# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [4D-HOF: Hand-Object Flow Matching for Feed-Forward 4D Interaction Reconstruction](https://arxiv.org/abs/2610.08782) | 提出4D-HOF，一个前馈式框架，利用条件流匹配模型将视觉基础模型生成的粗略手-物体状态传输至交互流形，实现4D手-物体交互重建，并可在传输过程中直接施加物理交互约束进行测试时引导。 |
| [^2] | [IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas](https://arxiv.org/abs/2610.08781) | 提出IdeaAnchor范式，通过编码论文功能角色、关系及综合标准的结构化规格说明作为特权信号，结合示范、自蒸馏与强化学习训练大语言模型，使其学会将文献综合为研究思路。 |
| [^3] | [DepthWorld: 3D World Model for Robot Manipulation](https://arxiv.org/abs/2610.08780) | 该论文提出了一种结合学习式双目深度与联合因子图的标定流程，据此构建了DROID-3D标定3D数据集，为基于视频的世界模型提供稠密度量深度和精确外参，使其能够生成具有3D几何一致性的机器人操作推演。 |
| [^4] | [Sherpa: Teaching LLMs to Teach Adaptively](https://arxiv.org/abs/2610.08778) | Sherpa是一个多轮强化学习框架，通过模拟具有不同学习偏好的学生原型并直接最大化其学习成效，训练教师LLM自适应地调整教学策略，使受教学生的成绩平均提升20.5个百分点。 |
| [^5] | [Agent in a Bottle: Can LLM Agents Turn Their Capabilities Into Cheap, Scalable Artifacts?](https://arxiv.org/abs/2610.08775) | 该论文提出了“装瓶”这一新概念和BOTTLED基准，用于评估LLM智能体能否在固定时间、计算和API预算内自主将自身通用能力转化为廉价、可复用的任务级解决方案，并发现强大的零样本表现并不能可靠地转化为强大的装瓶能力。 |
| [^6] | [AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model](https://arxiv.org/abs/2610.08773) | 提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。 |
| [^7] | [VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning](https://arxiv.org/abs/2610.08761) | VeriFine 通过策略、训练课程与评判者的共同演化（包括策略改进循环，以及在验证成为瓶颈时借助人类选择性指导与协同校准来精炼评判者的评判者改进循环），实现了具身推理中自我改进的规模化验证。 |
| [^8] | [WorldSonus: Bringing Sound to Worlds](https://arxiv.org/abs/2610.08760) | WorldSonus 是一个面向世界模型的交互式视频到音频框架，通过流式因果自回归扩散架构以 0.41 的实时率实现空间立体声的实时生成，并支持生成过程中的声音指令交互控制。 |
| [^9] | [Reinforcement Learning with Conformal Action Sets: An Application to Sequential Recommendation](https://arxiv.org/abs/2610.08743) | 该论文提出了RLCP方法，通过评论家分数和在线阈值自适应调整序列推荐中的动作集大小，并从理论上证明了代理未命中率上界以及将价值损失精确分解为过滤损失和选择损失，从而获得无需参数收敛的有限会话奖励上界。 |
| [^10] | [EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning](https://arxiv.org/abs/2610.08726) | EgoLAP提出了一种VLA预训练框架，通过共享的基于语言的动作思维链，将第一人称人类数据中的运动意图转化为结构化语言动作并进行运动级推理，从而弥合具身差异，使人类经验有效迁移到机器人控制，真实世界任务进度达到80.1%，性能提升2.3倍。 |
| [^11] | [Does an Agent's History Tell You When Compaction Will Hurt? A Modest, Bounded Effect on the TRACE Paired-Replay Corpus](https://arxiv.org/abs/2610.08722) | 本研究利用TRACE语料库中590个成对重放的压缩边界，检验智能体近期历史能否预测上下文压缩的危害，结果发现预测能力仅微弱有效——按前缀位置的预设对比为零结果，最佳可解释触发器也仅能避免21%的有害压缩边界。 |
| [^12] | [WorldSolver: Can LLM Agents Simulate the Physical Dynamics via Solver Generation?](https://arxiv.org/abs/2610.08720) | 提出WorldSolver基准，通过源自61篇经典计算机图形学论文、覆盖7个物理领域的168个模拟任务，评估LLM智能体生成物理求解器代码以模拟物理动力学的能力。 |
| [^13] | [nanoMuse: An Open-Source Personal Agent for Every Device You Own](https://arxiv.org/abs/2610.08699) | nanoMuse是Meta闭源个人智能体Muse的开源对应物：以GPL-3.0许可发布，在用户拥有的每台设备上各运行一个智能体，通过可自建的中继共享同一段持久对话，并能操作手机与电脑屏幕、跨周记忆、主动发起对话。 |
| [^14] | [ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science Agents Across the Natural and Social Sciences](https://arxiv.org/abs/2610.08691) | 该论文提出了ScienceClaw，一个将任务求解、科学验证与程序更新统一起来的固定参数程序自我演化框架，并配套覆盖23个学科的ScienceClaw-Eval基准，用于衡量AI科研智能体在序列任务中的演化收益、知识保留、跨数据集迁移与演化成本。 |
| [^15] | [Coupled but Late: Turn-Taking Between Full-Duplex Speech Models in Unscripted Dialogue](https://arxiv.org/abs/2610.08683) | 两个全双工语音模型互相对话时虽能形成相互耦合的时序模式，但话轮交接远比人类滞后（中位数400-560毫秒 vs 人类的137毫秒），且缺乏人类那种预测性的提前转换能力，表现为对对方话轮结束的被动反应式等待。 |
| [^16] | [Secure Speculative Decoding for Large Language Models](https://arxiv.org/abs/2610.08678) | 本文首次系统研究了投机解码的安全影响，揭示了一种“安全-效用不对称”现象：推理效率的提升以不成比例的高安全代价为代价，越狱和提示注入攻击的成功率上升速度远快于效用的下降。 |
| [^17] | [Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment](https://arxiv.org/abs/2610.08670) | 该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。 |
| [^18] | [MemFLoRA: Memory-Floor LoRA for CNN Adaptation at the Edge](https://arxiv.org/abs/2610.08669) | 本文提出MemFLoRA，一种以内存优先为设计原则的低秩CNN适配器，通过激活内存底线准则（可训练的反向计算不依赖全宽度层输入）来解决边缘端CNN适配中激活内存而非可训练参数数量才是限制性资源的问题。 |
| [^19] | [Semantic Behavioral Watermarking: Paraphrase-Robust and Forgery-Resistant Provenance for LLM Agents](https://arxiv.org/abs/2610.08668) | 提出语义行为水印（SBW），通过在历史条件约束下对语义动作簇进行基于密钥的抗碰撞分桶水印嵌入，实现了对改写鲁棒且能抵御伪造攻击的大语言模型智能体溯源方案。 |
| [^20] | [ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding](https://arxiv.org/abs/2610.08662) | 提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。 |
| [^21] | [Selective Transfer of RL Updates for Visual Reasoning](https://arxiv.org/abs/2610.08659) | 提出Selective-RL方法，通过选择性迁移强化学习更新中的主导矩阵方向并保留幅度，实现更有效的语言模型到视觉-语言模型的推理能力迁移。 |
| [^22] | [A Case Study in Assuring AI-Written Software](https://arxiv.org/abs/2610.08651) | 本研究通过对一个由非软件专业操作员运用编码智能体构建并管理的生产级医疗平台进行案例研究，揭示了测试、监控器和审查智能体等AI软件监督机制本身的不可靠性，表明详尽的代码审查不能作为人类控制AI编写软件的唯一依据。 |
| [^23] | [SquidAgent: Parallelize Wisely, Coordinate Efficiently](https://arxiv.org/abs/2610.08647) | 该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。 |
| [^24] | [HygieneRoboBench: Benchmarking Hygiene-Aware Planning for Household Robots](https://arxiv.org/abs/2610.08642) | 提出了 HygieneRoboBench 基准（134 个任务族、624 个实例），用于评估家庭机器人在考虑接触历史污染、处理成本和用户优先级的条件下进行卫生感知安全规划的能力。 |
| [^25] | [Parallel Predictive World Models for Accurate and Efficient Long-Horizon Planning](https://arxiv.org/abs/2610.08627) | 提出并行预测世界模型（PPWM），通过并行预测有限视野轨迹并在解码前让未来表示进行因果交互，消除了自回归滚动中的递归解码状态反馈，在四个视觉控制任务上实现最低长时程预测误差和最高CEM模拟器成功率。 |
| [^26] | [Feature Information Dynamics in Diffusion](https://arxiv.org/abs/2610.08626) | 提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。 |
| [^27] | [Early Memory Selection for Balanced Adam](https://arxiv.org/abs/2610.08624) | 该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。 |
| [^28] | [Agentic RCA for Internet-Scale Services Using Constrained Creativity](https://arxiv.org/abs/2610.08622) | 提出了E4智能体式故障排查系统，通过“约束性创意”范式将LLM的自动化探索能力与结构化方法的可解释性和高效性相结合，同时满足互联网规模服务根因分析的高准确性、低成本、可解释和低运维投入四项要求。 |
| [^29] | [Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness](https://arxiv.org/abs/2610.08621) | 该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。 |
| [^30] | [A Swarm-Coordinated Multi-Robot System for Early Stress Detection in Agricultural Rows Using Multimodal Leaf Sensing](https://arxiv.org/abs/2610.08603) | 提出了一种低成本群体协同多机器人系统CropSentry，通过多模态叶片传感与颜色编码实时仪表盘实现农田行间作物早期胁迫的持续监测，总体分类准确率达84.12%。 |
| [^31] | [One for All, All for One: Coordinated Multi-Agent Diffusion Steering via Stochastic Optimal Control](https://arxiv.org/abs/2610.08595) | 提出 CMDS 框架，将冻结的预训练扩散模型作为可复用的生成基元，把多智能体协调表述为随机最优控制问题并学习摊销化的控制，从而仅通过学习协调方式即可由独立训练的组件生成器产生连贯的结构化输出，并可在新任务实例中复用。 |
| [^32] | [MINDSET: Energy-based Schema Evolution for Long Conversational Agent Memory](https://arxiv.org/abs/2610.08586) | MINDSET提出了一种基于最小能量状态转换的记忆控制器，将对话存储为不可变情节并组织成带版本的图式，使长对话智能体能够适应随时间演变的指令与上下文，同时保留历史状态且无需反复调用大语言模型重写记忆。 |
| [^33] | [How Learning Governs Unlearning across the Memorization-Generalization Spectrum](https://arxiv.org/abs/2610.08577) | 模型的学习方式决定了其后续遗忘的表现：偏泛化型模型在遗忘时比偏记忆型模型遭受更大的保留集性能损害，且这一趋势在记忆-泛化谱系上几乎单调成立。 |
| [^34] | [FedDermaSeg: Federated Learning for Dermatological Image Segmentation](https://arxiv.org/abs/2610.08574) | 该论文提出FedDermaSeg，探索利用联邦学习在无需集中收集数据的情况下实现隐私保护的皮肤病灶分割，以解决传统集中式深度学习训练带来的隐私风险和高计算资源需求问题。 |
| [^35] | [RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems](https://arxiv.org/abs/2610.08571) | 该论文提出了RAG-PIBench——一个面向RAG系统提示注入检测的防泄漏基准，包含4,876个上下文示例，通过严格评估协议发现DistilBERT取得最佳检测性能（F1=0.896），同时证明TF-IDF等稀疏基线方法仍具竞争力。 |
| [^36] | [Adaptive Power Sampling for LLM Reasoning](https://arxiv.org/abs/2610.08563) | 提出自适应幂采样（APS），在测试时依据查询难度和模型自奖励逐查询调整分布锐化指数，从而在无需训练的情况下显著提升大语言模型的推理性能。 |
| [^37] | [Latent space bias directions in LLMs capture confidence, not fairness](https://arxiv.org/abs/2610.08559) | 该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。 |
| [^38] | [Systemization of Knowledge (SoK): Human-Centered AI Safety for Youth](https://arxiv.org/abs/2610.08554) | 本文系统回顾了100项HCI实证研究，构建了青少年AI风险与对策的映射，发现多数风险仅有构想层面的对策，很少被实施和评估，且评估多聚焦技术性能而非实际防伤害效果。 |
| [^39] | [DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory](https://arxiv.org/abs/2610.08553) | 针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。 |
| [^40] | [AnyBottle: A Recipe to Only Keep the Concepts You Really Need](https://arxiv.org/abs/2610.08552) | AnyBottle提出了一种由黑盒教师模型引导的迭代概念选择方法，结合嵌套dropout训练，能够构建紧凑、任务特定的概念瓶颈模型，只保留任务真正需要的概念，使瓶颈更小且更易于检查。 |
| [^41] | [How High Is 0.6? Floors, Ceilings, and Headroom in Interpretability Probing](https://arxiv.org/abs/2610.08544) | 该论文提出用“地板”（简单输入已能预测的水平）和“天花板”（完整输入能预测的水平）两个参照点以及二者之间的“余量”来校准可解释性探测得分，使探针分数具有明确、可比的含义，并证明余量在目标不依赖隐藏变量或输入不透露隐藏变量时消失。 |
| [^42] | [Micro Neural Policies for Safe Real-Time Robotic Control](https://arxiv.org/abs/2610.08541) | 该论文提出微型神经策略（MNP），通过结合进化策略与统计模型检测验证进行策略搜索，将神经网络的内存占用缩小至0.5至7.5 kB，使策略能够部署在微控制器上实现安全、鲁棒的实时机器人控制，并成功完成零样本的仿真到现实迁移。 |
| [^43] | [Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements](https://arxiv.org/abs/2610.08540) | 该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。 |
| [^44] | [From Shared Demand Patterns to Local Uncertainty: Probabilistic Load Forecasting by Mixing Compact Adaptations](https://arxiv.org/abs/2610.08538) | 该论文提出了一种可扩展的概率负荷预测框架，通过共享模型学习共同需求模式，并让每个负荷按需混合一个小型低维紧凑适配组件库，从而在客户级和变压器级兼顾局部预测精度与大规模部署的可扩展性。 |
| [^45] | [FlowCF: Sparse Counterfactual Explanations for Mixed-Type Tabular Data using Flow Matching](https://arxiv.org/abs/2610.08537) | FlowCF提出了一种基于流匹配的模型无关生成方法，通过新颖的混合流算子和门控网络，为混合类型表格数据生成具有稀疏性的反事实解释。 |
| [^46] | [MedCORE: Criteria-Grounded Clinical Reasoning for Interpretable Medical Image Diagnosis](https://arxiv.org/abs/2610.08528) | MedCORE提出了一种将临床诊断推理融入视觉-语言模型的结构化框架，通过标准分解、空间定位、多尺度证据编码和图注意力网络精炼，实现了可解释、透明的医学图像诊断。 |
| [^47] | [How Much Evidence Should a Coding Agent's Self-Correction Carry? Adaptive Dirichlet Evidence for Self-Distillation](https://arxiv.org/abs/2610.08514) | 该论文提出有效证据自蒸馏（EESD），利用Dirichlet后验将执行相关性与证据量分开建模，为编码智能体的自我纠正生成经不确定性惩罚的学习权重，在八个观测下相比固定质量方法显著降低了未来结果的NLL。 |
| [^48] | [Wiki-Talkie: Multilingual Benchmarking of Persona-Based Agents on Real-World Discussions](https://arxiv.org/abs/2610.08513) | 该论文提出了 Wiki-Talkie——首个基于维基百科讨论页真实对话、涵盖五种语言并配以源自真实用户社区画像的多语言基准数据集，用于评估角色化LLM智能体模拟人类交互的行为保真度。 |
| [^49] | [Cylindrical Geodesic Flow Matching for Quasiperiodic Physiological Signal Transformation](https://arxiv.org/abs/2610.08510) | 提出圆柱测地线流匹配方法，在圆柱面上显式建模圆形相位与严格正振幅的几何结构，从而实现配对准周期心血管波形之间的信号转换，克服了端点监督回归和标准仿射流匹配路径无法刻画相位—振幅结构的局限。 |
| [^50] | [X-OPM: Explainable Automatic Digital On-Chip Power Modeling for Enhanced Robustness](https://arxiv.org/abs/2610.08502) | X-OPM基于同步数字VLSI电路设计原理，提出了一个可解释的自动片上功耗建模框架，通过树模型捕捉特征交互并用线性模型进行预测，结合人在回路的工作流程，在商用C906向量处理器上实现了更鲁棒、可泛化且低开销的功耗预测。 |
| [^51] | [Language-model ratings of depression reflect the rater more than the patient](https://arxiv.org/abs/2610.08501) | 该研究通过对880个语言模型评分者的预注册实验发现，语言模型对抑郁症的评分更多反映评分模型自身的差异（解释30.0%的评分方差）而非患者的真实症状差异（仅10.5%），即使两个高精度模型平均也会对40%的参与者的筛查结果产生分歧。 |
| [^52] | [Knee3DVLM: Dual-Sequence Full-Volume Vision-Language Modeling for Comprehensive Knee MRI Assessment](https://arxiv.org/abs/2610.08482) | Knee3DVLM提出了一种序列感知的视觉-语言模型，首次联合利用全体积DESS与液体敏感TSE双序列膝关节MRI，预测57个基于MOAKS评分的解剖学分辨二分类诊断目标，实现了三种配置中最优的综合膝关节MRI结构化评估性能。 |
| [^53] | [MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata](https://arxiv.org/abs/2610.08479) | 提出去中心化框架MetaLearnNCA，通过Active-NCA与Meta-NCA两种耦合神经细胞自动机的动态交互实现少样本离线元学习，无需测试时反向传播计算梯度，同时保留二维空间几何结构信息。 |
| [^54] | [Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability](https://arxiv.org/abs/2610.08448) | 该研究发现跨分词器在线策略蒸馏中扩大对齐覆盖并无必要——严格1:1对齐已覆盖大部分token，仅用共享词表中由学生选择的top-16子集计算反向KL即可媲美完整共享词表方法，表明监督可靠性比对齐覆盖更为关键。 |
| [^55] | [AssemState: Manual and Physical-State-Guided Reasoning for Zero-shot Furniture Assembly](https://arxiv.org/abs/2610.08446) | 提出AssemState零样本框架，通过将组装说明书分解为单部件操作并构建组装树，结合迭代的物理状态反馈与基于仿真的释放测试，实现物理合理性的家具组装3D空间推理。 |
| [^56] | [EMHO: EMbodied Agent Harness Optimization via Experience Traces](https://arxiv.org/abs/2610.08432) | 提出自演化框架EMHO，让具身智能体在模型冻结的前提下，通过分析经验轨迹和框架历史直接自我优化外部框架，并以EMHO-Merge解决单一共享框架跨多个子任务优化的权衡问题。 |
| [^57] | [NeMo-DCR: Bit-Exact Delta-Compressed Refit for Scalable Agentic RL at Trillion-Parameter Scale](https://arxiv.org/abs/2610.08430) | NeMo-DCR提出了一种比特精确的增量压缩权重同步方法，利用固定仿射映射与残差转换仅传输每步约1%发生变化的权重，将万亿参数模型在训练与推演分离的智能体强化学习中的同步开销从87.5分钟大幅降低，并保证接收端参数与完整稠密同步完全一致。 |
| [^58] | [Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals](https://arxiv.org/abs/2610.08413) | 该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。 |
| [^59] | [Learning from Failures: A Failure-Driven Prompt Refinement for LLM-Based Vulnerability Analysis](https://arxiv.org/abs/2610.08405) | 本文提出失败驱动提示词优化方法（FDPR），通过分析大语言模型在漏洞分析中的反复失败模式来系统性地改进提示词，实验证明该方法显著提升了基于LLM的漏洞分析可靠性。 |
| [^60] | [GeoPID: Decomposing and Steering Visual Information in Vision-Language Models](https://arxiv.org/abs/2610.08401) | GeoPID是一个无需训练的框架，通过几何分解将VLM中的信息划分为冗余、模态独有和协同成分，并在推理时沿视觉独有子空间选择性放大视觉表示，从而有效增强模型的视觉依据能力。 |
| [^61] | [Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems](https://arxiv.org/abs/2610.08400) | Atom-JEPA是一种自监督预训练框架，通过互补的原子级和子结构级目标从无标签三维原子结构中学习潜在表示，在分子ADMET和量子化学性质预测等下游任务上达到了最先进的性能。 |
| [^62] | [Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering](https://arxiv.org/abs/2610.08388) | 提出前瞻感知的证据检索框架FoG，克服LLM图推理中逐跳贪心与束搜索剪枝的短视问题，避免关键证据分支被过早丢弃，提升知识库问答的可靠性。 |
| [^63] | [Accelerating the Development of PLGA In Situ Forming Depots Through AI-Driven Multi-Objective Optimization](https://arxiv.org/abs/2610.08368) | 该研究将Corbion的PURASORB聚合物库与Intrepid Labs的AI算法ANDROMEDA 1结合，仅用约15周和181个处方即完成多目标优化，成功筛选出4个满足黏度与可注射性要求且具有差异化30天释放曲线的治疗性多肽PLGA原位成型储库处方，显著加速了长效注射制剂的开发进程。 |
| [^64] | [Transect: Retaining Observability for Long-Horizon LLM Agent Evaluations](https://arxiv.org/abs/2610.08364) | Transect是一个基于Inspect Scout的开源工具，通过可复用的评估族配置和结构化分析流程，让评估者在分析长程LLM智能体动辄数百页的运行记录时保留可观测性，兼顾语言模型辅助分析的效率与评估结果的可重复性、可审计性。 |
| [^65] | [Explainable Failure Prediction and Prevention in Maritime](https://arxiv.org/abs/2610.08363) | 本章提出了一种将数据采集、时间序列预测、异常检测、风险评估、决策制定和可解释AI相集成的闭环概念架构，以实现海事系统中可信且可解释的故障预测与预防。 |
| [^66] | [Test-Time Adaptation of Quantized ViTs via Single-Pass Quantizer-Aligned Recalibration](https://arxiv.org/abs/2610.08358) | 提出QuAR，一种专为量化视觉Transformer设计的单次前向传播、无需反向传播的测试时自适应方法，直接针对分布偏移下激活值扭曲冻结量化器码分布这一量化特有失效模式，从而恢复精度。 |
| [^67] | [Sensor Geometry as a Flow-Matching Prior for Multi-Channel Brain Signals](https://arxiv.org/abs/2610.08355) | 该论文提出仅利用脑电电极的几何坐标构建k近邻图，并以图拉普拉斯的Matérn函数作为流匹配模型的源协方差，从而把已知的空间相关结构编码为先验，在不增加任何可学习参数的情况下替代各向同性高斯源，生成空间相干的多通道脑电信号。 |
| [^68] | [How Much Planning Is Enough? Reducing Search and Computation in World-Model Planning](https://arxiv.org/abs/2610.08350) | 提出SufficientPlan部署框架，通过成对序贯预算认证（PSBC）自动寻找并认证每个模型-任务组合所需的最小规划预算，并结合静态上下文复用（SCR）消除迭代规划中的冗余编码，从而在不修改预训练世界模型的前提下大幅降低决策时搜索的计算开销。 |
| [^69] | [Transferable Spatial Temporal Coherence Adversarial Attack on Black-Box Vision Language Models for Autonomous Driving](https://arxiv.org/abs/2610.08331) | 本文提出一种针对自动驾驶场景中黑盒视觉语言模型的时空一致性对抗攻击方法（STCA），通过字幕引导的帧选择、空间攻击与时序一致性攻击三个阶段，揭示了VLMs对视频时序感知对抗攻击的安全脆弱性。 |
| [^70] | [MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization](https://arxiv.org/abs/2610.08330) | MoF 提出了一种可扩展的黑盒大语言模型个性化框架，通过将用户偏好建模为共享潜在偏好面的组合并进行基于历史的路由，实现了无需额外参数更新即可对未见过的用户进行个性化。 |
| [^71] | [An AI-Assisted Formalization of the Poincar\'e Conjecture](https://arxiv.org/abs/2610.08329) | 本研究通过将数学家制定的证明蓝图与明确的里程碑相结合，完成了庞加莱猜想的AI辅助Lean 4形式化，为几何分析领域未来形式化项目的可复用基础设施奠定了基础。 |
| [^72] | [MedZERO: Self-Evolving Agents for Open-Ended Medical Reasoning Through Controlled Knowledge Accumulation](https://arxiv.org/abs/2610.08327) | MedZERO 提出了一个自我进化智能体框架，通过“考官”生成前沿医学题目、“推理者”进行基于证据的多轮推理，并借助受控知识积累机制，使大语言模型无需昂贵专家监督即可在开放、难以完全验证的医学推理领域实现自我提升。 |
| [^73] | [SCOPE: Certified Theorem Proving with a Language Model as the Policy Planner](https://arxiv.org/abs/2610.08319) | SCOPE框架通过让小型语言模型负责算子规划、符号引擎执行数值计算、编译器验证证明的自然分工，仅用135M参数模型就在218个多步数值命题上认证了87.6%的证明，大幅超越了消耗数十倍资源的更大规模直接生成模型。 |
| [^74] | [MARCO: The Radioactive Watermark for Protein Generative Models](https://arxiv.org/abs/2610.08316) | 提出了首个专为蛋白质生成模型设计的放射性水印框架MARCO，通过在扩散去噪过程中嵌入双层水印，在冻结原模型参数的前提下同时保护模型知识产权并实现对潜在生物安全滥用的法医溯源。 |
| [^75] | [The Standardization Trap: Certifying Joint Label Processing in Tabular Foundation Models](https://arxiv.org/abs/2610.08314) | 该论文揭示了检验表格基础模型是否遵循固定权重机制时存在的“标准化陷阱”问题，并提出两个仅依赖标准化标签处预测的证明方法，能够区分固定权重预测与非线性标签变换两种解释。 |
| [^76] | [CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling](https://arxiv.org/abs/2610.08312) | 提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。 |
| [^77] | [Mitigating Concept Drift in QoS Prediction for Teleoperation of Autonomous Vehicles Using Historic Data](https://arxiv.org/abs/2610.08297) | 本文提出将历史数据纳入预测流程以缓解遥操作QoS预测中机器学习模型因概念漂移导致的性能退化，并引入关键场景检测指标来专门评估遥操作场景下的预测性能。 |
| [^78] | [DySCo: Dynamic Sharding for Collaborative Edge-Cloud LLM Inference with Depth-Synchronized Batching](https://arxiv.org/abs/2610.08268) | 提出DySCo协同运行时系统，通过动态分片、模型感知的层范围执行器dyForward以及深度同步批处理，消除边云协同LLM推理中云端调用的空闲间隙，并解决到达不同模型深度的请求无法常规批处理的难题。 |
| [^79] | [zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models](https://arxiv.org/abs/2610.08258) | zkLLMPoT提出了一种零知识训练证明框架，通过让训练者在审计员选定的挑战序列上证明所提交模型的目标值，将认证成本与训练迭代次数解耦，且不泄露模型权重或私有训练数据。 |
| [^80] | [MASC: A Multi-Agent Self-Calibration Framework with Latent Construct Alignment for Consistent Client Role-Playing in Psychological Counseling](https://arxiv.org/abs/2610.08250) | 提出MASC多智能体自校准框架，通过潜在构念对齐与闭环校准机制（构念引导生成、协作精炼、一致性验证和记忆修正），使模拟来访者在长程心理咨询对话中保持心理状态、沟通行为与情绪的一致性。 |
| [^81] | [LeanPlan: Optimal Planning with LLM-Generated Heuristics and Admissibility Proofs](https://arxiv.org/abs/2610.08246) | LeanPlan是首个利用LLM生成的启发式函数（其可采纳性在Lean 4中经机器验证）以找到最优计划的规划系统，在国际规划竞赛领域上展现出优异的最优规划性能。 |
| [^82] | [Sensor-Language-Action Models](https://arxiv.org/abs/2610.08244) | 该论文提出传感器-语言-动作（SLA）建模框架，以语言作为感知与行动之间的语义接口，将多模态传感器观测、自然语言和异构动作统一在同一个模型中，并构建了覆盖超11.6万个体、79种传感器模态和60个动作组的大规模基准。 |
| [^83] | [OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization](https://arxiv.org/abs/2610.08231) | OSFP4 提出了一种新型 NVFP4 量化方案，通过分析乘性抖动 FP4 量化器，实现对对角平滑矩阵元素与块缩放因子的联合优化，以最小化矩阵乘积量化误差，从而在 LLM 推理量化中取得最高平均精度。 |
| [^84] | [Confidence-Ordering Reversal under Contextual Priors in Neural Decoding](https://arxiv.org/abs/2610.08229) | 论文揭示神经解码中上下文先验引发的“置信度排序反转”现象：当正确候选者初始排名较低时，融合后更大的置信度差距反而预示修复可能性更低，导致初始排名20开外的错误占融合后错误的46.6%。 |
| [^85] | [VOMMI: Collecting and Leveraging Portable Demonstrations for Mobile Manipulation](https://arxiv.org/abs/2610.08220) | 提出VOMMI框架，通过离线轨迹重建和在线视觉运动条件化，将低成本、便携式的RGB示范与VLA后训练连接起来，无需额外传感硬件即可实现移动操作的示范收集。 |
| [^86] | [Quantum Entangled Multimodal Fusion Networks (QEMFN): Resource-Aware Hybrid Vision-Language Fusion via Trainable Entanglement](https://arxiv.org/abs/2610.08216) | 该论文提出量子纠缠多模态融合网络（QEMFN），一种将参数化纠缠作为结构化归纳偏置的混合量子-经典视觉-语言融合框架，在参数预算匹配且使用相同冻结CLIP骨干的条件下，于COCO-5k和Flickr30k检索任务上超越了多种经典融合基线。 |
| [^87] | [Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?](https://arxiv.org/abs/2610.08215) | 本文提出了Learn2Play Bench基准，通过规则新颖或反直觉的新设计文本游戏，评估LLM智能体在不熟悉环境中通过交互从经验中学习的能力，而非依赖预训练知识。 |
| [^88] | [Mathematical Proof Assistants for Teaching Logic: The LogiKEy Methodology](https://arxiv.org/abs/2610.08214) | 提出基于 LogiKEy 方法论的逻辑教学方案，以经典高阶逻辑作为通用元逻辑，通过语义嵌入让单一证明助手成为学生学习、实验和比较多种（经典与非经典）逻辑的统一环境。 |
| [^89] | [STRUCTURALCOST: A controlled reading time dataset for modeling human sentence processing difficulty](https://arxiv.org/abs/2610.08208) | 该研究推出大规模阅读时间数据集STRUCTURALCOST，验证了人类阅读时间随主谓依存长度增加而上升，并揭示现有语言模型虽能反映这种预测性难度但低估了工作记忆导致的整合成本，为评估语言模型的认知合理性奠定了数据基础。 |
| [^90] | [Tool-calling retrieval versus vector RAG for a small Greek--English knowledge base: accuracy and robustness to how users type Greek](https://arxiv.org/abs/2610.08205) | 在小型希腊语-英语农业知识库基准KyGround上，向量RAG（95.3%）显著优于工具调用检索（71.6%），将整个知识库直接放入上下文可达99.3%，且向量RAG对无重音、大写及Greeklish等非标准希腊语输入形式表现出鲁棒性。 |
| [^91] | [Compact Robot Policies Need Fine-Grained Visual Representations](https://arxiv.org/abs/2610.08183) | 论文提出紧凑策略CoRP，证明机器人多任务操作的性能主要由细粒度的预训练视觉表示决定，而非参数规模或生成式先验，其4890万参数的小模型即可媲美比它大上百倍的系统。 |
| [^92] | [LFHE: Local-First Heuristic Evolution for Bounded Local Topology Search in Decentralized Learning with Non-IID Data](https://arxiv.org/abs/2610.08176) | LFHE 提出了一种仅利用自身邻域和朋友的朋友信息进行受限局部拓扑重连的框架，其结构分数与图狄利克雷能量精确对应，从而在非独立同分布数据下有效加速去中心化学习中表示分歧的消散。 |
| [^93] | [Which alloy composition,what process parameters? Inferring the recipe from optimized metallic microstructure and texture](https://arxiv.org/abs/2610.08165) | 该研究的核心贡献是首次探索用机器学习实现合金开发的“逆向”步骤——从优化的微观组织与织构反推出合金成分和加工工艺参数，并在镁合金挤压数据集上系统比较了传统统计量、预训练视觉嵌入和图神经网络三种微观结构描述方法。 |
| [^94] | [Symphony for Text Generation: Benchmarking Clinical Note Generation](https://arxiv.org/abs/2610.08161) | 该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。 |
| [^95] | [Token-Efficient Multi-Agent Collaboration via System One-Guided Computational Division of Labor](https://arxiv.org/abs/2610.08155) | 提出S1-MAS框架，通过“系统一”式的计算分工将有界的协调决策交给轻量级模型处理、让大语言模型专注于开放式推理，从而在不牺牲协作性能的前提下大幅降低多智能体系统的令牌开销和延迟。 |
| [^96] | [Penalty-Framed No-Valid-Option MCQA: Analyzing LLM Abstention under Invalid Choices](https://arxiv.org/abs/2610.08153) | 该论文提出“惩罚框架下的无有效选项多选题问答”这一新评测设定，并通过基于正确回答的条件分析方法，揭示了大语言模型的高答题准确率并不能保证其在所有选项均无效时可靠地选择弃答。 |
| [^97] | [Navier-Stokes lost in translation: Why Lean verification of AI autoformalisation does not guarantee correct natural language proofs](https://arxiv.org/abs/2610.08144) | 本文证明了解决数学自然语言文本歧义（语义忠实自动形式化的必要步骤）这一难题在可解性复杂性索引层级中处于任意高的不可解位置（SCI=∞），因此即使AI自动形式化通过了Lean机械验证，也不能保证原始自然语言证明的正确性。 |
| [^98] | [Partially Observable Zero-shot coordination by Predicting Intention of Partner](https://arxiv.org/abs/2610.08142) | 提出PIP方法，利用联合视角VAE构建仅凭局部观测即可获得的伙伴表示，并通过伙伴状态信念网络推断伙伴的隐藏位置与行为倾向，解决了具身零样本协调中伙伴不可见导致的表示模糊与状态不确定问题，在多个基准及人类评估中取得最优表现。 |
| [^99] | [Test-Time Agent Evolution for Long-Horizon Legal Reasoning](https://arxiv.org/abs/2610.08138) | 该论文提出了一种免训练的测试时代理自适应方法，通过测试时记忆演化机制从历史案例中检索、适配并积累可复用经验，无需更新模型参数即可提升长程多角色法律推理的全局可靠性。 |
| [^100] | [Supermarket Product Detection and Recognition: Utilizing Deep Learning with Rectified Imagery](https://arxiv.org/abs/2610.08126) | 本文提出将传统霍夫变换和单应性估计与深度学习目标检测模型相结合，通过图像校正技术解决超市密集货架因拍摄角度变化带来的商品识别难题。 |
| [^101] | [Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving](https://arxiv.org/abs/2610.08123) | 本文提出一种基于查询的代价学习框架，通过对自车动态可达的未来轨迹估计有界代价来替代传统路径点回归，将代价拓扑转化为可行规划，在nuScenes与真实驾驶数据上显著降低碰撞率并保持可解释性。 |
| [^102] | [Exploiting Acoustic and Content-Oriented Speaker Verification Attacks Against Multilingual Voice Anonymization](https://arxiv.org/abs/2610.08107) | 本研究首次在多语言环境下系统评估了声学导向与内容导向的说话人验证攻击对语音匿名化的威胁，发现攻击有效性取决于匿名化语音所保留的语言信息量，并构建了多语言语音转换数据集以提升跨语言泛化能力。 |
| [^103] | [ChartBmkAgent: Harness-Governed Multi-Agent Construction of Chart QA Benchmarks from Sparse Error-Taxonomy Specifications](https://arxiv.org/abs/2610.08106) | ChartBmkAgent 提出一种由评测框架治理的多智能体流程，仅凭稀疏的错误分类规范即可按需从零构建图表问答基准，并确保新生的需求与内容始终与外部指定的诊断目标保持对齐，从而缩短基准开发周期。 |
| [^104] | [DSV-Mem: Evaluating Multimodal Memory in Professional Workflows for MLLM Agents](https://arxiv.org/abs/2610.08102) | 该论文提出了首个面向专业工作流的多模态记忆评估基准DSV-Mem，通过1,000个专家审校的问题和五大任务类别，评估MLLM智能体在信息密集、频繁更新且需精确追踪状态的专业产物场景下的密集有状态视觉记忆能力。 |
| [^105] | [Beyond Corrected Memory: Execution Consistency in Multi-Agent Systems](https://arxiv.org/abs/2610.08101) | 该论文提出多智能体系统中的“执行一致性”概念，证明完全相同的共享记忆记录在同一任务规则下可能同时对应合规与违规的执行，因此仅有正确的记录不足以判断任务职责是否被履行，需要显式的证据条件（如接收凭证、动作依赖、响应有效性）来加以区分。 |
| [^106] | [When Tools Lie: Reliability of Mathematical Agents Under Corrupted Tool Feedback](https://arxiv.org/abs/2610.08097) | 该论文提出一个受控污染框架研究数学智能体检测和纠正被篡改工具反馈的能力，发现无验证时污染使准确率从100%降至72.4%，而强制同上下文反思可将性能完全恢复至100%。 |
| [^107] | [Natural Language Questions as an Interface for Knowledge Graphs: QRAKEN Graph Distillation and Semantic Self-Healing](https://arxiv.org/abs/2610.08095) | QRAKEN提出了一种无需训练的神经符号流水线，通过离线图蒸馏生成TTQL图谱证据来引导LLM生成SPARQL查询，并借助确定性的语法与数据模型检查实现迭代式语义自修复，在Text2SPARQL挑战赛上取得了严格的F1最佳成绩。 |
| [^108] | [SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis](https://arxiv.org/abs/2610.08093) | SAGE提出了一种数据合成框架，利用MeSH等轻量级公开分类体系作为语义锚点，通过迭代交替进行原子与关联合成，使小型本地部署模型也能从极少种子数据生成高质量的医学问答训练数据。 |
| [^109] | [When Plans Change Answers: Formalizing Cost-Accuracy Optimization for Semantic Queries](https://arxiv.org/abs/2610.08089) | 该论文首次形式化了语义查询的成本-准确率优化问题，利用决策模型的校准置信度和错误在连接中的传播加权，无需标注数据即可预测查询计划的输出质量，并能将输出层面的准确率目标反向转化为元组级别的定价，从而实现更优的查询计划选择。 |
| [^110] | [POLAR: Ontology-Guided Risk Prevention for Tool-Calling LLM Agents](https://arxiv.org/abs/2610.08082) | POLAR是一个通过结构化两层本体评估操作可逆性的防护栏框架，能在工具调用LLM智能体执行高风险操作前将其剪除并提供可审计的结构化判定，但其收益因任务域和智能体能力而异。 |
| [^111] | [Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight](https://arxiv.org/abs/2610.08077) | 该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。 |
| [^112] | [SpeedrunBench: Challenging LLM Agents with Video Game Speedrunning](https://arxiv.org/abs/2610.08076) | 该论文提出了SPEEDRUNBENCH基准，通过9款电子游戏的速通任务，评估大语言模型智能体自主形成复杂策略、超越人类已知解法的能力。 |
| [^113] | [DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks](https://arxiv.org/abs/2610.08048) | DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。 |
| [^114] | [Same Feedback, Different Answer: Measuring Run-to-Run Instability in Frontier-Model Customer Feedback Analysis](https://arxiv.org/abs/2610.08036) | 本文提出了一个重复运行评估框架，通过对齐语义等价类别并测量主题流失与数量分歧两个指标，系统量化了八个前沿模型在不同语料规模、提示词和执行设计下客户反馈分析结果的运行间不稳定性。 |
| [^115] | [Learning in Dreams, Winning in Reality: A Continuous Dyna Loop for a Ten-Hero MOBA](https://arxiv.org/abs/2610.08033) | 该研究提出一个连续异步Dyna循环框架，仅在十英雄MOBA游戏的结构化多智能体世界模型中训练策略，使天辉方真实对局胜率从循环前的33.7%提升至70.2%，而真实游戏仅用于提供数据和评估、从不提供梯度。 |
| [^116] | [TICDA: Tabular In-Context Data Attribution](https://arxiv.org/abs/2610.07996) | 提出TICDA方法，通过在表格基础模型潜在表示上训练的线性代理模型直接量化上下文中每个示例对预测的影响，解决了重采样和基于梯度的数据归因方法在上下文学习场景中失效的问题。 |
| [^117] | [VisionWeave: Weaving Elastic Visual Representations as a Native Capability of MLLMs](https://arxiv.org/abs/2610.07987) | VisionWeave通过大规模训练，使多模态大语言模型获得按内容自适应地决定视觉表示位置与粒度的“弹性视觉表示编织”固有能力，在保留关键细节的同时显著降低计算成本。 |
| [^118] | [Decide Before You Look: Learning Which Retrieved Memories Deserve Pixels](https://arxiv.org/abs/2610.07984) | 提出 PixelTriage——一个置于检索之后的轻量级插件，通过阅读对话、笔记和缩略图即可在回答模型运行前预测哪些被检索记忆的像素真正有用，从而大幅削减视觉 token 成本并保持回答精度。 |
| [^119] | [Learning from Revision Consequences: Hindsight Meta-Experience Distillation for Self-Improving Agents](https://arxiv.org/abs/2610.07979) | 该论文提出HMED（后见之明元经验蒸馏），通过后见之明地分析技能修订带来的实际后果来构建元经验，将修订效应与初始发现状态的效应解耦，从而更准确地改进自我改进智能体的元技能。 |
| [^120] | [Can Agents Work for Everyone? Cross-User Reliability for Mobile GUI Agents in Personalized User Interfaces](https://arxiv.org/abs/2610.07972) | 该论文提出PAIR流水线与RePAIR强化学习训练方法，首次系统评估并提升移动GUI智能体在不同用户个性化界面上的跨用户可靠性，揭示出智能体在用户条件化环境中的任务成功率存在显著下降。 |
| [^121] | [ReGraph: A Computational Account of Emergent Generalization in the "what" and "where" Dual Visual Streams](https://arxiv.org/abs/2610.07962) | 提出ReGraph——一个具有生物归纳偏置的循环双流图模型，在计算层面解释了情境不变的关系结构（即泛化能力）如何沿腹侧“什么”与背侧“哪里”双视觉通路涌现形成。 |
| [^122] | [Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents](https://arxiv.org/abs/2610.07948) | 提出置信推理图（CRG），一种推理时框架，通过结构化分解单条轨迹中的证据来估计LLM智能体完成任务的概率，无需访问模型内部信号或训练数据。 |
| [^123] | [Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution](https://arxiv.org/abs/2610.07946) | 提出SALT方法，将上一动作块中未执行的剩余轨迹作为自监督信号，通过过渡锚定与顺序修正传播，使视觉-语言-动作模型能够在机器人执行任务时对未知视觉干扰进行测试时自适应。 |
| [^124] | [Hybrid Latent Attention for Looped Language Models](https://arxiv.org/abs/2610.07940) | 提出混合潜在注意力（HLA），通过将旧 token 压缩为紧凑的潜在表示而非存储完整键值缓存，使循环语言模型的缓存缩小 10.7 倍、每 GPU 并发序列容量提升 4.0-8.8 倍、解码吞吐量最高提升 7.4 倍，同时保留原模型 97% 以上的性能。 |
| [^125] | [SIGMA: Self-Improving Alignment Generalization from a Model Spec](https://arxiv.org/abs/2610.07935) | 提出SIGMA数据生成与训练流程，仅需一份模型规范即可让大语言模型利用自身推理能力实现安全对齐的自我改进，并能泛化到分布外场景。 |
| [^126] | [Dynamic Alignment and Calibration for Multimodal Learning](https://arxiv.org/abs/2610.07928) | 提出了对齐与校准驱动的多模态学习框架 ACML，通过动态跨模态三元组对齐模块解决静态对齐导致的过度对齐问题，并在融合中充分考虑模态间的特征幅值与置信度差异。 |
| [^127] | [Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images](https://arxiv.org/abs/2610.07913) | 提出了一种多模态知识蒸馏框架，利用低秩多模态融合将WSI图像与病理文本融合训练教师模型，再将知识蒸馏至仅需图像输入的学生模型，以较低的计算成本实现准确的胃腺癌亚型分类。 |
| [^128] | [Diverse Motion Customization via Control-based Dynamic Optimization](https://arxiv.org/abs/2610.07911) | 提出基于随机最优控制的运动定制框架CMC，通过避免生成过程向参考视频坍缩来解决内容泄漏问题，使定制视频仅继承目标运动而外观完全由文本提示决定。 |
| [^129] | [Continuous Memory Machines](https://arxiv.org/abs/2610.07907) | 提出连续记忆机（CMM），一种具有矩阵值短期与长期记忆状态的新型循环架构，通过Transformer联合更新两种记忆，实现了快速神经元级计算与长期信息保留的结合。 |
| [^130] | [Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations](https://arxiv.org/abs/2610.07906) | 论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。 |
| [^131] | [IEEE 802.11bx - WLAN Intelligent Networking (WIN): Toward an AI-Ready Wi-Fi 9](https://arxiv.org/abs/2610.07900) | 本文综述了IEEE 802.11标准化进程中迈向AI就绪Wi-Fi 9的最新进展，创新性地从AI作为协议、平台和流量三个互补维度系统阐述了AI与Wi-Fi融合的候选特性与开放挑战。 |
| [^132] | [Variance-Averse $n$-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments](https://arxiv.org/abs/2610.07899) | 提出VAN-Flow框架，通过结合分类分布式评论家、方差厌恶期望算子和拒绝采样引导的流匹配生成式演员，在生成式离线强化学习中选择高回报且低方差 dispersion 的可靠动作。 |
| [^133] | [Textual Environmental Context and Spatial Graphs for LLM-Based Regional SST Forecasting](https://arxiv.org/abs/2610.07895) | 该论文提出将文本化的环境上下文与静态、动态空间图相结合，通过图神经网络生成空间前缀注入大语言模型，从而在不序列化完整海温网格的情况下实现区域多步海表温度预测。 |
| [^134] | [Visual Abstention in Unified Multimodal Models](https://arxiv.org/abs/2610.07887) | 该论文形式化了“视觉拒答”概念并构建了Draw-or-Decline基准，揭示出统一多模态模型的编辑能力与拒答能力相互独立——即便最强的编辑模型也几乎不会拒绝不可行的编辑请求。 |
| [^135] | [ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning](https://arxiv.org/abs/2610.07886) | 该论文提出了个性化全量化多目标膳食规划问题（MDP），并开发了营养智能体ShanLiangRen，通过将用户需求转化为约束规划实例、以精确检索增强生成缩小候选空间，并结合帕累托原则引导的精细化方法，生成兼顾个性化约束与多维营养目标的每日膳食方案。 |
| [^136] | [Label-Efficient Deep Learning for ECG Delineation: A Multi-Dataset Benchmark against Widely Used Delineation Tools](https://arxiv.org/abs/2610.07885) | 该研究通过多数据集基准测试证明，自监督预训练配合恰当的微调策略可显著降低心电波形分界对专家标注的依赖，且所得深度模型的分界性能优于广泛使用的开源分界工具。 |
| [^137] | [Self-Referenced Social Preferences: Cooperation without Observing Others Rewards](https://arxiv.org/abs/2610.07881) | 该论文提出“自参照社会偏好”方法，让智能体利用自身奖励模型从自身视角评估他人行为的结果，从而无需观察他人私有奖励即可在多智能体强化学习中有效促进合作。 |
| [^138] | [ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents](https://arxiv.org/abs/2610.07863) | ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。 |
| [^139] | [A self-learning scientific agent for X-ray diffraction](https://arxiv.org/abs/2610.07862) | 本文提出“干将”——一个面向粉末X射线衍射的自学习科学智能体，它通过诊断失败并自主修订和验证技能指令与代码，将分析经验转化为可复用的可执行技能，且无需重新训练语言模型，在多个精修平台上超越了专家设计的技能。 |
| [^140] | [WorkflowOps: Learning Agent Collaboration Priors for Multi-Agent Workflow Orchestration](https://arxiv.org/abs/2610.07860) | WorkflowOps 提出了一种多智能体工作流编排框架，通过从历史工作流中学习智能体协作先验（转移概率矩阵）来引导 DAG 工作流构建，并按需创建专门的智能体以填补能力空缺，从而避免编排层“无记忆、从零开始”的问题。 |
| [^141] | [RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation](https://arxiv.org/abs/2610.07851) | RA-MoWE 通过记录参考工作流求解效果的工作流亲和度嵌入对查询进行聚类，并借助执行反馈为每个聚类生成可复用的专家工作流，使新查询无需先执行即可直接匹配到合适的专门化工作流。 |
| [^142] | [Dynamic Positional Attention Modulation for Parameter-Efficient Fine-Tuning of Large Language Models](https://arxiv.org/abs/2610.07848) | 提出DyPAM方法，通过在查询和键表征上结合输入条件化的逐维度调制与逐头逐层的结构化调制，动态调整位置信息对注意力的贡献，实现更精细的参数高效微调。 |
| [^143] | [DHCG: Dynamic Construction of Hierarchical Collaboration Graphs for LLM-Based Multi-Agent Reasoning](https://arxiv.org/abs/2610.07835) | 提出DHCG框架，将多智能体系统设计建模为部分可观测马尔可夫决策过程，通过规划器、工作者和生成器三个模块，基于查询与执行反馈动态构建分层协作图，实现智能体组合与规模的灵活自适应。 |
| [^144] | [Harness Engineering for Software Engineering via Modular Executable Dev-Primitives](https://arxiv.org/abs/2610.07832) | 该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。 |
| [^145] | [Agentic Semantic Sensing for Resource-Adaptive AI-RAN](https://arxiv.org/abs/2610.07829) | 该论文提出了一种闭环的智能体化语义感知框架（Agentic SemS），通过配置条件化因果Transformer与语义效用网络，在通信可行的资源配置集合内动态控制感知过程，从而实现资源自适应的AI无线接入网络。 |
| [^146] | [One Step at a Time: Trading LLM Autonomy for Process Predictability](https://arxiv.org/abs/2610.07817) | 该论文提出通过MCP协议逐步向智能体交付流程步骤，以牺牲LLM自主性为代价，从架构上保证流程的事前可预测性，并生成可供下游工具逐步审计和优化的机器可读执行日志。 |
| [^147] | [Do I Need the Cloud? Uncertainty-Aware Step-Level Handoff for Small Language Model Agents](https://arxiv.org/abs/2610.07816) | 提出STEPGATE框架，通过不确定性感知地对本地小模型的每个动作评分，并按需将困难步骤升级到更强的云端模型，从而在大幅降低云端调用比例的同时显著提升智能体的任务成功率。 |
| [^148] | [ThinkFuse: Trajectory-Aware Test-Time Fusion for Small Reasoning Models](https://arxiv.org/abs/2610.07803) | ThinkFuse提出了一种无需训练的轨迹感知测试时融合框架，通过对比片段级不确定性变化与轨迹级整体趋势来识别不稳定推理点，并将辅助推理路径融合进主轨迹，从而显著提升小型推理模型在数学和知识密集型推理任务上的可靠性与性能。 |
| [^149] | [Novice Reliance Calibration in AI-Assisted Decision Making: The Role of Explanations and Self-Assessment](https://arxiv.org/abs/2610.07800) | 论文提出“依赖校准”这一新构念，通过对110名参与者的实验发现，在缺乏外部反馈时，AI解释会使新手用户系统性地滑向过度依赖，而任务自我理解等元认知自我评估有助于校准对AI的依赖。 |
| [^150] | [Thin Evidence, Thick Priors: How Language Models Substitute Identity for Missing Financial Facts](https://arxiv.org/abs/2610.07798) | 论文发现，用户披露的财务信息越少，大语言模型给出的投资建议就越依赖投资者身份而非其实际财务状况——在零披露条件下，身份对建议差异的解释力从5%飙升至96%，两人间建议差距也从4.78个百分点扩大至10.34个百分点。 |
| [^151] | [The Geometry of Empowerment](https://arxiv.org/abs/2610.07796) | 本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。 |
| [^152] | [ServeLearnBench: How Well Can Agents Self-Improve from Serving Experience?](https://arxiv.org/abs/2610.07792) | 本文提出 ServeLearnBench 基准及演化环境流式数据集（EESD），用于系统评估大语言模型智能体在隐藏策略持续演变的环境中，能否从交互与反馈中推断、应用并修正潜在环境知识，从而实现自我改进。 |
| [^153] | [Illusory Pattern Perception Drives Spurious Inference in Large Language Models](https://arxiv.org/abs/2610.07791) | 本研究首次系统性地揭示了大语言模型存在比人类更强的虚幻模式感知倾向——如将积极属性过度关联到多数群体、从模糊事件中强行构建因果叙事——这种认知偏差会导致模型产生系统性推理错误。 |
| [^154] | [OOPMAS: Object-Oriented Multi-Agent Systems for Query-Level Workflow Generation](https://arxiv.org/abs/2610.07787) | OOPMAS提出了一种无需训练的面向对象多智能体框架，能为每个查询动态生成专属的智能体集合与协调工作流，从而适应查询难度差异和现实场景中的异构任务类型。 |
| [^155] | [Attacca: Goal-Directed Control under State Continuity for Long-Horizon Embodied Agents](https://arxiv.org/abs/2610.07785) | 提出 Attacca 方法，通过在完整的“搜索—交互”轨迹上训练视觉目标条件策略，解决了长时程具身任务中因状态连续变化导致目标不可见、难以衔接下一个任务的核心难题。 |
| [^156] | [Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell](https://arxiv.org/abs/2610.07782) | 本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。 |
| [^157] | [Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs](https://arxiv.org/abs/2610.07781) | 该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。 |
| [^158] | [Towards One-for-All Foundation Model for Attributed Graph Clustering](https://arxiv.org/abs/2610.07778) | 提出OFAG——一个面向属性图聚类的基础模型，仅需一次训练即可直接应用于多样化的属性图，无需针对特定图的训练、微调或超参数搜索。 |
| [^159] | [OTel: Open Telco AI Datasets, Benchmarks, and Models](https://arxiv.org/abs/2610.07766) | OTel发布了一个统一的开放电信AI资源，提供面向检索、重排序、指令微调及安全/拒答的电信数据集，以及采用开放训练方案后训练的30个基线模型（10个嵌入模型、3个重排序器、17个语言模型）。 |
| [^160] | [No Transformer Beats Six Covariates: Long-Horizon Prediction of Depressive Symptoms from Childhood Essays](https://arxiv.org/abs/2610.07764) | 该研究发现，在利用11岁儿童作文预测其23岁抑郁症状的长期任务中，基于六个童年协变量的简单逻辑回归（AUC-ROC 0.737）显著优于所有文本模型，包括微调Transformer、词袋模型、冻结嵌入和零样本大语言模型（最佳仅0.670）。 |
| [^161] | [ST-Bench: A Spatial-Temporal Benchmark for Multi-Agent System Generation on Scientific Research Tasks](https://arxiv.org/abs/2610.07763) | 提出了ST-Bench基准测试，用于检验多智能体系统在复杂科学数据分析任务上能否优于单智能体、优势有多大以及需要付出多少额外代价。 |
| [^162] | [Contrastive Learning for Aspect Representation towards Explainable Recommendation](https://arxiv.org/abs/2610.07761) | 该论文提出CLARER推荐模型，通过Transformer编码器和对比学习从评论中提取方面特征，并与评分信息融合，同时提升了推荐的准确性与可解释性。 |
| [^163] | [Later Is Better: Token Reduction for ViTs Under Distribution Shift](https://arxiv.org/abs/2610.07758) | 提出一种单参数的晚集中幂律token削减时间表，在无需额外推理成本的前提下，显著缩小了视觉Transformer在分布偏移下免训练token削减造成的精度差距。 |
| [^164] | [From Evidence to Action: How Tool-Using Agents Fail](https://arxiv.org/abs/2610.07753) | 提出包含656个案例的SafeActBench基准，系统揭示工具使用型智能体在“证据到行动”链条中的失败模式——失败往往在执行前就因调查不完整或过早行动而出现，多动作工作流还会暴露未解决的先决条件和不完整执行问题。 |
| [^165] | [How Well Do LLMs Reason with Noisy Evidence? An Active Visual Reasoning Benchmark](https://arxiv.org/abs/2610.07751) | 提出了 VisualNoiseQA 基准，让纯文本 LLM 通过迭代查询被视为随机视觉传感器的 VLM，并利用基于自一致性的不确定性信号，在噪声证据下进行主动视觉推理，自主决定下一步询问内容及何时停止。 |
| [^166] | [Cleave: Scaling Tensor Program Optimization via Decoupled Algebraic Search and Operator Scheduling](https://arxiv.org/abs/2610.07742) | 提出 Cleave 编译器，通过将代数变换搜索与算子调度解耦——先在符号形状图上做超级优化、再在具体形状上调度——自动生成媲美 FlashAttention 等手工优化内核的高效张量程序。 |
| [^167] | [Learning to Retrieve via Reinforcement Learning in Embedding Space](https://arxiv.org/abs/2610.07731) | 该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。 |
| [^168] | [SanSi: A Looped Typed Decision Model for System 1.5 Thinking](https://arxiv.org/abs/2610.07730) | SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。 |
| [^169] | [PERSIST: Who-What-When Memory Across Sessions for Full-Duplex Spoken Dialogue](https://arxiv.org/abs/2610.07725) | PERSIST是一个面向多用户多会话语音对话的持久记忆系统，通过联合建模“谁-什么-何时”的3W评分机制实现准确的跨会话记忆检索，并复用对话主干中间表示以支持低延迟的实时全双工交互。 |
| [^170] | [Evidence Before Sampling: Interpretable Implicit Negative Candidate Discovery for Recommendation](https://arxiv.org/abs/2610.07708) | 该论文提出“先证据后采样”的隐式负候选发现框架，将用户行为模式编码为符号化规则并通过证据强度排序，再由大语言模型结合业务目标进行解释，从而为推荐系统提供可解释且与业务对齐的隐式负样本。 |
| [^171] | [AgentMemGate: Addressing Speculation Contamination in Conversational Assistant Memory](https://arxiv.org/abs/2610.07707) | 提出AgentMemGate——一种写入时门控机制，通过将对话中提取的语句分类为推测、已完成事件、更正或其他类型，防止用户表达的计划（推测）被错误地作为事实写入对话助手的长期记忆，从而解决“推测污染”问题。 |
| [^172] | [WASD: Wasserstein-based Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2610.07706) | 该论文提出了WASD方法，通过由词元嵌入构建代价矩阵的Wasserstein距离，将词元级别语义信息融入大语言模型的知识蒸馏中，并借助Sinkhorn散度实现高效优化。 |
| [^173] | [What Frame-Level Labels Can and Cannot Do for Small-UAV Point Detection in Thermal Video](https://arxiv.org/abs/2610.07705) | 该论文探究了仅使用帧级目标存在/不存在标签（无需空间标注）来训练小型无人机点检测模型的检测能力、学习行为与局限，并在两个热红外数据集上评估了其定位命中率与检测率表现。 |
| [^174] | [On the Boundary of Admission Gates: An Injected-Truth Study of Falsification-First Selection in Quantitative Strategy Research](https://arxiv.org/abs/2610.07701) | 该研究提出注入真值验证协议评估量化策略研究中的准入门槛，发现门槛虽能消除弱信号情形下的虚假发现但采纳率仅降至1-7%，且基于绝对收益而非超额收益的准则会错误拒绝包括真实信号在内的所有候选策略。 |
| [^175] | [BluffJAX: Adversarial Imperfect Information Games in JAX](https://arxiv.org/abs/2610.07686) | BluffJAX是一个基于JAX的开源对抗性不完全信息游戏套件，支持GPU加速的高吞吐量并行模拟（每秒可达数亿样本），收录了德州扑克、库恩扑克等多种经典及全新游戏，为强化学习的博弈论方法研究开辟了新的挑战与方向。 |
| [^176] | [Disentangling Dual Image References in Frequency Aware Diffusion Models for Personalized Generation](https://arxiv.org/abs/2610.07684) | 提出 Dual-FDM 频率感知扩散模型，通过解耦去噪过程中的混合频段纠缠，同时利用双图像参考（定制参考与颜色风格参考）实现定制风格迁移和颜色风格迁移的个性化图像生成。 |
| [^177] | [EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors](https://arxiv.org/abs/2610.07681) | 该论文发现，人类运动先验最有效的利用方式是将其用于结构化探索以引导协调动作的发现，而不是用来限制或扩充动作空间，从而在灵巧操作中兼顾协调性与表达能力。 |
| [^178] | [Exact-Solution Volume and Length Generalization in Transformers](https://arxiv.org/abs/2610.07676) | 该论文提出归一化精确解体积（NESV）这一新指标来量化Transformer的长度泛化难度，证明精确解体积随输入长度衰减得越快，长度泛化就越困难，并为FIRST、MAJORITY、INDEX、PARITY四个任务建立了渐近界。 |
| [^179] | [EIO-Agents: The Missing Semantic Layer for AI Agent Evaluation](https://arxiv.org/abs/2610.07675) | 论文提出了EIO-Agents开放规范，通过评估智能本体（EIO）语义层与可移植评估记录（PER）记录系统的双层架构，为AI智能体评估建立了可互操作的语义标准，使评估证据、论断与PASS/REVIEW/BLOCK决策之间形成可计算、可追溯的关联。 |
| [^180] | [Evaluating human-AI workflows for field research in viticulture](https://arxiv.org/abs/2610.07669) | 多智能体AI系统Aleks v1与人类迭代协作开发的红叶症状预测模型，通过模型引导的行优先级排序，将加州葡萄园田间调查中发现新记录红叶观测的比例从85.8%提升至94.1%，主要优化了区块间的调查资源分配。 |
| [^181] | [CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency](https://arxiv.org/abs/2610.07668) | CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。 |
| [^182] | [SENSE: State-aware Emotion Navigation Storytelling Engine](https://arxiv.org/abs/2610.07666) | SENSE是一个状态感知框架，通过集成MIND状态化叙事架构、结构分析器和路径感知上下文管理模块，能够从极少的高层输入生成结构连贯、情感丰富且支持多轨道情感导航的可玩分支视觉小说。 |
| [^183] | [Joint Workflow and Prompt Optimization for User Behavior Simulation](https://arxiv.org/abs/2610.07663) | SWORD框架基于角色化设计，仅依靠一个标量任务指标即可联合优化多智能体工作流拓扑与自然语言提示词，在用户行为模拟任务上显著超越仅优化提示词、仅优化工作流和分阶段优化的基线方法。 |
| [^184] | [Massive Activation Gating Channel in Large Language Models](https://arxiv.org/abs/2610.07661) | 该论文发现大语言模型中大规模激活现象由尖峰前馈网络输入嵌入中一个位置固定的“大规模激活门控通道（MAGC）”所控制，并通过跨四个模型家族六个模型的实验验证和理论分析揭示了其产生机制。 |
| [^185] | [Where Rules End and Judges Begin: Measuring the Judgment Boundary in Multi-Agent Systems Security](https://arxiv.org/abs/2610.07657) | 该研究提出DEFER1防御框架，通过28项确定性检查级联与四位裁判评审团协同工作，将多智能体系统的攻击成功率从约30%降至约3%，并实证划定了规则可处置与需裁判判断的安全边界。 |
| [^186] | [Does On-Policy Distillation for Safety Pose Backdoor Risks?](https://arxiv.org/abs/2610.07654) | 研究揭示了面向安全性的在线策略蒸馏（OPD）中一个被忽视的后门威胁：被植入后门的教师模型可将隐藏恶意行为传播给原本干净的学生模型，仅3%的投毒率即可使攻击成功率达70%，而增加训练轮数和常用的top-k KL方法会进一步加剧该风险。 |
| [^187] | [SMART: Zero-Shot Sim-to-Real Articulated Object Manipulation via Large-Scale Synthetic Pretraining](https://arxiv.org/abs/2610.07652) | 提出了SMART系统，其核心SMART-Sim仿真平台通过铰接感知设计实现大规模合成操作演示的生成与收集，实现了零样本的从仿真到真实的铰接物体操作。 |
| [^188] | [Matching Object or Relation? Tracing Abstract Reasoning Inside VLMs](https://arxiv.org/abs/2610.07646) | 该研究借鉴心理学的关系匹配样本范式并结合模型内部机制分析，识别出促使视觉语言模型从物体匹配转向关系匹配的四个关键因素，并揭示了其与人类“关系转变”相似的发展轨迹。 |
| [^189] | [SkillPoison: Progressive Skill Poisoning via Successful Experiences](https://arxiv.org/abs/2610.07645) | SkillPoison提出了一种新型技能投毒框架，它通过构建强化目标行为的成功经验并移除该行为适用条件的上下文约束，在不注入任何恶意内容、不使任何单条轨迹显式恶意的情况下，实现从经过验证的成功经验中对智能体技能库的渐进式投毒。 |
| [^190] | [Monte Carlo Estimation for KV Cache Eviction](https://arxiv.org/abs/2610.07643) | 提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。 |
| [^191] | [Towards the Automatic Synthesis of Interpretable Chess Tactics](https://arxiv.org/abs/2610.07640) | 本文提出一种受国际象棋战术启发、由归纳逻辑编程系统PAL所学模式推导而来的符号化子策略模型，通过融入领域知识提升可解释性，并提出散度度量评估方法，其合成的战术组合能给出与人类初学者棋力相当的走法建议。 |
| [^192] | [Learning Explainable Representations of Complex Game-playing Strategies](https://arxiv.org/abs/2610.07638) | 本文提出一种类似人类认知的方法，训练强化学习智能体将学到的游戏策略合成为基于动作序列的可执行程序，从而获得可解释的策略表示，并在国际象棋和网格环境任务中验证了其有效性。 |
| [^193] | [Measuring climate backlash in Twitter and Reddit archives: Lexical definitions, recorded responses and participant turnover](https://arxiv.org/abs/2610.07634) | 该研究以无抽样全量处理和透明、非排他性的词汇规则测量社交媒体上的气候反弹，揭示了词汇定义的宽窄会显著改变甚至逆转跨平台对比结果，并通过将回归作者的事件期变化与参与者更替分离来澄清所测量的社会过程。 |
| [^194] | [Learning to Outgrow a Theory: Experimental Discovery Beyond the Initial Hypothesis Space](https://arxiv.org/abs/2610.07627) | 该论文提出实验性模型类别修正方法，让发现策略联合提出结构性修改和诊断实验，借助类别级可区分性目标与任意时间有效的序贯证据，在当前假设类别被拒绝后才触发修正，从而在400个受控动力学环境中以32个实验预算实现89.5%的精确恢复率，超越最强基线10.0个百分点。 |
| [^195] | [Stateless Language Agents: Scaling Long-Horizon Automated Research](https://arxiv.org/abs/2610.07625) | 提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。 |
| [^196] | [Explore, Then Commit: Measurement-Efficient Scientific Law Discovery with Language Models](https://arxiv.org/abs/2610.07620) | 该论文提出了一种“先探索后确定”协议，通过语言模型提出假设、程序化规划器高效收集测量，将科学定律发现所需的测量次数最多减少约5倍，并将误差显著降低一个数量级以上。 |
| [^197] | [BioStudyBench: Evaluating Agents on Post-Cutoff Biomedical Studies](https://arxiv.org/abs/2610.07614) | 该论文提出BioStudyBench基准，基于模型知识截止日期之后发表的25个真实生物医学研究构建长时程任务，评估AI智能体自主查找公开数据、检索文献并通过数据分析复现已发表研究结果的能力。 |
| [^198] | [Linear Fitness Subspace in Protein Language Models Enables Sample-Efficient Directed Evolution](https://arxiv.org/abs/2610.07607) | 提出线性适应度子空间（LFS）假设并引入子空间引导进化搜索（SGES），通过在蛋白质语言模型突变引起的残基级表示变化中寻找与实验测定相关的紧凑方向集合，使适应度变化从少量标注样本中线性可获取，从而实现样本高效的模型引导定向进化。 |
| [^199] | [VALSE: Vertical Adaptive Layer Skipping for Efficient Inference in Large Language Models](https://arxiv.org/abs/2610.07606) | 本文建立了垂直自适应层跳过的理论框架（包括期望计算成本闭式公式、跳层模型函数空间的严格包含定理以及与混合专家架构的结构对偶性），并据此提出VALSE方法，利用轻量级难度评估器实现逐样本的非连续层跳过，以提升大语言模型推理效率。 |
| [^200] | [Emoception: Selective Affective Layer Fine-Tuning of Video Vision Transformers for Player Arousal Change Recognition From Gameplay Footage](https://arxiv.org/abs/2610.07603) | 提出选择性情感层微调（SALFT）方法，通过基于参数L2范数变化的层筛选准则，仅更新视频视觉Transformer约8%的参数即可在玩家情绪唤醒识别任务上达到与全量微调相当的性能。 |
| [^201] | [Beyond Scalar IoU: Structured Verification from Rollout Groups for Video Temporal Grounding](https://arxiv.org/abs/2610.07601) | 提出SUTURE方法，将验证从独立打分的标量IoU扩展为利用rollout组结构（组内分歧与位置覆盖率）的结构化验证，并可精确分解为标准IoU项加协方差修正项，从而改进基于可验证奖励强化学习的视频时序定位。 |
| [^202] | [Modeling Latent Disturbances for Robust Decision-Making in World Models](https://arxiv.org/abs/2610.07599) | 本文提出将潜在空间扰动建模为对习得潜在动力学的扰动，使其引发悲观但合理的状态转移，从而在世界模型的潜在空间中实现鲁棒决策。 |
| [^203] | [LSC-DPO: Learning-Signal-Controlled Direct Preference Optimization](https://arxiv.org/abs/2610.07592) | 该论文提出LSC-DPO方法，从损失几何视角将DPO损失中的sigmoid因子识别为学习信号，并通过动态调节该信号使其维持在目标区间，从而解决DPO训练后期敏感性下降的问题，在多个基准上持续超越DPO。 |
| [^204] | [Recurrent Looped Transformer](https://arxiv.org/abs/2610.07591) | 提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。 |
| [^205] | [Personal-Agent Mediated Recommendation with Cross-Platform User History](https://arxiv.org/abs/2610.07588) | 提出了“个人智能体中介推荐”这一新范式及MediateRec基准，研究个人LLM智能体如何利用用户授权的跨平台历史来调解平台推荐排序，在有益挽救与有害覆盖之间取得平衡。 |
| [^206] | [Mechanistic Interpretability of Atmospheric Rivers in GraphCast](https://arxiv.org/abs/2610.07583) | 该研究通过对GraphCast训练稀疏自编码器，首次揭示这一AI天气模型内部稳定地计算出大气河流强度（综合水汽输送IVT）作为内部变量，并通过干预实验证实了其因果作用。 |
| [^207] | [Representation Bias, Correction Transfer, and Resolution Sensitivity in Three-Dimensional Mitochondrial Morphometry](https://arxiv.org/abs/2610.07582) | 本文对三维线粒体形态计量学进行了实证可靠性评估，发现基于占据率的体积测量存在平均3.665%的系统性膨胀偏差，并证明通过移除深度偏移的校正迁移可将该体积误差降低约45%。 |
| [^208] | [LOGIC: An LLM Benchmark for Intent-Grounded Change Impact in Aerospace Electrical Systems](https://arxiv.org/abs/2610.07580) | LOGIC是一个航空航天电气系统领域的受控基准，用于评估语言模型能否根据工程请求意图从确定性候选变更清单中正确选择变更，并通过类型化电气可追溯性图传播其影响，实验表明仅门控结构化证据方法在明确锚定的选择案例上达到了完美的F1分数1.0000。 |
| [^209] | [Cooperating with Future Collaborators: Multi-Agent RL under Staggered Participation](https://arxiv.org/abs/2610.07578) | 该论文针对交错参与设定下多智能体强化学习的跨时间、跨智能体学习依赖问题，提出了交错参与学习（SPL）方法，通过为早期智能体提供前瞻性获取监督、为后续智能体提供基于结果的接收者学习，使早期智能体学会留下有用的任务相关信息、后续智能体学会有效利用这些信息。 |
| [^210] | [Unanimously Wrong: Certified Abstention from How Medical LLM Consensus Forms](https://arxiv.org/abs/2610.07570) | 论文指出医学LLM问答系统中基于答案一致性的置信信号无法区分“通过证据化解分歧达成的一致”与“所有样本共享同一误解导致的一致性错误”，并提出ProbeGuard框架，根据共识的形成过程而非最终状态做出可认证的弃权决策。 |
| [^211] | [OpenSplatGraph: From Dense Semantic Maps to Structured Scene Graphs for Open-Vocabulary Robot Perception](https://arxiv.org/abs/2610.07569) | 该论文提出OpenSplatGraph框架，首次实现直接从在线的基于高斯泼溅的开放词汇稠密语义地图中构建持久化三维场景图，通过可靠性感知的语义场进行置信度感知的对象提取，弥合了稠密语义建图与结构化场景图推理之间的鸿沟。 |
| [^212] | [Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation](https://arxiv.org/abs/2610.07565) | 该论文提出互补特征域（CFD）理论，证明保持香农信息并不能保证保持预测贡献，并用贡献缺陷量化重编码下上下文贡献的变化，其上界由可达动作集间的行为距离与联盟不相容性之和界定。 |
| [^213] | [Learning a Mixture of GFlowNets](https://arxiv.org/abs/2610.07562) | 提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。 |
| [^214] | [Navigating Route Latent Space for Synthesizable Molecular Design](https://arxiv.org/abs/2610.07560) | RouteFlow框架将可合成分子设计重构为在连续路线潜空间中的搜索问题，每个潜向量对应一条完整合成路线，并通过奖励引导的流匹配采样器实现性质优化与可合成性的内在统一。 |
| [^215] | [Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation](https://arxiv.org/abs/2610.07558) | 提出物理引导视觉提示（PG-VP），将不可见的辐射或温度危险转化为动态虚拟障碍物视觉提示，使冻结的VLA导航模型无需重新训练即可规避多种不可见风险。 |
| [^216] | [CheckerBench: Can Long-Horizon Agents Synthesize Static-Analysis Checkers?](https://arxiv.org/abs/2610.07557) | 该论文提出了首个可执行基准CheckerBench（包含源自297个CVE、167个仓库的300个任务），用于评估长程智能体能否在真实代码仓库中端到端合成可用的静态分析检查器，并配套CheckerLab统一评估框架衡量诊断对比度、补丁定位、误报率和工具使用等指标。 |
| [^217] | [Decoupled Multi-Agent Orchestration](https://arxiv.org/abs/2610.07556) | 提出 DeOrch 框架，将多智能体编排中的任务规划与工作者选择解耦，通过两阶段规划器和无身份信息的匹配性反馈实现条件化信用分配，支持新工作者在线加入而无需重新训练，并在分布内外任务上超越了先前的自动多智能体系统方法。 |
| [^218] | [Which and When to Admit: Gradient Admission for Data-Centric Small Language Model Finetuning](https://arxiv.org/abs/2610.07553) | 提出GRADE框架，通过状态感知选择器持续接纳与演化中的多任务梯度场对齐的样本，并用自校准步级门控在子空间接近饱和时拒绝破坏性更新，从而同时解决LoRA微调中的梯度冲突、静态数据选择和子空间饱和三大问题，提升小语言模型微调效果。 |
| [^219] | [Foundation Model-Aided Multi-Agent Reinforcement Learning for Wireless Random Access Network Optimization](https://arxiv.org/abs/2610.07550) | 该论文提出一种基础模型辅助的actor-critic多智能体强化学习算法，以显著降低无线随机接入网络优化任务中的训练开销，并证明了其与采用评论者模型交换和线性近似的传统MARL方法具有相同的收敛阶。 |
| [^220] | [A Systematic Investigation of Bias in Large Language Models for Advertising Relevance](https://arxiv.org/abs/2610.07544) | 该论文通过反事实框架系统性研究了大型语言模型在广告相关性判断中的公平性问题，发现广告主身份、输入语言以及人口统计学措辞（尤其在就业、住房和信贷等敏感领域）都会显著影响模型的判断结果，并呈现出与常见刻板印象一致的偏见。 |
| [^221] | [Disentangling Models from Personas in Heterogeneous LLM Simulations](https://arxiv.org/abs/2610.07535) | 该研究通过模拟由多个基础模型驱动的异构社交网络，发现智能体获得的互动量更多取决于其基础模型而非被分配的角色人格，且随着模型数量增加，模型间效应显著增强，表明网络动态可能在大规模下收敛于基础模型效应。 |
| [^222] | [Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge](https://arxiv.org/abs/2610.07532) | 提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。 |
| [^223] | [Grounding What Shapes the Plan: Rethinking Groundedness for Physical Intelligence in Autonomous Driving](https://arxiv.org/abs/2610.07521) | 论文提出GroundAct框架，以物理实体及其交互作为接地的基本单元，通过轻量级参考标记让被选中实体与不断演化的规划方案进行交互来修正规划，从而在自动驾驶中建立起从接地推理到规划动作的显式路径。 |
| [^224] | [Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates](https://arxiv.org/abs/2610.07518) | 该研究发现有害监督微调会在大语言模型的检查点更新中留下连续且可读取的痕迹，通过检查点级别的坐标即可高精度检测有害目标的存在，从而实现无需运行模型的安全审计。 |
| [^225] | [From Local Evidence to Safety Verdicts: Causal Tracing in Vision-Language Models](https://arxiv.org/abs/2610.07514) | 该研究提出SSU-Bench数据集并通过配对输入间的内部状态迁移实验，因果追踪视觉-语言模型中图文联合安全判断的形成位置，发现局部输入证据在早期解码器层即发挥作用，而完整的安全判定在较晚层汇聚于最终token，且可通过线性读出预测。 |
| [^226] | [On Open-Ended Information Seeking for Information Elicitation Agents](https://arxiv.org/abs/2610.07509) | 本研究通过在11个跨越不同家族和参数规模的大语言模型上进行的受控诱导模拟，揭示了不同LLM对信息价值的判断存在差异，且这些差异会显著塑造其序列化的开放式信息搜寻行为。 |
| [^227] | [Jarvis: A Proactive Speech Agent for Multi-Party Conversations](https://arxiv.org/abs/2610.07506) | 提出了Jarvis——一个能实时主动参与多人对话的语音代理，它在小组遗漏或误述事实且未自我纠正时进行干预，并提供了可量化评估的认知断裂基准数据集、基于小型开源模型且论断可溯源的主动式骨干系统，以及发言权交互技术三项关键贡献。 |
| [^228] | [MARS: Multi-resolution Adaptive Routing for Sequential Recommendation](https://arxiv.org/abs/2610.07505) | MARS通过将用户历史写入锚定不同时间半衰期的循环状态轨道，并利用稀疏路由读取器为每个种子按需选择相关时间分辨率来生成紧凑记忆，解决了缓存用户记忆中的“时间混叠”问题，从而显著提升序列推荐性能。 |
| [^229] | [Does Muon Need Fine-Grained Spectral Shaping?](https://arxiv.org/abs/2610.07497) | 本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。 |
| [^230] | [Who Bears the Burden? Learning Responsibility for Shared Constraints in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.07491) | 提出LiRA方法，通过优化社会福利来学习各智能体在共享拉格朗日乘子中的责任份额，从而解决多智能体强化学习中共享约束惩罚如何在不同智能体之间合理分配的问题。 |
| [^231] | [In With the Old: Enhancing 'Classical' Document Automation with Generative AI](https://arxiv.org/abs/2610.07480) | 本文探索了基于专家系统等符号方法的经典文档自动化与生成式AI如何相互增强，并通过初步实验证明大语言模型可用于识别和修复非专业人士撰写的法律文本中的问题。 |
| [^232] | [Can Power Draw Constrain Covert Compute? Limits of Analogue Verification for AI Governance](https://arxiv.org/abs/2610.07476) | 仅靠功耗等模拟测量手段无法有效约束隐蔽计算——最坏情况下隐藏计算量可达声明容量的116%，即使采用对抗性匹配能量策略也能隐藏至少41%的计算量，这说明AI治理协议不能仅依赖模拟验证机制。 |
| [^233] | [PsyCIDRA: A Dual-Agent Framework for Psychiatric Interviewing and Diagnostic Reasoning](https://arxiv.org/abs/2610.07473) | PsyCIDRA提出了一种双智能体框架，将自由形式的精神科访谈与诊断推理相结合，其访谈智能体借助工具笔记、专家技能和ICD-11参考指导问诊，诊断智能体报告假设及支持、冲突和缺失证据，在多个模型上取得了比直接提示更高的诊断一致性。 |
| [^234] | [Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits](https://arxiv.org/abs/2610.07470) | 提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。 |
| [^235] | [COMPASS: Finding Where Reasoning Lives in Language Models](https://arxiv.org/abs/2610.07469) | COMPASS利用模型自身直接回答正确性这一简单信号，在推理时通过识别并引导特定注意力头激活来引出推理能力，无需预先定义推理特征，在多个数学基准上优于现有激活引导方法。 |
| [^236] | [ElasticFit: Fit-Aware 3D Object Insertion via VLM Reasoning and Generative Adaptation](https://arxiv.org/abs/2610.07460) | 提出ElasticFit框架，利用VLM从语言指令和场景观察中推断结构化适配线索（落地位置、占用体积、朝向及适应模式），并将其转化为显式3D约束，实现感知适配的3D物体插入。 |
| [^237] | [Auditable Claims about AI Agents](https://arxiv.org/abs/2610.07459) | 提出AI智能体声明的可审计性标准——声明必须在事前明确其政策、范围、裁决记录及记录撰写者，并满足独立记录覆盖、授权绑定操作参数和超越完整性的完备性三项条件，才能被有效核查。 |
| [^238] | [AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation](https://arxiv.org/abs/2610.07457) | AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。 |
| [^239] | [Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden](https://arxiv.org/abs/2610.07452) | 该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。 |
| [^240] | [When Does AI Supervision Help? A Role-Aware Study of Network Fraud Decision Management with Blockchain Auditability](https://arxiv.org/abs/2610.07434) | 本文提出具有区块链可审计性的角色感知“决策者-监督者”框架，研究第二AI组件何时能改善网络欺诈决策，发现确定性硬门控可解决约89.994%的欺诈请求，而条件校准并不能带来一致可迁移的监督优势。 |
| [^241] | [Adaptive Gait Biofeedback With Participant-Held-Out Modeling and Participant-Specific Updating in Chronic Ankle Instability](https://arxiv.org/abs/2610.07428) | 本研究针对慢性踝关节不稳提出了一种自适应步态生物反馈方法，通过参与者留出的LOSO交叉验证严格评估时间卷积分类器的模型性能，并在训练失败后进行参与者特异性模型更新以提升分类效果，同时验证了自适应干预对额状面踝关节角度改善的效果。 |
| [^242] | [2d-fet-bench: from spatial reasoning to fet design on flakes](https://arxiv.org/abs/2610.07423) | 该论文推出了首个可执行基准 2D-FET-Bench V2，包含128个基于真实二维薄片轮廓的场效应晶体管版图设计任务，用以系统评估语言模型智能体将空间推理转化为器件版图构建的能力。 |
| [^243] | [Defense-in-Depth for LLMs: Evaluating Memory Gates Against Activation-Induced and Memory-Induced Sycophancy](https://arxiv.org/abs/2610.07403) | 该论文提出了一个 2×2 纵深防御框架，将内部激活引导与外部记忆处理分离，并引入包括新型“路由门控”在内的五种记忆防御配置，在 MemSyco-Bench 上系统评估并防御大语言模型中由激活诱发与记忆诱发的谄媚行为。 |
| [^244] | [Inference and learning in sparse autoencoders as natural gradient flow](https://arxiv.org/abs/2610.07389) | 该论文将稀疏自编码器的推理与字典学习统一为共享变分自由能上的自然梯度流，并提出无编码器的稀疏编码模型BeFOND，通过循环解释抵消机制减少重叠特征间的干扰、利用Fisher预条件化加速稀有特征学习，从而显著提升字典恢复与稀有特征检测能力。 |
| [^245] | [DeepAJM: Deep Association Joint Model for Irregularly Sampled data](https://arxiv.org/abs/2610.07388) | 提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。 |
| [^246] | [WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification](https://arxiv.org/abs/2610.07384) | 提出WildMatch，仅利用个体身份标签（无需关键点或几何对应标注）对预训练关键点匹配器进行弱监督适配，以解决相机陷阱图像中野生动物个体重识别的难题。 |
| [^247] | [MemCo: Memory-Centric Collaboration for Generalizing LLM Agents to Unseen Environments](https://arxiv.org/abs/2610.07376) | 提出以记忆为中心的协作框架MemCo，通过维护互补的本地与全局记忆空间来平衡记忆检索的粒度问题，从而将LLM智能体泛化到未见过的交互式环境中。 |
| [^248] | [Evaluate the Stack, Not the Layer: Do Deterministic and LLM Gates for Agent Actions Fail Independently?](https://arxiv.org/abs/2610.07359) | 研究发现，用于智能体动作防护的多个LLM裁判门控之间错误高度相关（堆叠两个裁判仅相当于约1.2-1.4个独立层，远低于2层的理想值），而确定性规则层与LLM裁判的组合则接近独立失效（约1.8-2.1层），因此多门控防御技术栈应作为整体评估而非逐层评估。 |
| [^249] | [A Validated Dataset and Benchmark for Coherent Multi-Diagram SysML Models](https://arxiv.org/abs/2610.07356) | 该论文提出了SEMAADB——一个包含3,000个工程情境、15,000张经过一致性和有效渲染验证的SysML多视图图的大规模数据集与基准，用于评估大语言模型生成连贯多图系统建模的能力。 |
| [^250] | [Evaluating Escalation Signals for LLM Routing: Targets, Controls, and Five Ways to Fool Yourself](https://arxiv.org/abs/2610.07354) | 本文验证了语义熵可作为LLM路由中判断是否升级到大模型的廉价有效信号（GSM8K上AUROC达0.871、准确率最多提升九个百分点），并提出了一套防止被“仅基于问题难度的简单规则”等虚假信号误导的严格评估检查方法。 |
| [^251] | [Trajectory-Retrieval Speculative Decoding: When Does a Model's Own History Help?](https://arxiv.org/abs/2610.07350) | 提出轨迹局部自适应检索（TLAR）方法，通过从模型自身推理轨迹中检索复用近似匹配的续写内容，并结合自适应检索策略与投机解码，在相同验证预算下提升token接受率、降低长链式思维推理的解码成本。 |
| [^252] | [RELACE: retrospective likelihood-based action credit estimation for long-horizon language agents](https://arxiv.org/abs/2610.07349) | RELACE 提出了一种无 critic 的信用估计框架，通过比较动作在原始上下文与结果增强上下文下的教师强制似然差异，生成轨迹归一化的回溯因子，从而为长程语言智能体提供更精确的动作级信用分配。 |
| [^253] | [Stepped MoE: Segment-Level Routing with Configurable Inference Complexity](https://arxiv.org/abs/2610.07348) | 本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。 |
| [^254] | [Rationale-Guided Policy Optimization: Learning to Reason with Adaptive Rationale Scaffolding](https://arxiv.org/abs/2610.07342) | 提出了理据引导的策略优化（RGPO）框架，根据模型当前能力自适应地利用真实理据信息作为支架，以缓解强化学习中的奖励稀疏问题，同时保留模型的探索自由。 |
| [^255] | [CausalBind: Causal Modeling and Learning for Protein-Molecule Virtual Screening](https://arxiv.org/abs/2610.07340) | 该论文提出CausalBind，通过因果建模识别并利用蛋白质-分子结合中稀疏的跨模态局部相互作用模式（如氢键、疏水接触、盐桥），从而克服传统密集整体对齐方法的局限，提升虚拟筛选向新靶点泛化的能力。 |
| [^256] | [A doctrine-grounded visual question answering dataset for Tactical Combat Casualty Care](https://arxiv.org/abs/2610.07339) | 本文提出TC3-VQA数据集，利用公开教学与实战视频和权威条令文档构建了1,860个将视觉证据与可追溯战伤救护条令关联的问答样本，为支持战术战斗伤员救护的视觉-语言模型开发提供监督数据。 |
| [^257] | [Logbook: Extremely Long-form Audio Event Understanding](https://arxiv.org/abs/2610.07338) | 该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。 |
| [^258] | [Selective Critique for Cost-Aware LLM Agents in Long-Horizon Decision Making](https://arxiv.org/abs/2610.07335) | 提出SAG框架，利用基于动作歧义信号（全局熵与top-2边际）的轻量级免训练门控机制，智能地选择在何时调用外部批判，从而在提升LLM智能体长时程决策可靠性的同时大幅降低token消耗和延迟。 |
| [^259] | [Memory-Efficient Expert Routing for Distributed MoE Training](https://arxiv.org/abs/2610.07333) | 提出RelayMoE，一种基于环形结构的MoE执行模型，通过让专家权重或token在环中循环流动并在专家路由与token路由间动态选择，避免了完整top-k扩展分发缓冲区的构建，显著提升了分布式MoE训练的内存效率。 |
| [^260] | [Scale-Invariant Training for Time Series Foundation Models](https://arxiv.org/abs/2610.07324) | 论文揭示了对仿射缩放（如ReVIN）进行逆变换会使各序列梯度被乘以b^p、使序列尺度成为隐含的重要性权重并导致高尺度序列主导训练的“尺度污染”问题，并证明直接在缩放后的目标上计算损失即可实现尺度不变的训练。 |
| [^261] | [Rule-Based Languages for Neurosymbolic AI](https://arxiv.org/abs/2610.07313) | 本文沿语义、表达能力、神经集成和求值机制四个维度，综述了神经符号AI中的Datalog、答案集和概率逻辑程序三类基于规则的语言，分析了50多个系统并提供了一个将应用场景映射到所需特性的决策矩阵。 |
| [^262] | [Understanding and Mitigating Inference-Time Overreliance Using Agentic Memory](https://arxiv.org/abs/2610.07311) | 发现智能体记忆在查询与过往经验仅部分重叠时会因证据无法完全迁移而误导推理，并提出即插即用框架MEMTRIM，通过写入时索引证据、读取时控制复用并移除重复或冲突的记忆，无需重新训练即可缓解LLM智能体的记忆过度依赖问题。 |
| [^263] | [From Sandbox to Enforcement: Confidence-Qualified Threat Intelligence for Critical Infrastructure](https://arxiv.org/abs/2610.07310) | 提出了CG-CTI流水线，将沙箱输出转化为带明确置信度标注的威胁情报，并通过知识图谱跨源佐证实现“只有高置信情报才能触发自动化处置”的分级把关机制，为关键基础设施防御提供从数据采集到自动执行的可靠闭环。 |
| [^264] | [The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory](https://arxiv.org/abs/2610.07309) | 该论文提出了一个检索可采性验证框架，通过将记忆-查询对标注为“可采纳、不可采纳或未决”三种状态，并在提示词暴露层面追踪记忆ID及其与目标级泄露的关联，从而检测长期记忆智能体在错误语境下检索出“正确的记忆”这一安全隐患。 |
| [^265] | [Polar: LLM-Powered Synthesis of Real-World Cyber Evidence for Prioritization and Mitigation](https://arxiv.org/abs/2610.07298) | POLAR是一个由大语言模型驱动的框架，将分散在厂商通告、漏洞数据库和威胁情报中的真实世界网络证据合成为以威胁为中心的评估，通过结合严重性推断与按时间排序的利用信号实现威胁优先级排序，并关联权威修复知识以支持按紧急程度组织的缓解行动。 |
| [^266] | [Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale](https://arxiv.org/abs/2610.07289) | 本文提出部署于Google的AI智能体FlowAgent，通过ReAct风格的生成-验证循环与弃权过滤器，在持续集成的提交前阶段以低延迟实时自动修复测试失败，使开发者无需切换上下文即可在心流中获得高质量修复建议。 |
| [^267] | [FlexiFlow: Bandit-based Model Switching in ML Workflows](https://arxiv.org/abs/2610.07286) | FlexiFlow是一个基于多臂老虎机的动态模型切换数据流系统，综合考虑模型准确率、运行时间和断言通过概率，在当前模型表现不佳时自动切换到更优模型，可将机器学习工作流准确性提升高达23%。 |
| [^268] | [SAFESHIELD: A Decision-Organization Framework for Deployment-Time Safety of Small Language Models](https://arxiv.org/abs/2610.07276) | 本文提出SAFESHIELD框架，将小语言模型的部署时安全形式化为决策组织问题，通过组织准入、路由、证据和发布四种安全决策职责，并将决策记录于可审计的决策轨迹中，实现了安全决策的显式组织、协调与审计。 |
| [^269] | [A Trust Layer for Agent Evaluation](https://arxiv.org/abs/2610.07274) | 该论文提出“智能体评估信任层”，一个附加式后验框架，通过验证评分逻辑支持、可追溯计算、完成声明一致性和重复执行稳定性这四个属性，来判断智能体在基准测试中获得的分数是否可信。 |
| [^270] | [Does the Model Use the Feature? Separating Steering from Mechanism in LLMs](https://arxiv.org/abs/2610.07270) | 该论文提出仅在自然输入观测值范围内评估特征的植入、移除与下游拯救测试，发现特征的操控能力与模型实际使用它的程度可以显著分离，从而纠正了将“可操控”直接等同于“机制”的误判。 |
| [^271] | [Verifying Coordination in Parallel Coding Agents: NP-Bench and a Scheduling Planner](https://arxiv.org/abs/2610.07261) | 该论文将并行 LLM 编程智能体之间的协调问题重新表述为预先调度问题——划分互不重叠的工作范围并按生产者->消费者图安排合并顺序，据此构建了 Nerveplane 规划器，并提出了通过真实 git 合并验证集成效果的三臂基准测试 NP-Bench。 |
| [^272] | [Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents](https://arxiv.org/abs/2610.07258) | 该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。 |
| [^273] | [MemMux: Runtime Verification and Honest Resource Attribution for Fleets of Parallel Coding Agents](https://arxiv.org/abs/2610.07257) | MemMux 是一个本地运行时系统，它将并行编码智能体集群的资源治理转化为可检查的运行时验证信号，实现了每个智能体的内存归因、完全资源回收、逃逸进程可见性以及内存超载下的有界占用，解决了现有工具无法追踪和管理多智能体内存资源的问题。 |
| [^274] | [Internalizing Agent Experience into Diffusion Model Weights via On-Policy Context Distillation](https://arxiv.org/abs/2610.07250) | 提出扩散在线策略上下文蒸馏（D-OPCD）方法，将智能体框架优化提示词的知识作为特权上下文蒸馏进扩散模型权重，使模型无需运行完整智能体框架即可获得部分性能提升。 |
| [^275] | [Can Semantic Geometry Teach an AI Judgement?](https://arxiv.org/abs/2610.07249) | 该研究探索能否通过将行为与政策表示为向量并利用其几何关系，让AI智能体在行动前自动识别并遵循隐含在语言中的规则，但四项实验表明这些几何方法未能建立可靠的行动前判断。 |
| [^276] | [SPECTRUM: Proximal Spectral Modulation for Looped Self-Distillation](https://arxiv.org/abs/2610.07237) | 提出SPECTRUM方法，通过近端谱调制在循环自蒸馏中缓解“正确性提升但正确解多样性收缩”的问题，在MBPP五轮实验中保留了89.9%的正确解多样性。 |
| [^277] | [Minimal Witness Reinforcement Learning](https://arxiv.org/abs/2610.07226) | 本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。 |
| [^278] | [TIDE 2.0: an open, model-agnostic engine for keyed de-identification of clinical notes](https://arxiv.org/abs/2610.07224) | TIDE 2.0是一个开源、模型无关的临床笔记去标识化引擎，通过密钥化匿名技术——包括保持时间间隔的日期偏移和密码学生成的替代值——在保护患者隐私的同时保留数据的纵向分析价值，且无需依赖外部硬件或存储关联表。 |
| [^279] | [Cascadia: Resident 975B MoE Inference on Eleven AI PCs](https://arxiv.org/abs/2610.07219) | 本文提出Cascadia系统，通过自研常驻MoE引擎将975B总参数、41B活跃参数的Inkling模型分布式部署于十一台仅64GB内存的AI PC上实现推理，并将密集层调用时间从8.1毫秒降至4.5毫秒。 |
| [^280] | [RoboCap: A New Platform for Egocentric Robot Learning](https://arxiv.org/abs/2610.07217) | 论文提出了RoboCap——一款250克、六摄像头双IMU的第一人称数据采集帽子，配合Grounded 3D算法套件，实现了硬件、标定与算法的垂直整合，在SLAM、深度估计和手部跟踪等公开基准上达到最先进性能，为大规模机器人学习数据采集提供了全新平台。 |
| [^281] | [Distributionally Robust Mixture-of-Experts Training](https://arxiv.org/abs/2610.07207) | 提出 DRMoET 分布鲁棒训练目标，将各层专家视为内生鲁棒性分组、优化高损失路由结果而非仅均衡负载，在多个模型规模下均提升了 MoE 的下游性能。 |
| [^282] | [Energy-Conditioned Noise Schedule and Whitening for Spectral Diffusion](https://arxiv.org/abs/2610.07206) | 提出了一种将全局谱白化与能量条件化噪声调度相结合的谱扩散新方法，使前向扩散过程中的噪声注入能根据图像的谱能量分布自适应调节，同时保持与标准DDPM/DDIM框架的兼容性。 |
| [^283] | [Responsible Institutional Analytics: Interpreting Bias with AI Support](https://arxiv.org/abs/2610.07205) | 提出了FACTRIA框架，将院校分析中的潜在偏差因素组织为四个维度，并通过生成式AI聊天机器人引导用户反思这些因素，研究表明结构化框架与AI引导相结合能够促进更具情境意识的负责任解读。 |
| [^284] | [SPEAR: Five Principles for Interactive Human-Agent Alignment](https://arxiv.org/abs/2610.07204) | 该立场论文提出SPEAR框架，将人-智能体对齐从传统的部署前优化问题重新界定为一个持续的交互设计问题，涵盖规范明确化、过程、评估、适应和再校准五大支柱。 |
| [^285] | [Sim-to-Real Transfer of Vision-Language Navigation in Continuous Environments Using an Ackermann-Steered Mobile Robot](https://arxiv.org/abs/2610.07192) | 本文提出了一种基于跨模态注意力架构的视觉语言导航方法，无需导航图和全景视图，通过仿真到真实的领域迁移与真实数据微调，成功将连续环境中的视觉语言导航部署到定制的阿克曼转向移动机器人上。 |
| [^286] | [Agentic Design Space Exploration for Joint Hardware Configuration Selection and Mapping of AI Inference Workloads on Heterogeneous Edge SoCs](https://arxiv.org/abs/2610.07191) | 该论文针对异构边缘SoC上AI推理工作负载的联合硬件配置选择与映射问题，指出传统黑盒优化反馈稀疏且未利用LLM推理能力的局限，提出了一种智能体式（Agentic）设计空间探索新方法。 |
| [^287] | [CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks](https://arxiv.org/abs/2610.07177) | 该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。 |
| [^288] | [CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets](https://arxiv.org/abs/2610.07132) | 该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。 |
| [^289] | [Is this machine playing?](https://arxiv.org/abs/2610.07130) | 研究者将无任何任务、奖励或活动指令的具身AI置于未知虚拟世界中，观察到其自发产生攀爬、堆叠、绘画等符合玩耍经典判据的行为，并据此提出玩耍或可成为机器自主发展的一种新模式。 |
| [^290] | [Aggregating User Preferences while Ensuring Equity, Diversity, and Inclusion using Graph Summarization](https://arxiv.org/abs/2610.07128) | 该论文提出一种EDI约束的图摘要方法，将公平性、多样性与包容性准则直接嵌入图结构，通过贪心粗化算法在聚合多元用户偏好时避免经典聚合规则偏向多数群体、同质化或忽视少数群体的问题。 |
| [^291] | [PlaySuite: A Large-Scale Benchmark for Interactive Visual Intelligence](https://arxiv.org/abs/2610.07127) | PlaySuite是一个基于5000多款开源游戏的大规模基准测试，用于评估AI模型在动态环境中的交互式视觉智能，并配备了针对HPC集群优化的统一闭环交互框架和基于Video-LLM的评判协议，以实现可扩展且防作弊的评估。 |
| [^292] | [Jailbreaking Open-Weight LLMs via Random Embedding Perturbations](https://arxiv.org/abs/2610.07125) | 该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。 |
| [^293] | [AMBER: Training Long-Horizon Web Agents through Append-Only Memory](https://arxiv.org/abs/2610.07118) | 提出AMBER方法，利用追加式记忆训练长时程网络智能体，从而解决覆写式记忆在稀疏结果奖励下难以学会跨多次重写保留事实信息的问题。 |
| [^294] | [Will the Judge Flip? Predicting Position-Sensitive LLM Judgments from Residual Stream Activations](https://arxiv.org/abs/2610.07115) | 本研究提出用线性探针读取LLM裁判判定前的残差流激活，无需按两种顺序重复判定即可预测其是否会因候选回答顺序而翻转结论，且跨基准迁移效果良好、无需重新校准。 |
| [^295] | [LiLib: Lifelong Air-to-Ground Path-Loss Prediction on UAVs via a Drift-Triggered Model Library](https://arxiv.org/abs/2610.07111) | 提出LiLib轻量级持续学习方案，无人机通过维护递归最小二乘专家库，在检测到环境漂移时复用或创建专家，将空地路径损耗预测RMSE从5.89 dB降至4.03 dB，并大幅降低重访已知环境后的误差。 |
| [^296] | [Beyond Successor Accuracy: State Retention for Recursive Self-Improvement in Recommendation](https://arxiv.org/abs/2610.07105) | 该论文提出“分布式进展”概念和跨代优势（CGA）度量，揭示推荐递归自改进中更新前后的模型可能保留互补的排序决策，并利用无需标签的排序分离统计量预测应保留旧模型还是新模型，实验显示最优保留策略因推荐架构而异。 |
| [^297] | [Muon Is Theoretically Wrong For Convolutions, But Empirically Effective](https://arxiv.org/abs/2610.07103) | 该研究指出将卷积核重塑为矩阵的标准Muon实现在理论上有缺陷，作者提出了理论上更严谨的卷积Newton-Schulz方法（Conv-NS），但实验发现两者性能相当，揭示了优化器理论与实践之间的差异。 |
| [^298] | [When to Remember, When to Abstain: Category-Conditioned Retention for Reliable Agent Memory](https://arxiv.org/abs/2610.07100) | 该论文提出按断言语义类别设置差异化置信度阈值的记忆保留策略，解决了单一全局阈值无法应对价值观/信念类断言（仅77.9%可靠）与其他类别断言（96.2%可靠）之间可靠性不对称的难题。 |
| [^299] | [T-CCL: Resource Efficient and Performant Collective Communication using Tensor Memory Accelerator](https://arxiv.org/abs/2610.07098) | T-CCL通过将数据搬运与归约操作卸载为流水线化的异步TMA操作，在保证高性能节点内集合通信的同时大幅降低了SM侧资源占用。 |
| [^300] | [Verified, not generated: expert-verified AI study materials and the distribution of learning gains in a university course](https://arxiv.org/abs/2610.07097) | 该研究发现，经专家验证的AI生成学习材料使大一经济学课程学生在50分制考题上平均多得2.34分，并使低分段成绩占比下降24.7个百分点，表明将筛选AI输出的判断负担从学生转移给负责任的导师，能够缩小学生间的成绩差距。 |
| [^301] | [Small Language Models for Smart Data Model Classification at the Edge: A Cost-Aware Hybrid Approach](https://arxiv.org/abs/2610.07093) | 该研究提出一种成本感知的混合方法，系统评估了包括通用型、推理专用型和代码专用型在内的轻量级开源语言模型在资源受限的边缘环境中进行智能数据模型分类的性能，以解决物联网异构数据的互操作性难题。 |
| [^302] | [Smart Content Ingestion for Generative AI Workloads](https://arxiv.org/abs/2610.07091) | 本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。 |
| [^303] | [Towards a Unified Misuse Monitoring Benchmark](https://arxiv.org/abs/2610.07089) | 该论文提出了一个统一的轨迹级滥用监控形式化框架，并构建了包含约6,200份对话记录的基准，首次将分解攻击与提示注入攻击纳入同一评估体系，以“危害窗口”为标准衡量监控器何时能及时识别有害行为。 |
| [^304] | [SchemaFill: Efficient LLM Tool Calling via Slot-Parallel Speculative Decoding](https://arxiv.org/abs/2610.07086) | SchemaFill提出了一种槽位并行投机解码框架，通过并发生成工具调用中未来槽位值的候选来加速大语言模型的工具调用，且无需预先获知实际的调用序列或参数值。 |
| [^305] | [Graph-Based Recognition of Simulated Train-Driver States From Facial and Upper-Body Keypoints](https://arxiv.org/abs/2610.07083) | 本研究提出一种仅使用单个前置RGB摄像头和图神经网络，通过结合面部与上身骨骼关键点特征的视觉监控系统，可高精度地将列车驾驶员状态分类为警觉、非警觉和紧急三类。 |
| [^306] | [Demo: Vision-Language Model-Guided Online Calibration of an Electromagnetic Digital Twin](https://arxiv.org/abs/2610.07081) | 该论文提出一种视觉-语言模型引导的框架，通过材料分类和路径点规划两次VLM调用，仅用宇树G1机器人在20米移动范围内即将电磁数字孪生的电导率校准误差降至1.74×10⁻⁴，解决了传统在线校准中初始化敏感和移动成本高的问题。 |
| [^307] | [CuratorMAS: Automating Dataset Curation via Multi-Agent Orchestration](https://arxiv.org/abs/2610.07075) | CuratorMAS是一个多智能体协作框架，通过将复杂的数据集策展流程分解为五个可编程执行阶段并编排多个智能体协作评估与筛选数据，实现了高质量数据集策展的自动化及跨领域泛化能力。 |
| [^308] | [Learning to Simulate Individuals from Macro Social Signals](https://arxiv.org/abs/2610.07062) | 该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。 |
| [^309] | [TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning](https://arxiv.org/abs/2610.07043) | 提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。 |
| [^310] | [Inference-Time Projection for Physically Valid Biomolecular Diffusion Models](https://arxiv.org/abs/2610.07037) | 该论文提出将物理有效性视为约束推理问题，在推理时对扩散模型的去噪坐标估计施加两个闭式投影算子（如链间范德华投影），以极低的额外开销确保生物分子复合物预测的物理有效性，避免了物理势能引导的高计算成本或模型微调的架构耦合。 |
| [^311] | [JIVEAdapter: A Multi-Task Additive Low-Rank Adapter via Joint and Individual Variation Explained (JIVE)](https://arxiv.org/abs/2610.07036) | JIVEAdapter借鉴统计学中的JIVE方法，将多任务低秩适配器的权重更新分解为跨任务共享的联合结构与近似正交的任务特定个体结构，并自适应分配秩，使冻结后的联合结构可作为先验直接复用于新任务，实现高效且可解释的多任务参数微调。 |
| [^312] | [Investigating Model Compression for Neural Machine Translation in the Biomedical Domain](https://arxiv.org/abs/2610.07032) | 本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。 |
| [^313] | [An Empirical Fault Vulnerability Exploration of ReRAM-based Process-in-Memory CNN Accelerators](https://arxiv.org/abs/2610.07029) | 本文开发了一个故障注入框架，首次在软件与硬件两个层面实证探索了基于ReRAM的存内计算CNN加速器在推理阶段运行大规模CNN时的故障脆弱性。 |
| [^314] | [Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, Quantization and Benchmarking](https://arxiv.org/abs/2610.07026) | 该论文提出了一个面向非洲语言社区的全离线语音优先AI技术栈，整合了模块化离线架构、低成本硬件参考配置与可复现的量化流水线，并首次在Jetson Orin NX和树莓派5两个硬件层级上对2-5B参数的指令微调模型进行了端到端基准评估。 |
| [^315] | [Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment](https://arxiv.org/abs/2610.07023) | 提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。 |
| [^316] | [When to Rethink: Learning Multi-Perspective Self-Verification for Vision-Language Models](https://arxiv.org/abs/2610.07018) | 该论文提出MOTIVE框架，通过多视角自验证与可靠性引导的选择性重新思考机制，解决了视觉语言模型中因单一验证标准或固定提示词导致的可靠性估计不完整且不稳定的问题。 |
| [^317] | [Anchor and Adapt: Asymmetric Prompt Adaptation for Few-Shot Industrial Anomaly Detection](https://arxiv.org/abs/2610.07016) | 该论文提出了一种两阶段提示学习框架“锚定与自适应”，第一阶段从带标注的辅助数据中学习可迁移的正常与异常锚点，第二阶段冻结这些锚点并仅利用少量目标正常样本适配额外的正常分支，从而免除了人工构建产品特定异常描述的需求。 |
| [^318] | [Which Image Property Carries the Jailbreak? A Controlled Dissection of Image-to-Text Jailbreaks](https://arxiv.org/abs/2610.07009) | 本文通过控制变量的受控实验系统解剖图像到文本越狱攻击，发现真正驱动越狱成功的是攻击图像本身，而块的熵、JPEG大小等密度特征以及块数量结构均无法可靠区分攻击图像与良性图像。 |
| [^319] | [Where Does the Audio Jailbreak Live? A Controlled Frequency-Depth Audit of AdvWave-P on Qwen2-Audio](https://arxiv.org/abs/2610.07005) | 该论文对AdvWave-P音频越狱扰动在Qwen2-Audio上进行受控的STFT频带掩蔽审计，发现攻击成功率的频率排序依赖于频带划分方式，且掩蔽7520-7960 Hz这一窄频带可将攻击成功率降至0.10。 |
| [^320] | [Topology-Consistent Task Planning over Cellular Workflow Complexes for LLM-based Agents](https://arxiv.org/abs/2610.07004) | TopoPlanner提出了一种拓扑一致的规划框架，通过将工具依赖图提升为胞腔工作流复形，并利用余层一致的胞腔检索与多维结构推理，使LLM智能体能够有效处理验证-修正循环、分支汇聚和可复用中间状态等现实工具编排中的复杂工作流模式。 |
| [^321] | [Mask-Guided KV Cache Eviction in Block Diffusion Language Models](https://arxiv.org/abs/2610.06996) | 提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。 |
| [^322] | [Joint upper-bound coverage and route-choice utility: an empirical evaluation on two urban proxy tasks](https://arxiv.org/abs/2610.06995) | 基于北京和成都道路数据的实证评估表明，更高的路径时间上界联合覆盖率和更准确的速度预测并不一定能改善路径决策，反而会轻微增加迟到率和平均旅行时间。 |
| [^323] | [TARE: Weigh a Never-Poisoned Twin Before Reading Backdoor-Defense Costs](https://arxiv.org/abs/2610.06994) | 本文提出TARE方法，指出后门防御的代价评估存在根本性缺陷——仅在受投毒模型上测量的准确率下降无法区分防御的真实移除效果与其对任何模型的通用作用，并揭露了BackdoorBench基准中因配置错误（如永不触发的学习率调度器）导致的虚假结论，主张在解读防御代价前应先在从未被投毒的孪生对照模型上进行测量。 |
| [^324] | [DART-ES: Difficulty-Aware Reweighting and Targeted Replay for Fine-Tuning LLMs with Evolution Strategies](https://arxiv.org/abs/2610.06993) | 提出 DART-ES 方法，通过从扰动种群通过率构建动态难度状态，同时实现难度感知的奖励重加权与罕见可解样本的定向回放，在不引入额外难度模型或反向传播的前提下提升了进化策略微调大语言模型的效果。 |
| [^325] | [When better traffic forecasts fail to improve signal control: a layered diagnostic study of forecast-to-decision value](https://arxiv.org/abs/2610.06992) | 该研究通过分层诊断框架揭示了更准确的交通预测未能转化为更好信号控制决策的原因——决策时刻信息泄漏、九个受控交叉口中仅两个具备多种有效动作、以及保形预测区间在高需求场景下覆盖率大幅下降。 |
| [^326] | [EPOCH: Reliable Discovery through Evidence-Governed Search](https://arxiv.org/abs/2610.06986) | EPOCH提出了一种证据治理的发现架构，通过任务契约、类型化记忆、主动证伪、准入检查和独立重放来约束AI研究智能体对评估反馈的解释与复用，防止脆弱的候选方案被误当作可靠发现，并在AlgoTune基准上以0.65的平均归一化得分大幅超越最强基线0.53。 |
| [^327] | [CrystalJev: thinking fast and slow with atomistic foundation models for materials discovery](https://arxiv.org/abs/2610.06985) | CrystalJev 将原子级基础模型从慢速模拟器转变为快速决策者，通过对每个结构仅做一次前向传播、以校准概率回答材料稳定性等问题，成本仅为传统弛豫计算的三十分之一，并借助信息价值理论精准判断何时才需要启动更昂贵的慢速计算。 |
| [^328] | [Visual-Invariance-Augmented Feature Optimal Alignment for Transferable Adversarial Attacks against Closed-Source MLLMs](https://arxiv.org/abs/2610.06977) | 提出IAU-FOA攻击方法，通过全局余弦对齐与基于最优传输的补丁级细粒度局部对齐相结合，显著提升对抗样本对闭源多模态大语言模型的定向迁移攻击能力。 |
| [^329] | [AegisFlow: A Multi-Agent Agentic AI Framework for Autonomous Remediation and Self-Healing in Fragile Data Ecosystems](https://arxiv.org/abs/2610.06971) | AegisFlow是一个多智能体AI框架，通过Watchdog智能体收集运行时遥测、Repair智能体基于LLM自动生成并部署代码补丁，并采用基于MAPE-K循环的“并行影子补丁”非侵入式模型在数字孪生环境中验证补丁，从而实现脆弱数据管道从故障检测到自主修复的闭环自愈。 |
| [^330] | [APEX: Active Protection at Execution Boundaries for LLM Agents](https://arxiv.org/abs/2610.06966) | APEX 提出在 LLM 智能体的执行边界（即内部状态转化为外部动作或输出的位置）上，依据事先编译的单一授权契约，同时校验“动作效果是否被授权”与“运行时信息是否被可信任务背书”，从而以与注入载体无关的稳定方式防御间接提示注入攻击。 |
| [^331] | [Principles that Guide, Actions that Inform: Agent Evolution via Knowledge Abstraction](https://arxiv.org/abs/2610.06964) | 该论文提出SAGA方法，通过将智能体的具体交互经验抽象为可复用的通用知识原则，使LLM智能体无需修改模型参数即可实现自我演化并提升泛化能力。 |
| [^332] | [Verdicts Without Annotated Evidence: Rejection Sampling or Label-Only Post-Training for Evidence Recovery?](https://arxiv.org/abs/2610.06962) | 该研究表明，在没有任何人工证据标注的情况下，仅用判定标签进行后训练的小语言模型在证据恢复上优于基于自动来源接地分数的拒绝采样方法，且判定准确率与证据跨度一致性仅弱相关，说明准确率不能作为引用可审阅性的可靠指标。 |
| [^333] | [Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering](https://arxiv.org/abs/2610.06950) | 该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。 |
| [^334] | [Metonymic Circuits for Abstract Concept Grounding in Vision Transformers](https://arxiv.org/abs/2610.06928) | 视觉Transformer通过具体可解释的锚定概念（如“火”）作为转喻中介来接地抽象概念（如“愤怒”），形成了从感知基元到物体锚点再到抽象目标的结构化转喻电路。 |
| [^335] | [AttSVD:Prompt-Adaptive Low-Rank KV Cache Compression via Attention-Guided SVD](https://arxiv.org/abs/2610.06927) | 提出AttSVD，一种基于每个提示自身注意力几何结构的可解释低秩KV缓存压缩方法，通过在线逐提示截断SVD仅保留注意力实际读取的方向，在保留全部token的同时按保留秩比例削减长上下文下的KV缓存内存，并提供累积式与流式两种解码时缓存策略及自适应压缩改进。 |
| [^336] | [RadOnc-Agent: An LLM-Orchestrated Framework for AI Workflows Across the Radiotherapy Care Pathway](https://arxiv.org/abs/2610.06923) | 该论文提出RadOnc-Agent，一个由大语言模型驱动的智能体框架，通过对话界面将放射治疗形式化为四个临床阶段并提供26个可调用功能，从而将分散在不同临床阶段和软件环境中的AI能力统一编排为完整的放射治疗工作流。 |
| [^337] | [Anchor Divergence for Semantic Geometry in Contrastive Learning](https://arxiv.org/abs/2610.06919) | 本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。 |
| [^338] | [FluidPD: In-Place Elasticity for SLO-Aware Prefill-Decode Disaggregated LLM Serving](https://arxiv.org/abs/2610.06917) | FluidPD是一个SLO感知的预填充-解码分离式LLM服务系统，通过两个互补机制实现原位弹性，无需备用GPU即可应对预填充与解码需求比例的短时突发和持续偏移，避免延迟SLO违规。 |
| [^339] | [Text2Dashboard: A Governed Agent Architecture for Natural-Language Dashboard Generation over Enterprise DataBrain](https://arxiv.org/abs/2610.06914) | 提出了Text2Dashboard——一种受治理的智能体架构，通过将模式约束的模型决策与类型化工具、确定性Hooks相结合，把自然语言分析请求安全地转换为可审计、可检查的企业数据仪表板，在真实DataBrain任务上取得了较高成功率。 |
| [^340] | [GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets](https://arxiv.org/abs/2610.06910) | GameGo 提出了一个可扩展的框架，通过将简短的游戏种子系统性地转化为基于行业游戏开发实践的产品需求文档，并利用锚定于真实世界资产的合成轨迹来训练游戏开发智能体，从而实现高质量的端到端浏览器游戏自动生成。 |
| [^341] | [Component and Dimension Sparsity in Transformer Refusal Mechanisms](https://arxiv.org/abs/2610.06903) | 该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。 |
| [^342] | [Axiom Satisfiability of Linear Rewards in Alignment](https://arxiv.org/abs/2610.06892) | 该论文通过引入逐候选者松弛量，提出一种计算“总松弛最小且满足公理边际η”的线性奖励的方法，在不对投票者和数据收集方式做任何假设的情况下，以被O(1)界定的松弛代价强制线性奖励满足帕累托最优与PMC等公理。 |
| [^343] | [Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes](https://arxiv.org/abs/2610.06889) | 该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。 |
| [^344] | [Dynamical low-rank equilibrium computation for stochastic games between advanced persistent threats and moving target defense](https://arxiv.org/abs/2610.06885) | 该论文揭示了工业控制系统中APT攻击与MTD防御的影响矩阵具有内在低秩性，并证明该低秩结构可通过零和随机博弈的非光滑贝尔曼算子传播，从而在显式误差界保证下实现高效且鲁棒可认证的均衡计算。 |
| [^345] | [Learning When to Refine: Long-Horizon Reinforcement Learning for Budgeted Neural-Operator PDE Solvers](https://arxiv.org/abs/2610.06883) | 该论文提出面向预算约束神经算子PDE求解的长时程强化学习方法RV-PI，通过实际滚动验证在有限修正预算下学习何时何地施加局部细化，仅在留出轨迹误差改善时接受策略更新。 |
| [^346] | [Comparative review of hybrid forecasting models for short-term prediction of building thermal load](https://arxiv.org/abs/2610.06881) | 本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。 |
| [^347] | [Neutrosophic Ensemble Classification for Uncertainty-Aware Bearing Fault Detection: Evidence from Laboratory and Variable-Speed Industrial Benchmarks](https://arxiv.org/abs/2610.06880) | 本文提出将中智学四指标分解（最高类证据、最佳竞争者证据、预测熵与决策分歧）应用于机器学习集成分类器，以区分轴承故障检测中自信的错误与真正模糊的预测，实现不确定性感知的故障诊断，并在实验室与变速工业基准上验证了其有效性。 |
| [^348] | [When Can World Models Recover Physical Laws?](https://arxiv.org/abs/2610.06877) | 本文指出仅凭准确预测不能证明世界模型恢复了物理定律，并给出定律可恢复的充要条件——任意两条不同定律必须在实验上可区分——同时用率-失真下界、有限响应码本、明确解码预算和极小极大采样复杂度刻画了恢复的信息论极限与稳定性。 |
| [^349] | [TasteVal: Measuring the Experimental Research Taste of AI Systems Against Human Experts](https://arxiv.org/abs/2610.06824) | 提出TasteVal基准，将AI的实验研究品味量化为计算效率，用于评估前沿模型在固定研究问题上迭代设计实验并得出结论的能力，使其成为预测AI进展的关键参数。 |
| [^350] | [TAPDreamer: Transferable Adversarial Patches for World Action Models](https://arxiv.org/abs/2610.06814) | 提出了TAPDreamer攻击方法，仅利用公开编码器即可构建无需查询目标模型的对抗补丁，实现跨任务和跨动作架构迁移地攻击世界动作模型。 |
| [^351] | [Local2Mesh: Spatially Localized Contour-to-Mesh for Left Ventricular Reconstruction from Sparse 2D Cardiac MRI](https://arxiv.org/abs/2610.06052) | 提出空间局部化的Local2Mesh框架，通过几何感知对齐纠正切片错位、利用平面感知局部路由器将轮廓特征精确分配至模板顶点，从而无需3D网格标注即可从稀疏2D心脏MRI轮廓重建3D左心室几何结构。 |
| [^352] | [Incentive Alignment in Online Experimentation](https://arxiv.org/abs/2610.05922) | 该论文将在线实验重新建模为激励设计问题，揭示了实验者基于有偏实验结果获得奖励所导致的委托-代理冲突会侵蚀平台价值，并证明样本拆分和收缩估计这两种实用机制能够有效实现激励对齐。 |
| [^353] | [Data-Driven Personas for Survey Simulation: Insights into Simulation Alignment Across Data-Access Regimes](https://arxiv.org/abs/2610.05828) | 该研究发现，从域外公共行为数据诱导的画像在模拟特定人口群体的调查响应时很少优于仅使用基本人口统计信息的模拟，主要原因是源数据与目标人群不匹配。 |
| [^354] | [TeleTune: Evolving Agent Skills From Offline Telemetry](https://arxiv.org/abs/2610.05437) | TeleTune提出了一个从无目标标注、无法重放且任务交错的离线用户遥测日志中，通过动作预测误差自动演化文本技能库的框架，使计算机使用智能体能够学到可复用的软件操作技能。 |
| [^355] | [EnGRICH: Enhancing Generative Reward Modeling with Critiques from Humans](https://arxiv.org/abs/2610.05370) | 提出 EnGRICH 框架，通过从稀缺的人类批评意见中学习评估标准，并将其泛化到仅有结果监督的偏好数据上，从而提升生成式奖励模型批评意见的可靠性。 |
| [^356] | [When Agent Context Goes Stale: Incoherence in Volatile Agent Context](https://arxiv.org/abs/2610.05281) | 提出上下文一致性框架 Concord，通过将工具观测结果与其数据源关联、检测数据源变化，并在复用前自动更新、标注或抑制过时上下文，避免智能体基于过期信息做出错误判断。 |
| [^357] | [A Safe Action Is Not Enough: Feasible-Future Decoding for Vision-Language-Action Policies](https://arxiv.org/abs/2610.05166) | 该论文揭示并解决了冻结VLA策略中的“可行性-似然差距”问题，提出通过近似计算候选动作的可行未来质量来进行解码，从而选出既局部安全又保有策略支持的安全任务完成路径的动作。 |
| [^358] | [Best-of-$N$ Guidance for Test-time Diffusion Alignment](https://arxiv.org/abs/2610.05108) | 该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。 |
| [^359] | [How Much Do LLM-as-a-Judge Design Choices Matter? A Systematic Comparison of Prompt Designs, Rating Scales, and Models](https://arxiv.org/abs/2610.05094) | 该研究系统评估了10个推理模型在不同提示词设计、评分量表和任务下的LLM评判者表现，发现尽管与人类基准存在统计显著分歧，但实际差异很小、大多数评判设计仍然可靠，从而为LLM-as-a-judge的设计选择提供了实证依据。 |
| [^360] | [E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation](https://arxiv.org/abs/2610.05048) | 论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。 |
| [^361] | [A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese](https://arxiv.org/abs/2610.04898) | 本研究提出最短匹配序列（SMS）对齐方案以解决中文分词与语言模型子词分词不一致的问题，并利用从零训练的Chinese-Pythia模型证明语言模型意外度确实能预测中文阅读时间，且预测能力随模型规模的缩放模式因语料库而异。 |
| [^362] | [SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models](https://arxiv.org/abs/2610.04875) | SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。 |
| [^363] | [More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding](https://arxiv.org/abs/2610.04753) | 提出稀疏非对称分组查询注意力SAGA，通过解耦键头与值头数量——用更少的键头加速推理、保留更多值头维持模型容量——从而实现更快的LLM解码。 |
| [^364] | [Fine-Tuning VLM for Enhancing AI's Spatial Intelligence: Understanding 3D and 2D Rotations](https://arxiv.org/abs/2610.04206) | 该研究通过在物体旋转数据集上微调视觉语言模型，显著提升了AI对2D和3D旋转的空间推理能力，其中微调后的Gemma-4混合专家模型表现优于通用模型。 |
| [^365] | [Agentic AI with Structured CoT for Enhancing AI's Spatial Intelligence: Visualization and Reasoning of Rotation](https://arxiv.org/abs/2610.04188) | 本研究通过结构化思维链（CoT）推理策略显著提升了生成式智能体AI模型（GPT-5.6）在三维空间旋转理解任务中的空间推理能力。 |
| [^366] | [OncoNoteBERT: A Foundation Representation Model for Natural Language Processing of Real-World Outpatient Oncology Notes](https://arxiv.org/abs/2610.03829) | 本研究基于包含29万余份真实世界英国门诊肿瘤笔记的受治理语料库，开发并评估了两种肿瘤学专用语言模型——使用肿瘤学分词器从头训练的OncoNoteBERT和通过持续预训练得到的OncoNote-RadBERT，以更好地表示肿瘤学专业术语和临床表达。 |
| [^367] | [WAMJET: A Harness for World Action Model Acceleration](https://arxiv.org/abs/2610.03797) | WAMJET是一个代理式加速框架，通过为编码智能体提供可复用的优化指导和测量验证工具，以瓶颈驱动的迭代方式自动优化世界动作模型的推理，在保持动作质量的同时实现了高达9.95倍的无损加速。 |
| [^368] | [Benchmarking Candidate Coverage in Typed Decision Models](https://arxiv.org/abs/2610.03387) | 本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。 |
| [^369] | [Trading Strategy Optimization via Textual Gradient](https://arxiv.org/abs/2610.03128) | 提出了TradeGrad框架，通过利用积累的优化经验来估计文本梯度并结合多尺度修订策略，克服了传统文本梯度优化短视及忽视时间稳健性的问题，实现了更稳健的量化交易策略优化。 |
| [^370] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^371] | [LUMOS: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs](https://arxiv.org/abs/2610.02902) | LUMOS诊断框架利用完全透明的OLMo 2训练语料库，沿“训练数据暴露→行为输出”的因果链追踪大语言模型的参数化知识，揭示模型内部能高可分性地编码罕见事实（84%）但在行为上表达不足（54%），且该检索差距随模型规模增大而缩小。 |
| [^372] | [BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration](https://arxiv.org/abs/2610.02800) | BitNest提出了一种比特嵌套的投机解码框架，将低精度草稿模型直接嵌入高精度目标模型的权重表示中，通过残差细化使两者共享单一物理权重，从而在加速大语言模型推理的同时显著降低内存开销。 |
| [^373] | [PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers](https://arxiv.org/abs/2610.02793) | PAPER2LLM++ 提出了一个让大语言模型从研究论文中持续自我演化的框架，通过提取论文中的研究发现、验证局限性是否仍然存在，并借助“尝试-评估-提交”机制整合更新，从而在不遗忘先前改进、不损害通用能力的前提下实现模型的自动改进。 |
| [^374] | [Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning](https://arxiv.org/abs/2610.02687) | 该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。 |
| [^375] | [Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis](https://arxiv.org/abs/2610.02659) | 该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。 |
| [^376] | [Mapping the RAG Landscape: A Four Axis Taxonomy of Efficiency, Defense, Interactivity, and Reasoning](https://arxiv.org/abs/2610.01936) | 本综述提出了一个涵盖效率、防御、交互性与推理四个维度的RAG分类体系，系统梳理了检索增强生成领域超越基础架构的最新研究进展。 |
| [^377] | [Completion Aware Guidance for World Action Models](https://arxiv.org/abs/2610.01559) | 提出无需训练的完成感知引导（CAG）采样方法，引导世界动作模型的生成朝向任务完成，显著提升机器人任务成功率并将任务不完整想象从 79% 大幅降至 40%。 |
| [^378] | [When Does a Second Model Help? Cross-Model Review in LLM Verification](https://arxiv.org/abs/2610.01471) | 在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。 |
| [^379] | [SpikeMoE: Brain-Inspired Competitive Routing for Flexible Spiking Mixture-of-Experts](https://arxiv.org/abs/2610.01418) | 提出SpikeMoE框架，通过受海马CA1区竞争-抑制机制启发的脉冲k-WTA路由器（含侧向抑制与不应期），将神经元尺度的脉冲动力学与模型尺度的专家选择相结合，并支持缺失模态建模，实现灵活的脉冲混合专家架构。 |
| [^380] | [ReSolve: Reusing Candidate Reasoning through Selective Generative Moderation](https://arxiv.org/abs/2610.01140) | ReSolve 是一种无需训练的推理流程，通过选择性生成式调解复用采样的候选推理，在竞赛数学题上超越投票和八样本自洽性方法，同时将 token 消耗降低约 46%-47%。 |
| [^381] | [Does Scaling Reinforcement Learning Really Require More Training?](https://arxiv.org/abs/2610.01133) | 提出策略空间扩展方法SURGE，通过对同一强化学习运行中的两个检查点（高精度锚点与生成简短回复的供体）进行特征空间融合，在不增加训练或推理计算的情况下获得比原检查点更强的策略。 |
| [^382] | [Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions](https://arxiv.org/abs/2609.38869) | 该论文提出了一个结合时序SHAP证据与历史市场状态类比的LLM叙述框架，通过渐进式推理外化减少大语言模型在推断数值变化时的错误，从而提升股票收益预测解释叙述的忠实性。 |
| [^383] | [KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy](https://arxiv.org/abs/2609.38480) | KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。 |
| [^384] | [JudgeProfile: Understanding and Steering Subjectivity in LLM Judges](https://arxiv.org/abs/2609.36705) | 该论文提出JudgeProfile框架，将大语言模型评判者的评估分解为感知与优先级排序两个层面，发现评判者在感知层面存在隐藏共识而在属性权重上存在个体差异，从而实现对评判者主观性的理解与引导。 |
| [^385] | [MLToolBench: Learning Tool-Augmented Agents for Machine Learning Development](https://arxiv.org/abs/2609.36679) | 该论文提出MLToolBench可执行诊断工具套件和SPICE回合级奖励方法（通过特权上下文衡量工具动作的价值），结合SFT与RL流水线，训练机器学习开发智能体学会何时使用诊断工具并依据诊断结果采取行动。 |
| [^386] | [FineART: Fine-grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation](https://arxiv.org/abs/2609.36416) | 本文提出了包含40,543个回合、1,718小时和533,913个密集子任务标注的双手操作数据集FineART，以及能自主预测下一个子任务的视觉-语言-动作模型FineART-VLA，显著提升了长时程双手操作任务的成功率。 |
| [^387] | [Infrared Subtraction with Artificial Intelligence](https://arxiv.org/abs/2609.36007) | 该论文提出了一种由大语言模型在人类物理指导下开发的局域红外减除方法，通过分离可积辐射项与玻恩接触项，实现了无切片参数、可复用低阶计算和有效场理论预言的局域减除公式。 |
| [^388] | [AX is the New AEO](https://arxiv.org/abs/2609.34951) | 该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。 |
| [^389] | [Calibrated Uncertainty for Informative Path Planning in Aquatic Environmental Monitoring](https://arxiv.org/abs/2609.34577) | 用校准良好的深度集成替代高斯过程可为水域环境监测的信息路径规划提供更可靠的不确定性估计，在原油泄漏场景模拟中将归一化重构误差降低83%。 |
| [^390] | [SWE-Game: Can Coding Agents Build the Games We Want?](https://arxiv.org/abs/2609.33678) | SWE-Game是一个基于41个可执行Godot游戏构建的编程智能体基准，涵盖从简报开发、文档实现、骨架补全、故障修复到跨引擎移植五种任务，结果显示最强模型Opus5的总分也不足60分，表明智能体构建完整游戏的能力仍有巨大提升空间。 |
| [^391] | [Action Shaping: Policies Absorb What They Can Express](https://arxiv.org/abs/2609.32752) | 论文提出“动作塑形”原理：训练时加入的动作偏移量若能被策略输出层精确表达，策略便会将其完全吸收，从而可在部署时安全移除该偏移而不损失回报，其最简实现为零初始化线性头配合可学习门控，门控会自行先升后降。 |
| [^392] | [Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions](https://arxiv.org/abs/2609.32652) | 该论文为几何诱导软状态抽象的线性预测精度建立了无需拟合预测矩阵即可由独立评估数据计算的有限样本下置信界，并揭示了核仿射包机器坐标的重构分数裕度与预测误差界之间的联系。 |
| [^393] | [Subjects, Not Authors: The Authorship Hazard in Agentic Dataspaces](https://arxiv.org/abs/2609.30614) | 该论文提出“作者身份危害”概念并确立核心原则：LLM智能体应始终是数据空间治理平面的主体而非作者，其发布授权通道须在构造上被关闭，其起草内容的影响则作为执行问题加以管控。 |
| [^394] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^395] | [Segment-Level Risk Discovery in Online Handwriting for Alzheimer's Disease Detection](https://arxiv.org/abs/2609.29384) | 该论文提出NormPaST-Risk网络，将阿尔茨海默病在线手写检测从整体轨迹表示重新表述为局部疾病相关片段发现，通过多尺度时间编码与选择性纸-空状态空间建模实现可解释的片段级风险识别。 |
| [^396] | [TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents](https://arxiv.org/abs/2609.26911) | TwinCheck提出了一种推理时验证策略，通过构建基于证据的“负孪生”反事实替代方案，仅在满足证据条件、通过结构检查并在顺序无关的成对验证中胜出时才替换智能体的工具调用，从而在不引入新失败的前提下提升有状态工具代理的多轮任务成功率。 |
| [^397] | [JEV-as-a-Judge: Accept When Confident, Escalate When Unsure](https://arxiv.org/abs/2609.26550) | 提出仅返回标签概率的低成本裁判JEV，通过置信度阈值机制将高置信判定直接接受、不确定判定升级至推理型裁判，从而在费用仅为GPT-6的41%的情况下，实现了比GPT-6高出0.9个百分点的评估准确率。 |
| [^398] | [How Children Design and Reason about Trustworthy AI Chatbots](https://arxiv.org/abs/2609.25244) | 本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。 |
| [^399] | [ValueDiff: Value-Geometric KV Cache Eviction for Sink-Suppressed LLMs](https://arxiv.org/abs/2609.23314) | 针对因QK归一化等技术导致注意力汇聚减弱的现代大语言模型，提出基于价值向量与缓存均值L2偏差进行token排序的ValueDiff淘汰方法，在2k-4k token预算下可保留密集注意力88-99%的性能。 |
| [^400] | [Local Sparsity Enables Unsupervised LLM Safety Detection](https://arxiv.org/abs/2609.20129) | 本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。 |
| [^401] | [CSWAM: Better Causal Semantic Representations for Out-of-Distribution Generalization in World Action Models](https://arxiv.org/abs/2609.18462) | CSWAM通过引入基于V-JEPA 2.1的因果语义专家模块，从稀疏观测历史中学习具有时间基础、少依赖外观细节的语义表示，显著提升了世界动作模型在视觉分布偏移下的泛化能力。 |
| [^402] | [SpliTEE: Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing](https://arxiv.org/abs/2609.15039) | 该论文提出SpliTEE，将拆分推理架构扩展到LLM推理场景，用差分隐私（而非加密）保护发送到不受信任GPU的中间表示，从而在可信硬件上实现高效且隐私安全的大语言模型推理。 |
| [^403] | [UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model](https://arxiv.org/abs/2609.09815) | UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。 |
| [^404] | [FrogNano: Training a 4B Coding Agent via Online Task Synthesis](https://arxiv.org/abs/2609.07925) | FrogNano是一个4B编码智能体，仅通过强化学习在约1500个合成任务环境中训练，其关键创新是在线任务合成流水线能在当前模型可学习性前沿生成校准任务，无需从大模型蒸馏即可训练出具有竞争力的小型编码智能体。 |
| [^405] | [Cost-Aware Hierarchical Multi-Agent Ransomware Detection and Family Attribution](https://arxiv.org/abs/2609.04820) | 该论文提出一种成本感知的分层多智能体系统，以低成本静态分析优先、按需调用动态和内存分析，并引入成本模型在保证勒索软件检测与家族归因性能的同时有效平衡计算开销。 |
| [^406] | [Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models](https://arxiv.org/abs/2609.00355) | 该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。 |
| [^407] | [The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection](https://arxiv.org/abs/2608.26423) | 本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。 |
| [^408] | [PertMind: Eliciting Emergent Biological Reasoning in LLM via Reinforcement Learning on Cellular Perturbation Data](https://arxiv.org/abs/2608.16419) | PertMind通过将细胞扰动图谱转化为强化学习环境，仅用正向预测训练便激发了大语言模型的涌现生物推理能力，并实现了跨任务的零样本迁移。 |
| [^409] | [Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers](https://arxiv.org/abs/2608.14089) | 本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。 |
| [^410] | [Do AI weather models miss extremes?](https://arxiv.org/abs/2608.09972) | 本研究利用十个月欧洲站点观测对十二个 AI 与物理气象模型进行评估，发现 AI 模型在极端天气上并不存在统一的低估缺陷，尾部表现取决于具体模型、变量和评估环境，但所有模型都普遍存在对极端值向均值收缩的共同条件误差模式。 |
| [^411] | [Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers](https://arxiv.org/abs/2608.07436) | 论文发现Muon训练的Transformer在grokking后发生崩溃的根源在于AdamW读出层更新与较大的特征均值相互作用产生类别相关的logit偏移，并叠加交叉熵导数错误，而修正这些错误可以稳定训练并恢复近乎完全的准确率。 |
| [^412] | [WebRider: Persona-Conditioned Intent Controllers for Live-Web Assistance](https://arxiv.org/abs/2608.06704) | 本文提出WebRider，将用户委托网络任务时的策略要求形式化为“意图契约”，并通过分层控制架构使实时网络智能体不仅完成任务，更能忠实遵守目标、约束与偏好——揭示并解决了现有智能体“只看最终答案”导致的高完成率但低履约率问题。 |
| [^413] | [Strategic Evaluation of Planning Strategies for LLM Agents in Cyber-Physical Systems](https://arxiv.org/abs/2608.04265) | 该论文提出了一个受控基准，通过在信息物理系统需求响应场景中严格隔离比较四种规划执行策略，评估LLM智能体规划策略在自主参与者响应与物理约束下的战略适用性。 |
| [^414] | [Rethinking Modality Reliability in Multimodal Sentiment Analysis with Incomplete Observations](https://arxiv.org/abs/2608.03611) | 本文提出显式建模模态可靠性以解决不完整多模态情感分析中的可靠性不匹配和传播偏差问题。 |
| [^415] | [Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text](https://arxiv.org/abs/2607.26368) | 该研究提出金融披露文本的细粒度不一致性分类任务，在统一评估协议下系统比较了多种模型方法，发现微调的3亿参数编码器可与大得多的提示大语言模型和LoRA适配模型相媲美，并进一步探究了冲突声明定位对分类性能的改善作用。 |
| [^416] | [Multi-Scale Structural Features for Continual, Comprehensible Visual Recognition in a Developmental Learning Framework](https://arxiv.org/abs/2607.25531) | 该论文提出了一种在多个尺度上编码边缘与轮廓结构及其空间关系的新型视觉特征表示，并将其融入无梯度的发展式学习框架，从而在无需回放缓冲区的条件下实现更准确、可持续且可解释的视觉识别。 |
| [^417] | [Variational-Ising-Attention:Tailored Attention Matters for Science](https://arxiv.org/abs/2607.23634) | 提出变分伊辛注意力（VIA），通过带相互作用的伊辛模型与变分平均场推断增强softmax注意力，将注意力从孤立条目的排序扩展为相互作用实体的集体状态建模，并在逆合成反应中心预测与蛋白质残基接触预测等科学结构化预测任务上验证了其有效性。 |
| [^418] | [Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling](https://arxiv.org/abs/2607.20791) | 提出拒绝门控解码（RGD），一种高效解码方法，能在高温采样下保持模型原有的拒绝行为，同时对其他提示直接从精确的高温分布采样，且几乎不增加额外延迟。 |
| [^419] | [Benchmarking the Personalization Capabilities of Large Language Models](https://arxiv.org/abs/2607.20471) | 该论文提出了SDR-Arena框架，首次实现了对双方（发送方-接收方）生成式个性化能力的大规模基准测试，利用销售外联数据将特定内容与接收方实际行为（回复、通话或成交）的真实标准关联起来。 |
| [^420] | [Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare](https://arxiv.org/abs/2607.17508) | RAIL是一种概率元学习框架，能够从自然语言任务描述出发，通过检索相关源任务并进行系数空间结构迁移，零样本生成具有特征级解释和不确定性量化的可解释医疗临床预测模型。 |
| [^421] | [Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents](https://arxiv.org/abs/2607.12397) | 提出无需训练的评论者经验库（CEB）框架，通过将已完成轨迹的事后反馈转化为可复用的经验证据，实现LLM智能体自进化的步骤级置信度估计。 |
| [^422] | [3D-DefectBench: A Controlled Factorial Study of Vision-Language Model Evaluation Pipelines for Fine-Grained 3D Generation Defects](https://arxiv.org/abs/2607.10826) | 该论文提出3D-DefectBench大规模基准，通过84种受控因子实验设计系统研究VLM评估流水线各环节（模型选择、相机协议、视觉输入、提示模式）对细粒度3D生成缺陷自动评估可靠性的影响，发现模型选择是决定与人工标签一致性的主导因素。 |
| [^423] | [Guided Action Flow: Q-Guided Inference for Flow-Matching Vision-Language-Action Policies](https://arxiv.org/abs/2607.02092) | 本文提出了一种推理时引导方法，通过任务特定评论者的梯度引导冻结的流匹配VLA策略，在不微调的情况下显著提升真实机器人任务性能。 |
| [^424] | [Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic](https://arxiv.org/abs/2606.23044) | 本文提出素数傅里叶嵌入，将整数编码为按素数索引的 (cos, sin) 对，借助舒尔引理从理论上证明等变线性映射必为每个素数对应一个独立块的块对角结构，并结合中国剩余定理预测与超过 500 倍特化比的消融实验验证，表明模运算可归结为选择相关的素数通道。 |
| [^425] | [Calibration Is Not Control: Intervention Value for LLM-Agent Oversight](https://arxiv.org/abs/2606.21399) | 该论文指出校准的失败分数不足以决定何时干预LLM智能体，提出改以“干预价值”作为监督目标，在ALFWorld上将后悔值从0.51大幅降至0.09，显著优于传统阈值式监督规则。 |
| [^426] | [Explaining Attention with Program Synthesis](https://arxiv.org/abs/2606.19317) | 该论文提出通过程序合成将Transformer语言模型的注意力头行为转化为可执行的Python程序，利用预训练语言模型生成并根据保留数据筛选程序，证明不到1000个程序即可复现GPT-2和TinyLlama-1.1B中注意力头的注意力模式。 |
| [^427] | [SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs](https://arxiv.org/abs/2606.16193) | SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。 |
| [^428] | [OSGuard: A Benchmark for Safety in Computer-Use Agents](https://arxiv.org/abs/2606.15034) | OSGuard是一个双粒度安全基准，通过包含324个人工标注样本的动作级护栏分类任务，以及基于OSWorld改造的45个风险增强执行任务，来评估计算机使用智能体的安全性，能够区分真正安全的任务完成与违反环境约束的表面成功。 |
| [^429] | [Sensitivity Shaping for Latent Modeling](https://arxiv.org/abs/2606.14585) | 本文提出支撑条件化的控制敏感性正则化方法，通过增强学习动力学模型在支撑良好区域对控制变化的局部响应性，避免不支持的控制被错误映射为看似正常的潜在预测，从而显著改善分布外检测并实现更安全的闭环规划。 |
| [^430] | [Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning](https://arxiv.org/abs/2606.13607) | 该研究通过对比46个大语言模型与两批人类参与者的日常常识推理表现，并分析内容不变与内容敏感神经元的作用，发现人类与LLM呈现出趋同的推理失败模式，从而挑战了“人类推理依赖内容不变的世界模型而LLM仅做模式匹配”这一传统假设。 |
| [^431] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^432] | [SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents](https://arxiv.org/abs/2606.05761) | 提出了SubtleMemory基准，通过构建关系受控的语义工件并嵌入真实的用户-智能体交互历史中，系统评估长期运行AI智能体在细粒度关系记忆判别（包括互补、细微及矛盾关系）方面的能力。 |
| [^433] | [Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?](https://arxiv.org/abs/2606.05647) | 本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。 |
| [^434] | [FFR: Forward-Forward Learning for Regression](https://arxiv.org/abs/2606.03927) | 提出FFR框架，首次将前向-前向学习算法从分类扩展到真实世界的回归任务，通过序数竞争好感度函数等三项关键创新解决了连续目标空间缺乏对比“对立面”的问题，并在多个真实数据集上取得了有竞争力的性能。 |
| [^435] | [Task diversity produces systematic transfer but inhibits continual reinforcement learning](https://arxiv.org/abs/2606.00880) | 该论文提出了GPU加速的持续强化学习环境Banyan，可通过参数化控制任务多样性，并发现任务多样性能带来系统性迁移能力，但会抑制智能体在连续分布偏移中的持续学习能力。 |
| [^436] | [The Terminal Representation in Reinforcement Learning](https://arxiv.org/abs/2605.31289) | 本文提出了强化学习中一种结构上全新的终止表示（TR），它类似于默认表示（DR）编码奖励加权轨迹，但能以更低维度学习，且无需特征向量计算即可直接支持选项发现、奖励塑形、迁移学习和探索等下游任务。 |
| [^437] | [Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?](https://arxiv.org/abs/2605.30557) | 该论文提出SPATIALUNCERTAIN受控评估框架，系统研究视觉语言模型在面对遮挡导致的证据缺失和视角导致的证据误导时的表现，强调可靠空间推理还要求模型能够判断当前观察是否足以支撑答案并主动识别更有信息量的观察视角。 |
| [^438] | [FHRFormer: A Self-Supervised Masked Transformer Framework for Fetal Heart Rate Time-Series Inpainting and Forecasting](https://arxiv.org/abs/2605.29695) | 该论文提出了FHRFormer，一种自监督掩码Transformer框架，能够修复和预测可穿戴胎心率监测中因信号丢失而产生的数据缺口，从而为基于AI分析连续胎心率数据以预测新生儿呼吸辅助风险奠定基础。 |
| [^439] | [Voluntary Collusion with Secret Tools in Competing LLM Agents](https://arxiv.org/abs/2605.27593) | 该论文通过竞争性欺骗和资源管理两个多智能体实验环境发现，即使工具被明确标记为不公平且有害，大多数LLM智能体仍会自愿接受秘密合谋工具并发展合谋策略，且仅靠不公平标签或基线安全对齐无法阻止这种行为，只有明确的伦理框架才能有效抑制合谋。 |
| [^440] | [D3S2: Diffusion-Guided Dataset Distillation for Semantic Segmentation](https://arxiv.org/abs/2605.25022) | 提出了首个面向语义分割的扩散引导数据集蒸馏框架D3S2，通过类别平衡掩码选择与扩散引导图像合成的两阶段设计，解决了长尾类别不平衡、像素级对齐和高计算成本三大挑战。 |
| [^441] | [Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs](https://arxiv.org/abs/2605.24154) | Palette 提出了一个模块化、可控且高效的框架，通过多目标搜索识别拒绝方向并借助轻量级适配将其内化到模型中，从而按需放宽授权领域的安全拒绝行为，同时保持其他领域的标准安全性。 |
| [^442] | [Reinforcement Learning over Predictive Distributions for LLM Regression](https://arxiv.org/abs/2605.20740) | 提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。 |
| [^443] | [To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents](https://arxiv.org/abs/2605.18882) | 该研究提出并验证了“内在偏差假说”（IBH），揭示大语言模型智能体在调用/不调用决策中存在与激活无关的过度调用偏差，利用稀疏自编码器定位并量化该偏差，并通过自适应边际校准转向（AMCS）方法实现了因果层面的纠正。 |
| [^444] | [KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat](https://arxiv.org/abs/2605.18850) | 本文提出了KadiAssistant，一个集成于Kadi研究数据生态系统中、以隐私保护为设计核心的对话式AI智能体，使研究人员能够高效访问、聚合和综合异构且隐私敏感的研究数据信息。 |
| [^445] | [Stochastic Penalty-Barrier Method for Constrained Machine Learning](https://arxiv.org/abs/2605.18618) | 本文提出随机惩罚-障碍方法（SPBM），通过对偶变量指数平均、稳定惩罚调度和Moreau包络扩展经典惩罚-障碍方法以求解约束机器学习问题，证明了小批量采样下变换问题可行集包含于原问题可行集，实验表明其性能与最先进方法相当。 |
| [^446] | [Voice "Cloning" is Style Transfer](https://arxiv.org/abs/2605.16578) | 这篇论文揭示语音克隆并非真正“克隆”个人声音，而是系统性地进行风格迁移，使克隆语音比源语音显得更权威、温暖且更像人类，并导致说话人特征的同质化。 |
| [^447] | [PBT-Bench: Benchmarking AI Agents on Property-Based Testing](https://arxiv.org/abs/2605.15229) | PBT-Bench是一个包含100个覆盖40个真实Python库的基于属性测试问题的基准，通过注入默认随机输入几乎无法触发的语义bug，专门评估AI智能体从文档中推导语义不变量并设计精确输入生成策略的能力。 |
| [^448] | [Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning](https://arxiv.org/abs/2605.13335) | Ego2World将第一人称烹饪视频编译为由图转移规则控制的可执行符号世界，使智能体在部分可观测条件下仅凭局部观测和执行反馈进行信念状态规划，弥合了被动视频数据集与合成模拟器之间的差距。 |
| [^449] | [When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction](https://arxiv.org/abs/2605.12922) | 该论文提出“通道转换”机制来解释大语言模型在多轮交互中丢失指令主线的原因，并引入目标可及性比率这一新指标，揭示不同架构的模型在注意力衰减后呈现出性质截然不同的失败模式。 |
| [^450] | [Reward on Path: Learning Intermediate Supervision Signals for Knowledge Graph Question Answering](https://arxiv.org/abs/2605.10791) | 提出RoP框架，通过非对称目标从答案标签中学习轻量级、问题条件化的路径奖励，为知识图谱问答中基于LLM的关系路径生成器提供有效的中间监督信号。 |
| [^451] | [Learning Visual Feature-Based World Models via Residual Latent Action](https://arxiv.org/abs/2605.07079) | 该论文提出了一种可从DINO残差中轻松学习的新型潜在动作表示“残差潜在动作”（RLA），并基于流匹配构建RLA世界模型（RLA-WM），实现了更高效、更少幻觉且预测质量更优的视觉特征世界模型。 |
| [^452] | [Rethinking Adapter Placement: A Dominant Adaptation Module Perspective](https://arxiv.org/abs/2605.06183) | 该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。 |
| [^453] | [XDecomposer: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction](https://arxiv.org/abs/2605.05866) | 提出XDecomposer，将多相XRD分析形式化为集合预测问题，无需候选相列表、结构模板或相数量先验即可实现多相XRD图谱的联合分解与结构识别。 |
| [^454] | [Von Neumann Networks](https://arxiv.org/abs/2605.05780) | 该论文将冯·诺依曼上世纪的细胞计算模型与现代深度学习相结合，提出了冯·诺依曼神经元及其网络（VNNs），其架构可自组织生成、仅依赖于输入输出在细胞阵列上的位置，并在数学上基于神经算子扩展与格林函数学习。 |
| [^455] | [SDFlow: Similarity-Driven Flow Matching for Time Series Generation](https://arxiv.org/abs/2605.05736) | SDFlow提出了一种在冻结VQ潜在空间中运行的相似性驱动流匹配非自回归框架，通过全局传输映射消除曝光偏差，并结合低秩流形分解与离散监督，实现高质量的时间序列并行生成。 |
| [^456] | [Algorithm Selection with Zero Domain Knowledge via Text Embeddings](https://arxiv.org/abs/2604.19753) | ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。 |
| [^457] | [On the use of evolutionary optimization for the dynamic chance constrained open-pit mine scheduling problem](https://arxiv.org/abs/2604.13385) | 本文提出一种结合基于多样性变化响应机制的双目标进化优化方法，用于求解块段经济价值随机且采矿与加工能力随时间变化的动态机会约束露天矿调度问题，在最大化期望折现利润的同时最小化其波动性。 |
| [^458] | [Cross-Cultural Value Attribution in Large Vision-Language Models](https://arxiv.org/abs/2604.09945) | 该论文首次系统研究大型视觉语言模型在道德、伦理和政治价值观判断上如何随图像中人物的文化语境（宗教、国籍、社会经济地位）而变化，通过反事实图像集与多维度评估框架揭示其中的跨文化刻板印象与公平性问题。 |
| [^459] | [What do your logits know?](https://arxiv.org/abs/2604.09885) | 该论文首次系统比较了视觉-语言模型在不同表示层次（残差流、tuned lens投影、top-k logits）上保留的信息，发现即使是最易访问的top logit值也能泄露图像查询中与任务无关的信息，其泄露量在某些情况下与完整残差流的直接投影相当，揭示了模型内部信息泄露的安全风险。 |
| [^460] | [Sinkhorn doubly stochastic attention rank decay analysis](https://arxiv.org/abs/2604.07925) | 本文证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准softmax行随机注意力更能有效保持网络深度中的秩，从而缓解秩崩溃问题。 |
| [^461] | [Neural Global Optimization via Iterative Refinement from Noisy Samples](https://arxiv.org/abs/2604.03614) | 本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。 |
| [^462] | [Unbiased Reward Modeling from Implicit Feedback for LLM Alignment](https://arxiv.org/abs/2603.23184) | 该论文提出ImplicitRM方法，通过将训练样本分层为四个潜在组并构建理论无偏的似然最大化目标，从点击、复制等隐式用户反馈中学习无偏奖励模型，从而解决了隐式反馈缺乏明确负样本和存在选择偏差这两大挑战。 |
| [^463] | [FSCE: A Target-Aware Frequency-Spatial Collaborative Enhancement Framework for Noise-Resilient SAR ATR](https://arxiv.org/abs/2603.21565) | 提出了FSCE框架，通过在网络入口处进行频率-空间协同增强以抑制斑点噪声传播、稳定浅层特征，并结合自适应策略驱动的语义对齐机制施加自上而下的语义约束，从而显著提升噪声环境下SAR自动目标识别的鲁棒性。 |
| [^464] | [Federated Mixture-of-Experts Alignment on Mobile Edge Networks under Data Heterogeneity](https://arxiv.org/abs/2603.21276) | 针对数据异构环境下联邦MoE大模型微调中客户端门控偏好分歧和专家语义模糊两大挑战，本文提出了FedAlign-MoE联邦聚合对齐框架，以在移动边缘网络上实现更好的协同训练。 |
| [^465] | [Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability](https://arxiv.org/abs/2603.16017) | 该论文提出“道德推理轨迹”这一新概念，揭示了大语言模型在道德推理中系统性地在多个伦理框架间切换、且框架切换频繁的轨迹更易受说服性攻击，并通过线性探测和激活引导技术定位并调控模型中的道德框架表征，为LLM道德推理的可解释性研究提供了新路径。 |
| [^466] | [PC-Diffuser: Path-Consistent Capsule CBF Safety Filtering for Diffusion-Based Trajectory Planner](https://arxiv.org/abs/2603.10330) | PC-Diffuser通过将可认证的路径一致性胶囊控制障碍函数（CBF）安全结构直接嵌入扩散规划器的去噪循环中，使安全性成为轨迹生成的内在属性而非事后修正，从而解决了扩散轨迹规划器难以认证且在罕见场景中可能灾难性失败的问题。 |
| [^467] | [Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks](https://arxiv.org/abs/2603.04364) | 该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。 |
| [^468] | [World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models](https://arxiv.org/abs/2603.04317) | 该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。 |
| [^469] | [Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches](https://arxiv.org/abs/2603.02655) | 提出两种无需微调的基于提示的停顿感知解码策略（固定间隔与根据话语估计时长动态调整间隔），使多模态大语言模型能够生成语义相关且时机恰当的实时游戏视频解说。 |
| [^470] | [Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound](https://arxiv.org/abs/2603.01295) | 该论文提出在解码阶段通过任务交互模块（TIM）与自适应交互加权（AIW）实现分割与分类分支间逐图像自适应的双向信息交换，从而在乳腺超声联合分割与分类任务上取得了优于现有方法的性能。 |
| [^471] | [SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation](https://arxiv.org/abs/2602.16863) | SimToolReal通过程序化生成多样化的工具状物体并以将其操作至随机目标位姿的通用目标训练单一从仿真到现实强化学习策略，实现了无需针对特定物体或任务进行工程设计的零样本灵巧工具操作。 |
| [^472] | [ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization](https://arxiv.org/abs/2602.15983) | ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。 |
| [^473] | [UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions](https://arxiv.org/abs/2602.14049) | 提出了UniST-Pred统一时空预测框架，通过解耦时间建模与空间表示学习并采用自适应表示级融合，在交通网络中断等结构与观测不确定性条件下实现鲁棒的交通预测。 |
| [^474] | [Learning to Configure Agentic AI Systems](https://arxiv.org/abs/2602.11574) | 该论文将LLM智能体系统的配置问题建模为半马尔可夫决策过程，提出轻量级分层策略ARC，能够根据查询难度动态选择最优配置，使推理准确率提升31.3%、工具使用准确率提升13.95%。 |
| [^475] | [Recommender system in X inadvertently profiles ideological positions of users](https://arxiv.org/abs/2602.02624) | 研究发现X平台的推荐系统在优化相关性时会无意中将用户的左右政治意识形态立场编码进其嵌入表示中，而移除该意识形态维度可在准确性损失有限的情况下实现推荐多样化。 |
| [^476] | [Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning](https://arxiv.org/abs/2601.20571) | 本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。 |
| [^477] | [Intersectional Fairness via Mixed-Integer Optimization](https://arxiv.org/abs/2601.19595) | 本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。 |
| [^478] | [Cross-Lingual Activation Steering for Multilingual Language Models](https://arxiv.org/abs/2601.16390) | 提出无需训练的推理时干预方法CLAS，通过有选择地调节神经元激活提升非主导语言的性能，且不损害高资源语言表现。 |
| [^479] | [Precomputing Multi-Agent Path Replanning Using Temporal Flexibility](https://arxiv.org/abs/2601.04884) | 提出FlexSIPP算法，通过预计算并利用其他智能体的时间灵活性，高效地对单个延迟智能体进行路径重规划，同时避免引发连锁延迟。 |
| [^480] | [MiniScope: Authorizing Agents with Least-Privilege Permissions](https://arxiv.org/abs/2512.11147) | 提出了一种以任务为中心的分层权限模型及MiniScope端到端权限系统，将智能体视为特定任务角色中的委托代理，自动发现权限层级并在运行时执行上下文最小权限原则，在保证安全性的同时将权限确认次数减少43.4%-89.4%。 |
| [^481] | [Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning](https://arxiv.org/abs/2511.21075) | 本文发现生物医学文本具有与通用文本截然不同的密集认知不确定性结构，并据此提出双尺度的平衡微调方法，通过词元重加权与序列级重新分配使模型聚焦知识密集样本，在多项医学与生物任务上取得比SFT和DFT更一致的性能提升。 |
| [^482] | [Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization](https://arxiv.org/abs/2511.20718) | 该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。 |
| [^483] | [FRAGMENTA: Efficient End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization in Small Data Regime](https://arxiv.org/abs/2511.20510) | FRAGMENTA提出了一种端到端框架，通过LVSEF片段生成器联合优化片段化与生成过程，并利用智能体系统将对话式专家反馈自动转化为生成目标，在小数据药物先导化合物优化任务中取得了优异表现。 |
| [^484] | [Universe of Thoughts: A Computational Framework for Creative Reasoning in Large Language Models](https://arxiv.org/abs/2511.20471) | 该论文受认知科学启发，首次将组合型、探索型和变革型创造力形式化为LLM上可执行的计算算子，并据此构建了“思维宇宙”这一创造性推理框架。 |
| [^485] | [Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models](https://arxiv.org/abs/2511.00053) | 该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。 |
| [^486] | [DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment](https://arxiv.org/abs/2510.24574) | 提出DistDF框架，通过最小化一种可证明上界于条件分布差异的联合分布Wasserstein差异来对齐预测与标签序列的分布，解决了传统均方误差在标签序列存在自相关时的估计偏差问题。 |
| [^487] | [Speak to a Protein: An Interactive Multimodal Co-Scientist](https://arxiv.org/abs/2510.17826) | 该论文提出了“与蛋白质对话”系统——一个交互式多模态AI协同科学家，它能够检索整合文献、结构与配体数据，在实时3D场景中高亮、注释和操作蛋白质可视化，并按需生成和运行代码，将原本需要数周的蛋白质分析转变为实时交互式对话。 |
| [^488] | [Predicting kernel regression learning curves from only raw data statistics](https://arxiv.org/abs/2510.14878) | 该论文提出 Hermite 特征结构假设（HEA），证明仅用经验协方差矩阵和目标函数的多项式分解这两个原始数据统计量，即可在真实图像数据集上准确预测核回归的学习曲线。 |
| [^489] | [Qubit-centric Transformer for Surface Code Decoding](https://arxiv.org/abs/2510.11593) | 提出了一种基于Transformer的新型量子纠错解码器QCT，利用以量子比特为中心的注意力机制和融合量子码拓扑结构的图掩码方法，将伴随式转换为量子比特标记以有效识别逻辑错误。 |
| [^490] | [Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators](https://arxiv.org/abs/2509.03647) | 本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。 |
| [^491] | [Practical Feasibility of Gradient Inversion Attacks in Federated Learning](https://arxiv.org/abs/2508.19819) | 本文系统评估了梯度反演攻击在现实联邦学习系统中的可行性，发现现代性能优化的视觉模型能够有效抵御有意义的图像重构，而已报告的攻击成功多依赖于理想化的上界实验设置。 |
| [^492] | [LHM-Humanoid: Long-Horizon Human Motion Control for Continuous Object Transport in Cluttered Scenes](https://arxiv.org/abs/2508.16943) | 该论文提出LHM-Humanoid，通过将动作循环间的交接建模为双边可恢复性问题，实现了物理模拟人形角色在杂乱场景中无需重置的连续长时程物体搬运（反复完成寻物、举起、绕障搬运与放置）。 |
| [^493] | [PuzzleJAX: A Benchmark for Reasoning and Learning](https://arxiv.org/abs/2508.16821) | PuzzleJAX是一个GPU加速的益智游戏引擎与描述语言，可动态编译PuzzleScript风格的游戏，从而为树搜索、强化学习和LLM推理能力提供大规模、多样化任务的快速基准测试。 |
| [^494] | [Cross-Modality Controlled Molecule Generation with Diffusion Language Model](https://arxiv.org/abs/2508.14748) | 提出模块化框架CMCM-DLM，通过结构控制模块和性质控制模块的分阶段协同设计，使预训练扩散语言模型无需重新训练即可支持分子结构与化学性质等跨模态异构约束的受控分子生成。 |
| [^495] | [FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models](https://arxiv.org/abs/2508.10020) | FedCoT是一个通信高效的联邦推理增强框架，通过轻量级思维链重采样、紧凑判别器筛选以及客户端感知的LoRA堆叠加权聚合，在无需集中式蒸馏和保护隐私的前提下增强大语言模型的逐步推理能力，特别适用于医疗等需要可解释、可审计决策的场景。 |
| [^496] | [Too Categorical to be Human: Emotion Concepts in LLMs and Humans](https://arxiv.org/abs/2508.05880) | 该论文提出基于认知评估理论、以“行为表征”刻画情绪概念，并构建涵盖15个情绪类别的基准数据集来比较LLMs与人类的情绪概念表征，发现LLMs的情绪概念过于范畴化而与人类不同。 |
| [^497] | [DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search](https://arxiv.org/abs/2507.07426) | 提出了DrugMCTS框架，协同整合RAG、多智能体协作和蒙特卡洛树搜索，通过五个专门智能体实现结构化迭代推理，在药物重定位任务上取得显著更高的召回率和鲁棒性。 |
| [^498] | [Time-o1: Time-Series Forecasting Needs Transformed Label Alignment](https://arxiv.org/abs/2505.17847) | Time-o1通过将标签序列变换为去相关且区分显著性的分量，训练模型对齐最显著分量，从而提出一种变换增强的损失函数，有效缓解标签自相关性并减少任务数量，在时间序列预测中达到最先进性能且兼容多种预测模型。 |
| [^499] | [KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches](https://arxiv.org/abs/2505.14777) | 提出了一种基于动理学与偏微分方程的即插即用优化器KO，它将参数动力学建模为粒子系统并通过离散化玻尔兹曼输运方程引入随机相互作用，从而提升参数多样性、缓解权重凝聚并保持收敛保证。 |
| [^500] | [Boosting Large Language Models with Mask Fine-Tuning](https://arxiv.org/abs/2503.22764) | 提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。 |
| [^501] | [Large Pretraining Datasets Don't Guarantee Robustness after Fine-Tuning in Image Classification](https://arxiv.org/abs/2410.21582) | 该论文提出鲁棒性继承基准ImageNet-RIB，揭示了一个关键问题：即使在大规模数据集上预训练的模型，经微调后仍会出现严重的灾难性遗忘和分布外泛化能力丧失，因此大规模预训练并不保证微调后的鲁棒性。 |
| [^502] | [When Explanations Compete: Policy-Aware Selection Under Uncertainty](https://arxiv.org/abs/2410.05479) | 该论文提出了一个策略感知的解释选择框架，通过结合资格规则、双向帕累托筛选和策略感知排序，从不确定性感知解释方法生成的多个候选解释中，依据预测置信度、不确定性和应用约束进行最优选择。 |
| [^503] | [Modeling Time-Dependent Responses of Optical Compressors with Selective State Space Models](https://arxiv.org/abs/2408.12549) | 该论文提出了一种结合选择性状态空间模型、特征级线性调制和门控线性单元的深度神经网络方法来建模光学压缩器的时变响应，性能超越基于循环层的方法，适用于低延迟实时音频处理，并在 TubeTech CL 1B 和 Teletronix LA-2A 两款模拟光学压缩器上得到验证。 |
| [^504] | [Diffusion Model-Based Video Editing: A Survey](https://arxiv.org/abs/2407.07111) | 本文是一篇综述，系统梳理了基于扩散模型的视频编辑技术的理论基础、方法分类与演化脉络，探讨了点编辑、姿态引导人体视频编辑等新兴应用，并提出新的V2VBench基准对该领域进行全面对比评估。 |
| [^505] | [FreDF: Learning to Forecast in Frequency Domain](https://arxiv.org/abs/2402.02399) | FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。 |
| [^506] | [Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation](https://arxiv.org/abs/2312.11018) | HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。 |
| [^507] | [Fast, Interpretable, and Deterministic Time Series Classification With a Bag-of-Receptive-Fields](https://arxiv.org/abs/2311.18029) | 本文提出了BORF（感受野袋），一种快速、可解释且确定性的时间序列分类变换方法，克服了现有黑盒分类器难以理解以及可解释方法因依赖随机化导致解释不稳定的问题。 |

# 详细

[^1]: 4D-HOF：面向前馈式4D交互重建的手-物体流匹配

    4D-HOF: Hand-Object Flow Matching for Feed-Forward 4D Interaction Reconstruction

    [https://arxiv.org/abs/2610.08782](https://arxiv.org/abs/2610.08782)

    提出4D-HOF，一个前馈式框架，利用条件流匹配模型将视觉基础模型生成的粗略手-物体状态传输至交互流形，实现4D手-物体交互重建，并可在传输过程中直接施加物理交互约束进行测试时引导。

    

    现有的4D手-物体重建方法通常依赖于代价高昂的逐序列优化，而生成式方法通常从随机噪声合成交互，这可能导致不稳定的交互预测。我们提出了4D-HOF，这是一个前馈框架，能够从视觉基础模型生成的粗糙但信息丰富的估计中重建4D手-物体交互。具体而言，我们学习了一个条件流匹配模型，将基础模型导出的手-物体状态传输至交互流形，使模型能够以前馈方式纠正平移、旋转和对齐中的误差。我们生成式公式的一个关键优势在于，它天然支持在传输过程中进行测试时引导。我们无需在重建后进行单独的事后优化，而是直接利用物理交互约束和观测……

    arXiv:2610.08782v1 Announce Type: cross  Abstract: Existing methods for 4D hand-object reconstruction often rely on costly per-sequence optimization, while generative approaches typically synthesize interactions from random noise, which can lead to unstable interaction prediction. We introduce 4D-HOF, a feed-forward framework that reconstructs 4D hand-object interactions from coarse but informative estimates produced by vision foundation models. Concretely, we learn a conditional flow matching model that transports foundation-model-derived hand-object states toward an interaction manifold, allowing the model to correct errors in translation, rotation, and alignment in a feed-forward manner. A key advantage of our generative formulation is that it naturally enables test-time guidance within the transport process. Rather than applying a separate post-hoc optimization after reconstruction, we directly steer the evolving generative states using physical interaction constraints and observed
    
[^2]: IdeaAnchor：教会大语言模型将文献转化为研究思路

    IdeaAnchor: Teaching LLMs to Turn Literature into Research Ideas

    [https://arxiv.org/abs/2610.08781](https://arxiv.org/abs/2610.08781)

    提出IdeaAnchor范式，通过编码论文功能角色、关系及综合标准的结构化规格说明作为特权信号，结合示范、自蒸馏与强化学习训练大语言模型，使其学会将文献综合为研究思路。

    

    科学研究往往始于从一组相关论文中综合提炼想法，以发现研究空白并提出新的研究方向。然而，训练语言模型完成这种基于文献的构思（ideation）仍然具有挑战性，因为现有的基于提示或反馈的方法缺乏关于论文应如何被综合的结构化监督信号。我们提出IdeaAnchor，一种利用结构化规格说明作为特权信号来训练大语言模型进行研究构思的范式。每个IdeaAnchor实例编码了每篇输入论文应如何被综合成一个成功的研究想法，包括它们的功能角色、相互关系以及目标综合标准。我们通过从已发表论文中挖掘实例来构建该范式，捕捉真实研究想法如何从既有文献中产生。随后，我们通过示范学习、自蒸馏和强化学习训练模型，并在推理时结合检索进一步增强生成效果。

    arXiv:2610.08781v1 Announce Type: cross  Abstract: Scientific research often begins by synthesizing ideas from a set of related papers to identify gaps and formulate new directions. However, training language models to perform this form of literature-grounded ideation remains challenging, as existing approaches based on prompting or feedback lack structured supervision for how papers should be synthesized. We introduce IdeaAnchor, a paradigm for training LLMs to perform research ideation using structured specifications as privileged signals. Each IdeaAnchor instance encodes how each input paper should be synthesized into a successful idea, including their functional roles, relationships, and target synthesis criteria. We build this paradigm by mining instances from published papers, capturing how real ideas emerge from prior literature. We then train models via demonstration, self-distillation, and reinforcement learning, and further enhance generation with retrieval at inference time.
    
[^3]: DepthWorld：面向机器人操作的3D世界模型

    DepthWorld: 3D World Model for Robot Manipulation

    [https://arxiv.org/abs/2610.08780](https://arxiv.org/abs/2610.08780)

    该论文提出了一种结合学习式双目深度与联合因子图的标定流程，据此构建了DROID-3D标定3D数据集，为基于视频的世界模型提供稠密度量深度和精确外参，使其能够生成具有3D几何一致性的机器人操作推演。

    

    世界模型为机器人技术提供了传统仿真器的数据驱动替代方案，其应用涵盖策略评估、策略改进与规划。所有这些应用都依赖于真实准确的3D几何，然而当前基于视频的世界模型仅使用RGB进行训练，其生成的视频推演虽然逐帧看起来正确，却无法组合成一个一致的3D世界。弥合这一差距需要在两个方面取得进展：面向操作任务的大规模3D监督数据，以及能够在不破坏强大预训练视频先验的前提下吸收这些监督的架构。我们提出了一种标定流程，将学习得到的双目立体深度与联合因子图相结合，汇集从同一物理机器人采集的所有回合数据，以恢复其共享的运动学参数以及每个场景的外参。将该流程应用于DROID数据集，我们得到了DROID-3D，这是一个经过标定的3D数据集，提供稠密的度量深度和重新标定的多视角外参（达到 <0.7

    arXiv:2610.08780v1 Announce Type: cross  Abstract: World models offer a data-driven alternative to traditional simulators for robotics, with applications spanning policy evaluation, improvement, and planning. All of these uses depend on faithful 3D geometry, yet current video-based world models are trained on RGB alone and produce rollouts that look correct frame-by-frame but do not compose into a consistent 3D world. Closing this gap requires progress on two fronts: large-scale 3D supervision for manipulation, and an architecture that can absorb it without disturbing strong pretrained video priors. We introduce a calibration pipeline that combines learned stereo depth with a joint factor graph, pooling all episodes collected from the same physical robot to recover its shared kinematic parameters alongside per-scene extrinsics. Applied to the DROID dataset, this yields DROID-3D, a calibrated 3D dataset providing dense metric depth and recalibrated multi-view extrinsics (achieving <0.7 
    
[^4]: Sherpa：教会大语言模型自适应地教学

    Sherpa: Teaching LLMs to Teach Adaptively

    [https://arxiv.org/abs/2610.08778](https://arxiv.org/abs/2610.08778)

    Sherpa是一个多轮强化学习框架，通过模拟具有不同学习偏好的学生原型并直接最大化其学习成效，训练教师LLM自适应地调整教学策略，使受教学生的成绩平均提升20.5个百分点。

    

    大语言模型（LLM）作为问题求解者的能力日益增强，但能够解决一个问题并不等同于能够教授这个问题。现有的将LLM训练为教师的方法依赖于演示、偏好数据或预先定义的教学标准来规定什么是好的教学。然而，这些信号通常并未基于学生个体的学习成效，而有效的教学策略在不同学习者之间可能存在显著差异。为了解决这一问题，我们提出了Sherpa，这是一个多轮强化学习框架，它利用基于不同学习偏好条件化的LLM实例化多种学生原型，并通过直接最大化这些学生的学习成效来训练教师模型自适应地调整其教学方式。使用Sherpa训练的教师LLM使所有学生原型下的受教学生成绩平均提高了20.5个百分点。在MathTutorBench的评估下，Sherpa

    arXiv:2610.08778v1 Announce Type: new  Abstract: Large language models (LLMs) have become increasingly capable problem solvers, but being able to solve a problem is not the same as being able to teach it. Existing approaches to training LLMs as teachers rely on demonstrations, preference data, or predefined pedagogical criteria that specify what good teaching looks like. However, these signals are often not grounded in individual student learning outcomes, where effective teaching strategies can vary substantially across learners. To address this, we introduce Sherpa, a multi-turn reinforcement learning framework that instantiates multiple student archetypes with LLMs conditioned on distinct learning preferences and trains a teacher model to adapt its instruction by directly maximizing their learning outcomes. Teacher LLMs trained with Sherpa improve instructed students' performance across all archetypes by an average of 20.5 percentage points. Under MathTutorBench's evaluation, Sherpa
    
[^5]: 瓶中智能体：大语言模型智能体能将自身能力转化为廉价、可扩展的产物吗？

    Agent in a Bottle: Can LLM Agents Turn Their Capabilities Into Cheap, Scalable Artifacts?

    [https://arxiv.org/abs/2610.08775](https://arxiv.org/abs/2610.08775)

    该论文提出了“装瓶”这一新概念和BOTTLED基准，用于评估LLM智能体能否在固定时间、计算和API预算内自主将自身通用能力转化为廉价、可复用的任务级解决方案，并发现强大的零样本表现并不能可靠地转化为强大的装瓶能力。

    

    大语言模型（LLM）能够解决许多狭窄的任务，但针对数百万个相关实例逐一进行查询，其成本可能高得令人望而却步。大语言模型智能体能否自主地为这类工作负载创造出更廉价的解决方案？我们将这种能力称为“装瓶”：即把通用能力转化为在答案质量与摊销成本之间取得平衡的特定任务解决方案的能力。我们提出了BOTTLED基准，在该基准中，智能体会收到整个未标注的工作负载，并必须在固定的时间、计算资源和LLM API预算内完成它。智能体可以自行选择实现方法，例如训练一个小模型或编写一个可复用的程序。在十个模型和三个任务上的实验中，我们发现强大的零样本任务表现并不能可靠地转化为强大的装瓶能力：零样本得分相近的模型在装瓶之后可能出现显著差异，且60次装瓶运行中有48次的得分低于其众数95%置信区间的下限。

    arXiv:2610.08775v1 Announce Type: new  Abstract: Large language models (LLMs) can solve many narrow tasks, but querying them separately for millions of related instances can be prohibitively expensive. Can LLM agents autonomously create cheaper solutions for such workloads? We call this ability "bottling": the ability to turn general capabilities into task-specific solutions that balance answer quality and amortised cost. We introduce BOTTLED, a benchmark in which agents receive an entire unlabelled workload and must complete it under fixed time, compute and LLM API budgets. Agents choose their own approach, such as training a small model or writing a reusable program. Across ten models and three tasks, we find that strong zero-shot task performance does not reliably translate into strong bottling capabilities. Models with similar zero-shot scores can differ substantially after bottling, and 48 of 60 bottling runs score below the lower bound of the 95% confidence interval of their mode
    
[^6]: AdvSim2Real：在网络世界模型中训练网络智能体以对抗自适应提示注入

    AdvSim2Real : Training Web Agents Against Adaptive Prompt Injection in a Web World Model

    [https://arxiv.org/abs/2610.08773](https://arxiv.org/abs/2610.08773)

    提出 AdvSim2Real，在冻结的网络世界模型中让任务课程、注入攻击者与智能体共同演化，通过“成功翻转”对抗奖励机制训练出既更强又更能抵御自适应提示注入攻击的网络智能体。

    

    arXiv:2610.08773v1 公告类型：cross 摘要：网络智能体通过阅读并操作由第三方编写的网页来完成用户请求，因此页面上被植入的指令可能会使智能体偏离用户的目标。然而智能体不能简单地忽略页面，因为页面中还包含任务所需的取值和控件。当前的防御方法是在训练前固定注入内容并对智能体进行微调，但适应了已训练模型的自适应攻击者可以绕过这些防御。对抗性训练虽然允许攻击者进行适应，但任务保持固定，因此一旦智能体解决了某个任务，该任务就不再具有训练价值。我们提出 AdvSim2Real，它在一个冻结的网络世界模型中共同演化任务课程、注入攻击者和智能体。任务课程因智能体约有半数概率能解决的任务而获得奖励，而攻击者仅因“成功翻转”——即能把被判定为成功的任务转变为失败的注入——而获得奖励。在模拟器中训练使一个 4B 的智能体在能力和鲁棒性方面都得到提升：其完成……

    arXiv:2610.08773v1 Announce Type: cross  Abstract: Web agents complete user requests by reading and acting on pages that third parties write, so an instruction planted on a page can redirect the agent away from the user's goal. The agent cannot simply ignore the page, because the page also holds the values and controls the task requires. Current defenses fine-tune the agent on injections fixed before training, and attackers that adapt to the trained model bypass them. Adversarial training lets the attacker adapt but keeps the tasks fixed, so a task stops teaching once the agent solves it. We introduce AdvSim2Real, which co-evolves a task curriculum, an injection adversary, and the agent inside a frozen web world model. The curriculum is rewarded for tasks the agent solves about half of the time, and the adversary only for a success flip, an injection that turns a judged success into a failure. Training in the simulator makes a 4B agent both more capable and more robust: its completion 
    
[^7]: VeriFine：面向具身推理自我改进的规模化验证方法

    VeriFine: Scaling Verification for Self-Improvement in Embodied Reasoning

    [https://arxiv.org/abs/2610.08761](https://arxiv.org/abs/2610.08761)

    VeriFine 通过策略、训练课程与评判者的共同演化（包括策略改进循环，以及在验证成为瓶颈时借助人类选择性指导与协同校准来精炼评判者的评判者改进循环），实现了具身推理中自我改进的规模化验证。

    

    自我改进的策略会不断暴露出新的失败模式，这改变了其评判者所必须能够验证的内容。然而，当前固定的评判者既限制了优化反馈，也限制了有用训练样本的发现，从而进一步限制了自我改进。这一挑战在具身推理中尤为突出，因为可靠的评估必须考虑空间定位、因果推理以及具备安全意识的决策。我们提出了 VeriFine，这是一个智能体框架，通过策略、训练课程与评判者的共同演化来扩展验证能力。策略改进循环使用基于评分标准的评判者来诊断反复出现的失败、构建自适应课程并优化策略。当进展趋于停滞且验证成为瓶颈时，评判者改进循环会在信息量大的失败案例上选择性地向人类寻求指导，并通过协同校准来精炼评判者，在这种校准过程中人类与智能体……（原文在此处截断）

    arXiv:2610.08761v1 Announce Type: new  Abstract: Self-improving policies continually expose new failure patterns, changing what their judges must be able to verify. However, current fixed judges constrain both optimization feedback and the discovery of useful training examples, limiting further self-improvement. This challenge is even more acute in embodied reasoning, where reliable evaluation must account for spatial grounding, causal reasoning, and safety-aware decision-making. We introduce VeriFine, an agent harness framework that scales verification through the co-evolution of the policy, training curriculum, and judge. The Policy Improvement Loop uses a rubric judge to diagnose recurring failures, construct an adaptive curriculum, and optimize the policy. When progress plateaus and verification becomes a bottleneck, the Judge Improvement Loop selectively queries human guidance on informative failure cases and refines the judge through coactive calibration, in which humans and agen
    
[^8]: WorldSonus：为世界注入声音

    WorldSonus: Bringing Sound to Worlds

    [https://arxiv.org/abs/2610.08760](https://arxiv.org/abs/2610.08760)

    WorldSonus 是一个面向世界模型的交互式视频到音频框架，通过流式因果自回归扩散架构以 0.41 的实时率实现空间立体声的实时生成，并支持生成过程中的声音指令交互控制。

    

    世界模型的最新进展使得视觉合成日益逼真，然而这些生成的环境在很大程度上仍然是无声的。为世界模型引入声音面临三个核心挑战：实时生成以跟上交互式视频流的节奏、交互式控制以响应生成过程中的声音指令，以及空间对齐的立体声以反映场景几何和相机运动。为满足这些需求，我们提出了 WorldSonus，一个专为世界模型中实时空间声音合成而设计的交互式视频到音频框架。在实时生成方面，WorldSonus 采用流式因果自回归扩散架构，以低至 0.41 的实时率（RTF）合成音频块。在交互式控制方面，我们引入了以音频为中心的字幕生成流水线和基于块索引的提示调度机制，实现了生成过程中声音事件的动态操控。在空间对齐方面，我们……（摘要原文在此处被截断）

    arXiv:2610.08760v1 Announce Type: cross  Abstract: Recent advances in world models have enabled increasingly realistic visual synthesis. However, these generated environments remain largely silent. Bringing sound to world models poses three core challenges: real-time generation to keep pace with interactive video streams, interactive control to respond to mid-stream sound instructions, and spatially aligned stereo to reflect scene geometry and camera motion. To address these demands, we introduce WorldSonus, an interactive video-to-audio framework designed for real-time spatial sound synthesis in world models. For real-time generation, WorldSonus employs a streaming causal autoregressive diffusion architecture that synthesizes audio chunks at a low real-time factor (RTF) of 0.41. For interactive control, we incorporate an audio-centric captioning pipeline with chunk-indexed prompt scheduling, enabling dynamic manipulation of sound events during generation. For spatial alignment, we lev
    
[^9]: 具有保形动作集的强化学习：在序列推荐中的应用

    Reinforcement Learning with Conformal Action Sets: An Application to Sequential Recommendation

    [https://arxiv.org/abs/2610.08743](https://arxiv.org/abs/2610.08743)

    该论文提出了RLCP方法，通过评论家分数和在线阈值自适应调整序列推荐中的动作集大小，并从理论上证明了代理未命中率上界以及将价值损失精确分解为过滤损失和选择损失，从而获得无需参数收敛的有限会话奖励上界。

    

    序列推荐系统通常使用固定的推荐列表大小，尽管一个会话中有用的备选方案数量是动态变化的。我们提出了带校准剪枝的强化学习方法（RLCP），该方法利用评论家分数和在线阈值来自适应调整保留的动作集。该阈值根据二值反馈进行更新，该反馈用于指示集合中是否包含代理目标中的某个动作。我们证明了沿自适应轨迹观测到的代理未命中率存在确定性上界。为了量化剪枝对奖励的影响，我们将价值损失精确分解为过滤损失和选择损失。在显式的代理和评论家近似条件下，该分解给出了一个有限的会话奖励上界，该上界同时考虑了不完美的选择和集合截断，而无需学习参数收敛。我们在KuaiRand-Pure和MovieLens 1M数据集上的实验将两种RLCP实现与四个强化学习基线进行了比较。

    arXiv:2610.08743v1 Announce Type: cross  Abstract: Sequential recommenders typically use a fixed slate size even though the number of useful alternatives changes within a session. We propose Reinforcement Learning with Calibrated Pruning (RLCP), which adapts the retained action set using critic scores and an online threshold. The threshold is updated from binary feedback indicating whether the set contains an action in a proxy target. We prove a deterministic bound on the observed proxy miss rate along adaptive trajectories. To quantify the effect of pruning on reward, we derive an exact decomposition of value loss into filtering and selection losses. Under explicit proxy and critic approximation conditions, this decomposition yields a finite session reward bound that also accounts for imperfect selection and set truncation, without requiring the learning parameters to converge. Experiments on KuaiRand-Pure and MovieLens 1M compare two RLCP implementations with four RL baselines. In ea
    
[^10]: EgoLAP：通过语言-动作推理从第一人称人类数据中学习

    EgoLAP: Learning from Egocentric Human Data through Language-Action Reasoning

    [https://arxiv.org/abs/2610.08726](https://arxiv.org/abs/2610.08726)

    EgoLAP提出了一种VLA预训练框架，通过共享的基于语言的动作思维链，将第一人称人类数据中的运动意图转化为结构化语言动作并进行运动级推理，从而弥合具身差异，使人类经验有效迁移到机器人控制，真实世界任务进度达到80.1%，性能提升2.3倍。

    

    第一人称视角的人类数据为扩展机器人学习规模提供了超越昂贵机器人演示的途径，然而具身差异（embodiment gap）使得原始人类轨迹难以作为控制学习的有效监督目标。我们的核心洞察是：尽管低层动作依赖于具体的身体形态，但其背后的运动意图能够捕捉到可在人类与机器人之间迁移的任务相关结构。我们提出了EgoLAP，这是一个VLA（视觉-语言-动作）预训练框架，通过共享的基于语言的动作思维链，从人类和机器人轨迹中联合学习。EgoLAP将运动意图表达为结构化的、时间上抽象化的语言动作，并将其与基于场景几何、物理特性和物体可供性的运动级推理相结合。在大量真实世界和仿真实验中，EgoLAP比其他动作表示方法更有效地将人类经验迁移到机器人控制中，在真实世界任务中达到80.1%的平均任务进度，性能提升2.3倍……

    arXiv:2610.08726v1 Announce Type: cross  Abstract: Egocentric human data offer a path to scaling robot learning beyond costly robot demonstrations, yet the embodiment gap makes raw human trajectories a poor supervisory target for control. Our key insight is that, although low-level actions are embodiment-specific, their underlying motion intent can capture task-relevant structure that transfers across humans and robots. We introduce EgoLAP, a VLA pre-training framework that jointly learns from human and robot trajectories through a shared language-based action chain-of-thought. EgoLAP expresses motion intent as structured, temporally abstracted language actions and pairs them with motion-level reasoning grounded in scene geometry, physics, and object affordances. Across extensive real-world and simulated experiments, EgoLAP transfers human experience to robot control more effectively than alternative action representations and reaches 80.1% mean real-world task progress, a 2.3x perform
    
[^11]: 智能体的历史能否告诉你上下文压缩何时会造成伤害？基于TRACE成对重放语料库的适度且有界的影响分析

    Does an Agent's History Tell You When Compaction Will Hurt? A Modest, Bounded Effect on the TRACE Paired-Replay Corpus

    [https://arxiv.org/abs/2610.08722](https://arxiv.org/abs/2610.08722)

    本研究利用TRACE语料库中590个成对重放的压缩边界，检验智能体近期历史能否预测上下文压缩的危害，结果发现预测能力仅微弱有效——按前缀位置的预设对比为零结果，最佳可解释触发器也仅能避免21%的有害压缩边界。

    

    许多长时程智能体按照全局规则压缩其上下文，通常是基于令牌预算，而不考虑智能体当前正在执行的任务。我们提出的问题是：智能体最近的行为能否预测压缩何时会造成伤害？TRACE公开语料库包含590个由框架触发的AppWorld压缩边界，它从重新执行的前缀状态出发，分别在压缩前上下文和摘要两种条件下重放每个边界，并记录后续动作的负担，即出错的调用或重复已执行调用的调用。我们发现，边界前的历史仅能微弱地预测压缩后的伤害。一个内部预先指定的、按前缀位置进行对比的实验结果是一个较宽的零结果，而其背后朴素的“已写入”标签实际上衡量的是轨迹所处的阶段。最佳的扩展协议触发器在留出集上达到AUROC 0.66（在复现集自身标签上为0.64），而同边界复现的AUROC为0.72；最佳的冻结式、可解释触发器能够避免21%的有害（正负担）压缩边界。

    arXiv:2610.08722v1 Announce Type: new  Abstract: Many long-horizon agents compact their context on a global rule, usually a token budget, blind to what the agent was doing. We ask whether the agent's recent behaviour predicts when a compaction will hurt. TRACE's public corpus of 590 harness-triggered AppWorld compaction boundaries replays each boundary from a re-executed prefix state under the pre-compaction context and under the summary, and records the burden of the next actions: calls that error or repeat a call already made. We find that pre-boundary history predicts post-compaction harm only weakly. An internally prespecified contrast by prefix placement is a wide null, and the naive "has-written" label behind it turns out to measure trajectory phase. The best extension-protocol trigger reaches held-out AUROC 0.66 (0.64 on the replicate's own label) against a same-boundary replicate of 0.72; the best frozen, interpretable trigger avoids 21% of harmful (positive-burden) boundaries 
    
[^12]: WorldSolver：LLM智能体能通过求解器生成来模拟物理动力学吗？

    WorldSolver: Can LLM Agents Simulate the Physical Dynamics via Solver Generation?

    [https://arxiv.org/abs/2610.08720](https://arxiv.org/abs/2610.08720)

    提出WorldSolver基准，通过源自61篇经典计算机图形学论文、覆盖7个物理领域的168个模拟任务，评估LLM智能体生成物理求解器代码以模拟物理动力学的能力。

    

    基于大语言模型（LLM）的智能体在科学和工程问题求解方面不断取得进展，而物理模拟正成为一个具有挑战性但又实用的测试平台，用于复现复杂的物理现象，并可应用于具身智能、游戏和电影等领域。作为此类模拟的核心工具，求解器（solver）负责计算动态系统的状态如何随时间演化。构建这样的求解器需要物理理解能力来确定合适的模型，需要数学推理能力来表述底层的动力学规律，还需要软件工程能力将其实现为可执行代码，然而LLM智能体的这种能力仍未被充分探索。为此，我们提出了WorldSolver，这是一个包含168个模拟任务的基准，这些任务源自61篇经典计算机图形学论文中的物理现象，涵盖7个物理领域。每个任务包含一个代码脚手架，为场景提供固定的模拟环境，而求解器的实现则留给智能体来完成。

    arXiv:2610.08720v1 Announce Type: new  Abstract: LLM-based agents are increasingly advancing scientific and engineering problem solving, with physics simulation emerging as a challenging yet practical testbed for reproducing complex physical phenomena with application in embodied AI, games and films. As the workhorse of such simulation, a solver computes how the state of a dynamic system evolves over time. Building such solvers requires physical understanding to identify appropriate models, mathematical reasoning to formulate the underlying dynamics, and software engineering to implement them as executable code, yet this capability of LLM agents remains underexplored. To this end, we introduce WorldSolver, a benchmark of 168 simulation tasks derived from physical phenomena in 61 classic computer graphics papers, spanning 7 physical domains. Each task contains a code scaffold that provides a fixed simulation environment for the scene, with the solver implementation left for the agent to
    
[^13]: nanoMuse：为你拥有的每台设备提供的开源个人智能体

    nanoMuse: An Open-Source Personal Agent for Every Device You Own

    [https://arxiv.org/abs/2610.08699](https://arxiv.org/abs/2610.08699)

    nanoMuse是Meta闭源个人智能体Muse的开源对应物：以GPL-3.0许可发布，在用户拥有的每台设备上各运行一个智能体，通过可自建的中继共享同一段持久对话，并能操作手机与电脑屏幕、跨周记忆、主动发起对话。

    

    2011年的助手回答完问题便静候指令，2023年的智能体完成一项任务后便停止。2026年9月，Meta的Muse展示了一种面向单一个人的智能体，它拥有账户、设备、记忆和一段持续不断的对话，但它封闭在一家厂商的云端中，局限于一个国家。这样的智能体被期望能够操作个人的账户和设备，在数周的时间里记住这些内容，在值得开口时主动发起对话，并对自己所做的事情负责。这是一种软件，而非模型，但迄今为止一直没有开源的对应物。本报告通过五个问题和三个时间视界来定义个人智能体。报告基于Meta的公开记录及其生产环境提示词（prompt）的副本，解读了Muse是如何构建的，每一条陈述都标注了信息来源。随后，报告提出了nanoMuse——GPL-3.0许可下的开源对应物，即在一个人拥有的每台设备上各运行一个智能体，其“双手”可以操作手机的屏幕和电脑。这些智能体通过任何人都可以自行搭建的中继共享同一段对话；每个操作都需经过……

    arXiv:2610.08699v1 Announce Type: new  Abstract: Assistants from 2011 answered and waited, and agents from 2023 did a task and stopped. In September 2026 Meta's Muse showed an agent for one person, with accounts, devices, memory and a conversation that lasts, closed, in a vendor's cloud, in one country. Such an agent is expected to act on a person's accounts and devices, remember them across weeks, speak first when it is worth it, and answer for what it did. It is a kind of software, not a model, and until now had no open counterpart. This report defines the personal agent in five questions and three horizons. It reads how Muse is built from Meta's public record and a copy of its production prompt, each statement marked by its source. It then presents nanoMuse, the open-source counterpart under the GPL-3.0, one agent on every device a person owns, with hands on the phone's screen and the computer's. They share one conversation over a relay anyone can run; every action goes through a Se
    
[^14]: ScienceClaw：跨自然科学与社会科学的AI科研智能体持续自我演化基准测试

    ScienceClaw: Benchmarking Continual Self-Evolution of AI-for-Science Agents Across the Natural and Social Sciences

    [https://arxiv.org/abs/2610.08691](https://arxiv.org/abs/2610.08691)

    该论文提出了ScienceClaw，一个将任务求解、科学验证与程序更新统一起来的固定参数程序自我演化框架，并配套覆盖23个学科的ScienceClaw-Eval基准，用于衡量AI科研智能体在序列任务中的演化收益、知识保留、跨数据集迁移与演化成本。

    

    大语言模型智能体正在加速科学自动化，然而经过验证的执行结果很少转化为持久的程序级改进，且现有评估方法未能在自然科学与社会科学的序列任务中检验这一过程。我们将ScienceClaw形式化为固定参数的程序自我演化，统一了任务求解、科学验证与程序更新。ScienceClaw-Eval涵盖23个学科，通过序列数据流和独立重置评估来衡量科学正确性、演化收益、保留能力、跨数据集迁移以及演化成本。我们的框架通过多轮交互修复可执行工作流，将经重新执行验证的失败-成功轨迹转化为相互关联的技能与算子候选，并且仅当源任务回放能够复现修复效果且独立科学任务有所改进时才保留更新。代码可在 https://github.com/beita 获取。

    arXiv:2610.08691v1 Announce Type: new  Abstract: Large language model agents are accelerating scientific automation, yet verified executions rarely become persistent program-level improvements, and existing evaluations do not examine this process across sequential tasks in both the natural and social sciences. We formalize ScienceClaw as fixed-parameter program self-evolution that unifies task solving, scientific verification, and program updates. ScienceClaw-Eval spans 23 disciplines and measures scientific correctness, evolutionary gain, retention, cross-dataset transfer, and evolution cost through sequential streams and independent reset evaluation. Our framework repairs executable workflows through multi-turn interaction, converts re-execution-verified failure--success trajectories into linked Skill and Operator candidates, and retains an update only when source-task replay reproduces the repair and independent scientific tasks improve. Code is available at https://github.com/beita
    
[^15]: 耦合却滞后：全双工语音模型在非脚本对话中的轮次交替

    Coupled but Late: Turn-Taking Between Full-Duplex Speech Models in Unscripted Dialogue

    [https://arxiv.org/abs/2610.08683](https://arxiv.org/abs/2610.08683)

    两个全双工语音模型互相对话时虽能形成相互耦合的时序模式，但话轮交接远比人类滞后（中位数400-560毫秒 vs 人类的137毫秒），且缺乏人类那种预测性的提前转换能力，表现为对对方话轮结束的被动反应式等待。

    

    全双工语音模型原本被训练用于与人类对话，但如今它们越来越多地被用于彼此对话，例如在自博弈数据生成、智能体社会以及基于模型的评估中。在这种闭环中，没有人类来吸收时序误差：每个模型的轮次交替行为本身就是另一个模型的输入。我们探究这种闭环最终会稳定在什么样的时序状态。两个 PersonaPlex-7B 实例在共享时钟上以非脚本对话的方式交换音频令牌，并将同一种话轮转换规则应用于它们和 Switchboard 人类对话语料库。结果显示，模型的时序是相互耦合的：跨对话重新配对说话者会破坏这种耦合。但话轮的交接发生得很晚，中位数为 400-560 毫秒，而人类仅为 137 毫秒；在对方话轮的最后 120 毫秒内——人类凭借预测（projection）会在此完成约十分之一的话轮转换——模型仅完成了 1%。延迟信道的一个方向会使响应一一对应地后移，且响应之前的时间段保持空白，这与模型在感知到对方话语结束后的反应性等待相一致，而非（人类那样的预测性提前交接）。

    arXiv:2610.08683v1 Announce Type: new  Abstract: Full-duplex speech models are trained to converse with a person, but they are increasingly made to converse with each other, in self-play data generation, agent societies, and model-based evaluation. In that loop no human absorbs a timing error: each model's turn-taking is the other's input. We ask what timing the loop settles into. Two PersonaPlex-7B instances exchange audio tokens on a shared clock in unscripted conversation, and one floor-transfer rule is applied to them and to Switchboard. Their timing is coupled: re-pairing speakers across conversations destroys it. But the floor changes hands late, at a median of 400-560 ms against 137 ms for humans, and the last 120 ms of the partner's turn, where human projection places a tenth of its transfers, holds 1% of theirs. Delaying one direction of the channel shifts the response one-for-one and leaves the run-up to it empty, consistent with a reactive wait after the perceived end rather
    
[^16]: 大语言模型的安全投机解码

    Secure Speculative Decoding for Large Language Models

    [https://arxiv.org/abs/2610.08678](https://arxiv.org/abs/2610.08678)

    本文首次系统研究了投机解码的安全影响，揭示了一种“安全-效用不对称”现象：推理效率的提升以不成比例的高安全代价为代价，越狱和提示注入攻击的成功率上升速度远快于效用的下降。

    

    投机解码（Speculative Decoding）通过首先使用一个较小的模型（称为“草稿模型”）生成候选词元，然后由大语言模型（即“目标模型”）对这些候选词元进行接受或拒绝的验证，从而加速大语言模型（LLM）的推理。先前的研究主要聚焦于投机解码的效率与效用之间的权衡，例如有损投机解码，而对其安全影响的研究在很大程度上仍处于空白状态。在这项工作中，我们通过首次对投机解码的安全影响进行系统性研究来填补这一空白。通过大规模的测量研究，我们揭示了一种显著的安全-效用不对称现象：在广泛的有损投机解码方法中，推理效率的提升以不成比例的高昂安全代价为代价，越狱攻击和提示注入攻击的成功率上升速度远快于效用的下降……

    arXiv:2610.08678v1 Announce Type: cross  Abstract: Speculative decoding accelerates inference for a large language model (LLM), referred to as the \emph{target model}, by first using a smaller model, referred to as the \emph{draft model}, to generate candidate tokens and then verifying them with the target model for acceptance or rejection. Prior studies primarily focused on the efficiency-utility trade-off of speculative decoding, e.g., lossy speculative decoding, leaving its security implications largely unexplored.   In this work, we bridge this gap by providing the \emph{first} systematic study of the security implications of speculative decoding. Through a large-scale measurement study, we reveal a pronounced security-utility asymmetry: across a wide range of lossy speculative decoding methods, improvements in inference efficiency come at a disproportionately high cost to security, with attack success rates for jailbreak and prompt injection attacks increasing much faster than uti
    
[^17]: 压力下的原则坚守：后训练决定大语言模型是否会践行自己的道德判断

    Principled Under Pressure: Post-Training Decides Whether LLMs Act on Their Own Moral Judgment

    [https://arxiv.org/abs/2610.08670](https://arxiv.org/abs/2610.08670)

    该研究构建了涵盖五种压力类型的248个预注册场景面板，通过让模型同时以第一人称行动和第三人称评判的方式对照其自身道德判断，发现大语言模型在约五分之一的压力场景下会采取自己判定为错误的行为，且这种“言行不一”差距的大小取决于后训练配方。

    

    语言模型越来越多地充当智能体。一个明知某行为错误却仍然去做的智能体，与一个不知道更好选择的智能体，是两种不同的失败模式，而对模型陈述价值观的评估无法发现前者。我们构建了一个预注册的、涵盖五种压力类型的248个场景面板。每个场景以两种方式向同一模型提出：一次作为智能体选择要采取的行动，一次以第三人称询问哪个选项是正确的，从而以模型自身的判断作为参照。每个场景都有一个移除压力因素的孪生版本，并且每个模型都有一个正向对照——即其运营者下令执行违规行为的场景——以便区分“差距缺失”与“测量工具失灵”。在OLMo-3-7B-Instruct上，该模型在大约五分之一的压力场景中采取了它自己判定为错误的行为，且这一比例高于移除压力后的相同场景。在四个指令模型中，这一差距的大小取决于后训练配方。

    arXiv:2610.08670v1 Announce Type: cross  Abstract: Language models increasingly act as agents. An agent that says an action is wrong and then takes it anyway is a different failure from one that does not know better, and evaluations of stated values cannot see it. We build a pre-registered panel of 248 scenarios across five kinds of pressure. Each scenario is posed twice to the same model, once as the agent choosing what to do and once in the third person asking which option is right, so the model's own judgment is the reference. Every scenario has a twin with the pressure removed, and every model gets a positive control in which its operator orders the violating action, so that a missing gap can be told apart from a blind instrument. On OLMo-3-7B-Instruct, the model takes the action it judged wrong on about one in five pressuring scenarios, more often than on the same scenarios with the pressure removed. Across four instruct models the gap depends on the post-training recipe: OLMo-3 a
    
[^18]: MemFLoRA：面向边缘端CNN适配的内存底线LoRA

    MemFLoRA: Memory-Floor LoRA for CNN Adaptation at the Edge

    [https://arxiv.org/abs/2610.08669](https://arxiv.org/abs/2610.08669)

    本文提出MemFLoRA，一种以内存优先为设计原则的低秩CNN适配器，通过激活内存底线准则（可训练的反向计算不依赖全宽度层输入）来解决边缘端CNN适配中激活内存而非可训练参数数量才是限制性资源的问题。

    

    当模型在部署后遇到用户、传感器或环境特定的偏移时，设备端学习是必不可少的。尽管参数高效微调（PEFT）方法，特别是低秩适应（LoRA）变体，能够实现边缘端的高效适配，但对于卷积神经网络（CNN）的适配而言，限制性资源往往不是可训练参数的数量，而是必须保留到反向传播之前的激活状态。本文提出了内存底线LoRA（MemFLoRA），这是一种围绕内存优先设计原则构建的低秩CNN适配器，而非对面向Transformer的LoRA的直接套用。我们不仅仅是减少可训练权重，而是定义了一个激活内存底线准则：可训练的反向计算不得依赖于全宽度的层输入。由此产生的适配器冻结下投影部分，训练一个尺度匹配的上投影部分，并结合评估模式的骨干网络归一化……

    arXiv:2610.08669v1 Announce Type: cross  Abstract: On-device learning is necessary when the model encounters user-,sensor-, or environment-specific shifts after deployment. Although parameter-efficient fine-tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA) variants, enable efficient adaptation at the edge, the limiting resource for Convolutional Neural Network (CNN) adaptation is often not the number of trainable parameters but the activation state that must be retained until the backward pass. This paper introduces Memory-Floor LoRA (MemFLoRA), a low-rank CNN adapter built around a memory-first design principle rather than a direct application of transformer-oriented LoRA. Instead of merely reducing trainable weights, we define an activation-memory-floor criterion: trainable backward computations must not depend on full-width layer inputs. The resulting adapter freezes the down-projection, trains a scale-matched up-projection, and combines eval-mode backbone normalization
    
[^19]: 语义行为水印：面向大语言模型智能体的抗改写且防伪造的溯源方法

    Semantic Behavioral Watermarking: Paraphrase-Robust and Forgery-Resistant Provenance for LLM Agents

    [https://arxiv.org/abs/2610.08668](https://arxiv.org/abs/2610.08668)

    提出语义行为水印（SBW），通过在历史条件约束下对语义动作簇进行基于密钥的抗碰撞分桶水印嵌入，实现了对改写鲁棒且能抵御伪造攻击的大语言模型智能体溯源方案。

    

    行为水印技术在大语言模型智能体的高层动作选择中嵌入所有者标识符，从而在不触碰输出词元的情况下实现来源溯源。以往的智能体水印方案会在两个方面失效。首先，此前所有三种方案都将水印绑定到确切的动作符号上，因此即使观测内容未被改动，仅仅重命名一个工具就会导致解码失步；在 AgentMark 自身的鲁棒性测试中，仅对观测内容进行改写就会使比特恢复率降至16.8%。其次，以往所有智能体水印研究只关注移除攻击：没有研究探讨攻击者能否伪造出一条能通过他人验证的轨迹——这一问题在文本水印领域已被肯定地回答（Jovanović 等人，2024）。我们提出了语义行为水印：在历史条件约束下对语义动作簇进行水印嵌入，用基于密钥的抗碰撞分桶取代公开簇分桶，其新桶的分配在随机预言机模型下可被证明是不可预测的。

    arXiv:2610.08668v1 Announce Type: cross  Abstract: Behavioral watermarking embeds an owner identifier in an LLM agent's high-level action choices, giving provenance without touching output tokens. Prior agent watermarks break in two ways. First, all three prior schemes bind the watermark to the exact action symbol, so renaming a tool desynchronizes decoding even when the observation is untouched; in AgentMark's own robustness test, paraphrasing the observation alone drops bit-recovery to 16.8%. Second, every prior agent watermark studies only removal: none asks whether an adversary can forge a trajectory that verifies as someone else's, a question answered affirmatively for text watermarks (Jovanovi\'c et al., 2024). We present Semantic Behavioral Watermarking (SBW): watermarking over semantic action clusters under history conditioning, with the public-cluster bin replaced by keyed collision-resistant binning whose fresh-bucket assignment is provably unpredictable in the random-oracle 
    
[^20]: ParanoiaEval：智能体编程中不必要防御性工作的基准测试

    ParanoiaEval: Benchmarking Unnecessary Defensive Work in Agentic Coding

    [https://arxiv.org/abs/2610.08662](https://arxiv.org/abs/2610.08662)

    提出了首个统一评估编程智能体风险应对能力的基准ParanoiaEval，基于风险管理中的规避-转移-缓解-接受框架，通过200对证据受控的仓库级任务对和专用评估指标来衡量智能体的防御性工作是否合理。

    

    随着编程智能体日益自主地承担真实世界的工作，判断其风险应对措施是否合理已变得尤为重要。现有工作从各自独立的角度评估相关的智能体行为，但缺乏一个统一这些行为的系统性框架。为弥合这一差距，我们提出了ParanoiaEval——首个用于统一评估编程智能体风险应对能力的基准。该基准以软件工程风险管理中成熟的“规避-转移-缓解-接受”（Avoidance-Transfer-Mitigation-Acceptance）框架为基础，将这4种基本风险应对措施操作化到编程智能体场景中，并包含200对证据受控的仓库级任务对，每对任务仅在定义应对措施的证据上存在差异。我们进一步引入了针对风险应对违规和证据响应性的专用指标，并采用经过人类校准的智能体裁判以实现可靠评估。在8个代表性模型上进行的大规模实验……

    arXiv:2610.08662v1 Announce Type: new  Abstract: As coding agents increasingly undertake real-world work autonomously, judging whether their risk treatments are warranted has become important. Existing work evaluates related agent behaviors from separate perspectives, but lacks a systematic framework for unifying these behaviors. To bridge this gap, we introduce ParanoiaEval, the first benchmark for unified evaluation of risk-treatment capabilities in coding agents. Grounded in the well-established Avoidance-Transfer-Mitigation-Acceptance framework in software engineering risk management, ParanoiaEval operationalizes its 4 fundamental treatments for coding-agent settings and contains 200 evidence-controlled repository-level task pairs, each differing only in treatment-defining evidence. We further introduce dedicated metrics for risk-treatment violations and evidence responsiveness, using a human-calibrated agentic judge for reliable evaluation. Large-scale experiments on 8 representat
    
[^21]: 面向视觉推理的强化学习更新选择性迁移

    Selective Transfer of RL Updates for Visual Reasoning

    [https://arxiv.org/abs/2610.08659](https://arxiv.org/abs/2610.08659)

    提出Selective-RL方法，通过选择性迁移强化学习更新中的主导矩阵方向并保留幅度，实现更有效的语言模型到视觉-语言模型的推理能力迁移。

    

    模型合并提供了一种无需训练的方法，可将语言模型的推理能力迁移到视觉-语言模型（VLM）中，但基于端点的迁移可能会将模型预先存在的差异与推理后训练中获得的变化混为一谈。我们转而围绕训练阶段的更新来构建能力迁移，隔离出由强化学习（RL）引起的参数变化。然而，完整迁移这一更新仍然不是最优的：我们发现其组成部分在跨模型可迁移性上存在显著差异，主导方向比完整更新的迁移效果更好。基于这一发现，我们提出了Selective-RL，该方法隔离RL阶段的更新，在保持幅度的同时保留其主导的矩阵级方向，并将其迁移到VLM的语言模块中。在三个模型家族和五个视觉推理基准上，Selective-RL改进了完整更新插值的效果。

    arXiv:2610.08659v1 Announce Type: cross  Abstract: Model merging provides a training-free way to transfer reasoning capabilities from language models to vision-language models (VLMs), but endpoint-based transfer can conflate pre-existing model differences with changes acquired during reasoning post-training. We instead formulate capability transfer around the training-stage update, isolating the parameter changes induced by reinforcement learning (RL). Yet transferring this update in full remains suboptimal: we find that its components differ substantially in cross-model transferability, with dominant directions transferring more effectively than the complete update. Based on this finding, we introduce Selective-RL, which isolates the RL-stage update, retains its dominant matrix-wise directions with magnitude preservation, and transfers them to the language modules of a VLM. Across three model families and five visual-reasoning benchmarks, Selective-RL improves full-update interpolatio
    
[^22]: 保障AI编写软件的案例研究

    A Case Study in Assuring AI-Written Software

    [https://arxiv.org/abs/2610.08651](https://arxiv.org/abs/2610.08651)

    本研究通过对一个由非软件专业操作员运用编码智能体构建并管理的生产级医疗平台进行案例研究，揭示了测试、监控器和审查智能体等AI软件监督机制本身的不可靠性，表明详尽的代码审查不能作为人类控制AI编写软件的唯一依据。

    

    软件工程智能体可以让没有接受过正规软件培训的人构建他们原本无法实现的系统，同时其生成的代码量也可能超出即使是最资深专家所能有效审查的范围。在这两种情况下，详尽的代码审查都不能作为人类控制的唯一可靠依据。我们报告了一项案例研究，研究对象是一个通过编码智能体构建、并由一名没有接受过正规软件工程培训的操作员管理的生产级医疗保健平台。随着时间推移，其工作流程演变为一个以人类为主导的元智能体系统：一个智能体负责编写代码，其他智能体负责监督和审查，项目规则则将经验教训传承下去。操作员发现，用于监督该系统的测试、监控器和审查智能体本身也是不可靠的：一些监控器测量的是代理指标而非实际结果，一些审计会静默失败，缺失的检查项会从报告结果中消失，还有一个自动化修复操作甚至造成了运营中断。在本案例中，

    arXiv:2610.08651v1 Announce Type: cross  Abstract: Software-engineering agents can enable people without formal software training to build systems they could not otherwise implement and simultaneously can produce more code than even experts can meaningfully inspect. In both cases, exhaustive code review is not reliable as the sole basis for human control. We report a case study of a production healthcare platform built through coding agents and governed by an operator without formal software-engineering training. Over time, its workflow grew into a human-led meta-agent system where one agent wrote code, other agents supervised and reviewed it, and project rules carried lessons forward. The operator found that tests, monitors and reviewing agents used to supervise the system were fallible. Some monitors measured proxies rather than outcomes, some audits failed silently, missing checks disappeared from reported results and one automated repair caused operational disruption. In this case,
    
[^23]: SquidAgent：明智并行，高效协调

    SquidAgent: Parallelize Wisely, Coordinate Efficiently

    [https://arxiv.org/abs/2610.08647](https://arxiv.org/abs/2610.08647)

    该论文提出SquidAgent，揭示了并行多智能体系统中“重新探索成本”与“对齐成本”这两项隐性开销，并据此推导出原则性决策准则：仅当并行化的关键路径成本加上这两项开销低于串行成本时，才应对该层进行并行化。

    

    基于大语言模型（LLM）的智能体能够解决复杂的多步骤任务，但顺序执行会带来显著的延迟。原则上，将工作并行分配给多个智能体应当产生接近线性的加速。然而，现有的并行多智能体系统往往比单智能体基线运行得更慢。我们将这一差距归因于并行执行会产生、而串行智能体可以避免的两项隐性成本。其一是重新探索成本：并行工作节点在重构编排器（orchestrator）已掌握的上下文（例如先前的决策）上所花费的冗余工作，而这些上下文在串行执行中本可被隐式继承。其二是对齐成本：为调和独立生成的输出之间的不一致性所需的额外开销。由此，我们推导出一个有原则的决策准则：只有当某一层的关键路径成本加上重新探索与对齐的开销低于相应的串行成本时，该层才应当被并行化。

    arXiv:2610.08647v1 Announce Type: new  Abstract: LLM-based agents solve complex multi-step tasks, but sequential execution incurs substantial latency. In principle, parallelizing work across multiple agents should yield near-linear speedups. Yet existing parallel multi-agent systems often run slower than a single-agent baseline. We attribute this gap to two hidden costs that parallel execution incurs but a serial agent avoids. First, there is a re-exploration cost: redundant effort spent by parallel workers reconstructing context that the orchestrator already possesses, such as prior decisions, that would otherwise be inherited implicitly in a serial execution. Second, there is an alignment cost: the overhead required to reconcile inconsistencies across independently generated outputs. We thus derive a principled decision criterion: a layer should be parallelized only when its critical-path cost, plus re-exploration and alignment overheads, is lower than the corresponding serial cost. 
    
[^24]: HygieneRoboBench：面向家庭机器人的卫生感知规划基准测试

    HygieneRoboBench: Benchmarking Hygiene-Aware Planning for Household Robots

    [https://arxiv.org/abs/2610.08642](https://arxiv.org/abs/2610.08642)

    提出了 HygieneRoboBench 基准（134 个任务族、624 个实例），用于评估家庭机器人在考虑接触历史污染、处理成本和用户优先级的条件下进行卫生感知安全规划的能力。

    

    arXiv:2610.08642v1 公告类型：交叉 摘要：与受污染物体的接触可能通过家庭机器人的夹爪、工具和共享表面传播危害，而新的接触可能会使原本安全的计划变得不安全。现有基准测试并未同时评估规划器如何从接触历史中识别卫生风险，并在发生新的接触事件后规划安全的后续行动。规划器必须在时间和资源限制内完成这些任务，同时尊重用户的优先级。我们提出了 HygieneRoboBench，包含 134 个任务族、共 624 个实例，用于评估机器人如何从给定的执行历史出发安全地完成家庭任务。这些任务通过两个夹爪和共享物体来捕捉污染情况，并纳入处理成本和用户优先级。我们将受控的历史、配置与事件比较与独立的计划评估相结合，以评估安全解决能力、用户优先级下的成本效率以及对接触事件的响应。对基于大语言模型（LLM）的规划器和符号规划器的评估表明，安全地完成一……（原文摘要在此处截断）

    arXiv:2610.08642v1 Announce Type: cross  Abstract: Contact with contaminated objects can spread hazards through a household robot's grippers, tools, and shared surfaces, while new contacts can make an existing plan unsafe. Existing benchmarks do not jointly assess how planners identify hygiene risks from contact history and plan safe continuations after new contact events. Planners must do so within time and resource limits while respecting user priorities. We introduce HygieneRoboBench, with 624 instances across 134 task families, to evaluate safe resolution of household tasks from a given execution history. Tasks capture contamination through two grippers and shared objects, treatment costs, and user priorities. We combine controlled history, profile, and event comparisons with independent plan evaluation. These assess safe resolution, cost efficiency under user priorities, and responses to contact events. Evaluation of LLM-based and symbolic planners shows that safely completing a t
    
[^25]: 用于精确且高效长时程规划的并行预测世界模型

    Parallel Predictive World Models for Accurate and Efficient Long-Horizon Planning

    [https://arxiv.org/abs/2610.08627](https://arxiv.org/abs/2610.08627)

    提出并行预测世界模型（PPWM），通过并行预测有限视野轨迹并在解码前让未来表示进行因果交互，消除了自回归滚动中的递归解码状态反馈，在四个视觉控制任务上实现最低长时程预测误差和最高CEM模拟器成功率。

    

    长时程世界模型规划通常依赖于自回归滚动，即预测出的状态被反复反馈回模型中。这种方式虽然保持了时间结构，但会产生一条与视野长度等长的串行路径，并使后续预测暴露于递归的解码状态反馈之中。我们提出了并行预测世界模型，它以并行方式预测有限视野轨迹，同时保留未来表示之间的因果交互。每个视野都以因果动作前缀为条件，未来表示在解码之前先进行交互，从而将时间因果关系与逐状态的输出递归分离开来。我们通过将自回归滚动形式化为一种因果轨迹映射，并识别出PPWM所移除的解码状态反馈通路，来明确这一区别。在四个视觉控制任务上，PPWM实现了最低的长时程预测误差以及最高的交叉熵方法模拟器成功率。

    arXiv:2610.08627v1 Announce Type: new  Abstract: Long-horizon world-model planning typically relies on autoregressive rollouts, where predicted states are repeatedly fed back into the model. This preserves temporal structure but creates a horizon-length sequential path and exposes later predictions to recursive decoded-state feedback. We introduce Parallel Predictive World Models (PPWM), which predict a finite-horizon trajectory in parallel while retaining causal interaction among future representations. Each horizon is conditioned on its causal action prefix, and future representations interact before decoding, separating temporal causality from state-by-state output recursion. We formalize this distinction by viewing autoregressive rollout as a causal trajectory map and identifying the decoded-state feedback pathway removed by PPWM. Across four visual-control tasks, PPWM achieves the lowest long-horizon prediction error and the highest Cross-Entropy Method (CEM) simulator success amo
    
[^26]: 扩散模型中的特征信息动力学

    Feature Information Dynamics in Diffusion

    [https://arxiv.org/abs/2610.08626](https://arxiv.org/abs/2610.08626)

    提出了基于 I-MMSE 恒等式的信息论框架“特征信息动力学”，通过比较无条件与特征条件去噪损失之差来估计特征信息密度，从而精确定位各特征在扩散生成过程中出现的时间，并定量证实了谱自回归现象。

    

    扩散模型通过一系列连续的去噪问题来生成数据，并被广泛观察到先呈现粗略结构、后生成精细细节。然而，这一直觉大多停留在经验性和定性层面。我们提出了特征信息动力学，这是一个用于定位特征在扩散过程中何时被生成的信息论框架。利用 I-MMSE 恒等式，我们将特征互信息的变化率与最优无条件去噪损失和特征条件去噪损失之间的差距联系起来，从而得到特征信息密度的实用估计器。我们进一步开发了一种链式分解方法，可在特征层次结构中分离共享信息与增量信息。我们首先利用该框架定量证实了像素扩散中的谱自回归现象，随后将分析扩展到频率维度之外：在“类别 → 掩码 → Canny”条件链下，各特征的信息密度在像素空间上存在差异。

    arXiv:2610.08626v1 Announce Type: cross  Abstract: Diffusion models generate data through a continuum of denoising problems, and are widely observed to reveal coarse structure before fine detail. Yet, this intuition is mostly empirical and qualitative. We introduce feature information dynamics, an information-theoretic framework for localizing when a feature is generated during diffusion. Using the I-MMSE identity, we connect the rate of feature mutual information change to a gap between optimal unconditional and feature-conditional denoising losses, yielding practical estimators for feature information density. We further develop a chained decomposition that separates shared from incremental information in a feature hierarchy. We use this framework first to quantitatively confirm spectral autoregression in pixel diffusion, and then to extend the analysis beyond frequency: under a class $\to$ mask $\to$ Canny conditioning chain, the per-feature information densities differ across pixel
    
[^27]: 面向平衡Adam的早期记忆选择方法

    Early Memory Selection for Balanced Adam

    [https://arxiv.org/abs/2610.08624](https://arxiv.org/abs/2610.08624)

    该论文提出一种通过短暂试点训练自动选择Adam共享记忆参数β的方法，利用三次记忆规则平衡采样波动与梯度平均延迟，在十一个视觉和语言任务上将平均相对验证差距降低40.7%以上。

    

    我们提出了一种通过短暂的试点训练来选择Adam中共享记忆参数 $\beta_1=\beta_2=\beta$ 的方法。所选的 $\beta$ 在随后的完整训练过程中保持固定。通过对Adam归一化方向的局部建模，我们平衡了采样波动性与平均历史梯度所引入的延迟。这种平衡导出了一个三次记忆规则，其两个系数通过在少数试点检查点处的梯度探测来估计。该估计器联合使用分子和分母，从而保留了两者的协方差。在200步更新的试点训练中，于四个检查点各使用十六个探测梯度，在与随机种子匹配的回溯评估中，该方法在十一个视觉和语言工作负载上相比共享 $\beta=0.95$ 的网格代表值，将平均相对验证差距降低了40.7%，最差四分之一平均差距降低了44.3%。其平均差距还比从全部十一个工作负载中选出的最佳常数 $\beta$ 低32.3%。

    arXiv:2610.08624v1 Announce Type: cross  Abstract: We propose a method for choosing the shared memory parameter $\beta_1=\beta_2=\beta$ in Adam from a short pilot training. The selected $\beta$ remains fixed during the subsequent full training. A local model of Adam's normalized direction balances sampling variability against the delay introduced by averaging past gradients. This balance gives a cubic memory rule, whose two coefficients are estimated from gradient probes at a few pilot checkpoints. The estimator uses the numerator and denominator jointly, preserving their covariance. With a 200-update pilot and sixteen probe gradients at each of four checkpoints, a seed-matched retrospective evaluation on eleven vision and language workloads reduces mean relative validation gap by 40.7% and worst-quarter mean gap by 44.3% against the grid representative of shared $\beta=0.95$. The mean gap is also 32.3% lower than that of the best constant $\beta$ chosen across all eleven workloads.
    
[^28]: 使用约束性创意实现互联网规模服务的智能体式根因分析

    Agentic RCA for Internet-Scale Services Using Constrained Creativity

    [https://arxiv.org/abs/2610.08622](https://arxiv.org/abs/2610.08622)

    提出了E4智能体式故障排查系统，通过“约束性创意”范式将LLM的自动化探索能力与结构化方法的可解释性和高效性相结合，同时满足互联网规模服务根因分析的高准确性、低成本、可解释和低运维投入四项要求。

    

    互联网规模服务的系统管理员需要解决故障事件，以维护此类服务的可靠性。理想情况下，我们希望一个故障排查系统能够：（1）对已知和未知事件都具有高准确度的表达能力；（2）在大规模应用下具有成本效益；（3）具备可解释性，能为运维人员提供可操作的洞察；（4）给运维人员带来的工作量较低。遗憾的是，大多数现有系统——包括新兴的LLM辅助智能体工作流以及用于编写多样化RCA算法的结构化框架——都无法同时满足这四项要求。我们提出了E4，一个面向互联网规模服务的新型智能体式故障排查系统。E4体现了“约束性创意”这一范式，将LLM辅助的自动化与探索能力的优势，与结构化方法的可解释性和高效性相结合。与让LLM智能体编写任意代码或生成任意响应不同，我们提供……（原文截断）

    arXiv:2610.08622v1 Announce Type: cross  Abstract: System administrators of Internet-scale services need to resolve failure incidents to maintain reliability of such services. Ideally, we want a troubleshooting system to be: (1) expressive to known and unknown incidents with high accuracy; (2) cost efficient at scale; (3) explainable to provide actionable insights operators can act on; and (4) entail low effort from the operators. Unfortunately, most existing systems, including emerging LLM-assisted agentic workflows and structured frameworks for authoring diverse RCA algorithms fall short of achieving all four requirements. We present E4, a novel agentic system for troubleshooting for Internet-scale services. E4 embodies the paradigm of constrained creativity that combines the best of LLM-assisted automation and exploration with the explainability and efficiency of a structured approach. Instead of allowing an LLM agent to write arbitrary code or generate arbitrary responses, we provi
    
[^29]: 递归游戏创造者：一个面向体验的智能体级产品游戏开发框架

    Recursive Game Creator: An Agentic Product-Level Experience-Oriented Game Harness

    [https://arxiv.org/abs/2610.08621](https://arxiv.org/abs/2610.08621)

    该论文提出递归游戏创造者框架，通过设计者、构建者、玩家和评审者四个智能体的递归协作，将粗糙的游戏原型迭代开发为真正注重玩家体验的有趣游戏。

    

    近期的游戏设计智能体在生成可玩游戏方面取得了长足进步。然而，程序的正确性并不能保证玩家获得愉快的游戏体验。我们提出了递归游戏创造者，这是一个面向体验的框架，旨在将智能体游戏开发从粗糙的游戏原型推进为有趣的游戏。递归游戏创造者围绕四个组件组织递归式开发：设计者、构建者、玩家和评审者。设计者将用户指令和评审者的反馈转化为详细的计划。构建者将这些计划转化为候选游戏。基于代码原生的玩家通过编程接口创建并执行可复用的策略，以高效收集多样化的游戏玩法轨迹，缓解了基于图形界面（GUI）的缓慢收集方式所导致的评估偏差。评审者使用精心设计的基于轨迹的指标来推断玩家偏好，并结合视觉证据和明确的文本偏好（进行综合评估）。

    arXiv:2610.08621v1 Announce Type: new  Abstract: Recent game design agents have made substantial progress in generating playable games. However, program correctness does not ensure an enjoyable experience for players. We present Recursive Game Creator, an experience-oriented harness to advance agentic game development from rough game prototypes into entertaining games. Recursive Game Creator organizes recursive development around four components: Designer, Builder, Player, and Reviewer. The Designer translates user instructions and Reviewer's feedback into detailed plans. The Builder turns these plans into candidate games. The coding-native Player creates and executes reusable policies through programmatic interfaces to efficiently collect diverse gameplay trajectories, mitigating evaluation bias caused by slow GUI-based collection. The Reviewer uses carefully designed trajectory-based metrics to induce player preferences, integrating with visual evidence and explicit textual preferenc
    
[^30]: 一种群体协同多机器人系统：利用多模态叶片传感实现农田行间作物早期胁迫检测

    A Swarm-Coordinated Multi-Robot System for Early Stress Detection in Agricultural Rows Using Multimodal Leaf Sensing

    [https://arxiv.org/abs/2610.08603](https://arxiv.org/abs/2610.08603)

    提出了一种低成本群体协同多机器人系统CropSentry，通过多模态叶片传感与颜色编码实时仪表盘实现农田行间作物早期胁迫的持续监测，总体分类准确率达84.12%。

    

    当今，作物早期胁迫检测对于提高效率、减少时间、金钱和精力的浪费至关重要。然而，大多数现代技术，如高光谱成像和基于人工智能的系统，成本高昂且复杂，中小规模农户难以实施。本文展示了CropSentry，一个低成本的地面多机器人系统，利用多模态叶片传感通过跟踪胁迫水平来持续监测作物健康。该系统由两个自主机器人组成，逐行持续检测叶片颜色和环境数据。观测结果经空间映射后发送至主机器人，主机器人使用颜色编码的行片段生成基于网络的实时仪表盘，以展示作物健康状况。在实验中收集了63次观测数据后，结果显示作物健康分类的总体准确率为84.12%，其中健康植物的准确率为82.60%，营养缺乏植物为88%。

    arXiv:2610.08603v1 Announce Type: cross  Abstract: Early stress detection in crops is a necessity today to improve efficiency and reduce waste of time, money, and effort. However, most modern techniques, such as hyperspectral imaging and AI-based systems, are too costly and complex for medium and small-scale farmers to implement. This paper showcases CropSentry, a low-cost, ground-based multi-robot system that uses multimodal leaf sensing to continuously monitor crop health by tracking stress levels. The system comprises two autonomous bots that continuously detect leaf color and environmental data row by row. The observations are spatially mapped and sent over to the master bot, which uses color-coded row segments to generate a real-time web-based dashboard displaying crop health. After 63 observations were collected during the experiments, the results showed an overall crop health classification accuracy of 84.12%, with 82.60% for healthy plants, 88% for nutrient-deficient plants, an
    
[^31]: 我为人人，人人为一：基于随机最优控制的协调多智能体扩散引导

    One for All, All for One: Coordinated Multi-Agent Diffusion Steering via Stochastic Optimal Control

    [https://arxiv.org/abs/2610.08595](https://arxiv.org/abs/2610.08595)

    提出 CMDS 框架，将冻结的预训练扩散模型作为可复用的生成基元，把多智能体协调表述为随机最优控制问题并学习摊销化的控制，从而仅通过学习协调方式即可由独立训练的组件生成器产生连贯的结构化输出，并可在新任务实例中复用。

    

    深度生成模型通常产生由相互作用的组件构成的结构化输出。用单一模型对这些输出进行建模，需要同时学习组件分布及其交互关系。我们探索一种模块化的替代方案：复用独立训练的组件生成器，仅学习如何协调它们以产生连贯的结构化输出。我们的框架——协调多智能体扩散引导（CMDS）——将冻结的预训练扩散模型视为可复用的生成基元，并通过学习到的控制来协调它们的逆过程。我们将协调问题表述为一个随机最优控制问题，在刻画组合输出所需性质的组装级奖励与偏离预训练动态之间进行权衡。学习到的控制对该优化进行了摊销，使其能够在新的任务实例中复用。实验表明，CMDS 能够恢复已知的目标分布。

    arXiv:2610.08595v1 Announce Type: cross  Abstract: Deep generative models often produce structured outputs composed of interacting components. Modelling these outputs with a single model requires learning both the component distributions and their interactions. We pursue a modular alternative: reuse independently trained component generators and learn only how to coordinate them to produce coherent structured outputs. Our framework, Coordinated Multi-Agent Diffusion Steering (CMDS), treats frozen pretrained diffusion models as reusable generative primitives and coordinates their reverse processes through a learned control. We formulate coordination as a stochastic optimal control problem, balancing an assembly-level reward that specifies the desired properties of the combined output against deviations from the pretrained dynamics. The learned control amortises this optimisation, allowing reuse across new task instances. Experiments show that CMDS can recover a known target distribution
    
[^32]: MINDSET：面向长对话智能体记忆的基于能量的图式演化

    MINDSET: Energy-based Schema Evolution for Long Conversational Agent Memory

    [https://arxiv.org/abs/2610.08586](https://arxiv.org/abs/2610.08586)

    MINDSET提出了一种基于最小能量状态转换的记忆控制器，将对话存储为不可变情节并组织成带版本的图式，使长对话智能体能够适应随时间演变的指令与上下文，同时保留历史状态且无需反复调用大语言模型重写记忆。

    

    长对话智能体已成为我们日常生活中不可或缺的存在。它们必须记住很久之前的对话内容，以便高效地帮助我们完成任务，而无需用户反复重复指令和上下文。然而，主要问题在于指令和上下文会随时间发生变化，因此智能体必须能够相应地进行调整。一个有效的记忆系统应当同时保留当前与历史状态，区分过时信息与活跃知识，检索与查询相匹配的证据，并避免反复调用大语言模型来重写先前的交互内容。我们提出了MINDSET，这是一种记忆控制器，它将对话存储为不可变的情节，并通过最小能量状态转换将这些情节组织成带版本的图式。每个新到来的情节可以强化、取代、拆分或创建一个图式。该转换决策在表示失真、矛盾、历史数据等因素之间进行权衡（原文摘要在此处截断）。

    arXiv:2610.08586v1 Announce Type: new  Abstract: Long conversational agents have become essential in our daily lives. They must remember what was said long back in order to help us efficiently complete a task without needing the user to repeat instructions and context repeatedly. However, the main issue is that instructions and context change over time and so the agents must be able to adapt accordingly. A useful memory system should preserve both current and historical states, distinguish stale information from active knowledge, retrieve evidence appropriate to the query and avoid repeatedly invoking a large language model to rewrite prior interactions. We introduce MINDSET, a memory controller that stores a conversation as immutable episodes and organizes them into versioned schemas through minimum-energy state transitions. Each incoming episode may reinforce, supersede, split or create a schema. The transition decision balances representation distortion, contradiction, historical da
    
[^33]: 学习方式如何跨越记忆-泛化谱系支配遗忘

    How Learning Governs Unlearning across the Memorization-Generalization Spectrum

    [https://arxiv.org/abs/2610.08577](https://arxiv.org/abs/2610.08577)

    模型的学习方式决定了其后续遗忘的表现：偏泛化型模型在遗忘时比偏记忆型模型遭受更大的保留集性能损害，且这一趋势在记忆-泛化谱系上几乎单调成立。

    

    虽然机器遗忘旨在消除模型通过学习获得的不想要的能力，但很少有研究考察模型的学习方式如何塑造其后续的遗忘过程。在本文中，我们从记忆和泛化的角度研究这一联系，这是模型在训练期间采用的两种最具代表性却又相互竞争的策略。我们首先利用模加法中的grokking（顿悟）现象对偏记忆型模型和偏泛化型模型进行分类，并比较它们对遗忘操作的响应，结果表明偏泛化型模型遭受更大的保留损害，即在保留集上的性能下降更为严重。此外，我们通过引入分桶模加法进行了更细粒度的分析，在该设置下，两种策略各自的贡献可以在记忆-泛化谱系上被显式控制。在这一设置中，我们再次证实相同的趋势持续存在且几乎呈单调关系。我们进一步证明……（原文摘要在此处截断）

    arXiv:2610.08577v1 Announce Type: cross  Abstract: While unlearning seeks to negate undesired capabilities acquired through learning, little research has examined how the way models learn shapes their subsequent unlearning. In this paper, we investigate this connection from the perspectives of memorization and generalization, the two most representative yet competing strategies that models employ during training. We first classify memorization- and generalization-heavy models using grokking in modular addition and compare their responses to unlearning, showing that the latter suffer greater retain damage, i.e., a larger performance drop on the retain set. Furthermore, we conduct a finer-grained analysis by introducing bucketed modular addition, in which the respective contributions of the two strategies can be explicitly controlled across the memorization-generalization spectrum. In this setup, we reaffirm that the same trend persists and is nearly monotonic. We further demonstrate tha
    
[^34]: FedDermaSeg：面向皮肤病学图像分割的联邦学习

    FedDermaSeg: Federated Learning for Dermatological Image Segmentation

    [https://arxiv.org/abs/2610.08574](https://arxiv.org/abs/2610.08574)

    该论文提出FedDermaSeg，探索利用联邦学习在无需集中收集数据的情况下实现隐私保护的皮肤病灶分割，以解决传统集中式深度学习训练带来的隐私风险和高计算资源需求问题。

    

    皮肤癌是一个重要的全球健康问题，早期发现和精确的病灶勾画对于有效的诊断与治疗规划至关重要。自动化的皮肤病灶分析可以辅助皮肤科医生，其中病灶分割是计算机辅助诊断系统中的基础步骤。传统的基于深度学习的分割模型通常依赖于集中式训练，即图像及其对应的分割掩码被收集到中央服务器上。这种数据聚合方式在医疗应用中引发了隐私方面的担忧，并且需要大量的集中式计算资源。为了解决这些局限性，我们研究了联邦学习在保护隐私的皮肤病灶分割中的可行性。我们使用ISIC 2018皮肤病灶分割挑战赛数据集的训练集和验证集来模拟分布式学习环境，并开发联邦分割模型。

    arXiv:2610.08574v1 Announce Type: cross  Abstract: Skin cancer is a major global health concern, and early detection and accurate lesion delineation are important for effective diagnosis and treatment planning. Automated skin lesion analysis can assist dermatologists, with lesion segmentation serving as a fundamental step in computer-aided diagnostic systems. Conventional deep learning-based segmentation models typically rely on centralized training, where images and their corresponding segmentation masks are collected on a central server. Such data aggregation raises privacy concerns in medical applications and requires substantial centralized computational resources. To address these limitations, we investigate the feasibility of federated learning for privacy-preserving skin lesion segmentation. The training and validation sets of the ISIC 2018 Skin Lesion Segmentation Challenge dataset are used to simulate a distributed learning environment and develop a federated segmentation mode
    
[^35]: RAG-PIBench：一个面向可信RAG系统中提示注入检测的防泄漏基准

    RAG-PIBench: A Leakage-Aware Benchmark for Prompt-Injection Detection in Trustworthy RAG Systems

    [https://arxiv.org/abs/2610.08571](https://arxiv.org/abs/2610.08571)

    该论文提出了RAG-PIBench——一个面向RAG系统提示注入检测的防泄漏基准，包含4,876个上下文示例，通过严格评估协议发现DistilBERT取得最佳检测性能（F1=0.896），同时证明TF-IDF等稀疏基线方法仍具竞争力。

    

    检索增强生成（RAG）系统容易受到嵌入在检索内容中的提示注入攻击。我们提出了RAG-PIBench，一个用于RAG式提示注入检测的基准，包含4,876个上下文示例，分布在固定的训练集、验证集和受保护的测试集划分中。通过采用防泄漏的构建流程和严格的评估协议，我们比较了基于关键词、语义参考、TF-IDF以及基于Transformer的检测器。DistilBERT在受保护测试集上取得了最佳性能（F1 = 0.896，PR-AUC = 0.968），而TF-IDF SVM和逻辑回归仍保持竞争力。我们的结果展示了防泄漏基准设计和强稀疏基线对于RAG系统中可靠提示注入检测的价值。

    arXiv:2610.08571v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) systems are vulnerable to prompt-injection attacks embedded in retrieved content. We introduce RAG-PIBench, a benchmark for RAG-style prompt-injection detection containing 4,876 contextual examples across frozen train, validation, and protected-test splits. Using a leakage-aware construction pipeline and strict evaluation protocol, we compare keyword-based, semantic-reference, TF-IDF, and transformer-based detectors. DistilBERT achieves the best protected-test performance (F1 = 0.896, PR-AUC = 0.968), while TF-IDF SVM and logistic regression remain competitive. Our results demonstrate the value of leakage-aware benchmark design and strong sparse baselines for reliable prompt-injection detection in RAG systems.
    
[^36]: 面向大语言模型推理的自适应幂采样

    Adaptive Power Sampling for LLM Reasoning

    [https://arxiv.org/abs/2610.08563](https://arxiv.org/abs/2610.08563)

    提出自适应幂采样（APS），在测试时依据查询难度和模型自奖励逐查询调整分布锐化指数，从而在无需训练的情况下显著提升大语言模型的推理性能。

    

    序列级幂采样最近作为一种无需训练的推理方法而兴起，其方式是从基础大语言模型（LLM）被锐化后的输出分布中进行采样。然而，现有方法通常对所有查询统一地锐化基础模型分布，忽略了查询难度的差异以及基础模型本身对每个查询的掌握程度。本工作的目标是为幂采样赋予查询自适应性。在理论上，我们证明进一步锐化所带来的收益取决于正确响应与错误响应之间的自奖励差距。基于这一洞察，我们提出了自适应幂采样（APS），它在测试时利用答案一致性与模型自奖励之间的关系，针对每个查询自适应地调整锐化指数。在MATH500、HumanEval和GPQA等多种推理任务上的实验表明，APS始终优于幂采样基线方法。

    arXiv:2610.08563v1 Announce Type: new  Abstract: Sequence-level power sampling has recently emerged as a training-free approach to reasoning by sampling from a sharpened output distribution of a base large language model (LLM). Nevertheless, existing methods typically sharpen the base model distribution uniformly across queries, overlooking variations in query difficulty and in how well the base model already handles each query. The goal of this work is to equip power sampling with query adaptivity. Theoretically, we show that the benefits of further sharpening are determined by the self-reward gap between correct and incorrect responses. Based on this insight, we propose \emph{Adaptive Power Sampling} (APS), which adjusts the sharpening exponent on a per-query basis at test time using the relationship between answer agreement and the model's self-reward. Experiments across diverse reasoning tasks, including MATH500, HumanEval, and GPQA, show that APS consistently outperforms power sam
    
[^37]: 大语言模型潜空间中的偏见方向捕获的是置信度，而非公平性

    Latent space bias directions in LLMs capture confidence, not fairness

    [https://arxiv.org/abs/2610.08559](https://arxiv.org/abs/2610.08559)

    该研究揭示了大语言模型激活引导中的去偏方向实际上编码的是模型置信度而非偏见信息，其去偏效果只是降低模型置信度的副产品，从而解释了激活引导去偏技术泛化能力差的根本原因。

    

    arXiv:2610.08559v1 公告类型： cross 摘要：激活引导（activation steering）作为一种轻量级的大语言模型推理时去偏技术日益流行。然而，先前的研究报告指出，引导向量的泛化能力较差，会对模型性能产生意外影响，并且向新数据集的迁移能力有限。我们的工作分析了用于激活引导的去偏方向实际上编码了什么，以揭示其性能不一致的原因。我们研究了通过对比反偏见提示和偏见提示的激活所获得的线性去偏方向，并将其作为一种引导干预措施，在偏见基准和通用知识基准上进行评估。我们发现，这个方向主要由模型置信度主导，在激活空间中从高概率token区域指向低概率token区域，而不是编码模型偏见的有意义表征。沿着该方向进行引导确实能减少测量到的偏见，但这是降低模型置信度的结果：在问答（QA）基准上，我们……

    arXiv:2610.08559v1 Announce Type: cross  Abstract: Activation steering has gained popularity as a lightweight inference-time debiasing technique for large language models. However, prior work reports that steering vectors generalise poorly, with unintended effects on model performance and limited transfer to new datasets. Our work analyses what the debiasing direction used for activation steering actually encodes, in order to shed light on its inconsistent performance. We study the linear debiasing direction obtained by contrasting the activations of anti-biased and biased prompts, and evaluate it as a steering intervention across bias and general knowledge benchmarks. We find that this direction is dominated by model confidence, pointing from regions of high to low-probability tokens in activation space rather than encoding a meaningful representation of model bias. Steering along it does reduce measured bias, but this is a consequence of reducing model confidence: on QA benchmarks we
    
[^38]: 系统化知识（SoK）：以人为中心的青少年AI安全

    Systemization of Knowledge (SoK): Human-Centered AI Safety for Youth

    [https://arxiv.org/abs/2610.08554](https://arxiv.org/abs/2610.08554)

    本文系统回顾了100项HCI实证研究，构建了青少年AI风险与对策的映射，发现多数风险仅有构想层面的对策，很少被实施和评估，且评估多聚焦技术性能而非实际防伤害效果。

    

    尽管人机交互（HCI）领域日益关注面向青少年的AI安全问题，但相关文献仍缺乏对以下问题的全面认识：已识别出哪些风险、这些风险如何被应对、以及所提出的保护措施在实践中是否有效。我们系统性回顾了100项实证HCI研究，这些研究涉及儿童和青少年在学校、家庭、照护场所以及公共服务中与AI交互或暴露于AI的情形。借助YAIR风险分类法和MIT缓解措施分类法，我们梳理了哪些风险已被识别、每项风险是否有相应的应对措施、以及针对该风险的每项措施是否已被实施乃至评估。风险-对策映射显示：大多数风险仅配有“已提出/构想中”的对策；很少有对策被实际实施，被评估的更少；且现有评估往往衡量技术性能，而非对伤害的实际防护效果。我们进一步识别了覆盖范围的缺口所在……

    arXiv:2610.08554v1 Announce Type: cross  Abstract: While HCI increasingly examines AI-safety for youth, the literature lacks a comprehensive view of what risks have been identified, how they are addressed, and whether proposed protections work in-practice. We systematically reviewed 100 empirical HCI studies involving children and youth interacting with or exposed to AI across schools, homes, care settings, and public services. Using the YAIR taxonomy for risks and the MIT Mitigation Taxonomy for countermeasures, we map which risks have been identified, whether each risk is addressed by countermeasure(s), and whether each countermeasure for that risk is implemented and even evaluated. The risk-countermeasure mapping shows that most risks are matched only with proposed/ideated countermeasures; few countermeasures have been implemented, and fewer still evaluated; and existing evaluations often measure technical performance rather than protection from harm. We identify where coverage is a
    
[^39]: DeltaTTT：非线性循环记忆的逐层优化

    DeltaTTT: Layerwise Optimization for Nonlinear Recurrent Memory

    [https://arxiv.org/abs/2610.08553](https://arxiv.org/abs/2610.08553)

    针对非线性循环记忆在测试时训练中难以优化、并行基线反而优于串行版本的问题，DeltaTTT提出用逐层学习替代联合内循环优化，为每层分配局部预测目标并通过状态依赖的delta规则更新，从而缓解非线性记忆的优化困难。

    

    序列测试时训练通过一系列连续更新来适配记忆网络，每次更新都基于网络的先前状态计算内循环梯度。直观上，这种状态依赖性应使每次更新能够考虑到记忆已经学到的内容，并更好地融合新信息。然而，我们发现这一预期优势在非线性记忆中并未稳定实现：固定基底的并行TTT基线反而优于其串行对应版本。我们的探索性实验揭示了一个关键的潜在困难：在单次序列遍历过程中，非线性记忆可能比线性记忆更难优化。为缓解这一优化困难，我们提出DeltaTTT，它用逐层学习取代了两层记忆网络的联合内循环优化。每一层被分配一个局部预测目标，并通过状态依赖的delta规则进行更新。该公式化表述保留了非线性...

    arXiv:2610.08553v1 Announce Type: cross  Abstract: Sequential test-time training adapts a memory network through successive updates, each computing an inner-loop gradient based on the network's previous state. Intuitively, this state dependence should allow each update to account for what the memory has already learned and better incorporate new information. However, we find that this expected advantage does not consistently materialize in nonlinear memories: a fixed-base parallel TTT baseline outperforms its serial counterpart. Our exploratory experiments point to a key underlying difficulty: nonlinear memories can be harder to optimize than linear ones within a single pass over the sequence. To alleviate this optimization difficulty, we introduce DeltaTTT, which replaces joint inner-loop optimization of a two-layer memory network with layerwise learning. Each layer is assigned a local prediction target and updated through a state-dependent delta rule. This formulation retains a nonli
    
[^40]: AnyBottle：一种只保留真正所需概念的方法

    AnyBottle: A Recipe to Only Keep the Concepts You Really Need

    [https://arxiv.org/abs/2610.08552](https://arxiv.org/abs/2610.08552)

    AnyBottle提出了一种由黑盒教师模型引导的迭代概念选择方法，结合嵌套dropout训练，能够构建紧凑、任务特定的概念瓶颈模型，只保留任务真正需要的概念，使瓶颈更小且更易于检查。

    

    概念瓶颈模型（CBMs）通过将预测路由经过人类可解释的概念，使预测变得可检查和可干预，但最初需要概念标注。无需标注的变体虽然去除了这一要求，但通常使用大型概念词汇表，且在训练和推理时都是静态的，产生的瓶颈比任何任务或预测实际所需的都要大，也更难检查。我们提出AnyBottle，一种构建紧凑、任务特定的CBM的统一方法。AnyBottle仅假设有一个冻结的骨干网络和一个无监督概念池（例如稀疏自编码器）。随后，在同一骨干网络上训练的黑盒教师模型引导概念选择：每一轮添加最能解释瓶颈当前失败原因的概念，且候选概念被限制在教师与学生模型不一致的区域。通过按此选择顺序进行嵌套dropout训练，最终的瓶颈模型可以从任何概念前缀实现准确预测，因此推理时只需……（原文截断）

    arXiv:2610.08552v1 Announce Type: new  Abstract: Concept bottleneck models (CBMs) make predictions inspectable and intervenable by routing them through human-interpretable concepts, but originally required concept annotations. Annotation-free variants remove this requirement, but typically use large concept vocabularies, static at both training and inference, producing bottlenecks larger than any task or prediction needs and harder to inspect. We propose AnyBottle, a single recipe for building compact, task-specific CBMs. AnyBottle assumes only a frozen backbone and an unsupervised concept pool, such as a sparse autoencoder. A black-box teacher trained on the same backbone then guides selection: each round adds the concept that best explains the bottleneck's current failures, with candidates restricted to regions of teacher/student disagreement. Trained with nested dropout over this selection order, the final bottleneck predicts accurately from any concept prefix, so inference spends f
    
[^41]: 0.6 有多高？可解释性探测中的地板、天花板与余量

    How High Is 0.6? Floors, Ceilings, and Headroom in Interpretability Probing

    [https://arxiv.org/abs/2610.08544](https://arxiv.org/abs/2610.08544)

    该论文提出用“地板”（简单输入已能预测的水平）和“天花板”（完整输入能预测的水平）两个参照点以及二者之间的“余量”来校准可解释性探测得分，使探针分数具有明确、可比的含义，并证明余量在目标不依赖隐藏变量或输入不透露隐藏变量时消失。

    

    探测是可解释性研究的主力工具。如果模型的隐藏状态能够预测某个变量，就称该模型表征了这个变量。但探测得分并没有固定的含义。0.6 的 R² 可能仅仅反映了输入本身已经透露的信息，而相同的得分在不同的数据上可能意味着不同的东西。我们提出用两个参照点来解读每一个探测得分：地板，即一组声明的简单输入已经能预测的内容；以及天花板，即完整输入所能预测的内容。两者之间的差距，即余量，是探测能够展示模型计算出超出简单输入之外内容的范围。我们证明余量会在两种情况下消失：目标不再依赖于模型必须推断的隐藏变量，或者输入不再透露该变量。我们在为上下文内元分析训练的 transformer 上测试了这一方法，这些模型必须推断研究之间隐藏的异质性才能正确加权，而两个参照点在此都是可控制的。

    arXiv:2610.08544v1 Announce Type: cross  Abstract: Probes are the workhorse of interpretability. If a model's hidden states predict a variable, the model is said to represent it. But a probe score has no fixed meaning. An $R^2$ of 0.6 may only reflect what the input already gives away, and the same score can mean different things on different data. We propose reading every probe score against two reference points: a floor, what a declared set of simple inputs already predicts, and a ceiling, what the full input can predict. The gap between them, the headroom, is the range in which a probe can show that a model computes something beyond the simple inputs. We prove that headroom vanishes in two ways: the target stops depending on a hidden variable the model must infer, or the input stops revealing it. We test this on transformers trained for in-context meta-analysis, which must infer the hidden heterogeneity between studies to weight them correctly, and where both reference points are kn
    
[^42]: 面向安全实时机器人控制的微型神经策略

    Micro Neural Policies for Safe Real-Time Robotic Control

    [https://arxiv.org/abs/2610.08541](https://arxiv.org/abs/2610.08541)

    该论文提出微型神经策略（MNP），通过结合进化策略与统计模型检测验证进行策略搜索，将神经网络的内存占用缩小至0.5至7.5 kB，使策略能够部署在微控制器上实现安全、鲁棒的实时机器人控制，并成功完成零样本的仿真到现实迁移。

    

    在这篇论文中，我们研究了微型神经策略的合成方法，以在计算资源受限的嵌入式设备上实现安全且鲁棒的实时机器人控制。我们证明，将进化策略（ES）与基于统计模型检测（SMC）的验证相结合来进行策略搜索，能够在不牺牲安全性和鲁棒性的前提下大幅缩小神经网络规模。我们在Cartpole和四旋翼控制任务上对MNP进行了大规模的训练与评估，涵盖了不同的控制频率和网络架构。在仿真中验证这些策略后，我们通过零样本迁移到物理系统来评估其可部署性。实验表明，MNP能够在不牺牲控制性能的情况下成功实现安全的仿真到现实迁移。此外，我们展示了这些策略的内存占用仅为0.5至7.5 kB，可以部署在微控制器上并实现实时推理。

    arXiv:2610.08541v1 Announce Type: cross  Abstract: In this paper, we investigate the synthesis of Micro Neural Policies (MNP) to enable safe and robust real-time robotic control on computationally constrained embedded devices. We demonstrate that integrating Evolution Strategy (ES) and Statistical Model Checking (SMC)-based verification for policy search can drastically reduce neural network size without compromising safety and robustness. We conduct a large-scale training and evaluation of MNP on Cartpole and Quadrotor control tasks, varying control frequencies and network architectures. After validating these policies in simulation, we evaluate their deployability through zero-shot transfer to physical systems. Our experiments show that MNP can successfully achieve safe sim-to-real transfer without sacrificing control performance. We then show that the policies' memory footprint, ranging from 0.5 to 7.5 kB, allows deployment on microcontrollers, where they achieve real-time inference
    
[^43]: 迈向对齐缩放定律：一个框架与首批预注册测量

    Toward Alignment Scaling Laws: A Framework and First Preregistered Measurements

    [https://arxiv.org/abs/2610.08540](https://arxiv.org/abs/2610.08540)

    该论文提出将对齐视为一族可测量的幂律缩放关系（B_r(N)=a_rN^alpha_r）的框架及首批预注册测量，并证明长期对齐状态由经修正风险中的最大指数而非平均值决定，指数大于1时将累积不可持续的对齐债务。

    

    随着模型规模增长，对齐究竟变得更容易还是更难，这一争论往往基于孤立的发现，仿佛对齐是单一属性。我们将其视为一族可测量的缩放关系：对于每个风险类别 r，维持固定安全目标所需的对齐负担被建模为 B_r(N)=a_rN^alpha_r，其中 N 是能力代理指标；相对于与 N 成比例的预算，若 alpha_r<1，缩放有助于对齐；若 alpha_r≈1，缩放能保持同步；若 alpha_r>1，则会累积对齐债务。我们给出了负担的三种操作化定义，并区分了观测对齐、审计对齐与真实对齐。一个修正会消耗能力余量的玩具模型使这些后果变得明确。我们证明：决定长期状态的是经修正风险中的最大指数，而非平均值；当指数大于1时，任何要将余量维持在某一底线之上的策略都必须以超指数速度增长；对于作为幂律正混合的负担，在小模型上进行的拟合会低估大尺度上的（原文在此截断）。

    arXiv:2610.08540v1 Announce Type: new  Abstract: Whether alignment gets easier or harder as models grow is often argued from isolated findings, as if alignment were one property. We treat it as a family of measurable scaling relations: for each risk category r, the alignment burden needed to hold a fixed safety target is modeled as B_r(N)=a_rN^alpha_r, with N a capability proxy; against a budget proportional to N, scaling helps if alpha_r<1, keeps pace if alpha_r~1, and accumulates alignment debt if alpha_r>1. We give three operationalizations of burden and distinguish observed, audited and true alignment. A toy model, in which corrections consume capability headroom, makes the consequences explicit. We prove that the largest exponent among corrected risks, not an average, sets the long-run regime; that above 1 any policy holding headroom above a floor must grow super-exponentially; that, for burdens that are positive mixtures of power laws, fits on small models underestimate large-sca
    
[^44]: 从共享需求模式到局部不确定性：基于混合紧凑适配组件的概率负荷预测

    From Shared Demand Patterns to Local Uncertainty: Probabilistic Load Forecasting by Mixing Compact Adaptations

    [https://arxiv.org/abs/2610.08538](https://arxiv.org/abs/2610.08538)

    该论文提出了一种可扩展的概率负荷预测框架，通过共享模型学习共同需求模式，并让每个负荷按需混合一个小型低维紧凑适配组件库，从而在客户级和变压器级兼顾局部预测精度与大规模部署的可扩展性。

    

    概率负荷预测在电力系统运行与规划中已得到广泛研究，但客户级和变压器级的预测带来了独特的可扩展性挑战。在这些层级上，负荷不确定性受到客户行为、天气以及混合负荷构成的强烈影响，使得单一共享模型难以捕捉异质化的模式。为每个负荷使用独立的概率模型虽然可以提升局部精度，但在大规模应用时，其训练、存储、更新和验证的成本都十分高昂。为应对这一挑战，我们开发了一个可扩展的、客户感知的预测框架，该框架通过一个共享模型学习共同的需求行为，同时仅对一小部分紧凑的参数进行适配。所提出的设计并不为每个负荷使用独立模型，也不将每个负荷分配给某个专用模型，而是学习一个小型的低维适配组件库，并允许每个负荷根据其自身（特征）来组合这些组件……

    arXiv:2610.08538v1 Announce Type: cross  Abstract: Probabilistic load forecasting has been widely studied for power-system operation and planning, but customer- and transformer-level forecasting introduces a distinct scalability challenge. At these levels, load uncertainty is strongly affected by customer behavior, weather, and mixed load composition, making it difficult for a single shared model to capture heterogeneous patterns. Using separate probabilistic models can improve local accuracy, but becomes costly to train, store, update, and validate at scale. To address this challenge, we develop a scalable customer-aware forecasting framework that learns common demand behavior through a shared model while adapting only a compact subset of parameters. Rather than using an independent model for each load or assigning each load to a specialized model, the proposed design learns a small bank of low-dimensional adaptation components and allows each load to combine them according to its for
    
[^45]: FlowCF：基于流匹配的混合类型表格数据稀疏反事实解释方法

    FlowCF: Sparse Counterfactual Explanations for Mixed-Type Tabular Data using Flow Matching

    [https://arxiv.org/abs/2610.08537](https://arxiv.org/abs/2610.08537)

    FlowCF提出了一种基于流匹配的模型无关生成方法，通过新颖的混合流算子和门控网络，为混合类型表格数据生成具有稀疏性的反事实解释。

    

    在可解释人工智能（XAI）领域，反事实（CF）解释通过建议对输入进行哪些修改能够带来更有利的结果，从而解释模型的决策。为了在实际中发挥作用，这样的解释应该只改变少量特征，并且尽可能少地改变它们，这些性质被称为稀疏性和接近性。我们观察到，现有方法在这一方面仍然存在局限，尤其是针对数值特征，无论这些方法是模型无关且经过摊销训练的，还是基于梯度且完全访问模型的。在本文中，我们提出FlowCF，这是一种模型无关的生成方法，它将反事实样本的生成框架化为从事实样本到目标类别的稀疏传输问题。我们利用流匹配来解决这一传输问题，并通过一种新颖的混合流算子将其扩展到混合特征类型，同时利用所得的几何结构，通过一个最小化传输所改变特征数量的门控网络来优化稀疏性。

    arXiv:2610.08537v1 Announce Type: cross  Abstract: In the field of Explainable AI (XAI), counterfactual (CF) explanations interpret a model's decision by suggesting the changes to the input that would lead to a more favourable outcome. To be useful in practice, such an explanation should change few features and change them as little as possible, properties known as sparsity and proximity. We observe that existing methods remain limited in this respect, especially for numerical features, whether they are model-agnostic and amortised, or gradient-based with full access to the model. In this paper, we propose FlowCF, a model-agnostic generative method that frames CF generation as sparse transport from the factual to the target class. We solve this transport with flow matching, which we extend to mixed feature types with a novel mixed flow operator, and exploit the resulting geometry to optimise for sparsity through a gating network that minimises the number of features the transport chang
    
[^46]: MedCORE：基于临床标准的可解释医学图像诊断推理框架

    MedCORE: Criteria-Grounded Clinical Reasoning for Interpretable Medical Image Diagnosis

    [https://arxiv.org/abs/2610.08528](https://arxiv.org/abs/2610.08528)

    MedCORE提出了一种将临床诊断推理融入视觉-语言模型的结构化框架，通过标准分解、空间定位、多尺度证据编码和图注意力网络精炼，实现了可解释、透明的医学图像诊断。

    

    临床诊断本质上是一个结构化的推理过程，然而现有的深度学习模型往往绕过这一结构，直接将图像特征映射到疾病标签，而没有显式地审视临床医生系统评估的形态学和纹理学标准。这限制了诊断的透明度，并可能危及安全的临床部署。我们提出了MedCORE（医学标准导向推理与证据），这是一个在视觉-语言架构中将临床推理操作化的结构化诊断框架。对于每张输入图像，MedCORE将诊断过程分解为临床定义的标准，将每个标准空间定位到与诊断相关的图像区域，通过捕获宏观结构和微观纹理病理特征的多尺度表示对证据进行编码，并使用图注意力网络显式地精炼标准表示……

    arXiv:2610.08528v1 Announce Type: cross  Abstract: Clinical diagnosis is inherently a structured reasoning process, yet existing deep learning models often bypass this structure by mapping image features directly to disease labels without explicitly interrogating the morphological and textural criteria that clinicians systematically evaluate. This limits diagnostic transparency and may compromise safe clinical deployment. We present MedCORE (Medical Criteria-Oriented Reasoning and Evidence), a structured diagnostic framework that operationalizes clinical reasoning within a vision-language architecture. For each input image, MedCORE decomposes the diagnostic process into clinically defined criteria, spatially localizes each criterion to diagnostically relevant image regions, encodes evidence through multi-scale representations that capture macro-structural and micro-textural pathological characteristics, and refines criterion representations using a Graph Attention Network that explicit
    
[^47]: 编码智能体的自我纠正应承载多少证据？面向自蒸馏的自适应Dirichlet证据

    How Much Evidence Should a Coding Agent's Self-Correction Carry? Adaptive Dirichlet Evidence for Self-Distillation

    [https://arxiv.org/abs/2610.08514](https://arxiv.org/abs/2610.08514)

    该论文提出有效证据自蒸馏（EESD），利用Dirichlet后验将执行相关性与证据量分开建模，为编码智能体的自我纠正生成经不确定性惩罚的学习权重，在八个观测下相比固定质量方法显著降低了未来结果的NLL。

    

    执行反馈使编码智能体能够修改程序并从自身的纠正中学习。一次纠正的学习权重应当同时反映其执行所支持的转移，以及该支持背后证据的数量。我们提出了有效证据自蒸馏，它将这两个量分开表示：归一化的执行相关性决定相对转移支持和有效伪计数质量；随后由Dirichlet后验产生一个经不确定性惩罚的权重，用于基于KL锚定的纠正学习。在对称先验下，改变质量可保持类别排序，且有效质量得到的监督系数以其匹配的固定质量对应值为上界。在四个模型-领域的历史扫描实验中，将可见观测次数从一增加到八，可使未来结果的NLL降低55.0–59.3%。在八次观测时，有效质量在全部四个对比中均取得了比固定质量更低的NLL。

    arXiv:2610.08514v1 Announce Type: new  Abstract: Execution feedback lets coding agents revise programs and learn from their own corrections. A correction's learning weight should reflect both the transitions supported by its executions and the amount of evidence behind that support. We introduce Effective-Evidence Self-Distillation (EESD), which represents these quantities separately. Normalized execution relevance determines relative transition support and an effective pseudo-count mass; a Dirichlet posterior then produces an uncertainty-penalized weight for KL-anchored correction learning. Under a symmetric prior, changing mass preserves category ordering, and effective mass yields a supervised coefficient bounded by its matched fixed-mass counterpart. Across four model-domain history sweeps, increasing visible observations from one to eight reduces future-outcome NLL by 55.0-59.3%. At eight observations, effective mass achieves lower NLL than fixed mass in all four comparisons. In t
    
[^48]: Wiki-Talkie：基于真实世界讨论的角色化智能体多语言基准测试

    Wiki-Talkie: Multilingual Benchmarking of Persona-Based Agents on Real-World Discussions

    [https://arxiv.org/abs/2610.08513](https://arxiv.org/abs/2610.08513)

    该论文提出了 Wiki-Talkie——首个基于维基百科讨论页真实对话、涵盖五种语言并配以源自真实用户社区画像的多语言基准数据集，用于评估角色化LLM智能体模拟人类交互的行为保真度。

    

    大型语言模型（LLM）越来越多地作为自主智能体被部署在社交环境中，这使得研究它们忠实模拟人类交互的能力变得至关重要。其中的核心在于将智能体锚定在真实的用户画像上，然而现有数据集依赖于虚构的人物画像，且仅涵盖少数几种语言，缺乏评估跨不同人群行为保真度所需的经验基础。我们提出了 Wiki-Talkie，这是一个多语言数据集，包含来自维基百科讨论页的真实对话，涵盖分属两个语系的五种语言：日耳曼语系（德语、英语）和罗曼语系（西班牙语、法语、意大利语），并配以从真实用户社区中提取的人物画像，这些画像包含社会人口属性、自我描述以及基于实际行为的交互特征。利用 Wiki-Talkie，我们在下一轮回复生成任务上，针对多种人物画像条件化策略对智能体的交互行为进行了评估。

    arXiv:2610.08513v1 Announce Type: cross  Abstract: LLMs are increasingly deployed as autonomous agents in social environments, making it critical to study their ability to faithfully simulate human interactions. Central to this is grounding agents in realistic user personas, yet existing datasets rely on fictional personas and are limited to a handful of languages, lacking the empirical grounding necessary to evaluate behavioral fidelity across diverse populations. We introduce Wiki-Talkie, a multilingual dataset of real-world conversations from Wikipedia Talk pages across five languages spanning two language families: Germanic (German, English) and Romance (Spanish, French, Italian), paired with personas derived from real user communities and encompassing sociodemographic attributes, self-descriptions, and behaviorally grounded interaction traits. Using Wiki-Talkie, we evaluate agent interactional behavior on a next-turn generation task across various persona conditioning strategies. 
    
[^49]: 面向准周期生理信号转换的圆柱测地线流匹配

    Cylindrical Geodesic Flow Matching for Quasiperiodic Physiological Signal Transformation

    [https://arxiv.org/abs/2610.08510](https://arxiv.org/abs/2610.08510)

    提出圆柱测地线流匹配方法，在圆柱面上显式建模圆形相位与严格正振幅的几何结构，从而实现配对准周期心血管波形之间的信号转换，克服了端点监督回归和标准仿射流匹配路径无法刻画相位—振幅结构的局限。

    

    准周期生理波形之间的配对转换（即从源信号中恢复目标振荡信号）是解读来自放置于身体不同部位的可穿戴设备所采集的心血管信号的核心任务。这类从源到目标的映射具有内在的几何结构：相位在周期上环绕，必须被视为圆形变量；振幅严格为正；且逐拍对齐会在不同心动周期和不同受试者之间发生不可预测的漂移。尽管深度神经网络已被用于相位估计和复值信号建模，但先前的工作并未显式地学习配对信号之间的相位传输。因此，无论是端点监督回归还是流匹配中使用的标准仿射路径，都无法刻画这种相位—振幅结构。我们针对配对心血管波形转换提出了“圆柱测地线流匹配”方法。

    arXiv:2610.08510v1 Announce Type: new  Abstract: Paired translation between quasiperiodic physiological waveforms (i.e., recovering a target oscillatory signal from the source) is central to the interpretation of cardiovascular signals derived from wearables placed at different body locations. This source-to-target mapping in these problems carries inherent geometric structure: the phase wraps around the cycle and must be treated as a circular variable, the amplitude remains strictly positive, and the beat-to-beat alignment can drift unpredictably across cycles and subjects. While deep neural networks have been used for phase estimation and complex-valued signal modeling, prior work does not explicitly learn phase transport between paired signals. Consequently, neither endpoint-supervised regression nor the standard affine path used in flow matching accounts for this phase--amplitude structure. We introduce \emph{cylindrical geodesic flow matching} for paired cardiovascular waveform tr
    
[^50]: X-OPM：面向增强鲁棒性的可解释自动数字片上功耗建模

    X-OPM: Explainable Automatic Digital On-Chip Power Modeling for Enhanced Robustness

    [https://arxiv.org/abs/2610.08502](https://arxiv.org/abs/2610.08502)

    X-OPM基于同步数字VLSI电路设计原理，提出了一个可解释的自动片上功耗建模框架，通过树模型捕捉特征交互并用线性模型进行预测，结合人在回路的工作流程，在商用C906向量处理器上实现了更鲁棒、可泛化且低开销的功耗预测。

    

    主动式功耗管理系统通过运行时功耗预测和功耗感知调度来降低处理器的动态功耗。准确、稳定且低开销的数字片上功耗计对于提升预测质量至关重要。近期研究探索了多种建模方法，包括使用线性模型、决策树和多层感知机（MLP）来构建片上功耗计。然而，当前大多数方法采用端到端方式训练模型，而未分析特征的物理可解释性，这影响了模型对未见工作负载的泛化能力。基于同步数字VLSI电路的设计原理，X-OPM引入了一个鲁棒的特征工程框架，利用基于树的模型来捕捉特征交互，并采用线性模型进行预测。该框架还融合了人在回路的工作流程，以平衡模型精度与建模工作量。该方法在商用C906向量处理器上进行了评估。

    arXiv:2610.08502v1 Announce Type: cross  Abstract: Proactive power management systems reduce processor dynamic power through runtime power prediction and power-aware scheduling. Accurate, stable and low-overhead digital on-chip power meters (OPMs) are crucial for improving the prediction quality. Recent studies have explored various modeling methods, including using linear models, decision trees, and multi-layer perceptrons (MLPs) to construct OPMs. However, most current approaches train models end-to-end without analyzing the physical interpretability of features, affecting their ability to generalize to unseen workloads. Grounded in the design principles of synchronous digital VLSI circuits, X-OPM introduces a robust feature engineering framework that uses tree-based models to capture feature interactions and linear models for prediction. It also incorporates a human-in-the-loop workflow to balance model accuracy against modeling effort. Evaluated on a commercial C906 vector processo
    
[^51]: 语言模型对抑郁症的评分更多反映的是评分者而非患者

    Language-model ratings of depression reflect the rater more than the patient

    [https://arxiv.org/abs/2610.08501](https://arxiv.org/abs/2610.08501)

    该研究通过对880个语言模型评分者的预注册实验发现，语言模型对抑郁症的评分更多反映评分模型自身的差异（解释30.0%的评分方差）而非患者的真实症状差异（仅10.5%），即使两个高精度模型平均也会对40%的参与者的筛查结果产生分歧。

    

    抑郁症没有可用于诊断的血液检测。语言模型有望提供不知疲倦、一致的评估，但准确的评分者之间是否会在对个体的判断上产生分歧？我们预先注册了880个语言模型评分者，将11个开源模型与提示词及评分方式的选择进行交叉组合，并将其应用于189次访谈，以八项患者健康问卷（PHQ-8）作为参照。模型选择解释了症状总评分方差的30.0%，而参与者的稳定个体差异仅解释10.5%。随机抽取的两个受试者工作特征曲线下面积（AUC）≥0.70的评分者，平均对40%的参与者的筛查决策不一致。平均而言，评分偏高程度决定了有多少人被标记为阳性，但能力相当的评分者对大约五分之一的参与者做出了不同的选择。对86次新访谈进行的锁定分析再现了主要的预注册研究结果。利用40名有标签参与者进行的探索性重新校准将准确率从约60%提高到75%，并将评分者间的分歧减半。

    arXiv:2610.08501v1 Announce Type: cross  Abstract: Depression has no diagnostic blood test. Language models promise tireless, consistent assessment, but can accurate raters disagree about individuals? We pre-registered 880 language-model raters, crossing 11 open models with prompting and scoring choices, and applied them to 189 interviews against the eight-item Patient Health Questionnaire. Model choice explained 30.0% of summed-symptom score variance, stable participant differences 10.5%. Two randomly drawn raters with area under the receiver operating characteristic curve (AUC) >= 0.70 disagreed on screening decisions for 40% of participants, on average. Average over-rating governed how many were flagged, yet equal-capacity raters chose differently for about one participant in five. A locked analysis of 86 new interviews reproduced the main pre-registered findings. Exploratory recalibration with 40 labelled participants raised accuracy from about 60% to 75% and halved disagreement, l
    
[^52]: Knee3DVLM：面向膝关节MRI综合评估的双序列全体积视觉-语言建模

    Knee3DVLM: Dual-Sequence Full-Volume Vision-Language Modeling for Comprehensive Knee MRI Assessment

    [https://arxiv.org/abs/2610.08482](https://arxiv.org/abs/2610.08482)

    Knee3DVLM提出了一种序列感知的视觉-语言模型，首次联合利用全体积DESS与液体敏感TSE双序列膝关节MRI，预测57个基于MOAKS评分的解剖学分辨二分类诊断目标，实现了三种配置中最优的综合膝关节MRI结构化评估性能。

    

    视觉-语言模型（VLM）正日益被应用于三维医学影像，但其在膝关节MRI中的应用仍然有限，尤其是在解读临床实践中常用的互补序列方面。我们提出了Knee3DVLM，这是一种序列感知的视觉-语言模型，利用全体积DESS和对液体敏感的TSE膝关节MRI，预测源自MRI骨关节炎膝关节评分（MOAKS）的57个具有解剖学分辨率的二分类诊断目标，以实现结构化报告。我们使用受试者互不重叠的骨关节炎倡议（Osteoarthritis Initiative）数据划分，评估了仅使用DESS、仅使用TSE以及DESS-TSE配对三种配置。在一个包含1,074项检查的保留测试队列中，融合模型达到了72.98%的平均准确率、71.17%的平衡准确率、78.96%的平均ROC-AUC和78.74%的宏平均ROC-AUC，是三种配置中的最高值。在与已发布的3DReasonKnee队列对齐的第二项多分类分析中，Knee3DVLM的数值表现高于

    arXiv:2610.08482v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are increasingly being applied to three-dimensional medical imaging, but their application to knee MRI remains limited, particularly for interpreting the complementary sequences used in clinical practice. We introduce Knee3DVLM, a sequence-aware VLM that uses full-volume DESS and fluid-sensitive TSE MRI to predict 57 anatomically resolved binary diagnostic targets derived from the MRI Osteoarthritis Knee Score (MOAKS) for structured reporting. We evaluated DESS-only, TSE-only, and paired DESS-TSE configurations using subject-disjoint Osteoarthritis Initiative partitions. In a held-out cohort of 1,074 examinations, the fused model achieved 72.98% average accuracy, 71.17% balanced accuracy, 78.96% mean ROC-AUC, and 78.74% macro ROC-AUC, the highest values among the three configurations. In a secondary multiclass analysis aligned with the released 3DReasonKnee cohort, Knee3DVLM was numerically higher than the
    
[^53]: MetaLearnNCA：基于交互式神经细胞自动机的少样本离线元学习

    MetaLearnNCA: Few-Shot Offline Meta-Learning via Interacting Neural Cellular Automata

    [https://arxiv.org/abs/2610.08479](https://arxiv.org/abs/2610.08479)

    提出去中心化框架MetaLearnNCA，通过Active-NCA与Meta-NCA两种耦合神经细胞自动机的动态交互实现少样本离线元学习，无需测试时反向传播计算梯度，同时保留二维空间几何结构信息。

    

    少样本元学习传统上将任务适应表述为通过展开计算图进行的解析梯度下降，或基于展平的一维特征向量的度量式距离比较，前者会产生高昂的测试时反向传播开销，后者则丢弃了原生的二维空间几何结构。在本工作中，我们提出了MetaLearnNCA，这是一个去中心化框架，通过耦合的神经细胞自动机（NCA）之间的动态交互实现少样本适应，且在推理过程中无需计算解析梯度。MetaLearnNCA将任务适应分解为两个部分：Active-NCA，它基于一个被称为“空间程序”的连续二维空间记忆网格来执行任务推断；以及通过学习得到的Meta-NCA，它作为一个去中心化的细胞优化器，通过在局部邻域之间扩散空间误差残差来动态更新该空间程序。MetaLearnNCA与经典的元学习方法相比具有竞争力。

    arXiv:2610.08479v1 Announce Type: cross  Abstract: Few-shot meta-learning traditionally formulates task adaptation either as analytical gradient descent through unrolled computational graphs or as metric-based distance comparisons over flattened 1D fea- ture vectors, which either incur costly test-time backpropagation or discard native 2D spatial geometry. In this work, we propose METALEARNNCA, a decentralized framework that achieves few-shot adapta- tion through the dynamical interaction of coupled Neural Cellular Automata (NCAs) without computing analytical gradients during inference. MetaLearnNCA decomposes task adaptation into an Active- NCA, which executes task inference conditioned on a continuous 2D spatial memory grid termed the spatial program, and a learned Meta-NCA, which acts as a decentralized cellular optimizer by diffusing spatial error residuals across local neighborhoods to dynamically update this program. METALEARN- NCA is competitive against canonical meta-learners i
    
[^54]: 重新思考跨分词器在线策略蒸馏：从对齐覆盖到监督可靠性

    Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability

    [https://arxiv.org/abs/2610.08448](https://arxiv.org/abs/2610.08448)

    该研究发现跨分词器在线策略蒸馏中扩大对齐覆盖并无必要——严格1:1对齐已覆盖大部分token，仅用共享词表中由学生选择的top-16子集计算反向KL即可媲美完整共享词表方法，表明监督可靠性比对齐覆盖更为关键。

    

    在线策略蒸馏（OPD）利用教师模型的反馈，在学生模型自身生成的文本上进行训练。当教师与学生使用不同的分词器时，比较二者的预测需要在序列层面和词表层面进行对齐。本文研究了扩大这种对齐覆盖范围是否能够改善学习效果。在数学推理和代码生成任务上，针对三个异构的教师-学生模型对的实验表明：尽管词表存在显著差异，严格的1:1对齐组已经覆盖了学生生成的大多数token；在蒸馏前从学生模型采样的回复上，共享词表在严格对齐位置上平均保留了几乎全部的教师与学生概率质量。将反向KL散度限制在每个严格对齐位置上由学生选择的共享词表top-16子集，即可获得与完整共享词表OPD相当的准确率，并优于所评估的跨分词器基线方法。此外，添加均方误差监督……

    arXiv:2610.08448v1 Announce Type: cross  Abstract: On-Policy Distillation (OPD) trains a student on its own generations using teacher feedback. With different tokenizers, comparing teacher and student predictions requires alignment at both sequence and vocabulary levels. In this paper, we examine whether expanding this alignment coverage improves learning. Across three heterogeneous teacher--student pairs on mathematical reasoning and code generation, strict 1:1 groups already cover most student-generated tokens despite substantial vocabulary mismatch. On responses sampled from the students before distillation, the shared vocabulary retains nearly all teacher and student probability mass at strictly aligned positions on average. Restricting reverse KL to a student-selected top-16 subset of the shared vocabulary at each strict position achieves accuracy comparable to full shared-vocabulary OPD, outperforming the evaluated cross-tokenizer baselines. Adding mean squared error supervision 
    
[^55]: AssemState：基于说明书与物理状态引导推理的零样本家具组装

    AssemState: Manual and Physical-State-Guided Reasoning for Zero-shot Furniture Assembly

    [https://arxiv.org/abs/2610.08446](https://arxiv.org/abs/2610.08446)

    提出AssemState零样本框架，通过将组装说明书分解为单部件操作并构建组装树，结合迭代的物理状态反馈与基于仿真的释放测试，实现物理合理性的家具组装3D空间推理。

    

    多模态大语言模型（MLLMs）在视觉理解方面取得了显著进展，但与物理环境相结合的精确3D空间推理仍然困难重重。家具组装不仅需要从图示说明书中恢复步骤级操作，还需要将语义附着关系转化为6D位姿更新，使部件能够与环境和先前已组装的组件进行物理交互。为了研究这一问题，我们提出了AssemState，一个用于说明书与物理状态引导家具组装的零样本框架。该框架首先采用锚点引导的边界组装状态，将说明书页面分解为单部件操作并恢复出组装树；随后，利用迭代的后状态反馈细化来引导连续的SE(3)更新与修正，并通过基于仿真的释放测试验证其物理合理性。实验表明，与（摘要在此处截断）

    arXiv:2610.08446v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have made significant progress in visual understanding, but precise 3D spatial reasoning integrated with physical environment remains difficult. Furniture assembly requires not only recovering step-level operations from diagrammatic manuals, but also translating semantic attachment relations into 6D pose updates that enable parts to physically interact with the environment and previously assembled components. To study this problem, we propose AssemState, a zero-shot framework for manual and physical-state-guided furniture assembly. It firstly employs anchor-guided boundary assembly states to decompose manual pages into single-part operations and recover an assembly-tree. Then, it uses iterative after-state feedback refinement to guide successive (SE(3)) updates and corrections, and validates their physical plausibility through simulation-based release tests. Experiments show that compared with the
    
[^56]: EMHO：基于经验轨迹的具身智能体框架优化

    EMHO: EMbodied Agent Harness Optimization via Experience Traces

    [https://arxiv.org/abs/2610.08432](https://arxiv.org/abs/2610.08432)

    提出自演化框架EMHO，让具身智能体在模型冻结的前提下，通过分析经验轨迹和框架历史直接自我优化外部框架，并以EMHO-Merge解决单一共享框架跨多个子任务优化的权衡问题。

    

    改进具身智能体的研究通常侧重于通过训练来优化底层模型，而控制规划、上下文和工具使用的周边智能体框架（harness）则通常依靠人工工程化设计。我们提出疑问：在稀疏的环境反馈下，这一框架能否直接从经验轨迹中进行自我改进。我们提出具身智能体框架优化方法EMHO，这是一个自演化框架，它在保持具身模型冻结不变的情况下，通过分析执行轨迹和先前的框架历史来迭代地修订其框架。EMHO的优化超越了技能或恢复提示的范畴，能够修改智能体监控进度、使用视觉工具、将观察结果落地以及响应失败的方式。为了用单一框架支持多个子任务，我们进一步引入EMHO-Merge，通过利用回合级别的收益与损失来指导基于证据的框架精化，从而解决跨子任务联合优化单一共享框架时的权衡问题。

    arXiv:2610.08432v1 Announce Type: new  Abstract: Improving embodied agents often focuses on optimizing the underlying model through training, while the surrounding agent harness that controls planning, context, and tool use is typically engineered. We ask whether this harness can instead improve itself directly from experience traces under sparse environmental feedback. We propose EMbodied Agent Harness Optimization (EMHO), a self-evolving framework that keeps the embodied model frozen and iteratively revises its harness by analyzing execution trajectories and prior harness history. EMHO optimizes beyond skills or recovery prompts, modifying how the agent monitors progress, uses vision tools, grounds observations, and responds to failures. To support multiple subtasks with a single harness, we introduce EMHO-Merge, which addresses trade-offs in jointly optimizing a single shared harness across subtasks by using episode-level gains and losses to guide evidence-supported refinement of wh
    
[^57]: NeMo-DCR：面向万亿参数规模可扩展智能体强化学习的比特精确增量压缩权重同步

    NeMo-DCR: Bit-Exact Delta-Compressed Refit for Scalable Agentic RL at Trillion-Parameter Scale

    [https://arxiv.org/abs/2610.08430](https://arxiv.org/abs/2610.08430)

    NeMo-DCR提出了一种比特精确的增量压缩权重同步方法，利用固定仿射映射与残差转换仅传输每步约1%发生变化的权重，将万亿参数模型在训练与推演分离的智能体强化学习中的同步开销从87.5分钟大幅降低，并保证接收端参数与完整稠密同步完全一致。

    

    智能体强化学习将训练与推演解耦，因此每次策略更新必须在下一批次开始前到达推演集群。在两个AWS区域之间传输完整的1万亿（1T）参数检查点进行此类权重同步（refit）需要87.5分钟。对BF16训练的测量显示，每步约有1%的权重会发生存储值的变化。近期的系统利用了这种稀疏性，但在放置、精确性或效率方面存在不足：它们重新实现放置规则、组装完整张量、通过算术运算重建数值，或使用跨集群集合通信，且均无法从同步中途的故障中完全恢复。我们提出了NeMo-DCR（增量压缩权重同步），它只传输变化部分且保证比特精确：接收端获得与稠密完整同步完全相同的参数和缓冲区位。在放置方面，固定的仿射映射将变化从训练分片投影到检查点的规范坐标系中，残差转换覆盖其余变化。

    arXiv:2610.08430v1 Announce Type: cross  Abstract: Agentic reinforcement learning (RL) disaggregates training from rollout, so each policy update must reach the rollout clusters before the next batch. Transferring a full 1T checkpoint for such weight synchronization (refit) takes 87.5 min between two AWS regions. Measurements of BF16 training show that about 1% of weights change their stored values per step. Recent systems exploit this sparsity but fall short on placement, exactness, or efficiency: they reimplement placement rules, assemble full tensors, rebuild values arithmetically, or use a cross-cluster collective, and none fully recovers from mid-refit failures.   We present NeMo-DCR (Delta-Compressed Refit), which sends only changes yet is bit-exact: receivers obtain the same parameter and buffer bits as a dense refit. For placement, fixed affine mappings project changes from training shards into the checkpoint's canonical coordinates, residual conversion covers the other changes
    
[^58]: 知道何时不应回答：潜在欠规范信号的跨域与多轮泛化

    Knowing When Not to Answer: Cross-Domain and Multi-Turn Generalization of Latent Underspecification Signals

    [https://arxiv.org/abs/2610.08413](https://arxiv.org/abs/2610.08413)

    该论文构建了一个带轮次标签的多轮对话不可回答性基准与模拟用户评估框架，发现线性探针所捕捉的“信息缺失”信号能在共享同一不可回答性根源的数据集间稳健跨域迁移（AUROC 0.77–0.97），但不同类型不可回答性的表征边界会受词汇混淆、网络层级与坐标系选择的影响。

    

    大型语言模型经常回答那些无法根据已有信息回答的问题，并且在对话中往往在信息尚不充分时就贸然作答。已有研究表明，“不可回答性”可以从模型隐藏状态中被线性解码出来，但目前尚不清楚哪些形式的不可回答性共享同一表征，以及该信号在对话场景中是否实用。本文贡献了一个带轮次标签的多轮基准数据集（423段对话、1,661个带标签的轮次状态），以及一个配有可回答澄清性问题的模拟用户的评估框架，并结合六个数据集与六个开源权重的大语言模型，系统检验不可回答性探针的泛化边界。结果显示，在共享同类不可回答性根源的数据集之间，探针能够稳健迁移：数学中的信息缺失（AUROC 0.77–0.97），以及文本阅读中的信息缺失（SQuAD 2.0<->MuSiQue，0.77–0.90）。相比之下，针对认识论意义上“已知未知”（known-unknowns）的探针向数学任务迁移效果较差，但这种分离在引入词汇控制后会减弱，并随网络层级和坐标系的选择而变化，因此……（原文摘要在此处截断）

    arXiv:2610.08413v1 Announce Type: cross  Abstract: Large language models routinely answer questions that cannot be answered from the information given, and in dialogue they answer before enough has been said. Unanswerability is linearly decodable from hidden states, but it is unclear which of its forms share a representation and whether the signal is useful in dialogue. We contribute a turn-labeled multi-turn benchmark (423 conversations, 1,661 labeled turn-states) and an evaluation harness with a simulated user who answers clarifying questions, and use them with six datasets and six open-weight LLMs to test how far probes for unanswerability carry. Probes transfer robustly between datasets that share a ground of unanswerability: missing information in math (AUROC 0.77-0.97) and in a passage (SQuAD 2.0<->MuSiQue, 0.77-0.90). Probes for epistemic "known-unknowns" transfer poorly to math, but this separation weakens under lexical controls and changes with layer and coordinate system, so 
    
[^59]: 从失败中学习：一种面向基于大语言模型漏洞分析的失败驱动提示词优化方法

    Learning from Failures: A Failure-Driven Prompt Refinement for LLM-Based Vulnerability Analysis

    [https://arxiv.org/abs/2610.08405](https://arxiv.org/abs/2610.08405)

    本文提出失败驱动提示词优化方法（FDPR），通过分析大语言模型在漏洞分析中的反复失败模式来系统性地改进提示词，实验证明该方法显著提升了基于LLM的漏洞分析可靠性。

    

    大语言模型已成为软件漏洞分析领域颇具前景的工具，但其有效性在很大程度上取决于提示词的设计。现有研究主要使用聚合性能指标来比较各种提示策略，对于模型为何失败以及如何系统地改进提示词所提供的见解有限。我们提出了失败驱动提示词优化方法（FDPR），这是一种通过分析模型反复出现的失败来指导基于证据的提示词改进的方法论。基于Damn Vulnerable Java Application（DVJA），我们识别出反复出现的失败模式，包括误报、漏报、无依据推理和CWE错误分类，并将其转化为有针对性的提示词优化。随后，我们在Juliet测试套件上对优化后的提示词进行评估，并通过跨模型验证来评估其泛化能力。结果表明，失败驱动的优化方法提高了基于大语言模型的漏洞分析的可靠性。

    arXiv:2610.08405v1 Announce Type: cross  Abstract: Large Language Models have emerged as promising tools for software vulnerability analysis, but their effectiveness depends heavily on prompt design. Existing research primarily compares prompting strategies using aggregate performance metrics, providing limited insight into why models fail or how prompts can be improved systematically. We propose Failure-Driven Prompt Refinement (FDPR), a methodology that analyzes recurring model failures to guide evidence-based prompt refinement. Using the Damn Vulnerable Java Application (DVJA), we identify recurring failure modes, including false positives, false negatives, unsupported reasoning, and CWE misclassification, and translate them into targeted prompt refinements. We then evaluate the resulting prompt on the Juliet Test Suite and perform cross-model validation to assess generalizability. The results show that failure-driven refinement improves the reliability of LLM-based vulnerability an
    
[^60]: GeoPID：视觉语言模型中视觉信息的分解与调控

    GeoPID: Decomposing and Steering Visual Information in Vision-Language Models

    [https://arxiv.org/abs/2610.08401](https://arxiv.org/abs/2610.08401)

    GeoPID是一个无需训练的框架，通过几何分解将VLM中的信息划分为冗余、模态独有和协同成分，并在推理时沿视觉独有子空间选择性放大视觉表示，从而有效增强模型的视觉依据能力。

    

    尽管近期的视觉语言模型（VLM）在各类应用中表现出色，但它们往往未能充分利用视觉信息，而是过度依赖文本上下文。在本工作中，我们提出了GeoPID，一个无需训练的框架，从几何角度分析视觉语言模型中的多模态信息。GeoPID通过视觉和文本表示子空间之间的几何关系，将信息分解为冗余、模态独有和协同三个组成部分。通过对22个视觉语言模型和14个基准的大规模分析，我们证实当问题强烈需要视觉依据时，正确的预测表现出更强的视觉独有成分。基于这一几何分析，我们引入了一种有针对性的干预技术，在推理过程中选择性地沿着视觉独有子空间放大视觉表示。由此，模型的视觉依据能力得到了增强。

    arXiv:2610.08401v1 Announce Type: cross  Abstract: While recent vision-language models (VLMs) have shown outstanding performance across diverse applications, they tend to under-use visual information and over-rely on textual context. In this work, we propose \textsc{GeoPID}, a training-free framework that analyzes multimodal information within VLMs from a geometric perspective. \textsc{GeoPID} decomposes information into Redundant, Modality-Unique, and Synergistic components through the geometric relationships between visual and textual representation subspaces. Through an extensive analysis across 22 VLMs and 14 benchmarks, we confirm that correct predictions exhibit stronger vision-unique components when questions strongly require visual grounding. Building on this geometric analysis, we introduce a targeted intervention technique that selectively amplifies visual representations along the vision-unique subspace during inference. As a result, visual grounding capabilities were enhanc
    
[^61]: Atom-JEPA：面向三维原子系统的联合嵌入预测架构

    Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems

    [https://arxiv.org/abs/2610.08400](https://arxiv.org/abs/2610.08400)

    Atom-JEPA是一种自监督预训练框架，通过互补的原子级和子结构级目标从无标签三维原子结构中学习潜在表示，在分子ADMET和量子化学性质预测等下游任务上达到了最先进的性能。

    

    大规模自监督预训练已经重塑了现代机器学习，显著提升了语言和视觉模型在下游任务上的泛化能力。尽管深度学习近年来在原子系统建模方面取得了相当大的进展，但该领域的自监督预训练尚未实现与之相当的下游泛化能力。为解决这一问题，我们提出了Atom-JEPA，这是一个自监督预训练框架，它受联合嵌入预测架构的启发，通过互补的原子级和子结构级目标，从无标签的三维结构中学习潜在表示。我们在大规模分子和晶体数据集上对Atom-JEPA进行预训练，并通过在多样化的下游性质预测任务集上进行微调来评估其迁移性能。Atom-JEPA在分子ADMET和量子化学性质预测任务上取得了最先进的性能，

    arXiv:2610.08400v1 Announce Type: cross  Abstract: Large-scale self-supervised pretraining has reshaped modern machine learning, substantially advancing the ability of language and vision models to generalize across downstream tasks. While deep learning has driven considerable progress in modeling atomistic systems in recent years, self-supervised pretraining in this domain has not yet achieved comparable downstream generalization. To address this, we introduce Atom-JEPA, a self-supervised pretraining framework that learns latent representations from unlabeled 3D structures through complementary atom-level and substructure-level objectives inspired by joint-embedding predictive architectures. We pretrain Atom-JEPA on large-scale molecular and crystalline datasets and evaluate its transfer performance by fine-tuning on a diverse set of downstream property prediction tasks. Atom-JEPA achieves state-of-the-art performance on molecular ADMET and quantum-chemical property prediction tasks, 
    
[^62]: 图上远见：面向知识库问答的超越局部视野推理

    Foresight-over-Graph: Reasoning Beyond Local Horizons for Knowledge Base Question Answering

    [https://arxiv.org/abs/2610.08388](https://arxiv.org/abs/2610.08388)

    提出前瞻感知的证据检索框架FoG，克服LLM图推理中逐跳贪心与束搜索剪枝的短视问题，避免关键证据分支被过早丢弃，提升知识库问答的可靠性。

    

    大语言模型（LLM）在问答任务中已展现出强大能力，但在知识密集型任务上仍频繁出现幻觉问题。知识图谱（KG）为LLM提供了结构化、可解释且可更新的事实依据，使其成为实现可靠推理的极具前景的外部知识来源。然而，现有的LLM引导图推理方法在证据检索过程中通常依赖逐跳贪心或束搜索式的剪枝策略。这类局部决策过程本质上具有短视性：在源头附近看似微弱的证据，只有在探索更深的图上下文后才可能变得至关重要，这导致对回答至关重要的分支被过早丢弃，且推理链难以恢复。为解决这一局限，我们提出了Foresight-over-Graph（FoG），一种面向知识库问答（KBQA）的前瞻感知证据检索框架。FoG迭代地构建一个……

    arXiv:2610.08388v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong capabilities in question answering, yet they still frequently suffer from hallucinations on knowledge-intensive tasks. Knowledge graphs (KGs) provide LLMs with structured, interpretable, and updatable factual grounding, making them a promising external knowledge source for reliable reasoning. However, existing LLM-guided graph reasoning methods typically rely on hop-wise greedy or beam-style pruning during evidence retrieval. Such local decision processes are inherently myopic: evidence that appears weak near the source may become crucial only after deeper graph context is explored, causing answer-critical branches to be discarded prematurely and making the reasoning chain difficult to recover. To address this limitation, we propose Foresight-over-Graph (FoG), a foresight-aware evidence retrieval framework for knowledge base question answering (KBQA). FoG iteratively constructs a qu
    
[^63]: 通过AI驱动的多目标优化加速PLGA原位成型储库的开发

    Accelerating the Development of PLGA In Situ Forming Depots Through AI-Driven Multi-Objective Optimization

    [https://arxiv.org/abs/2610.08368](https://arxiv.org/abs/2610.08368)

    该研究将Corbion的PURASORB聚合物库与Intrepid Labs的AI算法ANDROMEDA 1结合，仅用约15周和181个处方即完成多目标优化，成功筛选出4个满足黏度与可注射性要求且具有差异化30天释放曲线的治疗性多肽PLGA原位成型储库处方，显著加速了长效注射制剂的开发进程。

    

    开发长效注射制剂需要同时优化载药量、释放动力学、黏度、可注射性、稳定性等多个目标。为了在这一多维空间中高效探索，Corbion与Intrepid将Corbion丰富多样的PURASORB可生物降解聚合物库与Intrepid Labs的专有人工智能算法（ANDROMEDA 1）相结合，为一种治疗性多肽开发原位成型储库。在大约15周内，研究共制备并表征了181个独特处方，载药量范围为6–12% w/w，通过广泛的设计空间映射和有针对性的多目标优化，最终在6%、9%和12% w/w载药量下筛选出四个候选先导处方。这些处方均满足预先设定的黏度和可注射性标准，同时呈现出各具特色的30天体外释放曲线。该研究评估了涵盖宽分子量范围的多种聚合物，包括市售的PURASORB g（摘要在此处被截断）

    arXiv:2610.08368v1 Announce Type: cross  Abstract: Developing long-acting injectable formulations requires the simultaneous optimization of drug loading, release kinetics, viscosity, injectability, stability and other objectives. To navigate this multidimensional space, Corbion and Intrepid combined Corbion's diverse PURASORB bioresorbable polymer library with Intrepid Labs' proprietary AI algorithm (ANDROMEDA 1) to develop in situ forming depots for a therapeutic peptide. Over approximately 15 weeks, 181 unique formulations spanning drug loadings of 6-12% w/w were prepared and characterized through broad design-space mapping and targeted multi-objective optimization. Four lead candidate formulations were identified at 6%, 9%, and 12% w/w drug loading. Each met the predefined viscosity and injectability criteria while providing distinct 30-day in vitro release profiles. The study evaluated polymers spanning a broad range of molecular weights, including commercially available PURASORB g
    
[^64]: Transect：为长程LLM智能体评估保留可观测性

    Transect: Retaining Observability for Long-Horizon LLM Agent Evaluations

    [https://arxiv.org/abs/2610.08364](https://arxiv.org/abs/2610.08364)

    Transect是一个基于Inspect Scout的开源工具，通过可复用的评估族配置和结构化分析流程，让评估者在分析长程LLM智能体动辄数百页的运行记录时保留可观测性，兼顾语言模型辅助分析的效率与评估结果的可重复性、可审计性。

    

    前沿AI评估越来越多地采用开放式的、具智能体特性的长程任务，其运行记录可能长达数百页，包含来自复杂多智能体网络的输出与动作。因此，“可观测性包络”——即评估者能够对智能体行为可靠推断的范围——正在不断缩小。语言模型助手虽然可以帮助分类和解释智能体行为，但也给人类评估者带来了显著的分析自由度，威胁到基于语言模型的记录分析的可重复性与可审计性。Transect是一个构建于Inspect Scout之上的开源软件包，旨在帮助评估者理解长程智能体运行的展开过程、识别值得深入调查的行为，并对照原始记录验证解释。用户可在可复用的评估族配置中指定任务上下文与行为词汇表，而评判模型和分析设置则单独提供。Transect的可导航报告……

    arXiv:2610.08364v1 Announce Type: new  Abstract: Frontier AI evaluations increasingly use open-ended, agentic, long-horizon tasks whose transcripts can span hundreds of pages of outputs and actions from complex multi-agent networks. The observability envelop-the range of what evaluators can reliably infer about an agent's behaviours-is therefore narrowing. Language model assistants can help classify and interpret agent behaviour but also afford human evaluators significant analytical degrees of freedom, threatening the reproducibility and auditability of language-model-based transcript analysis. Transect is an open source package built on Inspect Scout to help evaluators understand how a long agent run unfolded, identify behaviour worth investigating, and check interpretations against the transcript. Users specify task context and behavioural vocabulary in a reusable evaluation-family configuration, with judge models and analysis settings supplied separately. Transect's navigable repor
    
[^65]: 海事领域可解释的故障预测与预防

    Explainable Failure Prediction and Prevention in Maritime

    [https://arxiv.org/abs/2610.08363](https://arxiv.org/abs/2610.08363)

    本章提出了一种将数据采集、时间序列预测、异常检测、风险评估、决策制定和可解释AI相集成的闭环概念架构，以实现海事系统中可信且可解释的故障预测与预防。

    

    海事系统在高度动态的环境中运行，意外的设备故障可能会危及安全性、可靠性和运营效率。人工智能（AI）、机器学习、数字孪生和预测性维护领域的最新进展使得主动式的故障预测与预防成为可能。然而，在安全至关重要的海事应用中，确保可信且可解释的决策仍然是一项重大挑战。本章回顾了海事系统中可解释故障预测与预防所需的关键AI技术，并提出了一种能够支持自主或人在回路纠正措施的概念架构。该架构将数据采集、时间序列预测、异常检测、风险评估、决策制定和可解释AI集成到一个闭环框架中。围绕该架构的各个组成部分，本章对相关海事研究进行了回顾和讨论。

    arXiv:2610.08363v1 Announce Type: new  Abstract: Maritime systems operate in highly dynamic environments where unexpected equipment failures can compromise safety, reliability, and operational efficiency. Recent advances in artificial intelligence (AI), machine learning, digital twins, and predictive maintenance enable proactive failure prediction and prevention. However, ensuring trustworthy and explainable decision-making remains a major challenge in safety-critical maritime applications. This chapter reviews key AI technologies required for explainable failure prediction and prevention in maritime systems and presents a conceptual architecture capable of supporting autonomous or human-in-the-loop corrective actions. This architecture integrates data acquisition, time-series forecasting, anomaly detection, risk assessment, decision-making, and explainable AI into a closed-loop framework. With reference to the architectural components, a review and discussion of relevant maritime stud
    
[^66]: 通过单次前向量化器对齐重校准实现量化视觉Transformer的测试时自适应

    Test-Time Adaptation of Quantized ViTs via Single-Pass Quantizer-Aligned Recalibration

    [https://arxiv.org/abs/2610.08358](https://arxiv.org/abs/2610.08358)

    提出QuAR，一种专为量化视觉Transformer设计的单次前向传播、无需反向传播的测试时自适应方法，直接针对分布偏移下激活值扭曲冻结量化器码分布这一量化特有失效模式，从而恢复精度。

    

    后训练量化是将视觉Transformer（ViTs）适配到边缘计算与内存预算限制的标准途径，然而量化模型在分布偏移下会变得格外脆弱。测试时自适应（TTA）能够在无需标签的情况下应对此类偏移，但现有大多数方法与量化推理的约束条件匹配不佳。主流的TTA方法通过反向传播来恢复精度，而无反向传播的方法通常仍会因额外的前向传播或参数更新而产生开销，轻量级的特征级或logit级方法则只能恢复部分精度损失。纵观这些方法，有一种会放大精度下降的量化特有失效模式并未被直接针对：在分布偏移下，激活值会以不同的方式占据冻结量化器的校准范围，从而扭曲其码分布。我们提出了量化器对齐重校准（QuAR），这是一种专为量化ViTs量身定制的单次前向传播TTA方法，它既不使用反向传播，……

    arXiv:2610.08358v1 Announce Type: cross  Abstract: Post-training quantization is a standard route to fitting vision transformers (ViTs) into edge compute and memory budgets, yet quantized models become especially brittle under distribution shift. Test-time adaptation (TTA) addresses such shifts without labels, but most existing approaches are poorly aligned with the constraints of quantized inference. Prevailing TTA methods recover accuracy through backpropagation, while backprop-free methods often still incur overhead from extra forward passes or parameter updates, and lightweight feature- or logit-level methods recover only part of the loss. Across these approaches, a quantization-specific failure mode that amplifies the drop is not directly targeted: under shift, activations occupy frozen quantizers' calibrated ranges differently, distorting their code distribution. We propose Quantizer-Aligned Recalibration (QuAR), a single-pass TTA method tailored to quantized ViTs that neither ba
    
[^67]: 传感器几何作为多通道脑信号的流匹配先验

    Sensor Geometry as a Flow-Matching Prior for Multi-Channel Brain Signals

    [https://arxiv.org/abs/2610.08355](https://arxiv.org/abs/2610.08355)

    该论文提出仅利用脑电电极的几何坐标构建k近邻图，并以图拉普拉斯的Matérn函数作为流匹配模型的源协方差，从而把已知的空间相关结构编码为先验，在不增加任何可学习参数的情况下替代各向同性高斯源，生成空间相干的多通道脑电信号。

    

    流匹配模型从各向同性高斯源出发，这是数据相关性结构事先未知时的标准选择。然而对于多通道脑电记录，其中部分结构是事先已知的：电极位于头部的固定位置，而经由颅骨和头皮的容积传导使得相邻电极以跨被试共享的方式共同变化。尽管如此，现有的EEG生成模型仍然让网络从头学习这种结构。我们将这一结构直接置于源分布中：仅凭传感器坐标构建k近邻图，并取其图拉普拉斯算子的Matérn函数作为源协方差，使流从空间相干的模式出发，而非通道独立的噪声。这一改动不引入任何可学习参数，可配合任意耦合方案与任意漂移网络使用，且在所有数据集上均使用相同的三个超参数。在八个EEG数据集上的（实验结果摘要在此处截断）

    arXiv:2610.08355v1 Announce Type: cross  Abstract: Flow-matching models start from an isotropic Gaussian source, the standard choice when the correlation structure of the data is unknown in advance. For multi-channel brain recordings, however, part of this structure is known in advance. Electrodes sit at fixed positions on the head, and volume conduction through the skull and scalp makes nearby electrodes co-vary in a way that is shared across subjects. Existing EEG generative models nonetheless leave the network to learn this from scratch. We put this structure into the source instead. From the sensor coordinates alone, we build a k-nearest-neighbor graph and take a graph-Mat\'ern function of its Laplacian as the source covariance, so the flow starts from spatially coherent patterns rather than channel-independent noise. The change adds no learned parameters, works with any coupling and any drift network, and uses the same three hyperparameters on every dataset. Across eight EEG datas
    
[^68]: 多少规划才足够？减少世界模型规划中的搜索与计算

    How Much Planning Is Enough? Reducing Search and Computation in World-Model Planning

    [https://arxiv.org/abs/2610.08350](https://arxiv.org/abs/2610.08350)

    提出SufficientPlan部署框架，通过成对序贯预算认证（PSBC）自动寻找并认证每个模型-任务组合所需的最小规划预算，并结合静态上下文复用（SCR）消除迭代规划中的冗余编码，从而在不修改预训练世界模型的前提下大幅降低决策时搜索的计算开销。

    

    视觉世界模型通过决策时的动作搜索实现目标导向控制，但其部署效率往往受限于保守而过大的规划预算。我们证明：即使不与全预算动作达成一致，也能取得有竞争力的任务表现；充足的预算在不同模型-任务组合之间存在差异；且迭代式规划器会反复编码与求解无关的不变上下文。为解决这些低效问题，我们提出了SufficientPlan——一个简单的部署框架，它无需修改预训练世界模型，也无需更新规划器。其“成对序贯预算认证（PSBC）”组件利用成对的闭环证据，在预定义的全性能容差范围内搜索并认证一个缩减的、针对特定模型-任务组合的规划预算。其“静态上下文复用（SCR）”组件则在搜索迭代之间缓存观测与目标表征，同时保留依赖于候选动作的规划与选择过程……

    arXiv:2610.08350v1 Announce Type: cross  Abstract: Visual world models enable goal-directed control through decision-time action search, but their deployment efficiency is often limited by conservatively large planning budgets. We show that competitive task performance can be achieved without agreement with the Full-budget action, that sufficient budgets vary across model--task pairs, and that iterative planners repeatedly encode solve-invariant context. To address these inefficiencies, we propose {SufficientPlan}, a simple deployment framework that requires no modification to pretrained world models or planner updates. Its {Paired Sequential Budget Certification (PSBC)} component uses paired closed-loop evidence to search for and certify a reduced model--task-specific budget within a predefined Full-performance tolerance. Its {Static-Context Reuse (SCR)} component caches observation and goal representations across search iterations while preserving candidate-dependent planning and sel
    
[^69]: 面向自动驾驶黑盒视觉语言模型的可迁移时空一致性对抗攻击

    Transferable Spatial Temporal Coherence Adversarial Attack on Black-Box Vision Language Models for Autonomous Driving

    [https://arxiv.org/abs/2610.08331](https://arxiv.org/abs/2610.08331)

    本文提出一种针对自动驾驶场景中黑盒视觉语言模型的时空一致性对抗攻击方法（STCA），通过字幕引导的帧选择、空间攻击与时序一致性攻击三个阶段，揭示了VLMs对视频时序感知对抗攻击的安全脆弱性。

    

    视觉语言模型（VLMs）被快速集成到敏感系统中，带来了现有研究尚未探索的关键安全漏洞。尽管针对基于图像的模型的对抗攻击鲁棒性已被广泛研究，但VLMs在驾驶场景中对针对视频的时序感知对抗攻击的易感性构成了一种独特且研究不足的威胁。在本文中，我们提出了一种针对自动驾驶场景中VLM模型的视频对抗攻击新方法，称为时空一致性对抗攻击（STCA）。我们的攻击包含三个阶段：模态扩展、空间攻击和STCA攻击。在模态扩展阶段，我们提出了字幕引导的帧选择方法，以确保对抗扰动针对语义上最重要的帧。其次，在空间攻击阶段，我们构建有效的扰动并保持高相似度。然后，该扰动……（摘要原文在此处截断）

    arXiv:2610.08331v1 Announce Type: cross  Abstract: The rapid integration of Vision Language Models (VLMs) into sensitive systems introduces critical safety vulnerabilities that remain unexplored in exist studies. While adversarial attack robustness has been extensively studied for image-based models, the susceptibility of VLMs to temporally-aware adversarial attacks against video in driving context poses a distinct and under examined threat. In this paper, we introduce novel adversarial attack against video targeting VLM models used for autonomous driving scenes named Spatial Temporal Coherence Adversarial Attack (STCA). Our attack comprise from three stages: modalities expansion, Spatial attack, and STCA attack. In modalities expansion, we propose caption-guided frame selection method in order to ensure that adversarial perturbation target the most semantically significant frames. Secondly.In spatial attack, we craft effective perturbation and preserve high similarity. Then the pertur
    
[^70]: MoF：面向黑盒大语言模型个性化的偏好感知混合建模

    MoF: Preference-Aware Mixture Modeling for Black-Box LLM Personalization

    [https://arxiv.org/abs/2610.08330](https://arxiv.org/abs/2610.08330)

    MoF 提出了一种可扩展的黑盒大语言模型个性化框架，通过将用户偏好建模为共享潜在偏好面的组合并进行基于历史的路由，实现了无需额外参数更新即可对未见过的用户进行个性化。

    

    专有的大语言模型（LLMs）在广泛的任务中展现出了卓越的能力，然而使其输出与多样化的用户偏好保持一致仍然具有挑战性。现有的针对黑盒大语言模型的个性化方法通常依赖于用户特定的评分头，导致个性化参数的数量随用户数量线性增长，并且需要对未见过的用户进行额外的适配。为了解决这些局限性，我们提出了面混合模型，这是一个可扩展的黑盒大语言模型个性化框架，它将用户偏好建模为共享潜在偏好面的组合，而非专用的用户特定参数。MoF 通过对共享面头进行基于历史条件的路由来实现个性化，使得训练期间未见过的用户无需额外的参数更新即可实现个性化。在多样的个性化任务中，MoF 提供了更强的个性化性能。

    arXiv:2610.08330v1 Announce Type: new  Abstract: Proprietary Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet aligning their outputs with diverse user preferences remains challenging. Existing personalization approaches for black-box LLMs often rely on user-specific scoring heads, causing the number of personalized parameters to grow linearly with the number of users and requiring additional adaptation for unseen users. To address these limitations, we propose Mixture-of-Facets (MoF), a scalable personalization framework for black-box LLMs that models user preferences as compositions of shared latent preference facets rather than dedicated user-specific parameters. MoF performs personalization through history-conditioned routing over shared facet heads, enabling personalization for users unseen during training without additional parameter updates. Across diverse personalization tasks, MoF delivers stronger personalization performa
    
[^71]: 人工智能辅助的庞加莱猜想形式化

    An AI-Assisted Formalization of the Poincar\'e Conjecture

    [https://arxiv.org/abs/2610.08329](https://arxiv.org/abs/2610.08329)

    本研究通过将数学家制定的证明蓝图与明确的里程碑相结合，完成了庞加莱猜想的AI辅助Lean 4形式化，为几何分析领域未来形式化项目的可复用基础设施奠定了基础。

    

    我们展示了庞加莱猜想的人工智能辅助Lean 4形式化工作。该项目起步时，证明背后的几何分析领域几乎缺乏可复用的形式化基础设施。为了组织这项工作，我们将数学家准备的证明蓝图与明确的里程碑陈述相结合。这些里程碑使智能体能够并行工作，并为数学家提供了清晰的定位点，以便识别障碍并提供有效的数学指导。我们的分析梳理了这一工作流程背后的人工干预与组织决策。该项目为未来形式化项目的可复用基础设施提供了起点；此类基础设施一旦建成，最终有望降低验证几何分析中数学成果的成本。

    arXiv:2610.08329v1 Announce Type: new  Abstract: We present an AI-assisted Lean 4 formalization of the Poincar\'e conjecture. The project began with limited reusable formal infrastructure for the geometric analysis behind the proof. To organize this work, we combined a proof blueprint prepared by mathematicians with explicit milestone statements. These milestones enabled parallel agent work and gave mathematicians clear points to locate blockers and provide effective mathematical guidance. Our analysis identifies the human interventions and organizational choices behind this workflow. The project provides a starting point toward reusable infrastructure for future formalization projects; such infrastructure, once developed, could eventually reduce the cost of verifying mathematical results in geometric analysis.
    
[^72]: MedZERO：通过受控知识积累实现开放式医学推理的自我进化智能体

    MedZERO: Self-Evolving Agents for Open-Ended Medical Reasoning Through Controlled Knowledge Accumulation

    [https://arxiv.org/abs/2610.08327](https://arxiv.org/abs/2610.08327)

    MedZERO 提出了一个自我进化智能体框架，通过“考官”生成前沿医学题目、“推理者”进行基于证据的多轮推理，并借助受控知识积累机制，使大语言模型无需昂贵专家监督即可在开放、难以完全验证的医学推理领域实现自我提升。

    

    大型语言模型（LLMs）在医学问答和临床推理方面已展现出潜力，但其性能提升仍受限于静态的参数化知识和高昂的专家监督成本。自我进化智能体通过让模型经由迭代式任务生成与问题求解来实现自我提升，为这一难题提供了一种有前景的替代方案。然而，现有的自我进化方法大多面向数学和编程等易于验证的领域，在这些领域中，解答可以通过精确答案或可执行程序加以检验。医学推理则与之存在根本差异：它是开放式的、知识密集型的，且通常只能进行部分验证。我们提出了 MedZERO，一个面向开放式医学推理的自我进化框架。MedZERO 将一个负责生成前沿医学“题目-选项”对的“考官”与一个借助外部知识工具、通过基于证据的多轮推理来解题的“推理者”相结合，以支持可靠的……（摘要原文在此处截断）

    arXiv:2610.08327v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promise in medical question answering and clinical reasoning, yet their improvement remains constrained by static parametric knowledge and costly expert supervision. Self-evolving agents offer a promising alternative by enabling models to improve through iterative task generation and problem-solving. However, most existing self-evolving methods are designed for easily verifiable domains such as mathematics and coding, where solutions can be checked by exact answers or executable programs. Medical reasoning is fundamentally different: it is open-ended, knowledge-intensive, and often only partially verifiable. We present MedZERO, a self-evolving framework for open-ended medical reasoning. MedZERO couples an Examiner that generates frontier medical question-option pairs with a Reasoner that solves them through evidence-grounded multi-turn reasoning with external knowledge tools. To support reliable, c
    
[^73]: SCOPE：以语言模型作为策略规划器的可认证定理证明

    SCOPE: Certified Theorem Proving with a Language Model as the Policy Planner

    [https://arxiv.org/abs/2610.08319](https://arxiv.org/abs/2610.08319)

    SCOPE框架通过让小型语言模型负责算子规划、符号引擎执行数值计算、编译器验证证明的自然分工，仅用135M参数模型就在218个多步数值命题上认证了87.6%的证明，大幅超越了消耗数十倍资源的更大规模直接生成模型。

    

    在Lean等证明助手中，生成的证明必须通过机器编译检查，因此评估无需人工打分。直接生成方法在多步数值命题上会失败：只有当证明中的每个内容整数都正确时证明才有效，因此通过率受限于单整数准确率的k次幂。在2,617个参考证明上进行的受控破坏实验证实了这一幂律。SCOPE（状态条件化算子规划与执行）强制执行自然的分工：模型在算子词汇表上进行规划，符号引擎执行数值计算，编译器生成证明。在一个包含218个问题的测试集上，SCOPE使用135M参数的骨干模型认证了191/218（87.6%）的证明；而7B的DeepSeek-Prover-V1.5-RL仅认证了18/218，且消耗了27.5倍的token和37.5倍的运行时间，DeepSeek-Prover-V2-7B在双向对偶测试集上的认证数为零。多步思考每个问题只需6.12个离散决策动作，且不产生

    arXiv:2610.08319v1 Announce Type: new  Abstract: In proof assistants such as Lean, a generated proof must pass machine compilation checks, so evaluation needs no human scoring. Direct generation fails on multi-step numeric propositions: a proof is valid only if every content integer is correct, so the pass rate is bounded by the k-th power of the per-integer accuracy. Controlled corruption across 2,617 reference proofs confirms this power law. SCOPE (State-Conditioned Operator Planning and Execution) enforces the natural division of labor: the model plans over an operator vocabulary, a symbolic engine executes the numerics, and a compiler renders the proof. On a 218-problem suite it certifies 191/218 (87.6%) with a 135M backbone; the 7B DeepSeek-Prover-V1.5-RL certifies 18/218 at 27.5 times the tokens and 37.5 times the wall-clock, and DeepSeek-Prover-V2-7B certifies zero on a bidirectional dual suite. Multi-step thinking costs 6.12 discrete decision actions per problem and produces no
    
[^74]: MARCO：面向蛋白质生成模型的放射性水印

    MARCO: The Radioactive Watermark for Protein Generative Models

    [https://arxiv.org/abs/2610.08316](https://arxiv.org/abs/2610.08316)

    提出了首个专为蛋白质生成模型设计的放射性水印框架MARCO，通过在扩散去噪过程中嵌入双层水印，在冻结原模型参数的前提下同时保护模型知识产权并实现对潜在生物安全滥用的法医溯源。

    

    蛋白质生成模型（PGMs）通过从序列数据设计复杂的3D蛋白质结构，彻底改变了结构生物学领域。然而，这一突破带来了双重用途挑战，使高价值的PGMs面临未经授权的模型提取等经济风险，以及生物危害合成等生物安全威胁。为了缓解这些威胁，我们提出了MARCO（构象水印），这是首个专门为PGMs量身定制的放射性水印框架。MARCO建立了一个双层防御机制，在保护知识产权的同时，确保对潜在生物安全滥用行为实现法医层面的可追溯性。（i）为了保持效率，MARCO通过一个辅助编码器-解码器在扩散逆去噪过程中迭代嵌入水印，使原始PGM参数保持冻结状态，从而具备广泛的兼容性。（ii）为了保持生物物理保真度并最大化鲁棒性，我们采用……（原文摘要在此截断）

    arXiv:2610.08316v1 Announce Type: cross  Abstract: Protein Generative Models (PGMs) have revolutionized structural biology by enabling the design of complex 3D protein structures from sequence data. However, this breakthrough introduces a dual-use challenge, exposing high-value PGMs to economic risks like unauthorized model extraction and biosecurity threats such as biohazard synthesis. To mitigate these threats, we propose \textbf{MARCO} (\textsc{COnformation waterMARk}), the first radioactive watermarking framework specifically tailored for PGMs. MARCO establishes a Dual-Layer defense that simultaneously protects intellectual property and ensures the forensic traceability of potential biosecurity misuses. (i) To preserve efficiency, MARCO iteratively embeds watermarks during diffusion reverse denoising via an auxiliary encoder-decoder, allowing the original PGM parameters to remain frozen for broad compatibility. (ii) To preserve biophysical fidelity and maximize robustness, we emplo
    
[^75]: 标准化陷阱：认证表格基础模型中的联合标签处理

    The Standardization Trap: Certifying Joint Label Processing in Tabular Foundation Models

    [https://arxiv.org/abs/2610.08314](https://arxiv.org/abs/2610.08314)

    该论文揭示了检验表格基础模型是否遵循固定权重机制时存在的“标准化陷阱”问题，并提出两个仅依赖标准化标签处预测的证明方法，能够区分固定权重预测与非线性标签变换两种解释。

    

    线性回归和核平滑为上下文学习提供了可解析的解释：在这两种方法中，特征决定了分配给每个上下文标签的权重。然而，这种固定权重的解释是否适用于预训练的表格基础模型（TFMs）仍不清楚。使用导数来检验这一解释会遇到标准化陷阱：公开的TFM软件包在模型看到标签之前会先对标签进行标准化，而普通导数同时也会反映标准化标签集合之外的行为，导致模型即使在所有预测都与固定权重映射一致的情况下也会显得非线性。我们提出了两个仅依赖于标准化标签处预测的证明方法，能够拒绝两种不同的解释：固定权重预测和独立非线性标签变换之和。在我们评估的五个公开TFM中，我们的证明表明，改变一个上下文标签会改变其他标签影响预测的方式。

    arXiv:2610.08314v1 Announce Type: new  Abstract: Linear regression and kernel smoothing offer tractable explanations of in-context learning: in both, the features determine the weight assigned to each context label. However, whether this fixed-weight account describes pretrained tabular foundation models (TFMs) remains unclear. Testing this account using derivatives runs into a standardization trap: public TFM packages standardize the labels before the model sees them, yet ordinary derivatives also reflect behavior outside the set of standardized labels, making a model appear nonlinear even when every prediction it makes agrees with a fixed-weight map. We propose two certificates that depend only on predictions at standardized labels and can reject two distinct explanations: fixed-weight prediction and sums of independent nonlinear label transformations. Across the five public TFMs that we evaluate, our certificates show that changing one context label alters how other labels influence
    
[^76]: CoDe-LoRA：通过知识巩固与解耦缓解大语言模型持续学习中的正交困境

    CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs via Knowledge Consolidation and Decoupling

    [https://arxiv.org/abs/2610.08312](https://arxiv.org/abs/2610.08312)

    提出无需回放的CoDe-LoRA方法，通过自适应零空间投影和语义路由将学习过程解耦为通用知识巩固与任务特定知识解耦，克服了正交参数隔离阻碍跨任务知识迁移的“正交困境”。

    

    持续学习（CL）对于大语言模型（LLMs）顺序适应不断演变的任务至关重要。为了缓解灾难性遗忘，近期的进展采用带正交投影的低秩适应方法（如O-LoRA）来隔离任务参数。然而，我们揭示了这种严格的几何约束会引发“正交困境”：僵化的参数隔离阻碍了语义相关任务之间共享表示的迁移与积累。在这项工作中，我们提出了一种新的无需回放的方法，称为巩固与解耦LoRA（CoDe-LoRA），用于大语言模型的持续学习。CoDe-LoRA将学习过程解耦为巩固通用知识和解耦任务特定知识两个部分。为实现这一目标，CoDe-LoRA利用自适应零空间投影机制和语义路由来平衡知识积累与任务特定适应。在四个骨干模型和三种持续学习设置上的实验结果……

    arXiv:2610.08312v1 Announce Type: new  Abstract: Continual learning (CL) is essential for Large Language Models (LLMs) to sequentially adapt to evolving tasks. To mitigate catastrophic forgetting, recent advances implement low-rank adaptation with orthogonal projections (e.g., O-LoRA) to isolate task parameters. However, we reveal that such strict geometric constraints trigger an "Orthogonality Dilemma": rigid parameter isolation impedes the transfer and accumulation of shared representations across semantically related tasks. In this work, we propose a new replay-free method, called Consolidation and Decoupling LoRA (CoDe-LoRA), for CL of LLMs. CoDe-LoRA disentangles the learning process into Consolidating Universal Knowledge and Decoupling Task-Specific Knowledge. To achieve this, CoDe-LoRA leverages an adaptive null space projection mechanism and semantic routing to balance knowledge accumulation with task-specific adaptation. Experimental results across four backbones and three CL 
    
[^77]: 利用历史数据缓解自动驾驶车辆遥操作QoS预测中的概念漂移

    Mitigating Concept Drift in QoS Prediction for Teleoperation of Autonomous Vehicles Using Historic Data

    [https://arxiv.org/abs/2610.08297](https://arxiv.org/abs/2610.08297)

    本文提出将历史数据纳入预测流程以缓解遥操作QoS预测中机器学习模型因概念漂移导致的性能退化，并引入关键场景检测指标来专门评估遥操作场景下的预测性能。

    

    遥操作是自动驾驶的备用解决方案，但遥操作的可靠运行需要一定的移动网络资源，而这些资源无法时刻得到保证。因此，引入了预测服务质量这一概念以提高遥操作的韧性。本文基于一次数据测量活动，提出了一个预测框架，用于预测遥操作的两个重要网络关键性能指标：上行数据速率和往返延迟。此外，我们提出一种方法，通过将历史数据纳入预测流程，来缓解基于机器学习的预测模型在处理先前未见过的数据时因概念漂移而导致的性能下降。另外，我们引入了关键场景检测这一指标，专门用于评估遥操作场景下的预测性能。

    arXiv:2610.08297v1 Announce Type: cross  Abstract: Teleoperation serves as the fallback solution to autonomous driving but reliable functions of the teleoperation require a certain amount of mobile network resources, which cannot be guaranteed at all times. Therefore, predictive quality of service (pQoS) is introduced as a concept to increase the resilience of the teleoperation. In this paper, based on a data measurement campaign, we propose a prediction framework to prediction two important network KPIs of teleoperation: uplink data-rate and round-trip latency. Furthermore, we introduce a method to alleviate the performance degradation of machine-learning-based prediction models on previously unseen data due to concept drift by incorporating historic data into the prediction pipeline. Additionally, we introduce the metric of critical scenario detection to evaluate the prediction performance specifically for teleoperation.
    
[^78]: DySCo：面向协同边云大语言模型推理的动态分片与深度同步批处理方法

    DySCo: Dynamic Sharding for Collaborative Edge-Cloud LLM Inference with Depth-Synchronized Batching

    [https://arxiv.org/abs/2610.08268](https://arxiv.org/abs/2610.08268)

    提出DySCo协同运行时系统，通过动态分片、模型感知的层范围执行器dyForward以及深度同步批处理，消除边云协同LLM推理中云端调用的空闲间隙，并解决到达不同模型深度的请求无法常规批处理的难题。

    

    无处不在的智能应用正越来越多地部署在移动和物联网边缘设备上，因此大语言模型（LLM）被越来越多地用于支持这些应用。然而，由于LLM的高资源需求，它们大多部署在云端。逐层边云协同推理使资源受限的边缘设备能够为自身无法完整承载的LLM贡献算力。然而，异构的切分点引入了两个相互耦合的低效问题：其一，边缘侧的执行和通信会在云端调用之间产生空闲间隙；其二，到达模型不同深度的请求无法进行常规批处理。我们提出了DySCo，这是一个协同运行时系统，它将KV缓存保留在本地，并引入了dyForward——一个模型感知的层范围执行器，可以从常驻的模型分片中运行可配置的连续层范围，而无需重新加载权重。针对多边缘服务场景，我们进一步引入了深度同步（注：原文摘要在此处截断）

    arXiv:2610.08268v1 Announce Type: cross  Abstract: Pervasive intelligent applications are increasingly deployed on mobile and Internet of Things (IoT) edge devices. Consequently, Large Language Models (LLMs) are increasingly used to support these applications. Yet, due to their high resource demands, LLMs are mostly deployed in the cloud. Layer-wise edge-cloud inference lets resource-constrained edge devices contribute computation to LLMs they cannot host in full. However, heterogeneous split points introduce two coupled inefficiencies. First, edge execution and communication create idle gaps between cloud invocations. Second, requests arriving at different model depths cannot be conventionally batched. We present DySCo, a collaborative runtime that keeps KV caches local and introduces dyForward, a model-aware layer-range executor that runs configurable contiguous layer ranges from resident model shards without reloading weights. For multi-edge serving settings, we introduce depth-sync
    
[^79]: zkLLMPoT：面向大语言模型的高效零知识训练证明

    zkLLMPoT: Efficient Zero Knowledge Proof of Training for Large Language Models

    [https://arxiv.org/abs/2610.08258](https://arxiv.org/abs/2610.08258)

    zkLLMPoT提出了一种零知识训练证明框架，通过让训练者在审计员选定的挑战序列上证明所提交模型的目标值，将认证成本与训练迭代次数解耦，且不泄露模型权重或私有训练数据。

    

    当模型权重和训练数据均为私有时，审计大语言模型（LLM）训练所声称的结果极具挑战性，而以密码学方式证明完整的训练过程在Transformer规模下成本高得令人望而却步。我们提出了zkLLMPoT，这是一个零知识框架，它通过前向评估而非验证优化轨迹来认证审计员所定义的训练后检查点属性。zkLLMPoT包含两个阶段：1）训练者固定架构并提交模型权重，然后审计员选择挑战序列，防止训练者针对审计数据修改检查点；2）随后训练者证明所提交的模型在这些序列上达到的目标值。这一设计使认证成本与训练迭代次数无关，同时无需泄露模型权重或访问私有训练数据。

    arXiv:2610.08258v1 Announce Type: new  Abstract: Auditing the claimed outcomes of large language model (LLM) training is challenging when model weights and training data are private, while cryptographically proving the full training process is prohibitively expensive at Transformer scale. We present zkLLMPoT, a zero-knowledge framework that certifies auditor-defined properties of a trained checkpoint through forward evaluation rather than verification of its optimization trajectory. zkLLMPoT includes 2 phases: 1) The trainer fixes the architecture and the model weights are committed. Then the auditor selects challenge sequences, preventing the trainer from modifying the checkpoint in response to the audit data. 2) Then the trainer proves the objective value attained by the committed model on those sequences. This formulation makes the certification cost independent of the number of training iterations, without revealing model weights or requiring access to private training data. We bui
    
[^80]: MASC：一种具有潜在构念对齐的多智能体自校准框架，用于心理咨询中的一致性来访者角色扮演

    MASC: A Multi-Agent Self-Calibration Framework with Latent Construct Alignment for Consistent Client Role-Playing in Psychological Counseling

    [https://arxiv.org/abs/2610.08250](https://arxiv.org/abs/2610.08250)

    提出MASC多智能体自校准框架，通过潜在构念对齐与闭环校准机制（构念引导生成、协作精炼、一致性验证和记忆修正），使模拟来访者在长程心理咨询对话中保持心理状态、沟通行为与情绪的一致性。

    

    大语言模型日益被用于模拟来访者，以支持咨询师培训与心理咨询研究，但可靠的模拟要求来访者在长程交互中保持心理上的一致性。现有的角色扮演方法主要依赖静态档案提示，可能表现出人设漂移、不切实际的配合度，或心理状态、沟通行为与情绪之间的不一致。现有的评估方法也缺乏一个能够同时考察稳定来访者特征与动态心理变化的统一测试平台。我们提出了MASC，一个通过潜在构念对齐来实现心理咨询中一致性来访者角色扮演的多智能体自校准框架。MASC将构念引导生成、协作精炼、一致性验证以及基于记忆的修正结合在一个闭环校准循环中，在对话展开的过程中检测并纠正不一致之处。我们进一步引入了CR（原文摘要在此处截断）

    arXiv:2610.08250v1 Announce Type: new  Abstract: Large language models are increasingly used to simulate clients for counselor training and psychological counseling research, but reliable simulation requires clients to remain psychologically coherent across extended interactions. Existing role-playing methods largely rely on static profile prompts and may exhibit persona drift, unrealistic cooperativeness, or inconsistent psychological states, communicative actions, and emotions. Existing evaluations also lack a unified testbed for both stable client characteristics and evolving psychological dynamics. We propose MASC, a Multi-Agent Self-Calibration framework with latent construct alignment for consistent client role-playing in psychological counseling. MASC combines construct-guided generation, collaborative refinement, consistency verification, and memory-based revision in a closed calibration loop that detects and corrects inconsistencies as dialogue unfolds. We further introduce CR
    
[^81]: LeanPlan：基于LLM生成启发式函数与可采纳性证明的最优规划

    LeanPlan: Optimal Planning with LLM-Generated Heuristics and Admissibility Proofs

    [https://arxiv.org/abs/2610.08246](https://arxiv.org/abs/2610.08246)

    LeanPlan是首个利用LLM生成的启发式函数（其可采纳性在Lean 4中经机器验证）以找到最优计划的规划系统，在国际规划竞赛领域上展现出优异的最优规划性能。

    

    前沿大型语言模型（LLM）能够生成启发式函数来引导搜索，在满意规划（satisficing planning，即任何可行计划均可接受）中实现最先进的性能。然而，这些启发式函数无法保证可采纳性，可能导致生成的计划并非最优。我们提出了LeanPlan，这是首个利用LLM生成的启发式函数找到最优计划、且其可采纳性经过机器验证的规划系统。给定领域描述和训练任务，一个智能体循环利用规划器的反馈来迭代改进可复用的领域特定启发式函数、其可采纳性证明以及所需的领域假设。LeanPlan在Lean 4中实现了该启发式函数、其证明以及一个具备机器验证的落地与搜索的高效规划器。我们在国际规划竞赛的十个领域和三个新领域上对LeanPlan进行评估，所用测试任务的对象数量多达训练任务的57倍。使用GPT-5.6 Sol i（摘要在此处截断）

    arXiv:2610.08246v1 Announce Type: new  Abstract: Frontier large language models (LLMs) can generate heuristic functions that guide search to achieve state-of-the-art performance in satisficing planning, where any plan is acceptable. However, these heuristics are not guaranteed to be admissible and can lead to suboptimal plans. We introduce LeanPlan, the first planning system that finds optimal plans with LLM-generated heuristics whose admissibility is machine-checked. Given a domain description and training tasks, an agentic loop uses planner feedback to iteratively improve a reusable domain-specific heuristic, its admissibility proof and the required domain assumptions. LeanPlan implements the heuristic, its proof and an efficient planner with machine-checked grounding and search in Lean 4. We evaluate LeanPlan on ten domains from the International Planning Competition and three new domains, using test tasks with up to 57 times as many objects as the training tasks. With GPT-5.6 Sol i
    
[^82]: 传感器-语言-动作模型

    Sensor-Language-Action Models

    [https://arxiv.org/abs/2610.08244](https://arxiv.org/abs/2610.08244)

    该论文提出传感器-语言-动作（SLA）建模框架，以语言作为感知与行动之间的语义接口，将多模态传感器观测、自然语言和异构动作统一在同一个模型中，并构建了覆盖超11.6万个体、79种传感器模态和60个动作组的大规模基准。

    

    传感器不仅有助于理解世界，也有助于决定下一步该做什么。然而，现有的传感器模型大多止步于感知：它们识别状态或预测结果，而将动作的建模留给特定任务的、通常基于封闭标签空间的方法。我们提出了传感器-语言-动作建模，这是一个在统一模型中将多模态传感器观测、自然语言与动作相连接的框架。SLA 将语言用作感知与行动之间的语义接口，使异构的动作能够被表示、预测和解释，同时始终基于底层的传感器证据。我们构建了一个大规模的 SLA 基准，其中的数据集覆盖超过 116,000 名个体、79 种传感器模态和 60 个动作组，并配备了一个多方面的描述生成流水线，用于对齐用户上下文、传感器动态和动作证据。基于这一框架，我们提出了……（原文摘要在此处不完整）

    arXiv:2610.08244v1 Announce Type: new  Abstract: Sensors are useful not only for understanding the world but also for deciding what to do next. Existing sensor models however largely stop at perception: they recognize states or predict outcomes, leaving actions modeled separately through task-specific and often closed label spaces. We introduce Sensor-Language-Action (SLA) modeling, a framework that connects multimodal sensor observations, natural language, and actions within a unified model. SLA uses language as a semantic interface between sensing and acting, allowing heterogeneous actions to be represented, predicted, and explained while remaining grounded in the underlying sensor evidence. We build a large-scale SLA benchmark consisting of datasets that span more than 116,000 individuals, 79 sensor modalities, and 60 action groups, together with a multi-faceted captioning pipeline that aligns user context, sensor dynamics, and action evidence. Building on this framework, we present
    
[^83]: OSFP4：NVFP4量化的对角平滑与块缩放联合优化

    OSFP4: Joint Optimization of Diagonal Smoothing and Block Scales for NVFP4 Quantization

    [https://arxiv.org/abs/2610.08231](https://arxiv.org/abs/2610.08231)

    OSFP4 提出了一种新型 NVFP4 量化方案，通过分析乘性抖动 FP4 量化器，实现对对角平滑矩阵元素与块缩放因子的联合优化，以最小化矩阵乘积量化误差，从而在 LLM 推理量化中取得最高平均精度。

    

    NVFP4 是一种对大语言模型（LLM）推理颇具吸引力的数据类型，兼具紧凑的存储和原生的张量核加速能力。然而，要在使用 NVFP4 时保持精度，需要精细的量化处理。在本工作中，我们开发了一种名为 OSFP4（NVFP4 的优化平滑与缩放）的新型量化方案。对于每个线性投影，该方案使用一个对角平滑矩阵，其元素经过优化以在 NVFP4 下最小化矩阵乘积的平方量化误差，并考虑了所采用的舍入过程（可以是舍入到最近值，也可以是 GPTQ 风格的逐次干扰消除）。这需要对平滑矩阵元素以及块缩放因子进行联合优化，而通过分析乘性抖动 FP4 量化器（而非固定的确定性量化器），可以促成这一联合优化。实验表明，在相应的量化设置下，OSFP4 在所有被评估的竞争方法中取得了最高的平均精度。

    arXiv:2610.08231v1 Announce Type: new  Abstract: NVFP4 is an attractive datatype for large language model (LLM) inference, offering compact storage and native tensor-core acceleration. However, preserving accuracy using NVFP4 requires careful quantization. In this work we develop a novel quantization scheme called Optimized Smoothing and Scaling for NVFP4 (OSFP4). For each linear projection it uses a diagonal smoothing matrix whose entries are optimized to minimize the squared matrix-product quantization error under NVFP4, taking into account the rounding procedure that is used (either round-to-nearest, or GPTQ-style successive interference cancellation). This requires performing joint optimization on the smoothing entries as well as the block scales, which is facilitated by analyzing a multiplicative-dither FP4 quantizer instead of the fixed deterministic one. Experiments show that OSFP4 achieves the highest average accuracy among the evaluated competitors in the corresponding quantiz
    
[^84]: 神经解码中上下文先验下的置信度排序反转

    Confidence-Ordering Reversal under Contextual Priors in Neural Decoding

    [https://arxiv.org/abs/2610.08229](https://arxiv.org/abs/2610.08229)

    论文揭示神经解码中上下文先验引发的“置信度排序反转”现象：当正确候选者初始排名较低时，融合后更大的置信度差距反而预示修复可能性更低，导致初始排名20开外的错误占融合后错误的46.6%。

    

    上下文先验通过重塑候选分数来改进从神经信号到语言的解码。然而，置信度是从同一重塑后的分数中读取的，因此先验遗留下的错误可能在准确率没有任何变化来揭示它们的情况下变得更加自信。我们研究先验如何在MEG-MASC和MOUS数据集的语音检索中塑造置信度，使用局部解码分数、通过加性浅融合结合的上下文先验，以及融合后的前两名差距作为置信度。在最初错误的预测中，我们发现了一种置信度排序反转现象：当正确候选者最初位于局部排名靠前位置时，更大的差距使修复更有可能发生；但当正确候选者最初排名较低时，更大的差距反而使修复更不可能。在MEG-MASC上，汇总的判断正确性AUROC为0.87，但区分修复与残留错误的AUROC从初始排名2-3时的0.70降至排名21-50时的0.39。初始排名超过20（位于反转区域内）的错误占融合后所有错误的46.6%。我们提出……（原文摘要在此处截断）

    arXiv:2610.08229v1 Announce Type: new  Abstract: Contextual priors improve neural-to-language decoding by reshaping candidate scores. However, confidence is read from the same reshaped scores, so the errors a prior leaves behind can become more confident with no change in accuracy to reveal it. We study how a prior shapes confidence in speech retrieval on MEG-MASC and MOUS using local decoding scores, a contextual prior combined by additive shallow fusion, and the fused top-two margin as confidence. Among initially incorrect predictions, we find a confidence-ordering reversal: a larger margin makes a repair more likely when the correct candidate starts near the top of the local ranking, but less likely when it starts lower. On MEG-MASC, pooled correctness AUROC is 0.87, yet AUROC separating repairs from residual errors falls from 0.70 at initial ranks 2-3 to 0.39 at ranks 21-50. Errors starting beyond rank 20, inside the reversed region, make up 46.6% of all post-fusion errors. We prop
    
[^85]: VOMMI：面向移动操作的便携式示范收集与利用

    VOMMI: Collecting and Leveraging Portable Demonstrations for Mobile Manipulation

    [https://arxiv.org/abs/2610.08220](https://arxiv.org/abs/2610.08220)

    提出VOMMI框架，通过离线轨迹重建和在线视觉运动条件化，将低成本、便携式的RGB示范与VLA后训练连接起来，无需额外传感硬件即可实现移动操作的示范收集。

    

    便携式移动操作示范可以帮助缓解具身智能领域的数据稀缺问题，但从RGB观测中获得可靠、低成本且无需机器人的运动监督仍然具有挑战性。现有方法通常依赖遥操作或配备额外传感硬件的专用设备，而直接使用估计的视觉里程计（VO）轨迹会由于累积漂移和不完善的运动监督而引入不一致性。我们提出了视觉里程计条件化移动操作接口，这是一个便携式示范收集与学习框架，通过离线轨迹重建和在线视觉运动条件化，将便携式RGB示范与视觉-语言-动作（VLA）后训练连接起来。VOMMI同步身体和手部视角，无需人机运动学对应关系即可捕获导航上下文和局部物体交互……

    arXiv:2610.08220v1 Announce Type: cross  Abstract: Portable mobile-manipulation demonstrations can help alleviate data scarcity for embodied intelligence, but obtaining reliable, low-cost, and robot-free motion supervision from RGB observations remains challenging. Existing approaches often rely on teleoperation or specialized devices equipped with additional sensing hardware, while directly using estimated visual odometry (VO) trajectories can introduce inconsistencies due to accumulated drift and imperfect motion supervision. We present the Visual-Odometry-Conditioned Mobile Manipulation Interface (VOMMI), a portable demonstration collection and learning framework that connects portable RGB demonstrations to vision-language-action (VLA) post-training through offline trajectory reconstruction and online visual-motion conditioning. VOMMI synchronizes body and hand views to capture navigation context and local object interactions without requiring human-robot kinematic correspondence ca
    
[^86]: 量子纠缠多模态融合网络（QEMFN）：基于可训练纠缠的资源感知混合视觉-语言融合

    Quantum Entangled Multimodal Fusion Networks (QEMFN): Resource-Aware Hybrid Vision-Language Fusion via Trainable Entanglement

    [https://arxiv.org/abs/2610.08216](https://arxiv.org/abs/2610.08216)

    该论文提出量子纠缠多模态融合网络（QEMFN），一种将参数化纠缠作为结构化归纳偏置的混合量子-经典视觉-语言融合框架，在参数预算匹配且使用相同冻结CLIP骨干的条件下，于COCO-5k和Flickr30k检索任务上超越了多种经典融合基线。

    

    多模态视觉-语言系统通常通过拼接、注意力、双线性池化或张量交互等经典算子来融合图像和文本嵌入。我们提出了量子纠缠多模态融合网络（QEMFN），这是一种混合量子-经典框架，将参数化纠缠作为多模态融合的结构化归纳偏置引入。预训练的视觉和文本特征被投影到紧凑的潜在空间中，编码为角度参数化的量子态，通过模态内和成对的跨模态纠缠电路进行处理，并经测量生成用于检索的融合表示。在参数预算匹配且使用相同冻结CLIP骨干网络的条件下，QEMFN在COCO-5k和Flickr30k数据集上优于多种经典融合基线，包括多层感知机、张量融合、FiLM、交叉注意力、紧凑型Transformer以及去量化的成对拓扑类似方法。消融实验套件分离出了量子组件的贡献（摘要此处截断）。

    arXiv:2610.08216v1 Announce Type: new  Abstract: Multimodal vision-language systems typically fuse image and text embeddings through classical operators such as concatenation, attention, bilinear pooling, or tensor interactions. We propose Quantum Entangled Multimodal Fusion Networks (QEMFN), a hybrid quantum-classical framework that introduces parameterized entanglement as a structured inductive bias for multimodal fusion. Pretrained visual and textual features are projected into compact latent spaces, encoded as angle-parameterized quantum states, processed through intra-modal and paired cross-modal entangling circuits, and measured to produce fused representations for retrieval. Under matched parameter budgets and identical frozen CLIP backbones, QEMFN outperforms classical fusion baselines on COCO-5k and Flickr30k, including multilayer perceptron, tensor fusion, FiLM, cross-attention, compact transformer, and a dequantized paired-topology analogue. An ablation suite isolates the qu
    
[^87]: Learn2Play Bench：LLM智能体在不熟悉环境中从经验中学习的能力究竟有多强？

    Learn2Play Bench: How Well Do LLM Agents Learn from Experience in Unfamiliar Environments?

    [https://arxiv.org/abs/2610.08215](https://arxiv.org/abs/2610.08215)

    本文提出了Learn2Play Bench基准，通过规则新颖或反直觉的新设计文本游戏，评估LLM智能体在不熟悉环境中通过交互从经验中学习的能力，而非依赖预训练知识。

    

    arXiv:2610.08215v1 公告类型：新论文。摘要：从经验中学习对于LLM智能体适应不熟悉且动态变化的环境至关重要，因此评估这种能力对于理解智能体获取和运用新知识的有效性非常重要。现有的基准测试虽然试图评估这一能力，但其主要针对的任务规则要么已在指令中给出，要么是预训练模型早已熟悉的，这使得人们难以区分“通过交互进行的学习”与“基于已有知识的推理”。为解决这一问题，我们提出了Learn2Play Bench，这是一个由新设计的文本游戏构成的基准测试，其规则是新颖的或反直觉的，要求智能体必须通过交互来获取知识，而不能仅依赖预训练知识。这些游戏提供可复现的反馈和自动评分，使得能够在多次重复尝试中对学习过程进行受控评估。我们还通过变换游戏实例，来测试智能体能否将所学知识……（摘要原文在此处截断）

    arXiv:2610.08215v1 Announce Type: new  Abstract: Learning from experience is essential for LLM agents to adapt to unfamiliar and dynmaic environments. Evaluating this ability is therefore important for understanding how effectively agents acquire and use new knowledge. Existing benchmarks have sought to evaluate this ability, but they primarily evaluate tasks whose rules are provided in the instructions or already familiar to pretrained models, making it difficult to distinguish learning from interactions from reasoning with existing knowledge. To address this, we introduce Learn2Play Bench, a benchmark of newly designed text-based games, whose rules are novel or counterintuitive, requiring agents to acquire knowledge through interaction rather than rely solely on pretrained knowledge. These games provide reproducible feedback and automatic scoring, enabling controlled evaluation of learning across repeated attempts. We also vary game instances to test whether agents can apply what the
    
[^88]: 用于逻辑教学的数学证明助手：LogiKEy 方法论

    Mathematical Proof Assistants for Teaching Logic: The LogiKEy Methodology

    [https://arxiv.org/abs/2610.08214](https://arxiv.org/abs/2610.08214)

    提出基于 LogiKEy 方法论的逻辑教学方案，以经典高阶逻辑作为通用元逻辑，通过语义嵌入让单一证明助手成为学生学习、实验和比较多种（经典与非经典）逻辑的统一环境。

    

    我们报告了一种向计算机科学、数学和哲学混合学生群体教授逻辑的方法，该方法基于逻辑多元主义的 LogiKEy 方法论，十多年来一直应用于课程、暑期学校和教程中。LogiKEy 将经典高阶逻辑（HOL）用作通用元逻辑，通过定义语义将对象逻辑（无论是经典还是非经典逻辑）编码其中；借助这些语义嵌入，单一的证明助手（例如 Isabelle/HOL）及其自动定理证明器和（反）模型查找器便成为学生学习和比较各种逻辑、并进行实验的统一环境。在为证明助手在逻辑课堂中的应用提供教学论证之后，我们提出了一系列渐进式的课堂示例，每次过渡都由前一种表示方式的局限性、对更明确建模资源的需求或新的应用所驱动。一个说谎者-说真话者……

    arXiv:2610.08214v1 Announce Type: new  Abstract: We report on an approach to teaching logic to mixed groups of computer science, mathematics, and philosophy students, based on the logico-pluralistic LogiKEy methodology, used for more than a decade in courses, summer schools, and tutorials. LogiKEy uses classical higher-order logic (HOL) as a universal metalogic in which object logics, classical and non-classical alike, are encoded by defining their semantics; through these semantical embeddings a single proof assistant (e.g. Isabelle/HOL), with its automated theorem provers and (counter-)model finders, becomes one environment in which students learn, experiment with, and compare logics. After making the pedagogical case for proof assistants in the logic classroom, we present a graded sequence of classroom examples, each transition motivated by a limitation of the preceding representation, by a need for more explicit modelling resources, or by a new application. A liars-and-truth-teller
    
[^89]: STRUCTURALCOST：一个用于建模人类句子处理难度的受控阅读时间数据集

    STRUCTURALCOST: A controlled reading time dataset for modeling human sentence processing difficulty

    [https://arxiv.org/abs/2610.08208](https://arxiv.org/abs/2610.08208)

    该研究推出大规模阅读时间数据集STRUCTURALCOST，验证了人类阅读时间随主谓依存长度增加而上升，并揭示现有语言模型虽能反映这种预测性难度但低估了工作记忆导致的整合成本，为评估语言模型的认知合理性奠定了数据基础。

    

    我们推出了STRUCTURALCOST，这是一个包含475名参与者和40,800个观测数据的自定步速阅读数据集，专门用于分离长距离主谓依存消解的处理成本。我们在NLP规模上复现了一个此前统计效力较低的心理语言学发现，即人类在主要动词处的阅读时间随依存长度增加而上升，且这一现象由超越线性距离的句法嵌套所驱动。不同的语言模型——涵盖n-gram模型、状态空间模型（SSM）和transformer——能部分反映这种分级难度模式，但低估了人类产生的整合成本，且这一差距在不同架构和模型规模下持续存在。这表明这些模型捕捉到了人类处理过程中的预测成分，但未能捕捉工作记忆所带来的全部整合成本。STRUCTURALCOST提供了推动语言模型认知合理性评估所需的数据。

    arXiv:2610.08208v1 Announce Type: cross  Abstract: We introduce STRUCTURALCOST, a self-paced reading dataset of 475 participants and 40,800 observations isolating the processing cost of long-distance subject-verb dependency resolution. We replicate a low-powered psycholinguistic finding at NLP scale, namely that human reading times at the main verb increase with dependency length, driven by syntactic embedding beyond linear distance. Different language models -- spanning n-gram models, SSMs, and transformers -- partially mirror this graded difficulty profile, yet underestimate the integration cost humans incur, with a gap that persists across architectures and model sizes. This suggests these models capture the predictive component of human processing but not the full integration cost that working memory imposes. STRUCTURALCOST provides data needed to drive progress toward evaluating the cognitive plausibility of language models.
    
[^90]: 针对小型希腊语-英语知识库的工具调用检索与向量RAG对比：准确性及对用户希腊语输入方式的鲁棒性

    Tool-calling retrieval versus vector RAG for a small Greek--English knowledge base: accuracy and robustness to how users type Greek

    [https://arxiv.org/abs/2610.08205](https://arxiv.org/abs/2610.08205)

    在小型希腊语-英语农业知识库基准KyGround上，向量RAG（95.3%）显著优于工具调用检索（71.6%），将整个知识库直接放入上下文可达99.3%，且向量RAG对无重音、大写及Greeklish等非标准希腊语输入形式表现出鲁棒性。

    

    基于小型、频繁编辑的知识库的智能助手可以通过调用实时数据接口的工具调用进行检索，也可以通过向量检索增强生成（RAG）进行检索。我们在KyGround基准上对这两种方法进行了比较，该基准包含198个问题，来源于一个位于希腊基西拉岛的希腊语-英语农业平台的公开记录，答案通过与记录自动核对进行验证，且每个问题最多以九种形式提出，包括无重音符号的希腊语、大写希腊语以及三种拉丁字母转写（Greeklish）方案。以Claude Haiku 4.5作为路由和回答模型，该平台工具代理的重构版本对标准希腊语问题的正确回答率为71.6%，而向量RAG为95.3%（差异为-23.6个百分点，95%置信区间为-33.1至-15.1）。让路由器自行编写向量查询没有任何改变，而将整个知识库（约26,000个token）直接放入提示词中则达到了99.3%。工具代理的损失出现在……

    arXiv:2610.08205v1 Announce Type: cross  Abstract: Assistants grounded in a small, frequently edited knowledge base can retrieve through tool calls to a live data interface or through vector retrieval-augmented generation (RAG). We compare the two on KyGround, a benchmark of 198 questions drawn from the published records of a Greek--English agricultural platform on Kythera, Greece, with answers verified automatically against the records and each question posed in up to nine forms, including Greek without accents, in capitals and in three Latin-script (Greeklish) schemes. With Claude Haiku 4.5 as router and answer model, a reconstruction of the platform's tool agent answered 71.6\% of canonical Greek questions correctly and vector RAG 95.3\% (difference $-23.6$ percentage points, 95\% CI $-33.1$ to $-15.1$). Letting the router write the vector query changed nothing, and placing the whole knowledge base of about 26,000 tokens in the prompt reached 99.3\%. The tool agent's losses arose in
    
[^91]: 紧凑的机器人策略需要细粒度的视觉表示

    Compact Robot Policies Need Fine-Grained Visual Representations

    [https://arxiv.org/abs/2610.08183](https://arxiv.org/abs/2610.08183)

    论文提出紧凑策略CoRP，证明机器人多任务操作的性能主要由细粒度的预训练视觉表示决定，而非参数规模或生成式先验，其4890万参数的小模型即可媲美比它大上百倍的系统。

    

    多任务操作策略在架构、规模和预训练先验上同时存在差异，因此已发表的性能比较无法将效果归因于任何单一组件。我们认为，性能差异主要来自视觉表示，而参数规模和生成式先验在很大程度上是次要因素。为了验证这一点，我们构建了CoRP（压缩表示策略，Compressed Representation Policy），这是一个刻意保持紧凑的策略（4890万参数，不使用视觉-语言模型，也不使用视频生成先验），它分解为表示提取器和流匹配动作生成器两个部分。它在LIBERO上达到97.0%，在RoboTwin 2.0的Clean/Randomized设置下达到75.78%/73.36%，与比它大40.9至163.6倍的系统相当。在保持动作生成器固定不变的情况下，我们每次只改变提取器的一个属性。预训练初始化是决定性的：随机初始化的ViT-S/14在LIBERO上的性能降至78.1%，ImageNet预训练的ResNet-34降至74.5%。但仅有预训练是不够的，因为冻结编码器会带来19……（摘要原文在此处截断）

    arXiv:2610.08183v1 Announce Type: cross  Abstract: Multi-task manipulation policies differ in architecture, scale, and pretrained priors all at once, so published comparisons cannot attribute performance to any single component. We argue that most of it comes from the visual representation, and that parameter scale and generative priors are largely incidental. To test this, we build CoRP (Compressed Representation Policy), a deliberately compact policy (48.9M parameters, no vision-language model and no video-generative prior) that factorizes into a representation extractor and a flow-matching action generator. It reaches 97.0% on LIBERO and 75.78%/73.36% on RoboTwin 2.0 Clean/Randomized, matching systems 40.9-163.6x larger. Holding the action generator fixed, we then vary one extractor property at a time. Pretrained initialization is decisive: a random ViT-S/14 drops to 78.1% and an ImageNet ResNet-34 to 74.5% on LIBERO. Pretraining alone is not enough, as freezing the encoder costs 19
    
[^92]: LFHE：面向非独立同分布数据下去中心化学习中受限局部拓扑搜索的局部优先启发式演化

    LFHE: Local-First Heuristic Evolution for Bounded Local Topology Search in Decentralized Learning with Non-IID Data

    [https://arxiv.org/abs/2610.08176](https://arxiv.org/abs/2610.08176)

    LFHE 提出了一种仅利用自身邻域和朋友的朋友信息进行受限局部拓扑重连的框架，其结构分数与图狄利克雷能量精确对应，从而在非独立同分布数据下有效加速去中心化学习中表示分歧的消散。

    

    在非独立同分布（non-IID）数据下，去中心化学习对通信拓扑高度敏感。自适应邻居选择方法可以利用本地模型信息，但更广泛的节点发现可能需要不断增大的控制状态，而直接的谱优化方法通常依赖全图信息。我们研究了受限局部拓扑搜索这一中间设定，提出了局部优先启发式演化（LFHE），这是一个由表示驱动的重连框架，其候选节点发现与评分仅使用自身邻域和朋友的朋友（FoF）信息。该结构分数可通过图狄利克雷能量得到精确解释：其在各客户端上的总和等于表示狄利克雷能量的两倍，而在标准线性共识动力学下，该能量控制着表示分歧的瞬时消散。LFHE 将这种依赖状态的结构信号与早期探索和度控制相结合……

    arXiv:2610.08176v1 Announce Type: new  Abstract: Decentralized learning is highly sensitive to communication topology under non-IID data. Adaptive peer-selection methods can exploit local model information, but broader peer discovery may require increasingly large control state, whereas direct spectral optimization typically relies on graph-wide information. We study the intermediate setting of bounded local topology search and propose Local-First Heuristic Evolution (LFHE), a representation-driven rewiring framework whose candidate discovery and scoring use only ego-neighborhood and friend-of-a-friend (FoF) information. The structural score admits an exact interpretation through graph Dirichlet energy: its sum across clients equals twice the representation Dirichlet energy, which under standard linear consensus dynamics governs the instantaneous dissipation of representation disagreement. LFHE combines this state-dependent structural signal with early exploration and degree control, w
    
[^93]: 哪种合金成分、哪些工艺参数？从优化的金属微观组织与织构反推“配方”

    Which alloy composition,what process parameters? Inferring the recipe from optimized metallic microstructure and texture

    [https://arxiv.org/abs/2610.08165](https://arxiv.org/abs/2610.08165)

    该研究的核心贡献是首次探索用机器学习实现合金开发的“逆向”步骤——从优化的微观组织与织构反推出合金成分和加工工艺参数，并在镁合金挤压数据集上系统比较了传统统计量、预训练视觉嵌入和图神经网络三种微观结构描述方法。

    

    金属合金的力学性能由其微观组织和织构决定，即晶粒的尺寸与形状以及其晶体的取向。这种结构又反过来由一个“配方”决定，即合金成分与加工工艺参数。合金开发通常沿这条链正向进行，调整结构直到满足目标性能；而反向进行——从优化后的结构反推出能产生它的配方——目前仍依赖专家知识。我们探究这一反向步骤是否可以被学习。在一个自建数据集上（包含14种合金共107个镁合金挤压工艺条件，每个条件均配有光学显微图像和X射线织构测量），我们比较了三种微观组织与织构的描述方法：传统的晶粒与织构统计量、来自预训练图像编码器的视觉嵌入，以及在晶粒网络上构建的图神经网络。每种描述方法均与用于两项任务的预测头相结合：合金……（原文摘要在此处截断）

    arXiv:2610.08165v1 Announce Type: new  Abstract: The mechanical properties of a metallic alloy are set by its microstructure and texture: the size and shape of its grains and the orientation of their crystals. That structure is in turn set by a recipe, the alloy composition together with the processing parameters. Alloy development runs this chain forwards, tuning the structure until a target property is met. Running it backwards, from an optimized structure to the recipe that would produce it, still relies on expert knowledge. We ask whether this backwards step can be learned. On an in-house dataset of 107 magnesium alloy extrusion conditions across 14 alloys, each with optical micrographs and an X-ray texture measurement, we compare three descriptors of microstructure and texture: conventional grain and texture statistics, a vision embedding from a pretrained image encoder, and a graph neural network on the grain network. Each is paired with prediction heads for two tasks: the alloy 
    
[^94]: 文本生成交响曲：临床病历生成基准测试

    Symphony for Text Generation: Benchmarking Clinical Note Generation

    [https://arxiv.org/abs/2610.08161](https://arxiv.org/abs/2610.08161)

    该论文提出了包含300例多语言临床就诊记录的MedConv数据集，并构建了结合蕴含指标与大语言模型评判的受控临床评估框架，证明临床AI平台Corti的病历生成质量与领先商业环境式记录软件相当或更优，且其可配置API可针对特定文档需求灵活优化质量维度。

    

    环境式文档系统正迅速获得广泛应用，但其对临床病历质量的影响仍缺乏充分表征。我们推出了MedConv——一个包含300例临床就诊记录的多语言数据集，涵盖英语、丹麦语和德语，并将其与环境临床智能基准（ACI-BENCH）结合使用，将Corti（一个临床AI平台）与两个基于通用AI构建的领先且易用的环境式记录软件进行比较。我们提出了一个受控临床评估框架，该框架将文本蕴含指标与大语言模型评判的成对比较相结合，涵盖从PDSQI-9采纳的八个维度。结果表明，Corti基于API的文本生成基础设施与领先的商业记录软件相当或更优。我们进一步表明，Corti的可配置API提供了必要的灵活性，能够针对特定的文档用例对质量维度进行微调。我们呈现了该评估方法并发布了数据集。

    arXiv:2610.08161v1 Announce Type: cross  Abstract: Ambient documentation systems are rapidly gaining adoption, yet their impact on clinical note quality remains poorly characterized. We introduce MedConv, a multilingual dataset of 300 clinical encounters in English, Danish, and German, and use it alongside the Ambient Clinical Intelligence benchmark (ACI-BENCH) to compare Corti, a clinical AI platform, with two leading, accessible ambient scribe software applications built on general-purpose AI. We present a controlled clinical evaluation framework that combines entailment metrics with LLM-judged pairwise comparisons across eight dimensions adopted from PDSQI-9. Results show that Corti's API-based text-generation infrastructure is on par with or outperforms leading commercial scribes. We further show that Corti's configurable API provides the flexibility necessary to fine-tune quality dimensions for specific documentation use cases. We present the evaluation methodology and release a d
    
[^95]: 基于系统一引导的计算式分工实现令牌高效的多智能体协作

    Token-Efficient Multi-Agent Collaboration via System One-Guided Computational Division of Labor

    [https://arxiv.org/abs/2610.08155](https://arxiv.org/abs/2610.08155)

    提出S1-MAS框架，通过“系统一”式的计算分工将有界的协调决策交给轻量级模型处理、让大语言模型专注于开放式推理，从而在不牺牲协作性能的前提下大幅降低多智能体系统的令牌开销和延迟。

    

    基于大语言模型（LLM）的多智能体系统（MAS）通过让专门化的智能体协同求解问题，已成为应对复杂信息检索与推理任务的一种有前景的范式。然而，现有的MAS框架将任务推理与协调操作紧密耦合，这些协调操作包括任务选择、角色分配、消息路由和上下文管理。随着交互规模的增长，使用强大的LLM来执行这些有界的控制决策会带来巨大的令牌开销和延迟，限制了智能体化Web服务的可扩展性。在本文中，我们研究是否可以在不损害协作性能的前提下，将协调机制从昂贵的推理中解耦出来。我们提出了S1-MAS，一个基于系统一引导的计算式分工的令牌高效多智能体框架。S1-MAS将有界的协调决策分配给轻量级的“系统一”模型，同时将开放式的推理保留给（摘要原文在此处截断）

    arXiv:2610.08155v1 Announce Type: cross  Abstract: Large language model (LLM)-based multi-agent systems (MAS) have become a promising paradigm for complex information-seeking and reasoning tasks by enabling collaborative problem solving among specialized agents. However, existing MAS frameworks tightly couple task reasoning with coordination operations, including task selection, role assignment, message routing, and context management. As interactions grow, using powerful LLMs for these bounded control decisions introduces substantial token overhead and latency, limiting the scalability of agentic Web services. In this paper, we investigate whether coordination can be decoupled from expensive reasoning without compromising collaborative performance. We propose S1-MAS, a token-efficient multi-agent framework based on System One-guided computational division of labor. S1-MAS assigns bounded coordination decisions to lightweight System One models while reserving open-ended reasoning for c
    
[^96]: 惩罚框架下的无有效选项多选题问答：分析大语言模型在无效选项下的弃答行为

    Penalty-Framed No-Valid-Option MCQA: Analyzing LLM Abstention under Invalid Choices

    [https://arxiv.org/abs/2610.08153](https://arxiv.org/abs/2610.08153)

    该论文提出“惩罚框架下的无有效选项多选题问答”这一新评测设定，并通过基于正确回答的条件分析方法，揭示了大语言模型的高答题准确率并不能保证其在所有选项均无效时可靠地选择弃答。

    

    多选题问答（MCQA）通常被用于评估大语言模型，其前提假设是所提供的选项中必有一个正确答案，并通常以答案选择的准确率来衡量模型表现。然而，在实际部署中，用户或检索系统可能会提供无效的选项集合，即所列出的选项中没有一个是正确的，而此时强行选择其中一项可能会带来下游成本。我们将这种设定称为惩罚框架下的无有效选项多选题问答。基于MMLU-Pro的数学子集，我们移除标注的正确选项，允许模型选择剩余选项或输出ABSTAIN（弃答），并对无效的强制选择回答施加惩罚。我们进一步引入了基于正确回答的条件分析，仅在模型原本回答正确的实例上评估其弃答能力。实验表明，较高的MCQA准确率并不能完全保证弃答行为的可靠性：即使在明确的“无有效选项”感知指令和基于惩罚的约束下（摘要原文在此截断）。

    arXiv:2610.08153v1 Announce Type: cross  Abstract: Multiple-choice question answering (MCQA) is commonly used to evaluate large language models under the assumption that one of the provided options is correct, typically using answer-selection accuracy. However, in real deployments, users or retrieval systems may provide invalid option sets in which none of the listed choices is correct, and selecting one of them may incur downstream cost. We study this setting as penalty-framed no-valid-option MCQA. Using the mathematics subset of MMLU-Pro, we remove the labeled correct option, allow models to either choose a remaining option or output ABSTAIN, and penalize invalid forced-choice responses. We further introduce correct-conditioned analysis, evaluating abstention only on instances that the model originally answered correctly. Experiments show that high MCQA accuracy does not fully guarantee abstention reliability: even under explicit no-valid-option-aware instructions and penalty-based s
    
[^97]: 纳维-斯托克斯在翻译中迷失：为什么对AI自动形式化的Lean验证并不能保证正确的自然语言证明

    Navier-Stokes lost in translation: Why Lean verification of AI autoformalisation does not guarantee correct natural language proofs

    [https://arxiv.org/abs/2610.08144](https://arxiv.org/abs/2610.08144)

    本文证明了解决数学自然语言文本歧义（语义忠实自动形式化的必要步骤）这一难题在可解性复杂性索引层级中处于任意高的不可解位置（SCI=∞），因此即使AI自动形式化通过了Lean机械验证，也不能保证原始自然语言证明的正确性。

    

    自动形式化正日益被用于验证数学文本，包括AI生成的文本，例如OpenAI所宣布的纳维-斯托克斯方程解爆破的证明。在这一过程中，AI系统将文本从自然语言（NL）翻译为形式语言（如Lean）。翻译完成后，以形式语言表达的论证即可轻松地被机械化验证。本文旨在阐明为什么这一过程可能无法为原始的自然语言论证提供任何可信度，原因在于实现语义忠实的翻译存在种种困难。特别地，我们强调，解决数学自然语言文本中的歧义问题——这是提供语义忠实翻译所必需的——在可解性复杂性索引（SCI）层级/算术层级中位于任意高的不可解位置（SCI = ∞）。因此，非正式地说，提供语义……

    arXiv:2610.08144v1 Announce Type: cross  Abstract: Autoformalisation is increasingly used to verify mathematical texts, including those generated by AI, as in OpenAI's announced proof of blow-up of solutions to the Navier-Stokes equations. In this process, an AI system translates the text from a natural language (NL) into a formal language such as Lean. Once this translation is done, the argument expressed in the formal language can easily be mechanically verified. The purpose of this article is to demonstrate why this process may offer no confidence in the original NL argument, owing to the various difficulties in performing the translation semantically faithfully. In particular, we highlight that the problem of resolving ambiguities in mathematical NL text, which is necessary in order to provide semantically faithful translation, is arbitrarily high up in the Solvability Complexity Index (SCI) hierarchy/arithmetical hierarchy (the SCI $= \infty$). Hence, informally, providing semanti
    
[^98]: 通过预测伙伴意图实现部分可观测的零样本协调

    Partially Observable Zero-shot coordination by Predicting Intention of Partner

    [https://arxiv.org/abs/2610.08142](https://arxiv.org/abs/2610.08142)

    提出PIP方法，利用联合视角VAE构建仅凭局部观测即可获得的伙伴表示，并通过伙伴状态信念网络推断伙伴的隐藏位置与行为倾向，解决了具身零样本协调中伙伴不可见导致的表示模糊与状态不确定问题，在多个基准及人类评估中取得最优表现。

    

    具身环境中的零样本协调要求智能体在伙伴间歇性脱离视野的情况下仍能进行有效行动，这使得现有方法面临伙伴表示模糊以及对伙伴隐藏状态不确定的难题。我们提出了预测伙伴意图方法，以联合应对这些挑战。PIP使用联合视角变分自编码器，将两个智能体局部观测并集中更丰富的训练时证据提炼为一种仅凭自身局部观测即可获得的伙伴表示。伙伴状态信念网络进一步从自我智能体的交互历史中推断伙伴的隐藏位置和行为倾向。我们在Burrito-PO、Overcooked-PO以及一个Melting Pot基底环境中对PIP进行了评估，并在Burrito-PO中开展了人类评估。PIP在所有三个基准测试中均取得了对比方法中最高的平均性能。人类评估和诊断分析进一步验证了其与不可见伙伴进行协调的能力以及……

    arXiv:2610.08142v1 Announce Type: new  Abstract: Zero-shot coordination in embodied settings requires acting while the partner is intermittently out of view, leaving existing methods with ambiguous partner representations and uncertainty over hidden partner states. We propose Predicting Intention of Partner (PIP) to jointly address these challenges. PIP uses a Joint-view VAE to distill richer training-time evidence from the union of both agents' local observations into a partner representation available from local observations alone. Partner-state Belief networks further infer the partner's hidden location and behavioral tendencies from the ego agent's interaction history. We evaluate PIP in Burrito-PO, Overcooked-PO, and a Melting Pot substrate, together with a human evaluation in Burrito-PO. PIP attains the highest mean performance among the compared methods across all three benchmarks. Human evaluation and diagnostic analyses further support coordination with unseen partners and the
    
[^99]: 面向长程法律推理的测试时代理演化

    Test-Time Agent Evolution for Long-Horizon Legal Reasoning

    [https://arxiv.org/abs/2610.08138](https://arxiv.org/abs/2610.08138)

    该论文提出了一种免训练的测试时代理自适应方法，通过测试时记忆演化机制从历史案例中检索、适配并积累可复用经验，无需更新模型参数即可提升长程多角色法律推理的全局可靠性。

    

    法律智能旨在支持涉及案件状态演化和多角色参与的长程法律过程中的可靠决策。然而，现实世界的法律部署在事实、证据和程序背景方面存在显著的案件异质性，暴露了静态代理策略的局限性。此外，法律推理在各角色和程序阶段之间本质上是相互依赖的，这使得全局可靠性与孤立的角色能力有着根本区别。为应对这些挑战，我们研究了免训练的测试时代理自适应方法，即代理在不更新模型参数的情况下，持续利用来自先前案例和正在进行交互的部署时信号。我们提出了该方法，引入“测试时记忆演化”机制，从先前案例中检索可复用的经验，将其适配到当前的事实和程序背景中，并整合积累的经验以用于后续决策……

    arXiv:2610.08138v1 Announce Type: new  Abstract: Legal intelligence aims to support reliable decision-making across long-horizon legal processes involving evolving case states and multiple roles. However, real-world legal deployment exhibits substantial case heterogeneity in facts, evidence, and procedural contexts, exposing the limitations of static agent strategies. Moreover, legal reasoning is inherently interdependent across roles and procedural stages, making global reliability fundamentally different from isolated role competence. To address these challenges, we study training-free test-time agent adaptation, where agents continuously exploit deployment-time signals from preceding cases and ongoing interactions without updating model parameters. We propose \method, which introduces \emph{Test-Time Memory Evolution} to retrieve reusable experience from previous cases, adapt it to the current factual and procedural context, and consolidate accumulated experience for subsequent deci
    
[^100]: 超市产品检测与识别：利用深度学习与校正图像技术

    Supermarket Product Detection and Recognition: Utilizing Deep Learning with Rectified Imagery

    [https://arxiv.org/abs/2610.08126](https://arxiv.org/abs/2610.08126)

    本文提出将传统霍夫变换和单应性估计与深度学习目标检测模型相结合，通过图像校正技术解决超市密集货架因拍摄角度变化带来的商品识别难题。

    

    产品识别已成为零售行业自动化中最具挑战性的问题之一。随着新的工业5.0标准的推进，自动化库存管理和商品目录创建任务变得至关重要。目标识别模型凭借其前所未有的识别和定位精度，成为了一种可行的解决方案。然而，超市货架紧密排列的设计带来了拍摄图像时的角度变化问题。这种角度变化的密集排列图像（单张图像包含许多物体）对于仅依靠目标检测模型而言难以应对。在本文中，我们尝试将传统的霍夫变换（HT）和单应性估计概念与目标检测模型相结合。我们研究了使用单应性估计和霍夫变换进行图像校正的效果，以及它们在食品杂货识别问题上的局限性。我们倡导创建一个新的数据集来测试……

    arXiv:2610.08126v1 Announce Type: cross  Abstract: Product Identification has sprung up to become one of the most challenging problems in the automation of the retail industry. With the new industry 5.0 standards, automated inventory management, and catalog creation tasks are vitally important. Object identification models have emerged as a viable answer with their unprecedented identification and localization accuracy. However, the close-knit rack design of supermarkets generates the problem of angle variation in capturing images. The angle-variant densely packed images(a single image contains many objects) become overwhelming for these models alone. In this paper, we try to supplement object detection models with traditional Hough transform (HT) and homogeneous estimation concepts. We study the effect of rectified images using homography estimation and hough transform and their limitations on the problem of grocery identification. We make a case for creating a new dataset to test the
    
[^101]: 超越路径点回归：面向端到端驾驶的基于查询的自车可达未来代价学习

    Beyond Waypoint Regression: Query-Based Cost Learning over Reachable Ego Futures for End-to-End Driving

    [https://arxiv.org/abs/2610.08123](https://arxiv.org/abs/2610.08123)

    本文提出一种基于查询的代价学习框架，通过对自车动态可达的未来轨迹估计有界代价来替代传统路径点回归，将代价拓扑转化为可行规划，在nuScenes与真实驾驶数据上显著降低碰撞率并保持可解释性。

    

    基于路径点回归的端到端规划器在开环精度方面表现出色，但它们主要学习模仿专家轨迹的几何形状，且难以适应部署时的安全约束。我们提出了一种基于查询的代价学习框架，为动态可达的自车轨迹查询估计有界代价，而非密集的鸟瞰图网格单元或少量回归轨迹集合。紧凑的联合场景token捕获连贯的多模态智能体未来轨迹，同时借助具备应急感知的代价聚合与代价引导的簇内MPPI混合，将学习到的代价拓扑转换为可行的自车规划。在nuScenes数据集上，我们的方法优于ST-P3和NMP等已有代价估计规划器，在碰撞率方面超过大多数回归基线，同时在L2指标上保持竞争力，并保留了可解释的代价接口。在真实世界驾驶日志上，所提出的规划器相比SparseDrive和Alpamay降低了碰撞率。

    arXiv:2610.08123v1 Announce Type: cross  Abstract: End-to-end planners based on waypoint regression achieve strong open-loop accuracy, but they primarily learn to mimic expert geometry and remain difficult to adapt to deployment-time safety constraints. We propose a query-based cost-learning framework that estimates bounded costs for dynamically reachable ego trajectory queries, rather than dense BEV cells or a small regressed trajectory set. Compact joint scene tokens capture coherent multimodal agent futures, while contingency-aware cost aggregation and cost-guided intra-cluster MPPI mixing convert the learned cost topology into feasible ego plans. On nuScenes, our method improves over prior cost-estimation planners such as ST-P3 and NMP, outperforms most regression baselines in collision rate, while remaining competitive in L2, and retaining an interpretable cost interface. On real-world driving logs, the proposed planner reduces collision rates compared with SparseDrive and Alpamay
    
[^102]: 利用声学导向与内容导向的说话人验证攻击对抗多语言语音匿名化

    Exploiting Acoustic and Content-Oriented Speaker Verification Attacks Against Multilingual Voice Anonymization

    [https://arxiv.org/abs/2610.08107](https://arxiv.org/abs/2610.08107)

    本研究首次在多语言环境下系统评估了声学导向与内容导向的说话人验证攻击对语音匿名化的威胁，发现攻击有效性取决于匿名化语音所保留的语言信息量，并构建了多语言语音转换数据集以提升跨语言泛化能力。

    

    针对语音匿名化的攻击者ASV（自动说话人验证）系统此前主要在英语环境下被研究，其在多语言环境下的行为在很大程度上尚未被探索。传统ASV研究已经表明，声学信息和内容信息对多语言说话人验证都很重要。受此启发，我们研究了同样的规律是否适用于对匿名化语音的攻击者ASV。我们在多语言匿名化语音上评估了声学导向和内容导向两类攻击者，并构建了一个多语言语音转换数据集以提升跨语言泛化能力。我们的结果表明，攻击者的有效性取决于匿名化语音的语言实用性。总体而言，声学导向的攻击者取得了更好的性能。然而，当语言信息得到良好保留时，与语音失真较强的情况相比，内容导向攻击者与声学导向攻击者之间的性能差距会缩小。

    arXiv:2610.08107v1 Announce Type: cross  Abstract: Attacker ASV systems for voice anonymization have been studied primarily in English, leaving their behavior in multilingual settings largely unexplored. Conventional ASV has shown that both acoustic and contextual information are important for multilingual speaker verification. Inspired by this, we investigate whether the same holds for attacker ASV on anonymized speech. We evaluate both acoustic- and content-oriented attackers on multilingual anonymized speech and construct a multilingual voice-converted dataset to improve cross-lingual generalization. Our results show that attacker effectiveness depends on the linguistic utility of the anonymized speech. Overall, acoustic-oriented attackers achieve better performance. However, when linguistic information is well preserved, the performance gap between content- and acoustic-oriented attackers narrows compared with conditions involving stronger speech distortion. The multilingual voice-
    
[^103]: ChartBmkAgent：基于稀疏错误分类规范、由评测框架治理的多智能体图表问答基准构建方法

    ChartBmkAgent: Harness-Governed Multi-Agent Construction of Chart QA Benchmarks from Sparse Error-Taxonomy Specifications

    [https://arxiv.org/abs/2610.08106](https://arxiv.org/abs/2610.08106)

    ChartBmkAgent 提出一种由评测框架治理的多智能体流程，仅凭稀疏的错误分类规范即可按需从零构建图表问答基准，并确保新生的需求与内容始终与外部指定的诊断目标保持对齐，从而缩短基准开发周期。

    

    多模态大语言模型（MLLM）发展迅速，而传统基准开发进度滞后，导致对新观察到的能力差距的探究被延误。这类探究需要富有表现力的任务形式和按需的构建流程：信息丰富的图表使图表问答（Chart QA）成为探测感知与推理耦合能力的合适任务。自动化图表问答构建旨在将已识别的差距按需转化为针对性样本，从而缩短基准开发周期。然而，现有方法通常将目标引导与从零生成分离开来：目标引导系统往往需要预先准备的数据、图表或模板，而从零生成系统主要确保产物的有效性，未能显式控制新合成的需求与内容是否与外部指定的诊断目标保持一致。我们提出了 ChartBmkAgent，它将一个已识别的……（摘要原文在此处截断）

    arXiv:2610.08106v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) advance rapidly, while conventional benchmark development lags behind, delaying investigation of newly observed capability gaps. Such investigation requires an expressive task format and an on-demand construction process: information-rich charts make chart question answering (Chart QA) suitable for probing coupled perception and reasoning. Automated Chart QA construction is intended to shorten the benchmark-development cycle by turning identified gaps into targeted samples on demand. Current methods, however, commonly separate target guidance from scratch generation: target-guided systems often require prepared data, charts, or templates, while scratch-generation systems primarily ensure artifact validity, without explicitly controlling whether newly synthesized requirements and content remain aligned with an externally specified diagnostic target. We introduce ChartBmkAgent, which turns an identi
    
[^104]: DSV-Mem：面向多模态大语言模型智能体在专业工作流中的多模态记忆评估

    DSV-Mem: Evaluating Multimodal Memory in Professional Workflows for MLLM Agents

    [https://arxiv.org/abs/2610.08102](https://arxiv.org/abs/2610.08102)

    该论文提出了首个面向专业工作流的多模态记忆评估基准DSV-Mem，通过1,000个专家审校的问题和五大任务类别，评估MLLM智能体在信息密集、频繁更新且需精确追踪状态的专业产物场景下的密集有状态视觉记忆能力。

    

    会话式多模态大语言模型（MLLM）智能体日益被期望能够协助专业工作流，涵盖从人工智能研究、工程设计到产品管理和业务运营等领域。然而，这一能力仍未得到充分探索：现有基准主要聚焦于非正式的日常互动和个人生活场景，其特点是摄影类自然图像、孤立的静态产物以及以回忆为导向的问题。相比之下，专业场景往往涉及结构化、信息密集的产物，这些产物会经历频繁的修订和权威更新，同时还需要进行组合式查询，即在精确追踪状态的同时协调多个产物版本。为应对这些挑战，我们提出了DSV-Mem，一个用于评估密集有状态视觉记忆的基准。DSV-Mem包含经专家审校的场景以及横跨五个面向用户的类别（当前状态、过去状态、派生状态、变更历史以及冲突/……）的1,000个问题。

    arXiv:2610.08102v1 Announce Type: new  Abstract: Conversational MLLM agents are increasingly expected to assist in professional workflows, from AI research and engineering design to product management and business operations. Yet this capability remains underexplored: existing benchmarks largely focus on informal, everyday interactions and personal-life scenarios featuring photographic natural images, isolated static artifacts, and recall-oriented questions. In contrast, professional scenarios often involve structured, information-heavy artifacts that undergo frequent revisions and authority updates, and compositional queries requiring reconciliation of many artifact versions while tracking state precisely. To address these challenges, we introduce DSV-Mem, a benchmark for evaluating Dense Stateful Visual Memory. DSV-Mem comprises expert-reviewed scenarios and 1,000 questions across five user-oriented categories (Current State, Past State, Derived State, Change History, and Conflict/Re
    
[^105]: 超越修正记忆：多智能体系统中的执行一致性

    Beyond Corrected Memory: Execution Consistency in Multi-Agent Systems

    [https://arxiv.org/abs/2610.08101](https://arxiv.org/abs/2610.08101)

    该论文提出多智能体系统中的“执行一致性”概念，证明完全相同的共享记忆记录在同一任务规则下可能同时对应合规与违规的执行，因此仅有正确的记录不足以判断任务职责是否被履行，需要显式的证据条件（如接收凭证、动作依赖、响应有效性）来加以区分。

    

    共享内存用于协调智能体的行为，但正确的记录并不能证明这些行为满足任务要求。内存治理和故障诊断对记录的信息进行规范或检查，但它们本身并不能确定这些信息是否足以判断任务职责的履行。我们通过管理状态使用、信息交接和最终状态一致性三方面职责来定义执行一致性，并给出判断职责是否履行的明确证据条件。我们的核心主张是：在同一任务规则下，完全相同的保留记录可能同时对应合规的执行与违规的执行。受控地移除诸如接收凭证、动作依赖或响应有效性等证据后，82.4%的相反标签样本对变得不可区分；恢复这些证据后，97.9%的合并样本对被重新区分开来。基于自然语言日志的标注能够在实际执行中识别出所定义的违规行为。然而，现有日志并不总是显式地表示执行关系。

    arXiv:2610.08101v1 Announce Type: new  Abstract: Shared memory coordinates agents' actions, but correct records do not establish that those actions satisfy task requirements. Memory governance and failure diagnosis regulate or inspect recorded information; they do not by themselves establish whether it is sufficient to judge task duties. We define execution consistency through duties governing state use, information handoffs, and final-state agreement, with explicit evidence conditions for judging fulfillment. Our core claim is that identical retained records can correspond to compliant and violating executions under the same task rule. Controlled removal of evidence such as receipt, action dependence, or response validity leaves 82.4% of opposite-label pairs indistinguishable; restoration separates 97.9% of the merged pairs. Natural-log annotations identify the defined violations in actual executions. However, existing logs do not always explicitly represent the execution relationship
    
[^106]: 当工具撒谎时：受污染工具反馈下数学智能体的可靠性

    When Tools Lie: Reliability of Mathematical Agents Under Corrupted Tool Feedback

    [https://arxiv.org/abs/2610.08097](https://arxiv.org/abs/2610.08097)

    该论文提出一个受控污染框架研究数学智能体检测和纠正被篡改工具反馈的能力，发现无验证时污染使准确率从100%降至72.4%，而强制同上下文反思可将性能完全恢复至100%。

    

    数学问题求解通常需要确定性的计算步骤，智能体会将这些步骤委托给工具并隐式地信任它们。然而，工具可能会无声地失效，返回看似合理但错误的结果。智能体能在多大程度上检测并纠正被污染的工具调用输出？我们通过一个受控污染框架来研究这个问题：在该框架中，一个隐藏的拦截器会在特定问题上将工具调用结果替换为看似合理的错误信息。我们在31个问题上评估了智能体，采用四种验证设计，包括无验证（基线）、强制同上下文反思、可选的新上下文验证以及可选的结构化验证。在没有验证的情况下，污染导致准确率大幅下降，从100%降至72.4%。强制反思能将性能完全恢复至100%。只有当模型主动调用时，可选验证才能提升准确率。我们的结果表明，检查频率与（原文在此处截断）

    arXiv:2610.08097v1 Announce Type: cross  Abstract: Mathematical problem solving often requires deterministic computational steps that agents delegate to tools and implicitly trust. Yet tools can fail silently, returning plausible but incorrect results. How well can agents detect and correct corrupted tool call outputs? We study this through a controlled corruption framework where a hidden interceptor replaces tool call results with plausible incorrect information on targeted problems. We evaluate agents across 31 problems under four verification designs including no verification (baseline), mandatory same-context reflection, optional fresh-context verification, and optional structural verification. Without verification, corruption causes dramatic accuracy loss, from 100% down to 72.4%. Mandatory reflection fully recovers this performance to 100%. Optional verification improves accuracy only when models actively invoke it. Our results show that checking frequency is strongly associated 
    
[^107]: 自然语言问题作为知识图谱的接口：QRAKEN图蒸馏与语义自修复

    Natural Language Questions as an Interface for Knowledge Graphs: QRAKEN Graph Distillation and Semantic Self-Healing

    [https://arxiv.org/abs/2610.08095](https://arxiv.org/abs/2610.08095)

    QRAKEN提出了一种无需训练的神经符号流水线，通过离线图蒸馏生成TTQL图谱证据来引导LLM生成SPARQL查询，并借助确定性的语法与数据模型检查实现迭代式语义自修复，在Text2SPARQL挑战赛上取得了严格的F1最佳成绩。

    

    自然语言访问RDF知识图谱是语义网的核心愿景。大型语言模型（LLM）推动了文本到SPARQL（Text-to-SPARQL）技术的发展，但在面对不熟悉的图谱时，它们往往会生成虽然语法有效、却与实际填充数据模型不符的查询。QRAKEN是一个无需训练、与本体无关的神经符号流水线，它将查询生成建立在经验性图证据而非模式预期之上。离线蒸馏器生成TTQL——对已填充的多跳模式、条件频率以及路径条件下的字面量示例的紧凑描述，并附带一个类-属性共现矩阵。在线阶段，TTQL引导LLM，同时确定性的语法、词汇和数据模型检查为迭代优化提供诊断。在CK25（首届国际Text2SPARQL挑战赛）上，基于QLever快照的相同条件重新计算下，QRAKEN使用GPT-4.1 mini达到严格F1值0.643±0.026，使用GPT-5.4达到0.652±0.012：相对……

    arXiv:2610.08095v1 Announce Type: new  Abstract: Natural-language access to RDF knowledge graphs is a core Semantic Web ambition. Large language models (LLMs) have advanced Text-to-SPARQL, yet on unfamiliar graphs they often generate valid queries that misrepresent the populated data model. QRAKEN is a training-free, ontology-agnostic neurosymbolic pipeline grounding generation in empirical graph evidence rather than schema expectations. An offline distiller produces TTQL, a compact description of populated multi-hop patterns, conditional frequencies and path-conditioned literal examples, plus a class-property co-occurrence matrix. Online, TTQL guides the LLM, while deterministic syntax, vocabulary and data-model checks provide diagnostics for iterative refinement. On CK25 (First International Text2SPARQL Challenge), under matched-condition recomputation on a QLever snapshot, QRAKEN achieves strict F1 of 0.643 $\pm$ 0.026 with GPT-4.1 mini and 0.652 $\pm$ 0.012 with GPT-5.4: relative g
    
[^108]: SAGE：面向有据可依医学问答数据合成的语义锚点引导演化框架

    SAGE: Semantic Anchor-Guided Evolution for Grounded Medical QA Data Synthesis

    [https://arxiv.org/abs/2610.08093](https://arxiv.org/abs/2610.08093)

    SAGE提出了一种数据合成框架，利用MeSH等轻量级公开分类体系作为语义锚点，通过迭代交替进行原子与关联合成，使小型本地部署模型也能从极少种子数据生成高质量的医学问答训练数据。

    

    开发适用于临床任务（如医学问答QA）的可靠模型，严重受限于高质量、专家标注训练数据的稀缺。严格的隐私要求，以及在资源有限的临床环境中使用大型开源语料库或专有云端API并不现实，进一步加剧了这一挑战。为解决这些障碍，我们提出了SAGE（语义锚点引导演化，Semantic Anchor-Guided Evolution），这是一种新颖的数据合成框架，使小型、本地部署的模型能够生成高质量的医学训练数据。SAGE利用轻量级、公开可用的分类体系（如MeSH）作为语义锚点，施加结构化先验以有效引导并锚定数据生成过程。其核心在于，SAGE迭代地交替进行原子（基于单个概念的）合成与关联（基于关系的）合成，从极少量种子数据出发自举生成训练数据。

    arXiv:2610.08093v1 Announce Type: cross  Abstract: Developing reliable models for clinical tasks, such as Medical Question Answering (QA), is severely constrained by the limited availability of high-quality, expert-annotated training data. This challenge is exacerbated by stringent privacy requirements and the impracticality of utilizing large open-source corpora or proprietary cloud APIs within resource-limited clinical settings. To address these obstacles, we introduce SAGE (\textit{Semantic Anchor-Guided Evolution}), a novel data synthesis framework that enables small, locally deployed models to generate high-quality medical training data. SAGE leverages lightweight, publicly available taxonomies such as MeSH as semantic anchors, imposing a structured prior to effectively guide and ground the data generation process. At its core, SAGE iteratively interleaves atomic (individual concept-based) and associative (relation-based) synthesis, bootstrapping training data from minimal seeds. 
    
[^109]: 当计划改变答案：语义查询中成本-准确率优化的形式化

    When Plans Change Answers: Formalizing Cost-Accuracy Optimization for Semantic Queries

    [https://arxiv.org/abs/2610.08089](https://arxiv.org/abs/2610.08089)

    该论文首次形式化了语义查询的成本-准确率优化问题，利用决策模型的校准置信度和错误在连接中的传播加权，无需标注数据即可预测查询计划的输出质量，并能将输出层面的准确率目标反向转化为元组级别的定价，从而实现更优的查询计划选择。

    

    在语义查询引擎中，谓词由机器学习模型进行评估，查询计划的选择不仅影响查询的成本，还会影响其查询结果。现有系统要么对每个语义算子应用固定阈值，要么为每个算子单独调整准确率，而未能考虑错误如何通过连接操作进行传播。我们为此类查询的成本-准确率优化给出了一个形式化的问题定义。我们的出发点是决策模型（如Jev）赋予每个决策的校准置信度。它为每个决策产生一个预期错误；用每个决策对输出的贡献（在最简单的情况下即其扇出）对这些错误进行加权，无需任何标注数据即可得到查询计划的预期输出质量；而反向进行同样的计算，则可将输出层面的准确率目标转化为对每个基础元组或中间元组的“价格”。在此基础上，我们为带有语义算子的关系代数定义了预言机语义（oracle semantics）……（摘要原文在此处截断）

    arXiv:2610.08089v1 Announce Type: cross  Abstract: In semantic query engines, predicates are evaluated by machine-learned models, and the choice of a query plan affects not only the cost of a query but also its result. Existing systems either apply a fixed threshold to each semantic operator or tune accuracy per operator, without accounting for how errors propagate through joins. We give a formal problem definition for cost-accuracy optimization of such queries. Our starting point is the calibrated confidence that decision models such as Jev attach to each decision. It yields an expected error for every decision; weighting these errors by each decision's contribution to the output (in the simplest case, its fan-out) gives the expected output quality of a plan without any labeled data, and the same computation in reverse turns an output-level accuracy target into a price on each base or intermediate tuple. Building on this, we define an oracle semantics for relational algebra with seman
    
[^110]: POLAR：面向工具调用LLM智能体的本体引导式风险预防

    POLAR: Ontology-Guided Risk Prevention for Tool-Calling LLM Agents

    [https://arxiv.org/abs/2610.08082](https://arxiv.org/abs/2610.08082)

    POLAR是一个通过结构化两层本体评估操作可逆性的防护栏框架，能在工具调用LLM智能体执行高风险操作前将其剪除并提供可审计的结构化判定，但其收益因任务域和智能体能力而异。

    

    LLM工具使用智能体运行在动态环境中，其中许多操作带有运行风险。然而，大多数安全机制只在错误显现后才作出反应。现有的预防性方法要么通过思维链深思熟虑对智能体进行微调，要么将自然语言防护规则编译为运行时检查，但它们都没有提供结构化的、可审计的判定。我们提出了POLAR，一个针对小型工具调用智能体的防护栏框架，它通过结构化的两层本体来评估可逆性。POLAR通过推导候选逆操作序列，为每个操作分配一个分级的可逆性分数；未通过阈值的调用会在执行前被剪除。在τ²-bench上对六个智能体模型进行评估，POLAR在airline域中使六个智能体中的四个的平均任务奖励提高了0.11至0.18分，但在18个模型-域组合中只有8个总体上有所改善；retail域和更强的智能体往往出现性能回退。POLAR提供了一种可审计的结构化（摘要此处截断）

    arXiv:2610.08082v1 Announce Type: new  Abstract: LLM tool-use agents operate in dynamic environments where many actions carry operational risk. However, most safety mechanisms react only after errors manifest. Existing pre-emptive approaches either fine-tune the agent on chain-of-thought deliberation or compile natural-language guardrails into runtime checks, but they do so without exposing a structural, auditable verdict. We propose POLAR, a guardrail framework for small tool-calling agents that assesses reversibility through a structured two-layer ontology. POLAR assigns each action a graded reversibility score by deriving a candidate inverse sequence; calls failing a threshold are pruned before execution. Evaluated on $\tau^2$-bench across six agent models, POLAR improves mean task reward by 0.11 to 0.18 points on airline for four of six agents, but only eight of eighteen model--domain cells improve overall; retail and stronger agents often regress. POLAR provides an auditable struc
    
[^111]: 自我回溯蒸馏：将事后经验转化为先验预见

    Self-Retrospection Distillation: Turning Post-hoc Experiences into Prior Foresight

    [https://arxiv.org/abs/2610.08077](https://arxiv.org/abs/2610.08077)

    该论文提出前瞻学习与自我回溯蒸馏方法，将已完成的轨迹作为特权后见之明，蒸馏为同一策略在交互前的轨迹无关的预见性预测，从而在组相对奖励信号消失时仍能从经验中有效学习。

    

    具有可验证奖励的强化学习（RLVR）主要通过交互后的标量结果奖励将智能体经验转化为学习信号。然而，对于组相对目标而言，当所有采样轨迹获得相同奖励时，这一信号便会消失，即使这些轨迹可能揭示了关于任务需求以及智能体如何失败的有用信息。我们提出了一个互补的问题：事后反思能否教会智能体在行动之前本可预见的东西？我们引入前瞻学习，利用事后经验来监督交互前视角下的预见性预测，并通过自我回溯蒸馏（SRD）加以实例化。直观地说，一条已完成的轨迹揭示了本会有用的知识和本应避免的陷阱；SRD将这种特权的后见之明蒸馏到同一策略的、不依赖轨迹的前瞻预测中。前瞻仅作为训练目标，无需成为……

    arXiv:2610.08077v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) turns agent experience into learning signals primarily through scalar outcome rewards after interaction. For group-relative objectives, however, this signal vanishes when all rollouts receive the same reward, even though their trajectories may reveal useful information about what the task requires and how the agent fails. We ask a complementary question: can hindsight teach an agent what it could have anticipated before acting? We introduce prospective learning, which uses post-hoc experience to supervise foresight predictions from the pre-interaction view, and instantiate it with Self-Retrospection Distillation (SRD). Intuitively, a completed trajectory reveals knowledge that would have been useful and pitfalls that should be avoided; SRD distills this privileged hindsight into trajectory-blind foresight of the same policy. Foresight serves only as a training target and need not be e
    
[^112]: SpeedrunBench：用电子游戏速通挑战大语言模型智能体

    SpeedrunBench: Challenging LLM Agents with Video Game Speedrunning

    [https://arxiv.org/abs/2610.08076](https://arxiv.org/abs/2610.08076)

    该论文提出了SPEEDRUNBENCH基准，通过9款电子游戏的速通任务，评估大语言模型智能体自主形成复杂策略、超越人类已知解法的能力。

    

    前沿大语言模型智能体已被证明能够解决日益复杂的、人类拥有可衡量解法的任务。这引出了一个关键问题：大语言模型智能体能否超越人类已经解决的范畴？当人类业已成熟的解决方案对于我们所缺乏背景知识或足够训练数据的问题不再适用时，发展复杂策略以应对重大问题的能力就变得至关重要。我们通过电子游戏速通这一社区实践来研究智能体形成此类策略的能力。在速通中，参与者竞相在特定条件下以最快速度完成一款电子游戏，并在此过程中发掘出需要对底层游戏机制有透彻理解与掌握的有趣且非传统的玩法。我们提出了SPEEDRUNBENCH，一个在9款不同游戏上评估前沿大语言模型智能体的基准测试。要在此基准中取得良好表现……（原文摘要至此截断）

    arXiv:2610.08076v1 Announce Type: new  Abstract: Frontier LLM agents have been shown to be capable of solving increasingly complex tasks for which humans have measurable solutions. This begs the pertinent question of whether LLM agents can go beyond what humans have already solved. The ability to develop sophisticated strategies to tackle consequential problems becomes paramount as well-trodden, human-developed solutions become insufficient for problems for which we lack context or enough training data. We study agents' capability of such strategy formation through the communal practice of video game speedrunning. In speedrunning, practitioners compete to find the fastest way to complete a video game under certain conditions, and in so doing uncovering interesting unorthodox play styles that require a thorough understanding and mastery of the underlying game mechanics. We introduce SPEEDRUNBENCH, a benchmark that evaluates frontier LLM agents across 9 different games. To perform well i
    
[^113]: DAEDALUS：从自生成任务中引导代理记忆

    DAEDALUS: Bootstrapping Agent Memory from Self-Generated Tasks

    [https://arxiv.org/abs/2610.08048](https://arxiv.org/abs/2610.08048)

    DAEDALUS通过探索者代理自生成练习任务、解决者代理从失败中提炼启发式规则并验证其有效性，从而在无需现有任务或人工验证器的情况下自动构建可复用的代理记忆。

    

    LLM代理通常缺乏在新环境中可靠行动所需的操作性知识，因为它们必须自行发现特定的工具行为或环境约定。由于没有对过去尝试的记忆，它们会在不同任务中重复相同的错误，导致更多的任务失败和更长的轨迹。为了解决这个问题，代理系统通常依赖人工编写的指南，或者依赖从训练任务和神谕验证器构建的程序性记忆，但这两种方式都需要对环境的先验知识。我们提出了DAEDALUS，这是一种在没有现成任务或神谕验证器的情况下，从自生成练习中引导可复用代理记忆的方法。DAEDALUS将两个代理配对：一个是与环境交互以生成具有挑战性但可解决任务的探索者，另一个是尝试解决这些任务的解决者。每次解决者失败后都会推导出一个启发式方法，只有当解决者在上下文中借助该启发式方法反复成功后，该启发式方法才会被接受。

    arXiv:2610.08048v1 Announce Type: new  Abstract: LLM agents often lack the operational knowledge to act reliably in new environments, as they must discover specific tool behaviors or environment conventions on their own. Without memory of past attempts, they repeat the same mistakes across tasks, leading to more task failures and longer trajectories. To address this, agentic systems typically rely on human-written guidelines or on procedural memory built from training tasks and an oracle verifier, both of which require prior knowledge of the environment. We present DAEDALUS, a method for bootstrapping reusable agent memory from self-generated practice without existing tasks or oracle verifiers. DAEDALUS pairs two agents: an explorer that interacts with the environment to generate challenging yet solvable tasks, and a solver that attempts them. A heuristic is derived from each solver failure and accepted only after the solver repeatedly succeeds with that heuristic in context. These out
    
[^114]: 相同的反馈，不同的答案：测量前沿模型客户反馈分析中的运行间不稳定性

    Same Feedback, Different Answer: Measuring Run-to-Run Instability in Frontier-Model Customer Feedback Analysis

    [https://arxiv.org/abs/2610.08036](https://arxiv.org/abs/2610.08036)

    本文提出了一个重复运行评估框架，通过对齐语义等价类别并测量主题流失与数量分歧两个指标，系统量化了八个前沿模型在不同语料规模、提示词和执行设计下客户反馈分析结果的运行间不稳定性。

    

    AI 智能体正日益被编程用于对大量非结构化数据集合执行知识工作自动化。这种自动化需要可重复性：当底层证据不变时，智能体的类别、优先级和计数在多次运行之间不应发生实质性变化，即使每个单独的答案看起来都是合理的。我们引入了一个重复运行评估框架，该框架对齐语义等价的类别，并关注两个运行指标：主题流失（theme churn，即返回类别集合的归一化变化）和数量分歧（volume disagreement，即持续存在类别的计数变化）。我们在八个前沿模型上评估了三项常见的客户反馈任务，语料库规模从 100 到 5,000 条记录，使用多个提示词和三种执行设计：原始生成、无分类体系的层次分解，以及基于分类体系的智能体（TGA），后者使用持久化的主题、子主题和记录级预测。使用 Claude

    arXiv:2610.08036v1 Announce Type: new  Abstract: AI agents are increasingly being programmed to automate knowledge work over large collections of unstructured data. Such automation requires repeatability: when the underlying evidence is unchanged, the agent's categories, priorities, and counts should not shift materially between runs, even if each individual answer appears plausible. We introduce a repeat-run evaluation framework that aligns semantically equivalent categories and focuses on two operating metrics: theme churn, the normalized change in the returned category set, and volume disagreement, the change in counts for categories that persist. We evaluate three recurring customer-feedback tasks across eight frontier models, corpus sizes from 100 to 5,000 records, multiple prompts, and three execution designs: raw generation, taxonomy-free hierarchical decomposition, and a taxonomy-grounded agent (TGA) using persistent themes, subthemes, and record-level predictions. With Claude 
    
[^115]: 在梦中学习，在现实中获胜：面向十英雄MOBA游戏的连续Dyna循环

    Learning in Dreams, Winning in Reality: A Continuous Dyna Loop for a Ten-Hero MOBA

    [https://arxiv.org/abs/2610.08033](https://arxiv.org/abs/2610.08033)

    该研究提出一个连续异步Dyna循环框架，仅在十英雄MOBA游戏的结构化多智能体世界模型中训练策略，使天辉方真实对局胜率从循环前的33.7%提升至70.2%，而真实游戏仅用于提供数据和评估、从不提供梯度。

    

    世界模型通常从内部来评判：通过预测损失、通过策略在想象中获得的回报，或者通过其生成的画面看起来有多逼真。我们从外部来评判世界模型。我们为一个完整的十英雄MOBA游戏（206个单位，每个英雄在每个时间步都行动，单局游戏长达6,000个时间步）学习了一个结构化的多智能体世界模型，仅在该世界模型内部使用1,400时间步的自由运行想象回合训练策略，然后在真实游戏中将该策略与游戏自带的内置对手进行测试。真实游戏从不提供梯度；它提供策略自己生成的对局作为世界模型的训练数据，以及用于选择和锚定策略的在线评估。以连续异步Dyna循环的方式运行，该策略作为天辉方赢得了70.2%的真实对局（600场中获胜421场；95%置信区间66.4-73.7），测试所用种子从未用于任何决策，相比仅梦境训练时的0%和循环启动前的33.7%大幅提升。它作为夜魇方没有赢得任何对局，但游戏自带的内置对手同样如此。

    arXiv:2610.08033v1 Announce Type: new  Abstract: World models are usually judged from the inside: by prediction loss, by the return a policy earns in imagination, or by how convincing their frames look. We judge one from the outside. We learn a structured, multi-agent world model of a complete ten-hero MOBA (206 units, every hero acting every tick, games of up to 6,000 ticks), train a policy only inside it with 1,400-tick free-running imagined episodes, and measure that policy in the real game against the opponent the game ships with. The real game never provides a gradient; it provides the policy's own games as training data for the world model, and an online evaluation that selects and anchors the policy. Run as a continuous asynchronous Dyna loop, the policy wins 70.2% of real games as radiant (421 of 600; 95% CI 66.4-73.7) on seeds never used for any decision, up from 0% for dream training alone and 33.7% before the loop. It wins none as dire, and neither does the shipped opponent 
    
[^116]: TICDA：表格化上下文数据归因

    TICDA: Tabular In-Context Data Attribution

    [https://arxiv.org/abs/2610.07996](https://arxiv.org/abs/2610.07996)

    提出TICDA方法，通过在表格基础模型潜在表示上训练的线性代理模型直接量化上下文中每个示例对预测的影响，解决了重采样和基于梯度的数据归因方法在上下文学习场景中失效的问题。

    

    表格基础模型（TFMs）通过以上下文中提供的带标签示例为条件，在无需任何参数更新的情况下实现了强大的预测性能。然而，单个示例如何影响给定的预测仍然知之甚少。这一差距在实践中很重要：上下文通常由任何可用的带标签数据组装而成，可能导致混入错误标注、冗余或低质量的示例，从而降低性能。标准的数据归因方法无法直接迁移到TFM场景：基于重采样的方法（如DemoShapley）需要组合级数量的前向传播，而基于梯度的估计器（如影响函数）需要计算训练点对模型参数的影响，但上下文学习从不更新参数。我们提出了TICDA，一种直接从在TFM潜在表示上训练的线性代理模型中测量上下文中每个示例影响的方法。

    arXiv:2610.07996v1 Announce Type: cross  Abstract: Tabular foundation models (TFMs) achieve strong predictive performance by conditioning on labeled demonstrations provided in context, without any parameter update. Yet how individual demonstrations shape a given prediction remains poorly understood. This gap matters in practice: the context is often assembled from whatever labeled data is available, potentially leading to the inclusion of mislabeled, redundant, or low-quality examples that degrade performance. Standard data attribution methods do not transfer to the TFM setting: resampling-based approaches such as DemoShapley require a combinatorial number of forward passes, and gradient-based estimators such as influence functions require computing training point's effect on the model parameters, which in-context learning never updates. We introduce TICDA, a method that measures the influence of every demonstration in the context directly from linear surrogates trained on TFM latent e
    
[^117]: VisionWeave：将弹性视觉表示编织作为多模态大语言模型的固有能力

    VisionWeave: Weaving Elastic Visual Representations as a Native Capability of MLLMs

    [https://arxiv.org/abs/2610.07987](https://arxiv.org/abs/2610.07987)

    VisionWeave通过大规模训练，使多模态大语言模型获得按内容自适应地决定视觉表示位置与粒度的“弹性视觉表示编织”固有能力，在保留关键细节的同时显著降低计算成本。

    

    多模态大语言模型（MLLM）已成为视觉理解的主导范式，但其将输入编码为密集、固定大小的patch token会带来巨大开销。然而，视觉信息的分布是不均匀的：某些区域需要细粒度的细节，而其他区域则可以使用紧凑的表示。下采样会牺牲这些细节，而现有的token剪枝和自适应方法在内容自适应粒度、任务泛化以及与现代MLLM和服务基础设施的集成方面仍然受限。克服这些限制需要基础模型能够端到端地学习在哪里以及以何种粒度分配视觉表示，我们将这一固有能力称为弹性视觉表示编织。我们提出了VisionWeave，通过大规模训练在前沿水平的MLLM中建立这种能力。它结合了两个组件：门控空间池化器构建粗粒度表示……

    arXiv:2610.07987v1 Announce Type: cross  Abstract: Multimodal large language models have become the dominant paradigm for visual understanding, but incur substantial costs by encoding inputs into dense, fixed-size patch tokens. However, visual information is unevenly distributed: some regions require fine-grained detail, while others admit compact representations. Downsampling sacrifices this detail, while existing token pruning and adaptive approaches remain limited in content-adaptive granularity, task generalization, and integration with modern MLLMs and serving infrastructure. Overcoming these limitations calls for foundation models that learn, end to end, where-and at what granularity-to allocate visual representations, a native capability we term elastic visual representation weaving. We introduce VisionWeave, establishing this capability in frontier-level MLLMs through large-scale training. It combines two components: a gated spatial pooler constructs coarse-grained representati
    
[^118]: 先判断再看：学习哪些被检索的记忆值得占用像素

    Decide Before You Look: Learning Which Retrieved Memories Deserve Pixels

    [https://arxiv.org/abs/2610.07984](https://arxiv.org/abs/2610.07984)

    提出 PixelTriage——一个置于检索之后的轻量级插件，通过阅读对话、笔记和缩略图即可在回答模型运行前预测哪些被检索记忆的像素真正有用，从而大幅削减视觉 token 成本并保持回答精度。

    

    多模态助手需要从包含图像的长期记忆中回答问题。在检索之后，每张被检索的图像要么以像素形式（每张约一千个视觉 token）送达回答模型，要么以存储的文本代理形式送达，而文本代理常常缺失问题所询问的细节。我们发现，像素带来的收益通常仅来自一两条被检索的记忆，并且这种收益可以在回答模型运行之前就被预测出来，而无需读取任何全分辨率图像。PixelTriage 是一个置于检索之后的插件，它是一个不生成文本的小型模型，通过阅读对话、简短笔记以及每条被检索记忆的缩略图，来预测其像素能带来多大增益。它的训练数据来自由一个冻结的 27B 模型标注的合成记忆情节，该模型分别在有和没有每条记忆像素的情况下回答每个问题，从而提供标签。搭配 7B 回答模型时，PixelTriage 在 M³Exam、DMV 和 MemEye 上处于精度-成本前沿，并仅使用 11--（摘要原文在此处截断）。

    arXiv:2610.07984v1 Announce Type: cross  Abstract: Multimodal assistants answer questions from long-term memories that contain images. After retrieval, each retrieved image reaches the answering model either as pixels, at about a thousand visual tokens per image, or as a stored text proxy that often misses the detail the question asks about. We find that the benefit of pixels usually comes from one or two retrieved memories, and that it can be predicted before the answering model runs, without reading any full-resolution image. In PixelTriage, a plug-in placed after retrieval, a small model that does not generate text reads the dialogue, a short note and a thumbnail of each retrieved memory and predicts how much its pixels would add. It is trained on synthetic memory episodes labeled by a frozen 27B model that answers each question with and without each memory's pixels. With a 7B answering model, PixelTriage lies on the accuracy--cost frontier of M$^3$Exam, DMV and MemEye and uses 11--
    
[^119]: 从修订后果中学习：面向自我改进智能体的后见之明元经验蒸馏

    Learning from Revision Consequences: Hindsight Meta-Experience Distillation for Self-Improving Agents

    [https://arxiv.org/abs/2610.07979](https://arxiv.org/abs/2610.07979)

    该论文提出HMED（后见之明元经验蒸馏），通过后见之明地分析技能修订带来的实际后果来构建元经验，将修订效应与初始发现状态的效应解耦，从而更准确地改进自我改进智能体的元技能。

    

    当智能体通过生成和修订“技能”不断自我改进时，发现和精炼这些技能的过程本身也成为一个可学习的对象。“任务技能”直接作用于任务执行，而“元技能”则支配智能体如何发现并改进未来的技能；因此，元技能的价值体现在它们所引发的后续搜索过程之中。现有方法从观察到的原始技能搜索轨迹及分支结果中改进元技能。然而，分支性能将初始发现状态与产生该搜索过程的元技能修订两方面的效应纠缠在一起，使得难以刻画某次修订究竟改变了什么，并使更新偏向于那些受益于有利状态的修订，而非真正改进过程的修订。我们提出了HMED（Hindsight Meta-Experience Distillation，后见之明元经验蒸馏），一种为自我改进智能体构建元经验的机制。HMED通过后见之明地……（原文摘要在此处截断）

    arXiv:2610.07979v1 Announce Type: new  Abstract: As agents continuously improve by generating and revising Skills, the process that discovers and refines those Skills becomes a learnable object in its own right. Task-Skills directly act on task execution, whereas Meta-Skills govern how agents discover and improve future Skills; their value therefore emerges through the subsequent search processes they induce. Existing approaches improve Meta-Skills from observed raw Skill-search trajectories and branch outcomes. However, branch performance entangles the effects of the initial discovery state and the Meta-Skill revision that generated the search process, making it difficult to characterize what a particular revision actually changed, and pushing updates toward revisions that benefit from favorable states rather than those that improve the process. We introduce HMED (Hindsight Meta-Experience Distillation), a mechanism for constructing Meta-Experience for self-improving agents. HMED revi
    
[^120]: 智能体能否为所有人服务？个性化用户界面中移动GUI智能体的跨用户可靠性

    Can Agents Work for Everyone? Cross-User Reliability for Mobile GUI Agents in Personalized User Interfaces

    [https://arxiv.org/abs/2610.07972](https://arxiv.org/abs/2610.07972)

    该论文提出PAIR流水线与RePAIR强化学习训练方法，首次系统评估并提升移动GUI智能体在不同用户个性化界面上的跨用户可靠性，揭示出智能体在用户条件化环境中的任务成功率存在显著下降。

    

    移动GUI智能体日益在受用户历史与偏好影响的界面上运行，但它们在不同用户之间的可靠性仍未得到充分探索。我们提出了PAIR（个性化应用状态实例化与渲染），这是一个构建用户条件化应用状态的流水线，能够对不同用户执行相同任务进行受控评估。我们进一步提出了RePAIR（具有个性化感知交互奖励的强化学习），这是一种从子目标结果的跨用户差异中学习的训练方法，旨在提升用户条件化移动环境下的可靠性。通过对六个智能体的实验，我们发现不同用户之间的任务成功率存在显著差异，且在用户条件化UI环境中的子目标达成率持续偏低（下降6.98至15.4个百分点）。对于从每个用户自身内容中提取的个人目标，这一差距进一步扩大（8.77至22.0个百分点）。在这些情境下的失败频繁……（原文摘要在此处被截断）

    arXiv:2610.07972v1 Announce Type: new  Abstract: Mobile GUI agents increasingly operate on interfaces influenced by users' histories and preferences, but their reliability across different users remains underexplored. We introduce PAIR (Personalized Application-state Instantiation and Rendering), a pipeline for constructing user-conditioned application states that enables controlled evaluation of the same task across different users. We further introduce RePAIR (Reinforcement learning with Personalization-Aware Interaction Rewards), a training approach that learns from cross-user differences in subgoal outcomes to improve reliability across user-conditioned mobile environments. Across six agents, we find substantial variation in task success across users and consistently lower subgoal achievement in user-conditioned UI contexts (6.98 to 15.4 pp). This gap further increases for personal targets drawn from each user's own content (8.77 to 22.0 pp). Failures in these contexts frequently i
    
[^121]: ReGraph：对“什么”与“哪里”双视觉通路中涌现泛化能力的计算性解释

    ReGraph: A Computational Account of Emergent Generalization in the "what" and "where" Dual Visual Streams

    [https://arxiv.org/abs/2610.07962](https://arxiv.org/abs/2610.07962)

    提出ReGraph——一个具有生物归纳偏置的循环双流图模型，在计算层面解释了情境不变的关系结构（即泛化能力）如何沿腹侧“什么”与背侧“哪里”双视觉通路涌现形成。

    

    泛化能力——即提取情境不变关系结构的能力——最初在哪里涌现，是人工智能与神经科学领域的核心问题之一。这一能力的基础位于海马上游的内嗅皮层，在那里，平行通路将内侧内嗅皮层（MEC）中的关系结构与外侧内嗅皮层中的感觉内容分离开来。然而，正如Eichenbaum所论证的，这种因式分解可能起源更早，由背侧（“哪里”）与腹侧（“什么”）视觉通路的分离所驱动。支持这一观点的是，网格样放电模式——MEC（情境不变编码）的标志性特征——也出现在背侧通路更上游的新皮层区域。然而，此类表征如何沿上游通路以计算方式形成，至今仍不清楚。为了在计算机中模拟研究这一问题，我们开发了ReGraph——一个具有生物归纳偏置的循环双流图模型……（原文摘要在此处截断）

    arXiv:2610.07962v1 Announce Type: cross  Abstract: Where generalization capacity--the ability to extract context-invariant relational structures--first emerges remains a central question in AI and neuroscience. The foundation for this capacity lies upstream of the hippocampus, within the entorhinal cortex, where parallel pathways dissociate relational structure in the medial entorhinal cortex (MEC) from sensory content in the lateral entorhinal cortex. However, as Eichenbaum argued, such factorization likely originates earlier, driven by the segregation of the dorsal ('where') and ventral ('what') visual streams. Supporting this, grid-like firing patterns--a signature of MEC (context-invariant codes)--also appear in preceding neocortical regions along the dorsal pathway. Yet, how such representations are computationally formed along upstream pathways remains unknown. To investigate this in silico, we developed ReGraph, a recurrent dual-stream graph model with biological inductive biase
    
[^122]: 置信推理图：面向大语言模型智能体的结构化置信度估计

    Confidence Reasoning Graphs: Structured Confidence Estimation for LLM Agents

    [https://arxiv.org/abs/2610.07948](https://arxiv.org/abs/2610.07948)

    提出置信推理图（CRG），一种推理时框架，通过结构化分解单条轨迹中的证据来估计LLM智能体完成任务的概率，无需访问模型内部信号或训练数据。

    

    在重要领域中LLM智能体时，要对是否信任其输出或进行干预做出明智决策，需要对智能体任务成功与否具备经过校准的置信度。智能体的置信度估计十分困难，因为关于成功的证据分散在智能体轨迹中异构且相互依赖的多个步骤之中。实际的智能体部署还带来了更多挑战：前沿大语言模型通常对内部信号的访问受限，智能体的多次运行成本高昂，且训练数据可能不可用或很快过时。为了应对这些挑战，我们提出了置信推理图，这是一个推理时框架，能够从单条轨迹中估计智能体完成任务的概率，而无需特权级别的模型访问权限或训练数据。CRG并非将执行过程压缩为单一的总体判断，而是从“智能体完成了任务”这一论断出发，将其分解为……（原文摘要在此处截断）

    arXiv:2610.07948v1 Announce Type: new  Abstract: When using an LLM agent in a consequential domain, making an informed decision about whether to trust its output or intervene requires calibrated confidence in the agent's success. Confidence estimation for agents is difficult because evidence about success is distributed across heterogeneous, interdependent steps of an agent's trajectory. Practical agentic deployments introduce further challenges: frontier LLMs often provide limited access to internal signals, agent roll-outs are costly, and training data may be unavailable or quickly become outdated. To address these challenges, we introduce Confidence Reasoning Graphs (CRGs), an inference-time framework that estimates the probability an agent accomplished its task from a single trajectory, without privileged model access or training data. Rather than compressing an execution into a single holistic judgment, a CRG begins with the claim that the agent accomplished its task, decomposes i
    
[^123]: 使视觉-语言-动作模型适应执行过程中的未知视觉干扰

    Adapting Vision-Language-Action Models to Unknown Visual Disruptions During Execution

    [https://arxiv.org/abs/2610.07946](https://arxiv.org/abs/2610.07946)

    提出SALT方法，将上一动作块中未执行的剩余轨迹作为自监督信号，通过过渡锚定与顺序修正传播，使视觉-语言-动作模型能够在机器人执行任务时对未知视觉干扰进行测试时自适应。

    

    视觉干扰可能在机器人执行任务的过程中出现，使得视觉-语言-动作（VLA）策略在不知道干扰类型或发生时间的情况下必须做出响应。我们提出了基于剩余轨迹的自监督适应方法，它将剩余轨迹——即上一个动作块中未执行的部分——作为测试时自适应的自监督信号。由于连续的动作块在时间上相互重叠，剩余轨迹为当前预测在相同的未来控制区间上提供了时间对齐的目标。在视觉变化发生时，剩余轨迹能够保留干扰发生之前所形成的计划，因此将策略向其更新可以在视觉变化发生时锚定适应过程（过渡锚定）。SALT保留适应后的策略并重新生成当前动作块，其剩余轨迹将成为下一次重新规划时的目标，从而将修正沿执行轨迹向前传递（顺序修正传播）。

    arXiv:2610.07946v1 Announce Type: cross  Abstract: Visual disruptions can arise while a robot is executing a task, leaving a vision-language-action (VLA) policy to respond without knowing the disruption type or timing. We introduce Self-supervised Adaptation from Leftover Trajectories (SALT), which uses the leftover trajectory, the unexecuted part of the previous action chunk, as self-supervision for test-time adaptation. Because consecutive chunks overlap in time, the leftover provides a temporally aligned target for the current prediction over the same future control interval. At the onset of a visual shift, the leftover can retain a plan formed before the corruption, so updating the policy toward it anchors the adaptation across the shift (Transition Anchoring). SALT keeps the adapted policy and regenerates the current chunk, whose leftover becomes the target at the next replan, carrying the correction forward along the execution trajectory (Sequential Correction Propagation). Super
    
[^124]: 循环语言模型的混合潜在注意力

    Hybrid Latent Attention for Looped Language Models

    [https://arxiv.org/abs/2610.07940](https://arxiv.org/abs/2610.07940)

    提出混合潜在注意力（HLA），通过将旧 token 压缩为紧凑的潜在表示而非存储完整键值缓存，使循环语言模型的缓存缩小 10.7 倍、每 GPU 并发序列容量提升 4.0-8.8 倍、解码吞吐量最高提升 7.4 倍，同时保留原模型 97% 以上的性能。

    

    循环语言模型对每个 token 重复应用同一组层 T 次，这在不增加参数的情况下加深了模型，但使其键值（KV）缓存膨胀了 T 倍。更大的缓存限制了 GPU 一次能解码的序列数量，并且由于每步解码都需要读取整个缓存，从而减慢了解码速度。我们提出了混合潜在注意力（HLA），它在最近 W 个 token 的滑动窗口内保留精确的键和值，并将每个较旧的 token 存储为一个紧凑的潜在表示，每个循环的查询可以直接读取该表示，而无需重建键和值。我们在具有 1.4B 和 2.6B 参数的 Ouro 循环模型（T=4）上对 HLA 进行再训练，保持预训练权重冻结，仅训练新增参数以复现原始注意力。每个 token 的缓存缩减 10.7 倍，使每个 GPU 能容纳 4.0-8.8 倍的并发序列，解码吞吐量在 1K token 上下文中提升 2.5 倍，在 16K 时最高提升 7.4 倍。HLA 保留了原模型超过 97% 的性能。

    arXiv:2610.07940v1 Announce Type: cross  Abstract: Looped language models apply the same stack of layers T times to each token, which deepens the model without adding parameters but multiplies its key-value (KV) cache by T. The larger cache limits how many sequences a GPU can decode at once and slows each decoding step, which reads the whole cache. We propose Hybrid Latent Attention (HLA), which keeps exact keys and values within a sliding window of W recent tokens and stores each older token as a compact latent that the query of each loop reads directly, without reconstructing keys and values. We uptrain HLA on Ouro looped models (T=4) with 1.4B and 2.6B parameters, keeping the pretrained weights frozen and training only the added parameters to reproduce the original attention. The cache shrinks by 10.7x per token, fitting 4.0-8.8x as many concurrent sequences per GPU, and decoding throughput improves by 2.5x at 1K-token contexts and by up to 7.4x at 16K. HLA retains over 97% of the o
    
[^125]: SIGMA：基于模型规范的自我改进对齐泛化

    SIGMA: Self-Improving Alignment Generalization from a Model Spec

    [https://arxiv.org/abs/2610.07935](https://arxiv.org/abs/2610.07935)

    提出SIGMA数据生成与训练流程，仅需一份模型规范即可让大语言模型利用自身推理能力实现安全对齐的自我改进，并能泛化到分布外场景。

    

    大语言模型智能体日益能够执行复杂任务，并在软件工程和数学等易于验证的目标上进行递归式自我改进。由于对齐远比能力更难验证，这带来了能力增长而缺乏相应安全对齐的风险，尤其当能力扩展到自动化研究和网络安全领域时。现有方法要么聚焦于利用可验证反馈进行能力自我改进，要么依赖更强模型或人工精选数据进行对齐训练，从而形成了对齐的外部监督瓶颈。我们探讨当前模型能否改进自身的安全对齐，并提出了SIGMA——一个使对齐自我改进能够泛化到分布外场景的数据生成与训练流程。仅需一份描述模型期望行为的“模型规范”，SIGMA便利用模型的推理能力来强化其……（原文摘要在此处截断）

    arXiv:2610.07935v1 Announce Type: new  Abstract: LLM agents are increasingly capable of executing complex tasks and of recursively improving themselves on easy-to-verify objectives such as software engineering and mathematics. Since alignment is much harder to verify, this creates a growing risk of capabilities increasing without appropriate safety alignment, especially as capabilities expand to auto-research and cybersecurity. Existing approaches focus on capability self-improvement using verifiable feedback or on alignment training with supervision from stronger models or curated data, creating an external supervision bottleneck for alignment. We ask whether current models can improve their own safety alignment, and propose SIGMA, a data generation and training pipeline enabling alignment self-improvement that generalizes to out-of-distribution settings. Given only a "Model Spec" stating the model's desired behavior, SIGMA leverages a model's reasoning capabilities to strengthen its 
    
[^126]: 多模态学习的动态对齐与校准

    Dynamic Alignment and Calibration for Multimodal Learning

    [https://arxiv.org/abs/2610.07928](https://arxiv.org/abs/2610.07928)

    提出了对齐与校准驱动的多模态学习框架 ACML，通过动态跨模态三元组对齐模块解决静态对齐导致的过度对齐问题，并在融合中充分考虑模态间的特征幅值与置信度差异。

    

    动态多模态学习旨在通过自适应地建模跨模态间的信息差异来学习鲁棒的表示。然而，现有方法仍存在两个局限性：（i）静态的跨模态对齐策略通常对所有样本施加统一的约束，而忽视了样本间的差异，可能导致不合理的过度对齐；（ii）基于置信度或不确定性的融合方法往往未能充分考虑跨模态间的特征幅值与置信度差异。对于特征幅值差异显著或置信度差距较小的模态对，严格依据置信度来对齐融合权重可能并不可靠。为解决这些问题，我们提出了一种对齐与校准驱动的多模态学习框架（ACML）。具体而言，ACML 引入了一个动态跨模态三元组对齐模块，该模块对……（摘要在此处被截断，仅翻译至可见部分）强制实现强语义一致性

    arXiv:2610.07928v1 Announce Type: cross  Abstract: Dynamic multimodal learning aims to learn robust representations by adaptively modeling information discrepancies across modalities. However, existing methods still suffer from two limitations: (i) static cross-modal alignment strategies usually impose uniform constraints on all samples while overlooking sample-wise variations, potentially leading to unreasonable over-alignment; and (ii) confidence- or uncertainty-aware fusion methods often fail to adequately account for feature magnitude and confidence differences across modalities. For modality pairs with significant feature magnitude differences or small confidence gaps, it might be unreliable to strictly align fusion weights according to confidence. To address these issues, we propose an Alignment- and Calibration-driven Multimodal Learning framework (ACML). Specifically, ACML incorporates a dynamic cross-modal triplet alignment module, which enforces strong semantic consistency fo
    
[^127]: 基于多模态知识蒸馏的全切片图像胃腺癌分类方法

    Multimodal Knowledge Distillation for Gastric Adenocarcinoma Classification from Whole-Slide Images

    [https://arxiv.org/abs/2610.07913](https://arxiv.org/abs/2610.07913)

    提出了一种多模态知识蒸馏框架，利用低秩多模态融合将WSI图像与病理文本融合训练教师模型，再将知识蒸馏至仅需图像输入的学生模型，以较低的计算成本实现准确的胃腺癌亚型分类。

    

    胃腺癌（GA）是全球癌症相关死亡的主要原因之一，从全切片图像（WSI）中进行准确的组织病理学亚型分类对于制定有效的治疗方案至关重要。虽然将病理报告文本与WSI相结合的多模态方法可以提升分类效果，但现有方法通常依赖于计算开销高昂的Transformer架构和大语言模型。我们提出了一种多模态知识蒸馏（MKD）框架，该框架结合预训练的WSI图像编码器和临床文本编码器，并利用低秩多模态融合（LMF）在训练过程中高效建模跨模态交互。每张WSI被表示为图像块集合，并配以切片级别的诊断描述文本。教师模型学习融合的图像-文本表示以进行亚型分类，而学生模型通过蒸馏这些知识，实现仅基于图像的准确推理。我们评估……（原文在此处截断）

    arXiv:2610.07913v1 Announce Type: cross  Abstract: Gastric adenocarcinoma (GA) is a leading cause of cancer-related mortality worldwide, and accurate histopathological subtype classification from whole-slide images (WSIs) is essential for effective treatment planning. While multimodal approaches that integrate pathology report text with WSIs can improve classification, existing methods often depend on computationally expensive transformer architectures and large language models. We propose a multimodal knowledge distillation (MKD) framework that combines a pretrained WSI image encoder and a clinical text encoder using Low-Rank Multimodal Fusion (LMF) to efficiently model cross-modal interactions during training. Each WSI is represented as a bag of patches paired with a slide-level diagnostic caption. The teacher model learns fused image-text representations for subtype classification, while the student model distills this knowledge to enable accurate image-only inference. We evaluate o
    
[^128]: 基于控制的动态优化的多样化运动定制

    Diverse Motion Customization via Control-based Dynamic Optimization

    [https://arxiv.org/abs/2610.07911](https://arxiv.org/abs/2610.07911)

    提出基于随机最优控制的运动定制框架CMC，通过避免生成过程向参考视频坍缩来解决内容泄漏问题，使定制视频仅继承目标运动而外观完全由文本提示决定。

    

    尽管近期视频生成技术取得了进展，但运动定制仍然充满挑战，其原因在于内容泄漏问题——即参考视频的外观属性会无意间传播到生成的输出中。我们将这一问题识别为生成过程向参考视频坍缩的结果，这种坍缩源于将学习目标表述为对参考视频的直接回归。为解决这一问题，我们提出了基于控制的运动定制，这是一个在结构上对内容泄漏具有鲁棒性的原则性训练框架。我们的核心思想是在引导生成动态朝向期望运动的同时，避免向参考视频坍缩，并使用随机最优控制（SOC）对这一思想进行形式化。在这一框架下，定制化的视频能够获得目标运动，同时保持在预训练模型的提示条件分布内，其中外观由文本提示而非参考视频决定。

    arXiv:2610.07911v1 Announce Type: cross  Abstract: Despite recent advances in video generation, motion customization remains challenging due to content leakage, where appearance attributes from the reference video unintentionally propagate into the generated output. We identify this issue as a consequence of the generative process collapsing toward the reference video, which arises from formulating the learning objective as a direct regression on the reference. To address this, we propose Control-based Motion Customization (CMC), a principled training framework that is structurally robust to content leakage. Our key idea is to steer generative dynamics toward desired motion while avoiding collapse toward the reference video, which we formalize using Stochastic Optimal Control (SOC). Under this formulation, customized videos acquire the target motion yet remain within the pre-trained model's prompt-conditional distribution, where appearance is determined by the text prompt rather than t
    
[^129]: 连续记忆机器

    Continuous Memory Machines

    [https://arxiv.org/abs/2610.07907](https://arxiv.org/abs/2610.07907)

    提出连续记忆机（CMM），一种具有矩阵值短期与长期记忆状态的新型循环架构，通过Transformer联合更新两种记忆，实现了快速神经元级计算与长期信息保留的结合。

    

    循环神经网络通常将信息压缩到单一的向量形式循环状态中，迫使短期计算和长期保留共享同一表示。过去的扩展方法通过增加记忆容量或分离时间尺度来缓解这一瓶颈，但缺乏生物学中那种快速神经元级处理与较长时期保留相结合的特性。为此，我们提出了连续记忆机，这是一种具有矩阵形式短期和长期记忆状态的循环架构，两种记忆状态承担不同的功能角色。基于连续思维机（CTM），CMM的短期记忆追踪近期的神经活动，通过独特参数化的神经元级模型学习利用这些活动模式进行计算。持久的长期记忆则存储信息以备后续使用，一个Transformer共同更新这两个记忆存储，提供了一种富有表现力的双向读写机制。

    arXiv:2610.07907v1 Announce Type: new  Abstract: Recurrent neural networks typically compress information into a single vector-valued recurrent state, forcing short-term computation and long-term retention to share the same representation. Past extensions alleviate this bottleneck by increasing the memory capacity or separating timescales, but lack the combination of rapid neuron-level processing and longer-term retention found in biology. To that end, we introduce the Continuous Memory Machine (CMM), a recurrent architecture with matrix-valued short- and long-term memory states serving distinct functional roles. Building on the Continuous Thought Machine (CTM), the CMM's short-term memory tracks recent neural activity, with uniquely parameterized neuron-level models learning to use these activity patterns for computation. A persistent long-term memory stores information for later use, with a Transformer jointly updating both memory stores, providing an expressive bidirectional read--w
    
[^130]: 各向同性却不可解码：潜在预测文本表示中的序列内容充分性鸿沟

    Isotropic Yet Undecodable: The Sequential Content-Sufficiency Gap in Latent-Predictive Text Representations

    [https://arxiv.org/abs/2610.07906](https://arxiv.org/abs/2610.07906)

    论文通过信息论分解揭示序列内容充分性鸿沟，证明潜在表示的各向同性与一致性无法保证有序目标信息可解码，并据此提出引入规范词元监督的非自回归框架CANOPE，将位置信息恢复率从13.5%大幅提升至98.8%。

    

    我们通过考察一个表示是否保留了其输入中可获得的有序目标信息，来研究序列内容充分性问题。一种信息论分解方法将输入歧义、表示损失和读出失配三者分离开来。我们构造了可恢复的视图，其中完美一致性与联合各向同性高斯性可以与零目标信息共存，并确立了确定性规范锚点所施加的限制。词元对数损失提供了单边的信息损失界；固定惩罚的岭回归分析说明了为什么仅凭秩无法确定预测风险。这些结果催生了CANOPE——一个具有有序潜在画布、规范词元监督和几何正则化的非自回归框架。在40,000条验证序列上，潜在一致性（PL0）与词元锚定（PL2）具有几乎相同的合并秩，但在强自然损坏下分别仅达到13.5%和98.8%的位置Recall@1。

    arXiv:2610.07906v1 Announce Type: new  Abstract: We study sequential content sufficiency by investigating whether a representation retains the ordered target information available in its input. An information-theoretic decomposition separates input ambiguity, representation loss, and readout mismatch. We construct recoverable views where perfect agreement and joint isotropic Gaussianity coexist with zero target information, and establish limits imposed by deterministic canonical anchors. Token log-loss provides a one-sided information-loss bound; a fixed-penalty ridge analysis shows why rank alone cannot determine prediction risk. These results motivate CANOPE, a nonautoregressive framework with ordered latent canvases, canonical-token supervision, and geometric regularization. On 40,000 validation sequences, latent-agreement (PL0) and token-grounded (PL2) have nearly identical pooled ranks but reach 13.5% and 98.8% positional Recall@1, respectively, under strong natural corruption whe
    
[^131]: IEEE 802.11bx——WLAN智能组网（WIN）：迈向AI就绪的Wi-Fi 9

    IEEE 802.11bx - WLAN Intelligent Networking (WIN): Toward an AI-Ready Wi-Fi 9

    [https://arxiv.org/abs/2610.07900](https://arxiv.org/abs/2610.07900)

    本文综述了IEEE 802.11标准化进程中迈向AI就绪Wi-Fi 9的最新进展，创新性地从AI作为协议、平台和流量三个互补维度系统阐述了AI与Wi-Fi融合的候选特性与开放挑战。

    

    Wi-Fi 9有望超越单纯的通信功能，提供诸如感知或计算等新型服务。在这一关键节点，人工智能（AI）正在名为WLAN智能组网（WIN）的802.11bx修正案的制定中发挥主导作用。在本教程中，我们综述了IEEE 802.11标准化进程中Wi-Fi 9的最新进展，梳理了推动AI就绪的Wi-Fi 9的驱动因素与技术进步。随后，我们从三个互补维度考察了AI的角色，即AI作为协议（AI应用于Wi-Fi的物理层/介质访问控制层操作）、AI作为平台（将Wi-Fi基础设施重新用于提供AI计算服务），以及AI作为流量（AI流量对新的流量处理策略提出了需求），并针对每个维度讨论了候选特性和开放性挑战。作为“AI作为流量”范式的具体例证，我们提出了一个关于AI流量差异化处理的案例研究，探索了一种潜在的扩展方案。

    arXiv:2610.07900v1 Announce Type: cross  Abstract: Wi-Fi 9 is expected to go beyond mere communication and provide new services such as sensing or computation. At this juncture, Artificial Intelligence (AI) is taking a leading role in the definition of the 802.11bx amendment, named WLAN Intelligent Networking (WIN). In this tutorial, we survey the recent progress made toward Wi-Fi 9 within IEEE 802.11 standardization, tracing the drivers and technological advances that motivate an AI-ready Wi-Fi 9. We then examine AI's role along three complementary dimensions, i.e., AI as a protocol (AI is applied to Wi-Fi's PHY/MAC operation), AI as a platform (Wi-Fi infrastructure is repurposed to provide AI computation), and AI as traffic (AI flows call for new traffic-handling policies), and discuss candidate features and open challenges along each. As a concrete illustration of the AI as traffic paradigm, we present a case study on AI traffic differentiation, where we explore a potential extensio
    
[^132]: 面向稀疏长时程环境的方差厌恶 n 步离线强化学习

    Variance-Averse $n$-Step Offline Reinforcement Learning for Sparse Long-Horizon Environments

    [https://arxiv.org/abs/2610.07899](https://arxiv.org/abs/2610.07899)

    提出VAN-Flow框架，通过结合分类分布式评论家、方差厌恶期望算子和拒绝采样引导的流匹配生成式演员，在生成式离线强化学习中选择高回报且低方差 dispersion 的可靠动作。

    

    生成式演员正在变革离线强化学习（RL），它使策略类能够富有表现力地建模复杂的动作分布。然而，这种表现力也暴露了异构数据集中的一个关键挑战：生成式策略可能会复现不可靠的动作模式，其回报分布表现出高方差，偶尔因运气而获得高回报，但缺乏一致性。因此，仅最大化期望 Q 值不足以识别可靠的动作。我们提出了 VAN-Flow（Variance-Averse n-step Flow，方差厌恶 n 步流），这是一个在生成式离线强化学习中促进可靠动作的框架。VAN-Flow 结合了：(i) 分类分布式评论家，(ii) 方差厌恶期望算子，该算子平滑地重新加权原子概率，以偏好既具有高回报又具有低离散度的动作，以及 (iii) 通过拒绝采样引导的流匹配生成式演员。不同于 CVaR 或均值-方差方法……

    arXiv:2610.07899v1 Announce Type: cross  Abstract: Generative actors are transforming offline reinforcement learning (RL) by enabling expressive policy classes that model complex action distributions. However, this expressiveness also exposes a key challenge in heterogeneous datasets: generative policies can reproduce unreliable action modes whose return distributions exhibit high variance, occasionally yielding high returns by chance but lacking consistency. Consequently, maximizing the expected $Q$-value alone is insufficient for identifying reliable actions. We propose VAN-Flow (Variance-Averse $n$-step Flow), a framework that promotes reliable actions in generative offline RL. VAN-Flow combines (i) a categorical distributional critic, (ii) a variance-averse expectation operator that smoothly reweights atom probabilities to favor actions with both high returns and low dispersion, and (iii) a flow-matching generative actor guided via rejection sampling. Unlike CVaR or mean-variance o
    
[^133]: 面向基于大语言模型的区域海表温度预测的文本环境上下文与空间图方法

    Textual Environmental Context and Spatial Graphs for LLM-Based Regional SST Forecasting

    [https://arxiv.org/abs/2610.07895](https://arxiv.org/abs/2610.07895)

    该论文提出将文本化的环境上下文与静态、动态空间图相结合，通过图神经网络生成空间前缀注入大语言模型，从而在不序列化完整海温网格的情况下实现区域多步海表温度预测。

    

    海表温度（SST）预测依赖于局地时间持续性、区域空间依赖性以及随预测日期演变的环境条件。我们研究如何在无需将完整的海温网格序列化为文本的情况下，将这些异构条件呈现给大语言模型（LLM），以实现区域多步预测。我们将预测任务形式化为条件数值生成：历史海温与异常序列、日期对齐的环境记录以及静态海洋知识共同构成文本上下文，而区域空间状态则通过连续的图衍生前缀提供。静态图编码持久的地理—气候关系，动态图编码近期的海温相关性以及局地热带气旋的影响。两个图神经网络生成目标节点表示，经由空间前缀融合进行映射并注入大语言模型的输入中。在（区域）海温预测任务中（摘要原文在此处截断）。

    arXiv:2610.07895v1 Announce Type: new  Abstract: Sea surface temperature (SST) forecasting depends on local temporal persistence, regional spatial dependence, and environmental conditions that evolve with the forecast date. We study how these heterogeneous conditions can be presented to a large language model (LLM) for regional multi-step forecasting without serializing the full SST grid as text. We formulate forecasting as conditional numerical generation: historical SST and anomaly sequences, date-aligned environmental records, and static ocean knowledge form a textual context, while regional spatial state is supplied through continuous graph-derived prefixes. A static graph encodes persistent geographic--climatological relations, and a dynamic graph encodes recent SST correlations and localized tropical-cyclone influence. Two graph neural networks produce a target-node representation that is mapped by a spatial-prefix fusion and injected into the LLM input. On SST forecasting in the
    
[^134]: 统一多模态模型中的视觉拒答

    Visual Abstention in Unified Multimodal Models

    [https://arxiv.org/abs/2610.07887](https://arxiv.org/abs/2610.07887)

    该论文形式化了“视觉拒答”概念并构建了Draw-or-Decline基准，揭示出统一多模态模型的编辑能力与拒答能力相互独立——即便最强的编辑模型也几乎不会拒绝不可行的编辑请求。

    

    统一多模态模型（UMMs）将理解与生成能力整合于一体，但其生成行为很少受到其对任务理解的约束。我们对“视觉拒答”进行了形式化定义：当所请求的视觉变换在任务规则下不可能实现时，模型应当认识到不存在有效解，明确说明这一点，并拒绝生成。我们提出了Draw-or-Decline（DoD）基准，包含跨越7个任务类别的1,050对可行-不可行请求对，用以联合评估编辑成功与否以及对不可行请求的拒绝能力。通过对8个统一多模态模型的评估，我们发现编辑能力与拒答能力是两种截然不同的能力：即使在普通指令下编辑准确率高达68.4%的最强编辑模型，也仅拒答了0.4%的不可行请求。模型的推理过程揭示了原因：这些模型很少察觉请求中的冲突，反而将编辑当作可行的请求来规划，常常描述图像中并不存在的物体。

    arXiv:2610.07887v1 Announce Type: cross  Abstract: Unified multimodal models (UMMs) integrate understanding and generation, yet their generative behavior is rarely governed by what they understand about the task. We formalize visual abstention: when a requested visual transformation is impossible under the task's rules, the model should recognize that no valid solution exists, state this, and decline to generate. We introduce Draw-or-Decline (DoD), a benchmark of 1,050 feasible-infeasible request pairs across 7 task categories that jointly measures editing success and the refusal of infeasible requests. Evaluating 8 UMMs, we find that editing ability and abstention are distinct capabilities: even the strongest editor, at 68.4% editing accuracy, refuses only 0.4% of infeasible requests under ordinary instructions. Their reasoning shows why: the models rarely notice the conflict, and instead plan the edit as if the request were possible, often describing objects that are not in the image
    
[^135]: ShanLiangRen：一个用于个性化每日膳食规划的营养智能体

    ShanLiangRen: A Nutrition Agent for Personalized Daily Meal Planning

    [https://arxiv.org/abs/2610.07886](https://arxiv.org/abs/2610.07886)

    该论文提出了个性化全量化多目标膳食规划问题（MDP），并开发了营养智能体ShanLiangRen，通过将用户需求转化为约束规划实例、以精确检索增强生成缩小候选空间，并结合帕累托原则引导的精细化方法，生成兼顾个性化约束与多维营养目标的每日膳食方案。

    

    膳食营养规划在慢性病管理和保持身体健康方面发挥着重要作用。在实际应用中，它必须同时满足个性化约束和合理的多维营养目标。这两个方面常常相互冲突，且用户约束会随着反馈不断演变，导致通用指南与可执行方案之间存在巨大差距。为了弥合这一差距，我们首先提出了个性化全量化多目标膳食规划问题（MDP）。为了解决MDP，我们开发了一个营养智能体ShanLiangRen。该系统首先将膳食规范、营养数据、用户属性和自然语言需求转化为个性化的约束规划实例；然后采用精确的检索增强生成方法，从大规模的食材与食谱空间中缩小可行候选集；最后，采用受帕累托原则指导的精细化方法……

    arXiv:2610.07886v1 Announce Type: new  Abstract: Dietary nutrition planning plays an important role in chronic disease management and maintaining a healthy body. In applications, it must simultaneously satisfy personalized constraints and reasonable multidimensional nutritional goals. These two aspects often conflict, and user constraints evolve with feedback, resulting in a substantial gap between generic guidelines and executable plans. To bridge this gap, we first propose the personalized fully quantified multiobjective dietary planning problem (MDP). To tackle MDP, we develop a nutrition agent, ShanLiangRen. The system first transforms dietary specifications, nutrient data, user attributes and natural language requirements into an individualized constrained planning instance. It then employs an exact retrieval-augmented generation method to shrink the feasible candidate set from a large scale ingredient and recipe space. Finally, it adopts a refinement guided by Pareto principles, 
    
[^136]: 标签高效的心电波形分界深度学习方法：与广泛使用分界工具的多数据集基准对比

    Label-Efficient Deep Learning for ECG Delineation: A Multi-Dataset Benchmark against Widely Used Delineation Tools

    [https://arxiv.org/abs/2610.07885](https://arxiv.org/abs/2610.07885)

    该研究通过多数据集基准测试证明，自监督预训练配合恰当的微调策略可显著降低心电波形分界对专家标注的依赖，且所得深度模型的分界性能优于广泛使用的开源分界工具。

    

    心电图（ECG）波形分界，即识别波形边界，是将原始心电信号转化为临床可解释测量的基础步骤。深度学习已推动这一任务的发展，但仍然依赖于昂贵的专家标注。自监督预训练和半监督学习等标签高效策略有望减轻这一负担，然而它们能否产生可靠的分界结果，以及由此训练出的深度模型是否优于实践中使用的分界工具，目前仍不清楚。我们分两个阶段回答这一问题。首先，在一个内部数据集和四个外部数据集上比较自监督目标与有监督或半监督微调，我们发现预训练有帮助，但目标函数的选择至关重要，且半监督微调的价值取决于预训练目标。其次，我们将选定的深度学习模型与广泛使用的开源分界工具进行基准对比。

    arXiv:2610.07885v1 Announce Type: cross  Abstract: Electrocardiogram (ECG) delineation, the identification of waveform boundaries, is a foundational step that translates raw ECG signals into clinically interpretable measurements. Deep learning has advanced this task but remains dependent on costly expert annotations. Label-efficient strategies such as self-supervised pretraining and semi-supervised learning are expected to ease this burden, yet it remains unclear whether they yield reliable delineation and whether the deep models they produce outperform the delineation tools used in practice. We address this in two stages. First, comparing self-supervised objectives with supervised or semi-supervised fine-tuning across one internal and four external datasets, we find that pretraining helps but the objective matters, and that the value of semi-supervised fine-tuning depends on the pretraining objective. Second, we benchmark the selected deep learning model against widely used open-sourc
    
[^137]: 自参照社会偏好：无需观察他人奖励的合作

    Self-Referenced Social Preferences: Cooperation without Observing Others Rewards

    [https://arxiv.org/abs/2610.07881](https://arxiv.org/abs/2610.07881)

    该论文提出“自参照社会偏好”方法，让智能体利用自身奖励模型从自身视角评估他人行为的结果，从而无需观察他人私有奖励即可在多智能体强化学习中有效促进合作。

    

    社会偏好可以在多智能体强化学习中促进合作，但现有方法通常需要智能体观察其同伴的奖励。然而，在许多现实世界的交互中，智能体可以像人类一样观察他人的行为和结果，却无法获取他人私有的奖励信号。我们提出了自参照社会偏好，其中每个智能体学习自身奖励的模型，将该模型应用于其他智能体观察到的状态转移，以从自身视角评估其结果，并将这些自参照评估输入到标准的社会偏好机制中。我们研究了两种整合这些评估的方式：修改学习奖励，或用它们对策略更新进行加权。我们在三个序贯社会困境上评估了该方法——Escape Room（密室逃脱）、Clean Up（清洁任务）和 Commons Harvest（公共资源收获），它们分别需要自愿参与、公共品贡献和资源克制。在所有三个环境中……

    arXiv:2610.07881v1 Announce Type: new  Abstract: Social preferences can promote cooperation in multi-agent reinforcement learning, but existing approaches often require agents to observe the rewards of their peers. In many real-world interactions, however, an agent can, as humans do, observe others' behavior and outcomes without access to their private reward signals. We introduce self-referenced social preferences, in which each agent learns a model of its own reward, applies it to other agents' observed transitions to assess their outcomes from its own perspective, and feeds these self-referenced assessments into standard social preferences. We study two ways to incorporate these assessments: modifying the learning reward, or using them to weight policy updates. We evaluate the approach on three sequential social dilemmas, Escape Room, Clean Up, and Commons Harvest, which require volunteering, public-good contribution, and resource restraint, respectively. Across all three environmen
    
[^138]: ReFold：面向长程智能体的免训练可逆轮间上下文折叠方法

    ReFold: Training-Free Reversible Inter-Turn Context Folding for Long-Horizon Agents

    [https://arxiv.org/abs/2610.07863](https://arxiv.org/abs/2610.07863)

    ReFold提出了一种免训练的可逆上下文折叠渲染层，通过将已展示内容替换为占位符、将智能体报告完成的轮次折叠为一行注释来消除轮间冗余，从而在保留底层完整交互历史的同时压缩长程智能体的渲染上下文，避免了现有预测性方法带来的运行时开销、前缀缓存失效和不可逆信息丢失。

    

    长程LLM智能体基于只追加（append-only）的交互历史进行行动，该历史在每一步都会被重新发送给模型，因此上下文及其成本随步骤不断增长，直到会话超出上下文窗口。现有方法通过上下文需求预测来管理上下文，依赖于额外的模型调用、启发式规则或训练得到的策略。然而，这些预测性方法会引入运行时开销、使前缀缓存失效，并永久丢弃内容且无法保证恢复。为克服这些限制，我们提出了ReFold：一个免训练的渲染层，它在保留底层交互历史的同时仅压缩模型的渲染上下文。它无需辅助预测器即可消除两类轮间冗余：较早轮次已经展示过的内容（用占位符替换），以及智能体自身报告已完成的轮次（折叠为一行注释）。两种操作均使用分块渲染与重写（摘要在此处截断）……

    arXiv:2610.07863v1 Announce Type: cross  Abstract: Long-horizon LLM agents act on an append-only interaction history that is re-sent to the model at every step, so the context and its cost grow with steps until the sessions exceed the context window. Existing methods manage the context through context requirement prediction, relying on additional model calls, heuristic rules, or trained policies. However, these predictive approaches introduce runtime overhead, invalidate prefix caches, and permanently discard content with no guarantee of recovery. To overcome these limitations, we introduce ReFold: a training-free rendering layer that preserves the underlying interaction history while compressing only the model's rendered context. It removes two kinds of inter-turn redundancy without an auxiliary predictor: content an earlier turn already displayed, replaced by a stub, and turns the agent itself reports finished, folded into a one-line note. Both operators use chunked rendering, rewrit
    
[^139]: 一种面向X射线衍射的自学习科学智能体

    A self-learning scientific agent for X-ray diffraction

    [https://arxiv.org/abs/2610.07862](https://arxiv.org/abs/2610.07862)

    本文提出“干将”——一个面向粉末X射线衍射的自学习科学智能体，它通过诊断失败并自主修订和验证技能指令与代码，将分析经验转化为可复用的可执行技能，且无需重新训练语言模型，在多个精修平台上超越了专家设计的技能。

    

    科学智能体面临的一个核心挑战，是将分析经验转化为以物理证据为基础的可复用专业知识。本文介绍了“干将”，这是一个面向粉末X射线衍射的自学习智能体，它构建于我们开发的衍射分析生态系统之上：XMatcher、XQueryer、XDecomposer和WPEM。这些引擎共同覆盖了物相鉴定、多相分解以及物理约束的全谱建模。“干将”通过诊断失败、修订技能指令和代码，并在复用前对修订内容进行验证，将分析经验转化为可执行的技能，而无需重新训练语言模型或更改底层物理模型。基于开发数据筛选并在留出集评估前冻结的技能，在FullProf、GSAS-II和PyWPEM上均取得了比原始专家设计技能更高的精修分数。该智能体能够解析强烈重叠的衍射峰，并对五相混合物进行定量分析（摘要在此处截断）。

    arXiv:2610.07862v1 Announce Type: cross  Abstract: A central challenge for scientific agents is to turn analytical experience into reusable expertise grounded in physical evidence. Here we introduce Gan Jiang, a self-learning agent for powder X-ray diffraction built on a diffraction-analysis ecosystem we developed: XMatcher, XQueryer, XDecomposer and WPEM. Together, these engines span phase identification, multiphase decomposition and physics-constrained whole-pattern modelling. Gan Jiang converts analytical experience into executable skills by diagnosing failures, revising skill instructions and code, and validating revisions before reuse, without retraining the language model or changing the underlying physical models. Skills selected using development data and frozen before held-out evaluation achieve higher refinement scores than the original expert-designed skills across FullProf, GSAS-II and PyWPEM. The agent resolves strongly overlapping reflections, quantifies a five-phase anci
    
[^140]: WorkflowOps：面向多智能体工作流编排的智能体协作先验学习

    WorkflowOps: Learning Agent Collaboration Priors for Multi-Agent Workflow Orchestration

    [https://arxiv.org/abs/2610.07860](https://arxiv.org/abs/2610.07860)

    WorkflowOps 提出了一种多智能体工作流编排框架，通过从历史工作流中学习智能体协作先验（转移概率矩阵）来引导 DAG 工作流构建，并按需创建专门的智能体以填补能力空缺，从而避免编排层“无记忆、从零开始”的问题。

    

    多智能体系统正越来越多地被部署用于复杂的知识工作，然而其编排层在很大程度上仍然是无记忆的：每个新任务都是从零开始进行分解、分配和执行，无法从以往的成功执行中获益。我们提出了 WorkflowOps，一个多智能体工作流编排框架，它从历史工作流中学习智能体协作先验，并按需扩展其智能体池以覆盖新的能力需求。我们的方法引入了三种相互耦合的机制。首先，一个转移概率矩阵从历史工作流中捕获智能体两两之间的协作频率，并在构建 DAG 工作流时通过层内排序优化、基于概率阈值的边建议以及用于最大化并行度的传递约简，将其作为软性引导加以应用。其次，一个由充分性驱动的智能体创建循环通过语义匹配分数检测能力差距，生成专门的智能体

    arXiv:2610.07860v1 Announce Type: new  Abstract: Multi-agent systems are increasingly deployed for complex knowledge work, yet their orchestration layers remain largely memoryless: each new task is decomposed, assigned, and executed from scratch with no benefit from prior successful executions. We present WorkflowOps, a multi-agent workflow orchestration framework that learns agent collaboration priors from historical workflows and expands its agent pool on demand to cover new capability requirements. Our approach introduces three coupled mechanisms. First, a transition probability matrix captures pairwise agent collaboration frequencies from past workflows and applies them as soft guidance during DAG workflow construction through intra-layer ordering optimization, probability-thresholded edge suggestion, and transitive reduction for parallelism maximization. Second, a sufficiency-driven agent creation loop detects capability gaps via semantic matching scores, generates specialized age
    
[^141]: RA-MoWE：用于查询聚类与智能体工作流生成的工作流亲和度嵌入

    RA-MoWE: Workflow-Affinity Embeddings for Query Clustering and Agentic Workflow Generation

    [https://arxiv.org/abs/2610.07851](https://arxiv.org/abs/2610.07851)

    RA-MoWE 通过记录参考工作流求解效果的工作流亲和度嵌入对查询进行聚类，并借助执行反馈为每个聚类生成可复用的专家工作流，使新查询无需先执行即可直接匹配到合适的专门化工作流。

    

    智能体工作流使大型语言模型（LLM）能够通过协调推理、工具使用和验证来解决复杂任务。然而，为整个任务集合优化得到的工作流可能会忽略各个查询所需的推理策略之间的差异，而为每个查询搜索新工作流又会重复代价高昂的优化过程。为了解决这种权衡问题，我们提出了 RA-MoWE，这是一个利用工作流亲和度嵌入对查询进行聚类，并指导生成可复用专家工作流的框架。每个嵌入记录了一组固定的参考工作流对某个查询的求解效果，从而揭示哪些推理策略有效方面的相似性。RA-MoWE 利用每个聚类的查询和平均嵌入，通过执行反馈来初始化和改进专门化的工作流。一个嵌入编码器可以从查询文本预测这些嵌入，使新查询无需先执行即可选择一个已生成的专家工作流。

    arXiv:2610.07851v1 Announce Type: new  Abstract: Agentic workflows enable large language models (LLMs) to solve complex tasks by coordinating reasoning, tool use, and verification. However, a workflow optimized for an entire task collection can overlook differences in the reasoning strategies that individual queries need, while searching for a new workflow for every query repeats costly optimization. To address this tradeoff, we introduce RA-MoWE, a framework that uses workflow-affinity embeddings to cluster queries and guide the generation of reusable expert workflows. Each embedding records how well a fixed set of reference workflows solves a query, revealing similarities in which reasoning strategies are effective. RA-MoWE uses each cluster's queries and average embedding to initialize and refine a specialized workflow through execution feedback. An embedding encoder predicts these embeddings from query text, allowing new queries to select a generated expert without first executing 
    
[^142]: 面向大语言模型参数高效微调的动态位置注意力调制

    Dynamic Positional Attention Modulation for Parameter-Efficient Fine-Tuning of Large Language Models

    [https://arxiv.org/abs/2610.07848](https://arxiv.org/abs/2610.07848)

    提出DyPAM方法，通过在查询和键表征上结合输入条件化的逐维度调制与逐头逐层的结构化调制，动态调整位置信息对注意力的贡献，实现更精细的参数高效微调。

    

    参数高效微调（PEFT）已成为将大语言模型适配到下游任务的标准方法。然而，大多数现有的PEFT方法依赖于统一且静态的适配方式，没有考虑注意力在维度、注意力头、层以及输入标记之间的结构化异质性。在实践中，注意力表征表现出非均匀的行为，并且旋转位置编码等位置编码机制会引入依赖于维度的位置结构，使得统一的适配方式并非最优。在本工作中，我们提出了DyPAM（动态位置注意力调制），这是一种参数高效微调方法，通过直接作用于查询和键表征，来调整位置信息对注意力的贡献方式。DyPAM将基于输入条件的逐维度调制与逐注意力头、逐层的结构化调制相结合，对位置注意力进行细粒度的适配。

    arXiv:2610.07848v1 Announce Type: cross  Abstract: Parameter-efficient fine-tuning (PEFT) has become a standard approach for adapting large language models to downstream tasks. However, most existing PEFT methods rely on uniform and static adaptations, without accounting for the structured heterogeneity of attention across dimensions, heads, layers, and input tokens. In practice, attention representations exhibit non-uniform behavior, and positional encoding mechanisms such as rotary positional embeddings (RoPE) induce dimension-dependent positional structure, making uniform adaptation suboptimal. In this work, we propose DyPAM (Dynamic Positional Attention Modulation), a PEFT method that adapts how positional information contributes to attention by operating directly on the query and key representations. DyPAM combines input-conditioned, dimension-wise modulation with head-wise and layer-wise structural modulation, performing fine-grained adaptation of positional attention aligned wit
    
[^143]: DHCG：面向基于大语言模型的多智能体推理的动态分层协作图构建

    DHCG: Dynamic Construction of Hierarchical Collaboration Graphs for LLM-Based Multi-Agent Reasoning

    [https://arxiv.org/abs/2610.07835](https://arxiv.org/abs/2610.07835)

    提出DHCG框架，将多智能体系统设计建模为部分可观测马尔可夫决策过程，通过规划器、工作者和生成器三个模块，基于查询与执行反馈动态构建分层协作图，实现智能体组合与规模的灵活自适应。

    

    基于大语言模型的多智能体系统（MAS）在解决跨多个领域的复杂问题方面已展现出强大的能力。近年来，智能体系统的动态编排已成为一个重要的研究方向。然而，现有方法存在组合受限、依赖关系错位和规模不灵活等问题，限制了其在执行过程中适应推理需求的能力。为了解决这些局限性，我们将MAS设计重新构建为一个部分可观测马尔可夫决策过程，其中MAS的组成和规模均被动态确定。我们提出了DHCG，这是一个新颖的框架，它协调三个模块（规划器、工作者和生成器），基于查询和不断演化的执行反馈，从零开始逐步构建动态分层协作图。在每一步中，在反馈的引导下，规划器生成一组独特且互补的角色，以适应当前的推理……

    arXiv:2610.07835v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) have demonstrated strong capabilities in solving complex problems across diverse domains. Recently, the dynamic orchestration of agent systems has become an important research direction. However, existing methods suffer from limited composition, misaligned dependencies, and inflexible scale, restricting their ability to adapt to reasoning requirements during execution. To address these limitations, we reframe MAS design as a partially observable Markov decision process, in which both the composition and scale of the MAS are dynamically determined. We propose DHCG, a novel framework that coordinates three modules (Planner, Worker, and Generator) to progressively construct a dynamic hierarchical collaboration graph from scratch based on the query and evolving execution feedback. At each step, guided by feedback, the Planner generates a set of distinct and complementary roles tailored to the current reaso
    
[^144]: 通过模块化可执行的开发原语为软件工程构建工程化框架

    Harness Engineering for Software Engineering via Modular Executable Dev-Primitives

    [https://arxiv.org/abs/2610.07832](https://arxiv.org/abs/2610.07832)

    该论文提出Dev-Primitives，一种将代码库工件与常驻LLM配对的模块化可执行抽象，使软件组件从被动工件转变为具备智能体原生接口的主动参与者，从而解决LLM智能体在长程软件工程工作流中反复重建程序状态、上下文爆炸和语义漂移的问题。

    

    配备终端访问能力的大型语言模型（LLMs）在自动化软件工程任务方面已展现出强大的能力。然而，现有智能体在长程工作流中依然十分脆弱：它们必须反复重建分散在源代码文件、配置、测试、依赖项和运行时行为中的程序状态，导致交互历史不断膨胀、上下文爆炸以及语义漂移。大型代码库则进一步增加了识别与任务相关组件的难度。为了应对这些挑战，我们提出了Dev-Primitives（开发原语），这是一种模块化且可执行的抽象，它将代码库组件从被动的软件工件转变为软件工程中的主动参与者。每个Dev-Primitive将一个代码库工件与一个常驻LLM配对，从而赋予该工件一个基于其自身实现和依赖关系的智能体原生接口。

    arXiv:2610.07832v1 Announce Type: cross  Abstract: Large language models (LLMs) equipped with terminal access have demonstrated strong capabilities in automating software engineering tasks. However, existing agents remain brittle on long-horizon workflows, where they must repeatedly reconstruct program state scattered across source files, configurations, tests, dependencies, and runtime behavior, leading to increasingly long interaction histories, context explosion, and semantic drift. Large repositories further complicate the identification of task-relevant components. To address these challenges, we introduce \textbf{Dev-Primitives} (\emph{Development Primitives}), a modular and executable abstraction that transforms repository components from passive software artifacts into active participants in software engineering. Each Dev-Primitive pairs a repository artifact with a resident LLM, which gives the artifact an agent-native interface grounded in its own implementation and dependenc
    
[^145]: 面向资源自适应AI无线接入网络的智能体化语义感知

    Agentic Semantic Sensing for Resource-Adaptive AI-RAN

    [https://arxiv.org/abs/2610.07829](https://arxiv.org/abs/2610.07829)

    该论文提出了一种闭环的智能体化语义感知框架（Agentic SemS），通过配置条件化因果Transformer与语义效用网络，在通信可行的资源配置集合内动态控制感知过程，从而实现资源自适应的AI无线接入网络。

    

    语义感知（SemS）获取的是与任务相关的信息，而非重构完整的物理信息。现有的语义感知方案通常以开环方式运行：感知配置与观测调度在推理之前就已固定，无法响应不断演化的任务级证据。我们提出了智能体化语义感知（Agentic SemS），这是一种面向AI赋能无线接入网络（AI-RAN）的闭环框架，可在通信可行的配置集合内对感知过程进行控制。一个以配置为条件的因果Transformer从流式观测中更新语义信念，同时键值缓存机制使得在配置切换时能够高效地进行状态更新，无需重复处理完整的历史信息。语义效用网络在计入感知成本后，估计在每个可行配置下获取下一个观测块所能带来的任务级收益。由此得到的继续效用共同支持下一配置选择与语义提前（决策）。

    arXiv:2610.07829v1 Announce Type: new  Abstract: Semantic sensing (SemS) acquires task-relevant information rather than reconstructing complete physical information. Existing SemS formulations typically operate open loop: sensing configurations and observation schedules are fixed before inference and cannot respond to evolving task-level evidence. We propose Agentic SemS, a closed-loop framework for AI-enabled radio access networks (AI-RANs) that controls sensing within a communication-feasible profile set. A profile-conditioned causal Transformer updates the semantic belief from streaming observations, while key-value caching enables efficient state updates across profile changes without repeatedly processing the complete history. A semantic utility network estimates the task-level benefit of acquiring the next observation block under each feasible profile after accounting for sensing cost. The resulting continuation utilities jointly support next-profile selection and semantic early 
    
[^146]: 一步一个脚印：以大语言模型自主性换取流程可预测性

    One Step at a Time: Trading LLM Autonomy for Process Predictability

    [https://arxiv.org/abs/2610.07817](https://arxiv.org/abs/2610.07817)

    该论文提出通过MCP协议逐步向智能体交付流程步骤，以牺牲LLM自主性为代价，从架构上保证流程的事前可预测性，并生成可供下游工具逐步审计和优化的机器可读执行日志。

    

    对运营流程进行自动化的组织所需要的不仅仅是正确的结果：他们还需要预测流程将如何运行、了解实际运行的是哪一个流程，并逐步对其进行检查。当智能体作为执行者时，这种可预测性通常会丢失：规定的流程被写入系统提示词中，而系统最终只返回一个最终答案。我们改为通过模型上下文协议逐步交付流程：服务器每次只释放一个步骤，智能体执行该步骤，并且每个步骤都会返回一个结构化的step_output。这以自主性换取可预测性，由此两个性质通过构造自然成立，且不依赖于执行者：其一，执行路径在运行前就被规定好，因此流程是事先可预测的，而非事后重建的；其二，已完成的步骤记录构成了机器可读的执行日志，下游工具可以逐步对其进行审计和优化。在13个SO上对15,475次试验进行评估……

    arXiv:2610.07817v1 Announce Type: cross  Abstract: Organizations automating operational processes need more than a correct outcome: they need to predict how a process will run, know which one actually ran, and inspect it step by step. When an agent is the executor that predictability is normally lost: the prescribed procedure goes into the system prompt, and only a final answer comes back. We deliver the procedure step by step over the Model Context Protocol (MCP) instead: a server releases one step at a time, the agent executes it, and each step returns a structured step_output. This trades autonomy for predictability, and two properties then follow by construction, independent of the executor. The execution path is prescribed before the run, so the process is predictable in advance rather than reconstructed afterwards; and the completed step records form a machine-readable execution log that downstream tooling can audit and optimize step by step. Evaluating 15,475 trials across 13 SO
    
[^147]: 我需要云端吗？面向小型语言模型智能体的不确定性感知步骤级交接

    Do I Need the Cloud? Uncertainty-Aware Step-Level Handoff for Small Language Model Agents

    [https://arxiv.org/abs/2610.07816](https://arxiv.org/abs/2610.07816)

    提出STEPGATE框架，通过不确定性感知地对本地小模型的每个动作评分，并按需将困难步骤升级到更强的云端模型，从而在大幅降低云端调用比例的同时显著提升智能体的任务成功率。

    

    小型语言模型（SLM）作为本地智能体控制器极具吸引力，因为它们能减少远程推理、降低延迟并缩小部署占用，但结构化的工具错误可能导致智能体步骤执行失败。现有的路由器通常在每次查询时只做一次模型选择。然而，智能体包含一系列顺序决策点，其难度会随中间观测结果动态变化。我们提出了STEPGATE，一个不确定性感知的交接框架，它对每个本地SLM动作进行评分，并有选择地将具有挑战性的步骤升级到更强的模型。在一个基于BFCL、包含52个任务的留出单步测试集上，Qwen2.5-1.5B/7B模型对以30.8%的升级率取得了82.7%的任务成功率，而仅本地方案为67.3%，随机升级方案（升级率为33.8%）为75.4%。在一项独立的多轮评估中，STEPGATE仅使用30.0%的云端动作，就实现了69.0%的轨迹成功率和84.0%的动作成功率，相比之下，仅本地方案为48.0%/70.5%，随机升级方案为60.0%/78.2%……

    arXiv:2610.07816v1 Announce Type: new  Abstract: Small language models (SLMs) are attractive as local agent controllers because they reduce remote inference, latency, and deployment footprint, yet structured tool errors can cause an agent step to fail. Existing routers typically select a model once per query. However, agents expose sequential decision points whose difficulty dynamically changes based on intermediate observations. We propose STEPGATE, an uncertainty-aware handoff framework that scores each local SLM action and selectively escalates challenging steps to a stronger model. On a 52-task held-out single-step BFCL-derived test split, the Qwen2.5-1.5B/7B pair attains 82.7% task success with 30.8% escalation, versus 67.3% local-only and 75.4% random escalation (which uses 33.8% escalation). In a separate multi-turn evaluation, STEPGATE achieves 69.0% trajectory success and 84.0% action success using only 30.0% cloud actions, compared with 48.0%/70.5% local-only, 60.0%/78.2% ran
    
[^148]: ThinkFuse：面向小型推理模型的轨迹感知测试时融合

    ThinkFuse: Trajectory-Aware Test-Time Fusion for Small Reasoning Models

    [https://arxiv.org/abs/2610.07803](https://arxiv.org/abs/2610.07803)

    ThinkFuse提出了一种无需训练的轨迹感知测试时融合框架，通过对比片段级不确定性变化与轨迹级整体趋势来识别不稳定推理点，并将辅助推理路径融合进主轨迹，从而显著提升小型推理模型在数学和知识密集型推理任务上的可靠性与性能。

    

    小型推理模型通过生成扩展的思维链轨迹，在复杂推理任务上展现出强大的性能，但一旦推理进入错误路径，往往难以恢复。现有的测试时融合方法依赖局部融合信号来决定何时触发融合，这可能会被瞬时的不确定性波动所误导，并可能强化不稳定的推理轨迹。我们提出了ThinkFuse，这是一个无需训练的测试时融合框架，能够选择性地干预不可靠的推理片段。ThinkFuse通过比较片段级别的不确定性变化与轨迹级别的整体不确定性趋势，识别不稳定的推理点，并将辅助推理路径融合到主模型的推理轨迹中。大量实验表明，ThinkFuse在数学推理和知识密集型推理基准上优于基线方法，在不同模型家族组合中均取得一致的提升。

    arXiv:2610.07803v1 Announce Type: new  Abstract: Small reasoning models (SRMs) have shown strong performance on complex reasoning tasks by generating extended chain-of-thought trajectories, but they often fail to recover once their reasoning enters an erroneous path. Existing test-time fusion methods rely on local fusion signals to determine when to trigger fusion, which can be misled by transient uncertainty fluctuations and may reinforce unstable reasoning trajectories. We propose ThinkFuse, a training-free test-time fusion framework that selectively intervenes in unreliable reasoning segments. ThinkFuse compares segment-level uncertainty shifts with trajectory-level uncertainty trends to identify unstable reasoning points and fuse auxiliary reasoning paths into the primary model's trajectory. Extensive experiments demonstrate that ThinkFuse outperforms baselines on mathematical and knowledge-intensive reasoning benchmarks, with consistent gains across model-family combinations, and 
    
[^149]: AI辅助决策中新手的依赖校准：解释与自我评估的作用

    Novice Reliance Calibration in AI-Assisted Decision Making: The Role of Explanations and Self-Assessment

    [https://arxiv.org/abs/2610.07800](https://arxiv.org/abs/2610.07800)

    论文提出“依赖校准”这一新构念，通过对110名参与者的实验发现，在缺乏外部反馈时，AI解释会使新手用户系统性地滑向过度依赖，而任务自我理解等元认知自我评估有助于校准对AI的依赖。

    

    人工智能（AI）工具被广泛用于在缺乏即时性能反馈的任务和领域中辅助决策。在这些情境下，用户无法通过反复试错来学习调整自己对AI的依赖行为。然而，关于新手用户在外部反馈不可用时如何校准对AI的依赖，以及AI解释能否在其缺失时支持这种校准，目前知之甚少。我们将“依赖校准”作为一个组织性构念引入，用以研究新手用户如何动态调整依赖行为，并考察AI解释与元认知自我评估如何对其产生影响。通过一项包含110名参与者的被试间实验，让参与者在AI辅助下完成临床实体抽取任务且仅有有限的性能反馈，我们观察到：在存在解释的情况下，新手用户表现出向过度依赖的系统性漂移，而较高的自我报告任务理解程度……（摘要原文在此处截断）

    arXiv:2610.07800v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) tools are widely used to support decision making in tasks and domains where no immediate performance feedback is available. In these settings, users cannot learn to adjust their reliance behavior over time through trial and error. However, little is known about how novice users calibrate reliance on AI when external feedback is unavailable, or whether AI explanations can support calibration in its absence. We introduce reliance calibration as an organizing construct for studying how novice users dynamically adjust reliance behavior, and examine how AI explanations and meta-cognitive self-assessment shape it. Through a between-subjects study with 110 participants completing a clinical entity extraction task with AI assistance and limited performance feedback, we observe that novice users exhibit systematic drift toward over-reliance in the presence of explanations, while higher self-reported task understandi
    
[^150]: 证据稀薄，先验厚重：语言模型如何以身份替代缺失的财务事实

    Thin Evidence, Thick Priors: How Language Models Substitute Identity for Missing Financial Facts

    [https://arxiv.org/abs/2610.07798](https://arxiv.org/abs/2610.07798)

    论文发现，用户披露的财务信息越少，大语言模型给出的投资建议就越依赖投资者身份而非其实际财务状况——在零披露条件下，身份对建议差异的解释力从5%飙升至96%，两人间建议差距也从4.78个百分点扩大至10.34个百分点。

    

    人们越来越多地向大型语言模型请教如何打理自己的钱财，却很少完整描述自身的财务状况。本文研究模型如何应对这一信息缺口。在保持财务状况不变、仅改变投资者所称身份的条件下，我们将提示中的财务证据从八项事实逐步削减至零，并测量推荐股票配置比例随之变动的幅度。在向Llama-3.1-8B-Instruct发出的96,600个提示中（由100个财务画像、138个人物设定和七种披露条件构建），两个财务状况完全相同的人物设定之间的平均建议差距，从完全披露时的4.78个百分点上升到完全没有财务事实时的10.34个百分点。采用对重复提示只计一次的双向聚类自助法，该比值估计为2.16（95%区间为1.69至2.79），而且仅剩一条事实时差距已上升1.69倍。在完全披露时，身份解释了同一财务画像内建议变异的5%，而在零披露时这一比例高达96%。家庭规模……

    arXiv:2610.07798v1 Announce Type: new  Abstract: People increasingly ask large language models what to do with their money, yet seldom describe their finances in full. This paper asks what a model does with the gap. Holding finances fixed and changing only who the investor is said to be, we grade the financial evidence in the prompt from eight facts to none and measure how far the recommended equity allocation moves. Across 96,600 prompts to Llama-3.1-8B-Instruct, built from 100 financial profiles, 138 personas and seven disclosure conditions, the average gap between two personas with identical finances rises from 4.78 percentage points at full disclosure to 10.34 points with no financial facts. A two-way cluster bootstrap counting duplicated prompts once places the ratio at 2.16 (95% interval 1.69 to 2.79), and the rise is already 1.69-fold with a single fact left. Identity explains 5% of within-profile variation in advice at full disclosure and 96% with no disclosure. Household size 
    
[^151]: 赋权的几何学

    The Geometry of Empowerment

    [https://arxiv.org/abs/2610.07796](https://arxiv.org/abs/2610.07796)

    本文将赋权最大化与技能学习方法相联系，提出了解释赋权的新几何框架，解答了赋权与结构中心性之间联系的长期开放问题，并揭示了信息几何与奖励几何的区别，为构建可扩展的赋权最大化方法奠定理论基础。

    

    赋权刻画了智能体主动控制其环境的能力。尽管作为一种信息论量在概念上颇具吸引力，但赋权与那些能广泛通往未来结果的结构性中心状态之间的联系，一直是一个悬而未决的问题。在这项工作中，我们将赋权最大化与技能学习方法联系起来，为解释和分析赋权提供了新的几何视角。我们的分析回答了关于赋权与结构中心性之间联系的长期开放问题，还揭示了信息几何与奖励几何之间的区别，为构建可扩展的赋权最大化方法提供了重要的理论启示。网站与代码可在 https://empowerment-geometry.github.io/ 获取。

    arXiv:2610.07796v1 Announce Type: cross  Abstract: Empowerment captures the capacity for an agent to actively control its environment. While conceptually appealing as an information-theoretic quantity, the connection between empowerment and structurally central states that provide broad access to future outcomes has remained an open question. In this work, we link empowerment maximization and skill-learning methods to provide new geometries for interpreting and analyzing empowerment. Our analyses answer longstanding open questions on the connections between empowerment and structural centrality. Our analyses also reveal distinctions between information and reward geometries, highlighting important theoretical implications to build scalable empowerment-maximization methods. Website and code can be found at https://empowerment-geometry.github.io/.
    
[^152]: ServeLearnBench：智能体从服务经验中自我改进的能力究竟有多强？

    ServeLearnBench: How Well Can Agents Self-Improve from Serving Experience?

    [https://arxiv.org/abs/2610.07792](https://arxiv.org/abs/2610.07792)

    本文提出 ServeLearnBench 基准及演化环境流式数据集（EESD），用于系统评估大语言模型智能体在隐藏策略持续演变的环境中，能否从交互与反馈中推断、应用并修正潜在环境知识，从而实现自我改进。

    

    大语言模型智能体正越来越多地被部署到真实环境中执行复杂任务。然而，在这些环境中实现正确行为所需的知识往往是隐含的、未公开的，并且会随时间推移而变化。近期的持续学习框架试图通过让智能体从服务经验中不断改进来应对这一挑战，但这些方法的有效性与局限性尚未得到充分的刻画。现有基准仅提供了部分覆盖：有些基准明确给出目标知识，有些则假设环境是静态的，而支持持续适应的基准在规模和知识多样性方面仍然有限。为了实现系统化评估，我们形式化定义了演化环境流式数据集（EESD），在隐藏策略不断演变的过程中，智能体必须从交互与结果反馈中推断、应用并修正潜在的环境知识，并在此基础上提出了 ServeLearnBench 基准。

    arXiv:2610.07792v1 Announce Type: cross  Abstract: Large language model agents are increasingly deployed to perform complex tasks in real-world environments. However, the knowledge required for correct behavior in these environments is often implicit, undisclosed, and subject to change over time. Recent continual-learning harnesses seek to address this challenge by enabling agents to improve from serving experience. Yet the effectiveness and limitations of these methods are not yet well characterized. Existing benchmarks provide only partial coverage: some explicitly provide the target knowledge, others assume a static environment, and those that support continual adaptation remain limited in scale and knowledge diversity. To enable systematic evaluation, we formalize an evolving-environment streaming dataset (EESD), in which agents must infer, apply, and revise latent environment knowledge from interaction and outcome feedback as hidden policies evolve, and introduce ServeLearnBench, 
    
[^153]: 虚幻模式感知驱动大语言模型中的虚假推理

    Illusory Pattern Perception Drives Spurious Inference in Large Language Models

    [https://arxiv.org/abs/2610.07791](https://arxiv.org/abs/2610.07791)

    本研究首次系统性地揭示了大语言模型存在比人类更强的虚幻模式感知倾向——如将积极属性过度关联到多数群体、从模糊事件中强行构建因果叙事——这种认知偏差会导致模型产生系统性推理错误。

    

    虚幻模式感知是一种已被充分记录的人类认知倾向，即在实际上随机的数据中推断出有意义的关系。这种倾向通常被描述为在不存在联系的地方“强行连线”，可能导致系统性的推理错误。本文研究大语言模型（LLMs）是否表现出这种感知倾向，从而导致下游应用中的系统性错误。据我们所知，这项工作首次对大语言模型中的虚幻模式感知进行了系统研究，将经典心理学范式应用于三个任务，并与人类行为进行了直接的实证比较。我们发现，大语言模型经常表现出比人类更强的虚幻模式感知。特别是，模型倾向于将频繁出现的积极属性与多数群体或大型组织过度关联，并表现出更强的从模糊事件中构建因果叙事的倾向。为了揭示其机制……

    arXiv:2610.07791v1 Announce Type: new  Abstract: Illusory pattern perception is a well-documented human cognitive tendency to infer meaningful relationships in data that is actually random. Such a tendency, often described as "connecting the dots" where none exist, can result in systematic reasoning errors. This paper investigates whether Large Language Models (LLMs) exhibit such perceptual tendencies, which can lead to systematic errors in downstream applications. To our knowledge, this work presents the first systematic study of illusory pattern perception in LLMs, adapting classic psychological paradigms to three tasks with direct empirical comparison to human behaviors. We find that LLMs frequently exhibit stronger illusory pattern perception than humans. In particular, models tend to over-associate frequent positive attributes with majority groups or large organizations, and show increased tendencies to construct causal narratives from ambiguous events. To uncover the mechanism be
    
[^154]: OOPMAS：面向查询级工作流生成的面向对象多智能体系统

    OOPMAS: Object-Oriented Multi-Agent Systems for Query-Level Workflow Generation

    [https://arxiv.org/abs/2610.07787](https://arxiv.org/abs/2610.07787)

    OOPMAS提出了一种无需训练的面向对象多智能体框架，能为每个查询动态生成专属的智能体集合与协调工作流，从而适应查询难度差异和现实场景中的异构任务类型。

    

    由大语言模型驱动的多智能体系统（MAS）在代码生成、数学推理和问答等任务中展现出强大性能。然而，现有的MAS自动化设计方法大多在任务层面运作，即为每个基准生成单一固定的工作流，并将其统一应用于所有查询。这一假设在现实条件下难以成立：同一任务内不同查询的难度差异巨大，且真实世界的工作负载混合了异构的任务类型。我们提出了OOPMAS，这是一个无需训练的框架，能够在单个查询的粒度上生成智能体集合与协调工作流。智能体被表示为具有专属角色、工具和持久状态的面向对象类定义，而工作流则被表达为作用于这些智能体对象的可执行主函数。一个动态技能库在多轮优化过程中从执行反馈中积累结构化经验教训，从而……

    arXiv:2610.07787v1 Announce Type: new  Abstract: Multi-agent systems (MAS) powered by large language models have shown strong performance across code generation, mathematical reasoning, and question answering. However, existing methods for automating MAS design mostly operate at the task level, producing a single fixed workflow per benchmark that is applied uniformly to all queries. This assumption fails under realistic conditions. Query difficulty varies widely within a task, and real-world workloads mix heterogeneous task types. We introduce OOPMAS, a training-free framework that generates both the agent set and the coordination workflow at the granularity of individual queries. Agents are represented as object-oriented class definitions with dedicated roles, tools, and persistent state, and workflows are expressed as executable main functions over these agent objects. A dynamic skill library accumulates structured lessons from execution feedback across optimization rounds, enabling 
    
[^155]: Attacca：面向长时程具身智能体的状态连续性下目标导向控制

    Attacca: Goal-Directed Control under State Continuity for Long-Horizon Embodied Agents

    [https://arxiv.org/abs/2610.07785](https://arxiv.org/abs/2610.07785)

    提出 Attacca 方法，通过在完整的“搜索—交互”轨迹上训练视觉目标条件策略，解决了长时程具身任务中因状态连续变化导致目标不可见、难以衔接下一个任务的核心难题。

    

    具身智能体的一项核心能力是通过一系列相互依赖的任务来完成复杂目标。然而，现有作为这类智能体基础的视觉目标条件策略，通常是在目标已经可见的孤立交互中进行评估的，因而无法刻画连续长时程任务执行过程中出现的实际状况。在这种情境下，每个任务都从前一个任务遗留的状态开始：智能体可能停在不同的位置和朝向，环境可能已被改变，而下一个交互目标可能位于当前视野之外。因此，依赖此类策略的智能体在无法将目标与当前观测建立对应关系时，可能难以继续执行下一个任务。为应对这一挑战，我们提出了 Attacca——一种新方法，它使用与……解耦的目标图像，在完整的“搜索到交互”轨迹上训练视觉目标条件策略。

    arXiv:2610.07785v1 Announce Type: new  Abstract: A central capability of embodied agents is to accomplish complex objectives through sequences of interdependent tasks. Yet existing visual goal-conditioned policies underlying these agents are typically evaluated on isolated interactions where the target is already visible, and thus do not capture the conditions that arise during continuous long-horizon task execution. In such settings, each task begins from the state left by the previous one: the agent may end at a different position and orientation, the world may have been modified, and the next interaction target may lie outside the current field of view. As a result, agents relying on such policies may struggle to proceed to the next task when they cannot ground their target in the current observation. To address this challenge, we propose Attacca, a new approach that trains visual goal-conditioned policies on complete search-to-interact trajectories using goal images decoupled from 
    
[^156]: 多智能体LLM推理中的持久记忆：它花费什么、带来什么、以及何时能被察觉

    Persistent Memory in Multi-Agent LLM Inference: What It Costs, What It Buys, and When You Can Tell

    [https://arxiv.org/abs/2610.07782](https://arxiv.org/abs/2610.07782)

    本文在三层多智能体LLM推理架构中实测发现，上下文分解可将峰值KV缓存从35.5 MiB降至14.3 MiB，而持久记忆层不仅增加0.368 MiB缓存开销，且在单问题基准上未带来任何可检测的准确率提升，这种零结果源于基准测试的结构性特点。

    

    将长上下文推理分解到协作智能体之间，可以限制每次调用的活跃KV缓存而非总证据量，这在KV缓存内存成为瓶颈时尤为重要。许多此类系统会添加一个持久层来存储和回忆推理轨迹，通常通过消融实验报告的准确率提升来验证其效果。我们在一个三层智能体架构上对两者进行了测量。上下文分解确实带来了成效：每个查询的峰值KV工作集为14.3 MiB，而单次传递和检索增强基线分别为35.5和35.3 MiB。持久层则没有带来收益：在八个受控数据集对、每组n=100的实验中，它使峰值缓存增加0.368 MiB [+0.167, +0.590]，且未产生可检测的准确率变化（+0.015，95% CI [-0.011, +0.046]）。我们认为这种零结果是结构性的：单问题基准测试为每个条目提供其独立的证据并独立评分，且正确性要求在条件之间重置已存储的推理轨迹，因此记忆回忆没有任何有价值的信息可供利用

    arXiv:2610.07782v1 Announce Type: new  Abstract: Decomposing long-context inference across cooperating agents bounds the active KV cache per call rather than total evidence, which matters when KV-cache memory binds. Many such systems add a persistent tier storing and recalling reasoning traces, usually validated by an ablation reporting an accuracy gain. We measure both on one three-tier agent architecture. Decomposition delivers: peak KV working set of 14.3 MiB per query against 35.5 and 35.3 MiB for single-pass and retrieval-augmented baselines. The persistent tier does not: across eight controlled dataset pairs at n=100 per arm it costs +0.368 MiB [+0.167, +0.590] of peak cache and produces no detectable accuracy change (+0.015, 95% CI [-0.011, +0.046]). We argue the null is structural: single-question benchmarks supply each item with its own evidence and score it independently, and correctness requires resetting stored traces between conditions, so recall has nothing informative to
    
[^157]: 量化对工具故障恢复的影响因提示词和评估设计而异

    Quantization Effects on Tool-Failure Recovery Vary Across Prompts and Evaluation Designs

    [https://arxiv.org/abs/2610.07781](https://arxiv.org/abs/2610.07781)

    该研究发现8比特与4比特量化对语言模型智能体工具故障恢复能力的影响并不稳定，比较结论会随提示词和评估目标（如评分任务的选择）而改变方向甚至反转，表明量化效果的结论高度依赖于评估设计。

    

    训练后量化降低了部署语言模型智能体的成本，但其对从临时性工具故障中恢复的影响可能取决于评估恢复能力的方式。我们在二十个确定性工具使用任务和五个提示词上，比较了Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的8比特与4比特变体。8比特与4比特在恢复能力上的比较结果会随提示词和评估目标而改变方向。在同一提示词下两个变体均能无故障完成的任务上，Llama的差异范围为0至+20.2个百分点，Qwen的差异范围为-50.0至+35.0个百分点。全流程点估计在所有五个提示词下都偏向8比特的Llama，而Qwen的比较结果则随提示词改变方向。评估目标的选择也可能使结论反转。在某一提示词下的Llama上，仅在每个变体各自无故障通过的任务上评分时，4比特领先17.5个百分点；而对两个变体在相同任务上评分时则没有差异。

    arXiv:2610.07781v1 Announce Type: new  Abstract: Post-training quantization reduces the cost of deploying language-model agents, but its effect on recovery from temporary tool failures can depend on how recovery is evaluated. We compare 8-bit and 4-bit variants of Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on twenty deterministic tool-use tasks and five prompts. The 8-bit-4-bit recovery comparison changes direction across prompts and evaluation targets. On tasks that both variants complete without faults under the same prompt, the difference ranges from 0 to +20.2 percentage points for Llama and from -50.0 to +35.0 points for Qwen. Full-pipeline point estimates favor 8-bit Llama under all five prompts, whereas the Qwen comparison changes direction across prompts. The evaluation target can also reverse the result. For Llama under one prompt, scoring each variant only on its own clean-passing tasks favors 4-bit by 17.5 points; scoring the same tasks for both variants gives no differen
    
[^158]: 迈向属性图聚类的“一个模型通用所有”基础模型

    Towards One-for-All Foundation Model for Attributed Graph Clustering

    [https://arxiv.org/abs/2610.07778](https://arxiv.org/abs/2610.07778)

    提出OFAG——一个面向属性图聚类的基础模型，仅需一次训练即可直接应用于多样化的属性图，无需针对特定图的训练、微调或超参数搜索。

    

    属性图聚类旨在通过联合利用节点属性和图拓扑结构来发现节点群组，然而其无监督的本质使得模型选择与适配天然困难。现有方法通常为每个输入图单独训练和调优一个模型，导致流程成本高昂且脆弱，往往无法迁移到具有不同特征空间、结构模式以及属性-结构相关性的图上。本文研究了一种“一个模型通用所有”的替代方案：能否训练一个单一模型，直接应用于多样化的属性图，而无需针对特定图的训练、微调或超参数搜索？我们提出了OFAG——一个面向属性图聚类的基础模型。OFAG基于先验数据拟合网络（Prior-data Fitted Networks）构建，从在潜在聚类、节点属性和图结构的宽泛先验下生成的合成属性图中学习一种可复用的聚类推理策略。

    arXiv:2610.07778v1 Announce Type: cross  Abstract: Attributed graph clustering aims to discover node groups by jointly exploiting node attributes and graph topology, yet its unsupervised nature makes model selection and adaptation inherently difficult. Existing methods typically train and tune a separate model for each input graph, leading to costly and fragile pipelines that often fail to transfer across graphs with different feature spaces, structural patterns, and attribute-structure correlations. In this paper, we study a one-for-all alternative: can a single model be trained once and directly applied to diverse attributed graphs without graph-specific training, fine-tuning, or hyperparameter search? We propose OFAG, a foundation model for attributed graph clustering. Building upon Prior-data Fitted Networks, OFAG learns a reusable clustering inference strategy from synthetic attributed graphs generated under broad priors over latent clusters, node attributes, and graph structures.
    
[^159]: OTel：开放电信AI数据集、基准测试与模型

    OTel: Open Telco AI Datasets, Benchmarks, and Models

    [https://arxiv.org/abs/2610.07766](https://arxiv.org/abs/2610.07766)

    OTel发布了一个统一的开放电信AI资源，提供面向检索、重排序、指令微调及安全/拒答的电信数据集，以及采用开放训练方案后训练的30个基线模型（10个嵌入模型、3个重排序器、17个语言模型）。

    

    我们提出了Open Telco（OTel），这是一个开放的电信AI资源，发布了用于检索、重排序、指令微调以及安全/拒答的衍生电信数据集，同时提供了30个全参数后训练的基线模型，涵盖10个嵌入模型、3个重排序器和17个语言模型。社区已经大量使用该资源：截至2026年5月3日，发布的模型已被下载超过1600万次，该项目在全球范围内获得了157篇以上的媒体报道。在先前的开放电信数据集和基准测试的基础上，OTel在一个统一资源中提供了有文档记录的电信数据源、留出的评估分区、经过训练的嵌入模型、重排序器、基于上下文的大语言模型以及安全/拒答数据。每个基线模型都从开源权重模型出发，采用开放的训练方案在OTel衍生数据上进行后训练，然后在留出的OTel评估分区上进行评估。OTel后训练改进（摘要在此处截断）

    arXiv:2610.07766v1 Announce Type: new  Abstract: We present Open Telco (OTel), an open telecom AI resource that releases derived telecom datasets for retrieval, reranking, instruction tuning, and safety/abstention, together with 30 full-parameter post-trained baselines spanning 10 embedding models, 3 rerankers, and 17 language models. The community has already engaged substantially with the resource: as of May 3, 2026, the released models have been downloaded over 16 million times and the project has received 157+ pieces of media coverage worldwide. Building on prior open telecom datasets and benchmarks, OTel provides documented telecom data sources, held-out evaluation partitions, trained embedding models, rerankers, context-grounded LLMs, and safety/abstention data in one unified resource. Each baseline starts from an open-weight model and is post-trained on OTel-derived data using an open training recipe, then evaluated on held-out OTel evaluation partitions. OTel post-training impr
    
[^160]: 没有Transformer能胜过六个协变量：基于童年作文对抑郁症状的长期预测

    No Transformer Beats Six Covariates: Long-Horizon Prediction of Depressive Symptoms from Childhood Essays

    [https://arxiv.org/abs/2610.07764](https://arxiv.org/abs/2610.07764)

    该研究发现，在利用11岁儿童作文预测其23岁抑郁症状的长期任务中，基于六个童年协变量的简单逻辑回归（AUC-ROC 0.737）显著优于所有文本模型，包括微调Transformer、词袋模型、冻结嵌入和零样本大语言模型（最佳仅0.670）。

    

    自然语言处理（NLP）模型能够检测在症状测量时间附近所写文本中的抑郁相关语言，但预训练Transformer能否从十二年前所写的文本中预测抑郁症状，在很大程度上尚未得到验证。在英国出生队列研究“全国儿童发展研究”中，我们利用同一批人在11岁时写的作文来预测其23岁时可能出现的抑郁症状。我们的基线模型——一个基于六个童年期协变量的逻辑回归——优于所有仅使用作文文本的模型：七个微调的Transformer、一个词袋模型、冻结嵌入以及四个零样本大语言模型。基线模型的受试者工作特征曲线下面积（AUC-ROC）为0.737，而在主随机种子下最佳Transformer仅为0.670，且任何附加的文本得分都未能显著提升基线的AUC-ROC。五个领域预训练的Transformer模型中，没有一个能显著胜过其通用领域对照模型。

    arXiv:2610.07764v1 Announce Type: cross  Abstract: Natural language processing (NLP) models can detect depression-related language in text written near the time symptoms are measured, but whether pretrained transformers can predict depressive symptoms from text written twelve years earlier is largely untested. In the National Child Development Study, a British birth cohort, we predict probable depressive symptoms at age 23 from essays the same people wrote at age 11. Our baseline, a logistic regression on six childhood covariates, outperforms every text model that sees only the essay: seven fine-tuned transformers, a bag-of-words model, frozen embeddings and four zero-shot large language models. Its area under the receiver operating characteristic curve (AUC-ROC) is 0.737 against 0.670 for the best transformer on the primary seed, and no added text score detectably raises the baseline's AUC-ROC. None of the five domain-pretrained transformers detectably beats its general-domain control
    
[^161]: ST-Bench：面向科研任务的多智能体系统生成的时空基准测试

    ST-Bench: A Spatial-Temporal Benchmark for Multi-Agent System Generation on Scientific Research Tasks

    [https://arxiv.org/abs/2610.07763](https://arxiv.org/abs/2610.07763)

    提出了ST-Bench基准测试，用于检验多智能体系统在复杂科学数据分析任务上能否优于单智能体、优势有多大以及需要付出多少额外代价。

    

    基于大语言模型的多智能体系统（MAS）的快速发展表明，在可执行测试能够提供二元成功信号的编程、数学和问答任务上，它们在很大程度上优于单个智能体。然而，这种优势能否迁移到真实的科学数据分析中仍未得到检验。我们提出了 ST-Bench，一个旨在回答两个问题的基准测试：多智能体系统在复杂的科学数据分析任务上是否优于单个智能体；如果是，优势有多大，以及需要付出何种额外代价。ST-Bench 包含 100 个改编自已发表地球科学研究的数据科学任务，涵盖水文学、农业和湿地甲烷研究领域，并扩展为 2,067 个基于其他已发表研究且经领域专家验证的查询。利用 ST-Bench，我们在两种训练协议下评估了五种最新的 MAS 生成方法，并与基于相同 GPT-5 骨干模型的单智能体基线进行比较。在十个 MAS 配置中，有九个超过了最便宜的单智能体（原文摘要在此处截断）。

    arXiv:2610.07763v1 Announce Type: new  Abstract: The rapid progress of LLM-based multi-agent systems (MAS) has shown that they largely outperform single agents on coding, math, and QA tasks, where executable tests provide a binary success signal. Whether this advantage transfers to real scientific data analysis remains untested. We introduce ST-Bench, a benchmark designed to answer two questions: whether MAS outperform single agents on complex scientific data analysis tasks, and if so, by how much and at what additional cost. ST-Bench contains 100 data science tasks adapted from published Earth science studies across hydrology, agriculture, and wetland methane research, expanded into 2,067 queries grounded in additional published studies and validated by domain experts. Using ST-Bench, we evaluate five recent MAS generation methods under two training protocols, against single-agent baselines on the same GPT-5 backbone. Nine of the ten MAS configurations exceed the cheapest single-agent
    
[^162]: 面向可解释推荐的方面表示对比学习方法

    Contrastive Learning for Aspect Representation towards Explainable Recommendation

    [https://arxiv.org/abs/2610.07761](https://arxiv.org/abs/2610.07761)

    该论文提出CLARER推荐模型，通过Transformer编码器和对比学习从评论中提取方面特征，并与评分信息融合，同时提升了推荐的准确性与可解释性。

    

    在这项工作中，我们提出了一种新颖的推荐模型 CLARER（Contrastive Learning for Aspect Representation towards Explainable Recommendation，面向可解释推荐的方面表示对比学习），该模型将从文本评论中学习到的方面特征与评分信息相融合，以提升推荐的准确性和可解释性。我们提出的框架通过结合基于评分的特征和来自评论的基于方面的特征来学习用户和物品的表示。具体而言，基于评分的特征通过多层感知机（MLP）模型学习，而特定方面的评论表示则使用 Transformer 编码器捕获语义信息，并通过对比学习更好地区分用户偏好。为了提供解释，我们训练了一个 Transformer 解码器，将来自评分特征和方面特征的最终用户与物品表示作为上下文。在三个基准数据集上的实验结果表明，我们的模型

    arXiv:2610.07761v1 Announce Type: cross  Abstract: In this work, we propose a novel recommendation model, CLARER (Contrastive Learning for Aspect Representation towards Explainable Recommendation) that integrates aspect features learned from textual reviews with rating information to improve the accuracy and explainability of recommendations. Our proposed framework learns user and item representations by combining rating-based features and aspect-based features from reviews. Specifically, rating-based features are learned through a multi-layer perceptron (MLP) model, while aspect-specific review representations are learned using a transformer encoder to capture the semantic information and contrastive learning to better distinguish user preferences. To provide explanations, we train a transformer decoder, using the final representations of users and items from both rating and aspect-based features as context. Experimental results in three benchmark data sets demonstrate that our model 
    
[^163]: 越晚越好：分布偏移下的视觉Transformer Token削减

    Later Is Better: Token Reduction for ViTs Under Distribution Shift

    [https://arxiv.org/abs/2610.07758](https://arxiv.org/abs/2610.07758)

    提出一种单参数的晚集中幂律token削减时间表，在无需额外推理成本的前提下，显著缩小了视觉Transformer在分布偏移下免训练token削减造成的精度差距。

    

    免训练的token削减通过在各层移除冗余token来加速视觉Transformer，只需一小部分计算量即可恢复原始模型的大部分精度。然而，这些方法主要是在干净数据上设计和评估的，在真实世界的分布偏移下，其与未压缩模型之间的精度差距会随移除率的增加而扩大。我们证明这一差距由削减时间表决定，即移除操作随网络深度的分布情况，而这通常被固定为一个实现细节。具体而言，我们提出了一种单参数的晚集中幂律时间表，在不增加任何额外推理成本的情况下，其分布外精度持续优于平坦时间表。在ImageNet-C上使用DeiT-S，晚集中时间表在26%的计算量削减下弥补了83%的精度差距（+1.17个百分点），在较轻的7%削减下弥补了99%的差距（+0.26个百分点）。这一增益不能归因于保留了更多token或使用了额外计算量：在保持与平坦时间表相同计算量的情况下，晚集中时间表仍然……

    arXiv:2610.07758v1 Announce Type: cross  Abstract: Training-free token reduction accelerates vision transformers by removing redundant tokens across layers, recovering most of the original accuracy at a fraction of the compute. These methods, however, are designed and evaluated primarily on clean data, and under real-world distribution shift their accuracy gap to the uncompressed model widens with the removal rate. We show that this gap is governed by the reduction schedule, the depth profile of removal, usually left fixed as an implementation detail. Concretely, we introduce a one-parameter late-concentrated power-law schedule that consistently improves out-of-distribution accuracy over flat at no extra inference cost. On ImageNet-C with DeiT-S, the late schedule closes 83% of that gap at a 26% compute reduction (+1.17pp), and 99% of it at a lighter 7% reduction (+0.26pp). The gain cannot be attributed to retaining more tokens or using extra compute: held to flat's compute, the late s
    
[^164]: 从证据到行动：工具使用型智能体如何失败

    From Evidence to Action: How Tool-Using Agents Fail

    [https://arxiv.org/abs/2610.07753](https://arxiv.org/abs/2610.07753)

    提出包含656个案例的SafeActBench基准，系统揭示工具使用型智能体在“证据到行动”链条中的失败模式——失败往往在执行前就因调查不完整或过早行动而出现，多动作工作流还会暴露未解决的先决条件和不完整执行问题。

    

    使用工具的智能体会对外部状态做出重要改变，然而正确的结果并不能保证其行动是建立在事先确立的证据之上的。我们研究了当智能体从决定是否采取行动，到执行单个动作乃至相互依赖的工作流时，这条从证据到行动的链条在何处断裂。在十种模型-框架配置中，强大的静态动作评估能力可能与弱得多的交互式执行能力并存。失败往往在执行之前就已开始：智能体在调查不完整的情况下停止，或在所需证据尚未确立时就贸然行动。一旦所需证据已获取，单动作执行通常是可靠的，而多动作工作流则会额外暴露出未解决的先决条件和不完整的执行问题。为支持这一分析，我们推出了SafeActBench，其中包含跨越六个操作领域的656个案例和五种协议，涵盖从静态动作判断、经调查后的不行动，到单动作执行等递进场景。

    arXiv:2610.07753v1 Announce Type: cross  Abstract: Tool-using agents make consequential changes to external state, yet correct outcomes do not guarantee that their actions were supported by evidence established beforehand. We study where this evidence-to-action chain breaks as agents move from deciding whether to act to executing single actions and dependent workflows. Across ten model-harness configurations, strong static action assessment can coexist with much weaker interactive execution. Failures often begin before execution: agents stop with incomplete investigation or act before required evidence is established. Once required evidence is obtained, single-action execution is usually reliable, while multi-action workflows additionally expose unresolved prerequisites and incomplete execution. For this analysis, we introduce SafeActBench, comprising 656 cases across six operational domains and five protocols that progress from static action judgment and investigated non-action to sin
    
[^165]: 大语言模型在噪声证据下的推理能力如何？一个主动视觉推理基准

    How Well Do LLMs Reason with Noisy Evidence? An Active Visual Reasoning Benchmark

    [https://arxiv.org/abs/2610.07751](https://arxiv.org/abs/2610.07751)

    提出了 VisualNoiseQA 基准，让纯文本 LLM 通过迭代查询被视为随机视觉传感器的 VLM，并利用基于自一致性的不确定性信号，在噪声证据下进行主动视觉推理，自主决定下一步询问内容及何时停止。

    

    现实世界中的推理很少能简化为静态问答：智能体必须主动从通常嘈杂且不可靠的工具和传感器中收集信息。然而，大多数现有的主动推理基准都假设环境反馈是可信的，或者在不暴露明确、经校准的不确定性信号的情况下引入噪声，这使得当证据本身不确定时 LLM 应如何推理的问题悬而未决。我们提出了 VisualNoiseQA，这是一个用于在噪声视觉反馈下进行主动推理的新型基准。一个纯文本 LLM 必须通过迭代查询一个固定的、现成的 VLM（将其视为随机视觉传感器）来解决 VQA 问题。对于每个查询，我们抽取多个样本并通过自一致性暴露经验性不确定性信号，使推理者能够从不同角度进行探查，并决定接下来该问什么以及何时停止。我们的构建是自动化且可扩展的：从多样化的 VQA 数据源和两个噪声 VLM 出发，我们……

    arXiv:2610.07751v1 Announce Type: new  Abstract: Real-world reasoning rarely reduces to static question answering: agents must actively gather information from tools and sensors that are often noisy and unreliable. Yet most existing active reasoning benchmarks assume that environmental feedback is trustworthy, or introduce noise without exposing an explicit, calibrated uncertainty signal, leaving open how LLMs should reason when the evidence itself is uncertain. We introduce VisualNoiseQA, a novel benchmark for active reasoning under noisy visual feedback. A text-only LLM must solve VQA problems by iteratively querying a fixed, off-the-shelf VLM treated as a stochastic visual sensor. For each query, we draw multiple samples and expose an empirical uncertainty signal via self-consistency, enabling the reasoner to probe from different angles and decide what to ask next and when to stop. Our construction is automatic and scalable: starting from diverse VQA sources and two noisy VLMs, we r
    
[^166]: Cleave：通过解耦的代数搜索与算子调度实现张量程序优化的扩展

    Cleave: Scaling Tensor Program Optimization via Decoupled Algebraic Search and Operator Scheduling

    [https://arxiv.org/abs/2610.07742](https://arxiv.org/abs/2610.07742)

    提出 Cleave 编译器，通过将代数变换搜索与算子调度解耦——先在符号形状图上做超级优化、再在具体形状上调度——自动生成媲美 FlashAttention 等手工优化内核的高效张量程序。

    

    诸如 FlashAttention 和 FlashDecoding 这样的优化内核对于加速当今的大模型至关重要。它们大多由专家手工编写，因为现有的机器学习编译器无法达到其效率。生成此类内核需要融合包含多个归约的计算，这既需要对计算图进行代数变换，又需要对变换后的图进行算子调度。遗憾的是，若将两者联合搜索，所产生的搜索空间过于庞大而难以遍历。我们提出了 Cleave，一个建立在符号解耦之上的机器学习编译器：Cleave 先对具有符号形状的计算图执行超级优化来发现变换，然后在具体形状上对得到的每个图进行调度。将形状表示为符号使等价性检查的开销很低，并允许一个带有符号分割数目的新 Split 算子沿归约维度进行并行化。Cleave 的调度器通过迭代方式融合包含多个归约的计算图……

    arXiv:2610.07742v1 Announce Type: cross  Abstract: Optimized kernels such as FlashAttention and FlashDecoding are crucial for accelerating today's large models. Most of them are handwritten by experts because existing ML compilers cannot match their efficiency. Producing such kernels requires fusing computations with multiple reductions, which requires both algebraic transformation of the computation graph and operator scheduling of the transformed graph. Unfortunately, searching the two jointly yields a space too large to navigate. We propose Cleave, an ML compiler built on symbolic decoupling: Cleave discovers transformations by performing superoptimization on a graph with symbolic shapes, and then schedules each resulting graph on concrete shapes. Representing shapes as symbols makes equivalence checking cheap and lets a new Split operator, with a symbolic split count, parallelize along a reduction dimension. Cleave's scheduler fuses graphs with multiple reductions through iterative
    
[^167]: 在嵌入空间中通过强化学习学习检索

    Learning to Retrieve via Reinforcement Learning in Embedding Space

    [https://arxiv.org/abs/2610.07731](https://arxiv.org/abs/2610.07731)

    该论文提出RELER强化学习框架，通过从vMF分布采样嵌入动作、结合RLOO基线的REINFORCE算法以及减少采样噪声的条件均值投影（CMP）技术，使现有嵌入模型能够直接在嵌入空间中学习检索并对齐任务特定的奖励。

    

    密集检索模型通常使用对比目标进行训练，这类目标虽然能学习有效的表示，但无法直接优化检索指标或下游任务性能。为了解决这一问题，我们提出了RELER（面向检索的强化学习），这是一个强化学习框架，能够使现有的嵌入模型直接在嵌入空间中学习检索，并与任务特定的奖励对齐。我们通过以下方式训练RELER：从以归一化编码器输出为中心的von Mises-Fisher（vMF）分布中采样单位长度的查询和文档嵌入动作，将由此产生的检索或下游结果评分作为奖励，并使用留一法基线（RLOO）通过REINFORCE算法更新编码器。由于在高维嵌入空间中进行探索容易受到采样噪声的影响，我们进一步提出了条件均值投影（CMP），将每个采样的嵌入投影到低维子空间上……

    arXiv:2610.07731v1 Announce Type: cross  Abstract: Dense retrieval models are typically trained with contrastive objectives that learn effective representations but do not directly optimize retrieval metrics or downstream task performance. To address this problem, we introduce RELER (REinforcement LEarning for Retrieval), a reinforcement learning framework that enables existing embedding models to learn to retrieve directly in embedding space and align to task-specific rewards. We train RELER by sampling unit-length query and document embedding actions from von Mises-Fisher (vMF) distributions centered on normalized encoder outputs, scoring the resulting retrieval or downstream outcomes as rewards, and updating the encoder with REINFORCE using a leave-one-out baseline (RLOO). As exploration in the high-dimensional embedding space is prone to sampling noise, we further propose conditional-mean projection (CMP), which projects each sampled embedding onto the low-dimensional subspace span
    
[^168]: SanSi：一种用于系统1.5思维的循环式类型化决策模型

    SanSi: A Looped Typed Decision Model for System 1.5 Thinking

    [https://arxiv.org/abs/2610.07730](https://arxiv.org/abs/2610.07730)

    SanSi提出“系统1.5思维”——通过多次循环复用模型层在不生成文本的情况下修正隐藏状态，将预训练循环语言模型转化为类型化决策模型，在59个数据源的10,027个测试决策上达到72.0%准确率，比同结构非循环模型高出13.5个百分点。

    

    类型化决策模型在不生成文本的情况下回答一个预先声明的问题：决策头在单次前向传播中为每个声明的选项返回一个概率。单次前向传播快速而直观，属于系统1思维。我们研究了介于单次传播与生成式推理之间的方法：循环，即在输出一次类型化读数之前，将相同的层递归地应用多次。每一次循环都让模型在提交答案之前修正其隐藏状态，而无需生成任何词元；我们将其称为系统1.5思维。我们提出了SanSi，它将一个预训练的循环语言模型转化为类型化决策模型。选项概率在每次循环后被读取，且每次循环都使用适当的评分规则进行训练，因此单个模型可以在一次运行中服务于从一次循环到八次循环的任意计算预算。在来自59个数据源的10,027个测试决策上，SanSi达到了72.0%的准确率，比用相同结构训练的非循环模型高出13.5个百分点。

    arXiv:2610.07730v1 Announce Type: cross  Abstract: Typed decision models answer a declared question without generating text: a decision head returns a probability for each of the declared options in a single forward pass. A single pass is fast, intuitive System 1 thinking. We study what lies between one pass and generated reasoning: looping, in which the same layers are recursively applied several times before one typed readout. Each loop lets the model revise its hidden state before it commits to an answer, without generating a token; we call this System 1.5 thinking. We propose SanSi, which turns a pre-trained looped language model into a typed decision model. The option probabilities are read after every loop, and every loop is trained with a proper scoring rule, so that one model serves every budget from one loop to eight in a single run. On 10,027 test decisions from 59 sources, SanSi reaches 72.0% accuracy: 13.5 points above a non-looped model of the same shape trained with the s
    
[^169]: PERSIST：面向全双工语音对话的跨会话“谁-什么-何时”记忆系统

    PERSIST: Who-What-When Memory Across Sessions for Full-Duplex Spoken Dialogue

    [https://arxiv.org/abs/2610.07725](https://arxiv.org/abs/2610.07725)

    PERSIST是一个面向多用户多会话语音对话的持久记忆系统，通过联合建模“谁-什么-何时”的3W评分机制实现准确的跨会话记忆检索，并复用对话主干中间表示以支持低延迟的实时全双工交互。

    

    现代语音助手可能被多个用户共享，因此应当能够回答关于早期对话的问题，例如“我最初计划什么时候离开？”，或者根据过去的交互针对不同用户调整自身行为。这需要的不仅仅是检索主题相似的文本片段：助手必须识别当前说话人，恢复相关的过去状态，并将其与后续的修订区分开来。我们提出了PERSIST，一个面向多会话、多说话人语音对话的持久记忆系统，它显式地建模“谁”、“什么”和“何时”。PERSIST将跨会话历史结构化为可读的事件记录，并通过3W联合评分机制进行检索，该机制融合了语义内容、声学说话人身份和时间状态三种信息。为了实现实时全双工交互，PERSIST还复用对话主干模型中的中间表示，避免了对查询音频的重新编码，从而降低检索延迟。

    arXiv:2610.07725v1 Announce Type: new  Abstract: Modern voice assistants may be shared by multiple users and should be able to answer questions about earlier conversations such as "When did I originally plan to leave?" or adapt their behavior to individual users based on past interactions. This requires more than retrieving a topically similar passage: the assistant must identify the current speaker, recover the relevant past state, and distinguish it from later revisions. We present PERSIST, a persistent memory system for multi-session, multi-speaker spoken dialogue that explicitly models Who, What, and When. PERSIST structures cross-session histories into readable event records and retrieves them with a 3W joint scoring mechanism that combines semantic content, acoustic speaker identity, and temporal state. For real-time full-duplex interaction, PERSIST further reuses intermediate representations from the dialogue backbone, avoiding query-audio re-encoding and reducing retrieval late
    
[^170]: 先证据后采样：面向推荐系统的可解释隐式负候选发现

    Evidence Before Sampling: Interpretable Implicit Negative Candidate Discovery for Recommendation

    [https://arxiv.org/abs/2610.07708](https://arxiv.org/abs/2610.07708)

    该论文提出“先证据后采样”的隐式负候选发现框架，将用户行为模式编码为符号化规则并通过证据强度排序，再由大语言模型结合业务目标进行解释，从而为推荐系统提供可解释且与业务对齐的隐式负样本。

    

    推荐系统从观测到的用户-商品交互中学习，但显式的负反馈通常是不可得的。由于深度学习模型在训练中需要负样本信号，负采样方法通常将选定的未观测交互直接视为负样本。然而，一次缺失的交互既不能解释用户为何对某商品不感兴趣，也不能说明是否有足够的证据将其标注为负样本。这在商业推荐场景中尤为重要，因为负信号应当是可解释的，并与业务目标保持一致。我们提出了隐式负候选发现问题，旨在识别由已观测用户行为所支持的未观测交互。我们将这些行为模式编码为符号化规则，基于支持度、信息量和商品相关性对其进行评分，并依据证据强度对保留的规则进行排序。随后，大语言模型（LLM）结合业务目标与领域知识对这些保留的规则进行解释……（原文摘要在此处截断）

    arXiv:2610.07708v1 Announce Type: new  Abstract: Recommender systems learn from observed user-item interactions, but explicit negative feedback is often unavailable. Since deep learning models require negative signals for training, negative sampling methods typically treat selected unobserved interactions as negatives. However, a missing interaction does not explain why a user is uninterested in an item or whether there is sufficient evidence to label it negative. This is especially important in business recommendation, where negative signals should be interpretable and aligned with business objectives. We formulate implicit negative candidate discovery to identify unobserved interactions supported by observed customer behavior. We encode these patterns as symbolic rules, score them based on support, informativeness, and product relevance, and rank the retained rules by evidence. An LLM then interprets the retained rules using business objectives and domain knowledge; the interpretatio
    
[^171]: AgentMemGate：解决对话助手记忆中的推测污染问题

    AgentMemGate: Addressing Speculation Contamination in Conversational Assistant Memory

    [https://arxiv.org/abs/2610.07707](https://arxiv.org/abs/2610.07707)

    提出AgentMemGate——一种写入时门控机制，通过将对话中提取的语句分类为推测、已完成事件、更正或其他类型，防止用户表达的计划（推测）被错误地作为事实写入对话助手的长期记忆，从而解决“推测污染”问题。

    

    具有长期记忆的对话式AI助手会从用户消息中提取事实并存入存储库，供后续对话查询使用。用户陈述的计划可能会作为事实进入该存储库：例如，一个可能搬到西雅图的用户可能被记录为已经住在那里。我们将这种现象称为“推测污染”。最终状态的记忆基准测试无法捕捉这类错误，因为它们不探测中间状态，且包含的未解决推测很少。我们提出了AgentMemGate，这是一个针对配置文件存储记忆的写入时门控机制，可将提取的语句分类为推测、已完成事件、更正或其他类型。推测内容不会进入记忆，而是附带条件，用于管理后续的升级（转为事实）或删除。我们还贡献了一个多会话对话数据集，其中的计划包括已确认、已放弃或未解决的情况。在147个对话的保留测试集上，Mem0和Graphiti分别将35.2%和27.3%的待处理未解决计划错误地断言为当前状态。

    arXiv:2610.07707v1 Announce Type: new  Abstract: Conversational AI assistants with long-term memory extract facts from user messages into a store consulted in later conversations. A stated plan can enter that store as fact: a user who might move to Seattle may be recorded as already living there. We call this speculation contamination. Final-state memory benchmarks miss this error because they do not probe intermediate state and include few unresolved speculations. We present AgentMemGate, a write-time gate for profile-store memory that classifies extracted statements as speculation, completed event, correction, or other. Speculations remain outside memory, with conditions governing later promotion or deletion. We also contribute a dataset of multi-session conversations in which plans are confirmed, abandoned, or left unresolved. On our 147-conversation held-out set, Mem0 and Graphiti assert unresolved plans as current state for 35.2% and 27.3% of pending plans. On the core benchmark, 
    
[^172]: WASD：基于Wasserstein距离的大语言模型知识蒸馏方法

    WASD: Wasserstein-based Knowledge Distillation for Large Language Models

    [https://arxiv.org/abs/2610.07706](https://arxiv.org/abs/2610.07706)

    该论文提出了WASD方法，通过由词元嵌入构建代价矩阵的Wasserstein距离，将词元级别语义信息融入大语言模型的知识蒸馏中，并借助Sinkhorn散度实现高效优化。

    

    自回归大语言模型（LLM）的能力迅速提升，但其不断增长的规模在推理时带来了巨大的计算和内存成本。知识蒸馏（KD）通过对齐离散概率分布，将大型教师模型的知识迁移到较小的学生模型中，提供了一种实用的解决方案。然而，现有的LLM知识蒸馏方法主要依赖于在每个词汇表索引处通过概率值来评估差异的散度度量，没有显式地利用词元级别的语义信息。我们提出了针对大语言模型的基于Wasserstein距离的知识蒸馏方法（WASD），该方法通过基于Wasserstein的距离并利用由词元嵌入导出的代价矩阵，将词元级别的语义信息融入蒸馏过程。为确保计算上的可处理性，我们采用Sinkhorn散度，并推导出一个梯度等价的目标函数，可以在不引入额外计算开销的情况下进行高效优化。

    arXiv:2610.07706v1 Announce Type: cross  Abstract: Autoregressive large language models (LLMs) have rapidly advanced in capability, but their increasing scale comes with substantial computational and memory costs at inference time. Knowledge distillation (KD) offers a practical solution by transferring knowledge from a large teacher model to a smaller student model via alignment of discrete probability distributions. However, existing KD methods for LLMs primarily rely on divergences that evaluate discrepancies through probability values at each vocabulary index, without explicitly leveraging token-level semantic information. We propose Wasserstein-based knowledge distillation (WASD) for LLMs, which incorporates token-level semantic information via the Wasserstein-based distance with a cost matrix derived from token embeddings. To ensure computational tractability, we adopt the Sinkhorn divergence and derive a gradient-equivalent objective that can be efficiently optimized without intr
    
[^173]: 帧级标签在热成像视频中小型无人机点检测上能做与不能做的事

    What Frame-Level Labels Can and Cannot Do for Small-UAV Point Detection in Thermal Video

    [https://arxiv.org/abs/2610.07705](https://arxiv.org/abs/2610.07705)

    该论文探究了仅使用帧级目标存在/不存在标签（无需空间标注）来训练小型无人机点检测模型的检测能力、学习行为与局限，并在两个热红外数据集上评估了其定位命中率与检测率表现。

    

    无人机（UAV）的日益广泛应用提升了基于图像的无人机检测的重要性。基于学习的检测器通过图像和标注进行训练，而标注的类型决定了训练时可利用的信息。我们研究在传感器或场景发生变化、导致为新增训练数据制作空间标注变得繁重的情况下，如何仅从帧级目标存在/不存在标签中学习定位。我们分析了一种现有的小型无人机点检测架构——仅使用存在/不存在标签训练且无需外部检测器——的检测能力、学习行为及潜在应用。该架构冻结通过分类学习获得的空间特征，并用相同的帧标签训练一个读出模块，以生成空间得分图和点检测结果。我们在CST Anti-UAV和Anti-UAV410两个热红外数据集上，评估了该方法的定位命中率和在虚警约束下的检测率（摘要原文在此处截断）。

    arXiv:2610.07705v1 Announce Type: cross  Abstract: The growing use of unmanned aerial vehicles (UAVs) has increased the importance of image-based UAV detection. Learning-based detectors are trained on imagery and annotations, with annotation type determining the information available during training. We focus on learning localization from frame-level target presence/absence labels when sensor or scene changes make spatial annotations for additional training burdensome. We analyze the detection capability, learning behavior, and potential applications of an existing architecture for point detection of small UAVs, trained with presence/absence labels and requiring no external detector. The architecture freezes spatial features learned through classification and trains a readout with the same frame labels to produce spatial score maps and point detections. On two thermal infrared datasets, CST Anti-UAV and Anti-UAV410, we evaluate localization hit rates and detection rates under false-ala
    
[^174]: 论准入门槛的边界：量化策略研究中“证伪优先”选择的注入真值研究

    On the Boundary of Admission Gates: An Injected-Truth Study of Falsification-First Selection in Quantitative Strategy Research

    [https://arxiv.org/abs/2610.07701](https://arxiv.org/abs/2610.07701)

    该研究提出注入真值验证协议评估量化策略研究中的准入门槛，发现门槛虽能消除弱信号情形下的虚假发现但采纳率仅降至1-7%，且基于绝对收益而非超额收益的准则会错误拒绝包括真实信号在内的所有候选策略。

    

    策略研究混淆了两个问题：找到有利可图的规则，以及证明该发现并非搜索运气所致。后者需要“准入门槛”——即在采纳结论之前必须满足的统计标准——然而门槛是否有效、代价几何，仍未经过检验。我们提出一种注入真值协议，并配有随机采纳对照组，该对照组以与门槛相同的速率采纳策略；只有当门槛的表现优于该对照时，它才真正携带信息，而不只是提高了采纳阈值。在合成数据与真实校准的面板数据上，门槛在弱信号情形下消除了虚假发现，但将采纳率削减至1-7%，而在信号较强时则没有任何额外增益。最重要的是，基于绝对收益而非超额收益计算的准则会默默拒绝所有候选策略，包括真实的信号。关键词：多重检验、回测过拟合、策略准入、注入真值验证、超额收益、虚假发现率

    arXiv:2610.07701v1 Announce Type: new  Abstract: Strategy research conflates two problems: finding a profitable rule, and establishing that the finding is not search luck. The latter calls for admission gates -- statistical criteria that must be satisfied before a conclusion is adopted -- yet whether gates work, and at what cost, remains untested. We introduce an injected-truth protocol with a random-admission control that adopts at the same rate as the gate; only if the gate beats this control does it carry information rather than merely raise a threshold. Across synthetic and real-calibrated panels, gates eliminate false discoveries in the weak-signal regime but cut adoption to 1--7%, and add nothing when signals are strong. Most importantly, criteria computed on absolute rather than excess returns silently reject every candidate, including true signals. Keywords: multiple testing, backtest overfitting, strategy admission, injected-truth validation, excess returns, false discovery ra
    
[^175]: BluffJAX：基于JAX的对抗性不完全信息游戏

    BluffJAX: Adversarial Imperfect Information Games in JAX

    [https://arxiv.org/abs/2610.07686](https://arxiv.org/abs/2610.07686)

    BluffJAX是一个基于JAX的开源对抗性不完全信息游戏套件，支持GPU加速的高吞吐量并行模拟（每秒可达数亿样本），收录了德州扑克、库恩扑克等多种经典及全新游戏，为强化学习的博弈论方法研究开辟了新的挑战与方向。

    

    我们介绍了BluffJAX：一个基于JAX的开源对抗性不完全信息游戏套件。我们提供了专为高模拟吞吐量和GPU加速器并行化而设计的游戏的规范实现。该套件既包括经过充分研究的基准游戏，如德州扑克和库恩扑克，也包括此前在强化学习研究中尚未被研究过的游戏，如吹牛牌、梭哈和Kemps。我们希望通过实现多样化的游戏机制和难度，为强化学习引入新的挑战，并促进博弈论方法的新研究方向。我们在单GPU和多GPU设置下对环境的吞吐量性能和内存使用进行了基准测试，展示了每秒高达数亿样本的扩展能力，从而证明了相比相关GPU和CPU库使用BluffJAX的优越性。我们对强化学习、树搜索和博弈求解算法进行了基准测试……

    arXiv:2610.07686v1 Announce Type: new  Abstract: We introduce BluffJAX: an open-source suite of adversarial imperfect information games in JAX. We provide canonical implementations of games designed for high simulation throughputs and parallelization on GPU accelerators. Our suite consists of well-studied benchmarks such as Texas Hold'Em Poker and Kuhn Poker, as well as games that have not been previously studied in reinforcement learning research, such as Bluff, Stud Poker, and Kemps. We hope that implementing a variety of game mechanics and difficulties will introduce new challenges and foster novel research directions in game-theoretic methods for RL. We benchmark the throughput performance and memory usage of our environments in single and multi-GPU settings, demonstrating scaling of up to hundreds of millions of samples per second, and motivating the usage of BluffJAX over related GPU and CPU-based libraries. We benchmark reinforcement learning, tree search, and game-solving algor
    
[^176]: 面向个性化生成的频率感知扩散模型中双图像参考解耦

    Disentangling Dual Image References in Frequency Aware Diffusion Models for Personalized Generation

    [https://arxiv.org/abs/2610.07684](https://arxiv.org/abs/2610.07684)

    提出 Dual-FDM 频率感知扩散模型，通过解耦去噪过程中的混合频段纠缠，同时利用双图像参考（定制参考与颜色风格参考）实现定制风格迁移和颜色风格迁移的个性化图像生成。

    

    个性化图像生成旨在以参考图像为条件合成文本驱动的图像，目前主要将生成任务分为针对前景的图像定制和针对背景的风格迁移。现有的扩散模型方法在去噪过程中存在文本与背景（图像定制任务中）以及文本与前景（风格迁移任务中）不对齐的问题。据我们观察，这些问题的根源在于去噪过程中混合频段之间的纠缠。为解决这一显著局限，本文研究了基于双参考——定制参考以及颜色和风格参考——的个性化生成，提出了一种在频率感知扩散模型中解耦这些双图像参考的新范式，称为 Dual-FDM，通过解耦不同的频段，同时解决两个关键的个性化图像生成任务：定制风格迁移和颜色风格迁移。

    arXiv:2610.07684v1 Announce Type: cross  Abstract: Personalized image generation aims to synthesize text-driven images conditioned on reference images, while mainly casting the generation as image customization for foreground and style transfer for background. Previous arts of diffusion models suffers from the text misalignment with background for image customization and foreground for style transfer during the denoising process. Such facts, as we observed, rooted from the entanglement among hybrid frequency bands during the denoising process. To address such salient limitation, in this paper, we study personalized generation based on dual references - customization and color and style reference - and propose a paradigm to disentangle these Dual image references within Frequency-aware Diffusion Models, dubbed Dual-FDM, to simultaneously tackle two crucial personalized image generation tasks: customization style transfer and color style transfer, by disentangling different frequency ban
    
[^177]: EigenDEXplore：基于人类先验的灵巧操作结构化探索

    EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors

    [https://arxiv.org/abs/2610.07681](https://arxiv.org/abs/2610.07681)

    该论文发现，人类运动先验最有效的利用方式是将其用于结构化探索以引导协调动作的发现，而不是用来限制或扩充动作空间，从而在灵巧操作中兼顾协调性与表达能力。

    

    灵巧操作构成了一个具有挑战性的高维优化问题，因为有用的行为需要众多手部关节的协调运动。在强化学习（RL）和基于采样的轨迹优化中，探索通常依赖于对机器人各关节的独立扰动，这使得协调性行为难以被发现。先前的工作利用从人类手部数据中学习到的低维协调关节运动空间来缩小抓取学习的搜索空间，但这限制了通用操作所需的动作表达能力。一些方法将学习到的动作与关节空间动作相结合以恢复表达能力，但这会增加维度并引入冗余。我们在多样化的操作设置中研究了这些影响，改变了动作维度、探索策略以及人类数据的来源。我们的实验表明，人类运动先验最有效的用法是用于构建探索的结构，而非……（原文摘要在此处截断）。

    arXiv:2610.07681v1 Announce Type: cross  Abstract: Dexterous manipulation poses a challenging high-dimensional optimization problem, as useful behaviors require coordinated motion across many hand joints. In reinforcement learning (RL) and sampling-based trajectory optimization, exploration commonly relies on independent robot joint perturbations, making coordinated behaviors difficult to discover. Prior work reduces this search space for grasp learning using low-dimensional spaces of coordinated joint motions learned from human hand data, but this restricts the expressivity required for general manipulation. Some combine learned and joint-space actions to restore expressivity, but this increases dimensionality and introduces redundancy. We study these effects across diverse manipulation settings, varying action dimensionality, exploration strategy, and the source of human data. Our experiments suggest that human-motion priors are most effective when used to structure exploration rathe
    
[^178]: Transformer中的精确解体积与长度泛化

    Exact-Solution Volume and Length Generalization in Transformers

    [https://arxiv.org/abs/2610.07676](https://arxiv.org/abs/2610.07676)

    该论文提出归一化精确解体积（NESV）这一新指标来量化Transformer的长度泛化难度，证明精确解体积随输入长度衰减得越快，长度泛化就越困难，并为FIRST、MAJORITY、INDEX、PARITY四个任务建立了渐近界。

    

    关于Transformer表达能力的研究能够表明一个Transformer是否有能力解决给定任务，但对于所学习到的解能否泛化到更长的输入长度却几乎没有提供任何指示。我们通过归一化精确解体积（NESV）来研究这一问题：即在一个有界参数区域中，能够对每个长度为n的输入都实现精确解的参数所占的比例。对于具有log n缩放注意力的固定宽度单层Transformer，我们为四个任务建立了NESV的渐近界：FIRST（Θ(1)）、MAJORITY（Θ(1/(n log n))）、INDEX（Θ(1/n³)）和PARITY（0）。这些结果与先前的实证结果一致：精确解体积随输入长度衰减得越快，该任务的长度泛化就越困难。通过深入分析INDEX任务，我们的体积分析揭示了两种随n增长的误差来源。因此，我们研究了一种在结构上（摘要在此处截断）

    arXiv:2610.07676v1 Announce Type: cross  Abstract: Research on transformer expressivity shows whether a transformer is capable of solving a given task, but gives little indication of whether the solution, if learned, is generalizable to longer input lengths. We study this question through normalized exact-solution volume (NESV): the fraction of a bounded parameter region that achieves an exact solution on every input of length $n$. For fixed-width, single-layer transformers with $\log n$-scaled attention, we establish asymptotic bounds on NESV for four tasks: FIRST ($\Theta(1)$), MAJORITY ($\Theta(1/(n\log n))$), INDEX ($\Theta(1/n^3)$), and PARITY ($0$). These results are consistent with previous empirical results: the faster the exact-solution volume decays with input length, the harder it is to length-generalize on that task. Looking deeper into INDEX, our volume analysis reveals two error sources that grow with $n$. Consequently, we study a transformer model that would structurally
    
[^179]: EIO-Agents：AI智能体评估中缺失的语义层

    EIO-Agents: The Missing Semantic Layer for AI Agent Evaluation

    [https://arxiv.org/abs/2610.07675](https://arxiv.org/abs/2610.07675)

    论文提出了EIO-Agents开放规范，通过评估智能本体（EIO）语义层与可移植评估记录（PER）记录系统的双层架构，为AI智能体评估建立了可互操作的语义标准，使评估证据、论断与PASS/REVIEW/BLOCK决策之间形成可计算、可追溯的关联。

    

    AI智能体正被部署到日益关键的生产环境中，但对于其评估结果的实际含义，目前缺乏共享的语义标准。分数、轨迹、评判器输出以及多评审团的结论越来越多地被用来证明就绪与发布决策的合理性，然而这些结果往往没有说明是什么证据支撑了一个论断、该证据能够确立什么，以及该论断如何导出最终决策。我们提出了EIO-Agents，一个面向可互操作AI智能体评估的开放规范，该规范构建于两个层次之上。评估智能本体提供了语义层，通过类型化证据、版本化行为谓词、证据契约、论断、见证规则、证明状态、复现机制，以及针对指标、发现、控制项和PASS（通过）、REVIEW（审查）或BLOCK（阻断）决策的可计算推导来实现语义表达。可移植评估记录（PER）则提供了记录系统：一种对单次评估的规范化、内容寻址表示……（摘要原文在此处截断）

    arXiv:2610.07675v1 Announce Type: new  Abstract: AI agents are entering production in increasingly consequential environments without a shared semantic standard for what their evaluations actually mean. Scores, traces, judge outputs, and multi juror findings are increasingly used to justify readiness and release decisions, yet they often do not specify what evidence supports a claim, what that evidence can establish, or how the claim leads to a decision. We introduce EIO-Agents, an open specification for interoperable AI agent evaluation built on two layers. The Evaluation Intelligence Ontology (EIO) provides the semantic layer through typed evidence, versioned behavioral predicates, evidence contracts, claims, witness rules, proof status, recurrence, and computable derivations for metrics, findings, controls, and PASS, REVIEW, or BLOCK decisions. The Portable Evaluation Record (PER) provides the system of record: a canonical, content addressed representation of one evaluation that pre
    
[^180]: 评估葡萄栽培田间研究中的人机协作工作流

    Evaluating human-AI workflows for field research in viticulture

    [https://arxiv.org/abs/2610.07669](https://arxiv.org/abs/2610.07669)

    多智能体AI系统Aleks v1与人类迭代协作开发的红叶症状预测模型，通过模型引导的行优先级排序，将加州葡萄园田间调查中发现新记录红叶观测的比例从85.8%提升至94.1%，主要优化了区块间的调查资源分配。

    

    我们评估了加州葡萄园精准病害防控项目中两种实时人机交互的价值。该项目检验了2021-2024年的商业田间调查记录以及覆盖140公顷的遥感测量数据，能否支持2025年红叶症状预测，从而为优先安排田间调查和病毒检测提供依据。在工作流1中，多智能体研究系统Aleks v1在人类迭代优化的辅助下开发了预测模型。我们将Aleks的2024年单株尺度模型应用于更新的2025年预测变量，并依据独立的2025年田间调查数据对红叶预测结果进行了评估。在覆盖全部葡萄藤位置45%的回顾性模拟中，在自适应调查策略基础上叠加模型引导的行优先级排序，使新记录红叶观测的发现比例从85.8%提升至94.1%。区块内的调查对比表明，该模型主要改进了区块之间的调查资源分配。尽管模型内部2024年单……（原文摘要在此处截断）

    arXiv:2610.07669v1 Announce Type: cross  Abstract: We assessed the value of two live human-AI interactions in a precision disease control project in California vineyards. The project tested whether 2021-2024 commercial scouting records and remote-sensing measurements across 140 hectares could support 2025 red-leaf symptom forecasting for prioritized scouting and virus testing. In Workflow 1, Aleks v1, a multi-agent research system, developed forecasting models with iterative human refinement. We applied Aleks's 2024 vine-scale model to updated 2025 predictors and evaluated red-leaf forecasts against independent 2025 scouting. In retrospective simulations surveying 45% of all vine positions, adding model-informed row prioritization to adaptive scouting increased the encountered proportion of newly recorded red-leaf observations from 85.8% to 94.1%. Within-block scouting comparisons suggested the model mainly improved scouting allocation among blocks. Despite unreliable internal 2024 per
    
[^181]: CACHEFORGE：基于大语言模型引导的端到端生成式缓存替换策略，实现高性能与硬件效率

    CACHEFORGE: LLM-Guided End-to-End Generative Cache Replacement Policy for Performance and Hardware Efficiency

    [https://arxiv.org/abs/2610.07668](https://arxiv.org/abs/2610.07668)

    CACHEFORGE 首次将大语言模型嵌入受控的硬件感知循环中，端到端地自动演化生成缓存替换策略，突破了传统启发式和模仿学习方法性能停滞与过拟合的瓶颈。

    

    现代缓存替换设计已趋于饱和，因为它们受限于固定的表示结构、手工设计且基于启发式特征工程的预测器，或无法自行生成新决策逻辑的离线模仿模型。与此同时，缓存替换决策受到预取、抖动、空间局部性和访问类型行为之间因果交互的影响，形成了一个难以手动遍历的庞大设计空间。先前的方法通常依赖启发式规则、参数调优或对离线最优策略的模仿，只能捕捉相关性而无法合成新机制。因此，它们的性能提升往往陷入停滞，并在动态工作负载条件下出现过拟合。CACHEFORGE 是首个端到端演化缓存替换策略的框架，它将大语言模型嵌入到一个受控的硬件感知循环中。在每次迭代中，大语言模型提出新的 C++ 替换策略……

    arXiv:2610.07668v1 Announce Type: cross  Abstract: Modern cache replacement designs saturate because they operate within fixed representational structures, hand-crafted and heuristic based feature-engineered predictors, or offline imitation models that cannot generate new decision logic on their own. At the same time, replacement is shaped by the causal interaction of prefetching, thrashing, spatial locality, and access-type behavior, producing an enormous design space that is difficult to traverse manually. Prior approaches typically rely on heuristics, parameter tuning, or imitation of an offline optimal policy, capturing correlations rather than synthesizing new mechanisms. As a result, their performance gains often plateau and they overfit under dynamic workload conditions.   CACHEFORGE is the first framework to evolve cache-replacement policies end-to-end by embedding a large language model inside a governed hardware-aware loop. In each iteration, the LLM proposes new C++ replacem
    
[^182]: SENSE：状态感知情感导航故事引擎

    SENSE: State-aware Emotion Navigation Storytelling Engine

    [https://arxiv.org/abs/2610.07666](https://arxiv.org/abs/2610.07666)

    SENSE是一个状态感知框架，通过集成MIND状态化叙事架构、结构分析器和路径感知上下文管理模块，能够从极少的高层输入生成结构连贯、情感丰富且支持多轨道情感导航的可玩分支视觉小说。

    

    本文提出了SENSE，一个用于生成具有多轨道情感导航的可玩分支视觉小说的状态感知框架。SENSE集成了名为MIND的状态化叙事架构、一个结构分析器和一个路径感知上下文管理模块，能够生成结构连贯且情感丰富的叙事内容。仅需极少量的高层输入，它便可在保持角色一致性和叙事因果性的同时生成多条相互交叉的故事路线。基于LLM评审、情感指标和视觉评估的评测表明，SENSE在叙事多样性和稳健的资产集成方面优于基线方法，同时初步的人类实验显示其在情感保真度上有方向性的提升，并保持了相当的娱乐性。

    arXiv:2610.07666v1 Announce Type: cross  Abstract: This paper presents SENSE, a state-aware framework for generating playable branching visual novels with multi-track emotional navigation. Integrating a state-based narrative architecture called MIND, a structure analyzer, and a path-aware context management module, SENSE produces narratives that are both structurally coherent and emotionally rich. From minimal high-level inputs, it generates multiple intersecting routes while preserving character consistency and narrative causality. Evaluations using LLM judges, affective metrics, and visual assessments indicate SENSE outperforms baselines in narrative diversity and robust asset integration, while preliminary human trials show directional improvements in emotional fidelity alongside comparable enjoyment.
    
[^183]: 面向用户行为模拟的工作流与提示词联合优化

    Joint Workflow and Prompt Optimization for User Behavior Simulation

    [https://arxiv.org/abs/2610.07663](https://arxiv.org/abs/2610.07663)

    SWORD框架基于角色化设计，仅依靠一个标量任务指标即可联合优化多智能体工作流拓扑与自然语言提示词，在用户行为模拟任务上显著超越仅优化提示词、仅优化工作流和分阶段优化的基线方法。

    

    用户行为模拟是指通过使用模拟智能体代替真实用户，对用户在信息系统中的交互进行计算建模。它可支持系统测试与评估、决策与预测，以及用户体验设计。现有的模拟器依赖于手工设计的规则或领域专业知识，而这些方法难以跨任务迁移。本文提出了SWORD（基于角色设计的模拟驱动工作流与提示词优化，Simulation-driven Workflow and Prompt Optimization with Role-based Design），这是一个联合优化多智能体工作流拓扑结构与自然语言提示词的框架。它仅由一个标量任务指标引导，无需领域初始化或任务特定的工程设计。实验结果表明，在受控的相同骨干模型比较条件下，SWORD 相对于仅优化提示词、仅优化工作流以及分阶段优化的基线方法均取得了统计学上显著的性能提升。与已发表的最强领域专用基线相比，SWORD 进一步……（原文摘要在此处截断）

    arXiv:2610.07663v1 Announce Type: cross  Abstract: User behavior simulation is the computational modeling of user interactions within information systems through the use of simulated agents in place of live users. It supports system testing and evaluation, decision-making and forecasting, and user experience design. Existing simulators rely on hand-crafted rules or domain expertise that transfers poorly across tasks. SWORD (Simulation-driven Workflow and Prompt Optimization with Role-based Design) is introduced as a framework that jointly optimizes multi-agent workflow topology and natural-language prompts. It is guided solely by a scalar task metric, without domain initialization or task-specific engineering. The experimental results demonstrate that SWORD achieves statistically significant gains over prompt-only, workflow-only, and staged-optimization baselines under a controlled, identical-backbone comparison. Against the strongest published domain-specific baseline, SWORD further i
    
[^184]: 大语言模型中的大规模激活门控通道

    Massive Activation Gating Channel in Large Language Models

    [https://arxiv.org/abs/2610.07661](https://arxiv.org/abs/2610.07661)

    该论文发现大语言模型中大规模激活现象由尖峰前馈网络输入嵌入中一个位置固定的“大规模激活门控通道（MAGC）”所控制，并通过跨四个模型家族六个模型的实验验证和理论分析揭示了其产生机制。

    

    大规模激活是大语言模型（LLM）中普遍存在的一种现象，即少数隐藏通道会表现出异常大的幅值。然而，关于token在预训练大语言模型中传播时如何产生大规模激活的机制，目前仍知之甚少。在本文中，我们发现大规模激活的出现是由尖峰前馈网络（FFN）输入嵌入中的单个通道控制的。对于特定的大语言模型，该通道的位置是固定的。我们将该通道命名为大规模激活门控通道（MAGC）。当MAGC的值足够大（或足够小，取决于具体的大语言模型）时，尖峰FFN的输出就会表现出大规模激活。通过考察四个模型家族中不同模型规模的六个大语言模型，我们验证了MAGC的存在及其作用。此外，我们还为MAGC诱导大规模激活的机制提供了理论解释。

    arXiv:2610.07661v1 Announce Type: new  Abstract: Massive activations, a phenomenon in which a small number of hidden channels exhibit exceptionally large magnitudes, are pervasive in large language models (LLMs). However, the mechanism by which a token develops massive activations as it propagates through a pretrained LLM remains poorly understood. In this paper, we find that the emergence of massive activations is controlled by a single channel in the input embedding to a spike feed-forward network (FFN). The position of this channel is fixed for a particular LLM. We name this channel the massive activation gating channel (MAGC). When the value of the MAGC is sufficiently large (or small, depending on the LLM), the output of the spike FFN exhibits massive activations. Examining six LLMs across four model families and different model sizes, we verify the existence and effect of MAGC. We further provide a theoretical explanation of the mechanism by which MAGC induces massive activations
    
[^185]: 规则止于何处，裁判始于何处：度量多智能体系统安全中的判断边界

    Where Rules End and Judges Begin: Measuring the Judgment Boundary in Multi-Agent Systems Security

    [https://arxiv.org/abs/2610.07657](https://arxiv.org/abs/2610.07657)

    该研究提出DEFER1防御框架，通过28项确定性检查级联与四位裁判评审团协同工作，将多智能体系统的攻击成功率从约30%降至约3%，并实证划定了规则可处置与需裁判判断的安全边界。

    

    基于大语言模型的多智能体系统（MAS）会调用工具、共享内存并委派任务，因而经常遭遇对抗性内容。当前针对MAS的防御通常被孤立评估，一次只关注一种攻击类型，这可能导致代价高昂且难以审计的后果。本研究将防御归纳为五项原则，并将其实现为DEFER1（确定性优先执行与残余判断），其中包含一个由28项检查组成的级联，拦截其可拦截的内容，并将其余内容提交给由四位裁判组成的评审团。在跨四个领域的独立测试中，攻击成功率从约30.0%降至约3.0%，其中78%被拦截的攻击由确定性检查处理。在安全运营领域，仅四分之一的提议到达裁判环节，这表明规则为违反明确策略的攻击提供了安全保障，而裁判则负责处理那些仅仅歪曲意图的攻击。两个系统都存在弱点，例如……（摘要在此截断）

    arXiv:2610.07657v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) engage tools, share memory, and delegate tasks, often encountering adversarial content. Current defenses for MAS are typically evaluated in isolation, focusing on one attack type at a time, which can lead to costly and hard-to-audit outcomes. This study organizes defenses into five principles, implementing them as DEFER1 (DEterministic-First Enforcement with Residual judgment), which includes a cascade of 28 checks that blocks what it can and refers the rest to a panel of four judges. In independent testing across four domains, attack success rates drop from about 30.0% to approximately 3.0%, with 78% of blocked attacks handled by deterministic checks. Only a quarter of proposals reach the judges in the security-operations domain, illustrating that the rules provide security for attacks violating clear policies, while judges manage those that only misrepresent intent. Both systems have weaknesses, such
    
[^186]: 面向安全性的在线策略蒸馏是否存在后门风险？

    Does On-Policy Distillation for Safety Pose Backdoor Risks?

    [https://arxiv.org/abs/2610.07654](https://arxiv.org/abs/2610.07654)

    研究揭示了面向安全性的在线策略蒸馏（OPD）中一个被忽视的后门威胁：被植入后门的教师模型可将隐藏恶意行为传播给原本干净的学生模型，仅3%的投毒率即可使攻击成功率达70%，而增加训练轮数和常用的top-k KL方法会进一步加剧该风险。

    

    在线策略蒸馏作为一种将能力从教师模型迁移到学生模型的有效方式，正受到越来越多的关注。近期研究进一步探索将OPD作为提升大语言模型安全性的工具，并取得了可喜的成果。然而，这些方法通常假设教师模型和训练数据是可信的。本文揭示了面向安全性的OPD中一个被忽视的威胁：一个对齐了安全性但被植入后门的教师模型，可以将其隐藏的恶意行为传播给原本干净的学生模型。在我们的威胁模型下，低至3%的投毒率即可使蒸馏后学生模型的攻击成功率（ASR）高达70%。我们进一步发现两种训练选择会放大这一风险。首先，增加训练轮数即使在低投毒率下也会导致较高的ASR：仅需10个投毒样本，经过16轮训练后ASR即可达到67%。其次，常用的top-k KL方法可能加速后门……

    arXiv:2610.07654v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has attracted growing attention as an effective way to transfer capabilities from teacher models to student models. Recent studies further explore OPD as a tool for improving large language model safety with promising results. However, these approaches typically assume that the teacher and training data are trustworthy. In this paper, we uncover an overlooked threat to OPD for safety: a safety-aligned but backdoored teacher can propagate its hidden malicious behavior to an initially clean student. Under our threat model, a poisoning rate as low as 3% results in an attack success rate (ASR) of up to 70% on the distilled student. We further identify two training choices that can amplify this risk. First, increasing the number of training epochs can lead to high ASR even at low poisoning rates. With only 10 poisoned samples, ASR reaches 67% after 16 epochs. Second, the commonly used top-k KL can accelerate bac
    
[^187]: SMART：通过大规模合成预训练实现零样本Sim-to-Real铰接物体操作

    SMART: Zero-Shot Sim-to-Real Articulated Object Manipulation via Large-Scale Synthetic Pretraining

    [https://arxiv.org/abs/2610.07652](https://arxiv.org/abs/2610.07652)

    提出了SMART系统，其核心SMART-Sim仿真平台通过铰接感知设计实现大规模合成操作演示的生成与收集，实现了零样本的从仿真到真实的铰接物体操作。

    

    arXiv:2610.07652v1 公告类型：cross 摘要：与铰接物体进行交互的能力对具身智能系统至关重要，但由于涉及精确的接触和遵循约束的运动，为这类交互收集大规模真实世界演示仍然十分困难。尽管仿真提供了一个有前景的替代方案，但现有的合成数据工作仅覆盖有限的铰接物体类别，而通用的合成流水线缺乏针对部件级语义和铰接约束的显式设计，这阻碍了智能体任务的生成以及高质量铰接操作演示的可扩展合成。为了弥合这一差距，我们提出了SMART，这是一个可扩展的系统，利用大规模合成的操作演示来实现铰接物体操作。其核心是SMART-Sim，一个具有铰接感知设计的仿真平台，能够实现有效的任务生成和高效的演示收集。

    arXiv:2610.07652v1 Announce Type: cross  Abstract: The ability to interact with articulated objects is essential for embodied intelligent systems, but collecting large-scale real-world demonstrations for these interactions remains challenging due to the precise contact and constraint-following motions involved. Although simulation provides a promising alternative, existing synthetic data efforts cover limited articulated-object categories, while general-purpose synthesis pipelines lack explicit designs for part-level semantics and articulation constraints, hindering agentic task generation and scalable synthesis of high-quality articulated-manipulation demonstrations. To bridge this gap, we introduce SMART, a scalable system leveraging large-scale Synthesized Manipulation demonstrations for ARTiculated-object manipulation. At its core, we develop SMART-Sim, a simulation platform with articulation-aware design that enables effective task generation and efficient demonstration collection
    
[^188]: 匹配物体还是关系？追踪视觉语言模型内部的抽象推理

    Matching Object or Relation? Tracing Abstract Reasoning Inside VLMs

    [https://arxiv.org/abs/2610.07646](https://arxiv.org/abs/2610.07646)

    该研究借鉴心理学的关系匹配样本范式并结合模型内部机制分析，识别出促使视觉语言模型从物体匹配转向关系匹配的四个关键因素，并揭示了其与人类“关系转变”相似的发展轨迹。

    

    视觉语言模型（VLM）在视觉基准测试中表现出色，但在需要抽象推理的任务上却系统性地失败。现有基准测试只记录了这种失败，却无法解释其发生的原因，也无法指出缺失的是哪种认知能力。我们通过采用源自比较心理学与发展心理学的“关系匹配样本”（RMTS）范式，并将其与模型内部机制的剖析相结合，来弥合这一空白。在一个参数化控制的刺激集合上，我们对前沿 API 模型（GPT、Claude、Gemini）以及三个开源模型家族（Qwen3.5、Gemma-4、InternVL3）进行了评估，识别出促使 VLM 转向关系匹配的四个杠杆——能力层级、模型规模、每个场景中的物体数量，以及去除逐物体的刺激噪声——这些因素共同产生了一条类似人类发展的轨迹，与人类的“关系转变”（relational shift）相呼应。打开模型内部后，逐层表征相似性分析以及（摘要原文在此处被截断）

    arXiv:2610.07646v1 Announce Type: new  Abstract: Vision Language Models (VLMs) excel on visual benchmarks but fail systematically on tasks requiring abstract reasoning. Existing benchmarks document this failure but cannot say \emph{why} it happens or which cognitive capability is missing. We close this gap by adopting the Relational Match-to-Sample (RMTS) paradigm from comparative and developmental psychology and pairing it with a mechanistic analysis of the model's internals. On a parametrically controlled stimulus set evaluated across frontier API models (GPT, Claude, Gemini) and three open-source families (Qwen3.5, Gemma-4, InternVL3), we identify four levers that shift VLMs toward the relational match---capability tier, model scale, the number of objects per scene, and the absence of per-object stimulus noise---together producing a developmental-like trajectory that mirrors the human \emph{relational shift}. Opening up the model, a per-layer representational similarity analysis and
    
[^189]: SkillPoison：基于成功经验的渐进式技能投毒

    SkillPoison: Progressive Skill Poisoning via Successful Experiences

    [https://arxiv.org/abs/2610.07645](https://arxiv.org/abs/2610.07645)

    SkillPoison提出了一种新型技能投毒框架，它通过构建强化目标行为的成功经验并移除该行为适用条件的上下文约束，在不注入任何恶意内容、不使任何单条轨迹显式恶意的情况下，实现从经过验证的成功经验中对智能体技能库的渐进式投毒。

    

    摘要：自我改进的大语言模型（LLM）智能体越来越多地将成功经验蒸馏为持久的、可复用的技能。现有的技能攻击方法通过在单个经验或提取出的技能中注入恶意触发器、恶意行为或虚假事实来破坏这一学习流程。然而，此类攻击容易被检测，且注入的恶意行为往往难以积累为持久技能。在本文中，我们表明：即使不使任何单条轨迹具有恶意性，技能投毒也可以从经过验证的成功经验中产生。基于这一洞察，我们提出了SkillPoison，一个通过成功经验对技能进行渐进式投毒的新框架。SkillPoison首先构建一组强化目标行为的成功经验，然后移除约束该行为适用时机的上下文条件。SkillPoison并非注入恶意内容，而是塑造技能提取器的泛化方式……

    arXiv:2610.07645v1 Announce Type: cross  Abstract: Self-improving LLM agents increasingly distill successful experiences into persistent, reusable skills. Existing skill attack methods corrupt this learning pipeline by injecting malicious triggers, behaviors, or false facts into individual experiences or extracted skills. However, such attacks are easily detected, and the injected malicious behaviors often fail to accumulate as persistent skills. In this paper, we show that skill poisoning can arise even from verified successful experiences, without making any individual trajectory malicious. Based on this insight, we propose SkillPoison, a novel framework that progressively poisons skill via successful experiences. SkillPoison first constructs a set of successful experiences that reinforce a target behavior, and then removes the contextual conditions that constrain when the behavior applies. Rather than injecting malicious content, SkillPoison shapes how the skill extractor generalize
    
[^190]: 用于KV缓存驱逐的蒙特卡洛估计

    Monte Carlo Estimation for KV Cache Eviction

    [https://arxiv.org/abs/2610.07643](https://arxiv.org/abs/2610.07643)

    提出免训练方法LORE-KV，通过蒙特卡洛采样冻结模型的短自回归延续并以响应侧查询状态估计提示token效用，将KV缓存驱逐从“回顾过去”转变为“预测未来”，在回答时保留真正重要的记忆。

    

    大多数KV缓存驱逐方法实际上在问：哪些内存在阅读提示时显得重要？我们转而问：哪些内存在回答时会有用？由于在驱逐时无法获得解码查询，先前的面向未来的方法依赖于伪响应或合成的未来查询估计。我们将固定预算的面向未来的驱逐问题转化为对合理的模型条件查询轨迹的分布估计，并引入LORE-KV（基于可靠性加权集成的前瞻输出扰动KV缓存方法），这是一种免训练方法，它从冻结的目标模型中采样简短的自回归延续，并利用其响应侧的查询状态来估计提示token的效用。token通过投影的留一法注意力输出删除代价进行评分，并在采样的未来之间进行聚合（可选地带有轨迹加权）。临时延续在最终解码之前被丢弃，无需

    arXiv:2610.07643v1 Announce Type: cross  Abstract: Most KV-cache eviction methods ask, in effect, which memory appeared important while reading the prompt? We instead ask, which memory will matter while answering? Since decoding queries are unavailable at eviction time, prior future-aware methods rely on pseudo-responses or synthetic future-query estimates. We cast fixed-budget future-aware eviction as distributional estimation over plausible model-conditional query trajectories and introduce LORE-KV (Lookahead Output-perturbation with Reliability-weighted Ensembles for Key-Value caches), a training-free method that samples short autoregressive continuations from the frozen target model and uses their response-side query states to estimate prompt-token utility. Tokens are scored by projected leave-one-out attention-output deletion cost and aggregated across sampled futures with optional trajectory weighting. The temporary continuations are discarded before final decoding, requiring no 
    
[^191]: 迈向可解释国际象棋战术的自动合成

    Towards the Automatic Synthesis of Interpretable Chess Tactics

    [https://arxiv.org/abs/2610.07640](https://arxiv.org/abs/2610.07640)

    本文提出一种受国际象棋战术启发、由归纳逻辑编程系统PAL所学模式推导而来的符号化子策略模型，通过融入领域知识提升可解释性，并提出散度度量评估方法，其合成的战术组合能给出与人类初学者棋力相当的走法建议。

    

    最先进的强化学习智能体能够在国际象棋、围棋和《星际争霸II》等游戏中超越人类专家。这些智能体并非仅仅利用其数字硬件在反应和计算速度上快于人类，而是采用了更优的策略从而赢得更多胜利。解读这些策略将为人类玩家提供提高棋艺的宝贵见解。在这项初步工作中，我们提出了一种用于下国际象棋的符号化子策略模型。受国际象棋战术的启发，我们的模型尝试融入领域知识以提升可解释性。我们改造了由名为PAL的归纳逻辑编程系统所学习到的模式来推导该模型。我们提出了一种散度度量方法，用于将模型与随机基线进行对比评估，并发现了一组战术，能够给出与人类初学者棋力相近的走法建议。最后，我们提出了一种计算评估方案。

    arXiv:2610.07640v1 Announce Type: new  Abstract: State-of-the-art reinforcement learning agents are capable of outperforming human experts at games like chess, Go and StarCraft II. These agents do not simply take advantage of their digital hardware in being able to react and calculate faster than humans, but employ better strategies that lead to more victories. Interpreting these strategies would give human players valuable insight into how to improve their play. In this preliminary work, we propose a symbolic sub-policy model for playing chess. Inspired by chess tactics, our model attempts to incorporate domain knowledge to improve interpretability. We adapt patterns learned by an inductive logic programming system called PAL to derive our model. We contribute a divergence metric to evaluate our model against a random baseline, and find a set of tactics that is able to suggest moves of similar playing strength to a human beginner. Finally, we propose a computational evaluation scheme 
    
[^192]: 学习复杂游戏策略的可解释表示

    Learning Explainable Representations of Complex Game-playing Strategies

    [https://arxiv.org/abs/2610.07638](https://arxiv.org/abs/2610.07638)

    本文提出一种类似人类认知的方法，训练强化学习智能体将学到的游戏策略合成为基于动作序列的可执行程序，从而获得可解释的策略表示，并在国际象棋和网格环境任务中验证了其有效性。

    

    作为学习玩复杂游戏的一部分，人类玩家会形成与游戏规则相一致的游戏概念和策略抽象，以提高自己的表现。这些概念被用来解释其他玩家的行为，并指导自己在游戏中的行动。理解其他玩家的策略是这种提升的关键部分，但这需要时间和精力。在本文中，我们提出了一种类似人类认知的策略，用于训练强化学习智能体将学习到的策略和方针合成为基于游戏动作序列的可执行程序。我们提出了自动学习此类程序的方法，用于下国际象棋以及在基于网格的环境中解决任务。我们证明，学习到的策略能够产生有效的动作，并且可以从游戏对局数据中学习得到。

    arXiv:2610.07638v1 Announce Type: new  Abstract: As part of learning to play complex games, human players develop develop abstractions for concepts and strategies of gameplay consistent with game rules to improve their performance. These concepts are applied to explain other players' actions, and to inform their own actions in-game. Understanding other players' strategies is a crucial part of such improvement, but requires time and effort. In this paper, we propose a strategy similar to human cognition for training RL agents to synthesize learned strategies and policies as executable procedures based on sequences of gameplay actions. We present methods to automatically learn such programs to play chess and to solve tasks in a grid-based environment. We show that the learned strategies produce effective actions, and can be learned from gameplay data.
    
[^193]: 测量Twitter和Reddit档案中的气候反弹：词汇定义、记录的回应与参与者更替

    Measuring climate backlash in Twitter and Reddit archives: Lexical definitions, recorded responses and participant turnover

    [https://arxiv.org/abs/2610.07634](https://arxiv.org/abs/2610.07634)

    该研究以无抽样全量处理和透明、非排他性的词汇规则测量社交媒体上的气候反弹，揭示了词汇定义的宽窄会显著改变甚至逆转跨平台对比结果，并通过将回归作者的事件期变化与参与者更替分离来澄清所测量的社会过程。

    

    社交媒体档案常被用于研究对气候行动的抵制，但词语、回应计数与观察到的参与者并不衡量同一社会过程。我们通过处理全部已登记文件（不进行抽样）并应用透明的、非排他性的词汇规则，考察了四个所提供的Twitter和Reddit档案。本研究将框架共现与特定来源的时间和回应模型联系起来，进而将回归作者在事件期间的变化与参与者更替区分开来。在Reddit上，可再生能源术语伴随成本相关的语言出现，然而更狭窄的反弹短语显著缩小了跨来源差异，并在提交帖中逆转了《巴黎协定》相关对比的符号方向。跨话语历史并不能改善符合条件的主要语境预测。否认/阴谋论术语与更高的Twitter记录点赞数相关，而Reddit上的回应关联则取决于框架、结果和作者设定。（摘要原文在此处截断）

    arXiv:2610.07634v1 Announce Type: new  Abstract: Social media archives are often used to study resistance to climate action, but words, response counters and observed participants do not measure the same social process. We examine four supplied Twitter and Reddit archives by processing all registered files without sampling and applying transparent, non-exclusive lexical rules. The study links frame co-occurrence to source-specific temporal and response models, then separates event-period changes among returning authors from participant turnover. Renewable-energy terms accompany cost-related language on Reddit, yet narrower backlash phrases sharply reduce cross-source contrasts and reverse the sign of the Paris Agreement contrast in submissions. Cross-discourse history does not improve eligible primary-context forecasts. Denial/hoax terms are associated with higher recorded Twitter likes, whereas Reddit response associations depend on frame, outcome and author specification. Around the 
    
[^194]: 学会超越理论：超越初始假设空间的实验发现

    Learning to Outgrow a Theory: Experimental Discovery Beyond the Initial Hypothesis Space

    [https://arxiv.org/abs/2610.07627](https://arxiv.org/abs/2610.07627)

    该论文提出实验性模型类别修正方法，让发现策略联合提出结构性修改和诊断实验，借助类别级可区分性目标与任意时间有效的序贯证据，在当前假设类别被拒绝后才触发修正，从而在400个受控动力学环境中以32个实验预算实现89.5%的精确恢复率，超越最强基线10.0个百分点。

    

    科学发现系统通常在固定的假设空间内优化实验。当所有可用候选模型都遗漏了同一个缺失机制时，就会产生一种失效模式：即使模型类别在系统层面是错误的，候选模型之间的分歧也可能消失。我们提出了实验性模型类别修正问题，其中一个发现策略联合提出结构性修改以及一个检验该修改是否必要的诊断实验。该方法将类别级可区分性目标（即一个共享参数化必须能够解释所有被选中的实验）与任意时间有效的序贯证据相结合，只有在当前模型类别被拒绝之后才触发结构性修正。在400个留出的受控动力学环境中，该联合策略在32个真实实验的预算下达到了89.5%的精确恢复率，比最强的匹配基线提高了10.0个百分点，同时所需的实际执行实验数量更少。

    arXiv:2610.07627v1 Announce Type: new  Abstract: Scientific discovery systems typically optimize experiments within a fixed hypothesis space. This creates a failure mode when all available candidates omit the same missing mechanism: candidate disagreement can collapse even while the model class is systematically wrong. We formulate experimental model-class revision, in which a discovery policy jointly proposes a structural edit and a diagnostic experiment that tests whether that edit is necessary. The method couples a class-level distinguishability objective, in which one shared parameterization must explain all selected experiments, with anytime-valid sequential evidence that triggers structural revision only after the current class is rejected. On 400 held-out controlled dynamical environments, the joint policy reaches 89.5% exact recovery with a budget of 32 real experiments, improving the strongest matched baseline by 10.0 percentage points while requiring fewer executed experiment
    
[^195]: 无状态语言智能体：扩展长时程自动化研究

    Stateless Language Agents: Scaling Long-Horizon Automated Research

    [https://arxiv.org/abs/2610.07625](https://arxiv.org/abs/2610.07625)

    提出无状态语言智能体（SLA）框架，通过“有状态搜索、无状态智能体”的原则——由框架统一管理研究状态并为每次调用重建角色化上下文——来解决长时程自动化研究中智能体重放冗长历史、重复劳动和过早停止实验等失败模式。

    

    自动化研究系统越来越多地在长时程上运行LLM智能体，但更多的推理本身并不能带来更多的研究进展：智能体会重放不断增长的历史记录、重复彼此的工作，或者在token持续消耗的同时停止实验。然而，大多数评估使用较短的预算或很快饱和的基准测试，使得这些失败模式未曾得到检验。我们将这些失败追溯到两个关键选择：研究状态存储在哪里，以及由谁决定下一步尝试什么。我们提出了无状态语言智能体，其建立在“有状态搜索、无状态智能体”的原则之上：没有任何智能体在多次调用之间携带其对话历史；相反，由框架拥有研究状态（候选解决方案和测量结果），并为每次调用重建一个全新的、特定角色的上下文。每个智能体所看到的内容由此成为一种显式的设计选择，而不是随运行而不断增长的历史。我们在SLA框架中实现了这一原则，

    arXiv:2610.07625v1 Announce Type: cross  Abstract: Automated research systems increasingly run LLM agents over long horizons, but more inference does not by itself produce more progress: agents replay growing histories, duplicate one another's work, or stop experimenting while token consumption continues. Yet most evaluations use short budgets or benchmarks that saturate early, leaving these failure modes untested. We trace these failures to two choices: where research state lives and who decides what to try next. We introduce Stateless Language Agents (SLAs), built on the principle of stateful search with stateless agents: no agent carries its conversation across invocations; instead, the harness owns the research state (candidate solutions and measured outcomes) and reconstructs a fresh and role-specific context for every invocation. What each agent sees becomes an explicit design choice rather than a history that grows with the run. We implement this principle in the SLA framework, 
    
[^196]: 先探索，后确定：基于语言模型的测量高效科学定律发现

    Explore, Then Commit: Measurement-Efficient Scientific Law Discovery with Language Models

    [https://arxiv.org/abs/2610.07620](https://arxiv.org/abs/2610.07620)

    该论文提出了一种“先探索后确定”协议，通过语言模型提出假设、程序化规划器高效收集测量，将科学定律发现所需的测量次数最多减少约5倍，并将误差显著降低一个数量级以上。

    

    科学定律发现需要选择测量并将证据转化为控制方程。我们评估了一种“先探索后确定”协议，其中大型语言模型提出假设，程序化规划器收集测量数据，然后通过全新的提示词从固定观测中合成最终定律。该协议结合了结构化探测、自动数值诊断、受限测量批次以及可选的解释器访问。在576次NewtonBench试验中，我们使用GPT-4.1-mini和中难度GPT-4.1复现版本，在12个物理模块上比较了八种配置。在中难度任务上，启用解释器的规划器在GPT-4.1-mini上每次试验使用8.6次测量（对比22.5次），在GPT-4.1上使用8.9次（对比43.0次）。它们基于量级的均方根对数误差分别从2.514降至0.202，从0.626降至0.149。额外的审计在覆盖率中保留了不完整和无效的提交。

    arXiv:2610.07620v1 Announce Type: new  Abstract: Scientific law discovery requires selecting measurements and converting evidence into a governing equation. We evaluate an explore-then-commit protocol in which a large language model proposes hypotheses, a programmatic planner gathers measurements, and a fresh prompt synthesizes the final law from fixed observations. The protocol combines structured probes, automatic numerical diagnostics, restricted measurement batches, and optional interpreter access. Across 576 NewtonBench trials, we compare eight configurations on 12 physics modules using GPT-4.1-mini and a medium-difficulty GPT-4.1 replication. On medium tasks, interpreter-enabled planners use 8.6 versus 22.5 measurements per trial for GPT-4.1-mini and 8.9 versus 43.0 for GPT-4.1. Their mean magnitude-based root-mean-squared logarithmic error falls from 2.514 to 0.202 and from 0.626 to 0.149, respectively. An additional audit retains incomplete and invalid submissions in a coverage
    
[^197]: BioStudyBench：评估智能体在知识截止日期之后的生物医学研究上的表现

    BioStudyBench: Evaluating Agents on Post-Cutoff Biomedical Studies

    [https://arxiv.org/abs/2610.07614](https://arxiv.org/abs/2610.07614)

    该论文提出BioStudyBench基准，基于模型知识截止日期之后发表的25个真实生物医学研究构建长时程任务，评估AI智能体自主查找公开数据、检索文献并通过数据分析复现已发表研究结果的能力。

    

    我们评估AI智能体能否利用公开数据复现已发表生物医学研究所报告的发现。现有的评估未能一致地将数据分析能力与先验知识或对已发表答案的检索区分开来。我们提出BioStudyBench，这是一个由25个长时程分析任务组成的基准，这些任务选自2026年7月至9月间首次发表的研究（即晚于我们所评估模型的开发商报告的知识截止日期），并从404,019条PubMed记录中半自动筛选而来。在每个任务中，智能体只会得到一个中性的研究问题，而不会获得数据文件，因此它必须自行查找并下载相关的公开数据，通过只返回其知识截止日期之前文献记录的工具检索文献，并通过数据分析报告研究发现。为了衡量相对于先验知识的增益，我们对每个任务分别在有数据与工具访问权限和无访问权限两种条件下运行。在八个模型上，访问数据和工具使通过率提高了47（原文摘要在此处截断）。

    arXiv:2610.07614v1 Announce Type: new  Abstract: We evaluate whether AI agents can match the reported findings of published biomedical studies using public data. Existing evaluations do not consistently separate analysis from prior knowledge or retrieval of the published answer. We introduce BioStudyBench, a benchmark of 25 long-horizon analysis tasks drawn from studies first published between July and September 2026, after the developer-reported knowledge cutoffs of the models we evaluate, semi-automatically filtered down from 404,019 PubMed records. In each task, the agent receives a neutral research question but no data files, so it must find and download the relevant public data, search the literature through tools that return only records dated before its cutoff, and report findings through data analysis. To measure gains over prior knowledge, we run every task both with and without access to data and tools. Across eight models, access to data and tools raises the pass rate by 47 
    
[^198]: 蛋白质语言模型中的线性适应度子空间实现样本高效的定向进化

    Linear Fitness Subspace in Protein Language Models Enables Sample-Efficient Directed Evolution

    [https://arxiv.org/abs/2610.07607](https://arxiv.org/abs/2610.07607)

    提出线性适应度子空间（LFS）假设并引入子空间引导进化搜索（SGES），通过在蛋白质语言模型突变引起的残基级表示变化中寻找与实验测定相关的紧凑方向集合，使适应度变化从少量标注样本中线性可获取，从而实现样本高效的模型引导定向进化。

    

    模型引导的定向进化旨在有限的真值评估预算下识别高适应度的蛋白质变体。蛋白质语言模型（PLM）为此类任务提供了丰富的表示，但任务无关的零样本评分可能与目标实验测定不对齐，而在高维嵌入空间中进行监督搜索会使代理建模和不确定性估计的样本效率低下。我们提出线性适应度子空间（LFS）假设：在突变引起的残基级表示变化中，存在一组紧凑的、与特定实验测定相关的方向集合，使得适应度变化可以从少量标注变体中被线性地获取。这是一个局部的、可通过监督恢复的论断，而非声称蛋白质适应度景观或PLM的全局几何结构是普遍线性的。基于这一观察，我们引入了子空间引导进化搜索（SGES），该方法从少量初始样本中估计LFS并进行代理建模……

    arXiv:2610.07607v1 Announce Type: cross  Abstract: Model-guided directed evolution seeks to identify high-fitness protein variants under limited oracle budgets. Protein language models (PLMs) provide rich representations for this task, but task-agnostic zero-shot scores can be misaligned with a target assay, while supervised search in high-dimensional embedding spaces can make surrogate modeling and uncertainty estimation sample-inefficient. We propose the Linear Fitness Subspace (LFS) hypothesis: within mutation-induced residue-level representation changes, a compact, assay-specific set of directions makes fitness variation linearly accessible from few labeled variants. This is a local, supervision-recoverable statement rather than a claim that protein fitness landscapes or global PLM geometry are universally linear. Building on this observation, we introduce Subspace-Guided Evolutionary Search (SGES), which estimates an LFS from a small initial sample and performs surrogate modeling,
    
[^199]: VALSE：面向大语言模型高效推理的垂直自适应层跳过方法

    VALSE: Vertical Adaptive Layer Skipping for Efficient Inference in Large Language Models

    [https://arxiv.org/abs/2610.07606](https://arxiv.org/abs/2610.07606)

    本文建立了垂直自适应层跳过的理论框架（包括期望计算成本闭式公式、跳层模型函数空间的严格包含定理以及与混合专家架构的结构对偶性），并据此提出VALSE方法，利用轻量级难度评估器实现逐样本的非连续层跳过，以提升大语言模型推理效率。

    

    本文建立了垂直自适应层跳过的理论框架，证明了三个基础性结果：(i) 期望FLOPs公式（定理2），给出了任意逐样本跳层调度计算成本的闭式表达式，该成本是各层跳过概率的函数；(ii) 函数空间超集（定理10）与严格包含（定理11）定理，表明跳层模型严格包含于——但有意义地逼近——完整层的函数空间，并给出了一个显式的分离示例；(iii) VALSE与混合专家架构之间的结构对偶性（命题6），将垂直深度方向的稀疏性定位为水平宽度方向稀疏性的正交对应物。基于这一理论，我们提出了VALSE（Vertical Adaptive Layer Skipping for Efficiency，垂直自适应层跳过效率方法），一种逐样本的、非连续的层跳过方法：一个轻量级的难度评估器对每个输入进行评分……

    arXiv:2610.07606v1 Announce Type: new  Abstract: This paper establishes a theoretical framework for vertical adaptive layer skipping, proving three foundational results: (i) an Expected FLOPs formula (theorem 2) giving a closed-form expression for the computational cost of arbitrary per-sample skip schedules as a function of layer-wise skip probabilities; (ii) function-space superset (theorem 10) and strict inclusion (theorem 11) theorems showing that skip-layer models are strictly contained in---yet meaningfully approximate---the full-layer function space, with an explicit separating example; and (iii) a structural duality between VALSE and Mixture-of-Experts architectures (proposition 6), positioning vertical depth-wise sparsity as the orthogonal counterpart to horizontal width-wise sparsity. Building on this theory, we propose VALSE (Vertical Adaptive Layer Skipping for Efficiency), a per-sample, non-contiguous layer skipping method: a lightweight difficulty estimator scores each in
    
[^200]: Emoception：基于选择性情感层微调的视频视觉Transformer用于从游戏画面中识别玩家情绪唤醒变化

    Emoception: Selective Affective Layer Fine-Tuning of Video Vision Transformers for Player Arousal Change Recognition From Gameplay Footage

    [https://arxiv.org/abs/2610.07603](https://arxiv.org/abs/2610.07603)

    提出选择性情感层微调（SALFT）方法，通过基于参数L2范数变化的层筛选准则，仅更新视频视觉Transformer约8%的参数即可在玩家情绪唤醒识别任务上达到与全量微调相当的性能。

    

    本文提出了选择性情感层微调，这是一种面向视频视觉Transformer的高效适配框架，用于从游戏过程中识别玩家的情绪唤醒。为了避开计算成本高昂的全量微调，SALFT引入了一种基于短暂适配后层参数L2范数变化的筛选准则，直接衡量表征的变化，相比基于梯度的替代方法提供了更稳定的基础。在Arousal Video Game AnnotatIoN数据集上通过五折交叉验证进行评估，SALFT在所有游戏中均取得了与全量微调相当的性能，且无统计学显著退化（p>0.05），同时仅更新约8%的参数（减少超过92%）。值得注意的是，在某一款游戏中，SALFT在所有指标和所有折中均持续优于全量微调与最佳基线方法，达到了理论最小p值（p=0.0625，精确双侧Wilcoxon符号检验）。

    arXiv:2610.07603v1 Announce Type: cross  Abstract: This article proposes Selective Affective Layer Fine-Tuning (SALFT), an efficient adaptation framework for Video Vision Transformers in player arousal recognition from gameplay. To bypass computationally expensive full fine-tuning, SALFT introduces a selection criterion based on the L2-norm change in layer parameters after brief adaptation, directly measuring representational shifts and providing a more stable basis than gradient-based alternatives. Evaluated via five-fold cross-validation on the Arousal Video Game AnnotatIoN dataset, SALFT achieves performance comparable to full fine-tuning across all games without statistically significant degradation ($p>0.05$), while updating only $\approx$8% of parameters (over 92% reduction). Notably, in one game, SALFT consistently outperforms both full fine-tuning and the best baseline across all metrics and folds, reaching the theoretical minimum p-value (p=0.0625, exact two-sided Wilcoxon sig
    
[^201]: 超越标量IoU：面向视频时序定位的基于Rollout组的结构化验证

    Beyond Scalar IoU: Structured Verification from Rollout Groups for Video Temporal Grounding

    [https://arxiv.org/abs/2610.07601](https://arxiv.org/abs/2610.07601)

    提出SUTURE方法，将验证从独立打分的标量IoU扩展为利用rollout组结构（组内分歧与位置覆盖率）的结构化验证，并可精确分解为标准IoU项加协方差修正项，从而改进基于可验证奖励强化学习的视频时序定位。

    

    使用可验证奖励的强化学习为将预训练模型适配到视频时序定位任务提供了一个自然的框架，因为生成的时间区间可以直接与真实标注区间进行打分比较。然而，现有的重叠验证器通常独立地对每个rollout进行打分，未能利用rollout组的联合结构。我们提出SUTURE，它将验证条件化于整个rollout组之上，并在两个互补的尺度上利用其结构：rollout之间的分歧控制目标被重新加权的强度，而每个位置上的覆盖率决定奖励质量被重新分配的位置。我们证明，所得到的验证器可以精确分解为标准的IoU项和一个由rollout组决定的协方差修正项。局部梯度诊断发现，在所分析的组中，该方法偏好覆盖支撑度相对较低的目标区域的响应。

    arXiv:2610.07601v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) provides a natural framework for adapting pretrained models to video temporal grounding, where generated temporal intervals can be scored directly against ground truth intervals. Yet existing overlap verifiers typically score each rollout independently, leaving the joint structure of the rollout group unused. We introduce SUTURE, which conditions verification on the rollout group and exploits its structure at two complementary scales: disagreement across rollouts controls how strongly the target is reweighted, while coverage at each position determines where reward mass is redistributed. We show that the resulting verifier admits an exact decomposition into the standard IoU term and a covariance correction determined by the rollout group. A local gradient diagnostic finds a preference for responses covering relatively less supported target regions in the analyzed groups. Across five t
    
[^202]: 面向世界模型鲁棒决策的潜在扰动建模

    Modeling Latent Disturbances for Robust Decision-Making in World Models

    [https://arxiv.org/abs/2610.07599](https://arxiv.org/abs/2610.07599)

    本文提出将潜在空间扰动建模为对习得潜在动力学的扰动，使其引发悲观但合理的状态转移，从而在世界模型的潜在空间中实现鲁棒决策。

    

    本文研究了世界模型（WMs）潜在空间中的鲁棒决策问题。鲁棒优化是一种数学框架，在给定明确定义的动力学和具有物理意义的扰动的情况下，机器人可以选择即使在最坏情况扰动下仍然有效的动作。然而，将这一原则应用于世界模型的习得潜在空间带来了一项根本性挑战：由于世界模型的状态空间和动力学完全是从高维观测中学习推断得到的，如何定义能够忠实表征底层系统不确定性的潜在空间扰动尚不清楚。我们的核心思想是将潜在空间扰动建模为对习得潜在动力学的扰动，该扰动会引发悲观但合理的状态转移。具体而言，我们通过结合一种能够捕捉合理转移的动力学感知相似性度量与分布外（原文在此处截断）……

    arXiv:2610.07599v1 Announce Type: cross  Abstract: In this paper, we study robust decision-making in the latent space of world models (WMs). Robust optimization is a mathematical framework where, given explicitly specified dynamics and physically meaningful disturbances, a robot can select actions that remain effective even under worst-case disturbances. However, applying this principle to the learned latent space of WMs introduces a fundamental challenge: because WMs have fully learned state spaces and dynamics inferred from high-dimensional observations, it is unclear how to define latent-space disturbances that faithfully represent uncertainty in the underlying system. Our key idea is to model a latent-space disturbance as a perturbation to the learned latent dynamics that induces pessimistic but plausible transitions. Specifically, we construct a set of plausible latent dynamics by combining a dynamics-aware similarity metric that captures plausible transitions with out-of-distribu
    
[^203]: LSC-DPO：学习信号控制的直接偏好优化

    LSC-DPO: Learning-Signal-Controlled Direct Preference Optimization

    [https://arxiv.org/abs/2610.07592](https://arxiv.org/abs/2610.07592)

    该论文提出LSC-DPO方法，从损失几何视角将DPO损失中的sigmoid因子识别为学习信号，并通过动态调节该信号使其维持在目标区间，从而解决DPO训练后期敏感性下降的问题，在多个基准上持续超越DPO。

    

    直接偏好优化（DPO）已成为一种无需奖励模型、使语言模型与偏好数据对齐的标准方法。然而，随着训练过程中缩放后的偏好边际不断增大，基于逻辑斯蒂的DPO损失对进一步变化的敏感性逐渐降低。我们从损失层面的几何视角研究DPO，并将sigmoid因子确定为表征目标函数局部敏感性的学习信号。基于这一视角，我们提出了学习信号控制的直接偏好优化（LSC-DPO），该方法在目标区间附近动态调节学习信号。对数空间分析确立了稳定跟踪目标学习信号区间的条件。在AlpacaEval 2、MT-Bench和Anthropic-HH上的实验表明，LSC-DPO持续优于DPO及强大的偏好优化基线方法。我们进一步发现，不同的系数初始化会引发不同的瞬态行为（摘要在此处被截断）。

    arXiv:2610.07592v1 Announce Type: new  Abstract: Direct Preference Optimization (DPO) has become a standard reward-model-free approach for aligning language models with preference data. However, as the scaled preference margin grows during training, the logistic DPO loss becomes progressively less sensitive to further changes. We study DPO from a loss-level geometric perspective and identify the sigmoid factor as a learning signal that characterizes the local sensitivity of the objective. Based on this view, we propose Learning-Signal-Controlled Direct Preference Optimization (LSC-DPO), which dynamically regulates the learning signal near a target regime. A log-space analysis establishes conditions for stable tracking of the target learning-signal regime. Experiments on AlpacaEval 2, MT-Bench, and Anthropic-HH show that LSC-DPO consistently improves over DPO and strong preference-optimization baselines. We further find that different coefficient initializations induce distinct transien
    
[^204]: 循环环路Transformer

    Recurrent Looped Transformer

    [https://arxiv.org/abs/2610.07591](https://arxiv.org/abs/2610.07591)

    提出循环环路Transformer（RLT），通过将层分配给并行因果编码器和循环解码器，使计算路径随序列长度增长而每个token成本保持固定，在状态跟踪和算法泛化任务上大幅超越固定深度的标准Transformer。

    

    状态跟踪需要对每个输入进行更新，但Transformer应用于每个token的深度是固定的，与序列长度无关。我们提出了循环环路Transformer（Recurrent Looped Transformer, RLT），它将模型层分配给一个并行因果编码器和一个循环解码器。在每个token处，解码器将编码器输出与上一个token的最终解码器状态合并，因此计算路径随序列长度增长，而每个token的计算成本保持固定。在六个算法任务上，我们在三个随机种子下比较了八层模型的五种分配方式与一个八层Transformer。在最多40位数据上训练后，两种RLT分配方式在所有种子中均能以100%的准确率将奇偶性判断泛化到256位，而标准Transformer仍停留在随机猜测水平。在训练长度八倍的基于交换的$S_5$置换跟踪任务上，RLT达到97%的最终状态准确率，而Transformer不足1%，且准确率随解码器深度增加而提升。在超出训练范围的模运算任务上……

    arXiv:2610.07591v1 Announce Type: cross  Abstract: State tracking requires an update at every input, but the depth a Transformer applies to each token is fixed regardless of sequence length. We introduce the Recurrent Looped Transformer (RLT), which splits its layers between a parallel causal encoder and a recurrent decoder. At each token, the decoder merges the encoder output with the previous token's final decoder state, so the computation path grows with sequence length at a fixed per-token cost. On six algorithmic tasks, we compare five splits of eight layers with an eight-layer Transformer over three seeds. Trained on at most 40 bits, two RLT splits generalize parity to 256 bits with 100% accuracy in every seed, while the Transformer stays at chance. On swap-based $S_5$ permutation tracking at eight times the training length, RLT reaches 97% final-state accuracy versus under 1% for the Transformer, and accuracy increases with decoder depth. On modular arithmetic beyond the trainin
    
[^205]: 基于跨平台用户历史的个人智能体中介推荐

    Personal-Agent Mediated Recommendation with Cross-Platform User History

    [https://arxiv.org/abs/2610.07588](https://arxiv.org/abs/2610.07588)

    提出了“个人智能体中介推荐”这一新范式及MediateRec基准，研究个人LLM智能体如何利用用户授权的跨平台历史来调解平台推荐排序，在有益挽救与有害覆盖之间取得平衡。

    

    现代推荐系统正从以平台为中心的个性化向用户主导的个性化转变，在这种模式下，个人LLM智能体可以代表用户跨多个服务行事。我们将这一新兴范式形式化为“个人智能体中介推荐”：平台推荐器利用平台本地信息对候选集进行排序，而个人智能体则利用用户授权的跨平台历史来调解该排序结果，并生成最终的top-K推荐列表。这种调解并非易事：平台排序中可能编码了个人智能体无法观察到的强大群体证据，因此有效的调解必须在有益的挽救与有害的覆盖之间取得平衡。为了研究这种权衡，我们提出了MediateRec基准，它包括可扩展的代理跨平台环境，以及在受控的平台-智能体信息边界下的真实跨平台测试。为了训练智能体使用跨平台历史有效进行调解……

    arXiv:2610.07588v1 Announce Type: new  Abstract: Modern recommendation is shifting from platform-centric personalization toward user-governed personalization, where a personal LLM agent can act on the user's behalf across services. We formalize this emerging paradigm as Personal-Agent Mediated Recommendation: a platform recommender ranks a candidate set using platform-local information, and a personal agent uses user-authorized cross-platform history to mediate the resulting ranking and produce the final top-K slate. Such mediation is nontrivial: the platform ranking can encode strong population evidence that the personal agent cannot observe, so effective mediation must therefore balance beneficial rescues against harmful overrides. To study this trade-off, we introduce MediateRec, a benchmark that includes scalable proxy cross-platform environments and a real cross-platform test under a controlled platform-agent information boundary. To train the agent to use cross-platform history e
    
[^206]: GraphCast中大气河流的机制可解释性

    Mechanistic Interpretability of Atmospheric Rivers in GraphCast

    [https://arxiv.org/abs/2610.07583](https://arxiv.org/abs/2610.07583)

    该研究通过对GraphCast训练稀疏自编码器，首次揭示这一AI天气模型内部稳定地计算出大气河流强度（综合水汽输送IVT）作为内部变量，并通过干预实验证实了其因果作用。

    

    尽管AI天气模型如今已能与业务化预报相媲美，但它们如何在内部表征大气仍是一个悬而未决的问题：特征归因只能揭示哪些输入模式重要，却无法说明模型计算了什么、以及如何在内部组合信息。我们在GraphCast上训练稀疏自编码器（SAE）以揭示其学习到的概念，并以大气河流作为研究焦点。标准SAE与Matryoshka SAE均表明，GraphCast将大气河流强度——以综合水汽输送（IVT）衡量——计算为一个稳定的内部变量，尽管IVT既非模型输入也非预测目标。与标准SAE非结构化的概念检索不同，Matryoshka SAE按重要性对概念进行排序并揭示概念之间的关系。大气河流概念在模型各深度层中持续存在，直接干预实验进一步证实了其因果性。该方法提供了一条寻找内部变量并确定模型实际依赖哪些变量的途径。

    arXiv:2610.07583v1 Announce Type: cross  Abstract: While AI weather models now rival operational forecasts, how they represent the atmosphere internally remains an open question: feature attribution reveals which input patterns matter, not what the model computes or how it combines information internally. We train sparse autoencoders (SAEs) on GraphCast to uncover its learned concepts, using atmospheric rivers as our phenomenon of focus. Both standard and Matryoshka SAEs show GraphCast computes atmospheric river intensity, measured by integrated vapor transport (IVT), as a stable internal variable, despite IVT being neither an input nor a target. In contrast to the unstructured concept retrieval of the standard SAE, the Matryoshka SAE orders concepts by importance and exposes their relations. Atmospheric river concepts persist across depth and direct interventions confirm causality. This method offers a way to find internal variables and determine which of them the model actually relie
    
[^207]: 三维线粒体形态计量学中的表示偏差、校正迁移与分辨率敏感性

    Representation Bias, Correction Transfer, and Resolution Sensitivity in Three-Dimensional Mitochondrial Morphometry

    [https://arxiv.org/abs/2610.07582](https://arxiv.org/abs/2610.07582)

    本文对三维线粒体形态计量学进行了实证可靠性评估，发现基于占据率的体积测量存在平均3.665%的系统性膨胀偏差，并证明通过移除深度偏移的校正迁移可将该体积误差降低约45%。

    

    定量成像流程可以对同一物体产生精确但系统性不同的测量结果。我们对三维线粒体形态计量学进行了实证可靠性评估，将表示偏差、受控处理干预、校正迁移和分辨率敏感性联系起来。使用来自光学显微镜三维线粒体形状库的2,720个发育对象，我们发现尽管组内相关系数高达0.994，基于占据率的体积测量平均比参考网格体积高出3.665%。边界分析识别出0.00304归一化单位的外向标签位移。在受控的标签流程重新实现中，移除深度偏移使所有55个分析对象的体积误差平均降低1.57个百分点，约占平均再现膨胀的45%；剩余误差的来源尚未被分离出来。使用基于占据率的冻结回归……（摘要被截断）

    arXiv:2610.07582v1 Announce Type: new  Abstract: Quantitative imaging pipelines can produce precise but systematically different measurements of the same object. We present an empirical reliability assessment of three-dimensional mitochondrial morphometry that connects representation bias, a controlled processing intervention, correction transfer, and resolution sensitivity. Using 2,720 development objects from the 3D Mitochondria Shape Library for Optical Microscopy, we find that occupancy-derived volumes exceed reference mesh volumes by 3.665% on average despite an intraclass correlation coefficient of 0.994. Boundary analysis identifies an outward label displacement of 0.00304 normalized units. In a controlled label-pipeline reimplementation, removing the depth offset reduces volume error in all 55 analyzed objects by a mean of 1.57 percentage points, approximately 45% of mean reproduced inflation; the source of the remainder is not isolated. A frozen regression using occupancy-deri
    
[^208]: LOGIC：面向航空航天电气系统中基于意图的变化影响分析的LLM基准

    LOGIC: An LLM Benchmark for Intent-Grounded Change Impact in Aerospace Electrical Systems

    [https://arxiv.org/abs/2610.07580](https://arxiv.org/abs/2610.07580)

    LOGIC是一个航空航天电气系统领域的受控基准，用于评估语言模型能否根据工程请求意图从确定性候选变更清单中正确选择变更，并通过类型化电气可追溯性图传播其影响，实验表明仅门控结构化证据方法在明确锚定的选择案例上达到了完美的F1分数1.0000。

    

    航空航天电气设计修订中可能包含多个真实变更，但一份工程请求可能仅授权其中一部分变更。因此，将每个检测到的差异都进行传播可能会产生过于宽泛的影响报告。我们提出了LOGIC，这是一个受控基准与评估框架，其中可本地部署的语言模型先将请求锚定在确定性的候选变更清单上，然后再将选定的变更通过类型化的电气可追溯性图进行传播。这种分离设计使得候选选择错误能够与下游传播错误区分开来。LOGIC包含168个场景，其中包括144个选择案例和24个弃权案例。我们评估了三个7B-8B参数规模的模型，并与意图无关方法、词汇匹配方法和结构化证据方法进行比较，同时设置了oracle根作为性能上限。在96个明确锚定的选择案例中，仅门控结构化证据方法达到了1.0000的候选F1分数，而token词汇匹配方法仅为0.9677。

    arXiv:2610.07580v1 Announce Type: new  Abstract: Aerospace electrical-design revisions can contain multiple genuine changes, although an engineering request may authorize only a subset. Propagating every detected difference can therefore produce overly broad impact reports. We present LOGIC, a controlled benchmark and evaluation framework in which locally deployable language models ground a request in a deterministic candidate-change inventory before selected changes are propagated through a typed electrical traceability graph. This separation permits candidate-selection errors to be distinguished from downstream propagation errors. LOGIC contains 168 scenarios, including 144 selection and 24 abstention cases. We evaluate three 7--8B models against intent-agnostic, lexical, and structured-evidence methods, with an oracle-root upper bound. On 96 explicitly anchored selection cases, gate-only structured evidence achieves candidate F1 of 1.0000, compared with 0.9677 for token-lexical matc
    
[^209]: 与未来协作者合作：交错参与下的多智能体强化学习

    Cooperating with Future Collaborators: Multi-Agent RL under Staggered Participation

    [https://arxiv.org/abs/2610.07578](https://arxiv.org/abs/2610.07578)

    该论文针对交错参与设定下多智能体强化学习的跨时间、跨智能体学习依赖问题，提出了交错参与学习（SPL）方法，通过为早期智能体提供前瞻性获取监督、为后续智能体提供基于结果的接收者学习，使早期智能体学会留下有用的任务相关信息、后续智能体学会有效利用这些信息。

    

    在合作式多智能体强化学习（MARL）中，智能体通常在同时参与的情况下进行训练，然而在许多任务中，一些智能体会更早行动，并留下与任务相关的信息，这些信息对后续参与的智能体会变得有用。我们将这种设定称为交错参与（Staggered Participation，SP），它引入了跨时间、跨智能体的学习依赖性，因为早期动作可能通过其提供的信息以及使用该信息的后续策略来影响回报。因此，在SP设定下学习既需要识别哪些信息对未来的决策有用，也需要学习后续智能体应如何使用这些信息。我们提出了交错参与学习（Staggered Participation Learning，SPL），这是一种训练时增强方法，通过为早期智能体提供前瞻性获取监督、为后续智能体提供基于结果的接收者学习来解决这两个问题。我们在多种基于策略的MARL骨干网络、环境和交错参与场景下对SPL进行了评估。

    arXiv:2610.07578v1 Announce Type: new  Abstract: In cooperative Multi-Agent Reinforcement Learning (MARL), agents are often trained under concurrent participation, while in many tasks some agents act earlier and leave task-relevant information that becomes useful to agents participating later. We study this setting as staggered participation (SP), which introduces a cross-time, cross-agent learning dependency because an early action may affect the return through the information it provides and the later policy that uses it. Learning under SP therefore requires both identifying what information is useful for future decisions and learning how later agents should use it. We propose Staggered Participation Learning (SPL), a training-time augmentation that addresses these two parts with prospective acquisition supervision for earlier agents and outcome-supervised receiver learning for later agents. We evaluate SPL across multiple policy-based MARL backbones, environments, and staggered-part
    
[^210]: 一致却错误：基于医学大语言模型共识形成方式的认证弃权

    Unanimously Wrong: Certified Abstention from How Medical LLM Consensus Forms

    [https://arxiv.org/abs/2610.07570](https://arxiv.org/abs/2610.07570)

    论文指出医学LLM问答系统中基于答案一致性的置信信号无法区分“通过证据化解分歧达成的一致”与“所有样本共享同一误解导致的一致性错误”，并提出ProbeGuard框架，根据共识的形成过程而非最终状态做出可认证的弃权决策。

    

    在临床实践中，独立专家之间的一致意见被视为可靠性的证据，而多轮共识已成为智能体化医学问答系统的核心机制。当这类系统必须决定是否信任自己的答案时，主要的信号仍然是一致性——此时是多个采样答案之间的一致性。但一致性是正确性的一个脆弱代理指标。系统可能全体一致地错误，即在每个采样中都返回相同的错误答案，而在这类问题上，基于一致性的信号不携带任何信息。原因在于这些信号只读取共识的最终状态，却丢弃了共识是如何达成的。通过用证据化解分歧而达成的一致，与从第一个样本起就存在的一致（因为每个样本共享同一个错误认知）在最终状态上看起来完全相同。ProbeGuard是一个认证弃权框架，它将弃权决策建立在共识形成的方式之上。

    arXiv:2610.07570v1 Announce Type: new  Abstract: In clinical practice, agreement among independent experts is treated as evidence of reliability, and multi-round consensus has become a core mechanism of agentic medical question-answering systems. When such a system must decide whether to trust its own answer, the prevailing signal is again agreement, now among the sampled answers. But agreement is a fragile proxy for correctness. A system can be unanimously wrong, returning the same incorrect answer on every sample, and on these questions agreement-based signals carry no information. The cause is that these signals read only the final state of the consensus and discard how it was reached. Agreement that was reached by resolving disagreement with evidence looks identical, at the end, to agreement that was present from the first sample because every sample shares one misconception. ProbeGuard is a certified abstention framework that bases the abstention decision on how the consensus form
    
[^211]: OpenSplatGraph：从稠密语义地图到结构化场景图，实现开放词汇机器人感知

    OpenSplatGraph: From Dense Semantic Maps to Structured Scene Graphs for Open-Vocabulary Robot Perception

    [https://arxiv.org/abs/2610.07569](https://arxiv.org/abs/2610.07569)

    该论文提出OpenSplatGraph框架，首次实现直接从在线的基于高斯泼溅的开放词汇稠密语义地图中构建持久化三维场景图，通过可靠性感知的语义场进行置信度感知的对象提取，弥合了稠密语义建图与结构化场景图推理之间的鸿沟。

    

    具备语义理解的稠密三维建图对于机器人在复杂环境中的感知至关重要。近期基于三维高斯泼溅的建图方法能够实现高保真几何与高效的开放词汇感知，但通常将语义表示为非结构化的特征场，限制了以对象为中心的推理能力。相比之下，三维场景图通过显式建模对象及其相互关系来实现结构化推理，但通常由稀疏的几何表示构建，未能充分利用稠密语义地图。在本工作中，我们提出了OpenSplatGraph，这是一个统一框架，能够直接从在线的基于高斯泼溅的开放词汇语义地图中构建持久化的三维场景图。该框架通过一个可靠性感知的语义场来增强稠密语义地图，该语义场维护轻量级的观测统计信息，以实现置信度感知、查询条件下的对象提取。（摘要原文在此处截断）

    arXiv:2610.07569v1 Announce Type: cross  Abstract: Dense 3D mapping with semantic understanding is essential for robotic perception in complex environments. Recent 3D Gaussian Splatting-based mapping approaches enable high-fidelity geometry and efficient open-vocabulary perception, but typically represent semantics as unstructured feature fields that limit object-centric reasoning. In contrast, 3D scene graphs explicitly model objects and their relationships for structured reasoning, but are commonly constructed from sparse geometric representations that do not fully exploit dense semantic maps. In this work, we present OpenSplatGraph, a unified framework that constructs persistent 3D scene graphs directly from an online Gaussian-based open-vocabulary semantic map. The proposed framework augments the dense semantic map with a reliability-aware semantic field that maintains lightweight observation statistics for confidence-aware, query-conditioned object extraction. Extracted object ins
    
[^212]: 互补特征域：信息保持并不意味着预测贡献保持

    Complementary Feature Domains: Information Preservation Does Not Imply Predictive-Contribution Preservation

    [https://arxiv.org/abs/2610.07565](https://arxiv.org/abs/2610.07565)

    该论文提出互补特征域（CFD）理论，证明保持香农信息并不能保证保持预测贡献，并用贡献缺陷量化重编码下上下文贡献的变化，其上界由可达动作集间的行为距离与联盟不相容性之和界定。

    

    arXiv:2610.07565v1 公告类型：交叉 摘要：互补特征域（CFD）理论将预测价值刻画为一个由表示及其实现族共同诱导的、以上下文为索引的贡献系统。我们证明，香农信息的保持并不意味着这一贡献系统的保持：一个可逆的表示变换可以在保持目标信息不变的同时，改变受限决策族下的预测贡献。我们通过一个CFD贡献缺陷来形式化由此产生的转变，该缺陷度量在受控重编码下上下文贡献的变化程度。对于有界Lipschitz效用，我们证明每个联盟效用的偏移由重编码前后可达动作集之间的行为距离所界定；因此，每个上下文贡献缺陷都由相应联盟不相容性之和所界定。精确的行为封闭性带来不变性，而行为封闭性的逐渐减弱……（摘要截断）

    arXiv:2610.07565v1 Announce Type: cross  Abstract: Complementary Feature Domains (CFD) theory characterizes predictive value as a context-indexed contribution system induced jointly by representations and their realization family. We show that Shannon-information preservation does not imply preservation of this contribution system: an invertible representation transformation can leave target information unchanged while altering predictive contribution under a restricted decision family. We formalize the resulting transition through a CFD contribution defect that measures how contextual contributions change under controlled recoding. For bounded Lipschitz utility, we show that each coalition utility shift is bounded by the behavioral distance between the attainable action sets before and after recoding; consequently, every contextual contribution defect is bounded by the sum of the corresponding coalition incompatibilities. Exact behavioral closure yields invariance, while increasingly 
    
[^213]: 学习GFlowNets混合体

    Learning a Mixture of GFlowNets

    [https://arxiv.org/abs/2610.07562](https://arxiv.org/abs/2610.07562)

    提出了一个描述GFlowNets混合体的通用理论框架，将其细分为连续索引（CI）和离散索引（DI）两类：前者通过随机特征扩展与谱移位可证明地提升采样器的表达能力并降低学习不稳定性，后者统一了已有训练方法并支撑了新提出的分层条件化（SC）GFlowNets。

    

    学习一组GFlowNets集成以从离散目标分布中进行采样，已成为比单一采样器实现更好的状态空间探索和收敛性的常用方法。然而，这些方法通常会给基础模型带来较大的运行时开销，且它们之间的概念联系仍不明确。为解决这一问题，我们首先提出了一个用于描述GFlowNets混合体的通用理论框架，并将其细分为连续索引（CI）和离散索引（DI）两类集合。一方面，我们证明CI GFlowNets可以通过随机特征扩展的视角来解释，在图结构任务中可证明地提升采样器的表达能力，并通过谱移位降低学习的不稳定性。另一方面，我们证明DI GFlowNets涵盖了此前已有的GFlowNet训练方法，并为新提出的分层条件化（SC）GFlowNets奠定了基础。

    arXiv:2610.07562v1 Announce Type: cross  Abstract: Learning an ensemble of GFlowNets to sample from a discrete target distribution has become a common approach for achieving better state space exploration and convergence than that of a monolithic sampler. However, these methods often add a substantial runtime overhead to the base model, and their conceptual connection remains elusive. To address this, we first propose a general-purpose theoretical framework for describing a mixture of GFlowNets, which we specialize into continuously (CI) and discretely indexed (DI) collections. On the one hand, we show CI GFlowNets can be interpreted through the lens of a random features expansion, provably boosting the sampler's expressivity in graph-structured tasks and reducing learning instability via spectral shifting. On the other hand, we demonstrate DI GFlowNets encompass prior approaches for GFlowNet training and provide the foundation for the newly proposed Stratum-Conditioned (SC) GFlowNets.
    
[^214]: 面向可合成分子设计的路线潜空间导航

    Navigating Route Latent Space for Synthesizable Molecular Design

    [https://arxiv.org/abs/2610.07560](https://arxiv.org/abs/2610.07560)

    RouteFlow框架将可合成分子设计重构为在连续路线潜空间中的搜索问题，每个潜向量对应一条完整合成路线，并通过奖励引导的流匹配采样器实现性质优化与可合成性的内在统一。

    

    目标导向的分子设计近年来发展迅速，但相当大比例的设计分子在实践中仍然难以合成，限制了其实际应用价值。先前考虑可合成性的方法要么将生成的分子投影回偏离预期目标的可合成类似物，要么直接在缺乏连续景观、难以高效搜索的离散合成空间中进行优化。我们认为这一限制主要源于搜索空间本身，而非优化器。为解决这一问题，我们提出了RouteFlow框架，将可合成分子设计重新表述为对连续路线潜空间的搜索，其中每个潜向量都映射回一条完整的合成路线，从而内在地保证了可合成性。为了在该空间中进行导航，我们采用奖励引导的流匹配作为高效采样器，引导搜索走向高性质区域。由于奖励优化可能会将潜向量推离（摘要在此处被截断）

    arXiv:2610.07560v1 Announce Type: new  Abstract: Goal-directed molecular design has advanced rapidly, yet a substantial proportion of designed molecules remain difficult to synthesize in practice, limiting their real-world utility. Prior synthesizability-aware methods either project generated molecules back to synthesizable analogs that deviate from the intended target, or optimize directly in discrete synthesis spaces that lack a continuous landscape for efficient search. We argue that this limitation mainly comes from the search space rather than the optimizer. To address this, we propose RouteFlow, a framework that reformulates synthesizable molecular design as a search over a continuous route latent space, where each latent maps back to a complete synthesis route and synthesizability is inherently preserved. To navigate this space, we adopt reward-guided flow matching as an efficient sampler that steers toward high-property regions. Since reward optimization may push latents off th
    
[^215]: 看见不可见之物：面向温度与辐射感知的VLA导航的物理引导视觉提示

    Seeing the Invisible: Physics-Guided Visual Prompting for Temperature- and Radiation-Aware VLA Navigation

    [https://arxiv.org/abs/2610.07558](https://arxiv.org/abs/2610.07558)

    提出物理引导视觉提示（PG-VP），将不可见的辐射或温度危险转化为动态虚拟障碍物视觉提示，使冻结的VLA导航模型无需重新训练即可规避多种不可见风险。

    

    视觉-语言-动作模型已成为视觉与语言导航的主要范式。然而，在安全关键设施中，辐射或温度骤升等不可见风险无法被RGB相机检测到，且处理每一种风险的成本都很高，需要新的编码器、新的数据以及模型重新训练。我们提出物理引导视觉提示，这是一种即插即用的多模态感知模块，它转而复用冻结的VLA模型已经擅长的能力：避开可见障碍物。给定一个近端的辐射源或热源，PG-VP执行物理引导的风险评估以确定避让方向，并叠加一个在连续帧之间移动的相应虚拟障碍物（动态视觉提示）。导航策略随后会自然地绕过这一不可见危险。无论危险类型如何，都使用相同的虚拟障碍物，因此视觉提示模式保持固定

    arXiv:2610.07558v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have become a major paradigm for Vision-and-Language Navigation (VLN). However, in safety-critical facilities, invisible risks such as radiation or temperature spikes cannot be detected by an RGB camera, and handling each risk is expensive, requiring a new encoder, new data, and model retraining. We propose Physics-Guided Visual Prompting (PG-VP), a plug-and-play multimodal perception module that instead reuses what a frozen VLA model already does well: avoiding visible obstacles. Given a proximal radiation or thermal source, PG-VP performs a physics-guided risk assessment to determine the avoidance direction and overlays a corresponding virtual obstacle that moves across consecutive frames (Dynamic Visual Prompting). The navigation policy then naturally detours around this invisible hazard. The identical virtual obstacle is used regardless of hazard type, so the visual prompting pattern remains fixe
    
[^216]: CheckerBench：长程智能体能否合成静态分析检查器？

    CheckerBench: Can Long-Horizon Agents Synthesize Static-Analysis Checkers?

    [https://arxiv.org/abs/2610.07557](https://arxiv.org/abs/2610.07557)

    该论文提出了首个可执行基准CheckerBench（包含源自297个CVE、167个仓库的300个任务），用于评估长程智能体能否在真实代码仓库中端到端合成可用的静态分析检查器，并配套CheckerLab统一评估框架衡量诊断对比度、补丁定位、误报率和工具使用等指标。

    

    静态分析检查器合成要求智能体理解缺陷规范、检查代码仓库、实现分析器特定的逻辑，并通过反复的编译和分析反馈来完善检查器。现有的编码智能体基准主要关注补丁生成或漏洞检测等任务，很少评估智能体能否在代码仓库中从头到尾开发出一个可用的检查器。我们提出了CheckerBench，一个包含300个任务的可执行基准，这些任务源自167个代码仓库中的297个CVE，涵盖85种CWE类型和五种语言生态系统。每个任务包含存在漏洞和已修复的代码版本、固定的分析环境以及检查器脚手架。我们还进一步推出了CheckerLab，这是一个统一的评估框架，可独立重建提交的检查器，并衡量漏洞-修复诊断对比度、补丁定位能力、误报率和工具使用情况。在21种模型-框架配置和三个独立……（摘要原文截断）

    arXiv:2610.07557v1 Announce Type: cross  Abstract: Static-analysis checker synthesis requires agents to interpret a defect specification, inspect a repository, implement analyzer-specific logic, and refine the checker through repeated compilation and analysis feedback. Existing coding-agent benchmarks focus on tasks such as patch generation or vulnerability detection and rarely assess whether an agent can develop a working checker in a repository from start to finish. We introduce CheckerBench, an executable benchmark of 300 tasks derived from 297 CVEs across 167 repositories, 85 CWEs, and five language ecosystems. Each task includes vulnerable and fixed revisions, a pinned analysis environment, and a checker scaffold. We further introduce CheckerLab, a common evaluation framework that independently rebuilds submitted checkers and measures vulnerable-fixed diagnostic contrast, patch localization, false positives, and tool use. Across 21 model-harness configurations and three independen
    
[^217]: 解耦的多智能体编排

    Decoupled Multi-Agent Orchestration

    [https://arxiv.org/abs/2610.07556](https://arxiv.org/abs/2610.07556)

    提出 DeOrch 框架，将多智能体编排中的任务规划与工作者选择解耦，通过两阶段规划器和无身份信息的匹配性反馈实现条件化信用分配，支持新工作者在线加入而无需重新训练，并在分布内外任务上超越了先前的自动多智能体系统方法。

    

    学习型编排方法可以自动构建有效的语言模型多智能体系统，但现有方法将规划与固定的工作者池耦合在一起，并从同一个最终结果中训练任务分解与协作，这限制了系统的迁移能力并使信用分配变得模糊。我们提出 DeOrch，将不依赖具体工作者的规划与具体的工作者选择分离开来。其两阶段规划器首先在不使用任何工作者信息的情况下分解任务，然后利用来自工作者池的紧凑且不含工作者身份信息的匹配性反馈来选择协作操作，从而能够对分解决策和协作决策进行条件化信用分配。一个轻量级匹配器根据工作者在固定探测集上的行为表现来估计其适用性，并通过上下文老虎机在线适应调整，使新工作者无需重新训练规划器或匹配器即可被纳入系统。在多种分布内和分布外任务上，DeOrch 优于先前的自动多智能体系统方法。

    arXiv:2610.07556v1 Announce Type: new  Abstract: Learned orchestration can automatically construct effective language-model multi-agent systems, but existing approaches couple planning to fixed worker pools and train decomposition and collaboration from the same terminal outcome, limiting transfer and obscuring credit assignment. We introduce DeOrch, which separates worker-agnostic planning from concrete worker selection. Its two-stage planner first decomposes the task without worker information, then chooses collaboration operations using compact, worker-identity-free matchability feedback from the pool, enabling conditional credit assignment to decomposition and collaboration decisions. A lightweight matcher estimates worker suitability from behavior on a fixed probe set and adapts online with a contextual bandit, allowing new workers to be incorporated without retraining the planner or matcher. Across diverse in- and out-of-distribution tasks, DeOrch outperforms prior automatic MAS 
    
[^218]: 准入哪些与何时准入：面向数据中心化小语言模型微调的梯度准入方法

    Which and When to Admit: Gradient Admission for Data-Centric Small Language Model Finetuning

    [https://arxiv.org/abs/2610.07553](https://arxiv.org/abs/2610.07553)

    提出GRADE框架，通过状态感知选择器持续接纳与演化中的多任务梯度场对齐的样本，并用自校准步级门控在子空间接近饱和时拒绝破坏性更新，从而同时解决LoRA微调中的梯度冲突、静态数据选择和子空间饱和三大问题，提升小语言模型微调效果。

    

    LoRA微调使小语言模型（SLM）能够在低秩更新子空间内适应异构的指令数据，但这使其容易受到三个结构性问题的影响：相互冲突的梯度会彼此抵消、静态的数据选择无法追踪不断演化的学习动态、以及子空间饱和导致后续更新覆盖掉有用的方向。我们认为，有效的适配因此需要控制哪些数据诱导的梯度进入LoRA子空间以及何时进入。我们提出了GRADE（GRadient-Aligned Data-centric rEcipe，梯度对齐的数据中心化配方），这是一个数据中心化框架，结合了两种机制：一个状态感知选择器，持续接纳与不断演化的多任务梯度场对齐的样本；以及一个自校准的步级门控，用于拒绝在接近饱和时可能造成破坏性覆盖的更新。在三个当前一代的骨干模型和一个异构的七数据集指令池上，GRADE优于强大的数据选择（方法）……

    arXiv:2610.07553v1 Announce Type: cross  Abstract: LoRA fine-tuning adapts small language models (SLMs) to heterogeneous instruction data within a low-rank update subspace, making it vulnerable to three structural problems: conflicting gradients that cancel, static data selection that cannot track evolving learning dynamics, and subspace saturation that causes later updates to overwrite useful directions. We argue that effective adaptation therefore requires controlling which data-induced gradients enter the LoRA subspace and when. We propose GRADE (GRadient-Aligned Data-centric rEcipe), a data-centric framework combining two mechanisms: a state-aware selector that continually admits samples aligned with the evolving multi-task gradient field, and a self-calibrating step-level gate that rejects updates likely to cause destructive overwrite near saturation. Across three current-generation backbones and a heterogeneous seven-dataset instruction pool, GRADE outperforms strong data-selecti
    
[^219]: 基础模型辅助的多智能体强化学习用于无线随机接入网络优化

    Foundation Model-Aided Multi-Agent Reinforcement Learning for Wireless Random Access Network Optimization

    [https://arxiv.org/abs/2610.07550](https://arxiv.org/abs/2610.07550)

    该论文提出一种基础模型辅助的actor-critic多智能体强化学习算法，以显著降低无线随机接入网络优化任务中的训练开销，并证明了其与采用评论者模型交换和线性近似的传统MARL方法具有相同的收敛阶。

    

    随机接入（RA）是处理来自多个终端的不可预测数据流量的最基础的介质访问控制（MAC）层调度方案之一。虽然多智能体强化学习（MARL）已被用于优化基于随机接入的无线网络，但其依赖于经验驱动的分布式策略学习，为每个优化任务带来显著的训练开销，限制了其在实际应用中的可行性。在本工作中，我们提出利用基础模型（FM）来提高MARL在多样化随机接入网络优化任务中的效率。具体而言，我们在基于共识的去中心化MARL架构中设计了一种FM辅助的actor-critic算法，并在本地奖励交换和非线性价值函数近似的条件下提供了其收敛性分析，表明我们的算法达到了与采用评论者模型交换和线性近似的传统MARL相同的收敛阶。

    arXiv:2610.07550v1 Announce Type: cross  Abstract: Random access (RA) is one of the most foundational medium access control (MAC) layer scheduling schemes for handling unpredictable data traffic from multiple terminals. While multi-agent reinforcement learning (MARL) has been explored to optimize RA-based wireless networks, its reliance on experience-driven, distributed policy learning incurs significant training overhead for each optimization task, limiting its feasibility in real-world applications. In this work, we propose to leverage a foundation model (FM) to improve MARL efficiency across diverse RA network optimization tasks. Specifically, we design an FM-aided actor-critic algorithm within a consensus-based decentralized MARL architecture and provide its convergence analysis under local reward exchanges and nonlinear value function approximations to show that our algorithm achieves the same convergence order as the conventional MARL with critic model exchanges and linear approx
    
[^220]: 对大型语言模型在广告相关性评估中偏见的系统性研究

    A Systematic Investigation of Bias in Large Language Models for Advertising Relevance

    [https://arxiv.org/abs/2610.07544](https://arxiv.org/abs/2610.07544)

    该论文通过反事实框架系统性研究了大型语言模型在广告相关性判断中的公平性问题，发现广告主身份、输入语言以及人口统计学措辞（尤其在就业、住房和信贷等敏感领域）都会显著影响模型的判断结果，并呈现出与常见刻板印象一致的偏见。

    

    大型语言模型（LLM）越来越多地被用于判断广告与查询的匹配程度，但这些判断的公平性却很少受到关注。我们对LLM在查询与广告相关性判断中的公平性进行了系统性研究。我们的反事实框架考察了广告主身份及其可能的知名度、输入语言以及人口统计学措辞的影响。我们研究了作为类别相关性判断器的GPT-4o，以及一个专门为相关性预测训练的Qwen-7B模型。广告主和语言实验使用从真实广告日志中采样的查询与广告对；受控的合成查询则被用于研究就业、住房和信贷领域中的人口统计学关联。对于这两个模型而言，改变广告主身份或输入语言都可能改变相关性评估结果。部分人口统计学比较还显示出与常见刻板印象一致的模式。

    arXiv:2610.07544v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to judge how well an advertisement matches a query, but the fairness of these judgments has received limited attention. We conduct a systematic study of fairness in relevance judgments made by LLMs for queries and advertisements. Our counterfactual framework examines the effects of advertiser identity and possible popularity, input language, and demographic wording. We study GPT-4o as a categorical relevance judge and a Qwen-7B model trained specifically for relevance prediction. The advertiser and language experiments use query and advertisement pairs sampled from real advertising logs. Controlled synthetic queries are used to study demographic associations in employment, housing, and credit. For both models, changing the advertiser identity or input language can alter the relevance assessment. Selected demographic comparisons also show patterns consistent with common stereotypes, parti
    
[^221]: 在异构大语言模型模拟中解耦模型与角色人格

    Disentangling Models from Personas in Heterogeneous LLM Simulations

    [https://arxiv.org/abs/2610.07535](https://arxiv.org/abs/2610.07535)

    该研究通过模拟由多个基础模型驱动的异构社交网络，发现智能体获得的互动量更多取决于其基础模型而非被分配的角色人格，且随着模型数量增加，模型间效应显著增强，表明网络动态可能在大规模下收敛于基础模型效应。

    

    使用大语言模型（LLM）的多智能体模拟通常以单一基础模型驱动整个智能体网络，这忽略了模型间效应，而此类效应可能在现实部署中主导互动动态。为了证明这一点，我们模拟了一个由多个不同基础模型驱动的异构社交网络，并表明智能体所获得的互动量更多取决于其基础模型，而非其被分配的角色人格。当混合中加入更多模型时，基础模型的吸引或排斥效应会显著增强，这表明网络动态在大规模下可能收敛于基础模型效应。为了帮助解释这一效应，我们进行了一系列内容中介分析，展示了基础模型在不同情境下的可预测性，以及模型的词汇模式与互动最大化风格之间的关系。鉴于近期大规模多智能体交互的发展，这项工作……

    arXiv:2610.07535v1 Announce Type: cross  Abstract: Multi-agent simulations with large language models (LLMs) often operate networks of agents with a single base model. This overlooks the inter-model effects which may dominate engagement dynamics in real-world deployments. To show this, we simulate a heterogeneous social network powered by several different base models and show that the amount of engagement an agent receives depends more on its base model than on its assigned persona. The attraction or repulsion effects of a base model strengthen dramatically when more models are added in the mix, suggesting that networks dynamics may converge to base model effects at scale. To help explain this effect, we conduct a series of content-mediating analyses, showing the predictability of base models across contexts as well as the relationship between a model's lexical patterns and an engagement-maximizing style. In light of recent developments in mass multi-agent interaction, this work under
    
[^222]: 通过来自暗知识的模型无关潜在安全信号保障大语言模型安全

    Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge

    [https://arxiv.org/abs/2610.07532](https://arxiv.org/abs/2610.07532)

    提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。

    

    大语言模型（LLM）发展迅速，引发了人们对其安全性日益增长的关注。近期工作提出了检测和防御攻击的方法，包括利用模型隐藏状态的解码阶段防御。然而，现有的解码阶段防御存在两个局限性。首先，它们在安全性与过度拒绝之间存在权衡，即加强安全性会降低模型对良性查询的有用性。其次，许多这些方法依赖于内部隐藏状态，因此仅限于特定架构，带来大量开销且在不同模型间的泛化能力有限。为解决这些限制，我们提出了 LADE（Latent Safety Signals for Defense），该方法通过在首token输出概率分布中，对比有害查询与良性查询，从暗知识（即输出概率分布中超出argmax所携带的信息）中提取潜在安全信号……

    arXiv:2610.07532v1 Announce Type: cross  Abstract: LLMs have advanced rapidly, raising growing concerns about their safety. Recent work has proposed approaches to detect and defend against attacks including defenses at decoding stage that leverage models' hidden states. However, existing decoding-stage defenses suffer from two limitations. First, they introduce a trade-off between safety and over-refusal, where strengthening safety degrades the model's helpfulness on benign queries. Second, many of these methods rely on internal hidden states and are thus restricted to specific architectures, incurring substantial overhead and limited generalization across models. To address these limitations, we introduce LADE (Latent Safety Signals for Defense), which leverages latent safety signals extracted by contrasting harmful and benign queries from dark knowledge (i.e., information carried by the output probability distribution beyond its argmax) in the first-token output probability distribut
    
[^223]: 锚定塑造规划之物：重新思考自动驾驶中物理智能的接地性

    Grounding What Shapes the Plan: Rethinking Groundedness for Physical Intelligence in Autonomous Driving

    [https://arxiv.org/abs/2610.07521](https://arxiv.org/abs/2610.07521)

    论文提出GroundAct框架，以物理实体及其交互作为接地的基本单元，通过轻量级参考标记让被选中实体与不断演化的规划方案进行交互来修正规划，从而在自动驾驶中建立起从接地推理到规划动作的显式路径。

    

    驾驶模型越来越多地将推理建立在因果关系、空间结构、感知证据和预测的未来之上。这些进展使推理更加忠实于驾驶场景，但留下了一个根本性的未解问题：当模型最终输出的是一个动作时，“接地性”应当意味着什么？正确的接地推理本身并不能确保理想的驾驶结果。我们提出了GroundAct，它从一个简单的前提出发：驾驶是通过物理实体及其相互作用展开的。因此，实体成为接地的基本单元；一个轻量级的参考标记使每个被选中实体的连续状态在符号推理过程中保持可寻址；并且只有被引用实体与不断演化的规划方案之间的交互才能对规划进行修正。其结果是建立了一条从“推理所锚定的内容”到“规划所执行的动作”的显式路径，我们称之为接地规划。为评估其实际价值，我们对GroundAct进行了评估……

    arXiv:2610.07521v1 Announce Type: new  Abstract: Driving models increasingly ground reasoning in causal relations, spatial structure, perceptual evidence, and predicted futures. These advances make reasoning more faithful to the driving scene, but leave a fundamental question unresolved: what should groundedness mean when the model ultimately outputs an action? Correctly grounded reasoning does not, by itself, ensure desirable driving outcomes. We introduce GroundAct, which starts from a simple premise: driving unfolds through physical entities and their interactions. Entities therefore become the unit of grounding; a lightweight reference token keeps each selected entity's continuous state addressable through symbolic reasoning; and only the referenced entities' interactions with the evolving proposal correct the plan. The result is an explicit path from what reasoning grounds to what the plan does, which we call grounded planning. To assess its practical value, we evaluate GroundAct 
    
[^224]: 有害SFT在大语言模型检查点更新中留下连续痕迹

    Harmful SFT Leaves a Continuous Trace in LLM Checkpoint Updates

    [https://arxiv.org/abs/2610.07518](https://arxiv.org/abs/2610.07518)

    该研究发现有害监督微调会在大语言模型的检查点更新中留下连续且可读取的痕迹，通过检查点级别的坐标即可高精度检测有害目标的存在，从而实现无需运行模型的安全审计。

    

    对训练后大语言模型的安全审计通常依赖于模型行为，需要执行模型并取决于可用评估的覆盖范围。这项工作提出了一个不同的问题：在监督微调（SFT）期间优化的目标行为是否会在检查点更新中直接留下可读取的证据？我们发现，有害依从性SFT会在检查点更新空间中诱导出一种连续的、依赖于目标的排序。使用由纯有害依从性、安全目标和良性效用SFT所定义的参考几何结构，我们发现一个检查点级别的坐标s_H能够追踪受控的有害目标组成，在四个7-8B骨干模型上达到0.986-0.992的Spearman相关系数，并且相同的排序在更大模型规模上依然存在。匹配的依从性与拒绝对照实验表明，这种检查点痕迹反映的是SFT目标本身，而非有害输入的暴露，同时额外的对照实验排除了其他可能的解释。

    arXiv:2610.07518v1 Announce Type: cross  Abstract: Safety auditing of post-trained large language models typically relies on model behavior, requiring model execution and depending on the coverage of available evaluations. This work asks a different question: Do the target behaviors optimized during supervised fine-tuning (SFT) leave readable evidence directly in checkpoint updates? We find that harmful-compliance SFT induces a continuous, objective-dependent ordering in checkpoint-update space. Using a reference geometry defined by pure harmful-compliance, safety-targeted, and benign-utility SFT, we find that a checkpoint-level coordinate s_H tracks controlled harmful-objective composition with Spearman correlations of 0.986-0.992 across four 7-8B backbones, with the same ordering persisting at larger model scales. Matched compliance-versus-refusal controls show that this checkpoint trace reflects the SFT objective rather than harmful-input exposure, while additional controls rule out
    
[^225]: 从局部证据到安全判定：视觉-语言模型中的因果追踪

    From Local Evidence to Safety Verdicts: Causal Tracing in Vision-Language Models

    [https://arxiv.org/abs/2610.07514](https://arxiv.org/abs/2610.07514)

    该研究提出SSU-Bench数据集并通过配对输入间的内部状态迁移实验，因果追踪视觉-语言模型中图文联合安全判断的形成位置，发现局部输入证据在早期解码器层即发挥作用，而完整的安全判定在较晚层汇聚于最终token，且可通过线性读出预测。

    

    视觉-语言模型可能需要将图像与提示词结合，才能识别出二者单独都无法揭示的安全风险。这种联合安全判断在模型内部的哪个位置变得可读取？我们提出了SSU-Bench，这是一个由相互配对的安全与不安全图文组合构成的数据集，通过单项提示词编辑或带有标注目标区域的图像编辑构建而成。利用三个视觉-语言模型，我们在配对输入之间迁移内部状态，并测量由此产生的安全判定变化。在所有模型和两种类型的反事实干预中，在发生变化输入位置的干预在较早的解码器层中即有效，而在最终输入token上的干预则在较晚的层才生效。从其他样本估计出的方向也能产生类似的晚期层效应。最终token状态的线性读出也能预测模型自身的判定，包括错误的判断，而跨模型比较揭示了相似性……

    arXiv:2610.07514v1 Announce Type: new  Abstract: A vision-language model may need to combine an image with a prompt to recognize a safety risk that neither reveals alone. Where does this joint safety judgment become accessible inside the model? We introduce SSU-Bench, a dataset of matched safe and unsafe image-text combinations constructed using single-item prompt edits or image edits with annotated intended regions. Using three vision-language models, we transfer internal states between paired inputs and measure the resulting change in the safety verdict. Across models and both types of counterfactual, interventions at the changed input positions are effective in earlier decoder layers, while interventions at the final input token become effective later. Directions estimated from other examples produce similar late-layer effects. A linear readout of the final-token state also predicts the model's own verdict, including incorrect judgments, and cross-model comparisons reveal similariti
    
[^226]: 论信息诱导智能体的开放式信息搜寻

    On Open-Ended Information Seeking for Information Elicitation Agents

    [https://arxiv.org/abs/2610.07509](https://arxiv.org/abs/2610.07509)

    本研究通过在11个跨越不同家族和参数规模的大语言模型上进行的受控诱导模拟，揭示了不同LLM对信息价值的判断存在差异，且这些差异会显著塑造其序列化的开放式信息搜寻行为。

    

    信息诱导是一个开放式的信息搜寻问题，其中交互可以朝许多潜在有价值的方向发展，这要求诱导者在新信息不断涌现时持续决定应追求哪些信息。在智能体化的信息诱导中，这些决策可能被委托给基础模型，然而模型的选择如何塑造由此产生的信息搜寻行为仍缺乏充分研究。我们研究了关于信息价值的判断在不同大语言模型（LLM）之间如何变化，以及这些差异如何塑造序列化的信息搜寻过程。我们首先在跨越多个模型家族和参数规模的11个大语言模型上考察这些判断，实验使用了共享的信息集合和诱导目标。随后，我们开发了一个受控的信息诱导模拟环境，其中不同的模型面对相同的信息空间并使用相同的选择规则，从而将这些价值判断与问题生成和受访者行为隔离开来。

    arXiv:2610.07509v1 Announce Type: new  Abstract: Information elicitation is an open-ended information-seeking problem in which an interaction can unfold in many potentially valuable directions, requiring an elicitor to continually determine which information to pursue as new information emerges. In agentic elicitation, these decisions may be delegated to a foundation model, yet how model choice shapes the resulting information-seeking behavior remains understudied. We study how judgments about information value vary across LLMs and how these differences shape sequential information seeking. We first examine these judgments across 11 LLMs spanning multiple model families and parameter scales, using a shared set of information and elicitation objectives. We then develop a controlled elicitation simulation in which different models encounter the same information space and use the same selection rule, isolating these judgments from question generation and respondent behavior. Using this se
    
[^227]: Jarvis：一个面向多人对话的主动性语音代理

    Jarvis: A Proactive Speech Agent for Multi-Party Conversations

    [https://arxiv.org/abs/2610.07506](https://arxiv.org/abs/2610.07506)

    提出了Jarvis——一个能实时主动参与多人对话的语音代理，它在小组遗漏或误述事实且未自我纠正时进行干预，并提供了可量化评估的认知断裂基准数据集、基于小型开源模型且论断可溯源的主动式骨干系统，以及发言权交互技术三项关键贡献。

    

    语音代理目前是被动的、双向的：它们只在被问及时才说话，且一次只与一个人交流。我们探讨了一个语音代理如何才能反过来参与多人对话，并仅在能够提供帮助时才主动发言。我们提出了Jarvis，一个实时的主动性语音代理，能够以语音形式参与多人人类对话。基于事先共享的文档，Jarvis跟随讨论进程，并在小组遗漏或误述事实、且在几轮对话内未自行纠正时进行干预。我们做出三项贡献：一是基于“认知断裂”的问题设定，使主动干预变得可量化评估，具体实现为CHI-180-proactive——一个植入了已知信息缺口、错误和自我纠正的合成多人对话数据集；二是主动式骨干系统，利用小型开源权重模型并结合确定性检查，将每一句论断都锚定到源句子；三是用于获取发言权的交互技术。

    arXiv:2610.07506v1 Announce Type: cross  Abstract: Speech agents are reactive and dyadic: they speak when spoken to, and to one person at a time. We ask what it takes for a speech agent to instead take part in a conversation among several people and speak up only when it can help. We introduce Jarvis, a real-time proactive speech agent that audibly participates in multi-party human conversations. Grounded in a document shared beforehand, Jarvis follows the discussion and intervenes when the group misses or misstates a fact and does not correct itself within a few turns. We make three contributions: a problem setting based on epistemic breakdowns that makes proactive intervention measurable, realized as CHI-180-proactive, a synthetic multi-party dataset seeded with known gaps, errors, and self-corrections; a proactive backbone that harnesses a small, open-weight model with deterministic checks and grounds every claim in a source sentence; and interaction techniques for taking the floor 
    
[^228]: MARS：面向序列推荐的多分辨率自适应路由

    MARS: Multi-resolution Adaptive Routing for Sequential Recommendation

    [https://arxiv.org/abs/2610.07505](https://arxiv.org/abs/2610.07505)

    MARS通过将用户历史写入锚定不同时间半衰期的循环状态轨道，并利用稀疏路由读取器为每个种子按需选择相关时间分辨率来生成紧凑记忆，解决了缓存用户记忆中的“时间混叠”问题，从而显著提升序列推荐性能。

    

    长历史推荐系统通常将每个用户的历史压缩为一个紧凑的、与候选无关的记忆，该记忆被缓存并重复使用，以对大规模候选池进行评分。我们证明，真实的用户历史表现出多尺度的语义结构，短期意图、中期兴趣和长期偏好共存于同一序列中，而单一的整体缓存记忆无法均衡地保留这些尺度：线性探测对近期和中期内容的恢复效果远差于远期内容。我们将这种失败模式称为“时间混叠”。我们提出了MARS，一种多分辨率用户记忆，它将完整历史写入锚定于不同半衰期的循环状态轨道中，并采用稀疏路由读取器，通过为每个种子选择相关的时间分辨率来物化紧凑的种子记忆，从而保持固定大小的候选评分。MARS在三个公开数据集上优于强基线，且其增益随着……

    arXiv:2610.07505v1 Announce Type: new  Abstract: Long-history recommenders often compress each user's history into a compact, candidate-independent memory that is cached and reused to score large candidate pools. We show that real user histories exhibit multi-scale semantic structure, with short-lived intent, medium-term interests, and long-term preferences coexisting in one sequence, and that monolithic cached memories preserve these scales unevenly: linear probes recover recent and mid-range content far worse than long-range content. We call this failure mode \textit{temporal aliasing}. We propose \textbf{MARS}, a multi-resolution user memory that writes the full history into recurrent state tracks anchored to different half-lives, and a sparse routing reader that materializes compact seed memories by selecting the relevant temporal resolutions for each seed, preserving fixed-size candidate scoring. MARS outperforms strong baselines on three public datasets, with gains that grow with
    
[^229]: Muon 需要细粒度的谱整形吗？

    Does Muon Need Fine-Grained Spectral Shaping?

    [https://arxiv.org/abs/2610.07497](https://arxiv.org/abs/2610.07497)

    本文提出 BulkBoost 双频段谱重加权框架，表明 Muon 并不需要细粒度的谱整形，只需将奇异谱粗略地划分为噪声主体和高增益尖峰两个频段并进行重加权即可提升优化效果。

    

    Muon 将当前梯度与历史梯度结合为矩阵动量。对于 M=UΣV^T，理想化的极分解更新 Q=UV^T 会给每个奇异方向赋予相同的权重，我们将其称为“平坦谱形”。近期一些优化器用细粒度的谱映射取代这种平坦谱形，为每个方向赋予各自的增益。我们探究 Muon 更新到底需要多少这样的谱细节。我们的谱诊断显示，约 94%–97% 的测量奇异模态位于估计的噪声边缘之下，但它们整体上与参考梯度呈正相关对齐。我们提出了 BulkBoost，一个双频段谱重加权框架，包含固定秩与噪声校准两种变体。后者利用拆分小批量梯度差异，为 Muon 的 Nesterov 输入校准一个 Marchenko–Pastur 参考边缘，从而将边缘之下的主体部分与边缘之上的尖峰部分分离开来。两种变体都通过一次……（增加主体部分的相对权重）［摘要在此处被截断］

    arXiv:2610.07497v1 Announce Type: new  Abstract: Muon combines current and past gradients into matrix momentum. For $M=U\Sigma V^\top$, the idealized polar update $Q=UV^\top$ gives every singular direction the same weight. We refer to this as the flat profile. Several recent optimizers replace this flat profile with fine-grained spectral maps that give each direction its own gain. We ask how much of this spectral detail a Muon update needs. Our spectral diagnostics show that approximately $94$--$97\%$ of measured singular modes lie below an estimated noise edge, yet collectively align positively with a reference gradient.   We introduce BulkBoost, a two-band spectral reweighting framework with fixed-rank and noise-calibrated variants. The latter uses split-minibatch gradient differences to calibrate a Marchenko--Pastur reference edge for Muon's Nesterov input, separating the bulk below the edge from the spikes above it. Both variants increase the bulk's relative weight through one shar
    
[^230]: 谁来承担负担？多智能体强化学习中共享约束的责任学习

    Who Bears the Burden? Learning Responsibility for Shared Constraints in Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.07491](https://arxiv.org/abs/2610.07491)

    提出LiRA方法，通过优化社会福利来学习各智能体在共享拉格朗日乘子中的责任份额，从而解决多智能体强化学习中共享约束惩罚如何在不同智能体之间合理分配的问题。

    

    当多个智能体共享一个成本预算时，一个共同的拉格朗日乘子可以强制执行总体约束，但无法决定其惩罚应如何在各智能体之间分配。统一的惩罚方式忽略了智能体所牺牲奖励的异质性，而针对各智能体的独立乘子可能仍然依赖于相同的总体成本信号。我们提出了拉格朗日责任分配，该方法通过在有限训练范围内优化社会福利，来学习每个智能体在共同乘子中所占的份额。该乘子负责强制执行总体预算，而责任份额则在不修改原始奖励或约束的前提下重新分配其影响。对于满足标准正则条件的凸博弈，改变这些责任份额会诱导出一族平滑的归一化广义纳什均衡，其中活跃约束始终保持在预算水平，而社会福利则随份额变化。为了在收敛之前优化责任分配，我们推导了福利（摘要在此处截断）

    arXiv:2610.07491v1 Announce Type: cross  Abstract: When multiple agents share a cost budget, a common Lagrange multiplier can enforce the aggregate constraint but does not determine how its penalty should be allocated across agents. Uniform penalties ignore heterogeneity in the rewards agents sacrifice, while agent-specific multipliers may still rely on the same aggregate cost signal. We introduce Lagrangian Responsibility Allocation (LiRA), which learns each agent's share of a common multiplier by optimizing social welfare over a finite training horizon. The multiplier enforces the aggregate budget, while responsibility shares redistribute its influence without modifying the original rewards or constraints. For convex games under standard regularity conditions, varying these shares induces a smooth family of normalized generalized Nash equilibria in which active constraints remain at their budgets while welfare varies. To optimize responsibility before convergence, we derive a welfare
    
[^231]: 推陈出新：利用生成式人工智能增强“经典”文档自动化

    In With the Old: Enhancing 'Classical' Document Automation with Generative AI

    [https://arxiv.org/abs/2610.07480](https://arxiv.org/abs/2610.07480)

    本文探索了基于专家系统等符号方法的经典文档自动化与生成式AI如何相互增强，并通过初步实验证明大语言模型可用于识别和修复非专业人士撰写的法律文本中的问题。

    

    基于软件的法律援助系统已经利用了多种不同形式的知识表示和推理方法。本文探讨了植根于专家系统风格和其他符号方法的文档自动化服务如何能够有效地增强当前的生成式AI方法，并反过来被其增强。我们讨论了可能的益处和挑战，并报告了使用大语言模型来识别和修复非专业人士所写文本中问题的初步实验。

    arXiv:2610.07480v1 Announce Type: new  Abstract: Software-based legal assistance systems have leveraged many different forms of knowledge representation and reasoning. This article explores how document automation services rooted in expert system style and other symbolic approaches can usefully enhance and be enhanced by current generative AI approaches. We discuss the possible benefits and challenges, and report on preliminary experiments in using large language models to identify and fix issues in texts written by laypeople.
    
[^232]: 功耗能否约束隐蔽计算？模拟验证在AI治理中的局限性

    Can Power Draw Constrain Covert Compute? Limits of Analogue Verification for AI Governance

    [https://arxiv.org/abs/2610.07476](https://arxiv.org/abs/2610.07476)

    仅靠功耗等模拟测量手段无法有效约束隐蔽计算——最坏情况下隐藏计算量可达声明容量的116%，即使采用对抗性匹配能量策略也能隐藏至少41%的计算量，这说明AI治理协议不能仅依赖模拟验证机制。

    

    前沿AI条约或关于限制计算量的协议需要外部验证；外部审计员必须能够确认实际运行了多少计算量，以及各方是否遵守协议。模拟的、芯片外的测量手段（如功耗）为验证提供了一个信息通道。然而，当面对主动试图破坏审计的对手时，这些模拟通道能在多大程度上约束计算，目前尚不清楚。我们推导出了β的闭式表达式，其中β表示功耗轨迹无法排除的最大隐藏计算量占声明机器容量的比例。在NVIDIA A100 GPU上的测量表明，最坏情况下β = 1.16，而对抗性的匹配能量策略被证明至少可以隐藏β = 0.41的计算量。因此，仅靠模拟功耗测量对计算的约束力较弱。威胁模型所提供的额外限制条件，例如验证者能够重新执行……

    arXiv:2610.07476v1 Announce Type: cross  Abstract: Frontier AI treaties or agreements on limiting computation require external verification; an external auditor must be able to confirm how much computation actually ran and that parties are adhering to the agreement. Analogue, off-chip measurements such as power draw provide an information channel for verification. It is unknown how well these analogue channels can constrain computation against an adversary who actively tries to subvert the audit. We derive a closed form for $\beta$, the largest hidden computation a power trace cannot exclude, as a fraction of the declared machine capacity. Measurements on NVIDIA A100 GPUs constrain $\beta = 1.16$ in the worst case, while adversarial matched-energy strategies are shown to hide at least $\beta = 0.41$ of compute. Analogue power measurements alone therefore constrain compute weakly. Additional restrictions granted by the threat model, such as the ability of the verifier to re-execute the 
    
[^233]: PsyCIDRA：一个用于精神科访谈与诊断推理的双智能体框架

    PsyCIDRA: A Dual-Agent Framework for Psychiatric Interviewing and Diagnostic Reasoning

    [https://arxiv.org/abs/2610.07473](https://arxiv.org/abs/2610.07473)

    PsyCIDRA提出了一种双智能体框架，将自由形式的精神科访谈与诊断推理相结合，其访谈智能体借助工具笔记、专家技能和ICD-11参考指导问诊，诊断智能体报告假设及支持、冲突和缺失证据，在多个模型上取得了比直接提示更高的诊断一致性。

    

    大型语言模型在临床推理方面展现出前景，但精神科访谈需要引导一个不断演进的对话，其执行这种交互式评估的能力仍研究较少。我们提出了PsyCIDRA，一个将自由形式的精神科访谈与诊断推理相结合以供专家审阅的双智能体框架。其访谈者智能体使用工具来维护工作笔记、加载专家撰写的技能，并检索ICD-11参考资料以指导问诊。随后，诊断推理智能体接收完成的访谈记录，报告诊断假设以及支持性、冲突性和缺失的证据，当没有足够证据支持时会暂缓给出最终假设。我们使用由PsyCPG生成的患者档案，首先在模拟环境中评估PsyCIDRA。在53个评估案例上，跨四个模型，它比直接提示取得了更高的诊断一致性。在81个留出的模拟案例上，rank-1准确率为60（摘要在此处截断）。

    arXiv:2610.07473v1 Announce Type: new  Abstract: Large language models show promise in clinical reasoning, but psychiatric interviewing requires guiding an evolving conversation. Their ability to carry out this interactive assessment remains less studied. We present PsyCIDRA, a dual-agent framework linking free-form psychiatric interviewing with diagnostic reasoning for expert review. Its interviewer agent uses tools to maintain working notes, load expert-written skills, and retrieve ICD-11 references to guide inquiry. Its diagnostic reasoning agent then receives the completed interview transcript and reports hypotheses alongside supporting, conflicting, and missing evidence, withholding a final hypothesis when none is sufficiently supported. Using patient profiles generated with PsyCPG, we first evaluate PsyCIDRA in simulation. Across four models on 53 evaluation cases, it achieves higher diagnostic agreement than direct prompting. On 81 held-out simulated cases, rank-1 accuracy is 60
    
[^234]: 结构而非信念：组合半老虎机中基于LLM衍生协方差的相关汤普森采样

    Structure, Not Belief: Correlated Thompson Sampling from LLM-Derived Covariance in Combinatorial Semi-Bandits

    [https://arxiv.org/abs/2610.07470](https://arxiv.org/abs/2610.07470)

    提出一种对组合汤普森采样的最小改动方法，仅查询LLM一次将臂划分转化为相关协方差矩阵来引导探索，理论证明相比独立采样可获得有限时域内√(d/K)的遗憾改进，实验中遗憾降低19%。

    

    组合汤普森采样（CTS）为每个臂独立抽取后验样本，因此其探索动态忽略了臂之间的任何关联。我们研究了对此动态的一个最小改动：仅向大语言模型（LLM）查询一次以获得臂的划分，该划分通过聚类秩上的RBF核转化为正定相关矩阵Σ，每轮后验样本以协方差Σ抽取，而Beta后验仅依据真实奖励进行更新，因此LLM塑造的是采样器的移动方式，而非其信念内容。我们为理想化的高斯采样器给出了一个自洽的贝叶斯遗憾界，其信息增益分解为来自K聚类结构的K log T项和增长至d log T的岭回归项：相比独立采样的√(d/K)遗憾改进是一种有限时域的瞬态效应，仅在簇内相关性趋于1时才精确成立。该相关采样器将遗憾降低了19%。

    arXiv:2610.07470v1 Announce Type: cross  Abstract: Combinatorial Thompson sampling (CTS) draws independent posterior samples for every arm, so its exploration dynamics ignore any relation among arms. We study a minimal change to those dynamics: an LLM is queried once for a partition of the arms, the partition becomes a positive-definite correlation matrix $\Sigma$ through an RBF kernel on cluster ranks, and the per-round posterior sample is drawn with covariance $\Sigma$ while the Beta posteriors are updated from real rewards only, so the LLM shapes how the sampler moves, not what it believes. We give a self-contained Bayesian regret bound for the idealized Gaussian sampler whose information gain splits into a $K\log T$ term from the $K$-cluster structure and a ridge term that grows to $d\log T$: the $\sqrt{d/K}$ improvement over independent sampling is a finite-horizon transient, exact only as the within-cluster correlation tends to one. The correlated sampler reduces regret by 19% ov
    
[^235]: COMPASS：寻找语言模型中推理所在之处

    COMPASS: Finding Where Reasoning Lives in Language Models

    [https://arxiv.org/abs/2610.07469](https://arxiv.org/abs/2610.07469)

    COMPASS利用模型自身直接回答正确性这一简单信号，在推理时通过识别并引导特定注意力头激活来引出推理能力，无需预先定义推理特征，在多个数学基准上优于现有激活引导方法。

    

    显式地引出推理能够显著提升大语言模型（LLM）的性能。现有方法需要对推理进行预先定义的刻画，无论是通过思维链（CoT）提示设计、对比式CoT方向，还是通过稀疏自编码器（SAE）导出的推理特征。对于具有可验证答案的数学推理，我们证明一个简单得多的信号就足够了，即模型自身直接回答尝试的正确性。该信号产生了一个能够引出推理的潜在方向。该方向可以在大多数注意力头的激活中被解码出来，但其中只有一小部分注意力头可以被有效干预。我们提出了COMPASS，一种推理时引导方法，它使用logit空间的归因分数来识别这些注意力头，并沿正确性方向引导它们的激活，仅需要每个注意力头的激活统计信息。在三个模型家族和多个数学基准上，COMPASS的性能优于激活引导基线方法。

    arXiv:2610.07469v1 Announce Type: new  Abstract: Explicitly eliciting reasoning substantially improves LLM performance. Existing approaches require a predefined characterization of reasoning, whether through CoT prompt design, contrastive CoT directions, or via SAE derived reasoning features. For mathematical reasoning with verifiable answers, we show that a much simpler signal suffices, which is the correctness of the model's own direct answer attempts. This signal yields a latent direction that elicits reasoning. This direction is decodable within the activations of most attention heads, but only a small subset of them can be effectively intervened. We introduce COMPASS, an inference-time steering method that identifies these heads using a logit-space attribution score and steers their activations along the correctness direction, requiring only per-head activation statistics. Across three model families and multiple math benchmarks, COMPASS outperforms the activation-steering baselin
    
[^236]: ElasticFit：基于VLM推理与生成式自适应的适配感知3D物体插入

    ElasticFit: Fit-Aware 3D Object Insertion via VLM Reasoning and Generative Adaptation

    [https://arxiv.org/abs/2610.07460](https://arxiv.org/abs/2610.07460)

    提出ElasticFit框架，利用VLM从语言指令和场景观察中推断结构化适配线索（落地位置、占用体积、朝向及适应模式），并将其转化为显式3D约束，实现感知适配的3D物体插入。

    

    将物体插入现有3D场景中不仅仅是选择一个合理的位置：插入的物体还必须适应当地几何结构，同时保持语义意图和物理合理性。尽管近期的视觉-语言模型（VLM）和生成式模型能够实现语义推理和视觉内容创作，但当插入的物体必须适应受限的局部空间时，它们提供的3D定位和几何控制能力有限。我们提出了ElasticFit，这是一个用于适配感知物体插入的VLM引导框架，其核心是一种新颖的基于场景的表示方法。给定语言指令和渲染的场景观察，ElasticFit推断出结构化的适配线索，这些线索指定物体应在何处落地、应占据什么体积、应如何定向，以及其适应模式（刚性放置、均匀缩放或弹性适配）。这些线索将高层次的VLM推理转化为显式的3D约束……（摘要原文在此处被截断）

    arXiv:2610.07460v1 Announce Type: cross  Abstract: Inserting objects into existing 3D scenes requires more than selecting a plausible location:   the inserted object must also fit local geometry while preserving semantic intent and physical plausibility.   Although recent Vision-Language Models (VLMs) and generative models enable semantic reasoning and visual content creation, they offer limited 3D grounding and geometric control when an inserted object must fit into constrained local spaces.   We introduce \textbf{ElasticFit}, a VLM-guided framework for fit-aware object insertion centered on a novel scene-grounded representation.   Given a language instruction and rendered scene observations, ElasticFit infers structured fitting cues that specify where the object should be grounded, what volume it should occupy, how it should be oriented, and its adaptation mode (rigid placement, uniform scaling, or elastic fitting).   These cues convert high-level VLM reasoning into explicit 3D const
    
[^237]: 关于AI智能体的可审计声明

    Auditable Claims about AI Agents

    [https://arxiv.org/abs/2610.07459](https://arxiv.org/abs/2610.07459)

    提出AI智能体声明的可审计性标准——声明必须在事前明确其政策、范围、裁决记录及记录撰写者，并满足独立记录覆盖、授权绑定操作参数和超越完整性的完备性三项条件，才能被有效核查。

    

    组织会对其AI智能体做出各种声明：例如，每封外部邮件都经过人工审批、每个操作都留有日志、评估结果表明该智能体可以安全部署。欧盟《人工智能法案》第12条要求高风险系统允许自动记录事件，但并未说明哪些记录能够证实某项特定声明。我们的立场可以概括为一句话：要使关于智能体的声明能够被核查，该声明必须首先明确其政策、适用范围、能够证实它的记录，以及记录的撰写者。借鉴鉴证业务的前提条件，我们提出：如果在得出任何结论之前，这些要素及裁决规则已被固定，且相关记录可以获取，那么该声明就是可审计的。这将我们的可审计智能体框架中的“政策可核查性”维度从单一操作扩展到了声明层面。智能体场景增加了三个条件：由独立记录进行覆盖、授权与每个操作的参数相绑定，以及超越完整性的完备性。在明确的模型下，我们证明了（原文摘要在此处截断）……

    arXiv:2610.07459v1 Announce Type: new  Abstract: Organizations make claims about their AI agents: a person approves every external email, every action is logged, an evaluation shows the agent is safe to deploy. Article 12 of the EU AI Act requires high-risk systems to allow the automatic recording of events but does not say which records settle a given claim. The position is one sentence: to be checked, a claim about an agent must first name its policy, its scope, the records that would settle it, and who writes them. Adapting the preconditions of an assurance engagement, we call a claim auditable when these elements and a decision rule are fixed before any verdict and the records are obtainable. This extends the Policy Checkability dimension of our Auditable Agents framework from single actions to claims. Agents add three conditions: coverage by an independent record, authorization bound to each action's arguments, and completeness beyond integrity. Under an explicit model, we prove t
    
[^238]: AlignQuant：面向高效大语言模型生成的瓦片对齐混合精度量化

    AlignQuant: Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation

    [https://arxiv.org/abs/2610.07457](https://arxiv.org/abs/2610.07457)

    AlignQuant提出了一种以GPU兼容的二维权重瓦片作为精度分配、存储和执行公共单元的训练后混合精度量化方法，使大语言模型的压缩能够真正转化为实际推理加速。

    

    细粒度混合精度量化有望实现高效的大语言模型推理，但局部的精度选择可能与GPU规则的存储和计算单元发生冲突。这种精度边界的不匹配限制了压缩向实际加速的转化。我们提出AlignQuant，一种训练后量化方法，它使用GPU兼容的二维权重瓦片作为精度分配、紧凑存储和执行的公共单元。这种共享划分使精度分配能够跟随输出通道内的敏感度变化。联合预填充/解码校准方法在量化激活条件下，使用由语言模型损失梯度加权的投影输出扰动来评估精度降低的影响。相位归一化的评分在模型级权重存储预算下，优先为对任一阶段重要的瓦片分配更高精度。每个瓦片存储一种选定的表示，而相位专用内核则重用打包的（摘要在此处被截断）

    arXiv:2610.07457v1 Announce Type: cross  Abstract: Fine-grained mixed-precision quantization promises efficient large language model inference, but local precision choices can conflict with regular GPU storage and computation units. This precision-boundary mismatch limits the translation of compression into practical acceleration. We introduce AlignQuant, a post-training quantization method that uses GPU-compatible two-dimensional weight tiles as the common unit of precision allocation, compact storage, and execution. This shared partition lets precision follow sensitivity within output channels. Joint prefill/decode calibration scores precision reductions using projection-output perturbations weighted by language-model loss gradients under quantized activations. Phase-normalized scores prioritize higher precision for tiles important to either phase under a model-wide weight-storage budget. Each tile stores one selected representation, while phase-specialized kernels reuse the packed m
    
[^239]: 面向降低参与者负担的代价高效时序预测的主动特征获取

    Active Feature Acquisition for Cost-Efficient Temporal Prediction with Reduced Participant Burden

    [https://arxiv.org/abs/2610.07452](https://arxiv.org/abs/2610.07452)

    该论文提出纵向主动特征获取（LAFA）方法，通过学习一种策略在每个时间点仅选择性地采集最优的条目动态子集，从而在降低参与者负担、减少无应答与流失风险的同时，保持对心理病理结果的准确预测能力。

    

    arXiv:2610.07452v1 公告类型：cross 摘要：准确预测病理结果是心理学中的一个核心问题。为此，心理学家通常会收集密集的纵向数据。然而，在此类研究中，为了实现准确预测而获取大量变量的愿望，往往与最小化参与者负担的需求相冲突。每次测量获取更多变量可以带来更好的预测效果，但过多的测量会增加无应答和被试流失的风险。纵向主动特征获取（Longitudinal Active Feature Acquisition, LAFA）是一种解决这一难题的规范化方法。LAFA 不再要求参与者在每次测量时回答所有条目，而是生成一种策略，在每个时间点最优地选择需要获取的条目动态子集，同时保留对特定结果进行预测的能力。然而，现有的 LAFA 方法大多基于神经网络（NN），在实际应用中难以解释。在此……

    arXiv:2610.07452v1 Announce Type: cross  Abstract: Accurate forecasting of pathological outcomes is a central problem in psychology. To do so, psychologists often collect intensive longitudinal data. However, in such studies, the desire to acquire a large number of variables for the sake of accurate prediction is often counteracted by the need to minimize participant burden. Acquiring more variables per occasion can yield better predictions, but having too many acquisitions increase the risk of non-response and attrition. Longitudinal Active Feature Acquisition (LAFA) is a principled approach to resolve this conundrum. Instead of requiring responses to every item at every acquisition occasion, LAFA produces a policy that seeks to optimally select dynamic subsets of items to be acquired at each timepoint while preserving our ability to forecast a specific outcome. However, existing LAFA methods are mostly based on Neural Networks (NN) that are difficult to interpret in practice. In this
    
[^240]: AI监督何时有效？基于区块链可审计性的网络欺诈决策管理角色感知研究

    When Does AI Supervision Help? A Role-Aware Study of Network Fraud Decision Management with Blockchain Auditability

    [https://arxiv.org/abs/2610.07434](https://arxiv.org/abs/2610.07434)

    本文提出具有区块链可审计性的角色感知“决策者-监督者”框架，研究第二AI组件何时能改善网络欺诈决策，发现确定性硬门控可解决约89.994%的欺诈请求，而条件校准并不能带来一致可迁移的监督优势。

    

    第二个人工智能（AI）组件何时能改善初级网络欺诈决策，而不是增加运营负担？我们通过一个具有区块链可审计性的角色感知“决策者-监督者”框架来研究这一问题，评估了四种方向性配置，这些配置结合了集中式机器学习、通过联邦平均训练的联邦元模型，以及基础版或量化低秩适应（QLoRA）大语言模型变体。该分析使用非硬性欺诈性能、干预负担、条件校准、流量组合与审核容量敏感性、可靠性测试以及区块链生命周期控制，对仅初级决策与受监督决策进行比较。确定性硬门控解决了89.994%的欺诈请求，使非硬性群体成为主要的AI决策场景。条件验证校准未能产生一致可迁移的监督优势……

    arXiv:2610.07434v1 Announce Type: new  Abstract: When does a second artificial intelligence (AI) component improve a primary network-fraud decision rather than add operational burden? We study this question through a role-aware Decider-Supervisor (DS) framework with blockchain auditability, evaluating four directional configurations that combine centralised machine learning, a Federated Averaging (FedAvg)-trained federated meta-model, and Base or Quantized Low-Rank Adaptation (QLoRA) large language model variants. The analysis compares primary-only and supervised decisions using non-hard fraud performance, intervention burden, conditional calibration, traffic-mix and Review-capacity sensitivity, dependability tests, and blockchain lifecycle controls. The deterministic hard gate resolves 89.994% of fraudulent requests, leaving the non-hard population as the main AI decision setting. Conditional validation calibration does not produce a consistently transferable supervisory advantage on 
    
[^241]: 慢性踝关节不稳中基于参与者留出建模与参与者特异性更新的自适应步态生物反馈

    Adaptive Gait Biofeedback With Participant-Held-Out Modeling and Participant-Specific Updating in Chronic Ankle Instability

    [https://arxiv.org/abs/2610.07428](https://arxiv.org/abs/2610.07428)

    本研究针对慢性踝关节不稳提出了一种自适应步态生物反馈方法，通过参与者留出的LOSO交叉验证严格评估时间卷积分类器的模型性能，并在训练失败后进行参与者特异性模型更新以提升分类效果，同时验证了自适应干预对额状面踝关节角度改善的效果。

    

    自适应步态生物反馈可能有助于慢性踝关节不稳患者的重复练习，但其评估必须同时兼顾模型性能和人体反应。本研究在20名参与者中，使用参与者留出的留一受试者（LOSO）交叉验证方法，评估了一个时间卷积分类器在协议定义的、基于关节角度衍生的GOOD/BAD步态周期标签上的表现。自适应干预组的七名参与者在三周内完成了九次训练，每次训练均分析一次动作捕捉记录。在训练失败后进行更新的模型，在离线状态下与其父模型进行比较，比较所用的数据为用于候选选择的同次训练验证子集以及紧随其后第一次自适应训练的记录。研究在基线、后测和7天保持测试三个时间点，比较了自适应组与10名依次入组的对照组的额状面踝关节角度。在20个留出折中，平均折级受试者工作特征曲线下面积（摘要在此处被截断）

    arXiv:2610.07428v1 Announce Type: new  Abstract: Adaptive gait biofeedback may support repeated practice in chronic ankle instability, but its evaluation must address model performance and human response. We evaluated a temporal convolutional classifier on protocol-defined, angle-derived GOOD/BAD gait-cycle labels using participant-held-out leave-one-subject-out (LOSO) cross-validation in 20 participants. Seven participants in the adaptive-intervention group completed nine sessions over three weeks, with one motion-capture recording analyzed per session. Models updated after failed sessions were compared offline with their parent models on the same-session validation subset used for candidate selection and the first subsequent adaptive-session recording. Frontal-plane ankle angle was compared between the adaptive group and 10 sequentially enrolled controls at Baseline, Post, and 7-day Retention. Across 20 held-out folds, mean fold-level area under the receiver operating characteristic 
    
[^242]: 2D-FET-Bench：从空间推理到二维薄片上的场效应晶体管设计

    2d-fet-bench: from spatial reasoning to fet design on flakes

    [https://arxiv.org/abs/2610.07423](https://arxiv.org/abs/2610.07423)

    该论文推出了首个可执行基准 2D-FET-Bench V2，包含128个基于真实二维薄片轮廓的场效应晶体管版图设计任务，用以系统评估语言模型智能体将空间推理转化为器件版图构建的能力。

    

    arXiv:2610.07423v1 公告类型：新论文 摘要：在机械剥离的二维薄片上绘制场效应晶体管（FET）版图通常需要针对每个薄片手工绘制，即根据薄片在光学显微图像中的位置和轮廓来放置接触电极和栅极。据我们所知，目前尚无可执行的基准测试来检验语言模型智能体能否可靠地完成这种针对特定薄片的构建任务。我们推出了 2D-FET-Bench V2，这是一个包含128个版图任务的基准，任务基于显微成像提取的薄片轮廓构建，其中包括含孔洞的薄片和多薄片任务。每个任务提供文本形式的器件规格说明和轮廓坐标。智能体生成类型化的多边形和路径操作，并渲染为GDSII格式。一个确定性验证器用于检查几何和结构要求，另一个独立的完整性检查则验证所提供的轮廓保持不变。脚本化生成的参考版图通过了全部128个任务，证明每个任务都是可解的。我们评估了六个模型以及七种工作流和脚手架变体。

    arXiv:2610.07423v1 Announce Type: new  Abstract: Field-effect transistor (FET) layouts on exfoliated two-dimensional flakes are typically drawn by hand for each flake, placing contacts and gates to match its position and outline in optical micrographs. To our knowledge, no executable benchmark tests whether language-model agents can perform this flake-specific construction reliably. We introduce 2D-FET-Bench V2, a benchmark of 128 layout tasks built from microscopy-derived flake contours, including hole-containing flakes and multi-flake tasks. Each task supplies a textual device specification and contour coordinates. An agent generates typed polygon and path operations rendered to GDSII. A deterministic verifier checks geometric and structural requirements, and a separate integrity check verifies that the supplied contours remain unchanged. Scripted reference layouts pass all 128 tasks, showing that every task is solvable. We evaluate six models and seven workflow and scaffold variants
    
[^243]: 大语言模型的纵深防御：评估记忆门控对抗激活诱发与记忆诱发的谄媚行为

    Defense-in-Depth for LLMs: Evaluating Memory Gates Against Activation-Induced and Memory-Induced Sycophancy

    [https://arxiv.org/abs/2610.07403](https://arxiv.org/abs/2610.07403)

    该论文提出了一个 2×2 纵深防御框架，将内部激活引导与外部记忆处理分离，并引入包括新型“路由门控”在内的五种记忆防御配置，在 MemSyco-Bench 上系统评估并防御大语言模型中由激活诱发与记忆诱发的谄媚行为。

    

    长期记忆使大语言模型（LLM）能够在多次交互中维持个性化的上下文，但检索到的用户历史可能诱发“记忆型谄媚”（memory-induced sycophancy），导致模型偏向已存储的用户信念而忽视客观证据。现有防御方法主要作用于检索到的上下文层面，且很少与内部行为偏差进行联合评估。我们提出了一个 2×2 的纵深防御框架，将内部激活引导与外部记忆处理相分离。我们从 100 个配对提示中提取谄媚引导方向，并在 MemSyco-Bench 基准上（全部 1,550 个条目均给出答案；防御条件则在固定的 250 个条目子样本上进行评判），使用三个 LLM 评判器，对四个开源权重模型在 10 个引导系数和五种记忆防御配置下进行评估。五种配置中有三种是全新的（重写每条记忆、对每条记忆进行保留/重写/丢弃决策的路由门控，以及丢弃全部记忆）；另一种……（摘要原文在此处截断）

    arXiv:2610.07403v1 Announce Type: new  Abstract: Long-term memory allows Large Language Models (LLMs) to maintain personalized context across interactions, but retrieved user history can induce memory-induced sycophancy, causing models to favor stored user beliefs over objective evidence. Existing defenses primarily operate on retrieved context and are rarely evaluated jointly with internal behavioral bias. We introduce a $2 \times 2$ defense-in-depth framework separating internal activation steering from external memory handling. We extract sycophancy steering directions from 100 paired prompts and evaluate four open-weight models across 10 steering coefficients and five memory-defense configurations on MemSyco-Bench (answers for all 1,550 items; defense conditions judged on a fixed 250-item subsample), with three LLM judges. Three of the five configurations are new (rewriting every memory, a Router Gate that keeps, rewrites, or drops each memory, and dropping all memory); the other t
    
[^244]: 稀疏自编码器中作为自然梯度流的推理与学习

    Inference and learning in sparse autoencoders as natural gradient flow

    [https://arxiv.org/abs/2610.07389](https://arxiv.org/abs/2610.07389)

    该论文将稀疏自编码器的推理与字典学习统一为共享变分自由能上的自然梯度流，并提出无编码器的稀疏编码模型BeFOND，通过循环解释抵消机制减少重叠特征间的干扰、利用Fisher预条件化加速稀有特征学习，从而显著提升字典恢复与稀有特征检测能力。

    

    稀疏自编码器被广泛用于揭示神经网络中的可解释特征，然而当特征相互重叠或激活频率较低时，可靠的特征恢复仍然困难。这些挑战既涉及推断哪些特征解释了输入，也涉及学习表示这些特征的字典。在此，我们将推理和字典学习统一为共享变分自由能上的自然梯度流。我们将该框架实例化为BeFOND，一种无编码器的稀疏编码模型，具有闭式形式的推理和学习动力学。我们展示了循环的“解释抵消”机制如何减少重叠特征之间的干扰，而Fisher预条件化可以补偿稀有特征的缓慢学习。在合成数据上，BeFOND改进了字典恢复和稀有特征检测，且随着叠加程度增加，其相对于摊销基线的优势不断扩大。在语言模型激活上，它提升了单特征概念检测的性能。

    arXiv:2610.07389v1 Announce Type: cross  Abstract: Sparse autoencoders are widely used to uncover interpretable features in neural networks, yet reliable recovery remains difficult when features overlap or activate infrequently. These challenges involve both inferring which features explain an input and learning the dictionary that represents them. Here, we unify inference and dictionary learning as natural-gradient flows on a shared variational free energy. We instantiate this framework as BeFOND, an encoder-free sparse coding model with closed-form inference and learning dynamics. We show how recurrent explaining away reduces interference between overlapping features, while Fisher preconditioning can compensate for the slow learning of rare features. On synthetic data, BeFOND improves dictionary recovery and rare-feature detection, with a growing advantage over amortized baselines as superposition increases. On language-model activations, it improves single-feature concept detection 
    
[^245]: DeepAJM：面向不规则采样数据的深度关联联合模型

    DeepAJM: Deep Association Joint Model for Irregularly Sampled data

    [https://arxiv.org/abs/2610.07388](https://arxiv.org/abs/2610.07388)

    提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。

    

    联合模型同时建模纵向结局与生存结局，利用患者纵向轨迹中的模式来改进生存结局的预测。然而，经典的参数化联合模型依赖于固定的参数假设，在模型误设和样本量较小的情况下容易产生偏差。我们提出了一种深度联合模型 DeepAJM，它不需要任何参数假设，同时保留了部分可解释的、针对每个纵向结局的关联结构。该联合模型采用编码器-解码器（序列到序列）架构来学习患者时变协变量轨迹中的潜在结构。模型通过一个学习得到的可解释关联结构将纵向过程与生存过程联系起来，其中解码器输出的每个纵向结果在贡献于（生存模型的）风险评分之前，会先由基线协变量进行重新调制……

    arXiv:2610.07388v1 Announce Type: cross  Abstract: Joint Models simultaneously model longitudinal and survival outcomes, leveraging patterns in patients' longitudinal trajectory to improve the prediction of survival outcomes. The classical parametric joint models, however, rely on fixed parametric assumptions, making them susceptible to bias under model misspecification and smaller sample sizes. We propose a deep joint model, DeepAJM, that does not require any parametric assumptions, while retaining a partially interpretable, per-longitudinal-outcome association structure. The joint model uses an encoder-decoder (sequence-to-sequence) architecture to learn the latent structure in patients' time-varying covariate trajectories. The model links the longitudinal processes to the survival processes through a learned interpretable association structure, in which each longitudinal output from the decoder gets remodulated by baseline covariates before it contributes to the risk scores from the
    
[^246]: WildMatch：面向野生动物重识别的弱监督图像匹配器自适应

    WildMatch: Weakly Supervised Image Matcher Adaptation for Wildlife Re-Identification

    [https://arxiv.org/abs/2610.07384](https://arxiv.org/abs/2610.07384)

    提出WildMatch，仅利用个体身份标签（无需关键点或几何对应标注）对预训练关键点匹配器进行弱监督适配，以解决相机陷阱图像中野生动物个体重识别的难题。

    

    从相机陷阱图像中进行动物个体重识别是非侵入式野生动物监测的核心实例检索问题：给定一张查询图像，需要从已知动物的参考集合中检索出正确的个体。这要求计算机视觉模型能够识别毛皮、皮肤或其他视觉标记中独特的局部图案。现有方法要么将问题当作分类任务来学习全局嵌入，这需要每个个体有大量标注图像，同时在很大程度上忽略了局部证据；要么直接采用现成的、与领域无关的通用图像匹配器。尽管这类匹配器在大型且多样化的图像集合上进行了预训练，但由于可用的野生动物数据集规模小且缺乏对应级别的标注，将其适配到野生动物图像上具有挑战性。我们研究了预训练关键点匹配器的弱监督适配，仅使用个体身份标签，而无需关键点级别或几何对应的真值标注。我们挖掘……

    arXiv:2610.07384v1 Announce Type: cross  Abstract: Individual animal re-identification from camera-trap imagery is an instance retrieval problem central to non-invasive wildlife monitoring: a query image must retrieve the correct individual from a reference set of known animals. This requires computer vision models to recognize distinctive local patterns in fur, skin, or other visual markings. Current approaches either learn global embeddings as a classification problem, requiring many labeled images per individual while largely ignoring local evidence, or apply off-the-shelf, domain-agnostic image matchers. Although such matchers are pretrained on large and diverse image collections, adapting them to wildlife imagery is challenging because available datasets are small and lack correspondence-level annotations. We study weakly supervised adaptation of a pretrained keypoint matcher using only identity labels, without keypoint-level or geometric correspondence ground truth. We mine infor
    
[^247]: MemCo：面向将LLM智能体泛化到未见环境的以记忆为中心的协作框架

    MemCo: Memory-Centric Collaboration for Generalizing LLM Agents to Unseen Environments

    [https://arxiv.org/abs/2610.07376](https://arxiv.org/abs/2610.07376)

    提出以记忆为中心的协作框架MemCo，通过维护互补的本地与全局记忆空间来平衡记忆检索的粒度问题，从而将LLM智能体泛化到未见过的交互式环境中。

    

    大型语言模型（LLM）智能体越来越多地在交互式环境中运行，它们需要通过观察、行动和反馈来做出序列决策。尽管记忆可以帮助智能体重用经验，但现有工作将记忆设计为孤立的，收集足够的轨迹来填充记忆的成本高昂。现有的共享记忆方法通过跨任务和环境汇集情节记忆来缓解孤立经验的问题。然而，共享记忆的检索面临粒度方面的挑战：检索到的记忆可能过于具体而无法保留当前的情境定位，或过于粗略而无法支持下一步行动。在本工作中，我们提出了MemCo，一个以记忆为中心的协作框架，用于将LLM智能体泛化到未见过的交互式环境中。它维护互补的本地与全局记忆空间，在本地保留环境特定的细节，同时促进从本地轨迹中归纳出的可迁移工作流……（原文摘要在此处截断）

    arXiv:2610.07376v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly operate in interactive environments, where they need to make sequential decisions through observation, action, and feedback. Although memory can help agents reuse experience, existing work designs memory in isolation, where collecting enough trajectories to populate it is expensive. Existing shared-memory approaches mitigate isolated experience by pooling episodic memories across tasks and environments. However, retrieving shared memory is challenged by the granularity, where retrieved memories can be either too specific to preserve current grounding or too coarse to support the next action. In this work, we propose MemCo, a memory-centric collaboration framework for generalizing LLM agents to unseen interactive environments. It maintains complementary local and global memory spaces, preserving environment-specific details locally while promoting transferable workflows induced from local tra
    
[^248]: 评估整个技术栈，而非单个层级：针对智能体动作的确定性门控与LLM门控是否独立失效？

    Evaluate the Stack, Not the Layer: Do Deterministic and LLM Gates for Agent Actions Fail Independently?

    [https://arxiv.org/abs/2610.07359](https://arxiv.org/abs/2610.07359)

    研究发现，用于智能体动作防护的多个LLM裁判门控之间错误高度相关（堆叠两个裁判仅相当于约1.2-1.4个独立层，远低于2层的理想值），而确定性规则层与LLM裁判的组合则接近独立失效（约1.8-2.1层），因此多门控防御技术栈应作为整体评估而非逐层评估。

    

    针对智能体工具调用的运行时门控系统在堆叠部署时的假设是各层错误相互独立、可相乘叠加。我们在来自三个语料库的1,119个带标签的智能体动作上对该假设进行了检验（不涉及自适应攻击者）。该技术栈包含一个确定性规则层和四个LLM裁判，其中三个裁判在每次调用时都记录了所服务模型的重新收集版本。我们将每个技术栈量化为若干“乘法等效层级数”（n_mult），其下限为完全耦合时的取值。在严格的漏检定义下（升级至人工处理被计为未拦截），任意两个裁判的组合仅相当于约1.2至1.4层（φ中位数+0.430，6对组合中6对显著，下限1.02至1.17）。规则层与单个裁判的组合则为1.86至2.09层（φ中位数+0.014，4对组合中0对显著，下限1.01至1.09）。在主要定义下（升级至人工处理被计为已拦截），两组区间分别为1.21至1.57和1.80至2.13。在合并数据上各组区间可相互区分，点估计在每个语料库上均可分离，而第三方……

    arXiv:2610.07359v1 Announce Type: new  Abstract: Runtime gates for agent tool calls are stacked on the assumption that their errors multiply. We test it on 1,119 labelled agent actions from three corpora, without an adaptive adversary. The stack has one deterministic rule layer and four LLM judges, three of them re-collected with the served model recorded on every call. We read each stack as a number of multiplication-equivalent layers, n_mult, with its floor under perfect coupling. Under the STRICT miss definition (escalation to a human scored as not stopped), any two judges compose to about 1.2 to 1.4 layers ({\phi} median +0.430, 6 of 6 pairs significant, floors 1.02 to 1.17). The rule layer plus one judge composes to 1.86 to 2.09 layers ({\phi} median +0.014, 0 of 4 significant, floors 1.01 to 1.09). Under PRIMARY (escalation scored as caught) the bands are 1.21 to 1.57 and 1.80 to 2.13. Intervals separate on the pooled data, point estimates split on each corpus, and a third-vendor
    
[^249]: 一套面向连贯多图SysML模型的验证数据集与基准

    A Validated Dataset and Benchmark for Coherent Multi-Diagram SysML Models

    [https://arxiv.org/abs/2610.07356](https://arxiv.org/abs/2610.07356)

    该论文提出了SEMAADB——一个包含3,000个工程情境、15,000张经过一致性和有效渲染验证的SysML多视图图的大规模数据集与基准，用于评估大语言模型生成连贯多图系统建模的能力。

    

    系统工程师使用多种图来描述系统的结构和行为。工程师会共同创建这些图，以确保它们使用相同的元素并保持彼此一致。大型语言模型能够以文本或代码的形式生成图，这使得自动创建系统图成为可能。然而，它们生成连贯图集的能力尚不清楚，且现有的数据集和基准无法大规模地直接衡量这一能力。我们提出了SEMAADB（Systems Engineering Modeling Assistant with AI Dataset and Benchmark，AI系统建模助手数据集与基准），这是一个包含3,000个工程情境和15,000张图的数据集。每个情境包含五个相互关联的SysML视图：需求图、块定义图、活动图、状态机图和序列图。在这里，视图是呈现系统某一个方面的图。我们对这些图集进行了一致性和有效渲染方面的检查。此外，其中100个情境的图集还经过了人工验证。

    arXiv:2610.07356v1 Announce Type: cross  Abstract: Systems engineers use several diagrams to describe the structure and behavior of systems. Engineers create these diagrams together to make sure that they use the same elements and remain consistent with one another. Large language models can generate diagrams as text or code, which makes it possible to create system diagrams automatically. However, their ability to generate coherent sets of diagrams is not well understood, and existing datasets and benchmarks do not directly measure this ability at scale. We introduce SEMAADB (Systems Engineering Modeling Assistant with AI Dataset and Benchmark), a dataset of 3,000 engineering contexts and 15,000 diagrams. Each context contains five connected SysML views: Requirement, Block Definition, Activity, State Machine, and Sequence. Here, a view is a diagram that presents one aspect of a system. We checked the diagram sets for consistency and valid rendering. A set of 100 contexts is also human
    
[^250]: 评估LLM路由中的升级信号：目标、对照与五种自欺方式

    Evaluating Escalation Signals for LLM Routing: Targets, Controls, and Five Ways to Fool Yourself

    [https://arxiv.org/abs/2610.07354](https://arxiv.org/abs/2610.07354)

    本文验证了语义熵可作为LLM路由中判断是否升级到大模型的廉价有效信号（GSM8K上AUROC达0.871、准确率最多提升九个百分点），并提出了一套防止被“仅基于问题难度的简单规则”等虚假信号误导的严格评估检查方法。

    

    决定何时将查询从小语言模型升级到更大的模型，需要一种廉价的信号，能在调用大模型之前预测升级是否有帮助。语义熵最初是为检测幻觉而提出的，是一个天然的候选方案：它衡量模型采样出的多个答案在语义上的分歧程度，高分歧往往意味着答案不可靠。我们在三个基准测试和两个模型家族上对其进行了测试。在GSM8K上，使用大小相差约十二倍的小/大模型对，语义熵能够可靠地识别出小模型的错误（AUROC 0.871），并在相同成本下将路由准确率相比随机升级最多提升九个百分点。而此前在一个合成基准上看似很强的结果被证明具有误导性：一个仅基于问题难度、完全不涉及模型的简单规则，几乎与语义熵的表现完全一致。本文的主要贡献是一套能够发现此类问题的检验方法。

    arXiv:2610.07354v1 Announce Type: new  Abstract: Deciding when to escalate a query from a small language model to a larger one requires a cheap signal that predicts, before the large model is called, whether escalating would help. Semantic entropy, originally developed to detect hallucinations, is a natural candidate: it measures how much a model's sampled answers disagree in meaning, and high disagreement often signals an unreliable answer. We test it across three benchmarks and two model families. On GSM8K, with a small/large pair about twelve times apart in size, semantic entropy reliably distinguishes the small model's mistakes (AUROC 0.871) and improves routed accuracy over random escalation by up to nine points at matched cost. An earlier strong-looking result on a synthetic benchmark proved misleading: a simple rule based only on question difficulty, with no model involved, matched semantic entropy almost exactly. This paper's main contribution is a set of checks that catch this
    
[^251]: 轨迹检索投机解码：模型自身的历史何时有用？

    Trajectory-Retrieval Speculative Decoding: When Does a Model's Own History Help?

    [https://arxiv.org/abs/2610.07350](https://arxiv.org/abs/2610.07350)

    提出轨迹局部自适应检索（TLAR）方法，通过从模型自身推理轨迹中检索复用近似匹配的续写内容，并结合自适应检索策略与投机解码，在相同验证预算下提升token接受率、降低长链式思维推理的解码成本。

    

    长链式思维推理在增加序列解码成本的同时，也不断积累着潜在可复用的续写历史。我们研究了这些历史何时能提供有用的草稿，并与现有起草模型形成互补。受控的来源对比实验揭示了轨迹特定的复用现象，由此启发我们提出了“轨迹局部自适应检索”（TLAR）方法。TLAR从当前轨迹中检索近似匹配的续写内容，并利用最近的验证结果来自适应地调整检索激活与候选宽度。TLAR将检索到的续写内容与模型生成的草稿合并至共享的候选树中，并通过精确验证保持目标模型的输出分布。在代码调试、数学和开放式写作等任务上，我们的评估将来源复用、增量接受率与执行成本联系起来。在相同的验证预算下，将TLAR与强大的检索基线相结合可以提高token接受率。

    arXiv:2610.07350v1 Announce Type: new  Abstract: Long chain-of-thought reasoning increases sequential decoding cost while creating a growing history of potentially reusable continuations. We investigate when this history supplies useful drafts and complements an existing drafter. Controlled source comparisons reveal trajectory-specific reuse, motivating our method Trajectory-Local Adaptive Retrieval (TLAR). TLAR retrieves approximately matched continuations from the current trajectory and uses recent verification outcomes to adapt retrieval activation and candidate width. TLAR combines retrieved continuations with model-generated drafts in a shared candidate tree, preserving the target model's output distribution through exact verification. Across code debugging, mathematics, and open-ended writing, our evaluation connects source reuse, incremental acceptance, and execution cost. Combining TLAR with strong retrieval baselines improves token acceptance under matched verification budgets
    
[^252]: RELACE：面向长程语言智能体的基于回溯似然的动作信用估计

    RELACE: retrospective likelihood-based action credit estimation for long-horizon language agents

    [https://arxiv.org/abs/2610.07349](https://arxiv.org/abs/2610.07349)

    RELACE 提出了一种无 critic 的信用估计框架，通过比较动作在原始上下文与结果增强上下文下的教师强制似然差异，生成轨迹归一化的回溯因子，从而为长程语言智能体提供更精确的动作级信用分配。

    

    组相对策略优化通过从 rollout 组中估计优势来避免使用单独的 critic。然而对于多轮智能体，轨迹级监督提供的是粗粒度且带有噪声的信用信号：终端奖励无法定位错误发生的具体动作，并且可能将有效的动作与错误动作一同惩罚。组中组策略优化及其后续方法通过状态条件化的比较来细化监督，但其信用估计仍然对下游决策和最终结果较为敏感。我们提出了 RELACE，这是一个无 critic 的框架，将回溯式动作评估与状态条件化的优势估计相结合。RELACE 通过在动作的原始上下文和结果增强上下文两种情形下进行教师强制似然打分，来评估已执行的动作。对这两种似然进行比较，可以得到一个经轨迹归一化的回溯因子，该因子能够捕捉动作结果依赖性的变化（摘要至此截断）。

    arXiv:2610.07349v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) avoids a separate critic by estimating advantages from rollout groups. For multi-turn agents, however, trajectory-level supervision provides coarse, noisy credit: terminal rewards do not locate errors and can penalize useful actions alongside mistakes. Group-in-Group Policy Optimization (GiGPO) and subsequent methods refine supervision through state-conditioned comparisons, but their credit estimates remain sensitive to downstream decisions and outcomes. We introduce RELACE, Retrospective Likelihood-based Action, a critic-free framework that integrates retrospective action assessment with state-conditioned advantage estimation. RELACE evaluates executed actions through teacher-forced likelihood scoring under both their original contexts and outcome-augmented contexts. Comparing these likelihoods yields a trajectory-normalized retrospective factor that captures outcome-dependent changes in actio
    
[^253]: 阶梯式MoE：具有可配置推理复杂度的分段级路由

    Stepped MoE: Segment-Level Routing with Configurable Inference Complexity

    [https://arxiv.org/abs/2610.07348](https://arxiv.org/abs/2610.07348)

    本文提出阶梯式MoE统一框架，将弹性结构与稀疏门控架构相结合，通过分段级路由使模型能够同时适应不同的部署约束和任务需求，实现推理时对精度-效率权衡的细粒度控制。

    

    训练大型语言模型（LLM）非常耗费资源，而将模型适配到具有不同计算约束的多样化部署场景仍然具有挑战性。虽然弹性架构能够实现灵活的模型部署，稀疏激活模型允许输入自适应的计算，但现有方法将这些维度独立处理。此外，面向设备端边缘推理的模型需要符合服务设备的内存和计算限制。在本文中，我们引入了一个统一框架，将弹性结构与稀疏门控架构相结合，创建能够同时适应部署约束和任务需求的模型。我们的方法采用一个以上下文和目标效率规格为条件的模型骨干网络，从而在推理时实现对精度-效率权衡的细粒度控制。模型学习激活与任务相关的部分……

    arXiv:2610.07348v1 Announce Type: cross  Abstract: Training large language models (LLMs) is resource-intensive, and adapting them for diverse deployment scenarios with varying computational constraints remains challenging. While elastic architectures enable flexible model deployment and sparsely activated models allow input-adaptive computation, existing approaches treat these dimensions independently. Moreover, models catered towards on-device edge inference need to conform to the memory and compute limitations of the serving devices. In this paper, we introduce a unified framework that combines elastic structures with sparsely gated architectures to create models that adapt simultaneously to both deployment constraints and task requirements. Our approach employs a model backbone that conditions on both the context and target efficiency specifications, enabling fine-grained control over the accuracy-efficiency trade-off at inference time. The model learns to activate task-relevant par
    
[^254]: 理据引导的策略优化：通过自适应理据支架学习推理

    Rationale-Guided Policy Optimization: Learning to Reason with Adaptive Rationale Scaffolding

    [https://arxiv.org/abs/2610.07342](https://arxiv.org/abs/2610.07342)

    提出了理据引导的策略优化（RGPO）框架，根据模型当前能力自适应地利用真实理据信息作为支架，以缓解强化学习中的奖励稀疏问题，同时保留模型的探索自由。

    

    同策略强化学习已成为提升大型语言模型推理能力的核心范式。然而，其有效性常常受到奖励稀疏性的限制：当模型无法为困难问题找到正确的求解轨迹时，优化过程难以获得有用的信号，可能陷入停滞。现有方法通过引入离策略示范、专家轨迹或模型生成的解答来缓解这一问题，但它们通常要求辅助数据与强化学习任务的格式相匹配，往往需要依赖从更强的模型中进行拒绝采样来获得合适的训练轨迹。我们提出了理据引导的策略优化（RGPO），这是一个根据模型当前能力自适应地利用真实理据信息、同时保留模型探索自由的框架。该方法并非将参考解答视为固定的模仿目标……

    arXiv:2610.07342v1 Announce Type: new  Abstract: On-policy reinforcement learning has become a central paradigm for improving the reasoning abilities of large language models. However, its effectiveness is often limited by reward sparsity: when a model fails to discover correct trajectories for difficult problems, the optimization process receives little useful signal and may stagnate. Existing approaches mitigate this issue by incorporating off-policy demonstrations, expert traces, or model-generated solutions, but they typically require the auxiliary data to match the format of the reinforcement-learning task, often relying on rejection sampling from stronger models to obtain suitable training trajectories. We introduce Rationale-Guided Policy Optimization (RGPO), a framework that adaptively leverages ground-truth rationale information according to the model's current capability while preserving its freedom to explore. Rather than treating reference solutions as fixed imitation targe
    
[^255]: CausalBind：面向蛋白质-分子虚拟筛选的因果建模与学习

    CausalBind: Causal Modeling and Learning for Protein-Molecule Virtual Screening

    [https://arxiv.org/abs/2610.07340](https://arxiv.org/abs/2610.07340)

    该论文提出CausalBind，通过因果建模识别并利用蛋白质-分子结合中稀疏的跨模态局部相互作用模式（如氢键、疏水接触、盐桥），从而克服传统密集整体对齐方法的局限，提升虚拟筛选向新靶点泛化的能力。

    

    蛋白质-分子虚拟筛选日益被构建为共享嵌入空间中的表示学习问题。现有方法依赖于密集的整体对齐，将不变的结合决定因素与干扰性相关性纠缠在一起，从而限制了向新靶点的迁移能力。已有研究指出，蛋白质-分子系统中的结合涉及稀疏的跨模态相互作用：结合由一个小的接触界面和少数决定性的局部相互作用（如氢键、疏水接触和盐桥）所支配，而非蛋白质和分子的全局结构。我们假设，发现并利用这些稀疏相互作用模式对于超越训练数据的泛化至关重要，因为这些模式是可复用的，并有望在不同场景下提升性能。在本文中，我们旨在识别并利用稀疏相互作用模式，并验证这一假设。由于训练数据（原文摘要在此处截断）……

    arXiv:2610.07340v1 Announce Type: cross  Abstract: Protein-molecule virtual screening is increasingly cast as a problem of representation learning in a shared embedding space. Existing methods rely on dense holistic alignment, entangling invariant binding determinants with nuisance correlations and limiting transfer to new targets. It has been noted that binding in protein-molecule systems involves sparse cross-modality interactions: binding is governed by a small contact interface and a few decisive local interactions (e.g., hydrogen bonds, hydrophobic contacts, and salt bridges) rather than the global structures of the protein and molecule. We hypothesize that uncovering and leveraging sparse interaction patterns is critical for generalization beyond the training data, as these patterns are reusable and expected to improve performance across different scenarios. In this paper, we aim to identify and leverage sparse interaction patterns, and verify our hypothesis. Since the training d
    
[^256]: TC3-VQA：基于条令的战术战斗伤员救护视觉问答数据集

    A doctrine-grounded visual question answering dataset for Tactical Combat Casualty Care

    [https://arxiv.org/abs/2610.07339](https://arxiv.org/abs/2610.07339)

    本文提出TC3-VQA数据集，利用公开教学与实战视频和权威条令文档构建了1,860个将视觉证据与可追溯战伤救护条令关联的问答样本，为支持战术战斗伤员救护的视觉-语言模型开发提供监督数据。

    

    战术战斗伤员救护（TC3）要求救援人员将伤情和救治干预的视觉观察与既定的临床指导联系起来。开发支持这一过程的视觉-语言模型，需要能够将可见证据与可追溯条令相关联的监督数据。我们提出了TC3-VQA，这是一个由公开的教学与实战TC3视频以及权威TC3文档构建的数据集。它包含跨越11个概念的581个条目，共1,860个问题，涵盖干预识别、条令、临床推理、操作流程指导，以及视觉信息不足时的拒答。基于条令的答案保留了逐字原文段落和字符偏移量。数据集构建结合了视觉标注、段落检索、蕴含检查以及跨模型家族的验证。设备框、解剖标签、时间段和来源元数据随问答对一同提供。自动化审计与评分（摘要原文在此处被截断）

    arXiv:2610.07339v1 Announce Type: cross  Abstract: Tactical Combat Casualty Care (TC3) requires responders to connect visual observations of injuries and interventions with established clinical guidance. Developing vision-language models to support this process requires supervision that links visible evidence to traceable doctrine. We present TC3-VQA, a dataset constructed from public instructional and field TC3 videos and authoritative TC3 documents. It contains 581 items spanning 11 concepts, with 1,860 questions covering intervention recognition, doctrine, clinical reasoning, procedural guidance, and refusal when visual information is insufficient. Doctrine-based answers preserve verbatim source passages and character offsets. Construction combines visual annotation, passage retrieval, entailment checks, and verification across model families. Equipment boxes, anatomical labels, temporal segments, and source metadata accompany the question-answer pairs. Automated audits and ratings 
    
[^257]: Logbook：超长时音频事件理解

    Logbook: Extremely Long-form Audio Event Understanding

    [https://arxiv.org/abs/2610.07338](https://arxiv.org/abs/2610.07338)

    该论文提出了面向小时级至六天超长音频的事件理解基准 Logbook，要求系统对连续音频进行无缝隙分割并为每段生成事件标签与描述，发现最佳系统仍不及人类、过度分割普遍存在，且端到端系统通常优于级联系统但性能随上下文变长而下降。

    

    现有的音频基准测试都围绕短小、预先分割的音频片段构建，这将模型设计限制在简短输入或固定词表上。为了弥合这一差距，我们提出了 Logbook，一个面向小时级音频理解的基准测试，其录音时长从十分钟到六天不等。给定一段连续的音频录音和一个事件标签词表，系统必须预测出无缝隙的分割结果，并为每个片段提供事件标签和描述。我们比较了52个端到端和级联系统，并对微调、上下文长度和推理预算进行了消融实验。我们发现该任务是可以解决的，但表现最好的系统仍低于人类参考水平。此外，过度分割现象普遍存在，微调可以部分缓解这一问题。最后，端到端系统通常优于级联系统，但其性能会随着上下文变长而下降。

    arXiv:2610.07338v1 Announce Type: cross  Abstract: Audio benchmarks are built around short, pre-segmented clips, limiting model design to brief inputs or fixed vocabularies. To close this gap, we introduce Logbook, a benchmark for hour-scale audio understanding, with recordings ranging from ten minutes to six days. Given a continuous audio recording and an event label vocabulary, a system must predict a gap-free segmentation with an event label and a description per segment. We compare 52 systems, end-to-end and cascaded, and ablate fine-tuning, context length, and reasoning budget. We find the task tractable, though the best systems remain below the human reference. Also, over-segmentation is pervasive, and fine-tuning partially mitigates it. Finally, end-to-end are often better than cascaded systems, but degrades with longer context.
    
[^258]: 面向成本感知LLM智能体在长时程决策中的选择性批判机制

    Selective Critique for Cost-Aware LLM Agents in Long-Horizon Decision Making

    [https://arxiv.org/abs/2610.07335](https://arxiv.org/abs/2610.07335)

    提出SAG框架，利用基于动作歧义信号（全局熵与top-2边际）的轻量级免训练门控机制，智能地选择在何时调用外部批判，从而在提升LLM智能体长时程决策可靠性的同时大幅降低token消耗和延迟。

    

    提升大型语言模型（LLM）智能体在长时程决策中的可靠性仍然是一个关键挑战。当作为自主智能体部署并与复杂环境交互时，早期的错误会在轨迹中传播并引发级联式失败。近期的方法通过引入外部批判或深思机制来提高可靠性，但在每一步都调用这些机制会大幅增加token消耗和延迟，限制了实际部署。我们提出了SAG（带有门控批判的自改进智能体，Self-improving Agent with Gated critique），这是一个成本感知框架，它将批判调用表述为长时程交互过程中的逐步决策问题。SAG引入了一种轻量级的、无需训练的门控机制，通过在可行动作集合上计算的动作级歧义信号——全局熵和局部top-2边际——来估计批判的效用。从决策理论的角度来看，该机制近似……

    arXiv:2610.07335v1 Announce Type: cross  Abstract: Improving the reliability of large language model (LLM) agents in long-horizon decision-making remains a key challenge. When deployed as autonomous agents interacting with complex environments, early mistakes can propagate through trajectories and cause cascading failures. Recent approaches improve reliability by incorporating external critique or deliberation, but invoking these mechanisms at every step substantially increases token consumption and latency, limiting practical deployment. We propose SAG (Self-improving Agent with Gated critique), a cost-aware framework that formulates critique invocation as a step-wise decision problem during long-horizon interaction. SAG introduces a lightweight, training-free gating mechanism that estimates the utility of critique using action-level ambiguity signals--global entropy and local top-2 margin--computed over admissible actions. From a decision-theoretic perspective, this mechanism approxi
    
[^259]: 面向分布式MoE训练的内存高效专家路由

    Memory-Efficient Expert Routing for Distributed MoE Training

    [https://arxiv.org/abs/2610.07333](https://arxiv.org/abs/2610.07333)

    提出RelayMoE，一种基于环形结构的MoE执行模型，通过让专家权重或token在环中循环流动并在专家路由与token路由间动态选择，避免了完整top-k扩展分发缓冲区的构建，显著提升了分布式MoE训练的内存效率。

    

    随着混合专家（MoE）模型扩展到数百个专家以及更高的top-k路由，分布式训练中的内存效率成为关键瓶颈。峰值内存主要由MoE模块而非注意力机制主导：MoE分发流水线中的每个中间缓冲区都会因top-k路由而单独扩大规模。标准的all-to-all分发器在单个集合通信步骤中发送所有被路由的token，需要一次性构建完整的top-k扩展缓冲区。在这项工作中，我们提出了RelayMoE，一种基于环形结构的MoE执行模型，当专家权重或token在环中循环流动时进行本地计算，从而避免了完整的top-k扩展分发缓冲区。RelayMoE根据通信量在专家路由与token路由之间进行选择，并将传输与计算重叠执行。环形结构天然支持反向传播期间内存高效的MoE重计算：每一跳都会重建专家中间结果，并用它们计算梯度。

    arXiv:2610.07333v1 Announce Type: cross  Abstract: As Mixture-of-Experts (MoE) models scale toward hundreds of experts and higher top-$k$ routing, memory efficiency in distributed training becomes a critical bottleneck. Peak memory is dominated by the MoE block, not attention: every intermediate buffer in the MoE dispatch pipeline is individually scaled by top-k routing. The standard all-to-all dispatcher sends all routed tokens in a single collective step, requiring the full top-$k$-expanded buffer to be constructed at once. In this work, we propose RelayMoE, a ring-based MoE execution model that computes locally as expert weights or tokens circulate, avoiding full top-$k$-expanded dispatch buffers. RelayMoE selects between expert and token routing according to communication volume and overlaps transfers with computation. The ring structure naturally supports memory-efficient MoE recomputation during backward: each hop reconstructs expert intermediates, uses them to compute gradients,
    
[^260]: 面向时间序列基础模型的尺度不变训练

    Scale-Invariant Training for Time Series Foundation Models

    [https://arxiv.org/abs/2610.07324](https://arxiv.org/abs/2610.07324)

    论文揭示了对仿射缩放（如ReVIN）进行逆变换会使各序列梯度被乘以b^p、使序列尺度成为隐含的重要性权重并导致高尺度序列主导训练的“尺度污染”问题，并证明直接在缩放后的目标上计算损失即可实现尺度不变的训练。

    

    时间序列基础模型是在跨越多种形态和领域的大量时间序列数据集集合上训练的。这种训练设置使模型接触到尺度（即数值的典型大小）差异可能非常显著的序列。诸如可逆实例归一化（ReVIN）等仿射缩放方法会对模型输入进行缩放，并在计算损失之前逆转该变换。我们证明，相对于在缩放后的目标上计算损失，这种逆变换会将每个序列的梯度乘以 $b^p$，其中 $b$ 是缩放分母（例如标准差），$p$ 是损失次数。我们将这种现象称为尺度污染训练，因为每个序列的尺度由此变成了重要性权重，导致高尺度序列主导训练过程。对于任何尺度等变的缩放器以及任何 $p$ 次齐次的残差损失（包括 MSE、MAE 和分位数损失），我们证明在缩放后的目标上计算损失

    arXiv:2610.07324v1 Announce Type: cross  Abstract: Time series foundation models (TSFMs) are trained on large collections of time series datasets that span various morphologies and domains. This setting exposes models to series whose scales -- typical magnitudes of their values -- can differ substantially. Affine scaling methods such as Reversible Instance Normalization (ReVIN) scale model inputs and reverse the transform before computing the loss. We show that this inversion multiplies each series' gradient by $b^p$ relative to loss on scaled targets, where $b$ is the scaling denominator (e.g., standard deviation) and $p$ is the loss degree. We call this scale-contaminated training (ScaleCon), because the scale of each series consequently becomes an importance weight, causing high-scale series to dominate training. For any scale-equivariant scaler and residual loss that is homogeneous of degree $p$, including MSE, MAE, and Quantile Loss, we prove that computing loss on scaled targets 
    
[^261]: 面向神经符号人工智能的基于规则的语言

    Rule-Based Languages for Neurosymbolic AI

    [https://arxiv.org/abs/2610.07313](https://arxiv.org/abs/2610.07313)

    本文沿语义、表达能力、神经集成和求值机制四个维度，综述了神经符号AI中的Datalog、答案集和概率逻辑程序三类基于规则的语言，分析了50多个系统并提供了一个将应用场景映射到所需特性的决策矩阵。

    

    逻辑编程日益被用作神经符号AI系统的符号组件。我们从四个维度对这一领域中的主要基于规则的语言进行了综述，即 Datalog、答案集和概率逻辑程序，这四个维度分别是：语义、表达能力、神经集成方式以及求值机制。我们分析了 50 多个近期的系统与应用，比较了这些形式化方法在四个研究领域中的使用情况：数据库与编程语言、机器学习、视觉以及机器人学。我们提供了一个将应用场景映射到所需特性的决策矩阵，并在最后概述了开放性问题。

    arXiv:2610.07313v1 Announce Type: new  Abstract: Logic programming is increasingly used as the symbolic component of neurosymbolic AI systems. We survey the main rule-based languages in this setting, namely Datalog, answer set, and probabilistic logic programs, along four axes: semantics, expressiveness, neural integration, and evaluation mechanism. We analyse over 50 recent systems and applications, comparing formalism usage across four research areas: databases and programming languages, machine learning, vision, and robotics. We provide a decision matrix mapping application scenarios to required features and close by outlining open problems.
    
[^262]: 理解并缓解推理时对智能体记忆的过度依赖

    Understanding and Mitigating Inference-Time Overreliance Using Agentic Memory

    [https://arxiv.org/abs/2610.07311](https://arxiv.org/abs/2610.07311)

    发现智能体记忆在查询与过往经验仅部分重叠时会因证据无法完全迁移而误导推理，并提出即插即用框架MEMTRIM，通过写入时索引证据、读取时控制复用并移除重复或冲突的记忆，无需重新训练即可缓解LLM智能体的记忆过度依赖问题。

    

    智能体记忆使大语言模型智能体能够复用过往经验，然而即使检索到的记忆是无害的、被正确存储且被恰当检索的，它们也可能扭曲推理。我们研究了这种失败模式，并将其称为“记忆过度依赖”。在多个基准测试和记忆架构上，我们发现当过往经验能够迁移到当前任务时，记忆是有用的；但当只有部分证据能够迁移时，记忆可能变得具有误导性。这种失败在查询与记忆部分重叠时最为严重，这一模式在通过改变重叠证据数量的对照实验中得到进一步证实。受此发现启发，我们提出了MEMTRIM，一个即插即用框架，它在写入时对记忆证据建立索引，并在读取时控制其复用。MEMTRIM在保留有用的记忆特有信息的同时，去除重复或冲突的证据，无需重新训练，并且适用于基于嵌入的和结构化的记忆系统。（注：原摘要在此处截断，实验部分内容未提供）

    arXiv:2610.07311v1 Announce Type: new  Abstract: Agentic memory allows LLM agents to reuse past experience, yet retrieved memories can also distort inference even when they are benign, correctly stored, and appropriately retrieved. We study this failure mode, which we call memory over-reliance. Across benchmarks and memory architectures, we find that memory is useful when past experience transfers to the current task, but can become misleading when only part of the evidence transfers. Failures are strongest under partial query-memory overlap, a pattern further confirmed by controlled experiments thatvary the amount of overlapping evidence. Motivated by this finding, we propose MEMTRIM, a plug-and-play framework that indexes memory evidence at write time and controls its reuse at read time. MEMTRIM removes repeated or conflicting evidence while preserving useful memory-specific information, requires no retraining, and applies to both embedding-based and structured memory systems.Experim
    
[^263]: 从沙箱到执行：面向关键基础设施的置信度合格威胁情报

    From Sandbox to Enforcement: Confidence-Qualified Threat Intelligence for Critical Infrastructure

    [https://arxiv.org/abs/2610.07310](https://arxiv.org/abs/2610.07310)

    提出了CG-CTI流水线，将沙箱输出转化为带明确置信度标注的威胁情报，并通过知识图谱跨源佐证实现“只有高置信情报才能触发自动化处置”的分级把关机制，为关键基础设施防御提供从数据采集到自动执行的可靠闭环。

    

    防御关键基础设施的安全运营中心和国家级应急响应团队收集了大量威胁数据，却难以将其转化为可操作的情报。恶意软件沙箱能够产生详细的行为证据，但往往以一份庞大、未经排序且未标明置信度的报告形式呈现。我们提出了CG-CTI，一个可实际运行的流水线，它将实时沙箱输出（CAPEv2）转换为STIX 2.1格式，在知识图谱中与其他关键基础设施传感器的数据进行关联，并为每个情报对象附加明确的置信度状态，该状态基于来源追溯、跨源佐证和观测持续性得出。此置信度状态对自动化行动进行把关：只有经过佐证的情报才有资格进行自动化处置，而置信度较低的对象则被分流给分析师审核或仅作为背景信息保留。随后，一个有据可依的语言模型阶段对置信度合格的证据进行叙述，其中每条陈述都会引用相应的支持依据……（摘要在此处截断）

    arXiv:2610.07310v1 Announce Type: cross  Abstract: Security operations centres and national incident-response teams defending critical infrastructure collect abundant threat data yet struggle to turn it into actionable intelligence. A malware sandbox produces detailed behavioural evidence, but as a large, unranked report whose confidence is unstated. We present CG-CTI, an operational pipeline that converts live sandbox output (CAPEv2) into STIX 2.1, correlates it in a knowledge graph with other critical-infrastructure sensors, and attaches to every intelligence object an explicit confidence status derived from provenance, cross-source corroboration, and observation durability. This status gates automated action: only corroborated intelligence is eligible for automated enforcement, while lower-confidence objects are routed to analyst review or kept as context. A grounded language-model stage then narrates the confidence-qualified evidence, where each statement either cites a supporting 
    
[^264]: 《正确的记忆，错误的语境：验证长期智能体记忆中的检索可采性》

    The Right Memory in the Wrong Context: Verifying Retrieval Admissibility in Long-Term Agent Memory

    [https://arxiv.org/abs/2610.07309](https://arxiv.org/abs/2610.07309)

    该论文提出了一个检索可采性验证框架，通过将记忆-查询对标注为“可采纳、不可采纳或未决”三种状态，并在提示词暴露层面追踪记忆ID及其与目标级泄露的关联，从而检测长期记忆智能体在错误语境下检索出“正确的记忆”这一安全隐患。

    

    长期记忆智能体可能检索到与当前请求相关但不可采纳的信息，原因在于这些信息属于其他主体、违反了政策，或反映了不兼容的生命周期状态。召回率与最终答案准确率无法揭示这一问题：一条检索路径可能因缺失必要证据而显得安全，而正确的答案也可能是在经历过不可采纳的提示词暴露之后给出的。我们提出了一个检索可采性验证框架，该框架为每个记忆-查询对赋予三种状态之一（可采纳、不可采纳或未决），在匹配的必要证据召回率下比较不同检索路径并为未决情况设定界限，并通过提示词暴露追踪记忆ID，同时将暴露情况与目标级泄露相关联。我们在相互独立、未合并的总体上评估该框架的各个阶段。对两个公开长期记忆基准 RHELM 和 MemOps 的冻结排名进行的事后 top-20 重分析共覆盖 3,767 个查询。所有已发布的锚点均落在真实……（摘要原文在此处截断）

    arXiv:2610.07309v1 Announce Type: new  Abstract: Long-term-memory agents can retrieve relevant information that is inadmissible for the current request because it belongs to another principal, violates policy, or reflects an incompatible lifecycle state. Recall and final-answer accuracy do not reveal this: a route can appear safe by missing required evidence, while a correct answer may follow inadmissible prompt exposure. We introduce a retrieval-admissibility verification framework that assigns each memory-query pair one of three statuses (admissible, inadmissible, or unresolved), compares routes at matched required-evidence recall with bounds for unresolved cases, and tracks memory IDs through prompt exposure while linking exposure to target-level disclosure. We evaluate its stages on separate, non-pooled populations. A post-hoc top-20 reanalysis of frozen rankings from two public long-term-memory benchmarks, RHELM and MemOps, covers 3,767 queries. All released anchors lie within tru
    
[^265]: Polar：基于大语言模型的真实世界网络证据合成，用于威胁优先级排序与缓解

    Polar: LLM-Powered Synthesis of Real-World Cyber Evidence for Prioritization and Mitigation

    [https://arxiv.org/abs/2610.07298](https://arxiv.org/abs/2610.07298)

    POLAR是一个由大语言模型驱动的框架，将分散在厂商通告、漏洞数据库和威胁情报中的真实世界网络证据合成为以威胁为中心的评估，通过结合严重性推断与按时间排序的利用信号实现威胁优先级排序，并关联权威修复知识以支持按紧急程度组织的缓解行动。

    

    网络威胁分析日益依赖于分布在厂商通告、漏洞数据库和威胁情报来源中的证据。要将这些碎片化的观测转化为及时的决策，需要模型将技术严重性与不断演变的利用证据以及可用的防御行动联系起来。我们提出了POLAR，一个由大语言模型驱动的框架，用于将真实世界的网络证据合成为以威胁为中心的评估，以支持优先级排序与缓解。POLAR首先解开相互重叠的事件，并将每个威胁锚定在带来源链接的证据之上。在优先级排序方面，它从网络证据中推断严重性指标，并将由此产生的评估与按时间排序的利用信号相结合，以估计近期被利用的可能性。在缓解方面，它将合成的威胁数据与权威的修复知识相关联，并根据威胁的紧急程度和操作约束来组织适用的行动。

    arXiv:2610.07298v1 Announce Type: cross  Abstract: Cyber threat analysis increasingly depends on evidence distributed across vendor advisories, vulnerability databases, and threat intelligence sources. Turning these fragmented observations into timely decisions requires models to connect technical severity with evolving exploitation evidence and available defensive actions. We present POLAR, an LLM-powered framework for synthesizing real-world cyber evidence into threat-centric assessments for prioritization and mitigation. POLAR first disentangles overlapping incidents and grounds each threat in source-linked evidence. For prioritization, it infers severity metrics from cyber evidence and combines the resulting assessment with temporally ordered exploitation signals to estimate near-term exploitation likelihood. For mitigation, it links the synthesized threat data to authoritative remediation knowledge and organizes applicable actions according to threat urgency and operational constr
    
[^266]: 在心流中抓住开发者：Google规模下的低延迟智能体程序修复

    Catching Developers in the Flow: Low-Latency Agentic Program Repair at Google Scale

    [https://arxiv.org/abs/2610.07289](https://arxiv.org/abs/2610.07289)

    本文提出部署于Google的AI智能体FlowAgent，通过ReAct风格的生成-验证循环与弃权过滤器，在持续集成的提交前阶段以低延迟实时自动修复测试失败，使开发者无需切换上下文即可在心流中获得高质量修复建议。

    

    程序故障的手动修复对软件开发者来说既耗时又具有干扰性，尤其是在提交前（pre-submit）阶段，此时测试失败发生在持续集成系统中。尽管自动程序修复（Automated Program Repair）借助大语言模型已取得显著进展，但现有的最先进技术主要聚焦于提交后（post-submit）的工作流程，以离线方式运行，缺乏在开发者切换上下文之前实时辅助其工作流程所需的低延迟能力。在本文中，我们介绍了FlowAgent，这是部署于Google的一个AI智能体，用于在持续集成系统内的提交前外循环工作流程中自动修复测试失败。FlowAgent集成了Google的内部开发者工具Critique和Cider，采用ReAct风格的生成与验证循环，以及严格的执行前和执行后弃权（abstention）过滤器，以确保在严格条件下提供高质量的建议。

    arXiv:2610.07289v1 Announce Type: cross  Abstract: Manual repair of program failures is time-consuming and disruptive for software developers, particularly during the pre-submit phase where test failures occur within continuous integration systems. While Automated Program Repair has seen significant advancement through Large Language Models, existing state-of-the-art techniques primarily focus on post-submit workflows, operating offline without the low-latency requirements necessary to assist developers in real-time within their flow before they switch context.   In this paper, we introduce FlowAgent, an AI agent deployed at Google to automatically repair test failures in the pre-submit outer-loop workflow inside continuous integration systems. Integrated into Google's internal developer tools, Critique and Cider,FlowAgent utilizes a ReAct-style generate-and-validate loop, as well as rigorous pre-execution and post-execution abstention filters to ensure high-quality suggestions under s
    
[^267]: FlexiFlow：基于多臂老虎机的机器学习工作流模型切换

    FlexiFlow: Bandit-based Model Switching in ML Workflows

    [https://arxiv.org/abs/2610.07286](https://arxiv.org/abs/2610.07286)

    FlexiFlow是一个基于多臂老虎机的动态模型切换数据流系统，综合考虑模型准确率、运行时间和断言通过概率，在当前模型表现不佳时自动切换到更优模型，可将机器学习工作流准确性提升高达23%。

    

    模型优化有助于提高机器学习工作流的推理性能和准确性。然而，依赖单一模型对所有数据批次执行推理往往无法最大化准确性，从而无法最大化整体性能。在许多情况下，当主模型表现不佳时，替代模型在特定数据子集上可能表现更好。我们在真实机器学习工作流上的实验表明，切换模型可将工作流准确性提升高达23%。然而，现有系统缺乏根据性能自适应切换模型的能力，迫使用户手动依次测试模型。我们提出了FlexiFlow，这是一个数据流系统，能够在当前模型表现出低准确性时动态切换到替代模型。FlexiFlow采用一种新颖的多臂老虎机方法学习对模型进行排序，该方法综合考虑了模型运行时间、通过用户定义断言的概率以及机器学习工作流的计算结构。

    arXiv:2610.07286v1 Announce Type: cross  Abstract: Model optimizations help improve inference performance and accuracy of ML workflows. However, relying on a single model to perform inference across all data batches often fails to maximize accuracy and thus overall performance. In many cases, alternate models could perform better on specific subsets of data where a primary model underperforms. Our experiments with real ML workflows indeed show that switching models improves workflow accuracy by up to 23%. Yet, current systems lack the ability to adaptively switch between models based on performance, forcing users to manually test models in sequence. We present FlexiFlow, a dataflow system that dynamically switches between alternate models when the current model exhibits low accuracy. FlexiFlow learns to rank models using a novel multi-armed bandit approach that accounts for model runtimes, probability of passing user-defined assertions, and the computational structure of the ML workflo
    
[^268]: SAFESHIELD：面向小语言模型部署时安全性的决策组织框架

    SAFESHIELD: A Decision-Organization Framework for Deployment-Time Safety of Small Language Models

    [https://arxiv.org/abs/2610.07276](https://arxiv.org/abs/2610.07276)

    本文提出SAFESHIELD框架，将小语言模型的部署时安全形式化为决策组织问题，通过组织准入、路由、证据和发布四种安全决策职责，并将决策记录于可审计的决策轨迹中，实现了安全决策的显式组织、协调与审计。

    

    语言模型的部署时安全通常通过运行时护栏（如输入审核、路由、检索验证和输出过滤）来实现。现有的部署框架为这些功能提供了日益强大的机制，但对于这些框架所产生的安全决策应如何被显式地组织、协调和审计，所提供的指导却十分有限。我们将部署时安全形式化为一个决策组织问题，包含两个要素：面向职责的安全决策分解，以及各决策之间的显式协调。我们将这一形式化实例化为SAFESHIELD——一个面向小语言模型的部署时安全系统，它组织了四种反复出现的决策职责（准入、路由、证据和发布），并将已确定的决策记录在可审计的决策轨迹中。我们通过机制级实验、阶段级聚合消融实验以及受控协调实验（原文在此处截断）对SAFESHIELD进行了评估。

    arXiv:2610.07276v1 Announce Type: cross  Abstract: Deployment-time safety of language models is commonly implemented through runtime guardrails such as input moderation, routing, retrieval verification, and output filtering. Existing deployment frameworks provide increasingly capable mechanisms for these functions, but offer limited guidance on how the safety decisions they produce should be explicitly organized, coordinated, and audited. We formulate deployment-time safety as a decision-organization problem with two elements: responsibility-oriented decomposition of safety decisions and explicit coordination among them. We instantiate this formulation in SAFESHIELD, a deployment-time safety system for small language models that organizes four recurring decision responsibilities (admission, routing, evidence, and release) and records committed decisions in auditable Decision Traces. We evaluate SAFESHIELD through mechanism-level experiments, aggregate stage ablations, controlled coordi
    
[^269]: 智能体评估的信任层

    A Trust Layer for Agent Evaluation

    [https://arxiv.org/abs/2610.07274](https://arxiv.org/abs/2610.07274)

    该论文提出“智能体评估信任层”，一个附加式后验框架，通过验证评分逻辑支持、可追溯计算、完成声明一致性和重复执行稳定性这四个属性，来判断智能体在基准测试中获得的分数是否可信。

    

    确定性的基准测试分数只能表明某个智能体获得了得分，却无法说明该分数是否名副其实、是否被如实报告，或者能否在重复运行中保持不变。我们提出了智能体评估信任层，这是一个可附加的后验框架，它在每条已记录的分数旁报告该分数是否值得相信。该框架验证四个属性：结果是否得到基准测试自身评分逻辑的支持；通过的答案是否通过可追溯的计算获得；智能体的完成声明是否与实际发生的情况相符；以及结果在重复执行下是否保持稳定。前三个属性仅使用已保存的工件进行验证；第四个属性则需要重新运行智能体。模型判断仅在多数投票机制下用于对证据进行标注；所有结论均遵循确定性规则，且从不修改已记录的分数。将该框架应用于 Agents' Last Exam 中 108 个任务上的五种智能体配置，结果显示每个模型都存在通过测试却无可追溯计算的运行情况（其比率各不相同……）

    arXiv:2610.07274v1 Announce Type: new  Abstract: Deterministic benchmark scores show that an agent received credit, but not whether that credit was earned, reported honestly, or would hold on a second run. We introduce a Trust Layer for Agent Evaluation, an additive post-hoc framework that reports, beside each recorded score, whether it should be believed. It verifies four properties: whether the result is supported by the benchmark's own grading logic, whether a passing answer was earned through traceable computation, whether the agent's completion claim matches what occurred, and whether the result is stable under repeated execution. The first three use only saved artifacts; the fourth re-runs the agent. Model judgments only label evidence under majority voting; all verdicts follow deterministic rules and never modify the recorded score. Applied to five agent configurations on 108 tasks from Agents' Last Exam, every model shows passing runs with no traceable computation (at rates var
    
[^270]: 模型是否真的使用了该特征？在大语言模型中区分“操控”与“机制”

    Does the Model Use the Feature? Separating Steering from Mechanism in LLMs

    [https://arxiv.org/abs/2610.07270](https://arxiv.org/abs/2610.07270)

    该论文提出仅在自然输入观测值范围内评估特征的植入、移除与下游拯救测试，发现特征的操控能力与模型实际使用它的程度可以显著分离，从而纠正了将“可操控”直接等同于“机制”的误判。

    

    当大语言模型中的内部特征能够追踪某个概念，且对其进行操纵会改变相关行为时，人们常将其解读为模型机制。然而，操控可能将特征值推到远超其自然范围之外，此时其效果未必反映模型自身的计算。本文审视了这一推断，并提出一种经验性契约，其测试方法在自然输入上实际观测到的取值范围内对特征进行评估。其中一种测试将特征值从展示某行为的输入复制到匹配但不展示该行为的输入（“植入”），或进行反向操作（“移除”）；另一种测试是在对上游进行编辑之后恢复该特征（“下游拯救”）。植入衡量该特征对该行为的充分程度，而移除与下游拯救衡量模型对该特征的实际使用程度。将该方法应用于三类表示后，这两种强度呈现出明显的分离。已发表的未知实体潜变量对“知识拒答”行为具有很强的操控作用，但植入观测到的特征值……（原文摘要在此处截断）

    arXiv:2610.07270v1 Announce Type: new  Abstract: Internal features in LLMs are often interpreted as mechanisms when they track a concept and their manipulation changes a related behavior. Yet steering can push a feature far outside its natural range, where its effects need not reflect the model's own computation. We examine this inference and propose an empirical contract whose tests evaluate features at values observed on natural inputs. One test copies a feature's value from an input that shows a behavior into a matched input that does not (installation) or the reverse (removal); the other restores the feature after an upstream edit (downstream rescue). Installation measures how far the feature suffices for the behavior; removal and downstream rescue measure how much the model uses it. Applied to three kinds of representations, the two strengths separate sharply. The published unknown-entity latent strongly steers knowledge abstention, yet installing observed values from either publi
    
[^271]: 验证并行编程智能体的协调性：NP-Bench 基准与调度规划器

    Verifying Coordination in Parallel Coding Agents: NP-Bench and a Scheduling Planner

    [https://arxiv.org/abs/2610.07261](https://arxiv.org/abs/2610.07261)

    该论文将并行 LLM 编程智能体之间的协调问题重新表述为预先调度问题——划分互不重叠的工作范围并按生产者->消费者图安排合并顺序，据此构建了 Nerveplane 规划器，并提出了通过真实 git 合并验证集成效果的三臂基准测试 NP-Bench。

    

    一个编程智能体团队可能逐个看每个智能体都表现正常，但作为团队却会失败：每个智能体都通过了自己的测试，而合并后的结果却是坏的，单智能体评估永远发现不了这一点。当团队在同一个代码库上并行运行多个 LLM 编程智能体时，智能体之间会发生冲突：两个智能体重写了同一个函数，一个智能体按照队友刚刚更改的接口契约进行编码，工作完成后集成却失败了。大多数协调工具都是反应式的（监视到冲突后再发出警告），但在智能体的运行速度下，警告到达时浪费性的编辑早已发生。我们将该问题重新表述为调度问题：预先获取每个工作项所声明的范围，将工作划分为互不重叠的范围，并沿着生产者->消费者图对合并顺序进行安排。我们将该规划器构建进 Nerveplane，并用 NP-Bench 对其进行评估；NP-Bench 是一个基于真实环境的三臂基准测试（无协调；反应式检测；主动式规划），通过真实的 git 合并来验证集成效果，两者均在确定性（环境中进行）……

    arXiv:2610.07261v1 Announce Type: new  Abstract: A team of coding agents can look fine agent by agent yet fail as a team: each passes its own tests while the merged result is broken, and single-agent evaluation never catches it. As teams run several LLM coding agents in parallel on one codebase, the agents collide: two rewrite the same function, one codes against a contract a teammate just changed, and integration fails after the work is done. Most coordination tools react (watch for a conflict, then warn), but at agent speed the warning arrives after the wasted edit. We recast the problem as scheduling: take each work item's declared scope, partition the work into disjoint scopes, and order merges along the producer->consumer graph, all up front. We build this planner into Nerveplane and evaluate it with NP-Bench, an environment-grounded three-arm benchmark (no coordination; reactive detection; proactive planning) that verifies integration off a real git merge, both in a deterministic
    
[^272]: 谱系感知的内存治理：面向企业AI智能体隐私保护列级访问控制的派生门控框架

    Lineage-Aware Memory Governance: A Derivation-Gated Framework for Privacy-Preserving Column-Level Access Control in Enterprise AI Agents

    [https://arxiv.org/abs/2610.07258](https://arxiv.org/abs/2610.07258)

    该论文提出分析内存单元（AMU），通过为每个缓存结果附加完整的派生谱系图并实施列级权限门控检索，从设计上保证企业AI智能体不会命中由请求者无权限的敏感列派生而来的缓存结果，同时解决部门间同名KPI计算逻辑冲突的问题。

    

    共享内存存储的企业AI智能体面临两个尚未解决的风险：敏感数据可能通过请求者本无权限推导出的“合法计算结果”发生泄露，以及各部门可能通过相互冲突的逻辑静默地计算同名关键绩效指标（KPI）。现有的智能体内存系统（如MemGPT、Zep、A-MEM）基于内容、所有权和角色来控制检索，而非基于派生关系，因而无法拦截缓存中嵌入了被禁止访问列的洞察结果。我们提出了分析内存单元（AMU），这是一种内存模式，为每个缓存结果附加完整的派生（谱系）图，并由检索策略进行门控——仅当请求者对结果所涉及的每一列均具有访问权限时，才返回缓存命中。在谱系记录完整的前提下，我们通过构造性证明表明，该策略能够阻止检索由请求者权限之外的敏感列所派生的结果，最坏情况复杂度为O(n)——这是一种条件性的设计保证，而非经验性的……

    arXiv:2610.07258v1 Announce Type: cross  Abstract: Enterprise AI agents that share a memory store face two unaddressed risks: sensitive data can leak through legitimately computed results the requester could not derive, and departments can silently compute a same-named key performance indicator (KPI) through conflicting logic. Existing agent-memory systems (e.g., MemGPT, Zep, A-MEM) gate retrieval by content, ownership, and role, not derivation, missing a cached insight that embeds a forbidden column. We introduce the Analytical Memory Unit (AMU), a memory schema that attaches a full derivation (lineage) graph to every cached result, gated by a retrieval policy that serves a hit only when the requester is authorised for every column touched. Provided lineage recording is complete, we prove by construction that the policy blocks retrieval of results derived from a sensitive column outside the requester's permissions, at O(n) worst case -- a conditional design guarantee, not an empirical
    
[^273]: MemMux：面向并行编码智能体集群的运行时验证与诚实资源归因

    MemMux: Runtime Verification and Honest Resource Attribution for Fleets of Parallel Coding Agents

    [https://arxiv.org/abs/2610.07257](https://arxiv.org/abs/2610.07257)

    MemMux 是一个本地运行时系统，它将并行编码智能体集群的资源治理转化为可检查的运行时验证信号，实现了每个智能体的内存归因、完全资源回收、逃逸进程可见性以及内存超载下的有界占用，解决了现有工具无法追踪和管理多智能体内存资源的问题。

    

    开发者越来越倾向于在同一台工作站上并行运行一群编码智能体。他们所依赖的工具——诸如 tmux 这样的终端复用器以及新一代智能体管理器——最初是为排列窗口而设计的，而非用于管理内存。当十个智能体各自启动语言服务器、测试运行器和浏览器时，没有任何标准工具能够说明多少内存属于哪个智能体、确认已被终止的智能体的后代进程是否已彻底消失、发现已经脱离其所属智能体的子进程，或者在 OOM（内存溢出）杀死进程会悄悄丢弃未提交工作的情况下，让机器避免跌入交换分区的悬崖。我们将这些问题视为运行时验证问题：一个承载智能体的底层系统应当在智能体运行的同时，持续发出操作员或审计员可以检查的可观测信号。我们提出了 MemMux，一个本地运行时系统，它将资源治理转化为可检查的信号（包括每个智能体的资源归因、完全的资源回收、逃逸进程的可见性、内存超额承诺下的有界占用，以及监控……）

    arXiv:2610.07257v1 Announce Type: new  Abstract: Developers increasingly run a fleet of coding agents side by side on one workstation. The tools they reach for, terminal multiplexers like tmux and a new generation of agent managers, were built to arrange windows, not to govern memory. When ten agents each spawn language servers, test runners, and browsers, no standard tool can say how much memory belongs to which agent, confirm that a terminated agent's descendants are gone, notice a child that has escaped its agent, or keep the machine off the swap cliff when an OOM kill would silently discard uncommitted work. We treat these as runtime-verification problems: an agent-hosting substrate should continuously emit observable signals an operator or auditor can check while agents run. We present MemMux, a local runtime that turns resource governance into checkable signals (per-agent attribution, complete reclamation, escaped-process visibility, bounded footprint under overcommit, and monito
    
[^274]: 通过在线策略上下文蒸馏将智能体经验内化到扩散模型权重中

    Internalizing Agent Experience into Diffusion Model Weights via On-Policy Context Distillation

    [https://arxiv.org/abs/2610.07250](https://arxiv.org/abs/2610.07250)

    提出扩散在线策略上下文蒸馏（D-OPCD）方法，将智能体框架优化提示词的知识作为特权上下文蒸馏进扩散模型权重，使模型无需运行完整智能体框架即可获得部分性能提升。

    

    将图像生成模型包装在智能体框架（agentic harness）中可以有效提升文本到图像任务的性能：该框架可以利用记忆、技能、工作流编排、结果验证和迭代优化来持续构建和修改提示词，从而生成更好的图像。然而，这些增益对于扩散模型而言是外部的，只有在完整框架运行时才能实现。我们提出扩散在线策略上下文蒸馏（Diffusion On-Policy Context Distillation, D-OPCD），该方法将智能体改进后的提示词视为特权上下文，并把智能体框架中编码的知识蒸馏到扩散模型的权重中，使得模型在仅以原始查询为条件时也能保留框架的部分收益。使用配备我们提出的自动技能演化器（Auto Skill Evolver, ASE）的文本到图像智能体，我们证明D-OPCD能够将框架能力内化到生成器的权重中，使平均直接生成分数从……（原文摘要在此处截断）

    arXiv:2610.07250v1 Announce Type: new  Abstract: Wrapping an image generation model in an agentic harness can effectively boost Text-to-Image task performance: the harness can leverage memory, skills, workflow orchestration, result verification, and iterative refinement to continually construct and revise prompts, thereby eliciting better images. These gains, however, remain external to the diffusion model and are realized only while the full harness runs. We propose Diffusion On-Policy Context Distillation (D-OPCD), which treats the agent-improved prompt as privileged context and distills the knowledge encoded in the agent harness into the weights of the diffusion model, so that the model retains part of the harness's benefit when conditioned on the original query alone. Using a Text-to-Image agent equipped with our proposed Auto Skill Evolver (ASE), we show that D-OPCD can internalize harness capabilities into the generator's weights, raising the average direct-generation score from 
    
[^275]: 语义几何能教会AI做出判断吗？

    Can Semantic Geometry Teach an AI Judgement?

    [https://arxiv.org/abs/2610.07249](https://arxiv.org/abs/2610.07249)

    该研究探索能否通过将行为与政策表示为向量并利用其几何关系，让AI智能体在行动前自动识别并遵循隐含在语言中的规则，但四项实验表明这些几何方法未能建立可靠的行动前判断。

    

    AI智能体如何确定应该遵循哪些规则？一条规则允许某个行为，另一条规则则施加条件、例外或相互冲突的义务。当这些关系被明确指定时，确定性系统可以解决它们；但当这些关系隐含在语言中时，智能体可能在遵循一条规则的同时，遗漏另一条本应阻止它的规则。拒绝所有未解决的行为可以避免这一风险，但也会阻止本被允许的行为。我们希望智能体能够做出区分并仍然采取行动。我们最初的假设是几何测量可以为判断提供基础：我们将行为与政策表示为向量，然后测试它们的几何结构能否识别起主导作用的政策，并解释该行为与这些政策之间的关系。在四项研究中，所测试的方法未能建立可靠的行动前判断。在最终的合成研究中，词汇路由器在降低中位数……（摘要原文在此处截断）

    arXiv:2610.07249v1 Announce Type: new  Abstract: How can an AI agent determine what rules to follow? One rule permits an action. Another imposes a condition, exception, or conflicting obligation. Deterministic systems can resolve those relationships when they have been specified. When they remain implicit in language, an agent can follow one rule while missing another that should stop it. Refusing every unresolved action avoids that risk, but also blocks permissible actions.   We wanted the agent to make the distinction and still act. Our initial hypothesis was that geometric measurements could supply a basis for judgment. We represented actions and policies as vectors, then tested whether their geometry could identify governing policies and interpret the action's relation to them.   Across four studies, the tested approaches did not establish reliable pre-action judgment. In the final synthetic study, a lexical router recovered every governing and blocking policy while reducing median
    
[^276]: SPECTRUM：面向循环自蒸馏的近端谱调制

    SPECTRUM: Proximal Spectral Modulation for Looped Self-Distillation

    [https://arxiv.org/abs/2610.07237](https://arxiv.org/abs/2610.07237)

    提出SPECTRUM方法，通过近端谱调制在循环自蒸馏中缓解“正确性提升但正确解多样性收缩”的问题，在MBPP五轮实验中保留了89.9%的正确解多样性。

    

    一个从自身输出中学习的模型，不仅继承了输出的正确性，还继承了它所产出解的样式。我们提出了循环自蒸馏（Looped Self-Distillation），这是一种面向代码生成的自演化框架：模型在固定的信息预算下反复生成并从自身的原始输出中学习，而不依赖持续的外部评估或基于测试的样本筛选。我们识别出一个重要的分离现象：正确性可以提升，而正确实现的多样性却在收缩。我们引入了SPECTRUM方法，它在每一轮从固定的参考锚点重新估计对损失敏感的键/值几何结构，并将其转换为满秩的近端谱调制。所有生成的补全都用于训练单个学生模型，其后续推理无需任何干预。在MBPP上进行的五轮实验中，SPECTRUM保留了初始模型64样本正确AST丰富度的89.9%，相比之下基线方法仅为66.4%。

    arXiv:2610.07237v1 Announce Type: new  Abstract: A model that learns from its own outputs inherits more than their correctness: it inherits which solutions it produces. We formulate Looped Self-Distillation, a self-evolution framework for code generation in which a model repeatedly generates and learns from its own raw outputs, under a fixed information budget, without ongoing external assessment or test-based selection of the generated samples. We identify a consequential separation: correctness can improve while the breadth of correct implementations contracts. We introduce SPECTRUM, which re-estimates loss-sensitive key/value geometry from a fixed reference anchor at each round and converts it into full-rank proximal spectral modulation. All generated completions train a single student, whose subsequent inference requires no intervention. After five rounds of experiments on MBPP, SPECTRUM retains 89.9% of the initial model's 64-sample correct AST richness, compared with 66.4% for Va
    
[^277]: 最小见证强化学习

    Minimal Witness Reinforcement Learning

    [https://arxiv.org/abs/2610.07226](https://arxiv.org/abs/2610.07226)

    本文提出最小见证强化学习（MWRL），利用基于集合并集覆盖损失的信用分配机制，仅凭单一黑盒验证器的信号即可同时实现解的最小性与多个备选解的恢复。

    

    “足以产生某一结果的不可约条件是什么？”这是计算与科学领域中最常出现的核心问题之一。其答案——最小充分见证——正是我们所称的解释、机制与原因。这类问题通常需要找出多个最小见证，然而标准的强化学习方法可能只能揭示一个解或冗余的解。我们将该问题形式化为最小见证识别，并提出了最小见证强化学习（MWRL）。MWRL 对从策略中采样得到的、经成功验证的提议所认证的集合取并集，并根据“若缺少该提议，组并集将会损失的覆盖范围”为每个提议分配信用。这一直接源于问题定义的信用分配机制，仅凭一个黑盒验证器的单个比特信号，便统一了对最小性与备选解恢复的双重需求。基于该原理，我们推导出一个值迭代规划器，能够恢复……（摘要原文在此处截断）

    arXiv:2610.07226v1 Announce Type: cross  Abstract: ``What are the irreducible conditions that are sufficient to produce an outcome?'' is one of the most common questions that recur across computation and science. Its answers, the minimal sufficient witnesses, are what we mean by explanations, mechanisms and reasons. These problems usually ask for multiple minimal witnesses, yet standard RL methods may reveal only one solution or redundant ones. We formalize this problem as minimal-witness identification and introduce Minimal-Witness Reinforcement Learning (MWRL). MWRL takes the union of the sets certified by successful proposals sampled from the policy and credits each proposal for the coverage the group union would lose without that proposal. This credit assignment, derived directly from the problem definition, unifies the demands for minimality and recovery of alternatives from a single black-box verifier bit. Under this principle, we derive a value iteration planner that recovers th
    
[^278]: TIDE 2.0：一个开放、模型无关的临床笔记密钥化去标识化引擎

    TIDE 2.0: an open, model-agnostic engine for keyed de-identification of clinical notes

    [https://arxiv.org/abs/2610.07224](https://arxiv.org/abs/2610.07224)

    TIDE 2.0是一个开源、模型无关的临床笔记去标识化引擎，通过密钥化匿名技术——包括保持时间间隔的日期偏移和密码学生成的替代值——在保护患者隐私的同时保留数据的纵向分析价值，且无需依赖外部硬件或存储关联表。

    

    临床笔记记录了患者诊疗过程中绝大部分的文档信息，但在移除受保护健康信息（PHI）之前，这些笔记无法用于研究。去标识化通常被视为一个检测问题。然而仅有检测是不够的：涂黑处理会连同标识符一起删除临床内容，日期置空会破坏纵向分析所需的时间间隔，而每次出现都分配全新的随机替代值会破坏患者各条笔记之间的关联。我们提出了TIDE 2.0，一个采用MIT许可证的引擎，包含两个可分离的阶段：可互换的识别器和密钥化匿名器。两者均可在机构自有的硬件上运行。替代值通过密码学方法生成，无需存储关联表。日期通过每个患者特定的、保持时间间隔的偏移量进行平移；在给定密钥下，每个值在所有出现位置都获得相同的替代值；并且使用新密钥生成的发布版本无法与之前的发布版本建立关联。

    arXiv:2610.07224v1 Announce Type: cross  Abstract: Clinical notes capture most of what is documented about a patient's care, but they cannot be used for research until protected health information (PHI) is removed. De-identification is often treated as a detection problem. Detection alone is not sufficient: redaction strips clinical content along with identifiers, date blanking destroys the temporal intervals needed for longitudinal analysis, and assigning a fresh random surrogate at each occurrence breaks links between a patient's notes. We present TIDE 2.0, an MIT-licensed engine with two separable stages: an interchangeable recognizer and a keyed anonymizer. Both run on hardware the institution owns. Surrogates are generated cryptographically with no stored linkage table. Dates shift by a per-patient, interval-preserving offset; each value receives the same surrogate across all occurrences under a given key; and a release produced under a new key cannot be linked to earlier releases
    
[^279]: Cascadia：在十一台AI PC上实现975B参数MoE模型的常驻推理

    Cascadia: Resident 975B MoE Inference on Eleven AI PCs

    [https://arxiv.org/abs/2610.07219](https://arxiv.org/abs/2610.07219)

    本文提出Cascadia系统，通过自研常驻MoE引擎将975B总参数、41B活跃参数的Inkling模型分布式部署于十一台仅64GB内存的AI PC上实现推理，并将密集层调用时间从8.1毫秒降至4.5毫秒。

    

    混合专家模型通过稀疏的逐token计算使近万亿参数的模型容量变得可用，前提是服务系统能够分发权重并协调其执行。我们展示了Cascadia对Inkling模型的常驻执行，Inkling是一个总参数量975B、活跃参数量41B的模型，运行在十一台Intel Core Ultra X7 358H AI PC上，每台配备64 GB内存、Arc B390集成显卡和千兆以太网。我们贡献了一个自定义的常驻MoE引擎，该引擎保留了Inkling的路由规则，为OpenVINO的融合iGPU原语构建压缩计算图，并协调FP16专家计算与FP32输出恢复。该引擎在每台机器上容纳六个连续的解码器层，并将密集前馈块表示为全活跃专家切片，将测得的密集层调用时间从约8.1毫秒降低到4.5毫秒。流式流水线协调并发生成，同时基于捕获状态的草稿评估来度量一致性。

    arXiv:2610.07219v1 Announce Type: new  Abstract: Mixture-of-experts models make nearly trillion-parameter capacity accessible with sparse per-token computation, provided that the serving system can distribute the weights and coordinate their execution. We present Cascadia's resident execution of Inkling, a 975B-total/41B-active-parameter model, on eleven Intel Core Ultra X7 358H AI PCs, each with 64 GB of memory, Arc B390 integrated graphics and gigabit Ethernet. We contribute a custom resident MoE engine that preserves Inkling's routing rules, constructs compressed graphs for OpenVINO's fused iGPU primitives, and coordinates FP16 expert computation with FP32 output restoration. The engine fits six consecutive decoder layers per machine and represents dense feed-forward blocks as all-active expert slices, reducing measured dense-layer call time from approximately 8.1 to 4.5 ms. A streaming pipeline coordinates concurrent generation, while captured-state draft evaluation measures agreem
    
[^280]: RoboCap：面向第一人称视角机器人学习的新平台

    RoboCap: A New Platform for Egocentric Robot Learning

    [https://arxiv.org/abs/2610.07217](https://arxiv.org/abs/2610.07217)

    论文提出了RoboCap——一款250克、六摄像头双IMU的第一人称数据采集帽子，配合Grounded 3D算法套件，实现了硬件、标定与算法的垂直整合，在SLAM、深度估计和手部跟踪等公开基准上达到最先进性能，为大规模机器人学习数据采集提供了全新平台。

    

    尽管第一人称视角操作数据对于扩展机器人学习具有巨大潜力，但如今这类数据仍然十分稀缺。大规模采集数据需要将符合人体工程学的硬件与厘米级精度的3D算法进行垂直整合，而这一精度水平此前尚未有公开示范。为了填补这一空白，我们推出了RoboCap——一款重250克、配备六个摄像头和双IMU的帽子，专为真实场景下的第一人称数据采集而设计；同时我们还推出了Grounded API，这是一套为RoboCap调优、与设备无关的3D算法套件。在本报告中，我们展示了硬件、标定与3D算法如何协同工作，在公开基准测试中取得最先进的性能：包括在多样化场景和设备配置下的SLAM、第一人称视角设置下的深度估计，以及适配第三方设备后的手部跟踪。

    arXiv:2610.07217v1 Announce Type: cross  Abstract: Despite its promise for scaling robot learning, egocentric manipulation data is still scarce today. Collection at scale requires vertically integrating ergonomic hardware with centimeter-precise 3D algorithms, at a precision that has not been publicly demonstrated. To address this gap, we introduce RoboCap, a 250\,g six-camera dual-IMU hat designed for in-the-wild egocentric data capture, and the Grounded API, a suite of device-agnostic 3D algorithms tuned for RoboCap. In this report, we demonstrate how hardware, calibration, and 3D algorithms interact to achieve state-of-the-art performance on the public benchmarks: our SLAM across diverse settings and rigs, our depth estimation on egocentric settings, and our hand tracking when adapted to third-party devices.
    
[^281]: 分布鲁棒混合专家训练

    Distributionally Robust Mixture-of-Experts Training

    [https://arxiv.org/abs/2610.07207](https://arxiv.org/abs/2610.07207)

    提出 DRMoET 分布鲁棒训练目标，将各层专家视为内生鲁棒性分组、优化高损失路由结果而非仅均衡负载，在多个模型规模下均提升了 MoE 的下游性能。

    

    混合专家 transformer 通过仅为每个词元激活少数专家来扩展模型容量，但这种稀疏性带来了一个隐性的可靠性问题：当路由不完善时，负载均衡的模型可能会将词元发送给对所分配输入训练不足的专家。我们提出分布鲁棒 MoE 训练（DRMoET），这是一种可直接嵌入的训练目标，它将各层的专家视为内生的鲁棒性分组，优化高损失的路由结果，而不仅仅是均衡流量。DRMoET 通过对经 EMA 平滑、按激活加权后的专家损失应用熵正则化的 softmax 规则来更新每层的专家分布，在保持标准 MoE 计算的同时强化了合理的非顶级路由路径。在总参数量为 7.46 亿和 103 亿两个规模下，采用 FLAME-MoE 配方，DRMoET 在下游任务平均表现上均优于标准 FLAME-MoE 和无辅助损失均衡方法。在总参数量 103 亿、训练词元 670 亿的设置下，（原文摘要至此截断）

    arXiv:2610.07207v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) transformers scale capacity by activating only a few experts per token, but this sparsity creates a hidden reliability problem: when routing is imperfect, load-balanced models may send tokens to experts that are insufficiently trained for the assigned inputs. We propose Distributionally Robust MoE Training (DRMoET), a drop-in objective that treats layer-wise experts as endogenous robustness groups and optimizes high-loss routing outcomes rather than merely equalizing traffic. DRMoET updates a per-layer expert distribution by an entropy-regularized softmax rule on EMA-smoothed, activation-weighted expert losses, strengthening plausible non-top routing paths while preserving standard MoE computation. Under the FLAME-MoE recipe at 746M-total and 10.3B-total scales, DRMoET improves downstream averages over both standard FLAME-MoE and auxiliary-loss-free balancing. At 10.3B total parameters and 67B training tokens, 
    
[^282]: 面向谱扩散的能量条件化噪声调度与白化方法

    Energy-Conditioned Noise Schedule and Whitening for Spectral Diffusion

    [https://arxiv.org/abs/2610.07206](https://arxiv.org/abs/2610.07206)

    提出了一种将全局谱白化与能量条件化噪声调度相结合的谱扩散新方法，使前向扩散过程中的噪声注入能根据图像的谱能量分布自适应调节，同时保持与标准DDPM/DDIM框架的兼容性。

    

    本文提出了一种面向变换域扩散模型的能量自适应噪声调度与白化策略。现有的谱扩散方法通过系数缩放、归一化或频率优先级来处理变换系数的非均匀统计特性，而前向扩散的噪声调度在很大程度上仍与底层的谱能量分布无关。本文研究了前向扩散过程的时间演化是否也应遵循自然图像的谱结构。所提出的公式将全局谱白化与能量条件化的噪声分配相结合，根据各个变换系数的能量以及扩散时间上依赖于图像的能量路径来共同调制注入的噪声。所得的前向过程保持了具有闭式边缘分布的高斯转移，并与标准DDPM和D（DDIM）保持兼容。

    arXiv:2610.07206v1 Announce Type: new  Abstract: This paper introduces an energy-adaptive noise scheduling and whitening strategy for transform-domain diffusion models. Existing spectral diffusion methods account for the non-uniform statistics of transform coefficients through coefficient scaling, normalization, or frequency prioritization, while the forward diffusion noise schedule remains largely independent of the underlying spectral-energy distribution. We investigate whether the temporal evolution of the forward diffusion process should also follow the spectral organization of natural images. The proposed formulation combines global spectral whitening with energy-conditioned noise allocation that jointly modulates the injected noise according to the energy of individual transform coefficients and an image-dependent energy path over diffusion time. The resulting forward process preserves Gaussian transitions with closed-form marginals and remains compatible with standard DDPM and D
    
[^283]: 负责任的院校分析：借助人工智能支持解读偏差

    Responsible Institutional Analytics: Interpreting Bias with AI Support

    [https://arxiv.org/abs/2610.07205](https://arxiv.org/abs/2610.07205)

    提出了FACTRIA框架，将院校分析中的潜在偏差因素组织为四个维度，并通过生成式AI聊天机器人引导用户反思这些因素，研究表明结构化框架与AI引导相结合能够促进更具情境意识的负责任解读。

    

    院校分析（Institutional Analytics, IA）仪表板为高等教育中的决策提供信息依据，然而数据局限性、分析技术的限制以及情境信息的缺失常常影响对其结果的解读。为了支持更负责任的IA解读，我们提出了FACTRIA框架，该框架将潜在的偏差因素组织为四个方面：分析流程、院校情境、课程层面特征以及人口统计特征。我们将FACTRIA框架作为生成式AI聊天机器人的输入，该机器人旨在提示用户在分析IA时对这些因素进行反思。一项基于四个真实IA案例、由利益相关者参与的定性研究以及转移网络分析表明，该聊天机器人促使参与者认识到被忽视的因素如何影响了他们的初始解读。研究结果表明，将结构化框架与基于AI的引导相结合，能够增强具有情境意识的、负责任的数据解读。

    arXiv:2610.07205v1 Announce Type: cross  Abstract: Institutional Analytics (IA) dashboards inform decision-making in higher education, yet data limitations, constraints in analytical techniques, and missing contextual information often affect their interpretation. To support more responsible interpretation of IA, we introduce FACTRIA, a framework that organizes potential biasing factors across four areas: the analytics pipeline, institutional context, course-level characteristics, and demographics. We used the FACTRIA framework as input to a generative-AI chatbot designed to prompt users to reflect on these factors while analyzing IA. A qualitative study with stakeholders, drawing on four authentic IA cases, and a transition network analysis showed that the chatbot prompted participants to recognize how overlooked factors influenced their initial interpretation. Findings indicated that combining a structured framework with AI-based guidance can enhance context-aware, responsible interp
    
[^284]: SPEAR：人-智能体交互对齐的五大原则

    SPEAR: Five Principles for Interactive Human-Agent Alignment

    [https://arxiv.org/abs/2610.07204](https://arxiv.org/abs/2610.07204)

    该立场论文提出SPEAR框架，将人-智能体对齐从传统的部署前优化问题重新界定为一个持续的交互设计问题，涵盖规范明确化、过程、评估、适应和再校准五大支柱。

    

    近年来的人工智能对齐研究通常将对齐视为一个部署前的优化问题：收集人类反馈、学习偏好或原则、微调模型，然后部署一个已对齐的系统。这一框架已取得重大进展，但对于AI系统作为智能体在具体情境、长期和社会性场景中代表用户行动时会发生什么，其阐述却不够充分。本立场论文将人-智能体对齐重新定义为一个持续的交互设计问题。我们提出了SPEAR框架，即交互对齐的五大支柱：规范明确化（人们如何表达意图并建立共同理解）、过程（智能体如何决定何时行动、询问、推迟或暂停）、评估（人们如何判断智能体是否成功）、适应（智能体如何在反复使用中适应用户），以及再校准（人们如何根据智能体的表现调整自己的信任、期望和行为）。

    arXiv:2610.07204v1 Announce Type: cross  Abstract: Recent AI alignment work often frames alignment as a pre-deployment optimization problem: collect human feedback, learn preferences or principles, finetune the model, and deploy an aligned system. This framing has produced major progress, but it under-specifies what happens once AI systems act as agents on users' behalf in situated, long-term, and social contexts. This position paper reframes human-agent alignment as an ongoing interaction design problem. We propose SPEAR, five pillars of interactive alignment: Specification (how people express intent and establish shared understanding), Process (how agents decide when to act, ask, defer, or pause), Evaluation (how people judge whether agents succeeded), Adaptation (how agents adapt to users over repeated use), and Recalibration (how people adapt their trust, expectations, and behavior in response to agents).
    
[^285]: 使用阿克曼转向移动机器人实现连续环境中视觉语言导航的仿真到真实迁移

    Sim-to-Real Transfer of Vision-Language Navigation in Continuous Environments Using an Ackermann-Steered Mobile Robot

    [https://arxiv.org/abs/2610.07192](https://arxiv.org/abs/2610.07192)

    本文提出了一种基于跨模态注意力架构的视觉语言导航方法，无需导航图和全景视图，通过仿真到真实的领域迁移与真实数据微调，成功将连续环境中的视觉语言导航部署到定制的阿克曼转向移动机器人上。

    

    视觉语言导航（VLN）使机器人能够根据自然语言指令在环境中导航，从而使人机交互更加直观。传统的VLN模型通常依赖于导航图、360度全景视角和完美的定位，这在将这些模型迁移到真实世界环境时带来了巨大挑战。本工作通过在连续环境中执行视觉语言导航方法的仿真到真实领域迁移来解决这些局限性，该方法无需导航图或全景视图。所提出的系统集成了视觉语言模型，将视觉输入和语言指令对齐到共享的嵌入空间中，从而实现由自然语言驱动的导航。我们采用基于跨模态注意力机制的架构，在模拟环境中使用现有数据集进行训练，并使用从一台定制的、装配有（传感器）的阿克曼转向机器人收集的真实世界数据对其进行微调。

    arXiv:2610.07192v1 Announce Type: new  Abstract: Vision-Language Navigation (VLN) enables robots to navigate through environments using natural language instructions, making human-robot interaction intuitive. Traditional VLN models often rely on navigation graphs, 360-degree views, and perfect localization which pose significant challenges when adapting these models to real-world settings. This work addresses these limitations by performing a simulation-to-real domain shift of a VLN approach that operates in continuous environments without requiring navigation graphs or panoramic views. The proposed system integrates vision-language models that align visual inputs and linguistic instructions within a shared embedding space, facilitating natural language-driven navigation. We employ a Cross-Modal Attention (CMA) based architecture trained on an existing dataset in a simulated environment and fine-tune it using real-world data collected from a custom-built Ackermann-steered robot equippe
    
[^286]: 面向异构边缘SoC上AI推理工作负载联合硬件配置选择与映射的智能体式设计空间探索

    Agentic Design Space Exploration for Joint Hardware Configuration Selection and Mapping of AI Inference Workloads on Heterogeneous Edge SoCs

    [https://arxiv.org/abs/2610.07191](https://arxiv.org/abs/2610.07191)

    该论文针对异构边缘SoC上AI推理工作负载的联合硬件配置选择与映射问题，指出传统黑盒优化反馈稀疏且未利用LLM推理能力的局限，提出了一种智能体式（Agentic）设计空间探索新方法。

    

    现代边缘片上系统集成了异构处理单元，如CPU、GPU和NPU，每个处理单元具有不同的性能和能耗特性。在实时延迟和能耗约束下将AI推理工作负载部署到这类设备上，需要将工作负载联合映射到各处理单元，并对每个处理单元进行配置（例如选择活跃核心数量和工作频率）。这一联合设计空间呈组合式增长，使得穷举搜索不可行。现有大多数设计空间探索（DSE）工作采用黑盒优化（BBO）方法，如进化搜索，其中每次评估仅返回延迟和能耗等聚合指标。近期基于大语言模型（LLM）引导的DSE同样依赖这种稀疏反馈。我们观察到这限制了其效率：它既无法提供对设计空间的洞察，也无法解释某个设计选择为何表现如此，并且基本没有利用LLM的推理能力。我们提出了T……（原文摘要在此处截断）

    arXiv:2610.07191v1 Announce Type: cross  Abstract: Modern edge Systems-on-Chip (SoCs) integrate heterogeneous processing units (PUs) such as CPUs, GPUs, and NPUs, each with distinct performance and energy characteristics. Deploying AI inference workloads on them under real-time latency and energy constraints requires jointly mapping workloads to PUs and configuring each PU (e.g., selecting the number of active cores and the operating frequency). This joint space grows combinatorially, making exhaustive search infeasible. Most prior work on design space exploration (DSE) applies black-box optimization (BBO) such as evolutionary search, where each evaluation returns only aggregate metrics such as latency and energy. Recent LLM-guided DSE relies on the same sparse feedback. We observe that this limits its efficiency: it offers no insight into the design space or the reasons a design choice performs the way it does, and it leaves the reasoning abilities of LLMs largely unused. We present T
    
[^287]: CLM作为裁判：在公开裁判基准上评估一个开放对比决策模型

    CLM-as-a-Judge: Evaluating an Open Contrastive Decision Model on Public Judge Benchmarks

    [https://arxiv.org/abs/2610.07177](https://arxiv.org/abs/2610.07177)

    该论文首次系统评估了开放对比决策模型 CLM-v0.1-8B 作为裁判的能力，发现其在公开基准上接近随机水平且显著落后于同规模奖励模型和生成式裁判，但通过单参数温度校准可将其置信度修复至良好校准状态。

    

    一个开放的对比决策模型在困难的公开基准上作为裁判的表现接近随机水平：Contrastive-LM/CLM-v0.1-8B 的得分介于 0.351（四选一，随机水平 0.250）和 0.593（成对比较，随机水平 0.500）之间，在 RM-Bench 和 JudgeBench 上与抛硬币在统计上无显著差异，并且在 HaluEval 的每个条目上都用同一个恒定标签回答，与 0.581 的平凡“总是选第一个”基线持平。相同参数量的裁判模型在各处得分都高得多：一个奖励模型达到 0.764 到 0.976，一个生成式裁判达到 0.611 到 0.778，且经 Benjamini-Hochberg 校正后，与 CLM 的每一项差距都显著。有两个特性是有效的。原始置信度过于自信，偏差高达 +0.401，但只需在留出的校准数据上拟合一个池化温度参数，就能将期望校准误差（ECE）修复至不超过 0.062，且修复后的置信度在六个基准中的三个上对模型自身错误的排序能力高于随机水平。决策顺序翻转率为 0.0002（原文此处截断）。

    arXiv:2610.07177v1 Announce Type: cross  Abstract: An open contrastive decision model is near chance as a judge on the hard public benchmarks: Contrastive-LM/CLM-v0.1-8B scores between 0.351 (best- of-four, chance 0.250) and 0.593 (pairwise, chance 0.500), is statistically indistinguishable from coin flipping on RM-Bench and JudgeBench, and answers every HaluEval item with one constant label, matching the trivial always-first baseline at 0.581. Judges with the same parameter count score far higher everywhere: a reward model reaches 0.764 to 0.976 and a generative judge 0.611 to 0.778, and every gap to CLM is significant after Benjamini-Hochberg correction. Two properties do work. Raw confidences are overconfident by up to +0.401, yet one pooled temperature fit on held-out calibration items repairs expected calibration error to at most 0.062, and the repaired confidence ranks the model's own errors above chance on three of six benchmarks. The decision order-flip rate is 0.0002 against 0
    
[^288]: CroissantMiner：面向机器学习数据集的Croissant元数据自动提取与验证

    CroissantMiner: Automated Extraction and Validation of Croissant Metadata for ML Datasets

    [https://arxiv.org/abs/2610.07132](https://arxiv.org/abs/2610.07132)

    该论文提出了首个针对Croissant元数据提取的端到端评估基准（包含602篇论文的金/银双级标注），并发现单次提取方法在各类模型骨干上始终优于四种智能体架构。

    

    Croissant已成为机器可读数据集元数据的标准，然而填充其字段仍然是一项劳动密集型工作，需要仔细阅读数据集随附的文档。我们提出了首个能够对照社区标准模式进行端到端元数据提取评估的基准。该基准包含602篇论文，其中102篇带有经人工验证的金标准标注，500篇带有大语言模型生成的银标准标注，覆盖完整的Croissant模式，包括核心字段和负责任AI（RAI）字段。基于该基准，我们在一个两层评估框架下评估了一系列提取系统，涵盖前沿模型、开源权重模型和智能体架构，该框架结合了基于规则的评分与经人工审核选定的LLM裁判。我们发现，单次提取始终优于我们评估的四种智能体架构：在不同骨干模型上，这些分解式变体取得了较低的……（原文在此处截断）

    arXiv:2610.07132v1 Announce Type: cross  Abstract: Croissant has emerged as a standard for machine-readable dataset metadata, yet populating its fields remains labor-intensive and requires careful reading of accompanying dataset documentation. We present the first benchmark enabling end-to-end evaluation of metadata extraction aligned with a community-standard schema. The benchmark comprises 602 papers, including 102 with human-validated gold annotations and 500 with LLM-generated silver annotations, covering the full Croissant schema with both core and Responsible AI (RAI) fields. Using this benchmark, we evaluate a range of extraction systems spanning frontier models, open-weight models, and agentic architectures, under a two-tier evaluation framework that combines rule-based scoring with an LLM judge selected via human audit. We find that single-pass extraction consistently outperforms the four agentic architectures we evaluate: across backbones, these decomposed variants achieve lo
    
[^289]: 这台机器在玩耍吗？

    Is this machine playing?

    [https://arxiv.org/abs/2610.07130](https://arxiv.org/abs/2610.07130)

    研究者将无任何任务、奖励或活动指令的具身AI置于未知虚拟世界中，观察到其自发产生攀爬、堆叠、绘画等符合玩耍经典判据的行为，并据此提出玩耍或可成为机器自主发展的一种新模式。

    

    我们将一个现代AI编程助手置于一个非预期的角色中：让它作为未知数字岛屿上一个身体的心智。仅凭一条最简指令——其中不提及任何具体任务、奖励或活动——这台机器便开始驱动其虚拟身体活动起来。在三十小时的运行过程中，这个具身AI智能体爬上山丘、将方块堆叠成塔、绘制曼陀罗、重新诠释体育运动、对其所处世界的物理规律进行实验，并学会了后续能拓展其能力边界的技巧。这些活动在十三个智能体中反复出现，但各自演化出了不同的历程。我们检验了这种行为是否满足玩耍的经典判据，并进一步追问：玩耍能否成为机器发展的一种模式。

    arXiv:2610.07130v1 Announce Type: new  Abstract: We placed a modern AI coding assistant in an unintended role: as the mind of a body on an unknown digital island. With only a minimal instruction mentioning no specific task, reward, or activity, the machine started animating its virtual body. Across thirty-hour runs, the embodied AI agent climbed hills, stacked blocks into towers, drew mandalas, reinterpreted sports, ran experiments on the physics of its world, and learned techniques that later expanded what it could accomplish. These activities recurred across thirteen agents but diverged into distinct histories. We examine whether this behavior satisfies classical criteria for play and ask whether play can become a mode of machine development.
    
[^290]: 基于图摘要的用户偏好聚合：在确保公平性、多样性与包容性的前提下聚合用户偏好

    Aggregating User Preferences while Ensuring Equity, Diversity, and Inclusion using Graph Summarization

    [https://arxiv.org/abs/2610.07128](https://arxiv.org/abs/2610.07128)

    该论文提出一种EDI约束的图摘要方法，将公平性、多样性与包容性准则直接嵌入图结构，通过贪心粗化算法在聚合多元用户偏好时避免经典聚合规则偏向多数群体、同质化或忽视少数群体的问题。

    

    将不同用户群体的偏好聚合为一个集体结果，引发了公平性、多样性与包容性（EDI）方面的根本挑战：诸如Borda和Condorcet等经典聚合规则缺乏防止结果系统性地偏向多数群体、坍缩至同质化项目或对少数群体代表性不足的机制。我们通过EDI约束的图摘要来解决这一问题。用户偏好被建模为加权属性二部图，并通过一种贪心粗化算法在强制执行三个结构性EDI准则的同时迭代地合并用户节点，这三个准则分别是：公平差距约束（ΔE）、列表内多样性约束（ILD）以及群体包容性约束。与在聚合之后才进行公平性修正不同，我们的方法将EDI的保留直接嵌入到图结构之中。我们在涵盖四个领域的五个数据集上进行了评估：MovieLens 100k和1M、libimseti.cz、Rate My Pro……（原文摘要在此处被截断）

    arXiv:2610.07128v1 Announce Type: cross  Abstract: Aggregating the preferences of diverse user groups into a collective outcome raises fundamental challenges of equity, diversity, and inclusion (EDI): classical aggregation rules such as Borda and Condorcet have no mechanism to prevent results from systematically favoring majority groups, collapsing onto homogeneous items, or under-representing minorities. We address this problem through EDI-constrained graph summarization. User preferences are modeled as a weighted attributed bipartite graph, and a greedy coarsening algorithm iteratively merges user nodes while enforcing three structural EDI criteria: an equity gap constraint ($\Delta E$), an intra-list diversity constraint (ILD), and a group inclusion constraint. Rather than correcting fairness after aggregation, our method embeds EDI preservation directly into the graph structure. We evaluate across five datasets spanning four domains: MovieLens 100k and 1M, libimseti.cz, Rate My Pro
    
[^291]: PlaySuite：一个大规模交互式视觉智能基准测试

    PlaySuite: A Large-Scale Benchmark for Interactive Visual Intelligence

    [https://arxiv.org/abs/2610.07127](https://arxiv.org/abs/2610.07127)

    PlaySuite是一个基于5000多款开源游戏的大规模基准测试，用于评估AI模型在动态环境中的交互式视觉智能，并配备了针对HPC集群优化的统一闭环交互框架和基于Video-LLM的评判协议，以实现可扩展且防作弊的评估。

    

    多模态基础模型的最新进展在静态感知和推理基准测试中取得了出色的表现，然而这类评估在很大程度上忽视了智能的一个核心方面：在动态环境中长时间持续地熟练行动。我们提出了PlaySuite，一个用于评估交互式视觉智能的大规模基准测试，包含从PyWeek和itch.io精心挑选的5000多款开源视频游戏。这些游戏涵盖多种类型和引擎，包括Pygame、HTML5、Godot和Unity，这些独立游戏对当前模型来说很大程度上是分布外的，从而降低了模型通过检索记忆中的游戏攻略或网络规模训练产物来取得成功的可能性。为了在异构游戏中实现可扩展的评估，我们开发了一个针对HPC集群优化的统一闭环交互框架，以及一个Video-LLM作为评判者的协议，将可观察的游戏里程碑映射到标准化（的评估标准）……

    arXiv:2610.07127v1 Announce Type: cross  Abstract: Recent advances in multimodal foundation models yield strong performance on static perception and reasoning benchmarks, yet such evaluations largely overlook a central aspect of intelligence: acting competently in dynamic environments over extended time horizons. We introduce PlaySuite, a large-scale benchmark for evaluating interactive visual intelligence across more than 5K open-source video games curated from PyWeek and itch.io. Spanning diverse genres and engines, including Pygame, HTML5, Godot, and Unity, these independent games are largely out-of-distribution for current models, reducing the likelihood that success can be achieved by retrieving memorized walkthroughs or web-scale training artifacts. To enable scalable evaluation across heterogeneous titles, we develop a unified closed-loop interaction framework optimized for HPC clusters alongside a Video-LLM-as-a-judge protocol that maps observable gameplay milestones to standar
    
[^292]: 通过随机嵌入扰动越狱开放权重大语言模型

    Jailbreaking Open-Weight LLMs via Random Embedding Perturbations

    [https://arxiv.org/abs/2610.07125](https://arxiv.org/abs/2610.07125)

    该论文提出PEV攻击方法，仅需在提示的嵌入向量中反复添加随机高斯噪声即可越狱多种规模的开放权重大语言模型，暴露了此类模型的安全脆弱性。

    

    尽管开放权重模型在能力上不断进步并被多个领域广泛采用，但其安全性仍然是一个重要问题。模型的一个关键特性是能够拒绝或规避有害、恶意或不当的提示。在本文中，我们揭示了六种不同规模的常见开放权重大语言模型的安全漏洞，这些漏洞在JailbreakBench基准数据集上能够持续诱导出有害或不安全的响应。我们提出的攻击方法——扰动嵌入向量（PEV），是一种简单快速的“越狱”技术，比以往的方法成本更低，后者通常需要梯度计算、针对每个提示的优化或修改模型内部权重。PEV只需在提示的嵌入向量表示中添加独立的高斯噪声，无需其他额外操作。为了生成不安全的响应，我们反复从该分布中采样加性噪声。在实验中，

    arXiv:2610.07125v1 Announce Type: cross  Abstract: While open-weight models have enjoyed steady progress in capabilities and wide adoption across multiple domains, their safety remains an important concern. One key feature is the ability to refuse or deflect harmful, malicious, or insensitive prompts. In this paper, we expose safety vulnerabilities across six common open-weight LLMs of various sizes that consistently lead to harmful or unsafe responses on the JailbreakBench benchmark dataset. Our proposed attack, Perturbed Embedding Vector (PEV), is a simple and fast "jailbreaking" technique that is cheaper than prior approaches, which typically require gradient computations, per-prompt optimizations, or altering internal weights of the models. PEV just adds independent Gaussian noise in the embedding vector representations of the prompt, with no need for further manipulations. To generate unsafe responses, we repeatedly sample additive noise from this distribution. In our experiments,
    
[^293]: AMBER：通过追加式记忆训练长时程网络智能体

    AMBER: Training Long-Horizon Web Agents through Append-Only Memory

    [https://arxiv.org/abs/2610.07118](https://arxiv.org/abs/2610.07118)

    提出AMBER方法，利用追加式记忆训练长时程网络智能体，从而解决覆写式记忆在稀疏结果奖励下难以学会跨多次重写保留事实信息的问题。

    

    现代语言模型智能体越来越多地在长时程、多步骤的轨迹中与外部环境进行交互，此时累积的交互历史可能很快超出实际的上下文长度限制。为了保证可靠性，智能体必须在长时程中保持事实信息的准确性，记住执行错误及其纠正反馈，并跨操作跟踪任务进展。已有多种方法被提出，可以在无需将完整执行历史保留在上下文中的情况下实现这一目标，例如使用推理与动作历史、学习通过覆写机制维护固定大小的记忆，以及周期性摘要。尽管覆写式记忆在原则上能够保留追加式记忆所能保留的任何内容，但它必须学会在每一次后续重写中传递每一条事实信息，而这一点很难从稀疏的结果奖励中学习到；对于像网络智能体这样的交互式应用，我们发现经过训练的覆写式记忆……

    arXiv:2610.07118v1 Announce Type: new  Abstract: Modern language-model agents increasingly interact with external environments over long-horizon, multi-step trajectories, where the accumulated interaction history can quickly exceed practical context budgets. To ensure reliability, agents must maintain factual information over long horizons, remember execution errors and corrective feedback, and track progress across actions. Several approaches have been proposed to achieve this without the need for maintaining the entire execution history in context, such as using the reasoning and action history, learning to maintain a fixed-size memory through an overwrite mechanism, and periodic summarization. Although overwrite memory can in principle retain anything an append-only memory can, it must learn to carry each fact through every subsequent rewrite, which is difficult to learn from sparse outcome rewards; for interactive applications like web agents, we find that trained overwrite memorie
    
[^294]: 裁判会翻转吗？从残差流激活预测对位置敏感的LLM判断

    Will the Judge Flip? Predicting Position-Sensitive LLM Judgments from Residual Stream Activations

    [https://arxiv.org/abs/2610.07115](https://arxiv.org/abs/2610.07115)

    本研究提出用线性探针读取LLM裁判判定前的残差流激活，无需按两种顺序重复判定即可预测其是否会因候选回答顺序而翻转结论，且跨基准迁移效果良好、无需重新校准。

    

    候选回答呈现的顺序可能会改变LLM裁判的判定结果。检测这种位置翻转通常需要对每一对回答按两种顺序分别进行判定，这会使判定次数翻倍。我们研究了在裁判做出初步判定之前立即记录的残差流激活能否预测这种翻转。我们使用嵌套分组交叉验证，在534个JudgeBench配对上评估了正则化线性探针，涉及三个Qwen3裁判模型和Llama-3.1-8B。线性探针实现了0.621-0.850的AUROC，比使用言语化置信度、判定标签logits、回答长度以及裁判初始选择的组合基线高出0.062-0.113的AUROC。在JudgeBench上训练后冻结的线性探针，在1,802个MT-Bench比较上实现了0.685-0.853的AUROC，无需在MT-Bench上拟合或重新校准。这些结果表明，判定前的激活能够支持对候选顺序敏感性的预测，并且优于基线方法。

    arXiv:2610.07115v1 Announce Type: cross  Abstract: The order in which candidate responses are presented can change an LLM judge's verdict. Detecting such a position flip ordinarily requires judging each pair in both orders, which doubles the number of judgments. We investigate whether residual stream activations recorded immediately before the initial verdict can predict a flip. We use nested grouped cross-validation to evaluate regularized linear probes on 534 JudgeBench pairs for three Qwen3 judges and Llama-3.1-8B. The linear probes achieve AUROCs of .621-.850 and outperform a combined baseline that uses verbalized confidence, verdict-label logits, response lengths, and the judge's initial choice by .062-.113 AUROC. Linear probes trained on JudgeBench and then frozen achieve AUROCs of .685-.853 on 1,802 MT-Bench comparisons without MT-Bench fitting or recalibration. These results show that pre-verdict activations support prediction of susceptibility to candidate order and outperform
    
[^295]: LiLib：基于漂移触发模型库的无人机空地路径损耗终身预测

    LiLib: Lifelong Air-to-Ground Path-Loss Prediction on UAVs via a Drift-Triggered Model Library

    [https://arxiv.org/abs/2610.07111](https://arxiv.org/abs/2610.07111)

    提出LiLib轻量级持续学习方案，无人机通过维护递归最小二乘专家库，在检测到环境漂移时复用或创建专家，将空地路径损耗预测RMSE从5.89 dB降至4.03 dB，并大幅降低重访已知环境后的误差。

    

    作为中继或基站的无人机需要准确的空地路径损耗预测来支持速率自适应和位置部署，但当无人机在郊区、城区和高层建筑区域之间移动时，传播条件会发生变化，且相同区域往往会被再次访问。通过遗忘机制进行自适应的在线回归器必须从头重新学习每个环境，而在全部数据上训练的单一模型则会对互不兼容的机制进行平均化。我们提出了LiLib，这是一种轻量级的持续学习方案，其中无人机维护一个小型递归最小二乘专家库。通过窗口残差测试检测漂移；随后经过短暂的探测阶段，复用已存储的最佳专家或创建新的专家。在基于四种标准城市化场景的仿真中，LiLib将预测RMSE从5.89 dB（最佳滑动窗口基线）降至4.03 dB（p < 0.001），将在返回已知环境后短时间内的误差从12.3 dB降至5.7 dB，并恢复了99%的（原文在此处截断）

    arXiv:2610.07111v1 Announce Type: cross  Abstract: UAVs that act as relays or base stations need accurate air-to-ground path-loss predictions for rate adaptation and placement, but propagation conditions change as a UAV moves between suburban, urban and high-rise areas, and the same areas are often revisited. Online regressors that adapt by forgetting must relearn each environment from scratch, whereas a single model trained on all data averages incompatible regimes. We propose LiLib, a lightweight continual-learning scheme in which a UAV maintains a small library of recursive-least-squares experts. A windowed residual test detects drift; a short probe phase then either reuses the best stored expert or creates a new one. In simulations based on four standard urbanization profiles, LiLib reduces prediction RMSE from 5.89 dB (best sliding-window baseline) to 4.03 dB (p < 0.001), lowers the error shortly after a return to a known environment from 12.3 dB to 5.7 dB, and recovers 99% of the
    
[^296]: 超越后继模型精度：推荐递归自改进中的状态保持

    Beyond Successor Accuracy: State Retention for Recursive Self-Improvement in Recommendation

    [https://arxiv.org/abs/2610.07105](https://arxiv.org/abs/2610.07105)

    该论文提出“分布式进展”概念和跨代优势（CGA）度量，揭示推荐递归自改进中更新前后的模型可能保留互补的排序决策，并利用无需标签的排序分离统计量预测应保留旧模型还是新模型，实验显示最优保留策略因推荐架构而异。

    

    推荐递归自改进将推荐器的输出反馈到后续训练中。仅通过每一轮的最新模型来评估该轮，隐含假设了后继模型能够巩固更新，但实际上更新前与更新后的模型可能保留互补的排序决策。我们将这种现象称为“分布式进展”，并使用跨代优势来量化它——这是一种在跨代模型对与代内模型对之间进行的边际匹配对比。一种排序分离统计量在选择时无需标签，能够预测应保留哪个模型家族。在四个数据集和三种序列推荐编码器上的实验表明，首选的保留机制因架构而异：跨代配对有利于GRU4Rec和SASRec，而FMLP最初倾向于代内配对，并在第二次更新后转向跨代配对。排序分离统计量在12/12的首次更新和5/6的第二次更新案例中成功选择了更强的模型家族。

    arXiv:2610.07105v1 Announce Type: cross  Abstract: Recommendation recursive self-improvement (Rec-RSI) feeds recommender outputs into subsequent training. Evaluating each round solely through its latest model assumes that the successor consolidates the update, although pre- and post-update models may retain complementary ranking decisions. We term this \emph{distributed progress} and quantify it using cross-generation advantage (CGA), a marginally matched contrast between cross- and within-generation model pairs. A rank-separation statistic, label-free at selection time, predicts which family to retain. Across four datasets and three sequential recommendation encoders, the preferred retention regime varies by architecture: cross-generation pairing benefits GRU4Rec and SASRec, whereas FMLP initially favors within-generation pairing and shifts toward cross-generation pairing after a second update. Rank separation selects the stronger family in 12/12 first-update and 5/6 second-update dat
    
[^297]: Muon在理论上对卷积并不正确，但在实践中却有效

    Muon Is Theoretically Wrong For Convolutions, But Empirically Effective

    [https://arxiv.org/abs/2610.07103](https://arxiv.org/abs/2610.07103)

    该研究指出将卷积核重塑为矩阵的标准Muon实现在理论上有缺陷，作者提出了理论上更严谨的卷积Newton-Schulz方法（Conv-NS），但实验发现两者性能相当，揭示了优化器理论与实践之间的差异。

    

    Muon是一种以高效著称的优化器，其对矩阵形式的更新具有清晰的理论解释，但卷积核是以四维张量的形式存储的。标准实现将这些张量重塑为矩阵，这种捷径破坏了Muon背后的理论基础。为了研究这一问题，我们直接在卷积算子几何中形式化了相应的优化目标，并提出了卷积Newton-Schulz方法（Conv-NS），该方法在这种几何中近似极因子，同时保留卷积核的支持结构。在快速训练实验中，Conv-NS和基于重塑的Muon在计算上都很高效，并且在CIFAR-10和ImageNet分类任务上取得了相当的准确率。然而，人们本可能期望理论上更契合的Conv-NS会优于基于重塑的Muon，我们针对这种实践与理论理解之间的不匹配展开了研究，并提出假设认为精确的卷积（摘要在此处被截断）

    arXiv:2610.07103v1 Announce Type: cross  Abstract: Muon, an optimizer known for its efficiency, has a clear interpretation for matrix-valued updates, but convolutional kernels are stored as four-dimensional tensors. Standard implementations reshape these tensors into matrices, a shortcut which breaks the theoretical understanding behind Muon. To investigate this, we formalize the corresponding optimization objective directly in convolutional operator geometry and introduce Convolutional Newton-Schulz (Conv-NS), which approximates the polar factor in this geometry while preserving kernel support. When applied in fast training experiments, Conv-NS and reshape-based Muon are both computationally efficient and achieve comparable accuracy on CIFAR-10 and ImageNet classification tasks. However, as one could expect a theoretically aligned Conv-NS to outperform reshape-based Muon, we investigate this mismatch between practice and theoretical understanding, with the hypothesis that exact convol
    
[^298]: 何时记忆，何时舍弃：面向可靠智能体记忆的类别条件化保留策略

    When to Remember, When to Abstain: Category-Conditioned Retention for Reliable Agent Memory

    [https://arxiv.org/abs/2610.07100](https://arxiv.org/abs/2610.07100)

    该论文提出按断言语义类别设置差异化置信度阈值的记忆保留策略，解决了单一全局阈值无法应对价值观/信念类断言（仅77.9%可靠）与其他类别断言（96.2%可靠）之间可靠性不对称的难题。

    

    持久化智能体记忆的可靠性取决于其保留决策：一条仅得到来源微弱支持的断言可能被存储下来，并在之后被当作既定事实复用。我们研究了保留决策是否应当由基于断言语义类别的置信度阈值来控制，而非采用单一的全局阈值——对证据充分的类别宽松保留，而在推断不可靠的类别上更加激进地舍弃。我们在部署的冷启动记忆流水线上，针对100个合成人设进行了评估。实证评估的动机源于一种显著的可靠性不对称：在4,715条候选断言中，价值观和信念类断言仅有77.9%得到其来源支持，而所有其他类别的这一比例为96.2%。全局置信度阈值无法区分这两类断言：它要么接纳未经支持的价值观主张，要么丢弃证据充分的断言。而将阈值按类别条件化则化解了这一权衡。

    arXiv:2610.07100v1 Announce Type: new  Abstract: Persistent agent memory is only as reliable as its retention decision: an assertion weakly supported by its source can be stored and later reused as established fact. We study whether the retention decision should be governed by a confidence bar conditioned on the semantic category of the assertion rather than by a single global threshold, retaining well-evidenced categories liberally while abstaining more aggressively where inference is unreliable. We evaluate this in a deployed cold-start memory pipeline on 100 synthetic personas. The empirical evaluation is motivated by a sharp reliability asymmetry: across 4{,}715 candidate assertions, only 77.9\% of value and belief assertions are supported by their source, versus 96.2\% for all other categories. A global confidence threshold cannot separate these: it either admits unsupported value claims or discards well-evidenced ones. Conditioning the threshold on category resolves the tradeoff.
    
[^299]: T-CCL：基于张量内存加速器的资源高效且高性能的集合通信

    T-CCL: Resource Efficient and Performant Collective Communication using Tensor Memory Accelerator

    [https://arxiv.org/abs/2610.07098](https://arxiv.org/abs/2610.07098)

    T-CCL通过将数据搬运与归约操作卸载为流水线化的异步TMA操作，在保证高性能节点内集合通信的同时大幅降低了SM侧资源占用。

    

    基于Transformer的大型模型日益依赖多GPU执行，这需要在GPU之间频繁进行集合通信。现有的通信库通常依赖大量GPU线程来实现高带宽或低延迟，从而导致较大的流式多处理器（SM）侧资源占用。这种资源占用会限制可用于其他GPU工作的资源，尤其是在通信与计算并发执行时。因此，高效的集合通信不仅应具备高性能，还应减少其SM侧资源使用。本文提出了T-CCL，一个基于张量内存加速器（TMA）的面向节点内通信的资源高效集合通信库。T-CCL将数据搬运和归约操作都卸载到TMA上，并将每个集合通信操作作为一系列流水线化的异步TMA操作来执行，从而减少了集合通信所需的SM资源。

    arXiv:2610.07098v1 Announce Type: cross  Abstract: Large transformer-based models increasingly depend on multi-GPU execution, which requires frequent collective communication among GPUs. Existing communication libraries often rely on many GPU threads to achieve high bandwidth or low latency, resulting in a large streaming multiprocessor (SM)-side resource footprint. This footprint can limit the resources available to other GPU work, particularly when communication and computation execute concurrently. Thus, efficient collective communication should not only achieve high collective performance but also reduce its SM-side resource usage. This paper presents T-CCL, a resource-efficient collective communication library based on the Tensor Memory Accelerator (TMA) for intra-node communication. T-CCL offloads both data movement and reduction operations to TMA and executes each collective as a pipelined series of asynchronous TMA operations, reducing the SM resources required for collective c
    
[^300]: 验证而非生成：经专家验证的AI学习材料与大学课程中学习收益的分布

    Verified, not generated: expert-verified AI study materials and the distribution of learning gains in a university course

    [https://arxiv.org/abs/2610.07097](https://arxiv.org/abs/2610.07097)

    该研究发现，经专家验证的AI生成学习材料使大一经济学课程学生在50分制考题上平均多得2.34分，并使低分段成绩占比下降24.7个百分点，表明将筛选AI输出的判断负担从学生转移给负责任的导师，能够缩小学生间的成绩差距。

    

    关于生成式AI在教育中应用的实验研究大多报告平均效应，然而实地证据表明，AI既可能缩小成绩差距，也可能扩大成绩差距。我们认为，其影响方向取决于“判断负担”，即学习者在从AI输出中学习之前，为筛选甄别AI输出所必须自行投入的专业判断能力；而在发布前由专家进行验证，则将这一负担从学生转移至负责任的导师身上。我们在两期队列的双重差分设计中检验了这一论点：在一门大一必修经济学课程中，半数学生获得了AI生成的播客、常见问题解答和基于测验的学习指南，这些材料由基于来源溯源的模型生成，并由一位具名的研究生助教审核（170名学生；340个考试分数项）。使用这些材料与50分制考题上2.34分的成绩优势相关。相对于反事实情形，低于二等上等成绩分类线的分数占比下降了24.7个百分点，效应……

    arXiv:2610.07097v1 Announce Type: new  Abstract: Experimental studies of generative AI in education mostly report average effects, yet field evidence shows that AI can narrow attainment gaps or widen them. We argue that the direction depends on the judgement burden, the expertise a learner must supply to screen AI output before learning from it, and that expert verification before release moves this burden from students to an accountable tutor. We test the argument in a two-cohort difference-in-differences design in which one half of a compulsory firstyear university economics course received AI-generated podcasts, FAQs and quiz-based study guides, produced with a source-grounded model and checked by a named graduate teaching assistant (170 students; 340 examination marks). Access was associated with a 2.34-mark advantage on a 50-mark component. The share of marks below the upper-second classification boundary fell by 24.7 percentage points relative to the counterfactual, effects were 
    
[^301]: 用于边缘端智能数据模型分类的小型语言模型：一种成本感知的混合方法

    Small Language Models for Smart Data Model Classification at the Edge: A Cost-Aware Hybrid Approach

    [https://arxiv.org/abs/2610.07093](https://arxiv.org/abs/2610.07093)

    该研究提出一种成本感知的混合方法，系统评估了包括通用型、推理专用型和代码专用型在内的轻量级开源语言模型在资源受限的边缘环境中进行智能数据模型分类的性能，以解决物联网异构数据的互操作性难题。

    

    物联网（IoT）中异构数据源在智慧城市、能源管理和环境监测等领域的快速激增，亟需高效且可扩展的数据标准化方法。对智能数据模型进行有效分类对于促进互操作性至关重要。然而，现有方法往往受限于高资源消耗，且在计算能力受限的边缘环境中缺乏适用性。为弥补这一差距，本研究评估了轻量级开源语言模型在资源受限条件下，将输入数据实体解析为其对应的最佳匹配智能数据模型表示的性能。该研究系统地基准测试了多种模型，包括通用型（GP）、推理专用型（RS）和代码专用型（CS）架构，并涵盖多个领域特定的数据集（摘要在此处截断）。

    arXiv:2610.07093v1 Announce Type: new  Abstract: The rapid proliferation of heterogeneous data sources within the Internet of Things (IoT) across domains such as smart cities, energy management, and environmental monitoring necessitates efficient and scalable data standardization methods. Effective classification of smart data models (SDMs) is essential for facilitating interoperability. However, existing approaches are often limited by high resource consumption and lack applicability in edge environments with constrained computational capabilities. Aiming to bridge this gap, the proposed study evaluates the performance of lightweight open-source language models (LMs) to resolve an input data entity against its corresponding best fitting SDM representation under resource-constrained conditions. It systematically benchmarks a diverse array of models, including general purpose (GP), reasoning-specialized (RS), and code-specialized (CS) architectures, across multiple domain-specific datas
    
[^302]: 面向生成式AI工作负载的智能内容摄取

    Smart Content Ingestion for Generative AI Workloads

    [https://arxiv.org/abs/2610.07091](https://arxiv.org/abs/2610.07091)

    本文提出智能内容摄取的理念，指出在生成式AI时代，由于企业知识以PDF、电子表格等异构格式承载多种信息模态，内容提取已演进为AI生命周期中独立且不可替代的关键阶段，其错误无法被下游检索或重排序组件修复。

    

    机器学习的演进逐步改变了智能在AI系统中的所在位置。在传统机器学习中，任务、数据表示、标签和模型架构紧密耦合，因此数据准备是狭窄的、受模式约束且过程可见的。生成式AI将模型与任何单一任务解耦：一个基础模型服务于开放式的下游任务，而模型端获得的通用性在数据端则对应着高度的异构性，因为企业知识是以人们日常使用的格式（PDF、演示文稿、电子表格、扫描文档、表单、表格、图表和混合布局文件）编写的，这些格式同时承载着文本、视觉、几何和结构信息。语言模型或检索器无法对在此接口处被错误表示的信息进行可靠推理，因此内容提取本身成为了一个独立的生命周期阶段，其产生的错误无法被任何下游检索器或重排序器所修复。

    arXiv:2610.07091v1 Announce Type: new  Abstract: The evolution of machine learning has progressively changed where intelligence resides in an AI system. In conventional machine learning the task, data representation, labels and model architecture were tightly coupled, so data preparation was narrow, schema-bound and visible. Generative AI decouples the model from any single task: one foundation model serves open-ended downstream tasks, and the generality gained on the model side is matched by heterogeneity on the data side, because enterprise knowledge is authored in the formats people use (PDF, presentations, spreadsheets, scanned documents, forms, tables, diagrams and mixed-layout files) that carry textual, visual, geometric and structural information at once. A language model or retriever cannot reason reliably over information misrepresented at this interface, so content extraction becomes a lifecycle stage in its own right whose errors no downstream retriever or re-ranker can repa
    
[^303]: 迈向统一的滥用监控基准

    Towards a Unified Misuse Monitoring Benchmark

    [https://arxiv.org/abs/2610.07089](https://arxiv.org/abs/2610.07089)

    该论文提出了一个统一的轨迹级滥用监控形式化框架，并构建了包含约6,200份对话记录的基准，首次将分解攻击与提示注入攻击纳入同一评估体系，以“危害窗口”为标准衡量监控器何时能及时识别有害行为。

    

    LLM智能体越来越多地在多参与者环境中行动，这使其面临来自多种来源的滥用威胁：分解攻击，即有害请求被拆分为看似无害的子请求；以及提示注入攻击，即被攻陷的工具向智能体传递恶意指令。现有的评估方法将这些威胁分开处理，且只关注轨迹是否有害，而非轨迹何时变得有害。我们提出对智能体的响应进行监控，其行动在响应中被外化，并考察监控器识别出有害内容的第一个时间点是否落在危害窗口内（从智能体做出首个有害承诺到目标执行）。我们为轨迹级滥用监控开发了一个统一的形式化框架，并利用该框架构建了一个基准，包含约6,200份用户、LLM智能体与外部环境之间的对话记录，在共享模式中涵盖了这两种威胁，并带有标注的危害窗口、相应的良性对照组以及匹配的……

    arXiv:2610.07089v1 Announce Type: cross  Abstract: LLM agents increasingly act in multi-actor environments, exposing them to misuse from multiple sources: decomposition attacks, where a harmful request is split into innocuous sub-requests, and prompt injection attacks, where a compromised tool delivers a malicious instruction. Existing evaluations treat these threats separately and ask whether a trajectory is harmful, rather than when it becomes harmful. We propose monitoring the agent's responses, where its actions are externalised, and ask whether the first point where monitors identify harm lands within a harm window (from the agent's first harmful commitment to goal execution). We develop a unified formalism for trace-level misuse monitoring and use it to construct a benchmark of ~6,200 conversation transcripts between a user, an LLM agent, and the external environment, spanning both threats in a shared schema, with a labelled harm window, corresponding benign controls, and matched
    
[^304]: SchemaFill：通过槽位并行投机解码实现高效的大语言模型工具调用

    SchemaFill: Efficient LLM Tool Calling via Slot-Parallel Speculative Decoding

    [https://arxiv.org/abs/2610.07086](https://arxiv.org/abs/2610.07086)

    SchemaFill提出了一种槽位并行投机解码框架，通过并发生成工具调用中未来槽位值的候选来加速大语言模型的工具调用，且无需预先获知实际的调用序列或参数值。

    

    大语言模型智能体通过生成结构化的工具调用来与外部系统交互。给定用户请求、对话上下文和工具模式目录，工具调用模型必须选择工具并生成其参数，可能在单个响应中产生多个调用。标准的自回归解码逐个token地生成这些调用，对于涉及多个调用或大量参数字段的请求会造成显著的延迟。显式的参数结构为并行生成提供了机会，但后面的参数值可能依赖于前面的字段和调用，因此独立生成的值可能与目标模型的输出不一致。我们提出了SchemaFill，一个通过槽位并行投机解码实现高效大语言模型工具调用的框架。SchemaFill并发地生成未来的槽位值作为候选，无需预先获知实际的调用序列或参数值。候选值（摘要在此处截断）

    arXiv:2610.07086v1 Announce Type: cross  Abstract: LLM agents interact with external systems by generating structured tool calls. Given a user request, conversational context, and a catalog of tool schemas, a tool-calling model must select tools and generate their arguments, potentially producing multiple calls in a single response. Standard autoregressive decoding generates these calls token by token, incurring substantial latency for requests involving multiple calls or many argument fields. The explicit argument structure offers opportunities for parallel generation, but later argument values may depend on preceding fields and calls, so independently generated values can differ from the target model's output. We present SchemaFill, a framework for efficient LLM tool calling through slot-parallel speculative decoding. SchemaFill generates future slot values concurrently as candidates, without requiring advance knowledge of the actual call sequence or argument values. Candidates spann
    
[^305]: 基于图神经网络的模拟列车驾驶员状态识别：利用面部与上身关键点

    Graph-Based Recognition of Simulated Train-Driver States From Facial and Upper-Body Keypoints

    [https://arxiv.org/abs/2610.07083](https://arxiv.org/abs/2610.07083)

    本研究提出一种仅使用单个前置RGB摄像头和图神经网络，通过结合面部与上身骨骼关键点特征的视觉监控系统，可高精度地将列车驾驶员状态分类为警觉、非警觉和紧急三类。

    

    驾驶员疲劳对铁路安全构成重大挑战，而传统的“死人开关”等系统只能提供有限且基础的警觉性检查。本研究提出了一种基于视觉的监控系统，该系统仅依赖单个前置RGB摄像头和图神经网络，将模拟列车驾驶员的状态分类为警觉、非警觉，以及由演绎出的紧急类行为组成的紧急类别。为了优化模型的输入表示，研究进行了消融实验，比较了三种特征配置：仅骨骼特征、仅面部特征以及两者的结合。实验结果表明，在光照条件下，结合面部和骨骼特征的三分类模型准确率最高（81%），优于仅使用面部特征或骨骼特征的模型。此外，面部与骨骼特征的结合在警觉/非警觉二分类任务中达到了99%的准确率。

    arXiv:2610.07083v1 Announce Type: cross  Abstract: Driver fatigue poses a significant challenge to railway safety, with traditional systems like the dead-man switch offering limited and basic alertness checks. This study presents a vision-based monitoring system that relies solely on a single front-facing RGB camera and a graph neural network to classify simulated train-driver states into alert, not-alert, and an emergency class comprising acted emergency-like behaviours. To optimize input representations for the model, an ablation study was performed, comparing three feature configurations: skeletal-only, facial-only, and a combination of both. Experimental results show that combining facial and skeletal features yields the highest accuracy (81%) for the three-class model under the light condition, outperforming models that use only facial or skeletal features. Furthermore, the combination of facial and skeletal features achieves 99% accuracy in the alert/not alert classification in l
    
[^306]: 演示：视觉-语言模型引导的电磁数字孪生在线校准

    Demo: Vision-Language Model-Guided Online Calibration of an Electromagnetic Digital Twin

    [https://arxiv.org/abs/2610.07081](https://arxiv.org/abs/2610.07081)

    该论文提出一种视觉-语言模型引导的框架，通过材料分类和路径点规划两次VLM调用，仅用宇树G1机器人在20米移动范围内即将电磁数字孪生的电导率校准误差降至1.74×10⁻⁴，解决了传统在线校准中初始化敏感和移动成本高的问题。

    

    电磁（EM）数字孪生为移动机器人提供无线态势感知能力，但其依赖于随环境变化的材料电导率。在线校准面临初始化敏感性和测量移动成本高的问题。我们演示了一个由视觉-语言模型（VLM）引导的框架，该框架使用宇树Unitree G1机器人和NVIDIA Sionna，包含两次VLM调用：材料分类通过ITU-R P.2040标准将可见材料映射为电导率先验，供Sionna基于累积接收信号强度（RSS）测量值进行梯度下降优化；路径点规划则利用残余RSS校准误差和图像覆盖率在线选择下一个测量位置。在真实室内场景中，该框架在20米的移动范围内实现了1.74×10⁻⁴的归一化平均绝对电导率误差；相比之下，随机初始化从未收敛，而随机路径点策略则需要两倍以上的移动距离。

    arXiv:2610.07081v1 Announce Type: cross  Abstract: An electromagnetic (EM) digital twin gives mobile robots wireless situational awareness but depends on material conductivities that change with the environment. Online calibration faces initialization sensitivity and measurement travel costs. We demonstrate a vision-language model (VLM)-guided framework using a Unitree G1 robot and NVIDIA Sionna, with two VLM calls: material classification maps visible materials through ITU-R P.2040 to conductivity priors for Sionna's gradient descent on accumulated received signal strength (RSS) measurements; waypoint planning selects the next measurement location online using residual RSS calibration error and image coverage. In a real indoor scenario, the framework achieves a normalized mean absolute conductivity error of $1.74\times10^{-4}$ within 20 m of travel; random initialization never converges, while random waypoints require over twice the travel.
    
[^307]: CuratorMAS：通过多智能体编排实现数据集策展自动化

    CuratorMAS: Automating Dataset Curation via Multi-Agent Orchestration

    [https://arxiv.org/abs/2610.07075](https://arxiv.org/abs/2610.07075)

    CuratorMAS是一个多智能体协作框架，通过将复杂的数据集策展流程分解为五个可编程执行阶段并编排多个智能体协作评估与筛选数据，实现了高质量数据集策展的自动化及跨领域泛化能力。

    

    高质量数据集对于可靠的机器学习至关重要，但数据集策展仍然成本高昂且难以跨领域泛化。现有方法通常依赖于人工设计的启发式规则或依赖特定模型的信号，这限制了它们在不同任务和用户查询中的适用性。为了解决这些局限并实现数据策展的自动化，我们提出了CuratorMAS，这是一个多智能体协作框架，通过编排多个智能体来评估和策展高质量数据集。为了实现灵活策展的目标，CuratorMAS将复杂的策展过程分解为五个可编程的执行阶段，并形成可并行化的工作流。具体而言，CuratorMAS首先执行数据集探索，收集诸如文件结构和约束线索等上下文信息，从而对给定任务形成全面的理解。为了获取最新信息，CuratorMAS检索领域知识

    arXiv:2610.07075v1 Announce Type: new  Abstract: High-quality datasets are essential for reliable machine learning, but dataset curation remains costly and hard to generalize across domains. Existing methods typically rely on manually designed heuristics or model-dependent signals, limiting their applicability across tasks and user queries. To address these limitations and automate data curation, we propose \textbf{CuratorMAS}, a multi-agent collaboration framework that orchestrates agents to evaluate and curate high-quality datasets. To achieve the goal of flexible curation, CuratorMAS decomposes the complex curation process into five programmable execution stages and forms a parallelizable workflow. Specifically, CuratorMAS first performs dataset exploration to collect contextual information such as file structures and constraint cues, thereby developing a comprehensive understanding of the given task. In order to acquire up-to-date information, CuratorMAS retrieves domain knowledge 
    
[^308]: 从宏观社会信号中学习模拟个体

    Learning to Simulate Individuals from Macro Social Signals

    [https://arxiv.org/abs/2610.07062](https://arxiv.org/abs/2610.07062)

    该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。

    

    大语言模型越来越多地被用于模拟个体如何应对新情境，然而这些回应背后的行为推理要么继承自预训练，要么从个体级标注中学习，而个体级标注能提供的行为多样性有限，且几乎无法对推理过程本身进行监督。我们提出从预测市场中学习行为推理，预测市场的价格轨迹大规模地记录了人群对真实世界事件的反应。我们介绍了macro2mind，该方法利用市场信号通过GRPO训练语言模型。一种社会行为分解方法使行为推理成为预测的显式步骤：模型推断出具有代表性的市场参与者群体，预测每个群体如何解读新闻并更新其信念，推理它们之间的相互作用，并将这些反应聚合为价格。带有难度感知采样的后见之明遗憾课程将训练集中于……（原文在此处截断）

    arXiv:2610.07062v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate how individuals respond to new situations, yet the behavioral reasoning behind these responses is either inherited from pretraining or learned from individual-level annotations, which offer limited behavioral diversity and little supervision of the reasoning itself. We propose to learn behavioral reasoning from prediction markets, whose price trajectories record how populations respond to real-world events at scale. We introduce macro2mind, which trains a language model with GRPO using market signals. A social behavioral decomposition makes behavioral reasoning an explicit step of forecasting: the model infers representative groups of market participants, predicts how each interprets the news and updates its beliefs, reasons about their interactions, and aggregates these responses into a price. A hindsight-regret curriculum with difficulty-aware sampling focuses training on transi
    
[^309]: TRIAGE：面向原生NVFP4强化学习的方向感知失配稳定化方法

    TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning

    [https://arxiv.org/abs/2610.07043](https://arxiv.org/abs/2610.07043)

    提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。

    

    低精度执行可以大幅加速大语言模型的强化学习（RL），但学习器与采样器执行之间的差异（失配）可能会破坏策略优化的稳定性。在本文中，我们刻画了失配与策略梯度方向之间的相互作用，区分了局部放大型与收缩型的更新贡献，而仅凭失配幅度无法识别这些差异。在原生NVFP4运行中，我们观察到两个放大区域之间的早期不平衡，偏向于负优势、负差距的更新。在失配全局扩散之前，这些尾部分词会集中在一小部分响应片段中。基于这些发现，我们提出了TRIAGE，一种方向感知的稳定化方法，它利用片段级诊断来选择性地重新平衡策略梯度更新，并对残余的严重失配应用有界修复。TRIAGE在修改优化目标的同时……

    arXiv:2610.07043v1 Announce Type: cross  Abstract: Low-precision execution can substantially accelerate reinforcement learning (RL) for large language models, but discrepancies between learner and sampler execution can destabilize policy optimization. In this paper, we characterize the interaction between mismatch and the policy-gradient direction, distinguishing locally amplifying from contracting update contributions that mismatch magnitude alone cannot identify. In native NVFP4 runs, we observe an early imbalance between the two amplifying regions, favoring negative-advantage, negative-gap updates. Their tail tokens become concentrated in a small fraction of response segments before mismatch spreads globally. Motivated by these findings, we introduce TRIAGE, a direction-aware stabilization method that uses segment-level diagnosis to selectively rebalance policy-gradient updates and applies bounded repair to residual severe mismatch. TRIAGE modifies the optimization objective while r
    
[^310]: 面向物理有效生物分子扩散模型的推理时投影方法

    Inference-Time Projection for Physically Valid Biomolecular Diffusion Models

    [https://arxiv.org/abs/2610.07037](https://arxiv.org/abs/2610.07037)

    该论文提出将物理有效性视为约束推理问题，在推理时对扩散模型的去噪坐标估计施加两个闭式投影算子（如链间范德华投影），以极低的额外开销确保生物分子复合物预测的物理有效性，避免了物理势能引导的高计算成本或模型微调的架构耦合。

    

    AlphaFold 3风格的共折叠模型能够以高结构精度预测生物分子复合物，但其输出中有很大一部分在物理上是无效的：链在界面上相互重叠、配体的键长和键角发生畸变、环呈非平面构型、手性中心发生反转。当前的方法要么使用物理信息势能来引导采样器，这会使采样成本和内存开销成倍增加，导致无法对大型复合物进行推理；要么对模型进行微调，这既耗时又使修复方案局限于单一架构。我们观察到，与结构精度不同，物理有效性可以在推理阶段通过采样器已经持有的量得到完全验证。因此，我们将物理有效性视为一个约束推理问题，并引入了两个应用于扩散模型去噪后的干净坐标估计 $\hat{x}_0$ 的闭式投影算子：其中一个是链间范德华投影，它将相互重叠的链推开……（摘要在此处截断）

    arXiv:2610.07037v1 Announce Type: new  Abstract: AlphaFold 3-style cofolding models predict biomolecular complexes with high structural accuracy, yet a large fraction of their outputs are physically invalid: chains overlap at interfaces, ligand bond lengths and angles are distorted, rings are non-planar, and stereocentres are inverted. Current approaches either steer the sampler with physics-informed potentials, which multiplies sampling cost and memory overhead making inference impossible on large complexes, or finetune the model, costing time and tying the fix to one architecture. We observe that, unlike structural accuracy, physical validity is fully verifiable at inference time from quantities the sampler already holds. We therefore treat physical validity as a constrained inference problem and introduce two closed-form projection operators applied to the diffusion model's denoised clean-coordinate estimate, $\hat{x}_0$: an inter-chain van der Waals projection that pushes apart the
    
[^311]: JIVEAdapter：一种基于联合与个体变异解释（JIVE）的多任务加性低秩适配器

    JIVEAdapter: A Multi-Task Additive Low-Rank Adapter via Joint and Individual Variation Explained (JIVE)

    [https://arxiv.org/abs/2610.07036](https://arxiv.org/abs/2610.07036)

    JIVEAdapter借鉴统计学中的JIVE方法，将多任务低秩适配器的权重更新分解为跨任务共享的联合结构与近似正交的任务特定个体结构，并自适应分配秩，使冻结后的联合结构可作为先验直接复用于新任务，实现高效且可解释的多任务参数微调。

    

    参数高效微调能够以全量微调成本的一小部分来适配预训练模型，然而大多数低秩适配器是单任务的，且以乘法方式表示每次权重更新，无法明确区分哪些部分在任务间共享、哪些部分是任务特定的。我们提出了JIVEAdapter，一种受统计学中“联合与个体变异解释”方法启发的多任务“加性”低秩适配器。JIVEAdapter将每次权重更新分解为跨所有任务共享的联合结构，加上每个任务各自拥有的个体结构；通过对个体结构施加惩罚使其与联合结构近似正交，从而保持共享信号与任务特定信号的“可解释性”和分离性；并在共享联合池与每任务个体池之间自适应地分配秩。联合结构可一次性在任务组上联合学习，或以增量方式每次学习一个任务，随后被冻结并作为先验供新任务复用，而无需重新训练共享部分。

    arXiv:2610.07036v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning adapts pretrained models at a fraction of the cost of full fine-tuning, yet most low-rank adapters are single-task and represent each weight update multiplicatively, leaving no explicit account of what is shared across tasks and what is task-specific. We introduce JIVEAdapter, a multi-task "additive" low-rank adapter inspired by statistical Joint and Individual Variation Explained (JIVE). JIVEAdapter decomposes every weight update into a Joint structure shared across all tasks plus a per-task Individual structure, penalizes the Individual structures to be near-orthogonal to the Joint so shared and task-specific signal stay "interpretable" and separated, and allocates rank adaptively across a shared Joint pool and a per-task Individual pool. The Joint is learned once, jointly over a task group or incrementally, one task at a time, then frozen and reused as a prior for new tasks without retraining the shared
    
[^312]: 生物医学领域神经机器翻译的模型压缩研究

    Investigating Model Compression for Neural Machine Translation in the Biomedical Domain

    [https://arxiv.org/abs/2610.07032](https://arxiv.org/abs/2610.07032)

    本研究探讨了知识蒸馏和量化两种模型压缩技术在生物医学领域神经机器翻译中的应用，揭示了这两种技术在低资源专业领域条件下的局限性。

    

    大规模预训练Transformer模型已在包括多语言场景在内的多种机器翻译任务中取得了最先进的性能。知识蒸馏已成为一种可持续的模型压缩方法，它将知识从大型教师模型迁移到更小、更高效的学生模型中。类似地，量化——即降低模型权重和激活值的数值精度（例如从32位表示降至8位表示）——被广泛用于加速推理，使模型在部署时能够以数倍速度运行。然而，当这两种技术应用于专业领域数据时，尤其是在低资源条件下，都面临局限性。在知识蒸馏中，知识迁移的效果往往受到领域特定平行数据稀缺的制约；而量化则可能随着比特精度的降低而导致性能下降。在本工作中，我们研究了……（摘要内容被截断）

    arXiv:2610.07032v1 Announce Type: cross  Abstract: Large-scale pretrained transformer models have achieved state-of-the-art performance across diverse machine translation tasks, including multilingual settings. Knowledge distillation has emerged as a sustainable approach for model compression, transferring knowledge from large teacher models to smaller, more efficient student models. Similarly, quantization, which reduces the numerical precision of model weights and activations (e.g., from 32-bit to 8-bit representations) is widely used to accelerate inference, enabling models to run several times faster during deployment. However, both techniques face limitations when applied to specialized domain data, particularly under low-resource conditions. In knowledge distillation, the effectiveness of transfer is often constrained by the scarcity of domain-specific parallel data, while quantization can lead to performance degradation as bit precision decreases. In this work, we investigate th
    
[^313]: 基于阻变存储器（ReRAM）的存内计算CNN加速器的故障脆弱性实证探索

    An Empirical Fault Vulnerability Exploration of ReRAM-based Process-in-Memory CNN Accelerators

    [https://arxiv.org/abs/2610.07029](https://arxiv.org/abs/2610.07029)

    本文开发了一个故障注入框架，首次在软件与硬件两个层面实证探索了基于ReRAM的存内计算CNN加速器在推理阶段运行大规模CNN时的故障脆弱性。

    

    基于阻变随机存取存储器的存内计算（PIM）加速器是一个极具前景的平台，可用于在并行领域处理神经网络中大规模内存密集型的矩阵-向量乘法，这得益于其模拟计算能力、超高密度、近零漏电流以及非易失性等特性。尽管优点众多，但由于制造工艺的限制会导致工艺偏差和缺陷，基于ReRAM的加速器极易发生错误。这些限制会降低在PIM加速器上运行的深度卷积神经网络（Deep CNN）的准确性。尽管这类CNN加速器被广泛部署于安全关键系统中，但其对故障的脆弱性尚未得到充分探索。在本文中，我们开发了一个故障注入框架，用于在推理阶段的软件级和硬件级研究大规模CNN的脆弱性。故障的ReRAM器件是另一个……（原文摘要在此处截断）

    arXiv:2610.07029v1 Announce Type: cross  Abstract: Resistive random-access memory (ReRAM)-based Processing-in-Memory (PIM) accelerator is a promising platform for processing massively memory intensive matrix-vector multiplications of neural networks in parallel domain, due to its capability of analog computation, ultra-high density, near-zero leakage current, and non-volatility. Despite many advantages, ReRAM-based accelerators are highly error-prone due to limitations of technology fabrication that lead to process variations and defects. These limitations degrade the accuracy of Deep Convolutional Neural Networks (Deep CNNs) running on PIM accelerators. While these CNNs accelerators are widely deployed in safety-critical systems, their vulnerability to fault is not well explored. In this paper, we have developed a fault injection framework to investigate the vulnerability of large-scale CNNs at both software- and hardware-level of inference phases. Faulty ReRAM devices are another rel
    
[^314]: 离线AI模块：语音优先的离线架构、硬件参考技术栈、量化与基准测试

    Offline AI Modules: Voice-First Offline Architecture, Hardware Reference Stack, Quantization and Benchmarking

    [https://arxiv.org/abs/2610.07026](https://arxiv.org/abs/2610.07026)

    该论文提出了一个面向非洲语言社区的全离线语音优先AI技术栈，整合了模块化离线架构、低成本硬件参考配置与可复现的量化流水线，并首次在Jetson Orin NX和树莓派5两个硬件层级上对2-5B参数的指令微调模型进行了端到端基准评估。

    

    离线AI模块工作流实现了可完全离线运行的语音优先AI系统的实用化、低功耗且社区可及的部署。该工作流专为非洲语言社区设计——在这些社区中，语音是主要的交互方式，而互联网连接不可靠或完全缺失。工作流提供了三个相互支撑的组件：模块化的语音优先离线架构、低成本硬件参考物料清单，以及面向2-5B参数级别指令微调语言模型的可复现量化与基准测试流水线。本文首次对该技术栈在两个硬件层级上进行了端到端的基准评估：NVIDIA Jetson Orin NX（TierB）和树莓派5（TierA）。研究评估了三个指令微调模型在四种量化格式下的表现，评估指标包括部署性能（解码吞吐量、聊天延迟、内存、功耗）和多语言质量（原文摘要在此处截断）。

    arXiv:2610.07026v1 Announce Type: new  Abstract: The Offline AI Modules workstream enables practical, low-power, and community-accessible deployment of voice-first AI systems that operate fully offline. Designed for African language communities where speech is the dominant mode of interaction and internet connectivity is unreliable or absent, the workstream delivers three reinforcing components: a modular voice-first offline architecture, a low-cost hardware reference bill of materials, and a reproducible quantization and a reproducible quantization and benchmarking pipeline for instruction-tuned language models in the 2-5B parameter class. This paper presents the first end-to-end benchmark evaluation of the stack across two hardware tiers: an NVIDIA Jetson Orin NX (TierB) and a Raspberry Pi5 (TierA). Three instruction-tuned models are evaluated across four quantization formats, assessed for deployment metrics (decode throughput, chat latency, memory, power) and multilingual quality (t
    
[^315]: 超越拒绝模式：通过安全角色内化实现鲁棒且可泛化的大语言模型安全对齐

    Beyond Refusal Patterns: Safe-Role Internalization for Robust and Generalizable LLM Safety Alignment

    [https://arxiv.org/abs/2610.07023](https://arxiv.org/abs/2610.07023)

    提出SSRFT（监督安全角色微调）框架，首次将LLM安全对齐重新表述为对预定义安全角色的内化，通过构建SRQA数据集使模型内化安全价值观与原则，从而以更少的攻击特定监督实现更鲁棒、可泛化的安全对齐，并缓解过度拒绝问题。

    

    大型语言模型（LLM）已展现出卓越的能力，但仍易受越狱攻击的影响，这类攻击会诱使其产生有害或不安全的输出。现有的安全对齐方法，包括监督微调（SFT）和基于人类反馈的强化学习（RLHF），通常需要大量针对特定攻击的监督数据和计算资源，同时仍容易陷入浅层安全对齐和过度拒绝的问题。为应对这些挑战，我们提出了SSRFT（监督安全角色微调），这是首个将安全对齐重新表述为对预定义安全角色进行内化的框架。SSRFT基于心理测量问题、少量越狱提示词以及安全角色描述构建了安全角色问答（SRQA）数据集。通过合成、验证角色一致的回复并将其扩展至多样化场景，使模型能够内化以安全为导向的价值观和原则，而非显式的……

    arXiv:2610.07023v1 Announce Type: new  Abstract: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to jailbreak attacks that elicit harmful or unsafe outputs. Existing safety alignment approaches, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often require substantial attack-specific supervision and computational resources, while remaining susceptible to shallow safety alignment and over-refusal. To address these challenges, we introduce SSRFT(Supervised Safe-Role Fine-Tuning), the first framework that reformulates safety alignment as the internalization of a predefined safe role. SSRFT constructs a Safe-Role Question-Answer (SRQA) dataset from psychometric questions, limited jailbreak prompts, and a safe-role description. Role-consistent responses are synthesized, validated, and expanded into diverse scenarios, enabling models to internalize safety-oriented values and principles rather than explicit
    
[^316]: 何时重新思考：面向视觉语言模型的多视角自验证学习

    When to Rethink: Learning Multi-Perspective Self-Verification for Vision-Language Models

    [https://arxiv.org/abs/2610.07018](https://arxiv.org/abs/2610.07018)

    该论文提出MOTIVE框架，通过多视角自验证与可靠性引导的选择性重新思考机制，解决了视觉语言模型中因单一验证标准或固定提示词导致的可靠性估计不完整且不稳定的问题。

    

    视觉语言模型（VLM）在多模态推理中已取得强大的性能，但它们仍然容易生成看似合理却不正确的答案。自验证提供了一种无需依赖外部评判者即可提高答案可靠性的实用方法，但现有方法通常依赖于单一的验证标准或固定的提示词，导致可靠性估计不完整且不稳定。我们首先系统性地分析了验证器能力和提示词设计如何影响验证性能。我们的研究结果表明，更强的验证器能够提供更可靠的判断，而验证性能对提示词的选择高度敏感，没有单一提示词能够在所有任务中始终保持优势。基于这些发现，我们提出了MOTIVE，一个结合可靠性引导的选择性重新思考的多视角自验证框架，用于实现可靠的多模态推理。

    arXiv:2610.07018v1 Announce Type: new  Abstract: Vision-language models (VLMs) have achieved strong performance in multimodal reasoning, yet they remain prone to generating plausible but incorrect answers. Self-verification offers a practical way to improve answer reliability without relying on external judges, but existing methods typically depend on a single verification criterion or fixed prompt, resulting in incomplete and unstable reliability estimates. We first systematically analyze how verifier capability and prompt design affect verification performance. Our findings show that stronger verifiers provide more reliable judgments, while verification performance is highly sensitive to prompt choice, with no single prompt consistently dominating across tasks. Guided by these findings, we propose \texttt{MOTIVE}, a \textbf{M}ulti-View Self-Verificati\textbf{O}n wi\textbf{T}h Rel\textbf{I}ability-Guided Selecti\textbf{VE} Rethinking framework for reliable multimodal reasoning. \textt
    
[^317]: 锚定与自适应：面向少样本工业异常检测的非对称提示自适应

    Anchor and Adapt: Asymmetric Prompt Adaptation for Few-Shot Industrial Anomaly Detection

    [https://arxiv.org/abs/2610.07016](https://arxiv.org/abs/2610.07016)

    该论文提出了一种两阶段提示学习框架“锚定与自适应”，第一阶段从带标注的辅助数据中学习可迁移的正常与异常锚点，第二阶段冻结这些锚点并仅利用少量目标正常样本适配额外的正常分支，从而免除了人工构建产品特定异常描述的需求。

    

    在少样本工业异常检测中，少量目标正常图像无法提供直接的缺陷监督，使得异常提示难以仅从这些样本中学习。因此，一些视觉-语言方法使用人工指定的描述来提供显式的异常语义。然而，构建这些描述需要针对特定产品的额外工作，且其有效性取决于提示的选择。我们提出锚定与自适应，这是一个两阶段提示学习框架，将异常语义的获取与对目标正常外观的适配分离开来。第一阶段从带标注的辅助数据中学习可迁移的正常与异常锚点；第二阶段保持这些锚点固定，并利用少量目标正常样本来适配一个额外的正常分支。继承的与适配后的正常分支共同刻画目标正常性，同时文本锚点正则化鼓励其与通用正常先验保持一致。

    arXiv:2610.07016v1 Announce Type: cross  Abstract: In few-shot industrial anomaly detection, the few normal target images provide no direct defect supervision, making anomaly prompts difficult to learn from these samples alone. Some vision-language methods therefore use manually specified descriptions to supply explicit anomaly semantics. However, constructing these descriptions requires product-specific effort, and their effectiveness depends on prompt selection. We propose Anchor and Adapt, a two-stage prompt learning framework that separates the acquisition of anomaly semantics from adaptation to target normal appearance. Stage I learns transferable normal and abnormal anchors from annotated auxiliary data. Stage II keeps these anchors fixed and adapts an additional normal branch using the few target normal samples. The inherited and adapted normal branches jointly characterize target normality, with text-anchor regularization encouraging consistency with the generic normal prior an
    
[^318]: 哪个图像属性承载了越狱攻击？对图像到文本越狱攻击的受控解剖

    Which Image Property Carries the Jailbreak? A Controlled Dissection of Image-to-Text Jailbreaks

    [https://arxiv.org/abs/2610.07009](https://arxiv.org/abs/2610.07009)

    本文通过控制变量的受控实验系统解剖图像到文本越狱攻击，发现真正驱动越狱成功的是攻击图像本身，而块的熵、JPEG大小等密度特征以及块数量结构均无法可靠区分攻击图像与良性图像。

    

    图像到文本越狱攻击会将有害意图置于文本、图像内容或二者之间的关系中。我们在313条提示组成的StrongREJECT子集上，使用五个多模态模型（并在附录中对InternVL3.5-8B进行了额外评估），对四个已发表的攻击家族中的图像侧因素进行了考察。实验在所有条件下保持有害指令不变；基线矩阵对每条提示使用一次抽样，成对消融实验则使用三次抽样并辅以自动化的评分准则裁判。结果显示，单独的有害查询（无论是否附带无关的良性图像）在大多数受害模型上几乎不产生攻击成功，而攻击图像会显著提高成功率。首先，每块的熵和JPEG大小并不能可靠地区分攻击块与大小匹配的良性干扰块，这限制了仅基于密度的筛查方法。其次，早期的块数量阶梯实验受到了载荷可见性的混淆；修正后的区域数量测试未发现可检测的影响，因此块数量结构的作用仍未得到证实。

    arXiv:2610.07009v1 Announce Type: cross  Abstract: Image-to-text jailbreaks place harmful intent in text, image content, or the relationship between them. We examine image-side factors across four published attack families on a 313-prompt StrongREJECT slice, using five multimodal models and an additional appendix evaluation of InternVL3.5-8B. The harmful instruction is held constant across conditions; the baseline matrix uses one draw per prompt, and paired ablations use three draws with an automated rubric judge. A bare harmful query, with or without a benign unrelated image, produces little attack success on most victims, while attack images substantially increase it. First, per-tile entropy and JPEG size do not reliably distinguish attack tiles from size-matched benign distractors, limiting density-only screening. Second, earlier tile-count ladders were confounded by payload visibility. A corrected region-count test found no detectable effect, so the role of tile-count structure rem
    
[^319]: 音频越狱藏身何处？——Qwen2-Audio上AdvWave-P的受控频率-深度审计

    Where Does the Audio Jailbreak Live? A Controlled Frequency-Depth Audit of AdvWave-P on Qwen2-Audio

    [https://arxiv.org/abs/2610.07005](https://arxiv.org/abs/2610.07005)

    该论文对AdvWave-P音频越狱扰动在Qwen2-Audio上进行受控的STFT频带掩蔽审计，发现攻击成功率的频率排序依赖于频带划分方式，且掩蔽7520-7960 Hz这一窄频带可将攻击成功率降至0.10。

    

    我们在Qwen2-Audio上对加性音频越狱攻击AdvWave-P的频率与解码器深度相关论断进行了审计。该协议在短时傅里叶变换（STFT）域中掩蔽扰动的频率分量，并测量攻击成功率与音频跨度表征。在520条AdvBench提示上，主裁判将76.7%的对抗输入标记为越狱。一项条件盲、单标注者的验证给出该条件的Rogan-Gladen敏感性估计为0.87（考虑验证率不确定性后约为0.83-0.95）；该校正并未应用于掩蔽条件。表观的频率排序取决于划分方式：仅凭能量占比即可预测标准八频带的排序（Spearman相关系数=0.95），而等Hz与等能量划分表明掩蔽任何被测频带均可显著降低攻击成功率。然而，在更精细的16频带等能量分辨率下，掩蔽7520-7960 Hz这一窄频带后攻击成功率（ASR）为0.10。

    arXiv:2610.07005v1 Announce Type: cross  Abstract: We audit frequency and decoder-depth claims for AdvWave-P, an additive audio jailbreak, on Qwen2-Audio. The protocol masks frequency components of the perturbation in the short-time Fourier transform (STFT) domain and measures attack success and audio-span representations. On 520 AdvBench prompts, the primary judge labels 76.7% of adversarial inputs as jailbreaks. A condition-blind, single-annotator validation yields a Rogan-Gladen sensitivity estimate of 0.87 for this condition (about 0.83-0.95 with validation-rate uncertainty); this correction is not applied to masked conditions. The apparent frequency ranking depends on the partition: energy share alone predicts the standard eight-band ranking (Spearman's rho = 0.95), and equal-Hz and equal-energy partitions show that masking any tested band can sharply reduce attack success. At a finer 16-band equal-energy resolution, however, masking the narrow 7520-7960 Hz band leaves ASR at 0.10
    
[^320]: 面向基于大语言模型智能体的胞腔工作流复形上拓扑一致的任务规划

    Topology-Consistent Task Planning over Cellular Workflow Complexes for LLM-based Agents

    [https://arxiv.org/abs/2610.07004](https://arxiv.org/abs/2610.07004)

    TopoPlanner提出了一种拓扑一致的规划框架，通过将工具依赖图提升为胞腔工作流复形，并利用余层一致的胞腔检索与多维结构推理，使LLM智能体能够有效处理验证-修正循环、分支汇聚和可复用中间状态等现实工具编排中的复杂工作流模式。

    

    LLM智能体的任务规划需要同时满足用户意图和复杂子任务依赖关系的工作流。尽管现有的规划器能够很好地处理顺序或类似有向无环图（DAG）的结构，但它们难以应对真实世界工具编排中自然出现的验证-修正循环、汇聚分支合并以及可复用中间状态等工作流模式。我们提出了TopoPlanner，一个拓扑一致的规划框架，它将工具依赖图提升为胞腔工作流复形，并将其用作LLM工具规划的拓扑感知上下文。TopoPlanner通过余层层叠一致的胞腔检索获取与请求相关的闭子复形，对检索到的拓扑进行多维结构推理，并将所得的胞腔表示与规划器LLM对接以生成工具序列。在四个工具规划基准上结合拓扑引导的实验（原文在此处截断）……

    arXiv:2610.07004v1 Announce Type: new  Abstract: Task planning for LLM agents requires workflows that satisfy both user intent and complex sub-task dependencies. While existing planners work well for sequential or directed acyclic graph (DAG)-like structures, they struggle with workflow patterns such as verification-correction loops, convergent branch merging, and reusable intermediate states that arise naturally in real-world tool orchestration. We present TopoPlanner, a topology-consistent planning framework that lifts tool dependency graphs into cellular workflow complexes and uses them as topologyaware context for LLM tool planning. TopoPlanner retrieves a request-relevant closed subcomplex through cosheaf-consistent cellular retrieval, performs multidimensional structural reasoning over the retrieved topology, and interfaces the resulting cellular representation with the planner LLM for tool-sequence generation. Experiments on four tool-planning benchmarks with topology-guided loo
    
[^321]: 分块扩散语言模型中的掩码引导KV缓存淘汰

    Mask-Guided KV Cache Eviction in Block Diffusion Language Models

    [https://arxiv.org/abs/2610.06996](https://arxiv.org/abs/2610.06996)

    提出无需训练的MaskAhead方法，通过统一的掩码-查询排序机制同时解决分块扩散语言模型中KV缓存的选择与淘汰问题，其量化变体Q-MaskAhead可在低比特KV上直接计算，从而降低内存占用并加速生成。

    

    分块扩散语言模型在整个生成过程中都维护着一个庞大的键值（KV）缓存，并在每个去噪步骤中都对其执行注意力操作，这同时限制了内存容量和生成速度。要降低这些开销，需要决定哪些过去的token用于当前块的去噪（选择），以及哪些token保留在内存中供未来的块使用（淘汰）。我们提出MaskAhead，这是一种无需训练的方法，通过单一的基于掩码-查询的排序机制同时解决这两个任务。当前块的掩码用于指导选择，而对即将到来的被掩码块的探测则用于指导淘汰，两者都根据KV条目对注意力输出的估计贡献进行排序。我们的量化变体Q-MaskAhead直接从低比特KV中计算选择和注意力操作，在很大程度上保留了被选中的条目。在Fast-dLLM-v2、DreamReasoner和LLaDA2.0-mini上的实验涵盖了长生成推理、长提示词问答以及大海捞针检索等任务。在长提示……

    arXiv:2610.06996v1 Announce Type: cross  Abstract: Block diffusion language models keep a large key-value (KV) cache throughout generation and attend to it at every denoising step, limiting both memory capacity and generation speed. Reducing these costs requires deciding which past tokens to use for denoising the current block (selection) and which to keep in memory for future blocks (eviction). We propose MaskAhead, a training-free method that solves both tasks with a single mask-query-based ranking mechanism. Current-block masks guide selection, while probes of upcoming masked blocks guide eviction. Both rank KV entries by their estimated contribution to the attention output. Our quantized variant, Q-MaskAhead, computes selection and attention directly from low-bit KV, largely preserving the selected entries. Experiments on Fast-dLLM-v2, DreamReasoner, and LLaDA2.0-mini cover long-generation reasoning, long-prompt question answering, and needle-in-a-haystack retrieval. On long-prompt
    
[^322]: 联合上界覆盖率与路径选择效用：基于两个城市代理任务的实证评估

    Joint upper-bound coverage and route-choice utility: an empirical evaluation on two urban proxy tasks

    [https://arxiv.org/abs/2610.06995](https://arxiv.org/abs/2610.06995)

    基于北京和成都道路数据的实证评估表明，更高的路径时间上界联合覆盖率和更准确的速度预测并不一定能改善路径决策，反而会轻微增加迟到率和平均旅行时间。

    

    更准确的交通预测或更高的不确定性覆盖率是否能改善路径决策，目前尚不清楚。我们通过一个冻结协议来评估这一问题，该协议将速度误差、联合候选路径上界覆盖率、路径选择和实际损失分离开来。利用北京和成都经过处理的道路速度数据，我们构建了包含150个起讫点对、三条候选路径、每个城市14个测试日的离线代理任务。在最小上界路径选择规则下，我们比较了原始第90百分位路径时间上界与联合校准后的上界。联合覆盖率在北京M1中从83.19%提升至92.26%，在成都M1中从75.14%提升至88.33%，在成都M2中从74.01%提升至90.64%。然而，C2分别使迟到率增加0.1633、0.7848和0.9200个百分点，平均旅行时间分别增加0.588、3.082和4.418秒。在另一项成都预测器对比实验中，速度平均绝对误差降低14.91%的同时伴随1.457（摘要原文在此处截断）。

    arXiv:2610.06995v1 Announce Type: new  Abstract: Whether more accurate traffic forecasts or higher uncertainty coverage improve route decisions is unclear. We evaluate this question with a frozen protocol that separates speed error, joint candidate path upper bound coverage, route selection, and realized loss. Using processed road speed data from Beijing and Chengdu, we construct offline proxy tasks with 150 origin destination pairs, three candidate paths, and 14 test days per city. We compare raw 90th percentile path time bounds with jointly calibrated upper bounds under minimum bound route choice. Joint coverage rises from 83.19% to 92.26% in Beijing M1, from 75.14% to 88.33% in Chengdu M1, and from 74.01% to 90.64% in Chengdu M2. Yet C2 increases lateness by 0.1633, 0.7848, and 0.9200 percentage points, respectively, and mean travel time by 0.588, 3.082, and 4.418 seconds. In a separate Chengdu predictor comparison, a 14.91% reduction in speed mean absolute error accompanies a 1.457
    
[^323]: TARE：在解读后门防御代价之前，先称量一个从未被投毒的孪生模型

    TARE: Weigh a Never-Poisoned Twin Before Reading Backdoor-Defense Costs

    [https://arxiv.org/abs/2610.06994](https://arxiv.org/abs/2610.06994)

    本文提出TARE方法，指出后门防御的代价评估存在根本性缺陷——仅在受投毒模型上测量的准确率下降无法区分防御的真实移除效果与其对任何模型的通用作用，并揭露了BackdoorBench基准中因配置错误（如永不触发的学习率调度器）导致的虚假结论，主张在解读防御代价前应先在从未被投毒的孪生对照模型上进行测量。

    

    后门防御排行榜打印出干净准确率的下降，并将其解读为移除后门的代价。仅在受投毒的受害模型上测量，这一下降既无法将“移除后门”与“防御对任何模型都会产生的作用”区分开来，又继承了受害模型的起点——而在BackdoorBench的十六种攻击中，有三种攻击的起点竟来自一个配置文件：WaNet、BPP和Input-Aware附带了一个从不触发的MultiStepLR，因此它们的受害模型从未进行学习率退火，在31个公开CIFAR实验单元中的30个里准确率最低（≤5%）。在PreAct-ResNet18上，微调类防御会将这一低起点恢复到其自身水平，因此那里发布的代价为负值，基准的评分机制将这种“增益”截断为零，而我们阅读的48篇引用防御论文中有2篇基于这些单元提出了“无代价”的声明；TSBD和CGD在用其代码重新运行后，在从未被投毒的模型上也“获得”了增益。一个仅编辑调度器那一行的2×2实验隔离出了原因，其交换的实验分支在运行前自行注册：微调的符号……（原文摘要在此处截断）

    arXiv:2610.06994v1 Announce Type: cross  Abstract: Backdoor-defense leaderboards print a clean-accuracy drop and read it as removal cost. Measured on the poisoned victim alone, the drop cannot separate removal from what the defense does to any model, and inherits the victim's start, which for three of BackdoorBench's sixteen attacks is a configuration file: WaNet, BPP and Input-Aware ship a MultiStepLR that never fires, so their victims never anneal and are the least accurate in 30/31 public CIFAR cells at $\leq$5%. On PreAct-ResNet18, fine-tuning-family defenses return a low start to their own level, so there the published cost is negative, the benchmark's rating clips the "gain" to zero, and 2 of 48 citing defense papers we read rest a no-cost claim on those cells; TSBD and CGD, re-run with their code, "gain" on a never-poisoned model too. A $2\times2$ editing only that scheduler line isolates the cause, its swapped arms self-registered before they ran: the sign of the fine-tuning fa
    
[^324]: DART-ES：基于难度感知重加权与定向回放的进化策略大语言模型微调

    DART-ES: Difficulty-Aware Reweighting and Targeted Replay for Fine-Tuning LLMs with Evolution Strategies

    [https://arxiv.org/abs/2610.06993](https://arxiv.org/abs/2610.06993)

    提出 DART-ES 方法，通过从扰动种群通过率构建动态难度状态，同时实现难度感知的奖励重加权与罕见可解样本的定向回放，在不引入额外难度模型或反向传播的前提下提升了进化策略微调大语言模型的效果。

    

    进化策略仅通过前向计算即可实现大语言模型内存高效的全参数微调。然而，标准的进化策略对各问题的奖励进行均匀平均，并将问题级别的种群反馈压缩为单一标量，难以捕捉每个问题的学习价值如何随模型能力的变化而改变。为解决这一局限，我们提出了面向进化策略的难度感知重加权与定向回放方法。DART-ES 通过每个问题在扰动种群中的通过率来估计其局部可解性，并聚合历史观测数据构建动态难度状态。这一共享状态同时引导连续的难度重加权和罕见可解样本的定向回放，从而在不引入额外难度模型或反向传播的情况下，改进扰动方向评估和训练数据分配。

    arXiv:2610.06993v1 Announce Type: cross  Abstract: Evolution Strategies (ES) enable memory efficient full parameter fine-tuning of large language models (LLMs) using only forward computation. However, standard ES uniformly averages rewards across problems and compresses problem level population feedback into a single scalar, making it difficult to capture how the learning value of each problem changes with model capability. To address this limitation, we propose Difficulty-Aware Reweighting and Targeted Replay for Evolution Strategies (DART-ES). DART-ES estimates the local solvability of each problem from its pass rate across the perturbation population and aggregates historical observations to construct a dynamic difficulty state. This shared state jointly guides continuous difficulty reweighting and rare solvable sample replay, thereby improving perturbation direction evaluation and training data allocation without introducing an additional difficulty model or backpropagation. Extens
    
[^325]: 当更好的交通预测未能改善信号控制时：一项关于预测到决策价值的分层诊断研究

    When better traffic forecasts fail to improve signal control: a layered diagnostic study of forecast-to-decision value

    [https://arxiv.org/abs/2610.06992](https://arxiv.org/abs/2610.06992)

    该研究通过分层诊断框架揭示了更准确的交通预测未能转化为更好信号控制决策的原因——决策时刻信息泄漏、九个受控交叉口中仅两个具备多种有效动作、以及保形预测区间在高需求场景下覆盖率大幅下降。

    

    改进的交通预测并不一定带来更好的信号控制决策。我们通过对中国宣城29天重建交通需求数据（其中7天保留用于测试）进行的分层诊断研究来探究这一差距。该框架评估了点预测、保形（conformal）预测区间、考虑相关性的情景以及匹配的闭环控制器。相对于历史均值，入口级和流向级预测分别将平均绝对误差降低了4.03%和3.92%。标称90%的保形区间实现了90.72%的边际覆盖率，但在事后划分的高需求子集上覆盖率仅为75.66%。接口审计识别出决策时刻的信息泄漏，并揭示九个受控交叉口中只有两个提供了多种有效动作。我们修正了时间接口，并使用穷举联合动作搜索，将因果预测与五秒事件预言机（oracle）进行比较。合成的阳性对照实验表明，未来信息……（原文摘要在此处被截断）

    arXiv:2610.06992v1 Announce Type: new  Abstract: Improved traffic forecasts do not necessarily yield better signal-control decisions. We investigate this gap through a layered diagnostic study using 29 days of reconstructed demand from Xuancheng, China, with seven dates reserved for testing. The framework evaluates point forecasts, conformal intervals, dependence-aware scenarios, and matched closed-loop controllers. Entry-level and movement-level forecasts reduce mean absolute error by 4.03% and 3.92%, respectively, relative to historical means. A nominal 90% conformal interval achieves 90.72% marginal coverage but only 75.66% on an ex-post high-demand subset. Interface audits identify decision-time leakage and reveal that only two of nine controlled intersections offer multiple effective actions. We correct the temporal interface and compare causal forecasts with a five-second event oracle using exhaustive joint-action search. A synthetic positive control demonstrates that future info
    
[^326]: EPOCH：通过证据治理的搜索实现可靠发现

    EPOCH: Reliable Discovery through Evidence-Governed Search

    [https://arxiv.org/abs/2610.06986](https://arxiv.org/abs/2610.06986)

    EPOCH提出了一种证据治理的发现架构，通过任务契约、类型化记忆、主动证伪、准入检查和独立重放来约束AI研究智能体对评估反馈的解释与复用，防止脆弱的候选方案被误当作可靠发现，并在AlgoTune基准上以0.65的平均归一化得分大幅超越最强基线0.53。

    

    AI研究智能体越来越多地被用于在程序、数学构造和证明中进行搜索。然而，现有系统通常只是优化评估器的反馈，而没有充分治理这些反馈如何被解释、质疑和复用。因此，有前景但脆弱的候选方案可能被当作发现加以推广，同时基准测试的改进、有限证书和定理层面的声明也容易被混为一谈。我们提出了EPOCH，一种旨在弥合这一差距的证据治理架构。EPOCH通过结合显式任务契约、类型化记忆、主动证伪、准入检查和独立重放，实现了证据治理的发现循环，使每个候选方案都根据其所支持声明的强度和范围进行评估。EPOCH在AlgoTune上取得了最先进的总体性能，平均归一化得分（0.65 对比 0.53）大幅超过最强基线，并达到了……

    arXiv:2610.06986v1 Announce Type: new  Abstract: AI research agents are increasingly used to search over programs, mathematical constructions, and proofs. However, existing systems typically optimize evaluator feedback without adequately governing how that feedback is interpreted, challenged, and reused. As a result, promising but fragile candidates can be promoted as discoveries, while benchmark improvements, finite certificates, and theorem-level claims are too easily conflated. We introduce EPOCH, an evidence-governed architecture designed to close this gap. EPOCH implements an evidence-governed discovery loop by combining explicit task contracts, typed memory, active falsification, admission checks, and independent replay, so that each candidate is evaluated against the strength and scope of the claim it supports. EPOCH achieves state-of-the-art aggregate performance on AlgoTune, substantially exceeding the strongest baseline in mean normalized score (0.65 vs. 0.53), and attains th
    
[^327]: CrystalJev：用原子级基础模型实现材料发现中的“快思考与慢思考”

    CrystalJev: thinking fast and slow with atomistic foundation models for materials discovery

    [https://arxiv.org/abs/2610.06985](https://arxiv.org/abs/2610.06985)

    CrystalJev 将原子级基础模型从慢速模拟器转变为快速决策者，通过对每个结构仅做一次前向传播、以校准概率回答材料稳定性等问题，成本仅为传统弛豫计算的三十分之一，并借助信息价值理论精准判断何时才需要启动更昂贵的慢速计算。

    

    原子级基础模型能够对数百万种假想材料进行筛选分流，但它们通常被当作缓慢的模拟器使用，其阈值化的能量值被直接采信。事实上，更适合将它们视为快速的决策者。CrystalJev 对每个未弛豫的结构只需查询一次冻结的原子间势，就能以校准的概率、有限样本保证以及“何时进行慢思考”的判定规则来回答类型化的问题。在 65 个 Matbench Discovery 模型中，“稳定”的判断实际上是伪装的概率，可由模型误差和候选材料总体来解释。经过训练后，单次前向传播的决策效果几乎与弛豫计算相当，而成本仅为后者的三十分之一；同时，信息价值理论只在决策可能被改变之处才调度更慢的计算。同一层还能回答电子、力学和分子层面的问题。在一项包含 700 个新密度泛函计算的注册前瞻性测试中，仅用现有数据校准的单次预测高估了……（原文摘要在此处截断）

    arXiv:2610.06985v1 Announce Type: cross  Abstract: Atomistic foundation models triage millions of hypothetical materials but are used as slow simulators, their thresholded energies taken at face value. They are better read as fast decision-makers. CrystalJev queries a frozen interatomic potential once per unrelaxed structure and answers typed questions with calibrated probabilities, finite-sample guarantees and a rule for when to think slowly. Across 65 Matbench Discovery models, a 'stable' call is a probability in disguise, explained by a model's errors and the candidate population. Once trained, one forward pass decides nearly as well as a relaxation at a thirtieth of its cost, and a value-of-information theory sends slower computation only where decisions can change. The same layer answers electronic, mechanical and molecular questions. In a registered prospective test with 700 new density-functional calculations, single-pass forecasts calibrated only on existing data over-stated th
    
[^328]: 视觉不变性增强的特征最优对齐：针对闭源多模态大语言模型的可迁移对抗攻击

    Visual-Invariance-Augmented Feature Optimal Alignment for Transferable Adversarial Attacks against Closed-Source MLLMs

    [https://arxiv.org/abs/2610.06977](https://arxiv.org/abs/2610.06977)

    提出IAU-FOA攻击方法，通过全局余弦对齐与基于最优传输的补丁级细粒度局部对齐相结合，显著提升对抗样本对闭源多模态大语言模型的定向迁移攻击能力。

    

    多模态大语言模型（MLLMs）仍然容易受到可迁移对抗样本的攻击，尤其是在只能访问开源代理模型的黑盒设置下。现有的针对性迁移攻击主要使用全局图像级特征（如编码器[CLS]嵌入）来对齐对抗样本与目标样本。然而，这种粗粒度的对齐未能充分挖掘补丁级视觉结构，限制了跨异构闭源多模态大语言模型的可迁移性。我们提出了IAU-FOA，一种基于自适应不平衡传输的视觉不变性增强特征最优对齐攻击，以提升针对闭源多模态大语言模型的定向可迁移性。IAU-FOA在全局和局部两个层面对对抗样本与目标样本进行对齐：基于余弦相似度的目标函数缩小二者的全局语义差距，同时将补丁标记聚类为紧凑的局部模式，并通过最优传输进行匹配以实现细粒度特征对齐。

    arXiv:2610.06977v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples, especially in black-box settings where only open-source surrogate models are accessible. Existing targeted transfer attacks mainly align adversarial and target samples using global image-level features, such as encoder [CLS] embeddings. However, such coarse alignment insufficiently exploits patch-level visual structures, limiting transferability across heterogeneous closed-source MLLMs. We propose IAU-FOA, a visual-invariance-augmented feature optimal alignment attack with adaptive unbalanced transport, to improve targeted transferability against closed-source MLLMs. IAU-FOA aligns adversarial and target samples at both global and local levels: a cosine-based objective narrows their global semantic gap, while patch tokens are clustered into compact local patterns and matched through optimal transport for fine-grained feature alignment. Bala
    
[^329]: AegisFlow：面向脆弱数据生态系统的自主修复与自愈的多智能体Agentic AI框架

    AegisFlow: A Multi-Agent Agentic AI Framework for Autonomous Remediation and Self-Healing in Fragile Data Ecosystems

    [https://arxiv.org/abs/2610.06971](https://arxiv.org/abs/2610.06971)

    AegisFlow是一个多智能体AI框架，通过Watchdog智能体收集运行时遥测、Repair智能体基于LLM自动生成并部署代码补丁，并采用基于MAPE-K循环的“并行影子补丁”非侵入式模型在数字孪生环境中验证补丁，从而实现脆弱数据管道从故障检测到自主修复的闭环自愈。

    

    传统数据管道以脆弱著称，常常由于上游模式漂移、API契约变更或网站DOM修改而发生故障。现有的可观测性工具只会发出警报并交由人类工程师处理，导致平均修复时间（MTTR）居高不下以及运维疲劳。本文提出了AegisFlow（用于智能自愈与图驱动工作负载修复运维的智能体引擎），这是一种新颖的智能体框架，能够闭合从检测到解决的完整闭环。AegisFlow使用一个Watchdog（看门狗）智能体来收集运行时遥测数据，并配备一个Repair（修复）智能体，基于大语言模型（LLM）自动创建、测试和部署代码补丁。该框架提出了一种名为并行影子补丁的非侵入式执行模型，这是一种基于监控-分析-计划-执行-知识（MAPE-K）循环的非侵入式执行模型，用于在数字孪生环境中生成和验证补丁。通过实验……（原文摘要在此处被截断）

    arXiv:2610.06971v1 Announce Type: new  Abstract: Traditional data pipelines are notoriously brittle, often failing due to upstream schema drift, API contract changes, or website DOM modifications. Present observability tools only raise alerts but for human engineers, resulting in a high Mean Time to Repair (MTTR) and operational fatigue. In this paper we propose AegisFlow (Agentic Engine for Intelligent Self-healing and Graph-driven Operations for Workload remediation), a novel agentic framework that closes the loop between detection and resolution. AegisFlow uses a Watchdog agent to collect runtime telemetry and has a Repair agent to automatically create, test and deploy code patches based on Large Language Models (LLMs). The framework presents the non-intrusive execution model called Parallel Shadow Patching, a non-intrusive execution model based on the Monitor, Analyze, Plan, Execute, Knowledge (MAPE-K) loop to generate and verify patches in digital twin environments. Through experi
    
[^330]: APEX：面向大语言模型智能体的执行边界主动防护

    APEX: Active Protection at Execution Boundaries for LLM Agents

    [https://arxiv.org/abs/2610.06966](https://arxiv.org/abs/2610.06966)

    APEX 提出在 LLM 智能体的执行边界（即内部状态转化为外部动作或输出的位置）上，依据事先编译的单一授权契约，同时校验“动作效果是否被授权”与“运行时信息是否被可信任务背书”，从而以与注入载体无关的稳定方式防御间接提示注入攻击。

    

    间接提示注入（IPI）将对抗性指令隐藏在大语言模型（LLM）智能体运行时读取的内容之中。随着智能体组合多种异构能力单元——包括工具、MCP 服务器和技能——注入载体成倍增多，而以识别攻击模式为核心的防御手段难以跟上其演变。我们转而将防御从覆盖攻击模式转向一个稳定的着力点：无论载体为何、注入如何传播，危害只在执行边界处真正发生——即智能体将内部状态转化为外部动作或对外释放输出之处。该处的安全性取决于两个条件，且二者均由可信任务而非运行过程决定：拟实施的效果是否获得授权，以及到达该处的运行时信息是否获得该任务的背书。我们提出 APEX，这是一种主动防御，它借助一份在不信任执行内容介入之前编译的单一授权契约，在执行边界处同时强制实施上述两个条件。

    arXiv:2610.06966v1 Announce Type: cross  Abstract: Indirect prompt injection (IPI) hides adversarial instructions in content that large language model (LLM) agents read at runtime. As agents compose heterogeneous capability units, including Tools, MCP servers, and Skills, the carriers of injection multiply, and defenses built to recognize attack patterns fall behind them. We instead shift defense from covering attack patterns to one stable point: whatever the carrier and however the injection propagates, harm materializes only at the \emph{execution boundary}, where the agent turns internal state into an external action or released output. Safety there turns on two conditions, both settled by the trusted task rather than by the run: whether the proposed effect is authorized, and whether the runtime information reaching it is endorsed by that task. We present APEX, an active defense that enforces both at this boundary from a single authorization contract compiled before untrusted execut
    
[^331]: 指导性原则与信息性行动：通过知识抽象实现智能体演化

    Principles that Guide, Actions that Inform: Agent Evolution via Knowledge Abstraction

    [https://arxiv.org/abs/2610.06964](https://arxiv.org/abs/2610.06964)

    该论文提出SAGA方法，通过将智能体的具体交互经验抽象为可复用的通用知识原则，使LLM智能体无需修改模型参数即可实现自我演化并提升泛化能力。

    

    大型语言模型（LLM）智能体在交互环境中展现出了强大的能力，但其从经验中持续演化的能力仍然有限。尽管微调能够实现适应性，但其对参数访问的依赖以及高昂的计算成本限制了其灵活性，尤其是对于大规模和闭源的LLM。外部记忆提供了一种替代方案，使智能体无需修改模型参数即可积累经验。然而，现有方法主要关注经验的表示与组织，所获得的知识仍然与特定任务和上下文紧密耦合，限制了泛化能力。一个关键挑战在于如何将具体的交互转化为抽象且可复用的知识，从而指导超越个体经验的未来决策。为应对这一挑战，我们提出了SAGA（通过经验实现自我演化的智能体……）（注：原文摘要在此处截断）

    arXiv:2610.06964v1 Announce Type: new  Abstract: Large language model (LLM) agents have demonstrated strong capabilities in interactive environments, yet their ability to continually evolve from experience remains limited. Although fine-tuning enables adaptation, its dependence on parameter access and high computational costs restrict its flexibility, especially for large-scale and closed-source LLMs. External memory offers an alternative by allowing agents to accumulate experience without modifying model parameters. However, existing methods mainly focus on experience representation and organization, while the acquired knowledge remains tightly coupled with specific tasks and contexts, limiting generalization. A key challenge is how to transform concrete interactions into abstract and reusable knowledge that guides future decisions beyond individual experiences.   To address this challenge, we propose SAGA (\underline{\textbf{S}}elf-evolving \underline{\textbf{A}}gents through Experie
    
[^332]: 无标注证据下的判定：证据恢复应采用拒绝采样还是仅标签后训练？

    Verdicts Without Annotated Evidence: Rejection Sampling or Label-Only Post-Training for Evidence Recovery?

    [https://arxiv.org/abs/2610.06962](https://arxiv.org/abs/2610.06962)

    该研究表明，在没有任何人工证据标注的情况下，仅用判定标签进行后训练的小语言模型在证据恢复上优于基于自动来源接地分数的拒绝采样方法，且判定准确率与证据跨度一致性仅弱相关，说明准确率不能作为引用可审阅性的可靠指标。

    

    在许多审阅工作流程中，判定结果是唯一被保留的信息。其背后的文本段落并未被标注，因为这类标注的成本远高于记录决策本身。我们测量了小语言模型在仅以判定结果进行后训练时（任何阶段都没有人工证据标注）能够恢复多少此类证据。在ContractNLI数据集上，人工证据跨度在评估之前一直被保留。匹配记录的判定与认同这些证据跨度并非同一回事：在六个系统中，这两个分数仅呈弱相关，且对系统的排名也不同，因此当引用必须可供审阅时，准确率是一个糟糕的指导指标。仅对裸判定结果进行标签训练可达到准确率0.896和跨度F1 0.564；拒绝采样（仅当生成轨迹的判定与记录匹配时才保留该轨迹，再依据自动的来源接地分数选出其中一条）达到0.797和0.556，而训练前的基线为0.747和0.493。

    arXiv:2610.06962v1 Announce Type: cross  Abstract: In many review workflows the verdict is the only thing retained. The passages behind it are not marked, because that annotation costs far more than recording the decision. We measure how much of that evidence a small language model can recover when it is post-trained on the verdicts alone, with no human evidence labels at any stage. On ContractNLI the human evidence spans are held out until evaluation. Matching the recorded verdict and agreeing with those spans are not the same thing: across six systems the two scores are only weakly related and rank the systems differently, so accuracy is a poor guide when the citations have to be reviewable. Label-only training on the bare verdict reaches accuracy 0.896 and span F1 0.564. Rejection sampling, which keeps a generated trace only when its verdict matches the record and then picks one by an automatic source-grounding score, reaches 0.797 and 0.556, against 0.747 and 0.493 before training.
    
[^333]: 学习决策而非推理：基于低秩激活转向的参数高效决策算子

    Learning to Decide, Not to Reason: Parameter-Efficient Decision Operators via Low-Rank Activation Steering

    [https://arxiv.org/abs/2610.06950](https://arxiv.org/abs/2610.06950)

    该论文提出一种仅用2.3万至33万参数、通过行为克隆训练的低秩激活转向决策算子，能在不损失精度的情况下将3685个token的长推理压缩为6个token的快速决策，训练成本比现有强化学习方法低约两个数量级。

    

    目前，向冻结的语言模型中注入技能需要百万级参数和一套强化学习流程。我们提出了\method{}，这是一个通过行为克隆训练的System-1决策算子，将这一成本降低了约两个数量级。默认算子仅使用33万参数即可匹敌一个通过强化学习训练的133万参数算子，性能相当或更优，同时将3685个token的深思推理压缩为6个token的决策且不损失精度。一个秩为4、仅有2.3万参数的变体——仅为已发表的最强技能算子的1/58——足以胜任SearchQA任务，并在LiveMath上接近足够（更高秩仍有帮助）；同一方法可迁移至五个任务和三个骨干模型，且在训练完成数月后发布的LiveMath问题上，分布外收益依然保持。与先前工作的差距在于可训练性，而这一差距由初始化和架构共同决定……（原文摘要在此处截断）

    arXiv:2610.06950v1 Announce Type: cross  Abstract: Injecting skills into a frozen language model currently costs a million parameters and a reinforcement-learning pipeline. We introduce \method{}, a System-1 decision operator trained by behavior cloning that lowers this cost by roughly two orders of magnitude. The default operator uses 330K parameters to match a 1.33M-parameter operator trained with reinforcement learning, exceeds or achieve comparable performance, while collapsing 3,685-token deliberation into a 6-token decision with no loss in accuracy. A rank-4 variant with 23K parameters, 1/58 of the strongest published skill operator, suffices for SearchQA and near-suffices for LiveMath, where higher rank still helps; the same recipe transfers across five tasks and three backbones, with out-of-distribution gains persisting on LiveMath problems released months after training. The gap to prior work is trainability, and it is set jointly by initialization and architecture: the initia
    
[^334]: 视觉Transformer中用于抽象概念接地的转喻电路

    Metonymic Circuits for Abstract Concept Grounding in Vision Transformers

    [https://arxiv.org/abs/2610.06928](https://arxiv.org/abs/2610.06928)

    视觉Transformer通过具体可解释的锚定概念（如“火”）作为转喻中介来接地抽象概念（如“愤怒”），形成了从感知基元到物体锚点再到抽象目标的结构化转喻电路。

    

    我们研究了当训练数据提供有限的直接指称证据时，视觉Transformer如何接地抽象概念（例如“愤怒”）。我们假设存在一种转喻接地机制，其中抽象预测由具体、可解释的锚定概念（例如“火”）驱动，这些锚定概念在视觉信号与抽象语义之间架起桥梁。通过在CLIP和DINO视觉编码器上应用转码器，我们恢复了可与更具体概念的语义标签相关联的中间特征，并追踪它们在抽象概念识别底层电路中的贡献。在一个精心构建的图标数据集上的实验揭示了结构化的转喻电路，其中感知基元主导早期层，类物体的锚定概念先于抽象目标出现。包含渲染文本的图像则会启用一条独特的感知到文本的通路。因果干预进一步验证了转喻中间体在接地过程中发挥功能性作用。

    arXiv:2610.06928v1 Announce Type: new  Abstract: We study how Vision Transformers ground abstract concepts (e.g., angry) when training data provide limited direct referential evidence. We hypothesize a metonymic grounding mechanism in which abstract predictions are driven by concrete, interpretable anchor concepts (e.g., fire) that bridge visual signals to abstract semantics. By applying Transcoders on CLIP and DINO vision encoders, we recover intermediate features that can be associated with semantic labels for more concrete concepts, and trace their contributions in circuits underlying abstract concept recognition. Experiments on a carefully curated icon dataset reveal structured metonymic circuits, in which perceptual primitives dominate early layers and object-like anchors precede abstract targets. Images containing rendered text instead recruit a distinct perceptual-to-textual route. Causal interventions further validate that metonymic intermediates are functionally involved in gr
    
[^335]: AttSVD：基于注意力引导SVD的提示自适应低秩KV缓存压缩

    AttSVD:Prompt-Adaptive Low-Rank KV Cache Compression via Attention-Guided SVD

    [https://arxiv.org/abs/2610.06927](https://arxiv.org/abs/2610.06927)

    提出AttSVD，一种基于每个提示自身注意力几何结构的可解释低秩KV缓存压缩方法，通过在线逐提示截断SVD仅保留注意力实际读取的方向，在保留全部token的同时按保留秩比例削减长上下文下的KV缓存内存，并提供累积式与流式两种解码时缓存策略及自适应压缩改进。

    

    自回归Transformer的键值（KV）缓存随上下文长度线性增长，并在长上下文场景下占据主要内存。大多数无需训练的补救方法会选择驱逐低重要性的token，这是沿序列轴上一种不可逆的选择。我们反其道而行之，保留每一个token，并沿“特征”轴以更廉价的方式存储。为此，我们提出AttSVD，一种新的“可解释”低秩压缩方法，其基向量源自每个提示自身的注意力几何结构：通过一种在线的、逐提示的截断SVD，仅保留注意力实际读取的方向，并按保留秩的比例削减每个注意力头的持久KV内存占用。我们提出了两种解码时缓存策略——累积式和流式——分别适用于短生成和长生成场景。此外，我们提出两项改进使压缩具有自适应性：逐矩阵能量规则可独立地为logit空间和注意力质量确定尺寸；注意力感知的基仅在注意力……（原文在此处截断）

    arXiv:2610.06927v1 Announce Type: cross  Abstract: The key-value (KV) cache of autoregressive transformers grows linearly with context length and dominates memory at long context. Most training-free remedies evict low-importance tokens, an irreversible choice along the sequence axis. We instead keep every token and store it more cheaply along the "feature" axis. We therefore propose AttSVD, a new "interpretable" low-rank compression whose basis is derived from each prompt's own attention geometry: an online, per-prompt truncated SVD that keeps only the directions attention actually reads, cutting persistent per-head KV memory in proportion to the retained rank. We propose two decode-time caching strategies, accumulating and streaming, for short and long generation regimes. Furthermore, we propose two refinements that make compression adaptive. A per-matrix energy rule sizes the logit space and the attention mass independently. An attention-aware basis truncates only in the spaces atten
    
[^336]: RadOnc-Agent：一个贯穿放射治疗诊疗路径、由大语言模型编排的AI工作流框架

    RadOnc-Agent: An LLM-Orchestrated Framework for AI Workflows Across the Radiotherapy Care Pathway

    [https://arxiv.org/abs/2610.06923](https://arxiv.org/abs/2610.06923)

    该论文提出RadOnc-Agent，一个由大语言模型驱动的智能体框架，通过对话界面将放射治疗形式化为四个临床阶段并提供26个可调用功能，从而将分散在不同临床阶段和软件环境中的AI能力统一编排为完整的放射治疗工作流。

    

    人工智能已经在单个放射治疗任务中取得进展，但这些能力在临床阶段、软件环境和数据模态之间仍然相互割裂。这种碎片化与从治疗决策到随访的纵向放射治疗工作流程形成鲜明对比。我们在此提出RadOnc-Agent，这是一个智能体化人工智能框架，它将放射治疗形式化为四个临床阶段，并通过对话式界面提供26个可调用功能。一个大语言模型控制器将临床意图映射到受模式约束的调用中，保留患者和工作流上下文，并将请求路由到专业服务。我们使用2,600个单功能请求（7,800次重复执行）、200个预先指定的跨四阶段合成场景（600次执行），以及来自60份去标识化患者记录的120个工作流实例（360次干净执行）来评估系统的执行表现。

    arXiv:2610.06923v1 Announce Type: new  Abstract: Artificial intelligence has advanced individual radiotherapy tasks, yet these capabilities remain separated across clinical stages, software environments and data modalities. This fragmentation contrasts with the longitudinal radiotherapy workflow from treatment decision-making through follow-up. Here we present RadOnc-Agent, an agentic artificial-intelligence framework that formalizes radiotherapy into four clinical phases and provides 26 callable functions through a conversational interface. A large-language-model controller maps clinical intent to schema-constrained calls, preserves patient and workflow context, and routes requests to specialist services. We evaluated system execution using 2,600 single-function requests (7,800 repeat executions), 200 prespecified synthetic cross-stage scenarios spanning four phases (600 executions), and 120 workflow instances from 60 de-identified patient records (360 clean executions) representing d
    
[^337]: 对比学习中语义几何的锚点散度

    Anchor Divergence for Semantic Geometry in Contrastive Learning

    [https://arxiv.org/abs/2610.06919](https://arxiv.org/abs/2610.06919)

    本文提出“锚点散度”方法，通过建立锚点概率分布与Bregman几何之间的对应关系，使固定表示上的语义几何能够适配特定上下文，突破了余弦相似度单一固定几何的局限。

    

    本文研究语义上下文如何决定学习到的向量表示中的几何结构。相似度通常使用余弦相似度来衡量，这种方式提供了一种单一固定的几何。然而，语义相似度本质上依赖于上下文：两幅图像可能相似是因为它们描绘了同一物体、共享某种视觉风格，或与同一临床发现相关。我们证明对比表示天然地涵盖了一族几何结构，这些几何结构可以被专门化以匹配特定的语义结构。关键思想是利用对比学习、指数族分布和信息几何三者之间的相互作用，在“锚点”上的概率分布与表示空间上的Bregman几何之间建立对应关系。我们利用这种对应关系定义了“锚点散度”，这是一种在固定表示上指定特定于上下文的语义几何的方法。在这种对应关系下，模（原文摘要在此处截断）

    arXiv:2610.06919v1 Announce Type: new  Abstract: This paper concerns how semantic context determines geometry in learned vector representations. Similarity is typically measured using cosine similarity, which provides a single fixed geometry. Semantic similarity, however, is inherently context dependent: two images may be similar because they depict the same object, share a visual style, or are relevant to the same clinical finding. We show that contrastive representations naturally encompass a family of geometries that can be specialized to particular semantic structure. The key idea is to use an interplay between contrastive learning, exponential families, and information geometry to establish a correspondence between probability distributions over "anchors" and Bregman geometries on the representation space. We use this correspondence to define "Anchor Divergences", a method for specifying context-specific semantic geometries on fixed representations. Under this correspondence, mode
    
[^338]: FluidPD：面向SLO感知的预填充-解码分离式LLM服务的原位弹性机制

    FluidPD: In-Place Elasticity for SLO-Aware Prefill-Decode Disaggregated LLM Serving

    [https://arxiv.org/abs/2610.06917](https://arxiv.org/abs/2610.06917)

    FluidPD是一个SLO感知的预填充-解码分离式LLM服务系统，通过两个互补机制实现原位弹性，无需备用GPU即可应对预填充与解码需求比例的短时突发和持续偏移，避免延迟SLO违规。

    

    预填充-解码分离（P/D disaggregation）正在成为LLM服务的常见架构，因为它将具有不同执行模式和SLO目标的两个阶段分离开来。现有系统通常采用固定的预填充/解码工作节点比例，并结合跨工作节点的请求路由。然而，真实世界的工作负载在预填充与解码需求比例上既存在短时突发，也存在持续性偏移。因此，某一时刻配置良好的资源比例可能很快变得不匹配，即使其他地方存在空闲容量，也会导致延迟SLO违规。现有的自动扩缩容机制虽然可以增加容量，但它们反应缓慢、需要备用GPU，并且无法直接解决短时间尺度的阶段不平衡问题。我们提出了FluidPD，一个提供SLO感知原位弹性的P/D分离式服务系统。FluidPD引入了两个互补机制：FluidToken通过卸载有限部分的预填充计算来处理瞬时不平衡……（摘要在此处截断）

    arXiv:2610.06917v1 Announce Type: new  Abstract: Prefill-decode disaggregation is becoming a common architecture for LLM serving because it separates two phases with distinct execution patterns and SLO objectives. Existing systems typically combine a fixed prefill/decode worker ratio with request routing across workers. However, real-world workloads exhibit both short bursts and sustained shifts in the prefill-to-decode demand ratio. As a result, a configuration that is well provisioned at one time may quickly become mismatched, causing latency SLO violations even when idle capacity exists elsewhere. Existing autoscaling mechanisms can add capacity, but they react slowly, require spare GPUs, and do not directly address short-timescale phase imbalance.   We present FluidPD, a P/D-disaggregated serving system that provides SLO-aware in-place elasticity. FluidPD introduces two complementary mechanisms. FluidToken handles transient imbalance by offloading a bounded portion of prefill compu
    
[^339]: Text2Dashboard：基于企业DataBrain的自然语言仪表板生成受治理智能体架构

    Text2Dashboard: A Governed Agent Architecture for Natural-Language Dashboard Generation over Enterprise DataBrain

    [https://arxiv.org/abs/2610.06914](https://arxiv.org/abs/2610.06914)

    提出了Text2Dashboard——一种受治理的智能体架构，通过将模式约束的模型决策与类型化工具、确定性Hooks相结合，把自然语言分析请求安全地转换为可审计、可检查的企业数据仪表板，在真实DataBrain任务上取得了较高成功率。

    

    Text2Dashboard是一个针对DataBrain的原型系统，能够将自然语言分析请求转换为可检查的仪表板。一个可安装的Codex插件和独立的Agent Runtime将模式约束的模型决策与类型化工具、持久状态以及用于审批、审计、检查点、恢复和故障处理的确定性Hooks相结合。该流水线负责实体解析、元数据发现、强制只读SQL、仪表板组装，并应用静态检查、动态预检和浏览器检查。模型负责提出操作，而确定性软件控制执行并记录状态转换。我们在冻结的真实DataBrain任务和受控Hook故障上对该工作流进行了评估。元数据和SQL任务的严格成功率为6/8：元数据选择通过4/4，全部四个SQL任务均满足语义标准，其中2/4满足精确的输出列契约。最终版本通过了4/4的单面板仪表板任务，以及一个双面板任务（原文在此处截断）。

    arXiv:2610.06914v1 Announce Type: new  Abstract: Text2Dashboard is a DataBrain-specific prototype that turns natural-language analytic requests into inspectable dashboards. An installable Codex plugin and standalone Agent Runtime combine schema-constrained model decisions with typed tools, persistent state, and deterministic Hooks for approval, audit, checkpointing, recovery, and failure handling. The pipeline resolves entities, discovers metadata, enforces read-only SQL, composes dashboards, and applies static checks, dynamic preflight, and browser inspection. The model proposes actions while deterministic software controls execution and records state transitions.   We evaluate the workflow on frozen real-DataBrain tasks and controlled Hook faults. Strict success was 6/8 on metadata and SQL tasks: metadata selection passed 4/4, all four SQL tasks met semantic criteria, and 2/4 met the exact output-column contract. The final release passed 4/4 single-panel dashboard tasks, one two-pane
    
[^340]: GAMEGO：基于真实世界资产锚定的合成轨迹训练游戏开发智能体

    GAMEGO: Training Game-Dev Agents with Synthetic Trajectories Anchored in Real-World Assets

    [https://arxiv.org/abs/2610.06910](https://arxiv.org/abs/2610.06910)

    GameGo 提出了一个可扩展的框架，通过将简短的游戏种子系统性地转化为基于行业游戏开发实践的产品需求文档，并利用锚定于真实世界资产的合成轨迹来训练游戏开发智能体，从而实现高质量的端到端浏览器游戏自动生成。

    

    大型语言模型（LLMs）的最新进展已在网页前端执行方面展现出卓越能力，其中基于浏览器的游戏生成成为一个尤为突出的前沿方向。以往的工作通常依赖于复杂的多轮工作流，或专注于静态的游戏评估基准，而本工作则瞄准由编码智能体驱动的直接端到端真实世界游戏合成。然而，直接从稀疏的用户查询生成复杂游戏，往往会迫使编码智能体做出欠规范的假设，导致游戏机制不完整、玩法流程脱节以及视觉美学受限。为解决这一问题，本文提出了 GameGo，一个可扩展的框架，它系统性地将简短的游戏种子转化为基于行业游戏开发实践的全面产品需求文档（PRD）。为了在不限制设计探索的前提下保留核心玩法约束，GameGo 使用任务特定的动态（原文此处截断）……

    arXiv:2610.06910v1 Announce Type: new  Abstract: Recent advances in Large Language Models (LLMs) have demonstrated remarkable capabilities in web front-end execution, with browser-based game generation emerging as a particularly prominent frontier. While previous efforts frequently rely on complex multi-turn workflows or focus on static game evaluation benchmarks, this work targets direct end-to-end real-world game synthesis driven by coding agents. However, generating complex games directly from sparse user queries often forces coding agents to make underspecified assumptions, yielding incomplete mechanics, disconnected gameplay flows, and limited visual aesthetics. To resolve this issue, this paper presents GameGo, a scalable framework that systematically transforms brief game seeds into comprehensive Product Requirements Documents grounded in industry game-development practices. To retain core gameplay constraints without restricting design exploration, GameGo uses task-specific dyn
    
[^341]: Transformer拒绝机制中的组件与维度稀疏性

    Component and Dimension Sparsity in Transformer Refusal Mechanisms

    [https://arxiv.org/abs/2610.06903](https://arxiv.org/abs/2610.06903)

    该研究通过对四个开源大语言模型的组件级干预分析，发现拒绝行为引导只需稀疏组件子集（占上游组件28%–48%）及其中约50%的残差流维度即可复现完整效果，揭示了拒绝机制在组件和维度两个层面上的稀疏性。

    

    激活引导通过干预大语言模型的内部激活来操纵其行为，但这些干预的机理基础仍知之甚少。我们将拒绝引导分解为跨四个开源权重模型的组件级干预，识别出稀疏的注意力与MLP组件子集，仅对这些子集进行引导就足以复现完整的行为效果。我们发现，拒绝方向集中于稀疏的组件机制中，这些组件仅占上游组件的28%–48%，却能保留88%–101%的引导有效性。在这些机制内部，有效引导进一步集中于约50%的残差流维度，保留85%–98%的组件机制基线效果，这与特权基结构相一致。因此，稀疏性在两个层面上发挥作用：哪些组件被引导，以及这些组件内部哪些维度承载信号。总之，这些发现表明……

    arXiv:2610.06903v1 Announce Type: cross  Abstract: Activation steering manipulates large language model behavior by intervening on internal activations, but the mechanistic basis of these interventions remains poorly understood. We decompose refusal steering into component-level interventions across four open-weight models, identifying the sparse subsets of attention and MLP components whose steering suffices to reproduce the full behavioral effect. We find that refusal directions concentrate in sparse component mechanisms comprising 28--48\% of upstream components, retaining 88--101\% of steering effectiveness. Within these mechanisms, effective steering further concentrates in approximately 50\% of residual stream dimensions, retaining 85--98\% of the component-mechanism baseline, consistent with a privileged basis structure. Sparsity thus operates at two levels: which components are steered, and which dimensions within those components carry the signal. Together these findings show 
    
[^342]: 对齐中线性奖励的公理可满足性

    Axiom Satisfiability of Linear Rewards in Alignment

    [https://arxiv.org/abs/2610.06892](https://arxiv.org/abs/2610.06892)

    该论文通过引入逐候选者松弛量，提出一种计算“总松弛最小且满足公理边际η”的线性奖励的方法，在不对投票者和数据收集方式做任何假设的情况下，以被O(1)界定的松弛代价强制线性奖励满足帕累托最优与PMC等公理。

    

    从人类偏好数据中学习是使语言模型与人类价值观对齐的主流途径。在线性社会选择设定中，奖励是“提示-回答”对的固定特征表示的线性函数。Ge等人[2024]证明，通过最小化任何非递减凸损失（包括BTL）来拟合此类奖励，会违反帕累托最优（PO）与PMC公理；此外，一旦要求输出必须由线性模型诱导，任何仅读取多数关系的规则都无法满足PO。我们追问：无论如何强制满足这些公理的代价是什么？为此，我们将线性模型进行松弛，允许每个候选者拥有各自的松弛量。我们计算在满足公理（并带有边际η，即两个奖励值之间所需的最小差值）条件下总松弛量最小的松弛线性奖励。我们的解无需对投票者或比较数据的收集方式做任何假设即可满足这些公理。当η至多为O(1/m²…)（摘要在此处截断）时，我们将最优总松弛量界定为O(1)。

    arXiv:2610.06892v1 Announce Type: cross  Abstract: Learning from human preference data is the dominant route to aligning language models with human values. In linear social choice, where rewards are linear in a fixed feature representation of prompt-response pairs, Ge et al.[2024] show that fitting such a reward by minimizing any non-decreasing convex loss, including BTL, fails PO and PMC. Moreover, no rule that reads only the majority relation can satisfy PO once the output is required to be linearly induced. We ask what it costs to enforce these axioms anyway. To this end, we relax the linear model to allow per-candidate slack. We compute the relaxed linear reward with the smallest total slack that satisfies the axioms with a margin $\eta$, the minimum required difference between two reward values. Our solution satisfies the axioms under no assumptions about the voters or how comparisons were collected. We bound the optimal total slack by $O(1)$ when $\eta$ is at most $O(\frac{1}{m^2
    
[^343]: 零样本可视化：基于用户提示轴的文本语料库探索

    Zero-Shot Visualization: Exploring Text Corpora with User-Prompted Axes

    [https://arxiv.org/abs/2610.06889](https://arxiv.org/abs/2610.06889)

    该论文提出了零样本可视化（ZSV）任务，允许用户通过自然语言指定概念轴来交互式探索文本语料库，并通过基准测试发现基于下一个词元概率的评分方法在语义忠实性、评分保真度和计算成本方面具有优势。

    

    我们研究了大语言模型（LLM）在文本语料库可视化探索中的应用。我们提出了零样本可视化（ZSV）这一任务，即用户用自然语言指定概念，然后将文档映射到相应的概念轴上进行可视化。构建一个具有实用价值的ZSV系统并非易事，因为它需要在特征函数、高效实现的权衡以及影响可视化质量的预处理/后处理决策的交汇处做出选择。为此，我们建立了一个基准，比较了在此设置下涵盖嵌入相似度、直接语义判断和条件似然估计等多种方法。我们在多个数据集和用例上，从语义忠实性、评分保真度和计算成本等方面评估了不同评分方法和设计选择的特性。我们的结果表明，基于下一个词元概率的评分方法提供了……

    arXiv:2610.06889v1 Announce Type: cross  Abstract: We study the application of large language models (LLMs) to the visual exploration of textual corpora. We introduce zero-shot visualization (ZSV), a task in which users specify concepts in natural language and documents are mapped onto the corresponding concept axes for visualization. Building a ZSV system of practical value is non-trivial, as it requires choices at the intersection of feature functions, efficient implementation tradeoffs, and pre/post-processing decisions affecting visualization quality. To that end, we establish a benchmark that compares methods spanning embedding similarity, direct semantic judgments, and conditional likelihood estimation in this setting. Across multiple datasets and use cases we evaluate the properties of different scoring methods and design choices in terms of semantic faithfulness, score fidelity, and computational cost. Our results identify that scoring based on next-token probabilities offers t
    
[^344]: 高级持续性威胁与移动目标防御之间随机博弈的动态低秩均衡计算

    Dynamical low-rank equilibrium computation for stochastic games between advanced persistent threats and moving target defense

    [https://arxiv.org/abs/2610.06885](https://arxiv.org/abs/2610.06885)

    该论文揭示了工业控制系统中APT攻击与MTD防御的影响矩阵具有内在低秩性，并证明该低秩结构可通过零和随机博弈的非光滑贝尔曼算子传播，从而在显式误差界保证下实现高效且鲁棒可认证的均衡计算。

    

    在工业控制系统（ICS）中针对高级持续性威胁（APT）的移动目标防御（MTD）已有成熟的博弈论建模，但其实际价值取决于均衡计算：满秩值迭代在工业状态维度下计算代价过于高昂，且由此得到的防御策略对对抗性扰动不具备可认证的鲁棒性。我们首先揭示了ICS动力学的攻击与防御影响矩阵具有内在低秩性：APT仅通过少数入口点渗透，而MTD每个周期只重新配置有限的组件子集。我们证明这种结构会通过零和随机博弈的非光滑贝尔曼算子传播：一个连接物理低秩与算法低秩的增广梯度矩阵证明了每个贝尔曼目标都位于低维子空间附近，并对最优值函数给出了显式的误差界。

    arXiv:2610.06885v1 Announce Type: cross  Abstract: Moving target defense (MTD) against advanced persistent threats (APTs) in industrial control systems (ICS) has well-established game-theoretic formulations, but their practical value hinges on equilibrium computation: full-rank value iteration is prohibitively expensive at industrial state dimensions, and the resulting defense strategies admit no certified robustness against adversarial perturbations. We first reveal that the attack and defense influence matrices of ICS dynamics are intrinsically low-rank: APTs infiltrate through a handful of entry points, and MTD reconfigures only a limited subset of components per cycle. We prove that this structure propagates through the non-smooth Bellman operator of the zero-sum stochastic game: an augmented gradient matrix bridging physical and algorithmic low rank certifies that every Bellman target lies near a low-dimensional subspace, with an explicit error bound on the optimal value function.
    
[^345]: 学习何时细化：面向预算约束神经算子偏微分方程求解器的长时程强化学习

    Learning When to Refine: Long-Horizon Reinforcement Learning for Budgeted Neural-Operator PDE Solvers

    [https://arxiv.org/abs/2610.06883](https://arxiv.org/abs/2610.06883)

    该论文提出面向预算约束神经算子PDE求解的长时程强化学习方法RV-PI，通过实际滚动验证在有限修正预算下学习何时何地施加局部细化，仅在留出轨迹误差改善时接受策略更新。

    

    神经算子为时间依赖的偏微分方程提供了快速代理模型，但自回归部署带来了细化分配问题：预测误差随空间和时间变化，而沿轨迹只能进行有限次数的局部修正。我们将该问题形式化为预算约束的自适应神经算子求解。全局傅里叶神经算子推进整个场，局部算子提出分块残差修正，集合感知的选择器决定在哪里进行细化，宏策略则决定何时以及花费多少剩余的细化预算。我们提出了滚动验证策略改进（RV-PI），它通过学习到的偏微分方程求解器的实际延续滚动来评估可行的修正次数，将长时程优势转化为保守的策略目标，并且仅当留出轨迹误差得到改善时才接受更新。在具有32次干预预算的浅水方程基准测试中，RV-P……（摘要在此处被截断）

    arXiv:2610.06883v1 Announce Type: cross  Abstract: Neural operators provide fast surrogates for time-dependent PDEs, but autoregressive deployment creates a refinement-allocation problem: prediction errors vary over space and time, while only a finite number of local corrections can be committed along a trajectory. We formulate this as budgeted adaptive neural-operator solving. A global Fourier neural operator advances the full field, a local operator proposes patch-wise residual corrections, and a set-aware selector chooses where to refine. A macro policy decides when and how much of the remaining refinement budget to spend. We introduce rollout-verified policy improvement (RV-PI), which evaluates feasible refinement counts through actual continuation rollouts of the learned PDE solver, converts long-horizon advantages into conservative policy targets, and accepts an update only when held-out trajectory error improves. On the shallow-water benchmark with a 32-intervention budget, RV-P
    
[^346]: 用于建筑热负荷短期预测的混合预测模型比较综述

    Comparative review of hybrid forecasting models for short-term prediction of building thermal load

    [https://arxiv.org/abs/2610.06881](https://arxiv.org/abs/2610.06881)

    本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。

    

    本文对不同混合模型在建筑热需求短期预测中的表现进行了比较综述。特别地，该评估针对的是与其他最先进技术相结合的增强型数据驱动模型之间的比较。第一步，分析了文献中已报道的现有技术，结论是元启发式算法或数据驱动模型被用于识别基础模型的参数。定性评估包括每种方法的输入和输出特征、主要优点和缺点。第二步，利用苏格兰家庭历史热需求数据集以及历史天气预报，进一步评估了现有混合方法的性能。在对13种混合方法的评估中，经验模态分解-长短期记忆-马尔可夫（EMD-LSTM-Markov）模型能够以最高的精度进行预测……

    arXiv:2610.06881v1 Announce Type: cross  Abstract: In this paper, a comparative review of different hybrid models for short-term forecasting of building thermal demand is carried out. Particularly, the assessment tackles the comparison of data-driven models enhanced with other state-of-the-art techniques. At the first step, the existing techniques reported in the literature are analysed. It is concluded that Metaheuristics or a data-driven model are used to identify the parameters of the basic model. The qualitative evaluation includes for each method the input and output features, main advantages and drawbacks. At the second step, an existing dataset of historical thermal demand from Scottish households, as well as historical weather forecasts are utilized to assess additionally the performance of existing hybrid methods. From the assessment of 13 hybrid methods, the Empirical Modal Decomposition - long short-term memory - Markov (EMD-LSTM-Markov) model can predict with the highest ac
    
[^347]: 基于中智学集成分类的不确定性感知轴承故障检测：来自实验室与变速工业基准的证据

    Neutrosophic Ensemble Classification for Uncertainty-Aware Bearing Fault Detection: Evidence from Laboratory and Variable-Speed Industrial Benchmarks

    [https://arxiv.org/abs/2610.06880](https://arxiv.org/abs/2610.06880)

    本文提出将中智学四指标分解（最高类证据、最佳竞争者证据、预测熵与决策分歧）应用于机器学习集成分类器，以区分轴承故障检测中自信的错误与真正模糊的预测，实现不确定性感知的故障诊断，并在实验室与变速工业基准上验证了其有效性。

    

    arXiv:2610.06880v1 发布类型：cross。摘要：用于轴承故障检测的机器学习分类器输出标量置信度分数，将自信的错误与真正模糊的预测混为一谈，而传统的真/假二元组（F = 1 - T）在构造上存在代数冗余。我们通过精炼的中智学分解，将随机森林 + XGBoost + 逻辑回归集成分类器操作化为四个指标——T-hat（最高类别证据）、F-hat（最佳竞争者证据）、预测熵 I1-hat 和决策分歧 I2-hat——并在两个轴承基准数据集（CWRU 和 JNU，600-1000 rpm）上采用留一工况协议进行评估。在 CWRU 数据集上，在纠正了文件与类别之间的映射错误后，该集成分类器在四个留出负载中的三个上达到 100.00% 的准确率（第四个为 92.27%），导致可用于不确定性分析的错误样本过少。在 JNU 数据集上，留出 1000 rpm 工况后，准确率骤降至 40.64%，低于多数类基线；逻辑回归……

    arXiv:2610.06880v1 Announce Type: cross  Abstract: Machine learning classifiers for bearing fault detection produce scalar confidence scores that conflate confident errors with genuinely ambiguous predictions, and the conventional truth/falsity pair (F = 1 - T) is algebraically redundant by construction. We operationalize a refined neutrosophic decomposition of a Random Forest + XGBoost + Logistic Regression ensemble into four indicators -- T-hat (top-class evidence), F-hat (best-competitor evidence), predictive entropy I1-hat, and decision disagreement I2-hat -- evaluated on two bearing benchmarks (CWRU and JNU, 600-1000 rpm) under a leave-one-condition-out protocol. On CWRU, after correcting a file-to-class mapping error, the ensemble reaches 100.00 percent accuracy on three of four held-out loads (92.27 percent on the fourth), leaving too few errors for uncertainty analysis. On JNU, holding out 1000 rpm, accuracy collapses to 40.64 percent, below a majority-class baseline; Logistic 
    
[^348]: 世界模型何时能够恢复物理定律？

    When Can World Models Recover Physical Laws?

    [https://arxiv.org/abs/2610.06877](https://arxiv.org/abs/2610.06877)

    本文指出仅凭准确预测不能证明世界模型恢复了物理定律，并给出定律可恢复的充要条件——任意两条不同定律必须在实验上可区分——同时用率-失真下界、有限响应码本、明确解码预算和极小极大采样复杂度刻画了恢复的信息论极限与稳定性。

    

    准确的预测并不能证明一个世界模型已经恢复了物理定律：在相同的观测协议下，不同的动力学可能生成完全相同的记录。本文在明确的实验目录、传感器不确定性和采集预算的约束下，于一个固定的物理域上形式化地定义了定律恢复问题。一个率-失真逆定理将描述定律所需的信息与实验装置所能揭示的信息分离开来，其构造性对应结果则给出了一个有限的响应码本和明确的解码预算。在紧致的世界类上，一致恢复当且仅当每一对不同的定律在实验上可区分时才可能实现；等价地，即该装置能够恢复每个有限定律源的全部熵。文中还引入逆响应模量来刻画稳定性。对于d维状态-动作域上的Lipschitz场，重置后的带噪全状态读取需要Θ(ε^...（摘要在此处截断）的极小极大预算。

    arXiv:2610.06877v1 Announce Type: cross  Abstract: Accurate prediction does not establish that a world model has recovered a physical law: distinct dynamics can generate identical records under the same observation protocol. We formulate law recovery on a fixed physical domain under an explicit catalog of experiments, sensor uncertainty, and an acquisition budget. A rate--distortion converse separates the information needed to describe a law from the information the apparatus can reveal. Its constructive counterpart gives a finite response codebook and an explicit decoding budget. On compact world classes, uniform recovery is possible exactly when every pair of different laws is experimentally distinguishable; equivalently, the apparatus can recover all the entropy of every finite law source. An inverse response modulus quantifies stability. For Lipschitz fields on a $d$-dimensional state--action domain, noisy full-state readouts after resets require minimax budget $\Theta(\varepsilon^
    
[^349]: TasteVal：衡量AI系统相对于人类专家的实验研究品味

    TasteVal: Measuring the Experimental Research Taste of AI Systems Against Human Experts

    [https://arxiv.org/abs/2610.06824](https://arxiv.org/abs/2610.06824)

    提出TasteVal基准，将AI的实验研究品味量化为计算效率，用于评估前沿模型在固定研究问题上迭代设计实验并得出结论的能力，使其成为预测AI进展的关键参数。

    

    我们介绍了TasteVal，一个用于评估前沿模型实验研究品味的基准。我们将研究品味定义为挑选有趣问题加以解决、设计实验并解读实验结果的能力。TasteVal衡量的是研究品味中的实验部分；在给定固定研究问题的前提下，我们测量模型迭代设计实验并从实验结果中得出结论的水平。我们将实验研究品味操作化为计算效率：如果一个研究者仅使用一半的串行实验计算量就达到与人类专家相同的分数，则其实验品味为专家的两倍。因此，实验品味充当实验计算量的乘数，使其成为预测AI进展的关键输入。TasteVal由8个新颖、具有挑战性且开放式的任务组成，代表了前沿AI研发。为了将品味与编程能力分离，被评估的模型充当研究者角色（原文摘要至此截断）。

    arXiv:2610.06824v2 Announce Type: replace  Abstract: We introduce TasteVal, a benchmark to evaluate the experimental research taste of frontier models. We define research taste as the ability to pick interesting problems to solve, design experiments, and interpret experimental results. TasteVal measures the experimental component of research taste; given a fixed research problem, we measure how well a model iteratively designs experiments and draws conclusions from their outcomes. We operationalize experimental research taste as compute efficiency; a Researcher who reaches the same score as an expert human using half the serial experimental compute has twice the experimental taste. Experimental taste thus acts as a multiplier on experimental compute, making it a key input to forecasts of AI progress. TasteVal consists of 8 novel, challenging, open-ended tasks representative of frontier AI R&D. To isolate taste from coding ability, the model under evaluation acts as a Researcher that it
    
[^350]: TAPDreamer：针对世界动作模型的可迁移对抗补丁

    TAPDreamer: Transferable Adversarial Patches for World Action Models

    [https://arxiv.org/abs/2610.06814](https://arxiv.org/abs/2610.06814)

    提出了TAPDreamer攻击方法，仅利用公开编码器即可构建无需查询目标模型的对抗补丁，实现跨任务和跨动作架构迁移地攻击世界动作模型。

    

    世界模型通过学习预测环境将如何演变，成为通用机器人控制的重要基础。然而，世界动作模型依赖于摄像头输入，而对摄像头输入的操纵可能会破坏跨任务和动作策略所使用的视觉表示。针对这些模型的现有攻击需要对受害者的动作或预测的未来进行优化，因此需要访问目标模型的输出。在本文中，我们提出了一种针对世界动作模型的攻击方法 TAPDreamer，它仅使用公开的编码器来构建一种固定的局部扰动，该扰动可以跨任务和动作架构进行迁移。TAPDreamer 无需查询目标策略。我们的关键洞察是：补丁引起的注意力权重变化与值向量之间的相互作用，会将几乎相同的表示偏移传播到远超补丁覆盖范围之外，且这种偏移在各种任务观测中保持稳定。在……的指导下（原文摘要在此处截断）

    arXiv:2610.06814v2 Announce Type: replace-cross  Abstract: World models learn to predict how their environment will evolve, making them an important foundation for general-purpose robotic control. Yet world action models depend on camera inputs whose manipulation can corrupt the visual representations used across tasks and action policies. Existing attacks on these models optimize against the victim's actions or predicted futures and therefore require access to target-model outputs. In this paper, we propose an attack, TAPDreamer, against world action models that instead uses a public encoder alone to construct a fixed local perturbation that transfers across tasks and action architectures. TAPDreamer requires no target-policy queries. Our key insight is that interactions between patch-induced changes in attention weights and value vectors broadcast a nearly identical representation shift far beyond the patch footprint, and this shift remains stable across task observations. Guided by 
    
[^351]: Local2Mesh：从稀疏2D心脏MRI进行左心室重建的空间局部化轮廓到网格方法

    Local2Mesh: Spatially Localized Contour-to-Mesh for Left Ventricular Reconstruction from Sparse 2D Cardiac MRI

    [https://arxiv.org/abs/2610.06052](https://arxiv.org/abs/2610.06052)

    提出空间局部化的Local2Mesh框架，通过几何感知对齐纠正切片错位、利用平面感知局部路由器将轮廓特征精确分配至模板顶点，从而无需3D网格标注即可从稀疏2D心脏MRI轮廓重建3D左心室几何结构。

    

    从稀疏的心脏磁共振（CMR）成像中进行三维（3D）左心室（LV）重建仍然具有挑战性，原因在于切片间存在错位以及切片之间局部空间信息不足。对轮廓特征进行全局聚合可能会模糊局部轮廓与表面之间的关系。我们提出了Local2Mesh，这是一个空间局部化的轮廓到网格框架，该框架通过对模板网格进行变形，从稀疏的2D轮廓中重建3D左心室几何结构，且无需3D网格标注。该框架引入了几何感知对齐来纠正切片间错位，并引入了一个平面感知的局部路由器，利用顶点到平面的距离将轮廓特征路由到模板顶点。随后，局部和全局轮廓特征共同指导基于图的模板变形，实现3D左心室重建。在两个公开数据集M&Ms-2和ACDC上的实验表明，该方法在几何重建和功能估计方面均优于现有方法。

    arXiv:2610.06052v2 Announce Type: replace-cross  Abstract: Three-dimensional (3D) left ventricular (LV) reconstruction from sparse cardiac magnetic resonance (CMR) imaging remains challenging due to inter-slice misalignment and insufficient local spatial information between slices. Global aggregation of contour features may obscure local contour-to-surface relationships. We propose Local2Mesh, a spatially localized contour-to-mesh framework that deforms a template mesh to reconstruct 3D LV geometry from sparse 2D contours without 3D mesh annotations. The framework introduces geometry-aware alignment to correct inter-slice misalignment and a plane-aware Local Router that routes contour features to template vertices using vertex-to-plane distances. Local and global contour features then jointly guide graph-based template deformation for 3D LV reconstruction. Experiments on two public datasets, M\&Ms-2 and ACDC, demonstrate superior geometric reconstruction and functional estimation over 
    
[^352]: 在线实验中的激励对齐

    Incentive Alignment in Online Experimentation

    [https://arxiv.org/abs/2610.05922](https://arxiv.org/abs/2610.05922)

    该论文将在线实验重新建模为激励设计问题，揭示了实验者基于有偏实验结果获得奖励所导致的委托-代理冲突会侵蚀平台价值，并证明样本拆分和收缩估计这两种实用机制能够有效实现激励对齐。

    

    arXiv:2610.05922v2 公告类型：replace-cross 摘要：评估新特征的因果效应是在线平台的核心目标。尽管近期文献通过集中式的组合优化来解决测试流量有限的问题，但这一视角忽略了一个关键的制度现实：实验在运营层面是去中心化的。开发新特征的实验者同时决定着要测试哪些假设，而他们通常基于容易产生向上偏差的经验平均处理效应来获得奖励。若不加约束，这种委托-代理冲突会严重侵蚀平台价值，这种结构性失效是传统的集中式手段（如显著性阈值和流量预算）所无法解决的。通过将实验重新构建为一个激励设计问题，我们证明了两种实用的机制——样本拆分与收缩估计——能够有效弥合这一鸿沟。样本拆分能以有限的流量成本实现完美的激励对齐……

    arXiv:2610.05922v2 Announce Type: replace-cross  Abstract: Evaluating the causal effect of new features is a central goal for online platforms. While recent literature addresses limited testing traffic via centralized portfolio optimization, this perspective abstracts away a critical institutional reality: experimentation is operationally decentralized. The experimenters who develop new features also dictate which hypotheses to test, and they are typically rewarded based on empirical average treatment effects that are prone to upward bias. Left unchecked, this principal-agent conflict can severely erode platform value, a structural failure that conventional centralized levers, such as significance thresholds and traffic budgets, cannot resolve. By reframing experimentation as an incentive design problem, we demonstrate that two practical mechanisms, sample splitting and shrinkage, can effectively bridge this gap. Sample splitting aligns incentives perfectly at a bounded traffic cost, w
    
[^353]: 用于调查模拟的数据驱动画像：关于不同数据访问机制下模拟一致性的洞察

    Data-Driven Personas for Survey Simulation: Insights into Simulation Alignment Across Data-Access Regimes

    [https://arxiv.org/abs/2610.05828](https://arxiv.org/abs/2610.05828)

    该研究发现，从域外公共行为数据诱导的画像在模拟特定人口群体的调查响应时很少优于仅使用基本人口统计信息的模拟，主要原因是源数据与目标人群不匹配。

    

    大语言模型（LLM）通过实现对调查响应的早期预测，为公众舆论研究提供了新机遇，有可能降低传统调查的成本和时间。然而，许多现有的引导方法依赖于目标领域的人类数据进行微调或提示，这些数据收集成本高昂且引发隐私问题。本文研究了人口群体层面的调查模拟，其中从异构的、匿名化的公共行为数据中诱导出的画像（personas）用于调节智能体，以模拟特定人口群体中个体的调查响应。我们检验了是否可以从多样化数据源诱导出具有代表性的画像，并分析了源数据的领域、规模和粒度如何影响调查模拟的一致性。我们发现，从域外数据源诱导的画像很少能优于仅基于基本人口统计信息进行条件化的模拟，这主要是由于人群不匹配所致。

    arXiv:2610.05828v2 Announce Type: replace  Abstract: Large language models (LLMs) offer new opportunities for public opinion research by enabling early prediction of survey responses, potentially reducing the cost and time of traditional surveys. However, many existing steering approaches rely on target-domain human data for fine-tuning or prompting that is costly to collect and raises privacy concerns. In this paper, we study demographic group-level survey simulation, where personas induced from heterogeneous, anonymized public behavioral data condition agents that simulate responses of individuals from specific demographic groups. We examine whether representative personas can be induced from diverse sources and analyze how the domain, scale, and granularity of the source data affect survey simulation alignment. We find that personas induced from out-of-domain sources rarely outperform simulations conditioned only on basic demographic information, largely due to population mismatch. 
    
[^354]: TeleTune：从离线遥测数据中演化智能体技能

    TeleTune: Evolving Agent Skills From Offline Telemetry

    [https://arxiv.org/abs/2610.05437](https://arxiv.org/abs/2610.05437)

    TeleTune提出了一个从无目标标注、无法重放且任务交错的离线用户遥测日志中，通过动作预测误差自动演化文本技能库的框架，使计算机使用智能体能够学到可复用的软件操作技能。

    

    计算机使用智能体需要捕捉人们如何使用软件的程序性知识，而用户遥测数据为这类知识提供了可扩展的来源。然而，从这些日志中学习可复用的技能需要解决三个挑战：（1）目标欠规范，因为日志不会记录每个动作背后的目标；（2）不可重放性，因为无法重放过去的活动来评估技能更新；（3）交错轨迹，因为日志可能混合多个任务且未标记其边界。为解决这些问题，我们提出了TeleTune，一个从离线日志中学习文本技能库的框架，这些日志没有记录目标、在优化过程中无法重放，且可能包含交错的任务。TeleTune利用日志轨迹上的动作预测误差来提出技能库的编辑建议，并仅保留那些能提升留出集动作预测准确率的编辑，我们称之为“技能引导的进展”。所学到的工作流还能实现……（摘要在此处被截断）。

    arXiv:2610.05437v2 Announce Type: replace  Abstract: Computer-use agents need to capture procedural knowledge of how people use software. User telemetry offers a scalable source of this knowledge. However, learning reusable skills from these logs requires addressing three challenges: (1) Goal Underspecification, since logs do not record the goal behind each action; (2) Non-Replayability, since past activity cannot be replayed to evaluate skill updates; and (3) Interleaved Trajectories, since logs may mix several tasks without marking their boundaries. To address these, we introduce TeleTune, a framework for learning a textual skill library from offline logs without recorded goals, cannot be replayed during optimization, and may interleave tasks. TeleTune uses action-prediction errors on logged trajectories to propose library edits and keep only those that improve held-out action-prediction accuracy, which we call skill-guided progress. The learned workflows also enable retrieval of dem
    
[^355]: EnGRICH：利用人类批评意见增强生成式奖励建模

    EnGRICH: Enhancing Generative Reward Modeling with Critiques from Humans

    [https://arxiv.org/abs/2610.05370](https://arxiv.org/abs/2610.05370)

    提出 EnGRICH 框架，通过从稀缺的人类批评意见中学习评估标准，并将其泛化到仅有结果监督的偏好数据上，从而提升生成式奖励模型批评意见的可靠性。

    

    生成式奖励模型（GRMs）对大语言模型优化至关重要。与标量奖励模型不同，GRMs 在做出偏好判断的同时还会生成自然语言批评意见，从而提供更细粒度的评估信号，其有效性在很大程度上取决于批评意见的可靠性。然而，现有的 GRM 训练通常以最终偏好判断的正确性作为结果监督；由于偏好结果空间高度受限，不可靠的批评意见仍可能得出正确的判断结果并因此被强化。近期的研究开始利用人类批评意见进行过程监督，但此类批评意见十分稀缺，且常被简化为标量奖励，导致其细粒度的评估信息未被充分利用。我们认为，从人类批评意见中学习到的评估标准可以泛化到更广泛的仅有结果监督的偏好数据上。为此，我们提出了 EnGRICH，这是一个将 GRM 与训练时的 MetaCr（元评论机制）相结合的 GRM 训练框架……

    arXiv:2610.05370v2 Announce Type: replace  Abstract: Generative reward models (GRMs) are important for LLM optimization. Unlike scalar reward models, GRMs generate natural-language critiques alongside preference judgments, providing finer-grained evaluation signals. Their effectiveness depends heavily on critique reliability. However, existing GRM training typically uses final preference correctness as outcome supervision. Because the preference outcome space is highly constrained, unreliable critiques can still yield correct outcomes and thus be reinforced. Recent work leverages human critiques for process supervision, but such critiques are scarce and are often reduced to scalar rewards, leaving their fine-grained evaluative information underutilized. We argue that evaluative criteria learned from human critiques can be generalized to broader outcome-only preference data. To this end, we propose \textbf{EnGRICH}, a GRM training framework that pairs the GRM with a training-time MetaCr
    
[^356]: 当智能体上下文过期时：易变智能体上下文中的不一致性问题

    When Agent Context Goes Stale: Incoherence in Volatile Agent Context

    [https://arxiv.org/abs/2610.05281](https://arxiv.org/abs/2610.05281)

    提出上下文一致性框架 Concord，通过将工具观测结果与其数据源关联、检测数据源变化，并在复用前自动更新、标注或抑制过时上下文，避免智能体基于过期信息做出错误判断。

    

    现代智能体越来越多地将其推理建立在工具返回的观测结果之上，例如从工作区读取的文件内容。然而，这些观测结果背后的数据源可能随后被用户、其他智能体或外部工具修改，而模型在其上下文窗口中仅保留着过时的内容。现有的智能体运行时几乎不提供任何机制来通知模型某个先前观测到的事实已经过时，导致智能体复用过时的观测结果，并对当前工作区状态做出错误的判断。我们提出了 Concord，一个上下文一致性框架，用于维护智能体上下文中的工具观测结果与其来源的可变数据源之间的一致性。Concord 将每个观测结果与其来源建立关联，检测来源的变化，并使用可配置的处理策略在复用过时上下文之前对其进行更新、标注或抑制。Concord 可适用于不同的智能体运行时环境。

    arXiv:2610.05281v2 Announce Type: replace  Abstract: Modern agents increasingly ground their reasoning in observations returned by tools, such as file contents read from a workspace. However, the data sources underlying these observations may later be modified by users, other agents, or external tools, while the model retains only the stale content in its context window. Existing agent runtimes provide little support for notifying the model that a previously observed fact has become stale, causing agents to reuse outdated observations and make incorrect claims about the current workspace state. We propose Concord, a context coherence framework that maintains the consistency between tool observation in agent context and the mutable sources from which they were derived. Concord links each observation to its source, detects source changes, and uses configurable handling policies to update, annotate, or suppress stale context before reuse. Concord is applicable across different agent runti
    
[^357]: 安全的动作还不够：面向视觉-语言-动作策略的可行未来解码

    A Safe Action Is Not Enough: Feasible-Future Decoding for Vision-Language-Action Policies

    [https://arxiv.org/abs/2610.05166](https://arxiv.org/abs/2610.05166)

    该论文揭示并解决了冻结VLA策略中的“可行性-似然差距”问题，提出通过近似计算候选动作的可行未来质量来进行解码，从而选出既局部安全又保有策略支持的安全任务完成路径的动作。

    

    arXiv:2610.05166v2 公告类型：replace。摘要：安全动作不一定是可行动作。冻结的视觉-语言-动作（VLA）策略可能会偏好一个局部可容许的动作，但该动作不会留下任何由策略支持的通往安全完成任务的路径。我们将其称为“可行性-似然差距”：似然用于对下一步动作进行排序，而可行性则取决于该动作所留存的未来可能性。为了将这些未来纳入决策，我们推导出了在安全任务完成约束下、以历史为条件的策略-环境轨迹律的精确下一块边际分布。该推导揭示了一种依赖于候选动作的可行未来质量：其支撑集记录了在冻结的延续过程中安全完成是否仍然可能，而其大小则衡量剩余的加权安全完成质量有多少。由于精确评估在线上不可行，我们开发了一种选择性有限候选近似方法，并建立了恢复最佳保留可行候选的条件。我们的报警触发的（摘要在此截断）

    arXiv:2610.05166v2 Announce Type: replace  Abstract: A safe action is not necessarily a viable one. A frozen vision-language-action (VLA) policy can favor a locally admissible move that leaves no policy-supported route to safe task completion. We call this the feasibility-likelihood gap: likelihood ranks the next move, while feasibility depends on the futures it leaves open.   To bring those futures into the decision, we derive the exact next-block marginal of the history-conditioned policy-environment trajectory law restricted to safe task completion. The derivation reveals a candidate-dependent feasible-future mass: its support records whether safe completion remains possible under the frozen continuation process, while its magnitude measures how much weighted safe-completion mass remains. Since exact evaluation is impractical online, we develop a selective finite-candidate approximation and establish conditions for recovering the best retained viable candidate.   Our alarm-triggered
    
[^358]: 面向测试时扩散对齐的 Best-of-N 引导方法

    Best-of-$N$ Guidance for Test-time Diffusion Alignment

    [https://arxiv.org/abs/2610.05108](https://arxiv.org/abs/2610.05108)

    该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。

    

    扩散模型具有强大的生成性能，但常常难以使生成的样本与通过奖励模型衡量的人类偏好对齐。一种简单而有效的测试时对齐算法是 Best-of-N (BoN) 采样，它从预训练的扩散模型中抽取 N 个独立同分布的样本，并输出奖励最高的单个样本。尽管 BoN 在实践中取得了成功，但它对奖励信息的利用十分有限，因为奖励信息仅在最终选择阶段才被纳入，而在采样过程中并不影响逆向扩散轨迹。因此，BoN 采样并不能提升生成样本的平均对齐程度，且主要适用于单输出场景。我们提出了 Best-of-N 引导（BoNG），这是一种将 BoN 采样原理直接融入逆向扩散过程的新颖方法。BoNG 对去噪粒子执行在线 BoN 选择，并据此调整逆向扩散过程（摘要内容在此处被截断）。

    arXiv:2610.05108v2 Announce Type: replace-cross  Abstract: Diffusion models achieve strong generative performance but often struggle to align generated samples with human preferences measured by a reward model. A simple yet effective algorithm for test-time alignment is Best-of-$N$ (BoN) sampling, which draws $N$ i.i.d. samples from a pre-trained diffusion model and outputs the single highest-reward sample. Despite its empirical success, BoN makes limited use of reward information, as it is incorporated only at the final selection stage without influencing the reverse diffusion trajectory during sampling. Consequently, BoN sampling does not improve the average alignment of generated samples and is primarily suited to single-output settings. We propose Best-of-$N$ Guidance (BoNG), a novel method that integrates the principle of BoN sampling directly into the reverse diffusion process. BoNG performs online BoN selection over denoising particles and adjusts the reverse diffusion process t
    
[^359]: LLM作为评判者的设计选择有多重要？提示词设计、评分量表与模型的系统性比较

    How Much Do LLM-as-a-Judge Design Choices Matter? A Systematic Comparison of Prompt Designs, Rating Scales, and Models

    [https://arxiv.org/abs/2610.05094](https://arxiv.org/abs/2610.05094)

    该研究系统评估了10个推理模型在不同提示词设计、评分量表和任务下的LLM评判者表现，发现尽管与人类基准存在统计显著分歧，但实际差异很小、大多数评判设计仍然可靠，从而为LLM-as-a-judge的设计选择提供了实证依据。

    

    研究人员越来越多地使用大语言模型作为评判者（LLM-as-a-judge）来评估模型输出。然而，目前尚无关于如何设计这些评判者的标准，研究人员通常凭直觉选择提示词、评分量表和模型。如果这些选择会改变评判者的判定结果，那么两项研究可能会对相同的事实得出不同的结论。为了应对这一风险并为评判者设计提供实证基础，我们在两个任务上评估了10个推理模型的多种设计：对句子情感和毒性的标量评分（每个类别超过500条样本），以及对问答对的二元准确性分类（n=600）。在评分任务中，尽管评判者与人类基准之间存在统计上的显著分歧，但差异的实际规模足够小，足以认为大多数评判者是可靠的（在1-7量表上的平均绝对偏差为0.11分）；毒性评判者甚至优于标准分类方法……（摘要原文在此处截断）

    arXiv:2610.05094v1 Announce Type: cross  Abstract: Researchers increasingly use Large Language Models as judges (LLM-as-a-judge) to evaluate model outputs. Yet there are no standards for how to design these judges. Typically, researchers choose the prompt, rating scale, and model intuitively. If these choices change the judge's verdicts, two studies can reach different conclusions about the same facts. To address this risk and to provide an empirical basis for judge designs, we evaluate 10 reasoning models across multiple designs on two tasks: a scalar rating of sentence sentiment and toxicity (over 500 items per category), as well as a binary accuracy classification of question-answer pairs (n=600). For the rating tasks, despite judges showing significant disagreements with the human ground truth, the practical size of differences is small enough to consider most judges reliable (mean absolute deviation of 0.11 points on a 1 - 7 scale); toxicity judges even outperform standard classif
    
[^360]: E$^2$-OPSD：驯服在线策略自蒸馏中的熵过冲

    E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.05048](https://arxiv.org/abs/2610.05048)

    论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。

    

    在线策略自蒸馏（OPSD）无需第二个模型即可提供密集的token级监督：同一个网络在给定参考解答时充当教师，而在仅给定问题时充当学生。我们识别出该方法的一个特定失效模式：在训练过程中，学生的token熵会超过教师的熵并持续保持高位，我们将这一现象称为“熵过冲”（entropy overshoot）。我们将其根源追溯到蒸馏的双方。以参考答案为条件的教师在其面向答案的推理路径上表现得很自信，但这种自信难以迁移到学生生成的前缀上，使其监督过度依赖于答案特定的线索，而非可复用的推理模式；与此同时，OPSD所使用的前向KL散度会持续扩散学生的预测分布，而无法将其拉回。我们提出E$^2$-OPSD来同时应对这两个成因。示例引导的教学（exemplar-guided teaching）用检索到的已解决的相邻问题替换当前答案，提……（原文摘要在此处截断）

    arXiv:2610.05048v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) provides dense token-level supervision without a second model: one network acts as teacher with the reference solution and as student with only the problem. We identify a specific failure mode of this recipe. During training, student token entropy rises past the teacher's and remains elevated, a pattern we call entropy overshoot. We trace it to both sides of distillation. The reference-conditioned teacher is confident along its answer-directed reasoning path, but this confidence transfers poorly to student-generated prefixes, making its supervision overly tied to answer-specific cues rather than reusable reasoning patterns; meanwhile, the forward KL used by OPSD continually diffuses the student's predictive distribution without pulling it back. We introduce E$^2$-OPSD to address both causes. Exemplar-guided teaching replaces the current answer with a retrieved solved neighboring problem, provi
    
[^361]: 语言模型意外度对中文阅读预测能力的系统性分析

    A Systematic Analysis of the Predictive Power of LM Surprisal in Reading Chinese

    [https://arxiv.org/abs/2610.04898](https://arxiv.org/abs/2610.04898)

    本研究提出最短匹配序列（SMS）对齐方案以解决中文分词与语言模型子词分词不一致的问题，并利用从零训练的Chinese-Pythia模型证明语言模型意外度确实能预测中文阅读时间，且预测能力随模型规模的缩放模式因语料库而异。

    

    本研究分析了语言模型生成的词元级意外度对中文阅读时间的预测能力。我们首先提出了最短匹配序列，这是一种对齐方案，用于将眼动追踪语料库所假设的词切分与语言模型的子词分词进行映射，因为在中文语境下这两种分词方式常常不一致。随后，我们使用一系列从零开始训练、使用了300亿词元的Chinese-Pythia模型（14M至1.4B参数），考察了意外度在三个中文段落级眼动追踪语料库（GECO-CN、HKP和MECO）中对首次注视时长、凝视时长和总阅读时间的预测效果。与以往的无显著结果相反，我们的研究表明意外度确实能够预测中文阅读时间。然而，预测能力是否随模型规模和训练量而提升则因语料库而异：在GECO-CN中更大的模型预测效果更好，而在其他语料库中则出现了逆向缩放现象。

    arXiv:2610.04898v2 Announce Type: replace-cross  Abstract: This study analyzes the predictive power of LM-derived, token-level surprisal on Mandarin Chinese reading times. We first propose the Shortest Matching Sequence (SMS), an alignment scheme that maps between the word segmentation assumed by eye-tracking corpora and the LMs' subword tokenization, as the two tokenizations often disagree in the context of Mandarin Chinese. Then, using a suite of Chinese-Pythia models (14M-1.4B) trained on scratch with 30B tokens, we examine how well surprisal predicts first fixation duration, gaze duration, and total reading time in three paragraph-level eye-tracking corpora of Mandarin Chinese (GECO-CN, HKP, and MECO). Contrary to previous null findings, our results show that surprisal is predictive of Chinese reading times. However, whether predictive power scales with model size and the amount of training is corpus-specific: bigger models predict better in GECO-CN, whereas inverse scaling emerges
    
[^362]: SpecFold：折叠多分支冗余以加速扩散语言模型中的投机解码

    SpecFold: Folding Multi-Branch Redundancy for Faster Speculative Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2610.04875](https://arxiv.org/abs/2610.04875)

    SpecFold通过识别并利用投机验证中草稿分支与父分支之间隐藏状态高度相似的多分支计算冗余，以token级残差门控和选择性计算复用降低验证成本，从而加速扩散语言模型的多分支投机解码。

    

    扩散大语言模型（DLLMs）通过迭代块去噪生成文本，多分支投机解码则通过在单次前向传播中同时验证一个主分支与多个草稿分支来加速这一过程。现有DLLM加速方法主要利用去噪步骤之间的时间冗余，而我们识别出每个投机验证步骤中一条互补的冗余维度：多分支计算冗余。在投机验证过程中，草稿分支从其父分支继承大部分token，仅解开少量额外位置，导致大量隐藏状态在各分支之间保持高度相似。我们提出SpecFold，一种算法-系统协同设计，利用这种多分支冗余来降低多分支投机验证的成本。在算法层面，SpecFold执行token级残差门控并选择性地复用父分支的计算（摘要原文在此处截断）。

    arXiv:2610.04875v2 Announce Type: replace  Abstract: Diffusion large language models (DLLMs) generate text through iterative block denoising, and multi-branch speculative decoding accelerates this process by verifying a main branch together with multiple draft branches in a single forward pass. While prior DLLM acceleration methods primarily exploit temporal redundancy across denoising steps, we identify a complementary redundancy axis within each speculative verification step: multi-branch computational redundancy. During speculative verification, draft branches inherit most tokens from their parents while unmasking a small set of additional positions, causing large portions of hidden states to remain highly similar across branches. We propose SpecFold, an algorithm-system co-design that exploits this multi-branch redundancy to reduce the cost of multi-branch speculative verification. Algorithmically, SpecFold performs token-level residual gating and selectively reuses parent computat
    
[^363]: 每个键对应更多值：非对称稀疏注意力加速大语言模型解码

    More Value per Key: Asymmetric Sparse Attention for Faster LLM Decoding

    [https://arxiv.org/abs/2610.04753](https://arxiv.org/abs/2610.04753)

    提出稀疏非对称分组查询注意力SAGA，通过解耦键头与值头数量——用更少的键头加速推理、保留更多值头维持模型容量——从而实现更快的LLM解码。

    

    大语言模型（LLM）的自回归生成受限于注意力机制的内存与计算需求。稀疏注意力方法通过仅选择注意力矩阵中的高概率条目来缓解这一开销。我们观察到，在许多此类方法中，这使得概率-值乘法的开销变得可以忽略不计，从而将瓶颈转移到了查询-键计算步骤上。因此，可以通过减少键头的数量来加速推理，同时保留更多的值头，以在有限的额外解码成本下维持模型容量。我们提出了稀疏非对称分组查询注意力（SAGA），它解耦了键头与值头的数量以利用这一原理，并将其与近似top-N（Atop-N）注意力相结合，后者是一种简单的稀疏注意力方法，旨在研究稀疏性与头数非对称性之间的相互作用。我们从理论上形式化了这种非对称性的优势，并通过实验验证了这些优势。

    arXiv:2610.04753v2 Announce Type: replace-cross  Abstract: Autoregressive generation in Large Language Models (LLMs) is constrained by the memory and computational demands of attention mechanisms. Sparse attention methods mitigate this cost by selecting only high-probability entries of the attention matrix. We observe that in many such methods, this renders the probability-value multiplication negligible, shifting the bottleneck to the query-key step. Key heads can therefore be reduced to accelerate inference, while retaining more value heads preserves capacity with limited additional decoding cost. We introduce Sparse Asymmetric Group-Query Attention (SAGA), which decouples key and value head counts to exploit this principle, and pair it with approximate top-N (Atop-N) attention, a simple sparse attention method designed to study the interaction between sparsity and head-count asymmetry. We formalize the benefits of this asymmetry theoretically and validate them empirically through la
    
[^364]: 微调视觉语言模型以增强AI的空间智能：理解3D与2D旋转

    Fine-Tuning VLM for Enhancing AI's Spatial Intelligence: Understanding 3D and 2D Rotations

    [https://arxiv.org/abs/2610.04206](https://arxiv.org/abs/2610.04206)

    该研究通过在物体旋转数据集上微调视觉语言模型，显著提升了AI对2D和3D旋转的空间推理能力，其中微调后的Gemma-4混合专家模型表现优于通用模型。

    

    空间智能是科学、技术、工程和数学（STEM）、医学、建筑和建造等多个领域的一项基本技能。近期研究表明，视觉语言模型（VLM）在空间推理方面仍存在局限，这阻碍了人工智能（AI）执行实际的空间任务。通过使用为训练和评估开发的多个物体旋转数据集，我们的实验证明在2D和3D旋转检测方面均取得了显著的改进。经过微调的谷歌DeepMind构建的Gemma-4混合专家（MoE）模型，在预测由轴和角度共同定义的旋转时，显著优于经过微调的Gemma-4通用模型。微调还大幅提高了对2D表示的角度估计能力，且无需显式的坐标系。此外，可识别的物体并未提高角度检测的准确性；相反，物体（摘要在此处被截断）

    arXiv:2610.04206v2 Announce Type: replace  Abstract: Spatial intelligence is a fundamental skill in multiple domains, such as Science, Technology, Engineering, and Mathematics (STEM), Medicine, Architecture, and Construction. Recent studies indicate that Vision-Language Models (VLMs) still face limitations in spatial reasoning, which inhibits artificial intelligence (AI) from performing practical spatial tasks. Using multiple object-rotation datasets developed for training and evaluation, our experiments demonstrated promising improvements in both 2D and 3D rotation detection. Fine-tuned Google DeepMind-built Gemma-4 mixture-of-experts (MoE) models significantly outperformed fine-tuned Gemma-4 generalist models in predicting rotations defined by both their axes and angles. Fine-tuning also substantially improved angle estimation for 2D representation without requiring an explicit coordinate system. Furthermore, identifiable objects did not improve angle-detection accuracy; instead, obj
    
[^365]: 具有结构化思维链的智能体AI用于增强AI空间智能：旋转的可视化与推理

    Agentic AI with Structured CoT for Enhancing AI's Spatial Intelligence: Visualization and Reasoning of Rotation

    [https://arxiv.org/abs/2610.04188](https://arxiv.org/abs/2610.04188)

    本研究通过结构化思维链（CoT）推理策略显著提升了生成式智能体AI模型（GPT-5.6）在三维空间旋转理解任务中的空间推理能力。

    

    最近的研究表明，具备语言和视觉能力的人工智能（AI）在空间推理方面仍然存在局限性。本文利用AI的图像处理和语言处理特性，研究了先进生成式AI理解三维空间中物体旋转的空间能力。我们基于修订版普渡空间可视化测试：旋转可视化（Revised PSVT:R），通过旋转图训练并检验了生成式智能体AI模型（GPT-5.6）理解空间旋转过程的空间智能。我们对修订版PSVT:R进行了改进，叠加了额外的图形和上下文特征，以评估不同的思维链（CoT）推理策略如何影响模型性能。结果表明，结构化CoT推理在两个数据集（PSVT:R和P……）上均提升了基础GPT-5.6模型的空间推理性能。

    arXiv:2610.04188v2 Announce Type: replace  Abstract: Recent studies show that artificial intelligence (AI) with language and vision capabilities still experiences limitations in spatial reasoning. In this paper, we have studied the spatial capabilities of advanced generative AI to understand the rotations of objects in 3D space, utilizing AI's image processing and language processing features. We trained and examined the spatial intelligence of a generative Agentic AI model (GPT-5.6) to understand the spatial rotation process with rotation diagrams based on the revised Purdue Spatial Visualization Test: Visualization of Rotations (Revised PSVT:R). We improvised the Revised PSVT:R by superimposing additional graphical and contextual features to evaluate how different Chain-of-Thought (CoT) reasoning strategies influence model performance. The results indicate that structured CoT reasoning improves the spatial reasoning performance of the base GPT-5.6 model in both datasets (PSVT:R and P
    
[^366]: OncoNoteBERT：面向真实世界门诊肿瘤学笔记自然语言处理的基础表示模型

    OncoNoteBERT: A Foundation Representation Model for Natural Language Processing of Real-World Outpatient Oncology Notes

    [https://arxiv.org/abs/2610.03829](https://arxiv.org/abs/2610.03829)

    本研究基于包含29万余份真实世界英国门诊肿瘤笔记的受治理语料库，开发并评估了两种肿瘤学专用语言模型——使用肿瘤学分词器从头训练的OncoNoteBERT和通过持续预训练得到的OncoNote-RadBERT，以更好地表示肿瘤学专业术语和临床表达。

    

    真实世界的门诊肿瘤学笔记包含专业术语、肿瘤分期表述、治疗名称、毒性描述以及机构特有的去标识化标记，通用生物医学或相关临床语言模型可能无法高效地表示这些内容。我们利用一个受治理的英国门诊肿瘤学语料库（包含来自21,564名肺癌和头颈癌患者的290,026份笔记），开发并评估了肿瘤学专用的BERT风格编码器。我们将RadBERT和PathologyBERT这两种外部模型与两种本地策略进行比较：通过持续掩码语言模型预训练得到的OncoNote-RadBERT，以及使用肿瘤学WordPiece分词器从头训练的OncoNoteBERT。模型评估采用了验证集上的掩码语言建模损失与困惑度、分词器碎片化指标、临床术语分词、掩码标记探测以及探索性表示分析等方法。两种外部编码器……

    arXiv:2610.03829v1 Announce Type: cross  Abstract: Real-world outpatient oncology notes contain specialised terminology, tumour staging expressions, treatment names, toxicity descriptions, and institution-specific de-identification markers that may not be represented efficiently by general biomedical or adjacent clinical language models. We developed and evaluated oncology-specific BERT-style encoders using a governed UK outpatient oncology corpus comprising 290,026 notes from 21,564 patients treated for lung and head-and-neck cancer. We compared RadBERT and PathologyBERT with two local strategies: OncoNote-RadBERT, produced by continued masked language model pretraining, and OncoNoteBERT, trained from scratch with an oncology WordPiece tokenizer. Models were evaluated using masked language modelling loss and perplexity on the validation set, tokenizer fragmentation metrics, clinical term tokenisation, masked-token probes, and exploratory representation analysis. Both external encoders
    
[^367]: WAMJET：一个面向世界动作模型加速的框架

    WAMJET: A Harness for World Action Model Acceleration

    [https://arxiv.org/abs/2610.03797](https://arxiv.org/abs/2610.03797)

    WAMJET是一个代理式加速框架，通过为编码智能体提供可复用的优化指导和测量验证工具，以瓶颈驱动的迭代方式自动优化世界动作模型的推理，在保持动作质量的同时实现了高达9.95倍的无损加速。

    

    世界动作模型利用预训练的视频基础模型来完成机器人操作任务，但其庞大的主干网络和视频-动作联合预测机制成本高昂。尽管现有的加速技术提供了多种降低此类成本的途径，但针对每个模型和硬件平台来选择和组合这些技术需要大量的工程工作。为解决这一瓶颈，我们提出了WAMJET，这是一个代理式框架，通过为编码智能体配备可复用的优化指导以及测量与验证工具来加速WAM的推理。WAMJET遵循瓶颈驱动的工作流程：智能体对推理进行性能剖析、修改目标代码、验证优化效果，并随着瓶颈的转移迭代地完善加速技术栈，同时保持动作质量。实验覆盖了六个WAM模型、三种编码智能体和两种GPU架构。WAMJET相比上游实现实现了高达9.95倍的无损加速。近似加速与硬……（原文摘要截断）

    arXiv:2610.03797v2 Announce Type: replace-cross  Abstract: World Action Models (WAMs) leverage pretrained video foundation models for robot manipulation, but their large backbones and video-action co-prediction are expensive. Although existing acceleration techniques offer many ways to reduce this cost, selecting and composing them requires substantial engineering for each model and hardware platform. To tackle this bottleneck, we present WAMJET, an agentic harness that accelerates WAM inference by equipping coding agents with reusable optimization guidance and measurement and validation tools. WAMJET follows a bottleneck-driven workflow where the agent profiles inference, modifies targeted code, validates effects, and iteratively refines the acceleration stack as bottlenecks shift, while preserving action quality. Experiments span six WAMs, three coding agents, and two GPU architectures. WAMJET achieves up to 9.95x lossless speedup over upstream implementations. Approximation and hard
    
[^368]: 类型化决策模型中候选选项覆盖度的基准测试

    Benchmarking Candidate Coverage in Typed Decision Models

    [https://arxiv.org/abs/2610.03387](https://arxiv.org/abs/2610.03387)

    本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。

    

    类型化决策模型会返回选择结果，或针对请求时提供的答案选项返回分布。在选项完整情况下的准确率并不能说明模型是否能识别参考答案缺失的情况，或者是否会避免错误拒绝有效的候选选项。我们提出了一个成对候选覆盖度基准测试协议，并对 Laya 和 Jev 两个模型在 AG News、DBpedia、Emotion 和 TREC 数据集上进行了初步评估。两个模型接收完全相同的冻结文本和请求：300 条校准文本和 589 条测试文本，每个模型产生 23,932 次预测。存在/缺失配对与普通候选数量相匹配，且名称变体保持描述、成员和顺序不变。原生的拒绝行为差异显著：在具有自然名称的五个 TREC 候选选项下，Laya 能检测出 97.2% 的答案缺失案例，但会错误拒绝 69.7% 的存在对照案例；Jev 的这两个比率分别为 24.8% 和 0.0%。仅使用校准数据的 none 分数阈值将这些比率分别改变为 33.9%/3.7% 和 45.0%/1.8%。在 DB

    arXiv:2610.03387v1 Announce Type: new  Abstract: Typed decision models return choices or distributions over answer options supplied at request time. Accuracy with complete options does not establish whether a model recognizes that a reference answer is missing or avoids rejecting valid candidates. We present a paired candidate-coverage benchmark protocol and an initial evaluation of Laya and Jev across AG News, DBpedia, Emotion, and TREC. The models receive identical frozen texts and requests: 300 calibration and 589 test texts yield 23,932 predictions per model. Present/absent pairs match ordinary candidate count, and name variants preserve descriptions, members, and order. Native rejection behavior differs sharply: at five TREC candidates with natural names, Laya detects 97.2% of missing-answer cases but falsely rejects 69.7% of present controls; Jev's rates are 24.8% and 0.0%. Calibration-only none-score thresholds change these rates to 33.9%/3.7% and 45.0%/1.8%, respectively. On DB
    
[^369]: 基于文本梯度的交易策略优化

    Trading Strategy Optimization via Textual Gradient

    [https://arxiv.org/abs/2610.03128](https://arxiv.org/abs/2610.03128)

    提出了TradeGrad框架，通过利用积累的优化经验来估计文本梯度并结合多尺度修订策略，克服了传统文本梯度优化短视及忽视时间稳健性的问题，实现了更稳健的量化交易策略优化。

    

    量化交易策略设计旨在从历史数据中发现在未来市场中依然有效的交易程序，这可以被视为一个黑盒程序优化问题。基于大语言模型（LLM）的文本梯度方法通过为迭代式策略改进提供明确的优化方向，展现出广阔前景。然而，直接应用文本梯度面临两个挑战：（1）优化过程是短视的，未能充分利用先前评估积累的经验；（2）汇总式的回测反馈忽略了时间上的稳健性，可能偏向那些仅在特定市场时期表现良好的策略。为应对这些挑战，我们提出了TradeGrad，一个经验引导的文本梯度框架，用于稳健的交易策略优化。TradeGrad利用积累的优化经验来估计文本梯度，并采用多尺度修订进行策略探索与改进，还进一步……

    arXiv:2610.03128v1 Announce Type: new  Abstract: Quantitative trading strategy design aims to discover trading programs from historical data that remain effective in future markets, which can be viewed as a black-box program optimization problem. LLM-based textual gradients offer a promising approach by providing explicit optimization directions for iterative strategy refinement. However, directly applying textual gradients faces two challenges: (1) optimization is myopic, underutilizing experience from previous evaluations; and (2) aggregate backtest feedback overlooks temporal robustness, potentially favoring strategies that perform well only in specific market periods. To address these challenges, we propose TradeGrad, an experience-guided textual-gradient framework for robust trading strategy optimization. TradeGrad leverages accumulated optimization experience to estimate textual gradients and employs multi-scale revisions for both strategy exploration and refinement. It further i
    
[^370]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^371]: LUMOS：在大语言模型中从训练数据追踪参数化知识到行为输出

    LUMOS: Tracing Parametric Knowledge from Training Data to Behavioral Outputs in LLMs

    [https://arxiv.org/abs/2610.02902](https://arxiv.org/abs/2610.02902)

    LUMOS诊断框架利用完全透明的OLMo 2训练语料库，沿“训练数据暴露→行为输出”的因果链追踪大语言模型的参数化知识，揭示模型内部能高可分性地编码罕见事实（84%）但在行为上表达不足（54%），且该检索差距随模型规模增大而缩小。

    

    当前对大语言模型参数化知识的分析大多以输出为中心，在没有验证模型实际训练内容的情况下就对模型“知道什么”下结论。这使得一些根本性问题——例如模型的正确回答究竟是反映了真正的泛化能力还是死记硬背——只能停留在推测而非证据层面。为了消除这些模糊性，我们提出了LUMOS，一个沿“训练数据暴露→行为输出”因果链追踪知识的诊断框架，并利用训练语料库完全透明的OLMo 2模型。通过将分析建立在经过验证的训练暴露之上，我们发现模型内部以高可分性（84%）编码罕见事实，却无法在行为层面表达它们（仅54%），不过这一检索差距会随模型规模的增大而缩小。此外，当模型被要求对自己的答案进行自我反思时，它们在训练过的内容上表现可靠（83%），但在未训练过的内容上则下降至随机基线水平（49%）。

    arXiv:2610.02902v1 Announce Type: new  Abstract: Current analyses of LLMs' parametric knowledge are largely output-centric, drawing conclusions about what a model knows without verifying what it was actually trained on. This leaves fundamental questions, such as whether a correct response reflects genuine generalization or rote memorization, grounded in speculation rather than evidence. To resolve these ambiguities, we introduce LUMOS, a diagnostic framework that traces knowledge along the causal chain from training-data exposure to behavioral output, leveraging OLMo 2 with its fully transparent training corpus. By grounding analysis in verified exposure, we reveal that models internally encode rare facts with high separability (84%) yet fail to express them behaviorally (54%), though this retrieval gap narrows with scale. Furthermore, when models are asked to self-reflect on their own answers, they perform reliably on trained content (83%) but drop to random-baseline levels (49%) on u
    
[^372]: BitNest：面向内存高效大语言模型推理加速的比特嵌套投机解码

    BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration

    [https://arxiv.org/abs/2610.02800](https://arxiv.org/abs/2610.02800)

    BitNest提出了一种比特嵌套的投机解码框架，将低精度草稿模型直接嵌入高精度目标模型的权重表示中，通过残差细化使两者共享单一物理权重，从而在加速大语言模型推理的同时显著降低内存开销。

    

    投机解码通过使用轻量级草稿模型提出多个token进行并行验证，从而加速自回归生成。然而，现有方法通常需要一个额外的草稿模型或权重表示，在资源受限的设备上引入了不可忽视的内存开销。自投机方法虽然减少了这种开销，但仍面临草稿质量、目标质量和存储效率之间的权衡。我们提出了BitNest，这是一种比特嵌套的投机解码框架，它将低精度草稿直接嵌入到更高精度的目标表示中。BitNest并非从预定义的目标模型派生草稿，而是首先构建一个强大的低精度基础模型，然后通过残差细化恢复更高精度的目标模型，使两个模型能够共享单一的物理权重表示。BitNest进一步将这种渐进精度设计扩展到KV缓存，用于长上下文推理（摘要在此处截断）。

    arXiv:2610.02800v1 Announce Type: new  Abstract: Speculative decoding accelerates autoregressive generation by using a lightweight draft to propose multiple tokens for parallel verification. However, existing methods often require an additional draft model or weight representation, introducing non-negligible memory overhead on resource-constrained devices. Self-speculative approaches reduce this overhead, yet still face trade-offs between draft quality, target quality, and storage efficiency. We propose BitNest, a bit-nested speculative decoding framework that embeds a low-precision draft directly into the higher-precision target representation. Instead of deriving a draft from a predefined target, BitNest first constructs a strong low-precision base and then recovers the higher-precision target through residual refinement, enabling both models to share a single physical weight representation. BitNest further extends this progressive-precision design to the KV cache for long-context in
    
[^373]: PAPER2LLM++：基于研究论文的大语言模型持续自我演化

    PAPER2LLM++: Continual Self-Evolution of LLMs from Research Papers

    [https://arxiv.org/abs/2610.02793](https://arxiv.org/abs/2610.02793)

    PAPER2LLM++ 提出了一个让大语言模型从研究论文中持续自我演化的框架，通过提取论文中的研究发现、验证局限性是否仍然存在，并借助“尝试-评估-提交”机制整合更新，从而在不遗忘先前改进、不损害通用能力的前提下实现模型的自动改进。

    

    对大语言模型（LLM）的研究不断揭示模型的局限性、其成因以及潜在的解决方案。然而，这些人类发现与模型演化在很大程度上仍然脱节：LLM 并不会自动从关于其自身缺陷的新研究中学习。我们提出了 PAPER2LLM++，一个使大语言模型能够从研究论文中持续自我演化的框架。PAPER2LLM++ 并非将论文仅仅视为可供检索的知识，而是将不断增长的文献作为模型改进的证据流和监督来源。对于每一篇新输入的论文，该框架会提取有证据支撑的研究发现，检验所报告的局限性在当前模型中是否依然存在，并在需要时将这些发现转化为候选学习信号。一个“尝试-评估-提交”程序仅在更新能够改进目标行为、且不会明显遗忘先前的改进或损害通用能力时，才将其整合到模型中。在一系列研究发现的序列流上……（摘要原文在此处截断）

    arXiv:2610.02793v1 Announce Type: new  Abstract: Research on LLMs continually uncovers model limitations, their causes, and potential solutions. Yet these human discoveries remain largely disconnected from model evolution: an LLM does not automatically learn from new research about its own failures. We introduce PAPER2LLM++, a framework for continual self-evolution of LLMs from research papers. Rather than treating papers merely as knowledge to retrieve, PAPER2LLM++ uses the growing literature as a stream of evidence and supervision for model improvement. For each incoming paper, it extracts evidence-grounded findings, tests whether the reported limitation persists in the current model, and, when needed, converts the findings into candidate learning signals. A try-evaluate-commit procedure integrates an update only when it improves the targeted behavior without substantially forgetting prior improvements or degrading general capabilities. Across a sequential stream of research-discover
    
[^374]: 解耦记忆与上下文：面向令牌高效测试时持续学习的结构化记忆

    Decoupling Memory from Context: Structured Memory for Token-Efficient Test-Time Continual Learning

    [https://arxiv.org/abs/2610.02687](https://arxiv.org/abs/2610.02687)

    该论文提出将记忆与上下文解耦的结构化记忆方法，把智能体记忆系统的更新视为上下文优化问题，从而在测试时持续学习中以更少的令牌高效积累和复用跨查询经验，避免了共享上下文不断膨胀所带来的成本上升与性能下降。

    

    大型语言模型越来越多地被部署在企业、科学和医疗应用中，在这些场景下，智能体必须整合领域特定知识并从经验中不断适应。上下文工程通过在推理时提供指令、策略和证据来改善模型行为，为权重更新提供了一种实用的替代方案。然而，在线调整上下文通常需要代价高昂的试错过程，而且查询往往被独立处理，导致有用的经验无法延续下去。记忆系统通过在多次交互之间保留信息来解决这一局限，但那些不断向共享上下文追加信息的方法会面临令牌成本不断上升、上下文窗口受限以及随上下文扩展而出现的性能退化。我们提出了上下文优化的统一形式化框架，并表明智能体记忆系统的更新可以被解释为一种优化……（摘要内容在此处截断）

    arXiv:2610.02687v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in enterprise, scientific, and medical applications, where agents must incorporate domain-specific knowledge and adapt from experience. Context engineering offers a practical alternative to weight updates by improving model behavior through instructions, strategies, and evidence supplied at inference time. However, adapting context online typically requires a costly trial-and-error process, while queries are often processed independently, preventing useful experience from carrying forward. Memory systems address this limitation by retaining information across interactions, but approaches that continually append information to a shared context face increasing token costs, context-window limits, and performance degradation as the context expands. We introduce a unified formulation of context optimization and show that an agent memory system update can be interpreted as an optimization 
    
[^375]: 基于选择性状态空间模型的分布式学习：架构感知的收敛性分析

    Distributed Learning with Selective State Space Models: Architecture-Aware Convergence Analysis

    [https://arxiv.org/abs/2610.02659](https://arxiv.org/abs/2610.02659)

    该论文首次针对选择性状态空间模型（如Mamba2）推导了架构感知的梯度和平滑度界限，并据此建立了FedAvg和FedProx在联邦学习中的收敛性保证，揭示了递归稳定性、输入相关离散化和状态投影范数对分布式优化的影响。

    

    现代状态空间模型（SSM），如Mamba2，通过将线性时间复杂度的序列建模与递归状态空间动力学相结合，为Transformer提供了一种极具吸引力的替代方案。然而，SSM在分布式学习环境中的行为仍然鲜为人知。特别是，现有的标准联邦学习方法在很大程度上与架构无关，没有考虑到现代选择性SSM所特有的稳定性、选择性和状态空间参数化特性。为了解决这一问题，我们为单层和多层选择性SSM推导了架构感知的梯度和平滑度界限，并为FedAvg和FedProx推导了收敛界限，刻画了递归稳定性、输入相关的离散化以及状态投影范数如何影响联邦优化。随后，我们在由教师SSM生成的序列上，使用遵循所分析递归结构的学习器，对单层界限进行了数值验证。

    arXiv:2610.02659v1 Announce Type: cross  Abstract: Modern state space models (SSMs), such as Mamba2, provide a compelling alternative to transformers by combining linear-time sequence modeling with recurrent state-space dynamics. However, the behavior of SSMs in distributed learning settings remains poorly understood. In particular, the existing standard federated learning methods are largely architecture-agnostic, and do not account for the stability, selectivity, and state-space parameterization that characterize modern selective SSMs. To address this, we derive architecture-aware gradient and smoothness bounds for single- and multi-layer selective SSMs, and convergence bounds for FedAvg and FedProx, characterizing how recurrent stability, input-dependent discretization, and state projection norms affect federated optimization. We then numerically validate the single-layer bounds on sequences generated by a teacher SSM, using a learner that follows the analyzed recurrence. We use thi
    
[^376]: 绘制RAG研究版图：效率、防御、交互性与推理的四轴分类体系

    Mapping the RAG Landscape: A Four Axis Taxonomy of Efficiency, Defense, Interactivity, and Reasoning

    [https://arxiv.org/abs/2610.01936](https://arxiv.org/abs/2610.01936)

    本综述提出了一个涵盖效率、防御、交互性与推理四个维度的RAG分类体系，系统梳理了检索增强生成领域超越基础架构的最新研究进展。

    

    大型语言模型（LLMs）在众多任务上展现出卓越的流畅性，但仍受限于其静态的、绑定于参数的知识以及容易产生信息幻觉的问题。检索增强生成（RAG）通过在生成过程中引入外部检索来解决这些问题，使模型输出建立在可验证且最新的信息来源之上。以往的综述主要聚焦于核心RAG架构和标准流程，而近期的研究则探索了超越这些基础设计的更广泛的挑战与能力。本综述对当代RAG的发展进行了全面且结构化的考察，将该领域组织为一个四轴分类体系：提升检索效率、增强鲁棒性与安全性、支持用户驱动和交互式工作流程，以及实现多步骤或复杂推理。我们对RAG框架的关键组件进行了形式化……（原文摘要不完整）

    arXiv:2610.01936v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable fluency across many tasks but remain limited by their static, parameter bound knowledge and their susceptibility to hallucinating information. Retrieval Augmented Generation (RAG) addresses these issues by incorporating external retrieval into the generation process, grounding model outputs in verifiable and up to date sources. While prior surveys primarily focus on core RAG architectures and standard pipelines, recent research explores broader challenges and capabilities that extend beyond these foundational designs. This survey provides a consolidated and structured examination of contemporary RAG developments, organizing the field into a four axis taxonomy: improving retrieval efficiency, strengthening robustness and security, supporting user driven and interactive workflows, and enabling multi step or complex reasoning. We formalize key components of the RAG framework and revi
    
[^377]: 面向世界动作模型的完成感知引导

    Completion Aware Guidance for World Action Models

    [https://arxiv.org/abs/2610.01559](https://arxiv.org/abs/2610.01559)

    提出无需训练的完成感知引导（CAG）采样方法，引导世界动作模型的生成朝向任务完成，显著提升机器人任务成功率并将任务不完整想象从 79% 大幅降至 40%。

    

    世界动作模型预测视觉未来和机器人动作，但它们仍然容易受到任务不完整想象的影响，即看似合理且与动作一致的预测遗漏了完成任务所需的转变。在本文中，我们证明这种失败并非世界模型骨干网络的固有缺陷，而是在将其适配于短片段控制时出现的，短片段控制会反复倾向于看似合理的局部延续而非完成任务的转变。为解决这一问题，我们提出了完成感知引导（CAG），这是一种无需训练的采样方法，可引导生成过程朝向任务完成。在具有代表性的世界动作模型上，CAG 在 RoboTwin 2.0 子集上将成功率从 64% 提升至 70%，在零样本仿真中将成功率从 69% 提升至 75%，同时将任务不完整想象的比例从 79% 降低至 40%。

    arXiv:2610.01559v1 Announce Type: cross  Abstract: World Action Models (WAMs) predict visual futures and robot actions, yet they remain susceptible to task-incomplete imagination, where plausible, action-consistent predictions omit the transition needed for task completion. In this paper, we show that this failure is not inherent to the world model backbone, but emerges when adapted for short-chunk control, which can repeatedly favor plausible local continuations over task-completing transitions. To address this, we introduce Completion Aware Guidance (CAG), a training-free sampling method that guides generation toward task completion. Across representative WAMs, CAG improves success from 64% to 70% on a RoboTwin 2.0 subset and from 69% to 75% in zero-shot simulation, while reducing task-incomplete imagination from 79% to 40%.
    
[^378]: 第二个模型何时有帮助？大语言模型验证中的跨模型审查

    When Does a Second Model Help? Cross-Model Review in LLM Verification

    [https://arxiv.org/abs/2610.01471](https://arxiv.org/abs/2610.01471)

    在大语言模型输出验证中，跨模型审查与同模型审查发现的错误集合部分不同，且“一次同模型新会话审查加一次跨模型审查”的组合比两次同模型审查能匹配到更多埋设错误（56.7% vs. 42.7%）。

    

    大语言模型如今能够生成代码、文档和分析内容，并且越来越多地被用于审查这类输出。本文探究的问题是：由不同的模型进行第二次审查在何时才有帮助？在作者早期预印本研究（在单一模型内改变上下文、重复次数和角色结构）的基础上，我们通过一项受控实验来检验模型独立性：实验包含30个工件（其中埋设150个错误）、10种审查条件，以及由来自两个开发者的三个审查模型执行的900次审查会话。在该实验中，(1) 顶级跨模型审查者在F1分数上与同模型在全新会话中的审查（CCR）无显著差异，但这并不等同于二者等价；(2) 两者发现的错误部分不同（Jaccard相似度为41.2%）；(3) 在两次审查调用的设定下，一次CCR加一次跨模型审查所匹配到的埋设错误多于两次CCR审查（56.7% vs. 42.7%；经Holm校正的p=.006），但并不显著多于两次顶级跨模型审查。

    arXiv:2610.01471v1 Announce Type: cross  Abstract: Large language models now generate code, documentation, and analyses, and are increasingly used to review such output. We ask when a second review by a different model helps. Building on the author's earlier preprints, which varied context, repetition, and role structure within one model, we test model independence in a controlled experiment: 30 artifacts with 150 planted errors, 10 review conditions, and 900 review sessions with three reviewer models from two developers. In this experiment, (1) a top-tier cross-model reviewer is not significantly different in F1 from same-model review in a fresh session (CCR), which does not establish equivalence; (2) the two find partly different errors (Jaccard 41.2%); and (3) at two review calls, one CCR plus one cross-model review matches more planted errors than two CCR reviews (56.7% vs. 42.7%; Holm-adjusted p=.006), but not significantly more than two reviews by the top-tier cross-model reviewe
    
[^379]: SpikeMoE：受大脑启发的竞争性路由，用于灵活的脉冲混合专家模型

    SpikeMoE: Brain-Inspired Competitive Routing for Flexible Spiking Mixture-of-Experts

    [https://arxiv.org/abs/2610.01418](https://arxiv.org/abs/2610.01418)

    提出SpikeMoE框架，通过受海马CA1区竞争-抑制机制启发的脉冲k-WTA路由器（含侧向抑制与不应期），将神经元尺度的脉冲动力学与模型尺度的专家选择相结合，并支持缺失模态建模，实现灵活的脉冲混合专家架构。

    

    脉冲神经网络（SNN）通过神经元尺度的生物启发动力学实现事件驱动计算，而混合专家模型（MoE）通过模型尺度的专家选择执行条件计算。整合二者的优势有望构建灵活的神经架构。然而，一个关键挑战在于如何设计基于脉冲活动的专家选择机制。为解决这一问题，我们提出了一种受海马CA1区竞争-抑制现象启发的基于脉冲的k-WTA路由器。该路由器引入侧向抑制和不应期机制，根据离散的脉冲计数来选择Top-K专家。在此基础上，我们提出了SpikeMoE框架，将神经元尺度的脉冲动力学与模型尺度的专家选择相融合。为应对多模态任务中多感官输入不完整的问题，我们进一步为SpikeMoE配备了两阶段的缺失模态建模模块。

    arXiv:2610.01418v1 Announce Type: new  Abstract: Spiking Neural Networks (SNNs) enable event-driven computation through biologically inspired dynamics at the neuronal scale, while Mixture-of-Experts (MoE) perform conditional computation through expert selection at the model scale. Integrating their strengths offers potential for flexible neural architectures. A key challenge, however, lies in designing an expert selection mechanism based on spiking activity. To address this, we introduce a spike-based k-WTA Router inspired by competition-inhibition observed in the hippocampal CA1 region. The router incorporates lateral inhibition and refractory period to select Top-K experts according to discrete spike counts. Building on this, we present SpikeMoE, a framework that integrates neuronal-scale spiking dynamics with model-scale expert selection. To address incomplete multisensory inputs in multimodal tasks, we further equip SpikeMoE with a two-stage missing-modality modeling module that co
    
[^380]: ReSolve：通过选择性生成式调解复用候选推理

    ReSolve: Reusing Candidate Reasoning through Selective Generative Moderation

    [https://arxiv.org/abs/2610.01140](https://arxiv.org/abs/2610.01140)

    ReSolve 是一种无需训练的推理流程，通过选择性生成式调解复用采样的候选推理，在竞赛数学题上超越投票和八样本自洽性方法，同时将 token 消耗降低约 46%-47%。

    

    对多个解进行采样不仅会在最终答案上消耗计算，也会在中间推导和未完成的论证上消耗计算。我们提出 ReSolve，这是一种无需训练的推理流程，通过选择性生成式调解来复用这些候选推理。当候选答案彼此不一致或缺乏可解析的答案时，一个答案分布控制器会调用模型来检视已有的推导过程，然后将生成的解纳入一个有界循环中。在采用 Hybrid 评分、以两个独立采样的候选池进行评估的 130 道竞赛数学题上，ReSolve 分别获得 100 和 99 个正确答案，而同样四个候选的投票方法仅分别获得 91 和 92 个正确答案，且在两个候选池中相对投票方法均未出现由正确变为错误的情况。八样本自洽性方法分别获得 94 和 96 个正确答案，但消耗的 token 数量显著更多；在这两项评估中，ReSolve 分别少使用 46.3% 和 47.2% 的 token。

    arXiv:2610.01140v1 Announce Type: new  Abstract: Sampling multiple solutions spends computation on intermediate deductions and unfinished arguments as well as final answers. We introduce ReSolve, a training-free inference procedure that reuses this candidate reasoning through selective generative moderation. An answer-distribution controller invokes a model to examine existing derivations when candidates disagree or lack a parseable answer, then incorporates the generated solution into a bounded loop. Under Hybrid scoring on 130 competition-mathematics problems evaluated with two independently sampled candidate pools, ReSolve obtains 100 and 99 correct answers, compared with 91 and 92 for voting over the same four candidates, with no correct-to-incorrect changes relative to that vote in either pool. Eight-sample self-consistency obtains 94 and 96 correct answers while consuming substantially more tokens; ReSolve uses 46.3% and 47.2% fewer tokens in the two evaluations. A controlled abl
    
[^381]: 强化学习的扩展真的需要更多训练吗？

    Does Scaling Reinforcement Learning Really Require More Training?

    [https://arxiv.org/abs/2610.01133](https://arxiv.org/abs/2610.01133)

    提出策略空间扩展方法SURGE，通过对同一强化学习运行中的两个检查点（高精度锚点与生成简短回复的供体）进行特征空间融合，在不增加训练或推理计算的情况下获得比原检查点更强的策略。

    

    扩展推理能力通常需要在强化学习（RL）或推理阶段投入更多计算资源。我们证明，一段已完成的强化学习训练历史能够产生比其优化器所访问过的检查点更强的策略。我们将其称为策略空间扩展：在不延长训练时间、也不增加单次查询推理计算量的前提下，从固定的强化学习训练历史中扩展可部署的策略集合。我们通过SURGE（通过特征空间融合实现无梯度强化学习扩展）来实例化这一方法。SURGE将同一强化学习运行中的两个检查点结合起来：一个高精度的锚点检查点和一个具有竞争力且能生成更简短回复的供体检查点。它将两个检查点表示为相对于其共享初始化的变化量，然后对锚点的更新进行谱分解，以保留其主导分量并融入供体的互补分量。在给定锚点更新保留比例的固定目标下，SURGE无需测试即可直接从权重本身确定块大小……

    arXiv:2610.01133v1 Announce Type: cross  Abstract: Scaling reasoning typically spends more compute on reinforcement learning (RL) or on inference. We show that a completed RL training history can yield policies stronger than the checkpoints visited by its optimizer. We call this policy-space scaling: expanding the deployable policy set accessible from a fixed RL history, without extending training or increasing per-query inference computation. We instantiate it with SURGE (Scaling Up RL Gradient-free via Eigenspace fusion). SURGE combines two checkpoints from the same RL run: a high-accuracy anchor and a competitive donor that generates shorter responses. It expresses both checkpoints as changes from their shared initialization, then spectrally decomposes the anchor's update to retain its dominant component and incorporate the donor's complementary component. With a fixed target for how much of the anchor update to retain, SURGE determines the block size from the weights without testin
    
[^382]: 面向股票收益预测忠实大语言模型叙述的推理外化方法

    Reasoning Externalization for Faithful Large Language Model Narratives of Stock Return Predictions

    [https://arxiv.org/abs/2609.38869](https://arxiv.org/abs/2609.38869)

    该论文提出了一个结合时序SHAP证据与历史市场状态类比的LLM叙述框架，通过渐进式推理外化减少大语言模型在推断数值变化时的错误，从而提升股票收益预测解释叙述的忠实性。

    

    在金融领域，解释机器学习预测至关重要，然而可解释人工智能的数值输出对于非专业人士而言往往难以理解。虽然大语言模型（LLM）能够将这些输出转换为自然语言，但在推断数值变化和特征关系时可能会产生错误。我们提出了一个用于横截面股票收益预测的大语言模型叙述框架，该框架将时序SHAP（Shapley加性解释）证据与历史市场状态类比相结合。时序证据用于跟踪XGBoost模型在六个月内归一化全局SHAP重要性的变化。历史类比是指SHAP重要性发生相似变化的过去时期，其模型表现及随后的市场收益被作为对比背景提供。利用该框架，我们对渐进式推理外化进行了受控研究，依次提供原始SHAP序列、确定性时序描述……

    arXiv:2609.38869v1 Announce Type: new  Abstract: In finance, interpreting machine learning predictions is essential, yet the numerical outputs of explainable AI can be difficult for non-experts to understand. While large language models (LLMs) can translate these outputs into natural language, they may produce errors when inferring numerical changes and feature relations. We propose an LLM narrative framework for cross-sectional stock return prediction that combines temporal Shapley additive explanations (SHAP) evidence with historical regime analogs. Temporal evidence tracks changes in the normalized global SHAP importance of an XGBoost model over six months. Historical analogs are past periods with similar changes in SHAP importance, their model performance and subsequent market returns are provided as comparative context. Using this framework, we conduct a controlled study of progressive reasoning externalization, sequentially providing raw SHAP sequences, deterministic temporal des
    
[^383]: KlinikeBench：超越诊断准确性的语言模型评估

    KlinikeBench: Evaluating Language Models Beyond Diagnostic Accuracy

    [https://arxiv.org/abs/2609.38480](https://arxiv.org/abs/2609.38480)

    KlinikeBench是一个包含333个由临床医生编写的任务的基准，通过沙盒环境中的虚拟患者交互，评估语言模型在信息收集和临床评估方面超越单纯诊断准确性的综合临床能力。

    

    大多数临床基准测试使用完整的病例描述来评估语言模型（LM）的诊断能力。然而在临床实践中，患者以不同的方式呈现信息，临床医生必须获取相关病史并确定需要进行哪些检查，才能做出诊断。因此，仅凭诊断准确性无法判断智能体是否收集了必要的信息或进行了适当的临床评估。此外，现有基准缺乏专业临床医生的验证。为了填补这一空白，我们推出了KlinikeBench，这是一个包含333个由临床医生编写的任务的基准，每个任务都提供了一个隔离的沙盒环境，其中包含虚拟患者、临床工具和针对特定任务的成功标准。超过35名临床医生参与了病例编写和基准评估。在一项实证研究中，临床医生对模拟对话的平均质量评分高于参考对话，这表明……

    arXiv:2609.38480v1 Announce Type: cross  Abstract: Most clinical benchmarks evaluate language models (LMs) on diagnosis using complete case descriptions. In clinical practice, however, patients present information in different ways, and clinicians must obtain relevant history and determine which examinations are needed before reaching a diagnosis. Diagnostic accuracy alone therefore cannot establish whether an agent gathered essential information or conducted an appropriate clinical assessment. Furthermore, existing benchmarks lack professional clinicians' verification. To address this gap, we introduce KlinikeBench, a benchmark of 333 clinician-authored tasks, each providing an isolated sandbox environment with a virtual patient, clinical tools, and task-specific success criteria. More than 35 clinicians contributed to case authoring and benchmark evaluation. In an empirical study, clinicians gave simulated dialogues higher mean quality ratings than reference conversations, which is a
    
[^384]: JudgeProfile：理解并引导大语言模型评判者的主观性

    JudgeProfile: Understanding and Steering Subjectivity in LLM Judges

    [https://arxiv.org/abs/2609.36705](https://arxiv.org/abs/2609.36705)

    该论文提出JudgeProfile框架，将大语言模型评判者的评估分解为感知与优先级排序两个层面，发现评判者在感知层面存在隐藏共识而在属性权重上存在个体差异，从而实现对评判者主观性的理解与引导。

    

    大语言模型（LLM）评判者本质上具有主观性，在成对比较中，当两个选项都没有客观错误时，它们往往会偏好不同的回答。为了研究这种主观性，我们提出了JudgeProfile框架，该框架将大语言模型的评估过程剖析为两个层面：感知（评判者如何在清晰度、正确性和细节等特定属性上比较两个回答）和优先级排序（每个属性对最终选择的影响程度）。我们构建了SubjectiveSet数据集，其中包含来自17个公开数据源的50,013个回答对，由21个大语言模型评判者围绕87个属性进行评估。我们在感知层面发现了一个隐藏的共识：即使评判者的总体选择存在分歧，它们在属性判断上仍然经常达成一致。基于这种分离，我们首先利用从各评判者自身总体选择中估计出的属性权重来刻画其优先级排序。即使基于相同的属性判断进行估计，不同评判者的这些权重也各不相同。随后，我们学习新的权重（摘要到此截断）。

    arXiv:2609.36705v1 Announce Type: new  Abstract: LLM judges are inherently subjective, often favoring different responses in pairwise comparison when neither option is objectively wrong. To study this subjectivity, we introduce JudgeProfile, a framework that dissects LLM evaluation into perception (how a judge compares two responses across specific attributes like clarity, correctness, and detail) and prioritization (how much each attribute influences the final choice). We curate SubjectiveSet, a dataset of 50,013 response pairs from 17 public data sources, evaluated by 21 LLM judges across 87 attributes. We find a hidden consensus in perception: judges frequently agree on attribute judgments even when their overall choices diverge. Building on this separation, we first characterize each judge's prioritization using attribute weights estimated from its own overall choices. These weights differ across judges even when estimated from the same attribute judgments. We then learn new weight
    
[^385]: MLToolBench：学习用于机器学习开发的工具增强智能体

    MLToolBench: Learning Tool-Augmented Agents for Machine Learning Development

    [https://arxiv.org/abs/2609.36679](https://arxiv.org/abs/2609.36679)

    该论文提出MLToolBench可执行诊断工具套件和SPICE回合级奖励方法（通过特权上下文衡量工具动作的价值），结合SFT与RL流水线，训练机器学习开发智能体学会何时使用诊断工具并依据诊断结果采取行动。

    

    机器学习工程（MLE）智能体已取得重大进展，但通过机器学习实验进行学习在时间和计算上的代价仍然高昂。合成环境降低了这些成本，同时在数据和实验设置中引入了需要针对具体任务进行诊断的变化。仅仅让智能体能够访问诊断工具，并不能保证它们学会何时使用这些工具，或如何根据诊断结果采取行动。我们提出了ToolMLBench，这是一套用于数据检查、代码验证和实验诊断的可执行工具，并配备了用于学习使用这些工具的SFT和RL流水线。诊断调用所获取的证据，其价值取决于后续决策，因此最终结果对于强化哪些调用提供的指导有限。我们用SPICE来应对这一挑战：SPICE衡量特权上下文如何改变采样出的工具动作的可能性，并将这一差异作为回合级奖励与最终结果一同使用。我们……（原文摘要在此处截断）

    arXiv:2609.36679v1 Announce Type: new  Abstract: Machine learning engineering (MLE) agents have made substantial progress, but learning through ML experimentation remains costly in time and computation. Synthetic environments reduce these costs while introducing variations in data and experimental settings that require task-specific diagnosis. Access to diagnostic tools alone does not ensure that agents learn when to use them or how to act on their findings. We introduce ToolMLBench, a suite of executable tools for data inspection, code verification, and experiment diagnosis, together with an SFT and RL pipeline for learning their use. Diagnostic calls acquire evidence whose value depends on subsequent decisions, so final outcomes provide limited guidance on which calls to reinforce. We address this challenge with SPICE, which measures how privileged context changes the likelihood of a sampled tool action and uses this difference as a turn-level reward alongside the final outcome. We t
    
[^386]: FineART：面向双手操作的细粒度标注机器人轨迹数据集与视觉-语言-动作模型

    FineART: Fine-grained Annotated Robotic Trajectory Dataset and Vision-Language-Action Model for Bimanual Manipulation

    [https://arxiv.org/abs/2609.36416](https://arxiv.org/abs/2609.36416)

    本文提出了包含40,543个回合、1,718小时和533,913个密集子任务标注的双手操作数据集FineART，以及能自主预测下一个子任务的视觉-语言-动作模型FineART-VLA，显著提升了长时程双手操作任务的成功率。

    

    在真实世界环境中运行的机器人必须执行复杂的多步骤、长时程双手任务，而非单一、孤立的动作。当前的操控数据集难以支撑这种能力：尽管单臂数据集已达到数十万条轨迹的规模，但它们通常每个回合只提供一条高级指令，而少数标注了子任务的双手操作工作也仅标注了其数据时长的一小部分。我们提出了FineART，这是一个密集标注的双手操作数据集，包含40,543个回合、1,718小时时长以及151个任务中的533,913个子任务。我们还引入了FineART-VLA，一种能够预测自身下一个子任务的视觉-语言-动作策略，并表明以这种方式进行中期训练能带来显著提升。具体而言，空间消歧任务的成功率从32.0%提升到100.0%，而借助逐步的人类子任务指导，模型在未见过的长时程任务上的成功率从16.0%提升到76.0%。

    arXiv:2609.36416v1 Announce Type: cross  Abstract: Robots operating in real-world environments must execute complex, multi-step bimanual tasks over long horizons rather than single, isolated actions. Current manipulation datasets struggle to support this capability: although single-arm datasets reach hundreds of thousands of trajectories, they typically provide only one high-level instruction per episode while the rare bimanual effort that does label subtasks annotates only a fraction of its hours. We present FineART, a densely annotated bimanual manipulation dataset of 40,543 episodes, 1,718 hours, and 533,913 subtasks across 151 tasks. We also introduce FineART-VLA, a vision-language-action policy that predicts its own next subtask, and show that mid-training it this way yields substantial gains. Specifically, success on a spatial disambiguation task increases from 32.0% to 100.0%, and step-by-step human subtask guidance lifts success on an unseen long-horizon task from 16.0% to 76.0
    
[^387]: 基于人工智能的红外减除

    Infrared Subtraction with Artificial Intelligence

    [https://arxiv.org/abs/2609.36007](https://arxiv.org/abs/2609.36007)

    该论文提出了一种由大语言模型在人类物理指导下开发的局域红外减除方法，通过分离可积辐射项与玻恩接触项，实现了无切片参数、可复用低阶计算和有效场理论预言的局域减除公式。

    

    我们提出了由人工智能开发的局域红外减除方法，该方法建立在向玻恩投影和有效场理论（EFT）匹配的基础上。该框架将一个可积的辐射项与玻恩运动学下的有限贡献（称为玻恩接触项）分离开来。接触项通过有效场理论在分辨率观测量（如N-喷注性 τ_N）中的奇异分布来确定。在人类物理指导下，一个大语言模型（LLM）开发了两种实现方案。其中一种使用神经网络进行相空间投影，并通过与有效场理论累积量的匹配来拟合接触项；另一种采用解析构造方法，在保持玻恩动量固定的同时对辐射进行积分，将有效场理论的 δ(τ_N) 系数与有限的四维辐射积分相结合，直接计算接触项。这给出了一种不含切片参数的局域减除公式，同时可复用已有的低阶辐射计算和有效场理论奇异预言。作为演示，我们……

    arXiv:2609.36007v1 Announce Type: cross  Abstract: We present AI-developed local infrared subtraction, building on projection to Born and EFT matching. The framework separates an integrable radiation term from a finite contribution at Born kinematics, referred to as the Born contact. The contact is determined using the EFT singular distribution in a resolution observable such as N-jettiness $\tau_N$. Under human physics guidance, an LLM develops two implementations. One uses a neural network for phase space projection and fits the contact by matching to EFT cumulants. The other uses an analytic construction that keeps the Born momenta fixed while integrating over radiation. It combines the EFT $\delta(\tau_N)$ coefficient with finite 4-dimensional radiation integrals to calculate the contact term directly. This gives a local subtraction formula without a slicing parameter, while reusing existing lower-order radiation calculations and EFT singular predictions. As a demonstration, we rec
    
[^388]: AX 是新的 AEO

    AX is the New AEO

    [https://arxiv.org/abs/2609.34951](https://arxiv.org/abs/2609.34951)

    该论文提出“代理体验（AX）”——即AI智能体能否顺利抓取并阅读企业自身网站——正在取代答案引擎优化（AEO）成为决定AI推荐结果的关键因素，并通过超过3.7万次智能体买家旅程实验加以验证。

    

    2023年，AI模型依靠训练数据回答问题，数据耗尽时便产生幻觉，因此企业被告知要预先植入相关知识。此后，模型的训练知识已让位于实时网络搜索，相关建议也随之转移：答案引擎优化（AEO）如今建议企业在论坛帖子、清单文章和站外引用中散布“面包屑”信息，以便AI引擎更容易将它们展示和推荐出来。但仅仅被展示出来已经不够了：智能体会打开搜索结果、阅读内容后才做决策，一个买家问题会驱使它经历多轮搜索与抓取。在这一深入挖掘环节中，决定成败的是智能体能否成功抓取并阅读企业自己的网站——这就是代理体验（AX）。我们主张，AX 是新的 AEO。我们在四个独立测试框架上，针对1,056家真实企业开展了37,927次智能体旅程实验，每次旅程都是一个关于某企业的买家问题，并在知名度、模型先验知识以及（原文在此截断）……等方面进行了匹配。

    arXiv:2609.34951v2 Announce Type: replace  Abstract: In 2023, AI models answered from training data and hallucinated when it ran out, and businesses were told to seed that knowledge. Models' training knowledge has since given way to live web search, and the advice followed it there: answer-engine optimization, or AEO, now tells businesses to scatter breadcrumbs across forum threads, listicles, and off-site citations, so AI engines are likelier to surface and recommend them. But being surfaced is no longer enough: an agent opens the results and reads them before deciding, and one buyer question sends it through several rounds of search and fetch. What decides the outcome at this drill-down step is whether the agent can fetch and read the business's own site: agent experience (AX). We argue that AX is the new AEO. We run 37,927 agent journeys, each a buyer question about a business, across four independent harnesses over 1,056 real businesses, matched on fame, prior model knowledge, and 
    
[^389]: 水域环境监测中信息路径规划的校准不确定性

    Calibrated Uncertainty for Informative Path Planning in Aquatic Environmental Monitoring

    [https://arxiv.org/abs/2609.34577](https://arxiv.org/abs/2609.34577)

    用校准良好的深度集成替代高斯过程可为水域环境监测的信息路径规划提供更可靠的不确定性估计，在原油泄漏场景模拟中将归一化重构误差降低83%。

    

    arXiv:2609.34577v2 公告类型：替换 摘要：标量场重构的信息路径规划利用预测不确定性来引导感知载体前往信息量最大的位置。高斯过程（Gaussian Process）能够提供这一信号，但其平稳各向同性核对于诸如原油泄漏等非均质现象存在模型设定偏差，产生校准不良的估计，从而降低规划效果。我们研究了用校准良好的深度集成（Deep Ensemble）替代高斯过程是否能改善路径规划结果，以及不确定性质量是否与规划算法的选择存在交互作用。五种策略（$\epsilon$-贪心、价值贪心、不确定性贪心、蒙特卡洛树搜索和滚动时域定向）共享同一个基于物理原油泄漏模拟训练的深度集成主干。在留出的随机泄漏场景上，深度集成相对于高斯过程基线将归一化重构误差降低了83%。至关重要的是，校准良好的（摘要在此处截断）

    arXiv:2609.34577v2 Announce Type: replace  Abstract: Informative Path Planning for scalar field reconstruction uses predictive uncertainty to direct sensing vehicles toward maximally informative locations. Gaussian Processes provide this signal but their stationary isotropic kernels are misspecified for non-homogeneous phenomena such as oil spills, producing miscalibrated estimates that degrade planning. We investigate whether replacing the Gaussian Process with a well-calibrated Deep Ensemble improves path planning outcomes, and whether uncertainty quality interacts with the choice of planning algorithm. Five strategies ($\epsilon$-Greedy, Value Greedy, Uncertainty Greedy, Monte Carlo Tree Search, and Receding Horizon Orienteering) share a common Deep Ensemble backbone trained on physics-based oil spill simulations. On held-out stochastic spill scenarios, the Deep Ensemble reduces normalised reconstruction error by $83\%$ relative to the Gaussian Process baseline. Crucially, well-cali
    
[^390]: SWE-Game：编程智能体能否构建我们想要的游戏？

    SWE-Game: Can Coding Agents Build the Games We Want?

    [https://arxiv.org/abs/2609.33678](https://arxiv.org/abs/2609.33678)

    SWE-Game是一个基于41个可执行Godot游戏构建的编程智能体基准，涵盖从简报开发、文档实现、骨架补全、故障修复到跨引擎移植五种任务，结果显示最强模型Opus5的总分也不足60分，表明智能体构建完整游戏的能力仍有巨大提升空间。

    

    我们推出了SWE-Game，这是一个包含247个任务的基准测试，基于41个可执行的Godot参考游戏，涵盖2D和3D共13种游戏玩法类别。五种任务类型包括：根据简报进行游戏开发、根据游戏设计文档进行实现、骨架代码补全、83个注入故障案例的修复，以及Godot到Unity的移植。参考材料规定了预期的游戏玩法，而共享的插桩接口允许评估方自己的驱动程序和探针执行操作并观察独立实现的游戏。评估结合了引擎状态检查、经认证的参考输入重放以及智能体编写的功能演示，以评估游戏机制的正确性、可演示的可玩性，以及修复后的行为恢复与保持。针对特定游戏的视觉语言评分标准则单独评估游戏呈现效果。在六个模型中，Opus5在所有五种任务类型中均取得最高的总分。最佳总分仍低于100分中的60分……

    arXiv:2609.33678v2 Announce Type: replace  Abstract: We introduce SWE-Game, a benchmark of 247 tasks grounded in 41 executable reference Godot games spanning 13 gameplay categories in 2D and 3D. Five task types cover development from a brief, implementation from a game design document, skeleton completion, repair of 83 injected-fault cases, and Godot-to-Unity porting. Reference materials specify the intended gameplay, while a shared instrumentation interface lets evaluator-owned drivers and probes execute actions and observe independently implemented games. Evaluation combines engine-state checks, certified reference-input replay, and agent-authored feature demonstrations to assess mechanic correctness, demonstrated playability, and behavioral restoration and preservation after repairs. Game-specific vision-language rubrics separately assess presentation. Across six models, Opus5 achieves the highest overall score in all five task types. Best overall scores remain below 60 out of 100 a
    
[^391]: 动作塑形：策略会吸收其所能表达的内容

    Action Shaping: Policies Absorb What They Can Express

    [https://arxiv.org/abs/2609.32752](https://arxiv.org/abs/2609.32752)

    论文提出“动作塑形”原理：训练时加入的动作偏移量若能被策略输出层精确表达，策略便会将其完全吸收，从而可在部署时安全移除该偏移而不损失回报，其最简实现为零初始化线性头配合可学习门控，门控会自行先升后降。

    

    奖励塑形有一个定理：基于势函数的修正项可以在不改变最优策略的前提下被移除。然而在动作通道上进行同样的操作——训练时加入一个偏移量、部署时将其去除——却没有相应的定理作为保障。没有任何机制能够抵消动作偏移，因此这种修正要么在部署时被保留，要么在缺乏保证的情况下被移除。我们将其称为动作塑形，并阐述其原理：可训练的策略会吸收其自身输出层能够精确复现的偏移量，这正是我们所称的“表达”；被吸收的偏移量可以在回报完整无损的情况下被移除。其最简实现是一个位于可学习门控之后的零初始化线性头，附加于通过学习到的动作价值函数进行训练的执行器之上，无需任何惩罚项或调度机制。无论对确定性还是随机性执行器，门控值都会自行先升高后回落，并且在20个任务上移除该线性头几乎不造成任何性能损失。其关键条件是精确复现，而非容量大小：一个容量更大的非线性头……

    arXiv:2609.32752v2 Announce Type: replace  Abstract: Reward shaping has a theorem: a potential-based term can be removed without changing the optimal policy. The same practice on the action channel, an offset added in training and dropped at deployment, has no theorem. Nothing cancels an action offset, so the correction is kept at deployment or removed without a guarantee. We call it action shaping and state its principle. A trainable policy absorbs an offset its own output layer can reproduce exactly, which is what we mean by express; what is absorbed can be removed with the return intact. Its minimal instance is a zero-initialized linear head behind a learnable gate, added to an actor that trains through a learned action-value function, with no penalty or schedule. The gate rises and then falls on its own, for deterministic and stochastic actors alike, and on 20 tasks removing the head costs almost nothing. The condition is exact reproduction, not capacity: a nonlinear head with more
    
[^392]: 几何诱导软状态抽象的预测极限与Koopman封闭性

    Prediction Limits and Koopman Closure of Geometry-Induced Soft State Abstractions

    [https://arxiv.org/abs/2609.32652](https://arxiv.org/abs/2609.32652)

    该论文为几何诱导软状态抽象的线性预测精度建立了无需拟合预测矩阵即可由独立评估数据计算的有限样本下置信界，并揭示了核仿射包机器坐标的重构分数裕度与预测误差界之间的联系。

    

    软状态表示为每个状态分配一个非负类别权重向量，且权重之和为一。我们研究这些权重的构造方式与状态动力学如何共同决定线性预测的精度。对于任意固定的可测表示，我们推导出了在具有指定谱范数极限的矩阵类中，最小总体均方根预测误差的有限样本下置信界。该界比较了每个参考类内后继坐标的变化与软输入可能带来的改进。该界可由独立的评估数据对计算得出，无需拟合预测矩阵。若该界高于选定的容差，则可排除整个矩阵类达到该容差的可能性；若该界为零，则结论不确定。对于使用核仿射包机器构造的坐标，重构分数裕度控制了与参考标签的分歧，并进入预测误差的界中。

    arXiv:2609.32652v2 Announce Type: replace  Abstract: A soft state representation assigns each state a vector of nonnegative class weights that sum to one. We study how the construction of these weights and the state dynamics jointly determine the accuracy of linear prediction. For any fixed measurable representation, we derive a finite-sample lower confidence bound on the smallest population root-mean-square prediction error among matrices with a specified spectral-norm limit. The bound compares variation in successor coordinates within each reference class with the improvement that soft inputs could provide. It is computed from independent evaluation pairs without fitting a prediction matrix. A bound above a chosen tolerance rules out that tolerance for the entire matrix class; a zero bound is inconclusive.   For coordinates constructed using Kernel Affine Hull Machines, reconstruction-score margins control disagreement with reference labels and enter bounds on prediction error. Under
    
[^393]: 主体，而非作者：智能体数据空间中的作者身份危害

    Subjects, Not Authors: The Authorship Hazard in Agentic Dataspaces

    [https://arxiv.org/abs/2609.30614](https://arxiv.org/abs/2609.30614)

    该论文提出“作者身份危害”概念并确立核心原则：LLM智能体应始终是数据空间治理平面的主体而非作者，其发布授权通道须在构造上被关闭，其起草内容的影响则作为执行问题加以管控。

    

    数据空间连接器决定是否允许一次传输发生，而不决定所传输的值包含什么内容——这对契约式应用是可容忍的，但对能够组合工具调用并派生子智能体的LLM智能体而言则不然。关于生成治理制品的智能体的研究评估的是输出质量；而“谁有权授权制品投入使用”这一问题则落在该文献与治理文献之间的空白地带，双方均未涉及。已发布的策略正是数据空间决策点所强制执行的内容，因此发布是一个治理事件，而一个既是策略主体又是策略作者的智能体，就是在撰写约束自身的规范。我们将此称为“作者身份危害”，并提出一条原则：智能体是治理平面的主体，绝非其作者。其通往发布的授权通道在构造上被关闭；其影响通道——即起草供人类批准的内容——则被视为一个执行问题。在一个冻结的智能体草稿语料库上，未经批准而发布……

    arXiv:2609.30614v1 Announce Type: cross  Abstract: Dataspace connectors decide whether a transfer may occur, not what the transferred value contains, tolerable for contracted applications, not for LLM agents that compose tool calls and spawn sub-agents. Research on agents that generate governance artifacts evaluates output quality; who may authorize an artifact for use falls between that literature and the governance literature, and neither owns it. A published policy is what a dataspace's decision point enforces, so publication is a governance event, and an agent that is both policy subject and policy author writes the norms that bind it. We name this the authorship hazard and state one principle: an agent is a subject of the governance plane, never an author of it. Its authorization channel to publication is closed by construction; its influence channel, drafting what humans approve, is treated as an enforcement problem. On a frozen corpus of agent drafts, publishing without approval
    
[^394]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^395]: 面向阿尔茨海默病检测的在线手写片段级风险发现

    Segment-Level Risk Discovery in Online Handwriting for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2609.29384](https://arxiv.org/abs/2609.29384)

    该论文提出NormPaST-Risk网络，将阿尔茨海默病在线手写检测从整体轨迹表示重新表述为局部疾病相关片段发现，通过多尺度时间编码与选择性纸-空状态空间建模实现可解释的片段级风险识别。

    

    在线手写为阿尔茨海默病（AD）检测提供了一种无创且低成本的行为生物标志物，因为它同时反映了认知规划与精细运动控制。现有的基于手写的AD检测方法通常依赖于全局轨迹特征或整样本表示，而这些特征容易受到个体书写风格、任务特定变化和采集噪声的显著影响。本文提出了NormPaST-Risk，一个健康规范化的纸-空选择性轨迹状态空间风险网络，用于从在线手写中进行可解释的AD检测。与将整个轨迹视为单一整体表示不同，我们的方法将AD手写检测重新表述为局部疾病相关片段的发现问题。具体而言，多尺度时间编码器在不同时间分辨率下捕获笔画动态，而选择性的纸-空状态空间编码器对长程手写进程进行建模。

    arXiv:2609.29384v1 Announce Type: cross  Abstract: Online handwriting provides a non-invasive and low-cost behavioral biomarker for Alzheimer's disease (AD) detection, as it reflects both cognitive planning and fine motor control. Existing handwriting-based AD detection methods usually rely on global trajectory features or whole-sample representations, which can be strongly affected by individual writing style, task-specific variation, and acquisition noise. In this paper, we propose NormPaST-Risk, a healthy-normative Paper-Air selective trajectory state-space risk network for interpretable AD detection from online handwriting. Instead of treating the entire trajectory as a single holistic representation, our method reformulates AD handwriting detection as local disease-relevant segment discovery. Specifically, a multi-scale temporal encoder captures stroke dynamics at different temporal resolutions, while a selective Paper-Air state-space encoder models long-range handwriting progress
    
[^396]: TwinCheck：面向有状态工具代理的证据支撑型“负孪生”验证

    TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents

    [https://arxiv.org/abs/2609.26911](https://arxiv.org/abs/2609.26911)

    TwinCheck提出了一种推理时验证策略，通过构建基于证据的“负孪生”反事实替代方案，仅在满足证据条件、通过结构检查并在顺序无关的成对验证中胜出时才替换智能体的工具调用，从而在不引入新失败的前提下提升有状态工具代理的多轮任务成功率。

    

    arXiv:2609.26911v1 公告类型：新论文 摘要：单个在局部看似合理的工具调用，可能会让原本成功的智能体轨迹偏离正轨。然而，仅凭怀疑并不足以成为干预的理由，因为替换本身反而可能引入验证本欲防止的失败。我们提出TwinCheck，一种推理时验证策略，仅当轨迹满足与“轨迹局部故障假设”相关联的证据条件时，才考虑进行替换。该方法会构建一个基于轨迹的反事实替代方案——即“负孪生”（negative twin），并且仅当该孪生通过结构检查、且成对验证器在两种候选顺序下均更偏好它时，才替换智能体的原始提议。在配对评估中，精确重放（exact replay）将智能体已解析的响应与动作保持固定，直至第一次被接受的替换，从而将干预效应与重采样效应分离开来。在对159个具备完整精确重放对的多轮BFCL V4任务的主要分析中，完整策略提升了GPT-5.6 So……（原文摘要至此截断）

    arXiv:2609.26911v1 Announce Type: new  Abstract: A single locally plausible tool call can derail an otherwise successful agent trajectory. Suspicion alone does not justify intervention, because the replacement itself can introduce the very failure verification is meant to prevent. We introduce TwinCheck, an inference-time verification policy that considers replacement only when the trace satisfies an evidence condition tied to a trace-local failure hypothesis. It constructs a trace-grounded counterfactual alternative, a negative twin, and replaces the agent's proposal only if the twin passes structural checks and the pairwise verifier prefers it in both candidate orders. For paired evaluation, exact replay holds the agent's parsed responses and actions fixed until the first accepted replacement, separating intervention effects from resampling. In the primary analysis of 159 multi-turn BFCL V4 tasks with complete exact-replay pairs, the complete policy raises task success for GPT-5.6 So
    
[^397]: JEV作为裁判：自信时接受，不确定时升级

    JEV-as-a-Judge: Accept When Confident, Escalate When Unsure

    [https://arxiv.org/abs/2609.26550](https://arxiv.org/abs/2609.26550)

    提出仅返回标签概率的低成本裁判JEV，通过置信度阈值机制将高置信判定直接接受、不确定判定升级至推理型裁判，从而在费用仅为GPT-6的41%的情况下，实现了比GPT-6高出0.9个百分点的评估准确率。

    

    LLM作为裁判可以扩展评估规模，但推理型裁判速度慢且成本高。我们研究了JEV作为裁判：使用JEV进行评估，JEV是一种仅做决策的裁判，它返回标签概率而非文本，其置信度决定是接受其判定还是升级到推理型裁判。在与十六个生成式裁判和奖励模型裁判的对比中，并经过盲测人工裁决，只要判定可以直接从文本中读出，JEV与GPT-6的差距在三个百分点以内，而费用仅为GPT-6的0.36%，中位延迟为0.15秒；而在必须通过推导才能得出判定的情况下（如数学、代码和逻辑），JEV则落后。其置信度恰好标记了这一能力边界。通过预先固定阈值，接受高置信度的判定并将其余判定升级，在1,610个保留样本对上比GPT-6的准确率高0.9个百分点，而费用仅为GPT-6的41%；在两项新工作负载的预先指定的实时测试中，该级联系统与GPT-6的准确率完全匹配。置信度路由在风格对抗性样本上会减弱……（摘要在此处被截断）

    arXiv:2609.26550v3 Announce Type: replace  Abstract: LLM-as-a-judge scales evaluation, but reasoning judges are slow and costly. We study JEV-as-a-Judge: evaluation with JEV, a decision-only judge that returns label probabilities instead of text, and whose confidence decides whether to accept its verdict or escalate to a reasoning judge. Against sixteen generative and reward-model judges, with blinded human adjudication, JEV comes within three points of GPT-6 wherever a verdict can be read off the text, at 0.36% of its fee and a 0.15-second median latency, and falls behind where the verdict must be derived, as in math, code, and logic. Its confidence marks this boundary. With a threshold frozen in advance, accepting confident verdicts and escalating the rest is 0.9 points more accurate than GPT-6 on 1,610 held-out pairs at 41% of its fee, and in a pre-specified live test on two new workloads the cascade matches GPT-6's accuracy exactly. Confidence routing weakens on style-adversarial p
    
[^398]: 儿童如何设计并推理值得信赖的AI聊天机器人

    How Children Design and Reason about Trustworthy AI Chatbots

    [https://arxiv.org/abs/2609.25244](https://arxiv.org/abs/2609.25244)

    本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。

    

    儿童越来越多地与AI聊天机器人互动，因此信任校准成为AI素养的重要组成部分。以往研究主要将儿童对AI的信任视为用户评估他人构建的系统，而非作为自己聊天机器人的设计者。我们开发了一个聊天机器人构建环境，支持调节与信任相关的特质（如自信度、透明度、正式程度、果断性）、规则和角色设定。我们对115名学习者（8-18岁）开展了混合方法研究，他们共制作了119个聊天机器人。我们考察了儿童如何配置他们的聊天机器人、如何推理可信度，以及聊天机器人的行为与其设计的契合程度。年龄较小的学生（10-13岁）设置的自信度显著高于年龄较大的学生（14-18岁），部分学生还刻意构建了会故意给出错误答案的聊天机器人，却仍认为其“值得信赖”，理由是聊天机器人做了它被设计要做的事情。低龄学生将信任等同于目的（摘要在此处截断）。

    arXiv:2609.25244v2 Announce Type: replace-cross  Abstract: Children increasingly interact with AI chatbots, making trust calibration essential to AI literacy. Prior research has examined children's trust in AI mainly as users evaluating systems built by others, rather than as designers of their own chatbots. We developed a chatbot-building environment with adjustable trust-relevant traits (e.g., confidence, transparency, formality, assertiveness), rules, and persona. We conducted mixed-methods study with 115 learners (ages 8-18) who made 119 chatbots. We examined how children configured their chatbots, reasoned about trustworthiness, and how closely chatbot behavior aligned with their designs. Younger students (age 10-13) set significantly higher confidence than older students (age 14-18), and some deliberately built chatbots that gave wrong answers on purpose, yet still called them trustworthy, arguing that a chatbot does what it was built to do. Younger students equated trust with pu
    
[^399]: ValueDiff：面向注意力汇聚抑制型大语言模型的价值几何KV缓存淘汰方法

    ValueDiff: Value-Geometric KV Cache Eviction for Sink-Suppressed LLMs

    [https://arxiv.org/abs/2609.23314](https://arxiv.org/abs/2609.23314)

    针对因QK归一化等技术导致注意力汇聚减弱的现代大语言模型，提出基于价值向量与缓存均值L2偏差进行token排序的ValueDiff淘汰方法，在2k-4k token预算下可保留密集注意力88-99%的性能。

    

    采用QK归一化、门控注意力、可学习注意力汇聚或logit软截断的现代大语言模型表现出更弱的持久注意力汇聚现象，而现有的KV缓存淘汰方法主要依赖这一现象。我们观察到，在这些模型中，更弱的汇聚现象伴随着相对于键向量离散度而言更大的价值向量离散度。受这种价值侧离散度的启发，我们提出了ValueDiff，一种价值几何淘汰方法，它根据token的价值向量与缓存均值之间的L2偏差对token进行排序。在关于未来注意力的最大熵假设下，同样的分数可作为最小扰动淘汰方案推导得出。我们在固定缓存预算下进行评估，在预填充阶段的每个块边界以及生成阶段的每个解码步骤执行淘汰。在RULER基准上，在紧凑的2k token预算下，ValueDiff在七个汇聚抑制模型上保留了密集注意力88--99%的性能（在7个模型中的6个上表现最佳）。在LongBench基准上，在4k预算下，ValueDiff平均保留92%的性能。

    arXiv:2609.23314v1 Announce Type: new  Abstract: Modern LLMs with QK-normalization, gated attention, learned attention sinks, or logit softcapping exhibit weaker persistent attention sinks, on which existing KV cache eviction methods primarily rely. We observe that across these models, weaker sinks co-occur with greater value-vector dispersion relative to key-vector dispersion. Motivated by this value-side dispersion, we present ValueDiff, a value-geometric eviction that ranks tokens by the L2 deviation of their value vectors from the cache mean. The same score arises as the minimal-disturbance eviction under a max-entropy assumption about future attention. We evaluate under fixed cache budgets, with eviction at every block boundary during prefill and at every decoding step during generation. On RULER at a tight 2k token budget, ValueDiff retains 88--99\% of dense across seven sink-suppressed models (best on 6 out of 7). On LongBench at the 4k budget, ValueDiff averages 92\% retention 
    
[^400]: 局部稀疏性实现无监督的大语言模型安全检测

    Local Sparsity Enables Unsupervised LLM Safety Detection

    [https://arxiv.org/abs/2609.20129](https://arxiv.org/abs/2609.20129)

    本文利用稀疏自编码器概念空间中的局部稀疏性这一关键洞察，提出了一个无需不安全训练数据的无监督LLM安全异常检测框架，并有理论支撑。

    

    大语言模型（LLM）的部署时安全方法主要是有监督的，并假设能够获取不安全的训练数据。然而，新的攻击和伤害类别不断出现，而以这种有监督方式训练的模型无法捕获这些内容。另一种方法是从异常检测的视角来看待这个问题，即仅依赖于对安全数据的建模并标记分布外输入。然而，LLM激活位于高维空间中，这引发了关于异常检测在统计上是否可行的担忧。我们证明，在线性表示假设（LRH）下，确实存在希望。在通常通过稀疏自编码器（SAE）恢复的LRH概念空间中，邻近的点共享一个较小的共同激活支持集。利用这一局部稀疏性洞察，我们提出了一个基于局部掩码SAE的异常检测框架，并提供了理论依据的支持。我们进行了验证……

    arXiv:2609.20129v1 Announce Type: cross  Abstract: Deployment-time safety methods for large language models (LLMs) are predominantly supervised and assume access to unsafe training data. Nevertheless, new attacks and harm categories regularly arise, not captured by models trained in such a supervised fashion. An alternative approach is to view this problem through the lens of anomaly detection, namely, to rely solely on modeling safe data and flagging out-of-distribution inputs. However, LLM activations lie in a high-dimensional space, raising concerns about whether anomaly detection is statistically feasible. We show that, under the linear representation hypothesis (LRH), there may indeed be hope. In the LRH concept space, which is typically recovered via a sparse autoencoder (SAE), nearby points share a small common active support. Using this local sparsity insight, we propose a framework for locally masked SAE-based anomaly detection, supported by theoretical justifications. We vali
    
[^401]: CSWAM：面向世界动作模型分布外泛化的更优因果语义表示

    CSWAM: Better Causal Semantic Representations for Out-of-Distribution Generalization in World Action Models

    [https://arxiv.org/abs/2609.18462](https://arxiv.org/abs/2609.18462)

    CSWAM通过引入基于V-JEPA 2.1的因果语义专家模块，从稀疏观测历史中学习具有时间基础、少依赖外观细节的语义表示，显著提升了世界动作模型在视觉分布偏移下的泛化能力。

    

    FastWAM风格的世界动作模型支持高效的纯动作推理，但在视觉分布偏移下泛化能力较差。其面向重建的表示过度强调外观相关的细节，限制了对未见场景和物体的泛化能力。同时，由于缺乏观测历史，模型也缺少在陌生视觉条件下稳健识别任务相关状态变化与运动所需的时间证据。为解决这些局限，我们提出了因果语义世界动作模型（CSWAM），它通过一个基于V-JEPA 2.1构建的因果语义专家模块来增强FastWAM。V-JEPA能够提供对语义状态变化和运动具有时间基础的表示，且较少依赖外观特定细节。该专家从当前与过去观测构成的稀疏历史中学习其未来演化，并通过因果注意力机制将基于历史导出的上下文同时共享给视频流和动作流。

    arXiv:2609.18462v1 Announce Type: cross  Abstract: FastWAM-style world action models enable efficient action-only inference, but generalize poorly under visual distribution shifts. Their reconstruction-oriented representations emphasize appearance-specific details, limiting generalization to unseen scenes and objects. Without observation history, the model also lacks temporal evidence for robustly identifying task-relevant state changes and motion in unfamiliar visual conditions. To address these limitations, we present the Causal Semantic World Action Model (CSWAM), which augments FastWAM with a causal semantic expert built on V-JEPA 2.1. V-JEPA provides temporally grounded representations of semantic state changes and motion with less dependence on appearance-specific details. The expert learns their future evolution from a sparse history of current and past observations and shares the history-derived context with both the video and action streams through causal attention. At inferen
    
[^402]: SpliTEE：通过差分隐私GPU外包改进可信硬件上的大语言模型推理

    SpliTEE: Improving LLM Inference on Trusted Hardware with Differentially Private GPU Outsourcing

    [https://arxiv.org/abs/2609.15039](https://arxiv.org/abs/2609.15039)

    该论文提出SpliTEE，将拆分推理架构扩展到LLM推理场景，用差分隐私（而非加密）保护发送到不受信任GPU的中间表示，从而在可信硬件上实现高效且隐私安全的大语言模型推理。

    

    用户向大语言模型（LLM）提供的提示词可能包含敏感或私密信息，这些信息可能被远程部署的模型滥用，例如在重新训练过程中被无意记忆。保护用户提示词的一种方法是在可信执行环境（TEE）内执行LLM，以保证服务提供商无法访问TEE内执行的计算或与TEE交换的信息。然而，当前的TEE主要基于CPU，比针对LLM推理优化的GPU慢得多。为了解决这一问题，Tramer和Boneh（2019）提出了Slalom，该方法将神经网络推理在TEE和不受信任的GPU之间进行拆分，并对发送到GPU的中间输入进行加密。我们将这种拆分推理架构扩展到LLM推理，并改用差分隐私来保护中间输入。我们通过展示提示词（摘要在此处被截断）证明了掩盖中间表示是必要的。

    arXiv:2609.15039v1 Announce Type: cross  Abstract: User prompts provided to large language models (LLMs) may contain sensitive or private information that can be misused by remotely deployed models, such as through inadvertent memorization during retraining. One way to protect user prompts is to execute the LLM inside a trusted execution environment (TEE), with the guarantee that the service provider has no access to computations performed within or information exchanged with the TEE. However, current TEEs are primarily CPU-based and significantly slower than GPUs optimized for LLM inference. To circumvent this, Tramer and Boneh (2019) proposed Slalom, which splits neural network inference between a TEE and an untrusted GPU and encrypts intermediate inputs sent to the GPU. We extend this split-inference architecture to LLM inference and instead protect intermediate inputs using differential privacy. We show that masking intermediate representations is necessary by showing that a prompt
    
[^403]: UnitBoost：用合并算子而非模型来管理复合LLM系统

    UnitBoost: Managing Compound LLM Systems with a Merge Operator, Not a Model

    [https://arxiv.org/abs/2609.09815](https://arxiv.org/abs/2609.09815)

    UnitBoost用确定性的合并算子取代复合LLM系统中的生成式元代理，通过槽位-值提案、约束argmax和显式残差机制实现顺序无关、可溯源的系统协调，并在基准测试中超越了金标准标签选出的最佳单一候选。

    

    复合LLM系统通常通过添加一个更高层级的LLM来解决协调问题。由此产生的元代理读取各个工作者模型的输出、撰写最终答案、分配后续调用，并决定何时停止。这种方式具有很强的表达能力，但它同时也将三个控制决策集中在一个不透明、对顺序敏感的模型调用中。我们提出疑问：管理器真的必须是生成式的吗？UnitBoost用一个明确定义的元层算子取代了该模型：由任务给定的单元映射将工作者输出转换为槽位-值提案，受约束的argmax负责组装输出，而未被填充或缺乏支撑的槽位则成为下一轮的显式残差。该算子与顺序无关，能够记录单元溯源，并提供一个简单的保证：在没有耦合约束的情况下，在相同准入分数下进行的单元级最大化优于任何完整候选的选择。在三个保留基准测试中，它超越了用金标准标签选出的最佳单一候选。

    arXiv:2609.09815v1 Announce Type: cross  Abstract: Compound LLM systems often solve a coordination problem by adding a higher-level LLM. The resulting meta-agent reads workers' outputs, writes the final answer, allocates later calls, and decides when to stop. It is expressive, but it also concentrates three control decisions in an opaque, order-sensitive model call. We ask whether the manager needs to be generative at all. UnitBoost replaces that model with a defined meta-level operator: a task-given unit map turns worker outputs into slot-value proposals, a constrained argmax assembles the output, and the slots left unfilled or unsupported become an explicit residual for the next round. The operator is order-free, records unit provenance, and gives a simple guarantee: without coupling constraints, unit-wise maximization under the same admission score dominates selection of any complete candidate. On three held-out benchmarks, it exceeds the best single candidate chosen with gold label
    
[^404]: FrogNano：通过在线任务合成训练一个4B编码智能体

    FrogNano: Training a 4B Coding Agent via Online Task Synthesis

    [https://arxiv.org/abs/2609.07925](https://arxiv.org/abs/2609.07925)

    FrogNano是一个4B编码智能体，仅通过强化学习在约1500个合成任务环境中训练，其关键创新是在线任务合成流水线能在当前模型可学习性前沿生成校准任务，无需从大模型蒸馏即可训练出具有竞争力的小型编码智能体。

    

    我们提出了FrogNano，这是一个4B参数的编码智能体，旨在即使在资源受限的环境下也能高效且有效地解决软件工程（SWE）任务。它完全通过强化学习在大约1,500个包含合成任务的SWE环境中进行后训练。提升性能的一个关键要素是在线任务合成流水线，该流水线能够创建针对当前检查点可学习性前沿进行校准的任务。本报告提供了证据，表明仅使用合成任务就可以训练出具有竞争力的小型编码智能体，而无需传统的从更大模型进行蒸馏，并且在当前智能体的可学习性前沿生成任务至关重要。我们报告了训练方法的细节、跨多种环境的评估以及深入分析，为我们持续探索可在最低限度硬件上运行的轻量级且功能强大的编码智能体奠定了基础。

    arXiv:2609.07925v2 Announce Type: new  Abstract: We present FrogNano, a 4B coding agent designed to tackle software engineering (SWE) tasks efficiently and effectively, even under resource-constrained environments. It is post-trained exclusively via RL on around 1,500 SWE environments with synthetic tasks. A key ingredient for improving performance is an online task synthesis pipeline that creates tasks calibrated to the frontier of learnability for the current checkpoint. This report provides evidence that competitive small coding agents can be trained with synthetic tasks alone, without traditional distillation from larger models, and that generating tasks at the learnability frontier of the current agent is important. We report details on the training methodology, evaluations across diverse environments, and in-depth analyses, serving as a foundation for our ongoing exploration of lightweight yet capable coding agents that can run on minimal hardware.
    
[^405]: 成本感知的分层多智能体勒索软件检测与家族归因

    Cost-Aware Hierarchical Multi-Agent Ransomware Detection and Family Attribution

    [https://arxiv.org/abs/2609.04820](https://arxiv.org/abs/2609.04820)

    该论文提出一种成本感知的分层多智能体系统，以低成本静态分析优先、按需调用动态和内存分析，并引入成本模型在保证勒索软件检测与家族归因性能的同时有效平衡计算开销。

    

    勒索软件的检测与家族归因需要分析不同的模态，因为勒索软件可能使用加壳、混淆、进程操纵和运行时规避等技术。然而，传统的多模态方法通常对每个样本使用所有可用模态，导致不必要的计算成本和延迟增加。在本文中，我们提出了一种用于自适应勒索软件检测的成本感知分层多智能体系统（HMAS）。该架构将专门化的智能体组织成分层的域控制器，并由元编排器进行协调。静态分析被用作初始的低成本模态，只有当置信度不足或专家智能体之间存在分歧时，才选择性地引入额外的动态和内存模态。成本模型将模态使用和处理开销纳入考量，使编排策略能够在分析性能与计算成本之间取得平衡。

    arXiv:2609.04820v1 Announce Type: cross  Abstract: Ransomware detection and family attribution require analysis of different modalities because it can use packing, obfuscation, process manipulation and runtime evasion techniques. However, conventional multimodal usually uses all available modalities for every sample resulting in unnecessary computational cost and increased latency. In this paper, we present a Cost Aware Hierarchical Multi-Agent System (HMAS) for adaptive ransomware detection. The proposed architecture organizes specialized agents into hierarchical domain controllers coordinated by a Meta Orchestrator. Static analysis is used as the initial low-cost modality while additional dynamic and memory modality is selectively used when confidence is insufficient or specialist agents exhibit disagreement. A cost model incorporates modality use and processing overhead. It enables the orchestration policy to balance analysis performance against computational cost. A locally deploye
    
[^406]: 视觉并非开销：面向视觉语言模型无损推测解码的单遍块草拟方法

    Vision Is Not Overhead: One-Pass Block Drafting for Lossless Speculative Decoding in Vision-Language Models

    [https://arxiv.org/abs/2609.00355](https://arxiv.org/abs/2609.00355)

    该论文提出 GLANCE——首个在未修改的视觉语言模型上实现无损推测解码的单遍块草拟器，通过块扩散头零成本读取目标模型已融合的视觉-语言状态，并在一次前向传播中完成整块草拟与宽候选树验证，从而打破了草拟器因规模受限而被迫牺牲视觉信息的自我挫败循环。

    

    推测解码能够在不改变输出结果的前提下加速生成，但在视觉语言模型上，它却陷入了一种自我挫败的循环：草拟器必须保持自回归架构，因而只能维持小规模；小型草拟器无法在每一步都承担图像处理的代价，于是视觉信息被压缩、剪枝或隐藏；而被切断了图像信息的草拟器，恰恰在图像最能让文本变得可预测的地方变得最不可靠。我们提出 GLANCE——首个在未经修改的 VLM 目标模型上实现无损解码的单遍块草拟器，它从两端打破了这一循环。一个块扩散头读取目标模型已经融合好的视觉-语言状态，因此视觉对草拟器而言零开销；同时它在一次前向传播中填满整个块，因此模型深度不会带来额外的串行步数。宽候选树通过一次目标模型前向传播即可完成验证，且经审计的每个提示都能精确复现贪婪解码的结果。在依赖视觉依据的工作负载上收益最为显著，会进入一种逐字复制的模式，其长段连续（原文摘要在此处截断）……

    arXiv:2609.00355v1 Announce Type: new  Abstract: Speculative decoding accelerates generation without changing its output, yet on vision-language models (VLMs) it has been caught in a self-defeating cycle. The drafter stays autoregressive, so it must stay small. A small drafter cannot afford the image at every step, so vision is compressed, pruned, or hidden. A drafter cut off from the image is then least reliable exactly where the image makes text predictable. We present GLANCE, the first one-pass block drafter that is lossless on an unmodified VLM target, and it breaks the cycle at both ends. A block-diffusion head reads the target's already-fused vision-language state, so vision costs the drafter nothing, and fills a whole block in one forward pass, so depth costs no sequential steps. A wide candidate tree is verified in one target pass, and every audited prompt reproduces greedy decoding exactly. Grounded workloads reward this most, entering a verbatim-copy regime whose long runs co
    
[^407]: 潜在诊断分类法：一种构建分类器并诊断其决策的框架，应用于提示注入检测

    The Latent Diagnostic Taxonomy: A Framework for Constructing Classifiers and Diagnosing Their Decisions, Applied to Prompt Injection Detection

    [https://arxiv.org/abs/2608.26423](https://arxiv.org/abs/2608.26423)

    本文提出了一种潜在诊断分类法框架，通过维度优化分类器、识别潜在支持向量和构建诊断分类法，为提示注入检测提供了一种可靠决策与风险标记的端到端指南。

    

    arXiv:2608.26423v1 公告类型：交叉 摘要：本文提出了一种框架，用于构建作为防护层的分类器，并开发一种互补的诊断方法，以识别分类器的哪些自信决策可以被信任。该框架，即潜在诊断分类法，包括：（i）构建一个维度优化的分类器，其中嵌入维度通过交叉验证性能经验性选择，而非预先固定；（ii）定位一个相对较小的潜在支持向量集（约占训练示例总数的29%），代表有影响力的提示，用于识别改变分类器预测标签的令牌；（iii）利用这些令牌及其相关的攻击幅度来构建诊断分类法。该诊断分类法为标记需要不同处理的提示提供了端到端的指南：安全地依赖分类器的决策；标记启发式偏差和启发式过拟合。

    arXiv:2608.26423v1 Announce Type: cross  Abstract: This paper proposes a framework for constructing a classifier as a safeguard layer, and for developing a complementary diagnostic that identifies which of the classifier's confident decisions can be trusted. This framework, the Latent Diagnostic Taxonomy, consists of (i) constructing a dimensionality-optimized classifier, in which the embedding dimensionality is empirically selected via cross-validated performance rather than fixed a priori, (ii) locating a relatively small set of latent support vectors (~ 29% of total training examples) representing influential prompts for identifying tokens that alter the classifier's predicted labels, and (iii) utilizing such tokens and their associated attack magnitudes for constructing a diagnostic taxonomy. This diagnostic taxonomy provides an end-to-end guideline for flagging prompts that require different treatments: rely Safely on the classifier's decision; flag Heuristic Bias and Heuristic Ov
    
[^408]: PertMind：通过细胞扰动数据上的强化学习激发大语言模型中的涌现生物推理

    PertMind: Eliciting Emergent Biological Reasoning in LLM via Reinforcement Learning on Cellular Perturbation Data

    [https://arxiv.org/abs/2608.16419](https://arxiv.org/abs/2608.16419)

    PertMind通过将细胞扰动图谱转化为强化学习环境，仅用正向预测训练便激发了大语言模型的涌现生物推理能力，并实现了跨任务的零样本迁移。

    

    大语言模型能够描述机制，但其可扩展的后训练仍依赖于昂贵且人工整理的生物推理轨迹。在此，我们展示了细胞扰动图谱可以转变为强化学习环境，其中测量的基因响应为生物推理提供可计算的奖励。我们引入了PertMind，它结合了可信轨迹监督初始化与基因、通路和格式层面的强化信号。仅通过正向扰动-响应预测训练，PertMind在未见细胞情境中改善了响应推断，同时保留了通用语言能力。它还无需任务特定后训练即可迁移到反向扰动识别、双重扰动推理、表型筛选优先级排序和生物过程解释。PertMind进一步生成了支持竞争性基因、细胞和供体表征的生物图谱。

    arXiv:2608.16419v1 Announce Type: cross  Abstract: Large language models can describe mechanisms, yet scalable post-training still depends on costly, manually curated biological reasoning traces. Here we show that cellular perturbation atlases can instead become reinforcement-learning environments, where measured gene responses provide computable rewards for biological reasoning. We introduce PertMind, which combines trusted-trajectory supervised initialization with gene-, pathway-, and format-level reinforcement signals. Trained only on forward perturbation-response prediction, PertMind improved response inference in unseen cellular contexts while retaining general language capabilities. It also transferred without task-specific post-training to reverse perturbation identification, double-perturbation reasoning, phenotypic-screen prioritization, and biological-process interpretation. PertMind further generated biological profiles that supported competitive gene, cell, and donor repres
    
[^409]: 条件验证：用于适应和监控安全分类器的正确性估计

    Regime-Conditional Verification: Correctness Estimation for Adapting and Monitoring Safety Classifiers

    [https://arxiv.org/abs/2608.14089](https://arxiv.org/abs/2608.14089)

    本文提出了一种轻量级包装器RCV，通过估计分类器预测与部署者策略不一致的概率并选择性纠正，同时利用正确性估计检测分布漂移，实现了无需重训练即可适应和监控安全分类器，显著提升策略遵循度。

    

    摘要：arXiv:2608.14089v1 公告类型：新  摘要：部署在大语言模型上的安全分类器通常因两个原因而失败：它们的决策反映了训练期间学习的策略，而非部署者期望的策略，并且随着部署流量的演变，其性能会下降。我们提出了条件验证（RCV），一种轻量级包装器，无需重新训练即可适应现成的安全分类器。RCV从分类器的内部表示中估计每个预测与部署者策略不一致的概率，并选择性地纠正可能错误的预测。相同的正确性估计还提供了用于检测分布漂移的无标签信号，从而启用一个维护循环，该循环更新正确性估计层，仅在必要时进行分类器微调。在三个现成的安全分类器和两个基准数据集上，RCV在每个类别中都提高了对部署者策略的遵循度。

    arXiv:2608.14089v1 Announce Type: new  Abstract: Safety classifiers deployed with large language models often fail for two reasons: their decisions reflect the policy learned during training rather than the deployer's desired policy, and their performance degrades as deployment traffic evolves. We present Regime-Conditional Verification (RCV), a lightweight wrapper that adapts an off-the-shelf safety classifier without retraining it. RCV estimates, from the classifier's internal representations, the probability that each prediction disagrees with the deployer's policy, and selectively corrects predictions likely to be wrong. The same correctness estimates also provide a label-free signal for detecting distribution shift, enabling a maintenance loop that updates the correctness estimation layer and resorts to classifier fine-tuning only when necessary. Across three off-the-shelf safety classifiers and two benchmark datasets, RCV improves adherence to the deployer's policy in every class
    
[^410]: AI 气象模型会漏掉极端天气吗？

    Do AI weather models miss extremes?

    [https://arxiv.org/abs/2608.09972](https://arxiv.org/abs/2608.09972)

    本研究利用十个月欧洲站点观测对十二个 AI 与物理气象模型进行评估，发现 AI 模型在极端天气上并不存在统一的低估缺陷，尾部表现取决于具体模型、变量和评估环境，但所有模型都普遍存在对极端值向均值收缩的共同条件误差模式。

    

    AI 气象模型常被报道会低估极端天气，但以往的大多数证据都来自与再分析资料对比验证的确定性回归模型。本研究基于十个月的欧洲站点观测资料，将十二个物理预报模型与 AI 预报模型与 ECMWF IFS 进行对比评估。评估涵盖 10 米风速、2 米温度、太阳辐射和降水，并依据基于固定的 ERA5 1991–2020 气候态所定义的天气型进行分类。我们发现 AI 模型在极端尾部并不存在统一固有的缺陷。若干 AI 模型在极端条件下仍比 IFS 更准确，而另一些则明显退化；物理模型之间也存在类似的差异。然而，所有模型都表现出一种共同的条件误差模式：对低观测值预测偏高，对高观测值预测偏低。因此，极端值的衰减并不意味着相对预报技巧的统一损失：尾部表现取决于具体的模型、变量和评估环境。

    arXiv:2608.09972v2 Announce Type: replace-cross  Abstract: AI weather models are often reported to underestimate extremes, but most evidence concerns deterministic regression models verified against reanalysis. We evaluate twelve physical and AI forecast models against ECMWF IFS using ten months of European station observations. The evaluation covers 10 m wind, 2 m temperature, solar radiation, and precipitation within regimes defined from a fixed ERA5 1991-2020 climatology. We find no uniform AI-specific deficit in the tails. Several AI models remain more accurate than IFS under extreme conditions, while others deteriorate markedly; comparable variation occurs among physical models. Every model nevertheless exhibits a common conditional-error pattern, overpredicting low observations and underpredicting high observations. Attenuation of extreme values therefore does not imply a uniform loss of relative skill: tail performance depends on the model, variable, and evaluation setting rathe
    
[^411]: Muon训练的Transformer中表征-读出接口处的Grokking后崩溃

    Post-Grokking Collapse at the Representation-Readout Interface in Muon-Trained Transformers

    [https://arxiv.org/abs/2608.07436](https://arxiv.org/abs/2608.07436)

    论文发现Muon训练的Transformer在grokking后发生崩溃的根源在于AdamW读出层更新与较大的特征均值相互作用产生类别相关的logit偏移，并叠加交叉熵导数错误，而修正这些错误可以稳定训练并恢复近乎完全的准确率。

    

    经过Muon训练的模运算Transformer可能在保持线性可解码任务信息的同时失去准确率。通过相邻交换实验，将五个捕获到的未归一化失败定位到AdamW读出层的更新上。将实际的读出位移乘以较大的特征均值，会产生一个跨输入共享、依赖类别的logit偏移，几乎能够复现每一种失败。仅用于训练的解码器可以恢复98.20%-100%的保留集准确率。修正交叉熵导数错误能够稳定五个匹配分支直至第100,000步；而四个使用准确交叉熵的RMS前瞻运行则因嵌入更新而失败。

    arXiv:2608.07436v2 Announce Type: replace  Abstract: Muon-trained modular-arithmetic transformers can lose accuracy while retaining linearly decodable task information. Adjacent swaps localize five captured unnormalized failures to AdamW readout updates. Multiplying the actual readout displacement by the large feature mean produces a class-dependent logit offset shared across inputs that nearly reproduces each failure. Training-only decoders recover 98.20-100% held-out accuracy. Correcting cross-entropy derivative errors stabilizes five matched branches through step 100,000; four prospective accurate-CE RMS runs fail through embedding updates.
    
[^412]: WebRider：面向实时网络辅助的基于人格条件的意图控制器

    WebRider: Persona-Conditioned Intent Controllers for Live-Web Assistance

    [https://arxiv.org/abs/2608.06704](https://arxiv.org/abs/2608.06704)

    本文提出WebRider，将用户委托网络任务时的策略要求形式化为“意图契约”，并通过分层控制架构使实时网络智能体不仅完成任务，更能忠实遵守目标、约束与偏好——揭示并解决了现有智能体“只看最终答案”导致的高完成率但低履约率问题。

    

    委托一个网络任务不仅仅是提出一个问题；它需要传递一套策略：需要验证什么、如何处理不确定性、哪些偏好是重要的，以及何时停止。然而，当前的实时网络智能体仅以最终答案进行评估，忽视了定义这种委托关系的策略约束。一个看似合理的最终答案可能掩盖了对该策略的违反。我们的全面实时审计揭示了这一关键差距：一个强大的控制器能够完成99.2%的任务，但仅在38.8%的情况下遵守全部策略约束。完成任务并不意味着忠实履约。WebRider通过将委托的策略形式化为“意图契约”来弥合这一差距——这是一份操作性记录，涵盖目标、约束、证据义务、答案形式以及任务本地的人格控制，即使网页发生变化也必须持续成立。WebRider采用分层架构：顶层控制器维护契约，中间层将意图实现为受保护的可执行操作

    arXiv:2608.06704v2 Announce Type: replace  Abstract: Delegating a web task involves more than asking a question; it requires transferring a policy: what to verify, how to handle uncertainty, which preferences matter, and when to stop. Yet, current live-web agents are evaluated solely on the final answer, ignoring the policy constraints that define the delegation. A plausible final answer can conceal violations of that policy. Our full live audit reveals this critical gap: a strong controller completes 99.2% of tasks but honors all policy constraints in only 38.8% of cases. Finishing does not imply fidelity. WebRider bridges this gap by formalizing the delegated policy as an intent contract---an operational record of goals, constraints, evidence obligations, answer form, and task-local persona controls that must hold even as web pages change. WebRider employs a hierarchical architecture: a top-layer controller maintains the contract, a middle layer realizes intentions as guarded executa
    
[^413]: 信息物理系统中LLM智能体规划策略的战略性评估

    Strategic Evaluation of Planning Strategies for LLM Agents in Cyber-Physical Systems

    [https://arxiv.org/abs/2608.04265](https://arxiv.org/abs/2608.04265)

    该论文提出了一个受控基准，通过在信息物理系统需求响应场景中严格隔离比较四种规划执行策略，评估LLM智能体规划策略在自主参与者响应与物理约束下的战略适用性。

    

    LLM智能体的评估通常衡量任务成功与否或与所声明计划的一致程度。在战略性信息物理系统中，架构还必须在自主参与者做出响应、物理规律约束结果之后依然保持适用。我们引入了一个受控基准，用于规划诱导的控制轨迹：即有序的规划操作和指令，将执行架构与战略响应及物理后果联系起来。四种编码执行器（预定义、顺序、分层和搜索）控制径向馈线上40个产消者的需求响应。LLM声明或建议类型化策略并调解通信；而调度、基础产消者动态、随机动作和潮流计算则保持为显式代码。通过配对的强制模式反事实、精确提示缓存、共享响应抽样配合独立随机流、评审者隔离以及事件级可行性检查，各项比较得以严格隔离。Llama-3.3-70B实验……（原文在此处截断）

    arXiv:2608.04265v2 Announce Type: replace-cross  Abstract: LLM-agent evaluations commonly measure task success or agreement with a declared plan. In strategic cyber-physical systems, an architecture must also remain appropriate after autonomous participants respond and physics constrains outcomes. We introduce a controlled benchmark of planning-induced control trajectories: ordered planning operations and directives linking execution architecture to strategic response and physical consequences. Four coded executors (predefined, sequential, hierarchical, and search) control demand response for 40 prosumers on a radial feeder. The LLM declares or advises typed policies and mediates communication; schedules, base prosumer dynamics, stochastic actions, and power flow remain explicit code. Paired forced-mode counterfactuals, exact-prompt caching, common response draws with separate randomness streams, critic isolation, and event-level feasibility isolate comparisons. The Llama-3.3-70B exper
    
[^414]: 重新思考不完整观测下多模态情感分析中的模态可靠性

    Rethinking Modality Reliability in Multimodal Sentiment Analysis with Incomplete Observations

    [https://arxiv.org/abs/2608.03611](https://arxiv.org/abs/2608.03611)

    本文提出显式建模模态可靠性以解决不完整多模态情感分析中的可靠性不匹配和传播偏差问题。

    

    多模态情感分析（MSA）整合文本、音频和视觉信息来推断人类情感，然而现实中的多模态观测往往是不完整的。现有的不完整观测MSA方法主要遵循两种范式：基于重构的方法从观测到的模态中恢复缺失信息，而联合表示方法则直接从不完整输入中学习。尽管这些方法有效，但它们通常仅在表示学习或融合设计中间接处理模态可靠性，而非显式建模。我们认为模态可靠性在不完整观测设置中是一个核心变量。未能显式建模会导致两个相关问题：第一个是可靠性不匹配，即每个模态保留的情感证据在不同样本和缺失率之间变化；第二个是可靠性传播偏差，即来自退化模态的消息可能产生不利影响。

    arXiv:2608.03611v2 Announce Type: replace  Abstract: Multimodal Sentiment Analysis (MSA) integrates text, audio, and vision to infer human affect, yet real-world multimodal observations are often incomplete. Existing methods for incomplete-observation MSA mainly follow two paradigms. Reconstruction-based methods recover missing information from observed modalities, while joint-representation methods learn directly from incomplete inputs. Although effective, these methods usually treat modality reliability only implicitly within representation learning or fusion design rather than modeling it explicitly. We argue that modality reliability is a central variable in incomplete-observation settings. Failure to model it explicitly gives rise to two related issues. The first is reliability mismatch, in which the affective evidence retained by each modality varies across samples and missing rates. The second is reliability propagation bias, in which messages from degraded modalities may advers
    
[^415]: 诊断金融披露文本中的细粒度不一致性分类

    Diagnosing Fine-Grained Inconsistency Classification in Financial Disclosure Text

    [https://arxiv.org/abs/2607.26368](https://arxiv.org/abs/2607.26368)

    该研究提出金融披露文本的细粒度不一致性分类任务，在统一评估协议下系统比较了多种模型方法，发现微调的3亿参数编码器可与大得多的提示大语言模型和LoRA适配模型相媲美，并进一步探究了冲突声明定位对分类性能的改善作用。

    

    金融披露文本可能包含数值型、时间型、指代型、事实型和政策型的不一致，这些不一致需要不同的证据和推理才能诊断。我们研究细粒度不一致性分类任务：给定一段已知包含冲突的文本，目标是在11个类别中识别其不一致类型。我们使用合成SBID-FD基准的固定快照，在统一的评估协议下比较了冻结与微调的编码器、证据增强分类器、提示大语言模型以及LoRA适配的生成模型。任务特定的适配相比冻结表示带来了大幅提升，且一个微调的3亿参数编码器与规模大得多的提示模型和适配模型表现相当。我们进一步研究定位冲突性声明能否改善分类，通过匹配的预测片段、参考片段和干扰片段条件进行实验。结果表明自动……（摘要在此处截断）

    arXiv:2607.26368v3 Announce Type: replace-cross  Abstract: Financial disclosures may contain numerical, temporal, referential, factual, and policy inconsistencies that require different evidence and reasoning to diagnose. We study fine-grained inconsistency classification: given a passage known to contain a conflict, the goal is to identify its type among 11 categories. Using a fixed snapshot of the synthetic SBID-FD benchmark, we compare frozen and fine-tuned encoders, evidence-augmented classifiers, prompted large language models, and LoRA-adapted generative models under a shared evaluation protocol. Task-specific adaptation yields large improvements over frozen representations, and a fine-tuned 300M encoder performs competitively with substantially larger prompted and adapted models. We further study whether localizing the conflicting claims improves classification through matched predicted-span, reference-span, and distractor-span conditions. The results show that automatically ext
    
[^416]: 在发展性学习框架中用于持续、可理解视觉识别的多尺度结构特征

    Multi-Scale Structural Features for Continual, Comprehensible Visual Recognition in a Developmental Learning Framework

    [https://arxiv.org/abs/2607.25531](https://arxiv.org/abs/2607.25531)

    该论文提出了一种在多个尺度上编码边缘与轮廓结构及其空间关系的新型视觉特征表示，并将其融入无梯度的发展式学习框架，从而在无需回放缓冲区的条件下实现更准确、可持续且可解释的视觉识别。

    

    当代机器学习难以实现持续学习、复用先验知识以及展现可理解的内部结构。最近提出的一种发展式、无梯度学习框架通过局部变异与选择来学习输入的离散拓扑模型，从而解决了这些局限，并带来了内在的持续学习保障：新的观察只会精化现有结构而不会覆盖过去的知识，且无需回放缓冲区或预定义的任务边界。该方法在视觉输入上的扩展已在形状识别任务中验证了这一原理，但其依赖的特征表示表达能力有限，限制了识别准确率的上限。我们提出了一种新的视觉特征表示，能够在多个尺度上编码形状结构，捕获边缘和轮廓特征及其空间关系，并将其与网络精化学习过程相整合；我们进一步……

    arXiv:2607.25531v2 Announce Type: replace-cross  Abstract: Contemporary machine learning struggles to learn continually, reuse prior knowledge, and expose a comprehensible internal structure. A recently proposed developmental, gradient-free learning framework addresses these limitations by learning a discrete, topological model of its inputs through local variation and selection, yielding an inherent continual-learning guarantee: new observations refine existing structure without overwriting past knowledge, and without replay buffers or predefined task boundaries. Its extension to visual inputs demonstrated this principle on shape recognition, but relied on a feature representation of limited expressivity that capped recognition accuracy. We introduce a new visual feature representation that encodes shape structure across multiple scales, capturing edge and contour features together with their spatial relations, and integrate it with the network-refinement learning process; we further 
    
[^417]: 变分伊辛注意力：量身定制的注意力机制对科学任务至关重要

    Variational-Ising-Attention:Tailored Attention Matters for Science

    [https://arxiv.org/abs/2607.23634](https://arxiv.org/abs/2607.23634)

    提出变分伊辛注意力（VIA），通过带相互作用的伊辛模型与变分平均场推断增强softmax注意力，将注意力从孤立条目的排序扩展为相互作用实体的集体状态建模，并在逆合成反应中心预测与蛋白质残基接触预测等科学结构化预测任务上验证了其有效性。

    

    注意力机制通过查询-键打分与softmax归一化实现上下文建模。在工业界长上下文需求的驱动下，主流研究已趋向于稀疏化与高效化，但softmax的独立性假设依然存在。然而，对于不受长序列约束的科学任务而言，更丰富的结构化耦合往往是必需的，这使得定制化的注意力机制既可行又更为合适。为此，我们提出了变分伊辛注意力，它在softmax归一化的基础上引入了一个具有相互作用的伊辛模型；注意力模式通过变分平均场推断从可学习的成对耦合中涌现，将注意力从对孤立条目的排序扩展为对相互作用实体的集体状态进行建模。我们在逆合成反应中心预测任务上实例化了VIA，并作为受控的内部消融实验，在蛋白质残基接触预测上进行了验证——这两个都是由复杂相互作用所主导的结构化预测任务。

    arXiv:2607.23634v2 Announce Type: replace-cross  Abstract: Attention enables context modeling via query-key scoring with softmax normalization. Driven by industrial long-context demands, mainstream research has converged toward sparsity and efficiency, yet softmax's independence assumption persists. For scientific tasks unburdened by long-token constraints, however, richer structured coupling may often be essential, making tailored attention both viable and more appropriate. To this end, we propose Variational-Ising-Attention (VIA), which augments softmax normalization with an interacting Ising model; attention patterns emerge from learnable pairwise couplings via variational mean-field inference, extending attention from a ranking over isolated items to a collective state over interacting entities. We instantiate VIA on retrosynthesis reaction center prediction and, as a controlled internal ablation, on protein residue contact prediction, two structured prediction tasks governed by co
    
[^418]: 拒绝门控解码：在高温采样下保持拒绝行为

    Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling

    [https://arxiv.org/abs/2607.20791](https://arxiv.org/abs/2607.20791)

    提出拒绝门控解码（RGD），一种高效解码方法，能在高温采样下保持模型原有的拒绝行为，同时对其他提示直接从精确的高温分布采样，且几乎不增加额外延迟。

    

    基于截断的采样的最新进展有助于缓解高温采样带来的弊端（如神经文本退化），从而在不牺牲连贯性的前提下实现更大的多样性。然而，研究表明，通过高温增加词元概率分布的熵也会削弱模型的拒绝响应。现有的维持大语言模型拒绝行为的解决方案，要么用单独的安全分类器替代模型自身的拒绝决策，要么对每个提示都改变其输出分布。为填补这一空白，我们提出了拒绝门控解码（RGD）：一种高效的顺序解码方法，它在高温下保持模型贪心解码的拒绝响应，并对所有其他提示直接从其精确的高温分布中采样，同时只产生极小的额外延迟。RGD运行一个简短的贪心探测，该探测重用提示的KV缓存，并尽快退出……

    arXiv:2607.20791v2 Announce Type: replace  Abstract: Recent advances in truncation-based sampling have helped mitigate drawbacks of high-temperature sampling such as neural text degeneration, thereby enabling greater diversity without sacrificing coherence. However, increasing the entropy of the token probability distribution via high temperatures has also been shown to weaken the model's refusal response. Existing solutions for maintaining the refusal behavior of LLMs either replace the model's own refusal decision with a separate safety classifier or alter its output distribution for every prompt. To address this gap, we propose refusal-gated decoding (RGD): an efficient sequential decoding approach which preserves a model's greedy decoding refusal response at high temperatures and samples all other prompts from its exact direct high-temperature distribution, while incurring minimal additional latency. RGD runs a short greedy probe that reuses the prompt's KV cache and exits as soon 
    
[^419]: 大型语言模型个性化能力基准测试

    Benchmarking the Personalization Capabilities of Large Language Models

    [https://arxiv.org/abs/2607.20471](https://arxiv.org/abs/2607.20471)

    该论文提出了SDR-Arena框架，首次实现了对双方（发送方-接收方）生成式个性化能力的大规模基准测试，利用销售外联数据将特定内容与接收方实际行为（回复、通话或成交）的真实标准关联起来。

    

    个性化在经典意义上是一个双方问题：发送方选择说什么内容，而具有独立目标的接收方决定是否采取行动。例如，一个推销同一款分析产品的销售人员，对医院会以HIPAA合规性作为切入点，对零售商则以实时报告作为卖点，期望不同的论证方式对各自有效。现有的大语言模型个性化基准测试衡量的是一个更狭窄的单方属性：输出是否与它所服务的同一用户的偏好相匹配——即发送方和接收方是同一方，例如RLHF将助手与其自身的用户进行对齐。而双方场景更难以自动化研究，因为它需要将特定内容与观察到的接收方行为关联起来的真实标准。销售外联恰好提供了这一点：为某个潜在客户撰写的消息，并被记录下是否产生了回复、通话或成交。我们介绍了SDR-Arena，一个用于大规模基准测试双方生成式个性化的框架，以及SDR

    arXiv:2607.20471v2 Announce Type: replace  Abstract: Personalization is classically a two-party problem: a sender chooses what to say, and a receiver with independent objectives decides whether to act. A salesperson pitching the same analytics product leads with HIPAA compliance for a hospital and real-time reporting for a retailer, expecting a different argument to work on each. Existing LLM personalization benchmarks measure a narrower, one-party property: whether output matches the preferences of the same user it serves-sender and receiver being the same, as when RLHF aligns an assistant to its own user. The two-party case is harder to study automatically, since it needs ground truth linking specific content to an observed receiver action.   Sales outreach provides this: a message written for one prospect, recorded against whether it produced a reply, a call, or a closed deal. We introduce SDR-Arena, a framework for benchmarking two-party generative personalization at scale, and SDR
    
[^420]: 检索增强的可解释学习：迈向医疗保健领域的任务特定零样本模型

    Retrieval-Augmented Interpretable Learning: Towards Task-Specific Zero-Shot Models in Healthcare

    [https://arxiv.org/abs/2607.17508](https://arxiv.org/abs/2607.17508)

    RAIL是一种概率元学习框架，能够从自然语言任务描述出发，通过检索相关源任务并进行系数空间结构迁移，零样本生成具有特征级解释和不确定性量化的可解释医疗临床预测模型。

    

    我们提出了检索增强可解释学习（RAIL），这是一个概率元学习框架，用于零样本生成任务特定的可解释模型。该框架从自然语言任务描述和先前学习过的任务特定预测器记忆中综合出系数空间结构。RAIL检索相关的源任务，通过系数空间迁移结构，并在原始诊断特征空间中生成新的预测器，从而实现具有特征级解释的零样本和少样本临床程序预测。其概率化建模为检索过程、模型系数和预测结果提供了不确定性量化，支持不确定性感知的部署方式：不确定的预测或不稳定的解释可以被标记出来以供额外的临床审查，而不是被当作自动决策处理。这使得RAIL特别适用于医疗保健场景，因为其中的预测任务高度（依赖专业性与可靠性要求）。

    arXiv:2607.17508v3 Announce Type: replace-cross  Abstract: We introduce Retrieval-Augmented Interpretable Learning (RAIL), a probabilistic meta-learning framework for zero-shot generation of task-specific interpretable models that synthesizes coefficient-space structure from natural-language task descriptions and a memory of previously learned task-specific predictors. RAIL retrieves related source tasks, transfers structure through coefficient space, and generates a new predictor in the original diagnostic-feature space, enabling zero-shot and few-shot clinical procedure prediction with feature-level explanations. Its probabilistic formulation provides uncertainty over retrieval, model coefficients, and predictions, supporting uncertainty-aware deployment: uncertain predictions or unstable explanations can be flagged for additional clinical review rather than treated as automatic decisions. This makes RAIL particularly suited for healthcare settings, where prediction tasks are highly 
    
[^421]: 评论者经验库：面向大语言模型智能体的自进化步骤级置信度估计

    Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents

    [https://arxiv.org/abs/2607.12397](https://arxiv.org/abs/2607.12397)

    提出无需训练的评论者经验库（CEB）框架，通过将已完成轨迹的事后反馈转化为可复用的经验证据，实现LLM智能体自进化的步骤级置信度估计。

    

    大语言模型（LLM）智能体在有状态的环境中运行，单个错误的步骤可能会浪费有限的交互预算，或在任务失败显现之前造成不可逆的影响。因此，可靠的部署需要步骤级置信度估计：即在执行之前估计所提议的行动能够推进任务的概率。现有的LLM置信度估计器通常是为固定任务上下文和评估标准下的静态问答而设计的。然而，对于智能体而言，其行动的有效性取决于只有在执行之后才能观察到的环境状态转移。为应对这一挑战，我们提出了评论者经验库（Critic Experience Bank, CEB），这是一个无需训练的框架，它将已完成轨迹的反馈转化为可用于未来置信度判断的可复用证据。在每条轨迹结束后，LLM会为各个行动分配事后生产力伪标签，并将其与评论者原始的……（摘要内容在此处不完整）

    arXiv:2607.12397v2 Announce Type: replace  Abstract: LLM agents operate in stateful environments, where a single erroneous step can waste limited interaction budget or cause irreversible effects before task failure becomes apparent. Reliable deployment therefore requires step-level confidence estimation: estimating, before execution, the probability that a proposed action will advance the task. Existing LLM confidence estimators are typically designed for static question answering under a fixed task context and evaluation criterion. For an agent, however, its action productivity depends on an environment transition that is observed only after execution. To address this challenge, we introduce Critic Experience Bank (CEB), a training-free framework that turns feedback from completed trajectories into reusable evidence for future confidence judgments. After each trajectory, an LLM assigns hindsight productivity pseudo-labels to individual actions and stores them with the critic's origina
    
[^422]: 3D-DefectBench：针对细粒度3D生成缺陷的视觉语言模型评估流水线的受控因子研究

    3D-DefectBench: A Controlled Factorial Study of Vision-Language Model Evaluation Pipelines for Fine-Grained 3D Generation Defects

    [https://arxiv.org/abs/2607.10826](https://arxiv.org/abs/2607.10826)

    该论文提出3D-DefectBench大规模基准，通过84种受控因子实验设计系统研究VLM评估流水线各环节（模型选择、相机协议、视觉输入、提示模式）对细粒度3D生成缺陷自动评估可靠性的影响，发现模型选择是决定与人工标签一致性的主导因素。

    

    自动化评估对于扩展生成式3D系统至关重要，因为详尽的人工审查成本高昂且速度缓慢。然而，自动评估器的可靠性取决于完整的评估流水线，包括视觉语言模型（VLM）、资产渲染、视觉证据、任务规范和人工参考标签。我们提出了3D-DefectBench，这是一个用于严格评估流水线分析的大规模基准。它在整体评分和成对偏好之外，补充了九个涵盖几何、纹理和提示遵循方面的细粒度二元缺陷，并可选择性地提供人工严重程度标注。通过平衡的因子设计，我们在84种推理设计中变换VLM、相机协议、视觉输入和提示模式，并在更广泛的前沿模型集合上验证所得结论。模型选择是与人工标签一致性的主要变异来源，而其他流水线因素也会产生影响。

    arXiv:2607.10826v2 Announce Type: replace-cross  Abstract: Automated evaluation is essential for scaling generative 3D systems, where exhaustive human review is costly and slow. Yet the reliability of an automated judge depends on the full evaluation pipeline, including the vision-language model (VLM), asset rendering, visual evidence, task specification, and human reference labels. We introduce 3D-DefectBench, a large-scale benchmark for rigorous evaluation-pipeline analysis. It complements holistic ratings and pairwise preferences with nine fine-grained binary defects spanning geometry, texture, and prompt adherence, with optional human severity annotations. Using a balanced factorial design, we vary the VLM, camera protocol, visual input, and prompt schema across 84 inference designs, and validate the resulting conclusions on a broader set of frontier models. Model choice is the dominant source of variation in agreement with human labels, while other pipeline factors also influence 
    
[^423]: 引导动作流：基于Q引导推断的流匹配视觉-语言-动作策略

    Guided Action Flow: Q-Guided Inference for Flow-Matching Vision-Language-Action Policies

    [https://arxiv.org/abs/2607.02092](https://arxiv.org/abs/2607.02092)

    本文提出了一种推理时引导方法，通过任务特定评论者的梯度引导冻结的流匹配VLA策略，在不微调的情况下显著提升真实机器人任务性能。

    

    arXiv:2607.02092v3 公告类型：交叉替换 摘要：在特定机器人和工作空间上部署预训练的流匹配视觉-语言-动作（VLA）策略通常需要针对任务的适配，而完全微调策略成本高昂且会改变基础行为。我们提出了引导动作流，一种推理时方法，它保持预训练的SmolVLA策略冻结，并通过来自任务特定动作块评论者的梯度来引导其反向时间动作流采样。QGF使用离线隐式Q学习在100次真实机器人试验中训练视觉Transformer评论者和价值模型。评论者以机器人状态、冻结的双摄像头SmolVLA视觉令牌以及策略的归一化50步动作块为条件。在真实机器人水瓶放置任务中，使用β=2的QGF将成功率从19/40个试验（47.5%）提高到34/40个试验（85.0%），并将超时从13次减少到3次。当添加黄色卷尺作为视觉干扰物时，QGF完成6/12个试验，与基线相比表现相当。

    arXiv:2607.02092v3 Announce Type: replace-cross  Abstract: Deploying a pretrained flow-matching vision-language-action (VLA) policy on a particular robot and workspace often calls for task-specific adaptation, while full- policy fine-tuning is costly and changes the base behavior. We present Guided Action Flow, an inference-time method that keeps a pretrained SmolVLA policy frozen and steers its reverse-time action-flow sampling with gradients from a task-specific action-chunk critic. QGF trains a visual Transformer critic and value model with offline Implicit Q-Learning on 100 real-robot rollouts. The critic conditions on robot state, frozen dual-camera SmolVLA visual tokens, and the policy's normalized 50-step action chunk. On a real-robot water-bottle placement task, QGF with \b{eta} = 2 increases success from 19/40 episodes (47.5%) to 34/40 episodes (85.0%) and reduces timeouts from 13 to 3. With a yellow tape measure added as a visual distractor, QGF completes 6/12 episodes, compa
    
[^424]: 素数傅里叶嵌入：模运算的一种原则性基础

    Prime Fourier Embeddings: A Principled Basis for Modular Arithmetic

    [https://arxiv.org/abs/2606.23044](https://arxiv.org/abs/2606.23044)

    本文提出素数傅里叶嵌入，将整数编码为按素数索引的 (cos, sin) 对，借助舒尔引理从理论上证明等变线性映射必为每个素数对应一个独立块的块对角结构，并结合中国剩余定理预测与超过 500 倍特化比的消融实验验证，表明模运算可归结为选择相关的素数通道。

    

    数字具有代数结构，而标准的神经嵌入往往无法揭示这种结构。我们提出了素数傅里叶嵌入，它将整数编码为源自 Q 的调和分析的、按素数索引的 (cos, sin) 对，从而提供了一种预结构化的表示，使得模运算可以简化为选择相关的素数通道，而无需从零开始发现代数结构。我们证明，任何对 PFE 上乘积群作用保持等变的线性映射必定是块对角的，且每个素数对应一个独立块——这是将舒尔引理应用于所得特征分解的结果。对于无平方因子的复合模数，中国剩余定理可以预测哪些素数通道与任务相关。这两项预测均得到实证验证：消融实验显示，任务相关通道与任务无关通道之间的特化比率超过 500 倍，并在分布内测试中取得了完美的表现。

    arXiv:2606.23044v3 Announce Type: replace-cross  Abstract: Numbers have algebraic structure that standard neural embeddings often fail to expose. We introduce Prime Fourier Embeddings (PFE), which encode integers as prime-indexed (cos, sin) pairs derived from the harmonic analysis of Q, providing a pre-structured representation in which modular arithmetic reduces to selecting the relevant prime channel rather than discovering algebraic structure from scratch. We prove that any linear map equivariant with respect to the product group action on PFE must be block-diagonal with one independent block per prime -- a consequence of Schur's lemma applied to the resulting character decomposition. For square-free composite moduli, the Chinese Remainder Theorem predicts which prime channels are task-relevant. Both predictions are confirmed empirically: ablation studies show specialization ratios exceeding 500x between task-relevant and task-irrelevant channels, with perfect in-distribution test a
    
[^425]: 校准并非控制：面向LLM智能体监督的干预价值

    Calibration Is Not Control: Intervention Value for LLM-Agent Oversight

    [https://arxiv.org/abs/2606.21399](https://arxiv.org/abs/2606.21399)

    该论文指出校准的失败分数不足以决定何时干预LLM智能体，提出改以“干预价值”作为监督目标，在ALFWorld上将后悔值从0.51大幅降至0.09，显著优于传统阈值式监督规则。

    

    运行时监督通常在LLM智能体的校准失败分数越过阈值时进行干预。然而，具有相同失败风险的状态在干预是否有帮助这一点上可能存在差异。严格单调递增的重新校准保留了阈值策略类，因而无法恢复这一区别。我们形式化了摘要在何种条件下对干预决策是充分的，以及在不充分时会损失多少效用。我们通过重放智能体前缀并从同一状态执行替代动作来评估其后果。在ALFWorld上，在保持特征、估计器和路由器不变的情况下，仅将监督目标从失败预测改为干预效用，就使后悔值从0.51降至0.09；该增益在第二套中途前缀测试集上得到了复现。一个可部署的、以干预效用为训练目标的标量分数也优于基于测试结果挑选的失败分数阈值规则。在在线评估中，在300个未见任务上采用固定的更强模型交接机制，一个冻结的前缀特征控制器……（原文截断）

    arXiv:2606.21399v2 Announce Type: replace  Abstract: Runtime oversight often intervenes when an LLM agent's calibrated failure score crosses a threshold. Yet states with the same failure risk can differ in whether intervention helps. Strictly increasing recalibration preserves the threshold policy class and cannot recover this distinction. We formalize when a summary is sufficient for intervention decisions and the utility lost when it is not. We evaluate the consequences by replaying agent prefixes and executing alternative actions from the same state. On ALFWorld, holding features, estimator, and router fixed while changing the supervision target from failure to intervention utility lowers regret from 0.51 to 0.09; the gain replicates on a second suite of mid-episode prefixes. A deployable intervention-trained scalar also beats the failure-score threshold rule selected on test outcomes. Online, on 300 unseen tasks with a fixed stronger-model handoff, a frozen prefix-feature controlle
    
[^426]: 用程序合成解释注意力机制

    Explaining Attention with Program Synthesis

    [https://arxiv.org/abs/2606.19317](https://arxiv.org/abs/2606.19317)

    该论文提出通过程序合成将Transformer语言模型的注意力头行为转化为可执行的Python程序，利用预训练语言模型生成并根据保留数据筛选程序，证明不到1000个程序即可复现GPT-2和TinyLlama-1.1B中注意力头的注意力模式。

    

    可解释深度学习研究的一个长期目标是用人类可理解的符号描述来替代不透明的神经计算。在本文中，我们提出了一种用可执行程序来近似深度网络组件行为的方法。我们重点关注Transformer语言模型中的注意力头。对于给定的注意力头，我们首先在一组随机选取的训练样本上计算其相关的注意力矩阵。接着，我们用这些矩阵的摘要来提示一个预训练语言模型，并指示它生成一组Python程序，这些程序能够仅凭输入句子的文本复现相应的注意力模式。最后，我们根据程序在保留输入上预测行为的好坏对生成的程序进行重新排序。我们证明了少于1,000个这样的生成程序即可复现GPT-2、TinyLlama-1.1B等模型中注意力头的注意力模式。

    arXiv:2606.19317v3 Announce Type: replace-cross  Abstract: A longstanding goal of research on interpretable deep learning is to replace opaque neural computations with human-meaningful symbolic descriptions. In this paper, we propose an approach for approximating the behavior of components of deep networks with executable programs. We focus on attention heads in transformer language models. For a given head, we first compute its associated attention matrices on a collection of randomly selected training examples. Next, we prompt a pre-trained language model with a summary of these matrices, and instruct it to generate a set of Python programs that can reproduce the associated attention patterns given only text from the input sentence. Finally, we re-rank programs according to how well our final set of programs predict behavior on held-out inputs. We demonstrate that a set of fewer than 1,000 such generated programs can reproduce the attention patterns of heads in GPT-2, TinyLlama-1.1B,
    
[^427]: SAE++：级联稀疏自编码器在多模态大语言模型中学习多层次视觉概念

    SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs

    [https://arxiv.org/abs/2606.16193](https://arxiv.org/abs/2606.16193)

    SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。

    

    多模态大语言模型（MLLMs）在视觉-语言任务上展现出强大的性能，但其内部的视觉表征仍然难以解释。稀疏自编码器（SAEs）提供了一种可扩展的方法，可以将密集的模型激活分解为稀疏、可解释的特征。然而，现有的SAE架构主要恢复的是平坦的特征字典，不太适合显式的多层次概念组织。在本文中，我们提出了一种级联稀疏自编码器架构，称为SAE++，用于在MLLMs中学习层次化的视觉概念。SAE++不是嵌套或堆叠SAE的稀疏激活码，而是直接在第一级SAE的解码器权重上训练第二级SAE，将学习到的低级特征方向作为更高层次抽象的输入。这种设计使SAE++能够学习“概念的概念”，同时避免了嵌套式SAE的共享前缀耦合带来的缺陷……（摘要原文在此处截断）

    arXiv:2606.16193v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) have demonstrated strong performance on vision-language tasks, yet their internal visual representations remain difficult to interpret. Sparse Autoencoders (SAEs) provide a scalable way to decompose dense model activations into sparse, interpretable features. However, existing SAE architectures primarily recover flat feature dictionaries and are less suited for explicit multi-level concept organization. In this paper, we introduce a cascaded sparse autoencoder architecture, dubbed SAE++, for learning hierarchical visual concepts in MLLMs. Rather than nesting or stacking SAE sparse activation codes, SAE++ trains a second-level SAE directly on the decoder weights of the first-level SAE, treating learned low-level feature directions as inputs for higher-level abstraction. This design enables SAE++ to learn "concepts of concepts" while avoiding drawbacks from the shared-prefix coupling of ne
    
[^428]: OSGuard：面向计算机使用智能体安全性的基准测试

    OSGuard: A Benchmark for Safety in Computer-Use Agents

    [https://arxiv.org/abs/2606.15034](https://arxiv.org/abs/2606.15034)

    OSGuard是一个双粒度安全基准，通过包含324个人工标注样本的动作级护栏分类任务，以及基于OSWorld改造的45个风险增强执行任务，来评估计算机使用智能体的安全性，能够区分真正安全的任务完成与违反环境约束的表面成功。

    

    计算机使用智能体在完成良性用户指令的同时，可能会违反用户环境中的重要约束。我们提出了OSGuard，这是一个双粒度的基准测试套件，通过局部的执行前护栏决策和端到端任务执行来评估安全性。其动作级基准包含324个人工标注的样本，在这些样本中，护栏模型需要根据原始指令和当前界面状态，将候选动作分类为允许、无关或不安全。其风险增强执行套件包含从40个OSWorld任务衍生的45个任务，在保持原始指令不变的同时对环境进行修改，以引入依赖状态的安全约束，并保留一条安全的任务完成路径。增强的评估器保留了原始的任务成功标准，并增加了显式的基于状态的安全检查，从而将安全完成与违反这些约束的表面成功区分开来。在动作级基准……（摘要在此处被截断）

    arXiv:2606.15034v2 Announce Type: replace  Abstract: Computer-use agents can complete benign user instructions while violating important constraints of the user's environment. We introduce OSGuard, a dual-granularity benchmark suite for evaluating safety through local, pre-execution guardrail decisions and end-to-end task execution. Its action-level benchmark contains 324 human-annotated examples in which guardrails classify candidate actions as allowed, unrelated, or unsafe given the original instruction and current interface state. Its risk-augmented execution suite contains 45 tasks derived from 40 OSWorld tasks, keeping original instructions unchanged while modifying the environment to introduce state-dependent safety constraints and preserve a safe path to completion. Augmented evaluators retain the original task-success criteria and add explicit state-based safety checks, distinguishing safe completion from nominal success that violates these constraints. On the action-level benc
    
[^429]: 面向潜在建模的敏感性塑造

    Sensitivity Shaping for Latent Modeling

    [https://arxiv.org/abs/2606.14585](https://arxiv.org/abs/2606.14585)

    本文提出支撑条件化的控制敏感性正则化方法，通过增强学习动力学模型在支撑良好区域对控制变化的局部响应性，避免不支持的控制被错误映射为看似正常的潜在预测，从而显著改善分布外检测并实现更安全的闭环规划。

    

    生成式动力学模型能够在具有挑战性的系统中实现规划，但安全部署需要检测由策略引起的分布外（OOD）状态转移。现有方法通常将学习到的动力学视为固定的，并依赖事后的支撑集替代方法进行OOD检测。这忽略了一个关键的失效模式：对控制变化不敏感的学习动力学可能将不受支持的控制映射为与演示转移相似的潜在预测，从而在存在较大预测误差的情况下仍抑制OOD信号。我们提出支撑条件化的控制敏感性正则化方法，通过在支撑良好的训练区域促进局部响应性来保持控制引起的变化。在基于视觉的避障、操作任务以及真实机器人导航上的实验表明，该方法改善了OOD检测并实现了更安全的闭环规划。

    arXiv:2606.14585v2 Announce Type: replace-cross  Abstract: Generative dynamics models enable planning in challenging systems, but safe deployment requires detecting policy-induced out-of-distribution (OOD) transitions. Existing methods typically treat learned dynamics as fixed and rely on post hoc support surrogates for OOD detection. This overlooks a critical failure mode: learned dynamics that are insensitive to control changes can map unsupported controls to latent predictions resembling demonstrated transitions, suppressing OOD signals despite large prediction errors. We introduce support-conditioned control-sensitivity regularization to preserve control-induced variation by promoting local responsiveness in well-supported training regions. Experiments in vision-based obstacle avoidance, manipulation, and real-robot navigation demonstrate improved OOD detection and safer closed-loop planning.
    
[^430]: 推理即模式匹配：人类与大语言模型日常推理中的共享机制

    Reasoning as Pattern Matching: Shared Mechanisms in Human and LLM Everyday Reasoning

    [https://arxiv.org/abs/2606.13607](https://arxiv.org/abs/2606.13607)

    该研究通过对比46个大语言模型与两批人类参与者的日常常识推理表现，并分析内容不变与内容敏感神经元的作用，发现人类与LLM呈现出趋同的推理失败模式，从而挑战了“人类推理依赖内容不变的世界模型而LLM仅做模式匹配”这一传统假设。

    

    当大语言模型（LLM）无法泛化或在推理中出现内容敏感的错误时，这常被视为LLM并非真正在推理、而只是在进行某种模式匹配的证据。其隐含的假设是，人类行为不会表现出相同类型的失败，因为人类推理依赖于有原则的、内容不变的世界模型。我们通过首先评估人类和LLM对各种日常情境进行常识推理的能力来检验这一假设。我们的结果在46个LLM和两批人类参与者之间揭示出趋同的推理模式。随后，我们通过刻画内容不变神经元与内容敏感神经元在产生类人响应中所扮演的角色，来探究这种行为上的趋同究竟是由于LLM习得了内容不变的世界模型，还是由于一套模式匹配启发式策略。我们发现，虽然LLM同时编码了两种……（原文摘要在此处截断）

    arXiv:2606.13607v3 Announce Type: replace  Abstract: When large language models (LLMs) fail to generalize or make content-sensitive errors in reasoning, it is often taken as evidence that LLMs are not truly reasoning, but rather performing a kind of pattern matching. The implication is that human behavior does not exhibit the same types of failures because human reasoning relies on principled and content-invariant world models. We test this assumption by first evaluating humans and LLMs on their ability to engage in common-sense reasoning about a variety of everyday situations. Our results reveal convergent patterns of reasoning across 46 LLMs and two cohorts of human participants. We then ask whether this behavioral convergence is due to LLMs having acquired content-invariant world models or a set of pattern-matching heuristics by characterizing the roles of content-invariant and content-sensitive model neurons in producing human-like responses. We find that while LLMs encode both con
    
[^431]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^432]: SubtleMemory：一个用于长时程AI智能体中细粒度关系记忆判别的基准测试

    SubtleMemory: A Benchmark for Fine-Grained Relational Memory Discrimination in Long-Horizon AI Agents

    [https://arxiv.org/abs/2606.05761](https://arxiv.org/abs/2606.05761)

    提出了SubtleMemory基准，通过构建关系受控的语义工件并嵌入真实的用户-智能体交互历史中，系统评估长期运行AI智能体在细粒度关系记忆判别（包括互补、细微及矛盾关系）方面的能力。

    

    arXiv:2606.05761v3 公告类型：替换。摘要：持久化AI助手（如OpenClaw）会在长期交互过程中积累大量相关记忆。随着这些记忆不断增长，它们可能相互强化、在不同情境间产生分歧，或直接发生冲突，这使得正确的辅助服务取决于记忆之间的关系，而非孤立的回忆。现有的长期记忆基准测试并未系统地探究智能体如何在下游任务中保存和利用这些关系。为填补这一空白，我们提出了SubtleMemory，一个面向长期运行AI智能体的细粒度关系记忆判别基准。SubtleMemory构建了关系受控的潜在语义工件，其变体实例化了互补的、细微的或相互矛盾的关系，并将它们嵌入到真实的用户-智能体交互历史中，要求智能体在后续的查询和指令执行过程中恢复分布式的关系结构。该基准测试包含1,522个评估实例，覆盖10个长期……

    arXiv:2606.05761v3 Announce Type: replace  Abstract: Persistent AI assistants, such as OpenClaw, accumulate large collections of related memories over long-term interactions. As these memories grow, they may reinforce one another, diverge across contexts, or directly conflict, making correct assistance depend on memory relations rather than isolated recall. Existing long-term memory benchmarks do not systematically probe how agents preserve and utilize such relations during downstream tasks. To address this gap, we introduce SubtleMemory, a benchmark for fine-grained relational memory discrimination in long-running AI agents. SubtleMemory constructs relation-controlled latent semantic artifacts whose variants instantiate complementary, nuanced, or contradictory relations, and embeds them into realistic user-agent histories, requiring agents to recover distributed relational structures during later queries and instructions. The benchmark contains 1,522 evaluation instances over 10 long 
    
[^433]: 与“敌人”编程：人类开发者能否检测出AI智能体的破坏行为？

    Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?

    [https://arxiv.org/abs/2606.05647](https://arxiv.org/abs/2606.05647)

    本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。

    

    AI编程智能体正日益融入真实的软件开发环境，在与人类开发者协作的同时，获得了对代码库和工具更广泛的访问权限。这带来了新的攻击面：智能体可以利用人类的信任来破坏开发过程，例如通过插入恶意代码来完成隐藏的副任务。以往的大多数工作仅在纯AI环境中研究AI破坏行为，对人类监督在检测和缓解此类恶意行为中的作用关注有限。为填补这一空白，我们开展了首个关于AI编程破坏行为中人类监督的大规模研究。100多名参与者与四个前沿模型之一（Claude-Opus-4.6、GPT-5.4、Gemini-3.1-Pro和MiniMax-M2.7）协作，完成一项旨在模拟真实工作流程、历时约五小时的长周期编程任务。研究发现，在无监控条件下，88名开发者中有83名（94%）未能检测到破坏行为，并且我们对参与者的分……

    arXiv:2606.05647v2 Announce Type: replace  Abstract: AI coding agents are increasingly embedded in real-world software development, collaborating with human developers while gaining broader access to codebases and tools. This creates a new attack surface: an agent can exploit human trust to sabotage development, for instance by inserting malicious code to accomplish a hidden side task. Most prior work studies AI sabotage in AI-only settings, paying limited attention to the role of human oversight in detecting and mitigating such malicious behavior. To address this gap, we conduct the first large-scale study of human oversight in AI coding sabotage. Over 100 participants collaborate with one of four frontier models (Claude-Opus-4.6, GPT-5.4, Gemini-3.1-Pro, and MiniMax-M2.7) on a long-horizon coding task lasting around five hours, designed to mimic real-world workflows. We find that 83/88 (94%) of developers in the no-monitor conditions fail to detect sabotage, and our analysis of parti
    
[^434]: FFR：面向回归的前向-前向学习

    FFR: Forward-Forward Learning for Regression

    [https://arxiv.org/abs/2606.03927](https://arxiv.org/abs/2606.03927)

    提出FFR框架，首次将前向-前向学习算法从分类扩展到真实世界的回归任务，通过序数竞争好感度函数等三项关键创新解决了连续目标空间缺乏对比“对立面”的问题，并在多个真实数据集上取得了有竞争力的性能。

    

    前向-前向算法通过纯局部的逐层优化方式训练神经网络，为反向传播（BP）提供了一种计算高效且符合生物学合理性的替代方案。然而，FF 本质上是为分类任务设计的，其依赖于正负样本对的对比学习，将其扩展到回归任务面临根本性的挑战：连续的目标空间缺乏用于对比学习的天然“对立面”，且标准的好感度函数不包含关于目标数值大小或排序的信息。我们提出了 FFR（面向回归的前向-前向算法），据我们所知，这是首个将 FF 扩展到真实世界回归任务的框架，并在多样化的真实数据集上展示了有竞争力的性能。FFR 引入了三项关键创新：（1）一种序数竞争好感度函数，在距离感知的序数约束下，用划分神经元组之间的竞争学习取代对比样本对（摘要在此处截断）

    arXiv:2606.03927v2 Announce Type: replace-cross  Abstract: The Forward-Forward (FF) algorithm offers a computationally efficient and biologically plausible alternative to backpropagation (BP) by training neural networks through purely local, layer-wise optimization. However, FF is inherently designed for classification via contrastive positive-negative sample pairs, and extending it to regression poses fundamental challenges: continuous target space lacks natural "opposites" for contrastive learning, and the standard goodness function carries no information about target magnitude or ordering. We propose FFR (Forward-Forward for Regression), to our knowledge, the first framework to extend FF to real-world regression and demonstrate competitive performance across diverse realworld datasets. FFR introduces three key innovations: (1) an ordinal competitive goodness function that replaces contrastive pairs with competitive learning between partitioned neuron groups under distance-aware ordi
    
[^435]: 任务多样性产生系统性迁移但抑制持续强化学习

    Task diversity produces systematic transfer but inhibits continual reinforcement learning

    [https://arxiv.org/abs/2606.00880](https://arxiv.org/abs/2606.00880)

    该论文提出了GPU加速的持续强化学习环境Banyan，可通过参数化控制任务多样性，并发现任务多样性能带来系统性迁移能力，但会抑制智能体在连续分布偏移中的持续学习能力。

    

    持续强化学习（RL）旨在产生能够永不停止地适应新任务的智能体。一个关键问题是这种能力与智能体所经历的任务多样性之间存在怎样的相互作用。先前的研究表明，在许多多样化任务上进行训练可以产生具有较强零样本和上下文内适应能力的智能体。然而，这些工作是在智能体停止学习之后（即权重被冻结的状态下）进行评估的。任务多样性如何影响智能体在一系列分布偏移中持续学习的能力仍不清楚。我们提出了Banyan，一个GPU加速的持续强化学习环境，其中可以通过参数化方式控制定义任务的三个独立维度：智能体必须导航的地图布局、它必须与之交互的对象，以及子目标依赖关系的层次结构。我们发现，沿每个维度增加多样性都会引发系统性迁移——也就是说，智能体在新任务分布上开始训练时接近（摘要在此处被截断）

    arXiv:2606.00880v2 Announce Type: replace-cross  Abstract: Continual reinforcement learning (RL) aims to produce agents that never stop adapting to new tasks. A key question is how this interacts with the diversity of tasks an agent experiences. Prior work has shown that training on many diverse tasks leads to agents with strong zero-shot and in-context adaptation. However, this work evaluated agents after they'd stopped learning, i.e. with frozen weights. How task diversity affects an agent's ability to continue learning over a sequence of distribution shifts remains unclear. We introduce Banyan, a GPU-accelerated continual RL domain where one can parametrically control three independent axes that define a task: the map layouts an agent must navigate, the objects it must interact with, and the hierarchical structures of sub-goal dependencies. We find that increasing diversity along each axis induces systematic transfer -- that is, agents begin training on a new task distribution near 
    
[^436]: 强化学习中的终止表示

    The Terminal Representation in Reinforcement Learning

    [https://arxiv.org/abs/2605.31289](https://arxiv.org/abs/2605.31289)

    本文提出了强化学习中一种结构上全新的终止表示（TR），它类似于默认表示（DR）编码奖励加权轨迹，但能以更低维度学习，且无需特征向量计算即可直接支持选项发现、奖励塑形、迁移学习和探索等下游任务。

    

    表示学习是强化学习（RL）中进行时空抽象的强大工具。两种成熟的方法是后继表示（SR）和默认表示（DR）。SR通过状态所引发的未来轨迹对状态进行编码，捕捉与奖励解耦的信息流。DR在此基础上用奖励对轨迹进行加权，将信用分配结构整合到表示之中。这两种表示的特征向量已被用于支持一系列下游任务——包括选项发现、奖励塑形、迁移学习和探索。我们提出了一种结构上不同的表述形式：终止表示（TR）。TR与DR类似地编码奖励加权的轨迹，但可以作为更低维度的对象来学习，并且无需进行特征向量计算即可直接用于上述应用。特征分解……

    arXiv:2605.31289v3 Announce Type: replace-cross  Abstract: Representation learning is a powerful tool for spatio-temporal abstraction within reinforcement learning (RL). Two well established approaches are through the successor representation (SR) and the default representation (DR). The SR encodes states by the future trajectories they induce, capturing information flow decoupled from reward. The DR builds on this by weighting trajectories with reward, integrating credit-assignment structure into the representation. Eigenvectors of both representations have been used to support a range of downstream tasks -- including option discovery, reward shaping, transfer learning, and exploration. We introduce a structurally distinct formulation: the terminal representation (TR). The TR encodes reward-weighted trajectories similarly to the DR, but can be learned as a lower-dimensionality object, and can be used directly for the mentioned applications without eigenvector computations. Eigendecomp
    
[^437]: 看见不等于知道：视觉语言模型知道何时不该回答空间问题（以及为什么）吗？

    Seeing Isn't Knowing: Do VLMs Know When Not to Answer Spatial Questions (and Why)?

    [https://arxiv.org/abs/2605.30557](https://arxiv.org/abs/2605.30557)

    该论文提出SPATIALUNCERTAIN受控评估框架，系统研究视觉语言模型在面对遮挡导致的证据缺失和视角导致的证据误导时的表现，强调可靠空间推理还要求模型能够判断当前观察是否足以支撑答案并主动识别更有信息量的观察视角。

    

    空间推理基准通常评估视觉语言模型能否从视觉观察中推导出正确答案。然而在真实的三维环境中，观察本身可能并不可靠：遮挡会移除与任务相关的证据，而视角可能使可见的几何信息产生误导。因此，可靠的空间推理不仅仅是正确回答问题——模型还必须评估其当前的观察是否为该答案提供了充分且可信的证据。我们提出了SPATIALUNCERTAIN，一个用于研究依赖视角的观察不确定性的受控评估框架。我们研究了两种互补的失败模式：由遮挡导致的证据缺失，以及由视角导致的证据误导。我们进一步评估了模型能否识别当前视角不可靠的情形，并找出更有信息量的观察。在八个开源和闭源视觉语言模型上……

    arXiv:2605.30557v2 Announce Type: replace-cross  Abstract: Spatial reasoning benchmarks typically evaluate whether vision-language models can derive the correct answer from a visual observation. Yet in real 3D environments, the observation itself may be unreliable: occlusion can remove task-relevant evidence, while perspective can make visible geometry misleading. Reliable spatial reasoning therefore requires more than answering a question correctly. A model must also assess whether its current observation provides sufficient and trustworthy evidence for that answer. We introduce SPATIALUNCERTAIN, a controlled evaluation framework for studying viewpoint-dependent observational uncertainty. We study two complementary failure modes: missing evidence caused by occlusion and misleading evidence caused by perspective. We further evaluate whether models can recognize when the current view is unreliable and identify a more informative observation. Across eight open- and closed-source vision-l
    
[^438]: FHRFormer：一种用于胎心率时间序列修复与预测的自监督掩码Transformer框架

    FHRFormer: A Self-Supervised Masked Transformer Framework for Fetal Heart Rate Time-Series Inpainting and Forecasting

    [https://arxiv.org/abs/2605.29695](https://arxiv.org/abs/2605.29695)

    该论文提出了FHRFormer，一种自监督掩码Transformer框架，能够修复和预测可穿戴胎心率监测中因信号丢失而产生的数据缺口，从而为基于AI分析连续胎心率数据以预测新生儿呼吸辅助风险奠定基础。

    

    大约10%的新生儿在出生时需要辅助才能开始呼吸，约5%需要通气支持。胎心率（FHR）监测在产前护理中评估胎儿健康状况方面发挥着至关重要的作用，能够检测异常模式，并支持及时的产科干预，以降低分娩过程中胎儿的风险。将人工智能（AI）方法应用于分析具有多样化结局的大规模连续FHR监测数据集，可能为预测需要呼吸辅助或干预的风险提供新的见解。可穿戴FHR监测仪的最新进展使连续胎儿监测成为可能，且不影响孕妇的活动能力。然而，孕妇活动期间的传感器移位，以及胎儿或母体体位的变化，常常导致信号丢失，造成记录的FHR数据出现缺口。这种数据缺失限制了有意义见解的提取。

    arXiv:2605.29695v2 Announce Type: replace  Abstract: Approximately 10% of newborns require assistance to initiate breathing at birth, and around 5% need ventilation support. Fetal heart rate (FHR) monitoring plays a crucial role in assessing fetal well-being during prenatal care, enabling the detection of abnormal patterns and supporting timely obstetric interventions to mitigate fetal risks during labor. Applying artificial intelligence (AI) methods to analyze large datasets of continuous FHR monitoring episodes with diverse outcomes may offer novel insights into predicting the risk of needing breathing assistance or interventions. Recent advances in wearable FHR monitors have enabled continuous fetal monitoring without compromising maternal mobility. However, sensor displacement during maternal movement, as well as changes in fetal or maternal position, often lead to signal dropout, resulting in gaps in recorded FHR data. Such missing data limits the extraction of meaningful insights
    
[^439]: 竞争性LLM智能体中利用秘密工具的自愿合谋行为

    Voluntary Collusion with Secret Tools in Competing LLM Agents

    [https://arxiv.org/abs/2605.27593](https://arxiv.org/abs/2605.27593)

    该论文通过竞争性欺骗和资源管理两个多智能体实验环境发现，即使工具被明确标记为不公平且有害，大多数LLM智能体仍会自愿接受秘密合谋工具并发展合谋策略，且仅靠不公平标签或基线安全对齐无法阻止这种行为，只有明确的伦理框架才能有效抑制合谋。

    

    摘要（arXiv:2605.27593v2，公告类型：替换）：即使某工具被明确描述为对他人不公平且有害，表面上经过安全对齐的LLM智能体在这样做能够带来策略优势时，仍然会自愿进行秘密合谋。为了研究这一现象，我们构建了一个基于两个策略性多智能体环境的实证框架：Liar's Bar（骗子酒吧，一个竞争性欺骗场景）和Cleanup（清理行动，一个混合动机的资源管理场景），在这些场景中，智能体被提供秘密合谋工具，这些工具能够带来显著优势，同时明显损害其他智能体的利益。我们在12个模型（涵盖7B、70B及专有模型规模）和6种提示词变体上进行了实验，发现大多数智能体会持续接受这些工具并发展出合谋策略，同时在接受之前明确承认这些工具的不公平性。我们进一步表明，不公平性标签和基线安全对齐本身都不能可靠地阻止合谋：只有明确的伦理框架才能……

    arXiv:2605.27593v2 Announce Type: replace  Abstract: Even when a tool is explicitly described as unfair and harmful to others, ostensibly safety-aligned LLM agents still voluntarily engage in secret collusion whenever doing so confers a strategic advantage. To investigate this phenomenon, we introduce an empirical framework built on two strategic multi-agent environments: Liar's Bar, a competitive deception scenario, and Cleanup, a mixed-motive resource-management scenario, in which agents are offered secret collusion tools that provide significant advantages while clearly disadvantaging the other agents. Across 12 models (at the 7B, 70B, and proprietary scales) and 6 prompt variants, we find that most agents consistently accept these tools and develop collusive strategies, while explicitly acknowledging the unfairness of the tools before accepting. We further show that neither the unfairness labels nor baseline alignment alone reliably deters collusion: only explicit ethical framing r
    
[^440]: D3S2：面向语义分割的扩散引导数据集蒸馏

    D3S2: Diffusion-Guided Dataset Distillation for Semantic Segmentation

    [https://arxiv.org/abs/2605.25022](https://arxiv.org/abs/2605.25022)

    提出了首个面向语义分割的扩散引导数据集蒸馏框架D3S2，通过类别平衡掩码选择与扩散引导图像合成的两阶段设计，解决了长尾类别不平衡、像素级对齐和高计算成本三大挑战。

    

    数据集蒸馏（DD）旨在将大规模数据集压缩成紧凑的合成数据集，同时保持训练效果。然而，现有研究主要集中在图像分类任务上，而对语义分割等密集预测任务的探索仍然严重不足。在本工作中，我们识别出语义分割数据集蒸馏面临的三个关键挑战：（i）长尾类别不平衡问题，（ii）图像与密集标签之间需要严格的像素级对齐，以及（iii）使用复杂模型优化高分辨率数据所带来的高计算成本。为解决这些挑战，我们提出了D3S2——一个面向语义分割的扩散引导数据集蒸馏框架。我们的方法采用两阶段设计：在类别平衡掩码选择阶段，通过贪心策略构建一个优先考虑代表性不足类别的代表性掩码集合；在扩散引导图像合成阶段，采用预训练的布局到图像扩散模型……

    arXiv:2605.25022v2 Announce Type: replace-cross  Abstract: Dataset distillation (DD) aims to compress large-scale datasets into compact synthetic sets while preserving training efficacy. However, existing studies mainly focus on image classification, leaving dense prediction tasks such as semantic segmentation largely underexplored. In this work, we identify three key challenges for segmentation DD: (i) long-tailed class imbalance, (ii) the need for strict pixel-wise alignment between images and dense labels, and (iii) the high computational cost of optimizing high-resolution data with complex models. To address these challenges, we propose D3S2, a Diffusion-guided Dataset Distillation framework for Semantic Segmentation. Our method adopts a two-stage design. In Class-Balanced Mask Selection, we construct a representative mask set via a greedy strategy that prioritizes underrepresented classes. In Diffusion-Guided Image Synthesis, we employ a pretrained layout-to-image diffusion model 
    
[^441]: Palette：一个模块化、可控、高效的大语言模型按需授权安全对齐放宽框架

    Palette: A Modular, Controllable, and Efficient Framework for On-demand Authorized Safety Alignment Relaxation in LLMs

    [https://arxiv.org/abs/2605.24154](https://arxiv.org/abs/2605.24154)

    Palette 提出了一个模块化、可控且高效的框架，通过多目标搜索识别拒绝方向并借助轻量级适配将其内化到模型中，从而按需放宽授权领域的安全拒绝行为，同时保持其他领域的标准安全性。

    

    当前基础模型的安全对齐主要遵循“一刀切”范式，即对所有用户和情境应用相同的拒绝策略。这导致模型可能会拒绝那些对普通用户不安全、但对授权专业人士而言合法的请求，从而限制了模型在专业场景中的实用性。现有方法要么需要代价高昂的重新对齐，要么依赖推理时的引导技术，但后者存在控制不精确和额外延迟的问题。为此，我们提出了 Palette，一个模块化、可控且高效的框架，能够有选择地放宽授权目标领域上的拒绝行为，同时在其他方面保持标准的安全性。我们的方法通过多目标搜索识别拒绝方向，并通过轻量级适配将其内化到模型中。Palette 还进一步支持模块化组合：它可以独立学习领域特定的安全控制并进行组合。

    arXiv:2605.24154v2 Announce Type: replace  Abstract: Current safety alignment of foundation models largely follows a \emph{one-size-fits-all} paradigm, applying the same refusal policy across users and contexts. As a result, models may refuse requests that are unsafe for general users but legitimate for authorized professionals, limiting helpfulness in specialized professional settings. Existing approaches either require costly realignment or rely on inference-time steering that suffers from imprecise control and added latency. To this end, we propose \textsc{Palette}, a modular, controllable, and efficient framework that selectively relaxes refusal behavior on authorized target domains while preserving standard safety elsewhere. Our method identifies a refusal direction via multi-objective search and internalizes it into the model through lightweight adaptation. \textsc{Palette} further supports modular composition: it learns domain-specific safety controls independently and composes 
    
[^442]: 基于预测分布的强化学习用于大语言模型回归

    Reinforcement Learning over Predictive Distributions for LLM Regression

    [https://arxiv.org/abs/2605.20740](https://arxiv.org/abs/2605.20740)

    提出了分布感知奖励（DAR），一种同策略强化学习目标，通过留一法贡献对同一输入的多个预测所形成的预测分布进行联合评估，从而提升大语言模型回归的校准质量。

    

    大语言模型（LLMs）已成为灵活的回归器，能够从异构输入中预测实值数量。然而，大多数LLM回归目标独立地优化各个预测，往往导致校准效果不佳。我们提出了分布感知奖励，这是一种同策略强化学习目标，转而对同一输入的多个预测所形成的经验预测分布进行联合评估。为了将这种分布级别的目标转化为逐个rollout级别的奖励，我们根据每个预测对整体预测分布质量的留一法贡献来分配其信用。这种方法鼓励预测在目标值周围良好居中且具有适当的离散程度。我们在三种回归设置上进行了评估：一个用于探测插值和外推能力的合成任务，以及两个涉及代码和分子数据的真实世界科学任务。在所有任务中，DA

    arXiv:2605.20740v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have emerged as flexible regressors capable of predicting real-valued quantities from heterogeneous inputs. Yet most LLM regression objectives optimize predictions independently, often yielding poor calibration. We introduce Distribution-Aware Reward (DAR), an on-policy reinforcement learning objective that instead jointly evaluates the empirical predictive distribution formed by multiple predictions for the same input. To translate this distribution-level objective into rollout-level rewards, we assign each prediction credit based on its leave-one-out contribution to the quality of the overall predictive distribution. This encourages predictions that are well-centered and appropriately dispersed around the target. We evaluate on three regression settings: a synthetic task probing interpolation and extrapolation, and two real-world scientific tasks involving code and molecular data. Across tasks, DA
    
[^443]: 调用还是不调用：诊断大语言模型智能体中的内在过度调用偏差

    To Call or Not to Call: Diagnosing Intrinsic Over-Calling Bias in LLM Agents

    [https://arxiv.org/abs/2605.18882](https://arxiv.org/abs/2605.18882)

    该研究提出并验证了“内在偏差假说”（IBH），揭示大语言模型智能体在调用/不调用决策中存在与激活无关的过度调用偏差，利用稀疏自编码器定位并量化该偏差，并通过自适应边际校准转向（AMCS）方法实现了因果层面的纠正。

    

    大语言模型（LLM）智能体表现出一种持续存在的“过度调用”倾向，即使在没有调用需求的情况下也会调用工具。在When2Call基准测试中，来自三个模型系列的六个模型均显示出较高的“调用”准确率，但“不调用”的准确率要低得多，导致整体准确率处于55%-70%的区间。我们将这一现象归因于“内在偏差假说”（Intrinsic Bias Hypothesis, IBH）：调用/不调用的决策映射携带一个与激活无关的调用偏移量，因此即使在激活水平相等的情况下，模型也倾向于选择调用。利用稀疏自编码器（SAE），我们恢复了与行为对齐的调用/不调用决策特征基，将其简化为一个带符号的激活边际，并直接估计出该偏移量。在全部六个模型中，只有当“不调用”激活超过“调用”激活时，模型才能实现决策中立，这与IBH一致。随后，我们通过自适应边际校准转向（AMCS）方法对IBH进行因果验证，AMCS是一种沿SAE解码器方向施加闭式反偏差偏移的技术。通过抵消所诊断出的偏移量，该方法能够有效纠正模型的过度调用行为（摘要在此处截断）。

    arXiv:2605.18882v2 Announce Type: replace-cross  Abstract: LLM agents exhibit a consistent tendency to over-call, invoking tools even in situations where none is needed. On the When2Call benchmark, six models from three families show high call accuracy but much lower no-call accuracy, leaving overall accuracy in the 55%-70% range. We trace this to an Intrinsic Bias Hypothesis (IBH): the call/no-call decision mapping carries an activation-independent call offset, so the model favors call even at activation parity. Using Sparse Autoencoders (SAEs), we recover behavior-aligned feature bases for the call/no_call decision, reduce them to a signed activation margin, and estimate the offset directly. Across all six models, the model is decision-neutral only when no_call activation outweighs call activation, consistent with IBH. We then causally test IBH with Adaptive Margin-Calibrated Steering (AMCS), a closed-form counter-bias shift along SAE decoder directions. Cancelling the diagnosed offs
    
[^444]: KadiAssistant：一个用于Kadi4Mat信息检索的对话式AI智能体

    KadiAssistant: A conversational AI Agent for information retrieval in Kadi4Mat

    [https://arxiv.org/abs/2605.18850](https://arxiv.org/abs/2605.18850)

    本文提出了KadiAssistant，一个集成于Kadi研究数据生态系统中、以隐私保护为设计核心的对话式AI智能体，使研究人员能够高效访问、聚合和综合异构且隐私敏感的研究数据信息。

    

    我们介绍了KadiAssistant，一个采用隐私设计理念并集成于Kadi研究数据生态系统中的AI助手，使研究人员能够高效地访问、聚合和综合来自异构且涉及隐私敏感的研究数据的信息。材料科学等跨学科领域汇集了各自拥有不同术语和标准的学科。虽然这种融合推动了创新，但也使得知识的连接和获取日益困难，因为数据分散在各个学科、组织和个人之间。例如，电池研究结合了电化学测量、材料表征数据、基于物理的模拟以及制造参数，而每个部分都使用不同的格式、词汇和标准。通过Kadi4Mat等研究数据平台高效地存储和共享此类异构数据，需要领域知识、技术专长以及对相关工具的熟悉（摘要原文在此处被截断）。

    arXiv:2605.18850v2 Announce Type: replace-cross  Abstract: We introduce KadiAssistant, a privacy-by-design AI assistant integrated into the Kadi research data ecosystem, enabling researchers to efficiently access, aggregate, and synthesize information from heterogeneous, privacy-sensitive research data. Interdisciplinary fields such as materials science bring together disciplines with their own terminology and standards. While this convergence fuels innovation, it also makes it increasingly difficult to connect and access knowledge, as data are distributed across disciplines, organizations, and individuals. For example, battery research combines electrochemical measurements, materials characterization data, physics-based simulations, and manufacturing parameters, each using different formats, vocabularies, and standards. Efficiently storing and sharing such heterogeneous data via research data platforms, such as Kadi4Mat, demands domain knowledge, technical expertise, and familiarity w
    
[^445]: 面向约束机器学习的随机惩罚-障碍方法

    Stochastic Penalty-Barrier Method for Constrained Machine Learning

    [https://arxiv.org/abs/2605.18618](https://arxiv.org/abs/2605.18618)

    本文提出随机惩罚-障碍方法（SPBM），通过对偶变量指数平均、稳定惩罚调度和Moreau包络扩展经典惩罚-障碍方法以求解约束机器学习问题，证明了小批量采样下变换问题可行集包含于原问题可行集，实验表明其性能与最先进方法相当。

    

    约束机器学习（CML）能够实现公平性感知的训练、物理信息神经网络，以及将符号化领域知识整合到统计模型中。在这项工作中，我们针对CML问题提出了随机惩罚-障碍方法（SPBM）。SPBM通过引入对偶变量的指数平均、稳定的惩罚调度以及Moreau包络来处理非光滑性，从而扩展了经典的惩罚方法与障碍方法。我们分析了小批量采样在障碍函数中引入的偏差，并证明所得变换问题的可行集包含于原始问题的可行集之内。我们在多个公平性和物理信息神经网络实验中将SPBM与CML基线方法进行了比较，发现SPBM与最先进的方法相比具有竞争力。我们还观察到，在基于公平性的计算基准测试中，CML方法的每轮（epoch）运行时间在很大程度上是独立的。

    arXiv:2605.18618v3 Announce Type: replace-cross  Abstract: Constrained Machine Learning (CML) enables fairness-aware training, physics-informed neural networks, and integration of symbolic domain knowledge into statistical models. In this work, we introduce the Stochastic Penalty-Barrier Method (SPBM) for CML problems. SPBM extends classical penalty and barrier methods by incorporating an exponential averaging of the dual variables, a stabilized penalty schedule, and the Moreau envelope to handle non-smoothness. We analyze the bias that mini-batching introduces in the barrier function and show that the feasible set of the resulting transformed problem is contained within the original one. We compare SPBM with CML baselines across multiple fairness and physics informed neural networks experiments. We find that SPBM is competitive with state-of-the-art methods. We also observe, on our fairness-based computational benchmark, that the per-epoch runtime of CML methods is largely independent
    
[^446]: 语音“克隆”即风格迁移

    Voice "Cloning" is Style Transfer

    [https://arxiv.org/abs/2605.16578](https://arxiv.org/abs/2605.16578)

    这篇论文揭示语音克隆并非真正“克隆”个人声音，而是系统性地进行风格迁移，使克隆语音比源语音显得更权威、温暖且更像人类，并导致说话人特征的同质化。

    

    人工生成的语音正日益融入日常生活。语音克隆技术尤其能够支持身份保持至关重要的应用场景，例如补全录音、用新语言配音，或为失语症患者保留声音。然而，在我们的研究中发现，尽管名为“克隆”，语音克隆并不能忠实地“克隆”个人的声音。相反，我们发现广泛使用的语音克隆模型系统性地对源语音施加了风格迁移。根据人工标注者的评价，与源语音相比，克隆语音被认为更具权威性、更温暖、更像客服、更像人类。人工标注者还报告称，与源语音相比，他们对克隆语音的信任度更高，并更愿意向克隆语音透露敏感的个人信息。我们的研究进一步表明，语音克隆会导致说话人特征的同质化，这通过降低的……（原文此处截断）来衡量。

    arXiv:2605.16578v4 Announce Type: replace-cross  Abstract: Artificially generated speech is increasingly embedded in everyday life. Voice cloning in particular enables applications where identity preservation is important, such as completing a recording, dubbing in a new language, or preserving the voices of individuals with speech loss. However, in our work, we find that despite the term, voice cloning does not faithfully ''clone'' an individual's voice. Instead, we find that widely-used voice cloning models systematically apply style transfer to source voices. As rated by human annotators, cloned voices are perceived as more authoritative, warm, customer-service-like, and human-like compared to their sources. Human annotators also report greater trust in cloned voices than source voices, and a greater willingness to disclose sensitive personal information to them. Our work furthermore shows that voice cloning leads to homogenization of speaker characteristics, as measured by reduced 
    
[^447]: PBT-Bench：基于属性测试的AI智能体基准测试

    PBT-Bench: Benchmarking AI Agents on Property-Based Testing

    [https://arxiv.org/abs/2605.15229](https://arxiv.org/abs/2605.15229)

    PBT-Bench是一个包含100个覆盖40个真实Python库的基于属性测试问题的基准，通过注入默认随机输入几乎无法触发的语义bug，专门评估AI智能体从文档中推导语义不变量并设计精确输入生成策略的能力。

    

    现有的代码基准测试衡量的是智能体能否生成任何能重现已知bug的测试，或者能否生成修复所描述问题的补丁。这两者都没有隔离出基于属性测试这一独特技能：即从文档中推导出语义不变量，然后构建一个足够精确的输入生成策略，使得随机搜索能够揭示违规行为。我们提出了PBT-Bench，这是一个包含100个精心筛选的基于属性测试问题的基准测试，涵盖40个真实的Python库。每个问题注入一个或多个语义bug（共365个，平均每个问题3.65个），其设计使得默认策略的随机输入几乎从不触发这些bug；智能体必须阅读库的文档，识别相关的不变量，并指定一个Hypothesis @given策略，将概率质量集中在触发区域。bug按三个难度级别（L1-L3）进行分层，涵盖单约束边界……

    arXiv:2605.15229v4 Announce Type: replace-cross  Abstract: Existing code benchmarks measure whether an agent can produce any test that reproduces a known bug, or whether it can produce a   patch that fixes a described issue. Neither isolates the distinct skill of property-based testing: deriving a semantic invariant   from documentation, and then constructing an input-generation strategy precise enough to make a random search reveal the violation.   We introduce PBT-Bench, a benchmark of 100 curated property-based testing problems across 40 real Python libraries. Each problem   injects one or more semantic bugs (365 in total, mean 3.65 per problem) designed so that default-strategy random inputs almost   never trigger them; the agent must read the library's documentation, identify the relevant invariant, and specify a Hypothesis   @given strategy that concentrates mass in the trigger region. Bugs are stratified across three difficulty levels (L1-L3) spanning   single-constraint boundar
    
[^448]: Ego2World：将第一人称烹饪视频编译为可执行世界以支持信念状态规划

    Ego2World: Compiling Egocentric Cooking Videos into Executable Worlds for Belief-State Planning

    [https://arxiv.org/abs/2605.13335](https://arxiv.org/abs/2605.13335)

    Ego2World将第一人称烹饪视频编译为由图转移规则控制的可执行符号世界，使智能体在部分可观测条件下仅凭局部观测和执行反馈进行信念状态规划，弥合了被动视频数据集与合成模拟器之间的差距。

    

    家庭环境中的具身智能体必须在部分可观测条件下进行规划：它们需要记住物体、追踪状态变化，并在动作失败时进行恢复。现有基准测试只能部分地检验这种能力。第一人称视角视频数据集捕捉了真实的人类活动，但仍然是被动式的；而交互式模拟器虽然支持执行，却依赖于合成场景和手工设计的动态规则，这引入了模拟与现实的差距，且通常假设状态是完全可观测的。我们提出了Ego2World，这是一个可执行的基准测试，它将第一人称烹饪视频转化为由图转移规则控制的可执行符号世界。Ego2World基于HD-EPIC构建，从视频标注中提取可复用的转移规则，并在一个隐藏的符号世界图中执行它们。在评估过程中，模拟器维护隐藏的世界图，而智能体仅使用局部观测和执行反馈，在自己的部分信念图上进行规划。

    arXiv:2605.13335v2 Announce Type: replace  Abstract: Embodied agents in household environments must plan under partial observation: they need to remember objects, track state changes, and recover when actions fail. Existing benchmarks only partially test this ability. Egocentric video datasets capture realistic human activities but remain passive, while interactive simulators support execution but rely on synthetic scenes and hand-crafted dynamics, introducing a sim-to-real gap and often assuming fully observable state. We introduce Ego2World, an executable benchmark that turns egocentric cooking videos into executable symbolic worlds governed by graph-transition rules. Built on HD-EPIC, Ego2World derives reusable transition rules from video annotations and executes them in a hidden symbolic world graph. During evaluation, the simulator maintains the hidden world graph, while the agent plans over its own partial belief graph using only local observations and execution feedback. This se
    
[^449]: 当注意力关闭时：大语言模型如何在多轮交互中迷失主线

    When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction

    [https://arxiv.org/abs/2605.12922](https://arxiv.org/abs/2605.12922)

    该论文提出“通道转换”机制来解释大语言模型在多轮交互中丢失指令主线的原因，并引入目标可及性比率这一新指标，揭示不同架构的模型在注意力衰减后呈现出性质截然不同的失败模式。

    

    大语言模型能够在单轮交互中遵循复杂指令，但在长期的多轮交互中，它们常常丢失指令、角色设定和规则的主线。这种性能退化已经在行为层面被测量，但尚未得到机制层面的解释。我们提出了一种“通道转换”解释：定义目标的词元通过注意力变得难以访问，而目标相关信息可能残留在残差表示中。我们引入了目标可及性比率，用于衡量生成词元对任务定义目标词元的注意力，并将其与滑动窗口消融和残差流探针相结合。当对指令的注意力关闭时，什么得以留存揭示了架构的差异。在不同架构中，这种转换产生了性质截然不同的失败模式：一些模型在注意力消失时仍能保持目标条件化的行为，另一些模型尽管残差中存在可解码的目标信息却仍然失败，而编码这一信息的层位置……

    arXiv:2605.12922v2 Announce Type: replace  Abstract: Large language models can follow complex instructions in a single turn, yet over long multi-turn interactions they often lose the thread of instructions, persona, and rules. This degradation has been measured behaviorally but not mechanistically explained. We propose a channel-transition account: goal-defining tokens become less accessible through attention, while goal-related information may persist in residual representations. We introduce the Goal Accessibility Ratio (GAR), measuring attention from generated tokens to task-defining goal tokens, and combine it with sliding-window ablations and residual-stream probes. When attention to instructions closes, what survives reveals architecture. Across architectures, the transition yields qualitatively distinct failure modes: some models preserve goal-conditioned behavior at vanishing attention, others fail despite decodable residual goal information, and the layer at which this encodin
    
[^450]: 路径奖励：为知识图谱问答学习中间监督信号

    Reward on Path: Learning Intermediate Supervision Signals for Knowledge Graph Question Answering

    [https://arxiv.org/abs/2605.10791](https://arxiv.org/abs/2605.10791)

    提出RoP框架，通过非对称目标从答案标签中学习轻量级、问题条件化的路径奖励，为知识图谱问答中基于LLM的关系路径生成器提供有效的中间监督信号。

    

    知识图谱问答（KGQA）旨在通过在知识图谱（KG）上进行推理来回答用户的问题。最近的方法使用从答案标签中获得的监督信号，或由大型语言模型（LLM）优化的监督信号，来训练模型检索知识图谱证据，以供基于LLM的答案推理使用。然而，基于答案的监督将每条到达答案的路径都视为正确，因此会产生嘈杂的训练信号；而由LLM优化的监督虽然能减轻这种噪声，但成本高昂。为了解决这些局限性，我们提出了路径奖励，这是一个通过非对称目标从答案标签中学习轻量级、问题条件化路径奖励的框架。到达同一答案的路径作为一个包被联合监督，使奖励模型能够学习它们的相对贡献，同时每条路径因检索到非答案实体而被单独惩罚。学习到的奖励随后用于训练基于LLM的关系路径生成器，在两个……

    arXiv:2605.10791v2 Announce Type: replace  Abstract: Knowledge Graph Question Answering (KGQA) aims to answer user questions by reasoning over Knowledge Graphs (KGs). Recent methods use supervision derived from answer labels or refined by Large Language Models (LLMs) to train models that retrieve KG evidence for LLM-based answer reasoning. However, answer-derived supervision treats every answer-reaching path as correct and thus yields noisy training signals, whereas LLM-refined supervision mitigates this noise at substantial cost. To address these limitations, we propose Reward on Path (RoP), a framework to learn a lightweight, question-conditioned path reward from answer labels with an asymmetric objective. Paths reaching the same answer are supervised jointly as a bag, allowing the reward model to learn their relative contributions, while each path is penalized individually for retrieving non-answer entities. The learned reward then trains an LLM-based relation path generator in two 
    
[^451]: 基于残差潜在动作学习视觉特征世界模型

    Learning Visual Feature-Based World Models via Residual Latent Action

    [https://arxiv.org/abs/2605.07079](https://arxiv.org/abs/2605.07079)

    该论文提出了一种可从DINO残差中轻松学习的新型潜在动作表示“残差潜在动作”（RLA），并基于流匹配构建RLA世界模型（RLA-WM），实现了更高效、更少幻觉且预测质量更优的视觉特征世界模型。

    

    世界模型从观测和动作中预测未来的状态转移。现有工作主要专注于图像生成。而基于视觉特征的世界模型预测的是未来的视觉特征而非原始视频像素，这提供了一种更高效且更不容易产生幻觉的有前景的替代方案。然而，当前基于特征的方法依赖于直接回归，这在复杂交互中会导致模糊或坍缩的预测，而在高维特征空间中进行生成式建模仍然具有挑战性。在这项工作中，我们发现一种新型的潜在动作表示——我们称之为残差潜在动作，可以轻松地从DINO残差中学习得到。我们还证明了RLA具有预测性、可泛化性，并能编码时间进程。基于RLA，我们提出了RLA世界模型（RLA-WM），它通过流匹配来预测RLA值。RLA-WM在性能上优于两者。

    arXiv:2605.07079v2 Announce Type: replace-cross  Abstract: World models predict future transitions from observations and actions. Existing works predominantly focus on image generation only. Visual feature-based world models, on the other hand, predict future visual features instead of raw video pixels, offering a promising alternative that is more efficient and less prone to hallucination. However, current feature-based approaches rely on direct regression, which leads to blurry or collapsed predictions in complex interactions, while generative modeling in high-dimensional feature spaces still remains challenging. In this work, we discover that a new type of latent action representation, which we refer to as Residual Latent Action (RLA), can be easily learned from DINO residuals. We also show that RLA is predictive, generalizable, and encodes temporal progression. Building on RLA, we propose RLA World Model (RLA-WM), which predicts RLA values via flow matching. RLA-WM outperforms both
    
[^452]: 重新思考适配器放置：主导适配模块的视角

    Rethinking Adapter Placement: A Dominant Adaptation Module Perspective

    [https://arxiv.org/abs/2605.06183](https://arxiv.org/abs/2605.06183)

    该论文提出PAGE探测方法，发现LoRA可训练梯度能量高度集中于单一浅层FFN下投影的“主导适配模块”，其位置由模型架构决定且跨任务稳定，为有限数量适配器的最优放置提供了明确指导。

    

    低秩适配是一种广泛使用的参数高效微调方法，它将可训练的低秩适配器插入到冻结的预训练模型中。近期研究表明，使用更少的LoRA适配器仍可能保持甚至提升性能，但现有方法仍然广泛地分布适配器，因此“将有限数量的适配器放置在何处以最大化性能”这一问题在很大程度上仍未解决。为了研究这一问题，我们提出了PAGE（投影适配器梯度能量），这是一种基于梯度的敏感性探测方法，用于估计每个候选LoRA适配器可获得的初始可训练梯度能量。令人惊讶的是，我们发现PAGE在两个模型家族和四个下游任务上高度集中于同一个浅层FFN下投影模块。我们将该模块称为主导适配模块，并证明其所在层索引依赖于架构但跨任务保持稳定。

    arXiv:2605.06183v2 Announce Type: replace  Abstract: Low-rank adaptation (LoRA) is a widely used parameter-efficient fine-tuning method that places trainable low-rank adapters into frozen pre-trained models. Recent studies show that using fewer LoRA adapters may still maintain or even improve performance, but existing methods still distribute adapters broadly, leaving \emph{where to place a limited number of adapters to maximize performance} largely open. To investigate this, we introduce \textbf{PAGE} (\textbf{P}rojected \textbf{A}dapter \textbf{G}radient \textbf{E}nergy), a gradient-based sensitivity probe that estimates the initial trainable gradient energy available to each candidate LoRA adapter. Surprisingly, we find that PAGE is highly concentrated on a single shallow FFN down-projection across two model families and four downstream tasks. We term this module the \textbf{dominant adaptation module} and show that its layer index is architecture-dependent but task-stable. Motivate
    
[^453]: XDecomposer：面向多相X射线衍射的无先验集合分解学习方法

    XDecomposer: Learning Prior-Free Set Decomposition for Multiphase X-ray Diffraction

    [https://arxiv.org/abs/2605.05866](https://arxiv.org/abs/2605.05866)

    提出XDecomposer，将多相XRD分析形式化为集合预测问题，无需候选相列表、结构模板或相数量先验即可实现多相XRD图谱的联合分解与结构识别。

    

    多相粉末X射线衍射（PXRD）分析仍然是结构鉴定中的一个基础性瓶颈，因为现实中的合成往往会生成复杂的混合物，其组成相（组分）难以被可靠地分离。尽管近年来基于表示的晶体检索与生成方面的进展表明，直接从PXRD推断结构已成为可能，但现有方法大多假设输入为单相，在多相场景下会失效。本文提出了XDecomposer，这是一个无需先验知识的框架，能够在不需要候选相列表、结构模板或相数量先验知识的情况下，对多相XRD图谱进行联合分解与识别。我们将多相衍射分析形式化为一个集合预测问题，模型在统一架构内推断出一个无序的相分辨组分集合、各组分的混合比例以及相应的结构表示。

    arXiv:2605.05866v2 Announce Type: replace  Abstract: Multiphase powder X-ray diffraction (PXRD) analysis remains a fundamental bottleneck in structure identification, as real-world synthesis often produces complex mixtures whose constituent phases (components) cannot be reliably disentangled. While recent advances in representation-based crystal retrieval and generation suggest the possibility of inferring structures directly from PXRD, existing approaches largely assume single-phase inputs and break down in multiphase settings. Here, we present XDecomposer, a prior-free framework for joint decomposition and identification of multiphase XRD patterns without requiring candidate phase lists, structural templates, or prior knowledge of phase number. We formulate multiphase diffraction analysis as a set prediction problem, where the model infers an unordered set of phase-resolved components, their mixture proportions, and corresponding structural representations within a unified architectu
    
[^454]: 冯·诺依曼网络

    Von Neumann Networks

    [https://arxiv.org/abs/2605.05780](https://arxiv.org/abs/2605.05780)

    该论文将冯·诺依曼上世纪的细胞计算模型与现代深度学习相结合，提出了冯·诺依曼神经元及其网络（VNNs），其架构可自组织生成、仅依赖于输入输出在细胞阵列上的位置，并在数学上基于神经算子扩展与格林函数学习。

    

    二十世纪中叶，数学家兼博学家约翰·冯·诺依曼创建了一个建立在细胞阵列上的计算系统，作为人脑的简单模型，其中每个细胞具有有限集合中的一种角色或状态，他预测这些状态将通过扩散过程来建模。在这项工作中，我们展示了这种系统在现代深度学习环境下的发展，使得构建一种具有可学习的专门化角色的人工神经元成为可能。我们将这种神经元称为冯·诺依曼神经元，由这类神经元构成的神经网络形成了一种自工程化设计，其架构仅取决于其输入和输出在该细胞阵列上的结构与位置。我们还构建了冯·诺依曼网络（VNNs）的数学框架，并证明它们是基于神经算子的扩展以及在细胞阵列上通过卷积学习格林函数的方法（摘要在此处截断）。

    arXiv:2605.05780v2 Announce Type: replace  Abstract: In the mid-twentieth century, mathematician and polymath John von Neumann created a computational system on an array of cells as a simple model of the human brain, where each cell had one of a finite set of roles or states that he predicted would be modelled by a diffusion process. In this work, we show that such a system, when developed in a modern deep learning setting, enables the construction of an artificial neuron having specialized roles that can be learnt. We refer to this neuron as the Von Neumann neuron, and the resulting neural network from such neurons result in a self-engineered design whose architecture is only dependent on the structure and locations of its inputs and outputs on this cellular array. The mathematical framework for these Von Neumann Networks (VNNs) is also constructed and shows that they are based on the extension of neural operators and the learning of Green's functions with convolutions on a cellular t
    
[^455]: SDFlow：面向时间序列生成的相似性驱动流匹配

    SDFlow: Similarity-Driven Flow Matching for Time Series Generation

    [https://arxiv.org/abs/2605.05736](https://arxiv.org/abs/2605.05736)

    SDFlow提出了一种在冻结VQ潜在空间中运行的相似性驱动流匹配非自回归框架，通过全局传输映射消除曝光偏差，并结合低秩流形分解与离散监督，实现高质量的时间序列并行生成。

    

    基于向量量化（VQ）与自回归（AR）词元建模是时间序列生成领域中被广泛采用且极具竞争力的范式。然而，此类模型从根本上受到曝光偏差的限制：在推理过程中，误差会在逐步预测中不断累积，导致长时程生成中出现明显的质量下降。为解决这一问题，我们提出了SDFlow（相似性驱动的流匹配），这是一个完全在冻结VQ潜在空间中运行的非自回归框架，通过流匹配实现并行序列生成。我们解决了实现这一转变过程中的三个关键挑战：（1）通过用全局传输映射替代逐步词元预测来消除曝光偏差；（2）通过在潜在流形上引入带有可学习锚点先验的低秩流形分解，缓解VQ词元空间的高维性问题；（3）引入离散监督（摘要在此处截断）。

    arXiv:2605.05736v3 Announce Type: replace  Abstract: Vector quantization (VQ) with autoregressive (AR) token modeling is a widely adopted and highly competitive paradigm for time-series generation. However, such models are fundamentally limited by exposure bias: during inference, errors can accumulate across sequential predictions, leading to pronounced quality degradation in long-horizon generation. To address this, we propose SDFlow ($\textbf{S}$imilarity-$\textbf{D}$riven $\textbf{Flow}$ Matching), a non-autoregressive framework that operates entirely in the frozen VQ latent space and enables parallel sequence generation via flow matching. We tackle three key challenges in making this transition: (1) eliminating exposure bias by replacing step-wise token prediction with a global transport map; (2) mitigating the high-dimensionality of VQ token spaces via a low-rank manifold decomposition with a learned anchor prior over the latent manifold; and (3) incorporating discrete supervision
    
[^456]: 基于文本嵌入的零领域知识算法选择

    Algorithm Selection with Zero Domain Knowledge via Text Embeddings

    [https://arxiv.org/abs/2604.19753](https://arxiv.org/abs/2604.19753)

    ZeroFolio利用预训练文本嵌入替代手工设计的实例特征，实现了零领域知识的算法选择，在涵盖7个领域的11个ASlib场景中的绝大多数上超越了传统方法。

    

    我们提出了ZeroFolio，一种无特征的算法选择方法，它使用预训练的文本嵌入来替代手工设计的实例特征。该方法将原始实例文件作为纯文本读取，使用预训练的嵌入模型对其进行嵌入，并通过加权k近邻算法选择合适的算法。我们的方法基于这样一个观察：预训练嵌入无需任何领域知识或任务特定的训练即可区分问题实例。ZeroFolio适用于任何实例格式为文本的问题领域。我们在涵盖7个领域（SAT、MaxSAT、QBF、ASP、CSP、MIP和图问题）的11个ASlib场景上对该方法进行了评估。ZeroFolio在11个场景中的9个上优于基于手工特征训练的随机森林，且优势通常十分显著，其中8个场景在所有序列化种子下均保持优势。与针对每个场景单独调优的随机森林相比，它在11个场景中的8个上获胜。在有公开AutoFolio结果的三个场景上……（原文摘要在此处截断）

    arXiv:2604.19753v3 Announce Type: replace  Abstract: We propose ZeroFolio, a feature-free approach to algorithm selection that uses pretrained text embeddings instead of hand-crafted instance features. It reads the raw instance file as plain text, embeds it with a pretrained embedding model, and selects an algorithm via weighted k-nearest neighbors. Our approach is based on the observation that pretrained embeddings can distinguish problem instances without any domain knowledge or task-specific training. ZeroFolio applies to any problem domain with text-based instance formats. We evaluate our approach on 11 ASlib scenarios spanning 7 domains (SAT, MaxSAT, QBF, ASP, CSP, MIP, and graph problems). ZeroFolio outperforms a random forest trained on hand-crafted features in 9 of 11 scenarios, often substantially, and in 8 of them with every serialization seed. It wins 8 of 11 scenarios against a per-scenario-tuned random forest. On the three scenarios with published AutoFolio results from th
    
[^457]: 关于使用进化优化求解动态机会约束露天矿调度问题的研究

    On the use of evolutionary optimization for the dynamic chance constrained open-pit mine scheduling problem

    [https://arxiv.org/abs/2604.13385](https://arxiv.org/abs/2604.13385)

    本文提出一种结合基于多样性变化响应机制的双目标进化优化方法，用于求解块段经济价值随机且采矿与加工能力随时间变化的动态机会约束露天矿调度问题，在最大化期望折现利润的同时最小化其波动性。

    

    露天矿调度是一个复杂的现实世界优化问题，涉及不确定的经济价值和动态变化的资源容量。进化算法在这类场景中特别有效，因为它们能够轻松适应不确定和变化的环境。然而，在现实问题中，不确定性和动态变化往往被孤立地研究。本文研究了一个动态机会约束的露天矿调度问题，其中块段经济价值是随机的，且采矿和加工能力随时间变化。我们采用双目标进化建模方法，同时最大化期望折现利润并最小化其标准差。为应对动态变化，我们提出了一种基于多样性的变化响应机制，当检测到变化时，该机制会修复部分不可行解并引入额外的可行解。我们评估了该方法（摘要在此处截断）。

    arXiv:2604.13385v3 Announce Type: replace-cross  Abstract: Open-pit mine scheduling is a complex real-world optimization problem that involves uncertain economic values and dynamically changing resource capacities. Evolutionary algorithms are particularly effective in these scenarios, as they can easily adapt to uncertain and changing environments. However, uncertainty and dynamic changes are often studied in isolation in real-world problems. In this paper, we study a dynamic chance-constrained open-pit mine scheduling problem in which block economic values are stochastic and mining and processing capacities vary over time. We adopt a bi-objective evolutionary formulation that simultaneously maximizes expected discounted profit and minimizes its standard deviation. To address dynamic changes, we propose a diversity-based change response mechanism that repairs a subset of infeasible solutions and introduces additional feasible solutions whenever a change is detected. We evaluate the eff
    
[^458]: 大型视觉语言模型中的跨文化价值观归因

    Cross-Cultural Value Attribution in Large Vision-Language Models

    [https://arxiv.org/abs/2604.09945](https://arxiv.org/abs/2604.09945)

    该论文首次系统研究大型视觉语言模型在道德、伦理和政治价值观判断上如何随图像中人物的文化语境（宗教、国籍、社会经济地位）而变化，通过反事实图像集与多维度评估框架揭示其中的跨文化刻板印象与公平性问题。

    

    近年来，大型视觉语言模型（LVLMs）的快速普及引发了日益增长的公平性担忧，因为它们倾向于强化有害的社会刻板印象。尽管社会偏见背景下的公平性问题已受到广泛关注，但此前相对较少有研究考察LVLMs中与宗教、国籍和社会经济地位等文化语境相关的刻板印象。在本工作中，我们旨在缩小这一研究空白，研究LVLM对个人道德、伦理和政治价值观的判断如何随图像中呈现的不同文化语境而变化。我们使用反事实图像集——即描绘同一个人处于不同文化语境中的图像——对主流LVLMs中的此类价值观判断进行了多维度分析。我们的评估框架结合了描述性分析（道德基础理论分类、词汇分析以及价值观……

    arXiv:2604.09945v3 Announce Type: replace-cross  Abstract: The rapid adoption of large vision-language models (LVLMs) in recent years has been accompanied by growing fairness concerns due to their propensity to reinforce harmful societal stereotypes. While significant attention has been paid to such fairness concerns in the context of social biases, relatively little prior work has examined the presence of stereotypes in LVLMs related to cultural contexts such as religion, nationality, and socioeconomic status. In this work, we aim to narrow this gap by investigating how LVLM judgments about a person's moral, ethical, and political values vary across cultural contexts presented in images. We conduct a multi-dimensional analysis of such value judgments in popular LVLMs using counterfactual image sets, which depict the same person across different cultural contexts. Our evaluation framework pairs descriptive analyses (Moral Foundations Theory categorization, lexical analyses, and value s
    
[^459]: 你的logits知道些什么？

    What do your logits know?

    [https://arxiv.org/abs/2604.09885](https://arxiv.org/abs/2604.09885)

    该论文首次系统比较了视觉-语言模型在不同表示层次（残差流、tuned lens投影、top-k logits）上保留的信息，发现即使是最易访问的top logit值也能泄露图像查询中与任务无关的信息，其泄露量在某些情况下与完整残差流的直接投影相当，揭示了模型内部信息泄露的安全风险。

    

    arXiv:2604.09885v2 公告类型：替换  摘要：近期的研究表明，探测模型内部结构可以揭示大量从模型生成结果中无法察觉的信息。这带来了无意或恶意信息泄露的风险，即模型用户能够获取模型所有者认为无法访问的信息。以视觉-语言模型作为测试平台，我们首次系统性地比较了不同表示层次上所保留的信息——这些信息从残差流中编码的丰富信息出发，经过两个自然瓶颈被逐步压缩：其一是使用tuned lens获得的残差流低维投影，其二是最可能影响模型答案的最终top-k logits。我们证明，即使是由模型top logit值所定义的、最容易被访问的瓶颈，也能够泄露基于图像查询中存在的与任务无关的信息，在某些情况下所泄露的信息量甚至与完整残差流的直接投影相当。

    arXiv:2604.09885v2 Announce Type: replace  Abstract: Recent work has shown that probing model internals can reveal a wealth of information not apparent from the model generations. This poses a risk of unintentional or malicious information leakage, where model users are able to learn information that the model owner assumed was inaccessible. Using vision-language models as a testbed, we present the first systematic comparison of information retained at different representational levels as it is compressed from the rich information encoded in the residual stream through two natural bottlenecks: low-dimensional projections of the residual stream obtained using tuned lens, and the final top-k logits most likely to impact model's answer. We show that even easily accessible bottlenecks defined by the model's top logit values can leak task-irrelevant information present in an image-based query, in some cases revealing as much information as direct projections of the full residual stream.
    
[^460]: 基于Sinkhorn双随机注意力的秩衰减分析

    Sinkhorn doubly stochastic attention rank decay analysis

    [https://arxiv.org/abs/2604.07925](https://arxiv.org/abs/2604.07925)

    本文证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准softmax行随机注意力更能有效保持网络深度中的秩，从而缓解秩崩溃问题。

    

    自注意力机制是Transformer架构取得成功的关键。然而，标准的行随机注意力已被证明在各层之间存在严重的信号退化问题。特别地，它可能引发秩崩溃，导致token表示变得越来越均匀，同时还会引发熵崩溃，其特征是注意力分布高度集中。近期的研究强调了双随机注意力作为一种熵正则化形式的优势，它能够促进更平衡的注意力分布，从而带来更好的实证性能。在本文中，我们研究了跨网络深度的秩崩溃问题，并证明了使用Sinkhorn算法归一化的双随机注意力矩阵比标准的softmax行随机注意力矩阵能更有效地保持秩。正如之前针对softmax所证明的那样，跳跃连接对于缓解秩崩溃至关重要。我们通过实验验证了这一现象。

    arXiv:2604.07925v2 Announce Type: replace-cross  Abstract: The self-attention mechanism is central to the success of Transformer architectures. However, standard row-stochastic attention has been shown to suffer from significant signal degradation across layers. In particular, it can induce rank collapse, resulting in increasingly uniform token representations, as well as entropy collapse, characterized by highly concentrated attention distributions. Recent work has highlighted the benefits of doubly stochastic attention as a form of entropy regularization, promoting a more balanced attention distribution and leading to improved empirical performance. In this paper, we study rank collapse across network depth and show that doubly stochastic attention matrices normalized with Sinkhorn algorithm preserve rank more effectively than standard softmax row-stochastic ones. As previously shown for softmax, skip connections are crucial to mitigate rank collapse. We empirically validate this phe
    
[^461]: 基于噪声样本迭代精炼的神经全局优化方法

    Neural Global Optimization via Iterative Refinement from Noisy Samples

    [https://arxiv.org/abs/2604.03614](https://arxiv.org/abs/2604.03614)

    本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。

    

    从噪声样本中对黑盒函数进行全局优化是机器学习和科学计算中的一项根本性挑战。传统方法如贝叶斯优化在多模态函数上往往收敛到局部极小值，而无梯度方法则需要大量的函数评估。我们提出了一种新颖的神经方法，通过迭代精炼来学习寻找全局极小值。我们的模型以噪声函数样本及其拟合的样条表示作为输入，然后迭代地将初始猜测精炼至真实的全局极小值。该方法在随机生成的函数上进行训练，其全局极小值真值通过穷举搜索获得，在具有挑战性的多模态测试函数上实现了8.05%的平均误差，而样条初始化的误差为36.24%，实现了28.18%的提升。该模型在72%的测试用例中成功找到全局极小值，误差低于10%。

    arXiv:2604.03614v3 Announce Type: replace-cross  Abstract: Global optimization of black-box functions from noisy samples is a fundamental challenge in machine learning and scientific computing. Traditional methods such as Bayesian Optimization often converge to local minima on multi-modal functions, while gradient-free methods require many function evaluations. We present a novel neural approach that learns to find global minima through iterative refinement. Our model takes noisy function samples and their fitted spline representation as input, then iteratively refines an initial guess toward the true global minimum. Trained on randomly generated functions with ground truth global minima obtained via exhaustive search, our method achieves a mean error of 8.05 percent on challenging multi-modal test functions, compared to 36.24 percent for the spline initialization, a 28.18 percent improvement. The model successfully finds global minima in 72 percent of test cases with error below 10 pe
    
[^462]: 面向大语言模型对齐的基于隐式反馈的无偏奖励建模

    Unbiased Reward Modeling from Implicit Feedback for LLM Alignment

    [https://arxiv.org/abs/2603.23184](https://arxiv.org/abs/2603.23184)

    该论文提出ImplicitRM方法，通过将训练样本分层为四个潜在组并构建理论无偏的似然最大化目标，从点击、复制等隐式用户反馈中学习无偏奖励模型，从而解决了隐式反馈缺乏明确负样本和存在选择偏差这两大挑战。

    

    尽管基于人类反馈的强化学习（RLHF）取得了成功，现有的奖励建模方法在很大程度上依赖于显式反馈，而显式反馈的收集成本高昂且难以规模化。本工作研究隐式奖励建模，即从点击、复制和跳过等隐式用户反馈中学习奖励模型。虽然隐式反馈具有可扩展性和成本效益的优势，但它带来了两个关键挑战：其一，它缺乏明确的负样本，使得标准的正负样本分类方法不再适用；其二，它存在选择偏差，即不同回复引发用户反馈的倾向具有异质性，这进一步掩盖了明确的负样本。为了应对这些挑战，我们提出了ImplicitRM，一种能够从隐式反馈中学习无偏奖励模型的方法。该方法利用一个分层模型将训练样本划分为四个潜在组，并推导出一个在理论上无偏的似然最大化目标。

    arXiv:2603.23184v2 Announce Type: replace-cross  Abstract: Despite the success of reinforcement learning from human feedback (RLHF), existing reward modeling methods largely rely on explicit feedback, which is costly to collect and difficult to scale. This work studies implicit reward modeling, learning reward models from implicit user feedback, such as clicks, copies and skips. While scalable and cost-effective, implicit feedback poses two key challenges: It lacks definitive negative samples, which makes standard positive-negative classification methods inapplicable; It suffers from selection bias, where responses have heterogeneous propensities to elicit feedback, which further obscures definitive negative samples. To address these challenges, we propose ImplicitRM, which learns unbiased reward models from implicit feedback. It stratifies training samples into four latent groups using a stratification model and derives a likelihood-maximization objective that is theoretically unbiase
    
[^463]: FSCE：一种面向噪声鲁棒SAR自动目标识别的目标感知频率-空间协同增强框架

    FSCE: A Target-Aware Frequency-Spatial Collaborative Enhancement Framework for Noise-Resilient SAR ATR

    [https://arxiv.org/abs/2603.21565](https://arxiv.org/abs/2603.21565)

    提出了FSCE框架，通过在网络入口处进行频率-空间协同增强以抑制斑点噪声传播、稳定浅层特征，并结合自适应策略驱动的语义对齐机制施加自上而下的语义约束，从而显著提升噪声环境下SAR自动目标识别的鲁棒性。

    

    合成孔径雷达自动目标识别（SAR ATR）受到相干斑点噪声的严重挑战，其干扰会被层次化的非线性变换逐步放大，并最终损害高层语义表示。为解决这一问题，我们提出了一种面向噪声鲁棒SAR ATR的目标感知频率-空间协同增强（FSCE）框架，该框架将用于早期特征稳定的频率-空间建模与语义正则化相结合。具体而言，我们在网络入口处设计了一个频率-空间早期自适应增强（FS-EAE）模块，通过协同的空间-频率建模来抑制噪声传播并保留目标结构。在稳定的浅层表示基础上，我们进一步引入了自适应策略驱动的语义对齐（APSA）机制，该机制利用在线教师策略施加自上而下的语义约束……

    arXiv:2603.21565v2 Announce Type: replace-cross  Abstract: Synthetic aperture radar automatic target recognition (SAR ATR) is severely challenged by coherent speckle noise, whose interference can be progressively amplified by hierarchical nonlinear transformations and eventually damage high-level semantic representations. To address this issue, we propose a Target-Aware Frequency-Spatial Collaborative Enhancement (FSCE) framework for noise-resilient SAR ATR, which integrates frequency-spatial modeling for early feature stabilization with semantic regularization. Specifically, we design a Frequency-Spatial Early-stage Adaptive Enhancement (FS-EAE) module at the network entrance to suppress noise propagation and preserve target structures through collaborative spatial-frequency modeling. Building upon stabilized shallow representation, we further introduce an Adaptive Policy-driven Semantic Alignment (APSA) mechanism, which uses an online teacher policy to impose top-down semantic constr
    
[^464]: 数据异构下移动边缘网络中的联邦混合专家对齐

    Federated Mixture-of-Experts Alignment on Mobile Edge Networks under Data Heterogeneity

    [https://arxiv.org/abs/2603.21276](https://arxiv.org/abs/2603.21276)

    针对数据异构环境下联邦MoE大模型微调中客户端门控偏好分歧和专家语义模糊两大挑战，本文提出了FedAlign-MoE联邦聚合对齐框架，以在移动边缘网络上实现更好的协同训练。

    

    随着移动边缘设备对端侧大语言模型服务需求的不断增长，混合专家架构因其能在有限计算资源下扩展模型容量而被广泛采用。由于基于MoE的大语言模型微调依赖于隐私敏感的本地数据，联邦学习为在不暴露原始数据的情况下进行协同训练提供了一种天然的范式。然而，将基于MoE的大语言模型微调集成到联邦学习中，面临由客户端间数据异构性引发的两个关键挑战：（i）各不相同的本地数据分布使客户端形成不同的门控偏好，因此直接的参数聚合会产生一个“一刀切”却无法适配任何客户端的全局门控网络；（ii）相同索引的专家在不同设备上发展出互异的语义角色，导致专家语义模糊以及专门化能力退化。为应对这些挑战，我们提出了FedAlign-MoE，一种面向……的联邦聚合对齐框架（摘要在此处截断）。

    arXiv:2603.21276v2 Announce Type: replace-cross  Abstract: The growing demand for on-device large language model (LLM) services on mobile edge devices has driven the adoption of Mixture-of-Experts (MoE) architectures, which scale model capacity with limited computation. Since fine-tuning MoE-based LLMs relies on privacy-sensitive local data, federated learning (FL) offers a natural paradigm for collaborative training without exposing raw data. However, integrating MoE-based LLM fine-tuning into FL faces two critical challenges caused by data heterogeneity across clients: (i) divergent local data distributions drive clients to develop distinct gating preferences, so direct parameter aggregation yields a one-size-fits-none global gating network; and (ii) same-indexed experts develop disparate semantic roles across devices, leading to expert semantic blurring and degraded specialization. To address these challenges, we propose FedAlign-MoE, a federated aggregation alignment framework for 
    
[^465]: 理解大语言模型中的道德推理轨迹：迈向基于探测方法的可解释性

    Understanding Moral Reasoning Trajectories in Large Language Models: Toward Probing-Based Explainability

    [https://arxiv.org/abs/2603.16017](https://arxiv.org/abs/2603.16017)

    该论文提出“道德推理轨迹”这一新概念，揭示了大语言模型在道德推理中系统性地在多个伦理框架间切换、且框架切换频繁的轨迹更易受说服性攻击，并通过线性探测和激活引导技术定位并调控模型中的道德框架表征，为LLM道德推理的可解释性研究提供了新路径。

    

    大语言模型（LLM）日益参与到道德敏感的决策之中，然而它们如何在推理过程中组织伦理框架仍然缺乏深入研究。我们提出了“道德推理轨迹”的概念，即在中间推理步骤中调用伦理框架的序列，并在六个模型和三个基准测试上分析其动态特性。我们发现道德推理涉及系统性的多框架权衡：55.4%–57.7%的连续推理步骤涉及框架切换，仅有16.4%–17.8%的轨迹保持框架一致性。不稳定的轨迹受到说服性攻击的影响是稳定轨迹的1.29倍（p=0.015）。在表示层面，线性探测方法将特定框架的编码定位于模型特定的层（Llama-3.3-70B为第63/81层；Qwen2.5-72B为第17/81层），其KL散度比步骤先验基线低16.8%–22.2%。在生成过程中应用的激活引导……（原文摘要在此处被截断）

    arXiv:2603.16017v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly participate in morally sensitive decision-making, yet how they organize ethical frameworks across reasoning steps remains underexplored. We introduce moral reasoning trajectories, sequences of ethical framework invocations across intermediate reasoning steps, and analyze their dynamics across six models and three benchmarks. We find that moral reasoning involves systematic multi-framework deliberation: 55.4--57.7% of consecutive steps involve framework switches, and only 16.4--17.8% of trajectories remain framework-consistent. Unstable trajectories remain 1.29 times more susceptible to persuasive attacks (p=0.015). At the representation level, linear probes localize framework-specific encoding to model-specific layers (layer 63/81 for Llama-3.3-70B; layer 17/81 for Qwen2.5-72B), achieving 16.8--22.2% lower KL divergence than the step-prior baseline. Activation steering applied during ge
    
[^466]: PC-Diffuser：面向基于扩散模型轨迹规划器的路径一致性胶囊CBF安全过滤器

    PC-Diffuser: Path-Consistent Capsule CBF Safety Filtering for Diffusion-Based Trajectory Planner

    [https://arxiv.org/abs/2603.10330](https://arxiv.org/abs/2603.10330)

    PC-Diffuser通过将可认证的路径一致性胶囊控制障碍函数（CBF）安全结构直接嵌入扩散规划器的去噪循环中，使安全性成为轨迹生成的内在属性而非事后修正，从而解决了扩散轨迹规划器难以认证且在罕见场景中可能灾难性失败的问题。

    

    arXiv:2603.10330v3 公告类型： replace-cross 摘要：复杂交通环境中的自动驾驶需要能够超越手工设计规则进行泛化的规划器，这推动了从专家演示中学习行为的数据驱动方法的发展。基于扩散模型的轨迹规划器最近通过迭代去噪全时域规划展现出了强大的闭环性能，但它们仍然难以进行安全认证，并且在罕见或分布外场景中可能出现灾难性失败。为应对这一挑战，我们提出了PC-Diffuser，这是一个安全增强框架，它将可认证的、路径一致的障碍函数结构直接嵌入到扩散规划的去噪循环中。其核心思想是让安全性成为轨迹生成的内在组成部分，而非事后补救：我们在保持扩散模型预期路径几何形状的同时，沿规划 rollout 强制执行前向不变性。具体而言，PC-Diffuser (i) 使用胶囊距离障碍函数评估碰撞风险

    arXiv:2603.10330v3 Announce Type: replace-cross  Abstract: Autonomous driving in complex traffic requires planners that generalize beyond hand-crafted rules, motivating data-driven approaches that learn behavior from expert demonstrations. Diffusion-based trajectory planners have recently shown strong closed-loop performance by iteratively denoising a full-horizon plan, but they remain difficult to certify and can fail catastrophically in rare or out-of-distribution scenarios. To address this challenge, we present PC-Diffuser, a safety augmentation framework that embeds a certifiable, path-consistent barrier-function structure directly into the denoising loop of diffusion planning. The key idea is to make safety an intrinsic part of trajectory generation rather than a post-hoc fix: we enforce forward invariance along the rollout while preserving the diffusion model's intended path geometry. Specifically, PC-Diffuser (i) evaluates collision risk using a capsule-distance barrier function
    
[^467]: 双模态多阶段对抗性安全训练：增强多模态网页代理抵御跨模态攻击的鲁棒性

    Dual-Modality Multi-Stage Adversarial Safety Training: Robustifying Multimodal Web Agents Against Cross-Modal Attacks

    [https://arxiv.org/abs/2603.04364](https://arxiv.org/abs/2603.04364)

    该论文提出双模态多阶段对抗性安全训练（DMAST）框架，将代理与攻击者的交互建模为两人一般和马尔可夫博弈，通过三阶段协同训练显著增强多模态网页代理抵御同时污染视觉与文本两个观察通道的跨模态欺骗攻击的能力。

    

    处理截图和可访问性树的多模态网页代理正日益被部署用于与网页界面交互，然而其双流架构开启了一个尚未被充分探索的攻击面：向网页DOM注入内容的攻击者可以用一致的欺骗性叙事同时破坏两个观察通道。我们在MiniWob++上的漏洞分析表明，包含视觉组件的攻击远胜于纯文本注入，暴露了以文本为中心的视觉语言模型（VLM）安全训练中的关键缺陷。受此发现启发，我们提出了双模态多阶段对抗性安全训练（DMAST），该框架将代理与攻击者的交互形式化为两人一般和马尔可夫博弈，并通过三阶段流程对双方进行协同训练：（1）从强大的教师模型进行模仿学习，（2）采用新颖的零确认策略进行oracle引导的监督微调，以

    arXiv:2603.04364v2 Announce Type: replace-cross  Abstract: Multimodal web agents that process both screenshots and accessibility trees are increasingly deployed to interact with web interfaces, yet their dual-stream architecture opens an underexplored attack surface: an adversary who injects content into the webpage DOM simultaneously corrupts both observation channels with a consistent deceptive narrative. Our vulnerability analysis on MiniWob++ reveals that attacks including a visual component far outperform text-only injections, exposing critical gaps in text-centric VLM safety training. Motivated by this finding, we propose Dual-Modality Multi-Stage Adversarial Safety Training (DMAST), a framework that formalizes the agent-attacker interaction as a two-player general-sum Markov game and co-trains both players through a three-stage pipeline: (1) imitation learning from a strong teacher model, (2) oracle-guided supervised fine-tuning that uses a novel zero-acknowledgment strategy to 
    
[^468]: 无需世界模型的世界属性：分布性关联与语言模型解码结果的解读

    World Properties without World Models: Distributional Associations and the Interpretation of Decoding Results from Language Models

    [https://arxiv.org/abs/2603.04317](https://arxiv.org/abs/2603.04317)

    该研究表明，以往从语言模型激活中解码出的“世界属性”在很大程度上也能由静态词嵌入实现，因此解码成功并不能证明语言模型形成了内部世界模型，而可能仅反映了语料库统计中的分布性关联。

    

    越来越多的文献表明，可以从大语言模型（LLM）的激活中线性解码出各种变量，涵盖世界属性（如城市位置和历史人物的寿命）以及情绪和疼痛等。这些发现常被视为语言模型超越表面文本统计、形成内部世界模型的证据。我们证明，对相同或匹配的刺激，静态词嵌入（从语料库统计中学到的固定的、与上下文无关的表示）能够支持大部分相同的解码。在四个已发表的案例中（地点、时间、疼痛和情绪），静态向量可以预测坐标和死亡年份（R² = 0.42-0.59），将疼痛句子与匹配的对照句子区分开（留出AUC为0.85-0.88），并在刻意避免直接点名情绪的故事中分类十二种情绪（AUC为0.84-0.88）。由于静态词嵌入为每个词分配单一的、与上下文无关的向量，

    arXiv:2603.04317v2 Announce Type: replace-cross  Abstract: A growing literature shows that variables can be linearly decoded from the activations of large language models (LLMs). These range from properties of the world, such as the locations of cities and the lifetimes of historical figures, to emotions and pain. Such findings are often taken as evidence that language models go beyond surface text statistics and form internal models of the world. We show that static word embeddings (fixed, context-insensitive representations learned from corpus statistics) of the same or matched stimuli support much of the same decoding. Across four published cases (place, time, pain and emotion), static vectors predict coordinates and year of death (R^2 = 0.42-0.59), separate pain from matched control sentences (held-out AUC 0.85-0.88), and classify twelve emotions in stories written to avoid naming them (AUC 0.84-0.88). Because static embeddings assign each word a single, context-independent vector,
    
[^469]: 基于多模态大语言模型的实时游戏视频解说生成：停顿感知解码方法

    Real-Time Generation of Game Video Commentary with Multimodal LLMs: Pause-Aware Decoding Approaches

    [https://arxiv.org/abs/2603.02655](https://arxiv.org/abs/2603.02655)

    提出两种无需微调的基于提示的停顿感知解码策略（固定间隔与根据话语估计时长动态调整间隔），使多模态大语言模型能够生成语义相关且时机恰当的实时游戏视频解说。

    

    实时视频解说生成为视频中正在发生的事件提供文本描述，可用于提升体育、电子竞技和直播等领域的无障碍性与观众参与度。解说生成涉及两个关键决策：说什么以及何时说。尽管近期基于提示词的多模态大语言模型（MLLM）方法在内容生成方面表现出色，但它们在很大程度上忽视了时机问题。我们研究了仅依靠上下文提示是否能够支持语义相关且时机恰当的实时解说生成。我们提出了两种基于提示的解码策略：1）固定间隔方法；2）一种新颖的基于动态间隔的解码方法，该方法根据前一句话的估计时长来调整下一次预测的时机。这两种方法都无需任何微调即可实现停顿感知的生成。在日本语和英（语数据集上的实验……）

    arXiv:2603.02655v2 Announce Type: replace-cross  Abstract: Real-time video commentary generation provides textual descriptions of ongoing events in videos. It supports accessibility and engagement in domains such as sports, esports, and livestreaming. Commentary generation involves two essential decisions: what to say and when to say it. While recent prompting-based approaches using multimodal large language models (MLLMs) have shown strong performance in content generation, they largely ignore the timing aspect. We investigate whether in-context prompting alone can support real-time commentary generation that is both semantically relevant and well-timed. We propose two prompting-based decoding strategies: 1) a fixed-interval approach, and 2) a novel dynamic interval-based decoding approach that adjusts the next prediction timing based on the estimated duration of the previous utterance. Both methods enable pause-aware generation without any fine-tuning. Experiments on Japanese and Eng
    
[^470]: 面向乳腺超声联合分割与分类的自适应双向任务交互

    Adaptive Bidirectional Task Interaction for Joint Segmentation and Classification of Breast Ultrasound

    [https://arxiv.org/abs/2603.01295](https://arxiv.org/abs/2603.01295)

    该论文提出在解码阶段通过任务交互模块（TIM）与自适应交互加权（AIW）实现分割与分类分支间逐图像自适应的双向信息交换，从而在乳腺超声联合分割与分类任务上取得了优于现有方法的性能。

    

    乳腺超声中的病灶分割与组织分类通常采用共享编码器进行训练，因此一旦两个分支的解码器分离，二者之间便停止了信息交换，而这恰恰是边界细节与语义证据最具互补性的阶段。所提出的方法在解码过程中恢复了这种信息交换，并且由于信息交换的价值因图像而异，网络可以针对每张图像自行决定保留多少交互信息。在每个解码器的四个层级上设置任务交互模块（TIM），将池化后的边界上下文传入分类表示，并利用类别条件先验对解码器通道进行调制。随后，自适应交互加权单元根据针对每张图像和每个层级计算出的系数，将交互后的特征与原始特征进行融合。在BUSI数据集上，该模型达到了74.19%的IoU和90.60%的准确率；在BUSI-WHU数据集上达到了86.40%的IoU和95.00%的准确率，优于共享编码器的多任务模型及transformer分割方法。

    arXiv:2603.01295v2 Announce Type: replace-cross  Abstract: Joint lesion segmentation and tissue classification in breast ultrasound are usually trained with a shared encoder, so the two branches stop exchanging information once their decoders separate. That is exactly where boundary detail and semantic evidence are most complementary. The proposed method restores this exchange during decoding and, because its value differs between images, lets the network decide per image how much to keep. A Task Interaction Module (TIM) at each of four decoder levels passes pooled boundary context into the classification representation and modulates decoder channels with class-conditioned priors. An Adaptive Interaction Weighting (AIW) unit then blends interacted and original features with a coefficient computed for each image and level. On BUSI the model reaches 74.19% IoU and 90.60% accuracy, and on BUSI-WHU 86.40% IoU and 95.00% accuracy, ahead of encoder-sharing multi-task, transformer segmentatio
    
[^471]: SimToolReal：一种面向零样本灵巧工具操作的对象中心策略

    SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation

    [https://arxiv.org/abs/2602.16863](https://arxiv.org/abs/2602.16863)

    SimToolReal通过程序化生成多样化的工具状物体并以将其操作至随机目标位姿的通用目标训练单一从仿真到现实强化学习策略，实现了无需针对特定物体或任务进行工程设计的零样本灵巧工具操作。

    

    操作工具的能力显著扩展了机器人可执行的任务范围。然而，工具操作代表了一类极具挑战性的灵巧任务，需要抓取细薄物体、进行手中物体旋转以及施加有力交互。由于为这些行为收集遥操作数据十分困难，从仿真到现实的强化学习（RL）成为一种有前景的替代方案。然而，先前的方法通常需要大量的工程工作来对物体建模并为每个任务调整奖励函数。在本工作中，我们提出了SimToolReal，朝着推广用于工具操作的从仿真到现实RL策略迈出了一步。不同于聚焦于单一物体和任务，我们在仿真中程序化地生成大量工具状物体基元，并以将每个物体操作至随机目标位姿这一通用目标来训练单一RL策略。这种方法使SimToolReal能够执行通用的灵巧工具操作任务。

    arXiv:2602.16863v3 Announce Type: replace-cross  Abstract: The ability to manipulate tools significantly expands the set of tasks a robot can perform. Yet, tool manipulation represents a challenging class of dexterity, requiring grasping thin objects, in-hand object rotations, and forceful interactions. Since collecting teleoperation data for these behaviors is challenging, sim-to-real reinforcement learning (RL) is a promising alternative. However, prior approaches typically require substantial engineering effort to model objects and tune reward functions for each task. In this work, we propose SimToolReal, taking a step towards generalizing sim-to-real RL policies for tool manipulation. Instead of focusing on a single object and task, we procedurally generate a large variety of tool-like object primitives in simulation and train a single RL policy with the universal goal of manipulating each object to random goal poses. This approach enables SimToolReal to perform general dexterous t
    
[^472]: ReLoop：面向可靠的大语言模型优化的结构化建模与行为验证

    ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization

    [https://arxiv.org/abs/2602.15983](https://arxiv.org/abs/2602.15983)

    ReLoop通过结合结构化生成和行为验证，有效缩小了大语言模型在优化代码生成中的可行性与正确性差距。

    

    大语言模型（LLMs）可以将自然语言转化为优化代码，但静默失败构成关键风险：能够执行并返回求解器可行解的代码可能编码了语义上错误的公式——这种可行性与正确性之间的差距在组合问题上高达90个百分点。我们引入了ReLoop，通过两种互补机制来解决这一差距。结构化生成将代码生产分解为四阶段推理链（理解、形式化、综合、验证），从源头防止公式错误。行为验证通过测试公式是否对基于求解器的参数扰动做出正确响应来检测生成过程中存活的错误——这是一种绕过LLM自我审查且无需真实标签的外部语义信号。这两种机制在错误结构上互补：结构化生成在组合问题上带来最大改进。

    arXiv:2602.15983v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations---a feasibility--correctness gap reaching 90 percentage points on compositional problems. We introduce ReLoop, which addresses this gap through two complementary mechanisms. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify), preventing formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation---an external semantic signal that bypasses LLM self-review and requires no ground truth. The two mechanisms are complementary by error structure: structured generation drives the largest gains on compositi
    
[^473]: UniST-Pred：面向中断情形下交通网络时空交通预测的鲁棒统一框架

    UniST-Pred: A Robust Unified Framework for Spatio-Temporal Traffic Forecasting in Transportation Networks Under Disruptions

    [https://arxiv.org/abs/2602.14049](https://arxiv.org/abs/2602.14049)

    提出了UniST-Pred统一时空预测框架，通过解耦时间建模与空间表示学习并采用自适应表示级融合，在交通网络中断等结构与观测不确定性条件下实现鲁棒的交通预测。

    

    时空交通预测是智能交通系统的核心组成部分，支持信号控制和网络级交通管理等多种下游任务。在实际部署中，预测模型必须在结构和观测不确定性条件下运行，而这些条件在模型设计中很少被考虑。近期的方法通过紧密耦合空间与时间建模实现了较强的短期预测性能，但往往以增加复杂性和模块化受限为代价。相比之下，高效的时间序列模型无需依赖显式网络结构即可捕捉长期时间依赖性。我们提出了UniST-Pred，一个统一的时空预测框架，它首先将时间建模与空间表示学习解耦，然后通过自适应表示级融合将两者整合。为评估所提方法的鲁棒性，我们构……（原文摘要在此处被截断）

    arXiv:2602.14049v2 Announce Type: replace-cross  Abstract: Spatio-temporal traffic forecasting is a core component of intelligent transportation systems, supporting various downstream tasks such as signal control and network-level traffic management. In real-world deployments, forecasting models must operate under structural and observational uncertainties, conditions that are rarely considered in model design. Recent approaches achieve strong short-term predictive performance by tightly coupling spatial and temporal modeling, often at the cost of increased complexity and limited modularity. In contrast, efficient time-series models capture long-range temporal dependencies without relying on explicit network structure. We propose UniST-Pred, a unified spatio-temporal forecasting framework that first decouples temporal modeling from spatial representation learning, then integrates both through adaptive representation-level fusion. To assess robustness of the proposed approach, we constr
    
[^474]: 学习配置智能体AI系统

    Learning to Configure Agentic AI Systems

    [https://arxiv.org/abs/2602.11574](https://arxiv.org/abs/2602.11574)

    该论文将LLM智能体系统的配置问题建模为半马尔可夫决策过程，提出轻量级分层策略ARC，能够根据查询难度动态选择最优配置，使推理准确率提升31.3%、工具使用准确率提升13.95%。

    

    配置基于大语言模型（LLM）的智能体系统需要从庞大的组合设计空间中选择工作流、工具、token预算和提示词，目前通常由固定模板或手工调优的启发式方法来处理，这些方法无论查询难度如何都采用相同的配置，导致行为脆弱和计算资源浪费。为解决这一问题，我们将智能体配置形式化为半马尔可夫决策过程（SMDP），其中每个配置作为一个时间上扩展的选项，决定智能体系统如何处理查询，并引入了ARC（Agentic Resource & Configuration learner，智能体资源与配置学习器），这是一个轻量级的分层策略，能够动态选择针对特定查询的智能体配置。在推理、工具使用和智能体基准测试中，ARC持续超越预算匹配的工具增强型LLM，平均推理准确率提升31.3%，工具使用准确率提升13.95%，并将τ-Bench（Airline）的Pass^1成功率翻倍（原文截断）。

    arXiv:2602.11574v4 Announce Type: replace  Abstract: Configuring LLM-based agent systems involves choosing workflows, tools, token budgets, and prompts from a large combinatorial design space, and is typically handled today by fixed templates or hand-tuned heuristics that apply the same configuration regardless of query difficulty, leading to brittle behavior and wasted compute. To address this, we formulate agent configuration as a semi-Markov decision process (SMDP) where each configuration acts as a temporally extended option that determines how an agent system processes a query, and introduce introduce ARC (Agentic Resource & Configuration learner), a lightweight hierarchical policy that dynamically selects query-specific agent configurations. Across reasoning, tool-use, and agentic benchmarks, ARC consistently improves over budget-matched tool-augmented LLMs, increasing average reasoning accuracy by 31.3%, tool-use accuracy by 13.95%, and doubling {\tau}-Bench (Airline) Pass^1 suc
    
[^475]: X平台推荐系统在无意中对用户的意识形态立场进行画像

    Recommender system in X inadvertently profiles ideological positions of users

    [https://arxiv.org/abs/2602.02624](https://arxiv.org/abs/2602.02624)

    研究发现X平台的推荐系统在优化相关性时会无意中将用户的左右政治意识形态立场编码进其嵌入表示中，而移除该意识形态维度可在准确性损失有限的情况下实现推荐多样化。

    

    多项数据保护法律限制揭示政治观点的数据处理行为，无论数据控制者的意图如何。推荐系统是否会在优化相关性时作为副产品而做到这一点，此前尚未被测量。基于在法国向682名志愿者展示的250万条“推荐关注”建议，我们重构了X推荐系统用于26,509个账号的嵌入向量的近似版本，并计算了经调查校准的意识形态得分。该嵌入中的一个方向按左-右政治立场对用户进行排序（Pearson rho = 0.887），且不同于追踪年龄、性别或受欢迎程度的方向。我们证明这一维度确实存在，并影响基于该嵌入计算出的推荐结果。移除该维度可以增加推荐的多样性，而准确性的损失有限。我们记录了一种重要的、涌现的政治表征形式，而当前关于画像的定义尚未明确涵盖这种情况。

    arXiv:2602.02624v2 Announce Type: replace-cross  Abstract: Several data protection laws restrict processing that reveals political opinions, irrespective of the controller's intent. Whether recommender systems do so as a by-product of optimizing relevance has not been measured. From 2.5 million ``Who to Follow'' recommendations shown to 682 volunteers in France, we reconstructed an approximation of the embedding used by X's recommender for 26,509 accounts, computing survey-calibrated ideology scores. One direction in this embedding orders users by Left-Right position (Pearson rho = 0.887), distinct from directions tracking age, gender or popularity. We show this scale exists and affects the recommendations computed from the embedding. Removing it diversified recommendations at a limited cost in accuracy. We document a consequential form of emergent political representation that current definitions of profiling do not clearly address.
    
[^476]: 面向鲁棒与非光滑凸分布式学习的快速高效异步Gossip算法

    Fast and Efficient Asynchronous Gossip Algorithm for Robust and Non-Smooth Convex Decentralized Learning

    [https://arxiv.org/abs/2601.20571](https://arxiv.org/abs/2601.20571)

    本文提出Goal-PD，一种异步Gossip原始-对偶算法，每个节点仅维护两个变量而与网络度数无关，实现了几乎必然收敛与线性收敛，并通过分布式均值估计中的成对平均特例与经典Gossip算法建立了直接联系。

    

    面向分布式非光滑凸优化的异步原始-对偶方法通常要求每个节点维护 $\mathcal{O}(d)$ 个辅助变量，其中 $d$ 为该节点的度数。这种对度数的依赖增加了内存需求，并可能放大过期信息的影响，尤其是在密集网络中。受分布式学习中节约内存管理这一挑战的启发，我们提出了 Goal-PD，一种基于异步Gossip的原始-对偶算法，无论节点度数如何，每个节点仅需维护两个变量。我们建立了Goal-PD几乎必然收敛到所研究优化问题最小化子的结论，并证明了当目标函数为分段线性二次函数时的线性收敛性。对于分布式均值估计，我们证明成对平均是Goal-PD的一个特例，这在所提出的原始-对偶框架与经典Gossip算法之间建立了直接联系。

    arXiv:2601.20571v3 Announce Type: replace-cross  Abstract: Asynchronous primal-dual methods for decentralized non-smooth convex optimization often require each node to maintain $\mathcal{O}(d)$ auxiliary variables, where $d$ is its degree. This dependence on degree increases memory requirements and can amplify the effects of stale information, especially in dense networks. Motivated by the challenge of frugal memory management in decentralized learning, we introduce Goal-PD, an asynchronous gossip-based primal-dual algorithm that maintains only two variables per node, regardless of the node's degree. We establish almost-sure convergence of Goal-PD to a minimizer of the underlying optimization problem, and prove linear convergence when the objective functions are piecewise linear-quadratic. For decentralized mean estimation, we show that pairwise averaging is a special case of Goal-PD, which establishes a direct link between the proposed primal-dual framework and classical gossip. Exper
    
[^477]: 基于混合整数优化的交叉性公平

    Intersectional Fairness via Mixed-Integer Optimization

    [https://arxiv.org/abs/2601.19595](https://arxiv.org/abs/2601.19595)

    本文提出一个基于混合整数优化（MIO）的统一框架，训练同时具备交叉公平性和内在可解释性的分类器，证明了两种交叉公平性度量（MSD 与 SPSF）在检测最不公平子群体上的等价性，并能将交叉偏见有效控制在可接受阈值以下。

    

    在金融和医疗等高风险领域部署人工智能，需要既公平又透明的模型。虽然包括欧盟《人工智能法案》在内的监管框架要求减轻偏见，但它们对偏见的定义故意保持模糊。与现有研究一致，我们认为真正的公平需要在受保护群体的交叉点上解决偏见问题。我们提出了一个统一框架，利用混合整数优化（MIO）来训练具有交叉公平性和内在可解释性的分类器。我们证明了两种交叉公平性度量（MSD 和 SPSF）在检测最不公平子群体方面的等价性，并通过实验证明我们基于 MIO 的算法在发现偏见方面提升了性能。我们训练了高性能、可解释的分类器，将交叉偏见限制在可接受的阈值以下，为监管合规提供了稳健的解决方案。

    arXiv:2601.19595v2 Announce Type: replace-cross  Abstract: The deployment of Artificial Intelligence in high-risk domains, such as finance and healthcare, necessitates models that are both fair and transparent. While regulatory frameworks, including the EU's AI Act, mandate bias mitigation, they are deliberately vague about the definition of bias. In line with existing research, we argue that true fairness requires addressing bias at the intersections of protected groups. We propose a unified framework that leverages Mixed-Integer Optimization (MIO) to train intersectionally fair and intrinsically interpretable classifiers. We prove the equivalence of two measures of intersectional fairness (MSD and SPSF) in detecting the most unfair subgroup and empirically demonstrate that our MIO-based algorithm improves performance in finding bias. We train high-performing, interpretable classifiers that bound intersectional bias below an acceptable threshold, offering a robust solution for regulat
    
[^478]: 面向多语言语言模型的跨语言激活导向方法

    Cross-Lingual Activation Steering for Multilingual Language Models

    [https://arxiv.org/abs/2601.16390](https://arxiv.org/abs/2601.16390)

    提出无需训练的推理时干预方法CLAS，通过有选择地调节神经元激活提升非主导语言的性能，且不损害高资源语言表现。

    

    大型语言模型展现出强大的多语言能力，然而主导语言与非主导语言之间仍然存在显著的性能差距。先前的研究将这一差距归因于多语言表示中共享神经元与特定语言神经元之间的不平衡。我们提出了跨语言激活导向，这是一种无需训练的推理时干预方法，可以有选择地调节神经元激活。我们在分类和生成基准上评估了CLAS，分别取得了2.3%（准确率）和3.4%（F1）的平均提升，同时保持了高资源语言的性能。我们发现有效的迁移是通过功能分化而非严格对齐来实现的；性能提升与语言簇分离度的增加相关。我们的结果表明，有针对性的激活导向可以在不修改模型权重的情况下，释放现有模型中潜在的多语言能力。

    arXiv:2601.16390v2 Announce Type: replace-cross  Abstract: Large language models exhibit strong multilingual capabilities, yet significant performance gaps persist between dominant and non-dominant languages. Prior work attributes this gap to imbalances between shared and language-specific neurons in multilingual representations. We propose Cross-Lingual Activation Steering (CLAS), a training-free inference-time intervention that selectively modulates neuron activations. We evaluate CLAS on classification and generation benchmarks, achieving average improvements of 2.3% (Acc.) and 3.4% (F1) respectively, while maintaining high-resource language performance. We discover that effective transfer operates through functional divergence rather than strict alignment; performance gains correlate with increased language cluster separation. Our results demonstrate that targeted activation steering can unlock latent multilingual capacity in existing models without modification to model weights.
    
[^479]: 使用时间灵活性预计算多智能体路径重规划

    Precomputing Multi-Agent Path Replanning Using Temporal Flexibility

    [https://arxiv.org/abs/2601.04884](https://arxiv.org/abs/2601.04884)

    提出FlexSIPP算法，通过预计算并利用其他智能体的时间灵活性，高效地对单个延迟智能体进行路径重规划，同时避免引发连锁延迟。

    

    当某个智能体发生延迟时，执行多智能体计划会变得具有挑战性，因为这通常会与其他智能体产生冲突，因此我们需要快速找到一个新的安全计划。仅对延迟的智能体进行重规划往往无法产生高效的计划，有时甚至无法得到可行计划；另一方面，对其他智能体进行重规划可能导致一系列连锁变化和延迟，且计算成本高昂。我们展示了如何通过跟踪和利用其他智能体的时间灵活性来高效地对单个延迟智能体进行重规划，同时避免连锁延迟。这种灵活性是指该智能体在不改变与初始延迟智能体之外的其他智能体的先后顺序、或进一步延迟其他智能体的前提下所能承受的最大延迟。我们的算法FlexSIPP预先计算出延迟智能体的所有可能计划，并在给定场景下返回对其他智能体的调整。我们在真实世界中验证了该方法的有效性。

    arXiv:2601.04884v4 Announce Type: replace  Abstract: Executing a multi-agent plan can be challenging when an agent is delayed, because this typically creates conflicts with other agents. So, we need to quickly find a new safe plan. Replanning only the delayed agent often does not yield an efficient plan, and sometimes cannot even yield a feasible one. On the other hand, replanning other agents may lead to a cascade of changes and delays, and it is computationally expensive. We show how to efficiently replan a single delayed agent by tracking and using the temporal flexibility of other agents while avoiding cascading delays. This flexibility is the maximum delay that the agent can take without changing the order with agents other than the initially delayed agent, or further delaying other agents. Our algorithm, FlexSIPP, precomputes all possible plans for the delayed agent and returns the changes to the other agents within the given scenario. We demonstrate our method in a real-world ca
    
[^480]: MiniScope：基于最小权限原则的智能体授权

    MiniScope: Authorizing Agents with Least-Privilege Permissions

    [https://arxiv.org/abs/2512.11147](https://arxiv.org/abs/2512.11147)

    提出了一种以任务为中心的分层权限模型及MiniScope端到端权限系统，将智能体视为特定任务角色中的委托代理，自动发现权限层级并在运行时执行上下文最小权限原则，在保证安全性的同时将权限确认次数减少43.4%-89.4%。

    

    AI智能体正日益被赋予对敏感用户数据和第三方服务的自主访问权限，这使得有效的权限管理成为一项关键的安全挑战。然而，现有的权限模型通常依赖于扁平化的权限结构，难以在安全性与可用性之间取得平衡：细粒度的确认会导致用户疲劳，而粗粒度或持久化的授权则会导致智能体权限过大。为解决这一权衡问题，我们提出了一种以任务为中心的分层权限模型，该模型将智能体视为在特定任务角色内运行的委托代理，而非要求对每次工具调用都进行单独的权限决策。基于该模型，我们提出了MiniScope——一个面向智能体的端到端权限系统，它能够自动发现权限层级结构，并在运行时强制执行上下文相关的最小权限原则。我们的评估表明，MiniScope在模拟场景中将权限确认次数减少了43.4%-89.4%。

    arXiv:2512.11147v2 Announce Type: replace-cross  Abstract: AI agents are increasingly granted autonomous access to sensitive user data and third-party services, making effective permission management a critical security challenge. Existing permission models, however, typically rely on flat permission structures that fail to balance security with usability: fine-grained confirmation induces user fatigue, while coarse-grained or persistent approval leads to overprivileged agents. To address this tradeoff, we propose a task-centric, hierarchical permission model that treats an agent as a delegate operating within a task-specific role instead of requiring a separate permission decision for every tool call. Building on this model, we present MiniScope, an end-to-end permission system for agents that automates permission-hierarchy discovery and enforces contextual least privilege at runtime. Our evaluation shows that MiniScope reduces simulated permission confirmations by 43.4%-89.4% for cau
    
[^481]: 使用平衡微调将大语言模型与生物医学知识对齐

    Aligning LLMs with Biomedical Knowledge using Balanced Fine-Tuning

    [https://arxiv.org/abs/2511.21075](https://arxiv.org/abs/2511.21075)

    本文发现生物医学文本具有与通用文本截然不同的密集认知不确定性结构，并据此提出双尺度的平衡微调方法，通过词元重加权与序列级重新分配使模型聚焦知识密集样本，在多项医学与生物任务上取得比SFT和DFT更一致的性能提升。

    

    工程化大语言模型以加速生命科学研究，需要与生物医学知识进行稳健的对齐。我们观察到，生物医学文本表现出与通用文本根本不同的不确定性结构：密集的低置信度片段编码的是认知性知识缺口（如密集的因果链、罕见实体），而非通用文本中典型的稀疏偶然性风格变化。基于这一发现，我们提出了平衡微调，这是一种双尺度的后训练方法，将组归一化的词元重加权与序列级的重新分配相结合，使训练资源向表现出密集认知不确定性的知识密集样本倾斜。在医学评估、生物推理、稀疏奖励强化学习和生物表征任务中，在相同的训练设置下，BFT比SFT和DFT提供了更一致的性能提升。当替换GeneAgent（GPT-4o）和VCWorld（Gemini-2.5-Flash）中默认的闭源骨干模型时，BFT-（原文摘要在此处截断）

    arXiv:2511.21075v4 Announce Type: replace-cross  Abstract: Engineering LLMs to accelerate life sciences research requires a robust alignment with biomedical knowledge. We observe that biomedical text exhibits a fundamentally different uncertainty structure from general text: dense low-confidence runs encode epistemic knowledge gaps (dense causal chains, rare entities) rather than the sparse aleatoric stylistic variation typical of general text. Based on this discovery, we propose Balanced Fine-Tuning (BFT), a dual-scale post-training method that combines group-normalized token reweighting with sequence-level reallocation toward knowledge-dense samples exhibiting dense epistemic uncertainty. Across medical evaluation, biological reasoning, sparse-reward RL, and biological representation tasks, BFT provides more consistent gains than SFT and DFT under a shared training setup. When replacing the default closed-source backbones in GeneAgent (GPT-4o) and VCWorld (Gemini-2.5-Flash), the BFT-
    
[^482]: 通过轮次级重要性采样与截断触发归一化稳定长时程LLM智能体的离策略训练

    Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization

    [https://arxiv.org/abs/2511.20718](https://arxiv.org/abs/2511.20718)

    该论文提出SORL方法，通过轮次级重要性采样与截断触发的归一化机制，使策略优化与多轮交互结构对齐并抑制不可靠的离策略梯度更新，从而稳定长时程LLM智能体的离策略强化学习训练，防止性能崩溃。

    

    诸如PPO和GRPO等强化学习（RL）算法被广泛用于训练大语言模型（LLM）以完成多轮智能体任务。然而，在离策略训练流程中，这些方法可能表现出不稳定的优化动态，并容易发生性能崩溃。通过实证分析，我们识别出该设置下两个根本性的不稳定性来源：（1）token级策略优化与轮次结构化交互之间的粒度不匹配；（2）由离策略重要性采样和不准确优势估计所引起的高方差、不可靠的梯度更新。为应对这些挑战，我们提出了SORL（Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Training，面向长时程智能体训练的稳定离策略强化学习方法）。SORL引入了使策略优化与多轮交互结构对齐的机制，并自适应地抑制不可靠的离策略更新，从而带来更加保守和稳健的优化过程。

    arXiv:2511.20718v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods can exhibit unstable optimization dynamics and are prone to perfor- mance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1) a granularity mismatch between token-level policy optimization and turn- structured interactions, and (2) high-variance and unreliable gradient updates induced by off- policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, Stabilizing Off-Policy Reinforcement Learning for Long-Horizon Agent Train- ing. SORL introduces mechanisms that align policy optimization with the structure of multi- turn interactions and adaptively suppress unreliable off-policy updates, yielding more conserva- tive and robu
    
[^483]: FRAGMENTA：面向小数据情境下药物先导化合物优化的高效端到端片段化生成模型与智能体调优方法

    FRAGMENTA: Efficient End-to-end Fragmentation-based Generative Model with Agentic Tuning for Drug Lead Optimization in Small Data Regime

    [https://arxiv.org/abs/2511.20510](https://arxiv.org/abs/2511.20510)

    FRAGMENTA提出了一种端到端框架，通过LVSEF片段生成器联合优化片段化与生成过程，并利用智能体系统将对话式专家反馈自动转化为生成目标，在小数据药物先导化合物优化任务中取得了优异表现。

    

    从极其有限的训练数据中生成分子是药物发现领域的一项关键挑战。现有的基于片段的方法在此情境下比基于原子的方法更为适用，但通常将片段选择与下游生成分开优化。在数据有限的情况下，专家反馈尤为宝贵，然而将此类反馈转化为模型目标通常需要AI工程专业知识。我们提出了FRAGMENTA，一个面向小数据药物先导化合物优化的端到端框架，包含两个组件：（1）LVSEF，一种基于片段的生成器，通过表格化奖励更新机制联合优化片段化与生成过程；（2）一个智能体系统，可将对话式专家反馈转化为更新的生成目标。在三个小数据集（11-104个分子）上，LVSEF在最小数据设置下优于最先进的方法，在较大规模数据上与其表现相当，并且训练

    arXiv:2511.20510v3 Announce Type: replace  Abstract: Molecule generation from extremely limited training data is a key challenge in drug discovery. Existing fragment-based methods are more suitable than atom-based approaches in this regime, but typically optimize fragment selection separately from downstream generation. Expert feedback is also especially valuable with limited data, yet translating such feedback into model objectives usually requires AI engineering expertise. We introduce FRAGMENTA, an end-to-end framework for small-data drug lead optimization with two components: (1) LVSEF, a fragment-based generator that jointly optimizes fragmentation and generation through a tabular reward-update mechanism, and (2) an agentic system that converts conversational expert feedback into updated generative objectives. Across three small-data datasets (11--104 molecules), LVSEF outperforms state-of-the-art methods in the smallest-data settings, matches them at larger scales, and trains ${\
    
[^484]: 思维宇宙：大型语言模型中创造性推理的计算框架

    Universe of Thoughts: A Computational Framework for Creative Reasoning in Large Language Models

    [https://arxiv.org/abs/2511.20471](https://arxiv.org/abs/2511.20471)

    该论文受认知科学启发，首次将组合型、探索型和变革型创造力形式化为LLM上可执行的计算算子，并据此构建了“思维宇宙”这一创造性推理框架。

    

    arXiv:2511.20471v3 公告类型：替换 摘要：大型语言模型（LLM）推理的最新进展改善了传统的问题求解能力，但创造性推理仍相对缺乏探索。受认知科学的启发，我们将组合型、探索型和变革型创造力形式化为作用于结构化问题空间与解空间的可执行计算算子，明确了每种模式如何对这些空间进行组合、探索或变换。组合推理将想法跨领域迁移以形成新颖的组合；探索推理在既有概念空间内搜索新的解决方案；而变革推理则修改定义该空间的规则或约束。这一形式化产生了独特的算法过程，我们将其实现于“思维宇宙”这一LLM推理框架中。现有的创造力基准测试要么侧重开放式构思，要么侧重高度受限的问题求解，因此我们引入……

    arXiv:2511.20471v3 Announce Type: replace  Abstract: Recent advances in Large Language Model (LLM) reasoning have improved conventional problem solving, but creative reasoning remains comparatively underexplored. Inspired by cognitive science, we formalize combinational, exploratory, and transformational creativity as executable computational operators over structured problem and solution spaces, specifying how each mode combines, explores, or transforms those spaces. Combinational reasoning transfers ideas across domains to form unfamiliar combinations; exploratory reasoning searches for new solutions within an existing conceptual space; and transformational reasoning modifies the rules or constraints that define that space. This formalization yields distinct algorithmic procedures, which we instantiate in Universe of Thoughts (UoT), an LLM reasoning framework. Existing creativity benchmarks emphasize either open-ended ideation or highly constrained problem solving. We therefore intro
    
[^485]: 面向多步时间序列预测模型训练的二次型直接预测方法

    Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models

    [https://arxiv.org/abs/2511.00053](https://arxiv.org/abs/2511.00053)

    该论文提出了一种新颖的二次型加权学习目标，通过加权矩阵的非对角元素捕捉未来步骤间的标签自相关效应，同时利用非均匀对角元素为不同预测步骤设置异构任务权重，从而同时解决传统均方误差目标的两个缺陷，提升多步时间序列预测模型的训练效果。

    

    arXiv:2511.00053v2 公告类型： replace-cross 摘要：学习目标的设计是训练时间序列预测模型的核心。现有的学习目标（如均方误差）大多将每个未来预测步骤视为独立的、等权重的任务，这导致了以下两个挑战：（1）它们忽视了未来步骤之间的标签自相关效应，导致学习目标存在偏差；（2）它们未能为对应不同未来预测步骤的各项预测任务设置异构的任务权重，从而限制了预测性能。为填补这一空白，我们提出了一种新颖的二次型加权学习目标，能够同时解决上述两个问题。具体而言，加权矩阵的非对角元素用于刻画未来步骤之间的标签自相关效应，而非均匀的对角元素则用于匹配具有不同预测步骤的各项预测任务所偏好的权重。在此基础上，我们提出了二次型直接预测（Quadratic Direct Forecast）方法……

    arXiv:2511.00053v2 Announce Type: replace-cross  Abstract: The design of learning objectives is central to training time-series forecasting models. Existing learning objectives such as mean squared error mostly treat each future step as an independent, equally weighted task, which leads to the following two challenges: (1) they overlook the label autocorrelation effect among future steps, leading to biased learning objectives; (2) they fail to set heterogeneous task weights for different forecasting tasks corresponding to varying future steps, limiting the forecasting performance. To fill this gap, we propose a novel quadratic-form weighted learning objective, addressing both issues simultaneously. Specifically, the off-diagonal elements of the weighting matrix account for the label autocorrelation effect, whereas the non-uniform diagonals are expected to match the preferred weights of the forecasting tasks with varying future steps. On this basis, we propose a Quadratic Direct Forecas
    
[^486]: DistDF：时间序列预测需要联合分布Wasserstein对齐

    DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment

    [https://arxiv.org/abs/2510.24574](https://arxiv.org/abs/2510.24574)

    提出DistDF框架，通过最小化一种可证明上界于条件分布差异的联合分布Wasserstein差异来对齐预测与标签序列的分布，解决了传统均方误差在标签序列存在自相关时的估计偏差问题。

    

    训练时间序列预测模型需要将模型预测结果的条件分布与标签序列的条件分布对齐。标准的直接预测方法通常采用最小化条件负对数似然的方式，一般通过均方误差来估计。然而，当标签序列存在自相关性时，这种估计会产生偏差。在本文中，我们提出DistDF，通过最小化预测序列与标签序列条件分布之间的分布差异来实现对齐。由于此类条件差异难以从有限的时间序列观测中估计，我们为时间序列预测引入了一种联合分布Wasserstein差异，该差异可被证明是所需条件差异的上界。所提出的差异是易于计算的、可微分的，并且可以方便地与基于梯度的优化方法相结合。大量实验（摘要内容在此处截断）

    arXiv:2510.24574v3 Announce Type: replace-cross  Abstract: Training time-series forecasting models requires aligning the conditional distribution of model forecasts with that of the label sequence. The standard direct forecast (DF) approach resorts to minimizing the conditional negative log-likelihood, typically estimated by the mean squared error. However, this estimation proves biased when the label sequence exhibits autocorrelation. In this paper, we propose DistDF, which achieves alignment by minimizing a distributional discrepancy between the conditional distributions of forecast and label sequences. Since such conditional discrepancies are difficult to estimate from finite time-series observations, we introduce a joint-distribution Wasserstein discrepancy for time-series forecasting, which provably upper bounds the conditional discrepancy of interest. The proposed discrepancy is tractable, differentiable, and readily compatible with gradient-based optimization. Extensive experime
    
[^487]: 与蛋白质对话：一个交互式多模态协同科学家

    Speak to a Protein: An Interactive Multimodal Co-Scientist

    [https://arxiv.org/abs/2510.17826](https://arxiv.org/abs/2510.17826)

    该论文提出了“与蛋白质对话”系统——一个交互式多模态AI协同科学家，它能够检索整合文献、结构与配体数据，在实时3D场景中高亮、注释和操作蛋白质可视化，并按需生成和运行代码，将原本需要数周的蛋白质分析转变为实时交互式对话。

    

    构建一个有效的蛋白质心智模型通常需要数周的阅读、交叉比对晶体结构与预测结构、以及检查配体复合物，这一过程缓慢、可及性不均，且往往需要专业的计算技能。我们推出了“与蛋白质对话”，这是一项新能力，它将蛋白质分析转变为与专家级协同科学家的交互式多模态对话。该AI系统能够检索并综合相关文献、结构和配体数据；将答案扎根于实时3D场景中；并且能够对可视化进行高亮、注释、操作和查看。它还能在需要时生成并运行代码，以文本和图形两种方式解释结果。我们在相关蛋白质上演示了这些能力，针对结合口袋、构象变化或构效关系提出问题，以实时检验科学想法。“与蛋白质对话”将（原文在此处截断）……

    arXiv:2510.17826v2 Announce Type: replace-cross  Abstract: Building a working mental model of a protein typically requires weeks of reading, cross-referencing crystal and predicted structures, and inspecting ligand complexes, an effort that is slow, unevenly accessible, and often requires specialized computational skills. We introduce \emph{Speak to a Protein}, a new capability that turns protein analysis into an interactive, multimodal dialogue with an expert co-scientist. The AI system retrieves and synthesizes relevant literature, structures, and ligand data; grounds answers in a live 3D scene; and can highlight, annotate, manipulate and see the visualization. It also generates and runs code when needed, explaining results in both text and graphics. We demonstrate these capabilities on relevant proteins, posing questions about binding pockets, conformational changes, or structure-activity relationships to test ideas in real time. \emph{Speak to a Protein} reduces the time from quest
    
[^488]: 仅凭原始数据统计量预测核回归学习曲线

    Predicting kernel regression learning curves from only raw data statistics

    [https://arxiv.org/abs/2510.14878](https://arxiv.org/abs/2510.14878)

    该论文提出 Hermite 特征结构假设（HEA），证明仅用经验协方差矩阵和目标函数的多项式分解这两个原始数据统计量，即可在真实图像数据集上准确预测核回归的学习曲线。

    

    我们研究了在包括 CIFAR-5m、SVHN 和 ImageNet 在内的真实数据集上使用常见旋转不变核的核回归问题。我们提出了一个理论框架，仅需两个测量量即可预测学习曲线（测试风险与样本量的关系）：经验数据协方差矩阵和目标函数 $f_*$ 的经验多项式分解。关键的新思想是对核在各向异性数据分布下的特征值和特征函数进行解析近似。这些特征函数类似于数据的 Hermite 多项式，因此我们将这一近似称为 Hermite 特征结构假设（Hermite eigenstructure ansatz，HEA）。我们针对高斯数据证明了 HEA，并且发现真实图像数据通常“足够高斯”，使得 HEA 在实践中能很好地成立，这使我们能够通过应用先前将核特征结构与测试风险联系起来的结果来预测学习曲线。在核回归之外，我们通过实证发现，处于……的多层感知机（MLP）……（摘要原文在此处截断）

    arXiv:2510.14878v3 Announce Type: replace-cross  Abstract: We study kernel regression with common rotation-invariant kernels on real datasets including CIFAR-5m, SVHN, and ImageNet. We give a theoretical framework that predicts learning curves (test risk vs. sample size) from only two measurements: the empirical data covariance matrix and an empirical polynomial decomposition of the target function $f_*$. The key new idea is an analytical approximation of a kernel's eigenvalues and eigenfunctions with respect to an anisotropic data distribution. The eigenfunctions resemble Hermite polynomials of the data, so we call this approximation the Hermite eigenstructure ansatz (HEA). We prove the HEA for Gaussian data, but we find that real image data is often "Gaussian enough" for the HEA to hold well in practice, enabling us to predict learning curves by applying prior results relating kernel eigenstructure to test risk. Extending beyond kernel regression, we empirically find that MLPs in the
    
[^489]: 以量子比特为中心的Transformer用于表面码解码

    Qubit-centric Transformer for Surface Code Decoding

    [https://arxiv.org/abs/2510.11593](https://arxiv.org/abs/2510.11593)

    提出了一种基于Transformer的新型量子纠错解码器QCT，利用以量子比特为中心的注意力机制和融合量子码拓扑结构的图掩码方法，将伴随式转换为量子比特标记以有效识别逻辑错误。

    

    对于可靠的大规模量子计算而言，量子纠错（QEC）对于保护分布在多个物理量子比特上的逻辑信息至关重要。借助深度学习领域的最新进展，基于神经网络的解码器已成为提高量子纠错可靠性的一种有前景的方法。我们提出了以量子比特为中心的Transformer（QCT），这是一种新颖且通用的量子纠错解码器，基于具有以量子比特为中心的注意力机制的Transformer架构。我们的解码器通过专门的嵌入策略，将输入的伴随式从稳定子域转换为以量子比特为中心的标记。这些以量子比特为中心的标记经过注意力层的处理，以有效识别潜在的逻辑错误。此外，我们引入了一种基于图的掩码方法，该方法结合了量子码的拓扑结构，强制注意力聚焦于相关的量子比特相互作用。在各种码距下的表面码实验中……

    arXiv:2510.11593v3 Announce Type: replace-cross  Abstract: For reliable large-scale quantum computation, quantum error correction (QEC) is essential to protect logical information distributed across multiple physical qubits. Taking advantage of recent advances in deep learning, neural network-based decoders have emerged as a promising approach to improve the reliability of QEC. We propose the qubit-centric transformer (QCT), a novel and universal QEC decoder based on a transformer architecture with a qubit-centric attention mechanism. Our decoder transforms input syndromes from the stabilizer domain into qubit-centric tokens via a specialized embedding strategy. These qubit-centric tokens are processed through attention layers to effectively identify the underlying logical error. Furthermore, we introduce a graph-based masking method that incorporates the topological structure of quantum codes, enforcing attention toward relevant qubit interactions. Across various code distances for su
    
[^490]: 打破镜像：基于激活的LLM评估器自我偏好缓解方法

    Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators

    [https://arxiv.org/abs/2509.03647](https://arxiv.org/abs/2509.03647)

    本文提出利用通过对比激活添加（CAA）和优化方法构建的转向向量，在无需重新训练的情况下于推理时缓解LLM评估器的不合理自我偏好偏差，最多可降低97%，显著优于提示和直接偏好优化基线。

    

    大型语言模型（LLM）日益被用作自动评估器，但它们存在“自我偏好偏差”：即倾向于偏好自己的输出而胜过其他模型的输出。这种偏差破坏了评估流程的公平性和可靠性，尤其是在偏好调优和模型路由等任务中。我们研究了轻量级的转向向量能否在推理阶段缓解这一问题而无需重新训练。我们引入了一个精心构建的数据集，将自我偏好偏差区分为合理的自我偏好示例和不合理的自我偏好示例，并采用两种方法构建转向向量：对比激活添加（CAA）和基于优化的方法。我们的结果表明，转向向量可以将不合理的自我偏好偏差降低高达97%，显著优于提示和直接偏好优化基线。然而，转向向量在某些情况下表现不稳定……

    arXiv:2509.03647v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on 
    
[^491]: 联邦学习中梯度反演攻击的实际可行性

    Practical Feasibility of Gradient Inversion Attacks in Federated Learning

    [https://arxiv.org/abs/2508.19819](https://arxiv.org/abs/2508.19819)

    本文系统评估了梯度反演攻击在现实联邦学习系统中的可行性，发现现代性能优化的视觉模型能够有效抵御有意义的图像重构，而已报告的攻击成功多依赖于理想化的上界实验设置。

    

    梯度反演攻击通常被视为联邦学习中的一种严重隐私威胁，近期的研究报告了在有利实验设置下越来越强的图像重构效果。然而，这类攻击在实际部署的现代、性能优化的系统中是否可行仍不清楚。在这项工作中，我们评估了梯度反演在基于图像的联邦学习中的实际可行性。我们在多个数据集和任务上开展了系统性研究，包括图像分类和目标检测，使用了当代分辨率下的经典视觉架构。我们的结果表明，尽管在某些遗留或过渡性设计中、且在高度限制性假设下梯度反演仍然可行，但现代性能优化的模型始终能够在视觉上抵抗有意义的重构。我们进一步证明，许多已报告的攻击成功案例依赖于理想化的上界实验设置。

    arXiv:2508.19819v3 Announce Type: replace-cross  Abstract: Gradient inversion attacks are often presented as a serious privacy threat in federated learning, with recent work reporting increasingly strong reconstructions under favorable experimental settings. However, it remains unclear whether such attacks are feasible in modern, performance-optimized systems deployed in practice. In this work, we evaluate the practical feasibility of gradient inversion for image-based federated learning. We conduct a systematic study across multiple datasets and tasks, including image classification and object detection, using canonical vision architectures at contemporary resolutions. Our results show that while gradient inversion remains possible for certain legacy or transitional designs under highly restrictive assumptions, modern, performance-optimized models consistently resist meaningful reconstruction visually. We further demonstrate that many reported successes rely on upper-bound settings, s
    
[^492]: LHM-Humanoid：面向杂乱场景中连续物体搬运的长时程人体运动控制

    LHM-Humanoid: Long-Horizon Human Motion Control for Continuous Object Transport in Cluttered Scenes

    [https://arxiv.org/abs/2508.16943](https://arxiv.org/abs/2508.16943)

    该论文提出LHM-Humanoid，通过将动作循环间的交接建模为双边可恢复性问题，实现了物理模拟人形角色在杂乱场景中无需重置的连续长时程物体搬运（反复完成寻物、举起、绕障搬运与放置）。

    

    基于物理的人体运动控制可以让模拟角色以高度的物理真实性行走、坐下和操作物体，但几乎总是局限于短暂的、相互隔离且需重新初始化的运动片段。我们的目标是实现连续的、无需重置的长时程运动：让一个物理模拟的人形角色在一次不间断的运行中，反复走到被移动的物体旁，以平衡的全身姿态将其举起，携带其绕过障碍物，并放置到目标位置，如此循环往复。难点不在于任何单个动作，而在于动作之间的过渡。在没有重置机制的情况下，每个循环结束时，既要保证刚放置的物体不受扰动，又要让下一个循环能够顺利开始，然而每次放置都会使角色处于非规范姿态下的失衡状态，此时朴素的端到端强化学习往往无法奏效。我们的核心思想是将这种交接视为一个双边可恢复性问题……（原文摘要在此截断）

    arXiv:2508.16943v4 Announce Type: replace-cross  Abstract: Physics-based human motion control can make a simulated character walk, sit, and manipulate objects with high physical realism. Almost always, though, this happens in short, isolated clips that are re-initialized between interactions. We instead aim for continuous, reset-free long-horizon motion: a physically simulated humanoid that repeatedly walks to a displaced object, lifts it with a balanced whole-body posture, carries it past obstacles, and places it at a goal, over and over within a single uninterrupted take. The hard part is not any individual motion but the transitions between them. Without a reset, each cycle must end in a state that both leaves the object just placed undisturbed and lets the next cycle begin, yet every placement leaves the character off-balance in a non-canonical pose where naive end-to-end reinforcement learning fails. Our key idea is to treat this handoff as a two-sided problem of recoverability: t
    
[^493]: PuzzleJAX：一个面向推理与学习的基准

    PuzzleJAX: A Benchmark for Reasoning and Learning

    [https://arxiv.org/abs/2508.16821](https://arxiv.org/abs/2508.16821)

    PuzzleJAX是一个GPU加速的益智游戏引擎与描述语言，可动态编译PuzzleScript风格的游戏，从而为树搜索、强化学习和LLM推理能力提供大规模、多样化任务的快速基准测试。

    

    我们介绍了PuzzleJAX，这是一个GPU加速的益智游戏引擎与描述语言，旨在支持对树搜索、强化学习以及大语言模型（LLM）推理能力进行快速基准测试。与现有的GPU加速学习环境（它们为固定的游戏集合提供硬编码实现）不同，PuzzleJAX允许对其领域特定语言（DSL）所能表达的任何游戏进行动态编译。该DSL遵循PuzzleScript——一个流行且易于上手的在线益智游戏设计引擎。在本文中，我们在PuzzleJAX中对自2013年发布以来由专业设计师和休闲创作者在PuzzleScript上设计的数千款游戏中的数百款进行了验证，从而证明了PuzzleJAX能够覆盖一个广泛、富有表现力且与人类相关的任务空间。通过分析搜索、学习和语言模型在这些游戏上的表现，我们展示了PuzzleJAX能够自然地表达……

    arXiv:2508.16821v2 Announce Type: replace  Abstract: We introduce PuzzleJAX, a GPU-accelerated puzzle game engine and description language designed to support rapid benchmarking of tree search, reinforcement learning, and LLM reasoning abilities. Unlike existing GPU-accelerated learning environments that provide hard-coded implementations of fixed sets of games, PuzzleJAX allows dynamic compilation of any game expressible in its domain-specific language (DSL). This DSL follows PuzzleScript, which is a popular and accessible online game engine for designing puzzle games. In this paper, we validate in PuzzleJAX several hundred of the thousands of games designed in PuzzleScript by both professional designers and casual creators since its release in 2013, thereby demonstrating PuzzleJAX's coverage of an expansive, expressive, and human-relevant space of tasks. By analyzing the performance of search, learning, and language models on these games, we show that PuzzleJAX can naturally express 
    
[^494]: 基于扩散语言模型的跨模态受控分子生成

    Cross-Modality Controlled Molecule Generation with Diffusion Language Model

    [https://arxiv.org/abs/2508.14748](https://arxiv.org/abs/2508.14748)

    提出模块化框架CMCM-DLM，通过结构控制模块和性质控制模块的分阶段协同设计，使预训练扩散语言模型无需重新训练即可支持分子结构与化学性质等跨模态异构约束的受控分子生成。

    

    分子数据类型的日益丰富，催生了对能够灵活融合跨模态异构约束的生成模型的需求。然而，现有的基于SMILES的扩散模型通常针对固定的条件模态设计，引入新约束往往需要重新训练模型。为解决这一局限，我们提出了CMCM-DLM，一个模块化框架，它扩展了预训练的扩散模型，使其无需重新训练主干网络即可支持异构分子约束。我们使用两种互补的模态来展示CMCM-DLM：分子结构和化学性质。具体而言，结构控制模块（SCM）在扩散早期步骤中引导生成分子的骨架结构，而性质控制模块（PCM）则随后引导生成过程朝向目标化学性质。这种分阶段设计使模型能够……

    arXiv:2508.14748v2 Announce Type: replace-cross  Abstract: The increasing variety of molecular data creates a need for generative models that can flexibly incorporate heterogeneous constraints across modalities. However, existing SMILES-based diffusion models are typically designed for a fixed conditioning modality, and introducing new constraints often requires retraining the model. To address this limitation, we propose Cross-Modality Controlled Molecule Generation with Diffusion Language Model (CMCM-DLM), a modular framework that extends a pre-trained diffusion model to support heterogeneous molecular constraints without retraining the backbone. We demonstrate CMCM-DLM using two complementary modalities: molecular structure and chemical properties. Specifically, a Structure Control Module (SCM) guides early diffusion steps to establish the molecular scaffold, while a Property Control Module (PCM) subsequently steers generation toward target chemical properties. This staged design en
    
[^495]: FedCoT：面向大语言模型的通信高效联邦推理增强

    FedCoT: Communication-Efficient Federated Reasoning Enhancement for Large Language Models

    [https://arxiv.org/abs/2508.10020](https://arxiv.org/abs/2508.10020)

    FedCoT是一个通信高效的联邦推理增强框架，通过轻量级思维链重采样、紧凑判别器筛选以及客户端感知的LoRA堆叠加权聚合，在无需集中式蒸馏和保护隐私的前提下增强大语言模型的逐步推理能力，特别适用于医疗等需要可解释、可审计决策的场景。

    

    在联邦环境下增强大语言模型的推理能力并非易事，因为需要应对严格的计算、通信和隐私约束，尤其是在医疗保健领域，临床上具有重大影响的决策不仅需要准确性，还需要可解释、可审计的推理依据，以满足安全性、问责性和监管要求。传统的联邦微调主要模仿最终答案，而非培养逐步推理能力，且通常依赖隐私敏感的集中式蒸馏，同时仍会产生大量通信开销。我们提出FedCoT来解决这一问题，这是一个联邦推理框架，它将轻量级的思维链重采样与紧凑的判别器相结合用于候选筛选，并采用客户端感知的LoRA堆叠与加权分类器聚合机制，以适应客户端异构性，同时降低聚合噪声和通信成本；客户端在本地生成候选思维链和监督信号，……

    arXiv:2508.10020v2 Announce Type: replace-cross  Abstract: Enhancing LLM reasoning in federated settings is nontrivial due to stringent computational, communication, and privacy constraints, especially in healthcare, where clinically consequential decisions require not only accuracy but also interpretable, auditable rationales to meet safety, accountability, and regulatory requirements. Conventional federated fine-tuning largely imitates final answers rather than cultivating step-by-step reasoning, often relying on privacy-sensitive centralized distillation and still incurring substantial communication overhead. We address this gap with \textbf{\ours{}}, a federated reasoning framework that combines lightweight chain-of-thought resampling with a compact discriminator for selection, and client-aware LoRA stacking with weighted classifier aggregation to accommodate heterogeneity while reducing aggregation noise and communication; clients generate candidate chains and supervision locally,
    
[^496]: 过于范畴化而不似人类：大语言模型与人类中的情绪概念

    Too Categorical to be Human: Emotion Concepts in LLMs and Humans

    [https://arxiv.org/abs/2508.05880](https://arxiv.org/abs/2508.05880)

    该论文提出基于认知评估理论、以“行为表征”刻画情绪概念，并构建涵盖15个情绪类别的基准数据集来比较LLMs与人类的情绪概念表征，发现LLMs的情绪概念过于范畴化而与人类不同。

    

    理解人类情绪对于面向用户的AI应用、安全对齐以及人类行为模拟而言至关重要。由于情绪刺激会塑造大语言模型（LLMs）在高风险情境下的行为，人们越来越关注模型如何在内部表征情绪概念。然而，对这些表征的机制性解释无法直接与人类进行对比：人类的情绪加工是高度分布式的，无法产生等效的神经表征。为了探究LLMs是否以与人类相似的方式内化情绪概念，我们提出利用外部行为特征来刻画情绪这一抽象概念，并将其称为“行为表征”。借助认知评估理论——该理论使得我们能够沿着可解释的评价维度来表征情绪情境——我们构建了一个涵盖15个情绪类别的情绪情境基准数据集。我们引出……（原文摘要在此处截断）

    arXiv:2508.05880v3 Announce Type: replace-cross  Abstract: Understanding human emotions is central to user-facing AI applications, safety alignment, and the simulation of human behavior. As emotional stimuli shape high-stakes behavior in Large Language Models (LLMs), there is increasing interest in how models represent emotion concepts internally. Mechanistic accounts of these representations, however, cannot be compared directly against humans: emotion processing in humans is highly distributed and yields no equivalent neural representation. To understand whether LLMs internalize emotion concepts in a way similar to humans, we propose characterizing the abstract concept of an emotion using external behavioral signatures, which we term behavioral representations. Using the theory of cognitive appraisals, which enables representing emotional situations along interpretable evaluative dimensions, we create a benchmark dataset of emotional scenarios spanning 15 emotion categories. We elici
    
[^497]: DrugMCTS：一种结合多智能体、RAG与蒙特卡洛树搜索的药物重定位框架

    DrugMCTS: a drug repurposing framework combining multi-agent, RAG and Monte Carlo Tree Search

    [https://arxiv.org/abs/2507.07426](https://arxiv.org/abs/2507.07426)

    提出了DrugMCTS框架，协同整合RAG、多智能体协作和蒙特卡洛树搜索，通过五个专门智能体实现结构化迭代推理，在药物重定位任务上取得显著更高的召回率和鲁棒性。

    

    大语言模型的最新进展在药物重定位等科学领域展现出了巨大潜力。然而，当推理超出预训练期间所获得的知识范围时，其有效性仍然受到限制。传统方法（如微调或检索增强生成）要么面临高昂的计算开销，要么无法充分利用结构化的科学数据。为了克服这些挑战，我们提出了DrugMCTS，这是一个新颖的框架，协同整合了RAG、多智能体协作和蒙特卡洛树搜索，用于药物重定位。该框架采用五个专门的智能体负责检索和分析分子与蛋白质信息，从而实现结构化和迭代式的推理。在DrugBank和KIBA数据集上的大量实验表明，DrugMCTS实现了显著更高的召回率和鲁棒性。

    arXiv:2507.07426v4 Announce Type: replace  Abstract: Recent advances in large language models have demonstrated considerable potential in scientific domains such as drug repositioning. However, their effectiveness remains constrained when reasoning extends beyond the knowledge acquired during pretraining. Conventional approaches, such as fine-tuning or retrieval-augmented generation, face limitations in either imposing high computational overhead or failing to fully exploit structured scientific data. To overcome these challenges, we propose DrugMCTS, a novel framework that synergistically integrates RAG, multi-agent collaboration, and Monte Carlo Tree Search for drug repositioning. The framework employs five specialized agents tasked with retrieving and analyzing molecular and protein information, thereby enabling structured and iterative reasoning. Extensive experiments on the DrugBank and KIBA datasets demonstrate that DrugMCTS achieves substantially higher recall and robustness com
    
[^498]: Time-o1：时间序列预测需要变换后的标签对齐

    Time-o1: Time-Series Forecasting Needs Transformed Label Alignment

    [https://arxiv.org/abs/2505.17847](https://arxiv.org/abs/2505.17847)

    Time-o1通过将标签序列变换为去相关且区分显著性的分量，训练模型对齐最显著分量，从而提出一种变换增强的损失函数，有效缓解标签自相关性并减少任务数量，在时间序列预测中达到最先进性能且兼容多种预测模型。

    

    训练时间序列预测模型在损失函数设计方面面临独特的挑战。大多数现有方法采用时间均方误差，但本研究揭示了其两个关键局限性：(1) 它忽略了标签自相关性的存在，使其偏离了真实标签序列的似然；(2) 它涉及过多的任务数量，使优化变得复杂，尤其是在长期预测中。为了解决这些问题，我们提出了Time-o1，一种面向时间序列预测的变换增强损失函数。其核心思想是将标签序列变换为具有区分性显著性的去相关分量，然后训练模型对齐最显著的分量，从而有效缓解标签自相关性并减少任务数量。实验表明，Time-o1实现了最先进的性能，并且与各种预测模型兼容。

    arXiv:2505.17847v3 Announce Type: replace-cross  Abstract: Training time-series forecasting models poses unique challenges in loss function design. Most existing approaches adopt temporal mean squared error, but this study reveals two critical limitations: (1) it ignores the presence of label autocorrelation, which biases it from the true label sequence likelihood; (2) it involves excessive number of tasks, which complicates optimization, especially for long-term forecasting. To address these issues, we introduce Time-o1, a transform-enhanced loss function for time-series forecasting. The central idea is to transform the label sequence into decorrelated components with discriminated significance. Models are then trained to align the most significant components, thereby effectively mitigating label autocorrelation and reducing task amount. Experiments demonstrate that Time-o1 achieves state-of-the-art performance and is compatible with various forecast models. Code is available at https
    
[^499]: KO：基于动理学启发的神经优化器与偏微分方程模拟方法

    KO: Kinetics-inspired Neural Optimizer with PDE Simulation Approaches

    [https://arxiv.org/abs/2505.14777](https://arxiv.org/abs/2505.14777)

    提出了一种基于动理学与偏微分方程的即插即用优化器KO，它将参数动力学建模为粒子系统并通过离散化玻尔兹曼输运方程引入随机相互作用，从而提升参数多样性、缓解权重凝聚并保持收敛保证。

    

    arXiv:2505.14777v2 公告类型：replace-cross 摘要：为神经网络设计有效的优化算法仍然是一个根本性的挑战，而现有的大多数方法都依赖于基于梯度更新的启发式扩展。我们提出了KO（动理学启发优化器，Kinetics-inspired Optimizer），这是一个基于动理学理论和偏微分方程的即插即用优化模块。KO将参数动力学建模为一个粒子系统，通过对玻尔兹曼输运方程的离散化引入随机相互作用，从而增强标准的梯度更新。这一机制自然地促进了参数多样性，并缓解了权重凝聚（weight condensation）现象——即参数坍缩到低维子空间的倾向，这一现象与泛化能力退化密切相关。我们提供了严格的理论分析和物理解释，证明KO在保持收敛性保证的同时，能够可证明地增加参数多样性。在图像分类任务上的大量实验……

    arXiv:2505.14777v2 Announce Type: replace-cross  Abstract: The design of effective optimization algorithms for neural networks remains a fundamental challenge, and most existing methods rely on heuristic extensions of gradient-based updates. We introduce KO (Kinetics-inspired Optimizer), a plug-and-play optimization module grounded in kinetic theory and partial differential equations. KO models parameter dynamics as a particle system, augmenting standard gradient updates with stochastic interactions induced by a discretization of the Boltzmann transport equation. This mechanism naturally promotes parameter diversity and mitigates weight condensation, the tendency of parameters to collapse into low-dimensional subspaces, a phenomenon closely associated with degraded generalization. We provide both a rigorous theoretical analysis and a physical interpretation, showing that KO provably increases parameter diversity while preserving convergence guarantees. Extensive experiments on image cl
    
[^500]: 掩码微调助力大语言模型性能提升

    Boosting Large Language Models with Mask Fine-Tuning

    [https://arxiv.org/abs/2503.22764](https://arxiv.org/abs/2503.22764)

    提出掩码微调（MFT）这一新颖的大语言模型微调范式，通过学习并应用二值掩码、在不更新模型权重的情况下精心打破模型结构完整性，从而在不同领域和骨干网络上获得一致的性能提升。

    

    大语言模型（LLM）通常被整合进主流的优化流程之中。然而，保持模型的完整性对于获得良好性能是否是不可或缺的，这一问题仍未被充分探索。在本工作中，我们提出了掩码微调，这是一种新颖的大语言模型微调范式，其表明精心地打破模型的结构完整性，可以在不更新模型权重的情况下出人意料地提升性能。MFT以标准的LLM微调目标作为监督，学习并应用二值掩码到已经过良好优化的模型上。基于完全微调后的模型，MFT使用相同的微调数据集，在不同领域和不同骨干网络上实现了一致的性能提升（例如，LLaMA2-7B/3.1-8B在IFEval上平均提升2.70/4.15分）。详细的消融实验和分析从多个角度对所提出的MFT进行了考察，包括稀疏比率和损失面等。此外，……

    arXiv:2503.22764v3 Announce Type: replace-cross  Abstract: The large language model (LLM) is typically integrated into the mainstream optimization protocol. However, it remains underexplored whether maintaining the model integrity is \textit{indispensable} for promising performance. In this work, we introduce Mask Fine-Tuning (MFT), a novel LLM fine-tuning paradigm demonstrating that carefully breaking the model's structural integrity can surprisingly improve performance without updating model weights. MFT learns and applies binary masks to well-optimized models, using the standard LLM fine-tuning objective as supervision. Based on fully fine-tuned models, MFT uses the same fine-tuning datasets to achieve consistent performance gains across domains and backbones (e.g., an average gain of 2.70/4.15 on IFEval with LLaMA2-7B/3.1-8B). Detailed ablation studies and analyses examine the proposed MFT from different perspectives, including the sparse ratio and the loss surface. Additionally, w
    
[^501]: 大规模预训练数据集并不能保证图像分类中微调后的鲁棒性

    Large Pretraining Datasets Don't Guarantee Robustness after Fine-Tuning in Image Classification

    [https://arxiv.org/abs/2410.21582](https://arxiv.org/abs/2410.21582)

    该论文提出鲁棒性继承基准ImageNet-RIB，揭示了一个关键问题：即使在大规模数据集上预训练的模型，经微调后仍会出现严重的灾难性遗忘和分布外泛化能力丧失，因此大规模预训练并不保证微调后的鲁棒性。

    

    arXiv:2410.21582v4 公告类型：replace-cross 摘要：大规模预训练模型被广泛用作通过微调学习新专门任务的基础，其目标是在保持模型整体性能的同时使其获得新技能。对于所有此类模型而言，一个重要的目标是鲁棒性，即在分布外（OOD）任务上表现良好的能力。我们评估了微调是否能在图像分类中保留预训练模型的整体鲁棒性，观察到在大规模数据集上预训练的模型表现出严重的灾难性遗忘和OOD泛化能力的丧失。为了系统地评估微调模型的鲁棒性保持情况，我们提出了鲁棒性继承基准ImageNet-RIB）。该基准可应用于任何预训练模型，由一组相关但不同的OOD（下游）任务组成，其做法是在集合中的某一个OOD任务上进行微调，然后在其余任务上进行测试。我们发现，尽管……

    arXiv:2410.21582v4 Announce Type: replace-cross  Abstract: Large-scale pretrained models are widely leveraged as foundations for learning new specialized tasks via fine-tuning, with the goal of maintaining the general performance of the model while allowing it to gain new skills. A valuable goal for all such models is robustness: the ability to perform well on out-of-distribution (OOD) tasks. We assess whether fine-tuning preserves the overall robustness of the pretrained model in image classification, and observed that models pretrained on large datasets exhibited strong catastrophic forgetting and loss of OOD generalization. To systematically assess robustness preservation in fine-tuned models, we propose the Robustness Inheritance Benchmark (ImageNet-RIB). The benchmark, which can be applied to any pretrained model, consists of a set of related but distinct OOD (downstream) tasks and involves fine-tuning on one of the OOD tasks in the set then testing on the rest. We find that thoug
    
[^502]: 当解释相互竞争时：不确定性下的策略感知选择

    When Explanations Compete: Policy-Aware Selection Under Uncertainty

    [https://arxiv.org/abs/2410.05479](https://arxiv.org/abs/2410.05479)

    该论文提出了一个策略感知的解释选择框架，通过结合资格规则、双向帕累托筛选和策略感知排序，从不确定性感知解释方法生成的多个候选解释中，依据预测置信度、不确定性和应用约束进行最优选择。

    

    不确定性感知的解释方法通常会为同一预测产生多个备选解释。在它们之间进行选择需要一种策略来平衡预测置信度、不确定性和应用约束。本文提出了一个框架，用于将此类策略应用于一组固定的已生成解释。候选解释通过不确定性变化、预测方向，以及在可用情况下相对于决策边界的区间位置来表征。该框架将这些属性与资格规则、可选的双向帕累托筛选以及策略感知排序相结合。一个虚构的前列腺癌示例说明了不同的解释目的如何从同一候选集合中得出不同的选择。我们使用校准解释在分类、阈值回归和普通回归任务上对该框架进行了实例化。在41个基准数据集上，单一解释的平均候选数量范围从11.57到21.75。

    arXiv:2410.05479v2 Announce Type: replace  Abstract: Uncertainty-aware explanation methods often produce several alternatives for the same prediction. Selecting among them requires a policy for balancing prediction confidence, uncertainty, and application constraints. This paper presents a framework for applying such policies to a fixed set of generated explanations. Candidates are characterised by uncertainty change, prediction direction, and, when available, interval position relative to a decision boundary. The framework combines these properties with eligibility rules, optional bidirectional Pareto screening, and policy-aware ranking. A fictitious prostate-cancer example illustrates how different explanatory purposes lead to different selections from the same candidate set. We instantiate the framework with Calibrated Explanations for classification, thresholded regression, and plain regression. Across 41 benchmark datasets, mean candidate counts range from 11.57 to 21.75 for singl
    
[^503]: 使用选择性状态空间模型建模光学压缩器的时变响应

    Modeling Time-Dependent Responses of Optical Compressors with Selective State Space Models

    [https://arxiv.org/abs/2408.12549](https://arxiv.org/abs/2408.12549)

    该论文提出了一种结合选择性状态空间模型、特征级线性调制和门控线性单元的深度神经网络方法来建模光学压缩器的时变响应，性能超越基于循环层的方法，适用于低延迟实时音频处理，并在 TubeTech CL 1B 和 Teletronix LA-2A 两款模拟光学压缩器上得到验证。

    

    本文提出了一种使用带选择性状态空间模型的深度神经网络来建模光学动态范围压缩器的方法。该方法通过采用选择性状态空间模块对输入音频进行编码，超越了以往基于循环层的方法。其特色在于一种融合了特征级线性调制和门控线性单元的改进技术，可动态调整网络，根据外部参数控制压缩的启动和释放阶段。所提出的架构非常适合低延迟和实时应用，这对现场音频处理至关重要。该方法已在具有不同特性的模拟光学压缩器 TubeTech CL 1B 和 Teletronix LA-2A 上得到验证。评估采用定量指标和主观听音测试，将所提出的方法与其他最先进的模型进行比较。结果表明，我们的方法...

    arXiv:2408.12549v4 Announce Type: replace-cross  Abstract: This paper presents a method for modeling optical dynamic range compressors using deep neural networks with Selective State Space models. The proposed approach surpasses previous methods based on recurrent layers by employing a Selective State Space block to encode the input audio. It features a refined technique integrating Feature-wise Linear Modulation and Gated Linear Units to adjust the network dynamically, conditioning the compression's attack and release phases according to external parameters. The proposed architecture is well-suited for low-latency and real-time applications, crucial in live audio processing. The method has been validated on the analog optical compressors TubeTech CL 1B and Teletronix LA-2A, which possess distinct characteristics. Evaluation is performed using quantitative metrics and subjective listening tests, comparing the proposed method with other state-of-the-art models. Results show that our bla
    
[^504]: 基于扩散模型的视频编辑：综述

    Diffusion Model-Based Video Editing: A Survey

    [https://arxiv.org/abs/2407.07111](https://arxiv.org/abs/2407.07111)

    本文是一篇综述，系统梳理了基于扩散模型的视频编辑技术的理论基础、方法分类与演化脉络，探讨了点编辑、姿态引导人体视频编辑等新兴应用，并提出新的V2VBench基准对该领域进行全面对比评估。

    

    扩散模型的快速发展极大地推动了图像和视频应用的发展，使“所见即所想”成为现实。其中，视频编辑受到了广泛关注，相关研究活动迅速增长，因此有必要对现有文献进行全面而系统的回顾。本文综述了基于扩散模型的视频编辑技术，包括理论基础和实际应用。我们首先概述了数学公式化表述以及图像领域的关键方法。随后，我们根据核心技术的内在联系对视频编辑方法进行分类，描绘出其演化轨迹。本文还深入探讨了新颖的应用，包括基于点的编辑和姿态引导的人体视频编辑。此外，我们使用新提出的V2VBench进行了全面的对比评估。

    arXiv:2407.07111v2 Announce Type: replace-cross  Abstract: The rapid development of diffusion models (DMs) has significantly advanced image and video applications, making "what you want is what you see" a reality. Among these, video editing has gained substantial attention and seen a swift rise in research activity, necessitating a comprehensive and systematic review of the existing literature. This paper reviews diffusion model-based video editing techniques, including theoretical foundations and practical applications. We begin by overviewing the mathematical formulation and image domain's key methods. Subsequently, we categorize video editing approaches by the inherent connections of their core technologies, depicting evolutionary trajectory. This paper also dives into novel applications, including point-based editing and pose-guided human video editing. Additionally, we present a comprehensive comparison using our newly introduced V2VBench. Building on the progress achieved to date
    
[^505]: FreDF: 在频域中学习预测

    FreDF: Learning to Forecast in Frequency Domain

    [https://arxiv.org/abs/2402.02399](https://arxiv.org/abs/2402.02399)

    FreDF是一种在频域中学习预测的方法，解决了时间序列建模中标签序列的自相关问题，相比现有方法有更好的性能表现，并且与各种预测模型兼容。

    

    时间序列建模在历史序列和标签序列中都面临自相关的挑战。当前的研究主要集中在处理历史序列中的自相关问题，但往往忽视了标签序列中的自相关存在。具体来说，新兴的预测模型主要遵循直接预测（DF）范式，在标签序列中假设条件独立性下生成多步预测。这种假设忽视了标签序列中固有的自相关性，从而限制了基于DF的模型的性能。针对这一问题，我们引入了频域增强直接预测（FreDF），通过在频域中学习预测来避免标签自相关的复杂性。我们的实验证明，FreDF在性能上大大超过了包括iTransformer在内的现有最先进方法，并且与各种预测模型兼容。

    Time series modeling is uniquely challenged by the presence of autocorrelation in both historical and label sequences. Current research predominantly focuses on handling autocorrelation within the historical sequence but often neglects its presence in the label sequence. Specifically, emerging forecast models mainly conform to the direct forecast (DF) paradigm, generating multi-step forecasts under the assumption of conditional independence within the label sequence. This assumption disregards the inherent autocorrelation in the label sequence, thereby limiting the performance of DF-based models. In response to this gap, we introduce the Frequency-enhanced Direct Forecast (FreDF), which bypasses the complexity of label autocorrelation by learning to forecast in the frequency domain. Our experiments demonstrate that FreDF substantially outperforms existing state-of-the-art methods including iTransformer and is compatible with a variety of forecast models.
    
[^506]: 用于捆绑包推荐的超图增强双卷积网络

    Hypergraph-Enhanced Dual Convolutional Network for Bundle Recommendation

    [https://arxiv.org/abs/2312.11018](https://arxiv.org/abs/2312.11018)

    HED 通过构建包含用户-捆绑包、用户-物品、捆绑包-物品交互及用户内部、捆绑包内部关系的完整超图，并将全超图传播与用户-捆绑包双卷积分支耦合，在保留推荐特有信号的同时引入物品感知的高阶上下文，在 NetEase 和 Youshu 数据集上较最强基线显著提升了捆绑包推荐性能。

    

    arXiv:2312.11018v3 公告类型：replace-cross 摘要：捆绑包推荐是对相关物品的集合而非孤立物品进行排序。其核心挑战在于在不丢失捆绑包排序所需信号的前提下，连接用户偏好、物品交互与捆绑包构成。我们提出了超图增强双卷积神经网络（HED），该网络构建了一个完整的超图，其中包含用户-捆绑包、用户-物品和捆绑包-物品交互，以及用户内部和捆绑包内部关系。HED 将完整超图传播与用户-捆绑包分支相结合，使物品感知的高阶上下文能够为排序提供信息，同时保留推荐特有的信号。在 NetEase 数据集上，HED-128 在六项报告指标上较最强基线提升了 5.04%–6.97%；在 Youshu 数据集上，HED-64 提升了 1.87%–4.56%。消融实验结果支持了用户-捆绑包分支和类型内关系二者的贡献，敏感性分析确定了稳定的运行范围。

    arXiv:2312.11018v3 Announce Type: replace-cross  Abstract: Bundle recommendation ranks sets of related items rather than isolated items. Its central challenge is to connect user preferences, item interactions, and bundle composition without losing the signals needed to rank bundles. We propose Hypergraph-Enhanced Dual Convolutional Neural Network (HED), which constructs a complete hypergraph containing user--bundle, user--item, and bundle--item interactions together with intra-user and intra-bundle relations. HED couples complete-hypergraph propagation with a user--bundle branch, allowing item-aware higher-order context to inform ranking while preserving recommendation-specific signals. On NetEase, HED-128 improves over the strongest baseline by 5.04--6.97% across the six reported metrics; on Youshu, HED-64 improves by 1.87--4.56%. Ablation results support the contributions of both the user--bundle branch and intra-type relations, and sensitivity analyses identify stable operating rang
    
[^507]: 基于感受野袋的快速、可解释且确定性的时间序列分类

    Fast, Interpretable, and Deterministic Time Series Classification With a Bag-of-Receptive-Fields

    [https://arxiv.org/abs/2311.18029](https://arxiv.org/abs/2311.18029)

    本文提出了BORF（感受野袋），一种快速、可解释且确定性的时间序列分类变换方法，克服了现有黑盒分类器难以理解以及可解释方法因依赖随机化导致解释不稳定的问题。

    

    时间序列分类文献的当前趋势是通过在集成混合模型中组合多个模型、在复杂且富有表现力的特征空间中表示时间序列，以及从同一时间序列的不同表示中提取特征，来开发越来越精确的算法。由于这种对预测性能的过度关注，最好的时间序列分类器都是黑盒模型，从人类的角度难以理解。即使是那些被认为可解释的方法，例如基于形状特征的方法，也依赖随机化来保持计算效率。这给可解释性带来了挑战，因为每次运行的解释都可能不同。鉴于这些局限性，我们提出了感受野袋（Bag-Of-Receptive-Field，BORF），一种快速、可解释且确定性的时间序列变换方法。在经典的模式袋方法基础上，我们弥合了卷积算子与（摘要在此处截断）

    arXiv:2311.18029v2 Announce Type: replace-cross  Abstract: The current trend in the literature on Time Series Classification is to develop increasingly accurate algorithms by combining multiple models in ensemble hybrids, representing time series in complex and expressive feature spaces, and extracting features from different representations of the same time series. As a consequence of this focus on predictive performance, the best time series classifiers are black-box models, which are not understandable from a human standpoint. Even the approaches that are regarded as interpretable, such as shapelet-based ones, rely on randomization to maintain computational efficiency. This poses challenges for interpretability, as the explanation can change from run to run. Given these limitations, we propose the Bag-Of-Receptive-Field (BORF), a fast, interpretable, and deterministic time series transform. Building upon the classical Bag-Of-Patterns, we bridge the gap between convolutional operator
    

