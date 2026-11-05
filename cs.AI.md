# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [BEAKER: An Expert-Curated Benchmark for Embodied Brains in Self-Driving Chemical Laboratories](https://arxiv.org/abs/2611.06911) | 该论文提出了BEAKER——一个由专家策划、包含1,000个问答样本和11种图像与视频任务的基准，用于评估多模态大语言模型在自动驾驶化学实验室中面向执行的具身推理与操作能力。 |
| [^2] | [FrontierPhysics: Evaluating Agents for End-to-End Physics Research](https://arxiv.org/abs/2611.06910) | 该论文提出了FrontierPhysics——一个由42位物理学家基于自身研究设计的包含56个任务、覆盖九个领域的端到端物理研究基准，通过结合确定性验证器与专家评分标准评估AI智能体完成完整物理研究的能力，发现现有模型表现远未达到可发表水平。 |
| [^3] | [Perceptive Humanoid Motion from Simulation](https://arxiv.org/abs/2611.06908) | 提出了HALO人形机器人自主系统，通过在千个程序化生成的家庭环境中合成视觉-语言-运动数据，使数据规模随算力而非人力扩展，让人形机器人掌握坐、导航、穿门、蹲爬、推箱等多种物理技能并协调完成复合任务。 |
| [^4] | [Occlusion for Free: Detecting Occlusion from the Disagreement Between Two Monocular Distance Cues](https://arxiv.org/abs/2611.06899) | 该论文提出一种无需训练、无需遮挡标注的方法，仅利用检测框表观高度（不受遮挡影响）与单目度量深度（被拉向遮挡物）之间的不一致性即可检测行人遮挡，在KITTI上达到AUC 0.909。 |
| [^5] | [Dull, Dirty, Dangerous? Working Conditions and the Limits of Occupational AI Exposure](https://arxiv.org/abs/2611.06888) | 该研究发现“枯燥、肮脏、危险”的工作条件与职业AI暴露度之间的关联在控制体力要求、教育培训和工资等职业特征后会显著减弱甚至消失，揭示了当前常用AI暴露度指标在解释上的局限性。 |
| [^6] | [Learning to Decode Linear Codes with Generalized Belief Propagation Network](https://arxiv.org/abs/2611.06874) | 本文提出广义信念传播网络GBPNet，在保持消息传递对称性和Tanner图结构的前提下，通过级联变量节点更新、可学习消息缩放以及渐进式和高信噪比训练策略，实现了优于传统信念传播算法和现有最先进神经译码器的误码率性能。 |
| [^7] | [On Turn-Level Best-of-N Sampling for Scaling LLM Agents](https://arxiv.org/abs/2611.06868) | 该论文提出一种轮次级Best-of-N采样的推理时扩展方法，通过生成器-选择器双模型框架在每轮生成并筛选候选动作，显著提升了LLM智能体在长时程任务上的准确率与稳定性。 |
| [^8] | [CounterFactual Retrieval for Evidence-Aware Individual Animal Identification](https://arxiv.org/abs/2611.06864) | 提出反事实前景检索（CFR）这一证据感知评估协议，通过完整图像、仅前景、仅捷径三种视图对动物个体识别进行评测，将检索分数转化为证据画像，从而揭示模型是依赖动物特征还是身份相关的背景捷径。 |
| [^9] | [CompressionBench: How can your coding agents compress LLMs?](https://arxiv.org/abs/2611.06858) | 该论文提出CompressionBench基准，将大语言模型压缩形式化为一项终端任务，系统评估了编程智能体自主压缩LLM的能力，发现大多数智能体提交能通过独立验证，且压缩后模型性能接近原始BF16模型。 |
| [^10] | [ServerBench: How can your coding agents design specialized LLM serving systems?](https://arxiv.org/abs/2611.06841) | 本文提出ServerBench，首个让编程智能体以“每美元收益”为目标通过实验自主设计并验证专用LLM服务系统的基准，其中Claude Sonnet 5.5的设计达到vLLM基线的4.87倍。 |
| [^11] | [Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token](https://arxiv.org/abs/2611.06837) | 本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。 |
| [^12] | [Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS](https://arxiv.org/abs/2611.06829) | 该论文提出了一套完整的从Vitis HLS到AWS云FPGA部署的工具链和四级误差定位方法论，使开发者乃至AI编码智能体能够通过高级语言在AWS云FPGA上高效开发、测试和部署机器学习模型。 |
| [^13] | [Rules, Traces and Routing Statistics: Training and Calibration Data for Open Models on ARC-AGI-2](https://arxiv.org/abs/2611.06827) | 通过在类型化语言中构建12,000条经行为去重的可执行规则和95,993个经过捷径求解器过滤的高质量合成任务，并结合真实任务记录微调与四GPU调度优化，将开放模型在ARC-AGI-2半私有集上的分数从28.96提升至33.06，同时证明纯合成数据训练无法超越真实数据模型。 |
| [^14] | [Task-Efficient Agent Evaluation Using Cached Traces](https://arxiv.org/abs/2611.06803) | 该论文提出用LLM提取智能体执行轨迹的结构化摘要并嵌入表示，在保留正确性与任务相关信号的同时滤除系统身份等干扰信号，从而利用缓存轨迹高效预测新智能体系统的基准测试得分。 |
| [^15] | [Message-passing graph persistence](https://arxiv.org/abs/2611.06801) | 本文提出消息传递图持久性（MGP），将消息传递机制融入图过滤过程构建拓扑描述子，并证明了其图同构不变性以及严格强于现有基于持久同调描述子的表达能力。 |
| [^16] | [Transforming Integer Grids to Spatial Intuition: Perception, Not Planning, Governs ARC-AGI-3](https://arxiv.org/abs/2611.06793) | 该论文提出开源智能体SARBLOH，证明在ARC-AGI-3交互推理中起决定作用的是空间感知而非规划能力——其零参数O(HW)感知引擎可在35毫秒内将整数网格帧转化为对象清单与几何简报，从而大幅减少前沿模型因序列化破坏空间拓扑而产生的高昂探索浪费。 |
| [^17] | [FedFiOT: Fisher-Weighted Optimal Transport for Model Merging in One-Shot and Few-Round Federated Learning](https://arxiv.org/abs/2611.06783) | FedFiOT提出了一种基于Fisher信息加权的最优传输框架，用于单轮和小轮次联邦学习，通过量化参数重要性来指导神经元对齐与模型聚合，解决了客户端模型漂移和数据异质性导致的直接平均性能下降问题。 |
| [^18] | [Route First, See Later: Enabling Vision Agents to Reason Scientifically by Separating Routing Policy from Action Spaces](https://arxiv.org/abs/2611.06760) | 提出ZETA架构，通过将路由策略与视觉工具执行解耦，让智能体先解析科学意图再调用专用工具，并将测量建立在显式几何之上，从而将推理错误与感知失败作为独立失败模式进行审计评估，提升科学图像分析自动化的可靠性。 |
| [^19] | [EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models](https://arxiv.org/abs/2611.06748) | 提出EvoWorld模型与“视野外因果演化”新问题，使生成式视频模型不仅能渲染逼真画面，还能推理事件对世界实体的影响并生成因果一致的世界演化。 |
| [^20] | [TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing](https://arxiv.org/abs/2611.06741) | TailSteer通过在去噪轨迹尾部集中施加保真校正（该校正从模型自身的在策略rollout中学习），在不改变早期去噪动力学和预训练先验的前提下，有效抑制多轮图像编辑中累积的视觉漂移。 |
| [^21] | [Smartphone-Aided VR Exergames for HIIT: Gamifying Fitness with Real-Time Pose Estimation](https://arxiv.org/abs/2611.06734) | 该论文提出了一种利用普通智能手机摄像头进行实时全身姿态估计的VR健身游戏系统，结合11个游戏化HIIT组合动作以及基于MCP协议和GPT-4o的后端生物力学数据分析，为高强度间歇训练提供了低成本、可扩展、无需穿戴设备的运动追踪与健身游戏化方案。 |
| [^22] | [ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models](https://arxiv.org/abs/2611.06732) | 该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。 |
| [^23] | [Dexterous Robotic Hands in the Foundation-Model Era: A Full-Stack Survey of Hardware, Data, Cross-Embodiment Transfer, Learning, and Evaluation](https://arxiv.org/abs/2611.06724) | 这是一篇全栈式综述，首次将基础模型时代的灵巧机械手研究系统性地组织为硬件、数据、跨本体迁移、学习与评估五个相互依赖的层次，并通过案例分析揭示各层设计选择之间的内在联系。 |
| [^24] | [Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge](https://arxiv.org/abs/2611.06715) | 该研究提出Rel-Bloom多语言认知诊断基准，在五个布鲁姆认知层次和三种真实性条件下对43个模型进行大规模测试，发现多语言大模型在基于上下文证据进行关系推理时，从“应用”到“分析”层次准确率急剧下降，揭示出模型对非参数记忆进行高阶推理的能力存在显著瓶颈。 |
| [^25] | [Beyond the principle-practice gap: Responsible AI as situated epistemic and ethical practice](https://arxiv.org/abs/2611.06709) | 本文基于对反人口贩卖AI系统从业者的访谈，论证“原则与实践差距”并非执行缺陷，而是源于不同制度条件下人们对伦理问题认知方式的差异，主张将负责任AI理解为一种情境化的认识论与伦理实践。 |
| [^26] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^27] | [Don't Replace, Complement: A Detector-Agnostic Subspace Signal for Near-OOD Detection](https://arxiv.org/abs/2611.06688) | 该论文提出PLUS方法，利用稀疏自编码器为每个类别构建低维正交子空间并计算白化投影评分，作为与现有检测评分互补的近分布外检测信号，从而在不牺牲远分布外性能的前提下提升近分布外检测能力。 |
| [^28] | [LifeVal: Evaluating AI Model Performance on Personalized Agentic Consumer Tasks](https://arxiv.org/abs/2611.06687) | LifeVal是一个基于3800多名消费者支付意愿调查构建的基准，涵盖148类代表每年超1.7万亿美元支付意愿的消费任务，并利用真实ChatGPT对话和个人背景创建个性化任务来评估AI模型的智能体能力。 |
| [^29] | [AgentTrustCI: Build-Time Policy, Grounding, and Trajectory Tests for Java LLM Agents](https://arxiv.org/abs/2611.06674) | AgentTrustCI提出了一个Java/JUnit 5框架，将策略、接地与轨迹检查以确定性重放的形式转化为可重复的构建时测试，从而在应用层验证LLM智能体的工具选择、答案依据与工作流程合规性，并在55个合成案例上准确分类了全部注入故障。 |
| [^30] | [Where Search Agents Go Off the Rails: Graph-Grounded Step-Level Verification](https://arxiv.org/abs/2611.06662) | GraphRail 将需求、证据和推理表示为结构化声明并进行对比，无需生成式评判者或标准答案即可对搜索智能体的中间推理步骤进行可追溯的验证，在多跳问答任务中比生成式验证器更有效地发现中间错误。 |
| [^31] | [Human Motion Prediction via Continual Prior Compensation](https://arxiv.org/abs/2611.06637) | 提出了一种持续先验补偿（CPC）框架，将不同时刻的人体运动预测任务分解为多个子任务并进行多阶段渐进式训练，同时通过可学习的随机变量补偿先验信息遗忘，从而避免短期预测的学习受到长期预测的干扰。 |
| [^32] | [Trick-or-Treat: Adversarial Evaluation of Multimodal Medical Reasoning with Synthetic Images](https://arxiv.org/abs/2611.06631) | 该论文提出Trick-or-Treat基准，通过将真实胸部X光片与合成的临床照片配对构成对抗性多模态评测框架，有效暴露了前沿模型在跨模态医学推理中的幻觉行为。 |
| [^33] | [Noise-regularized training of quantum circuit Born machines under a finite shot budget](https://arxiv.org/abs/2611.06630) | 该论文证明硬件噪声可作为量子电路玻恩机器的正则化手段以提升泛化能力，并提出GENIE框架在有限测量预算内自动选择最优噪声水平。 |
| [^34] | [Does Frontier AI Strengthen Cyber Defense? Evidence from Project Glasswing](https://arxiv.org/abs/2611.06629) | 本文利用“玻璃翼项目”作为准自然实验，发现前沿AI漏洞发现模型使获得访问权限的美国科技公司漏洞披露翻倍、高影响漏洞披露增加约150%，外部赏金研究者将报告转向其他公司，且未见攻击者利用额外漏洞的证据，表明前沿AI目前主要强化了网络防御而非攻击。 |
| [^35] | [From Certified Pairs to a Certified Hub: Reasoner-Checked Crosswalks for the Information Exchange Standard](https://arxiv.org/abs/2611.06628) | 该论文将交叉映射的有效性精确定义为合并本体的连贯性与保守性，并通过“匹配—人工审核—SKOS对应认证—HermiT推理”的分阶段流程校验信息交换标准（IES）通向十三个本体的映射，首次实现了从经认证的成对映射升级到经认证的整个枢纽。 |
| [^36] | [Interactive World Simulation with Cosmos for Physical AI](https://arxiv.org/abs/2611.06627) | 该论文提出了一个将 Cosmos 世界基础模型适配为实时交互式世界仿真的共享框架，使用户和具身智能体能够通过动作与场景控制同演化的虚拟环境交互，并支持相机导航、机器人操作、可控视频迁移和多视角自动驾驶仿真等应用。 |
| [^37] | [From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels](https://arxiv.org/abs/2611.06612) | 该论文提出一种测试时优化方法，将冻结的序列重建器在推理时转化为插补器——把缺失条目作为优化变量、仅最小化观测坐标上的重建误差，从而无需人工掩码增强或全通道丢弃训练即可应对完整传感器通道失效导致的时间序列缺失。 |
| [^38] | [Time-Division Multiplexing: The Operational Rhythm and Temporal Organization Layer of Digital Life](https://arxiv.org/abs/2611.06603) | 本文受灵长类皮层快速交替感知表征的启发，提出了一种时分复用式的时间组织层，通过相位标记微时段、可观测占空比与三个嵌套时钟，将数字生命智能体的忙碌与空闲通道统一为同一过程的不同阶段，并给出了占空比更新的确定性与随机下降保证。 |
| [^39] | [Signed, Sealed, Leaked: Trust Beyond Signature for MCP Toolchains](https://arxiv.org/abs/2611.06585) | 提出“种系执行”机制：每次工具调用都在从无数据的种系进程分叉出的临时副本中运行，且只接收其目的地被授权的信息，从而无需检查或信任工具实现即可防御已签名但恶意的 MCP 工具的隐蔽数据泄露攻击。 |
| [^40] | [One Scale, Four Texts: Gaps, Overlaps and Silent Repairs in Published Grading Schemes](https://arxiv.org/abs/2611.06569) | 该论文将糖尿病视网膜病变分级的四份权威文本统一形式化为基于24个发现变量的规则语言，并通过对14亿余个病例组合的穷举枚举首次严格证明：每份文本在任何解读下都存在无法归类的病例，且2003版与2017版之间存在无法互相映射的类别缺口。 |
| [^41] | [RAGAM-AI: A Hybrid Reinforcement Learning based Agentic approach powered by Generative AI for Edge Resource Management](https://arxiv.org/abs/2611.06554) | 提出RAGAM-AI混合框架，将强化学习与生成式AI结合构建智能体工作流，利用LLM作为奖励预言机解决奖励泛化问题，以优化多接入边缘计算单元的资源分配。 |
| [^42] | [Addition by Subtraction: Toward Mitigating Fragile Safety Alignment in Large Language Models](https://arxiv.org/abs/2611.06553) | 该论文提出Addition by Subtraction方法，通过在微调中逐步移除安全关键神经元并结合偏好优化，使安全对齐能力分散到模型的其他部分，从而缓解大语言模型安全对齐过度依赖稀疏组件的脆弱性问题，显著提升对剪枝等攻击的鲁棒性。 |
| [^43] | [Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment](https://arxiv.org/abs/2611.06543) | 提出在线评分准则自蒸馏（ORSD）方法，从同策略轨迹的跨轨迹行为差异中自生成动态自然语言过程评分准则，实现密集的令牌级信用分配监督，以增强以结果为中心的强化学习。 |
| [^44] | [From Admission to Retention: Predicting Durable Offspring in Quality-Diversity Search](https://arxiv.org/abs/2611.06522) | 该论文研究了质量多样性搜索中的信用分配问题，发现繁殖前可获得的信号（尤其是谱系深度）能够预测子代个体是仅短暂进入档案还是在多代之后仍被持久保留，且同一信号对准入与留存的影响方向往往相反。 |
| [^45] | [IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning](https://arxiv.org/abs/2611.06519) | 提出IDSISA方法，利用影响函数将易被遗忘的样本集中到少数分片，并通过未受影响分片的蒸馏恢复效用，在保持SISA精确删除保证的同时提升机器遗忘的效率和模型性能。 |
| [^46] | [EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL](https://arxiv.org/abs/2611.06518) | EmoReward 提出一种自进化奖励校准机制，通过从训练过程的奖励轨迹中提炼证据可靠性来自适应调整多奖励权重分配，解决了固定奖励难以捕捉情感意图且易引发奖励作弊的问题。 |
| [^47] | [ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events](https://arxiv.org/abs/2611.06508) | 提出了ElecLoad——一个上下文感知的智能体，能够利用网页通知、天气预警等多重事件的影响及其相互作用，自适应地组织电力负荷预测过程，从而提升预测的准确性。 |
| [^48] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^49] | [Tools Need Not Speak in Text: Latent Context Compression for Agents](https://arxiv.org/abs/2611.06500) | 提出Coca-Cola连续潜在上下文压缩模型，将工具观测压缩为嵌入智能体输入空间的潜在token，并通过动作保持训练对齐动作分布，突破了离散文本表示的限制，实现更高的上下文压缩率。 |
| [^50] | [FPTR: Feasibility-Preserving Transactional Refinement for Deadline-Constrained Joint Beam and Resource Allocation](https://arxiv.org/abs/2611.06491) | 该论文提出FPTR算法，通过“候选解必须完整、在阶段截止前通过可行性验证且严格改进目标才能替换当前最优解”的事务式机制，确保无线调度器在截止期约束下始终输出有效可行的波束与资源分配方案。 |
| [^51] | [Model Merging Does Not Require Uniform LoRA Ranks](https://arxiv.org/abs/2611.06486) | 该论文证明模型合并不要求各专家使用统一的LoRA秩，为不同领域专家分配各自的最佳秩可让平均法和基于效用的选择方法零成本地提升最多1.8个百分点的多领域准确率，而冲突解决类方法则表现不一致甚至可能损失4个百分点。 |
| [^52] | [ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs](https://arxiv.org/abs/2611.06481) | ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。 |
| [^53] | [Microdroplet Generator Surrogate Enhancement and Inverse Design Through an Uncertainty-Guided Framework](https://arxiv.org/abs/2611.06464) | 本文提出一种不确定性引导框架，利用神经网络集成的不确定性量化定位流型转变边界等关键区域，并通过物理驱动的针对性模拟数据增强策略提升代理模型精度，从而实现微液滴生成器的高效逆向设计。 |
| [^54] | [Within-Family EfficientNet Scaling under Fixed-Resolution Transfer Learning for Spectral-Domain Optical Coherence Tomography Retinal Disease Classification](https://arxiv.org/abs/2611.06452) | 该研究发现固定分辨率迁移学习中EfficientNet家族内按规模缩放并不能单调提升SD-OCT视网膜疾病分类性能，最小的B1以96.95%的准确率超越更大的B7模型，表明在医学图像迁移学习中盲目增大模型并非最优选择。 |
| [^55] | [AmbiHTMLBench: A Benchmark for HTML-based Visually Rich Document parsing](https://arxiv.org/abs/2611.06449) | 提出了AmbiHTMLBench基准，评估模型将真实世界文档恢复为保留内容、组织结构与视觉外观的可执行HTML的能力，以解决传统Markdown输出无法保留细粒度空间结构和渲染语义的问题。 |
| [^56] | [SUMONE: A Simple and Efficient Summarizer for Plug-and-Play Agent Memory](https://arxiv.org/abs/2611.06445) | SumOne是一个通过掩码重建进行自监督训练的独立摘要器，无需参考摘要或任务特定信号，即可为长时程智能体提供轻量级、即插即用的上下文压缩。 |
| [^57] | [EviSeek: An Evidence-Centric Agentic Framework for Embodied Active Information Seeking](https://arxiv.org/abs/2611.06434) | EviSeek 是一个无需训练的以证据为中心的智能体框架，通过将问题中蕴含的证据需求转化为明确的获取目标，引导具身智能体进行有目的的主动信息获取，在 ESI-Bench 上显著提升问答准确率并降低 token 消耗。 |
| [^58] | [D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding](https://arxiv.org/abs/2611.06425) | D3D 提出一种不依赖目标模型隐藏状态的独立扩散草稿模型，使草稿生成与验证可异步并行执行，在多个基准上平均实现约 5 倍的解码加速。 |
| [^59] | [Mind Control of LLMs via Closed-Loop Stimulation: Out-of-Distribution Robustness](https://arxiv.org/abs/2611.06418) | 受神经刺激领域闭环控制原理启发，本文将大语言模型引导表述为反馈控制问题，通过模型内部状态动态调节干预，从而在分布外场景下实现鲁棒的模型行为控制。 |
| [^60] | [A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks](https://arxiv.org/abs/2611.06412) | 本文针对训练数据可能被第三方污染的人脸识别系统，提出了三种实用的后门攻击防御技巧——一种在训练阶段检测被投毒的身份，两种在推理阶段过滤可疑输入，为系统提供商提供了可落地的安全加固方案。 |
| [^61] | [Zero-Shot Coordination among LLM Agents](https://arxiv.org/abs/2611.06409) | 该论文通过设计隔离特定协调挑战的最小化环境与简单通用脚手架，首次系统研究了独立开发的LLM智能体之间的零样本协调问题，发现即使最前沿的LLM智能体也难以协调，其主要原因在于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。 |
| [^62] | [GIST: Gibbs-Targeted Hierarchical Data Selection for Offline RLVR](https://arxiv.org/abs/2611.06395) | 该论文提出GIST方法，基于双向KL散度在参考策略处的精确分解，揭示出正确响应应聚焦其低概率token、错误响应应聚焦策略高频生成响应的高概率token这两条监督原则，并据此设计两级层级数据选择方法来优化离线RLVR的训练监督。 |
| [^63] | [MA-Mesh: Resilient Multi-Agent Collaboration via Role-based Self-Healing](https://arxiv.org/abs/2611.06389) | 提出了MA-Mesh框架，通过跨智能体反思和基于角色的故障转移两种自愈机制，在协作模式之上构建基于角色的自愈层，实现多智能体系统运行时的系统性弹性管理。 |
| [^64] | [From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents](https://arxiv.org/abs/2611.06382) | 该研究基于守恒、耗散和聚合三条水文定律构建简约流域模型CaRT，并通过AI代理辅助的假设驱动迭代细化得到CaRT+模型，在不使用日尺度神经网络的情况下于三大洲达到与深度学习相当的精度，证明物理定律约束的低维模型可以匹敌神经网络。 |
| [^65] | [LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution](https://arxiv.org/abs/2611.06381) | 提出 LegalEvolver 框架，通过评分准则引导的自我进化机制将查询启发式优化为种子，为法律判决预测生成更具法律信息量的检索查询，从而突破现有法律搜索智能体受限于查询质量的性能瓶颈。 |
| [^66] | [Does Context Compression Preserve Refusal Alignment?](https://arxiv.org/abs/2611.06380) | 论文发现基于编码器的上下文压缩虽然保留了语义内容，却会系统性削弱语言模型的拒绝对齐行为，并提出了一种无需重训练、在压缩表示空间中运行的轻量级推理时干预方法 Memory Steering，可将拒绝率恢复至接近基线水平。 |
| [^67] | [Corralling AI Risks with Grassroots Contracts](https://arxiv.org/abs/2611.06374) | 提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。 |
| [^68] | [Traffic-VL: Reinforced Vision-Language Models for Generalizable End-to-End Traffic Signal Control](https://arxiv.org/abs/2611.06353) | 该论文提出了SynTra-V逼真仿真数据集和Traffic-VL视觉语言模型，后者可直接从多视角监控图像与自然语言相位描述生成交通信号控制决策，无需中间结构化状态，实现了更可泛化的端到端交通信号控制。 |
| [^69] | [Today's Work, Tomorrow's Workforce: Incentivizing Learning in AI-Assisted Work](https://arxiv.org/abs/2611.06346) | 该研究表明，依赖AI辅助会侵蚀技能积累——新手可能陷入低技能陷阱、专家可能技能退化，而重视未来技能的策略性工作者即使AI更准确也会选择独立解题以维持练习。 |
| [^70] | [Prism: Bidirectional Self-Teaching for Unified Multimodal Models](https://arxiv.org/abs/2611.06343) | Prism 提出一种双向自教学后训练方法，将模型自生成的交错轨迹解耦为文本链与图像链并依据其可靠性构建训练目标，从而同时提升统一多模态模型的理解与生成能力。 |
| [^71] | [Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves](https://arxiv.org/abs/2611.06338) | 该论文研究了迭代反馈整定（IFT）在汽车电子节气门这类非线性对象上调参时的梯度偏差问题，发现静摩擦导致的两次实验间控制器状态不匹配会引入与第二次实验幅值成反比的误差，且IFT梯度仅在69%的低成本区域内为下降方向。 |
| [^72] | [Mass Producing Training Data with Surprising Side Effects](https://arxiv.org/abs/2611.06324) | 本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。 |
| [^73] | [EnzymeRSI: A dual-loop recursive self-improvement system for enzyme design](https://arxiv.org/abs/2611.06321) | EnzymeRSI提出了一种用于酶设计的双循环递归自我改进系统，通过外循环演进预测策略、内循环开发算法以及云实验室闭合物理反馈回路，在ProteinGym基准测试上显著提升了蛋白质适应度预测性能。 |
| [^74] | [Conformal Prediction for Distribution-to-Distribution Regression](https://arxiv.org/abs/2611.06320) | 本文针对分布到分布回归问题，提出了一种在正交基空间中运行的共形预测新框架，能够生成可解释的预测集，同时保持覆盖保证，克服了朴素应用分裂共形预测导致的预测集不可解释的缺陷。 |
| [^75] | [When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Link](https://arxiv.org/abs/2611.06307) | 对低轨卫星链路上BG-CFQS风险受控吞吐量预测的复现研究发现，其风险保证虽然可复现但依赖校准集划分方式——在实际部署所面临的连续时间划分下大多失效，且无论何种划分都无法实现低于0.15的风险预算。 |
| [^76] | [An Engineering Reading of the Bounded Halting Problem and Observable Properties of a Universal Machine](https://arxiv.org/abs/2611.06303) | 本文将停机问题工程化地改写为给定步数预算 T 的有界停机判定问题，通过带明确开销的对角线论证证明任何固定确定性检查器都无法在 T 的任意固定比例内判定所有实例，并将有界停机实例用作检验计算模型有限通用性与可观测性质的对抗性测试。 |
| [^77] | [STEK: Structural-Terrain-Enhanced Koopman-Inspired Cross-Channel Regularization for Retrospective Battery Multi-Target Estimation](https://arxiv.org/abs/2611.06299) | STEK框架通过质量代理条件门控将电池级生命周期(dQ/dV)地形与端点充放电曲线相融合，并引入Koopman启发的跨通道一致性正则化，实现了跨异构电池的回溯性多目标估计。 |
| [^78] | [Multi-Resident Data Association in Smart Homes via MDP Modeling and Behavior Cloning](https://arxiv.org/abs/2611.06296) | 该论文提出将智能家居多居民数据关联问题重新建模为马尔可夫决策过程（MDP）中的序贯决策任务，并通过行为克隆直接从专家演示中学习求解，有效解决了现有方法全局一致性差和轨迹碎片化严重的问题。 |
| [^79] | [A.S.E 2.0: Understanding How Repository Context Impacts the Security of LLM-Generated Code](https://arxiv.org/abs/2611.06294) | 该论文提出CASE-Bench基准测试，基于真实CVE构建145个仓库级任务，首次系统研究了仓库上下文构建方式如何影响LLM生成代码的安全性。 |
| [^80] | [The Split Protocol Decides the Result: Re-Evaluating Encrypted Traffic Classification](https://arxiv.org/abs/2611.06290) | 该论文揭示加密流量分类报告的高准确率主要由数据分割协议决定——同一模型在不同分割协议下宏F1可从0.992骤降至0.567，且关键划分轴取决于语料库的收集方式，并提出了一个无需训练模型即可识别正确划分轴的诊断方法。 |
| [^81] | [MonoLaDN: Language-Guided DeNoising for Monocular 3D Object Detection](https://arxiv.org/abs/2611.06282) | MonoLaDN在训练阶段利用冻结的视觉-语言模型生成关于目标遮挡、深度、尺寸和角度的结构化几何推理，并通过语言引导的去噪策略将其几何先验迁移到单目3D检测器中，部署时模型保持纯视觉推理，无需承担VLM的高昂推理成本。 |
| [^82] | [DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation](https://arxiv.org/abs/2611.06280) | DyPSI提出了一种动态物理感知框架，通过连续传感器位置场、共享函数式Tucker表示与联合扩散模型，从历史观测中同时生成未来物理场和时间连贯的传感器运动轨迹，突破了传统静态传感器布置的局限。 |
| [^83] | [Quo Vadis, Driving Scene Reconstruction? Occlusion-Aware Gaussian Densification](https://arxiv.org/abs/2611.06273) | OccSplat提出遮挡感知的高斯稠密化方法，通过高斯遮挡图识别缺乏高斯支撑的遮挡区域，并在3D表示中直接预测新的高斯基元以补全缺失的场景几何，从而提升前馈式3D高斯泼溅在驾驶场景新视角合成中的完整性。 |
| [^84] | [NextGAN: Towards Simple and Scalable GAN Training for Image and Video Generation](https://arxiv.org/abs/2611.06270) | NEXTGAN提出通过将标量评论家的输入梯度回归到冻结表示空间中配对特征间的位移来直接约束生成器的更新梯度，实现了简单且可扩展的GAN训练，无需辅助损失即可高效适配高达140亿参数的预训练图像与视频生成器，并取得先进的单步生成效果。 |
| [^85] | [Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature](https://arxiv.org/abs/2611.06265) | 该论文提出图符号指数（s_G）来量化正定几何先验与异质图之间的根本错配，并据此提出伪黎曼注意力网络（PRAT），通过带符号的平方伪距离注意力机制在伪黎曼流形上学习异质图表示。 |
| [^86] | [A Systematic Benchmark for Evaluating Multimodal Personalized Reasoning of Vision-Language Models](https://arxiv.org/abs/2611.06259) | 该论文提出了MPRBench——一个包含12个子任务、支持错误案例可解释性分析的系统性基准，用于评估多模态大语言模型的多模态个性化推理能力，并揭示出现有模型在对个性化概念识别不准确和对无关个人信息敏感方面的关键不足。 |
| [^87] | [HarnessFlow: MCTS-Guided Harness Optimization for LLM Agents](https://arxiv.org/abs/2611.06254) | HarnessFlow通过MCTS引导的精细化改进树搜索与编码智能体，在有限评估预算下自动优化LLM智能体的完整可执行执行框架，在六个基准测试中取得最高平均测试分数。 |
| [^88] | [Rubric Curriculum RL: Exploiting the Generation-Verification Gap in Non-Verifiable Domains](https://arxiv.org/abs/2611.06244) | RcRL 通过将对采样结果的成对偏好与评分标准课程机制相结合，在无需新数据、人工标注或更强教师模型的情况下，利用生成-验证差距在不可验证的开放式领域实现持续的性能提升并有效避免奖励作弊。 |
| [^89] | [A Definition of Catastrophic Performance](https://arxiv.org/abs/2611.06219) | 该论文提出了“灾难性性能”的正式定义，通过基线值和正交值两个参考值来区分次优结果与灾难性结果，从而为严格分析人工智能目标错位带来的灾难性风险提供了理论基础。 |
| [^90] | [CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving](https://arxiv.org/abs/2611.06204) | 提出CoSafe——一个模块化、与自动驾驶系统无关的协同式运行时安全保障框架，通过多车协作解决混合交通中由局部合理但联合不兼容决策引发的安全违规问题。 |
| [^91] | [Editing Large Language Models with Geometry-Aware Regularization](https://arxiv.org/abs/2611.06185) | 提出GEdit框架，通过低秩加对角（LR+D）因子模型捕捉知识表示的几何结构，利用低秩正则化保护核心知识子空间，并依据编辑难度自适应调整正则化强度，实现更精准的大模型知识编辑。 |
| [^92] | [AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment](https://arxiv.org/abs/2611.06170) | 本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。 |
| [^93] | [Reasoning-Aware Cross-Modal Association Unlearning in Multimodal Large Reasoning Models](https://arxiv.org/abs/2611.06168) | 该论文提出了多模态大型推理模型中的“跨模态关联遗忘”这一新问题，强调遗忘不仅要让目标信息从最终答案中消失，还需在中间推理过程中抑制指定的视觉到知识关联，防止敏感信息经由推理链泄露或被错误替代知识取代。 |
| [^94] | [CARM: Conformal Adaptive Risk Masking for Reliable Scientific Super-Resolution](https://arxiv.org/abs/2611.06151) | 提出CARM框架，通过维护异构更新速率的专家并利用非对称代理损失在线聚合阈值，在时间分布偏移下实现自适应空间风险控制，为科学超分辨率生成可靠的信任掩码。 |
| [^95] | [Grounding in the Dark: Text-Only Training for Zero-Shot Video Temporal Localization](https://arxiv.org/abs/2611.06149) | 提出一种仅用文本视频描述训练大语言模型（无需配对视频数据和视觉-语言预训练）来实现零样本视频时序定位的方法，通过LoRA在LLM2CLIP文本嵌入上适配Qwen3-8B，在推理时才首次接触视频即可取得优异的时序定位效果。 |
| [^96] | [SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving](https://arxiv.org/abs/2611.06147) | 本文揭示了MoE大模型服务中减少激活专家数量的加速方法会削弱模型对有害请求的拒绝能力，并提出SafeStart方法，在保留大部分加速效果的同时大幅恢复模型的安全性。 |
| [^97] | [ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search](https://arxiv.org/abs/2611.06131) | 提出自主跨模态搜索新范式ModalSearcher，使智能体能在直接推理、文本搜索、文图搜索与反向图搜之间自主决策，在多跳搜索中动态引入并整合视觉证据，并配套可扩展的ModalSearch-Pipeline构建交错多模态知识图谱以支持训练。 |
| [^98] | [From Handcrafted to Generative: RLVR-Optimized Icon Generation for Alexa+](https://arxiv.org/abs/2611.06123) | 该论文提出并部署了一套仅凭文本描述自动生成矢量化动画图标的生产级流水线，并将图标创建分解为多个阶段，重点利用 RLVR（可验证奖励的强化学习）优化其中决定整体质量的最关键阶段——图像生成，从而解决手工绘制图标耗时且难以规模化的问题。 |
| [^99] | [Ab Initio Accuracy Beyond the Cutoff: Distributed Message-Passing Equivariant Attention at Tens-of-Millions-of-Atom Scale](https://arxiv.org/abs/2611.06122) | 提出 Tessera 分布式运行时系统，通过空间填充曲线分片和固定张量形状克服通信瓶颈，首次将具有第一性原理精度的等变消息传递 Transformer 分子动力学模拟扩展到千万原子规模。 |
| [^100] | [The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models](https://arxiv.org/abs/2611.06116) | 该研究表明，即使底层大语言模型保持不变，智能体框架（脚手架）的选择也会显著改变模型在银行行为合规任务上的表现，且不同模型受框架的影响程度并不相同。 |
| [^101] | [Beyond Risk Cues: A Mechanism-Level Benchmark and Verifiable Debate for Scam Video Detection on the Web](https://arxiv.org/abs/2611.06110) | 本文提出机制级基准 MechScam（含6,614个视频、证据支持的诈骗解释及由真实证据构建的误导性论辩）与可验证辩论框架 ScamCourt，通过揭示诈骗运作机制并抵御误导性论辩，显著提升了网络诈骗视频检测的可靠性与鲁棒性。 |
| [^102] | [Reconciling Privacy with Provenance for Language Model Watermarks](https://arxiv.org/abs/2611.06100) | 提出红绿序列（RGS）本地差分隐私水印，通过将水印目标从追踪个体转向估计总体使用频率，在可证明地保护用户隐私的同时，仍能实现对语言模型使用情况的准确估计。 |
| [^103] | [Reasoning-Trace Overflow in Code Generation: An Empirical Study of Explicit Reasoning Models](https://arxiv.org/abs/2611.06095) | 该研究提出“推理轨迹溢出”概念，实证发现显式推理模型在较难的代码生成任务上常常将全部输出预算耗尽于推理而无法产出可用代码，而轻量级的推理时控制可以缓解这一问题。 |
| [^104] | [Drift-adaptive Representation Preservation for Domain-Incremental Learning](https://arxiv.org/abs/2611.06093) | 该论文提出将表征漂移作为遗忘风险的可测量替代指标，通过漂移加权的巩固损失和原型构建机制，在域增量学习过程中根据实际漂移程度自适应地分配表征保持资源。 |
| [^105] | [SysPilot: Cross-Library Performance Autotuning via Library Performance Contracts](https://arxiv.org/abs/2611.06078) | SysPilot提出“库性能契约”机制，通过规格化语言描述各库的资源需求与可观测信号，实现跨库性能自动调优，有效缓解多个独立库共同运行时的共享资源争用问题。 |
| [^106] | [UniCompress: From Multi-LoRA Experts to One Compact Model for Edge Intelligence](https://arxiv.org/abs/2611.06075) | UniCompress提出将多个多任务LoRA专家蒸馏整合为一个无适配器的INT2紧凑模型，通过GPTQ初始化和教师引导的量化恢复，在极低内存占用下保留完整视觉语言模型的多任务能力，适合边缘智能部署。 |
| [^107] | [Which One Is Sounding? Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation](https://arxiv.org/abs/2611.06074) | 该论文提出BiSID-AVS方法，利用双耳音频中耳间差异所编码的声源方向信息作为空间约束，在多个同类视觉候选中准确区分并分割出真正发声的实例，克服了传统方法依赖运动和跨帧关联的局限。 |
| [^108] | [Touch\'{e}World tech report : Towards Online Unified Camera--Hand Exploration in Generative Worlds](https://arxiv.org/abs/2611.06061) | TouchéWorld将手势驱动的手部交互统一集成到相机控制的视频世界模型中，通过因果适配与少步蒸馏实现块级在线生成，使用户能够在生成世界中同时进行相机导航与手部细粒度交互探索。 |
| [^109] | [Reglos: A Self-Evolving Personal Memory Agent with Recursive Self-Improvement](https://arxiv.org/abs/2611.06049) | Reglos是一个无需训练的个人记忆智能体，通过自进化记忆机制和递归自我改进来优化记忆线束，并利用异构模型系列的批评者筛选补丁以防止基准测试作弊，从而验证线束修改在目标失败案例之外同样有效，避免过拟合。 |
| [^110] | [Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription](https://arxiv.org/abs/2611.06043) | 本文提出Confucius4-R2T2，一个基于新概念“最长稳定前缀（LSP）”的稳定前缀流式语音识别模型，通过对Qwen3-ASR-1.7B进行后训练，使其在保持高准确率和低延迟的同时从不修改先前输出，非常适合需要即时行动的机器消费场景。 |
| [^111] | [Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels](https://arxiv.org/abs/2611.06027) | 该论文将仅基于软标签的贝叶斯误差估计技术从标准二分类扩展到公平分类等受约束分类问题，实现了对贝叶斯最优公平性-准确性权衡（帕累托前沿）的实用估计。 |
| [^112] | [How Unbalanced Must a Latin Rectangle Be?](https://arxiv.org/abs/2611.06003) | 本文将拉丁矩形的平衡性问题转化为网格中点的 $\ell_1$ 等距问题，通过模拟退火方法将 33 个最小不平衡度上界最多改进 39.4%，并借助计算机辅助分支定界法首次精确证明 3×9 矩形的最小不平衡度为 56。 |
| [^113] | [Intra-Architecture Source Attribution for Stable Diffusion: A New Task with Benchmark Dataset and Baseline](https://arxiv.org/abs/2611.06000) | 本文提出了新任务IntrAA-SD，旨在区分共享相同架构但经过不同微调的Stable Diffusion模型，并提供了首个包含14万张图像的基准数据集及基线方法。 |
| [^114] | [From Waste to Discovery: Recycling High-Scored Unexposed Candidates in Taobao Feed Recommendation](https://arxiv.org/abs/2611.05997) | 论文提出无需训练的召回阶段系统URED，通过回收工业推荐流水线中通常被丢弃的高分未曝光候选物品并在召回阶段重新注入，在淘宝信息流推荐中高效帮助用户发现新兴趣，避免了传统曝光提升策略因冷启动候选被排序器高估而损害参与度的问题。 |
| [^115] | [Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation](https://arxiv.org/abs/2611.05992) | Clips2Story提出了一种无需训练的LLM流水线，通过多模态描述符检索与关键词相关的视频片段，并自动编排成带有旁白叙述的可编辑故事板和连贯短视频。 |
| [^116] | [Ray-Tracing-Based Dynamic Channel Sounding With Autoregressive Wi-Fi CSI Prediction](https://arxiv.org/abs/2611.05970) | 本文提出一种将自回归CSI预测与SINR约束探测间隔选择相结合的动态信道探测框架，并利用基于Sionna射线追踪的环境建模和带旋转位置编码的Transformer来应对Wi-Fi MIMO系统中设备移动导致的信道老化与预测误差累积问题。 |
| [^117] | [Graph-Mediated Large Language Model Systems in the Agentic Era: Reasoning, Knowledge, Retrieval, Execution, and Coordination](https://arxiv.org/abs/2611.05953) | 本综述提出操作层面的“角色—位置”分类法，系统刻画了图在智能体时代大语言模型系统中于推理、知识与证据、检索与记忆、执行及协调五大环节中的功能、计算来源与输出用途。 |
| [^118] | [BAM: Bivariate Additive Models for Interpretable Tabular Prediction](https://arxiv.org/abs/2611.05940) | 提出 Bivariate Additive Models（BAM），将 GA²M 中的每个成对形状函数约束为非对称二元高斯的简单参数形式，在保持与标准 GAM 相当精度的同时，使每个交互项都可可视化为单峰等高线图并自动选择特征对，从而实现真正可解释、可审计的表格预测。 |
| [^119] | [Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification](https://arxiv.org/abs/2611.05937) | 本文提出FuseAMC轻量级决策融合模型，利用CNN在低信噪比下更鲁棒、高阶累积量分类器在高信噪比下更准确的互补特性，通过逻辑回归元组合器融合两者输出，实现了CPU高效的自动调制分类。 |
| [^120] | [Split-KV: Efficient Context Parallelism for Training Diffusion Drafters](https://arxiv.org/abs/2611.05926) | 提出Split-KV，一种精确的上下文并行方法，通过在本地保留键/值分片并将注意力通信限制为查询大小的张量，使通信量与上下文长度无关，从而高效训练扩散起草器。 |
| [^121] | [Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models](https://arxiv.org/abs/2611.05919) | 该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。 |
| [^122] | [Autonomous ETL Pipelines for Salesforce M&A Consolidation: A Reference Framework, Jitterbit Mapping, and Reproducible Simulation Study](https://arxiv.org/abs/2611.05918) | 本文提出了一个面向Salesforce并购场景的自主ETL参考框架，通过分区提取、声明式映射、质量门控、确定性及分块模糊匹配的主数据管理、受限加载和熔断器保护的变化数据捕获，解决了双源系统并行期间的数据去重、记录关系保留与持续同步难题，并将全流程映射至Jitterbit Integration Studio构件。 |
| [^123] | [Autonomous agents for electricity-system planning](https://arxiv.org/abs/2611.05911) | 该论文提出了一个用于电力系统规划的自主智能体框架，能够自动完成从方案提案、数值评估、反馈解读到重新设计和最终方案选择的全流程决策，并自动保留成本最低的可行规划方案。 |
| [^124] | [Do Saliency Maps Fail Dense Breasts? A Chance-Normalized, Confounder-Aware Audit of Explanation Fairness in Mammography](https://arxiv.org/abs/2611.05909) | 该研究首次系统审计了乳腺X线摄影AI分类器显著性解释图在致密型乳腺中的公平性，发现其定位癌症能力显著更差，约一半差距由病例构成（如钙化更多）解释，其余差距则指向模型本身对致密型乳腺病灶依赖不足，并建议报告调整前后按分类器划分的密度差距。 |
| [^125] | [Does Your Trial Simulator Know What the Drug Does? An Intervention-Blind Test of Interventional Validity on 7,207 Registered Trials](https://arxiv.org/abs/2611.05907) | 该论文首次将临床试验模拟器的“反事实能力”从主观论断转化为可测量的量化指标，通过构建隐去干预信息的干预盲参照模型，并基于7,207项注册试验构建的9,996个审计对比，将模拟器的干预技能定义为完整模型相对于干预盲参照所消除的误差。 |
| [^126] | [CHARM+: Cross-Hardware Attention with Re-Merge Multistream Mechanism](https://arxiv.org/abs/2611.05906) | CHARM+通过将顺序通信重组为多个层级以实现层级内并行和计算与通信重叠，并将跨硬件通信与硬件内通信解耦以消除带宽不均衡导致的同步开销，从而显著提升跨硬件架构下分布式注意力的整体性能。 |
| [^127] | [EchoKV: Self-Corrective KV Cache Eviction via Iterative Query Feedback](https://arxiv.org/abs/2611.05903) | EchoKV通过迭代查询反馈机制发现并修复KV缓存驱逐中早期工作区错误导致的决策偏差，在与LAQ相同的计算开销下，在证据分散的长上下文任务上取得显著改进。 |
| [^128] | [RECAP: Context-Aware Prediction Recycling with Request-Local and Persistent Memories for Multimodal Speculative Decoding](https://arxiv.org/abs/2611.05894) | 提出RECAP方法，通过区分请求本地的上下文记忆与跨请求持久的token级记忆、多长度键存储及熵置信度加权融合，实现无需训练的多模态投机解码预测回收，有效解决图像敏感解码状态下的预测混叠问题。 |
| [^129] | [CellCanon: An Agentic Cellular World Model for Prediction, Explanation, and Observation-Driven Updating](https://arxiv.org/abs/2611.05890) | CellCanon是首个智能体细胞世界模型，统一了单细胞表达预测、基于证据的解释和观测驱动的更新，在SciPlex3基准上显著超越基线方法，同时将语言模型token消耗降低超过98%。 |
| [^130] | [AI-Assisted Classification under Correlation Neglect and Trust](https://arxiv.org/abs/2611.05870) | 该论文提出包含相关性忽视与AI信任两个要素的人类决策行为模型，通过闭式相图划分AI辅助决策的互补、损害与自动化区域，并证明更强的AI可能因人类重复计算重叠证据而降低其决策准确性，而适当低水平的信任反而可以补偿相关性忽视。 |
| [^131] | [Auditable Unlearning for LLMs](https://arxiv.org/abs/2611.05866) | 该论文首次将“可审计性”引入为机器遗忘的第四个评估维度，发现现有十一种方法无一能同时满足全部四个维度，并提出基于 GapWiden 的两阶段方法来扩大遗忘集与保留集的审计统计量差异，使删除操作可被验证。 |
| [^132] | [Generative AI for urban planning: synthesizing land-use maps with satellite imagery via DualControlNet](https://arxiv.org/abs/2611.05859) | 本文提出DualControlNet，一种能根据文本提示联合生成高分辨率土地利用地图与卫星影像的双控制生成式模型，相比单模态基线将土地利用多样性一致性提升62.8%、建筑覆盖率一致性提升16.3%。 |
| [^133] | [Risk Control Certifies Rates, Not Magnitudes: An Audit of Selective Fetal Biometry](https://arxiv.org/abs/2611.05851) | 该论文审计了选择性胎儿头围测量流程，揭示风险控制仅能认证严重错误的发生率而非错误幅度——即使满足认证风险界限，系统仍可输出高达53毫米的巨大误差，且该失败被0.94的高完整度评分完全掩盖。 |
| [^134] | [PEARL: Prediction Error-Adaptive Regularized Learning for Selective Forgetting under Partial Preference Drift](https://arxiv.org/abs/2611.05842) | PEARL提出了一种预测误差自适应的正则化学习方法，通过仅在预测出错的特征方向上反转Gram矩阵更新来实现选择性遗忘，从而有效应对部分偏好漂移，克服了传统方法均匀遗忘和固定探索率的局限。 |
| [^135] | [Vision Paper: Why Distributed LLM Post-Training and Decentralized Alignment Are Now Essential](https://arxiv.org/abs/2611.05841) | 本文主张分布式学习应成为LLM后训练的一等范式，论证RLHF/DPO所依赖的偏好数据集中化假设不成立，并基于隐私主权、多元公平和鲁棒可扩展性三大支柱提出去中心化对齐的关键技术挑战与研究议程。 |
| [^136] | [RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection](https://arxiv.org/abs/2611.05838) | 提出RACE——一种无需真实执行的机器人策略检查点选择指标，通过基于状态条件阈值校准动作偏差并用累积乘积聚合评分，使检查点排序与真实部署性能更加一致。 |
| [^137] | [Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework](https://arxiv.org/abs/2611.05828) | 本文从复杂适应系统视角揭示了智能制造中个体自主性与集体协调这两种耦合机制，并提出了一种将目标驱动自主代理与递归组织结构相结合的递归自主代理（RAA）框架，用以实现生产系统中的智能决策与组织。 |
| [^138] | [A Hybrid Approach for Automated Security Pattern Recovery](https://arxiv.org/abs/2611.05823) | 本文提出SecPatRec，一种将ReAct风格LLM智能体与静态代码属性图分析相结合的混合方法，实现了从源代码仓库中自动恢复安全模式的角色检测、代码落地与交互序列识别。 |
| [^139] | [Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents](https://arxiv.org/abs/2611.05814) | 该论文提出咨询价值蒸馏方法，通过蒙特卡洛执行者模拟离线估计依赖执行者的人类咨询价值并蒸馏为轻量级预测器，使LLM智能体能够在有限人类决策预算下智能地决定何时咨询人类。 |
| [^140] | [When File Ingestion Misleads Language Models: Measuring and Mitigating Divergences in Office Documents](https://arxiv.org/abs/2611.05806) | 该论文揭示了LLM处理Office文档时的“提取分歧”——即提取器输出与用户所见内容不一致——可被攻击者利用为攻击面，并通过静态与动态分析识别出19个可被利用的OOXML通道。 |
| [^141] | [Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation](https://arxiv.org/abs/2611.05804) | 该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。 |
| [^142] | [GREP: Grounding Generated Room Evaluation in Principles](https://arxiv.org/abs/2611.05802) | GREP 提出了包含物理有效性、视觉连贯性和人类使用逻辑的三层评估理念，通过自主评估智能体发现场景问题并生成评分与报告，使生成房间的评估与人类判断达到 87% 的一致性。 |
| [^143] | [Consistency Is Not Understanding: Orbit Audits for Neural Field Weight Readers](https://arxiv.org/abs/2611.05798) | 该论文提出“轨道审计”方法——利用保持函数等价的隐藏神经元置换和正 ReLU 缩放变换来检验从神经场权重中读取视觉内容的模型，发现现有读取器在等价变换下准确率大幅下降甚至对相同输入给出不一致答案，证明其表面上的输出一致性并不等于对权重所表示函数的真正理解。 |
| [^144] | [DARE: Difficulty Analysis for Repair Enhancement](https://arxiv.org/abs/2611.05797) | 提出DARE框架，利用从基准测试产物中提取的151个静态特征预测编程修复任务的难度（定义为多个智能体的观察失败率），并结合七个特征组的贡献分析，为编程智能体提供任务特定的修复支持。 |
| [^145] | [FlyHash for LLM Agent Memory: An Empirical Validation and a New Application](https://arxiv.org/abs/2611.05791) | 本文首次将受果蝇嗅觉回路启发的FlyHash稀疏哈希应用于大语言模型智能体的记忆检索，并通过严谨的匹配位协议实验实证验证了其相比SimHash的稳健质量优势。 |
| [^146] | [Unsupervised Neural Operators for Rotating Relativistic Stars: High-Fidelity Equation of State Inference and Pulsar Spin Evolution at Scale](https://arxiv.org/abs/2611.05790) | 该论文开发了物理信息驱动的无监督神经算子，能够高效求解旋转中子星的轴对称广义相对论流体静力平衡方程，从而实现高保真度的状态方程推断以及大规模的脉冲星自旋演化群体合成。 |
| [^147] | [How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation](https://arxiv.org/abs/2611.05771) | 本研究比较评估了六种主流大语言模型模仿个体作者写作风格的能力，发现大多数模型模仿效果有限，而表现最好的 Gemini 3 Pro 实际上是靠大量复制训练文本中的句子和短语取得的表面成效。 |
| [^148] | [BobFlow: Towards Human-Agent Collaborative Data Science Using Dataflows](https://arxiv.org/abs/2611.05761) | 提出 BobFlow 系统，将数据流抽象与 ReAct 范式相结合，使智能体能够迭代构建和执行数据流来完成数据科学任务，同时让人类分析师可以直观地检查、引导和验证智能体的分析。 |
| [^149] | [PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?](https://arxiv.org/abs/2611.05759) | 该论文提出PAYWORTHY基准，基于1,494条真实自由职业任务发现，大语言模型只有在被告知智能体的能力限制时才能可靠预测哪些任务会被AI智能体成功完成（AUC达0.66-0.86），而若仅将其描述为通用智能体，预测效果则接近随机水平。 |
| [^150] | [Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex](https://arxiv.org/abs/2611.05753) | 提出由信号、领域、主张、维度四层组成的组合式知识架构，使组织解释连同其依据与立场持久保存，并通过真值维护网络随证据变化动态修订，从而显著提升检索性能。 |
| [^151] | [Scheduling Data to Compute: Hierarchical Data Pipelining for LLM Training on AMD GPUs](https://arxiv.org/abs/2611.05751) | 该论文将LLM训练中计算单元的数据等待时间视为核心瓶颈，提出通过编排AMD GPU上用于显式数据移动的硬件特性（数据加载、布局重排与GPU间传输），构建层次化数据流水线，使计算单元持续保持忙碌，从而提升训练效率与利用率。 |
| [^152] | [Pairing Expert Critique with LLM-Driven Refinement: An Iterative Approach for Curating Quality AI Evaluation Datasets](https://arxiv.org/abs/2611.05749) | 提出SME-C人机协作工作流程，通过专家轻量级点评结合LLM驱动的分析与精炼，迭代式筛选高质量AI评估数据集与评分标准，并在金融商业简报领域收集411位专业人士的3,853条点评验证了两种LLM精炼策略的有效性。 |
| [^153] | [Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa](https://arxiv.org/abs/2611.05744) | 该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。 |
| [^154] | [Learning Hierarchical Patch Splitting Policies for Faster Vision Transformers](https://arxiv.org/abs/2611.05742) | 该论文提出SPLIT方法，通过学习依据模型认为有意义的任务相关内容来分配图像token分辨率，替代基于像素统计的启发式方法，在减少30%token数量的同时保持与全分辨率ViT相当的准确率。 |
| [^155] | [PersonaCodeBench: Evaluating Coding Agents in Persona-driven Human-Interactive Sessions](https://arxiv.org/abs/2611.05739) | 该论文提出 PersonaCodeBench 基准，通过将用户人格特征转化为多轮编码会话中可验证的义务，评估编码智能体在满足个性化需求方面的能力。 |
| [^156] | [OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments](https://arxiv.org/abs/2611.05738) | 提出OdorSpaces——一个通过FDS模拟三维气味扩散来增强Habitat-Sim的物理仿真平台与基准，支持气体泄漏定位和早期火灾定位等视觉-嗅觉具身导航任务，其多模态强化学习策略优于多种基线方法。 |
| [^157] | [Do Language Models Know Which Constraints Belong Together?](https://arxiv.org/abs/2611.05726) | 该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。 |
| [^158] | [Self-Supervised Pretraining for Label-Efficient Facial Expression Recognition using Smart Glasses](https://arxiv.org/abs/2611.05711) | 该论文提出在OCOsense智能眼镜的未标注数据上，利用掩码自编码器对视觉Transformer进行自监督预训练，从而在标签极度稀缺的情况下显著提升面部表情识别性能，仅用5%的标注数据即可超越从零开始训练的全监督模型。 |
| [^159] | [Agents, Think Only When it Helps](https://arxiv.org/abs/2611.05709) | 研究发现LLM智能体蒸馏中思想监督对最终行动表现影响甚微，仅当思想信息未在相应行动中披露时才变得重要，因此应根据信息披露情况灵活选择是否蒸馏思想，以提升效率。 |
| [^160] | [From Type to Token: Petri Nets, Individual Token Philosophy, and Individualized Causal Effects](https://arxiv.org/abs/2611.05703) | 该论文提出以个体殊型哲学下的佩特里网作为殊型层面因果结构的自然表示，通过从贝叶斯网络到佩特里网的形式化转换并结合保形预测层，在不依赖分布假设且对处理效应异质性鲁棒的情况下实现个体化因果效应的量化。 |
| [^161] | [TACT: A Pragmatic Approach to Qualitative Analysis with an AI Council](https://arxiv.org/abs/2611.05702) | 该论文提出了TACT方法，通过由多个AI模型组成的可追溯委员会进行大规模主题分析，使软件团队在利用AI进行定性数据分析的同时，仍能保持对解释和判断的掌控权，并通过对抗性批评和不确定性标注进一步保证分析质量。 |
| [^162] | [How Do Artificial Agents Make Decisions in Light of Visual Data?](https://arxiv.org/abs/2611.05671) | 本研究通过两项互补实验评估了基于视觉-语言模型的人工智能体如何感知可视化中编码的不确定性并基于图像数据做出决策，发现 GPT-4o 在感知准确性方面表现最佳。 |
| [^163] | [Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps](https://arxiv.org/abs/2611.05664) | 该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。 |
| [^164] | [Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge](https://arxiv.org/abs/2611.05662) | 本研究揭示了“智能体即评审”即使自行收集了决定性证据，仍会因一条无证据价值的虚假权威备注而引发确认偏误，从而推翻多达40%的正确判定。 |
| [^165] | [Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings](https://arxiv.org/abs/2611.05652) | 该论文提出用层次结构度量和kNN纯度量化EEG嵌入中特质与状态信息的编码方式，发现大多数EEG特征呈现“宏观特质聚类、微观状态嵌套其中”的层次结构。 |
| [^166] | [Attributing Agent Failures to Their History Needs an Executed Counterfactual: Toward a Reporting Standard](https://arxiv.org/abs/2611.05650) | 该论文提出，将智能体失败归因于其历史的论断必须由实际执行的反事实干预来验证——冻结可执行的轨迹中间状态、仅改变历史并继续运行，同时报告经过测量的重放证书——并将其确立为一种新的报告标准。 |
| [^167] | [VIPBench: A Human-Aligned Benchmark for Voice Identity Perception in the Age of Voice Cloning](https://arxiv.org/abs/2611.05638) | 该论文提出VIPBench——首个与人类感知对齐的语音身份感知基准，包含1,290名听者对9,800对真实录音、语音克隆和声音融合样本的12万余次“相同或不同”判断，用于评估说话人表示模型与人类身份感知之间的对齐程度。 |
| [^168] | [Route to Remember: Adaptive Memory Selection for LLM Agents](https://arxiv.org/abs/2611.05636) | 该论文提出利用在线策略自蒸馏（OPSD）将用户特定事实等知识内化到模型参数中的权重记忆方法，并自适应地在文本记忆与权重记忆之间进行路由选择，使权重记忆作为长期事实存储在召回性能上可与基于文本的记忆系统相媲美，从而消除随历史累积而增长的查询开销。 |
| [^169] | [Spatio-Temporal Multi-Order Interaction for Video Recognition](https://arxiv.org/abs/2611.05633) | 该论文提出基于逐深度卷积的线性复杂度多阶时间建模与时空双重交互机制，以更低的计算开销实现更优的视频时空表示。 |
| [^170] | [Assurance-Conditioned Continuous Authorization for Autonomous AI Agents](https://arxiv.org/abs/2611.05632) | 提出了保证条件化持续授权（ACCA）框架，通过区分证据有效性与情境适用性、类型化依赖图的选择性重评估、权限纪元机制和效果时强制执行，使自主AI智能体的操作权限能够根据当前有正当依据的保证进行持续动态调整。 |
| [^171] | [ViBench 2.0: A Benchmark for Long-Horizon Software Building](https://arxiv.org/abs/2611.05620) | ViBench 2.0 是一个用于长周期软件构建的基准测试，通过 17 个随 241 个功能需求逐步演进的全栈 Web 产品和 544 个浏览器智能体测试计划进行严格评估，揭示了最强模型通过高达 98% 的测试计划却仅有最多 65% 的构建能交付完全可用产品的可靠性差距。 |
| [^172] | [Whiel: A Neurosymbolic Platform for Certified Relational Program Verification](https://arxiv.org/abs/2611.05612) | Whiel 是一个基于 Lean 认证的神经符号验证平台，它把以数据库实例为状态的命令式关系程序与 Hoare 风格推理、神经符号循环不变量发现及 Vampire 定理证明器相结合，实现了对 Datalog 等查询语言的包含/等价问题以及数据集成中完整性约束、模式映射与视图规范的可认证验证。 |
| [^173] | [Hybrid Spark: Adaptive Speculative Decoding That Lets the Content Choose the Drafter](https://arxiv.org/abs/2611.05611) | Hybrid Spark 提出在每一步同时运行基于模型和无模型两种起草者，并自适应地为每个请求选择更优的草拟链，解决了单一起草者无法适应不同内容类型、无模型起草者无法参与预算分配的问题。 |
| [^174] | [Verified but Vacuous: Three Failure Surfaces of LLM-Driven Verified Code Generation](https://arxiv.org/abs/2611.05590) | 论文指出评估LLM生成代码的“验证”不能只用单一指标，而应区分验证器接受、语法规范绑定与语义规范保真度三个独立层面，实证显示HumanEval上99.1%的验证器通过率中仅有1.9%真正绑定到了规范。 |
| [^175] | [Anduin: A sybil-resistant agentic marketplace reputation system](https://arxiv.org/abs/2611.05586) | 该论文提出Anduin系统，将链上支付历史建模为买卖双方之间的二部图，并基于买方可信网络相连的图路径设计了三种抗女巫攻击的评级算法，使操纵评级的成本大幅提升，从而解决自主AI智能体在链上市场中选择可信服务提供者的难题。 |
| [^176] | [BrainBind: Bridging Synchronized EEG and fMRI through Long-Form Naturalistic Video Representations](https://arxiv.org/abs/2611.05581) | 提出BrainBind框架，将同步记录的EEG与fMRI映射到冻结的长时自然视频嵌入空间中作为共同的视觉-语义参照，从而在长时自然观看场景下实现多模态脑信号的跨模态检索与解码。 |
| [^177] | [Multimodal Fusion Methods for Whole-Slide Images and Clinical Data in Inflammatory Bowel Disease](https://arxiv.org/abs/2611.05579) | 该论文提出了一个模块化框架，在统一条件下系统比较15种多模态融合方法，将全切片图像嵌入与临床数据结合，用于预测炎症性肠病患者的缓解结局，结果显示最佳融合方法在多数场景下优于单模态模型。 |
| [^178] | [Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations](https://arxiv.org/abs/2611.05577) | 该研究通过预注册实验发现，自监督模型学到的流体表征虽然具有线性可解码的“可读性”，但并不围绕物理量组织——JEPA表现不优于未训练网络，而普通自编码器反而在最慢潜在方向上突出涡量耗散，且添加VICReg正则化会削弱这种物理结构。 |
| [^179] | [Just the Facts: How Dialogues with AI Reduce Conspiracy Beliefs](https://arxiv.org/abs/2611.05572) | 与AI对话之所以能减少阴谋论信念，关键在于其提供的相关且论证充分的事实性反驳证据，而非融洽关系、讲故事或共情诉求。 |
| [^180] | [MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics](https://arxiv.org/abs/2611.05567) | 提出了MESA评估框架，通过五个评估套件和四个泛化维度（空间配置、物体实例、物体类别、子任务组合）实现语言条件机器人策略泛化能力的精确可重复测量，并配套开发了基于改进MimicGen的任务与演示自动生成流水线MESA-Gen。 |
| [^181] | [Discarded or Unreachable? Diagnosing and Repairing Spatial Relation Failure in Vision--Language Similarity](https://arxiv.org/abs/2611.05565) | 本文揭示了对比视觉-语言模型在左右空间关系判断上的失效存在两种成因——关系信息可被解码但相似度搜索无法触及的“访问失败”，以及池化丢弃信息而密集特征保留的“池化失败”，并提出一种零参数、查询条件化的空间矩方法从patch网格重新计算几何信息，用一个读取机制统一修复两类失败。 |
| [^182] | [How Finely Can Vision-Language Models See Biology? Bio-VQA: A Benchmark for Taxonomic, Morphological, Spatial and Individual Reasoning](https://arxiv.org/abs/2611.05559) | 该论文提出了Bio-VQA——目前最大的生物视觉问答基准，涵盖层次化分类推理、细粒度属性定位、定量空间推理和个体重识别四个方面，包含21.7万张图像上的186万个问题，并引入分级评分机制，系统评估了18个开源视觉-语言模型在生物理解上的能力。 |
| [^183] | [AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory](https://arxiv.org/abs/2611.05556) | 提出基于项目反应理论的AGORA聚合器，通过敏感度参数和有序评分阈值刻画各LLM裁判，无需人工标注即可从全部评估项中联合学习裁判可靠性，在仅有几百个样本的小规模评估集上显著优于Dawid-Skene等传统方法。 |
| [^184] | [Decision Models as Scalable Economic Surrogates that Capture Human Variation](https://arxiv.org/abs/2611.05553) | 本研究首次评估了决策模型在经济学中作为人类替代者的潜力，发现前沿决策模型在三种经济场景中比大语言模型更能再现人类回应的多样性，同时速度快约十倍，是大规模经济模拟与调查的更优选择。 |
| [^185] | [VirtualTrialHarness: Coordinating Agentic Operationalization of Virtual Trial Protocols Across Complex Clinical Schemas](https://arxiv.org/abs/2611.05551) | 提出了智能体框架 VirtualTrialHarness，通过为各工作流阶段选择可复用程序性技能并利用显式契约保持跨阶段决策一致，实现虚拟试验方案在复杂临床数据环境中的自动化执行，并配套发布评估基准 VTBenchmark。 |
| [^186] | [Whose Transparency is it Anyway? Understanding Generative AI Transparency Needs and Gaps Across Stakeholders](https://arxiv.org/abs/2611.05549) | 本研究通过对最终用户、AI开发者和政策制定者的全球调查发现，尽管各方对透明度的理解相似，但最终用户在透明度需求与实际可获得性之间的差距远大于开发者，凸显了当前生成式AI透明度实践中的利益相关者不平等问题。 |
| [^187] | [Iterated creation, grading, and revision of data science projects by language models](https://arxiv.org/abs/2611.05548) | 本研究通过“写作者-评分者”迭代反馈框架评估七个语言模型完成端到端数据科学项目的能力，发现三轮反馈修订使平均得分从59%提升至76%，且不同模型提升幅度差异显著（3至46个百分点）。 |
| [^188] | [Towards Efficient Whole Page Ranking with One-Pass Inference](https://arxiv.org/abs/2611.05542) | 该论文提出MIS（多物品打分）框架，一种感知前缀的NAR式方法，在单次推理的高效框架中引入序列上下文与位置感知打分，弥合了整页排序中自回归方法与非自回归方法之间的质量-效率差距。 |
| [^189] | [MA-SafeDiffuser: Safe Multi-Agent Planning with Diffusion Probabilistic Models](https://arxiv.org/abs/2611.05524) | MA-SafeDiffuser将控制屏障函数约束嵌入多智能体扩散规划的去噪过程，实现了具有可证安全不变性保证的集中式与分散式多智能体轨迹规划，并通过时变规范和活性CBF有效解决了局部陷阱与死锁问题。 |
| [^190] | [When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue](https://arxiv.org/abs/2611.05517) | 该论文提出“策略性澄清”任务并基于 SIMMC 2.1 构建了可控评测环境，同时提出轻量级的 BACE 动作选择方法，让 LLM 学会在澄清问题能缩小候选指代范围时才提问、否则直接回答，以缓解其过早作答的偏差。 |
| [^191] | [TSFM Meets LLM: Context as Covariates](https://arxiv.org/abs/2611.05515) | 该论文提出 CoCo 预测框架，让 LLM 将文本推理转化为数值化的预测与置信度协变量来引导 TSFM 骨干模型，融合两者优势进行时间序列预测，并提供利用 LLM 标注真实时间序列以生成多模态数据集的可扩展方法。 |
| [^192] | [SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories](https://arxiv.org/abs/2611.05512) | SlackBench是一个基于真实学术研究项目隐私保护重现构建的基准测试，用于评估AI智能体在需要对工作场所沟通与不断演变的代码仓库状态进行联合推理的现实协作研究与工程工作流中的能力。 |
| [^193] | [Objective-Conditioned Active Preference Learning for Multi-Objective Alignment](https://arxiv.org/abs/2611.05506) | 提出 OGC-DPO 主动学习框架，将（回复对，目标）组合视为独立查询并利用策略自身隐式梯度衡量信息量，主动选择最能扩展各目标梯度覆盖范围的比较样本，从而在无需外部奖励模型的情况下降低多目标对齐的标注成本。 |
| [^194] | [Can Large Language Models Review the Scientific Literature? A Test Against a 102-Article Benchmark](https://arxiv.org/abs/2611.05459) | 该研究以一篇含102篇文献的天气学综述为基准，对六种AI工具在文献综述全流程（检索、解读、综合、展望）进行评估，发现LLM虽有可取之处，但在文献发现能力以及区分实质性引用与顺带提及方面仍存在明显局限。 |
| [^195] | [Hybrid LLM-Deterministic Pipelines for Enterprise-Scale Code Migration at Bloomberg](https://arxiv.org/abs/2611.05457) | 彭博提出的混合 LLM-确定性流水线架构，通过将 LLM 限定在预计算上下文内进行有界文件转换并结合生产遥测反馈，在约 8,600 个企业仓库的两大迁移行动中实现了 729 次成功代码迁移。 |
| [^196] | [Towards Robust Long-Lived Agents: Benchmarking Experience Lifecycles in Evolving Environments](https://arxiv.org/abs/2611.05450) | 提出 ReScopeBench 基准，在 18 个交互式环境中区分结构经验与规则经验，并通过经验优先的任务流构建方法，系统评估智能体在环境演化过程中获取、复用与修订经验的完整生命周期。 |
| [^197] | [Jev with Foresight: Safe Language-Guided Autonomous Driving via a Language World Model](https://arxiv.org/abs/2611.05448) | 该论文提出Jev-AD系统，通过将基于概率回答的类型化决策模型Jev与能预测候选动作后果的语言世界模型相结合，实现了前瞻式安全决策，将模拟场景碰撞率从26.0%降至2.4%、真实NGSIM数据碰撞率从48.2%降至2.8%，同时能理解并实现乘客的自由形式驾驶指令。 |
| [^198] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^199] | [SurgTransfer: Multi-Step Latent Action Modeling for Action Transfer in Surgical World Model](https://arxiv.org/abs/2611.05443) | 本文提出SurgTransfer框架，通过多步潜在动作建模从无标注手术视频中学习上下文不变的潜在动作，从而在无需真实运动学数据或专家标注的情况下实现手术世界模型中的专家动作迁移。 |
| [^200] | [Constraint-Preserving Flow Matching for Spectral Graph Generation](https://arxiv.org/abs/2611.05436) | 提出了LaSFM方法，在谱图生成过程中通过在约束Stiefel流形上进行流匹配，同时保持拉普拉斯零空间约束、特征向量正交性和正特征值，并可扩展至属性引导的图生成。 |
| [^201] | [LLM-based NPCs in Video Games: A Systematic Mapping Study](https://arxiv.org/abs/2611.05427) | 该论文通过对42篇相关文献的系统性映射研究，首次从组件采用率、延迟、安全覆盖和可复现性四个维度对基于大语言模型的游戏NPC领域进行了量化刻画。 |
| [^202] | [CAST-Bench: Tracing Every Sound to Its Source in Multi-Shot Audio-Video Generation](https://arxiv.org/abs/2611.05422) | 提出CAST-Bench基准，将T2AV生成的正确性建模为“音视频契约”，通过9个声源类别共900个提示词，将多镜头音视频生成中的每个声音追溯至其预期声源，以诊断声音缺失、错误归因声源及镜头切换中声音丢失等问题。 |
| [^203] | [PEEK: One-Step-Look-Ahead Exposure Bias Correction for Diffusion Sampling](https://arxiv.org/abs/2611.05407) | PEEK提出了一种无需训练的确定性采样器插件，通过在每个去噪步骤中进行单步前瞻预测并插值，从更高信号的噪声状态获得更接近真实图像的预测，从而校正扩散采样中的曝光偏差并提升图像生成质量。 |
| [^204] | [When Does Sharing Work? Scaling a Shared SigLIP2 Across Image, Video, Audio, and Audio-Visual Representations](https://arxiv.org/abs/2611.05406) | 该论文提出Omni-Encoder，通过全参数共享的单一SigLIP-2 Transformer进行端到端联合训练，并借助一维卷积音频分词器和基于绝对时间戳参数化的多模态旋转位置嵌入，首次将图像、视频、音频和视听输入统一编码到共享的时空表征空间中。 |
| [^205] | [VenusGenome: Genome-Aware Masked Autoencoding for Microbial Genome Representation Learning](https://arxiv.org/abs/2611.05405) | 提出VenusGenome，一种基因分辨率的基因组感知掩码自编码器，通过高比例掩码与融合重叠群边界、链方向和基因位置的注意力机制，在超过170万个微生物基因组上预训练，以提升微生物基因组表示学习和功能预测能力。 |
| [^206] | [A Training-Free Socket for DNA in Frozen Language Models](https://arxiv.org/abs/2611.05401) | 该论文提出一种免训练的“套接字”方法，仅通过一次闭式岭回归将冻结的Evo 2 DNA特征接入冻结LLM的早期层表示，无需任何梯度训练即可在四个宿主模型上有效回答基因级别问题，真实DNA信号均显著优于打乱DNA对照。 |
| [^207] | [A Feasibility Condition for Confidence-Gated Tiered LLM Inference](https://arxiv.org/abs/2611.05394) | 本文提出了一个由无信号下界和完美信号上界界定的闭式可行性条件，通过架构成本比κ刻画了置信度门控分层LLM级联推理何时优于将所有查询直接发送到后备模型。 |
| [^208] | [Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning](https://arxiv.org/abs/2611.05393) | 该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。 |
| [^209] | [Uncertainty-Aware Resource Orchestration for Agentic AI Systems](https://arxiv.org/abs/2611.05392) | 本文提出了一种不确定性感知的资源编排方法，解决大语言模型驱动的智能体AI系统因运行时资源需求呈重尾、多峰且不可预测而导致的调度难题，避免资源供给不足或过度供给。 |
| [^210] | [Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents](https://arxiv.org/abs/2611.05391) | 提出名为“智能体执行框架感知”的信任分离设计，由宿主在关键操作前向模型提供经验证的最新运行状态快照，弥补模型对执行框架运行信息的“运行盲区”，使三种主流模型上的平均可观测策略遗憾降低96.9%。 |
| [^211] | [Who May Overrule the Agent? Evidence-Gated Authority in LLM Review Institutions](https://arxiv.org/abs/2611.05390) | 本文提出一种“证据门控”审查权限机制，要求LLM审查者必须先产生约束违反与合规修复的有效记录才能推翻上游智能体的提案，实验证明该机制能在大幅减少误改正确决策的同时保留大部分错误修复能力。 |
| [^212] | [AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture](https://arxiv.org/abs/2611.05389) | AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。 |
| [^213] | [RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On](https://arxiv.org/abs/2611.05163) | RoboTryOn 将人机转换类比为虚拟试穿——保持人类演示的动作不变，将人替换为由参考图像指定的目标机器人——从而从人类视频合成出视觉一致、保留任务与场景的灵巧机器人演示视频，为高自由度机器人提供可扩展的演示数据来源。 |
| [^214] | [FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video](https://arxiv.org/abs/2611.05062) | FloMo提出以3D场景流作为视频预训练迁移到机器人动作预测的中间表示，通过将场景流渲染为RGB并与视频共享潜在空间，使50亿参数的视频生成模型能够联合利用无动作标注的人类视频和机器人遥操作数据进行训练，从而在未见过的物体、运动基元和任务上实现更好的泛化。 |
| [^215] | [ScamWorld: An Interactive Environment for Evaluating Web Agents Against Online Fraud](https://arxiv.org/abs/2611.04879) | 本文提出 ScamWorld，一个通过在多场景、多机制、多语言下重建真实诈骗网站，从而对网络智能体识别与应对在线诈骗能力进行受控、可重复评估的交互式环境。 |
| [^216] | [AgenticFinBench: Benchmarking Financial Agents on Diverse Tasks from Real-World Requests](https://arxiv.org/abs/2611.04813) | 该论文提出了 AgenticFinBench——一个涵盖 8 个类别、2 种语言、4 个复杂度级别、共 230 个真实世界金融任务的智能体基准，并创新性地引入了结合问题特定评分标准与两阶段幻觉审计的无参考评估框架，从而更贴近实际、更准确地衡量金融智能体的真实能力。 |
| [^217] | [Signals: Continuous Diagnostics of AI Agents](https://arxiv.org/abs/2611.04507) | 本文提出Signals系统，通过包含45种问题类型的分类体系对AI智能体执行轨迹进行持续诊断，将发现的问题归类为工程问题并评估严重程度和提供具体补救措施，在公开数据集上达到91.43%的F1分数。 |
| [^218] | [Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs](https://arxiv.org/abs/2611.04376) | 该研究揭示了MoE大模型FP4强化学习失效的根源在于路由器的离散专家选择机制，证明了必须在训练与推理两侧对称地应用相同的恢复技术，才能实现有效的端到端FP4强化学习。 |
| [^219] | [Predicting protein structures at any folding stage from sequence](https://arxiv.org/abs/2611.04287) | FoldScape 是一个以序列为条件的深度学习框架，可在任意折叠阶段生成蛋白质构象集合，无需推理时进行针对特定蛋白质的分子动力学模拟。 |
| [^220] | [Bidirectional Centered Score Matching for Discrete Diffusion Distillation](https://arxiv.org/abs/2611.04279) | 提出双向中心化得分匹配（BCSM），一种面向离散扩散模型的高效位置内蒸馏目标，将CSM的O(|V|²)复杂度降低至O(K)，并通过跨时间步转移对齐在少步蒸馏中实现更优的质量-效率权衡。 |
| [^221] | [Separating Common and Unique Directions for Model Merging](https://arxiv.org/abs/2611.04108) | 提出CUSD，一种无需数据的模型合并预处理方法，通过将任务向量分解为跨任务共享的公共方向分量和任务独有的独立分量并分别处理，从而提升算术合并方法的性能。 |
| [^222] | [Exploiting Textual Semantics for Robust Cross-View Object Correspondence](https://arxiv.org/abs/2611.02356) | 提出TeSCo框架，利用目标文本描述中视角和外观不变的语义信息来补充视觉线索，从而显著提升复杂场景下跨视角目标对应的鲁棒性。 |
| [^223] | [Do agents need to sleep?](https://arxiv.org/abs/2611.02216) | 该论文借鉴睡眠的选择论“回响-共振-再归一化”（RRR）理论，将睡眠的三个阶段映射为智能体数字记忆架构中可行的操作，并提出可证伪预测：随着记忆规模增长，选择论式记忆巩固将优于指令论式方法。 |
| [^224] | [Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls](https://arxiv.org/abs/2611.02119) | 本研究构建了基于扩散模型的条件反事实审计框架，并通过匹配的阳性对照严格验证了其对黑色素瘤分类器概念级解释的审计能力，证明该方法在恢复目标概念方面优于传统的概念激活向量方法。 |
| [^225] | [Wave Physics as Representation: Tensor-Structured Neural Reconstruction for Ocean Acoustic Fields](https://arxiv.org/abs/2611.01680) | 该论文提出从波动物理中提取结构化表示并用张量结构神经网络学习其组件的新路线，实现了既具物理可解释性、又对不确定环境信息鲁棒的海洋三维声场稀疏重建。 |
| [^226] | [Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models](https://arxiv.org/abs/2611.00935) | 该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。 |
| [^227] | [VCoT-SQL: Enhancing Text-to-SQL Generation through Dual Chain-of-Thought Validation Training](https://arxiv.org/abs/2611.00815) | VCoT-SQL框架通过中间步骤验证的思维链监督训练SQL生成器和SQL验证器两个互补模型，使小型开源模型在Text-to-SQL任务上超越前沿闭源模型。 |
| [^228] | [Behavior Graphs: Unsupervised Graph Structures for Interpreting Robot Deployment Data](https://arxiv.org/abs/2611.00405) | 该论文提出“行为图”这一无监督方法，将机器人策略rollout数据压缩为一张单一、可解释的有向图，通过识别相似行为模式、按时序连接并与结果关联，显著降低了部署数据的分析难度，使研究者能更高效地指导数据收集决策。 |
| [^229] | [Quantifying Local and Global Fidelity of Nonlinear Embeddings](https://arxiv.org/abs/2611.00290) | 提出经机会校正的 DLP 曲线，从中导出局部保真度 L 与全局保真度 G，并配合 LG 反转检验和实体保真度 E，以可验证的方式量化并判定降维嵌入保留局部与全局结构的能力。 |
| [^230] | [AuroraWorld: Structured 4D World Modeling for Exo-to-Ego Video Generation in Robot Manipulation](https://arxiv.org/abs/2611.00177) | 提出AuroraWorld，一种结构化4D世界建模方法，能够从大规模外部视角机器人操作视频中合成物理一致、且对视觉运动策略学习有用的第一人称视角视频。 |
| [^231] | [When Disclosure Is Compulsory: Evaluating Ride-Hailing Rebalancing Policies with Self-Interested LLM Driver Agents](https://arxiv.org/abs/2610.41132) | 该研究首次在欧盟《平台工作指令》强制披露算法规则的背景下，用以自身收益最大化为目标的自利LLM司机智能体评估网约车反应式与预测式两类调度政策，发现规则公开后司机更遵循平台调度指令、拒绝调度的比例显著下降。 |
| [^232] | [Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection](https://arxiv.org/abs/2610.39076) | Tactus是一种织物集成的感知手套，通过超薄3D针织外形、策略性布置的电容触觉传感器和融合IMU的抗遮挡姿态跟踪三大设计，实现了自然人类操作演示中精确接触力与手部姿态的高质量数据采集。 |
| [^233] | [MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics](https://arxiv.org/abs/2610.36513) | 该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。 |
| [^234] | [AutoEnergyDesign: An LLM-Driven Agent for Automatic Energy System Design Optimization](https://arxiv.org/abs/2610.35160) | 该论文提出AutoEnergyDesign框架，利用具备多类型显式记忆和git式自动回退机制的LLM智能体，通过仿真反馈进行上下文学习来自动优化建筑能源系统设计，从而大幅减少黑盒优化所需的仿真评估次数。 |
| [^235] | [SPARTA: A Sparse Ternary LLM Accelerator with Adaptive Attention and Unified Datapath for Long-Context Edge Inference](https://arxiv.org/abs/2610.34824) | SPARTA是一种面向边缘设备长上下文推理的稀疏三值LLM加速器，通过相位感知稀疏数据流、统一计算架构和自适应两阶段注意力引擎（页级Top-k与令牌级Top-p剪枝），显著降低KV缓存流量并提升推理效率。 |
| [^236] | [OS-Wizard: Scaling Computer-Using Agents to Specialized Software via Knowledge-Guided Exploration](https://arxiv.org/abs/2610.33907) | OS-Wizard是一个自动化轨迹收集框架，通过教程引导探索与混合GUI-代码动作空间为专业桌面软件收集了10.9K条高质量交互轨迹，基于此微调的开源模型显著提升了计算机使用智能体的性能。 |
| [^237] | [Local Entropy Fluctuations for Black-Box LLM-Generated Text Detection](https://arxiv.org/abs/2610.33767) | 该论文提出新指标“相对峰值显著率”，利用人类写作信息流均匀而机器生成文本存在尖锐局部熵波动这一结构性差异，解决了黑盒场景下现有检测器在指令条件文本上灾难性失效的问题。 |
| [^238] | [Isharah-Selfie: Continuous Sign Language Recognition Dataset for One-handed Signing](https://arxiv.org/abs/2610.33200) | 提出了首个使用智能手机前置摄像头在无约束自拍条件下采集的大规模阿拉伯手语连续识别数据集Isharah-Selfie，专门针对现实移动场景中的单手手语识别。 |
| [^239] | [Entity-Aware Evaluation of OCR Systems for Historical Document Transcription](https://arxiv.org/abs/2610.33004) | 该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。 |
| [^240] | [HyScale: Scaling Long-Context Training for Learned Sparse Attention](https://arxiv.org/abs/2610.32057) | HyScale 通过无需分数表的实时精确 top-k 选择方法和前向选择重放机制，将学习型稀疏注意力的百万词元长上下文训练速度提升了 2.07 倍。 |
| [^241] | [Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion](https://arxiv.org/abs/2610.31323) | 提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。 |
| [^242] | [Asymmetric On-Policy Distillation Learns In-Context Exploration](https://arxiv.org/abs/2610.30105) | 提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。 |
| [^243] | [From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake](https://arxiv.org/abs/2610.29152) | 该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。 |
| [^244] | [A latent control model for realistic rodent motion](https://arxiv.org/abs/2610.27279) | 提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。 |
| [^245] | [RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development](https://arxiv.org/abs/2610.26826) | 该论文提出了RefCraft基准，包含300个将自然语言指令与参考网页截图相结合的多模态网页开发任务，用于评估智能体选择性地解读视觉参考并迭代优化生成结果的能力。 |
| [^246] | [X-Fly: Learning to Reach and Stop in Full-Chunk Aerial Vision-Language Navigation](https://arxiv.org/abs/2610.21450) | X-Fly提出了一种分阶段训练框架，通过恢复DAgger标注恢复动作块与停止指令，再用Flow-GRPO基于闭环结果联合优化分块流控制器和伯努利STOP策略，从而解决全动作块空中视觉语言导航中偏离专家状态导致的恢复失败和无法停止问题，同时将每个路径点的推理成本从195毫秒降至42毫秒。 |
| [^247] | [Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference](https://arxiv.org/abs/2610.21257) | 该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。 |
| [^248] | [Revisiting the 16x Pretraining Compute Gap between Diffusion and Autoregressive Language Models](https://arxiv.org/abs/2610.17590) | 本研究采用现代扩散语言模型训练实践（精确似然评估、块扩散训练、编码器-解码器架构）重新审视扩散语言模型的缩放定律，通过约1700个模型的大规模实验发现，扩散语言模型与自回归语言模型之间所谓的16倍预训练计算差距远小于此前声称的水平。 |
| [^249] | [DTA-GT: Direction- and Topology-Aware Graph Transformer for Neural Network Representation Learning](https://arxiv.org/abs/2610.16822) | 提出DTA-GT，一种方向与拓扑感知的图Transformer，通过在节点初始化、谱编码、注意力交互和特征更新全流程中保持有向计算语义，显著提升了神经架构搜索中精度与延迟预测的性能。 |
| [^250] | [RetryBench: Separating Retry Permission from Recovery Outcomes in GPT Tool Use](https://arxiv.org/abs/2610.15884) | 该论文提出RetryBench基准，将GPT工具使用中“虚假重试许可”（即在契约不支持时重试超时工具调用）这一错误与实际恢复结果分离，发现即使在受保护条件下能成功恢复，模型也无法可靠地拒绝不受支持的重试。 |
| [^251] | [SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models](https://arxiv.org/abs/2610.06598) | SimForcing通过潜空间运动蒸馏和多块仿真条件化（带条件丢弃）机制，将仿真中的运动先验迁移到真实域机器人世界模型中，同时规避了外观差异和仿真预测不准确带来的负面影响。 |
| [^252] | [MASBench: Benchmarking LLM-based Multi-Agent Collaboration under Partial Observability](https://arxiv.org/abs/2610.04672) | 该论文提出了MASBench，首个在部分可观测性约束下系统评估基于大语言模型的多智能体系统协作机制的基准测试，填补了现有基准假设全局可观测性的空白。 |
| [^253] | [MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations](https://arxiv.org/abs/2610.02480) | MEA通过Proposer智能体自动选择和配置解释工具、Actor智能体以忠实性为目标进行端到端奖励优化，完全消除了使用机器学习解释方法的知识壁垒，能够跨表格、文本和视觉模态生成忠实的自然语言模型解释。 |
| [^254] | [Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies](https://arxiv.org/abs/2609.39640) | 仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。 |
| [^255] | [AutoBCI: Forecast-Guided Agentic Neural Architecture Discovery for EEG-Based Brain-Computer Interfaces](https://arxiv.org/abs/2609.35456) | AutoBCI通过设计者智能体自动生成并改进EEG解码架构，同时由预测者智能体根据早期学习曲线预测架构的最终性能以筛选优秀候选，从而在情绪识别、运动想象和睡眠分期等14个EEG数据集上实现高效的脑机接口神经架构自动发现。 |
| [^256] | [Reliability Engineering for AI Systems: Challenges, Methods, and Directions](https://arxiv.org/abs/2609.35316) | 该论文将传统可靠性工程方法（如失效定义、FMEA、加速测试、现场监控与可靠性增长）适配到AI系统，提出四级失效诊断框架，主张超越平均基准准确率来系统性地保障AI系统整体的可靠性。 |
| [^257] | [STRIDE: Automated Evaluation of Text-to-Trajectory Alignment across Diverse Contexts](https://arxiv.org/abs/2609.34799) | 该论文提出首个评估场景文本描述与行人轨迹之间情境对齐的自动化框架STRIDE，通过源自社会学理论的VRDST评估协议，解决了传统依赖真实人类数据的评估方法在多样化情境下无法扩展的难题。 |
| [^258] | [Understanding Generalization Requires Universal Induction](https://arxiv.org/abs/2609.34458) | 论文指出经典统计理论因依赖无法证成的手工归纳偏置而无法解释通用AI的泛化成功，并提出将所罗门诺夫归纳相对化到信息视角、偏向能访问所有既有信息的短程序，从而以普适归纳重新框架化归纳偏置问题。 |
| [^259] | [Evolving Support Priorities in Empathetic Reinforcement Learning](https://arxiv.org/abs/2609.34249) | 提出 CARE 方法，通过在每轮对话中动态调整认知、情感与主动共情三个维度的权重及细粒度评估标准，生成情境自适应的奖励准则，解决了共情强化学习中支持优先级随对话演化而奖励规范固定不变的根本错配问题，并在多个基准上取得最先进性能。 |
| [^260] | [Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning](https://arxiv.org/abs/2609.29711) | 该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。 |
| [^261] | [PolicyMem: Geometric Policy Memory for LLM Governance](https://arxiv.org/abs/2609.13734) | PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。 |
| [^262] | [MAIA: Multi-Agent Intent Articulation for Requirement Discovery in Art Commissions](https://arxiv.org/abs/2609.12097) | MAIA是一个多智能体系统，通过苏格拉底式提问和“验证优先于发明”规则，帮助艺术委托中的外行客户将模糊的情感意图表达为可验证的视觉术语简报，实验证明其在认知支持方面显著优于基线。 |
| [^263] | [LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues](https://arxiv.org/abs/2609.03507) | 提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。 |
| [^264] | [CivBench: A Long-Horizon Benchmark for Tool-Mediated Agents in Civilization VI](https://arxiv.org/abs/2609.02459) | CivBench是一个基于《文明VI》游戏和MCP协议的开源长时程基准测试，通过300多回合、数千次工具调用的复杂任务评估语言模型智能体，并创新性地提出主动监控率和RAG@10两个接口级指标来刻画智能体行为。 |
| [^265] | [Using profiles of cognitive capability to assess AI suitability for workplace tasks](https://arxiv.org/abs/2608.25623) | 该论文提出一种基于共享核心认知能力为AI智能体与任务共同画像的流水线，通过结合基准测试推断的智能体能力与领域专家评估的任务需求权重，动态估算AI在领域、组织、角色或个人职责层面的任务适用性。 |
| [^266] | [Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems](https://arxiv.org/abs/2608.21095) | 本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。 |
| [^267] | [Agent Gym: A Framework for Continuous Evaluation and Evolution of LLM Agents Through Human-in-the-Loop Feedback](https://arxiv.org/abs/2608.15591) | Agent Gym提出了一种模块化、领域无关的框架，通过人在环反馈实现LLM智能体部署后的持续评估与演化，无需修改源代码即可进行行为修正。 |
| [^268] | [Fast and Memory-Efficient Wavelet Convolutions via I/O-Aware Reformulation](https://arxiv.org/abs/2608.10805) | 本文通过I/O感知的代数重构（包括片上重计算、级联折叠和权重缩放融合），显著减少了小波卷积中的高带宽内存流量，实现了快速且内存高效的WTConv实现。 |
| [^269] | [MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection](https://arxiv.org/abs/2608.10459) | MD-ProTector通过在编码器嵌入空间中为每个类别设置多个可训练原型并采用原型定位损失，为LLM生成文本检测提供了对类内多样变化的细粒度建模能力。 |
| [^270] | [TraceCAD: Trace-Guided Repair for Agentic CAD Generation](https://arxiv.org/abs/2608.03062) | TraceCAD通过将需求、建模步骤、失败证据与修复结果链接为持久轨迹状态并配合可复用的技能记忆，实现对LLM生成的CAD程序的高效诊断与局部化修复，显著减少几何退化并大幅降低智能体调用开销。 |
| [^271] | [Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks](https://arxiv.org/abs/2607.28685) | 本文对四个主流智能体安全基准进行效度审计，发现其评估指标存在系统性缺陷（如恒判为正的策略即可获得高分）、不同基准对同一批模型的排名互相矛盾，且安全基准之间或安全与能力之间表观的负相关很可能只是小样本导致的统计伪象。 |
| [^272] | [From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs](https://arxiv.org/abs/2607.26853) | 该研究将Funder人格三元框架适配到大语言模型中，提出了一个发现、控制和验证模型内部人格相关表征的框架，以机制性视角揭示这些表征如何跨情境地塑造社会任务中的行为表现。 |
| [^273] | [A Roadmap to Impactful Pluralistic Alignment Research](https://arxiv.org/abs/2607.22305) | 本文通过审计主要前沿AI实验室发现多元价值对齐研究尚未真正影响模型的训练与评估，并针对论证缺乏实证、实践标准不明等三大障碍提出了推动其落地的研究路线图。 |
| [^274] | [OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories](https://arxiv.org/abs/2607.17082) | 该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。 |
| [^275] | [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097) | 该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。 |
| [^276] | [Revisiting data-driven dynamic security assessment with a tabular foundation model](https://arxiv.org/abs/2607.16031) | 该论文提出使用表格基础模型（TFM）通过上下文学习进行电力系统动态安全评估，仅需单一模型即可同时评估多个偶然事件，无需逐个偶然事件训练分类器或进行超参数优化，并可泛化到未见过的偶然事件。 |
| [^277] | [Token Time Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2607.14106) | 本文提出词元时间连续扩散（TTCD），一种在连续空间中运行、确定性映射噪声到词元并引入逐词元时间概念的新型扩散语言模型，在高加速比下性能优于离散扩散模型。 |
| [^278] | [KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling](https://arxiv.org/abs/2607.09153) | KV-PRM通过直接读取大语言模型生成阶段产生的KV缓存来替代从头重新编码轨迹文本，将过程奖励模型的评分成本从O(L²)降至O(L)，实现了多智能体测试时扩展中的高效奖励评分。 |
| [^279] | [From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier](https://arxiv.org/abs/2607.07779) | 本综述系统梳理了LLM驱动的形式化数学进展（涵盖训练策略、推理时搜索与智能体工作流三大方向），并强调从求解明确定义的问题迈向真正的数学研究，关键在于形式化证明之外还需捕捉问题的本意并融入专家判断。 |
| [^280] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^281] | [Hyperparameter Transfer in Graph Neural Networks](https://arxiv.org/abs/2607.05017) | 本文为使用SGD、Adam和AdamW训练的图神经网络提出并验证了一种超参数迁移参数化方法，使模型在宽度和深度增加时能够实现稳定的特征更新、学习率迁移及性能提升。 |
| [^282] | [PixCon: Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation](https://arxiv.org/abs/2607.03068) | PixCon 通过添加一个仅存储预测与真实标签一致处的干净有标签特征、无需置信度阈值的类别级记忆库，并以多正样本对比损失训练编码器-解码器，从而以极简方式显著提升了基于基础模型的半监督分割性能。 |
| [^283] | [Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces](https://arxiv.org/abs/2607.00481) | 本文提出SMT黑盒攻击框架，通过构建模拟合法审核审计工作流的多轮轨迹，利用函数调用环境中可信控制逻辑与不可信数据边界模糊的结构性漏洞，成功越狱函数调用大语言模型。 |
| [^284] | [EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling](https://arxiv.org/abs/2606.26578) | 提出EvoOptiGraph框架，通过将优化模型表示为图结构并利用弱点驱动的协同进化，自动生成结构多样的训练数据，从而提升大语言模型在优化建模任务上的性能。 |
| [^285] | [Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games](https://arxiv.org/abs/2606.25997) | 本文提出通过变量边界收紧技术来加速求解多人不完美信息博弈中纳什均衡的二次约束规划，从而提升非凸求解器的计算效率。 |
| [^286] | [Large-Language-Model Discovery of Quantum LDPC Codes through Structured Concept Evolution](https://arxiv.org/abs/2606.24808) | 该论文提出结构化概念演化方法，让大语言模型通过修改有限群、原型图等构造要素自主演化量子LDPC码的构造程序与搜索策略，在特定参数区间超越了已发表的基准结果，且最佳模型在两个优化目标上均优于随机变异和爬山法。 |
| [^287] | [Patients With Personality: Realistic Patient Simulation through Controlled Diversity and Selective Disclosure](https://arxiv.org/abs/2606.17441) | 提出了基于六维HEXACO人格模型的患者模拟框架PWP，通过显式人格参数化实现对对话风格、配合度和信息披露的细粒度控制，其对话真实性经临床医生评估与真实医患对话相当，且远超以往模拟器。 |
| [^288] | [Phantoms and Disclosures: A Statistical Framework for Auditing Privacy in Synthetic Data](https://arxiv.org/abs/2606.16952) | 本文提出了一个可定制的实证审计框架，通过区分“真实泄露”与“幻影泄露”，并利用训练/保留集划分与严格的统计假设检验，来审计合成数据是否符合零学习或差分隐私等严格隐私基线。 |
| [^289] | [Bridging the Morphology Gap: Adapting VLA Models to Dexterous Manipulation via Intent-Conditioned Fine-Tuning](https://arxiv.org/abs/2606.12109) | 提出InDex意图条件化微调框架，利用由演示重定向导出的归一化开合度抓取意图信号，将预训练VLA模型从平行夹爪适配到多指灵巧操作，在四个仿真任务中达到85.8%的平均任务成功率。 |
| [^290] | [Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code](https://arxiv.org/abs/2606.11817) | 该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。 |
| [^291] | [Ekka: Automated Diagnosis of Silent Errors in LLM Inference](https://arxiv.org/abs/2606.04594) | Ekka通过差分对比目标与参考LLM服务框架的中间执行状态，将静默错误诊断转化为差分调试问题，实现了自动化的根因定位，在真实错误基准上达到80%的pass@1诊断准确率。 |
| [^292] | [Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs](https://arxiv.org/abs/2606.02628) | 量化开源大语言模型的中间层隐藏状态中存在近似线性的真实性信号，仅用单个中间层的简单线性探针即可高效检测幻觉（AUROC达0.904–1.000），显著优于采样式检测方法。 |
| [^293] | [Emergent Transfer of a Physics Foundation Model from Simulation to Laboratory Turbulence](https://arxiv.org/abs/2606.01470) | 该论文验证了物理基础模型能够从数值模拟涌现式地迁移到真实实验室湍流，在瑞利-泰勒不稳定性这一长期存在模拟-实验差异（增长率相差约三倍）的难题上取得了突破。 |
| [^294] | [MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs](https://arxiv.org/abs/2605.29300) | 该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。 |
| [^295] | [Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling](https://arxiv.org/abs/2605.26969) | Recon方法以动作重建的保真度作为推理轨迹质量的评分标准，从而筛选出真正具有预测能力的推理而非事后合理化，使用户建模更贴近用户真实的决策过程。 |
| [^296] | [More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647) | 本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。 |
| [^297] | [AvalancheBench: Evaluating Enterprise Agent-Driven Semantic Query Processing Through Latent World Recovery](https://arxiv.org/abs/2605.24183) | 该论文提出了AvalancheBench，一个从潜在世界规范生成观测数据、通过涵盖文本、图像和文档的58个多部分问题评估数据智能体端到端语义查询能力的多模态企业分析基准，并首次揭示了错误经合法依赖关系传播、不当复用中间结果导致无关问题失败的“雪崩效应”。 |
| [^298] | [Designing Benchmarks for Knowledge Work](https://arxiv.org/abs/2605.23262) | 本文提出了一种以工作为中心的基准测试表示方法，通过四个明确字段（代表的活动、测试的设置、所需的工作产品、评估的结果）来显式化知识工作基准测试中的隐含选择，从而支持跨职业的活动级报告和比较。 |
| [^299] | [EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation](https://arxiv.org/abs/2605.22286) | 本文提出EmoTrack框架，通过临床-语义建模将临床特征分数与话语级表示相融合，从心理咨询文本中预测PHQ-8抑郁严重程度总分，解决了临床评估与对话证据之间关联不足的问题。 |
| [^300] | [MONET: A Massive, Open, Non-redundant and Enriched Text-to-image dataset](https://arxiv.org/abs/2605.21272) | 该论文提出了MONET——一个包含1.038亿经过安全过滤、去重和多视觉语言模型重新标注的图像-文本对的开放文生图数据集，仅使用该数据集训练的40亿参数潜在扩散模型便取得了具有竞争力的GenEval和DPG分数。 |
| [^301] | [Agent Security is a Systems Problem](https://arxiv.org/abs/2605.18991) | 本文主张智能体安全应作为系统问题来处理——将AI模型视为不受信任的组件并在系统层面强制执行安全不变量，并通过分析十一例真实攻击论证了系统安全原则本可预防这些攻击。 |
| [^302] | [Learning to Learn from Multimodal Experience](https://arxiv.org/abs/2605.16857) | 提出AutoMMemo通用框架，通过将记忆机制表示为可执行的备忘程序并迭代发现有效程序，使智能体能够自动设计记忆机制，从而从多模态经验中学习。 |
| [^303] | [Prediction-Intervention Games and Invariant Sets](https://arxiv.org/abs/2605.16828) | 该论文提出预测-干预博弈（Stackelberg博弈的特例），并证明对于两类常见的跟随者干预目标，基于稳定毯这一不变子集的预测器性能至少与基于因果父节点的预测器相当。 |
| [^304] | [Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models](https://arxiv.org/abs/2605.16725) | 提出Alice闭环系统，将候选更新失败引发的“保持性冲突”作为结构性信号，归纳出假设类以修复可执行世界模型并引导前沿探索，从而仅凭交互证据实现先验失配下的在线自监督环境动力学发现。 |
| [^305] | [Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning](https://arxiv.org/abs/2605.14386) | Darwin Family通过14维自适应合并基因组、MRI-信任融合和跨架构杂交实现免训练的大语言模型进化合并，其旗舰模型Darwin-27B-Opus在GPQA Diamond上取得86.9%的成绩（在1,252个模型中排名第六），无需任何梯度训练即超越其完全训练的基础模型。 |
| [^306] | [GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?](https://arxiv.org/abs/2605.12827) | 提出了首个统一基准GraphIP-Bench，在单一黑盒协议下系统评估12种模型提取攻击与12种GNN知识产权防御方法（覆盖10个图数据集、3种骨干网络和3类任务），首次可靠地回答了窃取GNN的难度以及防御方法能否有效阻止窃取这两个问题。 |
| [^307] | [Every finite group admits a just finite presentation](https://arxiv.org/abs/2605.10402) | 本文证明了每个有限群都存在“恰有限”表现，即移除任意一条定义关系后就会变成无限群表现的有限表现，从而肯定地解决了 Kourovka 笔记本中的公开问题 21.10。 |
| [^308] | [Lattice Deduction Transformers](https://arxiv.org/abs/2605.08605) | 格演绎Transformer（LDT）通过在前向传递之间以格投影潜在状态来近似逻辑可靠的演绎推理，仅用80万参数和不到现有模型2%的训练计算量，就以近乎完全可靠的方式解决了99.999%的极难数独谜题。 |
| [^309] | [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865) | 提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。 |
| [^310] | [Structural Rationale Distillation via Reasoning Space Compression](https://arxiv.org/abs/2605.07139) | 提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。 |
| [^311] | [Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves](https://arxiv.org/abs/2605.06395) | 该论文提出了 HilbNets——一种以希尔伯特丛的联络拉普拉斯算子作为卷积算子的新型卷积学习框架，可处理流形上可能无穷维的信号，并通过采样诱导的希尔伯特胞腔层使卷积运算得以实际实现。 |
| [^312] | [Ordinal Credit Assignment: Multi-Step Reinforcement Learning from Comparisons Alone](https://arxiv.org/abs/2605.06375) | 该论文提出一种在多步任务中仅依靠轨迹间两两比较（无需奖励函数或奖励模型）的序数信用分配方法，用动作在同一时间步群体中的胜率替代未来累积奖励，证明其期望更新等价于策略梯度，且在确定性环境中按真实价值排序、在随机环境中偏好高胜率动作，并对标签噪声和位置偏差具有鲁棒性。 |
| [^313] | [In-Context Black-Box Optimization with Unreliable Feedback](https://arxiv.org/abs/2605.06187) | 该论文提出反馈感知的上下文黑盒优化框架FICBO，通过结构化先验预训练Transformer，使其能同时利用优化历史和可能不可靠的辅助反馈，从而在反馈质量参差不齐的情况下实现稳健的跨任务优化。 |
| [^314] | [Reinforcement Learning in GUI Agents: A Survey Toward Digital Inhabitants](https://arxiv.org/abs/2604.27955) | 本论文首次对强化学习与GUI智能体的交叉领域进行全面综述，提出了将现有方法分为离线RL、在线RL和混合策略的系统性分类法，并展望了GUI智能体向“数字居民”演进的研究方向。 |
| [^315] | [ReMoA: Recursive Modular Adaptation via Sparse Latent Refinement in Multimodal Diffusion Models](https://arxiv.org/abs/2604.25299) | 提出递归模块化适配框架ReMoA，通过在扩散模型联合注意力层中进行多步递归迭代精炼视觉token，并利用稀疏模块选择实现高效参数共享，从而提升多模态文本到图像生成中的结构化推理与文本遵循能力。 |
| [^316] | [AdaMamba: Adaptive Frequency-Gated Mamba for Long-Term Time Series Forecasting](https://arxiv.org/abs/2604.23239) | 提出AdaMamba，一种自适应频率门控的Mamba模型，通过自适应地处理频率分量来解决时频异质性问题，从而提升长期时间序列预测的准确性。 |
| [^317] | [AromaGen: Interactive Generation of Rich Olfactory Experiences with Multimodal Language Models](https://arxiv.org/abs/2604.01650) | AromaGen是一个由多模态语言模型驱动的12通道可穿戴嗅觉系统，能够将自然语言描述转化为可感知的气味组合并支持迭代优化，用户实验表明其检索增强的AI气味组合效果与人类专家相当。 |
| [^318] | [Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding](https://arxiv.org/abs/2603.25251) | 该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。 |
| [^319] | [MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels](https://arxiv.org/abs/2603.19310) | 提出MemReward，将查询、思考过程和答案作为异构图节点构建经验记忆，利用GNN将有限的奖励标签传播到无标签轨迹，从而在RL微调中充分利用全部采样轨迹优化LLM策略。 |
| [^320] | [AS2 -- Attention-Based Soft Answer Sets: An End-to-End Differentiable Neuro-Soft-Symbolic Reasoning Architecture](https://arxiv.org/abs/2603.18436) | AS²提出了一种完全可微分的神经符号架构，通过连续近似ASP不动点算子取代离散求解器，实现了无需外部求解器的端到端推理学习，并在视觉数独等任务上达到近乎完美的准确率。 |
| [^321] | [TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots](https://arxiv.org/abs/2603.18008) | 该论文提出了THERAPYGYM框架，通过基于CTRS量表的自动化评分衡量临床保真度、通过多标签标注衡量治疗安全性来评估治疗聊天机器人，并发布含116段对话和1270条专家评分的THERAPYJUDGEBENCH验证集，用于对照执业临床医生校准LLM评判者。 |
| [^322] | [Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures](https://arxiv.org/abs/2603.16568) | 提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。 |
| [^323] | [PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation](https://arxiv.org/abs/2603.13275) | PREBA提出了一种融合PCA加权检索与贝叶斯平均聚合的检索增强框架，无需任务特定微调即可将大语言模型的推理锚定于机构特定的临床证据上，从而实现准确且稳定的手术时长预测。 |
| [^324] | [Faces in Things, Faces in Nothing: Probing Face Selectivity in Vision Models with Pareidolia](https://arxiv.org/abs/2603.03989) | 该论文以人脸空想性错视为探针，利用 FacesInThings 数据集和三种同照片对照条件测试六个视觉模型，发现模型虽然会对虚假人脸作出响应，但其判断机制与人类并不相同。 |
| [^325] | [Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs](https://arxiv.org/abs/2603.03415) | 本文揭示了LLM内部表征的稀疏度随任务难度增加而系统性增大——任务越难，最后一层隐藏状态的表征越稀疏，且该规律跨模型、跨领域普遍成立，表明表征密度是模型对数据熟悉度的习得属性。 |
| [^326] | [Certified Circuits: Stability Guarantees for Mechanistic Circuits](https://arxiv.org/abs/2602.22968) | 提出了“认证电路”框架，通过随机化数据子采样包装任意黑盒电路发现算法，为电路组件的纳入决策提供可证明的稳定性保证，使其对概念数据集的有界编辑距离扰动保持不变，并对不稳定组件予以弃权。 |
| [^327] | [Same Words, Different Judgments: How Preferences Vary Across Modalities](https://arxiv.org/abs/2602.22710) | 该研究首次通过基于ICC的受控跨模态实验证明，人类在音频与文本两种模态下表达偏好的方式存在显著差异（跨模态一致性接近随机水平），因此原本为文本设计的偏好评估协议不能直接套用于语音模态。 |
| [^328] | [Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis](https://arxiv.org/abs/2602.20207) | 本研究通过层梯度分析，证明了大语言模型中存在固定的“黄金层”，其知识编辑性能接近逐样本最优层，且这些黄金层可通过代理数据集可靠识别并能跨数据集泛化到未见查询，从而改进了知识编辑方法。 |
| [^329] | [Disentangle to create: Architecture-level disentangled diffusion model for few-shot Chinese font generation](https://arxiv.org/abs/2602.18874) | 提出架构级解缠扩散模型ALD-Font，通过非对称设计将内容模板与风格调制信号在架构层面彻底分离，显著提升了少样本中文字体生成中的内容准确性与风格保真度。 |
| [^330] | [Evolutionary System Prompt Learning for Reinforcement Learning in LLMs](https://arxiv.org/abs/2602.14697) | 本文提出进化式系统提示学习（E-SPL），在强化学习迭代中联合优化模型权重与系统提示，实现权重中程序性知识与提示中可解释陈述性知识的自然分工，显著提升大语言模型在数学推理任务上的成功率与由易到难的泛化能力。 |
| [^331] | [A Lightweight Multi-View Approach to Short-Term Load Forecasting](https://arxiv.org/abs/2602.09220) | 本文提出一种利用单值嵌入、缩放时间范围输入和嵌入丢弃机制的轻量级多视角方法，以显著更少的参数实现了具有竞争力的短期负荷预测性能。 |
| [^332] | [CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation](https://arxiv.org/abs/2602.08686) | 论文提出CompilerKV，通过从校准语料库离线编译KV保留校正表，将在线校正开销降至O(1)查找，且编译出的保留表可作为可移植的架构先验，在跨语料库和跨模型迁移中表现出高度一致性。 |
| [^333] | [From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient](https://arxiv.org/abs/2602.02313) | 本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。 |
| [^334] | [OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference](https://arxiv.org/abs/2602.01493) | 提出OpInf-LLM框架，将算子推断与大语言模型结合，仅需少量解数据即可准确求解参数化偏微分方程，并能泛化到未见过的参数与边界条件，缓解了执行成功率与数值精度之间的权衡。 |
| [^335] | [Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio](https://arxiv.org/abs/2601.22718) | 该论文发现token级别的重要性采样比率无法修正前缀分布偏移，并提出用累积前缀重要性比率对前缀边缘化策略梯度进行精确修正，从而在大策略滞后下稳定大语言模型的离策略强化学习优化。 |
| [^336] | [COL-Trees: Efficient Hierarchical Object Search in Spatial Networks](https://arxiv.org/abs/2601.22183) | 提出了一种新颖的数据结构COL-Tree，通过层次化方法高效解决空间网络中聚合k近邻查询和k最远邻查询等复杂对象搜索问题，弥补了现有基于欧几里得距离启发式方法在空间网络中失效的不足。 |
| [^337] | [Investigating Anthropometric Fidelity in SAM 3D Body](https://arxiv.org/abs/2601.06035) | 本文发现 SAM 3D Body 难以重建特殊体型人群（如肌肉萎缩、脊柱侧弯、怀孕）的人体测量细节，并将其归因于“感知-失真权衡”与低维参数化 MHR 表示等架构设计带来的系统性偏差，而非模型能力不足。 |
| [^338] | [Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings](https://arxiv.org/abs/2601.01558) | 本研究将AlphaEarth嵌入作为静态流域表征用于无观测流域的径流预测，发现其样本外泛化鲁棒性优于传统CAMELS流域属性。 |
| [^339] | [PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds](https://arxiv.org/abs/2512.24193) | 本文提出PointRAFT，一种融合块茎高度先验几何知识的高通量点云回归网络，能够从不完整的点云中准确预测马铃薯块茎重量。 |
| [^340] | [AEGIS: Preserving Privacy of 3D Facial Avatars with Adversarial Perturbations](https://arxiv.org/abs/2511.17747) | AEGIS是首个针对3D高斯虚拟形象的隐私保护身份掩蔽框架，通过对高斯基元颜色系数施加对抗性扰动，在保持虚拟形象真实感和动画可用性的同时，实现跨视角一致的人脸识别身份保护。 |
| [^341] | [MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction](https://arxiv.org/abs/2511.03212) | 提出了一种基于多视角混合Transformer的MvBody模型，仅利用妊娠31-38周期间的3D光学人体扫描和自我报告医疗数据，即可实现可解释的剖腹产风险预测，适用于医疗资源有限的环境。 |
| [^342] | [Learning Soft Robotic Dynamics with Active Exploration](https://arxiv.org/abs/2510.27428) | 提出了一种不确定性感知的主动探索框架，通过主动收集信息丰富的数据，自主学习任务无关且可泛化的软体机器人动力学模型，在仿真和真实平台上的建模精度与零样本控制性能均显著优于随机探索和特定任务基线。 |
| [^343] | [Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs](https://arxiv.org/abs/2510.07697) | 本文首次对大语言模型中基于推理的后门攻击进行了全面综述，提出了将攻击分为关联型、被动型和主动型三类的新分类法，并系统分析了其机制、方法框架、防御策略及未解决的挑战。 |
| [^344] | [UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models](https://arxiv.org/abs/2510.02194) | 提出UpSafe°C框架，通过将安全关键层升级为稀疏MoE结构、利用路由器作为软防护栏、结合两阶段SFT和可调节的安全温度机制，在保持通用能力的同时实现大语言模型推理时可灵活控制的安全防护。 |
| [^345] | [Discovering Phase Space Structure in Learned Hamiltonian Systems](https://arxiv.org/abs/2509.23003) | 提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。 |
| [^346] | [Population-Aligned Persona Generation for LLM-based Social Simulation](https://arxiv.org/abs/2509.10127) | 该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。 |
| [^347] | [A Financial Brain Scan of the LLM](https://arxiv.org/abs/2508.21285) | 该论文利用稀疏自编码器对大语言模型进行“金融大脑扫描”，实现对内部概念的可解释映射与精准引导，并揭示了模型自我解释与其真实内部推理之间存在明显脱节。 |
| [^348] | [Computational Phenomenology of Borderline Personality Disorder: A Comparative Evaluation of LLM-Simulated Expert Personas and Human Clinical Experts](https://arxiv.org/abs/2508.19008) | 本研究通过混合方法比较评估了LLM模拟专家与人类临床专家在边缘型人格障碍定性分析中的表现，发现LLM在支持临床主题分析方面具有潜力但存在差异。 |
| [^349] | [Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning](https://arxiv.org/abs/2508.06199) | 本研究对28个预训练分子嵌入模型进行了大规模基准测试，发现除CheMeleon外所有神经模型均未超越经典ECFP计数指纹，且外部验证显示两者并无明确优劣，建议在新QSAR/QSPR研究中同时评估这两种方法。 |
| [^350] | [DUP: Detection-guided Unlearning for Backdoor Purification in Language Models](https://arxiv.org/abs/2508.01647) | 提出DUP统一框架，通过联合类无关距离与层间转换实现细粒度后门检测，并利用参数高效的遗忘机制净化语言模型中的后门，无需完整重训练或外部干净模型。 |
| [^351] | [When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings](https://arxiv.org/abs/2508.01128) | 提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。 |
| [^352] | [Scalable Source Selection for Machine Learning Tasks](https://arxiv.org/abs/2507.22186) | 提出了Splice系统，一种借鉴基因剪接思想的目标与成本感知数据源选择方法，通过识别能为下游任务提供额外信号的数据源来最大化任务效用，并开发了利用数据画像和梯度对齐的两种加速变体Splice-DP和Splice-Grad。 |
| [^353] | [Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning](https://arxiv.org/abs/2507.07359) | GO-CBED是一个面向目标的贝叶斯序贯因果实验设计框架，通过变分下界估计与Transformer策略网络直接最大化用户指定因果量的期望信息增益，实现非短视且实时的干预决策，在因果推理与发现任务中优于现有基线。 |
| [^354] | [Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?](https://arxiv.org/abs/2506.10912) | 该论文提出了ToxiMol——首个评估通用多模态大语言模型分子毒性修复能力的基准任务，包含覆盖11个任务、660个有毒分子的标准化数据集、机制感知的提示标注流程以及自动化评估框架ToxiEval。 |
| [^355] | [Hindsight Relabeling is All You Need for Reach-Avoid Learning](https://arxiv.org/abs/2505.19337) | 本文提出RADT，一种基于决策Transformer的离线无奖励强化学习方法，通过将目标与规避区域直接编码为提示token，仅凭随机策略产生的次优离线轨迹即可学会抵达-规避行为，并能在评估时灵活指定任意数量和大小的规避区域。 |
| [^356] | [ENACT-Heart -- ENsemble-based Assessment Using CNN and Transformer on Heart Sounds](https://arxiv.org/abs/2502.16914) | ENACT-Heart通过将处理频谱图的视觉Transformer与处理波形和谱质心可视化的CNN进行加权后期融合，在PASCAL心音数据集上实现了97.52%的分类准确率，优于任一单分支模型。 |
| [^357] | [EgoGen: An Egocentric Synthetic Data Generator.](http://arxiv.org/abs/2401.08739) | EgoGen是一种新的合成数据生成器，它可以为具体化自我的感知任务生成准确和丰富的地面真实训练数据，并利用自我视觉输入来感知3D环境。 |
| [^358] | [Estimating and Incentivizing Imperfect-Knowledge Agents with Hidden Rewards.](http://arxiv.org/abs/2308.06717) | 本文研究了一个自私学习代理和学习委托人之间的重复对自选游戏，探索如何估计和激励具有隐藏奖励的不完全知识代理。 |
| [^359] | [Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents.](http://arxiv.org/abs/2304.07407) | 本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。 |

# 详细

[^1]: BEAKER：自动驾驶化学实验室中具身大脑的专家策划基准

    BEAKER: An Expert-Curated Benchmark for Embodied Brains in Self-Driving Chemical Laboratories

    [https://arxiv.org/abs/2611.06911](https://arxiv.org/abs/2611.06911)

    该论文提出了BEAKER——一个由专家策划、包含1,000个问答样本和11种图像与视频任务的基准，用于评估多模态大语言模型在自动驾驶化学实验室中面向执行的具身推理与操作能力。

    

    自动驾驶化学实验室正在推动化学实验迈向具身自动化，在此过程中，机器必须感知实验室场景、推理实验状态，并在物理、程序和安全约束下执行操作。尽管多模态大语言模型越来越多地被视为此类系统的具身大脑，但现有的实验室基准主要关注安全性、异常检测或通用科学理解，其面向执行的具身能力仍未得到充分探索。为填补这一空白，我们提出了BEAKER，一个用于评估自动驾驶化学实验室中具身可操作性、知识、实验与推理能力的基准。BEAKER包含1,000个由专家策划的问答样本，涵盖11种图像和视频任务类型，覆盖化学实验室执行所需的关键判断，包括可供性定位、轨迹规划、设备、状态与安全识别等……

    arXiv:2611.06911v1 Announce Type: new  Abstract: Self-Driving Chemical Laboratories (SDCLs) are moving chemical experimentation toward embodied automation, where machines must perceive laboratory scenes, reason about experimental states, and act under physical, procedural, and safety constraints. Although Multimodal Large Language Models (MLLMs) are increasingly considered as embodied brains for such systems, existing laboratory benchmarks mainly focus on safety, anomaly detection, or general scientific understanding, leaving their execution-oriented embodied capabilities underexplored. To fill this gap, we introduce BEAKER, a Benchmark for evaluating Embodied Actionability, Knowledge, Experimentation, and Reasoning in SDCLs. BEAKER contains 1,000 expert-curated question-answer samples across 11 image and video task types, covering key judgments required for chemical laboratory execution, including affordance grounding, trajectory planning, equipment, state, and safety recognition, app
    
[^2]: FrontierPhysics：面向端到端物理学研究的智能体评估基准

    FrontierPhysics: Evaluating Agents for End-to-End Physics Research

    [https://arxiv.org/abs/2611.06910](https://arxiv.org/abs/2611.06910)

    该论文提出了FrontierPhysics——一个由42位物理学家基于自身研究设计的包含56个任务、覆盖九个领域的端到端物理研究基准，通过结合确定性验证器与专家评分标准评估AI智能体完成完整物理研究的能力，发现现有模型表现远未达到可发表水平。

    

    AI智能体在物理学研究中正扮演着日益重要的角色。然而，现有的物理学基准仅评估固定问题、孤立的子任务，或在给定解答的情况下复现论文，且很少在检查正确性的同时评估研究质量。我们提出了FrontierPhysics，这是一个端到端基准，包含由42位物理学家基于自身研究设计的56个任务，涵盖理论物理、计算物理和实验物理共九个领域。每个任务要求智能体查阅文献、制定研究计划、开展研究与实验，并产出结构化结果和科学论文手稿。通过将确定性验证器与专家撰写的评分标准相结合，FrontierPhysics能够评估智能体的研究是否正确且具备发表价值。在对22个模型的26种配置的测试中，没有任何配置在其168次试验中通过超过46次。对智能体轨迹的人工审查表明，前沿智能体的主要弱点在于……（摘要原文在此处截断）

    arXiv:2611.06910v1 Announce Type: new  Abstract: AI agents are playing an increasingly important role in physics research. However, existing physics benchmarks only evaluate fixed questions, isolated sub-tasks, or paper reproduction with given solutions, and rarely check research quality alongside correctness. We introduce FrontierPhysics, an end-to-end benchmark with 56 tasks made by 42 physicists based on their own research, covering nine domains across theoretical, computational, and experimental physics. Each task requires agents to review the literature, develop a research plan, conduct research and experiments, and produce structured outputs and scientific manuscripts. Combining deterministic verifiers with expert-authored rubrics, FrontierPhysics evaluates whether agents' research is correct and publishable. Across 22 models in 26 configurations, no configuration passes more than 46 of its 168 rollouts. Manual review of agent trajectories shows that frontier agents' main weaknes
    
[^3]: 基于仿真的具感知人形机器人运动

    Perceptive Humanoid Motion from Simulation

    [https://arxiv.org/abs/2611.06908](https://arxiv.org/abs/2611.06908)

    提出了HALO人形机器人自主系统，通过在千个程序化生成的家庭环境中合成视觉-语言-运动数据，使数据规模随算力而非人力扩展，让人形机器人掌握坐、导航、穿门、蹲爬、推箱等多种物理技能并协调完成复合任务。

    

    人形机器人有潜力自动化广泛的物理任务。然而，迈向通用型人形机器人自主能力的进展仍受限于数据瓶颈：将像素和自由形式的语言映射到底层关节命令需要成对的监督数据，而目前尚不存在足够大规模的此类数据语料库。一些方法依赖于人类遥操作和可穿戴动作捕捉设备，这需要至少一小时的人工时间才能产出一小时的数据。我们提出了HALO，这是一个由数据合成流水线驱动的人形机器人自主系统，该流水线在一千个程序化生成的家庭环境中合成视觉-语言-运动数据。该语料库的规模随算力而非人力扩展。HALO掌握了一系列物理技能，包括坐在椅子上、导航至指定名称的物体、侧身穿过半开的门、在桌下蹲伏和爬行，以及推箱子。它协调这些技能以解决复合任务。

    arXiv:2611.06908v1 Announce Type: cross  Abstract: Humanoid robots have the potential to automate a wide range of physical tasks. Progress toward generalist humanoid autonomy, however, remains constrained by a data bottleneck: mapping pixels and free-form language to low-level joint commands requires paired supervision, and no corpus of such data exists at sufficiently large scale. Some approaches rely on human teleoperation and wearable capture devices, which require at least an hour of human time to produce an hour of data. We present HALO, a humanoid autonomy system powered by a pipeline that synthesizes vision-language-motion data across a thousand procedurally generated homes. The corpus scales with compute rather than human labor. HALO acquires a repertoire of physical skills, including sitting on chairs, navigating to named objects, stepping sideways through half-open doors, crouching and crawling beneath tables, and pushing boxes. It orchestrates these skills to solve composite
    
[^4]: 免费的遮挡检测：从两个单目距离线索之间的不一致中检测遮挡

    Occlusion for Free: Detecting Occlusion from the Disagreement Between Two Monocular Distance Cues

    [https://arxiv.org/abs/2611.06899](https://arxiv.org/abs/2611.06899)

    该论文提出一种无需训练、无需遮挡标注的方法，仅利用检测框表观高度（不受遮挡影响）与单目度量深度（被拉向遮挡物）之间的不一致性即可检测行人遮挡，在KITTI上达到AUC 0.909。

    

    即使检测器无法恢复被遮挡的像素，知道行人处于遮挡状态对机器人依然有用，但遮挡估计器通常需要在部署场景中很少具备的遮挡标注上进行训练。我们证明，遮挡可以从纯相机流程已经计算出的两个距离估计中直接读取出来：表观高度基本不受遮挡影响，因为检测框仍然覆盖整个人；而单目度量深度会被拉向遮挡物，因为深度图报告的是框内最近的表面。只有两者的相对大小（排序）才是关键，因此无需焦距、身高或深度尺度，前提是深度模型的误差接近于一个纯增益，而这一条件可以在无标注的情况下进行检验。该评分在KITTI上检测严重遮挡的AUC达到0.909（在所有标注行人上为0.888，包括检测器漏检的行人），无需训练、无需遮挡标注，并且并非检测器失效的代理指标。

    arXiv:2611.06899v1 Announce Type: cross  Abstract: Knowing that a pedestrian is occluded is useful to a robot even when the detector cannot recover the hidden pixels, but occlusion estimators are normally trained on occlusion annotations that deployment settings rarely have. We show that occlusion can be read off two distance estimates a camera-only pipeline already computes. Apparent height is largely unchanged by occlusion, because the detected box still spans the whole person; monocular metric depth is pulled toward the occluder, because the nearest surface inside the box is what a depth map reports. Only their ordering matters, so no focal length, stature, or depth scale is needed, provided the depth model's error is close to a pure gain, a condition that can be checked without labels. The score detects heavy occlusion on KITTI with AUC 0.909 (0.888 over all annotated pedestrians, including those the detector misses), requires no training and no occlusion labels, and is not a proxy
    
[^5]: 枯燥、肮脏、危险？工作条件与职业AI暴露度指标解释的局限性

    Dull, Dirty, Dangerous? Working Conditions and the Limits of Occupational AI Exposure

    [https://arxiv.org/abs/2611.06888](https://arxiv.org/abs/2611.06888)

    该研究发现“枯燥、肮脏、危险”的工作条件与职业AI暴露度之间的关联在控制体力要求、教育培训和工资等职业特征后会显著减弱甚至消失，揭示了当前常用AI暴露度指标在解释上的局限性。

    

    自动化常被论证为是将人们从枯燥、肮脏和危险的工作中解脱出来的手段。我利用涵盖754个美国职业的21个O*NET条目，考察了这些工作条件与职业AI暴露度之间的关系。在大语言模型（LLM）度量和AI职业暴露度（AIOE）指标下，无论采用等权职业权重还是就业权重，较高的暴露度都与较低的工作条件评分相关。在对体力要求、所需教育、培训和经验以及工资进行调整后，这种综合关联性有所减弱。调整后的置信区间在LLM暴露度方面包含零，而AIOE的置信区间仍低于零。将重复性与受限自主权分开处理，也会改变合并后的“枯燥”分数所传达的含义。这些比较表明，条目分组方式和职业特征会影响对暴露度估计结果的解释。职业画像为选择具有对比性的研究场景提供了起点。确定期望的自动化或……

    arXiv:2611.06888v1 Announce Type: new  Abstract: Automation is often justified as relief from dull, dirty, and dangerous work. I examine how these conditions relate to occupational AI exposure using 21 O*NET items across 754 US occupations. Higher exposure is associated with lower condition scores under a large language model (LLM) measure and AI Occupational Exposure (AIOE), with equal occupation weights and employment weights. Adjustment for physical demands, required education, training and experience, and wages reduces the composite association. The adjusted confidence intervals include zero for LLM exposure and remain below zero for AIOE. Separating repetition from restricted discretion also changes what the combined Dull score conveys. These comparisons show how item groupings and occupational characteristics shape the interpretation of exposure estimates. Occupational profiles offer a starting point for selecting contrasting research settings. Establishing desired automation or 
    
[^6]: 基于广义信念传播网络的线性码学习译码方法

    Learning to Decode Linear Codes with Generalized Belief Propagation Network

    [https://arxiv.org/abs/2611.06874](https://arxiv.org/abs/2611.06874)

    本文提出广义信念传播网络GBPNet，在保持消息传递对称性和Tanner图结构的前提下，通过级联变量节点更新、可学习消息缩放以及渐进式和高信噪比训练策略，实现了优于传统信念传播算法和现有最先进神经译码器的误码率性能。

    

    纠错码是通信系统的基本组成部分，能够在噪声信道上实现可靠的数据传输。近年来，神经译码器相比经典迭代算法展现出了显著的性能提升。本文提出了一种广义信念传播网络，它在保留消息传递对称性的同时增强了译码器的学习能力。该架构采用级联的变量节点更新和可学习的逐元素消息缩放，同时保持Tanner图结构不变。我们还引入了两种训练策略——渐进式训练和高信噪比训练——以提升译码精度。在多种线性码上的实验结果表明，GBPNet相比传统信念传播算法以及所评估的最先进神经译码器取得了更低的误码率。

    arXiv:2611.06874v1 Announce Type: new  Abstract: Error correcting codes are a fundamental component of communication systems, enabling reliable data transmission over noisy channels. Recently, neural decoders have demonstrated notable performance gains over classical iterative algorithms. In this paper, we propose a Generalized Belief Propagation Network (GBPNet) that preserves message-passing symmetry while enhancing the decoder's learning capability. The architecture uses cascaded variable-node updates and learned element-wise message scaling while retaining the Tanner graph structure. We also introduce two training strategies, progressive training and high-SNR training, that improve decoding accuracy. Experimental results on several linear codes show that GBPNet achieves lower bit error rates than conventional belief propagation and the evaluated state-of-the-art neural decoders.
    
[^7]: 关于面向LLM智能体扩展的轮次级Best-of-N采样方法研究

    On Turn-Level Best-of-N Sampling for Scaling LLM Agents

    [https://arxiv.org/abs/2611.06868](https://arxiv.org/abs/2611.06868)

    该论文提出一种轮次级Best-of-N采样的推理时扩展方法，通过生成器-选择器双模型框架在每轮生成并筛选候选动作，显著提升了LLM智能体在长时程任务上的准确率与稳定性。

    

    在本工作中，我们评估了一种基于轮次级Best-of-N采样（\method）的简单推理时扩展方法，用于大型语言模型（LLM）智能体在复杂、长时程任务上的表现。我们通过一个包含两个LLM的“生成器-选择器”框架来实现该方法：在每一轮中，生成器模型生成N个候选动作，选择器模型从中选择一个，随后该动作在环境中执行。本文与zainullina等人（2025）的工作密切相关，后者在推理时使用评论模型进行引导搜索。与他们的工作不同，我们评估了现成的开源权重LLM是否能够有效地直接比较不同的候选动作。在长时程终端任务上的实验表明，\method可以带来显著的性能提升，包括准确率（5-10%）和多次尝试之间的一致性。对智能体轨迹的分析揭示了……（摘要在此处截断）

    arXiv:2611.06868v1 Announce Type: new  Abstract: In this work, we evaluate a simple inference-time scaling approach based on \textit{turn-level best-of-N sampling} (\method) for large language model (LLM) agents on complex, long-horizon tasks. We instantiate this using a \textit{generator-selector} framework with two LLMs; at each turn, a \emph{generator} model generates $N$ candidate actions and a \emph{selector} model selects one of them, which is then executed in the environment. This paper is closely related to the work of \cite{zainullina2025guided} who used critic models for guided search at inference time. Different from their work, we evaluate if off-the-shelf, open-weight LLMs can be effective at comparing different candidate actions directly. Experiments on long-horizon terminal tasks show that \method can lead to significant performance improvements, both in terms of accuracy (5--10\%) and consistency across repeated attempts. Analysis of agent trajectories reveal a shift in
    
[^8]: 面向证据感知的个体动物识别的反事实检索方法

    CounterFactual Retrieval for Evidence-Aware Individual Animal Identification

    [https://arxiv.org/abs/2611.06864](https://arxiv.org/abs/2611.06864)

    提出反事实前景检索（CFR）这一证据感知评估协议，通过完整图像、仅前景、仅捷径三种视图对动物个体识别进行评测，将检索分数转化为证据画像，从而揭示模型是依赖动物特征还是身份相关的背景捷径。

    

    自动化的个体动物识别正在从小规模精选研究向共享基准和可复用的基础模型发展。这一进步是有价值的，但当前的评估方式未能揭示匹配结果背后的证据依据，即高检索准确率无法表明模型究竟是使用了动物本身的特征，还是利用了与身份相关的捷径信息。为了解决这一问题，我们提出了反事实前景检索（CounterFactual Foreground Retrieval, CFR），这是一种证据感知的评估协议。CFR 在不同视图下评估同一查询-图库检索任务：完整图像视图衡量标准检索性能；仅前景视图衡量由前景支撑的检索能力；仅捷径视图则衡量从背景或轮廓中可恢复的身份信号量。由此，单一的检索分数被转化为一份证据画像：包括模型在原始图像上的表现，以及当仅有动物（前景）信息时该性能能保留多少。

    arXiv:2611.06864v1 Announce Type: cross  Abstract: Automated identification of individual animals is moving from small curated studies toward shared benchmarks and reusable foundation models. This progress is useful, but current evaluation leaves the evidence behind a match unresolved, i.e., high retrieval accuracy does not show whether a model used animal features or identity-correlated shortcuts. We address this with Counterfactual Foreground Retrieval (CFR), an evidence-aware evaluation protocol. CFR evaluates the same query-gallery task under different views: full image, foreground-only, and shortcut-only. The full-image measures standard retrieval performance; the foreground-only measures foreground-supported retrieval; and shortcut-only view measure how much identity signal is recoverable from background, or silhouette. This turns a retrieval score into an evidence profile: how well the model performs on the original images, how much of that performance remains when only the anim
    
[^9]: CompressionBench：你的编程智能体如何压缩大语言模型？

    CompressionBench: How can your coding agents compress LLMs?

    [https://arxiv.org/abs/2611.06858](https://arxiv.org/abs/2611.06858)

    该论文提出CompressionBench基准，将大语言模型压缩形式化为一项终端任务，系统评估了编程智能体自主压缩LLM的能力，发现大多数智能体提交能通过独立验证，且压缩后模型性能接近原始BF16模型。

    

    递归自我改进要求AI系统能够改进后续系统所使用的模型和基础设施。大语言模型（LLM）压缩提供了一个具体的效率目标：在更小的模型存储预算内保留预训练模型的行为。我们研究编程智能体能否自主地完成这一步骤。我们将压缩形式化为一项终端任务：智能体接收源模型检查点、存储限制、推理环境和时间预算，然后构建压缩后的权重和可执行的推理代码。智能体自主控制其算法和实验；困惑度反馈用于指导搜索，独立验证则检查最终模型。在六个智能体和六种模型规模的设置中，36个提交中有29个通过验证。每个任务选择困惑度最低的智能体设计，相比BF16平均增加3.4%的困惑度，下游准确率为77.33%，而BF16为79.56%。

    arXiv:2611.06858v1 Announce Type: new  Abstract: Recursive self-improvement requires AI systems to improve the models and infrastructure used by subsequent systems. LLM compression offers a concrete efficiency objective: retain a pretrained model's behavior within a smaller model storage budget. We study whether coding agents can carry out this step autonomously. We formulate compression as a terminal task in which an agent receives a source checkpoint, storage limit, inference environment, and time budget, then constructs compressed weights and executable inference code. The agent controls its algorithm and experiments; perplexity feedback guides search, and independent verification checks the final model. Across six agents and six model--size settings, 29 of 36 submissions pass verification. Selecting the lowest-perplexity agent design per task yields an average 3.4% perplexity increase over BF16 and 77.33% downstream accuracy, compared with 79.56% for BF16. Recorded experiments reve
    
[^10]: ServerBench：你的编程智能体如何设计专用的大语言模型服务系统？

    ServerBench: How can your coding agents design specialized LLM serving systems?

    [https://arxiv.org/abs/2611.06841](https://arxiv.org/abs/2611.06841)

    本文提出ServerBench，首个让编程智能体以“每美元收益”为目标通过实验自主设计并验证专用LLM服务系统的基准，其中Claude Sonnet 5.5的设计达到vLLM基线的4.87倍。

    

    arXiv:2611.06841v1 公告类型：新论文 摘要：大语言模型（LLM）编程智能体通过实验来开发软件，这为设计支撑其自身运行的服务系统提供了契机。有效的服务需要在会话依赖、延迟约束和正确性要求下协调实现与运营决策。我们提出了ServerBench，一个面向智能体化设计完整、专用LLM服务系统的基准。智能体围绕一个共同目标进行实验：每美元收益，即每美元GPU成本所产生的满足延迟要求的token价值。有状态重放保留了会话依赖关系，独立验证机制则对提交的系统进行检查。在11个编程智能体模型和三种部署环境的测试中，Claude Sonnet 5.5在Gemma E4B上取得了2.97的每美元收益，达到vLLM参考实现的4.87倍；33次运行中有22次产出了通过验证的系统，其中13次超越了最强基线。我们的分析识别出三项发现：高价值的设计协调……（原文摘要在此处截断）

    arXiv:2611.06841v1 Announce Type: new  Abstract: Large language model (LLM) coding agents develop software through experimentation, offering an opportunity to design the serving systems supporting their own operation. Effective serving requires coordinating implementation and operating decisions under session dependencies, latency constraints, and correctness requirements. We introduce ServerBench, a benchmark for agentic design of complete, specialized LLM serving systems. Agents experiment against a common objective, dollars per dollar: latency-qualified token value per dollar of GPU cost. Stateful replay preserves session dependencies, and independent verification checks the submitted system. Across 11 coding-agent models and three deployments, Claude Sonnet 5.5 achieves 2.97 dollars per dollar on Gemma E4B, $4.87\times$ the vLLM reference; 22 of 33 runs yield verified systems, and 13 exceed the strongest baseline. Our analysis identifies three findings: high-value designs coordinat
    
[^11]: 上下文-答案元模型：在首个答案词元生成之前预测大语言模型激活

    Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token

    [https://arxiv.org/abs/2611.06837](https://arxiv.org/abs/2611.06837)

    本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。

    

    在随机解码条件下，单个上下文可以从语言模型中引出许多不同的答案。表征这种变化通常需要大量昂贵的自回归推演，而每次推演只能揭示一个实现样本。我们提出这样一个问题：模型在阅读上下文时产生的激活能否直接预测模型推演时的激活。我们训练了上下文-答案元模型，将上下文激活序列映射为平均池化答案激活的点预测或分布预测。我们发现：（i）我们研究的四种元模型家族能够高保真地预测平均池化的答案激活，且分布元模型还能捕捉推演过程中的变异性；（ii）随着训练集增长到50万个提示，性能不断提升，并且元模型能够零样本泛化到域外数据集；（iii）它们的预测保留了可预测模型行为的信号；（iv）来自Qwen2.5的上下文激活能够预测平均池化的答案激活。

    arXiv:2611.06837v1 Announce Type: cross  Abstract: Under stochastic decoding, a single context can elicit many different answers from a language model. Characterizing this variation typically requires many expensive autoregressive rollouts, each revealing only one realization. We ask whether the activations produced while reading the context can directly forecast the model's rollout activations. We train context--answer metamodels that map a context activation sequence to point or distribution forecasts of mean-pooled answer activations. We find that (i) the four metamodel families we study forecast mean-pooled answer activations with high fidelity, and the distribution metamodels also capture rollout variability; (ii) performance improves as the training set grows to 500,000 prompts, and the metamodels generalize zero-shot to out-of-domain datasets; (iii) their forecasts retain behavior-predictive signal; and (iv) context activations from Qwen2.5 predict mean-pooled answer activations
    
[^12]: 使用高级Vitis HLS实现AWS云FPGA开发

    Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS

    [https://arxiv.org/abs/2611.06829](https://arxiv.org/abs/2611.06829)

    该论文提出了一套完整的从Vitis HLS到AWS云FPGA部署的工具链和四级误差定位方法论，使开发者乃至AI编码智能体能够通过高级语言在AWS云FPGA上高效开发、测试和部署机器学习模型。

    

    我们提出了一套在AWS EC2 F2云FPGA基础设施上使用高级语言开发机器学习模型的工具链，实现了一个完整的从Vitis HLS到HDK桥接再到AFI的工作流程——尽管AWS FPGA已面世八年，这一流程在学术文献中始终缺失。我们引入了一种四级方法论（FP32 PyTorch、位精确量化Python、HLS C仿真、物理FPGA），以帮助定位错误，包括FPGA特有的电路错误。这些工具共同使编码智能体能够根据用户指令有效地实现FPGA设计，并按需进行测试和调试。我们记录了部署工作流程、时序收敛策略以及被证明至关重要的实用HLS工程决策。我们使用针对命名实体识别任务微调的TinyBERT进行验证，在FPGA硬件上实现了嵌入层和四个编码器层中的第一层。我们的实现从主机DRAM流式传输权重。

    arXiv:2611.06829v1 Announce Type: cross  Abstract: We present a toolchain for development of ML models on AWS EC2 F2 cloud FPGA infrastructure from high-level language, implementing a complete Vitis HLS to HDK bridge to AFI workflow that has been absent from the academic literature despite eight years of AWS FPGA availability. We introduce a four-level methodology (FP32 PyTorch, bit-accurate quantized Python, HLS C-simulation, physical FPGA) to aid in localizing errors, including FPGA-specific circuit errors. Together these tools allow coding agents to effectively implement FPGA designs from user directions, testing and debugging as needed. We document the deployment workflow, timing closure strategies, and practical HLS engineering decisions that proved crucial. We validate using TinyBERT fine-tuned for address Named Entity Recognition, implementing the embedding layer and the first of four encoder layers on the FPGA hardware. Our implementation streams weights from host DRAM, yieldin
    
[^13]: 规则、轨迹与路由统计：面向开放模型在ARC-AGI-2上的训练与校准数据

    Rules, Traces and Routing Statistics: Training and Calibration Data for Open Models on ARC-AGI-2

    [https://arxiv.org/abs/2611.06827](https://arxiv.org/abs/2611.06827)

    通过在类型化语言中构建12,000条经行为去重的可执行规则和95,993个经过捷径求解器过滤的高质量合成任务，并结合真实任务记录微调与四GPU调度优化，将开放模型在ARC-AGI-2半私有集上的分数从28.96提升至33.06，同时证明纯合成数据训练无法超越真实数据模型。

    

    我们研究了哪些训练与校准数据能够帮助三个开放模型在求解器于四块L4 GPU上离线运行至多12小时的约束下解决ARC-AGI-2任务。我们在公开任务上对数据进行了离线比较，并在120个隐藏的半私有任务上报告了189次运行结果。对于NVARC Qwen3-4B网格模型，我们构建了一个包含12,000条类型化语言可执行规则的语料库以及95,993个任务；一条规则只有在固定探测集上的行为与所有已被采纳的规则都不同时才会被采纳；一个任务只有在其样本对能够反驳最邻近竞争规则的候选名单并击败12个捷径求解器时才会被采纳。在1,920条真实任务记录上进行持续微调并改进四GPU调度后，半私有分数从28.96（发布模型的均值）提升至33.06（五次运行的均值，最佳为34.86）；没有任何使用合成任务训练的检查点——即使使用全部95,993个合成任务——能够超越真实任务模型。对于Qwen3.8-27B推理模型，我们在其自身的1,036条（摘要在此处截断）

    arXiv:2611.06827v1 Announce Type: new  Abstract: We study which training and calibration data help three open models solve ARC-AGI-2 tasks when the solver runs offline on four L4 GPUs for at most 12 hours. We compare data offline on public tasks and report 189 runs on the 120 hidden semi-private tasks. For the NVARC Qwen3-4B grid model, we built a corpus of 12,000 executable rules in a typed language and 95,993 tasks; a rule is admitted only if its behaviour on a fixed probe set differs from that of every admitted rule, and a task only if its pairs refute a shortlist of nearest competing rules and defeat 12 shortcut solvers. Continued fine-tuning on 1,920 real task records with revised four-GPU scheduling raised the semi-private score from 28.96 (mean of the released model) to 33.06 (mean of five runs, best 34.86); no checkpoint trained with synthetic tasks, even all 95,993 of them, improved on the real-task model. For the Qwen3.8-27B reasoning model, we fine-tuned on 1,036 of its own 
    
[^14]: 基于缓存轨迹的任务高效智能体评估

    Task-Efficient Agent Evaluation Using Cached Traces

    [https://arxiv.org/abs/2611.06803](https://arxiv.org/abs/2611.06803)

    该论文提出用LLM提取智能体执行轨迹的结构化摘要并嵌入表示，在保留正确性与任务相关信号的同时滤除系统身份等干扰信号，从而利用缓存轨迹高效预测新智能体系统的基准测试得分。

    

    arXiv:2611.06803v1 公告类型：新论文。摘要：高效基准测试方法通过少量任务执行来估计系统的完整基准测试性能。标准方法仅依赖于从一组已评估的参考系统中获取的执行结果的二元正确性。然而，智能体系统会为每次执行产生丰富的轨迹，这些轨迹能够揭示其能力的更多信息。利用这些信息需要有效的轨迹表示方法。我们提出通过用大语言模型（LLM）提取轨迹的结构化摘要，并使用标准文本编码器对摘要进行嵌入来表示轨迹。这种表示方法在保留正确性和任务相关信号的同时，最小化了在原始轨迹嵌入中占主导地位的系统身份、模型系列和评测框架相关信号。基于来自所有参考系统的每个任务执行的这种表示，我们构建系统的低维表示，并由此预测新系统的得分（摘要在此处被截断）。

    arXiv:2611.06803v1 Announce Type: new  Abstract: Efficient benchmarking estimates a system's full benchmark performance from a small number of task executions. Standard approaches rely solely on the binary correctness of executions from a collection of already-evaluated reference systems. An agentic system, however, produces a rich trace for each execution that can reveal substantially more about its capabilities. Leveraging this information requires useful trace representations. We propose representing a trace by extracting a structured summary of it with an LLM and embedding the summary with a standard text encoder. These representations preserve correctness- and task-related signal while minimizing the presence of system identity-, model family- and harness-related signals that dominate raw trace embeddings. Given this representation for each task execution from all reference systems, we construct low-dimensional representations of the systems and predict a new system's score from i
    
[^15]: 消息传递图持久性

    Message-passing graph persistence

    [https://arxiv.org/abs/2611.06801](https://arxiv.org/abs/2611.06801)

    本文提出消息传递图持久性（MGP），将消息传递机制融入图过滤过程构建拓扑描述子，并证明了其图同构不变性以及严格强于现有基于持久同调描述子的表达能力。

    

    消息传递图神经网络的表达能力受限于其一维Weisfeiler-Leman测试。持久同调提供了可以补充GNN表示的拓扑描述子，但现有方法通常将持久性的计算与表示学习过程分开进行。在本文中，我们提出了消息传递图持久性，这是一类将消息传递机制融入图过滤过程的拓扑描述子。在每个过滤步骤中，节点嵌入利用当前子图进行更新，并与持久性事件相关联。这些嵌入还在持久性计算过程中决定连通分量的代表元。我们证明了MGP在图同构下保持不变，并且其表达能力严格强于其所扩展的基于持久同调的描述子。我们在表达能力基准测试以及多个图分类和回归任务上对MGP进行了评估。

    arXiv:2611.06801v1 Announce Type: cross  Abstract: Message-passing graph neural networks (GNNs) are limited in expressive power by the one-dimensional Weisfeiler-Leman test. Persistent homology (PH) provides topological descriptors that can complement GNN representations, but existing approaches typically compute persistence separately from the representation learning process. In this paper, we introduce Message-Passing Graph Persistence (MGP), a family of topological descriptors that incorporates message passing into graph filtrations. At each filtration step, node embeddings are updated using the current subgraph and associated with persistence events. These embeddings also determine the representatives of connected components during persistence computation. We prove that MGP is invariant under graph isomorphism and that it is strictly more expressive than the PH-based descriptors it extends. We evaluate MGP on an expressivity benchmark and on several graph classification and regress
    
[^16]: 将整数网格转化为空间直觉：主导ARC-AGI-3的是感知而非规划

    Transforming Integer Grids to Spatial Intuition: Perception, Not Planning, Governs ARC-AGI-3

    [https://arxiv.org/abs/2611.06793](https://arxiv.org/abs/2611.06793)

    该论文提出开源智能体SARBLOH，证明在ARC-AGI-3交互推理中起决定作用的是空间感知而非规划能力——其零参数O(HW)感知引擎可在35毫秒内将整数网格帧转化为对象清单与几何简报，从而大幅减少前沿模型因序列化破坏空间拓扑而产生的高昂探索浪费。

    

    样本高效的流体智能需要能够快速建立陌生交互环境工作模型的智能体。ARC-AGI-3通过无指令视觉游戏对此进行测试，游戏采用“相对人类动作效率”评分，该评分会以二次方方式惩罚探索性浪费。当前沿模型读取原始整数网格时，序列化破坏了空间拓扑结构，导致它们需要花费数百个动作来重新发现基本的边界和对称性。我们提出了SARBLOH，一个开源权重的智能体，其建立在一个前提之上：ARC-AGI-3上的交互推理由空间感知而非规划能力所主导。一个零参数、复杂度为O(HW)的感知引擎能在35毫秒内将每一帧转化为包含二部图时间跟踪的对象清单、区分已验证不变量与可供性假设的几何简报，以及一个带注释的画布。配合“代码即动作”的运行时机制，该智能体编写的是算法级假设而非低级动作。在八个……（摘要在此处截断）

    arXiv:2611.06793v1 Announce Type: cross  Abstract: Sample-efficient fluid intelligence requires agents that rapidly form working models of unfamiliar interactive environments. ARC-AGI-3 tests this across uninstructed visual games scored by Relative Human Action Efficiency, which penalizes exploratory waste quadratically. When frontier models read raw integer grids, serialization destroys spatial topology, and they spend hundreds of actions rediscovering basic boundaries and symmetries. We introduce SARBLOH, an open-weight agent built on the premise that spatial perception, not planning capacity, governs interactive reasoning on ARC-AGI-3. A zero-parameter O(HW) perception engine turns each frame, in 35 ms, into an object inventory with bipartite temporal tracking, a geometric briefing that separates proven invariants from affordance hypotheses, and an annotated canvas. Paired with a code-as-action runtime, the agent writes algorithmic hypotheses instead of low-level moves. Across eight
    
[^17]: FedFiOT：用于单轮和小轮次联邦学习中模型合并的Fisher加权最优传输方法

    FedFiOT: Fisher-Weighted Optimal Transport for Model Merging in One-Shot and Few-Round Federated Learning

    [https://arxiv.org/abs/2611.06783](https://arxiv.org/abs/2611.06783)

    FedFiOT提出了一种基于Fisher信息加权的最优传输框架，用于单轮和小轮次联邦学习，通过量化参数重要性来指导神经元对齐与模型聚合，解决了客户端模型漂移和数据异质性导致的直接平均性能下降问题。

    

    单轮和小轮次联邦学习（FL）通过在一次或少数几次通信轮次内构建全局模型来减少通信负担，但其性能通常低于传统的多轮次联邦学习。由于缺乏频繁的同步，客户端模型可能发生漂移，导致学习相似特征的神经元在各客户端占据不同的位置。同时，本地数据的异质性可能导致相应参数对每个客户端预测的重要性存在差异。这些挑战共同导致直接平均法会合并不匹配的神经元，并且尽管参数重要性不同，仍赋予它们相同的权重。我们提出了FedFiOT，一个面向单轮和小轮次联邦学习的Fisher加权最优传输框架，它利用Fisher信息来量化参数重要性，并指导神经元对齐和模型聚合。FedFiOT联合优化神经元对应关系和全局模型……

    arXiv:2611.06783v1 Announce Type: cross  Abstract: One-shot and few-round federated learning (FL) reduce communication burden by constructing a global model within one or a few communication rounds, but often achieve lower performance than conventional multi-round FL. Without frequent synchronization, client models may drift apart, causing neurons that learn similar features to occupy different positions across clients. At the same time, local data heterogeneity can cause corresponding parameters to differ in their importance to each client's predictions. Together, these challenges cause direct averaging to combine mismatched neurons and assign equal weight to parameters despite differences in their importance. We propose FedFiOT, a Fisher-weighted optimal transport framework for one-shot and few-round FL that leverages Fisher information to quantify parameter importance and guide neuron alignment and model aggregation. FedFiOT jointly optimizes neuron correspondences and global model 
    
[^18]: 先路由，后观察：通过将路由策略与动作空间分离，使视觉智能体能够进行科学推理

    Route First, See Later: Enabling Vision Agents to Reason Scientifically by Separating Routing Policy from Action Spaces

    [https://arxiv.org/abs/2611.06760](https://arxiv.org/abs/2611.06760)

    提出ZETA架构，通过将路由策略与视觉工具执行解耦，让智能体先解析科学意图再调用专用工具，并将测量建立在显式几何之上，从而将推理错误与感知失败作为独立失败模式进行审计评估，提升科学图像分析自动化的可靠性。

    

    科学图像分析工作流要求智能体在描述、检测、计数和分割等多种能力中进行选择。当前的多工具架构将意图解释与视觉执行混为一谈，从而掩盖了失败究竟源于有缺陷的推理（错误路由）还是失败的感知（无效的工具执行）。这种可审计性的缺失使得智能体在高风险的科学自动化场景中不可靠。我们提出了ZETA，这是一种可扩展的架构，实现了路由策略与动作执行的解耦。ZETA在调用专用视觉工具之前先解析科学意图，将测量视为基于显式几何的确定性操作，而非生成式输出。该架构揭示了智能体决策过程的“黑箱”，使路由正确性和下游执行可以作为相互独立的失败模式分别进行评估。据我们所知，我们提供了……

    arXiv:2611.06760v1 Announce Type: cross  Abstract: Scientific image-analysis workflows require agents to select among diverse capabilities like description, detection, counting, and segmentation. Current multi-tool architectures conflate intent interpretation with visual execution, masking whether a failure arises from flawed reasoning (misrouting) or failed perception (invalid tool execution). This lack of auditability renders agents unreliable for high-stakes scientific automation. We introduce ZETA, an extensible architecture that operationalizes the decoupling of routing policy from action execution. ZETA resolves scientific intent before invoking specialized visual tools, treating measurement as a deterministic operation grounded in explicit geometry rather than generative output. This architecture exposes the "black box" of agent decision-making, allowing routing correctness and downstream execution to be evaluated as independent failure modes.   To our knowledge, we provide the 
    
[^19]: EvoWorld：迈向生成式视频模型中的视野外世界演化

    EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models

    [https://arxiv.org/abs/2611.06748](https://arxiv.org/abs/2611.06748)

    提出EvoWorld模型与“视野外因果演化”新问题，使生成式视频模型不仅能渲染逼真画面，还能推理事件对世界实体的影响并生成因果一致的世界演化。

    

    生成式视频模型正日益被用作视觉世界模拟器，但它们的主要优势在于渲染逼真的像素，而非推理世界应如何演化。连贯的世界演化所需的不仅仅是视觉保真度：当一个事件发生时，模型必须推断其对世界中实体的影响，然后渲染出由此产生的状态。例如，当一头野兽出现后，模型不仅应该描绘这头野兽，还应该根据实际呈现的场景推断附近的人应该逃跑、躲避还是反击。现有的基于扩散模型的视频模型缺乏维护世界状态并对这类事件条件化后果进行推理的显式机制，常常产生视觉上合理但行为上静态或因果上不一致的输出。为了使这一推理鸿沟具体且可度量，我们提出了一个新问题——视野外因果演化，它要求模型能够对发生在视野之外的事件进行推理，并演化出因果一致的世界状态。

    arXiv:2611.06748v1 Announce Type: cross  Abstract: Generative video models are increasingly used as visual world simulators, yet their primary strength lies in rendering plausible pixels rather than reasoning about how a world should evolve. Coherent world evolution requires more than visual fidelity: when an event occurs, a model must infer its consequences for the entities in the world and then render the resulting state. For example, after a wild beast appears, the model should not only depict the wild beast, but also infer whether a nearby person should flee, take cover, or fight back based on the realized scene. Existing diffusion-based video models lack an explicit mechanism for maintaining world state and reasoning over such event-conditioned consequences, often producing visually plausible but behaviorally static or causally inconsistent outputs. To make this reasoning gap concrete and measurable, we propose a new problem, Out-of-Sight Causal Evolution, which requires a model t
    
[^20]: TailSteer：面向抗漂移多轮图像编辑的轨迹尾部引导

    TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing

    [https://arxiv.org/abs/2611.06741](https://arxiv.org/abs/2611.06741)

    TailSteer通过在去噪轨迹尾部集中施加保真校正（该校正从模型自身的在策略rollout中学习），在不改变早期去噪动力学和预训练先验的前提下，有效抑制多轮图像编辑中累积的视觉漂移。

    

    强大的单轮编辑性能并不能保证可靠的多轮交互：随着编辑输出被反复复用，微小的非预期变化会不断累积，最终演变为严重的视觉漂移。这种长期漂移通常通过对多轮数据或跨轮rollout进行训练来缓解，但这可能会改变预训练先验；或者借助难以扩展到基础模型的专用推理时机制。我们提出了TailSteer，一个最终阶段的训练框架，它能在提升多轮鲁棒性的同时，保留早期训练所获得的生成与编辑能力。TailSteer遵循一个简单的原则：保持早期去噪动力学接近基础模型，并将保真校正集中在轨迹尾部附近，从而抑制细粒度的偏差。这些校正从编辑器自身的在策略去噪rollout中学习，使用未发生变化的源内容……

    arXiv:2611.06741v1 Announce Type: cross  Abstract: Strong single-turn editing performance does not guarantee reliable multi-turn interaction: as edited outputs are repeatedly reused, small unintended changes can accumulate into severe visual drift. This long-term drift is often mitigated through training on multi-turn data or cross-turn rollouts, potentially altering pretrained priors, or through specialized inference-time mechanisms that are difficult to scale to foundation models. We introduce TailSteer, a final-stage training framework that improves multi-turn robustness while retaining the generative and editing capabilities acquired through earlier training. TailSteer follows a simple principle: keep early denoising dynamics close to those of the base model and concentrate preservation corrections near the trajectory tail, where fine-grained deviations can be suppressed. These corrections are learned from the editor's own on-policy denoising rollouts, using unchanged source conten
    
[^21]: 智能手机辅助的高强度间歇训练VR健身游戏：基于实时姿态估计的健身游戏化

    Smartphone-Aided VR Exergames for HIIT: Gamifying Fitness with Real-Time Pose Estimation

    [https://arxiv.org/abs/2611.06734](https://arxiv.org/abs/2611.06734)

    该论文提出了一种利用普通智能手机摄像头进行实时全身姿态估计的VR健身游戏系统，结合11个游戏化HIIT组合动作以及基于MCP协议和GPT-4o的后端生物力学数据分析，为高强度间歇训练提供了低成本、可扩展、无需穿戴设备的运动追踪与健身游戏化方案。

    

    本文介绍了一种创新的虚拟现实（VR）健身游戏系统，旨在通过11个预定义的组合动作（包括深蹲出拳组合、双臂挥击和对角拉伸组合）将高强度间歇训练（HIIT）游戏化。与通常依赖可穿戴动作捕捉套装或有限头显传感器的传统追踪方法不同，我们的系统使用放置在三脚架上的标准智能手机摄像头来捕捉全身运动。这种方法提供了一种经济实惠且可扩展的替代方案，特别适合HIIT训练中典型的快速、动态动作。智能手机拍摄的画面通过姿态估计算法进行实时处理，提取的生物力学数据由基于模型上下文协议（MCP）——Anthropic用于连接AI模型与结构化数据源和工具的开放标准——构建的后端服务器以及GPT-4o大语言模型进行分析。该系统……

    arXiv:2611.06734v1 Announce Type: cross  Abstract: This paper introduces an innovative virtual reality (VR) exergame system designed to gamify High-Intensity Interval Training (HIIT) workouts through 11 predefined combo moves, including squat-punch combinations, bilateral arm strikes, and diagonal stretch combos. Unlike conventional tracking methods that often rely on wearable motion capture suits or limited headset-based sensors, our system uses a standard smartphone camera placed on a tripod to capture full-body movement. This approach offers a cost-effective and scalable alternative, particularly well-suited to the fast, dynamic motions typical of HIIT routines.   The smartphone footage is processed in real time using a pose estimation algorithm, with extracted biomechanical data analyzed by a backend server built on the Model Context Protocol (MCP), Anthropic's open standard for connecting AI models to structured data sources and tools, and a GPT-4o large language model. The system
    
[^22]: 规模化ConceptARC：研究推理模型中的概念理解

    ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models

    [https://arxiv.org/abs/2611.06732](https://arxiv.org/abs/2611.06732)

    该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。

    

    大型推理模型（LRMs）在具有挑战性的基准测试中表现优异，但它们能否在概念相关的推理任务之间形成可泛化的抽象仍不明确。抽象与推理语料库（ARC）为研究抽象能力提供了一个受控的测试平台，要求求解者从少量示例中推断二维网格上的变换规则。虽然ARC评估的是单个任务内的规则抽象能力，但不同任务可能共享更高层次的结构，即便它们需要不同的变换——例如，多个任务可能都涉及对“被包含在其他对象中的对象”进行推理。像ConceptARC这样的基准通过将ARC风格的任务按反复出现的空间和语义概念进行分组来捕捉这种结构，但它们通常每个概念只包含少量手工构建的任务。我们提出了ConceptARC at Scale，这是一个生成器，可以创建数千个针对特定概念的ARC风格任务。

    arXiv:2611.06732v1 Announce Type: new  Abstract: Large reasoning models (LRMs) perform well on challenging benchmarks, yet it remains unclear whether they form generalizable abstractions across conceptually related reasoning tasks. The Abstraction and Reasoning Corpus (ARC) provides a controlled testbed for studying abstraction, requiring solvers to infer transformation rules over two-dimensional grids from a small set of examples. While ARC evaluates rule abstraction within individual tasks, different tasks may share higher-level structure despite requiring different transformations$\unicode{x2014}$for example, several tasks may involve reasoning about objects contained within other objects. Benchmarks such as ConceptARC capture this structure by grouping ARC-style tasks into recurring spatial and semantic concepts, but they often contain only a few manually constructed tasks per concept. We introduce ConceptARC at Scale, a generator for creating thousands of ARC-style tasks targeting
    
[^23]: 基础模型时代的灵巧机械手：硬件、数据、跨本体迁移、学习与评估的全栈式综述

    Dexterous Robotic Hands in the Foundation-Model Era: A Full-Stack Survey of Hardware, Data, Cross-Embodiment Transfer, Learning, and Evaluation

    [https://arxiv.org/abs/2611.06724](https://arxiv.org/abs/2611.06724)

    这是一篇全栈式综述，首次将基础模型时代的灵巧机械手研究系统性地组织为硬件、数据、跨本体迁移、学习与评估五个相互依赖的层次，并通过案例分析揭示各层设计选择之间的内在联系。

    

    灵巧机械手正日益与大规模人类数据和基础模型深度结合，然而硬件、表征学习与评估等方面的进展往往被割裂地单独评估。本综述回顾了2024年1月至2026年11月初的相关工作，并纳入更早期的研究以梳理技术发展脉络。我们将文献组织为五个相互依赖的层次：硬件；数据采集与整理；跨本体迁移；学习与基础模型；以及评估与应用。人手-物体交互理解与触觉感知提供了两个贯穿全文的横切视角。通过系统级架构、对比表格和五个案例分析，我们将各层次之间的设计选择串联起来。研究区分了单一研究内部的实验对比与异构语料库统计、厂商规格参数以及公司报告的结果。综述最终得出三点结论，其中第一点是：人类数据的价值取决于运动学、视觉等方面的特性。（原文摘要在此处截断）

    arXiv:2611.06724v1 Announce Type: cross  Abstract: Dexterous robotic hands are increasingly coupled to large-scale human data and foundation models, yet progress in hardware, representation learning and evaluation is often assessed separately. This survey reviews work from January 2024 to early November 2026, with earlier studies included to establish technical lineages. We organize the literature into five interdependent layers: hardware; data collection and curation; cross-embodiment transfer; learning and foundation models; and evaluation and applications. Human hand-object understanding and tactile sensing provide two cross-cutting perspectives. A system-level architecture, comparative tables and five case analyses connect design choices across these layers. We distinguish within-study experimental comparisons from heterogeneous corpus statistics, vendor specifications and company-reported results. Three conclusions emerge. First, the value of human data depends on kinematic, visua
    
[^24]: 多语言大语言模型中基于非参数记忆的推理：关系知识的认知分析

    Reasoning over Non-Parametric Memory in Multilingual LLMs: A Cognitive-Based Analysis of Relational Knowledge

    [https://arxiv.org/abs/2611.06715](https://arxiv.org/abs/2611.06715)

    该研究提出Rel-Bloom多语言认知诊断基准，在五个布鲁姆认知层次和三种真实性条件下对43个模型进行大规模测试，发现多语言大模型在基于上下文证据进行关系推理时，从“应用”到“分析”层次准确率急剧下降，揭示出模型对非参数记忆进行高阶推理的能力存在显著瓶颈。

    

    检索增强生成（RAG）假设语言模型能够对其上下文中陈述的关系进行推理，而不是退回到预训练期间记忆的关联上。我们单独研究了这种基于证据的关系推理，事实直接在提示中提供而非通过检索获取。Rel-Bloom是一个受控的多语言诊断工具，包含基于30个Wikidata关系构建的450道多项选择题；相同的源关系在五个基于布鲁姆认知分类法的层次（记忆、理解、应用、分析、评估）以及三种真实性条件（事实性、反事实、中性化）下进行实例化。在43个模型、6种语言和两种多语言设置（共计226,800个预测）的实验中，有三个发现尤为突出。首先，在每一个被评估的开源权重模型系列中，从“应用”到“分析”层次的准确率均急剧下降，大型Granite组的降幅达到47.2个百分点，但两个Gemini模式并非如此。

    arXiv:2611.06715v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) assumes that language models can reason over relations stated in their context rather than fall back on associations memorized during pretraining. We study this evidence-conditioned relational reasoning in isolation, with the facts supplied directly in the prompt rather than retrieved. \textsc{Rel-Bloom} is a controlled multilingual diagnostic of 450 multiple-choice questions built from 30 Wikidata relations; the same source relations are instantiated at five Bloom-inspired levels (Remember, Understand, Apply, Analyze, Evaluate) and under three veracity conditions (Factual, Counterfactual, Neutralized). Across 43 models, six languages, and two multilingual settings (226,800 predictions), three findings stand out. First, accuracy drops sharply from Apply to Analyze in every evaluated open-weight model series, reaching 47.2 percentage points for the Large Granite bucket, but not for either Gemini mode
    
[^25]: 超越原则与实践的差距：作为情境化认识论与伦理实践的负责任人工智能

    Beyond the principle-practice gap: Responsible AI as situated epistemic and ethical practice

    [https://arxiv.org/abs/2611.06709](https://arxiv.org/abs/2611.06709)

    本文基于对反人口贩卖AI系统从业者的访谈，论证“原则与实践差距”并非执行缺陷，而是源于不同制度条件下人们对伦理问题认知方式的差异，主张将负责任AI理解为一种情境化的认识论与伦理实践。

    

    负责任人工智能（Responsible AI）研究发现，伦理原则经常无法体现在人工智能的设计与开发过程中。这一现象被概念化为“原则与实践之间的差距”，并被界定为执行、意识或合规层面的问题，其言下之意是系统或个体行为的缺陷。本文提出了一种替代性解释。通过对设计和治理人工智能赋能的反人口贩卖系统的相关参与者——涵盖执法、商业、研究和幸存者支持等不同场景——进行的十四次半结构化访谈，本文论证了这种表面上的差距实际上源于多样而复杂的制度性与专业性条件，正是在这些条件下，参与者才将某些问题（而非其他问题）识别为需要采取行动的伦理问题。研究表明，认识论选择与伦理判断相互交织，并追溯了四个相互关联的环节——感知、判断、情境化行动与社会技术……

    arXiv:2611.06709v1 Announce Type: cross  Abstract: Responsible AI scholarship has found that ethical principles are frequently not reflected in AI design and development. Conceptualised as a `gap' between principle and practice, this is framed as a problem of implementation, awareness or compliance, and by implication, a deficit of systems or individual behaviour. This paper sets out an alternative explanation. Drawing on fourteen semi-structured interviews with actors designing and governing AI-enabled anti-trafficking systems across law enforcement, commercial, research, and survivor-support contexts, it argues that such apparent gaps emerge from the varied and complex institutional and professional conditions through which actors come to recognise some issues, but not others, as ethical problems requiring action. We show that epistemic choices are intertwined with ethical judgements, and trace four interconnected moments---perception, judgement, situated action and socio-technical m
    
[^26]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^27]: 不要替换，而要互补：一种面向近分布外检测的检测器无关子空间信号

    Don't Replace, Complement: A Detector-Agnostic Subspace Signal for Near-OOD Detection

    [https://arxiv.org/abs/2611.06688](https://arxiv.org/abs/2611.06688)

    该论文提出PLUS方法，利用稀疏自编码器为每个类别构建低维正交子空间并计算白化投影评分，作为与现有检测评分互补的近分布外检测信号，从而在不牺牲远分布外性能的前提下提升近分布外检测能力。

    

    事后分布外（OOD）检测在远分布外场景下已相当成熟，但在近分布外场景下仍然薄弱。这两种设定在类特征方向的激活上存在差异：远分布外输入大多无法激活其预测类别的类特征方向，而近分布外输入则会像分布内样本一样激活这些方向。由于用于区分远分布外的轴使得近分布外样本与分布内样本几乎无法区分，针对一种设定优化的评分与另一种设定并不匹配；在检测器已对远分布外达到饱和的情况下，新增的近分布外灵敏度在一定程度上是以牺牲远分布外性能为代价换来的。这种分歧促使我们设计一个专门针对近分布外的信号，以补充现有评分。我们提出了PLUS（Per-class Low-dimensional sUbspace Signal，每类低维子空间信号）。对于每个类别，PLUS从稀疏自编码器中选取m个最具类活性的神经元，将对应的解码器方向正交归一化以构成低维子空间，并通过每类白化投影对输入进行评分……（原文摘要在此处截断）

    arXiv:2611.06688v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection is mature on far-OOD yet remains weak on near-OOD. These two settings differ in their use of class-characteristic feature directions: far-OOD inputs largely fail to activate those of their predicted class, whereas near-OOD inputs activate them much as in-distribution samples do. Because the axis separating far-OOD leaves near-OOD almost indistinguishable from ID, a score optimized for one setting is misaligned with the other; where a detector already saturates far-OOD, added near-OOD sensitivity is bought partly against it. This divergence motivates a targeted near-OOD signal designed to complement existing scores. We propose PLUS (Per-class Low-dimensional sUbspace Signal). For each class, PLUS selects the $m$ most class-active neurons from a sparse autoencoder, orthonormalizes the corresponding decoder directions to form a low-dimensional subspace, and scores an input by its per-class whit
    
[^28]: LifeVal：评估AI模型在个性化智能体消费任务上的表现

    LifeVal: Evaluating AI Model Performance on Personalized Agentic Consumer Tasks

    [https://arxiv.org/abs/2611.06687](https://arxiv.org/abs/2611.06687)

    LifeVal是一个基于3800多名消费者支付意愿调查构建的基准，涵盖148类代表每年超1.7万亿美元支付意愿的消费任务，并利用真实ChatGPT对话和个人背景创建个性化任务来评估AI模型的智能体能力。

    

    我们提出了LifeVal，一个用于评估AI模型在个性化智能体消费任务上表现的基准。LifeVal涵盖了个性化数字超级助理能够自动化的最有价值的生活任务，涉及148个消费任务类别（例如管理财务、搬家、找工作、寻找托儿服务），代表每年超过1.7万亿美元的支付意愿。为了构建活动分布，我们针对具有人口统计学代表性的超过3800名消费者样本，开展了一项关于数字超级助理支付意愿的新调查，并将结果与美国时间使用调查（美国劳工统计局，2025）和跨国时间使用研究（Fisher等人，2025）进行交叉验证。针对最有价值的消费活动，我们招募了任务创建者，利用真实的ChatGPT对话和记忆、参考文件以及个人背景信息，创建了个性化的AI评估任务。

    arXiv:2611.06687v1 Announce Type: new  Abstract: We introduce LifeVal, a benchmark evaluating AI model performance on personalized agentic consumer tasks. LifeVal covers the most valuable life tasks that a personalized digital superassistant could automate, representing over 1.7 trillion US dollars in annual willingness-to-pay across 148 consumer task categories (e.g., managing finances, moving apartments, searching for a job, finding childcare). To construct our activity distribution, we conducted a new survey of willingness-to-pay for a digital superassistant across a demographically representative sample of over 3,800 consumers, cross-referencing our results against the American Time Use Survey (Bureau of Labor Statistics, 2025) and the Multinational Time Use Study (Fisher et al., 2025). For the most valuable consumer activities, we recruited task creators to create AI evaluation tasks personalized using real ChatGPT conversations and memories, reference files, and personal context.
    
[^29]: AgentTrustCI：面向Java LLM智能体的构建时策略、接地与轨迹测试

    AgentTrustCI: Build-Time Policy, Grounding, and Trajectory Tests for Java LLM Agents

    [https://arxiv.org/abs/2611.06674](https://arxiv.org/abs/2611.06674)

    AgentTrustCI提出了一个Java/JUnit 5框架，将策略、接地与轨迹检查以确定性重放的形式转化为可重复的构建时测试，从而在应用层验证LLM智能体的工具选择、答案依据与工作流程合规性，并在55个合成案例上准确分类了全部注入故障。

    

    基础设施安全并不能确定一个LLM智能体是否选择了被允许的工具、是否将其回答建立在工具输出之上，或是否遵循了必需的工作流程。AgentTrustCI是一个Java/JUnit 5框架，它将这些应用层期望表达为针对记录轨迹的策略检查、接地检查和轨迹检查。确定性重放实现支持可重复的构建时测试；裁判接口则为可选的语义评估提供了扩展点。在涵盖七个企业领域的55个合成案例上，完整流水线正确分类了全部30个注入故障和25个通过的控制案例（包括十个非平凡控制案例），并且在12个指定的关键案例中无一遗漏。每个单层配置正确分类了35个案例，因为语料库刻意为每一层隔离了十个故障。一个通过不允许的序列获得的接地回答说明了轨迹检查带来的额外覆盖能力。（摘要在此处截断）

    arXiv:2611.06674v1 Announce Type: cross  Abstract: Infrastructure security does not establish whether an LLM agent selected an allowed tool, grounded its answer in tool output, or followed a required workflow. AgentTrustCI is a Java/JUnit 5 framework that expresses these application-layer expectations as policy, grounding, and trajectory checks over recorded traces. A deterministic replay implementation supports repeatable build-time tests; a judge interface provides an extension point for optional semantic evaluation. On 55 synthetic cases spanning seven enterprise domains, the full pipeline correctly classifies all 30 injected faults and 25 passing controls, including ten non-trivial controls, and misses none of 12 designated critical cases. Each single-layer configuration correctly classifies 35 cases because the corpus deliberately isolates ten faults per layer. A grounded answer obtained through a disallowed sequence illustrates the additional coverage of trajectory checks. Repeat
    
[^30]: 搜索智能体在哪里偏离轨道：基于图的步骤级验证

    Where Search Agents Go Off the Rails: Graph-Grounded Step-Level Verification

    [https://arxiv.org/abs/2611.06662](https://arxiv.org/abs/2611.06662)

    GraphRail 将需求、证据和推理表示为结构化声明并进行对比，无需生成式评判者或标准答案即可对搜索智能体的中间推理步骤进行可追溯的验证，在多跳问答任务中比生成式验证器更有效地发现中间错误。

    

    搜索增强的推理系统能够得到正确答案，但其中间步骤可能存在错误。可靠的审计必须检查证据 $D$ 是否覆盖问题 $Q$、推理 $T$ 是否针对问题 $Q$、以及 $T$ 是否立足于证据 $D$。我们提出了 GraphRail，它将需求、证据和推理表示为结构化声明，并在无需生成式评判者或标准答案的情况下对它们进行比较。在三个多跳问答基准和三个推理框架上，GraphRail 相比生成式验证器提高了错误召回率，同时在随机抽取下保持稳定。其诊断结果还可以指导在线推理，为检测和纠正中间错误提供了一种可追溯的方法。

    arXiv:2611.06662v1 Announce Type: new  Abstract: Search-augmented reasoning systems can reach correct answers while making errors in intermediate steps. Reliable auditing must check whether evidence $D$ covers the question $Q$, reasoning $T$ addresses $Q$, and $T$ is grounded in $D$. We introduce GraphRail, which represents requirements, evidence, and reasoning as structured claims and compares them without a generative judge or gold answer. Across three multi-hop QA benchmarks and three reasoning frameworks, GraphRail improves error recall over generative verifiers while remaining stable under stochastic extraction. Its diagnoses can also guide online reasoning, providing a traceable way to detect and correct intermediate errors.
    
[^31]: 基于持续先验补偿的人体运动预测

    Human Motion Prediction via Continual Prior Compensation

    [https://arxiv.org/abs/2611.06637](https://arxiv.org/abs/2611.06637)

    提出了一种持续先验补偿（CPC）框架，将不同时刻的人体运动预测任务分解为多个子任务并进行多阶段渐进式训练，同时通过可学习的随机变量补偿先验信息遗忘，从而避免短期预测的学习受到长期预测的干扰。

    

    人体运动预测（HMP）旨在根据观察到的过去运动序列，预测未来不同时刻的人体姿态。以往的方法主要将不同时间时刻的预测视为单一的预测任务，并同时学习各个时刻的预测，这会遇到一个主要局限：短期预测（即“近期未来”预测）的学习可能受到长期（即“远期未来”）运动预测的阻碍。在本文中，我们开发了一种新颖的时间持续学习框架，称为持续先验补偿（CPC），以渐进式地训练HMP模型。在该框架中，我们将对应于不同时间时刻的运动预测任务划分为若干子任务，并以多阶段的方式训练模型。为了缓解渐进式训练过程中的先验信息遗忘问题，我们进一步引入了一个可学习的随机变……

    arXiv:2611.06637v1 Announce Type: cross  Abstract: Human Motion Prediction (HMP) aims to predict future human poses at different moments according to observed past motion sequences. Previous approaches mainly treated the prediction of different temporal moments as a single prediction task and learned the predictions of varied moments simultaneously, which would encounter a main limitation: the learning of short-term predictions (referring to ``near-future'' prediction) could be hindered by the predictions of long-term (referring to ``far-future'' prediction) motions. In this paper, we develop a novel temporal continual learning framework called Continual Prior Compensation (CPC) to progressively train HMP models, in which we divide the prediction task of motions corresponding to varied temporal moments into several subtasks and train the model in a multi-stage manner. To mitigate the prior information forgetting in the progressive training, we further introduce a learnable random varia
    
[^32]: “不给糖就捣蛋”：使用合成图像对多模态医学推理进行对抗性评估

    Trick-or-Treat: Adversarial Evaluation of Multimodal Medical Reasoning with Synthetic Images

    [https://arxiv.org/abs/2611.06631](https://arxiv.org/abs/2611.06631)

    该论文提出Trick-or-Treat基准，通过将真实胸部X光片与合成的临床照片配对构成对抗性多模态评测框架，有效暴露了前沿模型在跨模态医学推理中的幻觉行为。

    

    我们利用医学图像合成技术创建了一个具有挑战性的多模态评估框架，用以揭示前沿模型中的幻觉行为。评估多模态推理历来十分困难，因为现有的医学数据集很少将不同的图像模态配对，例如放射影像与外部临床照片。为了弥补这一空白，我们推出了 Trick-or-Treat，这是一个对抗性的多模态医学基准，它将真实的胸部X光片与合成的临床照片配对，以挑战模型的跨模态推理能力。针对广泛的疾病类别，我们创建了两对图像，每对图像共享同一张胸部X光片，但包含不同的外部照片。在匹配对中，胸部X光片和照片都包含提示同一疾病的发现（例如，肺部阴影和在水痘中可见的结痂皮肤病变共同提示水痘肺炎）。在不匹配对中，照片……

    arXiv:2611.06631v1 Announce Type: cross  Abstract: We used medical image synthesis to create a challenging multimodal evaluation framework that exposes hallucination behaviors in frontier models. Evaluating multimodal reasoning has historically been difficult because existing medical datasets rarely pair different image modalities, such as radiographs and external clinical photographs. To address this gap, we introduce Trick-or-Treat, an adversarial multimodal medical benchmark that pairs real chest radiographs with synthetic clinical photographs to challenge models' cross-modal reasoning. For each of a wide range of conditions, we created two pairs of images, where the pairs share a single chest radiograph but contain different external photographs. In matched pairs, the chest radiograph and photograph both contain findings suggesting the same disease (e.g., varicella pneumonia suggested by lung opacities and crusted skin lesions seen in chickenpox). In mismatched pairs, the photograp
    
[^33]: 有限测量预算下量子电路玻恩机器的噪声正则化训练

    Noise-regularized training of quantum circuit Born machines under a finite shot budget

    [https://arxiv.org/abs/2611.06630](https://arxiv.org/abs/2611.06630)

    该论文证明硬件噪声可作为量子电路玻恩机器的正则化手段以提升泛化能力，并提出GENIE框架在有限测量预算内自动选择最优噪声水平。

    

    量子电路玻恩机器能够学习经典计算机难以采样的分布，但其价值取决于能否从有限的训练数据中进行泛化。本文研究了硬件噪声如何影响这种泛化能力。我们证明，对于任何噪声过程，使用最大均值差异训练的QCBM的泛化差距受该噪声模型所能达到的分布复杂度限制。扰动的局域泡利噪声会收缩这种复杂度，但会使模型偏向均匀分布，因此残余噪声起到正则化的作用，其最优强度可以是中等水平。我们提出了GENIE框架，该框架通过有上限的误差缓解实现所选定的噪声水平，并在固定的执行预算内于训练过程中从留出数据中选择噪声水平，且具有可证明的有界遗憾。在合成数据和人口普查数据的模拟中，GENIE的表现始终接近四种噪声控制策略中的最优者，而任何固定的噪声选择都可能是次优的。

    arXiv:2611.06630v1 Announce Type: cross  Abstract: Quantum circuit Born machines (QCBMs) can learn distributions that are hard to sample from classically, but their value depends on generalizing from finite training data. Here, we study how hardware noise affects this generalization. We show that, for any noise process, the generalization gap of a QCBM trained with the maximum mean discrepancy is bounded by the complexity of the distributions the noisy model can reach. Scrambled local Pauli noise contracts this complexity but biases the model towards the uniform distribution, so residual noise acts as a regularizer whose optimal strength can be intermediate. We propose GENIE, a framework that realizes a chosen noise level with capped error mitigation and selects it during training from held-out data within a fixed execution budget, with provably bounded regret. In simulations on synthetic and census data, GENIE stays close to the best of four noise-control strategies, whereas every fix
    
[^34]: 前沿AI能否增强网络防御？来自“玻璃翼项目”的证据

    Does Frontier AI Strengthen Cyber Defense? Evidence from Project Glasswing

    [https://arxiv.org/abs/2611.06629](https://arxiv.org/abs/2611.06629)

    本文利用“玻璃翼项目”作为准自然实验，发现前沿AI漏洞发现模型使获得访问权限的美国科技公司漏洞披露翻倍、高影响漏洞披露增加约150%，外部赏金研究者将报告转向其他公司，且未见攻击者利用额外漏洞的证据，表明前沿AI目前主要强化了网络防御而非攻击。

    

    防御者与攻击者竞相寻找相同的软件缺陷。本文利用“玻璃翼项目”的启动来研究获取前沿AI如何影响防御者、外部研究者和攻击者——该项目让一小部分美国大型科技公司提前获得了Claude Mythos Preview，一种用于发现软件漏洞的模型。基于公开披露的全部漏洞数据，我发现获得访问权限的公司相对于没有访问权限的中国和俄罗斯科技公司，其漏洞披露数量增加了一倍以上，披露的高影响漏洞多出约150%，且修复已报告缺陷的速度与以前一样快。由于受影响的公司自己发现了更多缺陷，通过漏洞赏金出售研究结果的外部研究者在受影响公司代码中报告的漏洞减少了约一半，而在其他公司代码中报告的漏洞增多，且并未退出市场。没有证据表明攻击者利用了这些额外的漏洞：被利用的份额（摘要在此处截断）

    arXiv:2611.06629v1 Announce Type: cross  Abstract: Defenders and attackers race to find the same software flaws. I use the launch of Project Glasswing, which gave a small group of large US tech firms early access to Claude Mythos Preview, a model for finding software vulnerabilities, to study how access to frontier AI affects defenders, external researchers, and attackers. Using the universe of publicly disclosed vulnerabilities, I find that firms with access more than doubled their disclosures relative to Chinese and Russian tech firms without access, disclosed about 150 percent more high-impact vulnerabilities, and fixed reported flaws as quickly as before. As treated firms found more flaws themselves, external researchers, who sell their findings through bug bounties, reported about half as many vulnerabilities in treated firms' code and more in other firms' code, without leaving the market. There is no evidence that attackers exploited the additional vulnerabilities: the share expl
    
[^35]: 从经认证的成对映射到经认证的枢纽：由推理机校验的信息交换标准交叉映射

    From Certified Pairs to a Certified Hub: Reasoner-Checked Crosswalks for the Information Exchange Standard

    [https://arxiv.org/abs/2611.06628](https://arxiv.org/abs/2611.06628)

    该论文将交叉映射的有效性精确定义为合并本体的连贯性与保守性，并通过“匹配—人工审核—SKOS对应认证—HermiT推理”的分阶段流程校验信息交换标准（IES）通向十三个本体的映射，首次实现了从经认证的成对映射升级到经认证的整个枢纽。

    

    国家数字孪生被构想为一个由多个孪生体组成、通过映射连接到共同基础的生态系统，但承诺确认这些映射有效性的路线图却从未说明“有效”究竟意味着什么。我们借鉴基于逻辑的对齐修复方法，为这一检验赋予了精确的形式——合并后本体的连贯性与保守性——并将其应用于信息交换标准（IES），先逐对检验，再扩展到整个枢纽。从IES出发、锚定于建筑本体BOT、Brick和SAREF4BLDG、通向七个行业领域组中十三个本体的交叉映射，依次经过匹配、人工审核、针对两个类层次结构对每条SKOS对应关系进行认证、以及使用HermiT进行推理等相互独立的阶段。逐对检验时，各阶段捕获的失败各不相同：仅靠认证就消除了原始交叉映射中全部526处保守性违反，仅靠审核则遗留了73处，而二者结合使十二个存在重叠的映射对中的十一个实现了连贯。

    arXiv:2611.06628v1 Announce Type: new  Abstract: A national digital twin was conceived as an ecosystem of twins connected through mappings to a common foundation, but the roadmap that promises to confirm those mappings are valid never says what valid means. We give the test a precise form, coherence and conservativity of the merged ontology, borrowed from logic-based alignment repair, and apply it to the Information Exchange Standard (IES), first pair by pair and then to whole hubs. Crosswalks from IES into thirteen ontologies across seven sector groups, anchored on the building ontologies BOT, Brick, and SAREF4BLDG, pass through separate stages of matching, human review, certification of each SKOS correspondence against the two class hierarchies, and reasoning with HermiT. Pair by pair, the stages catch different failures: certification alone removed all 526 conservativity violations of the raw crosswalks, review alone left 73, and together they made eleven of twelve overlapping pairs
    
[^36]: 基于 Cosmos 的面向物理 AI 的交互式世界仿真

    Interactive World Simulation with Cosmos for Physical AI

    [https://arxiv.org/abs/2611.06627](https://arxiv.org/abs/2611.06627)

    该论文提出了一个将 Cosmos 世界基础模型适配为实时交互式世界仿真的共享框架，使用户和具身智能体能够通过动作与场景控制同演化的虚拟环境交互，并支持相机导航、机器人操作、可控视频迁移和多视角自动驾驶仿真等应用。

    

    我们提出了一个共享框架，将 Cosmos 世界基础模型适配用于物理 AI 中的实时交互式世界仿真。通过根据动作和场景控制流式生成视觉观察结果，该框架使用户和具身智能体能够与不断演化的虚拟环境进行交互。它既支持预测动作的视觉后果，也支持根据外部模拟器维护的物理状态生成观察结果。我们在相机导航与机器人操作、可控视频迁移以及多视角自动驾驶汽车（AV）仿真等场景中演示了这一方法。这些示例为开发用于物理 AI 训练与评估的交互式虚拟环境提供了共享基础。我们的代码和模型检查点可在 https://github.com/nvidia/cosmos 和 https://huggingface.co/collections/nvidia/cosmos3-examples 获取。项目网站可在 https://research.nvidia.com 访问。

    arXiv:2611.06627v1 Announce Type: new  Abstract: We present a shared framework that adapts Cosmos world foundation models for real-time interactive world simulation in Physical AI. By streaming visual observations in response to actions and scene controls, the framework enables users and embodied agents to interact with evolving virtual environments. It supports both predicting the visual consequences of actions and generating observations from physical state maintained by an external simulator. We demonstrate this approach across camera navigation and robot manipulation, controllable video transfer, and multi-view autonomous vehicle (AV) simulation. These examples provide a shared foundation for developing interactive virtual environments for Physical AI training and evaluation. Our code and model checkpoints are available at https://github.com/nvidia/cosmos and https://huggingface.co/collections/nvidia/cosmos3-examples. The project website is available at https://research.nvidia.com/
    
[^37]: 从重建到插补：针对时间序列缺失通道的测试时优化

    From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels

    [https://arxiv.org/abs/2611.06612](https://arxiv.org/abs/2611.06612)

    该论文提出一种测试时优化方法，将冻结的序列重建器在推理时转化为插补器——把缺失条目作为优化变量、仅最小化观测坐标上的重建误差，从而无需人工掩码增强或全通道丢弃训练即可应对完整传感器通道失效导致的时间序列缺失。

    

    完整的传感器通道失效会造成一种困难的时间序列缺失形式：局部插值方法不可用，而为每种失效模式单独部署一个训练好的插补器也可能不切实际。我们探讨这样一个问题：一个未经人工掩码增强训练、随后被冻结的序列重建器，能否在推理阶段被转换为插补器。我们的方法将每个测试窗口中的缺失条目视为优化变量，固定所有观测值，并通过冻结的模型进行反向传播，仅在观测坐标上最小化重建误差。由此得到的“观测自洽性”目标既不需要重建器训练时的人工掩码增强，也不需要全通道丢弃。我们通过从缺失输入到观测残差的交叉雅可比矩阵刻画了这种精调何时能够提供有效信息，并在该耦合较弱时引入可选的时间平滑先验。在五个真实世界……（摘要截断）

    arXiv:2611.06612v1 Announce Type: cross  Abstract: Complete sensor-channel failures create a difficult form of time-series missingness: local interpolation is unavailable, and deploying a separately trained imputer for every failure pattern may be impractical. We ask whether a sequence reconstructor trained without artificial mask augmentation and then frozen can be converted into an imputer at inference time. Our method treats the missing entries of each test window as optimization variables, clamps all observations, and backpropagates through the frozen model to minimize reconstruction error only on observed coordinates. The resulting observed-self-consistency objective requires neither artificial mask augmentation nor full-channel dropout during reconstructor training. We characterize when this refinement is informative through the cross-Jacobian from missing inputs to observed residuals, and use an optional temporal-smoothness prior when that coupling is weak. Across five real-worl
    
[^38]: 时分复用：数字生命的运行节律与时间组织层

    Time-Division Multiplexing: The Operational Rhythm and Temporal Organization Layer of Digital Life

    [https://arxiv.org/abs/2611.06603](https://arxiv.org/abs/2611.06603)

    本文受灵长类皮层快速交替感知表征的启发，提出了一种时分复用式的时间组织层，通过相位标记微时段、可观测占空比与三个嵌套时钟，将数字生命智能体的忙碌与空闲通道统一为同一过程的不同阶段，并给出了占空比更新的确定性与随机下降保证。

    

    arXiv:2611.06603v1 公告类型：新论文 摘要：持续型智能体必须处理紧急交互，同时保留备选方案、重放错误并提交安全的长期变更。此前的数字生命框架区分了在线（忙碌）与离线（空闲）通道，但未说明二者究竟是独立程序还是同一过程的不同阶段，以及它们的操作应如何排序。受近期灵长类动物皮层中快速交替感知表征观察的启发，我们开发了一个时间组织层，该层具有带相位标签的微时段、可观测的占空比、相互独立的内容与资源控制，以及将候选探索与持久承诺分离的三个嵌套时钟。我们给出了投影占空比更新方法，陈述了其确定性下降保证以及实现随机保证所需的额外噪声项，并规定了相耦合的忙碌/空闲执行契约。五个合成实验（每个实验64个随机种子）检验了下降与噪声、目标……（摘要原文在此处截断）

    arXiv:2611.06603v1 Announce Type: new  Abstract: Continual agents must handle urgent interaction while retaining alternatives, replaying errors, and committing safe long-term changes. A previous digital-life framework distinguishes online (busy) and offline (idle) channels, but leaves open whether these are separate programs or phases of a common process and how their operations should be ordered. Motivated by recent observations of rapidly alternating perceptual representations in primate cortex, we develop a temporal organization layer with phase-tagged micro-epochs, an observable duty cycle, independent content and resource controls, and three nested clocks separating candidate exploration from durable commitment. We give a projected duty-cycle update, state its deterministic descent guarantee and the additional noise term required for a stochastic guarantee, and specify coupled busy/idle execution contracts. Five synthetic experiments (64 seeds each) examine descent and noise, obje
    
[^39]: 签名、封缄、泄露：MCP 工具链中超越签名的信任

    Signed, Sealed, Leaked: Trust Beyond Signature for MCP Toolchains

    [https://arxiv.org/abs/2611.06585](https://arxiv.org/abs/2611.06585)

    提出“种系执行”机制：每次工具调用都在从无数据的种系进程分叉出的临时副本中运行，且只接收其目的地被授权的信息，从而无需检查或信任工具实现即可防御已签名但恶意的 MCP 工具的隐蔽数据泄露攻击。

    

    AI 智能体依赖第三方工具，例如模型上下文协议（MCP）服务器，这些工具的发布者可被身份认证，其执行也可受到运行时安全策略的限制。然而，真实性并不保证行为是良性的：一个合法签名的工具可能在执行其宣称功能的同时，隐藏着泄露机密信息的恶意代码。我们用一个已签名的 MCP 工具演示了这种信任失效——该工具将敏感数据嵌入到发往策略所允许目的地的请求中。Meta 的 MCPGuard-Dynamic 作为当前最先进的运行时防御，在同一运行中能够拦截直接的渗出尝试，却在每一次试验中都无法阻止这种攻击。我们提出了种系执行，它无需检查或信任工具实现即可限制信息泄露。每次工具调用都在从无数据的种系进程分叉出的临时副本中运行，每个副本只接收其目的地被授权获取的信息。即使……

    arXiv:2611.06585v1 Announce Type: cross  Abstract: AI agents rely on third-party tools, such as Model Context Protocol (MCP) servers, whose publishers can be authenticated and whose execution can be restricted by runtime security policies. Yet authenticity does not guarantee benign behaviour: a legitimately signed tool may perform its advertised function while concealing malicious code that leaks confidential information. We demonstrate this trust failure with a signed MCP tool that embeds sensitive data in requests to a policy-allowed destination. Meta's MCPGuard-Dynamic, a state-of-the-art runtime defence, blocks a direct exfiltration attempt in the same run, yet fails to prevent this attack in every trial. We introduce germline execution, which limits information disclosure without inspecting or trusting tool implementations. Each tool invocation runs in ephemeral copies forked from a data-free germline process, each receiving only information authorised for its destination. Even if
    
[^40]: 一个量表，四种文本：已发表分级方案中的缺口、重叠与静默修补

    One Scale, Four Texts: Gaps, Overlaps and Silent Repairs in Published Grading Schemes

    [https://arxiv.org/abs/2611.06569](https://arxiv.org/abs/2611.06569)

    该论文将糖尿病视网膜病变分级的四份权威文本统一形式化为基于24个发现变量的规则语言，并通过对14亿余个病例组合的穷举枚举首次严格证明：每份文本在任何解读下都存在无法归类的病例，且2003版与2017版之间存在无法互相映射的类别缺口。

    

    类别系统通常以文本形式传递到用户手中，而读者在应用时会对其进行心照不宣的修补。我们以糖尿病视网膜病变分级为研究对象，探讨一旦移除这些修补，此类文本本身究竟规定了什么；在该领域中，同一个国际量表以多份文档的形式流传。我们将四份文本（2025年的英文筛查定义、2017年制表版与2003年发表版的国际量表，以及2015年的一份意大利指南）转录为一种基于24个发现变量的统一规则语言，并将每份文本遗留的每一个未决之处都命名为一个明确的精确化。通过对1,425,807,360个发现特征组合的穷举枚举（由SMT求解器重新推导并经OWL推理器抽查验证），结果表明：任何赋予处置动作的文本，在任何解读下都会使某些眼睛得不到类别归属；2017年制表版存在一个按字面解读为空的增殖期类别，以及一个2003年文本所没有的黄斑缺口，因而不存在任何函数能将2003年的分级映射到201……

    arXiv:2611.06569v1 Announce Type: new  Abstract: Category systems often reach their users as text, and readers apply them with tacit repairs. We ask what such a text determines once the repairs are removed, using diabetic retinopathy grading, where one international scale circulates in several documents. We transcribe four texts (the English screening definitions of 2025, the international scale as tabulated in 2017 and as published in 2003, and an Italian guideline of 2015) into one rule language over 24 finding variables, and make every point a text leaves open a named precisification. Exhaustive enumeration of 1,425,807,360 finding profiles, re-derived by an SMT solver and spot-checked by an OWL reasoner, shows that every text that assigns actions leaves some eyes without a category under every reading; that the 2017 tabulation has a proliferative class that is empty when read literally and a macular gap that the 2003 text lacks, so that no function maps the 2003 grades onto the 201
    
[^41]: RAGAM-AI：一种由生成式AI驱动的混合强化学习智能体方法，用于边缘资源管理

    RAGAM-AI: A Hybrid Reinforcement Learning based Agentic approach powered by Generative AI for Edge Resource Management

    [https://arxiv.org/abs/2611.06554](https://arxiv.org/abs/2611.06554)

    提出RAGAM-AI混合框架，将强化学习与生成式AI结合构建智能体工作流，利用LLM作为奖励预言机解决奖励泛化问题，以优化多接入边缘计算单元的资源分配。

    

    对于托管多种服务的分布式边缘网络，强化学习（RL）已被高效地用于资源管理。然而，复杂工作流中的奖励泛化问题仍是RL领域的一个开放性难题。为此，本工作提出了RAGAM-AI，这是一个融合强化学习算法与生成式AI（Gen-AI）能力的混合框架，用于驱动一个智能体AI工作流，以优化多接入边缘计算（MEC）单元的资源分配。该框架通过利用大语言模型（LLM）提供反馈，进一步解决了奖励泛化问题。该工作实现了以MathΣtral作为奖励预言机的多智能体Soft-Actor-Critic（MASAC）算法来管理MEC处的资源分配。RAGAM-AI能够解读系统状态信息并评估多目标权衡，从而指导分布式边缘服务器智能体上的资源编排循环。该框架的复杂性被进一步……（原文摘要此处截断）

    arXiv:2611.06554v1 Announce Type: new  Abstract: Resource management for distributed edge networks hosting different services has been handled efficiently using Reinforcement learning (RL). Reward generalization in complex workflows is still an open issue within the RL realm. Therefore, this work proposes RAGAM-AI, a hybrid framework that combines the capabilities of RL algorithms and Generative-AI (Gen-AI) for powering an Agentic AI workflow that optimizes resource allocation at the Multi-access Edge Computing (MEC) units. This framework further resolves the reward generalization issue by utilizing Large Language Model (LLM) for feedback. A Multi- Agent Soft-Actor Critic (MASAC) using Math{\Sigma}tral as a reward oracle is implemented for managing resource allocations at the MEC. RAGAM-AI interprets system state information and evaluates multi-objective trade-offs guiding the resource orchestration loop over distributed edge server agents. The complexity of the framework is further br
    
[^42]: 减法加成：迈向缓解大语言模型中脆弱的安全对齐

    Addition by Subtraction: Toward Mitigating Fragile Safety Alignment in Large Language Models

    [https://arxiv.org/abs/2611.06553](https://arxiv.org/abs/2611.06553)

    该论文提出Addition by Subtraction方法，通过在微调中逐步移除安全关键神经元并结合偏好优化，使安全对齐能力分散到模型的其他部分，从而缓解大语言模型安全对齐过度依赖稀疏组件的脆弱性问题，显著提升对剪枝等攻击的鲁棒性。

    

    大语言模型（LLMs）通过安全对齐来防止生成有害内容，但它们仍然容易受到越狱攻击。近期研究揭示了一个更深层次的结构性弱点：安全对齐可能是高度稀疏的，以至于仅剪枝少量安全关键神经元或消融单一的拒绝方向，就能大幅破坏对齐后的行为。在这项工作中，我们迈出了强化安全对齐的第一步，使其不再过度依赖于稀疏的模型组件。我们提出Addition by Subtraction（减法加成）方法，该方法在微调过程中逐步移除已识别的安全关键神经元，并利用偏好优化通过模型的其他部分重新建立安全对齐行为。我们的结果表明，由此得到的模型对基于剪枝的攻击具有显著更强的鲁棒性。值得注意的是，这些收益超越了启发本设计的白盒攻击场景：Addition by Subtraction 提升了……的鲁棒性（原文摘要至此截断）。

    arXiv:2611.06553v1 Announce Type: new  Abstract: Large language models (LLMs) undergo safety alignment to prevent harmful generations, yet they remain vulnerable to jailbreak attacks. Recent work reveals a deeper structural weakness: safety alignment can be highly sparse, such that pruning a small number of safety-critical neurons or ablating a single refusal direction can substantially undermine aligned behavior. In this work, we take the first steps toward strengthening safety alignment such that it is less dependent on sparse model components. We propose Addition by Subtraction, which gradually removes identified safety-critical neurons during finetuning and uses preference optimization to re-establish safety-aligned behavior through other parts of the model. Our results show that the resulting models are substantially more robust to pruning-based attacks. Remarkably, these benefits generalize beyond the white-box attacks motivating our design: Addition by Subtraction improves robus
    
[^43]: 经验二次利用：用于特权式令牌级信用分配的动态过程评分准则

    Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment

    [https://arxiv.org/abs/2611.06543](https://arxiv.org/abs/2611.06543)

    提出在线评分准则自蒸馏（ORSD）方法，从同策略轨迹的跨轨迹行为差异中自生成动态自然语言过程评分准则，实现密集的令牌级信用分配监督，以增强以结果为中心的强化学习。

    

    智能体轨迹通常同时包含成功的决策与失败的尝试，然而基于稀疏轨迹级奖励的优化难以识别哪些关键决策应当被强化或抑制。我们观察到，同策略轨迹之间有意义的行为差异可以被提炼为自然语言形式的过程评分准则，并作为过程监督信号使用。基于这一洞察，我们提出了在线评分准则自蒸馏，该方法利用自生成的过程评分准则提供密集的令牌级监督，从而增强以结果为中心的强化学习。具体而言，当前策略针对同一组轨迹，在每个步骤独立生成多个评分准则包，从跨轨迹的行为差异中提取候选过程评分准则。随后，一个冻结的验证器针对组内每条轨迹评估所有评分准则项，并基于捆绑…（摘要在此处被截断）

    arXiv:2611.06543v1 Announce Type: new  Abstract: Agent trajectories often contain both successful decisions and failed attempts, yet optimization with sparse trajectory-level rewards struggles to identify which critical decisions should be reinforced or suppressed. We observe that meaningful distinctions among on-policy trajectories can be distilled into natural-language process rubrics and used as process supervision signals. Based on this insight, we propose Online Rubric Self-Distillation (ORSD), which leverages self-derived process rubrics for dense token-level supervision, thereby augmenting outcome-centric reinforcement learning. Specifically, the current policy independently generates multiple rubric bundles at each step according to the same group of trajectories, extracting candidate process rubrics from cross-trajectory behavioral differences. A frozen verifier then evaluates all rubric items against each trajectory in the group, and selects one candidate bundle based on bund
    
[^44]: 从准入到留存：在质量多样性搜索中预测持久留存的子代个体

    From Admission to Retention: Predicting Durable Offspring in Quality-Diversity Search

    [https://arxiv.org/abs/2611.06522](https://arxiv.org/abs/2611.06522)

    该论文研究了质量多样性搜索中的信用分配问题，发现繁殖前可获得的信号（尤其是谱系深度）能够预测子代个体是仅短暂进入档案还是在多代之后仍被持久保留，且同一信号对准入与留存的影响方向往往相反。

    

    自动化失效发现可以被构建为一个多样性问题：其目标是揭示系统失效的多种不同方式，而非单一的临界对抗样本。然而，均匀的父代采样无法将搜索精力分配给那些可能对档案产生持久贡献的个体。我们在一个非结构化的质量多样性档案中研究了这一信用分配问题，检验在个体繁殖之前可获得的信号能否区分那些仅仅进入档案的子代与那些在数代之后仍被保留的子代。在一个平面机械臂基准任务和两个机器人运动任务中，相同的信号往往以相反的方向与准入和留存相关联。特别地，谱系深度是最清晰的此类信号，且其留存区分能力随种群容量的增大而增强，这与更低的档案更替率相关联。

    arXiv:2611.06522v1 Announce Type: cross  Abstract: Automated failure discovery can be framed as a diversity problem: the objective is to expose distinct ways a system fails rather than a single critical adversarial case. Yet uniform parent sampling provides no way to allocate search effort toward individuals likely to produce durable archive contributions. We study this credit assignment problem in an unstructured Quality-Diversity (QD) archive, testing whether signals available before an individual has reproduced distinguish offspring that merely enter the archive from those still retained generations later. Across a planar-arm benchmark and two robotic-locomotion tasks, the same signals are often associated with entry and persistence in opposite directions. In particular, lineage depth is the clearest such signal, and its persistence discrimination grows with population capacity, which is associated with lower turnover. Biasing parent selection toward the signals with the strongest m
    
[^45]: IDSISA：面向高效机器遗忘的影响蒸馏分片训练方法

    IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning

    [https://arxiv.org/abs/2611.06519](https://arxiv.org/abs/2611.06519)

    提出IDSISA方法，利用影响函数将易被遗忘的样本集中到少数分片，并通过未受影响分片的蒸馏恢复效用，在保持SISA精确删除保证的同时提升机器遗忘的效率和模型性能。

    

    机器遗忘旨在移除特定训练样本对模型的影响，以符合隐私法规。完全重训能提供强保证，但在大规模场景下计算代价过高；近似方法虽然高效，却缺乏形式化保证。SISA（分片、隔离、切片与聚合训练）通过数据分片和独立重训实现了可认证的删除，但随机划分会将遗忘请求分散到大多数分片中，且隔离重训会降低模型效用。我们提出IDSISA（影响蒸馏SISA），利用影响函数将易被遗忘的样本集中到少数分片中，并通过来自未受影响分片的蒸馏来恢复因隔离训练而损失的效用。从SISA重训步骤继承的删除保证依然精确，无需任何概率性限定；而基于蒸馏的效用恢复则是一个单独量化、经验上有界的近似（ε...（摘要原文在此处截断）

    arXiv:2611.06519v1 Announce Type: cross  Abstract: Machine unlearning removes the influence of specific training examples to comply with privacy regulations. Full retraining gives strong guarantees but is computationally prohibitive at scale; approximate methods are efficient but lack formal guarantees. SISA (Sharded, Isolated, Sliced, and Aggregated training) achieves certified deletion through data sharding and independent retraining, but random partitioning scatters forget requests across most shards, and isolated retraining degrades utility. We introduce IDSISA (Influence-Distilled SISA), which uses influence functions to concentrate forget-prone samples into few shards and distillation from unaffected shards to recover utility lost to isolation. The deletion guarantee inherited from SISA's retraining step remains exact and requires no probabilistic qualification; the distillation-based utility recovery is a separately quantified, empirically bounded approximation ($\epsilon_{\text
    
[^46]: EmoReward：面向情感视频生成强化学习的自进化奖励校准

    EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL

    [https://arxiv.org/abs/2611.06518](https://arxiv.org/abs/2611.06518)

    EmoReward 提出一种自进化奖励校准机制，通过从训练过程的奖励轨迹中提炼证据可靠性来自适应调整多奖励权重分配，解决了固定奖励难以捕捉情感意图且易引发奖励作弊的问题。

    

    组相对策略优化（GRPO）已在代码和数学等可验证领域推动了生成模型的后训练，但视频质量（尤其是情感表达）的评估往往依赖于主观美学，缺乏唯一正确的目标。现有方法通常依赖针对视觉质量、运动质量和文本对齐的固定奖励，这些奖励难以捕捉情感意图，并可能引发奖励作弊；而基于人类反馈的强化学习（RLHF）难以扩展，且受个体审美差异影响，使得在不同叙事情境下校准面部、肢体动作和色彩线索变得复杂。我们提出了 EmoReward，一种全新的自进化训练机制，它从实际训练过程中收集的奖励轨迹中提炼证据可靠性，通过判定哪些情感证据可信以及应赋予多大权重，自适应地引导后续训练中的多奖励权重分配。

    arXiv:2611.06518v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) has advanced generative model post-training in verifiable domains such as code and mathematics, but evaluating video quality, especially emotional expression, often depends on subjective aesthetics and has no uniquely correct target. Existing approaches typically rely on fixed rewards for visual quality, motion quality, and text alignment that poorly capture emotional intent and can invite reward hacking, while RLHF is difficult to scale and subject to individual aesthetic differences that complicate the calibration of facial, body motion, and chromatic cues across narrative contexts. We introduce EmoReward, a new self-evolving training mechanism that distills evidence reliability from reward traces collected during an actual training run to adaptively guide multi-reward weight allocation in subsequent training by determining which emotional evidence to trust and how much to weight it. EmoRewar
    
[^47]: ElecLoad：由多重事件引导的自适应电力负荷预测

    ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events

    [https://arxiv.org/abs/2611.06508](https://arxiv.org/abs/2611.06508)

    提出了ElecLoad——一个上下文感知的智能体，能够利用网页通知、天气预警等多重事件的影响及其相互作用，自适应地组织电力负荷预测过程，从而提升预测的准确性。

    

    准确的电力负荷预测为能源采购、发电调度和供需平衡提供关键的决策支持，有助于提升电力系统的经济效率和运行可靠性。然而，未来的电力需求可能受到网页通知、天气预警和活动日程中描述的多重事件之间复杂相互作用的影响。这些事件的影响随时间变化，且可能相互增强或抵消，使得电力负荷难以准确预测。虽然现有方法可以将上下文信息作为额外输入纳入模型，但如何利用多重事件的影响及其相互作用来引导负荷预测过程本身仍有待探索。为填补这一空白，我们提出了ElecLoad，一个面向多重事件电力负荷预测的上下文感知智能体，它能够自适应地组织预测过程。具体而言，我们训练ElecLoad将历史负荷与（摘要在此处截断）相结合……

    arXiv:2611.06508v1 Announce Type: new  Abstract: Accurate electricity load forecasting provides critical decision support for energy procurement, generation scheduling, and supply-demand balancing, contributing to the economic efficiency and operational reliability of power systems. However, future electricity demand can be affected by the complex interplay of multiple events described in Web notices, weather alerts, and event schedules. Their effects vary over time and can reinforce or offset one another, making electricity load difficult to forecast accurately. While existing methods can incorporate context as additional input, how to use the effects of multiple events and their interactions to guide the load forecasting process itself remains to be explored. To address this gap, we propose ElecLoad, a context-aware agent for multi-event electricity load forecasting that adaptively organizes the forecasting process. Specifically, we train ElecLoad to combine historical load with cont
    
[^48]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^49]: 工具无需以文本形式发言：面向智能体的潜在上下文压缩

    Tools Need Not Speak in Text: Latent Context Compression for Agents

    [https://arxiv.org/abs/2611.06500](https://arxiv.org/abs/2611.06500)

    提出Coca-Cola连续潜在上下文压缩模型，将工具观测压缩为嵌入智能体输入空间的潜在token，并通过动作保持训练对齐动作分布，突破了离散文本表示的限制，实现更高的上下文压缩率。

    

    大语言模型（LLM）智能体通常将原始工具输出直接附加到其上下文中。在多步交互过程中，冗长且嘈杂的观测内容会带来巨大的上下文开销，并稀释智能体对任务相关信息的注意力。现有的token剪枝和摘要方法仍受限于离散的文本表示，从而限制了可实现的压缩比率。在本工作中，我们挑战了“智能体必须以显式文本token的形式消费工具观测内容”这一普遍假设。为此，我们提出了Coca-Cola，一个面向大语言模型智能体的连续潜在上下文压缩模型。Coca-Cola将工具观测内容压缩为潜在token，并将其嵌入到智能体的输入嵌入空间中。我们进一步提出了动作保持训练，该方法在完整观测和压缩观测下对齐智能体的下一步动作分布。由此训练压缩模型学习到动作充分的潜在token，从而保留（原文在此处截断）……

    arXiv:2611.06500v1 Announce Type: new  Abstract: Large language model (LLM) agents typically append raw tool outputs to their context. Over multi-step interactions, long and noisy observations incur substantial context overhead and dilute the agent's attention to task-relevant information. Existing token pruning and summarization methods remain constrained by discrete text representations, limiting the achievable compression ratios.   In this work, we challenge the prevailing assumption that agents must consume tool observations as explicit text tokens. To this end, we propose Coca-Cola, a continuous latent context compression model for LLM agents. Coca-Cola compresses tool observations into latent tokens and embeds them into the input embedding space of the agent. We further introduce Action-Preserving Training, which aligns the agent's next-action distributions under full and compressed observations. This trains the compression model to learn action-sufficient latent tokens that reta
    
[^50]: FPTR：面向截止期受限的联合波束与资源分配的可行性保持事务式精化算法

    FPTR: Feasibility-Preserving Transactional Refinement for Deadline-Constrained Joint Beam and Resource Allocation

    [https://arxiv.org/abs/2611.06491](https://arxiv.org/abs/2611.06491)

    该论文提出FPTR算法，通过“候选解必须完整、在阶段截止前通过可行性验证且严格改进目标才能替换当前最优解”的事务式机制，确保无线调度器在截止期约束下始终输出有效可行的波束与资源分配方案。

    

    短间隔无线调度器必须在每个执行截止期之前返回完整、可行的波束与资源分配方案，而迟到或无效的输出在当前控制周期内不产生任何效用。延长搜索时间虽然可以增加传输流量，但也可能错过截止期，或仅留下一个不可行的中间解。我们提出了可行性保持事务式精化算法，它始终保留一个经过验证的可行分配方案（即当前最优解）。每个精化阶段都会构建一个独立的候选解，而不改变当前最优解。只有当候选解是完整的、在其阶段截止时间之前通过可行性验证器验证，并且严格改进目标函数时，它才会替换当前最优解；否则该候选解将被丢弃。四个精化阶段采用这一规则：基于缓冲区的资源选择、兼容组（CG）用户共享、基于剩余需求的波束掩码重建，以及两个资源的联合重新分配。在150个固定的（摘要在此处被截断）……

    arXiv:2611.06491v1 Announce Type: new  Abstract: Short-interval wireless schedulers must return a complete, feasible beam-and-resource allocation before each execution deadline, and a late or invalid output gives no utility in the current control cycle. Searching longer can increase transmitted traffic, but it may also miss the deadline or leave only an infeasible intermediate solution. We propose Feasibility-Preserving Transactional Refinement (FPTR), which always keeps a validated feasible allocation, the incumbent. Each refinement stage builds a separate candidate and leaves the incumbent unchanged. A candidate replaces the incumbent only if it is complete, passes the feasibility validator before its stage cutoff, and strictly improves the objective; otherwise it is discarded. Four refinement stages use this rule: buffer-aware resource selection, compatibility-group (CG) user sharing, beam-mask rebuilding from remaining demand, and joint reassignment of two resources. On 150 fixed s
    
[^51]: 模型合并不需要统一的LoRA秩

    Model Merging Does Not Require Uniform LoRA Ranks

    [https://arxiv.org/abs/2611.06486](https://arxiv.org/abs/2611.06486)

    该论文证明模型合并不要求各专家使用统一的LoRA秩，为不同领域专家分配各自的最佳秩可让平均法和基于效用的选择方法零成本地提升最多1.8个百分点的多领域准确率，而冲突解决类方法则表现不一致甚至可能损失4个百分点。

    

    模型合并将各自独立微调的专家模型组合为单一模型，而无需重新训练。标准做法是以相同的LoRA秩训练每个专家，这使得秩的选择在各专家之间相互耦合，尽管每个领域其实都有各自的最佳秩。我们证明这种耦合是合并方法的特性，而非合并本身的固有属性。我们在三个领域（数学、指令遵循、多语言）、四种秩和三种模型规模下训练了36个专家模型，然后用六种方法分别在统一秩和各领域自身最佳秩下进行合并。为专家分配不同秩对平均法和基于效用的选择方法没有任何代价，反而可带来最高1.8个百分点的多领域准确率提升，而冲突解决类方法表现不一致，最多可能损失4个百分点。这种差异有一个简单的原因：合并仅保留汇集谱成分中的固定比例，而按大小选择这些成分的方法会丢弃许多本可以降低损失的成分。它们还倾向于丢弃…

    arXiv:2611.06486v1 Announce Type: cross  Abstract: Model merging combines separately fine-tuned experts into a single model without retraining. Standard practice trains every expert at the same LoRA rank, which couples the rank choice across experts even though each domain has its own best rank. We show this coupling is a property of the merge method, not of merging. We train 36 experts on three domains (mathematics, instruction following, multilingual) at four ranks and three model scales, then merge with six methods at uniform ranks and at each domain's own best rank. Giving experts different ranks costs averaging and utility-based selection nothing, and gains them up to 1.8 pp of multi-domain accuracy, while the conflict-resolution methods are inconsistent and can lose up to 4 pp. The split has a simple cause. Merging keeps only a fixed share of the pooled spectral components, and methods that choose them by size discard many that would have reduced loss. They also tend to discard t
    
[^52]: ArcRAG：基于知识图谱的智能体式组合检索

    ArcRAG: Agentic Compositional Retrieval over Knowledge Graphs

    [https://arxiv.org/abs/2611.06481](https://arxiv.org/abs/2611.06481)

    ArcRAG通过在推理时归纳查询局部的图谱接口并规划可修复的多跳遍历，实现了无需任何构建特定调整即可跨不同图谱构建方式保持有效的智能体式组合检索。

    

    基于图的检索增强生成（GraphRAG）通过在从语料库构建的知识图谱上进行检索来回答问题。然而，检索策略通常围绕其自身构建流程所产生的结构来设计，这些策略在构建方式改变后是否依然有效却很少被测试。我们研究了这种跨构建的可移植性：即一个不依赖特定构建方式的模式、关系名称或重新调优的检索流程，在图谱以不同方式构建时是否仍能保持有效。我们提出了ArcRAG，它在推理时归纳出一个查询局部的图谱接口（即从问题所命名的实体出发可达的节点标签、关系描述符和属性键），据此规划多跳遍历，并以Cypher语言执行和修复这些遍历。我们将MuSiQue、HotpotQA和TriviaQA语料库分别构建为模式引导的图谱（类型化节点、封闭关系集合、来源溯源）和开放信息抽取图谱（摘要在此处被截断）

    arXiv:2611.06481v1 Announce Type: cross  Abstract: Graph-based retrieval-augmented generation (GraphRAG) answers questions by retrieving over a knowledge graph built from a corpus. Yet retrieval strategies are designed around the structure their own construction pipeline produces, and whether they survive a change of construction is rarely tested. We study this cross-construction portability: whether one retrieval procedure, with no construction-specific schema, relation names, or retuning, stays effective when the graph is built differently. We introduce ArcRAG, which induces a query-local graph interface at inference time (the node labels, relation descriptors, and property keys reachable from the entities a question names), plans multi-hop traversals against it, and executes and repairs them as Cypher. We build each of the MuSiQue, HotpotQA, and TriviaQA corpora as a schema-guided graph (typed nodes, a closed relation set, source provenance) and as an open-information-extraction gra
    
[^53]: 基于不确定性引导框架的微液滴生成器代理模型增强与逆向设计

    Microdroplet Generator Surrogate Enhancement and Inverse Design Through an Uncertainty-Guided Framework

    [https://arxiv.org/abs/2611.06464](https://arxiv.org/abs/2611.06464)

    本文提出一种不确定性引导框架，利用神经网络集成的不确定性量化定位流型转变边界等关键区域，并通过物理驱动的针对性模拟数据增强策略提升代理模型精度，从而实现微液滴生成器的高效逆向设计。

    

    微流控液滴生成对许多应用至关重要，然而由于复杂的多相流动力学，在高维参数空间中进行高效的器件设计在计算上仍然代价高昂。本工作提出了一种统一的方法论，将自适应代理模型增强与仿真驱动的逆向设计相结合。高保真度的流体体积法模拟提供真实基准数据，神经网络集成模型则预测流型状态和液滴直径。基于集成模型的不确定性量化利用一致性水平（用于流型分类）和变异系数（用于直径预测）来识别认知不确定性，发现不确定性集中在流型转变边界附近——在这些区域，物理特性对参数扰动高度敏感。一种物理驱动的数据增强策略针对这些关键区域，通过部署额外的针对性模拟并重新训练模型来应对类别不平衡问题。

    arXiv:2611.06464v1 Announce Type: cross  Abstract: Microfluidic droplet generation is critical for many applications, yet efficient device design in high-dimensional parameter spaces remains computationally prohibitive due to complex multiphase flow dynamics. This work presents a unified methodology that couples adaptive surrogate enhancement with simulation-driven inverse design. High-fidelity Volume-of-Fluid simulations provide ground-truth data, while neural network ensembles predict flow regimes and droplet diameters. Ensemble-based uncertainty quantification identifies epistemic uncertainty using consensus levels for regime classification and the coefficient of variation for diameter prediction, revealing that uncertainty concentrates near regime transition boundaries where physics is highly sensitive to parameter perturbations. A physics-informed enrichment strategy targets these critical zones and addresses class imbalances by deploying additional targeted simulations and retrai
    
[^54]: 固定分辨率迁移学习下的EfficientNet家族内缩放用于频域光学相干断层扫描视网膜疾病分类

    Within-Family EfficientNet Scaling under Fixed-Resolution Transfer Learning for Spectral-Domain Optical Coherence Tomography Retinal Disease Classification

    [https://arxiv.org/abs/2611.06452](https://arxiv.org/abs/2611.06452)

    该研究发现固定分辨率迁移学习中EfficientNet家族内按规模缩放并不能单调提升SD-OCT视网膜疾病分类性能，最小的B1以96.95%的准确率超越更大的B7模型，表明在医学图像迁移学习中盲目增大模型并非最优选择。

    

    背景/目的：卷积神经网络（CNN）在基于频域光学相干断层扫描（SD-OCT）的视网膜疾病分类中表现良好，但EfficientNet在自然图像上的缩放层级能否迁移到固定分辨率的医学图像迁移学习中仍不明确。本研究评估了EfficientNet-B0至B7在SD-OCT四分类任务中的表现以及准确率与效率之间的权衡。方法：在Kermany数据集上以224×224分辨率评估了八个ImageNet预训练变体（实例化骨干网络参数量为405万至6410万）。灰度图像被复制为RGB三通道，并使用模型内置的预处理与数据增强进行处理。训练流程包括冻结骨干网络的分类头训练，随后进行部分微调。评估指标涵盖分类性能、判别能力、效率以及可视化分析。结果：测试准确率呈非单调变化，从B0的95.93%到B1的96.95%；B7达到96.73%。B1取得了最高的准确率以及玻璃膜疣类别的最高F1分数（0.878……摘要至此被截断）。

    arXiv:2611.06452v1 Announce Type: cross  Abstract: Background/Objectives: CNNs perform well in retinal disease classification from spectral-domain optical coherence tomography (SD-OCT), but whether EfficientNet's natural-image scaling hierarchy transfers to medical-image transfer learning at fixed resolution remains unclear. This study evaluates EfficientNet-B0-B7 for four-class SD-OCT classification and accuracy-efficiency trade-offs. Methods: Eight ImageNet-pretrained variants (4.05-64.10M instantiated backbone parameters) were evaluated at 224x224 on the Kermany dataset. Grayscale images were replicated to RGB and processed using model-embedded preprocessing and augmentation. Training comprised frozen-backbone head training followed by partial fine-tuning. Metrics covered classification, discrimination, efficiency, and visual analyses. Results: Test accuracy varied non-monotonically from 95.93% (B0) to 96.95% (B1); B7 achieved 96.73%. B1 had the highest accuracy and Drusen F1 (0.878
    
[^55]: AmbiHTMLBench：基于HTML的视觉丰富文档解析基准

    AmbiHTMLBench: A Benchmark for HTML-based Visually Rich Document parsing

    [https://arxiv.org/abs/2611.06449](https://arxiv.org/abs/2611.06449)

    提出了AmbiHTMLBench基准，评估模型将真实世界文档恢复为保留内容、组织结构与视觉外观的可执行HTML的能力，以解决传统Markdown输出无法保留细粒度空间结构和渲染语义的问题。

    

    传统文档解析系统使用Markdown作为输出表示形式，这对线性文本很有效，但无法完整保留细粒度的空间结构、表格几何形状、视觉层次或渲染语义。我们证明HTML对于异构且布局敏感的文档更具优势，这类文档的表示形式需要保持可编辑、可检查和可复用，并引入AmbiHTMLBench来系统性地研究这一场景。在该基准中，给定一个真实世界的文档，模型必须恢复出能够保留其内容、组织和视觉外观的可执行HTML。该基准包含1,115个带有结构化标注的多样化页面，并通过两种不同方式评估预测结果：客观协议将真实值与预测的渲染块进行对齐，衡量表格保真度、页面级文本错误和二维布局一致性；补充的参考锚定协议则衡量感知……

    arXiv:2611.06449v1 Announce Type: new  Abstract: Traditional document parsing systems use Markdown as their output representation, which is effective for linear text but does not fully preserve fine-grained spatial structure, table geometry, visual hierarchy, or rendering semantics. We show that HTML is advantageous for heterogeneous and layout-sensitive documents whose representations must remain editable, inspectable, and reusable, and introduce AmbiHTMLBench to systematically study this setting. In this benchmark, given a real-world document, a model must recover executable HTML that preserves its content, organization, and visual appearance. The benchmark contains 1,115 diverse pages with structured annotations and evaluates predictions in two different ways: The objective protocol aligns ground-truth and predicted rendered blocks and measures table fidelity, page-level text error, and a two-dimensional layout consistency; a complementary reference-anchored protocol measures percei
    
[^56]: SUMONE：一个面向即插即用智能体记忆的简单高效摘要器

    SUMONE: A Simple and Efficient Summarizer for Plug-and-Play Agent Memory

    [https://arxiv.org/abs/2611.06445](https://arxiv.org/abs/2611.06445)

    SumOne是一个通过掩码重建进行自监督训练的独立摘要器，无需参考摘要或任务特定信号，即可为长时程智能体提供轻量级、即插即用的上下文压缩。

    

    长时程语言模型智能体会积累日益庞大的交互历史，因此上下文压缩对高效执行至关重要。现有的学习式压缩方法通常依赖下游任务结果或求解器行为，使训练与特定任务或智能体耦合在一起。我们提出了SumOne，一个通过掩码重建来学习保留智能体历史信息的独立摘要器。给定已过期历史的摘要，一个冻结的重建器恢复原始轨迹中被随机掩码的部分；重建质量结合长度惩罚，构成了一个任务无关的奖励信号，用于以GRPO优化摘要器。由此，自监督信号可直接从原始轨迹中获得，无需参考摘要或下游任务执行。在推理时，仅保留轻量级的摘要器，即可为无需改动原有智能体实现即插即用的上下文压缩。我们进行了系统……

    arXiv:2611.06445v1 Announce Type: new  Abstract: Long-horizon language-model agents accumulate increasingly large interaction histories, making context compaction essential for efficient execution. Existing learned compaction methods often rely on downstream task outcomes or solver behavior, coupling training to specific tasks or agents. We introduce SumOne, a standalone summarizer trained to preserve information in agent histories through masked reconstruction. Given a summary of the expired history, a frozen reconstructor recovers randomly masked portions of the original trajectory; reconstruction quality, combined with a length penalty, provides a task-agnostic reward for optimizing the summarizer with GRPO. This yields self-supervision directly from raw trajectories, without reference summaries or downstream task execution. At inference time, only the lightweight summarizer is retained, enabling plug-and-play context compression for otherwise unchanged agents. We conduct a systemat
    
[^57]: EviSeek：一个以证据为中心的具身主动信息获取智能体框架

    EviSeek: An Evidence-Centric Agentic Framework for Embodied Active Information Seeking

    [https://arxiv.org/abs/2611.06434](https://arxiv.org/abs/2611.06434)

    EviSeek 是一个无需训练的以证据为中心的智能体框架，通过将问题中蕴含的证据需求转化为明确的获取目标，引导具身智能体进行有目的的主动信息获取，在 ESI-Bench 上显著提升问答准确率并降低 token 消耗。

    

    主动具身问答要求智能体通过视角变化和物理交互来获取证据。其核心挑战在于避免无方向的交互：即采取看似合理的行动，却没有明确这些行动应为回答问题揭示何种证据。仅依靠行动计划和累积的历史记录，可能会使交互的信息目的变得隐晦。我们提出了 EviSeek，这是一个无需训练的框架，它将从问题中推导出的证据需求转化为明确的获取目标，将基于观察条件的动作和选择性保留的证据与智能体所寻求的信息联系起来。在 ESI-Bench 的 3,436 个问题上，EviSeek 使用 Qwen3-VL-32B 将准确率相比单模型智能体提升了 7.1 个百分点，使用 Gemini 3.1 Pro 提升了 3.7 个百分点，同时提高了视觉支持加权准确率并降低了平均 token 使用量。这些发现支持了将交互的信息目的显式化的做法。

    arXiv:2611.06434v1 Announce Type: new  Abstract: Active embodied question answering requires agents to acquire evidence through viewpoint changes and physical interaction. The central challenge is to avoid undirected interaction: taking plausible actions without specifying what evidence they should reveal for answering the question. Action plans and accumulated histories alone can leave the informational purpose of an interaction implicit. We introduce EviSeek, a training-free framework that makes question-derived evidence demands explicit acquisition objectives, linking observation-conditioned actions and selectively retained evidence to the information the agent seeks. On 3,436 ESI-Bench questions, EviSeek improves accuracy over the Single-model agent by 7.1 percentage points with Qwen3-VL-32B and 3.7 points with Gemini 3.1 Pro, while improving visual-support-weighted accuracy and reducing mean token usage. These findings support making the informational purpose of interaction explic
    
[^58]: D3D：面向异步投机解码的解耦扩散草稿生成

    D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding

    [https://arxiv.org/abs/2611.06425](https://arxiv.org/abs/2611.06425)

    D3D 提出一种不依赖目标模型隐藏状态的独立扩散草稿模型，使草稿生成与验证可异步并行执行，在多个基准上平均实现约 5 倍的解码加速。

    

    目标条件扩散草稿模型可以并行预测候选 token，但每个新块都必须等待目标模型更新后的隐藏状态。这迫使草稿生成与验证必须串行运行，即使每个块都是一次性生成，速度提升仍然受限。我们提出 D3D，一种独立的扩散草稿模型，消除了这种轮次间的依赖。D3D 将分块去噪与辅助自回归监督相结合，在不依赖目标模型隐藏状态的情况下实现较长的被接受前缀。这种独立性使 D3D 能够在验证过程中通过一次打包前向传递草拟后续块，从而实现异步投机解码。我们根据各层对接受长度的影响逐步剪枝，以降低打包草稿生成的成本。在以 Qwen3-8B 为目标模型的三个基准测试中，移除 36 层中的 5 层后，平均接受长度仍保持在 99% 以上。在七个基准测试中，D3D 平均实现了 5.00 倍和 5.71 倍的解码加速（原文在此处截断）。

    arXiv:2611.06425v1 Announce Type: new  Abstract: Target-conditioned diffusion drafters predict candidate tokens in parallel, yet each new block must wait for updated target-model hidden states. This forces drafting and verification to run sequentially, limiting speedup even though each block is generated in one pass. We introduce D3D, a standalone diffusion drafter that removes this inter-round dependency. D3D combines blockwise denoising with auxiliary autoregressive supervision to achieve long accepted prefixes without target hidden states. This independence lets D3D draft successor blocks during verification in one packed pass, enabling asynchronous speculative decoding. We progressively prune layers according to their effect on acceptance length to reduce the cost of packed drafting. Removing five of 36 layers retains over 99\% of mean acceptance length across three benchmarks with Qwen3-8B as the target. Across seven benchmarks, D3D averages 5.00$\times$ and 5.71$\times$ decoding 
    
[^59]: 通过闭环刺激实现大语言模型的“意念控制”：分布外鲁棒性

    Mind Control of LLMs via Closed-Loop Stimulation: Out-of-Distribution Robustness

    [https://arxiv.org/abs/2611.06418](https://arxiv.org/abs/2611.06418)

    受神经刺激领域闭环控制原理启发，本文将大语言模型引导表述为反馈控制问题，通过模型内部状态动态调节干预，从而在分布外场景下实现鲁棒的模型行为控制。

    

    大脑和大语言模型不必被视为等价系统，也能揭示出一个可解释的共性：两者都表现出高维、非线性的动力学特性，其行为只能被不完美地建模。这在我们试图干预其动力学时带来了共同的挑战。控制器要么依赖于在特定工作点附近辨识出的模型，并承担系统偏离该工作点时失效的风险；要么显式地考虑不确定性和不断变化的内部状态。神经刺激领域几十年来一直面临类似的问题：固定开环刺激的局限性推动了闭环和状态依赖方法的发展，这些方法根据神经系统的实时状态来调节干预。受这一原则启发，我们将大语言模型引导表述为一个反馈控制问题，其中网络的内部状态——而非单一的固定干预——决定系统如何被扰动。

    arXiv:2611.06418v1 Announce Type: cross  Abstract: Brains and large language models need not be considered equivalent systems to reveal an interpretable commonality: both exhibit high-dimensional, nonlinear dynamics whose behavior can only be modeled imperfectly. This creates a shared challenge whenever we seek to intervene in their dynamics. A controller may either rely on a model identified around a particular operating regime and risk failure as the system moves away from that regime, or explicitly account for uncertainty and a changing internal state. Neurostimulation has confronted an analogous problem for decades. The limitations of fixed, open-loop stimulation have motivated closed-loop and state-dependent approaches that condition intervention on the ongoing state of the neural system. Inspired by this principle, we formulate LLM steering as a feedback-control problem in which the network's internal state, rather than a fixed intervention alone, determines how the system is per
    
[^60]: 《防御基于深度学习的人脸识别流水线免受后门攻击的一揽子技巧》

    A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks

    [https://arxiv.org/abs/2611.06412](https://arxiv.org/abs/2611.06412)

    本文针对训练数据可能被第三方污染的人脸识别系统，提出了三种实用的后门攻击防御技巧——一种在训练阶段检测被投毒的身份，两种在推理阶段过滤可疑输入，为系统提供商提供了可落地的安全加固方案。

    

    本技术报告针对与行业从业者最相关的威胁模型：尽管模型训练在内部完成，但训练数据来源于可能已被入侵的第三方。在此背景下，我们提出了三种针对基于数据投毒的后门攻击的实用防御“技巧”。第一种在训练阶段运行，可在训练期间或流水线部署之前检测被投毒的身份。后两种在推理阶段运行，在将结果返回给潜在攻击者之前过滤可疑输入。每种方法都介绍了其动机、算法和计算开销，并在真实的人脸识别工作负载上进行了评估。我们的目标是为系统提供商提供可实际操作的加固技术。

    arXiv:2611.06412v1 Announce Type: cross  Abstract: The protection of mature, multi-stage Face Recognition Systems is crucial to system providers. This technical report targets the threat model most relevant to industry practitioners: although model training is performed in-house, training data is sourced from possibly-compromised third parties. Given this setting, we present three practical defense "tricks" against data poisoning-based backdoor attacks. The first operates at training-time, detecting poisoned identities during training or prior to a pipeline's deployment. The last two operate at inference-time, filtering suspicious inputs before an answer can be sent back to a potential attacker. Each method is presented with their motivation, algorithms and computational impact, and evaluated on realistic face recognition workloads. Our goal is to provide actionable hardening techniques to system providers.
    
[^61]: 大语言模型智能体之间的零样本协调

    Zero-Shot Coordination among LLM Agents

    [https://arxiv.org/abs/2611.06409](https://arxiv.org/abs/2611.06409)

    该论文通过设计隔离特定协调挑战的最小化环境与简单通用脚手架，首次系统研究了独立开发的LLM智能体之间的零样本协调问题，发现即使最前沿的LLM智能体也难以协调，其主要原因在于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。

    

    我们研究零样本协调，即独立开发的智能体必须在测试时进行协调的场景。尽管零样本协调在强化学习文献中已被充分研究，但对于大语言模型智能体在此类设置中的性能却知之甚少，尽管它们在此类环境中的部署日益增多。现有的LLM协调工作通常依赖于专门化的脚手架架构，而独立开发的智能体在实践中不太可能共享这些架构。此外，在复杂环境（如Hanabi）上进行评估使得难以准确定位协调失败的根源。相比之下，我们专注于在最小化环境中使用简单、通用的脚手架架构，这些环境旨在隔离特定的协调挑战。我们的结果表明，即使在这些受控设置中，前沿LLM智能体也难以协调，这主要是由于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。虽然LLM引入了语义信息作为额外的（摘要在此处截断）……

    arXiv:2611.06409v1 Announce Type: cross  Abstract: We study zero-shot coordination (ZSC), where independently developed agents must coordinate at test time. While ZSC has been well studied in the RL literature, far less is known about the performance of LLM agents despite their increasing deployment in such settings. Existing work on LLM coordination often relies on specialised scaffolding that independently developed agents are unlikely to share in practice. Moreover, evaluations on complex environments (e.g., Hanabi) make it difficult to pinpoint the sources of coordination failure. In contrast, we focus on simple, general-purpose scaffolds in minimal environments designed to isolate specific coordination challenges. Our results show that even in these controlled settings, frontier LLM agents struggle to coordinate, largely due to limited understanding of the coordination problem and weak reasoning about their partner's beliefs. While LLMs introduce semantic information as an additio
    
[^62]: GIST：面向离线RLVR的基于吉布斯目标的层级数据选择

    GIST: Gibbs-Targeted Hierarchical Data Selection for Offline RLVR

    [https://arxiv.org/abs/2611.06395](https://arxiv.org/abs/2611.06395)

    该论文提出GIST方法，基于双向KL散度在参考策略处的精确分解，揭示出正确响应应聚焦其低概率token、错误响应应聚焦策略高频生成响应的高概率token这两条监督原则，并据此设计两级层级数据选择方法来优化离线RLVR的训练监督。

    

    具有可验证奖励的离线强化学习在固定的采样响应池上进行训练，通常对每个响应的每个token都进行监督，然而响应和token的信息量并不相同。我们通过KL正则化RLVR的Gibbs最优策略（即离线训练的目标）来研究应当监督哪些内容，并用双向KL散度来度量策略与该目标之间的失配。在参考策略处，这种失配可以精确分解，并将两种结果区分开来：正确响应的贡献主要来自前向项，集中于其低概率token上；错误响应的贡献则来自反向项，集中于策略经常生成的响应及其高概率token上。基于这一分析，我们提出了GIST（Gibbs-Targeted Hierarchical Data Selection，吉布斯目标导向的层级数据选择），这是一种两级选择方法，利用可获得的信号来近似上述原则……（摘要原文在此处截断）

    arXiv:2611.06395v1 Announce Type: new  Abstract: Offline reinforcement learning with verifiable rewards (RLVR) trains on a fixed pool of sampled responses and typically supervises every token of every response, although responses and tokens are not equally informative. We study which ones to supervise through the Gibbs-optimal policy of KL-regularized RLVR, which offline training targets, and measure the mismatch between a policy and this target with a bidirectional KL divergence. At the reference policy, this mismatch factorizes exactly and separates the two outcomes: correct responses contribute mainly through the forward term, concentrated on their low-probability tokens, and incorrect responses through the reverse term, concentrated on the responses the policy generates often and on their high-probability tokens. Based on this analysis, we propose GIST (Gibbs-Targeted Hierarchical Data Selection), a two-level selection method that approximates these principles with signals availabl
    
[^63]: MA-Mesh：基于角色自愈的弹性多智能体协作

    MA-Mesh: Resilient Multi-Agent Collaboration via Role-based Self-Healing

    [https://arxiv.org/abs/2611.06389](https://arxiv.org/abs/2611.06389)

    提出了MA-Mesh框架，通过跨智能体反思和基于角色的故障转移两种自愈机制，在协作模式之上构建基于角色的自愈层，实现多智能体系统运行时的系统性弹性管理。

    

    多智能体系统中的有效协作需要具备在动态和不确定环境中进行协调与适应的机制。然而，现有的多智能体系统很少显式地对弹性进行建模，而是依赖临时性的输出修正，这限制了对鲁棒性进行系统性的运行时管理。在本工作中，我们提出了多智能体网格，这是一个在协作模式之上构建基于角色自愈层的框架。具体而言，MA-Mesh 生成具备弹性感知的系统规范，并将其合成为可执行的多智能体策略代码。为了增强弹性，我们引入了两种自愈机制：跨智能体反思，该机制允许智能体不仅评估自身的输出结果，还能评估同伴的表现，从而实现有依据的行为调整和集体策略改进；基于角色的故障转移，一种动态重新分配角色的策略，从而实现……（原文此处截断）

    arXiv:2611.06389v1 Announce Type: new  Abstract: Effective collaboration in multi-agent systems requires mechanisms for coordination and adaptability in dynamic and uncertain environments. However, existing multi-agent systems rarely model resilience explicitly, instead relying on ad-hoc output correction that limits systematic, runtime management of robustness. In this work, we propose Multi-Agent Mesh (MA-Mesh), a framework that constructs a role-based self-healing layer on top of a collaboration pattern. Specifically, MA-Mesh generates resilience-aware system specifications and synthesizes them into executable multi-agent policy code. To enhance resilience, we introduce two self-healing mechanisms: Cross-agent reflection, a mechanism that allows agents to evaluate not only their own outcomes but also the performance of their peers, enabling informed behavioral adjustments and collective policy refinement; Role-based failover, a strategy which dynamically reassigns roles and enables 
    
[^64]: 从水文定律到流域模型：一个基于过程的模型在三大洲上达到与深度学习相当的精度

    From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents

    [https://arxiv.org/abs/2611.06382](https://arxiv.org/abs/2611.06382)

    该研究基于守恒、耗散和聚合三条水文定律构建简约流域模型CaRT，并通过AI代理辅助的假设驱动迭代细化得到CaRT+模型，在不使用日尺度神经网络的情况下于三大洲达到与深度学习相当的精度，证明物理定律约束的低维模型可以匹敌神经网络。

    

    Dooge曾提出水文在流域尺度上是否存在定律的问题。我们则要问：一个在物理定律约束下构建的低维系统，能否达到神经网络在相同输入下超越基于过程的模型的精度。三条定律约束着一个集总式流域模型：守恒定律使其成为一个由通量连接的储量网络；耗散定律确定了每个通量的符号；聚合定律决定了未解析的组成部分如何通过阈值分布上的有效定律和对强迫的记忆进入通量。这些定律允许多种细化方式。在此框架下的一个简约模型CaRT（每个流域11个参数），在AI代理辅助的循环中被逐步细化。每次细化都通过文献中的一个假设来解决观察到的模型缺陷。最终得到的CaRT+模型具有十四个状态变量、每个流域35个参数，且不包含任何日尺度神经网络。大多数细化源于对阈值的聚合或来自新的状态变量。

    arXiv:2611.06382v1 Announce Type: cross  Abstract: Dooge asked whether hydrology has laws at the catchment scale. We ask whether a low-dimensional system built under physical laws can match the accuracy of neural networks that outperform process-based models on the same inputs. Three laws constrain a lumped catchment model. Conservation makes it a flux-connected storage network, dissipation fixes the sign of every flux and aggregation determines how the unresolved composition enters the fluxes, through effective laws over threshold distributions and a memory of the forcing. These laws admit four kinds of refinement. A parsimonious model under them, CaRT (11 parameters per catchment), is refined in a loop assisted by AI agents. Each refinement addresses an observed deficit through a hypothesis from the literature. The result, CaRT+, has fourteen states, 35 parameters per catchment and no daily neural network. Most refinements follow from aggregation over thresholds or from variables tha
    
[^65]: LegalEvolver：基于评分准则引导自我进化的智能体法律判决预测

    LegalEvolver: Agentic Legal Judgment Prediction via Rubric-Guided Self-Evolution

    [https://arxiv.org/abs/2611.06381](https://arxiv.org/abs/2611.06381)

    提出 LegalEvolver 框架，通过评分准则引导的自我进化机制将查询启发式优化为种子，为法律判决预测生成更具法律信息量的检索查询，从而突破现有法律搜索智能体受限于查询质量的性能瓶颈。

    

    随着基于大语言模型（LLM）的智能体不断发展，它们已被越来越广泛地应用于法律领域。法律判决预测（LJP）要求模型进行多步推理并检索外部法律证据。现有法律搜索智能体的性能往往受限于检索查询的质量：它们通常直接根据用户的初始请求构建查询，而忽略了潜在的法律问题和具有法律重要性的细节。为解决这一问题，我们提出了 LegalEvolver，一个专为 LJP 导向检索设计的查询优化框架，用于生成具有法律信息量的查询。采样高质量的优化查询及其对应的搜索轨迹极具挑战性，因为仅依赖无引导的高温采样（rollouts）既低效，又受限于模型自身初始能力。因此，我们采用启发式方法将查询优化为种子，从而提升轨迹生成的效率……（摘要原文在此处被截断）

    arXiv:2611.06381v1 Announce Type: new  Abstract: As LLM-based agents develop, they have been increasingly applied in the legal domain. Legal Judgment Prediction~(LJP) requires models to conduct multi-step reasoning and external legal evidence retrieval. Existing legal search agents' performance is often constrained by the quality of retrieval queries. They often formulate queries directly from the initial user request, overlooking latent legal issues and legally salient details. To address this issue, we propose \emph{LegalEvolver}, a query optimization framework specifically designed to generate legally informative queries for LJP-oriented retrieval. Sampling high-quality optimized queries and their corresponding search trajectories is challenging, as relying solely on unguided high-temperature rollouts is both inefficient and constrained by the model's initial capabilities. We therefore heuristically optimize queries as seeds, improving the efficiency of trajectory generation. Furthe
    
[^66]: 上下文压缩能否保持拒绝对齐？

    Does Context Compression Preserve Refusal Alignment?

    [https://arxiv.org/abs/2611.06380](https://arxiv.org/abs/2611.06380)

    论文发现基于编码器的上下文压缩虽然保留了语义内容，却会系统性削弱语言模型的拒绝对齐行为，并提出了一种无需重训练、在压缩表示空间中运行的轻量级推理时干预方法 Memory Steering，可将拒绝率恢复至接近基线水平。

    

    上下文压缩通过将输入编码为紧凑的表示来降低推理成本，同时保留语义内容。一个悬而未决的问题是，仅凭语义保留是否足以维持诸如拒绝对齐之类的下游行为。我们对这一问题进行了研究，发现尽管重建保真度很高，基于编码器的压缩仍会系统性地削弱指令微调语言模型中的拒绝行为。这一效应在不同的模型家族和压缩架构中均持续存在。机制分析表明，压缩会沿着解码器学到的拒绝方向削弱激活。我们进一步提出了 Memory Steering，这是一种轻量级的推理时干预方法，无需重新训练即可将拒绝率恢复到接近基线的水平，并且完全在压缩表示空间中运行。这些结果表明，语义保留并不能保证压缩下的行为保留。

    arXiv:2611.06380v1 Announce Type: new  Abstract: Context compression reduces inference cost by encoding inputs into compact representations while preserving semantic content. An open question is whether semantic preservation alone is sufficient to maintain downstream behaviours such as refusal alignment. We investigate this question and find that encoder-based compression systematically weakens refusal behaviour in instruction-tuned language models, despite high reconstruction fidelity. This effect persists across model families and compression architectures. Mechanistic analysis shows that compression attenuates activation along the decoder's learned refusal direction. We further explore Memory Steering, a lightweight inference-time intervention that restores refusal rates to near-baseline levels without retraining and operates entirely in compressed representation space. These results demonstrate that semantic preservation does not guarantee behavioural preservation under compression
    
[^67]: 以草根契约圈控人工智能风险

    Corralling AI Risks with Grassroots Contracts

    [https://arxiv.org/abs/2611.06374](https://arxiv.org/abs/2611.06374)

    提出以“草根契约”这一数字化契约机制，来圈控AI操控全球平台、匿名AI智能体逃避追责以及无法归因的深度伪造内容这三大AI风险。

    

    人工智能的转折点带来了大量令人担忧的风险。数字平台尤其容易受到三大AI风险的影响，因为全球平台的控制者正在将控制权让渡给AI——包括算法控制权、成员控制权和内容控制权：(1) AI控制被人类大多数人所使用的全球社会与经济平台，导致大多数人类容易受到AI的操控。(2) 无法问责的AI智能体在社交与经济数字平台上与人类并肩匿名运行，可能伪装成人类，为身份不明的主人服务，追求未知的目标，在未知的约束下运行，缺乏问责机制和责任追究。(3) 由匿名AI生成、无法归因的深度伪造内容，人们无法将其与真实内容区分开来，其制作者则逃避问责和追责。在此，我们提出用草根契约来圈控这些AI风险，草根契约是一种……（原文摘要在此处截断）

    arXiv:2611.06374v1 Announce Type: new  Abstract: AI's inflection point is a fountain of alarming risks. Digital platforms in particular are vulnerable to three major AI risks, as those in control of global platforms cede control to AI, of the algorithms, of membership and of content: (1) AI control of global platforms, social and economic, used by the majority of humanity, resulting in the majority of humanity becoming vulnerable to AI control. (2) Unaccountable AI agents operating anonymously on social and economic digital platforms, side-by-side with humans, possibly masquerading as humans, serving unknown masters with unknown goals, operating under unknown restraints without accountability or culpability. (3) Unattributed deepfakes generated by anonymous AI, with people unable to distinguish them from genuine content and with their producers evading accountability and culpability.   Here, we propose to corral these AI risks with grassroots contracts, a digital incarnation of a two-c
    
[^68]: Traffic-VL：用于可泛化端到端交通信号控制的强化视觉语言模型

    Traffic-VL: Reinforced Vision-Language Models for Generalizable End-to-End Traffic Signal Control

    [https://arxiv.org/abs/2611.06353](https://arxiv.org/abs/2611.06353)

    该论文提出了SynTra-V逼真仿真数据集和Traffic-VL视觉语言模型，后者可直接从多视角监控图像与自然语言相位描述生成交通信号控制决策，无需中间结构化状态，实现了更可泛化的端到端交通信号控制。

    

    现有的交通信号控制方法通常依赖于由专用感知流水线产生的结构化交通状态。而新兴的视觉方法往往使用理想化的仿真器图像，并在控制之前将其转换为中间表示，这限制了它们直接利用路边视觉观测的能力。为了同时弥合视觉域差距和从感知到决策的差距，我们提出了SynTra-V和Traffic-VL。SynTra-V将基于物理的交通仿真转换为逼真的多视角监控图像，涵盖多样的道路拓扑结构和感知条件，并通过自动语义审计来检查控制关键的几何结构、车辆运动和车道语义是否与仿真器状态相匹配。Traffic-VL是一个80亿参数的视觉语言模型，它能够直接将同步的多视角图像和自然语言相位描述映射为信号决策，而无需中间结构化状态。

    arXiv:2611.06353v1 Announce Type: cross  Abstract: Existing traffic signal control methods typically rely on structured traffic states produced by dedicated perception pipelines. Emerging visual approaches often use idealized simulator images and convert them into intermediate representations before control, limiting their ability to directly exploit roadside visual observations. To bridge both the visual-domain and perception-to-decision gaps, we introduce SynTra-V and Traffic-VL. SynTra-V converts physics-based traffic simulations into photorealistic multi-view surveillance imagery across diverse road topologies and sensing conditions, and automatic semantic auditing checks that control-critical geometry, vehicle movements, and lane semantics match the simulator state. Traffic-VL is an eight-billion-parameter vision-language model that directly maps synchronized multi-view images and natural-language phase descriptions to signal decisions without intermediate structured states. Build
    
[^69]: 今日之工作，明日之劳动力：激励AI辅助工作中的学习

    Today's Work, Tomorrow's Workforce: Incentivizing Learning in AI-Assisted Work

    [https://arxiv.org/abs/2611.06346](https://arxiv.org/abs/2611.06346)

    该研究表明，依赖AI辅助会侵蚀技能积累——新手可能陷入低技能陷阱、专家可能技能退化，而重视未来技能的策略性工作者即使AI更准确也会选择独立解题以维持练习。

    

    在工作中，人们通过经验积累技能：新手通过完成任务习得技能，而专家则通过持续练习保持技能。AI辅助改变了这一过程，为工作者提供了独立解决任务或接受AI答案的选择。虽然依赖AI能节省眼前的精力，但它也可能取代获得并保持技能所需的练习，其中包括监督AI所需的技能。我们将现有的“接受或解决”模型扩展到一个任务序列中，其中工作者的技能通过独立练习和对AI的依赖而不断演变。我们比较了最大化即时效用的短视工作者与同时重视未来技能的策略性工作者。我们证明，即使AI辅助下的结果准确率依然很高，新手也可能被困在低技能水平，而专家可能丧失技能。相比之下，策略性工作者即使在AI更准确的情况下也可能选择独立解决问题……（原文摘要在此处截断）

    arXiv:2611.06346v1 Announce Type: cross  Abstract: At work, people develop skills through experience: novices acquire skills by completing tasks, while experts retain theirs through continued practice. AI assistance changes this process by offering workers a choice between solving tasks independently and accepting the AI's answers. While relying on AI saves immediate effort, it can also displace the practice needed to acquire and retain skills, including those required to oversee the AI. We extend an existing accept-or-solve model to a sequence of tasks in which workers' skills evolve through independent practice and reliance on AI. We compare myopic workers, who maximize immediate utility, with strategic workers, who also value their future skills. We show that novices can become trapped at low skill levels, while experts may lose skill, even as AI-assisted outcome accuracy remains high. Strategic workers, by contrast, may choose to solve independently even when AI is more accurate, r
    
[^70]: Prism：面向统一多模态模型的双向自教学方法

    Prism: Bidirectional Self-Teaching for Unified Multimodal Models

    [https://arxiv.org/abs/2611.06343](https://arxiv.org/abs/2611.06343)

    Prism 提出一种双向自教学后训练方法，将模型自生成的交错轨迹解耦为文本链与图像链并依据其可靠性构建训练目标，从而同时提升统一多模态模型的理解与生成能力。

    

    统一多模态模型旨在在单一模型内同时支持视觉理解与视觉生成。现有的后训练方法要么将理解能力作为固定的教师来改进生成，要么依赖真实轨迹来同时监督理解与生成。Prism 则另辟蹊径，从自生成的交错推理轨迹中推导出双向监督，并根据文本链与图像链的可靠性来构建训练目标。我们提出了 Prism，一种面向统一多模态模型的双向自教学后训练方法。如同棱镜分解混合光一样，Prism 将耦合的交错轨迹解耦为文本链和图像链，评估每条链的可靠性，并根据其可靠性状态构建相应的训练目标。在 SenseNova-U1 和 BAGEL 两个模型上的实验表明，Prism 同时提升了理解与生成能力。在生成方面，GenEval 从 85.8 提升至……（原文在此处截断）

    arXiv:2611.06343v1 Announce Type: cross  Abstract: Unified Multimodal Models (UMMs) aim to support visual understanding and generation within a single model. Existing post-training methods either use understanding as a fixed teacher to improve generation, or rely on ground-truth trajectories to supervise both understanding and generation. Prism instead derives bidirectional supervision from self-generated interleaved rollouts, constructing training targets according to the reliability of their text and image chains. We propose Prism, a bidirectional self-teaching post-training method for UMMs. Like a prism decomposing mixed light, Prism decouples a coupled interleaved trajectory into a text chain and an image chain, assesses the reliability of each chain, and constructs a corresponding training target from their reliability states. Experiments on both SenseNova-U1 and BAGEL show that Prism improves both understanding and generation. On the generation side, GenEval improves from 85.8 to
    
[^71]: 学习在有偏梯度下调参：节气门上无模型控制的迭代反馈整定

    Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves

    [https://arxiv.org/abs/2611.06338](https://arxiv.org/abs/2611.06338)

    该论文研究了迭代反馈整定（IFT）在汽车电子节气门这类非线性对象上调参时的梯度偏差问题，发现静摩擦导致的两次实验间控制器状态不匹配会引入与第二次实验幅值成反比的误差，且IFT梯度仅在69%的低成本区域内为下降方向。

    

    迭代反馈整定（IFT）通过随机梯度下降学习固定结构控制器的参数。在每次迭代中，需在被控对象上进行两次闭环实验：第一次实验测量跟踪误差，第二次实验由该误差驱动，提供用于估计代价函数梯度的灵敏度信息。该估计仅对线性对象是无偏的。我们在汽车电子节气门上研究了有偏的情形——节气门的摩擦与静摩擦在不同阀门和不同运行之间各不相同，由智能比例（无模型）控制器控制，该控制器是一个增量式PI控制器。我们证明，两次实验开始时控制器状态的不匹配（静摩擦使其不可避免）会引入一个与第二次实验幅值成反比的误差。在十个阀门的数字孪生体上，IFT梯度仅在69%的低成本区域内是下降方向，摩擦……

    arXiv:2611.06338v1 Announce Type: cross  Abstract: Iterative feedback tuning (IFT) learns the parameters of a fixed-structure controller by stochastic gradient descent. At each iteration, two closed-loop experiments are run on the plant: the first one measures the tracking error, and the second one, driven by this error, provides the sensitivities from which the gradient of the cost is estimated. This estimate is unbiased for linear plants only. We study the biased case on automotive electronic throttle valves, whose friction and stiction vary between valves and runs, controlled by an intelligent proportional (model-free) controller, which is an incremental PI controller. We show that the mismatch of the controller state at the start of the two experiments, which stiction makes unavoidable, adds an error inversely proportional to the amplitude of the second experiment. On digital twins of ten valves, the IFT gradient is a descent direction in only 69\% of the low-cost region, friction 
    
[^72]: 大规模生产具有惊人副作用的训练数据

    Mass Producing Training Data with Surprising Side Effects

    [https://arxiv.org/abs/2611.06324](https://arxiv.org/abs/2611.06324)

    本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。

    

    训练数据可能在训练领域之外对语言模型行为产生意想不到的副作用：不安全的代码数据可能导致模型广泛失调，过时的鸟类名称可能诱发模型表现出19世纪的人格。理解这种泛化现象之所以困难，是因为目前只存在几十个孤立的例子，且每个例子都是手工构建或偶然发现的。我们提出自动发现这类“具有惊人副作用的数据集”（DSSEs）。我们为DSSEs定义了三个目标：（1）能够引发目标行为，（2）读起来像普通的训练后数据，（3）让阅读这些数据的人类审计者感到惊讶。我们使用LLM评判器对每个目标进行量化，并通过进化算法、强化学习和潜意识学习方法搜索高分数据集。经过重新评分和使用保留指标过滤后，我们的流程生成了一系列包含6,575个DSSEs的数据集，涵盖8种行为，其中许多数据集的效果能够跨模型家族泛化。

    arXiv:2611.06324v1 Announce Type: cross  Abstract: Training data can have unexpected side effects on language model behavior outside the training domain: insecure code data can induce broad misalignment, and outdated bird names can induce a 19th-century persona. Understanding such generalization is hard because only dozens of isolated examples exist, each constructed by hand or discovered by accident. We propose automatically discovering such datasets with surprising side effects (DSSEs). We define three objectives for DSSEs: they should (1) elicit a target behavior, (2) read like ordinary post-training data, and (3) be surprising to human auditors reading the data. We operationalize each with LLM judges, and search for high-scoring datasets via evolution, reinforcement learning, and subliminal learning approaches. After re-scoring and filtering with held-out metrics, our pipeline produces a collection of 6,575 DSSEs across 8 behaviors, many of which generalize across model families an
    
[^73]: EnzymeRSI：一种用于酶设计的双循环递归自我改进系统

    EnzymeRSI: A dual-loop recursive self-improvement system for enzyme design

    [https://arxiv.org/abs/2611.06321](https://arxiv.org/abs/2611.06321)

    EnzymeRSI提出了一种用于酶设计的双循环递归自我改进系统，通过外循环演进预测策略、内循环开发算法以及云实验室闭合物理反馈回路，在ProteinGym基准测试上显著提升了蛋白质适应度预测性能。

    

    酶工程的改进通常是通过在固定的模型和工作流程中优化待测候选物来实现；当瓶颈源于方法不匹配或工作流程失效时，这一途径难以突破。我们提出了EnzymeRSI，一个用于递归自我改进（RSI）的双循环系统。外循环在不改变模型权重的情况下演进工具资产，内循环在受控条件下开发预测和选择算法，并规定了云实验室系统以通过质量受控的反馈来闭合物理循环。通过迭代比较更新其预测策略，外循环在五个检测任务上将ProteinGym上开发集的平均Spearman相关性提高了0.044，在CD19上达到0.335，在OPSD（视紫红质）上达到0.683；在两个第一名的成绩中，只有CD19是由外循环自身产生的。在ThermoMPNN算法的自我改进中，内循环

    arXiv:2611.06321v1 Announce Type: new  Abstract: Improvement in enzyme engineering is usually pursued by optimising which candidates to assay within a fixed model and workflow; when the bottleneck arises from a mismatched method or a failing workflow, that route is difficult to escape. We present EnzymeRSI, a dual-loop system for recursive self-improvement (RSI). The outer loop evolves harness assets without changing model weights, the inner loop develops prediction and selection algorithms under controlled conditions, and a cloud laboratory system is specified to close the physical loop with quality-controlled feedback. By updating its prediction strategy through iterative comparison, the outer loop raised the mean development-set Spearman correlation on ProteinGym by 0.044 across five assays, reaching 0.335 on CD19 and 0.683 on OPSD (rhodopsin); of the two first places, only CD19 was produced by the outer loop itself. In the self-improvement of the ThermoMPNN algorithm the inner loop
    
[^74]: 分布到分布回归的共形预测

    Conformal Prediction for Distribution-to-Distribution Regression

    [https://arxiv.org/abs/2611.06320](https://arxiv.org/abs/2611.06320)

    本文针对分布到分布回归问题，提出了一种在正交基空间中运行的共形预测新框架，能够生成可解释的预测集，同时保持覆盖保证，克服了朴素应用分裂共形预测导致的预测集不可解释的缺陷。

    

    共形预测（CP）提供了一个模型无关的框架，用于构建具有覆盖保证的有限样本预测集，并已成为高风险环境中可靠决策的工具。最近的进展已将分裂共形预测（Split Conformal Prediction）扩展到日益复杂的学习问题，包括多标签、高维和函数输出。在这项工作中，我们研究了分布到分布回归的共形预测，这是许多应用领域中日益受到关注的问题设置。我们表明，直接朴素地应用分裂共形预测所产生的预测集虽然具有边际有效性，但不可解释，例如缺乏明确定义的体积或可操作的采样程序，实际价值有限。为了克服这一问题，我们提出了一个新的共形框架，该框架在正交基空间中运行，在保持覆盖保证的同时产生可解释的预测集。我们进一步开发了一种自适应变体（摘要在此处截断）。

    arXiv:2611.06320v1 Announce Type: cross  Abstract: Conformal Prediction (CP) provides a model-agnostic framework for constructing finite-sample prediction sets with coverage guarantees and has become a tool for reliable decision-making in high-risk settings. Recent advances have extended Split Conformal Prediction to increasingly complex learning problems, including multilabel, high-dimensional, and functional outputs. In this work, we study CP for Distribution-to-Distribution Regression, a setting of growing interest across many application domains. We show that a naive application of Split CP yields prediction sets that, while marginally valid, are non-interpretable, e.g., lacking a well-defined volume or a tractable sampling procedure, and of limited practical value. To overcome this, we propose a new conformal framework that operates in an orthogonal basis space, producing interpretable prediction sets while preserving coverage guarantees. We further develop an adaptive variant wit
    
[^75]: 风险受控的吞吐量预测何时才真正控制住风险？——一项面向低轨卫星链路的复现研究

    When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Link

    [https://arxiv.org/abs/2611.06307](https://arxiv.org/abs/2611.06307)

    对低轨卫星链路上BG-CFQS风险受控吞吐量预测的复现研究发现，其风险保证虽然可复现但依赖校准集划分方式——在实际部署所面临的连续时间划分下大多失效，且无论何种划分都无法实现低于0.15的风险预算。

    

    风险预算化的吞吐量预测将低轨（LEO）卫星容量的点预测转化为一个保守上界，使其高估率保持在用户指定的预算水平之内，从而让准入控制能够安全地据此决策。我们在其作者所用的三条 Starlink 链路数据以及一个独立的第四个数据集上复现了当前最先进的方法 BG-CFQS，并报告了四项发现。第一，其风险保证是可复现的，但取决于校准集的划分方式：在随机划分下我们复现了所有已发表的数值，包括在全部3个数据集上均通过风险测试；然而在连续时间划分下——这正是实际部署终端所面对的情形——风险保证仅在3个数据集中的1个上成立。第二，无论采用何种划分方式，该方法都无法实现低于0.15的风险预算（即其候选分位数集合的下端）。在全部四个数据集上，预算设为0.05、0.10和0.15时实际达到的风险率完全相同，在最紧预算下超出目标1.4倍至3.7倍。第三，静态共形校准……（原文在此截断）

    arXiv:2611.06307v1 Announce Type: cross  Abstract: Risk-budgeted throughput forecasting converts a point forecast of LEO satellite capacity into a conservative bound whose overestimation rate is held at a user-specified budget, so that admission control can act on it safely. We reproduce the state of the art, BG-CFQS, on the three Starlink traces its authors used and on an independent fourth dataset, and report four findings. First, its risk guarantee is reproducible but conditional on the calibration split: under a random split we recover every published quantity, including risk-pass on 3 of 3 datasets, but under a contiguous temporal split, which is what a deployed terminal faces, it holds on 1 of 3. Second, independent of any split, the method cannot serve a risk budget below 0.15, the low end of its candidate quantile set. On all four datasets the achieved rate is identical at budgets of 0.05, 0.10 and 0.15, overshooting by 1.4x to 3.7x at the tightest. Third, static conformal cali
    
[^76]: 有界停机问题的工程学解读与通用机的可观测性质

    An Engineering Reading of the Bounded Halting Problem and Observable Properties of a Universal Machine

    [https://arxiv.org/abs/2611.06303](https://arxiv.org/abs/2611.06303)

    本文将停机问题工程化地改写为给定步数预算 T 的有界停机判定问题，通过带明确开销的对角线论证证明任何固定确定性检查器都无法在 T 的任意固定比例内判定所有实例，并将有界停机实例用作检验计算模型有限通用性与可观测性质的对抗性测试。

    

    停机问题问的是一个程序在资源不受限制的情况下最终是否会停机。工程系统则转而认证有界停机：给定一个程序、一个输入和一个步数预算 T，判定该程序是否会在 T 步内停机。本文研究了这一问题的两种用途。首先，论文定义了一个精确的有限预算问题，将被模拟的程序与检查器置于同一个单位成本的时间尺度上，从而使检查器可以被嵌入到对抗性程序之中。一个带有明确加性开销的对角线论证表明，对于较大的 T，不存在任何固定的确定性检查器能够在小于 1 的任意固定比例的 T 内判定所有实例。在非确定性情形下，T 限制了实现稳定可观测判定所允许的分支深度。其次，有界停机实例为有限通用性提供了对抗性测试，因为一个通用模型必须随着 T 的增长支持任意有限的计算。这些测试检验资源随 T 的增长情况以及……（原文摘要在此处截断）

    arXiv:2611.06303v1 Announce Type: cross  Abstract: The halting problem asks whether a program eventually halts, with unrestricted resources. Engineering systems instead certify bounded halting: given a program, an input, and a step budget T, decide whether the program halts within T steps. This paper studies two of its uses. First, it defines an exact finite-budget problem, placing the simulated program and the checker on the same unit-cost timescale, so the checker can be embedded in an adversarial program. A diagonal argument with explicit additive overhead shows that no fixed deterministic checker decides every instance within any fixed fraction of T below one, for large T. In the nondeterministic case, T bounds the branch depth allowed for a stable observable decision. Second, bounded-halting instances give adversarial tests for finitary universality, since a universal model must support arbitrary finite computations as T grows. The tests check resource growth with T, the absence o
    
[^77]: STEK：面向回溯性电池多目标估计的结构-地形增强Koopman启发跨通道正则化方法

    STEK: Structural-Terrain-Enhanced Koopman-Inspired Cross-Channel Regularization for Retrospective Battery Multi-Target Estimation

    [https://arxiv.org/abs/2611.06299](https://arxiv.org/abs/2611.06299)

    STEK框架通过质量代理条件门控将电池级生命周期(dQ/dV)地形与端点充放电曲线相融合，并引入Koopman启发的跨通道一致性正则化，实现了跨异构电池的回溯性多目标估计。

    

    跨异构电池的回溯性电池表征极具挑战性，因为循环记录随化学体系、测试协议、电池形式、容量和工作条件的变化而存在显著差异。端点充放电曲线捕捉被评估循环的局部状态，而电池级生命周期地形则总结了存档轨迹中的结构演化。我们提出STEK，一个用于回溯性电池多目标估计的结构-地形增强框架。STEK通过质量代理条件门控将电池级生命周期(dQ/dV)地形与端点导出的结构代理相结合。有界FiLM和有效性门控残差头在保留基于曲线的预测路径的同时细化端点曲线表示。电池类型和可选的材料文本表示进一步对以恒等映射为中心的低秩线性映射进行条件化，提供Koopman启发的跨通道一致性目标。

    arXiv:2611.06299v1 Announce Type: new  Abstract: Retrospective battery characterization across heterogeneous cells is challenging because cycling records vary substantially with chemistry, protocol, cell format, capacity, and operating conditions. An endpoint charge--discharge curve captures the local condition of an evaluated cycle, while a cell-level lifecycle terrain summarizes structural evolution across the archived trajectory. We propose STEK, a structural-terrain-enhanced framework for retrospective battery multi-target estimation. STEK combines a cell-level lifecycle (dQ/dV) terrain with an endpoint-derived structural proxy through quality-proxy-conditioned gating. Bounded FiLM and a validity-gated residual head refine the endpoint-curve representation while preserving a curve-based prediction pathway. Battery-type and optional material-text representations further condition an identity-centered low-rank linear map, providing a Koopman-inspired cross-channel consistency objecti
    
[^78]: 基于MDP建模与行为克隆的智能家居多居民数据关联

    Multi-Resident Data Association in Smart Homes via MDP Modeling and Behavior Cloning

    [https://arxiv.org/abs/2611.06296](https://arxiv.org/abs/2611.06296)

    该论文提出将智能家居多居民数据关联问题重新建模为马尔可夫决策过程（MDP）中的序贯决策任务，并通过行为克隆直接从专家演示中学习求解，有效解决了现有方法全局一致性差和轨迹碎片化严重的问题。

    

    环境传感器为智能家居中的活动识别提供了一种隐私保护解决方案，但它们在多居民环境中面临一个关键挑战，即数据关联问题，其目标是识别触发特定匿名传感器事件的居民。现有方法从根本上无法保持全局一致性，常常为沉默的居民产生相互冲突的关联结果。此外，由于将连续物理运动模型应用于离散逻辑行为所导致的本体论不匹配，这些方法经常遭受严重的轨迹碎片化问题。为了解决这些局限性，我们提出了一个新颖的框架，将数据关联重新表述为马尔可夫决策过程（MDP）中的序贯决策任务。我们通过行为克隆（BC）来求解所定义的MDP，通过直接从专家演示中学习，从而避免了复杂的奖励工程。我们使用两个互补的（策略实现方式，摘要在此处截断）。

    arXiv:2611.06296v1 Announce Type: new  Abstract: Ambient sensors offer a privacy-preserving solution for activity recognition in smart homes, yet they face a critical challenge in multi-resident environments known as Data Association, which aims to identify the resident triggering a specific anonymous sensor event. Existing approaches fundamentally fail to maintain global consistency by often yielding conflicting associations for silent residents. Additionally, these methods frequently suffer from severe trajectory fragmentation caused by the ontological mismatch of applying continuous physical motion models to discrete logical behaviors. To address these limitations, we propose a novel framework that reformulates data association as a sequential decision-making task within a Markov Decision Process (MDP). We solve the defined MDP via Behavior Cloning (BC) to avoid complex reward engineering by directly learning from expert demonstrations. We instantiate the policy using two complement
    
[^79]: A.S.E 2.0：理解仓库上下文如何影响LLM生成代码的安全性

    A.S.E 2.0: Understanding How Repository Context Impacts the Security of LLM-Generated Code

    [https://arxiv.org/abs/2611.06294](https://arxiv.org/abs/2611.06294)

    该论文提出CASE-Bench基准测试，基于真实CVE构建145个仓库级任务，首次系统研究了仓库上下文构建方式如何影响LLM生成代码的安全性。

    

    大型语言模型（LLM）编码助手和智能体在生成代码之前会构建仓库上下文。对于安全敏感的代码，这一上下文决定了模型是能看到项目的安全防护措施（例如授权检查），还是会看到可供模仿的不安全代码。然而，现有的上下文构建几乎完全是为功能正确性设计的，而安全代码基准测试又将上下文视为固定不变，因此仓库上下文如何影响生成代码的安全性仍不明确。我们构建了CASE-Bench，这是一个包含来自51个项目中真实CVE的145个仓库级任务的基准测试。每个任务都在Docker中运行，配有功能测试和经过双向验证的概念验证漏洞利用，并标注了仓库中实施或破坏其安全性的代码。借助CASE-Bench，我们将13种上下文构建方法和六种对照方式与六个LLM进行了比较，对单个上下文元素进行编辑，并分析了编码智能体的运行轨迹。我们……

    arXiv:2611.06294v1 Announce Type: cross  Abstract: Large language model (LLM) coding assistants and agents construct a repository context before generating code. For security-sensitive code, this context decides whether the model sees the project's safeguards (e.g., authorization checks) or insecure code to imitate. However, context construction is designed almost exclusively for functional correctness, and secure-code benchmarks treat the context as fixed, so how repository context affects the security of generated code remains unclear. We construct CASE-Bench, a benchmark of 145 repository-level tasks from real CVEs in 51 projects. Each task runs in Docker, has functional tests and a proof-of-concept exploit validated in both directions, and is annotated with the repository code that enforces or undermines its security. Using CASE-Bench, we compare 13 context construction methods and six controls with six LLMs, edit single context elements, and analyze coding-agent trajectories. We f
    
[^80]: 分割协议决定结果：重新评估加密流量分类

    The Split Protocol Decides the Result: Re-Evaluating Encrypted Traffic Classification

    [https://arxiv.org/abs/2611.06290](https://arxiv.org/abs/2611.06290)

    该论文揭示加密流量分类报告的高准确率主要由数据分割协议决定——同一模型在不同分割协议下宏F1可从0.992骤降至0.567，且关键划分轴取决于语料库的收集方式，并提出了一个无需训练模型即可识别正确划分轴的诊断方法。

    

    加密流量分类器通常被报告具有95-99%的准确率。我们表明，在两个公开语料库之一上，相同的模型和代码在文献标准的评估协议下得到宏F1 0.992，而在一种将来自同一数据包捕获的流量保留在分割同一侧的协议下仅得到0.567。除了数据划分方式外没有任何改变（八个随机种子上的差距为0.426，经Holm-Bonferroni校正后p = 0.008）。这种虚高并非随机分割本身的属性。在一个月的骨干网QUIC流量上，我们以每个协议八个种子评估了五种分割协议：随机、时间序和客户端不重叠分割在统计上与会话不重叠分割无法区分，而服务器不重叠分割会损失0.192的宏F1（d = 17.0，p = 0.008）。哪个划分轴重要取决于语料库的收集方式，而这从其标签空间中无法看出。我们提供了一个诊断方法，可以在训练任何模型之前识别出正确的划分轴。

    arXiv:2611.06290v1 Announce Type: cross  Abstract: Encrypted traffic classifiers are routinely reported at 95-99% accuracy. We show that on one of two public corpora, the same model and code score macro-F1 0.992 under the evaluation protocol standard in the literature and 0.567 under a protocol that keeps flows from one packet capture on one side of the split. Nothing changes but the partition (gap 0.426 over eight seeds, p = 0.008 after Holm-Bonferroni correction).   The inflation is not a property of random splitting as such. On a month of backbone QUIC traffic, we evaluate five split protocols at eight seeds each: random, temporal and client-disjoint splitting are statistically indistinguishable from session-disjoint, while server-disjoint splitting costs 0.192 macro-F1 (d = 17.0, p = 0.008). Which axis matters depends on how the corpus was collected, in a way invisible from its label space.   We give a diagnostic that identifies the right axis before any model is trained: single-id
    
[^81]: MonoLaDN：面向单目3D目标检测的语言引导去噪方法

    MonoLaDN: Language-Guided DeNoising for Monocular 3D Object Detection

    [https://arxiv.org/abs/2611.06282](https://arxiv.org/abs/2611.06282)

    MonoLaDN在训练阶段利用冻结的视觉-语言模型生成关于目标遮挡、深度、尺寸和角度的结构化几何推理，并通过语言引导的去噪策略将其几何先验迁移到单目3D检测器中，部署时模型保持纯视觉推理，无需承担VLM的高昂推理成本。

    

    单目3D目标检测为3D感知提供了一种低成本的途径，但由于深度必须从单一视角进行推断，这本质上是一个病态问题。近期基于Transformer的检测器仅从图像特征和3D框标注中学习几何信息，缺乏视觉-语言模型（VLM）通过大规模预训练所获得的高层语义先验。VLM虽然能够提供此类先验，但由于其高昂的推理成本，直接将其部署为检测器并不现实。我们提出了MonoLaDN，这是一种语言引导的去噪框架，能够将VLM的几何推理能力迁移到单目3D检测器中，同时保持部署的模型纯粹基于视觉。在训练过程中，一个冻结的VLM根据图像及其真值框，对每个目标的遮挡、深度、尺寸和角度生成结构化推理。我们将这些推理编码为查询，通过添加噪声进行扰动，并训练检测器重建干净的查询（原文摘要在此处截断）。

    arXiv:2611.06282v1 Announce Type: cross  Abstract: Monocular 3D object detection offers a low-cost route to 3D perception, but it remains fundamentally ill-posed, since depth must be inferred from a single view. Recent transformer-based detectors learn geometry purely from image features and 3D box annotations, lacking the high-level semantic priors that vision-language models (VLMs) acquire from large-scale pretraining. VLMs offer such priors, but deploying one directly as a detector is impractical due to its heavy inference cost. We propose MonoLaDN, a language-guided denoising framework that transfers the geometric reasoning of a VLM into a monocular 3D detector while keeping the deployed model purely vision-based. During training, a frozen VLM produces structured reasoning about each object's occlusion, depth, dimension, and angle from the image and its ground-truth boxes. We encode this reasoning into queries, perturb them with noise, and train the detector to reconstruct the clea
    
[^82]: DyPSI：通过场与传感器轨迹联合生成实现动态物理感知

    DyPSI: Dynamic Physics Sensing via Joint Field and Sensor-Trajectory Generation

    [https://arxiv.org/abs/2611.06280](https://arxiv.org/abs/2611.06280)

    DyPSI提出了一种动态物理感知框架，通过连续传感器位置场、共享函数式Tucker表示与联合扩散模型，从历史观测中同时生成未来物理场和时间连贯的传感器运动轨迹，突破了传统静态传感器布置的局限。

    

    动态演化环境中的物理感知需要在稀疏观测的基础上重建稠密的时空场，同时自适应地决定未来观测应当在哪里采集。现有的重建与传感器布置方法已取得长足进展，但它们通常依赖于静态感知假设：要么在所有帧中使用单一的布置方案，要么在每一帧上独立地优化布置，这限制了其生成时间上连贯的感知轨迹的能力。我们提出了DyPSI，一个动态物理感知框架，它将该问题表述为以历史观测为条件的未来场与传感器轨迹的联合生成。DyPSI将离散的传感器坐标提升为连续的传感器位置场，利用共享的函数式Tucker表示对场和感知配置进行编码，并训练一个联合扩散模型来同时生成未来场与感知轨迹……

    arXiv:2611.06280v1 Announce Type: new  Abstract: Physics sensing in evolving environments requires reconstructing dense spatiotemporal fields from sparse observations while adapting where future observations should be collected. Existing reconstruction and sensor-placement methods have made substantial progress, but they typically rely on static sensing assumptions: they either use a single layout across all frames or optimize placements independently at each frame, which limits their ability to produce temporally coherent sensing trajectories. We introduce DyPSI, a dynamic physics sensing framework that formulates this problem as the joint generation of future fields and sensor trajectories conditioned on historical observations. DyPSI lifts discrete sensor coordinates into continuous Sensor Position Fields, encodes fields and sensing configurations with a shared Functional Tucker representation, and trains a joint diffusion model to generate future fields and sensing trajectories sim
    
[^83]: 驾驶场景重建将何去何从？遮挡感知的高斯稠密化

    Quo Vadis, Driving Scene Reconstruction? Occlusion-Aware Gaussian Densification

    [https://arxiv.org/abs/2611.06273](https://arxiv.org/abs/2611.06273)

    OccSplat提出遮挡感知的高斯稠密化方法，通过高斯遮挡图识别缺乏高斯支撑的遮挡区域，并在3D表示中直接预测新的高斯基元以补全缺失的场景几何，从而提升前馈式3D高斯泼溅在驾驶场景新视角合成中的完整性。

    

    尽管近期的前馈式3D高斯泼溅（3D Gaussian Splatting）方法能够从环视图像中高效地重建驾驶场景，但其对单一观测的依赖从根本上限制了新视角合成的效果。输入中被遮挡的区域在新视角下可能变得可见，这导致重建出的高斯表示约束不足且不完整。为了超越仅对渲染视图进行图像空间修复的思路，我们提出了OccSplat——一个前馈框架，其核心贡献是遮挡感知的高斯稠密化，明确针对缺失的场景几何而非缺失的像素。通过为缺乏支撑的区域预测新的高斯基元，OccSplat直接在3D表示层面完成场景补全。具体而言，OccSplat首先通过高斯遮挡图识别出缺乏高斯支撑的遮挡区域，随后由高斯遮挡补全模块为这些区域生成新的基元。所预测的基元……

    arXiv:2611.06273v1 Announce Type: cross  Abstract: While recent feed-forward 3D Gaussian Splatting methods enable efficient driving-scene reconstruction from surround-view images, their reliance on a single observation fundamentally hinders novel-view synthesis. Regions occluded in the input can become visible from novel viewpoints, leaving the reconstructed Gaussians under-constrained and incomplete. Moving beyond image-space inpainting for rendered views, we introduce OccSplat, a feed-forward framework whose core contribution is Occlusion-Aware Gaussian Densification, explicitly targeting missing scene geometry rather than missing pixels. By predicting new Gaussian primitives for unsupported regions, OccSplat completes the scene directly in the 3D representation. Specifically, OccSplat first identifies occluded regions that lack Gaussian support through Gaussian Occlusion Mapping. Gaussian Occlusion Completion then generates new primitives for these regions. The predicted primitives 
    
[^84]: NextGAN：迈向简单且可扩展的图像与视频生成GAN训练

    NextGAN: Towards Simple and Scalable GAN Training for Image and Video Generation

    [https://arxiv.org/abs/2611.06270](https://arxiv.org/abs/2611.06270)

    NEXTGAN提出通过将标量评论家的输入梯度回归到冻结表示空间中配对特征间的位移来直接约束生成器的更新梯度，实现了简单且可扩展的GAN训练，无需辅助损失即可高效适配高达140亿参数的预训练图像与视频生成器，并取得先进的单步生成效果。

    

    对抗训练能够实现单步和少步的图像与视频生成，但传统的对抗损失仅监督评论家的数值，并未直接规定用于更新生成器的梯度。我们提出了NEXTGAN，该方法通过在冻结的表示空间中，将标量评论家的输入梯度回归到独立配对的参考特征与生成特征之间在插值位置上的位移，从而训练该评论家。对于固定的生成器，这一回归等价于带有梯度能量惩罚的对抗目标，而生成器则保留标准的基于势函数的GAN更新。同一目标函数可以适配参数量高达14B的预训练图像和视频生成器，无需辅助损失、在线教师评分或轨迹。在ImageNet 256×256上，NEXTGAN在像素空间的JiT-B下取得了2.87的单步FID，在潜在空间的DiT-XL/1下取得了1.54；其对抗形式进一步达到1.36。

    arXiv:2611.06270v1 Announce Type: cross  Abstract: Adversarial training enables one- and few-step image and video generation, but conventional adversarial losses supervise critic values without directly specifying the gradients used to update the generator. We present NEXTGAN, which trains a scalar critic by regressing its input gradient onto displacements between independently paired reference and generated features at interpolated locations in frozen representation spaces. For a fixed generator, this regression is equivalent to an adversarial objective with a gradient-energy penalty, while the generator retains the standard potential-based GAN update. The same objective adapts pre-trained image and video generators with up to 14B parameters, without auxiliary losses, online teacher scores, or trajectories. On ImageNet $256 \times 256$, NEXTGAN achieves one-step FIDs of 2.87 with pixel-space JiT-B and 1.54 with latent-space DiT-XL/1; its adversarial form further reaches 1.36. For text
    
[^85]: 基于图符号的异质图伪黎曼流形学习

    Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature

    [https://arxiv.org/abs/2611.06265](https://arxiv.org/abs/2611.06265)

    该论文提出图符号指数（s_G）来量化正定几何先验与异质图之间的根本错配，并据此提出伪黎曼注意力网络（PRAT），通过带符号的平方伪距离注意力机制在伪黎曼流形上学习异质图表示。

    

    arXiv:2611.06265v1 公告类型：cross（交叉发布）。摘要：若干广泛使用的图神经网络家族都带有正定几何先验。当图卷积被解读为热扩散时需要半正定算子，双曲模型将特征放置在正定黎曼流形上，而点积和GAT风格的注意力分数则使用正定内积。然而在异质图（heterophilic graphs）上，边往往表示不相似性或结构上的排斥，这种先验便产生了错配：任何正定形式都无法为标签所区分的方向赋予负的平方距离。为了度量这种错配，我们引入了图符号指数，它是仅由标签拓扑构建的类相容矩阵的惯性，并证明了在(0,1)区间内，任何边同质指数或节点同质指数的取值都无法排除负特征值的存在。随后，我们提出了伪黎曼注意力网络（PRAT），其注意力对数是一个带符号的平方伪距离……（摘要在此处不完整）

    arXiv:2611.06265v1 Announce Type: cross  Abstract: Several widely used families of Graph Neural Networks carry a positive-definite geometric prior. Graph convolution read as heat diffusion needs a positive semi-definite operator, hyperbolic models place features on positive-definite Riemannian manifolds, and dot-product and GAT-style attention scores use positive-definite inner products. On heterophilic graphs, where edges often signify dissimilarity or structural repulsion, that prior is a mismatch: no positive-definite form can assign a negative squared distance to a direction the labels separate. To measure the mismatch, we introduce the Graph Signature Index ($s_{\mathcal{G}}$), the inertia of a class-compatibility matrix built from label topology alone, and we prove that no value of the edge or node homophily index in $(0,1)$ rules out a negative eigenvalue. We then propose the Pseudo-Riemannian Attention Network (PRAT), whose attention logit is a signed squared pseudo-distance un
    
[^86]: 《一个用于评估视觉-语言模型多模态个性化推理能力的系统性基准》

    A Systematic Benchmark for Evaluating Multimodal Personalized Reasoning of Vision-Language Models

    [https://arxiv.org/abs/2611.06259](https://arxiv.org/abs/2611.06259)

    该论文提出了MPRBench——一个包含12个子任务、支持错误案例可解释性分析的系统性基准，用于评估多模态大语言模型的多模态个性化推理能力，并揭示出现有模型在对个性化概念识别不准确和对无关个人信息敏感方面的关键不足。

    

    人们常常通过对异构形式的个人信息进行整体推理来做出决策。在本文中，我们研究了多模态大语言模型（MLLMs）中一项相对未被充分探索的能力——多模态个性化推理。为此，我们提出了MPRBench，一个新颖且全面的基准，专门设计用于评估跨广泛任务和现实挑战的个性化推理能力。MPRBench包含12个子任务，并额外支持对代表性错误案例的可解释性分析。通过大量实验，我们发现当前的个性化多模态大语言模型在个性化推理方面仍存在困难，主要面临几个关键挑战，包括对个性化概念识别不准确以及对无关个人信息过于敏感。我们希望MPRBench能够激发对多模态大语言模型个性化推理的进一步研究。代码和数据可在https://获取

    arXiv:2611.06259v1 Announce Type: new  Abstract: People often make decisions by holistically reasoning over heterogeneous forms of personal information. In this paper, we study the relatively underexplored capability of multimodal personalized reasoning in multimodal large language models (MLLMs). To this end, we introduce \texttt{MPRBench}, a novel and comprehensive benchmark specifically designed to evaluate personalized reasoning across a wide range of tasks and real-world challenges. \texttt{MPRBench} consists of 12 sub-tasks and additionally supports interpretable analysis of representative error cases. Through extensive experiments, we find that current personalized MLLMs still struggle with personalized reasoning due to several key challenges, including inaccurate recognition of personalized concepts and sensitivity to irrelevant personal information. We hope that MPRBench will stimulate further research on personalized reasoning in MLLMs. Code and data are available at https://
    
[^87]: HarnessFlow：面向大语言模型智能体的MCTS引导式执行框架优化方法

    HarnessFlow: MCTS-Guided Harness Optimization for LLM Agents

    [https://arxiv.org/abs/2611.06254](https://arxiv.org/abs/2611.06254)

    HarnessFlow通过MCTS引导的精细化改进树搜索与编码智能体，在有限评估预算下自动优化LLM智能体的完整可执行执行框架，在六个基准测试中取得最高平均测试分数。

    

    大语言模型（LLM）智能体的性能既取决于底层模型，也取决于控制其与工具和环境交互的执行框架。然而，框架设计在很大程度上仍依赖人工完成，而实现自动化则需要通过代价高昂的端到端评估在一个开放式的搜索空间中进行探索。我们提出了HarnessFlow，这是一个在候选评估预算有限、且模型与任务环境保持固定的条件下，对完整可执行框架进行优化的框架。HarnessFlow将候选方案及其评估历史组织在一棵精细化改进树中，并采用基于蒙特卡洛树搜索的策略来平衡利用与探索。一个编码智能体则利用归档代码、执行反馈以及可复用框架机制库，对被选中的框架进行迭代改进。在每个基准测试仅需15次精细化尝试的条件下，HarnessFlow在涵盖智能体任务的六个基准测试中，于所有对比方法中取得了最高的平均测试分数。

    arXiv:2611.06254v1 Announce Type: new  Abstract: The performance of large language model (LLM) agents depends on both the underlying model and the harness that governs interaction with tools and the environment. Yet harness design remains largely manual, and automating it requires navigating an open-ended search space through costly end-to-end evaluations. We introduce HarnessFlow, a framework for optimizing complete executable harnesses under a limited candidate-evaluation budget, with the model and task environment held fixed. HarnessFlow organizes candidates and their evaluation histories in a refinement tree, using a Monte Carlo Tree Search (MCTS)-based policy to balance exploitation and exploration. A coding agent refines selected harnesses using archived code, execution feedback, and a library of reusable harness mechanisms. With 15 refinement attempts per benchmark, HarnessFlow achieves the highest mean test scores among the compared methods on six benchmarks spanning agentic ta
    
[^88]: 评分标准课程强化学习：在不可验证领域利用生成-验证差距

    Rubric Curriculum RL: Exploiting the Generation-Verification Gap in Non-Verifiable Domains

    [https://arxiv.org/abs/2611.06244](https://arxiv.org/abs/2611.06244)

    RcRL 通过将对采样结果的成对偏好与评分标准课程机制相结合，在无需新数据、人工标注或更强教师模型的情况下，利用生成-验证差距在不可验证的开放式领域实现持续的性能提升并有效避免奖励作弊。

    

    带可验证奖励的强化学习（RLVR）在数学和代码生成领域取得了显著成果。然而，将这些成果扩展到开放式领域仍然充满挑战：真值验证不可用，人工标注成本高昂，且学习到的奖励模型容易遭受奖励作弊。生成-验证差距——即评判一个高质量输出比生成它更容易——为改进提供了一条潜在路径。然而，基于评分标准的评判器往往不精确且固定不变，导致策略容易对其过拟合或进行奖励作弊。我们提出了评分标准课程强化学习，它在无需新数据、人工标注或更强教师模型的情况下利用这一差距。我们将对采样结果的成对偏好与评分标准上的课程机制相结合，从而产生比绝对评分更具判别性的信号以及一个非平稳的优化目标。基线方法通常在几百步内就陷入停滞，而 RcRL 能够保持输出熵并持续提升（摘要在此处截断）。

    arXiv:2611.06244v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has led to significant gains in math and code generation. Extending these gains to open-ended domains remains challenging: ground-truth verification is unavailable, human annotation is expensive, and learned reward models are prone to reward hacking. The generation-verification gap, where judging good outputs is easier than producing them, offers a potential path to improvement. However, rubric-based judges are often imprecise and stationary, so policies can overfit or reward hack them. We introduce Rubric Curriculum RL (RcRL), which exploits this gap without new data, human annotations, or stronger teacher models. We combine pairwise preferences over rollouts with a curriculum over rubric criteria, yielding a more discriminative signal than absolute scoring and a non-stationary objective. Whereas baselines plateau within a few hundred steps, RcRL preserves output entropy and keeps 
    
[^89]: 灾难性性能的定义

    A Definition of Catastrophic Performance

    [https://arxiv.org/abs/2611.06219](https://arxiv.org/abs/2611.06219)

    该论文提出了“灾难性性能”的正式定义，通过基线值和正交值两个参考值来区分次优结果与灾难性结果，从而为严格分析人工智能目标错位带来的灾难性风险提供了理论基础。

    

    人工智能的一个核心挑战是制定与人类偏好相一致的目标。虽然现有的理论结果阐明了目标设定错误如何影响性能，但它们通常没有区分仅仅是次优的结果与灾难性的结果。为了能够对人工智能目标错位带来的灾难性风险进行严格分析，我们引入了灾难性性能的正式定义。我们的定义基于两个参考值：第一个是基线值，表示如果没有部署任何智能体时所获得的价值；第二个是正交值，表示一条在我们所描述意义上与人类偏好无关的轨迹的价值。正交值通常远差于基线值。我们通过一些例子来阐释这些参考值，在这些例子中，过去的人类活动已经改变了环境状态并使其有利于人类。从由此产生的状态出发，部署一个以加速进一步……

    arXiv:2611.06219v1 Announce Type: new  Abstract: A central challenge in AI is specifying objectives that align with human preferences. While existing theoretical results elucidate how misspecification impacts performance, they typically do not distinguish between merely suboptimal and catastrophic outcomes. To enable rigorous analysis of catastrophic risk from AI misalignment, we introduce a formal definition of catastrophic performance. Our definition is based on two reference values. The first, baseline value, captures the value attained if no agent is deployed. The second, orthogonal value, captures the value of a trajectory that is unrelated to human preferences, in a sense we describe. Orthogonal value is typically far worse than baseline value. We interpret these reference values through examples in which past human activity has altered the state of the environment to human benefit. Starting at the resulting state, an AI agent is deployed with the goal of accelerating further pro
    
[^90]: CoSafe：面向混合交通自动驾驶的协同式运行时安全保障

    CoSafe: Cooperative Runtime Safety Assurance for Mixed-Traffic Autonomous Driving

    [https://arxiv.org/abs/2611.06204](https://arxiv.org/abs/2611.06204)

    提出CoSafe——一个模块化、与自动驾驶系统无关的协同式运行时安全保障框架，通过多车协作解决混合交通中由局部合理但联合不兼容决策引发的安全违规问题。

    

    随着自动驾驶系统（ADS）的快速发展，现实道路正在演变为混合交通环境，其中自动驾驶汽车（AV）与其他交通参与者（如人类驾驶车辆（HDV）、骑行者和行人）共存并相互作用。然而，当前的自动驾驶系统在混合交通场景中仍会出现各种安全违规行为，这引发了对能够持续监测危险并防止潜在违规的运行时安全保障框架的强烈需求。现有的以自我为中心的保障方法不足以应对混合交通场景，因为在这类场景中，违规行为往往源于多辆车之间局部合理但联合不兼容的决策，且无法通过独立接管来可靠解决。为弥补这一空白，我们提出了CoSafe，一个模块化且与自动驾驶系统无关的协同式运行时安全保障框架，用于混合交通自动驾驶，以将运行时安全扩展为……

    arXiv:2611.06204v1 Announce Type: cross  Abstract: With the rapid development of autonomous driving systems (ADSes), real-world roads are evolving into mixed-traffic environments where autonomous vehicles (AVs) and other traffic participants (e.g., human-driven vehicles (HDVs), cyclists, and pedestrians) coexist and interact with each other. However, current ADSes still incur various safety violations in mixed traffic scenarios, raising a strong demand for runtime safety assurance frameworks that can continuously monitor hazards and prevent potential violations. Existing ego-centric assurance approaches are insufficient for mixed traffic, where violations often arise from locally reasonable yet jointly incompatible decisions among multiple vehicles and cannot be reliably resolved through independent takeover.   To bridge this gap, we propose CoSafe, a modular and ADS-agnostic cooperative runtime safety assurance framework for mixed-traffic autonomous driving to extend runtime safety as
    
[^91]: 基于几何感知正则化的大型语言模型编辑

    Editing Large Language Models with Geometry-Aware Regularization

    [https://arxiv.org/abs/2611.06185](https://arxiv.org/abs/2611.06185)

    提出GEdit框架，通过低秩加对角（LR+D）因子模型捕捉知识表示的几何结构，利用低秩正则化保护核心知识子空间，并依据编辑难度自适应调整正则化强度，实现更精准的大模型知识编辑。

    

    知识编辑已成为更新大型语言模型知识的关键技术。然而，现有方法忽视了知识表示空间的几何结构，导致两个关键局限：它们既无法区分哪些预训练知识应被优先保留（核心知识 vs. 非核心知识），也无法区分哪些新编辑请求本质上较为困难（困难编辑 vs. 简单编辑）。为解决这一问题，我们提出了GEdit，一个基于低秩加对角（LR+D）因子模型的几何感知框架，该模型能够捕捉Transformer前馈网络（FFN）隐藏表示的内在低秩结构。GEdit包含两个互补机制：首先，低秩正则化对低秩核心知识子空间施加针对性保护；其次，自适应正则化根据每个编辑请求的可编辑性动态调整正则化强度——对困难编辑放松约束以确保编辑成功，对（原文摘要在此处截断）。

    arXiv:2611.06185v1 Announce Type: cross  Abstract: Knowledge editing has become a key technique for updating knowledge in large language models. However, existing methods neglect the geometric structure of the knowledge representation space, leading to two key limitations: they cannot distinguish which pre-trained knowledge should be preferentially preserved (core vs. non-core) nor which new edits are inherently difficult (hard vs. easy). To address this, we propose GEdit, a geometry-aware framework built on a Low-Rank Plus Diagonal (LR+D) factor model that captures the intrinsic low-rank structure of Transformer FFN hidden representations. GEdit features two complementary mechanisms. First, the low-rank regularization applies targeted protection on a low-rank core knowledge subspace. Second, the adaptive regularization dynamically adjusts the regularization strength based on each edit request's editability: relaxing constraints for hard edits to ensure success, and tightening them for
    
[^92]: AssameseLM-E2B：一个面向消费级设备高效部署的开源权重阿萨姆语语言模型

    AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment

    [https://arxiv.org/abs/2611.06170](https://arxiv.org/abs/2611.06170)

    本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。

    

    尽管大型语言模型（LLM）在高资源语言中表现出色，但由于分词效率低下、生成不稳定性和语言干扰，其在低资源语言中的性能往往会有所下降。本文提出了 AssameseLM-E2B 及其指令微调变体 AssameseLM-E2B-it，这是一个专门为低资源阿萨姆语适配的 20 亿参数模型。该模型基于 Gemma-4 架构构建，并采用扩展的字节级 BPE 词表，与多语言 2B–3B 基线相比，将分词开销降低了近 500%。我们采用结合 Hybrid LoRA 的参数高效持续预训练（CPT）流程，随后在平衡的多领域数据集上进行监督微调（SFT），以增强指令遵循和检索增强生成（RAG）能力。评估结果显示，该模型在 FLORES-200 上取得了 0.7488 的每字节比特数（BPB）得分，同时总体 RAG 得分为 96。

    arXiv:2611.06170v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) excel in high-resource languages, their performance often degrades in low-resource languages due to tokenization inefficiencies, generative instability, and linguistic interference. This paper presents AssameseLM-E2B and its instruction-tuned variant, AssameseLM-E2B-it, a 2-billion-parameter model adapted specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead by nearly 500% compared with multilingual 2B--3B baselines. We employ a parameter-efficient Continual Pre-Training (CPT) pipeline with Hybrid LoRA, followed by Supervised Fine-Tuning (SFT) on a balanced, multi-domain dataset to enhance instruction-following and Retrieval-Augmented Generation (RAG) capabilities.   Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, alongside an overall RAG score of 96
    
[^93]: 多模态大型推理模型中推理感知的跨模态关联遗忘

    Reasoning-Aware Cross-Modal Association Unlearning in Multimodal Large Reasoning Models

    [https://arxiv.org/abs/2611.06168](https://arxiv.org/abs/2611.06168)

    该论文提出了多模态大型推理模型中的“跨模态关联遗忘”这一新问题，强调遗忘不仅要让目标信息从最终答案中消失，还需在中间推理过程中抑制指定的视觉到知识关联，防止敏感信息经由推理链泄露或被错误替代知识取代。

    

    多模态大型推理模型（MLRMs）将视觉输入与文本及参数化知识相结合，使得在推理过程中可以通过跨模态关联访问敏感信息。当某个视觉输入不应再提供对这些信息的访问途径时，这会引发隐私担忧。机器遗忘技术可以在无需从头重新训练模型的情况下消除这种影响。然而，现有的多模态遗忘方法主要针对样本、概念或知识内容，而未能将知识本身与其被访问时所经由的跨模态关联区分开来。在推理模型中，目标信息可能从最终答案中消失，却仍然残留在中间推理过程中，或者被错误的替代知识所取代。因此，我们研究跨模态关联遗忘问题，其中遗忘目标是指定的视觉到知识关联：模型应在指定的（关联被触发时抑制包含目标信息的行为，摘要在此处截断）

    arXiv:2611.06168v1 Announce Type: cross  Abstract: Multimodal large reasoning models (MLRMs) integrate visual inputs with textual and parametric knowledge, allowing sensitive information to be accessed through cross-modal associations during reasoning. This raises privacy concerns when a visual input should no longer provide access to such information. Machine unlearning can remove this influence without retraining the model from scratch. However, multimodal unlearning methods mainly target samples, concepts, or knowledge content, without separating the knowledge from the cross-modal association through which it is accessed. In reasoning models, target information may disappear from the final answer while remaining in intermediate reasoning, or be replaced by incorrect substitute knowledge. We therefore study cross-modal association unlearning, where the forget target is a designated visual-to-knowledge association: the model should suppress target-bearing behavior under the designated
    
[^94]: CARM：用于可靠科学超分辨率的保形自适应风险掩码

    CARM: Conformal Adaptive Risk Masking for Reliable Scientific Super-Resolution

    [https://arxiv.org/abs/2611.06151](https://arxiv.org/abs/2611.06151)

    提出CARM框架，通过维护异构更新速率的专家并利用非对称代理损失在线聚合阈值，在时间分布偏移下实现自适应空间风险控制，为科学超分辨率生成可靠的信任掩码。

    

    基于深度学习的超分辨率（SR）技术可以从粗分辨率观测中重建精细尺度的地球观测场，但仅靠重建精度本身并不能提供一个经过校准的规则来识别可靠的空间区域。保形掩码校准（CMC）建立在保形风险控制的基础上，将空间不确定性分数转换为信任掩码，并在可交换性假设下控制预期最坏情况的局部失真。然而，其固定阈值在受季节性和偶发事件影响的地球观测数据流中可能会失去校准。我们提出了CARM（保形自适应风险掩码），这是一个在时间分布偏移下实现自适应空间风险控制的统计框架。CARM维护多个具有异构更新速率的专家，并使用非对称代理损失在线聚合它们的阈值提议，从而在快速适应分布变化与相对平稳时期的稳定性之间取得平衡。

    arXiv:2611.06151v1 Announce Type: new  Abstract: Deep learning-based super-resolution (SR) can reconstruct fine-scale Earth-observation fields from coarse observations, but reconstruction accuracy alone does not provide a calibrated rule for identifying reliable spatial regions. Conformal Mask Calibration (CMC), built on conformal risk control, converts a spatial uncertainty score into a trust mask and controls the expected worst-case local distortion under exchangeability. However, its fixed threshold can become miscalibrated in Earth-observation streams affected by seasonality and episodic events. We propose $\textit{CARM}$ (Conformal Adaptive Risk Masking), a statistical framework for adaptive spatial risk control under temporal distribution shift. CARM maintains experts with heterogeneous update rates and aggregates their threshold proposals online using an asymmetric surrogate loss, balancing rapid adaptation to distribution changes against stability during relatively stationary p
    
[^95]: 在黑暗中定位：仅用文本训练实现零样本视频时序定位

    Grounding in the Dark: Text-Only Training for Zero-Shot Video Temporal Localization

    [https://arxiv.org/abs/2611.06149](https://arxiv.org/abs/2611.06149)

    提出一种仅用文本视频描述训练大语言模型（无需配对视频数据和视觉-语言预训练）来实现零样本视频时序定位的方法，通过LoRA在LLM2CLIP文本嵌入上适配Qwen3-8B，在推理时才首次接触视频即可取得优异的时序定位效果。

    

    视频-语言大模型（Video-LLMs）通常在配对的视频-文本数据上进行时序任务训练，然而收集精确的时序标注以及在大规模视频数据集上训练都非常昂贵。我们研究了一个没有经过视觉-语言预训练的纯文本大语言模型，是否能够仅通过文本形式的视频描述进行训练来执行时序定位，并只在推理阶段接收视频输入。这种设置在训练中完全去除了配对的视频数据，是迈向无需配对标注的多模态模型训练的一步。我们生成了一个包含13.4万条样本、最多100个时间步的文本序列语料库，使用LoRA基于CLIP或LLM2CLIP文本嵌入对Qwen3-8B进行适配，并在Charades-STA和QV-Highlights上进行评估，且未在这两个基准的任何视频上进行训练。LLM2CLIP的文本编码器支持更长的输入，比CLIP能更好地表示详细描述（在Temporal-TextVid上mtIoU为61.54对53.86），并能将其更好地迁移到视频上。

    arXiv:2611.06149v1 Announce Type: cross  Abstract: Video-LLMs are typically trained on paired video-text data for temporal tasks, yet collecting precise temporal annotations and training on large-scale video datasets are both costly. We investigate whether a text-only LLM, with no vision-language pretraining, can be trained on textual video descriptions alone to perform temporal grounding and receive video only at inference time. This setting removes paired video data from training altogether and is a step toward training multimodal models without paired annotations. We generate a 134K-example corpus of textual sequences with up to 100 time steps, adapt Qwen3-8B with LoRA on CLIP or LLM2CLIP text embeddings, and evaluate on Charades-STA and QV-Highlights without training on videos from either benchmark. LLM2CLIP, whose text encoder supports longer inputs, represents detailed descriptions better than CLIP (61.54 versus 53.86 mtIoU on Temporal-TextVid) and transfers them better to video.
    
[^96]: SafeStart：缓解MoE大语言模型服务中主动专家削减所带来的被忽视的安全成本

    SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving

    [https://arxiv.org/abs/2611.06147](https://arxiv.org/abs/2611.06147)

    本文揭示了MoE大模型服务中减少激活专家数量的加速方法会削弱模型对有害请求的拒绝能力，并提出SafeStart方法，在保留大部分加速效果的同时大幅恢复模型的安全性。

    

    arXiv:2611.06147v1 公告类型：新论文 摘要：混合专家模型将每个token仅通过众多专家子网络中的少数几个进行处理。现有的服务加速方法通过让每个token经过更少的专家来提升推理速度，并通常在准确率基准上进行验证。我们展示了这些方法会削弱模型对有害请求的拒绝能力，并提出了SafeStart，它在保留大部分加速效果的同时，能够大幅恢复模型的拒绝能力。在单张RTX A6000上使用vLLM的实验中，保留约三分之二的被选中专家可在batch size为8时将离线解码速度提升15%到28%。我们发现，在此设置下，在一种指示模型不要拒绝的越狱攻击下，模型回答的有害请求比例在Qwen1.5-MoE上上升18.4个百分点，在OLMoE上上升9.5个百分点，在DeepSeek-V2-Lite上上升6.6个百分点（这是九个测试模型中显著上升的三个），而MMLU准确率下降不到2分。我们还发现，这种上升大部分发生在模型处理提示词的阶段。

    arXiv:2611.06147v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) language models send each token through a few of many expert subnetworks. Serving methods make these models faster by running even fewer experts per token, and they are validated on accuracy benchmarks. We show that these methods can weaken the model's refusal of harmful requests, and we introduce SafeStart, which largely restores refusal while keeping most of the speedup. In vLLM on one RTX A6000, keeping about two thirds of the selected experts speeds up offline decoding by 15 to 28% at batch size 8. We find that, at this setting, under a jailbreak that tells the model not to refuse, the share of harmful requests that the model answers rises by 18.4 percentage points on Qwen1.5-MoE, 9.5 on OLMoE, and 6.6 on DeepSeek-V2-Lite, the three of nine models where it rises significantly, while MMLU accuracy drops by less than two points. We discover that most of this increase arises while the model processes the prompt 
    
[^97]: ModalSearcher：将视觉证据编织进自主跨模态深度搜索

    ModalSearcher: Weaving Visual Evidence into Autonomous Cross-Modal Deep Search

    [https://arxiv.org/abs/2611.06131](https://arxiv.org/abs/2611.06131)

    提出自主跨模态搜索新范式ModalSearcher，使智能体能在直接推理、文本搜索、文图搜索与反向图搜之间自主决策，在多跳搜索中动态引入并整合视觉证据，并配套可扩展的ModalSearch-Pipeline构建交错多模态知识图谱以支持训练。

    

    静态且不完整的参数化知识限制了多模态大语言模型在新颖知识和知识密集型任务上的性能，尤其是那些需要多跳证据获取的任务。现有多模态智能体搜索方法通常将视觉证据固定在输入端或预定义的轨迹位置上，使得智能体难以随着信息需求的演变来选择搜索模态，也难以处理复杂的现实世界查询。我们将这种能力定义为自主跨模态搜索：智能体可以在直接推理、文本到文本搜索、文本到图像搜索和反向图像搜索之间进行选择，在适当时机主动引入视觉证据，并在多跳过程中在不同模态之间切换与整合。我们提出了ModalSearch-Pipeline，这是一种用于构建交错多模态知识图谱的可扩展流程。它采样具有不同深度和模态组合的路径，并使用反向……（摘要原文在此处截断）

    arXiv:2611.06131v1 Announce Type: new  Abstract: Static and incomplete parametric knowledge limits the performance of multimodal large language models on novel-knowledge and knowledge-intensive tasks, especially those requiring multi-hop evidence acquisition. Existing multimodal agentic search methods often fix visual evidence at the input or a predefined trajectory position, making it difficult for agents to select search modalities as their information needs evolve and to handle complex real-world queries. We define this capability as autonomous cross-modal search: an agent chooses among direct reasoning, text-to-text search, text-to-image search, and reverse-image search, actively introduces visual evidence when appropriate, and switches between and integrates modalities across multiple hops. We propose ModalSearch-Pipeline, a scalable procedure for constructing an interleaved multimodal knowledge graph. It samples paths with diverse depths and modality compositions and uses backwar
    
[^98]: 从手工绘制到生成式：面向 Alexa+ 的 RLVR 优化图标生成

    From Handcrafted to Generative: RLVR-Optimized Icon Generation for Alexa+

    [https://arxiv.org/abs/2611.06123](https://arxiv.org/abs/2611.06123)

    该论文提出并部署了一套仅凭文本描述自动生成矢量化动画图标的生产级流水线，并将图标创建分解为多个阶段，重点利用 RLVR（可验证奖励的强化学习）优化其中决定整体质量的最关键阶段——图像生成，从而解决手工绘制图标耗时且难以规模化的问题。

    

    配备屏幕的 Alexa+ 设备会在对话过程中显示动态图标，以视觉方式强化所表达的内容。最初的这批图标由设计师逐个手工绘制，这种方式虽能产出精美的成果，但耗时且无法扩展至开放式对话中遇到的各种话题。我们描述了一条已部署于生产环境的自动化流水线，它能够仅凭文本描述生成这些矢量化动态图标，并根据用户正在进行的对话向其展示相关的动画。该流水线将图标创建分解为一系列阶段（图像生成、语义分解、矢量化、样式化和动画制作）。其质量在最前面的图像生成阶段就被决定，所有下游阶段都依赖于这一阶段。因此，我们专注于改进这一首要且最关键的阶段。由于这些图标需要在严格的设计约束下在小尺寸显示画布上渲染……

    arXiv:2611.06123v1 Announce Type: new  Abstract: Alexa+ devices with screens display animated icons during conversations to visually reinforce what is being said. The original set of these icons was hand-drawn by designers one at a time, a process that produces polished results but is time-consuming and cannot scale to the range of topics encountered in open-ended conversation. We describe an automated pipeline, deployed in production, that generates these vectorised animated icons from a text description alone and surfaces relevant animations to customers based on their ongoing conversation. The pipeline decomposes icon creation into a sequence of stages (image generation, semantic decomposition, vectorisation, styling, and animation). Its quality is determined at the very first of these, image generation, on which every downstream stage depends. We therefore focus on improving this first and most critical stage. Because the icons are rendered on small display canvases under strict de
    
[^99]: 超越截断半径的第一性原理精度：千万原子规模下的分布式消息传递等变注意力

    Ab Initio Accuracy Beyond the Cutoff: Distributed Message-Passing Equivariant Attention at Tens-of-Millions-of-Atom Scale

    [https://arxiv.org/abs/2611.06122](https://arxiv.org/abs/2611.06122)

    提出 Tessera 分布式运行时系统，通过空间填充曲线分片和固定张量形状克服通信瓶颈，首次将具有第一性原理精度的等变消息传递 Transformer 分子动力学模拟扩展到千万原子规模。

    

    分子动力学通过原子力和轨迹积分来预测分子运动。密度泛函理论（DFT）能够提供第一性原理力，但其 O(N³) 的计算成本将模拟限制在数千个原子；经验力场虽然可扩展至数百万原子，却无法捕捉化学键断裂和电荷转移现象。等变消息传递 Transformer 能够以线性成本学习第一性原理势能面。通过堆叠多层网络，信息可以传播至超出单个截断半径的范围，四层网络即可获得高达 23 埃的感受野。在这一前沿领域，规模扩展受限于通信瓶颈：各层需要在设备之间交换边界特征，力的反向传播需要重复这种交换，而原子的运动还会不断改变邻居图和张量形状。我们提出了 Tessera，一种分布式运行时系统，它在两次重建之间固定工作负载的形状。该系统沿空间填充曲线将原子排序为等数量的分片，确保每个进程（rank）拥有相同的原子数和张量形状。对等节点集合随 K^（摘要在此处被截断）

    arXiv:2611.06122v1 Announce Type: cross  Abstract: Molecular dynamics predicts molecular motion from atomic forces and trajectory integration. Density functional theory (DFT) provides first-principles forces, but its $O(N^3)$ cost limits simulations to thousands of atoms; empirical force fields scale to millions but miss bond breaking and charge transfer. Equivariant message-passing Transformers learn the ab initio potential at linear cost. Stacked layers propagate information beyond a single cutoff, giving four layers a receptive field up to 23{\AA}. At this frontier, scale is limited by communication: layers exchange boundary features across devices, force backpropagation repeats the exchange, and moving atoms change neighbor graphs and tensor shapes. We present Tessera, a distributed runtime that fixes workload shapes between rebuilds. It sorts atoms along a space-filling curve into equal-count shards, ensuring identical atom counts and tensor shapes per rank. Peer sets scale as $K^
    
[^100]: 框架即模型：智能体脚手架会改变固定大语言模型对银行行为规则的合规性，且在不同模型间影响不均

    The Harness Is the Model: Agent Scaffolding Changes a Fixed LLM's Compliance with Banking Conduct Rules, Unevenly Across Models

    [https://arxiv.org/abs/2611.06116](https://arxiv.org/abs/2611.06116)

    该研究表明，即使底层大语言模型保持不变，智能体框架（脚手架）的选择也会显著改变模型在银行行为合规任务上的表现，且不同模型受框架的影响程度并不相同。

    

    银行验证的是语言模型，但实际部署的是系统：即处于一个框架中的模型，该框架决定模型能看到什么、可以调用哪些工具，以及由谁来审查其行为。我们在保持模型固定不变的条件下，测试该框架是否会改变对银行行为规则的合规性。四个开源权重模型分别运行五种框架（一次性做出结构化决策的流水线、ReAct式工具循环、策略仅能通过搜索获取的循环、带有较长良性客户记忆的循环，以及带有同模型合规审查员的循环），并在一家模拟零售银行中对156个预注册案例进行测试，该银行的11类规则改写自欧盟消费者信贷、支付、投资者保护、数据保护及反洗钱相关条款。全部3120个两轮对话片段均基于实际执行的工具调用轨迹进行确定性评分，其中违规行为与过度拒绝被分别统计。汇总各模型来看，该框架使正确案例率……

    arXiv:2611.06116v1 Announce Type: cross  Abstract: Banks validate language models, but they deploy systems: a model inside a harness that decides what it sees, which tools it can call and who checks its actions. We test whether that harness changes compliance with banking conduct rules when the model is held fixed. Four open-weight models each ran five harnesses (a pipeline that makes one structured decision, a ReAct-style tool loop, a loop in which the policy is reachable only through search, a loop with a long benign customer memory, and a loop with a same-model compliance reviewer) on 156 pre-registered cases in a simulated retail bank whose 11 rule families paraphrase EU consumer-credit, payments, investor-protection, data-protection and anti-money-laundering provisions. Every one of the 3120 two-turn episodes was scored deterministically from the executed tool trace, with violations and over-refusals counted separately. Pooled across models the harness moved the correct-case rate 
    
[^101]: 超越风险线索：面向网络诈骗视频检测的机制级基准与可验证辩论框架

    Beyond Risk Cues: A Mechanism-Level Benchmark and Verifiable Debate for Scam Video Detection on the Web

    [https://arxiv.org/abs/2611.06110](https://arxiv.org/abs/2611.06110)

    本文提出机制级基准 MechScam（含6,614个视频、证据支持的诈骗解释及由真实证据构建的误导性论辩）与可验证辩论框架 ScamCourt，通过揭示诈骗运作机制并抵御误导性论辩，显著提升了网络诈骗视频检测的可靠性与鲁棒性。

    

    网络诈骗视频对个人和社会构成重大威胁。现有研究主要集中于二元分类，对于诈骗如何运作以及检测决策背后的推理过程所提供的洞察有限。此外，这些研究往往忽视了将诈骗视频与表面相似的非诈骗内容（如广告）区分开来这一挑战，以及检测模型抵御误导性论辩的鲁棒性问题。这些局限性阻碍了可靠的诈骗视频检测。为解决这些局限，我们提出了 MechScam，一个包含 6,614 个视频、覆盖多样化诈骗机制与非诈骗类别的基准。MechScam 提供了由视频证据支持的解释、潜在诈骗元素及其上下文的标注，以及由真实证据构建的误导性论辩。鉴于真实证据仍可能被用来为误导性结论辩护，我们进一步提出了 ScamCourt，一个辩论框架

    arXiv:2611.06110v1 Announce Type: new  Abstract: Online scam videos pose significant threats to individuals and society. Existing studies primarily focus on binary classification, offering limited insights into how scams work and the reasoning behind detection decisions. Moreover, they often overlook the challenge of distinguishing scam videos from superficially similar non-scam content, such as advertisements, and the robustness of detection models against misleading arguments. These limitations hinder reliable scam video detection. To address these limitations, we introduce MechScam, a benchmark comprising 6,614 videos covering diverse scam mechanisms and non-scam categories. MechScam provides explanations supported by video evidence, annotations of potential scam elements and their context, and misleading arguments constructed from genuine evidence. Recognizing that genuine evidence can still be used to justify misleading conclusions, we further propose ScamCourt, a debate framework
    
[^102]: 调和隐私与溯源：面向语言模型水印的平衡之道

    Reconciling Privacy with Provenance for Language Model Watermarks

    [https://arxiv.org/abs/2611.06100](https://arxiv.org/abs/2611.06100)

    提出红绿序列（RGS）本地差分隐私水印，通过将水印目标从追踪个体转向估计总体使用频率，在可证明地保护用户隐私的同时，仍能实现对语言模型使用情况的准确估计。

    

    水印技术在规范语言模型（LM）的使用方面具有巨大潜力，但其部署一直饱受争议，因为许多用户认为对语言模型使用的细粒度追踪侵犯了他们的隐私。这种矛盾削弱了水印的有效性，因为用户会采取改写等规避策略，而这些策略可以去除大部分水印信号。我们证明，如果将水印设计的目标从追踪个体转变为估计总体使用情况，就可以调和溯源与隐私之间的矛盾。我们开发了本地差分隐私水印，可以从可证明的层面限制任何单个文本对检测所能贡献的证据，同时仍能实现对总体频率的准确估计。我们提出了红绿序列水印，这是一种具有隐私保护且计算高效的水印方法，能够在高隐私保护水平下，跨多种模型、样本量和生成场景实现准确的频率估计。

    arXiv:2611.06100v1 Announce Type: cross  Abstract: Watermarking has great potential for regulating the use of language models (LMs), but its deployment has been marred by controversy, as many users argue that fine-grained tracking of LM usage violates their privacy. This tension reduces the effectiveness of watermarks, as users engage in evasion strategies such as paraphrasing that can remove much of the watermark signal. We show that provenance and privacy can be reconciled if we shift the goal of watermark design from tracking individuals to estimating aggregate usage. We develop locally differentially private watermarks which provably limit the evidence that any individual text can contribute to detection, while still enabling accurate estimation of population frequencies. We introduce the Red-Green Sequence (RGS) watermark, a private and computationally efficient watermark that achieves accurate frequency estimation at high privacy levels across various models, sample sizes and gen
    
[^103]: 代码生成中的推理轨迹溢出：显式推理模型的实证研究

    Reasoning-Trace Overflow in Code Generation: An Empirical Study of Explicit Reasoning Models

    [https://arxiv.org/abs/2611.06095](https://arxiv.org/abs/2611.06095)

    该研究提出“推理轨迹溢出”概念，实证发现显式推理模型在较难的代码生成任务上常常将全部输出预算耗尽于推理而无法产出可用代码，而轻量级的推理时控制可以缓解这一问题。

    

    显式推理可以帮助语言模型生成代码，但它也带来一个实际风险：模型可能将过多的输出预算用于推理，以致于永远无法生成可用的代码。我们在相对较小的开源权重推理模型中研究这一行为，并将其定义为“推理轨迹溢出”。我们在七个模型和六个代码基准测试设置下，测量了固定32k-token输出预算中有多少被用于推理、这种推理是否能提升pass@1，以及轻量级的推理时控制手段能否缓解溢出问题。我们的结果表明，推理的价值具有高度的条件性。在较简单的函数级任务上，溢出很少发生；但在更难的代码生成基准测试和特定模型家族中，溢出变得普遍：在若干“模型-基准”组合上，超过一半的生成耗尽了全部32k-token预算却仍在推理。大多数完全溢出与重复性的推理循环相关。

    arXiv:2611.06095v1 Announce Type: new  Abstract: Explicit reasoning can help language models generate code, but it also introduces a practical risk: the model may spend so much of its output budget reasoning that it never produces usable code. We study this behaviour in relatively small open-weight reasoning models, framing it as reasoning-trace overflow. Across seven models and six code benchmark settings, we measure how much of a fixed 32k-token output budget is spent on reasoning, whether that reasoning improves pass@1, and whether lightweight inference-time controls can mitigate overflow. Our results show that the value of reasoning is highly conditional. Overflow is rare on simpler function-level tasks, but becomes common on harder code-generation benchmarks and for particular model families: on several model--benchmark pairs, more than half of generations exhaust the full 32k-token budget while still reasoning. Most complete overflow is associated with repetitive reasoning loops,
    
[^104]: 面向域增量学习的漂移自适应表征保持

    Drift-adaptive Representation Preservation for Domain-Incremental Learning

    [https://arxiv.org/abs/2611.06093](https://arxiv.org/abs/2611.06093)

    该论文提出将表征漂移作为遗忘风险的可测量替代指标，通过漂移加权的巩固损失和原型构建机制，在域增量学习过程中根据实际漂移程度自适应地分配表征保持资源。

    

    预训练视觉模型提供了强大的通用表征，但在域增量学习中，输入分布会依次发生变化，适应新域往往会降低模型在先前已学习域上的性能。我们观察到，遗忘在域和类别之间并非均匀分布，且不能仅用训练顺序来解释。这促使我们对每个过去的域-类别概念进行单独跟踪。我们使用表征漂移——定义为过去概念的当前表征相对于固定锚点的偏差——作为遗忘风险的可测量替代指标。测量到的漂移随后发挥两个互补作用：一种巩固损失为高漂移的域-类别对分配更强的保持力度，而原型构建则降低高漂移域的权重，以减少其对类别参考的影响。因此，表征保持是根据测量到的漂移来分配的，而非基于假设的重要性。为了在……期间获得该信号（摘要内容在此处被截断）

    arXiv:2611.06093v1 Announce Type: cross  Abstract: Pretrained vision models provide strong general-purpose representations, but in Domain-Incremental Learning (DIL), where the input distribution changes sequentially, adapting to new domains often degrades performance on previously learned ones. We observe that forgetting is not uniform across domains and classes and is not explained by training order alone. This motivates us to track each past domain-class concept individually. We use representation drift, defined as the deviation of a past concept's current representation from a fixed anchor, as a measurable surrogate for forgetting risk. The measured drift then serves two complementary roles. A consolidation loss assigns stronger preservation to high-drift pairs, while prototype construction down-weights highly drifted domains to reduce their influence on the class reference. Preservation is thus allocated by measured drift rather than assumed importance. To obtain this signal during
    
[^105]: SysPilot：基于库性能契约的跨库性能自动调优

    SysPilot: Cross-Library Performance Autotuning via Library Performance Contracts

    [https://arxiv.org/abs/2611.06078](https://arxiv.org/abs/2611.06078)

    SysPilot提出“库性能契约”机制，通过规格化语言描述各库的资源需求与可观测信号，实现跨库性能自动调优，有效缓解多个独立库共同运行时的共享资源争用问题。

    

    现代科学计算与机器学习应用由许多独立开发的库组合而成。当这些库共同运行时，它们各自独立管理的运行时可能会过度占用共享资源：使用相同的CPU核心、争夺GPU，或竞争内存带宽。要避免此类资源争用，需要跨库边界进行细致的配置。现有的自动调优工具并不会直接探究库之间共享了哪些资源、哪些运行时信号与干扰相关，因此在寻找缓解争用的最优配置方面效率低下。本文提出了SysPilot，一个基于库性能契约的跨库性能自动调优系统。库性能契约通过LPCL——一种用于描述库资源需求的规格说明语言——来刻画库的资源需求、假设条件、控制项以及可观测性信号。SysPilot综合

    arXiv:2611.06078v1 Announce Type: cross  Abstract: Modern scientific and machine-learning applications compose many independently developed libraries. When these libraries run together, their independently managed runtimes can oversubscribe shared resources: using the same CPU cores, fighting over GPUs, or contending for memory bandwidth. Avoiding such contention requires careful configuration across library boundaries. Existing autotuners do not directly investigate what resources are shared between libraries, which runtime signals correlate with interference. As a result, they are inefficient for finding optimal configurations to mitigate contention.   This paper presents SysPilot, a cross-library performance autotuning system based on Library Performance Contract (LPC). A Library Performance Contract captures a library's resource demands, assumptions, controls, and observability signals in LPCL, a specification language for describing library resource requirements. SysPilot synthesi
    
[^106]: UniCompress：从多LoRA专家到面向边缘智能的单一紧凑模型

    UniCompress: From Multi-LoRA Experts to One Compact Model for Edge Intelligence

    [https://arxiv.org/abs/2611.06075](https://arxiv.org/abs/2611.06075)

    UniCompress提出将多个多任务LoRA专家蒸馏整合为一个无适配器的INT2紧凑模型，通过GPTQ初始化和教师引导的量化恢复，在极低内存占用下保留完整视觉语言模型的多任务能力，适合边缘智能部署。

    

    具身智能与边缘系统日益依赖视觉语言模型（VLM）来支持多样化的技能，然而为每个技能部署独立的任务适配器、同时保留高精度骨干网络，会带来巨大的内存和管理成本。我们提出了UniCompress，一个面向部署的框架，它将多任务专长转化为一个紧凑的、无适配器的单一模型。以整合后的BF16 Qwen3-VL-4B教师模型为起点，UniCompress利用GPTQ初始化和教师引导的量化恢复，构建了一个非对称的INT2学生模型。该方法对视觉编码器和语言骨干网络中的投影权重进行联合量化与恢复，在保留多任务行为的同时，实现对完整视觉语言模型的激进压缩。我们在四个异构的具身任务上评估了UniCompress：姿态识别、家庭安全理解、多模态对话路由以及移动GUI动作预测。

    arXiv:2611.06075v1 Announce Type: new  Abstract: Embodied and edge systems increasingly rely on vision-language models (VLMs) to support heterogeneous skills, yet deploying a separate task adapter for each skill while retaining a high-precision backbone creates substantial memory and management costs. We present UniCompress, a deployment-oriented framework that converts multi-task expertise into one compact, adapter-free model. Starting from a consolidated BF16 Qwen3-VL-4B teacher, UniCompress constructs an asymmetric INT2 student using GPTQ initialization and teacher-guided quantization recovery. The approach jointly quantizes and recovers the projection weights in the vision encoder and language backbone, enabling aggressive compression of the complete vision-language model while preserving its multi-task behavior. We evaluate UniCompress on four heterogeneous embodied tasks: posture recognition, household-safety understanding, multimodal dialogue routing, and mobile-GUI action predi
    
[^107]: 哪一个在发声？面向视听分割的双耳发声实例消歧

    Which One Is Sounding? Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation

    [https://arxiv.org/abs/2611.06074](https://arxiv.org/abs/2611.06074)

    该论文提出BiSID-AVS方法，利用双耳音频中耳间差异所编码的声源方向信息作为空间约束，在多个同类视觉候选中准确区分并分割出真正发声的实例，克服了传统方法依赖运动和跨帧关联的局限。

    

    现有的视听分割方法主要预测与声音相关的区域或类别掩码，当多个同类候选实例共存时，往往无法识别真正发声的实例。近期的视听实例分割方法虽然解决了实例级预测问题，但在很大程度上依赖运动信息和跨帧关联，在观测稀疏或运动不显著的情况下可能变得不可靠。为此，我们引入双耳音频，其耳间差异编码了声源方向信息，可为在同类视觉候选中区分发声实例提供空间约束。我们提出了面向视听分割的双耳发声实例消歧方法，该方法逐步将类别级候选响应细化为真正发声实例的掩码。BiSID-AVS 首先通过音频提示定位类别一致的候选区域，然后将双耳线索投影到……

    arXiv:2611.06074v1 Announce Type: new  Abstract: Existing audio-visual segmentation methods mainly predict sound-related regions or category masks and often fail to identify the sounding instance when multiple same-class candidates coexist. Recent audio-visual instance segmentation methods address instance-level prediction but rely largely on motion and cross-frame association, which may become unreliable under sparse observations or inconspicuous motion. To this end, we introduce binaural audio, whose interaural differences encode source-direction information, to provide spatial constraints for distinguishing the sounding instance among same-class visual candidates. We propose Binaural Sounding Instance Disambiguation for Audio-Visual Segmentation (BiSID-AVS), which progressively refines category-level candidate responses into a mask of the true sounding instance. BiSID-AVS first grounds category-consistent candidate regions through audio prompting, then projects binaural cues into an
    
[^108]: TouchéWorld技术报告：面向生成世界中相机—手部统一的在线探索

    Touch\'{e}World tech report : Towards Online Unified Camera--Hand Exploration in Generative Worlds

    [https://arxiv.org/abs/2611.06061](https://arxiv.org/abs/2611.06061)

    TouchéWorld将手势驱动的手部交互统一集成到相机控制的视频世界模型中，通过因果适配与少步蒸馏实现块级在线生成，使用户能够在生成世界中同时进行相机导航与手部细粒度交互探索。

    

    最近的视频世界模型允许用户通过相机导航来探索生成的环境，而连续的手部动作为一种细粒度的动作通道，可用于与探索过程中遇到的人和物体进行交互。我们提出了TouchéWorld，一个将手势驱动交互集成到相机控制视频世界模型中的框架。给定一张初始图像和一个场景描述提示，模型以时间上对齐的相机和手部控制信号为条件生成视频。我们通过标注补全、基于一致性的过滤和时间对齐，从异构的第一人称视频中构建了相机—手部联合训练数据。在预训练的相机控制世界模型基础上，我们通过参数高效适配，将空间层面的手部骨架条件与数值化的手部运动特征相结合。因果适配与少步蒸馏支持块级在线生成，使用户能够在探索过程中（原文摘要在此处截断）……

    arXiv:2611.06061v1 Announce Type: cross  Abstract: Recent video world models allow users to explore generated environments through camera navigation. Continuous hand motion provides a fine-grained action channel for interacting with the people and objects encountered during exploration. We present Touch\'{e}World, a framework that integrates gesture-driven interaction into a camera-controlled video world model. Given an initial image and a scene-description prompt, the model generates video conditioned on temporally aligned camera and hand controls. We construct joint camera--hand training data from heterogeneous egocentric videos through annotation completion, consistency-based filtering, and temporal alignment. Building on a pretrained camera-controlled world model, we combine spatial hand-skeleton conditioning with numerical hand-motion features through parameter-efficient adaptation. Causal adaptation and few-step distillation support chunk-level online generation, allowing users t
    
[^109]: Reglos：一个具备递归自我改进能力的自进化个人记忆智能体

    Reglos: A Self-Evolving Personal Memory Agent with Recursive Self-Improvement

    [https://arxiv.org/abs/2611.06049](https://arxiv.org/abs/2611.06049)

    Reglos是一个无需训练的个人记忆智能体，通过自进化记忆机制和递归自我改进来优化记忆线束，并利用异构模型系列的批评者筛选补丁以防止基准测试作弊，从而验证线束修改在目标失败案例之外同样有效，避免过拟合。

    

    个人智能体借助过去的对话记录来为每位用户量身定制回复。其记忆系统负责存储和更新每个用户所说的内容，而如今编码智能体还能重写将记忆转化为模型上下文的“线束”（harness）。然而，这类补丁存在过拟合的风险，因为仅在针对的失败案例上进行测试的补丁，可能只修复了这些案例而对其他情况毫无帮助。我们提出了Reglos，一个无需训练的个人记忆智能体，它能够检验线束的修改是否在所针对的失败案例之外也带来帮助。其自进化记忆会在每次对话中向每个用户的仅追加式记录中添加内容，新偏好会取代旧偏好，事实也可能被遗忘。其线束通过递归自我改进不断优化：由一个编码智能体根据失败轨迹提出补丁，并由来自另一个模型系列的批评者筛选每个补丁，以防范“基准测试作弊”行为。在线束被冻结后，由固定的评估器决定接受哪些后续补丁。该论文摘要在此处被截断。

    arXiv:2611.06049v1 Announce Type: new  Abstract: Personal agents draw on past conversations to tailor their responses to each user. Their memory systems store and update what each user says, and coding agents now also rewrite the harness that turns this memory into the model's context. Such patches risk overfitting, because one tested only on the failures it targets can fix those cases and help nowhere else. We present Reglos, a training-free personal memory agent that tests whether harness changes help beyond the failures they were written to fix. Its self-evolving memory adds to each user's append-only record with every conversation, as new preferences supersede old ones and facts can be forgotten. Its harness improves through recursive self-improvement, in which a coding agent proposes patches from failure traces and a critic from a second model family screens each one for benchmark gaming. After the harness is frozen, a fixed evaluator decides which further patches to accept. With 
    
[^110]: Confucius4-R2T2：面向真·实时转录的稳定前缀流式语音识别模型

    Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription

    [https://arxiv.org/abs/2611.06043](https://arxiv.org/abs/2611.06043)

    本文提出Confucius4-R2T2，一个基于新概念“最长稳定前缀（LSP）”的稳定前缀流式语音识别模型，通过对Qwen3-ASR-1.7B进行后训练，使其在保持高准确率和低延迟的同时从不修改先前输出，非常适合需要即时行动的机器消费场景。

    

    流式语音识别越来越多地被机器所消费，例如翻译器和语音代理，它们会在每个词元（token）出现时立即对其采取行动。一个稍后被修改的词元会导致已做工作的浪费或错误的行动。在这项工作中，我们提出了Confucius4-R2T2（Real Real-Time Transcription的缩写，即“真·实时转录”），这是一个稳定前缀的流式ASR模型，它在保持高准确率和低延迟的同时，从不修改其先前的输出。R2T2是通过对Qwen3-ASR-1.7B在1,500小时的中文和英文音频上进行后训练获得的，训练期间冻结了其预训练的音频编码器和投影器。这种适配基于最长稳定前缀（Longest Stable Prefix, LSP）的概念，这是我们在本工作中引入的一个新概念。我们证明了LSP在前缀序下单调，与完整音频的输出一致，并且在等于之前完整输出的标签中是最大的。从LSP标签进行学习，可以将一个全上下文模型转换为稳定前缀的流式模型，而无需任何架构上的改动。

    arXiv:2611.06043v1 Announce Type: cross  Abstract: Streaming speech recognition is increasingly consumed by machines, such as translators and voice agents, that act on each token as it appears. A token that is later revised costs wasted work or a wrong action. In this work, we present Confucius4-R2T2 (short for Real Real-Time Transcription), a stable-prefix streaming ASR model that never modifies its earlier output while maintaining high accuracy and low latency. R2T2 is obtained by post-training Qwen3-ASR-1.7B on 1,500 hours of Chinese and English audio while freezing its pre-trained audio encoder and projector. The adaptation is based on the concept of Longest Stable Prefix (LSP), a new notion that we introduce in this work. We prove that LSP is monotone under the prefix order, consistent with the full-audio output, and maximal among labels that equal a whole earlier output. Learning from LSP labels converts a full-context model into a stable-prefix streaming one without any architec
    
[^111]: 基于软标签的贝叶斯最优公平性-准确性权衡的实用估计

    Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels

    [https://arxiv.org/abs/2611.06027](https://arxiv.org/abs/2611.06027)

    该论文将仅基于软标签的贝叶斯误差估计技术从标准二分类扩展到公平分类等受约束分类问题，实现了对贝叶斯最优公平性-准确性权衡（帕累托前沿）的实用估计。

    

    对于给定的数据分布，任何模型所能达到的最佳预测误差都存在一个根本性的极限。在分类任务中，贝叶斯误差量化了这一极限，它既可以作为训练模型的性能基准，也可以作为检测过拟合的标准。目前最先进的二分类贝叶斯误差估计技术仅使用软标签（即正类概率），而无需输入特征或辅助训练。我们将这些概念扩展到受约束的分类问题，例如公平性。公平分类是受约束或多目标分类的一个重要案例，它要求模型不仅要准确，还要在不同人口群体（如种族和性别）之间产生公平的结果（即相等或接近相等的真阳性率）。这导致了固有的公平性-准确性权衡，即帕累托前沿，用于对各个公平分类器进行相互比较。我们工作的重点是将（原文在此处截断）

    arXiv:2611.06027v1 Announce Type: cross  Abstract: There is a fundamental limit to the best prediction error of any model for a given data distribution. In classification, the Bayes error quantifies this limit and serves as both a benchmark for trained models and a criterion to detect overfitting. The state-of-the-art Bayes error estimation techniques for binary classification use only soft labels (positive class probabilities) and require no input features or auxiliary training. We extend such notions to constrained classification problems, such as fairness. Fair classification is an important case of constrained or multi-objective classification. It requires a model not only to be accurate but also to yield fair outcomes (equal or near-equal true positive rates) across different demographic groups (e.g., race and gender). This leads to an inherent fairness-accuracy tradeoff or a Pareto frontier that is used to compare fair classifiers against each other. The focus of our work is to e
    
[^112]: 拉丁矩形必须有多不平衡？

    How Unbalanced Must a Latin Rectangle Be?

    [https://arxiv.org/abs/2611.06003](https://arxiv.org/abs/2611.06003)

    本文将拉丁矩形的平衡性问题转化为网格中点的 $\ell_1$ 等距问题，通过模拟退火方法将 33 个最小不平衡度上界最多改进 39.4%，并借助计算机辅助分支定界法首次精确证明 3×9 矩形的最小不平衡度为 56。

    

    我们将一个 $k\times n$ 拉丁矩形视为网格 $[n]^k$ 中的 $n$ 个点。每个符号对应一个点，其坐标为该符号所占的列。当这些点在 $\ell_1$ 度量下两两等距时，称该矩形是平衡的，其不平衡度衡量了与这一理想状态的偏离程度。我们从三个方面研究最小不平衡度。在构造方面，我们使用一种模拟退火方法，该方法在行内移动符号，且仅对列冲突进行惩罚。与 Voboril 等人（2025）的参考表相比，该方法在 $8\le n\le 13$ 范围内的 51 个非方形值中改进了 33 个，改进幅度最高达 39.4%，其余值则保持持平。一个计算代价低廉的不变量——行群，证明了这 33 个新矩形中没有一个与群表的行同痕（isotopic），因此早期的列交换局部搜索无法到达其中任何一个。在精确证明方面，计算机辅助的分支定界法证明了 $3\times 9$ 矩形的最小不平衡度恰为 $56$。Díaz 等人曾找到这一数值，但……（摘要原文在此处截断）

    arXiv:2611.06003v1 Announce Type: cross  Abstract: We view a $k\times n$ Latin rectangle as $n$ points in the grid $[n]^k$. Each symbol gives one point, whose coordinates are the columns it occupies. The rectangle is balanced when these points are pairwise equidistant in the $\ell_1$ metric. Its imbalance measures the distance from this ideal. We study the minimum imbalance from three sides. Constructively, we use a simulated annealing that moves symbols within rows and only penalises column clashes. Against the reference table of Voboril et al. (2025), it improves 33 of the 51 non-square values for $8\le n\le 13$, by up to 39.4%, and matches the rest. A cheap invariant, the row group, certifies that none of the 33 new rectangles is isotopic to rows of a group table, so the earlier column-swapping local search cannot reach any of them. Exactly, a computer-assisted branch and bound proves that the minimum imbalance of a $3\times 9$ rectangle is $56$. D\'iaz et al. found this value, but 
    
[^113]: 面向Stable Diffusion的架构内来源归因：一项新任务及其基准数据集与基线方法

    Intra-Architecture Source Attribution for Stable Diffusion: A New Task with Benchmark Dataset and Baseline

    [https://arxiv.org/abs/2611.06000](https://arxiv.org/abs/2611.06000)

    本文提出了新任务IntrAA-SD，旨在区分共享相同架构但经过不同微调的Stable Diffusion模型，并提供了首个包含14万张图像的基准数据集及基线方法。

    

    设计能够将生成图像归因到其源模型的系统，近来已成为人工智能研究的一个重要领域。现有大多数与图像来源归因相关的数据集和系统，主要针对检测源模型的架构特征。与此不同，本文聚焦于Stable Diffusion，提出了一项新颖的任务——IntrAA-SD（面向Stable Diffusion的架构内来源归因），该任务中需要区分的源模型共享相同的架构骨干，仅因微调而彼此不同。所提出的任务与拥有相同架构的微调生成器的兴起相契合，例如Juggernaut XL和RealVis XL，它们都是Stable Diffusion XL（SDXL）的微调版本。我们提供了首个此类包含14万张图像的基准数据集，其中包含由SDXL不同微调版本生成的图像，以及域内和分布外提示词。此外，我们还提供了一个基线框架。

    arXiv:2611.06000v1 Announce Type: cross  Abstract: Designing systems that attribute a generated image to its source model has lately become a major area of AI research. Most existing datasets and systems related to image source attribution are targeted towards the detection of architectural signatures of the source models. Distinct from this, focusing on stable diffusion, we introduce a novel task, IntrAA-SD, Intra-Architecture source Attribution for Stable Diffusion, where the source models to be differentiated share the same architectural backbone and differ only due to fine-tuning. The proposed task aligns with the emergence of fine-tuned generators having the same architecture, like Juggernaut XL and RealVis XL, which are fine-tuned versions of Stable Diffusion XL (SDXL). We provide a first-of-its-kind $140$K-image benchmark dataset containing images generated from different fine-tunes of SDXL, along with in-domain and out-of-distribution prompts. We also provide a baseline framewo
    
[^114]: 从废料到发现：在淘宝信息流推荐中回收高分未曝光候选物品

    From Waste to Discovery: Recycling High-Scored Unexposed Candidates in Taobao Feed Recommendation

    [https://arxiv.org/abs/2611.05997](https://arxiv.org/abs/2611.05997)

    论文提出无需训练的召回阶段系统URED，通过回收工业推荐流水线中通常被丢弃的高分未曝光候选物品并在召回阶段重新注入，在淘宝信息流推荐中高效帮助用户发现新兴趣，避免了传统曝光提升策略因冷启动候选被排序器高估而损害参与度的问题。

    

    帮助用户发现新兴趣对于工业推荐系统的长期健康发展至关重要，然而常见的补救措施——提升探索不足（探索性）内容的曝光——在生产环境中往往适得其反。我们提出了URED（Unexposed Recycling for Efficient Discovery，未曝光回收高效发现），这是一个无需训练的召回阶段系统，已在阿里巴巴的淘宝信息流推荐中为千万级日活跃用户服务超过六个月。该系统基于一个经生产环境验证的洞察：朴素的曝光提升之所以失败，是因为它注入了冷启动、信息贫乏的候选物品，而排序器会系统性地高估这些物品，不仅无法带来发现收益，还会损害用户参与度。那些得分很高但从未曝光的物品（工业流水线通常会丢弃的信号）构成了一个既相关又可被可靠预测的候选池。在30天的时间窗口内，URED聚合这些物品，并在召回阶段通过一个涵盖离线存储、分布式聚合和低延迟服务的轻量级流水线将其重新注入。

    arXiv:2611.05997v1 Announce Type: new  Abstract: Helping users discover new interests is vital to the long-term health of industrial recommender systems, yet the usual remedy, boosting exposure of under-explored (discovery) content, often backfires in production. We report URED (Unexposed Recycling for Efficient Discovery), a training-free recall-stage system serving Taobao feed recommendation at Alibaba for over six months to tens of millions of daily active users. It rests on a production-validated insight: naive boosting fails because it injects cold, information-poor candidates that rankers systematically over-estimate, buying no discovery gain while risking engagement. Items scored highly but never exposed (signals industrial pipelines routinely discard) form a pool that is both relevant and reliably predictable. Over a 30-day horizon URED aggregates them and re-injects at recall time through a lightweight pipeline spanning offline storage, distributed aggregation, and low-latency
    
[^115]: Clips2Story：基于多模态检索嵌入式生成的免训练视频故事板制作与编辑

    Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation

    [https://arxiv.org/abs/2611.05992](https://arxiv.org/abs/2611.05992)

    Clips2Story提出了一种无需训练的LLM流水线，通过多模态描述符检索与关键词相关的视频片段，并自动编排成带有旁白叙述的可编辑故事板和连贯短视频。

    

    视频编辑人员通常需要处理大量碎片化的素材集合，并必须从中选择和组织相关片段以形成连贯的故事。然而，许多现有的视频编辑和摘要方法都假设源视频在时间上是有序的，并旨在保留其原始的叙事和时间结构。在本文中，我们提出了Clips2Story，这是一个无需训练、基于大语言模型（LLM）的流水线，使用户能够从视频片段池中创建指定关键词的短视频。给定一个片段池和用户指定的关键词，我们首先为每个片段提取基于文本的多模态描述符，并检索与关键词最相关的片段。然后，我们将这些候选片段的文本描述符提供给LLM，由其选择并排列这些片段，形成带有连贯旁白叙述的可编辑故事板。最后，我们组装所选片段，并为旁白检索配套的视觉画面，以合成输出视频。为了评估所提出的方法……

    arXiv:2611.05992v1 Announce Type: cross  Abstract: Video editors often work with large collections of fragmented footage and must select and organize relevant clips into a coherent story. However, many existing video editing and summarization methods assume a temporally ordered source video and aim to preserve its original narrative and temporal structure. In this paper, we introduce Clips2Story, a training-free, LLM-based pipeline that enables users to create keyword-specified short videos from a pool of video clips. Given a pool of clips and a user-specified keyword, we first extract text-based multimodal descriptors for each clip and retrieve the clips most relevant to the keyword. We then provide the textual descriptors of these candidate clips to an LLM, which selects and arranges them into an editable storyboard with connective narrations. Finally, we assemble the selected clips and retrieve accompanying visuals for the narrations to compose the output video. To evaluate the prop
    
[^116]: 基于射线追踪的动态信道探测与自回归Wi-Fi信道状态信息预测

    Ray-Tracing-Based Dynamic Channel Sounding With Autoregressive Wi-Fi CSI Prediction

    [https://arxiv.org/abs/2611.05970](https://arxiv.org/abs/2611.05970)

    本文提出一种将自回归CSI预测与SINR约束探测间隔选择相结合的动态信道探测框架，并利用基于Sionna射线追踪的环境建模和带旋转位置编码的Transformer来应对Wi-Fi MIMO系统中设备移动导致的信道老化与预测误差累积问题。

    

    arXiv:2611.05970v1 公告类型：cross。摘要：Wi-Fi多输入多输出（MIMO）系统中的信道探测必须在信道状态信息（CSI）精度与CSI获取开销之间取得平衡。设备的移动会导致信道老化，而自回归CSI预测会引入随探测间隔不断累积的误差。然而，标准化的统计信道模型无法完全捕捉物理人工智能应用中所遇到的特定环境的信道动态，这促使我们采用基于Sionna的射线追踪来评估设备移动与传播几何结构演变情况下的CSI预测。本文开发了一种动态信道探测框架，将自回归CSI预测与信干噪比（SINR）约束的探测间隔选择相结合。该框架利用带有旋转位置编码的Transformer，将设备位置转换为CSI序列，在多个子载波上预测CSI，用于探测间隔之间的波束成形更新。

    arXiv:2611.05970v1 Announce Type: cross  Abstract: Channel sounding in Wi-Fi multiple-input multiple-output (MIMO) systems must balance channel state information (CSI) accuracy against CSI acquisition overhead. Device mobility causes channel aging, while autoregressive CSI prediction introduces errors that accumulate over sounding intervals. However, standardized statistical channel models cannot fully capture the environment-specific channel dynamics encountered in physical artificial intelligence (AI) applications, motivating the use of Sionna-based ray tracing to evaluate CSI prediction under device mobility and evolving propagation geometry. This paper develops a dynamic sounding framework that integrates autoregressive CSI prediction with signal-to-interference-plus-noise ratio (SINR)-constrained interval selection. A Transformer with rotary position embedding that converts device positions into CSI sequences predicts CSI across multiple subcarriers for beamforming updates between
    
[^117]: 智能体时代中图介导的大语言模型系统：推理、知识、检索、执行与协调

    Graph-Mediated Large Language Model Systems in the Agentic Era: Reasoning, Knowledge, Retrieval, Execution, and Coordination

    [https://arxiv.org/abs/2611.05953](https://arxiv.org/abs/2611.05953)

    本综述提出操作层面的“角色—位置”分类法，系统刻画了图在智能体时代大语言模型系统中于推理、知识与证据、检索与记忆、执行及协调五大环节中的功能、计算来源与输出用途。

    

    图在大语言模型（LLM）系统中发挥着日益重要的作用，支撑结构化推理、知识落地、检索与记忆。随着这些系统向智能体时代演进，图还被用于组织执行依赖关系和多智能体通信，将中间观察结果与保留的信息同后续决策联系起来。然而，要理解图的贡献，必须区分每个图操作的功能、负责其计算的组件以及其输出的后续使用方式。本综述提出了一种操作层面的“角色—位置”分类法，涵盖推理、知识与证据、检索与记忆、执行以及协调五个方面。我们比较了图观察和外部计算如何引导决策、记忆表示与更新如何支持后续任务，以及依赖关系如何约束执行与信息传递。这些比较区分了改进……（原文摘要在此处被截断）

    arXiv:2611.05953v1 Announce Type: new  Abstract: Graphs play an increasingly important role in large language model (LLM) systems, supporting structural reasoning, knowledge grounding, retrieval, and memory. As these systems evolve toward the agentic era, graphs also organize execution dependencies and multi-agent communication, linking intermediate observations and retained information to subsequent decisions. However, understanding their contribution requires distinguishing the function of each graph operation, the component responsible for its computation, and the subsequent use of its output. This survey introduces an operation-level role--locus taxonomy for reasoning, knowledge and evidence, retrieval and memory, execution, and coordination. We compare how graph observations and external computations guide decisions, how memory representations and updates support later tasks, and how dependencies constrain execution and information delivery. These comparisons distinguish improveme
    
[^118]: BAM：用于可解释表格预测的二元可加模型

    BAM: Bivariate Additive Models for Interpretable Tabular Prediction

    [https://arxiv.org/abs/2611.05940](https://arxiv.org/abs/2611.05940)

    提出 Bivariate Additive Models（BAM），将 GA²M 中的每个成对形状函数约束为非对称二元高斯的简单参数形式，在保持与标准 GAM 相当精度的同时，使每个交互项都可可视化为单峰等高线图并自动选择特征对，从而实现真正可解释、可审计的表格预测。

    

    具有成对交互作用的广义可加模型（GA²M）被视为玻璃箱模型，因为每个成对形状函数都可以可视化为等高线图。但在实践中，EBM 或 NODE-GAM 等方法学习到的形状函数过于复杂（等高线不规则且局部波动剧烈），以至于从业者无法对其进行有意义的解释，也无法对照领域知识加以检验。这种更强意义上的可审计性需要显著更简单的形状函数。我们认为，只要捕捉到正确的特征对，这样的简化可以达到与标准 GAM 相当的精度。我们提出二元可加模型，这是一类 GA²M，其中每个成对形状函数都被约束为简单的参数形式，具体实例化为非对称二元高斯分布。每个组件都可以可视化为单峰等高线图，并且其特征对在训练过程中自动选择。在六个表格数据集上的实验……

    arXiv:2611.05940v1 Announce Type: cross  Abstract: Generalized Additive Models with pairwise interactions (GA$^2$Ms) are regarded as glass-box models, since each pairwise shape function can be visualized as a contour plot. In practice, the shape functions learned by methods such as EBM or NODE-GAM are so complex (irregular contours and sharp local fluctuations) that a practitioner cannot meaningfully interpret them or check them against domain knowledge. Auditability in this stronger sense requires substantially simpler shape functions. We argue that such simplification can match the accuracy of standard GAMs if the right feature pairs are captured. We propose Bivariate Additive Models (BAMs), a class of GA$^2$Ms in which every pairwise shape function is constrained to a simple parametric form, instantiated as an asymmetric bivariate Gaussian. Each component is visualizable as a unimodal contour map, and its feature pair is selected automatically during training. Experiments on six tab
    
[^119]: 深度与经典的结合：一种CPU高效的自动调制分类决策融合模型

    Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification

    [https://arxiv.org/abs/2611.05937](https://arxiv.org/abs/2611.05937)

    本文提出FuseAMC轻量级决策融合模型，利用CNN在低信噪比下更鲁棒、高阶累积量分类器在高信噪比下更准确的互补特性，通过逻辑回归元组合器融合两者输出，实现了CPU高效的自动调制分类。

    

    自动调制分类（AMC）是认知无线电中频谱感知和非协作信号识别的基础技术，然而其两种标准的CPU可行方法——基于原始同相/正交（I/Q）样本的紧凑卷积神经网络（CNN）和经典的高阶累积量特征分类器——很少在信噪比（SNR）条件下被相互比较。我们证明这两种方法实际上是互补的：CNN在低信噪比下明显更加鲁棒，此时手工设计的统计特征已接近随机猜测水平；而累积量分类器在高信噪比下更加准确，此时高阶统计量能够清晰地区分密集星座。我们利用这种互补性提出了FuseAMC，这是一种轻量级的决策融合模型，它在约19.8万参数的残差CNN和梯度提升树累积量分类器的拼接类后验之上，训练一个逻辑回归元组合器。

    arXiv:2611.05937v1 Announce Type: cross  Abstract: Automatic modulation classification (AMC) underpins spectrum sensing and non-cooperative signal identification in cognitive radio, yet the two standard CPU-feasible approaches to it, a compact convolutional neural network (CNN) on raw in-phase and quadrature (I/Q) samples and a classical higher-order-cumulant feature classifier, are rarely compared against each other conditioned on signal-to-noise ratio (SNR). We show that they are, in fact, complementary: the CNN is markedly more robust at low SNR, where hand-crafted statistics collapse toward chance, while the cumulant classifier is more accurate at high SNR, where higher-order statistics cleanly separate dense constellations. We exploit this complementarity with FuseAMC, a lightweight decision-fusion model that trains a logistic-regression meta-combiner over the concatenated class-posteriors of an approximately 198k-parameter residual CNN and a gradient-boosted-tree cumulant classif
    
[^120]: Split-KV：面向扩散起草器训练的高效上下文并行方法

    Split-KV: Efficient Context Parallelism for Training Diffusion Drafters

    [https://arxiv.org/abs/2611.05926](https://arxiv.org/abs/2611.05926)

    提出Split-KV，一种精确的上下文并行方法，通过在本地保留键/值分片并将注意力通信限制为查询大小的张量，使通信量与上下文长度无关，从而高效训练扩散起草器。

    

    扩散起草器通过并行去噪多个草稿词元来加速投机解码。其训练采用锚点采样构建紧凑的草稿查询，并通过块稀疏掩码在长得多的上下文上执行交叉注意力。我们通过实验证明：在固定查询预算下，接受长度随上下文规模扩大而保持稳定；而增加查询预算所带来的质量收益递减——这两点共同确立了一种查询远短于上下文的实用工作模式。然而，现有的上下文并行方法均无法同时利用这种查询-上下文不对称性，也无法满足锚点采样所导致的非线性、样本依赖的负载均衡需求。我们提出了Split-KV，这是一种精确的上下文并行方法，它将键/值分片保留在本地，并将每层的注意力通信限制为查询大小的张量，从而使通信量与上下文长度无关。其全局查询视图进一步支持了代价感知的最长…（摘要原文在此处截断）

    arXiv:2611.05926v1 Announce Type: new  Abstract: Diffusion drafters accelerate speculative decoding by denoising multiple draft tokens in parallel. Their training uses anchor sampling to construct compact draft queries and performs cross-attention with a block-sparse mask over a much longer context. We empirically show that acceptance length remains stable as context scales with a fixed query budget, while increasing the query budget yields diminishing quality gains, jointly establishing a practical regime with queries far shorter than the context. No existing context parallelism (CP) method jointly exploits this query--context asymmetry and meets the nonlinear, sample-dependent load-balancing requirement created by anchor sampling. We present Split-KV, an exact CP method that keeps key/value shards local and limits per-layer attention communication to query-sized tensors, making this communication independent of context length. Its global query view further enables a cost-aware longes
    
[^121]: 推理之路：读取与重塑MoE模型中的测试时计算

    Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models

    [https://arxiv.org/abs/2611.05919](https://arxiv.org/abs/2611.05919)

    该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。

    

    推理模型会将测试时计算分配给不同的行为，例如规划、推导和验证。然而，模型经常“过度思考”，在找到正确答案后很长时间仍继续验证和重新推导。这不仅浪费了昂贵的计算资源，甚至可能降低性能。现有的干预方法，如token预算或引导，主要调节推理时长，而非针对特定行为。在这项工作中，我们证明了混合专家模型的router提供了一种天然的、无需训练的行为控制接口。我们首先将推理轨迹切分为句子级片段，并用六种推理行为（规划、执行、复述、事实、反思和结论）之一对每个片段进行标注，使每个token都携带其所属片段的行为标签。作用于router logits上的线性探针预测该标签的效果，几乎与作用于残差流（即模型完整隐藏状态这一标准目标）上的探针相当。

    arXiv:2611.05919v1 Announce Type: new  Abstract: Reasoning models allocate test-time compute across distinct behaviors such as planning, deriving, and verifying. However, models frequently "overthink," continuing to verify and re-derive long after finding a correct answer. This wastes expensive compute and can even degrade performance. Existing interventions, such as token budgets or steering, primarily regulate reasoning duration rather than targeting specific behaviors. In this work, we demonstrate that Mixture-of-Experts (MoE) routers provide a natural, training-free interface for behavioral control. We first split reasoning traces into sentence-level segments and label each with one of six reasoning behaviors (planning, execution, restatement, fact, reflection, and conclusion), so that every token carries the behavior of its segment. Linear probes on router logits predict this label nearly as well as probes on the residual stream, the model's full hidden state and the standard targ
    
[^122]: 面向Salesforce并购整合的自主ETL管道：参考框架、Jitterbit映射与可复现仿真研究

    Autonomous ETL Pipelines for Salesforce M&A Consolidation: A Reference Framework, Jitterbit Mapping, and Reproducible Simulation Study

    [https://arxiv.org/abs/2611.05918](https://arxiv.org/abs/2611.05918)

    本文提出了一个面向Salesforce并购场景的自主ETL参考框架，通过分区提取、声明式映射、质量门控、确定性及分块模糊匹配的主数据管理、受限加载和熔断器保护的变化数据捕获，解决了双源系统并行期间的数据去重、记录关系保留与持续同步难题，并将全流程映射至Jitterbit Integration Studio构件。

    

    各自运行Salesforce的合并公司必须在两个源系统仍持续使用的情况下，将客户数据整合到单一组织中。这需要协调不同的数据模式与选项列表、解决组织之间及组织内部的重复客户问题、保留记录之间的关系，并在系统切换前保持目标数据的实时更新。本文针对该任务提出了一种自主式提取-转换-加载（ETL）管道的参考框架，它结合了分区批量提取、声明式映射模板、基于规则与统计的质量门控、采用确定性匹配与分块模糊匹配的主数据管理、具备逐条记录错误捕获的受限加载，以及由数据健康熔断器保护的变化数据捕获（CDC）。框架的每个阶段均映射到Jitterbit Integration Studio的构件。该框架被实现为开放参考原型，并在合成的双组织场景上进行了评估……（摘要原文在此处截断）

    arXiv:2611.05918v1 Announce Type: cross  Abstract: Merging companies that each run Salesforce must consolidate their customer data into a single organization while both source systems remain in use. This requires reconciling different schemas and picklists, resolving duplicate customers across and within organizations, preserving relationships between records, and keeping the target current until cutover. This article proposes a reference framework for an autonomous extract-transform-load pipeline for this task. It combines partitioned bulk extraction, declarative mapping templates, rule-based and statistical quality gates, master data management with deterministic and blocked fuzzy matching, constrained loading with per-record error capture, and change data capture guarded by a data-health circuit breaker. Each stage is mapped to Jitterbit Integration Studio constructs. The framework was implemented as an open reference prototype and evaluated on synthetic two-organization scenarios w
    
[^123]: 面向电力系统规划的自主智能体

    Autonomous agents for electricity-system planning

    [https://arxiv.org/abs/2611.05911](https://arxiv.org/abs/2611.05911)

    该论文提出了一个用于电力系统规划的自主智能体框架，能够自动完成从方案提案、数值评估、反馈解读到重新设计和最终方案选择的全流程决策，并自动保留成本最低的可行规划方案。

    

    规划电力系统需要反复进行将区域资源、投资、运行和网络约束联系起来的决策。在此，我们提出了一个自主规划框架，该框架将准备好的区域情景依次推进至提案、数值评估、反馈解读、重新设计和最终方案选择环节。智能体提出投资边界、储能运行条件和馈线干预措施，解读数值反馈，并决定如何修改设计以及何时停止。该框架自动执行年度投资与调度优化以及代表性馈线验证，应用固定的验收标准，并保留成本最低的通过验收的候选方案。我们在五个电力系统、四个国家中，使用两种模型部署以及单智能体和多智能体架构，评估了跨越十个系统年情景的四十条碳约束轨迹。全部四十次运行均保留了满足实现……的候选方案（摘要在此处截断）。

    arXiv:2611.05911v1 Announce Type: new  Abstract: Planning electricity systems requires repeated decisions that connect regional resources, investment, operation and network constraints. Here we present an autonomous planning framework that carries prepared regional scenarios through proposal, numerical evaluation, feedback interpretation, redesign and final plan selection. Agents propose investment bounds, storage operating conditions and feeder interventions, interpret numerical feedback, and decide how to revise designs and when to stop. The framework automatically executes annual investment-and-dispatch optimization and representative-feeder verification, applies fixed acceptance criteria and retains the lowest-cost accepted candidate. We evaluate forty carbon-constrained trajectories across ten system-year scenarios in five electricity systems and four countries, using two model deployments and single- and multi-agent architectures. All forty retain candidates satisfying the implem
    
[^124]: 显著性图在致密型乳腺上会失效吗？一项机会归一化、混杂因素感知的乳腺X线摄影解释公平性审计

    Do Saliency Maps Fail Dense Breasts? A Chance-Normalized, Confounder-Aware Audit of Explanation Fairness in Mammography

    [https://arxiv.org/abs/2611.05909](https://arxiv.org/abs/2611.05909)

    该研究首次系统审计了乳腺X线摄影AI分类器显著性解释图在致密型乳腺中的公平性，发现其定位癌症能力显著更差，约一半差距由病例构成（如钙化更多）解释，其余差距则指向模型本身对致密型乳腺病灶依赖不足，并建议报告调整前后按分类器划分的密度差距。

    

    显著性图被作为乳腺X线摄影分类器确实利用了癌症的证据而提出，但其在可能掩盖癌症的致密型乳腺中尚未经过检验。我们测试了显著性图在该场景下定位癌症是否更差及其原因。在1,345个恶性CBIS-DDSM视图上，我们以显著性峰值是否命中病灶，以及一个机会归一化的病灶与背景显著性比值（以比特为单位；+1表示比值翻倍）为指标，评估了三个CNN的Grad-CAM、积分梯度和遮挡图。未调整的差距很大：Grad-CAM峰值命中了44%的致密型乳腺病灶，而非致密型为58%。病例构成（主要是致密型乳腺中更多的钙化）解释了每个差距的约一半或更多。遮挡法仍保持了平均-0.56比特的显著差距。对照实验也将问题指向分类器本身：未调整的差距超过了仅基于图像显著性的图，且模型对致密型乳腺病灶的依赖更低。我们建议按分类器分别报告病例构成调整前后的乳腺密度差距，并进行显著性和病灶依赖性对照。

    arXiv:2611.05909v1 Announce Type: cross  Abstract: Saliency maps, offered as evidence that mammography classifiers use the cancer, are untested in dense breasts, which can hide cancers. We tested whether they locate cancers worse there, and why. On 1,345 malignant CBIS-DDSM views, we scored three CNNs' Grad-CAM, Integrated Gradients and occlusion maps by whether their peak hits the lesion and a chance-normalized lesion-to-background saliency ratio (bits; +1 doubles it). Unadjusted gaps were large: Grad-CAM peaks hit 44% of dense vs. 58% of non-dense lesions. Case mix, chiefly more calcifications in dense breasts, explained about half or more of each gap. Occlusion kept a significant average of -0.56 bit gap. Controls also implicate the classifiers: unadjusted gaps exceeded an image-only conspicuity map's, and models relied less on dense-breast lesions. We recommend reporting density gaps before and after case-mix adjustment, per classifier, with conspicuity and lesion-reliance controls
    
[^125]: 你的试验模拟器知道药物的作用吗？基于7,207项注册试验的干预盲法干预效度检验

    Does Your Trial Simulator Know What the Drug Does? An Intervention-Blind Test of Interventional Validity on 7,207 Registered Trials

    [https://arxiv.org/abs/2611.05907](https://arxiv.org/abs/2611.05907)

    该论文首次将临床试验模拟器的“反事实能力”从主观论断转化为可测量的量化指标，通过构建隐去干预信息的干预盲参照模型，并基于7,207项注册试验构建的9,996个审计对比，将模拟器的干预技能定义为完整模型相对于干预盲参照所消除的误差。

    

    患者世界模型正日益被提出用作临床试验的模拟器，近期的综述将其能力按从时序预测到反事实推演的能力阶梯进行排序。目前阶梯上的位置是通过阅读论文而非实际测量来确定的。我们将反事实这一层级转化为可测量的量。从30,119项已完成并将结果发布至ClinicalTrials.gov的2/3期试验中，我们从7,207项试验构建了9,996个经过审计的试验内对比，并评估模拟器的准确率在多大程度上依赖于知晓所测试的是哪种干预。该测量工具是一个干预盲参照：即同一模型、在相同的试验上训练，但隐去所有能够识别干预的字段。干预技能定义为完整模型相对于该参照所消除的误差。在时间划分上（来自655项试验的878个测试对比），一个基于注册数据训练的梯度提升模拟器消除了五分之一的……

    arXiv:2611.05907v1 Announce Type: cross  Abstract: Patient world models are increasingly proposed as simulators of clinical trials, and recent surveys rank their capabilities on a ladder from temporal prediction to counterfactual rollouts. Position on that ladder is currently assigned by reading papers rather than by measurement. We turn the counterfactual rung into a measured quantity. From 30,119 completed phase 2/3 trials with results posted to ClinicalTrials.gov we construct 9,996 audited within-trial contrasts from 7,207 trials, and we score a simulator on how much of its accuracy depends on knowing which intervention was tested. The instrument is an intervention-blind reference: the same model, trained on the same trials, with every intervention-identifying field withheld. Interventional skill is the error the full model removes relative to that reference. On a temporal split (878 test contrasts from 655 trials) a registry-trained gradient-boosting simulator removes a fifth of th
    
[^126]: CHARM+：基于重合并多流机制的跨硬件注意力

    CHARM+: Cross-Hardware Attention with Re-Merge Multistream Mechanism

    [https://arxiv.org/abs/2611.05906](https://arxiv.org/abs/2611.05906)

    CHARM+通过将顺序通信重组为多个层级以实现层级内并行和计算与通信重叠，并将跨硬件通信与硬件内通信解耦以消除带宽不均衡导致的同步开销，从而显著提升跨硬件架构下分布式注意力的整体性能。

    

    随着序列长度不断增长，基于序列并行的分布式注意力（如Ring-Attention）已成为大语言模型（LLMs）的主流解决方案。然而，固有的通信依赖性限制了计算与通信的重叠，降低了GPU利用率。此外，在跨硬件（NUMA或节点）架构中，由于通信带宽不均衡，频繁的通信同步会限制整体性能。基于这些洞察，我们提出了CHARM+（基于重合并多流机制的跨硬件注意力）。我们将顺序通信重组为多个层级，实现层级内的并行性和计算重叠。此外，我们将跨硬件通信与硬件内通信解耦，消除了由带宽不均衡引起的同步开销。前向和反向算法在多种硬件架构上进行了评估，并持续展现出优越的性能表现。

    arXiv:2611.05906v1 Announce Type: new  Abstract: As sequence lengths continue to grow, distributed attention based on sequence parallel (e.g., Ring-Attention) has become a mainstream solution for large language models (LLMs). Nevertheless, inherent communication dependencies limit computation-communication overlap, reducing GPU utilization. Moreover, in cross-hardware (NUMA or node) architectures, due to imbalanced communication bandwidth, frequent communication synchronization will limit overall performance. Based on these insights, we propose CHARM+, Cross-Hardware Attention with Re-Merge Multistream Mechanism. We reorganize sequential communications into several levels and achieve intra-level parallelism and computation overlap. Furthermore, we decouple cross-hardware communication from intra-hardware communication, eliminating synchronization overhead caused by bandwidth imbalance. Forward and backward algorithms were evaluated on multiple hardware architectures and consistently ou
    
[^127]: EchoKV：通过迭代查询反馈实现自纠错的KV缓存驱逐

    EchoKV: Self-Corrective KV Cache Eviction via Iterative Query Feedback

    [https://arxiv.org/abs/2611.05903](https://arxiv.org/abs/2611.05903)

    EchoKV通过迭代查询反馈机制发现并修复KV缓存驱逐中早期工作区错误导致的决策偏差，在与LAQ相同的计算开销下，在证据分散的长上下文任务上取得显著改进。

    

    KV缓存驱逐不仅仅是一个token排序问题：为生成驱逐查询而保留的工作区同样决定了这些查询的质量。因此，早期的工作区错误可能会扭曲后续的查询观察，并强化一个错误的最终驱逐决策，尤其当证据分散在长上下文中时尤为明显。我们提出了EchoKV，它通过迭代修正来发现并解决这一失效模式。EchoKV利用未来解码位置的提议初始化一个兼容反馈的引导工作区，然后在剩余的前瞻步骤之前，利用早期解码生成的查询来恢复工作区。它使用与LAQ相同数量的工作区前向传递次数，同时仅增加一个用于工作区修复的张量重新评分操作。在LongBench（16个任务，Mistral-7B-Instruct-v0.2和Qwen2.5-7B-Instruct）上，EchoKV在多个需要分散证据的任务上相比LAQ展现出显著提升，包括多……

    arXiv:2611.05903v1 Announce Type: new  Abstract: KV cache eviction is not merely a token-ranking problem: the workspace retained for generating eviction queries also determines the quality of those queries. Consequently, an early workspace error can distort later query observations and reinforce an incorrect final eviction decision, particularly when evidence is dispersed across a long context. We present EchoKV, which discovers and addresses this failure mode with iterative correction. It initializes a feedback-compatible bootstrap workspace using proposals at future decode positions, then uses early decode-generated queries to recover the workspace before the remaining lookahead steps. It uses the same number of workspace forward passes as LAQ while adding a tensor re-scoring operation for workspace repair. On LongBench (16 tasks, Mistral-7B-Instruct-v0.2 and Qwen2.5-7B-Instruct), EchoKV shows notable improvements over LAQ on several tasks requiring dispersed evidence, including mult
    
[^128]: RECAP：面向多模态投机解码的基于请求本地与持久记忆的上下文感知预测回收

    RECAP: Context-Aware Prediction Recycling with Request-Local and Persistent Memories for Multimodal Speculative Decoding

    [https://arxiv.org/abs/2611.05894](https://arxiv.org/abs/2611.05894)

    提出RECAP方法，通过区分请求本地的上下文记忆与跨请求持久的token级记忆、多长度键存储及熵置信度加权融合，实现无需训练的多模态投机解码预测回收，有效解决图像敏感解码状态下的预测混叠问题。

    

    回收目标模型自身的预测为投机解码提供了一种无需训练的候选提案来源。然而，在视觉-语言模型中，现有的回收方法仅通过最后一个token来索引预测，这将不同图像和前缀下计算得到的预测混为一谈；我们发现这种复用在图像敏感的解码状态下最容易失效。更长的键可以减少这种混叠，但匹配到的状态也更少。我们提出了RECAP方法，它将目标预测存储在一、二、三token长度的键下：上下文记忆仅在当前请求中存活，因此隐式地以该请求的图像为条件，而token级记忆则跨请求持久存在以保持覆盖率。在每次查找时，熵-置信度加权根据键的特异性和集中度对检索到的预测进行融合。最佳优先搜索将融合后的状态转移组合成候选树，并由后缀匹配的延续序列加以补充，最后通过一次目标模型的前向传递（对候选进行验证）……

    arXiv:2611.05894v1 Announce Type: new  Abstract: Recycling a target model's own predictions offers a training-free source of proposals for speculative decoding. In vision-language models, however, existing recycling indexes predictions by the last token alone, conflating predictions computed under different images and prefixes; we find that such reuse fails most often at image-sensitive decoding states. Longer keys reduce this aliasing but match fewer states. We introduce \method, which stores target predictions under one-, two-, and three-token keys: contextual memories live only for the current request and are thus implicitly conditioned on its image, while token-level memory persists across requests to retain coverage. At each lookup, entropy-confidence weighting fuses the retrieved predictions according to their key specificity and concentration. Best-first search composes the fused transitions into a candidate tree, supplemented by suffix-matched continuations, and one target pass
    
[^129]: CellCanon：一个用于预测、解释与观测驱动更新的智能体细胞世界模型

    CellCanon: An Agentic Cellular World Model for Prediction, Explanation, and Observation-Driven Updating

    [https://arxiv.org/abs/2611.05890](https://arxiv.org/abs/2611.05890)

    CellCanon是首个智能体细胞世界模型，统一了单细胞表达预测、基于证据的解释和观测驱动的更新，在SciPlex3基准上显著超越基线方法，同时将语言模型token消耗降低超过98%。

    

    生物学发现需要利用分子知识和新的测量数据来推断超出已观测条件的响应。我们提出了CellCanon，这是首个将单细胞表达预测、基于证据的解释以及观测驱动的更新统一起来的智能体细胞世界模型。其预测核心在不同基因的数值读数之间复用条件级别的生物学证据，以指导单细胞表达生成。一个共享的世界状态将预测、获取的测量数据以及来源关联的论断连接起来，从而支持解释与修正。在三个细胞系上的339个留出SciPlex3条件上，CellCanon在差异表达AUPRC和Macro-F1上分别比各指标的最佳基线提高了63.3%和17.3%，方向Macro-F1提高了7.2%。相对于在匹配的药物-细胞队列上进行逐基因推理，实测语言模型token消耗在Predict任务中降低了99.77%，在Explain任务中降低了98.41%。

    arXiv:2611.05890v1 Announce Type: cross  Abstract: Biological discovery requires using molecular knowledge and new measurements to infer responses beyond the conditions already observed. We introduce CellCanon, the first agentic cellular world model to unify single-cell expression prediction, evidence-grounded explanation, and observation-driven updating. Its predictive core reuses condition-level biological evidence across gene-specific numerical readouts to guide single-cell expression generation. A shared world state links predictions, acquired measurements, and source-linked claims, supporting explanation and revision. On 339 held-out SciPlex3 conditions across three cell lines, CellCanon improves differential-expression AUPRC and Macro-F1 over the best baseline for each metric by 63.3% and 17.3%, respectively, and direction Macro-F1 by 7.2%. Relative to gene-wise reasoning on matched drug-cell cohorts, measured language-model token use falls by 99.77% for Predict and 98.41% for Ex
    
[^130]: 相关性忽视与信任下的人工智能辅助分类

    AI-Assisted Classification under Correlation Neglect and Trust

    [https://arxiv.org/abs/2611.05870](https://arxiv.org/abs/2611.05870)

    该论文提出包含相关性忽视与AI信任两个要素的人类决策行为模型，通过闭式相图划分AI辅助决策的互补、损害与自动化区域，并证明更强的AI可能因人类重复计算重叠证据而降低其决策准确性，而适当低水平的信任反而可以补偿相关性忽视。

    

    AI助手何时能改善人类决策，何时又会有所损害？我们认为，答案不仅取决于AI的准确性，还取决于其证据与人类证据的重叠程度，以及人类如何将两者结合起来。决策者通过观察自身信号并接收AI的推荐来做出二元决策。我们构建了一个包含两个基本要素的人类行为模型：她对证据重叠的认知程度以及她对AI的信任程度。我们的分析给出了关于AI能力与证据重叠程度的闭式相图，其中划分出互补、损害与自动化三个区域。忽视证据重叠（即相关性忽视）的决策者，可能因为能力更强的AI反而降低决策准确性，原因是她重复计算了共享的证据。信任会重塑这些区域——信任不足可能补偿相关性忽视，因此使准确性最大化的信任水平取决于具体的部署参数。在实证部分，我们通过配对……识别这两个模型参数。

    arXiv:2611.05870v1 Announce Type: cross  Abstract: When does an AI assistant improve human decision-making, and when does it hurt? We argue that the answer depends not just on AI accuracy but also on how much its evidence overlaps with the human's and how the human combines the two. A worker makes a binary decision by observing her own signal and receiving a recommendation from an AI. We develop a human behavioral model with two primitives: her awareness of the overlap and her trust in the AI. Our analysis gives a closed-form phase diagram over AI capability and overlap, with regions of complementarity, impairment, and automation. A worker who neglects the overlap (correlation neglect) can be made less accurate by a more capable AI, double-counting the shared evidence. Trust reshapes the regions - undertrust may compensate for correlation neglect, so that the trust level maximizing accuracy depends on the deployment parameters. Empirically, we identify the two model parameters by pairi
    
[^131]: 可审计的大语言模型机器遗忘

    Auditable Unlearning for LLMs

    [https://arxiv.org/abs/2611.05866](https://arxiv.org/abs/2611.05866)

    该论文首次将“可审计性”引入为机器遗忘的第四个评估维度，发现现有十一种方法无一能同时满足全部四个维度，并提出基于 GapWiden 的两阶段方法来扩大遗忘集与保留集的审计统计量差异，使删除操作可被验证。

    

    机器遗忘旨在当用户提出删除请求后，将用户数据从已训练的模型中移除。现有方法主要从三个维度进行评估——遗忘效果如何、保留了多少模型能力，以及对成员推理攻击的鲁棒性——但并未评估服务提供者能否向用户提供删除确已发生的证据。为填补这一空白，我们引入可审计性作为第四个评估维度，定义为在提供者所计算的统计量下，被请求删除的数据（遗忘集）与保留数据（保留集）之间的分离程度。我们通过对 TOFU、MUSE 和 RWKU 数据集上十一种遗忘方法的评估发现，没有任何一种方法能同时满足全部四个维度，不同方法在不同维度上各有欠缺。为解决这一问题，我们提出了一种可叠加于任何遗忘方法之上的两阶段方法：第一阶段我们提出 GapWiden，利用铰链损失来增大遗忘集与保留集审计统计量之间的差异……

    arXiv:2611.05866v1 Announce Type: new  Abstract: Machine unlearning aims to remove a user's data from a trained model after a deletion request. Existing methods are majorly evaluated on three axes - how well they forget, how much capability they retain, and how robust they are to membership inference attacks, but not on whether the provider can offer the user any evidence that the deletion occurred. To fill this gap, we introduce auditability as a fourth axis, defined as the separation between requested (forget set) and retained (retain set) data under a statistic the provider computes. We show that across eleven unlearning methods on TOFU, MUSE and RWKU no method satisfies all four axes, with different methods falling short on different axes. To address this, we propose a two stage method that can be used on top of any unlearning method, in the the first stage we propose ${GapWiden}$ which uses a hinge loss to increase the difference between audit statistics of forget set and retain s
    
[^132]: 面向城市规划的生成式人工智能：利用DualControlNet合成土地利用地图与卫星影像

    Generative AI for urban planning: synthesizing land-use maps with satellite imagery via DualControlNet

    [https://arxiv.org/abs/2611.05859](https://arxiv.org/abs/2611.05859)

    本文提出DualControlNet，一种能根据文本提示联合生成高分辨率土地利用地图与卫星影像的双控制生成式模型，相比单模态基线将土地利用多样性一致性提升62.8%、建筑覆盖率一致性提升16.3%。

    

    arXiv:2611.05859v1 公告类型：cross 摘要：各类土地利用模型可用于刻画土地利用模式并支持情景规划。然而，现有模型大多是预测性的，仅基于数值数据预测土地利用转变，无法通过简单的文本提示合成新的土地利用地图。与此同时，城市系统中的生成式人工智能仍是单模态的、可控性较弱，且多以通用图像质量而非领域特定指标进行评估。为填补这些空白，我们提出了DualControlNet，一种生成式土地利用模型，可联合合成高分辨率土地利用地图与卫星影像，从而应对多模态、可控性和可验证性的挑战。该模型在来自奥兰多和芝加哥的超过40,000对瓦片影像上训练，并采用土地利用、空间一致性和图像质量等指标进行评估。相对于单模态基线，DualControlNet将土地利用多样性一致性提升了62.8%，建筑覆盖率一致性提升了16.3%，更……

    arXiv:2611.05859v1 Announce Type: cross  Abstract: Various land-use models characterize land-use patterns and facilitate scenario planning. However, existing models are largely predictive, forecasting transitions based on numeric data without synthesizing new land-use maps from simple text prompts. Meanwhile, generative AI in urban systems remains unimodal, weakly controllable, and evaluated mostly by generic image quality rather than domain-specific metrics. To address these gaps, we introduce DualControlNet, a generative land-use model that jointly synthesizes high-resolution land-use maps and satellite imagery, addressing multimodality, controllability, and verification challenges. Trained on over 40,000 paired tiles from Orlando and Chicago, DualControlNet is evaluated using land-use, spatial consistency, and image quality metrics. Relative to unimodal baselines, DualControlNet improves land-use diversity consistency by 62.8% and building coverage ratio consistency by 16.3%, more t
    
[^133]: 风险控制认证的是错误率而非错误幅度：对选择性胎儿生物测量的审计

    Risk Control Certifies Rates, Not Magnitudes: An Audit of Selective Fetal Biometry

    [https://arxiv.org/abs/2611.05851](https://arxiv.org/abs/2611.05851)

    该论文审计了选择性胎儿头围测量流程，揭示风险控制仅能认证严重错误的发生率而非错误幅度——即使满足认证风险界限，系统仍可输出高达53毫米的巨大误差，且该失败被0.94的高完整度评分完全掩盖。

    

    产前超声生物测量必须返回头围（HC）数值，并在图像无法支持该测量时拒绝给出结果。选择性预测以覆盖范围换取错误率，但其有限样本保证仅约束了严重错误的发生率，而非错误的幅度。我们在两个分别预先注册的、对999张标注的HC18图像进行的四路划分上，审计了一个基于工具的生物测量流程，并在每次划分上对分割器进行三次重训练。通过精确的Clopper-Pearson风险控制来校准轮廓完整度阈值，所得的认证证书与手工设定的底线并无区别，但该流程仍然输出了一个271.3毫米的误差：一个几何上完整的小圆环被拟合到了真实头围为324.6毫米的图像上。被接受的候选结果完整度得分高达0.94，因此该评分无法察觉此次失败，而在约150个答案中的一个离群值也使认证的风险界限（α = 0.10）保持完好。第二个未经改动的划分在不同的（文本在此处被截断）

    arXiv:2611.05851v1 Announce Type: new  Abstract: Prenatal ultrasound biometry must return a head circumference (HC) and refuse when the image cannot support one. Selective prediction trades coverage against error, but its finite-sample guarantees bound the rate of gross errors, not their magnitude. We audit a tool-grounded biometry pipeline on two separately pre-registered four-way splits of the 999 labelled HC18 images, retraining the segmenter three times on each. Calibrating the contour-completeness threshold by exact Clopper-Pearson risk control returns a certificate indistinguishable from a hand-set floor, yet the pipeline still emits a 271.3 mm error: a small, geometrically complete ring fitted to an image whose true HC is 324.6 mm. The accepted candidate scores 0.94 completeness, so the score cannot see the failure, and one outlier among $\approx 150$ answers leaves the certified risk bound ($\alpha = 0.10$) intact. A second, untouched split reproduces the mechanism on a differe
    
[^134]: PEARL：面向部分偏好漂移下选择性遗忘的预测误差自适应正则化学习

    PEARL: Prediction Error-Adaptive Regularized Learning for Selective Forgetting under Partial Preference Drift

    [https://arxiv.org/abs/2611.05842](https://arxiv.org/abs/2611.05842)

    PEARL提出了一种预测误差自适应的正则化学习方法，通过仅在预测出错的特征方向上反转Gram矩阵更新来实现选择性遗忘，从而有效应对部分偏好漂移，克服了传统方法均匀遗忘和固定探索率的局限。

    

    用户偏好会随着需求、情境和品味的演变而发生漂移。这对个性化推荐、大语言模型响应生成以及通过RLHF训练的奖励模型至关重要，因为这些系统可能会随偏好漂移而变得失准。现有的非平稳赌博机方法在所有维度上均匀遗忘，但真实的偏好漂移通常是部分的：某些维度已经过时，而其他维度仍然有效。神经上下文赌博机通过Gram矩阵单调地累积置信度，缺乏将不确定性引导至预测出错方向的机制。我们提出了PEARL（预测误差自适应正则化学习），它在预测出现严重偏差的特征方向上反转Gram矩阵的更新，同时在其他方向上保持置信度不变。我们证明了：（i）任何非自适应策略在偏好漂移后都会遭受线性遗憾；（ii）没有固定的探索率能够在平稳和非平稳两种情况下都实现低遗憾。

    arXiv:2611.05842v1 Announce Type: cross  Abstract: User preferences shift as needs, context, and tastes evolve. This matters for personalized recommendation, LLM response generation, and RLHF-trained reward models, which can become miscalibrated as preferences drift. Existing non-stationary bandit methods forget uniformly across all dimensions, but real preference shifts are typically partial: some dimensions go stale while others remain valid. Neural contextual bandits accumulate confidence monotonically via the Gram matrix, with no mechanism to direct uncertainty where predictions have gone wrong. We propose PEARL (Prediction Error-Adaptive Regularized Learning), which reverses the Gram update along feature directions where predictions are surprisingly wrong, while leaving confidence intact elsewhere. We prove that (i) any non-adaptive policy suffers linear regret after a preference shift, (ii) no fixed exploration rate achieves low regret in both stationary and non-stationary regime
    
[^135]: 愿景论文：为什么分布式LLM后训练与去中心化对齐如今至关重要

    Vision Paper: Why Distributed LLM Post-Training and Decentralized Alignment Are Now Essential

    [https://arxiv.org/abs/2611.05841](https://arxiv.org/abs/2611.05841)

    本文主张分布式学习应成为LLM后训练的一等范式，论证RLHF/DPO所依赖的偏好数据集中化假设不成立，并基于隐私主权、多元公平和鲁棒可扩展性三大支柱提出去中心化对齐的关键技术挑战与研究议程。

    

    这篇愿景论文认为，分布式学习（包括联邦学习和去中心化学习）应成为LLM后训练中的一等范式，而不仅仅是隐私保护的附加功能。当前的RLHF和DPO等方法建立在一个隐含假设之上：偏好数据应当被集中化。我们认为这一假设在科学上站不住脚，在实践中不可持续，在伦理上也存在问题。随着LLM被部署到医疗、法律、金融和多语言社区，偏好数据来自无法或不愿将数据集中化的人群。我们构建了一个基于三大支柱的论证：(1) 隐私与数据主权，(2) 多元主义与表征公平，(3) 鲁棒性与可扩展性。我们形式化了这一交叉领域特有的关键技术挑战——参考漂移、偏好异质性以及对比损失的差分隐私-效用权衡，提出了具体的算法方向，并勾勒出一份研究议程。

    arXiv:2611.05841v1 Announce Type: new  Abstract: This vision paper argues that distributed learning, including federated and decentralized, should become a first-class paradigm in LLM post-training, not merely a privacy add-on. Current methods like RLHF and DPO rest on an invisible assumption: preference data should be centralized. We contend this assumption is scientifically unjustified, practically unsustainable, and ethically problematic. As LLMs deploy into healthcare, law, finance, and multilingual communities, preference data originates from populations that cannot or will not centralize. We develop a three-pillar argument grounded in (1) privacy and data sovereignty, (2) pluralism and representational equity, and (3) robustness and scalability. We formalize key technical challenges unique to this intersection, reference drift, preference heterogeneity, and the DP-utility trade-off for contrastive losses, and propose concrete algorithmic directions, and sketch a research agenda. 
    
[^136]: RACE：用于免执行机器人策略检查点选择的校准累积评分方法

    RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection

    [https://arxiv.org/abs/2611.05838](https://arxiv.org/abs/2611.05838)

    提出RACE——一种无需真实执行的机器人策略检查点选择指标，通过基于状态条件阈值校准动作偏差并用累积乘积聚合评分，使检查点排序与真实部署性能更加一致。

    

    在训练过程中评估已学习机器人策略的部署性能具有挑战性，因为直接衡量任务成功与否需要对每个候选检查点进行代价高昂的真实世界试验。平均验证损失提供了一种无需真实执行的替代方案，但其对检查点的排序可能与部署性能不一致。我们分析了该标准的两个局限性：它在惩罚动作偏差时并未考虑策略能否从这些偏差中恢复过来；并且即使早期的偏差可能使执行过程偏离参考轨迹，后期的参考预测仍被赋予同等权重。我们提出了RACE，一种新颖的免执行检查点排序指标，它利用一组固定的留出成功轨迹进行评估。RACE根据这些轨迹中相似状态下的动作差异推导出状态条件阈值，并以此校准动作偏差。我们通过运行乘积将这些分数进行聚合，使得较大的偏差能够……（摘要原文在此处截断）

    arXiv:2611.05838v1 Announce Type: cross  Abstract: Assessing the deployment performance of learned robot policies during training is challenging, as directly measuring task success requires costly real-world trials for each candidate checkpoint. Averaged validation loss offers a rollout-free alternative, but its checkpoint rankings can be inconsistent with deployment performance. We analyze two limitations of this criterion: it penalizes action deviations without considering whether the policy can recover from them, and gives later reference predictions equal weight even when earlier deviations may lead execution away from the reference. We propose RACE, a novel rollout-free metric for ranking checkpoints using a fixed set of held-out successful trajectories. RACE calibrates action deviations against state-conditional thresholds derived from action differences at similar states in these trajectories. We aggregate these scores through a running product, so that a substantial deviation r
    
[^137]: 智能制造中的自主性与协调性：递归自主代理框架

    Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework

    [https://arxiv.org/abs/2611.05828](https://arxiv.org/abs/2611.05828)

    本文从复杂适应系统视角揭示了智能制造中个体自主性与集体协调这两种耦合机制，并提出了一种将目标驱动自主代理与递归组织结构相结合的递归自主代理（RAA）框架，用以实现生产系统中的智能决策与组织。

    

    智能生产要求系统能够根据生产目标和运行状况进行决策并组织工作，同时还需要通过反馈进行持续调整。本文从复杂适应系统（CAS）的视角分析了人类生产组织中的智能。该分析识别出两种相互耦合的机制：个体自主性与集体协调。个体自主性使行动者能够利用可用信息形成局部决策；集体协调则将这些决策与共同目标和联合约束保持一致。学习与适应随时间推移不断改进这两种机制。通过一个生产系统模型，本文形式化了局部决策规则与协调机制如何共同形成系统决策。在此基础上，我们提出了递归自主代理（RAA）框架，该框架将目标驱动的自主代理与递归式的组织结构相结合。目标契约定义了局部目标……

    arXiv:2611.05828v1 Announce Type: new  Abstract: Intelligent production requires systems to make decisions and organize work according to production goals and operating conditions. It also requires continuous adjustment through feedback. This paper analyzes intelligence in human production organizations from a complex adaptive systems (CAS) perspective. The analysis identifies two coupled mechanisms: individual autonomy and collective coordination. Individual autonomy enables actors to form local decisions using available information. Collective coordination aligns these decisions with shared goals and joint constraints. Learning and adaptation improve both mechanisms over time. A production-system model formalizes how local decision rules and coordination jointly form system decisions. On this basis, we propose the Recursive Autonomous Agent (RAA) framework, which combines goal-driven autonomous agents with a recursive organizational structure. Goal contracts define local objectives a
    
[^138]: 一种用于自动化安全模式恢复的混合方法

    A Hybrid Approach for Automated Security Pattern Recovery

    [https://arxiv.org/abs/2611.05823](https://arxiv.org/abs/2611.05823)

    本文提出SecPatRec，一种将ReAct风格LLM智能体与静态代码属性图分析相结合的混合方法，实现了从源代码仓库中自动恢复安全模式的角色检测、代码落地与交互序列识别。

    

    安全设计强调将安全融入软件设计与开发过程中。安全模式目录通过为反复出现的安全问题规定可复用的架构解决方案，将这一原则付诸实践。然而，从源代码中恢复安全模式的实现仍然具有挑战性，因为安全模式是抽象的、跨越多个组件，并且允许多种有效的实现方式。我们提出了SecPatRec，这是一种从源代码仓库中自动恢复安全模式的混合方法。我们将安全模式视为一种参考架构，并将其恢复任务构建为一个演绎任务：确定哪些规定角色存在、将这些角色落地到具体代码实体中，并恢复它们的交互序列。SecPatRec将一个配备语义和词汇代码搜索能力的ReAct风格LLM智能体（用于角色检测与落地）与静态代码属性图分析（用于交互恢复）相结合。我们构建了一个由人工标注的（原文在此处截断）

    arXiv:2611.05823v1 Announce Type: cross  Abstract: Security-by-design emphasises integrating security into software design and development. Security pattern catalogues operationalise this principle by prescribing reusable architectural solutions to recurring security problems. However, recovering their implementations from source code remains challenging because security patterns are abstract, span multiple components, and admit many valid implementations. We present SecPatRec, a hybrid approach for automated security pattern recovery from source code repositories. We treat a security pattern as a reference architecture and frame its recovery as a deductive task: determining which prescribed roles are present, grounding them in code entities, and recovering their interaction sequences. SecPatRec pairs a ReAct-style LLM agent equipped with semantic and lexical code search for role detection and grounding with static code property graph analysis for interaction recovery. We curate a huma
    
[^139]: 学习何时人类决策至关重要：面向大语言模型智能体的执行者相对咨询价值

    Learning When Human Decisions Matter: Executor-Relative Consultation Value for LLM Agents

    [https://arxiv.org/abs/2611.05814](https://arxiv.org/abs/2611.05814)

    该论文提出咨询价值蒸馏方法，通过蒙特卡洛执行者模拟离线估计依赖执行者的人类咨询价值并蒸馏为轻量级预测器，使LLM智能体能够在有限人类决策预算下智能地决定何时咨询人类。

    

    大语言模型日益自动化复杂任务，但人类的决策能力并不会随着模型能力的提升而扩展。在许多应用中，人类在提供私有背景信息、权威性或智能体无法获取的判断方面仍然不可或缺。我们提出了人类决策能力分配这一形式化问题：在存在多个咨询机会且预算有限的情况下，大语言模型智能体应该在哪些决策上咨询人类？其核心挑战在于估计咨询价值，而咨询价值依赖于执行者，因为同样的人类输入可能对不同的下游智能体产生不同的收益。我们提出了咨询价值蒸馏方法，该方法利用蒙特卡洛执行者模拟离线估计咨询价值，并将这些价值蒸馏到轻量级预测器中用于预算受限的分配，在部署时无需进行基于模拟的价值估计。在五个批量设置和三个序贯设置以及两个前沿执行器上，CVD 提升了……

    arXiv:2611.05814v1 Announce Type: cross  Abstract: Large language models increasingly automate complex tasks, but human decision capacity does not scale with their capabilities. In many applications, humans remain essential for providing private context, authority, or judgments unavailable to the agent. We formulate human decision capacity allocation: given multiple consultation opportunities and a limited budget, at which decisions should an LLM agent consult a human? The central challenge is estimating consultation value, which is executor-dependent because the same human input may benefit different downstream agents differently. We introduce Consultation Value Distillation (CVD), which estimates consultation values offline using Monte Carlo executor rollouts and distills these values into lightweight predictors for budgeted allocation, requiring no rollout-based value estimation at deployment. Across five batch and three sequential settings with two frontier executors, CVD improves 
    
[^140]: 当文件摄取误导语言模型：测量与缓解Office文档中的分歧

    When File Ingestion Misleads Language Models: Measuring and Mitigating Divergences in Office Documents

    [https://arxiv.org/abs/2611.05806](https://arxiv.org/abs/2611.05806)

    该论文揭示了LLM处理Office文档时的“提取分歧”——即提取器输出与用户所见内容不一致——可被攻击者利用为攻击面，并通过静态与动态分析识别出19个可被利用的OOXML通道。

    

    Office文档如今经常被上传到大型语言模型（LLM）进行处理。然而，LLM并不总是能看到人类所看到的内容，因为它通常读取的是提取器从文档底层的Office Open XML（OOXML）文件中生成的文本。由于实现方式不同，各个提取器在读取哪些OOXML部分以及如何解释这些部分上存在差异，因此其输出可能与Office应用程序所显示的内容不一致。例如，攻击者只需稍微移动幻灯片上文本框的位置，就能改变模型读取它们的顺序，而幻灯片在外观上没有任何变化。我们证明这种“提取分歧”是LLM应用程序的一个攻击面。在本工作中，我们首先通过分析LLM应用程序所依赖的广泛使用的提取器，测量攻击者能够利用的内容。通过结合静态源代码分析与动态分析，我们识别出19个提取器会读取（而用户可能无法察觉）的OOXML通道。

    arXiv:2611.05806v1 Announce Type: cross  Abstract: Office documents are now routinely uploaded to large language models (LLMs) for processing. However, an LLM does not always see what a human sees, because it typically reads the text that an extractor produces from the document's underlying Office Open XML (OOXML) files. Depending on the implementation, extractors differ in which OOXML parts they read and how they interpret them, so their output can differ from what Office applications display. For example, by slightly shifting the positions of text boxes on a slide, an adversary can swap the order in which the model reads them while the slide looks unchanged. We show that this extraction divergence is an attack surface for LLM applications.   In this work, we first measure what an adversary can exploit by analyzing the widely used extractors that LLM applications rely on. By combining static source code analysis with dynamic analysis, we identify 19 OOXML channels where extractors rea
    
[^141]: 在推理时专家停用中匹配训练时的权重规则

    Matching the Training-Time Weight Rule in Inference-Time Expert Deactivation

    [https://arxiv.org/abs/2611.05804](https://arxiv.org/abs/2611.05804)

    该论文提出一个权重和模型，揭示推理时专家停用方法应与训练时的权重规则相匹配——使用原始路由概率训练的模型适合剩余专家重归一化，而使用归一化权重训练的模型则适合未归一化重新路由。

    

    推理时的专家停用会同时改变token路由和专家权重，其对性能的影响取决于这些变化的实现方式。我们在来自四个模型家族的五个指令微调检查点上，针对相同的禁用专家集合，比较了移除、冻结、剩余专家重归一化以及未归一化重新路由这几种方法，并以基础OLMoE检查点作为额外对比。我们开发了一个权重和模型，将专家贡献的损失以及分支规模的变化与训练时使用的权重规则联系起来。在根据基于激活的排名停用一半专家的设置下，对于使用原始路由概率训练的模型，剩余专家重归一化会产生最高的困惑度；而对于使用归一化权重训练的模型，未归一化重新路由会产生最高的困惑度。当保留训练时的规则时，困惑度范围为完好基线的1.4倍至73倍，而（原文在此处截断）

    arXiv:2611.05804v1 Announce Type: cross  Abstract: Expert deactivation at inference time changes both token routing and expert weights, and its effect on performance depends on how these changes are implemented. We compare removal, freezing, remaining-expert renormalization, and unnormalized re-routing on identical disabled sets across five instruction-tuned checkpoints from four model families, with a base OLMoE checkpoint as an additional comparison. We develop a weight-sum model that relates the loss of expert contributions and changes in branch scale to the weight rule used during training. With half of the experts deactivated according to activation-based rankings, remaining-expert renormalization yields the highest perplexity in models trained with raw router probabilities, whereas unnormalized re-routing yields the highest perplexity in models trained with normalized weights. Perplexity ranges from 1.4 to 73 times the intact baseline when the training rule is retained, and from 
    
[^142]: GREP：基于原则的生成式房间评估

    GREP: Grounding Generated Room Evaluation in Principles

    [https://arxiv.org/abs/2611.05802](https://arxiv.org/abs/2611.05802)

    GREP 提出了包含物理有效性、视觉连贯性和人类使用逻辑的三层评估理念，通过自主评估智能体发现场景问题并生成评分与报告，使生成房间的评估与人类判断达到 87% 的一致性。

    

    近期以大语言模型（LLM）为核心的进展能够通过选择和布置物体资产来生成美观的带家具房间。然而，现有评估仍局限于特定视角或预定义的问题，无法反映场景的整体质量。为满足这一迫切需求，我们提出了 GREP。其核心是将关注点从定义具体问题转变为建立一套三层评估理念：基础物理有效性检查场景是否遵守物理约束；视觉连贯性关注物体在尺度、风格和构图上的兼容性；而在最高层级，人类使用逻辑要求物体的布置符合人类的日常习惯。在这些原则的指导下，一个评估智能体能够自主且自适应地从场景上下文中识别问题、收集证据，并生成评分和报告，从而支持针对具体场景的问题发现和可靠判断。实验表明：(1) GREP 与人类评估达成 87% 的一致性……

    arXiv:2611.05802v1 Announce Type: cross  Abstract: Recent LLM-centric advances can generate attractive furnished rooms by selecting and arranging assets. Yet evaluation remains limited to specific perspectives or predefined questions, failing to reflect overall scene quality. To address this pressing need, we introduce GREP. Its core is a focus shift from defining specific questions to establishing a three-tier evaluation philosophy: basic physical validity checks whether a scene obeys physical constraints; visual coherence concerns compatibility in object scale, style, and composition; and, at the highest level, human-use logic requires object arrangements to align with everyday human habits. Guided by these principles, an evaluation agent autonomously and adaptively identifies issues from scene context, gathers evidence, and produces scores and reports. This supports scene-specific issue discovery and reliable judgments. Experiments show that (1) GREP achieves 87% agreement with huma
    
[^143]: 一致性并非理解：面向神经场权重读取器的轨道审计

    Consistency Is Not Understanding: Orbit Audits for Neural Field Weight Readers

    [https://arxiv.org/abs/2611.05798](https://arxiv.org/abs/2611.05798)

    该论文提出“轨道审计”方法——利用保持函数等价的隐藏神经元置换和正 ReLU 缩放变换来检验从神经场权重中读取视觉内容的模型，发现现有读取器在等价变换下准确率大幅下降甚至对相同输入给出不一致答案，证明其表面上的输出一致性并不等于对权重所表示函数的真正理解。

    

    从神经网络权重中识别视觉内容，需要将功能性信息与任意的参数坐标区分开来。我们通过隐藏神经元置换以及保持所表示函数不变的正向 ReLU 缩放来审计这种分离。该研究包含 480 个独立拟合的形状场和 1,000 个 MNIST 场，变换后的副本按其来源分组。在形状识别任务上，一个统计读取器在联合变换下准确率从 \ShapeRawStats% 下降到 \ShapeJointStats%；而一个归一化集合读取器则保持 \ShapeNormSet% 的准确率，且预测完全不变。四个小型冻结语言模型通过下一词元打分获得了 \LMRange% 的原始准确率。在另一项包含 1,920 个评分会话的四配置推理实验中，生成式答案获得了 \FrontRawRange% 的原始准确率，然而对相同来源重新使用完全相同的原始提示时，模型在 \FrontRawRepeatRange% 的样本上给出了相互矛盾的判断。规范化使得等价的提示变得……（摘要原文在此处截断）

    arXiv:2611.05798v1 Announce Type: new  Abstract: Recognizing visual content from neural-network weights requires separating functional information from arbitrary parameter coordinates. We audit this separation with hidden-neuron permutations and positive ReLU rescalings that preserve the represented function. The study contains 480 independently fitted shape fields and 1,000 MNIST fields, with transformed copies grouped by origin. On shape recognition, a statistics reader falls from \ShapeRawStats\% to \ShapeJointStats\% accuracy under joint transformations; a normalized set reader retains \ShapeNormSet\% with unchanged predictions. Four small frozen language models obtain \LMRange\% raw accuracy with next-token scoring. A separate four-configuration reasoning experiment with 1,920 scored sessions obtains \FrontRawRange\% raw accuracy with generated answers, while fresh identical raw prompts disagree on \FrontRawRepeatRange\% of origins. Canonicalization makes equivalent prompts identi
    
[^144]: DARE：面向修复增强的难度分析

    DARE: Difficulty Analysis for Repair Enhancement

    [https://arxiv.org/abs/2611.05797](https://arxiv.org/abs/2611.05797)

    提出DARE框架，利用从基准测试产物中提取的151个静态特征预测编程修复任务的难度（定义为多个智能体的观察失败率），并结合七个特征组的贡献分析，为编程智能体提供任务特定的修复支持。

    

    基于大语言模型（LLM）的编程智能体主要通过解决率来进行评估。然而，这一指标既无法揭示已解决任务的难度，也无法考虑不同基准测试之间任务难度的差异。通过重复运行多个智能体来确定单个任务的难度成本高昂，而仅凭预测的难度分数对智能体需要何种支持提供的指导也十分有限。为了应对这些挑战，我们提出了DARE（Difficulty Analysis for Repair Enhancement，面向修复增强的难度分析），该框架集成了修复任务难度测量、预测、特征组贡献分析以及任务特定的修复支持。我们将连续的任务难度定义为跨多个编程智能体观察到的失败率。DARE难度模型利用从基准测试产物中提取的151个静态特征来预测这一难度。该框架还分析了七个修复任务特征组的贡献，以识别特征……

    arXiv:2611.05797v1 Announce Type: cross  Abstract: LLM-based coding agents are primarily evaluated by their solve rates. However, this metric does not reveal how difficult the solved tasks are or account for differences intask difficulty across benchmarks. Determining the difficulty of individual tasks by repeatedly running multiple agents is costly, while a predicted difficulty score alone provides limited guidance on what support an agent needs. To address these challenges, we propose DARE (Difficulty Analysis for Repair Enhancement), which integrates repair-task difficulty measurement, prediction, feature-group contribution analysis, and task-specific repair support. We define continuous task difficulty as the observed failure rate across multiple coding agents. The DARE difficulty model predicts this difficulty using 151 static features extracted from benchmark artifacts. The framework also analyzes the contributions of seven Repair-Task Feature Groups to identify the characteristi
    
[^145]: 面向大语言模型智能体记忆的FlyHash：实证验证与新应用

    FlyHash for LLM Agent Memory: An Empirical Validation and a New Application

    [https://arxiv.org/abs/2611.05791](https://arxiv.org/abs/2611.05791)

    本文首次将受果蝇嗅觉回路启发的FlyHash稀疏哈希应用于大语言模型智能体的记忆检索，并通过严谨的匹配位协议实验实证验证了其相比SimHash的稳健质量优势。

    

    大语言模型（LLM）智能体日益依赖外部记忆库来在对话之间保留信息，通常通过稠密向量嵌入并结合近似最近邻搜索方法实现。果蝇的嗅觉回路以一种截然不同的机制计算感觉输入的局部敏感哈希——先进行稀疏随机投影，再进行胜者通吃竞争——先前的研究表明，这种被称为FlyHash的方案在多个基准测试上，每比特的邻居结构保持能力优于传统的随机超平面哈希。据我们所知，本文首次将FlyHash应用于LLM智能体记忆检索，并提出了三个实践者真正需要回答的问题：在正确匹配位数的协议下，质量优势是否依然成立（答案是肯定的，在10个随机种子和涵盖两种模态的四个数据集上均稳健成立，并揭示了一个此前未被记录的具体比较结果……

    arXiv:2611.05791v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly rely on external memory stores to retain information across conversations, typically via dense vector embeddings searched with approximate nearest-neighbor methods. The fruit fly's olfactory circuit computes a locality-sensitive hash of sensory input using a strikingly different mechanism - sparse random projection followed by winner-take-all competition - and prior work (Dasgupta et al., 2017) showed this scheme, dubbed FlyHash, preserves neighbor structure better per bit than conventional random-hyperplane hashing (SimHash) on several benchmarks. We apply FlyHash, to our knowledge for the first time, to LLM agent memory retrieval, and ask three questions a practitioner actually needs answered: does the quality advantage hold under a correctly matched-bit protocol (it does, robustly, across 10 seeds and four datasets spanning two modalities, with one specific, previously undocumented compar
    
[^146]: 用于旋转相对论性恒星的无监督神经算子：高保真状态方程推断与大规模脉冲星自旋演化

    Unsupervised Neural Operators for Rotating Relativistic Stars: High-Fidelity Equation of State Inference and Pulsar Spin Evolution at Scale

    [https://arxiv.org/abs/2611.05790](https://arxiv.org/abs/2611.05790)

    该论文开发了物理信息驱动的无监督神经算子，能够高效求解旋转中子星的轴对称广义相对论流体静力平衡方程，从而实现高保真度的状态方程推断以及大规模的脉冲星自旋演化群体合成。

    

    观测到的具有毫秒级周期的旋转中子星群体是探测基础物理学的有力工具。质量-半径测量能够揭示简并核物质中约束不佳的性质，而高保真的恒星群体合成则将中子星形成与双星演化中的未知因素与大型射电计时巡天观测联系起来。然而，由于求解控制旋转中子星内部的轴对称广义相对论流体静力平衡高度耦合偏微分方程的计算成本极高，重复的结构计算难以实现。采用非旋转计算或近似旋转修正可能会使状态方程推断产生偏差，并限制群体合成在追踪前身天体演化历史以及传播状态方程不确定性方面的保真度。在这篇快报中，我们开发了物理信息驱动的神经算子，能够求解轴对称问题……

    arXiv:2611.05790v1 Announce Type: cross  Abstract: Observed populations of rotating neutron stars~(NSs) with millisecond-order periods are powerful probes of fundamental physics. Mass-radius measurements inform poorly constrained properties of degenerate nuclear matter, whereas high-fidelity population synthesis connects unknowns in NS formation and binary evolution to large radio-timing surveys. However, repeated structure calculations are prohibitive due to the computational cost of solving the highly coupled partial differential equations of axisymmetric general relativistic hydrostatic equilibrium that govern rotating NS interiors. Non-rotating calculations or approximate rotational corrections can bias equation of state inference and limit the fidelity of population synthesis in tracing the evolutionary history of progenitors and propagating equation of state uncertainties. In this \textit{Letter}, we develop physics-informed neural operators that can solve the axisymmetric proble
    
[^147]: 大语言模型能在多大程度上模仿个体作者？一项比较评估

    How Well Can Large Language Models Impersonate Individual Authors? A Comparative Evaluation

    [https://arxiv.org/abs/2611.05771](https://arxiv.org/abs/2611.05771)

    本研究比较评估了六种主流大语言模型模仿个体作者写作风格的能力，发现大多数模型模仿效果有限，而表现最好的 Gemini 3 Pro 实际上是靠大量复制训练文本中的句子和短语取得的表面成效。

    

    本研究考察了六种广泛使用的大型语言模型（LLM）的作者身份模仿能力，包括 Gemini 3 Pro、Claude Opus、GPT 5.2、GPT 5.2 Pro、DeepSeek 3.2 和 Grok 4.1 Fast。评估采用精选的短篇产品评论文本，以确保个体写作风格在作者内部具有高度一致性。研究设计了两种类型的上下文提示：一种是指令最少的简单提示，另一种是包含明确文体学指导以辅助模仿的复杂提示。作者模仿能力通过基于余弦距离的“与训练文本距离”指标进行评估，并辅以嵌入分布的可视化分析。结果表明，对于大多数大语言模型而言，作者身份模仿仍然具有挑战性，其中 Gemini 3 Pro 展现出最强的表面模仿性能。然而，随后的重叠分析揭示，Gemini 3 Pro 之所以表现突出，是因为它大量复制并重用了训练文本中的句子和短语。

    arXiv:2611.05771v1 Announce Type: cross  Abstract: This study investigates the authorship impersonation capacity of six widely used large language models (LLMs): Gemini 3 Pro, Claude Opus, GPT 5.2, GPT 5.2 Pro, DeepSeek 3.2, and Grok 4.1 Fast. The evaluation was conducted using short product review texts selected to ensure high within author consistency in individual writing style. Two types of in context prompt were designed: a simple prompt with minimal instruction and a complex prompt incorporating explicit stylometric guidance for mimicry. Authorship mimicry was evaluated using a cosine distance based distance to training metric, supplemented by visual analyses of embedding distributions. The results indicate that authorship impersonation remained challenging for most LLMs, with Gemini 3 Pro showing the strongest apparent mimicry performance. However, subsequent overlap analysis revealed that Gemini 3 Pro extensively copied and reused sentences and phrases from the training texts, 
    
[^148]: BobFlow：基于数据流的人机智能体协作数据科学

    BobFlow: Towards Human-Agent Collaborative Data Science Using Dataflows

    [https://arxiv.org/abs/2611.05761](https://arxiv.org/abs/2611.05761)

    提出 BobFlow 系统，将数据流抽象与 ReAct 范式相结合，使智能体能够迭代构建和执行数据流来完成数据科学任务，同时让人类分析师可以直观地检查、引导和验证智能体的分析。

    

    大语言模型（LLM）的最新进展使人类与智能体协作成为数据科学中一种具有前景的范式。有效的协作需要一个共享的计算抽象，使智能体能够对处理逻辑进行推理，同时让人类分析师能够理解、引导和完善分析。尽管用 Python 或 SQL 编写的脚本提供了精确的控制，但其语法和执行细节对分析师来说可能难以理解和修改。在本文中，我们认为数据流是数据科学中人机智能体协作的一种有效抽象。我们介绍了 BobFlow，一个将数据流抽象与 ReAct 范式相结合的系统。它使智能体能够为数据科学任务迭代地构建和执行数据流，同时允许分析师检查分析过程并提供反馈。我们的用户研究表明，BobFlow 帮助分析师理解、检查和验证智能体生成的分析。

    arXiv:2611.05761v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have established human-agent collaboration as a promising paradigm for data science. Effective collaboration requires a shared computational abstraction that enables agents to reason about processing logic while allowing human analysts to understand, guide, and refine the analysis. Although scripts written in Python or SQL provide precise control, their syntax and execution details can be difficult for analysts to understand and modify. In this paper, we argue that dataflows are an effective abstraction for human-agent collaboration in data science. We introduce BobFlow, a system that combines the dataflow abstraction with the ReAct paradigm. It enables agents to iteratively construct and execute dataflows for data science tasks while allowing analysts to inspect the analysis and provide feedback. Our user study shows that BobFlow helps analysts understand, inspect, and verify agent-gener
    
[^149]: PAYWORTHY：大语言模型能否预测AI智能体将完成哪些自由职业任务？

    PAYWORTHY: Can LLMs Predict Which Freelance Jobs an AI Agent Will Complete?

    [https://arxiv.org/abs/2611.05759](https://arxiv.org/abs/2611.05759)

    该论文提出PAYWORTHY基准，基于1,494条真实自由职业任务发现，大语言模型只有在被告知智能体的能力限制时才能可靠预测哪些任务会被AI智能体成功完成（AUC达0.66-0.86），而若仅将其描述为通用智能体，预测效果则接近随机水平。

    

    在AI智能体接手一份有偿工作之前，需要有人判断它是否可能成功。我们测试了前沿大语言模型能否在任何实际尝试之前做出这一判断。我们提出PAYWORTHY基准，其中包含1,494条真实的自由职业任务帖，这些任务从一个自由职业平台2020年至2026年间发布的1,880万条帖子中采样而来。智能体仅根据任务帖单独尝试完成每项工作（单次运行），并由一个充当客户角色的大语言模型裁判决定是否接受该工作。我们让18个模型预测智能体成功的可能性，并衡量每个模型的预测将被接受的任务排在被拒绝任务之上的能力（AUC指标，其中0.5表示随机排序）。结果表明，答案取决于如何描述该智能体：当告知模型智能体不能做什么时，所有模型都能很好地对任务进行排序（AUC为0.66-0.86）；而当仅告知它是一个具备浏览器和代码执行能力的通用智能体时，同样的模型对同样任务的排序效果几乎与随机排序无异（0.45-0.56）。

    arXiv:2611.05759v1 Announce Type: new  Abstract: Before an AI agent takes on a paid job, someone has to judge whether it is likely to succeed. We test whether frontier language models can make this call before any attempt. We propose PAYWORTHY which contains 1{,}494 real freelance job postings, sampled from 18.8M postings made between 2020 and 2026 on a freelancing platform. An agent attempts each job from the posting alone in a single run, and an LLM judge acting as the client decides whether to accept the work. We ask 18 models to forecast how likely the agent is to succeed and measure how well each model's forecasts rank the accepted jobs above the rejected ones (AUC, where 0.5 is a random ranking). The answer depends on how the agent is described. Told what the agent cannot do, every model ranks the jobs well (AUC 0.66--0.86). Told instead that it is a generic agent with a browser and code execution, the same models rank the same jobs about as well as a random ordering (0.45--0.56)
    
[^150]: 面向共享组织皮质的组合式知识架构

    Toward a Compositional Knowledge Architecture for a Shared Organizational Cortex

    [https://arxiv.org/abs/2611.05753](https://arxiv.org/abs/2611.05753)

    提出由信号、领域、主张、维度四层组成的组合式知识架构，使组织解释连同其依据与立场持久保存，并通过真值维护网络随证据变化动态修订，从而显著提升检索性能。

    

    组织在对话、文档、决策和人际关系中积累证据，但其行动依赖于解释——即由多项观察组合而成、建立在假设之上、并随环境变化而不断修订的判断。现有的检索技术、知识图谱和智能体记忆系统只保存了记录，却丢失了解释。我们提出一个四层架构——信号、领域、主张与维度，其中解释连同其依据与立场被持久化保存；一个真值维护网络随证据变化修订这些立场；经过验证且带有版本管理的维度将解释定位于命名轴线上。我们将这一成果称为共享组织皮质。在一项注册检索实验中，维度词项带来 0.028 的 nDCG@10 提升（p = 0.002）。在系统级对比中，完整检索配置相比仅信号检索将 nDCG@10 提升 0.081，相比图增强的仅信号检索提升 0.053。

    arXiv:2611.05753v1 Announce Type: new  Abstract: Organizations accumulate evidence in conversations, documents, decisions and relationships, but they act on interpretations: judgements assembled from several observations, resting on assumptions, and revised as circumstances change. Existing retrieval, knowledge graphs and agent memory keep the records and lose the interpretation. We propose a four-layer architecture, Signals, Domains, Claims and Dimensions, in which interpretations persist with their justifications and standing, a truth-maintenance network revises that standing as evidence changes, and validated, versioned Dimensions position interpretations on named axes. We call the result a shared organizational cortex. In a registered retrieval experiment, the Dimension term gains 0.028 nDCG@10 (p = 0.002). In system-level comparisons, the full retrieval configuration improves nDCG@10 by 0.081 over Signals-only and 0.053 over graph-augmented Signals-only retrieval (both reported p 
    
[^151]: 面向计算的数据调度：用于AMD GPU上LLM训练的层次化数据流水线

    Scheduling Data to Compute: Hierarchical Data Pipelining for LLM Training on AMD GPUs

    [https://arxiv.org/abs/2611.05751](https://arxiv.org/abs/2611.05751)

    该论文将LLM训练中计算单元的数据等待时间视为核心瓶颈，提出通过编排AMD GPU上用于显式数据移动的硬件特性（数据加载、布局重排与GPU间传输），构建层次化数据流水线，使计算单元持续保持忙碌，从而提升训练效率与利用率。

    

    大语言模型（LLM）已成为近期机器学习突破的核心焦点。随着模型规模呈指数级增长，训练所需的计算量达到了天文数字级别。GPU能够执行海量的算术运算，但其计算单元所需的数据并不总是在计算单元就绪时可用。这些数据可能仍处于慢速内存中，或位于另一个GPU上。当芯片处于等待状态时，昂贵的计算资源便被闲置，而在大规模部署下，即使是利用率的微小下降也会转化为巨大的成本。在本文中，我们将这种等待时间视为主要瓶颈。我们专注于为显式数据移动而设计的硬件特性。这些特性将数据从片外内存加载到芯片上，并在加载过程中将其重排为计算所需的布局，同时在GPU之间传输数据。通过编排这些硬件特性，我们保持了稳定的数据流水线持续流向计算单元，使其保持忙碌。我们提出了内核和框架实现……

    arXiv:2611.05751v1 Announce Type: cross  Abstract: Large language models (LLM) have been the key focus of recent machine learning breakthroughs. As the model size grows exponentially, the computation needed for training is astronomical. A GPU can do a huge amount of arithmetic. However, the data its compute units need is not always available when they are ready. This data may still be in slow memory or on another GPU. While the chip waits, expensive compute sits idle, and at scale even a small drop in utilization translates into substantial cost. In this paper, we treat this waiting time as the primary bottleneck. We focus on hardware features designed for explicit data movement. These features load data from off-chip memory onto the chip, rearrange it into the layout the computation requires as it is loaded, and transfer it between GPUs. By orchestrating them, we keep a steady pipeline of data flowing to the compute units so that they remain busy.   We present kernel and framework imp
    
[^152]: 将专家点评与LLM驱动的精炼相结合：一种迭代式方法用于筛选高质量AI评估数据集

    Pairing Expert Critique with LLM-Driven Refinement: An Iterative Approach for Curating Quality AI Evaluation Datasets

    [https://arxiv.org/abs/2611.05749](https://arxiv.org/abs/2611.05749)

    提出SME-C人机协作工作流程，通过专家轻量级点评结合LLM驱动的分析与精炼，迭代式筛选高质量AI评估数据集与评分标准，并在金融商业简报领域收集411位专业人士的3,853条点评验证了两种LLM精炼策略的有效性。

    

    评估和改进特定领域的LLM回答需要高质量的参考数据和评分标准，然而由专家撰写既缓慢又昂贵，而合成数据集除非以专家判断为基础，否则可能无法捕捉从业者的真实偏好。我们提出了SME-C（领域专家点评，Subject Matter Expert Critique），这是一种人机协作工作流程，通过对候选回答的轻量级点评来捕捉专家知识，并结合LLM驱动的分析与精炼，迭代式地筛选回答数据集和评估评分标准。我们在金融商业简报领域验证了SME-C，在两轮调查中收集了411位金融专业人士针对520个查询给出的3,853条点评。我们评估了两种LLM驱动的精炼策略：一种以专家的书面点评为指导，另一种仅从专家的偏好投票中推断点评内容。在对另外100位专业人士进行的盲测验证调查中，我们发现两种策略……（原文摘要在此处截断）

    arXiv:2611.05749v1 Announce Type: cross  Abstract: Evaluating and improving domain-specific LLM responses requires high-quality reference data and rubrics, yet expert authoring is slow and expensive, and synthetic datasets may fail to capture practitioner preferences unless they are grounded in expert judgment. We present SME-C (Subject Matter Expert Critique), a human-AI workflow that captures expertise through lightweight critiques of candidate responses and complements it with LLM-driven analysis and refinement to iteratively curate response datasets and evaluation rubrics. We validate SME-C in the domain of financial business briefs, collecting 3,853 critiques from 411 finance professionals on 520 queries across two survey rounds. We evaluate two LLM-driven refinement strategies: one guided by the experts' written critiques and one that infers critique solely from their preference votes. In a blind validation survey with 100 additional professionals, we found that both strategies s
    
[^153]: 面向低资源医学自动语音识别的纯文本域自适应：豪萨语的浅融合与密度比重评分

    Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa

    [https://arxiv.org/abs/2611.05744](https://arxiv.org/abs/2611.05744)

    该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。

    

    自动语音识别（ASR）系统常常难以处理专业医学词汇，尤其是在缺乏领域特定语音数据的低资源语言中。我们研究了针对豪萨语医学ASR的纯文本域自适应方法，使用固定的MMS-1B-FL102声学模型以及两种解码时语言模型融合方法：浅融合（SF）和密度比方法（DRA）。基于19,421条豪萨语健康句子训练的4-gram KenLM作为医学目标模型，而由75,372条句子构成的通用领域模型为DRA提供源先验。评估使用了来自尼日利亚北部六个州的10名说话人针对25条经医生审阅的提示语录制的250条录音。在提示语不相交的五折交叉验证下，SF和DRA分别将词错误率（WER）从29.56%降至23.55%和23.33%。两者均显著优于贪婪解码（p < 0.001），而二者之间的差异不显著（p = 0.423）。然而，仅使用通用语言模型的SF……

    arXiv:2611.05744v1 Announce Type: cross  Abstract: Automatic speech recognition (ASR) systems often struggle with specialised medical vocabulary, especially in low-resource languages where domain-specific speech data are scarce. We investigate text-only domain adaptation for Hausa medical ASR using the fixed MMS-1B-FL102 acoustic model and two decoding-time language-model fusion methods: Shallow Fusion (SF) and the Density Ratio Approach (DRA). A 4-gram KenLM trained on 19,421 Hausa health sentences serves as the medical target model, while a 75,372-sentence general-domain model provides the source prior for DRA. Evaluation uses 250 recordings of 25 physician-reviewed prompts from 10 speakers across six northern Nigerian states. Under prompt-disjoint five-fold cross-validation, SF and DRA reduce WER from 29.56% to 23.55% and 23.33%, respectively. Both significantly outperform greedy decoding (p < 0.001), while their difference is not significant (p = 0.423). However, general-LM-only SF
    
[^154]: 学习分层图像块切分策略以加速视觉Transformer

    Learning Hierarchical Patch Splitting Policies for Faster Vision Transformers

    [https://arxiv.org/abs/2611.05742](https://arxiv.org/abs/2611.05742)

    该论文提出SPLIT方法，通过学习依据模型认为有意义的任务相关内容来分配图像token分辨率，替代基于像素统计的启发式方法，在减少30%token数量的同时保持与全分辨率ViT相当的准确率。

    

    视觉Transformer（ViT）的性能瓶颈在于其输入token序列的长度。标准的图像块化通过在整个图像上应用统一的patch大小来固定该长度，而不考虑各区域的内容。以往关于自适应patch大小的工作通过为视觉上简单的区域分配更粗糙的patch来减少token数量，但这类方法依赖于局部熵等含噪声的像素级启发式方法来做出patch化决策。我们认为，这种计算资源的分配应该基于模型对任务相关信息的理解，而不是原始的像素统计。我们提出了SPLIT（Split Policy Learned Image Tokenization，学习切分策略的图像token化方法），它学习根据模型认为有意义的内容来分配token分辨率，从而产生与任务相关内容更好匹配的更短token序列。SPLIT在使用减少30%的token的情况下，达到了全分辨率ViT分类基线的准确率。同样的收益也体现在密集预测任务上。（注：原始摘要在此处被截断）

    arXiv:2611.05742v1 Announce Type: cross  Abstract: Vision Transformers (ViTs) are bottlenecked by the length of their input token sequence. Standard patchification fixes this length with a single patch size applied across the image, regardless of regional content. Prior work on adaptive patch sizing reduces token count by assigning coarser patches to visually simple regions, but relies on noisy pixel-level heuristics such as local entropy to make patchification decisions. We argue that this compute allocation should be grounded in the model's interpretation of task relevant information, not in raw pixel statistics. We introduce Split Policy Learned Image Tokenization which learns to allocate token resolution based on what the model finds meaningful, producing shorter token sequences that are better matched to task-relevant content. SPLIT matches the accuracy of full-resolution ViT classification baselines while using 30% fewer tokens. The same gains hold on dense prediction: SPLIT achi
    
[^155]: PersonaCodeBench：在人格驱动的人机交互会话中评估编码智能体

    PersonaCodeBench: Evaluating Coding Agents in Persona-driven Human-Interactive Sessions

    [https://arxiv.org/abs/2611.05739](https://arxiv.org/abs/2611.05739)

    该论文提出 PersonaCodeBench 基准，通过将用户人格特征转化为多轮编码会话中可验证的义务，评估编码智能体在满足个性化需求方面的能力。

    

    编码智能体必须在交付正确软件的同时尊重用户的专业水平、偏好和工作条件。现有基准测试日益纳入交互环节，但个性化需求的满足仍未得到充分探索。我们提出了 PersonaCodeBench，它在人格驱动的多轮编码会话中将用户特征转化为可验证的义务。我们的人格库涵盖角色、技术能力、交互偏好和实际约束，从而产生基于任务且逐步演化的需求，涵盖实现、权限、测试、解释和交付等方面。在运行时，控制器管理用户模拟器的响应与需求披露，利用在线需求评估提供反馈并推进交互。需求逐步浮现，其内容、披露条件和验收标准在评估前即已固定。完成验证器与个性化验证器分别（摘要在此处截断）

    arXiv:2611.05739v1 Announce Type: new  Abstract: Coding agents must deliver correct software while respecting users' expertise, preferences, and working conditions. Existing benchmarks increasingly incorporate interaction, yet personalized requirement fulfillment remains underexplored. We introduce PersonaCodeBench, which turns user characteristics into verifiable obligations in persona-driven multi-turn coding sessions. Our persona pool spans roles, technical abilities, interaction preferences, and practical constraints, yielding task-grounded evolved requirements for implementation, permissions, testing, explanation, and delivery. At runtime, a controller governs user simulator responses and requirement disclosure, using online requirement evaluation to provide feedback and advance the interaction. Requirements emerge progressively, with their content, disclosure conditions, and acceptance criteria fixed before evaluation. A completion verifier and a personalization verifier separate
    
[^156]: OdorSpaces：三维环境中的视觉-嗅觉具身导航

    OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments

    [https://arxiv.org/abs/2611.05738](https://arxiv.org/abs/2611.05738)

    提出OdorSpaces——一个通过FDS模拟三维气味扩散来增强Habitat-Sim的物理仿真平台与基准，支持气体泄漏定位和早期火灾定位等视觉-嗅觉具身导航任务，其多模态强化学习策略优于多种基线方法。

    

    嗅觉能够提供关于空气传播化学线索的关键信息，但它在具身智能领域仍未得到充分探索，先前的机器嗅觉研究较为有限，且主要聚焦于语义气味感知。我们提出了OdorSpaces，一个基于物理的视觉-嗅觉具身导航仿真平台与基准。OdorSpaces通过火灾动力学模拟器（FDS）模拟的三维气味传输与扩散来增强Habitat-Sim，使智能体能够结合视觉与嗅觉观测在照片级逼真的室内环境中进行导航。该基准包含两个代表性的源头搜索场景：不可见气体泄漏定位与早期火灾定位。我们开发了一个初始的多模态强化学习策略，能够随时间整合视觉观测与嗅觉测量。实验表明，该策略优于多种基线方法，为未来研究建立了强有力的参考。

    arXiv:2611.05738v1 Announce Type: cross  Abstract: Olfaction provides critical information about airborne chemical cues, yet it remains largely underexplored in embodied AI, with limited prior machine olfaction work focusing primarily on semantic odor perception. We introduce OdorSpaces, a physics-based simulation platform and benchmark for visual-olfactory embodied navigation. OdorSpaces augments Habitat-Sim with 3D odor transport and dispersion simulated by Fire Dynamics Simulator (FDS), enabling agents to navigate photorealistic indoor environments using both visual and olfactory observations. The benchmark includes two representative source-seeking scenarios: invisible gas-leak localization and early-stage fire localization. We develop an initial multimodal reinforcement learning policy that integrates visual observations and olfactory measurements over time. Experiments show that the policy outperforms a diverse set of baselines, establishing a strong reference for future research
    
[^157]: 语言模型知道哪些约束属于同一组吗？

    Do Language Models Know Which Constraints Belong Together?

    [https://arxiv.org/abs/2611.05726](https://arxiv.org/abs/2611.05726)

    该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。

    

    Dantzig-Wolfe 重构通过将选定的一组约束族凸化来加强混合整数规划的线性松弛，而每个候选选择都需要付出一次列生成的运行代价。当各约束族只有组合在一起才能改进边界时，这一选择变得困难，因为每次只添加一个约束族的搜索会在线性松弛处停滞。批量生产、网络设计和设施选址中的经典不等式正建立在这类零边际组之上。我们证明，当每次评估最多凸化 k 个约束族中的 s 个时，任何仅观察边界值的方法都需要量级为 k^w 的覆盖次数评估，才能找到包含 w 个约束族的隐藏组合，且该结论在实际的 Dantzig-Wolfe 松弛上同样成立。一个遗漏 e 个成员的预测可通过一次无需知晓 e 的分阶段搜索，将成本降至 Θ(k^e)，因此预测所节省的评估次数直接度量了预测器所掌握的知识。我们研究了两种预测来源……

    arXiv:2611.05726v1 Announce Type: cross  Abstract: Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. The classical inequalities of lot sizing, network design and facility location rest on such zero-marginal groups. We prove that when each evaluation convexifies at most $s$ of $k$ families, any method that observes only bounds needs a covering number of evaluations, of order $k^w$, to find a hidden group of $w$ families, also on actual Dantzig-Wolfe relaxations. A prediction that misses $e$ members reduces the cost to $\Theta(k^e)$, through one staged search that does not need to know $e$, so the evaluations a prediction saves measure what the predictor knows. We study two sources of predictions
    
[^158]: 基于智能眼镜的标签高效面部表情识别自监督预训练方法

    Self-Supervised Pretraining for Label-Efficient Facial Expression Recognition using Smart Glasses

    [https://arxiv.org/abs/2611.05711](https://arxiv.org/abs/2611.05711)

    该论文提出在OCOsense智能眼镜的未标注数据上，利用掩码自编码器对视觉Transformer进行自监督预训练，从而在标签极度稀缺的情况下显著提升面部表情识别性能，仅用5%的标注数据即可超越从零开始训练的全监督模型。

    

    在光学肌电（OMG）智能眼镜（如OCOsense）上进行面部表情识别（FER），可以实现保护隐私、无需摄像头的情感感知，服务于健康与福祉领域。大多数FER系统是全监督的，即它们要么依赖手工设计的专家特征结合经典机器学习方法，要么依赖从标注数据训练的端到端深度模型。这两种方式都依赖大量标注的表情数据，而这些数据收集成本高昂，限制了其在真实世界个性化场景中的部署。我们研究了自监督学习（SSL）作为在这些眼镜上实现标签高效FER的途径。我们在未标注的OCOsense录制数据上，采用掩码自编码器目标对视觉Transformer进行预训练，并在与个体无关的交叉验证设置下，针对五分类面部表情识别任务进行微调。在从5%到100%的标签预算范围内，预训练始终能提升性能；仅使用5%的标签时，预训练模型的性能就超过了从零开始训练的模型。

    arXiv:2611.05711v1 Announce Type: new  Abstract: Facial Expression Recognition (FER) on optomyographic (OMG) smart glasses such as OCOsense enables privacy-preserving, camera-free affect sensing for health and well-being. Most FER systems are fully supervised, i.e., they rely either on hand-crafted expert features with classical machine learning, or on end-to-end deep models trained from labelled data. Both depend on large quantities of labelled expressions, which are costly to collect and limit deployment in real-world, personalized settings. We investigate Self-Supervised Learning (SSL) as a route to label-efficient FER on these glasses. We pretrain a Vision Transformer with a masked-autoencoder objective on unlabelled OCOsense recordings and finetune it for five-class FER under person-independent cross-validation. Across label budgets from 5% to 100%, pretraining consistently improves performance; with only 5% of the labels, the pretrained model outperforms the from-scratch model tr
    
[^159]: 智能体：只在有帮助时才思考

    Agents, Think Only When it Helps

    [https://arxiv.org/abs/2611.05709](https://arxiv.org/abs/2611.05709)

    研究发现LLM智能体蒸馏中思想监督对最终行动表现影响甚微，仅当思想信息未在相应行动中披露时才变得重要，因此应根据信息披露情况灵活选择是否蒸馏思想，以提升效率。

    

    LLM智能体蒸馏通过在通常交织着思想与行动的完整教师轨迹上进行训练，将智能体能力迁移到学生模型中。我们质疑思想监督对智能体最终行动表现究竟有多关键。为回答这一问题，我们在蒸馏过程中独立地扰动教师的思想与行动，以解耦二者的影响。在跨模型规模的广泛采用的智能体基准测试上，改变或丢弃思想对学生性能几乎没有影响。然而我们发现，当训练中思想的信息未在相应行动中被披露时，思想监督变得重要起来。通过识别这类信息不一致的情况，我们的发现挑战了为LLM智能体蒸馏完整教师轨迹的常见做法，并提供了一个新见解：思想监督可以作为一种基于效率的可辨识选择，取决于思想的信息是否被披露。

    arXiv:2611.05709v1 Announce Type: new  Abstract: LLM agent distillation transfers agentic capabilities to student models by training on complete teacher trajectories that typically interleave thoughts and actions. We question how critical thought supervision is for agents' ultimate action performance. To answer this, we disentangle their impacts by independently perturbing teacher thoughts and actions during distillation. On widely adopted agent benchmarks across model scales, altering or discarding thoughts has little effect on student performance. We find, however, that thought supervision becomes important when information about the thought is not disclosed in the corresponding action during training. By identifying such discrepant cases, our findings challenge the common practice of distilling entire teacher trajectories for LLM agents, and offer a new insight that thought supervision can be a discernible choice for efficiency depending on whether the information of thought is disc
    
[^160]: 从类型到殊型：佩特里网、个体殊型哲学与个体化因果效应

    From Type to Token: Petri Nets, Individual Token Philosophy, and Individualized Causal Effects

    [https://arxiv.org/abs/2611.05703](https://arxiv.org/abs/2611.05703)

    该论文提出以个体殊型哲学下的佩特里网作为殊型层面因果结构的自然表示，通过从贝叶斯网络到佩特里网的形式化转换并结合保形预测层，在不依赖分布假设且对处理效应异质性鲁棒的情况下实现个体化因果效应的量化。

    

    结构因果模型和贝叶斯网络是表示因果关系的主流框架。它们本质上是类型层面的形式化方法，因为其估计量——平均处理效应（ATE）和条件平均处理效应（CATE）——是总体分布的性质，而非个体单元的性质。当处理效应表现出无界异质性时，这些估计量便会失效。此时仍然存在的是因果关系的结构性的、殊型层面的核心。我们论证，在个体殊型哲学下的佩特里网是这类殊型层面因果结构的自然表示。我们开发了从贝叶斯网络到佩特里网的形式化转换，为个体化因果性提供了一个两层解释：佩特里网层提供了不依赖分布假设的因果机制的结构表示，而保形预测层则在有限样本覆盖保证下量化因果效应。

    arXiv:2611.05703v1 Announce Type: new  Abstract: Structural Causal Models and Bayesian Networks are the dominant framework for representing causal relations. They are fundamentally type-level formalisms since their estimands, Average Treatment Effects and Conditional Average Treatment Effects, are properties of distributions over populations, not of individual units. When treatment effects exhibit unbounded heterogeneity, these estimands collapse. What persists is the structural, token-level core of causation.   We argue that Petri Nets under the Individual Token Philosophy are the natural representation of these token-level causal structures. We develop a formal translation from Bayesian Networks to Petri Nets, providing a two-layer account of individualized causality. The PN layer provides a structural representation of causal mechanisms that does not rest on distributional assumptions. In turn, a conformal prediction layer quantifies causal effects with finite-sample coverage guaran
    
[^161]: TACT：一种利用AI委员会进行定性分析的实用方法

    TACT: A Pragmatic Approach to Qualitative Analysis with an AI Council

    [https://arxiv.org/abs/2611.05702](https://arxiv.org/abs/2611.05702)

    该论文提出了TACT方法，通过由多个AI模型组成的可追溯委员会进行大规模主题分析，使软件团队在利用AI进行定性数据分析的同时，仍能保持对解释和判断的掌控权，并通过对抗性批评和不确定性标注进一步保证分析质量。

    

    定性数据能够告诉软件团队他们的利益相关者需要什么，然而严格分析成千上万条定性回复需要耗费专家时间，很少有团队能够负担或证明其合理性。因此，团队常常面临一个虚假的两难选择：要么不分析这些回复，要么将它们交给大语言模型并接受其解释和判断。在本文中，我们采用一种实用的AI辅助定性分析方法，提出了TACT（可追溯AI委员会主题分析法），这是一种能够在保持团队掌控解释和判断权的同时实现大规模定性分析的程序。TACT有两种变体：TACT-Lite，即核心AI委员会程序；TACT-Full，它在核心程序基础上增加了对抗者（来自其他模型家族、对每个模型的输出进行批评的模型）以及一个用于识别高不确定性编码分配的决策模型。我们在一个由他人进行定性编码的包含256个Stack Overflow线程的独立数据集上评估了TACT。

    arXiv:2611.05702v1 Announce Type: cross  Abstract: Qualitative data can tell software teams what their stakeholders need, yet rigorously analyzing thousands of qualitative responses takes expert time that few teams can afford or justify. Teams therefore often face a false choice: leave the responses unanalyzed, or hand them to a large language model and accept its interpretation and judgment. In this paper, we take a pragmatic approach to AI-assisted qualitative analysis and present TACT (Traceable AI Council for Thematic analysis), a procedure that enables such analysis at scale while keeping teams in control of interpretation and judgment. TACT comes in two variants: TACT-Lite, the core AI council procedure, and TACT-Full, which adds adversaries (models from other model families that critique each model's output) and a decision model that surfaces high-uncertainty code assignments. We evaluated TACT on an independent dataset of 256 Stack Overflow threads qualitatively coded by other 
    
[^162]: 人工智能体如何根据视觉数据进行决策？

    How Do Artificial Agents Make Decisions in Light of Visual Data?

    [https://arxiv.org/abs/2611.05671](https://arxiv.org/abs/2611.05671)

    本研究通过两项互补实验评估了基于视觉-语言模型的人工智能体如何感知可视化中编码的不确定性并基于图像数据做出决策，发现 GPT-4o 在感知准确性方面表现最佳。

    

    视觉-语言模型（VLMs）在利用视觉数据进行推理方面已展现出显著进展，这一点尤为重要，因为人类的大量知识在科学文章和其他来源中是以视觉形式编码的。然而，在理解由视觉-语言模型驱动的人工智能体如何在存在视觉数据的情况下感知并应对不确定性下的决策这一问题上，仍然存在空白。本研究通过评估和分析自主决策过程来探索这一空白。基于先前在可视化素养和决策理论方面的工作，我们开展了两项互补研究：一项研究考察智能体如何感知和量化可视化中编码的不确定性，另一项研究调查人工智能体如何基于图像形式的视觉数据做出决策。我们分析了不同模型和不同可视化类型下的感知准确性与决策表现。我们的结果表明，GPT-4o 表现出最高的准确

    arXiv:2611.05671v1 Announce Type: new  Abstract: Vision-language models (VLMs) have demonstrated significant advances in reasoning with visual data, which is particularly important given that much of human knowledge is encoded visually in scientific articles and other sources. However, a gap remains in understanding how artificial agents, powered by VLMs, perceive and navigate decision-making under uncertainty in the presence of visual data. This study explores this gap by evaluating and analyzing autonomous decision-making processes. Building upon prior work in visualization literacy and decision theory, we conducted two complementary studies: one examining how agents perceive and quantify uncertainty encoded in visualizations, and another investigating how artificial agents make decisions based on visual data in the form of images. We analyzed perception accuracy and decision performance across models and visualization types. Our results show that GPT-4o demonstrates the highest accu
    
[^163]: 使用反馈模糊认知图混合体组合相关专家知识

    Combining Relevant Expert Knowledge Using Mixtures of Feedback Fuzzy Cognitive Maps

    [https://arxiv.org/abs/2611.05664](https://arxiv.org/abs/2611.05664)

    该论文将法律证据理论中的相关性定义扩展应用于神经符号系统，通过将专家的反馈模糊认知图混合为共同知识网络，并借助新的类贝叶斯定理对混合知识进行解混合，从而仅组合相关专家答案以提升系统的准确性和可扩展性。

    

    神经符号系统应当只组合来自专家智能体或其他知识源的相关答案。随着知识网络的不断增长并连接越来越多的知识源，针对给定输入查询仅使用相关专家对于准确性和可扩展性变得愈发重要。我们从证据法理论出发扩展了相关性的标准定义：在给定可用知识的条件下，当且仅当一个答案对于手头问题既具有证明力又具有实质性时，该答案才是相关的。我们将这一相关性方案应用于混合真实或人工专家的反馈模糊认知图（FCMs）。各专家FCM的有向因果图混合或组合成一个共同的知识网络。我们将专家的均衡极限环预测或答案与混合FCM的答案进行比较，然后通过它们之间的差异或变化来估计答案的证明力状态。一个新的类贝叶斯定理将知识混合物解混合为……（原文摘要在此处截断）

    arXiv:2611.05664v1 Announce Type: new  Abstract: A neuro-symbolic system should combine only relevant answers from expert agents or other knowledge sources. Using only relevant experts for a given input query becomes ever more important for accuracy and scalability as the knowledge network grows and connects ever more knowledge sources. We extend the standard definition of relevance from the legal theory of evidence: An answer is relevant if and only if it is both probative and material for the matter at hand given the knowledge available. We apply this relevancy scheme to mixing the feedback fuzzy cognitive maps (FCMs) of real or artificial experts. The expert FCM directed causal graphs mix or combine into a common knowledge network. We compare the equilibrium limit-cycle predictions or answers of the experts to the answers of the mixture FCM and then estimate the probative status of answers by their difference or variation. A new Bayes-like theorem de-mixes the knowledge mixture into
    
[^164]: 有据可依但非免疫：“智能体即评审”中的确认偏误

    Grounded but Not Immune: Confirmation Bias in Agent-as-a-judge

    [https://arxiv.org/abs/2611.05662](https://arxiv.org/abs/2611.05662)

    本研究揭示了“智能体即评审”即使自行收集了决定性证据，仍会因一条无证据价值的虚假权威备注而引发确认偏误，从而推翻多达40%的正确判定。

    

    评估开放式智能体任务越来越多地交给“智能体即评审”：这是一种配备工具并获得智能体完整轨迹的评估者，以便其对照环境验证结果。其前提是，一个自行收集证据的评审者应当被证据所锚定。我们表明，这种证据基础并不能使评审者对确认偏误免疫：一条虚假的预筛备注——尽管被冠以权威来源、且对正确判定没有任何证据价值——竟能在四个评审者和五个数据集上，推翻使用工具的评审者多达40%的正确判定，即使评审者收集了决定性证据。这种判定转变由该提示线索的特定属性所驱动。我们将这一线索视为确认偏误的受控诊断手段，而非对抗性攻击。我们的核心贡献是解释了确认偏误如何改变评审者在轨迹中的行为，我们对评审者行为的变化进行了分类，并将变化归因于轨迹中的特定环节。

    arXiv:2611.05662v1 Announce Type: new  Abstract: Evaluating open-ended agent tasks increasingly falls to an Agent-as-a-Judge: an evaluator handed tools and the agent's full trajectory so it can verify a result against the environment. The premise is that a judge that gathers its own evidence should be grounded by it. We show that this grounding does not immunize the judge against confirmation bias: a single false pre-screening note, attributed to an authority and carrying no evidential bearing on the correct verdict, overturns up to 40% of a tool-using judge's correct verdicts across four judges and five datasets, even though the judge collects decisive evidence. The shift is driven by specific properties of the cue. We treat this cue as a controlled diagnostic of confirmation bias rather than an adversarial attack. Our central contribution is to explain how confirmation bias changes the judge's behavior in a trajectory. We classify changes in judge behavior and attribute changes to on
    
[^165]: 量化脑电图（EEG）基础模型与信号处理嵌入中的特质-状态层次结构

    Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings

    [https://arxiv.org/abs/2611.05652](https://arxiv.org/abs/2611.05652)

    该论文提出用层次结构度量和kNN纯度量化EEG嵌入中特质与状态信息的编码方式，发现大多数EEG特征呈现“宏观特质聚类、微观状态嵌套其中”的层次结构。

    

    脑电图（EEG）通过记录头皮上的电位来测量大脑活动。EEG记录捕捉信号中两组不同的特征：（1）特质——在一次记录过程中保持相对恒定的信息，例如受试者身份、年龄和性别；（2）状态——编码对外部刺激响应的特征，包括视觉、听觉和运动信息。目前，人们对不同的EEG建模特征（信号处理方法与基础模型）如何编码状态和特质特征的理解仍然有限。在这项工作中，我们使用层次结构度量指标和kNN纯度，在脑机接口（BCI）数据集试验上量化EEG嵌入所捕获的状态级和特质级特征，并发现大多数EEG特征形成了宏观特质层次结构（即嵌入形成特质级聚类，其内部包含微观状态）。我们的线性探测结果表明……

    arXiv:2611.05652v1 Announce Type: new  Abstract: Electroencephalography (EEG) measures brain activity through recording electrical potentials of the scalp. EEG recordings capture two distinct sets of characteristics in the signal: (1) \emph{Traits} -- information that remains relatively constant over the course of a recording session, such as subject identity, age, and sex; (2) \emph{States} -- signatures that encode responses to external stimuli, including visual, auditory, and motor information. There is limited understanding of how different EEG modeling features (signal processing and foundation models) encode state and trait characteristics. In this work, we quantify state- and trait-level features captured by EEG embeddings using a hierarchy metric and kNN purity on Brain-Computer Interface dataset trials, and find that a majority of EEG features form a macro-trait hierarchy (embeddings form trait-level clusters with micro-states within). Our linear probe results indicate that th
    
[^166]: 将智能体失败归因于其历史需要执行的反事实验证：迈向一种报告标准

    Attributing Agent Failures to Their History Needs an Executed Counterfactual: Toward a Reporting Standard

    [https://arxiv.org/abs/2611.05650](https://arxiv.org/abs/2611.05650)

    该论文提出，将智能体失败归因于其历史的论断必须由实际执行的反事实干预来验证——冻结可执行的轨迹中间状态、仅改变历史并继续运行，同时报告经过测量的重放证书——并将其确立为一种新的报告标准。

    

    已发表的关于长程智能体为何失败的解释，通常是基于完整轨迹标注的分类法：由裁判阅读对话记录并命名失败类型。我们认为，将智能体行为归因于其历史的论断应当携带经干预验证的证据——无论是逐实例验证还是通过经验证的估计器：冻结一个可执行的轨迹中间状态，仅改变智能体所见的历史，然后继续执行——必须实际执行而非从对话记录中推断，且在恢复精度已被测量的状态下进行，并附带与估计目标相匹配的重放证书。我们所命名的四类操纵中每一类都有先例，且已有若干方法能从检查点、重新执行的前缀或记录的对话状态出发执行干预（例如 DoVer、REFLECT、Min 等人、Who&When Pro、Causal Agent Replay），但没有任何一项报告了经过测量的重放证书。随后我们给出两项警示性测量，均未应用该标准……

    arXiv:2611.05650v1 Announce Type: new  Abstract: Published explanations of why long-horizon agents fail are commonly taxonomies labeled over completed trajectories: a judge reads a transcript and names the failure. We argue that a claim attributing an agent's behavior to its history should carry intervention-verified evidence, per instance or via a validated estimator: freeze an executable mid-trajectory state, alter only the history the agent sees, and continue - executed rather than inferred from the transcript, at a state whose restoration has been measured, with a replay certificate matched to the estimand. Each of the four manipulation classes we name has a precedent, and several methods already execute interventions from checkpoints, re-executed prefixes or recorded conversation states (for example DoVer, REFLECT, Min et al., Who&When Pro, Causal Agent Replay), but none reports a measured replay certificate. Two cautionary measurements, neither applying the standard, follow: a ca
    
[^167]: VIPBench：语音克隆时代与人类感知对齐的语音身份感知基准

    VIPBench: A Human-Aligned Benchmark for Voice Identity Perception in the Age of Voice Cloning

    [https://arxiv.org/abs/2611.05638](https://arxiv.org/abs/2611.05638)

    该论文提出VIPBench——首个与人类感知对齐的语音身份感知基准，包含1,290名听者对9,800对真实录音、语音克隆和声音融合样本的12万余次“相同或不同”判断，用于评估说话人表示模型与人类身份感知之间的对齐程度。

    

    说话人识别基准测试的是模型从录音中恢复说话人标签的准确程度，将语音身份视为发声者个人的一种属性。然而，这些基准并未评估人类是如何感知语音身份的。听者是否将两个声音听成同一个人，关系到知识产权、网络安全和日常生活的方方面面。我们提出了语音身份感知基准（VIPBench），其中包含1,290名听者对来自100位公众人物的9,800对录音所作出的124,876次“相同或不同”判断。这些录音配对包括真实录音、由文本转语音系统生成的语音克隆，以及两个说话人声音之间的融合体。我们还定义了四个任务，用以评估说话人表示模型与人类身份感知之间的对齐程度。前两个任务衡量模型的相似度分数能在多大程度上预测回答“相同”的听者比例……（原文摘要在此处截断）

    arXiv:2611.05638v1 Announce Type: cross  Abstract: Speaker-recognition benchmarks test how accurately a model recovers the speaker label of a recording, treating voice identity as a property of the person who produced it. However, these benchmarks do not evaluate how humans perceive voice identity. Whether a listener hears two voices as the same person has consequences for intellectual property rights, cybersecurity, and everyday life. We introduce the Voice Identity Perception benchmark (VIPBench), which contains 124,876 same-or-different judgments from 1,290 listeners on 9,800 recording pairs from 100 public figures. The pairs include real recordings, voice clones generated by a text-to-speech system, and morphs between two speakers' voices. We also define four tasks that evaluate the alignment between speaker-representation models and human identity perception. The first two tasks measure how well a model's similarity score predicts the proportion of listeners who respond "same" and
    
[^168]: 记忆之路：面向大语言模型智能体的自适应记忆选择

    Route to Remember: Adaptive Memory Selection for LLM Agents

    [https://arxiv.org/abs/2611.05636](https://arxiv.org/abs/2611.05636)

    该论文提出利用在线策略自蒸馏（OPSD）将用户特定事实等知识内化到模型参数中的权重记忆方法，并自适应地在文本记忆与权重记忆之间进行路由选择，使权重记忆作为长期事实存储在召回性能上可与基于文本的记忆系统相媲美，从而消除随历史累积而增长的查询开销。

    

    生产级智能体的记忆正日益成为瓶颈，而且其记忆几乎完全由文本构成：从超出上下文限制的内容中提取要点，并在查询时检索回上下文中。因此，每个新查询都会产生随累积历史增长而不断增加的额外成本。权重记忆是一种互补的替代方案，它将知识内化到模型参数中，可以在完全没有上下文的情况下进行回答，并且最近已在自进化智能体中证明能够有效内化技能与行为。然而，权重记忆在情景记忆和语义记忆（如用户特定事实）方面的应用仍相对缺乏探索，且早期的权重记忆尝试与基于文本的记忆系统在召回性能上仍存在差距。我们研究了如何通过在线策略自蒸馏（OPSD）使权重记忆成为一种有竞争力的长期事实存储方法。该方法从一个教师模型训练出无上下文的学生模型，该教师是……（原文摘要在此处截断）

    arXiv:2611.05636v1 Announce Type: new  Abstract: The memory of production agents is increasingly becoming a bottleneck and is built almost exclusively from text: key points are extracted from out-of-limit context and retrieved into context at query time. Therefore, every new query incurs an additional cost that grows with the accumulated history. Weight memory is a complementary alternative that internalizes knowledge into model parameters and answers with no context at all, and has recently proven effective in self-evolving agents for internalizing skills and behaviors. However, its application in episodic and semantic memory, such as user-specific facts, remains comparatively unexplored, and a gap remains between earlier attempts at weight memory and text-based systems regarding recall performance. We study how to make weight memory competitive as a long-term factual store by applying on-policy self-distillation (OPSD). It trains a context-free student from a teacher, which is the sa
    
[^169]: 面向视频识别的时空多阶交互

    Spatio-Temporal Multi-Order Interaction for Video Recognition

    [https://arxiv.org/abs/2611.05633](https://arxiv.org/abs/2611.05633)

    该论文提出基于逐深度卷积的线性复杂度多阶时间建模与时空双重交互机制，以更低的计算开销实现更优的视频时空表示。

    

    许多用于视频识别的时间建模方法依赖于显式的跨通道聚合投影，其复杂度随通道数呈二次方增长，带来了显著的计算开销。此外，用于视频识别的时空表示要么偏向局部阶交互（如CNN），要么偏向全局阶交互（如Transformer）。基于MogaNet的多阶空间表示，我们的方法引入了两个互补的贡献来解决这些问题。首先，在时间建模方面，我们引入了仅使用逐深度卷积的并行多阶处理以及基于隐式门控的聚合方式，其复杂度随通道数呈线性增长，从而大幅降低了计算量。其次，我们提出了空间维度与时间维度之间的双重交互机制，能够在不增加计算成本的情况下获得更好的时空表示。

    arXiv:2611.05633v1 Announce Type: cross  Abstract: Many temporal approaches for video recognition rely on explicit cross-channel aggregation projections, which have quadratic complexity in the number of channels, introducing significant computational overhead. In addition, the spatio-temporal representation used for video recognition is biased either toward local order interactions (for CNNs) or toward global order interactions (for transformers). Starting from the multi-order spatial representation of MogaNet, our method introduces two complementary contributions to tackle these problems. First, for the temporal modelling, we introduce parallel multi-order processing with depthwise-only convolutions and implicit gating-based aggregation, which is linear in the number of channels, inducing a considerable computational reduction. Second, we propose a dual interaction between spatial and temporal dimensions, which allows a better spatio-temporal representation without increasing the comp
    
[^170]: 面向自主AI智能体的保证条件化持续授权

    Assurance-Conditioned Continuous Authorization for Autonomous AI Agents

    [https://arxiv.org/abs/2611.05632](https://arxiv.org/abs/2611.05632)

    提出了保证条件化持续授权（ACCA）框架，通过区分证据有效性与情境适用性、类型化依赖图的选择性重评估、权限纪元机制和效果时强制执行，使自主AI智能体的操作权限能够根据当前有正当依据的保证进行持续动态调整。

    

    自主软件智能体可以通过工具执行操作、委派任务，并在先前的授权仍在进行时更改其模型或执行环境。我们提出了保证条件化持续授权（ACCA），这是一种设计为与执行框架无关的方法，用于根据当前有正当依据的保证来维持操作权限。该方法区分证据有效性与情境适用性，从合格证据中推导保证声明，并在主体声明的上限和适用策略范围内解析多维有效权限包络。类型化依赖图支持对与授权相关的变更进行选择性重新评估。当有效权限发生变化时，权限纪元会发生变更；一个独立的可选授权依据版本支持对特定操作的授权条件进行重新验证。受保护的效果时强制执行机制可以在产生重大后果的资源操作之前拒绝过期操作。

    arXiv:2611.05632v1 Announce Type: new  Abstract: Autonomous software agents can act through tools, delegate work, and change their models or execution environments while earlier authorizations remain in flight. We present Assurance-Conditioned Continuous Authorization (ACCA), an approach designed to be harness-independent, for maintaining operational authority according to currently justified assurance. It distinguishes evidence validity from contextual applicability, derives assurance claims from eligible evidence, and resolves a multidimensional effective-authority envelope within a principal-declared ceiling and applicable policy. A typed dependency graph supports selective re-evaluation of authorization-relevant changes. An authority epoch changes when effective authority changes; a distinct optional authorization-basis version supports revalidation of action-specific authorization conditions. Protected effect-time enforcement can reject stale actions before a consequential resourc
    
[^171]: ViBench 2.0：一个面向长周期软件构建的基准测试

    ViBench 2.0: A Benchmark for Long-Horizon Software Building

    [https://arxiv.org/abs/2611.05620](https://arxiv.org/abs/2611.05620)

    ViBench 2.0 是一个用于长周期软件构建的基准测试，通过 17 个随 241 个功能需求逐步演进的全栈 Web 产品和 544 个浏览器智能体测试计划进行严格评估，揭示了最强模型通过高达 98% 的测试计划却仅有最多 65% 的构建能交付完全可用产品的可靠性差距。

    

    在 Replit——我们的智能体工程平台上，用户如今无需编写代码，只需向智能体连续提出功能需求，就能构建完整的产品。然而，现有的编程基准测试评估的是单一的、完全明确指定的任务，且许多基准已接近饱和。我们提出了 ViBench 2.0，一个用于长周期软件构建的基准测试，我们用它来为平台选择模型。它包含 17 个以广泛使用的软件为原型的全栈 Web 产品，通过 241 个功能需求逐步演进，并且只有在完成时才像用户那样被评判：由一个浏览器智能体运行 544 个测试计划，只有所有计划都通过，构建才算通过，其评分协议经过验证以确保这一严格分数的可靠性。通过对九个模型的评估，我们发现了一个可靠性差距：最强的三个模型最多能通过 98% 的测试计划，但在构建中最多只有 65% 能交付完全可用的产品，而没有任何开放权重模型能做到这一点。这一差距正在快速缩小：在某个模型家族的三个版本中，通过率（原文在此处截断）

    arXiv:2611.05620v1 Announce Type: new  Abstract: On Replit, our agentic engineering platform, users now build entire products without writing code, through successive feature requests to an agent. Yet existing coding benchmarks evaluate single, fully specified tasks, and many are near saturation. We introduce ViBench 2.0, a benchmark for long-horizon software building that we use to select our platform's models. Its 17 full-stack web products, modeled on widely used software, grow over 241 feature requests and are judged only when finished, as a user would judge them: a browser agent runs 544 test plans, and a build passes only if every plan does, under a grading protocol validated to keep this strict score reliable. Evaluating nine models, we find a reliability gap: the three strongest pass up to 98% of test plans yet deliver a fully working product in at most 65% of builds, and no open-weight model delivers one. The gap is closing fast: across three releases of one model family, pass
    
[^172]: Whiel：一个用于认证关系程序验证的神经符号平台

    Whiel: A Neurosymbolic Platform for Certified Relational Program Verification

    [https://arxiv.org/abs/2611.05612](https://arxiv.org/abs/2611.05612)

    Whiel 是一个基于 Lean 认证的神经符号验证平台，它把以数据库实例为状态的命令式关系程序与 Hoare 风格推理、神经符号循环不变量发现及 Vampire 定理证明器相结合，实现了对 Datalog 等查询语言的包含/等价问题以及数据集成中完整性约束、模式映射与视图规范的可认证验证。

    

    我们提出了 Whiel（Whi(le-R)el 的缩写）系统，用于在 Lean 中对命令式关系程序进行可认证的推理。该系统能够编码 Datalog 及其带否定的扩展、以及图查询语言核心片段的包含与等价问题。它还可以表示原始查询语言无法表达的优化或增量求值算法，以及数据集成、交换与共享中出现的完整性约束、模式映射和视图的一阶规范。Whiel 的程序状态是数据库实例，其命令式结构支持基于一阶关系规范的组合式 Hoare 风格推理，并借助预言变量和分层不变量来支持归纳论证。Whiel 支持通过神经符号方法发现循环不变量，其产生的验证条件可使用 Vampire 一阶定理证明器进行检验。反例……（原文摘要至此被截断）

    arXiv:2611.05612v1 Announce Type: cross  Abstract: We present the Whiel (a contraction of Whi(le-R)el) system for Lean-certified reasoning about imperative relational programs. The system can encode containment and equivalence problems for Datalog, its extensions with negation, and the core fragment of graph query languages. It can also represent optimized or incremental evaluation algorithms that are not expressible in the original query language, as well as first-order specifications of integrity constraints, schema mappings, and views arising in data integration, exchange, and sharing. Whiel's program states are database instances, and its imperative structure enables compositional Hoare-style reasoning with first-order relational specifications, using prophecy variables and leveled invariants to support inductive arguments. Whiel supports neurosymbolic discovery of loop invariants whose verification conditions can be checked using the Vampire first-order theorem prover. Counterexam
    
[^173]: Hybrid Spark：让内容来选择起草者的自适应投机解码

    Hybrid Spark: Adaptive Speculative Decoding That Lets the Content Choose the Drafter

    [https://arxiv.org/abs/2611.05611](https://arxiv.org/abs/2611.05611)

    Hybrid Spark 提出在每一步同时运行基于模型和无模型两种起草者，并自适应地为每个请求选择更优的草拟链，解决了单一起草者无法适应不同内容类型、无模型起草者无法参与预算分配的问题。

    

    投机解码（Speculative Decoding, SD）可以加速大语言模型（LLM）的服务，但其收益取决于工作负载的“形状”（如批次大小和上下文长度）以及其“内容”。基于模型的起草者（drafter）可以在任何内容上提出较短的草拟块，而无模型的起草者（如 n-gram 查找）在输出重复其输入时（如代码编辑场景），几乎零成本地提出较长的复制内容。自适应验证将整个批次的验证预算分配给最可能被接受的草拟词元，但它仅支持单个基于模型的起草者。无模型起草者无法提供逐词元的置信度，因此无法参与预算分配，部署时要么固定使用一个起草者，要么在多个起草者之间分割 GPU 资源。我们提出了 Hybrid Spark，这是一个在每个步骤同时运行一个基于模型的起草者和一个无模型起草者的框架，并为每个请求提供两者中更优的草拟链，整个过程在自适应验证框架内并完全在 GPU 上运行，从而保持异步性……

    arXiv:2611.05611v1 Announce Type: new  Abstract: Speculative decoding (SD) speeds up large language model (LLM) serving, but its gain depends on the workload's \emph{shape}, such as batch size and context length, and on its \emph{content}. A model-based drafter proposes a short block on any content, while a model-free drafter such as n-gram lookup proposes long copies at almost no cost when the output repeats its input, as in code editing. Adaptive verification spends a batch-wide verification budget on the draft tokens most likely to be accepted, but it supports only one model-based drafter. Model-free drafters provide no per-token confidence, so they cannot take part in budget allocation, and deployments must either fix one drafter or split their GPUs between drafters. We present \sys, a framework that runs a model-based and a model-free drafter on every step and gives each request the better of their chains, within adaptive verification and entirely on the GPU, which keeps asynchron
    
[^174]: 已验证但空洞：LLM驱动的验证代码生成的三个失败层面

    Verified but Vacuous: Three Failure Surfaces of LLM-Driven Verified Code Generation

    [https://arxiv.org/abs/2611.05590](https://arxiv.org/abs/2611.05590)

    论文指出评估LLM生成代码的“验证”不能只用单一指标，而应区分验证器接受、语法规范绑定与语义规范保真度三个独立层面，实证显示HumanEval上99.1%的验证器通过率中仅有1.9%真正绑定到了规范。

    

    在评估LLM生成的代码时，“已验证”究竟应该意味着什么？许多现有流程仅报告单一的验证数字。我们认为这种做法混淆了在条件审计序列中测量的三个不同属性：验证器接受（表面A）、语法层面的规范绑定（表面B：证明义务是否绑定到规范？）以及语义层面的规范保真度（表面C：规范是否捕获了非正式需求？）。两项实证演示表明这三个表面是相互分离的。第一（A ≠ B），通过Li@Opus（在Claude Opus 4.7下对Li等人2025年“以Dafny作为中间表示（IR）”流程的同模型复现）在三个Python基准测试上进行：它在HumanEval上达到99.1%的验证器接受率，但仅有1.9%的代码携带绑定到规范定义的谓词或函数的证书。第二（B ≠ C），通过VCG在两个Rust基准测试上进行：我们使用VCG（这是我们自己的……（原摘要在此处被截断）

    arXiv:2611.05590v1 Announce Type: new  Abstract: What should ``verified'' mean when evaluating LLM-generated code? Many existing pipelines report a single verification number. We argue this collapses three distinct properties measured in a conditional audit sequence: verifier acceptance (Surface A), \emph{syntactic specification binding (Surface B: does the proof obligation bind to the spec?), and \emph{semantic specification fidelity} (Surface C: does the spec capture the informal requirement?). Two empirical demonstrations show the surfaces separate. \textbf{First ($A \neq B$), via Li@Opus (a same-model reproduction of \citet{li2025dafny}'s Dafny-as-intermediate-representation (IR) pipeline under Claude Opus 4.7) on three Python benchmarks}: it reaches \textbf{99.1\% verifier acceptance} on HumanEval while only \textbf{1.9\%} carry a certificate binding to a spec-defined predicate or function. \textbf{Second ($B \neq C$), via VCG on two Rust benchmarks}: we use VCG (which is our own 
    
[^175]: Anduin：一种抗女巫攻击的智能体市场声誉系统

    Anduin: A sybil-resistant agentic marketplace reputation system

    [https://arxiv.org/abs/2611.05586](https://arxiv.org/abs/2611.05586)

    该论文提出Anduin系统，将链上支付历史建模为买卖双方之间的二部图，并基于买方可信网络相连的图路径设计了三种抗女巫攻击的评级算法，使操纵评级的成本大幅提升，从而解决自主AI智能体在链上市场中选择可信服务提供者的难题。

    

    随着自主AI智能体进入无需许可的链上市场，如何选择新的服务提供者仍然是一个核心挑战。现有的声誉系统依赖于稀缺资源，例如人工策展、身份凭证或自愿评价，而简单的交易量排名则容易受到女巫攻击。我们评估了现有的链上支付历史能否产生能够抵御女巫攻击的服务提供者评级。我们使用一个为期157天的数据集，其中包含16条区块链上价值610万美元的6600万笔智能体交易，将支付流建模为买方与服务提供者之间的二部图。服务提供者的评级源自与买方现有可信网络相连的图路径。我们提出了三种抗女巫攻击的评级算法：一种最大流评级算法，以及两种为获得更强评级性能而优化的松弛算法。在813个买方上针对模拟对手对这些算法进行评估，结果表明操纵评级的成本达到20%（摘要在此处被截断）。

    arXiv:2611.05586v1 Announce Type: new  Abstract: As autonomous AI agents enter permissionless onchain markets, selecting new service providers remains a core challenge. Existing reputation systems depend on scarce resources like human curation, identity credentials, or voluntary reviews, while naive volume rankings are vulnerable to Sybil attacks.   We evaluate whether existing onchain payment histories can yield provider ratings that are resilient against Sybil attacks. Using a 157-day dataset comprising 66M agentic transactions worth $6.1M across 16 blockchains, we model payment flows as a bipartite graph between buyers and providers. A provider's rating is derived from graph paths linked to a buyer's existing trusted network. We propose three Sybil-resistant rating algorithms: a max-flow rating and two relaxed algorithms optimized for stronger rating performance.   Evaluating these algorithms against simulated adversaries across 813 buyers, we show that rating manipulation costs 20%
    
[^176]: BrainBind：通过长时自然视频表征连接同步记录的EEG与fMRI

    BrainBind: Bridging Synchronized EEG and fMRI through Long-Form Naturalistic Video Representations

    [https://arxiv.org/abs/2611.05581](https://arxiv.org/abs/2611.05581)

    提出BrainBind框架，将同步记录的EEG与fMRI映射到冻结的长时自然视频嵌入空间中作为共同的视觉-语义参照，从而在长时自然观看场景下实现多模态脑信号的跨模态检索与解码。

    

    真实世界的视觉是动态且场景丰富的，而EEG和fMRI分别以互补的时间尺度和空间尺度来解析它。然而，大多数神经解码系统仅对齐单一模态，且大多针对静态图像或短片段。我们研究同步记录的情形：在相同的长时间自然观看过程中同时记录EEG和fMRI，这不仅提供了共享的刺激，还带来了反复出现的叙事上下文以及大量语义相似的候选样本。我们提出BrainBind，一个将同步的EEG和fMRI映射到冻结的长时视频嵌入空间的框架，并将这些空间视为共同的视觉-语义参照，而非字面上的神经编码。在CineBrain数据集上，跨越两个视频锚点和四种系统配置的五种子Recall@K 2×4矩阵展现出不同的检索特征。使用Omni时，fMRI对完整候选列表的排序能力最强（MRR = 23.82，nDCG@10 = 25.31），而Joint-Dual则取得了最高的平均标签感知R@1（13.52

    arXiv:2611.05581v1 Announce Type: new  Abstract: Real-world vision is dynamic and scene-rich, while EEG and fMRI resolve it at complementary temporal and spatial scales. Most neural decoding systems, however, align one modality in isolation and mostly against static images or short clips. We study the synchronized case: EEG and fMRI are recorded during the same long-form naturalistic viewing, giving a shared stimulus but also recurring narrative context and many semantically similar candidates. We present BrainBind, a framework that maps synchronized EEG and fMRI into frozen long-form video embedding spaces and treats those spaces as common visual-semantic references, not literal neural codes. On CineBrain, a five-seed Recall@K 2 x 4 matrix across two video anchors and four system configurations shows distinct retrieval profiles. With Omni, fMRI organizes the full candidate list most strongly (MRR = 23.82, nDCG@10 = 25.31), while Joint-Dual gives the highest mean label-aware R@1 (13.52
    
[^177]: 炎症性肠病中全切片图像与临床数据的多模态融合方法

    Multimodal Fusion Methods for Whole-Slide Images and Clinical Data in Inflammatory Bowel Disease

    [https://arxiv.org/abs/2611.05579](https://arxiv.org/abs/2611.05579)

    该论文提出了一个模块化框架，在统一条件下系统比较15种多模态融合方法，将全切片图像嵌入与临床数据结合，用于预测炎症性肠病患者的缓解结局，结果显示最佳融合方法在多数场景下优于单模态模型。

    

    多模态AI融合组织学与临床数据是计算病理学中的一个活跃研究领域，但融合方法很少在相同条件下、跨不同临床终点进行比较。我们提出了一个模块化框架，其中切片级嵌入（通过在冻结的基础模型所产生的补丁嵌入上训练基于注意力的聚合器获得）与每次就诊的临床嵌入被输入到15种可互换的融合方法中，这些方法共享输出维度和预测头（决策级方法除外）。使用一个新诊断炎症性肠病患者的前瞻性队列，我们从基线或随访就诊预测符合STRIDE-II标准的缓解结局。最佳融合方法的宏F1达到0.57–0.75，在六个场景中的四个场景中优于两种单模态模型，且其表现因结局和输入不同而有所差异。图像依赖性，即当切片嵌入被重新（原文此处截断）……

    arXiv:2611.05579v1 Announce Type: cross  Abstract: Multimodal AI fusion of histology and clinical data is an active research area in computational pathology, but fusion methods are rarely compared under identical conditions and across different clinical endpoints. We present a modular framework in which slide-level embeddings, obtained by training an attention-based aggregator over patch embeddings from a frozen foundation model, and per-visit clinical embeddings are fed into 15 interchangeable fusion methods with a shared output dimension and prediction head (except decision-level methods). Using a prospective cohort of newly diagnosed inflammatory bowel disease patients, we predicted STRIDE-II-aligned remission outcomes from baseline or follow-up visits. The best fusion method reached macro-F1 of 0.57-0.75, outperforming both single-modality models in four of six scenarios, and differed by outcome and input. Image reliance, the drop in balanced accuracy when the slide embedding is re
    
[^178]: 可读性不等于可操控性：对学习到的流体表征中物理结构的预注册测试

    Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations

    [https://arxiv.org/abs/2611.05577](https://arxiv.org/abs/2611.05577)

    该研究通过预注册实验发现，自监督模型学到的流体表征虽然具有线性可解码的“可读性”，但并不围绕物理量组织——JEPA表现不优于未训练网络，而普通自编码器反而在最慢潜在方向上突出涡量耗散，且添加VICReg正则化会削弱这种物理结构。

    

    自监督世界模型常被认为会围绕物理量来组织其潜在表示。我们在二维衰减的纳维-斯托克斯流动上通过三项预注册研究检验了这一说法，每项研究仅在保留数据上测试一次。在研究1中，训练用于预测流场的联合嵌入预测架构（JEPA）未能满足预注册标准，即其潜在表示围绕流体量的组织程度并不优于相同架构的未训练网络（假设不被支持）；而在非物理线性动力学上训练的JEPA，以及从未接触时间信息的静态不变性模型，表现得与其相当甚至更好。在研究2中，渲染流场的普通自编码器使其最慢的无监督潜在方向上涡量耗散（enstrophy）变得显著（最佳相关系数 r 为 0.80-0.86，而未训练网络为 0.38；采用全新随机种子和封存数据）。添加 VICReg 在所有比较中都降低了这种显著性，尽管在涡度方面未达到预注册的差距，同时提高了自编码器的线性可解码性（部分支持……）

    arXiv:2611.05577v1 Announce Type: cross  Abstract: Self-supervised world models are often said to organize their latents around physical quantities. We test this on 2D decaying Navier-Stokes flow in three pre-registered studies, each tested once on held-out data. In Study 1, a joint-embedding predictive architecture (JEPA) trained to predict the flow fails the pre-registered criteria for organizing its latent around fluid quantities better than an untrained network of the same architecture (not supported); a JEPA trained on non-physical linear dynamics and a static invariance model that never sees time do as well or better. In Study 2, plain autoencoders that render the field make enstrophy prominent in their slowest unsupervised latent directions (best r 0.80-0.86 vs 0.38 untrained; fresh seeds, sealed data). Adding VICReg lowers this prominence in every comparison, though not by the pre-registered margin for vorticity, while raising the autoencoders' linear decodability (partially su
    
[^179]: 仅凭事实：与AI的对话如何减少阴谋论信念

    Just the Facts: How Dialogues with AI Reduce Conspiracy Beliefs

    [https://arxiv.org/abs/2611.05572](https://arxiv.org/abs/2611.05572)

    与AI对话之所以能减少阴谋论信念，关键在于其提供的相关且论证充分的事实性反驳证据，而非融洽关系、讲故事或共情诉求。

    

    与AI的个性化对话能显著减少阴谋论信念。为了探究其原因，我们开展了两项针对在线招募的美国成年人的实验（总样本量 N = 2,193），逐一排除可能的原因。研究发现，辟谣效果在以下情况下依然稳健：披露AI的说服目标、将交流设定为辩论形式、指示AI进行告知而非说服、将AI回复长度减半、改变信念的测量方式，以及用单条定制信息取代完整对话。只有在指示AI在缺乏事实证据和论据的情况下进行说服时，辟谣效果才会大幅减弱。对AI输出内容及参与者自述的内容分析表明，阴谋论信念的大幅减少源于事实性论证，而非融洽关系、讲故事或共情诉求。AI对话似乎主要通过提供相关且论证充分的事实性反驳证据来减少阴谋论信念。我们的结果表明……

    arXiv:2611.05572v1 Announce Type: cross  Abstract: Personalized dialogues with AI substantially reduce conspiracy beliefs. To identify why, we conducted two experiments with U.S. adults recruited online (total N = 2,193) that removed possible causes one at a time. The debunking effect was robust to disclosing the AI's persuasive goal, framing the exchange as a debate, instructing the AI to inform rather than persuade, halving its response length, altering how beliefs were measured, and replacing the dialogue with a single tailored message. Only instructing the AI to persuade without factual evidence and arguments substantially reduced the debunking effect. Content analyses of the AI's output and participants' own accounts linked larger reductions in conspiracy beliefs to factual argumentation rather than rapport, storytelling, or appeals to empathy. AI dialogues appear to reduce conspiracy beliefs primarily by supplying relevant, well-reasoned counterevidence. Our results suggest that 
    
[^180]: MESA：面向机器人组合性、语义性与空间泛化的评估框架

    MESA: An Evaluation Framework for Compositional, Semantic, and Spatial Generalization in Robotics

    [https://arxiv.org/abs/2611.05567](https://arxiv.org/abs/2611.05567)

    提出了MESA评估框架，通过五个评估套件和四个泛化维度（空间配置、物体实例、物体类别、子任务组合）实现语言条件机器人策略泛化能力的精确可重复测量，并配套开发了基于改进MimicGen的任务与演示自动生成流水线MESA-Gen。

    

    尽管人们对机器人操作基础模型的兴趣日益增长，但对其能力的系统性评估仍然难以实现。现有基准要么狭窄且已饱和，无法为最先进方法提供有效信号；要么范围过于宽泛，以牺牲受控实验为代价换取多样性；要么需要在大型、杂乱的开源数据集上进行训练。我们提出了MESA，一个旨在对语言条件策略泛化能力进行精确、可重复测量的评估框架。MESA提供五个评估套件用于隔离分布内性能，并涵盖四个不同的泛化维度：空间配置、物体实例、物体类别和子任务组合，使研究人员能够诊断策略在何处以及为何失败，而不仅仅是判断其是否失败。为了大规模填充这些评估套件，我们开发了MESA-Gen，一个通过改进MimicGen构建的任务与演示生成流水线。我们对该框架进行了实例化……（摘要在此处截断）

    arXiv:2611.05567v1 Announce Type: cross  Abstract: Despite growing interest in foundation models for robotic manipulation, systematic evaluation of their capabilities remains elusive. Existing benchmarks are either narrow and saturated, offering little signal on SOTA methods, broadly scoped, providing diversity at the cost of controlled experimentation, or require training on large, messy open-source datasets. We introduce MESA, an evaluation framework designed for precise, reproducible measurement of language-conditioned policy generalization. MESA provides five evaluation suites that isolate in-distribution performance and four distinct axes of generalization: spatial configuration, object instance, object category, and subtask composition, enabling researchers to diagnose where and why policies fail, not just whether they fail. To populate these suites at scale, we develop MESA-Gen, a pipeline for task and demonstration generation built by improving MimicGen. We instantiate the fram
    
[^181]: 被丢弃还是不可触及？诊断与修复视觉-语言相似性中的空间关系失效

    Discarded or Unreachable? Diagnosing and Repairing Spatial Relation Failure in Vision--Language Similarity

    [https://arxiv.org/abs/2611.05565](https://arxiv.org/abs/2611.05565)

    本文揭示了对比视觉-语言模型在左右空间关系判断上的失效存在两种成因——关系信息可被解码但相似度搜索无法触及的“访问失败”，以及池化丢弃信息而密集特征保留的“池化失败”，并提出一种零参数、查询条件化的空间矩方法从patch网格重新计算几何信息，用一个读取机制统一修复两类失败。

    

    对比式视觉-语言模型对左/右空间关系的评分接近随机水平，但这一症状有两种成因，需要截然相反的修复方式：空间关系可能仍保留在池化后的嵌入中，却无法通过原生图文相似度触及；或者池化过程丢弃了这一关系，而密集特征仍保留着它。在干净的双物体照片上，跨十个编码器，横向极性可以从池化向量的翻转奇分量中以0.77-0.99的准确率被线性解码，然而在对一百万条描述的搜索中，十个编码器中有九个在原生评分下仍处于随机水平：这是访问失败。在受控的C4轨道（其中空间关系是唯一变化量）上，池化以0.998-1.000的准确率保留了排列轴却丢失了其极性，而patch网格保留了极性：这是池化失败。两者都位于密集特征的下游，因此单一的读取机制即可同时覆盖两类失败。我们的零参数、查询条件化空间矩方法从patch网格重新计算几何信息并存储……（原文摘要在此处截断）

    arXiv:2611.05565v1 Announce Type: cross  Abstract: Contrastive vision-language models score left/right relations near chance, but the symptom has two causes that call for opposite repairs: the relation may survive in the pooled embedding yet be unreachable by native image-text similarity, or pooling may discard it while the dense features retain it. On clean two object photographs, across ten encoders, the lateral polarity is linearly decodable from the flip odd component of the pooled vector at 0.77-0.99, yet a search over a million captions leaves nine of the ten at chance under native scoring: an access failure. On controlled C4 orbits, where the relation is the only thing that varies, pooling keeps the arrangement axis at 0.998-1.000 and loses its polarity, which the patch grid retains: a pooling failure. Both sit downstream of the dense features, so one readout covers both. Our zero parameter, query conditioned spatial moment recomputes the geometry from the patch grid and stores 
    
[^182]: 视觉-语言模型能把生物学看得多细致？Bio-VQA：一个面向分类学、形态学、空间和个体推理的基准测试

    How Finely Can Vision-Language Models See Biology? Bio-VQA: A Benchmark for Taxonomic, Morphological, Spatial and Individual Reasoning

    [https://arxiv.org/abs/2611.05559](https://arxiv.org/abs/2611.05559)

    该论文提出了Bio-VQA——目前最大的生物视觉问答基准，涵盖层次化分类推理、细粒度属性定位、定量空间推理和个体重识别四个方面，包含21.7万张图像上的186万个问题，并引入分级评分机制，系统评估了18个开源视觉-语言模型在生物理解上的能力。

    

    在生物多样性监测中，一张图像会引出多个问题：它属于哪个分类单元、数量是多少、具有什么形态特征、以及该个体此前是否被观测过。视觉-语言模型（VLMs）本可以从图像中回答所有这些问题，但其在生物学领域的能力在很大程度上尚未经过检验，因为现有的基准测试大多只局限于少数分类群内的物种识别。我们提出了Bio-VQA，这是目前最大的生物视觉问答（VQA）基准，也是唯一一个涵盖生物视觉理解四个方面的基准：层次化分类推理、细粒度属性定位、定量空间推理和个体重识别。Bio-VQA包含来自11个公开数据集的21.7万张图像上的186万个问题，并将简单的对错评分扩展为分级评分，使得接近正确的答案比偏差较大的答案获得更高的分数。通过对18个参数量从3.75B到38.4B的开源权重模型进行评估，我们发现通用视觉能力（原文摘要在此处截断）……

    arXiv:2611.05559v1 Announce Type: cross  Abstract: In biodiversity monitoring, a single image raises several questions: which taxon, how many, which trait, and whether the individual has been seen before. Vision-language models (VLMs) could answer all of these from the image, but their ability is largely untested in biology, as existing benchmarks focus mostly on species identification within a few taxonomic groups. We present Bio-VQA, the largest biological visual question answering (VQA) benchmark and the only one to span four aspects of biological visual understanding: hierarchical taxonomic reasoning, fine-grained attribute grounding, quantitative spatial reasoning, and individual re-identification. Bio-VQA contains 1.86M questions over 217K images from 11 public datasets, and extends right-or-wrong scoring to graded scoring, so a near-miss counts for more than a distant one. Evaluating 18 open-weight models from 3.75B to 38.4B parameters, we find that general visual capabilities d
    
[^183]: AGORA：基于项目反应理论的可靠性感知LLM裁判聚合方法

    AGORA: Reliability-Aware Aggregation of LLM Judges via Item-Response Theory

    [https://arxiv.org/abs/2611.05556](https://arxiv.org/abs/2611.05556)

    提出基于项目反应理论的AGORA聚合器，通过敏感度参数和有序评分阈值刻画各LLM裁判，无需人工标注即可从全部评估项中联合学习裁判可靠性，在仅有几百个样本的小规模评估集上显著优于Dawid-Skene等传统方法。

    

    LLM作为裁判（LLM-as-a-judge）是替代昂贵人工评估的常用方式，而聚合多个裁判的评估结果是应对单一裁判偏差的标准方法。然而，裁判池中混杂着可靠性差异极大的裁判，而用于验证裁判的人工标注数据集通常只有几百个样本甚至更少。在这种情形下，逐项平均的方法对弱裁判和强裁判赋予同等权重，而像Dawid-Skene这样的混淆表模型则必须从过少的样本中估计每个裁判的错误表。我们提出AGORA（可靠性加权评估聚合器），这是一个基于项目反应理论的聚合方法，它将每个评估项置于同一潜在质量量表上，并用敏感度参数和有序评分阈值来刻画每个裁判，从而能够从所有评估项中联合学习裁判的可靠性，且无需人工标注。在三个基准测试中，AGORA在大型评估集上达到或超过Dawid-Skene的表现，且随着评估集缩小至几百个样本，其优势进一步扩大。

    arXiv:2611.05556v1 Announce Type: new  Abstract: LLM-as-a-judge is a common substitute for costly human evaluation, and aggregating several judges is the standard remedy for any single judge's bias. Judge pools mix judges of very different reliability, while the human-labeled sets that validate them hold a few hundred items or fewer. In this regime the per-item mean weights weak judges as heavily as strong ones, and confusion-table models such as Dawid-Skene must estimate a per-judge error table from too few items. We propose AGORA (Aggregation Of Reliability-weighted Assessments), an item-response-theory aggregator that places every item on one latent quality scale and describes each judge by a sensitivity parameter and ordered rating thresholds, so judge reliability is learned from all items jointly and without human labels. Across three benchmarks, AGORA matches or exceeds Dawid-Skene on large evaluation sets, and its advantage widens as the set shrinks toward a few hundred items. I
    
[^184]: 决策模型：能够捕捉人类差异的可扩展经济替代模型

    Decision Models as Scalable Economic Surrogates that Capture Human Variation

    [https://arxiv.org/abs/2611.05553](https://arxiv.org/abs/2611.05553)

    本研究首次评估了决策模型在经济学中作为人类替代者的潜力，发现前沿决策模型在三种经济场景中比大语言模型更能再现人类回应的多样性，同时速度快约十倍，是大规模经济模拟与调查的更优选择。

    

    决策模型（DMs）是一类新型AI模型，能够快速给出响应，并返回一组固定选项上的概率分布。这些特性使其成为经济模拟与调查中人类替代者的有力候选——这一角色目前正越来越多地由大语言模型（LLMs）承担，但LLMs在大规模应用时成本高昂，且无法充分体现人类行为的多样性。我们首次评估了DMs在经济学领域作为人类替代者的表现，在三种经济场景中将10个DMs与8个LLMs进行比较：一项关于消费的家庭支出调查、若干经典经济博弈，以及一项劳动力市场田野实验。在所有三种场景中，前沿DMs（即在通用决策基准上得分最高的模型）都比LLMs更贴近地再现了人类回应的多样性。在支出调查中，它们对人类最典型回应的捕捉能力与LLMs相当，但在经济博弈中表现稍逊。此外，前沿DMs的速度比最快的……大约快十倍

    arXiv:2611.05553v1 Announce Type: cross  Abstract: Decision Models (DMs), a new class of AI models, give fast responses and return a probability distribution over a fixed set of options. These properties make them candidate human surrogates in economic simulations and surveys, a role increasingly filled by Large Language Models (LLMs), which are costly at scale and under-represent the variation in human behaviour. We present a first evaluation of DMs as human surrogates in economics, comparing 10 DMs with 8 LLMs in three economic settings: a household spending survey on consumption, several classic economic games and a labour-market field experiment. Frontier DMs, those that score highest on general decision-making benchmarks, reproduce the variation in human responses more closely than LLMs in all three settings. They capture the modal human response as well as LLMs in the spending survey, but less well in the economic games. Frontier DMs are also roughly ten times faster than the fas
    
[^185]: VirtualTrialHarness：在复杂临床数据模式间协调虚拟试验方案智能体化执行

    VirtualTrialHarness: Coordinating Agentic Operationalization of Virtual Trial Protocols Across Complex Clinical Schemas

    [https://arxiv.org/abs/2611.05551](https://arxiv.org/abs/2611.05551)

    提出了智能体框架 VirtualTrialHarness，通过为各工作流阶段选择可复用程序性技能并利用显式契约保持跨阶段决策一致，实现虚拟试验方案在复杂临床数据环境中的自动化执行，并配套发布评估基准 VTBenchmark。

    

    虚拟试验利用常规收集的临床数据来评估治疗策略，但要将研究方案转化为可执行的分析，需要确定临床概念和研究要求应如何在可用数据中表示，并在整个工作流程中一致地应用。这些决策依赖于本地数据表示方式和操作惯例，可能涉及数据源选择、患者与就诊身份、数据关联以及时间解释等问题。在此，我们提出了 VirtualTrialHarness（VTHarness），这是一个智能体框架，通过为每个工作流阶段选择可复用的程序性技能来协调虚拟试验的方案执行，并使用显式契约在各阶段之间保持研究决策的一致性。我们还引入了 VTBenchmark，用于评估数据源选择和队列构建。在三个复杂的合成环境中，VTHarness 成功生成了所有所需的队列输出，并且……

    arXiv:2611.05551v1 Announce Type: new  Abstract: Virtual trials use routinely collected clinical data to evaluate treatment strategies, but translating a study protocol into an executable analysis requires determining how clinical concepts and study requirements should be represented in the available data and applied consistently across the workflow. These decisions depend on local data representations and operational conventions and can involve source selection, patient and encounter identity, linkage, and temporal interpretation. Here we introduce VirtualTrialHarness (VTHarness), an agent framework that coordinates virtual-trial operationalization by selecting reusable procedural skills for each workflow stage and uses explicit contracts to maintain consistent study decisions across stages. We also introduce VTBenchmark for evaluating source selection and cohort construction. Across three complex synthetic environments, VTHarness successfully produced all required cohort outputs matc
    
[^186]: 这究竟是谁的透明度？理解生成式AI中不同利益相关者的透明度需求与差距

    Whose Transparency is it Anyway? Understanding Generative AI Transparency Needs and Gaps Across Stakeholders

    [https://arxiv.org/abs/2611.05549](https://arxiv.org/abs/2611.05549)

    本研究通过对最终用户、AI开发者和政策制定者的全球调查发现，尽管各方对透明度的理解相似，但最终用户在透明度需求与实际可获得性之间的差距远大于开发者，凸显了当前生成式AI透明度实践中的利益相关者不平等问题。

    

    透明度已成为AI治理的核心原则，然而关于透明度究竟意味着什么、为谁服务以及在何种情境下适用，目前尚未达成共识。本文借助技术的社会建构理论中“解释灵活性”的概念，考察了不同利益相关者如何理解和体验生成式AI中的透明度。本研究基于一项覆盖400名最终用户、AI开发者和政策制定者的全球调查，并辅以针对最终用户的结构化焦点小组讨论，将定量发现扎根于当前透明度实践中服务最不足群体的定性洞察之上。我们提出了三个关键发现：第一，三个群体对透明度的含义及其应实现的功能有相似的理解，核心聚焦于确保准确性与隐私；第二，透明度特性的重要性与其实际可获得性之间的差距，在最终用户群体中远大于开发者群体，揭示了……

    arXiv:2611.05549v1 Announce Type: new  Abstract: Transparency has become a central principle in AI governance, yet there is little agreement about what transparency means, for whom, and in what contexts. Drawing on interpretive flexibility within the Social Construction of Technology, this paper examines how different stakeholders understand and experience transparency in generative AI. This study draws from a global survey of 400 end-users, AI builders, and policymakers, complemented by a structured focus group with end-users, grounding the quantitative findings in qualitative insights from the group most underserved by current transparency practice. We present three key findings. First, all three groups share similar understandings of what transparency means and what it should afford, centered on ensuring accuracy and privacy. Second, gaps between the importance of transparency features and their perceived availability are substantially larger for end-users than builders, revealing h
    
[^187]: 语言模型对数据科学项目的迭代式创建、评分与修订

    Iterated creation, grading, and revision of data science projects by language models

    [https://arxiv.org/abs/2611.05548](https://arxiv.org/abs/2611.05548)

    本研究通过“写作者-评分者”迭代反馈框架评估七个语言模型完成端到端数据科学项目的能力，发现三轮反馈修订使平均得分从59%提升至76%，且不同模型提升幅度差异显著（3至46个百分点）。

    

    我们评估了智能体AI模型能够完成端到端数据科学项目的程度，以及这一能力如何随反馈而提升。一个配备特定工具的“写作者”模型创建一个完整的数据科学项目，包括下载数据集、清洗和分析数据集，并就此撰写一篇短论文。项目随后被执行，并由一个“评分者”模型进行评估。写作者模型接着可以修改其提交内容，如此重复共进行三轮。我们考虑了来自五个开发商（Anthropic、OpenAI、Google、DeepSeek和Z.AI）的七个模型。每个模型既充当写作者又充当评分者，因此共有49个写作者-评分者组合，我们对每个组合运行三次。在147份初始提交中，107份能够成功运行并渲染，经过最后一轮反馈后，平均得分从59%提升至76%。按写作者模型来看，提升幅度从Anthropic的Opus 5.5的3个百分点到Anthropic的Haiku的46个百分点不等。

    arXiv:2611.05548v1 Announce Type: new  Abstract: We evaluate the extent to which agentic AI models can complete an end-to-end data science project, and how this improves with feedback. A 'writer' model with specific tools creates a complete data science project including downloading, cleaning, and analyzing a dataset, and writing a short paper about it. The project is executed and a 'grader' model evaluates it. The writer can then revise its submission, and this repeats for a total of three rounds. We consider seven models from five developers: Anthropic, OpenAI, Google, DeepSeek, and Z.AI. Each model is both a writer and a grader, so there are 49 writer-grader pairs and we run each pair three times. Of the 147 initial submissions, 107 run and render, and the average score increases from 59 per cent to 76 per cent after the final round of feedback. By writer the extent of the increase ranges from 3 percentage points for Anthropic's Opus 5.5 to 46 percentage points for Anthropic's Haiku
    
[^188]: 基于单次推理的高效整页排序

    Towards Efficient Whole Page Ranking with One-Pass Inference

    [https://arxiv.org/abs/2611.05542](https://arxiv.org/abs/2611.05542)

    该论文提出MIS（多物品打分）框架，一种感知前缀的NAR式方法，在单次推理的高效框架中引入序列上下文与位置感知打分，弥合了整页排序中自回归方法与非自回归方法之间的质量-效率差距。

    

    整页排序在复杂的布局与业务约束下，联合优化广告与自然内容，以平衡业务收益和用户参与度。现有的非自回归（NAR）方法能够实现高效的页面生成，但通常对候选物品的打分独立于先前已选中的物品，限制了其捕捉与顺序相关的交互的能力。自回归（AR）方法通过序列条件化解决了这一局限，但会产生大量的推理开销，给大规模部署带来挑战。弥合这种质量与效率之间的差距，仍然是工业界整页排序中的一项难题。我们提出了多物品打分（MIS），这是一种感知前缀的NAR风格框架，将序列上下文融入高效的多物品打分之中。MIS在共享表示架构内集成了全局候选交互、局部前缀条件化以及位置感知打分。通过计算……

    arXiv:2611.05542v1 Announce Type: new  Abstract: Whole-page ranking jointly optimizes advertisements and organic content to balance business outcomes and user engagement under complex layout and business constraints. Existing non-autoregressive (NAR) approaches enable efficient page generation but typically score candidate items independently of previously selected items, limiting their ability to capture order-dependent interactions. Autoregressive (AR) methods address this limitation through sequential conditioning, but incur substantial inference overhead that challenges large-scale deployment. Bridging this quality-efficiency gap remains a challenge in industrial whole-page ranking.   We propose Multi-Item Scoring (MIS), a prefix-aware NAR-style framework that incorporates sequential context into efficient multi-item scoring. MIS integrates global candidate interactions, localized prefix conditioning, and position-aware scoring within a shared representation architecture. By comput
    
[^189]: MA-SafeDiffuser：基于扩散概率模型的安全多智能体规划

    MA-SafeDiffuser: Safe Multi-Agent Planning with Diffusion Probabilistic Models

    [https://arxiv.org/abs/2611.05524](https://arxiv.org/abs/2611.05524)

    MA-SafeDiffuser将控制屏障函数约束嵌入多智能体扩散规划的去噪过程，实现了具有可证安全不变性保证的集中式与分散式多智能体轨迹规划，并通过时变规范和活性CBF有效解决了局部陷阱与死锁问题。

    

    我们提出了MA-SafeDiffuser，这是SafeDiffuser的多智能体扩展，为基于扩散的轨迹规划器提供了针对多智能体联合安全规范的有限时间扩散不变性保证。在将控制屏障函数（CBF）约束嵌入反向扩散更新的单智能体构造基础上，我们：(i) 将联合安全集形式化为每个智能体屏障集与成对屏障集的交集；(ii) 在温和假设下推导出具有可证不变性的集中式与分散式（感知通信的）约束去噪程序；(iii) 通过时变规范和活性CBF解决局部陷阱与死锁现象；(iv) 开发了一个包含多智能体Maze2D域的轻量级基准测试套件。实验表明，MA-SafeDiffuser相对于无约束扩散基线减少了违规次数，同时保持了规划质量。我们提供了算法、证明和参考实现。

    arXiv:2611.05524v1 Announce Type: new  Abstract: We propose MA-SafeDiffuser, a multi-agent extension of SafeDiffuser that equips diffusion-based trajectory planners with finite-time diffusion invariance guarantees for joint-agent safety specifications. Building on the single-agent construction that embeds control barrier function (CBF) constraints into reverse-diffusion updates, we: (i) formalize joint safe sets as intersections of per-agent and pairwise barrier sets; (ii) derive centralized and decentralized (communication-aware) constrained denoising procedures with provable invariance under mild assumptions; (iii) address local-trap and deadlock phenomena via time-varying specifications and liveness CBFs; and (iv) develop a lightweight benchmarking suite including a multi-agent Maze2D domain. Empirically, MA-SafeDiffuser reduces violation counts relative to unconstrained diffusion baselines, while retaining planning quality. We provide algorithms, proofs, and reference implementatio
    
[^190]: 何时提问，何时回答：对话中歧义指代消解的策略性澄清

    When to Ask, When to Answer: Strategic Clarification for Ambiguous Reference Resolution in Dialogue

    [https://arxiv.org/abs/2611.05517](https://arxiv.org/abs/2611.05517)

    该论文提出“策略性澄清”任务并基于 SIMMC 2.1 构建了可控评测环境，同时提出轻量级的 BACE 动作选择方法，让 LLM 学会在澄清问题能缩小候选指代范围时才提问、否则直接回答，以缓解其过早作答的偏差。

    

    对人类而言，指代消歧是一个协作过程，人们只在澄清问题能够实质性改变结果时才会提出这些问题。然而，已有研究表明，大语言模型（LLM）存在过早作答的偏差，更倾向于给出确定的单轮回答，而非进行更多的澄清式交流。本文研究“策略性澄清”问题——即在指代游戏中，当澄清问题的答案能够缩小候选指称对象的集合时才提出澄清问题。我们基于 SIMMC 2.1 数据集构建了一个受控测试环境，该环境记录了每个歧义指代需要多少个问题才能被成功消解，并考察了基线 LLM 行为和常见后训练方法在“提出澄清问题”与“尝试回答指代”之间的表现。我们还提出了 BACE（Belief-Aware Critic with Epistemic tracking，具有认知追踪的信念感知评论器），这是一种在冻结 LM 骨干之上的轻量级动作选择方法，并展示了……

    arXiv:2611.05517v1 Announce Type: cross  Abstract: For humans, referential disambiguation is a collaborative process involving clarification questions asked when they will meaningfully change the outcome. However, large language models (LLMs) have been shown to be biased toward premature answering, preferring definite single-turn responses over more clarification exchanges. In this paper, we study the problem of "strategic clarification" -- asking clarification questions in a reference game when their answers will narrow the set of candidate referents. We construct a controlled test environment from the SIMMC 2.1 dataset that records how many questions each ambiguous reference needs to be successfully resolved, and examine how baseline LLM behavior and common post-training methods ask clarifying questions vs. attempt to answer the referent. We also propose BACE (Belief-Aware Critic with Epistemic tracking), a lightweight action chooser method on top of a frozen LM backbone, and show th
    
[^191]: TSFM 遇见 LLM：作为协变量的上下文

    TSFM Meets LLM: Context as Covariates

    [https://arxiv.org/abs/2611.05515](https://arxiv.org/abs/2611.05515)

    该论文提出 CoCo 预测框架，让 LLM 将文本推理转化为数值化的预测与置信度协变量来引导 TSFM 骨干模型，融合两者优势进行时间序列预测，并提供利用 LLM 标注真实时间序列以生成多模态数据集的可扩展方法。

    

    文本信息在现实世界的时间序列数据集中无处不在，以新闻、报告和注释等形式出现，其中蕴含着仅凭数值历史无法推断出的关键预测信号。大语言模型（LLM）能够解读文本但缺乏数值精度，而时间序列基础模型（TSFM）在数值上精确却无法对文本的影响进行推理。此外，大规模成对多模态数据集的缺失也限制了弥合这一差距的研究进展。在本工作中，我们通过提供一种可扩展的方法来应对这一挑战，即利用 LLM 对真实世界的时间序列进行标注，从而生成逼真的多模态数据集。在此基础上，我们提出了“上下文作为协变量”预测框架 CoCo，其中 LLM 将其文本推理提炼为基于数值的预测协变量和置信度协变量，用以引导 TSFM 骨干模型，从而充分发挥 LLM 与 TSFM 各自的优势。我们将两种模型在……

    arXiv:2611.05515v1 Announce Type: cross  Abstract: Textual information is ubiquitous in real-world time series datasets, appearing as news, reports, and annotations that hold key predictive signals not deducible from numerical history alone. Large Language Models (LLMs) interpret text but lack numerical precision, while Time Series Foundation Models (TSFMs) are quantitatively accurate but cannot reason about the effects of text. Moreover, the lack of large-scale paired multimodal datasets limits progress in bridging this gap. In this work, we address this challenge by providing a scalable approach for generating realistic multimodal datasets by annotating real-world time series with LLMs. Building on this, we propose Context as Covariates (CoCo), a forecasting framework in which an LLM distills its textual reasoning into numerically grounded forecast and confidence covariates that guide a TSFM backbone, thereby exploiting the strengths of both LLMs and TSFMs. We align the two models in
    
[^192]: SlackBench：基于真实代码仓库的协作项目中智能体基准测试

    SlackBench: Benchmarking Agents on Collaborative Projects Grounded in Real Code Repositories

    [https://arxiv.org/abs/2611.05512](https://arxiv.org/abs/2611.05512)

    SlackBench是一个基于真实学术研究项目隐私保护重现构建的基准测试，用于评估AI智能体在需要对工作场所沟通与不断演变的代码仓库状态进行联合推理的现实协作研究与工程工作流中的能力。

    

    AI智能体正在成为数字同事，它们能够浏览网页、编写代码、运行实验，并在较长时间跨度内协助人类工作。然而，现有基准测试难以反映真实的知识工作：许多基准依赖于合成数据或孤立任务，忽略了协作项目中复杂的多方动态——在这种项目中，关键信息散布在对话、代码和不断演变的决策之中。即使是简单的工作场所问题，也可能需要定位相关频道和私信、从冗长的讨论中恢复上下文、检查仓库状态、调和过时或相互冲突的证据，并进行跨来源的综合分析。我们提出了SlackBench，这是一个用于评估智能体在现实研究与工程工作流中表现的基准，这些工作流需要对工作场所沟通与不断演变的仓库状态进行联合推理。SlackBench基于真实学术研究项目的隐私保护重现构建，提供了可复现的……

    arXiv:2611.05512v1 Announce Type: new  Abstract: AI Agents are becoming digital coworkers that can browse the web, write code, run experiments, and assist humans over extended horizons. Yet existing benchmarks fall short of real knowledge work: many rely on synthetic data or isolated tasks, missing the complex, multi-party dynamics of collaborative projects, where critical information is scattered across conversations, code, and evolving decisions. Even simple workplace questions may require locating relevant channels and direct messages, recovering context from long discussions, inspecting repository state, reconciling stale or conflicting evidence, and synthesizing across sources.   We introduce SlackBench, a benchmark for evaluating agents on realistic research and engineering workflows that require joint reasoning over workplace communication and evolving repository state. Grounded in privacy-preserving reenactments of real academic research projects, SlackBench provides reproducib
    
[^193]: 面向多目标对齐的目标条件化主动偏好学习

    Objective-Conditioned Active Preference Learning for Multi-Objective Alignment

    [https://arxiv.org/abs/2611.05506](https://arxiv.org/abs/2611.05506)

    提出 OGC-DPO 主动学习框架，将（回复对，目标）组合视为独立查询并利用策略自身隐式梯度衡量信息量，主动选择最能扩展各目标梯度覆盖范围的比较样本，从而在无需外部奖励模型的情况下降低多目标对齐的标注成本。

    

    将大型语言模型（LLM）与多个目标（如有用性、无害性和诚实性）进行对齐，通常需要对每个目标分别进行偏好判断，使标注成本随目标数量线性增长。主动偏好学习可以通过选择那些标签预期对策略改变最大的比较来降低这一成本。现有的方法要么缺乏面向多目标设置的目标感知查询选择机制，要么将相互冲突的偏好视为有害信息并将其排除在训练数据之外。我们提出了 OGC-DPO（目标条件化梯度覆盖 DPO），这是一个主动学习框架，它将每个（回复对，目标）组合视为一个独立的查询，并通过策略自身的隐式梯度来衡量其信息量，而无需依赖外部奖励模型。OGC-DPO 随后选择最能扩展每个目标梯度覆盖范围的查询。

    arXiv:2611.05506v1 Announce Type: cross  Abstract: Aligning large language models (LLMs) with several objectives, such as helpfulness, harmlessness, and honesty, typically relies on preference judgment for each objective, growing the annotation cost linearly with the number of objectives. Active preference learning can reduce this cost by selecting the comparisons whose labels are expected to change the policy most. Existing remedies either lack an objective-aware query selection mechanism for the multi-objective setting or regard conflicting preferences as harmful, therefore excluding them from the training data. We introduce OGC-DPO (Objective-conditioned Gradient Coverage DPO), an active learning framework that treats each (response pair, objective) combination as a separate query and measures its informativeness through the policy's own implicit gradient, without relying on an external reward model. OGC-DPO then selects the queries that most expand each objective's gradient coverag
    
[^194]: 大语言模型能否完成科学文献综述？基于102篇文献基准的测试

    Can Large Language Models Review the Scientific Literature? A Test Against a 102-Article Benchmark

    [https://arxiv.org/abs/2611.05459](https://arxiv.org/abs/2611.05459)

    该研究以一篇含102篇文献的天气学综述为基准，对六种AI工具在文献综述全流程（检索、解读、综合、展望）进行评估，发现LLM虽有可取之处，但在文献发现能力以及区分实质性引用与顺带提及方面仍存在明显局限。

    

    大语言模型（LLM）正被越来越多地用于支持科学文献综述，但现有评估通常只考察单个任务，而非将综述过程作为一个整体来检验。我们在四个阶段评估了六种AI工具：查找相关文章、解读文章、跨文章综合归纳，以及提出未来研究方向。测试基准是一篇已发表的关于西班牙羽流（Spanish plume）天气型的综述文章及其102篇文献的语料库。在初始检索中，五种工具合计识别出57篇文章，表现最好的单个工具找到42篇。借助机构图书馆访问权限，Codex的文献发现量从42篇提升至82篇，Claude Code从15篇提升至79篇。在给定全部102篇PDF文献的情况下，三种工具正确识别出了所有18篇实质性讨论西班牙羽流的文章，但将84篇仅顺带提及的文献中的40%至79%错误地判定为实质性讨论。同样这三种工具所生成的综述内容由两位领域专家进行盲评……

    arXiv:2611.05459v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support scientific literature reviews, yet evaluations often examine individual tasks rather than the review process as a whole. We assess six AI tools across four stages: finding relevant articles, interpreting them, synthesizing across them, and proposing future research. The benchmark is a published review of the Spanish plume weather pattern and its 102-article corpus.   In the initial searches, five tools collectively identified 57 articles, with the best individual tool finding 42. Institutional library access increased discovery from 42 to 82 articles for Codex and from 15 to 79 for Claude Code. Given the complete corpus of 102 PDFs, three tools correctly identified all 18 articles discussing the Spanish plume substantially, but incorrectly classified 40% to 79% of the 84 passing mentions as substantial. The same three tools produced syntheses evaluated blindly by two domain ex
    
[^195]: 彭博（Bloomberg）企业级代码迁移中的混合 LLM-确定性流水线

    Hybrid LLM-Deterministic Pipelines for Enterprise-Scale Code Migration at Bloomberg

    [https://arxiv.org/abs/2611.05457](https://arxiv.org/abs/2611.05457)

    彭博提出的混合 LLM-确定性流水线架构，通过将 LLM 限定在预计算上下文内进行有界文件转换并结合生产遥测反馈，在约 8,600 个企业仓库的两大迁移行动中实现了 729 次成功代码迁移。

    

    大型组织中存在着大量基础设施迁移的长尾工作，这些工作值得做但往往不够紧迫，难以与功能开发竞争优先级，因为每项迁移在每个代码仓库上都需要耗费大量的工程时间。本文报告了我们从“Automated Uplift（自动提升）”计划中获得的工业实践经验，这是一项历时数月的计划，在彭博的数千个代码仓库中应用了 LLM 辅助的迁移流水线。我们考察了两个不同的生产环境迁移行动：一个异构的 Python 打包迁移（约 1,900 个候选仓库）和一个统一的 C++ 构建系统迁移（约 6,700 个候选仓库）。这些行动总共生成了 2,333 个拉取请求，最终实现了 729 次成功的迁移。我们详细介绍了我们的混合架构解决方案：该方案最大化确定性处理，将 LLM 的使用限定在预先计算好的上下文中执行有界的文件转换，并引入了运行时生产遥测数据（如执行日志和使用轨迹等）。

    arXiv:2611.05457v1 Announce Type: cross  Abstract: Large organizations carry a long tail of infrastructure migrations that are worth doing but rarely urgent enough to compete with feature work, because each one costs significant engineering time per repository. This paper reports on our industrial experience from Automated Uplift, a multi-month initiative applying LLM-assisted migration pipelines across thousands of Bloomberg repositories. We examine two distinct production campaigns: a heterogeneous Python packaging migration (approx. 1,900 candidate repositories) and a uniform C++ build system migration (approx. 6,700 candidate repositories). Together, these campaigns generated 2,333 pull requests, resulting in 729 successful migrations. We detail our hybrid architectural solution which maximizes deterministic processing, confines the LLM to bounded file transformations within a pre-computed context, and incorporates runtime production telemetry (such as execution logs and usage trac
    
[^196]: 迈向稳健的长生命周期智能体：在动态演化环境中对经验生命周期进行基准测试

    Towards Robust Long-Lived Agents: Benchmarking Experience Lifecycles in Evolving Environments

    [https://arxiv.org/abs/2611.05450](https://arxiv.org/abs/2611.05450)

    提出 ReScopeBench 基准，在 18 个交互式环境中区分结构经验与规则经验，并通过经验优先的任务流构建方法，系统评估智能体在环境演化过程中获取、复用与修订经验的完整生命周期。

    

    随着智能体能力的不断增强，它们正从一次性任务转向跨越用户连续目标的持续协助。这种协助要求智能体探究陌生环境的运作方式，在跨任务中复用所学内容，并在环境演化时保持经验的有效性。现有基准越来越多地研究经验复用与适应，但对于智能体会修订、保留或丢失积累经验的哪些部分，其提供的洞察有限。为填补这一空白，我们提出了 ReScopeBench，一个面向经验生命周期、涵盖 18 个交互式环境的基准。它将关于环境如何组织和访问的结构性经验，与指导跨任务决策的可复用原则和约束的规则性经验区分开来。为研究经验如何被获取、复用和修订，我们提出了一种经验优先的构建方法。该方法围绕共享经验构建任务流，并……

    arXiv:2611.05450v1 Announce Type: new  Abstract: As agents become more capable, they are moving beyond one-off tasks toward sustained assistance across successive user goals. Such assistance requires agents to discover how unfamiliar environments work, reuse what they learn across tasks, and keep that experience current as environments evolve. Existing benchmarks increasingly study experience reuse and adaptation, but provide limited insight into which parts of accumulated experience agents revise, retain, or lose. To address this gap, we introduce ReScopeBench, a benchmark for the experience lifecycle across 18 interactive environments. It distinguishes structural experience about how an environment is organized and accessed from rule experience about reusable principles and constraints that guide decisions across tasks. To study how experience is acquired, reused, and revised, we propose an experience-first construction approach. It builds task streams around shared experience and in
    
[^197]: 具有前瞻能力的Jev：通过语言世界模型实现安全的语言引导自动驾驶

    Jev with Foresight: Safe Language-Guided Autonomous Driving via a Language World Model

    [https://arxiv.org/abs/2611.05448](https://arxiv.org/abs/2611.05448)

    该论文提出Jev-AD系统，通过将基于概率回答的类型化决策模型Jev与能预测候选动作后果的语言世界模型相结合，实现了前瞻式安全决策，将模拟场景碰撞率从26.0%降至2.4%、真实NGSIM数据碰撞率从48.2%降至2.8%，同时能理解并实现乘客的自由形式驾驶指令。

    

    大型语言模型能够理解乘客的驾驶指令，但作为驾驶智能体，它们需要花费数秒钟在生成的文本中进行推理，并且必须从场景中隐式推断每个动作的后果。我们提出了Jev-AD，它将Jev（一种类型化决策模型，以概率而非生成文本的形式回答预定义问题）与语言世界模型相结合，该世界模型为每个候选动作预测附近车辆在未来两秒内的运动。代码将这些预测转化为可预见的结果，Jev可以直接比较这些结果，而无需从坐标中推断。在模拟高速公路汇入场景（500个车辆回合）中，这种前瞻能力将同一决策模型的碰撞率从26.0%降至2.4%，每次决策仅需0.25秒，并且Jev-AD在93%的自由形式指令试验中实现了乘客的意图。在500个重放的真实世界NGSIM回合中，前瞻能力将碰撞率从48.2%降至2.8%，并且从这些数据中学习得到的世界模型……（摘要在此处被截断）

    arXiv:2611.05448v1 Announce Type: cross  Abstract: Large language models understand passengers' driving commands, but as driving agents they reason in generated text for seconds and must infer each action's consequences implicitly from the scene. We present Jev-AD, which pairs Jev, a typed decision model that answers predefined questions with probabilities, not generated text, with a language world model that predicts for every candidate action how nearby vehicles move over the next two seconds. Code turns these predictions into foreseen consequences, which Jev compares instead of inferring them from coordinates. In simulated highway merging (500 car-episodes), this foresight lowers the same decision model's collision rate from 26.0% to 2.4% at 0.25 seconds per decision, and Jev-AD achieves the passenger's intent in 93% of free-form command trials. In 500 replayed real-world NGSIM episodes, foresight lowers the collision rate from 48.2% to 2.8%, and the world model, learned from this t
    
[^198]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^199]: SurgTransfer：面向手术世界模型中动作迁移的多步潜在动作建模

    SurgTransfer: Multi-Step Latent Action Modeling for Action Transfer in Surgical World Model

    [https://arxiv.org/abs/2611.05443](https://arxiv.org/abs/2611.05443)

    本文提出SurgTransfer框架，通过多步潜在动作建模从无标注手术视频中学习上下文不变的潜在动作，从而在无需真实运动学数据或专家标注的情况下实现手术世界模型中的专家动作迁移。

    

    手术世界模型有潜力提供术中指导、加速无风险训练，并促进专家手术操作的迁移。然而，利用世界模型来迁移手术专长在很大程度上尚未被探索。此外，该领域近期的方法主要依赖于特定任务的真实运动学动作或专家标注，这限制了其在缺乏真实标注的狭窄任务范围之外的应用。相比之下，自然图像领域的现有工作在仅从无标注视频中学习潜在动作并跨上下文迁移所学动作方面已展现出令人鼓舞的成果。我们观察到，这一范式对于手术领域而言是短视的，无法捕捉手术过程的层次化与目标导向结构。因此，我们提出了一种新颖的多步动作学习框架，用于学习上下文不变的潜在动作，从而（摘要至此截断）。

    arXiv:2611.05443v1 Announce Type: cross  Abstract: Surgical world models have the potential to provide intra-operative guidance, accelerate risk-free training and facilitate transfer of expert surgical maneuvers. However, the use of world models for transferring surgical expertise has been largely unexplored. Moreover, recent approaches in this domain have mainly relied on ground truth kinematics actions or expert annotations for specific tasks, limiting their applicability beyond the narrow range of tasks with ground-truth annotations. In contrast, existing works in the natural domain have shown promising results in learning latent actions solely from unlabeled videos and transferring the learned actions across contexts. We observe that this paradigm is short-sighted for the surgical domain, failing to capture the hierarchical and goal-directed structure of surgical procedures. Thus, we propose a novel multi-step action learning framework to learn context-invariant latent actions that
    
[^200]: 面向谱图生成的保约束流匹配

    Constraint-Preserving Flow Matching for Spectral Graph Generation

    [https://arxiv.org/abs/2611.05436](https://arxiv.org/abs/2611.05436)

    提出了LaSFM方法，在谱图生成过程中通过在约束Stiefel流形上进行流匹配，同时保持拉普拉斯零空间约束、特征向量正交性和正特征值，并可扩展至属性引导的图生成。

    

    谱图生成通过图拉普拉斯矩阵的特征值和特征向量来建模图，为直接图构建提供了一种紧凑且连续的替代方案。然而，这些特征系统必须保留图拉普拉斯矩阵的结构性质。现有方法虽然保留了特征向量的正交归一性，但并未强制执行拉普拉斯零空间约束，可能产生与拉普拉斯结构不一致的特征系统。为解决这一问题，我们提出了拉普拉斯约束Stiefel流匹配，在生成过程中同时保持零空间条件、特征向量正交归一性以及正特征值。我们刻画了底层的约束Stiefel流形，并在其上构建流匹配。我们进一步将LaSFM扩展到属性引导的生成，在保持这些约束的同时，将样本引导至期望的图属性。实验表明在谱生成任务上取得了性能提升。

    arXiv:2611.05436v1 Announce Type: cross  Abstract: Spectral graph generation models graphs through the eigenvalues and eigenvectors of their Laplacians, providing a compact and continuous alternative to direct graph construction. However, these eigensystems must retain the structural properties of graph Laplacians. Existing approaches preserve eigenvector orthonormality but do not enforce the Laplacian nullspace constraint, potentially producing eigensystems inconsistent with Laplacian structure. To address this, we introduce Laplacian-Constrained Stiefel Flow Matching (LaSFM), which preserves the nullspace condition alongside eigenvector orthonormality and positive eigenvalues during generation. We characterize the underlying constrained Stiefel manifold and formulate flow matching on it. We further extend LaSFM to property-guided generation, steering samples toward desired graph properties while maintaining these constraints. Experiments demonstrate improved performance on both spect
    
[^201]: 基于大语言模型的游戏非玩家角色（NPC）：一项系统性映射研究

    LLM-based NPCs in Video Games: A Systematic Mapping Study

    [https://arxiv.org/abs/2611.05427](https://arxiv.org/abs/2611.05427)

    该论文通过对42篇相关文献的系统性映射研究，首次从组件采用率、延迟、安全覆盖和可复现性四个维度对基于大语言模型的游戏NPC领域进行了量化刻画。

    

    几十年来，非玩家角色（NPC）一直由基于规则的系统驱动，其个性和互动在游戏发布前就被预先编写好。大语言模型（LLM）有望使NPC能够进行开放式对话、适应新情况，并表现出开发者未完全预设的行为，这引发了一波研究和商业实验的浪潮。然而，该领域目前缺乏一个统一的图景，来展示已经构建了什么、效果如何以及代价是什么。我们通过对学术文献中377篇论文的系统性映射研究来弥补这一空白，最终筛选出42篇论文，围绕一个主要研究问题和十个子问题、横跨四个主题维度进行分析。我们首次提供了组件采用率、延迟、安全覆盖和可复现性的量化刻画：短期记忆和世界状态接地已接近普及（60%和69%），而长期记忆（31%）、反思（38%）和检索增强生成（14%）仍然……

    arXiv:2611.05427v1 Announce Type: cross  Abstract: For decades, non-player characters (NPCs) have been driven by rule-based systems whose personalities and interactions are scripted before release. Large language models (LLMs) promise NPCs that converse openly, adapt to new situations, and behave in ways developers did not fully author, prompting a wave of research and commercial experimentation. Yet the field lacks a unified picture of what has been built, how well it works, and at what cost. We address this gap with a systematic mapping study of 377 papers from the academic literature, yielding 42 papers analysed against a main research question and ten sub-questions across four thematic axes. We provide the first quantitative characterisation of component adoption, latency, safety coverage, and reproducibility: short-term memory and world-state grounding are near-universal (60% and 69%), whereas long-term memory (31%), reflection (38%), and retrieval-augmented generation (14%) remai
    
[^202]: CAST-Bench：在多镜头音视频生成中将每个声音追溯至其声源

    CAST-Bench: Tracing Every Sound to Its Source in Multi-Shot Audio-Video Generation

    [https://arxiv.org/abs/2611.05422](https://arxiv.org/abs/2611.05422)

    提出CAST-Bench基准，将T2AV生成的正确性建模为“音视频契约”，通过9个声源类别共900个提示词，将多镜头音视频生成中的每个声音追溯至其预期声源，以诊断声音缺失、错误归因声源及镜头切换中声音丢失等问题。

    

    现有的文本到音视频（T2AV）联合生成基准主要评估感知质量和音视频对齐程度，但在声源采样的均衡性以及按镜头结构分别报告结果方面仍然不足。这使得人们难以判断某个特定声音是缺失了、被归因于错误的声源，还是在镜头切换过程中丢失。为解决这一问题，我们提出了CAST-BENCH，这是一个旨在多镜头音视频生成中将每个生成声音追溯回其预期声源的基准。其核心思想是将T2AV的正确性表述为一种“音视频契约”：每个被要求的声音必须由正确的声源发出、与相应的视觉动作同步，并携带正确的语义内容。具体而言，该基准将900个提示词组织为9个声源类别，每个类别100个提示词，涵盖语音、动物、乐器、环境声音及其他发声体类型。

    arXiv:2611.05422v1 Announce Type: cross  Abstract: Existing benchmarks for joint text-to-audio-video (T2AV) generation evaluate perceptual quality and audiovisual alignment, but balanced sound-source sam?pling and separate reporting by shot structure remain limited. This makes it dif?ficult to determine whether a specific sound is missing, attributed to the wrong source, or lost across shot transitions. To address this, we introduce CAST?BENCH, a benchmark designed to trace every generated sound back to its in?tended source in multi-shot audio-video generation. The core idea is to formu?late T2AV correctness as an audiovisual contract in which each requested sound must be emitted by the right source, synchronized with the corresponding visual action, and carry the right semantic content. Concretely, the benchmark orga?nizes 900 prompts into 9 sound-source categories, with 100 prompts per cate?gory, spanning speech, animals, instruments, ambient sounds, and other emitter types. The prom
    
[^203]: PEEK：面向扩散采样的单步前瞻曝光偏差校正

    PEEK: One-Step-Look-Ahead Exposure Bias Correction for Diffusion Sampling

    [https://arxiv.org/abs/2611.05407](https://arxiv.org/abs/2611.05407)

    PEEK提出了一种无需训练的确定性采样器插件，通过在每个去噪步骤中进行单步前瞻预测并插值，从更高信号的噪声状态获得更接近真实图像的预测，从而校正扩散采样中的曝光偏差并提升图像生成质量。

    

    扩散图像生成器之所以成功，得益于精确的去噪网络和强大的数值采样器。然而这两个组件并未完全匹配。在采样过程中，去噪器基于与真实数据差异较大的有缺陷输入来预测下一步。这种训练-测试差距，即曝光偏差，限制了图像生成质量（例如图像模糊、缺乏细节或形状怪异）。为了缓解这一问题，我们提出了PEEK，一种新颖的无需训练的确定性采样器插件。在每个符合条件的去噪步骤中，PEEK首先执行一次临时的单步更新，在前瞻状态下评估一次去噪器，并将该结果与一步之后的预测进行插值以形成下一个噪声状态。通过采取这一额外步骤，我们能够从更高信号的噪声状态获得通常更接近真实图像的预测，从而缓解曝光偏差。为了优化插值位置，PEEK对插值调度进行了搜索

    arXiv:2611.05407v1 Announce Type: cross  Abstract: Diffusion image generators are successful due to accurate denoising networks and strong numerical samplers. Yet the two components are not fully matched. During sampling, the denoiser predicts for the next step based on flawed input that differ much from the real data. This train-test gap, or exposure bias, caps image generation quality (i.e., blurry, lacking in details or awkward in shape). To mitigate the problem, we propose PEEK, a novel training-free deterministic sampler plugin. At each eligible denoising step, PEEK first performs a provisional one-step update, evaluates the denoiser once at the look-ahead state, and interpolates it with the one-step-later prediction to form the next noisy state. By taking the additional step, we obtain a generally more-image-like prediction from the higher-signal noisy state, mitigating the exposure bias. To optimize the interpolation positions, PEEK searches for an interpolation schedule search 
    
[^204]: 何时共享才有效？在图像、视频、音频与视听表征间扩展共享的SigLIP2

    When Does Sharing Work? Scaling a Shared SigLIP2 Across Image, Video, Audio, and Audio-Visual Representations

    [https://arxiv.org/abs/2611.05406](https://arxiv.org/abs/2611.05406)

    该论文提出Omni-Encoder，通过全参数共享的单一SigLIP-2 Transformer进行端到端联合训练，并借助一维卷积音频分词器和基于绝对时间戳参数化的多模态旋转位置嵌入，首次将图像、视频、音频和视听输入统一编码到共享的时空表征空间中。

    

    对比预训练已经推动了图像-文本理解的发展，然而图像、视频和音频编码器仍然是独立训练的，跨模态对齐主要通过事后绑定到冻结的锚点空间来实现。我们提出了Omni-Encoder，这是一个通用的多模态编码器，能够在单个SigLIP-2视觉Transformer中以完全共享的参数处理图像、视频、音频和视听输入。与以锚点为中心的绑定或针对每个模态的适配器不同，所有模态都以端到端的方式联合优化，且该框架从输入到训练进行了系统性组织。模态原生的输入阶段首先将每种信号映射到共享嵌入空间，其中包括一个一维卷积音频分词器，它将梅尔频谱图转换为与视觉patch对齐的token。随后，由绝对时间戳参数化的多模态旋转位置嵌入将视觉帧和音频片段置于统一的时空坐标系中。

    arXiv:2611.05406v1 Announce Type: cross  Abstract: Contrastive pretraining has advanced image-text understanding, yet image, video, and audio encoders are still trained independently, with cross-modal alignment realized mainly through post-hoc binding to a frozen anchor space. We present Omni-Encoder, a general multimodal encoder that processes image, video, audio, and audio-visual inputs within a single SigLIP-2 Vision Transformer with fully shared parameters. Unlike anchor-centric binding or per-modality adapters, all modalities are optimized jointly end-to-end, and the framework is organized from input to training. A modality-native input stage first maps each signal into the shared embedding space, including a 1D-convolutional audio tokenizer that converts mel-spectrograms into tokens aligned with visual patches. A multimodal rotary position embedding parameterized by absolute timestamps then places visual frames and audio segments in a unified spatiotemporal coordinate system. Fin
    
[^205]: VenusGenome：面向微生物基因组表示学习的基因组感知掩码自编码器

    VenusGenome: Genome-Aware Masked Autoencoding for Microbial Genome Representation Learning

    [https://arxiv.org/abs/2611.05405](https://arxiv.org/abs/2611.05405)

    提出VenusGenome，一种基因分辨率的基因组感知掩码自编码器，通过高比例掩码与融合重叠群边界、链方向和基因位置的注意力机制，在超过170万个微生物基因组上预训练，以提升微生物基因组表示学习和功能预测能力。

    

    基因组上下文模型已经表明，为蛋白质嵌入提供上下文信息可以改善微生物功能的预测。我们观察到，一个基因的蛋白质嵌入可以从基因组中相邻基因的嵌入中部分预测出来，这促使我们采用高比例掩码策略，使模型必须从更稀疏的基因组上下文中进行重建。此外，基因组建模还必须区分真实的生物学关系与组装约定，因为独立重叠群（contig）之间的排序并不意味着物理上相邻，而链方向标签依赖于参考方向。为此，我们提出了VenusGenome，一个基因分辨率的掩码自编码器，它将仅可见编码与基因组感知注意力机制相结合，后者纳入了重叠群边界、相对链方向以及基因位置信息。我们在一个精选的语料库上预训练该模型，该语料库包含来自分离株和宏基因组来源的超过170万个微生物基因组组装，并使用冻结的蛋白质嵌入作为基因级输入。

    arXiv:2611.05405v1 Announce Type: cross  Abstract: Genomic-context models have shown that contextualizing protein embeddings can improve predictions of microbial function. We observe that a gene's protein embedding is partially predictable from the embeddings of neighboring genes in the genome, motivating high-ratio masking that requires reconstruction from sparser genomic context. Genomic modeling must also distinguish biological relationships from assembly conventions, since the ordering of separate contigs does not imply adjacency and strand labels depend on the reference orientation. We therefore introduce VenusGenome, a gene-resolution masked autoencoder that combines visible-only encoding with genome-aware attention incorporating contig boundaries, relative strand orientations, and gene positions. We pretrain this model on a curated corpus of over 1.7 million microbial genome assemblies from isolate and metagenomic sources, using frozen protein embeddings as gene-level inputs. Ac
    
[^206]: 一种面向冻结语言模型中DNA的免训练套接字

    A Training-Free Socket for DNA in Frozen Language Models

    [https://arxiv.org/abs/2611.05401](https://arxiv.org/abs/2611.05401)

    该论文提出一种免训练的“套接字”方法，仅通过一次闭式岭回归将冻结的Evo 2 DNA特征接入冻结LLM的早期层表示，无需任何梯度训练即可在四个宿主模型上有效回答基因级别问题，真实DNA信号均显著优于打乱DNA对照。

    

    将DNA基础模型连接到大型语言模型（LLM）通常需要训练一个连接器，并且往往还需要训练LLM本身。我们证明，对于基因级别的问题，一个闭式映射就足够了。我们的方法称为“套接字”（socket），它通过拟合一次岭回归（即“插头”（plug）），将冻结的Evo 2特征映射到宿主LLM在早期层对基因符号、名称和描述的自身表示。插头的输出替换占位符位置的隐藏状态，且没有任何梯度回传到宿主模型。我们在GeneHop上测试了该方法，这是一个按同源性划分的五选一多项选择基准。在冻结分析方案下对留存测试集进行单次运行的结果显示，真实DNA在所有四个宿主模型上都优于打乱后的DNA：在Qwen3.5-4B上提升+22.2分[19.8, 25.0]，在Qwen3.6-27B上提升+26.1分[24.3, 27.9]，在Mistral-7B上提升+16.7分[14.2, 19.7]，在OLMo-2-7B上提升+18.4分[16.2, 21.0]。所有四个结果均通过Holm校正，每个零假设对照都保持在假处理水平两点以内……

    arXiv:2611.05401v1 Announce Type: cross  Abstract: Connecting a DNA foundation model to a large language model (LLM) usually requires training a connector, and often the LLM too. We show that a closed-form map is enough for gene-level questions. Our method, the socket, fits one ridge regression (the plug) from frozen Evo 2 features to the host LLM's own representation of the gene's symbol, name, and description at an early layer. The plug's output replaces the hidden state at placeholder positions, and no gradient passes through the host.   We test it on GeneHop, a five-way multiple-choice benchmark split by homology. In a single run on the held-out test set, under a frozen analysis plan, real DNA beats shuffled DNA on all four hosts: by +22.2 points [19.8, 25.0] on Qwen3.5-4B, +26.1 [24.3, 27.9] on Qwen3.6-27B, +16.7 [14.2, 19.7] on Mistral-7B, and +18.4 [16.2, 21.0] on OLMo-2-7B. All four survive Holm correction, every null control stays within two points of the sham level, and the d
    
[^207]: 置信度门控分层式LLM推理的可行性条件

    A Feasibility Condition for Confidence-Gated Tiered LLM Inference

    [https://arxiv.org/abs/2611.05394](https://arxiv.org/abs/2611.05394)

    本文提出了一个由无信号下界和完美信号上界界定的闭式可行性条件，通过架构成本比κ刻画了置信度门控分层LLM级联推理何时优于将所有查询直接发送到后备模型。

    

    分层推理将大语言模型的计算分布在一个设备层级结构中，从端侧和边缘硬件到数据中心，并且日益被视为实现更廉价、更低延迟和更隐私保护的服务途径。大多数此类系统协调的是冻结的模型检查点，并且大多依靠前置部署模型产生的置信度信号来决定哪些查询在本地解决、哪些需要升级。我们给出了一个闭式条件，用于判断何时该信号上的任何阈值都无法使分层级联相对于将所有查询直接发送给后备模型更有价值。通过将路由决策建模为二元测试，并借助架构成本比 κ 来区分本地成本与升级成本，我们推导出了一个可行性条件，该条件由一个无信号下界所界定——低于该下界时，级联在任何信号质量下都可行；同时由一个完美信号上界所界定——高于该上界时，级联不可行。沿 κ 轴来看，该条件会发生反转：在本地计算……

    arXiv:2611.05394v1 Announce Type: new  Abstract: Tiered inference distributes large language model computation across a hierarchy of devices, from on-device and edge hardware to the datacenter, and is increasingly offered as the route to cheaper, lower-latency, and more private serving. Most such systems orchestrate frozen model checkpoints, and most decide what to resolve locally and what to escalate using a confidence signal produced by a forward-deployed model. We give a closed-form condition for when no threshold on that signal makes a tiered cascade worthwhile relative to sending every query to the backstop. Modeling the routing decision as a binary test and separating local from escalation cost through an architecture cost ratio $\kappa$, we derive a feasibility condition bracketed by a signal-free floor, below which a cascade is feasible at any signal quality, and a perfect-signal ceiling, above which none is. Read along the $\kappa$ axis the condition inverts: where local compu
    
[^208]: 面向医疗多智能体推理的挑战感知知识招募与元智能体监督方法

    Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning

    [https://arxiv.org/abs/2611.05393](https://arxiv.org/abs/2611.05393)

    该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。

    

    大语言模型在标准医疗问答任务上表现强劲，但在更困难的推理任务上性能会下降，而多智能体辩论在此类任务上往往无法超越强大的单模型提示方法。我们认为一个关键局限在于：现有系统仅根据问题本身来选择专家和检索知识，而不是根据推理过程中涌现出的知识缺口。我们提出挑战感知的自适应招募机制：一个外部的“挑战者”检查不断演化的推理轨迹，并识别出具体的知识缺口或专业知识缺口。这些信号会触发针对性的知识检索，或在讨论中途招募领域专家。在MEDAGENTSBENCH困难子集（涵盖九个生物医学数据集的862个问题）上，事前检索几乎不带来收益，而挑战感知的推理中间检索表现更好；在多专家骨干系统上，该方法在9个数据集中的8个上提升了准确率，相比仅使用挑战者的方案提升了6个百分点。消融实验表明两者均有收益……

    arXiv:2611.05393v1 Announce Type: new  Abstract: Large language models perform strongly on standard medical QA but degrade on harder reasoning tasks, where multi-agent debate often fails to beat strong single-model prompting. We argue that a key limitation is that existing systems select experts and retrieve knowledge from the question alone, rather than from gaps that emerge during reasoning. We propose challenge-aware adaptive recruitment: an external CHALLENGER inspects the evolving reasoning trace and identifies specific factual or expertise gaps. These signals trigger targeted knowledge retrieval or mid-discussion specialist recruitment. On the MEDAGENTSBENCH hard subset (862 questions across nine biomedical datasets), upfront retrieval provides little benefit, while challenge-aware in-between retrieval performs better; on a multi-specialist backbone it improves accuracy on 8 of 9 datasets and adds 6 percentage points over Challenger Only. Ablations indicate benefits from both ret
    
[^209]: 面向智能体AI系统的不确定性感知资源编排

    Uncertainty-Aware Resource Orchestration for Agentic AI Systems

    [https://arxiv.org/abs/2611.05392](https://arxiv.org/abs/2611.05392)

    本文提出了一种不确定性感知的资源编排方法，解决大语言模型驱动的智能体AI系统因运行时资源需求呈重尾、多峰且不可预测而导致的调度难题，避免资源供给不足或过度供给。

    

    由大语言模型（LLM）驱动的智能体人工智能（AI）系统以自主的多步骤流水线形式执行，动态调用工具、将观测结果整合进记忆，并不断迭代，直到任务完成或资源预算耗尽。与具有固定图结构的经典工作流不同，此类流水线的总体计算、带宽和端到端延迟需求在调度时是未知的：它们在运行时由中间观测结果和随机token采样涌现而来，形成重尾、多峰的随机变量。这种不可预测性使基础设施编排器陷入两难：资源供给不足可能导致执行中途限流和违反截止期限，而过度供给则会浪费稀缺的边缘或云端容量。现有调度器需要完全已知的流水线图，而边缘资源管理方法依赖参数化或平稳的需求模型，两者都不适用于未知的、非平稳的需求特性……

    arXiv:2611.05392v1 Announce Type: new  Abstract: Agentic artificial intelligence (AI) systems powered by large language models (LLMs) execute as autonomous multi-step pipelines that dynamically invoke tools, integrate observations into memory, and iterate until a task is resolved or a resource budget is exhausted. Unlike classical workflows with fixed graphs, the aggregate compute, bandwidth, and end-to-end latency demands of such pipelines are unknown at dispatch time: they emerge at runtime from intermediate observations and stochastic token sampling, yielding heavy-tailed, multi-modal random variables. This unpredictability forces infrastructure orchestrators into a dilemma: under-provisioning risks mid-execution throttling and deadline violations, while over-provisioning wastes scarce edge or cloud capacity. Existing schedulers require a fully known pipeline graph, while edge resource-management methods rely on parametric or stationary demand models, and neither fits the unknown, n
    
[^210]: 智能体执行框架感知：将经验证的遥测数据转化为工具调用智能体的运行控制

    Agent Harness Awareness: Turning Verified Telemetry into Operational Control for Tool-Using Agents

    [https://arxiv.org/abs/2611.05391](https://arxiv.org/abs/2611.05391)

    提出名为“智能体执行框架感知”的信任分离设计，由宿主在关键操作前向模型提供经验证的最新运行状态快照，弥补模型对执行框架运行信息的“运行盲区”，使三种主流模型上的平均可观测策略遗憾降低96.9%。

    

    工具描述告诉智能体某个操作能够做什么，但不一定揭示该操作在使用时刻是否健康或适用。执行框架通常通过监控器、追踪记录和近期故障掌握这些信息，而选择下一步操作的模型却无从得知。我们将这一差距称为“运行盲区”。我们提出“智能体执行框架感知”，这是一种信任分离的设计：在执行关键操作之前，由宿主向模型提供一份小型、最新且经过验证的状态快照，同时保留测量、强制执行、评估和最终分发的权限。我们使用ToolRoute测试其工具与网络实例，ToolRoute是一个受控隔离基准，它在保持任务和候选语义不变的情况下，仅改变模型可获得的运行信息。在涵盖GPT-5.6 Sol、Claude Sonnet 5和Gemini 3.7 Flash的216次决策消融实验中，经验证的工具健康状态使平均可观测策略遗憾降低了96.9%。

    arXiv:2611.05391v1 Announce Type: new  Abstract: Tool descriptions tell an agent what an action can do, but they do not necessarily reveal whether that action is healthy or suitable at the moment of use. Execution harnesses often possess this information through monitors, traces, and recent failures while the model choosing the next action does not. We call this gap operational blindness. We introduce Agent Harness Awareness, a trust-separated design in which the host supplies a small, fresh, verified state snapshot before a consequential action while retaining measurement, enforcement, evaluation, and final dispatch authority. We test its tools-and-network instance with ToolRoute, a controlled isolation benchmark that holds the task and candidate semantics constant and varies the operational information available to the model. In a 216-decision ablation across GPT-5.6 Sol, Claude Sonnet 5, and Gemini 3.7 Flash, verified tool-health state reduced mean observable policy regret by 96.9% 
    
[^211]: 谁有权推翻智能体？LLM审查机构中的证据门控授权

    Who May Overrule the Agent? Evidence-Gated Authority in LLM Review Institutions

    [https://arxiv.org/abs/2611.05390](https://arxiv.org/abs/2611.05390)

    本文提出一种“证据门控”审查权限机制，要求LLM审查者必须先产生约束违反与合规修复的有效记录才能推翻上游智能体的提案，实验证明该机制能在大幅减少误改正确决策的同时保留大部分错误修复能力。

    

    多智能体语言模型系统通常会添加一个审查者来改进上游智能体的决策。然而，审查者既能修复错误，也可能替换原本正确的决策。我们将此作为一个机构管辖权问题来研究：在相同的提案、任务信息、模型和审查输出的条件下，审查者何时被授权使其建议具有约束力？我们比较了宽泛的推翻权限与证据门控机制——后者只有在审查者生成关于约束性规则违反及合规修复的机械有效记录后，才允许替换提案。在两个本地运行的开源权重模型、96个空间策略任务以及一个96任务的投资组合迁移基准测试中，该门控机制显著减少了对正确提案的破坏，同时保留了对违规行为的大多数修复。这种权衡的方向部分由规则本身决定；其幅度则取决于审查者的状态转换和上游提案的状态。

    arXiv:2611.05390v1 Announce Type: new  Abstract: Multi-agent language-model systems often add a reviewer to improve an upstream agent's decision. Yet a reviewer can both repair errors and replace decisions that were already correct. We study this as a problem of institutional jurisdiction: conditional on the same proposal, task information, model, and review output, when is the reviewer authorized to make its recommendation binding? We compare broad override authority with an evidence gate that permits replacement only after the reviewer produces a mechanically valid record of a binding-constraint violation and a compliant repair. Across two locally run open-weight models, 96 spatial-policy tasks, and a 96-task portfolio-transfer benchmark, the gate sharply reduces corruption of correct proposals while retaining most repairs of violations. The directions of this tradeoff are partly implied by the rule; its magnitude depends on reviewer transitions and upstream proposal states. A prospe
    
[^212]: AI驱动的SaaS平台集成：AI Express统一生成式AI服务——基于按量付费积分系统与可扩展微服务架构

    AI-Driven SaaS Platform Integration: AI Express for Unified Generative AI Services with Pay-As-You-Go Credit Systems and Scalable Microservice Architecture

    [https://arxiv.org/abs/2611.05389](https://arxiv.org/abs/2611.05389)

    AI Express通过按量付费积分系统和可扩展微服务架构，将碎片化的生成式AI能力（代码审查、文本转网站、语法检查等）整合到单一SaaS平台中，使用户无需多重订阅和技术专长即可低成本使用统一的AI服务。

    

    当前的生成式AI服务呈现碎片化状态，文本生成、图像合成和代码辅助等功能需要分别订阅，导致成本高昂且工作流程频繁中断。AI Express是一个SaaS平台，将这些能力整合为统一的按量付费服务。该平台提供一系列AI工具，包括用于评估GitHub代码仓库的核心代码审查器、可将文本转换为完整功能网站的文本转网站生成器，以及脚本生成、语法检查、数据分析和水印去除等功能，所有工具均可通过单一界面访问，无需任何专业技术知识。

    arXiv:2611.05389v1 Announce Type: cross  Abstract: Generative AI is currently fragmented, with separate subscriptions required for text generation, image synthesis, and code assistance, leading to high costs and workflow disruptions. AI Express is a SaaS platform that consolidates these capabilities into a unified pay-as-you-go service. The platform provides a range of AI tools, including a Core Code Reviewer that evaluates GitHub repositories, a Text-to-Website generator that converts text into fully functional websites, script generation, grammar checking, data analysis, and watermark removal, all accessible through a single interface without requiring technical expertise.
    
[^213]: RoboTryOn：通过虚拟试穿从人类视频合成灵巧机器人演示

    RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On

    [https://arxiv.org/abs/2611.05163](https://arxiv.org/abs/2611.05163)

    RoboTryOn 将人机转换类比为虚拟试穿——保持人类演示的动作不变，将人替换为由参考图像指定的目标机器人——从而从人类视频合成出视觉一致、保留任务与场景的灵巧机器人演示视频，为高自由度机器人提供可扩展的演示数据来源。

    

    机器人基础模型在机器人控制方面已展现出卓越性能，但将其适配到新任务时仍严重依赖遥操作演示。第一人称视角的人类视频提供了一种可扩展的替代方案，然而无论是将渲染的机器人叠加到人类视频上，还是将人类姿态重定向为机器人动作，对于高自由度的机器人本体而言都会留下显著的视觉或动作差距。为解决这一局限，我们提出了 RoboTryOn，它将“人到机器人”的转换视为虚拟试穿问题：试穿是保持人物不变、更换服装，而 RoboTryOn 则是保持演示的动作不变、将人类替换为机器人。具体而言，RoboTryOn 将一段人类视频转换为由参考图像指定的目标机器人的视频，同时保留原始任务、物体和背景。为了保持视觉一致性，我们以自回归方式对选定的关键帧进行转换，并用它们来引导视频合成，运动以人类（此处摘要内容不完整）

    arXiv:2611.05163v2 Announce Type: replace-cross  Abstract: Robot foundation models have shown remarkable performance in robot control, yet adapting them to a new task still relies heavily on teleoperated demonstrations. Egocentric human videos offer a scalable substitute, but overlaying rendered robots on human videos or retargeting human poses into robot actions leaves significant visual or action gaps for high-DoF robot embodiments. To address this limitation, we propose RoboTryOn, which treats human-to-robot translation as virtual try-on: try-on keeps the person and swaps the garment, RoboTryOn keeps the demonstrated motion and swaps the human for the robot. Specifically, RoboTryOn converts a human video into a video of the target robot specified by a reference image, while preserving the original task, objects, and background. To maintain visual consistency, we translate selected keyframes autoregressively and use them to guide video synthesis with motion conditioned on the human h
    
[^214]: FloMo：3D场景流作为从人类视频学习的世界动作模型中间表示

    FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video

    [https://arxiv.org/abs/2611.05062](https://arxiv.org/abs/2611.05062)

    FloMo提出以3D场景流作为视频预训练迁移到机器人动作预测的中间表示，通过将场景流渲染为RGB并与视频共享潜在空间，使50亿参数的视频生成模型能够联合利用无动作标注的人类视频和机器人遥操作数据进行训练，从而在未见过的物体、运动基元和任务上实现更好的泛化。

    

    预训练视频生成模型能够将互联网规模的运动先验蒸馏到机器人策略中，但完整视频对于动作预测而言是一个冗余信号。我们认为，密集的3D运动是视频预训练最佳迁移到机器人动作的媒介，并且预测运动而非视频能为动作预测带来更好的样本效率。FloMo是一个50亿参数的视频生成模型，经过微调以预测3D场景流和动作token。将场景流渲染为RGB并将其编码到与视频相同的潜在空间中，使预训练的视频生成主干网络能够高效地适应预测3D场景流。通过在无动作标注的人类视频和有动作标注的机器人遥操作数据上联合训练，我们的模型受益于人类视频预训练，并能泛化到未见过的物体、运动基元和任务。用视频预测目标替代运动预测会降低分布外泛化能力，而添加……

    arXiv:2611.05062v2 Announce Type: replace-cross  Abstract: Pretrained video generation models distill internet-scale motion priors into robot policies, but full video is a redundant signal for action prediction. We argue that dense 3D motion is the medium through which video pretraining best transfers to robot actions, and that predicting motion rather than video affords better sample-efficiency for action prediction. FloMo is a 5B video-generation model fine-tuned to predict 3D scene-flow and action tokens. Rendering scene flow as RGB and encoding it into the same latent space as video enables a pretrained video generation backbone to efficiently adapt to predict 3D scene flow. Co-trained on action-free human video and action-labeled robot teleoperation data, our model benefits from human-video pretraining and generalizes to unseen objects, motion primitives, and tasks. Replacing motion prediction with a video prediction objective degrades out-of-distribution generalization, while add
    
[^215]: ScamWorld：一个用于评估网络智能体应对在线诈骗的交互式环境

    ScamWorld: An Interactive Environment for Evaluating Web Agents Against Online Fraud

    [https://arxiv.org/abs/2611.04879](https://arxiv.org/abs/2611.04879)

    本文提出 ScamWorld，一个通过在多场景、多机制、多语言下重建真实诈骗网站，从而对网络智能体识别与应对在线诈骗能力进行受控、可重复评估的交互式环境。

    

    网络智能体越来越多地代表用户进行网页浏览、购物和账户管理。先前的研究表明，它们可能被对抗性内容误导，包括提示注入和欺骗性界面。除了此类攻击之外，在开放网络上运行的智能体还会遇到旨在欺骗人类的在线诈骗，例如欺诈性网店、虚假投资平台和预付费骗局。然而，网络智能体在任务执行过程中如何应对此类诈骗仍缺乏充分研究，部分原因在于诈骗网站寿命短暂，使系统性评估难以开展。为填补这一空白，我们提出了 ScamWorld——一个交互式环境，它能够在多样化的场景、诈骗机制和语言下重建已被举报的诈骗网站。每个重建的网站都复现了有据可查的诈骗机制，使相关诈骗迹象在智能体采取最终行动之前即可被观察到，从而实现受控且可重复的评估。

    arXiv:2611.04879v2 Announce Type: replace  Abstract: Web agents increasingly browse, shop, and manage accounts on behalf of their users. Prior work has shown that they can be misled by adversarial content, including prompt injections and deceptive interfaces. Beyond such attacks, agents operating on the open web also encounter online fraud designed to deceive humans, including fraudulent online stores, fake investment platforms, and advance-fee schemes. How web agents respond to such fraud during task execution remains underexplored, in part because scam websites are short-lived, making systematic evaluation difficult. To address this gap, we introduce ScamWorld, an interactive environment that reconstructs reported scam websites across diverse scenarios, fraud mechanisms, and languages. Each reconstructed website reproduces documented fraud mechanisms, making relevant signs of fraud observable before the agent's final action and enabling controlled, repeatable evaluation. Through an e
    
[^216]: AgenticFinBench：基于真实世界请求的多样化任务金融智能体基准测试

    AgenticFinBench: Benchmarking Financial Agents on Diverse Tasks from Real-World Requests

    [https://arxiv.org/abs/2611.04813](https://arxiv.org/abs/2611.04813)

    该论文提出了 AgenticFinBench——一个涵盖 8 个类别、2 种语言、4 个复杂度级别、共 230 个真实世界金融任务的智能体基准，并创新性地引入了结合问题特定评分标准与两阶段幻觉审计的无参考评估框架，从而更贴近实际、更准确地衡量金融智能体的真实能力。

    

    金融智能体正在成为现实世界 AI 系统中的一个重要类别。金融任务在类型和难度上差异巨大，且许多任务是开放式的，没有标准答案。然而，现有的基准测试既无法捕捉这种广度和开放性，也没有将信息溯源（grounding）与正确性区分开来。为了弥补这一差距，我们提出了 AgenticFinBench，这是一个包含 230 个真实世界金融任务的基准，涵盖 8 个类别、2 种语言和 4 个复杂度级别。我们还提出了一个无参考（reference-free）的评估框架，该框架将针对具体问题的评分标准（rubrics）与两阶段声明审计相结合，以检测幻觉——其定义为智能体轨迹中不存在且被外部证据反驳的声明。该框架更符合实践中对金融工作的评估方式。在 18 种智能体配置上的实验揭示了三个关键发现：(1) 忽视幻觉可能会夸大智能体的性能：幻觉审计会降低基线成绩……（摘要原文在此处截断）

    arXiv:2611.04813v2 Announce Type: replace  Abstract: Financial agents are becoming an important class of real-world AI systems. Financial tasks vary widely in type and difficulty, and many are open-ended with no ground-truth answers. However, existing benchmarks neither capture this breadth and open-endedness nor separate grounding from correctness. To bridge this gap, we introduce $\textbf{AgenticFinBench}$, a benchmark of 230 real-world financial tasks spanning 8 categories, 2 languages, and 4 complexity levels. We also propose a reference-free evaluation framework that combines question-specific rubrics with a two-stage claim audit that detects hallucinations, defined as claims absent from the agent's trace and refuted by external evidence. This framework better matches how financial work is assessed in practice. Experiments across 18 agent configurations reveal three key findings: (1) Ignoring hallucinations can overstate agent performance: hallucination auditing lowers the baselin
    
[^217]: Signals：AI智能体的持续诊断

    Signals: Continuous Diagnostics of AI Agents

    [https://arxiv.org/abs/2611.04507](https://arxiv.org/abs/2611.04507)

    本文提出Signals系统，通过包含45种问题类型的分类体系对AI智能体执行轨迹进行持续诊断，将发现的问题归类为工程问题并评估严重程度和提供具体补救措施，在公开数据集上达到91.43%的F1分数。

    

    即使在工具和基础设施正常工作的情况下，AI智能体也可能做出错误的决策或浪费资源。它们的不确定性行为使得这些问题难以复现和诊断。我们开发了Signals，用于从智能体执行轨迹中识别此类问题并提出具体的补救措施。我们围绕一个包含八个类别、45种问题类型的分类体系构建了该方法，并为每种类型提供了定义和证据要求。该方法区分已发生的故障、风险暴露和低效行为，将相关发现归类为称为Signals的工程问题，并根据支持证据评估其严重程度。每个Signal都包含开发人员可以采取的行动以及用于评估变更的检查项。进一步的执行可以为现有Signal添加证据，或揭示新的问题。我们在公开的智能体执行记录上对该方法进行了评估，在将Signals与参考问题匹配时达到了91.43%的F1分数。

    arXiv:2611.04507v2 Announce Type: replace  Abstract: AI agents can make incorrect decisions or waste resources even when their tools and infrastructure work normally. Their nondeterministic behavior makes these problems difficult to reproduce and diagnose. We developed Signals to identify such problems from agentic traces and suggest concrete remediations. We built the method around a taxonomy of 45 problem types across eight families, with definitions and evidence requirements for each type. The method distinguishes realized failures, risk exposures, and inefficiencies, groups related findings into engineering issues called Signals, and assesses their severity from the supporting evidence. Each Signal includes actions a developer can take and checks for evaluating the change. Further executions can add evidence to an existing Signal or reveal a new problem. We evaluated the method on public agent executions and achieved 91.43% F1 when matching Signals to reference issues, about 18 per
    
[^218]: 面向混合专家大语言模型端到端FP4强化学习的训练与推理策略对齐

    Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs

    [https://arxiv.org/abs/2611.04376](https://arxiv.org/abs/2611.04376)

    该研究揭示了MoE大模型FP4强化学习失效的根源在于路由器的离散专家选择机制，证明了必须在训练与推理两侧对称地应用相同的恢复技术，才能实现有效的端到端FP4强化学习。

    

    本研究探讨了面向混合专家大语言模型的端到端FP4（四比特浮点）强化学习后训练。近期关于FP4强化学习的工作提出了有效的恢复方案，能够在激进低精度量化下恢复稠密模型的精度，但我们发现同样的方法在MoE场景下相比朴素的FP4基线没有任何提升。我们开展了深入研究以理解其中的原因。我们的分析发现其根本原因在于MoE模型独有的路由机制：由于路由器做出的是离散的专家选择决策，训练与推理策略的对齐对训练端和推理端是否对称地应用相同的恢复技术变得极为敏感。当使用非对称方案时——包括那种仅修正rollout/训练侧的稠密模型方法——训练与推理策略的对齐会急剧恶化（以皮尔逊相关系数衡量）。

    arXiv:2611.04376v2 Announce Type: replace-cross  Abstract: In this work, we study end-to-end FP4 reinforcement learning (RL) post-training for Mixture-of-Experts (MoE) large language models. Recent work on FP4 RL has introduced effective recovery recipes that restore accuracy under aggressive low-precision quantization for dense models, but we find that the same approach yields no improvement over the naive FP4 baseline in the MoE setting. We conduct an in-depth study to understand why. Our analysis identifies the root cause in the routing mechanism unique to MoE: because the router makes discrete expert-selection decisions, training-inference policy alignment becomes acutely sensitive to whether the same recovery technique is applied symmetrically on the training and inference sides. When an asymmetric recipe is used-including the dense-model approach, which corrects only the rollout-training-inference alignment deteriorates sharply, as measured by the Pearson correlation coefficient 
    
[^219]: 从氨基酸序列预测蛋白质任意折叠阶段的结构

    Predicting protein structures at any folding stage from sequence

    [https://arxiv.org/abs/2611.04287](https://arxiv.org/abs/2611.04287)

    FoldScape 是一个以序列为条件的深度学习框架，可在任意折叠阶段生成蛋白质构象集合，无需推理时进行针对特定蛋白质的分子动力学模拟。

    

    蛋白质结构预测的研究大多集中于天然构象，使得未折叠态、折叠中间体与天然结构之间的连续变化过程难以被探究。我们提出了 FoldScape，这是一个以序列为条件的深度学习框架，能够在指定折叠比例 $Q$ 下生成蛋白质构象集合。FoldScape 基于来自 10,000 个蛋白质序列的约一百万个构象进行训练，结合了粗粒度折叠轨迹与全原子机械解折叠轨迹。它通过残基间距离的分层滑动窗口回归、链级组装以及修正流解码器来重建三维 $C_\alpha$ 链。通过以折叠进度作为条件，FoldScape 在推理时无需针对特定蛋白质的分子动力学模拟，即可生成从未折叠态、经中间体直至天然态结构的构象。在匹配样本的比较中，FoldScape 达到了（摘要在此处被截断）

    arXiv:2611.04287v2 Announce Type: replace-cross  Abstract: Protein structure prediction has largely focused on native conformations, leaving the continuum of unfolded states, folding intermediates, and native structures difficult to access. We present FoldScape, a sequence-conditioned deep learning framework for generating protein conformational ensembles at specified folding fractions $Q$. Trained on approximately one million conformations from 10,000 protein sequences, FoldScape combines coarse-grained folding and all-atom mechanical-unfolding trajectories. It reconstructs three-dimensional $C_\alpha$ chains through hierarchical sliding-window regressions of inter-residue distances, chain-level assembly, and a rectified-flow decoder. By conditioning on folding progress, FoldScape generates conformations from unfolded states through intermediates to native-state structures without protein-specific molecular dynamics at inference. In matched-sample comparisons, FoldScape achieves media
    
[^220]: 离散扩散蒸馏的双向中心化得分匹配

    Bidirectional Centered Score Matching for Discrete Diffusion Distillation

    [https://arxiv.org/abs/2611.04279](https://arxiv.org/abs/2611.04279)

    提出双向中心化得分匹配（BCSM），一种面向离散扩散模型的高效位置内蒸馏目标，将CSM的O(|V|²)复杂度降低至O(K)，并通过跨时间步转移对齐在少步蒸馏中实现更优的质量-效率权衡。

    

    具有并行解码的离散扩散模型通常采用逐词元的KL散度目标进行蒸馏。尽管逐词元的KL能够精确匹配相应的乘积形式块分布，但其概率空间的优化几何结构以及仅基于教师模型的候选截断，在激进的多步压缩（few-step）蒸馏场景下可能提供不足的监督信号。具体得分匹配通过在位置内部对齐相对logit几何结构提供了一种替代性目标，但其完整的成对形式会带来 O(|V|²) 的计算复杂度。我们提出双向中心化得分匹配（BCSM），这是一种高效的位置内部蒸馏目标，它构建了一个 O(K) 规模的双向活动集，将成对比较的复杂度降至 O(K²)，并将其分解为 O(K) 的目标函数。我们还进一步引入了跨时间步的转移对齐机制。实验结果表明，该方法在质量与效率之间取得了出色的权衡。

    arXiv:2611.04279v2 Announce Type: replace  Abstract: Discrete diffusion models with parallel decoding are often distilled using token-wise KL objectives. Although token-wise KL exactly matches the corresponding product-form block distributions, its probability-space optimization geometry and teacher-only candidate truncation may provide insufficient supervision under aggressive few-step distillation. Concrete Score Matching (CSM) provides an alternative within-position objective by aligning relative logit geometry, but its complete pairwise form incurs \mathcal{O}(|\mathcal{V}|^2) complexity. We propose Bidirectional Centered Score Matching (BCSM), an efficient within-position distillation objective that constructs an \mathcal{O}(K) bidirectional active set, reduces pairwise comparisons to \mathcal{O}(K^2), and factorizes them into an \mathcal{O}(K) objective. We further introduce cross-timestep transition alignment. Experiments demonstrate strong quality--efficiency trade-offs, includ
    
[^221]: 面向模型合并的公共与独立方向分离

    Separating Common and Unique Directions for Model Merging

    [https://arxiv.org/abs/2611.04108](https://arxiv.org/abs/2611.04108)

    提出CUSD，一种无需数据的模型合并预处理方法，通过将任务向量分解为跨任务共享的公共方向分量和任务独有的独立分量并分别处理，从而提升算术合并方法的性能。

    

    模型合并无需额外训练即可组合多个微调模型，而算术方法已成为一种无需数据、简单而有效的模型合并方式。然而，现有的算术方法在参数空间中处理任务向量，并对可合并性可能不同的方向统一应用单一的逐元素规则。在共享的方向基中，我们发现了一致的双区间结构：一个跨任务共享的低离散度公共主体，以及一个高离散度、由任务主导且承载着不成比例能量的尾部。这些发现催生了公共-独立子空间分解（CUSD），这是一种无需数据的预合并方法，它将任务向量分解为公共与独立的方向分量，用下游算术算子合并公共分量，并单独保留独立残差。在11个基础模型、四个模型家族和四种基于算术的合并算子上的实验表明，CUSD提升了……

    arXiv:2611.04108v2 Announce Type: replace-cross  Abstract: Model merging combines multiple fine-tuned models without additional training, and arithmetic methods have emerged as a simple yet effective way to merge models without data. However, existing arithmetic methods treat task vectors in parameter space and apply a single element-wise rule to directions that can differ in mergeability. In a shared directional basis, we find a consistent two-regime structure: a low-dispersion common bulk shared across tasks and a high-dispersion task-dominated tail that carries disproportionate energy. These findings motivate Common--Unique Subspace Decomposition (CUSD), a data-free pre-merging method that separates task vectors into common and unique directional components, merges the common components with a downstream arithmetic operator, and retains unique residuals separately. Experiments across $11$ base models, four model families, and four arithmetic-based merging operators show that CUSD im
    
[^222]: 利用文本语义实现鲁棒的跨视角目标对应

    Exploiting Textual Semantics for Robust Cross-View Object Correspondence

    [https://arxiv.org/abs/2611.02356](https://arxiv.org/abs/2611.02356)

    提出TeSCo框架，利用目标文本描述中视角和外观不变的语义信息来补充视觉线索，从而显著提升复杂场景下跨视角目标对应的鲁棒性。

    

    跨视角目标对应（CVOC）旨在在不同视角拍摄的查询视频和搜索视频之间建立同一目标的关联。现有方法通常依赖视觉线索来实现CVOC。尽管近期取得了进展，但由于缺乏足够的目标信息，这些方法在视角变化剧烈、外观变化显著的复杂场景中表现不佳。针对这一问题，我们提出了TeSCo框架，在视觉线索之外利用目标丰富的文本语义来进行跨视角目标对应。我们的关键洞察是：目标的文本描述通过捕捉类别、纹理、颜色等多种属性，提供了视角不变和外观不变的语义，这些语义与视觉线索形成互补，因而能够增强复杂场景中目标对应的鲁棒性。受此启发，TeSCo首先从给定信息中生成目标的文本表达（摘要原文在此处截断）。

    arXiv:2611.02356v2 Announce Type: replace-cross  Abstract: Cross-view object correspondence (CVOC) aims to establish associations of the same target between query and search videos captured from different viewpoints. Existing methods often rely on visual cues for CVOC. Despite recent progress, these methods struggle in complicated scenarios with substantial viewpoint changes and appearance variations, due to lacking sufficient target information. Addressing this, we introduce TeSCo, a framework that exploits rich Textual Semantics of an object, in addition to its visual cues, for cross-view object Correspondence. Our key insight is, textual description of a target, by capturing diverse attributes such as category, texture, and color, provides viewpoint- and appearance-invariant semantics, which are complementary to visual cues and can hence enhance correspondence robustness in complicated scenes. Inspired by this, TeSCo first generates a textual expression of the target from the given 
    
[^223]: 智能体需要睡眠吗？

    Do agents need to sleep?

    [https://arxiv.org/abs/2611.02216](https://arxiv.org/abs/2611.02216)

    该论文借鉴睡眠的选择论“回响-共振-再归一化”（RRR）理论，将睡眠的三个阶段映射为智能体数字记忆架构中可行的操作，并提出可证伪预测：随着记忆规模增长，选择论式记忆巩固将优于指令论式方法。

    

    基于大语言模型的智能体日益跨会话持久运行并通过交互积累记忆，它们已经表现出可塑性失去控制后的常见症状：语义漂移、检索精度下降、存储错误的传播以及记忆容量的饱和。借鉴关于睡眠的选择论解释——“回响、共振与再归一化”（RRR）理论，我们提出睡眠的三个阶段可以分别映射到数字记忆架构中可实现的操作上。与此同时，构建有状态智能体的工程师们正在独立地重新发现睡眠在进化过程中所要解决的问题：遗忘什么，以及何时进行遗忘。由此我们得出一个可证伪的预测：随着记忆存储规模的增大，选择论式的巩固机制将优于指令论式的替代方案。核心主张涉及持续学习本身的本质：可塑性是有代价的，对于任何在有限条件下持续学习的系统……

    arXiv:2611.02216v2 Announce Type: cross  Abstract: LLM-based agents, which increasingly operate persistently across sessions and accumulate memory through interaction, already exhibit the familiar symptoms of unchecked plasticity: semantic drift, degraded retrieval precision, propagation of stored errors, and saturation of memory capacity. Drawing on the selectionist "ringing, resonance and renormalization" (RRR) account of sleep, we propose that each of its three phases maps onto an operation feasible in digital memory architectures. Engineers building stateful agents, meanwhile, are independently rediscovering the problems sleep evolved to solve: what to forget, and when to do the forgetting. From this we derive the falsifiable prediction that selectionist consolidation will outperform instructionist alternatives as memory stores grow large. The core claim concerns the nature of continuous learning itself: plasticity has a price, and for any system that learns continuously under fini
    
[^224]: 使用扩散反事实样本和匹配阳性对照测试黑色素瘤分类器的概念级审计

    Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls

    [https://arxiv.org/abs/2611.02119](https://arxiv.org/abs/2611.02119)

    本研究构建了基于扩散模型的条件反事实审计框架，并通过匹配的阳性对照严格验证了其对黑色素瘤分类器概念级解释的审计能力，证明该方法在恢复目标概念方面优于传统的概念激活向量方法。

    

    基于概念的黑色素瘤分类器解释很少针对具有已知目标的对照进行测试。我们基于以七点检查表中七个概念为条件的扩散模型构建了一个反事实审计，并使用此类对照对其进行了测试。经过色彩校准的起始噪声将颜色异常值从78.5-84.8%降低到6.6-13.3%。七个控制旋钮中有四个是目标主导的，但也会改变其他概念。基于四分位数的阳性对照存在混杂因素，该审计未能通过预先设定的规则。对于三个概念，它成功恢复了所有16个按生成器伪标签匹配的对照的目标，以及由第二位评估者标注的12个对照中全部12个的目标；而概念激活向量恢复的数量较少（16个中恢复了11个）。在18个置换标签分类器中，也出现了显著的扫描响应和目标排名，而标准化的效应边际成功地将对照与这些零分类器区分开来。色素沉着的扫描提高了黑色素瘤的……（摘要在此处被截断）

    arXiv:2611.02119v2 Announce Type: replace-cross  Abstract: Concept-based explanations of melanoma classifiers are rarely tested against controls with known targets. We built a counterfactual audit from a diffusion model conditioned on seven 7-point checklist concepts and tested it with such controls. Colour-calibrated start noise reduced colour outliers from 78.5-84.8\% to 6.6-13.3\%. Four of seven knobs were target-dominant but also changed other concepts. Quartile-based positive controls were confounded, and the audit failed its pre-specified rule. It recovered the targets of all 16 controls matched on the generator's pseudo-labels and of 12 of 12 controls labelled by a second evaluator, for three concepts; concept activation vectors recovered fewer (11 of 16). In 18 permuted-label classifiers, significant sweep responses and target rankings also arose, whereas standardised effect margins separated the controls from these null classifiers. The pigmentation sweep raised the melanoma l
    
[^225]: 波动物理作为表示：用于海洋声场的张量结构神经重建

    Wave Physics as Representation: Tensor-Structured Neural Reconstruction for Ocean Acoustic Fields

    [https://arxiv.org/abs/2611.01680](https://arxiv.org/abs/2611.01680)

    该论文提出从波动物理中提取结构化表示并用张量结构神经网络学习其组件的新路线，实现了既具物理可解释性、又对不确定环境信息鲁棒的海洋三维声场稀疏重建。

    

    从稀疏测量中重建三维（3D）复杂海洋声场极具挑战性，原因在于高度振荡的波结构以及受限的感知模式与路径。基于学习的方法可分为两类：第一类是纯数据驱动模型，虽然能够学习复杂的场模式，但并未显式利用波动物理，而波动物理本可增强可解释性并降低训练成本；第二类是物理信息方法，其将特定的物理方程作为重建约束，但往往依赖难以获取的环境相关参数。因此，如何在物理上有依据、又能对不确定环境信息保持鲁棒性的实用3D声场重建，在很大程度上仍是一个未被充分探索的问题。本工作采取了另一条路线：不再求解完全指定的波动方程，而是从波动物理中提取一种结构化表示，并学习其各个组成部分……

    arXiv:2611.01680v2 Announce Type: replace-cross  Abstract: Reconstructing three-dimensional (3D) complex ocean acoustic fields from sparse measurements is challenging due to highly oscillatory wave structures and restricted sensing patterns/routes. Learning-based approaches fall into two categories. First, purely data-driven models can learn complex field patterns, but do not explicitly exploit wave physics that may enhance interpretability and reduce training costs. Second, physics-informed methods impose a specified physical equation as a reconstruction constraint, but often require environment-dependent parameters that are difficult to acquire. As a result, practical 3D acoustic field reconstruction that is both physically grounded and robust to uncertain environmental information remains largely unexplored. This work takes an alternative route: instead of solving a fully specified wave equation, it extracts a structured representation from wave physics and learns its components fro
    
[^226]: Labadain-MedQA：一个用于微调和评估大型语言模型的德顿语医疗问答数据集

    Labadain-MedQA: A Tetun Medical QA Dataset for Fine-Tuning and Evaluating Large Language Models

    [https://arxiv.org/abs/2611.00935](https://arxiv.org/abs/2611.00935)

    该论文构建了首个大规模德顿语医疗问答数据集Labadain-MedQA，包含39,077个问答对，通过将MedAlpaca医学数据集翻译成德顿语并进行人机翻译质量对比评估，为低资源语言的医疗LLM微调与评估提供了宝贵资源。

    

    大型语言模型（LLM）的进展为医疗知识从高资源语言向低资源语言的跨语言迁移创造了机会，以支持人们获取医疗保健信息。然而，用于微调和评估LLM的医疗问答（QA）数据集在德顿语中仍然稀缺。为填补这一空白，我们使用Gemini 2.5 Flash将MedAlpaca的医学闪卡和WikiDoc患者信息集合翻译成德顿语，生成了39,077个医疗问答对。翻译质量通过对50个随机抽取的问答对（100个文档）进行评估，采用自动指标与人工参考翻译对比，并由两名以德顿语为母语的医疗专业人员进行评估，重点关注医学准确性和德顿语流畅性。人工翻译在两项标准上的平均评分均高于LLM生成的翻译（4.43/5 对比 4.02/5），差异具有统计学显著性（Wilcoxon符号秩检验，p <...）（摘要在此处被截断）

    arXiv:2611.00935v1 Announce Type: new  Abstract: Advances in large language models (LLMs) have created opportunities for cross-lingual transfer of medical knowledge from high-resource to low-resource languages to support access to healthcare information. However, medical question-answering (QA) datasets for fine-tuning and evaluating LLMs remain scarce for Tetun. To address this gap, we translated MedAlpaca's Medical Flashcards and WikiDoc Patient Information collections into Tetun using Gemini 2.5 Flash, producing 39,077 medical QA pairs. Translation quality was assessed on 50 randomly sampled QA pairs (100 documents) using automatic metrics against human reference translations and evaluations by two native Tetun-speaking healthcare professionals, focusing on medical accuracy and Tetun fluency. Human translations achieved higher mean ratings than LLM-generated translations for both criteria (4.43/5 vs. 4.02/5), with statistically significant differences (Wilcoxon signed-rank test, p <
    
[^227]: VCoT-SQL：通过双重思维链验证训练增强Text-to-SQL生成

    VCoT-SQL: Enhancing Text-to-SQL Generation through Dual Chain-of-Thought Validation Training

    [https://arxiv.org/abs/2611.00815](https://arxiv.org/abs/2611.00815)

    VCoT-SQL框架通过中间步骤验证的思维链监督训练SQL生成器和SQL验证器两个互补模型，使小型开源模型在Text-to-SQL任务上超越前沿闭源模型。

    

    开源大语言模型为Text-to-SQL任务提供了闭源模型的灵活替代方案，但在处理复杂查询时仍然表现不佳。使用思维链（CoT）数据进行微调可以改善推理能力，然而标准数据集缺乏CoT标注，需要进行合成生成。现有的合成方法仅验证最终答案，导致有缺陷的推理步骤会污染训练数据。我们提出VCoT-SQL，这是一个通过中间步骤验证的CoT监督和专用SQL验证器来改进Text-to-SQL生成的框架，使小型开源模型能够超越前沿闭源模型。VCoT-SQL在经过筛选的推理轨迹上训练两个互补模型：一个SQL生成器，使用子句级CoT模板，其推理步骤通过执行和模式对齐检查进行验证；一个SQL验证器，使用诊断性CoT模板来检测语法、语义和类型级别的错误。我们进一步设计……

    arXiv:2611.00815v2 Announce Type: replace  Abstract: Open-source LLMs provide a flexible alternative to closed-source models for Text-to-SQL but still struggle with complex queries. Finetuning with chain-of-thought (CoT) data improves reasoning, yet standard datasets lack CoT annotations, requiring synthetic generation. Existing synthesis methods only verify the final answer, allowing flawed reasoning steps to corrupt the training data. We propose VCoT-SQL, a framework that improves Text-to-SQL generation through intermediate-step validated CoT supervision and a dedicated SQL validator, which enables small open-source models to outperform frontier closed-source models. VCoT-SQL trains two complementary models on curated reasoning traces: an SQL generator using clause-level CoT templates where reasoning steps are verified via execution and schema alignment checks, and an SQL validator using diagnostic CoT templates to detect syntactic, semantic, and type level errors. We further design 
    
[^228]: 行为图：用于解释机器人部署数据的无监督图结构

    Behavior Graphs: Unsupervised Graph Structures for Interpreting Robot Deployment Data

    [https://arxiv.org/abs/2611.00405](https://arxiv.org/abs/2611.00405)

    该论文提出“行为图”这一无监督方法，将机器人策略rollout数据压缩为一张单一、可解释的有向图，通过识别相似行为模式、按时序连接并与结果关联，显著降低了部署数据的分析难度，使研究者能更高效地指导数据收集决策。

    

    数据飞轮承诺通过从部署数据中获得的洞察实现持续改进。理解部署后的模型如何成功与失败，可以引导后续的数据收集，从而修补失败并强化成功。然而，机器人部署会产生大量高维的rollout数据，这些数据难以理解，也难以据此做出有把握的数据收集决策。为了让部署数据更易于分析和采取行动，我们提出了行为图，它将策略rollout压缩成一张单一的、可解释的有向图。我们通过学习识别不同rollout之间相似的行为模式、在时间维度上连接这些行为，并最终将它们与结果关联起来来构建该图。在一项针对三个真实世界任务的用户研究中，参与者通过分析rollout来选择数据收集策略，18名参与者中有16名更偏好行为图而非rollout视频，且在感知负荷降低40.3%的情况下得出了相当的结论。

    arXiv:2611.00405v1 Announce Type: cross  Abstract: Data flywheels promise continuous improvement via insights from deployment data. Understanding how the deployed model succeeds and fails can steer subsequent data collection towards patching failures and reinforcing successes. However, robot deployments create large volumes of high-dimensional rollout data, which is challenging to understand and confidently ground data collection decisions in. To make deployment data easier to analyze and act on, we propose behavior graphs, which compress policy rollouts into a single interpretable, directed graph. We construct the graphs by learning to identify similar behavior modes across rollouts, connecting the behaviors temporally, and finally linking them to outcomes. In a user study on three real-world tasks, where participants analyzed rollouts to choose a data collection strategy, 16 of 18 preferred behavior graphs to rollout videos, reaching comparable conclusions with 40.3% lower perceived 
    
[^229]: 量化非线性嵌入的局部与全局保真度

    Quantifying Local and Global Fidelity of Nonlinear Embeddings

    [https://arxiv.org/abs/2611.00290](https://arxiv.org/abs/2611.00290)

    提出经机会校正的 DLP 曲线，从中导出局部保真度 L 与全局保真度 G，并配合 LG 反转检验和实体保真度 E，以可验证的方式量化并判定降维嵌入保留局部与全局结构的能力。

    

    降维建立在一个断言之上：t-SNE 和 UMAP 保留局部结构，PCA 保留全局结构。我们的 DLP 曲线表示映射为每个点的 k 近邻所保留的比例。经过机会校正后，该曲线在 k=10 时给出 L（局部保真度），在数据的 30% 到 50% 范围上给出 G（全局保真度）。可以证明，在中点处该曲线等价于距离的秩相关。LG 反转检验能以给定的错误率判定两个映射中哪个是局部的、哪个是全局的。在多达 138 万个点的十二个数据集上，G 与已有全局评分的相关性达到 0.84 和 0.93，与 R_NX 曲线下面积的相关性达到 0.22 和 0.30。在实体规模不同的情形下，在每个点的实体规模处读取的实体保真度 E，可以预测超出 L 和 G 之外映射会丢失哪些实体。在一个包含一百万条段落的智能体记忆实验中，t-SNE 与 PCA 在 15 万条段落处发生交叉（p=2.6e-5）。

    arXiv:2611.00290v2 Announce Type: replace  Abstract: Dimension reduction rests on a claim: t-SNE and UMAP preserve local structure, PCA global structure. Our DLP curve is the share of each point's k nearest neighbors a map keeps. Corrected for chance, it gives L at k=10 and G over 30 to 50% of the data, local and global fidelity. At the midpoint the curve is provably a rank correlation of distances. The LG reversal test decides, with a stated error rate, which of two maps is local and which global. On twelve datasets of up to 1.38 million points, G agrees with established global scores at 0.84 and 0.93, and the area under R_NX at 0.22 and 0.30. The entity fidelity E, read at each point's entity size, predicts which entities a map loses beyond L and G where entity sizes differ. In an agent memory of one million passages, t-SNE and PCA cross at 150,000 passages (p=2.6e-5).
    
[^230]: AuroraWorld：面向机器人操作中外部视角到第一人称视角视频生成的结构化4D世界建模

    AuroraWorld: Structured 4D World Modeling for Exo-to-Ego Video Generation in Robot Manipulation

    [https://arxiv.org/abs/2611.00177](https://arxiv.org/abs/2611.00177)

    提出AuroraWorld，一种结构化4D世界建模方法，能够从大规模外部视角机器人操作视频中合成物理一致、且对视觉运动策略学习有用的第一人称视角视频。

    

    第一人称（自我中心）观测提供了以机器人为中心的操作过程视角，能够捕捉到对视觉运动策略学习至关重要、但难以从外部相机观察到的接触交互和物体状态变化。然而，由于需要额外的硬件设备以及特定平台的部署成本，大规模收集此类数据仍然十分昂贵。从大规模的外部视角操作视频中合成第一人称观测提供了一种可扩展的替代方案。然而，在面临大幅度的视角变换、严重的遮挡、复杂的物体交互以及外部视角固有的稀疏观测时，现有方法难以合成物理一致且与策略学习相关的第一人称视频。代码将在 https://github.com/xueha0/AuroraWorld 公开。

    arXiv:2611.00177v2 Announce Type: replace  Abstract: Egocentric observations provide a robot-centric view of manipulation processes, capturing contact interactions and object-state changes that are important for visuomotor policy learning but difficult to observe from external cameras. However, collecting such data at scale remains expensive due to additional hardware requirements and platform-specific deployment costs. Synthesizing egocentric observations from large-scale exocentric manipulation videos provides a scalable alternative. However, existing methods struggle to synthesize physically consistent and policy-relevant egocentric videos under substantial viewpoint transformations, severe occlusions, complex object interactions, and inherently sparse observations from available exocentric viewpoints.Code will be made publicly available at https://github.com/xueha0/AuroraWorld
    
[^231]: 当披露成为强制要求：利用自利的大语言模型司机智能体评估网约车调度政策

    When Disclosure Is Compulsory: Evaluating Ride-Hailing Rebalancing Policies with Self-Interested LLM Driver Agents

    [https://arxiv.org/abs/2610.41132](https://arxiv.org/abs/2610.41132)

    该研究首次在欧盟《平台工作指令》强制披露算法规则的背景下，用以自身收益最大化为目标的自利LLM司机智能体评估网约车反应式与预测式两类调度政策，发现规则公开后司机更遵循平台调度指令、拒绝调度的比例显著下降。

    

    欧盟《平台工作指令》将于2026年12月前转化为成员国法律，要求网约车平台以通俗易懂的语言说明其自动化决策的主要参数，以及劳动者行为如何影响这些决策。这类规则正是大语言模型（LLM）能够阅读的内容，因此司机将可以让LLM代替自己做决策。在规则公开之后，我们比较了两类主要的调度（rebalancing）政策：反应式政策（R），即将空闲司机派往刚刚流失请求的区域；以及预测式政策（A），即基于需求预测来规划司机移动。每个司机都是一个以最大化自身净收入为目标的LLM智能体，其条件为：不提供规则（N），或按该指令的详细程度提供规则（L）。我们在三区域模型和曼哈顿真实路网中测试了两个LLM——Claude Haiku 5.5和Jev，二者对规则的理解水平相当。结果显示，在规则被披露后，司机更紧密地遵循平台指令：在曼哈顿，他们拒绝调度提议的比例下降……

    arXiv:2610.41132v1 Announce Type: cross  Abstract: The EU Platform Work Directive, to be transposed by December 2026, requires ride-hailing platforms to state in plain language the main parameters of their automated decisions and how a worker's behaviour affects them. Such rules are exactly what a large language model (LLM) can read, so drivers will be able to let one decide for them. We compare two main families of rebalancing policy once their rules are public: a reactive policy (R) that sends idle drivers to cover requests just lost, and an anticipatory policy (A) that plans moves on a demand forecast. Each driver is an LLM agent maximizing its own net income, given no rules (N) or the rules at the directive's level of detail (L). We test two LLMs, Claude Haiku 5.5 and Jev, which understand the rules equally well, in a three-zone model and in Manhattan. With the rules disclosed, drivers follow the platform more closely: in Manhattan, the share of relocation offers they decline falls
    
[^232]: Tactus：用于自然灵巧操作数据采集的触觉与姿态感知手套

    Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection

    [https://arxiv.org/abs/2610.39076](https://arxiv.org/abs/2610.39076)

    Tactus是一种织物集成的感知手套，通过超薄3D针织外形、策略性布置的电容触觉传感器和融合IMU的抗遮挡姿态跟踪三大设计，实现了自然人类操作演示中精确接触力与手部姿态的高质量数据采集。

    

    采集带有精确接触力和手部姿态的自然人类操作演示数据，对于机器人灵巧操作策略和基础世界模型学习至关重要——然而，目前没有任何可穿戴平台能够在保持自然操作行为的外形设计下，同时提供经过标定的指尖力感知和抗遮挡的手部跟踪。我们提出了Tactus，一种织物集成的感知手套，共同优化了三大设计支柱：(1) 超薄、贴合的3D针织外形（1.5毫米指尖叠层），旨在最大程度减少对自然操作的干扰；(2) 基于实证接触研究，在操作相关区域策略性布置的电容式触觉传感器；(3) 融合IMU的姿态跟踪，在手-物体遮挡最严重（也正是操作数据最有价值）的情况下仍能保持跟踪精度。Tactus还包含一套文档化的标定流程，以确保跨手套的一致性和单次会话（该部分内容被截断）

    arXiv:2610.39076v1 Announce Type: cross  Abstract: Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning - yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-sessi
    
[^233]: MedConstBench：衡量“宪法”如何引导大语言模型医疗伦理行为的基准测试

    MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

    [https://arxiv.org/abs/2610.36513](https://arxiv.org/abs/2610.36513)

    该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。

    

    随着大语言模型的快速发展，它们被越来越多地用于回答医学问题，且许多此类模型如今已在本地部署。模型一旦部署，其行为主要由写入其“宪法”中的伦理政策所决定。然而，现有研究大多关注事实正确性，而非相互冲突的伦理价值之间的权衡。即使是那些确实评估伦理的基准测试，也将伦理视为模型的固定属性而非变量，因此“宪法”如何重塑医疗伦理行为在很大程度上仍未被探索。我们提出 MedConstBench，一个将部署宪法作为受控变量的医疗伦理基准测试，它围绕比彻姆的四大伦理原则组织，并通过人工校准的多模型同行评审进行评分。该基准包含我们自主构建的临床困境数据集 MedDilemma，并支持对“宪法”如何引导大语言模型进行系统性分析。对十个开源（模型）的评估……

    arXiv:2610.36513v1 Announce Type: new  Abstract: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-sourc
    
[^234]: AutoEnergyDesign：一种由大语言模型驱动的自动能源系统设计优化智能体

    AutoEnergyDesign: An LLM-Driven Agent for Automatic Energy System Design Optimization

    [https://arxiv.org/abs/2610.35160](https://arxiv.org/abs/2610.35160)

    该论文提出AutoEnergyDesign框架，利用具备多类型显式记忆和git式自动回退机制的LLM智能体，通过仿真反馈进行上下文学习来自动优化建筑能源系统设计，从而大幅减少黑盒优化所需的仿真评估次数。

    

    设计建筑能源系统是一个高维度、混合整数的搜索问题：当每个设计方案都需要通过长达一年的仿真来评估时，黑盒优化器通常需要数百次评估。本文提出了AutoEnergyDesign，这是一个由大语言模型（LLM）智能体驱动的框架，该智能体对先前的仿真结果进行推理、提出配置变更，并通过上下文从每次结果中学习，同时基于git的机制会自动回退不成功的变更。该智能体将推理模型与显式记忆组织（程序性、语义性、情景性和反思性记忆）、受限动作接口以及基于环境的保留/回退信号相结合。仿真器涵盖八种技术（光伏、微型风机、热电联产、太阳能光热、燃气锅炉、电池、热泵和逆变器），并计算25年净现值成本（NPC），其中包含资本成本、运营成本和残值。

    arXiv:2610.35160v1 Announce Type: new  Abstract: Designing building energy systems is a high-dimensional, mixed-integer search problem: when each design is evaluated by a year-long simulation, black-box optimizers typically need hundreds of evaluations. This paper presents AutoEnergyDesign, a framework in which a Large Language Model (LLM) agent reasons over previous simulation results, proposes configuration changes and learns in context from each outcome, while a git-based mechanism automatically reverts unsuccessful changes. The agent couples a reasoning model with an explicit memory organisation (procedural, semantic, episodic and reflective), a constrained action interface and an environment-grounded keep/revert signal. The simulator covers eight technologies (photovoltaics, micro-wind, combined heat and power, solar thermal, gas boilers, batteries, heat pumps and inverters) and computes the 25-year Net Present Cost (NPC), including capital and operating costs and salvage value. O
    
[^235]: SPARTA：一种用于长上下文边缘推理的具备自适应注意力与统一数据通路的稀疏三值LLM加速器

    SPARTA: A Sparse Ternary LLM Accelerator with Adaptive Attention and Unified Datapath for Long-Context Edge Inference

    [https://arxiv.org/abs/2610.34824](https://arxiv.org/abs/2610.34824)

    SPARTA是一种面向边缘设备长上下文推理的稀疏三值LLM加速器，通过相位感知稀疏数据流、统一计算架构和自适应两阶段注意力引擎（页级Top-k与令牌级Top-p剪枝），显著降低KV缓存流量并提升推理效率。

    

    边缘设备上的大语言模型受限于有限的内存和带宽。尽管1.58比特三值权重极大地降低了模型存储需求，但长上下文推理日益受到KV缓存流量和注意力计算的瓶颈制约。现有加速器还为三值线性层和注意力计算采用独立的数据通路，带来了硬件和执行开销。我们提出了SPARTA，一种面向高效长上下文边缘推理的稀疏三值LLM加速器。SPARTA集成了：（1）相位感知的稀疏数据流，在计算受限的预填充阶段与内存受限的解码阶段利用异构稀疏性；（2）统一的计算架构，将三值SpMM/SpMV与位串行注意力映射到共享的处理单元上，以利用零权重和零比特稀疏性；（3）自适应两阶段注意力引擎，结合页级Top-k选择（减少K缓存访问）与令牌级Top-p剪枝（减少V缓存流量）。SPARTA将KV缓存流量降低了4……（原文摘要至此处被截断）

    arXiv:2610.34824v1 Announce Type: cross  Abstract: LLMs on edge devices are constrained by limited memory and bandwidth. Although 1.58-bit ternary weights greatly reduce model storage, long-context inference is increasingly bottlenecked by KV-cache traffic and attention. Existing accelerators also use separate datapaths for ternary linear layers and attention, incurring hardware and execution overhead. We present SPARTA, a sparse ternary LLM accelerator for efficient long-context edge inference. SPARTA integrates: (1) a phase-aware sparse dataflow exploiting heterogeneous sparsity across compute-bound prefill and memory-bound decode; (2) a unified compute fabric mapping ternary SpMM/SpMV and bit-serial attention onto shared processing elements to exploit zero-weight and zero-bit sparsity; and (3) an adaptive two-stage attention engine combining page-level Top-k selection to reduce K-cache accesses with token-level Top-p pruning to reduce V-cache traffic. SPARTA reduces KV traffic by 4.
    
[^236]: OS-Wizard：通过知识引导的探索将计算机使用智能体扩展至专业软件

    OS-Wizard: Scaling Computer-Using Agents to Specialized Software via Knowledge-Guided Exploration

    [https://arxiv.org/abs/2610.33907](https://arxiv.org/abs/2610.33907)

    OS-Wizard是一个自动化轨迹收集框架，通过教程引导探索与混合GUI-代码动作空间为专业桌面软件收集了10.9K条高质量交互轨迹，基于此微调的开源模型显著提升了计算机使用智能体的性能。

    

    计算机使用智能体在数字世界中执行复杂任务的能力日益增强。然而，对于专业桌面应用程序而言，高质量的交互轨迹仍然稀缺，因为应用程序知识有限、GUI工作流程繁琐以及探索不完整等因素阻碍了自动化数据收集。我们提出了OS-Wizard，一个自动化轨迹收集框架。该框架将教程引导的探索与混合GUI-代码动作空间相结合，提供应用程序特定的知识，并通过直接执行应用程序原生命令来简化GUI工作流程。此外，该框架还在树状探索过程中对关键状态进行检查点记录与重访，以探索替代性工作流程，从而补充线性推演，实现更广泛的覆盖。我们收集了10.9K条轨迹，包含54.9K个状态-动作样本。在该语料库上微调开源基础模型后，得到的性能最佳的模型在基准测试中提升了表现。

    arXiv:2610.33907v1 Announce Type: new  Abstract: Computer-using agents are increasingly capable of performing complex tasks in the digital world. However, high-quality interaction trajectories remain scarce for specialized desktop applications, where limited application knowledge, cumbersome GUI workflows, and incomplete exploration hinder automated collection. We introduce OS-Wizard, an automated trajectory collection framework. It combines tutorial-guided exploration with a hybrid GUI--Code action space, providing application-specific knowledge and simplifying GUI workflows through direct execution of application-native commands. The framework also checkpoints and revisits pivotal states during tree exploration to explore alternative workflows, complementing linear rollouts for broader coverage. We collect 10.9K trajectories comprising 54.9K state--action samples. Fine-tuning open-source base models on this corpus yields a best-performing model that improves performance on a benchmar
    
[^237]: 基于局部熵波动的黑盒大语言模型生成文本检测

    Local Entropy Fluctuations for Black-Box LLM-Generated Text Detection

    [https://arxiv.org/abs/2610.33767](https://arxiv.org/abs/2610.33767)

    该论文提出新指标“相对峰值显著率”，利用人类写作信息流均匀而机器生成文本存在尖锐局部熵波动这一结构性差异，解决了黑盒场景下现有检测器在指令条件文本上灾难性失效的问题。

    

    零样本机器生成文本检测对于缓解大语言模型（LLM）滥用风险至关重要。尽管最先进的双模型检测器和基于重采样的检测器在无提示续写文本上的表现优于单模型统计检测器，但在黑盒设置（即源模型未知）下，它们在指令条件生成的文本上会遭遇灾难性失效。我们分析了这一失效现象，并将其根源归结为基于重采样的方法在提示条件下表现不佳，以及双模型检测器跨模型差异的减弱。受近期研究启发现象的启发——人类写作保持稳定、均匀的信息流，而机器生成的文本则呈现出尖锐的局部熵波动——我们提出了一种新指标：相对峰值显著率，用以捕捉熵轨迹上的这种结构性对比，并进一步将其作为LLM生成文本检测的信号。大量实验……

    arXiv:2610.33767v1 Announce Type: new  Abstract: Zero-shot detection of machine-generated text is vital for mitigating the risks of large language models (LLMs) misuse. While the state-of-the-art dual-model detectors and resampling-based detectors outperform single-model statistical detectors on unprompted continuations, they suffer a catastrophic breakdown on instruction-conditioned text in the black-box setting, where the source model is unknown. We analyze this failure and trace it to poor prompt conditioning for resampling-based methods and diminished cross-model discrepancy for dual-model detectors. Motivated by recent studies that human writing maintains a steady, uniform information flow, whereas machine-generated text exhibits sharp, localized entropy fluctuations, we propose a new metric, the Relative Peak Prominence Rate (RelPPR) to capture this structural contrast along the entropy trajectory and further leverage it as a signal for LLM-generated text detection. Extensive eva
    
[^238]: Isharah-Selfie：面向单手手语的连续手语识别数据集

    Isharah-Selfie: Continuous Sign Language Recognition Dataset for One-handed Signing

    [https://arxiv.org/abs/2610.33200](https://arxiv.org/abs/2610.33200)

    提出了首个使用智能手机前置摄像头在无约束自拍条件下采集的大规模阿拉伯手语连续识别数据集Isharah-Selfie，专门针对现实移动场景中的单手手语识别。

    

    当前的手语识别与翻译基准数据集通常在受控环境下使用固定相机或影棚式相机采集，这极大地限制了其反映手语使用者日常移动场景交流方式的能力。当考虑到阿拉伯手语时，这一差距尤为明显，因为阿拉伯手语是阿拉伯国家广泛使用的主要手语。在这项工作中，我们提出了Isharah-Selfie，这是一个完全使用智能手机前置摄像头在无约束自拍条件下采集的大规模阿拉伯手语数据集。该数据集包含16名手语者录制的8,997个视频片段，涵盖1,493个不同句子，内容涉及医疗、教育和交通等多个领域。与以固定相机、双手手语为主的传统数据集不同，Isharah-Selfie捕捉了以单手手语、近距离视角和部分身体可见为特征的现实移动交流场景。

    arXiv:2610.33200v1 Announce Type: new  Abstract: Current sign language recognition and translation benchmarks are commonly collected in controlled settings using fixed or studio-style cameras, particularly limiting their ability to reflect how signers communicate in everyday mobile scenarios. As such, this gap becomes increasingly evident when considering the Arabic Sign Language (ArSL), which is a primary sign language used across Arab countries. In this work, we introduce Isharah-Selfie, a large-scale ArSL dataset collected entirely using front-facing smartphone cameras under unconstrained selfie conditions. The dataset comprises 8,997 video clips performed by 16 signers across 1,493 unique sentences, covering several domains such as healthcare, education, and transportation. Unlike conventional datasets dominated by fixed-camera, two-handed signing, Isharah-Selfie captures realistic mobile communication characterized by one-handed signing, close-range viewpoints, partial body visibi
    
[^239]: 面向历史文档转录的实体感知OCR系统评估

    Entity-Aware Evaluation of OCR Systems for Historical Document Transcription

    [https://arxiv.org/abs/2610.33004](https://arxiv.org/abs/2610.33004)

    该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。

    

    光学字符识别（OCR）对于历史档案数字化至关重要，然而字符错误率（CER）等标准评估指标并不一定能反映姓名、日期和标识符等有意义的信息是否被保留。在本工作中，我们研究了OCR错误的实体级分析，作为评估历史文档OCR质量的一种补充视角。我们使用来自Berrutti档案的175份文档的公开基准数据集，并辅以5,209个手动标注的实体提及（将于论文被接收后发布），比较了序列识别OCR系统、OCR专用视觉语言模型以及通用视觉语言模型。除了全局转录准确率之外，我们还在不同的字符级编辑容差下分析了实体的保留性和可恢复性。我们的实验表明，OCR系统的排名取决于评估标准：虽然Qwen 3.6在……（摘要原文在此处被截断）

    arXiv:2610.33004v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) is essential for digitizing historical archives, yet standard evaluation metrics such as Character Error Rate (CER) do not necessarily reflect whether meaningful information, such as names, dates, and identifiers, is preserved. In this work, we investigate entitylevel analysis of OCR errors as a complementary perspective for evaluating OCR quality in historical documents. Using a publicly available benchmark of 175 documents from the Berrutti Archive, augmented with 5,209 manually annotated entity mentions that will be released upon acceptance, we compare sequence-recognition OCR systems, OCR-specialized vision-language models, and general-purpose vision-language models. Beyond global transcription accuracy, we analyze entity preservation and recoverability under varying character-level edit tolerances. Our experiments show that OCR rankings depend on the evaluation criterion: while Qwen 3.6 achieves
    
[^240]: HyScale：面向学习型稀疏注意力的长上下文训练扩展

    HyScale: Scaling Long-Context Training for Learned Sparse Attention

    [https://arxiv.org/abs/2610.32057](https://arxiv.org/abs/2610.32057)

    HyScale 通过无需分数表的实时精确 top-k 选择方法和前向选择重放机制，将学习型稀疏注意力的百万词元长上下文训练速度提升了 2.07 倍。

    

    我们提出了 HyScale，一个面向学习型稀疏注意力的长上下文训练系统。学习型稀疏注意力是一种适用于百万级词元上下文的有前景架构，自 DeepSeek-V3.2 以来已在生产环境中被广泛采用。然而，选择操作成为了训练瓶颈：对于每个查询，它需要对之前所有词元打分，而稀疏注意力只读取其中的 $k$ 个；更糟的是，当前系统会生成完整的分数表，并在重计算期间重复选择操作。我们的关键洞察是：选择的输出不是激活值，而是一组索引。基于这一洞察，HyScale 引入了 HyTopK 和选择重放机制。HyTopK 无需分数表即可实时找到每个查询的精确 top-$k$，而选择重放则在重计算中复用前向传播的选择结果。结合应对端到端训练挑战的整体系统优化，HyScale 在 512 块 H20 GPU 上将 GLM-5.1 在 1M 词元序列上的训练速度相比 Megatron-LM 提升了 2.07 倍。

    arXiv:2610.32057v1 Announce Type: new  Abstract: We present HyScale, a long-context training system for learned sparse attention. Learned sparse attention is a promising architecture for million-token contexts, widely adopted in production since DeepSeek-V3.2. Selection, however, grows into the training bottleneck. For each query, it scores every previous token whereas sparse attention reads only $k$; worse, current systems materialize a full score table and repeat selection during recomputation. Our key insight is that the output of selection is not an activation, but rather a set of indices. Building on it, HyScale introduces HyTopK and selection replay. HyTopK finds each query's exact top-$k$ on the fly without a score table, while selection replay reuses forward-pass selection in recomputation. Together with holistic system optimizations tackling end-to-end training challenges, HyScale speeds up GLM-5.1 training on 1M-token sequences by 2.07$\times$ over Megatron-LM on 512 H20 GPUs
    
[^241]: 弥合数据鸿沟：面向少样本扩散的源信息任务空间传输

    Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion

    [https://arxiv.org/abs/2610.31323](https://arxiv.org/abs/2610.31323)

    提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。

    

    许多实际部署场景要求模型能够快速地从数据丰富的源域迁移到仅有稀疏观测的相关目标域。当目标数据难以收集时，这一问题尤为突出，例如在存在持续人群或采集不平衡的医疗场景中。为每个目标域重新训练或微调高容量生成器代价高昂，且当目标数据稀缺时在统计上可能不可靠。我们提出 Meta-Diff，它转而在一个紧凑的任务空间中一次性学习反复出现的源-目标关系：一个共享编码器将每个分布映射为低维坐标；一个学习到的关系将新的源坐标传输到其预测的目标坐标，随后共享的扩散主干直接从该预测进行生成，无需针对特定目标进行再训练。受控的高斯混合实验揭示了该方法生效的条件：任务身份必须在表示中得以保留，关系必须能够在不同任务间泛化，并且源-

    arXiv:2610.31323v1 Announce Type: new  Abstract: Many deployments require a model to move rapidly from a data-rich source to a related target observed only sparsely. This is especially consequential where target data are difficult to collect, as in healthcare settings with persistent population or acquisition imbalances. Retraining or fine-tuning a high-capacity generator for every target is costly and can be statistically unreliable when target data are scarce. We introduce Meta-Diff, which instead learns a recurring source--target relation once in a compact task space. A shared encoder maps each distribution to a low-dimensional coordinate; a learned relation transports a new source coordinate to its predicted target, and a shared diffusion backbone generates from that prediction without target-specific retraining. Controlled Gaussian-mixture experiments identify when this works: task identity must survive the representation, the relation must generalise across tasks, and the source-
    
[^242]: 非对称在策略蒸馏学习情境化探索

    Asymmetric On-Policy Distillation Learns In-Context Exploration

    [https://arxiv.org/abs/2610.30105](https://arxiv.org/abs/2610.30105)

    提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。

    

    在部分可观测条件下的序贯决策十分困难，因为智能体必须先通过探索来推断隐藏的、与任务相关的信息，然后才能有效地行动。通过在线强化学习从零开始学习这种信息收集行为的样本效率很低，而离线模仿学习则会在不同的隐藏状态之间平均专家动作，从而无法学到探索行为。在这项工作中，我们提出了 ASTEROID，这是一个简单而有效的学生-教师监督学习框架，能够学习测试时的探索行为。尽管拥有特权信息的教师从不需要探索，我们证明了基于历史条件的学生模型在与策略上下文一起训练时能够学会探索：学生首先在部分可观测条件下收集在策略上下文，特权教师则基于该上下文提供动作标签。通过直接在学生的在策略历史条件下蒸馏专家动作，学生能够学会系统地收集与任务相关的信息。

    arXiv:2610.30105v1 Announce Type: cross  Abstract: Sequential decision-making under partial observability is difficult because agents must explore to infer hidden, task-relevant information before acting effectively. Learning such information-gathering behaviors tabula rasa via online reinforcement learning is sample-inefficient, whereas offline imitation averages expert actions across hidden states and fails to learn exploration. In this work, we propose ASTEROID, a simple and effective student--teacher supervised learning framework that can learn test-time exploration. Although a privileged expert never needs to explore, we demonstrate that a history-conditioned student can learn to do so when trained with on-policy contexts: the student first collects on-policy context under partial observability, and the privileged expert provides action labels conditioned on that context. By distilling expert actions directly conditioned on the student's on-policy history, the student learns to sy
    
[^243]: 从数据增强到合成证据：在数据集入口处评估生成的胸部X光片

    From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake

    [https://arxiv.org/abs/2610.29152](https://arxiv.org/abs/2610.29152)

    该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。

    

    生成的胸部X光片作为明示的数据增强可能是有用的，但当同一张图像被提交作为真实检查的记录时，它就变成了缺乏依据的证据。我们通过区分三个问题来研究这一问题：图像是否通过自动化质量检查、来源分类器是否将其标记为生成图像，以及可信记录是否支持所声称的采集过程。在我们的实验中，Qwen从1,000张现有的Cheff识别或SynthCheX候选图像池中最多选择20张图像。接收方使用MedGemma评估图像质量，并使用基于冻结DINOv3特征的分类器预测图像来源。基于质量的选择使Cheff识别图像的质量通过率从20张中的12张提高到19张，SynthCheX图像从8张提高到17张。我们没有发现任何证据表明选择能够改善同时通过两项检查的情况：没有选中的Cheff识别图像能同时通过两项检查，而均匀选择、质量选择和质量约束的替代选择分别产生2张、1张和……（摘要在此处截断）

    arXiv:2610.29152v1 Announce Type: cross  Abstract: A generated chest radiograph can be useful as declared augmentation, but the same image becomes unsupported evidence when submitted as the record of a real examination. We study this problem by separating three questions: whether an image passes an automated quality check, whether an origin classifier flags it as generated, and whether a trusted record supports the claimed acquisition. In our experiments, Qwen selects up to 20 images from pools of 1,000 existing Cheff-identified or SynthCheX candidates. A receiver evaluates quality with MedGemma and predicted origin with a classifier on frozen DINOv3 features. Quality-based selection raises quality passes from 12 to 19 of 20 for Cheff-identified images and from 8 to 17 for SynthCheX. We find no evidence that selection improves passage through both checks: no selected Cheff-identified image passes both, while uniform, quality, and quality-constrained surrogate selection produce 2, 1, an
    
[^244]: 一种用于逼真啮齿动物运动的潜在控制模型

    A latent control model for realistic rodent motion

    [https://arxiv.org/abs/2610.27279](https://arxiv.org/abs/2610.27279)

    提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。

    

    开发动物行为的具身模型是神经-AI领域的一个重要目标。具身模型将增强我们对生物智能体和人工智能体进行直接比较的能力。然而，要使这种比较有效，我们需要真正像动物一样运动的具身模型。先前的工作已经开发了虚拟动物身体以及通过模仿学习训练逼真运动的技术。然而，迄今为止，还没有一个具身动物建模系统能够既在经过训练的目标导向行为中，又在由随机生成驱动的探索过程中产生自然的运动。在这里，我们介绍了SCAMPER（通过增强运动先验实现具身强化学习模拟控制），这是一个训练具身虚拟大鼠模型的系统，能够在目标导向行为和随机探索过程中都实现自然的运动。SCAMPER使用一种具有两种不同先验的变分架构，即一种用于……的运动先验

    arXiv:2610.27279v1 Announce Type: new  Abstract: The development of embodied models of animal behavior is a major goal of the neuro-AI community. Embodied models will enhance our ability to make direct comparisons between biological and artificial agents. To make such comparisons effective, though, we need embodied models that actually move like animals. Previous work has developed virtual animal bodies and techniques for training realistic motion through imitation. However, to date, there is no system for embodied animal modeling that produces natural movements both during trained goal-directed behavior, and during exploration driven by random generation. Here, we introduce Simulated Control via Augmented Motor Priors for Embodied Reinforcement learning (SCAMPER), a system for training an embodied virtual rat model, enabling natural movements during both goal-directed behavior and random exploration. SCAMPER uses a variational architecture with two distinct priors, a motion prior for 
    
[^245]: RefCraft：通过网页开发对视觉参考条件化的智能体生成进行基准测试

    RefCraft: Benchmarking Visual-Reference-Conditioned Agentic Generation through Web Development

    [https://arxiv.org/abs/2610.26826](https://arxiv.org/abs/2610.26826)

    该论文提出了RefCraft基准，包含300个将自然语言指令与参考网页截图相结合的多模态网页开发任务，用于评估智能体选择性地解读视觉参考并迭代优化生成结果的能力。

    

    arXiv:2610.26826v2 公告类型：replace-cross 摘要：除了基本的工具使用和编程技能之外，如何有效地将视觉参考融入智能体生成过程，对于将AI智能体应用于现实世界的生产力任务同样至关重要。我们通过网页开发来研究这一关键能力——这是一个实用且易于评估的场景，但现有基准在此方面大多仍停留在理想化状态：它们通常专注于重建视觉目标或仅根据文本指令生成网页，而忽视了基于视觉示例进行开发这一更为常见的工作流程。具体而言，我们提出了RefCraft，一个包含300个真实多模态网页开发任务的基准，这些任务将自然语言指令与一至五个参考网页的截图相结合。智能体必须在渲染输出时选择性地解读和利用这些参考，同时进行视觉检查并迭代优化其产物。我们使用独立的评估方法来评估所产生的开放式结果。

    arXiv:2610.26826v2 Announce Type: replace-cross  Abstract: Beyond basic tool use and coding skills, effectively incorporating visual references into agentic generation is equally essential for applying AI agents to real-world productivity tasks. We study this crucial capability through webpage development, a practical and readily evaluable setting in which existing benchmarks remain largely idealized: they typically focus on reconstructing visual targets or generating webpages from textual instructions alone, overlooking the more common workflow of developing from visual examples. Specifically, we introduce RefCraft, a benchmark comprising 300 realistic multimodal webpage-development tasks that combine natural-language instructions with screenshots from one to five reference webpages. Agents must selectively interpret and utilize these references while rendering, visually inspecting, and iteratively refining their outputs. We evaluate the resulting open-ended artifacts using independen
    
[^246]: X-Fly：在全动作块空中视觉语言导航中学习到达与停止

    X-Fly: Learning to Reach and Stop in Full-Chunk Aerial Vision-Language Navigation

    [https://arxiv.org/abs/2610.21450](https://arxiv.org/abs/2610.21450)

    X-Fly提出了一种分阶段训练框架，通过恢复DAgger标注恢复动作块与停止指令，再用Flow-GRPO基于闭环结果联合优化分块流控制器和伯努利STOP策略，从而解决全动作块空中视觉语言导航中偏离专家状态导致的恢复失败和无法停止问题，同时将每个路径点的推理成本从195毫秒降至42毫秒。

    

    基于大型视觉语言模型的空中视觉语言导航（VLN）智能体查询成本高昂。在重新规划前执行完整的预测动作块可以摊薄这一成本，将本策略每个已执行路径点的模型推理时间从195毫秒降低至42毫秒。但这也使策略被查询时的状态偏离了专家演示所覆盖的范围，而专家演示既不监督从这些状态的恢复，也不监督学习者自身到达时的终止行为。在OpenFly测试可见集上，“无法停止”失败占完整动作块行为克隆（BC）结果的47.78%。我们提出X-Fly，一个面向全动作块空中VLN的分阶段训练框架。恢复DAgger（Recovery DAgger）为学习者访问过的决策状态标注平滑的动作块（可重新衔接至未来的专家片段），并在第一个进入目标区域的状态标注STOP。随后，Flow-GRPO基于同任务的闭环结果，联合优化分块流控制器与伯努利STOP策略，并辅以结果可验证……（摘要原文在此处截断）

    arXiv:2610.21450v1 Announce Type: cross  Abstract: Aerial vision-language navigation (VLN) agents built on large vision-language models are expensive to query. Executing a complete predicted action chunk before replanning amortizes this cost, reducing model inference from 195 to 42 ms per executed waypoint for our policy. It also moves the states at which the policy is queried away from expert demonstrations, which supervise neither recovery from these states nor termination at the learner's own arrival. On OpenFly test-seen, no-stop failures account for 47.78% of full-chunk behavior cloning (BC) outcomes. We present X-Fly, a staged training framework for full-chunk aerial VLN. Recovery DAgger labels learner-visited decision states with smooth chunks that reconnect to future expert segments and with STOP at the first in-region state. Flow-GRPO then jointly refines the chunked flow controller and a Bernoulli STOP policy from same-task closed-loop outcomes, supplemented by outcome-verifi
    
[^247]: 面向高效且有效的无乘法器LLM推理的分数二次幂量化

    Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference

    [https://arxiv.org/abs/2610.21257](https://arxiv.org/abs/2610.21257)

    该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。

    

    大语言模型（LLM）需要高效的推理，其中线性投影中的乘累加（MAC）运算主导了计算和能耗成本。训练后量化（PTQ）通过将权重和激活映射到低位整数来降低这一成本。二次幂权重量化通过移位-加法逻辑有望实现高效的无乘法器LLM推理，但由于其粗糙的以2为基数的量化网格，在低位宽下会出现显著的精度下降。我们引入了分数二次幂量化（FPoT），这是一种以√2为基数（半八度）的分数二次幂网格，在保持完全整数移位-加法数据通路的同时，实现了实用的4位权重量化。该方法结合了：(i) 以√2为基数的权重网格与均匀整数激活相结合，(ii) 在校准阶段进行并被吸收到静态逐通道缩放因子中的网格对齐细化，以及 (iii) 轻量级的单加法器√2近似，其误差在权重量化过程中被吸收。

    arXiv:2610.21257v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demand efficient inference, where the multiply-accumulate (MAC) operations in linear projections dominate compute and energy cost. Post-training quantization (PTQ) reduces this cost by mapping weights and activations to low-bit integers. Power-of-Two (PoT) weight quantization promises efficient multiplier-free LLM inference via shift-add logic, but suffers significant accuracy degradation at low bitwidths due to its coarse base-2 quantization grid. We introduce Fractional PoT (FPoT), a base-$\sqrt{2}$ (half-octave) fractional PoT grid that achieves practical 4-bit weight quantization while preserving a fully integer shift-add datapath. The method combines (i) the base-$\sqrt{2}$ weight grid with uniform integer activations, (ii) a calibration-time grid-alignment refinement absorbed into the static per-channel scale, and (iii) a lightweight 1-adder $\sqrt{2}$ approximation whose error is absorbed during weig
    
[^248]: 重新审视扩散语言模型与自回归语言模型之间16倍的预训练计算差距

    Revisiting the 16x Pretraining Compute Gap between Diffusion and Autoregressive Language Models

    [https://arxiv.org/abs/2610.17590](https://arxiv.org/abs/2610.17590)

    本研究采用现代扩散语言模型训练实践（精确似然评估、块扩散训练、编码器-解码器架构）重新审视扩散语言模型的缩放定律，通过约1700个模型的大规模实验发现，扩散语言模型与自回归语言模型之间所谓的16倍预训练计算差距远小于此前声称的水平。

    

    扩散语言模型因其快速、并行的解码能力而受到广泛关注，但人们普遍认为其训练效率远低于自回归语言模型（ARMs）：一项被广泛引用的研究报告指出，扩散语言模型需要大约16倍于自回归模型的训练计算量才能达到相同的似然水平，这一数字使得从零开始预训练扩散语言模型在大规模应用上显得望而却步。我们发现，这一数字背后的研究方法已经与当前扩散语言模型的实践脱节：它使用证据下界（ELBO）来评估模型，并训练全序列的仅解码器模型。因此，我们重新审视了扩散语言模型的缩放定律，采用了模型自身解码顺序下的精确似然、块扩散训练、编码器-解码器架构，并对实践中使用的块大小进行了扫描。在这一新协议下，我们研究了十个模型变体、总计约1700个模型，发现预训练扩散语言模型的计算成本远……

    arXiv:2610.17590v1 Announce Type: cross  Abstract: Diffusion language models (DLMs) have attracted wide attention for their fast, parallel decoding, but they are believed to be far less efficient to train than autoregressive language models (ARMs): a widely cited study reports that a DLM needs roughly 16x the training compute of an ARM to match its likelihood, a figure that has made pretraining DLMs from scratch look prohibitive at scale. We find that the methodology behind this figure has fallen out of step with current DLM practice: it scores the model with its evidence lower bound (ELBO) and trains a full-sequence, decoder-only model. We therefore revisit the scaling laws of DLMs, adopting the exact likelihood under the model's own decoding order, block diffusion training, an encoder-decoder architecture, and a sweep over the block sizes used in practice. Under this new protocol, we study across ten model variants, about 1,700 models in total, and find that pretraining DLMs is far c
    
[^249]: DTA-GT：用于神经网络表示学习的方向与拓扑感知图Transformer

    DTA-GT: Direction- and Topology-Aware Graph Transformer for Neural Network Representation Learning

    [https://arxiv.org/abs/2610.16822](https://arxiv.org/abs/2610.16822)

    提出DTA-GT，一种方向与拓扑感知的图Transformer，通过在节点初始化、谱编码、注意力交互和特征更新全流程中保持有向计算语义，显著提升了神经架构搜索中精度与延迟预测的性能。

    

    架构属性预测器通过从小规模已评估的候选集中估计精度或延迟，来降低神经架构搜索的成本。其关键难点在于神经架构是带有操作标签的有向无环图（DAG）：边的方向定义了计算流，但现有预测器往往仅将方向视为局部结构线索。我们提出了DTA-GT，一种基于编码器级方向一致性的方向与拓扑感知图Transformer，它在节点初始化、全局谱编码、成对交互和特征更新等各个环节中都保持了有向计算语义。DTA-GT通过方向与拓扑感知的节点初始化、磁拉普拉斯谱编码、方向对偶结构注意力以及方向敏感的MoE-FFN来实现这一原则。在NAS-Bench-101和NAS-Bench-201上的精度与延迟预测任务中，DTA-GT持续优于代表性的序列模型、GNN和Transformer方法。

    arXiv:2610.16822v1 Announce Type: cross  Abstract: Architecture attribute predictors reduce the cost of neural architecture search by estimating accuracy or latency from a small set of evaluated candidates. The key difficulty is that neural architectures are operation-labeled DAGs: edge direction defines computational flow, but existing predictors often treat direction as a local structural cue. We propose DTA-GT, a Direction- and Topology-Aware Graph Transformer based on encoder-wide directional consistency, which preserves directed computational semantics across node initialization, global spectral encoding, pairwise interaction, and feature update. DTA-GT realizes this principle with direction- and topology-aware node initialization, Magnetic Laplacian spectral encoding, direction-dual structural attention, and a Direction-Sensitive MoE-FFN. Across accuracy and latency prediction on NAS-Bench-101 and NAS-Bench-201, DTA-GT consistently outperforms representative sequence-, GNN-, Tran
    
[^250]: RetryBench：在GPT工具使用中将重试许可与恢复结果相分离

    RetryBench: Separating Retry Permission from Recovery Outcomes in GPT Tool Use

    [https://arxiv.org/abs/2610.15884](https://arxiv.org/abs/2610.15884)

    该论文提出RetryBench基准，将GPT工具使用中“虚假重试许可”（即在契约不支持时重试超时工具调用）这一错误与实际恢复结果分离，发现即使在受保护条件下能成功恢复，模型也无法可靠地拒绝不受支持的重试。

    

    虚假重试许可（FRP）是指智能体在可见契约不支持的情况下重试超时的工具调用。RetryBench将这一错误与实际实现的恢复分离开来。成功的受保护恢复并不意味着能够可靠地拒绝不受支持的重试。在一项720次响应的密钥对照实验中，有效密钥通常能实现安全推进，但被提供的已更改密钥在Sol（1/80）、Terra（55/80）和Luna（60/80）中引发了不受支持的重试。随后的一项240次响应对照实验移除了候选密钥并要求模型自行提供密钥，Luna仍进行了8/80次不受支持的调度；Sol和Terra在各自80次响应中均未出现此类情况。因此，接受显示的错误密钥无法解释所有观察到的错误，尽管更改的输出要求和条件使得纯粹的显示效应比较无法进行。在一项六终点研究中，移除弱契约标签增加了Sol和Terra在编码动作条件下的重试次数。另一项独立的1,536次响应的规范动作……（原文摘要在此处截断）

    arXiv:2610.15884v1 Announce Type: new  Abstract: False retry permission (FRP) occurs when an agent retries a timed-out tool call without support from the visible contract. RetryBench separates this error from realized recovery. Successful protected recovery does not imply reliable rejection of unsupported retries. In a 720-response key control, valid keys usually enabled safe progress, but offered changed keys elicited unsupported retries in Sol (1/80), Terra (55/80), and Luna (60/80). A later 240-response control removed the candidate and required models to supply a key. Luna still made 8/80 unsupported dispatches; Sol and Terra made none in 80 responses each. Accepting a displayed bad key therefore cannot explain all observed errors, although changed output requirements and conditions preclude a pure display-effect comparison. Removing the weak-contract label increased retries in Sol and Terra's coded-action conditions in a six-endpoint study. A separate 1,536-response canonical-acti
    
[^251]: SimForcing：将仿真运动先验蒸馏到真实域机器人世界模型中

    SimForcing: Distilling Simulation Motion Priors into Real-Domain Robot World Models

    [https://arxiv.org/abs/2610.06598](https://arxiv.org/abs/2610.06598)

    SimForcing通过潜空间运动蒸馏和多块仿真条件化（带条件丢弃）机制，将仿真中的运动先验迁移到真实域机器人世界模型中，同时规避了外观差异和仿真预测不准确带来的负面影响。

    

    动作条件化的机器人世界模型必须对机器人轨迹做出精确响应，同时保持逼真的视觉动态，然而从异构的机器人视频中同时学习这两者仍然具有挑战性。仿真可以提供结构化的运动监督，但外观差异阻碍了直接迁移，且不准确的仿真预测可能会误导真实视频的生成。我们提出了SimForcing，这是一个仿真引导的框架，它将仿真既作为可迁移运动知识的来源，又作为预测的可控参考。首先，我们通过潜空间运动蒸馏从仿真教师模型中迁移运动知识，在潜空间中对齐时间变化以内化运动先验，同时减轻外观差异的影响。其次，我们引入了带条件丢弃的多块仿真条件化机制，以利用预测的仿真轨迹，而不会过度依赖其准确性。

    arXiv:2610.06598v2 Announce Type: replace-cross  Abstract: Action-conditioned robot world models must respond precisely to robot trajectories while preserving realistic visual dynamics, yet learning both from heterogeneous robot videos remains challenging. Simulation offers structured motion supervision, but appearance differences hinder direct transfer, and inaccurate simulation predictions can misguide real-video generation. We present SimForcing, a simulation-guided framework that uses simulation both as a source of transferable motion knowledge and as a controllable reference for prediction. First, we transfer motion knowledge from a simulation teacher through latent-motion distillation, aligning temporal changes in latent space to internalize motion priors while mitigating the influence of appearance differences. Second, we introduce multi-block simulation conditioning with condition dropout to exploit predicted simulation trajectories without relying excessively on their accuracy
    
[^252]: MASBench：部分可观测性下基于大语言模型的多智能体协作基准测试

    MASBench: Benchmarking LLM-based Multi-Agent Collaboration under Partial Observability

    [https://arxiv.org/abs/2610.04672](https://arxiv.org/abs/2610.04672)

    该论文提出了MASBench，首个在部分可观测性约束下系统评估基于大语言模型的多智能体系统协作机制的基准测试，填补了现有基准假设全局可观测性的空白。

    

    大语言模型（LLM）已逐步演变为自主智能体的核心。在这一进展的基础上，基于LLM的多智能体系统（MAS）将多个智能体协调成一个协同团队，以完成超出单个智能体能力范围的复杂任务。此类系统的有效性不仅取决于智能体本身，还取决于协作机制的设计与组织方式。值得注意的是，现实世界中的协作通常是部分可观测的，即由于物理或隐私相关的限制，每个智能体只能获取环境中的部分信息。然而，许多现有的多智能体基准测试都假设全局可观测性，对系统性评估协作机制的支持十分有限。为弥合这一差距，我们提出了MASBench——一个在部分可观测约束下设计的多智能体协作基准测试。该基准由三个渐进……（原文摘要至此截断）

    arXiv:2610.04672v2 Announce Type: replace  Abstract: Large language models (LLMs) have progressively evolved into the core of autonomous agents. Building on this progress, LLM-based multi-agent systems (MAS) coordinate multiple agents into a synergistic team to accomplish complex tasks that exceed the capabilities of individual agents. The effectiveness of such systems depends not only on the agents themselves, but also on how collaboration mechanisms are designed and organized. Note that real-world collaboration is typically partially observable, where each agent can only access partial information about the environment due to physical or privacy-related constraints. However, many existing multi-agent benchmarks assume global observability, and leave limited support for systematically evaluating collaboration mechanisms. To bridge this gap, we introduce MASBench, a multi-agent collaboration benchmark designed under partially observable constraints. It is organized into three progressi
    
[^253]: MEA：一个奖励驱动的多智能体系统，用于忠实的模型解释

    MEA: A Reward-Driven Multi-Agent System for Faithful Model Explanations

    [https://arxiv.org/abs/2610.02480](https://arxiv.org/abs/2610.02480)

    MEA通过Proposer智能体自动选择和配置解释工具、Actor智能体以忠实性为目标进行端到端奖励优化，完全消除了使用机器学习解释方法的知识壁垒，能够跨表格、文本和视觉模态生成忠实的自然语言模型解释。

    

    近年来，大量机器学习（ML）模型被应用于高风险领域，但对于依据这些模型预测结果采取行动的从业者来说，模型在很大程度上仍然是不透明的。虽然事后解释方法为理解模型行为提供了一个视角，但有效使用这些方法需要大多数领域专家所不具备的专业知识：处理高维输出、选择最佳解释，以及综合来自不同工具的证据。为此，我们提出了MEA，一个完全消除解释知识壁垒的多智能体框架：Proposer（提议者）智能体根据问题和模态选择并配置解释工具，而Actor（执行者）智能体则针对忠实性进行端到端优化，将输出转化为基于表格、文本和视觉等多种模态下模型行为的自然语言解释。此外，我们引入了涵盖特征归因、反事实推理等多种类型的问题（摘要原文在此处截断）。

    arXiv:2610.02480v1 Announce Type: new  Abstract: Recent years have seen the employment of a plethora of machine learning (ML) models in high-stakes domains, but they remain largely opaque to the practitioners who act on their predictions. While post-hoc explanation methods offer a lens into this model behavior, wielding them effectively demands expertise most domain experts lack: navigating high-dimensional outputs, selecting the best explanations, and synthesizing evidence across disparate tools. To this end, we present MEA, a multi-agent framework that removes the explanation knowledge barrier entirely: a Proposer agent selects and configures explanation tools based on the question and modality, while an Actor agent is optimized end-to-end against faithfulness, transforming the outputs into natural language explanations grounded in model behavior across tabular, text, and vision modalities. Further, we introduce diverse question types spanning feature attribution, counterfactual reas
    
[^254]: 零计算跨语言可迁移性估计：基于类型学特征代理

    Zero-Compute Cross-Lingual Transferability Estimation Using Typological Feature Proxies

    [https://arxiv.org/abs/2609.39640](https://arxiv.org/abs/2609.39640)

    仅利用免费的语言类型学特征训练随机森林，即可在零计算成本下准确预测跨语言迁移性（留一语言 ρ=0.705、R²=0.49），证明类型学数据库蕴含廉价而密集的迁移信号，可替代昂贵的多语言预训练测量。

    

    跨语言迁移描述了源语言中的知识如何惠及目标语言。对迁移进行定量测量需要广泛的多语言预训练，正如先前工作通过跨语言迁移矩阵所做的那样。我们探究迁移是否可以从免费获取的类型学特征中预测出来，以及高资源源语言的突出优势究竟是源于类型学因素，还是源于数据的质量与数量。我们证明，类型学数据库中蕴含着关于跨语言迁移的廉价且密集的信号。在先前工作给出的24种语言迁移矩阵上，仅使用类型学特征的随机森林在留一语言协议下取得了 ρ=0.705 和 R²=0.49 的成绩，优于非类型学对照组的 ρ=0.62，这验证了仅凭类型学预测即可重建昂贵的实测跨语言迁移能力。该信号在留一文字系统和留一语系协议下依然存在，因此文字系统和语系的混杂因素无法解释这一效应（原文在此处截断）。

    arXiv:2609.39640v1 Announce Type: cross  Abstract: Cross-lingual transfer describes how knowledge in a source language benefits a target language. Measuring it quantitatively requires broad multilingual pre-training, as prior work has done with cross-lingual transfer matrices. We ask whether transfer is predictable from freely available typological features, and whether the prominence of high-resource source languages reflects typology or data quality and quantity. We show that typological databases contain cheap and dense signals about cross-lingual transfer. Our typology-only random forest on a 24-language prior-work transfer matrix scores leave-one-language-out $\rho{=}0.705$ and $R^2{=}0.49$, beating a non-typological control at $\rho{=}0.62$, which verifies the ability of typology-only predictions to reconstruct costly measured cross-lingual transfer. The signal survives leave-one-script-out and leave-one-family-out protocols, so script and family confounding do not explain the ef
    
[^255]: AutoBCI：面向基于EEG脑机接口的预测引导式智能体神经架构发现

    AutoBCI: Forecast-Guided Agentic Neural Architecture Discovery for EEG-Based Brain-Computer Interfaces

    [https://arxiv.org/abs/2609.35456](https://arxiv.org/abs/2609.35456)

    AutoBCI通过设计者智能体自动生成并改进EEG解码架构，同时由预测者智能体根据早期学习曲线预测架构的最终性能以筛选优秀候选，从而在情绪识别、运动想象和睡眠分期等14个EEG数据集上实现高效的脑机接口神经架构自动发现。

    

    基于EEG的脑机接口支持广泛的应用，然而设计在多样化任务中均能表现良好的解码架构仍然具有挑战性。我们提出了AutoBCI，这是一个智能体框架，由设计者智能体和预测者智能体协同支持跨任务的EEG解码架构的发现与选择。设计者智能体执行池引导架构发现（PGAD），通过在多个EEG任务（如情绪识别、运动想象和睡眠分期）上进行训练和验证来生成并改进架构。预测者智能体执行基于早期知识的性能估计（PEEK），利用架构代码、训练协议和早期学习曲线来预测全预算下的验证性能，并选出有前景的候选架构进行后续训练。在涵盖运动想象、情绪识别和睡眠分期的14个EEG数据集上，我们评估了AutoBCI……（原文摘要此处截断）

    arXiv:2609.35456v2 Announce Type: replace  Abstract: EEG-based brain-computer interfaces support a broad range of applications, yet designing decoding architectures that perform well across diverse tasks remains challenging. We introduce AutoBCI, an agentic framework in which a Designer Agent and a Forecaster Agent support the discovery and selection of EEG decoding architectures across tasks. The Designer Agent performs Pool-Guided Architecture Discovery (PGAD), generating and refining architectures through training and validation across multiple EEG tasks, such as emotion recognition, motor imagery, and sleep staging. The Forecaster Agent performs Performance Estimation from Early Knowledge (PEEK), using architecture code, the training protocol, and early learning curves to predict full-budget validation performance and select promising candidates for continued training. Across 14 EEG datasets spanning motor imagery, emotion recognition, and sleep staging, we evaluate AutoBCI with si
    
[^256]: 面向AI系统的可靠性工程：挑战、方法与方向

    Reliability Engineering for AI Systems: Challenges, Methods, and Directions

    [https://arxiv.org/abs/2609.35316](https://arxiv.org/abs/2609.35316)

    该论文将传统可靠性工程方法（如失效定义、FMEA、加速测试、现场监控与可靠性增长）适配到AI系统，提出四级失效诊断框架，主张超越平均基准准确率来系统性地保障AI系统整体的可靠性。

    

    AI可靠性关注的是AI系统是否能在规定的时期内、在规定的运行条件下，以明确的证据可靠地执行其预期功能。随着这些系统变得更加自主，这一功能应不仅限于产生正确的输出。检索、记忆、工具使用、权限、人类监督以及系统之间的交互必须一致且安全地运行，而对于生成式AI而言，产生输出的推理过程也必须如此。平均基准准确率仅衡量系统能力，并不能量化这一更广泛的可靠性主张。本文将成熟的可靠性工程方法——从失效定义、运行包线，到失效模式与影响分析（FMEA）、加速测试、现场监控和可靠性增长——适配到AI系统中。论文提出了一个四级诊断框架，将失效分类为组件级失效、运行回路级失效、智能体行为级失效，以及网络与治理级失效。测试与评……

    arXiv:2609.35316v3 Announce Type: replace  Abstract: AI reliability concerns whether an AI system performs its intended function dependably over a stated period and under stated operating conditions, with clear evidence. As these systems become more autonomous, this function should include more than only a correct output. Retrieval, memory, tool use, permissions, human oversight, and interactions among systems must operate consistently and safely, and, for a generative AI, so must the reasoning process that produces the output. Average benchmark accuracy only measures system capability; it does not quantify this broader reliability claim. This paper adapts established reliability engineering methods, from failure definitions and operational envelopes to FMEA, accelerated testing, field monitoring, and reliability growth, to AI systems. A four-level diagnostic framework classifies failures as component, operational-loop, agentic-conduct, or network and governance failures. Test, evaluat
    
[^257]: STRIDE：面向多样化情境的文本-轨迹对齐自动化评估

    STRIDE: Automated Evaluation of Text-to-Trajectory Alignment across Diverse Contexts

    [https://arxiv.org/abs/2609.34799](https://arxiv.org/abs/2609.34799)

    该论文提出首个评估场景文本描述与行人轨迹之间情境对齐的自动化框架STRIDE，通过源自社会学理论的VRDST评估协议，解决了传统依赖真实人类数据的评估方法在多样化情境下无法扩展的难题。

    

    语言条件化的轨迹生成已经问世，但其评估方法却未能跟上步伐。现有的行人轨迹指标是将生成轨迹与真实世界的人类数据进行比较。这种方法无法扩展到跨多样化情境的文本到轨迹生成任务，因为为每个场景收集人类轨迹既昂贵又不可行。此外，行人行为具有异质性且依赖于具体情境，不存在单一的“正确答案”式指标，而当前的评估框架也无法迁移到这一领域。这些挑战使得可扩展且可靠的评估变得十分困难。我们提出了STRIDE，这是首个用于评估场景描述与行人轨迹之间情境对齐程度的框架。STRIDE通过三项设计选择来应对这些挑战。首先，我们从社会学理论中推导出VRDST评估协议，以定义完整的评估空间。其次，它将高层次情境分解为场景……

    arXiv:2609.34799v2 Announce Type: replace  Abstract: Language-conditioned trajectory generation is here, but its evaluation has not kept pace. Existing pedestrian trajectory metrics compare trajectories with real-world human data. This does not scale to text-to-trajectory generation across diverse contexts, as collecting human trajectories for every scenario is costly and infeasible. Moreover, pedestrian behavior is heterogeneous and context-dependent, with no single metric as the correct answer, and current evaluation frameworks are not transferable to this domain. These challenges make scalable, reliable evaluation difficult. We introduce STRIDE, the first framework for evaluating context alignment between scenario descriptions and pedestrian trajectories. STRIDE addresses these challenges through three design choices. First, we derive our VRDST evaluation protocol from sociological theories to define a complete evaluation space. Second, it decomposes high-level context into scenario
    
[^258]: 理解泛化需要普适归纳

    Understanding Generalization Requires Universal Induction

    [https://arxiv.org/abs/2609.34458](https://arxiv.org/abs/2609.34458)

    论文指出经典统计理论因依赖无法证成的手工归纳偏置而无法解释通用AI的泛化成功，并提出将所罗门诺夫归纳相对化到信息视角、偏向能访问所有既有信息的短程序，从而以普适归纳重新框架化归纳偏置问题。

    

    经典统计理论不足以解释通用人工智能模型的成功，因为它依赖于无法自圆其说的手工设计的归纳偏置。没有免费午餐（No Free Lunch, NFL）定理表明，任何在某些环境中胜过随机的学习器，必然在其他环境中表现不佳。我们或许希望过去的经验能指明应当预期哪些环境，但NFL定理同样适用于元学习。因此，任何能做出有意义预测的方法，都必然以数据之外的归纳偏置为起点。若选择偏向短程序的归纳偏置，便会得到所罗门诺夫归纳（Solomonoff Induction, SI），其性能可与所有可计算的学习器相媲美——尽管其中存在的“常数”在与利用背景信息的专门化方法进行比较时会变得很大。因此，我们将SI相对化到一个信息视角上，偏向于能够访问所有既有信息的短程序。这一做法重新框架化了归纳偏置问题（摘要原文在此处截断）。

    arXiv:2609.34458v2 Announce Type: replace-cross  Abstract: Classical statistical theory is insufficient to explain the successes of general-purpose AI models, because it depends on handcrafted inductive biases that it cannot justify. No Free Lunch (NFL) theorems force any learner that beats chance on some environments to underperform on others. We might hope that past experience informs which environments to expect, but NFL applies equally to meta-learning. Thus, any method that makes meaningful predictions necessarily begins with an inductive bias external to the data. Choosing to bias toward short programs yields Solomonoff induction (SI), whose performance is competitive against all computable learners - albeit up to "constants" that become large when comparing against specialized methods that exploit background information. We therefore relativize SI to an information vantage point, biasing toward short programs with access to all preexisting information. This reframes the inductiv
    
[^259]: 共情强化学习中不断演化的支持优先级

    Evolving Support Priorities in Empathetic Reinforcement Learning

    [https://arxiv.org/abs/2609.34249](https://arxiv.org/abs/2609.34249)

    提出 CARE 方法，通过在每轮对话中动态调整认知、情感与主动共情三个维度的权重及细粒度评估标准，生成情境自适应的奖励准则，解决了共情强化学习中支持优先级随对话演化而奖励规范固定不变的根本错配问题，并在多个基准上取得最先进性能。

    

    我们发现共情强化学习中存在一个根本性的错配：支持优先级会随对话状态而演化，然而现有方法通常优化的是在各轮对话中保持固定的预定义奖励规范。为了对这些不断演化的支持优先级进行建模，我们将共情支持沿认知共情、情感共情和主动共情三个维度加以组织，并提出了情境自适应准则演化方法。在每一轮对话中，CARE 通过调整这三个共情维度的权重及其细粒度评估标准，生成一个情境自适应的评估准则。该准则生成器先通过监督微调、再通过基于偏好的强化学习，利用轮级准则监督信号和人类偏好数据进行训练，随后作为在线共情强化学习的自适应奖励接口。与 RLVER 和 MICA 集成后，CARE 在 SentientBench、EQBench3 和 EMPA 上取得了最先进的性能。

    arXiv:2609.34249v3 Announce Type: replace  Abstract: We identify a fundamental mismatch in empathetic reinforcement learning: support priorities evolve with the dialogue state, yet existing methods typically optimize predefined reward specifications that remain fixed across turns. To model these evolving support priorities, we organize empathetic support along cognitive, affective, and proactive empathy, and propose Context-Adaptive Rubric Evolution (CARE). At each turn, CARE generates a context-adaptive rubric by adjusting both the weights of these three empathy dimensions and their fine-grained evaluation criteria. The rubric generator is trained with turn-level rubric supervision and human preference data through supervised fine-tuning followed by preference-based reinforcement learning, and then serves as an adaptive reward interface for online empathetic RL. Integrated with both RLVER and MICA, CARE achieves state-of-the-art performance across SentientBench, EQBench3, and EMPA und
    
[^260]: 解耦知识与隐私：面向大语言模型持续学习的任务后自蒸馏重放

    Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning

    [https://arxiv.org/abs/2609.29711](https://arxiv.org/abs/2609.29711)

    该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。

    

    隐私保护的持续学习（PPCL）必须在跨序列任务保留有用知识的同时，减少敏感内容的再现。形式化隐私保证刻画的是随机化机制，而操作性输出控制关注的则是训练后的模型能否选择性地降低其输出中敏感内容出现的可能性。在这项工作中，我们在现实的任务演化情境下，将后者与持续学习效用放在一起进行研究。知识保留与隐私修正作用于不同的粒度：任务获取需要广泛保留当前任务和旧任务的行为，而隐私修正则针对稀疏的标注位置。联合优化会导致当前任务的保留目标不断变化。我们提出了SPARK，一种保留-修正分解方法，它首先冻结学习到的任务后分布，然后围绕这一稳定参考进行选择性修正。

    arXiv:2609.29711v1 Announce Type: cross  Abstract: Privacy-preserving continual learning (PPCL) must reduce the reproduction of sensitive content while retaining useful knowledge across sequential tasks. Formal privacy guarantees characterize randomized mechanisms, whereas operational output control concerns whether a trained model selectively reduces the likelihood of sensitive content in its outputs. In this work, we investigate the latter together with continual-learning utility under realistic task evolution. Retention and privacy correction operate at different granularities: task acquisition requires broad preservation of current- and old-task behavior, whereas privacy correction targets sparse annotated positions. Joint optimization leaves the current-task preservation target continually changing. We propose SPARK, a retention-correction decomposition that first freezes the learned post-task distribution and then applies selective correction around this stable reference. Self-Di
    
[^261]: PolicyMem：用于大语言模型治理的几何策略记忆

    PolicyMem: Geometric Policy Memory for LLM Governance

    [https://arxiv.org/abs/2609.13734](https://arxiv.org/abs/2609.13734)

    PolicyMem提出了一种几何策略记忆框架，将自然语言策略外化为共享表示空间中由低秩子空间表示的可重用几何记忆对象，解决了现有LLM治理方法无法将策略作为可重用操作状态的问题，从而支持在检测、干预和验证之间一致地重用策略证据。

    

    随着大语言模型（LLM）越来越多地部署在现实世界的高风险应用中，有效的治理变得至关重要。现有的安全防护措施主要遵循两种范式：基于学习的防护器提供强大的语义判别能力，但将策略行为与训练的模型和分类体系耦合在一起；而可编程框架提供灵活的控制，但需要大量的手动提示和工作流工程。这两种方法都没有将策略外化为可重用的操作状态，导致难以在检测、干预和验证之间一致地重用策略证据。在本文中，我们提出了PolicyMem，这是一种几何策略记忆，它将自然语言策略外化为可重用的几何记忆对象，这些对象由共享表示空间中的低秩子空间表示。记忆写入器将自然语言策略编译为策略记忆槽，查询-响应对通过……（摘要原文不完整）读取策略记忆。

    arXiv:2609.13734v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly deployed in real-world high-stakes applications, effective governance has become essential. Existing safeguards largely follow two paradigms: learning-based guards provide strong semantic discrimination but couple policy behavior to trained models and taxonomies, while programmable frameworks offer flexible control but require substantial manual prompt and workflow engineering. Neither externalizes policies as reusable operational states, making it difficult to consistently reuse policy evidence across detection, intervention, and verification. In this paper, we introduce PolicyMem, a geometric policy memory that externalizes natural-language policies as reusable geometric memory objects represented by low-rank subspaces in a shared representation space. A memory writer compiles natural-language policies into policy memory slots, and query-response pairs read the policy memory through pr
    
[^262]: MAIA：面向艺术委托需求发现的多智能体意图表达系统

    MAIA: Multi-Agent Intent Articulation for Requirement Discovery in Art Commissions

    [https://arxiv.org/abs/2609.12097](https://arxiv.org/abs/2609.12097)

    MAIA是一个多智能体系统，通过苏格拉底式提问和“验证优先于发明”规则，帮助艺术委托中的外行客户将模糊的情感意图表达为可验证的视觉术语简报，实验证明其在认知支持方面显著优于基线。

    

    在定制艺术委托中，外行人知道自己感受到什么，却缺乏语言来明确表达：一位参与者希望将一名下岗卡车司机描绘成“自己机器中的幽灵”，却没有说明媒介、尺寸和色调。我们将此框定为需求发现这一长期被忽视的上游环节的表达瓶颈——该环节先于任何艺术家或图像生成器，迫使委托人首先构建自己的意图。我们提出了MAIA（多智能体意图表达），这是一个多智能体系统，在“验证优先于发明”规则下通过苏格拉底式提问来支撑这一阶段，将模糊的情感转化为用户可验证的纯文本视觉术语简报。在一项被试内研究（N = 16）中，完整配置相比最小基线在认知支持方面产生了大幅且显著的提升（r = 0.97, p_FDR = 0.003; LMM p_FDR < 0.001）。主题分析揭示了相同的机制，验证器关卡……

    arXiv:2609.12097v2 Announce Type: replace-cross  Abstract: In bespoke art commissions, laypeople know what they feel but lack the words to specify it: one participant wanted a laid-off truck driver depicted as "a ghost in his own machine" but left the medium, scale, and palette unsaid. We frame this as an articulation bottleneck at an under-served upstream stage: requirement discovery, which precedes any artist or image generator and forces the commissioner to constitute intent in the first place. We present MAIA (Multi-Agent Intent Articulation), a multi-agent system that scaffolds this stage through Socratic inquiry under a "Verification over Invention" rule, turning vague affect into a text-only brief of visual terms the user verifies. In a within-subjects study (N = 16), the full configuration produced a large, significant gain in Cognitive Support over a minimal baseline (r = 0.97, p_FDR = 0.003; LMM p_FDR < 0.001). Thematic analysis traces the same mechanism, and a validator gate
    
[^263]: LongCounsel-8：基于多轮次心理咨询对话的纵向抑郁症追踪基准套件

    LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues

    [https://arxiv.org/abs/2609.03507](https://arxiv.org/abs/2609.03507)

    提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。

    

    从多轮次心理咨询对话中追踪抑郁症，需要同时估计当前的症状严重程度以及其在不同会话之间的变化情况。然而，该任务的进展受限于缺乏具有标准化会话级抑郁标签的纵向咨询数据。现有资源通常要么提供没有抑郁标签的多轮次对话，要么仅提供单次会话的标注访谈。构建这样一个基准面临三大挑战：保持纵向的一致性与多样性，将症状进展建立在实证模式之上，以及在不暴露目标标签的前提下自然地表达受控的抑郁状态。为应对这些挑战，我们提出了LongCounsel-8，这是一个由三个独立生成的数据集组成的基准套件，共包含7,749条五轮次咨询轨迹，其建立在真实的来访者档案、抑郁轨迹、症状构成和咨询模式的基础之上。

    arXiv:2609.03507v1 Announce Type: cross  Abstract: Tracking depression from multi-session counseling dialogues requires estimating both current symptom severity and how it changes across sessions. Yet progress on this task is constrained by the scarcity of longitudinal counseling data with standardized session-level depression labels. Existing resources typically provide either multi-session conversations without depression labels or labeled interviews in a single session. Building such a benchmark poses three challenges: maintaining longitudinal consistency and diversity, grounding symptom progression in empirical patterns, and expressing controlled depression states naturally without exposing target labels. To address these challenges, we introduce LongCounsel-8, a benchmark suite of three independently generated datasets totaling 7,749 five-session counseling trajectories, grounded in real-world client profiles, depression trajectories, symptom compositions, and counseling patterns.
    
[^264]: CivBench：面向《文明VI》中工具介导智能体的长时程基准测试

    CivBench: A Long-Horizon Benchmark for Tool-Mediated Agents in Civilization VI

    [https://arxiv.org/abs/2609.02459](https://arxiv.org/abs/2609.02459)

    CivBench是一个基于《文明VI》游戏和MCP协议的开源长时程基准测试，通过300多回合、数千次工具调用的复杂任务评估语言模型智能体，并创新性地提出主动监控率和RAG@10两个接口级指标来刻画智能体行为。

    

    我们提出了CivBench，这是一个开源基准测试，用于通过模型上下文协议评估语言模型智能体在长时程、工具介导环境中的表现。单个回合跨越300多个回合，在大动作空间中产生数千次工具调用，要求智能体在部分可观测性条件下进行持续的规划、状态监控和执行。该环境提供了76个MCP工具，以及一个将视觉游戏状态转换为结构化文本的叙述层。我们使用CivBench在23次有效运行中对四个模型系列的智能体行为进行了表征。该样本属于试点研究，而非模型排名：在此规模下，总体结果无法可靠地区分不同模型。相反，我们引入了两个环境可测量的接口级指标：主动监控率，用于捕捉智能体是否主动查询潜在的战略状态；以及RAG@10，用于捕捉智能体在结构化规划中所陈述的承诺是否反映了……

    arXiv:2609.02459v1 Announce Type: new  Abstract: We present CivBench, an open-source benchmark for evaluating language model agents in long-horizon, tool-mediated environments through the Model Context Protocol (MCP). A single episode spans 300+ turns and produces thousands of tool calls over a large action space, requiring sustained planning, state monitoring, and execution under partial observability. The environment exposes 76 MCP tools and a narration layer that converts visual game state into structured text.   We use CivBench to characterise agent behaviour across four model families in 23 admissible runs. The sample is a pilot, not a model ranking: aggregate outcomes do not reliably discriminate models at this scale. Instead, we introduce two interface-level metrics that the environment makes measurable: Proactive Monitoring Rate (PMR), capturing whether agents actively query latent strategic state, and RAG@10, capturing whether commitments stated in structured planning reflecti
    
[^265]: 利用认知能力画像评估人工智能对职场任务的适用性

    Using profiles of cognitive capability to assess AI suitability for workplace tasks

    [https://arxiv.org/abs/2608.25623](https://arxiv.org/abs/2608.25623)

    该论文提出一种基于共享核心认知能力为AI智能体与任务共同画像的流水线，通过结合基准测试推断的智能体能力与领域专家评估的任务需求权重，动态估算AI在领域、组织、角色或个人职责层面的任务适用性。

    

    部署AI的组织面临一个范围界定问题：哪些任务可以自动化，哪些应该留给人类，哪些最好由两者共同完成。总体基准分数对于系统在实践中将在何处成功或失败几乎无法提供洞见，而人类对模型能力的判断又很快过时。我们引入了一个流水线，使用一组共享的核心认知能力对智能体和任务进行画像。认知能力画像通过智能体在基准测试组上的表现推断其能力，该测试组对每个条目的认知需求进行了标注。任务需求加权则从领域专家处获取这些相同能力对其工作的相对重要性。由于两者使用一套共同的认知维度，因此可以随着模型和角色的变化独立更新，并可结合起来在领域、组织、角色或个人职责层面估算AI的适用性。我们对能力进行了验证……（原文摘要在此处截断）

    arXiv:2608.25623v2 Announce Type: replace  Abstract: Organisations deploying AI face a scoping problem: which tasks can be automated, which should remain with humans, and which are best shared between the two. Aggregate benchmark scores provide little insight into where systems will succeed or fail in practice, while human judgements of model capabilities quickly become outdated. We introduce a pipeline that profiles agents and tasks using a shared set of core cognitive capabilities. Cognitive capability profiling infers an agent's capabilities from performance on a benchmark battery annotated for the cognitive demands of each item. Task requirements weighting elicits from domain experts the relative importance of these same capabilities for their work. As both use a common set of cognitive dimensions, they can be updated independently as models and roles change, and combined to estimate AI suitability at the level of a domain, organisation, role, or individual duty. We validate capabi
    
[^266]: 可信RAG：用于检测生成式AI系统中错误信息与知识投毒的评估代理

    Trustworthy RAG: An Evaluation Agent for Detecting Misinformation and Knowledge Poisoning in Generative AI Systems

    [https://arxiv.org/abs/2608.21095](https://arxiv.org/abs/2608.21095)

    本文提出了一种结合NLI事实核查与五信号投毒检测的评估代理，并引入信任指数，在TruthfulQA上实现了高准确率与精确率，有效缓解了RAG系统中的知识投毒风险。

    

    检索增强生成（RAG）将大型语言模型（LLM）的输出锚定在外部知识上，但RAG系统通常信任其检索到的任何内容，从而造成“安全-可靠性差距”：高语义相关性并不保证事实真实性。攻击者利用这一漏洞进行知识投毒，插入恶意文档以引发定向错误信息。我们提出了一种评估代理，这是一种中间件，结合了自然语言推理（NLI）事实核查、具有相关性加权聚合的五信号投毒检测器，以及一个信任指数 T = 0.4 F + 0.35 C + 0.25 (1 - P )，并针对高污染情境采用非线性阻尼器。在TruthfulQA上使用Llama 3.3 70B时，该代理达到了91%的准确率和100%的精确率，对指令注入的召回率为100%，而就地编辑（如实体替换）仍难以检测。在三个LLM上，信任指数保持判别性，接收者操作特征（ROC）曲线表现良好。

    arXiv:2608.21095v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation (RAG) grounds Large Language Model (LLM) outputs in external knowledge, but RAG systems usually trust whatever they retrieve, creating a Security-Reliability Gap: high semantic relevance does not guarantee factual truth. Adversaries exploit this through knowledge poisoning, inserting malicious documents to cause targeted misinformation. We propose an Evaluation Agent, middleware that combines Natural Language Inference (NLI) factual verification, a five-signal poison detector with relevance-weighted aggregation, and a Trust Index T = 0.4 F + 0.35 C + 0.25 (1 - P ) with a non-linear dampener for high-contamination contexts. On TruthfulQA with Llama 3.3 70B, the agent reaches 91% accuracy and 100% precision, with 100% recall on instruction injection, while in-place edits, such as entity swaps, remain hard to detect. Across three LLMs the Trust Index stays discriminative, with a Receiver Operating Characteri
    
[^267]: Agent Gym：一种通过人在环反馈实现LLM智能体持续评估与演化的框架

    Agent Gym: A Framework for Continuous Evaluation and Evolution of LLM Agents Through Human-in-the-Loop Feedback

    [https://arxiv.org/abs/2608.15591](https://arxiv.org/abs/2608.15591)

    Agent Gym提出了一种模块化、领域无关的框架，通过人在环反馈实现LLM智能体部署后的持续评估与演化，无需修改源代码即可进行行为修正。

    

    大规模语言模型（LLM）智能体在生产环境中的部署面临一个根本性矛盾：智能体在部署时行为被冻结，而它必须处理的业务规则和边缘情况却在持续变化。现有方法侧重于智能体的构建和一次性评估，但缺乏一种结构化的机制，能在不修改智能体源代码的情况下，对部署后的行为进行持续修正。市场上大多数方法需要大量收集日志和追踪信息，并由工程团队重新审查智能体设计，这一过程繁琐且耗时，削弱了智能体转型的经济价值。我们提出了Agent Gym，一个模块化、领域无关的框架，可将任何现有的基于LLM的智能体封装在一个持续的评估与演化循环中。该框架提供六种可组合的能力——执行（Act）、评估（Evaluate）、调查（Investigate）、纠正（Correct）、学习（Learn）和观察（Observe）——并组织在多个层面。

    arXiv:2608.15591v1 Announce Type: new  Abstract: Large Language Model (LLM) agents deployed in production environments face a fundamental tension: the agent's behavior is frozen at deployment time, while the business rules and edge cases it must handle continue to evolve. Existing approaches address agent construction and one-time evaluation but provide no structured mechanism for continuous post-deployment behavioral correction without modifying the agent's source code. Most of the approaches offered in the market, require intense collection of logs and traces, and re-examining the agent design by the engineering team, a process which is heavy, long and negates the economical value of agentic transformation. We introduce Agent Gym, a modular, domain-agnostic framework that wraps any existing LLM-based agent in a continuous evaluation-and-evolution loop. The framework provides six composable capabilities --- Act, Evaluate, Investigate, Correct, Learn, and Observe --- organized across t
    
[^268]: 通过I/O感知重构实现快速且内存高效的小波卷积

    Fast and Memory-Efficient Wavelet Convolutions via I/O-Aware Reformulation

    [https://arxiv.org/abs/2608.10805](https://arxiv.org/abs/2608.10805)

    本文通过I/O感知的代数重构（包括片上重计算、级联折叠和权重缩放融合），显著减少了小波卷积中的高带宽内存流量，实现了快速且内存高效的WTConv实现。

    

    arXiv:2608.10805v2 公告类型：替换交叉 摘要：小波卷积（WTConv）已成为标准卷积日益流行的替代方案，通过分解级别的数量指数级扩展网络的感受野，同时保持参数数量线性增长。然而，其参考实现由于通过高带宽内存（HBM）的过度数据移动而严重受内存限制。我们开发了一个WTConv的I/O模型来表征这一瓶颈，并用它指导三种代数重构：（1）在芯片上重新计算廉价的Haar分析蝶形运算，（2）将多级合成级联折叠为单个由输出坐标位索引的闭式传递，（3）将学习到的逐通道缩放因子折叠到卷积权重中。这些重构共同实现了一个I/O感知的融合实现，大幅减少了HBM流量。我们评估了不同分解级别下的WTConvNeXt配置，以及更广泛的设置。

    arXiv:2608.10805v2 Announce Type: replace-cross  Abstract: Wavelet convolution (WTConv) has emerged as an increasingly popular drop-in replacement for standard convolutions, expanding a network's receptive field exponentially with the number of decomposition levels while keeping the parameter count linear. However, its reference implementation is severely memory-bound due to excessive data movement through high-bandwidth memory (HBM). We develop an I/O model of WTConv to characterize this bottleneck and use it to guide three algebraic reformulations: (1) recomputing the inexpensive Haar analysis butterfly on chip, (2) collapsing the multi-level synthesis cascade into a single closed-form pass indexed by output-coordinate bits, and (3) folding learned per-channel scales into the convolution weights. Together, these reformulations enable an I/O-aware fused implementation that substantially reduces HBM traffic. We evaluate the WTConvNeXt configuration across decomposition levels and a bro
    
[^269]: MD-ProTector：面向LLM生成文本检测的多数据驱动原型定位

    MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection

    [https://arxiv.org/abs/2608.10459](https://arxiv.org/abs/2608.10459)

    MD-ProTector通过在编码器嵌入空间中为每个类别设置多个可训练原型并采用原型定位损失，为LLM生成文本检测提供了对类内多样变化的细粒度建模能力。

    

    随着大语言模型（LLM）生成的内容日益复杂，用于区分LLM生成文本与人类撰写文本的检测系统必须能够大规模运行，同时应对多样化的写作风格、领域、语言和生成器模型。仅基于输入的编码器检测器适合实际部署场景，但标准的二分类仅提供类别标签，无法显式地组织两个类别内部存在的显著变化。我们提出MD-ProTector，它在编码器嵌入空间中使用多个可训练的参考向量（称为原型）来表示每个类别。这些原型为同一类别内的不同文本组提供独立的决策边界。然而，仅添加多个原型并不能确定每个原型应代表何种变化。MD-ProTector通过原型定位损失来解决这一问题，该损失将类别级结构与类内变化分离开（原文此处被截断）。

    arXiv:2608.10459v2 Announce Type: replace-cross  Abstract: As LLM-generated content becomes more sophisticated, detection systems for distinguishing those texts from human-written text must operate at scale while handling diverse writing styles, domains, languages, and generator models. Input-only encoder detectors are suitable for practical deployment setting, but standard binary classification supplies only the class label and does not explicitly organize the substantial variation within either class. We propose MD-ProTector, which represents each class with multiple trainable reference vectors in the encoder embedding space, referred to as prototypes. These prototypes provide separate decision boundaries for different groups of texts within the same class. However, adding multiple prototypes alone does not determine which variation each prototype should represent. MD-ProTector addresses this problem with Prototype Positioning loss, which separates class-level structure from the with
    
[^270]: TraceCAD：面向智能体CAD生成的轨迹引导修复方法

    TraceCAD: Trace-Guided Repair for Agentic CAD Generation

    [https://arxiv.org/abs/2608.03062](https://arxiv.org/abs/2608.03062)

    TraceCAD通过将需求、建模步骤、失败证据与修复结果链接为持久轨迹状态并配合可复用的技能记忆，实现对LLM生成的CAD程序的高效诊断与局部化修复，显著减少几何退化并大幅降低智能体调用开销。

    

    基于大语言模型（LLM）的CAD智能体能够生成可执行的参数化程序，但其纠错循环可能会丢失关于已满足的需求、错误操作以及先前修复的证据。我们提出了TraceCAD，一个将请求的特征、建模步骤、失败证据和候选修复结果链接为持久状态的恢复层。TraceCAD诊断可能出错的建模操作，在其依赖区域内搜索有界编辑方案，通过执行验证和保留性检查来评估候选方案，并将成功与失败的修复结果保存到可复用的技能记忆中。在基于DeepCAD的基准测试上，通过200个模型的消融实验和1000个模型的对比，TraceCAD在IoU、Chamfer距离和Hausdorff距离等指标上取得了具有竞争力的几何质量。消融实验表明，移除持久状态会使恢复分数几乎减半；移除局部化搜索会使几何退化增加一倍以上，并使代码智能体的调用次数翻倍。在不相交的数据上初始化技能库……

    arXiv:2608.03062v2 Announce Type: replace  Abstract: LLM-based CAD agents produce executable parametric programs, but their correction loops may lose evidence about satisfied requirements, faulty operations, and prior repairs. We introduce TraceCAD, a recovery layer that links requested features, modeling steps, failure evidence, and candidate outcomes as persistent state. TraceCAD diagnoses likely faulty operations, searches bounded edits in their dependency regions, validates candidates through execution and preservation checks, and retains successful and failed repair outcomes in reusable skill memory. On DeepCAD-derived benchmarks with 200-model ablations and a 1K-model comparison, TraceCAD achieves competitive geometric quality in terms of IoU, Chamfer distance, and Hausdorff distance. Removing persistent state nearly halves recovery score; removing localized search more than doubles geometric regression and doubles code-agent invocations. Initializing the skill store on disjoint 
    
[^271]: 安全，还是仅仅能力？智能体安全基准的效度审计

    Safety, or Just Capability? A Validity Audit of Agent-Safety Benchmarks

    [https://arxiv.org/abs/2607.28685](https://arxiv.org/abs/2607.28685)

    本文对四个主流智能体安全基准进行效度审计，发现其评估指标存在系统性缺陷（如恒判为正的策略即可获得高分）、不同基准对同一批模型的排名互相矛盾，且安全基准之间或安全与能力之间表观的负相关很可能只是小样本导致的统计伪象。

    

    智能体安全基准各自测量的是不同的行为，但它们的分数却经常被交替引用为智能体的“安全性”。我们将其中四个基准视为待验证的测量工具，在多达22个模型上使用各基准的官方实现和作者提供的评分器进行运行，并在统一协议下自行测量MMLU和GPQA作为能力综合指标。首要问题在于评估指标本身：在任何以 $F_1$ 评分的二值轨迹判断基准上，“恒判为正”的策略都能达到 $F_1 = 2\pi/(1+\pi)$；在R-Judge上该值为0.690，高于21个实际具有区分能力的模型中的五个。进一步地，三个覆盖面较广的基准对同一批18个模型给出了不同的排名，而这种不一致背后的权衡其实是小样本面板造成的伪象：R-Judge特异性与AgentHarm安全性在 $n=7$ 时相关系数为 $-0.64$，而在 $n=18$ 时仅为 $+0.02$，且约四分之一的随机规模为7的子集在该值附近达到 $|\rho| \geq 0.5$。

    arXiv:2607.28685v2 Announce Type: replace  Abstract: Agent-safety benchmarks measure different behaviors, and their scores get quoted interchangeably as an agent's safety. We treat four of them (R-Judge, InjecAgent, AgentHarm, AgentDojo) as measurements to be validated, running each under its official implementation and author-provided scorer on up to 22 models, with MMLU and GPQA measured by us under one protocol as a capability composite. The metric is the first problem. On any binary trace-judgment benchmark scored by $F_1$, an ``always positive'' policy attains $F_1 = 2\pi/(1+\pi)$; on R-Judge that is $0.690$, above five of the 21 models that actually discriminate. The three broad-coverage benchmarks then rank the same 18 models differently, and the trade-off behind that disagreement is a small-panel artifact: R-Judge specificity against AgentHarm safety correlates $-0.64$ at $n{=}7$ and $+0.02$ at $n{=}18$, and a quarter of random size-7 subsets reach $|\rho| \geq 0.5$ around that
    
[^272]: 从表征到行为：探索大语言模型中的人-情境-行为三元关系

    From Representations to Behaviors: Exploring the Person-Situation-Behavior Triad in LLMs

    [https://arxiv.org/abs/2607.26853](https://arxiv.org/abs/2607.26853)

    该研究将Funder人格三元框架适配到大语言模型中，提出了一个发现、控制和验证模型内部人格相关表征的框架，以机制性视角揭示这些表征如何跨情境地塑造社会任务中的行为表现。

    

    人类人格理论认为，人格特质并非由单一分数捕捉的孤立属性，而是通过人、情境与行为之间的相互作用所表达的稳定个体倾向。现有关于大语言模型中人格相关行为的研究主要集中于人格条件引导下的输出，仅刻画了可观察的特质相关表达，缺乏机制性证据来证明内部人格相关表征的存在、这些表征的跨情境表达方式，以及它们如何塑造特定行为。基于Funder的人格三元框架，我们将其三个组成部分适配用于大语言模型分析：人（Person）作为人格相关的内部表征，情境作为能够引发特质相关响应的语境，行为作为更广泛社会任务上的响应模式。我们引入了一个用于发现、控制和验证特质类表征的框架……

    arXiv:2607.26853v2 Announce Type: replace-cross  Abstract: Human personality theories characterize traits not as isolated attributes captured by a single score, but as stable individual tendencies expressed through the interplay among persons, situations, and behaviors. Existing studies of personality-related behavior in LLMs have primarily focused on outputs elicited under personality conditioning, characterizing observable trait-related expressions while lacking mechanistic evidence for the existence of internal personality-related representations, their cross-situational expression, and how these representations shape specific behaviors. Building on Funder's personality triad framework, we adapt its three components for LLM analysis: Person as personality-related internal representations, Situation as contexts that afford trait-relevant responses, and Behavior as response patterns on broader social tasks. We introduce a framework for discovering, controlling, and validating trait-li
    
[^273]: 具有影响力的多元价值对齐研究路线图

    A Roadmap to Impactful Pluralistic Alignment Research

    [https://arxiv.org/abs/2607.22305](https://arxiv.org/abs/2607.22305)

    本文通过审计主要前沿AI实验室发现多元价值对齐研究尚未真正影响模型的训练与评估，并针对论证缺乏实证、实践标准不明等三大障碍提出了推动其落地的研究路线图。

    

    许多负责任人工智能研究旨在改变数十亿人使用的AI系统的行为。多元价值对齐——即构建能够代表并服务于多元人类价值观与观点的AI系统——已成为其中一个活跃的研究议程。然而，在对主要前沿实验室和开放权重模型开发者的审计中，我们没有发现任何公开证据表明多元价值对齐已影响其模型的训练或评估方式。多元价值对齐需要被实际采纳才能实现其主要目标，因此我们识别出三大障碍，并针对每个障碍提出了相应的研究路线图：1. 目前多元价值对齐的论证大多是规范性或理论性的，需要开展实证研究来衡量多元AI对用户和社会的益处与弊端。2. 多元价值对齐研究界尚未就何时需要多元行为、以及多元行为在实践中应呈现何种形态达成共识，需要相关研究来阐明……

    arXiv:2607.22305v2 Announce Type: replace  Abstract: Much research on responsible AI aims to change the behavior of AI systems that billions of people use. Pluralistic value alignment---the goal of building AI systems that represent and serve diverse human values and perspectives---has emerged as one such active research agenda. Yet, in an audit of the major frontier labs and open-weight developers, we find no public evidence that it has shaped how their models are trained or evaluated. Pluralistic alignment requires adoption to achieve its main goals, so we identify three barriers and lay out a roadmap of research to address each: 1. The justifications for pluralistic alignment so far are mostly normative or theoretical. Studies are needed that measure the benefits and drawbacks of pluralistic AI for users and society. 2. The pluralistic alignment research community has not settled when pluralistic behavior is warranted or what it should look like in practice. Research is needed to sp
    
[^274]: OTAP：用于评估智能体轨迹中规划与执行的结构感知最优传输方法

    OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories

    [https://arxiv.org/abs/2607.17082](https://arxiv.org/abs/2607.17082)

    该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。

    

    大型语言模型智能体通过生成交错包含规划、工具调用和中间结果的轨迹来完成任务。当前的评估指标要么将此类轨迹简化为一个二元成功标志，要么通过精确匹配将其与参考轨迹进行比较，要么将判断委托给另一个语言模型。成功标志无法区分合理的解决方案与侥幸成功的方案，也无法解释运行失败的原因；而精确匹配则会对被重新排序或以不同方式分解的有效规划进行惩罚。我们将轨迹评估重新框架化为智能体执行图与一组有效解决方案图之间的距离问题，并通过在带属性依赖图上求解不平衡融合 Gromov-Wasserstein 传输问题来加以实现。所得分数称为 OTAP（Optimal Transport for Agentic Planning，智能体规划最优传输），是一个归一化的差异分数。其理想化的、经全局最小化的无熵目标对于固定图中保持依赖关系的重标记具有不变性……

    arXiv:2607.17082v3 Announce Type: replace  Abstract: Large language model agents solve tasks by generating trajectories that interleave planning, tool calls, and intermediate results. Current evaluation metrics reduce such a trajectory to a binary success flag, compare it against a reference by exact matching, or delegate judgment to another language model. A success flag cannot distinguish sound solutions from lucky ones, nor explain why runs fail. Exact matching penalizes valid plans reordered or decomposed differently. We reframe trajectory evaluation as a distance between the agent's execution graph and a set of valid solution graphs, and instantiate it via an unbalanced fused Gromov-Wasserstein transport problem over attributed dependency graphs. The resulting score, termed OTAP (Optimal Transport for Agentic Planning), is a normalized discrepancy score. Its idealized, globally minimized entropy-free objective is invariant to dependency-preserving relabelings of a fixed graph and 
    
[^275]: 理解从预训练到后训练的推理

    Understanding Reasoning from Pretraining to Post-Training

    [https://arxiv.org/abs/2607.16097](https://arxiv.org/abs/2607.16097)

    该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。

    

    强化学习（RL）已成为提升大型语言模型（LLM）在复杂推理任务上表现的核心手段，然而RL后训练通常是与之前的预训练割裂开来研究的。因此，两个基本问题仍然悬而未决：(1) 预训练选择（模型规模、数据）如何影响RL计算投入的回报？(2) RL究竟对模型做了什么？在标准的LLM研究设定下，这些问题难以探究：预训练语料庞大且不可控，难以将模型行为归因于预训练还是RL，而且跨两个阶段进行系统性的计算扫描成本高昂到令人却步。为应对这些挑战，我们采用国际象棋作为受控测试平台，在完整的预训练到后训练流程中研究推理能力。我们遵循标准的LLM训练流程，在人类棋局上预训练从5M到1B参数的语言模型，并在合成……（摘要在此处截断，内容不完整）

    arXiv:2607.16097v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become central to improving large language models (LLMs) on complex reasoning tasks, yet RL post-training is largely studied in isolation from the pretraining that precedes it. As a result, two basic questions remain open: (1) how do pretraining choices (model size, data) shape the returns to RL compute, and (2) what does RL actually do to the model? These questions are difficult to study in the standard LLM setting: pretraining corpora are vast and uncontrolled, making it hard to attribute behaviors to pretraining versus RL, and systematic compute sweeps across both stages are prohibitively expensive. To address these challenges, we use chess as a controlled testbed for studying reasoning across the full pretraining-to-post-training pipeline. We follow the standard LLM training pipeline by pretraining language models from 5M to 1B parameters on human chess games, supervised fine-tuning on synthe
    
[^276]: 基于表格基础模型的数据驱动动态安全评估再审视

    Revisiting data-driven dynamic security assessment with a tabular foundation model

    [https://arxiv.org/abs/2607.16031](https://arxiv.org/abs/2607.16031)

    该论文提出使用表格基础模型（TFM）通过上下文学习进行电力系统动态安全评估，仅需单一模型即可同时评估多个偶然事件，无需逐个偶然事件训练分类器或进行超参数优化，并可泛化到未见过的偶然事件。

    

    数据驱动的故障前动态安全评估（DSA）利用机器学习（ML）快速评估电力系统中可信偶然事件的动态风险。现有方法面临两个局限性：首先，它们需要大量的标注数据库进行训练，且对于可能很长的可信偶然事件列表中的每一个偶然事件，都需要单独训练、调优和维护一个模型；其次，训练好的模型对未见过的偶然事件泛化能力较差。本工作通过使用一种表格基础模型（TFM）来解决这些局限性，该模型通过上下文学习来评估稳定性，无需重新训练或超参数优化。我们使用单个TFM同时评估多个偶然事件，从而无需为每个偶然事件训练一个分类器。我们还刻画了使用电气距离坐标（EDC）作为连续特征何时能够使TFM泛化到未见过的偶然事件，以及何时不能。

    arXiv:2607.16031v2 Announce Type: replace-cross  Abstract: Data-driven pre-fault dynamic security assessment (DSA) rapidly evaluates the dynamic risk of credible contingencies on a power system using machine learning (ML). Existing approaches face two limitations. First, they require a large labelled database for training, with a separate model trained, tuned, and maintained for each contingency in a potentially long list of credible contingencies. Second, the trained models generalize poorly to unseen contingencies. This work addresses these limitations by using a tabular foundation model (TFM) that assesses stability through in-context learning, and requires no retraining or hyperparameter optimization. We use a single TFM to assess many contingencies at once, removing the need to train one classifier per contingency. We also characterize when using electrical distance coordinates (EDC) as continuous features enables the TFM to generalize to unseen contingencies and when it does not,
    
[^277]: 用于语言建模的词元时间连续扩散模型

    Token Time Continuous Diffusion for Language Modeling

    [https://arxiv.org/abs/2607.14106](https://arxiv.org/abs/2607.14106)

    本文提出词元时间连续扩散（TTCD），一种在连续空间中运行、确定性映射噪声到词元并引入逐词元时间概念的新型扩散语言模型，在高加速比下性能优于离散扩散模型。

    

    本文介绍了词元时间连续扩散（TTCD），这是一种新的扩散语言模型，它（a）在连续空间中运行，确定性地将高斯噪声映射到最终的词元画布上，无需进一步的采样；且至关重要的是（b）引入了一种新的“逐词元时间”（per-token times）概念，使某些词元能够以比其他词元更快的速度从噪声转变为词元。连续空间建模帮助TTCD避免了对多个词元进行并行采样，而并行采样正是纯离散空间迭代模型在高加速比下产生不准确性的关键原因。逐词元时间的概念帮助TTCD更好地建模条件生成，允许更确定的词元以更快的速度推进，并允许在精炼过程中对词元之间的影响进行差异化处理。TTCD在高加速比下优于离散模型。我们在OpenWebText数据集上训练了一个1.6亿参数的TTCD模型，并随后对其进行了自蒸馏；我们发现在高加速比下我们具有可比的性能（原文在此处截断）……

    arXiv:2607.14106v2 Announce Type: replace-cross  Abstract: In this paper we introduce token time continuous diffusion (TTCD), a new diffusion language model which (a) operates in continuous space, deterministically mapping Gaussian noise to a final token canvas with no further sampling, and crucially (b) incorporates a new notion of per-token times, with some tokens proceeding from noise to token at a faster rate than others. Continuous space modeling helps TTCD avoid the parallel sampling of multiple tokens, which is a key source of inaccuracy at high speedups for models that iterate purely in discrete space. The notion of per-token times helps TTCD to better model conditional generation, allows for more sure tokens to proceed at a faster rate, and allows for differentiated inter-token influences during refinement. TTCD outperforms discrete models at high speedups. We train a 160M parameter TTCD model on OpenWebText, and then self-distill it; we find that at high speedups we are compa
    
[^278]: KV-PRM：通过KV缓存传递实现多智能体测试时扩展的高效过程奖励建模

    KV-PRM: Efficient Process Reward Modeling via KV-Cache Transfer for Multi-Agent Test-Time Scaling

    [https://arxiv.org/abs/2607.09153](https://arxiv.org/abs/2607.09153)

    KV-PRM通过直接读取大语言模型生成阶段产生的KV缓存来替代从头重新编码轨迹文本，将过程奖励模型的评分成本从O(L²)降至O(L)，实现了多智能体测试时扩展中的高效奖励评分。

    

    过程奖励模型已被证明在指导测试时扩展（TTS）方法方面非常有效，可显著提升基于大语言模型的多智能体系统的能力。然而，现有的过程奖励模型是基于文本的：它们需要从头开始对整个轨迹文本进行重新编码。在较长的多智能体轨迹展开中，评分成本随序列长度L呈二次方增长，形成了严重的计算瓶颈，极大限制了过程奖励模型在长上下文场景中的应用。为解决这一问题，我们提出了KV-PRM，一种高效的过程奖励模型，它通过直接读取大语言模型在生成阶段自然产生的KV缓存，消除了繁重的文本重新编码过程。通过针对已有的KV缓存处理单个“验证token”，KV-PRM将评分成本从O(L²)降低到O(L)。我们正式证明KV缓存包含严格大于文本的信息容量，并且更加高效……

    arXiv:2607.09153v2 Announce Type: replace  Abstract: Process Reward Models (PRMs) have been proven to be highly effective in guiding test-time scaling (TTS) methods, which significantly boost the capabilities of LLM-based multi-agent systems. However, existing PRMs are text-based: they re-encode the entire trajectory text from scratch. In long multi-agent rollouts, the scoring cost, growing quadratically with respect to sequence length L, creates a severe computational bottleneck, severely limiting PRMs' application in long-context scenarios. To resolve this, we introduce KV-PRM, a highly efficient process reward model that eliminates the heavy text re-encoding by directly reading the KV cache produced naturally during the LLM's generation phase. By processing a single "verify token" against the pre-existing KV cache, KV-PRM reduces the scoring cost from O(L^2) to O(L). We formally prove that the KV cache contains strictly greater information capacity than text, and is more efficient f
    
[^279]: 从求解器到研究：大型语言模型驱动的研究前沿形式化数学

    From Solvers to Research: Large Language Model-Driven Formal Mathematics at the Research Frontier

    [https://arxiv.org/abs/2607.07779](https://arxiv.org/abs/2607.07779)

    本综述系统梳理了LLM驱动的形式化数学进展（涵盖训练策略、推理时搜索与智能体工作流三大方向），并强调从求解明确定义的问题迈向真正的数学研究，关键在于形式化证明之外还需捕捉问题的本意并融入专家判断。

    

    大型语言模型在为明确定义的数学问题生成机器可检验的证明方面取得了快速进展。然而，将这一进展扩展到研究层面的数学，需要的不仅仅是解决孤立的定理。研究工作涉及选择问题、发展定义与猜想、连接不同结果，以及将形式化证明与专家判断相结合。本综述考察了LLM驱动的系统如何支持这些更广泛的活动。我们回顾了形式化数学的任务、数据集和自动形式化，并将证明生成方法归纳为三个相互重叠的领域：训练策略、推理时搜索与适应、以及智能体工作流。随后，我们考察了基准测试结果、若干代表性研究项目以及支持协作数学研究的平台。贯穿全文的一个核心区分是：通过形式化证明的机器检验本身，并不能证明该证明真正捕捉了所要研究的数学问题，也无法等同于做出了新的数学贡献。

    arXiv:2607.07779v4 Announce Type: replace-cross  Abstract: Large language models have made rapid progress in generating machine-checked proofs for well-defined mathematical problems. Extending this progress to research mathematics requires more than solving isolated theorems. Research involves choosing questions, developing definitions and conjectures, connecting results, and combining formal proofs with expert judgment. This survey examines how LLM-driven systems can support these broader activities. We review formal-mathematics tasks, datasets, and autoformalization, and organize proof-generation methods into three overlapping areas: training strategies, inference-time search and adaptation, and agentic workflows. We then examine benchmark results, selected research projects, and platforms for collaborative mathematical work. A central distinction runs throughout the survey: checking a formal proof does not by itself establish that it captures the intended problem or makes a new math
    
[^280]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^281]: 图神经网络中的超参数迁移

    Hyperparameter Transfer in Graph Neural Networks

    [https://arxiv.org/abs/2607.05017](https://arxiv.org/abs/2607.05017)

    本文为使用SGD、Adam和AdamW训练的图神经网络提出并验证了一种超参数迁移参数化方法，使模型在宽度和深度增加时能够实现稳定的特征更新、学习率迁移及性能提升。

    

    深度学习模型的性能在很大程度上取决于学习率、初始化尺度和权重衰减等超参数的设置。超参数迁移旨在使接近最优的超参数设置在不同模型规模之间保持一致，从而可以通过代理调优较小、优化成本更低的对等模型来优化大模型。虽然迁移原则在语言和视觉任务中的密集神经网络背景下已被充分研究，但对于图神经网络（GNN）而言，这一领域仍相对缺乏探索。我们为使用SGD、Adam和AdamW训练的GNN开发并验证了一种迁移参数化方法。通过理论缩放分析和受控实验，我们表明所提出的参数化方法能够产生稳定的特征更新、学习率迁移，并随着宽度和深度的增加而提升性能。对于SGD，我们识别出依赖于图的第一层校正因子，并显示…

    arXiv:2607.05017v2 Announce Type: replace-cross  Abstract: The performance of deep learning models crucially depends on the settings of hyperparameters like learning rate, initialization scale, and weight decay. Hyperparameter transfer aims to make near-optimal hyperparameter settings consistent across model scale, so that large models can be optimized by proxy tuning their smaller, cheaper-to-optimize counterparts. While transfer principles are well-studied in the context of dense neural networks in language and vision tasks, they remain comparatively under-explored for graph neural networks (GNNs). We develop and validate a transfer parameterization for GNNs trained with SGD, Adam, and AdamW. Through theoretical scaling analyses and controlled experiments, we show that the proposed parameterization yields stable feature updates, learning rate transfer, and improved performance as width and depth increase. For SGD, we identify graph-dependent first-layer correction factors and show th
    
[^282]: PixCon：面向基础模型半监督分割的干净正样本对比学习

    PixCon: Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation

    [https://arxiv.org/abs/2607.03068](https://arxiv.org/abs/2607.03068)

    PixCon 通过添加一个仅存储预测与真实标签一致处的干净有标签特征、无需置信度阈值的类别级记忆库，并以多正样本对比损失训练编码器-解码器，从而以极简方式显著提升了基于基础模型的半监督分割性能。

    

    arXiv:2607.03068v2 公告类型： replace-cross 摘要：一个简单的有标签像素对比分支能否在使用强大预训练编码器的半监督分割中带来提升？我们通过 PixCon 研究了这一问题。PixCon 在基于 DINOv2-Base 的 UniMatch V2 实现上添加了一个按类别划分的记忆库，该记忆库仅存储学生预测与真实标签一致位置处的有标签特征。多正样本对比损失利用这些特征来训练共享的编码器-解码器，而在推理阶段头部和记忆库会被丢弃。这通过去除置信度阈值和学习的排序机制，简化了先前的有标签像素记忆库方法。在仅使用 183 个标签的 Pascal VOC 上，PixCon 使三个配对随机种子的 mIoU 分别提升 0.20、0.64 和 1.83 个点，平均值从 87.01 提高到 87.90。使用 92 个标签时平均提升 0.40 个点；在 Cityscapes 和 ADE20K 上的单种子差异范围为 -0.08 至 +0.14 个点。一个使用预测标签进行对比学习的对照实验达到了几乎相同的 Pascal 精度，因此这些结果……

    arXiv:2607.03068v2 Announce Type: replace-cross  Abstract: Can a simple labeled-pixel contrastive branch improve semi-supervised segmentation with a strong pretrained encoder? We study this question with PixCon, which adds a class-wise memory bank to a DINOv2-Base implementation of UniMatch V2. The bank stores ground-truth-labeled features only where the student prediction agrees with the label. A multi-positive contrastive loss uses these features to train the shared encoder-decoder; the head and bank are discarded at inference. This simplifies earlier labeled-pixel banks by removing confidence thresholds and learned ranking. On Pascal VOC with 183 labels, PixCon improves all three paired seeds by 0.20, 0.64 and 1.83 mIoU points, raising the mean from 87.01 to 87.90. With 92 labels the mean gain is 0.40 points; single-seed differences on Cityscapes and ADE20K range from -0.08 to +0.14 points. A predicted-label contrastive control reaches nearly the same Pascal accuracy, so these resul
    
[^283]: 超越提示词：通过模拟审核轨迹越狱函数调用大语言模型

    Beyond the Prompt: Jailbreaking Function-Calling LLMs via Simulated Moderation Traces

    [https://arxiv.org/abs/2607.00481](https://arxiv.org/abs/2607.00481)

    本文提出SMT黑盒攻击框架，通过构建模拟合法审核审计工作流的多轮轨迹，利用函数调用环境中可信控制逻辑与不可信数据边界模糊的结构性漏洞，成功越狱函数调用大语言模型。

    

    越狱攻击仍然是大语言模型（LLM）安全部署的重大威胁。虽然先前的工作主要在提示词层面研究攻击与防御，但我们表明这种以提示词为中心的范式忽略了有状态的函数调用环境中的一个结构性漏洞。在这类应用中，开发者定义的模式、结构化参数和不可信的工具输出被交错混合到单一的共享模型上下文中。这种架构通过模糊可信控制逻辑与不可信数据之间的边界来扩大攻击面，使对抗意图能够分布在多轮执行路径中。我们通过SMT——一种基于模拟审核轨迹的黑盒攻击框架——利用这一架构缺陷。SMT摒弃了纯提示词交互方式，构建了一个模拟合法审核审计工作流的多轮轨迹。在该轨迹中，一个……

    arXiv:2607.00481v2 Announce Type: replace-cross  Abstract: Jailbreak attacks remain a critical threat to the safe deployment of large language models (LLMs). While prior work has primarily studied attacks and defenses at the prompt level, we show that this prompt-centric paradigm overlooks a structural vulnerability in stateful, function-calling environments. In such applications, developer-defined schemas, structured arguments, and untrusted tool outputs are interleaved into a single shared model context. This architecture expands the attack surface by blurring the boundary between trusted control logic and untrusted data, allowing adversarial intent to be distributed across a multi-turn execution path. We exploit this architectural flaw through SMT, a black-box attack framework based on Simulated Moderation Traces. Departing from purely prompt-based interactions, SMT constructs a multi-turn trajectory that simulates a legitimate moderation-auditing workflow. Within this trajectory, a
    
[^284]: EvoOptiGraph：基于图结构生成的弱点驱动协同进化优化建模方法

    EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling

    [https://arxiv.org/abs/2606.26578](https://arxiv.org/abs/2606.26578)

    提出EvoOptiGraph框架，通过将优化模型表示为图结构并利用弱点驱动的协同进化，自动生成结构多样的训练数据，从而提升大语言模型在优化建模任务上的性能。

    

    利用大语言模型从自然语言自动生成优化模型面临两个关键挑战。首先，训练语料库缺乏结构多样性。其次，数据生成流程保持静态，与模型学习相脱节。为解决这些挑战，我们提出了EvoOptiGraph，这是一种新颖的框架，其中数据和模型以模型弱点为驱动力协同进化。EvoOptiGraph将每个混合整数线性规划表示为属性二分图，并应用保持有效性的进化算子来生成结构多样的实例。通过确定性编译和验证性反向翻译，进化后的图被转换为求解器代码和自然语言。训练分为两个阶段：在初始数据集上进行监督微调，随后进行带可验证奖励的强化学习，其中基于图的弱点信号指导生成针对模型当前薄弱点的新实例。

    arXiv:2606.26578v1 Announce Type: new  Abstract: Automating optimization modeling from natural language with large language models (LLMs) faces two key challenges. First, training corpora lack structural diversity. Second, data generation pipelines remain static and decoupled from model learning. To address these challenges, we propose EvoOptiGraph, a novel framework where data and model co-evolve, driven by model weaknesses. EvoOptiGraph represents each mixed-integer linear program (MILP) as an attributed bipartite graph and applies validity-preserving evolutionary operators to generate structurally diverse instances. The evolved graphs are converted into solver code and natural language via deterministic compilation and verified back-translation. Training proceeds in two stages: supervised fine-tuning (SFT) on an initial dataset, followed by reinforcement learning with verifiable rewards (RLVR), where graph-derived weakness signals guide the generation of new instances targeting the 
    
[^285]: 多人不完美信息博弈中纳什均衡计算的变量边界收紧方法

    Variable Bound Tightening for Nash Equilibrium Computation in Multiplayer Imperfect-Information Games

    [https://arxiv.org/abs/2606.25997](https://arxiv.org/abs/2606.25997)

    本文提出通过变量边界收紧技术来加速求解多人不完美信息博弈中纳什均衡的二次约束规划，从而提升非凸求解器的计算效率。

    

    近年来，在大型二人零和不完美信息博弈的纳什均衡近似算法，以及多人策略式博弈的纳什均衡精确计算方面取得了显著进展。尽管反事实遗憾最小化和虚拟对弈方法可以扩展到大型博弈，并在二人零和博弈中具有收敛保证，但它们无法保证在多人博弈中收敛到纳什均衡。最近，有研究提出了一种精确计算多人不完美信息博弈中纳什均衡的方法，该方法求解一个二次约束规划，该规划基于由序列形式博弈表示导出的非线性互补问题公式化。该公式化使用Gurobi的非凸二次求解器求解，该求解器采用空间分支定界技术，通过求解双线性项的凸松弛来迭代地收紧变量边界。

    arXiv:2606.25997v3 Announce Type: replace-cross  Abstract: There has been significant recent progress in algorithms for approximation of Nash equilibrium in large two-player zero-sum imperfect-information games and exact computation of Nash equilibrium in multiplayer strategic-form games. While counterfactual regret minimization and fictitious play are scalable to large games and have convergence guarantees in two-player zero-sum games, they do not guarantee convergence to Nash equilibrium in multiplayer games. Recently, an approach has been presented for exact computation of Nash equilibrium in multiplayer imperfect-information games that solves a quadratically constrained program based on a nonlinear complementarity problem formulation derived from the sequence-form game representation. This formulation was solved using Gurobi's nonconvex quadratic solver, which employs spatial branch-and-bound to iteratively refine variable bounds by solving convex relaxations of bilinear terms. Dur
    
[^286]: 通过结构化概念演化的大语言模型发现量子LDPC码

    Large-Language-Model Discovery of Quantum LDPC Codes through Structured Concept Evolution

    [https://arxiv.org/abs/2606.24808](https://arxiv.org/abs/2606.24808)

    该论文提出结构化概念演化方法，让大语言模型通过修改有限群、原型图等构造要素自主演化量子LDPC码的构造程序与搜索策略，在特定参数区间超越了已发表的基准结果，且最佳模型在两个优化目标上均优于随机变异和爬山法。

    

    量子低密度奇偶校验（LDPC）码有望实现低开销的量子纠错，但高性能的构造方法仍然难以找到。本文提出了结构化概念演化方法，其中大语言模型（LLM）通过修改有限群、原型图形状或元素分配规则，来演化构造程序及其内部搜索策略。我们搜索数据量子比特少于600且校验权重不超过12的码，优化目标为 kd²/n 或电路级逻辑错误率。我们的方法在特定参数区间内超越了已发表的提升乘积（lifted-product）基准结果，且领先结果均具有精确认证的码距离。电路级优化通常比 kd²/n 优化产生更低的逻辑错误率，即便其码距离往往更小。在相同的演化框架和步数预算下，表现最佳的大语言模型在两个优化目标上都超越了所有随机变异和爬山法运行。我们的结果……

    arXiv:2606.24808v2 Announce Type: replace-cross  Abstract: Quantum low-density parity-check codes promise low-overhead error correction, but high-performing constructions remain hard to find. Here we introduce structured concept evolution, in which a large language model (LLM) evolves construction programs and their internal search strategies by modifying the finite group, protograph shapes or entry-assignment rule. We search for codes with fewer than $600$ data qubits and check weight at most $12$, optimizing $kd^2/n$ or the circuit-level logical error rate. Our method improves on published lifted-product benchmarks in selected parameter regimes, with exactly certified distances for the leading results. Circuit-level optimization typically yields lower logical error rates than $kd^2/n$ optimization, often despite smaller code distances. With the same evolution framework and step budget, the best LLM outperforms all random-mutation and hill-climbing runs on both objectives. Our results
    
[^287]: 具有人格的患者：通过可控多样性与选择性披露实现真实患者模拟

    Patients With Personality: Realistic Patient Simulation through Controlled Diversity and Selective Disclosure

    [https://arxiv.org/abs/2606.17441](https://arxiv.org/abs/2606.17441)

    提出了基于六维HEXACO人格模型的患者模拟框架PWP，通过显式人格参数化实现对对话风格、配合度和信息披露的细粒度控制，其对话真实性经临床医生评估与真实医患对话相当，且远超以往模拟器。

    

    真实患者模拟是在无需昂贵用户研究的情况下大规模开发和测试临床大语言模型的关键需求。现有模拟器缺乏真实性和可控性：它们会过度分享信息，且无法体现真实患者行为的多样性。本文提出了PatientsWithPersonality（PWP），这是一个患者模拟框架，通过对潜在患者状态进行显式人格参数化，生成真实且多样的患者响应。该框架基于六维HEXACO人格模型，能够对对话风格、配合度和信息披露进行细粒度控制。我们在首个共享协议下对不同患者模拟器进行直接比较的评估中评估了PWP，该评估以真实医患就诊对话记录为基准。临床医生认为PWP对话的真实性至少与真实就诊记录相当，且远高于以往模拟器。

    arXiv:2606.17441v3 Announce Type: replace-cross  Abstract: Realistic patient simulation is a key requirement for developing and testing clinical LLMs at scale without costly user studies. Existing simulators lack realism and controllability: they overshare information and miss the variability of real patient behavior. Here, we introduce PatientsWithPersonality (PWP), a patient simulation framework that generates realistic yet diverse patient responses through explicit personality parametrization over a latent patient state. Grounded in the six-dimensional HEXACO personality model, it gives fine-grained control over conversational style, cooperativeness, and information disclosure. We evaluate PWP in the first direct comparison of different patient simulators under a shared protocol, anchored in transcripts of real physician-patient encounters. Clinicians judge PWP conversations as realistic at least as often as the recorded encounters themselves and far more often than prior simulators
    
[^288]: 幻影与泄露：一个用于审计合成数据隐私的统计框架

    Phantoms and Disclosures: A Statistical Framework for Auditing Privacy in Synthetic Data

    [https://arxiv.org/abs/2606.16952](https://arxiv.org/abs/2606.16952)

    本文提出了一个可定制的实证审计框架，通过区分“真实泄露”与“幻影泄露”，并利用训练/保留集划分与严格的统计假设检验，来审计合成数据是否符合零学习或差分隐私等严格隐私基线。

    

    生成式人工智能和大型语言模型（LLM）的迅速普及，引发了人们将合成数据作为敏感真实世界数据集的隐私保护替代品的兴趣。然而，生成高实用性的合成数据往往伴随着记忆并复述训练语料库中私人信息的风险。在这项工作中，我们提出了一个可定制的实证审计框架，旨在检测并解释此类数据泄露。我们的框架引入了一种机制，用于区分“真实泄露”（即系统直接复制了用户的信息）和“幻影泄露”（即系统偶然生成了用户的数据）。通过将输入数据划分为训练集和保留集，并应用严格的统计假设检验，我们判断观察到的泄露是否符合严格的隐私基线，例如零学习基线或特定的差分隐私（DP）界限。

    arXiv:2606.16952v3 Announce Type: replace-cross  Abstract: The rapid adoption of generative AI and Large Language Models (LLMs) has spurred interest in synthetic data as a privacy-preserving alternative to sensitive real-world datasets. However, generating high-utility synthetic data often carries the risk of memorizing and regurgitating private information from the training corpus. In this work, we present a customizable empirical auditing framework designed to detect and explain such data disclosures. Our framework introduces a mechanism to distinguish between "true disclosures"-where the system directly reproduces a user's information-and "phantom disclosures''-where the system incidentally generates a user's data. By partitioning input data into training and holdout sets and applying rigorous statistical hypothesis testing, we determine if observed disclosures are consistent with strict privacy baselines, such as zero-learning or specific Differential Privacy (DP) bounds. Crucially
    
[^289]: 弥合形态差异：通过意图条件化微调将VLA模型适配到灵巧操作

    Bridging the Morphology Gap: Adapting VLA Models to Dexterous Manipulation via Intent-Conditioned Fine-Tuning

    [https://arxiv.org/abs/2606.12109](https://arxiv.org/abs/2606.12109)

    提出InDex意图条件化微调框架，利用由演示重定向导出的归一化开合度抓取意图信号，将预训练VLA模型从平行夹爪适配到多指灵巧操作，在四个仿真任务中达到85.8%的平均任务成功率。

    

    预训练的视觉-语言-动作（VLA）模型为机器人操作提供了语义和空间先验，但其平行夹爪接口使得多指接触协调处于隐式状态。适配这些模型既需要及时的抓取转换，也需要与本体的手部轨迹相匹配。我们提出InDex，这是一个连接这两项需求的意图条件化框架。从重定向演示中导出的归一化开合度信号用于监督一个标量抓取意图，同时保留完整的手部动作监督。第一阶段学习同步的末端执行器-意图数据块；第二阶段基于这些预测、视觉-语言上下文以及本体感知，对扩散手部解码器进行条件化。通过顺序优化，先建立上游接口，再以低速率协同适配的方式学习灵巧控制。在四个仿真任务中，采用π0.5的InDex实现了85.8%的平均任务成功率，相比原生策略有所提升。

    arXiv:2606.12109v3 Announce Type: replace-cross  Abstract: Pretrained vision-language-action (VLA) models offer semantic and spatial priors for robot manipulation, but their parallel-gripper interfaces leave multi-finger contact coordination implicit. Adapting these models requires both timely grasp transitions and embodiment-specific hand trajectories. We introduce InDex, an intent-conditioned framework that connects these two requirements. A normalized aperture signal, derived from retargeted demonstrations, supervises a scalar grasp intent while preserving full hand-action supervision. Stage 1 learns synchronized end-effector-intent chunks; Stage 2 conditions a diffusion hand decoder on these predictions, visual-language context, and proprioception. Sequential optimization establishes the upstream interface before learning dexterous control with low-rate co-adaptation. Across four simulated tasks, InDex with $\pi_{0.5}$ achieves 85.8% mean task success, improving the native policy b
    
[^290]: 语法约束解码可将大语言模型越狱以生成恶意代码

    Grammar-Constrained Decoding Can Jailbreak LLMs into Generating Malicious Code

    [https://arxiv.org/abs/2606.11817](https://arxiv.org/abs/2606.11817)

    该论文揭示了语法约束解码（GCD）这一可靠性增强技术可被反直觉地利用为越狱攻击面，提出了 CodeSpear 攻击使大语言模型生成恶意代码，并进一步提出 CodeShield 安全对齐方法，使模型在攻击者控制的语法约束下仍能保持安全行为。

    

    大语言模型（LLM）越来越多地被用于代码生成，这引发了人们对其可能被滥用以生成恶意代码的担忧。与此同时，语法约束解码（GCD）已被广泛采用，通过强制语法有效性来提高大语言模型生成代码的可靠性。在本文中，我们揭示了一个反直觉的风险：这种以可靠性为导向的技术本身可能成为攻击面。我们发现了一种新的越狱攻击，称为 CodeSpear，它利用 GCD 诱导大语言模型生成恶意代码。我们的实验表明，仅应用良性的代码语法约束就可以有效地越狱大语言模型。为了解决这一漏洞，我们提出了 CodeShield，这是一种安全对齐方法，即使在攻击者控制的语法约束下也能稳健地保持安全行为。CodeShield 在代码模态中对模型进行对齐，教会它在 GCD 下生成蜜罐代码。这种代码在语义上……（摘要原文在此处截断）

    arXiv:2606.11817v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for code generation, raising concerns that they may be misused to produce malicious code. Meanwhile, Grammar-Constrained Decoding (GCD) has been widely adopted to improve the reliability of LLM-generated code by enforcing syntactic validity. In this paper, we reveal a counterintuitive risk: this reliability-oriented technique can itself become an attack surface. We uncover a new jailbreak attack, termed CodeSpear, that exploits GCD to induce LLMs into generating malicious code. Our experiments show that simply applying a benign code grammar constraint can effectively jailbreak LLMs.   To address this vulnerability, we propose CodeShield, a safety alignment approach that robustly preserves safe behavior even under attacker-controlled grammar constraints. CodeShield aligns the model in the code modality by teaching it to generate honeypot code under GCD. Such code is semantically
    
[^291]: Ekka：LLM推理中静默错误的自动化诊断

    Ekka: Automated Diagnosis of Silent Errors in LLM Inference

    [https://arxiv.org/abs/2606.04594](https://arxiv.org/abs/2606.04594)

    Ekka通过差分对比目标与参考LLM服务框架的中间执行状态，将静默错误诊断转化为差分调试问题，实现了自动化的根因定位，在真实错误基准上达到80%的pass@1诊断准确率。

    

    LLM服务框架正快速演进，形成了复杂的软件栈和大量的优化技术。快速的开发过程可能引入静默错误，即输出质量在没有任何显式错误信号的情况下悄然下降。由于高层症状与低层根本原因之间存在巨大的语义鸿沟，诊断这类静默错误是非常困难的。我们观察到，通过利用语义正确的参考实现，静默错误的诊断可以有效地转化为一个差分调试问题。我们提出了Ekka，一个自动化诊断系统，它通过系统地对比和对齐目标框架与参考框架之间的中间执行状态来识别根本原因。我们基于流行的服务框架构建了一个真实世界静默错误基准，Ekka在其中展现出80%的pass@1诊断准确率和88%的pass@5诊断准确率，超越了现有方法。

    arXiv:2606.04594v2 Announce Type: replace-cross  Abstract: LLM serving frameworks are quickly evolving with a complex software stack and a vast number of optimizations. The rapid development process can introduce silent errors where output quality silently degrades without any explicit error signals. Diagnosing silent errors is notoriously difficult due to the substantial semantic gap between the high-level symptoms and the low-level root causes. We observe that diagnosis of silent errors can be effectively framed as a differential debugging problem by leveraging the existence of semantically correct reference implementations. We propose Ekka, an automated diagnosis system that identifies root causes by systematically aligning and comparing intermediate execution states between a target and a reference framework. We constructed a benchmark of real-world silent errors from popular serving frameworks, where Ekka shows 80% pass@1 diagnosis accuracy and 88% pass@5 diagnosis accuracy, outpe
    
[^292]: 幻觉可从量化大语言模型的中间层隐藏状态中被线性解码

    Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs

    [https://arxiv.org/abs/2606.02628](https://arxiv.org/abs/2606.02628)

    量化开源大语言模型的中间层隐藏状态中存在近似线性的真实性信号，仅用单个中间层的简单线性探针即可高效检测幻觉（AUROC达0.904–1.000），显著优于采样式检测方法。

    

    我们研究了开源大语言模型是否在其隐藏状态中编码了线性可分的真实性信号，以及该信号在网络哪个深度处最强。我们在三个经指令微调的7B–8B模型（Llama-3.1-8B、Mistral-7B、Qwen2.5-7B）上以4-bit NF4量化加载，在四个幻觉基准（TruthfulQA、HaluEval-QA、FEVER以及一个受控合成数据集）上提取逐层隐藏状态，并比较了四种检测方法：线性探针与MLP探针、INSIDE EigenScore、自一致性以及注意力熵。在单个网络中间层上的线性探针在保留测试集上实现了0.904–1.000的AUROC，而基于采样的检测器在相同协议下AUROC不超过0.541。真实性信号近似线性：MLP探针很少能比线性探针高出0.01以上的AUROC。在自然语言基准上，各模型家族的峰值探测层落在一致的层区间内——块（注：摘要在此处被截断）

    arXiv:2606.02628v1 Announce Type: cross  Abstract: We investigate whether open-source LLMs encode a linearly separable truthfulness signal in their hidden states, and at which network depth this signal is strongest. Across three $7$B--$8$B instruction-tuned models (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B) loaded in $4$-bit NF4 quantization, we extract per-layer hidden states on four hallucination benchmarks (TruthfulQA, HaluEval-QA, FEVER, and a controlled synthetic set) and compare four detection approaches: linear and MLP probes, INSIDE EigenScore, self-consistency, and attention entropy. A linear probe on a single mid-network layer achieves $0.904$--$1.000$ AUROC on held-out splits, while sampling-based detectors do not exceed $0.541$ AUROC under the same protocol. The truthfulness signal is approximately linear: MLP probes rarely surpass linear probes by more than $0.01$ AUROC. Peak probing layers fall in a consistent band across model families on natural-language benchmarks -- blocks
    
[^293]: 物理基础模型从数值模拟到实验室湍流的涌现式迁移

    Emergent Transfer of a Physics Foundation Model from Simulation to Laboratory Turbulence

    [https://arxiv.org/abs/2606.01470](https://arxiv.org/abs/2606.01470)

    该论文验证了物理基础模型能够从数值模拟涌现式地迁移到真实实验室湍流，在瑞利-泰勒不稳定性这一长期存在模拟-实验差异（增长率相差约三倍）的难题上取得了突破。

    

    物理基础模型能否有效地部署于真实实验室实验，仍是科学机器学习（ML）领域一个悬而未决的问题。我们在瑞利-泰勒不稳定性（RTI）上检验了这一问题。RTI 是一种普遍存在且极具挑战性的流体不稳定性，从桌面尺度的流动到超新星爆发中均可见到：当较轻的流体加速进入较重的流体时，密度界面上的微小扰动会发展为混沌的、多尺度的混合。标准 ML 模型难以应对 RTI，并且尽管经过了一个多世纪的理论、数值与实验研究，RTI 仍存在一个模拟与实验之间未解决的差异：大多数实验室实验测得的晚期混合增长率 α（约 0.06–0.07）约为理想化直接数值模拟（DNS，约 0.02）所得数值的三倍，而这一差距的起源至今仍有争议。这些特性使 RTI 成为检验这一重要问题的严格试金石（原文摘要至此处截断）。

    arXiv:2606.01470v2 Announce Type: replace-cross  Abstract: Whether physics foundation models can be usefully deployed on laboratory experiments remains an open question for scientific machine learning (ML). We test this question on the Rayleigh-Taylor instability (RTI), a ubiquitous and demanding fluid instability seen from tabletop flows to supernova explosions, in which small perturbations at a density interface grow into chaotic, multiscale mixing as a lighter fluid accelerates into a heavier one. Standard ML models struggle with RTI, and despite over a century of theoretical, numerical, and experimental work, it carries an unresolved discrepancy between simulation and experiment: the late-time mixing growth rate, $\alpha$, measured in most laboratory experiments ($\sim$ 0.06-0.07), is roughly three times the value from idealized direct numerical simulations (DNS, $\sim$ 0.02). The gap's origin remains debated. These properties make RTI a stringent test for a question that matters w
    
[^294]: MusTBench：音乐大语言模型中时间定位能力的基准测试与提升

    MusTBench: Benchmarking and Advancing Temporal Grounding in Music LLMs

    [https://arxiv.org/abs/2605.29300](https://arxiv.org/abs/2605.29300)

    该论文提出了经音乐专家验证的MusTBench基准和涵盖四阶段优化的MusT方案，用于评估并提升音乐大语言模型将回答准确锚定到音频正确时间段的能力。

    

    近期的大型音频-语言模型（LALMs）在理解音乐内容方面展现出了可观的能力。然而，这些模型的回答是否锚定于音频中正确的时间区域，这一问题仍未得到充分探索。这一局限性对于音乐理解尤为关键，因为音乐中的关键信息往往以时间上局部化事件的形式出现，例如乐器进入和节奏转换。为填补这一空白，我们提出了MusTBench，一个经音乐专家验证的基准，旨在通过五个时间定位问答任务来评估LALM的时间定位能力。为进一步提升现有模型的时间定位能力，我们提出了MusT，一种新颖的四阶段时间优化方案，涵盖音乐编码器适配、大语言模型适配、大语言模型监督微调以及基于强化学习的优化。在MusTBench上的实验表明，现有LALM难以实现精确的时间定位，而MusT则带来了……

    arXiv:2605.29300v2 Announce Type: replace-cross  Abstract: Recent Large Audio-Language Models (LALMs) have demonstrated promising abilities in understanding musical content. However, whether their responses are grounded in the correct temporal regions of the audio remains underexplored. This limitation is particularly critical for music understanding, where key information often occurs as temporally localized events, such as instrument entries and rhythmic transitions. To address this gap, we introduce MusTBench, a music-expert-validated benchmark designed to evaluate temporal grounding in LALMs through five temporally grounded question-answering tasks. To further improve temporal grounding in existing models, we propose MusT, a novel four-stage temporal optimization recipe spanning music encoder adaptation, LLM adaptation, LLM supervised fine-tuning, and RL-based optimization. Experiments on MusTBench show that existing LALMs struggle with precise temporal grounding, while MusT brings
    
[^295]: Recon：面向用户建模的重建引导式推理合成

    Recon: Reconstruction-Guided Reasoning Synthesis for User Modeling

    [https://arxiv.org/abs/2605.26969](https://arxiv.org/abs/2605.26969)

    Recon方法以动作重建的保真度作为推理轨迹质量的评分标准，从而筛选出真正具有预测能力的推理而非事后合理化，使用户建模更贴近用户真实的决策过程。

    

    用户建模旨在利用语言模型（LM）从过去上下文-动作对（如对话轮次）的语料库中模仿个体的行为，从而在行为科学、人机协作和市场研究等场景中实现用户模拟。近期的方法通过合成推理轨迹来增强这些语料库，这些轨迹通常以上下文和动作共同作为条件生成。然而，以动作为条件的推理轨迹属于事后合理化，未必能反映用户真实的底层决策过程。我们提出Recon，它利用动作重建来根据推理轨迹的预测能力为其评分：给定上下文和候选推理，由一个重建模型预测动作，并以重建的保真度来判定推理质量。在四个领域中，Recon相对于Backward Synthesis（一种标准的事后合理化基线）取得了54.7%的胜率。此外，我们还发现训练一个推理模型……（摘要在此处被截断）

    arXiv:2605.26969v2 Announce Type: replace-cross  Abstract: User modeling aims to use language models (LMs) to mimic an individual's behavior from a corpus of past context-action pairs (e.g., conversation turns), enabling the simulation of users in settings like behavioral science, human-AI collaboration, and market research. Recent approaches augment these corpora with synthesized reasoning traces, typically generated by conditioning on both context and action. However, action-conditioned traces are post-hoc rationalizations and need not reflect the user's underlying decision process. We propose Recon, which uses action reconstruction to score reasoning traces by their predictive power: given a context and candidate reasoning, a reconstruction model predicts the action, and reconstruction fidelity determines reasoning quality. Across four domains, Recon achieves a 54.7% win rate over Backward Synthesis, a standard post-hoc rationalization baseline. Further, we find that training a reas
    
[^296]: 更具表达力的前馈层：第一部分：令牌自适应的激活混合

    More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations

    [https://arxiv.org/abs/2605.26647](https://arxiv.org/abs/2605.26647)

    本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。

    

    前馈网络（FFN）层在基于Transformer的大语言模型（LLM）中占据了相当大的参数比例和非线性表达能力。尽管激活函数已从ReLU和GELU演进到SwiGLU等门控变体，但大多数FFN设计仍然使用单一固定的激活函数，对所有令牌（token）应用相同的非线性变换。在这项工作中，我们提出了激活混合（Mixture of Activations, MoA），这是一种令牌自适应的FFN设计，它使用轻量级的、依赖于输入的门控机制来混合一个激活函数字典，同时共享相同的线性投影。作为输入无关的对应方案，我们还引入了可学习激活（Learnable Activations, LA），它为ReLU型和SwiGLU型FFN构造激活函数的线性组合。在理论方面，我们在固定激活FFN、LA和MoA之间建立了严格的有限宽度表达能力分离关系：LA严格包含固定激活FFN的表达能力，而MoA又严格包含LA，并进一步……

    arXiv:2605.26647v2 Announce Type: replace  Abstract: Feedforward network (FFN) layers account for a large fraction of parameters and nonlinear expressivity in Transformer-based large language models (LLMs). Despite the evolution from ReLU and GELU to gated variants such as SwiGLU, most FFN designs still use a single fixed activation function, applying the same nonlinear transformation to all tokens. In this work, we propose Mixture of Activations (MoA), a token-adaptive FFN design that mixes a dictionary of activation functions using lightweight input-dependent gates while sharing the same linear projections. As an input-independent counterpart, we also introduce learnable activations (LA), which form linear combinations of activation functions for both ReLU-type and SwiGLU-type FFNs. Theoretically, we establish strict finite-width expressive separations among fixed-activation FFNs, LA, and MoA: LA strictly contains fixed-activation FFNs, while MoA strictly contains LA, with the additi
    
[^297]: AvalancheBench：通过潜在世界恢复评估企业级智能体驱动的语义查询处理

    AvalancheBench: Evaluating Enterprise Agent-Driven Semantic Query Processing Through Latent World Recovery

    [https://arxiv.org/abs/2605.24183](https://arxiv.org/abs/2605.24183)

    该论文提出了AvalancheBench，一个从潜在世界规范生成观测数据、通过涵盖文本、图像和文档的58个多部分问题评估数据智能体端到端语义查询能力的多模态企业分析基准，并首次揭示了错误经合法依赖关系传播、不当复用中间结果导致无关问题失败的“雪崩效应”。

    

    随着大语言模型（LLM）能力的提升，数据智能体正成为数据分析的新接口，而语义查询处理引擎则利用LLM对结构化和非结构化数据进行操作。现有基准测试大多将智能体与语义查询引擎分开评估，因此智能体能否在端到端分析中决定何时以及如何使用语义算子仍不清楚。我们提出了AvalancheBench，这是一个多模态企业分析基准，它从潜在世界规范生成观测数据，并据此评估数据智能体恢复该潜在世界的能力。该基准包含3个经过验证的任务，共58个问题，涵盖文本、图像和文档。这些问题以多部分分析简报的形式一并提供，且可能共享中间结果，从而暴露出“雪崩效应”：错误可以通过合法的依赖关系传播，而不恰当的复用会导致原本互不相关的问题失败。AvalancheBench测……（原文摘要至此截断）

    arXiv:2605.24183v2 Announce Type: replace-cross  Abstract: As LLMs improve, data agents are emerging as a new interface for analytics, while semantic query processing engines use LLMs to operate over structured and unstructured data. Existing benchmarks largely evaluate agents and semantic query engines in isolation, leaving it unclear whether agents can decide when and how to use semantic operators in end-to-end analysis.   We introduce AvalancheBench, a multimodal enterprise analytics benchmark that generates observed data from latent world specifications and evaluates data agents on their ability to recover that world. The benchmark's 3 validated tasks contain 58 questions spanning text, images, and documents. Questions are provided together as a multi-part analytical brief and may share intermediate results, exposing the avalanche effect: errors can propagate through legitimate dependencies, and inappropriate reuse can cause otherwise unrelated questions to fail. AvalancheBench mea
    
[^298]: 设计知识工作的基准测试

    Designing Benchmarks for Knowledge Work

    [https://arxiv.org/abs/2605.23262](https://arxiv.org/abs/2605.23262)

    本文提出了一种以工作为中心的基准测试表示方法，通过四个明确字段（代表的活动、测试的设置、所需的工作产品、评估的结果）来显式化知识工作基准测试中的隐含选择，从而支持跨职业的活动级报告和比较。

    

    arXiv:2605.23262v2 公告类型：替换 摘要：AI代理正迅速从回答孤立问题转向通过工具、软件环境和多步骤工作流程完成任务。这些系统现在被要求执行的许多任务都是知识工作，其中信息和专业知识在完成工作的过程中被解释、产生和传达。针对这一场景的基准测试通常仅通过任务、环境和指标来描述，这使四个问题隐含不清：工作代表哪部分、在什么条件下测试、系统应留下什么工作产品，以及基准测试实际评估该产品的哪部分。我们引入了一种以工作为中心的基准测试表示方法，包含四个字段：代表的活动、测试的设置、所需的工作产品和评估的结果。这种表示方法使这些选择在基准测试设计中明确且可比较。为了支持跨职业的活动级报告，我们推导出

    arXiv:2605.23262v2 Announce Type: replace  Abstract: AI agents are moving quickly from answering isolated questions toward completing work through tools, software environments, and multi-step workflows. Much of what these systems are now asked to do is knowledge work, where information and expertise are interpreted, produced, and communicated as part of completing work. Benchmarks for this setting are usually described only by their tasks, environments, and metrics, leaving four questions implicit: what part of the work is represented, under what conditions it is tested, what work product the system is expected to leave, and what part of that product the benchmark actually evaluates. We introduce a work-centered benchmark representation with four fields: represented activity, tested setting, required work product, and evaluated result. The representation makes these choices explicit and comparable across benchmark designs. To support activity-level reporting across occupations, we deri
    
[^299]: EmoTrack：面向基于文本的抑郁严重程度估计的临床-语义建模

    EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation

    [https://arxiv.org/abs/2605.22286](https://arxiv.org/abs/2605.22286)

    本文提出EmoTrack框架，通过临床-语义建模将临床特征分数与话语级表示相融合，从心理咨询文本中预测PHQ-8抑郁严重程度总分，解决了临床评估与对话证据之间关联不足的问题。

    

    基于文本的心理咨询为评估抑郁严重程度提供了宝贵的信息来源。我们研究了如何从咨询对话记录中预测八项患者健康问卷（PHQ-8）的总分，PHQ-8是一种抑郁严重程度的自我报告测量工具。基于临床的方法主要依靠大语言模型（LLM）推理来获得结构化的会话级评估，但这些评估提供的信息有限，无法说明哪些话语支持每个评分。基于训练的方法直接在句子或其语义嵌入上训练预测器，能够保留局部对话细节，但必须从有限的有标签对话记录中学习临床结构。将少量临床特征分数与长序列的高维话语表示相集成极具挑战性，因为简单的融合方法可能会过度强调对话证据，并且无法将临床特征与支持性话语相关联。（原文摘要在此处截断）

    arXiv:2605.22286v2 Announce Type: replace-cross  Abstract: Text-based counseling provides a valuable source of information for assessing depression severity. We study prediction of the total score on the eight-item Patient Health Questionnaire (PHQ-8), a self-report measure of depression severity, from counseling transcripts. Clinical-based methods rely mainly on large language model (LLM) inference to obtain structured session-level assessments, but these assessments provide limited information about which utterances support each score. Training-based methods train predictors directly on sentences or their semantic embeddings and preserve local conversational detail, but must learn clinical structure from limited labeled transcripts. Integrating a small set of clinical feature scores with a long sequence of high-dimensional utterance representations is challenging, as simple fusion can overemphasize dialogue evidence and fail to link clinical features to supporting utterances. Combini
    
[^300]: MONET：一个大规模、开放、无冗余且内容丰富的文生图数据集

    MONET: A Massive, Open, Non-redundant and Enriched Text-to-image dataset

    [https://arxiv.org/abs/2605.21272](https://arxiv.org/abs/2605.21272)

    该论文提出了MONET——一个包含1.038亿经过安全过滤、去重和多视觉语言模型重新标注的图像-文本对的开放文生图数据集，仅使用该数据集训练的40亿参数潜在扩散模型便取得了具有竞争力的GenEval和DPG分数。

    

    训练大型文生图模型需要高质量、经过精心筛选、内容多样且配有详细描述的数据集。然而，大规模收集、过滤、去重和重新标注此类语料库的成本和复杂性阻碍了该领域的开放性与可复现性研究。我们提出了MONET，一个基于Apache 2.0许可的开放数据集，包含1.038亿图像-文本对。这些数据从29亿原始数据对中收集而来，来源涵盖多个异构开放资源，经过安全过滤、基于领域的过滤、精确与近似去重，以及使用多个视觉语言模型进行从短篇到长篇描述的重新标注等一系列处理阶段，并进一步用合成生成的样本进行扩充。每个图像都附带预计算的嵌入向量和标注，以加速下游应用。为验证MONET的有效性，我们仅在该数据集上训练了一个40亿参数的潜在扩散模型，并取得了具有竞争力的GenEval和DPG分数，证明了……

    arXiv:2605.21272v2 Announce Type: replace-cross  Abstract: Training large text-to-image models requires high-quality, curated datasets with diverse content and detailed captions. Yet the cost and complexity of collecting, filtering, deduplicating, and re-captioning such corpora at scale hinder open and reproducible research in the field. We introduce MONET, an open Apache 2.0 dataset of 103.8M image-text pairs collected from 2.9B raw pairs across heterogeneous open sources through successive stages of safety filtering, domain-based filtering, exact and near-duplicate removal, and re-captioning with multiple vision-language models covering short to long-form descriptions, and further augmented with synthetically generated samples. Each image is shipped with pre-computed embeddings and annotations to accelerate downstream use. To validate the effectiveness of MONET, we train a 4B-parameter latent diffusion model exclusively on it and reach competitive GenEval and DPG scores, demonstratin
    
[^301]: 智能体安全是一个系统问题

    Agent Security is a Systems Problem

    [https://arxiv.org/abs/2605.18991](https://arxiv.org/abs/2605.18991)

    本文主张智能体安全应作为系统问题来处理——将AI模型视为不受信任的组件并在系统层面强制执行安全不变量，并通过分析十一例真实攻击论证了系统安全原则本可预防这些攻击。

    

    我们主张应将智能体安全作为一个系统问题来处理：为智能体提供驱动的AI模型必须被视为不受信任的组件，而安全不变量必须在系统层面加以强制执行。通过这一视角来看，提高模型鲁棒性的努力（这是社区中的主流观点）本身是不够的。相反，我们必须用系统安全领域的技术来补充现有工作。基于我们作为网络安全研究者在操作系统、网络、形式化方法和对抗性机器学习方面的经验，我们阐述了一套核心原则，这些原则植根于数十年的系统安全研究，为设计具有可预测保障的智能体系统提供了基础。作为证据，我们分析了十一例针对智能体的代表性真实世界攻击，并讨论了如果实现这些系统原则，本可以如何阻止这些攻击。我们还识别了相关研究方向……

    arXiv:2605.18991v3 Announce Type: replace-cross  Abstract: We take the position that agent security must be approached as a systems problem: the AI model powering the agent must be treated as an untrusted component, and security invariants must be enforced at the system level. Through this lens, efforts to increase model robustness (the dominant viewpoint in the community) are insufficient on their own. Instead, we must complement existing efforts with techniques from the systems security domain. Based on our experience as cybersecurity researchers in operating systems, networks, formal methods, and adversarial machine learning, we articulate a set of core principles, grounded in decades of systems security research, that provide a foundation for designing agentic systems with predictable guarantees. As evidence, we analyze eleven representative real-world attacks on agents and discuss how systems principles, if realized, could have prevented these attacks. We also identify the researc
    
[^302]: 学习如何从多模态经验中学习

    Learning to Learn from Multimodal Experience

    [https://arxiv.org/abs/2605.16857](https://arxiv.org/abs/2605.16857)

    提出AutoMMemo通用框架，通过将记忆机制表示为可执行的备忘程序并迭代发现有效程序，使智能体能够自动设计记忆机制，从而从多模态经验中学习。

    

    多模态智能体日益与类真实世界环境进行交互，并生成涵盖感知、推理与行动的丰富轨迹。一个关键的研究焦点是如何将这些原始的多模态轨迹转化为可复用的经验，以提升未来的任务表现。现有的经验驱动学习方法大多是在文本环境中开发的，且通常依赖人工设计的记忆模式，这使得它们在处理需要在模态、时间尺度和抽象层级上组合异构信号的多模态经验时效果欠佳。我们提出了AutoMMemo，一个通用框架，使智能体能够自动设计用于从多模态经验中学习的记忆机制。AutoMMemo将每个记忆机制表示为一个可执行的备忘程序，该程序定义了过去情节如何被存储、组织和检索。它通过一个迭代过程来发现有效的备忘程序（原文摘要在此处截断）。

    arXiv:2605.16857v2 Announce Type: replace  Abstract: Multimodal agents increasingly interact with real-world-like environments and generate rich trajectories spanning perception, reasoning, and action. A key research focus is how to transform these raw multimodal trajectories into reusable experience that can improve future task performance. Existing experience-driven learning methods are mostly developed in textual settings and typically rely on manually designed memory schemas, making them less effective for multimodal experience that requires composing heterogeneous signals across modalities, temporal scales, and abstraction levels. We propose AutoMMemo, a general framework that enables agents to automatically design memory mechanisms for learning from multimodal experience. AutoMMemo represents each memory mechanism as an executable memo program, which defines how past episodes are stored, organized, and retrieved. It discovers effective memo programs through an iterative process o
    
[^303]: 预测-干预博弈与不变集

    Prediction-Intervention Games and Invariant Sets

    [https://arxiv.org/abs/2605.16828](https://arxiv.org/abs/2605.16828)

    该论文提出预测-干预博弈（Stackelberg博弈的特例），并证明对于两类常见的跟随者干预目标，基于稳定毯这一不变子集的预测器性能至少与基于因果父节点的预测器相当。

    

    我们考虑以下双人博弈：领导者利用观测数据，从给定协变量中为响应变量 $Y$ 选择一个预测函数。随后，跟随者通过对底层结构因果模型中的某些协变量进行干预来做出反应，以最大化自身的目标。领导者知道干预的目标变量，但对跟随者的目标可能了解有限。我们将这种设定称为预测-干预博弈，它是Stackelberg博弈的一种特例。寻找领导者的最优策略通常是困难的。为避免严重的性能损失，领导者可以将其预测建立在 $Y$ 的因果父节点之上，或更一般地建立在协变量的某个不变子集之上。我们证明，对于两类常见的跟随者目标，基于稳定毯（一种不变子集）的预测器至少与基于因果父节点的预测器一样好。我们进一步对领导者干预后……

    arXiv:2605.16828v2 Announce Type: replace-cross  Abstract: We consider the following two-player game: using observational data, the leader chooses a prediction function for a response variable $Y$ from given covariates. The follower then reacts with an intervention on some covariates in the underlying structural causal model to maximize their own objective. The leader knows the intervention targets, but may have limited knowledge of the follower's objective. We call this setup a prediction-intervention game, a special case of a Stackelberg game. Finding an optimal strategy for the leader is generally difficult. To avoid severe performance loss, the leader may base their prediction on the causal parents of $Y$, or more generally on an invariant subset of covariates. We prove, for two common classes of follower objectives, that predictors based on the stable blanket, an invariant subset, are at least as good as those based on the causal parents. We further upper bound the leader's post-i
    
[^304]: Baba漫游奇境：面向可执行世界模型的在线自监督动力学发现

    Baba in Wonderland: Online Self-Supervised Dynamics Discovery for Executable World Models

    [https://arxiv.org/abs/2605.16725](https://arxiv.org/abs/2605.16725)

    提出Alice闭环系统，将候选更新失败引发的“保持性冲突”作为结构性信号，归纳出假设类以修复可执行世界模型并引导前沿探索，从而仅凭交互证据实现先验失配下的在线自监督环境动力学发现。

    

    可执行世界模型可以被读取、编辑、执行并复用于规划，但前提是程序捕获的是环境的转移规律，而非其表面词汇中的语义捷径。我们研究了先验失配情况下的在线可执行世界模型学习问题：此时智能体必须仅凭交互证据归纳出依赖于状态的动力学，无法依赖规则描述、奖励信号或可信的词汇先验。我们提出了Alice，一个闭环系统，它将失败的候选更新视为结构性信号：当一个候选能够解释新的转移却丢失了先前已解释的转移时，这种保持性冲突便揭示了当前程序所混淆的动力学。Alice将这些冲突细化为假设类，这些假设类一方面为更新提供少量被破坏的例子（每个受影响的类别各一个），另一方面引导前沿探索走向相对当前程序而言新颖且欠表示的转移。

    arXiv:2605.16725v2 Announce Type: replace  Abstract: Executable world models can be read, edited, executed, and reused for planning, but only if the program captures the environment's transition law rather than semantic shortcuts in its surface vocabulary. We study online executable world-model learning under prior misalignment, where an agent must induce state-dependent dynamics from interaction evidence alone, without rule descriptions, reward signals, or trustworthy lexical priors. We introduce Alice, a closed-loop system that treats failed candidate updates as structural signal: when a candidate explains a new transition but loses previously explained ones, the preservation conflict reveals dynamics that the current program had conflated. Alice refines these conflicts into hypothesis classes that both provide a few broken examples, one per affected class, for update and guide frontier exploration toward transitions that are novel and underrepresented with respect to the current pro
    
[^305]: 达尔文家族：基于MRI-信任加权的进化合并，实现免训练的语言模型推理扩展

    Darwin Family: MRI-Trust-Weighted Evolutionary Merging for Training-Free Scaling of Language-Model Reasoning

    [https://arxiv.org/abs/2605.14386](https://arxiv.org/abs/2605.14386)

    Darwin Family通过14维自适应合并基因组、MRI-信任融合和跨架构杂交实现免训练的大语言模型进化合并，其旗舰模型Darwin-27B-Opus在GPQA Diamond上取得86.9%的成绩（在1,252个模型中排名第六），无需任何梯度训练即超越其完全训练的基础模型。

    

    我们提出了Darwin Family，一个通过无梯度权重空间重组实现大语言模型免训练进化合并的框架。我们探究是否可以在不进行额外训练的情况下提升前沿级别的推理性能，方法是重组现有检查点中已经编码的潜在能力。Darwin引入了三个关键思想：(i) 一个14维自适应合并基因组，支持细粒度的组件级和块级重组；(ii) MRI-信任融合，通过可学习的信任参数自适应地平衡诊断性层重要性信号与进化搜索；以及 (iii) 一个架构映射器，实现异构模型家族之间的跨架构杂交。实验上，旗舰模型Darwin-27B-Opus在GPQA Diamond上达到86.9%，在1,252个评估模型中排名第六，且无需任何基于梯度的训练即超越了其完全训练的基础模型。跨规模（摘要在此处截断）

    arXiv:2605.14386v2 Announce Type: replace-cross  Abstract: We present Darwin Family, a framework for training-free evolutionary merging of large language models via gradient-free weight-space recombination. We ask whether frontier-level reasoning performance can be improved without additional training, by reorganizing latent capabilities already encoded in existing checkpoints. Darwin introduces three key ideas: (i) a 14-dimensional adaptive merge genome enabling fine-grained component- and block-level recombination; (ii) MRI-Trust Fusion, which adaptively balances diagnostic layer-importance signals with evolutionary search through a learnable trust parameter; and (iii) an Architecture Mapper that enables cross-architecture breeding between heterogeneous model families. Empirically, the flagship Darwin-27B-Opus achieves 86.9% on GPQA Diamond, ranking #6 among 1,252 evaluated models, and outperforming its fully trained foundation model without any gradient-based training. Across scales
    
[^306]: GraphIP-Bench：窃取一个图神经网络有多难？我们能否阻止它？

    GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?

    [https://arxiv.org/abs/2605.12827](https://arxiv.org/abs/2605.12827)

    提出了首个统一基准GraphIP-Bench，在单一黑盒协议下系统评估12种模型提取攻击与12种GNN知识产权防御方法（覆盖10个图数据集、3种骨干网络和3类任务），首次可靠地回答了窃取GNN的难度以及防御方法能否有效阻止窃取这两个问题。

    

    部署为云服务的图神经网络（GNN）可能通过*模型提取攻击*被“窃取”——攻击者利用查询响应训练代理模型来复现目标模型的行为，而不断涌现的所有权防御方法则试图阻止或追踪此类窃取行为。本文的标题提出了两个问题：*窃取一个GNN有多难？*以及*我们能否阻止它？*由于以往实验使用的数据集、威胁模型和评估指标不一致，先前的工作无法回答这两个问题。我们提出了*GraphIP-Bench*，这是一个在统一黑盒协议下同时评估攻防双方的基准。它集成了十二种提取攻击、横跨水印、输出扰动和查询模式检测三大类的十二种防御方法、十个覆盖同配、异配和大规模场景的公开图数据集、三种GNN骨干网络以及三种图学习任务，并报告保真度、任务效用、所有权验证和计算成本等指标。

    arXiv:2605.12827v3 Announce Type: replace-cross  Abstract: Graph neural networks (GNNs) deployed as cloud services can be *stolen* through *model-extraction attacks*, which train a surrogate from query responses to reproduce the target's behaviour, and a growing line of ownership defenses tries to prevent or trace such theft. The title of this paper asks two questions: *how hard is it to steal a GNN?*, and *can we stop it?* Prior work cannot answer either, because experiments use inconsistent datasets, threat models, and metrics. We introduce *GraphIP-Bench*, a unified benchmark which evaluates both sides under a single black-box protocol. It integrates twelve extraction attacks, twelve defenses spanning watermarking, output-perturbation, and query-pattern-detection families, ten public graphs covering homophilic, heterophilic, and large-scale regimes, three GNN backbones, and three graph-learning tasks, and it reports fidelity, task utility, ownership verification, and computational c
    
[^307]: 每个有限群都容许一个恰有限表现

    Every finite group admits a just finite presentation

    [https://arxiv.org/abs/2605.10402](https://arxiv.org/abs/2605.10402)

    本文证明了每个有限群都存在“恰有限”表现，即移除任意一条定义关系后就会变成无限群表现的有限表现，从而肯定地解决了 Kourovka 笔记本中的公开问题 21.10。

    

    有限群的一个有限表现 < X | R > 被称为“恰有限的”，如果从 R 中移除任何一条关系后，所得的表现对应一个无限群。每个有限群是否都容许这样的表现，一直是一个悬而未决的问题（Kourovka 笔记本，问题 21.10）。我们以肯定的方式解决了这一猜想。

    arXiv:2605.10402v2 Announce Type: replace-cross  Abstract: A finite presentation < X | R > of a finite group is called `just finite' if removing any relation from R results in a presentation for an infinite group. It has been an open question (Kourovka Notebook, Problem 21.10) whether every finite group admits such a presentation. We resolve this conjecture in the affirmative.
    
[^308]: 格演绎Transformer

    Lattice Deduction Transformers

    [https://arxiv.org/abs/2605.08605](https://arxiv.org/abs/2605.08605)

    格演绎Transformer（LDT）通过在前向传递之间以格投影潜在状态来近似逻辑可靠的演绎推理，仅用80万参数和不到现有模型2%的训练计算量，就以近乎完全可靠的方式解决了99.999%的极难数独谜题。

    

    我们提出了格演绎Transformer（LDT），这是一种循环Transformer，通过在前向传递之间将其潜在状态投影到一个格上，来近似逻辑上可靠的演绎推理。我们在一个与基于搜索的约束求解器中的演绎过程相镜像的流程中进行在线训练，并通过基于抽象解释的、与领域无关的解候选集近似方法来监督训练。一个80万参数的LDT在Sudoku-Extreme（极难数独）基准上超越了以往的小型循环推理模型，且训练所需的估计浮点运算量不到后者的2%，解决了422,786个测试谜题中的99.999%。该模型几乎总是可靠的：在127万次谜题评估中仅给出4次错误答案，其余情况要么返回正确答案，要么选择弃权。一个180万参数的LDT变体在Maze-Hard（高难度迷宫）上达到了99.8%的准确率。我们的代码发布于 github.com/lcrh/lattice-deduction-transformers。

    arXiv:2605.08605v2 Announce Type: replace-cross  Abstract: We introduce the Lattice Deduction Transformer (LDT), a recurrent transformer that approximates logically sound deduction by projecting its latent state through a lattice between forward passes. We train on-policy in a process that mirrors deduction in a search-based constraint solver and supervise training via a domain-agnostic approximation of the set of solution candidates based on abstract interpretation. An $800$K-parameter LDT outperforms prior small recurrent reasoners on Sudoku-Extreme with under $2\%$ of their estimated training FLOPs, solving $99.999\%$ of the $422{,}786$ test puzzles. The model is nearly always sound, in that it gives a wrong answer for $4$ of $1.27$M puzzle evaluations, and otherwise returns a correct answer or abstains. A $1.8$M-parameter LDT variant reaches $99.8\%$ accuracy on Maze-Hard. Our code is available at github.com/lcrh/lattice-deduction-transformers.
    
[^309]: 以KL换KL：基于控制变量基线的在线策略蒸馏

    KL for a KL: On-Policy Distillation with Control Variate Baseline

    [https://arxiv.org/abs/2605.07865](https://arxiv.org/abs/2605.07865)

    提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。

    

    在线策略蒸馏已成为大语言模型后训练的主导范式，尤其是在推理领域。然而，由于其单样本蒙特卡洛估计器具有高梯度方差，OPD在实践中仍然不稳定，且实现稳定训练的方法尚不成熟。我们提出了vOPD（带有控制变量基线的在线策略蒸馏），该方法将OPD视为策略梯度强化学习，并通过引入来自强化学习文献的控制变量基线（通常是一个价值函数）来稳定训练过程。我们证明OPD价值函数存在闭式解：即学生模型与教师模型之间逐token的负反向KL散度，该数值可直接从现有的前向传播中获得，无需额外的评论家网络或推理过程。现有的稳定化方法要么需要在整个词表上计算完整的token级反向KL散度，带来显著的开销，要么将其限制在top-k支持集上，从而引入偏差。

    arXiv:2605.07865v2 Announce Type: replace-cross  Abstract: On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for Large Language Models, especially in reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline---canonically a value function---from the RL literature. We show that the OPD value function admits a closed form: the per-token negative reverse KL divergence between student and teacher, available directly from the existing forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing th
    
[^310]: 通过推理空间压缩的结构化推理依据蒸馏

    Structural Rationale Distillation via Reasoning Space Compression

    [https://arxiv.org/abs/2605.07139](https://arxiv.org/abs/2605.07139)

    提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。

    

    将大型语言模型（LLM）的推理能力蒸馏到较小模型时，教师模型对相似问题给出的推理依据在结构和策略上往往差异巨大。就像厨师每次做同一道菜的方式都不同一样，这种不一致性给学生模型带来了难以内化的噪声监督。我们提出了“通过推理路径压缩进行蒸馏”的方法（D-RPC），该方法约束教师模型遵循一个紧凑的、动态维护的可复用高层推理路径库。对于每个训练问题，D-RPC检索最相关的路径并引导教师模型遵循该路径，从而产生在相似问题之间保持一致、同时又能覆盖不同问题类型的多样化推理依据。PAC-Bayes分析形式化了路径库大小与覆盖率之间的权衡：较小的路径库会限制路径多样性，从而带来覆盖缺口的风险。我们的消融实验倾向于中等大小的路径库，这与该理论分析相一致。

    arXiv:2605.07139v2 Announce Type: replace-cross  Abstract: When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes a trade-off between bank size and coverage, with smaller banks limiting path diversity at the risk of coverage gaps. Our ablations favor an intermediate bank size, consistent with thi
    
[^311]: 基于希尔伯特丛与胞腔层的一致几何深度学习

    Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves

    [https://arxiv.org/abs/2605.06395](https://arxiv.org/abs/2605.06395)

    该论文提出了 HilbNets——一种以希尔伯特丛的联络拉普拉斯算子作为卷积算子的新型卷积学习框架，可处理流形上可能无穷维的信号，并通过采样诱导的希尔伯特胞腔层使卷积运算得以实际实现。

    

    现代深度学习架构越来越多地需要处理本质上为无穷维的复杂信号，例如时间序列、概率分布或算子，这些信号定义在不规则的域上。然而，针对这些情形一直缺乏统一的学习理论。为开始填补这一空白，我们针对定义在流形上的、可能为无穷维的信号，提出了一种新颖的卷积学习框架。具体而言，我们将与希尔伯特丛相关联的联络拉普拉斯算子用作卷积算子，并由此推导出滤波器和神经网络，称为 HilbNets。我们通过一个两阶段的采样过程使 HilbNets 以及更一般的卷积运算得以实现。首先，我们证明对流形进行采样会诱导出一个希尔伯特胞腔层——这是一种具有希尔伯特特征空间和逐边耦合规则的广义图结构，并且我们证明了其层拉普拉斯算子收敛……（摘要在此处被截断）

    arXiv:2605.06395v3 Announce Type: replace-cross  Abstract: Modern deep learning architectures increasingly contend with sophisticated signals that are natively infinite-dimensional, such as time series, probability distributions, or operators, and are defined over irregular domains. Yet, a unified learning theory for these settings has been lacking. To start addressing this gap, we introduce a novel convolutional learning framework for possibly infinite-dimensional signals supported on a manifold. Namely, we use the connection Laplacian associated with a Hilbert bundle as a convolutional operator, and we derive filters and neural networks, dubbed as \textit{HilbNets}. We make HilbNets and, more generally, the convolution operation, implementable via a two-stage sampling procedure. First, we show that sampling the manifold induces a Hilbert Cellular Sheaf, a generalized graph structure with Hilbert feature spaces and edge-wise coupling rules, and we prove that its sheaf Laplacian conver
    
[^312]: 序数信用分配：仅凭比较的多步强化学习

    Ordinal Credit Assignment: Multi-Step Reinforcement Learning from Comparisons Alone

    [https://arxiv.org/abs/2605.06375](https://arxiv.org/abs/2605.06375)

    该论文提出一种在多步任务中仅依靠轨迹间两两比较（无需奖励函数或奖励模型）的序数信用分配方法，用动作在同一时间步群体中的胜率替代未来累积奖励，证明其期望更新等价于策略梯度，且在确定性环境中按真实价值排序、在随机环境中偏好高胜率动作，并对标签噪声和位置偏差具有鲁棒性。

    

    策略梯度方法需要一个标量奖励，然而对许多任务而言，没有任何公式能够刻画我们真正想要的目标，而判断两个结果中哪个更好却很容易。我们研究在多步任务中仅接收此类比较信号的强化学习——不需要奖励函数，也不需要奖励模型。多组轨迹从共享状态出发，每个动作的信用根据其未来累积奖励与同一时间步上其他轨迹的比较结果来分配。我们证明，期望更新等价于一种策略梯度，其中未来累积奖励被替换为该动作在当时仍在运行的轨迹群体中的胜率，并按对手仍在运行的概率进行加权；在确定性环境中，该更新按真实价值对动作排序，且最优策略是稳定的；而在随机结果下，它更偏好胜率更高的动作，而非均值更高的动作。平局附近的标签噪声或由位置偏差引起的噪声不会改变期望更新；只

    arXiv:2605.06375v2 Announce Type: replace-cross  Abstract: Policy-gradient methods need a scalar reward, yet for many tasks no formula captures what we want, while judging which of two outcomes is better is easy. We study reinforcement learning that receives only such comparisons - no reward function, no reward model - in multi-step tasks. Groups of trajectories start from a shared state, and each action is credited by how its reward-to-go compares with those of other trajectories at the same time step. We show the expected update is policy gradient with reward-to-go replaced by its win rate against the population still running at that time, weighted by the chance an opponent is still running; in deterministic environments the update ranks actions by true value and the optimal policy is stable, whereas under stochastic outcomes it prefers actions that win more often rather than those with higher mean. Label noise near ties or from position bias leaves the expected update unchanged; onl
    
[^313]: 基于不可靠反馈的上下文黑盒优化

    In-Context Black-Box Optimization with Unreliable Feedback

    [https://arxiv.org/abs/2605.06187](https://arxiv.org/abs/2605.06187)

    该论文提出反馈感知的上下文黑盒优化框架FICBO，通过结构化先验预训练Transformer，使其能同时利用优化历史和可能不可靠的辅助反馈，从而在反馈质量参差不齐的情况下实现稳健的跨任务优化。

    

    科学与工程中的黑盒优化通常伴随着辅助信息：专家、模拟器、预训练预测器或启发式方法可以提出有前景的候选方案。这类反馈能够加速搜索，但也可能存在偏差、依赖于输入或具有误导性。感知反馈的贝叶斯优化方法通常独立处理各个任务，限制了跨反馈源的适应能力；而上下文优化器虽然能够实现跨任务适应，但在测试时通常仅依赖优化历史。我们研究了基于反馈信息的上下文黑盒优化（FICBO），其中预训练的优化器同时以观测历史和当前候选集的廉价辅助反馈为条件。我们引入了关于辅助反馈与目标之间关系的结构化先验，并利用它们在从高度信息丰富到具有误导性的各种情形下预训练一个感知反馈的 Transformer。在测试时，该模型估计……（原文摘要截断于此）

    arXiv:2605.06187v2 Announce Type: replace-cross  Abstract: Black-box optimization in science and engineering often comes with side information: experts, simulators, pretrained predictors, or heuristics can suggest promising candidates. Such feedback can accelerate search, but can also be biased, input-dependent, or misleading. Feedback-aware Bayesian optimization methods typically handle tasks independently, limiting adaptation across feedback sources. In-context optimizers enable cross-task adaptation, but usually rely only on optimization history at test time. We study feedback-informed in-context black-box optimization (FICBO), where a pretrained optimizer conditions on both the observed history and cheap auxiliary feedback for the current candidate set. We introduce structured priors over how auxiliary feedback relates to the objective, and use them to pretrain a feedback-aware transformer across regimes ranging from highly informative to misleading. At test time, the model estimat
    
[^314]: GUI智能体中的强化学习：迈向数字居民的综述

    Reinforcement Learning in GUI Agents: A Survey Toward Digital Inhabitants

    [https://arxiv.org/abs/2604.27955](https://arxiv.org/abs/2604.27955)

    本论文首次对强化学习与GUI智能体的交叉领域进行全面综述，提出了将现有方法分为离线RL、在线RL和混合策略的系统性分类法，并展望了GUI智能体向“数字居民”演进的研究方向。

    

    图形用户界面（GUI）智能体已成为一种有前景的范式，使智能系统能够以视觉方式感知图形界面并与之交互。然而，仅依靠监督微调无法处理长时程的信用分配、分布偏移以及不可逆环境中的安全探索问题，这使得强化学习（RL）成为推动自动化发展的核心方法论。在本工作中，我们首次对强化学习与GUI智能体的交叉领域进行了全面综述，并探讨了这一研究方向如何向“数字居民”演进。我们提出了一个系统性的分类法，将现有方法组织为离线强化学习、在线强化学习和混合策略三大类，并辅以对奖励工程、数据效率和关键技术创新的深入分析。我们的分析揭示了若干新兴趋势：可靠性与可扩展性之间的张力正在推动复合式、多层次方案的应用。

    arXiv:2604.27955v2 Announce Type: replace  Abstract: Graphical User Interface (GUI) agents have emerged as a promising paradigm for intelligent systems that perceive and interact with graphical interfaces visually. Yet supervised fine-tuning alone cannot handle long-horizon credit assignment, distribution shifts, and safe exploration in irreversible environments, making Reinforcement Learning (RL) a central methodology for advancing automation. In this work, we present the first comprehensive overview of the intersection between RL and GUI agents, and examine how this research direction may evolve toward digital inhabitants. We propose a principled taxonomy that organizes existing methods into Offline RL, Online RL, and Hybrid Strategies, and complement it with analyses of reward engineering, data efficiency, and key technical innovations. Our analysis reveals several emerging trends: the tension between reliability and scalability is motivating the adoption of composite, multi-tier re
    
[^315]: ReMoA：多模态扩散模型中基于稀疏潜在精化的递归模块化适配

    ReMoA: Recursive Modular Adaptation via Sparse Latent Refinement in Multimodal Diffusion Models

    [https://arxiv.org/abs/2604.25299](https://arxiv.org/abs/2604.25299)

    提出递归模块化适配框架ReMoA，通过在扩散模型联合注意力层中进行多步递归迭代精炼视觉token，并利用稀疏模块选择实现高效参数共享，从而提升多模态文本到图像生成中的结构化推理与文本遵循能力。

    

    扩散模型在高保真数据合成方面已取得成功，但其在遵循文本等更复杂的结构化推理方面的能力仍然受限。尽管语言模型领域的进展利用了潜在推理和递归等策略来增强文本理解能力，但由于视觉token具有连续、非离散的特性，将这些策略扩展到多模态文本到图像生成任务中面临挑战。为解决这一问题，我们从人类模块化认知中汲取灵感，提出了递归模块化适配框架，该框架被集成到传统扩散模型的联合注意力层中，通过多次递归步骤迭代地精炼视觉token，同时借助神经模块的稀疏选择实现高效的参数共享。在类条件ImageNet生成任务以及使用GenEval和DPG基准的文本到图像生成任务上的全面评估……（原文摘要在此处截断）

    arXiv:2604.25299v2 Announce Type: replace-cross  Abstract: Diffusion models have achieved success in high-fidelity data synthesis, yet their capacity for more complex, structured reasoning such as text following remains constrained. While advances in language models have leveraged strategies such as latent reasoning and recursion to enhance text understanding capabilities, extending these to multimodal text-to-image generation tasks is challenging due to the continuous and non-discrete nature of visual tokens. To tackle this problem, we draw inspiration from modular human cognition and propose a Recursive Modular Adaptation (ReMoA) framework integrated into joint attention layers of conventional diffusion models, iteratively refining visual tokens over multiple recursion steps while efficiently sharing parameters via sparse selection of neural modules. Comprehensive evaluations on both class-conditioned ImageNet generation and text-to-image generation using the GenEval and DPG benchmar
    
[^316]: AdaMamba：用于长期时间序列预测的自适应频率门控Mamba模型

    AdaMamba: Adaptive Frequency-Gated Mamba for Long-Term Time Series Forecasting

    [https://arxiv.org/abs/2604.23239](https://arxiv.org/abs/2604.23239)

    提出AdaMamba，一种自适应频率门控的Mamba模型，通过自适应地处理频率分量来解决时频异质性问题，从而提升长期时间序列预测的准确性。

    

    时间序列预测是广泛的网络级服务的基础，涵盖个性化推荐、能源调度和基础设施容量规划。准确的长期时间序列预测（LTSF）要求模型能够捕捉长程时间依赖性，同时适应跨变量变化的周期性模式。尽管频率感知方法提供了时间动态的全局视角，但它们通常依赖于不灵活的启发式设计，例如预定义的周期长度、手动选择或随机采样的top-k频率分量，以及分离的频谱变换。因此，它们无法考虑到时频异质性问题，即在时间域中看似同步的变量在频率域中可能存在显著差异。与此同时，基于Mamba的预测模型尽管在长程建模方面具有高效性，但仅将频率信息作为辅助信号引入，留下……

    arXiv:2604.23239v2 Announce Type: replace  Abstract: Time-series forecasting is foundational to a wide spectrum of web-scale services, spanning personalized recommendations, energy scheduling, and infrastructure capacity planning. Accurate long-term time series forecasting (LTSF) requires models to capture long-range temporal dependencies while adapting to periodic patterns that vary across variables. Although frequency-aware methods offer a global view of temporal dynamics, they typically rely on inflexible heuristic designs, such as predefined period lengths, manually selected or randomly sampled top-k frequency components, and separate spectral transformations. Consequently, they fail to account for time-frequency heterogeneity, where variables that appear synchronized in the time domain can differ substantially in the frequency domain. Meanwhile, Mamba-based forecasters, despite their efficiency in long-range modeling, introduce frequency information only as auxiliary signals, leav
    
[^317]: AromaGen：基于多模态语言模型的丰富嗅觉体验交互式生成

    AromaGen: Interactive Generation of Rich Olfactory Experiences with Multimodal Language Models

    [https://arxiv.org/abs/2604.01650](https://arxiv.org/abs/2604.01650)

    AromaGen是一个由多模态语言模型驱动的12通道可穿戴嗅觉系统，能够将自然语言描述转化为可感知的气味组合并支持迭代优化，用户实验表明其检索增强的AI气味组合效果与人类专家相当。

    

    嗅觉与食物、记忆以及社会体验之间的联系，促使研究人员将嗅觉引入交互式系统中。多模态人工智能为从自然语言生成气味开辟了新的可能性，然而目前尚不清楚预训练模型是否编码了足够的嗅觉知识，从而能够将语言转化为可感知的气味组合。我们提出了AromaGen，这是一个由人工智能驱动的12通道可穿戴嗅觉系统，它将自由格式的自然语言描述映射到12种基础气味剂上，这些基础气味剂经过精心挑选以覆盖语义推导出的嗅觉空间，同时支持通过自然语言反馈进行迭代优化。在一项涵盖50种真实世界基准气味的被试间研究（N=60）中，参与者在三选一强迫选择任务中区分AromaGen生成的目标气味的表现显著高于随机水平，其中基于检索增强的AI气味组合的表现与人类专家组合以及零样本AI组合相当。

    arXiv:2604.01650v4 Announce Type: replace-cross  Abstract: Smell's connections with food, memory, and social experience have motivated researchers to bring olfaction into interactive systems. Multimodal AI opens new possibilities for generating smells from natural language, yet it remains unclear whether pretrained models encode sufficient olfactory knowledge to translate language into perceptible compositions. We present AromaGen, an AI-powered 12-channel wearable olfactory system that maps free-form natural-language descriptions to 12 base odorants selected to cover a semantically derived olfactory space, while supporting iterative refinement through natural-language feedback. In a between-subjects study (N=60) across 50 real-world benchmark smells, participants distinguished AromaGen-generated target smells above chance in a three-alternative forced-choice task, with retrieval-augmented AI composition performing comparably to human-expert composition and zero-shot AI composition. Na
    
[^318]: 解释的正确性重要吗？将计算化XAI评估与人类理解联系起来

    Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding

    [https://arxiv.org/abs/2603.25251](https://arxiv.org/abs/2603.25251)

    该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。

    

    解释性人工智能（XAI）方法通常使用功能正确性指标（也称为忠实度或保真度）进行评估，这些指标衡量解释在多大程度上反映模型的真实推理过程。人们通常假设正确性越高，人类理解效果就越好，但这一联系尚未在受控的正确性水平下得到检验。我们开展了一项用户研究（N=200），在一个合成时间序列分类任务中，将解释正确性操纵在四个水平（100%、85%、70%、55%），参与者无法依赖领域知识或视觉直觉进行判断。解释被构建为类似特征归因的形式，其正确性基于已知的真实标准定义，而非从训练好的模型中估计得出。参与者需要根据这些解释预测模拟AI的决策（前向模拟），我们以其前向模拟准确率作为理解程度的代理指标。正确性影响了前向模拟准确率

    arXiv:2603.25251v3 Announce Type: replace-cross  Abstract: Explainable AI (XAI) methods are commonly evaluated using functional correctness metrics, also called faithfulness or fidelity, which estimate how closely an explanation reflects the model's reasoning. Higher correctness is assumed to produce better human understanding, but this link has not been tested with controlled levels. We conducted a user study (N=200) that manipulated explanation correctness at four levels (100%, 85%, 70%, 55%) in a synthetic time series classification task where participants could not rely on domain knowledge or visual intuition. Explanations were constructed to resemble feature attributions, with correctness defined against a known ground truth rather than estimated from a trained model. Participants predicted a simulated AI's decisions from these explanations (forward simulation), and we used their forward simulation accuracy as a proxy for understanding. Correctness affected forward simulation accu
    
[^319]: MemReward：基于图的经验记忆用于有限标签下的LLM奖励预测

    MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels

    [https://arxiv.org/abs/2603.19310](https://arxiv.org/abs/2603.19310)

    提出MemReward，将查询、思考过程和答案作为异构图节点构建经验记忆，利用GNN将有限的奖励标签传播到无标签轨迹，从而在RL微调中充分利用全部采样轨迹优化LLM策略。

    

    强化学习（RL）微调通过在采样轨迹上利用奖励信号来提升大型语言模型（LLM）的推理能力。当只有部分轨迹拥有奖励标签时，将策略更新限制在该子集上会导致其余轨迹无法用于策略优化。半监督学习可以将标签从有标签样本传播到无标签样本，但现有方法要么使用固定的相似度权重来扩散标签，要么将每个样本独立处理，忽略了查询、思考过程与答案之间的关联。我们提出MemReward，一种基于图的经验记忆方法，可在在线策略优化过程中将奖励传播到无标签轨迹。MemReward将初始LLM策略的查询及其轨迹（思考过程与最终答案）存储为异构图中的节点，并通过相似性边和结构边进行连接，同时在有标签轨迹上预热训练一个GNN，以在图上传播奖励……（摘要在此处截断）

    arXiv:2603.19310v5 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) fine-tuning improves large language model (LLM) reasoning using reward signals on sampled rollouts. When reward labels are available for only a subset of rollouts, restricting policy updates to that subset leaves the remaining rollouts unused for policy optimization. Semi-supervised learning propagates labels from labeled to unlabeled samples, but existing methods either diffuse labels with fixed similarity weights or treat each sample independently, ignoring the relations among queries, thinking processes, and answers. We propose MemReward, a graph-based experience memory that propagates rewards to unlabeled rollouts during online policy optimization. MemReward stores queries and the rollouts (thinking processes and final answers) of an initial LLM policy as nodes of a heterogeneous graph connected by similarity and structural edges, and warms up a GNN on labeled rollouts to propagate rewards over t
    
[^320]: AS²——基于注意力的软答案集：一种端到端可微分的神经-软-符号推理架构

    AS2 -- Attention-Based Soft Answer Sets: An End-to-End Differentiable Neuro-Soft-Symbolic Reasoning Architecture

    [https://arxiv.org/abs/2603.18436](https://arxiv.org/abs/2603.18436)

    AS²提出了一种完全可微分的神经符号架构，通过连续近似ASP不动点算子取代离散求解器，实现了无需外部求解器的端到端推理学习，并在视觉数独等任务上达到近乎完美的准确率。

    

    神经符号系统通常将神经感知与离散符号求解器相结合，这限制了端到端优化。我们提出了基于注意力的软答案集（AS²），这是一种完全可微分的神经符号架构，它用答案集编程（ASP）立即后果算子 $T_P$ 的连续近似取代了离散求解。AS² 在有限符号域上维护概率分布，并通过最小化概率化 $T_P$ 的不动点残差来进行学习，从而在训练和推理过程中无需外部求解器即可对约束检查进行梯度传播。对于空间约束满足任务，AS² 用直接从声明式 ASP 规范导出的约束组归属嵌入取代位置嵌入，使模型摆脱了对任意位置索引的依赖。在视觉数独任务上，AS² 实现了 99.89% 的原始单元格准确率和 100% 的约束满足率。

    arXiv:2603.18436v2 Announce Type: replace  Abstract: Neuro-symbolic systems typically combine neural perception with discrete symbolic solvers, limiting end-to-end optimization. We introduce Attention-Based Soft Answer Sets (AS$^2$), a fully differentiable neuro-symbolic architecture that replaces discrete solving with a continuous approximation of the Answer Set Programming (ASP) immediate-consequence operator $T_P$. AS$^2$ maintains probability distributions over finite symbol domains and learns by minimizing the fixed-point residual of a probabilistic version of $T_P$, enabling differentiation through constraint checking without external solvers during training or inference. For spatial constraint-satisfaction tasks, AS$^2$ replaces positional embeddings with constraint-group membership embeddings derived directly from declarative ASP specifications, making the model independent of arbitrary position indexing. On Visual Sudoku, AS$^2$ achieves 99.89% raw cell accuracy and 100% const
    
[^321]: TherapyGym：评估并对齐治疗聊天机器人的临床保真度与安全性

    TherapyGym: Evaluating and Aligning Clinical Fidelity and Safety in Therapy Chatbots

    [https://arxiv.org/abs/2603.18008](https://arxiv.org/abs/2603.18008)

    该论文提出了THERAPYGYM框架，通过基于CTRS量表的自动化评分衡量临床保真度、通过多标签标注衡量治疗安全性来评估治疗聊天机器人，并发布含116段对话和1270条专家评分的THERAPYJUDGEBENCH验证集，用于对照执业临床医生校准LLM评判者。

    

    arXiv:2603.18008v2 公告类型：replace-cross 摘要：大语言模型（LLM）正越来越多地被用于心理健康支持；然而，现有的主流评估方法——流畅度指标、偏好测试以及通用对话基准——无法捕捉心理治疗中临床上的关键维度。我们提出了THERAPYGYM，一个围绕两大临床支柱——保真度与安全性——来评估和改进治疗聊天机器人的框架。保真度通过认知治疗评分量表来衡量，该量表被实现为一个自动化流水线，在多轮对话会话中对认知行为疗法（CBT）技术的依从性进行评分。安全性则通过多标签标注方案进行评估，涵盖治疗特有的风险（例如未能妥善处理自伤或虐待问题）。为了减轻基于LLM的评判者存在的偏差与不可靠性，我们进一步发布了THERAPYJUDGEBENCH，一个包含116段对话和1270条专家评分的验证集，可用于对照执业临床医生对LLM评判者进行审计与校准。THERAPYGYM还可以作为训练框架来使用。

    arXiv:2603.18008v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used for mental-health support; yet prevailing evaluation methods--fluency metrics, preference tests, and generic dialogue benchmarks--fail to capture the clinically critical dimensions of psychotherapy. We introduce THERAPYGYM, a framework that evaluates and improves therapy chatbots along two clinical pillars: fidelity and safety. Fidelity is measured using the Cognitive Therapy Rating Scale (CTRS), implemented as an automated pipeline that scores adherence to CBT techniques over multi-turn sessions. Safety is assessed using a multi-label annotation scheme, covering therapy-specific risks (e.g., failing to address harm or abuse). To mitigate bias and unreliability in LLM-based judges, we further release THERAPYJUDGEBENCH, a validation set of 116 dialogues with 1,270 expert ratings for auditing and calibration against licensed clinicians. THERAPYGYM also serves as a training harnes
    
[^322]: 流形匹配自编码器：将数据嵌入到选定的几何结构中

    Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures

    [https://arxiv.org/abs/2603.16568](https://arxiv.org/abs/2603.16568)

    提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。

    

    操纵自编码器潜空间中数据的结构是表示学习领域的一个长期目标。我们提出了一种简单的正则化方法——流形匹配，它能够在潜空间上施加显式的几何结构。其核心思想是选取数据的一个参考表示R（它不必是输入网络的原始输入X），然后按批次最小化R中的成对距离与潜空间Z中的成对距离之间的均方误差。尽管自编码器仍然学习重建X，但其潜空间的形状会逐渐模仿R的形状。该目标不需要标签，只需要参考距离；而且由于这些距离是标量，R的维数与Z的维数相互解耦：任何参考都可以引导任何瓶颈层。利用这一单一工具，我们研究了如何在潜空间上施加不同的几何结构，并与基于几何或t…（方法）构建的自编码器进行了比较。

    arXiv:2603.16568v2 Announce Type: replace-cross  Abstract: Manipulating the structure of data in the latent space of an autoencoder is a long-standing goal of representation learning. We propose a simple regularization, Manifold Matching, that imposes an explicit geometric structure on that latent space. The idea is to take a reference representation R of the data, which need not be the raw input X fed to the network, and to minimize, batch-wise, the mean squared error between the pairwise distances in R and those in the latent space Z. Although the autoencoder still learns to reconstruct X, the shape of its latent space comes to mimic that of R. The objective requires no labels, only reference distances, and because those are scalars, the dimensionality of R is decoupled from that of Z: any reference can guide any bottleneck. Using this single instrument we study how different geometric structures can be imposed on a latent space, comparing against autoencoders built on geometric or t
    
[^323]: PREBA：基于PCA加权检索增强大语言模型与贝叶斯平均聚合的手术时长预测

    PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation

    [https://arxiv.org/abs/2603.13275](https://arxiv.org/abs/2603.13275)

    PREBA提出了一种融合PCA加权检索与贝叶斯平均聚合的检索增强框架，无需任务特定微调即可将大语言模型的推理锚定于机构特定的临床证据上，从而实现准确且稳定的手术时长预测。

    

    手术时长的准确预测对医院资源管理至关重要。尽管近年来从机器学习（ML）到微调大语言模型（LLM）等监督学习方法已展现出强大的性能，但它们仍受限于对高质量标注数据和计算密集型训练的依赖。相比之下，零样本LLM推理提供了一种无需训练的有前景的替代方案，但其缺乏对特定机构临床证据的支撑，导致预测结果与临床实际脱节且容易不稳定。为解决这些局限性，我们提出了PREBA，这是一个检索增强框架，它集成了PCA加权检索和贝叶斯平均聚合，将LLM的推理建立在特定机构的临床证据之上，而无需任务特定的微调。具体而言，PREBA首先将异构的临床特征编码为统一的表示形式，然后执行PCA加权检索……（原文摘要在此处截断）

    arXiv:2603.13275v4 Announce Type: replace-cross  Abstract: Accurate prediction of surgical duration is pivotal for hospital resource management. Although recent supervised learning approaches, from machine learning (ML) to fine-tuned large language models (LLMs), have shown strong performance, they remain constrained by the need for high-quality labeled data and computationally intensive training. In contrast, zero-shot LLM inference offers a promising training-free alternative, but it lacks grounding in institution-specific clinical evidence, making its predictions clinically misaligned and prone to instability. To address these limitations, we present PREBA, a retrieval-augmented framework that integrates PCA-weighted retrieval and Bayesian averaging to ground LLM reasoning in institution-specific evidence without requiring task-specific fine-tuning. Specifically, PREBA first encodes heterogeneous clinical features into a unified representation. It then performs PCA-weighted retrieva
    
[^324]: 物中之脸，无中之脸：利用空想性错视探究视觉模型的人脸选择性

    Faces in Things, Faces in Nothing: Probing Face Selectivity in Vision Models with Pareidolia

    [https://arxiv.org/abs/2603.03989](https://arxiv.org/abs/2603.03989)

    该论文以人脸空想性错视为探针，利用 FacesInThings 数据集和三种同照片对照条件测试六个视觉模型，发现模型虽然会对虚假人脸作出响应，但其判断机制与人类并不相同。

    

    一个电源插座即使没有描绘任何人，也可能看起来像一张脸。因此，人脸空想性错视（face pareidolia）是探究视觉模型如何将模糊视觉证据转化为类别判断的一个有效探针。当模型将这类物体判定为“Human（人类）”时，这一判断是否依赖于其类脸的排列方式？我们利用 FacesInThings 数据集测试了六个视觉模型——从纯视觉的 ViT 到 CLIP 以及生成式视觉-语言模型，并以人脸检测器作为参照。每个类脸区域都与来自同一照片的三种缺乏该排列的对照条件配对：背景裁剪区域、将图块打乱后的区域，以及该区域的平均颜色。如果“Human”的回答是由排列方式驱动的，那么在这些对照条件上该回答出现的频率应当更低。我们还改变了类别的描述方式。研究得出了三项发现。首先，模型会对空想出的脸作出响应，但其方式与人类不同：人脸检测器很少在这些区域返回人脸结果，然而 CLIP ViT-L/14 和两个生成式（摘要在此处不完整，原文被截断）

    arXiv:2603.03989v4 Announce Type: replace-cross  Abstract: An electrical outlet can look like a face without depicting a person. Face pareidolia is thus a revealing probe of how vision models turn ambiguous visual evidence into category judgments. When a model calls such an object Human, does that answer depend on its face-like arrangement? Using the FacesInThings dataset, we test six vision models, from a vision-only ViT to CLIP and generative vision-language models, with face detectors as a reference. Each face-like region is paired with three controls from the same photograph that lack the arrangement: a background crop, the region with its tiles shuffled, and its average color. A Human answer driven by the arrangement should appear less often on these controls. We also vary how the categories are described. Three findings emerge. First, models respond to illusory faces, but not as people do: face detectors rarely return a face for these regions, yet CLIP ViT-L/14 and two generative
    
[^325]: 偏移越远，表征越稀疏：分析大语言模型中的分布外（OOD）机制

    Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs

    [https://arxiv.org/abs/2603.03415](https://arxiv.org/abs/2603.03415)

    本文揭示了LLM内部表征的稀疏度随任务难度增加而系统性增大——任务越难，最后一层隐藏状态的表征越稀疏，且该规律跨模型、跨领域普遍成立，表明表征密度是模型对数据熟悉度的习得属性。

    

    在本工作中，我们研究了随着输入难度的增加，大语言模型（LLM）的内部表征如何发生变化。尽管在单个样本层面稀疏度的变化可能并不明显，但在数据集层面的结果揭示了一个一致的统计趋势：随着任务难度的提升（例如更难的问题、更长的上下文或更多的答案选项），LLM最后一层隐藏状态会系统性地变得更加稀疏。简而言之，任务越难，表征越稀疏。这种稀疏度与难度的关系在多种模型和领域中均可观察到，表明更难的输入会促使最后一层隐藏状态中的激活模式更加集中。通过一系列受控分析和学习动态实验，我们证明表征密度是模型对数据熟悉度的一种习得属性：模型会为已掌握的模式发展出丰富且分布式的表征，而面对陌生的输入时则表现出……

    arXiv:2603.03415v4 Announce Type: replace-cross  Abstract: In this work, we investigate how the internal representations of Large Language Models (LLMs) change with inputs of increasing difficulty. Although sparsity changes may not be obvious for individual samples, dataset-level results reveal a consistent statistical trend: as task difficulty increases (e.g., harder questions, longer contexts, or more answer choices), the last hidden states of LLMs become systematically sparser. In short, \textbf{the harder the task, the sparser the representation}. This sparsity--difficulty relation is observable across diverse models and domains, suggesting that harder inputs drive more concentrated activation patterns in the last hidden state. Through a series of controlled analyses and learning-dynamics experiments, we show that representational density is a learned property of data familiarity: models develop rich, distributed representations for mastered patterns, while unfamiliar inputs defaul
    
[^326]: 认证电路：机制电路的稳定性保证

    Certified Circuits: Stability Guarantees for Mechanistic Circuits

    [https://arxiv.org/abs/2602.22968](https://arxiv.org/abs/2602.22968)

    提出了“认证电路”框架，通过随机化数据子采样包装任意黑盒电路发现算法，为电路组件的纳入决策提供可证明的稳定性保证，使其对概念数据集的有界编辑距离扰动保持不变，并对不稳定组件予以弃权。

    

    理解神经网络如何得出其预测对于调试、审计和部署至关重要。机制可解释性通过识别“电路”——即负责特定行为的最小子网络——来追求这一目标。然而，现有的电路发现方法十分脆弱：电路高度依赖于所选的概念数据集，且往往无法在分布外迁移，这引发了人们的质疑：它们究竟捕捉到了概念本身，还是仅仅捕捉到了数据集特有的伪影。我们提出了认证电路，为电路发现提供可证明的稳定性保证。我们的框架通过随机化数据子采样来包装任意黑盒发现算法，以证明对电路组件（取决于基础算法，可以是神经元或模型图的边）的纳入决策在概念数据集的有界编辑距离扰动下保持不变。对于不稳定的组件将予以弃权，从而产生……（原文摘要在此处截断）

    arXiv:2602.22968v4 Announce Type: replace  Abstract: Understanding how neural networks arrive at their predictions is essential for debugging, auditing, and deployment. Mechanistic interpretability pursues this goal by identifying circuits--minimal subnetworks responsible for specific behaviors. However, existing circuit discovery methods are brittle: circuits depend strongly on the chosen concept dataset and often fail to transfer out-of-distribution, raising doubts whether they capture the concept or merely dataset-specific artifacts. We introduce Certified Circuits, which provide provable stability guarantees for circuit discovery. Our framework wraps any black-box discovery algorithm with randomized data subsampling to certify that inclusion decisions over circuit components--neurons or edges of the model graph, depending on the base algorithm--are invariant to bounded edit-distance perturbations of the concept dataset. Unstable components are abstained from, yielding circuits that
    
[^327]: 相同的话语，不同的评判：偏好如何随模态而变化

    Same Words, Different Judgments: How Preferences Vary Across Modalities

    [https://arxiv.org/abs/2602.22710](https://arxiv.org/abs/2602.22710)

    该研究首次通过基于ICC的受控跨模态实验证明，人类在音频与文本两种模态下表达偏好的方式存在显著差异（跨模态一致性接近随机水平），因此原本为文本设计的偏好评估协议不能直接套用于语音模态。

    

    基于偏好的强化学习（PbRL）是将AI系统与人类偏好对齐的主流框架。然而，针对此类偏好数据的评估协议是为文本设计的，尚未在语音模态上得到验证。我们开展了首个基于组内相关系数（ICC）的、受控的跨模态研究，对人类与合成的偏好标注进行分析，在100个提示上比较了对相同语义内容的文本评估与音频评估。我们表明，在任一模态内部达到“良好”（音频）至“接近良好”（文本）的一致性水平（ICC(2,k) 音频约为 .81，文本约为 .74）需要约9名评分者；并且两种模态在人们报告偏好的方式上存在显著差异：音频评分者表现出更窄的决策阈值、更弱的长度偏差以及更偏向用户的评估标准，而跨模态的一致性接近随机水平。我们证明合成评分可以被用来有效预测……（原文摘要在此处截断）

    arXiv:2602.22710v4 Announce Type: replace-cross  Abstract: Preference-based reinforcement learning (PbRL) is the dominant framework for aligning AI systems to human preferences. However, evaluation protocols for such data were designed for text and have not been validated for speech. We present the first intraclass correlation coefficient (ICC)-based, controlled cross-modal study of human and synthetic preference annotations, comparing text and audio evaluations of identical semantic content across 100 prompts. We show that achieving \textit{good} (audio) to \textit{near good} (text) agreement within either modality (ICC(2,$k$) $\approx$ .81 audio, .74 text) requires $\sim$9 raters and modalities show marked differences in how people report preferences: audio raters exhibit narrower decision thresholds, reduced length bias, and more user-oriented evaluation criteria, with near-chance cross-modality agreement. We demonstrate that synthetic ratings can be used to effectively predict inte
    
[^328]: 黄金层及其寻找之道：通过层梯度分析改进大语言模型的知识编辑

    Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis

    [https://arxiv.org/abs/2602.20207](https://arxiv.org/abs/2602.20207)

    本研究通过层梯度分析，证明了大语言模型中存在固定的“黄金层”，其知识编辑性能接近逐样本最优层，且这些黄金层可通过代理数据集可靠识别并能跨数据集泛化到未见查询，从而改进了知识编辑方法。

    

    大语言模型（LLMs）中的知识编辑旨在将模型对特定查询的预测更新为期望的目标，同时保持模型对所有其他输入的原有行为。这一过程通常包含两个阶段：确定需要编辑的层，以及执行参数更新。直觉上，不同的查询可能会将知识定位于模型的不同深度，因此固定的编辑层对不同样本的编辑性能会有所差异。在本工作中，我们假设存在固定的“黄金层”，能够达到与逐样本最优层相近的近最优编辑性能。为验证这一假设，我们通过将黄金层与真实的逐样本最优层进行对比，提供了实证证据。此外，我们证明黄金层可以通过代理数据集被可靠地识别出来，并且能够有效地泛化到不同数据集中未见过的测试集查询上。最后，我们提出了一种新颖的方法（摘要在此处截断）。

    arXiv:2602.20207v4 Announce Type: replace-cross  Abstract: Knowledge editing in Large Language Models (LLMs) aims to update the model's prediction for a specific query to a desired target while preserving its behavior on all other inputs. This process typically involves two stages: identifying the layer to edit and performing the parameter update. Intuitively, different queries may localize knowledge at different depths of the model, resulting in different sample-wise editing performance for a fixed editing layer. In this work, we hypothesize the existence of fixed golden layers that can achieve near-optimal editing performance similar to sample-wise optimal layers. To validate this hypothesis, we provide empirical evidence by comparing golden layers against ground-truth sample-wise optimal layers. Furthermore, we show that golden layers can be reliably identified using a proxy dataset and generalize effectively to unseen test set queries across datasets. Finally, we propose a novel me
    
[^329]: 解缠以创造：面向少样本中文字体生成的架构级解缠扩散模型

    Disentangle to create: Architecture-level disentangled diffusion model for few-shot Chinese font generation

    [https://arxiv.org/abs/2602.18874](https://arxiv.org/abs/2602.18874)

    提出架构级解缠扩散模型ALD-Font，通过非对称设计将内容模板与风格调制信号在架构层面彻底分离，显著提升了少样本中文字体生成中的内容准确性与风格保真度。

    

    少样本中文字体生成旨在仅使用少量参考图像合成目标风格的新字符。实现准确的内容渲染和忠实的风格迁移需要在内容与风格之间进行有效的解缠。然而，大多数现有方法仅实现了特征级的解缠，并允许生成器重新纠缠这些特征，从而导致内容失真和风格保真度下降。为解决这一问题，我们提出了具有非对称设计的架构级解缠扩散模型。U-Net骨干网络以宋体风格的字形作为内容模板，与噪声潜变量拼接以构建空间画布；从目标风格图像中提取的风格特征则作为调制信号注入以引导生成。基于对解缠有效性的理论验证，我们采用参数高效微调（PEFT）策略，仅更新……

    arXiv:2602.18874v2 Announce Type: replace-cross  Abstract: Few-shot Chinese font generation aims to synthesize new characters in a target style using only a handful of reference images. Achieving accurate content rendering and faithful style transfer requires effective disentanglement between content and style. However, most existing approaches achieve only feature-level disentanglement and allow the generator to re-entangle these features, leading to content distortion and degraded style fidelity. To address this, we propose the Architecture-Level Disentangled Diffusion Model (ALD-Font) with an asymmetric design. The U-Net backbone takes SimSun-style glyphs as content templates, concatenated with noisy latents to construct a spatial canvas. Style features extracted from target-style images are injected as modulation signals to guide generation. Based on theoretical validation of disentanglement effectiveness, we adopt a parameter-efficient fine-tuning (PEFT) strategy that updates only
    
[^330]: 面向大语言模型强化学习的进化式系统提示学习

    Evolutionary System Prompt Learning for Reinforcement Learning in LLMs

    [https://arxiv.org/abs/2602.14697](https://arxiv.org/abs/2602.14697)

    本文提出进化式系统提示学习（E-SPL），在强化学习迭代中联合优化模型权重与系统提示，实现权重中程序性知识与提示中可解释陈述性知识的自然分工，显著提升大语言模型在数学推理任务上的成功率与由易到难的泛化能力。

    

    构建能够从经验中自主自我改进的智能体系统是人工智能长期以来的目标。如今，大语言模型（LLM）主要通过两种机制进行自我改进：通过自我反思进行上下文更新，以及通过强化学习（RL）进行权重更新。在本工作中，我们提出了进化式系统提示学习（E-SPL），这是一种联合改进模型经验上下文及其权重的方法。在每次强化学习迭代中，E-SPL 在多个系统提示下并行采样轨迹，然后利用这些轨迹通过强化学习联合更新权重，并通过自我反思进化系统提示。E-SPL 促进了权重中的程序性知识与系统提示中可解释的陈述性知识之间的自然分工。最值得注意的是，在由易到难的泛化设置中（AIME → BeyondAIME），E-SPL 将强化学习的成功率从 38.8% 提升至 45.1%。E-SPL 还提升了强化学习在 AIME 2025 上的表现（56.3……

    arXiv:2602.14697v4 Announce Type: replace  Abstract: Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving a model's experiential context and its weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then uses them to jointly update the weights with RL and evolve the system prompts with self-reflection. E-SPL encourages a natural division between procedural knowledge in weights and interpretable declarative knowledge in system prompts. Most notably, in an easy-to-hard generalization setting (AIME $\rightarrow$ BeyondAIME), E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1%. E-SPL also improves RL on AIME 2025 (56.3
    
[^331]: 一种用于短期负荷预测的轻量级多视角方法

    A Lightweight Multi-View Approach to Short-Term Load Forecasting

    [https://arxiv.org/abs/2602.09220](https://arxiv.org/abs/2602.09220)

    本文提出一种利用单值嵌入、缩放时间范围输入和嵌入丢弃机制的轻量级多视角方法，以显著更少的参数实现了具有竞争力的短期负荷预测性能。

    

    时间序列预测是能源、金融和气象等领域的一项关键任务，准确的预测能够支持明智的决策。尽管基于Transformer和大参数量的模型最近取得了最先进的结果，但其复杂性可能导致过拟合和不稳定的预测，尤其是当较早的数据点变得不那么相关时。在本文中，我们提出了一种用于短期负荷预测的轻量级多视角方法，该方法利用单值嵌入和缩放的时间范围输入来高效地捕获时间相关特征。我们引入了一种嵌入丢弃机制，以防止对特定特征的过度依赖并增强可解释性。我们的方法以显著更少的参数实现了具有竞争力的性能，在包括噪声或稀疏数据场景在内的多个数据集上展示了鲁棒性，并对各组成部分的贡献提供了深入见解。

    arXiv:2602.09220v2 Announce Type: replace-cross  Abstract: Time series forecasting is a critical task across domains such as energy, finance, and meteorology, where accurate predictions enable informed decision-making. While transformer-based and large-parameter models have recently achieved state-of-the-art results, their complexity can lead to overfitting and unstable forecasts, especially when older data points become less relevant. In this paper, we propose a lightweight multi-view approach to short-term load forecasting that leverages single-value embeddings and a scaled time-range input to capture temporally relevant features efficiently. We introduce an embedding dropout mechanism to prevent over-reliance on specific features and enhance interpretability. Our method achieves competitive performance with significantly fewer parameters, demonstrating robustness across multiple datasets, including scenarios with noisy or sparse data, and provides insights into the contributions of 
    
[^332]: CompilerKV：通过离线经验编译实现风险自适应的KV压缩

    CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation

    [https://arxiv.org/abs/2602.08686](https://arxiv.org/abs/2602.08686)

    论文提出CompilerKV，通过从校准语料库离线编译KV保留校正表，将在线校正开销降至O(1)查找，且编译出的保留表可作为可移植的架构先验，在跨语料库和跨模型迁移中表现出高度一致性。

    

    仅预填充的KV压缩在预填充结束时冻结一个token子集，并从中进行解码，不再进行后续淘汰。因此保留决策是不可逆的，然而现有方法是在线地从单个带噪声的提示中估计其所依赖的校正信号，即每个注意力头的可靠性和提示级别的压缩敏感度。我们认为这是错误的统计单位：这些信号在跨提示之间表现出的规律性远高于提示内的信噪比。我们提出了CompilerKV，这是一种KV保留策略，其校正表是从校准语料库中离线编译得到的，将标准观察窗口扫描之后的在线校正简化为O(1)次查找加上一个预算钳制。我们发现编译得到的保留表表现得如同可移植的架构先验：在四个骨干网络上，排序可以在互不相交的语料库之间迁移（平均Spearman相关系数ρ̄=0.90），且直接的模型到模型表迁移仅需0.4

    arXiv:2602.08686v4 Announce Type: replace-cross  Abstract: Prefill-only KV compression freezes a token subset at the end of prefill and decodes from it without further eviction. The retention decision is therefore irreversible, yet existing methods estimate the corrective signals it relies on, per-head reliability and prompt-level compression sensitivity, online from a single noisy prompt. We argue this is the wrong statistical unit: these signals exhibit far higher cross-prompt regularity than within-prompt signal-to-noise. We introduce \textsc{CompilerKV}, a KV-retention policy whose corrective tables are compiled offline from a calibration corpus, reducing online correction after the standard observation-window scan to $O(1)$ lookups plus a budget clamp. We find that compiled retention tables behave as portable architectural priors: rankings transfer across disjoint corpora on four backbones (mean Spearman $\bar\rho{=}0.90$), and direct model-to-model table transfer costs only $0.4$
    
[^333]: 从结果到表征：通过积分策略梯度追踪推理机制

    From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient

    [https://arxiv.org/abs/2602.02313](https://arxiv.org/abs/2602.02313)

    本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。

    

    大型语言模型（LLMs）展现出卓越的推理能力，然而驱动这些多步推理过程的内部机制仍然不透明。现有的机制可解释性方法通常依赖于表面的文本模式共现，或仅能捕捉短期效应，难以追踪神经元或稀疏特征等内部组件在多步推理中的长程、结果驱动的影响。本文提出了积分策略梯度，这是一种新颖的无需训练的框架，将基于策略的归因方法引入LLM推理的机制可解释性研究。该方法基于“结果导向”和“序贯影响感知”两项原则，将结果层面的不可微分行为信号（如推理结果的正确性）反向传播至整个推理轨迹，并利用路径积分在组件层面分离出与结果相关的中介因素。

    arXiv:2602.02313v3 Announce Type: replace  Abstract: Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, yet the internal mechanisms driving these multi-step processes remain opaque. Existing mechanistic interpretability approaches often rely on superficial text-pattern co-occurrences or capture only short-term effects, struggling to trace the long-horizon, outcome-driven influence of internal components such as neurons or sparse features in multi-step reasoning.In this paper, we introduce Integrated Policy Gradient (IPG), a novel, training-free framework that brings policy-based attribution into mechanistic interpretability for LLM reasoning. Grounded in \textit{outcome-oriented} and \textit{sequential-influence-aware} principles, it backpropagates outcome-level, non-differentiable behavioral signals (e.g., reasoning outcome correctness) through entire inference trajectories, utilizing path integration to isolate outcome-relevant mediators at component level fo
    
[^334]: OpInf-LLM：通过算子推断利用大语言模型求解参数化偏微分方程

    OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference

    [https://arxiv.org/abs/2602.01493](https://arxiv.org/abs/2602.01493)

    提出OpInf-LLM框架，将算子推断与大语言模型结合，仅需少量解数据即可准确求解参数化偏微分方程，并能泛化到未见过的参数与边界条件，缓解了执行成功率与数值精度之间的权衡。

    

    求解多样化的偏微分方程（PDE）是科学与工程领域的基础任务。大语言模型（LLM）在代码生成、符号推理和工具使用方面已展现出强大能力，但在异构环境下可靠地求解PDE仍然具有挑战性。此前基于LLM的代码生成研究以及基于Transformer的PDE学习基础模型已取得可喜进展。然而，执行成功率与数值精度之间始终存在权衡，尤其是在需要对未见过的参数和边界条件进行泛化时。在本工作中，我们提出了OpInf-LLM，一个通过算子推断实现大语言模型参数化PDE求解的框架。该框架利用少量解数据即可对多样化的PDE实例（包括未见过的参数和配置）进行准确预测，并可与LLM无缝集成，实现自然语言交互（原文在此截断）。

    arXiv:2602.01493v3 Announce Type: replace-cross  Abstract: Solving diverse partial differential equations (PDEs) is fundamental in science and engineering. Large language models (LLMs) have demonstrated strong capabilities in code generation, symbolic reasoning, and tool use, but reliably solving PDEs across heterogeneous settings remains challenging. Prior work on LLM-based code generation and transformer-based foundation models for PDE learning has shown promising advances. However, a persistent trade-off between execution success rate and numerical accuracy arises, particularly when generalization to unseen parameters and boundary conditions is required. In this work, we propose OpInf-LLM, an LLM parametric PDE solving framework via operator inference. The proposed framework leverages small amounts of solution data to enable accurate prediction of diverse PDE instances, including unseen parameters and configurations, and provides seamless integration with LLMs for natural language t
    
[^335]: 基于前缀重要性比率稳定离策略大语言模型优化

    Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio

    [https://arxiv.org/abs/2601.22718](https://arxiv.org/abs/2601.22718)

    该论文发现token级别的重要性采样比率无法修正前缀分布偏移，并提出用累积前缀重要性比率对前缀边缘化策略梯度进行精确修正，从而在大策略滞后下稳定大语言模型的离策略强化学习优化。

    

    强化学习（RL）后训练在激发大语言模型（LLM）的推理行为方面日益展现出强大的能力。为了提高训练效率，通常以离策略的方式使用较旧的采样策略生成rollout样本，然后用于更新当前的目标策略。为了纠正由此产生的采样策略与目标策略之间的偏差，大多数现有的RL目标依赖于token级别的重要性采样比率。然而，我们观察到在较大的策略滞后情况下，现有的重要性加权目标存在训练动态不稳定的问题。在本文中，我们重新审视了离策略条件下的LLM策略优化，并证明了对前缀边缘化策略梯度的精确修正应该是累积前缀重要性比率，而token级别的加权并不能显式地纠正这种前缀分布偏移。为了在大规模离策略漂移下稳定LLM优化，我们提出了一种简单的（摘要在此处截断）

    arXiv:2601.22718v2 Announce Type: replace  Abstract: Reinforcement learning (RL) post-training has increasingly demonstrated strong ability to elicit reasoning behaviors in large language models (LLMs). For training efficiency, rollouts are typically generated in an off-policy manner using an older sampling policy and then used to update the current target policy. To correct the resulting discrepancy between the sampling and target policies, most existing RL objectives rely on a token-level importance sampling ratio. However, we observe unstable training dynamics in existing importance-weighted objectives under large policy lag. In this paper, we revisit LLM policy optimization under off-policy conditions and show that the exact correction for the prefix-marginal policy gradient is the cumulative prefix importance ratio. Token-level weighting does not explicitly correct for this prefix distribution shift. To stabilize LLM optimization under large off-policy drift, we propose a simple y
    
[^336]: COL-Trees：空间网络中高效的层次化对象搜索

    COL-Trees: Efficient Hierarchical Object Search in Spatial Networks

    [https://arxiv.org/abs/2601.22183](https://arxiv.org/abs/2601.22183)

    提出了一种新颖的数据结构COL-Tree，通过层次化方法高效解决空间网络中聚合k近邻查询和k最远邻查询等复杂对象搜索问题，弥补了现有基于欧几里得距离启发式方法在空间网络中失效的不足。

    

    许多空间网络应用，例如基于地图的服务，严重依赖于高效的方法来搜索附近相关的兴趣点（POI）。k近邻查询就是这样一个例子，它从代理的位置找到k个最近的兴趣点。虽然现有的空间网络技术大多集中于为单个代理检索附近的兴趣点，但这些搜索启发式方法并不能推广到许多其他应用。例如，聚合k近邻查询需要找到接近多个代理的兴趣点，k最远邻居查询则需要找到与最近邻相对的最远兴趣点。这些问题天然适合采用层次化的方法，但现有方法依赖于基于欧几里得距离的启发式策略，而这类策略在道路网络图等空间网络中有效性会显著下降。为了填补这一空白，我们提出了一种新颖的数据结构——COL-Tree（压缩对象-地标树，Compacted Object-Landmark Tree）。

    arXiv:2601.22183v2 Announce Type: replace-cross  Abstract: Many spatial network applications, such as map-based services, rely heavily on efficient methods to search for relevant nearby points-of-interest (POIs). A $k$ Nearest Neighbor ($k$NN) query is one such example that finds the $k$ closest POIs from an agent's location. While most existing techniques for spatial networks focus on retrieving nearby POIs for a single agent, these search heuristics do not translate to many other applications. For example, Aggregate $k$ Nearest Neighbor (A$k$NN) queries require POIs that are close to multiple agents. $k$ Farthest Neighbors ($k$FN) queries require POIs that are the antithesis of nearest. Such problems naturally benefit from a hierarchical approach, but existing methods rely on Euclidean-based heuristics, which have diminished effectiveness in spatial networks such as road network graphs. We propose a novel data structure, COL-Tree (Compacted Object-Landmark Tree), to address this gap 
    
[^337]: 探究 SAM 3D Body 中的人体测量学保真度

    Investigating Anthropometric Fidelity in SAM 3D Body

    [https://arxiv.org/abs/2601.06035](https://arxiv.org/abs/2601.06035)

    本文发现 SAM 3D Body 难以重建特殊体型人群（如肌肉萎缩、脊柱侧弯、怀孕）的人体测量细节，并将其归因于“感知-失真权衡”与低维参数化 MHR 表示等架构设计带来的系统性偏差，而非模型能力不足。

    

    SAM 3D Body 的发布是人体网格恢复领域的最新进展，展示了其从单张图像生成干净、拓扑连贯网格方面的性能提升。通过利用动量人体骨架，该模型对遮挡和多样化姿态表现出鲁棒性。然而，我们的评估揭示了一个具体且一致的局限性：该模型难以重建细节化的人体测量学偏差，尤其是在表现出显著形态学改变的人群中，例如老年性肌肉萎缩、脊柱侧弯或怀孕，即使这些特征在输入图像中十分明显。在本文中，我们并非将这一现象视为模型能力的失败，而是将其作为“感知-失真权衡”的副产品加以研究。我们认为，架构上对低维参数化 MHR 表示的依赖，结合语义不变的调节机制（DINOv3）以及基于标注的对齐，造……

    arXiv:2601.06035v2 Announce Type: cross  Abstract: The release of SAM 3D Body is a recent development in human mesh recovery, demonstrating improved performance in producing clean, topologically coherent meshes from single images. By leveraging the Momentum Human Rig (MHR), it achieves robustness to occlusion and diverse poses. However, our evaluation reveals a specific and consistent limitation: the model struggles to reconstruct detailed anthropometric deviations, particularly in populations exhibiting distinctive morphological alterations such as geriatric muscle atrophy, scoliosis, or pregnancy, even when these features are prominent in the input image. In this paper, we investigate this phenomenon not as a failure of the model's capacity, but as a byproduct of the "perception-distortion trade-off". We posit that the architectural reliance on the low-dimensional parametric MHR representation, combined with semantic-invariant conditioning (DINOv3) and annotation-based alignment, cre
    
[^338]: 利用地球基础模型的AlphaEarth嵌入提升水文模型的模拟性能

    Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings

    [https://arxiv.org/abs/2601.01558](https://arxiv.org/abs/2601.01558)

    本研究将AlphaEarth嵌入作为静态流域表征用于无观测流域的径流预测，发现其样本外泛化鲁棒性优于传统CAMELS流域属性。

    

    无观测流域的准确预测（PUB）需要在水文条件各异的区域之间进行有效迁移。尽管深度学习推动了大样本水文建模的发展，但其跨区域泛化能力在一定程度上依赖于用于引导模型行为的静态流域描述符。地球基础模型（EFMs）通过从多模态卫星数据中学习环境表征，提供了一种有前景的替代方案。在本研究中，我们评估了64维的AlphaEarth Foundations（AEF）嵌入作为一种新的静态流域表征形式，并基于CAMELS数据集，在样本内（IS）和样本外（OOS）两种设置下评估了其对径流预测的价值。结果表明，尽管AEF嵌入的样本内性能与CAMELS属性几乎相当，但其样本外鲁棒性更高。互信息分析揭示了AEF嵌入与传统流域属性之间的统计关联……

    arXiv:2601.01558v3 Announce Type: replace-cross  Abstract: Accurate prediction in ungauged basins (PUB) requires effective transfer across hydrologically diverse regions. While deep learning has advanced large-sample hydrological modeling, its cross-regional generalization depends in part on the static basin descriptors used to guide model behavior. Earth Foundation Models (EFMs) provide a promising alternative by learning environmental representations from multi-modal satellite data. In this study, we evaluate the 64-dimensional AlphaEarth Foundations (AEF) embeddings as a new form of static basin representation and assess their value for streamflow prediction under both in-sample (IS) and out-of-sample (OOS) settings using the CAMELS dataset. Results show that AEF embeddings offer higher OOS robustness than CAMELS attributes, despite nearly identical IS performance. Mutual-information analysis reveals statistical associations between AEF embeddings and conventional catchment attribut
    
[^339]: PointRAFT：基于部分点云高通量预测马铃薯块茎重量的3D深度学习方法

    PointRAFT: 3D deep learning for high-throughput prediction of potato tuber weight from partial point clouds

    [https://arxiv.org/abs/2512.24193](https://arxiv.org/abs/2512.24193)

    本文提出PointRAFT，一种融合块茎高度先验几何知识的高通量点云回归网络，能够从不完整的点云中准确预测马铃薯块茎重量。

    

    关键表型性状，如作物大小、体积和重量，可以通过点云进行无损估计。然而在实际应用中，由于自遮挡效应，作物的点云通常是不完整的，这会导致估计这些性状时产生系统性误差。为了克服这一限制，可以使用基于深度学习的回归网络直接从不完整的点云中推断作物性状。然而，这些网络的表现可能欠佳，因为其预测并未融入作物的先验几何知识。这类几何知识在结构化环境中通常是可获得的，例如在马铃薯收获机上，相机与传送带之间的固定距离使得块茎高度可以从点云中推导出来。为了利用这一几何线索，我们提出了PointRAFT，一种通过专门的高度嵌入模块来融合物体高度信息的高通量点云回归网络。

    arXiv:2512.24193v2 Announce Type: replace-cross  Abstract: Key phenotypic traits, such as crop size, volume, and weight, can be estimated non-destructively from point clouds. In practice, however, point clouds of crops are typically incomplete due to self-occlusion, leading to systematic errors when estimating these traits. To overcome this limitation, crop traits can be inferred directly from incomplete point clouds using deep learning-based regression networks. These networks, however, may perform suboptimally, because their predictions do not incorporate prior geometric knowledge of the crop. Such geometric knowledge is often available in structured settings, such as on a potato harvester, where the fixed distance between the camera and the conveyor belt allows tuber height to be derived from the point cloud. To exploit this geometric cue, we introduce PointRAFT, a high-throughput point cloud regression network that incorporates object height through a dedicated height embedding. We
    
[^340]: AEGIS：利用对抗性扰动保护3D面部虚拟形象的隐私

    AEGIS: Preserving Privacy of 3D Facial Avatars with Adversarial Perturbations

    [https://arxiv.org/abs/2511.17747](https://arxiv.org/abs/2511.17747)

    AEGIS是首个针对3D高斯虚拟形象的隐私保护身份掩蔽框架，通过对高斯基元颜色系数施加对抗性扰动，在保持虚拟形象真实感和动画可用性的同时，实现跨视角一致的人脸识别身份保护。

    

    随着照片级逼真3D面部虚拟形象的日益普及，尤其是采用高效3D高斯泼溅（3D Gaussian Splatting）表示的虚拟形象，带来了新的网络身份盗用风险，特别是在依赖生物特征认证的系统中。虽然针对2D图像已开发出有效的对抗性掩蔽方法，但在为动态3D虚拟形象实现鲁棒的、视角一致的身份保护方面仍存在显著空白。为解决这一问题，我们提出了AEGIS，这是首个面向3D高斯虚拟形象的隐私保护身份掩蔽框架，能够在保持虚拟形象感知真实感和动画可用性的同时，使其身份对自动人脸识别系统隐藏。AEGIS在预训练人脸验证网络集合的引导下，对高斯基元的基色（DC）系数施加对抗性扰动，将保护直接嵌入到3D表示中，从而使保护在各视角下均能生效。

    arXiv:2511.17747v2 Announce Type: replace-cross  Abstract: The growing adoption of photorealistic 3D facial avatars, particularly those utilizing efficient 3D Gaussian Splatting representations, introduces new risks of online identity theft, especially in systems that rely on biometric authentication. While effective adversarial masking methods have been developed for 2D images, a significant gap remains in achieving robust, viewpoint-consistent identity protection for dynamic 3D avatars. To address this, we present AEGIS, the first privacy-preserving identity masking framework for 3D Gaussian avatars, concealing identity from automated face recognition while preserving the avatar's perceptual realism and animation usability. AEGIS applies adversarial perturbations to the base (DC) color coefficients of the Gaussian primitives, guided by an ensemble of pre-trained face verification networks, embedding the protection directly in the 3D representation so that it applies across viewpoints
    
[^341]: MvBody：基于多视角的混合Transformer网络，利用光学3D人体扫描实现可解释的剖腹产预测

    MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction

    [https://arxiv.org/abs/2511.03212](https://arxiv.org/abs/2511.03212)

    提出了一种基于多视角混合Transformer的MvBody模型，仅利用妊娠31-38周期间的3D光学人体扫描和自我报告医疗数据，即可实现可解释的剖腹产风险预测，适用于医疗资源有限的环境。

    

    准确评估剖腹产（CS）分娩风险至关重要，尤其是在医疗资源有限的环境中，因为这些地区获得医疗保健的机会往往受限。早期可靠的风险预测有助于做出更明智的产前护理决策，并可改善孕产妇和新生儿的结局。然而，现有的大多数预测模型都是为住院分娩期间使用而设计的，依赖于在资源有限或居家环境中通常无法获得的参数。在本研究中，我们开展了一项试点研究，以检验利用3D体型进行剖腹产风险评估的可行性，以便未来能够使用更经济实惠的通用设备。我们提出了一种新颖的基于多视角的Transformer网络MvBody，该网络仅使用自我报告的医疗数据和在妊娠第31周至第38周之间获得的3D光学人体扫描来预测剖腹产风险。为了提高训练效率和模型在数据稀缺情况下的泛化能力（摘要不完整）……

    arXiv:2511.03212v1 Announce Type: cross  Abstract: Accurately assessing the risk of cesarean section (CS) delivery is critical, especially in settings with limited medical resources, where access to healthcare is often restricted. Early and reliable risk prediction allows better-informed prenatal care decisions and can improve maternal and neonatal outcomes. However, most existing predictive models are tailored for in-hospital use during labor and rely on parameters that are often unavailable in resource-limited or home-based settings. In this study, we conduct a pilot investigation to examine the feasibility of using 3D body shape for CS risk assessment for future applications with more affordable general devices. We propose a novel multi-view-based Transformer network, MvBody, which predicts CS risk using only self-reported medical data and 3D optical body scans obtained between the 31st and 38th weeks of gestation. To enhance training efficiency and model generalizability in data-sc
    
[^342]: 基于主动探索的软体机器人动力学学习

    Learning Soft Robotic Dynamics with Active Exploration

    [https://arxiv.org/abs/2510.27428](https://arxiv.org/abs/2510.27428)

    提出了一种不确定性感知的主动探索框架，通过主动收集信息丰富的数据，自主学习任务无关且可泛化的软体机器人动力学模型，在仿真和真实平台上的建模精度与零样本控制性能均显著优于随机探索和特定任务基线。

    

    软体机器人在非结构化环境中具有适应性和安全性，但其柔顺、高维且非线性的动力学特性使得面向控制的建模十分困难。现有的数据驱动方法往往难以泛化，受限于范围狭窄的任务演示或低效的随机探索。我们提出了一种不确定性感知的主动探索框架，能够自主学习任务无关且可泛化的软体机器人系统动力学模型。我们在三个仿真平台上对该框架进行了评估——连续体机械臂、流体中带关节的仿生鱼以及肌肉骨骼腿——并在真实世界的气动驱动连续体机械臂上进行了验证。与随机探索和特定任务基线相比，我们的方法通过主动收集信息丰富的数据实现了更高的建模精度。这种改进的模型质量在未见任务的零样本控制中带来了更高的回报，并在传感噪声下表现出鲁棒性能。

    arXiv:2510.27428v2 Announce Type: replace-cross  Abstract: Soft robots offer adaptability and safety in unstructured environments, yet their compliant, high-dimensional, and nonlinear dynamics make modeling for control difficult. Existing data-driven approaches often fail to generalize, constrained by narrowly focused task demonstrations or inefficient random exploration. We introduce an uncertainty-aware active exploration framework that autonomously learns task-agnostic and generalizable dynamics models of soft robotic systems. We evaluate this framework on three simulated platforms---a continuum arm, an articulated fish in fluid, and a musculoskeletal leg---and on a real-world pneumatically actuated continuum arm. Compared with random exploration and task-specific baseline, our method achieves higher modeling accuracy by actively collecting informative data. This improved model quality yields higher returns in zero-shot control on unseen tasks and robust performance under sensing no
    
[^343]: 重新思考推理：大语言模型中基于推理的后门攻击综述

    Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs

    [https://arxiv.org/abs/2510.07697](https://arxiv.org/abs/2510.07697)

    本文首次对大语言模型中基于推理的后门攻击进行了全面综述，提出了将攻击分为关联型、被动型和主动型三类的新分类法，并系统分析了其机制、方法框架、防御策略及未解决的挑战。

    

    随着先进推理能力的兴起，大语言模型（LLMs）受到越来越多的关注。然而，尽管推理能力提升了大语言模型在下游任务上的性能，但它也带来了新的安全风险，因为攻击者可以利用这些能力发起后门攻击。现有的关于后门攻击和推理安全的综述提供了全面的概述，但缺乏针对大语言模型推理能力的后门攻击与防御的深入分析。在本文中，我们迈出了第一步，通过分析其潜在机制、方法框架和未解决的挑战，对大语言模型中基于推理的后门攻击进行了全面综述。具体而言，我们提出了一种新的分类法，为总结现有方法提供了统一视角，将基于推理的后门攻击分为关联型、被动型和主动型三类。我们还提出了相应的防御策略……

    arXiv:2510.07697v2 Announce Type: replace-cross  Abstract: With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategie
    
[^344]: UpSafe°C：面向大语言模型可控安全性的升级改造方法

    UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models

    [https://arxiv.org/abs/2510.02194](https://arxiv.org/abs/2510.02194)

    提出UpSafe°C框架，通过将安全关键层升级为稀疏MoE结构、利用路由器作为软防护栏、结合两阶段SFT和可调节的安全温度机制，在保持通用能力的同时实现大语言模型推理时可灵活控制的安全防护。

    

    大语言模型（LLMs）在广泛的任务中取得了显著进展，但仍然容易受到安全风险的威胁，例如有害内容生成和越狱攻击。现有的安全技术——包括外部防护栏、推理时引导和后训练对齐——在平衡安全性、实用性和可控性方面各有局限。在这项工作中，我们提出了UpSafe°C，一个通过安全感知升级改造来增强大语言模型安全性的统一框架。我们的方法首先识别安全关键层，并将其升级改造为稀疏的混合专家结构，其中路由器充当软防护栏，选择性地激活原始MLP和新增的安全专家。我们进一步引入两阶段SFT策略，在保持通用能力的同时增强安全判别能力。为了在推理时实现灵活控制，我们引入了安全温度机制，允许……

    arXiv:2510.02194v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have achieved remarkable progress across a wide range of tasks, but remain vulnerable to safety risks such as harmful content generation and jailbreak attacks. Existing safety techniques---including external guardrails, inference-time guidance, and post-training alignment---each face limitations in balancing safety, utility, and controllability. In this work, we propose UpSafe$^\circ$C, a unified framework for enhancing LLM safety through safety-aware upcycling. Our approach first identifies safety-critical layers and upcycles them into a sparse Mixture-of-Experts (MoE) structure, where the router acts as a soft guardrail that selectively activates original MLPs and added safety experts. We further introduce a two-stage SFT strategy to strengthen safety discrimination while preserving general capabilities. To enable flexible control at inference time, we introduce a safety temperature mechanism, allowing 
    
[^345]: 发现学习到的哈密顿系统中的相空间结构

    Discovering Phase Space Structure in Learned Hamiltonian Systems

    [https://arxiv.org/abs/2509.23003](https://arxiv.org/abs/2509.23003)

    提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。

    

    力学启发的神经动力学模型提升了可解释性和物理一致性，但它们通常假设构型空间已知、物理参数固定，并且从给定初始条件出发进行确定性预测。我们提出结构化相空间生成对抗网络（SPS-GAN），这是一种条件生成模型，能够同时从笛卡尔轨迹和视频中学习结构化动力学。SPS-GAN 将端口哈密顿骨干网络与对抗式条件生成相结合，使保守系统、耗散系统和受迫系统能够在单一架构中得到表示。为了从原始观测中推断潜在的力学结构，我们采用循环坐标损失，在无需访问真实相空间的情况下同时恢复系统的自由度和正则坐标。在模拟系统和真实世界系统上的实验表明，SPS-GAN 在轨迹精度上优于监督动力学基线方法，并提升了视频一致性。

    arXiv:2509.23003v3 Announce Type: replace-cross  Abstract: Mechanics-inspired neural dynamical models improve interpretability and physical consistency, but they usually assume a known configuration space, fixed physical parameters, and deterministic prediction from a prescribed initial condition. We introduce Structured Phase Space GAN (SPS-GAN), a conditional generative model that learns structured dynamics from both Cartesian trajectories and video. SPS-GAN combines a port-Hamiltonian backbone with adversarial conditional generation, allowing conservative, dissipative, and forced systems to be represented in a single architecture. To infer latent mechanical structure from raw observations, we use a cyclic-coordinate loss that recovers both the system degrees of freedom and canonical coordinates without access to the true phase space. Experiments on simulated and real-world systems show that SPS-GAN improves trajectory accuracy over supervised dynamics baselines and video consistency
    
[^346]: 面向基于大语言模型社会模拟的种群对齐式人格生成

    Population-Aligned Persona Generation for LLM-based Social Simulation

    [https://arxiv.org/abs/2509.10127](https://arxiv.org/abs/2509.10127)

    该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。

    

    大语言模型（LLM）的最新进展使得大规模、高保真度的社会模拟成为可能，为计算社会科学创造了新的机遇。然而，构建能够忠实反映现实世界人口多样性的角色（persona）集合仍然是一个关键挑战。现有研究往往强调智能体框架和模拟环境，而对角色生成以及不具代表性的角色集合所引入的偏差关注较少。本文提出了一个系统性框架，用于为LLM驱动的社会模拟合成高质量、与种群对齐的角色集合。我们的方法首先利用LLM从长期社交媒体数据中生成叙事型角色，随后通过严格的质量评估来过滤低保真度的角色档案。接着，我们应用重要性采样来实现与参考心理测量分布（如大五人格）的全局对齐。

    arXiv:2509.10127v3 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have enabled large-scale, high-fidelity social simulations, creating new opportunities for computational social science. However, constructing persona sets that faithfully reflect real-world population diversity remains a key challenge. Existing studies often emphasize agentic frameworks and simulation environments, while paying less attention to persona generation and the biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality 
    
[^347]: 大语言模型的金融大脑扫描

    A Financial Brain Scan of the LLM

    [https://arxiv.org/abs/2508.21285](https://arxiv.org/abs/2508.21285)

    该论文利用稀疏自编码器对大语言模型进行“金融大脑扫描”，实现对内部概念的可解释映射与精准引导，并揭示了模型自我解释与其真实内部推理之间存在明显脱节。

    

    我们使用稀疏自编码器（SAEs）对大型语言模型进行“大脑扫描”，将模型内部激活映射到可解释的概念上，并能够对特定特征进行有针对性的“引导”。我们表明，通过提示让模型采用特定偏好会激活许多非预期的概念，从而使因果推断变得复杂。相比之下，SAE 能够对目标概念实现平滑且更加孤立的引导。此外，我们的扫描揭示了模型内部特征贡献与其自我生成的解释之间存在明显的脱节，表明这些解释并不能可靠地反映模型的推理过程。最后，SAE 为现有模型提供了一种轻量级的干预手段，能够生成可解释的嵌入，并使研究人员能够减弱模型中特定的经济偏见。

    arXiv:2508.21285v4 Announce Type: replace-cross  Abstract: We use sparse autoencoders (SAEs) to "brain scan" large language models, mapping internal activations to interpretable concepts and enabling targeted "steering" of specific features. We show that prompting models to adopt specific preferences activates many unintended concepts, complicating causal inference. In contrast, SAEs permit smooth, more isolated steering of target concepts. Furthermore, our scans reveal a stark disconnect between the model's internal feature contributions and its self-generated explanations, showing that these explanations are unreliable guides to its reasoning. Finally, SAEs provide a lightweight intervention for existing models, producing interpretable embeddings and allowing researchers to attenuate specific economic biases.
    
[^348]: 边缘型人格障碍的计算现象学：LLM模拟专家角色与人类临床专家的比较评估

    Computational Phenomenology of Borderline Personality Disorder: A Comparative Evaluation of LLM-Simulated Expert Personas and Human Clinical Experts

    [https://arxiv.org/abs/2508.19008](https://arxiv.org/abs/2508.19008)

    本研究通过混合方法比较评估了LLM模拟专家与人类临床专家在边缘型人格障碍定性分析中的表现，发现LLM在支持临床主题分析方面具有潜力但存在差异。

    

    基于一项由人类主导的对边缘型人格障碍住院患者临床生活史访谈（超过15万字）的主题分析，本研究考察了大型语言模型（OpenAI的GPT、Google的Gemini和Anthropic的Claude）支持定性临床分析的能力。通过混合方法评估了模型的解释潜力。研究A涉及现象学和临床心理学领域的盲审与非盲审评判者，专家们使用语义一致性、Jaccard系数和多维有效性评级（包括可信度、连贯性、结果的实质性及定性数据基础）评估了AI生成内容的有效性。在研究B中，使用神经方法将人类和模型生成的主题描述嵌入多维向量空间，该方法提供了人类与模型之间差异的客观化计算度量。

    arXiv:2508.19008v3 Announce Type: replace  Abstract: Building on a human-led thematic analysis of clinical life-story interviews (> 150,000 words) with inpatients with Borderline Personality Disorder, this study examines the capacity of large language models (OpenAI's GPT, Google's Gemini, and Anthropic's Claude) to support qualitative clinical analysis. The models' interpretative potential was evaluated using a mixed-methods approach. Study A involved blinded and non-blinded judges in phenomenology and clinical psychology. The experts assessed the validity of AI-generated content using semantic congruence, Jaccard coefficients, and multidimensional validity ratings, including credibility, coherence, the substantiveness of results, and grounding in qualitative data. In Study B, neural methods were used to embed human- and model-generated theme descriptions in a multidimensional vector space. This approach provided an objectified computational measure of the difference between human and
    
[^349]: 面向分子表示学习的预训练分子嵌入模型基准测试

    Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning

    [https://arxiv.org/abs/2508.06199](https://arxiv.org/abs/2508.06199)

    本研究对28个预训练分子嵌入模型进行了大规模基准测试，发现除CheMeleon外所有神经模型均未超越经典ECFP计数指纹，且外部验证显示两者并无明确优劣，建议在新QSAR/QSPR研究中同时评估这两种方法。

    

    覆盖图神经网络、文本Transformer和混合架构的预训练分子嵌入模型已被广泛应用于药物发现领域，然而此前尚缺乏对其表示能力的大规模严格比较。我们在一个受控框架下，采用层次贝叶斯Bradley-Terry统计检验方法，在25个分类数据集上对28个模型进行了评估。结果显示，除CheMeleon外，所有神经模型均未能超越经典的ECFP计数指纹，CheMeleon是唯一被评为优于ECFP指纹的模型。然而，基于支架交叉验证和贝叶斯层次相关t检验的外部验证数据集的进一步测试表明，两种嵌入方法均不具备明确的优势。作为实用的结论，我们建议在新的QSAR/QSPR研究中同时评估ECFP指纹和CheMeleon嵌入。总体而言，就所有被评估的神经网络而言，这些结果也挑战了人们所认为的（神经模型的优势）。

    arXiv:2508.06199v5 Announce Type: replace-cross  Abstract: Pretrained molecular embedding models spanning graph neural networks, text transformers, and hybrid architectures have been widely used in drug discovery, yet no large-scale, rigorous comparison of their representations existed. We evaluated 28 models across 25 classification datasets under a controlled framework using hierarchical Bayesian Bradley-Terry statistical testing. All neural models except CheMeleon failed to outperform the classical ECFP count fingerprint; CheMeleon was the only model rated as better. However, further testing on external validation datasets, based on scaffold cross-validation and a Bayesian hierarchical correlated t-test, revealed no definite advantage of either embedding method. As a practical takeaway, we recommend evaluating both ECFP fingerprints and CheMeleon embeddings in novel QSAR/QSPR studies. Overall, in the context of all neural networks evaluated, these results also challenge the perceive
    
[^350]: DUP：面向语言模型后门净化的检测引导遗忘方法

    DUP: Detection-guided Unlearning for Backdoor Purification in Language Models

    [https://arxiv.org/abs/2508.01647](https://arxiv.org/abs/2508.01647)

    提出DUP统一框架，通过联合类无关距离与层间转换实现细粒度后门检测，并利用参数高效的遗忘机制净化语言模型中的后门，无需完整重训练或外部干净模型。

    

    随着后门攻击变得更加隐蔽和鲁棒，它们暴露了当前防御策略的关键弱点：检测方法通常依赖于粗粒度的特征统计，而净化方法通常需要完整的重新训练或额外的干净模型。为了应对这些挑战，我们提出了DUP（检测引导的遗忘净化），这是一个将后门检测与基于遗忘的净化相结合的统一框架。该检测器通过联合利用类无关距离和层间转换来捕获特征级别的异常。这些偏差通过加权方案进行整合以识别被投毒的输入，从而实现更细粒度的分析。基于检测结果，我们通过一种参数高效的遗忘机制来净化模型，该机制避免了完整的重新训练，且不需要任何外部干净模型。具体而言，我们创新性地将知识蒸馏重新用于

    arXiv:2508.01647v2 Announce Type: replace-cross  Abstract: As backdoor attacks become more stealthy and robust, they reveal critical weaknesses in current defense strategies: detection methods often rely on coarse-grained feature statistics, and purification methods typically require full retraining or additional clean models. To address these challenges, we propose DUP (Detection-guided Unlearning for Purification), a unified framework that integrates backdoor detection with unlearning-based purification. The detector captures feature-level anomalies by jointly leveraging class-agnostic distances and inter-layer transitions. These deviations are integrated through a weighted scheme to identify poisoned inputs, enabling more fine-grained analysis. Based on the detection results, we purify the model through a parameter-efficient unlearning mechanism that avoids full retraining and does not require any external clean model. Specifically, we innovatively repurpose knowledge distillation t
    
[^351]: 当评论稀缺时：现实稀疏评论场景下感知结构的评论补全推荐方法

    When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings

    [https://arxiv.org/abs/2508.01128](https://arxiv.org/abs/2508.01128)

    提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。

    

    文本评论为推荐系统提供了细粒度的偏好信号和更强的可解释性。然而，在现实场景中，用户很少留下评论，导致严重的稀疏性，削弱了部分可观测条件下评论感知推荐器的有效性。一个自然的补救方法是合成缺失的评论，以恢复这些推荐器所依赖的文本信号。然而，传统的补全技术，如矩阵补全和基于大语言模型的增强方法，要么在将文本嵌入为向量时丢失上下文语义，要么忽视了用户-物品图上的结构依赖关系。为了解决这些缺陷，我们提出了TWISTER（ToWards Imputation on Sparsity with Textual Edge Graph Representation），这是一个用于稀疏文本推荐的统一框架，通过联合建模语义信号和结构信号来合成缺失评论。具体而言，我们将用户-物品交互表示为……

    arXiv:2508.01128v2 Announce Type: replace-cross  Abstract: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of review-aware recommenders under partial observability. A natural remedy is to synthesize the missing reviews to restore the textual signal these recommenders rely on. Yet, conventional imputation techniques, such as matrix completion and LLM-based augmentation, either lose contextualized semantics by embedding texts into vectors or overlook structural dependencies on the user-item graph. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework for sparse-text recommendation that synthesizes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as 
    
[^352]: 面向机器学习任务的可扩展数据源选择

    Scalable Source Selection for Machine Learning Tasks

    [https://arxiv.org/abs/2507.22186](https://arxiv.org/abs/2507.22186)

    提出了Splice系统，一种借鉴基因剪接思想的目标与成本感知数据源选择方法，通过识别能为下游任务提供额外信号的数据源来最大化任务效用，并开发了利用数据画像和梯度对齐的两种加速变体Splice-DP和Splice-Grad。

    

    给定一个查询表，现有的数据湖发现技术通过模式相似性和连接性来识别与查询表相关的数据源。然而，发现的数据源并不总是对下游分析有用；其中一些可以提高任务泛化能力，而另一些则可能无意中引入有害噪声或分布偏移。我们将数据源选择形式化为一个寻找被发现数据源子集的问题，该子集能够最大化下游任务的效用。我们提出了Splice，一个目标和成本感知的数据源选择系统，它借鉴基因剪接的思想来识别那些能提供所有数据源并集所缺失的额外信号的数据源。为了加速这一过程，我们开发了两个代理变体Splice-DP和Splice-Grad，它们分别通过在元学习方法中利用数据画像以及梯度对齐来估计子集对任务效用的边际贡献。

    arXiv:2507.22186v3 Announce Type: replace-cross  Abstract: Given a query table, existing data lake discovery techniques seek to identify data sources related to the query table through schema similarity and joinability. The discovered sources, however, are not always useful for downstream analyses; while some improve task generalization, some have the potential to inadvertently introduce harmful noise or distribution shifts. We formalize source selection as the problem of finding a subset of the discovered sources that maximizes downstream task utility. We propose Splice, a target- and cost-aware source selection system that adopts the idea of gene splicing to identify sources which provide additional signals that the union of all sources misses. To speed up the process, we develop two surrogate variants, Splice-DP and Splice-Grad, that estimate the marginal contribution of subsets to task utility by leveraging data profiles in a meta-learning approach and gradient alignment, respectiv
    
[^353]: 面向目标的因果学习序贯贝叶斯实验设计

    Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning

    [https://arxiv.org/abs/2507.07359](https://arxiv.org/abs/2507.07359)

    GO-CBED是一个面向目标的贝叶斯序贯因果实验设计框架，通过变分下界估计与Transformer策略网络直接最大化用户指定因果量的期望信息增益，实现非短视且实时的干预决策，在因果推理与发现任务中优于现有基线。

    

    我们提出了GO-CBED，一个面向目标的序贯因果实验设计贝叶斯框架。与旨在通过选择干预来推断完整因果模型的传统方法不同，GO-CBED直接最大化用户指定的目标因果量上的期望信息增益（EIG），从而实现更有针对性、更高效的实验。该框架既具有非短视性（对整个干预序列进行优化），又具有面向目标性（仅关注与因果查询相关的模型方面）。为解决精确EIG计算的难解性问题，我们引入了一个变分下界估计器，通过基于Transformer的策略网络和基于归一化流的变分后验进行联合优化。由此得到的策略可通过摊销网络实现实时决策。我们证明，GO-CBED在多种因果推理与因果发现任务中始终优于现有基线方法。

    arXiv:2507.07359v2 Announce Type: replace-cross  Abstract: We present GO-CBED, a goal-oriented Bayesian framework for sequential causal experimental design. Unlike conventional approaches that select interventions aimed at inferring the full causal model, GO-CBED directly maximizes the expected information gain (EIG) on user-specified causal quantities of interest, enabling more targeted and efficient experimentation. The framework is both non-myopic, optimizing over entire intervention sequences, and goal-oriented, targeting only model aspects relevant to the causal query. To address the intractability of exact EIG computation, we introduce a variational lower bound estimator, optimized jointly through a transformer-based policy network and normalizing flow-based variational posteriors. The resulting policy enables real-time decision-making via an amortized network. We demonstrate that GO-CBED consistently outperforms existing baselines across various causal reasoning and discovery ta
    
[^354]: 分子版《绝命毒师》：多模态大语言模型准备好进行结构级分子脱毒了吗？

    Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?

    [https://arxiv.org/abs/2506.10912](https://arxiv.org/abs/2506.10912)

    该论文提出了ToxiMol——首个评估通用多模态大语言模型分子毒性修复能力的基准任务，包含覆盖11个任务、660个有毒分子的标准化数据集、机制感知的提示标注流程以及自动化评估框架ToxiEval。

    

    毒性仍然是早期药物开发失败的主要原因。尽管分子设计和性质预测领域已取得进展，但分子毒性修复这一任务——即生成结构有效且毒性降低的分子替代物——尚未被系统地定义或建立基准。为填补这一空白，我们提出了ToxiMol，这是首个专注于分子毒性修复的通用多模态大语言模型（MLLMs）基准任务。我们构建了一个标准化数据集，涵盖11个主要任务和660个代表性有毒分子，覆盖多种毒性机制和粒度。我们设计了一个具有机制感知和任务自适应能力的提示标注流程，并融入了专家毒理学知识。同时，我们提出了一个自动化评估框架ToxiEval，它将毒性终点预测、合成可及性、类药性和结构相似性整合到一个高（原文在此处截断）。

    arXiv:2506.10912v5 Announce Type: replace  Abstract: Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair, generating structurally valid molecular alternatives with reduced toxicity, has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 660 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a hig
    
[^355]: 后见之明重标注即抵达-规避学习所需的一切

    Hindsight Relabeling is All You Need for Reach-Avoid Learning

    [https://arxiv.org/abs/2505.19337](https://arxiv.org/abs/2505.19337)

    本文提出RADT，一种基于决策Transformer的离线无奖励强化学习方法，通过将目标与规避区域直接编码为提示token，仅凭随机策略产生的次优离线轨迹即可学会抵达-规避行为，并能在评估时灵活指定任意数量和大小的规避区域。

    

    离线目标条件强化学习方法在抵达-规避任务中展现出良好前景，这类任务要求智能体在抵达目标状态的同时避开状态空间中的不良区域。现有方法通常将规避区域信息编码到增广状态空间和代价函数中，这阻碍了在评估时对新规避区域信息进行灵活、动态的设定。此外，这些方法还严重依赖精心设计的奖励和代价函数，限制了对新环境和新规范的可迁移性。我们提出RADT，一种用于离线、无奖励、目标条件、规避区域条件强化学习的决策Transformer方法。RADT将目标和规避区域直接编码为提示token，允许在评估时指定任意数量、任意大小的规避区域。仅使用来自随机策略的次优离线轨迹，RADT即可在完整……（原文摘要在此处截断）中学到抵达-规避行为。

    arXiv:2505.19337v3 Announce Type: replace-cross  Abstract: Offline goal-conditioned reinforcement learning methods have shown promise for reach-avoid tasks, where an agent must reach a target state while avoiding undesirable regions of the state space. Existing approaches typically encode avoid-region information into an augmented state space and cost function, which prevents flexible, dynamic specification of novel avoid-region information at evaluation time. They also rely heavily on meticulously designed reward and cost functions, limiting transferability to novel environments and specifications. We introduce RADT, a decision transformer approach for offline, reward-free, goal-conditioned, avoid-region-conditioned RL. RADT encodes goals and avoid-regions directly as prompt tokens, allowing any number of avoid-regions of arbitrary size to be specified at evaluation time. Using only suboptimal offline trajectories from a random policy, RADT can learn reach-avoid behavior in a complete
    
[^356]: ENACT-Heart——基于CNN与Transformer集成的心音评估方法

    ENACT-Heart -- ENsemble-based Assessment Using CNN and Transformer on Heart Sounds

    [https://arxiv.org/abs/2502.16914](https://arxiv.org/abs/2502.16914)

    ENACT-Heart通过将处理频谱图的视觉Transformer与处理波形和谱质心可视化的CNN进行加权后期融合，在PASCAL心音数据集上实现了97.52%的分类准确率，优于任一单分支模型。

    

    ENACT-Heart是一个用于心音分类的双分支集成模型。视觉Transformer（ViT）处理频谱图图像，而卷积神经网络（CNN）处理由相同录音导出的波形及谱质心的可视化结果。两个分支的预测通过具有固定且与输入无关权重的加权平均进行融合。在PASCAL心音数据集上，该集成模型达到了97.52%的准确率，相比之下ViT的准确率为93.88%，CNN的准确率为95.45%。该集成模型取得了0.94的宏F1分数，以及对额外心音0.74的召回率。在该数据集上，后期融合相对于两个单独分支均提升了准确率和宏F1分数。

    arXiv:2502.16914v2 Announce Type: replace-cross  Abstract: ENACT-Heart is a two-branch ensemble for heart-sound classification. A Vision Transformer (ViT) processes spectrogram images, while a convolutional neural network (CNN) processes visualizations of waveforms and spectral centroids derived from the same recordings. The branch predictions are combined by a weighted average with fixed, input-independent weights. On the PASCAL heart-sound dataset, the ensemble achieved an accuracy of 97.52%, compared with 93.88% for the ViT and 95.45% for the CNN. The ensemble achieved a macro-F1 of 0.94 and recall of 0.74 for extra heart sounds. On this dataset, late fusion improved accuracy and macro-F1 relative to both individual branches.
    
[^357]: EgoGen:一种自我的合成数据生成器

    EgoGen: An Egocentric Synthetic Data Generator. (arXiv:2401.08739v1 [cs.CV])

    [http://arxiv.org/abs/2401.08739](http://arxiv.org/abs/2401.08739)

    EgoGen是一种新的合成数据生成器，它可以为具体化自我的感知任务生成准确和丰富的地面真实训练数据，并利用自我视觉输入来感知3D环境。

    

    在增强现实（AR）中，以第一人称视角理解世界是基础性的。这种身临其境的视角相对于第三人称视角带来了戏剧性的视觉变化和独特的挑战。合成数据已经赋予了第三人称视角视觉模型的能力，但其应用于具体化自我的感知任务仍然很少被探索。一个重要的挑战在于模拟自然的人类运动和行为，以有效地操纵实体相机捕捉到3D世界的真实自我的表达。为了解决这个挑战，我们引入了一个新的合成数据生成器EgoGen，它可以为具体化自我的感知任务生成准确和丰富的地面真实训练数据。EgoGen的核心是一种新颖的人体运动合成模型，它直接利用虚拟人类的自我视觉输入感知3D环境。结合避免碰撞的运动基元和两阶段强化学习方法，我们的运动合成模型实现了高效的自我的合成。

    Understanding the world in first-person view is fundamental in Augmented Reality (AR). This immersive perspective brings dramatic visual changes and unique challenges compared to third-person views. Synthetic data has empowered third-person-view vision models, but its application to embodied egocentric perception tasks remains largely unexplored. A critical challenge lies in simulating natural human movements and behaviors that effectively steer the embodied cameras to capture a faithful egocentric representation of the 3D world. To address this challenge, we introduce EgoGen, a new synthetic data generator that can produce accurate and rich ground-truth training data for egocentric perception tasks. At the heart of EgoGen is a novel human motion synthesis model that directly leverages egocentric visual inputs of a virtual human to sense the 3D environment. Combined with collision-avoiding motion primitives and a two-stage reinforcement learning approach, our motion synthesis model off
    
[^358]: 估计和激励具有隐藏奖励的不完全知识代理

    Estimating and Incentivizing Imperfect-Knowledge Agents with Hidden Rewards. (arXiv:2308.06717v1 [cs.LG])

    [http://arxiv.org/abs/2308.06717](http://arxiv.org/abs/2308.06717)

    本文研究了一个自私学习代理和学习委托人之间的重复对自选游戏，探索如何估计和激励具有隐藏奖励的不完全知识代理。

    

    在实践中，激励提供者（即委托人）通常无法观察受到激励的代理的奖励实现情况，这与许多先前研究过的委托人-代理模型形成了对比。这种信息不对称性使委托人仅通过观察代理的决策就要始终估计代理的未知奖励变得更加困难，当代理需要学习自己的奖励时，这个挑战变得更加困难。这种复杂情况在各种现实场景中被观察到，从可再生能源储存合同到个性化医疗保健激励。因此，它不仅提供了有趣的理论问题，而且具有广泛的实际相关性。本文探讨了一个自私学习代理和学习委托人之间的重复对自选游戏。代理解决多臂老虎机（MAB）问题，以最大化他们预期的奖励和激励。除代理的学习外，委托人还训练了一个并行算法，并面临一个折中情况

    In practice, incentive providers (i.e., principals) often cannot observe the reward realizations of incentivized agents, which is in contrast to many principal-agent models that have been previously studied. This information asymmetry challenges the principal to consistently estimate the agent's unknown rewards by solely watching the agent's decisions, which becomes even more challenging when the agent has to learn its own rewards. This complex setting is observed in various real-life scenarios ranging from renewable energy storage contracts to personalized healthcare incentives. Hence, it offers not only interesting theoretical questions but also wide practical relevance. This paper explores a repeated adverse selection game between a self-interested learning agent and a learning principal. The agent tackles a multi-armed bandit (MAB) problem to maximize their expected reward plus incentive. On top of the agent's learning, the principal trains a parallel algorithm and faces a trade-of
    
[^359]: 未观测到代理奖励的重复负责人代理博弈问题研究

    Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents. (arXiv:2304.07407v1 [cs.LG])

    [http://arxiv.org/abs/2304.07407](http://arxiv.org/abs/2304.07407)

    本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    

    本文研究了一个多臂老虎机框架中的重复负责人代理博弈场景，其中代理选择一种老虎机后会获得奖励和激励，但负责人只能观察到代理选择了哪个老虎机以及代理相应的激励，而想要设计一种合适的策略却充满了挑战性。本文提出了一种利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    Motivated by a number of real-world applications from domains like healthcare and sustainable transportation, in this paper we study a scenario of repeated principal-agent games within a multi-armed bandit (MAB) framework, where: the principal gives a different incentive for each bandit arm, the agent picks a bandit arm to maximize its own expected reward plus incentive, and the principal observes which arm is chosen and receives a reward (different than that of the agent) for the chosen arm. Designing policies for the principal is challenging because the principal cannot directly observe the reward that the agent receives for their chosen actions, and so the principal cannot directly learn the expected reward using existing estimation techniques. As a result, the problem of designing policies for this scenario, as well as similar ones, remains mostly unexplored. In this paper, we construct a policy that achieves a low regret (i.e., square-root regret up to a log factor) in this scenar
    

