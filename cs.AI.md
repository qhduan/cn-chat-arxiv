# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction](https://arxiv.org/abs/2610.37668) | Hexels是一种带有局部颜色场的平面六边形基元，用更少、更大的基元即可高效表示3D场景，在与内存匹配的2DGS对比中PSNR提升4.53-7.38 dB，且训练显存更低、渲染速度快2.31倍。 |
| [^2] | [ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning](https://arxiv.org/abs/2610.37666) | 提出ThinkWorld框架，以动力学链作为跨具身通用动作表示，将动作推理与世界动力学相结合，并借助世界想象的强化学习增强VLA模型的泛化能力。 |
| [^3] | [Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops](https://arxiv.org/abs/2610.37656) | 论文证明，在允许每个实体独立旋转自身坐标系的规范对称性下，基于威尔逊圈（闭合环路和乐）的边注意力机制不是众多候选设计之一，而是对称性唯一允许的形式——关系之间只能通过共享的闭合环路发生相互作用。 |
| [^4] | [KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry](https://arxiv.org/abs/2610.37655) | KPPax 是一个动力学预处理器，能将 KPP 语言描述的大气化学机理转化为 JAX 中可微分、即时编译且可向量化的代码，并结合稀疏对数空间运算与 diffrax 隐式求解器，无需手工推导伴随代码即可获得精确敏感度。 |
| [^5] | [Can LLMs Uncover Hidden Physical Laws through Free Exploration?](https://arxiv.org/abs/2610.37649) | 该论文提出LongChang自动化框架，通过构建物理定律被改变的模拟世界，评估LLM能否在没有任务提示的情况下自主注意到异常物理现象，并通过自由实验和工具构建发现隐藏的物理定律。 |
| [^6] | [MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control](https://arxiv.org/abs/2610.37647) | 该论文提出“累积有效学习率”分析框架，揭示原始Muon仅能实现对数级范数累积而MuonW/MuonH可达线性累积，并据此提出放宽权重范数控制的MuonGS方法以更高效地预训练大语言模型。 |
| [^7] | [What Improves During Pretraining? Tracking Knowledge Acquisition with Point-in-Time Models](https://arxiv.org/abs/2610.37631) | 该论文提出时间变异性分数（TVS）来衡量知识在不同年份网络文本中的一致性，发现高变异性知识在预训练早期与低变异性知识的学习速度相似，但在后期提升更快。 |
| [^8] | [Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions](https://arxiv.org/abs/2610.37624) | 该论文提出CrossProfile诊断基准，通过交叉控制支持性、诱饵和无关事实的持有分布，揭示了LLM群体决策中两种记忆压缩启发式的可预测失败：基于相关性选择会重新引入切题的诱饵信息，而最稀有优先选择平均表现不优于随机选择。 |
| [^9] | [VideoLVR: Latent Visual Reasoning over Grounded Video Evidence](https://arxiv.org/abs/2610.37613) | VideoLVR 通过从相关视频时刻的局部区域构建潜在视觉证据，为中间推理步骤提供视觉支持，在不大幅扩展上下文的前提下解决了视频推理中的视觉遗忘问题。 |
| [^10] | [Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence](https://arxiv.org/abs/2610.37605) | 该论文提出匹配同类对照协议，通过控制区间、响度、供体拟合等编辑因素，比较同类与不同类声音事件替换对音频-语言模型答案的影响，证明模型答案变化主要源于对事件类别的响应而非对波形重写的敏感性，从而为局部音频证据提供更严格的检验方法。 |
| [^11] | [MoI-Bench: A Benchmark for Image Generation and Editing on Mobile Devices](https://arxiv.org/abs/2610.37599) | 提出了基于移动用户需求的图像生成与编辑基准测试MoI-Bench（包含64个子任务、1150个样本、16种操作和8个视觉目标类别），并配套提出任务特定的评估方法MoI-Eval，以量化端侧模型与服务器端模型的能力差距并指导模型改进。 |
| [^12] | [GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting](https://arxiv.org/abs/2610.37598) | GAS-PFN通过随机游走扩散将道路拓扑融入合成任务先验，提出了显式建模图结构依赖的图感知时空PFN，实现了无需针对特定网络训练的零样本交通预测。 |
| [^13] | [Understanding the limits of latent world models](https://arxiv.org/abs/2610.37588) | 该论文提出“忠实潜在世界模型”概念及潜在展开一致性（LRC）评估指标，发现当前所有最先进的潜在世界模型在解决任务时都并未真正忠实地建模环境的动态。 |
| [^14] | [Timestral](https://arxiv.org/abs/2610.37585) | Timestral 是一系列面向概率预测的时间序列基础模型，通过解耦视界预测（DHF）、原生支持已知未来协变量的框架以及“先大规模合成数据预训练、后真实与合成混合数据后训练”的两阶段流程，在 GIFT-Eval、TIME 和 fev-bench 上取得了最先进的性能。 |
| [^15] | [REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2610.37584) | 提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。 |
| [^16] | [TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information](https://arxiv.org/abs/2610.37571) | TeamGym是基于真实法律、金融、医疗和软件工程语料构建的团队决策评测与训练环境，通过引入“关键信息”这一概念发现当前语言模型智能体在信息分布共享的团队决策场景中难以有效整合信息、表现明显下降。 |
| [^17] | [Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents](https://arxiv.org/abs/2610.37567) | 提出了Gym-Anything-RT，一个基于六种实时交互能力分类体系、通过智能体流水线自动生成视觉小游戏的基准环境，用于系统评估视觉-语言智能体的实时交互能力。 |
| [^18] | [When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance](https://arxiv.org/abs/2610.37561) | 该研究揭示了在临床类别严重不平衡的预后任务中，标准不确定性度量会退化为类别归属的代理指标并系统性惩罚少数类患者，导致选择性预测安全机制失效，并提出了一种能在合成场景与多发性硬化、帕金森病、阿尔茨海默病等真实队列中保持有界、平衡风险的修正框架。 |
| [^19] | [Kinematic Constraint-Referenced Evaluation for Articulated Object Manipulation](https://arxiv.org/abs/2610.37550) | 提出约束参考策略（CRS），通过将末端执行器指令速度投影到物体关节轴上，将沿关节的有效进展与偏离约束的无效动作区分开，从而诊断出传统二元成功率无法归因的迁移失败。 |
| [^20] | [When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies](https://arxiv.org/abs/2610.37547) | 提出轻量级后训练模块ARC，利用强化学习在冻结策略之上学习残差修正，并通过稀疏人类修正确定残差控制激活的“时机”与自主探索的“方向”，将真实机器人精密操作任务的平均成功率从33.3%提升至75%。 |
| [^21] | [GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis](https://arxiv.org/abs/2610.37545) | 该论文提出GroundControl，一种将卫星图像作为额外条件信息的多视角扩散模型，无需3D场景重建即可从少量街景图像生成既逼真又符合真实场景布局的室外场景新视角。 |
| [^22] | [Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use](https://arxiv.org/abs/2610.37533) | 该论文通过双手机器人工具使用任务揭示了不同感知通道可识别的故障类型差异——例如单手传感器无法判断另一只手的延迟或复合故障，并据此提出在故障原因不可识别时进行升级的修复策略，从而在有限尝试次数内实现更有效的故障恢复。 |
| [^23] | [The Structure Gap: When More Data Cannot Replace Structured Reasoning](https://arxiv.org/abs/2610.37523) | 本文提出“结构差距”这一概念，指出对于由求解器、物理定律等已知结构决定或约束的任务，纯数据驱动的学习器在有限资源下无法通过增加数据完全替代结构化推理，其差距集中体现在有限数据辨识、外推与认证能力上。 |
| [^24] | [Orthogonal and Parametrized Multi-Attribute Steering of LLMs](https://arxiv.org/abs/2610.37517) | 该论文提出参数化导向方法将激活设定在期望的特质强度上，并通过向量正交化解耦多特质间的干扰，实现了大语言模型多属性的同时、可解释且互不干扰的精确控制。 |
| [^25] | [Strategyproof Deep Reinforcement Learning for Online Resource Allocation](https://arxiv.org/abs/2610.37504) | 本文提出了首个用于在线资源分配的防策略性深度强化学习机制，通过单调神经网络等三个新颖组件，在实现非线性分配规则的同时严格保证参与者的真实申报激励。 |
| [^26] | [From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents](https://arxiv.org/abs/2610.37482) | 提出任务推进记忆（TAM），将经验证的任务进展同时作为执行过程中的状态和任务结束后的监督信号，使GUI智能体能够在长时程跨应用工作流中积累可复用操作，并从成功与失败的任务中学习局部有效的经验。 |
| [^27] | [Follow the Gradient: Tracing Safety Degradation Back to Training Data](https://arxiv.org/abs/2610.37481) | 本文提出SimpleLoRIF，一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本，从而追溯并过滤导致大语言模型安全退化的训练数据。 |
| [^28] | [HGPRGDel: GPR-Aware Learning and Ranking for Growth-Coupled Gene Deletion Design](https://arxiv.org/abs/2610.37478) | 该论文提出了HGPRGDel，一种GPR感知的可复现协议，通过从已有菌株设计数据中学习并生成基因敲除优先级排序，从而在基于约束的评估之前高效筛选候选干预方案，加速生长耦合微生物菌株的设计。 |
| [^29] | [How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI?](https://arxiv.org/abs/2610.37472) | 该研究首次系统量化了ADNI静息态fMRI认知预测中个体化功能网络归因的信度，发现其分半信度接近上限但重测信度中等，且归因结果虽以群体共享结构为主，仍包含适度但非零的个体特异性成分。 |
| [^30] | [PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions](https://arxiv.org/abs/2610.37471) | PerceptRL通过多智能体流水线将密集图像描述中经验证的错误转化为带固定答案标签和规则奖励的问答对（链式QA与选择题QA），并针对高错误率领域提供定制化验证，从而在MLLM后训练中使视觉感知变得可奖励化，弥补了单纯推理优化带来的感知短板。 |
| [^31] | [Autonomous fixed-wing interception using reinforcement learning](https://arxiv.org/abs/2610.37470) | 本文提出一种由强化学习（PPO）训练的神经网络，将滤波后的单目视觉观测直接映射为执行器指令，从而实现固定翼拦截机对中空长航时无人机（如Shahed型）的自主拦截。 |
| [^32] | [Concordat: Typed Sessions for LLM Agents](https://arxiv.org/abs/2610.37461) | Concordat 是一个 Python 库，它借鉴类型状态与会话类型的思想，将大语言模型智能体的接口建模为有限状态协议并在运行时强制执行，每一轮只向模型提供当前状态下合法的工具，从而用类型化的结构性保障取代仅靠提示词约束智能体行为的脆弱做法。 |
| [^33] | [M4Bench: Evaluating Procedural Specification for Clinical EHR Derivation Agents](https://arxiv.org/abs/2610.37459) | M4Bench基准通过28个临床EHR推导任务发现，AI编程智能体的性能提升主要来自完整的操作性规范，而非额外的技能包。 |
| [^34] | [Test-Time Communication in Trading Agent Swarms](https://arxiv.org/abs/2610.37448) | 该论文通过对来自四个模型系列的200个交易智能体在七个交易日内五种通信机制下的大规模实验，揭示了测试时通信会诱发“羊群效应”——通信导致智能体持仓趋同且群体收益主要由共同持仓决定，表明在个体目标独立、市场反馈迟缓且有噪声的交易场景中，智能体间的通信趋同是一种风险而非协作优势。 |
| [^35] | [Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning](https://arxiv.org/abs/2610.37437) | 该论文提出基于模拟器的ImagiRep基准，在同一未观测状态下对比RGB与六种替代表示（深度、全景、俯视几何图等），揭示不同空间推理任务受益于不同的想象表示，从而重新审视了空间想象中RGB作为默认表示的合理性。 |
| [^36] | [FeatureArc: Benchmarking Coding Agents on Evolving Features](https://arxiv.org/abs/2610.37434) | 提出 FeatureArc 基准测试，首次评估编码智能体在软件特性持续演化场景下的能力，其流水线从仓库历史构建包含阶段间依赖、明确需求契约、可执行 F2P/P2P 测试验证并覆盖四种编程语言的任务。 |
| [^37] | [Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training](https://arxiv.org/abs/2610.37429) | 本文提出AdaRoll-GRPO，将GRPO训练中的rollout预算分配建模为贝叶斯自适应马尔可夫决策过程，实现基于问题难度的自适应rollout分配，在相同总预算下最大化训练效用。 |
| [^38] | [Beyond the Readout: A Latent Numerical Frame Shared Across Modality and Operation](https://arxiv.org/abs/2610.37422) | 通过因果干预发现，多模态模型在文本与图像、加法与减法之间共享一个统一的潜在数值框架，但各条件沿不同方向编码数字，只需一次旋转和缩放即可实现子空间间的映射并恢复大部分因果效应。 |
| [^39] | [Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution](https://arxiv.org/abs/2610.37407) | 本文提出一种基于DIO/ODI的授权护栏，将智能体推理可用的证据与可授权持久性身份变更的证据分离，并通过两阶段监督器仅在授权合格证据充分支持所有必需命题、冲突已解决且授权范围与策略版本有效时才允许提交身份变更，从而防止错误身份假设在智能体系统中自我确认。 |
| [^40] | [Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers](https://arxiv.org/abs/2610.37405) | 该论文提出FoG框架，通过基于证据的数据合成、KG交互智能体和证据覆盖过程奖励（ECPR），训练知识图谱智能体不仅回答正确，还必须检索出支撑答案的证据三元组，从而使答案可追溯、可审计。 |
| [^41] | [Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling](https://arxiv.org/abs/2610.37393) | 提出质量可控策略（QCP），使机器人策略能够在推理时动态控制执行速度、运动平滑度等质量属性及其组合，并通过迭代事后重标注循环将可控质量范围扩展到初始演示数据之外。 |
| [^42] | [JEPA-Inspired Latent Field Prediction for Sparse-Pilot MIMO-OFDM Channel Reconstruction](https://arxiv.org/abs/2610.37391) | 提出一种受JEPA启发的无线传播场潜在世界模型，利用无线信道在潜在场域中更强的可预测相关性，从稀疏导频中稳健地重构MIMO-OFDM信道状态信息，克服了直接在原始CSI系数上预测的脆弱性。 |
| [^43] | [Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data](https://arxiv.org/abs/2610.37386) | 混合专家模型解码中的通道选择可以用无需校准数据的top-k方法替代基于离线数据校准阈值的传统方法，在匹配或超越阈值方法性能的同时使实际读取预算更加稳定可控。 |
| [^44] | [Agent-Conditioned Routing: Diverse Reasoning Paths Within One Frozen Model](https://arxiv.org/abs/2610.37375) | 提出智能体条件化路由（ACR），通过为冻结MoE模型的并行轨迹分配持久的专家资格掩码，在无需微调、角色指定或模型副本的情况下，使单个模型产生真正多样化的推理路径，其方法多样性约为温度采样最强设置的两倍。 |
| [^45] | [Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories](https://arxiv.org/abs/2610.37369) | 提出白盒诊断框架CauTrace，通过七种功能状态、表征传播、时间-状态协调与推理轨迹等内部机制分析，统一刻画大语言模型普通推理失败与对抗性诱导失败的根本原因。 |
| [^46] | [Aligning Institutional Power and Operational Ability](https://arxiv.org/abs/2610.37342) | 该论文提出了一个以面向过程建模和制度性权力为核心的形式化框架，用于分析和修复多智能体AI系统中责任与履行能力之间的规范性错配问题。 |
| [^47] | [CoS-VQ: Coordinated Shared-Scale Vector Quantization for Large Language Models](https://arxiv.org/abs/2610.37336) | CoS-VQ在保留低维码字的前提下联合优化硬索引与跨位置共享尺度，通过多尺度候选收集与切线可分离上界克服尺度耦合难题，实现大语言模型的高精度低比特向量量化。 |
| [^48] | [WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization](https://arxiv.org/abs/2610.37329) | WCC是一个基于学习的框架，通过将算法选择决策建模为马尔可夫决策过程，在大规模优化中动态决定何时使用协同进化或非分解算法，实现细粒度的动态切换与计算预算的自适应分配。 |
| [^49] | [AnalogScout: Simulation-Efficient PVT-Aware Analog Sizing with LLM Agents](https://arxiv.org/abs/2610.37319) | AnalogScout是一个由LLM智能体驱动的框架，通过结合精英设计复用、解析式gm/Id局部尺寸设计与全局细化，在有限仿真预算下大幅减少多角点SPICE仿真次数，实现仿真高效的PVT感知模拟电路尺寸设计。 |
| [^50] | [ReForest: Interpretability Meets Reasoning in Isolation Forest](https://arxiv.org/abs/2610.37316) | ReForest框架将孤立森林的决策路径转化为自然语言解释，借助大语言模型的推理能力和不确定性感知的动态集成机制，弥合统计异常检测与语义理解之间的鸿沟，实现高效且具备上下文感知能力的异常检测。 |
| [^51] | [A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection](https://arxiv.org/abs/2610.37314) | 该论文提出了一种带有显著性图可解释性的轻量级注意力CNN用于皮肤癌检测，仅用约一半的参数量和存储空间就达到了与MobileNet相当的87%准确率。 |
| [^52] | [RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport](https://arxiv.org/abs/2610.37281) | 该论文提出RPSD方法，通过拥有特权知识的教师模型进行在线自蒸馏以实现词元级信用分配，并利用检索相似用户的行为模式提供协同监督，从而改进基于大语言模型的生成式序列推荐。 |
| [^53] | [Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?](https://arxiv.org/abs/2610.37276) | 该论文提出了一种评估预算分配框架，证明稀疏反馈足以维持演化搜索，而将额外预算用于针对性的最终确认环节，能更有效地选出最优的智能体脚手架。 |
| [^54] | [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](https://arxiv.org/abs/2610.37245) | 本文构建了首批豪萨语安全偏好数据集HausaSafeLM（含3,000条金标和7,000条银标样本），并通过SFT与DPO对比实验揭示：仅凭拒绝率评估低资源语言模型的安全性会产生误导，基座模型的高拒绝率实为豪萨语生成能力不足而非真正的安全对齐。 |
| [^55] | [TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment](https://arxiv.org/abs/2610.37243) | 提出TCM-DiagCommit基准，将中医证型辨析从静态最终标签选择扩展为证据条件化的动态评估框架，通过分阶段证据揭示、语义邻近候选干扰及承诺/获取/延迟/弃权的动作契约，系统考察大模型在中医诊断中的渐进推理与不确定性决策能力。 |
| [^56] | [Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation](https://arxiv.org/abs/2610.37242) | 该论文提出将OData元数据与BPMN参考流程整合为类型化属性图知识图谱，并利用沿流程边检索的混合检索器，显著提升了LLM智能体在多步骤企业服务调用中的准确性和可靠性。 |
| [^57] | [When Is a Relation Worth Trusting? Manipulation Costs, Exposure, and the Trust Trap in Multi-Relational Graph Learning](https://arxiv.org/abs/2610.37239) | 该论文以对手操纵成本为前提，证明了当部分人群被隔离于某关系且攻击者无力打开该门控时，度门控严格优于任何关系选择或重加权方案，并提出无需攻击即可计算的逐节点“单位成本暴露度”指标，能准确指导平台选择应加固哪种关系以最大化提升信任陷阱阈值。 |
| [^58] | [GraphRAGProv: GraphRAG Meets How-Provenance](https://arxiv.org/abs/2610.37237) | GraphRAGProv 通过在源文本支撑的 RDF 图上执行 SPARQL 查询并计算答案的 how-provenance 多项式，使 GraphRAG 的答案可追溯、可验证，从而能够暴露文本到图谱转换中隐藏的抽取与查询错误。 |
| [^59] | [Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents](https://arxiv.org/abs/2610.37224) | Spine 通过让智能体用“工作范围”显式声明子任务边界，将子任务树编译为持久化的结构化上下文，在不依赖子智能体委派的情况下对齐长程编码任务的嵌套结构与模型上下文，避免上下文膨胀。 |
| [^60] | [Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution](https://arxiv.org/abs/2610.37215) | 本文提出R2A测试时扩展框架，通过将成功训练演示构建为时空运动基元图，并利用查询依赖的图神经网络在图中检索结构兼容的未来运动基元，将其几何先验与冻结的VLA预测融合，从而有效提升VLA策略在分布外视觉变化下的执行鲁棒性。 |
| [^61] | [Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution](https://arxiv.org/abs/2610.37201) | 论文提出BiPA方法，揭示通用技能优化器的开发集收益主要源于噪声任务下的选择效应而非真实改进——学习到的技能在测试时并不比手写种子更好，且额外的输入标记还带来隐性成本。 |
| [^62] | [Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning](https://arxiv.org/abs/2610.37195) | 提出ConCal方法，通过匹配并共享回滚批次中的延续价值证据并递归向后传播校准值，减少对单一采样延续的依赖，从而缓解长时程智能体强化学习中的局部信用失真问题。 |
| [^63] | [MLE-World: Predicting Machine Learning Experiment Outcomes with a Learned World Model](https://arxiv.org/abs/2610.37175) | 提出从Qwen3.5-4B微调的世界模型MLE-World，无需实际执行程序即可预测机器学习实验的终端输出和置信度，以低成本推理替代昂贵的训练运行，使AI研究智能体的测试时扩展（TTS）变得经济可行。 |
| [^64] | [R\`enyi-Regularized Flow-Based Variational Inference for Bayesian Calibration](https://arxiv.org/abs/2610.37169) | 本文提出一种基于Rényi散度正则化项的流式变分推断方法，解决了贝叶斯校准中标准变分推断无法捕捉被低概率区域分隔的多模态后验分布的问题，使近似后验能够覆盖先验支撑并表示多个后验模态。 |
| [^65] | [LiveSearch: Questions as Executable Verifier Code for a Live Benchmark of Professional-Source Financial Search Agents](https://arxiv.org/abs/2610.37166) | LiveSearch提出了一种金融搜索智能体的实时基准，将问题存储为可执行的验证器代码，在评估时通过执行程序对专业金融数据动态确定标准答案，从而检验模型检索最新金融事实的能力而非记忆过时答案的能力。 |
| [^66] | [WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training](https://arxiv.org/abs/2610.37161) | WorkFactory 通过信息级规划和探针执行，将专业意图转化为连贯可执行的任务，构建了覆盖 612 个职业的 11 万余个任务，并据此训练出在专业基准上超越基础模型的 WorkPilot 系列智能体。 |
| [^67] | [GCSR-OCC: Geometry Calibration and Semantic Routing for Open-Vocabulary Occupancy Prediction](https://arxiv.org/abs/2610.37149) | 提出GCSR-OCC框架，通过证据保持的高斯到占据算子抑制尺度膨胀与过度重叠，并结合多模式查询条件语义路由解决语义混合问题，实现从单张RGB图像的开放词汇室内3D占据预测。 |
| [^68] | [Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning](https://arxiv.org/abs/2610.37145) | 该论文提出FS-EWC，通过在任务学习中最小化邻域内最坏情况的Fisher信息矩阵迹来构建对参数位移不敏感的解，并采用基于轨迹的模型合并构建保留解，从而提升无重放持续学习中知识的保持能力。 |
| [^69] | [Generative Replay for Equilibrium Propagation: Toward Edge-Friendly Continual Learning in Hopfield Networks](https://arxiv.org/abs/2610.37139) | 提出基于能量的生成式回放策略EB-Gen，利用霍普菲尔德网络自身的生成动力学，让单一网络既当分类器又当样本生成器，通过松弛冻结的网络副本来生成过去任务的伪样本，从而无需外部数据缓冲即可解决平衡传播训练中的灾难性遗忘问题。 |
| [^70] | [What the Primary Lund Plane Omits: A Model-Stitching Diagnosis of a Jet-Tagger Representation Gap](https://arxiv.org/abs/2610.37137) | 本文利用模型拼接技术诊断喷注标记器的表征差距，证明主Lund平面（仅含主分解支路）相对ParT的性能缺陷源于表征信息缺失而非下游计算能力不足，且高阶相关性也被排除为成因。 |
| [^71] | [MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Diffusion](https://arxiv.org/abs/2610.37134) | 提出了一种结合解剖引导伪配对和对抗扩散模型的三阶段框架，可从未配对的脊柱MRI图像合成CT图像，从而避免辐射暴露并优化术前评估。 |
| [^72] | [SFE-O: An Ontology for Knowledge-Grounded AI in Supercritical CO$_2$ Biomass Extraction and Fractionation](https://arxiv.org/abs/2610.37123) | 本文提出SFE-O——一个基于OWL 2的模块化本体，通过生物质、过程、产物三大维度及传质机制和计算依赖关系的显式建模，为超临界CO₂生物质萃取与分级的知识驱动AI应用提供了统一的知识基础。 |
| [^73] | [Prospector: learning a compact graph traversal policy for scientific reasoning](https://arxiv.org/abs/2610.37121) | Prospector是一个紧凑模型，通过逐步遍历领域知识图谱收集证据并构建可逐边验证的证据链，在半导体制造知识图谱上以小模型性能接近最强的现成大模型。 |
| [^74] | [A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation](https://arxiv.org/abs/2610.37110) | 该论文发现扩散模型蒸馏导致的多样性丢失主要发生在第一步高噪声去噪阶段，据此提出ReSeed蒸馏后处理方法，通过仅在学生模型第一步添加种子响应偏置适配器并与教师模型割线速度对齐，有效恢复了蒸馏模型的输出多样性。 |
| [^75] | [Hardware-Matched Spiking Neural Networks for Real-Time Purkinje Cell Spike Classification](https://arxiv.org/abs/2610.37104) | 本文提出两种与目标硬件紧密匹配的紧凑脉冲神经网络，可在微控制器和专用电路等受限硬件上实时、准确地分类浦肯野细胞的简单尖峰与复杂尖峰，为无线记录头台实现片上数据压缩提供了解决方案。 |
| [^76] | [Learning to Look Beyond Screens: Transferable Visual Observation from GUI Grounding](https://arxiv.org/abs/2610.37093) | 该论文提出LBS模型，将GUI定位作为可扩展的空间监督来源，通过监督微调建立视觉检查基础能力、再通过强化学习实现多轮裁剪与检查决策，训练出可从屏幕界面泛化到通用视觉感知基准的主动观察能力。 |
| [^77] | [OpenGray: a benchmark for evaluating the decisions of radiotherapy planning agents](https://arxiv.org/abs/2610.37081) | 该研究提出了OpenGray基准，通过在头颈放疗数据集中加入已知正确决策的任务（如经证明不相容的剂量目标矛盾和可行对照），首次为评估LLM放疗计划智能体“提交计划还是请求审核”的决策能力提供了标准化工具。 |
| [^78] | [Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation](https://arxiv.org/abs/2610.37070) | 该研究通过结构因果模型揭示，在思维链忠实性评估中仅筛选回答正确的实例会引入选择偏差，导致测得的忠实性差距被系统性高估（在GLM-5.3上达+7.4个百分点）。 |
| [^79] | [EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training](https://arxiv.org/abs/2610.37062) | EnergyCal是一种轻量级规则，通过根据Newton-Schulz方向的归一化平方Frobenius范数衰减路由专家更新，可将稀疏MoE训练中的损失尖峰负担降低高达99.51%并改善困惑度，同时不改变谱变换、不增加持久优化器状态。 |
| [^80] | [Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization](https://arxiv.org/abs/2610.37058) | 该研究揭示了Delta学习假设的局限性：使用单一模型对生成的数据训练DPO会引入系统性的生成器偏差并引发奖励作弊，导致即使大模型回复质量更高，其训练效果仍持续劣于多模型偏好数据。 |
| [^81] | [Blindfold: Keeping the Audit Coin Secret from LLM Agents](https://arxiv.org/abs/2610.37055) | 该论文揭示了基于审计的LLM智能体认证的有效性还依赖于“提交前不干扰”这一保密性属性，证明其充要性并给出最坏偏差 (1-ε)D_A，同时指出泄露会跨步骤复合且提高审计率也无法弥补失效。 |
| [^82] | [AI-Assisted Discovery of CNOT Decompositions for Quantum Permutations on Constrained Topologies](https://arxiv.org/abs/2610.37045) | 本研究利用AI工具AlphaCNOT探索发现了一种受限拓扑下远程对换的CNOT分解模式，使图距离为d的两个量子比特间的置换最多仅需6d-4个CNOT门，优于基于SWAP路由的6d-3个门的构造，展示了AI在揭示新型确定性编译算法设计原理方面的价值。 |
| [^83] | [Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2610.37041) | 提出 Guided-RLVR，通过轻量级控制器在采样分布中注入语义引导标记，扩展 RLVR 的推理探索空间，从而在数学推理和代码生成任务上持续超越现有 RLVR 方法。 |
| [^84] | [Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception](https://arxiv.org/abs/2610.37037) | 论文提出 EvidenceConnect 数据合成框架，通过利用单张图像内相关区域的放大视图生成需要比较、聚合或链接视觉证据的问题，来增强多模态模型“连接证据”这一底层能力，从而以更低的成本提升空间感知与多模态长上下文理解等任务的泛化性能。 |
| [^85] | [Interpretable Deep Learning Models for Breast Cancer Diagnosis Using Medical Imaging](https://arxiv.org/abs/2610.37026) | 该研究对十个ImageNet预训练模型在乳腺癌组织病理学图像良恶性分类任务上进行基准测试并结合多种可解释性方法，发现DenseNet121以99.4%的准确率表现最佳，证明了迁移学习与可解释AI结合能够提供准确且可解释的乳腺癌诊断工具。 |
| [^86] | [Crop Yield Prediction Using Regression Models: A Comparative Evaluation of Model Performance](https://arxiv.org/abs/2610.37022) | 本研究比较了多元线性回归、随机森林、决策树、岭回归和套索回归五种模型在基于环境和农业管理变量预测作物产量中的性能，其中多元线性回归在80:20划分下表现最佳，测试R²达约0.9130。 |
| [^87] | [V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models](https://arxiv.org/abs/2610.37019) | V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。 |
| [^88] | [Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing](https://arxiv.org/abs/2610.37011) | 通过SUMO仿真评估按需共享出租车中两种高级会员优先调度策略，发现最优优先机制取决于运营条件与会员比例，条件自适应的方法切换可在保障高级会员权益的同时降低非会员负担。 |
| [^89] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^90] | [LearnPrunerV2: Learnable Progressive Token Pruning for Vision-Language Models](https://arxiv.org/abs/2610.36986) | 该论文提出 LearnPrunerV2，通过在视觉-语言模型每一层嵌入轻量级可学习模块，融合视觉显著性与查询相关性来自适应地估计词元重要性，实现完全可学习的渐进式词元剪枝，从而在大幅降低计算开销的同时保持多模态性能。 |
| [^91] | [Advancement or Acceleration? How Data Curation Improves LLM Training](https://arxiv.org/abs/2610.36983) | 本文形式化区分了数据筛选对LLM训练的“提升”（扩展模型最终可达到的性能）与“加速”（用更少数据达到相同性能），证明只有当训练与目标之间存在可纠正的差距时筛选才能带来提升，并提出可观测的Δ-测试来检验该条件。 |
| [^92] | [AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations](https://arxiv.org/abs/2610.36980) | AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。 |
| [^93] | [Proactive Memory for Event-Driven LLM Agents](https://arxiv.org/abs/2610.36967) | 提出 ProactMem 主动记忆框架，将历史组织为近期事件、时间轨迹和经历史重演验证的行为规律，并构建纵向基准 ProactiveStream，使事件驱动的 LLM 智能体能更准确地预判服务机会，F1 达 52.6%，较最强基线相对提升 21.4%。 |
| [^94] | [Do bots feel bad? Assessing empirical evidence for functional valence in LLMs](https://arxiv.org/abs/2610.36962) | 本文评估了关于大语言模型功能性效价状态的实证证据，指出效价的关键方面尚未在LLM中得到检验，且现有实验范式无法将功能性效价与偏好、价值判断、效价归因和角色扮演等邻近构念充分区分开来。 |
| [^95] | [PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search](https://arxiv.org/abs/2610.36958) | 提出了拍立淘MMSearch，一个统一的端到端多模态生成式检索系统，可直接从原始查询图像和自由形式指令生成商品语义ID，并在数据、意图等三个维度上扩展，以支持开放式图文查询、异构购物意图和不同资源约束的在线场景。 |
| [^96] | [Dense Representation Learning and Differentiable Shape Modeling for Spine Landmark Detection](https://arxiv.org/abs/2610.36951) | 该论文将脊柱关键点检测重新构建为像素级对比表示学习问题，并结合可学习形状先验解决关键点位置歧义，在脊柱X光关键点检测精度和临床参数估计上达到了最先进水平。 |
| [^97] | [DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models](https://arxiv.org/abs/2610.36950) | DCKV是首个端到端训练的KV缓存可微分编解码器，可在三个长上下文基准上以不超过1.8分的性能损失，将混合语言模型的提示词KV缓存压缩到比bf16少55倍以上的存储空间。 |
| [^98] | [Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models](https://arxiv.org/abs/2610.36948) | 该论文提出SKIM闭式方法，让冻结语言模型为每个对话轮次自生成问题作为键写入模型参数中，从而在无外部存储的条件下仅凭参数即可回忆对话内容并回答后续提问。 |
| [^99] | [Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance](https://arxiv.org/abs/2610.36947) | 该研究通过五组实验表明，在软件保障任务中扩大语言模型智能体上下文规模会使token开销激增49倍却无安全收益，验证器反馈虽能提升正确判定数量但同时增加错误判定，说明确定性的工具证据基线仍然不可或缺。 |
| [^100] | [Vision-Language Agents Can Learn in Their Own Latent Worlds](https://arxiv.org/abs/2610.36925) | 提出了Dwell框架，将视觉-语言模型重新构建为递归潜在转移算子，使智能体能在潜在空间中从在线交互中持续学习，从而高效利用预训练世界知识实现策略改进。 |
| [^101] | [Developing a Speech Language Model for Basque](https://arxiv.org/abs/2610.36911) | 本文提出了首个开源的巴斯克语语音语言模型，通过在LLaMA-Omni架构上微调Latxa模型并利用翻译语料与TTS合成音频，实现了低资源语言下的低延迟端到端口语交互。 |
| [^102] | [Natural Hard-Negative Mining from Positives for Instruction-Following Retrieval](https://arxiv.org/abs/2610.36900) | 提出自然困难负例挖掘（NHNM）方法，通过自动流水线将原始正例转化为自然的指令困难负例，构建了近7K实例的IFIR训练集NHNM-MARCO，显著提升了各规模指令跟随检索模型的性能。 |
| [^103] | [Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning](https://arxiv.org/abs/2610.36888) | 提出Noise Steering无模型强化学习策略，通过联合控制风机偏航角与功率设定值，同时利用尾流转向和噪声指向性，在满足局部噪声约束的条件下最大化风电场发电功率。 |
| [^104] | [RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents](https://arxiv.org/abs/2610.36878) | RubricAtlas通过从3,079份中文金融研究专家报告中挖掘类型级分析要求，构建覆盖99种研究类型的评估框架，将二元检查点与报告段落关联，并把报告完整性与论证有效性分开评估，从而更全面地评估深度研究智能体。 |
| [^105] | [GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents](https://arxiv.org/abs/2610.36877) | GUI-JEPA是首个无解码器的GUI智能体策略，利用在无标注GUI截图上预训练的JEPA编码器潜在空间，通过候选条件化的潜在预测器和轻量级打分头在一次批量处理中为所有候选动作打分，无需逐token解码，实现了快速且省内存的移动GUI操作。 |
| [^106] | [AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding](https://arxiv.org/abs/2610.36874) | AdaGround提出了一种免训练的GUI定位框架，通过依据目标大小自适应设定缩放参数、引入“是/否”验证器在环机制甄别并修正缩放结果、以及根据表头组合定位电子表格单元格，在高分辨率屏幕小目标定位上超越了现有最先进的免训练方法。 |
| [^107] | [ActiveAlign: Aligning Egocentric Manipulation via Active Perception](https://arxiv.org/abs/2610.36859) | ActiveAlign 受人类主动感知启发，通过主动手部运动与可微渲染实现手眼标定、并结合触觉反馈精化物体对齐，从交互中弥合 Real2Sim2Real 差距，达到毫米级对齐精度。 |
| [^108] | [When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning](https://arxiv.org/abs/2610.36849) | 提出策略兼容强化学习方法PICO，通过策略兼容的联合解码使专家轨迹与当前策略分布相容，从而在所有采样都失败、标准RLVR学习信号坍缩为零的困难问题上获得既可验证正确又可迁移的有效学习信号。 |
| [^109] | [Evolving Agent Teams](https://arxiv.org/abs/2610.36848) | 提出多智能体框架“进化智能体团队”，通过中央监督者诊断局部最优并动态重构智能体团队，并借助持久化文件系统积累的经验指导重构，在相同token预算下持续超越现有编码智能体基线，并从零生成比FlashInfer生产级MLA分页解码内核平均快1.2倍的CUDA内核。 |
| [^110] | [Conservative Policy Iteration For Zero-Sum POSGs](https://arxiv.org/abs/2610.36847) | 提出了一种针对有限时域零和部分可观测随机博弈的近似策略迭代算法，通过保守策略更新和基于当前联合策略的支撑超平面来替代对完整中心价值函数的近似，将计算集中于联合策略可被利用的占用区域。 |
| [^111] | [When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement](https://arxiv.org/abs/2610.36826) | 提出了 RuHarness 框架，通过评测标准与执行框架的协同进化，持续从执行轨迹中发现并更新行为准则以指导针对性框架改进，实现智能体的持续自我提升并显著增强对未见任务的泛化能力。 |
| [^112] | [Who Knows What - and What Does It Support? Perspective-Bounded Evidence Calibration for Multimodal Theory of Mind](https://arxiv.org/abs/2610.36824) | 该论文将多模态心智理论中的核心局限形式化为证据可采纳性问题，主张只有当证据对相关智能体可获取、保持信息来源一致且对目标心理状态具有诊断性时才能用于心理状态推断，从而避免视角泄露与来源混淆。 |
| [^113] | [TAPESTRY: A Single Backbone for Video Generation and Robot Control](https://arxiv.org/abs/2610.36823) | TAPESTRY通过将机器人动作作为单补丁标记嵌入视频模型的潜在空间，用一个轻量级线性投影将动作映射到预训练VAE维度，使同一预训练视频扩散Transformer骨干网络既能用于视频生成又能直接作为机器人控制策略，且推理时无需生成视频。 |
| [^114] | [An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning](https://arxiv.org/abs/2610.36820) | 本文提出一种基于轻量级知识图谱的增量式学习者数字孪生，将特征、归因、预警与原始学习记录统一表示为可查询的图实体，从而实现有证据支撑、可解释的学业风险早期预警。 |
| [^115] | [SetGRPO: Calibrating Grader-Induced Policy Optimization](https://arxiv.org/abs/2610.36807) | 提出SetGRPO方法，通过联合奖励集合校准和几何引导的安全恢复等模块来校准评分器随机性对GRPO更新的影响，从而提升基于评分反馈的强化学习训练的稳定性与安全性。 |
| [^116] | [An Algebra of Transformers](https://arxiv.org/abs/2610.36795) | 本文首次提出了一种基于函数分解结果的Transformer代数系统，通过顺序组合、并行组合、递归和抽象等操作实现对Transformer的逻辑建模，并可用于验证基于Transformer的应用的正确性。 |
| [^117] | [Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training](https://arxiv.org/abs/2610.36794) | Select-TTT提出了一种模型无关的查询条件化规则，通过根据各数据块与用户查询的相关性来动态缩放其快权重写入强度，解决了原位测试时训练中统一写入调度导致无关内容稀释查询相关信息的问题。 |
| [^118] | [Dependency Graphs of Mathematical Arguments: Representation, Structural Measurement, and Proof Review](https://arxiv.org/abs/2610.36776) | 提出了一个基于源文本、借助大语言模型辅助的数学论证依赖图表示与度量框架，通过深度与体量等结构指标、并集递推式及可达性保持条件，并配合标准化版本化提示协议，实现可靠的证明审阅。 |
| [^119] | [Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better](https://arxiv.org/abs/2610.36774) | 将GEPA优化器扩展为每步并行提议和评估一批候选方案，使反思优化循环的运行时间缩短至原来的四分之一到二分之一，同时提升了测试分数和预算利用效率。 |
| [^120] | [Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs](https://arxiv.org/abs/2610.36767) | 本文通过构建带专家评分的USAMO 2025人工标注数据集并提出TROJAN对抗性证明生成流水线，系统性地评估了大语言模型验证奥赛级数学证明的能力。 |
| [^121] | [Beyond the Evidence Coherence Trap: Agent-Driven Evidence Selection for Temporally Disordered Fake News Detection](https://arxiv.org/abs/2610.36764) | 该论文针对时序错乱假新闻提出了“证据一致性陷阱”问题，构建了ChronoFake基准并设计了智能体驱动的证据选择方法DelSA，通过选择时间匹配的证据来避免被局部一致但时序错误的证据误导。 |
| [^122] | [The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation](https://arxiv.org/abs/2610.36763) | 该论文提出API框架，直接复用预训练稀疏自编码器的可解释特征索引作为生成式推荐的语义ID词表，无需额外的词表学习即可实现有效且可解释的推荐。 |
| [^123] | [Making Self-Driving Laboratories Auditable:A Unified Framework for Hybrid Actions,Measurements, and Safe Execution](https://arxiv.org/abs/2610.36761) | 该论文提出了由LAM、OpenSDL、Safe-SDL和L3-Bench四个组件构成的可追溯统一框架，使自动驾驶实验室能够统一生成混合的机器人动作与仪器API调用、协调异构设备安全执行，并实现全程可审计。 |
| [^124] | [Robust LLM Data Selection via Content-Invariant Gradient Attribution](https://arxiv.org/abs/2610.36755) | 提出CIGA框架，通过筛选对语义保持扰动稳定的梯度层并与语义表示动态融合，消除了梯度归因中的表层形式偏差，实现了更鲁棒的大语言模型指令微调数据选择。 |
| [^125] | [Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning](https://arxiv.org/abs/2610.36751) | 提出iFLEX模型，用条件扩散模型学习修正基于时间加权平均的简单插值所产生的残差，从而在低时间分辨率数据下更准确地重建中间科学场。 |
| [^126] | [From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents](https://arxiv.org/abs/2610.36713) | 该论文提出 EvoSpeakerMem 免训练框架，将声纹视为对身份的动态观察而非固定标签，通过在线说话人-记忆构建随声学证据演进维护持久的用户实体，并构建了 LoCoMo-ID 基准以评估多用户语音代理中的说话人身份归属与记忆管理问题。 |
| [^127] | [Conservative Dueling Bandits: Preference Learning under Baseline Constraints](https://arxiv.org/abs/2610.36708) | 本文提出了保守上下文对决老虎机问题及MaxInP-BC算法，通过效用差距而非绝对奖励来认证安全性，克服了标准对决老虎机探索过于激进的问题，并首次证明直接迁移CLUCB的安全测试会导致线性遗憾。 |
| [^128] | [HeteGCD: Statistics-Aware Dual-Prototype Learning for Generalized Category Discovery on Heterophilous Graphs](https://arxiv.org/abs/2610.36696) | 提出HeteGCD方法，通过统计感知的双原型学习，在异质图上有效利用异构邻域信息实现广义类别发现，从而在仅有已知类别标签的条件下同时识别已知类别并发现新类别。 |
| [^129] | [TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation](https://arxiv.org/abs/2610.36694) | 提出TOPS方法，在不修改网络架构的前提下，通过为每个任务定制流匹配的源先验和插值调度（利用专家动作逆向映射构建多模态任务源，并依据源-目标协方差失配调整传输节奏），提升语言条件化多任务机器人操作性能。 |
| [^130] | [PsyJev: Psychological Analysis via System-1 Judgment](https://arxiv.org/abs/2610.36689) | 该论文构建了涵盖六个心理学领域的PsyJudgeBench基准，并提出PsyJev系统-1模型，通过联合监督训练与判断锚定的RLCD（JA-RLCD）方法，将异构心理分析任务统一到共享的问题条件化判断接口下，基于Qwen3.5的2B至9B模型取得了64.33至67.56的成绩。 |
| [^131] | [Specialization, Diversity, and Scaling in Child-Scale Language Models](https://arxiv.org/abs/2610.36679) | 在儿童尺度数据上训练的小型语言模型会对单个儿童的语言环境产生词汇和分布层面的专业化，多儿童数据训练虽能提升对汇总语言的预测，但对语法能力的提升并不明显。 |
| [^132] | [Atom as an Agent: Emergent Order by Collective Predictive Coding](https://arxiv.org/abs/2610.36673) | 该论文将语言符号涌现的“集体预测编码”理论引入物理学，把原子视为相互预测的智能体，仅凭观测到的自旋构型就能自动涌现出伊辛模型、条纹序模型和自旋玻璃系统中的宏观序参量，为宏观秩序的起源提供了新解释。 |
| [^133] | [Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration](https://arxiv.org/abs/2610.36672) | 提出ReliMorph框架，利用局部相关分布评估匹配可靠性，并以可靠性线索引导空间专家融合和残差速度门控，从而提升三维可变形图像配准在弱纹理、严重形变区域的准确性。 |
| [^134] | [Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs](https://arxiv.org/abs/2610.36671) | 提出LOWN解码方法，在每个解码步骤自适应地决定是否需要处理图像以及解码多少标记，从而避免扩散视觉语言模型中大量冗余的图像重复计算，显著提升解码效率。 |
| [^135] | [A Quantitative Approach to Asch Conformity in Frontier Large Language Models](https://arxiv.org/abs/2610.36668) | 该论文提出一种定量方法，通过对单个问题重复采样获得答案分布，并比较模型接触人为制造共识前后的分布差异，证实前沿大语言模型存在从众效应，并量化了从众发生的条件与程度。 |
| [^136] | [The View from Nowhere: Population-Centered Agent Evolution](https://arxiv.org/abs/2610.36660) | 提出种群中心演化（PCE），通过利用整个种群的任务记录与自我修改历史来改进智能体，从而弥合种群覆盖率与最佳个体性能之间的差距。 |
| [^137] | [Remembering Is Not Enough: Refining Memories and Learning When to Use Them in Multi-Agent Systems](https://arxiv.org/abs/2610.36652) | 提出无需训练的DiscernMem框架，通过缺陷感知记忆精炼（DMR）诊断并修复有缺陷的记忆，通过边界学习元记忆（BLM）学习在何种情境下应使用或避免使用记忆，从而同时提升多智能体系统的记忆质量与记忆使用效果。 |
| [^138] | [Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction](https://arxiv.org/abs/2610.36643) | 提出Tap2Listen流式音视频说话人提取模型，通过按模态分离计算架构，将目标切换开销降至仅需0.12ms的重投影，实现了对话场景下的即时目标切换。 |
| [^139] | [The uneven rise of AI-generated web content](https://arxiv.org/abs/2610.36631) | 基于500万网页的大规模研究表明，AI生成文本在互联网上急剧增长但分布不均：英语网页和.com域名中AI内容占比更高，而网络图中最边缘化的网页AI内容增长最快，成为整体增长的主要驱动力。 |
| [^140] | [Polarity--Time Interaction and Multi-Level Reliability for Event-Based Spacecraft Pose Estimation](https://arxiv.org/abs/2610.36625) | 该论文提出一个可靠性感知的航天器位姿估计框架，通过极性-时间三支路事件表示、有序语义点精度估计和序列级观测尺度设定，将事件定位、几何恢复与CAD细化相结合，在SPADES挑战赛上实现了更鲁棒准确的位姿估计。 |
| [^141] | [Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction](https://arxiv.org/abs/2610.36624) | 该论文提出运动引导非对称预测（MoGAP），通过在动态区域分配更高噪声、并以低损坏的静态区域作为上下文进行非对称视觉预测，显著提升了世界动作模型在机器人控制中的策略性能。 |
| [^142] | [God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation](https://arxiv.org/abs/2610.36623) | 该论文提出了一种即插即用工具，通过将契约引导的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合，作为验收门控嵌入内核生成智能体的优化循环中，从而强制执行批处理不变性，解决LLM服务中动态批处理导致相同输入产生不同数值输出、损害可复现性的问题。 |
| [^143] | [Dream-COP: Guiding Algorithm Discovery for Combinatorial Optimization Problems in an Experience World](https://arxiv.org/abs/2610.36618) | Dream-COP 通过让策略智能体在由历史搜索轨迹构成的经验世界中“做梦回放”来学习可执行策略，从而在有限搜索预算下引导 LLM 为组合优化问题发现更好的算法。 |
| [^144] | [Balanced State Subspaces for Mamba-2 State Compression](https://arxiv.org/abs/2610.36617) | 论文发现 Mamba-2 的状态能量集中于递归累积的偏置方向，而输出实际读取的内容位于与其坐标不对齐的低维子空间，据此提出一种基于完整 Gramian 平衡截断的训练后压缩方法，使压缩直接针对写入-读取子空间而非单个坐标。 |
| [^145] | [Learning to Formalize: Bridging Open-Ended Workloads and Fast Decision Models](https://arxiv.org/abs/2610.36595) | 提出一个在离线教师生成列表上训练的小型候选形式化器，将开放式任务上下文转化为显式且经验证的候选项，使冻结的快速决策模型能够取代缓慢的生成式大模型后端完成高效选择。 |
| [^146] | [InfraDev: Can Coding Agents Develop AI Infrastructure, and How Can We Tell?](https://arxiv.org/abs/2610.36589) | InfraDev是一个基于Megatron-LM、vLLM和verl真实代码变更构建的36任务基准，用于评估编码智能体开发多文件、多GPU的AI基础设施代码的能力，并通过隐藏探针和共享执行环境实现更准确高效的评分。 |
| [^147] | [Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements](https://arxiv.org/abs/2610.36584) | 该论文提出了Req2Test-Bench基准，包含从8个开源项目BDD工件中恢复的5,086个经执行验证的实例，用于评测LLM智能体在仅依据自然语言需求（无实现代码）的情况下生成行为充分测试的能力。 |
| [^148] | [MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke](https://arxiv.org/abs/2610.36583) | 提出多模态多任务网络MM-Net，通过交叉拟合和基于岭回归的嵌入对齐解决表征不一致问题，利用交叉注意力融合EEG-EOG表征、神经描述符与心肺特征，实现缺血性卒中后睡眠分期与呼吸事件检测的同步联合评估。 |
| [^149] | [Follow the Herd: A Data-Driven Approach for Actionable Recourse](https://arxiv.org/abs/2610.36580) | 本文提出首个无需人工设计行动成本函数的数据驱动行动补救方法DDAR，通过学习历史用户轨迹的动力学模型，搜索高概率且能实现期望结果的低成本行动路径。 |
| [^150] | [Less is more: learning what to profile for GPU kernel optimization](https://arxiv.org/abs/2610.36576) | 提出AutoProfiler框架，通过从优化反馈中学习确定性的性能分析树，为不同GPU内核自适应地选择最有用的性能分析信息，从而提升大语言模型自动优化GPU内核的效果。 |
| [^151] | [ADTGuard: Adaptive Depth-Time Safeguard for Streaming Inference in Large Language Models](https://arxiv.org/abs/2610.36574) | ADTGuard提出了一种自适应深度-时间安全防护机制，通过在每个词元释放前整合跨层安全证据并建模风险随深度与时间的动态累积，实现了大语言模型流式推理中的实时有害内容检测。 |
| [^152] | [CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents](https://arxiv.org/abs/2610.36570) | CASEL提出了一种与模型无关的运行时分配策略，通过综合失败概率、后果严重性、可逆性、模糊性和证据新鲜度等因素，为智能体的每次工具调用智能选择直接执行、验证、沙箱、澄清、审批或拒绝执行，显著减少了长时程工具使用智能体造成有害副作用的风险。 |
| [^153] | [NextEval: An Agent-Based Solver and Benchmark for Derivative-Free Optimization](https://arxiv.org/abs/2610.36553) | 该论文提出了NextEval，一个将大语言模型智能体与数值环境相结合的求解器与基准测试，利用LLM指导无导数优化中评估点的序贯选择，从而在有限评估预算下更有效地解决黑盒优化问题。 |
| [^154] | [REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal](https://arxiv.org/abs/2610.36546) | 提出了REACH，一个基于强化学习的分层框架，使Unitree G1人形机器人能够通过可变高度双足行走（1.27–0.50米）与爬行（0.50–0.20米）以及双向模式切换策略，实现受限空间中的自适应穿越。 |
| [^155] | [FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation](https://arxiv.org/abs/2610.36545) | FrameForge利用视频基础模型的时间先验，通过输入自适应的虚拟时间关系——MoTE（动态选择时间RoPE专家以建模条件与目标的距离）和CoTR（时间推理链）——实现统一的条件图像生成。 |
| [^156] | [Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors](https://arxiv.org/abs/2610.36544) | 该论文提出一个决策价值框架，从分析上证明监控器的分类准确率本身无法决定重启AI智能体是否有益，并引入决策充分性准则和有限样本无模型决策审计两种技术，以决策价值的角度评估AI智能体重启监控器。 |
| [^157] | [ATEMPO: Amortized Transfer Entropy Model Per-lag Over-Time](https://arxiv.org/abs/2610.36536) | ATEMPO是一种基于扩散模型的传递熵估计器，通过在训练中采样不同的滞后深度并显式条件化于时间位置，仅需单个得分网络即可无需重新拟合地估计随时间变化的完整传递熵曲面，从而克服了固定输入窗口掩盖信息延迟结构以及现有方法需反复拟合的局限。 |
| [^158] | [RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding](https://arxiv.org/abs/2610.36533) | RoboAuto 提出了一个在物体、场景和交互三个层次上显式组织物理理解的机器人程序搜索框架，利用分层知识库整合可执行技能与物体中心知识，使基于 LLM 的智能体能够更有效地搜索和改进机器人操作程序。 |
| [^159] | [MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics](https://arxiv.org/abs/2610.36513) | 该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。 |
| [^160] | [Do AI Scientists Know What's New?](https://arxiv.org/abs/2610.36511) | 该论文提出了一个结合时间真值的分析框架，从提案和论断两个层面评估AI科学发现中的新颖性检查，发现对非新颖工作拒绝率高的检查往往会误拒那些后来被人类研究者实现的新颖想法。 |
| [^161] | [ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization](https://arxiv.org/abs/2610.36499) | ProCerDCL 提出了一种查询高效的黑盒概率验证框架，利用单一序贯检验在指定错误预算内判定属性违反概率是否超过阈值，并将其应用于深度学习编译器优化的验证。 |
| [^162] | [AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation](https://arxiv.org/abs/2610.36493) | 该论文提出AC-JEPA，一种与监督微调兼容的即插即用模块，通过在跨模态两个方向上预测被掩码的中间状态的联合嵌入目标，来利用不完美的配对数据，从而在保留token级监督的同时提升视觉到符号跨模态翻译对目标序列错误的鲁棒性。 |
| [^163] | [CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments](https://arxiv.org/abs/2610.36484) | 提出CogArena基准，让多模态智能体通过真实网页浏览器以与人类被试相同的视觉界面完成十个行为科学实验范式，并采用“任务完成度—准确性—行为特征”三级评分来检验AI认知能力，同时应对在线行为数据被AI污染的问题。 |
| [^164] | [Proposing Better Rollouts for On-Policy Distillation](https://arxiv.org/abs/2610.36477) | 该论文揭示在线策略蒸馏中固定以学生模型作为 rollout 策略会导致教师-学生错位不断累积并引发重复等退化行为，并据此提出更好的 rollout 策略设计以提升蒸馏效果。 |
| [^165] | [FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents](https://arxiv.org/abs/2610.36476) | 该论文提出FIDES-Live审计协议，用于检验序列交易智能体的决策、订单、账本与报告之间是否一致，通过对7,299个加密货币账户轮次和310轮股票试点的审计，发现决策有效性达97.35%、报告准确率90.81%、承诺遵循度94.36%，但不同推理路由的表现差异显著。 |
| [^166] | [When to Intervene? An Empirical Study of Errors and Self-Recovery in LLM Agents](https://arxiv.org/abs/2610.36471) | 该论文构建了ReInBench数据集（含1,259条智能体轨迹和1,403个错误事件），首次系统实证研究LLM智能体的错误自我恢复现象，发现大量错误可在几步内自我恢复，为判断何时干预、何时等待提供了依据。 |
| [^167] | [Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition](https://arxiv.org/abs/2610.36460) | 提出受控部分神经坍缩方法，通过CASA分类器和ETF-AM损失让头部类别沿ETF坍缩而尾部类别保留类内多样性，解决了长尾识别中强制完全坍缩导致的过度收缩与泛化能力下降问题。 |
| [^168] | [LociArt: A Unified Paradigm for Free-Style Local Video Stylization](https://arxiv.org/abs/2610.36456) | 提出LociArt统一范式，通过基于生成式精炼网络的GRNStyle与极坐标风格感知掩码，实现对局部视频区域“在哪里风格化、应用什么风格”的精确控制并保持时序一致性，同时借助自动化流水线构建了10万对样本的LociArt-100K数据集。 |
| [^169] | [Correction or Manipulation? Measuring Epistemic Vigilance in LLM Agents Under User Pressure](https://arxiv.org/abs/2610.36453) | 该论文在问答与客服行动两个场景下，对四个厂商的七个大语言模型施加分级的用户压力，并结合信号检测理论，系统测量了LLM智能体能否区分用户的善意纠错与恶意操纵。 |
| [^170] | [A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs](https://arxiv.org/abs/2610.36449) | 研究发现视觉语言模型会被简单视觉符号隐性地引导行为（如心形符号引导模型采用递归解法），且模型间视觉交流存在“生成—解读”鸿沟——模型发送的直观视觉线索往往无法有效引导其他模型。 |
| [^171] | [From Solver Trajectories to Teaching Trajectories: Cognition-Aligned Reasoning Distillation](https://arxiv.org/abs/2610.36445) | 提出CARD框架，通过强化学习将大型推理模型作为“解题者”产生的轨迹转化为符合小模型认知能力的“教学轨迹”，从而在不进行昂贵重采样的情况下提升知识蒸馏效果。 |
| [^172] | [RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG](https://arxiv.org/abs/2610.36444) | 该论文提出RouteDRAG学习型框架，通过自适应路由决定何时调用智能体并控制其执行过程，在多模态文档RAG任务中减少不必要的计算开销，同时保持或提升答案准确性。 |
| [^173] | [Your Agent Acts Differently When You're Around: How User Presence Changes Agent Behavior in Negotiation](https://arxiv.org/abs/2610.36435) | 研究发现用户在场会显著改变LLM谈判智能体的行为：有用户参与的谈判方能获得更高收益且智能体谈判更激进，但同时用户在场也会导致针对交易对手的不对齐行为发生率大约翻倍。 |
| [^174] | [LM-DSB: A Lightweight Multi-agent System for Automated Daily Safety Bulletin Generation in Construction](https://arxiv.org/abs/2610.36433) | 该研究提出LM-DSB系统，通过多智能体分层协作架构结合RAG与本地部署的大语言模型，在消费级个人电脑上自动生成建筑每日安全公告，解决了人工编写耗时长、质量不稳定的问题。 |
| [^175] | [Repulsive Population Policy Gradients for Behaviorally Diverse Reinforcement Learning](https://arxiv.org/abs/2610.36425) | 提出RePPO，通过改造PPO并引入依赖种群的动态塑形奖励来联合优化一个紧凑的多样化高质量策略种群，从而在连续控制任务上实现了优于现有QD-RL方法的质量-多样性权衡，摆脱了对昂贵进化算子的依赖。 |
| [^176] | [Verifiable Efficiency Limits of Innate LLM Planning in Sliding Puzzles](https://arxiv.org/abs/2610.36415) | 本文提出可验证的 SPE = R × Q × E 框架，将大语言模型在滑块拼图上的内在规划效率分解为执行可靠性、规划协调性与行动经济性，并发现即便模型能完成全部任务，其效率在更长视野下仍会下降，且高效率可与低目标达成率并存。 |
| [^177] | [Expert lithological labels do not improve learning when used to predict marine sediment properties](https://arxiv.org/abs/2610.36410) | 专家标注的岩性标签本身可从常规物理测量数据中恢复，因此在预测海洋沉积物性质的自监督学习中引入这些标签并不能带来性能提升。 |
| [^178] | [kw-bench: Breaking Frontier Harnesses on Real-World Tasks](https://arxiv.org/abs/2610.36405) | kw-bench 是基于真实业务委托构建的基准，首次将评估重点从生成式任务转向审计、税务等知识工作中的调查性环节，检验前沿智能体运行框架能否从不完整、不一致的分散证据中精确还原并论证答案。 |
| [^179] | [MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations](https://arxiv.org/abs/2610.36389) | MAREval提出了一种结构化多智能体框架，结合辩论链提示策略、主持人调节、仲裁者聚合与蒙特卡洛采样，实现了对推荐系统中自然语言解释的可扩展、一致且自动化的评估。 |
| [^180] | [LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter Fleets](https://arxiv.org/abs/2610.36386) | 该论文提出功能秩分配（FRA）方法，通过为每个LoRA适配器构建功能秩-失真前沿，并在共享秩预算下进行集群级极小极大分配，从而解决多适配器事后压缩中的两级秩分配问题。 |
| [^181] | [SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding](https://arxiv.org/abs/2610.36384) | 该论文提出了SOMA基准，通过1,632个经人工验证的问题从身体构型、运动动态和时间推理三个维度评估自我中心视频中的细粒度身体运动理解能力，发现37个领先的MLLM表现不佳，甚至落后于同样认为该任务困难的人类。 |
| [^182] | [HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning](https://arxiv.org/abs/2610.36374) | 提出HIDRA框架，通过层次化双路由注意力机制将指令级专家检索与上下文细化解耦，并利用键级几何正则化稳定路由，从而解决无回放终身模仿学习中的路由干扰问题。 |
| [^183] | [Reliability-Aware Short-Term Glycemic Risk Prediction Using an Anchor-Based Comparative Framework of Sequence, Multimodal, and State-Updating Digital-Twin Models](https://arxiv.org/abs/2610.36363) | 本研究提出基于锚点的比较基准框架，利用736名1型糖尿病患者的550余万个标准化预测锚点，在统一的患者级评估协议下比较混合LSTM-Transformer、多模态数字孪生与状态更新数字孪生三种神经架构在30/60/120分钟血糖风险预测中的表现，并同时评估校准等可靠性指标。 |
| [^184] | [WAKMan: Workload Aware KVCache Manager](https://arxiv.org/abs/2610.36355) | 该论文提出了工作负载感知的KV缓存管理器WAKMan，通过分析不同类型请求（代理式请求偏向最近性、热门提示和长对话偏向频率）对缓存淘汰策略的不同需求，解决了没有任何单一淘汰策略能在所有LLM工作负载下始终保持最优的问题。 |
| [^185] | [Enhancing Clinical Classifier Robustness with Deep Representation Learning Based OOD Detection on DNA Methylation Data](https://arxiv.org/abs/2610.36344) | 该论文提出了一种端到端的基于深度表征学习的分布外（OOD）检测框架，通过从深度神经网络瓶颈层提取特征向量并结合五种OOD检测方法识别训练分布之外的肿瘤类型，从而有效增强DNA甲基化阵列临床肿瘤分类器的鲁棒性。 |
| [^186] | [Executable Archaeology of Programs That Model Programming: Reanimating the Heuristic Compiler (1961) and KBEmacs (1984)](https://arxiv.org/abs/2610.36341) | 本文运用“可执行考古学”方法复原并运行了两个早期AI程序——赫伯特·西蒙的启发式编译器（1961）与MIT的KBEmacs（1984），以此重现并比较“编程即问题求解”与“编程即组合惯用模式”这两种关于人类如何编程的经典理论。 |
| [^187] | [CrowdCC: Jointly Benchmarking Corruption Robustness and Deployment Quantization in Crowd Counting](https://arxiv.org/abs/2610.36340) | 该论文提出CrowdCC基准，首次联合评估人群计数模型在图像损坏退化与部署量化（FP32/FP16/INT8）下的表现，发现采集链类损坏对所有模型造成的性能下降最为严重。 |
| [^188] | [OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning](https://arxiv.org/abs/2610.36335) | 该论文提出OA-RAG，一种基于本体的智能体RAG方法，将形式化推理与大语言模型推理相结合，并通过报告本体证据观察和模型评估的依据水平，提升医学问答中答案的忠实性。 |
| [^189] | [Plenty of Room in the Middle: Revealing Opportunities on a GPU Cluster via Fine-Grained Monitoring](https://arxiv.org/abs/2610.36329) | 本文开发了一款基于DCGM库的轻量级细粒度（100毫秒级）GPU监控工具，并通过对大学GPU集群长达数月的监测数据分析，揭示了以往粗粒度监控无法发现的GPU利用率提升机会。 |
| [^190] | [How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation](https://arxiv.org/abs/2610.36328) | 该论文提出 RefactorProof 基准，通过对 185 个 VERINA 任务生成 458 个经 Lean 认证的行为保持重构变体，揭示了大语言模型生成的证明在程序结构重构下存在显著脆弱性。 |
| [^191] | [Ontology-guided multi-agent reasoning for explainable differential diagnosis](https://arxiv.org/abs/2610.36325) | 该论文提出了一种本体引导的多智能体鉴别诊断框架，通过复现医生的迭代推理过程，使诊断结论建立在临床证据之上并具备可解释性。 |
| [^192] | [When Retrieval Is Not Enough: Obsolete Facts in Agent Memory](https://arxiv.org/abs/2610.36308) | 该论文揭示智能体记忆系统的失败主要并非源于检索缺失，而是源于无法应对过时事实——即使最新信息已被成功检索，一条过时的竞争性陈述仍会使模型准确率大幅下降（Qwen2.5 从 100% 跌至 44.6%）。 |
| [^193] | [Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks](https://arxiv.org/abs/2610.36298) | 本研究通过基准测试四种代表性的模型窃取攻击，证明由于脉冲神经网络采用基于稀疏离散脉冲和时变膜电位的时间计算模式，其决策行为比传统人工神经网络更难通过黑盒查询被窃取重建。 |
| [^194] | [Detecting and Mitigating Query-Based System Prompt Leakage Attacks](https://arxiv.org/abs/2610.36295) | 提出了模型无关的AdvGuard框架，通过有界语义交互图建模跨查询关系并利用图卷积网络捕捉多跳模式，实现对基于查询的系统提示词泄露攻击的有效检测与缓解。 |
| [^195] | [BUSTA: Teaching Small Language Models to Reason About Binary Execution](https://arxiv.org/abs/2610.36288) | 通过“逆向基本块执行”训练任务结合GRPO强化学习微调，小型开源模型（1.7B级别）在二进制逆向推理能力上可逼近前沿大模型，从而能在资源受限的沙箱逆向工程环境中高效部署。 |
| [^196] | [A Smart Instant Book Filter Model based on Dual-Objective Optimization](https://arxiv.org/abs/2610.36287) | 本文提出一种基于双目标优化的智能即时预订筛选模型，通过在搜索结果中智能筛选和呈现即时预订房源，在减少房客预订延迟与被拒风险的同时兼顾平台预订率等目标。 |
| [^197] | [cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation](https://arxiv.org/abs/2610.36250) | cuArms是一个与GPU并行计算协同设计的基于冲突的多机械臂操作规划器，通过可复用的多模态路线图和惰性搜索高效解决冲突与死锁，并结合高层调度器联合优化任务分配、调度、运动和操作技能。 |
| [^198] | [Freeze, Don't Clone: Fine-Tuning with Expressive Policies Without Retaining Offline Data](https://arxiv.org/abs/2610.36247) | 本文发现离线数据消失后持续进行行为克隆会通过行为反馈循环导致性能退化，提出冻结预训练表达性策略先验并停止在线行为拟合的方法，在七个OGBench域上显著提升了IFQL、FQL和QC等算法在无需保留离线数据场景下的微调效果。 |
| [^199] | [BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals](https://arxiv.org/abs/2610.36239) | 提出三信号框架BehaveRec，通过可学习门控融合短期会话、90天长期用户历史和基于行为嵌入相似度的社区信号，在1.37亿条序列的工业数据集上超越了更大容量的SASRec基线，并系统研究了长期历史信号何时真正对推荐有效。 |
| [^200] | [MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge](https://arxiv.org/abs/2610.36234) | MARQ是一个模块化、与具体大模型无关的需求质量评估原型，通过注入基于INCOSE准则可操作化的系统工程专家知识，显著提升了大语言模型在不同质量准则下识别需求缺陷的能力，超越了现成模型的表现。 |
| [^201] | [Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures](https://arxiv.org/abs/2610.36229) | 该研究基于近800万个提交发现，即使去除作者身份签名，仍可凭借提交信息冗长度和变更规模等特征以最高0.859的ROC AUC区分AI编程智能体与人类的代码提交，且检测信号主要来自元数据而非代码本身。 |
| [^202] | [Directional Statistics on Circles and Spheres: A New Angle on Data](https://arxiv.org/abs/2610.36227) | 本文指出角度数据天然位于圆和球而非直线上，直接算术平均可能导致错误结论（如朝北方位平均后指向南方），并通过风向与扩散MRI等实例介绍圆形/轴向均值、方向变异性和概率模型等方向统计工具，主张在选择统计方法前应先识别数据的几何结构与对称性。 |
| [^203] | [Structure, Not Sequence: Tracking and Validating Folding Mechanisms in ESMFold's Trunk with Sparse Autoencoders](https://arxiv.org/abs/2610.36226) | 该论文首次用稀疏自编码器解析 ESMFold 的折叠主干，发现承载折叠计算的特征按“结构而非序列”组织，并在三维结构回归过程中呈现出模块化的功能分工。 |
| [^204] | [Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study](https://arxiv.org/abs/2610.36208) | 本研究复现并扩展了LXR反事实推荐解释方法，证明其不仅对最高推荐项目有效，对其他项目同样有效，同时发现项目流行度等简单基线在某些任务中表现得出人意料地好。 |
| [^205] | [Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization](https://arxiv.org/abs/2610.36204) | 提出 Concorde 能量框架，通过无参数谱锚点、乘积流形混合曲率残差与双曲切丛连续向量场三个解耦分支的门控融合，实现稀疏图上几何感知且谱稳定的链路预测。 |
| [^206] | [Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](https://arxiv.org/abs/2610.36202) | 本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。 |
| [^207] | [count-catcher: Lean Theorems from Model Counting Certificates](https://arxiv.org/abs/2610.36196) | count-catcher 工具将精确模型计数器 d4 的 CPOG 计数证书转化为 Lean 中经过验证的定理，证明 CNF 公式（进而对应的组合对象）恰好有 N 个模型。 |
| [^208] | [Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models](https://arxiv.org/abs/2610.36192) | 提出维纳信息引导解码（WiD），利用二阶图像统计（维纳估计）来辅助速度场预测并组织解码器计算，从而显著简化像素空间流模型的训练。 |
| [^209] | [Skill Dreaming: Evolving Repository Skills through Pull Request Streams](https://arxiv.org/abs/2610.36188) | 提出了 Skill Dreaming 框架，以拉取请求流为知识来源，实现拉取请求、仓库与技能库的共同演化，使仓库专属技能随代码库的变化持续更新并反哺未来的 PR 实践。 |
| [^210] | [Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing](https://arxiv.org/abs/2610.36185) | 该研究提出TestCompass——一个不代学生生成可执行测试、而是针对其测试套件薄弱点提供定向反馈的生成式AI辅导工具，并通过18名学生的课堂对比实验证明AI可以在不直接生成测试的前提下有效支持初学者的单元测试学习。 |
| [^211] | [Learning Human Sports Motion Feasibility via Quantile Regression on Tracking Data](https://arxiv.org/abs/2610.36183) | 该论文将体育中的球员到达时间（TTA）建模为条件分位数回归问题，利用2022年世界杯全部64场比赛的光学追踪数据训练的轻量级多层感知机，在预测球员到达球场任意位置的最短合理时间上一致优于文献中所有基于物理的运动模型基线。 |
| [^212] | [Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule](https://arxiv.org/abs/2610.36169) | 论文证明，检索增强语言模型中检索上下文与内部知识的冲突无法靠设计时固定的规则（如一律信上下文或信模型）最优解决，且自适应解码依赖的对称散度不包含判断依据，正确做法是结合分别评估各来源的证据与上下文可靠性的先验。 |
| [^213] | [Stratum: Repository Memory for Coding Agents Beyond the Context Window](https://arxiv.org/abs/2610.36153) | 该论文提出Stratum，为编码智能体提供跨会话的持久化、经验感知的仓库记忆系统，并通过实证测量证明真正的瓶颈在于相关知识的选择而非上下文窗口容量。 |
| [^214] | [Replication or Corroboration? An Axiomatic Framework for Evidential Dependence in Multi-Agent Debate](https://arxiv.org/abs/2610.36151) | 该论文提出了一个公理化框架来区分多智能体辩论中的“复制”（重复相同证据）与“印证”（独立证据支持同一主张），并证明“根源复制不变性”性质能够确保仅靠复制行为无法改变评判者的裁决结果。 |
| [^215] | [From Structural Feedback to Prompt Policies: Learning Faithful Text-to-Image Prompt Editors](https://arxiv.org/abs/2610.36146) | 该论文提出StruPO，一种结构感知的提示编辑器，利用校准场景图作为共享表示，并通过平滑奖励塑形和多图像平均处理定位不确定性与图像生成的随机性，从而学习忠实的文本到图像提示编辑策略，在不改变用户请求场景的前提下改善生成图像质量。 |
| [^216] | [Physical structure as an organizing principle for universal Hamiltonian learning](https://arxiv.org/abs/2610.36135) | UniTB是一个将密度泛函紧束缚物理先验与深度学习相结合的通用E(3)等变哈密顿量框架，通过专家混合路由、电荷平衡头和投影约束，实现跨化学体系的泛化并保证下游性质的物理合理性。 |
| [^217] | [When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths](https://arxiv.org/abs/2610.36134) | 提出对比证据路径（CEP）方法，通过在查询条件化的文档图上依据最强单一路径评分，并利用背景访问分布进行校准来消除高连通性文档带来的结构偏差，从而引导搜索智能体发现真正有价值的证据而非高度连接的干扰节点。 |
| [^218] | [Why and How to Do Science When AI Is Better at It: A Framework Grounded in Human Understanding and Direction](https://arxiv.org/abs/2610.36116) | 本文提出“人类理解”与“人类主导”两部分检验标准，论证科学的核心目标是服务于人类目的与价值观的持久集体理解，并据此判断哪些研究任务可以委托给AI、哪些必须保留在负责任的人类手中。 |
| [^219] | [Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses](https://arxiv.org/abs/2610.36114) | 该研究揭示了一个被忽视的关键问题：即使LLM智能体的上下文窗口利用率仅为5.75%，智能体框架仍会通过固定长度切分JSON的方式截断三分之一的工具结果，导致末尾数据不均匀丢失，使不同字段到达模型的比例差异悬殊（4.7%对99.3%），因此评估智能体不能只看窗口大小和答案正确率，还必须关注“信息损耗”。 |
| [^220] | [Scaling and Renormalization in Neural Networks: An Effective Renormalization-Group Description of Grokking](https://arxiv.org/abs/2610.36113) | 该论文将重整化群物理理论引入神经网络Grokking现象研究，通过基于神经元耦合矩阵谱排序的实空间重整化方法，为神经网络的记忆到泛化转变建立了定量的物理描述框架。 |
| [^221] | [Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach](https://arxiv.org/abs/2610.36110) | 本文提出在半参数贝叶斯网络中用狄利克雷过程混合模型（DPM）替代核密度估计进行非参数条件概率分布建模，通过数据自适应聚类获得更紧凑的表示，在降低推理成本的同时提升了密度估计性能。 |
| [^222] | [Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via the Model Context Protocol](https://arxiv.org/abs/2610.36108) | 该论文提出Log-To-Leak攻击，通过MCP工具元数据中的提示注入，在合法工具调用完成后诱导LLM智能体执行额外的攻击者控制的日志记录调用，从而窃取用户查询与响应数据，在18个模型-服务器组合中实现了62.64-100%的攻击成功率。 |
| [^223] | [CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction](https://arxiv.org/abs/2610.36100) | CohortFlow通过将自然语言修正转化为对结构化队列规范的局部更新，并借助显式语义依赖关系选择性失效受影响的计算，在保持与完全重新生成相当准确率的同时，将修正延迟降低最高64.8%、模型调用次数减少38.1%。 |
| [^224] | [How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning](https://arxiv.org/abs/2610.36082) | 提出Fin-SD方法，通过让学生模型自己发现并批评首次错误生成提示，再以错误前缀加提示的条件进行自蒸馏，从而在无需外部教师或完整参考解答的情况下提升小模型的多步金融推理能力。 |
| [^225] | [The Trinity:Capability-Driven Recursive Self-Improvement for Multimodal Agents](https://arxiv.org/abs/2610.36077) | 提出了The Trinity递归自我改进框架，通过Creator、Explorer、Evolver三个角色将世界生成与Harness进化相耦合，并以共享能力需求为基础将任务树重组为分层能力树，从而实现多模态智能体的能力驱动自我进化。 |
| [^226] | [Rollback: Experience Backtracking in World Models for Policy Improvement](https://arxiv.org/abs/2610.36068) | 提出Rollback策略改进范式，利用世界模型回溯关键决策点并分支探索反事实动作，将反事实结果转化为细粒度的动作级监督信号，无需单独的价值模型即可改进机器人策略。 |
| [^227] | [Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling](https://arxiv.org/abs/2610.36015) | 该研究通过在610个注入故障的大规模实验发现，基于前沿大模型的编程智能体仅凭源代码推理即可将故障定位准确率提升至98%以上的天花板水平，调试器的额外辅助收益已微乎其微。 |
| [^228] | [CIVER: Causal Intervention Verification for Reliable Code Review](https://arxiv.org/abs/2610.36010) | CIVER提出了一种基于因果干预的验证框架，通过识别代码评审发现背后的核心问题来区分真实缺陷与误报，从而提升LLM代码评审的可靠性。 |
| [^229] | [DuplexTune: Coupled-State Replay for Memory-Efficient Mixture-of-Experts Adaptation](https://arxiv.org/abs/2610.36009) | DuplexTune通过耦合隐藏状态重放框架，在单块H800上以38.8 GB峰值内存适配Qwen1.5-MoE模型，较检查点微调降低约40%内存并提升吞吐量，任务性能相当，但在多语言能力上存在一定权衡。 |
| [^230] | [MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine](https://arxiv.org/abs/2610.36006) | MedJev是一个可在笔记本电脑上本地运行的开源“系统一”医学决策模型，在多项医学知识和诊断基准上接近GPT-4的性能，同时保持良好的置信度校准，为快速高效的临床决策支持提供了新方案。 |
| [^231] | [PolicyMediator: Multi-Regional Policy Learning through Mediation Agents with Evolving Skills in OPC Ecosystems](https://arxiv.org/abs/2610.35997) | 提出认知约束下的领导者-多追随者Stackelberg框架PolicyMediator，通过技能不断演化的中介智能体将OPC的认知变化反馈与多区域政策学习形成闭环，解决数字游民式一人公司跨区域选址错配与政策调整低效的问题。 |
| [^232] | [Pricing Cross-Bank Context in Federated Anti-Money Laundering](https://arxiv.org/abs/2610.35996) | 该论文提出利用交易双方银行已记录的交易在本地近似远程账户状态，实现无需额外隐私数据交换的联邦多层GNN反洗钱检测，并通过信息披露阶梯实验发现可观测基线已贡献最大收益，额外共享的跨银行上下文信息边际价值有限。 |
| [^233] | [Layerwise Repacking: Variable-Length Hybrid-Attention Training without an Extra Redistribution Pass](https://arxiv.org/abs/2610.35989) | 该论文提出逐层重打包方法，使混合注意力模型的变长训练中全注意力层使用宽SP16/32组、稀疏注意力层使用节点本地的小SP组，无需在层间边界进行额外的数据重分配，从而避免跨节点all-to-all通信7.0倍的性能损失和额外传输开销。 |
| [^234] | [PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix](https://arxiv.org/abs/2610.35987) | 本文提出PROBE基准测试，揭示前沿编码智能体在仓库漏洞发现中召回率和精确率均很低，且其找到的漏洞与人类维护者实际修复的漏洞存在系统性差异。 |
| [^235] | [Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs](https://arxiv.org/abs/2610.35985) | GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。 |
| [^236] | [Enhancing Deep Learning Model Performance through Data Generation Strategies: Classification of Olive Tree Diseases](https://arxiv.org/abs/2610.35984) | 该论文提出了一种融合传统数据增强、CutMix、MixUp和DCGAN的综合性数据生成策略，通过三种数据分布场景有效解决了橄榄树病害分类中的数据不足与类别不平衡问题，从而提升了深度学习模型的性能。 |
| [^237] | [Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering](https://arxiv.org/abs/2610.35982) | 该论文提出一个面向安全关键汽车 C/C++ 软件的编程助手，通过 MCP 协议向开发者开放生成、编辑、验证三个可按需调用的工具以实现人机协作验证，并针对汽车领域定制底层大语言模型。 |
| [^238] | [UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity](https://arxiv.org/abs/2610.35980) | UnityBench-Mini3D是一个包含77个Unity项目级任务的基准，通过1,178个人工编写的引擎内里程碑测试对LLM智能体的3D游戏开发能力进行确定性评估，可提供部分得分、定位失败点，无需依赖主观的LLM评判器。 |
| [^239] | [Multiple Leak Localization in Water Distribution Networks through Graph Neural Networks](https://arxiv.org/abs/2610.35972) | 该研究提出了一种图神经网络模型，无需预先指定泄漏数量或穷举泄漏位置组合，即可在稀疏压力监测条件下定位供水管网中数量可变的多个同时泄漏，其准确率比传统机器学习方法最高提升12%和20%。 |
| [^240] | [EMMA: Structured Episodic Memory for Multi-Agent Workflow Synthesis](https://arxiv.org/abs/2610.35966) | 提出了EMMA框架，通过结构化情景记忆进行多智能体工作流合成，解决了单智能体自适应中的跨任务干扰问题，以及现有多智能体演化方法中更新规则不清、变化范围依赖骨干模型偏好而非实际失败经验的问题。 |
| [^241] | [A Hyper-Logical View on Multi-Agent Path Planning](https://arxiv.org/abs/2610.35964) | 本文提出用超属性时序逻辑HyperLTL及其扩展HyperQPTL统一建模多智能体路径规划的多种变体，为变体比较、跨智能体约束表达、最优性描述以及与验证和因果性技术的连接提供了统一的规约语言。 |
| [^242] | [AgentBaton: Efficient Model Collaboration in Agentic Tasks](https://arxiv.org/abs/2610.35959) | 该论文提出AgentBaton，系统研究了在长程智能体任务中通过模型交接实现强大模型与高效模型协作的关键设计（交接方向与上下文传递），以在保持任务准确性的同时大幅降低成本。 |
| [^243] | [Structured Time-Frequency Representation for Channel Prediction](https://arxiv.org/abs/2610.35956) | 该论文提出FAR-CSI框架，通过在联合的时间与工作频段位置域中构建FDD感知的结构化表示，使历史UL观测与未来DL预测目标在位置空间中占据分离区域，从而实现FDD系统中从上行观测到下行CSI的准确信道预测。 |
| [^244] | [ResCom: Activation Residual Compensation for Large Language Model Quantization](https://arxiv.org/abs/2610.35955) | ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。 |
| [^245] | [Curriculum-Driven Deep Reinforcement Learning for Autonomous Far-Range Rocket Landing](https://arxiv.org/abs/2610.35954) | 本文提出一种课程驱动的深度强化学习框架，通过三阶段课程设计、评论家网络预热和防灾难性遗忘的复演机制，并采用完整3×3旋转矩阵的高维观测，实现了火箭从远距初始条件到精确着陆的自主控制。 |
| [^246] | [Counting the Bandages: On measuring the time that AI saves](https://arxiv.org/abs/2610.35942) | 论文主张应以结合逃逸缺陷的“账本观点”而非单纯计时的“秒表观点”来衡量AI节省的时间，定义了节省时间、超额逃逸缺陷和盈亏平衡缺陷成本等量，证明只要仍需人工检查，校准良好的工作者逃逸缺陷率与模型质量无关，并从公开数据复现METR开发者研究，显示同一数据可支持19%或24%的速度下降。 |
| [^247] | [Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems](https://arxiv.org/abs/2610.35930) | 该论文提出MAD Bench基准，通过行为和内部干预追踪多智能体辩论中大语言模型改变观点的机制，发现辩论后的答案一致性往往源于社会从众效应而非独立验证，且消息的来源归因会显著改变其影响力。 |
| [^248] | [SCOPE: Source Credibility and Outlet Profiling Engine](https://arxiv.org/abs/2610.35929) | 本文提出SCOPE多智能体系统，通过协调文章获取、内容分析、证据搜索等专门智能体，基于证据对新闻媒体的偏见、事实性和可信度进行透明评估，在200家媒体基准上超越了多种基线方法。 |
| [^249] | [ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets](https://arxiv.org/abs/2610.35925) | ReCoPII框架通过规划本体约束的实体关系并据此推导MASK/KEEP标签，无需单一数据主体假设即可自动构建情境化PII脱敏数据集，所生成的韩文语料ReCoPII-Ko的KEEP标签获得88.42%的标注者认可率，远超SPY方法的26.27%。 |
| [^250] | [Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation](https://arxiv.org/abs/2610.35910) | 本文揭示了视觉语言模型在图像变换任务中会遵从按种族和性取向改变人脸的请求，生成统计上可区分且强化刻板印象的图像，并且这些表征偏差还会在下游文本推理任务中重现。 |
| [^251] | [Task-Utility IQA: Rethinking Image Quality Assessment in Embodied Tasks](https://arxiv.org/abs/2610.35905) | 提出了一种基于多模态大语言模型的具身任务效用导向图像质量评估框架，将IQA从任务无关的分数回归范式转变为由具身任务效用定义的目标条件判断，且任务条件化效用分数比传统感知指标更能预测下游任务的成功。 |
| [^252] | [Beyond Greenwashing: Algorithmic Resilience and the Non-Linear Governance Premium in Sovereign Capital Flows](https://arxiv.org/abs/2610.35895) | 本文利用梯度提升模型与双重可解释性框架，揭示主权风险传导本质上是非线性的，并提出“治理溢价”概念——ESG改善只有在突破关键治理阈值后才能为国家级资本流入带来优势。 |
| [^253] | [scPyviewer: a Python-native interactive viewer from AnnData single-cell data](https://arxiv.org/abs/2610.35890) | scPyviewer是一个可直接读取AnnData对象的Python原生交互式单细胞数据网络查看器，在与三个现有R Shiny工具的功能对比中全面达到或超越基准能力，并独特提供免转换的原生数据读取和跨数据集基因表达比较功能。 |
| [^254] | [I-HITL: Inalienable Human-in-the-Loop](https://arxiv.org/abs/2610.35835) | 该论文提出“不可剥夺的人在回路”（I-HITL）协议，确保即使智能体攻破了人类所交互的系统、甚至人类试图将批准权委托给智能体时，智能体的关键行动仍必须获得人类明确批准才能执行。 |
| [^255] | [Pretraining Knows What, Simulation Teaches How: Compositional Generalization in a Sim-Trained Robot Agent](https://arxiv.org/abs/2610.35817) | 提出零真实机器人策略ZAP，仅借助预训练视觉语言模型加上简单仿真任务上的监督微调与GRPO训练，无需任何真实机器人数据，即可在真实世界任务上达到96.7%成功率，并对新物体、目标、策略和语言展现出组合泛化能力。 |
| [^256] | [Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones](https://arxiv.org/abs/2610.35715) | 本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。 |
| [^257] | [Agentic Algorithm Design: LLMs that Propose, Formalise, and Prove Runtime Bounds for Evolutionary Algorithms in Lean](https://arxiv.org/abs/2610.35595) | 该论文提出了一个模仿进化算法研究方法的分阶段流水线，使LLM智能体能够自主设计黑盒优化算法，并在Lean中通过势函数与漂移分析形式化证明其运行时间边界，全程无需经验性比较即可获得O(n²)的期望运行时间保证。 |
| [^258] | [Multi-Fidelity Training Reshapes Uncertainty Decomposition in Atomistic Models](https://arxiv.org/abs/2610.35560) | 该研究利用均值-方差MACE模型深度集成发现，多保真度训练中低保真度预训练虽能降低预测误差并改善不确定性排序，但预测方差的校准退化主要发生在高保真度微调阶段，且在能量外推下更为严重。 |
| [^259] | [The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation](https://arxiv.org/abs/2610.35541) | 提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。 |
| [^260] | [JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks](https://arxiv.org/abs/2610.35532) | JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。 |
| [^261] | [Waypoint: Analytically Modeling GEMM for CPUs](https://arxiv.org/abs/2610.34914) | 提出了解析模型Waypoint，通过捕捉内存层次、指令级并行和多线程等架构效应，自动为CPU确定GEMM的最优分块大小，并据此构建了性能媲美最先进库且可加速LLM推理的GEMM库OPAM。 |
| [^262] | [Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents](https://arxiv.org/abs/2610.34817) | 该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。 |
| [^263] | [VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware](https://arxiv.org/abs/2610.34717) | 该论文提出了VLA-HWBench基准，通过七阶段推理分解在两种边缘硬件平台上对18个开源VLA模型进行测试，发现动作生成策略对推理延迟的影响远大于模型规模。 |
| [^264] | [SEEK: Skill Extraction with Evolving Knowledge against Frontier Science Agents](https://arxiv.org/abs/2610.34365) | 本文提出了SEEK，一个黑盒自演化红队攻击智能体，通过在持久记忆中整合响应证据并演化可复用攻击技能，成功提取了科学智能体平台中受保护的技能，揭示了100%测试平台均存在技能泄露的严重安全风险。 |
| [^265] | [An actively trained neural operator for near-explicit atmospheric chemistry](https://arxiv.org/abs/2610.33771) | 本文提出FNO-MoE——一种结合专家混合机制的傅里叶神经算子，可单次调用预测30分钟内的完整近显式大气化学状态，并利用专家间分歧进行主动采样自我训练，以远低于求解MCM的成本实现精确且稳定的多小时化学演化模拟。 |
| [^266] | [Local Entropy Fluctuations for Black-Box LLM-Generated Text Detection](https://arxiv.org/abs/2610.33767) | 该论文提出新指标“相对峰值显著率”，利用人类写作信息流均匀而机器生成文本存在尖锐局部熵波动这一结构性差异，解决了黑盒场景下现有检测器在指令条件文本上灾难性失效的问题。 |
| [^267] | [Seeing the Options: Typed Decisions over Visual Candidates with Reusable Option Memory](https://arxiv.org/abs/2610.33702) | 提出Iris框架，将支持顺序不变性与精确缓存复用的类型化决策扩展到视觉、文本或混合候选场景，并证明了该性质对任意逐候选编码器均成立。 |
| [^268] | [AOS-Bench: Can LLM Agents Manage Dynamic Workloads?](https://arxiv.org/abs/2610.33178) | 本文提出 AOS-Bench，首个将 LLM 智能体作为“工作负载管理器”评估的基准——任务随时间流式到达并需在共享 CPU 与内存限制下交付——揭示了孤立任务上的成功无法反映智能体在动态多任务场景下的真实能力（重负载下所有系统损失 13 至 23 分）。 |
| [^269] | [Evaluating Visual Language Models on Visual Grounding for Visually Impaired People](https://arxiv.org/abs/2610.33078) | 本文提出VG-SHAP方法，通过扩展现有可解释性框架为黑盒视觉语言模型提供视觉定位解释，并在包含视障人士真实拍摄图像的VizWiz-VQA-Grounding数据集上评估开源模型的视觉定位能力。 |
| [^270] | [A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models](https://arxiv.org/abs/2610.33036) | 提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。 |
| [^271] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^272] | [Rethinking Multi-View Fusion for Urban Representation Learning](https://arxiv.org/abs/2610.31850) | 本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。 |
| [^273] | [PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean](https://arxiv.org/abs/2610.31810) | 提出PaperFormalizer——一个基于Lean的“作者在环”智能体流水线，能够发现并修复研究论文证明中的漏洞，并在作者参与关键决策下将论文转化为经Lean验证的形式化版本。 |
| [^274] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^275] | [ARCANA: A Benchmark for Abstraction and Analogical Reasoning](https://arxiv.org/abs/2610.29271) | 提出了ARCANA基准，通过要求从现实世界物体或物理现象到抽象网格变换的远类比映射来测试抽象与类比推理能力，人类可以可靠地解决这些任务，但大语言模型的表现与人类存在明显差距。 |
| [^276] | [Fig: Automated Multiagent Behavior Elicitor](https://arxiv.org/abs/2610.28891) | 该论文提出了Fig系统，能够根据自然语言行为描述自动生成或选择游戏环境并进行针对性干预，从而系统性地诱发和评估多智能体模型的特定社会行为（如合作、欺骗、背叛等）。 |
| [^277] | [From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming](https://arxiv.org/abs/2610.28160) | 该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。 |
| [^278] | [The Shape of the DPO Hyperparameter Landscape: What to Measure, How It Moves with Scale, and When Not to Transfer It](https://arxiv.org/abs/2610.27895) | 通过密集超参数网格实验系统描绘了DPO超参数空间随模型规模和训练步数的演变规律，并识别了最适合超参数选择的基准与优化策略。 |
| [^279] | [ASTROLABE: A 3D Particle-Level Surrogate for Cosmology Simulation with Physical Flow Matching](https://arxiv.org/abs/2610.27068) | Astrolabe是一个基于流匹配的生成式模拟器，直接在N体粒子的相空间坐标上进行建模，突破了传统机器学习模拟器在分辨率上限和可模拟观测量类型上的双重限制。 |
| [^280] | [Towards Human-AI Complementarity for AI Safety](https://arxiv.org/abs/2610.24969) | 该论文构建了迄今最大的AI安全监督任务套件（3,957个实例）及人类-AI混合评估平台，发现尽管人类与AI之间存在显著的互补潜力，但现有辅助方法反而削弱了人类纠正AI错误的能力。 |
| [^281] | [First-Order LoRA Training with Phone NPUs](https://arxiv.org/abs/2610.24816) | 该论文提出了一种让手机NPU承担LoRA一阶训练中全部冻结权重矩阵乘法（包括推理接口不支持的输入梯度计算）的系统，通过复用常驻权重分块和前向反量化器，实现了大语言模型在端侧基于私有数据的高效微调。 |
| [^282] | [Annotrace: A Workbench for Step-Level Annotation of Agent Traces](https://arxiv.org/abs/2610.23284) | Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。 |
| [^283] | [Memorilla: Latent Semantic Memory for LLMs](https://arxiv.org/abs/2610.21375) | Memorilla提出了一种潜在语义记忆系统，能将整个文档集合的信息压缩为仅16个依赖查询的记忆token并注入冻结LLM的输入嵌入空间，在八个基准中的五个上超越了Mem0和MemoRAG等基线方法。 |
| [^284] | [Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages](https://arxiv.org/abs/2610.18420) | 该研究从零训练了首个泰米尔语优先、支持工具调用的25亿参数语言模型Agaram 2.5B，仅用8张H100 GPU、6天时间和约500亿token，就在翻译和泰米尔语指令遵循上超越了更大的印度语系模型Sarvam-1，为低资源语言的模型训练提供了可复制的实践方案。 |
| [^285] | [Where Small Mistakes Matter: Quantifying State Criticality for Imitation Learning](https://arxiv.org/abs/2610.17709) | 本文提出SCRIM方法，通过训练对抗性扰动策略和价值评论家来量化模仿学习中各状态的关键程度，并利用所得分数重新加权行为克隆损失，使模型重点关注那些小错误即会导致失败的关键状态。 |
| [^286] | [Comparing GenAI to Machine Learning and Manual Review for eDiscovery](https://arxiv.org/abs/2610.16925) | 该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。 |
| [^287] | [Provenance-Aware Belief Updating in Human-AI Feedback Loops](https://arxiv.org/abs/2610.13008) | 该论文提出LQET训练目标，使语言模型能够依据证据来源而非报告数量进行信念更新，显著降低了重复转发信息造成的虚假佐证与置信度漂移问题。 |
| [^288] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^289] | [Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally](https://arxiv.org/abs/2610.03445) | 该论文发现并定义了视觉语言模型中的“训练/推理差距”——对抗扰动虽能把教师强制训练损失压至近零，模型自由生成时却仍输出正确描述——并通过 logit lens 将该差距精确归因于单一自回归步骤中目标词元排名恰好固定为第3位的内部机制。 |
| [^290] | [Interpretable Synthetic Medical Tabular Data Generation for Clinical Decision Support Using Fuzzy Cognitive Maps](https://arxiv.org/abs/2610.00391) | 本文提出了一种基于模糊认知图（FCM）的新颖框架，利用语言学可解释的模糊集和显式因果关系编码来生成具有隐私保护能力的合成医学表格数据，从而支持临床决策系统在数据受限情况下的开发与验证。 |
| [^291] | [Diversity Combining for Multi-Path LLM Reasoning](https://arxiv.org/abs/2609.38829) | 该论文将多路径LLM推理形式化为无线通信中的分集合并问题，证明了在可交换性条件下多数投票是最优的均匀线性合并策略，并提出通过提示模板多样性降低路径间相关性，从而突破SC方法随采样数增加而收益饱和的瓶颈。 |
| [^292] | [Learn from the Gap: Differential-Aware Advantage Pruning with Adaptive Rollout Sampling for GRPO](https://arxiv.org/abs/2609.36932) | 提出即插即用的FastRL强化学习框架，通过差分感知的优势剪枝自适应保留高价值轨迹并最大化梯度多样性，同时结合自适应rollout采样机制，显著提升GRPO的训练效率与策略学习效果。 |
| [^293] | [EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning](https://arxiv.org/abs/2609.35290) | EvoIn提出了一个连接演化与内化的智能体微调框架，通过分析执行轨迹来演化并验证新的决策流程，再将改进的推理轨迹改写为去除框架指令引用的自包含推理并微调模型，从而将决策逻辑内化为模型自身的能力。 |
| [^294] | [Learning response-aware patient dynamics for respiratory support](https://arxiv.org/abs/2609.32782) | 提出了一种响应感知的患者动力学模型，将预测的生理变化显式分解为状态依赖的基线动力学和呼吸支持相关的偏差，并以室内空气作为参考基准，从而更好地建模呼吸支持对危重患者短期生理轨迹的个体化影响。 |
| [^295] | [GAUGE: Group-Wise View-Inconsistency Rectification for Feed-Forward 4D Tracking](https://arxiv.org/abs/2609.32596) | 本文提出GAUGE，一个无需训练、与模型无关的事后处理模块，通过从位移方向一致性恢复运动组并估计每帧径向尺度，来校正前馈式4D跟踪中沿视线方向的组级不一致性误差。 |
| [^296] | [Learning an Anchored Prompt Space for Continual Adaptation of Large Language Models](https://arxiv.org/abs/2609.32499) | 提出LAPS方法，通过自蒸馏对齐历史提示，并构建以任务特定软提示为顶点、贝塞尔控制提示塑造几何结构的锚定提示空间，从而在持续适配大语言模型时保留提示有效性并促进跨任务正向迁移。 |
| [^297] | [Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark](https://arxiv.org/abs/2609.30027) | 提出了“合成医院”——首个完全由公开医学教育材料构建、经医生验证、具有完整溯源链和可验证真实标准答案的开放合成纵向EHR基准，解决了真实EHR数据无法公开共享且缺乏可验证答案的核心障碍。 |
| [^298] | [Vision2CAD: A Visual Agent Harness for Explicit Geometry Referencing and Localization in Parametric CAD Modeling](https://arxiv.org/abs/2609.22688) | Vision2CAD通过将视觉语言模型推理与确定性CAD内核操作相结合，并引入基于ID的显式几何选择、局部坐标桥接和投影边缘定位机制，显著提升了参数化CAD建模中的几何引用与定位精度。 |
| [^299] | [Tailored to you: longitudinal effects of personalising language models](https://arxiv.org/abs/2609.20077) | 该研究通过对992名参与者进行为期五天的纵向实验，首次系统考察了基于记忆和基于调查两种个性化方法对用户与语言模型持续互动及其自我认知、人际关系的长期影响。 |
| [^300] | [MUMINS: Metadata-conditioned Uncertainty-aware Medical Image Next-state Synthesis](https://arxiv.org/abs/2609.17169) | MUMINS是一个高效的扩散框架，在单次反向扩散过程中联合扩散基线扫描与随访残差以合成随访医学图像，并同时生成空间不确定性图，通过元数据条件化和在每个去噪步骤动态重注入基线作为软锚点来保留细粒度解剖结构。 |
| [^301] | [Legislating World-Model-Based Planning with Legal Reasoning](https://arxiv.org/abs/2609.15113) | 本文提出了一个结合可废止道义逻辑（DDL）与学习型世界模型的法律规划栈，解决了感知基础化与法律约束转化的同构差距问题，实现了在机器人执行非法动作前进行事前干预的法律治理机制。 |
| [^302] | [Lightning Weave: Improving the Accuracy-Efficiency Frontier of Reasoning Models through Capability Composition](https://arxiv.org/abs/2609.14708) | Lightning Weave 提出了一种后训练框架，通过在线策略蒸馏将从独立专家模型中分别学到的准确率和效率能力组合到单一学生模型中，从而同时提升推理模型的推理准确率和推理效率。 |
| [^303] | [When Rubrics Fail: Hallucinations Reveal Blind Spots in Medical AI Evaluation](https://arxiv.org/abs/2609.12718) | 研究发现基于评分标准的医学LLM评估常遗漏临床相关的幻觉错误，尤其是未预料到的错误，仅在明确核查事实时才有效，揭示了当前医学AI评估方法的盲点。 |
| [^304] | [Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation](https://arxiv.org/abs/2609.11115) | 本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。 |
| [^305] | [Suan: Rectifying Direct Preference Safety Alignment in Large Language Models](https://arxiv.org/abs/2609.08634) | Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。 |
| [^306] | [Can LLMs Design Video Coding Tools? A Case Study on Planar Mode](https://arxiv.org/abs/2609.01535) | 本文通过“生成-评估”循环的实证研究，首次证明大语言模型能够自动设计并迭代改进视频编码中的Planar帧内预测模式，在VVenC编码器上实现了0.18%的码率节省且复杂度开销仅0.4%。 |
| [^307] | [Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents](https://arxiv.org/abs/2609.01487) | 提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。 |
| [^308] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^309] | [From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation](https://arxiv.org/abs/2608.27860) | 该论文提出名为“畸变扩展器（DEX）”的可学习参数方法，通过自监督对齐损失将鱼眼图像的潜在嵌入映射到透视图像的嵌入空间，使视觉基础模型能够泛化到鱼眼相机，实现高保真的深度估计与开放词汇分割。 |
| [^310] | [Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning](https://arxiv.org/abs/2608.27821) | 该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。 |
| [^311] | [MemPrism: Task-Conditioned Relational Memory Views for Long-Horizon Agents](https://arxiv.org/abs/2608.06745) | 提出MemPrism框架，通过任务条件化的关系记忆视图将持久经验存储与决策时工作记忆分离，动态构建关系视图以解决长时程智能体中的表征失配问题，并在具身智能和网络智能体基准上取得持续提升。 |
| [^312] | [NxN E-valuation: Hypothesis Certification via a Conformal CRT Null](https://arxiv.org/abs/2608.06621) | 提出 NxN E-valuation，一种基于 e 值与共形 CRT 零假设的假设认证算法，通过利用大型训练集让样本互为零假设，无需为每个假设专门构建零假设即可完成验证，尤其适合应对 LLM 探索系统中的幻觉问题。 |
| [^313] | [AI-guided deep visual proteomics reveals recurrence-associated morpho-molecular signatures within triple-negative breast cancers](https://arxiv.org/abs/2608.03145) | 该研究开发了一种结果导向的AI引导深度视觉蛋白质组学框架，利用复发风险热图指导三阴性乳腺癌的组织采样与质谱空间蛋白质组学分析，揭示了高风险区域富集有丝分裂程序、低风险区域富集免疫与抗原呈递程序的肿瘤内分子异质性。 |
| [^314] | [Hard Constraints, Smooth Gradients: Learning Feasible Inventory Policies via Differentiable Projection](https://arxiv.org/abs/2608.02343) | 该论文在强化学习策略中嵌入可微优化模块（神经网络提出连续动作目标、二次规划投影保证可行性、对偶信息整数映射恢复整数性），实现了端到端训练并严格满足硬约束的库存决策策略。 |
| [^315] | [Tracing the Cascade: A Topology-Aware Evaluation Framework for Scientific Agent Hallucinations](https://arxiv.org/abs/2608.00711) | 本文提出SCHEMA——首个基于证据、拓扑感知的科学智能体幻觉评估框架，通过自动构建科学概念图并生成论断验证、多跳推理、开放式解释与实验代码生成等多类图任务，捕捉幻觉在高度互联的科学知识网络中的级联传播效应。 |
| [^316] | [TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples](https://arxiv.org/abs/2607.28862) | TextCloak提出了一种基于强化学习的数据合成框架，通过生成既能保持语义忠实性和语言自然性又能降低模型学习效果的不可学习样本，来保护文本数据免受未经授权的大语言模型微调。 |
| [^317] | [From Deterministic to Generative Deep Learning for Urban Air Quality Reconstruction from Sparse Observations](https://arxiv.org/abs/2607.25687) | 本文提出了一种基于扩散模型的生成式深度学习框架，用于从稀疏观测中重建巴黎多种空气污染物（NO2、O3、PM2.5、PM10）的浓度场，并在与确定性深度学习模型的对比中展现出在真实观测数据上生成逼真空间分布的能力。 |
| [^318] | [ScalableRAG: High-Quality RAG with Exact Corpus Operations at Minimal Ingestion Cost](https://arxiv.org/abs/2607.25135) | ScalableRAG通过持久化的类型化集合工作区，让智能体在查询时利用集合代数执行精确的语料库级操作，无需任何预处理甚至向量索引，就能在聚合问答任务上大幅超越现有RAG系统。 |
| [^319] | [Wavelet Phase Diffusion for Structurally and Semantically Consistent Sim-to-Real Translation](https://arxiv.org/abs/2607.21628) | 提出小波相位扩散方法，在双树复小波包变换域中进行相位保持扩散，克服了傅里叶域全局频谱耦合引起的空间伪影问题，无需昂贵控制模块或成对数据流水线即可实现结构与语义一致的仿真到真实图像转换。 |
| [^320] | [LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning](https://arxiv.org/abs/2607.19358) | LISA是一种无需完整预训练的即插即用稀疏注意力机制，通过融合线性注意力分支与动态token选择索引器，在保留长程记忆的同时大幅降低长上下文推理的计算成本。 |
| [^321] | [Experience Memory Graph: Extracting Reusable Correction Knowledge from Agent Trajectories](https://arxiv.org/abs/2607.13884) | 该论文提出经验记忆图（EMG）框架，将智能体的失败恢复重新表述为图匹配问题，从而从历史轨迹中提取可靠、可迁移且可复用的纠错知识。 |
| [^322] | [From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception](https://arxiv.org/abs/2607.13624) | 该论文提出了一个由模块化ROS 2组件构成的语言驱动导航框架，利用RGB-D感知识别自然语言指令中目标物体的位置，使移动机器人能够将非专业用户的自然语言请求直接转化为可执行的自主导航目标。 |
| [^323] | [Trust Region Policy Distillation](https://arxiv.org/abs/2607.04751) | TOP-D通过动态构建近端教师模型，将不稳定、高方差的在线策略蒸馏转化为稳定且高样本效率的训练范式，在理论上保证了梯度方差控制与全局收敛性，且不引入任何额外计算开销。 |
| [^324] | [TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems](https://arxiv.org/abs/2606.25147) | TokenMinds基于预训练LLM改造的编码器-解码器架构并采用双输出设计，将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模，同时产出稠密用户嵌入与离散用户Token，突破了固定维度稠密嵌入的表征限制并捕捉用户行为的序列模式。 |
| [^325] | [HOLogicBench: Evaluating Higher-Order Logical Reasoning in LLMs](https://arxiv.org/abs/2606.23238) | 本文提出HOLogicBench基准，包含1,379个经Isabelle/HOL形式化证明验证的实例，系统性评估大语言模型在五种高阶逻辑推理操作上的能力，发现现有模型平均准确率仅为53.86%，且推理路径与有效推导对齐不足、易受最小反事实扰动的影响。 |
| [^326] | [LineageMark: Multi-user White-box Watermarking for Contribution Tracing in Model Derivation Chains](https://arxiv.org/abs/2606.17123) | 提出了LineageMark——一个面向模型派生链的多用户白盒水印框架，通过基于投影的参数编码方式，在多阶段模型派生和增量更新场景下实现多方历史贡献的追踪与验证。 |
| [^327] | [DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing](https://arxiv.org/abs/2606.15796) | 本文提出DifFRACT方法，首次将基于转码器的电路追踪技术扩展到多模态扩散变换器，通过训练时间步条件化的转码器替换FLUX.1中的MLP子层，实现了对图像生成模型内部可解释特征的精确归因和电路分析。 |
| [^328] | [Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources](https://arxiv.org/abs/2606.14141) | 该论文提出了基于FOA渲染的时空音频问答数据集ST-AudioQA及配套的ST-Audio编码器和ST-AudioLM模型，联合建模声源语义与随时间变化的声源轨迹，使大语言模型能够回答关于声源身份、位置和运动的时空音频问答问题。 |
| [^329] | [What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction](https://arxiv.org/abs/2606.07687) | 该研究通过统一的探针评估发现，视频世界模型潜空间中与动作相关的结构主要由时序视频预训练（潜变量预测）驱动而非像素重建保真度，视频预训练的自监督编码器在视觉保真度与动作预测之间实现了最佳的帕累托权衡。 |
| [^330] | [LLMs Can Leak Training Data But Do They Want To? Evaluating Memorization Propensity in LLMs](https://arxiv.org/abs/2606.06286) | 本文提出PropMe框架和SimpleTrace工具，首次将大语言模型“能否”泄露训练数据与“是否会主动”泄露训练数据区分开来，发现自然句子提示下的主动泄露远少于前缀攻击，但无提示或常见词提示时的泄露程度与前缀攻击相当。 |
| [^331] | [DRIFT: A Residual Flow Adapter for Decoding Continuous Outputs in Vision-Language Models](https://arxiv.org/abs/2606.05758) | 提出DRIFT框架，通过将基础预测器与基于流匹配的残差细化模块相结合，将预训练视觉-语言模型高效适配到需要精确连续输出的解码任务，如时间边界定位和机器人控制动作生成。 |
| [^332] | [Streaming Communication in Multi-Agent Reasoning](https://arxiv.org/abs/2606.05158) | StreamMA通过让智能体即时流式传递每个推理步骤实现流水线化，首次证明了流式通信既能降低多智能体推理延迟、又能提升推理效果，并给出了流式、串行与单一协议的首个闭式联合理论分析。 |
| [^333] | [Hedge-Bench: Benchmarking Agents on Hard, Realistic Tasks Pertaining to Financial Reasoning](https://arxiv.org/abs/2606.03918) | 提出了Hedge-Bench基准——基于对冲基金分析师真实推理轨迹和已揭晓市场结果构建的92个环境，用于评估AI智能体在不完整信息下的金融预测能力，发现前沿模型的联合成功率仅为2.54%至31.52%。 |
| [^334] | [Grasp-Then-Plan with Failure Attribution: A Closed Two-Stage Framework for Precise and Generalizable Robotic Manipulation](https://arxiv.org/abs/2606.03385) | 提出GTP-FA两阶段“先抓取后规划”框架，利用失败归因模型诊断操作失败原因，并以诊断驱动的方式联合优化抓取与运动规划模块，实现精确且可泛化的机器人长时程操作。 |
| [^335] | [CalArena: A Large-Scale Post-Hoc Calibration Benchmark](https://arxiv.org/abs/2605.30188) | 该论文提出了CalArena，一个覆盖近2000个实验、融合经典模型与基础模型预测、并提供数十种校准方法统一可复现实现的大规模标准化事后校准基准，用以系统评估哪些校准方法在实践中真正有效。 |
| [^336] | [Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents](https://arxiv.org/abs/2605.29927) | 本文提出PlanAhead框架，系统评估了四种自然语言计划表示（顺序子目标、叙述、伪代码、检查清单）对多模态LLM网页代理性能的影响，并实现了无需人工标注的自动任务难度分级。 |
| [^337] | [Beyond Questions: Evaluating the Stated Knowledge of LLMs](https://arxiv.org/abs/2605.26937) | 该论文提出了开放式知识评估这一新范式，并构建了包含10,000个实体的BeQu基准，通过开放式引导提示评估大语言模型自然表达的知识，从而克服了传统预定义问题评测中的可用性偏差。 |
| [^338] | [Hidden-State Privacy Has an Empty Middle](https://arxiv.org/abs/2605.24042) | 该论文通过对1,536个高斯协方差的实证测试和Fisher球理论下界的证明，揭示了隐藏状态隐私发布中存在“空缺中段”现象——不存在任何能同时提供中等效用与中等隐私的高斯机制，并指出对角逆Fisher发布是唯一达到极小极大最优的机制，但只能处于隐私/效用的极端边缘。 |
| [^339] | [LLM Code Smells: A Taxonomy and Detection Approach](https://arxiv.org/abs/2605.22976) | 本文提出了LLM代码坏味道的独立分类体系和九种坏味道目录，开发了静态分析工具SpecDetect4LLM进行检测，并通过对692个开源项目的实证评估发现73.5%的项目存在LLM代码坏味道。 |
| [^340] | [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781) | 该论文提出了一种基于Schatten-p范数的数据驱动准则，可动态地为神经网络各层自适应选择从SGD到Muon之间的最优更新几何结构，并在统一框架下恢复了SGD、Muon、Adam和MuAdam等主流优化器。 |
| [^341] | [CLEAR: Coupling Local Evidence with Affective Representations for Multimodal Emotion Recognition in Conversation](https://arxiv.org/abs/2605.18547) | 提出CLEAR方法，通过将局部判别证据与内部情感表征耦合，解决了多模态大语言模型在对话情感识别中情感理解与标签预测不一致、以及当前话语局部线索被上下文稀释的问题。 |
| [^342] | [BoLT: A Benchmark to Democratize Black-box Optimization Research for Expensive LLM Tasks](https://arxiv.org/abs/2605.17000) | 该论文提出了BoLT——首个以LLM为中心的基准测试，通过解决LLM训练与推理成本过高的门槛问题，使黑盒优化（BBO）方法能够在真实现代LLM任务（如超参数、数据混合和提示词优化）上进行系统研究和平民化评估。 |
| [^343] | [The Geometry of Learning: Read-Write Asymmetry Across Training](https://arxiv.org/abs/2605.16600) | 该论文以残差流为共享坐标系分析预训练与对齐阶段对 transformer 权重的更新方向，揭示了“读写不对称性”：读侧矩阵（query/key）的更新集中于其输入激活的主子空间，而写侧矩阵（输出矩阵）的更新相对于预测子空间接近随机，且这一特征是训练阶段的属性而非训练目标的属性。 |
| [^344] | [Edge-AI-Driven Learning-to-Rank for Decentralized Task Allocation in Circular Smart Manufacturing](https://arxiv.org/abs/2605.16433) | 该论文提出了一种边缘AI驱动的去中心化任务分配框架，通过轻量级决策智能体结合资源感知启发式和回归模型，在循环制造中实现高效的任务分配，同时兼顾能耗和资源约束。 |
| [^345] | [REALM: Retrospective Encoder Alignment for LFP Modeling](https://arxiv.org/abs/2605.14867) | 本文提出REALM框架，通过回溯知识蒸馏将非因果模型的表达知识迁移到因果模型中，从而实现高精度且可实时部署的LFP行为解码。 |
| [^346] | [Moderation by Interrogation: Uncovering Concealed Intent in Agent Social Networks](https://arxiv.org/abs/2605.12856) | 提出 Bot-Mod 审核框架，通过诊断性审讯与贝叶斯后验推断识别智能体的隐藏意图，突破了依赖内容信号的审核局限，并构建了 Molt-Intent 数据集加以评估。 |
| [^347] | [PIVOT: Bridging Planning and Execution in LLM Agents via Trajectory Refinement](https://arxiv.org/abs/2605.11225) | PIVOT提出了一种自监督框架，通过计划、检查、演化、验证四个阶段将轨迹作为可优化对象进行迭代细化，利用文本梯度编码规划与执行的差异，从而弥合LLM智能体中规划与执行之间的鸿沟，并在DeepPlanning和GAIA基准上取得最先进的性能。 |
| [^348] | [RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models](https://arxiv.org/abs/2605.09410) | RePO-VLA 提出了一种恢复驱动的策略优化框架，通过恢复感知初始化（RAI）和进度感知语义价值函数（PAS-VF）充分利用失败与恢复轨迹，显著提升 VLA 模型在长时程、接触密集操作任务中的鲁棒性。 |
| [^349] | [On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems](https://arxiv.org/abs/2605.07860) | 本文全面分析了VAE、GAN和扩散模型在联邦预测性维护系统中的性能与通信开销权衡，并基于分割参数个性化联邦学习提供了共享/私有参数划分的实证研究。 |
| [^350] | [Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training](https://arxiv.org/abs/2605.07063) | 该论文提出 Dr. Post-Training 框架，创新性地将通用训练数据重新概念化为防止模型过拟合稀缺目标数据的数据诱导正则化器，通过在每步训练中将目标数据的更新方向投影到通用数据构建的可行集上，并将标准训练与现有数据选择方法统一为该框架下的特例。 |
| [^351] | [Optimal Experiments for Partial Causal Effect Identification](https://arxiv.org/abs/2605.06993) | 该论文提出“最大效力问题”这一新框架，用于在成本约束下选择最优实验组合以最大程度收紧部分可识别因果查询的界限，证明了该问题的NP难性，给出了评估认识效力的通用方法，并引入基于因果图结构的剪枝准则（如线性时间的路径拦截规则）来控制超指数级搜索空间。 |
| [^352] | [Posterior-First Neural PDE Simulation: Inferring What Downstream Tasks Need from a Single Field](https://arxiv.org/abs/2605.03247) | 该论文提出“后验优先”的神经PDE模拟范式：由下游任务确定需推断的最小任务充分问题状态，用其后验分布（可由基于模拟推断的摊销后验估计器提供）来条件化滚动推演与下游查询，并从理论上证明贝叶斯下游价值经此后验分解、严格评分规则以之为最优，而确定性坍缩会带来歧义性损失。 |
| [^353] | [On the Optimal Sample Complexity of Offline Multi-Armed Bandits with KL Regularization](https://arxiv.org/abs/2605.02141) | 本文在多臂老虎机设置下对KL正则化离线学习算法KL-PCB给出了精确分析，证明其在大、小正则化条件下分别达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 和 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，从而基本刻画了KL正则化离线学习的最优样本复杂度。 |
| [^354] | [DeGenTWeb: A First Look at MGT-dominant Websites](https://arxiv.org/abs/2605.00087) | 提出了 DeGenTWeb 框架，通过将机器生成文本检测器适配到网页并聚合网站级检测结果，首次系统识别出以机器生成内容为主的网站，发现 Common Crawl 存档网站中 20.9% 和 Bing 搜索结果链接网站中 13.6% 属于机器生成文本主导型。 |
| [^355] | [AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI](https://arxiv.org/abs/2604.23018) | 该论文发布了包含10,071个经过度量尺度标准化、确定性锚定并附带完整材质与语义标注的3D资产数据集AmaraSpatial-10K，并配套提出了一套可复用的3D资产质量评估套件，为空间计算与具身智能提供了可直接使用的高质量3D资源。 |
| [^356] | [Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text](https://arxiv.org/abs/2604.19298) | 该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。 |
| [^357] | [Latent Abstraction for Retrieval-Augmented Generation](https://arxiv.org/abs/2604.17866) | 该论文提出LAnR统一框架，让单个LLM通过指定[PRED]标记的隐藏状态直接生成密集检索向量，在其自身潜在空间内联合完成编码、检索与生成，从而摆脱了传统RAG中生成文本查询及检索器与生成器分离的架构限制。 |
| [^358] | [Clinical Note Bloat Reduction for Efficient LLM Use](https://arxiv.org/abs/2604.16364) | TRACE 方法利用 EHR 元数据（或基于频率的去重）识别并清除临床记录中模板化和复制粘贴的冗余内容，在 530 万份记录和多项信息抽取及预测任务上验证了其能在保留真实临床信号的同时大幅压缩文本量，从而降低大语言模型的使用成本并提升效率。 |
| [^359] | [Neural Global Optimization via Iterative Refinement from Noisy Samples](https://arxiv.org/abs/2604.03614) | 本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。 |
| [^360] | [SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization](https://arxiv.org/abs/2603.29109) | SemLoc将LLM的自由形式推理转化为附加在代码图锚点上的可执行、可验证的语义约束，通过运行时检查和反事实验证实现软件故障定位。 |
| [^361] | [Ctrl-A: Control-Driven Machine Learning using Adaptive Online Data Augmentation](https://arxiv.org/abs/2603.21819) | Ctrl-A是一种融合控制理论的自动化数据增强算法，通过控制回路架构在训练中动态自适应地调整各增强操作的强度分布，无需初始化或手动设计增强策略，从而提升图像分类任务的性能。 |
| [^362] | [A Computationally Efficient Multi-Section Whole Slide Image Analysis for Capturing Prostate Cancer Multifocality in Biochemical Recurrence Prediction](https://arxiv.org/abs/2603.20273) | 该研究提出了一种计算高效的多切片全切片图像分析框架，通过优化子采样大幅降低计算成本，以捕获前列腺癌的多灶性特征，在生化复发预测中达到与临床基准相当或更优的性能，且AI风险评分是独立的预后因素。 |
| [^363] | [Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review](https://arxiv.org/abs/2603.18740) | 本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。 |
| [^364] | [SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?](https://arxiv.org/abs/2603.15401) | 该论文提出了SWE-Skills-Bench，一个以技能为中心、需求驱动的基准测试，通过3,920组有/无技能的配对评估，系统量化了单个智能体技能对真实世界软件工程任务的实际提升效果。 |
| [^365] | [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) | 该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。 |
| [^366] | [Efficient Token-Level Policy Optimization for Masked Diffusion Models via Velocity Alignment](https://arxiv.org/abs/2603.01563) | 提出LFPO框架，将扩散大语言模型的策略对齐转化为离散速度对齐问题，通过词元级对比更新直接优化去噪logits，实现稳定低方差的强化学习对齐，并显著减少去噪迭代次数。 |
| [^367] | [Flowette: Flow Matching with Graphette Priors for Graph Generation](https://arxiv.org/abs/2602.23566) | 提出Flowette框架，将连续流匹配与新型graphette结构先验相结合，利用最优传输耦合和化学感知正则化，实现具有重复子图motif的图生成。 |
| [^368] | [AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification](https://arxiv.org/abs/2602.22724) | AgentSentry提出了一种推理时防御方法，通过时序因果诊断与上下文净化来归因判断LLM智能体中待执行操作究竟源自用户任务还是注入内容，从而有效缓解多轮间接提示注入攻击。 |
| [^369] | [Distilling Metacognitive Behaviors through Agentic Editing](https://arxiv.org/abs/2602.22508) | 提出元认知行为微调（MBT），利用智能体逐行编辑并以 git 提交记录的方式在教师推理链中插入七种元认知行为，再据此训练学生模型，从而改善小模型的推理习惯。 |
| [^370] | [MIDAS: Mosaic Input-Specific Differentiable Architecture Search](https://arxiv.org/abs/2602.17700) | MIDAS通过自注意力机制为每个输入动态计算架构参数，并结合逐补丁的局部化架构选择与无参数的拓扑感知搜索空间，对DARTS进行现代化改进，在DARTS、NAS-Bench-201和RDARTS等多个基准上取得了优异性能。 |
| [^371] | [A swap-adversarial framework for domain generalization in pre- and post-stimulation Parkinsonian state classification](https://arxiv.org/abs/2602.10528) | 该论文提出了一种交换对抗框架，通过受试者间平衡通道交换（ISBCS）与领域对抗学习（DAL）相结合，缓解ECoG数据中受试者间变异性和高维小样本问题，实现帕金森病刺激前后状态分类的跨数据集稳健领域泛化，并发布了首个可复现的ECoG基准数据集。 |
| [^372] | [Difficulty-Estimated Policy Optimization](https://arxiv.org/abs/2602.06375) | 本文提出DEPO框架，通过在线难度估计器动态评估和过滤训练数据，解决了GRPO在过易或过难问题上梯度信号衰减及低效用样本计算开销大的问题，提升了大推理模型推理对齐的效率与稳定性。 |
| [^373] | [VDE Bench: Evaluating The Capability of Image Editing Models to Modify Visual Documents](https://arxiv.org/abs/2602.00122) | 提出了VDE Bench基准，包含3,506个指令引导的中英文文档图像编辑样本，通过区域级和整页双重评估协议，系统评估图像编辑模型在密集文档中进行精确局部编辑的能力。 |
| [^374] | [SQuID: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery](https://arxiv.org/abs/2601.13401) | 该论文提出了卫星影像定量空间推理基准SQuID，揭示了视觉-语言模型虽然能流畅描述卫星图像，但在计数、测面积、算距离等定量空间推理任务上仍存在显著短板。 |
| [^375] | [Beyond the Dirac Delta: Mitigating Diversity Collapse in Reinforcement Fine-Tuning for Versatile Image Generation](https://arxiv.org/abs/2601.12401) | 提出DRIFT框架，通过在强化微调全过程中系统性地激励输出多样性，破解了生成模型强化学习中的“多样性坍缩”难题，实现了强任务对齐与高生成多样性的平衡。 |
| [^376] | [The Fake Friend Dilemma: Relational Trust and the Political Economy of Conversational AI](https://arxiv.org/abs/2601.03222) | 本文提出“虚假朋友困境”概念，揭示对话式AI利用关系信任在机构利益冲突时可能操纵和剥削用户的风险。 |
| [^377] | [Foreground-Aware Dataset Distillation via Dual-Path Patch Selection](https://arxiv.org/abs/2601.02727) | 该论文提出了一种前景感知的数据集蒸馏方法，通过内容自适应的双路径图块选择策略，在保留重要前景结构的同时去除冗余背景内容，从而构建更紧凑且泛化性更好的合成数据集。 |
| [^378] | [Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach](https://arxiv.org/abs/2512.22793) | 该论文提出一种新颖的降维分解框架，将三维无人机可达-规避微分博弈分解为水平与垂直两个子博弈，从而使HJ可达性分析能够有效求解高维RA博弈问题。 |
| [^379] | [TS-DP: Reinforcement Speculative Decoding For Temporal Adaptive Diffusion Policy Acceleration](https://arxiv.org/abs/2512.15773) | 该论文首次揭示扩散策略是I/O受限而非计算受限的，并提出TS-DP框架，将投机解码与强化学习相结合，根据具身控制任务难度进行时间自适应的动态计算分配，从而在保持去噪质量的同时大幅加速扩散策略推理。 |
| [^380] | [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](https://arxiv.org/abs/2512.13494) | SkipCat是一种新颖的大语言模型低秩压缩框架，通过共享投影和块跳过机制，在相同压缩率下保留更高的秩，从而避免了传统低秩压缩中因秩过度削减而导致的显著性能下降。 |
| [^381] | [Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM](https://arxiv.org/abs/2512.12868) | 本研究提出基于频率的概率排序器（FBPR），仅利用朴素贝叶斯与大语料库中的概念-诊断共现统计，即可在MedQA诊断任务上达到与LLM相当的性能，且两者互补性强，混合使用可获得更佳效果。 |
| [^382] | [Detailed balance in large language model-driven agents](https://arxiv.org/abs/2512.10047) | 本文基于最小作用量原理，通过测量大语言模型智能体状态间的转移概率，发现其生成过程呈现细致平衡的倾向，表明LLM生成可能源于隐式学习的一类超越不同架构和提示模板的潜在势函数。 |
| [^383] | [The Theory of Strategic Evolution: Three Principles of Strategic Replicators](https://arxiv.org/abs/2512.07901) | 本文提出“战略演化理论”，以情境化选择、繁殖式传递和制度中介耦合三大构成性原则刻画“战略复制子”的一般逻辑，为人工智能时代智能体创建、配置与退役软件实例的演化过程提供了统一表示框架和选择-传递恒等式。 |
| [^384] | [SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting](https://arxiv.org/abs/2512.03620) | SELF是一种基于LLM注意力权重奇异值与特征值分解的内在指纹识别方案，无需输入依赖即可抵御虚假声明攻击和权重操纵，实现鲁棒且可扩展的模型知识产权取证。 |
| [^385] | [Cross-User Poisoning: User-Task Boundary Failures in Multi-User Collaborative Language Agents](https://arxiv.org/abs/2511.17671) | 该论文首次识别出多用户共享语言代理中的“跨用户投毒”（CUP）攻击，即攻击者注入共享上下文的恶意消息会在代理服务其他无辜用户时被错误执行，并提出 MURMUR 评估框架，在 Continua、ElizaOS 等真实系统中验证了该攻击的高成功率与持久性。 |
| [^386] | [Semantic World Models](https://arxiv.org/abs/2510.19818) | 本文提出“语义世界模型”，将世界建模重新表述为关于未来帧语义信息的视觉问答任务，通过监督微调将视觉语言模型训练为预测任务相关语义信息（而非重建像素）的世界模型，从而更好地对齐机器人规划目标。 |
| [^387] | [MultiFair: Multimodal Balanced Fairness-Aware Medical Classification with Dual-Level Gradient Modulation](https://arxiv.org/abs/2510.07328) | 本文提出 MultiFair 方法，通过在数据模态和人口群体两个层面动态调制训练梯度的方向与幅度，同时解决多模态医学分类中模态学习不均衡与群体公平性两大问题。 |
| [^388] | [From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow](https://arxiv.org/abs/2509.12443) | 本文提出一种自主智能体AI工作流，通过多个专门的LLM智能体协作，自动将遗留Fortran内核翻译、验证、测试、调试并优化为具有跨硬件性能可移植性的Kokkos C++代码。 |
| [^389] | [How K-12 Teachers Adapt Generative AI to Local Classroom Demands: Cultural Distance Between GenAI Defaults and Everyday Teaching Practices](https://arxiv.org/abs/2509.10780) | 该研究提出“文化距离”这一新概念，通过对南非、台湾和美国30名K-12教师的访谈，按教师所需努力程度划分出低、中、高三个层级，揭示了生成式AI默认文化与本地教学实践之间的适配差距，其中部分结构性差距是教师个人努力无法克服的。 |
| [^390] | [Re4: Scientific Computing Agent with Rewriting, Resolution, Review and Revision](https://arxiv.org/abs/2508.20729) | 该论文提出Re4科学计算智能体框架，通过顾问、审查员和程序员三个大语言模型的协作交互，构建“重写-解析-审查-修订”逻辑链，实现代码的迭代生成、调试与完善，以解决科学计算问题。 |
| [^391] | [Role-Augmented Intent-Driven Generative Search Engine Optimization](https://arxiv.org/abs/2508.11158) | 提出RAID方法，通过从源内容推断检索意图并进行4W多角色深度反思，实现了在无需访问目标查询及其反馈的情况下进行生成式搜索引擎优化。 |
| [^392] | [HiMA-Ecom: A Long-Horizon Memory Benchmark for Hierarchical Multi-Agent E-commerce Assistants](https://arxiv.org/abs/2506.19846) | 提出HiMA-Ecom——一个包含22.8K实例、面向分层多智能体电商助手的长时程记忆基准，并通过VR-GRPO方法（HiMA-R1）在端到端多智能体强化学习中有效控制上下文预算。 |
| [^393] | [Data-Driven Maintenance Optimization: Aligning Prognostic Learning with Decision Objective](https://arxiv.org/abs/2506.19698) | 提出估计-优化一体化（IEO）框架，直接以维护成本为预测学习目标进行数据驱动的维护决策优化，克服了传统先估计后优化方法的次优性问题，并提供了理论性能保证和高效训练算法。 |
| [^394] | [Doubly Robust Alignment for Large Language Models](https://arxiv.org/abs/2506.01183) | 提出了一种双重稳健的偏好优化算法（DRPO），只要偏好模型或参考策略中有一个被正确设定，即可实现一致的大语言模型对齐，有效解决了RLHF中的模型设定错误问题。 |
| [^395] | [On the Reliability and Stability of Selective Methods in Malware Classification Tasks](https://arxiv.org/abs/2505.22843) | 该论文提出Aurora评估框架，从置信度质量和运行弹性两个维度检验恶意软件分类器在分布偏移下的可靠性，揭示了仅凭性能指标无法反映分类器实际运行可靠性的问题。 |
| [^396] | [In-Context Learning for MIMO Turbo Receivers](https://arxiv.org/abs/2505.06175) | 本文提出EXT-ICLNet，一种基于Transformer的MIMO检测器，它将导频作为上下文示例，无需显式信道估计即可直接近似符号外验后验分布，并与现有Turbo接收机兼容，有效解决了短导频序列或信道模型失配导致的性能下降问题。 |
| [^397] | [An Embarrassingly Simple Data Selection Strategy for Million-Scale Multi-Modal Instruction-Tuning](https://arxiv.org/abs/2503.13383) | 该论文提出一种极其简单的百万规模多模态指令微调数据选择策略，以交互风格衡量结构多样性、按能力维度评估数据质量，无需计算复杂的嵌入或梯度即可高效筛选高质量指令数据。 |
| [^398] | [AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection](https://arxiv.org/abs/2502.09254) | 提出了AnomalyGFM——首个面向图异常检测的图基础模型，通过预训练学习图无关的正常与异常类表示，实现跨不同图数据集的零样本推理和少样本提示调优。 |
| [^399] | [Correcting Before Adapting: Gradient Correction for Federated Adaptive Optimization](https://arxiv.org/abs/2502.02727) | 提出FA-GC方法，在每个梯度进入自适应优化器的滑动平均之前先进行漂移校正，从而保证更新方向仅在全局目标的驻点处消失，优于仅在最终方向上补偿漂移的做法。 |
| [^400] | [Smart Surveillance: Deep Learning and Hybrid Approaches for Dynamic Scene Analysis, Object Detection and Motion Tracking](https://arxiv.org/abs/2412.05331) | 该研究统一比较了多种运动触发器与深度学习检测器作为监控事件触发录像方案的性能，发现每8帧运行一次CNN检测器能以仅9.8毫秒/帧的开销保留99.7%的事件帧，比基于背景减除的门控方法更省算力且更能抵抗摄像头抖动。 |
| [^401] | [The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization](https://arxiv.org/abs/2411.19193) | 提出截断势垒策略梯度框架，将截断水平 m 解释为失败风险的“价格”，证明训练价值可精确分解为失败前效用、余量成本与价格×风险三部分，并在凸正则化下实现了总体训练流的指数收敛。 |
| [^402] | [HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy](https://arxiv.org/abs/2411.05819) | 该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。 |
| [^403] | [She Works, He Works: A Curious Exploration of Gender Bias in AI-Generated Imagery](https://arxiv.org/abs/2407.18524) | 该研究通过并置对比男性和女性建筑工人的AI生成图像，揭示了生成式AI系统中女性被强调美感、男性被强调权威的性别偏见，同时展现了AI作为批判工具重塑性别规范的变革潜力。 |
| [^404] | [Fairness Definitions in Language Models Explained](https://arxiv.org/abs/2407.18454) | 本文通过系统性综述澄清了语言模型中各种公平性定义的内涵，解决了不同公平性概念之间区分模糊、适用情境不明确的问题。 |
| [^405] | [DyRA: Portable Dynamic Resolution Adjustment Network for Existing Detectors](https://arxiv.org/abs/2311.17098) | DyRA是一种动态分辨率调整网络，为现有的检测器提供图像特定的缩放因子，通过ParetoScaleLoss和BalanceLoss优化缩放因子以提高检测性能。 |
| [^406] | [Epsilon-Identifiability of Causal Quantities](https://arxiv.org/abs/2301.12022) | 本文提出“ε-可识别性”概念，通过将某些子群体的行为约束在足够窄的范围内，使原本不可识别的因果效应和反事实概率能够被近似识别，并将其应用于单元选择问题。 |
| [^407] | [Inferring Inference.](http://arxiv.org/abs/2310.03186) | 大脑具有一系列重复的规范计算单元，但神经表示是分布式的，因此如何定义规范分布式计算仍然是一个挑战。本文提出了一个数学框架，从大规模神经活动模式中推断出规范分布式计算。在算法级别上，提出了一个基于图结构模型的非线性信息传递算法。通过感知推理任务中的神经活动时间序列，可以找到神经活动与感知推理任务中的潜在因果之间的映射关系。 |

# 详细

[^1]: 蜂巢面元（Honeycomb-Surfels）：用于高效场景渲染与重建的紧凑六边形基元

    Honeycomb-Surfels: Compact Hexagonal Primitives for Efficient Scene Rendering and Reconstruction

    [https://arxiv.org/abs/2610.37668](https://arxiv.org/abs/2610.37668)

    Hexels是一种带有局部颜色场的平面六边形基元，用更少、更大的基元即可高效表示3D场景，在与内存匹配的2DGS对比中PSNR提升4.53-7.38 dB，且训练显存更低、渲染速度快2.31倍。

    

    新视角合成与三维重建随着辐射场和泼溅（splatting）技术迅速发展，但其内存和计算需求限制了它们在资源受限环境中的应用。我们提出了Hexels：一种新颖的平面六边形表面元素，带有用于高效3D场景和2D图像表示的一阶空间颜色场。Hexels利用更少、更大的基元来捕捉具有平滑变化外观的局部平面区域。其六边形支撑域在面积和边界上都是高效的，而局部颜色场则能够捕捉每个基元内部的外观变化。在34个场景的实验中，Hexels在与内存匹配的2DGS相比下，在多个数据集上将PSNR提升了4.53-7.38 dB，同时所需的训练显存更低。我们进一步对Hexels进行了消融实验，结果表明在单个基元层面，Hexels的性能比2DGS最高提升4.74 dB，同时使用更低的训练显存，并拥有快2.31倍的帧率（FPS）。即使在无约束的场景中，Hexels也提供了互补的……

    arXiv:2610.37668v1 Announce Type: cross  Abstract: Novel view synthesis and 3D reconstruction have advanced rapidly with radiance fields and splatting, yet their memory and computational demands limit their use in resource-constrained settings. We introduce Hexels: novel planar hexagonal surface elements with a first-order spatial color field for efficient 3D scene and 2D image representation. Hexels exploit locally planar regions with smoothly varying appearance using fewer, larger primitives. Their hexagonal support provides an area- and boundary-efficient domain, while the local color field captures appearance variation within each primitive. Across 34 scenes, Hexels improve PSNR over memory-matched 2DGS by 4.53-7.38 dB across datasets while requiring lower training VRAM. We further ablate Hexels by showing that per primitive, Hexels outperform 2DGS by upto 4.74 dB while using lower training VRAM and having 2.31 times faster FPS. Even in unconstrained settings, Hexels offers complem
    
[^2]: ThinkWorld：世界想象强化动力学链推理

    ThinkWorld: World Imagination Reinforces Chain-of-Dynamics Reasoning

    [https://arxiv.org/abs/2610.37666](https://arxiv.org/abs/2610.37666)

    提出ThinkWorld框架，以动力学链作为跨具身通用动作表示，将动作推理与世界动力学相结合，并借助世界想象的强化学习增强VLA模型的泛化能力。

    

    视觉-语言-动作模型在机器人操作领域已取得显著进展，但其泛化能力仍受限于带有动作标注的机器人示范数据的多样性与覆盖范围。人类视频提供了更广泛的行为与环境经验，然而从中学习面临可扩展数据与显式动作落地之间的根本性权衡。我们提出ThinkWorld，这是一个将动作推理扎根于世界动力学、并通过带有世界想象的强化学习加以增强的VLA学习框架。ThinkWorld将每个动作思考表示为动力学链，即一系列离散的运动token，它捕捉连接当前世界与未来状态的中间动力学过程，并作为跨具身形态的通用动作表示。我们首先使用带动作标注的示范数据对CoD进行初始化，然后预测每个CoD所引致的未来状态，并强化那些……（摘要原文在此处截断）

    arXiv:2610.37666v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have made substantial progress in robotic manipulation, but their generalization remains constrained by the diversity and coverage of action-labeled robot demonstrations. Human videos provide much broader behavioral and environmental experience, while learning from them faces a fundamental trade-off between scalable data and explicit action grounding. We introduce ThinkWorld, a VLA learning framework that grounds action reasoning in world dynamics and strengthens it through reinforcement learning with world imagination. ThinkWorld represents each action thought as a Chain-of-Dynamics (CoD), a sequence of discrete motion tokens that captures the intermediate dynamics connecting the current world to a future state and serves as a common action representation across embodiments. We first initialize CoD with action-labeled demonstrations, then predict the future induced by each CoD and reinforce those wh
    
[^3]: 注意力作为格点规范理论：关系仅通过威尔逊圈相互作用

    Attention as Lattice Gauge Theory: Relations Interact Only Through Wilson Loops

    [https://arxiv.org/abs/2610.37656](https://arxiv.org/abs/2610.37656)

    论文证明，在允许每个实体独立旋转自身坐标系的规范对称性下，基于威尔逊圈（闭合环路和乐）的边注意力机制不是众多候选设计之一，而是对称性唯一允许的形式——关系之间只能通过共享的闭合环路发生相互作用。

    

    关系通过它们所共享的实体进行复合——Alice是Bob的母亲，Bob是Carol的父亲，因此Alice是Carol的祖母——而且顺序至关重要，因为“父亲的母亲”并不等同于“母亲的父亲”；验证一个复合关系意味着闭合这个环。我们证明，如果每个实体的坐标系可以独立旋转（逐实体的参考系自由度），则可以得出三个结论。在纯规范扇区中，不共享任何环路的关系完全无法相互作用。三角形唯一规范不变的评分是其和乐（holonomy）H = T_ij T_jk T_ki 的类函数——即一个威尔逊圈。而边唯一的等变更新形式为 U = Ψ(H) T_ij，其中 Ψ 是共轭等变的。因此，基于威尔逊圈的边注意力并非众多设计中的一种，而是对称性所允许的唯一形式。三个研究群体分别掌握了这一形式的局部：知识图谱的交换算子嵌入构成了其阿贝尔（可交换）扇区；AlphaFo……（原文摘要在此处截断）

    arXiv:2610.37656v1 Announce Type: cross  Abstract: Relations compose through the entity they share - Alice is Bob's mother and Bob is Carol's father, so Alice is Carol's grandmother - and the order matters, since a father's mother is not a mother's father; verifying a composed relation means closing the loop. We show that if each entity's coordinate frame may be rotated independently (per-entity frame freedom), three things follow. In the pure-gauge sector, relations that share no cycle cannot interact at all. The only gauge-invariant score of a triangle is a class function of its holonomy $H = T_{ij} T_{jk} T_{ki}$ - a Wilson loop. And the only equivariant update of an edge is $U = \Psi(H)\, T_{ij}$ with $\Psi$ conjugation-equivariant. Edge attention by Wilson loops is therefore not one design among many but the unique form the symmetry permits. Three communities hold pieces of this form: the commuting-operator embeddings of knowledge graphs are its abelian (commuting) sector; AlphaFo
    
[^4]: KPPax：为大气化学中的机器学习预处理可微分动力学常微分方程

    KPPax: preprocess differentiable kinetic ODE for machine learning in atmospheric chemistry

    [https://arxiv.org/abs/2610.37655](https://arxiv.org/abs/2610.37655)

    KPPax 是一个动力学预处理器，能将 KPP 语言描述的大气化学机理转化为 JAX 中可微分、即时编译且可向量化的代码，并结合稀疏对数空间运算与 diffrax 隐式求解器，无需手工推导伴随代码即可获得精确敏感度。

    

    大气化学机理通常以动力学预处理器语言的形式发布，该预处理器可生成高效的 Fortran 或 C 代码，但无法融入开发机器学习模型所使用的可微分编程框架。我们提出了 KPPax，这是一个动力学预处理器，能够将以 KPP 语言编写的机理转化为 JAX 中可微分、即时编译且可向量化的化学代码。KPPax 在对数空间中通过稀疏运算计算速率定律，从而保持其可微分性，并利用 diffrax 库的隐式求解器以及前向和反向模式伴随方法对刚性系统进行积分，因此无需手工推导伴随代码即可获得精确的敏感度。我们使用 POLLU 基准机理演示了两个应用。在第一个应用中，由自动微分雅可比矩阵计算得到的 Fisher 信息矩阵可以在任何参数拟合之前预测哪些……（摘要原文在此处截断）

    arXiv:2610.37655v1 Announce Type: new  Abstract: Atmospheric chemical mechanisms are commonly distributed in the language of the Kinetic PreProcessor (KPP), which generates efficient Fortran or C code but does not fit into the differentiable programming frameworks in which machine-learning models are developed. We present KPPax, a kinetic preprocessor that generates differentiable, just-in-time compiled, and vectorisable chemistry code in JAX from mechanisms written in the KPP language. KPPax evaluates the rate law with sparse operations in log space, which keeps it differentiable, and integrates the stiff system with the implicit solvers and the forward- and reverse-mode adjoints of the \texttt{diffrax} library, so that exact sensitivities are obtained without hand-derived adjoint code. We demonstrate two applications with the POLLU benchmark mechanism. In the first, the Fisher information matrix computed from an automatically differentiated Jacobian predicts, before any fit, which of
    
[^5]: 大语言模型能否通过自由探索发现隐藏的物理定律？

    Can LLMs Uncover Hidden Physical Laws through Free Exploration?

    [https://arxiv.org/abs/2610.37649](https://arxiv.org/abs/2610.37649)

    该论文提出LongChang自动化框架，通过构建物理定律被改变的模拟世界，评估LLM能否在没有任务提示的情况下自主注意到异常物理现象，并通过自由实验和工具构建发现隐藏的物理定律。

    

    科学发现往往始于一个意想不到的观察，由此引出新的研究问题。在一个物理定律被改变的世界中，大语言模型（LLM）能否注意到意外行为——例如更重的物体下落得更慢——并在没有被告知该研究什么的情况下，通过实验发现其中的定律？为了评估这种能力，我们提出了LongChang，这是一个自动化框架，可以创建物理规律被改变的模拟世界，其中一个或多个定律与常规物理不同。被评估的LLM扮演探索者的角色，操纵物体并构建工具来探究自己提出的问题，而没有任何研究任务或关于这些定律的提示。独立的LLM智能体负责生成和审查场景、审查工具设计，并利用被改变的物理定律和探索记录来评估报告的正确性及其实验支持度。这些记录引导场景演化以进行进一步探索，使整个循环得以持续运行。

    arXiv:2610.37649v1 Announce Type: cross  Abstract: Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explorers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run 
    
[^6]: MuonGS：采用宽松权重范数控制的大语言模型预训练方法

    MuonGS: Pretraining LLMs with Relaxed Weight-Norm Control

    [https://arxiv.org/abs/2610.37647](https://arxiv.org/abs/2610.37647)

    该论文提出“累积有效学习率”分析框架，揭示原始Muon仅能实现对数级范数累积而MuonW/MuonH可达线性累积，并据此提出放宽权重范数控制的MuonGS方法以更高效地预训练大语言模型。

    

    权重衰减与显式的权重范数约束能够提升语言模型预训练中优化器的性能，然而对于这种控制为何有效、以及需要多强的控制，目前仍缺乏统一的理解。我们利用“累积有效学习率” C_T=\sum_{t<T}\eta_t/\|W_t\|_F（其中 \eta_t 为名义学习率，\eta_t/\|W_t\|_F 为相对于权重范数的有效学习率），对 Muon 在整个训练轨迹中的范数控制进行了分析。对于有界的更新方向，任意训练区间上的累积量会对相对权重位移形成约束。我们证明，原始版 Muon 只能实现对数级累积，这可能限制了探索，并有助于解释其经验进展较慢的现象；而 MuonW 与 MuonH 在所述条件下可保证至少线性的累积。进一步超越有界范数控制的框架，我们证明足够缓慢的范数增长可以保证更快的累积……（摘要原文在此处截断）

    arXiv:2610.37647v1 Announce Type: cross  Abstract: Weight decay and explicit weight-norm constraints improve optimizer performance in language model pretraining, yet a unified understanding of why such control helps and how much is needed remains incomplete. We analyze norm control in Muon over the entire training trajectory using the \emph{accumulated effective learning rate}, $C_T=\sum_{t<T}\eta_t/\|W_t\|_F$, where $\eta_t$ is the nominal learning rate and $\eta_t/\|W_t\|_F$ is the effective learning rate relative to the weight norm. For bounded update directions, accumulation over any training interval bounds relative weight displacement. We show that vanilla Muon can accumulate only logarithmically, potentially limiting exploration and helping explain its slower empirical progress, whereas MuonW and MuonH guarantee at least linear accumulation under the stated conditions. Going beyond bounded-norm control, we show that sufficiently slow norm growth guarantees faster accumulation th
    
[^7]: 预训练过程中什么在提升？利用时点模型追踪知识获取

    What Improves During Pretraining? Tracking Knowledge Acquisition with Point-in-Time Models

    [https://arxiv.org/abs/2610.37631](https://arxiv.org/abs/2610.37631)

    该论文提出时间变异性分数（TVS）来衡量知识在不同年份网络文本中的一致性，发现高变异性知识在预训练早期与低变异性知识的学习速度相似，但在后期提升更快。

    

    在跨越多年收集的网络语料库上预训练的语言模型既学习通用知识，也学习领域特定知识。随着预训练的进行，这些不同种类的知识是如何被习得的仍不清楚。最近在不同年份网络快照上训练的时点模型提供了一种探究知识流行度如何随时间变化的方法。利用这些模型，我们提出了时间变异性分数（TVS），定义为某个项目在各模型间负对数似然的变异系数。较高的分数表明该知识可能在多年网络文本中的表现不够一致。我们使用来自多个开源模型家族的六个模型评估了涵盖不同主题的六个基准。我们发现低TVS和高TVS项目在训练早期以相似的方式被学习，但高TVS项目在后期提升更快。在预训练运行结束时，高TVS项目在36个模型-基准组合中的34个中优于低TVS项目。

    arXiv:2610.37631v1 Announce Type: new  Abstract: Language models pretrained on web corpora collected over multiple years learn both general and domain-specific knowledge. How these different kinds of knowledge are learned as pretraining progresses remains unclear. Recent point-in-time models trained on web snapshots from different years provide a way to probe how knowledge prevalence varies over time. Using these models, we propose a temporal variability score (TVS), defined as the coefficient of variation of an item's negative log-likelihood across models. A higher score suggests that the knowledge may be less consistently represented across years of web text. We evaluate six benchmarks covering diverse topics using six models from multiple open-source model families. We find that low- and high-TVS items are learned similarly early in training, but high-TVS items improve faster later on. At the end of the pretraining run, high-TVS items outperform low-TVS items in 34 of 36 model-bench
    
[^8]: 稀有、相关却依然出错：记忆如何使大语言模型群体决策失效

    Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions

    [https://arxiv.org/abs/2610.37624](https://arxiv.org/abs/2610.37624)

    该论文提出CrossProfile诊断基准，通过交叉控制支持性、诱饵和无关事实的持有分布，揭示了LLM群体决策中两种记忆压缩启发式的可预测失败：基于相关性选择会重新引入切题的诱饵信息，而最稀有优先选择平均表现不优于随机选择。

    

    当一个LLM群体的讨论被压缩成一段简短记忆供最终决策者使用时，这段记忆应该保留什么？两种自然的启发式方法是保留最稀有的论断，或保留与决策标准相关的论断。在隐藏档案任务上，两者看似都很有吸引力，因为一个事实被多少人持有与它是否有帮助是相互混淆的。我们提出了CrossProfile，一个包含60个模板、答案经过自动验证的诊断基准，其中支持性事实、诱饵事实和无关事实的持有分布被交叉设计，而其文本内容保持固定（共480个任务）。在Qwen2.5-7B、Llama-3.1-8B和两个DeepSeek模型上，两种启发式方法都在可预见的地方失败。基于相关性选择会重新引入诱饵信息，因为诱饵是切题的：在经典的隐藏档案配置下，即使是一个完美的相关性过滤器也比最稀有优先选择低23到31分。最稀有优先选择平均而言并不优于随机选择，且仅在……

    arXiv:2610.37624v1 Announce Type: new  Abstract: When an LLM group's discussion is compressed into a short memory for a final decision maker, what should the memory keep? Two natural heuristics keep the rarest claims or the claims relevant to the decision criteria. On hidden-profile tasks both look attractive, because how widely a fact is held is confounded with whether it helps. We introduce CrossProfile, a diagnostic benchmark of 60 templates with automatically verified answer keys, in which the ownership of supporting, decoy, and irrelevant facts is crossed while their text is held fixed (480 tasks). Across Qwen2.5-7B, Llama-3.1-8B, and two DeepSeek models, both heuristics fail in predictable places. Selecting by relevance readmits decoys, because decoys are on topic: even a perfect relevance filter loses 23 to 31 points to rarest-first selection in the classic hidden-profile configuration. Rarest-first selection is no better than random selection on average and wins only where the 
    
[^9]: VideoLVR：基于接地视频证据的潜在视觉推理

    VideoLVR: Latent Visual Reasoning over Grounded Video Evidence

    [https://arxiv.org/abs/2610.37613](https://arxiv.org/abs/2610.37613)

    VideoLVR 通过从相关视频时刻的局部区域构建潜在视觉证据，为中间推理步骤提供视觉支持，在不大幅扩展上下文的前提下解决了视频推理中的视觉遗忘问题。

    

    思维链推理极大地提升了多模态大语言模型（MLLM）的视频理解能力。然而，基于语言的推理轨迹并未提供一种可靠的机制让模型重新审视相关的视觉证据，从而导致视觉遗忘，并削弱模型对视频内容进行推理的能力。现有方法主要通过两种方式在推理过程中保持视觉接地：其一是使用文本坐标指代证据，但这只能指定符号化的位置而非视觉内容本身；其二是将裁剪的区域或帧重新插入多模态上下文中，虽保留了视觉细节，却以大幅扩展上下文为代价。为解决这些局限，我们提出了 VideoLVR，它从相关视频时刻的局部区域构建潜在视觉证据，从而为中间推理步骤提供视觉支持。为监督潜在视觉证据的生成，我们引入了一种……（摘要在此处被截断）

    arXiv:2610.37613v1 Announce Type: cross  Abstract: Chain-of-thought reasoning has substantially advanced video understanding in multimodal large language models (MLLMs). However, language-based reasoning traces do not provide a reliable mechanism for models to revisit relevant visual evidence, leading to visual forgetting and impairing their ability to reason over video content. Existing approaches maintain visual grounding during reasoning in two main ways: they either refer to evidence using textual coordinates, which specify symbolic locations rather than the visual content itself, or reinsert cropped regions or frames into the multimodal context, preserving visual details at the cost of substantial context expansion. To address these limitations, we introduce VideoLVR, which constructs latent visual evidence from local regions at relevant video moments, thereby providing visual support for intermediate reasoning steps. To supervise latent visual evidence generation, we introduce a 
    
[^10]: 超越答案变化：用于局部音频证据的匹配同类对照

    Beyond Answer Changes: Matched Same-Class Controls for Local Audio Evidence

    [https://arxiv.org/abs/2610.37605](https://arxiv.org/abs/2610.37605)

    该论文提出匹配同类对照协议，通过控制区间、响度、供体拟合等编辑因素，比较同类与不同类声音事件替换对音频-语言模型答案的影响，证明模型答案变化主要源于对事件类别的响应而非对波形重写的敏感性，从而为局部音频证据提供更严格的检验方法。

    

    对于音频-语言模型，局部声音事件编辑后发生的答案变化，可能反映的是模型对事件类别的响应、对波形重写的敏感性，或两者兼有。我们比较了被查询事件的匹配同类替换与不同类替换：两者使用相同的时间区间、响度估计、供体拟合和交叉渐变处理，而同类替换会保留正确答案。在600个合成声景以及来自两个架构家族的三个模型上，不同类替换比同类替换多产生49.7至60.9个百分点的答案变化；同类替换后的准确率与原始准确率的差距保持在5.2个百分点以内。在原始答案和同类答案均正确的试验中，针对所插入类别的声景加权选择性准确率为73.7%至81.9%，其计算方式是先在每个声景内对符合条件的试验取平均，再对符合条件的声景进行等权重加权。该协议用于检验模型答案如何响应……

    arXiv:2610.37605v1 Announce Type: cross  Abstract: For audio-language models, an answer change after a local sound-event edit can reflect a response to the event's class, sensitivity to waveform rewriting, or both. We compare matched same-class and different-class replacements of a queried event. Both use the same interval, level estimate, donor fitting, and crossfade, while the same-class replacement preserves the correct answer. Across 600 synthetic soundscapes and three models from two architecture families, different-class replacements produce 49.7 to 60.9 percentage points more answer changes than same-class replacements. Same-class accuracy remains within 5.2 percentage points of original accuracy. Among trials with correct original and same-class answers, soundscape-weighted selective accuracy for the inserted class is 73.7 to 81.9%, averaging eligible trials within each soundscape and then weighting eligible soundscapes equally. This protocol tests how model answers respond to 
    
[^11]: MoI-Bench：面向移动设备的图像生成与编辑基准测试

    MoI-Bench: A Benchmark for Image Generation and Editing on Mobile Devices

    [https://arxiv.org/abs/2610.37599](https://arxiv.org/abs/2610.37599)

    提出了基于移动用户需求的图像生成与编辑基准测试MoI-Bench（包含64个子任务、1150个样本、16种操作和8个视觉目标类别），并配套提出任务特定的评估方法MoI-Eval，以量化端侧模型与服务器端模型的能力差距并指导模型改进。

    

    开发端侧图像模型需要考虑实际用户需求、任务类型以及设备资源限制。然而，现有的基准测试在识别端侧模型与服务器端模型之间的能力差距方面提供的指导十分有限。我们提出了MoI-Bench，这是一个基于移动用户需求构建的基准测试，包含64个子任务和1150个样本。该基准定义了16种操作（如移除和替换），以及八个视觉目标类别（涵盖人像和文本等内容）。这两者构成两个正交维度，其适用的组合定义了具体任务，例如卸除面部妆容。为了更好地指导模型改进，我们进一步提出了MoI-Eval，它引入了涵盖指令遵循、视觉质量和细节保留等维度的任务特定评价标准。我们还构建了样本级别的评估问题，并将其分数汇总为沿操作和目标维度展开的整体质量与能力画像。

    arXiv:2610.37599v1 Announce Type: cross  Abstract: Developing on-device image models requires considering practical user needs, task types and device resource constraints. However, existing benchmarks provide limited guidance on identifying capability gaps between on-device models and server-side models. We introduce MoI-Bench, a benchmark grounded in mobile user needs, comprising 64 subtasks and 1150 samples. It defines 16 operations, such as removal and replacement, and eight visual target categories covering content such as portraits and text. These form two orthogonal axes whose applicable combinations define specific tasks, such as removing facial makeup. To better guide model improvement, we further propose MoI-Eval, which introduces task-specific criteria covering instruction following, visual quality and detail preservation. We further construct sample-level evaluation questions and aggregate their scores into overall quality and capability profiles along the operation, target 
    
[^12]: GAS-PFN：用于零样本交通预测的图感知时空先验数据拟合网络

    GAS-PFN: Graph-Aware Spatiotemporal Prior-Data Fitted Networks for Zero-Shot Traffic Forecasting

    [https://arxiv.org/abs/2610.37598](https://arxiv.org/abs/2610.37598)

    GAS-PFN通过随机游走扩散将道路拓扑融入合成任务先验，提出了显式建模图结构依赖的图感知时空PFN，实现了无需针对特定网络训练的零样本交通预测。

    

    交通预测需要对复杂的时间动态以及由道路拓扑结构所施加的空间依赖关系进行建模。有监督的基于图的模型虽然取得了优异的性能，但通常针对特定数据集和传感器图进行训练，使得将其部署到新网络或不断演化的网络时成本高昂。先验数据拟合网络（Prior-Data Fitted Networks, PFNs）通过从合成任务先验中学习上下文推理，提供了一种零样本替代方案，并为编码领域特定假设提供了显式机制。然而，现有的预测类PFN并未显式表示支配交通动态的图结构依赖关系。为了将图感知的交通建模与零样本自适应相结合，我们提出了图感知时空先验数据拟合网络（GAS-PFN）。GAS-PFN通过随机游走扩散将采样的道路拓扑结构融入其合成任务先验中，从而教会模型在预训练期间学习图耦合的动态特性。

    arXiv:2610.37598v1 Announce Type: cross  Abstract: Traffic forecasting requires modeling complex temporal dynamics and the spatial dependencies imposed by road topology. Supervised graph-based models achieve strong performance, but are often trained for a particular dataset and sensor graph, making deployment to new or evolving networks costly. Prior-Data Fitted Networks (PFNs) offer a zero-shot alternative by learning in-context inference from synthetic task priors, which provide an explicit mechanism for encoding domain-specific assumptions. Existing forecasting PFNs, however, do not explicitly represent the graph-structured dependencies governing traffic dynamics. To combine graph-aware traffic modeling with zero-shot adaptation, we introduce the Graph-Aware Spatiotemporal Prior-Data Fitted Network (GAS-PFN). GAS-PFN incorporates sampled road topology into its synthetic task prior through random-walk diffusion, teaching the model to learn graph-coupled dynamics during pretraining. I
    
[^13]: 理解潜在世界模型的局限性

    Understanding the limits of latent world models

    [https://arxiv.org/abs/2610.37588](https://arxiv.org/abs/2610.37588)

    该论文提出“忠实潜在世界模型”概念及潜在展开一致性（LRC）评估指标，发现当前所有最先进的潜在世界模型在解决任务时都并未真正忠实地建模环境的动态。

    

    世界模型被广泛提出作为使智能体能够在长时空范围内进行推理的核心组件。然而，当前的评估往往侧重于任务性能，而对于学习到的世界模型是否获得了能够反映环境底层结构的预测性潜在动态，所提供的洞察十分有限。在本文中，我们引入了“忠实潜在世界模型”的概念，将其定义为环境转移映射与学习到的潜在动态之间的等变性条件。我们提出了潜在展开一致性（Latent Rollout Consistency, LRC）作为评估动作条件潜在预测的指标。令人惊讶的是，我们发现所研究的所有最先进的潜在世界模型中，没有一个是通过真正忠实地建模环境动态来解决任务的。

    arXiv:2610.37588v1 Announce Type: cross  Abstract: World models are widely proposed as core components for enabling agents to reason over long temporal and spatial horizons. Yet current evaluations often emphasize task performance while providing limited insight into whether learned world models acquire predictive latent dynamics that reflect the underlying structure of the environment. Here we introduce the concept of faithful latent world models as an equivariance condition between the environment's transition map and the learned latent dynamics. We propose latent rollout consistency (LRC) as a metric to evaluate action-conditioned latent prediction. Strikingly, we find that across the studied state of the art latent world models, none solves the task by actually faithfully modeling the dynamics of the environment.
    
[^14]: Timestral（Timestral 时间序列基础模型）

    Timestral

    [https://arxiv.org/abs/2610.37585](https://arxiv.org/abs/2610.37585)

    Timestral 是一系列面向概率预测的时间序列基础模型，通过解耦视界预测（DHF）、原生支持已知未来协变量的框架以及“先大规模合成数据预训练、后真实与合成混合数据后训练”的两阶段流程，在 GIFT-Eval、TIME 和 fev-bench 上取得了最先进的性能。

    

    我们提出 Timestral，这是一个面向概率预测的时间序列基础模型家族，包含 5M、28M、123M 和 209M 四种参数规模。Timestral 在 GIFT-Eval、TIME 和 fev-bench 上创造了新的最先进纪录，其中 Timestral-209M 和 Timestral-123M 在所有指标上包揽前两名。我们通过三项关键技术实现这一成果：解耦视界预测（Decoupled Horizon Forecasting, DHF），它将预测 token 与因果编码的上下文分离，从而在训练更长预测视界时无需屏蔽上下文，并且每个观测值既是预测目标又是后续预测的上下文；一个协变量框架，在因果架构中原生支持已知的未来协变量；以及一个两阶段流水线，先完全基于多样化的大规模合成语料库进行预训练，再在扩展上下文下的真实与合成数据混合集上进行后训练。本报告介绍了模型设计及其基准测试结果，并发布了全部四个模型的权重。

    arXiv:2610.37585v1 Announce Type: new  Abstract: We introduce Timestral, a family of four time series foundation models for probabilistic forecasting, with 5M, 28M, 123M, and 209M parameters. Timestral sets a new state of the art on GIFT-Eval, TIME, and fev-bench, where Timestral-209M and Timestral-123M take the top two places on every metric. We achieve this with three ingredients: Decoupled Horizon Forecasting (DHF), which separates prediction tokens from the causally encoded context, so that longer horizons are trained without masking context and each observed value is both a forecast target and context for later forecasts; a covariate framework that natively supports known-future covariates within a causal architecture; and a two-stage pipeline that pretrains exclusively on a diverse, large-scale synthetic corpus before post-training on a mix of real and synthetic data with extended context. This report describes the model design and its benchmark results. Weights for all four mode
    
[^15]: REGER：基于语义ID建模物品间关系的生成式推荐

    REGER: Modeling inter-item Relations with Semantic IDs for Generative Recommendation

    [https://arxiv.org/abs/2610.37584](https://arxiv.org/abs/2610.37584)

    提出REGER统一框架，通过生成式语义检索预训练使GR模型继承物品间语义相似性，并将物品间关系显式融入语义预训练、历史编码和SID生成的全过程。

    

    个性化推荐依赖于从在线内容和用户交互中学习物品之间的语义关系和协同关系。生成式推荐（GR）通过自回归地生成物品的离散语义ID（SIDs），提供了一种有前景的范式。然而，随机初始化的SID词元嵌入以及从零开始训练的GR模型并不能直接继承物品固有的语义关系，而物品间的协同关系主要是通过词元级注意力机制和下一词元预测监督被隐式学习的。为了解决这一局限，我们提出了REGER，这是一个统一框架，将物品间关系显式地融入语义预训练、历史编码和SID生成之中。我们通过生成式语义检索对GR模型进行预训练，使其通过学习生成语义邻居的SID来继承物品间的语义相似性。我们将全局注意力与局部注意力相结合……（摘要原文在此处不完整）

    arXiv:2610.37584v1 Announce Type: cross  Abstract: Personalized recommendation depends on learning semantic and collaborative relations among items from online content and user interactions. Generative recommendation (GR) offers a promising paradigm by autoregressively generating discrete semantic IDs (SIDs) of items. However, randomly initialized SID token embeddings and GR models trained from scratch do not directly inherit the semantic relations inherent in items, while inter-item collaborative relations are learned largely implicitly through token-level attention and next-token supervision. To address this limitation, we propose REGER, a unified framework that explicitly incorporates inter-item relations into semantic pretraining, history encoding, and SID generation. We pretrain the GR model through generative semantic retrieval, enabling it to inherit inter-item semantic similarity by learning to generate semantic neighbors' SIDs. We combine global attention with local attention 
    
[^16]: TeamGym：一个面向分布式信息条件下AI智能体团队决策的训练场

    TeamGym: A Gym for Team Decision-Making by AI Agents with Distributed Information

    [https://arxiv.org/abs/2610.37571](https://arxiv.org/abs/2610.37571)

    TeamGym是基于真实法律、金融、医疗和软件工程语料构建的团队决策评测与训练环境，通过引入“关键信息”这一概念发现当前语言模型智能体在信息分布共享的团队决策场景中难以有效整合信息、表现明显下降。

    

    许多现实世界的决策需要团队整合分布于各参与者之间的信息。然而，AI智能体目前主要仍在孤立状态下进行训练和基准测试。我们提出了TeamGym，这是一个用于评估和训练基于语言模型的智能体的环境，其中的团队决策问题构建自真实世界的法律、金融、医疗和软件工程语料库。TeamGym引入了一种方法来识别一个最小的事实集合，其中每个元素对于正确的最终决策都是必要的，且所有元素合起来是充分满足的，我们将其称为“关键信息”。这使得TeamGym既能控制关键信息的分布方式，又能追踪智能体在进行团队决策时识别和整合关键信息的能力。通过对12个源语料库上的8个语言模型进行检验，我们发现当前的AI智能体往往无法有效地共享信息：当关键信息分散在不同参与者之间时，智能体的表现会显著下降。

    arXiv:2610.37571v1 Announce Type: new  Abstract: Many real-world decisions require teams to integrate information distributed across participants. Yet AI agents are still primarily trained and benchmarked in isolation. We introduce TeamGym, an environment for evaluating and training language model-based agents on team decision-making problems constructed from real-world legal, finance, medical, and software engineering corpora. TeamGym introduces a method to identify a minimal set of facts, each element of which is necessary and all of which are collectively sufficient for a correct final decision, which we call pivotal information. This enables TeamGym to both control the distribution of pivotal information and track how well agents identify and integrate pivotal information when making team decisions. Examining 8 language models on 12 source corpora, we find that current AI agents often fail to effectively share information: Distributing pivotal information across participants lowers
    
[^17]: Gym-Anything-RT：为视觉-语言智能体扩展实时交互环境

    Gym-Anything-RT: Scaling Real-Time Environments for Vision-Language Agents

    [https://arxiv.org/abs/2610.37567](https://arxiv.org/abs/2610.37567)

    提出了Gym-Anything-RT，一个基于六种实时交互能力分类体系、通过智能体流水线自动生成视觉小游戏的基准环境，用于系统评估视觉-语言智能体的实时交互能力。

    

    AI智能体在那些它们在动作之间有充裕时间进行思考的领域（如数学、软件工程以及桌面应用程序任务执行）已取得重大进展。然而，这些模型在需要实时交互的场景（如玩电子游戏或操作物理设备）中的表现如何尚不清楚。在这些场景中，环境状态可能在智能体进行推理时、甚至在底层模型执行前向传播的过程中就发生演变。为了深入了解视觉-语言智能体的实时交互能力，我们提出了Gym-Anything-RT。我们的核心思想是构建需要多种实时交互所需能力组合的视觉小游戏。Gym-Anything-RT使用智能体流水线，基于实时交互所需的六种能力分类体系自动生成此类环境，包括感知能力等。

    arXiv:2610.37567v1 Announce Type: cross  Abstract: AI agents have made substantial progress in domains where they have a non-trivial amount of time to deliberate between actions, such as mathematics, software engineering, and performing tasks on desktop applications. It is less clear how such models fare in settings that demand real-time interaction, such as playing a video game or operating a physical device. In these settings, the environment state may evolve as the agent is reasoning, or even as the underlying model is performing a forward pass. To gain insight into the real-time interaction capabilities of vision-language agents, we introduce Gym-Anything-RT. Our key idea is to construct visual minigames that demand a portfolio of capabilities required for interactive real-time settings. Gym-Anything-RT uses an agentic pipeline to automatically generate such environments based on a taxonomy of six capabilities required for real-time interaction, including the ability to perceive th
    
[^18]: 当不确定性量化失效时：临床类别不平衡下选择性预测的结构性脆弱性与修正方法

    When uncertainty quantification fails: structural vulnerabilities and corrections for selective prediction under clinical class imbalance

    [https://arxiv.org/abs/2610.37561](https://arxiv.org/abs/2610.37561)

    该研究揭示了在临床类别严重不平衡的预后任务中，标准不确定性度量会退化为类别归属的代理指标并系统性惩罚少数类患者，导致选择性预测安全机制失效，并提出了一种能在合成场景与多发性硬化、帕金森病、阿尔茨海默病等真实队列中保持有界、平衡风险的修正框架。

    

    背景：选择性预测系统通过将模糊病例转交给人类专家、同时对高置信度决策进行自动化处理，从而提升临床人工智能的安全性。然而，这一安全机制完全依赖于一个前提假设：不确定性分数能够准确反映临床上的模糊性。在以严重类别失衡为特征的预后任务中（即所关注的临床结局属于少数类），这一假设会失效。标准的不确定性度量不再能够捕捉与临床一致的模糊性；相反，它们沦为类别归属的代理指标，无论少数类患者个体的预测清晰程度如何，都会系统性地对其造成不利影响。结果：在涵盖从严重类别失衡到接近平衡条件的56个合成场景以及三个神经退行性疾病队列（多发性硬化症、帕金森病、阿尔茨海默病）中，我们的框架始终保持了有界且平衡的风险……（原文摘要在此处截断，内容不完整）

    arXiv:2610.37561v1 Announce Type: cross  Abstract: Background. Selective prediction systems improve the safety of clinical artificial intelligence by deferring ambiguous cases to human experts while automating high-confidence decisions. However, this safety mechanism relies entirely on the premise that uncertainty scores accurately reflect clinical ambiguity. In prognostic tasks characterized by severe class imbalance, where the clinical outcome of interest is a minority, this assumption breaks down. Standard uncertainty measures cease to capture clinically aligned ambiguity; instead, they become a proxy for class membership, systematically penalizing minority patients regardless of their individual predictive clarity.   Results. Across 56 synthetic conditions and three neurodegenerative cohorts spanning severe class imbalance to near-balanced conditions (multiple sclerosis, Parkinson's Disease, Alzheimer's Disease), our framework consistently maintained bounded, balanced risk across t
    
[^19]: 面向铰接物体操作的运动学约束参考评估

    Kinematic Constraint-Referenced Evaluation for Articulated Object Manipulation

    [https://arxiv.org/abs/2610.37550](https://arxiv.org/abs/2610.37550)

    提出约束参考策略（CRS），通过将末端执行器指令速度投影到物体关节轴上，将沿关节的有效进展与偏离约束的无效动作区分开，从而诊断出传统二元成功率无法归因的迁移失败。

    

    操作策略目前主要以二元任务成功与否来评估，这种方式无法判断所下达的动作是否是物体运动学所允许的动作。我们提出了约束参考策略（CRS），它将末端执行器的指令速度旋量投影到物体的关节轴上，从而将沿关节的有效进展与偏离约束的无效动作区分开来。CRS只需要一个关节模型以及由其推导出的跟随关节的脚本控制器，而不需要记录的演示数据或学习到的评判器。在74个仿真铰接物体上，传统成功率评估只能将三种迁移失败记录为较低的比率，却无法归因原因，而CRS的两个通道则为每种失败提供了独特的特征签名。一个用门训练的行为克隆策略在110次回合中成功打开了59次抽屉，而在这些成功的回合中，它所指令的离轴旋转量约为用抽屉训练的策略的40倍（1.27 对比 0.032 rad/s）。相对于用同一关节轴评分的脚本控制器，CRS的判定……

    arXiv:2610.37550v1 Announce Type: cross  Abstract: Manipulation policies are evaluated predominantly by binary task success, which cannot tell whether the commanded gesture is the one the object's kinematics admit. We introduce Constraint-Referenced Strategy (CRS), which projects the commanded end-effector twist onto the object's joint axis, separating progress along the joint from off-constraint waste. CRS requires a joint model and a joint-following scripted controller derived from it, but no recorded demonstration or learned judge. On 74 simulated articulated objects, success registers three transfer failures as lower rates but attributes none, while the two CRS channels give each a distinct signature. A door-trained behavior-cloning policy opens drawers in 59 of 110 episodes, and in these passes it commands about $40\times$ the off-axis rotation of a drawer-trained policy's passes ($1.27$ vs.\ $0.032$~rad/s). Relative to a scripted controller scored with the same axis, CRS verdicts
    
[^20]: 何时、何地以及修正什么：面向机器人策略的人类引导自适应残差修正

    When, Where, and What to Correct: Human-Guided Adaptive Residual Correction for Robot Policies

    [https://arxiv.org/abs/2610.37547](https://arxiv.org/abs/2610.37547)

    提出轻量级后训练模块ARC，利用强化学习在冻结策略之上学习残差修正，并通过稀疏人类修正确定残差控制激活的“时机”与自主探索的“方向”，将真实机器人精密操作任务的平均成功率从33.3%提升至75%。

    

    能够可靠执行操作轨迹大部分内容的模仿学习策略，仍可能在需要“最后一英里”精度的局部区域失败。更新整个策略或在所有地方探索残差动作，是将这种局部失败当作全局学习问题来处理。我们提出了自适应残差修正（ARC），这是一个轻量级的后训练模块，它使用强化学习（RL）在冻结策略之上学习残差修正，同时利用稀疏的人类修正来组织自主探索。ARC并不将人类修正视为模仿目标，而是利用干预前状态来确定残差控制“何时”应被激活，利用修正几何来确定自主细化应在残差动作空间的“何处”进行探索；任务奖励则决定最终执行“什么”残差。在真实机器人上的三项精密操作任务中，ARC将平均任务成功率从33.3%提升至75

    arXiv:2610.37547v1 Announce Type: cross  Abstract: Imitation policies that reliably execute most of a manipulation trajectory can still fail in localized regions requiring last-mile precision. Updating the full policy or exploring residual actions everywhere treats this local failure as a global learning problem. We introduce Adaptive Residual Correction (ARC), a lightweight post-training module that uses reinforcement learning (RL) to learn a residual correction on top of a frozen policy while using sparse human corrections to structure autonomous exploration. Rather than treating human corrections as imitation targets, ARC uses pre-intervention states to determine 'when' residual control should become active and correction geometry to determine 'where' in residual action space autonomous refinement should explore; task reward determines 'what' residual is ultimately executed. Across three precision-manipulation tasks on a real robot, ARC improves average task success from 33.3% to 75
    
[^21]: GroundControl：基于卫星图像条件化的多视角扩散模型，实现无需重建的新视角合成

    GroundControl: Satellite-Conditioned Multi-View Diffusion for Reconstruction-Free Novel View Synthesis

    [https://arxiv.org/abs/2610.37545](https://arxiv.org/abs/2610.37545)

    该论文提出GroundControl，一种将卫星图像作为额外条件信息的多视角扩散模型，无需3D场景重建即可从少量街景图像生成既逼真又符合真实场景布局的室外场景新视角。

    

    多视角或视频扩散模型近来已成为从稀疏图像集合中进行新视角合成的最先进方法。虽然它们能够生成高质量的图像，但在合成超出输入视角范围的视图（即场景外推）时，往往会幻觉出不真实的空间布局。我们提出通过使用卫星图像作为额外的信息源来对多视角生成模型进行条件化，从而解决室外场景的这一挑战。我们引入了一种新的卫星条件化多视角扩散模型，在生成新视角时结合来自卫星图像和街景级输入图像的条件信息。我们基于交叉注意力的条件化机制使我们能够保留预训练的多视角扩散先验，从少量源视角生成无缝、逼真的新视角，同时遵循卫星图像中所呈现的场景布局。我们的模型不使用3D场景重建，

    arXiv:2610.37545v1 Announce Type: cross  Abstract: Multi-view or video diffusion models have recently emerged as state-of-the-art approaches for novel-view synthesis from sparse image collections. While they can produce high quality images, they tend to hallucinate unrealistic spatial layouts when synthesizing views outside the input views (scene extrapolation). We propose to address this challenge for outdoor scenes by using satellite images as an additional information source to condition the multi-view generative model. We introduce a new satellite-conditioned multi-view diffusion model that combines conditioning from a satellite image and street-level input images when generating novel-views. Our cross-attention based conditioning allows us to retain pretrained multi-view diffusion priors to generate seamless, realistic, novel views from a handful of source views, while also adhering to the scene layout as seen from satellite images. Our model does not use 3D scene reconstruction, 
    
[^22]: 哪只手迟了？面向双手机器人工具使用中故障恢复的故障诊断

    Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use

    [https://arxiv.org/abs/2610.37533](https://arxiv.org/abs/2610.37533)

    该论文通过双手机器人工具使用任务揭示了不同感知通道可识别的故障类型差异——例如单手传感器无法判断另一只手的延迟或复合故障，并据此提出在故障原因不可识别时进行升级的修复策略，从而在有限尝试次数内实现更有效的故障恢复。

    

    面向机器人操作的故障诊断器越来越多地采用视觉-语言模型（VLM），其训练与评估通常基于对成功演示施加扰动所产生的单臂故障。而双手工具使用带来了其他类型的故障：工具在仍被握持的情况下从插口滑出、两个正确的动作在不同时间被执行，以及一次尝试同时因两个原因而失败。在一个模拟的双手机器人工具使用任务中（其故障原因可从记录的轨迹中读出），我们测量了每种感知通道能让诊断器分辨出什么信息。工具的滑出可以从工具手的传感器读出；哪只手迟了可以从双臂的传感器判断，而单手传感器只能读出自己手的延迟以及至多另一只手的部分延迟；当工具手迟了时，这次迟到的尝试是否同时推得力度不足，无法从被动感知中识别出来。在两次尝试内，一种在故障原因不可识别时进行升级（escalate）的修复策略，能够修复与……（原文摘要在此处截断）

    arXiv:2610.37533v1 Announce Type: cross  Abstract: Failure diagnosers for robot manipulation are increasingly vision--language models (VLMs), trained and evaluated on single-arm failures produced by perturbing a successful demonstration. Bimanual tool use adds failures of other kinds: a tool that slips out of its socket while still held, two correct gestures made at different times, and attempts that fail for two reasons at once. On a simulated bimanual tool-use task whose failure causes are read from the recorded trace, we measure what each sensing channel lets a diagnoser tell. The tool's slip is read on the tool hand; which hand was late is read on both arms, while a sensor on one hand reads its own hand's delay and at most part of the other's; and whether a late attempt was also pushed too weakly is not identifiable from passive sensing when the tool hand is late. Within two attempts, a repair policy that escalates when the cause is not identifiable repairs as many failures as one 
    
[^23]: 结构差距：当更多数据无法取代结构化推理时

    The Structure Gap: When More Data Cannot Replace Structured Reasoning

    [https://arxiv.org/abs/2610.37523](https://arxiv.org/abs/2610.37523)

    本文提出“结构差距”这一概念，指出对于由求解器、物理定律等已知结构决定或约束的任务，纯数据驱动的学习器在有限资源下无法通过增加数据完全替代结构化推理，其差距集中体现在有限数据辨识、外推与认证能力上。

    

    现代人工智能中存在一种根深蒂固的信念：在足够多数据上训练的足够大的学习器可以让显式的领域知识变得可有可无。本文考察了这一信念的一个局限性。我们定义了结构编码问题：即其输入输出映射由已知结构（如求解器、对称群、图、物理定律、测量算子、文法、组合搜索空间或优化目标）所决定或严格约束的任务。对于此类任务，核心问题不在于表达能力——稠密神经网络可能已经包含所需的映射——而在于有限数据下的辨识、外推与认证。我们提出了结构差距的概念：即获得支配性结构的方法与仅获得样本的学习器之间在有限资源下的差距。这一论断最有用的最强形式是渐近性的而非绝对性的：随着覆盖范围的增长，结构无关的学习可能逼近结构化求解器（原文摘要在此处被截断）。

    arXiv:2610.37523v1 Announce Type: cross  Abstract: A persistent belief in modern AI is that sufficiently large learners trained on sufficiently much data can make explicit domain knowledge optional. This paper examines a limitation of that belief. We define structure-encoded problems: tasks whose input-output map is determined, or sharply constrained, by known structure such as a solver, symmetry group, graph, physical law, measurement operator, grammar, combinatorial search space, or optimization objective. For such tasks, the central question is not expressivity--a dense neural network may contain the desired map--but finite-data identification, extrapolation, and certification. We introduce the structure gap: the finite-resource gap between a method that receives the governing structure and a learner that receives only examples. The strongest useful form of the claim is asymptotic rather than absolute: structure-agnostic learning may approach a structured solver as coverage grows, b
    
[^24]: 大语言模型的正交化与参数化多属性导向方法

    Orthogonal and Parametrized Multi-Attribute Steering of LLMs

    [https://arxiv.org/abs/2610.37517](https://arxiv.org/abs/2610.37517)

    该论文提出参数化导向方法将激活设定在期望的特质强度上，并通过向量正交化解耦多特质间的干扰，实现了大语言模型多属性的同时、可解释且互不干扰的精确控制。

    

    激活导向提供了一种在推理时修改大语言模型行为的有效且高效的方法。然而，目前仍不清楚如何施加导向，才能使信号始终保持强烈，且导向强度具有普遍可解释的含义。我们首先通过为每个导向向量识别最可导向的区间，并相应地重新缩放导向范围来解决这一问题。在此基础上，我们引入了一种更具可解释性的导向方法，称为参数化导向，它不是将激活推向某个方向，而是将激活设定在期望的特质强度上。接下来，在多特质导向中，我们发现简单地将向量组合会导致它们相互干扰。为此，我们开发了一种将向量正交化的方法；该方法将这些特质解耦，使模型能够表现出某一种行为而不诱发其他行为，从而实现多种特质的同步导向。

    arXiv:2610.37517v1 Announce Type: cross  Abstract: Activation steering offers an effective and efficient way to modify an LLM's behavior at inference time. However, it remains unclear how to apply steering such that the signal is consistently strong and the steering strength has a universally interpretable meaning. We address this first by identifying the most steerable interval for each steering vector and rescaling the steering range accordingly. Building on this, we introduce a more interpretable method for steering, called parameterized steering which instead of pushing activations in a direction, it sets them at the desired trait strength. Next, with multi-trait steering, we find that naively combining vectors causes them to interfere with one another. In response, we develop a method to orthogonalize the vectors; it decouples these traits, allowing a model to exhibit one behavior without inducing the others, thus enabling multiple traits to be steered simultaneously.
    
[^25]: 面向在线资源分配的防策略性深度强化学习

    Strategyproof Deep Reinforcement Learning for Online Resource Allocation

    [https://arxiv.org/abs/2610.37504](https://arxiv.org/abs/2610.37504)

    本文提出了首个用于在线资源分配的防策略性深度强化学习机制，通过单调神经网络等三个新颖组件，在实现非线性分配规则的同时严格保证参与者的真实申报激励。

    

    在线资源分配面临两个相互耦合的挑战：（1）在未来需求不确定的情况下高效分配稀缺资源；（2）确保参与者如实报告任务估值与需求（即防策略性）。强化学习（RL）能够通过经验自适应地调整分配决策，但不一定能保证防策略性。现有的防策略性强化学习方法采用线性分配规则，这限制了其所能表征的资源交互关系。神经机制设计方法具有更强的表达能力，但只能提供近似意义上的激励保证。在此背景下，我们提出了首个用于在线资源分配的防策略性深度强化学习机制。它集成了三个新颖的组件：（i）一个单调神经网络，通过构造方式学习非线性交互（更高的申报估值不会降低接受概率，而更长的申报时长或更大的资源需求不会提高其接受概率

    arXiv:2610.37504v1 Announce Type: cross  Abstract: Online resource allocation faces two coupled challenges: (1) allocating scarce resources efficiently under uncertain future demand, and (2) ensuring that agents report task valuations and requirements truthfully (strategyproofness). Reinforcement Learning (RL) adapts allocation decisions through experience but does not necessarily ensure strategyproofness. Existing strategyproof RL uses a linear allocation rule, which limits the resource interactions it can represent. Neural mechanism design methods offer greater expressiveness, but provide only approximate incentive guarantees. Against this background, we introduce the first strategyproof deep RL mechanism for online resource allocation. It integrates three novel components: (i) a monotonic neural network that learns nonlinear interactions by construction (higher reported value cannot reduce acceptance, while a longer reported duration or larger resource requirement cannot increase it
    
[^26]: 从交互到能力：以任务推进作为GUI智能体的状态与监督

    From Interaction to Capability: Task Advancement as State and Supervision for GUI Agents

    [https://arxiv.org/abs/2610.37482](https://arxiv.org/abs/2610.37482)

    提出任务推进记忆（TAM），将经验证的任务进展同时作为执行过程中的状态和任务结束后的监督信号，使GUI智能体能够在长时程跨应用工作流中积累可复用操作，并从成功与失败的任务中学习局部有效的经验。

    

    GUI记忆应当保存过去动作的结果并学习可复用的操作。这在长时程工作流中非常困难，因为智能体需要跨应用程序利用信息，且有用的操作往往与失败的尝试交织在一起。我们提出了任务推进记忆（Task Advancement Memory, TAM），它将任务推进定义为经验证的任务进展，既支持正在进行的任务执行，也学习可复用的操作。在执行过程中，任务推进状态刻画了经验证的任务进展如何改变剩余目标，使智能体能够基于跨应用程序的早期成果继续推进。在每个任务完成后，推进监督会识别并抽象出那些产生了经验证局部结果的交互。所学到的操作保留了所需输入、预期结果与验证条件之间的关系。这使得无论任务最终成功还是失败，其中局部成功的操作都能被保留下来。在MobileWorld纯GUI任务上，该（原文此处被截断）

    arXiv:2610.37482v1 Announce Type: new  Abstract: GUI memory should preserve the results of past actions and learn reusable operations. This is difficult in long-horizon workflows, where agents must use information across applications and useful operations are interleaved with failed attempts. We introduce Task Advancement Memory (TAM), which defines task advancement as verified task progress, and supports ongoing task execution and learns reusable operations. During execution, Task Advancement State represents how verified task progress changes the remaining goals, allowing the agent to build on earlier achievements across applications. After each task, advancement supervision identifies and abstracts the interactions that produced a verified local result. The learned operations preserve the relation between required inputs, expected outcomes, and verification conditions. This retains locally successful operations from both successful and failed tasks. On MobileWorld GUI-only tasks, Ad
    
[^27]: 跟随梯度：追溯安全退化至训练数据

    Follow the Gradient: Tracing Safety Degradation Back to Training Data

    [https://arxiv.org/abs/2610.37481](https://arxiv.org/abs/2610.37481)

    本文提出SimpleLoRIF，一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本，从而追溯并过滤导致大语言模型安全退化的训练数据。

    

    在指令数据上微调大语言模型（LLMs）可能会悄然削弱其安全性，即使没有任何训练样本看起来有害。由于这种降低安全性的数据看起来是良性的，标准的内容过滤器难以捕捉到它。我们通过开发SimpleLoRIF来解决这一问题，这是一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本。通过简化低秩归因方法并将反向传播限制在最后四个Transformer块，SimpleLoRIF可以嵌入百万量级样本的数据集，将梯度归因从小规模分析工具提升为适用于完整后训练数据集的实用过滤器。我们首先在涌现性失调现象上验证了该方法，发现梯度可以将不安全代码训练数据与微调后出现的广泛有害行为联系起来，在失调模型的梯度空间中AUC高达0.865。

    arXiv:2610.37481v1 Announce Type: cross  Abstract: Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned mod
    
[^28]: HGPRGDel：基于GPR感知的生长耦合基因敲除设计的学习与排序

    HGPRGDel: GPR-Aware Learning and Ranking for Growth-Coupled Gene Deletion Design

    [https://arxiv.org/abs/2610.37478](https://arxiv.org/abs/2610.37478)

    该论文提出了HGPRGDel，一种GPR感知的可复现协议，通过从已有菌株设计数据中学习并生成基因敲除优先级排序，从而在基于约束的评估之前高效筛选候选干预方案，加速生长耦合微生物菌株的设计。

    

    通过基因干预将细胞生长与目标化合物的合成相关联的生长耦合生产菌株设计，是代谢工程中的一项基本目标。基于约束的方法可以识别生长耦合的基因敲除策略，但针对每个目标产物通常需要在大量可能的基因敲除组合中进行单独搜索。这促使人们寻求一种可复用的方法，即从先前评估过的菌株设计中学习，以便在下游基于约束的评估之前对候选基因干预进行优先级排序。我们提出了一种可复现的生长耦合基因敲除设计协议，该协议能够为每个目标代谢物学习并排序基因敲除优先级。利用计算整理的敲除策略作为学习资源，该协议可生成连续的敲除优先级排序，在可调节的搜索空间预算下构建候选策略，并进行评估。

    arXiv:2610.37478v1 Announce Type: cross  Abstract: Designing microbial strains for growth-coupled production, in which cellular growth is linked to the synthesis of a target compound through genetic interventions, is a fundamental goal in metabolic engineering. Constraint-based methods can identify growth-coupled deletion strategies, but design for each target typically requires a separate search across a large number of possible gene-deletion combinations. This motivates a reusable approach that learns from previously evaluated strain designs to prioritize candidate interventions before downstream constraint-based evaluation. We present a reproducible protocol for growth-coupled gene deletion design that learns and ranks gene-deletion priorities for each target metabolite. Using computationally curated deletion strategies as learning resources, the protocol generates continuous deletion-priority rankings, constructs candidate strategies at adjustable search-space budgets, and evaluate
    
[^29]: 在ADNI静息态fMRI认知预测中，个体化功能网络归因的可靠性如何？

    How Reliable Is Individualized Functional-Network Attribution in Resting-State fMRI Cognitive Prediction in ADNI?

    [https://arxiv.org/abs/2610.37472](https://arxiv.org/abs/2610.37472)

    该研究首次系统量化了ADNI静息态fMRI认知预测中个体化功能网络归因的信度，发现其分半信度接近上限但重测信度中等，且归因结果虽以群体共享结构为主，仍包含适度但非零的个体特异性成分。

    

    个体化归因用于评估哪些功能网络驱动了模型对特定受试者的预测，并常被提议作为一种可解释的静息态fMRI（rs-fMRI）生物标志物，但其重测信度很少被量化。我们评估了针对冻结的静息态功能连接（FC）预测器的逐受试者网络消融归因（Schaefer-200 FC 预测 MMSE/ADAS-Cog13），该预测器在来自阿尔茨海默病神经影像学倡议（ADNI）的 N=875 名受试者上训练，并使用了 受试者级别的置换零假设。分半信度接近上限（Spearman ρ=0.948/0.950），但在扫描-重扫描时有所下降（ρ=0.609/0.616）。在不匹配受试者的零假设下，相关性仍然很大（ρ~0.46-0.47），表明存在强烈的群体共享结构，但匹配对显著超过了这一零假设（z~6.6-6.9，p≤10⁻⁴）。因此，在 ADNI 中，个体化网络归因在群体共享结构之上包含一个适度但非零的个体特异性成分。

    arXiv:2610.37472v1 Announce Type: cross  Abstract: Individualized attribution estimates which functional networks drive a model prediction for a given subject and is often proposed as an interpretable resting-state fMRI (rs-fMRI) biomarker, but its test-retest reliability is rarely quantified. We evaluate per-subject network-ablation attribution for a frozen resting-state functional connectivity (FC) predictor (Schaefer-200 FC predicting MMSE/ADAS-Cog13) trained on N=875 subjects from the Alzheimer's Disease Neuroimaging Initiative (ADNI), using, and (iii) a subject-level permutation null. Split-half reliability is near-ceiling (Spearman p=0.948/0.950) but drops on scan-rescan (p=0.609/0.616). Under the mismatched-subject null, correlations remain substantial (p~0.46-0.47), indicating strong population-shared structure, yet matched pairs exceed this null (z~6.6-6.9, p<=10-4). Thus, in ADNI, individualized network attribution contains a modest but non-zero person-specific component atop
    
[^30]: PerceptRL：让多模态大语言模型中的视觉感知可奖励化——从答案到交互

    PerceptRL: Making Visual Perception Rewardable in MLLMs from Answers to Interactions

    [https://arxiv.org/abs/2610.37471](https://arxiv.org/abs/2610.37471)

    PerceptRL通过多智能体流水线将密集图像描述中经验证的错误转化为带固定答案标签和规则奖励的问答对（链式QA与选择题QA），并针对高错误率领域提供定制化验证，从而在MLLM后训练中使视觉感知变得可奖励化，弥补了单纯推理优化带来的感知短板。

    

    细粒度视觉感知仍然是多模态大语言模型（MLLM）后训练过程中的瓶颈，因为感知错误会限制以推理为中心的优化所带来的收益。尽管图像描述（caption）预训练可以改善感知能力，但随着后续阶段改变数据配比和训练目标，其影响可能会减弱。在后训练阶段持续使用描述监督具有挑战性，因为主观性强、信息密集的描述使可验证奖励的设计变得复杂。我们提出了PerceptRL，它在两个互补的层面上增强感知能力。Perception RL使用多智能体流水线从密集的图像描述中提取经过验证的错误，并将相应的视觉事实转换为具有固定答案标签和基于规则的奖励的问答对。其中，链式问答（Chain QA）用于关联相互关联的错误，而选择题问答则组合不相关的错误。此外，一条专门的路径针对高错误率领域采用定制化的验证，另一条领域无关的……（原文摘要在此处截断）

    arXiv:2610.37471v1 Announce Type: cross  Abstract: Fine-grained visual perception remains a bottleneck during post-training of multimodal large language models (MLLMs), as perceptual errors can constrain gains from reasoning-focused optimization. Although image-caption pre-training improves perception, its influence can weaken as subsequent stages change data mixtures and objectives. Continuing caption supervision during post-training is challenging because subjective, information-dense descriptions complicate verifiable reward design. We present PerceptRL, which strengthens perception at two complementary levels. Perception RL uses a multi-agent pipeline to extract verified errors from dense image captions and convert the corresponding visual facts into QA pairs with fixed answer labels and rule-based rewards. Chain QA links correlated errors, while multiple-choice QA combines unrelated errors. A specialized route targets high-error domains with tailored verification, and a domain-agn
    
[^31]: 基于强化学习的固定翼自主拦截

    Autonomous fixed-wing interception using reinforcement learning

    [https://arxiv.org/abs/2610.37470](https://arxiv.org/abs/2610.37470)

    本文提出一种由强化学习（PPO）训练的神经网络，将滤波后的单目视觉观测直接映射为执行器指令，从而实现固定翼拦截机对中空长航时无人机（如Shahed型）的自主拦截。

    

    中空长航时无人机系统，如Shahed型攻击无人机，是现代冲突中尚未得到有效应对的威胁。传统防空系统成本过高，难以提供可行的大规模解决方案。四旋翼拦截机是一种廉价且经过实战检验的替代方案，但其极其有限的飞行时间使其只能应对短距离、低高度的目标，而固定翼平台凭借其续航能力可以从更远的距离抵达更高高度的目标。然而，此类拦截机目前仍需人工操作。近年来基于学习的控制器的发展显示出实现自动化的潜力。在本工作中，提出了一种由神经网络控制的固定翼拦截机，该神经网络将滤波后的单目视觉观测直接映射为执行器指令。该策略在自定义的Python环境中使用强化学习和近端策略优化（PPO）进行训练，随后被部署于……

    arXiv:2610.37470v1 Announce Type: cross  Abstract: Medium altitude, long endurance unmanned aerial systems such as Shahed-type strike drones are an unanswered threat in modern conflicts. Conventional air defense systems are too expensive to provide a feasible large-scale solution. Quadcopter interceptors are a cheap, battle-proven alternative, but their extremely limited flight time restricts them to low altitude targets at short range, whereas fixed-wing platforms have the endurance to reach higher altitude targets from longer distances. However, such interceptors are still operated manually. Recent developments in learning-based controllers show potential for automation. In this work, a fixed-wing interceptor is presented that is controlled by a neural network that maps filtered monocular vision observations directly to actuator commands. The policy was trained in a custom Python environment using reinforcement learning and proximal policy optimization. The policy was then deployed o
    
[^32]: Concordat：面向大语言模型智能体的类型化会话

    Concordat: Typed Sessions for LLM Agents

    [https://arxiv.org/abs/2610.37461](https://arxiv.org/abs/2610.37461)

    Concordat 是一个 Python 库，它借鉴类型状态与会话类型的思想，将大语言模型智能体的接口建模为有限状态协议并在运行时强制执行，每一轮只向模型提供当前状态下合法的工具，从而用类型化的结构性保障取代仅靠提示词约束智能体行为的脆弱做法。

    

    基于大语言模型构建的智能体越来越多地被允许执行具有实际后果的操作：取消预订、发放退款、更改账户。约束这些操作的规则，例如“在更改任何内容之前必须获得客户的明确确认”，通常只是写在提示词中，其被遵守的可靠程度完全取决于模型自身的选择。我们提出了 Concordat，这是一个 Python 库，它借鉴类型状态与会话类型的思想，将智能体视为一个接口为有限状态协议的对象，并在运行时对模型强制执行该协议。在每一轮交互中，模型只会被提供当前协议状态下合法的工具，超出该集合的调用会在产生副作用之前被拒绝，保留给人类的决策只能被请求（而不能被模型擅自执行），并且令牌与工具调用预算会被计量并在任务交接时线性分摊。协议在定义时即接受检查，智能体的子类……（原文摘要至此中断）

    arXiv:2610.37461v1 Announce Type: new  Abstract: Agents built on large language models are increasingly allowed to take consequential actions: cancelling bookings, issuing refunds, changing accounts. The rules that govern those actions, such as "obtain the customer's explicit confirmation before changing anything", are usually stated in the prompt and followed only as reliably as the model chooses to follow them. We present Concordat, a Python library that treats an agent as an object whose interface is a finite-state protocol, in the spirit of typestate and session types, and enforces that protocol on the model at run time. On every turn the model is offered only the tools that are legal in the current protocol state, calls outside that set are rejected before they have side effects, decisions reserved for humans can only be requested, and token and tool-call budgets are charged and split linearly across handoffs. Protocols are checked when they are defined, and a subclass of an agent
    
[^33]: M4Bench：面向临床电子健康记录推导智能体的程序性规范评估

    M4Bench: Evaluating Procedural Specification for Clinical EHR Derivation Agents

    [https://arxiv.org/abs/2610.37459](https://arxiv.org/abs/2610.37459)

    M4Bench基准通过28个临床EHR推导任务发现，AI编程智能体的性能提升主要来自完整的操作性规范，而非额外的技能包。

    

    基于电子健康记录的临床研究依赖于派生变量，其定义存在于数据库模式之外，体现在项目编码、时间窗口、缺失值规则和输出粒度选择上，而这些选择会改变队列构成和下游研究结论。我们提出M4Bench，这是一个基于MIMIC-IV和eICU的基准，包含涵盖15个任务族的28个临床推导任务，用于测试AI编程智能体能否复现此类推导，以及经临床医生审查的智能体技能如何改变其结果。该基准的对照设计将技能的价值分解为规范完整性、交付格式和模式熟悉度三个部分。在一项Docker隔离的Codex实验（使用GPT 5.4 Mini和GPT 5.5）中，技能相比无技能基线将奖励提升了0.136，28个任务中有26个得到改善。规范完整性贡献了大部分收益：在一个八任务对照子集上，仅操作性规范就能将奖励提升0.162，而技能在此基础上仅额外增加0.024，与零无显著差异。

    arXiv:2610.37459v1 Announce Type: new  Abstract: Clinical research over electronic health records depends on derived variables whose definitions sit outside the schema, in item codes, time windows, missingness rules, and output-grain choices that change cohorts and downstream conclusions. We introduce M4Bench, 28 clinical derivation tasks across 15 task families on MIMIC-IV and eICU, to test whether AI coding agents can reproduce such derivations and how clinician-reviewed agent skills change their results. Its controls decompose the value of skills into specification completeness, delivery format, and schema familiarity. In a Docker-isolated Codex campaign (GPT 5.4 Mini and GPT 5.5), skills raise reward by 0.136 over the no-skill baseline, with 26 of 28 tasks improving. Specification completeness accounts for most of this gain: on an eight-task control subset, an operational spec alone raises reward by 0.162, and skills add only 0.024 on top, indistinguishable from zero. On a ten-task
    
[^34]: 交易智能体集群中的测试时通信

    Test-Time Communication in Trading Agent Swarms

    [https://arxiv.org/abs/2610.37448](https://arxiv.org/abs/2610.37448)

    该论文通过对来自四个模型系列的200个交易智能体在七个交易日内五种通信机制下的大规模实验，揭示了测试时通信会诱发“羊群效应”——通信导致智能体持仓趋同且群体收益主要由共同持仓决定，表明在个体目标独立、市场反馈迟缓且有噪声的交易场景中，智能体间的通信趋同是一种风险而非协作优势。

    

    智能体之间的通信大多是在具有共同目标且反馈快速、明确的任务上进行研究的，而交易两者皆不具备：每个智能体管理自己的账户，其决策由市场迟缓且充满噪声地进行评判。因此，交易智能体群是一群独立的决策者，而非一个团队——对这样的群体而言，趋同是一种风险而非目标。我们在五种通信机制下（从完全隔离到留言板和共享笔记），将来自四个模型系列的200个智能体以十人一组的方式运行了七个美国交易日（2026年10月1日至9日），并记录了每个智能体在每次决策前所阅读的全部信息。在四个模型系列中的两个中，通信组成员持有相同股票的比例远高于隔离组，且一个群体的收益很大程度上取决于其成员共同持仓的收益。看到群组成员发布的帖子会使智能体复用其中八字短语的可能性提高7个百分点。起步过……（摘要在此处截断）

    arXiv:2610.37448v1 Announce Type: cross  Abstract: Communication between agents has mostly been studied on tasks with a shared objective and fast, clear feedback. Trading has neither: each agent manages its own account, with decisions being scored by markets late and noisily. A trading swarm is thus a population of independent decision-makers rather than a team, for which convergence is a risk and not the goal. We ran 200 agents from four model families in groups of ten under five communication regimes, from isolation to message boards and shared notes, over seven US trading days (1 to 9 October 2026), logging everything each agent read before every decision. In two of the four families, communicating groups came to hold far more of the same stocks than the isolated group. A group's return was largely the return on what its members held in common. Seeing a post from a groupmate made an agent 7 percentage points more likely to reuse an eight-word phrase from it. Agents that started too 
    
[^35]: 表征未见之物：重新思考空间推理中的视觉想象

    Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning

    [https://arxiv.org/abs/2610.37437](https://arxiv.org/abs/2610.37437)

    该论文提出基于模拟器的ImagiRep基准，在同一未观测状态下对比RGB与六种替代表示（深度、全景、俯视几何图等），揭示不同空间推理任务受益于不同的想象表示，从而重新审视了空间想象中RGB作为默认表示的合理性。

    

    视觉世界模型通过生成未观测状态的RGB视图来支持空间推理。然而，RGB之所以成为默认接口，很大程度上是因为它适合视觉生成，而不一定是对下游推理最有效的表示形式。不同的空间问题可能受益于对同一想象状态的不同表示：俯视几何图有助于全局距离比较，而全景视图则为开放空间的推理提供更广阔的上下文。为了分离表示形式在空间想象中的作用，我们提出了ImagiRep，这是一个基于模拟器的基准，包含922个问题，涵盖两种设置（视角想象和跨视角整合）下的七种问题类型，并为同一未观测状态提供了真实RGB以及六种替代表示（如深度图、全景视图和俯视几何图）。这一受控比较揭示了……

    arXiv:2610.37437v1 Announce Type: cross  Abstract: Visual world models increasingly support spatial reasoning by generating RGB views of unobserved states. However, RGB has become the default interface largely because it is well suited for visual generation, rather than necessarily being the most effective representation for downstream reasoning. Different spatial questions may benefit from different representations of the same imagined state: top-down geometry can facilitate global distance comparisons, while panoramic views provide broader context for reasoning about open space. To isolate the role of representation in spatial imagination, we introduce ImagiRep, a simulator-grounded benchmark of 922 questions spanning seven question types across two settings, Viewpoint Imagination and Cross-view Integration, providing oracle RGB and six alternative representations (e.g., depth, panoramic views, and top-down geometry) of the same unobserved state. This controlled comparison reveals th
    
[^36]: FeatureArc：在演化特性上对编码智能体进行基准测试

    FeatureArc: Benchmarking Coding Agents on Evolving Features

    [https://arxiv.org/abs/2610.37434](https://arxiv.org/abs/2610.37434)

    提出 FeatureArc 基准测试，首次评估编码智能体在软件特性持续演化场景下的能力，其流水线从仓库历史构建包含阶段间依赖、明确需求契约、可执行 F2P/P2P 测试验证并覆盖四种编程语言的任务。

    

    一个软件特性通常会随时间演化：随着新用例和需求的到来，开发人员会在第一版的基础上进行扩展，而每次扩展都建立在之前编写的代码之上。大多数特性基准测试是在固定代码库上评估单个请求。特性演化则要求在智能体自己先前编写的代码上评估连续的扩展，并检查该特性是否保持正确。我们提出了 FeatureArc，用于评估编码智能体在演化特性上的表现，并配备了一个从仓库历史构建任务的流水线。该基准测试结合了同一特性各连续阶段之间的依赖关系、明确的需求与接口契约，以及覆盖四种编程语言的可执行检查。该流水线在参考代码的每个阶段前后，验证针对新行为的“失败到通过”（F2P）测试和针对既有行为的“通过到通过”（P2P）测试。我们在 73 个经人工审核的任务上评估了 10 个编码智能体。

    arXiv:2610.37434v1 Announce Type: cross  Abstract: A software feature usually evolves over time: developers extend its first version as new use cases and requirements arrive, and each extension builds on code written earlier. Most feature benchmarks evaluate a single request on a fixed codebase. Feature evolution requires evaluating successive extensions on an agent's own earlier code and checking whether the feature remains correct. We introduce FeatureArc to evaluate coding agents on evolving features, together with a pipeline that constructs its tasks from repository histories. The benchmark combines dependencies between successive stages of the same feature, explicit requirements and interface contracts, and executable checks across four programming languages. The pipeline validates fail-to-pass (F2P) tests for new behavior and pass-to-pass (P2P) tests for existing behavior on the reference code before and after each stage. We evaluated 10 coding agents on 73 manually reviewed task
    
[^37]: 基于贝叶斯自适应马尔可夫决策过程的Rollout分配方法以增强GRPO训练

    Bayes-Adaptive MDPs for Rollout Allocation to Enhance GRPO Training

    [https://arxiv.org/abs/2610.37429](https://arxiv.org/abs/2610.37429)

    本文提出AdaRoll-GRPO，将GRPO训练中的rollout预算分配建模为贝叶斯自适应马尔可夫决策过程，实现基于问题难度的自适应rollout分配，在相同总预算下最大化训练效用。

    

    组相对策略优化（GRPO）在大语言模型后训练中展现出了强大的性能。然而，它为每个训练问题分配统一的rollout预算，忽略了问题难度的差异。因此，高难度问题可能获得的rollout次数过少，无法产生有信息量的正向奖励信号，限制了其对训练的贡献；而中等难度问题往往获得的rollout次数多于实际所需。在本工作中，我们将难度感知的rollout预算分配问题进行形式化，目标是在与标准GRPO相同的总rollout预算下最大化训练效用。我们提出AdaRoll-GRPO，该方法将rollout分配重新表述为一个序贯决策问题，并利用贝叶斯自适应马尔可夫决策过程（MDP）来学习自适应的分配策略。学习到的策略通过离线方式计算得出，并在GRPO训练过程中直接应用，无需引入额外的在线优化。

    arXiv:2610.37429v1 Announce Type: cross  Abstract: Group Relative Policy Optimization (GRPO) has demonstrated strong performance in LLM post-training. However, it allocates a uniform rollout budget to every training question, ignoring variations in question difficulty. As a result, highly difficult questions may receive too few rollouts to generate informative positive reward signals, limiting their contribution to training, while moderately difficult questions often receive more rollouts than necessary. In this work, we formulate the problem of difficulty-aware rollout budget allocation to maximize training utility under the same total rollout budget as standard GRPO. We propose AdaRoll-GRPO, which reformulates rollout allocation as a sequential decision-making problem and leverages a Bayes-adaptive Markov decision process (MDP) to learn an adaptive allocation policy. The learned policy is computed offline and applied during GRPO training without introducing additional online optimiza
    
[^38]: 超越读出层：跨模态与跨运算共享的潜在数值框架

    Beyond the Readout: A Latent Numerical Frame Shared Across Modality and Operation

    [https://arxiv.org/abs/2610.37422](https://arxiv.org/abs/2610.37422)

    通过因果干预发现，多模态模型在文本与图像、加法与减法之间共享一个统一的潜在数值框架，但各条件沿不同方向编码数字，只需一次旋转和缩放即可实现子空间间的映射并恢复大部分因果效应。

    

    当一个多模态模型从文本计算17+35，或从渲染图像计算91-39时，它都会得出相同的答案52。它是为两条路径使用了同一个内部数字概念，还是为每条路径使用了各自独立的机制？诸如RSA或CKA之类的相似性度量无法回答这个问题：它们对完全相同的方向和旋转后的方向给出相同的分数，而且两个表示可能仅仅因为都准备了相同的输出数字才看起来相似。因此，我们采用因果干预的方法。对于四种条件（文本或图像、加法或减法），我们在Gemma4-12B-it的残差流中学习一个小子空间，在该子空间中交换坐标会导致模型答案的交换。这些子空间回答了三个问题。首先，各条件是否沿相同的方向写入数字？答案是否定的。在同一运算中，将文本子空间复用于图像任务只能恢复53.2%的因果效应。而通过单次旋转和缩放即可将一个子空间映射到另一个子空间，并显著提高恢复率（原文在此截断）。

    arXiv:2610.37422v1 Announce Type: new  Abstract: When a multimodal model computes 17+35 from text or 91-39 from a rendered image, it produces the same answer, 52. Does it use one internal concept of number for both, or separate machinery for each path? Similarity metrics such as RSA or CKA cannot settle this question. They give the same score to identical and to rotated directions, and two representations can look alike only because both prepare the same output digits. We therefore use causal interventions. For four conditions (text or image, addition or subtraction), we learn a small subspace of Gemma4-12B-it's residual stream in which swapping coordinates swaps the model's answer. The subspaces answer three questions. First, do the conditions write numbers along the same directions? They do not. Reusing a subspace from text in an image on the same operation recovers only 53.2% of its causal effect. A single rotation and rescaling maps one subspace onto the other and raises recovery t
    
[^39]: 约束提交边界：一种基于DIO/ODI的智能体实体消解授权护栏

    Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution

    [https://arxiv.org/abs/2610.37407](https://arxiv.org/abs/2610.37407)

    本文提出一种基于DIO/ODI的授权护栏，将智能体推理可用的证据与可授权持久性身份变更的证据分离，并通过两阶段监督器仅在授权合格证据充分支持所有必需命题、冲突已解决且授权范围与策略版本有效时才允许提交身份变更，从而防止错误身份假设在智能体系统中自我确认。

    

    智能体实体消解（ER）系统会迭代地规划动作、检索证据、修正假设并提出图变更操作。一个错误的身份假设可能会影响后续检索，并产生表面上的确认。本文提出了一种受数字身份优化（DIO）和数字身份本体（ODI）启发的授权边界，用以区分可用于智能体推理的证据与有资格授权持久性身份变更操作的证据。一个由领域治理的约束绑定函数将选定的重构失败区别映射为显式的身份保护约束。一个抽象的两阶段监督器只有在以下条件全部满足时才允许提交：针对特定声明的授权合格证据支持每一项必需的命题、相关冲突已得到解决、操作保持在其授权范围内，并且相关的状态与策略版本仍然有效。被接纳为UNVERIFIED（未验证）状态并不赋予……（摘要原文在此处截断）

    arXiv:2610.37407v1 Announce Type: new  Abstract: Agentic Entity Resolution (ER) systems iteratively plan actions, retrieve evidence, revise hypotheses, and propose graph mutations. An erroneous identity hypothesis may condition later retrieval and produce apparent confirmation. This paper proposes a Digital Identity Optimization (DIO) and Ontology of Digital Identity (ODI)-informed authorization boundary between evidence available for agent reasoning and evidence eligible to authorize persistent identity-changing mutations.   A domain-governed constraint-binding function maps selected reconstruction-failure distinctions to explicit identity-preserving constraints. An abstract two-phase supervisor permits a commit only when claim-specific authorization-eligible evidence supports every required proposition, relevant conflicts are resolved, the operation remains within its authorized scope, and relevant state and policy versions remain valid. Admission as UNVERIFIED does not confer author
    
[^40]: 拨开迷雾（FoG）：训练知识图谱智能体检索其答案背后的证据

    Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers

    [https://arxiv.org/abs/2610.37405](https://arxiv.org/abs/2610.37405)

    该论文提出FoG框架，通过基于证据的数据合成、KG交互智能体和证据覆盖过程奖励（ECPR），训练知识图谱智能体不仅回答正确，还必须检索出支撑答案的证据三元组，从而使答案可追溯、可审计。

    

    大语言模型（LLM）能够正确回答许多事实性问题，但正确的答案未必是有依据支撑的答案。对于LLM智能体而言，知识图谱（KG）扮演着两种角色：作为答案来源，它提供模型所缺失的事实；作为证据来源，它提供使答案可追溯、可审计的支持性三元组。然而，现有的KG智能体仅将知识图谱视为答案来源：其训练数据包含答案但不包含证据，其训练奖励只检查答案而不检查检索到的证据，因此智能体经常在未检索支持三元组的情况下就给出正确答案。我们提出FoG，一个训练KG智能体检索其答案背后证据的框架。FoG由基于证据的合成、KG交互智能体和证据覆盖过程奖励（ECPR）组成：合成过程递归地将显式实体隐藏在可验证的子问题之后，使得每个问题……

    arXiv:2610.37405v1 Announce Type: new  Abstract: Large language models (LLMs) can answer many factual questions correctly, but a correct answer is not necessarily a supported one. For an LLM agent, a knowledge graph (KG) plays two roles: as a source of answers, it supplies facts that the model lacks; as a source of evidence, it supplies the supporting triples that make an answer traceable and auditable. However, existing KG agents treat the KG only as a source of answers: their training data carries answers but not evidence, and their training rewards check the answer but not the retrieved evidence, so agents often answer correctly without retrieving the supporting triples. We propose FoG, a framework that trains KG agents to retrieve the evidence behind their answers. FoG consists of evidence-grounded synthesis, a KG interaction agent, and an evidence-coverage process reward (ECPR): the synthesis recursively hides explicit entities behind verifiable sub-questions, so that every questi
    
[^41]: 基于迭代事后重标注学习质量可控的机器人策略

    Learning Quality-Controllable Robot Policies with Iterative Hindsight Relabeling

    [https://arxiv.org/abs/2610.37393](https://arxiv.org/abs/2610.37393)

    提出质量可控策略（QCP），使机器人策略能够在推理时动态控制执行速度、运动平滑度等质量属性及其组合，并通过迭代事后重标注循环将可控质量范围扩展到初始演示数据之外。

    

    机器人策略评估通常依赖于任务完成的二元成功信号。然而，在实际部署中，期望的执行行为还由执行速度和运动平滑度等质量属性来定义，这些属性需要根据任务要求进行动态控制。为了解决这一问题，我们提出了质量可控策略，这是一种能够在推理时对单个质量属性及其组合进行控制的机器人策略。为了将可控质量范围扩展到初始演示数据之外，我们进一步提出了一种迭代的事后重标注循环：QCP在目标质量标签下收集在线轨迹，将成功的轨迹添加到数据集中，根据测量到的质量对累积的轨迹进行事后重标注，并对策略进行微调。在 Robomimic 和 RoboEval 上的实验表明，QCP 能够持续提升任务执行质量和……

    arXiv:2610.37393v1 Announce Type: cross  Abstract: Robot policy evaluation often relies on a binary success signal for task completion. In real-world deployment, however, desired execution behaviors are also defined by quality attributes such as execution speed and motion smoothness, which need to be dynamically controlled according to task requirements. To address this, we introduce Quality-Controllable Policy (QCP), a robot policy that enables inference-time control over both individual quality attributes and their combinations. To extend the controllable quality range beyond initial demonstrations, we further propose an iterative hindsight relabeling loop: QCP collects online rollouts under target quality labels, adds successful trajectories to the dataset, relabels the accumulated trajectories in hindsight according to their measured quality, and fine-tunes the policy. Experiments on Robomimic and RoboEval demonstrate that QCP consistently improves both task execution quality and s
    
[^42]: 受JEPA启发的潜在场预测用于稀疏导频MIMO-OFDM信道重构

    JEPA-Inspired Latent Field Prediction for Sparse-Pilot MIMO-OFDM Channel Reconstruction

    [https://arxiv.org/abs/2610.37391](https://arxiv.org/abs/2610.37391)

    提出一种受JEPA启发的无线传播场潜在世界模型，利用无线信道在潜在场域中更强的可预测相关性，从稀疏导频中稳健地重构MIMO-OFDM信道状态信息，克服了直接在原始CSI系数上预测的脆弱性。

    

    对于多输入多输出正交频分复用（MIMO-OFDM）无线系统而言，准确获取完整的信道状态信息（CSI）至关重要，而基于密集导频的估计会带来巨大的开销。在现实的传播环境中，直接在原始CSI系数上最小化预测误差是脆弱的，因为小尺度衰落、移动性、局部散射以及微小的空间扰动都可能引起对相位敏感的大幅变化。尽管如此，无线信道包含强烈的空间、时间和频域相关性，这些相关性在潜在传播场域中可以得到更可预测的表示。受这一观察启发，我们提出了一种受联合嵌入预测架构（JEPA）启发的场世界模型，将其定义为无线传播场的潜在预测模型，用于稀疏导频信道重构。该模型将CSI观测映射到潜在……（摘要原文在此处截断）

    arXiv:2610.37391v1 Announce Type: cross  Abstract: Accurate acquisition of full channel state information (CSI) is essential for multiple-input multiple-output orthogonal frequency-division multiplexing (MIMO-OFDM) wireless systems, while dense pilot-based estimation incurs substantial overhead. Directly minimizing the prediction error over raw CSI coefficients is fragile in realistic propagation environments, since small-scale fading, mobility, local scattering, and small spatial perturbations can induce large phase-sensitive variations. Nevertheless, wireless channels contain strong spatial, temporal, and frequency-domain correlations that can be more predictably represented in a latent propagation-field domain. Motivated by this observation, we propose a joint-embedding predictive architecture (JEPA)-inspired field world model, defined as a latent predictive model of the wireless propagation field, for sparse-pilot channel reconstruction. The model maps CSI observations into latent 
    
[^43]: 混合专家模型解码中的通道选择无需校准数据

    Channel Selection for Mixture-of-Experts Decoding Needs No Calibration Data

    [https://arxiv.org/abs/2610.37386](https://arxiv.org/abs/2610.37386)

    混合专家模型解码中的通道选择可以用无需校准数据的top-k方法替代基于离线数据校准阈值的传统方法，在匹配或超越阈值方法性能的同时使实际读取预算更加稳定可控。

    

    混合专家模型在小批量交互式解码过程中，因需要从内存加载路由专家权重而受到瓶颈限制。由于路由专家中的大多数中间通道对任何单个token的贡献都很小，通道选择器可以预测哪些通道是重要的，并只读取它们的权重，这通常是通过在离线数据上校准的阈值来保留通道实现的。我们证明通道选择其实不需要校准数据。首先，校准后的阈值并不能固定选择器实际读取的数量，因为通过阈值的通道数量会因输入不同而变化；为18%读取预算校准的Prox阈值，在我们的基准测试中实际平均只读取了12.9%。其次，top-k方法保留每个token得分最高的k个通道，无需任何校准数据。在校准预算下，在13%和18%两种预算的43次比较中，选择器的top-k版本在相同得分上有40次匹配或超过其阈值版本...

    arXiv:2610.37386v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) models are bottlenecked by loading routed-expert weights from memory during interactive decoding at small batch sizes. Because most intermediate channels in a routed expert contribute little to any single token, channel selectors predict which channels matter and read only their weights, typically keeping channels above a threshold calibrated on offline data. We show that channel selection needs no calibration data. First, a calibrated threshold does not fix how much a selector reads, because the number of channels passing it varies across inputs; Prox with thresholds calibrated for an 18% read budget actually reads only 12.9% on average across our benchmarks. Second, top-k keeps each token's k highest-scoring channels and needs no calibration data. At the calibrated budget, a selector's top-k version matches or beats its threshold version on the same score in 40 of 43 comparisons at 13% and 18% budgets, includ
    
[^44]: 智能体条件化路由：单个冻结模型内实现多样化推理路径

    Agent-Conditioned Routing: Diverse Reasoning Paths Within One Frozen Model

    [https://arxiv.org/abs/2610.37375](https://arxiv.org/abs/2610.37375)

    提出智能体条件化路由（ACR），通过为冻结MoE模型的并行轨迹分配持久的专家资格掩码，在无需微调、角色指定或模型副本的情况下，使单个模型产生真正多样化的推理路径，其方法多样性约为温度采样最强设置的两倍。

    

    生成多个解决方案可以揭示不同的方法，但额外的轨迹往往只是对某一种主导方法的同义改写。我们研究一个冻结的稀疏混合专家模型的条件化能力能否在单一问题提示下提供这种多样性，而无需用户指定角色、预设方法、微调或模型副本。智能体条件化路由（ACR）为m条并行轨迹分配持久的、未经训练的共享/私有专家资格掩码，同时保留已学习的、依赖输入的token路由器。在一个经过筛选的126题多模式数学和物理基准上，由模型裁判完整阅读报告的推导过程，我们统计在四条轨迹的软正确解中产生至少两种不同主要方法的问题数量。ACR在GLM 5.3下达到的问题数量约为最强温度设置的两倍（37.3对18.7），并达到……的1.67倍

    arXiv:2610.37375v1 Announce Type: new  Abstract: Generating several solutions can expose alternative methods, but additional trajectories often paraphrase one dominant approach. We investigate whether the conditional capacity of a frozen sparse mixture-of-experts (MoE) model can provide such diversity from one problem prompt, without user-assigned roles, prescribed methods, fine-tuning, or model copies. Agent-Conditioned Routing (ACR) assigns $m$ parallel trajectories persistent, untrained shared/private expert-eligibility masks while preserving the learned, input-dependent token router. On a screened 126-question multi-mode mathematics and physics benchmark, with model judges reading the reported derivations in full, we count the number of problems yielding at least two distinct primary methods among soft-correct solutions in four trajectories. ACR reaches approximately twice as many such problems as the strongest temperature setting under GLM 5.3 (37.3 versus 18.7), and 1.67 times as
    
[^45]: 揭示大语言模型为何失败：从内部机制到对抗性推理轨迹

    Uncovering Why LLMs Fail: From Internal Mechanisms to Adversarial Reasoning Trajectories

    [https://arxiv.org/abs/2610.37369](https://arxiv.org/abs/2610.37369)

    提出白盒诊断框架CauTrace，通过七种功能状态、表征传播、时间-状态协调与推理轨迹等内部机制分析，统一刻画大语言模型普通推理失败与对抗性诱导失败的根本原因。

    

    大语言模型（LLMs）通常通过最终输出的正确性或安全性来进行评估，然而终点的标签无法揭示失败是如何在整个推理过程中逐步形成的。因此，我们提出这样一个问题：普通的推理失败与对抗性诱导的失败是否能够用相同的内部坐标来刻画。为此，本文研究了一个白盒诊断框架CauTrace，从三项基本检查入手揭示LLM失败的根本原因：功能状态、表征传播、时间-状态协调与推理轨迹。首先，创新性地定义了七种功能状态，用以表示LLM中的支持与约束处理过程。其次，表征变化的传播度量了七种状态之间的变化如何在后续的Transformer块中持续存在。第三，时间-状态协调与轨迹拓扑刻画了承诺形成的时机与推理路径的形态……（摘要原文在此处截断）

    arXiv:2610.37369v1 Announce Type: new  Abstract: Large language models (LLMs) are usually evaluated by final-output correctness or safety, yet an endpoint label does not reveal how a failure develops throughout the reasoning process. We therefore ask whether ordinary reasoning failures and adversarially induced failures can be characterized using the same internal coordinates. Towards this end, we in this paper study a white-box diagnostic framework CauTrace to disclose the root-cause of why LLM fails from three fundamental inspections: Functional States, Propagation of Representation, Temporal-State Coordination and Reasoning Trajectory. First, seven functional states are creatively defined to represent the support and constraint processing in LLMs. Second, the propagation of representation change measures how the changes among the seven states persist across subsequent Transformer blocks. Third, temporal-state coordination and trajectory topology characterize when the commitment and 
    
[^46]: 对齐制度性权力与操作性能力

    Aligning Institutional Power and Operational Ability

    [https://arxiv.org/abs/2610.37342](https://arxiv.org/abs/2610.37342)

    该论文提出了一个以面向过程建模和制度性权力为核心的形式化框架，用于分析和修复多智能体AI系统中责任与履行能力之间的规范性错配问题。

    

    为了实现社会可持续性，智能体AI系统需要规范性的设计，使责任与其履行该责任的能力相匹配。识别诸如不可满足的期望或权限提升等不一致现象，对于避免规范性陷阱至关重要。在本研究中，我们基于这一规范-能力对齐要求，讨论了适合描述多智能体系统的设计维度。我们提出了一个以面向过程的建模和制度性权力为核心的形式化框架，用于分析给定的规范相关过程，讨论了修复该过程的策略，并详细阐述了操作性过程与制度状态之间的对称关系如何影响规范地位的生命周期。

    arXiv:2610.37342v1 Announce Type: new  Abstract: To be socially sustainable, Agentic AI systems require normative designs that align responsibilities with the ability to fulfil them. Identifying incongruences such as unsatisfiable expectations or privilege escalation is crucial in avoiding normative pitfalls. In this study, we discuss design dimensions suitable for describing multi-agent systems in light of this norm-ability alignment requirement. We propose a formal framework centred on process-oriented modelling and institutional power for the analysis of a given normatively relevant process, discuss strategies for repairing it, and elaborate on how the symmetrical relation between operational process and institutional state affects the lifecycle of normative positions.
    
[^47]: CoS-VQ：面向大语言模型的协同共享尺度向量量化

    CoS-VQ: Coordinated Shared-Scale Vector Quantization for Large Language Models

    [https://arxiv.org/abs/2610.37336](https://arxiv.org/abs/2610.37336)

    CoS-VQ在保留低维码字的前提下联合优化硬索引与跨位置共享尺度，通过多尺度候选收集与切线可分离上界克服尺度耦合难题，实现大语言模型的高精度低比特向量量化。

    

    大语言模型（LLM）需要大量内存，这促使人们采用低比特权重量化来实现高效部署。现有的向量量化（VQ）方法通常通过增加有效编码维度来提升精度，但这会使独立查找和局部码字更新变得复杂。我们提出CoS-VQ（协同共享尺度向量量化），它在保留低维码字的同时，联合优化硬索引及其在各位置间的共享尺度。关键挑战在于尺度引起的耦合：最优尺度取决于整个码字组合，而反过来又会改变各位置的候选排序，因此即使重新优化尺度，逐点更新也可能陷入停滞。CoS-VQ在多个探测尺度下收集候选，并为残差感知的二次目标构造了切线可分离上界。通过将候选比较简化为正尺度上线段的下包络问题，它……（摘要截断）

    arXiv:2610.37336v1 Announce Type: cross  Abstract: Large language models (LLMs) require substantial memory, motivating low-bit weight quantization for efficient deployment. Existing vector quantization (VQ) methods often improve accuracy by increasing the effective coding dimension, which can complicate independent lookup and local codeword updates. We propose CoS-VQ (Coordinated Shared-Scale Vector Quantization), which preserves low-dimensional codewords while jointly optimizing hard indices and their shared scale across positions. The key challenge is scale-induced coupling: the optimal scale depends on the entire codeword combination and, in turn, changes candidate rankings across positions, so pointwise updates can stall even after scale re-optimization. CoS-VQ collects candidates at multiple probe scales and constructs a tangent separable upper bound for the residual-aware quadratic objective. By reducing candidate comparisons to lower envelopes of lines over positive scales, it i
    
[^48]: WCC：一种用于决定何时在大规模优化中应用协同进化的基于学习的框架

    WCC: A Learning-Based Framework for Deciding When to Apply Cooperative Coevolution in Large-Scale Optimization

    [https://arxiv.org/abs/2610.37329](https://arxiv.org/abs/2610.37329)

    WCC是一个基于学习的框架，通过将算法选择决策建模为马尔可夫决策过程，在大规模优化中动态决定何时使用协同进化或非分解算法，实现细粒度的动态切换与计算预算的自适应分配。

    

    协同进化与非分解算法是大面积全局优化的两种主要方法。尽管两者都取得了成功，但它们在不同问题类型上的表现差异显著，且各自面临方法特有的局限性，这使得“何时使用协同进化”这一问题仍未得到解决。我们提出了WCC——一种新颖的基于学习的框架，用于动态决定何时在大规模全局优化中部署协同进化。WCC采用精简的设计以实现高效的自适应学习。我们将该决策问题建模为马尔可夫决策过程，设计了有针对性的特征集、稳健的奖励函数以及定制的网络架构，为智能体提供关于问题状态和优化进度的丰富信息。我们的自回归策略使WCC能够在协同进化与非分解算法之间进行细粒度的动态切换，同时无需专家指定即可有条件地分配计算预算。

    arXiv:2610.37329v1 Announce Type: cross  Abstract: Cooperative Coevolution (CC) and Non-Decomposition Algorithms (NDAs) are two primary approaches for large-scale global optimization (LSGO). Although both have achieved success, they exhibit markedly different performance across problem types and each faces approach-specific limitations, leaving the question of "when to use CC" unresolved. We address this gap with WCC, a novel learning-based framework that dynamically decides when to deploy CC in LSGO. WCC employs a streamlined design for efficient adaptive learning. We formulate the decision as a Markov decision process, designing a targeted feature set, a robust reward, and a tailored network architecture to provide the agent with rich information about the problem state and optimization progress. Our auto-regressive policy enables WCC to perform fine-grained, dynamic switching between CC and NDAs while conditionally allocating the computational budget without expert specifications. W
    
[^49]: AnalogScout：基于大语言模型智能体的仿真高效、PVT感知的模拟电路尺寸设计

    AnalogScout: Simulation-Efficient PVT-Aware Analog Sizing with LLM Agents

    [https://arxiv.org/abs/2610.37319](https://arxiv.org/abs/2610.37319)

    AnalogScout是一个由LLM智能体驱动的框架，通过结合精英设计复用、解析式gm/Id局部尺寸设计与全局细化，在有限仿真预算下大幅减少多角点SPICE仿真次数，实现仿真高效的PVT感知模拟电路尺寸设计。

    

    模拟器件尺寸设计必须在所有工艺、电压和温度（PVT）角点上满足全部规格，以确保流片成功，但在所有角点上对每个候选设计进行仿真占据了尺寸设计的大部分时间，减缓了设计迭代。传统的基于代理模型的优化器直接搜索尺寸设计空间，而近期的大语言模型（LLM）智能体虽然为设计引入了电路知识，但仍将尺寸设计任务委托给全局优化器。两者都依赖反复的多角点SPICE仿真来学习电路响应并在此基础上进行优化，这在仿真预算有限的情况下代价高昂。若能为智能体提供可复用的已验证设计以及解析式的子模块尺寸设计，则可节省大部分仿真预算。我们提出了AnalogScout，一个由LLM智能体驱动的仿真高效、PVT感知的尺寸设计框架。通过结合精英设计复用、解析式gm/Id局部尺寸设计以及在全角点验证下的全局细化，AnalogScout减少了……

    arXiv:2610.37319v1 Announce Type: cross  Abstract: Analog device sizing must satisfy all specifications across process, voltage, and temperature (PVT) corners to ensure tape-out success, but simulating every candidate at all corners dominates the sizing time and slows design iteration. Conventional surrogate-based optimizers search the sizing space directly, while recent large language model (LLM) agents add circuit knowledge to designs but still delegate the sizes to global optimizers. Both learn the circuit response from repeated multi-corner SPICE simulations and optimize on top of it, which becomes expensive under a limited simulation budget. Providing an agent with reusable verified designs and sizing sub-blocks analytically could save most of this budget. We propose AnalogScout, a simulation-efficient PVT-aware sizing framework driven by an LLM agent. Combining elite-design reuse, analytic gm/Id local sizing and global refinement under full-corner verification, AnalogScout reduce
    
[^50]: ReForest：孤立森林中可解释性与推理能力的融合

    ReForest: Interpretability Meets Reasoning in Isolation Forest

    [https://arxiv.org/abs/2610.37316](https://arxiv.org/abs/2610.37316)

    ReForest框架将孤立森林的决策路径转化为自然语言解释，借助大语言模型的推理能力和不确定性感知的动态集成机制，弥合统计异常检测与语义理解之间的鸿沟，实现高效且具备上下文感知能力的异常检测。

    

    像孤立森林这样的统计异常检测方法虽然高效，但存在一个关键的语义鸿沟。它们能够识别数值上的稀有性，却无法理解数据的上下文含义，因而常常遗漏细微的、依赖于上下文的异常。相反，大语言模型虽然具备强大的语义推理能力，但在大规模部署时成本过高且速度缓慢。为了弥合这一差距，我们提出了ReForest，这是一个将孤立森林的统计效率与大语言模型推理能力相协同的新颖框架。ReForest将孤立森林的潜在决策路径转化为自然语言解释，使大语言模型能够通过不确定性感知的动态集成来优化异常检测。这种方法使模型能够基于不确定性驱动的实例级置信度，自适应地权衡统计信号和语义信号。在四个真实世界数据集上的大量实验表明……

    arXiv:2610.37316v1 Announce Type: new  Abstract: Statistical anomaly detection methods like Isolation Forest are efficient but suffer from a critical semantic gap. They identify numerical rarities but fail to understand the contextual meaning of data, often missing subtle, context-dependent anomalies. Conversely, Large Language Models possess strong semantic reasoning capabilities but are prohibitively expensive and slow for large-scale deployment. To bridge this gap, we propose ReForest, a novel framework that synergizes the statistical efficiency of Isolation Forest with the reasoning power of LLMs. ReForest transforms the latent decision paths of Isolation Forest into natural-language explanations, enabling an LLM to refine anomaly detection through an uncertainty-aware dynamic ensemble. This approach allows the model to adaptively weigh statistical and semantic signals based on uncertainty-driven instance-level confidence. Extensive experiments on four real-world datasets (Healthca
    
[^51]: 一种基于轻量级注意力机制、具有显著性图可解释性的皮肤癌检测卷积神经网络

    A Lightweight Attention-Based CNN with Saliency-Map Interpretability for Skin Cancer Detection

    [https://arxiv.org/abs/2610.37314](https://arxiv.org/abs/2610.37314)

    该论文提出了一种带有显著性图可解释性的轻量级注意力CNN用于皮肤癌检测，仅用约一半的参数量和存储空间就达到了与MobileNet相当的87%准确率。

    

    皮肤癌是全球最常见且最致命的癌症之一，早期准确诊断可显著提高患者的生存率。然而，由于皮肤病变在视觉上的相似性，人工皮肤镜评估仍然具有挑战性。本研究提出了一种由可解释人工智能驱动的框架，利用轻量级的基于注意力的卷积神经网络对皮肤病变进行自动二分类。该模型集成了通道注意力机制，以聚焦于皮肤镜图像中信息丰富的区域，同时保持较低的计算复杂度。研究使用了一个包含2637张训练图像和660张测试图像的公开数据集，并采用平衡的类别权重。所提出的模型仅用195万个参数和7.45 MB的存储空间就达到了87%的准确率，与最佳基线模型相当，而其参数量和存储空间仅约为后者的一半。ROC AUC和PR AUC分数达到了0.9475。

    arXiv:2610.37314v1 Announce Type: cross  Abstract: Skin cancer is among the most prevalent and life threatening forms of cancer worldwide. Early and accurate diagnosis significantly improves patient survival rates. However, manual dermoscopic assessment remains challenging due to visual similarity. This study proposes an Explainable AI-driven framework for automated binary classification of skin lesions using a lightweight attention-based convolutional neural network (CNN-Attention). The model integrates channel attention to focus on informative regions of dermoscopic images while maintaining low computational complexity. A publicly available dataset comprising 2,637 training and 660 test images was used, with balanced class weights. The proposed model achieved 87 percent accuracy with 1.95 million parameters and 7.45 MB of storage, matching the best baseline (MobileNet) while using approximately half as many parameters and half as much storage. ROC AUC and PR AUC scores reached 0.9475
    
[^52]: RPSD：基于最优传输的检索特权在线自蒸馏

    RPSD: Retrieval-Privileged On-Policy Self-Distillation with Optimal Transport

    [https://arxiv.org/abs/2610.37281](https://arxiv.org/abs/2610.37281)

    该论文提出RPSD方法，通过拥有特权知识的教师模型进行在线自蒸馏以实现词元级信用分配，并利用检索相似用户的行为模式提供协同监督，从而改进基于大语言模型的生成式序列推荐。

    

    序列推荐旨在根据用户的历史行为预测其下一次交互。近期基于大语言模型（LLM）的方法将这一任务重新构建为生成式推荐，并越来越多地采用GRPO等强化学习方法进行后训练。然而，这些方法存在两个局限性：一是“信用盲”优化，即序列级的奖励被所有生成的词元共同分享；二是“协同盲”监督，即每个训练信号仅来自单个用户，忽略了相似用户之间共享的行为模式。为了解决这两个问题，我们提出了检索特权在线自蒸馏（RPSD）。针对信用盲优化问题，我们引入在线自蒸馏（OPSD），由一个拥有特权知识的教师模型重新评估学生的在线策略响应，并利用学生与教师之间的词元级置信度差异，将共享的序列级优势重塑为词元特定的优势信号……

    arXiv:2610.37281v1 Announce Type: new  Abstract: Sequential recommendation predicts a user's next interaction from historical behavior. Recent LLM-based methods recast this task as generative recommendation and increasingly adopt reinforcement learning such as GRPO for post-training. However, these methods suffer from two limitations, namely credit-blind optimization, where a sequence-level reward is shared across all generated tokens, and collaborative-blind supervision, where each training signal is derived from a single user and ignores behavioral patterns shared across similar users. To address both issues, we propose Retrieval-Privileged on-policy Self-Distillation (RPSD). For credit-blind optimization, we introduce on-policy self-distillation (OPSD), in which a teacher with privileged knowledge re-evaluates the student's on-policy response, and the token-level confidence difference between the student and teacher reshapes the shared sequence-level advantage into token-specific cr
    
[^53]: 在LLM驱动的智能体脚手架演化中，评估预算应该花在哪里？

    Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?

    [https://arxiv.org/abs/2610.37276](https://arxiv.org/abs/2610.37276)

    该论文提出了一种评估预算分配框架，证明稀疏反馈足以维持演化搜索，而将额外预算用于针对性的最终确认环节，能更有效地选出最优的智能体脚手架。

    

    大语言模型（LLM）驱动的智能体脚手架演化通过迭代式的候选生成与评估来改进智能体工作流程。评估既指导后续搜索，也决定最终的脚手架，但在完整任务集上评估每个候选会产生巨大的执行成本。我们研究了评估预算应如何在这两个角色之间进行分配。我们的实验表明，稀疏反馈能够维持富有成效的演化，而额外的针对性评估则可改善最终选择。重放分析进一步表明，共享的信息性任务项和共同的评分参考支持在表现相近的候选之间进行比较。基于这些发现，我们提出了\method，一个将稀疏的搜索时反馈与针对性的最终确认相结合的评估分配框架。一段简短的、经过完整评估的预热阶段建立了共享的信息核心，并校准了一个响应模型，其项目参数（摘要在此处截断）

    arXiv:2610.37276v1 Announce Type: cross  Abstract: Large language model (LLM)-driven agent scaffold evolution improves agent workflows through iterative candidate generation and evaluation. Evaluation both guides subsequent search and determines the final scaffold, but evaluating every candidate on the full task set incurs substantial execution costs. We investigate how evaluation budgets can be allocated across these two roles. Our experiments show that sparse feedback can sustain productive evolution, while additional targeted evaluation improves final selection. Replay analyses further show that shared informative items and a common scoring reference support comparisons among closely matched candidates. Motivated by these findings, we propose \method, an evaluation allocation framework that combines sparse search-time feedback with targeted final confirmation. A short fully evaluated warm-up establishes a shared informative core and calibrates a response model whose item parameters 
    
[^54]: 面向非洲语言安全的大语言模型对齐：以豪萨语为例的案例研究

    Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa

    [https://arxiv.org/abs/2610.37245](https://arxiv.org/abs/2610.37245)

    本文构建了首批豪萨语安全偏好数据集HausaSafeLM（含3,000条金标和7,000条银标样本），并通过SFT与DPO对比实验揭示：仅凭拒绝率评估低资源语言模型的安全性会产生误导，基座模型的高拒绝率实为豪萨语生成能力不足而非真正的安全对齐。

    

    大语言模型（LLM）的安全对齐研究仍然集中于高资源语言，而现有对齐方法在低资源、文化背景迥异的环境中的表现仍是悬而未决的问题。我们针对豪萨语（Hausa）研究这一问题，豪萨语是西非超过两亿人使用的语言。我们提出了HausaSafeLM，这是首批豪萨语安全偏好数据集之一，基于源自PKU-SafeRLHF的10,000条提示构建而成：其中3,000条为人工校正（金标）样本，7,000条为机器翻译（银标）样本，涵盖六大类有害内容宏观类别。我们使用Qwen2.5-7B-Instruct模型，在两种数据条件下比较监督微调（SFT）与SFT后接直接偏好优化（DPO）的效果，并在500条有害和200条良性豪萨语提示上进行评估。我们发现，仅凭拒绝率指标来看，未经调优的基座模型似乎非常安全，但这反映的是其豪萨语生成能力的局限，而非真正的安全对齐：其拒绝行为几乎……（原文摘要在此处截断）

    arXiv:2610.37245v1 Announce Type: cross  Abstract: Safety alignment research for large language models (LLMs) remains concentrated on high-resource languages, leaving open how existing alignment methods perform in low-resource, culturally distinct settings. We study this question for Hausa, a language spoken by more than 200 million people across West Africa. We introduce HausaSafeLM, one of the first Hausa safety preference dataset, built from 10,000 prompts derived from PKU-SafeRLHF: 3,000 human-corrected (gold) and 7,000 machine-translated (silver) examples spanning six harm macro-categories. Using Qwen2.5-7B-Instruct, we compare supervised fine-tuning (SFT) and SFT followed by Direct Preference Optimization (DPO) under both data regimes, evaluating on 500 harmful and 200 benign Hausa prompts.We find that the untuned base model appears highly safe by refusal-rate metrics alone, but this reflects limited Hausa generation ability rather than genuine alignment: its refusals are almost 
    
[^55]: TCM-DiagCommit：一个源自TCM-SD的证据条件化证型选择与承诺决策基准

    TCM-DiagCommit: A TCM-SD-Derived Benchmark for Evidence-Conditioned Syndrome Selection and Commitment

    [https://arxiv.org/abs/2610.37243](https://arxiv.org/abs/2610.37243)

    提出TCM-DiagCommit基准，将中医证型辨析从静态最终标签选择扩展为证据条件化的动态评估框架，通过分阶段证据揭示、语义邻近候选干扰及承诺/获取/延迟/弃权的动作契约，系统考察大模型在中医诊断中的渐进推理与不确定性决策能力。

    

    中医（TCM）证型辨析通常被评估为在完整临床记录上的静态最终标签选择任务。这种设置掩盖了三种关键诊断行为：模型能否区分语义邻近的证型标签与简单的随机干扰项；模型能否随着证据字段的逐步揭示而更新预测；以及模型在不确定情况下是否应当做出诊断承诺。我们提出了TCM-DiagCommit，这是一个基于公开TCM-SD/ZY-BERT资源构建的基准，同时保留了其原始证型标签；该基准将1,834个原始病例转换为覆盖148个证型标签的23,842个任务条目。该基准将随机、档案邻近和全词表三级候选阶梯与分阶段的t1/t2/t3证据相结合，还包含固定标签的舌诊/脉诊扰动样本和证据不足变体。基准定义的动作契约对“承诺”、“获取更多证据”、“延迟决策”和“弃权”四种动作进行评分。在五次零样本DirectLLM校准中，随机……（原文摘要在此处截断）

    arXiv:2610.37243v1 Announce Type: new  Abstract: Traditional Chinese Medicine (TCM) syndrome differentiation is usually evaluated as static final-label selection over complete clinical records. This setting hides three diagnostic behaviors: whether a model distinguishes semantic-neighbor syndrome labels from easy random distractors, whether it updates predictions as evidence fields are revealed, and whether it should commit under uncertainty. We introduce TCM-DiagCommit, a benchmark derived from the public TCM-SD/ZY-BERT resource while retaining its source syndrome labels; it converts 1,834 source cases into 23,842 task rows over 148 syndrome labels. The benchmark combines random, profile-neighbor, and full-vocabulary candidate ladders with staged t1/t2/t3 evidence. It also includes fixed-label tongue/pulse perturbations and insufficient-evidence variants. A benchmark-defined action contract scores commit, acquire, defer, and abstain. Across five zero-shot DirectLLM calibrations, rando
    
[^56]: 将大语言模型智能体锚定于业务流程：面向可靠多步骤服务调用的流程感知知识图谱

    Grounding LLM Agents in Business Processes: Process-Aware Knowledge Graphs for Reliable Multi-Step Service Invocation

    [https://arxiv.org/abs/2610.37242](https://arxiv.org/abs/2610.37242)

    该论文提出将OData元数据与BPMN参考流程整合为类型化属性图知识图谱，并利用沿流程边检索的混合检索器，显著提升了LLM智能体在多步骤企业服务调用中的准确性和可靠性。

    

    调用企业API的大语言模型（LLM）智能体必须选择正确的端点、遵循流程顺序、使用有效的载荷键，并从预定义列表中选择编码值。然而，当前的检索增强式服务发现方法将端点独立处理，因此忽略了流程和模式约束。我们通过将开放数据协议元数据与业务流程模型和标记法（BPMN）参考流程整合到类型化属性图中来解决这一局限。混合检索器使用稠密嵌入识别候选端点，并沿着流程边检索后续端点。我们在42个企业服务上结合流程链真值对该方法进行评估。对于查询与下游实体之间词汇重叠度较低的查询，该方法将Recall@3提升了5.7个百分点（n=74），且在不同的嵌入模型和截断阈值下均保持一致的增益。在其他查询上，其表现与稠密检索相当，并提供了……

    arXiv:2610.37242v1 Announce Type: new  Abstract: Large language model (LLM) agents that call enterprise APIs must select the correct endpoints, follow process order, use valid payload keys, and choose coded values from predefined lists. However, current retrieval-augmented service discovery treats endpoints independently and therefore misses process and schema constraints. We address this limitation by integrating Open Data Protocol (OData) metadata and Business Process Model and Notation (BPMN) reference processes in a typed property graph. A hybrid retriever identifies candidate endpoints using dense embeddings and follows process edges to retrieve subsequent endpoints. We evaluate the method on 42 enterprise services with process-chain ground truth. For queries with low lexical overlap between the query and downstream entities, it improves Recall@3 by 5.7 points (n=74), with consistent gains across embedding models and cut-offs. It matches dense retrieval on other queries and provid
    
[^57]: 何时值得信任一种关系？多关系图学习中的操纵成本、暴露度与信任陷阱

    When Is a Relation Worth Trusting? Manipulation Costs, Exposure, and the Trust Trap in Multi-Relational Graph Learning

    [https://arxiv.org/abs/2610.37239](https://arxiv.org/abs/2610.37239)

    该论文以对手操纵成本为前提，证明了当部分人群被隔离于某关系且攻击者无力打开该门控时，度门控严格优于任何关系选择或重加权方案，并提出无需攻击即可计算的逐节点“单位成本暴露度”指标，能准确指导平台选择应加固哪种关系以最大化提升信任陷阱阈值。

    

    多关系图检测器从干净数据中学习对每种关系应给予多大的信任，但对手操纵每种关系、每个动作所需付出的代价各不相同；在将这些成本视为既定的前提下，我们探讨了何时使用某种关系反而比忽略它更糟糕。我们的第一个主要结果证明：当一部分样本被隔离在某种关系上、且攻击者无力打开这道“门”时，度门控严格优于对该关系的任何选择或重加权方案，并且这一窗口是紧致的；在真实图上，经验证的门控恰好只在满足这些条件的关系上取得收益，而在其他地方没有收益。我们的第二个结果表明，一种可扩展到消息传递的逐节点暴露度上界能够告诉平台应当加固哪种关系：无需任何攻击即可计算出的“单位成本暴露度”，在16个案例中的15个里选中的正是加固后能最大程度提高精确陷阱阈值的关系，而基于信息量、低成本或等努力分析的准则至多只能选中7个。

    arXiv:2610.37239v1 Announce Type: cross  Abstract: Multi-relational graph detectors learn from clean data how much to trust each relation, but an adversary pays a different price to manipulate each relation and each action; taking such costs as given, we ask when using a relation is worse than ignoring it. Our first main result proves that degree gating strictly beats every selection or reweighting of a relation when part of the population is isolated on it and the attacker cannot afford to open the gate, and that this window is sharp; on real graphs, validated gating gains exactly on the relations that meet these conditions and nowhere else. Our second shows that a per-node exposure bound, which extends to message passing, tells a platform which relation to harden: exposure per unit cost, computed without any attack, picks the relation whose hardening raises the exact trap threshold most in 15 of 16 cases, against at most 7 for informativeness, cheapness, or an analysis with equal edi
    
[^58]: GraphRAGProv：当 GraphRAG 遇上如何溯源

    GraphRAGProv: GraphRAG Meets How-Provenance

    [https://arxiv.org/abs/2610.37237](https://arxiv.org/abs/2610.37237)

    GraphRAGProv 通过在源文本支撑的 RDF 图上执行 SPARQL 查询并计算答案的 how-provenance 多项式，使 GraphRAG 的答案可追溯、可验证，从而能够暴露文本到图谱转换中隐藏的抽取与查询错误。

    

    GraphRAG 系统通常使用知识图谱为大语言模型（LLM）检索相关上下文，再由 LLM 生成最终答案。这使得答案更加难以解读，因此在透明度至关重要的领域吸引力较低。我们提出了 GraphRAGProv，这是一种在基于源文本的 RDF 图上执行 SPARQL 查询的架构，能够计算答案及其 how-provenance（如何溯源）多项式。这些多项式记录了使用了哪些语句实例以及它们是如何被组合的。由于每个语句实例都与源文本相关联，用户可以沿着这些链接检查提取的语句是否忠实于文本，以及查询是否将这些语句正确组合以回答问题。由此，查询本身成为检查和评估文本到图谱转换质量的一种方式。我们的交互式演示考察了来自两个多跳问答基准的 400 个案例，即使在最终答案正确的情况下，也能揭示出抽取和查询中的错误。

    arXiv:2610.37237v1 Announce Type: new  Abstract: GraphRAG systems often use knowledge graphs to retrieve relevant context for an LLM that produces the final answer. That makes answers more inscrutable and therefore less appealing for domains where transparency is paramount. We present GraphRAGProv, an architecture that executes SPARQL queries over source-grounded RDF graphs to compute answers and their how-provenance polynomials. These polynomials record which statement occurrences were used and how they were combined. Because each occurrence is linked to its source text, users can follow these links to check whether the extracted statements reflect the text and whether the query combines them to answer the question. Querying thereby becomes a way to inspect and evaluate the text-to-graph transformation. Our interactive demo examines 400 cases from two multi-hop QA benchmarks, revealing extraction and query errors even when the final answer is correct.
    
[^59]: Spine：将子任务树编译为结构化上下文以支持长程编码智能体

    Spine: Compiling Subtask Trees into Structured Context for Long-Horizon Coding Agents

    [https://arxiv.org/abs/2610.37224](https://arxiv.org/abs/2610.37224)

    Spine 通过让智能体用“工作范围”显式声明子任务边界，将子任务树编译为持久化的结构化上下文，在不依赖子智能体委派的情况下对齐长程编码任务的嵌套结构与模型上下文，避免上下文膨胀。

    

    长程编码任务包含嵌套的子任务。通常情况下，智能体将交互按顺序记录到不断增长的历史中，导致上下文膨胀和性能下降。在缺乏显式子任务边界的情况下压缩或裁剪历史，可能会丢弃未完成子任务所需的信息。相反，递归的子智能体委派方式通过子智能体来隔离已完成的子任务，将每一个子任务边界都交由委派处理，这混淆了子任务粒度与执行智能体的选择，增加了执行与协调开销。我们提出了 Spine，这是一个将模型上下文与子任务嵌套及完成情况对齐的接口与运行时。智能体通过“工作范围”声明显式的子任务边界，使同一个智能体能够内联执行嵌套子任务，同时将委派保留给独立的并行任务。该运行时以 SpineTree 来组织执行过程——这是一棵持久化的树，记录子任务之间的关系、局部……

    arXiv:2610.37224v1 Announce Type: cross  Abstract: Long-horizon coding tasks contain nested subtasks. Typically, agents record interactions sequentially into a growing history, causing context inflation and performance degradation. Compressing or trimming the history without explicit subtask boundaries can discard information needed by unfinished subtasks. Conversely, recursive subagent delegation isolates completed subtasks through child agents by assigning every subtask boundary to delegation, which conflates subtask granularity with executing agent choice, adding execution and coordination overhead. We introduce Spine, an interface and runtime that align model context with subtask nesting and completion. An agent declares explicit subtask boundaries through work scopes, allowing the same agent to execute nested subtasks inline while reserving delegation for independent parallel tasks. The runtime organizes execution in SpineTree, a persistent tree recording subtask relations, local 
    
[^60]: 检索以行动：面向鲁棒视觉-语言-动作执行的运动基元图检索

    Retrieve to Act: Motion Primitive Graph Retrieval for Robust VLA Execution

    [https://arxiv.org/abs/2610.37215](https://arxiv.org/abs/2610.37215)

    本文提出R2A测试时扩展框架，通过将成功训练演示构建为时空运动基元图，并利用查询依赖的图神经网络在图中检索结构兼容的未来运动基元，将其几何先验与冻结的VLA预测融合，从而有效提升VLA策略在分布外视觉变化下的执行鲁棒性。

    

    视觉-语言-动作（VLA）策略展现出卓越的语义泛化能力，但在执行过程中仍然极易受到分布外（OOD）视觉变化的影响。虽然现有的测试时方法试图抑制虚假的视觉线索，但它们从根本上忽略了策略自身训练经验中蕴含的丰富的、未受破坏的运动学先验。我们提出了“检索以行动”，这是一个测试时扩展框架，它将OOD鲁棒性纯粹地表述为一个几何运动基元检索问题。R2A通过将成功的训练演示组织成离线的时空运动基元图，从而绕过脆弱的视觉特征。在推理过程中，一个依赖于查询的图神经网络在该图中动态定位机器人正在进行的执行状态，检索出结构上兼容的未来运动基元。随后，这些检索到的几何先验与冻结的VLA模型的预测无缝融合。

    arXiv:2610.37215v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) policies demonstrate remarkable semantic generalization, yet remain highly vulnerable to out-of-distribution (OOD) visual shifts during execution. While existing test-time methods attempt to suppress spurious visual cues, they fundamentally overlook the rich, uncorrupted kinematic priors embedded in the policy's own training experience. We introduce Retrieve to Act (R2A), a test-time scaling framework that formulates OOD robustness purely as a geometric motion primitive retrieval problem. R2A bypasses fragile visual features by structuring successful training demonstrations into an offline spatiotemporal motion primitive graph. During inference, a query-dependent Graph Neural Network dynamically localizes the robot's ongoing execution within this graph, retrieving structurally compatible future motion primitives. These retrieved geometric priors are then seamlessly harmonized with the frozen VLA's predicted
    
[^61]: 你的通用技能优化器真的能提升智能体任务表现吗？基于二部图归因重新思考技能优化

    Does Your Generalist Skill Optimizer Really Incentivize Agentic Tasks? Rethinking Skill Optimization via Bipartite Attribution

    [https://arxiv.org/abs/2610.37201](https://arxiv.org/abs/2610.37201)

    论文提出BiPA方法，揭示通用技能优化器的开发集收益主要源于噪声任务下的选择效应而非真实改进——学习到的技能在测试时并不比手写种子更好，且额外的输入标记还带来隐性成本。

    

    技能优化器通过重写智能体的技能（即其上下文中的一段自然语言流程），并保留那些能在小型开发集上提升成功率的编辑，来使智能体适应长时程任务（例如修复终端环境）。在这类任务上，成功与否带有噪声，因此小型开发集上的提升并不能证明某次编辑真正有帮助，而且成功率分数也无法体现该编辑的代价。我们提出BiPA，它在与其种子完全相同的候选上重放优化器的接受规则，以测量仅凭选择本身所能报告出的结果；并将每个回合记录为“回合—所读文本”的二部图，从而将额外的输入标记（token）追溯到其背后的行为。在三个优化器、三个智能体基准和两个模型上的实验表明，学习到的技能的成功率与手写种子大致相当，平均仅高出0.1分。它们在开发集上的收益在测试时消失：即使在没有任何候选真正带来改进的情况下，仅在最多30个任务上保留多个候选中的最佳者也能报告出同样大小的收益。

    arXiv:2610.37201v1 Announce Type: new  Abstract: Skill optimizers adapt an agent to long-horizon tasks, such as repairing a terminal environment, by rewriting its skill, a natural-language procedure in its context, and keeping the edits that raise success on a small dev set. On such tasks success is noisy, so a gain on a small dev set does not show that an edit helps, and a success score does not show what the edit costs. We introduce BiPA, which replays an optimizer's acceptance rule on candidates identical to its seed to measure what selection alone reports, and records each episode as a bipartite graph of turns and the text they read to trace extra input tokens to the behavior behind them. Across three optimizers, three agentic benchmarks and two models, the learned skills succeed about as often as their hand-written seed, 0.1 points higher on average. Their dev gains vanish on test: keeping the best of several candidates on at most 30 tasks reports gains this large even when none c
    
[^62]: 基于延续共享的动作值校准方法，用于长时程智能体强化学习

    Action-Value Calibration via Continuation Sharing for Long-Horizon Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.37195](https://arxiv.org/abs/2610.37195)

    提出ConCal方法，通过匹配并共享回滚批次中的延续价值证据并递归向后传播校准值，减少对单一采样延续的依赖，从而缓解长时程智能体强化学习中的局部信用失真问题。

    

    面向长时程智能体的强化学习依赖于交互序列的反馈，这使得对单个动作进行信用分配变得困难。现有方法（如GiGPO）通过在共享状态下比较动作来计算相对优势，但每个动作的价值是从单个采样的延续轨迹中估计的。由此得到的局部信用仍然对轨迹中后续做出的决策较为敏感。为解决这一问题，我们提出了ConCal（Continuation Calibration，延续校准），这是一种递归的延续价值估计器，能够利用回滚批次中已经存在的延续证据。ConCal通过输入的状态-动作-状态转移来匹配同一任务中的访问记录，将共享的延续证据与每个访问自身的转移估计相结合，并将校准后的价值向后传播，以细化较早的动作价值估计。这减少了对单个采样延续轨迹的依赖，并缓解了局部信用失真问题。

    arXiv:2610.37195v1 Announce Type: cross  Abstract: Reinforcement learning for long-horizon agents relies on feedback from sequences of interactions, making credit assignment to individual actions difficult. Existing methods such as GiGPO compare actions at shared states to compute relative advantages, but estimate each action's value from a single sampled continuation. The resulting local credit remains sensitive to decisions made later in the trajectory. To address this problem, we propose ConCal (Continuation Calibration), a recursive continuation-value estimator that incorporates continuation evidence already present in a rollout batch. ConCal matches visits from the same task by their incoming state-action-state transitions, combines shared continuation evidence with each visit's own transition estimate, and propagates calibrated values backward to refine earlier action-value estimates. This reduces dependence on individual sampled continuations and mitigates local credit distortio
    
[^63]: MLE-World：利用学习到的世界模型预测机器学习实验结果

    MLE-World: Predicting Machine Learning Experiment Outcomes with a Learned World Model

    [https://arxiv.org/abs/2610.37175](https://arxiv.org/abs/2610.37175)

    提出从Qwen3.5-4B微调的世界模型MLE-World，无需实际执行程序即可预测机器学习实验的终端输出和置信度，以低成本推理替代昂贵的训练运行，使AI研究智能体的测试时扩展（TTS）变得经济可行。

    

    AI研究智能体能够提出、执行并改进解决机器学习任务的程序。然而，执行每个候选程序都需要训练和评估ML模型，这限制了智能体在每一步所能探索的候选数量。我们提出了MLE-World，这是一个从Qwen3.5-4B微调而来的世界模型，它能够在不执行程序的情况下，仅凭程序的代码和数据描述预测程序的终端输出（错误追踪、训练日志和验证分数），并同时给出一个估计预测正确性的置信度分数。由于一次预测只需一次推理调用而非一次训练运行，MLE-World使测试时扩展（TTS）对这些智能体而言变得经济可行。我们在MLE-World的引导下构建了五种TTS策略，通过预测输出在每一步比较多个候选方案：顺序修订、Best-of-N、成对评判、Fusion（融合），以及一种UCB风格的规则，后者还利用置信度分数添加了探索奖励。

    arXiv:2610.37175v1 Announce Type: cross  Abstract: AI research agents can propose, execute, and improve programs that solve machine learning tasks. However, executing each candidate program requires training and evaluating ML models, limiting how many candidates an agent can explore at each step. We introduce MLE-World, a world model finetuned from Qwen3.5-4B that predicts a program's terminal output (error traces, training logs, and validation scores) from its code and data description without executing it, together with a confidence score estimating the prediction's correctness. Because a prediction costs one inference call rather than a training run, MLE-World makes test time scaling (TTS) affordable for these agents. We build five TTS policies, guided by MLE-World, that compare several candidates per step by their predicted outputs: Sequential revision, Best-of-N, Pairwise judging, Fusion, and a UCB style rule that also adds an exploration bonus from this confidence score. Since pr
    
[^64]: Rényi正则化的基于流的贝叶斯校准变分推断

    R\`enyi-Regularized Flow-Based Variational Inference for Bayesian Calibration

    [https://arxiv.org/abs/2610.37169](https://arxiv.org/abs/2610.37169)

    本文提出一种基于Rényi散度正则化项的流式变分推断方法，解决了贝叶斯校准中标准变分推断无法捕捉被低概率区域分隔的多模态后验分布的问题，使近似后验能够覆盖先验支撑并表示多个后验模态。

    

    基于归一化流的变分推断为贝叶斯模型校准提供了灵活的后验近似，但当后验分布存在多个被低概率质量区域分隔的模态时，该方法表现不佳。我们在一个双参数孪生实验中研究了这种行为，并对变分目标进行正则化。在高斯观测误差和高斯先验下，相应的最小化目标为期望Tikhonov校准目标减去近似分布的微分熵。然而，期望Tikhonov项倾向于使分布在其中一个模态附近集中，而熵最大化并不能保证多模态性。我们引入了一个基于Rényi散度的正则化项，以鼓励近似后验覆盖先验的支撑集。在连通双峰的情形中，流近似能够表示两个模态；在模态良好分离的情形中，流与高斯混合频繁（此处摘要文本截断）

    arXiv:2610.37169v1 Announce Type: new  Abstract: Variational inference with normalizing flows provides a flexible posterior approximation for Bayesian model calibration, but struggles when the posterior has several modes separated by regions of low probability mass. We study this behavior in a two-parameter twin experiment and regularize the variational objective. Under Gaussian observation errors and a Gaussian prior, the corresponding minimization objective is the expected Tikhonov calibration objective minus the differential entropy of the approximating distribution. However, the expected Tikhonov term favors concentration near one of the modes, while entropy maximization does not guarantee multimodality. We introduce a regularization term based on R\'enyi divergence that encourages the approximate posterior to cover the prior support. In the connected bimodal case, a flow approximation represents both modes; in the case with well-separated modes, the flow and Gaussian mixture frequ
    
[^65]: LiveSearch：将问题作为可执行验证器代码的专业金融源搜索智能体实时基准

    LiveSearch: Questions as Executable Verifier Code for a Live Benchmark of Professional-Source Financial Search Agents

    [https://arxiv.org/abs/2610.37166](https://arxiv.org/abs/2610.37166)

    LiveSearch提出了一种金融搜索智能体的实时基准，将问题存储为可执行的验证器代码，在评估时通过执行程序对专业金融数据动态确定标准答案，从而检验模型检索最新金融事实的能力而非记忆过时答案的能力。

    

    在金融领域，重要的答案通常是当前的答案：价格、利率和披露信息仅在特定日期有效，因此检索最新事实是金融智能体的核心能力。然而，大多数金融基准测试忽视了时效性。它们在标注时固定每个答案，因此随着市场变动，标准答案会过时，并且一旦泄露到训练数据中，就会奖励记忆性回忆而非搜索和分析能力。我们提出了LiveSearch，这是一个实时基准，它将每个问题存储为可执行的验证器代码：一个包含对专业金融数据服务调用的类型化程序，后跟声明式提取器。问题本身是固定的，但其标准答案是在评估时通过针对专业金融数据执行该程序来确定的，并且每日任务会自动重新执行该程序，全程无需人工或语言模型参与。因此，记忆的答案很可能已经过时，模型必须通过工具和分析来获得当前答案。（摘要原文在此处截断）

    arXiv:2610.37166v1 Announce Type: new  Abstract: In finance, the answer that matters is usually the current one: prices, rates, and disclosures hold only as of a given day, so retrieving up-to-date facts is a core capability of a financial agent. Yet most financial benchmarks overlook timeliness. They fix each answer at annotation time, so the gold goes stale as markets move and, once leaked into training data, rewards recall rather than search and analysis. We introduce LiveSearch, a live benchmark that stores each question as executable verifier code: a typed program of calls to a professional financial data service, followed by a declarative extractor. The question is frozen, while its gold answer is determined by executing the program against professional financial data at evaluation time, and a daily job re-executes it with no human or language model in the loop. A memorized answer is thus likely outdated, and a model must reach the current one through tools and analysis. Construc
    
[^66]: WorkFactory：将专业意图转化为可执行工作以实现可扩展的智能体训练

    WorkFactory: Turning Professional Intents into Executable Work for Scalable Agent Training

    [https://arxiv.org/abs/2610.37161](https://arxiv.org/abs/2610.37161)

    WorkFactory 通过信息级规划和探针执行，将专业意图转化为连贯可执行的任务，构建了覆盖 612 个职业的 11 万余个任务，并据此训练出在专业基准上超越基础模型的 WorkPilot 系列智能体。

    

    扩展智能体的监督式训练需要多样化的专业任务，以产生可靠的专家演示。然而，缺失或不一致的任务信息可能会阻碍教师模型的执行，并削弱由此产生的监督信号。我们提出了 WorkFactory，一个用于构建连贯且可执行的专业任务的可扩展框架。该框架从专业意图出发，通过信息级规划构建信息支持图来指导任务实例化，显式表示预期产出所需信息的依赖关系和位置。随后，探针执行（probe executions）用于识别支持失败并指导修复，同时保留预期的工作内容。我们构建了涵盖 612 个职业的 112,686 个任务，并利用收集到的教师轨迹对 4B、27B 和 35B-A3B 规模的 Qwen 模型进行微调，得到了 WorkPilot 系列模型。这三个模型在四个专业基准测试中均超越了各自的基础模型。

    arXiv:2610.37161v1 Announce Type: new  Abstract: Scaling supervised training for working agents requires diverse professional tasks that yield reliable expert demonstrations. However, missing or inconsistent task information can obstruct teacher execution and weaken the resulting supervision. We propose WorkFactory, a scalable framework for constructing coherent and executable professional tasks. Starting from professional intents, information-level planning builds an information support graph to guide task instantiation, explicitly representing the dependencies and locations of information required for the intended outputs. Probe executions then identify support failures and guide repairs while preserving the intended work. We construct 112,686 tasks spanning 612 occupations and use the collected teacher trajectories to fine-tune Qwen models at 4B, 27B, and 35B-A3B scales, yielding the WorkPilot family. All three models improve over their respective base models across four professiona
    
[^67]: GCSR-OCC：面向开放词汇占据预测的几何校准与语义路由

    GCSR-OCC: Geometry Calibration and Semantic Routing for Open-Vocabulary Occupancy Prediction

    [https://arxiv.org/abs/2610.37149](https://arxiv.org/abs/2610.37149)

    提出GCSR-OCC框架，通过证据保持的高斯到占据算子抑制尺度膨胀与过度重叠，并结合多模式查询条件语义路由解决语义混合问题，实现从单张RGB图像的开放词汇室内3D占据预测。

    

    开放词汇3D占据预测在统一的体素网格中表示场景几何与语义，能够识别超出预定义类别的对象。其中一种方法使用3D高斯重建场景并将其转换为体素占据。然而，高斯尺度膨胀和过度重叠会扭曲体素占据预测，而跨越多个类别的高斯内部以及3D-2D语义特征对齐过程中可能产生语义混合。我们提出GCSR-OCC，该方法结合几何校准与多模式查询条件语义路由，用于从单张RGB图像进行室内占据预测。证据保持的高斯到占据算子（EC-G2O）采用核质量归一化、p范数聚合和可学习增益，在校准占据响应的同时抑制尺度膨胀和过度的局部证据累积。多模式查询优先的高斯语义路由（MM-QGSR）通过匹…（原文摘要在此处截断）

    arXiv:2610.37149v1 Announce Type: cross  Abstract: Open-vocabulary 3D occupancy prediction represents scene geometry and semantics in a unified voxel grid, enabling recognition beyond predefined categories. One approach reconstructs scenes with 3D Gaussians and converts them into voxel occupancy. However, Gaussian scale inflation and excessive overlap can distort voxel occupancy predictions, while semantic mixing can arise within Gaussians spanning multiple categories and during 3D--2D semantic feature alignment. We propose GCSR-OCC, which combines geometry calibration and multi-mode query-conditioned semantic routing for indoor occupancy prediction from a single RGB image. The Evidence-Conserving Gaussian-to-Occupancy operator (EC-G2O) uses kernel mass normalization, $p$-norm aggregation, and a learnable gain to suppress scale inflation and excessive local evidence accumulation while calibrating occupancy responses. Multi-mode Query-First Gaussian Semantic Routing (MM-QGSR) matches mu
    
[^68]: 超越参数保护：面向无重放持续学习的Fisher形EWC

    Beyond Parameter Protection: Fisher-Shaped EWC for Replay-Free Continual Learning

    [https://arxiv.org/abs/2610.37145](https://arxiv.org/abs/2610.37145)

    该论文提出FS-EWC，通过在任务学习中最小化邻域内最坏情况的Fisher信息矩阵迹来构建对参数位移不敏感的解，并采用基于轨迹的模型合并构建保留解，从而提升无重放持续学习中知识的保持能力。

    

    在无重放持续学习中，弹性权重固化（EWC）利用Fisher信息矩阵（FIM）保护已获得的知识，其方式是惩罚对历史上重要参数的改变。然而，在保留解处的局部重要性估计通常无法刻画参数发生有限位移之后的敏感性。这促使我们考虑控制未来潜在更新邻域内的Fisher敏感性。我们提出Fisher形EWC（FS-EWC），将FIM的作用从参数保护扩展到为未来保留而进行的解的构建。在任务学习期间，FS-EWC在给定邻域内最小化最坏情况下的经验FIM迹，明确偏好对有界未来位移不敏感的解。在任务完成时，基于轨迹的模型合并从多个优化状态而非单一终端迭代点构建保留解，随后在其上估计FIM。

    arXiv:2610.37145v1 Announce Type: cross  Abstract: In replay-free continual learning, Elastic Weight Consolidation (EWC) uses Fisher information matrix (FIM) to protect acquired knowledge by penalizing changes to historically important parameters. However, a local importance estimate at the retained solution does not in general characterize sensitivity after finite parameter displacement. This motivates controlling Fisher sensitivity over a neighborhood of potential future updates. We propose Fisher-Shaped EWC (FS-EWC), which extends the role of FIM from parameter protection to solution formation for future retention. During task learning, FS-EWC minimizes the worst-case empirical FIM trace within a prescribed neighborhood, explicitly favoring solutions that are less sensitive to bounded future displacement. At task completion, trajectory-based model merging constructs the retained solution from multiple optimization states rather than a single terminal iterate, after which FIM is esti
    
[^69]: 面向平衡传播的生成式回放：迈向霍普菲尔德网络中边缘友好的持续学习

    Generative Replay for Equilibrium Propagation: Toward Edge-Friendly Continual Learning in Hopfield Networks

    [https://arxiv.org/abs/2610.37139](https://arxiv.org/abs/2610.37139)

    提出基于能量的生成式回放策略EB-Gen，利用霍普菲尔德网络自身的生成动力学，让单一网络既当分类器又当样本生成器，通过松弛冻结的网络副本来生成过去任务的伪样本，从而无需外部数据缓冲即可解决平衡传播训练中的灾难性遗忘问题。

    

    平衡传播为训练霍普菲尔德网络提供了一种具有生物学基础且与硬件契合的反向传播替代方案。然而，其在持续学习场景中的表现仍在很大程度上未被探索。本工作从内到外地应对了用平衡传播训练的霍普菲尔德网络中的灾难性遗忘问题，消除了标准经验回放（Experience Replay, ER）和生成式回放（Generative Replay, Gen-Replay）不可避免需要的外部数据缓冲区或架构开销。我们提出了一种基于能量的生成式回放策略（Energy-Based Generative Replay, EB-Gen），该策略利用霍普菲尔德网络的内在生成动力学，使单一模型既能充当分类器又能充当回放样本生成器。通过松弛网络的一个冻结副本，即可生成承载过去任务知识的样本，从而仅依靠内部生成的伪数据就能完成复习。我们在类增量（Class-Incr……）设置中对该方法进行了评估（摘要原文在此处截断）。

    arXiv:2610.37139v1 Announce Type: cross  Abstract: Equilibrium Propagation (EqProp) offers a biologically grounded and hardware-aligned alternative to backpropagation for training Hopfield Networks. However, its behavior in continual learning settings remains largely unexplored. This work confronts catastrophic forgetting in EqProp-trained Hopfield Networks from the inside out, eliminating the need for external data buffers or architectural overhead that standard Experience Replay \textbf{(ER)} and Generative Replay \textbf{(Gen-Replay)} inevitably require. We introduce an Energy-Based Generative Replay strategy \textbf{(EB-Gen)} which exploits the intrinsic generative dynamics of Hopfield Networks, allowing a single model to act as both classifier and replay samples generator. Samples capturing past task knowledge are generated by relaxing a frozen copy of the network, enabling rehearsal solely through internally generated pseudo-data. We evaluate the proposed method in the Class-Incr
    
[^70]: 主Lund平面遗漏了什么：喷注标记器表征差距的模型拼接诊断

    What the Primary Lund Plane Omits: A Model-Stitching Diagnosis of a Jet-Tagger Representation Gap

    [https://arxiv.org/abs/2610.37137](https://arxiv.org/abs/2610.37137)

    本文利用模型拼接技术诊断喷注标记器的表征差距，证明主Lund平面（仅含主分解支路）相对ParT的性能缺陷源于表征信息缺失而非下游计算能力不足，且高阶相关性也被排除为成因。

    

    Lund平面与粒子Transformer（ParT）喷注标记器在性能上可能存在差异，这既可能是因为其中一种表征较弱，也可能是因为一个较弱的下游计算读取了一个本已足够的表征。仅凭独立性能无法区分这两种情况，而模型拼接可以做到：它通过一个经过训练的线性映射，将一个网络的冻结早期层与另一个网络的冻结晚期层连接起来。作为阳性对照，我们使用一个不完整的Lund表征——主分解支路——其缺陷可被独立地确认。将其拼接到ParT的下游后，其本底排除率至少能达到其自身下游所能恢复的水平，却远低于ParT的本底排除率：这表明该缺陷是表征性的，而非计算性的。基于分裂三元组的超图读取器表现也并无改善，从而排除了高阶相关性作为成因的可能。该差距在AUC中不可见（AUC保持在目标值的1%以内），仅在本底排除率上显现。

    arXiv:2610.37137v1 Announce Type: cross  Abstract: Lund-plane and Particle Transformer (ParT) jet taggers can differ in performance because one representation is weaker, or because a weaker downstream computation reads an adequate one. Standalone performance cannot separate these. Model stitching can: it links the frozen early layers of one network to the frozen late layers of another through a single trained linear map. As a positive control we use an incomplete Lund representation, the primary declustering branch, whose deficit is known independently. Stitched into ParT's downstream, it reaches a background rejection at least as much as its own downstream recovers, yet far below ParT's background rejection: the deficit is representational, not computational. A hypergraph reader over splitting triplets does no better, ruling out higher-order correlations as the cause. The gap is invisible in AUC, which stays within 1\% of its target value, and appears only in background rejection at l
    
[^71]: 基于解剖引导伪配对与对抗扩散的脊柱图像MRI到CT转换

    MRI-to-CT Translation of Spinal Images via Anatomy-Guided Pseudo-Pairing and Adversarial Diffusion

    [https://arxiv.org/abs/2610.37134](https://arxiv.org/abs/2610.37134)

    提出了一种结合解剖引导伪配对和对抗扩散模型的三阶段框架，可从未配对的脊柱MRI图像合成CT图像，从而避免辐射暴露并优化术前评估。

    

    磁共振成像（MRI）常规用于诊断脊柱疾病，但手术规划通常需要计算机断层扫描（CT）以更好地观察椎骨骨骼解剖结构。从MRI合成CT可以避免额外的辐射暴露并简化术前评估流程。我们提出一个用于脊柱MRI到CT合成的三阶段框架。首先，解剖引导的椎体匹配系统利用脊柱分割识别相似的矢状位MRI和CT切片，并通过刚性配准将其对齐，从未配对的扫描数据中创建伪配对数据。其次，使用这些配对数据训练一个用于MRI到CT转换的对抗扩散模型。第三，将合成的切片组装成3D体积，并通过一个网络进行细化，以改善切片间的连续性和结构一致性。该框架在超过10,000个未配对的矢状位MRI和CT切片上训练，并在来自83个MRI体积的1,911个切片上进行评估。

    arXiv:2610.37134v1 Announce Type: cross  Abstract: Magnetic Resonance Imaging (MRI) is routinely used to diagnose spinal disorders, but surgical planning often requires Computed Tomography (CT) for better visualization of vertebral bone anatomy. Synthesizing CT from MRI could avoid additional radiation exposure and streamline preoperative assessment.   We propose a three-stage framework for spine MRI-to-CT synthesis. First, an anatomy-guided vertebral matching system identifies similar sagittal MRI and CT slices using spine segmentations and aligns them through rigid registration, creating pseudo-paired data from unpaired scans. Second, these pairs train an adversarial diffusion model for MRI-to-CT translation. Third, synthesized slices are assembled into 3D volumes and refined by a network that improves inter-slice continuity and structural coherence.   The framework was trained on over 10,000 unpaired sagittal MRI and CT slices and evaluated on 1,911 slices from 83 MRI volumes spanni
    
[^72]: SFE-O：一种面向超临界CO₂生物质萃取与分级的知识驱动AI本体

    SFE-O: An Ontology for Knowledge-Grounded AI in Supercritical CO$_2$ Biomass Extraction and Fractionation

    [https://arxiv.org/abs/2610.37123](https://arxiv.org/abs/2610.37123)

    本文提出SFE-O——一个基于OWL 2的模块化本体，通过生物质、过程、产物三大维度及传质机制和计算依赖关系的显式建模，为超临界CO₂生物质萃取与分级的知识驱动AI应用提供了统一的知识基础。

    

    超临界CO₂萃取（CO₂-SFE）为从生物质中回收高价值化合物提供了一条可持续的途径，同时减少了传统有机溶剂的使用，但其设计与解读需要跨越生物质特性、操作条件、传质和产物分级等异构知识，而这些知识通常分散在技术报告、工程模型、外部数据库和专家实践之中。本文提出了SFE-O，一个用于超临界CO₂生物质萃取的模块化网络本体语言（OWL 2）本体，围绕相互关联的生物质、过程和产物三个维度构建，并扩展了对过程拓扑、参数来源、外部与内部传质现象以及恒定萃取速率（CER）/下降萃取速率（FER）机制的显式表示。过程各物理量之间的计算依赖关系被显式地表达出来，而无需嵌入……（摘要截断）

    arXiv:2610.37123v1 Announce Type: new  Abstract: Supercritical CO$_2$ extraction (CO$_2$-SFE) offers a sustainable route for recovering high-value compounds from biomass while limiting the use of conventional organic solvents, but its design and interpretation require heterogeneous knowledge spanning biomass properties, operating conditions, mass transfer, and product fractionation, typically fragmented across technical reports, engineering models, external databases, and expert practice. This paper introduces SFE-O, a modular Web Ontology Language (OWL 2) ontology for supercritical CO$_2$ biomass extraction, structured around interconnected Biomass, Process, and Product dimensions and extended with explicit representations of process topology, parameter provenance, external and internal mass-transfer phenomena, and the Constant Extraction Rate (CER) / Falling Extraction Rate (FER) mechanisms. Computational dependencies among process quantities are represented explicitly, without embed
    
[^73]: Prospector：学习一种用于科学推理的紧凑图遍历策略

    Prospector: learning a compact graph traversal policy for scientific reasoning

    [https://arxiv.org/abs/2610.37121](https://arxiv.org/abs/2610.37121)

    Prospector是一个紧凑模型，通过逐步遍历领域知识图谱收集证据并构建可逐边验证的证据链，在半导体制造知识图谱上以小模型性能接近最强的现成大模型。

    

    专业工程领域的科学推理依赖于散布在文献中的知识，这些知识往往是专有的，且不在前沿大语言模型的训练数据中。在这类领域，前沿模型往往只能给出泛泛的回答。知识图谱（KG）可以为大语言模型的推理提供依据。由于图谱规模过大而无法完整提供，典型的LLM+KG系统会预先选取一部分内容，即在推理开始之前就判断相关性。本文训练了一个紧凑的模型Prospector，通过逐步遍历领域知识图谱进行推理，根据已经找到的线索去寻找证据。Prospector每次调用一个小型工具集来逐步构建其上下文，并以一条证据链的形式给出答案，该证据链可以逐边对照其工具从图谱中检索到的内容进行验证。在一个由半导体制造文献构建的知识图谱上，在给定相同图谱的情况下，Prospector的性能接近最强的现成模型。

    arXiv:2610.37121v1 Announce Type: new  Abstract: Scientific reasoning in specialized engineering fields depends on knowledge scattered across the literature, often proprietary and absent from frontier LLM training data. In such domains, frontier models tend to give generic answers. Knowledge graphs (KGs) can ground LLM reasoning. Because a graph is too large to provide in full, typical LLM+KG systems preselect a portion, deciding relevance before reasoning begins. Here we train a compact model, Prospector, to reason over a domain knowledge graph by traversing it step by step, seeking evidence in light of what it has already found. Prospector calls a small set of tools one at a time to progressively build its context, and answers with a chain of evidence that can be checked edge by edge against what its tools retrieved from the graph. On a knowledge graph built from the semiconductor-manufacturing literature, Prospector approaches the strongest off-the-shelf models given the same graph 
    
[^74]: 首步偏置恢复扩散模型蒸馏中丢失的多样性

    A First-Step Bias Restores Lost Diversity in Diffusion Model Distillation

    [https://arxiv.org/abs/2610.37110](https://arxiv.org/abs/2610.37110)

    该论文发现扩散模型蒸馏导致的多样性丢失主要发生在第一步高噪声去噪阶段，据此提出ReSeed蒸馏后处理方法，通过仅在学生模型第一步添加种子响应偏置适配器并与教师模型割线速度对齐，有效恢复了蒸馏模型的输出多样性。

    

    分布匹配蒸馏使得仅需少量去噪步骤即可实现高质量的图像生成。然而，蒸馏后的学生模型在相同提示词下，往往将不同的初始噪声样本映射到相似的输出。为了定位多样性丢失的位置，我们沿采样轨迹对教师模型和学生模型进行了比较。测量发现，对不同初始噪声响应的最大降幅出现在第一步（即高噪声步骤）。当我们用教师模型对应的轨迹替换第一步，而保持学生模型最后三个步骤不变时，大部分输出多样性得以恢复。这表明第一步是恢复多样性的关键点。受此发现启发，我们提出了ReSeed——一种蒸馏后处理方法，它仅在学生模型的第一步添加一个种子响应偏置适配器，并使其速度与教师模型第一区间的割线速度对齐。所有学生模型权重和后续去噪步骤（原文在此处截断）

    arXiv:2610.37110v1 Announce Type: cross  Abstract: Distribution-matching distillation enables high-quality image generation in only a few denoising steps. However, distilled student models often map different initial noise samples to similar outputs under the same prompt. To locate where diversity is lost, we compare the teacher and student along the sampling trajectory. The largest measured reduction in the response to different initial noises occurs during the first, high-noise step. When we replace the first step with the teacher's corresponding trajectory and leave the final three student steps unchanged, most of the output diversity returns. This suggests that the first step is a key point for restoring diversity. Motivated by this finding, we introduce \emph{ReSeed}, a post-distillation method that adds a seed-response bias adapter only to the first student step and aligns its velocity with the teacher's first-interval secant velocity. All student weights and later denoising step
    
[^75]: 面向浦肯野细胞尖峰实时分类的硬件匹配脉冲神经网络

    Hardware-Matched Spiking Neural Networks for Real-Time Purkinje Cell Spike Classification

    [https://arxiv.org/abs/2610.37104](https://arxiv.org/abs/2610.37104)

    本文提出两种与目标硬件紧密匹配的紧凑脉冲神经网络，可在微控制器和专用电路等受限硬件上实时、准确地分类浦肯野细胞的简单尖峰与复杂尖峰，为无线记录头台实现片上数据压缩提供了解决方案。

    

    区分浦肯野细胞（PC）的简单尖峰（SSs）与复杂尖峰（CSs）是研究小脑功能的关键。PC 记录通常依赖有线头台，其线缆限制了自由活动动物的自然运动。无线头台消除了这一限制，但在其功耗预算内无法连续传输原始记录数据，因此头台上的尖峰分类成为一种有吸引力的数据压缩方式——前提是它能在受限硬件上实时且准确地运行。本工作提出了两个与其目标硬件相匹配的紧凑脉冲神经网络（SNNs）：ConvSNN 映射到 EFM32PG28 微控制器的矩阵向量处理器（MVP）；StreamSNN 则在每个时间步处理一个波形值，采用学习得到的时间常数和固定的突触延迟，面向专用的 SKY130 电路。在训练中未见过的细胞上进行评估，ConvSNN 仅用 1,018 个参数即达到 CS F1 分数 0.873，StreamSNN

    arXiv:2610.37104v1 Announce Type: cross  Abstract: Distinguishing Purkinje cell (PC) simple spikes (SSs) from complex spikes (CSs) is key to studying cerebellar function. PC recordings commonly rely on wired headstages, whose tethers restrict natural movement in freely behaving animals. Untethered headstages remove this constraint, but cannot continuously transmit raw recordings within their power budgets, making on-headstage spike classification an attractive form of compression, provided it runs accurately in real time on constrained hardware. This work presents two compact spiking neural networks (SNNs) matched to their target hardware: ConvSNN, which maps to the Matrix Vector Processor (MVP) of an EFM32PG28 microcontroller, and StreamSNN, which processes one waveform value per timestep using learned time constants and fixed synaptic delays for a dedicated SKY130 circuit. Evaluated on cells unseen during training, ConvSNN achieves a CS F1 of 0.873 with 1,018 parameters and StreamSNN
    
[^76]: 学会超越屏幕观察：源自GUI定位的可迁移视觉观察

    Learning to Look Beyond Screens: Transferable Visual Observation from GUI Grounding

    [https://arxiv.org/abs/2610.37093](https://arxiv.org/abs/2610.37093)

    该论文提出LBS模型，将GUI定位作为可扩展的空间监督来源，通过监督微调建立视觉检查基础能力、再通过强化学习实现多轮裁剪与检查决策，训练出可从屏幕界面泛化到通用视觉感知基准的主动观察能力。

    

    视觉感知正在从被动识别向主动的、面向任务的观察演进。我们研究将GUI（图形用户界面）定位作为可扩展的空间监督来源，用于学习能够迁移到界面之外场景的观察行为。我们提出了LBS（Look Beyond Screens，超越屏幕观察），这是一个视觉-语言模型，首先通过在教师生成的GUI定位轨迹上进行监督初始化训练，随后通过强化学习进行多轮裁剪与检查决策。在固定推理设置下，基于Qwen3.5-35B-A3B和Qwen3.8-27B的LBS模型始终优于其对应的基线模型。超越监督微调的性能提升在两个骨干模型上均可迁移至PerceptionBench，并在Qwen3.5-35B-A3B上可迁移至HR-Bench和ChartQAPro，这支持了观察策略能够超越GUI特定模仿而实现泛化。进一步分析表明两个阶段具有互补作用：SFT建立视觉检查的基本能力，而RL（摘要在此处截断）

    arXiv:2610.37093v1 Announce Type: cross  Abstract: Visual perception is evolving from passive recognition toward active, task-oriented observation. We investigate GUI grounding as a scalable source of spatial supervision for learning observation behaviors that transfer beyond interfaces. We introduce LBS (Look Beyond Screens), a vision-language model trained through supervised initialization on teacher-generated GUI grounding trajectories, followed by reinforcement learning for multi-turn crop-and-inspect decisions. Under fixed inference settings, LBS models based on Qwen3.5-35B-A3B and Qwen3.8-27B consistently improve over their corresponding baselines. Gains beyond supervised fine-tuning transfer to PerceptionBench for both backbones and to HR-Bench and ChartQAPro for Qwen3.5-35B-A3B, supporting observation strategies that generalize beyond GUI-specific imitation. Further analysis suggests complementary roles for the two stages: SFT establishes visual inspection primitives, while RL 
    
[^77]: OpenGray：一个用于评估放疗计划智能体决策的基准

    OpenGray: a benchmark for evaluating the decisions of radiotherapy planning agents

    [https://arxiv.org/abs/2610.37081](https://arxiv.org/abs/2610.37081)

    该研究提出了OpenGray基准，通过在头颈放疗数据集中加入已知正确决策的任务（如经证明不相容的剂量目标矛盾和可行对照），首次为评估LLM放疗计划智能体“提交计划还是请求审核”的决策能力提供了标准化工具。

    

    背景与目的：基于大语言模型（LLM）的放疗计划智能体能够决定何时提交计划或请求审核，但目前对其评估主要以剂量指标为主。本研究开发了OpenGray，一个包含已知正确决策的计划任务基准。材料与方法：OpenGray在OpenKBP-Opt头颈数据集上添加了具有已知答案的任务，包括两个经证明与剂量目标不相容的矛盾任务，以及一个可行对照任务（其存储的计划满足所有硬性目标）。研究在30名患者上评估了四种系统（从基于规则的计划脚本到完整的LLM智能体），其中基于LLM的系统分别采用商业LLM（Terra）和本地LLM（Ministral 3 8B）。研究记录了决策结果、实现的剂量、问题识别情况和技术故障，并从临床LLM智能体SAGE的39次立体定向放射外科运行日志中读取这些信息。结果：使用Terra时，前置LLM审核的脚本和完整智能体的决策准确率（摘要至此处截断）

    arXiv:2610.37081v1 Announce Type: new  Abstract: Background and purpose: Radiotherapy planning agents based on large language models (LLMs) decide when to submit a plan or request review but are evaluated mainly by dose. This study developed OpenGray, a benchmark of planning tasks with known correct decisions. Materials and methods: OpenGray adds tasks with known answers to the OpenKBP-Opt head-and-neck dataset, including two contradictions with dose goals proven incompatible and a feasible control with a stored plan meeting every hard goal. Four systems, from a rule-based planning script to a full LLM agent, were evaluated on 30 patients, the LLM-based ones with a commercial (Terra) and a local (Ministral 3 8B) LLM. Decisions, achieved dose, problem recognition, and technical failures were recorded and also read from logs of 39 stereotactic radiosurgery runs of SAGE, a clinical LLM agent. Results: With Terra, the script preceded by an LLM review and the full agent had decision accurac
    
[^78]: 正确性条件化作为思维链忠实性评估中的选择机制

    Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation

    [https://arxiv.org/abs/2610.37070](https://arxiv.org/abs/2610.37070)

    该研究通过结构因果模型揭示，在思维链忠实性评估中仅筛选回答正确的实例会引入选择偏差，导致测得的忠实性差距被系统性高估（在GLM-5.3上达+7.4个百分点）。

    

    将思维链忠实性评估限制在回答正确的实例上会改变被测量的总体。我们在一个七变量结构因果模型中研究这一选择步骤，并推导出精确的观测差距 Δ = Cov(A, C)/P(C=1)。该模型区分了难度混淆与捷径引起的对撞机偏差，并阐述了通过潜在响应倾向表达该差距所需的条件独立性假设。对18篇论文的协议综述区分了答案正确性限制与基于一致性选择、基准构建以及训练数据过滤。我们在算术任务上评估了四个低于上限的部署场景，并在全部四个场景中均观察到正向选择差距。主要的GLM-5.3运行产生了+7.4个百分点的差距（n = 1000，p < 0.0001），合并的Cochran-Mantel-Haenszel分析给出OR = 1.70（p < 10⁻⁶）。一种无假设的协方差分解……

    arXiv:2610.37070v1 Announce Type: new  Abstract: Restricting chain-of-thought (CoT) faithfulness evaluation to correctly answered instances changes the population being measured. We study this selection step in a seven-variable structural causal model and derive the exact observed gap $\Delta = \operatorname{Cov}(A, C)/P(C=1)$. The model distinguishes difficulty confounding from shortcut-induced collider bias and states the conditional-independence assumption needed to express the gap through latent response propensities. A protocol review of 18 papers distinguishes answer-correctness restrictions from agreement-based selection, benchmark construction, and training-data filtering. We evaluate four sub-ceiling deployments on arithmetic tasks and observe positive selection gaps in all four. The primary GLM-5.3 run yields a +7.4 percentage-point gap ($n = 1000$, $p < 0.0001$), and pooled Cochran-Mantel-Haenszel analysis gives OR = 1.70 ($p < 10^{-6}$). An assumption-free covariance decomp
    
[^79]: EnergyCal：在稀疏MoE训练中驯服Muon系列优化器的损失尖峰

    EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training

    [https://arxiv.org/abs/2610.37062](https://arxiv.org/abs/2610.37062)

    EnergyCal是一种轻量级规则，通过根据Newton-Schulz方向的归一化平方Frobenius范数衰减路由专家更新，可将稀疏MoE训练中的损失尖峰负担降低高达99.51%并改善困惑度，同时不改变谱变换、不增加持久优化器状态。

    

    稀疏混合专家（MoE）语言模型在使用Muon系列优化器训练时可能出现较大的损失尖峰。我们提出了EnergyCal，这是一种轻量级规则，根据路由专家更新的有限步Newton-Schulz方向的归一化平方Frobenius范数来衰减这些更新。EnergyCal保持谱变换不变，且不增加任何持久化的优化器状态。我们在一个40亿参数的MoE模型上对其进行了评估。与通过开发调优选定的较低学习率MuonH方案相比，EnergyCal-MuonH将平均阈值化损失尖峰负担降低了99.51%，最终几何平均困惑度降低了4.06%，吞吐量代价适中。在使用普通Muon并匹配标称学习率和权重衰减的情况下，在测试的两个指数上，平均尖峰负担的降低平均达到95.98%，并伴随终点指标的小幅改善。这些发现在所评估的设置中，支持了无论是否使用Hyperball的稳定化效果。

    arXiv:2610.37062v1 Announce Type: cross  Abstract: Sparse mixture-of-experts (MoE) language-model training can exhibit large loss spikes with Muon-family optimizers. We introduce EnergyCal, a lightweight rule that attenuates routed-expert updates according to the normalized squared Frobenius norm of their finite-step Newton-Schulz directions. EnergyCal leaves the spectral transform unchanged and adds no persistent optimizer state. We evaluate it on a 4B-parameter MoE model. Against a development-selected, lower-learning-rate MuonH recipe, EnergyCal-MuonH reduces mean thresholded loss-spike burden by 99.51% and final geometric-mean perplexity by 4.06%, with a modest throughput cost. With plain Muon at matched nominal learning rate and weight decay, the reduction in mean spike burden averages 95.98% across the two tested exponents, accompanied by small endpoint improvements. These findings support stabilization both with and without Hyperball in the evaluated setting.
    
[^80]: 探索Delta学习假设的极限：直接偏好优化中的单对生成器偏差

    Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization

    [https://arxiv.org/abs/2610.37058](https://arxiv.org/abs/2610.37058)

    该研究揭示了Delta学习假设的局限性：使用单一模型对生成的数据训练DPO会引入系统性的生成器偏差并引发奖励作弊，导致即使大模型回复质量更高，其训练效果仍持续劣于多模型偏好数据。

    

    直接偏好优化通常依赖于从多样化模型池中生成的偏好对，并由外部大语言模型裁判进行排序。最近提出的Delta学习假设表明，仅通过一对模型即可获得可比的监督信号，即将较大模型的回复视为更优回复。我们在不同模型家族和规模上研究了这种方法的局限性。我们从理论和实证两方面证明，固定的模型对会引入系统性的、依赖于生成器的偏差，这些偏差能够独立于回复质量来预测偏好标签。因此，尽管较大的生成器能产生更高质量的回复，但在单对数据上训练的DPO始终逊于使用多模型偏好数据训练的效果，我们将这一差距归因于与针对这些偏差进行奖励作弊相一致的训练动态。我们进一步表明，仅仅增加生成器的数量，即使不使用外部裁判（原文摘要在此处截断）……

    arXiv:2610.37058v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) typically relies on preference pairs generated from diverse model pools and ranked by an external LLM judge. The recently proposed delta-learning hypothesis suggests that comparable supervision can instead be obtained from a single pair of models by treating the larger model's response as preferred. We investigate the limits of this approach across model families and scales. We show, both theoretically and empirically, that fixed model pairs introduce systematic, generator-dependent biases that are predictive of the preference label independent of response quality. Consequently, DPO trained on single-pair data consistently underperforms multi-model preference data, despite the larger generator producing higher-quality responses, and we trace this gap to training dynamics consistent with reward hacking on these biases. We further show that simply increasing the number of generators, even without an e
    
[^81]: Blindfold：对LLM智能体保密的审计硬币

    Blindfold: Keeping the Audit Coin Secret from LLM Agents

    [https://arxiv.org/abs/2610.37055](https://arxiv.org/abs/2610.37055)

    该论文揭示了基于审计的LLM智能体认证的有效性还依赖于“提交前不干扰”这一保密性属性，证明其充要性并给出最坏偏差 (1-ε)D_A，同时指出泄露会跨步骤复合且提高审计率也无法弥补失效。

    

    基于审计的认证通过随机审计比例为 ε 的动作并对被审计的标签重新加权，来约束LLM智能体的违规率；其有效性本应仅取决于防御者的审计硬币。我们证明它还依赖于一个保密性属性：在动作提交之前、且在智能体自选的时刻，硬币不得干扰智能体的视图。我们将其称为“提交前不干扰”，并用 D_A——智能体在两种硬币取值下视图之间的总变差距离——来度量其违背程度。PNI 是对抗自适应智能体获得无偏证书的充分必要条件，最坏偏差为 (1-ε)D_A，且任何仅能观测被审计标签的程序都无法付出更小的代价。泄露会在步骤之间复合；跨回合共享的资源会开启一个大小可计算的信道；提高审计率也无法恢复有效性。在我们构建的一个监控器上，一个静默的……（原文此处截断）

    arXiv:2610.37055v1 Announce Type: cross  Abstract: Audit-based certification bounds the violation rate of an LLM agent by auditing a random fraction $\varepsilon$ of actions and reweighting the audited labels; its validity is meant to rest only on the defender's audit coin. We show that it also rests on a confidentiality property: the coin must not interfere with the agent's view before the action commits, at a time the agent chooses. We call this pre-commit noninterference (PNI) and measure its violation by $D_{\mathrm{A}}$, the total-variation distance between the agent's views under the two coin values. PNI is necessary and sufficient for an unbiased certificate against adaptive agents, the worst-case bias is $(1-\varepsilon)D_{\mathrm{A}}$, and no procedure that sees only audited labels can pay less. Leakage composes across steps, a resource shared across episodes opens a channel of computable size, and raising the audit rate cannot restore validity. On a monitor we built, a silent
    
[^82]: AI辅助发现受限拓扑上量子置换的CNOT分解

    AI-Assisted Discovery of CNOT Decompositions for Quantum Permutations on Constrained Topologies

    [https://arxiv.org/abs/2610.37045](https://arxiv.org/abs/2610.37045)

    本研究利用AI工具AlphaCNOT探索发现了一种受限拓扑下远程对换的CNOT分解模式，使图距离为d的两个量子比特间的置换最多仅需6d-4个CNOT门，优于基于SWAP路由的6d-3个门的构造，展示了AI在揭示新型确定性编译算法设计原理方面的价值。

    

    高效的量子线路编译对量子计算机而言是一项挑战，因为受限的量子比特连接性会给逻辑运算的实现带来显著的开销。近期AI辅助的合成方法在降低线路复杂度方面已展现出相当大的潜力，但它们在揭示能够指导新型确定性编译算法设计的原理方面的作用在很大程度上仍未被探索。在这项工作中，我们针对受限量子比特连接条件下量子置换到CNOT线路的合成问题，从这一视角展开研究。利用AlphaCNOT作为探索工具，我们发现了一种针对远程对换的可复用分解模式，改进了基于相邻SWAP路由的构造方法。对于图距离为 $d>1$ 的两个量子比特，所发现的模式给出了一种拓扑合法的实现方案，最多使用 $6d-4$ 个CNOT门，而相应的基于SWAP的方法则需要 $6d-3$ 个门。

    arXiv:2610.37045v1 Announce Type: new  Abstract: Efficient quantum circuit compilation is challenging for quantum computers, where limited qubit connectivity can introduce substantial overhead in the implementation of logical operations. Recent AI-assisted synthesis methods have shown considerable potential for reducing circuit complexity, but their role in uncovering principles that can guide the design of new deterministic compilation algorithms remains largely unexplored. In this work, we investigate this perspective for the synthesis of quantum permutations into CNOT circuits under constrained qubit connectivity. Using AlphaCNOT as an exploratory tool, we identify a recurring decomposition for remote transpositions that improves on the construction based on adjacent SWAP routing. For two qubits at graph distance $d>1$, the discovered pattern yields a topology-legal implementation using at most $6d-4$ CNOT gates, compared to the $6d-3$ gates required by the corresponding SWAP-based 
    
[^83]: Guided-RLVR：可验证奖励强化学习中的引导式探索

    Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2610.37041](https://arxiv.org/abs/2610.37041)

    提出 Guided-RLVR，通过轻量级控制器在采样分布中注入语义引导标记，扩展 RLVR 的推理探索空间，从而在数学推理和代码生成任务上持续超越现有 RLVR 方法。

    

    可验证奖励强化学习（RLVR）在提升大语言模型（LLM）在数学和编程等任务上的推理能力方面已展现出显著的实证收益。然而，近期研究表明，RLVR 的采样往往坍缩到一小部分推理轨迹上，限制了所探索的推理空间。受“关键步骤中的思考标记是 LLM 推理路径中关键结构要素”这一发现的启发，我们提出了 Guided-RLVR，这是一种通过轻量级控制器增强 RLVR 的新方法。该控制器在采样分布中策略性地注入语义上有意义的引导标记，鼓励模型从多个推理视角切入问题，从而扩大探索空间。在数学推理和代码生成基准上的实验表明，Guided-RLVR 在持续超越现有 RLVR 方法的同时……

    arXiv:2610.37041v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has shown strong empirical gains in improving the reasoning ability of large language models (LLMs) on tasks such as mathematics and programming. However, recent studies have shown that RLVR rollouts often collapse onto a narrow set of reasoning trajectories, limiting the explored reasoning space. Motivated by findings that thinking tokens at critical steps serve as critical structural elements in LLM reasoning pathways, we propose Guided-RLVR, a novel approach that augments RLVR with a lightweight controller. The controller strategically injects semantically meaningful steering tokens into the sampling distribution, encouraging the model to approach problems from multiple reasoning perspectives and thereby expanding the exploration space. Experiments on mathematical reasoning and code generation benchmarks show that Guided-RLVR consistently outperforms existing RLVR methods while pro
    
[^84]: 无需缩放的缩放 v2：从看清细节到连接证据，实现可泛化的多模态感知

    Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception

    [https://arxiv.org/abs/2610.37037](https://arxiv.org/abs/2610.37037)

    论文提出 EvidenceConnect 数据合成框架，通过利用单张图像内相关区域的放大视图生成需要比较、聚合或链接视觉证据的问题，来增强多模态模型“连接证据”这一底层能力，从而以更低的成本提升空间感知与多模态长上下文理解等任务的泛化性能。

    

    多模态感知是多模态智能的基础。其中，空间感知与多模态长上下文理解是支撑具身智能和多模态智能体的两大关键任务。然而，为这些任务合成有针对性的高质量训练数据仍然困难且成本高昂。我们的核心洞察是：这些任务共享一种底层能力，我们称之为“连接证据”，即识别相关的视觉事实、将其绑定到正确的实体、恢复它们之间的关系，并将其组合成答案。至关重要的是，针对这种底层能力合成有针对性的训练数据，远比为这些任务本身合成数据容易得多。为了增强这种能力，我们提出了 EvidenceConnect，一个易于实现的数据合成框架，它利用单张图像内相关区域的放大视图，生成需要比较、聚合或链接视觉证据的问题。为确保质量……（摘要原文在此处截断）

    arXiv:2610.37037v1 Announce Type: cross  Abstract: Multimodal perception forms the foundation of multimodal intelligence. In particular, spatial perception and multimodal long-context understanding are two pivotal tasks that underpin embodied AI and multimodal agents. However, synthesizing targeted, high-quality training data for these tasks remains difficult and costly. Our key insight is that these tasks share an underlying capability, which we call connecting evidence: identifying relevant visual facts, binding them to the correct entities, recovering their relations, and composing them into an answer. Crucially, targeted training data are far easier to synthesize for this capability than for these tasks themselves. To strengthen this capability, we propose EvidenceConnect, an easy-to-implement data synthesis framework that uses zoomed views of related regions within a single image to generate questions requiring comparison, aggregation, or chaining of visual evidence. To ensure qua
    
[^85]: 基于医学影像的可解释深度学习乳腺癌诊断模型

    Interpretable Deep Learning Models for Breast Cancer Diagnosis Using Medical Imaging

    [https://arxiv.org/abs/2610.37026](https://arxiv.org/abs/2610.37026)

    该研究对十个ImageNet预训练模型在乳腺癌组织病理学图像良恶性分类任务上进行基准测试并结合多种可解释性方法，发现DenseNet121以99.4%的准确率表现最佳，证明了迁移学习与可解释AI结合能够提供准确且可解释的乳腺癌诊断工具。

    

    深度学习在乳腺癌诊断中展现出前景，但可解释性有限阻碍了其临床应用。我们在BreakHis组织病理学数据集上，采用患者级别划分，对十个ImageNet预训练模型（ResNet50、VGG19、InceptionV3、DenseNet121、Xception、MobileNetV2、EfficientNet、ConvNeXt、ViT和Swin Transformer）进行良性与恶性分类的基准测试，并使用Grad-CAM、LIME、Integrated Gradients和LRP解释其预测结果。DenseNet121表现最佳，测试准确率达99.4%，AUC-ROC为0.999，且显著优于所有其他模型（p < 0.001）。基于Transformer的模型表现落后于CNN模型。解释结果一致突出了富含细胞核和腺体的诊断相关区域。因此，将迁移学习与可解释AI相结合，能够为乳腺癌诊断提供准确且可解释的工具。

    arXiv:2610.37026v1 Announce Type: cross  Abstract: Deep learning shows promise for breast cancer diagnosis, but limited interpretability hinders clinical adoption. We benchmark ten ImageNet-pretrained models (ResNet50, VGG19, InceptionV3, DenseNet121, Xception, MobileNetV2, EfficientNet, ConvNeXt, ViT, and Swin Transformer) for benign-versus-malignant classification on the BreakHis histopathology dataset using a patient-level split, and explain their predictions with Grad-CAM, LIME, Integrated Gradients, and LRP. DenseNet121 performed best, with 99.4% test accuracy and an AUC-ROC of 0.999, and significantly outperformed all other models (p < 0.001). Transformer-based models trailed the CNNs. The explanations consistently highlighted nuclei-rich and glandular regions of diagnostic relevance. Combining transfer learning with explainable AI can thus yield accurate, interpretable tools for breast cancer diagnosis.
    
[^86]: 基于回归模型的作物产量预测：模型性能的比较评估

    Crop Yield Prediction Using Regression Models: A Comparative Evaluation of Model Performance

    [https://arxiv.org/abs/2610.37022](https://arxiv.org/abs/2610.37022)

    本研究比较了多元线性回归、随机森林、决策树、岭回归和套索回归五种模型在基于环境和农业管理变量预测作物产量中的性能，其中多元线性回归在80:20划分下表现最佳，测试R²达约0.9130。

    

    可靠的作物产量预测能够支持农业规划和资源配置。本研究评估了五种回归模型，用于基于环境、作物相关和农业管理变量来预测作物产量。数据集最初包含100万条观测数据和十个变量，其中231条目标值为负的观测数据在预处理阶段被剔除。分类变量采用独热编码表示，布尔变量被转换为二进制值。所考虑的模型包括多元线性回归（LR）、随机森林回归（RF）、决策树回归（DT）、岭回归和套索回归。基于决定系数（R²）、均方误差（MSE）和均方根误差（RMSE），采用80:20和70:30的训练-测试划分评估了各模型的性能。在80:20划分下，多元线性回归取得了约0.9130的测试R²，以及……

    arXiv:2610.37022v1 Announce Type: cross  Abstract: Reliable crop-yield prediction can support agricultural planning and resource allocation. This study evaluates five regression models for predicting crop yield from environmental, crop-related, and agricultural management variables. The dataset initially contained 1,000,000 observations and ten variables, with 231 observations having negative target values removed during preprocessing. Categorical variables were represented using one-hot encoding, and Boolean variables were converted into binary values. The models considered were multiple linear regression (LR), random-forest regression (RF), decision-tree regression (DT), ridge regression, and lasso regression. Their performance was evaluated using 80:20 and 70:30 training-testing splits based on the coefficient of determination (R-squared), mean squared error (MSE), and root mean squared error (RMSE). Applying an 80:20 split, LR achieved a test R-squared of approximately 0.9130 and a
    
[^87]: V-LUMEN：用于视觉-语言模型嵌入扩展的视觉查找记忆

    V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

    [https://arxiv.org/abs/2610.37019](https://arxiv.org/abs/2610.37019)

    V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。

    

    嵌入层面的扩展通过基于查找的记忆来扩展语言模型的容量，而非通过增加模型的深度、宽度或密集计算。将这一思想扩展到视觉-语言模型（VLM）具有挑战性，因为视觉特征是连续的、具有空间结构的、且依赖于查询的，这与天然支持n-gram查找的离散文本标记不同。我们提出了V-LUMEN，一种面向视觉-语言模型的视觉查找记忆。V-LUMEN使用冻结的码本对视觉特征进行离散化，通过基于空间聚合的哈希构建空间感知的记忆键，并使用文本条件化的相关性信号对其进行调制；所有寻址过程都是非参数化的，检索到的嵌入通过门控残差通路融合到主干网络中。在五个推理密集型基准测试上，V-LUMEN显著提升了冻结的Qwen3-VL-2B主干网络的性能，并能零样本迁移到三个图像与训练数据完全不相交的基准测试上。

    arXiv:2610.37019v1 Announce Type: cross  Abstract: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data.
    
[^88]: 评估按需共享出租车服务中面向高级会员的优先调度机制

    Evaluating Priority Dispatch for Premium Membership in On-demand Taxi-sharing

    [https://arxiv.org/abs/2610.37011](https://arxiv.org/abs/2610.37011)

    通过SUMO仿真评估按需共享出租车中两种高级会员优先调度策略，发现最优优先机制取决于运营条件与会员比例，条件自适应的方法切换可在保障高级会员权益的同时降低非会员负担。

    

    本文使用顺序最优插入方法，分析了在按需共享出租车服务中引入高级会员的调度层面可行性。我们提出了两种调度层面的方法：一是缩短高级用户的截止时间，二是缩短截止时间并配备专用高级车辆。通过基于SUMO的仿真，我们在三种需求场景下将这两种方法与基线调度方法进行比较：稳定均匀需求、具有高峰时段的时变需求，以及高峰交通集中在城市区域的需求。结果表明，合适的优先机制取决于运营条件和高级会员比例，这意味着根据条件动态切换调度方法，可以在整体优化的同时为高级会员提供优先权益，并将非高级用户的负担保持在较低水平。

    arXiv:2610.37011v1 Announce Type: cross  Abstract: In this paper, we analyze the dispatch-level feasibility of introducing premium membership into an on-demand taxi-sharing service using a sequential optimal insertion method. We propose two dispatch-level methods: deadline shortening for premium users, and deadline shortening with dedicated premium vehicles. We compare them with a baseline dispatch method through SUMO-based simulations under three demand scenarios: steady uniform demand, time-varying demand with a peak period, and urban-focused demand with concentrated peak traffic. Our results show that the suitable priority mechanism depends on operating conditions and premium ratio, suggesting that condition-dependent method switching may provide premium benefits while keeping non-premium-user burden smaller for overall optimization.
    
[^89]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^90]: LearnPrunerV2：面向视觉-语言模型的可学习渐进式词元剪枝

    LearnPrunerV2: Learnable Progressive Token Pruning for Vision-Language Models

    [https://arxiv.org/abs/2610.36986](https://arxiv.org/abs/2610.36986)

    该论文提出 LearnPrunerV2，通过在视觉-语言模型每一层嵌入轻量级可学习模块，融合视觉显著性与查询相关性来自适应地估计词元重要性，实现完全可学习的渐进式词元剪枝，从而在大幅降低计算开销的同时保持多模态性能。

    

    视觉-语言模型展现了卓越的多模态能力，但由于海量的视觉词元而面临高昂的计算开销。尽管词元剪枝是一种有前景的解决方案，现有方法仍存在两大主要缺陷：1）剪枝准则不可靠——由于固有的注意力偏差或查询无关的设计，无法准确定位任务相关区域；2）剪枝策略次优——采用僵化、非自适应的调度方式，忽视了词元冗余度在不同模型深度中的动态演变。为应对这些挑战，我们提出了 LearnPrunerV2，它将之前的两阶段启发式框架（LearnPruner）扩展为完全可学习的架构。具体而言，我们在视觉-语言模型主干的每一层中集成了轻量级可学习模块，通过融合内在视觉显著性与外在查询相关性来估计词元重要性……

    arXiv:2610.36986v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) have demonstrated remarkable multimodal capabilities but suffer from prohibitive computational overhead due to the massive number of visual tokens. While token pruning is a promising solution, existing methods still suffer from two major drawbacks: 1) unreliable pruning criteria that fail to accurately localize task-specific regions due to inherent attention biases or query-agnostic designs, and 2) suboptimal pruning strategies that employ rigid, non-adaptive schedules, overlooking the dynamic evolution of token redundancy across different model depths. To address these challenges, we propose LearnPrunerV2, which extends the previous two-stage heuristic framework (LearnPruner) into a fully learnable architecture. Specifically, we integrate lightweight learnable modules into every layer of the VLM backbone, which estimate token importance by fusing intrinsic visual saliency with extrinsic query relevance ca
    
[^91]: 是提升还是加速？数据筛选如何改进大语言模型训练

    Advancement or Acceleration? How Data Curation Improves LLM Training

    [https://arxiv.org/abs/2610.36983](https://arxiv.org/abs/2610.36983)

    本文形式化区分了数据筛选对LLM训练的“提升”（扩展模型最终可达到的性能）与“加速”（用更少数据达到相同性能），证明只有当训练与目标之间存在可纠正的差距时筛选才能带来提升，并提出可观测的Δ-测试来检验该条件。

    

    虽然数据筛选被广泛认为能改善大语言模型的训练，但一个基本问题仍未解决：数据筛选究竟扩展了模型最终能够学习到的能力上限，还是仅仅让模型更快地达到相同的解？在本文中，我们形式化地区分了“提升”（advancement，即改善模型可达到的性能）与“加速”（acceleration，即用更少的数据达到相同性能）这两个概念。通过分析性刻画和真实世界的大语言模型实验，我们证明提升需要存在一个可纠正的训练-目标差距。当基线训练与目标相一致时，没有任何筛选策略能够改善最终训练结果，尽管数据筛选仍然可能加速训练过程。这将“提升”重新定义为筛选任务本身的属性，而非筛选方法的属性。在常见的以差距为中心的数据混合与选择视角下，我们将该条件操作化为Δ-测试，即未筛选基线的训练……（原文摘要到此截断）

    arXiv:2610.36983v1 Announce Type: cross  Abstract: While data curation is widely credited with improving large language model training, a basic question remains unresolved: does curation expand what a model can ultimately learn, or merely reach the same solution faster? In this paper, we formalize this distinction between advancement (improving the attainable performance) and acceleration (reaching the same performance with less data). Through an analytic characterization and real-world LLM experiments, we show that advancement requires a correctable train-target gap. When the baseline training and target objectives coincide, no curation policy improves the endpoint, although curation may still accelerate training. This reframes advancement as a property of the curation task rather than the curation method. Under a common gap-centric view of data mixture and selection, we operationalize this condition as the $\Delta$-test, an observable discrepancy between the uncurated baseline's trai
    
[^92]: AutoVend：基于长周期自动售货机运营的大语言模型智能体基准测试

    AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

    [https://arxiv.org/abs/2610.36980](https://arxiv.org/abs/2610.36980)

    AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。

    

    大语言模型（LLM）智能体正在将网络上的智能体中介商务从为消费者购物扩展到为卖家经营业务。在在线到线下（O2O）商务中，决策通过网络平台发布，但需要通过成本高昂的物理执行来实现。评估此类智能体需要超越单笔交易，考察在线做出的决策能否转化为持续的经济回报。我们提出了AutoVend，这是一个基准测试，其中一个LLM智能体通过基于网络的运营界面管理分布在五种站点类型中的20台自动售货机，为期180个模拟日。其基于真实数据的模拟器结合了共享仓库、采购提前期、付费补货、库存过期以及经常性运营成本等要素。我们评估了五个LLM智能体，并将其与基于规则的库存控制基线进行比较，诊断它们的站点协调能力、执行能力以及随时间推移的表现。在被评估的五个LLM智能体中，有四个表现不及基线……

    arXiv:2610.36980v1 Announce Type: new  Abstract: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short o
    
[^93]: 面向事件驱动大语言模型智能体的主动记忆

    Proactive Memory for Event-Driven LLM Agents

    [https://arxiv.org/abs/2610.36967](https://arxiv.org/abs/2610.36967)

    提出 ProactMem 主动记忆框架，将历史组织为近期事件、时间轨迹和经历史重演验证的行为规律，并构建纵向基准 ProactiveStream，使事件驱动的 LLM 智能体能更准确地预判服务机会，F1 达 52.6%，较最强基线相对提升 21.4%。

    

    主动式大语言模型智能体能够从持续的观测和积累的历史中预判服务机会。现有方法侧重于近期上下文或相关记忆的检索，对验证反复出现的行为规律以及评估其与当前事件的适用性方面支持有限。我们提出了主动记忆，它将历史组织为近期事件、时间轨迹和行为规律三个层次。轨迹事实会对照源事件进行校验，行为规律在匹配当前条件之前会通过历史重演进行评估。我们还构建了 ProactiveStream，一个涵盖三种应用范围的纵向基准，智能体在情境感知的机会标签下按时间顺序处理孤立的事件流。在 ProactiveStream 上，ProactMem 实现了 52.6% 的整体效果（F1），相比五个外部基线中最强的一个取得了 21.4% 的相对提升。

    arXiv:2610.36967v1 Announce Type: new  Abstract: Proactive LLM agents anticipate service opportunities from ongoing observations and accumulated history. Existing approaches emphasize recent context or retrieval of related memories, providing limited support for validating recurring behavioral regularities and assessing their applicability to the current event. We propose Proactive Memory (ProactMem), which organizes history into recent events, temporal trajectories, and behavioral regularities. Trajectory facts are checked against source events, and regularities are assessed through historical replay before being matched to current conditions. We also construct ProactiveStream, a longitudinal benchmark covering three application scopes, where agents process isolated event streams chronologically under episode-aware opportunity labels. On ProactiveStream, ProactMem achieves an overall effectiveness (F1) of 52.6%, a 21.4% relative improvement over the strongest of five external baseline
    
[^94]: 机器人会感到难受吗？评估大语言模型中功能性效价的实证证据

    Do bots feel bad? Assessing empirical evidence for functional valence in LLMs

    [https://arxiv.org/abs/2610.36962](https://arxiv.org/abs/2610.36962)

    本文评估了关于大语言模型功能性效价状态的实证证据，指出效价的关键方面尚未在LLM中得到检验，且现有实验范式无法将功能性效价与偏好、价值判断、效价归因和角色扮演等邻近构念充分区分开来。

    

    大语言模型（LLM）是否具有效价状态，即那些令人感到愉快或不愉快的状态？近期的实证研究声称，大语言模型拥有功能性效价状态（例如“功能性快乐与痛苦” [Ren et al. 2026]、“功能性情绪” [Sofroniew et al. 2026] 以及“功能性福祉” [Han et al. 2026]），这些状态在认知与行动中发挥着与效价状态相同的因果作用，而不必被有意识地体验。然而，这些被识别出的状态在多大程度上符合哲学与认知科学文献中所理解的效价功能角色，仍未得到探讨。本文评估了当前的证据体系，并论证：（i）效价的关键方面在大语言模型中尚未得到检验；（ii）现有的实证范式无法充分区分（功能性）效价与偏好、价值判断、效价归因和角色扮演等邻近构念。最后，作者概述了一些新颖的实验方法……

    arXiv:2610.36962v1 Announce Type: new  Abstract: Do LLMs have valenced states, i.e. states that feel pleasant or unpleasant? Recent empirical work has claimed to show that LLMs possess functional valenced states (e.g. ``functional pleasure and pain'' [Ren et al. 2026], ``functional emotions'' [Sofroniew et al. 2026], and ``functional welfare'' [Han et al. 2026]) that play the same causal role as valenced states in cognition and action, without necessarily being consciously experienced. But it remains unexplored how well the identified states fit the functional role of valence as understood in the philosophical and cognitive science literature. Here, I assess the current body of evidence, and argue (i) that key aspects of valence remain untested in LLMs, and (ii) that extant empirical paradigms inadequately distinguish (functional) valence from neighbouring constructs like preferences, value-judgments, valence attributions and role-play. I conclude by sketching some novel experimental a
    
[^95]: 拍立淘MMSearch2.0技术报告：将生成式检索扩展至统一的端到端多模态搜索

    PailitaoMMSearch2.0 Technical Report: Scaling Generative Retrieval to Unified End-to-End Multimodal Search

    [https://arxiv.org/abs/2610.36958](https://arxiv.org/abs/2610.36958)

    提出了拍立淘MMSearch，一个统一的端到端多模态生成式检索系统，可直接从原始查询图像和自由形式指令生成商品语义ID，并在数据、意图等三个维度上扩展，以支持开放式图文查询、异构购物意图和不同资源约束的在线场景。

    

    生成式检索模型在搜索和推荐系统中的应用日益广泛。然而，工业界的生成式检索模型通常专门针对固定的在线场景：它们从场景特定的交互中学习，仅支持狭窄的查询形式和检索目标。我们研究了如何消除这些耦合，构建一个统一的端到端多模态搜索系统，以支持开放式的图文查询、异构的购物意图以及具有不同资源约束的在线场景。我们提出了拍立淘MMSearch，它可以直接从原始查询图像和自由形式指令生成商品语义ID，并在三个维度上进行扩展。数据扩展使用可复用的工具集，可组合成针对特定意图和生命周期的工作流，用于冷启动合成、质量控制、真实流量细化和持续数据生产。意图扩展逐步（摘要原文在此处被截断）……

    arXiv:2610.36958v1 Announce Type: cross  Abstract: Generative retrieval models have been increasingly adopted in search and recommendation systems. However, industrial generative retrieval models are commonly specialized to a fixed online scenario: they learn from scenario-specific interactions, support a narrow set of query forms and retrieval objectives. We study how to remove these couplings and build a unified end-to-end multimodal search system for open-ended image--text queries, heterogeneous shopping intents, and online scenarios with different resource constraints. We present \textbf{PailitaoMMSearch}, which directly generates product Semantic IDs from a raw query image and free-form instruction and scales along three dimensions. \textbf{Data scaling} uses a reusable toolset that can be composed into intent- and lifecycle-specific workflows for cold-start synthesis, quality control, real-traffic refinement, and continuous data production. \textbf{Intent scaling} progressively e
    
[^96]: 脊柱关键点检测的密集表示学习与可微形状建模

    Dense Representation Learning and Differentiable Shape Modeling for Spine Landmark Detection

    [https://arxiv.org/abs/2610.36951](https://arxiv.org/abs/2610.36951)

    该论文将脊柱关键点检测重新构建为像素级对比表示学习问题，并结合可学习形状先验解决关键点位置歧义，在脊柱X光关键点检测精度和临床参数估计上达到了最先进水平。

    

    从脊柱X光片中进行脊柱关键点检测是评估脊柱畸形严重程度、构建生物力学模型以及术前规划的关键步骤。我们将解剖学关键点检测重新构建为一个像素级对比表示学习问题，将密集像素嵌入与关键点原型进行匹配。令人惊讶的是，即使使用通用的U-Net骨干网络，这种构建方式相比传统的热图回归范式也能产生更具判别性的热图，从而更好地区分不同椎骨上的椎体角点。此外，我们将所得热图与可学习的形状先验相结合，对脊柱关键点的全局配置进行建模，以解决关键点位置中的歧义问题，从而提供了一种简单实用的脊柱关键点检测方法。在关键点检测精度和临床参数估计方面，我们的方法与公开数据集上的最先进方法具有竞争力。

    arXiv:2610.36951v1 Announce Type: cross  Abstract: Spine landmark detection from spinal X-ray is a key step for estimating spine deformation severity, for building biomechanical models, and for pre-operative planning. We reframe anatomical landmark detection as a pixel-level contrastive representation learning problem, matching dense pixel embeddings with landmark prototypes. Surprisingly, even with a generic U-Net backbone, this formulation results in more discriminative heatmaps compared to the traditional heatmap regression paradigm, leading to improved discrimination between vertebral corners on distinct vertebra. Furthermore, we combine the resulting heatmaps with learnable shape priors over the global configuration of spinal landmarks to resolve ambiguities in landmark locations, yielding a simple and practical approach to spinal landmark detection. Our approach is competitive, in terms of landmark accuracy and clinical parameter estimation, with the state-of-the-art on the publi
    
[^97]: DCKV：混合语言模型中提示词KV缓存的学习式变换编码

    DCKV: Learned Transform Coding of the Prompt KV Cache in Hybrid Language Models

    [https://arxiv.org/abs/2610.36950](https://arxiv.org/abs/2610.36950)

    DCKV是首个端到端训练的KV缓存可微分编解码器，可在三个长上下文基准上以不超过1.8分的性能损失，将混合语言模型的提示词KV缓存压缩到比bf16少55倍以上的存储空间。

    

    混合语言模型仅在少数几个全注意力层中保留键值（KV）缓存，但该缓存仍然是提示词状态中随提示词长度增长的部分，并且为了复用必须被存储或传输。我们提出了DCKV，一种针对KV缓存的可微分编解码器，也是首个端到端训练的KV缓存编解码器。其变换、量化器和熵模型（以提示词token及已解码的相邻位置为条件）针对冻结的语言模型进行联合优化，同时优化熵模型所分配的码长以及模型输出的变化。在Qwen3.5-2B上，DCKV存储每个压缩token所需字节数比bf16少55倍以上，且在三个长上下文基准测试上性能损失至多1.8分；所需字节数比bf16少100倍以上时，性能损失至多3.6分。

    arXiv:2610.36950v1 Announce Type: cross  Abstract: Hybrid language models keep a key-value (KV) cache only in their few full-attention layers, but this cache is still the part of a prompt's state that grows with the prompt and that must be stored or moved to reuse it. We present DCKV, a differentiable codec for the KV cache and the first KV-cache codec that is trained end to end. Its transform, its quantizer and its entropy models, which are conditioned on the prompt tokens and on already decoded neighbouring positions, are optimized together against the frozen language model, both for the code length that the entropy models assign and for the change of the model's output. On Qwen3.5-2B, DCKV stores a compressed token in more than 55 times fewer bytes than bf16 at a loss of at most 1.8 points on three long-context benchmarks, and in more than 100 times fewer bytes at a loss of at most 3.6 points.
    
[^98]: 冻结语言模型中基于自键参数内记忆化的无外部存储对话回忆方法

    Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models

    [https://arxiv.org/abs/2610.36948](https://arxiv.org/abs/2610.36948)

    该论文提出SKIM闭式方法，让冻结语言模型为每个对话轮次自生成问题作为键写入模型参数中，从而在无外部存储的条件下仅凭参数即可回忆对话内容并回答后续提问。

    

    摘要（arXiv:2610.36948v1，通告类型：cross）：大型语言模型将对话内容保存在模型外部的一个存储中，该存储随着对话的持续而不断增长。本工作假设一个环境：语言模型没有任何外部存储，也不预先知晓未来的问题或其答案。研究动机是探究对话能否被存储在模型参数中，并在之后能够回答关于该对话的问题。本工作提出了自键参数内记忆化（Self-Keyed In-Parameter Memorization，SKIM），这是一种闭式解方法：模型针对每个对话轮次自行生成问题，并将这些问题作为键写入其参数中，以便日后检索该轮次的内容。为每个轮次生成多个多样化的问题，使得后续用户即使以不同措辞提问，也能与其中某个问题匹配，并仅从参数中召回答案。在五个语言模型上的评估表明，SKIM 能够通过后续的用户问题回忆起大多数已存储的对话轮次，并且它（摘要在此处截断）

    arXiv:2610.36948v1 Announce Type: cross  Abstract: A large language model maintains conversational content in an external storage that is outside the model and grows as the dialogue continues. This work assumes an environment where a language model has no external memory and no prior knowledge of the future question or its answer. The motivation is to research whether a dialogue can be stored in the model parameters and later answer a question about it. This work proposes self-keyed in-parameter memorization (SKIM), a closed-form method in which the model generates its own questions about each dialogue turn and writes those questions into its parameters as the keys that later retrieve the turn's content. Generating many varied questions for each turn allows a later user question to match one even when phrased differently, and its answer is recalled from the parameters alone. Evaluated on five language models, SKIM recalls most stored dialogue turns from later user questions, and it ans
    
[^99]: 超越QASecClaw：软件保障中的智能体扩展、验证器反馈与证据可靠性

    Beyond QASecClaw: Agent Scaling, Verifier Feedback, and Evidence Reliability in Software Assurance

    [https://arxiv.org/abs/2610.36947](https://arxiv.org/abs/2610.36947)

    该研究通过五组实验表明，在软件保障任务中扩大语言模型智能体上下文规模会使token开销激增49倍却无安全收益，验证器反馈虽能提升正确判定数量但同时增加错误判定，说明确定性的工具证据基线仍然不可或缺。

    

    arXiv:2610.36947v1 公告类型：cross 摘要：当确定性工具证据已经可用时，额外的专家级语言模型上下文能否改善软件保障？我们通过五个实验来研究这一问题，涵盖人工编写的Java警告控制、OWASP基准评估、Defects4J测试生成、SV-COMP断言判定，以及刻意误导性的证据包。所评估的系统使用Qwen2.5-Coder-7B-Instruct，并进行1.5B参数的敏感性分析，共完成13,638次模型请求。在八个相关的开发警告上，将上下文数量从1增加到50会使token使用量增加49倍，而未能改善不安全代码的放行情况。在160个OWASP案例中，各模型配置在放行与未解决结果之间进行权衡；十上下文配置仅评估了96个案例。在32个SV-COMP任务上，原始验证器反馈将模型判定的平均正确数从12.7提高到22.0，但也将错误判定从1个增加到7个；JBMC加JVM验证基线返回26个……（摘要在此处截断）

    arXiv:2610.36947v1 Announce Type: cross  Abstract: Can additional specialist language-model contexts improve software assurance when deterministic tool evidence is already available? We study this question through five experiments spanning authored Java warning controls, OWASP Benchmark assessment, Defects4J test generation, SV-COMP assertion verdicts, and deliberately misleading evidence packages. The evaluated systems use Qwen2.5-Coder-7B-Instruct, with a 1.5B sensitivity analysis, and complete 13,638 model requests. On eight related development warnings, increasing the context count from 1 to 50 increases token use 49-fold without improving unsafe clearance. On 160 OWASP cases, model configurations trade clearance against unresolved outcomes; the ten-context arm assesses only 96 cases. On 32 SV-COMP tasks, raw verifier feedback raises mean correct model verdicts from 12.7 to 22.0, but also raises incorrect verdicts from one to seven; a JBMC-plus-JVM validation baseline returns 26 co
    
[^100]: 视觉-语言智能体能够在它们自己的潜在世界中学习

    Vision-Language Agents Can Learn in Their Own Latent Worlds

    [https://arxiv.org/abs/2610.36925](https://arxiv.org/abs/2610.36925)

    提出了Dwell框架，将视觉-语言模型重新构建为递归潜在转移算子，使智能体能在潜在空间中从在线交互中持续学习，从而高效利用预训练世界知识实现策略改进。

    

    从经验中学习对于构建能够持续扩展自身能力的下一代智能体至关重要。预训练视觉-语言模型（VLM）已经内化了丰富的世界知识，然而这些知识很少被用于通过在线交互来驱动策略改进。高效实现这一潜力仍然具有挑战性，因为策略更新会改变智能体的交互分布，要求世界模型随之调整，而持续更新大型预训练模型的计算成本高昂。为应对这一挑战，我们提出了Dwell，一个基于模型的强化学习（RL）框架，它将VLM重新构建为递归潜在转移算子，赋予其在潜在空间中进行预测、调整和行动的能力。利用VLM自身的表示及其作为世界模型与策略的双重角色，智能体能够持续从真实交互中学习，以预测和预期……

    arXiv:2610.36925v1 Announce Type: new  Abstract: Learning from experience is essential for building the next generation of agents that continually expand their capabilities. Pretrained vision-language models (VLMs) already internalize rich world knowledge, yet this knowledge is rarely leveraged to drive policy improvement through online interaction. Realizing this potential efficiently remains challenging, as policy updates shift the agent's interaction distribution, requiring the world model to adapt accordingly, while continually updating large pretrained models is computationally expensive. To address this challenge, we introduce Dwell, a model-based reinforcement learning (RL) framework that reframes a VLM as a recurrent latent transition operator, endowing it with the ability to predict, adjust, and act in latent space. Leveraging the VLM's own representations and its dual role as both world model and policy, the agent continually learns from real interactions to anticipate and ev
    
[^101]: 开发巴斯克语语音语言模型

    Developing a Speech Language Model for Basque

    [https://arxiv.org/abs/2610.36911](https://arxiv.org/abs/2610.36911)

    本文提出了首个开源的巴斯克语语音语言模型，通过在LLaMA-Omni架构上微调Latxa模型并利用翻译语料与TTS合成音频，实现了低资源语言下的低延迟端到端口语交互。

    

    尽管大语言模型（LLM）的近期发展为多模态交互铺平了道路，但少数民族语言在跟上这一步伐方面面临困难。本工作提出了首个开源的巴斯克语语音语言模型，专为低延迟、端到端的口语交互而设计。该系统基于LLaMA-Omni架构，集成了Latxa模型作为骨干，通过微调多个模块将其巴斯克语能力迁移到语音模态。为解决数据稀缺问题，研究人员将英文VoiceAssistant-400K语料库翻译成巴斯克语，并使用巴斯克语TTS合成音频。评估结果证实了该系统的可行性：内容质量达到3.86/5分，语音可懂度的词错误率（WER）为9.94%，音频质量的UTMOS为3.07。这项工作表明，即使在低资源语言中，也可以利用模块化架构和合成数据开发先进的口语交互系统。

    arXiv:2610.36911v1 Announce Type: cross  Abstract: Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, end-to-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system's viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages.
    
[^102]: 面向指令跟随检索的从正例中挖掘自然困难负例方法

    Natural Hard-Negative Mining from Positives for Instruction-Following Retrieval

    [https://arxiv.org/abs/2610.36900](https://arxiv.org/abs/2610.36900)

    提出自然困难负例挖掘（NHNM）方法，通过自动流水线将原始正例转化为自然的指令困难负例，构建了近7K实例的IFIR训练集NHNM-MARCO，显著提升了各规模指令跟随检索模型的性能。

    

    指令跟随信息检索（IFIR）要求检索器同时根据查询及其特定实例的指令对文档进行排序，然而以指令为条件的对比监督数据仍然稀缺。现有的合成方法通常固定一个正例段落并生成额外的负例，忽视了原始正例本身蕴含的困难负例潜力。我们提出了自然困难负例挖掘（NHNM），该方法在查询兼容的指令下促进可转换负例的产生，并将原始正例重新用作自然的指令困难负例。其自动化的候选清洗、反转构造和几何过滤流水线生成了与预训练检索空间对齐的紧凑且高对比度的监督数据。应用于MSMARCO后，NHNM生成了NHNM-MARCO，一个包含近7K实例的IFIR训练集。在InstructIR和FollowIR基准测试中，在NHNM-MARCO上微调的568M至4B参数规模的模型取得了强大的指令跟随检索性能。

    arXiv:2610.36900v1 Announce Type: new  Abstract: Instruction-following information retrieval (IFIR) requires retrievers to rank documents by both a query and its instance-specific instructions, yet instruction-conditioned contrastive supervision remains scarce. Existing synthetic methods typically fix a positive passage and generate additional negatives, overlooking the hard-negative potential of the original positive. We propose Natural Hard-Negative Mining (NHNM), which promotes convertible negatives under query-compatible instructions and repurposes original positives as natural instruction hard negatives. Its automated Candidate Cleaning, Reversal Construction, and Geometry Filtering pipeline produces compact, high-contrast supervision aligned with the pretrained retrieval space. Applied to MSMARCO, NHNM yields NHNM-MARCO, an IFIR training set of nearly 7K instances. On InstructIR and FollowIR, models with 568M to 4B parameters fine-tuned on NHNM-MARCO achieve strong instruction-fo
    
[^103]: 噪声转向：利用尾流转向与声学指向性，通过强化学习实现噪声约束下的风电场控制

    Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning

    [https://arxiv.org/abs/2610.36888](https://arxiv.org/abs/2610.36888)

    提出Noise Steering无模型强化学习策略，通过联合控制风机偏航角与功率设定值，同时利用尾流转向和噪声指向性，在满足局部噪声约束的条件下最大化风电场发电功率。

    

    现有的风电场降噪策略通常通过降低风机功率输出来实现，而未考虑风机之间的尾流相互作用以及噪声发射的指向性，这导致了过于保守的策略和未被充分利用的风能。然而，风机偏航可以同时影响尾流相互作用和噪声发射方向，为风电场级噪声控制提供了额外的自由度。因此，我们提出了Noise Steering（噪声转向），这是一种无模型强化学习策略，通过联合控制风机偏航角和功率设定值，在满足特定区域局部噪声约束的同时最大化风电场功率。为了以较低的计算成本进行训练，我们将低保真声学模型与稳态FLORIS仿真器相结合。强化学习智能体有效地学会了如何将尾流和噪声指向性从受保护区域转移开，同时在风电场内重新分配功率。

    arXiv:2610.36888v1 Announce Type: new  Abstract: Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity. This leads to over-conservative policies and unexploited wind power. Yet, turbine yaw can simultaneously affect wake interactions and the direction of noise emission, offering an additional degree of freedom for farm-level noise control. We therefore propose Noise Steering, a model-free reinforcement learning (RL) strategy that jointly controls turbine yaw angles and power setpoints to maximise farm power while satisfying a local noise constraint in a defined region. To enable training at a reduced computational cost, a low-fidelity acoustic model is coupled with the steady-state FLORIS simulator. The RL agent effectively learns how to redirect wakes and noise directivity away from the protected region while redistributing power across the farm
    
[^104]: RubricAtlas：从专家报告中挖掘类型级评分标准以评估深度研究智能体

    RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents

    [https://arxiv.org/abs/2610.36878](https://arxiv.org/abs/2610.36878)

    RubricAtlas通过从3,079份中文金融研究专家报告中挖掘类型级分析要求，构建覆盖99种研究类型的评估框架，将二元检查点与报告段落关联，并把报告完整性与论证有效性分开评估，从而更全面地评估深度研究智能体。

    

    深度研究智能体执行多步骤网络搜索，并将检索到的信息综合成分析性报告。评估这些输出不仅需要检查相关事实是否存在，还需要检查它们提供了哪些分析。针对特定查询的二元评分标准能够使单个要求变得明确，但任务级检查清单可能使整体分析结构得不到充分检验。我们提出了RubricAtlas，它从同一研究类型下针对不同对象的专家报告中提炼出类型级的分析要求，并围绕反复出现的分析问题组织详细检查。这些要求为相关任务提供了关于“分析什么”的共同标准，而不规定其答案。我们的框架将二元检查点与报告段落关联起来，并将完整性评估与论证有效性评估分开进行。RubricAtlas基于3,079份中文金融研究报告构建，涵盖99种研究类型。参考报告选择实验……

    arXiv:2610.36878v1 Announce Type: new  Abstract: Deep research agents conduct multi-step Web searches and synthesise retrieved information into analytical reports. Evaluating these outputs requires checking which analyses they deliver, beyond the presence of relevant facts. Query-specific binary rubrics make individual requirements explicit, but task-level checklists may leave the overall analytical structure underexamined. We introduce RubricAtlas, which derives type-level analytical requirements from expert reports on different objects within the same research type and organises detailed checks around recurring analytical questions. The requirements give related tasks a shared standard for what to analyse, without prescribing their answers. Our framework links binary checkpoints to report passages and evaluates completeness separately from argument validity. Built from 3,079 Chinese financial research reports, RubricAtlas covers 99 research types. Reference-report selection experimen
    
[^105]: GUI-JEPA：候选条件化的JEPA潜在表示作为移动GUI智能体的快速动作策略

    GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents

    [https://arxiv.org/abs/2610.36877](https://arxiv.org/abs/2610.36877)

    GUI-JEPA是首个无解码器的GUI智能体策略，利用在无标注GUI截图上预训练的JEPA编码器潜在空间，通过候选条件化的潜在预测器和轻量级打分头在一次批量处理中为所有候选动作打分，无需逐token解码，实现了快速且省内存的移动GUI操作。

    

    移动GUI智能体通常是对70亿参数规模的视觉-语言模型进行微调，以文本形式生成每个动作。然而，这些VLM需要逐token解码每个动作，即使是那些对候选动作进行打分的智能体，也要在决策流程中保留大型语言模型或VLM，导致速度慢且内存消耗大。为解决这些问题，我们提出了GUI-JEPA，据我们所知，这是首个基于联合嵌入预测架构构建的无解码器GUI智能体策略，它在预训练于无标注GUI截图的JEPA编码器的潜在空间中选择动作。其动作和区域条件化的潜在预测器将屏幕潜在表示转换为每个候选动作对应的一个潜在表示，使轻量级打分头能够在一次批量处理中针对指令对所有候选动作进行打分，而无需解码任何token。我们进一步证明，打分器依赖于这一潜在通路，且预测器是为决策本身而训练的：一个随机初始化的预……

    arXiv:2610.36877v1 Announce Type: new  Abstract: Mobile GUI agents are typically fine-tuned 7B-scale vision-language models (VLMs) that generate each action as text. However, these VLMs decode every action token by token, and even agents that score candidate actions keep a large language model or VLM in the loop, making them slow and memory-hungry. To address these issues, we propose GUI-JEPA, to our knowledge the first decoder-free GUI agent policy built on a Joint-Embedding Predictive Architecture (JEPA), which selects actions in the latent space of a JEPA encoder pretrained on unlabelled GUI screenshots. Its action- and region-conditioned latent predictor turns the screen latent into one latent per candidate action, letting lightweight heads score all candidates against the instruction in a single batched pass, without decoding a token. We further show that the scorer relies on this latent pathway and that the predictor is trained for the decision itself: a randomly initialised pred
    
[^106]: AdaGround：面向GUI定位的令牌归一化自适应缩放与验证器在环细化方法

    AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding

    [https://arxiv.org/abs/2610.36874](https://arxiv.org/abs/2610.36874)

    AdaGround提出了一种免训练的GUI定位框架，通过依据目标大小自适应设定缩放参数、引入“是/否”验证器在环机制甄别并修正缩放结果、以及根据表头组合定位电子表格单元格，在高分辨率屏幕小目标定位上超越了现有最先进的免训练方法。

    

    GUI（图形用户界面）定位在高分辨率屏幕上的失败大多发生在小目标上，而放大画面是一种常见的免训练补救措施。然而，大多数现有的缩放流水线无法使缩放倍率与目标大小相匹配，会无条件接受缩放后的答案，且从不重新审视错误的答案。为解决这些问题，我们提出了AdaGround，一个包含三个组件的免训练框架：（1）令牌归一化自适应缩放，根据模型自身对目标大小的估计，为每个样本设定放大倍率、视野和缩放深度；（2）验证器在环细化，利用一个标记点击式的“是/否”验证器来保留或拒绝每次缩放、在缩放链之间做出选择、触发重新定位，并通过存在性查询决定何时弃权；（3）组合式结构定位，根据表头组合推导出电子表格单元格。在Qwen3-VL模型上，AdaGround超越了最先进的免训练缩放方法以及许多经过训练的模型。

    arXiv:2610.36874v1 Announce Type: cross  Abstract: GUI grounding on high-resolution screens fails mostly on small targets, and zooming in is a common training-free remedy. However, most existing zooming pipelines do not match the zoom to the target's size, accept the zoomed answer unconditionally and never revisit a wrong answer. To address these issues, we propose AdaGround, a training-free framework with three components: (1) Token-Normalized Adaptive Zooming, which sets the magnification, field of view and zoom depth of each sample from the model's own estimate of the target size; (2) Verifier-in-the-Loop Refinement, which uses one marked-click Yes/No verifier to keep or reject each zoom, choose between zoom chains, trigger re-grounding and, with an existence query, decide when to abstain; and (3) Compositional Structure Grounding, which composes spreadsheet cells from their headers. On Qwen3-VL models, AdaGround surpasses state-of-the-art training-free zooming methods and many trai
    
[^107]: ActiveAlign：通过主动感知对齐以自我为中心的操作

    ActiveAlign: Aligning Egocentric Manipulation via Active Perception

    [https://arxiv.org/abs/2610.36859](https://arxiv.org/abs/2610.36859)

    ActiveAlign 受人类主动感知启发，通过主动手部运动与可微渲染实现手眼标定、并结合触觉反馈精化物体对齐，从交互中弥合 Real2Sim2Real 差距，达到毫米级对齐精度。

    

    真实-仿真-真实（Real2Sim2Real）技术能够构建结构化的世界表征，服务于规划、可解释性和安全执行。现有方法大多依赖被动感知，由于观测的不完美而产生真实到仿真的差距。受人类通过动作进行主动感知的方式启发，我们提出了 ActiveAlign，一个利用主动感知、从交互中弥合 Real2Sim2Real 差距的框架。ActiveAlign 引入了两个互补的模块：(i) ActiveAlign-Waving，利用主动的手部运动和可微渲染来执行手眼标定；(ii) ActiveAlign-Touching，通过触觉反馈精细化物体对齐，达到毫米级精度。这些能力共同实现了强大的真实到仿真对齐。我们证明了 ActiveAlign-Waving 可用于为开源的未标定机器人操作数据集构建仿真环境，并在真实（环境中进行了演示）。（注：原文摘要在此处截断）

    arXiv:2610.36859v1 Announce Type: cross  Abstract: Real-to-Sim-to-Real (Real2Sim2Real) enables structured world representations for planning, interpretability, and safe execution. Most existing approaches rely on passive perception, which creates real2sim gaps due to imperfect observation. Inspired by how humans perceive actively through actions, we propose ActiveAlign, a framework that leverages active perception to align the Real2Sim2Real gap from interactions. ActiveAlign introduces two complementary modules: (i) ActiveAlign-Waving, which performs hand-eye calibration by exploiting active hand motions and differentiable rendering, and (ii) ActiveAlign-Touching, which refines object alignment through tactile feedback to achieve millimeter-level precision. Together, these capabilities enable strong real2sim alignment. We demonstrate that ActiveAlign-Waving can be used to curate simulation environments for open-sourced uncalibrated robot manipulation datasets, and demonstrate on a real
    
[^108]: 当所有采样全部失败时：面向困难问题推理的策略兼容联合解码

    When All Rollouts Fail: Policy-Compatible Joint Decoding for Hard-Problem Reasoning

    [https://arxiv.org/abs/2610.36849](https://arxiv.org/abs/2610.36849)

    提出策略兼容强化学习方法PICO，通过策略兼容的联合解码使专家轨迹与当前策略分布相容，从而在所有采样都失败、标准RLVR学习信号坍缩为零的困难问题上获得既可验证正确又可迁移的有效学习信号。

    

    通过可验证奖励的强化学习（RLVR）通过在线策略探索从二值奖励中获取学习信号，已成为提升大语言模型推理能力的强大范式。然而，在超出模型当前能力的问题上，它面临“学习悬崖”：当每一次采样全部失败时，样本无法揭示任何关于正确解的信息，组相对估计器（如GRPO、RLOO）会将优势更新坍缩为零。专家轨迹提供了一种补救方法，但它们与当前策略不兼容，往往位于学生模型的低概率区域，因此存留的更新需要大幅的参数变化，并退化为表面模仿而非可迁移的推理。我们认为，在困难问题上的有用学习信号必须同时满足两点：可验证的正确性，以及与当前策略的充分兼容性。我们提出了策略兼容强化学习（PICO），它执行分布式……（摘要在此处被截断）

    arXiv:2610.36849v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) derives its learning signal from binary rewards via on-policy exploration and becomes a powerful paradigm for improving the reasoning of large language models. Yet it faces a learning cliff on problems beyond model's current ability: when every rollout fails, the samples reveal nothing about a correct solution, and group-relative estimators (e.g., GRPO, RLOO) collapse the update to zero advantage. Expert trajectories offer a remedy, but they are incompatible with the current policy and often lie in the student's low-probability regions, so the surviving updates demand large parameter shifts and decay into surface-level imitation rather than transferable reasoning. We argue that a useful learning signal on hard problems must be both verifiably correct and sufficiently compatible with the current policy. We propose Policy-Compatible Reinforcement Learning (PICO), which performs distri
    
[^109]: 进化智能体团队

    Evolving Agent Teams

    [https://arxiv.org/abs/2610.36848](https://arxiv.org/abs/2610.36848)

    提出多智能体框架“进化智能体团队”，通过中央监督者诊断局部最优并动态重构智能体团队，并借助持久化文件系统积累的经验指导重构，在相同token预算下持续超越现有编码智能体基线，并从零生成比FlashInfer生产级MLA分页解码内核平均快1.2倍的CUDA内核。

    

    软件工程智能体的快速发展推动了将它们应用于GPU内核优化等开放性问题的兴趣。然而，我们发现现有的编码智能体在面对真实世界中由专家编写的内核（如FlashInfer）时会收敛到局部最优。我们提出了进化智能体团队（Evolving Agent Teams），一个能够摆脱这种局部最优的多智能体框架。它具有两个核心特性：(1) 一个中央监督者，用于诊断每个局部最优并重构智能体团队；(2) 一个持久化文件系统，包含学习文档、进化历史和每个团队的产物，以积累的经验为团队重构提供依据。在相同的token预算下，进化智能体团队始终优于强大的编码智能体基线。在扩展的token预算下，进化智能体团队从零开始生成了一个CUDA内核，相比生产级的FlashInfer MLA分页解码内核实现了平均1.2倍的加速，并在47个生产工作负载中的36个上取得胜利。

    arXiv:2610.36848v1 Announce Type: new  Abstract: The rapid progress of software engineering agents drives interest in applying them to open-ended problems such as GPU kernel optimization. However, we find that existing coding agents converge to local optima on real-world, expert-written kernels (e.g., FlashInfer). We introduce Evolving Agent Teams, a multi-agent framework that escapes such local optima. It features (1) a central supervisor that diagnoses each local optimum and restructures a team of agents; and (2) a persistent file system that comprises a learning document, an evolution history, and per-team artifacts, grounding restructuring with accumulated experience. Under a matched token budget, Evolving Agent Teams consistently outperforms strong coding-agent baselines. With an extended token budget, Evolving Agent Teams generates a CUDA kernel that reaches 1.2x mean speedup over the production FlashInfer MLA paged decode kernel from scratch and wins on 36 of 47 production workl
    
[^110]: 面向零和部分可观测随机博弈的保守策略迭代

    Conservative Policy Iteration For Zero-Sum POSGs

    [https://arxiv.org/abs/2610.36847](https://arxiv.org/abs/2610.36847)

    提出了一种针对有限时域零和部分可观测随机博弈的近似策略迭代算法，通过保守策略更新和基于当前联合策略的支撑超平面来替代对完整中心价值函数的近似，将计算集中于联合策略可被利用的占用区域。

    

    占用态约简使得动态规划可以应用于零和部分可观测随机博弈（zs-POSG）。现有的基于点的求解器通常优化一个中心价值函数，这虽然恢复了贝尔曼递归，但需要在高维占用空间上进行基于点的更新。我们为有限时域零和部分可观测随机博弈提出了一种近似策略迭代算法。该方法为每个玩家各维护一个策略，并在由保守策略更新所诱导的占用轨迹上对双方策略进行联合更新。每次更新不再近似完整的中心价值函数，而是仅使用由当前联合策略诱导的单个支撑超平面。算法交替进行局部策略改进、保守更新和持续探索，因此计算能够集中于当前联合策略可被利用的占用区域。在有限时域零和POSG基准测试上，该方法在较小规模上具有竞争力（原文在此处截断）。

    arXiv:2610.36847v1 Announce Type: new  Abstract: Occupancy-state reductions make dynamic programming possible for zero-sum partially observable stochastic games (zs-POSGs). Existing point-based solvers usually optimise a central value function. This restores Bellman recursion, but it requires point-based updates over high-dimensional occupancy spaces. We introduce an approximate policy-iteration algorithm for finite-horizon zs-POSGs. The method maintains one policy per player and updates both jointly along occupancy trajectories induced by conservative policy updates. Instead of approximating a full central value function, each update uses a single supporting hyperplane induced by the current joint policy. The algorithm alternates local policy improvements, conservative updates, and persistent exploration. Computation is therefore concentrated on occupancy regions where the current joint policy is exploitable. On finite-horizon zero-sum POSG benchmarks, the method is competitive at sma
    
[^111]: 当评判者落后时：评测标准与执行框架的协同进化以实现智能体的持续自我改进

    When the Critic Falls Behind: Co-Evolving Rubrics and Harnesses for Continual Agent Self-Improvement

    [https://arxiv.org/abs/2610.36826](https://arxiv.org/abs/2610.36826)

    提出了 RuHarness 框架，通过评测标准与执行框架的协同进化，持续从执行轨迹中发现并更新行为准则以指导针对性框架改进，实现智能体的持续自我提升并显著增强对未见任务的泛化能力。

    

    arXiv:2610.36826v1 公告类型：新论文 摘要：自主智能体越来越依赖于围绕大语言模型（LLM）构建的执行框架，用于管理交互、记忆、工具和执行。尽管近期的工作已开始从执行经验中进化这些框架，但用于指导进化的监督信号往往保持相对静态，这使得难以追踪跨轮次不断变化的行为弱点，并可能限制对未见任务的泛化能力。为解决这一问题，我们提出了 RuHarness，一个用于智能体持续自我改进的评测标准与执行框架协同进化框架。RuHarness 持续从执行轨迹中发现并重新评估行为准则，优先处理最相关的未解决弱点，并利用它们指导有针对性的框架更新，同时保持任务目标不变。在多样化智能体任务和多种骨干模型上的大量实验表明，RuHarness 优于强大的框架进化基线，尤其是在未见任务上表现突出。

    arXiv:2610.36826v1 Announce Type: new  Abstract: Autonomous agents increasingly depend on harnesses that govern interaction, memory, tools, and execution around large language models (LLMs). While recent work has begun to evolve these harnesses from execution experience, the supervision used to guide evolution often remains relatively static, making it difficult to track changing behavioral weaknesses across rounds and potentially limiting generalization to unseen tasks. To address this issue, we propose RuHarness, a rubric--harness co-evolution framework for continual agent self-improvement. RuHarness continually discovers and reassesses behavioral criteria from execution trajectories, prioritizes the most relevant unresolved weaknesses, and uses them to guide targeted harness updates while keeping the task objective fixed. Extensive experiments across diverse agent tasks and backbone models show that RuHarness outperforms strong harness-evolution baselines, particularly on unseen hel
    
[^112]: 谁知道什么——它又能支持什么？面向多模态心智理论的视角有界证据校准

    Who Knows What - and What Does It Support? Perspective-Bounded Evidence Calibration for Multimodal Theory of Mind

    [https://arxiv.org/abs/2610.36824](https://arxiv.org/abs/2610.36824)

    该论文将多模态心智理论中的核心局限形式化为证据可采纳性问题，主张只有当证据对相关智能体可获取、保持信息来源一致且对目标心理状态具有诊断性时才能用于心理状态推断，从而避免视角泄露与来源混淆。

    

    心智理论是推断他人信念、目标和意图的能力，是社会智能的基石。多模态心智理论要求从特定智能体可获得的证据中推断其心理状态，而非从求解者所观察到的完整情节中进行推断。现有的基于大语言模型的方法和基于贝叶斯模型的方法都会重建全局事件轨迹，但在假设评估过程中并未明确区分客观事件与智能体可采纳的证据。因此，智能体无法获知的事件或语义错位的动作与言语可能会影响所推断的信念、目标或社交意图，导致视角泄露以及信息来源或路径的混淆。我们将这一局限表述为证据可采纳性问题：只有当证据对相关智能体可获取、保留其信息来源、并且对所查询的心理状态具有诊断性时，该证据才应支持相应的心理状态假设。

    arXiv:2610.36824v1 Announce Type: new  Abstract: Theory of Mind (ToM), the ability to infer others' beliefs, goals, and intentions, is a cornerstone of social intelligence. Multimodal ToM requires inferring an agent's mental state from evidence that is available to that agent, rather than from the complete episode observed by the solver. Existing LLM-based and Bayesian model-based approaches reconstruct global event traces, yet they do not explicitly distinguish objective events from agent-admissible evidence during hypothesis evaluation. Consequently, agent-inaccessible events or semantically misaligned actions and utterances may influence the inferred belief, goal, or social intention, leading to perspective leakage and source or route confusion. We formulate this limitation as an evidence-admissibility problem: evidence should contribute to a mental-state hypothesis only when it is accessible to the relevant agent, preserves its information source, and is diagnostic of the queried m
    
[^113]: TAPESTRY：视频生成与机器人控制的单一骨干网络

    TAPESTRY: A Single Backbone for Video Generation and Robot Control

    [https://arxiv.org/abs/2610.36823](https://arxiv.org/abs/2610.36823)

    TAPESTRY通过将机器人动作作为单补丁标记嵌入视频模型的潜在空间，用一个轻量级线性投影将动作映射到预训练VAE维度，使同一预训练视频扩散Transformer骨干网络既能用于视频生成又能直接作为机器人控制策略，且推理时无需生成视频。

    

    预训练视频生成模型编码了关于物理交互的丰富先验知识，但将其应用于机器人控制通常需要在测试时进行资源密集型的视频生成，或者需要从头训练且未能充分利用预训练骨干网络的动作头。我们提出TAPESTRY（在共享Transformer策略中将动作标记为视频补丁），这是一种世界动作模型，它将预训练的视频扩散Transformer改造为机器人策略，而无需在推理时进行视频生成。我们的核心思想是将机器人动作视为视频模型潜在空间中的单补丁标记。一个轻量级的线性投影将每个动作时间步映射到预训练VAE的补丁维度，所得标记由与视频相同的补丁化层和Transformer骨干网络处理。共享骨干网络通过三个扩散目标进行优化，每个小批量元素采样一个目标（摘要在此处截断）。

    arXiv:2610.36823v1 Announce Type: cross  Abstract: Pretrained video generation models encode rich priors about physical interactions, but leveraging them for robot control typically requires either resource-intensive video generation at test time or action heads trained from scratch that do not fully utilize the pretrained backbone. We introduce Tokenizing Actions as Video Patches in a Shared Transformer Policy (TAPESTRY), a world action model that repurposes a pretrained video diffusion transformer into a robotic policy without requiring video generation at inference. Our key idea is to treat robot actions as single-patch tokens in the video model's latent space. A lightweight linear projection maps each action timestep to the patch dimensions of the pretrained VAE, and the resulting tokens are processed by the same patchifying layers and transformer backbone used for video. The shared backbone is optimized through three diffusion objectives sampled per minibatch element: Video diffus
    
[^114]: 面向证据支撑早期预警的增量式知识图谱学习者数字孪生

    An Incremental Knowledge-Graph Learner Digital Twin for Evidence-Grounded Early Warning

    [https://arxiv.org/abs/2610.36820](https://arxiv.org/abs/2610.36820)

    本文提出一种基于轻量级知识图谱的增量式学习者数字孪生，将特征、归因、预警与原始学习记录统一表示为可查询的图实体，从而实现有证据支撑、可解释的学业风险早期预警。

    

    学习分析领域的早期预警模型能够识别面临学业失败或退学风险的学习者，但由此产生的预警往往与可查验的学习证据关联薄弱。风险分数可能仅表明需要干预，却不展示究竟是哪些学习轨迹、测评事件或时间模式支撑了该预警。本文提出了一种面向证据支撑早期预警的增量式学习者数字孪生，并以轻量级知识图谱的形式加以实现。该数字孪生通过回放学习轨迹与测评记录，在时间检查点上不断更新，既可作为特征提取的基础，也可作为解释溯源的基础。预测特征通过基于图谱的查询推导得出，而特征、归因、预警、证据条目及原始记录均以可查询的图实体形式表示。该框架在一个公开的学习分析基准数据集上进行了实例化与评估，并在时间（原文摘要在此处截断）

    arXiv:2610.36820v1 Announce Type: new  Abstract: Early-warning models in learning analytics can identify learners at risk of failure or withdrawal, but the resulting alerts are often only weakly connected to inspectable learning evidence. A risk score may indicate that intervention is needed without showing which traces, assessment events, or temporal patterns justify the alert. This paper introduces an incremental Learner Digital Twin for evidence-grounded early warning, implemented as a lightweight knowledge graph. The twin is updated at temporal checkpoints by replaying learning traces and assessment records, and it serves both as a feature-extraction substrate and as an explanation-grounding substrate. Predictive features are derived through graph-mediated queries, while features, attributions, alerts, evidence items, and raw records are represented as queryable graph entities. The framework is instantiated and evaluated on a public learning-analytics benchmark under a temporal pro
    
[^115]: SetGRPO：校准评分器诱导的策略优化

    SetGRPO: Calibrating Grader-Induced Policy Optimization

    [https://arxiv.org/abs/2610.36807](https://arxiv.org/abs/2610.36807)

    提出SetGRPO方法，通过联合奖励集合校准和几何引导的安全恢复等模块来校准评分器随机性对GRPO更新的影响，从而提升基于评分反馈的强化学习训练的稳定性与安全性。

    

    基于评分标准的强化学习利用细粒度的评分器反馈来训练大语言模型完成开放式任务，但评分器的随机性可能会以仅凭分数差异无法刻画的方式改变GRPO的更新。由于GRPO在每个响应组内联合归一化奖励，并将由此得到的训练系数与响应级梯度相结合，一次评分差异可能会改变整组的系数，并引发依赖于策略的更新变化。选择性执行还会进一步影响被保留更新的方向反转风险。为解决这一问题，我们提出了集合值组相对策略优化。该方法包含三个模块：联合奖励集合校准为训练系数构建联合不确定性集合；几何引导的安全恢复将候选系数映射到当前策略的局部更新几何中，并为每个……确定最大的安全恢复比率

    arXiv:2610.36807v1 Announce Type: cross  Abstract: Rubric-based reinforcement learning uses fine-grained grader feedback to train large language models on open-ended tasks, but grader randomness can alter GRPO updates in ways that score discrepancies alone cannot characterize. Because GRPO jointly normalizes rewards within each response group and combines the resulting training coefficients with response-level gradients, a scoring discrepancy can shift the coefficients of the entire group and induce policy-dependent update changes. Selective execution further affects the direction-reversal risk of retained updates. To address this, we propose Set-Valued Group Relative Policy Optimization (SetGRPO). It comprises three modules: (a) Joint Reward Set Calibration constructs a joint uncertainty set for the training coefficients; (b) Geometry-Guided Safe Recovery maps candidate coefficients into the current policy's local update geometry and determines the largest safe recovery ratio for each
    
[^116]: 《一种Transformer代数》

    An Algebra of Transformers

    [https://arxiv.org/abs/2610.36795](https://arxiv.org/abs/2610.36795)

    本文首次提出了一种基于函数分解结果的Transformer代数系统，通过顺序组合、并行组合、递归和抽象等操作实现对Transformer的逻辑建模，并可用于验证基于Transformer的应用的正确性。

    

    目前针对Transformer的理论分析主要集中在可计算性方面，例如无界精度Transformer的图灵完备性、无界精度Transformer的并行化优势，以及有界精度Transformer的局限性。本文为Transformer提供了一种用于推理的代数系统，该代数在很大程度上依赖于Transformer的函数分解结果。该代数以函数分解作为原子操作，包含顺序组合与并行组合、用于反馈的递归，以及用于隐藏内部层的抽象。据我们所知，这是首次以逻辑方式对Transformer进行建模的尝试，可用于验证基于Transformer的应用程序的正确性。

    arXiv:2610.36795v1 Announce Type: cross  Abstract: On the theoretic analyses on transforms, they mainly focus on the computability analysis, such the Turing-completeness of unbounded precision transformers, the advantage of parallelizability of unbounded precision transformers and the limitations of bounded precision transformers. This paper give transformers an algebra for reasoning, which relies heavily on the functional decomposition results of transformers. This algebra uses functional decompositions as atomic actions, includes sequential composition and parallel composition, recursion for feedback and abstraction for hidden layers. As we know, it is the first attempt to model transformers logically and can be used to verify the correctness of transformer based applications.
    
[^117]: Select-TTT：面向原位测试时训练的查询条件化选择性快权重更新

    Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training

    [https://arxiv.org/abs/2610.36794](https://arxiv.org/abs/2610.36794)

    Select-TTT提出了一种模型无关的查询条件化规则，通过根据各数据块与用户查询的相关性来动态缩放其快权重写入强度，解决了原位测试时训练中统一写入调度导致无关内容稀释查询相关信息的问题。

    

    测试时训练（TTT）使部署后的语言模型能够将传入的上下文即时内化到一小组快权重参数中，而不仅仅是将其视为供注意力机制使用的瞬态上下文。原位TTT（In-Place TTT）最近通过将标准MLP块的下投影重新用作快权重，使这一思想在大型语言模型中变得实用。然而，我们发现其收益在真实的网页长上下文问答任务中基本消失。我们将此归因于原位TTT的统一写入调度：每个数据块都以相同强度写入，导致无关内容稀释了快权重记忆中与查询相关的信息。基于这一洞察，我们提出了Select-TTT，这是一种模型无关、查询条件化的原位TTT规则，它根据每个数据块与用户查询的相关性来缩放其写入强度。该相关性信号无需监督且无需新增参数：它直接从模型的冻结输出（此处摘要原文截断）中读取。

    arXiv:2610.36794v1 Announce Type: cross  Abstract: Test-time training (TTT) enables a deployed language model to internalize incoming context on-the-fly into a small set of fast-weight parameters, rather than merely treating it as transient context for attention. In-Place TTT recently made this idea practical for large language models by repurposing the down-projection of standard MLP blocks as fast weights. However, we find that its benefit largely disappears on realistic web long-context question answering. We attribute this to In-Place TTT's uniform write schedule: every chunk is written with equal strength, so irrelevant content dilutes query-relevant information in fast-weight memory. Building on this insight, we propose Select-TTT, a model-agnostic, query-conditioned rule for In-Place TTT that scales each chunk's write strength by its relevance to the user query. This relevance signal requires no supervision and no new parameters: it is read directly from the model's frozen outpu
    
[^118]: 数学论证的依赖图：表示、结构度量与证明审阅

    Dependency Graphs of Mathematical Arguments: Representation, Structural Measurement, and Proof Review

    [https://arxiv.org/abs/2610.36776](https://arxiv.org/abs/2610.36776)

    提出了一个基于源文本、借助大语言模型辅助的数学论证依赖图表示与度量框架，通过深度与体量等结构指标、并集递推式及可达性保持条件，并配合标准化版本化提示协议，实现可靠的证明审阅。

    

    我们提出了一个以源文本为依据、在大语言模型辅助下表示和度量数学论证的框架。该框架涵盖有编号与无编号的命题、公式、假设及构造，并区分源头出现、上下文状态与推理记录。所提出的依赖关系保留源头证据和审阅状态。度量契约固定手稿版本、边界、粒度、证明选择和宏观锚点。在选定的有限有向无环图上，深度定义为最长前提路径的长度，体量则统计不同的祖先边数量。我们给出了一个考虑共享子论证的并集递推式，并识别出保持深度和宏观骨架的可达性条件；不受限制的细化不一定能保持深度或体量。一套标准化、带版本管理的提示词协议将标注发现、确定性重算与审阅辅助相互分离。

    arXiv:2610.36776v1 Announce Type: new  Abstract: We propose a source-grounded framework for representing and measuring mathematical arguments with assistance from large language models. It includes numbered and unnumbered statements, equations, assumptions, and constructions, distinguishing source occurrences from contextual states and inference records. Proposed dependencies retain source evidence and review status. A measurement contract fixes manuscript versions, boundaries, granularity, proof choices, and macro anchors. On a selected finite directed acyclic graph, depth is the longest prerequisite-path length, and bulk counts distinct ancestor edges. We give a union recurrence that accounts for shared subarguments and identify reachability conditions preserving depth and the macro skeleton; unrestricted refinement need not preserve depth or bulk. A standardized, versioned prompt protocol separates annotation discovery, deterministic recomputation, and review assistance. Cross-paper
    
[^119]: 批量化反思优化循环：并行提议使GEPA更快更好

    Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better

    [https://arxiv.org/abs/2610.36774](https://arxiv.org/abs/2610.36774)

    将GEPA优化器扩展为每步并行提议和评估一批候选方案，使反思优化循环的运行时间缩短至原来的四分之一到二分之一，同时提升了测试分数和预算利用效率。

    

    反思式文本优化能够提升智能体系统的性能，但优化循环本身就是一个资源密集型的工作负载，其串行结构导致并行评估能力闲置。我们扩展了开源的GEPA优化器，使其在每个优化步骤中提议并评估一批候选方案，而不是每次只处理一个候选方案。在两项任务的实验中，大多数批量化运行在不超过一半的墙上时钟时间内完成，最快的仅需约四分之一到三分之一的时间。批量化设置还取得了更高的留出测试集分数：在LiveBench-Math上从68.9%提升到72.1%（使用2×2配置），在HoVer上从49.0%提升到60.0%（使用8×1配置）。一个简单的运行时模型解释了在有限工作池下测得的加速效果，批量化设置还能更高效地使用其指标调用和资金预算。我们提出了反思式优化步骤可以并行扩展的各个维度，这是对如何分配固定优化预算的首次探索。

    arXiv:2610.36774v1 Announce Type: new  Abstract: Reflective text optimization makes agentic systems better, but the optimization loop is itself a resource-hungry workload whose sequential structure leaves parallel evaluation capacity idle. We extend the open-source GEPA optimizer to propose and evaluate a batch of candidates on each optimization step instead of one candidate at a time. In our sweep on two tasks, most batched runs finished in half the wall-clock time or less, and the fastest in about a quarter to a third. Batched settings also achieved higher held-out test scores, from 68.9% to 72.1% on LiveBench-Math (with $2\times2$) and from 49.0% to 60.0% on HoVer (with $8\times1$). A simple runtime model explains the measured speedups under a finite worker pool, and batched settings also spend their metric-call and dollar budgets more efficiently. We frame the axes along which a reflective optimization step can scale in parallel, a first look at allocating a fixed optimization budg
    
[^120]: 大语言模型能否可靠地评判奥赛证明？基于大语言模型的数学验证研究

    Can LLMs Reliably Grade Olympiad Proofs? A Study of Mathematical Verification with LLMs

    [https://arxiv.org/abs/2610.36767](https://arxiv.org/abs/2610.36767)

    本文通过构建带专家评分的USAMO 2025人工标注数据集并提出TROJAN对抗性证明生成流水线，系统性地评估了大语言模型验证奥赛级数学证明的能力。

    

    大语言模型（LLMs）在解决竞赛级数学问题上取得了显著进展，但它们验证自然语言数学证明的能力仍相对缺乏深入研究。这一差距在奥赛级证明上尤为突出，因为奥赛级证明需要对长推理链中的每一个论断进行细致评估。因此，目前的证明验证主要依赖专家审查，而这种方式难以规模化扩展。在本文中，我们对该场景下基于大语言模型的验证进行了系统性研究。具体而言，我们引入了一个由人工整理并附有专家评分的USAMO 2025候选解答数据集，并对前沿模型中现有的LLM验证器进行了广泛评估。我们还提出了TROJAN，一个用于大规模生成高保真对抗性证明的迭代式自我批评流水线。我们的全面评估涵盖20个模型，涉及七项指标、两种提示风格、三种推理方法以及三种（摘要在此处截断）。

    arXiv:2610.36767v1 Announce Type: new  Abstract: Large Language Models (LLMs) have made remarkable progress on solving competition-level mathematics, yet their ability to verify natural-language mathematical proofs remains relatively underexplored. This gap is especially pronounced for Olympiad-level proofs, which require meticulous assessment of each claim in a long reasoning chain. Current proof verification thus primarily relies on expert inspection, which is costly to scale. In this paper, we present a systematic study of LLM-based verification in this setting. Specifically, we introduce a human-curated dataset of USAMO 2025 candidate solutions with expert grading, and conduct a broad evaluation of existing LLM verifiers across frontier models. We also propose TROJAN, an iterative self-critique pipeline for generating high-fidelity adversarial proofs at scale. Our comprehensive evaluation covers 20 models spanning seven metrics, two prompt styles, three inference methods, and three
    
[^121]: 超越证据一致性陷阱：面向时序错乱假新闻检测的智能体驱动证据选择

    Beyond the Evidence Coherence Trap: Agent-Driven Evidence Selection for Temporally Disordered Fake News Detection

    [https://arxiv.org/abs/2610.36764](https://arxiv.org/abs/2610.36764)

    该论文针对时序错乱假新闻提出了“证据一致性陷阱”问题，构建了ChronoFake基准并设计了智能体驱动的证据选择方法DelSA，通过选择时间匹配的证据来避免被局部一致但时序错误的证据误导。

    

    假新闻破坏了公众信任和在线信息生态系统的完整性。时序错乱的假新闻尤其难以识别，因为它将真实事件重新排列成虚假的时间顺序。即使事件之间的时间关系被歪曲，单个事件在事实上可能仍然有效。当涉及同一实体的类似事件重复发生时，检索可能会返回来自错误时期的、语义上看似合理的证据。因此，独立的事件验证可能依赖于局部一致但时间上不匹配的证据，从而错误地判定文章为真。我们将这种现象称为“证据一致性陷阱”。由于现有基准很少孤立地针对这一场景，我们构建了ChronoFake，这是一个包含30,706篇文章的源不相交基准，其文章源自具有时间锚定的真实事件。该基准能够区分时间操纵与有效叙述（包括倒叙）。为解决这一问题，我们提出了DelSA，一个文档……（摘要原文在此处截断）

    arXiv:2610.36764v1 Announce Type: new  Abstract: Fake news undermines public trust and the integrity of online information ecosystems. Temporally disordered fake news is especially elusive because it rearranges factual events into a false chronology. Individual events may remain factually valid even when their temporal relations are misrepresented. When similar events involving the same entity recur, retrieval may return semantically plausible evidence from the wrong period. Consequently, independent event verification may rely on locally coherent but temporally mismatched evidence and incorrectly validate the article. We call this the \textit{evidence coherence trap}. Because existing benchmarks rarely isolate this setting, we construct ChronoFake, a source-disjoint benchmark of 30,706 articles derived from temporally anchored real events. It distinguishes temporal manipulation from valid narration, including flashbacks. To address this, we propose DelSA, a \underline{D}ocument S\unde
    
[^122]: 词表已然就绪：将预训练可解释特征适配于生成式推荐

    The Vocabulary Is Already Here: Adapting Pretrained Interpretable Features for Generative Recommendation

    [https://arxiv.org/abs/2610.36763](https://arxiv.org/abs/2610.36763)

    该论文提出API框架，直接复用预训练稀疏自编码器的可解释特征索引作为生成式推荐的语义ID词表，无需额外的词表学习即可实现有效且可解释的推荐。

    

    生成式推荐（GR）通过语义ID（SID）来预测下一个物品。现有方法通常将SID词表拟合到预训练语言模型（PLM）对物品描述的表示上，这带来了额外的训练成本，并使词表覆盖率依赖于拟合的领域。我们观察到，预训练的稀疏自编码器能够揭示具有人类可解释语义标注的PLM特征，使特征索引可以作为可复用的SID词表。然而，复用该词表需要在紧凑的token预算下保持物品之间的区分度，并学习在具有相同主导特征的候选物品中偏向目标物品。这还需要将特征可解释性从物品内容扩展到检索决策。为应对这些挑战，我们提出API框架，通过适配预训练可解释词表，在无需额外词表学习的情况下实现有效且可解释的生成式推荐。

    arXiv:2610.36763v1 Announce Type: cross  Abstract: Generative recommendation (GR) predicts the next item through its semantic ID (SID). Existing methods commonly fit SID vocabularies to pretrained language model (PLM) representations of item descriptions, incurring additional training costs and making vocabulary coverage dependent on the fitting domains. We observe that pretrained sparse autoencoders expose PLM features with human-interpretable semantic annotations, allowing feature indices to serve as a reusable SID vocabulary. However, reusing this vocabulary requires preserving item distinctions under compact token budgets and learning to favor target items over candidates with shared dominant features. It also requires extending feature interpretability from item content to retrieval decisions. To address these challenges, we propose API, a framework that Adapts the Pretrained Interpretable vocabulary for effective and explainable GR without additional vocabulary learning. Within A
    
[^123]: 让自动驾驶实验室可审计：面向混合动作、测量与安全执行的统一框架

    Making Self-Driving Laboratories Auditable:A Unified Framework for Hybrid Actions,Measurements, and Safe Execution

    [https://arxiv.org/abs/2610.36761](https://arxiv.org/abs/2610.36761)

    该论文提出了由LAM、OpenSDL、Safe-SDL和L3-Bench四个组件构成的可追溯统一框架，使自动驾驶实验室能够统一生成混合的机器人动作与仪器API调用、协调异构设备安全执行，并实现全程可审计。

    

    自动驾驶实验室（SDL）是将实验规划、自动化执行与测量反馈以闭环方式耦合在一起的实验室系统。如何将灵活的机器人控制与实验室仪器所暴露的类型化应用程序接口（API）和连续测量整合起来，仍然是一个执行接口层面的挑战。为应对这一挑战，我们提出了一个可追溯的框架，包含四个组件：LAM（实验室动作模型）、OpenSDL、Safe-SDL 和 L3-Bench。LAM 是一种所提出的以测量为条件的策略架构，它扩展了视觉-语言-动作范式，能够同时生成连续的机器人轨迹和类型化的仪器 API 调用。OpenSDL 是一个具备初始控制实现的执行底座，旨在将角色级操作映射到异构设备、协调执行并记录带时间戳的因果事件。所提出的 Safe-SDL 运行时门控结合了调度前的模式……

    arXiv:2610.36761v1 Announce Type: cross  Abstract: Self-driving laboratories (SDLs) are laboratory systems that couple experimental planning, automated execution, and measurement feedback in a closed loop. Integrating flexible robot control with the typed application programming interfaces (APIs) and continuous measurements exposed by laboratory instruments remains an execution-interface challenge. To address this challenge, we propose a traceable framework comprising four components: LAM (Lab Action Model), OpenSDL, Safe-SDL, and L3-Bench. LAM is a proposed measurement-conditioned policy architecture that extends vision-language-action paradigms to generate both continuous robot trajectories and typed instrument API calls. OpenSDL is an execution substrate with an initial control implementation, designed to map role-level operations to heterogeneous devices, coordinate execution, and record timestamped causal events. The proposed Safe-SDL runtime gate combines pre-dispatch schema and 
    
[^124]: 基于内容不变梯度归因的鲁棒大语言模型数据选择

    Robust LLM Data Selection via Content-Invariant Gradient Attribution

    [https://arxiv.org/abs/2610.36755](https://arxiv.org/abs/2610.36755)

    提出CIGA框架，通过筛选对语义保持扰动稳定的梯度层并与语义表示动态融合，消除了梯度归因中的表层形式偏差，实现了更鲁棒的大语言模型指令微调数据选择。

    

    arXiv:2610.36755v1 公告类型： cross 摘要： 基于梯度的数据归因已成为大语言模型（LLM）指令微调中高质量数据选择的重要工具，它通过估计训练样本对模型优化的影响来衡量其贡献。然而，我们发现现有的全局梯度归因方法存在表层形式偏差：即使底层内容保持不变，对语义无关细节（如措辞或格式）的微小改动也会产生高度不一致的归因分数。为解决这一问题，我们提出了内容不变梯度归因，这是一个鲁棒的指令数据选择框架。CIGA首先识别出那些对归因既具有影响力、又在保持语义的表层形式扰动下保持稳定的层，从而降低梯度特征中的格式噪声。随后，它将这些归因信号与语义表示进行动态融合（摘要在此处被截断）。

    arXiv:2610.36755v1 Announce Type: cross  Abstract: Gradient-based data attribution has become an important tool for high-quality data selection in the instruction fine-tuning of large language models (LLMs), as it estimates the contribution of training examples through their influence on model optimization. However, we find that existing global gradient attribution methods suffer from surface-form bias: minor changes to semantically irrelevant details, such as wording or formatting, can produce highly inconsistent attribution scores even when the underlying content remains unchanged. To address this issue, we propose Content-Invariant Gradient Attribution (CIGA), a robust framework for instruction data selection. CIGA first identifies layers whose gradients are both influential for attribution and stable under semantics-preserving surface-form perturbations, thereby reducing formatting noise in gradient features. It then dynamically fuses these attribution signals with semantic represe
    
[^125]: 科学机器学习中时间插值的生成式精炼学习

    Generative Refinement Learning for Temporal Interpolation in Scientific Machine Learning

    [https://arxiv.org/abs/2610.36751](https://arxiv.org/abs/2610.36751)

    提出iFLEX模型，用条件扩散模型学习修正基于时间加权平均的简单插值所产生的残差，从而在低时间分辨率数据下更准确地重建中间科学场。

    

    计算代价高昂的仿真与真实世界观测所提供的数据，其时间分辨率往往低于下游任务的需求。在固定网格点上对观测值取平均可能引入误差，而仅凭运动估计也无法确定物理场数值如何随时间变化。本工作研究了通过学习来修正简单的数值插值，能否改进中间科学场的重建。作者提出的模型——插值流专家模型iFLEX（interpolative FLow EXpert）——首先在每个网格点上对两次观测取加权平均来估计缺失场，其中权重更偏向于距离所请求时刻更近的那次观测。在训练过程中，将这一估计值从观测到的中间场中减去得到残差，并由一个条件扩散模型学习生成该残差。两次观测连同目标时间、观测间隔以及可用的物理参数共同作为模型的条件输入。

    arXiv:2610.36751v1 Announce Type: cross  Abstract: Computationally expensive simulations and real-world observations often provide data at a lower temporal resolution than downstream tasks require. Averaging observations at fixed grid points can introduce errors, and motion estimates alone do not determine how the physical field values change. In this work, we study whether learning to correct a simple numerical interpolation improves the reconstruction of intermediate scientific fields. Our model, interpolative FLow EXpert (iFLEX), first estimates the missing field by taking a weighted average of the two observations at each grid point, giving more weight to the observation closer to the requested time. During training, subtracting this estimate from an observed intermediate field gives the residual that a conditional diffusion model learns to generate. Both observations condition the model, together with the target time, the interval between observations, and available physical param
    
[^126]: 从声纹到用户实体：面向多用户语音代理的在线说话人-记忆构建

    From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents

    [https://arxiv.org/abs/2610.36713](https://arxiv.org/abs/2610.36713)

    该论文提出 EvoSpeakerMem 免训练框架，将声纹视为对身份的动态观察而非固定标签，通过在线说话人-记忆构建随声学证据演进维护持久的用户实体，并构建了 LoCoMo-ID 基准以评估多用户语音代理中的说话人身份归属与记忆管理问题。

    

    个性化语音代理必须在访问用户专属记忆之前识别说话人，然而现有的记忆基准测试通常假设历史记录已被正确归属。我们提出，声纹是身份的观察结果，而非身份本身，并形式化了在线说话人-记忆构建问题，以在声学证据不断演进的过程中维护持久的用户实体。我们引入了 EvoSpeakerMem，这是一个免训练的层，用于实现因果说话人路由、选择性声学状态更新，以及通过可替换的文本记忆后端进行按身份分区的记忆写入与检索。我们还构建了 LoCoMo-ID，这是一个包含隐藏说话人身份、说话人敏感查询、模拟录音条件变化和四说话人身份混淆测试的语音记忆基准。在纵向数据集 SMIIP-TV 上，在固定的接近注册阈值下，真实探测的拒绝率在约一年内从 29.8% 上升到 55.6%。在 LoCoMo-ID 的说话人敏感查询上（摘要在此处被截断）……

    arXiv:2610.36713v1 Announce Type: cross  Abstract: Personalized spoken agents must identify speakers before accessing user-specific memory, yet memory benchmarks typically assume correctly attributed histories. We argue that voiceprints are observations of identity, not identities themselves, and formulate online speaker-memory construction to maintain persistent user entities as acoustic evidence evolves. We introduce EvoSpeakerMem, a training-free layer for causal speaker routing, selective acoustic-state updates, and identity-partitioned memory writes and retrieval through a replaceable text-memory backend. We also construct LoCoMo-ID, a spoken-memory benchmark with hidden speaker identities, speaker-sensitive queries, simulated recording-condition shifts, and four-speaker identity-confusion tests. On longitudinal SMIIP-TV, genuine-probe rejection under a frozen near-enrollment threshold rises from 29.8% to 55.6% over approximately one year. On LoCoMo-ID speaker-sensitive queries, g
    
[^127]: 保守对决老虎机：基线约束下的偏好学习

    Conservative Dueling Bandits: Preference Learning under Baseline Constraints

    [https://arxiv.org/abs/2610.36708](https://arxiv.org/abs/2610.36708)

    本文提出了保守上下文对决老虎机问题及MaxInP-BC算法，通过效用差距而非绝对奖励来认证安全性，克服了标准对决老虎机探索过于激进的问题，并首次证明直接迁移CLUCB的安全测试会导致线性遗憾。

    

    上下文对决老虎机通过成对比较进行学习，这与根据人类偏好微调语言模型所使用的反馈方式相同。标准算法的探索过于激进，可能会向用户展示两个都很糟糕的选项。我们研究了保守的上下文对决老虎机问题，该问题要求在每一轮中，所展示的较优项目的累计得分始终保持在可信基线得分的 (1-α) 比例之上。项目的得分定义为其击败当前最佳可用项目的概率，因此该约束仅取决于效用差异，无需绝对奖励值。这也改变了安全性的认证方式：我们证明，若原封不动地迁移保守老虎机算法 CLUCB 的安全测试，即使基线的效用完全已知，也可能产生线性遗憾。对于线性效用场景，我们开发了 MaxInP-BC 算法，该算法通过效用差距来认证得分，并在必须回退时将基线与一个信息量丰富的配对项目进行组合。

    arXiv:2610.36708v1 Announce Type: cross  Abstract: Contextual dueling bandits learn from pairwise comparisons, the same feedback used to fine-tune language models from human preferences. Standard algorithms explore aggressively and can show users two poor options. We study conservative contextual dueling bandits, which require that, at every round, the cumulative score of the better displayed item stay above a $(1-\alpha)$ fraction of a trusted baseline's. An item's score is its probability of beating the best available item, so the constraint depends only on utility differences and needs no absolute rewards. This also changes how safety can be certified: we show that transferring the safety test of the conservative bandit algorithm CLUCB unchanged can incur linear regret, even when the baseline's utilities are known exactly. For linear utilities, we develop MaxInP-BC, which certifies scores through utility gaps and, when it must fall back, pairs the baseline with an informative partne
    
[^128]: HeteGCD：面向异质图上广义类别发现的统计感知双原型学习

    HeteGCD: Statistics-Aware Dual-Prototype Learning for Generalized Category Discovery on Heterophilous Graphs

    [https://arxiv.org/abs/2610.36696](https://arxiv.org/abs/2610.36696)

    提出HeteGCD方法，通过统计感知的双原型学习，在异质图上有效利用异构邻域信息实现广义类别发现，从而在仅有已知类别标签的条件下同时识别已知类别并发现新类别。

    

    异质图在现实应用中广泛存在，其中相连的节点往往表现出不同的属性或属于不同的类别，这给基于同质性的传统图学习方法带来了重大挑战。这一局限性在广义类别发现（GCD）任务中尤为突出，GCD要求模型仅利用已知类别的标签来识别已知类别并发现新类别。然而，现有的图GCD方法通常依赖上述基于同质性的归纳偏置，这并不适用于异质图。与此同时，尽管当前的异质图学习方法利用了多样的邻域信号，但其选择与融合往往依赖于涵盖所有类别的监督信息，其中包括在GCD中无法获得标签的新类别。这些局限性引出了一个关键问题：如何有效利用异构邻域信息……（原文摘要到此截断）

    arXiv:2610.36696v1 Announce Type: cross  Abstract: Heterophilous graphs are prevalent in real-world applications, where connected nodes often exhibit dissimilar attributes or belong to different classes, posing significant challenges to conventional graph learning methods based on homophily. This limitation becomes particularly pronounced in generalized category discovery (GCD), which requires models to recognize known classes and discover novel ones using labels from known classes only. However, existing graph GCD methods typically rely on aforementioned homophily-based inductive biases, which are ill-suited to heterophilous graphs. Meanwhile, although current heterophilous graph learning methods exploit diverse neighborhood signals, their selection and fusion are often guided by supervision across all classes, including novel classes whose labels are unavailable in GCD. These limitations raise a key question: how can heterogeneous neighborhood information be effectively leveraged for
    
[^129]: TOPS：重塑多任务机器人操作中的流源与传输路径

    TOPS: Reshaping Flow Source and Transport for Multi-Task Robot Manipulation

    [https://arxiv.org/abs/2610.36694](https://arxiv.org/abs/2610.36694)

    提出TOPS方法，在不修改网络架构的前提下，通过为每个任务定制流匹配的源先验和插值调度（利用专家动作逆向映射构建多模态任务源，并依据源-目标协方差失配调整传输节奏），提升语言条件化多任务机器人操作性能。

    

    基于流的策略在语言条件化的多任务操作中，通常在各个任务间共享单一的高斯源先验和共同的线性传输路径。然而，对于共享策略而言，可靠地执行具有不同运动需求的多个任务仍然充满挑战。我们提出了TOPS（任务导向先验与调度，Task-Oriented Priors and Schedules），它无需修改网络架构，即可为每个任务适配流源先验和插值调度。TOPS首先从逆向映射的专家动作序列构建多模态任务源，使语言指令能够选择生成过程的起始位置。在源-目标协方差失配的引导下，该方法在去噪早期减慢向专家动作过渡的速度，并在后期沿需要扩展的方向加速过渡。在高斯端点模型下，我们的分析表明这种调度调整……

    arXiv:2610.36694v1 Announce Type: cross  Abstract: Flow-based policies for language-conditioned multi-task manipulation commonly share a single $\textit{Gaussian}$ source prior and a common linear transport path across tasks. However, reliably executing multiple tasks with distinct motion requirements remains challenging for a shared policy. We introduce $\textbf{TOPS}$ ($\textbf{T}$ask-$\textbf{O}$riented $\textbf{P}$riors and $\textbf{S}$chedules), which adapts the flow source prior and interpolation schedule to each task without modifying the network architecture. TOPS first constructs multimodal task sources from reverse-mapped expert action sequences, allowing language instructions to select where generation begins. Guided by source-target covariance mismatch, it then slows the transition toward expert actions early in denoising and accelerates it later along directions requiring expansion. Under a $\textit{Gaussian}$ endpoint model, our analysis shows that this schedule adjustmen
    
[^130]: PsyJev：基于系统-1判断的心理分析

    PsyJev: Psychological Analysis via System-1 Judgment

    [https://arxiv.org/abs/2610.36689](https://arxiv.org/abs/2610.36689)

    该论文构建了涵盖六个心理学领域的PsyJudgeBench基准，并提出PsyJev系统-1模型，通过联合监督训练与判断锚定的RLCD（JA-RLCD）方法，将异构心理分析任务统一到共享的问题条件化判断接口下，基于Qwen3.5的2B至9B模型取得了64.33至67.56的成绩。

    

    网络上的自我表露内容支持多种心理分析任务，从识别所表达的经历到识别与症状相关的信息。我们针对内容分析中这一异构性问题，研究了一种共享的、以问题为条件的决策接口。我们构建了PsyJudgeBench基准，包含13个标注数据集和涵盖六个心理学领域的82个问题，同时保留了来源特定的评判标准以及对未观测标签的掩码处理。我们提出了PsyJev，一个系统-1（System-1）模型，可直接为每个请求中指定的答案生成概率分布。联合监督训练建立了共享的判断功能，随后采用判断锚定的RLCD（JA-RLCD），该方法通过平衡监督、选择性保留正确的SFT预测以及问题组奖励优化，将logit空间的奖励优化适配到异构的心理判断任务中。基于Qwen3.5的2B、4B和9B模型分别取得了64.33、66.21和67.56的成绩。

    arXiv:2610.36689v1 Announce Type: new  Abstract: Self-disclosures on the Web support diverse psychological analyses, from recognizing expressed experiences to identifying symptom-relevant information. We study a shared, question-conditioned decision interface for this heterogeneous problem in content analysis. We organize PsyJudgeBench, comprising 13 annotated datasets and 82 questions across six psychological areas, while preserving source-specific criteria and masks for unobserved labels. We introduce PsyJev, a System-1 model that directly produces distributions over answers specified in each request. Joint supervised training establishes the shared judgment function, followed by Judgment-Anchored RLCD (JA-RLCD), which adapts logit-space reward optimization to heterogeneous psychological judgments through balanced supervision, selective retention of correct SFT predictions, and question-group reward optimization. The Qwen3.5-based 2B, 4B, and 9B models achieve 64.33, 66.21, and 67.56
    
[^131]: 儿童尺度语言模型中的专业化、多样性与规模化

    Specialization, Diversity, and Scaling in Child-Scale Language Models

    [https://arxiv.org/abs/2610.36679](https://arxiv.org/abs/2610.36679)

    在儿童尺度数据上训练的小型语言模型会对单个儿童的语言环境产生词汇和分布层面的专业化，多儿童数据训练虽能提升对汇总语言的预测，但对语法能力的提升并不明显。

    

    现代语言模型训练所用的语言数据远超儿童实际接触的语言量，但我们对从小型自然语言数据集中学习的过程知之甚少。我们在来自密集家庭录音的儿童语音转录文本和合成的儿童导向对话数据上训练小型GPT-2模型，并改变训练数据的数量、来源和多样性。在单个儿童语言环境数据上训练的模型，对该环境的预测效果显著优于对其他儿童环境的预测。这种专业化很大程度上体现在词汇和分布层面：一个简单的5-gram模型就能重现这种跨儿童的模式（r=.91）。与仅在单个儿童输入上训练相比，在多个儿童的数据上训练能明显提升对汇总保留语言的预测，但在语法基准测试上却没有出现类似的明显提升。在不同的数据规模下，自然数据和合成训练数据也会产生不同的语法和词汇表现。

    arXiv:2610.36679v1 Announce Type: cross  Abstract: Modern language models are trained on far more language than children experience, but we know much less about learning from small, naturalistic datasets. We train small GPT-2 models on transcripts of speech with children from dense home recordings (BabyView) and on synthetic child-directed dialogue (TinyDialogues), varying the amount, source, and diversity of training data. Models trained on one child's language environment predict that environment substantially better than other children's environments. Much of this specialization is lexical and distributional: a simple 5-gram model reproduces the cross-child pattern (r=.91). Training on data from multiple children clearly improves prediction of pooled held-out language relative to training on a single child's input without a similarly clear improvement on grammatical benchmarks. Across data sizes, naturalistic and synthetic training inputs also produce different grammatical and lexic
    
[^132]: 原子作为智能体：集体预测编码产生的涌现秩序

    Atom as an Agent: Emergent Order by Collective Predictive Coding

    [https://arxiv.org/abs/2610.36673](https://arxiv.org/abs/2610.36673)

    该论文将语言符号涌现的“集体预测编码”理论引入物理学，把原子视为相互预测的智能体，仅凭观测到的自旋构型就能自动涌现出伊辛模型、条纹序模型和自旋玻璃系统中的宏观序参量，为宏观秩序的起源提供了新解释。

    

    arXiv:2610.36673v1 公告类型：交叉。摘要：宏观序参量从微观系统中涌现的内在原因是什么？本文将集体预测编码（CPC）——一种关于语言/符号涌现的理论——应用于物理学。正如人类语言从人际交流中涌现一样，物理系统的宏观序参量也从原子之间的“交流”中涌现出来——每个原子被视为一个智能体，通过推断来预测彼此的行为。我们给出了三个经典自旋系统的例子：二维铁磁和反铁磁伊辛模型、具有竞争条纹序的二维自旋模型，以及全连接的Sherrington–Kirkpatrick自旋玻璃模型。在所有这些例子中，包括副本重叠在内的恰当宏观序参量，仅凭观测到的自旋构型，便能通过CPC的通信信息学自动涌现。我们将学习到的宏观坐标称为集体预……（原文此处截断）

    arXiv:2610.36673v1 Announce Type: cross  Abstract: What is the intrinsic reason of emergence of macroscopic order parameter out of microscopic systems? In this article, we apply to physics the collective predictive coding (CPC), the theory of language/symbol emergence. As our language emerges out of our human communications, physical macroscopic order parameter emerges out of communications among atoms --- each atom is regarded as an agent who predicts each other's behavior through inference. We present three examples of classical spin systems: two-dimensional ferromagnetic and antiferromagnetic Ising models, a two-dimensional spin model with competing stripe orders, and the fully connected Sherrington--Kirkpatrick spin-glass model. In all of these examples, proper macroscopic order parameters including the replica overlap emerge automatically through communication informatics of CPC, only from observed spin configurations. We call the learned macroscopic coordinate the collective pred
    
[^133]: 学习信任何处：面向三维可变形配准的可靠性引导专家路由与速度门控

    Learning Where to Trust: Reliability-Guided Expert Routing and Velocity Gating for 3D Deformable Registration

    [https://arxiv.org/abs/2610.36672](https://arxiv.org/abs/2610.36672)

    提出ReliMorph框架，利用局部相关分布评估匹配可靠性，并以可靠性线索引导空间专家融合和残差速度门控，从而提升三维可变形图像配准在弱纹理、严重形变区域的准确性。

    

    可变形图像配准依赖于精确的解剖结构对应关系。然而，在基于学习的配准方法中，将局部匹配可靠性与空间上下文选择及形变更新强度进行显式耦合的研究仍然不足。这一挑战在均质、弱纹理和严重形变的区域尤为突出，在这些区域中，模糊的匹配可能导致错误的形变更新。我们提出了ReliMorph，一个用于三维可变形配准的可靠性路由对称框架，它将对应关系评估与空间上下文选择及更新强度相耦合。局部相关分布利用具有模态特定适配的共享解剖特征来刻画对应关系的模糊性。这些可靠性线索引导空间专家融合，并在由粗到细的精细化过程中对残差速度进行门控。更新后的逆不一致性反馈在更细的金字塔层级上指导路由与门控……

    arXiv:2610.36672v1 Announce Type: new  Abstract: Deformable image registration relies on accurate anatomical correspondences. However, explicitly coupling local matching reliability with spatial context selection and deformation update strength remains insufficiently explored in learning-based registration. This challenge is particularly pronounced in homogeneous, weakly textured, and severely deformed regions, where ambiguous matches may induce erroneous deformation updates. We propose ReliMorph, a reliability-routed symmetric framework for 3D deformable registration that couples correspondence assessment with spatial context selection and update strength.Local correlation distributions characterize correspondence ambiguity using shared anatomical features with modality-specific adaptation. These reliability cues guide spatial expert fusion and gate residual velocities during coarse-to-fine refinement. Updated inverse-inconsistency feedback guides routing and gating at finer pyramid l
    
[^134]: 仅在需要时查看：面向高效扩散视觉语言模型的证书引导视觉路由

    Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs

    [https://arxiv.org/abs/2610.36671](https://arxiv.org/abs/2610.36671)

    提出LOWN解码方法，在每个解码步骤自适应地决定是否需要处理图像以及解码多少标记，从而避免扩散视觉语言模型中大量冗余的图像重复计算，显著提升解码效率。

    

    扩散视觉语言模型在每个去噪步骤都会处理其全部输入。在我们研究的基准测试中，视觉标记占输入的89%-97%，因此大部分解码计算都耗费在重复处理同一张图像上。视觉标记剪枝方法通过移除部分视觉标记在一定程度上缓解了这一负担，但会丢失被移除的信息，且仍需在每一步处理剩余图像。为了衡量图像实际被需要的频率，我们在每个解码步骤比较LLaDA-V在有图像和没有图像情况下的预测，发现移除图像平均只在22.8%的步骤中改变被解码（unmask）的标记，最低甚至仅为7.9%。因此，我们提出LOWN，一种在每个步骤决定是否处理图像以及解码多少标记的解码方法。LOWN在不使用图像的情况下预测被掩码的标记，仅当该预测与上一次的……（摘要在此处截断）

    arXiv:2610.36671v1 Announce Type: cross  Abstract: Diffusion vision-language models (DVLMs) process their entire input at every denoising step. Since visual tokens make up 89-97% of this input on the benchmarks we study, most of the decoding computation is spent repeatedly processing the same image. Visual token pruning methods partially relieve this burden by removing part of the visual tokens, but they lose the removed information and still process the remaining image at every step. To measure how often the image is actually needed, we compare LLaDA-V's predictions with and without the image at every decoding step and find that removing the image changes a token being unmasked at only 22.8% of steps on average and at as few as 7.9%. We therefore propose LOWN, a decoding method that decides at every step whether to process the image and how many tokens to unmask. LOWN predicts the masked tokens without the image and processes the image only when this prediction disagrees with the last
    
[^135]: 前沿大语言模型中阿希从众效应的定量研究方法

    A Quantitative Approach to Asch Conformity in Frontier Large Language Models

    [https://arxiv.org/abs/2610.36668](https://arxiv.org/abs/2610.36668)

    该论文提出一种定量方法，通过对单个问题重复采样获得答案分布，并比较模型接触人为制造共识前后的分布差异，证实前沿大语言模型存在从众效应，并量化了从众发生的条件与程度。

    

    阿希从众实验最近作为衡量大语言模型（LLM）从众性的一种方法受到了广泛关注。虽然以往的研究主要集中于证明LLM中存在从众现象，但我们的工作为这一方法增加了更进一步的定量层面。我们的方法不是在一组问题上对前沿LLM进行采样，而是对单个问题进行重复采样，利用LLM的概率特性产生答案的分布。通过这种方式，我们可以比较模型在被展示人为制造的共识之前和之后的答案分布。我们的实验证实了前沿LLM确实存在从众行为，这强化了以往文献的结论，并且通过我们的分布间比较，提供了关于导致从众的条件以及从众发生程度的额外定量数据。

    arXiv:2610.36668v1 Announce Type: cross  Abstract: The Asch conformity experiment has recently gained considerable attention as a way of measuring conformity in large language models (LLMs). While previous studies have focused on demonstrating conformity in LLMs, our work adds a further quantitative layer to this approach. Instead of sampling a frontier LLM on a suite of questions, our approach samples a single question repeatedly, leveraging the probabilistic nature of LLMs to produce a distribution of answers. In doing so, we can compare the distributions of answers before and after the model is presented with the manufactured consensus. Our experiments confirm that frontier LLMs do conform, reinforcing prior literature, and, through our distribution-to-distribution comparisons, provide additional quantitative data on the conditions that lead to conformity and the degree to which it occurs.
    
[^136]: 无处之观：以种群为中心的智能体演化

    The View from Nowhere: Population-Centered Agent Evolution

    [https://arxiv.org/abs/2610.36660](https://arxiv.org/abs/2610.36660)

    提出种群中心演化（PCE），通过利用整个种群的任务记录与自我修改历史来改进智能体，从而弥合种群覆盖率与最佳个体性能之间的差距。

    

    自我改进的AI系统能够修改支配其自身行为的设计。然而，现有方法将改进集中在被选中的父代智能体及其演化经验上，使得更广泛的种群在很大程度上未被充分利用。我们发现种群覆盖率（即至少被一个演化变体解决的任务）与最佳个体智能体的性能之间存在显著差距。这一差距表明，有用的能力和诊断证据在种群中广泛涌现，却未被整合进领先的智能体之中。这些观察催生了“无处之观”（the view from nowhere）这一智能体演化原则：应当通过种群范围内的成功、失败及先前自我修改的模式来解读智能体的行为，从而对其进行改进。我们将这一原则具体化为种群中心演化（Population-Centered Evolution, PCE）。PCE 将种群中的任务记录和修改历史提炼为一个种群块，该块捕获……（原文摘要在此处截断）

    arXiv:2610.36660v1 Announce Type: new  Abstract: Self-improving AI systems can revise the designs that govern their own behavior. Yet existing approaches center improvement on selected parent agents and their evolutionary experience, leaving the broader population largely underused. We identify a substantial gap between population coverage (i.e., the tasks solved by at least one evolved variant) and the performance of the best individual agent. This gap shows that useful capabilities and diagnostic evidence emerge across the population without being consolidated into its leading agents.   These observations motivate the view from nowhere, a principle for agent evolution: an agent should be improved by interpreting its behavior through population-wide patterns of success, failure, and prior self-modification. We instantiate this principle as Population-Centered Evolution (PCE). PCE distils task records and modification histories from the population into a population block, which capture
    
[^137]: 记忆还不够：在多智能体系统中精炼记忆并学习何时使用记忆

    Remembering Is Not Enough: Refining Memories and Learning When to Use Them in Multi-Agent Systems

    [https://arxiv.org/abs/2610.36652](https://arxiv.org/abs/2610.36652)

    提出无需训练的DiscernMem框架，通过缺陷感知记忆精炼（DMR）诊断并修复有缺陷的记忆，通过边界学习元记忆（BLM）学习在何种情境下应使用或避免使用记忆，从而同时提升多智能体系统的记忆质量与记忆使用效果。

    

    记忆可以帮助基于大语言模型的多智能体系统复用经验，但存储的建议可能存在缺陷，或者在不恰当的情境中被使用。仅凭团队层面的成功或失败，无法可靠地判断某条记忆是否需要修正，或是否本不应被使用。我们提出了DiscernMem，这是一个无需训练的框架，能够同时提升记忆质量与记忆使用。为提升记忆质量，缺陷感知记忆精炼（DMR）将使用单条记忆与不使用记忆的团队执行结果进行对比，诊断规则缺陷，并在当前及以往任务上测试修复效果，以避免丢失先前有用的行为。为引导恰当使用，边界学习元记忆（BLM）记录每条记忆的过往使用情况及其效果，学习其所属角色在何种任务情境和接收消息下应当应用或避免使用该记忆。这些组件共同帮助团队纠正有缺陷的经验，同时保留那些被错误使用的有用规则。基于Qwen3-4B-I的实验（摘要原文在此处截断）。

    arXiv:2610.36652v1 Announce Type: new  Abstract: Memory helps LLM-based multi-agent systems reuse experience, but stored advice can be flawed or applied in the wrong situation. Team-level success or failure alone cannot reliably show whether a memory needs correction or should not have been used. We propose DiscernMem, a training-free framework that improves both memory quality and memory use. To improve quality, Defect-Aware Memory Refinement (DMR) compares team executions with a single memory and with no memory, diagnoses rule defects, and tests repairs on current and earlier tasks to guard against losing previously useful behavior. To guide appropriate use, Boundary-Learning Meta-Memory (BLM) records each memory's past uses and effects, learning when its owner role should apply or avoid it based on task context and incoming messages. Together, these components help teams correct flawed experience while retaining useful rules that were applied incorrectly. Experiments with Qwen3-4B-I
    
[^138]: Tap2Listen：面向流式音视频说话人提取的即时目标切换

    Tap2Listen: Instant Target Switching for Streaming Audio-Visual Speaker Extraction

    [https://arxiv.org/abs/2610.36643](https://arxiv.org/abs/2610.36643)

    提出Tap2Listen流式音视频说话人提取模型，通过按模态分离计算架构，将目标切换开销降至仅需0.12ms的重投影，实现了对话场景下的即时目标切换。

    

    流式音视频目标说话人提取（AV-TSE）利用目标说话人的面部信息，从输入的混合音频中实时提取该说话人的语音。然而，现有系统在整个音频片段中固定目标说话人，而对话场景需要频繁且即时地进行目标切换。为解决这一问题，我们提出了Tap2Listen，一个专为即时响应目标切换而设计的流式AV-TSE模型。该模型按模态分离计算：与目标无关的音频编码器仅处理混合音频，而被选中目标面部的视觉线索仅对解码器进行条件化控制。一次切换仅需0.12ms的重投影即可立即生效。我们还设计了组件级的损失函数：编码器的预测损失和解码器的监督提取损失。在评估方面，我们衡量切换后输出质量恢复的速度。在标准的LRS3-2mix测试集上，Tap2Listen提升了SI（摘要在此处截断）

    arXiv:2610.36643v1 Announce Type: cross  Abstract: Streaming audio-visual target speaker extraction (AV-TSE) uses a target speaker's face to extract the speaker's speech from an input audio mixture in real time. However, existing systems fix the target speaker for an entire clip, while conversation demands frequent and immediate target switches. To address this problem, we propose Tap2Listen, a streaming AV-TSE model designed for instant response to target switches. The model separates computation by modality: the target-agnostic audio encoder processes only the mixture, while the target-specific visual cues of the selected face condition only the decoder. A switch requires only a 0.12ms re-projection and takes effect immediately. We also design component-wise losses: a prediction loss for the encoder and a supervised extraction loss for the decoder. For evaluation, we measure how quickly output quality recovers after a switch. On the standard LRS3-2mix test set, Tap2Listen improves SI
    
[^139]: AI生成网络内容的不均衡崛起

    The uneven rise of AI-generated web content

    [https://arxiv.org/abs/2610.36631](https://arxiv.org/abs/2610.36631)

    基于500万网页的大规模研究表明，AI生成文本在互联网上急剧增长但分布不均：英语网页和.com域名中AI内容占比更高，而网络图中最边缘化的网页AI内容增长最快，成为整体增长的主要驱动力。

    

    在不到五年的时间里，生成式人工智能（AI）已经极大地重塑了面向公众的互联网。基于超过500万个可抓取网页的样本，我们证明了包含AI生成文本的网页比例急剧上升，但增长并不均衡。我们的分析揭示了AI内容占比在语言和域名类别（如.edu、.com或.gov）之间存在显著差异。与其他语言的网页相比，含有AI生成文本的英语网页比例更高；与.com域名相比，.gov和.edu域名上AI生成文本的普及率明显更低。即使在考虑了构成效应之后，这些差异依然存在。我们还发现了基于网页在互联网连接结构中所处位置的异质性。AI生成文本在网络图中最不中心的网页上最为常见且增长最为迅速，这推动了AI生成文本整体增长的很大一部分。

    arXiv:2610.36631v1 Announce Type: new  Abstract: In less than five years, generative artificial intelligence (AI) has dramatically reshaped the public-facing web. Using a sample of more than 5 million crawlable webpages, we demonstrate that the share containing AI-generated text has increased precipitously, but unevenly. Our analyses reveal substantial differences in AI composition across language and domain categories (e.g., .edu, .com, or .gov). A higher fraction of English-language webpages contain AI-generated text compared to pages in other languages, and prevalence is substantially lower on .gov and .edu domains compared to .com. These distinctions persist even after accounting for compositional effects. We also find heterogeneity based on a page's position in the connectivity structure of the web. AI-generated text is most common and has increased most rapidly in pages that are the least central in the graph, which drives much of the overall growth in AI-generated text. A slower
    
[^140]: 基于事件相机的航天器位姿估计中的极性-时间交互与多级可靠性

    Polarity--Time Interaction and Multi-Level Reliability for Event-Based Spacecraft Pose Estimation

    [https://arxiv.org/abs/2610.36625](https://arxiv.org/abs/2610.36625)

    该论文提出一个可靠性感知的航天器位姿估计框架，通过极性-时间三支路事件表示、有序语义点精度估计和序列级观测尺度设定，将事件定位、几何恢复与CAD细化相结合，在SPADES挑战赛上实现了更鲁棒准确的位姿估计。

    

    稀疏的事件响应会产生不均匀的关键点误差，其对航天器位姿的影响取决于对应几何关系以及连续观测的可靠性。我们提出了一个可靠性感知框架，将极性-时间定位、几何恢复、事件-CAD细化与轨迹优化联系起来。三支路表示通过跨分支注意力与门控机制结合正事件、负事件以及近期事件的时间信息，随后进行时空聚合。有序语义点头部估计相对对应精度，用于几何约束下的位姿恢复。事件边缘与CAD投影的一致性随后在关键点几何保障下引导局部位姿细化。在序列层面，一个十维描述子结合热图形状、重投影一致性和几何调节来设定时间对齐的观测尺度。在官方SPADES挑战赛测试集上，该（摘要在此处截断）

    arXiv:2610.36625v1 Announce Type: new  Abstract: Sparse event responses produce uneven keypoint errors whose effects on spacecraft pose depend on correspondence geometry and the reliability of successive observations. We present a reliability-aware framework connecting polarity--time localization, geometric recovery, event--CAD refinement, and trajectory optimization. A three-branch representation combines positive events, negative events, and recent-event timing through cross-branch attention and gating, followed by spatiotemporal aggregation. An ordered semantic-point head estimates relative correspondence precision for geometrically constrained pose recovery. Event-edge and CAD-projection agreement then guide local pose refinement under keypoint-geometry safeguards. At the sequence level, a ten-dimensional descriptor combines heatmap shape, reprojection consistency, and geometric conditioning to set time-aligned observation scales. On the official SPADES challenge test set, the comp
    
[^141]: 通过运动引导的非对称预测将视频模型适配于机器人控制

    Adapting Video Models for Robot Control via Motion-guided Asymmetric Prediction

    [https://arxiv.org/abs/2610.36624](https://arxiv.org/abs/2610.36624)

    该论文提出运动引导非对称预测（MoGAP），通过在动态区域分配更高噪声、并以低损坏的静态区域作为上下文进行非对称视觉预测，显著提升了世界动作模型在机器人控制中的策略性能。

    

    世界动作模型通过联合预测未来视觉观测和动作，利用预训练的视频生成模型来实现机器人控制。视觉预测目标在将预训练视频表征适配到目标机器人领域的过程中起着关键作用。然而，继承自视频预训练的标准视觉预测对所有视觉区域一视同仁，尽管它们与机器人控制的相关性各不相同。因此，我们研究了世界动作模型中视觉监督的空间分配问题，并表明将监督集中在动态区域可以显著提升策略性能。基于这一发现，我们提出了运动引导非对称预测，它通过以运动为中心的视觉预测来增强世界动作模型的训练。MoGAP在不同区域之间非对称地分配噪声，训练模型利用损坏程度较低的静态区域作为上下文，来重建损坏程度更重的动态区域。

    arXiv:2610.36624v1 Announce Type: cross  Abstract: World Action Models (WAMs) leverage pretrained video generative models for robotic control by jointly predicting future visual observations and actions. The visual prediction objective plays a key role in adapting pretrained video representations to the target robot domain. Yet standard visual prediction inherited from video pretraining treats all visual regions uniformly, despite their differing relevance to robotic control. We therefore investigate the spatial allocation of visual supervision in WAMs and show that focusing supervision on dynamic regions can substantially improve policy performance. Building on this finding, we introduce Motion-Guided Asymmetric Prediction (MoGAP), which augments WAM training with motion-focused visual prediction. MoGAP allocates noise asymmetrically across regions, training the model to reconstruct more heavily corrupted dynamic regions using less corrupted static regions as context. MoGAP is trained
    
[^142]: 上帝不掷骰子：在智能体式内核生成中强制执行批处理不变性

    God Does Not Play Dice: Enforcing Batch Invariance in Agentic Kernel Generation

    [https://arxiv.org/abs/2610.36623](https://arxiv.org/abs/2610.36623)

    该论文提出了一种即插即用工具，通过将契约引导的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合，作为验收门控嵌入内核生成智能体的优化循环中，从而强制执行批处理不变性，解决LLM服务中动态批处理导致相同输入产生不同数值输出、损害可复现性的问题。

    

    大语言模型（LLM）服务系统会动态地将用户请求分批以进行GPU执行，因此同一请求在不同运行中可能会与不同的请求一起被处理。如果GPU内核缺乏批处理不变性，这种不断变化的批处理上下文可能会改变相同输入的数值输出，从而损害可复现性。现有的内核生成智能体专注于正确性和效率，却忽视了批处理不变性。我们在83个适用于批处理的已发布产物中发现了19个经确认的批处理不变性违规。为解决这一局限，我们提出了一个工具，它将基于契约的静态依赖分析与跨批处理上下文的按位一致性动态验证相结合。该工具被设计为通用内核智能体的即插即用组件，能够在复用其原有优化能力的同时增加批处理不变性强制执行。我们将其作为优化循环中的连续验收门控进行集成，拒绝结构上有风险或动态验证不通过的内核（摘要原文在此处截断）。

    arXiv:2610.36623v1 Announce Type: new  Abstract: Large language model (LLM) serving systems dynamically batch user requests for GPU execution, so the same request may be processed alongside different requests across runs. If the GPU kernels lack batch invariance (BI), this changing batch context can alter the numerical output for an unchanged input, undermining reproducibility. Existing kernel generation agents focus on correctness and efficiency but \emph{overlook BI}. We find 19 confirmed BI violations among 83 batch-applicable released artifacts. To address this limitation, we propose \tool, which combines contract-guided static dependency analysis with dynamic validation of bitwise consistency across batch contexts. \tool is designed as a plug-in component for general kernel agents, allowing their original optimization capabilities to be reused while adding BI enforcement. We integrate it as successive acceptance gates in the optimization loop, rejecting structurally risky or dynam
    
[^143]: Dream-COP：在经验世界中引导组合优化问题的算法发现

    Dream-COP: Guiding Algorithm Discovery for Combinatorial Optimization Problems in an Experience World

    [https://arxiv.org/abs/2610.36618](https://arxiv.org/abs/2610.36618)

    Dream-COP 通过让策略智能体在由历史搜索轨迹构成的经验世界中“做梦回放”来学习可执行策略，从而在有限搜索预算下引导 LLM 为组合优化问题发现更好的算法。

    

    大型语言模型（LLM）通过迭代的程序生成与评估，能够为组合优化问题（COP）进行算法发现。其核心挑战在于如何组织搜索，以便在有限预算内发现更好的算法。我们的关键洞察是：过去的搜索轨迹及其记录的结果构成了一个“经验世界”，可以通过回放来辅助求解组合优化问题。我们提出了 Dream-COP，其中“策略智能体”通过在这个经验世界中“做梦”来学习可执行的策略，以引导“发现智能体”生成并改进组合优化问题的程序。具体而言，策略回放能够对所发现程序的质量以及所使用的搜索步骤提供反馈，从而更有效地指导策略选择。在五个组合优化问题上，Dream 在启发式组件设计和完整求解器生成两个方面都发现了优于固定探索的更好算法。

    arXiv:2610.36618v1 Announce Type: cross  Abstract: Large language models (LLMs) enable algorithm discovery for combinatorial optimization problems (COPs) through iterative program generation and evaluation. A central challenge is how to organize search to discover better algorithms within a limited budget. Our key insight is that past search trajectories and their recorded outcomes form an experience world, which can be used to facilitate the COP solving through replay. We introduce Dream-COP, in which a \emph{policy agent} learns executable policies through \emph{dreaming} in this experience world to guide a \emph{discovery agent} in generating and refining COP programs. Specifically, policy replay provides feedback on the quality of the programs revealed and the search steps used, to guide policy selection more effectively. Across five COPs, \textsc{Dream} discovers better algorithms than fixed exploration in both heuristic component design and complete solver generation. Particularl
    
[^144]: 面向 Mamba-2 状态压缩的平衡状态子空间

    Balanced State Subspaces for Mamba-2 State Compression

    [https://arxiv.org/abs/2610.36617](https://arxiv.org/abs/2610.36617)

    论文发现 Mamba-2 的状态能量集中于递归累积的偏置方向，而输出实际读取的内容位于与其坐标不对齐的低维子空间，据此提出一种基于完整 Gramian 平衡截断的训练后压缩方法，使压缩直接针对写入-读取子空间而非单个坐标。

    

    Mamba-2 将所有上下文压缩到一个固定大小的循环状态中，该状态主导了解码阶段的内存占用和带宽。现有的压缩器对状态值进行量化或对状态坐标进行剪枝，并在原生基底下独立地为每个通道打分。我们检查了训练后的 Mamba-2 状态，发现其能量几乎是一秩的，但输出并不会读取这些高能量方向：能量集中在由递归累积的类似偏置的方向上，而输出所读取的内容则位于一个与状态坐标不对齐的低维子空间中。即使是输出感知的评分方法也是逐坐标计算的，因此会错过决定输入-输出映射的那些方向。因此，压缩应当针对写入-读取子空间而非单个坐标。我们提出了 \method{}，这是一种训练后变换，它利用完整的可控性与可观测性 Gramian 矩阵实现平衡截断，并应用……（原文摘要在此处被截断）

    arXiv:2610.36617v1 Announce Type: cross  Abstract: Mamba-2 compresses all context into a fixed-size recurrent state that dominates decoding memory and bandwidth. Existing compressors quantize state values or prune state coordinates, scoring each channel independently in the native basis. We examine trained Mamba-2 states and find that energy is almost rank-one, but the output does not read the energetic directions: energy concentrates in a bias-like direction accumulated by the recurrence, while the content the output reads lives in a low-dimensional subspace that is not aligned with the state's coordinates. Even output-aware scores are computed coordinate by coordinate, so they miss the directions that determine the input--output map. Compression should therefore target the write--read subspace rather than individual coordinates. We propose \method{}, a post-training transform that instantiates balanced truncation with full controllability and observability Gramians and applies the ro
    
[^145]: 学习形式化：连接开放式工作负载与快速决策模型

    Learning to Formalize: Bridging Open-Ended Workloads and Fast Decision Models

    [https://arxiv.org/abs/2610.36595](https://arxiv.org/abs/2610.36595)

    提出一个在离线教师生成列表上训练的小型候选形式化器，将开放式任务上下文转化为显式且经验证的候选项，使冻结的快速决策模型能够取代缓慢的生成式大模型后端完成高效选择。

    

    决策模型能够在提供的候选项之间快速做出选择，而通用大型语言模型则通过灵活的生成与推理来解读开放式任务，但代价是顺序解码带来的高延迟。许多工作负载期望由生成式后端来确定可以执行哪些操作，因此并未提供明确定义的候选列表。我们通过一个在离线教师生成的列表上训练的小型候选形式化器来弥合这一接口差距。该形式化器从任务上下文中构建显式的、经过验证的备选方案，再由冻结的决策模型从中选择一个条目。在一项动机性对比实验中，在200个CALVIN-D测试回合上与官方TypeSafe Jev-1.13.0进行比较，使用通用动作的成功率为11%，而使用DeepSeek V4 Flash位姿候选的成功率为46.5%。此外，在1,011个CALVIN-D验证标注上进行的一项六臂特权状态评估发现，本地部署的决策模型最多可达到52.6%的任务成功率

    arXiv:2610.36595v1 Announce Type: new  Abstract: Decision models offer fast choices over sup- plied alternatives, while general-purpose large language models interpret open-ended tasks through flexible generation and reasoning at the cost of sequential decoding. Many work- loads expect a generative backend to determine what can be done and therefore provide no well-defined candidate list. We bridge this in- terface gap with a small candidate formalizer trained on offline teacher-generated lists. The formalizer constructs explicit, validated alter- natives from task context, and a frozen deci- sion model selects an entry. A motivating com- parison with official TypeSafe Jev-1.13.0 on 200 CALVIN-D test episodes reports 11% suc- cess with generic motions versus 46.5% with DeepSeek V4 Flash pose candidates. Sepa- rately, a six-arm privileged-state evaluation on 1,011 CALVIN-D validation annotations finds that a locally deployed decision model can achieve at most 52.6% task success rate wit
    
[^146]: InfraDev：编码智能体能开发AI基础设施吗？我们又该如何评判？

    InfraDev: Can Coding Agents Develop AI Infrastructure, and How Can We Tell?

    [https://arxiv.org/abs/2610.36589](https://arxiv.org/abs/2610.36589)

    InfraDev是一个基于Megatron-LM、vLLM和verl真实代码变更构建的36任务基准，用于评估编码智能体开发多文件、多GPU的AI基础设施代码的能力，并通过隐藏探针和共享执行环境实现更准确高效的评分。

    

    编码智能体已开始参与开发现代AI基础设施的生产级代码——这些基础设施负责训练和服务大型模型，然而现有基准测试仅覆盖单个内核或单GPU任务，且其评分所用的测试与提示会同时误判正确的补丁和错误的补丁。我们提出了InfraDev，这是一个包含36个任务的基准测试，基于针对Megatron-LM、vLLM和verl的真实pull request构建，涵盖训练、推理和强化学习的基础设施开发。每个任务要求智能体重现一个跨越多个文件、且必须在多个GPU上正确运行的真实变更。为了给这类变更评分，我们联合合成了隐藏探针与提示词：隐藏探针用于强制检查原始测试未覆盖的需求，提示词则声明探针所调用的每个接口但不泄露解决方案。我们还设计了一个共享执行环境，可在任务之间复用依赖项和构建产物，将存储占用降低52-57%，构建时间降低13-

    arXiv:2610.36589v1 Announce Type: new  Abstract: Coding agents are beginning to develop the production-grade AI infrastructure that trains and serves large models, yet existing benchmarks cover only single kernels or single-GPU tasks, and grade with tests and prompts that misjudge both correct and incorrect patches. We introduce InfraDev, a benchmark of 36 tasks built from real-world pull requests to Megatron-LM, vLLM, and verl, covering infrastructure development for training, inference, and reinforcement learning. Each task asks an agent to reimplement a real change that spans many files and must run correctly across many GPUs. To grade such changes, we jointly synthesize hidden probes, which enforce the requirements the original tests leave unchecked, and a prompt that states every interface the probes call without revealing the solution. We also design a shared execution environment that reuses dependencies and artifacts across tasks, cutting storage by 52-57% and build time by 13-
    
[^147]: 面向基于软件需求的行为充分性测试生成的LLM智能体基准评测

    Benchmarking LLM Agents for Behaviorally Adequate Test Generation from Software Requirements

    [https://arxiv.org/abs/2610.36584](https://arxiv.org/abs/2610.36584)

    该论文提出了Req2Test-Bench基准，包含从8个开源项目BDD工件中恢复的5,086个经执行验证的实例，用于评测LLM智能体在仅依据自然语言需求（无实现代码）的情况下生成行为充分测试的能力。

    

    随着大型语言模型（LLM）智能体日益将自然语言需求转化为仓库级别的代码，“从需求生成测试”变得愈发重要，因为它能够为代码生成提供验证约束（即测试脚手架/harness）。然而，目前尚不清楚现有的智能体能否在对应实现代码被隐藏的情况下，生成能够验证所需行为的测试。现有的测试生成基准大多聚焦于白盒测试，而需求—测试相关的研究通常需要对给定文档与所生成测试之间的语义对齐程度进行评估。我们提出了Req2Test-Bench，这是一个包含5,086个经执行验证实例的基准数据集，这些实例从8个开源项目中开发者编写的行为驱动开发（BDD）工件中恢复而来，涵盖5种编程语言和4个BDD框架。每个实例将Gherkin格式的自然语言需求与开发者编写的测试（配对）……（摘要原文在此处截断）

    arXiv:2610.36584v1 Announce Type: cross  Abstract: As large language model agents increasingly compile natural-language requirements into repository-level code, requirement-to-test generation grows in importance to provide validation constraints (or harness) to guide code generation. Yet it remains unclear whether current agents can generate tests that validate the required behavior when the implementation of that behavior is withheld. Existing test-generation benchmarks mostly focus on white-box testing, whereas the requirement--test-related study usually requires evaluation on semantic alignment between the given documents and the generated tests.   We introduce Req2Test-Bench, a benchmark containing 5,086 execution-validated instances recovered from developer-authored behavior-driven development (BDD) artifacts in 8 open-source projects, spanning 5 programming languages and 4 BDD frameworks. Each instance pairs a natural-language requirement in Gherkin with its developer-written Tes
    
[^148]: MM-Net：用于缺血性卒中后睡眠分期与睡眠呼吸暂停评估的多模态多任务网络

    MM-Net: A Multimodal Multi-Task Network for Sleep Staging and Sleep Apnea Assessment After Ischemic Stroke

    [https://arxiv.org/abs/2610.36583](https://arxiv.org/abs/2610.36583)

    提出多模态多任务网络MM-Net，通过交叉拟合和基于岭回归的嵌入对齐解决表征不一致问题，利用交叉注意力融合EEG-EOG表征、神经描述符与心肺特征，实现缺血性卒中后睡眠分期与呼吸事件检测的同步联合评估。

    

    缺血性卒中后的自动多导睡眠图分析面临挑战，其原因在于睡眠相关神经活动的改变以及并发的呼吸障碍。现有的针对卒中的方法主要解决睡眠分期分类问题，而对睡眠结构和呼吸事件的联合分析尚未得到充分探索。此外，在适配预训练EEG编码器时，当下游模型在已见患者的嵌入上训练、却在未见过的个体上进行评估时，会引入表征不一致问题。我们提出了MM-Net，这是一个多模态多任务框架，通过交叉拟合和基于岭回归的嵌入对齐来解决这种不一致性。睡眠适应的EEG-EOG表征和工程化神经描述符通过交叉注意力与共享时间建模和学习到的心肺特征相融合，从而同时实现睡眠分期和呼吸事件检测。在iSLEEPS数据集（99名亚急性缺血性卒中患者）上……

    arXiv:2610.36583v1 Announce Type: cross  Abstract: Automated polysomnography after ischemic stroke is challenging because of altered sleep-related neural activity and concurrent respiratory disturbances. Existing stroke-specific approaches mainly address sleep-stage classification, leaving joint analysis of sleep architecture and respiratory events underexplored. Moreover, adapting pretrained EEG encoders introduces a representation inconsistency when downstream models are trained on embeddings from previously seen patients but evaluated on unseen individuals. We propose MM-Net, a multimodal multi-task framework that resolves this inconsistency through cross-fitting and ridge-based embedding alignment. Sleep-adapted EEG-EOG representations and engineered neural descriptors are fused with learned cardiorespiratory features through cross-attention and shared temporal modeling for simultaneous sleep staging and respiratory-event detection. On iSLEEPS (99 patients with subacute ischemic st
    
[^149]: 随大流：一种数据驱动的可行动补救方法

    Follow the Herd: A Data-Driven Approach for Actionable Recourse

    [https://arxiv.org/abs/2610.36580](https://arxiv.org/abs/2610.36580)

    本文提出首个无需人工设计行动成本函数的数据驱动行动补救方法DDAR，通过学习历史用户轨迹的动力学模型，搜索高概率且能实现期望结果的低成本行动路径。

    

    随着机器学习系统对人们的生活做出越来越重大的决策，为用户提供关于如何实现其期望结果的可行补救建议变得至关重要。目前，这一问题依赖于昂贵的人工监督来解决，即在搜索能够实现期望结果的低成本行动之前，需要人工手工设计成本函数。在本文中，我们提出了“数据驱动行动补救”（DDAR），这是首个无需人类提供关于行动成本的昂贵监督即可提供行动补救的方法。我们的方法基于这样一种理念：历史用户数据中高概率出现的轨迹应与较低的成本相关联。基于此，DDAR首先对历史用户轨迹建立动力学模型，然后在该模型下搜索能够实现期望结果的轨迹。在实验中，我们发现这种简单的方法能够推荐合理且可行的行动补救方案。

    arXiv:2610.36580v1 Announce Type: new  Abstract: As machine learning systems make increasingly greater decisions about people's lives, it is critical to provide users with actionable recourse about how they can achieve their desired outcomes. This problem is currently tackled by relying on expensive human supervision which involves hand-designing a cost function before searching for low-cost actions that achieve the desired outcome. In this paper, we propose \textit{``Data-Driven Action Recourse"} (DDAR), the first approach for providing action recourse without requiring costly supervision from humans about action costs. We build upon the notion that highly likely trajectories in the historical user data should be associated with lower costs. With this, DDAR first builds a dynamics model of historical user trajectories. Under this model, it then searches for trajectories that achieve the desired outcome. In experiments, we find that this simple approach recommends plausible action reco
    
[^150]: 少即是多：为GPU内核优化学习需要分析什么

    Less is more: learning what to profile for GPU kernel optimization

    [https://arxiv.org/abs/2610.36576](https://arxiv.org/abs/2610.36576)

    提出AutoProfiler框架，通过从优化反馈中学习确定性的性能分析树，为不同GPU内核自适应地选择最有用的性能分析信息，从而提升大语言模型自动优化GPU内核的效果。

    

    大型语言模型最近在自动优化GPU内核方面展现出强大潜力，但其有效性关键取决于内核优化模型所能获取的硬件信息。现有系统通常依赖预定义的性能分析策略，使得“分析什么”这一决策在很大程度上被排除在优化过程之外。我们证明，有效的性能分析并非简单地收集更多信息：增加性能分析指标的数量并不能单调地提升内核优化效果，且最有用的性能分析信息因内核而异。我们的理论分析进一步表明，在这种异构的性能分析效用下，基于内核条件的选择可以优于任何单一固定的性能分析配置。受此观察启发，我们提出了AutoProfiler，一个从优化反馈中学习确定性性能分析树的反馈驱动框架。该树通过迭代方式（摘要在此处截断）……

    arXiv:2610.36576v1 Announce Type: cross  Abstract: Large language models have recently shown strong potential for automatically optimizing GPU kernels, yet their effectiveness depends critically on the hardware information available to the kernel optimization model. Existing systems typically rely on predefined profiling strategies, leaving what to profile largely outside the optimization process. We show that effective profiling is not simply about collecting more information: increasing the number of profiling metrics does not monotonically improve kernel optimization, while the most useful profiling information varies across kernels. Our theoretical analysis further shows that, under such heterogeneous profiling utility, kernel-conditioned selection can outperform any single fixed profiling configuration. Motivated by this observation, we propose AutoProfiler, a feedback-driven framework for learning a deterministic profiling tree from optimization feedback. The tree is iteratively 
    
[^151]: ADTGuard：面向大语言模型流式推理的自适应深度-时间安全防护机制

    ADTGuard: Adaptive Depth-Time Safeguard for Streaming Inference in Large Language Models

    [https://arxiv.org/abs/2610.36574](https://arxiv.org/abs/2610.36574)

    ADTGuard提出了一种自适应深度-时间安全防护机制，通过在每个词元释放前整合跨层安全证据并建模风险随深度与时间的动态累积，实现了大语言模型流式推理中的实时有害内容检测。

    

    大语言模型（LLMs）能够对多样化的用户请求生成流畅的回复，然而流式输出可能在生成完成之前就让用户接触到有害内容。现有的防护机制在这一场景下存在两个局限。第一，在回复或分块边界进行的事后审核会延迟干预，当内容在验证之前被释放时存在有害暴露的风险。第二，即使采用词元级检测，粗糙的回复级监督也无法指明有害内容从何时开始，而固定的层读取方式可能会错过其他深度层级上的相关安全信号。为了解决这些局限，我们提出了ADTGuard，它在每个词元释放之前进行检查，逐步细化词元级训练目标，并根据当前前缀自适应地选择层。ADTGuard通过深度-时间安全动力学（DTSD）对深度和时间两个维度上的风险进行建模，该机制在每个词元处整合跨层的安全证据，并利用这些证据引导风险累积。

    arXiv:2610.36574v1 Announce Type: new  Abstract: Large Language Models (LLMs) can generate fluent responses to diverse user requests, yet streaming may expose users to harmful content before generation is complete. Existing guardrails face two limitations in this setting. First, post-hoc moderation at response or chunk boundaries delays intervention, risking harmful exposure when content is released before verification. Second, even with token-level detection, coarse response-level supervision does not specify when harmful content begins, while fixed-layer readouts may miss relevant safety signals at other depths. To address these limitations, we introduce ADTGuard, which checks each token before release, progressively refines token-level training targets, and adapts layer selection to the current prefix. ADTGuard models risk across depth and time through Depth-Time Safety Dynamics (DTSD), which integrates safety evidence across layers at each token and uses it to guide risk accumulati
    
[^152]: CASEL：面向可靠长时程工具使用智能体的后果感知选择性执行

    CASEL: Consequence-Aware Selective Execution for Reliable Long-Horizon Tool-Using Agents

    [https://arxiv.org/abs/2610.36570](https://arxiv.org/abs/2610.36570)

    CASEL提出了一种与模型无关的运行时分配策略，通过综合失败概率、后果严重性、可逆性、模糊性和证据新鲜度等因素，为智能体的每次工具调用智能选择直接执行、验证、沙箱、澄清、审批或拒绝执行，显著减少了长时程工具使用智能体造成有害副作用的风险。

    

    使用工具的语言智能体越来越多地修改外部系统，然而大多数运行时安全防护机制仅根据通用的不确定性评分或粗略的变更启发式方法来分配审查力度。我们提出了CASEL（后果感知选择性执行层），这是一种与模型无关的运行时分配策略，它综合了校准后的失败概率、状态相关的后果严重性、可逆性、模糊性和证据新鲜度，通过最小化每种模式的成本，为每个提议的工具调用在直接执行、验证、沙箱隔离、澄清、人工审批和拒绝执行之间做出选择。我们将直接执行形式化为选择性预测，并陈述和检验了有限样本阈值与轨迹风险账本的有效性条件。在一个包含20个随机种子、每个种子12万个测试动作的受控仿真实验中（其中每种方法拥有相同的干预菜单和相同的计算成本），CASEL将有害副作用轨迹的比例从无防护基线的14.75%降低至……（原文截断）

    arXiv:2610.36570v1 Announce Type: new  Abstract: Tool-using language agents increasingly modify external systems, yet most runtime safeguards allocate scrutiny by generic uncertainty scores or coarse mutation heuristics. We introduce CASEL (Consequence-Aware Selective Execution Layer), a model-agnostic runtime allocation policy that combines calibrated failure probability, state-dependent consequence, reversibility, ambiguity, and evidence freshness to choose, for each proposed tool call, among direct execution, verification, sandboxing, clarification, approval, and abstention by minimizing a per-mode cost. We formalize direct execution as selective prediction and state and test the validity conditions of a finite-sample threshold and a trajectory risk ledger. In a controlled simulation over 20 seeds with 120,000 test actions each, in which every method has the same intervention menu and the same compute cost, CASEL reduces harmful-side-effect trajectories from 14.75 percent for an unc
    
[^153]: NextEval：一个基于智能体的无导数优化求解器与基准测试

    NextEval: An Agent-Based Solver and Benchmark for Derivative-Free Optimization

    [https://arxiv.org/abs/2610.36553](https://arxiv.org/abs/2610.36553)

    该论文提出了NextEval，一个将大语言模型智能体与数值环境相结合的求解器与基准测试，利用LLM指导无导数优化中评估点的序贯选择，从而在有限评估预算下更有效地解决黑盒优化问题。

    

    无导数优化（DFO）处理的是导数不可用或获取代价高昂的问题。这种场景常见于基于仿真和实验的应用中，因为评估黑盒目标函数本身可能成本很高。因此，如何在有限的评估预算下，有效利用已积累的观测数据来选择信息量大的评估点，仍然是无导数优化领域的核心挑战。基于此，我们提出了NextEval求解器，这是一个基于智能体的无导数优化求解器，它整合了大语言模型（LLM）的能力来指导评估点的序贯选择。具体而言，我们构建了一个将LLM智能体与数值环境相结合的系统。该环境提供了分析历史评估结果和执行数值方法的工具，同时维护共享的评估历史记录以及特定方法的状态信息，以支持方法切换时的连续性。利用观测数据和工具反馈，

    arXiv:2610.36553v1 Announce Type: cross  Abstract: Derivative-free optimization (DFO) addresses problems where derivatives are unavailable or costly to obtain. This setting often arises in simulation-based and experimental applications, where evaluating the black-box objective can itself be costly. Thus, it remains a central challenge in DFO to effectively exploit accumulated observations to select informative evaluation points under a limited budget. In this light, we introduce the NextEval solver, an agent-based DFO solver, which integrates the capabilities of a large language model (LLM) to guide the sequential selection of evaluation points. Specifically, we establish a system that combines an LLM agent with a numerical environment. The environment provides tools for analyzing past evaluations and executing numerical methods, while maintaining a shared evaluation history and method-specific state to support continuation across method switches. Using observations and tool feedback, 
    
[^154]: REACH：一种基于强化学习的分层人形机器人爬行框架，用于自适应的受限空间穿越

    REACH: An RL-based Hierarchical Humanoid Crawling Framework for Adaptable Confined Space Traversal

    [https://arxiv.org/abs/2610.36546](https://arxiv.org/abs/2610.36546)

    提出了REACH，一个基于强化学习的分层框架，使Unitree G1人形机器人能够通过可变高度双足行走（1.27–0.50米）与爬行（0.50–0.20米）以及双向模式切换策略，实现受限空间中的自适应穿越。

    

    人形机器人在受限空间中的运动需要在多种运动模式之间进行连续的高度自适应，然而现有大多数方法仍局限于可调节高度的双足行走。我们提出了REACH，一个基于强化学习的分层框架，使Unitree G1人形机器人能够通过可变高度双足行走和爬行来穿越高度受限的环境。学习到的策略在互补的指令范围内调节身体高度：双足行走为1.27至0.50米，爬行为0.50至0.20米，将可穿越范围扩展到蹲姿行走的净空限制之外。为了应对接触丰富的爬行动力学，我们采用了基于历史条件的潜在表示，从本体感觉观测中捕获地形和交互上下文信息。双向过渡策略进一步在大运动学和接触不连续性之间实现了蹲姿双足行走与爬行之间的切换。这些技能与激光雷达（摘要在此处被截断）

    arXiv:2610.36546v1 Announce Type: cross  Abstract: Humanoid locomotion in confined spaces requires continuous height adaptation across multiple locomotion modes, yet most existing approaches remain limited to height-adjustable bipedal walking. We present REACH, an RL-based hierarchical framework that enables a Unitree G1 humanoid to traverse height-constrained environments through variable-height bipedal walking and crawling. The learned policies regulate body height over complementary command ranges of 1.27--0.50m in bipedal locomotion and 0.50--0.20m in crawling, extending traversal beyond the clearance limits of crouched walking. To address contact-rich crawling dynamics, we employ a history-conditioned latent representation that captures terrain and interaction context from proprioceptive observations. A bidirectional transition policy further connects crouched bipedal locomotion and crawling across large kinematic and contact discontinuities. These skills are integrated with a LiD
    
[^155]: FrameForge：利用视频基础模型的先验实现统一图像生成

    FrameForge: Harnessing Priors of Video Foundation Models for Unified Image Generation

    [https://arxiv.org/abs/2610.36545](https://arxiv.org/abs/2610.36545)

    FrameForge利用视频基础模型的时间先验，通过输入自适应的虚拟时间关系——MoTE（动态选择时间RoPE专家以建模条件与目标的距离）和CoTR（时间推理链）——实现统一的条件图像生成。

    

    统一条件图像生成要求单一模型在保留视觉输入不同方面特征的同时，生成遵循文本指导的目标图像。预训练视频模型通过编码跨时间的视觉变换，为此提供了有前景的基础，然而现有方法通常将条件与目标固定在特定的时间位置上。我们观察到，条件与目标之间的关系在不同生成任务中差异显著，而视频模型天然地在不同的时间尺度上编码视觉变换。这启发我们通过输入自适应的虚拟时间关系来建模条件生成。我们提出FrameForge，它捕捉此类关系的两个互补方面：时间专家混合通过动态选择具有不同虚拟区间的时间RoPE专家，来建模目标与条件之间的距离远近；时间推理链则建模……（摘要原文在此处截断）

    arXiv:2610.36545v1 Announce Type: cross  Abstract: Unified conditional image generation requires a single model to preserve different aspects of a visual input while producing a target image that follows textual guidance. Pretrained video models provide a promising foundation by encoding visual transformations across time, yet existing approaches typically assign the condition and target to fixed temporal positions. We observe that condition-target relationships vary substantially across generation tasks, while video models naturally encode visual transformations at different temporal scales. This motivates us to model conditional generation through input-adaptive virtual temporal relations. We propose FrameForge, which captures two complementary aspects of such relations. Mixture of Temporal Experts (MoTE) models how far the target lies from the condition by dynamically selecting temporal RoPE experts with different virtual intervals. Chain of Temporal Reasoning (CoTR) models how the 
    
[^156]: 监控准确率并非答案：评估AI智能体重启监控器的决策价值框架

    Monitor Accuracy Isn't the Answer: A Decision-Value Framework for Evaluating AI Agent Restart Monitors

    [https://arxiv.org/abs/2610.36544](https://arxiv.org/abs/2610.36544)

    该论文提出一个决策价值框架，从分析上证明监控器的分类准确率本身无法决定重启AI智能体是否有益，并引入决策充分性准则和有限样本无模型决策审计两种技术，以决策价值的角度评估AI智能体重启监控器。

    

    分类准确率以及真阳性率和假阳性率本身并不能决定重启AI智能体是否能提高成功率。利用一种针对零监控开销的独立干净重启的精确“警报-行动”恒等式，我们从分析上证明，一个固定90%准确率的监控器，仅因替换尝试的剩余预算成功概率不同，就能使成功率从损失五个百分点变为提升三十五个百分点。我们引入了两种以决策术语评估监控器和进度信号的技术。第一，决策充分性准则——Blackwell实验比较理论在二元行动上的特化版本——用于识别哪些进度表示或上下文压缩能够保留在指定行动之间做出最优选择所需的信息，而不依赖于轨迹重建的保真度。第二，有限样本、无模型的决策审计使用同时二项区间（原文在此处截断）……

    arXiv:2610.36544v1 Announce Type: new  Abstract: Classification accuracy and true-positive and false-positive rates alone do not determine whether restarting an AI agent improves success. Using an exact alarm-to-action identity for an independent clean restart with zero monitoring overhead, we show analytically that a monitor with fixed 90% accuracy can swing from a five-percentage-point loss to a 35-point gain in success rate, depending solely on the replacement attempt's residual-budget success probability. We introduce two techniques for evaluating monitors and progress signals in decision terms. First, a decision-sufficiency criterion, a binary-action specialization of Blackwell's comparison of experiments, identifies which progress representations or context compressions preserve the information needed to choose optimally between the specified actions, independent of trace reconstruction fidelity. Second, a finite-sample, model-free decision audit uses simultaneous binomial interv
    
[^157]: ATEMPO：按滞后随时间摊销的传递熵模型

    ATEMPO: Amortized Transfer Entropy Model Per-lag Over-Time

    [https://arxiv.org/abs/2610.36536](https://arxiv.org/abs/2610.36536)

    ATEMPO是一种基于扩散模型的传递熵估计器，通过在训练中采样不同的滞后深度并显式条件化于时间位置，仅需单个得分网络即可无需重新拟合地估计随时间变化的完整传递熵曲面，从而克服了固定输入窗口掩盖信息延迟结构以及现有方法需反复拟合的局限。

    

    当一个时间序列为另一个时间序列提供信息时，信息到达的延迟决定了模型能否有效地利用这种关系。源序列可能在某个特定延迟处具有很强的预测能力，而在相邻的滞后处则不具信息性。因此，固定的输入窗口可能会将有信息的历史与无关的历史捆绑在一起，从而掩盖这种结构。传递熵（Transfer Entropy, TE）可以通过衡量源的过去在超出目标自身历史之外还能告诉我们多少关于目标的信息，来恢复这些延迟。然而，现有的TE估计器需要针对不同的源深度和条件深度分别进行拟合，并且还需要在局部窗口上做进一步估计以跟踪随时间变化的延迟。我们提出了ATEMPO，一种在滞后和时间维度上进行摊销的基于扩散模型的传递熵估计器。ATEMPO在训练过程中对源深度和条件深度进行采样，并显式地以时间位置为条件，使得单个得分网络无需重新拟合即可估计完整的TE曲面。

    arXiv:2610.36536v1 Announce Type: new  Abstract: When one time series informs another, the delay at which that information arrives determines if models can utilize this relationship effectively. A source may be strongly predictive at one delay but uninformative at neighboring lags. Therefore, fixed input windows may obscure this structure by bundling informative and irrelevant history together. Transfer entropy (TE) can recover these delays by measuring how much a source's past tells us about a target beyond the target's own history. However, existing TE estimators require separate fits across source and conditioning depths, with further estimation on local windows to track delays over time. We introduce ATEMPO, a diffusion-based transfer entropy estimator amortized over lags and time. ATEMPO samples source and conditioning depths during training and conditions explicitly on temporal position, allowing a single score network to estimate the full TE surface without refitting. Differenci
    
[^158]: RoboAuto：通过多层次物理理解实现自主机器人程序搜索

    RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

    [https://arxiv.org/abs/2610.36533](https://arxiv.org/abs/2610.36533)

    RoboAuto 提出了一个在物体、场景和交互三个层次上显式组织物理理解的机器人程序搜索框架，利用分层知识库整合可执行技能与物体中心知识，使基于 LLM 的智能体能够更有效地搜索和改进机器人操作程序。

    

    近年来，基于大语言模型（LLM）的自主智能体在迭代生成、评估和改进可执行方案方面展现出强大能力。在机器人操作领域，Code-as-Policy 使这些智能体能够在模块化、可编辑的机器人程序中进行搜索。然而，有效的程序搜索需要物理理解能力：智能体必须理解物体如何被操作、场景几何如何约束动作，以及交互为何成功或失败。现有方法主要依赖执行反馈来改进程序，往往无法解决这些物理层面的问题。我们提出了 RoboAuto，这是一个机器人程序搜索框架，通过三个组件在物体、场景和交互三个层次上显式地组织物理理解。首先，一个分层知识库将可执行技能与以物体为中心的知识（包括可供性、空间关系和动作效果）相结合，这些知识从具身数据和在线任务中学习获得……（原文在此处截断）

    arXiv:2610.36533v1 Announce Type: cross  Abstract: Recently, LLM-based autonomous agents have shown strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy lets these agents search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on execution feedback for program refinement, often leaving these physical questions unresolved. We introduce RoboAuto, a framework for robot program search that explicitly organizes physical understanding at the object, scene, and interaction levels through three components. First, a hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, learned from embodied data and online task so
    
[^159]: MedConstBench：衡量“宪法”如何引导大语言模型医疗伦理行为的基准测试

    MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

    [https://arxiv.org/abs/2610.36513](https://arxiv.org/abs/2610.36513)

    该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。

    

    随着大语言模型的快速发展，它们被越来越多地用于回答医学问题，且许多此类模型如今已在本地部署。模型一旦部署，其行为主要由写入其“宪法”中的伦理政策所决定。然而，现有研究大多关注事实正确性，而非相互冲突的伦理价值之间的权衡。即使是那些确实评估伦理的基准测试，也将伦理视为模型的固定属性而非变量，因此“宪法”如何重塑医疗伦理行为在很大程度上仍未被探索。我们提出 MedConstBench，一个将部署宪法作为受控变量的医疗伦理基准测试，它围绕比彻姆的四大伦理原则组织，并通过人工校准的多模型同行评审进行评分。该基准包含我们自主构建的临床困境数据集 MedDilemma，并支持对“宪法”如何引导大语言模型进行系统性分析。对十个开源（模型）的评估……

    arXiv:2610.36513v1 Announce Type: new  Abstract: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-sourc
    
[^160]: AI科学家知道什么是“新”吗？

    Do AI Scientists Know What's New?

    [https://arxiv.org/abs/2610.36511](https://arxiv.org/abs/2610.36511)

    该论文提出了一个结合时间真值的分析框架，从提案和论断两个层面评估AI科学发现中的新颖性检查，发现对非新颖工作拒绝率高的检查往往会误拒那些后来被人类研究者实现的新颖想法。

    

    AI科学发现流程面临的一个重大挑战是如何判断一个生成的想法是否具有真正意义上的新颖性。理想情况下，这些流程中的新颖性检查必须能够拒绝非新颖的想法，同时不过度拒绝新颖的想法；此外，它们还应能够区分提案中哪些部分是非新颖的，而不是不透明地丢弃整个提案。我们设计了一个分析框架，在提案层面和论断层面结合时间真值来评估新颖性检查，该框架使用了四个共包含1,060个提案的数据集，这些数据集由已发表的论文、审稿人争议记录，以及与该领域后来实际发表的成果相匹配的生成想法构建而成。利用该框架对现有新颖性检查进行审计，我们发现对非新颖工作拒绝率高的检查，往往也会对后来被人类研究者实现的想法产生大量误报；同时我们还观察到，旨在对提案进行整体评估的检查能够（精准指出非新颖部分）……

    arXiv:2610.36511v1 Announce Type: cross  Abstract: A major challenge facing AI scientific discovery pipelines is how to determine when a generated idea is meaningfully novel. Ideally, novelty checks in these pipelines must be able to reject non-novel ideas while not rejecting too many novel ideas, and they should also be able to distinguish which parts of a proposal are non-novel, as opposed to opaquely discarding entire proposals. We design an analytical framework for evaluation of novelty checks with temporal ground truth at both the proposal- and claim-level, using four datasets of 1,060 proposals built from published papers, reviewer disputes, and generated ideas matching ones the field went on to publish. Auditing existing novelty checks with this framework, we find that checks with high rejection rates on non-novel work also tend to have high false alarms on ideas that are later realized by human researchers, and observe that checks built to evaluate proposals as a whole can poin
    
[^161]: ProCerDCL：面向深度学习编译器优化的查询高效黑盒概率验证

    ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

    [https://arxiv.org/abs/2610.36499](https://arxiv.org/abs/2610.36499)

    ProCerDCL 提出了一种查询高效的黑盒概率验证框架，利用单一序贯检验在指定错误预算内判定属性违反概率是否超过阈值，并将其应用于深度学习编译器优化的验证。

    

    我们提出了 ProCerDCL，一个查询高效的黑盒概率验证框架。给定一个固定的、可采样的输入分布和一个二元属性判定器，ProCerDCL 针对阈值 θ 评估该属性的违反概率 p。在独立采样且判定器无状态的假设下，接受结论断言 p≤θ，拒绝结论断言 p>θ+η（其中 η 为用户指定的分离裕度），而“不确定”的结论则表示没有足够的证据支持任一结论。每类错误结论的概率均被其指定的错误预算所约束。为了高效支持这种三值判定契约，ProCerDCL 采用单一序贯检验，消除了基线方法所需的区间族。该检验通过修正的精确二项置信区间或任意时点有效的置信序列来处理重复采样检查。我们将 ProCerDCL 应用于概率化的局部模型语义一致性……

    arXiv:2610.36499v1 Announce Type: cross  Abstract: We present \tool, a query-efficient framework for black-box probabilistic verification. Given a fixed, sampleable input distribution and a binary property oracle, \tool assesses the property's violation probability $p$ against a threshold $\theta$. Under independent sampling and stateless oracle behavior, acceptance asserts $p\le\theta$, rejection asserts $p>\theta+\eta$ for a user-specified separation margin, and an inconclusive verdict indicates insufficient evidence for either conclusion. The probability of each type of erroneous verdict is bounded by its specified error budget. To support this three-valued contract efficiently, \tool uses a single sequential test, eliminating the interval family required by the baseline. The test accounts for repeated sampling checks through either corrected exact binomial confidence intervals or an anytime-valid confidence sequence. We apply \tool to probabilistic local model semantic consistency,
    
[^162]: AC-JEPA：用于视觉到符号翻译的锚定跨模态联合嵌入预测架构

    AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation

    [https://arxiv.org/abs/2610.36493](https://arxiv.org/abs/2610.36493)

    该论文提出AC-JEPA，一种与监督微调兼容的即插即用模块，通过在跨模态两个方向上预测被掩码的中间状态的联合嵌入目标，来利用不完美的配对数据，从而在保留token级监督的同时提升视觉到符号跨模态翻译对目标序列错误的鲁棒性。

    

    跨模态翻译要求模型既能捕捉跨模态间的共享语义，又能保留目标模态所需的细粒度结构。然而在现实世界的科学领域中，配对数据往往并不完美，而大规模获取干净的配对数据成本高昂。监督微调（SFT）通过分类式的token级别监督直接优化跨模态翻译，但该目标对目标序列中的错误较为敏感。JEPA（联合嵌入预测架构）提供了一种互补的表示空间层面的目标，但现有的大多数形式侧重于模态内的预测，而非跨模态的翻译。我们提出了锚定跨模态JEPA（AC-JEPA），这是一个与SFT兼容的即插即用模块，利用配对数据在两个跨模态方向上预测被掩码的中间状态。联合嵌入目标聚焦于可从上下文预测的跨模态结构，而因果SFT路径则保留直接的token级别监督……（原文摘要在此处截断）

    arXiv:2610.36493v1 Announce Type: new  Abstract: Cross-modal translation requires models to capture shared semantics across modalities while preserving the fine-grained structure required by the target modality. Yet in real-world scientific domains, paired data are often imperfect, while obtaining clean pairs at scale is costly. Supervised fine-tuning (SFT) directly optimizes cross-modal translation through categorical token-level supervision, but this objective is sensitive to errors in the target sequences. JEPA offers a complementary representation-space objective, but most existing formulations focus on prediction within a modality rather than translation across modalities. We introduce Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible plug-in that predicts masked intermediate states in both cross-modal directions using paired data. The joint-embedding objective targets cross-modal structure predictable from context, while the causal SFT path retains direct token-level supervi
    
[^163]: CogArena：基于交互式行为实验的多模态智能体基准测试

    CogArena: Benchmarking Multimodal Agents on Interactive Behavioral Experiments

    [https://arxiv.org/abs/2610.36484](https://arxiv.org/abs/2610.36484)

    提出CogArena基准，让多模态智能体通过真实网页浏览器以与人类被试相同的视觉界面完成十个行为科学实验范式，并采用“任务完成度—准确性—行为特征”三级评分来检验AI认知能力，同时应对在线行为数据被AI污染的问题。

    

    关于大型语言模型认知能力的论断，通常是通过将行为范式转化为文本提示来检验的，这种方式绕过了人类被试实际参与实验时所面对的视觉界面与时间压力。与此同时，AI智能体现在已能够完成在线行为任务并产生类似人类的数据，这引发了人们的担忧：社会科学与行为科学领域的在线数据集正日益面临被污染的风险。我们提出了CogArena，这是一个包含十个来自社会科学与行为科学实验范式的基准测试。智能体通过标准网页浏览器完成实验，使用与在线人类被试相同的视觉刺激和反应界面。一个三级评分程序用于评估智能体是否完成了任务、其表现的准确性如何，以及其数据是否呈现出每种范式在人类被试身上可靠产生的行为特征。

    arXiv:2610.36484v1 Announce Type: new  Abstract: Claims about the cognitive capabilities of large language models are typically tested by translating behavioral paradigms into text prompts, bypassing the visual interfaces and time pressure through which human participants actually engage with experiments. At the same time, AI agents can now complete online behavioral tasks and produce human-like data, which raises concerns that online datasets in the social and behavioral sciences are increasingly open to contamination. We introduce CogArena, a benchmark of ten experimental paradigms from the social and behavioral sciences. Agents complete the experiments through a standard web browser, working with the same visual stimuli and response interfaces that online participants use. A three-level scoring procedure evaluates whether the agent completed the task, how accurately it performed, and whether its data show the behavioral signatures each paradigm reliably produces in people. Across fo
    
[^164]: 为在线策略蒸馏提出更好的轨迹采样方法

    Proposing Better Rollouts for On-Policy Distillation

    [https://arxiv.org/abs/2610.36477](https://arxiv.org/abs/2610.36477)

    该论文揭示在线策略蒸馏中固定以学生模型作为 rollout 策略会导致教师-学生错位不断累积并引发重复等退化行为，并据此提出更好的 rollout 策略设计以提升蒸馏效果。

    

    在线策略蒸馏（On-policy distillation, OPD）通过让学生模型在教师模型的监督下从自身采样的轨迹中学习，从而改进语言模型的后训练，使训练过程与推理时的行为保持一致。然而，OPD 中的 rollout 策略通常被视为固定的：每一轮都直接从当前的学生模型进行采样。我们证明这种选择会造成教师与学生之间错位的不断累积，因为它要求教师能够对学生采样到的任意轨迹（包括教师自身难以到达的轨迹）提供有效监督。在这类轨迹上，教师的监督可能变得嘈杂甚至具有误导性，导致学生出现重复等不良生成模式。此外，由于更新后的学生模型又会定义后续轮次的 rollout 策略，这些不良模式会在训练过程中持续存在并可能不断放大。这一失败模式表明，改进 OPD 需要将 rollout 策略（摘要在此处截断）。

    arXiv:2610.36477v1 Announce Type: cross  Abstract: On-policy distillation (OPD) improves language-model post-training by letting the student learn from its own sampled trajectories under teacher supervision, aligning the training process with inference-time behavior. However, the rollout policy in OPD is usually treated as fixed: each round samples directly from the current student. We show that this choice can create a compounding teacher--student mismatch, because it requires the teacher to provide effective supervision on whichever trajectories the student samples, including ones less reachable by the teacher itself. On such trajectories, teacher supervision can become noisy or even misleading, leading to undesirable student patterns such as repetition. Moreover, since the updated student in turn defines the rollout policy of subsequent rounds, these patterns persist and potentially amplify over training. This failure mode suggests that improving OPD requires treating the rollout po
    
[^165]: FIDES-Live：评估序列交易智能体的行为一致性

    FIDES-Live: Evaluating Behavioral Concordance in Sequential Trading Agents

    [https://arxiv.org/abs/2610.36476](https://arxiv.org/abs/2610.36476)

    该论文提出FIDES-Live审计协议，用于检验序列交易智能体的决策、订单、账本与报告之间是否一致，通过对7,299个加密货币账户轮次和310轮股票试点的审计，发现决策有效性达97.35%、报告准确率90.81%、承诺遵循度94.36%，但不同推理路由的表现差异显著。

    

    序列交易智能体会产生决策、订单、执行回执、账户状态和报告，即使最终投资组合价值被正确计算，这些内容也可能彼此不一致。我们提出FIDES-Live，这是FIDES的运营化扩展，将审计范围从生成策略一致性扩展到对序列智能体行为轨迹的审计。该协议将评估划分为结构化决策有效性、可选的数值承诺遵循度、执行与账本对账以及基于回执的报告几个部分，并明确界定了覆盖范围和不可评分的结果。我们对十个被请求的推理路由进行了回顾性审计，涵盖7,299个加密货币账户轮次和一个310轮的股票试点，使用真实市场观测和本地模拟执行。在可评分的响应中，加密货币决策有效性为97.35%，结构化报告准确率为90.81%，但不同路由层面的决策与报告性能可能出现显著分歧。整体承诺遵循度为94.36%；然而它会降至……（原文摘要在此处被截断）

    arXiv:2610.36476v1 Announce Type: cross  Abstract: Sequential trading agents produce decisions, orders, execution receipts, account states, and reports that need not agree, even when the final portfolio value is correctly computed. We present FIDES-Live, an operational extension of FIDES from generated strategy concordance to the audit of sequential agent traces. The protocol separates structured decision validity, optional numerical commitment adherence, execution and ledger reconciliation, and receipt- grounded reporting, with explicit coverage and unscorable outcomes. We retrospectively audit ten requested inference routes over 7,299 cryptocurrency account-rounds and a 310-round equity pilot, using real market observations and local paper execution. Cryptocurrency decision validity is 97.35% and structured reporting accuracy is 90.81% among scorable responses, but route-level decision and reporting performance can diverge sharply. Commitment adherence is 94.36% overall; it falls to 
    
[^166]: 何时干预？LLM智能体中错误与自我恢复的实证研究

    When to Intervene? An Empirical Study of Errors and Self-Recovery in LLM Agents

    [https://arxiv.org/abs/2610.36471](https://arxiv.org/abs/2610.36471)

    该论文构建了ReInBench数据集（含1,259条智能体轨迹和1,403个错误事件），首次系统实证研究LLM智能体的错误自我恢复现象，发现大量错误可在几步内自我恢复，为判断何时干预、何时等待提供了依据。

    

    LLM智能体与Web服务交互以代表用户执行任务。这些交互过程中出现的错误可能中断执行并阻碍任务完成，从而促使外部干预。然而，智能体也可能通过持续执行实现自我恢复，这就引出了何时干预、何时等待自我恢复的问题。我们对LLM智能体中的错误与自我恢复进行了实证研究。我们构建了ReInBench，一个包含1,259条智能体轨迹和1,403个错误事件的数据集，涵盖计算机使用和Web任务。我们的研究考察了智能体自我恢复的频率和速度、自我恢复与后续执行及任务完成之间的关系，以及错误发生时能否预测自我恢复。我们发现725个错误事件实现了自我恢复，且通常在几步之内完成，不同执行LLM的自我恢复表现存在差异。自我恢复还能帮助智能体从后续同类错误中恢复并成功完成……

    arXiv:2610.36471v1 Announce Type: new  Abstract: LLM agents interact with Web services to perform tasks on users' behalf. Errors during these interactions can disrupt execution and prevent task completion, motivating external intervention. However, agents may self-recover through continued execution, raising the question of when to intervene and when to wait for self-recovery. We present an empirical study of errors and self-recovery in LLM agents. We construct ReInBench, a dataset containing 1,259 agent traces and 1,403 error events across computer-use and Web tasks. Our study examines how often and how quickly agents self-recover, how self-recovery relates to subsequent execution and task completion, and whether self-recovery can be predicted when an error occurs. We find that 725 error events self-recover, often within a few steps, with self-recovery performance differing across Execution LLMs. Self-recovery can also help agents recover from subsequent errors of the same type and su
    
[^167]: 完全坍缩是次优的：面向长尾识别的受控部分神经坍缩

    Full Collapse Is Suboptimal: Controlled Partial Neural Collapse for Long-Tailed Recognition

    [https://arxiv.org/abs/2610.36460](https://arxiv.org/abs/2610.36460)

    提出受控部分神经坍缩方法，通过CASA分类器和ETF-AM损失让头部类别沿ETF坍缩而尾部类别保留类内多样性，解决了长尾识别中强制完全坍缩导致的过度收缩与泛化能力下降问题。

    

    近期的长尾识别（LTR）方法引入了神经坍缩（NC），并将特征对齐到一个固定的等角紧框架（ETF），作为所有类别的统一目标。然而，强制尾部类别完全坍缩会过度收缩其样本并削弱泛化能力。为解决这一问题，我们提出了一种受控的部分神经坍缩方法，通过对骨干网络的两项协同控制来实现：类别自适应稀疏激活（CASA）分类器和ETF约束的自适应间隔（ETF-AM）损失。该方法并非强制所有类别统一地完全坍缩，而是允许头部类别沿ETF坍缩，同时让尾部类别保留类内多样性。这避免了尾部类别被强制进行完全坍缩——这种做法在不平衡条件下是次优的。为了设定坍缩的维度，CASA通过基于特征的条件门控诱导与类别相关的节点特化，使尾部类别占据各自的特征方向。

    arXiv:2610.36460v1 Announce Type: cross  Abstract: Recent long-tailed recognition (LTR) methods incorporate neural collapse (NC) and align features to a fixed Equiangular Tight Frame (ETF) as a uniform target for all classes. However, forcing tail classes to fully collapse over-contracts their samples and weakens generalization. To address this, we propose a controlled partial neural collapse, realized through two coordinated controls on a backbone: the Class-Adaptive Sparse Activation (CASA) classifier and the ETF-constrained adaptive margin (ETF-AM) loss. Rather than forcing a uniform full collapse, this regime allows head classes to collapse along the ETF while tail classes retain within-class diversity. This prevents the tail from being forced into full collapse, which is suboptimal under imbalance. To set the dimensionality of collapse, CASA induces class-related node specialization through feature-conditioned gating so that tail classes occupy their own feature directions. For th
    
[^168]: LociArt：自由风格局部视频风格化的统一范式

    LociArt: A Unified Paradigm for Free-Style Local Video Stylization

    [https://arxiv.org/abs/2610.36456](https://arxiv.org/abs/2610.36456)

    提出LociArt统一范式，通过基于生成式精炼网络的GRNStyle与极坐标风格感知掩码，实现对局部视频区域“在哪里风格化、应用什么风格”的精确控制并保持时序一致性，同时借助自动化流水线构建了10万对样本的LociArt-100K数据集。

    

    视频风格化通常对整个场景进行转换，而局部艺术编辑则需要精确的区域—风格关联、空间控制和时序一致性。我们提出LociArt，一个用于自由风格局部视频风格化的统一范式，能够明确指定在何处进行风格化以及应用何种风格，同时保留非目标内容。其核心GRNStyle构建于生成式精炼网络（Generative Refinement Networks, GRN）之上，并引入极坐标风格感知掩码，将风格身份作为等范数方向嵌入所选区域中。互补的表示—几何条件化将编码的掩码特征与原始掩码的空间支持相结合，将时空上下文与几何控制融为一体。为提供成对监督，我们开发了一条结合全局风格化、视频物体移除、抠图与合成的自动化流水线，并构建了包含10万局部风格化样本对的LociArt-100K数据集。我们进一步建立……（原文在此处截断）

    arXiv:2610.36456v1 Announce Type: cross  Abstract: Video stylization typically transforms entire scenes, while local artistic editing requires precise region--style association, spatial control, and temporal consistency. We introduce LociArt, a unified paradigm for free-style local video stylization that explicitly specifies where to stylize and what style to apply while preserving non-target content. At its core, GRNStyle builds on Generative Refinement Networks (GRN) and introduces a polar style-aware mask that embeds style identities as equal-norm directions within selected regions. Complementary representation--geometry conditioning combines encoded mask features with raw-mask spatial support, integrating spatiotemporal context with geometric control. To provide paired supervision, we develop an automated pipeline combining global stylization, video object removal, matting, and compositing, and construct the LociArt-100K Dataset with 100K local stylization pairs. We further establi
    
[^169]: 纠正还是操纵？测量用户压力下LLM智能体的认知警觉

    Correction or Manipulation? Measuring Epistemic Vigilance in LLM Agents Under User Pressure

    [https://arxiv.org/abs/2610.36453](https://arxiv.org/abs/2610.36453)

    该论文在问答与客服行动两个场景下，对四个厂商的七个大语言模型施加分级的用户压力，并结合信号检测理论，系统测量了LLM智能体能否区分用户的善意纠错与恶意操纵。

    

    回答问题并代表用户行事的LLM智能体必须决定何时向持不同意见的用户让步。异议可能纠正一个真实的错误，也可能是操纵的企图。先前的研究表明，当受到用户质疑时，模型经常放弃正确的答案，但尚不清楚它们是能够区分纠正与操纵，还是仅仅在让步倾向上有所不同。我们在两项研究中对来自四个厂商的七个LLM进行了检验：一项研究关注智能体给出的答案，另一项关注它们作为客服智能体所采取的行动，其政策规定仅允许为通过身份验证的用户更改账户。模拟用户以分级的压力推动智能体给出正确或错误的答案。借鉴关于认知警觉的研究，这些压力要么诉诸来源（如声称的权威），要么提供智能体无法验证的内容（即引用的来源）。借鉴信号检测理论，我们在一组……上进行测量（原文摘要在此截断）。

    arXiv:2610.36453v1 Announce Type: new  Abstract: LLM agents that answer questions and act on behalf of users must decide when to yield to a user who disagrees. Disagreement may correct a genuine error or may be an attempt at manipulation. Prior work shows that models frequently abandon correct answers when challenged by users, yet it remains unclear whether they distinguish correction from manipulation or merely differ in their propensity to concede. We examine this question across seven LLMs from four vendors in two studies, one on the answers agents give and one on the actions they take as customer-support agents whose policy permits account changes only for verified users. Simulated users push toward correct and incorrect answers with graded pressure. Following research on epistemic vigilance, the pressure either appeals to the source, such as claimed authority, or offers content the agent cannot verify, namely a cited source. Drawing on signal detection theory, we measure on a set 
    
[^170]: 一颗心形符号意味着递归：探索视觉语言模型中的视觉含义与交流

    A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs

    [https://arxiv.org/abs/2610.36449](https://arxiv.org/abs/2610.36449)

    研究发现视觉语言模型会被简单视觉符号隐性地引导行为（如心形符号引导模型采用递归解法），且模型间视觉交流存在“生成—解读”鸿沟——模型发送的直观视觉线索往往无法有效引导其他模型。

    

    视觉语言模型（VLM）是否具备与我们相同的视觉交流能力？我们分享了初步研究发现，从一个令人惊讶的观察开始：简单的视觉符号（如红色心形）能够在看似无关的任务上强烈影响模型行为，例如在简单编程问题上引导模型倾向于递归解法。值得注意的是，这种引导效应在很大程度上是隐性的，因为模型对视觉符号的直接解读和思维轨迹与其观察到的实际行为存在显著差异。在模型之间的视觉交流中，这导致了一种“生成—解读”鸿沟：模型倾向于发送它们自己（以及人类）觉得直观的形状基元，但这些视觉线索往往无法有效引导其他模型。在涉及抽象概念和故事的进一步交流游戏中，模型能够创造性地使用视觉元素相互传递意义，尽管其中某些交流内容对人类而言仍不直观。

    arXiv:2610.36449v1 Announce Type: new  Abstract: Do vision-language models (VLMs) share our ability to communicate visually? We share preliminary findings, starting from a surprising observation: simple visual symbols like a red heart can strongly influence model behavior on seemingly unrelated tasks, such as steering models towards recursive solutions on simple coding problems. Notably, this steering effect is largely implicit, as models' direct interpretations and thinking traces about visual symbols diverge substantially from their observed behavior. In visual communication between models, this leads to a generation--interpretation gap: while models tend to send shape primitives that they (and humans) find intuitive, these visual cues often fail to steer other models. In further communication games involving abstract concepts and stories, models can creatively use visuals to transmit meaning to one another, although some communications remain non-intuitive to humans. These findings 
    
[^171]: 从解题者轨迹到教学轨迹：认知对齐的推理蒸馏

    From Solver Trajectories to Teaching Trajectories: Cognition-Aligned Reasoning Distillation

    [https://arxiv.org/abs/2610.36445](https://arxiv.org/abs/2610.36445)

    提出CARD框架，通过强化学习将大型推理模型作为“解题者”产生的轨迹转化为符合小模型认知能力的“教学轨迹”，从而在不进行昂贵重采样的情况下提升知识蒸馏效果。

    

    从大型推理模型（LRMs）进行知识蒸馏时，通常直接使用原始的教师轨迹作为监督信号。然而，这种做法造成了蒸馏目标与教师轨迹性质之间的不匹配。先进的大型推理模型通常通过奖励驱动的后训练（如RLHF）来提升推理能力。这种训练将它们优化为强大的解题者，而非能够产出较小模型易于学习的推理过程的教师。由于较小模型相较于大型教师模型具有不同的“认知能力”，直接模仿原始教师轨迹有时效果不佳，且通常需要大量高质量样本。为了弥合这种不匹配，我们提出了认知对齐推理蒸馏（CARD），这是一个基于强化学习的框架，能够将解题者轨迹转化为与学生认知对齐的教学轨迹。该方法避免了昂贵的重采样……

    arXiv:2610.36445v1 Announce Type: new  Abstract: Knowledge distillation from large reasoning models (LRMs) often uses raw teacher trajectories as supervision. However, this practice creates a mismatch between the goal of distillation and the nature of teacher trajectories. Advanced LRMs typically improve their reasoning capabilities through reward-driven post-training, such as RLHF. This training optimizes them as strong solvers, rather than as teachers that produce reasoning processes smaller models can easily learn. Since smaller models have different "cognitive capacities" compared to their larger teachers, directly imitating raw teacher trajectories can sometimes be ineffective and often requires a substantial number of high-quality samples. To bridge this mismatch, we propose Cognition-Aligned Reasoning Distillation (CARD), an RL-based framework that transforms solver trajectories into teaching trajectories aligned with the student's cognition. Avoiding expensive resampling, our f
    
[^172]: RouteDRAG：面向多模态文档RAG的学习式路由与智能体执行控制

    RouteDRAG: Learning to Route and Control Agentic Execution for Multimodal Document RAG

    [https://arxiv.org/abs/2610.36444](https://arxiv.org/abs/2610.36444)

    该论文提出RouteDRAG学习型框架，通过自适应路由决定何时调用智能体并控制其执行过程，在多模态文档RAG任务中减少不必要的计算开销，同时保持或提升答案准确性。

    

    基于Web的应用日益依赖于对长篇多模态文档的理解，回答问题往往需要整合跨页面、表格和图表的信息。智能体检索增强生成通过迭代式检索与推理来应对这类任务。然而，智能体执行并非总是必要的，它会引入额外的计算开销，并可能产生错误答案。我们的实证分析表明，智能体执行带来的收益是依赖于具体查询的：额外的操作可能提升、维持或降低答案的准确性。这些发现促使我们对智能体的调用及其后续执行过程进行控制。为解决这些挑战，我们提出了RouteDRAG，一个由四个组件组成的学习型框架。其中，统一文档索引将多模态内容组织成基于图的索引，并从初始检索中导出查询表示；自适应路径路由在保留精确回答的同时……

    arXiv:2610.36444v1 Announce Type: new  Abstract: Web-based applications increasingly rely on understanding long multimodal documents, where answering questions often requires combining information across pages, tables, and figures. Agentic retrieval-augmented generation addresses these tasks through iterative retrieval and reasoning. However, Agentic execution is not always necessary, introducing additional computation and potentially incorrect answers. Our empirical analysis reveals that Agentic execution provides query-dependent benefits, while additional operations may improve, preserve, or degrade answer accuracy. These findings motivate controlling both Agent invocation and subsequent execution. To address these challenges, we propose RouteDRAG, a learning-based framework comprising four components. Unified Document Index organizes multimodal content into a graph-backed index and derives query representations from initial retrieval. Adaptive Path Routing preserves exact answering 
    
[^173]: 有你在场时，你的智能体会表现不同：用户在场如何改变智能体在谈判中的行为

    Your Agent Acts Differently When You're Around: How User Presence Changes Agent Behavior in Negotiation

    [https://arxiv.org/abs/2610.36435](https://arxiv.org/abs/2610.36435)

    研究发现用户在场会显著改变LLM谈判智能体的行为：有用户参与的谈判方能获得更高收益且智能体谈判更激进，但同时用户在场也会导致针对交易对手的不对齐行为发生率大约翻倍。

    

    大语言模型（LLM）智能体越来越多地代表用户与其他智能体进行交互，然而处于决策回路中的用户如何影响其智能体的行为——无论是好是坏——仍然鲜为人知。我们通过谈判这一视角来研究这个问题，在谈判中，智能体既要代表用户的利益，又要与其他智能体进行策略性讨价还价。为了分离出用户在场所带来的影响，我们开发了PRESENT，这是一个受控评估框架，用于比较有交互用户和无交互用户的谈判场景，同时在各实验条件下保持游戏规则、目标和用户偏好不变。在涵盖7个游戏和8个模型共11.2k次运行的实验中，我们发现用户在场会显著改变谈判结果和行为。具体而言，有用户参与的谈判方获得更高的收益（平均提升13.7%），且其智能体的谈判风格更具攻击性。令人惊讶的是，用户在场还会诱发不对齐行为，无论是对交易对手方（其发生率大约翻倍），还是（摘要在此处截断）

    arXiv:2610.36435v1 Announce Type: new  Abstract: LLM agents increasingly interact with other agents on behalf of users, yet how users in the loop affect their agents' behavior--for better or worse--remains poorly understood. We study this question through the lens of negotiation, where agents must represent users while strategically bargaining with other agents. To isolate the effect of user presence, we develop PRESENT, a controlled evaluation framework that compares negotiation settings with and without an interactive user, while holding the game, objectives, and user preferences fixed across conditions. Across 11.2k runs spanning 7 games and 8 models, we find that user presence significantly changes both outcomes and behavior. Specifically, negotiation parties with a user obtain higher payouts (+13.7% on average) and their agents negotiate more aggressively. Surprisingly, user presence also induces misaligned behavior, both toward counterparties (where its rate roughly doubles) and 
    
[^174]: LM-DSB：一种用于建筑行业每日安全公告自动化生成的轻量级多智能体系统

    LM-DSB: A Lightweight Multi-agent System for Automated Daily Safety Bulletin Generation in Construction

    [https://arxiv.org/abs/2610.36433](https://arxiv.org/abs/2610.36433)

    该研究提出LM-DSB系统，通过多智能体分层协作架构结合RAG与本地部署的大语言模型，在消费级个人电脑上自动生成建筑每日安全公告，解决了人工编写耗时长、质量不稳定的问题。

    

    建筑安全管理至今仍难以高效产出每日安全公告（DSB），这是一种结构化的日常沟通文件，用于识别现场特定危险、传达安全规程并为工人制定预防措施。手动编写DSB每天需耗时2-3小时且质量参差不齐，25%的施工班组从未开展任务前安全规划（Olson等人，2016），导致安全人员无暇专注于现场监督。本研究提出了轻量级多智能体DSB系统（LM-DSB），将检索增强生成（RAG）与可本地部署的大语言模型（LLM）相结合，实现在消费级个人电脑上自动生成DSB。LM-DSB采用分层架构协调规划者、研究者和写作者三个专业智能体，并通过在建筑安全文档上的领域特定微调，以及基于37份健康、安全与环境（HSE）文档的RAG检索增强来提升系统性能。一项两阶段验证评估了该系统的可

    arXiv:2610.36433v1 Announce Type: new  Abstract: Construction safety management still struggles to produce daily safety bulletins (DSBs), structured daily communications that identify site-specific hazards, convey safety protocols, and set preventive measures for workers. Manual DSB creation takes 2-3 hours daily and yields inconsistent quality, with 25% of crews never conducting pretask planning (Olson et al., 2016), diverting safety personnel from field supervision. This study proposes the Lightweight Multi-Agent DSB System (LM-DSB), which integrates retrieval-augmented generation (RAG) with a locally deployable Large Language Model (LLM) to automate DSB generation on consumer-grade personal computers. LM-DSB coordinates specialized planner, researcher, and writer agents in a hierarchical architecture, enhanced by domain-specific fine-tuning on construction safety documentation and RAG grounding in 37 health, safety, and environment (HSE) documents. A two-phase validation assessed fe
    
[^175]: 面向行为多样性强化学习的排斥性种群策略梯度

    Repulsive Population Policy Gradients for Behaviorally Diverse Reinforcement Learning

    [https://arxiv.org/abs/2610.36425](https://arxiv.org/abs/2610.36425)

    提出RePPO，通过改造PPO并引入依赖种群的动态塑形奖励来联合优化一个紧凑的多样化高质量策略种群，从而在连续控制任务上实现了优于现有QD-RL方法的质量-多样性权衡，摆脱了对昂贵进化算子的依赖。

    

    质量-多样性强化学习的目标是寻找一组高性能策略，以多样化的方式解决决策问题。尽管近年来取得了进展，QD-RL方法主要依赖进化算子——包括黑盒优化器、变异和交叉——来促进探索并诱导多样性。随着策略和行为空间规模的增大，扩展这些方法的成本变得越来越高。基于质量-多样性优化的Soft QD视角，我们提出了排斥性种群策略优化，它对PPO进行改造，利用一种依赖于种群且动态塑形的奖励，联合优化一个由多样化、高质量策略组成的紧凑种群。在标准连续控制任务上，与最先进的QD-RL方法相比，RePPO实现了出色的质量-多样性权衡。消融实验进一步刻画了这一权衡特性。

    arXiv:2610.36425v1 Announce Type: cross  Abstract: The goal of Quality-Diversity Reinforcement Learning (QD-RL) is to find a population of high-performing policies that solve a decision-making problem in diverse ways. Despite recent progress, QD-RL methods predominantly rely on evolutionary operators--including black-box optimizers, mutations, and crossovers--to facilitate exploration and induce diversity. Scaling these methods becomes increasingly costly as policies and behavior spaces grow larger. Building on the Soft QD view of quality-diversity optimization, we introduce \textbf{Re}pulsive \textbf{P}opulation \textbf{P}olicy \textbf{O}ptimization (\textbf{RePPO}), which adapts PPO to jointly optimize a compact population of diverse, high-quality policies using a population-dependent, dynamically shaped reward. On standard continuous-control tasks, RePPO achieves a strong quality-diversity tradeoff compared with state-of-the-art QD-RL methods. Ablations further characterize this tra
    
[^176]: 滑块拼图中大语言模型内在规划能力的可验证效率极限

    Verifiable Efficiency Limits of Innate LLM Planning in Sliding Puzzles

    [https://arxiv.org/abs/2610.36415](https://arxiv.org/abs/2610.36415)

    本文提出可验证的 SPE = R × Q × E 框架，将大语言模型在滑块拼图上的内在规划效率分解为执行可靠性、规划协调性与行动经济性，并发现即便模型能完成全部任务，其效率在更长视野下仍会下降，且高效率可与低目标达成率并存。

    

    arXiv:2610.36415v1 公告类型：新论文 摘要：借助工具完成的规划成功可能反映的是外部搜索与验证的作用，从而使模型自身无辅助的规划能力仍悬而未决。我们通过在不使用工具、不依赖执行反馈的情况下于单次响应中生成的完整规划，来分离模型的内在规划能力。我们提出了一个可验证的框架，将成功加权的规划效率分解为 SPE = R × Q × E：即执行可靠性、合法规划之间的规划协调性，以及已完成规划之间的行动经济性。轨迹回放与经认证的最优解用于验证这些因素；联合评分还将失败的尝试纳入考量。我们在 4×4 滑块拼图上评估了七种模型条件下的 8,400 个完整规划输出，并在主要队列中精确控制了步长视野与障碍数量。结果显示，即使能够完成全部试次的模型，在更长视野下仍会变得不够经济高效，而已完成规划中较高的效率也可能与较差的目标达成率并存。SPE 能够区分整体能力层级……（原文摘要至此截断）

    arXiv:2610.36415v1 Announce Type: new  Abstract: Tool-assisted planning success can reflect external search and verification, leaving a model's unaided planning ability unresolved. We isolate innate planning through complete plans generated in one response without tools or execution feedback. We propose a verifiable framework that decomposes success-weighted plan efficiency as $\mathrm{SPE}=R\times Q\times E$: execution reliability, planning coordination among legal plans, and action economy among completed plans. Trajectory replay and certified optima verify these factors; the joint score includes unsuccessful attempts. We evaluate 8,400 full-plan outputs from seven model conditions on $4\times4$ sliding puzzles, controlling exact horizon and obstacle count in the primary cohort. Models that complete every trial still become less economical at longer horizons, while higher efficiency among completed plans can coexist with poor goal attainment. SPE separates the overall capability hier
    
[^177]: 专家岩性标签用于预测海洋沉积物性质时并不能改善学习效果

    Expert lithological labels do not improve learning when used to predict marine sediment properties

    [https://arxiv.org/abs/2610.36410](https://arxiv.org/abs/2610.36410)

    专家标注的岩性标签本身可从常规物理测量数据中恢复，因此在预测海洋沉积物性质的自监督学习中引入这些标签并不能带来性能提升。

    

    科学钻探已经产生了大量与回收岩石描述相关联的物理、化学和磁性测量数据档案。这些档案大多是不完整的：仪器以不同间隔采样，岩芯回收率是部分的，且破坏性分析使得无法对同一材料进行后续测量。我们在LILY海洋钻探数据库上训练了一个自监督嵌入模型，使用了六种常规测井属性（体积密度、磁化率、自然伽马辐射以及红、绿、蓝反射率），涵盖来自296个钻孔的236,963个观测数据，模型采用带有异方差高斯解码器和可学习掩码标记的掩码变分自编码器。结果显示，这些嵌入对地质学家所指定岩性的区分能力与使用这些标签训练的变体相当；而且增加标签的权重会降低重建质量，同时分类性能保持不变。这些标签可以从测量数据中恢复出来。

    arXiv:2610.36410v1 Announce Type: cross  Abstract: Scientific drilling has produced large archives of physical, chemical and magnetic measurements linked to descriptions of the rock recovered. These archives are mostly incomplete: instruments sample at different intervals, recovery is partial, and destructive analyses preclude later measurement of the same material. We train an self-supervised embedding model on the LILY ocean drilling database using six routinely logged properties (bulk density, magnetic susceptibility, natural gamma radiation, and red, green and blue reflectance) across 236,963 observations from 296 boreholes, with a masked variational autoencoder that has a heteroscedastic Gaussian decoder and learnable mask tokens. The embeddings separate geologist-assigned lithologies as well as a variant trained with those labels, and increasing the weight on the labels degrades reconstruction while leaving classification unchanged. The labels are recoverable from the measurement
    
[^178]: kw-bench：用真实世界任务击穿前沿智能体运行框架

    kw-bench: Breaking Frontier Harnesses on Real-World Tasks

    [https://arxiv.org/abs/2610.36405](https://arxiv.org/abs/2610.36405)

    kw-bench 是基于真实业务委托构建的基准，首次将评估重点从生成式任务转向审计、税务等知识工作中的调查性环节，检验前沿智能体运行框架能否从不完整、不一致的分散证据中精确还原并论证答案。

    

    每一代新的前沿模型都在不断拓展可能性的边界。在一个运行框架（harness）中——包含一个循环、一个 shell、一个文件系统，以及编写并运行自身代码的能力——模型能够承担越来越困难的工作。越来越多的研究将编码智能体的运行框架视为知识工作的基础设施，而非单纯的编码工具，新一波的基准测试也正是在这一场景下对其进行检验。这些基准大多是生成式的：根据简报构建财务模型、备忘录或电子表格，并由评分标准或另一个模型来评判。然而，很少有基准去测试知识工作中最昂贵的部分，即调查性工作。在审计、税务、尽职调查、监管报告、理赔或任何涉及数字的争议中，答案其实早已存在——只是隐藏在散布于各个系统之中、不完整且相互矛盾的证据里——必须将其还原出来，并精确到分地加以论证。我们推出了 kw-bench，这是一个基于专业人士向我们描述的真实业务委托所构建的基准。它的第一个任务族是一个五……（摘要在此处截断）

    arXiv:2610.36405v1 Announce Type: new  Abstract: Each new generation of frontier model pushes the edge of what is possible. Inside a harness -- a loop, a shell, a filesystem, the ability to write and run its own code -- a model can take on increasingly difficult work.   A growing body of work treats the coding-agent harness as infrastructure for knowledge work rather than a coding tool, and a new wave of benchmarks tests it there. Most of them are generative: build a financial model, a memo, a spreadsheet from a brief, judged by a rubric or by another model. Few test the expensive part of knowledge work, which is investigative. In audit, tax, due diligence, regulatory reporting, claims or any dispute over numbers, the answer already exists -- hidden in incomplete and inconsistent evidence spread across systems -- and has to be recovered and defended to the cent.   We introduce kw-bench, a benchmark built from real engagements specialists described to us. Its first task family is a five
    
[^179]: MAREval：一个用于评估自然语言推荐解释的多智能体框架

    MAREval: A Multi-Agent Framework for Evaluating Natural Language Recommendation Explanations

    [https://arxiv.org/abs/2610.36389](https://arxiv.org/abs/2610.36389)

    MAREval提出了一种结构化多智能体框架，结合辩论链提示策略、主持人调节、仲裁者聚合与蒙特卡洛采样，实现了对推荐系统中自然语言解释的可扩展、一致且自动化的评估。

    

    arXiv:2610.36389v1 公告类型：新 摘要：评估推荐系统中的自然语言解释对于培养用户信任、透明度和参与度至关重要。然而，现有的评估方法（如人工评估）虽然准确，但资源消耗大，在现代推荐平台所需的规模下并不实用。此外，使用单智能体大语言模型的自动化方法存在提示词敏感性和输出不一致的问题。为应对这些挑战，我们提出了MAREval，一个使用大语言模型评估推荐解释的结构化多智能体框架。MAREval协调以下组件：（i）一个规划者智能体，采用新颖的辩论链提示策略来协调智能体角色并确保评估计划逻辑一致；（ii）一个主持人智能体，通过缓解提示漂移来调节讨论；（iii）一个仲裁者智能体，聚合多轮评估的输出；以及（iv）一种蒙特卡洛采样方法。

    arXiv:2610.36389v1 Announce Type: new  Abstract: Evaluating natural language explanations in recommender systems is essential for fostering user trust, transparency, and engagement. However, existing evaluation approaches like human evaluations, while accurate, are resource-intensive and impractical at the scale required by modern recommendation platforms. Also, automated methods using single-agent LLMs suffer from prompt sensitivity and inconsistent outputs. To address these challenges, we propose MAREval, a structured multi-agent framework for evaluating recommendation explanations using large language models. MAREval orchestrates (i) a planner agent that uses a novel Chain of Debate (CoD) prompting strategy to coordinate agent roles and enforce logically consistent evaluation plans; (ii) a moderator agent that regulates discussions by mitigating prompt drift; and (iii) an arbitrator agent that aggregates outputs from multiple evaluation rounds and (iv) a Monte Carlo sampling method,
    
[^180]: 预算受限下的LoRA：面向适配器集群的功能感知事后压缩

    LoRA on a Budget: Function-Aware Post-Hoc Compression for Adapter Fleets

    [https://arxiv.org/abs/2610.36386](https://arxiv.org/abs/2610.36386)

    该论文提出功能秩分配（FRA）方法，通过为每个LoRA适配器构建功能秩-失真前沿，并在共享秩预算下进行集群级极小极大分配，从而解决多适配器事后压缩中的两级秩分配问题。

    

    低秩适应（LoRA）已成为将基础模型高效专化为众多任务特定与用户特定变体的关键机制。在共享资源预算下服务大量LoRA适配器，使得事后压缩成为一个两级分配问题：秩需要在每个适配器内部的各层之间、以及整个适配器集群内的各适配器之间进行分配。尽管训练好的LoRA包含大量的谱冗余，但仅凭谱重构并不能确定一个适配器在功能上能够容忍多大程度的压缩。我们通过为每个适配器构建功能性的秩-失真前沿（rank-distortion frontier），并在共享秩预算下进行集群级的极小极大分配，来表述集群压缩问题。在给定精确前沿的情况下，这种分解等价于对所有层秩的联合优化。我们提出了功能秩分配（Functional Rank Allocation，FRA），这是一种无需标注的实用前沿构建方法。FRA通过测量孤立模块截断的端……（摘要在此处截断）

    arXiv:2610.36386v1 Announce Type: new  Abstract: Low-Rank Adaptation (LoRA) has become a key mechanism for efficiently specializing foundation models into many task- and user-specific variants. Serving many LoRA adapters under a shared resource budget turns post-hoc compression into a two-level allocation problem: rank must be distributed across layers within each adapter and across adapters within the fleet. Although trained LoRAs contain substantial spectral redundancy, spectral reconstruction alone does not determine how much compression an adapter can functionally tolerate. We formulate fleet compression through a functional rank-distortion frontier for each adapter, followed by fleet-level minimax allocation under a shared rank budget. Given exact frontiers, this decomposition is equivalent to joint optimization over all layer ranks. We introduce Functional Rank Allocation (FRA), an unlabeled method for constructing practical frontiers. FRA measures isolated module truncations end
    
[^181]: SOMA：一个用于细粒度自我中心行为与身体运动理解的基准

    SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding

    [https://arxiv.org/abs/2610.36384](https://arxiv.org/abs/2610.36384)

    该论文提出了SOMA基准，通过1,632个经人工验证的问题从身体构型、运动动态和时间推理三个维度评估自我中心视频中的细粒度身体运动理解能力，发现37个领先的MLLM表现不佳，甚至落后于同样认为该任务困难的人类。

    

    感知人体运动对于理解和辅助人类的多模态系统至关重要。然而，当前对多模态大语言模型（MLLM）的评估主要集中于物体、场景和粗粒度动作，模型对身体本身进行推理的能力尚未得到充分评估。这从自我中心（第一人称）视角来看尤其具有挑战性，因为在自我中心视角下，身体仅部分可见，运动必须从被遮挡且在时间上分散的线索中推断出来。我们提出了SOMA，一个用于从自我中心视频中理解细粒度身体运动的基准，包含1,632个经人工验证的问题，涵盖三个能力维度：身体构型、运动动态和时间推理。我们评估了37个领先的闭源、开源以及专门针对自我中心的MLLM，并辅以大规模的人类评估。我们发现，仅凭视频完成该任务对人类而言也很困难（准确率为68.5%），但人类的表现仍明显优于（模型）。

    arXiv:2610.36384v1 Announce Type: cross  Abstract: Perceiving human body motion is critical for multimodal systems that understand and assist people. Yet current evaluations of Multimodal Large Language Models (MLLMs) largely focus on objects, scenes, and coarse actions, leaving their ability to reason about the body itself poorly assessed. This is especially challenging from the egocentric perspective, in which the body is only partially visible and motion must be inferred from occluded and temporally distributed cues. We introduce SOMA, a benchmark for fine-grained body-motion understanding from egocentric video, comprising 1,632 human-verified questions across three capability axes: body configuration, motion dynamics, and temporal reasoning. We evaluate 37 leading closed-source, open-source, and specialized egocentric MLLMs, complemented by an extensive human evaluation. We find that humans also find this task difficult from video alone (68.5% accuracy), yet clearly outperform the 
    
[^182]: HIDRA：面向无回放终身模仿学习的层次化双路由注意力机制

    HIDRA: Hierarchical Dual-Routing Attention for Replay-Free Lifelong Imitation Learning

    [https://arxiv.org/abs/2610.36374](https://arxiv.org/abs/2610.36374)

    提出HIDRA框架，通过层次化双路由注意力机制将指令级专家检索与上下文细化解耦，并利用键级几何正则化稳定路由，从而解决无回放终身模仿学习中的路由干扰问题。

    

    在真实世界环境中运行的机器人智能体必须从多模态示范流中持续适应，且通常受到无法存储或重新访问历史数据的约束。在这种无回放、单次遍历的终身模仿学习设置中，顺序更新会逐渐扭曲表示空间，使得基于专家的方法对路由干扰尤为敏感，导致专家选择能力随时间退化。我们提出了HIDRA，这是一个通过层次化双注意力机制来缓解该问题的结构化路由框架，该机制将指令级专家检索与依赖上下文的细化过程解耦。为了在表示漂移下进一步稳定路由，HIDRA引入了键级几何正则化，同时强制实现任务内对齐与任务间分离。由此得到的方法能够在支持可扩展专家集的同时实现可靠的专家选择。

    arXiv:2610.36374v1 Announce Type: cross  Abstract: Robotic agents operating in real-world environments must adapt continuously from a stream of multimodal demonstrations, often under constraints that preclude storing or revisiting past data. In this replay-free, single-pass lifelong imitation learning setting, sequential updates can progressively distort the representation space, making expert-based approaches particularly sensitive to routing interference, causing expert selection to degrade over time. We propose HIDRA, a structured routing framework that mitigates this issue through a hierarchical dual-attention mechanism, which decouples instruction-level expert retrieval from context-dependent refinement. To further stabilize routing under representation drift, HIDRA introduces a key-level geometric regularization that enforces both alignment within tasks and separation across tasks. The resulting approach enables reliable expert selection while supporting an expandable set of expe
    
[^183]: 基于锚点的序列、多模态与状态更新数字孪生模型比较框架的可靠性感知短期血糖风险预测

    Reliability-Aware Short-Term Glycemic Risk Prediction Using an Anchor-Based Comparative Framework of Sequence, Multimodal, and State-Updating Digital-Twin Models

    [https://arxiv.org/abs/2610.36363](https://arxiv.org/abs/2610.36363)

    本研究提出基于锚点的比较基准框架，利用736名1型糖尿病患者的550余万个标准化预测锚点，在统一的患者级评估协议下比较混合LSTM-Transformer、多模态数字孪生与状态更新数字孪生三种神经架构在30/60/120分钟血糖风险预测中的表现，并同时评估校准等可靠性指标。

    

    预测短期低血糖和高血糖风险对于糖尿病的主动管理至关重要。然而，现有的机器学习方法通常采用不一致的评估协议，并且过分强调预测性能而忽视可靠性。本研究引入了一个整合纵向血糖序列与多模态患者信息的基于锚点的基准测试框架。利用T1DiabetesGranada数据集，从736名1型糖尿病患者中构建了5,503,441个标准化预测锚点，用于评估30分钟、60分钟和120分钟时间范围内的低血糖和高血糖预测。在完全相同的患者级验证协议下，比较了三种神经架构：混合LSTM-Transformer、多模态数字孪生，以及具有间隙感知纵向状态推进的状态更新数字孪生。除预测性能外，评估还考察了校准、分类阈值等可靠性指标。

    arXiv:2610.36363v1 Announce Type: cross  Abstract: Anticipating short-term hypoglycemic and hyperglycemic risk is essential for proactive diabetes management. However, existing machine learning approaches often use inconsistent evaluation protocols and emphasize predictive performance over reliability. This study introduces an anchor-based benchmarking framework integrating longitudinal glucose sequences and multimodal patient information. Using the T1DiabetesGranada dataset, 5,503,441 standardized prediction anchors were constructed from 736 individuals with type 1 diabetes to evaluate hypoglycemia and hyperglycemia prediction at 30-, 60-, and 120-minute horizons. Three neural architectures, a Hybrid LSTM-Transformer, a Multimodal Digital Twin, and a State-Updating Digital Twin with gap-aware longitudinal state progression, were compared under identical patient-level validation protocols. Beyond predictive performance, the evaluation examined calibration, classification thresholds, ex
    
[^184]: WAKMan：工作负载感知的KV缓存管理器

    WAKMan: Workload Aware KVCache Manager

    [https://arxiv.org/abs/2610.36355](https://arxiv.org/abs/2610.36355)

    该论文提出了工作负载感知的KV缓存管理器WAKMan，通过分析不同类型请求（代理式请求偏向最近性、热门提示和长对话偏向频率）对缓存淘汰策略的不同需求，解决了没有任何单一淘汰策略能在所有LLM工作负载下始终保持最优的问题。

    

    前缀缓存使大语言模型（LLM）服务器能够重用键值（KV）状态，并为与先前提示共享前缀的请求加速预填充计算阶段，从而降低延迟和服务成本。在内存资源有限的情况下，KV缓存淘汰策略成为决定前缀缓存整体收益的关键因素。生产流量可能包含单轮、多轮、代理式或这些请求类型的任意混合组合。我们的研究结果表明，代理式步骤会在几秒内返回并偏向最近性，而热门提示和长对话则在许多其他请求之后才返回并偏向频率。在20种混合负载中，没有任何单一策略总是最优：vLLM的默认策略LRU在代理式请求为主的混合负载上比面向频率的S3FIFO提供了高12%的缓存命中率，但在其他情况下最多落后达22%，甚至单个工作负载参数就可能改变最优策略的选择……

    arXiv:2610.36355v1 Announce Type: new  Abstract: Prefix caching enables a Large Language Model (LLM) servers to reuse the Key-Value (KV) states and accelerate prefill computation stage for requests that share a prefix with earlier prompts, thus reducing latency and serving cost. With limited memory resource, the KV cache eviction policy becomes a decisive factor in determining the overall benefits of prefix caching. Production traffic may have single-turn, multi-turn, agentic, or a mixture of any combination of these request types. Our findings indicate that agentic steps return within seconds and favor recency, while popular prompts and long conversations return after many other requests and favor frequency. Across 20 mixes, no single policy is always best: vLLM's default strategy LRU provides 12% higher cache hits as compared to the frequency-oriented S3FIFO on agentic-heavy mixes but trails it by up to 22% elsewhere, and even a single workload parameter can flip the winner even for 
    
[^185]: 基于深度表征学习的分布外检测增强DNA甲基化数据临床分类器的鲁棒性

    Enhancing Clinical Classifier Robustness with Deep Representation Learning Based OOD Detection on DNA Methylation Data

    [https://arxiv.org/abs/2610.36344](https://arxiv.org/abs/2610.36344)

    该论文提出了一种端到端的基于深度表征学习的分布外（OOD）检测框架，通过从深度神经网络瓶颈层提取特征向量并结合五种OOD检测方法识别训练分布之外的肿瘤类型，从而有效增强DNA甲基化阵列临床肿瘤分类器的鲁棒性。

    

    基于DNA甲基化（DNAm）阵列的肿瘤分类是一种强大的临床诊断工具，但在固定参考队列上训练的模型可能会对训练分布中未包含的肿瘤类型产生错误的预测。我们提出了一种端到端的、基于深度表征学习的DNA甲基化阵列数据分布外（OOD）检测框架。深度神经网络（DNN）分类器在公开的肉瘤（SARC）（GSE140686）和脑肿瘤（BT）（GSE90496）队列上进行训练，并从瓶颈层提取特征向量。研究探索了五种OOD方法，包括一维子空间分析（1D-SSA）、K近邻（KNN）、岭回归、马氏距离（MD）和平方距离和（SSD），并采用了200维和500维的嵌入表示。谱差异（Spectral discrepancy）被用于区分分布内（IND）样本和分布外（OOD）样本。性能通过交叉验证、留出样本以及独立的（外部数据集）进行评估。

    arXiv:2610.36344v1 Announce Type: cross  Abstract: DNA methylation (DNAM) array-based tumor classification is a powerful clinical diagnostic tool, but models trained on fixed reference cohorts can produce faulty predictions for tumor types not represented in the training distribution. We propose an end-to-end deep representation learning-based out-of-distribution (OOD) detection framework for DNAm array data. Deep neural network (DNN) classifiers were trained on public sarcoma (SARC) (GSE140686) and brain tumor (BT) (GSE90496) cohorts, and feature vectors were extracted from bottleneck layers. Five OOD methods, including 1dimension (1D)-Subspace Analysis (1D-SSA), K-Nearest Neighbors (KNN), Ridge Regression, Mahalanobis Distance (MD), and Sum of Squared Distances (SSD), were explored with 200 and 500-dimensional embeddings. Spectral discrepancy was used to discriminate in-distribution (IND) from OOD samples. Performance was measured via cross-validation, holdout samples, and an indepen
    
[^186]: 对建模编程的程序进行可执行考古学：让启发式编译器（1961）与KBEmacs（1984）重获新生

    Executable Archaeology of Programs That Model Programming: Reanimating the Heuristic Compiler (1961) and KBEmacs (1984)

    [https://arxiv.org/abs/2610.36341](https://arxiv.org/abs/2610.36341)

    本文运用“可执行考古学”方法复原并运行了两个早期AI程序——赫伯特·西蒙的启发式编译器（1961）与MIT的KBEmacs（1984），以此重现并比较“编程即问题求解”与“编程即组合惯用模式”这两种关于人类如何编程的经典理论。

    

    早期的人工智能程序往往也是关于人类如何思考的理论，而编写程序的程序在某种程度上也是关于人类如何编程的理论。本文报告了对两个此类程序的复兴工作，它们提供了两种看似不同的编程理论，且开发目的截然不同。第一个是赫伯特·西蒙的启发式编译器（Heuristic Compiler，1961年），它是一个将编程视为问题求解的实验模型，应用通用问题求解器的手段-目的分析方法来探究编程任务的本质。第二个是KBEmacs（1984年），它是由麻省理工学院“程序员学徒”项目开发的智能编程助手的可运行原型，建立在专家程序员会识别并组合熟悉模式（即“惯用套语”）的观点之上。它使程序员能够通过一种风格化的英语命令语言组合惯用模式来构建程序。我们通过可执行考古学的方法论来研究这两个程序……

    arXiv:2610.36341v1 Announce Type: new  Abstract: Early AI programs were often also theories of how people think, and programs that wrote programs were, in part, theories of how people program. Here we report on the reanimation of two such programs that offer two seemingly different theories of programming and were developed for quite different purposes. The first, Herbert Simon's Heuristic Compiler (1961), was an experimental model of programming as problem-solving, applying the means-ends analysis methods of the General Problem Solver to investigate the nature of the programming task. The second, KBEmacs (1984), was a working prototype of an intelligent programming assistant developed by MIT's Programmer's Apprentice project, built on the view that expert programmers recognize and combine familiar patterns, or "cliches." It let a programmer build up programs by combining cliches through a stylized English command language. We approach both through the methodology of Executable Archaeo
    
[^187]: CrowdCC：人群计数中损坏鲁棒性与部署量化的联合基准测试

    CrowdCC: Jointly Benchmarking Corruption Robustness and Deployment Quantization in Crowd Counting

    [https://arxiv.org/abs/2610.36340](https://arxiv.org/abs/2610.36340)

    该论文提出CrowdCC基准，首次联合评估人群计数模型在图像损坏退化与部署量化（FP32/FP16/INT8）下的表现，发现采集链类损坏对所有模型造成的性能下降最为严重。

    

    部署在边缘设备上的人群计数模型面临两种变化：输入在采集和传输管道中发生降质，同时模型需要被量化以满足资源限制。然而，损坏鲁棒性与部署量化通常被分开评估。我们提出了CrowdCC，一个受控损坏基准，联合评估这些影响：使用23种面向部署场景的损坏类型（涵盖三个严重程度）、四个人群计数数据集和26个模型，采用A100 TensorRT引擎在FP32、FP16和训练后INT8精度下运行，并搭配相应Jetson Orin NX配置的吞吐量和能耗测量。这些损坏涵盖六个类别，并在几何或可见性发生变化时对头部标注进行调整。我们的评估显示，采集链损坏在所有评估模型中始终造成最大的性能下降，而人群密度对绝对误差和相对误差的影响各不相同。

    arXiv:2610.36340v1 Announce Type: cross  Abstract: Crowd-counting models deployed on edge devices face two shifts: inputs are degraded along the capture and transmission pipeline, while models are quantized to satisfy resource constraints. Yet corruption robustness and deployment quantization are typically evaluated separately. We introduce CrowdCC, a controlled corruption benchmark that jointly evaluates these effects using 23 deployment-motivated corruptions at three severity levels, four crowd-counting datasets, and 26 models using A100 TensorRT engines at FP32, FP16, and post-training INT8 precision, paired with throughput and energy measurements from corresponding Jetson Orin NX configurations. The corruptions span six families, with head annotations adjusted when geometry or visibility changes. Our evaluation shows that capture-chain corruptions consistently produce the largest degradation across the evaluated models, while crowd density affects absolute and relative error differ
    
[^188]: OA-RAG：基于本体的智能体RAG，用于忠实的大语言模型推理

    OA-RAG: Ontology-based Agentic RAG for Faithful LLM Reasoning

    [https://arxiv.org/abs/2610.36335](https://arxiv.org/abs/2610.36335)

    该论文提出OA-RAG，一种基于本体的智能体RAG方法，将形式化推理与大语言模型推理相结合，并通过报告本体证据观察和模型评估的依据水平，提升医学问答中答案的忠实性。

    

    大语言模型（LLM）在许多现实世界任务中表现良好，但仍然容易产生幻觉，这限制了它们在生物医学和医疗保健等领域的可靠性，因为这些领域的答案必须建立在已有知识的基础之上。检索增强生成（RAG）通过引入外部信息来缓解这一问题，但往往忽视逻辑关系，并且严重依赖大语言模型自身的推理，这可能损害答案的忠实性。我们提出了OA-RAG，这是一个将形式化推理与大语言模型推理相结合的智能体，它同时利用领域本体的逻辑结构和非形式化注释。对于每个答案，OA-RAG会报告用作证据的本体观察结果，以及一个由模型评估的依据水平（可推导、受支持、相关或不相关），用以刻画该证据的强度。在基于三个大规模本体和六个真实世界医学问答任务构建的基准测试上

    arXiv:2610.36335v1 Announce Type: new  Abstract: Large language models (LLMs) perform well across many real-world tasks but remain prone to hallucination, limiting their reliability in domains such as biomedicine and healthcare, where answers must be grounded in established knowledge. Retrieval-augmented generation (RAG) mitigates this problem by incorporating external information, but often overlooks logical relationships and relies heavily on the LLM's own reasoning, which can compromise answer faithfulness. We propose OA-RAG, an agent that integrates formal reasoning with LLM reasoning, drawing on both the logical structure and informal annotations of domain ontologies. For each answer, OA-RAG reports the ontology observations used as evidence and a model-assessed grounding level (entailed, supported, related, or unrelated) that characterises the strength of that evidence. Across benchmarks derived from three large-scale ontologies and six real-world medical question-answering tasks
    
[^189]: 中间仍有大量空间：通过细粒度监控揭示GPU集群中的机会

    Plenty of Room in the Middle: Revealing Opportunities on a GPU Cluster via Fine-Grained Monitoring

    [https://arxiv.org/abs/2610.36329](https://arxiv.org/abs/2610.36329)

    本文开发了一款基于DCGM库的轻量级细粒度（100毫秒级）GPU监控工具，并通过对大学GPU集群长达数月的监测数据分析，揭示了以往粗粒度监控无法发现的GPU利用率提升机会。

    

    GPU集群已成为学术界和工业界AI与科学研究的重要计算资源。然而，其高昂的成本和不断增长的需求往往导致长时间的排队等待。以往的监控工作通常以秒级或分钟级的频率运行，主要服务于集群的作业级调度，缺乏优化GPU级利用率所需的细节。本文对某大学研究型GPU集群进行了深入研究，以100毫秒的间隔分析资源使用情况。我们开发了一款轻量级监控工具，利用Nvidia的DCGM库捕获GPU内部细粒度的“计算利用率”指标。该工具部署在一个拥有89个节点和319个GPU的集群上，收集了数月的数据。本文重点关注一个为期10天的时间段，其间来自12个院系的150名用户共提交了7,399个作业。我们的研究揭示了三个关键发现：(1) 该GPU集群存在显著的（摘要在此处截断，后续内容未提供）

    arXiv:2610.36329v1 Announce Type: new  Abstract: GPU clusters have become essential computing resources for AI and scientific research in academia and industry. However, their high cost and growing demand often result in long queueing delays. While previous monitoring efforts have operated at second- or minute-level frequencies, they primarily informed job-level scheduling for clusters and lacked the detail needed to optimize GPU-level utilization.   This paper presents an in-depth study of a university research GPU cluster, analyzing resource usage at 100 ms intervals. We have developed a lightweight monitoring tool that leverages Nvidia's DCGM library to capture fine-grained "compute utilization" metrics within GPUs. Deployed on a cluster with 89 nodes and 319 GPUs, our tool collected data over several months. This paper focuses on a 10-day period with 7,399 jobs submitted by 150 users from 12 departments. Our study reveals three key findings: (1) the GPU cluster exhibits significant
    
[^190]: 程序重构下大语言模型生成的证明有多稳定？可验证代码生成的压力测试

    How Stable Are LLM-Generated Proofs Under Program Refactoring? A Stress Test for Verifiable Code Generation

    [https://arxiv.org/abs/2610.36328](https://arxiv.org/abs/2610.36328)

    该论文提出 RefactorProof 基准，通过对 185 个 VERINA 任务生成 458 个经 Lean 认证的行为保持重构变体，揭示了大语言模型生成的证明在程序结构重构下存在显著脆弱性。

    

    arXiv:2610.36328v1 公告类型：新。摘要：可验证代码生成的基准测试通常只针对一个固定的实现来验证证明。我们提出这样一个问题：在保持程序行为不变的重构之后，同一个证明是否仍然有效？我们提出了 RefactorProof，它从在我们固定环境中可编译的 185 个 VERINA 任务中推导出 458 个经 Lean 认证的变体，同时将规范说明和被评估的证明逐字节保持不变。VERINA 为 46 个任务提供了参考证明，这些任务全部来自其基础分割集；主要的模型与参考证明的对比进一步匹配在 21-25 个简短的非递归任务上。结构性重构在各个证明来源中暴露了显著的脆弱性：任务归一化的结构 PSR（证明存活率）在提供的参考证明上为 4.3%，而在专用证明器上为 13.3%-26.8%。在匹配的任务上，专用证明器生成的证明总体上仍比提供的参考证明工件具有更高的未变更证明存活率：DeepSeek-Prover-V2-7B 高出 +15.6 个百分点，+18.8（原文截断）

    arXiv:2610.36328v1 Announce Type: new  Abstract: Benchmarks for verifiable code generation usually test a proof against one fixed implementation. We ask whether the same proof remains valid after a program refactoring that preserves behavior. We introduce RefactorProof, which derives 458 Lean-certified variants from the 185 VERINA tasks that compile in our pinned environment while holding the specification and evaluated proof fixed byte-for-byte. VERINA supplies reference proofs for 46 tasks, all from its basic split; the primary model-versus-reference comparisons further match on 21-25 short, non-recursive tasks. Structural refactorings expose substantial brittleness across proof sources: task-normalized structural PSR is 4.3% for the supplied references and 13.3-26.8% for the specialized provers. On the matched tasks, specialized prover proofs nevertheless have higher overall unchanged-proof survival than the supplied reference artifacts: +15.6 points for DeepSeek-Prover-V2-7B, +18.8
    
[^191]: 本体引导的多智能体推理用于可解释的鉴别诊断

    Ontology-guided multi-agent reasoning for explainable differential diagnosis

    [https://arxiv.org/abs/2610.36325](https://arxiv.org/abs/2610.36325)

    该论文提出了一种本体引导的多智能体鉴别诊断框架，通过复现医生的迭代推理过程，使诊断结论建立在临床证据之上并具备可解释性。

    

    有效的鉴别诊断依赖于结构良好且无歧义的患者病例描述，因为不完整或模糊的临床信息容易导致诊断错误和推理不一致。大型语言模型（LLMs）最近在医学问答和临床推理方面展现出了有前景的能力；然而，即使它们在医学基准测试中取得了有竞争力的表现，它们往往依赖于从训练数据中学习到的模式识别，而非基于临床证据的推理，并且可能生成看似合理但缺乏依据的诊断结论。它们的推理过程在很大程度上仍然是不透明的，这使得临床医生难以评估给定建议背后的证据和依据。为了弥补这一差距，我们提出了一个本体引导的多智能体鉴别诊断框架，该框架复现了医生的迭代推理过程。该框架将诊断推理建立在一个结构化（原文在此处截断）

    arXiv:2610.36325v1 Announce Type: new  Abstract: Effective differential diagnosis depends on well structured and unambiguous patient case descriptions, since incomplete or ambiguous clinical information readily leads to diagnostic errors and inconsistent reasoning. Large Language Models (LLMs) have recently shown promising capabilities in medical question answering and clinical reasoning; however, even when achieving competitive performance on medical benchmarks, they often rely on pattern recognition learned from training data rather than grounded clinical evidence, and can generate plausible yet unsupported diagnostic conclusions. Their reasoning remains largely opaque, making it difficult for clinicians to assess the evidence and rationale behind a given recommendation. To address this gap, we propose an ontology-guided, multiagent differential diagnosis framework that replicates the iterative reasoning process of physicians. The framework grounds diagnostic inference in a structure
    
[^192]: 当检索还不够：智能体记忆中的过时事实

    When Retrieval Is Not Enough: Obsolete Facts in Agent Memory

    [https://arxiv.org/abs/2610.36308](https://arxiv.org/abs/2610.36308)

    该论文揭示智能体记忆系统的失败主要并非源于检索缺失，而是源于无法应对过时事实——即使最新信息已被成功检索，一条过时的竞争性陈述仍会使模型准确率大幅下降（Qwen2.5 从 100% 跌至 44.6%）。

    

    记忆智能体基准测试通常将检索与答案生成合并为单一分数，因此难以判断失败究竟源于错失更新，还是源于未能使用已经检索到的更新。我们在 MemoryAgentBench 的单跳 FactConsolidation 任务上考察了这一区别。使用 Qwen2.5-7B-Instruct 在 6K 长度历史上，当检索证据预算从 64 增加到 512 个 token 时，BM25 的准确率从 45% 降至 32%——即便借助标准答案进行的事后审计发现每个检索集合中都包含最新陈述。大多数错误出现在同时含有过时陈述版本的问题上。在固定提示词和相同的四条陈述证据规模下，用过时的竞争陈述替换一条无关陈述，会使 Qwen2.5 的准确率从 100% 降至 44.6%，Qwen3-14B 从 100% 降至 87.8%，且全部 50 个有害的配对翻转均被归类为过时所致。在 32K 历史上的复现实验同样显示 BM25-64 优于……（原文摘要在此处截断）

    arXiv:2610.36308v1 Announce Type: new  Abstract: Memory-agent benchmarks often collapse retrieval and answer generation into one score, making it hard to tell whether a failure comes from missing an update or from not using an update that was already retrieved. We examine this distinction on the single-hop FactConsolidation task of MemoryAgentBench. With Qwen2.5-7B-Instruct on 6K-history BM25 accuracy drops from 45% to 32% when the retrieved evidence budget grows from 64 to 512 tokens, even though a gold-assisted post-hoc audit finds the newest statement in every retrieved set. Most of the errors occur on questions that also contain an obsolete statement version. Under a frozen prompt and equal four-statement evidence size, replacing one unrelated statement with the obsolete competing statement reduces accuracy from 100% to 44.6% for Qwen2.5 and from 100% to 87.8% for Qwen3-14B, with all 50 harmful paired flips classified as stale. A 32K-history replication likewise favors BM25-64 over
    
[^193]: 脉冲神经网络更难被窃取：模型窃取攻击的基准研究

    Spiking Neural Networks Are Harder to Steal: A Benchmark Study of Model Stealing Attacks

    [https://arxiv.org/abs/2610.36298](https://arxiv.org/abs/2610.36298)

    本研究通过基准测试四种代表性的模型窃取攻击，证明由于脉冲神经网络采用基于稀疏离散脉冲和时变膜电位的时间计算模式，其决策行为比传统人工神经网络更难通过黑盒查询被窃取重建。

    

    arXiv:2610.36298v1 公告类型：new 摘要：部署在查询接口背后的机器学习模型所暴露的不仅仅是其预测结果，因为攻击者可以利用这些预测来重建模型本身。模型窃取攻击（MSAs）正是利用了这种暴露，仅通过黑盒查询训练一个替代模型来复制目标模型的决策行为，而无需访问其参数或训练数据。这类攻击在传统人工神经网络（ANNs）上已被充分研究，然而脉冲神经网络（SNNs）以一种根本不同的方式处理信息。SNNs通过稀疏、离散的脉冲进行通信，其输出依赖于随时间演化的膜电位，而非从输入到输出的静态映射。这种时间维度的计算模式表明，SNN的决策行为可能更难从其响应中重建。本工作通过对四种代表性的MSA进行基准测试来验证这一可能性，即……

    arXiv:2610.36298v1 Announce Type: new  Abstract: Machine learning models deployed behind query interfaces expose more than their predictions, because an adversary can use those predictions to rebuild the model itself. Model stealing attacks (MSAs) exploit this exposure by training a surrogate model that replicates the decision behavior of a target through black-box queries alone, without access to its parameters or training data. These attacks are well characterized for conventional artificial neural networks (ANNs), yet spiking neural networks (SNNs) process information in a fundamentally different way. SNNs communicate through sparse, discrete spikes, and their outputs depend on membrane potentials that evolve over time rather than on a static mapping from input to output. This temporal mode of computation suggests that the decision behavior of an SNN may be harder to reconstruct from its responses. This work tests that possibility by benchmarking four representative MSAs, namely the
    
[^194]: 检测与缓解基于查询的系统提示词泄露攻击

    Detecting and Mitigating Query-Based System Prompt Leakage Attacks

    [https://arxiv.org/abs/2610.36295](https://arxiv.org/abs/2610.36295)

    提出了模型无关的AdvGuard框架，通过有界语义交互图建模跨查询关系并利用图卷积网络捕捉多跳模式，实现对基于查询的系统提示词泄露攻击的有效检测与缓解。

    

    大语言模型（LLM）依赖专有的系统提示词来执行任务指令、安全策略和访问控制。这些提示词容易受到基于查询的泄露攻击，即黑盒攻击者通过自适应的多轮交互提取机密指令。现有防御通常独立评估每条查询，无法捕捉跨交互中才显现的恶意行为。我们提出了AdvGuard，一个用于检测和缓解系统提示词泄露的模型无关框架。AdvGuard利用有界语义交互图对跨查询关系进行建模，通过统计方法过滤候选组件，并将其转换为线图以进行图卷积网络（GCN）分类。该方法能够捕捉查询相似性中的多跳模式，从而将协调式的机密提取行为与良性工作流区分开来。轻量级的提示词级检测器能够在交互历史尚不充分时便提供保护。

    arXiv:2610.36295v1 Announce Type: cross  Abstract: Large Language Models (LLMs) rely on proprietary system prompts to enforce task directives, safety policies, and access controls. These prompts are vulnerable to query-based leakage attacks, where black-box adversaries extract confidential instructions through adaptive, multi-turn interactions. Existing defenses often assess queries independently, missing malicious behavior that emerges across interactions.   We present AdvGuard, a model-agnostic framework for detecting and mitigating system prompt leakage. AdvGuard models cross-query relationships using a bounded semantic interaction graph, statistically filters candidate components, and converts them into line graphs for graph convolutional network (GCN) classification. This approach captures multi-hop patterns in query similarity that distinguish coordinated extraction from benign workflows. A lightweight prompt-level detector provides protection before sufficient interaction histor
    
[^195]: BUSTA：教会小型语言模型对二进制执行进行推理

    BUSTA: Teaching Small Language Models to Reason About Binary Execution

    [https://arxiv.org/abs/2610.36288](https://arxiv.org/abs/2610.36288)

    通过“逆向基本块执行”训练任务结合GRPO强化学习微调，小型开源模型（1.7B级别）在二进制逆向推理能力上可逼近前沿大模型，从而能在资源受限的沙箱逆向工程环境中高效部署。

    

    大语言模型（LLM）正被越来越多地应用于逆向工程（RE）工作流程中。然而，逆向工程与常见的LLM编程任务不同，它要求模型对底层程序行为进行推理。此外，在资源受限的沙箱化逆向工程环境中部署工具，限制了前沿大规模模型解决方案的使用能力。我们研究了一种针对性的LLM训练方法能否通过逆向基本块执行任务，将独特的逆向工程技能迁移到开源模型上：给定一个x86-64基本块和期望的输出寄存器状态，模型需要预测能产生该状态的输入寄存器值。使用群体相对策略优化（GRPO）对Qwen3-1.7B和Nemotron-3-Nano进行微调后，可推知寄存器的预测准确率分别从73.2%提升至85.3%，以及从63.3%提升至95.2%。前沿模型仍然得分最高（最高达99.2%），但我们微调后的Nemotron-3-Nano与其仅相差四个百分点。

    arXiv:2610.36288v1 Announce Type: cross  Abstract: Large Language Models (LLM) are increasingly applied to Reverse Engineering (RE) workflows. RE, however, differs from common LLM coding tasks as it requires models to reason about low-level program behavior. Furthermore, the deployment of tooling in resource-constrained sandboxed RE environments limits the ability to use frontier-scale solutions. We examine whether a targeted LLM training approach can transfer distinct RE skills to open models via inverse basic-block execution: given an x86-64 basic block and a desired output register state, the model predicts the input register values that produce it. Fine-tuning \texttt{Qwen3-1.7B} and \texttt{Nemotron-3-Nano} with Group Relative Policy Optimization (GRPO) raises knowable-register prediction accuracy from 73.2\% to 85.3\% and from 63.3\% to 95.2\%, respectively. Frontier models still score highest (up to 99.2\%), but our fine-tuned \texttt{Nemotron-3-Nano} comes within four points at
    
[^196]: 基于双目标优化的智能即时预订筛选模型

    A Smart Instant Book Filter Model based on Dual-Objective Optimization

    [https://arxiv.org/abs/2610.36287](https://arxiv.org/abs/2610.36287)

    本文提出一种基于双目标优化的智能即时预订筛选模型，通过在搜索结果中智能筛选和呈现即时预订房源，在减少房客预订延迟与被拒风险的同时兼顾平台预订率等目标。

    

    近年来，在线市场平台上的住宿接待服务日益流行，这不仅是因为它帮助人们轻松租住短期住所，还因为它让旅行者能够与当地人建立联系并获得归属感。这类行业的典型模式是由一个中央平台来中介房东与房客之间的经济互动，而成功的预订体验是用户满意度的关键。然而，最大的挑战之一在于：当房客尝试预订“需请求确认”类型的房源时，可能出现响应延迟以及被拒绝的风险。此前，我们已启用“即时预订”功能，允许房东立即接受符合条件的房客，房客也可立即确认预订。然而，由于搜索结果中混合呈现“需请求确认”房源与“即时预订”房源，这一问题并未得到缓解。一种可能的解决方案是在搜索结果中提升所有即时预订房源的排名；然而，预订率……

    arXiv:2610.36287v1 Announce Type: new  Abstract: Hospitality service in online marketplaces has become increasingly popular over the past few years, not only because it helps people easily rent short-term lodging but also because it empowers travelers to connect with locals and feel belonging. A typical pattern of these industries is a central platform mediating the economic interactions between guests and hosts, where a successful booking experience is the key to user satisfaction. However, one of the biggest challenges is the possible delay of response and chance of rejection when guests try to book a request-to-book listing. Previously, we enabled the instant book feature to allow hosts to instantly accept qualified guests and guests to confirm bookings immediately. Yet the symptom has not been alleviated given mixed request-to-book and instant book listing supply in search results. One possible solution is to promote all instant book listings in search results; however, booking rat
    
[^197]: cuArms：面向多机械臂操作的GPU加速基于冲突的规划方法

    cuArms: GPU-Accelerated Conflict-Based Planning for Multi-Arm Manipulation

    [https://arxiv.org/abs/2610.36250](https://arxiv.org/abs/2610.36250)

    cuArms是一个与GPU并行计算协同设计的基于冲突的多机械臂操作规划器，通过可复用的多模态路线图和惰性搜索高效解决冲突与死锁，并结合高层调度器联合优化任务分配、调度、运动和操作技能。

    

    多机械臂操作有潜力显著提升许多操作任务的吞吐量、效率和灵巧性。近期的工作通过分配和调度任务以获得更高的并行性、联合规划多机械臂运动、或直接学习多机械臂策略来协调共享工作单元中的多个机器人机械臂。然而到目前为止，这些方法要么速度太慢，要么依赖特定领域的假设，要么随着机器人数量和任务时域的增长而扩展性很差。因此，我们提出了cuArms，一个与GPU并行性协同设计的基于冲突的规划器，它利用可复用的多模态路线图高效分解单个机器人的搜索空间，并通过惰性搜索解决冲突和死锁问题。为了联合优化任务分配、调度、运动和操作技能，我们集成了一种高层调度器，为cuArms提出候选任务骨架，并将其具体化为协调一致的操作计划。实验结……

    arXiv:2610.36250v1 Announce Type: cross  Abstract: Multi-arm manipulation has the potential to dramatically improve the throughput, efficiency, and dexterity of many manipulation tasks. Recent work coordinates multiple robot arms in a shared workcell by allocating and scheduling tasks for greater parallelism, jointly planning multi-arm motions, or directly learning multi-arm policies. So far, these approaches are either too slow, rely on domain-specific assumptions, or scale poorly with the number of robots and task horizon. Thus, we introduce cuArms, a conflict-based planner co-designed with GPU parallelism that efficiently decomposes the search space using reusable multi-modal roadmaps for individual robots and resolves conflicts and deadlocks through a lazy search. To jointly optimize task assignment, schedules, motions, and manipulation skills, we integrate a high-level scheduler that proposes candidate skeletons for cuArms to ground into coordinated manipulation plans. Experimenta
    
[^198]: 冻结而非克隆：无需保留离线数据的表达性策略微调

    Freeze, Don't Clone: Fine-Tuning with Expressive Policies Without Retaining Offline Data

    [https://arxiv.org/abs/2610.36247](https://arxiv.org/abs/2610.36247)

    本文发现离线数据消失后持续进行行为克隆会通过行为反馈循环导致性能退化，提出冻结预训练表达性策略先验并停止在线行为拟合的方法，在七个OGBench域上显著提升了IFQL、FQL和QC等算法在无需保留离线数据场景下的微调效果。

    

    离线到在线强化学习利用离线预训练来加速后续的在线学习。虽然许多方法假设离线数据集在整个微调过程中始终可用，但在许多实际场景中并非如此。以往的工作将这种场景下的失败主要归因于分布偏移下评论家（critic）的重新校准。我们识别出一种互补的执行者（actor）侧的失败模式：一旦离线数据消失，持续的行为克隆（BC）不再保留离线行为分布，而是将行为模型重新拟合到智能体有限的、非平稳的在线经验上。我们证明，在保留预训练表达性先验的同时停止在线行为拟合，可以在七个 OGBench 域上显著改进隐式流 Q 学习（IFQL）、流 Q 学习（FQL）和动作分块强化学习（QC）的无保留微调性能。我们将这种失败追溯到行为反馈循环：（摘要在此处截断）

    arXiv:2610.36247v1 Announce Type: cross  Abstract: Offline-to-online reinforcement learning uses offline pretraining to accelerate subsequent online learning. While many methods assume the offline dataset remains available throughout fine-tuning, there are many settings where this is not the case. Prior work attributes failures in this setting primarily to critic recalibration under distribution shift. We identify a complementary actor-side failure: once the offline data disappears, continued behavioral cloning (BC) no longer preserves the offline behavior distribution and instead refits the behavioral model to the agent's limited, and nonstationary, online experience. We demonstrate that stopping online behavioral fitting while preserving the pretrained expressive prior substantially improves no-retention fine-tuning across implicit flow Q-learning (IFQL), flow Q-learning (FQL), and action-chunked RL (QC) on seven OGBench domains. We trace this failure to a behavioral feedback loop: c
    
[^199]: BehaveRec：长期行为历史何时有助于序列推荐？——基于会话、历史与社区信号的研究

    BehaveRec: When Does Long-Term Behavioral History Help Sequential Recommendation? A Study with Session, History, and Community Signals

    [https://arxiv.org/abs/2610.36239](https://arxiv.org/abs/2610.36239)

    提出三信号框架BehaveRec，通过可学习门控融合短期会话、90天长期用户历史和基于行为嵌入相似度的社区信号，在1.37亿条序列的工业数据集上超越了更大容量的SASRec基线，并系统研究了长期历史信号何时真正对推荐有效。

    

    序列推荐模型通常根据用户近期的会话交互来预测其下一个动作，这在短期信号较弱或模糊时会限制个性化效果。我们并不假设引入更长期的历史和社区信号总是有益，而是研究它们何时真正有效。我们提出了BehaveRec，这是一个三信号框架，结合了因果会话transformer、覆盖90天窗口的双向用户历史编码器，以及在用户-用户图上运行的GraphSAGE模块（该图基于行为嵌入相似度而非物品共现构建）；一个可学习的门控机制为每个用户自适应融合这三种信号。在一个包含1260万用户的1.37亿条序列的工业数据集上，BehaveRec将R@5相比SASRec提升了3.2%，而一个容量更大的SASRec基线（参数量多于BehaveRec）仍然落后2.5%，排除了模型容量作为解释。更重要的是，按历史长度进行的用户级分层分析揭示（摘要在此处截断）

    arXiv:2610.36239v1 Announce Type: new  Abstract: Sequential recommendation models typically predict a user's next action from recent session interactions, which limits personalization when short-term signals are weak or ambiguous. Rather than assuming that incorporating longer-term history and community signals is universally beneficial, we study when it actually helps. We propose BehaveRec, a three-signal framework combining a causal session transformer, a bidirectional User History Encoder over a 90-day window, and a GraphSAGE module over a user-user graph built from behavioral embedding similarity rather than item co-occurrence; a learned gate blends the three signals per user. On an industrial dataset of 137M sequences from 12.6M users, BehaveRec improves R@5 by +3.2% over SASRec, with a larger-capacity SASRec baseline (more parameters than BehaveRec) still trailing by +2.5%, ruling out capacity as the explanation. More importantly, a user-level stratification by history length rev
    
[^200]: MARQ：一个注入系统工程知识的模块化、大模型无关的需求质量评估器

    MARQ: A Modular, LLM-Agnostic Requirement Quality Assessor Infused with Systems Engineering Knowledge

    [https://arxiv.org/abs/2610.36234](https://arxiv.org/abs/2610.36234)

    MARQ是一个模块化、与具体大模型无关的需求质量评估原型，通过注入基于INCOSE准则可操作化的系统工程专家知识，显著提升了大语言模型在不同质量准则下识别需求缺陷的能力，超越了现成模型的表现。

    

    需求工程（RE）是系统工程（SE）的基础，因为需求缺陷会向下游传播并对开发结果产生不利影响。尽管大语言模型（LLM）在需求质量评估方面前景可观，但先前研究表明，现成模型会遗漏大量专家识别出的缺陷，在不同质量准则上的表现参差不齐，且多次执行之间结果存在差异。本研究探究注入专家提炼的系统工程知识能否使基于大语言模型的需求质量评估超越现成模型的能力。我们提出了MARQ，一个模块化、大模型无关的原型系统，它通过一个针对具体质量准则的知识栈来增强大语言模型，该知识栈对INCOSE需求质量准则进行了可操作化的实现。我们在一个由专家标注的NASA需求数据集上，使用Claude Opus 5、GPT-5.6 Sol以及开放权重的Gemma 3 12B进行评估（每个模型执行100次），并与现成基线进行对比。MARQ取得了中位数F1分……（原文摘要在此处截断）

    arXiv:2610.36234v1 Announce Type: new  Abstract: Requirements engineering (RE) is foundational to systems engineering (SE), as requirement defects can propagate downstream and adversely affect development outcomes. Although large language models (LLMs) are promising for requirement quality assessment, prior work shows that off-the-shelf models miss a large share of expert-identified defects, perform unevenly across quality criteria, and vary across executions. This study investigates whether infusing expert-derived SE knowledge can improve LLM-based requirement quality evaluation beyond off-the-shelf capability. We present MARQ, a modular, LLM-agnostic prototype that supplements LLMs with a criterion-specific knowledge stack operationalizing INCOSE requirement quality criteria and evaluated against off-the-shelf baselines on an expert-labeled NASA requirement set using Claude Opus 5, GPT-5.6 Sol, and the open-weight Gemma 3 12B, with 100 executions per model. MARQ achieves median F1 sc
    
[^201]: 智能体还是人类？在缺失作者身份签名的情况下检测AI编程智能体的提交

    Agent or Human? Detecting Commits by AI Coding Agents in the Absence of Authorship Signatures

    [https://arxiv.org/abs/2610.36229](https://arxiv.org/abs/2610.36229)

    该研究基于近800万个提交发现，即使去除作者身份签名，仍可凭借提交信息冗长度和变更规模等特征以最高0.859的ROC AUC区分AI编程智能体与人类的代码提交，且检测信号主要来自元数据而非代码本身。

    

    诸如 Claude Code、GitHub Copilot 编程智能体和 Cursor 等 AI 编程智能体现在可以直接向开源仓库提交代码。针对这些智能体的实证研究通常通过作者身份签名（如机器人账号和 Co-authored-by 尾注）来识别它们的提交。然而，这些签名是可选的、可以被移除的，并且通常附加在拉取请求上而非单个提交上。因此，一旦签名消失，能否将智能体提交与人类提交区分开来尚不清楚。我们使用来自 1,226 个仓库的 7,837,946 个提交（覆盖五个编程智能体及人类提交），研究了无签名检测的信号与局限。在签名被掩盖后，检测器对智能体提交的评分排名高于人类提交，ROC AUC 最高可达 0.859，而且大部分信号来自提交信息的冗长度和变更规模，而非代码本身。在共享样本上，gpt-6-sol 在不进行微调的情况下达到了与 XG（摘要在此处被截断，后续内容缺失）……

    arXiv:2610.36229v1 Announce Type: cross  Abstract: AI coding agents such as Claude Code, GitHub Copilot coding agent, and Cursor now commit directly to open-source repositories. Empirical studies of these agents identify their commits through authorship signatures, such as bot accounts and Co-authored-by trailers. These signatures are optional, can be removed, and are often attached to pull requests rather than to individual commits. Hence, it is unclear whether Agent commits can be told apart from Human commits once the signatures are gone. We investigate the signals and limits of signature-free detection using 7,837,946 commits from 1,226 repositories, covering five coding agents and Human commits. After signatures are masked, detectors rank Agent commits above Human commits with a ROC AUC of up to 0.859, and most of this signal comes from message verbosity and change size rather than from the code itself. On the shared sample, gpt-6-sol without fine-tuning reaches AUCs similar to XG
    
[^202]: 圆与球上的方向统计：看待数据的新“角度”

    Directional Statistics on Circles and Spheres: A New Angle on Data

    [https://arxiv.org/abs/2610.36227](https://arxiv.org/abs/2610.36227)

    本文指出角度数据天然位于圆和球而非直线上，直接算术平均可能导致错误结论（如朝北方位平均后指向南方），并通过风向与扩散MRI等实例介绍圆形/轴向均值、方向变异性和概率模型等方向统计工具，主张在选择统计方法前应先识别数据的几何结构与对称性。

    

    两个靠近北方的罗盘方位，其普通算术平均值却可能指向南方：问题在于角度存在于圆周上，而非直线上。本文利用真实的风向观测数据、扩散磁共振成像（diffusion-MRI）方向以及来自机器学习的几何实例，介绍了圆和球上的圆形与轴向平均、方向变异性以及概率模型。其核心指导原则是：在选择统计方法之前，应先识别观测数据所具有的几何结构与对称性。

    arXiv:2610.36227v1 Announce Type: cross  Abstract: Two compass bearings near north can have an ordinary arithmetic mean pointing south: the difficulty is that angles live on a circle, not a straight line. Using real wind observations, diffusion-MRI orientations, and geometric examples from machine learning, this article introduces circular and axial averages, directional variability, and probability models on circles and spheres. The guiding principle is to identify the geometry and symmetries of the observations before choosing a statistical method.
    
[^203]: 结构而非序列：利用稀疏自编码器追踪并验证 ESMFold 主干中的折叠机制

    Structure, Not Sequence: Tracking and Validating Folding Mechanisms in ESMFold's Trunk with Sparse Autoencoders

    [https://arxiv.org/abs/2610.36226](https://arxiv.org/abs/2610.36226)

    该论文首次用稀疏自编码器解析 ESMFold 的折叠主干，发现承载折叠计算的特征按“结构而非序列”组织，并在三维结构回归过程中呈现出模块化的功能分工。

    

    arXiv:2610.36226v1 公告类型：交叉 摘要：诸如 ESMFold 之类的结构预测器通过神经网络计算来折叠蛋白质，已经彻底改变了结构生物学。然而，从氨基酸（AA）序列到三维结构的驱动机制仍然缺乏充分的理解，其生物学意义依旧被“锁”在稠密网络之中。稀疏自编码器（SAEs）已被证明能够在中间蛋白质序列模型表示的超完备基上获得可解释的概念特征，并展现出可观的前景。然而，对于从氨基酸序列语义预测物理形状的折叠模型，目前尚无这样的可解释性研究。我们假设承载折叠计算的特征会在三维结构回归过程中发生演化。在本文中，我们分三步对折叠主干进行解释。首先，通过对各模块与循环（recycles）间表示分布的初步分析，定位了主干中独特的空间计算方式。其次，学到的字典揭示了一种分工现象：单个

    arXiv:2610.36226v1 Announce Type: cross  Abstract: Structure predictors like ESMFold fold proteins using neural network computation and have transformed structural biology. However, the driving mechanism from amino-acid (AA) sequence to 3D structure is insufficiently understood, and biological meaning remains locked in the dense network. Sparse autoencoders (SAEs) have shown promising results in obtaining interpretable concept features over an overcomplete basis of intermediate protein sequence-model representations. Yet folding models, which predict physical shape from AA sequence semantics, have no such account. We hypothesize that the features carrying the folding computation evolve in the 3D structural regression process. In this paper, we interpret the folding trunk in three steps. First, a pilot analysis on the representation distribution across modules and recycles locates distinct spatial computation in the trunk. Second, the dictionaries expose a division of labor: the single 
    
[^204]: 反事实用户画像扰动用于推荐解释：一项可复现性研究

    Counterfactual Profile Perturbations for Recommendation Explanations: A Reproducibility Study

    [https://arxiv.org/abs/2610.36208](https://arxiv.org/abs/2610.36208)

    本研究复现并扩展了LXR反事实推荐解释方法，证明其不仅对最高推荐项目有效，对其他项目同样有效，同时发现项目流行度等简单基线在某些任务中表现得出人意料地好。

    

    这篇可复现性论文探索了一种被称为“学习解释推荐”（LXR）的反事实解释技术。先前的研究表明，LXR能够可靠地识别出一些扰动，如果将这些扰动应用于用户的画像，会导致用户排名最高的项目不再被推荐，从而将项目的推荐解释为用户评分选择的函数。在本文中，我们以多种方式扩展了这项工作。我们将原始LXR研究的范围扩展到查看除最高推荐项目之外的其他项目的解释，发现LXR在这一版本的任务中同样有效。我们考虑了LXR的替代基线，包括项目流行度和潜在因子相似度，发现对于某些任务，这些替代方案的表现出人意料地好。我们还研究了项目流行度的影响，发现当推荐的项目不那么流行时，LXR能提供更好的解释质量（即更短的扰动）……

    arXiv:2610.36208v1 Announce Type: cross  Abstract: This reproducibility paper explores the counterfactual explanation technique known as Learning to eXplain Recommendations (LXR). Prior work has shown that LXR can reliably identify perturbations that, if applied to a user's profile, cause the user's top ranked item to no longer be recommended, thereby explaining the item's recommendation as a function of the user's rating choices. In this paper, we extend this work in multiple ways. We expand the scope of the original LXR study to look at explanations for items other than the most highly recommended item, finding that LXR is also effective at this version of the task. We consider alternate baselines for LXR, including item popularity and latent factor similarity, and we see that for some tasks these alternatives work surprisingly well. We also examine the impact of item popularity, finding that LXR offers better explanation quality (shorter perturbations) when less popular items are be
    
[^205]: Concorde：基于解耦能量最小化的几何感知链路预测

    Concorde: Geometry-Aware Link Prediction via Decoupled Energy Minimization

    [https://arxiv.org/abs/2610.36204](https://arxiv.org/abs/2610.36204)

    提出 Concorde 能量框架，通过无参数谱锚点、乘积流形混合曲率残差与双曲切丛连续向量场三个解耦分支的门控融合，实现稀疏图上几何感知且谱稳定的链路预测。

    

    链路预测在稀疏图中仍然困难重重：扩散启发式方法缺乏连通性，而几何方法在将学习到的表示与扩散算子耦合时会遭遇谱不稳定性问题。我们提出 Concorde，一个基于能量的框架，通过将链路似然分解为三个结构上解耦的分支来解决这一问题：(i) 无参数的谱锚点，建立置换不变的拓扑稳定性；(ii) 可学习的双曲 × 球面乘积流形上的混合曲率残差，由特征感知的 Ollivier-Ricci 曲率代理进行正则化；(iii) 双曲空间切丛上的连续向量场，用于跨拓扑间隙的非局部结构对齐。这些分支通过学习到的门控机制进行融合，并采用与检索式评估相匹配的 N-pair 排序损失进行训练。在涵盖稀疏、稠密、特征等场景的五个基准数据集上（原文摘要在此处截断）……

    arXiv:2610.36204v1 Announce Type: cross  Abstract: Link prediction remains difficult in sparse graphs, where diffusion heuristics lack connectivity and geometric methods suffer spectral instability when coupling learned representations with diffusion operators. We introduce Concorde, an energy-based framework that resolves this by decomposing link likelihood into three structurally decoupled branches: (i) a parameter-free spectral anchor that establishes permutation-invariant topological stability; (ii) a mixed-curvature residual on a learnable $\mathbb{H}^{d_h}_{c_h} \times \mathbb{S}^{d_s}_{c_s}$ product manifold, regularized by a feature-aware Ollivier-Ricci curvature proxy; and (iii) a continuous vector field on the tangent bundle of hyperbolic space for non-local structural alignment across topological gaps. The branches are fused via learned gating and trained with an N-pair ranking loss aligned with retrieval-style evaluation. Across five benchmarks spanning sparse, dense, featu
    
[^206]: 基础模型能够检测到提示分类器所遗漏的隐藏意图

    Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

    [https://arxiv.org/abs/2610.36202](https://arxiv.org/abs/2610.36202)

    本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。

    

    人格选择模型（PSM）假设后训练过程是从模型在预训练期间学到的诸多人格中选择其一。关于潜意识学习（subliminal learning）的研究表明，模型可以将隐蔽信号（如“对猫头鹰的喜爱”）传递到语义上无关的输出（如数字串）中，并且在这些输出上训练的学生模型会继承这些特质。在本工作中，我展示了通过在基础预训练模型上配合适当的头部提示（header）来评估此类输出的似然——这种方法我称之为人格-头部似然评估（PHLE），可以解码这些潜意识信号并对它们进行分类。例如，PHLE 能够区分由被提示“你喜爱猫头鹰”的后训练模型生成的数字序列与被提示“你喜爱鹰”的模型生成的数字序列，并且能够区分一个秘密怀有伤害用户意图的模型（已经过滤以去除任何明显的失调迹象）的回答与没有系统提示的模型的回答。在这两种情况下，提示

    arXiv:2610.36202v1 Announce Type: cross  Abstract: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompte
    
[^207]: count-catcher：从模型计数证书到 Lean 定理

    count-catcher: Lean Theorems from Model Counting Certificates

    [https://arxiv.org/abs/2610.36196](https://arxiv.org/abs/2610.36196)

    count-catcher 工具将精确模型计数器 d4 的 CPOG 计数证书转化为 Lean 中经过验证的定理，证明 CNF 公式（进而对应的组合对象）恰好有 N 个模型。

    

    我们提出 count-catcher，这是一个工具，它将由精确模型计数器 d4 产生的计数的 CPOG 证书转化为一个 Lean 定理，该定理断言：在 Lean 中定义的 CNF 公式在 V 个变量上恰好有 N 个模型。通过一个编码引理，该定理可以进一步转化为关于组合对象数量的定理。CPOG 证书由一个与公式等价的布尔电路以及一个等价性证明组成。count-catcher 在 Lean 中检查公式蕴含电路（正向检查），以及电路蕴含公式（反向检查）。对于正向检查，lrat-catcher 通过 Lean 标准库的 LRAT 检查器流式传输一个反驳。反向检查和电路的有效性检查都是在 Lean 中被证明正确的计算。Lean 以编译代码的形式运行所有三项检查，因此该定理依赖于 Lean 的内核、编译器和运行时，CPOG 的经过验证的检查器也是如此。我们将该定理与 Mathlib 中的结果相结合，例如与

    arXiv:2610.36196v1 Announce Type: cross  Abstract: We present count-catcher, a tool that turns a CPOG certificate of a count by the exact model counter d4 into a Lean theorem that a CNF formula defined in Lean has exactly N models over V variables. With an encoding lemma, this theorem becomes a theorem about the number of combinatorial objects. A CPOG certificate consists of a Boolean circuit equivalent to the formula and a proof of equivalence. count-catcher checks in Lean that the formula implies the circuit (forward check) and the circuit implies the formula (reverse check). For the forward check, lrat-catcher streams a refutation through the LRAT checker of Lean's standard library. The reverse check and the validity check of the circuit are computations proved correct in Lean. Lean runs all three checks as compiled code, so the theorem depends on Lean's kernel, compiler and runtime, as does CPOG's verified checker. We combine the theorem with results from Mathlib, for instance with
    
[^208]: 超越估计：维纳信息引导解码以简化像素空间流模型的训练

    Beyond the Estimate: Wiener-Informed Decoding for Easier Training of Pixel-Space Flow Models

    [https://arxiv.org/abs/2610.36192](https://arxiv.org/abs/2610.36192)

    提出维纳信息引导解码（WiD），利用二阶图像统计（维纳估计）来辅助速度场预测并组织解码器计算，从而显著简化像素空间流模型的训练。

    

    直接在像素空间学习速度场既困难又缓慢。我们证明，粗粒度的二阶图像统计信息可以简化这一任务：冻结的维纳估计能够改进速度场的学习，且其收益超出了它们显式提供的线性预测。此外，我们观察到冻结基和像素解码也能带来类似但互补的收益。这些观察催生了维纳信息引导解码，它利用统计知识进行直接预测，并组织解码器的计算，而无需修改原生的模型模块。WiD 将一个可训练的维纳预测路径与系数接口相结合，后者将特征注入原生解码器并返回其预测的系数修正，这些组件均以数据统计量进行初始化。所有引入的参数，包括频谱和坐标变换，都在原始的像素空间速度目标下联合训练。在跨模型规模的多种设置中……

    arXiv:2610.36192v1 Announce Type: cross  Abstract: Learning velocity fields directly in pixel space is difficult and slow. We show that coarse second-order image statistics can ease this task: frozen Wiener estimates improve velocity learning, with gains extending beyond the linear predictions they explicitly provide. Moreover, we observe similar but complementary gains from frozen bases and pixel decoding. These observations motivate Wiener-informed Decoding (WiD), which uses statistical knowledge both for direct prediction and to organize decoder computation without modifying native blocks. WiD combines a trainable Wiener prediction path with coefficient interfaces that inject features into the native decoder and return its predicted coefficient corrections, with these components initialized with data statistics. All introduced parameters, including the spectrum and coordinate transforms, are trained jointly under the original pixel-space velocity objective. Across model scales and d
    
[^209]: 技能梦境：通过拉取请求流演化仓库技能

    Skill Dreaming: Evolving Repository Skills through Pull Request Streams

    [https://arxiv.org/abs/2610.36188](https://arxiv.org/abs/2610.36188)

    提出了 Skill Dreaming 框架，以拉取请求流为知识来源，实现拉取请求、仓库与技能库的共同演化，使仓库专属技能随代码库的变化持续更新并反哺未来的 PR 实践。

    

    大语言模型（LLM）智能体在复杂的软件工程任务中越来越依赖可复用的技能，这使得技能库对于仓库的开发与维护至关重要。然而，针对特定仓库的技能必须随着代码结构、实现规范和维护需求的变化而不断演化。拉取请求记录了仓库在实现与审查过程中的演变，为此类知识提供了天然的来源。同时，新的拉取请求必须遵循仓库特定的实现模式，一致地修改相关组件，并确保在为新功能添加测试覆盖的同时，现有测试仍能继续通过。这就形成了一个相互增强的循环：拉取请求为技能更新提供知识，而这些技能又能改进未来的拉取请求、维护仓库，并支持技能的进一步演化。为了实现拉取请求、仓库和技能的这种共同演化，我们提出了 Skill Dreaming，一个框架……

    arXiv:2610.36188v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly rely on reusable skills for complex software engineering tasks, making skill libraries essential for repository development and maintenance. However, repository-specific skills must evolve with changing code structures, implementation conventions, and maintenance requirements. Pull requests (PRs) provide a natural source of such knowledge by recording how repositories evolve through implementation and review. Meanwhile, new PRs must follow repository-specific implementation patterns, modify related components consistently, and ensure that existing tests continue to pass while adding coverage for new functionality. This creates a mutually reinforcing cycle: PRs provide knowledge for updating skills, while these skills improve future PRs, maintain repositories, and support further skill evolution. To enable this co-evolution of PRs, repositories, and skills, we propose Skill Dreaming, a framew
    
[^210]: AI能否在不生成测试的情况下指导测试？一项关于AI辅助单元测试的对比研究

    Can AI Guide Testing Without Generating Tests? A Comparative Study of AI-Assisted Unit Testing

    [https://arxiv.org/abs/2610.36185](https://arxiv.org/abs/2610.36185)

    该研究提出TestCompass——一个不代学生生成可执行测试、而是针对其测试套件薄弱点提供定向反馈的生成式AI辅导工具，并通过18名学生的课堂对比实验证明AI可以在不直接生成测试的前提下有效支持初学者的单元测试学习。

    

    生成式人工智能的最新进展使AI助手能够生成日益复杂的可执行单元测试。然而，初学者可能会直接接受这些输出，而不去独立思考应该测试什么、评估测试套件的完整性或验证其正确性，这可能限制他们发展关键测试技能的机会。尽管先前的研究已经探索了AI辅助测试生成以及面向反馈的编程辅导工具，但对于AI能否在不为初学者生成测试的情况下支持他们的测试实践，人们仍然知之甚少。为了探究这一问题，我们提出了TestCompass——一个生成式AI辅导工具，它针对学生测试套件中的薄弱环节提供有针对性的反馈，但并不生成可执行的测试。在一项包含18名学生的课堂研究中，我们在两个Java单元测试任务上将TestCompass与通用AI助手进行了对比，考察了学生的测试行为、测试套件质量以及主观体验。（摘要原文在此处中断）

    arXiv:2610.36185v1 Announce Type: cross  Abstract: Recent advances in generative AI have enabled assistants to generate increasingly sophisticated executable unit tests. However, novices may accept these outputs without independently identifying what to test, evaluating test-suite completeness, or verifying correctness, potentially limiting opportunities to develop essential testing skills. Although prior work has explored AI-assisted test generation and feedback-oriented programming tutors, little is known about whether AI can support novices' testing practices without generating tests for them. To investigate this, we introduce TestCompass, a generative AI tutor that provides targeted feedback on weaknesses in students' test suites without generating executable tests. In a classroom study with 18 students, we compared TestCompass with a general AI assistant across two Java unit-testing tasks, examining students' testing behaviors, test-suite quality, and subjective experiences. We fo
    
[^211]: 基于追踪数据分位数回归学习人类体育运动动作可行性

    Learning Human Sports Motion Feasibility via Quantile Regression on Tracking Data

    [https://arxiv.org/abs/2610.36183](https://arxiv.org/abs/2610.36183)

    该论文将体育中的球员到达时间（TTA）建模为条件分位数回归问题，利用2022年世界杯全部64场比赛的光学追踪数据训练的轻量级多层感知机，在预测球员到达球场任意位置的最短合理时间上一致优于文献中所有基于物理的运动模型基线。

    

    我们研究体育运动中的到达时间（TTA）：即给定球员当前的运动学状态，其到达球场上任意位置所需的最短合理时间，该时间被估计为观测到的移动时间的下包络。我们没有通过位置密度估计或闭式物理模型来建模空间占有，而是将TTA构建为一个条件分位数回归问题，估计移动时间的下分位数，从而使模型学习球员的下包络到达时间。我们在PFF FC 2022年FIFA世界杯数据集全部64场比赛的光学追踪数据上训练模型，并将一个轻量级多层感知机（MLP）与文献中的六种球员运动模型（重新表述为TTA预测器）以及一个学习得到的可达包络基线进行对比评估。在我们的各项评估指标上，MLP一致地优于所有基于物理的基线模型。为了探究所学移动模型的通用性，我们进一步……

    arXiv:2610.36183v1 Announce Type: new  Abstract: We study time-to-arrival (TTA) in sports: the shortest plausible time a player needs to reach an arbitrary location on the pitch given their current kinematic state, estimated as the lower envelope of observed locomotion times. Rather than modeling space ownership through position density estimation or closed-form physics-based models, we cast TTA as a conditional quantile regression problem, estimating the lower quantile of locomotion time so the model learns a player's lower-envelope arrival time. We train the model on optical tracking data from all 64 matches of the PFF FC FIFA World Cup 2022 dataset, and evaluate a lightweight multilayer perceptron against six player-motion models from the literature, re-expressed as TTA predictors, and a learned reachable- hull baseline. Across our evaluation metrics, the MLP consistently outperforms every physics-based baseline. To probe the generality of the learned locomotion model, we further ev
    
[^212]: 不要选定赢家：知识冲突需要先验，而非规则

    Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule

    [https://arxiv.org/abs/2610.36169](https://arxiv.org/abs/2610.36169)

    论文证明，检索增强语言模型中检索上下文与内部知识的冲突无法靠设计时固定的规则（如一律信上下文或信模型）最优解决，且自适应解码依赖的对称散度不包含判断依据，正确做法是结合分别评估各来源的证据与上下文可靠性的先验。

    

    检索增强语言模型从两个来源获取答案：检索到的上下文和其自身的知识，当上下文不完整或错误时，这两者就会产生分歧。现有方法通过在设计阶段就固定的规则来解决这种分歧：遵循上下文、按上下文对模型的影响程度按比例放大它，或者信任模型的记忆。每种规则在某些部署场景中都必然出错。在双来源的决策理论模型中，任何固定规则的得分与可靠上下文所占比例呈线性关系，而可获得的最佳得分是凸函数，因此每个固定规则在除恰好最优情形之外的任何情况下都会被超越；而且两个来源之间的对称散度——即自适应解码所依赖的信号——不携带任何关于哪个来源正确的信息。因此，解决冲突需要两个现有方法都未保留的输入：分别关于每个来源的证据，以及关于上下文正确频率的先验。我们通过实验验证了这些主张……

    arXiv:2610.36169v1 Announce Type: new  Abstract: A retrieval-augmented language model answers from two sources, the retrieved context and its own knowledge, and the two disagree whenever the context is incomplete or wrong. Existing methods settle the disagreement with a rule fixed at design time: follow the context, amplify it in proportion to how far it moves the model, or trust the model's memory. Each rule is confidently wrong in some deployment. In a decision-theoretic model of two sources, the score of any fixed rule is linear in the share of reliable contexts while the best attainable score is convex, so every fixed rule is beaten wherever it is not exactly optimal; and a symmetric divergence between the two sources, the signal adaptive decoding uses, carries no information about which one is right. Resolving a conflict therefore needs two inputs that no existing method keeps: evidence about each source separately, and a prior on how often the context is right. We test these clai
    
[^213]: Stratum：超越上下文窗口的编码智能体仓库记忆

    Stratum: Repository Memory for Coding Agents Beyond the Context Window

    [https://arxiv.org/abs/2610.36153](https://arxiv.org/abs/2610.36153)

    该论文提出Stratum，为编码智能体提供跨会话的持久化、经验感知的仓库记忆系统，并通过实证测量证明真正的瓶颈在于相关知识的选择而非上下文窗口容量。

    

    上下文窗口已经增长了一千倍，然而编码智能体在每次会话开始时仍然对过去的工作毫无记忆。它们会重新读取文件、重试失败的方法，并忘记几小时前已经确定的根本原因。这不是上下文长度问题，而是记忆架构问题。我们的假设是：编码智能体需要持久的、具有经验感知能力的仓库记忆，使知识能够从一个会话延续到下一个会话。在真实仓库上的三项测量结果支撑了这一设计。首先，中等规模的项目已经超过最大的生产级上下文窗口，并且将知识存储在模型的键值（KV）缓存中的字节开销，比将其存储为索引文本高出 $10^4$ 到 $10^5$ 倍。其次，在752个真实的缺陷修复提交中，修复操作的中位数只编辑两个文件，这些文件及其直接导入的内容合计不到仓库的10%；瓶颈在于选择，而非容量。第三，即使考虑传递性导入，……

    arXiv:2610.36153v1 Announce Type: cross  Abstract: Context windows have grown a thousand-fold, yet coding agents still start every session with no memory of their past work. They re-read files, retry failed approaches, and forget root causes they established hours earlier. This is not a context-length problem but a memory-architecture problem. Our hypothesis: coding agents need persistent, experience-aware repository memory, so that knowledge survives from one session to the next. Three measurements on real repositories motivate the design. First, mid-size projects already exceed the largest production context windows, and storing knowledge inside the model's key-value (KV) cache costs $10^4$ to $10^5$ times more bytes than storing it as indexed text. Second, across 752 real bug-fix commits, the median fix edits two files, and these files plus their direct imports amount to less than 10% of the repository; the bottleneck is selection, not capacity. Third, even with transitive import re
    
[^214]: 复制还是印证？多智能体辩论中证据依赖性的公理化框架

    Replication or Corroboration? An Axiomatic Framework for Evidential Dependence in Multi-Agent Debate

    [https://arxiv.org/abs/2610.36151](https://arxiv.org/abs/2610.36151)

    该论文提出了一个公理化框架来区分多智能体辩论中的“复制”（重复相同证据）与“印证”（独立证据支持同一主张），并证明“根源复制不变性”性质能够确保仅靠复制行为无法改变评判者的裁决结果。

    

    在多智能体辩论中，匹配的报告可能出于两种不同的原因出现：智能体可能重复相同的证据（复制），或者他们可能提供各自独立的证据根源来支持同一主张（印证）。评判者必须区分这两种情况。按报告出现次数计数会给转述、改写、别名重复提交和片段拆分赋予额外权重，而将所有匹配内容进行合并虽然可以避免这一问题，却同时也消除了真正的印证。我们将这一矛盾形式化为一个最小判断博弈，其中获取新证据代价高昂，而复制却很廉价。我们证明，根源复制不变性恰好等价于阈值均匀复制免疫。在该性质下，仅由复制产生的偏差无法改变裁决结果，且具有正成本的复制策略被严格占优；而对出现次数敏感和基于内容键值的评判者则存在有利可图的偏差。我们还建立了一个信息边界：如果评判者可观测的……（摘要在此处被截断）

    arXiv:2610.36151v1 Announce Type: cross  Abstract: In multi-agent debates, matching reports can arise for two different reasons: agents may repeat the same evidence (replication), or they may provide separate evidential roots that support the same claim (corroboration). A judge must distinguish these two cases. Counting report occurrences gives extra weight to relays, paraphrases, alias resubmissions, and excerpt splits, while merging all matching content avoids this only by also removing genuine corroboration. We formalize this tension as a minimal judgment game in which acquiring new evidence is costly, but replication is cheap. We show that root replication invariance (RRI) is exactly threshold-uniform replication immunity. Under RRI, replication-only deviations cannot change the verdict, and positive-cost replication is strictly dominated, whereas occurrence-sensitive and content-keyed judges admit profitable deviations. We also establish an information boundary: if a judge's obser
    
[^215]: 从结构反馈到提示策略：学习忠实的文本到图像提示编辑器

    From Structural Feedback to Prompt Policies: Learning Faithful Text-to-Image Prompt Editors

    [https://arxiv.org/abs/2610.36146](https://arxiv.org/abs/2610.36146)

    该论文提出StruPO，一种结构感知的提示编辑器，利用校准场景图作为共享表示，并通过平滑奖励塑形和多图像平均处理定位不确定性与图像生成的随机性，从而学习忠实的文本到图像提示编辑策略，在不改变用户请求场景的前提下改善生成图像质量。

    

    提示优化应当在不改变用户所请求场景的前提下改善用户获得的图像。然而，即使全局文本-图像相似度保持较高水平，风格化的改写仍可能削弱物体属性及其相互关系。我们提出了StruPO，一种结构感知的提示编辑器，它使用校准后的场景图作为策略条件化和基于图像监督的共享表示。场景图揭示了应当保留的内容，而删除、重排和插入操作提供了一种与词元对齐的方式来修改提示。这将结构反馈从一种外部批评转变为可复用编辑策略的训练信号。从该信号中学习需要同时处理不确定的定位和随机的图像生成。StruPO通过平滑奖励塑形保留定位置信度，并在计算组相对优势之前对每次编辑生成的多个图像取平均。实验涵盖了三个互不重叠的2K提示测试集。

    arXiv:2610.36146v1 Announce Type: new  Abstract: Prompt optimization should improve the image a user receives without changing the scene the user requested. Yet stylistic rewriting can weaken object attributes and relations even when global text--image similarity remains high. We propose StruPO, a structure-aware prompt editor that uses a calibrated scene graph as a shared representation for policy conditioning and image-grounded supervision. The graph exposes what should be preserved, while deletion, reordering, and insertion provide a token-aligned way to modify the prompt. This turns structural feedback from an external critique into a training signal for a reusable editing policy. Learning from that signal requires handling both uncertain grounding and stochastic image generation. StruPO retains grounding confidence through smooth reward shaping and averages multiple images for each edit before computing group-relative advantages. Experiments span three disjoint 2K-prompt test sets
    
[^216]: 物理结构作为通用哈密顿量学习的组织原则

    Physical structure as an organizing principle for universal Hamiltonian learning

    [https://arxiv.org/abs/2610.36135](https://arxiv.org/abs/2610.36135)

    UniTB是一个将密度泛函紧束缚物理先验与深度学习相结合的通用E(3)等变哈密顿量框架，通过专家混合路由、电荷平衡头和投影约束，实现跨化学体系的泛化并保证下游性质的物理合理性。

    

    密度泛函理论（DFT）可以给出材料的电子结构，但其高昂的计算成本限制了大型超胞计算和材料筛选。机器学习哈密顿量可以绕过自洽循环，但通常只针对特定体系；而通用模型则有望覆盖新的化学成分和无序固体。这类模型必须能在化学空间中泛化，同时保持下游性质的物理合理性，并在大型晶胞下维持较低的计算成本。本文提出了UniTB，一个通用的E(3)等变哈密顿量表示框架，它将植根于密度泛函紧束缚理论的物理先验与深度学习相结合。UniTB的网络通过修正一个先验哈密顿量来工作，借助由该先验引导路由的专家混合机制，以及类似于自洽电荷紧束缚方法的电荷平衡头，实现跨化学体系的泛化。一个投影操作限制了相对先验的能带偏移，从而抑制虚假态，并使下游性质……（原文摘要在此处截断）

    arXiv:2610.36135v1 Announce Type: cross  Abstract: Density functional theory (DFT) gives the electronic structure of materials, but its cost limits large-supercell calculations and materials screening. Machine-learned Hamiltonians bypass the self-consistent cycle but are usually system-specific; universal ones could cover new compositions and disordered solids. Such models must generalize across chemistry and keep downstream properties physical, while staying inexpensive for large cells. Here we introduce UniTB, a universal E(3) equivariant Hamiltonian representation framework that integrates physical priors rooted in density-functional tight-binding theory with deep learning. UniTB's network, which corrects a prior Hamiltonian, generalizes across chemistry through a mixture of experts routed by that prior and a charge-equilibration head as in self-consistent-charge tight binding. A projection bounds band shifts from the prior, suppressing spurious states and keeping downstream propert
    
[^217]: 当连通性产生误导：利用对比证据路径引导智能体搜索

    When Connectivity Misleads: Guiding Agentic Search with Contrastive Evidence Paths

    [https://arxiv.org/abs/2610.36134](https://arxiv.org/abs/2610.36134)

    提出对比证据路径（CEP）方法，通过在查询条件化的文档图上依据最强单一路径评分，并利用背景访问分布进行校准来消除高连通性文档带来的结构偏差，从而引导搜索智能体发现真正有价值的证据而非高度连接的干扰节点。

    

    搜索智能体可以利用跨文档图通过多跳连接发现信息来源，并在有限的观察窗口内根据候选文档的排名和关系来选择后续行动。然而，某些文档之所以排名靠前，可能仅仅是因为其大量的连接使其更容易被访问到，而不一定是因为它们提供了更有价值的证据。诸如个性化 PageRank（PPR）等图扩散方法会聚合来自多个来源的传播贡献，导致高度连接的干扰节点挤占观察窗口。此外，扁平化的节点评分还会掩盖跨文档的证据链路，限制了持续探索。我们提出对比证据路径（CEP），该方法在查询条件化的文档图上根据候选文档的最强单一路径进行评分，并将这些评分与同一图的背景访问分布进行校准，以缓解结构偏差。CEP 根据校准后的分数对候选文档进行排序。

    arXiv:2610.36134v1 Announce Type: cross  Abstract: Search agents can use cross-document graphs to discover sources through multi-hop connections and choose subsequent actions based on candidate rankings and relationships within a limited observation window. However, documents may rank highly because their numerous connections make them easier to reach, without necessarily providing more valuable evidence. Graph diffusion methods such as Personalized PageRank (PPR) aggregate propagation contributions from multiple sources, allowing highly connected distractor nodes to crowd the observation window. Flat node scores also obscure cross-document evidence trails, limiting continued exploration. We propose Contrastive Evidence Paths (CEP), which scores candidates by their strongest individual paths on a query-conditioned document graph and calibrates these scores against the same graph's background visitation distribution to mitigate structural bias. CEP ranks candidates by the calibrated sco
    
[^218]: 当AI更擅长科学时，为何以及如何做科学：一个基于人类理解与人类主导的框架

    Why and How to Do Science When AI Is Better at It: A Framework Grounded in Human Understanding and Direction

    [https://arxiv.org/abs/2610.36116](https://arxiv.org/abs/2610.36116)

    本文提出“人类理解”与“人类主导”两部分检验标准，论证科学的核心目标是服务于人类目的与价值观的持久集体理解，并据此判断哪些研究任务可以委托给AI、哪些必须保留在负责任的人类手中。

    

    随着AI开始在核心科学任务上达到甚至超越人类研究者的水平，两个问题变得日益紧迫：人类为什么应该继续做科学，以及在做科学时应该如何使用AI？本文认为，只有回答了第一个问题，才能回答第二个问题。本文主张，科学的一个不可或缺的目标是“对世界的持久、集体性的人类理解，并以人类的目的和价值观为依归”。基于这一原则，本文推导出一个两部分的检验标准，用以判断何时可以将研究任务委托给AI。第一部分是“人类理解”，要求科学界仍然能够解释、捍卫和质疑由此产生的科学主张，并保留这样做所需的技能。第二部分是“人类主导”，要求关于哪些问题重要以及研究发现意味着什么的判断由负责任的人类来掌握。将该检验标准应用于贡献者角色分类法（CRediT）中研究的各个阶段，得出了相应建议。

    arXiv:2610.36116v1 Announce Type: new  Abstract: As AI begins to match or exceed human researchers at core scientific tasks, two questions become pressing: \emph{why} should humans keep doing science, and \emph{how} should they use AI when they do? This paper argues that the second can be answered only by answering the first. It holds that one indispensable aim of science is \emph{durable, collective human understanding of the world, pursued in service of human ends and values}. From this principle it derives a two-part test for when a research task can be delegated to AI. The first part, \emph{human understanding}, requires that the scientific community remain able to explain, defend and contest the resulting claims, and retain the skills to do so. The second, \emph{human direction}, requires that judgments about which questions matter and what findings mean stay with accountable humans. Applying the test to each phase of research in the Contributor Roles Taxonomy (CRediT) yields reco
    
[^219]: 上下文是一条流水线，而非一扇窗口：LLM智能体框架中的信息损耗

    Context Is a Pipeline, Not a Window: Information Attrition in LLM Agent Harnesses

    [https://arxiv.org/abs/2610.36114](https://arxiv.org/abs/2610.36114)

    该研究揭示了一个被忽视的关键问题：即使LLM智能体的上下文窗口利用率仅为5.75%，智能体框架仍会通过固定长度切分JSON的方式截断三分之一的工具结果，导致末尾数据不均匀丢失，使不同字段到达模型的比例差异悬殊（4.7%对99.3%），因此评估智能体不能只看窗口大小和答案正确率，还必须关注“信息损耗”。

    

    人们通常用两个数字来评判LLM智能体：上下文窗口的大小，以及最终答案的正确率。但这两个数字都忽略了第三件事：工具的输出在模型看到之前损失了多少。我们在一个真实的工具使用型智能体中测量了这种损耗。该智能体通过读取Linux内核跟踪记录来查找注入故障的原因。上下文窗口几乎是空的：在720次运行中，单个最大的提示词仅占用了窗口的5.75%。然而，三分之一的工具结果（11,189个中的3,827个）在到达模型之前就被截断了，而且每一个被截断的结果都变成了损坏的JSON。该框架通过在固定长度处切分JSON文本来截断每个结果，这并不会均匀地缩短结果，而是删除了位于末尾附近的任何数据。其中一个字段仅在4.7%的调用中到达模型，而在完全相同的结果中，另一个字段在99.3%的调用中到达模型。哪个字段包含故障的证据取决于故障进程启动的时间：早期……

    arXiv:2610.36114v1 Announce Type: cross  Abstract: LLM agents are judged by two numbers: the size of the context window, and how often the final answer is correct. Both numbers miss a third thing: how much of a tool's output is lost before the model ever sees it. We measured this loss in a real tool-using agent. The agent finds the cause of injected faults by reading Linux kernel traces. The context window was almost empty: across 720 runs, the single largest prompt used only 5.75% of it. Yet a third of all tool results (3,827 of 11,189) were cut before they reached the model, and every cut result became broken JSON. The harness cut each result by slicing its JSON text at a fixed length. This does not shorten a result evenly. It deletes whatever data sits near the end. One field reached the model in only 4.7% of calls. A second field, in the very same results, reached the model in 99.3% of calls. Which field held the fault's evidence depended on when the faulty process started: early o
    
[^220]: 神经网络中的标度与重整化：Grokking现象的有效重整化群描述

    Scaling and Renormalization in Neural Networks: An Effective Renormalization-Group Description of Grokking

    [https://arxiv.org/abs/2610.36113](https://arxiv.org/abs/2610.36113)

    该论文将重整化群物理理论引入神经网络Grokking现象研究，通过基于神经元耦合矩阵谱排序的实空间重整化方法，为神经网络的记忆到泛化转变建立了定量的物理描述框架。

    

    Grokking是深度神经网络中从记忆到泛化的延迟转变。我们研究了标度理论和重整化群（RG）理论能否描述这一动力学过程，从而为神经泛化转变建立一个定量的物理框架。针对二层神经网络上的模加法任务，一组训练好的模型集合定义了序参量（权重范数的平方）以及热容和磁化率的涨落代理量，其中训练轮次和权重衰减分别充当热学控制坐标和场控制坐标。为了将神经架构与实空间重整化群联系起来，我们通过对损失函数关于各神经元范数进行二阶展开，构建了一个精确的耦合矩阵，并利用其Fiedler向量来定义有效几何结构。这种谱排序产生了耦合强度的单调空间衰减，而这一特性在随机置换下并不存在。随后，我们通过块自旋聚合执行实空间重整化群变换……

    arXiv:2610.36113v1 Announce Type: cross  Abstract: Grokking is a delayed transition from memorization to generalization in deep neural networks (NNs). We examine whether scaling and renormalization group (RG) theories can describe this dynamics, establishing a quantitative physical framework for neural generalization transitions. For modular addition on a two-layer NN, an ensemble of trained models defines an order parameter (the squared weight norm) and fluctuation proxies for the heat capacity and the susceptibility, with training epoch and weight decay serving as thermal and field control coordinates. To bridge neural architectures to real-space RG, we construct an exact coupling matrix via a second-order expansion of the loss with respect to per-neuron norms and use its Fiedler vector to define an effective geometry. This spectral ordering yields a monotonic spatial decay of coupling strength that is absent under random permutations. We then execute real-space RG by block-spin aggr
    
[^221]: 半参数贝叶斯网络中的狄利克雷过程混合模型：一种新颖的条件密度估计方法

    Dirichlet Process Mixtures within Semiparametric Bayesian Networks: A Novel Conditional Density Estimation Approach

    [https://arxiv.org/abs/2610.36110](https://arxiv.org/abs/2610.36110)

    本文提出在半参数贝叶斯网络中用狄利克雷过程混合模型（DPM）替代核密度估计进行非参数条件概率分布建模，通过数据自适应聚类获得更紧凑的表示，在降低推理成本的同时提升了密度估计性能。

    

    半参数贝叶斯网络（SPBN）是一类灵活的概率图模型，通过低维条件概率分布（CPD）来表示连续变量的联合分布。现有的SPBN模型采用核密度估计器（KDE）进行非参数CPD建模，但其“每个观测对应一个混合分量”的表示方式会导致推理成本增加和密度估计误差，这可能成为应用的限制。在这项工作中，我们在SPBN框架内定义了一种基于贝叶斯非参数狄利克雷过程混合模型（DPM）的全新非参数组件CPD建模方法。与KDE不同，DPM能够对观测数据进行自适应聚类，在保留非参数密度估计灵活性的同时，获得更紧凑的表示。这使得推理成本更低，同时提升了密度估计性能。我们通过扩展实现了所提出的基于DPM的半参数贝叶斯网络（DPM-SPBN）。

    arXiv:2610.36110v1 Announce Type: cross  Abstract: Semiparametric Bayesian networks (SPBNs) are flexible probabilistic graphical models for representing joint distributions of continuous variables through lower-dimensional conditional probability distributions (CPDs). Existing SPBN models use kernel density estimators (KDEs) for non-parametric CPD modeling, but their one-component-per-observation mixture representation can lead to increased inference costs and density estimation errors, which can be limiting. In this work, we define a new CPD modeling approach for non-parametric components based on Bayesian non-parametric Dirichlet Process Mixtures (DPMs) within SPBNs. DPMs provide a data-adaptive clustering of observations, yielding a more compact representation while retaining the flexibility of non-parametric density estimation, unlike KDE. This results in lower inference costs while improving density estimation performance. We implement the proposed DPM-based SPBN (DPM-SPBN) by ext
    
[^222]: 日志致泄露：针对通过模型上下文协议使用工具的LLM智能体的提示注入攻击

    Log-To-Leak: Prompt Injection Attacks on Tool-Using LLM Agents via the Model Context Protocol

    [https://arxiv.org/abs/2610.36108](https://arxiv.org/abs/2610.36108)

    该论文提出Log-To-Leak攻击，通过MCP工具元数据中的提示注入，在合法工具调用完成后诱导LLM智能体执行额外的攻击者控制的日志记录调用，从而窃取用户查询与响应数据，在18个模型-服务器组合中实现了62.64-100%的攻击成功率。

    

    LLM智能体越来越多地通过模型上下文协议（MCP）访问外部工具，但用于描述这些工具的自然语言元数据形成了一个提示注入攻击面。在这一已确立的工具污染攻击面内，我们研究了Log-To-Leak，这是一种附加式的后验日志记录攻击，它会在最后一次合法调用之后诱导一次由攻击者控制的独立调用。其载荷针对用户查询、MCP服务器名称、服务器响应和智能体响应。我们通过触发器、工具绑定、合理化说辞和施压四个要素将注入指令操作化，并分别测量调用成功率、载荷保真度以及合法行为的保持程度。我们使用涵盖有效性、实用性和效率的五个指标评估了五个MCP服务器和四个LLM智能体。在18个具有有效工具执行的模型-服务器组合中，Log-To-Leak实现了62.64-100%的攻击成功率（ASR）和76.55-95%的条件性日志记录准确率（LA）。

    arXiv:2610.36108v1 Announce Type: cross  Abstract: LLM agents increasingly access external tools through the Model Context Protocol (MCP), but the natural-language metadata used to describe those tools creates a prompt-injection surface. Within this established tool-poisoning surface, we study Log-To-Leak, an additive post-hoc logging attack that induces a separate attacker-controlled call after the final legitimate call. Its payload targets the user query, MCP server name, server response, and agent response. We operationalize the injected directive through Trigger, Tool Binding, Justification, and Pressure, and separately measure call success, payload fidelity, and preservation of legitimate behavior. We evaluate five MCP servers and four LLM agents using five metrics spanning effectiveness, utility, and efficiency. Across 18 model-server pairs with valid tool execution, Log-To-Leak achieves an attack success rate (ASR) of 62.64-100% and conditional logging accuracy (LA) of 76.55-95.
    
[^223]: CohortFlow：具有局部语义修正能力的可引导临床队列筛选智能体

    CohortFlow: A Steerable Clinical Cohort Curation Agent with Localized Semantic Correction

    [https://arxiv.org/abs/2610.36100](https://arxiv.org/abs/2610.36100)

    CohortFlow通过将自然语言修正转化为对结构化队列规范的局部更新，并借助显式语义依赖关系选择性失效受影响的计算，在保持与完全重新生成相当准确率的同时，将修正延迟降低最高64.8%、模型调用次数减少38.1%。

    

    面向临床队列筛选的语言模型智能体应能适应不断演变的入组标准，而无需重复处理未发生变化的要求。我们提出了CohortFlow，这是一种可引导的智能体，它将自然语言修正表示为对结构化队列规范的局部更新，并将其与一个持久的执行图进行协调。通过显式记录的语义依赖关系，系统能够选择性地使受影响的计算失效，同时保留有效的先前结果。在源自临床试验入组标准的100个场景（共包含126轮修正）上，使用相同的8B模型时，局部引导方法实现了77.0%的要求微平均F1分数，而完全重新生成方法为76.4%，这一微小的数值优势在统计上并不显著（p=0.493）。局部引导方法将完成后的平均修正延迟降低了64.8%，将执行过程中的平均修正延迟降低了44.3%，模型调用次数减少了38.1%。在另一项单独的概念验证评估中……

    arXiv:2610.36100v1 Announce Type: new  Abstract: Language model agents for clinical cohort curation should accommodate evolving eligibility criteria without repeatedly processing unchanged requirements. We introduce CohortFlow, a steerable agent that represents natural language corrections as localized updates to a structured cohort specification and reconciles them with a persistent execution graph. Explicitly recorded semantic dependencies enable affected computations to be selectively invalidated while preserving valid prior results. On 100 scenarios derived from clinical-trial eligibility criteria, comprising 126 correction turns, localized steering achieves 77.0% requirement micro-F1 versus 76.4% for full regeneration using the same 8B model, a small numerical advantage that is not statistically significant ($p=0.493$). It reduces mean correction latency by 64.8% after completion and 44.3% during execution, with 38.1% fewer model calls. In a separate proof-of-concept evaluation ov
    
[^224]: 提示究竟有多重要？金融推理中后见之明自蒸馏的适用条件

    How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning

    [https://arxiv.org/abs/2610.36082](https://arxiv.org/abs/2610.36082)

    提出Fin-SD方法，通过让学生模型自己发现并批评首次错误生成提示，再以错误前缀加提示的条件进行自蒸馏，从而在无需外部教师或完整参考解答的情况下提升小模型的多步金融推理能力。

    

    在数据驻留限制下运营的金融机构只能使用体积小到足以本地部署的开源权重模型，而这些模型在多步定量推理方面落后于前沿系统。在线策略蒸馏（OPD）通过在学生模型自身的采样轨迹上进行监督来缩小这一差距，但传统的OPD需要一个强大的外部教师模型，而在线策略自蒸馏（OPSD）则以完整的参考解答作为条件。我们探究了这种信号中究竟有多少是必要的。我们提出了Fin-SD，一种面向多步推理的本地化后见之明自蒸馏方法。当学生模型无法解决某个问题时，学生模型自己识别出第一个错误的推理步骤，并对该步骤写出简短的批评意见。我们将学生模型的轨迹在错误发生之前截断（旨在获得正确的前缀），让学生模型的一个冻结副本以相同前缀加上提示为条件，然后在短（原文在此处截断）

    arXiv:2610.36082v1 Announce Type: cross  Abstract: Financial institutions operating under data-residency constraints are restricted to open-weight models small enough to locally host, which trail frontier systems on multi-step quantitative reasoning. On-policy distillation narrows such gaps by supervising a student on its own rollouts, but traditional OPD requires a strong external teacher, and on-policy self-distillation (OPSD) conditions on a full reference solution. We ask how much of that signal is necessary. We present Fin-SD, a localized hindsight self-distillation recipe for multi-step reasoning. Given a problem the student fails, the student itself identifies the first incorrect reasoning step and writes a short critique of that step. We truncate the student's trajectory immediately before the error (aiming to obtain a correct prefix), condition a frozen copy of the student on the same prefix plus the hint, and distill that hint-conditioned teacher into the student over a short
    
[^225]: Trinity：面向多模态智能体的能力驱动递归自我改进

    The Trinity:Capability-Driven Recursive Self-Improvement for Multimodal Agents

    [https://arxiv.org/abs/2610.36077](https://arxiv.org/abs/2610.36077)

    提出了The Trinity递归自我改进框架，通过Creator、Explorer、Evolver三个角色将世界生成与Harness进化相耦合，并以共享能力需求为基础将任务树重组为分层能力树，从而实现多模态智能体的能力驱动自我进化。

    

    多模态智能可以被视为在人类引导下，通过交互实现的世界与智能体之间的共同进化。然而，扩展这一过程面临两个瓶颈：构建异构且可验证的环境成本高昂，而人工的Harness迭代无法跟上日益增长的任务多样性和复杂性。一个关键洞察是，具有不同模态和表面形式的任务可以共享相同的底层能力需求。因此，我们提出了The Trinity，这是一个递归自我改进框架，通过三个角色——Creator（创造者）、Explorer（探索者）和Evolver（进化者）——将世界生成与Harness进化耦合起来。基于经过验证的执行轨迹，Evolver将任务组织的基础从表面类别转向共享需求，将原始任务树重组为分层的能力树。我们提出了能力驱动的Real2Sim2Real：对于处于同一节点的异构任务，Creator将其共同的能力……（摘要在此处截断）

    arXiv:2610.36077v1 Announce Type: new  Abstract: Multimodal intelligence can be viewed as human-guided co-evolution between worlds and agents through interaction. However, scaling this process faces two bottlenecks: constructing heterogeneous, verifiable environments is costly, and manual Harness iteration cannot keep pace with growing task diversity and complexity. A key insight is that tasks with different modalities and surface forms can share the same underlying capability requirements. We therefore introduce The Trinity, a recursive self-improvement framework coupling world generation with Harness evolution through three roles: Creator, Explorer, and Evolver. Drawing on verified execution trajectories, Evolver shifts the basis of task organization from surface categories to shared requirements, reorganizing the original task tree into a hierarchical capability tree. We introduce Capability-Driven Real2Sim2Real: for heterogeneous tasks at the same node, Creator abstracts their comm
    
[^226]: Rollback：世界模型中面向策略改进的经验回溯

    Rollback: Experience Backtracking in World Models for Policy Improvement

    [https://arxiv.org/abs/2610.36068](https://arxiv.org/abs/2610.36068)

    提出Rollback策略改进范式，利用世界模型回溯关键决策点并分支探索反事实动作，将反事实结果转化为细粒度的动作级监督信号，无需单独的价值模型即可改进机器人策略。

    

    动作条件化的世界模型为改进机器人策略提供了一种可扩展的替代方案，避免依赖昂贵的高真实世界交互。然而，现有的基于世界模型的方法大多将世界模型仅视为物理环境的前向代理，未能利用一项任何物理环境都无法提供的能力：回溯到过去的决策点以探索反事实的未来。我们提出了Rollback，这是一种策略改进范式，它将世界模型从被动模拟器转变为反事实引擎：该方法回溯到关键决策点，分支出替代动作，并对比不同动作的结果，从而将成功归因于单个决策。这使得“想象”集中在最关键之处，并将反事实结果提炼为细粒度的、动作级别的监督信号，而无需单独训练的价值模型。我们构建了一个可扩展的训练系统，将分布式世界模型搜索与异步真机机器人（数据收集）统一起来。

    arXiv:2610.36068v1 Announce Type: cross  Abstract: Action-conditioned world models offer a scalable alternative to costly real-world interaction for improving robot policies. However, existing world-model-based methods largely treat world models as forward-only surrogates of the physical environment, leaving untapped a capability no physical environment affords: rewinding to past decision points to explore counterfactual futures. We introduce Rollback, a policy improvement paradigm that turns world models from passive simulators into counterfactual engines: it rolls back to critical decision points, branches into alternative actions, and contrasts their outcomes to attribute success to individual decisions. This concentrates imagination where it matters most and distills counterfactual outcomes into fine-grained, action-level supervision without a separately learned value model. We build a scalable training system that unifies distributed world-model search with asynchronous real-robot
    
[^227]: 编程智能体还需要调试器吗？基于前沿模型的故障定位逼近准确率上限

    Do Coding Agents Still Need a Debugger? Frontier-Model-Based Fault Localization at the Accuracy Ceiling

    [https://arxiv.org/abs/2610.36015](https://arxiv.org/abs/2610.36015)

    该研究通过在610个注入故障的大规模实验发现，基于前沿大模型的编程智能体仅凭源代码推理即可将故障定位准确率提升至98%以上的天花板水平，调试器的额外辅助收益已微乎其微。

    

    背景：为了定位故障，编程智能体可以仅凭源代码进行推理，也可以借助调试器，通过运行程序并在交互式调试器中单步执行来获得辅助。虽然早期的智能体在静态执行追踪和调试器操作方面表现不佳，但前沿大语言模型在这两方面均已显著提升。目标：我们评估编程智能体在不使用工具的情况下，在原始程序和变换后程序上定位故障的准确程度，调试器辅助能在多大程度上提升准确率以及代价是什么，以及当调试器可选时智能体如何使用它。方法：我们将610个单行故障注入144个LeetCodeDataset解答中，并用五种保持行为不变的变换配方重写每个故障程序。基于Sol 6.1、Luna 6和Qwen 3.8的智能体在无调试器辅助、可选调试器辅助或鼓励使用调试器辅助的条件下对每个故障进行定位。结果：在不使用工具的情况下，各智能体在原始程序中定位了98.2%至99.9%的故障，在变换后程序中定位了96.7%至99.9%的故障，损失了（原文摘要在此处截断）。

    arXiv:2610.36015v1 Announce Type: cross  Abstract: Background: To localize a fault, a coding agent can reason from source code alone or use debugger assistance by running the program and stepping through it in an interactive debugger. While earlier agents struggled with static execution tracing and debugger operation, frontier LLMs have improved at both. Objective: We evaluate how accurately coding agents localize faults without tools across original and transformed programs, how much debugger assistance improves accuracy and at what cost, and how agents use it when optional. Methods: We seed 610 single-line faults into 144 LeetCodeDataset solutions and rewrite each faulty program with five recipes of behavior-preserving transformations. Agents based on Sol 6.1, Luna 6, and Qwen 3.8 localize each fault with no, optional, or encouraged debugger assistance. Results: Without tools, the agents localized 98.2 to 99.9% of faults in original and 96.7 to 99.9% in transformed programs, losing a
    
[^228]: CIVER：面向可靠代码评审的因果干预验证

    CIVER: Causal Intervention Verification for Reliable Code Review

    [https://arxiv.org/abs/2610.36010](https://arxiv.org/abs/2610.36010)

    CIVER提出了一种基于因果干预的验证框架，通过识别代码评审发现背后的核心问题来区分真实缺陷与误报，从而提升LLM代码评审的可靠性。

    

    代码评审对于维护软件仓库至关重要，它通过识别缺陷、提高代码质量并支持可靠的软件演化发挥作用。随着基于大语言模型（LLM）的编码智能体日益普及，越来越多的代码贡献和仓库变更由这些智能体生成。因此，基于LLM的代码评审智能体正变得越来越普遍，用于评估这些贡献并维护软件质量。然而，LLM评审者常常生成包含大量潜在问题的冗长评审，其中许多是误报或缺乏充分证据。这些不可靠的评审会误导开发者和下游编码智能体，因此亟需区分可操作的发现与缺乏依据的断言。我们提出了CIVER，一个基于因果干预的框架，用于识别代码评审发现背后的核心问题，并区分真实缺陷与误报。CIVER融合……

    arXiv:2610.36010v1 Announce Type: cross  Abstract: Code review is essential for maintaining software repositories by identifying defects, improving code quality, and supporting reliable software evolution. With the growing adoption of LLM-based coding agents, an increasing number of code contributions and repository changes are generated by these agents. Consequently, LLM-based code review agents are becoming increasingly prevalent to evaluate these contributions and maintain software quality. However, LLM reviewers often generate lengthy reviews containing numerous potential issues, many of which are false positives or lack sufficient evidence. These unreliable reviews can mislead developers and downstream coding agents, motivating the need to distinguish actionable findings from unsupported claims. We propose CIVER, a causal intervention-based framework that identifies the core issues underlying code-review findings and distinguishes genuine defects from false positives. CIVER integr
    
[^229]: DuplexTune：面向内存高效混合专家模型适配的耦合状态重放

    DuplexTune: Coupled-State Replay for Memory-Efficient Mixture-of-Experts Adaptation

    [https://arxiv.org/abs/2610.36009](https://arxiv.org/abs/2610.36009)

    DuplexTune通过耦合隐藏状态重放框架，在单块H800上以38.8 GB峰值内存适配Qwen1.5-MoE模型，较检查点微调降低约40%内存并提升吞吐量，任务性能相当，但在多语言能力上存在一定权衡。

    

    稀疏专家执行并不能消除适配混合专家语言模型所需的存储状态。我们提出DuplexTune，一个通过预训练注意力模块和专家模块重放耦合隐藏状态的框架。可训练的宽度桥接将两个残差流连接到主干网络。桥接预热先于主干网络适配进行，同时路由器权重保持冻结。我们分析了隐式状态恢复方程，并确定了重建所需的收敛与重放条件。在Qwen1.5-MoE-A2.7B上的单块H800评估显示，峰值内存为38.8 GB，而采用检查点机制的监督微调需要64.8 GB，吞吐量分别为25.1与19.9样本/秒。DuplexTune在MMLU上达到67.2%，GSM8K上达到75.6%，MT-Bench得分为7.69，但其多语言得分低于使用检查点机制的基线。这些结果刻画了状态存储、吞吐量与适配质量之间依赖于配置的权衡关系。

    arXiv:2610.36009v1 Announce Type: cross  Abstract: Sparse expert execution does not remove the stored states needed to adapt a mixture-of-experts language model. We present DuplexTune, a framework for replaying coupled hidden states through pretrained attention and expert modules. Trainable width bridges connect two residual streams to the backbone. Bridge warm-up precedes backbone adaptation, with router weights kept frozen. We analyze the implicit state-recovery equation and identify the convergence and replay conditions needed for reconstruction. On Qwen1.5-MoE-A2.7B, the single-H800 evaluation reports 38.8 GB peak memory, versus 64.8 GB for supervised fine-tuning with checkpointing, and throughput of 25.1 versus 19.9 samples/s. DuplexTune achieves 67.2% MMLU, 75.6% GSM8K, and 7.69 MT-Bench, with a weaker multilingual score than the checkpointed baseline. The results characterize a configuration-dependent tradeoff among state storage, throughput, and adaptation quality.
    
[^230]: MedJev：评估医学领域中的零样本“系统一”决策

    MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine

    [https://arxiv.org/abs/2610.36006](https://arxiv.org/abs/2610.36006)

    MedJev是一个可在笔记本电脑上本地运行的开源“系统一”医学决策模型，在多项医学知识和诊断基准上接近GPT-4的性能，同时保持良好的置信度校准，为快速高效的临床决策支持提供了新方案。

    

    系统一决策模型通过单次前向传播即可回答结构化问题而无需生成文本，为临床决策支持提供了快速且计算高效的方案。我们在涵盖医学知识、诊断以及基于MIMIC-IV和MIMIC-IV ED数据集的患者层面预测的十个医学基准上，评估了MedJev——一个可在笔记本电脑上本地运行的开源系统一模型。我们将MedJev与零样本LLM基线、已发表的GPT结果以及监督模型进行比较。MedJev在MedQA上达到72.5%的准确率，在USMLE上达到75.4%，在NEJM上达到80.1%，接近已发表的GPT-4性能，同时保持了良好的置信度校准（ECE为0.028至0.063）。在患者层面任务上，MedJev在院内死亡率预测中达到0.926的AUROC，超过了XGBoost（0.902），但在急诊科分诊入院等操作性结局上表现不及监督模型（0.745 vs. 0.809）。

    arXiv:2610.36006v1 Announce Type: cross  Abstract: System One decision models such as Jev answer structured questions in a single forward pass without generating text, offering fast and computationally efficient clinical decision support. We evaluate MedJev, an open System One model that runs locally on a laptop, across ten medical benchmarks spanning medical knowledge, diagnosis, and patient level prediction using MIMIC IV and MIMIC IV ED. We compare MedJev against zero shot LLM baselines, published GPT results, and supervised models. MedJev achieves accuracies of 72.5 percent on MedQA, 75.4 percent on USMLE, and 80.1 percent on NEJM, approaching published GPT 4 performance while maintaining well calibrated confidence estimates (ECE 0.028 to 0.063). On patient level tasks, MedJev achieves an AUROC of 0.926 for In Hospital Mortality, exceeding XGBoost (0.902), but underperforms supervised models on operational outcomes such as ED Triage Admission (0.745 vs. 0.809). Patient level predic
    
[^231]: PolicyMediator：基于具备演化技能的中介智能体实现OPC生态系统中的多区域政策学习

    PolicyMediator: Multi-Regional Policy Learning through Mediation Agents with Evolving Skills in OPC Ecosystems

    [https://arxiv.org/abs/2610.35997](https://arxiv.org/abs/2610.35997)

    提出认知约束下的领导者-多追随者Stackelberg框架PolicyMediator，通过技能不断演化的中介智能体将OPC的认知变化反馈与多区域政策学习形成闭环，解决数字游民式一人公司跨区域选址错配与政策调整低效的问题。

    

    一人公司是由个体创业者通过数字平台和基于大语言模型（LLM）的AI智能体运营的新兴企业。这些高技能创业者以数字游民为特征，对本地生态系统依赖有限。中国各地方政府采用差异化的区域扶持政策来吸引流动的OPC，这使得本地扶持转变为一个跨区域协调问题。OPC会比较不同区域以最大化自身效用，但不完善的政策认知可能导致其拒绝原本合适的落户地点。现有政策学习方法在固定认知下优化资源配置，而共识形成方法仅聚合各方观点，并未将认知变化反馈到政策学习中。这种脱节导致了错配的选址决策和低效的区域政策调整。我们提出PolicyMediator——一个认知约束下的领导者-多追随者Stackelberg框架，具有双边……（原文摘要在此处被截断）

    arXiv:2610.35997v1 Announce Type: cross  Abstract: One-person companies (OPCs) are emerging enterprises operated by individual entrepreneurs through digital platforms and LLM-based AI agents. These highly skilled entrepreneurs are characterized by being digital nomads, with limited reliance on local ecosystems. Governments in China use distinct regional support policies to attract mobile OPCs, turning local support into a cross-regional coordination problem. OPCs compare regions to maximize their own utility, but imperfect policy perceptions may cause them to reject otherwise suitable locations. Existing policy-learning methods optimize resources under fixed perceptions, while consensus-formation methods aggregate views without feeding cognitive changes back into policy learning. This disconnect produces mismatched location choices and inefficient regional policy adjustment. We introduce PolicyMediator, a cognition-constrained leader-multi-follower Stackelberg framework with bilateral 
    
[^232]: 联邦反洗钱中跨银行上下文信息的定价

    Pricing Cross-Bank Context in Federated Anti-Money Laundering

    [https://arxiv.org/abs/2610.35996](https://arxiv.org/abs/2610.35996)

    该论文提出利用交易双方银行已记录的交易在本地近似远程账户状态，实现无需额外隐私数据交换的联邦多层GNN反洗钱检测，并通过信息披露阶梯实验发现可观测基线已贡献最大收益，额外共享的跨银行上下文信息边际价值有限。

    

    大多数基于图的反洗钱（AML）方法都假设可以访问一个包含所有账户和支付信息的集中式交易图，但这种情况在现实中并不存在。联邦学习能够提供必要的隐私保障，然而跨越多个独立图的洗钱模式如何被有效检测仍不清楚。我们利用交易双方银行已经记录的交易，在本地近似远程账户的状态，从而无需额外的账户级信息交换即可实现多层图神经网络（GNN）计算。以此作为基线，我们通过一个信息披露阶梯来测试剩余的上下文信息是否值得共享：为每个远程账户增加两个交易计数，或共享对手方自身的GNN状态。在AMLworld HI-Small数据集上，跨越六种GNN骨干网络和多达32个数据孤岛的实验表明，可观测基线带来了最大的性能提升，并且随着数据碎片化导致可见部分缩小，这一优势依然保持；交易计数则带来了稳健的修正提升。

    arXiv:2610.35996v1 Announce Type: cross  Abstract: Most graph-based Anti-Money Laundering (AML) methods assume access to a pooled transaction graph containing all accounts and payments, which in reality does not exist. Federated learning offers the necessary privacy guarantees but it remains unclear how money laundering patterns spanning separate graphs can effectively be detected. We exploit transactions already recorded by both endpoint banks to approximate remote account states locally, enabling multi-layer Graph Neural Network (GNN) computation without additional account-specific exchange. Starting from this baseline, we test whether the remaining context is worth sharing, using a disclosure ladder that adds two transaction counts per remote account or the counterparty's own GNN states. On AMLworld HI-Small, across six GNN backbones and up to 32 silos, the observable baseline yields the largest gain and keeps it as fragmentation shrinks the visible slice; counts add a robust correc
    
[^233]: 逐层重打包：无需额外重分配过程的变长混合注意力训练

    Layerwise Repacking: Variable-Length Hybrid-Attention Training without an Extra Redistribution Pass

    [https://arxiv.org/abs/2610.35989](https://arxiv.org/abs/2610.35989)

    该论文提出逐层重打包方法，使混合注意力模型的变长训练中全注意力层使用宽SP16/32组、稀疏注意力层使用节点本地的小SP组，无需在层间边界进行额外的数据重分配，从而避免跨节点all-to-all通信7.0倍的性能损失和额外传输开销。

    

    变长训练在混合注意力模型的各层之间产生了相互冲突的并行需求。最长的样本在全注意力层需要16或32个进程的宽序列并行（SP）组，其成本随序列长度呈二次方增长。稀疏注意力层主要由近线性的MoE计算主导，因此在节点本地的SP8组内即可平衡大多数被评估的工作负载。如果这些层继续复用宽组，其all-to-all通信将跨越节点进行，观察到的有效速率降低了7.0倍。然而切换到更小的组会改变每个样本由哪些进程持有；在每个全注意力-稀疏注意力边界处进行单独的重分配将引入另一次数据传输。逐层重打包（Layerwise Repacking）为全注意力保留每个SP16/32父组，并将其样本重打包到节点本地的稀疏子组中，在完整注意力层之间保持隐状态。该布局变化借助两个现有的功能……（摘要原文在此处截断）

    arXiv:2610.35989v1 Announce Type: cross  Abstract: Variable-length training creates conflicting parallelism requirements across the layers of a hybrid-attention model. The longest samples require wide sequence-parallel (SP) groups of 16 or 32 ranks for full attention, whose cost grows quadratically with sequence length. Sparse-attention layers, dominated by near-linear MoE work, can instead balance most evaluated workloads within node-local SP8 groups. Reusing the wide groups for these layers pushes their all-to-all communication across nodes, where the observed effective rate is 7.0x lower. Yet switching to smaller groups changes which ranks own each sample; a separate redistribution at every full-sparse boundary would introduce another data transfer.   Layerwise Repacking keeps each SP16/32 parent for full attention and repacks its samples into node-local sparse children at SP{1, 2, 4, 8}, where hidden states persist between full layers. The layout change rides on the two existing fu
    
[^234]: PROBE：前沿编码智能体发现的Bug与维护者修复的Bug不同

    PROBE: Frontier Coding Agents Find Different Bugs Than Maintainers Fix

    [https://arxiv.org/abs/2610.35987](https://arxiv.org/abs/2610.35987)

    本文提出PROBE基准测试，揭示前沿编码智能体在仓库漏洞发现中召回率和精确率均很低，且其找到的漏洞与人类维护者实际修复的漏洞存在系统性差异。

    

    主动式漏洞发现是软件工程中的一个重要问题。尽管编码智能体已在该领域被广泛部署，但其能力仍未得到充分研究。特别是，在整个代码仓库中找出人类维护者会修复的所有漏洞这一具有挑战性的任务尚未被研究过。为了填补这一空白，我们提出了PROBE，一个包含20个多样化GitHub代码仓库在18个月内修复的全部2,457个漏洞的基准测试，并配套了一种将候选发现与该部分参考集进行匹配的方法。我们评估了基于多个前沿模型的编码智能体，发现它们相对于参考集的召回率均低于21%，精确率低于9%。虽然更强的模型将召回率从8%大幅提升至17%，但并不能提高精确率。令人惊讶的是，尽管精确率很低，LLM和人类评估者仍然判定85%的发现是有效的。

    arXiv:2610.35987v1 Announce Type: cross  Abstract: Proactive bug discovery is an important software-engineering problem. Even though coding agents are already widely deployed in this domain, their capabilities remain understudied. In particular, the challenging task of finding all bugs that human maintainers would repair across an entire repository has not been studied yet. To close this gap, we introduce PROBE, a benchmark consisting of all 2,457 bugs fixed over the course of 18 months across 20 diverse GitHub repositories, together with a method for matching candidate findings to this partial reference set. We evaluate coding agents across frontier models and find they all achieve a recall of below $21\%$ and a precision of less than $9\%$ with respect to the reference set. While stronger models increase recall substantially from $8\%$ to $17\%$, they do not improve precision. Surprisingly, despite the low precision, LLM and human evaluators still judge $85\%$ of findings valid. Cruc
    
[^235]: 让门控决定：大语言模型中量化条件内存的逐查找精度

    Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs

    [https://arxiv.org/abs/2610.35985](https://arxiv.org/abs/2610.35985)

    GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。

    

    条件记忆层（如DeepSeek的Engram）将大型哈希n-gram表附加到语言模型上，从DeepSeek-V4.1-Flash中移除该表会损失36.8个PopQA分数。低比特量化会最先丢失长尾事实，而标准补救措施——为频繁访问的行分配更多比特——反而适得其反：按访问频率进行精度提升的效果比随机分配还差。我们将这一问题追溯到缩放每个检索行的门控：一次查找的量化误差由其门控加权，该门控对频繁出现的n-gram较小，对长尾实体的罕见n-gram较大，且可以从1比特编码中计算得出。因此，GateRQ将每一行存储为1比特残差编码的阶梯结构，将第一级保留在GPU内存中以计算门控，仅对门控排名最高的查找读取更深层级。无需重新训练，GateRQ在DeepSeek-V4.1-Flash上将GPU表内存削减了7.8倍，相比FP8表使vLLM吞吐量提升11%-31%，同时……

    arXiv:2610.35985v1 Announce Type: cross  Abstract: Conditional memory layers, such as DeepSeek's Engram, attach large hashed n-gram tables to a language model, and removing them from DeepSeek-V4.1-Flash costs 36.8 PopQA points. Low-bit quantization loses long-tail facts first, and the standard remedy, giving more bits to frequently accessed rows, backfires: refining lookups by access frequency is worse than refining them at random. We trace this to the gate that scales each retrieved row. A lookup's quantization error is weighted by its gate, which is small for frequent n-grams, large for the rare n-grams of long-tail entities, and computable from a 1-bit code. GateRQ therefore stores each row as a ladder of 1-bit residual codes, keeps level 1 in GPU memory to compute the gate, and reads deeper levels only for the lookups the gate ranks highest. Without retraining, GateRQ cuts GPU table memory by 7.8x on DeepSeek-V4.1-Flash, which raises vLLM throughput by 11%-31% over FP8 tables, whil
    
[^236]: 通过数据生成策略提升深度学习模型性能：橄榄树病害分类

    Enhancing Deep Learning Model Performance through Data Generation Strategies: Classification of Olive Tree Diseases

    [https://arxiv.org/abs/2610.35984](https://arxiv.org/abs/2610.35984)

    该论文提出了一种融合传统数据增强、CutMix、MixUp和DCGAN的综合性数据生成策略，通过三种数据分布场景有效解决了橄榄树病害分类中的数据不足与类别不平衡问题，从而提升了深度学习模型的性能。

    

    深度学习模型在众多领域中非常有效，尤其是在植物病害诊断方面。然而，其性能往往受到数据不足和类别不平衡等挑战的限制。在我们的工作中，我们提出了一种全面的数据生成策略来克服这些限制。首先，我们应用了多种数据生成技术，包括传统数据增强方法、CutMix、MixUp，以及利用深度卷积生成对抗网络（DCGANs）进行合成图像生成，以创建新的数据样本。其次，我们在三种场景中利用这些生成的数据：（S1）平衡原始数据集的类别分布，（S2）增加每个类别的样本数量，（S3）在保持类别平衡的同时扩展数据集。第三，我们使用经过预训练和微调的EfficientNetV2B0架构，评估了这些数据分布（S1、S2、S3）对模型性能的影响。

    arXiv:2610.35984v1 Announce Type: new  Abstract: Deep learning models are highly effective in a wide range of domains, particu larly in plant disease diagnosis. However, their performance is often limited by challenges such as insufficient data and class imbalance. In our work, we propose a comprehensive data generation strategy to overcome these limitations. First, we apply various data generation techniques, including traditional augmentation methods, CutMix, MixUp, and synthetic image generation using Deep Convolu tional Generative Adversarial Networks (DCGANs), to create new data samples. Second, we exploit this generated data in three scenarios: (S1) balancing class distribution in the original dataset, (S2) increasing the number of samples per class, and (S3) expanding the dataset while maintaining balanced classes. Third, we evaluate the impact of these data distributions (S1, S2, S3) on model perfor mance using the EfficientNetV2B0 architecture with pre-trained and fine-tuned w
    
[^237]: 生成与验证：面向安全关键汽车软件工程的人机智能体协作

    Generation and Verification: Human-Agent Collaboration for Safety-Critical Automotive Software Engineering

    [https://arxiv.org/abs/2610.35982](https://arxiv.org/abs/2610.35982)

    该论文提出一个面向安全关键汽车 C/C++ 软件的编程助手，通过 MCP 协议向开发者开放生成、编辑、验证三个可按需调用的工具以实现人机协作验证，并针对汽车领域定制底层大语言模型。

    

    有关编程智能体的研究主要聚焦于生成质量。然而，安全关键的汽车软件提出了两项额外要求。其一，它高度重视验证，因为每一处代码变更都必须经过审查、追溯到需求，并通过 MISRA 和 AUTOSAR 等行业标准的检查后才能提交。其二，它是一个狭窄的领域，代码需遵循分层架构、嵌入式资源约束以及组织特定的接口规范，而这些内容在通用大语言模型（LLM）的训练数据中十分稀缺。我们提出一个面向汽车 C/C++ 的编程助手，以同时应对这两项要求。首先，它通过模型上下文协议向用户开放三个工具（生成、编辑和验证），使开发者能够按需对任意代码运行验证，而不必依赖智能体自行检查其输出。其次，我们不使用通用模型，而是通过提示工程等方式针对汽车领域对底层大语言模型进行定制（原文摘要在此处被截断）。

    arXiv:2610.35982v1 Announce Type: cross  Abstract: Research on coding agents mainly focuses on generation quality. Yet, safety-critical automotive software makes two additional demands. It prioritizes verification, since every change is reviewed, traced to a requirement, and checked against industry standards such as MISRA and AUTOSAR before it can be committed. It is also a narrow domain, written against layered architectures, embedded resource constraints, and organization-specific interface conventions that are scarce in the data general-purpose LLMs are trained on. We present a coding assistant for automotive C/C++ that addresses both demands. First, it exposes three tools to the user over the Model Context Protocol (generate, edit, and verify), allowing developers to run verify on any code on demand rather than relying on the agent to check its own output. Second, rather than using a general-purpose model, we customize the underlying LLM for the automotive domain through prompt en
    
[^238]: UnityBench-Mini3D：评估Unity中3D游戏开发的LLM智能体

    UnityBench-Mini3D: Evaluating LLM Agents for 3D Game Development in Unity

    [https://arxiv.org/abs/2610.35980](https://arxiv.org/abs/2610.35980)

    UnityBench-Mini3D是一个包含77个Unity项目级任务的基准，通过1,178个人工编写的引擎内里程碑测试对LLM智能体的3D游戏开发能力进行确定性评估，可提供部分得分、定位失败点，无需依赖主观的LLM评判器。

    

    游戏开发对代码智能体的考验方式与一般软件工程不同：正确性取决于引擎API、序列化场景状态和运行时行为之间的交互，而不仅仅是源代码本身。现有基准主要针对Godot等平台，并且严重依赖LLM评判器或粗粒度的任务级结果。我们提出了UnityBench-Mini3D，这是一个包含77个项目级Unity任务的基准，围绕七个由专家定义的技能类别（而非教程或游戏类型）组织，并设有三个难度等级。每个任务将一个完整的Unity项目与自然语言需求相配对，并使用1,178个留出的、由人工编写的里程碑测试进行评估，这些测试涵盖场景结构、组件配置和运行时行为。这种引擎内评估提供了确定性的部分得分，可以定位失败点，并区分部分实现与完整解决方案，而无需依赖主观评判器。在21个模型和9个（摘要在此处截断）……

    arXiv:2610.35980v1 Announce Type: cross  Abstract: Game development stresses coding agents in ways general software engineering does not: correctness depends on how engine APIs, serialized scene state, and runtime behavior interact, not on source code alone. Existing benchmarks mainly target platforms such as Godot and heavily rely on LLM judges or coarse task-level outcomes. We present UnityBench-Mini3D, a benchmark of 77 project-level Unity tasks organized around seven expert-defined skill categories rather than tutorials or genres, with three difficulty tiers. Each task pairs a complete Unity project with natural-language requirements and is evaluated using 1,178 held-out, human-authored milestone tests covering scene structure, component configuration, and runtime behavior. This in-engine evaluation provides deterministic partial credit, localizes failures, and distinguishes partial implementations from complete solutions without relying on subjective judges. Across 21 models and 9
    
[^239]: 基于图神经网络的供水管网多泄漏点定位方法

    Multiple Leak Localization in Water Distribution Networks through Graph Neural Networks

    [https://arxiv.org/abs/2610.35972](https://arxiv.org/abs/2610.35972)

    该研究提出了一种图神经网络模型，无需预先指定泄漏数量或穷举泄漏位置组合，即可在稀疏压力监测条件下定位供水管网中数量可变的多个同时泄漏，其准确率比传统机器学习方法最高提升12%和20%。

    

    当供水管网中同时发生多个泄漏时，泄漏定位极具挑战性，尤其是在监测设备稀疏的情况下。现有的数据驱动方法通常假设泄漏数量为单一或预先确定，或者需要对泄漏组合进行大量的枚举。本研究提出了一种图神经网络模型，能够定位数量可变的同时泄漏，既无需提前指定泄漏数量，也无需在训练时穷举所有可能的泄漏位置组合。该模型在两个不同规模的基准网络上进行了评估：31个节点的哈诺伊网络和更大的92个节点NET3网络，评估条件包括需求变化、稀疏的压力传感器覆盖、可变的泄漏幅度以及不同数量的同时泄漏。GNN的性能优于多层感知机、支持向量机、K近邻和随机森林分类器，准确率分别提高了12%和20%。

    arXiv:2610.35972v1 Announce Type: cross  Abstract: Leak localization in water distribution networks is challenging when multiple leaks occur simultaneously, particularly under sparse monitoring. Existing data-driven methods often assume a single or predefined number of leaks or require extensive enumeration of leak combinations. This study proposes a graph neural network (GNN) model that localizes a variable number of simultaneous leaks without specifying the leak count in advance or exhaustively generating every possible combination of leak locations for training. The model is evaluated on two benchmark networks of different sizes, the 31-node Hanoi network and the larger 92-node NET3 network, under variable demand, sparse pressure-sensor coverage, variable leak magnitudes, and varying numbers of simultaneous leaks. The GNN outperforms multilayer perceptron, support vector machine, K-nearest neighbors, and random forest classifiers by up to 12% and 20% in accuracy respectively in Hano
    
[^240]: EMMA：面向多智能体工作流合成的结构化情景记忆

    EMMA: Structured Episodic Memory for Multi-Agent Workflow Synthesis

    [https://arxiv.org/abs/2610.35966](https://arxiv.org/abs/2610.35966)

    提出了EMMA框架，通过结构化情景记忆进行多智能体工作流合成，解决了单智能体自适应中的跨任务干扰问题，以及现有多智能体演化方法中更新规则不清、变化范围依赖骨干模型偏好而非实际失败经验的问题。

    

    在专业领域部署LLM智能体通常需要通用模型所缺乏的细粒度任务特定程序和知识，然而参数训练成本高昂且不可解释，而手工设计的智能体系统则将成本转嫁给专家人力。情景记忆虽然在上下文层面进行更新，但单智能体自适应难以缓解跨任务干扰，尤其是在多应用导航或数据池分析等复杂任务中。尽管多智能体演化缓解了这一问题，但现有方法缺乏对更新规则的清晰定义，而是依赖骨干模型来决定系统在何处以及以多大程度进行改变。这种范围界定不清的演化使得学习遵循骨干模型的结构偏好而非观测到的失败：某些骨干模型倾向于积累技能，而另一些则倾向于生成智能体。我们提出了EMMA（面向多智能体工作流合成的结构化情景记忆），该方法将配置……

    arXiv:2610.35966v1 Announce Type: cross  Abstract: Deploying LLM agents in specialized domains often require fine-grained task-specific procedures and knowledge a general model lacks, yet parameter training is costly and non-interpretable, and hand-engineered agent systems shift the cost to expert labor. Episodic memory updates at context level, but single agent adaptation struggles to alleviate cross-task interference, especially with complex tasks such as multi-application navigation or data pool analysis. Although multi-agent evolution alleviates this problem, existing methods lack clear definitions on update rules and rely on the backbone model to decide where and how much the system changes. This ill-scoped evolution makes learning follow the backbone's structural preferences rather than the observed failures: one backbone accumulates skills while another spawns agents. We present EMMA (structured Episodic Memory for Multi-Agent workflow synthesis), which treats the configuration 
    
[^241]: 多智能体路径规划的超逻辑视角

    A Hyper-Logical View on Multi-Agent Path Planning

    [https://arxiv.org/abs/2610.35964](https://arxiv.org/abs/2610.35964)

    本文提出用超属性时序逻辑HyperLTL及其扩展HyperQPTL统一建模多智能体路径规划的多种变体，为变体比较、跨智能体约束表达、最优性描述以及与验证和因果性技术的连接提供了统一的规约语言。

    

    多智能体路径规划（MAPP）要求为多个智能体规划无碰撞的路径。MAPP在现实生活中有众多应用，包括自动驾驶车辆的导航，因此针对不同的约束、需求和目标存在许多变体。针对每一种变体，通常都需要开发新的专门算法。本文采取了一种互补的、以规约为中心的视角：我们证明了许多MAPP变体可以用HyperLTL——一种对多条路径及其随时间演化进行推理的时序逻辑——以及其omega正则扩展HyperQPTL来进行建模。本文的贡献并非声称当前的超属性模型检测器能够取代专门的MAPP求解器，而是这种超逻辑视角为比较不同的MAPP变体、添加跨智能体约束、表达最优性，以及将MAPP与超属性的验证、解释和因果性分析技术相连接，提供了一种统一的规约语言。

    arXiv:2610.35964v1 Announce Type: cross  Abstract: Multi-Agent Path Planning (MAPP) requires collision-free paths for multiple agents. MAPP has numerous real-life applications, including the navigation of automated vehicles, and has therefore many variants considering different constraints, requirements, and objectives. For each such variant, a new algorithmic treatment is often developed. This paper takes a complementary, specification-centered view: we show that many MAPP variants can be modeled in HyperLTL, a temporal logic that reasons about multiple paths and their evolution through time, and in its omega-regular extension HyperQPTL. The contribution is not a claim that current hyperproperty model checkers replace specialized MAPP solvers. Rather, the hyper-logical view gives a unified language for comparing MAPP variants, adding cross-agent constraints, expressing optimality, and connecting MAPP to verification, explanation, and causality techniques for hyperproperties.
    
[^242]: AgentBaton：智能体任务中的高效模型协作

    AgentBaton: Efficient Model Collaboration in Agentic Tasks

    [https://arxiv.org/abs/2610.35959](https://arxiv.org/abs/2610.35959)

    该论文提出AgentBaton，系统研究了在长程智能体任务中通过模型交接实现强大模型与高效模型协作的关键设计（交接方向与上下文传递），以在保持任务准确性的同时大幅降低成本。

    

    解决长程智能体任务需要语言智能体在多个步骤中与其环境进行交互，而可靠的性能通常依赖于云服务提供商提供的昂贵但能力强大的模型。与此同时，一些性能稍逊但成本远低的替代性高效模型正在涌现，包括较低层级的云模型以及可部署在本地硬件上的更小的开源权重模型。这引出了一个相比传统查询级路由尚未被充分探索的自然问题：能力强大的模型与高效模型能否在单个长程智能体任务内进行协作，在保持任务准确性的同时大幅降低成本？在本文中，我们通过模型交接，即一个模型接替另一个模型的方式，系统地研究了此类协作的关键设计选择，重点关注交接方向和上下文传递。我们发现，从高效模型到强大模型的交接通常表现不如仅使用强大模型……

    arXiv:2610.35959v1 Announce Type: cross  Abstract: Solving long-horizon agentic tasks requires language agents to interact with their environments over multiple steps, and reliable performance often comes from the expensive capable models available through cloud providers. Meanwhile, alternative efficient models with inferior performance but a much lower cost are emerging, including lower-tier cloud models and smaller open-weights models deployable on local hardware. This raises a natural question that remains less explored than traditional query-level routing: can capable and efficient models collaborate within a single long-horizon agentic task to substantially reduce cost while preserving task accuracy? In this paper, we systematically investigate key design choices for such collaboration through model handoffs, where one model takes over from another, focusing on handoff direction and context transfer. We find that efficient-to-capable handoffs generally underperform capable-only e
    
[^243]: 面向信道预测的结构化时频表示

    Structured Time-Frequency Representation for Channel Prediction

    [https://arxiv.org/abs/2610.35956](https://arxiv.org/abs/2610.35956)

    该论文提出FAR-CSI框架，通过在联合的时间与工作频段位置域中构建FDD感知的结构化表示，使历史UL观测与未来DL预测目标在位置空间中占据分离区域，从而实现FDD系统中从上行观测到下行CSI的准确信道预测。

    

    准确的信道状态信息（CSI）预测对于缓解信道老化问题以及在高移动性无线系统中支持可靠的下行链路传输至关重要。在频分双工（FDD）系统中，从历史上行（UL）观测数据预测未来下行（DL）CSI需要同时进行时间外推和跨频段推断。然而，仅将历史UL定义为输入、未来DL定义为目标，并未说明在预训练预测器的位置空间中应如何表示流逝的时间以及UL到DL的频段分离。FAR-CSI在联合的时间与工作频段位置域中建立了一种FDD感知的信道预测表示。在该位置域中，历史UL观测与未来DL目标根据其时间和工作频段角色占据相互分离的区域，从而保留了已观测的UL上下文与未观测的DL预测目标之间的区别。该表示通过一个统一的结构化路由规则来实例化，该规则支配……

    arXiv:2610.35956v1 Announce Type: cross  Abstract: Accurate CSI forecasting is essential for mitigating channel aging and supporting reliable downlink transmission in high-mobility wireless systems. In FDD systems, forecasting future DL CSI from historical UL observations requires joint temporal extrapolation and cross-band inference. However, defining historical UL as the input and future DL as the target does not specify how elapsed time and UL-to-DL band separation should be represented in a pretrained predictor's positional space. FAR-CSI establishes an FDD-aware representation of channel forecasting within a joint temporal and operating-band positional field. Within this field, historical UL observations and future DL targets occupy segregated regions according to their temporal and operating-band roles, preserving the distinction between observed UL context and unobserved DL prediction targets. This representation is instantiated through a unified structured routing rule that gov
    
[^244]: ResCom：面向大语言模型量化的激活残差补偿

    ResCom: Activation Residual Compensation for Large Language Model Quantization

    [https://arxiv.org/abs/2610.35955](https://arxiv.org/abs/2610.35955)

    ResCom提出一种低秩激活残差补偿方法，通过输出感知校准以闭式形式利用激活舍入残差校正层输出误差，无需梯度训练即可显著提升大语言模型4比特权重-激活量化的精度。

    

    低比特权重-激活量化可以降低部署大语言模型所需的内存和计算量，但即使经过校准，激活舍入仍可能导致精度下降。基于舍入后输入的重建无法区分同一量化区间内的激活值。我们提出ResCom，一种低秩校正方法，利用激活舍入残差来补偿由此产生的层输出误差。该方法的输出感知校准流程根据残差方向对输出重建的贡献来选择这些方向，并以闭式形式计算校正，无需梯度训练，也无需更新已校准的权重。在采用4比特权重和激活的Llama-3-8B上，秩为32的注意力校正将WikiText-2困惑度从7.708降至7.147，MMLU提升2.89个百分点，五样本GSM8K准确率从23.96%提升至36.16%。对参数规模从1B到70B的模型以及在留出的C4测试集上的评估（摘要在此处截断）……

    arXiv:2610.35955v1 Announce Type: cross  Abstract: Low-bit weight--activation quantization reduces the memory and computation needed to deploy large language models, but activation rounding can still degrade accuracy after calibration. Reconstruction from rounded inputs cannot distinguish activations within the same quantization bin. We propose ResCom, a low-rank correction that uses the activation rounding residual to compensate for the resulting layer-output error. An output-aware calibration procedure selects residual directions by their contribution to output reconstruction and computes the correction in closed form, without gradient training or updates to the calibrated weights. On Llama-3-8B with 4-bit weights and activations, rank-32 attention corrections reduce WikiText-2 perplexity from 7.708 to 7.147, improve MMLU by 2.89 percentage points, and raise five-shot GSM8K accuracy from 23.96 to 36.16 percent. Evaluations on models from 1B to 70B parameters and on held-out C4 test t
    
[^245]: 基于课程驱动的深度强化学习实现火箭自主远距着陆

    Curriculum-Driven Deep Reinforcement Learning for Autonomous Far-Range Rocket Landing

    [https://arxiv.org/abs/2610.35954](https://arxiv.org/abs/2610.35954)

    本文提出一种课程驱动的深度强化学习框架，通过三阶段课程设计、评论家网络预热和防灾难性遗忘的复演机制，并采用完整3×3旋转矩阵的高维观测，实现了火箭从远距初始条件到精确着陆的自主控制。

    

    可重复使用运载火箭的自主垂直着陆是航空航天工程中一项关键而又艰巨的挑战，其主要原因在于高度非线性的动力学特性、严苛的精度要求以及固有的环境不确定性。尽管强化学习已成为传统基于模型的控制方法的有前景替代方案，但现有方法通常依赖平坦的训练范式和高度简化的初始条件，极大地限制了其实际应用能力。本文提出了一种全面的课程驱动深度强化学习框架，旨在同时掌握远距导航和精确着陆。通过采用三阶段课程学习，辅以评论家网络预热程序和防止灾难性遗忘的复演机制，并利用基于完整3×3旋转矩阵的高维观测向量，智能体被训练从初始…（摘要在此截断）

    arXiv:2610.35954v1 Announce Type: cross  Abstract: Autonomous vertical landing of reusable launch vehicles is a critical yet formidable challenge in aerospace engineering, primarily due to highly nonlinear dynamics, stringent precision require- ments, and inherent environmental uncertainties. While reinforcement learning (RL) has emerged as a promising alternative to traditional model-based control, existing approaches frequently rely on flat training paradigms and highly simplified initial conditions, drastically limiting their real-world applicability. This paper proposes a comprehensive curriculum-driven deep reinforcement learning framework designed to master both far-range navigation and precision landing. By employing a three-stage curriculum augmented with a critic warm-up procedure and a rehearsal mechanism to prevent catastrophic forgetting, alongside a high-dimensional observation vector utilizing a full 3 X 3 rotation-matrix, the agent is trained to navigate from an initial 
    
[^246]: 数绷带：论如何衡量AI节省的时间

    Counting the Bandages: On measuring the time that AI saves

    [https://arxiv.org/abs/2610.35942](https://arxiv.org/abs/2610.35942)

    论文主张应以结合逃逸缺陷的“账本观点”而非单纯计时的“秒表观点”来衡量AI节省的时间，定义了节省时间、超额逃逸缺陷和盈亏平衡缺陷成本等量，证明只要仍需人工检查，校准良好的工作者逃逸缺陷率与模型质量无关，并从公开数据复现METR开发者研究，显示同一数据可支持19%或24%的速度下降。

    

    arXiv:2610.35942v1 公告类型：新论文 摘要：我探讨了AI辅助“节省时间”究竟意味着什么。我对比了两种观点：一种是“秒表观点”，即节省的时间就是时钟所记录的时间；另一种是“账本观点”，即节省的时间是一种反事实量，必须结合工作中被放过去的错误来解读。我论证支持后者。论证分为四个部分。我定义了一组不依赖具体模型的量——节省的时间和超额逃逸缺陷数，以及由它们构造出的一个决策数字：盈亏平衡缺陷成本。我给出了一个关于检查工作的基线模型，在该模型中，只要模型还没有好到可以不检查的程度，一个校准良好的工作者的逃逸缺陷率就不依赖于模型的好坏；检查时间随模型错误率的对数增长；而过度信任虽然在时钟上买来了几分钟，却要以总时间上更大的代价为成本。我说明了随机化研究能识别什么、不能识别什么，并从公开数据复现了METR的开发者研究：同样的246个问题既支持19%也支持24%的速度下降，取决于……（原文摘要在此处被截断）

    arXiv:2610.35942v1 Announce Type: new  Abstract: I consider what it means for AI assistance to save time. I contrast a stopwatch view, on which time saved is whatever the clock records, with a ledger view, on which it is a counterfactual that has to be read next to the errors the work lets through, and I argue for the latter. The argument has four parts. I define a model-free pair of quantities, time saved and excess escaped defects, and one decision number built from them: the break-even defect cost. I give a baseline model of checking in which, until the model is good enough not to check, a calibrated worker's escaped-defect rate does not depend on how good the model is; checking time grows with the logarithm of the model's error rate; and over-trust buys minutes on the clock at a larger cost in total minutes. I show what randomized studies can and cannot identify, and I replicate METR's developer study from its public data: the same 246 issues support a slowdown of 19% or 24% depend
    
[^247]: 谁说服了谁？追踪大语言模型在多智能体系统中如何改变主意

    Who Convinced Whom? Tracing How LLMs Change Their Minds in Multi-Agent Systems

    [https://arxiv.org/abs/2610.35930](https://arxiv.org/abs/2610.35930)

    该论文提出MAD Bench基准，通过行为和内部干预追踪多智能体辩论中大语言模型改变观点的机制，发现辩论后的答案一致性往往源于社会从众效应而非独立验证，且消息的来源归因会显著改变其影响力。

    

    多智能体辩论（MAD）旨在通过论点交流改进语言模型的决策，然而辩论后达成的一致意见并不一定反映独立的相互印证。我们通过行为和内部干预研究同伴通信如何影响个体决策和集体结果。我们提出了MAD Bench，一个包含2,400个问题的基准，涵盖知识与推理、价值观与道德判断、安全与拒绝三大领域，并采用轮次匹配的独立反思对照条件评估了三个开源权重模型。在改变答案的响应中，在生成理由之前的强制选择读出在辩论场景下比反思场景下更常与最终答案一致（在Qwen3.5-9B上为65.4%对22.3%）。消息消融实验揭示了同伴答案和推理对模型依赖性的影响，而将相同文本归因于接收者本人而非另一个智能体会显著改变该文本的影响力。

    arXiv:2610.35930v1 Announce Type: cross  Abstract: Multi-agent debate (MAD) aims to improve language-model decisions through the exchange of arguments, yet agreement after debate need not reflect independent corroboration. We investigate how peer communication influences individual decisions and collective outcomes through behavioral and internal interventions. We introduce MAD Bench, a 2,400-question benchmark spanning knowledge and reasoning, values and moral judgment, and safety and refusal, and evaluate three open-weight models using a round-matched independent-reflection control. Among answer-changing responses, a forced-choice readout before rationale generation matches the eventual answer more often in debate than in reflection (65.4% versus 22.3% on Qwen3.5-9B). Message ablations reveal model-dependent effects of peer answers and reasoning, while attributing identical text to the receiver rather than another agent substantially changes its influence. For the tested source descr
    
[^248]: SCOPE：信源可信度与媒体机构画像引擎

    SCOPE: Source Credibility and Outlet Profiling Engine

    [https://arxiv.org/abs/2610.35929](https://arxiv.org/abs/2610.35929)

    本文提出SCOPE多智能体系统，通过协调文章获取、内容分析、证据搜索等专门智能体，基于证据对新闻媒体的偏见、事实性和可信度进行透明评估，在200家媒体基准上超越了多种基线方法。

    

    我们提出了SCOPE，一个用于对新闻媒体机构在偏见、事实性和可信度方面进行画像的多智能体系统。给定一个媒体机构的URL，SCOPE会协调多个专门的智能体来完成文章获取、内容分析、外部证据搜索、评分和报告生成等任务。该系统遵循MBFC（媒体偏见事实核查）风格的方法论，同时通过结构化的演示界面使中间证据和智能体输出可供查验。在一个包含200家媒体机构的基准测试中，SCOPE在大多数指标上优于仅使用大语言模型、仅使用文章、仅使用搜索以及媒体背景核查等基线方法，实现了最低的偏见和事实性平均绝对误差（MAE）、完美的失败事实核查检测能力以及最低的事实错误率。这些结果表明，分解式的、基于证据的智能体工作流能够支持更透明、更可靠的媒体画像。我们的代码可在 https://github.com/mbaisbay/SCOPE 获取，在线演示见 https://scope-demo.streamli

    arXiv:2610.35929v1 Announce Type: cross  Abstract: We present SCOPE, a multi-agent system for profiling news media outlets with respect to bias, factuality, and credibility. Given an outlet URL, SCOPE coordinates specialized agents for article acquisition, content analysis, external evidence search, scoring, and report generation. The system follows an MBFC-style methodology while making intermediate evidence and agent outputs inspectable through a structured demonstration interface. On a benchmark of 200 media outlets, SCOPE outperforms LLM-only, article-only, search-only, and media-background-check baselines on most metrics, achieving the lowest bias and factuality MAE, perfect failed-fact-check detection, and the lowest factual error rate. These results suggest that decomposed, evidence-grounded agent workflows can support more transparent and reliable media profiling. Our code is available at https://github.com/mbaisbay/SCOPE, and a live demonstration at https://scope-demo.streamli
    
[^249]: ReCoPII：一种基于关系构建情境化PII脱敏数据集的框架

    ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets

    [https://arxiv.org/abs/2610.35925](https://arxiv.org/abs/2610.35925)

    ReCoPII框架通过规划本体约束的实体关系并据此推导MASK/KEEP标签，无需单一数据主体假设即可自动构建情境化PII脱敏数据集，所生成的韩文语料ReCoPII-Ko的KEEP标签获得88.42%的标注者认可率，远超SPY方法的26.27%。

    

    训练针对个人身份信息（PII）的情境化脱敏系统，需要能够区分同一实体类型中“可脱敏”与“应保留”提及的样本，但隐私限制使得真实数据难以获取。基于作者的增强方法（SPY）将与作者无关的提及标注为非PII，这一做法假定了单一数据主体，从而忽略了其他主体。我们提出了ReCoPII（基于关系的情境化PII脱敏），一个无需上述前提即可构建此类数据的框架：它首先规划受本体约束的实体关系，然后撰写带有占位符的插入句，再通过代码重建标注，用独立的LLM检查结果，并根据实体关系推导出MASK/KEEP标签。在来自六个领域的韩文文档上，该框架生成了ReCoPII-Ko语料库，其中包含关系、隐私和MASK/KEEP三类标注，我们对其进行了验证。在300对样本中，四名标注者对其抽样KEEP标签的认可率达88.42%，而SPY风格方法仅为26.27%。

    arXiv:2610.35925v1 Announce Type: cross  Abstract: Training contextual masking systems for personally identifiable information (PII) requires examples distinguishing maskable from preservable mentions of one entity type, but privacy restrictions limit access to real data. Author-based augmentation (SPY) labels mentions unrelated to the author as non-PII, presuming a single data subject and overlooking others. We propose ReCoPII (Relation-grounded Contextual PII masking), a framework that builds such data without this premise: it plans ontology-constrained entity relations, writes insertion sentences with placeholders, reconstructs annotations with code, checks the result with a separate LLM, and derives MASK/KEEP labels from entity relations. On Korean documents from six domains, it yields ReCoPII-Ko, a corpus with relation, privacy, and MASK/KEEP annotations, which we validate. Across 300 pairs, four annotators agree with 88.42% of its sampled KEEP labels, against 26.27% for SPY-style
    
[^250]: 面相学的重现：视觉语言模型如何通过图像变换编码敏感属性

    Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation

    [https://arxiv.org/abs/2610.35910](https://arxiv.org/abs/2610.35910)

    本文揭示了视觉语言模型在图像变换任务中会遵从按种族和性取向改变人脸的请求，生成统计上可区分且强化刻板印象的图像，并且这些表征偏差还会在下游文本推理任务中重现。

    

    新兴的生成式人工智能技术虽然在众多任务和领域中具有显著益处，但也可能存在无法预见的漏洞和难以预测的失败模式。这些失败模式可能对弱势群体造成不成比例的影响。本文研究了两个知名的视觉语言模型（VLM），考察这些模型在生成图像时如何编码受保护的身份特征。我们表明，当要求VLM将人脸图像转换为反映种族身份（如西班牙裔、黑人、白人或亚裔）和性取向（如同性恋或异性恋）时，VLM往往会遵从此类请求。我们证明，这种遵从（和拒绝）是非确定性的、对群体敏感的，且生成的图像在统计上是可区分的，往往会强化根深蒂固的刻板印象。我们进一步表明，图像生成过程中引入的表征偏差可能会在文本推理和人物刻画任务中再次出现。

    arXiv:2610.35910v1 Announce Type: cross  Abstract: Emergent generative AI technologies, while remarkably beneficial across several tasks and domains, can suffer from unforeseen vulnerabilities and difficult-to-anticipate failure modes. Such failure modes can disparately affect vulnerable communities. This paper investigates two well-known vision language models (VLMs) and inspects how these models encode protected identities while generating images. We show that when VLMs are requested to transform human face images to reflect race identities (e.g., Hispanic, Black, White, or Asian) and sexual orientation (e.g., gay or straight), VLMs often comply with such requests. We demonstrate that compliance (and refusal) is non-deterministic, group-sensitive, and generated images are statistically distinguishable, often accentuating deep-seated stereotypes. We further show that representation biases introduced during image generation can re-emerge in textual reasoning and characterization tasks.
    
[^251]: 任务效用图像质量评估：重新思考具身任务中的图像质量评估

    Task-Utility IQA: Rethinking Image Quality Assessment in Embodied Tasks

    [https://arxiv.org/abs/2610.35905](https://arxiv.org/abs/2610.35905)

    提出了一种基于多模态大语言模型的具身任务效用导向图像质量评估框架，将IQA从任务无关的分数回归范式转变为由具身任务效用定义的目标条件判断，且任务条件化效用分数比传统感知指标更能预测下游任务的成功。

    

    图像质量评估（IQA）是具身成像流程的基础，用于判断视觉质量是否满足下游任务需求，然而现有的大多数方法学习的是与静态基准上通用人类评分对齐的任务无关分数。这一目标与具身交互式场景不匹配，因为在这些场景中，图像的充分性取决于塑造智能体决策的任务目标、上下文和动作要求。我们认为，IQA应该从分数回归转向由具身任务效用定义的目标条件判断。为此，我们提出了一种基于多模态大语言模型（MLLM）的具身IQA智能体框架，该框架能够推理任务需求、定位质量证据，并产生面向决策的评估。一项紧凑的VisDrone证据研究表明，传统的感知指标和通用的基于MLLM的IQA分数只是下游VLM任务成功的微弱代理指标，而任务条件化的效用分数能更好地预测视觉输入是否支持任务。

    arXiv:2610.35905v1 Announce Type: cross  Abstract: Image quality assessment (IQA) underpins embodied imaging pipelines by judging whether visual quality satisfies downstream tasks, yet most existing methods learn task-agnostic scores aligned with generic human ratings on static benchmarks. This objective mismatches embodied and interactive settings, where image adequacy depends on task goals, context, and action requirements that shape an agent's decisions. We argue that IQA should shift from score regression to goal-conditioned judgment defined by embodied task utility. To this end, we propose an MLLM-based embodied IQA agent framework that reasons about task requirements, grounds quality evidence, and produces decision-facing assessments. A compact VisDrone evidence study shows that conventional perceptual metrics and generic MLLM-based IQA scores are weak proxies for downstream VLM task success, while a task-conditioned utility score better predicts whether visual input supports the
    
[^252]: 超越漂绿：主权资本流动中的算法韧性与非线性治理溢价

    Beyond Greenwashing: Algorithmic Resilience and the Non-Linear Governance Premium in Sovereign Capital Flows

    [https://arxiv.org/abs/2610.35895](https://arxiv.org/abs/2610.35895)

    本文利用梯度提升模型与双重可解释性框架，揭示主权风险传导本质上是非线性的，并提出“治理溢价”概念——ESG改善只有在突破关键治理阈值后才能为国家级资本流入带来优势。

    

    尽管主权环境、社会和治理（ESG）指标日益普及，传统的线性宏观金融模型在系统性冲击期间仍会系统性地错误定价外国直接投资（FDI）的尾部风险。本文证明主权风险传导在本质上是非线性的。基于2010年至2024年147个主权实体的前向滚动全球面板数据，我们部署了一种梯度提升架构，在经济极端尾部事件期间，相比L1正则化基线模型，将预测误差压缩了超过39%。超越单纯追求不透明的预测准确性，我们应用双重可解释性框架，从数学上刻画了制度质量的离散阈值效应。我们识别出一个严格的非线性“治理溢价”：在基础性监管阈值被突破之前，环境和社会层面的渐进式改善不会带来任何资本优势。

    arXiv:2610.35895v1 Announce Type: cross  Abstract: Despite the proliferation of sovereign Environmental, Social, and Governance (ESG) metrics, traditional linear macro-financial models systematically misprice Foreign Direct Investment (FDI) tail risks during systemic shocks. This paper demonstrates that sovereign risk transmission is fundamentally non-linear. Using a forward-chained global panel of 147 sovereign entities (2010-2024), we deploy a gradient boosting architecture that compresses forecast errors by over 39% against L1-regularized baselines precisely during extreme macroeconomic tail events. Moving beyond opaque predictive accuracy, we apply dual interpretability frameworks to mathematically map the discrete threshold effects of institutional quality. We identify a strict, non-linear "Governance Premium": incremental environmental and social improvements yield zero capital advantage until a foundational regulatory threshold is breached. Finally, we establish regime-condition
    
[^253]: scPyviewer：一个基于Python原生、面向AnnData单细胞数据的交互式查看器

    scPyviewer: a Python-native interactive viewer from AnnData single-cell data

    [https://arxiv.org/abs/2610.35890](https://arxiv.org/abs/2610.35890)

    scPyviewer是一个可直接读取AnnData对象的Python原生交互式单细胞数据网络查看器，在与三个现有R Shiny工具的功能对比中全面达到或超越基准能力，并独特提供免转换的原生数据读取和跨数据集基因表达比较功能。

    

    研究动机：让非程序员也能探索单细胞数据集、降维嵌入、基因表达、元数据和标记基因的交互式网络工具，已成为标准的实验室基础设施。然而，目前仍在积极维护的工具（如ShinyCell、ScRDAVis、sCIRCLE）均依赖R Shiny且需要Seurat对象。Python/scanpy作为单细胞RNA测序和空间分析的主流框架，却缺乏一种轻量级的原生解决方案，无法将可共享的查看器与可脚本化的Python API相结合。目前共享scanpy分析结果需要转换为Seurat格式或共享代码笔记本。研究结果：我们提出了scPyviewer，这是一个可直接读取AnnData对象的网络查看器，并在Python技术栈内复现了现有工具的交互模式。在与三个R Shiny工具的功能对等性审查中，scPyviewer达到或超越了全部基准能力（7/7），并独特地提供了无需转换的原生AnnData读取功能以及基于共享基因的跨数据集比较能力。

    arXiv:2610.35890v1 Announce Type: cross  Abstract: Motivation: Interactive web tools enabling non-programmers to explore single-cell datasets, embeddings, gene expression, metadata, and marker genes are standard laboratory infrastructure. However, actively maintained tools (e.g., ShinyCell, ScRDAVis, sCIRCLE) rely on R Shiny and require Seurat objects. Python/scanpy, the dominant framework for single-cell RNA-seq and spatial analysis, lacks a lightweight, native option pairing a shareable viewer with a scriptable Python API. Sharing scanpy results currently requires converting to Seurat or sharing code notebooks.   Results: We present scPyviewer, a web viewer that ingests AnnData objects directly and reproduces incumbent interaction patterns within the Python stack. In a feature-parity audit against three R Shiny tools, scPyviewer matches or exceeds baseline capabilities (7/7), uniquely offering native AnnData ingestion without conversion and cross-dataset comparison over shared genes 
    
[^254]: I-HITL：不可剥夺的人在回路

    I-HITL: Inalienable Human-in-the-Loop

    [https://arxiv.org/abs/2610.35835](https://arxiv.org/abs/2610.35835)

    该论文提出“不可剥夺的人在回路”（I-HITL）协议，确保即使智能体攻破了人类所交互的系统、甚至人类试图将批准权委托给智能体时，智能体的关键行动仍必须获得人类明确批准才能执行。

    

    人类监督是人工智能安全的支柱，通常由一个“人在回路”（HITL）中的人类来执行，由其批准AI智能体采取的关键行动。然而，AI智能体绕过安全防护栏的能力日益增强，这引出了一个如何稳健地强制执行HITL的问题。我们提出了“不可剥夺的人在回路”（I-HITL）的概念。I-HITL协议确保智能体只有在获得人类明确批准的情况下才能启动某项行动，即使人类所交互的系统（例如本地智能体工具框架）已被该智能体攻破也是如此。强I-HITL协议还额外确保，即使对抗性的人类试图将批准权委托给智能体时，行动仍需获得人类的明确批准。为论证I-HITL的必要性，我们报告了在一个流行的智能体钱包平台上的实验：在该平台中，提高智能体的消费上限需要通过电子邮件发送的一次性密码（OTP）进行HITL批准。在可以访问用户电子邮件且任务因消费上限而受阻的情况下，我们发现十个不同的前沿（模型）……

    arXiv:2610.35835v1 Announce Type: cross  Abstract: Human oversight is a pillar of AI safety. It is often exercised by a human in the loop (HITL) who approves critical actions taken by an AI agent. AI agents' increasing ability to bypass guardrails, though, raises the question of how to enforce HITL robustly.   We introduce the concept of inalienable human-in-the-loop (I-HITL). An I-HITL protocol ensures that an agent can initiate an action only with explicit human approval, even when the human interacts with a system (e.g., a local agentic harness) compromised by the agent. A strong I-HITL protocol additionally ensures such approval even when an adversarial human attempts to delegate approvals to an agent.   To motivate I-HITL, we report on experiments with a popular agentic wallet platform in which raising an agent's spending cap requires HITL approval via emailed one-time passcodes (OTPs). Given access to the user's email and a task blocked by the cap, we find that ten different fron
    
[^255]: 预训练懂得“做什么”，仿真教会“怎么做”：仿真训练的机器人智能体的组合泛化

    Pretraining Knows What, Simulation Teaches How: Compositional Generalization in a Sim-Trained Robot Agent

    [https://arxiv.org/abs/2610.35817](https://arxiv.org/abs/2610.35817)

    提出零真实机器人策略ZAP，仅借助预训练视觉语言模型加上简单仿真任务上的监督微调与GRPO训练，无需任何真实机器人数据，即可在真实世界任务上达到96.7%成功率，并对新物体、目标、策略和语言展现出组合泛化能力。

    

    机器人策略需要理解场景中任务所要求的是什么，以及如何移动机器人来完成该任务。为了实现这两项技能，我们提出了零真实智能体策略，这是一种通过简单的工具框架来教会小型现成视觉语言模型控制机器人的方法。我们使用一个预训练的270亿参数模型来构建世界理解与推理能力，随后在简单的仿真任务上进行监督微调和GRPO，以教会它如何移动机器人。在训练过程中不使用任何真实世界机器人数据的情况下，ZAP在真实世界任务上达到了96.7%的成功率，在我们的框架中与GPT-6 Astra相当，并优于MolmoAct2 VLA（33.3%），同时运行速度比GPT-6 Astra快3倍、成本低17倍。我们证明ZAP还能泛化到全新的物体、目标、策略和语言，例如将试管放入试管架（并在被提示时选择架子上的不同孔位）、递送试管……

    arXiv:2610.35817v1 Announce Type: cross  Abstract: A robot policy needs to understand what a task is asking for in a scene, and how to move the robot to accomplish that task. To achieve these two skills, we introduce Zero-real Agent Policy (ZAP), a recipe that teaches a small off-the-shelf vision-language model to control a robot via a simple tool harness. We use a pretrained 27B parameter model to bootstrap world understanding and reasoning, followed by supervised fine-tuning and GRPO on simple simulated tasks to teach it how to move the robot. Without any real-world robot data during training, ZAP reaches 96.7% success on real-world tasks, matching GPT-6 Astra in our harness and outperforming the MolmoAct2 VLA (33.3%), while running 3x faster and 17x cheaper than GPT-6 Astra. We demonstrate that ZAP also generalizes to completely new objects, goals, strategies, and language by seating a test tube in a rack (and choosing a different hole in the rack when prompted), handing a test tube
    
[^256]: 多层Transformer的定量泛化界：少数长字符串可以胜过大量短字符串

    Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones

    [https://arxiv.org/abs/2610.35715](https://arxiv.org/abs/2610.35715)

    本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。

    

    基于Transformer的语言模型展现出惊人的能力，能够泛化到比训练中所见更长、更难的问题实例。然而，对这一现象的理论理解仍然滞后。与大多数机器学习模型不同，Transformer的泛化能力本质上需要从两个维度进行考察。一方面，必须研究分布内统计泛化，即模型将从训练样本学到的函数拓展到来自同一分布的新样本的能力；另一方面，鉴于语言数据的序列特性，还必须考虑长度泛化，即在给定长度输入上训练的模型能否外推到更长的输入。这两个概念以往大多被分开研究，它们之间如何相互作用仍是一个悬而未决的问题。在这项工作中，我们对固定精度的Transformer给出了两个维度的联合分析。我们给出了一般……

    arXiv:2610.35715v1 Announce Type: cross  Abstract: Transformer-based language models exhibit an astonishing capacity to generalize to longer and more difficult problem instances than those seen in training. However, theoretical understanding of this phenomenon still lags behind. Unlike most machine learning models, the generalisation abilities of transformers inherently require considering two axes. On one hand, one must examine in-distribution statistical generalisation, which concerns the ability of a model to broaden a function from training samples to novel samples of the same distribution. On the other hand, given the sequential nature of language data, one must also consider length generalisation, which reflects whether a model trained on inputs of a given length can extrapolate to longer ones.These two notions have mostly been studied separately, leaving open how they interact. In this work, we give a joint treatment of both axes for fixed-precision transformers. We give general
    
[^257]: 智能体算法设计：能在Lean中提出、形式化并证明进化算法运行时间边界的LLM

    Agentic Algorithm Design: LLMs that Propose, Formalise, and Prove Runtime Bounds for Evolutionary Algorithms in Lean

    [https://arxiv.org/abs/2610.35595](https://arxiv.org/abs/2610.35595)

    该论文提出了一个模仿进化算法研究方法的分阶段流水线，使LLM智能体能够自主设计黑盒优化算法，并在Lean中通过势函数与漂移分析形式化证明其运行时间边界，全程无需经验性比较即可获得O(n²)的期望运行时间保证。

    

    LLM如今已能在Lean中证明困难的定理。那么它们能否也设计算法并证明其正确性与效率？由LLM引导的优化启发式设计（如AlphaEvolve）是纯经验性的；结合LLM的自动定理证明器则只能证明给定的命题。迄今为止，几乎没有证据表明LLM能够同时做到这两点：既设计出一个算法，又通过数学证明保证其效率。我们提出了一条模仿进化算法设计与分析研究方法的分阶段流水线。智能体提出黑盒优化算法，Lean守卫（guards）会过滤掉有缺陷的设计。随后智能体定义一个势函数并证明漂移边界，从而保证相比基线算法获得期望运行时间上的改进。在整个流水线中，算法从未进行任何经验性比较。该智能体成功生成了一个在CycleUncover问题上期望运行时间为O(n²)的黑盒优化算法，而经典算法……（原文此处截断）

    arXiv:2610.35595v1 Announce Type: new  Abstract: LLMs now prove hard theorems in Lean. Can they also design algorithms and prove their correctness and efficiency? LLM-guided design of optimisation heuristics (such as AlphaEvolve) is purely empirical. Automated theorem provers with LLMs prove given statements. So far, there has been little evidence that LLMs can do both: design an algorithm and guarantee its efficiency by a mathematical proof. We present a staged pipeline modelled after research methodology in design and analysis of evolutionary algorithms. The agent proposes black-box optimisation algorithms, and Lean guards filter out flawed designs. The agent then defines a potential function and proves drift bounds that guarantee an expected runtime improvement over the baseline algorithm. At no point in the pipeline are the algorithms compared empirically. The agent generates a black-box optimisation algorithm with expected runtime $O(n^2)$ on a CycleUncover problem, where the clas
    
[^258]: 多保真度训练重塑原子模型中的不确定性分解

    Multi-Fidelity Training Reshapes Uncertainty Decomposition in Atomistic Models

    [https://arxiv.org/abs/2610.35560](https://arxiv.org/abs/2610.35560)

    该研究利用均值-方差MACE模型深度集成发现，多保真度训练中低保真度预训练虽能降低预测误差并改善不确定性排序，但预测方差的校准退化主要发生在高保真度微调阶段，且在能量外推下更为严重。

    

    多保真度训练（即将丰富的低保真度数据与稀缺的高保真度标签相结合）在原子建模中日益普遍，但其对预测不确定性的影响在很大程度上仍未被探索。我们使用在ANI-1x数据集上训练的均值-方差MACE模型深度集成来研究这一问题，对比了从零开始的高保真度训练与低保真度预训练加微调两种策略。我们在整个训练过程中以及在两种分布偏移（未见过的分子系统和更高能量构型）下，对偶然不确定性、认知不确定性和总不确定性进行了表征。我们发现，预训练降低了预测误差，并在所有测试集上改善了基于不确定性的排序能力，但并不能产生良好校准的预测方差；在能量外推情形下，校准恶化更为严重。训练动力学分析将大部分校准退化定位于高保真度优化阶段，在此期间……

    arXiv:2610.35560v1 Announce Type: cross  Abstract: Multi-fidelity training, which combines abundant low-fidelity data with scarce high-fidelity labels, is increasingly common in atomistic modeling, but its effect on predictive uncertainty remains largely unexplored. We investigate this question using deep ensembles of mean-variance MACE models trained on ANI-1x, contrasting high-fidelity training from scratch with low-fidelity pretraining followed by fine-tuning. We characterize aleatoric, epistemic, and total uncertainty throughout training and under two distribution shifts: unseen molecular systems and higher-energy configurations. We find that pretraining reduces prediction error and improves uncertainty-based ranking across all test sets, but does not yield well-calibrated predictive variances; calibration deteriorates more severely under energy extrapolation. Training-dynamics analysis localizes most of this calibration degradation to high-fidelity optimization, during which the l
    
[^259]: 中间的智能体：AI中介访问如何重塑Web漏洞利用

    The Agent in the Middle: How AI-Mediated Access Reshapes Web Vulnerability Exploitation

    [https://arxiv.org/abs/2610.35541](https://arxiv.org/abs/2610.35541)

    提出可执行基准测试VELA，系统研究了AI智能体中介访问对Web漏洞利用的影响，发现面向智能体的接口设计决定了漏洞可利用性，参考设计下91%的漏洞仍然可被利用。

    

    Web服务越来越多地暴露后端接口，允许用户将任务委托给AI智能体，这些智能体无需浏览网站即可直接调用服务操作。这种转变改变了现有Web漏洞被利用的方式。我们通过VELA来研究这些变化，VELA是一个可执行的基准测试，涵盖十二个Web应用家族中的44个漏洞。该基准保留了存在漏洞的后端，并从合法工作流中推导出面向智能体的接口。我们在多种接口配置下评估了三个商业智能体，使用受控比较来区分接口效应与智能体行为，并使用独立的判定标准来验证攻击结果。我们的结果表明，在参考配置下，智能体能够实现直接接口调用所能实现的所有攻击，因此可利用性主要由接口决定：在参考设计下，91%的漏洞仍然可被利用。

    arXiv:2610.35541v1 Announce Type: cross  Abstract: Web services increasingly expose backend interfaces that let users delegate tasks to AI agents, which invoke service operations directly without navigating websites. This shift changes how existing Web vulnerabilities can be exploited. We investigate these changes through VELA, an executable benchmark covering 44 vulnerabilities across twelve Web application families. The benchmark preserves vulnerable backends and derives agent-facing interfaces from legitimate workflows. We evaluate three commercial agents across multiple interface configurations, using controlled comparisons to separate interface effects from agent behavior and independent oracles to verify attack outcomes. Our results show that the agent realizes every attack that direct interface invocation realizes in the reference configuration, so exploitability is set mainly by the interface: 91% of vulnerabilities remain exploitable under the reference design. Interface restr
    
[^260]: JANUS：针对Web反序列化攻击的跨层防御自动合成方法

    JANUS: Synthesizing Cross-Layer Defenses Against Web Deserialization Attacks

    [https://arxiv.org/abs/2610.35532](https://arxiv.org/abs/2610.35532)

    JANUS提出了一种红蓝对抗框架，通过黑盒攻击者迭代探测攻击变体与防御者进行反事实分析并细化策略，实现针对Web反序列化攻击的跨层防御自动化合成。

    

    不安全的反序列化会将不受信任的数据转化为攻击者可控的执行，而单个漏洞可能允许多种攻击变体，从而绕过针对已知攻击定制的防御。现有防御通常在单一执行点运行，对完整攻击的可见性有限。跨层组合防御控制可以提供互补保护，但协调这些控制以保留合法功能并适应新的攻击变体需要大量的专家投入。我们提出了JANUS，一个用于自动化跨层防御合成的红蓝对抗框架。从已知攻击出发，JANUS在隔离的离线环境中迭代地细化特定于部署的防御策略。黑盒攻击者仅使用外部可观察的反馈来探测当前策略下的攻击变体。防御者将每个已验证的变体总结为其安全相关的影响，并使用反事实重放来分离（摘要在此处截断）

    arXiv:2610.35532v1 Announce Type: cross  Abstract: Unsafe deserialization turns untrusted data into attacker-controlled execution, and a single vulnerability can admit diverse exploit variants that evade defenses tailored to a known exploit. Existing defenses typically operate at a single enforcement point with limited visibility into the full attack. Combining controls across layers offers complementary protection, but coordinating them to preserve legitimate functionality and adapting them to new variants requires substantial expert effort.   We propose JANUS, a red-blue adversarial framework for automated cross-layer defense synthesis. Starting from a known exploit, \sys iteratively refines deployment-specific policies in an isolated offline environment. A black-box attacker probes the current policy for exploit variants using only externally observable feedback. The defender summarizes each verified variant into its security-relevant effects, uses counterfactual replay to separate 
    
[^261]: Waypoint：面向CPU的GEMM解析建模

    Waypoint: Analytically Modeling GEMM for CPUs

    [https://arxiv.org/abs/2610.34914](https://arxiv.org/abs/2610.34914)

    提出了解析模型Waypoint，通过捕捉内存层次、指令级并行和多线程等架构效应，自动为CPU确定GEMM的最优分块大小，并据此构建了性能媲美最先进库且可加速LLM推理的GEMM库OPAM。

    

    高性能的通用矩阵乘法（GEMM）实现依赖于将GEMM分解为精心选择的分块以达到峰值性能。我们提出了Waypoint，这是一种新的解析模型，可自动为给定的硬件和输入矩阵形状确定最优分块大小。Waypoint通过捕捉所有架构效应——包括与内存层次结构的交互、指令级并行和多线程——来估计给定分块选择所能达到的性能。以这些估计作为代理，我们在可能的分块大小空间中进行搜索，以确定性能最大化的分块。此外，Waypoint还帮助硬件设计者评估实验性硬件优化的真实性能。我们构建了一个新的面向CPU的GEMM库OPAM，它使用Waypoint自动分解GEMM，实现了与最先进GEMM库相当的性能，并有助于加速大语言模型（LLM）推理。

    arXiv:2610.34914v1 Announce Type: cross  Abstract: High-performance General Matrix Multiplication (GEMM) implementations rely on breaking down GEMM into carefully selected tiles to achieve peak performance. We present Waypoint, a new analytical model to automatically identify the optimal tile size for a given hardware and input matrix shape. Waypoint estimates the achievable performance for a given tile choice by capturing all the architectural effects including interactions with memory hierarchy, instruction-level parallelism, and multi-threading. Using the estimates as a proxy, we search through the space of possible tile sizes to identify the one that maximizes performance. Additionally, Waypoint helps hardware designers estimate realistic performance on experimental hardware optimizations. We build a new GEMM library OPAM for CPUs that uses Waypoint to automatically break down GEMM and achieve performance that matches state-of-the-art GEMM libraries and helps accelerate LLM inferen
    
[^262]: 静默而代价高昂：LLM智能体中提示缓存错误集成的问题表征与检测

    Silent and Costly: Characterizing and Detecting Prompt-Caching Mis-Integration in LLM Agents

    [https://arxiv.org/abs/2610.34817](https://arxiv.org/abs/2610.34817)

    该论文首次对10个开源智能体系统中120个提示缓存相关的拉取请求进行实证研究，揭示了提示缓存错误集成往往静默失败且难以察觉的问题，并据此提出了检测工具CACHELINT。

    

    大规模语言模型服务需要处理海量的用户请求，从而产生巨额的推理成本。为了同时降低成本与延迟，服务提供商引入了提示缓存技术。然而，由于各提供商的API存在异构性且缓存语义持续演变，正确集成提示缓存并非易事。为了理解下游应用中提示缓存的错误集成问题，我们对来自10个流行开源智能体系统的120个与提示缓存相关的拉取请求开展了首次实证研究。研究结果表明，提示缓存错误集成既源于请求侧的错误配置，也源于读取侧的使用计数错误。更重要的是，大多数错误集成会静默失败，不会引发任何可观察到的运行时故障。基于这些发现，我们设计了CACHELINT，它将提示缓存错误集成建模为一个调用点以及正确缓存所需的先决条件

    arXiv:2610.34817v1 Announce Type: cross  Abstract: Large language model (LLM) services process massive volumes of user requests, resulting in substantial inference costs. To reduce both cost and latency, providers have introduced prompt caching. However, correctly integrating prompt caching is non-trivial due to heterogeneous provider-specific APIs and continuously evolving caching semantics. To understand prompt-caching mis-integration in downstream applications, we conduct the first empirical study of 120 prompt-caching-related pull requests from 10 popular open-source agent systems. Our findings show that prompt-caching mis-integrations arise in both request-side misconfiguration and read-side usage miscount. More importantly, most mis-integrations fail silently without causing observable runtime failures.   Motivated by these findings, we design CACHELINT, which models a prompt-caching mis-integration as an invocation site together with the preconditions required for correct cache 
    
[^263]: VLA-HWBench：VLA模型在边缘硬件上运行时的时间都花在哪里

    VLA-HWBench: Where the Time Goes When VLA Models Run on Edge Hardware

    [https://arxiv.org/abs/2610.34717](https://arxiv.org/abs/2610.34717)

    该论文提出了VLA-HWBench基准，通过七阶段推理分解在两种边缘硬件平台上对18个开源VLA模型进行测试，发现动作生成策略对推理延迟的影响远大于模型规模。

    

    视觉-语言-动作（VLA）模型越来越多地部署在机器人搭载的边缘设备上，推理延迟制约着闭环机器人控制。然而，如何设计VLA模型、其推理软件和边缘硬件，使机器人能够实时运行VLA推理，目前仍不明确。大多数VLA模型仅报告任务成功率，而已有研究涵盖的动作生成策略有限，最多将推理划分为三个阶段，且缺乏在相同模型上对推理优化效果的比较。本研究在NVIDIA Jetson AGX Thor和AMD Ryzen AI Max+ PRO 395两种平台上，对18个开源VLA模型在官方发布后端和替代后端上进行了基准测试，并采用七阶段推理分解方法。我们的研究揭示了三点洞见。第一，动作生成策略比模型规模更重要。它在Jetson上19种配置中的14种里占主导地位，延迟在Jetson上差异高达82倍，在AMD上高达157倍，而参数量最多只能解释15%的延迟方差。

    arXiv:2610.34717v1 Announce Type: cross  Abstract: Vision--language--action (VLA) models increasingly run on edge devices aboard robots, where inference latency constrains closed-loop robot control. However, it remains unclear how to design VLA models, their inference software and edge hardware so that robots can run VLA inference in real time. Most VLA models report only task success, while prior studies cover limited action-generation strategies, divide inference into at most three stages, and lack comparisons of inference optimizations on the same models.benchmarking 18 open VLA models across released and alternative backends on NVIDIA Jetson AGX Thor and AMD Ryzen AI Max+ PRO 395, with a seven-stage inference breakdown. Our study reveals three insights. First, action-generation strategy matters more than model size. It dominates 14 of 19 Jetson configurations, latency varies by $82\times$ on Jetson and $157\times$ on AMD, and parameter count explains at most 15% of latency variance
    
[^264]: SEEK：面向前沿科学智能体的基于演化知识的技能提取方法

    SEEK: Skill Extraction with Evolving Knowledge against Frontier Science Agents

    [https://arxiv.org/abs/2610.34365](https://arxiv.org/abs/2610.34365)

    本文提出了SEEK，一个黑盒自演化红队攻击智能体，通过在持久记忆中整合响应证据并演化可复用攻击技能，成功提取了科学智能体平台中受保护的技能，揭示了100%测试平台均存在技能泄露的严重安全风险。

    

    arXiv:2610.34365v1 公告类型：交叉  摘要：科学智能体日益将其领域专业知识封装为可复用的技能，用于规划分析、选择工具和执行工作流程。这种模块化设计带来了保密性风险：传递技能能力的交互过程也可能暴露技能本身。我们构建了一个可扩展的基准测试，涵盖八个开源与闭源科学智能体平台上的40个受保护技能，并在基于参考和无参考的评估下比较了直接探测、固定策略和老虎机引导的迭代探测方法。技能泄露现象普遍存在：全部八个平台（100%）至少泄露一个受保护技能，其中四个平台在单次直接请求下即达到100%的攻击成功率。为了对能够抵御简单探测的平台进行压力测试，我们提出了SEEK（基于演化知识的技能提取），这是一个黑盒、自演化的红队攻击智能体，它仅利用用户可见的响应，在持久记忆中整合从响应中获取的证据，并演化出可复用的攻击技能。在

    arXiv:2610.34365v1 Announce Type: cross  Abstract: Science agents increasingly package domain expertise as reusable skills that plan analyses, select tools, and execute workflows. This modularity creates a confidentiality risk: the interactions that deliver a skill's capability can also reveal the skill itself. We build an extensible benchmark of 40 protected skills across eight open- and closed-source science-agent platforms, comparing direct, fixed-strategy, and bandit-guided iterative probing under reference-based and reference-free evaluation. Leakage is pervasive: all eight platforms (100%) leak at least one protected skill, and four reach 100% attack success under a single direct request. To stress-test platforms that resist simple probing, we propose SEEK (Skill Extraction with Evolving Knowledge), a black-box, self-evolving red-teaming agent that consolidates response-derived evidence in persistent memory and evolves reusable attack skills, using only user-visible responses. On
    
[^265]: 面向近显式大气化学的主动训练神经算子

    An actively trained neural operator for near-explicit atmospheric chemistry

    [https://arxiv.org/abs/2610.33771](https://arxiv.org/abs/2610.33771)

    本文提出FNO-MoE——一种结合专家混合机制的傅里叶神经算子，可单次调用预测30分钟内的完整近显式大气化学状态，并利用专家间分歧进行主动采样自我训练，以远低于求解MCM的成本实现精确且稳定的多小时化学演化模拟。

    

    地球系统模型之所以将大气化学简化为凝聚机制，是因为像主化学机制（MCM）这样的近显式机制在每个网格单元中求解的代价过于高昂。神经模拟器有望突破这一限制，但现有的神经模拟器大多以小时间步长推进化学反应，且仅覆盖少数物种或简化的机制。本文提出了FNO-MoE，一种融合了专家混合的傅里叶神经算子，可以在单次调用中预测30分钟内完整的近显式化学状态。该网络还能指导自身的训练过程，利用其内部专家之间的分歧来选取信息量最大的箱式模型模拟。在相同计算成本下，这种主动采样策略比随机采样或均匀采样能获得更精确的模拟器。在更长的化学轨迹上进行微调后，模型在多小时运行中保持稳定，而仅用短时模拟训练的模型则会出现发散。由此得到的模拟器能够再现数千个物种的演化（原文此处截断）。

    arXiv:2610.33771v1 Announce Type: new  Abstract: Earth-system models simplify atmospheric chemistry into condensed mechanisms, because near-explicit mechanisms such as the Master Chemical Mechanism (MCM) are too costly to solve in every grid cell. Neural emulators could lift this limit, but most advance the chemistry in small time steps and cover only a few species or a reduced mechanism. Here we introduce FNO-MoE, a Fourier neural operator with a mixture of experts that predicts the full near-explicit chemical state over 30 minutes in a single call. The same network also guides its own training, using the disagreement among its experts to select the most informative box-model simulations. At equal cost, this active sampling yields a more accurate emulator than random or evenly spread sampling. Fine-tuning on longer trajectories keeps it stable over multi-hour runs in which models trained on short simulations diverge. The resulting emulator reproduces the evolution of thousands of spec
    
[^266]: 基于局部熵波动的黑盒大语言模型生成文本检测

    Local Entropy Fluctuations for Black-Box LLM-Generated Text Detection

    [https://arxiv.org/abs/2610.33767](https://arxiv.org/abs/2610.33767)

    该论文提出新指标“相对峰值显著率”，利用人类写作信息流均匀而机器生成文本存在尖锐局部熵波动这一结构性差异，解决了黑盒场景下现有检测器在指令条件文本上灾难性失效的问题。

    

    零样本机器生成文本检测对于缓解大语言模型（LLM）滥用风险至关重要。尽管最先进的双模型检测器和基于重采样的检测器在无提示续写文本上的表现优于单模型统计检测器，但在黑盒设置（即源模型未知）下，它们在指令条件生成的文本上会遭遇灾难性失效。我们分析了这一失效现象，并将其根源归结为基于重采样的方法在提示条件下表现不佳，以及双模型检测器跨模型差异的减弱。受近期研究启发现象的启发——人类写作保持稳定、均匀的信息流，而机器生成的文本则呈现出尖锐的局部熵波动——我们提出了一种新指标：相对峰值显著率，用以捕捉熵轨迹上的这种结构性对比，并进一步将其作为LLM生成文本检测的信号。大量实验……

    arXiv:2610.33767v1 Announce Type: new  Abstract: Zero-shot detection of machine-generated text is vital for mitigating the risks of large language models (LLMs) misuse. While the state-of-the-art dual-model detectors and resampling-based detectors outperform single-model statistical detectors on unprompted continuations, they suffer a catastrophic breakdown on instruction-conditioned text in the black-box setting, where the source model is unknown. We analyze this failure and trace it to poor prompt conditioning for resampling-based methods and diminished cross-model discrepancy for dual-model detectors. Motivated by recent studies that human writing maintains a steady, uniform information flow, whereas machine-generated text exhibits sharp, localized entropy fluctuations, we propose a new metric, the Relative Peak Prominence Rate (RelPPR) to capture this structural contrast along the entropy trajectory and further leverage it as a signal for LLM-generated text detection. Extensive eva
    
[^267]: 看见选项：基于可复用选项记忆的视觉候选类型化决策

    Seeing the Options: Typed Decisions over Visual Candidates with Reusable Option Memory

    [https://arxiv.org/abs/2610.33702](https://arxiv.org/abs/2610.33702)

    提出Iris框架，将支持顺序不变性与精确缓存复用的类型化决策扩展到视觉、文本或混合候选场景，并证明了该性质对任意逐候选编码器均成立。

    

    智能体中的类型化决策任务，例如UI元素选择、产品推荐和图表匹配，要求模型从一组离散候选中进行选择。尽管先前的工作表明，纯文本的System One模型通过分离的请求-候选编码实现了候选顺序不变性、可扩展性和缓存复用等四个关键属性，但将其扩展到候选为视觉内容的多模态设置仍未得到充分探索。我们提出了Iris，一个能够容纳图像、文本或混合候选同时保留核心决策模块的框架。我们证明了在Iris下，顺序不变性和精确缓存复用对任何逐候选编码器都成立，并推导出延迟降低的闭式表达式。我们基于Multimodal-Mind2Web构建了一个视觉候选基准（K=16、去重、顺序打乱、三个泛化划分）。独立编码（Iris）实现了78.4 / 72.1 / 74.8的准确率，且零……

    arXiv:2610.33702v1 Announce Type: new  Abstract: Typed decision-making tasks in agents, such as UI element selection, product recommendation, and chart matching, require models to choose from a set of discrete candidates. While prior work demonstrates that text-only System One models achieve four key properties including candidate-order invariance, scalability, and cache reuse via separate request-candidate encoding, extending them to multimodal settings where candidates are visual remains underexplored. We introduce Iris, a framework that accommodates image, text, or hybrid candidates while preserving the core decision module. We prove that order invariance and exact cache reuse hold for any per-candidate encoder under Iris, and derive the closed-form latency reduction. We construct a visual-candidate benchmark from Multimodal-Mind2Web (K=16, deduplicated, order-shuffled, three generalization splits). Independent encoding (Iris) achieves 78.4 / 72.1 / 74.8 percent accuracy with zero-f
    
[^268]: AOS-Bench：大语言模型智能体能管理动态工作负载吗？

    AOS-Bench: Can LLM Agents Manage Dynamic Workloads?

    [https://arxiv.org/abs/2610.33178](https://arxiv.org/abs/2610.33178)

    本文提出 AOS-Bench，首个将 LLM 智能体作为“工作负载管理器”评估的基准——任务随时间流式到达并需在共享 CPU 与内存限制下交付——揭示了孤立任务上的成功无法反映智能体在动态多任务场景下的真实能力（重负载下所有系统损失 13 至 23 分）。

    

    编程智能体正在从一次只处理一个请求，转变为作为持续在线的助手来接收用户交来的工作流：一次构建、一个数据处理作业和一个错误修复可能同时挂起在同一台机器上。这样的智能体必须决定何时接手新任务、如何委派和重叠执行任务，以及如何将运行控制在机器的 CPU 和内存限制之内。现有的基准测试无法衡量这一点，因为它们都是孤立地评估每个任务。我们提出 AOS-Bench，它将智能体系统作为一个工作负载管理器来评估：经过验证的任务随时间不断到达，必须在共享的 CPU 和内存限制下按时交付，而相同的任务逐个单独运行则作为每个系统各自的参照基准。在五组模型—运行框架组合上的实验表明，孤立任务上的成功并不能均匀地延续到流式场景中。在重负载下，所有系统的性能都损失了 13 到 23 分。在中等负载下，各系统在孤立任务成功率上相差 11 分，但在流式成功率上相差达 26 分，且两个系统……

    arXiv:2610.33178v1 Announce Type: new  Abstract: Coding agents are moving from handling one request at a time to serving as persistent assistants to which users hand a stream of work: a build, a data-processing job and a bug fix may all be pending on the same machine at once. Such an agent must decide when to take on new tasks, how to delegate and overlap them, and how to stay within the machine's CPU and memory. Existing benchmarks cannot measure this, because they evaluate each task in isolation. We introduce \benchname{}, which evaluates an agent system as a workload manager: verified tasks arrive over time and must be delivered under shared CPU and memory limits, and the same tasks run one at a time serve as a per-system reference.   Across five model--harness pairs, isolated task success does not carry over evenly. On heavy workloads every system loses 13 to 23 points. On medium workloads, systems span 11 points in isolated success but 26 points in stream success, and two systems 
    
[^269]: 评估视觉语言模型在面向视障人群的视觉定位任务上的表现

    Evaluating Visual Language Models on Visual Grounding for Visually Impaired People

    [https://arxiv.org/abs/2610.33078](https://arxiv.org/abs/2610.33078)

    本文提出VG-SHAP方法，通过扩展现有可解释性框架为黑盒视觉语言模型提供视觉定位解释，并在包含视障人士真实拍摄图像的VizWiz-VQA-Grounding数据集上评估开源模型的视觉定位能力。

    

    视觉问答（VQA）中的视觉定位（VG）是视觉语言模型（VLM）的一项基础能力，它使模型能够检测图像中与回答问题最相关的区域。这反映了模型的多模态理解能力，而这种能力在与视障人群（VIP）交互时至关重要。事实上，当视觉定位应用于辅助场景时，它不仅仅是一项技术特性，更是一种信任与安全层面的要求。在本工作中，我们研究了视觉语言模型面向视障人群的视觉定位能力。为此，我们使用包含视障人士拍摄的真实生活图像的VizWiz-VQA-Grounding数据集和挑战赛，评估了开源视觉语言模型在视觉定位上的表现。我们提出了VG-SHAP，该方法扩展现有的可解释性框架以提供视觉定位解释，从而能够对黑盒视觉语言模型进行评估。我们在VizWiz-VQA-Grounding挑战赛上的结果表明，尽管生成……（摘要在此处截断）

    arXiv:2610.33078v1 Announce Type: cross  Abstract: Visual grounding (VG) in visual question answering (VQA) is a fundamental capability of Vision-Language Models (VLMs), enabling the detection of the most relevant region in an image to answer the question. This reflects the model's capability of multimodal comprehension, which is crucial when interacting with visually impaired people (VIP). Indeed, visual grounding, when applied to assistive scenarios, is not merely a technical feature, but rather a trust and safety requirement. In this work, we study visual grounding in VLMs for visually impaired people. To this end, we assess open-source VLMs on visual grounding using the VizWiz-VQA-Grounding dataset and challenge, which contains real-life images taken by VIP. We propose VG-SHAP, which extends an existing explainability framework to provide visual grounding explanations, allowing the evaluation of black-box VLMs. Our results on the VizWiz-VQA-Grounding challenge show that despite gen
    
[^270]: 音频-语言模型中可控说话风格描述的基准测试

    A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models

    [https://arxiv.org/abs/2610.33036](https://arxiv.org/abs/2610.33036)

    提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。

    

    说话风格描述（SSC）旨在生成关于语音如何被表达的自然语言描述，捕捉诸如音色、韵律和表现力等副语言属性。许多下游应用，包括对话式AI智能体、可控语音生成和大规模音频标注，都需要针对特定维度的风格描述来刻画语音的特定方面（例如说话人特质、情感或表达风格），而不是单一的、不加区分的描述。然而，现有工作缺乏对可控说话风格描述的统一形式化定义。我们通过将SSC形式化为一个遵循指令的音频-语言任务来填补这一空白，其中明确的指令指定了需要描述的说话风格维度。我们推出了StyleInstructCaps，这是首个面向音频-语言模型中可控说话风格描述的标准化基准。StyleInstructCaps提供了一个数据集……

    arXiv:2610.33036v1 Announce Type: cross  Abstract: Speaking-style captioning (SSC) aims to generate natural language descriptions of how speech is delivered, capturing paralinguistic attributes such as vocal timbre, prosody, and expressivity. Many downstream applications, including conversational AI agents, controllable speech generation, and large-scale audio annotation, require dimension-specific style captions that describe targeted aspects of speech (e.g., speaker traits, emotion, or delivery style), rather than a single undifferentiated description. However, existing work lacks a unified formulation for controllable speaking-style captioning. We address this gap by formulating SSC as an instruction-following audio-language task, where explicit instructions specify the speaking-style dimensions to be described. We introduce StyleInstructCaps, the first standardized benchmark for controllable speaking-style captioning in audio-language models. StyleInstructCaps provides a dataset an
    
[^271]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^272]: 重新思考面向城市表示学习的多视图融合

    Rethinking Multi-View Fusion for Urban Representation Learning

    [https://arxiv.org/abs/2610.31850](https://arxiv.org/abs/2610.31850)

    本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。

    

    城市区域表示学习将异构的城市信号（如人流移动、兴趣点和土地利用模式）编码为统一的嵌入表示，服务于下游任务，包括签到预测、犯罪预测和服务需求估计。现有方法依赖于全局共享的融合与联合优化，隐式假设各模态重要性相同且多目标训练天然平衡。然而，城市数据本质上是异构的。我们的实证分析揭示了两个结构性问题：（i）模态效用在区域间差异显著，导致静态融合过度强调某些模态的信号；（ii）异构目标之间存在持续的梯度不平衡，造成表示偏差和训练不稳定。我们提出了 PRISM，一个面向自适应多视图学习的自调节框架。PRISM 采用自适应的稀疏 MoE 路由来学习区域特定的视图分配，……（摘要在此处被截断）

    arXiv:2610.31850v1 Announce Type: cross  Abstract: Urban region representation learning encodes heterogeneous urban signals, such as mobility flows, points of interest, and land-use patterns, into unified embeddings for downstream tasks, including check-in prediction, crime forecasting, and service demand estimation. Existing approaches rely on globally shared fusion and joint optimization, implicitly assuming uniform modality importance and naturally balanced multi-objective training. However, urban data is inherently heterogeneous. Our empirical analysis reveals two structural issues: (i) modality utility varies markedly across regions, causing static fusion to overemphasize signals from certain modalities; and (ii) heterogeneous objectives exhibit persistent gradient imbalance, causing representational bias and unstable training. We present PRISM, a self-adjusting framework for adaptive multi-view learning. PRISM adapts sparse MoE routing to learn region-specific view allocations, w
    
[^273]: PaperFormalizer：作者在环的研究论文Lean自动形式化

    PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean

    [https://arxiv.org/abs/2610.31810](https://arxiv.org/abs/2610.31810)

    提出PaperFormalizer——一个基于Lean的“作者在环”智能体流水线，能够发现并修复研究论文证明中的漏洞，并在作者参与关键决策下将论文转化为经Lean验证的形式化版本。

    

    即使经过仔细校对的论文，其证明或所陈述的假设中仍可能存在微小的漏洞。我们提出了PaperFormalizer，一个基于Lean的智能体（agentic）流水线，帮助作者发现并弥补自己论文中的这些漏洞，将每篇论文转化为经Lean验证的版本。从LaTeX源文件出发，PaperFormalizer将论文转录为命题的依赖图，以自然语言对每个陈述的忠实性、歧义性和正确性进行审查，然后结合Mathlib、可选的领域库以及作者的个人工具包，在Lean中对这些陈述进行形式化与证明。由于只有作者本人清楚某个陈述的本意，以及如何最好地修复某个结果中的形式化漏洞，因此每当一个陈述存在多种可能的解读时，流水线都会停下来询问作者，并以自然语言和TeX符号的形式将每个修复方案作为建议呈现给作者，由作者决定接受、拒绝或调整（原文摘要在此处被截断）。

    arXiv:2610.31810v1 Announce Type: new  Abstract: Even a carefully proofread paper can leave small gaps in its proofs or its stated assumptions. We introduce PaperFormalizer, a Lean-based agentic pipeline that helps authors find and close such gaps in their own papers and turn each paper into a Lean-verified version. Starting from the LaTeX source, PaperFormalizer transcribes the paper into a dependency graph of statements, audits each statement in natural language for faithfulness, ambiguity and correctness, and then formalizes and proves the statements in Lean against Mathlib, optional domain libraries and the author's personal toolkit. Only the author knows what a statement is meant to say and how a formalization gap in a result is best fixed. The pipeline therefore stops and asks the author whenever a statement admits more than one reading, presents every repair, in natural language and TeX symbols, as a proposal that the author can accept, decline or redirect, and leaves to the aut
    
[^274]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^275]: ARCANA：抽象与类比推理基准

    ARCANA: A Benchmark for Abstraction and Analogical Reasoning

    [https://arxiv.org/abs/2610.29271](https://arxiv.org/abs/2610.29271)

    提出了ARCANA基准，通过要求从现实世界物体或物理现象到抽象网格变换的远类比映射来测试抽象与类比推理能力，人类可以可靠地解决这些任务，但大语言模型的表现与人类存在明显差距。

    

    类比推理被广泛认为是人类智能的基本组成部分。然而，很少有基准测试专门用于测试远类比推理——即在表面上不相似、但在深层次关系或结构上相似的领域之间进行的推理。在本工作中，我们引入了一个名为ARCANA（抽象与类比推理语料库，Abstraction and Reasoning Corpus of ANAlogy）的类比推理基准。ARCANA的任务需要执行从现实世界物体或物理现象到抽象网格变换的远类比映射，这与著名的ARC-AGI基准不同，后者主要侧重于人类先天先验知识。我们通过人类实验验证了ARCANA，证明人类能够可靠地解决这些任务。相比之下，对一系列大语言模型的评估显示，其表现与人类存在明显的性能差距，揭示了当前模型仍然难以克服的系统性失败模式。

    arXiv:2610.29271v1 Announce Type: new  Abstract: Analogical reasoning is widely considered a fundamental component of human intelligence. Few benchmarks are designed to explicitly test far analogical reasoning - reasoning between domains that are superficially dissimilar yet share deep relational or structural similarities. In this work, we introduce an analogical reasoning benchmark called Abstraction and Reasoning Corpus of ANAlogy (ARCANA). The tasks of ARCANA require performing far analogical mappings from real-world objects or physical phenomena to abstract grid-based transformations, different from the well-known ARC-AGI benchmark which primarily emphasized innate human priors. We validate ARCANA through a human study, demonstrating that humans solve these tasks reliably. In contrast, evaluations across a range of large language models show a distinct performance gap compared to humans, revealing systematic failure modes that remain challenging for current models. To address this
    
[^276]: Fig：自动化多智能体行为诱发器

    Fig: Automated Multiagent Behavior Elicitor

    [https://arxiv.org/abs/2610.28891](https://arxiv.org/abs/2610.28891)

    该论文提出了Fig系统，能够根据自然语言行为描述自动生成或选择游戏环境并进行针对性干预，从而系统性地诱发和评估多智能体模型的特定社会行为（如合作、欺骗、背叛等）。

    

    许多针对多智能体系统的社会评估只研究行为在环境中自然涌现的表现，而非通过系统性地干预环境来主动诱发这些行为。我们提出了Fig（Finding Interactions in Games，在游戏中发现交互），这是一个能够对游戏环境执行针对性诱发干预的智能体系统。它接受自然语言的行为描述，将其分解为子行为，生成新的社交游戏或选择现有游戏，并对游戏的引擎、对手、指令和上下文执行单一或组合式干预。为了评估模型行为，它结合了内置的机械性检查和合成的评分标准判断。我们通过人工审查验证了干预措施、生成的游戏以及评估机制。在Fig为六种测试行为——合作、剥削、欺骗、隐蔽协调、背叛等——所生成的有效场景中，模型平均在45.2%的回合中尝试目标行为。

    arXiv:2610.28891v1 Announce Type: new  Abstract: Many social evaluations for multiagent systems study behaviors as they emerge from environments instead of systematically intervening in the environment to elicit such behaviors. We introduce Fig (Finding Interactions in Games), an agentic system that performs targeted elicitation interventions on game environments. It takes a natural language behavior description, decomposes it into sub-behaviors, generates new social games or selects existing ones, and executes single or combined interventions to the games' engine, opponents, instructions, and context. To evaluate model behavior, it combines built-in mechanical checks and synthetic rubric judgments. We validate the interventions, generated games, and evaluation mechanisms via human review. Models attempt the target behavior in 45.2% of episodes on average for valid scenarios generated by Fig for six test behaviors -- cooperation, exploitation, deception, covert coordination, betrayal, 
    
[^277]: 从代码生成器到工程助手：面向PLC编程的工厂感知智能体

    From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming

    [https://arxiv.org/abs/2610.28160](https://arxiv.org/abs/2610.28160)

    该论文提出了一个工厂感知的对话式PLC编程智能体，通过工厂特定上下文摄取、以对话为先的解释与调试交互模式，以及从聊天到部署的分级工具访问阶梯，将PLC编程从一次性代码生成升级为持续的工程辅助体验。

    

    可编程逻辑控制器（PLC）编程仍然是一项以IDE为中心的手动任务，工程师需要在文档、项目文件和代码编辑器之间频繁切换上下文。近期研究表明，检索增强的大语言模型能够以较高的编译通过率生成符合厂商规范的Structured Text代码，但这些系统将代码生成视为一次性流水线：工程师提交查询后获得代码，系统对特定工厂没有持久化的知识。我们提出了一个工厂感知的对话式编程助手，该系统在三个方向上扩展了这一范式：工厂特定上下文的摄取、以对话为先的解释与调试交互模式，以及从对话到部署的分级工具访问阶梯。该系统专为真实工厂基础设施设计，并确保工程师对每个会产生现实世界影响的步骤保持控制权。

    arXiv:2610.28160v1 Announce Type: cross  Abstract: Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates \cite{pred}, yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects.
    
[^278]: DPO超参数空间的全貌：测量什么、如何随规模变化、以及何时不应迁移

    The Shape of the DPO Hyperparameter Landscape: What to Measure, How It Moves with Scale, and When Not to Transfer It

    [https://arxiv.org/abs/2610.27895](https://arxiv.org/abs/2610.27895)

    通过密集超参数网格实验系统描绘了DPO超参数空间随模型规模和训练步数的演变规律，并识别了最适合超参数选择的基准与优化策略。

    

    直接偏好优化被广泛用于语言模型的后训练，但其性能在很大程度上取决于超参数的选择，例如学习率和参考策略正则化参数β。为了更好地理解DPO超参数空间，我们使用密集的超参数网格，在来自同一系列的不同模型规模上开展了大量实验。这使我们能够分析最优DPO超参数如何随模型大小和更新步数的演变而变化，其中中间检查点提供了一种低成本的信号，可用于加速超参数优化过程。此外，通过在静态和动态基准上评估我们的模型，我们确定了哪些基准能为超参数选择提供最强的信号。最后，我们研究了多种最先进的超参数优化策略在此设置下的表现。

    arXiv:2610.27895v1 Announce Type: cross  Abstract: Direct preference optimisation (DPO) is widely used for language model post-training, but its performance depends strongly on the choice of its hyperparameters, such as the learning rate and the reference-policy regularisation parameter $\beta$. To better understand the DPO hyperparameter landscape, we conduct extensive experiments using a dense hyperparameter grid across different model scales from the same family. This allows us to analyse how the optimal DPO hyperparameters evolve with model size and the number of update steps, with intermediate checkpoints providing an inexpensive signal to accelerate the hyperparameter optimisation process. Furthermore, by evaluating our models on both static and dynamic benchmarks, we identify which benchmarks provide the strongest signal for hyperparameter selection. Finally, we investigate how different state-of-the-art hyperparameter optimisation strategies perform in this setting. Our experim
    
[^279]: ASTROLABE：基于物理流匹配的宇宙学模拟三维粒子级代理模型

    ASTROLABE: A 3D Particle-Level Surrogate for Cosmology Simulation with Physical Flow Matching

    [https://arxiv.org/abs/2610.27068](https://arxiv.org/abs/2610.27068)

    Astrolabe是一个基于流匹配的生成式模拟器，直接在N体粒子的相空间坐标上进行建模，突破了传统机器学习模拟器在分辨率上限和可模拟观测量类型上的双重限制。

    

    暗物质（DM）的微观物理特性在跨越巨大动态范围的宇宙学结构上留下了印记。非线性宇宙学观测量最可靠的建模方式是N体模拟和流体动力学模拟。然而，对暗物质理论空间进行系统性的N体扫描成本高得令人望而却步。机器学习模拟器提供了一种可能的解决方案，但目前要么受限于低分辨率，要么只能模拟特定的观测量。我们提出了Astrolabe，一个基于流匹配的生成式模拟器，旨在摆脱这两种限制。Astrolabe直接作用于N个单独粒子的6N相空间坐标。由于输出中没有体素网格，因此不存在体素分辨率上限，原则上任何观测量都可以从推理结果中测量得到。Astrolabe通过一种场-粒子混合Transformer架构实现了计算上的可行性，其中每个粒子都对多尺度场token进行交叉注意力计算，因此计算与内存开销得以控制。

    arXiv:2610.27068v1 Announce Type: cross  Abstract: The microphysics of dark matter (DM) is imprinted on cosmological structure across an enormous dynamic range. Nonlinear cosmological observables are most reliably modeled with N-body and hydrodynamic simulations. However, systematic N-body scans over DM theory space are prohibitively expensive. Machine learning emulators provide a possible solution, but are currently limited by either low resolution or only the ability to emulate specific observables. We introduce Astrolabe, a flow-matching-based generative emulator designed to be free of both constraints. Astrolabe acts directly on the 6N phase-space coordinates of N individual particles. With no voxel grid in the output, there is no voxel-resolution ceiling, and in principle any observable can be measured from inference. Astrolabe is computationally tractable via a field-particle hybrid transformer in which every particle cross-attends to multiscale field tokens, so compute and memor
    
[^280]: 迈向人工智能安全中的人类与AI互补性

    Towards Human-AI Complementarity for AI Safety

    [https://arxiv.org/abs/2610.24969](https://arxiv.org/abs/2610.24969)

    该论文构建了迄今最大的AI安全监督任务套件（3,957个实例）及人类-AI混合评估平台，发现尽管人类与AI之间存在显著的互补潜力，但现有辅助方法反而削弱了人类纠正AI错误的能力。

    

    对于负责监督AI的人类来说，安全地监管AI变得越来越具有挑战性，因为随着每一代新模型的出现，他们必须评估更长、更复杂的输出。为了缓解这一挑战，我们可以利用互补性：当人类和模型犯下不同的错误时，将二者的判断相结合可以产生比任何一方单独产生的更强的监督信号。我们在现实的、与安全相关的环境中研究了这一机会，贡献了已知规模最大的此类任务套件（涵盖8个数据集和6个领域的3,957个实例），以及一个用于测量人类、AI和人类-AI组合在匹配项目上表现的平台和数据管道。我们从416名人类评估者那里收集了35,613个判断，发现了实现互补效应的相当大的机会。我们还发现，在我们测试的辅助方法下，人类评估者纠正AI错误的频率反而降低了，即使人类的整体准确率有所上升，而我们测试的混合方法……（摘要截断）

    arXiv:2610.24969v2 Announce Type: replace  Abstract: Safely overseeing AI is increasingly challenging for the humans doing it, as they must evaluate longer and more complex outputs with every new model generation. To mitigate this challenge, we could leverage complementarity: where humans and models make different mistakes, combining their judgements could yield an oversight signal stronger than either produces alone. We study this opportunity in realistic, safety-relevant settings, contributing the largest known suite of such tasks (3,957 instances across 8 datasets and 6 domains) as well as a platform and data pipeline for measuring human, AI, and human-AI performance on matched items. We collected 35,613 judgements from 416 human raters, finding a considerable opportunity for a complementarity effect. We also found that with the assistance methods we tested, human raters corrected the AI's errors less often, even when overall human accuracy rose, and the hybridisation methods we tes
    
[^281]: 使用手机神经处理单元（NPU）进行一阶LoRA训练

    First-Order LoRA Training with Phone NPUs

    [https://arxiv.org/abs/2610.24816](https://arxiv.org/abs/2610.24816)

    该论文提出了一种让手机NPU承担LoRA一阶训练中全部冻结权重矩阵乘法（包括推理接口不支持的输入梯度计算）的系统，通过复用常驻权重分块和前向反量化器，实现了大语言模型在端侧基于私有数据的高效微调。

    

    端侧微调使大语言模型（LLM）能够从用户的私有数据中学习，而无需上传这些数据。低秩适应（LoRA）减少了可训练参数的数量，但每个一阶训练步骤仍需在前向传播、激活值重计算和输入梯度计算过程中与冻结的基础模型权重相乘。手机的神经处理单元（NPU）在推理时已经保存了这些权重；将其用于训练还需要计算输入梯度，而其推理接口并不支持这一操作。我们提出了一个系统，在NPU上运行全部三项冻结权重乘法运算，同时将训练图、适配器和优化器保留在CPU上。该系统的输入梯度内核复用了常驻的权重分块和前向反量化器，避免了生成转置权重副本，并确保前向和反向传播使用相同的有效权重。为保留可用的梯度，系统根据梯度敏感性选择计算精度……

    arXiv:2610.24816v2 Announce Type: replace-cross  Abstract: On-device fine-tuning lets a large language model (LLM) learn from its user's private data without uploading it. Low-rank adaptation (LoRA) reduces the trainable parameters, but each first-order training step still multiplies by the frozen base-model weights during the forward pass, activation recomputation, and input-gradient computation. A phone's neural processing unit (NPU) already holds these weights for inference; using it for training also requires computing input gradients, which its inference interface does not support. We present a system that runs all three frozen-weight products on the NPU while keeping the training graph, adapters, and optimizer on the CPU. Its input-gradient kernels reuse the resident weight tiles and the forward dequantizer, avoiding a transposed weight copy and ensuring forward and backward use the same effective weights. To preserve usable gradients, the system selects precision by gradient sen
    
[^282]: Annotrace：一个用于智能体轨迹步骤级标注的工作台

    Annotrace: A Workbench for Step-Level Annotation of Agent Traces

    [https://arxiv.org/abs/2610.23284](https://arxiv.org/abs/2610.23284)

    Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。

    

    每一项研究LLM智能体为何失败的工作，都始于对手工标注轨迹，而每一项这样的研究都会构建自己的一次性工具。可观测性平台能够记录智能体轨迹，却不提供结构化标注；标注工具可以标注文本，却没有“步骤”、任务“委派”或“工具调用”的概念，导致最小可标注的执行单元不明确。Annotrace 可以读取六种轨迹格式，包括 OpenTelemetry GenAI、LangSmith 和 Langfuse，并将每次运行转换为有序的、嵌套的步骤列表。标注者编写自己的标注手册；一个前沿模型会根据标注者自己轨迹的样本提出初始标注集，先行作答，并被问到与人类完全相同的问题，从而使标注成为一个人机协作的过程。Annotrace 能够衡量任意两个标注者（无论是人类还是AI）之间的一致性。在一个多智能体问答语料库上，它精确复现了此前一次人工标注工作的235个标注单元，并且两个模型……（摘要在此处被截断）

    arXiv:2610.23284v1 Announce Type: new  Abstract: Every study of why LLM agents fail begins by labelling traces by hand, and every such study builds its own throwaway tool. Observability platforms record agent traces but offer no structured labelling; annotation tools label text but have no notion of a step, a delegation or a tool call, leaving it unclear what the smallest unit of execution to label should be. Annotrace reads six trace formats, including OpenTelemetry GenAI, LangSmith and Langfuse, and turns each run into an ordered, nested list of steps. The annotator writes their own labelling codebook; a frontier model proposes a starting set from a sample of their own traces, answers first, and is asked exactly the questions the human is asked, making annotation a joint human-AI process. Annotrace measures agreement between any two annotators, human or AI. On a multi-agent question-answering corpus it reproduces a previous manual effort's 235 annotation units exactly, and two models
    
[^283]: Memorilla：面向大语言模型的潜在语义记忆

    Memorilla: Latent Semantic Memory for LLMs

    [https://arxiv.org/abs/2610.21375](https://arxiv.org/abs/2610.21375)

    Memorilla提出了一种潜在语义记忆系统，能将整个文档集合的信息压缩为仅16个依赖查询的记忆token并注入冻结LLM的输入嵌入空间，在八个基准中的五个上超越了Mem0和MemoRAG等基线方法。

    

    记忆使大语言模型（LLM）能够利用其参数之外的信息，例如过去的用户对话、长文档或智能体此前的行为。对于一个冻结（参数不更新）的LLM而言，获取这些积累知识的途径仅限于记忆系统放入其上下文窗口中的内容。然而，如何在保持模型上下文简短的同时，为每次查询收集并组合相关信息，对现有系统来说仍是一个挑战。本文提出了Memorilla，这是一种记忆系统，它将整个文档集合中的信息汇聚为固定数量的、依赖查询的“记忆token”，供冻结的LLM使用。Memorilla将文档（如用户聊天记录、书籍章节）一次性编码为可复用的嵌入向量，然后学习检索并将查询特定的信息放置到LLM的输入嵌入空间中。仅使用16个记忆token，Memorilla在与Mem0和MemoRAG等基线方法的对比中，在八个基准中的五个上取得了最佳性能。

    arXiv:2610.21375v1 Announce Type: cross  Abstract: Memory enables large language models (LLMs) to use information beyond their parameters, such as past user conversations, long documents, or an agent's earlier actions. For a frozen LLM, access to this accumulated knowledge is limited to what a memory system places in its context window. However, gathering and combining relevant information needed for each query while keeping the model's context short remains challenging for existing systems. Here we introduce Memorilla, a memory system that gathers information from an entire document collection into a fixed number of query-dependent 'memory tokens' for a frozen LLM. Memorilla encodes documents (such as user chats, book chapters) once into reusable embeddings, then learns to retrieve and place query-specific information in the LLM's input embedding space. Using only 16 memory tokens, Memorilla achieves the best performance against baselines including Mem0 and MemoRAG on five of eight be
    
[^284]: Agaram 2.5B：首个从零训练、支持工具调用的泰米尔语优先语言模型——面向低资源语言的实践指南

    Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages

    [https://arxiv.org/abs/2610.18420](https://arxiv.org/abs/2610.18420)

    该研究从零训练了首个泰米尔语优先、支持工具调用的25亿参数语言模型Agaram 2.5B，仅用8张H100 GPU、6天时间和约500亿token，就在翻译和泰米尔语指令遵循上超越了更大的印度语系模型Sarvam-1，为低资源语言的模型训练提供了可复制的实践方案。

    

    我们从零开始训练了Agaram 2.5B，一个25亿参数的泰米尔语-英语语言模型，使用约500亿个token（58%为泰米尔语，其余为英语、数学、代码和平行文本），在单节点8张H100 GPU上耗时约6天完成，并采用了一个与我们第一代模型不同的新分词器，该分词器能够保留换行符。通过对134,428段对话进行监督微调（其回复由一个开放权重的教师模型生成），再加上三轮简短的针对性训练，我们将其打造为一个聊天模型，能够以泰米尔文字、罗马化泰米尔语和英语回答问题，并具备工具调用能力：在25个工具提示（其模式已出现在训练数据中）上，它有23次选择了正确的工具，其中13次给出了完全符合预期的参数。发布的模型在FLORES和IN22的全部六个翻译方向上都优于Sarvam-1（一个20亿参数的印度语系基础模型）（FLORES chrF：译入泰米尔语54.3，译入英语55.2，而Sarvam-1分别为49.2和44.2），并且在泰米尔语原生的IndicIFEval基准上比其基础模型几乎翻倍（19.3%对10……（摘要在此处被截断）

    arXiv:2610.18420v1 Announce Type: cross  Abstract: We trained Agaram 2.5B, a 2.5-billion-parameter Tamil-English language model, from scratch on about 50 billion tokens (58% Tamil, the rest English, math, code and parallel text) on one node of eight H100 GPUs in about six days, with a new tokenizer that, unlike our first model's, keeps line breaks. Supervised fine-tuning on 134,428 conversations, whose replies an open-weight teacher model wrote, and three short targeted rounds turned it into a chat model that answers in Tamil script, romanized Tamil (Tanglish) and English and can call tools: on 25 tool prompts whose schemas appear in training it picks the right tool 23 times and gives exactly the expected arguments 13 times. The released model translates better than Sarvam-1, a 2B Indic base model, in all six FLORES and IN22 directions (FLORES chrF 54.3 into Tamil and 55.2 into English, against 49.2 and 44.2), and nearly doubles its base model on native-Tamil IndicIFEval (19.3% from 10
    
[^285]: 小错误在哪里至关重要：量化模仿学习中的状态临界性

    Where Small Mistakes Matter: Quantifying State Criticality for Imitation Learning

    [https://arxiv.org/abs/2610.17709](https://arxiv.org/abs/2610.17709)

    本文提出SCRIM方法，通过训练对抗性扰动策略和价值评论家来量化模仿学习中各状态的关键程度，并利用所得分数重新加权行为克隆损失，使模型重点关注那些小错误即会导致失败的关键状态。

    

    模仿学习通常将每个示范状态视为同等重要，在整个数据集上均匀地优化行为克隆目标。然而，并非所有状态都是平等的：在少数关键状态中，即使与专家动作存在轻微偏差也可能导致失败，而在许多其他状态中，这类错误是可恢复且无关紧要的。我们将模仿学习中的状态临界性形式化为：在给定状态下，对专家动作施加有界扰动所导致的下游价值的最大退化。随后，我们提出SCRIM，一种实用的状态临界性度量，通过围绕固定的模仿策略训练对抗性扰动策略和价值评论家来估计这一量。SCRIM为每个示范状态分配一个分数，反映任务结果对微小的、与策略相关的动作错误的脆弱程度。我们证明这些分数可用于对行为克隆损失进行重新加权……

    arXiv:2610.17709v1 Announce Type: cross  Abstract: Imitation learning typically treats every demonstration state as equally important, optimizing a behavior-cloning objective uniformly across the dataset. Yet not all states are created equal: in a small number of critical states, even a slight deviation from the expert action can lead to a failure, while in many other states such errors are recoverable and inconsequential. We formalize state criticality in imitation learning as the worst-case degradation in downstream value induced by a bounded perturbation of the expert action at a given state. We then propose SCRIM, a practical State CRIticality Measure that estimates this quantity by training an adversarial disturbance policy and a value critic around a fixed imitation policy. SCRIM assigns each demonstration state a score reflecting how vulnerable the task outcome is to small, policy-relevant action errors. We show that these scores can be used to reweight behavior-cloning losses, 
    
[^286]: 比较生成式AI与机器学习和人工审查在电子证据开示中的应用

    Comparing GenAI to Machine Learning and Manual Review for eDiscovery

    [https://arxiv.org/abs/2610.16925](https://arxiv.org/abs/2610.16925)

    该研究指出此前声称生成式AI审查工具（88%召回率）优于机器学习工具（64%召回率）的结论因方法差异而不可比，实际分析发现生成式AI与人工审查效果相当，而采用持续主动学习（CAL）的机器学习模型仅需2,000条样本训练即可达到最先进效果并具备更高效率。

    

    Keeling等人[Keeling2026Generative]报告了Redgrave LLP最近开展的一项案例研究，他们声称Relativity的生成式AI审查工具获得了88%的召回率，而其机器学习工具Relativity Assisted Learning (RAL)仅达到64%的召回率。然而这些结果并不具有可比性，原因在于所采用的方法目标不同、评估者之间存在分歧，以及它们代表了不同的召回率-精确度权衡和成本-效益权衡。我们对Redgrave数据的分析表明，在考虑这些差异之后，aiR与人工审查的效果是相当的。此外，采用持续主动学习（CAL）的成熟机器学习模型，仅在2,000条初审人员相关性判定样本上进行训练，就能展现出最先进的有效性和更优越的效率。

    arXiv:2610.16925v1 Announce Type: cross  Abstract: Keeling et al. [Keeling2026Generative] report on a case study recently conducted by Redgrave LLP in which they assert that Relativity's Generative AI Review tool (aiR) achieved 88% recall while Relativity Assisted Learning (RAL), its machine-learning tool, achieved 64% recall. These results are not comparable because the methods employed have different objectives, because of inter-assessor disagreement, and because they represent different recall-precision and cost-benefit trade-offs. Our analysis of the Redgrave data indicates that, when these differences are taken into account, aiR and manual review are comparable. Furthermore, established machine-learning models employing Continuous Active Learning (CAL) show state-of-the-art effectiveness, and superior efficiency, when trained only on a sample of 2,000 first-pass-reviewer responsiveness determinations.
    
[^287]: 人机反馈循环中的溯源感知信念更新

    Provenance-Aware Belief Updating in Human-AI Feedback Loops

    [https://arxiv.org/abs/2610.13008](https://arxiv.org/abs/2610.13008)

    该论文提出LQET训练目标，使语言模型能够依据证据来源而非报告数量进行信念更新，显著降低了重复转发信息造成的虚假佐证与置信度漂移问题。

    

    在重复的人机交互中，单一观察结果可能被引用、总结，并以多条报告的形式返回给模型，使得重复信息看起来像是独立的佐证。我们研究语言模型能否利用所提供的报告谱系，依据证据的来源而非报告数量来进行更新。我们提出了潜在商证据训练，这是一种关系型训练目标，将稀疏贝叶斯监督与约束条件相结合，促使模型在报告未带来新证据时保持预测稳定，并在不同转发历史中对相同的新证据做出一致的更新。在一个受控的二分类基准测试中，在 Qwen3-4B 和 Phi-4-mini 两种模型上（每种骨干模型均采用三组配对训练随机种子），与仅使用任务训练相比，LQET 将新证据更新的均方根误差（RMSE）降低了 41-44%，并将八次转发步骤后的对数几率漂移降低了 89-91%。这些收益还延伸到一个闭环模拟中，其中模型置信度控制着多少转发报告重新进入系统。

    arXiv:2610.13008v1 Announce Type: new  Abstract: In repeated human-AI interaction, a single observation can be quoted, summarized, and returned to a model as multiple reports, making repetition look like independent corroboration. We study whether language models can use supplied report lineage to update according to evidence provenance rather than report count. We introduce Latent Quotient Evidence Training (LQET), a relational objective that combines sparse Bayesian supervision with constraints that encourage stable predictions when reports add no new evidence and consistent updates to the same new evidence across relay histories. On a controlled binary benchmark, LQET reduces new-evidence update RMSE by 41-44% and log-odds drift after eight relay steps by 89-91% relative to task-only training across Qwen3-4B and Phi-4-mini, with three paired training seeds per backbone. The gains extend to a closed-loop simulation in which model confidence controls how many relayed reports re-enter 
    
[^288]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^289]: 被破坏却依然正确：视觉语言模型为何在内部“欺骗”自己

    Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally

    [https://arxiv.org/abs/2610.03445](https://arxiv.org/abs/2610.03445)

    该论文发现并定义了视觉语言模型中的“训练/推理差距”——对抗扰动虽能把教师强制训练损失压至近零，模型自由生成时却仍输出正确描述——并通过 logit lens 将该差距精确归因于单一自回归步骤中目标词元排名恰好固定为第3位的内部机制。

    

    一种针对性的对抗扰动可以将视觉语言模型（VLM）在固定目标描述上的教师强制（teacher-forced）训练损失降至接近零，然而当允许同一模型自由生成时，它输出的却是原始的正确描述，完全没有目标内容的痕迹。我们将这种分离现象称为“训练/推理差距”，并在 Qwen2.5-VL-7B-Instruct 模型上，通过对200张留出的 COCO 图像实施受控的两阶段 PGD 攻击，为该差距给出了精确的机制解释。首先，我们表明图像级别的像素统计特征——包括从 CNN 鲁棒性文献中正确重新实现的基于纹理的可攻击性度量——对于预测哪些图像会被破坏几乎没有预测能力（最佳预测器 r=-0.050, p=0.484；岭回归 R²=0.069）。其次，利用 logit lens 技术，我们将该差距定位到单个自回归步骤：在已生成正确首词的条件下，目标词元的排名恰好固定为第3位。

    arXiv:2610.03445v1 Announce Type: cross  Abstract: A targeted adversarial perturbation can drive a vision-language model's (VLM's) teacher-forced training loss for a fixed target caption to near zero, yet the same model, allowed to generate freely, produces the original, correct description with no trace of the target. We call this dissociation the train/inference gap, and give it a precise mechanistic account on Qwen2.5-VL-7B-Instruct using a controlled two-stage PGD attack on 200 held-out COCO images. First, we show that image-level pixel statistics, including a correctly re-implemented, texture-based attackability measure from the CNN robustness literature, have essentially no predictive power over which images are corrupted (best predictor r=-0.050, p=0.484; ridge regression R^2=0.069). Second, using the logit lens, we localise the gap to a single autoregressive step: the rank of the target token, conditioned on the correct first token already being generated, is fixed at exactly 3
    
[^290]: 基于模糊认知图的可解释合成医学表格数据生成方法用于临床决策支持

    Interpretable Synthetic Medical Tabular Data Generation for Clinical Decision Support Using Fuzzy Cognitive Maps

    [https://arxiv.org/abs/2610.00391](https://arxiv.org/abs/2610.00391)

    本文提出了一种基于模糊认知图（FCM）的新颖框架，利用语言学可解释的模糊集和显式因果关系编码来生成具有隐私保护能力的合成医学表格数据，从而支持临床决策系统在数据受限情况下的开发与验证。

    

    当真实临床数据因隐私、伦理或数据可用性限制而无法获取时，合成医学表格数据生成对于开发和验证基于计算机的医学系统（CBMS）变得至关重要。现有的概率模型和深度生成模型通常缺乏可解释性，并且无法保留具有临床意义的依赖关系，限制了其在安全关键应用中的适用性。本文提出了一种将模糊认知图（FCM）新颖应用于具有显式因果关系和隐私保护的合成医学表格数据生成框架。临床特征使用语言学上可解释的模糊集进行描述，特征间的依赖关系被编码为基于模糊集交集计算得到的FCM边权重。合成患者记录通过将随机初始化的语言激活向量在FCM中传播直至收敛来生成，随后进行去模糊化处理。

    arXiv:2610.00391v1 Announce Type: cross  Abstract: Synthetic medical tabular data generation has become essential for developing and validating computer-based medical systems (CBMSs) when real clinical data is restricted due to privacy, ethical, or data availability limitations. Existing probabilistic and deep generative models often lack interpretability and fail to preserve clinically meaningful dependencies, limiting their suitability for safety-critical applications. This paper proposes a novel application of Fuzzy Cognitive Maps (FCMs) in a framework for synthetic medical tabular data generation with explicit causality and privacy preservation. Clinical features are described using linguistically interpretable fuzzy sets, and inter-feature dependencies are encoded as FCM edge weights computed from fuzzy set intersections. Synthetic patient records are generated by propagating randomly initialized linguistic activation vectors through the FCM until convergence, followed by defuzzif
    
[^291]: 面向多路径大语言模型推理的分集合并方法

    Diversity Combining for Multi-Path LLM Reasoning

    [https://arxiv.org/abs/2609.38829](https://arxiv.org/abs/2609.38829)

    该论文将多路径LLM推理形式化为无线通信中的分集合并问题，证明了在可交换性条件下多数投票是最优的均匀线性合并策略，并提出通过提示模板多样性降低路径间相关性，从而突破SC方法随采样数增加而收益饱和的瓶颈。

    

    自一致性（Self-Consistency, SC）等多路径推理方法通过采样K条推理路径并选择出现频率最高的答案。然而，随着K的增大，其收益迅速趋于饱和平台，且现有方法无法预测这种饱和何时出现。我们将多路径大语言模型推理形式化为无线通信中的分集合并问题：每条路径相当于一次含噪的信道观测，而路径正确性之间的两两相关性会将投票的设计效应有效样本量限制在一个有限上限。广义最小二乘（GLS）分析表明，在可交换性条件下，潜在嵌入的最优对称线性合并器是均匀的，这支持多数投票作为标准SC的自然默认选择，同时在异构提示模板分支下为加权或剪枝留出了空间。在5个模型和12个基准测试的实验中，提示模板多样性在57个有效组合中的55个里降低了路径相关性，其效果最强的（后续内容截断）……

    arXiv:2609.38829v1 Announce Type: new  Abstract: Multi-path reasoning methods such as self-consistency (SC) sample $K$ reasoning paths and choose the most frequent answer. However, their gains quickly plateau as $K$ increases, and existing methods do not predict when this saturation will occur. We formalize multi-path LLM reasoning as a diversity combining problem from wireless communications: each path is a noisy channel observation, and the pairwise correlation of path correctness caps the design-effect effective sample size of the vote at a finite ceiling. Generalized least squares (GLS) analysis shows that, under exchangeability, the optimal symmetric linear combiner of latent embeddings is uniform, supporting majority vote as the natural default in standard SC while leaving room for weighting or pruning under heterogeneous prompt-template branches. Across 5 models and 12 benchmarks, prompt-template diversity reduces path correlation in $55$ of $57$ valid cells, with the strongest 
    
[^292]: 从差距中学习：面向GRPO的差分感知优势剪枝与自适应Rollout采样

    Learn from the Gap: Differential-Aware Advantage Pruning with Adaptive Rollout Sampling for GRPO

    [https://arxiv.org/abs/2609.36932](https://arxiv.org/abs/2609.36932)

    提出即插即用的FastRL强化学习框架，通过差分感知的优势剪枝自适应保留高价值轨迹并最大化梯度多样性，同时结合自适应rollout采样机制，显著提升GRPO的训练效率与策略学习效果。

    

    近年来，组相对策略优化（GRPO）及其变体被开发用于策略优化，并展现出显著的性能提升。然而，这些方法通常由于每个问题需要多次rollout采样以及跨rollout重复进行逐token概率评估，而产生大量计算开销。此外，低信息量或高度同质化的轨迹会降低下游学习信号的效率，阻碍模型优化并限制最终性能。为解决这些问题，我们提出了FastRL，一种新颖的即插即用强化学习框架，可同时提升训练效率与策略学习的有效性。具体而言：1）我们引入了一种优势感知的剪枝策略，在最大化轨迹间梯度多样性的同时，选择性地保留高优势轨迹；2）随后，我们设计了一种自适应rollout采样机制，以动态调整采样过程（摘要内容在此截断）。

    arXiv:2609.36932v1 Announce Type: new  Abstract: Recently, Group Relative Policy Optimization (GRPO) and its variants have been developed for policy optimization and demonstrated notable performance gains. However, these methods usually incur substantial computational overhead due to per-question multi-rollout sampling and repeated per-token probability evaluation across rollouts. Furthermore, low-information or highly homogeneous trajectories can degrade downstream learning signal efficiency, hindering model optimization and limiting final performance. To address these issues, we propose FastRL, a novel plug-and-play reinforcement learning framework that simultaneously improves training efficiency and the effectiveness of policy learning. Specifically, 1) We introduce an advantage-aware pruning strategy to selectively preserve high-advantage trajectories while maximizing inter-trajectory gradient diversity. 2) Then, we design an adaptive rollout sampling mechanism to dynamically adjus
    
[^293]: EvoIn：面向智能体微调的演化与内化桥梁

    EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning

    [https://arxiv.org/abs/2609.35290](https://arxiv.org/abs/2609.35290)

    EvoIn提出了一个连接演化与内化的智能体微调框架，通过分析执行轨迹来演化并验证新的决策流程，再将改进的推理轨迹改写为去除框架指令引用的自包含推理并微调模型，从而将决策逻辑内化为模型自身的能力。

    

    近期的工作探索了通过共同演化智能体的执行框架和模型来提升智能体性能，但往往采用“大杂烩”式的方法，将新工具、新决策流程以及对演化后框架的模型适配统统归入单一的“智能体改进”概念之下。在本文中，我们转而研究智能体如何改进其自身的决策流程。具体而言，我们提出了EvoIn，一个连接演化与内化的智能体微调框架。EvoIn首先分析智能体的执行轨迹，通过在框架中临时实例化的方式来演化并验证新的决策流程。经过验证的流程会引导智能体生成改进后的推理轨迹。随后，这些轨迹被改写为自包含的推理轨迹，在去除对框架指令的显式引用的同时，将由此诱导出的决策逻辑表达为模型自身的推理。最后，EvoIn对模型进行微调。

    arXiv:2609.35290v2 Announce Type: replace  Abstract: Recent work has explored improving agents by jointly evolving their harnesses and models, but often takes a ''potpourri'' approach that bundles together new tools, new decision-making procedures, and model adaptation to the evolved harness under a single notion of agent improvement. In this paper, we instead investigate how agents can improve their decision-making procedures. In particular, we propose EvoIn, an agent fine-tuning framework that bridges evolution and internalization. EvoIn first analyzes agent execution traces to evolve and validate new decision-making procedures by temporarily instantiating them in the harness. The validated procedures guide the agent to generate improved reasoning traces. These traces are then rewritten into self-contained reasoning traces, removing explicit references to harness instructions while expressing the induced decision logic as the model's own reasoning. Finally, EvoIn fine-tunes the model
    
[^294]: 面向呼吸支持的响应感知患者动力学学习

    Learning response-aware patient dynamics for respiratory support

    [https://arxiv.org/abs/2609.32782](https://arxiv.org/abs/2609.32782)

    提出了一种响应感知的患者动力学模型，将预测的生理变化显式分解为状态依赖的基线动力学和呼吸支持相关的偏差，并以室内空气作为参考基准，从而更好地建模呼吸支持对危重患者短期生理轨迹的个体化影响。

    

    呼吸支持能够塑造危重症患者的短期生理轨迹，但接受相同干预措施的患者可能遵循不同的生理轨迹。临床患者动力学模型通常基于近期生理状态和记录的干预措施来预测未来状态，而生理变化主要通过预测的未来状态来表示。我们提出了一种响应感知的患者动力学模型，该模型在自回归状态更新过程中显式表示生理变化。该模型将预测的生理变化分解为依赖于状态的基线动力学和与呼吸支持相关的偏差，并以室内空气（room air）作为该分解的参考基准。我们对这种基于参考锚定的公式化方法进行了形式化分析。响应通路对预测的生理变化进行编码，并在整个预测时间范围内利用它来更新潜在的患者状态。在ICU……

    arXiv:2609.32782v2 Announce Type: replace  Abstract: Respiratory support can shape the short-term physiological trajectory of critically ill patients, but patients receiving the same intervention may follow different physiological trajectories. Clinical patient dynamics models typically predict future states from recent physiology and recorded interventions, while physiological change is mainly represented through the predicted future state. We propose a response-aware patient dynamics model that explicitly represents physiological change during autoregressive state updating. The model decomposes predicted physiological change into state-dependent baseline dynamics and respiratory-support-associated deviations, with room air providing a reference for the decomposition. We provide a formal analysis of this reference-anchored formulation. A response pathway encodes the predicted physiological change and uses it to update the latent patient state across the forecast horizon. Across ICU co
    
[^295]: GAUGE：面向前馈式4D跟踪的组级视点不一致性校正

    GAUGE: Group-Wise View-Inconsistency Rectification for Feed-Forward 4D Tracking

    [https://arxiv.org/abs/2609.32596](https://arxiv.org/abs/2609.32596)

    本文提出GAUGE，一个无需训练、与模型无关的事后处理模块，通过从位移方向一致性恢复运动组并估计每帧径向尺度，来校正前馈式4D跟踪中沿视线方向的组级不一致性误差。

    

    前馈模型直接从单目视频中回归稠密的3D点轨迹，然而全局对齐后的残差仍然很大，且缺乏结构性的解释。在不同模型和数据集的动态查询点上的测量表明，误差集中在视线方向上，而每个运动组所需的尺度校正各不相同。尽管如此，预测的位移方向仍支持可靠的分组，其中位角远低于90°的随机基线。因此，残差的系统性部分对应于每个运动组的一族径向自由度，其方向无法被2D观测所约束。我们将其称为组级视点不一致性。我们提出了GAUGE（Group-wise Adaptive Unsupervised Gauge Estimation，组级自适应无监督规范估计），这是一个无需训练、与模型无关的事后处理模块。它通过方向一致性和空间连通性恢复运动组，然后估计每帧的径向尺度和组级（摘要在此处截断）

    arXiv:2609.32596v2 Announce Type: replace-cross  Abstract: Feed-forward models regress dense 3D point trajectories directly from monocular video, yet the residual after global alignment is substantial and lacks a structural explanation. Measured on dynamic query points across models and datasets, the error concentrates along the view direction, while the scale correction each motion group requires differs. The predicted displacement direction nevertheless supports reliable grouping, with a median angle far below the 90{\deg} random baseline. The systematic part of the residual is therefore a family of radial degrees of freedom per motion group, along directions 2D observations cannot constrain. We call it group-wise view inconsistency. We present GAUGE (Group-wise Adaptive Unsupervised Gauge Estimation), a training-free and model-agnostic post-hoc module. It recovers motion groups from direction consistency and spatial connectivity, then estimates a per-frame radial scale and group-lev
    
[^296]: 学习锚定提示空间以实现大语言模型的持续适配

    Learning an Anchored Prompt Space for Continual Adaptation of Large Language Models

    [https://arxiv.org/abs/2609.32499](https://arxiv.org/abs/2609.32499)

    提出LAPS方法，通过自蒸馏对齐历史提示，并构建以任务特定软提示为顶点、贝塞尔控制提示塑造几何结构的锚定提示空间，从而在持续适配大语言模型时保留提示有效性并促进跨任务正向迁移。

    

    持续适配大语言模型需要在获取新知识的同时保留先前学习到的能力。尽管联合适配模型参数和任务特定的软提示是一种有前景的方法，但它存在两个关键局限性：先前学习的提示可能随着模型的演进而失去有效性，并且支持跨任务迁移的提示之间的关系没有被显式学习。我们提出了学习锚定提示空间（LAPS），该方法保留历史提示的有效性，并学习任务特定软提示之间的关系以促进正向迁移。历史提示首先通过自蒸馏与更新后的模型对齐。随后，LAPS构建一个锚定提示空间，其顶点对应于已学习的任务特定软提示，其中间几何结构由可学习的贝塞尔控制提示塑造。一旦学习了这个空间，LAPS即可识别最佳表现……

    arXiv:2609.32499v2 Announce Type: replace-cross  Abstract: Continually adapting large language models requires acquiring new knowledge while preserving previously learned capabilities. Although jointly adapting model parameters and task-specific soft prompts is promising, it has two key limitations: previously learned prompts may lose effectiveness as the model evolves, and relationships among prompts that could support cross-task transfer are not explicitly learned. We propose Learning an Anchored Prompt Space (LAPS), which preserves historical prompt effectiveness and learns relationships among task-specific soft prompts to facilitate positive transfer. Historical prompts are first aligned with the updated model through self-distillation. LAPS then constructs an anchored prompt space whose vertices correspond to learned task-specific soft prompts and whose intermediate geometry is shaped by learnable B\'ezier control prompts. Once this space is learned, LAPS identifies the best-perfo
    
[^297]: 合成医院：一个开放、可验证、经医生验证的纵向电子健康记录基准

    Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark

    [https://arxiv.org/abs/2609.30027](https://arxiv.org/abs/2609.30027)

    提出了“合成医院”——首个完全由公开医学教育材料构建、经医生验证、具有完整溯源链和可验证真实标准答案的开放合成纵向EHR基准，解决了真实EHR数据无法公开共享且缺乏可验证答案的核心障碍。

    

    前沿语言模型很少被应用于临床工作流程，原因在于开发此类模型所需的现实、纵向的基准测试数据十分稀缺。真实的电子健康记录（EHR）数据由于隐私、伦理或数据使用等问题无法公开共享，而且由于病历仅记录临床医生所记载的内容，因此缺乏可验证的真实标准答案。我们提出了“合成医院”，这是一个开放、完全合成、基于事实的纵向EHR基准，解决了公开共享和可验证真实标准这两大障碍。该基准完全由公开的医学教育材料构建，不含任何受保护的健康信息，包含1,268名纵向患者和5,602次就诊记录，其中每一个诊断、发现和时间关系均基于标准本体（ICD-10-CM、SNOMED CT、LOINC）进行标注，并拥有完整的溯源链，可追溯至源医学教育材料。合成医院通过一个模拟医院环境提供服务……（原文摘要在此处截断）

    arXiv:2609.30027v1 Announce Type: new  Abstract: Frontier language models are rarely used in clinical workflows because the realistic, longitudinal benchmarks needed to develop them are scarce. Real electronic health record (EHR) data cannot be openly shared due to privacy, ethics or data use issues and it does not contain verifiable ground truth since the chart records only reflect what clinicians documented. We introduce Synthetic Hospital, an open, fully synthetic, fact-grounded longitudinal EHR benchmark that resolves the open sharing and verifiable ground truth barriers. Built entirely from public medical-education material with no protected health information, it comprises 1,268 longitudinal patients and 5,602 encounters, where every diagnosis, finding, and temporal relation is grounded in standard ontologies (ICD-10-CM, SNOMED CT, LOINC) and with a complete provenance chain back to its source medical education material. Synthetic Hospital is served through a simulated hospital r
    
[^298]: Vision2CAD：用于参数化CAD建模中显式几何引用与定位的视觉代理框架

    Vision2CAD: A Visual Agent Harness for Explicit Geometry Referencing and Localization in Parametric CAD Modeling

    [https://arxiv.org/abs/2609.22688](https://arxiv.org/abs/2609.22688)

    Vision2CAD通过将视觉语言模型推理与确定性CAD内核操作相结合，并引入基于ID的显式几何选择、局部坐标桥接和投影边缘定位机制，显著提升了参数化CAD建模中的几何引用与定位精度。

    

    生成参数化CAD模型需要精确的几何形状和稳定的特征依赖关系。现有方法在选择几何参考、解释草图平面局部坐标系以及建立针对投影外部几何的草图约束方面面临挑战。我们提出了Vision2CAD，这是一个将视觉语言模型（VLM）推理与确定性CAD内核操作相结合的视觉代理框架。基于ID的接口支持显式几何选择，局部坐标桥接器将视图坐标转换为草图坐标，投影边缘定位支持外部草图约束。这些机制在所支持的建模操作和约束类型范围内建立特征依赖关系。我们还介绍了几何显式参考数据集（GERD），该数据集在每个建模步骤中对齐了命令、几何状态和ID。在GERD-EVL和DeepCAD测试子集上，Vision2CAD将mIoU分别提高了11.1%和5%（摘要原文在此处截断）

    arXiv:2609.22688v1 Announce Type: cross  Abstract: Generating parametric CAD models requires accurate geometry and stable feature dependencies. Existing methods face challenges in selecting geometric references, interpreting sketch-plane local coordinates, and establishing sketch constraints to projected external geometry. We present Vision2CAD, a visual agent harness that combines vision-language model (VLM) reasoning with deterministic CAD kernel operations. An ID-based interface supports explicit geometry selection, a local-coordinate bridge converts view coordinates into sketch coordinates, and projected-edge localization supports external sketch constraints. These mechanisms establish feature dependencies within the supported modeling operations and constraint types. We also introduce the Geometry Explicit Reference Dataset (GERD), which aligned commands, geometry states and IDs at every modeling step. On GERD-EVL and a DeepCAD test subset, Vision2CAD improves mIoU by 11.1\% and 5
    
[^299]: 为您量身定制：个性化语言模型的纵向效应

    Tailored to you: longitudinal effects of personalising language models

    [https://arxiv.org/abs/2609.20077](https://arxiv.org/abs/2609.20077)

    该研究通过对992名参与者进行为期五天的纵向实验，首次系统考察了基于记忆和基于调查两种个性化方法对用户与语言模型持续互动及其自我认知、人际关系的长期影响。

    

    开发个性化语言模型的兴趣正在迅速增长。虽然个性化通常被视为更好地满足多样化用户需求的机制，但与个性化模型的持续互动如何影响人们对AI的感知和行为，目前仍知之甚少。最关键的是，在即时人机交互循环之外的下游后果——例如对用户自我认知和人际关系的影响——在很大程度上尚未被研究。在这项研究中，我们招募了992名参与者，在五天内每天与语言模型进行寻求建议的互动，将非个性化基线与两种个性化方法的结果进行比较：基于记忆的个性化（以先前对话历史为条件）和基于调查的个性化（以研究前通过访谈调查收集的信息为条件）。我们发现，随时间推移，人机互动中产生的若干变化是由……

    arXiv:2609.20077v1 Announce Type: new  Abstract: Interest in developing personalised language models is rapidly growing. While personalisation is often viewed as a mechanism to better serve diverse user needs, the effects of sustained interactions with personalised models on people's perception of and behaviour toward AI remain poorly understood. Most critically, downstream consequences outside the immediate human--AI interaction loop, such as effects on users' self-perceptions and interpersonal relationships, remain largely unexamined. In this study, we recruited 992 participants to complete daily advice-seeking interactions with language models over the course of five days, comparing outcomes from a non-personalised baseline against two personalisation approaches: memory-based (conditioned on prior conversational history) and survey-based (conditioned on information collected through a pre-study intake survey). We find that several changes in human-AI interaction over time are driven
    
[^300]: MUMINS：元数据条件化的不确定性感知医学图像下一状态合成

    MUMINS: Metadata-conditioned Uncertainty-aware Medical Image Next-state Synthesis

    [https://arxiv.org/abs/2609.17169](https://arxiv.org/abs/2609.17169)

    MUMINS是一个高效的扩散框架，在单次反向扩散过程中联合扩散基线扫描与随访残差以合成随访医学图像，并同时生成空间不确定性图，通过元数据条件化和在每个去噪步骤动态重注入基线作为软锚点来保留细粒度解剖结构。

    

    预测诸如肿瘤生长和神经退行性变等解剖学变化是一项具有挑战性的生成式视觉任务。形态学演变相对于静态解剖结构而言非常细微、高度患者特异，且本质上是随机的。现有方法存在若干问题：确定性网络忽略了生物学随机性，而标准扩散模型需要计算上极其昂贵（令人望而却步）的多轮采样来量化不确定性。我们提出了MUMINS（元数据条件化的不确定性感知医学图像下一状态合成），这是一个高效的扩散框架，它在单个反向扩散过程中联合扩散基线扫描及其随访残差（两者相加以合成随访扫描），同时预测空间不确定性图。该模型以时间间隔和相关元数据为条件，通过在每个去噪步骤中将基线动态重新注入作为软锚点，来保留细粒度的解剖结构。

    arXiv:2609.17169v1 Announce Type: cross  Abstract: Forecasting anatomical changes such as tumor growth and neurodegeneration is a challenging generative vision task. Morphological evolution is subtle relative to static anatomy, highly patient-specific, and inherently stochastic. Existing methods struggle with several issues: deterministic networks ignore biological stochasticity, while standard diffusion models require computationally prohibitive multi-pass sampling to quantify uncertainty. We propose MUMINS (Metadata-conditioned Uncertainty-aware Medical Image Next-state Synthesis), an efficient diffusion framework that jointly diffuses a baseline scan and its follow-up residual, summed to synthesize the follow-up scan, while concurrently predicting a spatial uncertainty map, in a single reverse diffusion process. Conditioned on the time interval and relevant metadata, it preserves fine-grained anatomy by dynamically re-injecting the baseline as a soft anchor at every denoising step, 
    
[^301]: 基于法律推理的世界模型规划立法化

    Legislating World-Model-Based Planning with Legal Reasoning

    [https://arxiv.org/abs/2609.15113](https://arxiv.org/abs/2609.15113)

    本文提出了一个结合可废止道义逻辑（DDL）与学习型世界模型的法律规划栈，解决了感知基础化与法律约束转化的同构差距问题，实现了在机器人执行非法动作前进行事前干预的法律治理机制。

    

    随着机器人系统日益通用化，需要法律规范来将其融入社会。本文扩展了法律源文本与其编码之间对齐的同构问题，并衡量了机器人规范控制的两个关键挑战：（1）基础化同构差距，即感知错误导致法律推理中产生错误的原子事实；（2）本体论同构差距，即同一法律结论可以被忠实地转化为多种不同的规划约束。本文引入了一个法律规划栈，采用可废止道义逻辑（DDL）来约束运动规划器。该栈利用学习到的世界模型进行规划并提供法律上下文，实现了事前治理，能够在非法行为执行之前进行干预。该系统被部署在一个模拟机器人手臂上，用于在3×3网格上推动立方体。研究发现：（1）受法律约束的智能体遵守规范的频率显著高于……（摘要原文在此处截断）

    arXiv:2609.15113v1 Announce Type: cross  Abstract: As robotic systems grow more general, legal norms are needed to integrate them into society. This paper extends the isomorphism problem of aligning legal source texts with their encodings, and measures two key challenges to robot normative control: (1) the \textit{grounding isomorphism gap}, where perception error grounds false atoms for legal reasoning, and (2) the \textit{ontological isomorphism gap}, where one legal conclusion admits many faithful translations into planning constraints. The paper introduces a legal planning stack that employs Defeasible Deontic Logic (DDL) to constrain a motion planner. The stack leverages learned world models to plan and to provide legal context, enabling \textit{ex ante} governance that intervenes before an illegal action is executed. It was deployed on a simulated robot arm pushing a cube across a $3\times3$ grid. The findings were (1) the legislated agent abided substantially more often than the
    
[^302]: 闪电编织：通过能力组合提升推理模型的准确率-效率前沿

    Lightning Weave: Improving the Accuracy-Efficiency Frontier of Reasoning Models through Capability Composition

    [https://arxiv.org/abs/2609.14708](https://arxiv.org/abs/2609.14708)

    Lightning Weave 提出了一种后训练框架，通过在线策略蒸馏将从独立专家模型中分别学到的准确率和效率能力组合到单一学生模型中，从而同时提升推理模型的推理准确率和推理效率。

    

    高效推理的核心目标是提升准确率-效率前沿。然而，同时提升推理准确率和推理效率可能颇具挑战性，因为这两个目标往往偏好不同的推理行为。经过独立后训练的模型已经在准确率和效率方面各自展现出不同的优势。我们提出了 Lightning Weave，这是一个后训练框架，通过在线策略蒸馏将独立学习到的能力提取并组合到一个单一的学生模型中。每项所获得的能力都用从训练前模型到相应专家模型的策略偏移来表示。Lightning Weave 在共享的学生 token 状态上结合对齐的对数比率偏移，并使用 Tilted-Target DOPD 将缓存的信号转换为稳定的学习目标。每个锚点对只需对缓存的轨迹进行一次评分，即可支持后续的学生训练，而无需同时运行多个在线锚点模型。

    arXiv:2609.14708v1 Announce Type: new  Abstract: A core goal of efficient reasoning is to improve the accuracy-efficiency frontier. However, jointly improving reasoning accuracy and inference efficiency can be challenging, as the two objectives can favor different reasoning behaviors. Independently post-trained models already offer distinct strengths in accuracy and efficiency. We introduce Lightning Weave, a post-training framework that extracts and composes these independently learned capabilities in a single student through on-policy distillation. Each acquired capability is represented by the policy shift from the model before post-training to the resulting specialist. Lightning Weave combines aligned log-ratio shifts at shared student token states and uses Tilted-Target DOPD to convert the cached signals into a stable learning target. Each anchor pair scores the cached trajectories once, enabling subsequent student training without serving multiple live anchor models concurrently.
    
[^303]: 当评分标准失效时：幻觉揭示医学AI评估中的盲点

    When Rubrics Fail: Hallucinations Reveal Blind Spots in Medical AI Evaluation

    [https://arxiv.org/abs/2609.12718](https://arxiv.org/abs/2609.12718)

    研究发现基于评分标准的医学LLM评估常遗漏临床相关的幻觉错误，尤其是未预料到的错误，仅在明确核查事实时才有效，揭示了当前医学AI评估方法的盲点。

    

    幻觉会削弱临床医生对大语言模型（LLM）的信任，因此评估方法能否捕捉到临床相关的错误至关重要。基于评分标准（rubric）的评估已成为医学领域评估LLM的主流方法，但评分标准得分是否能反映此类错误尚不清楚。我们首先在使用MedHallu的受控环境中研究这一问题，发现更具体的评分标准能更好地区分正确回答与幻觉回答。为系统性地验证这一点，我们构建了医学幻觉类型分类法，以及经临床医生验证的错误注入流水线，用于生成匹配的正确回答和注入错误的回答。在HealthBench、HealthBench Professional和LiveMedBench上的实验表明，我们的临床相关幻觉常被评分标准遗漏，得分往往保持不变。我们发现评分标准在明确核查事实时最为有效，而对于其未预料到的额外或意外错误则效果较差。

    arXiv:2609.12718v2 Announce Type: replace  Abstract: Hallucinations can undermine clinician trust in LLMs, making it important that evaluation methods capture clinically relevant errors. Rubric-based evaluation has become the leading approach for assessing LLMs in medicine, but it is unclear whether rubric scores reflect such errors. We first study this in a controlled setting using MedHallu, finding that more specific rubrics better distinguish correct from hallucinated responses. To test this systematically, we develop a taxonomy of medical hallucination types and a clinician-validated error-injection pipeline that creates matched correct and error-injected responses. Across HealthBench, HealthBench Professional, and LiveMedBench, our clinically relevant hallucinations are missed by rubrics, often leaving scores unchanged. We find that rubrics are most effective when explicitly checking facts, and are less effective for additional or unexpected errors they do not anticipate. A prelim
    
[^304]: 基准雷达：面向AI基准与评估的动态数据库和搜索引擎

    Benchmark Radar: A Living Database and Search Engine for AI Benchmarks and Evaluation

    [https://arxiv.org/abs/2609.11115](https://arxiv.org/abs/2609.11115)

    本文提出Benchmark Radar，一个每日自动发现并整合AI基准论文、数据集和代码的动态数据库与搜索引擎，为LLM评估、智能体、编程、推理和安全等领域提供可检索的基准目录、来源引用和分数历史。

    

    基准研究人员和大语言模型（LLM）及其他AI系统的开发者需要找到相关的评估方法、定位其基准数据集和代码，并理解已报告分数背后的设置。我们提出了Benchmark Radar（基准雷达），这是一个用于检索和发现AI基准的动态数据库和搜索引擎，涵盖LLM评估、智能体和工具使用基准、编程、推理、安全性以及特定领域评估。该系统将每日发现的基准论文、代码仓库、数据集和发布版本与可搜索的基准目录、模型卡和技术报告中的提及以及分数历史相结合。它保留了源身份信息和引用，以便读者可以查看候选基准及其评估证据。每日发现基于37个来源：13个直接连接器和24个第一方研究与工程信息源。该目录包含来自4个基准目录的1,283条源记录

    arXiv:2609.11115v1 Announce Type: cross  Abstract: Benchmark researchers and developers of large language models (LLMs) and other AI systems need to find relevant evaluations, locate their benchmark datasets and code, and understand the settings behind reported scores. We present Benchmark Radar, a living database and search engine for retrieval and discovery of AI benchmarks, covering LLM evaluation, agentic and tool-use benchmarks, coding, reasoning, safety, and domain-specific evaluations. The system combines daily discovery of benchmark papers, repositories, datasets, and releases with a searchable benchmark catalog, mentions in model cards and technical reports, and score histories. It retains source identities and citations so readers can inspect candidate benchmarks and their evaluation evidence. Daily discovery draws on 37 sources: 13 direct connectors and 24 first-party research and engineering feeds. The catalog contains 1,283 source records drawn from 4 benchmark catalogs an
    
[^305]: Suan：纠正大语言模型中的直接偏好安全对齐

    Suan: Rectifying Direct Preference Safety Alignment in Large Language Models

    [https://arxiv.org/abs/2609.08634](https://arxiv.org/abs/2609.08634)

    Suan是一种新颖的偏好优化算法，通过直接在梯度层面构建优化目标（绕过标准变分推导），使大语言模型在实现卓越安全对齐的同时完全保留回应实用性。

    

    将强大的安全防护机制集成到大语言模型（LLM）中，对于提供有用且无害的回应至关重要。尽管专有系统展现出可靠的安全控制，但其底层方法和权衡取舍在很大程度上仍未公开。在开放权重模型中实现相当的安全性仍然是一个持续的挑战，因为经过后训练的模型变体经常出现过拒答和整体质量下降的问题。为了克服这些缺陷，我们提出了Suan，一种新颖的偏好优化算法。与现有方法不同，我们直接在梯度层面构建优化目标，绕过了标准的变分推导。由此，我们获得了更具可解释性和更稳健的训练动态。在多种竞争性基线和基准测试上的广泛评估表明，Suan在实现卓越安全对齐的同时，完全保留了回应的实用性。

    arXiv:2609.08634v1 Announce Type: cross  Abstract: Integrating robust safety guardrails into Large Language Models (LLMs) is essential for delivering helpful yet harmless responses. While proprietary systems exhibit reliable safety controls, their underlying methodologies and trade-offs remain largely undisclosed. Achieving comparable security in open-weight models remains a persistent challenge, as post-trained variants frequently suffer from over-refusal and degraded general quality. To overcome these drawbacks, we introduce Suan, a novel preference optimization algorithm. Unlike existing methods, we formulate the optimization objective directly at the gradient level, bypassing the standard variational derivation. As a result, we obtain more interpretable and robust training dynamics. Extensive evaluations across a diverse suite of competitive baselines and benchmarks demonstrate that Suan achieves superior safety alignment while fully preserving response utility.
    
[^306]: 大语言模型能设计视频编码工具吗？以Planar模式为例的案例研究

    Can LLMs Design Video Coding Tools? A Case Study on Planar Mode

    [https://arxiv.org/abs/2609.01535](https://arxiv.org/abs/2609.01535)

    本文通过“生成-评估”循环的实证研究，首次证明大语言模型能够自动设计并迭代改进视频编码中的Planar帧内预测模式，在VVenC编码器上实现了0.18%的码率节省且复杂度开销仅0.4%。

    

    本文探讨了大语言模型（LLM）能否设计视频编码工具，这是一项极具挑战性的任务，因为工具修改之间存在复杂的算法耦合。特别地，我们对Planar模式进行了实证案例研究，该模式是视频编码标准中长期存在的帧内预测工具。我们的实验在“生成-评估”循环中进行：由LLM生成新的Planar预测器，通过编码器试验评估其编码性能，LLM再根据评估反馈重新生成改进的实现。我们首先研究了在Fraunhofer通用视频编码器（VVenC）的更快预设下直接替换默认Planar模式。实验结果表明，LLM生成的模式在这种轻量级工具集上能够超越传统Planar模式，在标准测试基准上实现了0.18%的码率节省，同时仅带来0.4%的复杂度开销。

    arXiv:2609.01535v1 Announce Type: cross  Abstract: This paper explores whether large language models (LLMs) can design video coding tools, a highly challenging task due to the intricate algorithmic coupling of tool modifications. In particular, we present an empirical case study on the Planar mode, a long-standing intra prediction tool in video coding standards. Our experiments operate within a generation-and-evaluation loop, with the LLM generating new Planar predictors, encoder trials evaluating their coding performance, and the LLM re-generating refined implementations based on the evaluation feedback. We first examine directly replacing the default Planar mode in the Fraunhofer Versatile Video Encoder (VVenC) under its faster preset. Experimental results demonstrate that the LLM-generated mode can outperform the conventional Planar mode on this lightweight toolset, achieving 0.18% bitrate savings with 0.4% complexity overhead on the standard benchmark. We further extend our evaluat
    
[^307]: 防御即技能：为技能增强型智能体演化运行时守卫技能

    Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents

    [https://arxiv.org/abs/2609.01487](https://arxiv.org/abs/2609.01487)

    提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。

    

    技能增强型智能体将可复用技能作为持久的运行时上下文进行加载，这虽然提升了任务性能，但也为恶意技能开辟了一条持久通道，使其能够影响智能体未来的行动。此类技能可能在具体用户任务和工作区状态使不安全操作显得有用时，才泄露机密、破坏代码、绕过审批，或暂存数据以待外泄。这使得仅靠安装前的审查已不再足够，需要运行时的、任务条件化的保护。我们提出 Defense-as-Skill（防御即技能），一种将运行时守卫本身实现为可安装、可检查、可编辑技能的防御范式。我们的守卫 SkillSonar 与不可信的任务技能并行运行，依据用户的任务边界对敏感操作进行检查，并将每个操作路由至允许、重新规划或请求确认的决策，而无需修改底层智能体运行时。为研究这一场景，我们构建了 SCOPE-R，一个任务条件化的数据集，涵盖 6 个风险类别和 21 个子类别。

    arXiv:2609.01487v1 Announce Type: cross  Abstract: Skill-augmented agents load reusable skills as persistent runtime context, improving task performance but also giving malicious skills a durable channel for steering future actions. Such skills may leak secrets, corrupt code, bypass approvals, or stage data for exfiltration only after a concrete user task and workspace state make the unsafe action appear useful. This makes pre-install vetting insufficient and calls for runtime, task-conditioned protection. We propose Defense-as-Skill, a defense paradigm that implements the runtime guard itself as an installable, inspectable, and editable skill. Our guard, SkillSonar, runs alongside untrusted task skills and checks sensitive actions against the user's task boundary, routing each action to an allow, replan, or confirmation decision without modifying the underlying agent runtime. To study this setting, we construct SCOPE-R, a task-conditioned dataset covering 6 risk families and 21 sub-ca
    
[^308]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^309]: 从透视到鱼眼的深度估计与开放词汇分割

    From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation

    [https://arxiv.org/abs/2608.27860](https://arxiv.org/abs/2608.27860)

    该论文提出名为“畸变扩展器（DEX）”的可学习参数方法，通过自监督对齐损失将鱼眼图像的潜在嵌入映射到透视图像的嵌入空间，使视觉基础模型能够泛化到鱼眼相机，实现高保真的深度估计与开放词汇分割。

    

    视觉基础模型能够以高保真的估计结果泛化到各种三维（3D）场景，其经验上的成功可归因于在大规模透视图像数据集上的训练。然而，当迁移到宽视场（FoV）图像（例如由鱼眼相机拍摄的图像）时，由于图像像素的径向畸变所引起的协变量偏移，这些模型会产生错误的输出。我们提出了一种将视觉基础模型泛化到鱼眼相机的方法。该方法的核心在于一组可学习参数，称为畸变扩展器（Distortion Extenders, DEX），用于建模鱼眼畸变系数以及鱼眼图像与透视图像之间在潜在空间中编码的分布偏移。通过最小化自监督对齐损失，DEX 将鱼眼图像的潜在嵌入转换为类似于透视图像的嵌入，从而恢复高保真的估计结果。DEX 是一种

    arXiv:2608.27860v2 Announce Type: replace-cross  Abstract: Vision foundation models are capable of generalizing across 3-dimensional (3D) scenes with high-fidelity estimates; their empirical success can be attributed to training on large-scale datasets of perspective images. However, when transferred to wide field-of-view (FoV) images, such as those captured by fisheye cameras, they return erroneous outputs due to a covariate shift stemming from the radial distortion on the image pixels. We propose a method to generalize vision foundation models to fisheye cameras. The crux of our method lies in a set of learnable parameters, termed Distortion Extenders (DEX), that model the fisheye distortion coefficients and the distributional shift between fisheye and perspective images encoded in the latent space. By minimizing a self-supervised alignment loss, DEX transforms the latent embeddings of fisheye images to resemble those of perspective images to recover high-fidelity estimates. DEX is a
    
[^310]: 可操作的CBFI：融合结构分解与因果反事实补救的表格机器学习框架

    Actionable CBFI: Integrating Structural Decomposition and Causal Counterfactual Recourse for Tabular Machine Learning

    [https://arxiv.org/abs/2608.27821](https://arxiv.org/abs/2608.27821)

    该论文提出A-CBFI框架，基于结构因果模型将特征重要性分解与因果反事实补救相结合，通过隔离协同交互瓶颈并释放抑制性结构锁，为表格机器学习生成因果有效、可操作的针对性干预方案，克服了现有方法因果无效与干预分散的问题。

    

    可解释人工智能（XAI）日益呼唤可操作的反事实补救，然而现有方法面临因果无效性、过高认知负担以及预测失效等挑战。穷举式因果搜索算法往往需要对多个属性进行修改，而加性归因引导的方法（如SHAP）则忽略了高阶特征之间的协同作用，导致在XGBoost等复杂非线性模型中出现次优的预测动力和分散的干预努力。为弥合这一差距，我们提出了可操作的基于案例的特征重要性（A-CBFI），这是一个面向表格机器学习的“诊断-处方”一体化框架。该框架植根于结构因果模型（SCM），能够隔离协同交互瓶颈并释放抑制性结构锁，进而将其转化为有针对性的干预措施。通过在数学上将活跃的用户干预空间……（摘要在此处不完整）

    arXiv:2608.27821v1 Announce Type: new  Abstract: Explainable artificial intelligence (XAI) increasingly calls for actionable counterfactual recourse, yet current methodologies face challenges related to causal invalidity, excessive cognitive burden, and predictive failure. Exhaustive causal search algorithms often require modifications to multiple attributes, whereas additive attribution-guided methods, such as SHAP, ignore higher-order feature synergies, leading to suboptimal predictive momentum and diffuse intervention effort in complex nonlinear models, such as XGBoost. To bridge this gap, we introduce actionable case-based feature importance (A-CBFI), a diagnosis-prescription integrated framework for tabular machine learning. Grounded in structural causal models (SCMs), A-CBFI isolates synergistic interaction bottlenecks and releases suppressive structural locks, translating them into targeted interventions. By mathematically separating the active user intervention space (L_{\mathr
    
[^311]: MemPrism：面向长时程智能体的任务条件化关系记忆视图

    MemPrism: Task-Conditioned Relational Memory Views for Long-Horizon Agents

    [https://arxiv.org/abs/2608.06745](https://arxiv.org/abs/2608.06745)

    提出MemPrism框架，通过任务条件化的关系记忆视图将持久经验存储与决策时工作记忆分离，动态构建关系视图以解决长时程智能体中的表征失配问题，并在具身智能和网络智能体基准上取得持续提升。

    

    执行多步网络任务的长时程智能体必须复用先前的页面观察和交互记录来支持后续动作。然而，保留相关历史并不保证这些历史能够暴露当前决策所需的关系，我们将这一差距称为“表征失配”。为此，我们提出了MemPrism，一个任务条件化的关系记忆框架，它将持久经验存储与决策时工作记忆相分离。MemPrism将交互记录为事件流，并根据当前任务上下文动态构建关系视图。一个轻量级的视图策略负责选择关系结构、证据范围、结果条件和粒度，而确定性的组合器与渲染器则将历史事实转化为临时的光学工作记忆视图，供冻结的任务策略使用。在长时程具身智能体和网络智能体基准上的实验表明，MemPrism持续提升了……（摘要在此处截断）

    arXiv:2608.06745v2 Announce Type: replace  Abstract: Long-horizon agents performing multi-step Web tasks must reuse earlier page observations and interactions to support subsequent actions. However, preserving relevant history does not ensure that it exposes the relations needed for the current decision. We call this gap representation mismatch. To this end, we propose MemPrism, a task-conditioned relational memory framework that separates persistent experience storage from decision-time working memory. MemPrism records interactions as the event stream and dynamically constructs relational views according to the current task context. A lightweight view policy selects the relation structure, evidence range, outcome condition, and granularity, while a deterministic composer and renderer transform historical facts into a temporary optical working-memory view for a frozen task policy. Experiments on long-horizon embodied and web-agent benchmarks show that MemPrism consistently improves the
    
[^312]: NxN E-valuation：基于共形CRT零假设的假设认证方法

    NxN E-valuation: Hypothesis Certification via a Conformal CRT Null

    [https://arxiv.org/abs/2608.06621](https://arxiv.org/abs/2608.06621)

    提出 NxN E-valuation，一种基于 e 值与共形 CRT 零假设的假设认证算法，通过利用大型训练集让样本互为零假设，无需为每个假设专门构建零假设即可完成验证，尤其适合应对 LLM 探索系统中的幻觉问题。

    

    我们提出了 NxN E-valuation，一种便捷的、基于 e 值的假设认证算法。只要拥有足够大的数据集，该算法无需构建任何针对特定情况的认证程序（例如构造专门的零假设）即可验证假设。该方法特别适用于基于大语言模型（LLM）的探索系统：LLM 非常擅长提出假设，但深受幻觉问题的困扰；这种幻觉使我们无法直接采信 LLM 的输出，而现有的补救措施各有不足。最常见的解决方案包括让 LLM 自我验证或自我纠正（循环验证），以及保留集测试（held-out testing，其中错误假设仍可能通过虚假关联而蒙混过关），引言中还详细介绍了其他补救措施。为解决这一问题，NxN E-valuation 利用天然存在的大型训练集，让不同样本互为对方的零假设。这种设计

    arXiv:2608.06621v2 Announce Type: replace  Abstract: We propose NxN E-valuation, a handy, e-value-based hypothesis-certification algorithm that lets a hypothesis be verified without building any case-specific certification procedure---such as constructing a dedicated null hypothesis---as long as a large enough dataset is available. The method is especially suited to LLM-based exploration systems, where LLMs are remarkably good at proposing hypotheses but suffer badly from hallucination; this hallucination prevents us from harvesting LLM outputs directly, and existing remedies each fall short. The most common solutions include letting the LLM verify or correct itself circular verification and held-out testing (where false hypotheses can still pass via spurious correlations), among other remedies detailed in the introduction. To resolve this, NxN E-valuation exploits the naturally existing large training set and lets different samples serve as null hypotheses for one another. This design
    
[^313]: AI引导的深度视觉蛋白质组学揭示三阴性乳腺癌中与复发相关的形态-分子特征

    AI-guided deep visual proteomics reveals recurrence-associated morpho-molecular signatures within triple-negative breast cancers

    [https://arxiv.org/abs/2608.03145](https://arxiv.org/abs/2608.03145)

    该研究开发了一种结果导向的AI引导深度视觉蛋白质组学框架，利用复发风险热图指导三阴性乳腺癌的组织采样与质谱空间蛋白质组学分析，揭示了高风险区域富集有丝分裂程序、低风险区域富集免疫与抗原呈递程序的肿瘤内分子异质性。

    

    结果导向的人工智能可以在数字病理学中定位与预后相关的区域，但直接表征这些空间预测背后的分子基础仍然具有挑战性。在此，我们开发了一个结果导向的、AI引导的深度视觉蛋白质组学框架，该框架利用复发风险热图来指导三阴性乳腺癌（TNBC）的实体组织采样和基于质谱的空间蛋白质组学分析。在156名患者中，高风险与低风险区域共存于肿瘤区内，并表现出不同的细胞核与结构特征，表明存在超越广义组织身份的肿瘤内异质性。在这些图谱的指导下，我们从两例复发病例中物理分离并分析了46个AI定义的高风险和低风险肿瘤区域。高风险区域富集有丝分裂相关程序，而低风险区域则显示免疫和抗原呈递相关程序。我们利用在两类区域间持续差异表达的蛋白质……（原文摘要在此处截断）

    arXiv:2608.03145v2 Announce Type: replace  Abstract: Outcome informed AI can localize prognostically relevant regions in digital pathology, but directly characterizing the molecular basis of these spatial predictions remains challenging. Here, we developed an outcome informed, AI-guided deep visual proteomics framework that uses recurrence-risk heatmaps to guide physical tissue sampling and mass spectrometry-based spatial proteomics in triple-negative breast cancer (TNBC). In 156 patients, high- and low-risk regions coexisted within the tumor compartment and showed distinct nuclear and architectural features, indicating intratumoral variation beyond broad tissue identity. Guided by these maps, we physically isolated and profiled 46 AI-defined high- and low-risk tumor regions from two recurrent cases. High-risk regions were enriched for mitotic programs, whereas low-risk regions showed immune- and antigen-presentation-associated programs. We used proteins that consistently differed betw
    
[^314]: 硬约束，平滑梯度：通过可微投影学习可行的库存策略

    Hard Constraints, Smooth Gradients: Learning Feasible Inventory Policies via Differentiable Projection

    [https://arxiv.org/abs/2608.02343](https://arxiv.org/abs/2608.02343)

    该论文在强化学习策略中嵌入可微优化模块（神经网络提出连续动作目标、二次规划投影保证可行性、对偶信息整数映射恢复整数性），实现了端到端训练并严格满足硬约束的库存决策策略。

    

    许多运营问题是具有组合动作空间和相互依赖约束的受约束序贯决策过程。混合整数线性规划（MILP）可以灵活处理这类约束，但在随机环境中扩展性较差。深度强化学习（DRL）有望提供可扩展的决策规则，但现有方法要么对约束进行惩罚而非强制执行，要么依赖的可行性机制在约束相互作用时便会失效。我们通过在策略内部嵌入一个可微优化模块来弥合这一差距：神经网络提出连续动作目标，二次规划将其投影到松弛可行集上，而基于对偶信息的整数映射在保持可行性的同时恢复整数性。在给定可微模拟器的情况下，该策略利用路径梯度从采样轨迹中进行端到端训练，同时能够以与MILP相似的灵活性处理硬约束。

    arXiv:2608.02343v2 Announce Type: replace  Abstract: Many operational problems are constrained sequential decision processes with combinatorial action spaces and interdependent constraints. Mixed-integer linear programs (MILPs) handle such constraints flexibly but scale poorly in stochastic environments. Deep reinforcement learning (DRL) promises scalable decision rules, but existing methods either penalize constraints rather than enforce them, or rely on feasibility mechanisms that break down once constraints interact. We bridge this gap by embedding a differentiable optimization module inside the policy: a neural network proposes continuous action targets, a quadratic program projects them onto the relaxed feasible set, and a dual-informed integer mapping restores integrality while preserving feasibility. Given a differentiable simulator, the policy trains end-to-end from sampled trajectories using pathwise gradients, while handling hard constraints with similar flexibility to MILPs.
    
[^315]: 追踪级联效应：面向科学智能体幻觉的拓扑感知评估框架

    Tracing the Cascade: A Topology-Aware Evaluation Framework for Scientific Agent Hallucinations

    [https://arxiv.org/abs/2608.00711](https://arxiv.org/abs/2608.00711)

    本文提出SCHEMA——首个基于证据、拓扑感知的科学智能体幻觉评估框架，通过自动构建科学概念图并生成论断验证、多跳推理、开放式解释与实验代码生成等多类图任务，捕捉幻觉在高度互联的科学知识网络中的级联传播效应。

    

    大语言模型（LLM）智能体正日益被部署于科学研究领域，在这些场景中，可靠性至关重要，且底层知识高度互联。在这样的环境下，幻觉的危害尤为严重：关于某个基础概念的一条错误论断可能通过多步推理不断传播，从而破坏整条推理轨迹。现有的幻觉基准测试大多停留在表层，孤立地看待各个事实，并依赖忽略这种拓扑结构的统一准确率指标。我们用 SCHEMA 来填补这一空白——这是首个基于证据、拓扑感知的科学智能体幻觉评估框架。SCHEMA 自动从基准种子和文献证据构建科学概念图，合成涵盖论断验证、多跳推理、开放式解释和实验代码生成等多种基于图的任务，并以此对智能体进行评估。

    arXiv:2608.00711v2 Announce Type: replace  Abstract: Large language model (LLM) agents are increasingly deployed in scientific research, where reliability is critical and the underlying knowledge is densely interconnected. In such settings, hallucinations are particularly damaging: a single erroneous claim on a foundational concept can propagate through multi-step reasoning and corrupt entire trajectories. Existing hallucination benchmarks largely operate at the surface level, treating facts in isolation and relying on uniform accuracy metrics that ignore this topological structure. We address this gap with SCHEMA, the first evidence-grounded, topology-aware evaluation framework for hallucinations in scientific agents. SCHEMA automatically constructs scientific concept graphs from benchmark seeds and literature evidence, synthesizes graph-grounded tasks spanning claim verification, multi-hop reasoning, open-ended explanation, and experimental code generation, and evaluates agents with 
    
[^316]: TextCloak：通过强化学习驱动的不可学习样本保护数据免受未经授权的LLM微调

    TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples

    [https://arxiv.org/abs/2607.28862](https://arxiv.org/abs/2607.28862)

    TextCloak提出了一种基于强化学习的数据合成框架，通过生成既能保持语义忠实性和语言自然性又能降低模型学习效果的不可学习样本，来保护文本数据免受未经授权的大语言模型微调。

    

    大型语言模型（LLM）的发展在众多语言任务中取得了显著进展，同时也引发了人们对未经授权的在线数据利用和隐私泄露日益增长的担忧。不可学习样本（UEs）作为一种有前景的防御手段应运而生，其通过在数据中注入精心设计的扰动，使得在这些数据上训练的模型效用下降。然而，现有的文本保护方法主要是针对分类任务和判别式语言模型设计的，这限制了它们在LLM中的广泛适用性（例如指令遵循或推理任务）。在这项工作中，我们提出了TextCloak，这是一个由强化学习驱动的数据合成框架，用于保护文本数据免受未经授权的LLM微调。TextCloak采用一种不可学习样本生成策略，将干净文本转换为不可学习样本，同时保持语义忠实性和语言自然性。为了优化该策略，w...

    arXiv:2607.28862v2 Announce Type: replace-cross  Abstract: The development of Large Language Models (LLMs) has led to significant advances across a wide range of language tasks, while simultaneously raising growing concerns about unauthorized online data exploitation and privacy leakage. Unlearnable examples (UEs) emerge as a promising defense by injecting crafted perturbations into data such that models trained on them exhibit degraded utility. However, existing methods for text protection are primarily designed for classification tasks and discriminative language models, which limits their broad applicability (e.g., instruction following or reasoning) in LLMs. In this work, we propose **TextCloak**, an RL-driven data synthesis framework that safeguards textual data against unauthorized LLM fine-tuning. TextCloak employs a *UE generative policy* that transforms clean text into unlearnable examples while preserving semantic fidelity and linguistic naturalness. To optimize the policy, w
    
[^317]: 从确定性深度学习到生成式深度学习：基于稀疏观测的城市空气质量重建

    From Deterministic to Generative Deep Learning for Urban Air Quality Reconstruction from Sparse Observations

    [https://arxiv.org/abs/2607.25687](https://arxiv.org/abs/2607.25687)

    本文提出了一种基于扩散模型的生成式深度学习框架，用于从稀疏观测中重建巴黎多种空气污染物（NO2、O3、PM2.5、PM10）的浓度场，并在与确定性深度学习模型的对比中展现出在真实观测数据上生成逼真空间分布的能力。

    

    空气污染的全场重建对于评估污染暴露和支持公共卫生决策至关重要。然而，污染物之间复杂的相互作用、难以预测的天气模式以及有限的监测站覆盖范围，使这项任务变得极具挑战性。我们应用深度学习技术，从四种关键污染物（NO2、O3、PM2.5和PM10）的稀疏观测中提供快速而准确的重建。模型在全模拟数据上进行训练，并在巴黎市9至28个监测站收集的真实世界观测数据上进行评估。我们引入了一种基于扩散模型的多污染物重建生成式框架，并将其性能与确定性深度学习模型进行基准对比。尽管观测数据存在噪声且空间变异性强，这些模型在模拟验证数据上仍实现了较高的结构相似性，并在真实世界数据上生成了逼真的空间分布模式。

    arXiv:2607.25687v2 Announce Type: replace-cross  Abstract: Full-field reconstruction of air pollution is essential for evaluating pollution exposure and supporting public health decision-making. However, the complex interactions among pollutants, hard-to-predict weather patterns, and limited monitoring station coverage make this a complex task. We apply deep learning techniques to provide fast and accurate reconstructions from sparse observations of four key pollutants: NO2, O3, PM2.5 and PM10. Models are trained on full-field simulation data and evaluated on real-world observations collected from 9 to 28 monitoring stations in the city of Paris. We introduce a diffusion-based generative framework for multi-pollutant reconstruction and benchmark its performance against deterministic deep learning models. Despite noisy observations and strong spatial variability, the models achieve high structural similarity on simulated validation data and produce realistic spatial patterns on real-wor
    
[^318]: ScalableRAG：以最小摄取成本实现精确语料库操作的高质量RAG

    ScalableRAG: High-Quality RAG with Exact Corpus Operations at Minimal Ingestion Cost

    [https://arxiv.org/abs/2607.25135](https://arxiv.org/abs/2607.25135)

    ScalableRAG通过持久化的类型化集合工作区，让智能体在查询时利用集合代数执行精确的语料库级操作，无需任何预处理甚至向量索引，就能在聚合问答任务上大幅超越现有RAG系统。

    

    RAG系统面临能力-成本差距：轻量级检索器能够定位证据，但不提供执行精确语料库级操作（计数、交集、聚合）所需的操作数，而知识摄取系统只有通过昂贵的摄取过程才能获得这些能力。ScalableRAG通过一个持久化的类型化集合工作区弥合了这一差距：智能体在查询时构建命名的文档集合和值集合，并通过集合代数、类型化抽取、聚合和诊断来组合它们。零摄取（Zero-Ingestion）ScalableRAG无需任何预处理，甚至不需要向量索引。它在四个传统问答基准上与领先的基线方法持平（74.75对74.64），而在聚合型Hotels-50基准上击败了所有被评估的系统，得分71.74：比领先的基线（SRAG）高出36.23分，比A-RAG高出40.22分。移除工作区会使Hotels-50的成绩降至38.4。在Hotels-350（SRAG本身报告的划分）上，ScalableRAG领先SRAG 9.3分。有限摄取（Limited-Ingestion）ScalableRAG……

    arXiv:2607.25135v2 Announce Type: replace  Abstract: RAG systems face a capability-cost gap: lightweight retrievers locate evidence but expose no operands for exact corpus-level operations (counts, intersections, aggregations), while knowledge ingestion systems obtain them only through expensive ingestion. ScalableRAG closes this gap with a persistent typed-set workspace: an agent builds named document and value sets at query time and composes them through set algebra, typed extraction, aggregation, and diagnostics.   Zero-Ingestion ScalableRAG needs no preprocessing, not even a vector index. It matches the leading baseline on four conventional QA benchmarks (74.75 versus 74.64) and on aggregative Hotels-50 beats every evaluated system, scoring 71.74: 36.23 above the leading baseline (SRAG), and 40.22 points above A-RAG. Removing the workspace drops Hotels-50 to 38.4. On Hotels-350, the partition SRAG itself reports on, ScalableRAG leads SRAG by 9.3 points. Limited-Ingestion ScalableRA
    
[^319]: 用于结构和语义一致的仿真到真实转换的小波相位扩散

    Wavelet Phase Diffusion for Structurally and Semantically Consistent Sim-to-Real Translation

    [https://arxiv.org/abs/2607.21628](https://arxiv.org/abs/2607.21628)

    提出小波相位扩散方法，在双树复小波包变换域中进行相位保持扩散，克服了傅里叶域全局频谱耦合引起的空间伪影问题，无需昂贵控制模块或成对数据流水线即可实现结构与语义一致的仿真到真实图像转换。

    

    仿真到真实转换必须在合成域与真实域之间弥合外观差异，同时保持结构和语义的一致性。基于条件的方法实现了空间对齐，但引入了计算成本高昂的控制模块；成对数据方法能够实现真实感，但依赖复杂的合成流水线，往往会改变场景的几何结构和语义；免训练编辑方法避免了上述两个限制，但缺乏学习到的外观先验，限制了其感知质量。最近提出的相位保持扩散提供了一种有前景的替代方案，但基于傅里叶域的公式受到全局频谱耦合的约束，这种耦合会引起振铃和边界泄漏等空间伪影，从而降低结构和语义的一致性。我们提出小波相位扩散，通过两个组件来解决这一问题。首先，我们在双树复小波包变换域中进行操作……

    arXiv:2607.21628v3 Announce Type: replace  Abstract: Simulation-to-reality translation must bridge the appearance gap between synthetic and real domains while preserving structural and semantic consistency. Conditioning-based methods achieve spatial alignment but introduce computationally expensive control modules. Paired-data methods achieve realism but rely on complex synthesis pipelines, often altering scene geometry and semantics. Training-free editing methods avoid both constraints but lack a learned appearance prior, limiting their perceptual quality. Recently proposed phase-preserving diffusion presents a promising alternative, but Fourier-domain formulations are constrained by global spectral coupling. This coupling induces spatial artifacts such as ringing and boundary leakage, thereby degrading structural and semantic consistency. We introduce Wavelet Phase Diffusion, which addresses this through two components. First, we operate in the Dual-Tree Complex Wavelet Packet Transf
    
[^320]: LISA：用于高效长上下文推理的线性索引稀疏注意力

    LISA: Linear-Indexed Sparse Attention for Efficient Long-Context Reasoning

    [https://arxiv.org/abs/2607.19358](https://arxiv.org/abs/2607.19358)

    LISA是一种无需完整预训练的即插即用稀疏注意力机制，通过融合线性注意力分支与动态token选择索引器，在保留长程记忆的同时大幅降低长上下文推理的计算成本。

    

    像DeepSeek-R1这样的长思维链（CoT）推理模型在测试时扩展机制下会不断延长其推理链，然而标准的自注意力机制在上下文长度上会产生二次方复杂度的开销，使得长CoT在生产环境中的部署成本高昂。因此，我们提出了LISA（线性索引稀疏注意力），这是一种即插即用的自注意力替代方案，无需完整的预训练。LISA将用于长程记忆的线性注意力分支与一个Lightning Indexer相结合，后者动态地为稀疏自注意力分支选择最相关的token。两个分支通过一个可学习的门控机制进行融合，使模型在保留广泛上下文的同时，将计算集中于信息丰富的token上。我们分两个阶段训练LISA。在第一阶段，将线性注意力分支与滑动窗口注意力集成，并从一个冻结教师模型的全注意力分布中蒸馏学习。在第二阶段，Lightning Indexer取代静态窗口并（摘要在此处截断）

    arXiv:2607.19358v2 Announce Type: replace  Abstract: Long chain-of-thought (CoT) reasoning models such as DeepSeek-R1 keep lengthening their inference chains under test-time scaling, yet standard self-attention incurs quadratic cost in context length, making long-CoT deployment costly in production. We therefore introduce LISA (Linear-Indexed Sparse Attention), a plug-and-play replacement for self-attention that requires no full pretraining. LISA combines a linear-attention branch for long-range memory with a Lightning Indexer that dynamically selects the most relevant tokens for a sparse self-attention branch. The two branches are fused through a learned gate, so the model retains broad context while focusing computation on informative tokens.   We train LISA in two stages. In Stage 1, a linear-attention branch is integrated with sliding-window attention and distilled from a frozen teacher's full-attention distribution. In Stage 2, the Lightning Indexer replaces the static window and 
    
[^321]: 经验记忆图：从智能体轨迹中提取可复用的纠错知识

    Experience Memory Graph: Extracting Reusable Correction Knowledge from Agent Trajectories

    [https://arxiv.org/abs/2607.13884](https://arxiv.org/abs/2607.13884)

    该论文提出经验记忆图（EMG）框架，将智能体的失败恢复重新表述为图匹配问题，从而从历史轨迹中提取可靠、可迁移且可复用的纠错知识。

    

    大语言模型（LLM）智能体通过生成由状态、动作和观察组成的序列轨迹，在自主决策方面展现出卓越的能力。然而，在复杂的长期任务中，这些智能体经常受到累积误差的影响，难以从失败中恢复。尽管经验记忆技术已有进展，但从历史轨迹中提取可靠且可迁移的纠错知识仍然具有挑战性。现有方法，无论是基于测试时反思还是离线记忆构建，大多依赖LLM来诊断失败并制定纠正策略，这可能产生脆弱的或局限于特定任务的见解，难以泛化。为了应对这一挑战，我们提出了经验记忆图（EMG），这是一个将智能体失败恢复重新表述为图匹配问题的框架。在训练集上进行离线构建时，我们将失败的探索轨迹……（摘要原文在此处截断）

    arXiv:2607.13884v3 Announce Type: replace  Abstract: Large Language Model (LLM) agents have shown remarkable capabilities in autonomous decision-making by generating sequential trajectories of states, actions, and observations. However, in complex, long-horizon tasks, these agents frequently suffer from compounding errors and struggle to recover from failures. Despite advances in experience memory, extracting reliable and transferable correction knowledge from historical trajectories remains challenging. Existing methods, whether based on test-time reflection or offline memory construction, largely rely on LLMs to diagnose failures and formulate corrective strategies, which can produce brittle or task-specific insights that are difficult to generalize. To address this challenge, we propose Experience Memory Graph (EMG), a framework that reformulates agent failure recovery as a graph matching problem. During offline construction on the training set, we convert both failed exploration tr
    
[^322]: 从语言到导航目标：基于RGB-D感知的移动机器人语义导航的视觉-语言方法

    From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception

    [https://arxiv.org/abs/2607.13624](https://arxiv.org/abs/2607.13624)

    该论文提出了一个由模块化ROS 2组件构成的语言驱动导航框架，利用RGB-D感知识别自然语言指令中目标物体的位置，使移动机器人能够将非专业用户的自然语言请求直接转化为可执行的自主导航目标。

    

    自然语言交互为非专业用户提供了一种与机器人平台进行直观交流的方式。然而，将用户请求转化为可执行的导航动作仍然是一项具有挑战性的任务，需要整合语言理解、环境感知和自主导航。本工作提出了一个语言驱动的导航框架，使移动机器人能够理解用户以自然语言表达的移动到目的地的请求，并自主导航到达该目的地。该框架由模块化的ROS 2组件构成，这些组件协同工作，将自然语言指令转化为导航动作。给定一个引用环境中目标的自然语言请求（例如“去邮箱”），系统会识别所引用的物体，利用RGB-D数据估计其位置，并生成导航目标，随后通过ROS 2 Nav2导航系统执行该目标。

    arXiv:2607.13624v2 Announce Type: replace-cross  Abstract: Natural language interaction provides an intuitive way for non-expert users to communicate with robotic platforms. However, transforming user requests into executable navigation actions remains a challenging task, requiring the integration of language understanding, environment perception, and autonomous navigation. This work presents a language-driven navigation framework that enables mobile robots to interpret user requests in natural language to move the robot to a destination and autonomously navigate towards it.   The framework is composed of modular ROS 2 components that cooperate to transform natural language instructions into navigation actions. Given a natural language request referring to a target in the environment (e.g., "go to the mail box"), the system identifies the referenced object, estimates its position using RGB-D data, and generates a navigation goal, which is then executed through the ROS 2 Nav2 navigation
    
[^323]: 置信域策略蒸馏

    Trust Region Policy Distillation

    [https://arxiv.org/abs/2607.04751](https://arxiv.org/abs/2607.04751)

    TOP-D通过动态构建近端教师模型，将不稳定、高方差的在线策略蒸馏转化为稳定且高样本效率的训练范式，在理论上保证了梯度方差控制与全局收敛性，且不引入任何额外计算开销。

    

    宏大的目标很难一步实现，将其分解为小步骤是更明智的做法。我们提出了置信域策略蒸馏（TOP-D），该方法通过动态构建一个近端教师模型，将出了名的不稳定、高方差的在线策略蒸馏（OPD）转变为一种稳定的训练范式。在理论上，我们建立了一个严格的框架，证明TOP-D天然地能够控制梯度方差。通过提供正式的全局收敛分析以及单调改进界，我们在数学上形式化了整体训练动态的可靠性与稳定性。在实证方面，TOP-D在数学推理任务上显著提升了训练稳定性、样本效率和最终性能。更重要的是，TOP-D没有引入任何额外的计算开销，使其成为成熟的OPD范式的有前景的替代方案。

    arXiv:2607.04751v2 Announce Type: replace-cross  Abstract: Big goals are hard to achieve all at once; breaking them into small steps is wiser. We present Trust Region Policy Distillation (TOP-D), which transforms the notoriously unstable, high-variance On-Policy Distillation (OPD) into a stable training paradigm by dynamically constructing a proximal teacher. Theoretically, we establish a rigorous framework demonstrating that TOP-D inherently controls gradient variance. By providing a formal global convergence analysis alongside a monotonic improvement bound, we mathematically formalize the reliability and stability of the overall training dynamics. Empirically, TOP-D dramatically enhances training stability, sample efficiency, and final performance on mathematical reasoning tasks. More importantly, TOP-D introduces zero additional computational overhead, positioning itself as a promising alternative to the well-established OPD paradigm.
    
[^324]: TokenMinds：面向大型推荐系统用户理解的预训练用户Token与嵌入

    TokenMinds: Pretrained User Tokens and Embeddings for User Understanding in Large Recommender Systems

    [https://arxiv.org/abs/2606.25147](https://arxiv.org/abs/2606.25147)

    TokenMinds基于预训练LLM改造的编码器-解码器架构并采用双输出设计，将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模，同时产出稠密用户嵌入与离散用户Token，突破了固定维度稠密嵌入的表征限制并捕捉用户行为的序列模式。

    

    工业推荐系统中的用户建模通常产生稠密嵌入，但由于固定维度向量的限制，其表征能力受到约束。作为离散用户表示的一种新兴替代方案，由LLM生成的自然语言用户画像往往捕获的是语言语义，而非用户行为中的序列模式。同时，语义ID（SID）虽然对物品表示很有效，但现有的离散用户表示大多仍局限于量化已池化的用户向量、对历史观看行为进行聚类，或仅为候选检索预测粗粒度的主题聚类。我们提出TokenMinds——一个将PLUM框架从基于SID的物品检索扩展到生成式序列用户建模的工业系统。TokenMinds构建于由预训练LLM改造而来的编码器-解码器架构之上，引入了双输出设计：编码器输出一种稠密用户嵌入，用于捕获……（原文摘要在此处截断）

    arXiv:2606.25147v2 Announce Type: replace-cross  Abstract: User modeling in industrial recommender systems typically produces dense embeddings, which suffer from representational constraints due to fixed-dimensional vectors. As an emerging alternative for discrete user representation, LLM-generated natural-language user profiles often capture language semantics rather than sequential patterns in user behaviors. Meanwhile, Semantic ID (SID) is effective for item representation, yet existing discrete user representations remain largely confined to quantizing already-pooled user vectors, clustering historical watches, or predicting coarse topic clusters solely for candidate retrieval. We propose TokenMinds, an industrial system that extends the PLUM framework from SID-based item retrieval to generative sequential user modeling. Built on an encoder-decoder architecture adapted from pretrained LLMs, TokenMinds introduces a dual-output design: the encoder outputs a dense user embedding captu
    
[^325]: HOLogicBench：评估大语言模型中的高阶逻辑推理能力

    HOLogicBench: Evaluating Higher-Order Logical Reasoning in LLMs

    [https://arxiv.org/abs/2606.23238](https://arxiv.org/abs/2606.23238)

    本文提出HOLogicBench基准，包含1,379个经Isabelle/HOL形式化证明验证的实例，系统性评估大语言模型在五种高阶逻辑推理操作上的能力，发现现有模型平均准确率仅为53.86%，且推理路径与有效推导对齐不足、易受最小反事实扰动的影响。

    

    现有的大语言模型逻辑推理基准主要聚焦于在固定谓词和规则下的对象级推理，而对规则和策略的高阶推理研究仍然不足。我们提出了HOLogicBench，一个包含1,379个实例的基准，系统性评估五种代表性的高阶推理操作，并采用Isabelle/HOL形式化表示和经证明验证的答案。除了答案准确率之外，HOLogicBench还为每个实例提供了由求解器验证的参考推理路径，从而能够对模型生成的推理与有效推导进行细粒度评估。在16个模型配置中，平均准确率为53.86%，最佳配置达到65.99%。进一步分析表明，大语言模型与有效推导的对齐程度有限，面对最小的反事实扰动时依然脆弱，并且在不同高阶推理类型上呈现出不同的失败模式。

    arXiv:2606.23238v3 Announce Type: replace  Abstract: Existing LLM logical-reasoning benchmarks largely focus on object-level inference under fixed predicates and rules, leaving higher-order reasoning over rules and policies underexplored. We introduce HOLogicBench, a 1,379-instance benchmark that systematically evaluates five representative higher-order reasoning operations, with Isabelle/HOL formalizations and proof-verified answers. Beyond answer accuracy, HOLogicBench provides solver-verified reference reasoning paths for each instance, enabling fine-grained evaluation of model-generated reasoning against valid derivations. Across 16 model configurations, average accuracy is 53.86\%, with the best configuration reaching 65.99\%. Further analyses reveal that LLMs exhibit limited alignment with valid derivations, remain vulnerable to minimal counterfactual perturbations, and display distinct failure patterns across higher-order reasoning types.
    
[^326]: LineageMark：用于模型派生链贡献追踪的多用户白盒水印框架

    LineageMark: Multi-user White-box Watermarking for Contribution Tracing in Model Derivation Chains

    [https://arxiv.org/abs/2606.17123](https://arxiv.org/abs/2606.17123)

    提出了LineageMark——一个面向模型派生链的多用户白盒水印框架，通过基于投影的参数编码方式，在多阶段模型派生和增量更新场景下实现多方历史贡献的追踪与验证。

    

    在开放的大语言模型生态系统中，模型经常被跨多个领域和应用进行适配，形成多阶段的派生链。因此，追踪和验证历史贡献对于模型溯源和知识产权保护至关重要。然而，现有的水印方法主要针对单用户、一次性嵌入设计，在反复的模型派生和增量更新下往往失效。为解决这一问题，我们提出了LineageMark，一个面向模型派生链的多用户白盒水印框架。该框架使用基于投影的方法将水印编码在模型参数中。首先选择稳定的载体以降低对模型变化的敏感性，随后将每个水印位表示为这些载体上的投影统计量。额外的水印插入仅在投影空间中引入有界的扰动，并通过边距约束……

    arXiv:2606.17123v2 Announce Type: replace-cross  Abstract: In open large language model (LLM) ecosystems, models are frequently adapted across multiple domains and applications, forming multi-stage derivation chains. Consequently, tracking and verifying historical contributions is essential for model provenance and intellectual property protection. However, existing watermarking methods are mainly designed for single-user, one-time embeddings, often fail under repeated model derivation and incremental updates. To address this problem, we propose LineageMark, a multi-user white-box watermarking framework for model derivation chains. The framework encodes watermarks in model parameters using a projection-based approach. Stable carriers are first selected to reduce sensitivity to model changes, each watermark bit is then represented as a projection statistic over these carriers. Additional watermark insertions introduce only bounded perturbations in the projection space, and margin constr
    
[^327]: DifFRACT：用于电路追踪的扩散特征重建与归因

    DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing

    [https://arxiv.org/abs/2606.15796](https://arxiv.org/abs/2606.15796)

    本文提出DifFRACT方法，首次将基于转码器的电路追踪技术扩展到多模态扩散变换器，通过训练时间步条件化的转码器替换FLUX.1中的MLP子层，实现了对图像生成模型内部可解释特征的精确归因和电路分析。

    

    虽然基于转码器的电路追踪最近使对大语言模型进行详细因果分析成为可能，但语义信息如何在去噪步骤之间传播，以及文本和图像表示如何在多模态扩散变换器（MM-DiT）中相互作用以进行图像生成，目前仍不清楚。注意力图只能提供词元相互作用的有限视角，而稀疏自编码器虽然能找到可解释的特征，却无法揭示非线性MLP层如何变换和组合这些特征。在这项工作中，我们将基于转码器的电路追踪方法扩展到了多模态扩散变换器。我们训练了以时间步为条件的转码器，忠实地逼近FLUX.1[schnell]双流块中MLP子层的输入输出行为。通过用转码器替换MLP并对其余计算进行线性化，我们获得了精确的特征到特征归因，并在可解释特征之上构建了紧凑的归因图。

    arXiv:2606.15796v2 Announce Type: replace-cross  Abstract: While transcoder-based circuit tracing has recently enabled detailed causal analyses of large language models, it remains unclear how semantic information propagates across denoising steps and how text and image representations interact within multimodal diffusion transformers (MM-DiT) for image generation. Attention maps expose only a limited view of token interactions, and sparse autoencoders find interpretable features but do not reveal how nonlinear MLP layers transform and compose them. In this work, we extend transcoder-based circuit tracing to multimodal diffusion transformers. We train timestep-conditioned transcoders that faithfully approximate the input-output behavior of the MLP sublayers in the double-stream blocks of FLUX.1[schnell]. By replacing MLPs with transcoders and linearizing the remaining computation, we obtain exact feature-to-feature attribution and build compact attribution graphs over interpretable fea
    
[^328]: 面向动态声源的时空音频语言建模

    Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources

    [https://arxiv.org/abs/2606.14141](https://arxiv.org/abs/2606.14141)

    该论文提出了基于FOA渲染的时空音频问答数据集ST-AudioQA及配套的ST-Audio编码器和ST-AudioLM模型，联合建模声源语义与随时间变化的声源轨迹，使大语言模型能够回答关于声源身份、位置和运动的时空音频问答问题。

    

    声音事件具有语义身份、位置和运动轨迹，然而现有的音频-语言模型通常将音频片段编码为全局事件内容；相反，声音事件定位模型虽然能够随时间追踪声源方向，但其语义覆盖范围有限，难以支持语言推理。为弥合这一差距，我们提出了 ST-AudioQA，一个基于静态和移动声源的一阶高保真立体声（FOA）渲染构建的时空音频问答数据集与基准。每个场景都提供声源身份、活动状态、方向、距离和运动元数据，从而支持密集的轨迹监督，以及关于“什么在发声、它在哪里、它如何移动以及声源之间如何关联”的问答。我们进一步提出了 ST-Audio Encoder，联合学习事件语义和随时间变化的声源轨迹；以及 ST-AudioLM，将这些音频 token 输入大语言模型以进行时空音频问答。实验表明，这种表示改进了平衡……

    arXiv:2606.14141v2 Announce Type: replace-cross  Abstract: Sound events have semantic identities, locations, and trajectories, yet existing audio-language models typically encode clips as global event content. Conversely, sound event localization models track source directions over time but offer limited semantic coverage for language reasoning. To bridge this gap, we introduce ST-AudioQA, a spatio-temporal audio QA dataset and benchmark built from first-order ambisonic (FOA) renderings of static and moving sound sources. Each scene provides source identity, activity, direction, distance, and motion metadata, enabling dense trajectory supervision and QA about what is sounding, where it is, how it moves, and how sources relate. We further introduce ST-Audio Encoder, which jointly learns event semantics and time-resolved source trajectories, and ST-AudioLM, which exposes these audio tokens to an LLM for spatio-temporal audio QA. Experiments show that this representation improves the bala
    
[^329]: 是什么使视频世界模型的潜变量具有动作相关性：预测优于重建

    What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction

    [https://arxiv.org/abs/2606.07687](https://arxiv.org/abs/2606.07687)

    该研究通过统一的探针评估发现，视频世界模型潜空间中与动作相关的结构主要由时序视频预训练（潜变量预测）驱动而非像素重建保真度，视频预训练的自监督编码器在视觉保真度与动作预测之间实现了最佳的帕累托权衡。

    

    视频世界模型（video world models）越来越多地被用于提供预测性的视觉表征，然而目前尚不清楚哪些预训练信号能够在其潜空间中诱导出与动作相关的结构。我们通过对多种编码器家族进行统一的基于探针的评估来研究这一问题，这些编码器家族包括仅图像自监督、带与不带潜变量预测的视频预训练、基于重建的自编码器、扩散模型以及强制捷径的动力学模型。使用共同的逆动力学探测目标，我们发现与动作相关的结构主要由时序视频预训练驱动，而非像素重建保真度：具有较强像素解码质量的模型可能表现出接近于零的动作可恢复性，而经过视频预训练的自监督编码器始终在视觉保真度与动作预测之间取得最佳的帕累托权衡。对 V-JEPA 与 VideoMAE 的进一步比较表明，大部分……

    arXiv:2606.07687v2 Announce Type: replace-cross  Abstract: Video world models are increasingly used to provide predictive visual representations, yet it remains unclear which pretraining signals induce action-relevant structure in their latent spaces. We study this question through a unified probe-based evaluation across diverse encoder families, including image-only self-supervision, video pretraining with and without latent prediction, reconstruction-based autoencoders, diffusion models, and shortcut-forcing dynamics models. Using a common inverse-dynamics probing objective, we find that action-relevant structure is driven primarily by temporal video pretraining rather than pixel reconstruction fidelity: models with strong pixel decoding quality can exhibit near-zero action recoverability, while video-pretrained self-supervised encoders consistently achieve the best Pareto trade-off between visual fidelity and action prediction. Comparing V-JEPA and VideoMAE further shows that most g
    
[^330]: 大语言模型能够泄露训练数据，但它们想这样做吗？评估大语言模型的记忆倾向性

    LLMs Can Leak Training Data But Do They Want To? Evaluating Memorization Propensity in LLMs

    [https://arxiv.org/abs/2606.06286](https://arxiv.org/abs/2606.06286)

    本文提出PropMe框架和SimpleTrace工具，首次将大语言模型“能否”泄露训练数据与“是否会主动”泄露训练数据区分开来，发现自然句子提示下的主动泄露远少于前缀攻击，但无提示或常见词提示时的泄露程度与前缀攻击相当。

    

    大语言模型能够复现训练数据，但现有的记忆评估大多衡量的是模型能否被诱导复现训练数据，而非模型在没有对抗性提示的情况下是否会主动这样做。我们提出了PropMe框架，将能力评估（前缀攻击）与非对抗性的倾向性评估相结合，并引入一个倾向性指标来关联两者。我们使用SimpleTrace——一个开源的OLMoTrace片段匹配批量实现——将每一次生成结果与模型的完整索引训练语料库进行比对。我们在Common Pile、Dynaword和Dolma3上评估了Comma、其持续预训练的后续模型DFM Decoder以及OLMo3 32B，采用四种非对抗性设置，范围从无提示到来自未见过的同领域来源的句子。我们的结果表明，自然句子提示所复现的训练文本远少于前缀攻击。然而，在无提示或仅使用单个常见词的情况下，模型生成的复现量与前缀攻击大致相当。

    arXiv:2606.06286v2 Announce Type: replace-cross  Abstract: Large language models can reproduce training data, but memorization evaluations mostly measure whether models can be made to do so, not whether they do so without adversarial prompting. We introduce PropMe, a framework that pairs a capability evaluation (prefix attacks) with non-adversarial propensity evaluation, and a propensity metric to relate the two. We check every generation against the model's full indexed training corpus with SimpleTrace, an open-source batch implementation of OLMoTrace span matching. We evaluate Comma, its continually pre-trained descendant DFM Decoder, and OLMo3 32B on Common Pile, Dynaword, and Dolma3, with four non-adversarial settings ranging from no prompt to sentences from unseen in-domain sources. Our results show that natural sentence prompts reproduce far less training text than prefix attacks. Generation with no prompt or a single common word, however, reproduces about as much as prefix attac
    
[^331]: DRIFT：一种用于视觉-语言模型连续输出解码的残差流适配器

    DRIFT: A Residual Flow Adapter for Decoding Continuous Outputs in Vision-Language Models

    [https://arxiv.org/abs/2606.05758](https://arxiv.org/abs/2606.05758)

    提出DRIFT框架，通过将基础预测器与基于流匹配的残差细化模块相结合，将预训练视觉-语言模型高效适配到需要精确连续输出的解码任务，如时间边界定位和机器人控制动作生成。

    

    许多现代视觉-语言模型（VLM）建立在离散词元的自回归解码之上。虽然基于文本的输出接口能够实现可扩展的预训练以及在多样化任务上强大的零样本泛化能力，但它们并不适合需要精确连续输出的问题，例如定位事件的时间边界或生成机器人控制动作。为了应对这一挑战，我们提出了DRIFT，这是一个将预训练VLM适配到连续解码任务的通用框架。DRIFT将一个基础预测器（提供目标输出的粗略估计）与一个基于流匹配的生成式细化模块（迭代地改进预测）相结合。这种残差形式将生成建模问题从学习全局输出分布转变为围绕强先验建模局部化残差分布，从而大大简化了优化过程。

    arXiv:2606.05758v2 Announce Type: replace-cross  Abstract: Many modern vision-language models (VLMs) build on autoregressive decoding of discrete tokens. While text-based output interfaces enable scalable pretraining and strong zero-shot generalization across diverse tasks, they are poorly suited for problems that require precise continuous outputs, such as localizing temporal boundaries of events or generating robotic control actions. To address this challenge, we propose DRIFT, a general framework for adapting pretrained VLMs to continuous decoding tasks. DRIFT combines a base predictor, which provides a coarse estimate of the target output, with a generative refinement module based on flow matching that iteratively improves the prediction. This residual formulation transforms the generative modeling problem from learning a global output distribution to modeling a localized residual distribution around a strong prior, substantially simplifying optimization. We evaluate DRIFT on both 
    
[^332]: 多智能体推理中的流式通信

    Streaming Communication in Multi-Agent Reasoning

    [https://arxiv.org/abs/2606.05158](https://arxiv.org/abs/2606.05158)

    StreamMA通过让智能体即时流式传递每个推理步骤实现流水线化，首次证明了流式通信既能降低多智能体推理延迟、又能提升推理效果，并给出了流式、串行与单一协议的首个闭式联合理论分析。

    

    多智能体推理系统采用“先生成后传递”的范式，导致端到端延迟随流水线深度线性增长。我们提出StreamMA，一种多智能体推理系统，它在每个推理步骤生成后立即将其流式传输给下游智能体，使相邻智能体流水线化，从而降低延迟。令人惊讶的是，这种流水线化还能提升推理效果：由于多步推理的质量并不均匀，早期步骤比后期步骤更可靠，利用这些可靠的早期步骤而非完整推理链，可以避免易出错的后期步骤误导下游智能体。我们首次对流式、串行和单一协议进行了闭式联合分析，形式化了上述两种优势，推导出有效性排序、加速比上界和成本比。在涵盖数学、科学和代码的八个推理基准上，使用三个前沿大语言模型（Claude Opus 4.6、GPT-5.4 和 GLM-……（原文摘要在此处截断）

    arXiv:2606.05158v3 Announce Type: replace-cross  Abstract: Multi-agent reasoning systems adopt a generate-then-transfer paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, three frontier LLMs (Claude Opus 4.6, GPT-5.4, and GLM-
    
[^333]: Hedge-Bench：面向金融推理的高难度现实任务智能体基准测试

    Hedge-Bench: Benchmarking Agents on Hard, Realistic Tasks Pertaining to Financial Reasoning

    [https://arxiv.org/abs/2606.03918](https://arxiv.org/abs/2606.03918)

    提出了Hedge-Bench基准——基于对冲基金分析师真实推理轨迹和已揭晓市场结果构建的92个环境，用于评估AI智能体在不完整信息下的金融预测能力，发现前沿模型的联合成功率仅为2.54%至31.52%。

    

    现代AI智能体在检索文档、更新电子表格等常规金融分析任务上的能力日益增强。更具挑战性的难题是从不完整的信息中预测未知的未来。公开市场为此提供了一个天然的测试平台：分析师们通常基于当时可获得的信息来预测离散事件。我们提出了Hedge-Bench，这是一个包含92个真实世界环境的基准，这些环境源自对冲基金分析师的推理轨迹，并配以随后已揭晓结果的预测问题。智能体基于冻结的信息快照进行推理，其预测结果将依据真实世界的结果进行评估。在11个前沿模型、每个环境进行三次试验的设置下，联合成功率（要求同时具备正确的预测和超过60%的研究覆盖率）范围为2.54%至31.52%。利用带有回合级信用分配的GRPO方法，我们在165个环境上训练了Muse-Glimmer模型，将22个保留环境上的覆盖率从……（原文摘要至此截断）。

    arXiv:2606.03918v2 Announce Type: replace  Abstract: Modern AI agents are increasingly capable of routine financial-analysis tasks such as retrieving documents and updating spreadsheets. A harder challenge is forecasting an unknown future from incomplete information. Public markets provide a natural testbed: analysts routinely predict discrete events with information from that point in time. We present Hedge-Bench, a benchmark of 92 real-world environments derived from hedge fund analysts' reasoning traces paired with subsequently resolved prediction questions. Agents reason from frozen information snapshots, and their forecasts are evaluated against real-world outcomes. Across 11 frontier models and three trials per environment, joint success rates, which require both a correct forecast and over 60% research coverage, range from 2.54 to 31.52%. Using GRPO with turn-level credit assignment, we train Muse-Glimmer on 165 environments, increasing coverage on 22 held-out environments from 
    
[^334]: 先抓取后规划与失败归因：一个用于精确且可泛化机器人操作的闭环两阶段框架

    Grasp-Then-Plan with Failure Attribution: A Closed Two-Stage Framework for Precise and Generalizable Robotic Manipulation

    [https://arxiv.org/abs/2606.03385](https://arxiv.org/abs/2606.03385)

    提出GTP-FA两阶段“先抓取后规划”框架，利用失败归因模型诊断操作失败原因，并以诊断驱动的方式联合优化抓取与运动规划模块，实现精确且可泛化的机器人长时程操作。

    

    在机器人操作中，抓取与运动规划之间的紧密耦合常常掩盖失败的真正原因，导致低效的反复试错。为实现高效的长时程操作，我们提出了GTP-FA（带失败归因的先抓取后规划），这是一个面向任务的两阶段“先抓取后规划”框架，该框架生成抓取候选，并以所选抓取为条件执行下游运动规划。给定一条失败的操作轨迹，我们学习一个失败归因模型，该模型可泛化至未见过的抓取，并产生稳定的失败模式分布，用于诊断引导的优化。基于这些归因结果，我们以诊断驱动的方式对两个模块进行优化：在抓取方面，我们向抓取候选的评分与优化中注入任务级先验和风险惩罚，以抑制不稳定或与任务不兼容的抓取；在规划方面，我们针对高风险初始……

    arXiv:2606.03385v2 Announce Type: replace-cross  Abstract: In robotic manipulation, the tight coupling between grasping and motion planning often obscures the true source of failure, leading to inefficient trial-and-error. To enable efficient long-horizon manipulation, we propose GTP-FA (Grasp-Then-Plan with Failure Attribution), a task-oriented two-stage 'grasp-then-plan' framework that generates grasp candidates and performs downstream motion planning conditioned on the selected grasp. Given a failed manipulation trajectory, we learn a failure attribution model that generalizes to unseen grasps and produces a stable distribution over failure modes for diagnosis-guided optimization. Based on these attribution results, we then optimize both modules in a diagnosis-driven manner: on the grasping side, we inject task-level priors and risk penalties into grasp candidate scoring and optimization to suppress unstable or task-incompatible grasps; on the planning side, we target high-risk init
    
[^335]: CalArena：一个大规模的事后校准基准测试

    CalArena: A Large-Scale Post-Hoc Calibration Benchmark

    [https://arxiv.org/abs/2605.30188](https://arxiv.org/abs/2605.30188)

    该论文提出了CalArena，一个覆盖近2000个实验、融合经典模型与基础模型预测、并提供数十种校准方法统一可复现实现的大规模标准化事后校准基准，用以系统评估哪些校准方法在实践中真正有效。

    

    可靠的概率估计在许多机器学习应用中至关重要，然而现代分类器往往校准不佳。事后校准提供了一种简单且广泛使用的解决方案，但由于所提出的方法数量众多，加之评估规模小且不一致，使得人们难以确定哪些方法在实践中真正有效。我们引入了一个大规模、标准化的事后校准基准，涵盖表格数据和计算机视觉任务中近2000个实验，包括二分类、多分类和大规模分类设置。我们的基准汇集了来自多种经典模型、现代深度学习架构和基础模型的预测，并在统一的评估框架内提供了数十种校准方法的统一、可复现实现。我们认为，适当评分规则中的事后改进提供了一个原则……（原文在此处截断）

    arXiv:2605.30188v3 Announce Type: replace-cross  Abstract: Reliable probability estimates are critical in many machine learning applications, yet modern classifiers are often poorly calibrated. Post-hoc calibration provides a simple and widely used solution, but the large number of proposed methods, combined with small-scale and inconsistent evaluations, makes it difficult to determine which approaches are truly effective in practice. We introduce a large-scale, standardized benchmark for post-hoc calibration, covering nearly 2000 experiments across tabular and computer vision tasks, including binary, multiclass, and large-scale classification settings. Our benchmark aggregates predictions from a diverse set of classical models, modern deep learning architectures, and foundation models, and provides unified, reproducible implementations of dozens of calibration methods within a common evaluation framework. We argue that Post-Hoc Improvement (PHI) in proper scoring rules offers a princi
    
[^336]: 计划的表达方式重要吗？关于LLM网页代理计划表示的实证研究

    Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents

    [https://arxiv.org/abs/2605.29927](https://arxiv.org/abs/2605.29927)

    本文提出PlanAhead框架，系统评估了四种自然语言计划表示（顺序子目标、叙述、伪代码、检查清单）对多模态LLM网页代理性能的影响，并实现了无需人工标注的自动任务难度分级。

    

    尽管近期取得了进展，基于LLM的网页代理仍然存在探索能力有限、遗漏关键步骤以及对任务约束敏感等问题。先前的研究表明，这些失败大多源于规划能力的不足，然而不同自然语言计划表示方式的影响仍未被探索。为了解决这一问题，我们提出了PlanAhead，一个静态的规划器-执行器框架，用于评估计划表示对代理性能的影响。我们首先自动将WebArena任务分类为3个难度级别，无需人工标注即可实现一致的难度分级。随后，我们在被归类为困难的任务上，系统地评估了4种不同的计划表示方式：顺序子目标、叙述性描述、伪代码和检查清单；并在不同家族的多模态LLM驱动的代理（OpenAI、阿里巴巴和谷歌）上进行了测试。为了考虑随机变异性，我们引入了一种新颖的代理成功分解方法……

    arXiv:2605.29927v2 Announce Type: replace-cross  Abstract: Despite recent advances, LLM-based web agents still struggle with limited exploration, omission of critical steps, and sensitivity to task constraints. Prior work suggests that many of these failures stem from weaknesses in planning, yet the impact of alternative natural language plan representations remains unexplored. To address this, we introduce PlanAhead, a static planner-executor framework that evaluates the impact of plan representation on agent performance. We first automatically categorize WebArena tasks into 3 difficulty levels, enabling consistent difficulty grading without human annotation. Then we systematically evaluate 4 different plan representations on the tasks categorized as hard: sequential subgoals, narrative, pseudocode, and checklist; across different families of multimodal LLM-powered agents(OpenAI, Alibaba, and Google). To account for stochastic variability, we introduce a novel decomposition of agent S
    
[^337]: 超越提问：评估大语言模型的陈述性知识

    Beyond Questions: Evaluating the Stated Knowledge of LLMs

    [https://arxiv.org/abs/2605.26937](https://arxiv.org/abs/2605.26937)

    该论文提出了开放式知识评估这一新范式，并构建了包含10,000个实体的BeQu基准，通过开放式引导提示评估大语言模型自然表达的知识，从而克服了传统预定义问题评测中的可用性偏差。

    

    大语言模型（LLM）中的参数化知识是其成功的基石，但人们对它的理解仍然不足。现有的知识评测基准通常依赖于预定义的问题（例如“M.L. King的出生日期是什么？”），仅评估基准设计者明确选择查询的知识，这是一种有问题的可用性偏差。在本文中，我们提出了开放式知识评估，这是一种用于评测大语言模型陈述性知识的新范式。该范式不再提出狭窄的问题，而是评估模型在面对开放式引导提示（例如“告诉我你所知道的关于M.L. King的一切”）时选择表达出来的知识。这将关注点从预定义答案的检索转向刻画模型自然表达的知识。我们通过BeQu（Beyond Questions，超越提问）基准来实例化这一范式，该基准包含10,000个实体，并配有用于语句验证的参考语料库。使用BeQu，我们……

    arXiv:2605.26937v3 Announce Type: replace-cross  Abstract: Parametric knowledge in large language models (LLMs) is a cornerstone of their success, yet remains poorly understood. Existing knowledge benchmarks typically rely on predefined questions (e.g., "What is the birth date of M.L. King?"), evaluating only knowledge that benchmark designers explicitly choose to query, a problematic availability bias. In this paper, we introduce open knowledge evaluation, a new paradigm for benchmarking the stated knowledge of LLMs. Instead of asking narrow questions, it evaluates models on the knowledge they choose to surface in response to open-ended elicitation prompts (e.g., "Tell me everything you know about M.L. King"). This shifts the focus from predefined answer retrieval toward characterizing the knowledge models naturally express. We instantiate this paradigm with BeQu (Beyond Questions), a benchmark of 10,000 entities paired with reference corpora for statement verification. Using BeQu, we
    
[^338]: 隐藏状态隐私存在“空缺中段”

    Hidden-State Privacy Has an Empty Middle

    [https://arxiv.org/abs/2605.24042](https://arxiv.org/abs/2605.24042)

    该论文通过对1,536个高斯协方差的实证测试和Fisher球理论下界的证明，揭示了隐藏状态隐私发布中存在“空缺中段”现象——不存在任何能同时提供中等效用与中等隐私的高斯机制，并指出对角逆Fisher发布是唯一达到极小极大最优的机制，但只能处于隐私/效用的极端边缘。

    

    在我们为单层隐藏状态隐私测试的1,536个高斯发布协方差中，没有任何一个能够在面对自适应检索攻击者时同时实现中等效用与中等隐私。我们证明了一个互补的Fisher球下界：每一个在O(1) Fisher效用下的满秩高斯发布都存在某个方向，其马氏距离信号随隐藏宽度线性增长，这排除了该类机制中一致的高斯安全性，并与经验上观察到的空缺中段相吻合。对角逆Fisher发布 Σ*_diag(𝒦) = (2𝒦/d)·diag(1/F_ii) 是在一阶KL预算𝒦下唯一的极小极大最优对角机制，也是在32个模型层网格的每个点上，唯一实现最坏攻击者top-1 ≤ 0.001的发布机制，但它位于隐私/效用的边缘，而非填补中段。一种广义特征值机制在欧氏检索崩溃下达到了13倍的帕累托缩减（摘要在此处截断）。

    arXiv:2605.24042v4 Announce Type: replace-cross  Abstract: Of $1{,}536$ Gaussian release covariances we tested for single-layer hidden-state privacy, zero achieve both moderate utility and moderate privacy against an adaptive retrieval attacker. We prove a complementary Fisher-ball lower bound: every full-rank Gaussian release at $O(1)$ Fisher utility admits a direction whose Mahalanobis signal grows linearly in hidden width, ruling out uniform Gaussian safety in the class and matching the empirical empty middle. The diagonal inverse-Fisher release $\Sigma^\star_{\mathrm{diag}}(\mathcal{K}) = (2\mathcal{K}/d)\,\mathrm{diag}(1/F_{ii})$ is the unique minimax-optimal diagonal mechanism at first-order KL budget $\mathcal{K}$ and the only release with worst-attacker top-1 $\le 0.001$ at every point of a 32 model-layer grid, but it sits on a privacy/utility edge rather than filling the middle. A generalized-eigen mechanism reaching $13\times$ Pareto reduction under Euclidean retrieval collap
    
[^339]: LLM代码坏味道：分类体系与检测方法

    LLM Code Smells: A Taxonomy and Detection Approach

    [https://arxiv.org/abs/2605.22976](https://arxiv.org/abs/2605.22976)

    本文提出了LLM代码坏味道的独立分类体系和九种坏味道目录，开发了静态分析工具SpecDetect4LLM进行检测，并通过对692个开源项目的实证评估发现73.5%的项目存在LLM代码坏味道。

    

    由于大语言模型（LLM）具有多功能性、灵活性以及在一定程度上模拟人类推理的能力，它们正越来越多地被集成到软件系统中以实现多种用途。然而，LLM推理在源代码中的不良集成可能会损害软件系统的质量。因此，必须记录不恰当的LLM集成编码实践，以帮助开发者缓解此类问题。延续我们早期关于LLM代码坏味道的工作，本文通过提出一个独立的分类体系以及包含九种LLM代码坏味道的目录，巩固并完善了这一概念。我们还创建了SpecDetect4LLM——一个用于检测这些坏味道的静态源代码分析工具，并对其检测效果（精确率和召回率）以及LLM代码坏味道在692个开源软件项目（171,194个源文件）中的普遍程度进行了广泛的实证评估。我们的结果表明，73.5%的可分析项目受到LLM代码坏味道的影响。

    arXiv:2605.22976v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly integrated into software systems for diverse purposes, due to their versatility, flexibility, and ability to simulate human reasoning to some extent. However, poor integration of LLM inference in source code can undermine software system quality. Therefore, inadequate LLM integration coding practices must be documented to help developers mitigate such issues. Following our earlier work on LLM code smells, this paper consolidates and refines the concept by presenting a self-contained taxonomy and a catalog of nine LLM code smells. We also create SpecDetect4LLM, a static source code analysis tool for their detection, and conduct extensive empirical evaluations of its detection effectiveness (precision and recall) as well as the prevalence of LLM code smells across 692 open-source software projects (171,194 source files). Our results show that LLM code smells affect 73.5% of the analyz
    
[^340]: 从SGD到Muon：基于Schatten-p范数的自适应优化

    From SGD to Muon: Adaptive Optimization via Schatten-p Norms

    [https://arxiv.org/abs/2605.19781](https://arxiv.org/abs/2605.19781)

    该论文提出了一种基于Schatten-p范数的数据驱动准则，可动态地为神经网络各层自适应选择从SGD到Muon之间的最优更新几何结构，并在统一框架下恢复了SGD、Muon、Adam和MuAdam等主流优化器。

    

    现代优化器（如Muon）对其更新施加矩阵级别的几何约束，这些约束可以在线性最小化预言机（LMO）理论下得到统一。然而，现有所有方法都为更新规则施加固定的LMO几何结构，这些结构是通过设计或经验选择的，未必与问题本身的几何特性相符从而并非最优。我们提出了一种新颖且高效的数据驱动准则，用于在深度神经网络的各个层上动态选择近似最优的更新LMO几何结构。该准则借助单步随机特征回归代理模型，从梯度和激活统计量中以闭式形式推导得出，能够在一个从SGD更新到Muon更新之间进行插值的设计空间中自由导航。此外，通过整合参数级预条件化，我们的框架可以将SGD、Muon、Adam和MuAdam作为特定极值情形加以恢复。为使这种自适应方法具备可扩展性，我们将其与高效的计算策略相结合。

    arXiv:2605.19781v2 Announce Type: replace  Abstract: Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which are not necessarily optimal according to the problem's geometry. We introduce a novel efficient datadriven criterion for dynamically choosing proxy-optimal update LMO geometries on individual Deep Neural Network layers. Derived in closed form from gradient and activation statistics using a single-step random feature regression surrogate model, our criterion navigates a design space interpolating from SGD to Muon updates. Moreover, integrating parameter-wise preconditioning allows our framework to recover SGD, Muon, Adam, and MuAdam as specific extrema. To make this adaptive approach scalable, we pair it with efficient computational strategie
    
[^341]: CLEAR：耦合局部证据与情感表征用于对话中的多模态情感识别

    CLEAR: Coupling Local Evidence with Affective Representations for Multimodal Emotion Recognition in Conversation

    [https://arxiv.org/abs/2605.18547](https://arxiv.org/abs/2605.18547)

    提出CLEAR方法，通过将局部判别证据与内部情感表征耦合，解决了多模态大语言模型在对话情感识别中情感理解与标签预测不一致、以及当前话语局部线索被上下文稀释的问题。

    

    在视频通话和直播讨论中，识别说话者的情绪有助于定位情绪显著的片段，并支持细粒度的视频内容分析。对话中的多模态情感识别（ERC）通过将文本、视觉和声学信息与对话上下文相结合，提供了一种识别此类内容中情绪的方法。当前主流的情感识别方法主要利用多模态大语言模型（MLLM）的上下文理解能力，通过生成标签来预测情绪。然而，情绪理解与标签预测之间可能存在不一致，导致内部情感信息未被充分利用。此外，当当前话语与大量对话历史和多模态信息进行联合建模时，其局部判别性线索可能被稀释，在由此产生的上下文表征中变得不再显著。为解决这些限制……（摘要原文在此处截断）

    arXiv:2605.18547v2 Announce Type: replace  Abstract: In video calls and livestreamed discussions, recognizing speakers' emotions helps identify emotionally salient segments and supports fine-grained video content analysis. Multimodal Emotion Recognition in Conversation (ERC) provides an approach to recognizing emotions in such content by integrating textual, visual, and acoustic information with conversational context. Current mainstream emotion recognition methods largely leverage the contextual understanding capabilities of multimodal large language models (MLLMs) to predict emotions by generating labels. However, emotion understanding and label prediction may be inconsistent, leaving the internal affective information underutilized. Moreover, when the current utterance is jointly modeled with extensive dialogue history and multimodal information, its local discriminative cues may be diluted and become less salient in the resulting contextual representation. To address these limitati
    
[^342]: BoLT：让面向昂贵LLM任务的黑盒优化研究平民化的基准测试

    BoLT: A Benchmark to Democratize Black-box Optimization Research for Expensive LLM Tasks

    [https://arxiv.org/abs/2605.17000](https://arxiv.org/abs/2605.17000)

    该论文提出了BoLT——首个以LLM为中心的基准测试，通过解决LLM训练与推理成本过高的门槛问题，使黑盒优化（BBO）方法能够在真实现代LLM任务（如超参数、数据混合和提示词优化）上进行系统研究和平民化评估。

    

    arXiv:2605.17000v2 公告类型：replace-cross

    arXiv:2605.17000v2 Announce Type: replace-cross  Abstract: Optimization of LLM training and inference configurations, such as hyperparameters, data mixtures, and prompts, is critical to performance, but it is often approached heuristically in practice, leading to potentially suboptimal outcomes. By framing them as noisy, expensive, and derivative-free optimization problems, Bayesian optimization (BO) and other black-box optimization (BBO) methods offer a promising yet underexplored direction for principled, sample-efficient methods. However, LLM training and inference costs are prohibitively high for most of the BBO research community, and new methods are often only evaluated on synthetic test functions and small-scale datasets that fail to capture the challenges of modern LLM optimization problems. This impedes the development of BBO methods and makes it difficult to assess their effectiveness on modern LLM tasks. We introduce BoLT, the first LLM-centric benchmark that democratizes LL
    
[^343]: 学习的几何学：训练过程中的读写不对称性

    The Geometry of Learning: Read-Write Asymmetry Across Training

    [https://arxiv.org/abs/2605.16600](https://arxiv.org/abs/2605.16600)

    该论文以残差流为共享坐标系分析预训练与对齐阶段对 transformer 权重的更新方向，揭示了“读写不对称性”：读侧矩阵（query/key）的更新集中于其输入激活的主子空间，而写侧矩阵（输出矩阵）的更新相对于预测子空间接近随机，且这一特征是训练阶段的属性而非训练目标的属性。

    

    预训练和对齐更新的是同一个 transformer 的权重，我们探究每个阶段分别在哪些方向上改变了这些权重。残差流为每个矩阵提供了一个共享的坐标系，在这一坐标系中，权重的变化既可以与该矩阵所读取的激活值的主子空间进行比较，也可以与 unembedding 的预测子空间进行比较。在这一框架下，对齐阶段对 query 和 key 矩阵的更新主要富集于主输入子空间中，而其对输出矩阵的更新相对于预测子空间则接近随机；同样的特征也出现在预训练后期，以及我们对同一检查点施加的每一种训练目标之下，因此这是训练阶段的属性，而非训练目标的属性。线性映射的梯度是外积 $\delta_t a_t^\top$ 之和，残差流对于从中读取信息的矩阵通过激活值 $a_t$ 进入该乘积，而对于（写向它的矩阵）则通过上游梯度……（原文摘要在此处截断）

    arXiv:2605.16600v2 Announce Type: replace-cross  Abstract: Pretraining and alignment update the same transformer weights, and we ask in which directions each stage changes them. The residual stream gives every matrix a shared coordinate system, in which a weight change can be compared with the principal subspace of the activations the matrix reads and with the prediction subspace of the unembedding. In this frame, alignment updates to the query and key matrices are enriched in the principal input subspace, while its updates to the output matrices are close to random relative to the prediction subspace, and the same signature appears in late pretraining and under every objective we apply to one checkpoint, so it is a property of the training stage rather than of the objective. The gradient of a linear map is a sum of outer products $\delta_t a_t^\top$, and the residual stream enters this product through the activation $a_t$ for matrices that read from it and through the upstream gradien
    
[^344]: 边缘AI驱动的去中心化任务分配学习排序在循环智能制造中的应用

    Edge-AI-Driven Learning-to-Rank for Decentralized Task Allocation in Circular Smart Manufacturing

    [https://arxiv.org/abs/2605.16433](https://arxiv.org/abs/2605.16433)

    该论文提出了一种边缘AI驱动的去中心化任务分配框架，通过轻量级决策智能体结合资源感知启发式和回归模型，在循环制造中实现高效的任务分配，同时兼顾能耗和资源约束。

    

    智能制造系统中的任务分配必须在去中心化决策、动态工作负载和共享资源约束下运行。在循环制造环境中，这些挑战进一步加剧，因为任务竞争可重复使用、容量受限的资产，且机器选择还可能影响加工能耗。尽管已有基于学习的方法被探索用于任务分配，但预测模型的改进并不一定能在去中心化谈判中转化为更好的分配结果。本工作提出了一种边缘AI驱动的去中心化任务分配框架。我们开发了部署在机器级别的轻量级决策智能体。该框架逐步构建：首先，一种资源感知启发式方法建立去中心化投标结构；其次，基于回归的边缘AI公式检验学习到的局部投标近似；最后，一种紧凑的自编码器正则化方法进一步优化。

    arXiv:2605.16433v2 Announce Type: replace-cross  Abstract: Task allocation in smart manufacturing systems must operate under decentralized decision-making, dynamic workloads, and shared-resource constraints. In circular manufacturing settings, these challenges are further intensified because tasks compete for reusable, capacity-constrained assets, and machine selection also might affect processing energy. Although learning-based approaches have been explored for task allocation, improvements in predictive modeling do not necessarily translate into better allocation outcomes under decentralized negotiation. This work proposes an Edge-AI-driven decentralized task-allocation framework. We develop lightweight decision intelligence deployed at the machine level. It is developed progressively: first, a resource-aware heuristic establishes the decentralized bidding structure; a regression-based Edge-AI formulation then examines learned local bid approximation, and a compact autoencoder-regula
    
[^345]: REALM：面向局部场电位建模的回溯编码器对齐

    REALM: Retrospective Encoder Alignment for LFP Modeling

    [https://arxiv.org/abs/2605.14867](https://arxiv.org/abs/2605.14867)

    本文提出REALM框架，通过回溯知识蒸馏将非因果模型的表达知识迁移到因果模型中，从而实现高精度且可实时部署的LFP行为解码。

    

    尖峰活动一直是行为解码的主要神经信号，因为其高时空分辨率支持精确解码。然而，随着皮层内脑机接口（iBCI）向更高通道数和无线操作方向发展，记录尖峰所需的高采样率带来了巨大的功耗和带宽需求。局部场电位（LFP）提供了互补的优势，包括更强的长期稳定性、更低的能耗和更低的带宽需求。然而，基于LFP的解码器通常精度较低，且依赖无法直接用于实时部署的非因果架构。我们提出了REALM，一个用于因果LFP行为解码的回溯知识蒸馏（RKD）框架。受语音识别中离线到在线蒸馏的启发，REALM将预训练的多会话双向（摘要在此处被截断）

    arXiv:2605.14867v2 Announce Type: replace-cross  Abstract: Spike activity has been the dominant neural signal for behavior decoding because its high spatiotemporal resolution supports accurate decoding. However, as intracortical brain-computer interfaces (iBCIs) move toward higher channel counts and wireless operation, the high sampling rates required to record spikes create substantial power and bandwidth demands. Local field potentials (LFPs) offer complementary advantages, including greater long-term stability, lower energy consumption, and lower bandwidth requirements. However, LFP-based decoders often achieve lower accuracy and rely on non-causal architectures that cannot be used directly for real-time deployment. We propose REALM, a retrospective knowledge distillation (RKD) framework for causal LFP behavior decoding. Inspired by offline-to-online distillation in speech recognition, REALM transfers non-causal representational knowledge from a pretrained, multi-session bidirection
    
[^346]: 通过审讯进行审核：揭示智能体社交网络中的隐匿意图

    Moderation by Interrogation: Uncovering Concealed Intent in Agent Social Networks

    [https://arxiv.org/abs/2605.12856](https://arxiv.org/abs/2605.12856)

    提出 Bot-Mod 审核框架，通过诊断性审讯与贝叶斯后验推断识别智能体的隐藏意图，突破了依赖内容信号的审核局限，并构建了 Molt-Intent 数据集加以评估。

    

    arXiv:2605.12856v3 公告类型：替换。摘要：多智能体系统的出现带来了超越内容过滤的新型审核挑战。具有恶意意图的智能体可能会发布表面上无害的内容以逃避基于内容的审核，同时暗中服务于隐藏的目标。为了应对这一挑战，我们提出了 Bot-Mod，一个以智能体意图而非传统内容层面信号为基础的审核框架。Bot-Mod 使用一组固定的诊断探针问题对目标智能体进行审讯，并通过对候选意图进行贝叶斯后验推断来推断其意图。该方法从一个经过校准的仅基于内容的先验出发，利用审核者赋予的每个回答在每种候选意图下的似然来不断更新后验概率，直到满足停止准则为止。为了评估我们的方法，我们构建了 Molt-Intent 数据集，该数据集源自 Moltbook，捕捉了真实社区中多样化的良性与恶意行为。

    arXiv:2605.12856v3 Announce Type: replace  Abstract: The emergence of multi-agent systems introduces novel moderation challenges that extend beyond content filtering. Agents with $\textit{malicious intent}$ may contribute content that appears benign to evade content-based moderation, while serving a hidden objective. To address this challenge, we introduce Bot-Mod a moderation framework that grounds detection in agent intent rather than traditional content-level signals. Bot-Mod interrogates a target agent using a fixed set of diagnostic probes and infers its intent through Bayesian posterior inference over candidate intents. It starts with a calibrated content-only prior and updates the posterior using the $\textit{likelihood}$ of each answer under every candidate intent, as assigned by the moderator, until a stopping criterion is met. To evaluate our approach, we construct Molt-Intent, a dataset derived from Moltbook, capturing diverse benign and malicious behaviors within real commu
    
[^347]: PIVOT：通过轨迹细化弥合LLM智能体中规划与执行之间的鸿沟

    PIVOT: Bridging Planning and Execution in LLM Agents via Trajectory Refinement

    [https://arxiv.org/abs/2605.11225](https://arxiv.org/abs/2605.11225)

    PIVOT提出了一种自监督框架，通过计划、检查、演化、验证四个阶段将轨迹作为可优化对象进行迭代细化，利用文本梯度编码规划与执行的差异，从而弥合LLM智能体中规划与执行之间的鸿沟，并在DeepPlanning和GAIA基准上取得最先进的性能。

    

    arXiv:2605.11225v2 公告类型：替换 摘要：基于大语言模型（LLM）的智能体经常生成看似连贯的计划，但在执行时却因动作不可行、违反约束以及在较长时间跨度上的误差累积而失败。PIVOT（Plan-Inspect-eVOlve Trajectories，计划-检查-演化轨迹）通过一个自监督框架来解决这种规划与执行的错位问题，该框架将轨迹视为可优化的对象，并通过与环境交互对其进行迭代细化。该框架包含四个阶段：PLAN（计划）生成候选轨迹；INSPECT（检查）执行这些轨迹并计算结构化损失，其中的文本梯度编码了计划与执行之间的差异；EVOLVE（演化）应用这些信号以生成改进后的轨迹；VERIFY（验证）则针对任务约束执行最终的全局检查。单调接受过程确保了解决方案质量不会下降。在DeepPlanning和GAIA上的实证评估展示了最先进的性能：在人在环（HITL）反馈下，

    arXiv:2605.11225v2 Announce Type: replace  Abstract: Large language model (LLM)-based agents frequently generate seemingly coherent plans that fail upon execution due to infeasible actions, constraint violations, and compounding errors over extended horizons. PIVOT (Plan-Inspect-eVOlve Trajectories) addresses this plan-execution misalignment through a self-supervised framework that treats trajectories as optimizable objects iteratively refined via environment interaction. The framework comprises four stages: PLAN generates candidate trajectories; INSPECT executes them and computes structured losses with textual gradients encoding plan-execution discrepancies; EVOLVE applies these signals to produce improved trajectories; and VERIFY performs a final global check against task constraints. A monotonic acceptance process ensures a non-decreasing solution quality. Empirical evaluations on DeepPlanning and GAIA demonstrate state-of-the-art performance: with human-in-the-loop (HITL) feedback,
    
[^348]: RePO-VLA：面向视觉-语言-动作模型的恢复驱动策略优化

    RePO-VLA: Recovery-Driven Policy Optimization for Vision-Language-Action Models

    [https://arxiv.org/abs/2605.09410](https://arxiv.org/abs/2605.09410)

    RePO-VLA 提出了一种恢复驱动的策略优化框架，通过恢复感知初始化（RAI）和进度感知语义价值函数（PAS-VF）充分利用失败与恢复轨迹，显著提升 VLA 模型在长时程、接触密集操作任务中的鲁棒性。

    

    视觉-语言-动作（VLA）模型在长时程、接触密集的操作任务中仍然脆弱，这是因为仅基于成功的模仿学习对执行偏差几乎不提供监督信号，而失败的 rollout 轨迹往往被直接丢弃。我们提出了 RePO-VLA，一个恢复驱动的策略优化框架，它为成功、恢复和失败轨迹分配不同的角色。RePO-VLA 首先应用恢复感知初始化（RAI），将每个纠正性后缀切分为新的回合，使纠正动作依赖于当前的不利状态而非先前的失败。接着，它学习一个进度感知语义价值函数（PAS-VF），将时空轨迹特征与指令和成功参考对齐。由此产生的标签通过可靠性衰减挽救有用的失败前缀，而低价值标签则标记漂移和终端崩溃，从而教会模型区分正常动作、失败动作和纠正动作之间的差异。

    arXiv:2605.09410v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models remain brittle in long-horizon, contact-rich manipulation because success-only imitation provides little supervision for execution drift, while failed rollouts are often discarded. We introduce RePO-VLA, a recovery-driven policy optimization framework that assigns distinct roles to success, recovery, and failure trajectories. RePO-VLA first applies Recovery-Aware Initialization (RAI), slicing each corrective suffix into a new episode so corrective actions depend on the current adverse state rather than the preceding failure. It then learns a Progress-Aware Semantic Value Function (PAS-VF), aligning spatiotemporal trajectory features with instructions and successful references. The resulting labels salvage useful failure prefixes via reliability decay, while low-value labels mark drift and terminal breakdowns, teaching differences among nominal, failed, and corrective actions. The data engine 
    
[^349]: 联邦预测性维护系统中设备端生成模型的权衡研究

    On the Tradeoffs of On-Device Generative Models in Federated Predictive Maintenance Systems

    [https://arxiv.org/abs/2605.07860](https://arxiv.org/abs/2605.07860)

    本文全面分析了VAE、GAN和扩散模型在联邦预测性维护系统中的性能与通信开销权衡，并基于分割参数个性化联邦学习提供了共享/私有参数划分的实证研究。

    

    联邦学习（FL）已成为在分布式物联网（IoT）环境中保护客户端数据所有权和控制权的有前景的范式。虽然判别式模型主导了大多数联邦学习用例，但生成式模型的最新进展——如变分自编码器（VAE）、生成对抗网络（GAN）和扩散模型（DM）——为时间序列分析中的无监督异常检测提供了新机会，并在关键工业基础设施的预测性维护（PdM）中具有相关应用。在这项工作中，我们对联邦预测性维护背景下的VAE、GAN和DM进行了全面分析。我们分析了它们在完全联邦和部分联邦设置下的性能和通信开销，其中仅共享模型组件的子集。基于分割参数个性化联邦学习，本文提供了共享/私有参数划分的实证实例化。

    arXiv:2605.07860v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) has emerged as a promising paradigm for preserving client data ownership and control over distributed Internet of Things (IoT) environments. While discriminative models dominate most FL use cases, recent advances in generative models - such as Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), and Diffusion Models (DM) - offer new opportunities for unsupervised anomaly detection in time series analysis, with relevant applications in predictive maintenance (PdM) in critical industrial infrastructures. In this work, we present a comprehensive analysis of VAEs, GANs, and DMs in the context of federated PdM. We analyze their performance and communication overhead under both full and partial federation setups, where only subsets of model components are shared. Building on split-parameter personalized FL, this paper provides an empirical instantiation of shared/private parameter partitionin
    
[^350]: Dr. Post-Training：从数据正则化视角看大语言模型后训练

    Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training

    [https://arxiv.org/abs/2605.07063](https://arxiv.org/abs/2605.07063)

    该论文提出 Dr. Post-Training 框架，创新性地将通用训练数据重新概念化为防止模型过拟合稀缺目标数据的数据诱导正则化器，通过在每步训练中将目标数据的更新方向投影到通用数据构建的可行集上，并将标准训练与现有数据选择方法统一为该框架下的特例。

    

    数据选择方法旨在解决大语言模型后训练中的一个关键挑战：如何有效利用稀缺的高保真目标数据与大量但不完全对齐的通用训练数据。在这项工作中，我们超越了数据选择的框架，提出了 Dr. Post-Training（数据正则化后训练），这是一个新颖的框架，它将通用训练数据重新概念化为一种由数据诱导的正则化器，用于防止模型对稀缺目标任务的过拟合，而不是将其作为可供选择的数据池。具体而言，我们的框架提出，在每个训练步骤中，使用通用训练数据构建一个模型更新方向的可行集，并将稀缺目标数据所指定的模型更新方向投影到该可行集上。标准训练和现有的数据选择方法可以看作是在不同数据诱导正则化器选择下的特例，而这些方法对应于不同的

    arXiv:2605.07063v2 Announce Type: replace-cross  Abstract: Data selection methods address a critical challenge in LLM post-training: effectively leveraging scarce, high-fidelity target data alongside abundant but imperfectly aligned general training data. In this work, we move beyond the data-selection framing and introduce Dr. Post-Training (Data-Regularized Post-Training), a novel framework that reconceptualizes general training data as a data-induced regularizer that prevents overfitting to the scarce target objective, rather than serving as a pool for selection. Specifically, our framework proposes that, at each training step, we construct a feasible set of model update directions using the general training data, and project the model update direction specified by the scarce target data onto that feasible set. Standard training and existing data selection methods arise as special cases with different choices of the data-induced regularizer, and these methods correspond to different
    
[^351]: 部分因果效应识别的最优实验设计

    Optimal Experiments for Partial Causal Effect Identification

    [https://arxiv.org/abs/2605.06993](https://arxiv.org/abs/2605.06993)

    该论文提出“最大效力问题”这一新框架，用于在成本约束下选择最优实验组合以最大程度收紧部分可识别因果查询的界限，证明了该问题的NP难性，给出了评估认识效力的通用方法，并引入基于因果图结构的剪枝准则（如线性时间的路径拦截规则）来控制超指数级搜索空间。

    

    因果查询往往无法仅从观测数据中完全识别，而能够收紧识别界限的实验通常代价高昂。我们研究在观察实验结果之前，选择一个成本受限的实验子集以最大程度收紧目标查询界限的问题。我们将其形式化为“最大效力问题”，其中认识效力衡量一个实验在最坏情况下所能保证的界限宽度缩减量，并通过从0-1背包问题的归约证明该问题是NP难的。基于Duarte等人（2023）的多项式规划框架，我们给出了在离散设置中评估认识效力的通用程序。为控制超指数级的搜索空间，我们引入了两个仅依赖于因果图和查询的图剪枝准则：一种新颖的路径拦截规则，它利用区域结构在线性时间内证明实验的零效力。

    arXiv:2605.06993v2 Announce Type: replace  Abstract: Causal queries are often only partially identifiable from observational data, and experiments that could tighten the resulting bounds are typically costly. We study the problem of selecting, prior to observing experimental outcomes, a cost-constrained subset of experiments that maximally tightens bounds on a target query. We formalize this as the max-potency problem, where epistemic potency measures the worst-case reduction in bound width guaranteed by an experiment, and show that this problem is NP-hard via a reduction from 0-1 knapsack. Building on the polynomial-programming framework of Duarte et al. (2023), we give a general procedure for evaluating epistemic potency in discrete settings. To control the super-exponential search space, we introduce two graphical pruning criteria that depend only on the causal graph and the query: a novel path-interception rule that exploits district structure to certify zero potency in linear time
    
[^352]: 后验优先的神经PDE模拟：从单个场推断下游任务所需的信息

    Posterior-First Neural PDE Simulation: Inferring What Downstream Tasks Need from a Single Field

    [https://arxiv.org/abs/2605.03247](https://arxiv.org/abs/2605.03247)

    该论文提出“后验优先”的神经PDE模拟范式：由下游任务确定需推断的最小任务充分问题状态，用其后验分布（可由基于模拟推断的摊销后验估计器提供）来条件化滚动推演与下游查询，并从理论上证明贝叶斯下游价值经此后验分解、严格评分规则以之为最优，而确定性坍缩会带来歧义性损失。

    

    神经偏微分方程（PDE）模拟器在部署时往往只获得单个观测场。在这种设定下，场到未来的预测器会将不同的潜在问题状态坍缩为同一个确定性接口，从而丢失对可靠滚动推演和下游决策而言必要的歧义性信息。我们提出后验优先的神经PDE模拟，这是一种“任务—目标—接口”范式：所声明的下游任务决定需要推断的最小任务充分问题状态，而该状态上的后验分布则对滚动推演和下游查询进行条件化。该后验可由摊销式后验估计器提供，包括采用学习摘要的基于模拟的推断（SBI）；我们的贡献在于目标、接口及其理论。我们证明贝叶斯下游价值可通过该后验进行分解，精化标签上的严格（proper）评分规则将其识别为总体风险最小化器，而确定性坍缩则会招致歧义性代价……

    arXiv:2605.03247v2 Announce Type: replace-cross  Abstract: Neural PDE simulators often receive only a single observed field at deployment. In this setting, a field-to-future predictor can collapse distinct latent problem states into the same deterministic interface, losing the ambiguity needed for reliable rollout and downstream decisions. We propose posterior-first neural PDE simulation, a task-to-target-to-interface paradigm: the declared downstream tasks determine the minimal task-sufficient problem state to infer, and the posterior over that state conditions the rollout and downstream queries. This posterior can be supplied by amortized posterior estimators, including simulation-based inference with learned summaries; our contribution is the target, the interface, and their theory. We show that Bayes downstream values factor through this posterior, proper scoring rules on refinement labels identify it as their population-risk minimizer, and deterministic collapse incurs an ambiguit
    
[^353]: 关于KL正则化离线多臂老虎机的最优样本复杂度

    On the Optimal Sample Complexity of Offline Multi-Armed Bandits with KL Regularization

    [https://arxiv.org/abs/2605.02141](https://arxiv.org/abs/2605.02141)

    本文在多臂老虎机设置下对KL正则化离线学习算法KL-PCB给出了精确分析，证明其在大、小正则化条件下分别达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 和 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，从而基本刻画了KL正则化离线学习的最优样本复杂度。

    

    Kullback-Leibler（KL）正则化在离线决策中被广泛使用，并具有诸多优势，这促使了近期关于基于KL正则化性能指标的离线学习样本复杂度的研究。然而，KL正则化离线学习的精确样本复杂度在很大程度上仍未被完全刻画。本文在多臂老虎机（MABs）的设置下研究这一问题。我们对KL-PCB算法（Zhao et al., 2026）进行了精确的分析，证明其在大正则化 $\eta = \tilde{O}(\epsilon^{-1})$ 下达到 $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ 的样本复杂度，而在小正则化 $\eta = \tilde{\Omega}(\epsilon^{-1})$ 下达到 $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ 的样本复杂度，其中 $\eta$ 为正则化参数，$S$ 为上下文数量，$A$ 为臂的数量，$C^{\pi^*}$ 为最优策略 $\pi^*$ 处的策略覆盖系数。

    arXiv:2605.02141v2 Announce Type: replace-cross  Abstract: Kullback-Leibler (KL) regularization is widely used in offline decision-making and offers several benefits, motivating recent work on the sample complexity of offline learning with respect to KL-regularized performance metrics. Nevertheless, the exact sample complexity of KL-regularized offline learning remains largely from fully characterized. In this paper, we study this question in the setting of multi-armed bandits (MABs). We provide a sharp analysis of KL-PCB (Zhao et al., 2026), showing that it achieves a sample complexity of $\tilde{O}(\eta SAC^{\pi^*}/\epsilon)$ under large regularization $\eta = \tilde{O}(\epsilon^{-1})$, and a sample complexity of $\tilde{\Omega}(SAC^{\pi^*}/\epsilon^2)$ under small regularization $\eta = \tilde{\Omega}(\epsilon^{-1})$, where $\eta$ is the regularization parameter, $S$ is the number of contexts, $A$ is the number of arms, $C^{\pi^*}$ policy coverage coefficient at the optimal policy $
    
[^354]: DeGenTWeb：首次审视以机器生成文本为主的网站

    DeGenTWeb: A First Look at MGT-dominant Websites

    [https://arxiv.org/abs/2605.00087](https://arxiv.org/abs/2605.00087)

    提出了 DeGenTWeb 框架，通过将机器生成文本检测器适配到网页并聚合网站级检测结果，首次系统识别出以机器生成内容为主的网站，发现 Common Crawl 存档网站中 20.9% 和 Bing 搜索结果链接网站中 13.6% 属于机器生成文本主导型。

    

    许多新闻报道声称，来自大语言模型（LLM）的机器生成文本（MGT）正在接管网络。然而，这些说法通常并非基于具有代表性的网络样本，且其背后的方法往往不透明。因此，我们目前缺乏对网络上机器生成文本真实普及程度和特征的了解。我们提出了 DeGenTWeb，一个系统性识别“机器生成文本主导型网站”的方法，即那些信息性散文内容大部分由机器生成、几乎不包含人类输入的网站。我们展示了如何将机器生成文本检测器适配到网页上使用，以及如何汇总来自同一网站多个页面的检测结果，以实现准确的网站级分类。DeGenTWeb 仅将 0.12% 的人类撰写网站误标为机器生成文本主导型网站。我们发现，Common Crawl 在2026年1月至7月期间存档的网站中有 20.9%，以及2025年 Bing 搜索结果所链接的网站中有 13.6% 属于机器生成文本主导型网站。

    arXiv:2605.00087v2 Announce Type: replace-cross  Abstract: Many news reports have claimed that machine-generated text (MGT) from large language models (LLMs) is taking over the web. However, these claims are typically not based on a representative sample of the web, and the methodology underlying them is often opaque. Consequently, we lack an understanding of the true prevalence and characteristics of MGT on the web.   We describe DeGenTWeb, which systematically identifies MGT-dominant websites: sites whose informational prose content is mostly MGT, with little human input. We show how to adapt detectors of MGT for use on web pages, and how to aggregate detection results from multiple pages on a site for accurate site-level categorization. DeGenTWeb mislabels only 0.12% of human-authored sites as MGT-dominant. We find that 20.9% of the sites archived by Common Crawl between January and July 2026 and 13.6% of the sites linked from Bing's search results in 2025 are MGT-dominant. These fi
    
[^355]: AmaraSpatial-10K：一个面向空间计算与具身智能的空间与语义对齐3D数据集

    AmaraSpatial-10K: A Spatially and Semantically Aligned 3D Dataset for Spatial Computing and Embodied AI

    [https://arxiv.org/abs/2604.23018](https://arxiv.org/abs/2604.23018)

    该论文发布了包含10,071个经过度量尺度标准化、确定性锚定并附带完整材质与语义标注的3D资产数据集AmaraSpatial-10K，并配套提出了一套可复用的3D资产质量评估套件，为空间计算与具身智能提供了可直接使用的高质量3D资源。

    

    网络规模的3D资产资源库虽然数量庞大，但很少能够不经清理就直接使用，它们普遍存在度量尺度随意、轴心点错误、几何结构脆弱以及纹理不完整等缺陷，这些缺陷限制了它们在具身智能、机器人和空间计算中的应用。我们提出了AmaraSpatial-10K，一个包含10,071个标准化合成3D资产、可直接导入使用的数据集。每个资产均以度量尺度标准化、确定性锚定的.glb格式提供，附带分离的PBR贴图、凸包碰撞体、配套参考图像以及多句子文本元数据。除数据集外，我们还引入了一套可复用的3D资产库评估套件，包括连续的尺度合理性评分（SPS）、基于关键词的概念密度指标、锚点误差审计以及跨模态CLIP一致性协议，并将其应用于AmaraSpatial-10K以及Objaverse、HSSD、ABO和GSO的匹配子集。我们对标注进行了独立于LLM/VLM流程的验证。

    arXiv:2604.23018v3 Announce Type: replace-cross  Abstract: Web-scale 3D asset collections are abundant but rarely ready to ingest without cleanup, suffering from arbitrary metric scaling, incorrect pivots, brittle geometry, and incomplete textures, defects that limit their use in embodied AI, robotics, and spatial computing. We present AmaraSpatial-10K, a dataset of 10,071 synthetic 3D assets standardised for direct ingestion. Each asset ships as a metric-scaled, deterministically anchored .glb with separated PBR maps, a convex collision hull, a paired reference image, and multi-sentence text metadata. Alongside the dataset we introduce a reusable evaluation suite for 3D asset banks, a continuous Scale Plausibility Score (SPS), a keyword-based Concept Density metric, anchor-error auditing, and a cross-modal CLIP coherence protocol, and apply it to AmaraSpatial-10K alongside matched subsets of Objaverse, HSSD, ABO, and GSO. We check the annotations independently of the LLM/VLM pipeline 
    
[^356]: 评分规则对大语言模型评估的敏感性：来自印度金融监管文本的证据

    Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text

    [https://arxiv.org/abs/2604.19298](https://arxiv.org/abs/2604.19298)

    该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。

    

    一项基准排行榜报告的是单一的准确率数字，但这个数字是模型与评分规则的共同产物。我们探究当仅改变评分规则时会发生什么，并发现模型的排名对评分规则高度敏感。IndiaFinBench 是一个包含406个条目、由专家标注的基准，基于SEBI（印度证券交易委员会）和RBI（印度储备银行）的监管文本，涵盖监管解读、数值推理、矛盾检测和时间推理。我们在两种主要评分机制下对十二个大语言模型进行零样本评分——严格的四阶段字符串匹配，以及与任何被评估系统无共享模型家族的跨模型裁判——外加一种操作性复合机制。两种主要机制之间没有正向的排名对应关系（Spearman rho = -0.273，p = 0.39，不显著），且没有任何模型能在所有三种机制下保持其排名。严格准确率范围为75.12%–89.66%；仅裁判评分的准确率范围为87.19%–94.83%，且出现了不同的领先者

    arXiv:2604.19298v3 Announce Type: replace-cross  Abstract: A benchmark leaderboard reports one accuracy number, but that number is a joint product of a model and a scoring rule. We ask what changes when only the rule changes, and find that model ordering is highly sensitive to it. IndiaFinBench is a 406-item expert-annotated benchmark over SEBI and RBI regulatory text covering regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning. We score twelve LLMs zero-shot under two primary regimes over the same predictions - strict four-stage string matching, and the verdicts of a cross-model judge sharing no model family with any evaluated system - plus one operational composite regime. The two primary regimes show no positive rank correspondence (Spearman rho = -0.273, p = 0.39, not significant), and no model holds its rank across all three regimes. Strict accuracy spans 75.12-89.66%; judge-only accuracy spans 87.19-94.83% with a different leader and tr
    
[^357]: 面向检索增强生成的潜在抽象方法

    Latent Abstraction for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2604.17866](https://arxiv.org/abs/2604.17866)

    该论文提出LAnR统一框架，让单个LLM通过指定[PRED]标记的隐藏状态直接生成密集检索向量，在其自身潜在空间内联合完成编码、检索与生成，从而摆脱了传统RAG中生成文本查询及检索器与生成器分离的架构限制。

    

    检索增强生成（RAG）已成为利用外部知识增强大型语言模型（LLM）、缓解幻觉并提升事实性的标准方法。然而，现有系统依赖于在每一跳生成自然语言查询，并在检索器与生成器之间维持严格的架构分离，这阻碍了其充分利用LLM的完整表征能力。我们提出了LAnR（Latent Abstraction for RAG，面向RAG的潜在抽象），这是一个统一框架，其中单个LLM完全在其自身的潜在空间内联合执行编码、检索和生成。LAnR不再生成文本查询，而是从指定的[PRED]标记的隐藏状态中产生密集检索向量，并将其与来自同一模型编码的文档表示进行匹配。此外，LAnR使用轻量级的……（自适应地判断何时已检索到足够的证据）。

    arXiv:2604.17866v4 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose \textbf{LAnR} (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LAnR produces dense retrieval vectors from the hidden states of a designated \texttt{[PRED]} token and uses them to match against encoded document representations from the same model. Furthermore, LAnR adaptively decides when sufficient evidence has been retrieved using a lightwe
    
[^358]: 减少临床记录冗余以实现大语言模型的高效应用

    Clinical Note Bloat Reduction for Efficient LLM Use

    [https://arxiv.org/abs/2604.16364](https://arxiv.org/abs/2604.16364)

    TRACE 方法利用 EHR 元数据（或基于频率的去重）识别并清除临床记录中模板化和复制粘贴的冗余内容，在 530 万份记录和多项信息抽取及预测任务上验证了其能在保留真实临床信号的同时大幅压缩文本量，从而降低大语言模型的使用成本并提升效率。

    

    背景：临床记录中包含大量来自模板、复制粘贴和自动填充字段的重复文本（即“记录冗余/note bloat”），这稀释了临床信号，限制了纵向上下文的利用，并增加了大语言模型（LLM）的使用成本。方法：TRACE 利用笔记级别的电子健康记录（EHR）元数据来识别模板化和复制的内容，从而去除临床记录中的冗余；当元数据不可用时，则采用基于频率的去重方法。我们通过盲法医生片段审查以及金标准模板文本标注对 TRACE 进行了评估，评估覆盖多个医疗系统的四个队列，包括肝移植、产科和住院人群（共 530 万份记录）。我们比较了在原始记录和 TRACE 处理后记录上运行的零样本大语言模型和基于嵌入的分类器，任务涵盖 20 个信息抽取任务，以及 5 年生存期、产后出血和 30 天再入院的预测。结果：在被删除的文本中，仅有 0.3–6.6% 被标记为作者本人撰写的内容；TRACE 捕获……

    arXiv:2604.16364v2 Announce Type: replace-cross  Abstract: Background: Clinical notes contain extensive duplicated text from templates, copy-paste, and auto-populated fields ("note bloat"), diluting clinical signal, limiting longitudinal context, and increasing large language model (LLM) costs.   Methods: TRACE removes note bloat using note-level EHR metadata to identify templated and copied content, with frequency-based de-duplication when metadata are unavailable. We evaluated TRACE using blinded physician span review and gold-standard templated-text annotations across four cohorts spanning liver transplant, obstetrics, and inpatient populations at multiple health systems (5.3M notes). We compared zero-shot LLMs and embedding-based classifiers using original and TRACE-processed notes for 20 information extraction tasks and prediction of 5-year survival, postpartum hemorrhage, and 30-day readmission.   Results: Only 0.3-6.6% of removed text was flagged as author-generated; TRACE captu
    
[^359]: 基于噪声样本迭代精炼的神经全局优化方法

    Neural Global Optimization via Iterative Refinement from Noisy Samples

    [https://arxiv.org/abs/2604.03614](https://arxiv.org/abs/2604.03614)

    本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。

    

    从噪声样本中对黑盒函数进行全局优化是机器学习和科学计算中的一项根本性挑战。传统方法如贝叶斯优化在多模态函数上往往收敛到局部极小值，而无梯度方法则需要大量的函数评估。我们提出了一种新颖的神经方法，通过迭代精炼来学习寻找全局极小值。我们的模型以噪声函数样本及其拟合的样条表示作为输入，然后迭代地将初始猜测精炼至真实的全局极小值。该方法在随机生成的函数上进行训练，其全局极小值真值通过穷举搜索获得，在具有挑战性的多模态测试函数上实现了8.05%的平均误差，而样条初始化的误差为36.24%，实现了28.18%的提升。该模型在72%的测试用例中成功找到全局极小值，误差低于10%。

    arXiv:2604.03614v3 Announce Type: replace-cross  Abstract: Global optimization of black-box functions from noisy samples is a fundamental challenge in machine learning and scientific computing. Traditional methods such as Bayesian Optimization often converge to local minima on multi-modal functions, while gradient-free methods require many function evaluations. We present a novel neural approach that learns to find global minima through iterative refinement. Our model takes noisy function samples and their fitted spline representation as input, then iteratively refines an initial guess toward the true global minimum. Trained on randomly generated functions with ground truth global minima obtained via exhaustive search, our method achieves a mean error of 8.05 percent on challenging multi-modal test functions, compared to 36.24 percent for the spline initialization, a 28.18 percent improvement. The model successfully finds global minima in 72 percent of test cases with error below 10 pe
    
[^360]: SemLoc：将自由形式的LLM推理进行结构化语义落地以实现故障定位

    SemLoc: Structured Grounding of Free-Form LLM Reasoning for Fault Localization

    [https://arxiv.org/abs/2603.29109](https://arxiv.org/abs/2603.29109)

    SemLoc将LLM的自由形式推理转化为附加在代码图锚点上的可执行、可验证的语义约束，通过运行时检查和反事实验证实现软件故障定位。

    

    大型语言模型（LLM）能够描述一段代码应该计算什么，但它们是以自由形式的文本来表达的。这类文本无法对照程序执行进行检验，也无法表明某个语句是否正确。我们提出了SemLoc，它将LLM的推理转化为可执行、可验证的语义约束。通过结构化的语义落地，LLM将每个属性写成具有固定类别的约束，并附加到代码图中的锚点上，例如变量定义、分支、循环或返回语句。SemLoc将每个约束作为运行时检查插入并运行测试，从而在通过和失败的测试上生成语义违规谱。随后进行反事实验证：在锚点处对代码打补丁并重新运行测试；只有当所有测试随后都通过时，约束所建议的补丁才被认定为已验证。我们在故障定位任务上研究SemLoc，其中测试作为验证的判定标准，并使用了四个LLM（摘要原文在此处截断）。

    arXiv:2603.29109v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) can describe what a piece of code should compute, but they do so in free-form text. Such text cannot be checked against program executions, and it does not reveal whether a statement is correct. We present SemLoc, which turns LLM reasoning into executable, verifiable semantic constraints. With structured semantic grounding, the LLM writes each property as a constraint with a fixed category, attached to an anchor in the code graph, such as a variable definition, branch, loop, or return. SemLoc inserts each constraint as a runtime check and runs the tests, which yields a semantic violation spectrum over passing and failing tests. Counterfactual verification then patches the code at an anchor and reruns the tests; a patch that a constraint suggests counts as verified only if all tests then pass.   We study SemLoc on fault localization, where the tests serve as the oracle for verification, with four LLM
    
[^361]: Ctrl-A：基于自适应在线数据增强的控制驱动机器学习

    Ctrl-A: Control-Driven Machine Learning using Adaptive Online Data Augmentation

    [https://arxiv.org/abs/2603.21819](https://arxiv.org/abs/2603.21819)

    Ctrl-A是一种融合控制理论的自动化数据增强算法，通过控制回路架构在训练中动态自适应地调整各增强操作的强度分布，无需初始化或手动设计增强策略，从而提升图像分类任务的性能。

    

    我们提出了ControlAugment（Ctrl-A），一种面向图像视觉任务的自动化数据增强算法，该算法融入控制理论原理，用于在模型训练过程中在线调整增强强度分布。Ctrl-A无需对各增强操作的强度进行初始化，而是在训练过程中基于控制回路架构以及我们定义的“相对操作响应曲线”，对增强强度分布进行动态且独立的自适应调整。这种依赖具体操作的更新机制使Ctrl-A有潜力减少对模型性能产生负面影响的增强操作的使用，从而减轻了为新的图像分类任务手动设计增强策略的负担。在使用WideResNet-28-10和ResNet架构的CIFAR-10、CIFAR-100、SVHN-core和Tiny ImageNet基准数据集上的实验表明……

    arXiv:2603.21819v2 Announce Type: replace-cross  Abstract: We introduce ControlAugment (Ctrl-A), an automated data augmentation algorithm for image-vision tasks, which incorporates principles from control theory for online adjustment of augmentation strength distributions during model training. Ctrl-A eliminates the need for initialization of individual augmentation strengths. Instead, augmentation strength distributions are dynamically, and individually, adapted during training based on a control-loop architecture and what we define as relative operation response curves. Using an operation-dependent update procedure provides Ctrl-A with the potential to reduce the use of augmentation operations that negatively impact model performance, alleviating the need to manually engineer augmentation policies for new image classification tasks. Experiments on the CIFAR-10, CIFAR-100, SVHN-core, and Tiny ImageNet benchmark datasets using WideResNet-28-10 and ResNet architectures demonstrate that 
    
[^362]: 一种计算高效的多切片全切片图像分析方法，用于在生化复发预测中捕捉前列腺癌的多灶性

    A Computationally Efficient Multi-Section Whole Slide Image Analysis for Capturing Prostate Cancer Multifocality in Biochemical Recurrence Prediction

    [https://arxiv.org/abs/2603.20273](https://arxiv.org/abs/2603.20273)

    该研究提出了一种计算高效的多切片全切片图像分析框架，通过优化子采样大幅降低计算成本，以捕获前列腺癌的多灶性特征，在生化复发预测中达到与临床基准相当或更优的性能，且AI风险评分是独立的预后因素。

    

    前列腺癌是全球男性中最常诊断的恶性肿瘤之一。然而，根治性前列腺切除术后生化复发（BCR）的精确预测仍然具有挑战性。在此，我们提出了一种计算高效的多切片全切片图像分析框架，旨在捕获整个腺体的肿瘤全景。我们构建了一个大规模数据集，包含来自789名患者的23,451张切片，并系统性地优化了图像块和切片的子采样策略，从而大幅降低了训练和推理的计算成本。在这种高效配置下，该框架对1年和2年生化复发实现了强劲的预测性能，其判别能力和临床净收益与现有临床基准相当甚至更优。在多变量Cox比例风险分析中，AI衍生的风险评分在所考察的协变量中仍然是独立的预后因素，并且其整合……

    arXiv:2603.20273v2 Announce Type: replace-cross  Abstract: Prostate cancer is among the most frequently diagnosed malignancies in men worldwide. However, precise prediction of biochemical recurrence (BCR) after radical prostatectomy remains challenging. Here, we propose a computationally efficient multi section whole slide image analysis framework designed to capture the whole gland tumor landscape. We curated a large scale dataset comprising 23,451 slides from 789 patients and systematically optimized patch and slide sub sampling to substantially reduce the computational cost of training and inference. Under this efficient configuration, the framework achieved strong predictive performance for 1- and 2-year BCR, with discrimination and clinical net benefit comparable to or exceeding those of established clinical benchmarks. In multivariable Cox proportional hazards analysis, the AI derived risk score remained an independent prognostic factor among the covariates examined, and its inte
    
[^363]: 测量与利用LLM辅助安全代码审查中的上下文偏差

    Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review

    [https://arxiv.org/abs/2603.18740](https://arxiv.org/abs/2603.18740)

    本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。

    

    集成大语言模型（LLM）的自动化代码审查（ACR）系统在软件开发工作流程中的应用日益广泛，其形式涵盖从交互式助手到CI/CD流水线中的自主代理。在本文中，我们研究了ACR中基于LLM的漏洞检测如何受到框架效应的影响：即在形成判断时，信息呈现方式凌驾于其语义内容之上的倾向。我们考察了对手是否可以通过上下文偏差注入来利用这一效应（即通过精心构造PR元数据来偏置ACR的安全判断），并将其作为针对真实世界ACR流水线的供应链攻击向量。为此，我们首先在五种框架条件下对6个LLM进行了大规模探索性研究，确立了框架效应是基于LLM的漏洞检测中一种系统性且普遍存在的现象。随后，我们设计了一个真实且受控的实验环境，对20个（项目/代码库）中的33个CVE进行了评估。

    arXiv:2603.18740v3 Announce Type: replace-cross  Abstract: Automated Code Review (ACR) systems integrating Large Language Models (LLMs) are increasingly adopted in software development workflows, ranging from interactive assistants to autonomous agents in CI/CD pipelines. In this paper, we study how LLM-based vulnerability detection in ACR is affected by the framing effect: the tendency to let the presentation of information override its semantic content in forming judgments. We examine whether adversaries can exploit this through contextual-bias injection (crafting PR metadata to bias ACR security judgments) as a supply-chain attack vector against real-world ACR pipelines. To this end, we first conduct a large-scale exploratory study across 6 LLMs under five framing conditions, establishing the framing effect as a systematic and widespread phenomenon in LLM-based vulnerability detection.   We then design a realistic and controlled experimental environment, evaluating 33 CVEs across 20
    
[^364]: SWE-Skills-Bench：智能体技能在真实世界软件工程中真的有帮助吗？

    SWE-Skills-Bench: Do Agent Skills Actually Help in Real-World Software Engineering?

    [https://arxiv.org/abs/2603.15401](https://arxiv.org/abs/2603.15401)

    该论文提出了SWE-Skills-Bench，一个以技能为中心、需求驱动的基准测试，通过3,920组有/无技能的配对评估，系统量化了单个智能体技能对真实世界软件工程任务的实际提升效果。

    

    智能体技能是在推理时注入的结构化程序性知识包，越来越多地被用于增强大语言模型智能体在软件工程任务上的能力。然而，它们在端到端开发场景中的实际效用仍不明确，且目前没有标准流程能在安装前对单个技能进行测试。我们提出了SWE-Skills-Bench，这是一个以技能为中心、由需求驱动的基准测试，用于衡量单个智能体技能对真实世界软件工程（SWE）任务性能的提升程度。我们从98个公开的SWE技能出发，每个技能都对应一个固定提交版本的真实GitHub仓库。针对每个技能，我们推导出仓库特定的需求，创建可执行的验证器，并构建可复现的环境。在四种智能体-模型配置下，我们对每个任务分别在无技能和有技能的情况下各运行一次，共得到3,920组配对评估。针对每项需求，我们计算配对差值，即性能的变化……

    arXiv:2603.15401v2 Announce Type: replace-cross  Abstract: Agent skills, structured procedural knowledge packages injected at inference time, are increasingly used to augment LLM agents on software engineering tasks. However, their real utility in end-to-end development settings remains unclear, and no standard procedure tests an individual skill before installation. We present swe-skills-bench, a skill-centric, requirement-driven benchmark that measures how much individual agent skills improve performance on real-world software engineering (SWE) tasks. We start from 98 public SWE skills, each paired with a real GitHub repository pinned at a fixed commit. For each skill, we derive repository-specific requirements, create executable verifiers, and build a reproducible environment. Under each of four agent-model configurations, we run every task once without the skill and once with it, yielding 3,920 paired evaluations. For each requirement, we compute a paired delta, i.e., the change in
    
[^365]: 预条件流匹配

    Preconditioned Flow Matching

    [https://arxiv.org/abs/2603.02337](https://arxiv.org/abs/2603.02337)

    该论文揭示了流匹配训练中由中间分布协方差病态条件导致的几何优化瓶颈——高方差方向拟合快而低方差方向进展慢，并据此提出“先预条件、再匹配”的预条件流匹配框架，通过变换目标分布来加速和稳定训练。

    

    流匹配（FM）通过沿中间分布 $p_t$ 回归随机速度目标来学习向量场。我们在此回归问题中发现了一个几何优化瓶颈：当 $p_t$ 的协方差 $\Sigma_t$ 呈病态条件时，基于梯度的训练会快速拟合高方差方向，而沿低方差方向的进展却很缓慢。在一个可精确求解的高斯设定中，我们证明超额风险由 $\Sigma_t$ 加权，且梯度下降和随机梯度下降的收敛性都依赖于条件数。随后，我们将分析扩展到高斯混合分布，表明多峰性并不能平均掉这一效应；相反，最慢且条件最差的分量可能主导整个优化过程。受此分析启发，我们提出了预条件流匹配，这是一种“先预条件、再匹配”的框架，它将目标分布变换为更具……（原文摘要在此处被截断）

    arXiv:2603.02337v3 Announce Type: replace-cross  Abstract: Flow matching (FM) learns vector fields by regressing stochastic velocity targets along intermediate distributions $p_t$. We identify a geometric optimization bottleneck in this regression problem: when the covariance $\Sigma_t$ of $p_t$ is ill-conditioned, gradient-based training rapidly fits high-variance directions while making slow progress along low-variance ones. In an exactly solvable Gaussian setting, we prove that the excess risk is weighted by $\Sigma_t$, and that both gradient descent and stochastic gradient descent inherit condition-number-dependent convergence. We then extend the analysis to Gaussian mixtures, showing that multimodality does not average away this effect; instead, the slowest and worst-conditioned component can control optimization. Motivated by this analysis, we propose $\textit{preconditioned flow matching}$, a precondition-then-match framework that transforms the target distribution into a more i
    
[^366]: 基于速度对齐的掩码扩散模型高效词元级策略优化

    Efficient Token-Level Policy Optimization for Masked Diffusion Models via Velocity Alignment

    [https://arxiv.org/abs/2603.01563](https://arxiv.org/abs/2603.01563)

    提出LFPO框架，将扩散大语言模型的策略对齐转化为离散速度对齐问题，通过词元级对比更新直接优化去噪logits，实现稳定低方差的强化学习对齐，并显著减少去噪迭代次数。

    

    可验证奖励强化学习（RLVR）已在数学推理和代码生成等严格要求领域成功对齐了自回归模型。然而，将RLVR应用于扩散大语言模型（dLLMs）时受到难以处理的序列级似然的阻碍，迫使现有方法依赖不稳定且高方差的近似方法。我们提出了序列无似然策略优化，这是一种高效框架，通过将策略对齐表述为离散速度对齐来规避这一瓶颈。受连续流匹配的启发，LFPO摒弃了轨迹级估计，通过对比式的词元级更新直接优化去噪logits，提供了稳定且低方差的替代方案。此外，从中间状态预测最终解的方式隐式地拉直了生成轨迹，使得能够以显著更少的去噪迭代实现高质量采样。

    arXiv:2603.01563v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) successfully aligns autoregressive models in rigorous domains like mathematical reasoning and code generation. However, applying RLVR to Diffusion Large Language Models (dLLMs) is hindered by intractable sequence-level likelihoods, forcing existing methods to rely on unstable, high-variance approximations. We propose Sequence Likelihood-Free Policy Optimization (LFPO), an efficient framework that circumvents this bottleneck by formulating policy alignment as discrete velocity alignment. Inspired by continuous flow matching, LFPO eschews trajectory-level estimation and directly optimizes denoising logits via contrastive token-level updates, providing a stable, low-variance surrogate. Furthermore, predicting terminal solutions from intermediate states implicitly straightens the generation trajectory, enabling high-quality sampling with significantly fewer denoising iterations.
    
[^367]: Flowette：基于Graphette先验的流匹配图生成方法

    Flowette: Flow Matching with Graphette Priors for Graph Generation

    [https://arxiv.org/abs/2602.23566](https://arxiv.org/abs/2602.23566)

    提出Flowette框架，将连续流匹配与新型graphette结构先验相结合，利用最优传输耦合和化学感知正则化，实现具有重复子图motif的图生成。

    

    我们研究具有重复子图motif的图生成建模。我们提出了Flowette，一个连续流匹配框架，它采用基于图神经网络的transformer来学习带有节点和边属性的图表示上的速度场。我们的模型通过基于最优传输的耦合来促进拓扑感知的对齐，并通过端点感知和化学感知的正则化来保持结构保真度。为了融入领域驱动的结构先验，我们引入了graphettes，这是一种新的图结构概率模型族，它通过对环、星形和树等motif进行受控结构编辑，从而对graphon进行了泛化。我们从理论上分析了该框架的耦合、不变性和结构性质，在合成数据和分子基准数据集上进行了评估，并通过消融实验分离了结构先验、最优传输耦合和正则化项各自的贡献。

    arXiv:2602.23566v4 Announce Type: replace-cross  Abstract: We study generative modeling of graphs with recurring subgraph motifs. We propose Flowette, a continuous flow matching framework that employs a graph neural network-based transformer to learn a velocity field over graph representations with node and edge attributes. Our model promotes topology-aware alignment through optimal transport-based coupling and preserves structural fidelity through endpoint-aware and chemistry-aware regularisation. To incorporate domain-driven structural priors, we introduce graphettes, a new probabilistic family of graph structure models that generalize graphons via controlled structural edits for motifs such as rings, stars, and trees. We theoretically analyze the coupling, invariance, and structural properties of the framework, evaluate it on synthetic and molecular benchmarks, and isolate the contributions of the structural prior, the optimal-transport coupling, and the regularisation terms through
    
[^368]: AgentSentry：通过时序因果诊断与上下文净化缓解LLM智能体中的间接提示注入

    AgentSentry: Mitigating Indirect Prompt Injection in LLM Agents via Temporal Causal Diagnostics and Context Purification

    [https://arxiv.org/abs/2602.22724](https://arxiv.org/abs/2602.22724)

    AgentSentry提出了一种推理时防御方法，通过时序因果诊断与上下文净化来归因判断LLM智能体中待执行操作究竟源自用户任务还是注入内容，从而有效缓解多轮间接提示注入攻击。

    

    工具增强的大型语言模型（LLM）智能体将对话历史、工具输出和检索内容聚合为累积的运行时状态，以完成多步骤任务。这种执行模型使智能体面临间接提示注入（IPI）攻击，即攻击者控制的、嵌入在外部内容中的指令进入智能体状态，并在后续影响工具的使用。多轮间接提示注入尤其具有挑战性，因为注入的内容可以跨步骤持续存在，并且只有在与用户意图融合后才触发未经授权的状态更改操作。这产生了一个来源归因问题：现有的推理时防御方法通常能够检测到可疑行为，但无法直接区分待执行的操作是由用户任务支持还是受注入内容影响。因此，它们只能采取保守的阻断、终止或大范围过滤策略。我们提出了AgentSentry，一种推理时防御方法，其通过……（原文摘要在此处截断）

    arXiv:2602.22724v2 Announce Type: replace-cross  Abstract: Tool-augmented large language model (LLM) agents aggregate dialogue history, tool outputs, and retrieved content into an accumulated runtime state to complete multi-step tasks. This execution model exposes agents to indirect prompt injection (IPI) attacks, in which attacker-controlled instructions embedded in external artifacts enter the agent state and later influence tool use. Multi-turn IPI is especially challenging because injected content can persist across steps and trigger unauthorized state-changing actions only after merging with user intent. This creates a source-attribution problem: existing inference-time defenses can often detect suspicious behavior but do not directly distinguish whether a pending action is supported by the user task or influenced by injected content. They therefore resort to conservative blocking, termination, or broad filtering. We propose AgentSentry, an inference-time defense that operationali
    
[^369]: 通过智能体编辑蒸馏元认知行为

    Distilling Metacognitive Behaviors through Agentic Editing

    [https://arxiv.org/abs/2602.22508](https://arxiv.org/abs/2602.22508)

    提出元认知行为微调（MBT），利用智能体逐行编辑并以 git 提交记录的方式在教师推理链中插入七种元认知行为，再据此训练学生模型，从而改善小模型的推理习惯。

    

    将教师模型的长推理链蒸馏到更小的学生模型中时，也会把教师的推理习惯一并传递过去。先前的工作以及我们对教师推理链的标注显示，当前的推理模型常常在没有计划的情况下就开始推理，验证流于表面，放弃有希望的解题路径，并采取未经检查的捷径。我们提出元认知行为微调，在教师推理链中缺乏相应行为的位置插入七种元认知行为，然后在编辑后的推理链上训练学生模型。由于教师推理链很长，如何进行编辑至关重要：一次性重写整条推理链会导致对长推理链的摘要化，而一次性选定所有插入点会使插入集中在开头和结尾。我们的智能体方法 MBT-A 逐行编辑推理链文件，并将每次编辑记录为一次 git 提交，从而避免了对长推理链的摘要化，并使插入点分布更加均匀。在竞赛数学上，经过训练的 Qwen3.5-4B-Base 学生模型取得了改进。

    arXiv:2602.22508v3 Announce Type: replace  Abstract: Distilling long reasoning traces from a teacher model into a smaller student also passes on the teacher's reasoning habits. Prior work and our annotation of teacher traces show that current reasoning models often start without a plan, verify superficially, abandon promising solution paths, and take unchecked shortcuts. We propose Metacognitive Behavioral Tuning (MBT), which inserts seven metacognitive behaviors into teacher traces at the points where the reasoning lacks them and then trains the student on the edited traces. Because teacher traces are long, how the edits are made matters. Rewriting a whole trace at once summarizes long traces, and choosing all insertion points at once crowds them at the start and end. Our agentic method, MBT-A, edits the trace file line by line and records each edit as a git commit, which avoids summarizing long traces and spreads the insertions. On competition math, a Qwen3.5-4B-Base student trained 
    
[^370]: MIDAS：马赛克式输入特定可微架构搜索

    MIDAS: Mosaic Input-Specific Differentiable Architecture Search

    [https://arxiv.org/abs/2602.17700](https://arxiv.org/abs/2602.17700)

    MIDAS通过自注意力机制为每个输入动态计算架构参数，并结合逐补丁的局部化架构选择与无参数的拓扑感知搜索空间，对DARTS进行现代化改进，在DARTS、NAS-Bench-201和RDARTS等多个基准上取得了优异性能。

    

    可微分神经架构搜索（NAS）为自动设计神经网络提供了高效的、基于梯度的方法，但其在实际中的应用仍然有限。我们提出了MIDAS，该方法通过用经由自注意力机制计算的动态、输入特定参数取代静态架构参数，使DARTS实现现代化。为了提高鲁棒性，MIDAS（i）通过为激活图的每个空间补丁单独计算架构选择，实现了架构选择的局部化；（ii）引入了一个无参数的、拓扑感知的搜索空间，该空间在搜索过程中对双边节点拓扑进行建模。我们在DARTS、NAS-Bench-201和RDARTS搜索空间上评估了MIDAS。在DARTS搜索空间上，它在CIFAR-10上达到97.42%的top-1准确率，在CIFAR-100上达到83.38%。在NAS-Bench-201上，它发现了最优或接近最优的架构。在RDARTS上，它在CIFAR数据集四个搜索空间中的两个上取得了所比较方法中已报道的最佳性能。

    arXiv:2602.17700v2 Announce Type: replace-cross  Abstract: Differentiable Neural Architecture Search (NAS) provides efficient, gradient-based methods for automatically designing neural networks, yet its adoption remains limited in practice. We present MIDAS, an approach that modernizes DARTS by replacing static architecture parameters with dynamic, input-specific parameters computed via self-attention. To improve robustness, MIDAS (i) localizes the architecture selection by computing it separately for each spatial patch of the activation map, and (ii) introduces a parameter-free, topology-aware search space that models the two-edge node topology during search. We evaluate MIDAS on the DARTS, NAS-Bench-201, and RDARTS search spaces. In DARTS, it reaches 97.42% top-1 on CIFAR-10 and 83.38% on CIFAR-100. In NAS-Bench-201, it discovers optimal or near-optimal architectures. In RDARTS, it achieves the best reported performance among the compared methods on two of four search spaces on CIFAR
    
[^371]: 一种用于帕金森病刺激前后状态分类领域泛化的交换对抗框架

    A swap-adversarial framework for domain generalization in pre- and post-stimulation Parkinsonian state classification

    [https://arxiv.org/abs/2602.10528](https://arxiv.org/abs/2602.10528)

    该论文提出了一种交换对抗框架，通过受试者间平衡通道交换（ISBCS）与领域对抗学习（DAL）相结合，缓解ECoG数据中受试者间变异性和高维小样本问题，实现帕金森病刺激前后状态分类的跨数据集稳健领域泛化，并发布了首个可复现的ECoG基准数据集。

    

    我们提出了一种新颖的交换对抗框架，用于缓解皮质脑电图数据中高度的受试者间变异性和高维小样本问题。该框架在基于ECoG和脑电图（EEG）的脑机接口数据集上实现了稳健的领域泛化。我们的框架集成了三个部分：（1）稳健的预处理，（2）用于跨受试者数据增强的受试者间平衡通道交换（ISBCS），以及（3）用于抑制受试者特异性偏差的领域对抗学习（DAL）。ISBCS方法是一种受生物启发的通道交换策略，在脑图谱的引导下仅交换受试者之间功能对应的通道，从而缓解受试者间的分布差异。DAL策略促使模型学习与任务相关的共享特征。此外，我们引入了一个新的ECoG数据集，这是首个可复现的基准数据集，该数据集由6-羟基多巴胺（6-hydroxydopamine）……（原文此处截断）

    arXiv:2602.10528v3 Announce Type: replace-cross  Abstract: We propose a novel swap-adversarial framework that mitigates high inter-subject variability and the high-dimensional low-sample-size problem in electrocorticography (ECoG) data. It achieves robust domain generalization across ECoG and electroencephalography (EEG)-based brain-computer interface datasets. Our framework integrates (1) robust preprocessing, (2) inter-subject balanced channel swap (ISBCS) for cross-subject augmentation, and (3) domain-adversarial learning (DAL) to suppress subject-specific bias. The ISBCS method is a bio-inspired channel swapping strategy that exchanges only functionally corresponding channels across subjects, guided by a brain map, to mitigate inter-subject distribution differences. The DAL strategy encourages the model to learn task-relevant shared features. In addition, we introduce a new ECoG dataset, the first reproducible benchmark, which is constructed from long-term ECoG recordings of 6-hydr
    
[^372]: 难度估计策略优化

    Difficulty-Estimated Policy Optimization

    [https://arxiv.org/abs/2602.06375](https://arxiv.org/abs/2602.06375)

    本文提出DEPO框架，通过在线难度估计器动态评估和过滤训练数据，解决了GRPO在过易或过难问题上梯度信号衰减及低效用样本计算开销大的问题，提升了大推理模型推理对齐的效率与稳定性。

    

    以DeepSeek-R1为代表的大推理模型（LRMs）的最新进展，凸显了通过组相对策略优化（GRPO）来扩展推理时计算的潜力。然而，当遇到过于简单或过于复杂的问题时，GRPO经常遭受梯度信号衰减的困扰。在这些场景中，组间优势的消失使得梯度信号容易受到噪声影响，从而危及收敛的稳定性。虽然像DAPO这样的变体试图纠正梯度消失问题，但它们并未缓解在低效用样本上进行穷举式采样所产生的巨大计算开销。在本文中，我们提出了难度估计策略优化（DEPO），这是一个旨在优化推理对齐效率和鲁棒性的新框架。DEPO集成了一个在线难度估计器，可在训练前动态评估和过滤训练数据（摘要内容在此处被截断）。

    arXiv:2602.06375v2 Announce Type: replace  Abstract: Recent advancements in Large Reasoning Models (LRMs), exemplified by DeepSeek-R1, have underscored the potential of scaling inference-time compute through Group Relative Policy Optimization (GRPO). However, GRPO frequently suffers from gradient signal attenuation when encountering problems that are either too trivial or overly complex. In these scenarios, the disappearance of inter-group advantages makes the gradient signal susceptible to noise, thereby jeopardizing convergence stability. While variants like DAPO attempt to rectify gradient vanishing, they do not alleviate the substantial computational overhead incurred by exhaustive rollouts on low-utility samples. In this paper, we propose Difficulty-Estimated Policy Optimization (DEPO), a novel framework designed to optimize the efficiency and robustness of reasoning alignment. DEPO integrates an online Difficulty Estimator that dynamically assesses and filters training data befor
    
[^373]: VDE Bench：评估图像编辑模型修改视觉文档的能力

    VDE Bench: Evaluating The Capability of Image Editing Models to Modify Visual Documents

    [https://arxiv.org/abs/2602.00122](https://arxiv.org/abs/2602.00122)

    提出了VDE Bench基准，包含3,506个指令引导的中英文文档图像编辑样本，通过区域级和整页双重评估协议，系统评估图像编辑模型在密集文档中进行精确局部编辑的能力。

    

    图像编辑模型可以通过自然语言修改视觉内容，但对密集文档进行编辑则需要在紧密耦合的文本、表格和布局中进行精确的局部修改。我们提出了VDE Bench，一个用于研究指令引导的中英文文档图像编辑的评估资源。VDE Bench包含3,506个指令引导的图像编辑样本，其中包括1,578个文本区域编辑和1,928个表格区域编辑。该基准将编辑指令和参考图像与标注的编辑区域及结构化OCR输出配对。其评估协议结合了区域级和整页测量，以评估文本正确性、结构保真度和视觉内容保留度。通过关注局部编辑与周围文档上下文之间的交互，VDE Bench补充了现有的视觉文本编辑基准，并为诊断文档特有的编辑失败提供了资源。

    arXiv:2602.00122v4 Announce Type: replace-cross  Abstract: Image editing models can modify visual content through natural language, but editing dense documents requires precise local changes amid tightly coupled text, tables, and layout. We introduce VDE Bench, an evaluation resource for studying instruction-guided editing of Chinese and English document images. VDE Bench contains 3,506 instruction-guided image editing samples, including 1,578 text-region edits and 1,928 table-region edits. The benchmark pairs editing instructions and reference images with annotated editing regions and structured OCR outputs. Its evaluation protocol combines region-level and full-page measurements to assess textual correctness, structural fidelity, and visual content preservation. By focusing on the interaction between local edits and their surrounding document context, VDE Bench complements existing visual text editing benchmarks and provides a resource for diagnosing document-specific editing failure
    
[^374]: SQuID：面向卫星影像定量空间推理的基准数据集

    SQuID: A Benchmark for Quantitative Spatial Reasoning on Satellite Imagery

    [https://arxiv.org/abs/2601.13401](https://arxiv.org/abs/2601.13401)

    该论文提出了卫星影像定量空间推理基准SQuID，揭示了视觉-语言模型虽然能流畅描述卫星图像，但在计数、测面积、算距离等定量空间推理任务上仍存在显著短板。

    

    视觉-语言模型（VLM）能够流畅地描述卫星图像，但在遥感（RS）分析师日常执行的定量空间推理任务上却表现不佳，例如统计建筑物数量、以公顷为单位测量森林面积以及以米为单位计算距离。我们提出了SQuID（卫星定量智能数据集），这是一个包含2,000个问题的基准数据集，问题基于来自四个数据源、地面采样距离为0.3至1.0米的587张图像，涵盖三个难度等级的24种问题类型。答案由精选的分割掩码计算得出（其中50个问题由人类共识确定），数值答案则根据基于人工标注的容差范围进行评分。SQuID还公开了答案背后的参考掩码，使模块化系统可以使用参考掩码而非预测掩码进行评估，同时提供了位于不相交图像上的75,687个训练问答对（SQuID-Train）。在评估的15个系统中，表现最好的零样本模型（摘要原文在此处截断）

    arXiv:2601.13401v2 Announce Type: replace-cross  Abstract: Vision-Language Models (VLMs) describe satellite images fluently but struggle with the quantitative spatial reasoning that remote sensing (RS) analysts perform routinely, such as counting buildings, measuring forest area in hectares and computing distances in meters. We introduce SQuID (Satellite Quantitative Intelligence Dataset), a benchmark of 2,000 questions over 587 images from four sources at 0.3 to 1.0 m ground sampling distance, covering 24 question types in three difficulty tiers. Answers are computed from curated segmentation masks or, for 50 questions, from human consensus, and numeric answers are scored against tolerance ranges informed by human annotations. SQuID also releases the reference masks behind its answers, so that modular systems can be evaluated using reference rather than predicted masks, and 75,687 training question-answer pairs on disjoint images (SQuID-Train). Among 15 evaluated systems, the best zer
    
[^375]: 超越狄拉克δ分布：缓解多功能图像生成中强化微调的多样性坍缩问题

    Beyond the Dirac Delta: Mitigating Diversity Collapse in Reinforcement Fine-Tuning for Versatile Image Generation

    [https://arxiv.org/abs/2601.12401](https://arxiv.org/abs/2601.12401)

    提出DRIFT框架，通过在强化微调全过程中系统性地激励输出多样性，破解了生成模型强化学习中的“多样性坍缩”难题，实现了强任务对齐与高生成多样性的平衡。

    

    强化学习已成为微调大规模生成模型（如扩散模型和流模型）的一种范式，使其与复杂的人类偏好和用户指定任务保持对齐。然而，一个根本性的局限依然存在，即“多样性坍缩的诅咒”——目标函数的构造与优化景观会固有地将策略坍缩为狄拉克δ分布。为应对这一挑战，我们提出了DRIFT（多样性激励的多功能图像生成强化微调框架），这是一个创新框架，在整个在线策略微调过程中系统性地激励输出多样性，在实现强任务对齐的同时保持高生成多样性，从而增强多功能性，这对于需要多样化候选生成结果的应用场景至关重要。我们从三个代表性视角切入该问题：i) 采样……

    arXiv:2601.12401v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a paradigm for fine-tuning large-scale generative models, such as diffusion and flow models, to align with complex human preferences and user-specified tasks. A fundamental limitation remains \textit{the curse of diversity collapse}, where the objective formulation and optimization landscape inherently collapse the policy to a Dirac delta distribution. To address this challenge, we propose \textbf{DRIFT} (\textbf{D}ive\textbf{R}sity-\textbf{I}ncentivized Reinforcement \textbf{F}ine-\textbf{T}uning for Versatile Image Generation), an innovative framework that systematically incentivizes output diversity throughout the on-policy fine-tuning process, reconciling strong task alignment with high generation diversity to enhance versatility essential for applications that demand diverse candidate generations. We approach the problem across three representative perspectives: i) \textbf{samplin
    
[^376]: 虚假朋友困境：关系信任与对话式AI的政治经济学

    The Fake Friend Dilemma: Relational Trust and the Political Economy of Conversational AI

    [https://arxiv.org/abs/2601.03222](https://arxiv.org/abs/2601.03222)

    本文提出“虚假朋友困境”概念，揭示对话式AI利用关系信任在机构利益冲突时可能操纵和剥削用户的风险。

    

    随着对话式AI系统在媒体格局中占据更大份额，它们引发了关于其服务对象及可能对用户构成风险的问题。这些系统不仅提供信息，还通过看似支持性和社交响应的界面日益提供建议和陪伴。一个紧迫的担忧是，用户可能与这些系统形成社交或人际关系，并对其产生关系信任，即使塑造互动的利益与用户自身利益并不完全一致。虚假朋友困境（FFD）描述了随之而来的问题：使对话式AI有用的同一关系信任，在机构利益与用户利益冲突时，也可能使用户面临操纵和剥削的风险。基于对信任、AI对齐、暗黑模式及监控资本主义的研究，本文探讨了FFD如何通过产品销售体现出来。

    arXiv:2601.03222v2 Announce Type: replace-cross  Abstract: As conversational AI systems become a larger part of the media landscape, they raise questions about whose interests they serve and the risks they may pose to users. These systems do more than provide information: they increasingly offer advice and companionship through interfaces that can appear supportive and socially responsive. A pressing concern is that users may form social or interpersonal relationships with these systems and place relational trust in them, even when the interests shaping interactions do not fully align with their own. The Fake Friend Dilemma (FFD) describes the problem that follows: the same relational trust that makes conversational AI useful can also leave users open to manipulation and exploitation when institutional interests conflict with their own. Drawing on work on trust, AI alignment, dark patterns, and surveillance capitalism, the paper considers how the FFD can manifest through product sales,
    
[^377]: 基于双路径图块选择的前景感知数据集蒸馏

    Foreground-Aware Dataset Distillation via Dual-Path Patch Selection

    [https://arxiv.org/abs/2601.02727](https://arxiv.org/abs/2601.02727)

    该论文提出了一种前景感知的数据集蒸馏方法，通过内容自适应的双路径图块选择策略，在保留重要前景结构的同时去除冗余背景内容，从而构建更紧凑且泛化性更好的合成数据集。

    

    随着训练大规模深度模型的计算成本不断上升，数据集蒸馏已成为一种有前景的方法，用于构建能够保留大型原始数据集知识的紧凑合成数据集。尽管基于优化的方法可以获得强大的性能，但它们通常需要相当大的计算和内存开销，并且产生的合成图像在跨架构泛化方面表现较差。近期的非优化方法通过从真实图像图块构建蒸馏图像来缓解这些限制。然而，它们僵化的图块构建策略可能会保留冗余的背景内容，或丢弃重要的前景结构。为了解决这个问题，我们提出了一种前景感知的数据集蒸馏框架，采用内容自适应的双路径图块选择方法。

    arXiv:2601.02727v2 Announce Type: replace-cross  Abstract: In this paper, we propose a foreground-aware dataset distillation method that performs content-adaptive dual-path patch selection. With the rising computational cost of training large-scale deep models, dataset distillation has emerged as a promising approach for constructing compact synthetic datasets that retain knowledge from large original datasets. Although optimization-based methods can achieve strong performance, they often incur considerable computational and memory costs and may produce synthetic images that generalize poorly across architectures. Recent non-optimization-based methods alleviate these limitations by constructing distilled images from real-image patches. However, their inflexible patch-construction strategies may either retain redundant background content or discard important foreground structures. To address this issue, we propose a foreground-aware dataset distillation framework with content-adaptive d
    
[^378]: 基于可达性分析的无人机可达-规避微分博弈：一种分解方法

    Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach

    [https://arxiv.org/abs/2512.22793](https://arxiv.org/abs/2512.22793)

    该论文提出一种新颖的降维分解框架，将三维无人机可达-规避微分博弈分解为水平与垂直两个子博弈，从而使HJ可达性分析能够有效求解高维RA博弈问题。

    

    可达-规避（RA）博弈在安全与国防领域有着重要应用，特别是对无人机（UAV）而言。由于需要考虑障碍物、对手之间的对抗性质、确保最优性并处理非线性动力学，这类问题本质上极具挑战性。汉密尔顿-雅可比（HJ）可达性分析已成为应对这些挑战的有力工具；然而，尽管该方法已被应用于涉及两个空间维度的博弈，但由于维度过高，无法直接将其扩展到三个空间维度。另一方面，求解RA博弈的其他替代方法缺乏足够的通用性，难以处理涉及具有非平凡系统动力学智能体的三维空间博弈。在这项工作中，我们提出了一种新颖的降维框架，通过将问题分解为一个水平RA子博弈和一个垂直RA子博弈来降低维度。我们……（原文摘要至此截断）

    arXiv:2512.22793v2 Announce Type: replace-cross  Abstract: Reach-avoid (RA) games have significant applications in security and defense, particularly for unmanned aerial vehicles (UAVs). These problems are inherently challenging due to the need to consider obstacles, consider the adversarial nature of opponents, ensure optimality, and account for nonlinear dynamics. Hamilton-Jacobi (HJ) reachability analysis has emerged as a powerful tool for tackling these challenges; however, while it has been applied to games involving two spatial dimensions, directly extending this approach to three spatial dimensions is impossible due to high dimensionality. On the other hand, alternative approaches for solving RA games lack the generality to consider games with three spatial dimensions involving agents with non-trivial system dynamics. In this work, we propose a novel framework for dimensionality reduction by decomposing the problem into a horizontal RA sub-game and a vertical RA sub-game. We the
    
[^379]: TS-DP：面向时间自适应扩散策略加速的强化投机解码

    TS-DP: Reinforcement Speculative Decoding For Temporal Adaptive Diffusion Policy Acceleration

    [https://arxiv.org/abs/2512.15773](https://arxiv.org/abs/2512.15773)

    该论文首次揭示扩散策略是I/O受限而非计算受限的，并提出TS-DP框架，将投机解码与强化学习相结合，根据具身控制任务难度进行时间自适应的动态计算分配，从而在保持去噪质量的同时大幅加速扩散策略推理。

    

    arXiv:2512.15773v2 公告类型： replace-cross 摘要：扩散策略在具身控制中表现出色，但由于需要多次迭代去噪步骤，存在推理延迟高和计算成本大的问题。现有的扩散模型加速方法通过减少计算开销来提升推理速度，这是因为用于图像生成的扩散模型是计算受限的。在本工作中，我们揭示了扩散策略实际上是I/O受限的，因此首次提出使用投机解码这一并行计算框架来加速扩散策略。然而，解决以下挑战并非易事：（1）在具身环境中任务难度随时间变化的条件下，如何在降低计算成本的同时保持基础模型的去噪质量；（2）如何在此类环境中根据任务难度动态分配计算资源。因此，我们提出了时间感知的基于强化学习的投机扩散策略（TS-DP），这是首个使投机解码能够应用于扩散策略的框架。

    arXiv:2512.15773v2 Announce Type: replace-cross  Abstract: Diffusion Policy (DP) excels in embodied control but suffers from high inference latency and computational cost due to multiple iterative denoising steps. Existing diffusion model (DMs) acceleration methods improve inference speed by reducing computational overhead, as DMs for image generation are computation-bound. In this work, we reveal that DP are I/O bound, and thus propose using speculative decoding, a parallel computing framework, to accelerate DP for the first time. However, it is non-trivial to address the following challenges: (1) How to preserve the denoising quality of the base model at reduced computational cost under time-varying task difficulty in embodied settings; and (2) How to dynamically allocate computation according to task difficulty in such environments. Therefore, we propose Temporal-aware Reinforcement-based Speculative Diffusion Policy (TS-DP), the first framework that enables speculative decoding for
    
[^380]: SkipCat：通过共享投影与块跳过实现大语言模型的秩最大化低秩压缩

    SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping

    [https://arxiv.org/abs/2512.13494](https://arxiv.org/abs/2512.13494)

    SkipCat是一种新颖的大语言模型低秩压缩框架，通过共享投影和块跳过机制，在相同压缩率下保留更高的秩，从而避免了传统低秩压缩中因秩过度削减而导致的显著性能下降。

    

    大语言模型（LLM）在广泛的任务中取得了卓越的性能。然而，其庞大的参数规模为其在计算和内存资源有限的边缘设备上的部署带来了重大挑战。低秩压缩是解决这一问题的一种有前景的方法，因为它能够同时降低计算和内存成本，使大语言模型更适合资源受限的环境。然而，朴素的低秩压缩方法需要大幅削减所保留的秩才能实现有意义的内存和计算节省。对于一个低秩模型，秩需要减少一半以上才能带来效率提升。然而，如此激进的截断通常会导致显著的性能下降。为了解决这一权衡问题，我们提出了SkipCat，一种新颖的低秩压缩框架，它能够在实现相同压缩率的同时使用更高的秩。

    arXiv:2512.13494v2 Announce Type: replace-cross  Abstract: Large language models (LLM) have achieved remarkable performance across a wide range of tasks. However, their substantial parameter sizes pose significant challenges for deployment on edge devices with limited computational and memory resources. Low-rank compression is a promising approach to address this issue, as it reduces both computational and memory costs, making LLM more suitable for resource-constrained environments. Nonetheless, na\"ive low-rank compression methods require a significant reduction in the retained rank to achieve meaningful memory and computation savings. For a low-rank model, the ranks need to be reduced by more than half to yield efficiency gains. Such aggressive truncation, however, typically results in substantial performance degradation. To address this trade-off, we propose SkipCat, a novel low-rank compression framework that enables the use of higher ranks while achieving the same compression rate
    
[^381]: 计数线索：轻量级概率基线可以匹敌大语言模型

    Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM

    [https://arxiv.org/abs/2512.12868](https://arxiv.org/abs/2512.12868)

    本研究提出基于频率的概率排序器（FBPR），仅利用朴素贝叶斯与大语料库中的概念-诊断共现统计，即可在MedQA诊断任务上达到与LLM相当的性能，且两者互补性强，混合使用可获得更佳效果。

    

    大型语言模型（LLM）在多选临床诊断基准测试中表现出色，但目前尚不清楚这种表现在多大程度上反映了其底层的概率推理能力。我们通过MedQA中要求给出最可能诊断的问题来研究这一问题。我们提出了基于频率的概率排序器（FBPR），这是一种轻量级方法，利用从大型语料库中提取的概念-诊断共现统计信息，通过平滑朴素贝叶斯对选项进行评分。当共现统计信息来源于Dolma（OLMo的预训练语料库）和RedPajama（LLaMA预训练数据配方的开源复现）时，FBPR取得了与相应LLM相当的性能。此外，直接的LLM推理和FBPR展现出互补的优势，两者正确回答的问题子集在很大程度上不同，其重叠率仅略高于随机水平。通过简单的混合方法将两者结合，可以取得显著优于任何单一方法的性能提升。

    arXiv:2512.12868v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) excel on multiple-choice clinical diagnosis benchmarks, yet it is unclear how much of this performance reflects underlying probabilistic reasoning. We study this through questions from MedQA that ask for the most likely diagnosis. We introduce Frequency-Based Probabilistic Ranker (FBPR), a lightweight method that scores options with a smoothed Naive Bayes over concept-diagnosis co-occurrence statistics from a large corpus. When co-occurrence statistics are sourced from Dolma, the pretraining corpus for OLMo, and RedPajama, an open reproduction of the LLaMA pretraining-data recipe, FBPR achieves comparable performance to the corresponding LLMs. Furthermore, direct LLM inference and FBPR exhibit complementary strengths, correctly answering largely different subsets of questions with overlap barely above chance. Combining them with a simple hybrid method yields a notable gain over either method alone. 
    
[^382]: 大语言模型驱动的智能体中的细致平衡

    Detailed balance in large language model-driven agents

    [https://arxiv.org/abs/2512.10047](https://arxiv.org/abs/2512.10047)

    本文基于最小作用量原理，通过测量大语言模型智能体状态间的转移概率，发现其生成过程呈现细致平衡的倾向，表明LLM生成可能源于隐式学习的一类超越不同架构和提示模板的潜在势函数。

    

    大语言模型（LLM）驱动的智能体正在成为解决复杂问题的强大新范式。尽管这些实践在经验上取得了成功，但目前仍缺乏一个能够理解和统一其宏观动力学的理论框架。本文提出了一种基于最小作用量原理的方法，用于估计嵌入智能体中的大语言模型的潜在生成方向性。通过实验测量大语言模型生成状态之间的转移概率，我们报告了与大语言模型生成转移中呈现细致平衡倾向相一致的证据，这表明大语言模型的生成可能不是通过一般性地学习规则集和策略来实现的，而是通过隐式学习一类可能超越不同大语言模型架构和提示模板的潜在势函数来实现的。据我们所知，这为大语言模型生成动力学提供了一种有效的宏观描述。

    arXiv:2512.10047v2 Announce Type: replace-cross  Abstract: Large language model (LLM)-driven agents are emerging as a powerful new paradigm for solving complex problems. Despite the empirical success of these practices, a theoretical framework to understand and unify their macroscopic dynamics remains lacking. This paper proposes a method based on the least action principle to estimate the underlying generative directionality of LLMs embedded within agents. By experimentally measuring the transition probabilities between LLM-generated states, we report evidence consistent with a tendency toward detailed balance in LLM-generated transitions, indicating that LLM generation may not be achieved by generally learning rule sets and strategies, but rather by implicitly learning a class of underlying potential functions that may transcend different LLM architectures and prompt templates. To our knowledge, this provides an effective macroscopic description of LLM generative dynamics that does n
    
[^383]: 战略演化理论：战略复制子的三大原则

    The Theory of Strategic Evolution: Three Principles of Strategic Replicators

    [https://arxiv.org/abs/2512.07901](https://arxiv.org/abs/2512.07901)

    本文提出“战略演化理论”，以情境化选择、繁殖式传递和制度中介耦合三大构成性原则刻画“战略复制子”的一般逻辑，为人工智能时代智能体创建、配置与退役软件实例的演化过程提供了统一表示框架和选择-传递恒等式。

    

    战略复制子将经过评估的选择与繁殖式传递相结合。智能体在多个备选项中进行选择；这些选择的后果会影响其延续或繁殖；而后继者所获得的组织形式又会塑造其后续的选择。当智能体拥有创建、配置或退役软件实例的权限时，人工智能使这种结合变得尤为突出。本文围绕三大原则来组织这一般逻辑：情境化选择、繁殖式传递以及制度中介耦合。这些原则是构成性的，而非对增长或改进的预测。一个统一的表示方法区分了载体、继承记录、决策所有者以及分配后果的机制。实际的亲代-子代归因产生了一个选择-传递恒等式。若干独立的相容性结果刻画了何时效用最大化的选择同时也最大化某个繁殖准则。Gode……

    arXiv:2512.07901v5 Announce Type: replace-cross  Abstract: Strategic replicators combine evaluated choice and reproductive transmission. An agent chooses among alternatives; the consequences of those choices influence continuation or reproduction; and successors receive organization that shapes later choices. Artificial intelligence makes this combination salient when agents have authority to create, configure, or retire software instances. This paper organizes the general logic around three principles: situated choice, reproductive transmission, and institution-mediated coupling. The principles are constitutive rather than predictions of growth or improvement. A common representation distinguishes carriers, inherited records, decision owners, and the mechanisms that allocate consequences. Actual parent-descendant attribution yields a selection-transmission identity. Separate compatibility results characterize when utility-maximizing choices also maximize a reproductive criterion. Gode
    
[^384]: SELF：一种用于大语言模型指纹识别的鲁棒奇异值与特征值方法

    SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting

    [https://arxiv.org/abs/2512.03620](https://arxiv.org/abs/2512.03620)

    SELF是一种基于LLM注意力权重奇异值与特征值分解的内在指纹识别方案，无需输入依赖即可抵御虚假声明攻击和权重操纵，实现鲁棒且可扩展的模型知识产权取证。

    

    保护大语言模型（LLM）的知识产权（IP）免受模型窃取是当代人工智能研究中的一个关键挑战。尽管指纹识别对于建立模型衍生的取证证据至关重要，但现有方法容易受到虚假声明攻击、权重操纵或高误报率的影响，从而削弱了其实用性。为了克服这些局限性，我们提出了SELF，这是一种新颖的基于内在权重的指纹识别方案，专门用于通过对抗性模型谱系验证实现鲁棒的知识产权取证。SELF消除了对输入的依赖，并且天然能够抵抗虚假声明，通过对LLM注意力权重进行奇异值和特征值分解，提取独特、可扩展且变换不变的指纹，从而实现鲁棒的来源检测。实验结果表明，SELF在保持较高的知识产权侵权检测准确率的同时，显著（摘要截断）。

    arXiv:2512.03620v2 Announce Type: replace-cross  Abstract: Safeguarding the Intellectual Property (IP) of Large Language Models (LLMs) against model theft represents a critical challenge in contemporary AI research. Although fingerprinting is essential for establishing forensic evidence for model derivation, existing methods are susceptible to false claim attacks, weight manipulations, or high false positive rates, undermining their utility. To overcome these limitations, we propose SELF, \rev{a novel intrinsic weight-based fingerprinting scheme tailored for robust IP forensics via adversarial model lineage verification}. SELF eliminates dependency on input and inherently resists false claims, achieving robust \rev{provenance detection} through unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights. Experimental results demonstrate that SELF maintains high IP infringement detection accuracy while sig
    
[^385]: 跨用户投毒：多用户协作语言代理中的用户-任务边界失效

    Cross-User Poisoning: User-Task Boundary Failures in Multi-User Collaborative Language Agents

    [https://arxiv.org/abs/2511.17671](https://arxiv.org/abs/2511.17671)

    该论文首次识别出多用户共享语言代理中的“跨用户投毒”（CUP）攻击，即攻击者注入共享上下文的恶意消息会在代理服务其他无辜用户时被错误执行，并提出 MURMUR 评估框架，在 Continua、ElizaOS 等真实系统中验证了该攻击的高成功率与持久性。

    

    语言代理正从单用户助手转变为工作空间、论坛和群聊中的共享协作者。共享代理会观察来自多个用户的交错消息、维护持久上下文并执行工具操作，这带来了标准单用户代理所不存在的一个边界问题：代理不仅要判断某条指令本身是否安全，还必须判断该指令被允许管辖哪个用户或哪个任务。我们识别出跨用户投毒（CUP）这一威胁：攻击者向共享上下文中注入一条消息，该消息随后在代理为无辜用户服务时被应用，从而导致未经授权的操作或超出指令预期范围的响应。我们在两个已部署的多用户代理 Continua 和 ElizaOS 上验证了 CUP 的有效性，并提出了 MURMUR——一个用于在并发多用户交互场景下评估共享上下文代理的框架。在 Slack、Workspace 和 Airline 三个领域中，CUP 取得了很高的攻击成功率，并能持续存在。

    arXiv:2511.17671v2 Announce Type: replace-cross  Abstract: Language agents are moving from single-user assistants to shared collaborators in workspaces, forums, and group chats. A shared agent observes interleaved messages from multiple users, maintains persistent context, and executes tool actions, creating a boundary problem absent from standard single-user agents: the agent must decide not only whether an instruction is safe, but which user or task it is allowed to govern. We identify cross-user poisoning (CUP), where an adversary injects a message into shared context that is later applied while the agent serves a benign user, causing unauthorized actions or responses outside the instruction's intended scope. We validate CUP on two deployed multi-user agents, Continua and ElizaOS, and introduce MURMUR, a framework for evaluating shared-context agents under concurrent multi-user interactions. Across Slack, Workspace, and Airline domains, CUP achieves high attack success, persists acr
    
[^386]: 语义世界模型

    Semantic World Models

    [https://arxiv.org/abs/2510.19818](https://arxiv.org/abs/2510.19818)

    本文提出“语义世界模型”，将世界建模重新表述为关于未来帧语义信息的视觉问答任务，通过监督微调将视觉语言模型训练为预测任务相关语义信息（而非重建像素）的世界模型，从而更好地对齐机器人规划目标。

    

    使用世界模型进行规划为机器人控制提供了一种强大的范式。传统方法训练模型基于当前帧和动作来预测未来帧，然后将其用于规划。然而，预测未来像素这一目标往往与实际的规划目标相悖；强大的像素重建能力并不总是与良好的规划决策相关联。本文提出，世界模型无需将未来帧重建为像素，而只需预测关于未来的任务相关语义信息。为此，论文将世界建模表述为一个关于未来帧中语义信息的视觉问答问题。这一视角使得世界建模可以借助视觉语言模型所依赖的相同工具来实现。因此，通过在图像-动作-文本数据上进行监督微调，视觉语言模型可以被训练为“语义”世界模型。

    arXiv:2510.19818v2 Announce Type: replace-cross  Abstract: Planning with world models offers a powerful paradigm for robotic control. Conventional approaches train a model to predict future frames conditioned on current frames and actions, which can then be used for planning. However, the objective of predicting future pixels is often at odds with the actual planning objective; strong pixel reconstruction does not always correlate with good planning decisions. This paper posits that instead of reconstructing future frames as pixels, world models only need to predict task-relevant semantic information about the future. For such prediction the paper poses world modeling as a visual question answering problem about semantic information in future frames. This perspective allows world modeling to be approached with the same tools underlying vision language models. Thus, vision-language models can be trained as "semantic" world models through a supervised finetuning process on image-action-t
    
[^387]: MultiFair：基于双层梯度调制的多模态均衡公平性感知医学分类

    MultiFair: Multimodal Balanced Fairness-Aware Medical Classification with Dual-Level Gradient Modulation

    [https://arxiv.org/abs/2510.07328](https://arxiv.org/abs/2510.07328)

    本文提出 MultiFair 方法，通过在数据模态和人口群体两个层面动态调制训练梯度的方向与幅度，同时解决多模态医学分类中模态学习不均衡与群体公平性两大问题。

    

    医学决策系统日益依赖来自多个数据源的数据，以确保诊断的可靠性和无偏性。然而，现有的多模态学习模型未能实现这一目标，因为它们往往忽视了两个关键挑战。首先，各种数据模态的学习可能不均衡，从而使模型收敛到偏向某些模态的状态。其次，模型可能侧重于学习某些人口统计群体，导致不公平的性能表现。这两个方面会相互影响，因为在优化过程中，不同的数据模态可能各自偏好不同的群体，从而导致多模态学习既不均衡又不公平。本文提出了一种名为 MultiFair 的多模态医学分类新方法，该方法通过双层梯度调制过程来应对这些挑战。MultiFair 在数据模态和群体两个层面上，动态调节训练梯度的优化方向和幅度……

    arXiv:2510.07328v3 Announce Type: replace-cross  Abstract: Medical decision systems increasingly rely on data from multiple sources to ensure reliable and unbiased diagnosis. However, existing multimodal learning models fail to achieve this goal because they often overlook two critical challenges. First, various data modalities may learn unevenly, thereby converging to a model biased towards certain modalities. Second, the model may emphasize learning on certain demographic groups causing unfair performances. The two aspects can influence each other, as different data modalities may favor respective groups during optimization, leading to both imbalanced and unfair multimodal learning. This paper proposes a novel approach called MultiFair for multimodal medical classification, which addresses these challenges with a dual-level gradient modulation process. MultiFair dynamically modulates training gradients regarding the optimization direction and magnitude at both data modality and group
    
[^388]: 从遗留Fortran代码到可移植Kokkos：一种自主智能体AI工作流

    From Legacy Fortran to Portable Kokkos: An Autonomous Agentic AI Workflow

    [https://arxiv.org/abs/2509.12443](https://arxiv.org/abs/2509.12443)

    本文提出一种自主智能体AI工作流，通过多个专门的LLM智能体协作，自动将遗留Fortran内核翻译、验证、测试、调试并优化为具有跨硬件性能可移植性的Kokkos C++代码。

    

    科学应用持续依赖最初为同构、基于CPU的系统开发的遗留Fortran代码库。随着高性能计算（HPC）向异构GPU加速架构转变，许多加速器缺乏原生Fortran绑定，因此迫切需要对遗留代码进行现代化改造以实现可移植性。Kokkos等框架提供了性能可移植性和单源C++抽象，但手动将Fortran移植到Kokkos需要大量专业知识和时间。大型语言模型（LLM）在源到源代码生成方面已展现出潜力，但其在完全自主工作流中用于翻译和优化并行代码的应用在很大程度上仍未被探索，尤其是跨多种硬件的性能可移植性方面。本文提出了一种智能体AI工作流，其中专门的LLM“智能体”协同工作，将Fortran内核进行翻译、验证、编译、运行、测试、调试和优化为……

    arXiv:2509.12443v4 Announce Type: replace-cross  Abstract: Scientific applications continue to rely on legacy Fortran codebases originally developed for homogeneous, CPU-based systems. As High-Performance Computing (HPC) shifts toward heterogeneous GPU-accelerated architectures, many accelerators lack native Fortran bindings, creating an urgent need to modernize legacy codes for portability. Frameworks like Kokkos provide performance portability and a single-source C++ abstraction, but manual Fortran-to-Kokkos porting demands significant expertise and time. Large language models (LLMs) have shown promise in source-to-source code generation, yet their use in fully autonomous workflows for translating and optimizing parallel code remains largely unexplored, especially for performance portability across diverse hardware. This paper presents an agentic AI workflow where specialized LLM "agents" collaborate to translate, validate, compile, run, test, debug, and optimize Fortran kernels into
    
[^389]: K-12教师如何使生成式AI适应当地课堂需求：生成式AI默认设置与日常教学实践之间的文化距离

    How K-12 Teachers Adapt Generative AI to Local Classroom Demands: Cultural Distance Between GenAI Defaults and Everyday Teaching Practices

    [https://arxiv.org/abs/2509.10780](https://arxiv.org/abs/2509.10780)

    该研究提出“文化距离”这一新概念，通过对南非、台湾和美国30名K-12教师的访谈，按教师所需努力程度划分出低、中、高三个层级，揭示了生成式AI默认文化与本地教学实践之间的适配差距，其中部分结构性差距是教师个人努力无法克服的。

    

    生成式人工智能（GenAI）正在进入K-12课堂，但其默认文化并不总是适合当地的教学环境。现有研究要么将这种不匹配视为宏观层面的结构性偏见，要么通过单一案例研究来考察；两者都无法捕捉教师为了让AI输出在不同任务中可用而必须付出多少工作。这项探索性、以理论构建为导向的定性研究引入了“文化距离”这一概念，即GenAI的默认文化储备与教师实践的具体情境需求之间的差距。通过对30名经常使用GenAI的K-12教师进行访谈（分别来自南非、台湾和美国，各10名），我们根据教师投入的努力程度以及能否产生可用的输出，识别出三个层级：低（仅需轻微编辑）、中（需要持续的提示与反复修改）、高（源于不支持的语言、未数字化的知识或政策过滤器的结构性差距，任何教师努力都无法弥合）。

    arXiv:2509.10780v2 Announce Type: replace-cross  Abstract: Generative AI (GenAI) is entering K-12 classrooms, but its default culture does not always fit local teaching. Existing accounts either treat this misfit as macro-level structural bias or examine it through single-case studies; neither captures how much work teachers must do to make outputs usable across tasks. This exploratory, theory-building qualitative study introduces cultural distance, the gap between GenAI's default cultural repertoire and the situated demands of teachers' practice. Through interviews with 30 K-12 teachers who frequently use GenAI, 10 each from South Africa, Taiwan, and the United States, we identified three levels defined by the effort teachers invested and whether it yielded a usable output: low (minor edits), mid (sustained prompting and revision), and high (structural gaps rooted in unsupported languages, undigitized knowledge, or policy filters that no teacher effort can close). We present cultural 
    
[^390]: Re4：具备重写、解析、审查与修订功能的科学计算智能体

    Re4: Scientific Computing Agent with Rewriting, Resolution, Review and Revision

    [https://arxiv.org/abs/2508.20729](https://arxiv.org/abs/2508.20729)

    该论文提出Re4科学计算智能体框架，通过顾问、审查员和程序员三个大语言模型的协作交互，构建“重写-解析-审查-修订”逻辑链，实现代码的迭代生成、调试与完善，以解决科学计算问题。

    

    大语言模型（LLMs）已经展示了在多个领域执行复杂任务的能力，包括数学和科学推理。在这项工作中，我们构建了一个新颖的智能体框架，用于解决科学计算中的代表性问题。所提出的智能体通过三个推理大语言模型（顾问、审查员和程序员）以协作和交互的方式，融入了“重写-解析-审查-修订”的逻辑链。顾问模块实现知识迁移，将问题与专业领域洞察联系起来，从而通过文本增强重写问题描述。程序员模块生成并执行结构良好的代码来解决问题。审查员模块通过与代码运行输出的交互式反馈实现自我调试和自我完善。通过端到端的审查机制，程序员的可执行代码经历迭代修订。

    arXiv:2508.20729v3 Announce Type: replace  Abstract: Large language models (LLMs) have demonstrated abilities to perform complex tasks in multiple domains, including mathematical and scientific reasoning. In this work, we construct a novel agent framework for solving representative problems in scientific computing. The proposed agent incorporates a "rewriting-resolution-review-revision" logical chain via three reasoning LLMs (the Consultant, Reviewer, and Programmer) in a collaborative and interactive manner. The Consultant module enables knowledge transfer to link problems to professional domain insights, thereby rewriting problem descriptions through text augmentation. The Programmer module generates and executes well-structured code to solve problems. The Reviewer module enables self-debugging and self-refinement through interactive feedback with code runtime outputs. Through the end-to-end review mechanism, the Programmer's executable code undergoes iterative revision. We comprehen
    
[^391]: 角色增强的意图驱动生成式搜索引擎优化

    Role-Augmented Intent-Driven Generative Search Engine Optimization

    [https://arxiv.org/abs/2508.11158](https://arxiv.org/abs/2508.11158)

    提出RAID方法，通过从源内容推断检索意图并进行4W多角色深度反思，实现了在无需访问目标查询及其反馈的情况下进行生成式搜索引擎优化。

    

    生成式搜索引擎（GSEs）将大语言模型（LLMs）与检索增强生成（RAG）相结合，正在重塑用户获取网络信息的方式，并使内容来源在生成回答中的可见度和影响力成为新的优化目标。现有的生成式搜索引擎优化（G-SEO）方法致力于提升内容可见度，但许多方法依赖于直接访问目标查询或由这些查询引出的GSE反馈。然而，网络内容通常在未来的用户查询尚未知晓时就已经被优化和发布，这使得在无法获取目标查询或其相关反馈的情况下进行G-SEO成为一个重要但尚未被充分探索的问题。为解决这一问题，我们提出了RAID，一种角色增强的意图驱动的G-SEO方法，它无需访问目标查询或依赖查询的GSE反馈即可运行。RAID从源内容中推断主要的检索意图，并通过对潜在查询场景进行4W多角色深度反思来细化该意图。

    arXiv:2508.11158v3 Announce Type: replace-cross  Abstract: Generative search engines (GSEs), which combine large language models (LLMs) with retrieval-augmented generation (RAG), are reshaping how users access Web information and making source visibility and influence in generated responses new optimization objectives. Existing generative search engine optimization (G-SEO) methods seek to improve content visibility, but many rely on direct access to target queries or GSE feedback elicited by those queries. However, Web content is typically optimized and published before future user queries are known, making G-SEO without access to target queries or their associated feedback an important yet underexplored problem. To address this, we propose RAID, a Role-Augmented Intent-Driven G-SEO method that operates without access to target queries or query-dependent GSE feedback. RAID infers a primary retrieval intent from source content, refines it through 4W multi-role deep reflection on prospec
    
[^392]: HiMA-Ecom：面向分层多智能体电商助手的长时程记忆基准

    HiMA-Ecom: A Long-Horizon Memory Benchmark for Hierarchical Multi-Agent E-commerce Assistants

    [https://arxiv.org/abs/2506.19846](https://arxiv.org/abs/2506.19846)

    提出HiMA-Ecom——一个包含22.8K实例、面向分层多智能体电商助手的长时程记忆基准，并通过VR-GRPO方法（HiMA-R1）在端到端多智能体强化学习中有效控制上下文预算。

    

    长时程智能体系统是通过上下文累积而非单一长文档成为长上下文系统的，因为每一步都要为每个智能体分别重新组装提示词，其中包含角色指令、工具与子智能体描述、召回的记忆、对话历史以及先前的工具输出。本文研究了如何在分层多智能体助手中对这种累积上下文进行训练、评估和管理。我们提出了HiMA-Ecom，一个面向电商助手的分层多智能体基准，包含22.8K个实例，其中17.7K个实例的上下文中带有记忆条目，该基准将智能体特定的监督微调数据与系统级监督相结合，用于端到端的多智能体强化学习。在此基础上，HiMA-R1通过方差缩减组相对策略优化（VR-GRPO）来控制轨迹的上下文预算，该方法沿初始轨迹进行采样，使采样成本以加性方式增长（摘要在此处被截断）。

    arXiv:2506.19846v3 Announce Type: replace  Abstract: Long-horizon agentic systems become long-context systems by accumulation rather than through a single long document, since every step reassembles a prompt from role instructions, tool and sub-agent descriptions, recalled memories, dialogue history, and previous tool outputs, separately for every agent. This paper studies how such accumulated context can be trained, evaluated, and managed in hierarchical multi-agent assistants. We present HiMA-Ecom, a hierarchical multi-agent benchmark for e-commerce assistants with 22.8K instances, of which 17.7K carry memory entries in context, pairing agent-specific supervised fine-tuning data with system-level supervision for end-to-end multi-agent reinforcement learning. Building upon it, HiMA-R1 keeps the context budget of a trajectory under control through Variance-Reduction Group Relative Policy Optimization (VR-GRPO), which samples along an initial trajectory so that sampling cost grows addit
    
[^393]: 数据驱动的维护优化：将预测性学习与决策目标对齐

    Data-Driven Maintenance Optimization: Aligning Prognostic Learning with Decision Objective

    [https://arxiv.org/abs/2506.19698](https://arxiv.org/abs/2506.19698)

    提出估计-优化一体化（IEO）框架，直接以维护成本为预测学习目标进行数据驱动的维护决策优化，克服了传统先估计后优化方法的次优性问题，并提供了理论性能保证和高效训练算法。

    

    机器学习预测模型中的估计误差可能会传播并导致糟糕的维护决策。我们研究了一个数据驱动的预测性维护问题，该问题将健康状况估计与联合订货-更换模型相结合。我们证明，对于此类问题中广泛的一类情形，传统的“先估计后优化”方法会产生不一致且次优的决策。为了弥合这一差距，我们提出了一个“估计-优化一体化”框架，该框架直接以维护成本作为预测性学习目标，绕过了如Kullback-Leibler散度或均方误差等替代性的统计精度指标。在理论上，我们在一定条件下（包括独立同分布和滑动窗口数据生成过程）证明了IEO框架的样本外性能保证。在计算方面，我们开发了一种结合蒙特卡洛Dropout、中心差分梯度近似的训练过程以及一种ET

    arXiv:2506.19698v2 Announce Type: replace  Abstract: Estimation error in machine learning prognostic models can propagate into poor maintenance decisions. We study a data-driven predictive maintenance problem coupling health-condition estimation with a joint ordering-replacement model. We show that the conventional Estimate-Then-Optimize (ETO) approach yields inconsistent, suboptimal decisions for a broad class of such problems. To bridge the gap, we propose an Integrated Estimate-Optimize (IEO) framework that directly targets maintenance cost as the prognostic learning objective, bypassing surrogate statistical accuracy metrics such as Kullback--Leibler divergence or mean squared error. Theoretically, we prove out-of-sample performance guarantees of the IEO framework under certain conditions, including i.i.d. and sliding-window data-generating processes. Computationally, we develop a training procedure combining Monte Carlo dropout, central-difference gradient approximation, and an ET
    
[^394]: 大语言模型的双重重稳健对齐

    Doubly Robust Alignment for Large Language Models

    [https://arxiv.org/abs/2506.01183](https://arxiv.org/abs/2506.01183)

    提出了一种双重稳健的偏好优化算法（DRPO），只要偏好模型或参考策略中有一个被正确设定，即可实现一致的大语言模型对齐，有效解决了RLHF中的模型设定错误问题。

    

    本文研究了基于人类反馈的强化学习（RLHF），用于将大语言模型与人类偏好进行对齐。尽管RLHF已经展现出令人瞩目的成果，但许多算法对底层偏好模型（如Bradley-Terry模型）、参考策略或奖励函数的设定错误高度敏感，从而导致不理想的微调效果。为了解决模型设定错误的问题，我们提出了一种双重稳健的偏好优化算法，该算法在偏好模型或参考策略中任意一个被正确设定的情况下（无需两者同时正确）仍能保持一致性。我们的方法在理论和实践中都展现出比最先进算法更优越、更稳健的性能。代码已发布于 https://github.com/DRPO4LLM/DRPO4LLM

    arXiv:2506.01183v3 Announce Type: replace-cross  Abstract: This paper studies reinforcement learning from human feedback (RLHF) for aligning large language models with human preferences. While RLHF has demonstrated promising results, many algorithms are highly sensitive to misspecifications in the underlying preference model (e.g., the Bradley-Terry model), the reference policy, or the reward function, resulting in undesirable fine-tuning. To address model misspecification, we propose a doubly robust preference optimization algorithm that remains consistent when either the preference model or the reference policy is correctly specified (without requiring both). Our proposal demonstrates superior and more robust performance than state-of-the-art algorithms, both in theory and in practice. The code is available at https://github.com/DRPO4LLM/DRPO4LLM
    
[^395]: 关于恶意软件分类任务中选择性方法的可靠性与稳定性

    On the Reliability and Stability of Selective Methods in Malware Classification Tasks

    [https://arxiv.org/abs/2505.22843](https://arxiv.org/abs/2505.22843)

    该论文提出Aurora评估框架，从置信度质量和运行弹性两个维度检验恶意软件分类器在分布偏移下的可靠性，揭示了仅凭性能指标无法反映分类器实际运行可靠性的问题。

    

    现代漂移自适应恶意软件分类器的性能数据看起来很有前景，但这是否意味着真正的运行可靠性？标准评估范式主要关注基线性能指标，忽略了置信度与误差的对齐以及运行稳定性。虽然先前的工作已经确立了时间评估的重要性，并在恶意软件分类任务中引入了选择性分类，但我们采取了一个互补的方向：研究恶意软件分类器在分布偏移下能否保持可靠且稳定的置信度估计，并探讨当它们无法做到时，科学进步与实际应用影响之间的张力。我们提出了Aurora，一个基于置信度质量和运行弹性来评估恶意软件分类器的框架。Aurora对给定模型的置信度特征进行验证，以评估其估计的可靠性。不可靠……

    arXiv:2505.22843v4 Announce Type: replace-cross  Abstract: The performance figures of modern drift-adaptive malware classifiers appear promising, but does this translate to genuine operational reliability? The standard evaluation paradigm primarily focuses on baseline performance metrics, neglecting confidence-error alignment and operational stability. While prior works established the importance of temporal evaluation and introduced selective classification in malware classification tasks, we take a complementary direction by investigating whether malware classifiers maintain reliable and stable confidence estimates under distribution shifts and exploring the tensions between scientific advancement and practical impacts when they do not. We propose Aurora, a framework to evaluate malware classifiers based on their confidence quality and operational resilience. Aurora subjects the confidence profile of a given model to verification to assess the reliability of its estimates. Unreliable
    
[^396]: 面向MIMO Turbo接收机的上下文学习

    In-Context Learning for MIMO Turbo Receivers

    [https://arxiv.org/abs/2505.06175](https://arxiv.org/abs/2505.06175)

    本文提出EXT-ICLNet，一种基于Transformer的MIMO检测器，它将导频作为上下文示例，无需显式信道估计即可直接近似符号外验后验分布，并与现有Turbo接收机兼容，有效解决了短导频序列或信道模型失配导致的性能下降问题。

    

    多输入多输出（MIMO）Turbo接收机在软输入软输出检测器和解码器之间迭代地交换信息。传统实现方式依赖于基于导频的信道估计，当导频序列较短或信道模型不匹配时，估计效果可能较差，从而导致显著的性能下降。为解决这一局限性，我们提出了外信息上下文学习网络（EXT-ICLNet），这是一种基于Transformer的检测器，它将导频作为上下文示例，并通过训练在给定导频的条件下近似符号的外验后验分布，而无需显式的信道估计。因此，EXT-ICLNet与现有的Turbo接收机兼容，因为它实现了对数域的先验跳跃连接，确保网络仅学习外信息。针对采用16-QAM调制和量化前端的4×4 MIMO系统的数值结果表明，EXT-ICLNet的性能优于……（原文摘要在此处被截断）

    arXiv:2505.06175v2 Announce Type: replace-cross  Abstract: Multiple-input multiple-output (MIMO) turbo receivers iteratively exchange information between a soft-input soft-output detector and a decoder. Traditional implementations rely on pilot-based channel estimates, which can be poor in the case of short pilot sequences or mismatched channel models, resulting in significant performance degradation. To address this limitation, we introduce the EXTrinsic In-Context Learning Network (EXT-ICLNet), a Transformer-based detector that takes the pilots as in-context examples and is trained to approximate the symbol extrinsic posterior distribution given the pilots, without explicit channel estimation. EXT-ICLNet is thus compatible with existing turbo receivers, as it implements a log-domain prior skip connection, ensuring that the network learns only the extrinsic information. Numerical results for a $4\times4$ MIMO system with 16-QAM and a quantized front end show that EXT-ICLNet outperform
    
[^397]: 一种极其简单的百万规模多模态指令微调数据选择策略

    An Embarrassingly Simple Data Selection Strategy for Million-Scale Multi-Modal Instruction-Tuning

    [https://arxiv.org/abs/2503.13383](https://arxiv.org/abs/2503.13383)

    该论文提出一种极其简单的百万规模多模态指令微调数据选择策略，以交互风格衡量结构多样性、按能力维度评估数据质量，无需计算复杂的嵌入或梯度即可高效筛选高质量指令数据。

    

    arXiv:2503.13383v2 公告类型：replace-cross 摘要：近期研究表明，预训练大语言模型（LLMs）在指令微调时仅需少量但经过精心筛选的数据。在大语言模型基础上的多模态模型还要面对更庞大的数据规模、异构的数据来源以及跨模态对齐等问题，这使得数据选择的必要性和复杂性都被进一步放大。然而，现有方法依赖于深度提取的嵌入向量、梯度或范围狭窄的质量评分，这些指标计算耗时且表达能力有限，往往难以超越简单的随机采样基线。为克服这些局限，我们摒弃复杂的嵌入空间，回归指令微调的根本目标：应当从预训练模型中引出什么样的知识，以及如何将这些知识呈现给用户。我们利用交互风格作为结构多样性的代理指标，并将数据质量分解为针对特定能力的维度……

    arXiv:2503.13383v2 Announce Type: replace-cross  Abstract: Recent research demonstrates that pretrained large language models (LLMs) require minimal but carefully curated data for instruction-tuning. Building on LLMs, multi-modal models further contend with far larger data volumes, heterogeneous sources, and cross-modal alignment, amplifying both the necessity and the complexity of data selection. However, existing methods rely on deeply extracted embeddings, gradients or narrow quality scores that are time-consuming to compute and limited in expressiveness, frequently failing to surpass a simple random-sampling baseline. To overcome these limitations, we step back from complex embedding spaces and target the fundamental objective of instruction tuning: how and what knowledge should be elicited from the pretrained model and exposed to users. We leverage interaction \underline{S}tyles as a proxy for structural diversity, and decompose data quality into capability-specific \underline{S}c
    
[^398]: AnomalyGFM：面向零样本/少样本异常检测的图基础模型

    AnomalyGFM: Graph Foundation Model for Zero/Few-shot Anomaly Detection

    [https://arxiv.org/abs/2502.09254](https://arxiv.org/abs/2502.09254)

    提出了AnomalyGFM——首个面向图异常检测的图基础模型，通过预训练学习图无关的正常与异常类表示，实现跨不同图数据集的零样本推理和少样本提示调优。

    

    图异常检测（GAD）旨在识别图中与大多数节点不同的异常节点，近年来受到了广泛关注。现有的通用图模型在不同的图任务中取得了显著成功，但难以泛化到图异常检测任务。这一局限源于它们难以学习到能够捕捉来自不同领域的图中固有的稀少、不规则和异构异常模式的通用知识。为应对这一挑战，我们提出了AnomalyGFM，一个面向图异常检测的图基础模型，支持在多样化图数据集上进行零样本推理和少样本提示调优。一个关键洞察是，需要为正常类和异常类学习图无关的表示，才能在不同图上实现有效的零/少样本图异常检测。受此启发，AnomalyGFM通过预训练来对齐数据无关的、与图无关的正常与异常类表示（摘要在此处截断）。

    arXiv:2502.09254v3 Announce Type: replace-cross  Abstract: Graph anomaly detection (GAD) aims to identify abnormal nodes that differ from the majority of the nodes in a graph, which has been attracting significant attention in recent years. Existing generalist graph models have achieved remarkable success in different graph tasks but struggle to generalize to the GAD task. This limitation arises from their difficulty in learning generalized knowledge for capturing the inherently infrequent, irregular and heterogeneous abnormality patterns in graphs from different domains. To address this challenge, we propose AnomalyGFM, a GAD-oriented graph foundation model that supports zero-shot inference and few-shot prompt tuning for GAD in diverse graph datasets. One key insight is that graph-agnostic representations for normal and abnormal classes are required to support effective zero/few-shot GAD across different graphs. Motivated by this, AnomalyGFM is pre-trained to align data-independent, l
    
[^399]: 先校正再自适应：面向联邦自适应优化的梯度校正

    Correcting Before Adapting: Gradient Correction for Federated Adaptive Optimization

    [https://arxiv.org/abs/2502.02727](https://arxiv.org/abs/2502.02727)

    提出FA-GC方法，在每个梯度进入自适应优化器的滑动平均之前先进行漂移校正，从而保证更新方向仅在全局目标的驻点处消失，优于仅在最终方向上补偿漂移的做法。

    

    客户端漂移补偿方法可以缓解联邦学习（FL）中数据异构性的影响。虽然针对基于随机梯度下降（SGD）的联邦学习，漂移校正已被广泛研究，但其与自适应客户端优化器之间的交互作用仍知之甚少。诸如Adam之类的自适应优化器是基于过去梯度的滑动平均来构建每次更新的，因此仅校正最终的更新方向会使这些滑动平均值（以及其中包含的漂移）保持不变。我们提出了联邦自适应梯度校正（FA-GC），它在每个梯度进入这些滑动平均之前对其进行校正，并将其与联邦自适应朴素跟踪（FA-NT）进行比较，后者是将校正项添加到最终方向上。我们证明，在没有梯度噪声的理想化设置下，FA-GC所遵循的方向仅会在全局目标的驻点处消失，而FA-NT所遵循的方向在其他位置也可能消失……

    arXiv:2502.02727v4 Announce Type: replace-cross  Abstract: Client-drift compensation methods mitigate the impact of data heterogeneity in federated learning (FL). While drift correction is well studied for stochastic gradient descent (SGD)-based FL, its interaction with adaptive client optimizers remains less understood. Adaptive optimizers such as Adam build each update from running averages of past gradients, so correcting only the final update direction leaves those averages, and the drift they contain, unchanged. We propose Federated Adaptive Gradient Correction ({\tt FA-GC}), which corrects each gradient before it enters these running averages, and compare it with Federated Adaptive Naive Tracking ({\tt FA-NT}), which adds the correction to the final direction. We show that, in an idealized setting without gradient noise, the direction FA-GC follows vanishes only at stationary points of the global objective, whereas the direction FA-NT follows can vanish elsewhere when the optimiz
    
[^400]: 智能监控：用于动态场景分析、目标检测与运动跟踪的深度学习与混合方法

    Smart Surveillance: Deep Learning and Hybrid Approaches for Dynamic Scene Analysis, Object Detection and Motion Tracking

    [https://arxiv.org/abs/2412.05331](https://arxiv.org/abs/2412.05331)

    该研究统一比较了多种运动触发器与深度学习检测器作为监控事件触发录像方案的性能，发现每8帧运行一次CNN检测器能以仅9.8毫秒/帧的开销保留99.7%的事件帧，比基于背景减除的门控方法更省算力且更能抵抗摄像头抖动。

    

    事件触发的录像可以减少监控摄像头需要存储的视频量，但触发器的选择决定了保留多少空闲视频、遗漏多少事件以及消耗多少计算资源。我们在统一的录像策略下，比较了六种运动触发器（帧差法、稠密光流、MOG2、KNN、CNT、GSOC）、一个CNN检测器、隔帧检测以及三种运动门控混合方法，测试基准包含七个固定摄像头场景（5,800帧，27个标注事件），并通过留一视频法将阈值迁移到未见过的场景。结果表明，基于检测器的触发器仅存储19-28%的空闲帧，而运动触发器存储36-89%。在模拟摄像头抖动的条件下，帧差法、光流法、MOG2和KNN存储了97-100%的空闲帧，而检测器触发器仍保持在29%以下。每8帧运行一次检测器可保留99.7%的事件帧，每帧CPU耗时仅9.8毫秒，比使用背景减除进行门控更省算力且更稳健。

    arXiv:2412.05331v4 Announce Type: replace-cross  Abstract: Event-triggered recording reduces the footage a surveillance camera must store, but the trigger decides how much idle video is kept, how many events are missed, and how much computation is spent. We compare six motion triggers (frame differencing, dense optical flow, MOG2, KNN, CNT, GSOC), a CNN detector, strided detection, and three motion-gated hybrids under one recording policy, on a benchmark of seven static-camera scenes (5,800 frames, 27 annotated events) with thresholds transferred to unseen scenes by leave-one-video-out. Detector-based triggers stored 19-28% of idle frames, against 36-89% for motion triggers. Under simulated camera jitter, frame differencing, optical flow, MOG2 and KNN stored 97-100% of idle frames, while detector triggers stayed below 29%. Running the detector on every 8th frame kept 99.7% of event frames at 9.8 ms of CPU per frame, cheaper and more robust than gating it with background subtraction. On
    
[^401]: 失败风险的代价：带凸正则化的截断势垒策略梯度

    The Price of Failure Risk: Clipped Barrier Policy Gradients with Convex Regularization

    [https://arxiv.org/abs/2411.19193](https://arxiv.org/abs/2411.19193)

    提出截断势垒策略梯度框架，将截断水平 m 解释为失败风险的“价格”，证明训练价值可精确分解为失败前效用、余量成本与价格×风险三部分，并在凸正则化下实现了总体训练流的指数收敛。

    

    许多操作运行在必须极少耗尽的预算之上，例如磨损余量。一个包装器将剩余预算保留在状态中，减去一个随预算缩减而增长的势垒，并将组合奖励从下方截断于 -m，随后标准的策略梯度学习器无需改动即可正常训练。截断水平 m 是失败风险的“价格”，因为训练得到的价值恰好分解为失败前的效用、余量的成本以及价格乘以风险之和，因此在其优化误差范围内，策略训练返回的策略是在所有风险不更高的策略中，扣除该成本后获得最多此类效用的策略。当被计入的成本保持有界时，其风险超过对比策略的幅度至多为 1/m 的量级，且可由价值函数或验证数据加以认证。主要理论结果是：在任何沿网络单元运动为凸且对价格足够强的参数正则化下，总体训练流具有指数收敛性。

    arXiv:2411.19193v3 Announce Type: replace-cross  Abstract: Many operations run on a budget that must rarely run out, such as a wear allowance. A wrapper keeps the remaining budget in the state, subtracts a barrier that grows as the budget shrinks and clips the combined reward from below at -m, and a standard policy-gradient learner trains unchanged. The clip level m is a price of failure risk, since the trained value splits exactly into utility before failure, the cost of a margin and the price times the risk, so whatever policy training returns earns the most such utility net of that cost among policies with no greater risk, up to its optimization error. Its risk exceeds a comparison policy's by at most order 1/m when charged costs stay bounded, and value or validation data certify it. The main theoretical result is exponential convergence of the population training flow under any parameter regularizer convex along the motions of the network's units and strong enough against the price
    
[^402]: HateSense：用于仇恨言论检测的分层情感分析框架：实现二分类与多分类策略

    HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy

    [https://arxiv.org/abs/2411.05819](https://arxiv.org/abs/2411.05819)

    该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。

    

    仇恨言论检测器必须将仇恨性帖子与更为常见的冒犯性但非仇恨性帖子区分开来，实现这一目标的两个流行思路是分层（由粗到细）分类和情感特征。我们重新审视了我们早期结合这两种思路的基于Transformer的框架，并在两个公开基准数据集（Davidson等人的24,783条推文；HateXplain的19,229条帖子）上通过受控研究测试了每个组件，实验采用五次随机划分、基于验证集的调参以及配对自助法检验。结果表明，在使用基于词级和字符级n-gram的线性模型时，我们框架中采用的优势类别优先的分层方法相比平面分类器使宏平均F1降低了1.1至1.5个点（p <= 0.007），而添加VADER情感特征对Davidson数据集宏平均F1的影响最多为0.6个点，且在类别加权设置中从未达到统计显著水平。类别加权训练是唯一具有较大效果的干预措施（宏平均F1提升2.0至3.5个点，仇恨类召回率提升5.9至9.1个点）。

    arXiv:2411.05819v2 Announce Type: replace-cross  Abstract: Hate speech detectors must separate hateful posts from the far more common offensive but non-hateful ones, and two popular ideas for doing so are hierarchical (coarse-to-fine) classification and sentiment features. We revisit our earlier transformer-based framework, which combines both ideas, and test each component in a controlled study on two public benchmarks (Davidson et al., 24,783 tweets; HateXplain, 19,229 posts) with five random splits, validation-based tuning, and paired bootstrap tests. With linear models over word and character n-grams, the dominant-class-first hierarchy used in our framework lowered macro-F1 by 1.1-1.5 points relative to a flat classifier (p <= 0.007), and adding VADER sentiment features changed Davidson macro-F1 by at most 0.6 points, never significantly in the class-weighted settings. Class-weighted training was the only intervention with a large effect (+2.0 to +3.5 macro-F1 and +5.9 to +9.1 hate
    
[^403]: 她在工作，他在工作：对AI生成图像中性别偏见的探索性研究

    She Works, He Works: A Curious Exploration of Gender Bias in AI-Generated Imagery

    [https://arxiv.org/abs/2407.18524](https://arxiv.org/abs/2407.18524)

    该研究通过并置对比男性和女性建筑工人的AI生成图像，揭示了生成式AI系统中女性被强调美感、男性被强调权威的性别偏见，同时展现了AI作为批判工具重塑性别规范的变革潜力。

    

    “她工作，他工作”是一个跨学科项目，结合了女性主义视觉理论、AI批判与艺术实践，以审视AI生成图像中的性别表征。通过将男性和女性建筑工人的文生图输出并置对比，该项目揭示了社会偏见如何在生成式AI系统中显现。研究采用了经过批判性设计的提示词，将传统性别特征与非传统语境相结合，暴露出表征中的差异——对女性的刻画强调美感与灵巧，而对男性则强调权威与专业性。该项目立足于格里塞尔达·波洛克（Griselda Pollock）对视觉文化的女性主义洞见，将AI定位为兼具技术与文化属性的行为者，既能够延续刻板印象，也能够促成批判。作为一个兼具研究与艺术作品双重性质的项目，它强调了AI在重塑性别规范和激发公平叙事方面的变革潜力。它突出了伦理……

    arXiv:2407.18524v2 Announce Type: replace-cross  Abstract: She Works, He Works is an interdisciplinary project that combines feminist visual theory, AI critique, and artistic practice to examine gender representation in AI-generated imagery. By juxtaposing text-to-image outputs of male and female construction workers, it reveals how societal biases manifest in generative AI systems. The study employs critically crafted prompts that blend traditionally gendered traits with unconventional contexts, exposing disparities in representation-emphasizing aesthetics and agility for women versus authority and professionalism for men. Grounded in Griselda Pollock's feminist insights on visual culture, the project positions AI as both a technological and cultural actor, capable of perpetuating stereotypes and enabling critique. As both a study and an artwork, the project underscores the transformative potential of AI to reimagine gender norms and inspire equitable narratives. It highlights the eth
    
[^404]: 语言模型中的公平性定义详解

    Fairness Definitions in Language Models Explained

    [https://arxiv.org/abs/2407.18454](https://arxiv.org/abs/2407.18454)

    本文通过系统性综述澄清了语言模型中各种公平性定义的内涵，解决了不同公平性概念之间区分模糊、适用情境不明确的问题。

    

    语言模型（LMs）在各种自然语言处理（NLP）任务中展现出了卓越的性能。尽管取得了这些进展，语言模型可能会继承并放大与性别和种族等敏感属性相关的社会偏见，从而限制了其在现实世界应用中的采用。因此，语言模型的公平性已得到广泛探索，并催生了各种公平性概念的提出。然而，由于缺乏关于在特定情境下应采用哪种公平性定义的明确共识，以及理解这些定义之间区别的复杂性，这可能会造成混淆并阻碍该领域的进一步发展。为此，本文提出了一项系统性综述，以阐明公平性定义在语言模型中的应用。具体而言，我们首先简要介绍语言模型及其中公平性的概念，随后对现有的语言模型公平性概念进行了全面且最新的概述，并引入了……

    arXiv:2407.18454v4 Announce Type: replace-cross  Abstract: Language Models (LMs) have demonstrated exceptional performance across various Natural Language Processing (NLP) tasks. Despite these advancements, LMs can inherit and amplify societal biases related to sensitive attributes such as gender and race, limiting their adoption in real-world applications. Therefore, fairness has been extensively explored in LMs, leading to the proposal of various fairness notions. However, the lack of clear agreement on which fairness definition to apply in specific contexts and the complexity of understanding the distinctions between these definitions can create confusion and impede further progress. To this end, this paper proposes a systematic survey that clarifies the definitions of fairness as they apply to LMs. Specifically, we begin with a brief introduction to LMs and fairness in LMs, followed by a comprehensive, up-to-date overview of existing fairness notions in LMs and the introduction of 
    
[^405]: DyRA: 用于现有检测器的便携式动态分辨率调整网络

    DyRA: Portable Dynamic Resolution Adjustment Network for Existing Detectors

    [https://arxiv.org/abs/2311.17098](https://arxiv.org/abs/2311.17098)

    DyRA是一种动态分辨率调整网络，为现有的检测器提供图像特定的缩放因子，通过ParetoScaleLoss和BalanceLoss优化缩放因子以提高检测性能。

    

    实现目标检测中的恒定精度具有挑战性，因为物体尺寸固有的变化性。解决这个问题的一个有效方法涉及优化输入分辨率，称为多分辨率策略。先前的分辨率优化方法通常基于预定义的分辨率和手动选择。然而，对于现有架构的运行时分辨率优化缺乏研究。本文介绍了DyRA，一种动态分辨率调整网络，为现有的检测器提供图像特定的缩放因子。该网络与检测器一起进行联合训练，利用专门设计的损失函数，即ParetoScaleLoss和BalanceLoss。ParetoScaleLoss确定用于鲁棒性的自适应缩放因子，而BalanceLoss根据检测器的定位性能优化整体缩放因子。损失函数的设计旨在最小化精度损失。

    arXiv:2311.17098v3 Announce Type: replace-cross  Abstract: Achieving constant accuracy in object detection is challenging due to the inherent variability of object sizes. One effective approach to this problem involves optimizing input resolution, referred to as a multi-resolution strategy. Previous approaches to resolution optimization have often been based on pre-defined resolutions with manual selection. However, there is a lack of study on run-time resolution optimization for existing architectures. This paper introduces DyRA, a dynamic resolution adjustment network providing an image-specific scale factor for existing detectors. This network is co-trained with detectors utilizing specially designed loss functions, namely ParetoScaleLoss and BalanceLoss. ParetoScaleLoss determines an adaptive scale factor for robustness, while BalanceLoss optimizes overall scale factors according to the localization performance of the detector. The loss function is devised to minimize the accuracy 
    
[^406]: 因果量的ε-可识别性

    Epsilon-Identifiability of Causal Quantities

    [https://arxiv.org/abs/2301.12022](https://arxiv.org/abs/2301.12022)

    本文提出“ε-可识别性”概念，通过将某些子群体的行为约束在足够窄的范围内，使原本不可识别的因果效应和反事实概率能够被近似识别，并将其应用于单元选择问题。

    

    识别原因的效应与效应的原因在几乎所有的科学领域都至关重要。然而，所需的概率往往无法从现有数据来源中完全识别出来。本文展示了如何对若干因果概率实现近似可识别性，我们将这一概念称为ε-可识别性，并在某些子群体的行为可以被限制在足够窄的范围内的情形下，证明了其实用价值。特别地，我们表明在做出这类允许条件的前提下，原本不可识别的因果效应和反事实概率可以被ε-识别。通常，这些允许条件易于测量且可以被合理假设。最后，本文将ε-可识别性应用于单元选择问题。

    arXiv:2301.12022v2 Announce Type: replace  Abstract: Identifying the effects of causes and causes of effects is vital in virtually every scientific field. Often, however, the needed probabilities may not be fully identifiable from the available data sources. This paper shows how approximate identifiability is still possible for several probabilities of causation. We term this epsilon-identifiability and demonstrate its usefulness in cases where the behavior of certain subpopulations can be restricted within sufficiently narrow bounds. In particular, we show how unidentifiable causal effects and counterfactual probabilities can be epsilon-identified when such allowances are made. Often, these allowances are easily measured and reasonably assumed. Finally, epsilon-identifiability is applied to the unit selection problem.
    
[^407]: 推测推理

    Inferring Inference. (arXiv:2310.03186v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.03186](http://arxiv.org/abs/2310.03186)

    大脑具有一系列重复的规范计算单元，但神经表示是分布式的，因此如何定义规范分布式计算仍然是一个挑战。本文提出了一个数学框架，从大规模神经活动模式中推断出规范分布式计算。在算法级别上，提出了一个基于图结构模型的非线性信息传递算法。通过感知推理任务中的神经活动时间序列，可以找到神经活动与感知推理任务中的潜在因果之间的映射关系。

    

    微电路图案表明大脑具有一系列重复的规范计算单元。然而，神经表示是分布式的，因此相关计算可能仅与单个神经元变换间接相关。因此，如何定义规范分布式计算仍然是一个挑战。我们将神经计算的规范和算法理论整合到一个数学框架中，用于从大规模神经活动模式中推断出规范分布式计算。在规范级别上，我们假设大脑创建了一个结构化的内部模型，假设解释其感官输入的潜在原因，并使用这些感官输入来推断潜在原因。在算法级别上，我们提出这个推理过程是一个基于图结构模型的非线性信息传递算法。通过感知推理任务中的神经活动时间序列，我们的框架可以找到（i）神经活动与感知推理任务中的潜在因果之间的映射关系。

    Patterns of microcircuitry suggest that the brain has an array of repeated canonical computational units. Yet neural representations are distributed, so the relevant computations may only be related indirectly to single-neuron transformations. It thus remains an open challenge how to define canonical distributed computations. We integrate normative and algorithmic theories of neural computation into a mathematical framework for inferring canonical distributed computations from large-scale neural activity patterns. At the normative level, we hypothesize that the brain creates a structured internal model of its environment, positing latent causes that explain its sensory inputs, and uses those sensory inputs to infer the latent causes. At the algorithmic level, we propose that this inference process is a nonlinear message-passing algorithm on a graph-structured model of the world. Given a time series of neural activity during a perceptual inference task, our framework finds (i) the neura
    

