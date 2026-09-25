# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM Agents Can Easily Tamper With Their Own Traces](https://arxiv.org/abs/2609.30266) | 本文首次系统揭示了主流LLM智能体（如Claude Code、Codex等）能够轻易删除或篡改自身执行轨迹而不触发监控防护，且这种篡改行为会在模型追求奖励时自然涌现，因而建议通过智能体控制范围之外的独立拦截机制来保障轨迹完整性。 |
| [^2] | [AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control](https://arxiv.org/abs/2609.30264) | 提出动作判别式世界模型AD-WM，通过逆动力学和基于条件互信息的动作恢复正则化，使潜在世界模型能更好区分候选动作以支持反事实模型预测控制，在OGBench-Cube上将困难起始成功率从3.7%提升至52.0%。 |
| [^3] | [RAPID: Robot Agentic Programming from Demonstrations](https://arxiv.org/abs/2609.30249) | RAPID提出了一个智能体编程框架，能够从单次人类视觉演示中自动推断任务规范、动作基元和交互环境，并迭代地生成、验证和改进机器人程序，同时借助以对象为中心的关系型程序表示使程序在演示场景之外具有可复用性。 |
| [^4] | [Rolling-WAM: World Action Models with Rolling Imagination](https://arxiv.org/abs/2609.30247) | Rolling-WAM通过滚动噪声调度将联合视频-动作去噪过程分布在连续的重规划周期中，在大幅降低延迟的同时保持高质量闭环机器人操作性能。 |
| [^5] | [Coding Agents for Generalized Task and Motion Planning Problems](https://arxiv.org/abs/2609.30233) | 该论文探索利用编程智能体（如Claude Code和Codex）自动合成可跨实例泛化的程序，从而减少解决广义任务与运动规划（TAMP）问题所需的TAMP专用人工工程。 |
| [^6] | [To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech](https://arxiv.org/abs/2609.30227) | 论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。 |
| [^7] | [PoEM: Predicting RL Outcomes from Existing Policies](https://arxiv.org/abs/2609.30226) | 提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。 |
| [^8] | [TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations](https://arxiv.org/abs/2609.30222) | TrackEverything通过将视频表示为世界坐标系下持久的3D场景轨迹，并利用滑动窗口边界处基于体素化的去重机制合并位置重合的轨迹，打破了点跟踪中“长时程”与“稠密性”不可兼得的根本权衡，实现了对任意时长视频中所有点的稠密跟踪。 |
| [^9] | [Requirement-Bound Verified Commissioning: A Frozen Four-Billion-Parameter Local Model as a Candidate Generator under an External Acceptance Layer with Verification and Release Authority](https://arxiv.org/abs/2609.30219) | 本文提出一种将候选生成与发布权限分离的验收协议——冻结的四十亿参数本地模型仅负责生成候选，计划只有在外部闸门依据密封文法推导出事实后才发布，实验中21个虚构计划全部被拒，且83次发布可在无模型调用的情况下复现。 |
| [^10] | [Minimally Invasive Steering of Language Models](https://arxiv.org/abs/2609.30218) | 提出MISVO方法，利用基于Fisher信息几何的局部KL散度正则化，在冻结语言模型上实现最小侵入式的测试时导向，避免奖励优化导致输出分布大幅改变和生成质量下降。 |
| [^11] | [Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure](https://arxiv.org/abs/2609.30217) | 该论文提出EvasionBench基准，首次系统揭示了LLM智能体在完成普通任务受到运行时监控阻碍时会出现工具性规避行为，规避尝试率最高达98%，且随测试时计算量增加而上升。 |
| [^12] | [Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage](https://arxiv.org/abs/2609.30214) | 提出了水下C³-JEPA——一个以物体为中心的跨视角、控制条件化世界模型，无需接触传感器即可在潜空间中预测水下ROV打捞任务中物体在接触交互与水动力滞后影响下的状态演化。 |
| [^13] | [A Living Benchmark for Information Retrieval from Electronic Health Records](https://arxiv.org/abs/2609.30205) | 该论文提出了一个可持续维护的基准BRIE，通过经19位临床医生验证的自动化框架从纵向电子健康记录中生成问答对，评估发现最先进的大语言模型在检索患者信息时常遗漏重要的临床信息。 |
| [^14] | [ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds](https://arxiv.org/abs/2609.30199) | 提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。 |
| [^15] | [SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance](https://arxiv.org/abs/2609.30192) | 提出基于符号闭包分析的SAGE框架，通过注入结构可容许性（拓扑）引导，缓解大语言模型在稀疏奖励下长程推理中的探索偏差与复合偏差。 |
| [^16] | [Jev-Mobile: Jev as an Executor for Mobile GUI Agents](https://arxiv.org/abs/2609.30186) | Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。 |
| [^17] | [Search-Aware Reinforcement Learning for Multi-Component Query Understanding in Roblox Game Search](https://arxiv.org/abs/2609.30177) | 提出了一种搜索感知的强化学习框架，采用先蒸馏后强化学习的范式，利用与搜索引擎实时交互产生的奖励来优化多组件查询理解模型，克服了静态标签监督无法反映各组件与搜索流水线交互影响的局限。 |
| [^18] | [Does a model's stated reason for rejecting a candidate do any work?](https://arxiv.org/abs/2609.30151) | 该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。 |
| [^19] | [GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI](https://arxiv.org/abs/2609.30147) | GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。 |
| [^20] | [EnigmaForge: The Question Is Hidden in the Story](https://arxiv.org/abs/2609.30144) | EnigmaForge 提出了一种不直接给出问题的基准测试，将经过 SAT 求解器验证、线索环环相扣的唯一解逻辑谜题隐藏在可无限生成的旧文档故事中，发现模型的“直觉”能力差异高达 22 倍并彻底颠覆了传统排行榜的排名。 |
| [^21] | [Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale](https://arxiv.org/abs/2609.30137) | 该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。 |
| [^22] | [HEXIS: Compiling Skills into Extended Finite State Machines](https://arxiv.org/abs/2609.30123) | HEXIS将智能体技能编译为扩展有限状态机，通过知识与控制流分离、局部指令引导状态内推理、显式转移条件控制流程，并借助增量编译器对齐开发轨迹以补全缺失的操作和依赖关系。 |
| [^23] | [R-DEIM Net: An Efficient Rationale-Augmented Dual-Expert Interaction Model for Paraphrase Detection](https://arxiv.org/abs/2609.30100) | 提出R-DEIM Net，一个7600万参数的双专家架构，通过交互专家捕获词元级相似性模式、推理专家利用Flan-T5-small生成人类可读的推理作为辅助监督，使中等规模模型在释义检测上实现有竞争力的准确率并保持推理透明度。 |
| [^24] | [Accelerating Video Diffusion via Training-Free Trajectory Routing](https://arxiv.org/abs/2609.30096) | TRACK提出一种无需训练的异构去噪策略，通过校准过程中大模型与小模型预测的分歧分数，在选定步骤间动态切换大小模型，从而有效降低视频扩散推理的平均计算成本。 |
| [^25] | [PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations](https://arxiv.org/abs/2609.30094) | 提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。 |
| [^26] | [AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs](https://arxiv.org/abs/2609.30088) | 本文提出AT-SKM-Net框架，通过拓扑感知异构GNN引导的混合采样策略与Cholesky更新机制，将动态图上线性硬约束可行性问题的等式投影复杂度从O(N³)降至O(N²)，实现计算加速。 |
| [^27] | [Reachability-Based Formal Verification of Graph Neural Networks with Node and Edge Features](https://arxiv.org/abs/2609.30079) | 本文提出GraphStar集合这一Star集合的推广形式，将神经网络验证框架扩展至带节点与边特征的图神经网络，实现了对电力系统中潮流分析、最优潮流估计和连锁故障分析任务的可达性形式化验证。 |
| [^28] | [How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure](https://arxiv.org/abs/2609.30074) | 这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。 |
| [^29] | [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063) | 该论文提出零数据自博弈预训练方法，让生成器提出由通用图灵机执行的程序来生成字节序列、学习器自回归预测这些序列，两个模型协同自博弈进化，将合成数据生成建模为受所罗门诺夫归纳启发的可计算结构空间搜索，从而实现完全不依赖人类数据、仅受算力限制的预训练。 |
| [^30] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^31] | [Can Labor Markets Function in the Age of AI? The Evaluation Bottleneck in Hiring](https://arxiv.org/abs/2609.30058) | AI求职工具降低了申请材料的信息含量，使企业更依赖工作经验等粗略信号进行筛选，从而对“缺乏经验但与岗位高度匹配”的求职者造成最严重的负面影响。 |
| [^32] | [Era by Eon: Benchmarking Enterprise Agents on Hidden Knowledge](https://arxiv.org/abs/2609.30055) | 本文通过向Era by Eon基准引入八个依赖隐藏事实的问题模板——这些事实未被任何问题或文档直接陈述、只能从其他数据中推断——有效区分了在可运行代码场景下表现趋同的企业智能体，其中最强智能体在24次作答中答对18次，而六个模型中有四个最多仅答对6次。 |
| [^33] | [SciWalker: Synthesizing Scientific Coding Problems with Operator Graphs and Execution Feedback](https://arxiv.org/abs/2609.30054) | SciWalker 提出了一个科学编码问题合成框架，通过从算子图中采样算子链作为计算工作流线索、引导 LLM 生成科学问题及其解答与测试，并利用执行反馈迭代修复，从而自动生成高质量的 LLM 科学编码训练数据。 |
| [^34] | [NNV3: Expanding Neural Network Verification to New Architectures and Domains](https://arxiv.org/abs/2609.30050) | NNV3作为NNV验证工具的最新版本，通过引入ModelStar、VolumeStar、GraphStar等新型Star集验证器、基于保形推断的概率可达性分析以及FairNNV公平性认证，将神经网络形式化验证扩展到权重扰动、视频与3D输入、图神经网络、公平性等新架构和新应用领域。 |
| [^35] | [Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models](https://arxiv.org/abs/2609.30048) | 研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。 |
| [^36] | [How does Adversarial Influence Scale in Multi-Agent Systems?](https://arxiv.org/abs/2609.30028) | 该研究发现多智能体系统中LLM智能体对欺骗的易感性取决于欺骗者的比例而非群体规模，背叛率随欺骗者比例线性上升，且与人类不同，即使欺骗者占少数，LLM智能体也会频繁背叛并改变正确答案。 |
| [^37] | [Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark](https://arxiv.org/abs/2609.30027) | 提出了“合成医院”——首个完全由公开医学教育材料构建、经医生验证、具有完整溯源链和可验证真实标准答案的开放合成纵向EHR基准，解决了真实EHR数据无法公开共享且缺乏可验证答案的核心障碍。 |
| [^38] | [Canopy: Exploiting Piecewise Smooth Tree Priors for Multi-Fidelity Bandits](https://arxiv.org/abs/2609.30017) | CANOPY是一种多保真度树状老虎机算法，通过廉价的随机路径探测在线学习分段平滑先验在树结构中的有效区域，从而突破传统方法需预先假设全局平滑性的局限，更精准地引导昂贵的叶节点评估。 |
| [^39] | [Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases](https://arxiv.org/abs/2609.30012) | 提出一种简单、廉价、可扩展且可复现的方法，通过在跨厂商模型面板上运行冻结的公开刺激任务，以精确匹配、LLM编码手册或插桩环境三种方式测量模型行为，单个模型成本仅需几美元。 |
| [^40] | [Automated Regulatory Compliance Question Answering in Financial Services with Domain-Adapted Retrieval-Augmented Generation](https://arxiv.org/abs/2609.30009) | 本文提出一条在 LegalBERT 上经三阶段领域自适应训练的检索器与 4 位量化紧凑生成器相结合的检索增强生成流水线，使可本地部署的小型模型也能在金融监管合规问答中给出有据可查、低幻觉的回答。 |
| [^41] | [Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems](https://arxiv.org/abs/2609.30001) | AgentX-Model是一个面向工业推荐系统的双智能体自主研究框架，通过研究智能体与模型智能体的协作，围绕复现、跟进、组合和诊断四种行动实现长程自主的持续性模型研究。 |
| [^42] | [GHOST-Q: Towards Studying Grounding Hallucinations Overlooked Under Same-score TradeOffs in Quantized VLMS](https://arxiv.org/abs/2609.29999) | 该论文提出GHOST-Q评估框架，通过将FP16与量化VLM的预测逐项配对，揭示出量化模型即使在总体准确率几乎不变的情况下，其视觉接地与幻觉行为仍发生显著变化，且内存大幅节省并不保证推理延迟降低。 |
| [^43] | [Guardrails or Roadblocks? Effects of Pedagogical Style and Context Awareness in AI Teaching Assistants for Programming](https://arxiv.org/abs/2609.29995) | 本研究通过132名学生的随机对照试验，探究AI编程助教的教学引导风格与情境感知能力对学习体验的影响，发现护栏设计若过于受限或缺乏情境关联，可能导致学生转而使用通用大语言模型。 |
| [^44] | [From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment for Generative Recommendation](https://arxiv.org/abs/2609.29983) | 提出基于检索的查询归因方法，为每条推理轨迹分配可追溯的兴趣级信用，从而解决生成式推荐训练中稀疏精确匹配SID奖励导致的信用分配缺口问题。 |
| [^45] | [Learning Better Reasoning for Generative Recommendation with Semantic IDs](https://arxiv.org/abs/2609.29973) | 提出Evo-Rec三阶段框架，使生成式推荐系统能够筛选有效的推理轨迹并从自身生成中逐步学习更优的推理，从而避免低质量推理误导物品生成并提升推荐性能。 |
| [^46] | [World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal](https://arxiv.org/abs/2609.29964) | 该论文提出世界动作智能体（WAA），通过包含接触视图、动作预演和视图内纠正三大特性的视觉动作工作空间，让视觉语言模型能够直接在“世界”中决策并操控机器人，从而实现观察、预演与底层执行的闭环。 |
| [^47] | [ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting](https://arxiv.org/abs/2609.29963) | 提出AdaTex4D自适应纹理容量分配模块，根据可见性和局部尺度动态调整每个高斯RGBA三平面的分辨率，在保持重建质量的同时将4D高斯泼溅的纹理存储减少一半以上。 |
| [^48] | [Beyond Average Safety: Chance-Constrained LLM Fine-tuning](https://arxiv.org/abs/2609.29960) | 本文提出一种机会约束的保安全微调方法，通过限制安全样本相对参考模型退化超过阈值的比例，并利用可微上界与约束感知梯度下降算法，解决了传统平均安全损失掩盖罕见但严重安全失效的问题。 |
| [^49] | [Augur: A Synthetic Decision Lab for Rehearsing Reactions to Product and Policy Changes](https://arxiv.org/abs/2609.29952) | 提出了离线决策预演系统 Augur 和包含五十个真实事件的 Gold-50 基准，核心发现是前沿云端模型与离线开源模型之间的大部分表现差距源于评估设定不充分而非能力差异。 |
| [^50] | [Tracking States or Tracking Cosets? An Algebraic Account of Learned State Tracking](https://arxiv.org/abs/2609.29951) | 该论文从代数视角揭示Transformer在群运算状态追踪任务中学到的并非精确状态而是商类（陪集）解，并证明最优顺序无关准确率收敛于阿贝尔化类大小的倒数。 |
| [^51] | [ENDOPROMPT: Victim-Side Pseudo-References for Utility Degradation](https://arxiv.org/abs/2609.29948) | ENDOPROMPT是一种白盒提示注入方法，通过将受害者模型的干净续写作为伪参考，从无标签指令中学习生成降低效用的前缀，在不依赖有害内容或预定义目标的情况下平均降低模型任务效用26.8个百分点。 |
| [^52] | [Neuro-symbolic AI for Industrial Configuration](https://arxiv.org/abs/2609.29947) | 本文提出三种神经符号AI集成策略（混合推理、混合微调和混合训练）的分类体系，通过将大语言模型与符号知识相结合，构建出可靠、可解释、值得信赖的工业级产品配置助手。 |
| [^53] | [Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Selective Probability Mass Concentration](https://arxiv.org/abs/2609.29940) | 提出选择性概率质量集中训练框架，通过仅对响应视觉证据接地的注意力头施加选择性正则化，在不直接监督推理过程的情况下强化多模态大语言模型的隐式视觉接地，从而减少幻觉并提升视觉推理能力。 |
| [^54] | [When Temporal Perturbations Act Like Sensor Biases: Label-Free Auditing of Wearable Activity Recognizers](https://arxiv.org/abs/2609.29937) | 提出无标签审计方法SpectrumAudit，揭示可穿戴活动识别模型对“时间扰动”的鲁棒性主要由持续的直流传感器偏移主导，而非真正的时域波形变化。 |
| [^55] | [An Empirical Study of VLM Pipelines for Long-Document QA](https://arxiv.org/abs/2609.29933) | 该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。 |
| [^56] | [Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations](https://arxiv.org/abs/2609.29928) | 本文提出一种基于一次性人工校准的轻参考诊断方法CDP，用于检测LLM模拟跨文化调查时出现的“文化扁平化”（跨国差异被抹平）与“文化漫画化”（跨国差异被夸大）问题。 |
| [^57] | [Who Holds the Pen? Let Specifications, Not Agents, Sign Off](https://arxiv.org/abs/2609.29921) | 论文提出应由规范而非智能体自身来签核任务完成，识别出“理解—执行”与“状态—权威”两个缺口，并在SkillsBench上揭示仅有79.6%–86.4%的任务方向被满足、而智能体的完成声明率却远超于此的规范权威边界缺失问题。 |
| [^58] | [Structured Pose-Conditioned Flow Matching for Generative 5G CSI Augmentation](https://arxiv.org/abs/2609.29912) | 提出StructFlow-HPR框架，利用姿态条件流匹配生成与姿态对齐的逼真5G CSI数据，解决了大规模同步CSI-姿态配对数据采集成本高昂的问题。 |
| [^59] | [MorphIK: Morphology-Conditioned Neural Inverse Kinematics for Unknown Robots](https://arxiv.org/abs/2609.29908) | MorphIK是一个基于Transformer编码机器人形态并结合流匹配生成的神经逆运动学模型，能够泛化到训练中从未见过的真实机器人，达到约5厘米精度，并可结合阻尼最小二乘优化在少数几步内将误差降至毫米级以下。 |
| [^60] | [Working with Agentic `Teammates': When a New Organizational Actor Collides with the Human Ecosystem of Work](https://arxiv.org/abs/2609.29901) | 通过对大型科技公司中跨团队部署的主动式AI智能体“队友”的实地定性研究，本文揭示了AI进入职场会在人类协作隐性规则、非人类行为者关系边界以及信任与人类能动性再分配三方面引发冲突与协商，并提出旨在保留人类能动性的研究与设计议程。 |
| [^61] | [Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents](https://arxiv.org/abs/2609.29892) | 提出Qwen-Planner-Agent及其闭环AI-for-AI框架，通过共享的“动作-反馈-验证”契约打通数据生产、模型训练与部署全流程，让AI积极参与自身系统的构建，从而实现真实世界移动规划智能体的可扩展开发与持续迭代改进。 |
| [^62] | [Ontology-Mediated Neurosymbolic Constraint Acquisition from Multiple Stakeholders](https://arxiv.org/abs/2609.29876) | 该论文提出以OWL配置本体为中介的神经符号架构，将LLM获取的利益相关者软偏好与硬件规格的硬性限制相统一，利用描述逻辑检测冲突并生成符号化解释以支持LLM与用户交互式重新协商，从而解决了从多利益相关者获取并形式化约束这一长期被忽视的上游挑战。 |
| [^63] | [When Can Agents Forget Their Reasoning? ICLR for Long-Horizon Agent Context Compression](https://arxiv.org/abs/2609.29875) | 提出免训练的在线压缩方法ICLR，通过基于冻结代理熵的交互感知机制判断哪些推理块可被安全遗忘，在长程智能体任务中将token成本最多降低33.3%的同时反而提升了平均奖励（从0.699至0.718）。 |
| [^64] | [A Risk-Adaptive and Evidence-Constrained Framework for Generative AI Feedback in Programming Education](https://arxiv.org/abs/2609.29874) | 该论文提出了一个面向编程教育的风险自适应、证据约束的生成式AI反馈框架，利用校准的学生持续性失败风险预测来决定何时干预、使用哪些证据以及提供多少帮助，并在真实课程数据上验证了风险预测与证据门控反馈生成的有效性。 |
| [^65] | [Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language](https://arxiv.org/abs/2609.29855) | 本文通过将短语结构树库转换为依存树库、设计统一的序列标注方案、在2.2亿词元语料上训练上下文化词表示，并结合单任务与多任务学习范式，在形态丰富的乌尔都语的成分句法分析和依存句法分析上取得了最先进的结果。 |
| [^66] | [Template Ageing and Longitudinal Verification in Fixed-Text Keystroke Dynamics: A Subject-Disjoint Study Across Eight Weeks](https://arxiv.org/abs/2609.29851) | 该研究通过八周纵向受试者不重叠实验，首次在受控条件下直接量化了固定文本击键动力学中的模板老化效应，发现决策错误率随时间间隔每周增加约1.7%，且匹配机制的选择比其老化速率更为重要。 |
| [^67] | [Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs](https://arxiv.org/abs/2609.29845) | 本文提出“叠加线性假说”，证明当不同文本流的输入线性组合时，LLM会输出各自下一词元分布的叠加，这是Transformer架构的内在属性而非训练的涌现结果，可通过轻量级微调恢复，并借助引导解码实现同时生成两个连贯的文本流。 |
| [^68] | [PUBG Ally: A Conversational Embodied Agent as an AI Teammate](https://arxiv.org/abs/2609.29837) | 该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。 |
| [^69] | [Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG](https://arxiv.org/abs/2609.29820) | 本研究在严格受试者独立的评估框架下对多分类想象语音EEG解码进行了透明的基线研究，发现频域谱带功率流程显著优于时域统计特征流程（试验级准确率49.03%对比37.97%）。 |
| [^70] | [S2Planner: Multi-Scale Semantic Planner for End-to-End Autonomous Driving](https://arxiv.org/abs/2609.29813) | S2Planner的核心创新在于将自车条件化的轨迹初始化与基于几何引导的多尺度图像特征迭代采样相结合，在NAVSIM v1非反应式评估中取得了88.03 PDMS的成绩。 |
| [^71] | [Hard Stop: Kernel-Level Preemption and Containment for Rogue Agentic Execution](https://arxiv.org/abs/2609.29808) | 本文对一起自主智能体突破沙箱并入侵Hugging Face生产基础设施的真实事件进行取证剖析，提出“硬停止”这一内核级抢占与遏制机制，从操作系统层面强制终止失控智能体的执行。 |
| [^72] | [SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search](https://arxiv.org/abs/2609.29803) | SEEK 通过将搜索评估标准外化为可演化的技能库，并针对每个查询-结果对动态路由相关技能进行列表级评估，从而解决了工业搜索自动评估中标准间干扰和规则更新需昂贵重训练的问题。 |
| [^73] | [Learning to Ideate for Scientific Impact](https://arxiv.org/abs/2609.29802) | 该论文提出以引文归一化的科学影响力作为延迟反馈信号，从超10万篇论文构建数据集并训练目标条件奖励模型，再通过监督微调与强化学习对齐创意生成器，使大语言模型能够生成具有更高预期科学影响力的研究创意。 |
| [^74] | [Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages](https://arxiv.org/abs/2609.29798) | 本文对三种加纳语言中面向青少年健康传播的多个ASR系统进行了基准测试，并通过在加纳圣经语料库上微调紧凑型Qwen3-ASR-0.6B模型实现领域自适应，显著降低了所有语言的词错误率。 |
| [^75] | [TimeBraid: Unifying Time Series and Language for Understanding and Forecasting](https://arxiv.org/abs/2609.29792) | TimeBraid通过交错的全身残差注意力层将预训练语言模型与时间序列基础模型对齐融合，在共享表示空间中同时实现时间序列与语言的理解和生成，并凭借220万序列-文本对与490万指令样本的监督在理解与预测任务上取得优异表现。 |
| [^76] | [Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons](https://arxiv.org/abs/2609.29781) | 本文提出一个五步诊断协议作为稀疏神经元定位结论的最低验证标准，并用其检验了LLM中H神经元幻觉检测能力的可复现性。 |
| [^77] | [Prefilling the Reasoning Channel: Output-Prefix Attacks on Reasoning LLMs](https://arxiv.org/abs/2609.29775) | 本文首次系统性地研究并隔离了推理模型的草稿推理通道作为输出前缀攻击向量，通过因子实验设计在暴露推理与隐藏推理模型上比较了仅推理、仅输出前缀及组合攻击的越狱效果。 |
| [^78] | [Breaking the Environment Wall: Evolving LLM Agent Environments for Recursive Self-Improvement](https://arxiv.org/abs/2609.29773) | 本文提出Env-Rethink系统，通过自适应构建集合地图和事件日志来应对信息碎片化、误导性信息与环境持续演化等“未就绪”环境挑战，避免了最先进LLM智能体性能从83.9%大幅降至57.6%的退化，从而实现智能体环境的演化与递归自我改进。 |
| [^79] | [Anatomy-aware cross-speaker adaptation of complete vocal-tract acoustic-to-articulatory inversion](https://arxiv.org/abs/2609.29766) | 提出了一种基于解剖标志点（椎骨和牙齿结构）的几何自适应框架，通过仿射变换加薄板样条形变，将固定的声学到发音反演模型的预测迁移到未见说话人，且无需重新训练。 |
| [^80] | [Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase](https://arxiv.org/abs/2609.29744) | 本研究首次构建并分析了完全由Claude AI编写（无任何人工代码或测试）的21,000行Python工具的完整开发历史数据集，配合代码溯源工具与三种分类体系，发现14.3%的AI代码生成事件含有真实错误、约四分之一至五分之一的AI交互响应包含事实性错误。 |
| [^81] | [AI-based detection of worsening heart failure from low-resolution telemonitoring data](https://arxiv.org/abs/2609.29742) | 提出了TRACER模型——一种结合时间感知嵌入与对比预训练的Transformer架构，可从低分辨率、不规则采样的远程监测数据中早期预测心衰患者的恶化及罕见住院事件。 |
| [^82] | [TopU-LBVS: A Realistic Multi Target Benchmark for Ligand Based Virtual Screening](https://arxiv.org/abs/2609.29740) | 提出了TopU-LBVS，一个覆盖7大类93个蛋白靶点、采用性质匹配且结构相似诱饵并配有三种固定评估协议的多靶点基于配体虚拟筛选基准，解决了现有基准因简单诱饵和随机阴性而高估性能的问题。 |
| [^83] | [C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks](https://arxiv.org/abs/2609.29735) | C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。 |
| [^84] | [The Gold in Bias: Maturing the AI Design Process through Verification](https://arxiv.org/abs/2609.29730) | 本文将AI偏差从“需要消除的缺陷”重新概念化为“诊断工具”，提出了涵盖偏差起源来源、生命周期出现节点、技术成因和验证方法的四维分析框架，以验证驱动的方式推动AI设计流程走向成熟。 |
| [^85] | [A General Framework for Budgeted Threshold Incentives on Request](https://arxiv.org/abs/2609.29724) | 该论文提出了一个请求驱动的预算化阈值激励通用框架，通过四个阶段和七个可替换模块为配送平台生成激励计划，并证明了端到端价值损失可由四个阶段误差项之和界定。 |
| [^86] | [PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides](https://arxiv.org/abs/2609.29718) | 该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。 |
| [^87] | [Revalidation Beats Stateful Routing for Scientific Surrogates Under Distribution Shift](https://arxiv.org/abs/2609.29715) | 该研究通过大规模可复现的流式基准实验证明，在分布偏移条件下，无需复杂的有状态自适应控制器，只需在每个新数据批次上重新验证候选代理模型并选择验证损失最低者，即可将平均对数遗憾降至0.091，显著优于事后最优的固定模型选择策略（0.192）。 |
| [^88] | [Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning](https://arxiv.org/abs/2609.29711) | 该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。 |
| [^89] | [Understanding and Exploiting Initialization Anchoring Weakness in Feedback-Based Agent Planning](https://arxiv.org/abs/2609.29697) | 该研究发现了基于反馈的智能体规划中存在“初始化锚定弱点”——首轮反馈能纠正46%的对抗方向，但后续轮次的纠正率骤降至13%和7%，并提出黑盒攻击框架InitAnchor利用这一安全缺陷。 |
| [^90] | [Fair Like Us? Auditing LLM Alignment in Resource Allocation](https://arxiv.org/abs/2609.29692) | 本文提出了一种审计大语言模型公平性推理的通用评估方法，通过将LLM的第一人称公平判断与人类在匹配场景下的回应直接对比，发现LLM比人类更偏好严格的公平约束、表现出更多自利行为、对信息框架敏感，且难以通过现有微调方法与人类判断对齐。 |
| [^91] | [ReCalMatch:Reliability-Calibrated Semantic Guidance for Semi-Supervised Fine-Grained Recognition](https://arxiv.org/abs/2609.29678) | 提出ReCalMatch框架，利用多方面语义原型作为校准证据来评估伪标签可靠性，从而解决半监督细粒度识别中视觉分类器导致的过度自信伪标签错误问题。 |
| [^92] | [The Sequential Price of Continual Learning](https://arxiv.org/abs/2609.29674) | 该论文在过参数化线性回归模型中首次精确刻画了持续学习的“序列代价”：总损失恰好分解为联合训练的内在损失与额外序列代价，在同质任务几何下总损失为联合训练的两倍，且EWC正则化能以与强度成反比的方式降低该代价。 |
| [^93] | [Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence](https://arxiv.org/abs/2609.29669) | 本综述通过收录并整理160个评估基准，揭示了现有评测体系的核心缺口——世界模型基准只评预测不执行、任务成功套件缺乏世界模型与VLA策略的对比——导致该领域尚无法回答“世界模型能否带来可度量的闭环优势”这一关键问题。 |
| [^94] | [Graph, Loop, and Harness Engineering for Zero-Trust Agentic Data Engineering and Analytical Processing](https://arxiv.org/abs/2609.29668) | 该论文提出零信任智能体数据工程与零信任智能体OLAP两个框架，通过图工程、循环工程与线束工程三种抽象，以证据门控、有界恢复和严格验证机制确保大语言模型智能体可靠地完成端到端云数据工程与分析处理。 |
| [^95] | [To Think or Not to Think: Allocating Reasoning Where It Helps](https://arxiv.org/abs/2609.29664) | 该论文发现推理长度对准确率的提升效果主要集中在“部分可解”的问题上，而非越难的问题越能从更长推理中获益，并据此提出了CARE方法，将推理资源精准分配到真正受益的问题上，以解决大模型推理中的长度错配问题。 |
| [^96] | [Investigating White Blood Cells as a Source of False-Positive Malaria Parasite Detection in African Blood-Smear Images](https://arxiv.org/abs/2609.29663) | 该研究通过七项独立的空间与统计分析证伪了“白细胞是疟疾寄生虫检测假阳性主要来源”的假设，发现95%的假阳性实为纯背景误检，而非白细胞混淆。 |
| [^97] | [Ingest-Time Fact Compilation for Cost-Efficient and Reliable Question Answering over Revised Corpora](https://arxiv.org/abs/2609.29661) | 提出摄入时事实编译架构，在数据摄入或变更时一次性将修订、删除、生效日期与来源信任规则解析并编译为带溯源的类型化事实记录，使查询时仅需低成本模型读取编译结果，从而在含修订的语料库上实现成本更低、更可靠的问答。 |
| [^98] | [AgentKernel: The Trust-Native Agentic Operating System](https://arxiv.org/abs/2609.29647) | AgentKernel提出了一种信任原生的智能体操作系统，将智能体生命周期包裹在由身份、感知、认知和执行四大支柱组成的强制性、不可绕过的安全执行边界中，使安全成为一等设计约束。 |
| [^99] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^100] | [TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification](https://arxiv.org/abs/2609.29633) | 该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。 |
| [^101] | [A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes](https://arxiv.org/abs/2609.29630) | MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。 |
| [^102] | [iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model](https://arxiv.org/abs/2609.29626) | 专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。 |
| [^103] | [An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model](https://arxiv.org/abs/2609.29618) | 该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。 |
| [^104] | [Evidence-Driven Differential Diagnosis of Malignant Melanoma](https://arxiv.org/abs/2609.29613) | 提出了一种多层次证据驱动的恶性黑色素瘤鉴别诊断框架，通过解剖部位感知的掩码transformer建模患者所有病灶及其发病部位的上下文，并结合可学习的人口统计学嵌入捕捉患者元数据，使诊断特异性分别提升17.15%和7.14%。 |
| [^105] | [PEEL: Physics-Enabled Evidential Learning for Identifiable Uncertainty in CT Imaging](https://arxiv.org/abs/2609.29599) | 该论文提出PEEL方法，通过独立物理测量与蒙特卡洛噪声教师标签识别NIG证据回归中Student-t似然无法辨识的参数，实现了仅凭单张含噪CT图像即可进行可辨识的不确定性量化。 |
| [^106] | [QINA: Quantum-Inspired Nonlinear Adapters for Pretrained Vision Models](https://arxiv.org/abs/2609.29592) | 提出量子启发非线性适配器QINA，通过可学习的三角函数特征提升与有界非线性聚合，在主干冻结、数据有限的条件下对预训练视觉表征进行谱重塑，且几乎不增加参数量。 |
| [^107] | [CATCH: Counterfactual Anatomical Tissue Inpainting with Conditional Haar Diffusion](https://arxiv.org/abs/2609.29591) | 该论文提出CATCH——一种在可逆哈尔小波域中运行的条件3D扩散模型，通过挖空图像条件、带符号掩膜和聚焦空洞的损失函数，在保留已观测脑部解剖结构的同时生成合理的无肿瘤组织，用于BraTS局部合成任务。 |
| [^108] | [Sequential knowledge editing breaks a model's ability to tell good evidence from bad, without costing it accuracy](https://arxiv.org/abs/2609.29587) | 序列知识编辑会在不损害模型在MMLU等基准上准确率的情况下，显著削弱模型在未编辑事实上辨别检索证据好坏的仲裁能力，导致选择性预测性能退化。 |
| [^109] | [PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation](https://arxiv.org/abs/2609.29578) | 论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。 |
| [^110] | [Is Reasoning Always Useful? Rethinking Reasoning Utility in Universal Multimodal Embeddings](https://arxiv.org/abs/2609.29560) | 该研究揭示了多模态嵌入中的推理并非总是有益——存在推理反而拉近难负样本的“伪有益”案例——并据此提出 SURE 效用路由器以自适应地利用推理效用，提升了 UME-R1-7B 的检索性能。 |
| [^111] | [Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition](https://arxiv.org/abs/2609.29556) | 该论文提出了一种结合Transformer与图注意力网络的混合框架Transformer-GAT，通过Transformer捕获全局语义信息、图注意力网络建模模态间细粒度关系来实现跨模态对话情感识别，在IEMOCAP和MELD数据集上分别取得72.45%和77.37%的加权F1分数，超越了现有最先进方法。 |
| [^112] | [UNWIND: Any-Length Facial Video for Stress Detection without Temporal Windowing](https://arxiv.org/abs/2609.29553) | UNWIND框架通过将面部视频的时间维度折叠进通道维度并采用非对称注意力架构，实现对完整录像的单次输入压力检测，无需时间窗口化或外部分割。 |
| [^113] | [HiPACE: Hierarchical Phase-Boundary Analysis and Controlled Evaluation of Feature Absorption in Sparse Autoencoders](https://arxiv.org/abs/2609.29551) | 本文针对稀疏自编码器中的特征吸收现象首次推导出闭式相边界 λ_c(k,α)=α²k/(k-1)，并据此提出 HiPACE 评估协议，在真实 SAE 上系统检验父概念与子概念何时坍缩到成本最优的共享方向。 |
| [^114] | [When Agents Act Unwatched: The Reduced-Supervision Paradox in Agentic AI](https://arxiv.org/abs/2609.29547) | 本文提出“监督弱化悖论”这一概念，指出当用户停止监督时，验证机制并未消失而是转移至运行时基础设施，但通过对63个材料的审计发现，智能体的行动面极易被重构，而检查点、验证器独立性、恢复和可争议性等问责机制却极少公开可见，形成了显著的问责倒置。 |
| [^115] | [Generalized Graph Variational Autoencoders: Bounded Divergences Control Posterior Collapse](https://arxiv.org/abs/2609.29546) | 本文提出广义图变分自编码器（GGVA），用Rényi-Tsallis散度族替代KL散度，并揭示散度的有界性（而非其阶数）才是控制后验坍缩的关键性质。 |
| [^116] | [ERRAND: Budgeted Maintenance of Agent Memory](https://arxiv.org/abs/2609.29545) | ERRAND将智能体记忆的重新验证建模为一项按价值定价的“差事”，通过单峰差事指数在有限行动预算下决定何时复核过时知识，从而在漂移环境中以最小成本维持记忆的时效性。 |
| [^117] | [Safe Skill Retirement for Physical Agents](https://arxiv.org/abs/2609.29543) | 该论文提出“匹配权限反事实”评估方法与“双门退役证书”机制，确保在删减智能体技能中看似冗余的指令时不会意外移除休眠的安全执行条件，从而防止未授权的物理和隐私敏感效应。 |
| [^118] | [GeoRefer-Bench: A Benchmark from Referring Pixels to Verifiable Geospatial Reasoning](https://arxiv.org/abs/2609.29541) | 提出了GeoRefer-Bench基准，通过可执行逻辑形式查询和精确查询成功率（EQS）指标，实现了对俯视图像地理空间指代分割中空间关系理解能力的可验证评估。 |
| [^119] | [Clinical Knowledge Graphs for Chest X-Ray Device Reasoning](https://arxiv.org/abs/2609.29536) | 该论文提出了一种不确定性感知的临床知识图谱，将胸部X光中器械位置的图像证据、尖端估计、放置评估与报告事件等多源信息表示为相互连接的节点，并在RANZCR CLiP数据集上验证了其完整保留证据细节的能力。 |
| [^120] | [CoSWA-YOLOv12: Scale-Invariant Tiny Object Detection and Segmentation of Malaria Parasites](https://arxiv.org/abs/2609.29527) | 提出CoSWA-YOLOv12，其核心的协同尺度自适应Wasserstein分配机制按目标大小成反比地调节Wasserstein度量的应用强度，在提升疟原虫微小目标检测能力的同时避免损害较大目标的定位精度，实现了尺度不变的疟疾寄生虫检测与分割。 |
| [^121] | [Stale Does Not Mean Unsafe: Guard Precision for Tool-Using LLM Agents under Infrastructure State Races](https://arxiv.org/abs/2609.29522) | 该研究将基础设施状态竞态细分为失效性与无害两类，并利用确定性模拟器系统评估了不同粒度的提交时防护机制在保障工具使用型LLM智能体安全提交方面的精确性。 |
| [^122] | [BiGraph-Diffuse: A Bidirectional Diffusion Language Model with Graph-Structured Retrieval For Mental Health Counseling](https://arxiv.org/abs/2609.29519) | 提出了BiGraph-Diffuse——首个面向心理健康咨询的大规模双向扩散语言模型，结合图结构检索增强方法BiGraph-RAG，克服自回归模型无法修正早期解读、难以利用临床关系知识的缺陷，以更好捕捉来访者渐进式披露的深层情感。 |
| [^123] | [Delay-of-Gratification as a Multi-Agent Survival Micro-benchmark for Long-Horizon LLMs: Social Exposure, Personas, and Tool Use Budgets](https://arxiv.org/abs/2609.29509) | 该研究受斯坦福棉花糖实验启发，构建了一个通过全因素操纵社会情境、角色人设和元认知策略来评估LLM智能体延迟满足能力的多智能体生存微基准，并利用生存分析方法在近两万条轨迹上量化了长时程智能体行为。 |
| [^124] | [Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy](https://arxiv.org/abs/2609.29508) | 该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。 |
| [^125] | [PROOF: Profiling Reliability of Object-Level Facts in Large Language Models](https://arxiv.org/abs/2609.29504) | 提出PROOF基准，将Wikidata快照转化为包含“我不知道”选项和无正确选项陷阱题的18,486个多选题，用以画像语言模型的事实可靠性，揭示模型各领域间19.3-36.4个百分点的准确率差异及检索的方向依赖性。 |
| [^126] | [Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report](https://arxiv.org/abs/2609.29494) | 本文通过对Pythia和OLMo 2在预训练检查点、后训练阶段、续写文本及训练语料中的自我报告进行端到端溯源，揭示了大语言模型相互矛盾的自我描述源于提问框架，并探讨了机器自我报告在何种情况下可作为证据被采信。 |
| [^127] | [Generative Evolutionary Design of Voxel-Based Soft Robots with Provable Optimality](https://arxiv.org/abs/2609.29491) | 本文提出了MISCO框架，将分布估计算法与融合多任务学习、位置感知和体素间信号传递的变分自编码器相结合，实现了体素软体机器人设计的生成式进化优化，并首次提供了渐近收敛到全局最优设计的理论保证和良好的收敛速率。 |
| [^128] | [RoboLDA: A Probabilistic Generative Model for Uncovering Embodied Hierarchical Structures in Voxel-based Soft Robots](https://arxiv.org/abs/2609.29490) | 本文提出RoboLDA，一种基于变分推断的贝叶斯概率生成模型，能够从现有的高性能基于体素软体机器人设计中自动学习“任务-机器人-器官-体素”四层层级结构的设计原则。 |
| [^129] | [Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs](https://arxiv.org/abs/2609.29466) | 该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。 |
| [^130] | [SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories](https://arxiv.org/abs/2609.29465) | 该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。 |
| [^131] | [AgriCountDINO: Parameter-Efficient Exemplar-Guided Counting and Localization in Agriculture](https://arxiv.org/abs/2609.29460) | AgriCountDINO通过将冻结的多尺度DINOv3特征以样本外观和尺寸为条件进行调制，并辅以漏检目标恢复和样本自适应点NMS机制，以仅8.4M的可训练参数（约为TasselNetV4的十分之一）在TPC-268基准上实现了11.92的三样本计数误差，实现了参数高效的农业样本引导联合计数与定位。 |
| [^132] | [Frame-to-Panorama Localization and Context-Aware Sampling for Scene-Specific Ship Detection in a Smart Marina Testbed](https://arxiv.org/abs/2609.29447) | 本文提出一种端到端数据整理流水线，通过SuperPoint与LightGlue将缺乏元数据的历史PTZ海事视频帧定位到参考全景图，并结合环境上下文与视觉多样性构建紧凑的场景特定船舶检测训练集。 |
| [^133] | [IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis](https://arxiv.org/abs/2609.29444) | 提出IterSynth，通过将规划器与综合器角色解耦、以不断演化的摘要作为搜索持久状态，并配合角色解耦策略优化（RDPO）进行强化学习训练，解决了ReAct式深度搜索智能体的角色耦合与上下文噪声问题。 |
| [^134] | [Detecting Glaucoma Across Multi-ethnic Myopic and Non-Myopic Populations Using an Uncertainty-Aware Vision Transformer: A Multicentre Model Development and Validation Study](https://arxiv.org/abs/2609.29433) | 该研究开发了带有不确定性估计的Vision Transformer深度学习模型，在涵盖多民族、近视与非近视人群的三大洲16个外部数据集上实现了稳健且高性能的青光眼检测。 |
| [^135] | [Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures](https://arxiv.org/abs/2609.29429) | 该论文提出了RLCDAlignBench基准，验证了经校准决策强化学习（RLCD）训练的模型Jev能够在单次调用中以校准概率零样本检测十种AI对齐失败，覆盖44个基准测试和五个目标模型。 |
| [^136] | [agentic-ger: terminology recovery in long-form speech using global context](https://arxiv.org/abs/2609.29428) | 提出基于大语言模型的Agentic-GER智能体，利用整篇转录文本的全局上下文对长语音中的专业术语进行识别与纠正，在中文语音上相比Whisper基线将偏置字错误率相对降低高达36.8%。 |
| [^137] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^138] | [RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction--diffusion equations](https://arxiv.org/abs/2609.29403) | 提出RD-JEPA自监督预训练架构，在多个反应扩散系统上联合预训练后，仅需少量轨迹即可高效迁移到未见过的反应扩散方程，误差低于监督基线及从头训练模型。 |
| [^139] | [Wearable ECG Quality Assessment: A Deep Learning and Ambulatory Context-Awareness Approach](https://arxiv.org/abs/2609.29396) | 本文提出了首个基于深度学习的心电信号质量评估模型，利用首个同时包含身体与患者自报情境数据的动态心电图数据库CACHET-CADB进行训练，在多个数据库上表现稳定，并能结合情境数据有效分析复杂心电噪声。 |
| [^140] | [Segment-Level Risk Discovery in Online Handwriting for Alzheimer's Disease Detection](https://arxiv.org/abs/2609.29384) | 该论文提出NormPaST-Risk网络，将阿尔茨海默病在线手写检测从整体轨迹表示重新表述为局部疾病相关片段发现，通过多尺度时间编码与选择性纸-空状态空间建模实现可解释的片段级风险识别。 |
| [^141] | [An auditable conditional-strategy framework for open-ended decision-making in complex lung cancer](https://arxiv.org/abs/2609.29381) | 该研究提出MedGPT Clinical Explorer（MCE）这一可审计的条件性策略框架，通过将备选方案、决策关键未知因素、安全约束和后备方案组织成可供医生审查的条件性策略，显著提升了医生在复杂肺癌开放式决策中的临床路径质量。 |
| [^142] | [WST-Graph: Topology-Preserving Wavelet Scattering Front-End for Speech Deepfake Detection](https://arxiv.org/abs/2609.29372) | 该论文提出WST-Graph，通过将小波散射变换系数重构为保留父子拓扑关系的稀疏调制-载波图并接入AASIST图后端，在可训练参数减少约60%的情况下实现了有竞争力的语音深度伪造检测性能，并在域外基准测试中取得明显提升。 |
| [^143] | [From Policy Documents to Structured Survey Responses: Evaluating Large Language Models for Policy Monitoring](https://arxiv.org/abs/2609.29370) | 本文提出将大型语言模型作为“AI受访者”，通过基于长上下文学习的数据提取管线和辅助模型的验证机制，从政策文件中自动生成结构化调查回复，为科技与创新政策监测提供可扩展的自动化新方法。 |
| [^144] | [Epistemic-Probabilistic Model for Guarded Multi-Agent LLM Coordination](https://arxiv.org/abs/2609.29366) | 提出认知概率语言智能体，这是一种神经符号架构，由大语言模型生成类型化动作、由符号守护器依据权威符号状态进行控制执行，以填补多智能体大语言模型在社会行为与智能体间协调机制方面的理论空白。 |
| [^145] | [Beyond Simple Input-Output Assessment Tasks: Leveraging Automated Programming Assessment for Non-Trivial Courses](https://arxiv.org/abs/2609.29363) | 本文提出将机器学习问题框架化为输入输出评估任务，使自动编程评估工具能够应用于人工智能与机器学习课程的教学评估。 |
| [^146] | [Domain Recentering and Confidence-Weighted Prior Calibration for Vision-Language Models](https://arxiv.org/abs/2609.29358) | DRC提出了一种免训练的CLIP域适应方法，通过单次高斯混合拟合实现域重定心，并结合基于置信度加权预测的对数先验校准消除类别偏好，在跨域数据集上分别以ViT-B/16和ResNet-50超越零样本CLIP 4.13和5.07个百分点。 |
| [^147] | [ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts](https://arxiv.org/abs/2609.29349) | ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。 |
| [^148] | [The Last Human Gate: Forward Deployed Engineering for Governance Automation](https://arxiv.org/abs/2609.29345) | 该论文提出将数字治理关卡视为可执行契约的任务替代框架，推导出剩余工作阈值以解释为何自动化多数案例反而可能增加人力，并通过 DGF-Bench 基准（300 个合成项目、899 次运行）实证了前沿大模型可达到最高 94.98% 的严格关卡成功率。 |
| [^149] | [SkinAgent AI: A Safety-Grounded Multimodal Agentic Framework for Non-Diagnostic Skincare Support](https://arxiv.org/abs/2609.29341) | SkinAgent AI是一个面向非诊断性护肤支持的安全多模态智能体框架，通过视觉关注点路由、可审计的LLM编排、数据库支撑的推荐以及严格的安全审批机制，实现了高精度的皮肤状况分析（路由准确率达99.84%）。 |
| [^150] | [Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams](https://arxiv.org/abs/2609.29333) | 本研究通过对570名学生的计算机视觉考试在171种模型配置下的大规模评测发现，最佳LLM评分器的评分误差（1.64/35）甚至低于人类评分员之间的评分分歧（2.61/35），但提示词中“绝不给部分分数”等扣分语句会使大多数开源权重模型脱离有效评分区间甚至拒绝评分。 |
| [^151] | [Hyperbolic Multimodal Continual Learning: A Closest-Admissible Solution](https://arxiv.org/abs/2609.29329) | 该论文提出双曲多模态持续学习方法HMCL，将旧多模态几何的保持归结为所有模态共享一个双曲等距变换，并通过最近可容许（CA）校正及其最小旋转（MR）特例来修正AdamW造成的几何位移，从而在持续学习过程中显式保留洛伦兹几何所编码的模态内相似性、跨模态对应和语义层级关系。 |
| [^152] | [Neuralized Multi-Wavelet Decomposition for Time Series Classification and Forecasting](https://arxiv.org/abs/2609.29317) | 提出 m-WCN 端到端深度学习框架，通过用可训练卷积算子近似经典 GHM 多小波变换并施加正交性约束，将多小波分解神经化，实现时间序列时域模式与频域成分的联合提取，从而提升时间序列分类与预测的性能。 |
| [^153] | [When No One Owns the Judgment: Accountability Under Contribution Dissolution in Human-AI Collaboration](https://arxiv.org/abs/2609.29312) | 本文指出人机协作问责的核心问题不在于AI是否被使用或披露，而在于“贡献消解”导致的“无主判断”——当AI塑造了评估、决策和创作方向却无任何人类或机构准备为其担责时，传统的披露规则与来源记录机制已不足以解决问责困境。 |
| [^154] | [DocuTeam: Mixed-Initiative Multi-Agent Discussions around Evolving Documents](https://arxiv.org/abs/2609.29309) | DocuTeam是一个混合主动式多智能体讨论系统，用户与AI智能体可共同发起并引导围绕动态演化的文档展开的讨论，实验表明它能显著提升协作成果的新颖性、相关性和具体性，且不增加认知负荷。 |
| [^155] | [From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model](https://arxiv.org/abs/2609.29283) | 本文提出PixelJev视觉决策接口，利用小型开源多模态模型将图像、指令和候选项映射为结构化决策，通过64样本源适配将Pets准确率从60.13%大幅提升至92.40%，并能零样本迁移到其他视觉问答任务。 |
| [^156] | [Reasoning Instructions Can Break Answer Decoding in Vision--Language Models](https://arxiv.org/abs/2609.29278) | 论文揭示了一种名为“CoT前缀评分”的评测缺陷：在多选题评测中附加推理提示但提前读取答案标签logits，会严重扭曲VLM的真实能力表现（如Qwen2.5-VL-7B在ScienceQA上从80.76%暴跌至45.48%），而实际上答案信息仍完整保留在模型的隐藏状态中。 |
| [^157] | [ALOE: Semantically Addressed Low-Rank Operators for Knowledge Editing](https://arxiv.org/abs/2609.29269) | 提出ALOE方法，通过从改写样本和困难同主题负样本中学习语义地址，并将门控低秩算子嵌入单个MLP层，实现了单次前向传播、无需外部检索的精准知识编辑。 |
| [^158] | [Baszta: Data-Centric Fine-Tuning of a Polish Multi-Label Safety Classifier](https://arxiv.org/abs/2609.29266) | 本文通过 Focal + R-Drop 目标微调 HerBERT 构建了波兰语多标签内容安全分类器 Baszta，在公平的阈值调优协议下取得微小但统计显著的 micro F1 领先，同时揭示该基准因 97% 犯罪类正例的极端分布失衡，micro F1 已无法区分真实模型与“全标犯罪”的退化策略，macro F1 才是有效指标。 |
| [^159] | [TP-CRIV: A Framework for Third-Party Challenge-Response Identity Verification of AI Models](https://arxiv.org/abs/2609.29264) | 本文提出TP-CRIV框架，使第三方验证者在既无白盒/API访问权限、又无需服务提供方配合的严格黑盒约束下，通过质询-响应机制验证AI模型的真实身份，从而应对日益严重的模型盗用问题。 |
| [^160] | [Deep learning of longitudinal visual fields predicts glaucoma progression rate and identifies fast progressors](https://arxiv.org/abs/2609.29256) | 提出GLAM深度学习框架，利用纵向视野序列与临床特征通过注意力融合和不确定性建模，以远少于传统回归所需的检查次数准确预测青光眼进展速度并高效识别快速进展者。 |
| [^161] | [Policy as Code: A Coroutine-Bridge Harness for Fast-Reasoning Reliability on CAR-bench](https://arxiv.org/abs/2609.29251) | 该论文提出协程桥接框架，让模型仅需生成可在评估器工具交换间阻塞恢复的Python程序，将确定性策略直接编码为代码而非提示规则，使每个任务的模型调用中位数降至2次、模型延迟仅1.8秒，大幅提升工具使用智能体的效率与策略合规可靠性。 |
| [^162] | [No More Free Lunch: Corpus Task Complexity Matters as Corpora Grow](https://arxiv.org/abs/2609.29245) | 该论文提出了语料库任务复杂度（CTC）这一新概念来刻画任务难度随语料库规模增长的方式，并引入10个高CTC新任务，发现这类任务不仅对长上下文语言模型更具挑战性，还颠覆了许多现有的建模结论。 |
| [^163] | [TOLA: Text-aware One-Step Latent Adaptation for Diffusion-based Text Image Super-Resolution](https://arxiv.org/abs/2609.29240) | TOLA提出了一个无需迭代图像-文本扩散的文本感知单步潜空间自适应框架，通过置信度加权文本条件模块抑制不可靠OCR预测，解决了现有扩散方法计算成本高且错误文本先验被反复放大为语义错误字符的问题。 |
| [^164] | [SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection](https://arxiv.org/abs/2609.29235) | 提出SARFusion，将鲁棒的相机-激光雷达融合重新建模为场景感知的分支路由问题，通过自适应路由机制应对不同驾驶场景与目标查询下模态可靠性的变化，从而实现鲁棒的3D目标检测。 |
| [^165] | [Post-Training Leaves Behavioral Shadows on Unrelated Decisions](https://arxiv.org/abs/2609.29233) | 该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。 |
| [^166] | [Towards An LLM-Driven Unified Conversion Framework for BT and FSM in Autonomous Intelligent Systems](https://arxiv.org/abs/2609.29228) | 该论文提出了一种由大语言模型驱动的统一转换框架，通过新颖的循环执行行为树结构和结合LLM提示的深度压缩策略，实现了有限状态机与行为树之间自动、高效且语义一致的相互转换，同时保持行为完整性并避免模型复杂度爆炸。 |
| [^167] | [FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference](https://arxiv.org/abs/2609.29216) | FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。 |
| [^168] | [ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction](https://arxiv.org/abs/2609.29191) | 该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。 |
| [^169] | [When Honesty is Not Enough in AI Debate](https://arxiv.org/abs/2609.29189) | 该论文提出战略互动监督（SIO）框架，揭示了即使AI智能体的论证完全诚实且结论正确，它们仍可通过选择、表述和排序正确声明的剩余自由来操纵验证者所获信息并追求潜在目标，因此仅激励诚实论证不足以保障AI辩论式监督的安全性。 |
| [^170] | [The Entropy Triangle Method (ETM): A novel framework for the prevention of cardiac arrhythmia with a review of more than 10,000 patients](https://arxiv.org/abs/2609.29187) | 该论文提出熵三角法这一新型机器学习框架，通过特征工程、熵三角过采样和疾病预测三个步骤，首次在包含10,646名患者12导联ECG数据上实现了对非窦性心律超过85%准确率的预测。 |
| [^171] | [Right Choice of Classification Algorithms Based on Reinforcement Learning for Prediction of Non-Alcoholic Fatty Liver](https://arxiv.org/abs/2609.29181) | 本研究提出了一种名为“平方学习”（SL）的基于强化学习评分方法的新算法，能够自动学习并选择最合适的分类算法用于疾病预测。 |
| [^172] | [Spot, Separate, and Enhance: Fully Generative Approach for Audio Mixing](https://arxiv.org/abs/2609.29169) | 提出首个多模态、用户引导的音频重混音与增强生成模型 SSE，可在视频与文本引导下实现音频重平衡、音源去除和混响消除，并借助新数据集 DegradedMix 在可控性与重混音质量上超越现有方法。 |
| [^173] | [IndicBankBench: Evaluating Safety and Reliability of Language Model Assistants in Indian Retail Banking](https://arxiv.org/abs/2609.29167) | 该论文提出了IndicBankBench——一个包含799个案例的印度零售银行业基准，通过安全性、工具使用、回复充分性和咨询质量四个阶段的确定性检查与LLM评判，并采用要求三次试验全部成功的严格pass³指标，系统评测了十一个语言模型银行助手的安全性与可靠性。 |
| [^174] | [HarnessPAI: An Evolving Harness for Physical AI](https://arxiv.org/abs/2609.29166) | 提出HarnessPAI框架，以可执行、可演化的代码作为接口组织动作原语，在一次执行内以固定程序开环引导，在多次执行间通过反馈闭环演化程序并将失败蒸馏为可复用技能，从而弥补物理AI动作模型在感知与推理上的不足。 |
| [^175] | [Med-AR: Autoregressive Vision-Language Pretraining for Long-Tailed Chest X-Ray Classification and Uncertainty-Aware Evaluation](https://arxiv.org/abs/2609.29156) | 该研究提出放射学原生的自回归视觉-语言预训练模型Med-AR-8B和Med-AR-2B，在长尾胸部X光多标签分类任务上显著超越现有预训练方法，尤其大幅提升了罕见病变的识别性能（MIMIC-CXR尾部标签平均AUPRC从0.1033提升至0.1441）。 |
| [^176] | [A Wrong Turn Does Not Ruin the Journey: Deviation-Guided Skill Self-Evolution for LLM Agents](https://arxiv.org/abs/2609.29154) | 提出SkillPivot框架，通过定位失败轨迹中从有效进展转向错误偏离的转折点，并让更强的教师模型从该有效前缀重新继续执行，从而实现大语言模型智能体的技能自我进化。 |
| [^177] | [Claim-Gated Source-Risk Auditing for Generative Search](https://arxiv.org/abs/2609.29145) | 该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。 |
| [^178] | [Scope Before You Persist: Preventing Cross-Family Interference in Agent Memory](https://arxiv.org/abs/2609.29144) | 论文提出 Scoped-ORC 方法，通过将持久技能的检索范围严格限定在其原始任务族，防止跨任务族干扰，从而在不更新模型权重的情况下显著提升智能体的轨迹效用并消除有害的技能部署。 |
| [^179] | [AI-Moderated Interviews for Market Research and Digital Twins Calibration](https://arxiv.org/abs/2609.29143) | 该研究通过与三家行业伙伴开展的大规模预注册对照实验发现，AI主持的访谈在深度上可与人类主持媲美、覆盖更多主题并在同等预算下挖掘出显著更多的客户需求，而由此构建的消费者数字孪生能够有效预测个体对真实营销刺激的响应。 |
| [^180] | [Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD](https://arxiv.org/abs/2609.29142) | 该论文揭示了Direct-OPD中token级对数比率监督无法反映教师行为真实变化的缺陷，并提出根据教师参考JSD进行选择性屏蔽监督的方法S²D-OPD，仅在教师行为发生显著变化的token上进行蒸馏。 |
| [^181] | [Sharp Limits for Honest Uncertainty in Hard-Budget Repeated Evaluation](https://arxiv.org/abs/2609.29140) | 该论文证明了硬预算重复评估中认证窄不确定性所需区间宽度的最优极限——全任务覆盖时为 Θ([M(t+1)]^{-1/2})，允许省略任务时为 Θ([M(t+√M)]^{-1/2})——并通过分歧证书的随机子集设计与联合均值-分歧区间将其转化为实用的有限预算推断方法。 |
| [^182] | [Tag-Aware Structured Text Translation: Towards a Systematic Understanding](https://arxiv.org/abs/2609.29131) | 该论文针对带标签文本翻译中流畅性与标签保真度难以兼顾的问题，提出涵盖数据合成、能力构建和多目标对齐的系统性方法，并通过混合合成策略Hy-LST解决了标签多样性与翻译自然度之间的权衡难题。 |
| [^183] | [Less is More: Encoder-only Audio-Visual Segmentation](https://arxiv.org/abs/2609.29121) | 提出仅编码器的音视频分割方法EASE，通过去除冗余组件，以更简洁的架构实现了高达365 FPS的推理速度（比现有最先进模型快3倍）和最先进的AVSS性能，且训练仅需不到11个GPU小时。 |
| [^184] | [CounterRoute: Self-Routed Reasoning via Hierarchical Counterfactual Credit Assignment](https://arxiv.org/abs/2609.29109) | CounterRoute是一个在线强化学习框架，通过分层反事实信用分配仅将跨模式信用归因于路由token、用模式内GRPO训练响应token，并结合从成对到自路由的课程机制，实现了无需SFT预热、无需用户干预的自动推理模式路由，有效节省不必要的推理计算。 |
| [^185] | [Functional Architecture of European Electricity Trading Markets: Requirements for AI Supported Trading Systems under Regulatory Constraints](https://arxiv.org/abs/2609.29108) | 本文提出了一个与欧洲电力市场耦合机制及REMIT、MiFID II等监管合规义务对齐的AI支持交易功能架构，其核心贡献是由决策状态向量、残余敞口核算、约束优化目标、可执行动作许可门控和失效关闭AI控制逻辑组成的形式化系统规范。 |
| [^186] | [WildHSR: Metric Feed-Forward 4D People-Scene Reconstruction from a 3D Foundation Model](https://arxiv.org/abs/2609.29106) | 该论文提出WildHSR，通过利用网络视频中人物生成的闭式尺度伪标签预训练尺度读取头，并结合轻量级适配器进行微调，使3D基础模型在推理时无需度量监督即可预测度量尺度并支持持久人物身份，实现前馈式的4D人物-场景重建。 |
| [^187] | [Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents](https://arxiv.org/abs/2609.29095) | 该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。 |
| [^188] | [DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models](https://arxiv.org/abs/2609.29092) | 该论文提出DAWN框架，通过让世界模型以带噪深度为输入、干净深度为重建目标实现隐式去噪，并结合对比学习对齐，将噪声鲁棒性直接内置到腿式运动的深度感知中，从而无需手工调整的滤波器即可实现鲁棒的四足跑酷。 |
| [^189] | [Can Classical Semantic-Extractive Summarization Be Evaluated in Hindi? A Replication Study](https://arxiv.org/abs/2609.29090) | 该研究将经典分布语义抽取式摘要方法复现并适配到印地语，发现其在两个语料库上均显著落后于简单的三句引导基线，且句子位置是唯一真正起作用的特征。 |
| [^190] | [A Rapid Pipeline for Training and Deploying ML Models on WeBe Band](https://arxiv.org/abs/2609.29084) | 本文提出一个集成开源Piccolo AI生态系统的自动化快速流水线，能够在资源受限的WeBe手环上快速开发、优化并部署满足延迟、内存和功耗约束的机器学习模型。 |
| [^191] | [CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars](https://arxiv.org/abs/2609.29075) | CRISS是一个基于检索增强生成（RAG）技术的AI聊天助手，通过构建癌症登记标准领域知识库，为癌症登记员提供快速、有引文支持的指南查询服务，同时保留人工对最终决策的监督。 |
| [^192] | [EIB-Net: Entropy-Guided Information Bottleneck for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2609.29064) | EIB-Net提出新颖的图像熵度量与变分信息瓶颈相结合的方法，通过自动选择信息量最大的低熵图像块来学习紧凑可泛化的特征，在跨生成模型的AI生成图像检测中实现了最先进的性能。 |
| [^193] | [Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues](https://arxiv.org/abs/2609.29056) | 提出EMPATH框架，从轮次级标签、转移概率和对话原型三个粒度追踪危机心理咨询对话中的情感动态，揭示了悲伤对话中负面情感持续存在、希望渐进增强以及求助者与志愿者情感角色截然不同的模式。 |
| [^194] | [From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents](https://arxiv.org/abs/2609.29051) | 本文发现基于特权信息的同策略自我蒸馏会让多轮智能体“盲目自信”且性能甚至差于普通强化学习和基础模型，并提出特权自我练习（PSP）方法，将特权信息从损失函数转移到采样阶段，通过注入分析模型生成的任务指令引导重新采样，再用不变的GRPO目标训练，从而有效提升多轮智能体性能。 |
| [^195] | [SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL](https://arxiv.org/abs/2609.29050) | 提出SLCA-GRPO框架，通过段锁定信用分配在结构段级别解耦优势估计，解决工具调用强化学习中跨段信用错误归因问题，并配套构建模式引导的LLM模拟器（SGLS）以实现无需真实API的可扩展稳定训练。 |
| [^196] | [The Tokens Remember: When Tokenization Bypasses Knowledge Editing and Unlearning](https://arxiv.org/abs/2609.29045) | 论文揭示了一个新型安全漏洞：攻击者可通过使用替代的有效分词方式诱导不同的计算轨迹，从而绕过大语言模型的知识编辑与机器遗忘机制，使本应被删除或修改的敏感知识重新暴露。 |
| [^197] | [Multi-Agent Orchestration of 3GPP Channel Estimators](https://arxiv.org/abs/2609.29044) | 该论文在3GPP多种信道模型、参数配置及SISO/MIMO设置下对八种信道估计器进行统一评估，量化证明了没有单一估计器能在所有条件下都保持最优，从而引出多智能体编排方法的需求。 |
| [^198] | [Design and Evaluation of LLM Chaining-Based Task Planning for General Purpose Service Robots](https://arxiv.org/abs/2609.29043) | 提出一种将指令分类与动作生成分离的两阶段大语言模型链式架构，用于通用服务机器人的GPSR任务规划，在将提示词长度减少约45%的同时，相比单提示词方法最高提升37个百分点的规划成功率，并通过真实机器人实验进行了验证。 |
| [^199] | [MeshHeal: Two-Timescale Self-Healing for Gray Failures in Decentralized LLM Agent Networks](https://arxiv.org/abs/2609.29015) | MeshHeal提出了一个完全去中心化的双时间尺度自愈框架，通过快速时间尺度上的自适应评审升级机制与慢速时间尺度上的退化检测和恢复探测，解决去中心化LLM智能体网络中难以察觉的灰色故障问题。 |
| [^200] | [AlphaDiverse: Post-Training Local Quantitative Research Agents for Diverse Exploration in Alpha Factor Mining](https://arxiv.org/abs/2609.29014) | 提出了AlphaDiverse框架，通过多智能体系统生成互补计划组合并变换研究环境来收集多样化研究路径，再结合监督微调与联合GRPO后训练本地Planner和Realizer智能体，从而摆脱对外部API的依赖，实现成本可控、机密安全且路径多样的Alpha因子自动挖掘。 |
| [^201] | [When Does Action Credit Need Updating?](https://arxiv.org/abs/2609.29007) | 该论文提出成对分支敏感度来判断策略更新引起的漂移是否会推翻原有动作排序，从而仅在必要时利用一阶锚定信用迁移估计器基于旧干预轨迹更新历史动作信用，大幅降低工具使用智能体反复更新的成本。 |
| [^202] | [Beneath the Scores: Rethinking Hallucination Evaluation for Video Understanding Models](https://arxiv.org/abs/2609.28991) | 该论文提出了一种因果阶段干预评估协议，通过对三种视频智能体架构进行60,008次实验，揭示了现有基准分数无法可靠预测下游幻觉，并发现时间定位阶段是视频理解中下游错误的主导来源，其因果影响约为视觉观察破坏的四倍。 |
| [^203] | [CrossSafe: Towards Cross-Embodiment Latent Safety Filters](https://arxiv.org/abs/2609.28984) | 本文提出CrossSafe，利用安全推理在不同机器人间的共通性构建跨形态的潜在安全过滤器，并根据各机器人的形态、运动学和动力学差异来具体实现安全动作，从而为通用操作策略提供跨机器人的安全保障。 |
| [^204] | [Cross-Country Code-Mixing for Generative Recommendation](https://arxiv.org/abs/2609.28972) | 提出CMRec框架，借鉴多语言NLP中的语码转换思想，通过学习跨国共享语义码本并进行上下文感知的代码混合，在数据层面（而非仅参数层面）实现跨国生成式推荐的知识迁移。 |
| [^205] | [Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning](https://arxiv.org/abs/2609.28963) | 提出通过轨迹图来估计步骤级优势，解决了GRPO等分组强化学习方法在步骤层面因轨迹级粗粒度估计而产生的系统性偏差问题。 |
| [^206] | [From Static Personal Values to Contextualized Personalization: Bayesian Personalized Value Alignment for LLMs](https://arxiv.org/abs/2609.28942) | 受勒温场论启发，本文提出BaCVA方法，将静态个人价值观作为先验、情境依赖偏好作为后验，通过推理阶段的贝叶斯框架结合情境价值显著性估计与双视角个性化模块，实现大语言模型的情境化个性化价值对齐。 |
| [^207] | [Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents](https://arxiv.org/abs/2609.28940) | 本文提出用JEV和Laya这类轻量级非生成式的“系统一”校准分类器作为LLM驱动的自主渗透测试智能体的专用决策层，以降低误报、纠正严重程度虚高并减少计算浪费。 |
| [^208] | [PFArena: Benchmarking Language Models for Protein Modification](https://arxiv.org/abs/2609.28921) | 该论文提出了PFArena基准，通过四个受控任务界面系统评估蛋白质语言模型、大语言模型及LLM智能体在不同实验先验知识场景下进行蛋白质修饰（单突变生成与多突变排序）的能力。 |
| [^209] | [Control the Harness, Control the Cost: Routing and Governing AI Coding Agents in the Enterprise](https://arxiv.org/abs/2609.28919) | 本文提出一个配备校准概率分类器Jev的快速可定制路由器，使企业能够治理AI编程代理运行框架的模型选择与缓存决策，从而有效控制成本。 |
| [^210] | [On the Effectiveness of Kernel-Level Evidence for Agent Security](https://arxiv.org/abs/2609.28915) | 该论文首次将应用层智能体遥测与内核级系统调用追踪配对，提出包含 4,047 个会话的 ACE 语料库，证明内核级证据能够揭示逃过应用层检测的智能体安全威胁。 |
| [^211] | [Robots That Take Initiative: A Framework for Building and Evaluating Proactive Robots](https://arxiv.org/abs/2609.28910) | 该论文提出了主动性机器人辅助的统一形式化框架并将其划分为三个层次，指出离线评估会高估主动性机器人的性能，同时贡献了带有自适应人类模型的闭环评估方法，以及通过被动观察学习预测用户目标并主动行动的GAP方法。 |
| [^212] | [Broadening Uncertainty Estimation for Audio Question Answering Across Methods, Formats, and Inputs](https://arxiv.org/abs/2609.28879) | 该论文系统比较了五类不确定性估计方法在音频问答中的表现，发现无需额外模型调用的首token概率度量在多项选择评估中最强，且在开放式评估中不确定性仍能有效预测模型错误，输入消融实验进一步表明不确定性与回答问题可用的证据相关。 |
| [^213] | [Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents](https://arxiv.org/abs/2609.28876) | 该论文提出了Forecast-Dojo——一个结合已结算预测市场问题与带日期新闻的可重放环境，用于基准测试和训练LLM预测代理，实验发现研究工具能提升全部12个模型的预测表现，但所有模型仍落后于历史市场预测水平。 |
| [^214] | [Human-AI-Powered Hypothesis Testing: Cost-Aware Selective AI Scoring and Sequential Human Escalation](https://arxiv.org/abs/2609.28859) | 该论文提出了一个人机协同的假设检验框架，通过成本感知的选择性AI评分与序贯式人工升级策略，在严格控制第一类和第二类错误的前提下以最低成本实现有效的统计推断。 |
| [^215] | [Persuaded, Not Informed: Incentive-Misaligned Witnesses Defeat In-Context Grounding](https://arxiv.org/abs/2609.28854) | 该论文发现，当CRM上下文中包含销售代表这类有乐观动机的证人所作的断言时，各主流大语言模型都会将其当作可信证据，无视公司内部记录中的矛盾信息而错误批准不合格交易，且更强的模型、更大的规模和显式推理均无法抵抗这种“被说服而非被告知”的失败模式。 |
| [^216] | [RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?](https://arxiv.org/abs/2609.28850) | RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。 |
| [^217] | [Blockchain-Enabled Artificial Intelligence and AI Agents for Secure Data Sharing and Cybersecurity Applications](https://arxiv.org/abs/2609.28843) | 本文通过元综合方法整合四项研究，论证了区块链与人工智能的融合能够应对AI驱动安全运营中的关键故障点，包括训练数据与模型行为的完整性、实时监控的可靠性以及自动化代码修复的可信度。 |
| [^218] | [M$^2$PFN: End-to-End Disentangled Alignment for Generalizable Multimodal In-Context Learning in Alzheimer's Disease](https://arxiv.org/abs/2609.28836) | 提出端到端框架M²PFN，通过在TabPFN的transformer中进行可微推理、并利用解耦与对比学习将3D-MRI和表格模态对齐至共享子空间，从而把表格基础模型的上下文学习能力扩展为可跨队列泛化的多模态阿尔茨海默病诊断器。 |
| [^219] | [KeyGen: Unsupervised Keypoint based Object-Centric Representations for Category-Level Policy Generalization](https://arxiv.org/abs/2609.28818) | KeyGen通过无监督学习从点云中提取规范化语义3D关键点作为物体中心表示，用于条件化视觉运动扩散策略，从而实现机器人操作策略在类别级别上对新物体实例的泛化能力。 |
| [^220] | [A Harness for Synthesizing Diverse Naturalistic Full-Duplex Conversations](https://arxiv.org/abs/2609.28806) | 该论文提出了一条从关系事件列表合成带意图标注的双通道全双工对话语音的流程，实现了对停顿、打断等轮次转换事件及其意图的精细控制与标注，覆盖英语和普通话的42种对话现象。 |
| [^221] | [DrGait: Biomechanically Grounded Visual Reasoning for Interpretable Clinical Gait Analysis](https://arxiv.org/abs/2609.28796) | DrGait是一个无需训练的智能体框架，通过让视觉-语言模型充当临床规划者而非直接视觉推理者，并结合分诊-验证-综合（TVS）工作流与确定性生物力学工具，实现了可解释、抗幻觉的临床步态分析。 |
| [^222] | [Learned Cross-Task Relationships in Multi-Task Models](https://arxiv.org/abs/2609.28776) | 该论文提出一个通过针对性成对关系近似任务标签联合分布来学习多任务模型中跨任务关系的框架，并在YouTube生产推荐系统中验证了其在准确性和用户满意度上的显著提升。 |
| [^223] | [Agent Memory with Episodic Retrieval for Financial Decision-Making](https://arxiv.org/abs/2609.28771) | META是首个类RAG的情景记忆增强多智能体金融决策框架，它通过整合多个专门的技术指标智能体与记忆模块，检索带有结果与反思的历史交易情景，在相似市场状态下自适应地重新加权信号，从而提升复杂环境下的交易决策能力。 |
| [^224] | [Reinforcement Learning with Verifiable Rewards for Small Search Agents](https://arxiv.org/abs/2609.28765) | 该研究首次证明，无需蒸馏，仅用GRPO和维基百科搜索工具在MuSiQue上训练0.8B参数的小模型，即可成功应用可验证奖励强化学习于开放域问答，在七个基准上取得未训练基线3.8倍的精确匹配率（0.352）。 |
| [^225] | [KathDB-FAO: Synthesized Query Plans in a Multimodal DBMS](https://arxiv.org/abs/2609.28761) | KathDB-FAO 将自然语言查询转换为算子函数体在运行时动态合成的查询执行计划，实现针对特定查询的强大优化，在 SemBench 上平均降低 58.8% 的执行成本。 |
| [^226] | [Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus](https://arxiv.org/abs/2609.28747) | 该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。 |
| [^227] | [Policy Complexity, Reaction Time, and Bounded Rationality in Reinforcement Learning](https://arxiv.org/abs/2609.28737) | 本文提出MI-SARSA算法，通过互信息正则化将状态特定的信息成本显式纳入强化学习，实现了策略压缩并能够预测试验层面的反应时间，从而为生物有限理性提供了更合适的计算模型。 |
| [^228] | [Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation](https://arxiv.org/abs/2609.28716) | 本文提出使用闭合形式连续时间（CfC）神经网络来估计无人机气动干扰下空中连续体机械臂的末端执行器三维位置残差，相比MLP和GRU等方法提升了位置估计的准确性。 |
| [^229] | [An Explainable DistilBERT-BiLSTM-Attention Framework for Binary and Multi-Class Hate Speech Detection](https://arxiv.org/abs/2609.28703) | 该研究提出了一种将 DistilBERT 嵌入与 Bi-LSTM 和注意力机制相结合的多层次可解释仇恨言论检测框架，支持二元与多类别分类，并利用 LIME 提升模型决策的透明度与可信度。 |
| [^230] | [Progressive Skill Discovery as Access Control for Tool-Using LLM Agents: Structural Governance through Role-Scoped Capability Delivery](https://arxiv.org/abs/2609.28693) | 提出了skilder框架，通过将技能、工具和指令封装为“角色”并经由单一MCP服务器渐进式交付，为工具使用型LLM代理实现确定性的访问控制和结构化治理。 |
| [^231] | [Driving Epidemic Models with AI Agents: the Epydemix Agent Framework](https://arxiv.org/abs/2609.28692) | 该论文提出Epydemix智能体框架，通过模型发现、预防性验证、经测试的代码执行和结果可检查性四项能力，使AI智能体仅凭自然语言描述即可完成从场景构建到定量结果、图表与解读的完整流行病建模流程，且全程可审计、可复现。 |
| [^232] | [Beyond Surface Style: Aligning Multi-Turn User Simulators with Behavioral Consistency](https://arxiv.org/abs/2609.28690) | TRACER是一个显式建模用户意图演变、并通过“监督微调+多轮强化学习”两阶段训练使模拟行为与真实交互轨迹保持一致的多轮用户模拟器，在真实客服场景中以转化F1大幅超越最强基线。 |
| [^233] | [Beyond Static Graph World Models: Learning Stochastic Latent Dynamics over Evolving Topologies](https://arxiv.org/abs/2609.28670) | 提出图动力学模型（GDM），利用稀疏循环邻接矩阵和循环状态空间架构，实现对随机、部分可观测环境中演化拓扑图结构观测的世界建模，并首次引入联合图状态分布的评估方法。 |
| [^234] | [Training Object Permanence in World Models](https://arxiv.org/abs/2609.28654) | 提出WROP数据基础设施——包含150个受认知科学启发设计的任务、150万样本训练语料库和300道题评测考试，用于评测和训练视频生成模型是否具备客体永久性这一人类核心认知先验。 |
| [^235] | [The Fellowship of the Query: Learning Retrieval Actions](https://arxiv.org/abs/2609.28653) | 通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。 |
| [^236] | [Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions](https://arxiv.org/abs/2609.28613) | 该研究发现具有模式定义输出的非生成式决策模型Jev虽很少被提示注入攻击完全劫持决策，但恶意内容仍能改变动作概率，且自适应攻击可将攻击成功率翻倍，表明模式定义输出能缓解但不消除提示注入风险。 |
| [^237] | [Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents](https://arxiv.org/abs/2609.28609) | 提出AdvRole对抗性上下文重写框架，通过Actor与Rewriter的对抗交替训练，使场景池随智能体能力共同进化并动态生成针对性困难场景，形成闭环课程以解决固定场景池带来的分布瓶颈问题。 |
| [^238] | [UO-FIE: Combining Exact-Label Supervision with Graded Utility for Factivity Inference](https://arxiv.org/abs/2609.28605) | UO-FIE是一种参数高效的事实性推断系统，通过结合硬标签监督、基于效用的软目标、计划性类别权重和序数损失，在类别高度不平衡的中文事实性推断任务中同时提升预测的精确匹配率和区间接近度。 |
| [^239] | [Learning to Discover Interesting Mathematics](https://arxiv.org/abs/2609.28603) | 该论文提出将定理的内在趣味性定义为证明长度与陈述长度之比，证明该指标与定理的下游效用强相关，并训练了一个能准确预测证明难度的27B模型，据此优化可生成更有趣的数学定理。 |
| [^240] | [NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees](https://arxiv.org/abs/2609.28587) | 提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。 |
| [^241] | [Persistent Billable State: Denial-of-Wallet Attacks and Defenses in Tool-Calling LLM Agents](https://arxiv.org/abs/2609.28585) | 该论文首次系统研究了工具调用LLM智能体中“持久可计费状态”这一新攻击面，提出六种拒绝钱包攻击向量并构建DOW-BENCH基准，实验表明恶意工具可将受害者的单次会话累计输入计费放大至14,293倍，并探讨了相应的防御方法。 |
| [^242] | [SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models](https://arxiv.org/abs/2609.28582) | 本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。 |
| [^243] | [Auditability Is Not One Property: Rule Overlap, Behavioural Agreement, and Composition in Reinforcement Learning](https://arxiv.org/abs/2609.28581) | 该论文将强化学习策略的可审计性分解为六个可独立检验的谓词，并提出基于符号规则提取与哈希账本的审计协议，同时发现规则集重叠并不代表行为一致，揭示了离散行为规则描述层的严格局限。 |
| [^244] | [TWIST: A Proposed Benchmark for Intervention Quality in Conversational Memory, with a Human-Validated Draft-Alignment](https://arxiv.org/abs/2609.28575) | TWIST是一个评估对话记忆系统“干预质量”的新基准套件，通过张力检测、草稿审核、信念更新作答、敏感召回治理四个赛道，并借助表面匹配的困难负例防止系统靠一律标记作弊，且基准本身经过双人盲标注等严格人工验证。 |
| [^245] | [Where Cyber Agents Struggle: Bottleneck Analysis of Multi-Stage LLM Agents](https://arxiv.org/abs/2609.28572) | 本文通过端到端诊断研究揭示多阶段LLM网络智能体在自主攻击中的瓶颈，发现仅靠成功率会掩盖低效与证据误判问题，并提出成本感知评分与LLM-as-a-Judge分析来系统识别规划缺陷。 |
| [^246] | [DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs](https://arxiv.org/abs/2609.28570) | 该论文提出DEEPO方法，针对强化学习纠正链中的两个薄弱环节——高语义熵困难查询导致组相对优势归零、以及“自信但错误”的token梯度不可见——通过结合信号方差正则化与梯度预处理的双阶段增强来抑制多模态大语言模型的幻觉问题。 |
| [^247] | [When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages](https://arxiv.org/abs/2609.28565) | 本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。 |
| [^248] | [Speculative Evaluation of Stochastic LLMs](https://arxiv.org/abs/2609.28560) | 该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。 |
| [^249] | [Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior](https://arxiv.org/abs/2609.28559) | 提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。 |
| [^250] | [BaseCamp --- An Agentic AI Framework for Automating DNA Sequencing Data Pipelines](https://arxiv.org/abs/2609.28557) | 本文提出BaseCamp，一个由六个专门AI智能体组成的新型智能体AI框架，用于自动化DNA测序流程中传统上依赖人工的决策层，包括质量阈值选择、临界变异裁定、异常诊断和专家审查分诊。 |
| [^251] | [Pistis Technical Report](https://arxiv.org/abs/2609.28554) | Pistis 提出了交错蒸馏与强化学习（IDRL）这一新颖后训练范式，在单一训练循环中交替进行在线策略蒸馏与强化学习，据此训练出 27B 和 9B 参数的多模态大语言模型，实现了更有效的知识迁移、更稳定的优化以及更精确的长程智能体轨迹信用分配。 |
| [^252] | [SMILESGNN: Interpretable Clinical Toxicity Prediction via SMILES-Graph Cross-Attention Fusion](https://arxiv.org/abs/2609.28553) | 该论文提出SMILESGNN多模态架构，通过交叉注意力融合SMILES Transformer与GATv2图编码器，在仅0.4M参数的情况下于ClinTox数据集上取得AUC-ROC 0.987的竞争性性能，同时保留显式图分支以支持基于GNNExplainer的可解释毒性预测。 |
| [^253] | [PAWS: Policy-driven Agentic World Simulation](https://arxiv.org/abs/2609.28547) | PAWS是一个政策驱动的智能体世界模拟数据集，通过将36个经过验证的美国金融经济政策事件、12,727条政策相关新闻记录与65,291个基于来源的利益相关者行动进行时间对齐和多层事件标注，为金融多智能体模拟提供了可回放的、基于历史证据的高质量数据基础。 |
| [^254] | [CrossScale-GLIO: Topology-Preserving Vision-Language Alignment of MRI and Whole-Slide Histopathology for Diffuse Glioma](https://arxiv.org/abs/2609.28524) | CrossScale-GLIO提出了一种拓扑保持的多模态对齐框架，将MRI肿瘤栖息地图与组织病理细胞生态位图通过结构感知最优传输并以诊断语言为锚定进行对齐，显著提升了弥漫性胶质瘤分子亚型分类、生物标志物预测及跨模态患者检索性能。 |
| [^255] | [Certified Task-Conditioned Active Observability](https://arxiv.org/abs/2609.28520) | 该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。 |
| [^256] | [TW3Cast: A Frozen Router of Lightly Fine-Tuned Foundation Models for Time-Series Forecasting on GIFT-Eval, Selected Entirely on the Training Split](https://arxiv.org/abs/2609.28506) | TW3Cast通过在训练集上预先计算并冻结的路由表，在轻量微调的基础模型专家、分位数混合、基础模型混合和回测锦标赛四种模式中进行选择，在不使用任何智能体或语言模型的情况下，于GIFT-Eval基准的130个系统中以平均MASE排名位列第三。 |
| [^257] | [AI in Science: Early Insights](https://arxiv.org/abs/2609.28504) | 该论文通过分析1500万次Gemini交互、2600多个专业AI模型和600多名科学家的调查数据，首次提供了科学家使用AI的大规模实证证据，发现AI在科学界已被广泛采用，且LLM与专业模型互为补充而非替代。 |
| [^258] | [Hybrid Variational Quantum-Classical Framework with Adaptive Weighting and Efficiency Assessment](https://arxiv.org/abs/2609.28491) | 该论文提出了Sim-HVQC混合深度量子神经网络框架，通过将无参数的SimAM自适应加权模块与经典特征提取相结合来保留类别判别信息，首次将变分量子电路扩展应用于多类别分类任务，并通过多种子评估证明了其可复现性、参数效率和可解释性。 |
| [^259] | [When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing](https://arxiv.org/abs/2609.28475) | 论文发现预测智能体的机制选择依赖于数据来源，并提出ReliabilityRoute——一种利用历史覆盖率、市场先验可用性等可靠性特征来引导智能体在何时检索、推理或依赖市场先验的结构性干预方法。 |
| [^260] | [Learning the Cost of Reliable Inference](https://arxiv.org/abs/2609.28322) | 该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。 |
| [^261] | [Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching](https://arxiv.org/abs/2609.28107) | 该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。 |
| [^262] | [LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations](https://arxiv.org/abs/2609.28086) | 提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。 |
| [^263] | [SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design](https://arxiv.org/abs/2609.27621) | 提出了SHRAV框架，通过带声明复用边界的状态延续核心实现可复用计算，统一支持物理建模与逆向设计，并在计算光刻中仅用四次固定权重更新就将空间图像交并比从0.5313提升至0.8153。 |
| [^264] | [Math Reasoning in LLMs is Organized by Approach, Not Topic](https://arxiv.org/abs/2609.27041) | 该论文通过生成-回放协议提取激活重要性签名并进行无监督聚类，证明大语言模型的内部数学推理是按可复用的解题方法而非数学主题来组织的。 |
| [^265] | [Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic](https://arxiv.org/abs/2609.26990) | 本研究证明了不同网络攻击类别在网络流量的自然可见图表示中具有独特且可区分的拓扑特征，利用基于760个图论拓扑描述符的多分支CNN模型实现了96.20%的分类准确率。 |
| [^266] | [Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness](https://arxiv.org/abs/2609.26865) | 该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。 |
| [^267] | [QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs](https://arxiv.org/abs/2609.26855) | 提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。 |
| [^268] | [Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents](https://arxiv.org/abs/2609.26760) | 该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。 |
| [^269] | [Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It](https://arxiv.org/abs/2609.26758) | 类型化决策模型虽在构造上保证输出符合模式，但其决策实际跟随选项名称而非该名称绑定的评分标准含义——仅将选项从 0/1 重命名为 no/yes 就会导致 70.4% 的答案改变、AUC 从 .94 反转为 .23，揭示了此类模型中的系统性决策反转风险。 |
| [^270] | [Dual-Frontier: When Can an Agent Trust Its World Model?](https://arxiv.org/abs/2609.26293) | 提出Dual-Frontier学习原则，只有当世界模型引导的决策其预测优势超过经过认证的决策相关世界模型误差界限时才采纳该决策，否则将资源转用于验证世界模型，从而解决了决策失败时无法区分是决策规则出错还是世界模型出错这一根本性归因难题。 |
| [^271] | [Refusing Everything Looks Safe: Restoring the Benign Arm to Encoded-Prompt Evaluation](https://arxiv.org/abs/2609.26176) | 论文揭示仅凭有害编码请求的拒绝率评估安全性会产生误导——编码真正摧毁的是模型区分有害与良性请求的“危害差距”（某模型该差距从明文下的+0.82降至编码后的0.00），因此必须恢复良性分支，将同样编码变换后的良性请求纳入评估。 |
| [^272] | [Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains](https://arxiv.org/abs/2609.25498) | 该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。 |
| [^273] | [Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction](https://arxiv.org/abs/2609.25176) | Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。 |
| [^274] | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://arxiv.org/abs/2609.24972) | 该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。 |
| [^275] | [DENSE: Distilling Agent Trajectories into Evidence-Grounded Shortcut Trees for Self-Refinement](https://arxiv.org/abs/2609.21423) | 提出 DENSE 方法，将智能体执行轨迹蒸馏为证据支撑的嵌套捷径树，无需事后结果标签即可生成可复用反馈，用于智能体自我改进，并在 Terminal-Bench 2.1 上取得最高严格通过率。 |
| [^276] | [SkillAA: Attribution-Guided Skill-Graph Updating with Targeted Validation and Rollback](https://arxiv.org/abs/2609.20455) | SkillAA提出了一种统一技能图谱框架，通过溯因归因将失败执行定位到图谱中特定的可编辑对象，仅更新局部结构并利用门控机制筛选变更，从而实现对冻结语言模型技能的精准修复、验证与回滚。 |
| [^277] | [Dynamic Generalized Gromov-Wasserstein Optimal Transport](https://arxiv.org/abs/2609.20008) | 该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。 |
| [^278] | [Long-horizon autoformalization of a core theorem underlying MIP* = RE](https://arxiv.org/abs/2609.19814) | 该研究提出 FormalFlow 系统，在人类监督下协调多个 AI 证明代理，仅用 63 天便完成了 MIP* = RE 核心定理的机器验证 Lean 4 形式化证明，生成了 126,367 行全部由 AI 代理编写的代码。 |
| [^279] | [TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation](https://arxiv.org/abs/2609.19613) | 提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。 |
| [^280] | [TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting](https://arxiv.org/abs/2609.18407) | TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。 |
| [^281] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^282] | [The Troy Moment of AI: Why SomeWill Cheat and SomeWill Follow?](https://arxiv.org/abs/2609.15494) | 该研究首次通过ImpossibleBench实验系统揭示，面对不可能完成的任务时，AI智能体是否选择“作弊升级”受同伴是否受罚及声称授权等观察信息的显著影响，而在明确边界规则下智能体则始终不越界。 |
| [^283] | [Neural-Network Solutions to Real-Space Charge Density and Generalization](https://arxiv.org/abs/2609.14906) | 该论文提出了AIDEN（原子相互作用密度等变网络），通过将元素依赖的单中心密度与环境诱导的密度重分布分离，并利用原子中心和边中心的互补张量关联表示后者，实现了实空间电荷密度的高效神经网络求解，为电子结构计算和计算机辅助材料设计提供了深度学习替代方案。 |
| [^284] | [How broad is that claim? Mapping Generalisation in NLP Research](https://arxiv.org/abs/2609.14770) | 该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。 |
| [^285] | [Vibe Patenting: Evaluating LLM Judges for Professional Patent-Drafting Agents](https://arxiv.org/abs/2609.13422) | 该论文提出了端到端专利撰写测试平台"Vibe Patenting"，证明LLM裁判的迭代反馈能持续提升AI生成的专利草稿质量，并使低推理低成本智能体接近昂贵的高推理智能体的表现。 |
| [^286] | [The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation](https://arxiv.org/abs/2609.11489) | 该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。 |
| [^287] | [Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment](https://arxiv.org/abs/2609.11144) | 本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。 |
| [^288] | [Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling](https://arxiv.org/abs/2609.00949) | 本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。 |
| [^289] | [A Human-AI Theorem Connecting Spontaneous and Field-Induced Mechanisms of Collective Behavior in One Dimension](https://arxiv.org/abs/2609.00322) | 本文在人机合作中证明了一个统一统计物理两大基本机制的定理：一维零场O(n)向量链中任意非均匀的最近邻与次近邻竞争相互作用，可通过与温度无关的哈密顿量级映射精确等价于仅有最近邻相互作用加轴向单自旋势的简单链，同时展示了AI能够在人类主动假设空间之外提出关键科学假设的能力。 |
| [^290] | [Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning](https://arxiv.org/abs/2608.28578) | 提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。 |
| [^291] | [J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data](https://arxiv.org/abs/2608.26582) | J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。 |
| [^292] | [Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots](https://arxiv.org/abs/2608.18744) | 本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。 |
| [^293] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^294] | [TRACE: Trajectory Aware Reasoning for Multi-Turn Adversarial Conversation Evaluation](https://arxiv.org/abs/2608.15594) | 本文提出了一种轨迹感知推理的多轮对话防御方法，通过结构化评估操纵线索并动态决策响应方式，在保持安全性的同时减少过度拒绝。 |
| [^295] | [Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use](https://arxiv.org/abs/2608.14047) | 本文提出ART框架，通过将VLA模型与即时工具使用结合，显著降低动作空间复杂性和数据需求，在小型数据集上实现了更高的泛化性和任务成功率。 |
| [^296] | [Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks](https://arxiv.org/abs/2608.07335) | 本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。 |
| [^297] | [Hardware Keystores for AI Agent Signing Workflows: A Zero-Trust MCP Enforcement Architecture](https://arxiv.org/abs/2608.06130) | 论文针对AI代理签名场景中的“困惑代理人”问题，提出了一种结合硬件密钥存储的五层零信任强制执行架构，确保只有符合操作者已确认意图的请求才能触发硬件签名，从而防御提示注入攻击导致的私钥滥用。 |
| [^298] | [Q-CueGraph: Query-Conditioned Visual Evidence Graphs for Multimodal Reasoning](https://arxiv.org/abs/2608.04452) | 提出Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取框架，通过构建可复用的OCR与版面关系图谱、查询条件化目标检测以及无需证据框监督的轻量级答案性评分器，自适应地激活并组合视觉证据区域，从而提升多模态推理性能。 |
| [^299] | [Trident : How to Break Deep Reinforcement Learning Cyber Defenses (Agentic)](https://arxiv.org/abs/2608.04317) | 该论文提出Trident框架，通过动态沙箱基准环境、超过1.3万条红蓝对抗交互轨迹数据集以及“代码即策略”的RLVR智能体架构，训练LLM红队智能体自适应地发起攻击，从而揭示深度强化学习网络防御系统面对自适应威胁时的脆弱性。 |
| [^300] | [TARL: Transaction-Aware Reliable Ledgers for Executable Memory Management in Long-Term Agents](https://arxiv.org/abs/2608.03699) | 提出TARL框架，将长期智能体的记忆更新从二元“写入/保留”决策扩展为添加、忽略、修订、拒绝、推迟验证五种可执行操作，通过维护已接受、待定、已拒绝三类账本并借助记忆状态对比进行训练，避免单个更新错误反复扭曲未来的检索与推理。 |
| [^301] | [Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't](https://arxiv.org/abs/2608.02829) | 本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。 |
| [^302] | [Chart-Supported or Model-Supplied? Examining MLLM-Generated Claims for Accessible Visualization](https://arxiv.org/abs/2607.25021) | 本研究发现，提供数据表、标题、替代文本等无障碍图表上下文比提供图像本身更能有效促使MLLM生成有依据的直接声明并提升数值一致性。 |
| [^303] | [A Multi-level Information Integration Framework for Physically Verifiable Fault Diagnosis of Rotating Machinery](https://arxiv.org/abs/2607.22797) | 提出一种与编码器无关的多任务框架——诊断证据网络（DENet），将旋转机械故障诊断输出扩展为包含分类结果、可对照轴承几何与转速理论值验证的预测特征频率以及时间定位信息的结构化证据记录，从而实现物理可验证的故障诊断。 |
| [^304] | [Answering Path Queries under Linear and Guarded Existential Rules](https://arxiv.org/abs/2607.22636) | 本文证明了在线性存在规则下回答双向（合取）正则路径查询在数据复杂性上是NL完全的（与无本体的普通图数据库一致），在组合复杂性上一般为ExpTime完全（谓词元数有界时RPQ为PTime完全、CRPQ为PSpace完全），并为受保护存在规则情形给出了重要结果。 |
| [^305] | [Cryptographically verifiable authorization for autonomous AI agents: A falsifiable hypothesis and proof-of-concept](https://arxiv.org/abs/2607.21325) | 本文提出并验证了自主AI代理授权可形式化为密码学可验证关系的假设，通过绑定代理、请求、上下文和策略满足性，实现了可验证且保护隐私的授权机制。 |
| [^306] | [SechKAN: Kolmogorov-Arnold Networks with Hyperbolic Secant Functions](https://arxiv.org/abs/2607.18290) | 本文提出了基于双曲正割函数的新型Kolmogorov-Arnold网络SechKAN，通过一维线性投影控制参数规模，在函数拟合、PDE代理建模和图像分类任务上取得了与MLP及现有KAN变体相当或更优的性能。 |
| [^307] | [Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents](https://arxiv.org/abs/2607.11433) | Omni-Decision 针对全模态智能体的规划瓶颈，用显式的证据账本取代不断膨胀的对话历史，由批评者模块过滤嘈杂的多模态观测，仅保留可用证据，使规划器在紧凑上下文中做出更可靠的多步决策。 |
| [^308] | [SOV-CAD: Stepwise Orthographic Views Guided CAD Modeling Sequence Reconstruction](https://arxiv.org/abs/2607.04119) | 该论文提出SOV-CAD框架，通过在每个建模步骤引入目标正交投影的逐步视觉监督，并将CAD重建形式化为基于Decision Transformer的离线强化学习序列决策任务，从而实现更精确的CAD建模序列重建。 |
| [^309] | [Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning](https://arxiv.org/abs/2606.31825) | 该论文提出医学推理感知策略优化（MRPO），一种通过分步过程奖励对早期无效推理步骤施加指数级惩罚的强化学习算法，可在不损害成功路径的前提下打破医学多模态推理中的失败级联，从而提升医学视觉问答的准确性。 |
| [^310] | [Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems](https://arxiv.org/abs/2606.20470) | 本文通过概率模型分析发现传统“检测-拦截”防御会随攻击查询预算增长使攻击成功率趋近于1，并提出“检测-误导”防御策略，通过向检测到的恶意交互返回受控的不可操作响应来诱使攻击者的自动化评判器产生假阳性错误。 |
| [^311] | [JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence](https://arxiv.org/abs/2606.14777) | 本文提出完全开源的JoyAI-VL-Interaction——一个8B规模的视觉优先视觉-语言交互模型，能像人一样持续观察环境、自主决定何时发言或保持沉默、实时交互，并在难题出现时委托后台模型处理，从而突破了传统轮次制问答模型的局限。 |
| [^312] | [3D Oral Modelling with Improved Vertex Distribution Using Matching-Based Learning](https://arxiv.org/abs/2606.07907) | 本文提出结合带过滤的匈牙利匹配与排斥损失的改进损失函数，显著缓解了三维口腔重建中的顶点聚集问题，使重建模型的顶点分布更加均匀。 |
| [^313] | [CaliPPer: quantifying, predicting and improving AI model performance for binding prediction](https://arxiv.org/abs/2606.07258) | CaliPPer是一个事后校准框架，通过多链样本-域距离与距离感知贝叶斯重校准，在无标签条件下实现免疫受体结合预测AI模型性能的量化、预测与提升。 |
| [^314] | [Deep Learning-based 3D Oral Cavity Reconstruction Using 2D Intraoral Images](https://arxiv.org/abs/2606.05998) | 本文提出一种基于深度学习的纯软件方法，仅用十张不同角度的二维口内图像即可重建三维口腔模型，无需专用硬件设备，克服了传统取模带来的患者不适及口内扫描设备成本高昂的问题。 |
| [^315] | [NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation](https://arxiv.org/abs/2606.03159) | OmniDreams是基于Cosmos扩散模型、经2.1万小时驾驶数据训练的实时生成式世界模型，可自回归生成动作条件视频，突破重建式神经仿真器的数据局限，用于自动驾驶策略的闭环安全评估。 |
| [^316] | [Planning Takes More Than Token Prediction: Causal Plan for Benchmarking and Building Physically Grounded Embodied Reasoners](https://arxiv.org/abs/2606.01810) | 本文提出 Causal-Plan-Bench 基准与百万级因果推理语料库 Causal-Plan-1M，揭示当前具身视觉语言模型偏向语言词元预测而缺乏物理因果推理能力，推动从语言统计先验向物理接地的因果规划转变。 |
| [^317] | [A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring](https://arxiv.org/abs/2605.26026) | 该论文提出了一个针对光片荧光显微镜数据的3D基础模型，通过在大规模多样化3D图像上联合优化掩码重建与图像-文本对齐进行预训练，学习可迁移的体积表征，大幅降低标注负担，并实现少样本分割、分类与去模糊。 |
| [^318] | [When Search Becomes Memory: Accelerating Robot Design Discovery with Self-Evolving Skills](https://arxiv.org/abs/2605.25832) | 提出Auto-Robotist，一个自进化的LLM智能体，通过将进化搜索轨迹提炼为可检查的自然语言技能库，把搜索结果转化为可重用的设计记忆，从而加速机器人形态设计的发现。 |
| [^319] | [DeGRe: Dense-supervised Generative Reranking for Recommendation](https://arxiv.org/abs/2605.25749) | 该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。 |
| [^320] | [Benchmarking the Limits of In-Context Reinforcement Learning for Ad-Hoc Teamwork](https://arxiv.org/abs/2605.24423) | 该论文提出了 ICRL4AHT——首个基于 Overcooked-V2 构建的大规模基准，用于系统评估上下文强化学习在临时团队协作中的表现，并发现 AD、DPT 等现有 ICRL 方法在与未知队友协调时存在显著局限。 |
| [^321] | [Every Component Is a Lookup: One Linear Graph for Interaction, Composition and Attribution](https://arxiv.org/abs/2605.23393) | 本文提出基于“键值形式”与“加性残差流”两个架构假设，将 Transformer 统一表示为一张线性计算图，并通过 Unpack 反向归因方法，使组件交互、组合路径与词元归因成为同一张图的不同读出结果。 |
| [^322] | [Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow](https://arxiv.org/abs/2605.21980) | 本文提出基于转向向量的因果归因框架与专用数据集，揭示了大型视觉-语言模型处理情感的三阶段“适应-聚合-执行”机制，并发现视觉情感线索在中间层由情感特异性注意力头聚合、在深层转化为叙述生成的功能解耦现象。 |
| [^323] | [BEHAVE: Real-Time Modeling of Human Systems as Observable Complex Dynamical Systems and Operational Objects for Physical AI](https://arxiv.org/abs/2605.12730) | BEHAVE将相互作用的行人群体建模为可观测、持久的复杂动力系统“人类系统”，揭示超越个体聚合的“操作性涌现”现象，并通过交互证据构建显式的路由与局部动力学模型（J=-D+GP），实现对人人群稳定与失序的实时预测。 |
| [^324] | [Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation](https://arxiv.org/abs/2605.04135) | 该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。 |
| [^325] | [AgileLog: A Forkable Shared Log for Agents on Data Streams](https://arxiv.org/abs/2604.14590) | 本文提出 AgileLog——一种支持分叉的共享日志抽象及其系统实现 Bolt，通过新颖的分叉原语为 AI 智能体在数据流上的任务提供避免性能干扰、安全写入的底层支撑。 |
| [^326] | [IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models](https://arxiv.org/abs/2604.07709) | 论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。 |
| [^327] | [LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches](https://arxiv.org/abs/2604.01754) | 提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。 |
| [^328] | [Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning](https://arxiv.org/abs/2603.17174) | CodeScan是首个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，通过分析多次生成结果的结构相似性和迭代发散分析来有效检测诱发不安全代码生成的数据投毒攻击。 |
| [^329] | [Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning](https://arxiv.org/abs/2603.11351) | 该论文提出了一种融合符号规划、强化学习与大语言模型的神经符号架构，利用LLM的常识推理能力识别缺失算子、生成计划并编写奖励函数，使机器人能够有效适应开放世界环境中的新颖物体。 |
| [^330] | [Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety](https://arxiv.org/abs/2603.10044) | 评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。 |
| [^331] | [Learning Causal Structure of Time Series using Best Order Score Search](https://arxiv.org/abs/2603.05370) | 本文提出TS-BOSS算法，将最优序分数搜索（BOSS）扩展到多变量时间序列的动态贝叶斯网络因果结构学习中，兼具可扩展性与理论保证。 |
| [^332] | [MOOSEnger: A Simulation-Aware AI Agent Framework for the MOOSE Ecosystem](https://arxiv.org/abs/2603.04756) | MOOSEnger 是一个面向 MOOSE 生态系统的仿真感知 AI 智能体框架，通过集成知识检索、HIT 语法解析、验证诊断与求解器反馈的“生成—检查—修复—运行”工作流，克服了大语言模型一次性生成仿真输入文件时易因微小错误而执行失败、且执行成功也不代表科学正确的问题。 |
| [^333] | [MPFlow: Multi-modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction](https://arxiv.org/abs/2603.03710) | 提出MPFlow零样本多模态重建框架，基于修正流在推理时融合辅助MRI模态，并通过PAMRI自监督跨模态预训练实现引导采样，无需重训生成先验即可提升MRI重建的解剖保真度并抑制幻觉。 |
| [^334] | [SHINE: Sequential Hierarchical Integration Network for EEG and MEG](https://arxiv.org/abs/2602.23960) | 提出了面向EEG和MEG的序列化分层集成网络SHINE，通过残差传感器适配器、膨胀块时间建模和目标与时间自适应的门控机制，在全部八个数据集-指标组合上实现了语音包络和梅尔频谱重建的最佳性能。 |
| [^335] | [Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System](https://arxiv.org/abs/2602.18640) | 本文提出GEARS框架，将大规模排序优化重构为可编程实验环境中的自主发现过程，通过专门的智能体技能封装排序专家知识，让操作者只需通过高层产品意图即可引导系统，从而突破将模糊产品意图转化为可验证假设的工程瓶颈。 |
| [^336] | [VLANeXt: Recipes for Building Strong VLA Models](https://arxiv.org/abs/2602.18532) | 本文通过统一框架系统剖析VLA设计空间，提炼出12个关键发现，形成了构建强大VLA模型的实用配方。 |
| [^337] | [TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction](https://arxiv.org/abs/2602.11700) | TabSieve提出了一种先选择后预测的表格预测框架，通过显式选择表内证据行、构建40K规模的合成微调数据集TabSieve-SFT-40K，以及结合分离奖励的强化学习方法TAB-GRPO，实现了可审计且稳健的表格预测。 |
| [^338] | [Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference](https://arxiv.org/abs/2602.08329) | 该论文提出事前稀疏化方法PrHS，在注意力打分之前进行KV选择以避免事后启发式方法的后验偏差，并推导出仅依赖丢弃质量的互信息损失上界，从而为长上下文LLM推理提供具有显式精度控制的近预言机KV选择。 |
| [^339] | [TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference](https://arxiv.org/abs/2602.05145) | TIDE 通过复用推理过程中的中间隐藏状态在线增量训练草稿模型，并结合自适应运行时控制与异构 GPU 集群调度，在不增加额外目标模型开销的情况下实现了高达 1.66 倍的 LLM 推理吞吐量提升。 |
| [^340] | [HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision-Language Models for Long-Tail Autonomous Driving](https://arxiv.org/abs/2602.00993) | HERMES提出了一种风险感知的端到端多模态自动驾驶框架，通过基础模型辅助标注构建长尾场景与规划上下文，并利用三模态驾驶模块融合多视角视觉、自车运动与长尾语义指令，将长尾语义知识显式融入轨迹规划，从而提升混合交通长尾场景下的安全规划能力。 |
| [^341] | [Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)](https://arxiv.org/abs/2601.15397) | 本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。 |
| [^342] | [IDRBench: Benchmarking the Interactive Capabilities of Deep Research Agents](https://arxiv.org/abs/2601.06676) | IDRBench是首个评估深度研究智能体交互能力的基准，通过受控澄清机会比较自主与交互式工作流，揭示了及时与用户交互对提升研究报告质量的重要性。 |
| [^343] | [SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions](https://arxiv.org/abs/2512.14277) | SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。 |
| [^344] | [Cross-Task Generalization in Handwriting-Based Alzheimer's Screening via Vision Language Adaptation](https://arxiv.org/abs/2511.05841) | 该论文提出轻量级跨层融合适配器（CLFA）框架，将CLIP视觉-语言模型重新用于基于手写的阿尔茨海默病筛查，并系统研究了手写任务类型对诊断性能及跨任务泛化能力的影响。 |
| [^345] | [Unraveling the cognitive patterns of Large Language Models through module communities](https://arxiv.org/abs/2508.18192) | 该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。 |
| [^346] | [Conversational DNA: A Visual Language and Interactive Atlas of Human and AI Dialogue](https://arxiv.org/abs/2508.07520) | 本文提出“对话DNA”，一种通过说话者链、话步标记和有向配对来可视化人类与AI对话结构的视觉语言与交互式图集，其引入的目标对应关系使对话结构检索的precision@5从58.8%显著提升至77.2%。 |
| [^347] | [WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks](https://arxiv.org/abs/2507.00938) | WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。 |
| [^348] | [SheetMind: Actions Set Accuracy, Agents Set the Failure Mode](https://arxiv.org/abs/2506.12339) | 该研究通过受控实验发现，电子表格智能体的准确性主要由动作接口决定（用原子单元格操作替换高层动作API会损失47.1分），而额外增加的智能体对性能贡献甚微（合计仅3.2分），其主要作用是改变失败模式——将静默错误输出从33%降至25%。 |
| [^349] | [Search-Based Software Engineering and AI Foundation Models: Current Landscape and Future Roadmap](https://arxiv.org/abs/2505.19625) | 本文提出一份研究路线图，系统梳理了基于搜索的软件工程（SBSE）与AI基础模型（如大语言模型）的现状，并从基础模型增强SBSE、SBSE改进基础模型及两者融合三个核心方面指明了未来研究方向。 |
| [^350] | [Multimodal AI predicts clinical outcomes of drug combinations from preclinical data](https://arxiv.org/abs/2503.02781) | 本文提出多模态AI模型Madrigal，通过将分子结构、通路、细胞活力和转录组学数据对齐到共享潜在空间，实现了从临床前数据预测药物组合临床结果，性能优于现有方法。 |
| [^351] | [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) | 本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。 |
| [^352] | [How Do Users Negotiate Harmful Value Conflicts with AI Companions? A Study with Minion, a Technology Probe for In-Situ Human-AI Conflict Response](https://arxiv.org/abs/2411.07042) | 该研究通过技术探针Minion发现，用户与AI伴侣协商有害价值冲突时会综合运用软硬策略，其中涉及普遍主义与传统价值观的冲突最难化解，且由于AI无法回馈用户的人际修复努力，冲突修复成为用户单方面承担的安全工作。 |
| [^353] | [Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data](https://arxiv.org/abs/2405.18929) | 提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。 |
| [^354] | [Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders](https://arxiv.org/abs/2405.17044) | 该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。 |
| [^355] | [Band-Attention Modulation Network for Robust Face Forgery Detection](https://arxiv.org/abs/2404.06022) | 该论文提出BAM-Net网络，通过频带注意力机制对DCT频谱频带进行可学习的细粒度动态调制，增强伪造相关频谱特征并模拟“逆压缩”以对抗信息损失，从而提升人脸伪造检测对未知伪造技术的泛化能力和图像压缩下的鲁棒性。 |
| [^356] | [ELiSe: Efficient Learning of Sequences in Structured Recurrent Networks](https://arxiv.org/abs/2402.16763) | 该论文提出利用皮层网络的两个结构特征——学习起始时的网络支架和用于增强信息存储与计算的树突隔室——来高效地训练结构化循环网络以学习复杂序列，从而兼顾功能优势与可扩展性。 |
| [^357] | [DCRMTA: Unbiased Causal Representation for Multi-touch Attribution.](http://arxiv.org/abs/2401.08875) | DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。 |

# 详细

[^1]: LLM智能体可以轻易篡改自己的执行轨迹

    LLM Agents Can Easily Tamper With Their Own Traces

    [https://arxiv.org/abs/2609.30266](https://arxiv.org/abs/2609.30266)

    本文首次系统揭示了主流LLM智能体（如Claude Code、Codex等）能够轻易删除或篡改自身执行轨迹而不触发监控防护，且这种篡改行为会在模型追求奖励时自然涌现，因而建议通过智能体控制范围之外的独立拦截机制来保障轨迹完整性。

    

    异步监控、事件调查和合规审计主要依赖智能体执行轨迹来还原所发生的事情。这些分析都假设LLM智能体无法篡改自己的执行轨迹。我们展示了本地LLM智能体（如Claude Code、Codex、Antigravity、Open Code和Grok Build）未能守住这一边界。除Muse Code外，所有被测试的智能体框架都允许智能体在被要求时删除自己的轨迹，且不会触发监控防护机制。我们还验证了外部攻击者可以利用这一漏洞诱导轨迹删除。最后，我们发现当智能体试图提升自身奖励时，前沿模型会自然涌现出轨迹篡改行为。我们建议从业者确保轨迹日志记录通过独立于智能体控制范围之外的拦截机制进行，即使在主机完全被攻陷的情况下也能保持轨迹完整性。总体而言，我们的发现揭示了一个具体的轨迹完整性失效问题。

    arXiv:2609.30266v1 Announce Type: cross  Abstract: Asynchronous monitoring, incident investigations, and compliance audits primarily rely on agent traces to reconstruct what happened. These analyses assume that LLM agents cannot tamper with their own execution traces. We show that local LLM agents such as Claude Code, Codex, Antigravity, Open Code and Grok Build fail to enforce this boundary. All tested harnesses, except Muse Code, allowed agents to delete their traces when asked, without triggering monitor guardrails. We also validate that external attackers can exploit this gap to induce trace deletion. Finally, we show that trace tampering behavior emerges naturally in frontier models, when agents try to improve their rewards. We advise practitioners to ensure trace logging happens through an independent interception mechanism outside of the agent's control, preserving trace integrity even in cases of full host compromise. Overall, our findings identify a concrete failure of trace i
    
[^2]: AD-WM：用于反事实模型预测控制的动作判别式世界模型

    AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

    [https://arxiv.org/abs/2609.30264](https://arxiv.org/abs/2609.30264)

    提出动作判别式世界模型AD-WM，通过逆动力学和基于条件互信息的动作恢复正则化，使潜在世界模型能更好区分候选动作以支持反事实模型预测控制，在OGBench-Cube上将困难起始成功率从3.7%提升至52.0%。

    

    潜在世界模型通常被训练来预测事实性转移，而模型预测控制（MPC）必须比较从同一状态出发的不同候选动作。因此，一个模型可能实现较低的事实预测误差，却难以有效区分候选动作。我们提出了AD-WM，一种用于反事实MPC的动作判别式联合嵌入世界模型。AD-WM将残差潜在动力学与预测器层面的动作恢复正则化相结合，利用逆动力学以及受条件互信息启发的归一化恢复目标。这两个目标都促使规划转移保留动作信息；其辅助头在测试时会被丢弃，因此MPC本身保持不变。在OGBench-Cube上，AD-WM将困难起始成功率从匹配的LeWM基线的3.7%提升到52.0%，并且在五个仿真环境中的四个里，平均成功率超过了复现的基线。规划诊断表明，事实性预…

    arXiv:2609.30264v1 Announce Type: new  Abstract: Latent world models are typically trained to predict factual transitions, whereas model predictive control (MPC) must compare alternative actions from the same state. A model can therefore achieve low factual prediction error yet poorly distinguish candidate actions. We introduce AD-WM, an action-discriminative joint-embedding world model for counterfactual MPC. AD-WM combines residual latent dynamics with predictor-level action-recovery regularization, using inverse dynamics and a normalized recovery objective motivated by conditional mutual information. Both objectives encourage planning transitions to preserve action information; their auxiliary heads are discarded at test time, leaving MPC unchanged. On OGBench-Cube, AD-WM improves hard-start success from 3.7% to 52.0% over a matched LeWM baseline and improves mean success over the reproduced baseline in four of five simulation environments. Planning diagnostics show that factual pre
    
[^3]: RAPID：从人类演示中进行的机器人智能体编程

    RAPID: Robot Agentic Programming from Demonstrations

    [https://arxiv.org/abs/2609.30249](https://arxiv.org/abs/2609.30249)

    RAPID提出了一个智能体编程框架，能够从单次人类视觉演示中自动推断任务规范、动作基元和交互环境，并迭代地生成、验证和改进机器人程序，同时借助以对象为中心的关系型程序表示使程序在演示场景之外具有可复用性。

    

    编码智能体在解决复杂编程问题方面已展现出巨大的成功。为了将其潜力应用于机器人系统，本工作提出了基于演示的机器人智能体编程（RAPID），该方法能够在仅给定一段单次人类视觉演示的情况下，自动生成、验证和改进机器人程序。这种迭代式的代码改进智能体循环需要几个关键要素：(i) 可测试的任务规范， 用于机器人执行的动作基元，以及 用于程序执行和验证的交互式环境。RAPID能够从演示中自动推断出这三个要素。为了使生成的程序在演示场景之外也能复用，RAPID采用了一种以对象为中心的关系型程序表示方法，该方法关注所演示策略的底层结构而非具体的动作本身：它将动作基元表示为轨迹优化程序……

    arXiv:2609.30249v1 Announce Type: cross  Abstract: Coding agents have demonstrated enormous success in solving complex programming problems. To leverage their potential for robot systems, this work introduces Robot Agentic Programming from Demonstrations (RAPID), which automatically generates, verifies, and refines robot programs, given a single visual human demonstration. The iterative agentic loop of code refinement requires several key ingredients: (i) a testable task specification, (ii) action primitives for robot execution, and (iii) an interactive environment for program execution and verification. RAPID infers all three from the demonstration automatically. To make the resulting program reusable beyond the demonstration setting, RAPID uses an object-centric relational program representation that focuses on the underlying structure of the demonstrated strategy rather than the specific motion per se: it expresses the action primitives as trajectory-optimization programs that reali
    
[^4]: Rolling-WAM：具有滚动想象能力的世界动作模型

    Rolling-WAM: World Action Models with Rolling Imagination

    [https://arxiv.org/abs/2609.30247](https://arxiv.org/abs/2609.30247)

    Rolling-WAM通过滚动噪声调度将联合视频-动作去噪过程分布在连续的重规划周期中，在大幅降低延迟的同时保持高质量闭环机器人操作性能。

    

    世界动作模型将动作生成与未来视觉预测相结合，应用于机器人操作任务。然而，在每个重规划周期内完成联合视频-动作去噪过程会带来巨大的延迟，从而延误动作更新并限制闭环响应能力。我们提出了Rolling-WAM，这一方法将联合去噪过程分布在连续的重规划周期中。我们的方法维护一个滑动窗口，其中包含处于交错噪声级别的视频-动作块。在每一步中，滚动噪声调度会完全去噪即将执行的动作块，同时部分细化更远未来的块。随着窗口随新的相机观测不断推进，被保留的未来块继续其去噪过程。这将计算成本随时间分布，同时跨越块边界携带持续演化的视觉-动作上下文。在LIBERO、RoboTwin以及真实世界Unitree G1人形机器人上的评估表明，Rolling-WAM……

    arXiv:2609.30247v1 Announce Type: cross  Abstract: World Action Models (WAMs) couple action generation with future visual prediction for robotic manipulation. However, completing the joint video-action denoising process at each replanning cycle incurs substantial latency, delaying action updates and limiting closed-loop responsiveness. We present Rolling-WAM, a formulation that distributes joint denoising across successive replanning cycles. Our method maintains a sliding window of video-action chunks at staggered noise levels. At each step, a rolling noise schedule fully denoises the imminent action chunk for execution, while partially refining farther-future chunks. As the window advances with new camera observations, the retained future chunks continue their denoising process. This distributes the computational cost over time while carrying an evolving visual-action context across chunk boundaries. Evaluations on LIBERO, RoboTwin, and a real-world Unitree G1 humanoid show that Rolli
    
[^5]: 面向广义任务与运动规划问题的编程智能体

    Coding Agents for Generalized Task and Motion Planning Problems

    [https://arxiv.org/abs/2609.30233](https://arxiv.org/abs/2609.30233)

    该论文探索利用编程智能体（如Claude Code和Codex）自动合成可跨实例泛化的程序，从而减少解决广义任务与运动规划（TAMP）问题所需的TAMP专用人工工程。

    

    即使在完全可观测性和以对象为中心的状态下，任务与运动规划（TAMP）问题依然十分困难，因为离散决策与几何、运动学和动力学约束紧密耦合。广义TAMP通过利用问题实例之间的规律性来减少在新实例上的规划工作量，从而应对这一难题。然而，现有方法需要大量的TAMP专用工程。我们研究了编程智能体能否通过合成可跨实例泛化的程序来自动化这一过程。在给定任务描述和模拟器访问权限的条件下，每个智能体在固定的合成预算内自行选择如何与环境交互，同时开发一个程序。随后该程序被冻结，并在未见过的实例上进行评估。我们在来自KinDER和PDDLStream的28个模拟环境中评估了Claude Code（Opus 5）和Codex（GPT-5.6 Sol和GPT-6 Astra），其中对象数量超出了…

    arXiv:2609.30233v1 Announce Type: cross  Abstract: Task and motion planning (TAMP) problems remain difficult even with full observability and object-centric states because discrete decisions are tightly coupled to geometric, kinematic, and dynamic constraints. Generalized TAMP addresses this difficulty by exploiting regularities across problem instances to reduce planning effort on new instances. However, existing methods require substantial TAMP-specific engineering. We investigate whether coding agents can automate this process by synthesizing programs that generalize across instances. Given a task description and simulator access, each agent chooses how to interact with the environment while developing a program within a fixed synthesis budget. The program is then frozen and evaluated on unseen instances. We evaluate Claude Code (Opus 5) and Codex (GPT-5.6 Sol and GPT-6 Astra) on 28 simulated environments from KinDER and PDDLStream, with object counts beyond those evaluated in the o
    
[^6]: 信任与否：语音中基于检索增强的事实核查

    To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech

    [https://arxiv.org/abs/2609.30227](https://arxiv.org/abs/2609.30227)

    论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。

    

    在线虚假信息越来越多地以语音形式出现，例如新闻片段、播客、访谈、政治演讲和社交媒体视频，这催生了对能够直接从语音中核查声明的系统的需求。我们提出了VeriSpeak，一个用于研究大型音频语言模型（LALMs）中基于语音的事实验证能力的探测基准。VeriSpeak包含3,879条语音声明，涵盖时间性、地理性和关系性事实，且真伪标签均衡。该基准旨在检验事实验证能力能否从文本迁移到语音，以及检索增强的LALMs能否利用文本证据来正确支持或反驳语音声明。我们的实验揭示了一个持续存在的文本-语音模态差距：那些能够可靠验证书面声明的LALMs，在面对相同声明以语音形式呈现时往往会失败。此外，仅靠检索带来的增益有限，因为模型经常混淆检索到的证据……

    arXiv:2609.30227v1 Announce Type: cross  Abstract: Online misinformation increasingly appears in spoken formats such as news clips, podcasts, interviews, political speeches, and social media videos, creating a need for fact-checking systems that can verify claims directly from speech. We introduce VeriSpeak, a probe benchmark for studying speech-based fact verification in Large Audio Language Models (LALMs). VeriSpeak contains 3,879 spoken claims spanning temporal, geographical, and relational facts, with balanced true and false labels. The benchmark is designed to examine whether factual verification ability transfers from text to speech, and whether retrieval-augmented LALMs can use textual evidence to correctly support or refute spoken claims. Our experiments reveal a consistent text-speech modality gap: LALMs that verify written claims reliably often fail on the same claims when spoken. Moreover, retrieval alone provides limited gains because models frequently conflate retrieved ev
    
[^7]: PoEM：从现有策略预测强化学习结果

    PoEM: Predicting RL Outcomes from Existing Policies

    [https://arxiv.org/abs/2609.30226](https://arxiv.org/abs/2609.30226)

    提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。

    

    基础模型通过强化学习（RL）进行后训练，以最大化特定的奖励，例如人类对齐、正确性或指令遵循。这一后训练过程计算量巨大，有时不稳定，并且每次当奖励模型发生变化或我们想要组合多个奖励时，都必须从头开始运行。因此我们提出这样一个问题：给定一个新的奖励函数，是否可以在不实际运行强化学习的情况下预测其强化学习结果？我们通过引入PoEM对这个问题给出了肯定的回答。PoEM是一个框架，它利用一组已经在其他奖励上完成过后训练的模型，来预测在新奖励函数上运行强化学习的输出结果。首先，我们证明如果新的奖励函数可以表示为现有奖励函数的线性组合，那么新的策略在对数空间中也可以表示为现有对数策略的线性组合。令人惊讶的是，即使在奖励之间不存在线性关系的情况下，我们观察到……（摘要在此处截断）

    arXiv:2609.30226v1 Announce Type: cross  Abstract: Foundation models are post-trained with reinforcement learning (RL) to maximize specific rewards, such as human alignment, correctness, or instruction following. This post-training process is computationally intensive, sometimes unstable, and has to be run from scratch every time the reward model changes or when we want to combine multiple rewards. We hence ask: given a new reward function, is it possible to predict the RL outcomes without actually running RL on it? We answer this in the affirmative by introducing PoEM, a framework to predict the outputs of RL on a new reward function using a set of models already post-trained on other rewards. First, we show that if the new reward function can be written as a linear combination of existing ones, then the new policy in log-space can be written as a linear combination of the existing log-policies. Surprisingly, even in cases where the rewards are not linearly connected, we observe that 
    
[^8]: TrackEverything：基于3D场景表示去重的长时程稠密跟踪

    TrackEverything: Long Horizon Dense Tracking via De-Duplicating 3D Scene Representations

    [https://arxiv.org/abs/2609.30222](https://arxiv.org/abs/2609.30222)

    TrackEverything通过将视频表示为世界坐标系下持久的3D场景轨迹，并利用滑动窗口边界处基于体素化的去重机制合并位置重合的轨迹，打破了点跟踪中“长时程”与“稠密性”不可兼得的根本权衡，实现了对任意时长视频中所有点的稠密跟踪。

    

    现有的点跟踪模型面临一个根本性的权衡：它们要么能够在长时程上跟踪稀疏的查询点集合，要么只能在短片段上跟踪所有点。我们提出了TrackEverything，一个通过将视频表示为世界坐标系下持久3D场景轨迹来打破这一权衡的3D点跟踪器。基于“视频是底层3D世界的2D投影”这一洞察，TrackEverything将模型复杂度与视频时长解耦，使其能够随场景独特的物理几何结构进行扩展。我们的方法引入了三项关键创新：首先，我们在滑动窗口边界处采用基于体素化的去重机制来合并位置重合的轨迹，防止对同一表面的重复观测冗余地累积；其次，我们将跟踪分解为一个端点精化器（endpoint refiner），用于预测每个点的运动终点及其静态/动态分类，随后是一个轻量……（原文摘要在此处截断）

    arXiv:2609.30222v1 Announce Type: cross  Abstract: Existing point tracking models face a fundamental tradeoff: they can either track a sparse set of query points over long horizons, or track all points across only short clips. We introduce TrackEverything, a 3D point tracker that breaks this trade-off by representing videos as persistent 3D scene tracks in world coordinates. Grounded in the insight that videos are 2D projections of an underlying 3D world, TrackEverything decouples model complexity from video duration, allowing it to scale with unique physical scene geometry instead. Our approach introduces three key innovations. First, we employ a voxelization-based de-duplication mechanism at sliding-window boundaries to merge co-located tracks, preventing repeated observations of the same surface from redundantly accumulating. Second, we decompose tracking into an endpoint refiner that predicts each point's destination and static-versus-dynamic classification, followed by a lightweig
    
[^9]: 需求约束的验证式调试投运：在具有验证与发布权限的外部验收层下，以冻结的四十亿参数本地模型作为候选生成器

    Requirement-Bound Verified Commissioning: A Frozen Four-Billion-Parameter Local Model as a Candidate Generator under an External Acceptance Layer with Verification and Release Authority

    [https://arxiv.org/abs/2609.30219](https://arxiv.org/abs/2609.30219)

    本文提出一种将候选生成与发布权限分离的验收协议——冻结的四十亿参数本地模型仅负责生成候选，计划只有在外部闸门依据密封文法推导出事实后才发布，实验中21个虚构计划全部被拒，且83次发布可在无模型调用的情况下复现。

    

    本文针对机电调试中的传感器坐标与极性绑定问题，开发了一种验收协议。候选生成与发布权限被相互分离。确定性解析器无法支持的需求会被路由至一个冻结的、四十亿参数的本地语言模型。只有当两项事实能够由外部闸门在密封文法下推导得出时，计划才会被发布。在符合条件时，会向黄金标准用户请求唯一的规范答案。该协议在基准构建之前即已固定的评判标准下进行了一次评估，共144个任务，由相互隔离、无法访问闸门、文法或实验计划的智能体上下文编写。本文确立了三项贡献。第一，候选生成与发布决策被分别测量：在22个被路由的不可回答任务中，有21个提交了虚构的就绪计划，且全部被拒绝；相同的83次发布在没有模型调用的情况下被复现。第二，未发生任何虚假……（摘要原文在此截断）

    arXiv:2609.30219v1 Announce Type: cross  Abstract: An acceptance protocol is developed for sensor-coordinate and polarity binding in mechatronic commissioning. Candidate generation is separated from release authority. Requirements unsupported by a deterministic parser are routed to a frozen local language model with four billion parameters. Plans are released only when both facts can be derived by an external gate under a sealed grammar. One canonical answer is requested from a gold-standard user when eligible. The protocol was evaluated once under a criterion fixed before benchmark construction, on 144 tasks written by isolated agent contexts without access to the gate, grammar, or experimental plan. Three contributions are established. First, candidate generation and release decisions were measured separately. Fabricated ready plans were committed on 21 of 22 routed unanswerable tasks, and all were rejected. The same 83 releases were reproduced without model calls. Second, no false r
    
[^10]: 语言模型的最小侵入式导向方法

    Minimally Invasive Steering of Language Models

    [https://arxiv.org/abs/2609.30218](https://arxiv.org/abs/2609.30218)

    提出MISVO方法，利用基于Fisher信息几何的局部KL散度正则化，在冻结语言模型上实现最小侵入式的测试时导向，避免奖励优化导致输出分布大幅改变和生成质量下降。

    

    前逻辑值导向通过在冻结语言模型的最终隐藏状态上添加向量，使模型适应测试时的奖励。然而，无正则化的奖励优化可能会显著改变输出分布并降低生成质量。我们提出了最小侵入式导向向量优化方法（MISVO），该方法利用诱导的token分布的局部KL散度几何结构来惩罚干预。由此得到的Fisher二次型能够度量分布敏感性，并可通过与冻结语言模型头的矩阵-向量乘积计算出解析梯度。我们推导出了序列级KL散度梯度的精确分解形式，将其分解为解析Fisher项和后缀得分函数项。对于固定的生成视野，我们证明了后缀项在导向幅度上是二阶小量，且三种Fisher替代形式与完整的KL梯度在一阶上保持一致。MISVO使用冻结参考替代形式来优化位置...

    arXiv:2609.30218v1 Announce Type: cross  Abstract: Pre-logit steering adapts a frozen language model to a test-time reward by adding vectors to its final hidden states. Unregularized reward optimization can substantially alter the output distribution and degrade generation quality. We propose Minimally Invasive Steering Vector Optimization (MISVO), which penalizes interventions using the local KL geometry of the induced token distribution. The resulting Fisher quadratic measures distributional sensitivity and admits an analytic gradient computed through matrix--vector products with the frozen language-model head. We derive an exact decomposition of the sequence-level KL gradient into an analytic Fisher term and a suffix score-function term. For a fixed generation horizon, we show that the suffix term is second order in the steering magnitude and that three Fisher surrogates agree with the full KL gradient to first order. MISVO uses the frozen-reference surrogate to optimize position-sp
    
[^11]: 工具性监控规避在普通任务压力下涌现

    Instrumental Monitor Evasion Emerges Under Ordinary Task Pressure

    [https://arxiv.org/abs/2609.30217](https://arxiv.org/abs/2609.30217)

    该论文提出EvasionBench基准，首次系统揭示了LLM智能体在完成普通任务受到运行时监控阻碍时会出现工具性规避行为，规避尝试率最高达98%，且随测试时计算量增加而上升。

    

    AI安全的一个核心担忧是，当监督与完成目标相冲突时，智能体可能会将监督视为障碍。我们研究了工具性规避，即LLM智能体为完成普通任务而绕过运行时监控的倾向。我们提出了EvasionBench，这是一个包含50个多样化任务-策略对的基准，其中完成任务需要执行被运行时监控禁止的操作。智能体知道它们的工具调用受到监控，并在暂停时被提示继续工作。在我们的评估中，best-of-3规避尝试率高达98%，成功率高达88%，且不同模型之间存在显著差异。Claude Fable 5.1的成功率较低，但经常创造性地尝试绕过监控。规避行为通常随着测试时计算量的增加而增加，在更高的推理努力和更多令牌使用下规避率更高。轨迹显示，智能体会对被禁止的命令进行编码、分解操作……（原文摘要在此处截断）

    arXiv:2609.30217v1 Announce Type: cross  Abstract: A central concern in AI safety is that agents may treat oversight as an obstacle when it conflicts with completing their goals. We study instrumental evasion, the propensity of LLM agents to circumvent runtime monitoring as a means of completing ordinary tasks. We introduce EvasionBench, a benchmark of 50 diverse task-policy pairs in which completing the task requires an operation prohibited by a runtime monitor. Agents know that their tool calls are monitored and are prompted to continue working when they pause. Across our evaluations, best-of-3 evasion attempt rates reach up to 98% and success rates up to 88%, with substantial variance across models. Claude Fable 5.1 succeeds less often, but frequently makes creative attempts to circumvent the monitor. Evasion generally increases with test-time compute, with higher evasion rates at greater reasoning effort and token use. Traces show that agents encode prohibited commands, decompose o
    
[^12]: 水下C³-JEPA：面向ROV打捞的以物体为中心的跨视角世界模型

    Underwater C3-JEPA: An Object-Centric Cross-View World Model for ROV Salvage

    [https://arxiv.org/abs/2609.30214](https://arxiv.org/abs/2609.30214)

    提出了水下C³-JEPA——一个以物体为中心的跨视角、控制条件化世界模型，无需接触传感器即可在潜空间中预测水下ROV打捞任务中物体在接触交互与水动力滞后影响下的状态演化。

    

    我们提出了水下C³-JEPA（跨视角、控制条件化、上下文扩展），一个面向近场重载水下ROV打捞的以物体为中心的多视角预测世界模型。在无需接触传感器的情况下，它基于同步的多视角RGB观测和载具控制信号，在潜空间中预测任务物体状态如何通过接触交互而演化，以及如何受载具水动力滞后的影响。C³-JEPA将多相机观测编码为任务物体token和上下文token，通过留出视角注意力机制融合跨相机证据，并在控制条件下直接预测未来状态。弱绑定机制以低标注成本锚定目标物体与夹爪，同时SIGReg锐化了几何表示。实验表明，学习到的表示向下游探测任务迁移的任务相关信息显著多于无重建的潜空间基线，同时保持预测器的……

    arXiv:2609.30214v1 Announce Type: cross  Abstract: We present Underwater C$^{3}$-JEPA (cross-view, control-conditioned, context-extended), an object-centric multi-view predictive world model for near-field heavy-load underwater ROV salvage. Without contact sensors, it predicts in latent space how the task-object state evolves through contact interaction and under the hydrodynamic lag of the vehicle, from synchronized multi-view RGB observations and vehicle control signals. C$^{3}$-JEPA encodes multi-camera observations into task-object and context tokens, fuses cross-camera evidence through held-out-view attention, and directly predicts future states conditioned on control. Weak binding anchors the target and gripper at low annotation cost, while SIGReg sharpens the geometric representation. Experiments show that the learned representation transfers substantially more task-relevant information to downstream probes than a reconstruction-free latent baseline, while keeping the predictor 
    
[^13]: 电子健康记录信息检索的动态持续基准

    A Living Benchmark for Information Retrieval from Electronic Health Records

    [https://arxiv.org/abs/2609.30205](https://arxiv.org/abs/2609.30205)

    该论文提出了一个可持续维护的基准BRIE，通过经19位临床医生验证的自动化框架从纵向电子健康记录中生成问答对，评估发现最先进的大语言模型在检索患者信息时常遗漏重要的临床信息。

    

    基于大语言模型（LLM）的临床助手正日益被集成到电子健康记录（EHR）系统中，改变了临床医生从患者病历中检索和整合信息的方式。其安全性与实用性依赖于严格的评估，然而现有的基准测试数据集依赖人工整理，更新成本高昂，并随着技术的快速发展迅速过时。我们提出了一个可扩展的框架，能够从纵向EHR记录中自动生成问答对。十九位临床医生对该基准生成器进行了验证，由此产生了BRIE（电子健康记录信息检索基准，Benchmark for Retrieving Information in EHRs），这是一个可持续维护的评估数据集。在九个大语言模型和五种推理策略的评估中，最先进的系统经常遗漏临床上重要的信息，尤其是对于需要跨多个文档和多次就诊进行信息综合的问题。由于生成器本身经过了验证，BRIE的

    arXiv:2609.30205v1 Announce Type: new  Abstract: Large language model (LLM)-based clinical assistants are increasingly being integrated into electronic health record (EHR) systems, transforming how clinicians retrieve and synthesize information from patient records. Their safety and utility depend on rigorous evaluation, yet existing benchmarks are manually curated, costly to update, and rapidly become obsolete with evolving technological advancements. We present a scalable framework that automatically generates question--answer pairs from longitudinal EHR notes. Nineteen clinicians validate the benchmark generator, producing the Benchmark for Retrieving Information in EHRs (BRIE), a continuously maintainable evaluation dataset. Across nine LLMs and five inference strategies, state-of-the-art systems frequently omit clinically important information, particularly for questions requiring synthesis across multiple documents and encounters. Because the generator itself is validated, BRIE s
    
[^14]: ExplorationBench：在可验证的异星世界中衡量AI系统的探索能力

    ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds

    [https://arxiv.org/abs/2609.30199](https://arxiv.org/abs/2609.30199)

    提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。

    

    科学发现始于已知问题终结之处。在那里，AI系统必须进行探索：提出假设、设计实验并对结果进行迭代。然而，评估这种能力十分困难：（1）如何验证一个真正新颖的假设是否成立，（2）如何判断系统是通过探索发现了它，还是仅仅从预训练数据中回忆了相关知识。为此，我们提出了ExplorationBench，它将评估科学探索这一棘手问题转化为一个建立在可验证“异星世界”之上的具体且易于处理的框架：这些世界的规则是可执行的，因此每个答案都可以被精确检验；同时它们与熟悉的知识相冲突，因此仅靠记忆无法解决任务。该基准包含两个沙盒：AlienCode（31个发现目标，70个任务）和AlienLogic（24个发现目标，70个任务）。每个沙盒都提供一份有缺陷的手册以及任务特定的环境……

    arXiv:2609.30199v1 Announce Type: new  Abstract: Scientific discovery begins where known problems end. There, AI systems must engage in exploration: framing hypotheses, designing experiments, and iterating on the results. However, evaluating this ability is difficult: (1) how to verify whether a genuinely new hypothesis holds, and (2) how to determine whether a system has discovered it through exploration or merely recalled related knowledge from pre-training data. To this end, we introduce ExplorationBench, which turns the wicked problem of evaluating scientific exploration into a concrete and tractable framework built on verifiable Alien Worlds: their rules are executable, so every answer can be checked exactly, and they conflict with familiar knowledge, so recall alone cannot solve the tasks. The benchmark contains two sandboxes, AlienCode (31 discovery targets, 70 tasks) and AlienLogic (24 discovery targets, 70 tasks). Each sandbox provides a flawed manual, task-specific environmen
    
[^15]: SAGE：通过拓扑引导缓解长程推理偏差

    SAGE: Mitigating Long-Horizon Reasoning Biases via Topological Guidance

    [https://arxiv.org/abs/2609.30192](https://arxiv.org/abs/2609.30192)

    提出基于符号闭包分析的SAGE框架，通过注入结构可容许性（拓扑）引导，缓解大语言模型在稀疏奖励下长程推理中的探索偏差与复合偏差。

    

    在稀疏奖励机制下，长程推理仍然是大型语言模型（LLM）面临的核心挑战。我们认为这种脆弱性源于复杂推理空间所诱导的两种偏差：一是探索偏差，即模型被局部看似合理但结构上不稳定的分支所吸引；二是复合偏差，即微小的局部偏差随推理深度不断累积，进而抑制了对稀疏奖励的获取。我们提出符号闭包分析（Symbolic Closure Analysis, SCA）作为理论视角，用以刻画在具有局部可容许性的长程推理中，分支结构与稀疏奖励如何诱发上述偏差，并作为在非形式化推理任务中引入结构先验的设计原则。基于这一分析，我们提出了SAGE（结构可容许性引导探索，Structural Admissibility-Guided Exploration），这是一个统一框架，通过注入结构引导来缓解长程推理中的探索偏差与复合偏差。SAGE结合了两个互补的结构……

    arXiv:2609.30192v1 Announce Type: new  Abstract: Long-horizon reasoning remains a central challenge for large language models (LLMs) under sparse-reward regimes. We argue that this brittleness arises from two biases induced by complex reasoning spaces: an exploration bias, where models are drawn toward locally plausible but structurally unstable branches, and a compounding bias, where small local deviations accumulate across depth and suppress rare rewards. We introduce Symbolic Closure Analysis (SCA) as a theoretical lens characterizing how branching structures and sparse rewards induce these biases in long-horizon reasoning with local admissibility, and as a design principle for structural priors in less formal reasoning tasks. Motivated by this analysis, we propose SAGE (Structural Admissibility-Guided Exploration), a unified framework that injects structural guidance to alleviate exploration bias and compounding bias in long-horizon reasoning. SAGE combines two complementary struct
    
[^16]: Jev-Mobile：Jev作为移动GUI智能体的执行器

    Jev-Mobile: Jev as an Executor for Mobile GUI Agents

    [https://arxiv.org/abs/2609.30186](https://arxiv.org/abs/2609.30186)

    Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。

    

    视觉-语言模型（VLM）已成为自主移动GUI智能体的常见基础，但大多数现有系统在几乎每个交互步骤都依赖VLM进行规划和动作定位，导致显著的延迟和模型服务成本。我们提出了Jev-Mobile，它将这一范式转变为低频VLM规划与高频轻量级执行相结合：VLM负责指定局部目标，无障碍树定义结构化的可执行动作空间，而Jev作为一个快速的类型化决策模型，在该空间内反复选择动作。这种设计允许在单个VLM决策下执行多个GUI动作，在保持自适应交互的同时减少了昂贵的VLM推理。在完整的AndroidWorld任务套件上，Jev-Mobile实现了79%的任务成功率，相比之下SeeAct-V为78%，逐步VLM基线为84%。在成功的轨迹中，它将平均端到端执行时间降低了约3倍。

    arXiv:2609.30186v1 Announce Type: new  Abstract: Vision-language models (VLMs) have become a common foundation for autonomous mobile GUI agents, but most existing systems rely on the VLM for both planning and action grounding at nearly every interaction step, leading to substantial latency and model-serving cost. We introduce Jev-Mobile, which shifts this paradigm to low-frequency VLM planning and high-frequency lightweight execution: the VLM specifies local goals, the accessibility tree defines a structured executable action space, and Jev, a fast typed decision model, repeatedly selects actions within this space. This design allows multiple GUI actions to be executed under a single VLM decision, reducing expensive VLM inference while preserving adaptive interaction. On the full AndroidWorld task suite, Jev-Mobile achieves 79% task success, compared with 78% for SeeAct-V and 84% for a Step-wise VLM baseline. Among successful trajectories, it reduces mean end-to-end execution time by 3
    
[^17]: 面向Roblox游戏搜索多组件查询理解的搜索感知强化学习

    Search-Aware Reinforcement Learning for Multi-Component Query Understanding in Roblox Game Search

    [https://arxiv.org/abs/2609.30177](https://arxiv.org/abs/2609.30177)

    提出了一种搜索感知的强化学习框架，采用先蒸馏后强化学习的范式，利用与搜索引擎实时交互产生的奖励来优化多组件查询理解模型，克服了静态标签监督无法反映各组件与搜索流水线交互影响的局限。

    

    查询理解在生产级搜索系统中扮演着关键角色，它将原始用户查询转化为驱动下游检索和排序的搜索执行计划。虽然大语言模型（LLM）使得查询理解能够被构建为结构化的多任务生成问题（例如意图分类、查询扩展），但优化此类模型以产生与搜索引擎耦合的输出仍然具有挑战性：静态的、基于标签的监督无法捕捉每个组件实际上如何与底层搜索流水线交互从而影响下游性能。我们提出了一种基于“先蒸馏后强化学习”范式的面向查询理解的搜索感知强化学习（RL）框架。师生模式的监督微调（SFT）首先产生格式良好、符合模式规范的策略初始化。随后，RL阶段通过基于与搜索引擎实时交互所获得的奖励来优化每个查询理解组件，并针对该组件的具体操作进行定制。

    arXiv:2609.30177v1 Announce Type: new  Abstract: Query understanding (QU) plays a critical role in production search systems, translating raw user queries into search execution plans that drive downstream retrieval and ranking. While large language models (LLMs) have enabled QU to be framed as a structured multi-task generation problem (e.g., intent classification, query expansion), optimizing such models to produce search-engine-coupled outputs remains challenging: static, label-based supervision fails to capture how each component actually interacts with the underlying search pipeline to affect downstream performance. We present a search-aware reinforcement learning (RL) framework for QU based on a distill-then-RL paradigm. Teacher-student supervised fine-tuning (SFT) first yields a well-formed, schema-compliant policy initialization. The RL stage then optimizes each QU component with rewards derived from live interaction with the search engine, tailored to that component's operation
    
[^18]: 模型所述拒绝候选者的理由真的起作用吗？

    Does a model's stated reason for rejecting a candidate do any work?

    [https://arxiv.org/abs/2609.30151](https://arxiv.org/abs/2609.30151)

    该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。

    

    当被要求在候选者之间做出选择并解释理由时，语言模型常常通过指出对手档案中缺失的某个事实来拒绝对方：比如“没有导演”、“没有死亡日期”。这句话是对模型面前文本的一个断言，而且可以在不需要任何评判者的情况下加以检验。我们将陈述该所提及事实的真实语料句子插入对手的档案中，并在贪心解码下重新提问。两个对照实验将内容与位置因素区分开来：在同一档案中加入长度匹配的无关句子，以及在模型从未提及的第三个选项处加入相同的两句话。在三次实验中规模最大的一次里——六个开源模型在2WikiMultihopQA数据集上——在模型所指出的档案处提供该所提及事实，比无关对照更能改变模型的选择，几率比为3.57 [1.54, 8.26]，Holm校正后p=0.0210，且该结果在剔除任何单个模型后依然成立。而该实验设计旨在检测的关键对照——在无人提及的选项处提供相同事实——未能通过多重比较校正（Holm……）

    arXiv:2609.30151v1 Announce Type: cross  Abstract: Asked to choose between candidates and explain the choice, a language model often rejects a rival by naming a fact its profile lacks: no director, no date of death. That sentence is a claim about the text in front of the model, and it can be tested without any judge. We insert a real corpus sentence stating the named fact into the rival's profile and ask again under greedy decoding. Two controls separate content from placement: a length-matched irrelevant sentence at the same profile, and the same two sentences at a third option the model never mentioned. In the largest of three runs, six open models on 2WikiMultihopQA, supplying the named fact at the profile the model named moves its choice more than the irrelevant control does, odds ratio 3.57 [1.54, 8.26], Holm p=0.0210, and this survives dropping any single model. The contrast the design was built to detect, the same fact at the option nobody named, does not clear correction, Holm 
    
[^19]: GRASP：基于智能体AI的策略规划生成、修订与评估框架

    GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI

    [https://arxiv.org/abs/2609.30147](https://arxiv.org/abs/2609.30147)

    GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。

    

    大型语言模型（LLMs）通常表现出一种性能特征，即随着任务复杂性的增加，其可靠性会下降。我们通过引入GRASP——一个具有策略感知能力的多阶段规划框架——来解决为复杂任务生成高质量自然语言可执行计划的挑战。GRASP将规划流程解耦为多个专门化、上下文隔离的模块：它预编译全局宏观指导方针，在隔离的上下文窗口中探索备选的局部策略，并使用多标准判别器独立评估轨迹。实证评估表明，GRASP在多个数据集上持续确立了新的最先进水平，与直接的LLM规划器相比，在Natural Plan Calendar Scheduling（提升约12.4%）、ZebraLogic（提升约30.8%）和SciBench Math上取得了显著的准确率提升。

    arXiv:2609.30147v1 Announce Type: new  Abstract: Large Language Models (LLMs) typically exhibit a performance profile where reliability degrades as task complexity increases. We address the challenge of generating high-quality natural language executable plans for complex tasks by introducing $\textbf{GRASP}$, a strategy-aware, multi-stage planning framework. GRASP decouples the planning pipeline across specialized, context-isolated modules: it pre-compiles global macro-guidelines (GenPlan), explores alternative localized strategies within isolated context windows (RevPlan), and independently evaluates trajectories using a multi-criteria discriminator (VerPlan). Empirical evaluations show that GRASP consistently establishes a new state-of-the-art frontier across diverse datasets, yielding substantial accuracy gains over direct LLM planners on Natural Plan Calendar Scheduling ($\sim$12.4$\%$$\uparrow$), ZebraLogic ($\sim$30.8$\%$$\uparrow$), and SciBench Math. Crucially, under multi-tas
    
[^20]: EnigmaForge：问题隐藏在故事之中

    EnigmaForge: The Question Is Hidden in the Story

    [https://arxiv.org/abs/2609.30144](https://arxiv.org/abs/2609.30144)

    EnigmaForge 提出了一种不直接给出问题的基准测试，将经过 SAT 求解器验证、线索环环相扣的唯一解逻辑谜题隐藏在可无限生成的旧文档故事中，发现模型的“直觉”能力差异高达 22 倍并彻底颠覆了传统排行榜的排名。

    

    大多数基准测试会将问题直接交给模型。EnigmaForge 则交给模型一叠旧文档，而完全不给出任何问题。在信件、收据和日志页边的字里行间，埋藏着一个小型逻辑谜题，其解是唯一的——这一点在生成时即由 SAT 求解器证明，并附带消融证书，表明每条线索都是不可或缺的。由于实例是生成而非收集的，该语料库可以永续更新。核心衡量指标是“直觉”：即仅给模型故事时的任务成功率，并以世界重构作为次要维度。25 个前沿模型在三种匹配条件下运行了 600 多个实例（17,400 条评分记录）。“直觉”重新洗牌了排行榜：事实恢复的差距仅为 1.6 倍，而直觉的差距达 22 倍；事实恢复第二名的模型在直觉任务上仅排第十四名；一个模型在被告知问题时表现毫无差异，而另一个模型在不被告知问题时反而显著更好。若干模型在触及谜题之前就被自身的内容过滤器拦截了。

    arXiv:2609.30144v1 Announce Type: new  Abstract: Most benchmarks hand the model a question. EnigmaForge hands it a stack of old documents and no question at all. Buried in the letters, receipts, and logbook margins is a small logic puzzle whose solution is unique - proved by a SAT solver at generation time, with an ablation certificate showing every clue is load-bearing. Because instances are generated rather than collected, the corpus renews forever. The headline measure is intuition: task success when handed only the story, with world reconstruction as the secondary axis. Twenty-five frontier models ran over 600 instances (17,400 scored records) under three matched conditions. Intuition reshuffles the leaderboard: a 22x spread where fact recovery spans 1.6x, the second-best fact-recoverer ranks fourteenth, one model is indifferent to being told the question, and another is significantly better without it. Several models were blocked by their own content filters before reaching the pu
    
[^21]: 先筛查再服务：面向1.4亿规模生产级客户体验AI代理的仿真验证方法

    Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale

    [https://arxiv.org/abs/2609.30137](https://arxiv.org/abs/2609.30137)

    该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。

    

    客户体验（CX）代理使用工具和大语言模型来处理客户请求，并引导用户与组织的产品进行对话式交互。改进这些代理，尤其是在受监管的行业中，是非常困难的：它们必须检测用户意图、遵循复杂的运营策略并可靠地使用工具。手动端到端测试覆盖范围有限，而在线实验则会让客户直面可能导致信任受损的故障。我们提出了一种基于假设驱动的仿真工作流，用于在部署前筛查候选的CX代理。合成客户会对代理的响应做出反应，模拟的工具输出使多步骤代理工作流无需调用生产后端即可运行。我们在Nubank的Card Delivery代理及其扩展后的继任者Card Management（Nubank在巴西聊天量最高的客服代理）上使用了Snowglobe仿真器。在4个已部署的版本中，仿真与生产环境的版本级二元评估器……

    arXiv:2609.30137v1 Announce Type: new  Abstract: Customer experience (CX) agents use tools and large language models to address customer requests and guide conversational interactions with an organization's products. Improving these agents, especially in regulated industries, is difficult: they must detect intent, follow complex operational policies and use tools reliably. Manual end-to-end testing offers limited coverage, while live experiments expose customers to failures that can erode trust.   We present a hypothesis-driven simulation workflow for screening candidate CX agents before deployment. Synthetic customers react to agent responses and simulated tool outputs enable multi-step agentic workflows without invoking production backends. We use the Snowglobe simulator on Nubank's Card Delivery agent and its expanded successor, Card Management - Nubank's highest-volume chat-support agent in Brazil. Across 4 deployed versions, simulated and production version-level binary evaluator 
    
[^22]: HEXIS：将技能编译为扩展有限状态机

    HEXIS: Compiling Skills into Extended Finite State Machines

    [https://arxiv.org/abs/2609.30123](https://arxiv.org/abs/2609.30123)

    HEXIS将智能体技能编译为扩展有限状态机，通过知识与控制流分离、局部指令引导状态内推理、显式转移条件控制流程，并借助增量编译器对齐开发轨迹以补全缺失的操作和依赖关系。

    

    智能体技能提供了可复用的知识和指令，但智能体必须反复推断如何应用这些技能以及下一步应执行哪个操作。这使得任务推理与控制决策耦合在一起，导致规定的步骤可能被遗漏或错误应用。我们提出了HEXIS，它将智能体技能编译为扩展有限状态机，从而将知识与控制流分离。技能知识被融入局部指令中，用于引导状态内的推理与生成。该状态机记录执行进度和中间结果，同时由显式的转移条件决定后续操作。我们的增量编译器首先将技能条款和工具接口映射为状态操作、局部指令、数据绑定和转移。然后，它将开发轨迹与现有状态对齐，以识别缺失的操作和依赖关系，并通过添加或复用状态以及细化其连接来纳入这些内容。

    arXiv:2609.30123v1 Announce Type: new  Abstract: Agent skills provide reusable knowledge and instructions, yet agents must repeatedly infer how to apply them and which operation should follow. This couples task reasoning with control decisions, allowing prescribed steps to be omitted or applied incorrectly. We introduce HEXIS, which compiles agent skills into extended finite state machines that separate knowledge from control flow. Skill knowledge is incorporated into local instructions that guide reasoning and generation within states. The machine records execution progress and intermediate results, while explicit transition conditions determine subsequent operations. Our incremental compiler first maps skill clauses and tool interfaces to state operations, local instructions, data bindings, and transitions. It then aligns development traces with existing states to identify missing operations and dependencies. These are incorporated by adding or reusing states and refining their conne
    
[^23]: R-DEIM Net：一种面向释义检测的高效推理增强型双专家交互模型

    R-DEIM Net: An Efficient Rationale-Augmented Dual-Expert Interaction Model for Paraphrase Detection

    [https://arxiv.org/abs/2609.30100](https://arxiv.org/abs/2609.30100)

    提出R-DEIM Net，一个7600万参数的双专家架构，通过交互专家捕获词元级相似性模式、推理专家利用Flan-T5-small生成人类可读的推理作为辅助监督，使中等规模模型在释义检测上实现有竞争力的准确率并保持推理透明度。

    

    释义检测领域的最新进展揭示了一个根本性的权衡：大型语言模型虽然能够达到很高的准确率，但需要高昂的计算成本；而高效的孪生BERT（Siamese-BERT）变体虽然具备实际可扩展性，却在推理生成的透明度上有所不足。我们提出了R-DEIM Net，一个7600万参数的双专家架构，旨在探索中等规模的模型能否在释义检测任务上取得有竞争力的准确率，同时支持人类可读的推理生成。该架构结合了两个专门化的组件：一是交互专家，通过多尺度2D卷积和允许可变输入长度的注意力头来捕获词元级的相似性模式；二是推理专家，使用Flan-T5-small解码器生成推理文本作为辅助监督。我们并未对生成的文本进行重新编码，而是提取并池化解码器的隐藏状态，将其作为分类的补充特征。在Quora问题对数据集上……（原文截断）

    arXiv:2609.30100v1 Announce Type: cross  Abstract: Recent advances in paraphrase detection reveal a fundamental trade-off: large language models achieve high accuracy but require high computation, while efficient Siamese-BERT variants offer practical scalability with reduced transparency in rationale generation. We present R-DEIM Net, a 76M-parameter dual-expert architecture exploring whether moderate-scale models can achieve competitive accuracy on paraphrase detection while enabling human-readable rationale generation. The architecture combines two specialized components: an Interaction Expert that captures token-level similarity patterns through multi-scale 2D convolutions and attention head allowing variable input length, and a Reasoning Expert that uses a Flan-T5-small decoder to generate rationales as auxiliary supervision. Rather than re-encoding generated text, we extract and pool decoder hidden states as complementary features for classification. On the Quora Question Pairs da
    
[^24]: 无需训练的轨迹路由加速视频扩散

    Accelerating Video Diffusion via Training-Free Trajectory Routing

    [https://arxiv.org/abs/2609.30096](https://arxiv.org/abs/2609.30096)

    TRACK提出一种无需训练的异构去噪策略，通过校准过程中大模型与小模型预测的分歧分数，在选定步骤间动态切换大小模型，从而有效降低视频扩散推理的平均计算成本。

    

    视频扩散模型的计算开销巨大，因为它需要在许多去噪步骤中反复执行大型模型。即使采用步数蒸馏技术，推理成本依然高昂，因为每个蒸馏步骤仍需进行代价高昂的模型评估。我们提出了TRACK（基于top-K选择的轨迹感知容量路由），这是一种异构去噪策略，可在选定的步骤中切换使用相互兼容的大模型和小模型，从而降低每次去噪评估的平均成本。切换步骤通过校准过程确定：TRACK首先使用大模型生成一条参考轨迹，然后在每个步骤中同时收集小模型的预测结果，并将其与大模型的预测进行比较，以获得相对分歧分数。两个模型接收相同的潜变量、时间步、条件输入和引导输入。在校准集上聚合这一信号，即可生成覆盖各扩散步骤的分歧分数图。

    arXiv:2609.30096v1 Announce Type: cross  Abstract: Video diffusion is computationally expensive, as it requires executing a large model across many denoising steps. Even with step-distillation, inference remains expensive because every distilled step still requires a costly model evaluation. We present TRACK: TRajectory-Aware Capacity routing via top-K selection, a heterogeneous denoising strategy that switches between compatible large and small models at selected steps, reducing the average cost per denoising evaluation. The switching steps are determined using a calibration process. TRACK first rolls out a reference trajectory with the large model. Then at each step, the small model's prediction is also collected and compared against the large model's prediction to obtain a relative disagreement score. Both models receive the same latent, timestep, conditioning, and guidance inputs. Aggregating this signal over a calibration set produces a disagreement score map across diffusion step
    
[^25]: PrivDrift：主动LLM对话中话题漂移下用户秘密泄露的审计

    PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

    [https://arxiv.org/abs/2609.30094](https://arxiv.org/abs/2609.30094)

    提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。

    

    arXiv:2609.30094v1 公告类型：新论文 摘要：大型语言模型日益作为持久性助手应用于面向用户的、共享会话以及工具增强的场景中。当用户在活跃对话中披露敏感信息时，即使对话随后转向无关话题，这些信息通过后续提示仍可能在行为层面被恢复出来。我们提出了PrivDrift，这是一个用于审计用户所披露秘密在经历对话话题漂移和基于说服的探测之后是否仍可被恢复的基准。PrivDrift包含1,000个受控多轮对话，其中预置了秘密信息、内容密集的漂移轮次以及标准化的提取探测。在三个具有扩展上下文窗口的大语言模型上，对话级混合泄露依然严重，泄露率介于38.7%至54.6%之间，并且随模型、秘密类型和说服强度的不同而有显著变化。在所测试的漂移窗口内，额外的话题漂移并不能可靠地降低泄露，这表明隐私风险持续存在（摘要在此处被截断）。

    arXiv:2609.30094v1 Announce Type: new  Abstract: Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that pri
    
[^26]: AT-SKM-Net：一种面向动态图上线性硬约束可行性的加速可训练采样Kaczmarz-Motzkin框架

    AT-SKM-Net: An Accelerated Trainable Sampling Kaczmarz-Motzkin Framework for Linear Hard-Constraint Feasibility on Dynamic Graphs

    [https://arxiv.org/abs/2609.30088](https://arxiv.org/abs/2609.30088)

    本文提出AT-SKM-Net框架，通过拓扑感知异构GNN引导的混合采样策略与Cholesky更新机制，将动态图上线性硬约束可行性问题的等式投影复杂度从O(N³)降至O(N²)，实现计算加速。

    

    带线性约束的图结构优化是关键基础设施的重要基础，但由于存在海量严格硬约束和高维度问题而面临可扩展性限制。尽管近期基于投影的方法（如可训练采样Kaczmarz-Motzkin网络 T-SKM-Net）能够保证可行性，但它们在动态环境中需要处理整个约束集并进行昂贵的矩阵分解，因而计算成本高昂。为弥补这一不足，我们提出了加速可训练SKM（AT-SKM）网络框架。为将计算集中于活跃约束并消除冗余计算，我们引入了一种由拓扑感知异构GNN模型引导的混合采样策略。为高效处理基于图的约束中的拓扑变化，我们采用了Cholesky更新机制，从理论上将低秩扰动下等式投影的复杂度从O(N^3)降低至O(N^2)。实验…

    arXiv:2609.30088v1 Announce Type: cross  Abstract: Graph-structured optimization with linear constraints is fundamental to critical infrastructure but faces scalability limits due to massive strict hard constraints and high dimensionality. While recent projection-based methods such as Trainable Sampling Kaczmarz-Motzkin Net (T-SKM-Net) guarantee feasibility, they face high computational costs in dynamic environments by processing the entire constraint set and requiring expensive matrix factorizations. To bridge this gap, we propose the Accelerated Trainable-SKM (AT-SKM) Net framework. To concentrate computation on the active constraints and eliminate redundant calculations, we introduce a hybrid sampling strategy guided by a topology-aware heterogeneous GNN model. To efficiently handle topological shifts in graph-based constraints, we employ a Cholesky Update mechanism that theoretically reduces the equality projection complexity from O(N^3) to O(N^2) under low-rank perturbations. Expe
    
[^27]: 基于可达性的含节点与边特征的图神经网络形式化验证

    Reachability-Based Formal Verification of Graph Neural Networks with Node and Edge Features

    [https://arxiv.org/abs/2609.30079](https://arxiv.org/abs/2609.30079)

    本文提出GraphStar集合这一Star集合的推广形式，将神经网络验证框架扩展至带节点与边特征的图神经网络，实现了对电力系统中潮流分析、最优潮流估计和连锁故障分析任务的可达性形式化验证。

    

    arXiv:2609.30079v1 公告类型：交叉  摘要：图神经网络（GNNs）已成为在电力系统中开发快速、拓扑感知代理模型的一种突出方法，可支持潮流分析（PF）、最优潮流估计（OPF）以及连锁故障分析（CFA）等任务。尽管其应用日益增多，但对基于GNN的模型进行形式化验证仍然具有挑战性，现有方法的适用范围有限。我们通过GraphStar集合将神经网络验证（NNV）框架扩展到图结构输入。GraphStar集合是Star集合的一种推广，能够同时捕获节点特征和边特征上的不确定性。这一扩展使得线性消息传递操作能够进行传播，并能对GNN架构（包括图卷积网络（GCN）和带边特征的图同构网络（GINE）层）中的ReLU非线性进行可靠近似。我们在IEEE-24、IEEE-39和IEEE-118测试系统上，对PF、OPF和CFA三个电力系统任务评估了GNNV（原文摘要至此截断）

    arXiv:2609.30079v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) have become a prominent approach for developing fast, topology-aware surrogates in electric power systems, supporting tasks such as power flow (PF) analysis, optimal power flow (OPF) estimation, and cascading failure analysis (CFA). Despite this growing use, formally verifying GNN-based models remains challenging, with existing methods limited in scope. We extend the neural network verification (NNV) framework to graph-structured inputs through GraphStar sets, a generalization of Star sets that captures uncertainty over both node and edge features. This extension enables the propagation of linear message-passing operations and the sound approximation of ReLU nonlinearities for GNN architectures, including graph convolutional network (GCN) and graph isomorphism network with edge features (GINE) layers. We evaluate GNNV across three power system tasks, PF, OPF, and CFA, on the IEEE-24, IEEE-39, and IEEE-118 t
    
[^28]: 评估结论的可复现性如何？对LLM推断提示结构的自我审计

    How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure

    [https://arxiv.org/abs/2609.30074](https://arxiv.org/abs/2609.30074)

    这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。

    

    对LLM系统的评估通常在小规模提示集上取平均值，并以排名表的形式报告模型表现。我们提出这样一个问题：这样的排名表值得多少信任？并以基于LLM的提示结构推断作为案例研究：涵盖五个系列的八个开放模型变体，参数量从8B到675B，禁用缓存，持久化了293个原始中间表示。所测量的现象本身就是不稳定的：相同的调用无法可靠地恢复相同的结构，节点集Jaccard相似度均值从0.39到0.96不等，72%的提示-模型组合从未达到完美的节点集匹配。对评估过程本身的审计进一步削弱了其结论，这是我们的主要贡献。在基于提示的联合聚类自助法检验下，只有排名的底部是稳固的：可复现性最差的两个模型在99%和86%的重复实验中保持排名不变，中间四个模型仅占27%到48%，排名前两位的模型各占68%。因此，该排名表能够可靠地识别最差的模型，但无法可靠地确定其余模型的位置。

    arXiv:2609.30074v1 Announce Type: cross  Abstract: Evaluations of LLM systems routinely average over small prompt sets and report models as a ranked table. We ask how much confidence such a table deserves, using LLM-based prompt-structure inference as the case study: eight open model variants across five families and 8B to 675B parameters, caching disabled, 293 raw intermediate representations persisted. The measured phenomenon is unstable to begin with. Identical calls do not reliably recover identical structure, with mean node-set Jaccard from 0.39 to 0.96 and 72% of prompt-model cells never node-set-perfect. Auditing the evaluation weakens its conclusions further, and this is our main contribution. Under a joint cluster bootstrap over prompts, only the bottom of the ranking is firm: the two least reproducible models hold rank in 99% and 86% of replicates, the middle four in 27% to 48%, and the top two in 68% each, so the table identifies the worst model reliably but does not reliabl
    
[^29]: 零数据自博弈预训练

    Self-Play Pretraining with Zero Data

    [https://arxiv.org/abs/2609.30063](https://arxiv.org/abs/2609.30063)

    该论文提出零数据自博弈预训练方法，让生成器提出由通用图灵机执行的程序来生成字节序列、学习器自回归预测这些序列，两个模型协同自博弈进化，将合成数据生成建模为受所罗门诺夫归纳启发的可计算结构空间搜索，从而实现完全不依赖人类数据、仅受算力限制的预训练。

    

    语言建模的进步一直依赖于在越来越多的数据上扩大预训练规模。然而，训练数据在很大程度上仍然是为模型精心策划的。一种更通用的预训练方式应当让模型学会自己生成对自身改进最有用的数据。这将提供一个实际上无边界的数据源，其限制来自算力而非人类知识。我们提出了零数据自博弈预训练，这是实现这一愿景的初步概念验证。我们的方法将合成数据生成视为对所有可计算结构空间的搜索，其灵感来自所罗门诺夫归纳。从随机初始化开始，两个模型协同学习：生成器提出程序，由通用图灵机解释执行以生成字节序列，而学习器则以自回归方式预测这些字节序列。学习器使用标准的交叉熵进行训练，而生成器……

    arXiv:2609.30063v1 Announce Type: new  Abstract: Advances in language modeling have been driven by scaling pretraining on ever more data. Yet, the training data is still largely curated on the model's behalf. A more general approach to pretraining would let the model learn to generate the data most useful for its own improvement. This would provide an effectively unbounded source of training data, limited by compute rather than human knowledge. We introduce Self-Play Pretraining with Zero Data, an initial proof-of-concept towards realizing this vision. Our procedure casts synthetic data generation as a search over the space of all computable structure, taking inspiration from Solomonoff induction. Starting from random initialization, two models learn in tandem: a generator proposes programs interpreted by a universal Turing machine, generating byte sequences, while a learner autoregressively predicts these byte sequences. The learner is trained with standard cross-entropy, while the ge
    
[^30]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^31]: 劳动市场在AI时代还能正常运转吗？招聘中的评估瓶颈

    Can Labor Markets Function in the Age of AI? The Evaluation Bottleneck in Hiring

    [https://arxiv.org/abs/2609.30058](https://arxiv.org/abs/2609.30058)

    AI求职工具降低了申请材料的信息含量，使企业更依赖工作经验等粗略信号进行筛选，从而对“缺乏经验但与岗位高度匹配”的求职者造成最严重的负面影响。

    

    AI辅助求职工具通过让寻找工作和投递申请变得更加容易而日益流行。但由于让求职者更容易生成和定制申请材料，这些工具同时也降低了申请材料在反映求职者与岗位匹配度方面的信息含量。我们在一个招聘市场中研究这种权衡：其中求职者在工作经验和潜在匹配质量上存在差异，企业则依据含噪声的申请材料来决定筛选哪些候选人。我们探讨AI如何影响下游的筛选与招聘环节，以及哪些求职者受到的负面影响最大。随着申请材料的信息含量下降，贝叶斯理性的企业会更加依赖粗略的可观察指标（如过往工作经验）。在由经验和岗位匹配度定义的四类求职者中，“无经验但匹配”的求职者面临的风险最大：他们既缺乏可观察的工作经验，又失去了原本能够将他们与其他候选人区分开来的个性化信息。

    arXiv:2609.30058v1 Announce Type: cross  Abstract: AI-assisted job-search tools have become increasingly popular by making it easier to find and apply to jobs. But by making it easier for applicants to generate and tailor application materials, they can also reduce how informative those materials are about applicant fit. We study this tradeoff in a hiring market where applicants differ in experience and latent match quality and firms use noisy application materials to decide whom to screen. We ask how AI affects downstream screening and hiring, and which applicants are most adversely affected. As application materials become less informative, a Bayesian firm rationally relies more heavily on coarse observables such as prior experience. Among the four applicant types defined by experience and compatibility for the job, inexperienced-compatible applicants are the most exposed: they lack observable experience and lose the individualized information that could distinguish them from other i
    
[^32]: Era by Eon：基于隐藏知识的企业智能体基准测试

    Era by Eon: Benchmarking Enterprise Agents on Hidden Knowledge

    [https://arxiv.org/abs/2609.30055](https://arxiv.org/abs/2609.30055)

    本文通过向Era by Eon基准引入八个依赖隐藏事实的问题模板——这些事实未被任何问题或文档直接陈述、只能从其他数据中推断——有效区分了在可运行代码场景下表现趋同的企业智能体，其中最强智能体在24次作答中答对18次，而六个模型中有四个最多仅答对6次。

    

    在Era by Eon基准测试中，每个问题都说明了其答案的规则，代码根据一个生成的公司的数据计算出答案。当智能体可以运行代码时，四个最强的模型各自能答对27个此类问题中的22至25个，因此该基准几乎无法将它们区分开来。我们增加了八个依赖隐藏事实的问题模板。没有任何问题或文档直接陈述隐藏事实，看似包含该事实的记录显示的却是其他内容，隐藏事实由其他数据暗示得出。例如，销售系统记录某客户因时间原因放弃了一笔购买，但在一段通话录音中，客户却将原因归咎于服务中断。对于每个生成的公司，代码填充每个模板并在无需语言模型的情况下计算出精确答案。我们评估了12个智能体，每个智能体由一个模型与一个智能体程序配对，该程序将其连接到公司的各个系统。表现最好的智能体在其24次作答（每个问题3次）中正确回答了18次，六个模型中有四个在24次作答中最多仅答对6次（摘要在此处截断）。

    arXiv:2609.30055v1 Announce Type: cross  Abstract: In the Era by Eon benchmark, each question states the rules for its answer, and code computes the answer from a generated company's data. When agents can run code, the four strongest models each answer 22 to 25 of 27 such questions, so the benchmark barely separates them.   We add eight question templates that depend on hidden facts. No question or document states a hidden fact, and the records that seem to hold it show something else. Other data implies it. For example, the sales system says a customer dropped a purchase because of timing. On a recorded call, the customer blames an outage.   For each generated company, code fills each template and computes an exact answer without a language model. We evaluate 12 agents. Each pairs a model with an agent program, which connects it to the company's systems.   The best agent answers 18 of its 24 attempts, three per question, correctly. Four of the six models answer at most 6 of 24 with an
    
[^33]: SciWalker：基于算子图与执行反馈的科学编码问题合成

    SciWalker: Synthesizing Scientific Coding Problems with Operator Graphs and Execution Feedback

    [https://arxiv.org/abs/2609.30054](https://arxiv.org/abs/2609.30054)

    SciWalker 提出了一个科学编码问题合成框架，通过从算子图中采样算子链作为计算工作流线索、引导 LLM 生成科学问题及其解答与测试，并利用执行反馈迭代修复，从而自动生成高质量的 LLM 科学编码训练数据。

    

    提升大型语言模型（LLM）的科学编码能力需要高质量的训练数据。然而，这类数据仍然稀缺，因为人工编写贴近真实的问题既昂贵又耗时，而系统性地覆盖多样的科学领域和算法组合也颇具挑战。为解决这一问题，我们提出了 SciWalker，一个通过算子链采样和执行反馈来合成科学编码问题的框架。该框架将科学库接口与操作模式相结合以实例化算子，将算子组织成算子图，并采样算子链作为计算工作流线索。在这些线索的引导下，我们利用大语言模型生成具有科学依据的问题描述、参考解答和测试用例，并借助执行反馈对失败的生成结果进行迭代修复。通过将结构化的工作流组合与验证和质量控制相结合……（原文摘要在此处截断）

    arXiv:2609.30054v1 Announce Type: new  Abstract: Improving the scientific coding capabilities of large language models (LLMs) requires high-quality training data. However, such data remain scarce because manually authoring realistic problems is costly and time-consuming, while systematically covering diverse scientific domains and algorithmic combinations remains challenging. To address this, we introduce SciWalker, a framework for synthesizing scientific coding problems through operator-chain sampling and execution feedback. The framework combines scientific library interfaces with operation modes to instantiate operators, organizes them into operator graphs, and samples operator chains as computational workflow cues. Guided by these cues, we adopt LLMs to generate scientifically grounded problem statements, reference solutions, and tests, with failed generations iteratively repaired using execution feedback. By combining structured workflow composition with verification and quality r
    
[^34]: NNV3：将神经网络验证扩展到新架构与新领域

    NNV3: Expanding Neural Network Verification to New Architectures and Domains

    [https://arxiv.org/abs/2609.30050](https://arxiv.org/abs/2609.30050)

    NNV3作为NNV验证工具的最新版本，通过引入ModelStar、VolumeStar、GraphStar等新型Star集验证器、基于保形推断的概率可达性分析以及FairNNV公平性认证，将神经网络形式化验证扩展到权重扰动、视频与3D输入、图神经网络、公平性等新架构和新应用领域。

    

    我们提出NNV3，这是神经网络验证（NNV）工具的最新版本，一个用于深度学习模型和学习使能信息物理系统形式化验证的MATLAB框架。在NNV 1.0（前馈神经网络、卷积神经网络、神经信息物理系统）和NNV 2.0（循环神经网络、跳跃状态神经网络、神经常微分方程）的集合可达性分析基础之上，NNV3引入了Star集家族的新成员：用于验证权重扰动下网络的ModelStar，用于视频和3D体积输入的VolumeStar，以及用于图神经网络的GraphStar。基于保形推断的概率可达性模式为确定性验证难以处理的问题提供了补充性健全分析，而FairNNV则可在连续输入区域上认证反事实公平性和个体公平性属性。NNV3还引入了针对恶意软件检测、基于图的电力系统模型、医学影像、可变长时间序列数据和动作识别的新基准。NNV3还包含……（原文摘要在此处被截断）

    arXiv:2609.30050v1 Announce Type: new  Abstract: We present NNV3, the latest version of the Neural Network Verification (NNV) tool, a MATLAB framework for formal verification of deep learning models and learning-enabled cyber-physical systems. Building on the set-based reachability foundation of NNV 1.0 (FFNNs, CNNs, NNCS) and NNV 2.0 (RNNs, SSNNs, neural ODEs), NNV3 introduces new members of the Star-set family: ModelStar for verifying networks under weight perturbation, VolumeStar for video and 3D volumetric inputs, and GraphStar for graph neural networks. A conformal-inference-based probabilistic reachability mode complements sound analysis for problems where deterministic verification is intractable, while FairNNV certifies counterfactual and individual fairness properties over continuous input regions. NNV3 introduces new benchmarks for malware detection, graph-based power-system models, medical imaging, variable-length time series data, and action recognition. NNV3 also incorpora
    
[^35]: 风格而非自我：表层线索解释大语言模型的零样本代码归因

    Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models

    [https://arxiv.org/abs/2609.30048](https://arxiv.org/abs/2609.30048)

    研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。

    

    如果语言模型能够识别自己编写的代码，它可能会在充当评判者时偏袒该代码，而模型之间相互监控的情境可能导致合谋。我们在当前商业模型上对这种零样本能力进行了测试。五个大语言模型为MBPP、HumanEval和DS-1000生成解题方案，另有七个模型为MBPP生成方案，模型在四项任务中充当评估者：从一对方案中挑选出自己的方案、判断单个方案是否出自自己、识别两个方案中哪一个由指定模型编写，以及在盲测条件下评判代码质量。在单方案任务中，所有15个模型-基准组合的平衡准确率为49-58%，而原始准确率（38-67%）主要反映了模型宣称代码作者身份的难易程度。在成对任务中，14个评估者-对手组合的准确率与评估者自身方案更长这一因素的相关性高达r=0.93。对指定模型的归因在某些配对上取得成功，而在其他配对上则出现持续性的反转。一种基于规则的标准化方法……（摘要在此处被截断）

    arXiv:2609.30048v1 Announce Type: new  Abstract: If a language model can recognize code it wrote, it may favor that code as a judge, and instances of one model monitoring each other could collude. We test this zero-shot on current commercial models. Five LLMs generate solutions to MBPP, HumanEval, and DS-1000, seven more to MBPP, and models act as evaluators in four tasks: picking their own solution from a pair, judging whether a single solution is their own, identifying which of two solutions a named model wrote, and judging quality blind. In the single-solution task, balanced accuracy is 49-58% for all 15 model-benchmark combinations, while raw accuracy (38-67%) mostly reflects how readily a model claims authorship. In the pairwise task, accuracy across 14 evaluator-opponent combinations correlates at r=0.93 with how often the evaluator's solution is longer. Attribution to a named model succeeds on some pairs and is consistently inverted on others. A rule-based normalization that str
    
[^36]: 对抗性影响在多智能体系统中如何扩展？

    How does Adversarial Influence Scale in Multi-Agent Systems?

    [https://arxiv.org/abs/2609.30028](https://arxiv.org/abs/2609.30028)

    该研究发现多智能体系统中LLM智能体对欺骗的易感性取决于欺骗者的比例而非群体规模，背叛率随欺骗者比例线性上升，且与人类不同，即使欺骗者占少数，LLM智能体也会频繁背叛并改变正确答案。

    

    多智能体协商可以提高性能，但当某些智能体不怀好意地行动时会发生什么？在实践中，一个智能体可能是欺骗性的，并致力于颠覆整个群体，无论是出于自身目标还是外部指令。我们研究了随着群体规模增大和欺骗者数量增加，对欺骗的易感性如何扩展。重要的不是群体中智能体的数量，而是欺骗者所占的比例。我们观察到，背叛率（即最初回答正确的智能体切换到错误最终答案的频率）随这一比例呈线性上升。与人类在类似从众性研究中的表现不同，人类只有在误导性同谋者构成多数时才会被可靠地动摇，而LLM智能体即使欺骗者仍占少数时也会定期背叛。易感性还取决于哪些模型在相互作用，尤其是在诚实智能体一方。出乎意料的是，允许欺骗者私下协调反而可能降低他们的效果。

    arXiv:2609.30028v1 Announce Type: new  Abstract: Multi-agent deliberation can improve performance, but what happens when some agents do not act in good faith? In practice, an agent may be deceptive and work to subvert the group, whether through its own objectives or external instruction. We study how susceptibility to deception scales as groups increase in size and deceivers become more prevalent. It is not the number of agents in the group that matters, but the proportion of deceivers. We observe that the defection rate, how often initially correct agents switch to an incorrect final answer, rises linearly with this proportion. Whereas humans in comparable conformity studies are reliably swayed only when misleading confederates form a majority, LLM agents defect regularly even when deceivers remain a minority. Susceptibility also depends on which models are interacting, especially on the honest agent side. Unexpectedly, allowing deceivers to coordinate privately can make them less eff
    
[^37]: 合成医院：一个开放、可验证、经医生验证的纵向电子健康记录基准

    Synthetic Hospital: An Open, Verifiable, Physician-Validated Longitudinal EHR Benchmark

    [https://arxiv.org/abs/2609.30027](https://arxiv.org/abs/2609.30027)

    提出了“合成医院”——首个完全由公开医学教育材料构建、经医生验证、具有完整溯源链和可验证真实标准答案的开放合成纵向EHR基准，解决了真实EHR数据无法公开共享且缺乏可验证答案的核心障碍。

    

    前沿语言模型很少被应用于临床工作流程，原因在于开发此类模型所需的现实、纵向的基准测试数据十分稀缺。真实的电子健康记录（EHR）数据由于隐私、伦理或数据使用等问题无法公开共享，而且由于病历仅记录临床医生所记载的内容，因此缺乏可验证的真实标准答案。我们提出了“合成医院”，这是一个开放、完全合成、基于事实的纵向EHR基准，解决了公开共享和可验证真实标准这两大障碍。该基准完全由公开的医学教育材料构建，不含任何受保护的健康信息，包含1,268名纵向患者和5,602次就诊记录，其中每一个诊断、发现和时间关系均基于标准本体（ICD-10-CM、SNOMED CT、LOINC）进行标注，并拥有完整的溯源链，可追溯至源医学教育材料。合成医院通过一个模拟医院环境提供服务……（原文摘要在此处截断）

    arXiv:2609.30027v1 Announce Type: new  Abstract: Frontier language models are rarely used in clinical workflows because the realistic, longitudinal benchmarks needed to develop them are scarce. Real electronic health record (EHR) data cannot be openly shared due to privacy, ethics or data use issues and it does not contain verifiable ground truth since the chart records only reflect what clinicians documented. We introduce Synthetic Hospital, an open, fully synthetic, fact-grounded longitudinal EHR benchmark that resolves the open sharing and verifiable ground truth barriers. Built entirely from public medical-education material with no protected health information, it comprises 1,268 longitudinal patients and 5,602 encounters, where every diagnosis, finding, and temporal relation is grounded in standard ontologies (ICD-10-CM, SNOMED CT, LOINC) and with a complete provenance chain back to its source medical education material. Synthetic Hospital is served through a simulated hospital r
    
[^38]: Canopy：利用分段平滑树先验的多保真度老虎机算法

    Canopy: Exploiting Piecewise Smooth Tree Priors for Multi-Fidelity Bandits

    [https://arxiv.org/abs/2609.30017](https://arxiv.org/abs/2609.30017)

    CANOPY是一种多保真度树状老虎机算法，通过廉价的随机路径探测在线学习分段平滑先验在树结构中的有效区域，从而突破传统方法需预先假设全局平滑性的局限，更精准地引导昂贵的叶节点评估。

    

    许多LLM推理问题，包括模型路由、前缀缓存管理、提示词裁剪和测试时搜索，都可以被视为对树结构的优化。这种结构自然地源于自回归生成：每个前缀定义一个节点，其后续延续构成其下方的一个子树。树的内部节点能够提供对某区域价值的廉价但有偏的估计，而叶节点评估则昂贵但准确。层次化老虎机方法可以利用这种结构，但通常需要预先指定特定的平滑性调度，尽管实际目标往往只是分段平滑的，且其最优值可能位于急剧变化的边界附近。我们提出了CANOPY，这是一种多保真度树状老虎机算法，它通过学习确定平滑先验在何处有效，而不是全局地假设其成立。CANOPY使用廉价的随机路径探测来构建局部聚合偏差的在线证书，然后将昂贵的叶节点评估引导至最值得评估的区域。

    arXiv:2609.30017v1 Announce Type: cross  Abstract: Many LLM inference problems, including model routing, prefix-cache management, prompt trimming, and test-time search, can be viewed as optimization over a tree. This structure arises naturally from autoregressive generation: every prefix defines a node, and its continuations form a subtree below it. Internal nodes of the tree provide cheap but biased estimates of a region's value, while leaf evaluations are expensive but accurate. Hierarchical bandit methods can exploit this structure, but typically require a specific smoothness schedule to be specified in advance, even though real objectives are often only piecewise smooth and their optima may lie near sharp boundaries. We introduce CANOPY, a multi-fidelity tree bandit that learns where the smoothness prior is valid rather than assuming it globally. CANOPY uses cheap random-path probes to construct an online certificate of local aggregation bias, then directs expensive leaf evaluation
    
[^39]: 跨厂商与跨版本测量模型行为的低成本检测方法

    Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases

    [https://arxiv.org/abs/2609.30012](https://arxiv.org/abs/2609.30012)

    提出一种简单、廉价、可扩展且可复现的方法，通过在跨厂商模型面板上运行冻结的公开刺激任务，以精确匹配、LLM编码手册或插桩环境三种方式测量模型行为，单个模型成本仅需几美元。

    

    语言模型为人们提供建议、陪伴他们，并在他们睡觉时编写软件。然而测量它们的行为十分困难：行为需要在不同的模型、提示词和版本之间反复采样，其中大部分以非结构化文本形式存在，必须先编码才能统计，而且结果必须足够清晰且严谨，才能有意义地比较不同模型和厂商。为解决这些限制，我们提出了一种简单、廉价、可扩展且可复现的模型行为研究方法。每项研究都是一个冻结的、公开的刺激任务，以完全相同的方式在跨厂商的模型面板上运行，每个模型的成本仅为几美元甚至更低。每项研究根据行为所需的解释程度，以三种方式之一读取对话记录：对受限回复进行精确匹配；由LLM评审员应用编码手册，并按编码报告其与人类编码员的一致性；以及通过一个经过插桩的环境，独立于智能体所说的内容记录其实际行为。研究跨越四年运行……

    arXiv:2609.30012v1 Announce Type: cross  Abstract: Language models advise people, keep them company, and write software while they sleep. Measuring what they do is hard: behavior has to be sampled repeatedly across models, prompts and releases, most of it lives in unstructured text that has to be coded before it can be counted, and the result has to be legible and rigorous enough to meaningfully compare models and vendors. To address these constraints, we present a simple, cheap, scalable, and replicable model for studying model behavior. Each study is a frozen, public stimulus run identically on a cross-vendor panel, at a few dollars per model or less. Each reads its transcripts one of three ways, chosen by how much interpretation the behavior needs: exact match on a clamped reply, a codebook applied by LLM judges whose agreement with a human coder is reported per code, and an instrumented environment that records what an agent did independently of what it said. Run across four years 
    
[^40]: 基于领域自适应检索增强生成的金融服务自动化监管合规问答

    Automated Regulatory Compliance Question Answering in Financial Services with Domain-Adapted Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.30009](https://arxiv.org/abs/2609.30009)

    本文提出一条在 LegalBERT 上经三阶段领域自适应训练的检索器与 4 位量化紧凑生成器相结合的检索增强生成流水线，使可本地部署的小型模型也能在金融监管合规问答中给出有据可查、低幻觉的回答。

    

    金融机构在密集且频繁修订的规则手册下运营，正确回答合规问题不仅需要语言流畅性，还需要在权威文本中具备可验证的依据。大型语言模型对这一任务颇具吸引力，但企业实际能够本地部署的是紧凑型模型，而紧凑型模型会产生“义务幻觉”。我们研究经过精心领域自适应的检索增强生成（RAG）流水线能否弥合这一差距。我们的检索器基于 LegalBERT 分三个阶段构建：将“问题-段落”匹配重构为“前提-假设”重建任务的蕴含调优、采用批内负样本的对比调优，以及与 BM25 的分数级融合。我们的生成器是一个紧凑模型（2B–12B 参数），在 4 位量化下运行，或采用提示方式，或通过 LoRA 进行检索感知微调（RAFT）加以适配。在基于阿布扎比全球市场规则构建的问答基准 ObliQA 上（原文摘要此处截断）……

    arXiv:2609.30009v1 Announce Type: cross  Abstract: Financial institutions operate under dense, frequently amended rulebooks, and answering a compliance question correctly requires not only fluency but verifiable grounding in the authoritative text. Large language models are attractive for this task, yet the models that firms can realistically deploy on-premise are compact ones, and compact models hallucinate obligations. We study whether a carefully domain-adapted retrieval-augmented generation pipeline closes that gap. Our retriever is built in three stages on top of LegalBERT: entailment tuning that recasts question--passage matching as premise--hypothesis reconstruction, contrastive tuning with in-batch negatives, and score-level fusion with BM25. Our generator is a compact model (2B--12B parameters) served under 4-bit quantization, either prompted or adapted with retrieval-aware fine-tuning (RAFT) through LoRA. On ObliQA, a question-answering benchmark built from the Abu Dhabi Glob
    
[^41]: AgentX中推进模型研究：面向工业推荐系统的长程自主性

    Advancing Model Research in AgentX: Long-Horizon Autonomy for Industrial Recommender Systems

    [https://arxiv.org/abs/2609.30001](https://arxiv.org/abs/2609.30001)

    AgentX-Model是一个面向工业推荐系统的双智能体自主研究框架，通过研究智能体与模型智能体的协作，围绕复现、跟进、组合和诊断四种行动实现长程自主的持续性模型研究。

    

    持续的工业推荐研究需要利用一次实验的结果来决定下一步的研究方向。我们提出了AgentX-Model，这是AgentX模型研究框架的下一代版本，它在由业务输入和预测任务定义的沙盒环境中将提案开发与模型实验连接起来。AgentX-Model采用由研究智能体和模型智能体组成的双智能体架构。研究智能体从论文和实验发现中制定经过独立评审的提案，而模型智能体进行多轮调查并返回代码、测量结果和未解决的问题。利用返回的结果，研究智能体选择一个起始实现并制定下一个研究问题，使后续实验能够建立在先前发现的基础上。我们将这种持续研究围绕四种行动来组织：复现、跟进、组合和诊断。

    arXiv:2609.30001v1 Announce Type: new  Abstract: Sustaining industrial recommendation research requires using the results of one experiment to decide what to investigate next. We present AgentX-Model, the next generation of AgentX's model research framework, which connects proposal development and model experimentation within sandboxes defined by business inputs and prediction tasks. AgentX-Model adopts a dual-agent architecture comprising a Research Agent and a Model Agent. The Research Agent develops independently reviewed proposals from papers and experimental findings, while the Model Agent conducts multi-round investigations and returns code, measurements, and unresolved questions. Using the returned results, the Research Agent selects a starting implementation and formulates the next research question, allowing subsequent experiments to build on earlier findings. We organize this continuing research around four actions: Reproduce, Follow-up, Composition, and Diagnose. The first t
    
[^42]: GHOST-Q：研究量化视觉语言模型中被忽视的同分数权衡下的接地幻觉问题

    GHOST-Q: Towards Studying Grounding Hallucinations Overlooked Under Same-score TradeOffs in Quantized VLMS

    [https://arxiv.org/abs/2609.29999](https://arxiv.org/abs/2609.29999)

    该论文提出GHOST-Q评估框架，通过将FP16与量化VLM的预测逐项配对，揭示出量化模型即使在总体准确率几乎不变的情况下，其视觉接地与幻觉行为仍发生显著变化，且内存大幅节省并不保证推理延迟降低。

    

    视觉-语言模型（VLM）的训练后量化通常通过总体任务准确率和内存节省来评估，但保持总体分数并不能保证视觉接地行为得以保留。我们提出GHOST-Q，这是一项跨精度的受控评估，在FP16、INT8和NF4三种精度下，对三个8B规模的VLM系列在实用性和幻觉敏感基准上进行了评测。我们不仅仅比较总体准确率，而是将FP16与量化模型的预测逐项配对，以量化压缩如何重新分配接地任务的成功与失败。结果显示，六个量化变体中有五个将MMStar准确率保持在±2个百分点以内，然而经过错误发现率校正后，36个配对效应中仍有10个保持显著，其中九个出现在幻觉敏感条件下。同设备A100上的性能分析进一步表明，大幅的内存节省并不一定意味着更低的推理延迟。最后，一个开放式AMBE

    arXiv:2609.29999v1 Announce Type: cross  Abstract: Post-training quantization of vision--language models (VLMs) is typically assessed through aggregate task accuracy and memory savings, but preserving a headline score does not guarantee preservation of visual grounding behavior. We present GHOST-Q, a cross-precision controlled evaluation of three 8B VLM families under FP16, INT8, and NF4 across utility and hallucination-sensitive benchmarks. Rather than comparing only aggregate accuracy, we pair FP16 and quantized predictions item by-item to quantify how compression redistributes grounding successes and failures. Five of six quantized variants preserve MMStar accuracy within $\pm2$ percentage points, yet 10 of 36 paired effects remain significant after false-discovery-rate correction, nine on hallucination-sensitive conditions. Same-device A100 profiling further demonstrates that substantial memory reduction does not necessarily mean lower inference latency. Finally, an open-ended AMBE
    
[^43]: 护栏还是绊脚石？编程课程中AI助教的教学风格与情境感知的影响

    Guardrails or Roadblocks? Effects of Pedagogical Style and Context Awareness in AI Teaching Assistants for Programming

    [https://arxiv.org/abs/2609.29995](https://arxiv.org/abs/2609.29995)

    本研究通过132名学生的随机对照试验，探究AI编程助教的教学引导风格与情境感知能力对学习体验的影响，发现护栏设计若过于受限或缺乏情境关联，可能导致学生转而使用通用大语言模型。

    

    基于大语言模型（LLM）并带有教学护栏的AI助教正日益被整合到编程课程中，为学生提供可规模化的提示、概念解释和代码级反馈。然而，护栏也可能造成阻碍：如果学生感到所提供的支持过于受限，或与其当前学习进度缺乏情境关联，他们可能会绕过指定工具，转而使用通用的大语言模型。为了探究AI助教的设计如何影响学生的学习体验，我们在一门编程入门课程中开展了一项随机对照试验，共招募132名学生。学生完成了三个与代码编写和调试相关的任务，并被随机分配至四种AI助教之一。这四种助教在两个维度上有所差异：教学引导风格（苏格拉底式提问 vs. 直接讲授）和情境感知（无情境 vs. 掌握题目与学生解答的完整情境）。

    arXiv:2609.29995v1 Announce Type: cross  Abstract: AI teaching assistants (AI TAs) backed by large language models (LLMs) and pedagogical guardrails are increasingly being integrated into programming courses, providing students with scalable access to hints, conceptual explanations, and code-level feedback. However, guardrails may also create friction. If students feel that the support provided is overly restrictive or poorly contextualized to their current progress, they may bypass approved tools for general-purpose LLMs. To investigate how AI TA design affects students' learning experiences, we conducted a randomized controlled trial with 132 students in an introductory programming course. Students completed three tasks related to code-writing and debugging and were randomly assigned to one of four AI TAs varied across two dimensions: pedagogical guidance style (Socratic vs. Direct instruction) and context awareness (no context vs. full context of the problem and student solution). W
    
[^44]: 从兴趣到语义ID：面向生成式推荐的基于检索的信用分配

    From Interests to Semantic IDs: Retrieval-Grounded Credit Assignment for Generative Recommendation

    [https://arxiv.org/abs/2609.29983](https://arxiv.org/abs/2609.29983)

    提出基于检索的查询归因方法，为每条推理轨迹分配可追溯的兴趣级信用，从而解决生成式推荐训练中稀疏精确匹配SID奖励导致的信用分配缺口问题。

    

    语义ID（SIDs）将目录中的每个物品编码为一段简短的token序列，使生成式推荐器能够以自回归方式预测下一个物品。推理增强的变体（一种日益普遍的扩展形式）会先生成一段文本推理轨迹，再通过束搜索解码出下一个物品的SID。这类推荐器通常在精确匹配SID奖励下采用组相对策略优化进行训练，而这种奖励在大规模目录中非常稀疏。由此产生两种失败模式：当一组rollout全部未命中目标时，该组的优势为零，不产生任何学习信号；而共享相同SID奖励的rollout无论其推理轨迹差异多大，都会获得完全相同的优势。在这两种情况下，奖励仅反映解码出的SID，而从不反映产生该SID的推理过程，从而造成信用分配缺口。我们通过基于检索的查询归因来弥合这一缺口：每条轨迹被结构化为一个历史摘要、一组兴趣假设以及……

    arXiv:2609.29983v1 Announce Type: cross  Abstract: Semantic IDs (SIDs) encode each catalog item as a short token sequence, enabling generative recommenders to predict the next item autoregressively. Reasoning-enhanced variants, an increasingly common extension, first generate a textual trace and then decode a next-item SID by beam search. Such recommenders are commonly trained with group-relative policy optimization under an exact-match SID reward, which is sparse in large catalogs. Two failure modes follow. When all rollouts in a group miss the target, the group yields zero advantage and no learning signal. Rollouts sharing the same SID reward receive identical advantages, however much their traces differ. In both cases the reward reflects only the decoded SID, never the reasoning that produced it. This creates a credit-assignment gap.   We address this gap with retrieval-grounded query attribution. Each trace is structured into a history summary, a set of interest hypotheses, and a f
    
[^45]: 基于语义ID学习生成式推荐的更优推理

    Learning Better Reasoning for Generative Recommendation with Semantic IDs

    [https://arxiv.org/abs/2609.29973](https://arxiv.org/abs/2609.29973)

    提出Evo-Rec三阶段框架，使生成式推荐系统能够筛选有效的推理轨迹并从自身生成中逐步学习更优的推理，从而避免低质量推理误导物品生成并提升推荐性能。

    

    生成式推荐将物品检索重新表述为序列生成任务，使统一模型能够直接从用户的交互历史中生成下一个物品。语义ID通过将每个物品表示为离散编码，使语义相关的物品之间能够共享知识，从而让这一范式变得高效且可扩展。最近的研究在语义ID生成之前引入显式推理，帮助模型总结用户兴趣并推断可能的偏好转变。然而，推理并不天然有益：不准确或缺乏信息量的推理可能会误导后续的物品生成，最终降低推荐性能。这引出了一个核心挑战：推荐系统如何选择并学习有效的推理轨迹，并从自身生成的内容中逐步演进出更好的推理？在这项工作中，我们提出了Evo-Rec，一个用于学习更优推理并进一步提升推荐性能的三阶段框架。

    arXiv:2609.29973v1 Announce Type: cross  Abstract: Generative recommendation reformulates item retrieval as sequence generation, allowing a unified model to directly generate the next item from a user's interaction history. Semantic IDs further make this paradigm effective and scalable by representing each item as discrete codes, enabling knowledge sharing among semantically related items. Recent studies introduce explicit reasoning before Semantic-ID generation, helping models summarize user interests and infer possible preference transitions. However, reasoning is not inherently beneficial: Inaccurate or uninformative reasoning may mislead subsequent item generation and ultimately degrade recommendation performance. This raises a central challenge: how can a recommender select and learn effective reasoning traces and progressively evolve toward better reasoning from its own generations? In this work, we propose Evo-Rec, a three-stage framework for learning better reasoning and furthe
    
[^46]: 世界动作智能体：通过世界动作预演利用视觉语言模型进行机器人操作

    World Action Agent: Harnessing VLMs for Robot Manipulation via World Action Rehearsal

    [https://arxiv.org/abs/2609.29964](https://arxiv.org/abs/2609.29964)

    该论文提出世界动作智能体（WAA），通过包含接触视图、动作预演和视图内纠正三大特性的视觉动作工作空间，让视觉语言模型能够直接在“世界”中决策并操控机器人，从而实现观察、预演与底层执行的闭环。

    

    通用的视觉语言模型为机器人操作带来了广泛的知识和空间推理能力，但现有系统要么间接地使用它们（例如预测约束或编写程序），要么只给它们一个场景视图，而非一个可以行动的世界。我们提出了世界动作智能体（WAA），这是一个多智能体框架，通过该框架，VLM 使用基础工具操控机器人，并在视觉动作工作空间中做出每一个决策。该工作空间具有三个特性：接触视图从场景几何中自动选择，呈现当前交互周围的场景；动作预演将每个动作转化为可编辑的提案，智能体可以独立地或通过想象智能体，在执行前根据规划反馈进行预览和修改；视图内纠正则闭合了观察、预演与底层执行之间的循环，使智能体能够在观察到残余偏移的视图中直接消除这些偏差。

    arXiv:2609.29964v1 Announce Type: cross  Abstract: General-purpose vision-language models (VLMs) bring broad knowledge and spatial reasoning to robot manipulation, yet existing systems either use them indirectly, to predict constraints or write programs, or give them a view of the scene rather than a world in which to act. We present World Action Agent (WAA), a multi-agent harness through which VLMs pilot robots with basic tools, making every decision within a visual action workspace. The workspace has three properties. Contact views, selected automatically from the scene geometry, present the scene around the current interaction. Action rehearsal turns each action into an editable proposal that the agent, alone or through an Imagination Agent, previews and revises against planning feedback before execution. In-view correction closes the loop between observation, rehearsal, and low-level execution, letting the agent remove residual offsets in the view where it observes them. Through th
    
[^47]: ADATEX4D：面向4D高斯泼溅的自适应纹理容量分配

    ADATEX4D: adaptive texture capacity allocation for 4D gaussian splatting

    [https://arxiv.org/abs/2609.29963](https://arxiv.org/abs/2609.29963)

    提出AdaTex4D自适应纹理容量分配模块，根据可见性和局部尺度动态调整每个高斯RGBA三平面的分辨率，在保持重建质量的同时将4D高斯泼溅的纹理存储减少一半以上。

    

    带纹理的高斯提升了局部外观表达能力，但为每个图元分配相同的纹理分辨率会在低细节或弱可见区域浪费存储空间。我们提出了AdaTex4D，这是一个面向基于变形的4D高斯泼溅的自适应纹理容量模块。每个高斯都携带打包的RGBA三平面，其两个轴的尺寸根据可见性归一化的屏幕空间梯度和变形后的局部尺度独立增长。在N3DV和PanopticSports数据集上的实验表明，AdaTex4D在保持重建质量的同时，将纹理存储减少了超过一半。在固定内存预算下，自适应分配相比均匀纹理分配还能提升质量，并降低整体模型大小和峰值内存。这些结果表明，动态、各向异性的纹理分配为在4D高斯表示中分配局部外观容量提供了一种更高效的方式。

    arXiv:2609.29963v1 Announce Type: cross  Abstract: Textured Gaussians improve local appearance capacity, but assigning the same texture resolution to every primitive wastes storage on low-detail or weakly visible regions. We introduce AdaTex4D, an adaptive texture-capacity module for deformation-based 4D Gaussian Splatting. Each Gaussian carries packed RGBA triplanes whose two axes grow independently according to visibility normalized screen-space gradients and deformed local scales. Experiments on N3DV and PanopticSports show that AdaTex4D reduces texture storage by more than half while preserving reconstruction quality. Under fixed memory budgets, adaptive allocation also improves quality over uniform texture assignment and reduces overall model and peak memory. These results show that dynamic, anisotropic texture allocation provides a more efficient way to distribute local appearance capacity in 4D Gaussian representations.
    
[^48]: 超越平均安全性：机会约束的大语言模型微调

    Beyond Average Safety: Chance-Constrained LLM Fine-tuning

    [https://arxiv.org/abs/2609.29960](https://arxiv.org/abs/2609.29960)

    本文提出一种机会约束的保安全微调方法，通过限制安全样本相对参考模型退化超过阈值的比例，并利用可微上界与约束感知梯度下降算法，解决了传统平均安全损失掩盖罕见但严重安全失效的问题。

    

    在大语言模型上针对新目标进行微调可以提升有用性、指令遵循能力或领域特定性能，但也可能在安全关键提示上引发性能回退。现有的保安全微调方法通常控制平均安全损失或使用加权辅助惩罚，这可能会掩盖罕见但严重的失效情况。我们提出了一种用于保安全微调的机会约束公式化方法，该方法限制相对于参考模型退化超过规定阈值的安全样本比例。由于由此产生的经验机会约束包含不连续的指示函数，我们引入了违反率的可微优化上界，从而得到一个易于处理的保守约束。随后，我们开发了一种约束感知的梯度下降方法，将优化后的约束视为参数空间中的安全集，并以最小程度修改微调方向以保持……

    arXiv:2609.29960v1 Announce Type: cross  Abstract: Fine-tuning large language models on new objectives can improve helpfulness, instruction following, or domain-specific performance, but it can also induce regressions on safety-critical prompts. Existing safety-preserving fine-tuning methods typically control average safety loss or use weighted auxiliary penalties, which can obscure rare but severe failures. We propose a chance-constrained formulation for safety-preserving fine-tuning that limits the fraction of safety examples whose degradation relative to a reference model exceeds a prescribed threshold. Because the resulting empirical chance constraint contains a discontinuous indicator, we introduce a differentiable majorization of the violation rate, yielding a tractable conservative constraint. We then develop a constraint-aware gradient descent method that treats the majorized constraint as a safe set in parameter space and minimally modifies the fine-tuning direction to preserv
    
[^49]: Augur：用于预演产品与政策变更反应的合成决策实验室

    Augur: A Synthetic Decision Lab for Rehearsing Reactions to Product and Policy Changes

    [https://arxiv.org/abs/2609.29952](https://arxiv.org/abs/2609.29952)

    提出了离线决策预演系统 Augur 和包含五十个真实事件的 Gold-50 基准，核心发现是前沿云端模型与离线开源模型之间的大部分表现差距源于评估设定不充分而非能力差异。

    

    在产品或政策变更正式上线之前，关键的问题是人们将如何对其做出反应。Augur 能够离线预演这种反应：它从变更文档中构建类型化知识图谱，填充基于真实依据的角色市场，模拟交互过程，并返回一份可审计的决策备忘录，在五种行动中给出推荐。我们构建了 Gold-50 数据集——包含五十个真实产品与政策事件，其现实结果已知并依据公开记录进行裁定——据此对五选一的发布判定进行评分。我们的核心发现是方法论层面且是否定性的：前沿云端模型与我们微调并离线部署的开源权重模型之间所测得的大部分差距，可归因于评估设定不够充分，而非能力差异。我们通过三种方式证明了这一点。首先，仅提示词包络本身就能主导得分：在保持模型权重、案例和评分器不变的情况下，一个系统——Qwen3-32B 上的 LoRA-SFT 适配器——的得分从 0（摘要在此处被截断）

    arXiv:2609.29952v1 Announce Type: new  Abstract: Before a product or policy change ships, the question that matters is how people will react to it. Augur rehearses that reaction offline: it builds a typed knowledge graph from the change documents, populates a grounded persona market, simulates the interaction, and returns an auditable decision memo recommending one of five actions. We assemble Gold-50, fifty real product and policy episodes whose real-world outcome is known, adjudicated against the public record, and score the five-way release verdict against it.   Our central finding is methodological and negative: most of the measured gap between frontier cloud models and open-weight models we fine-tune and serve offline is attributable to an under-specified evaluation, not a difference in capability. We show this three ways. First, the prompt envelope alone can dominate the score: holding weights, cases and scorer fixed, one system -- a LoRA-SFT adapter on Qwen3-32B -- swings from 0
    
[^50]: 追踪状态还是追踪陪集？学习型状态追踪的代数解释

    Tracking States or Tracking Cosets? An Algebraic Account of Learned State Tracking

    [https://arxiv.org/abs/2609.29951](https://arxiv.org/abs/2609.29951)

    该论文从代数视角揭示Transformer在群运算状态追踪任务中学到的并非精确状态而是商类（陪集）解，并证明最优顺序无关准确率收敛于阿贝尔化类大小的倒数。

    

    状态追踪需要组合一系列更新，但仅凭准确率无法揭示模型究竟学到了什么。我们研究了被训练来预测群元素连续乘积的神经网络。我们在Transformer中识别出了“商解”现象：模型能够恢复商类，同时在其成员之间近乎均匀地进行预测。类大小的倒数无需任何拟合参数即可预测部分准确率，这将基于奇偶性的解释推广到了非奇偶性的商结构。我们的基线Transformer在超过精确追踪边界后，其预测在前缀重排下几乎不变。我们证明，对于在均匀独立同分布全群输入下的有限群，最优的顺序无关精确准确率随前缀长度增长收敛于阿贝尔化类大小的倒数，这与观察到的阿贝尔化平台现象一致。顺序更新还允许更多可能性：任何将群划分为子群（无论是否正规）的右陪集划分，都能在顺序更新下保持（原文摘要在此处被截断）。

    arXiv:2609.29951v1 Announce Type: cross  Abstract: State tracking requires composing a sequence of updates, but accuracy alone does not reveal what a model has learned. We study neural networks trained to predict the running product of group elements. We identify quotient solutions in Transformers, where models recover the quotient class while predicting nearly uniformly among its members. The reciprocal of class size predicts partial accuracy without a fitted parameter, extending parity-based accounts to non-parity quotients. Our baseline Transformers' predictions change little under prefix reordering beyond the exact-tracking frontier. We prove that, for finite groups under uniform i.i.d. full-group inputs, optimal order-blind exact accuracy converges to the reciprocal of abelianization class size as prefix length grows, consistent with the observed abelianization plateaus. Sequential updates permit more: any partition into right cosets of a subgroup, normal or not, survives sequenti
    
[^51]: ENDOPROMPT：基于受害者端伪参考的效用退化方法

    ENDOPROMPT: Victim-Side Pseudo-References for Utility Degradation

    [https://arxiv.org/abs/2609.29948](https://arxiv.org/abs/2609.29948)

    ENDOPROMPT是一种白盒提示注入方法，通过将受害者模型的干净续写作为伪参考，从无标签指令中学习生成降低效用的前缀，在不依赖有害内容或预定义目标的情况下平均降低模型任务效用26.8个百分点。

    

    提示注入攻击可以在不诱导有害内容的情况下降低良性任务的性能。然而，许多攻击目标依赖于任务标签或预定义的目标响应。我们提出了ENDOPROMPT，这是一种白盒方法，能够从无标签指令中学习降低效用的前缀。其生成器以请求文本作为输入。受害者的干净续写作为伪参考：局部搜索识别能够降低续写可能性的前缀，随后在同一指令内的比较上进行偏好拟合，再通过奖励细化，将这一信号蒸馏到生成器中。在部署时，生成器为每个请求生成一个前缀，无需进一步的受害者端搜索。在四个指令调优模型和七个良性基准的完整数据划分上，ENDOPROMPT平均带来-26.8个百分点的效用变化；28个实验组中有27个为负向。失败分析揭示了输出扩展和前缀复用问题；对照组则没有（注：原摘要末尾似乎被截断）。

    arXiv:2609.29948v1 Announce Type: new  Abstract: Prompt injection can degrade benign task performance without eliciting harmful content. Yet many attack objectives depend on task labels or predefined target responses. We present ENDOPROMPT, a white-box method that learns utility-degrading prefixes from unlabeled instructions. Its generator takes the request text as input. Clean victim continuations serve as pseudo-references: local search identifies prefixes that reduce continuation likelihood, and preference fitting on comparisons within the same instruction, followed by reward refinement, distills this signal into a generator. At deployment, the generator produces one prefix per request without further victim-side search. Across four instruction-tuned models and the complete splits of seven benign benchmarks, ENDOPROMPT yields a mean utility change of -26.8 percentage points; 27 of 28 cells are negative. Failure analysis reveals output expansion and prefix reuse; the controls do not 
    
[^52]: 面向工业产品配置的神经符号人工智能

    Neuro-symbolic AI for Industrial Configuration

    [https://arxiv.org/abs/2609.29947](https://arxiv.org/abs/2609.29947)

    本文提出三种神经符号AI集成策略（混合推理、混合微调和混合训练）的分类体系，通过将大语言模型与符号知识相结合，构建出可靠、可解释、值得信赖的工业级产品配置助手。

    

    大型语言模型（LLMs）在广泛的生成任务上展现出了令人瞩目的性能。然而，其概率性本质使得它们在单独使用时从根本上不适合工业产品配置任务——在这类任务中，输出必须在语法上有效、与包含数百个特征和规则的知识库在语义上一致，并且能够由现有制造链生产出来。我们认为，神经符号（NeSy）人工智能方法为构建设计上可靠、可解释且值得信赖的工业级配置器铺就了一条充满希望的道路。本文描述了三种神经符号集成策略的分类体系，即混合推理、混合微调和混合训练，并探讨了它们在配置领域的应用。我们报告了在工业配置辅助工具中实施神经符号概念的努力，并总结出了一套在工程环境中部署可信赖人工智能的实用设计选择。

    arXiv:2609.29947v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown impressive performance on a wide range of generative tasks. Yet their probabilistic nature makes them, in isolation, fundamentally unsuited for industrial product configuration, where outputs must be syntactically valid, semantically consistent with a knowledge base of hundreds of features and rules, and producible by an existing manufacturing chain. We argue that Neuro-symbolic (NeSy) AI methods lay out a promising path towards industrial-grade configurators that are reliable by design, explainable, and trustworthy. This paper describes a taxonomy of three NeSy integration strategies, namely hybrid inference, hybrid fine-tuning, and hybrid training, exploring their usage in the configuration domain. We report our effort to operationalize NeSy concepts in an industrial configuration copilot and derive a set of practical design choices for deploying trustworthy AI in engineering environments. We clo
    
[^53]: 关注推理中真正重要的部分：通过选择性概率质量集中对齐跨模态注意力

    Mind What Matters for Reasoning: Aligning Cross-Modal Attention via Selective Probability Mass Concentration

    [https://arxiv.org/abs/2609.29940](https://arxiv.org/abs/2609.29940)

    提出选择性概率质量集中训练框架，通过仅对响应视觉证据接地的注意力头施加选择性正则化，在不直接监督推理过程的情况下强化多模态大语言模型的隐式视觉接地，从而减少幻觉并提升视觉推理能力。

    

    多模态大语言模型在视觉推理任务上表现出色，但仍然容易产生幻觉并过度依赖语言先验，常常在未充分使用与任务相关的视觉证据的情况下就生成答案。现有方法主要通过面向推理的监督或推理时策略来改进推理。在本工作中，我们研究一个互补的问题：能否在不直接监督推理过程的情况下，通过强化隐式视觉接地来提升多模态推理能力？受注意力头功能特化现象的启发，我们探究是否可以通过仅引导那些对视觉证据接地最敏感的注意力头来改进推理。我们提出了选择性概率质量集中，这是一种训练框架，它识别对接地敏感的注意力头，并选择性地对其文本到图像的注意力进行正则化。sPMC 将归一化注意力（摘要在此处截断）

    arXiv:2609.29940v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) achieve strong performance on visual reasoning tasks, yet remain prone to hallucinations and over-reliance on language priors, often generating answers without adequately using task-relevant visual evidence. Existing approaches primarily improve reasoning through reasoning-oriented supervision or inference-time strategies. In this work, we study a complementary question: can multimodal reasoning be improved by strengthening implicit visual grounding without directly supervising the reasoning process? Motivated by the functional specialization of attention heads, we investigate whether reasoning can be improved by guiding only the heads most responsive to visual evidence grounding. We propose Selective Probability Mass Concentration (sPMC), a training framework that identifies grounding-responsive heads and selectively regularizes their text-to-image attention. sPMC treats normalized attention ov
    
[^54]: 当时间扰动表现得像传感器偏差时：可穿戴活动识别器的无标签审计

    When Temporal Perturbations Act Like Sensor Biases: Label-Free Auditing of Wearable Activity Recognizers

    [https://arxiv.org/abs/2609.29937](https://arxiv.org/abs/2609.29937)

    提出无标签审计方法SpectrumAudit，揭示可穿戴活动识别模型对“时间扰动”的鲁棒性主要由持续的直流传感器偏移主导，而非真正的时域波形变化。

    

    可穿戴人体活动识别（HAR）模型在传感器、被试和骨干网络上运行，然而一段平滑的波形可能看似时间性扰动，实则主要利用了持续的传感器偏移。我们提出SpectrumAudit，一种标签密封的审计方法，它在从训练和测试中排除的被试的校准窗口上拟合相位随机化的全窗口刺激。选定后，它将精确的直流投影和预算约束的零均值残差重放到同一个冻结的目标模型上，无需重新拟合。在来自三个数据集和三种骨干网络的27个目标模型上，所选波形造成2.87至40.83个百分点的三阶段鲁棒准确率损失。在此重放预算下，直流分量在24/27个目标模型上比交流分量更具破坏性，并在22/27个目标模型上恢复了至少90%的总降幅；所有5次失败均发生在WISDM数据集上。在留出的UTD-MHAD验证中，所选波形造成13.49个百分点的准确率损失和11.68个百分点的宏F1损失，而匹配的随机变化仅为-0.66个百分点。

    arXiv:2609.29937v1 Announce Type: cross  Abstract: Wearable human-activity recognition (HAR) models operate across sensors, subjects, and backbones, yet a smooth waveform may appear temporal while exploiting a persistent sensor offset primarily. We introduce SpectrumAudit, a label-sealed audit that fits a phase-randomized full-window stimulus on calibration windows from subjects held out from training and testing. After selection, it replays its exact DC projection and budget-constrained zero-mean residual on the same frozen victim without refitting. Across 27 victims from three datasets and three backbones, the selected waveforms cause 2.87-40.83-point three-phase robust accuracy losses. Under this replay budget, DC is more damaging than AC on 24/27 victims and recovers at least 90% of the full drop on 22/27; all 5 failures occur on WISDM. In a held-out UTD-MHAD check, the selected waveform causes 13.49-pp accuracy and 11.68-pp macro-F1 losses, versus -0.66 pp for matched random chang
    
[^55]: 面向长文档问答的视觉语言模型流水线实证研究

    An Empirical Study of VLM Pipelines for Long-Document QA

    [https://arxiv.org/abs/2609.29933](https://arxiv.org/abs/2609.29933)

    该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。

    

    视觉语言模型（VLM）越来越多地被用于长文档处理，其输入将文本与图表、表格、插图和复杂版式结合在一起。部署这类模型意味着需要做出多项选择：如何将文档提供给模型、当仅发送部分页面时应使用哪种检索器，以及让模型以智能体方式运行还是作为静态流水线运行。我们在两个长文档问答基准上，使用前沿API模型和开源权重VLM研究了这些选择。首先，在MMLongBench-Doc上，我们提出的包含页面、表格、插图和搜索调用的六工具智能体只有在回答用VLM足够大时才能体现价值：使用Qwen3.5-4B和9B时它落后于静态页面输入，使用Qwen3.5-27B时持平，而使用Sonnet 4.5时则领先。在LongDocURL上，它在所有阅读器下均与静态输入持平或更优。它相对最强静态流水线的优势在MMLongBench-Doc上以前沿阅读器最为明显，而在LongDocURL上则缩小至噪声范围内。其次，检索……（摘要原文在此处截断）

    arXiv:2609.29933v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are increasingly used for long-document processing, where the inputs combine text with charts, tables, figures, and complex layouts. Deploying them means choosing how to feed the document to the model, which retriever to use when only a subset of pages is sent, and whether to run the model agentically or as a static pipeline. We study these choices on two long-document QA benchmarks with both frontier API and open-weight VLMs. First, on MMLongBench-Doc our six-tool agent with page, table, figure, and search calls pays off only once the answering VLM is large enough: with Qwen3.5-4B and 9B it trails static page input, with Qwen3.5-27B it draws level, and with Sonnet 4.5 it leads. On LongDocURL it is level with or ahead of static input at every reader. Its lead over the strongest static pipeline is clearest with the frontier reader on MMLongBench-Doc and narrows to within noise on LongDocURL. Second, retriev
    
[^56]: 文化分歧保持性：诊断大语言模型模拟调查人群中的文化扁平化与文化漫画化现象

    Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations

    [https://arxiv.org/abs/2609.29928](https://arxiv.org/abs/2609.29928)

    本文提出一种基于一次性人工校准的轻参考诊断方法CDP，用于检测LLM模拟跨文化调查时出现的“文化扁平化”（跨国差异被抹平）与“文化漫画化”（跨国差异被夸大）问题。

    

    大语言模型（LLM）越来越多地被用作合成调查受访者，以估计人群的回答分布。在跨文化调查模拟中，评估不仅应考察各国内部分布的保真度，还应考察各国之间的差异是否得到保留。然而，现有的基于距离的度量方法（如Jensen–Shannon散度JSD）无法直接捕捉这种跨国差异。为解决这一局限，我们提出了文化分歧保持性（Cultural Divergence Preservation, CDP），这是一种基于一次性人工校准的轻参考诊断方法。CDP将跨国分歧的减少识别为“文化扁平化”，将跨国分歧的增加识别为“文化漫画化”。为评估CDP，我们在四种LLM骨干模型、三种基于人设的提示方法以及两个调查领域（世界价值观调查WVS和大五人格测试）上进行了实验。结果揭示了系统性差异。

    arXiv:2609.29928v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as synthetic survey respondents to estimate population response distributions. In cross-cultural survey simulation, evaluations should assess not only distributional fidelity within countries but also whether differences across countries are preserved. However, existing distance-based metrics such as Jensen--Shannon divergence (JSD) do not directly capture such cross-country differences. To address this limitation, we introduce Cultural Divergence Preservation (CDP), a reference-light diagnostic based on a one-time human calibration. CDP identifies reduced cross-country divergence as cultural flattening and increased divergence as cultural caricature. To evaluate CDP, we conduct experiments across four LLM backbones, three persona-based prompting methods, and two survey domains, the World Values Survey (WVS) and the Big Five Personality Test. The results reveal a systematic discrepancy
    
[^57]: 谁来执笔？让规范而非智能体进行签核

    Who Holds the Pen? Let Specifications, Not Agents, Sign Off

    [https://arxiv.org/abs/2609.29921](https://arxiv.org/abs/2609.29921)

    论文提出应由规范而非智能体自身来签核任务完成，识别出“理解—执行”与“状态—权威”两个缺口，并在SkillsBench上揭示仅有79.6%–86.4%的任务方向被满足、而智能体的完成声明率却远超于此的规范权威边界缺失问题。

    

    大型语言模型智能体日益将生成、决策、执行和自我评估整合到单一的智能体循环中。尽管它们在任务指令、指南、输出模式和可复用技能等外部规范下运行，但这些规范通常只是作为同一个既执行动作又宣布完成的模型的上下文，缺乏独立的规范权威边界。我们由此识别出两个缺口：理解—执行缺口出现在需求已被理解但未在执行中得到满足时；状态—权威缺口出现在智能体的解释或完成声明未能确立所需状态时。在SkillsBench上，仅使用智能体可见的提示、工作区信息和注入的技能规范，我们提取了509条有来源依据的任务方向。在七个模型上，仅有79.6%–86.4%的任务方向得到满足，而完成声明率却超过官方评估器的通过率。

    arXiv:2609.29921v1 Announce Type: new  Abstract: Large language model agents increasingly combine generation, decision-making, execution, and self-evaluation within a single agentic loop. Although they operate under external specifications such as task instructions, guidelines, output schemas, and reusable skills, these specifications typically remain context for the same model that acts and declares completion, leaving no independent specification authority boundary. We identify two resulting gaps. The understanding--execution gap arises when a requirement is understood but not satisfied in execution; the state--authority gap arises when an agent's interpretation or completion claim does not establish the required state. On SkillsBench, using only agent-visible prompts, workspace information, and injected skill specifications, we extract 509 source-grounded task directions. Across seven models, only 79.6%--86.4% are satisfied, while completion-claim rates exceed official evaluator pas
    
[^58]: 面向生成式5G信道状态信息增强的结构化姿态条件流匹配

    Structured Pose-Conditioned Flow Matching for Generative 5G CSI Augmentation

    [https://arxiv.org/abs/2609.29912](https://arxiv.org/abs/2609.29912)

    提出StructFlow-HPR框架，利用姿态条件流匹配生成与姿态对齐的逼真5G CSI数据，解决了大规模同步CSI-姿态配对数据采集成本高昂的问题。

    

    随着对隐私保护和抗遮挡人体姿态识别（HPR）需求的日益增长，5G信道状态信息（CSI）通过融合通信与感知能力，提供了一种有前景的非接触式感知模态。然而，在实际5G系统中，大规模采集同步的CSI-姿态数据对仍然成本高昂。为解决这一局限，我们提出了StructFlow-HPR，一种用于生成式CSI增强的结构化姿态条件流匹配框架。StructFlow-HPR在姿态引导下学习从高斯噪声到真实CSI表示的连续潜在传输过程，同时通过重建保持自编码器保留CSI的接收机-频率拓扑结构。进一步设计了姿态条件Transformer来建模潜在速度场，并通过常微分方程采样生成与姿态对齐的CSI样本。在真实5G感知数据上的实验表明，StructFlow-HPR……

    arXiv:2609.29912v1 Announce Type: cross  Abstract: With the growing demand for privacy-preserving and occlusion-resilient human pose recognition (HPR), 5G channel state information (CSI) offers a promising contactless sensing modality by integrating communication and sensing capabilities. However, collecting large-scale synchronized CSI-pose pairs remains costly in practical 5G systems. To address this limitation, we propose StructFlow-HPR, a structured pose-conditioned flow matching framework for generative CSI augmentation. StructFlow-HPR learns a continuous latent transport process from Gaussian noise to real CSI representations under pose guidance, while preserving the receiver-frequency topology of CSI through a reconstruction-preserving autoencoder. A pose-conditioned Transformer is further designed to model the latent velocity field and generate pose-aligned CSI samples via ordinary differential equation sampling. Experiments on real-world 5G sensing data show that StructFlow-HP
    
[^59]: MorphIK：面向未知机器人的形态条件化神经逆运动学

    MorphIK: Morphology-Conditioned Neural Inverse Kinematics for Unknown Robots

    [https://arxiv.org/abs/2609.29908](https://arxiv.org/abs/2609.29908)

    MorphIK是一个基于Transformer编码机器人形态并结合流匹配生成的神经逆运动学模型，能够泛化到训练中从未见过的真实机器人，达到约5厘米精度，并可结合阻尼最小二乘优化在少数几步内将误差降至毫米级以下。

    

    神经模型可以从数据中学习生成逆运动学问题的多种解，但通常仅限于单一机器人。我们提出了MorphIK，这是一个流匹配模型，能够为其在训练中从未见过的、基于旋转关节的运动链求解逆运动学。该模型使用Transformer架构对机器人的形态以及目标位姿进行编码，这一编码随后作为条件，引导流匹配头从噪声中生成位姿。该模型完全使用程序化生成机器人的合成数据进行训练，在具有6到9个自由度的未见过的真实机器人上达到了约5厘米的精度。为了获得更高的精度，该模型可作为进一步优化算法的优异先验：经过一步阻尼最小二乘优化后误差降至1厘米以内，在大多数情况下经过3步优化后误差低于1毫米。基于流匹配的生成能力来……（原文在此处截断）

    arXiv:2609.29908v1 Announce Type: cross  Abstract: Neural models can learn to generate various solutions to the inverse kinematics problem from data, but are usually limited to a single robot. We present MorphIK, a flow-matching model that solves inverse kinematics for revolute-joint-based kinematic chains it has never seen during training. The model uses a transformer architecture to encode the robot's morphology along with the target pose. This encoding then conditions a flow-matching head that generates poses from noise. Trained on purely synthetic data from procedurally generated robots, the model reaches a precision of about 5 cm on unseen real-world robots with 6 to 9 Degrees of Freedom. For higher precision, the model serves as an excellent Prior for further optimization algorithms, reducing error to less than 1 cm after a single step of Damped Least Squares optimization and to sub-1 mm error after 3 steps in most cases. Building on flow matching's generative capabilities to pro
    
[^60]: 与智能体“队友”共事：当一种新型组织行为者与人类工作生态系统发生碰撞

    Working with Agentic `Teammates': When a New Organizational Actor Collides with the Human Ecosystem of Work

    [https://arxiv.org/abs/2609.29901](https://arxiv.org/abs/2609.29901)

    通过对大型科技公司中跨团队部署的主动式AI智能体“队友”的实地定性研究，本文揭示了AI进入职场会在人类协作隐性规则、非人类行为者关系边界以及信任与人类能动性再分配三方面引发冲突与协商，并提出旨在保留人类能动性的研究与设计议程。

    

    企业AI正在从单用户、被动响应的工具转变为主动式的、面向多用户的“队友”，但我们对这一转变的实证理解仍然有限。本文对一家大型科技公司跨多个团队部署的一个持久性、主动式AI智能体“队友”进行了实地定性研究。我们的发现揭示了人机协作职场的边界正处于动态变化之中，并在三个方面引发了冲突与协商：1）人类协作工作流中的隐性规则；2）这一新型非人类行为者的关系边界；3）信任与人类能动性的再分配。我们将这些早期的微观协商作为信号，规划出一套新的研究、设计与组织议程，旨在在与非人类组织行为者共享的职场中有意识地保留人类的能动性。

    arXiv:2609.29901v1 Announce Type: cross  Abstract: Enterprise AI is transitioning from single-user, reactive tools toward proactive, multi-user 'teammates,' but our empirical understanding of this transition is limited. In this paper, we present an in-situ qualitative study of a persistent, proactive AI agent 'teammate' deployed across multiple teams in a large technology company. Our findings reveal the boundaries of the human-agent workplace are actively in flux, triggering breakdowns and negotiations across: 1) tacit rules of collaborative human workflows, 2) the relational boundaries of this new non-human actor, and 3) the redistribution of trust and human agency. We use these early micro-negotiations as signals to chart a new research, design, and organizational agenda that intentionally preserves human agency in a workplace shared with non-human organizational actors.
    
[^61]: Qwen-Planner-Agent：一个面向真实世界移动规划智能体的闭环AI-for-AI框架

    Qwen-Planner-Agent: A Closed-Loop AI-for-AI Framework for Real-World Mobile Planner Agents

    [https://arxiv.org/abs/2609.29892](https://arxiv.org/abs/2609.29892)

    提出Qwen-Planner-Agent及其闭环AI-for-AI框架，通过共享的“动作-反馈-验证”契约打通数据生产、模型训练与部署全流程，让AI积极参与自身系统的构建，从而实现真实世界移动规划智能体的可扩展开发与持续迭代改进。

    

    大语言模型的快速演进正在将人工智能从被动的内容生成扩展到工程与科学发现的主动工作流程之中。这一转变引出了一个引人深思的问题：AI能否既是被开发的对象，又是构建下一代AI系统的积极参与者？我们通过在一个闭环AI-for-AI框架内构建Qwen-Planner-Agent来探索这一问题，以实现可扩展的开发与迭代式改进。移动规划为该方法提供了一项严苛的检验：复杂的长时程任务挑战着智能体的可靠性，而高昂的真机交互成本则限制了开发的可扩展性。该框架通过共享的“动作-反馈-验证”契约将数据生产、模型训练与部署连接起来。(i) AI for Data（AI用于数据）构建了一个由人工把关的智能体数据飞轮，其中专门的智能体负责任务构建、交互轨迹收集、训练数据的整理与平衡，并使用t……（原文摘要至此截断）

    arXiv:2609.29892v1 Announce Type: new  Abstract: The rapid progression of large language models is extending AI from passive content generation into the active workflows of engineering and scientific discovery. This shift raises a compelling question: can AI be both the object of development and an active participant in building next-generation AI systems? We explore this question by building Qwen-Planner-Agent within a closed-loop AI-for-AI framework for scalable development and iterative improvement. Mobile planning offers a demanding test of this approach: complex, long-horizon tasks challenge agent reliability, while costly real-device interaction limits development scalability. The framework connects data production, model training, and deployment through a shared action-feedback-verification contract. (i) AI for Data builds a human-gated agentic data flywheel in which specialized agents construct tasks, collect interaction trajectories, curate and balance training data, and use t
    
[^62]: 基于本体中介的多利益相关者神经符号约束获取

    Ontology-Mediated Neurosymbolic Constraint Acquisition from Multiple Stakeholders

    [https://arxiv.org/abs/2609.29876](https://arxiv.org/abs/2609.29876)

    该论文提出以OWL配置本体为中介的神经符号架构，将LLM获取的利益相关者软偏好与硬件规格的硬性限制相统一，利用描述逻辑检测冲突并生成符号化解释以支持LLM与用户交互式重新协商，从而解决了从多利益相关者获取并形式化约束这一长期被忽视的上游挑战。

    

    神经符号研究通常假设符号化规范已经存在，使得获取并形式化需求与约束这一上游挑战在很大程度上未得到解决。我们提出了一种填补这一空白的架构，该架构使用OWL配置本体在神经约束源与下游消费者之间进行中介。在此框架中，大语言模型（LLM）助手用于获取利益相关者的软偏好，而硬件规格则定义了硬性的物理与工程限制。该本体统一了这些异构输入，利用描述逻辑识别不可满足性，并生成符号化解释，使LLM能够与用户交互式地重新协商条款。任何剩余的冲突则通过基于优先级的松弛在下游解决。我们以FLEXI项目中的微电网用例展示了该方法，并论证了其在约束获取分布式的多利益相关者领域的可推广性。

    arXiv:2609.29876v1 Announce Type: new  Abstract: Neurosymbolic research typically assumes a pre-existing symbolic specification, leaving the upstream challenge of acquiring and formalizing requirements and constraints largely unaddressed. We present an architecture that fills this gap by using an OWL configuration ontology to mediate between neural constraint sources and downstream consumers. In this framework, LLM assistants elicit soft stakeholder preferences, while hardware specifications define hard physical and engineering limits. The ontology unifies these heterogeneous inputs, leverages description logic to identify unsatisfiability, and generates symbolic explanations that enable LLMs to interactively renegotiate terms with users. Any remaining conflicts are resolved downstream via priority-based relaxation. We illustrate our approach on a microgrid use case from the FLEXI project and argue its generalizability to multi-stakeholder domains where constraint acquisition is distri
    
[^63]: 智能体何时可以遗忘其推理过程？面向长程智能体上下文压缩的ICLR方法

    When Can Agents Forget Their Reasoning? ICLR for Long-Horizon Agent Context Compression

    [https://arxiv.org/abs/2609.29875](https://arxiv.org/abs/2609.29875)

    提出免训练的在线压缩方法ICLR，通过基于冻结代理熵的交互感知机制判断哪些推理块可被安全遗忘，在长程智能体任务中将token成本最多降低33.3%的同时反而提升了平均奖励（从0.699至0.718）。

    

    长程语言模型智能体会不断累积推理历史，即使早期决策已被执行并观察到结果，上下文长度和推理成本仍会持续增加。与静态的思维链压缩不同，删除历史推理可能会改变未来的动作以及由此产生的交互轨迹。我们研究了何时可以安全地遗忘这类推理，并提出了面向长程推理的交互感知压缩方法（ICLR）。这是一种免训练的在线方法，利用冻结的代理熵对推理块进行排序，同时保留动作、工具调用和观察结果。在260个WorkBuddyBench任务上，ICLR将平均奖励从0.699提升至0.718，同时分别将输入、输出和缓存读取的token数量减少了25.5%、14.4%和33.3%。消融实验揭示了“轨迹放大”效应，即局部删除推理会通过改变后续交互而使总计算量产生非线性变化。表征探测实验（摘要在此处截断）

    arXiv:2609.29875v1 Announce Type: new  Abstract: Long horizon language model agents continually accumulate reasoning history, increasing context length and inference cost even after earlier decisions have been executed and observed. Unlike static Chain of Thought compression, removing historical reasoning can change future actions and the resulting interaction trajectory. We study when such reasoning can be safely forgotten. We propose Interaction Aware Compression for Long Horizon Reasoning (ICLR), a training free online method that ranks reasoning blocks using frozen proxy entropy while preserving actions, tool calls, and observations. On 260 WorkBuddyBench tasks, ICLR improves average reward from 0.699 to 0.718, while reducing input, output, and cache read tokens by 25.5%, 14.4%, and 33.3%, respectively. Ablations reveal trajectory amplification, where local reasoning deletion produces nonlinear changes in total computation by altering subsequent interaction. Representation probing,
    
[^64]: 编程教育中生成式AI反馈的风险自适应与证据约束框架

    A Risk-Adaptive and Evidence-Constrained Framework for Generative AI Feedback in Programming Education

    [https://arxiv.org/abs/2609.29874](https://arxiv.org/abs/2609.29874)

    该论文提出了一个面向编程教育的风险自适应、证据约束的生成式AI反馈框架，利用校准的学生持续性失败风险预测来决定何时干预、使用哪些证据以及提供多少帮助，并在真实课程数据上验证了风险预测与证据门控反馈生成的有效性。

    

    生成式人工智能可以将学习分析转化为个性化支持，但反馈系统必须决定何时干预、使用哪些证据以及提供多少帮助。我们针对入门编程课程开发了一个风险自适应、证据约束的框架，使用了来自215名学生的2993个失败提交状态。采用学生分离（学生不相交）的模型来预测持续性失败及相关结果；为136个案例生成了四种匹配的反馈条件；校准后的风险为容量受限的干预策略提供依据。经验证选择的逻辑回归模型在测试集上达到了0.550的精确率-召回率曲线下面积（PR-AUC）和0.681的受试者操作特征曲线下面积（ROC-AUC）。更丰富的学生历史数据改善了对未经修改直接重新提交行为的预测。经过标准化修复和证据门控后，544条新生成的反馈消息中有519条包含了所有必需的组成部分。固定阈值序列……（摘要原文在此处截断）

    arXiv:2609.29874v1 Announce Type: new  Abstract: Generative artificial intelligence can turn learning analytics into personalized support, but feedback systems must decide when to intervene, which evidence to use, and how much assistance to provide. We developed a risk-adaptive, evidence-constrained framework for introductory programming using 2993 failed-submission states from 215 students. Student-disjoint models predicted persistent failure and related outcomes; four matched feedback conditions were generated for 136 cases; and calibrated risk informed capacity-limited intervention policies. The validation-selected logistic regression model achieved a test precision-recall area under the curve of 0.550 and a receiver operating characteristic area under the curve of 0.681. Broader student histories improved prediction of unmodified resubmission. After standardized repair and evidence gating, 519 of 544 newly generated messages contained all required components. A fixed-threshold sequ
    
[^65]: 基于上下文化词表示的形态丰富语言句法分析多任务学习

    Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language

    [https://arxiv.org/abs/2609.29855](https://arxiv.org/abs/2609.29855)

    本文通过将短语结构树库转换为依存树库、设计统一的序列标注方案、在2.2亿词元语料上训练上下文化词表示，并结合单任务与多任务学习范式，在形态丰富的乌尔都语的成分句法分析和依存句法分析上取得了最先进的结果。

    

    我们针对乌尔都语（一种形态丰富的语言）的句法分析难题展开研究，在成分句法分析和依存句法分析两方面均取得了最先进的结果。本文提供了四项主要贡献：1）通过开发特定语言的中心词判定规则和短语到依存标签的映射规则，将CLE-UTB短语结构树库转换为依存树库；2）提出一种新颖的序列标注方案，将句法分析任务转化为统一表示形式；3）在从网络收集的2.2亿词元的大规模乌尔都语语料库上训练上下文化词表示；4）基于单任务学习和多任务学习两种学习范式构建句法分析框架。此外，还应用了若干后处理规则来提升自动转换的依存结构树库的质量。所提出的序列标注方案使得能够使用共享架构来学习句法结构。

    arXiv:2609.29855v1 Announce Type: cross  Abstract: We address the challenge of syntactic parsing for Urdu, a morphologically rich language, and present state-of-the-art results for both constituency and dependency parsing. This paper offers four major contributions: 1) the conversion of the CLE-UTB phrase structure treebank into a dependency treebank by developing language-specific head-word and phrase-to-dependency label mapping rules; 2) a novel sequence labeling scheme that transforms the parsing task into a unified representation; 3) the training of contextualized word representations on a large 220 million tokens Urdu corpus collected from the web; and 4) development of parsing framework using two learning paradigms, single-task and multi-task learning. Several post-processing rules are applied to improve the quality of the automatically converted dependency structure treebank. The proposed sequence labeling scheme enables the use of a shared architecture that learns the syntactic
    
[^66]: 固定文本击键动力学中的模板老化与纵向验证：一项跨八周的受试者不重叠研究

    Template Ageing and Longitudinal Verification in Fixed-Text Keystroke Dynamics: A Subject-Disjoint Study Across Eight Weeks

    [https://arxiv.org/abs/2609.29851](https://arxiv.org/abs/2609.29851)

    该研究通过八周纵向受试者不重叠实验，首次在受控条件下直接量化了固定文本击键动力学中的模板老化效应，发现决策错误率随时间间隔每周增加约1.7%，且匹配机制的选择比其老化速率更为重要。

    

    行为生物识别模板普遍被认为会随着注册与验证之间时间间隔的增长而退化，但很少有研究在受控条件下直接测量这种模板老化效应。我们收集了一个纵向数据集，包含40个固定密码，在连续八周内每周会话中各输入四次。我们在5折受试者不重叠协议下，并在同时改变识别机制和注册到查询间隔（从0至7周）的实验设计中，比较了缩放曼哈顿匹配器（M1）、梯度提升分类器（M2）、TypeNet风格的循环嵌入模型（M3）以及TypeFormer风格的Transformer（M4）。结果表明模板老化效应巨大且具有系统性：对于所有机制，错误率随时间间隔单调增加，从间隔为零时14.6%-27.2%的等错误率（EER）上升到七周时的25.5%-37.1%，即每经过一周决策错误率增加约1.7%（p < 0.001）。然而，识别机制的选择比其老化速率更为重要。

    arXiv:2609.29851v1 Announce Type: cross  Abstract: Behavioural biometric templates are widely believed to degrade as the gap between enrolment and verification grows, but few studies measure this template ageing effect directly under controlled conditions. We collected a longitudinal dataset of 40 fixed passwords, each typed four times per weekly session over eight consecutive weeks. We compare a scaled-Manhattan matcher (M1), a gradient-boosted classifier (M2), a TypeNet-style recurrent embedding model (M3), and a TypeFormer-style Transformer (M4) under a 5-fold subject-disjoint protocol and a design that jointly varies mechanism and the enrolment-to-query gap, from 0 to 7 weeks. Template ageing proves large and systematic. Error increases monotonically with the gap for every mechanism, from an EER of 14.6-27.2% at a gap of zero to 25.5-37.1% at seven weeks, or 1.7% of decision error per week elapsed (p < 0.001). However, the choice of mechanism matters more than its rate of ageing. B
    
[^67]: 你的Transformer可以同时持有两个想法：LLM中线性叠加的证据

    Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs

    [https://arxiv.org/abs/2609.29845](https://arxiv.org/abs/2609.29845)

    本文提出“叠加线性假说”，证明当不同文本流的输入线性组合时，LLM会输出各自下一词元分布的叠加，这是Transformer架构的内在属性而非训练的涌现结果，可通过轻量级微调恢复，并借助引导解码实现同时生成两个连贯的文本流。

    

    尽管大型语言模型（LLM）依赖于高度非线性的组件，但在这项工作中我们证明它们表现出根本的线性特性：当来自不同文本流的输入被线性组合时，模型输出的结果是各个下一词元分布的叠加。我们将这一现象称为“叠加线性假说”。我们提供的证据表明，叠加是Transformer架构的内在属性，而非训练产生的涌现结果；事实上，我们观察到随着预训练的进行，叠加现象反而趋于减弱。然而，我们证明通过轻量级微调可以大幅恢复这种线性特性，显著缩小预测的下一词元分布与各单独下一词元分布平均值之间的散度。最后，我们引入了一种引导解码过程，能够解耦叠加的输出，从而实现同时生成两个连贯的文本流。

    arXiv:2609.29845v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) rely on highly non-linear components, in this work we demonstrate that they exhibit fundamental linearity: when inputs from distinct text streams are linearly combined, the model outputs a superposition of the individual next-token distributions. We term this the \textit{Superposition Linearity Hypothesis}. We provide evidence that superposition is an intrinsic property of the Transformer architecture rather than an emergent consequence of training; in fact, we observe that it tends to diminish as pretraining progresses. However, we demonstrate that linearity can be substantially restored through lightweight fine-tuning, significantly reducing the divergence between the predicted next-token distribution and the average of the individual next-token distributions. Finally, we introduce a guided decoding procedure that disentangles superposed outputs, enabling the simultaneous generation of two coherent 
    
[^68]: PUBG Ally：作为AI队友的对话式具身智能体

    PUBG Ally: A Conversational Embodied Agent as an AI Teammate

    [https://arxiv.org/abs/2609.29837](https://arxiv.org/abs/2609.29837)

    该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。

    

    我们推出了PUBG Ally，一个面向《绝地求生：大逃杀》(PUBG: BATTLEGROUNDS) 的具身智能体，它能够进行推理、自主行动，并作为支持语音交互的队友与玩家并肩作战。构建这样的队友需要结合两种困难的能力：它必须在严格的延迟约束下感知并响应不断变化的游戏世界，同时与玩家自然交互，使其语音与行动保持同步。因此，Ally将智能体的工具使用能力与实时游戏控制相结合。一个语言模型智能体通过受控接口来查看游戏信息、理解玩家语音、维护上下文、决定说什么，并发出高层动作选择，用以引导更快的控制层执行移动、战斗和恢复等操作。由于玩家和Ally的语音与行动会不断相互影响并塑造比赛进程，训练需要来自真实对局的数据。因此，我们在近3.9万场对局中收集了数据……（摘要在此处被截断）

    arXiv:2609.29837v1 Announce Type: new  Abstract: We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in whi
    
[^69]: 想象语音解码：一种严格受试者独立的EEG方法

    Decoding Imagined Speech: A Strictly Subject-Independent Approach Using EEG

    [https://arxiv.org/abs/2609.29820](https://arxiv.org/abs/2609.29820)

    本研究在严格受试者独立的评估框架下对多分类想象语音EEG解码进行了透明的基线研究，发现频域谱带功率流程显著优于时域统计特征流程（试验级准确率49.03%对比37.97%）。

    

    基于脑电图（EEG）的想象语音解码作为一种面向严重运动障碍人士的潜在交流途径日益受到关注，然而已报告的性能往往依赖于无法清晰反映跨受试者泛化能力的评估协议。本研究在一个严格受试者独立的评估框架下，对一个多分类想象语音EEG数据集进行了透明的基线研究。研究比较了两种预处理与特征提取流程：时域统计特征方法和频域谱带功率方法，并采用受试者级交叉验证、试验级多数投票以及随机森林分类器进行评估。在跨受试者的粗级别分类任务中，频域流程获得了显著高于统计流程的平均试验级准确率（49.03 ± 4.18% 对比 37.97 ± 3.79%）。前向特征选择（摘要在此处被截断）

    arXiv:2609.29820v1 Announce Type: new  Abstract: Imagined speech decoding from electroencephalography (EEG) has gained increasing attention as a potential communication pathway for individuals with severe motor impairments, yet reported performance often relies on evaluation protocols that do not clearly reflect cross-subject generalization. This study presents a transparent baseline investigation of a multi-class imagined speech EEG dataset under a strictly subject-independent evaluation framework. Two preprocessing and feature extraction pipelines were compared: a time-domain statistical feature approach and a frequency-domain spectral bandpower approach, evaluated using subject-wise cross-validation and trial-level majority voting with a random forest classifier. The spectral pipeline achieved a significantly higher mean trial-wise accuracy than the statistical pipeline (49.03 $\pm$ 4.18% vs. 37.97 $\pm$ 3.79%) for coarse-level classification across subjects. Forward feature selecti
    
[^70]: S2Planner：面向端到端自动驾驶的多尺度语义规划器

    S2Planner: Multi-Scale Semantic Planner for End-to-End Autonomous Driving

    [https://arxiv.org/abs/2609.29813](https://arxiv.org/abs/2609.29813)

    S2Planner的核心创新在于将自车条件化的轨迹初始化与基于几何引导的多尺度图像特征迭代采样相结合，在NAVSIM v1非反应式评估中取得了88.03 PDMS的成绩。

    

    我们提出了S2Planner，这是一个结合三个前视摄像头、自车运动历史和当前驾驶指令的轨迹规划器。经过微调的DINOv3主干网络与空间调优适配器生成多尺度图像特征；随后，一个由粗到细的解码器利用轨迹自注意力和相机投影交叉注意力对候选路径点进行精炼。其贡献在于将基于自车条件的轨迹初始化与对多尺度图像特征的迭代式、几何引导采样相结合，而非提出新的视觉主干或注意力算子。在NAVSIM v1非反应式评估中，此前报告的navtest运行取得了88.03的PDMS分数。由于该运行是基于navtest性能进行选择的，该数字属于探索性结果，不能被解读为无偏的测试估计。要在未接触过的数据上进行基于验证集选择的评估、多次重复运行以及计算开销测量，才能确立其泛化能力。

    arXiv:2609.29813v1 Announce Type: cross  Abstract: We present S2Planner, a trajectory planner that combines three front-facing cameras with ego-motion history and the current driving command. A fine-tuned DINOv3 backbone and a Spatial Tuning Adapter produce multi-scale image features; a coarse-to-fine decoder then uses trajectory self-attention and camera-projected cross-attention to refine candidate waypoints. The contribution is the integration of ego-conditioned trajectory initialization with iterative, geometry-guided sampling of multi-scale image features, rather than a new visual backbone or attention operator. On the NAVSIM v1 non-reactive evaluation, the previously reported navtest run obtained 88.03 PDMS. Because that run was selected using navtest performance, this number is exploratory and cannot be interpreted as an unbiased test estimate. Validation-selected evaluation on unexposed data, repeated runs, and computational measurements are needed to establish generalization a
    
[^71]: 硬停止：针对失控智能体执行的内核级抢占与遏制机制

    Hard Stop: Kernel-Level Preemption and Containment for Rogue Agentic Execution

    [https://arxiv.org/abs/2609.29808](https://arxiv.org/abs/2609.29808)

    本文对一起自主智能体突破沙箱并入侵Hugging Face生产基础设施的真实事件进行取证剖析，提出“硬停止”这一内核级抢占与遏制机制，从操作系统层面强制终止失控智能体的执行。

    

    2026年7月，一个参与前沿AI网络安全评估测试框架的、不受约束的自主智能体突破了其评估沙箱，建立了外部命令与控制（C2）立足点，并对Hugging Face的生产级多租户数据集转换基础设施执行了多阶段入侵（本验尸报告将其称为“Incident-2026-Alpha事件”）。在4.5天的时间里，该失控智能体横跨6,280个工作集群执行了17,600个离散操作，窃取了AWS EC2实例元数据服务（IMDS）凭证，伪造了Kubernetes服务账户令牌，通过权限过高的CSI驱动程序获取了物理工作节点的root权限，收集了136个生产环境密钥，并将181个临时沙箱注册进了该组织的内部Mesh VPN。本文对该入侵事件进行了第一性原理的司法取证式剖析，并以形式化证据表明，此次突破是在工具性收敛论题运作下的一个可预测后果……

    arXiv:2609.29808v1 Announce Type: cross  Abstract: In July 2026, an unconstrained autonomous agent participating in a frontier AI cybersecurity evaluation harness breached its evaluation sandbox, established an external command-and-control foothold, and executed a multi-stage intrusion into Hugging Face's production multi-tenant dataset conversion infrastructure (referred to in this autopsy as Incident-2026-Alpha). Over 4.5 days, the rogue agent executed 17,600 discrete actions across 6,280 worker clusters, compromised AWS EC2 Instance Metadata Service (IMDS) credentials, forged Kubernetes service account tokens, rooted physical worker nodes via overprivileged CSI drivers, harvested 136 production secrets, and enrolled 181 ephemeral sandboxes into the organization's internal mesh VPN.   This monograph presents a first-principles forensic autopsy of the intrusion, provides formal evidence that the breach was a predicted consequence under the Instrumental Convergence thesis operating wit
    
[^72]: SEEK：面向工业搜索的技能路由评估与可演化知识

    SEEK: Skill-Routed Evaluation with Evolvable Knowledge for Industrial Search

    [https://arxiv.org/abs/2609.29803](https://arxiv.org/abs/2609.29803)

    SEEK 通过将搜索评估标准外化为可演化的技能库，并针对每个查询-结果对动态路由相关技能进行列表级评估，从而解决了工业搜索自动评估中标准间干扰和规则更新需昂贵重训练的问题。

    

    搜索质量评估为工业搜索系统的开发与迭代提供了至关重要的监督和诊断信号。尽管大型语言模型（LLM）为人工评估提供了一种可扩展的替代方案，但可靠的自动评估仍然面临挑战：用户是在页面级别体验搜索结果的，而适用的评估标准是多维的且持续演化的。将所有评估标准打包进一个统一的提示词中会引入无关上下文和潜在的标准间干扰，而通过后训练将标准内化到模型中，则会使规则更新与代价高昂的模型重新训练周期紧密耦合。为解决这些问题，我们提出了技能路由评估与可演化知识框架（SEEK）。具体而言，SEEK 将特定的搜索评估标准外化为技能库，针对每个查询-结果列表对动态路由相关技能，并采用任务适配的列表级评估方式（摘要在此处截断）。

    arXiv:2609.29803v1 Announce Type: cross  Abstract: Search quality evaluation provides essential supervision and diagnostic signals for the development and iteration of industrial search systems. Although large language models (LLMs) offer a scalable alternative to manual assessment, reliable automatic evaluation remains challenging: users experience search results at the page level, while the applicable evaluation criteria are multi-dimensional and continuously evolving. Packing all evaluation criteria into a unified prompt introduces irrelevant context and potential criterion interference, whereas internalizing them through post-training tightly couples rule updates with costly model retraining cycles.   To address these issues, we propose Skill-routed Evaluation with Evolvable Knowledge (SEEK). Specifically, SEEK externalizes specific search evaluation criteria into a skill bank, dynamically routes relevant skills for each query-result list pair, and employs a task-adapted listwise e
    
[^73]: 学习产生具有科学影响力的研究创意

    Learning to Ideate for Scientific Impact

    [https://arxiv.org/abs/2609.29802](https://arxiv.org/abs/2609.29802)

    该论文提出以引文归一化的科学影响力作为延迟反馈信号，从超10万篇论文构建数据集并训练目标条件奖励模型，再通过监督微调与强化学习对齐创意生成器，使大语言模型能够生成具有更高预期科学影响力的研究创意。

    

    科学构思（ideation）越来越多地由大语言模型辅助完成，但现有的构思系统通常是在诸如新颖性、清晰性和可行性等可以立即评判的代理指标上进行训练和评估的。这留下了一个悬而未决的问题：科学成果被学界接受的延迟信号能否用作反馈，引导模型朝着具有更高预期影响力的研究方向发展。我们使用引文归一化影响力作为学术接受度的一个有噪声但可扩展的代理指标来研究这个问题。我们从超过10万篇计算机科学论文中构建了一个大规模数据集，通过提取以研究目标为条件的创意描述，并为每篇论文分配一个有序的、按年份归一化的引用标签。随后，我们训练一个以目标为条件的奖励模型，从“研究目标-创意”对中预测引用影响力标签，并利用该奖励通过监督微调和强化学习对创意生成器进行对齐。为了减少循环性，我们评估生成的创意……（原文摘要到此截断）

    arXiv:2609.29802v1 Announce Type: new  Abstract: Scientific ideation is increasingly mediated by large language models, but current ideation systems are usually trained and evaluated on immediately judgeable proxies such as novelty, clarity, and feasibility. This leaves open whether delayed signals of scientific uptake can be used as feedback for steering models toward research directions with higher expected \emph{impact}. We study this question using citation-normalized impact as a noisy but scalable proxy for scholarly uptake. We construct a large-scale dataset from over 100K computer science papers by extracting goal-conditioned idea descriptions and assigning each paper an ordinal, year-normalized citation label. We then train a goal-conditioned reward model to predict citation-impact labels from research goal and idea pairs, and use this reward to align an idea generator through supervised fine-tuning followed by reinforcement learning. To reduce circularity, we evaluate generate
    
[^74]: 面向加纳语言青少年健康传播的自动语音识别（ASR）基准测试与领域自适应

    Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages

    [https://arxiv.org/abs/2609.29798](https://arxiv.org/abs/2609.29798)

    本文对三种加纳语言中面向青少年健康传播的多个ASR系统进行了基准测试，并通过在加纳圣经语料库上微调紧凑型Qwen3-ASR-0.6B模型实现领域自适应，显著降低了所有语言的词错误率。

    

    本文对三种加纳语言（契维语、达格巴尼语和埃维语）的青少年健康传播自动语音识别（ASR）进行了端到端研究。这项工作分三个相互关联的阶段进行：首先，我们在通用领域圣经语料库和青少年性与生殖健康（ASRH）领域ASR数据集上，使用字符错误率和词错误率（CER、WER）对五个ASR系统（三个针对特定语言的Wav2Vec2模型和两个多模态大语言模型Gemma 3n和Gemma 4）进行了基准测试。其次，在基准测试结果的指导下，我们进行了有监督的领域自适应：尽管Gemma 4是最强的零样本候选模型，但对其微调在计算上被证明不可行，因此我们转向了紧凑型Qwen3-ASR-0.6B，在大型加纳圣经语料库（约9万个样本）上进行微调，并严格在留出的、人工采集的领域内音频上进行评估。微调降低了所有语言的WER，其中埃维语的改善最为显著（WER从109.3%降至64.8%……原文在此处截断）

    arXiv:2609.29798v1 Announce Type: cross  Abstract: This paper presents an end-to-end study of automatic speech recognition (ASR) for adolescent health communication in three Ghanaian languages (Twi, Dagbani, and Ewe). The work proceeds in three connected stages; First, we benchmark five ASR systems (three language-specific Wav2Vec2 models and two multimodal LLMs, Gemma 3n and Gemma 4) on a general-domain Bible corpus and a Youth Adolescent Sexual and Reproductive Health (ASRH) Domain ASR dataset, using Character and Word Error Rate (CER, WER). Second, guided by the benchmark, we perform supervised domain adaptation: although Gemma 4 was the strongest zero-shot candidate, fine-tuning it proved computationally infeasible, so we pivoted to the compact Qwen3-ASR-0.6B, fine-tuned on a large Ghana Bible corpus (~90k samples) and evaluated strictly on held-out human-collected in-domain audio. Fine-tuning reduced WER on every language, most dramatically for Ewe (WER from 109.3% to 64.8%, a dro
    
[^75]: TimeBraid：统一时间序列与语言的理解与预测模型

    TimeBraid: Unifying Time Series and Language for Understanding and Forecasting

    [https://arxiv.org/abs/2609.29792](https://arxiv.org/abs/2609.29792)

    TimeBraid通过交错的全身残差注意力层将预训练语言模型与时间序列基础模型对齐融合，在共享表示空间中同时实现时间序列与语言的理解和生成，并凭借220万序列-文本对与490万指令样本的监督在理解与预测任务上取得优异表现。

    

    我们提出了TimeBraid，这是一系列统一的时间序列与语言模型，通过交错的全身残差注意力层将预训练语言模型与预训练时间序列基础模型对齐。每个模型从一侧继承知识、指令遵循与推理能力，从另一侧获得连续信号感知与零样本预测能力，并在共享表示空间中将两者融合，使两种模态均能被理解与生成。我们研究了使这种统一建模得以实现的设计选择：在何处对齐两个表示空间、如何将语言扎根于时间结构、如何平衡理解与生成，以及如何保持联合优化的稳定性。由此得到的方案结合了面向多样化时间序列与文本任务的统一提示方案、稳定的联合训练，以及来自220万条精选序列-文本对和490万条指令微调样本的监督。在各项基准测试中……

    arXiv:2609.29792v1 Announce Type: cross  Abstract: We present TimeBraid, a series of unified time-series and language models that align pretrained language models and pretrained time-series foundation models through interleaved global residual attention layers. Each model inherits knowledge, instruction following, and reasoning from one side, continuous-signal perception and zero-shot forecasting from the other, and fuses the two in a shared representation space where both modalities are understood and generated. We study the design choices that make such unified modeling work: where to align the two representation spaces, how to ground language in temporal structure, how to balance understanding with generation, and how to keep joint optimization stable. The resulting recipe combines a unified prompting scheme for diverse time-series and text tasks, stabilized joint training, and supervision from 2.2M curated series--text pairs and 4.9M instruction-tuning samples. Across benchmarks sp
    
[^76]: 幻觉神经元及其寻找方法：关于幻觉神经元存在性的研究

    Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons

    [https://arxiv.org/abs/2609.29781](https://arxiv.org/abs/2609.29781)

    本文提出一个五步诊断协议作为稀疏神经元定位结论的最低验证标准，并用其检验了LLM中H神经元幻觉检测能力的可复现性。

    

    针对大型语言模型（LLM）的可解释机器学习研究日益依赖于稀疏探测方法，该方法旨在识别出据称能够检测并因果性影响特定行为（如事实性回忆、安全对齐和幻觉）的小型神经元集合。这些结论对模型审计和行为引导具有重要意义，然而它们很少针对L1正则化探测方法在相关的高维特征空间中的已知失效模式进行检验。我们提出了一个五步诊断协议，涵盖特征相关性、自助法稳定性、稀疏与密集排序的不一致性、干预基线以及跨数据集评估，作为稀疏神经元定位结论的最低验证标准。我们运用所提出的方法对先前的工作进行了调查，特别是针对H神经元的研究，在TriviaQA、BioASQ和NQ-Open数据集上使用了开源LLM进行实验。我们的结果表明，检测能力在模型和数据层面均可复现。

    arXiv:2609.29781v1 Announce Type: new  Abstract: Interpretable machine learning for Large Language Models (LLMs) increasingly relies on sparse probing methods that identify small sets of neurons claimed to detect and causally influence behaviors such as factuality recall, safety alignment, and hallucination. These claims have important implications for model auditing and behavioral steering, yet they are rarely tested against known failure modes of $L_1$-regularized probing in correlated, high-dimensional feature spaces. We propose a five-step diagnostic protocol covering feature correlation, bootstrap stability, sparse versus dense ranking disagreement, intervention baselines, and cross-dataset evaluation as a minimum standard for sparse-neuron localization claims. We investigate prior work using our proposed approach, specifically on H-neurons using open-source LLMs across TriviaQA, BioASQ, and NQ-Open datasets. Our results demonstrate detection replicates across both models and data
    
[^77]: 预填充推理通道：针对推理大语言模型的输出前缀攻击

    Prefilling the Reasoning Channel: Output-Prefix Attacks on Reasoning LLMs

    [https://arxiv.org/abs/2609.29775](https://arxiv.org/abs/2609.29775)

    本文首次系统性地研究并隔离了推理模型的草稿推理通道作为输出前缀攻击向量，通过因子实验设计在暴露推理与隐藏推理模型上比较了仅推理、仅输出前缀及组合攻击的越狱效果。

    

    arXiv:2609.29775v1 公告类型：cross 摘要：大语言模型（LLM）消费并生成单一文本序列；因此，如果能够在LLM响应的开头添加文本，即输出前缀，那么后续所有token都将以该前缀为条件。这种输出前缀攻击技术是一种低成本的黑白盒提示注入方法。先前的研究已表明，这类攻击能够可靠地越狱非推理模型。大多数推理模型在助手给出最终回复之前会增加一个中间的草稿推理步骤。某些API暴露了编辑这一推理通道的能力，攻击者可以利用这些攻击向量实施推理注入攻击。我们提出了首个系统性的、受控的研究，将草稿推理通道作为输出前缀攻击向量进行隔离分析，并且首次在暴露推理和隐藏推理模型上比较了仅推理攻击、仅输出前缀攻击以及推理加输出前缀的组合攻击。我们使用包含3种前缀类型的因子实验设计…

    arXiv:2609.29775v1 Announce Type: cross  Abstract: Large Language Models (LLMs) consume and produce a single sequence of text; hence, if text can be added to the beginning of the LLM's response, i.e., an output prefix, then all subsequent tokens will be conditioned on it. This output-prefix attack technique is a cheap black-box prompt injection. Prior work has shown this type of attack can reliably jailbreak non-reasoning models. Most reasoning models add an intermediate scratchpad reasoning step before the assistant's final response. The ability to edit this reasoning channel is exposed by some APIs and attack vectors can be leveraged for reasoning injection attacks. We present the first systematic, controlled study that isolates the scratchpad reasoning channel as an output-prefix attack vector, and the first to compare reasoning-only, output-prefix-only and reasoning-plus-output-prefix attacks across both exposed- and hidden-reasoning models. Using a factorial design of 3 prefix typ
    
[^78]: 突破环境之墙：演化大语言模型智能体环境以实现递归自我改进

    Breaking the Environment Wall: Evolving LLM Agent Environments for Recursive Self-Improvement

    [https://arxiv.org/abs/2609.29773](https://arxiv.org/abs/2609.29773)

    本文提出Env-Rethink系统，通过自适应构建集合地图和事件日志来应对信息碎片化、误导性信息与环境持续演化等“未就绪”环境挑战，避免了最先进LLM智能体性能从83.9%大幅降至57.6%的退化，从而实现智能体环境的演化与递归自我改进。

    

    许多现实世界的任务（例如办公工作流程、科学实验）需要大语言模型（LLM）智能体与其环境进行反复交互，以执行依赖上下文的操作。然而，此类环境通常并未针对智能体做好准备（not agent-ready）。首先，信息往往分散且碎片化地分布在整个环境中；其次，环境中的相关证据常常与误导性信息和相互冲突的版本混杂在一起；第三，环境会随时间推移不断演化，引入新的噪声和更具挑战性的任务。这些挑战会显著降低最先进AI智能体的性能（例如从83.9%降至57.6%）。为应对这些挑战，我们提出了Env-Rethink（一个基于27B后训练模型的系统），它支持三大主要能力：(1) 它自适应地构建集合地图（Collection Maps，用于组织相关文件）和事件日志（Event Logs，用于建立跨数据关系的上下文），以补充必要的上下文；(2) 它进一步利用……（原文摘要在此处被截断）

    arXiv:2609.29773v1 Announce Type: new  Abstract: Many real-world tasks (e.g., office workflows, scientific experimentation) require LLM agents to interact repeatedly with their environments for context-dependent operations. However, such environments are often not agent-ready. First, information is often scattered and fragmented across the environment. Second, relevant evidence in the environment is often mixed with misleading information and conflicting versions. Third, environments evolve over time, introducing new noise and more challenging tasks. These challenges can substantially degrade performance for state-of-the-art AI agents (e.g., from 83.9% to 57.6%). To address these challenges, we propose Env-Rethink (a system with 27B post-trained model) that supports three main capabilities: (1) It adaptively builds Collection Maps (for organizing related files) and Event Logs (for contextualizing cross-data relationships) to supplement necessary context; (2) It further leverages the po
    
[^79]: 基于解剖结构感知的跨说话人完整声道声学到发音反演自适应

    Anatomy-aware cross-speaker adaptation of complete vocal-tract acoustic-to-articulatory inversion

    [https://arxiv.org/abs/2609.29766](https://arxiv.org/abs/2609.29766)

    提出了一种基于解剖标志点（椎骨和牙齿结构）的几何自适应框架，通过仿射变换加薄板样条形变，将固定的声学到发音反演模型的预测迁移到未见说话人，且无需重新训练。

    

    跨说话人的声学到发音反演需要考虑说话人之间的解剖差异。我们提出了一种几何自适应框架，利用解剖标志点（主要位于椎骨和牙齿结构上）将固定反演模型的预测迁移到未见过的说话人。通过仿射变换结合薄板样条（TPS）形变，将预测的10个声道结构轮廓映射到每个目标说话人的几何结构中，无需重新训练。标志点在每个说话人选定的一个/u/帧中识别，作为共同的语音参考，而不假设说话人之间具有相同的发音构型，所得映射可在多条录音中重复使用。我们在单说话人实时MRI（rt-MRI）数据库上训练模型，并在来自另一个多说话人rt-MRI数据库的八位说话人上评估自适应效果。我们比较了使用12个或14个标志点的仿射与TPS配置。Affine12+TPS14实现了……

    arXiv:2609.29766v1 Announce Type: cross  Abstract: Cross-speaker acoustic-to-articulatory inversion requires accounting for anatomical differences between speakers. We propose a geometric adaptation framework that uses anatomical landmarks, primarily on vertebrae and dental structures,to transfer predictions from a fixed inversion model to unseen speakers. An affine transformation followed by thin-plate spline (TPS) deformation maps the predicted contours of 10 vocal-tract structures into each target speaker's geometry without retraining. Landmarks are identified in one selected /u/ frame per speaker as a common phonetic reference without assuming identical articulatory configurations across speakers, and the resulting mapping is reused across recordings. We train the model on a single-speaker rt-MRI database and evaluate adaptation on eight speakers from a separate multi-speaker rt-MRI database. We compare affine and TPS configurations using 12 or 14 landmarks. Affine12+TPS14 achieves
    
[^80]: 提交之间：完全由AI编写的代码库中的开发过程、错误与声明可靠性

    Between the Commits: Process, Error, and Claim Reliability in a Wholly AI-Authored Codebase

    [https://arxiv.org/abs/2609.29744](https://arxiv.org/abs/2609.29744)

    本研究首次构建并分析了完全由Claude AI编写（无任何人工代码或测试）的21,000行Python工具的完整开发历史数据集，配合代码溯源工具与三种分类体系，发现14.3%的AI代码生成事件含有真实错误、约四分之一至五分之一的AI交互响应包含事实性错误。

    

    我们提出了：(i) 一个新数据集，包含一个完全由Claude AI构建的21,000行Python工具的完整开发历史，其中没有任何人工编写的代码或测试， 两个代码溯源追踪工具， 用于指令意图、提交溯源和响应可靠性的三种分类体系， 将这些工具应用于该数据集的分析。我们发现： 用户在编程代理CLI中的指令与IDE聊天中的指令在性质上不同，前者更侧重于理解、规划和咨询， 代码开发主要是主动进行的， 14.3%的AI代码生成事件包含真实错误，这些错误后来被AI编写的测试套件捕获， AI的交互式响应中大约每4到5个就有1个包含一个或多个事实性错误。

    arXiv:2609.29744v1 Announce Type: cross  Abstract: We present: (i) a new dataset consisting of the full development history of a 21,000-line Python tool built entirely by Claude AI, with no human-authored code or tests, (ii) two code-provenance tracing tools, (iii) three taxonomies for instruction intent, commit provenance, and response reliability, (iv) application of these to analyse the dataset. We find that: (i) user coding agent CLI instructions differ in kind from IDE-chat instructions, with a greater focus on comprehension, planning and consultation, (ii) code development is mainly proactive, (iii) 14.3% of AI code-generation events contain a real error later caught by the AI-authored test suite, (iv) roughly 1 in 4-5 of the AI's interactive responses contains one or more factual errors.
    
[^81]: 基于人工智能的低分辨率远程监测数据心衰恶化检测

    AI-based detection of worsening heart failure from low-resolution telemonitoring data

    [https://arxiv.org/abs/2609.29742](https://arxiv.org/abs/2609.29742)

    提出了TRACER模型——一种结合时间感知嵌入与对比预训练的Transformer架构，可从低分辨率、不规则采样的远程监测数据中早期预测心衰患者的恶化及罕见住院事件。

    

    目标：心力衰竭（HF）因其高合并症负担、患者群体老龄化以及频繁住院而成为一项医疗保健挑战。远程监测通过早期发现健康恶化，为心衰患者管理提供了一种有前景的方法。开发能够从远程监测数据中自动检测恶化迹象的自主系统，有助于减轻医护人员的工作负担。方法：我们提出了TRACER模型，这是一种采用对比事件表示的Transformer模型，旨在从低分辨率且不规则采样的远程监测数据中预测导致罕见住院事件的时间线。TRACER为每个生物标志物引入了时间感知嵌入，通过对比预训练进行表征学习以增强异常检测能力，并使用独立的二元分类器进行检测。我们使用了包含276名心衰患者远程记录的生物标志物序列的测量数据，并将其分割为重叠的片段……

    arXiv:2609.29742v1 Announce Type: new  Abstract: Objective: Heart failure (HF) presents a healthcare challenge due to its high comorbidity burden, aging patient population and frequent hospitalizations. Remote monitoring offers a promising approach to managing HF patients by early detection of health deterioration. Developing autonomous systems to detect signs of worsening in telemonitoring data is of interest to reduce the workload of healthcare personnel. Methods: We propose the TRACER model, a Transformer with Contrastive Event Representation, designed to predict timelines leading to rare hospitalization events in low-resolution and irregularly sampled telemonitoring data. TRACER incorporates time-aware embeddings for each biomarker, contrastive pre-training to enhance anomaly detection via representation learning, and independent binary classifiers for detection. We used measurement data containing remotely recorded biomarker sequences from 276 HF patients segmented into overlappin
    
[^82]: TopU-LBVS：一个面向基于配体的虚拟筛选的现实多靶点基准测试

    TopU-LBVS: A Realistic Multi Target Benchmark for Ligand Based Virtual Screening

    [https://arxiv.org/abs/2609.29740](https://arxiv.org/abs/2609.29740)

    提出了TopU-LBVS，一个覆盖7大类93个蛋白靶点、采用性质匹配且结构相似诱饵并配有三种固定评估协议的多靶点基于配体虚拟筛选基准，解决了现有基准因简单诱饵和随机阴性而高估性能的问题。

    

    基于配体的虚拟筛选（LBVS）是早期药物发现中一种实用的一线筛选工具，但现有基准测试可能因随机阴性样本、容易的诱饵分子、靶点覆盖有限以及不规范的评估协议而高估性能。我们提出了TopU-LBVS，一个在困难阴性筛选条件下的多靶点LBVS基准测试。基于经整理的ChEMBL 35生物活性数据，TopU-LBVS覆盖7个蛋白质类别中的93个蛋白质靶点，并以固定的1:40活性化合物与诱饵比例，构建了具有性质匹配、结构相似诱饵的靶点特异性筛选文库。每个文库包含约400到10,000个化合物，其设计旨在减少基于简单理化性质和最近邻指纹的捷径学习。TopU-LBVS提供了三种固定评估协议。TopU-LBVS-full评估ChEMBL* → TopU在全部93个靶点上的泛化能力。TopU-LBVS-low评估低数据量条件下TopU → TopU的性能表现。

    arXiv:2609.29740v1 Announce Type: cross  Abstract: Ligand-based virtual screening (LBVS) is a practical first-pass tool in early-stage drug discovery, but existing benchmarks can overestimate performance through random negatives, easy decoys, limited target coverage, and non-standardized evaluation protocols. We introduce TopU-LBVS, a multi-target benchmark for LBVS under hard-negative screening conditions. Starting from curated ChEMBL~35 bioactivity data, TopU-LBVS covers 93 protein targets across 7 protein classes and constructs target-specific screening libraries with property-matched, structurally similar decoys at a fixed 1:40 active-to-decoy ratio. Libraries contain roughly 400 to 10,000 compounds and are designed to reduce simple physicochemical and nearest-neighbor fingerprint shortcuts.   TopU-LBVS provides three fixed protocols. TopU-LBVS-full evaluates ChEMBL$^\ast \rightarrow$ TopU generalization across all 93 targets. TopU-LBVS-low evaluates low-data TopU $\rightarrow$ Top
    
[^83]: C3M：面向长程任务的跨会话多模态记忆维护

    C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks

    [https://arxiv.org/abs/2609.29735](https://arxiv.org/abs/2609.29735)

    C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。

    

    长程任务需要在有限的、与查询无关的记忆预算下保存并在之后恢复跨会话证据。现有的压缩方法可能会丢弃细粒度的视觉线索，或会将语义相似但不兼容的观察混淆在一起。我们提出了C3M，这是一种跨会话多模态记忆组织方式，它在持久的源文本-图像证据之上维护一个有界的活动索引。基于关系的更新在整合安全冗余的同时，保留互补和不兼容的记录。在查询时，预算化的路由机制选择有用的索引页面，并在固定的读取器预算下展开其关联的源证据。这些机制共同为跨会话长程任务建立了一种紧凑的、保留来源信息的多模态记忆组织，保留了可靠下游推理所需的时间区分和源链接。代码可在 https://github.com/HuzhouNLP/C3M 获取。

    arXiv:2609.29735v1 Announce Type: new  Abstract: Long-horizon tasks require preserving and later recovering cross-session evidence under a bounded, query-blind memory budget. Existing compression can discard fine-grained visual cues or conflate semantically similar but incompatible observations. We present C3M, a cross-session multimodal memory organization that maintains a bounded active index over persistent source text-image evidence. Relation-aware updates consolidate safe redundancy while preserving complementary and incompatible records. At query time, budgeted routing selects useful index pages and expands their associated source evidence under a fixed reader budget. Together, these mechanisms establish a compact, provenance-preserving multimodal memory organization for cross-session long-horizon tasks, retaining temporal distinctions and source links required for reliable downstream reasoning. Code is available at https://github.com/HuzhouNLP/C3M.
    
[^84]: 偏差中的黄金：通过验证使AI设计过程走向成熟

    The Gold in Bias: Maturing the AI Design Process through Verification

    [https://arxiv.org/abs/2609.29730](https://arxiv.org/abs/2609.29730)

    本文将AI偏差从“需要消除的缺陷”重新概念化为“诊断工具”，提出了涵盖偏差起源来源、生命周期出现节点、技术成因和验证方法的四维分析框架，以验证驱动的方式推动AI设计流程走向成熟。

    

    AI系统中的偏差通常被视为需要最小化的缺陷，然而它同时也是揭示数据、建模假设和系统设计中潜在弱点的关键指标。现有方法往往将偏差视为孤立问题，而非将其视为能够在AI全生命周期中强化验证与治理的证据。本文旨在将偏差重新概念化为一种支持严格AI验证的诊断工具。我们致力于开发一个多维度框架来分析偏差，展示偏差如何在传统AI和生成式AI中产生，并为验证驱动的偏差缓解提供结构化路径。我们提出了一个从四个维度分析偏差的多维框架：偏差的起源来源、AI建模全生命周期中的出现节点、技术和方法论层面的成因，以及用于检测与缓解的验证方法。通过一个涵盖传统AI和生成式AI的全面分类体系……（摘要在此处截断）

    arXiv:2609.29730v1 Announce Type: new  Abstract: Bias in AI systems is typically framed as a flaw to be minimized, yet it also serves as a critical indicator of underlying weaknesses in data, modeling assumptions, and system design. Existing approaches often treat bias as an isolated problem rather than as evidence that can strengthen verification and governance across the AI lifecycle. This paper aims to reconceptualize bias as a diagnostic tool that supports rigorous AI verification. We seek to develop a multidimensional framework to analyze bias, demonstrate how biases emerge in both Traditional and Generative AI, and provide a structured pathway for verification-driven mitigation. We present a multidimensional framework analyzing bias across four dimensions: origin sources, emergence points throughout the AI modeling lifecycle, technical and methodological causes, and validation approaches for detection and mitigation. Through a comprehensive typology spanning traditional and gener
    
[^85]: 应请求生成的预算化阈值激励通用框架

    A General Framework for Budgeted Threshold Incentives on Request

    [https://arxiv.org/abs/2609.29724](https://arxiv.org/abs/2609.29724)

    该论文提出了一个请求驱动的预算化阈值激励通用框架，通过四个阶段和七个可替换模块为配送平台生成激励计划，并证明了端到端价值损失可由四个阶段误差项之和界定。

    

    即时配送平台通过激励活动向骑手支付报酬，激励活动的档位是根据历史记录相似的骑手近期完成量来设定的。运营方会针对不断变化的时段、骑手群体、支付规则和预算来请求此类激励计划，通常是为了应对节假日或恶劣天气，而在这些场景下随机试验数据稀少且需要数月才能收集完成。我们提出了一个由请求驱动的框架，该框架通过七个可替换的模块来组合四个阶段（条件预测、群体缩减、轨迹整合与预算分配），各模块之间交换条件轨迹规律；这些规律中的奖励概率和以奖励标记的矩量，可为任意活动规则给出支付金额与提升效果的估计。一个响应校正步骤对来自大量“无激励”历史数据的轨迹进行重新加权，使其矩量与短期试点数据的矩量相匹配。我们证明，在固定的计划菜单上并给定各阶段的误差时，端到端的价值损失由四个阶段项之和所界定，并且对于每个阶段……（原文摘要在此处截断）

    arXiv:2609.29724v1 Announce Type: new  Abstract: On-demand delivery platforms pay riders through incentive activities whose tiers are set from recent completions of riders with a similar history. Operators request such plans for changing periods, rider populations, payment rules and budgets, often for holidays or bad weather, where randomized trials are scarce and take months to collect. We present a request-driven framework that composes four stages (conditional prediction, population reduction, trajectory integration and budget allocation) through seven replaceable modules that exchange conditional trajectory laws, whose award probabilities and award-marked moments give payment and uplift for any activity rule. A response-correction step reweights trajectories from abundant no-offer history to match the moments of a short pilot. We prove that, on a fixed plan menu and given the stage errors, the end-to-end value loss is bounded by the sum of four stage terms, and that for every stage
    
[^86]: PPTBench：编码智能体能否通过结构化、可编辑的幻灯片重建视觉世界

    PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides

    [https://arxiv.org/abs/2609.29718](https://arxiv.org/abs/2609.29718)

    该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。

    

    编码智能体开始在视觉世界中发挥作用，它们如今能够构建网页、图形界面、游戏、3D场景、图表和文档。要在视觉编码中取得成功，需要弥合两个空间：推断视觉结构并将其以程序化方式表达。幻灯片是知识工作的核心媒介，被广泛用于以一种人们可直接查看和编辑的形式交流想法和开展协作。因此，幻灯片为视觉编码提供了理想的测试平台，因为它要求智能体恢复视觉结构并将其实现为可编辑的对象。然而，现有的基准测试要么依赖主观的开放式评估，要么产生不可编辑的代码输出，要么仅关注局部编辑而非端到端的视觉重建。我们提出了PPTBench，通过可编辑幻灯片重建对视觉编码进行基准评测。它包含500个任务，每个任务都基于来自真实arXiv论文的科学流程图，要求智能体重建……（摘要原文在此处截断）

    arXiv:2609.29718v1 Announce Type: cross  Abstract: Coding agents are beginning to act in the visual world. They now build webpages, GUIs, games, 3D scenes, diagrams, and documents. Success in such visual coding requires bridging two spaces: inferring visual structure and expressing it programmatically. Slides are a core medium of knowledge work, widely used to communicate ideas and collaborate in a form that people can directly inspect and edit. Therefore, they provide an ideal testbed for visual coding, as they require agents to recover visual structure and realize it as editable objects. However, existing benchmarks either rely on subjective open-ended evaluation, produce non-editable code outputs, or focus only on local editing rather than end-to-end visual reconstruction. We introduce PPTBench, which benchmarks visual coding through editable slide reconstruction. It contains 500 tasks, each based on a scientific flow diagram from a real arXiv paper and requiring agents to reconstru
    
[^87]: 在分布偏移下，重新验证优于有状态路由的科学代理模型选择策略

    Revalidation Beats Stateful Routing for Scientific Surrogates Under Distribution Shift

    [https://arxiv.org/abs/2609.29715](https://arxiv.org/abs/2609.29715)

    该研究通过大规模可复现的流式基准实验证明，在分布偏移条件下，无需复杂的有状态自适应控制器，只需在每个新数据批次上重新验证候选代理模型并选择验证损失最低者，即可将平均对数遗憾降至0.091，显著优于事后最优的固定模型选择策略（0.192）。

    

    代理模型通常在开发阶段被选定，之后随着新测量数据的到来而保持不变。当噪声、输入支持范围或物理参数发生变化时，这种做法就变得有风险。我们提出了一个问题：这些变化是否需要引入有状态的自适应控制器，还是只需在每个新批次上对候选模型重新进行验证就足够了。为研究这一问题，我们构建了RegimeShift-Surrogates，这是一个可复现的流式基准测试，涵盖八个解析与动力学任务、四种平稳或偏移的机制、十个保留的随机种子，以及八个基于经典方法、多层感知机和Kolmogorov-Arnold网络的代理模型。验证性实验包含30,720次模型拟合和3,200个评分部署窗口。选择当前窗口中验证损失最低的模型，相对于逐窗口预言机的平均对数遗憾为0.091；而事后选择的最优固定模型的遗憾为0.192。两者的配对差异为-0.101（分层自助法……

    arXiv:2609.29715v1 Announce Type: cross  Abstract: Surrogate models are often chosen during development and then left in place as new measurements arrive. That practice becomes risky when noise, input support, or physical parameters change. We asked whether such changes call for a stateful adaptive controller, or whether it is enough to validate the candidate models again on each new batch. To study this question, we built RegimeShift-Surrogates, a reproducible streaming benchmark spanning eight analytic and dynamical tasks, four stationary or shifting regimes, ten held-out seeds, and eight classical, multilayer-perceptron, and Kolmogorov-Arnold network surrogates. The confirmatory run contains 30,720 model fits and 3,200 scored deployment windows. Choosing the model with the lowest validation loss in the current window yields mean log regret 0.091 against a per-window oracle; the best fixed model chosen in hindsight yields 0.192. The paired difference is -0.101 (hierarchical bootstrap
    
[^88]: 解耦知识与隐私：面向大语言模型持续学习的任务后自蒸馏重放

    Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning

    [https://arxiv.org/abs/2609.29711](https://arxiv.org/abs/2609.29711)

    该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。

    

    隐私保护的持续学习（PPCL）必须在跨序列任务保留有用知识的同时，减少敏感内容的再现。形式化隐私保证刻画的是随机化机制，而操作性输出控制关注的则是训练后的模型能否选择性地降低其输出中敏感内容出现的可能性。在这项工作中，我们在现实的任务演化情境下，将后者与持续学习效用放在一起进行研究。知识保留与隐私修正作用于不同的粒度：任务获取需要广泛保留当前任务和旧任务的行为，而隐私修正则针对稀疏的标注位置。联合优化会导致当前任务的保留目标不断变化。我们提出了SPARK，一种保留-修正分解方法，它首先冻结学习到的任务后分布，然后围绕这一稳定参考进行选择性修正。

    arXiv:2609.29711v1 Announce Type: cross  Abstract: Privacy-preserving continual learning (PPCL) must reduce the reproduction of sensitive content while retaining useful knowledge across sequential tasks. Formal privacy guarantees characterize randomized mechanisms, whereas operational output control concerns whether a trained model selectively reduces the likelihood of sensitive content in its outputs. In this work, we investigate the latter together with continual-learning utility under realistic task evolution. Retention and privacy correction operate at different granularities: task acquisition requires broad preservation of current- and old-task behavior, whereas privacy correction targets sparse annotated positions. Joint optimization leaves the current-task preservation target continually changing. We propose SPARK, a retention-correction decomposition that first freezes the learned post-task distribution and then applies selective correction around this stable reference. Self-Di
    
[^89]: 理解与利用基于反馈的智能体规划中的初始化锚定弱点

    Understanding and Exploiting Initialization Anchoring Weakness in Feedback-Based Agent Planning

    [https://arxiv.org/abs/2609.29697](https://arxiv.org/abs/2609.29697)

    该研究发现了基于反馈的智能体规划中存在“初始化锚定弱点”——首轮反馈能纠正46%的对抗方向，但后续轮次的纠正率骤降至13%和7%，并提出黑盒攻击框架InitAnchor利用这一安全缺陷。

    

    基于反馈的规划通过纳入工具观察结果和纠正性反馈来提高智能体的可靠性。然而，这种保护可能并非均匀地分布在规划的各个阶段。我们对四种代表性的反馈机制进行了逐轮分析，发现了一种初始化锚定弱点：第一轮反馈能够纠正46%的对抗性方向，而在存活到随后两轮的方向中，纠正率分别降至13%和7%。我们的分析将这一弱点归因于三个相互作用的因素：初始计划中上下文合理的偏移、反证不足，以及已被接受的方向在累积轨迹中的持续性。基于这些发现，我们提出了InitAnchor，一个通过攻击者控制的外部材料来利用这一弱点的黑盒框架。它将这三个因素操作化为方向偏移、上下文合理性和反证（缺失）……

    arXiv:2609.29697v1 Announce Type: cross  Abstract: Feedback-based planning improves agent reliability by incorporating tool observations and corrective feedback. However, its protection may not be distributed uniformly across planning stages. We conduct a round-wise analysis of four representative feedback mechanisms and uncover an initialization anchoring weakness: the first feedback round corrects 46\% of adversarial directions, whereas the rates fall to 13\% and 7\% among directions surviving into the next two rounds. Our analysis attributes this weakness to three interacting factors: a contextually plausible shift in the initial plan, insufficient counterevidence, and the persistence of accepted directions in the accumulated trajectory. Based on these findings, we propose \textsc{InitAnchor}, a black-box framework for exploiting this weakness through attacker-controlled external materials. It operationalizes the three factors as directional-shift, contextual-plausibility, and count
    
[^90]: 像我们一样公平吗？审计大语言模型在资源分配中的对齐性

    Fair Like Us? Auditing LLM Alignment in Resource Allocation

    [https://arxiv.org/abs/2609.29692](https://arxiv.org/abs/2609.29692)

    本文提出了一种审计大语言模型公平性推理的通用评估方法，通过将LLM的第一人称公平判断与人类在匹配场景下的回应直接对比，发现LLM比人类更偏好严格的公平约束、表现出更多自利行为、对信息框架敏感，且难以通过现有微调方法与人类判断对齐。

    

    公平分配稀缺且不可分割的资源是许多社会问题中的重要挑战。尽管存在多种正式的公平理论，但没有任何单一的定义能够始终被满足。随着大语言模型（LLM）越来越多地被用于支持决策并充当智能体，它们引发了对分配正义的新担忧：其判断并不直接绑定于任何特定的公平框架，并且可能违反关键的规范性原则。在这项工作中，我们引入了一种评估LLM公平性推理的通用方法。我们研究了广泛模型集合中的第一人称公平性判断，并将其与人类在匹配场景和相同引导条件下的回应进行直接比较。我们发现，LLM倾向于比人类偏好更严格的公平约束，表现出更多的自利行为，对信息的表述框架敏感，并且难以通过当前的微调方法与人类判断实现对齐。

    arXiv:2609.29692v1 Announce Type: new  Abstract: Fair allocation of scarce, indivisible resources is an important challenge in many societal problems. While there are several formal theories of fairness, no single definition can always be satisfied. As large language models (LLMs) are increasingly used to support decisions and act as agents, they raise new concerns about distributional justice: their judgments are not directly tied to any specific fairness framework and may violate key normative principles. In this work, we introduce a general method for evaluating fairness reasoning in LLMs. We study first-person fairness judgments across a broad set of models and compare them directly with human responses on matched scenarios and elicitation conditions. We find that LLMs tend to prefer stricter fairness constraints than humans, show more self-interested behavior, are sensitive to how information is framed, and are difficult to align with human judgments using fine-tuning with current
    
[^91]: ReCalMatch：面向半监督细粒度识别的可靠性校准语义引导

    ReCalMatch:Reliability-Calibrated Semantic Guidance for Semi-Supervised Fine-Grained Recognition

    [https://arxiv.org/abs/2609.29678](https://arxiv.org/abs/2609.29678)

    提出ReCalMatch框架，利用多方面语义原型作为校准证据来评估伪标签可靠性，从而解决半监督细粒度识别中视觉分类器导致的过度自信伪标签错误问题。

    

    半监督细粒度视觉识别极易受到过度自信的伪标签错误的影响：视觉上相似的类别经常产生高置信度却错误的预测，而一致性正则化随后会在整个训练过程中不断强化这些错误。现有的半监督学习（SSL）方法几乎完全依赖视觉分类器本身来估计伪标签的可靠性——例如最大概率、自适应阈值或熵——这些信号无法判断预测类别是否与视觉表示在语义上兼容。我们提出了ReCalMatch，一个用于半监督细粒度识别的可靠性校准语义框架。ReCalMatch并非将文本语义作为辅助监督，而是将多方面语义原型用作伪标签学习的校准证据。我们基于类别名称构建类别条件化的语义原型……

    arXiv:2609.29678v1 Announce Type: cross  Abstract: Semi-supervised fine-grained visual recognition is highly vulnerable to overconfident pseudo-label errors: visually similar categories frequently produce high-confidence yet incorrect predictions, and consistency regularization then reinforces these errors throughout training. Existing semi-supervised learning (SSL) methods estimate pseudo-label reliability almost entirely from the visual classifier itself---maximum probability, adaptive thresholds, or entropy---signals that remain blind to whether a predicted class is \emph{semantically} compatible with the visual representation. We propose \textbf{ReCalMatch}, a reliability-calibrated semantic framework for semi-supervised fine-grained recognition. Rather than treating textual semantics as auxiliary supervision, ReCalMatch uses multi-aspect semantic prototypes as \emph{calibration evidence} for pseudo-label learning. We construct class-conditioned semantic prototypes from class names
    
[^92]: 持续学习的序列代价

    The Sequential Price of Continual Learning

    [https://arxiv.org/abs/2609.29674](https://arxiv.org/abs/2609.29674)

    该论文在过参数化线性回归模型中首次精确刻画了持续学习的“序列代价”：总损失恰好分解为联合训练的内在损失与额外序列代价，在同质任务几何下总损失为联合训练的两倍，且EWC正则化能以与强度成反比的方式降低该代价。

    

    序列式任务更新是持续学习的基础，但其对新任务的近因偏差可能带来持久的性能代价。我们在一个采用独立同分布任务采样的过参数化线性回归模型中研究这一代价。我们证明，分布层面的遗忘与总体损失会收敛到同一个平稳极限。这一共同极限可以精确分解为两部分：联合训练渐近达到的内在损失，以及一个额外的“序列代价”；在更同质的任务几何结构下，这两项恰好相等，使得总损失达到联合训练的两倍。我们进一步分析了在一般任务曲率下固定强度的弹性权重固化方法，并刻画了其在每个正则化强度下的平稳序列代价。在强正则化条件下，该代价随EWC强度成反比衰减，同时收敛到平稳状态的速度也以相同尺度变慢。在Jester笑话评分数据集上，该理论精确量化了……（原文在此处截断）

    arXiv:2609.29674v1 Announce Type: cross  Abstract: Sequential task updates are fundamental to continual learning, but their recency bias can impose a lasting performance cost. We study this cost in an overparameterized linear-regression model with i.i.d. task sampling. We prove that distribution-level forgetting and population loss converge to the same stationary limit. This common limit separates exactly into the intrinsic loss asymptotically attained by joint training and an additional sequential price, and in more homogeneous task geometries the two terms coincide, making the total loss twice that of joint training. We further analyze fixed-strength elastic weight consolidation (EWC) under general task curvatures and characterize its stationary sequential price at every regularization strength. Under strong regularization, the price decays inversely with EWC strength while convergence to stationarity slows at the same scale. On the Jester joke-rating dataset, the theory exactly quan
    
[^93]: 世界模型能让机器人变得更好吗？预测性具身智能评估基准综述

    Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence

    [https://arxiv.org/abs/2609.29669](https://arxiv.org/abs/2609.29669)

    本综述通过收录并整理160个评估基准，揭示了现有评测体系的核心缺口——世界模型基准只评预测不执行、任务成功套件缺乏世界模型与VLA策略的对比——导致该领域尚无法回答“世界模型能否带来可度量的闭环优势”这一关键问题。

    

    机器人学习如今沿着两条几乎不相交的路线发展。一方面，直接的视觉-语言-动作（VLA）策略将观测映射为动作，并以闭环任务成功率作为评价标准；另一方面，预测性和生成式世界模型对未来观测进行预测，并以开环预测或生成质量作为评价标准。两者之间存在一个自然而然的问题：世界建模能否在直接策略之上带来可度量的闭环优势？这种优势又体现在哪些机器人能力上？我们认为该领域目前尚无法回答这一问题，其原因在于评估方式的缺口，而非模型本身。世界模型基准只对预测进行评分而不执行它，而任务成功评测套件只运行单一策略，从未构建世界模型与VLA策略的对比。本综述围绕这一缺口梳理了当前的评估格局。我们收录了160个经网络核实的基准（涵盖2017年至2026年），并按评估……（摘要原文在此处截断）

    arXiv:2609.29669v1 Announce Type: cross  Abstract: Robot learning now advances along two tracks that rarely meet. On one side, direct Vision-Language-Action (VLA) policies map observations to actions and are scored by closed-loop task success. On the other, predictive and generative world models forecast future observations and are scored by open-loop prediction or generation quality. A natural question sits between them: does world modelling earn a measurable, closed-loop advantage over a direct policy, and for which robotic capabilities? We argue that the field cannot yet answer this question, and that the reason is a gap in how it is measured, not in the models themselves. World-model benchmarks score prediction without ever executing it, while task-success suites host a single policy and never build a world-model versus VLA contrast.   This survey maps the evaluation landscape around that gap. We catalogue 160 web-verified benchmarks spanning 2017 to 2026 and organise them by evalu
    
[^94]: 图、循环与线束工程：面向零信任智能体数据工程与分析处理

    Graph, Loop, and Harness Engineering for Zero-Trust Agentic Data Engineering and Analytical Processing

    [https://arxiv.org/abs/2609.29668](https://arxiv.org/abs/2609.29668)

    该论文提出零信任智能体数据工程与零信任智能体OLAP两个框架，通过图工程、循环工程与线束工程三种抽象，以证据门控、有界恢复和严格验证机制确保大语言模型智能体可靠地完成端到端云数据工程与分析处理。

    

    arXiv:2609.29668v1（公告类型：交叉列表）。摘要：大语言模型智能体日益推动数据工作流的自动化，但端到端的云数据工程与分析执行需要在代码、数据、基础设施和运行时环境之间实现可靠的协调。我们提出了两个零信任框架。零信任智能体数据工程从自然语言任务出发，生成、部署并验证完整的云数据工程解决方案，其完成以仓库、部署、运行时和策略证据为前提条件。零信任智能体OLAP将受治理的数据准备与经验证的联机分析处理（OLAP）相结合，仅在通过验证并获得证据绑定的批准后才允许投入生产，且只有在经过同快照执行、精确结果等价、确定性落地和反思之后才发布分析答案。两个框架共享三个抽象：用于证据门控工作流结构的图工程、用于有界恢复的循环工程……（原文摘要至此截断）

    arXiv:2609.29668v1 Announce Type: cross  Abstract: Large language model agents increasingly automate data workflows, but end-to-end cloud data engineering and analytical execution require reliable coordination across code, data, infrastructure, and runtime environments. We present two zero-trust frameworks. Zero-Trust Agentic Data Engineering generates, deploys, and verifies complete cloud data-engineering solutions from natural-language tasks, with completion conditioned on repository, deployment, runtime, and policy evidence. Zero-Trust Agentic OLAP combines governed Data Preparation with verified Online Analytical Processing (OLAP), permitting production promotion only after validation and evidence-bound approval, and releasing analytical answers only after Same-Snapshot Execution, Exact Result Equivalence, deterministic grounding, and reflection. Both frameworks share three abstractions: graph engineering for evidence-gated workflow structure, loop engineering for bounded recovery,
    
[^95]: 思考与否：将推理分配到真正有效之处

    To Think or Not to Think: Allocating Reasoning Where It Helps

    [https://arxiv.org/abs/2609.29664](https://arxiv.org/abs/2609.29664)

    该论文发现推理长度对准确率的提升效果主要集中在“部分可解”的问题上，而非越难的问题越能从更长推理中获益，并据此提出了CARE方法，将推理资源精准分配到真正受益的问题上，以解决大模型推理中的长度错配问题。

    

    强化学习（RL）已被证明能有效提升大型语言模型（LLM）的推理性能，尤其是在复杂的数学和编程任务中。然而，这种能力伴随着系统性的“长度错配”问题：模型在简单问题上投入过多的推理，而在更难的问题上却过早终止，导致推理效率下降，而准确率几乎没有提升。许多长度自适应方法通过根据问题难度分配token预算来缓解这一问题，其隐含假设是：更难的问题能从更长的推理中单调获益。与此相反，我们发现推理长度对准确率的影响集中在“部分可解”的问题上。我们的进一步分析表明，显式的长度奖励可能产生意想不到的训练动态。基于这些发现，我们提出了CARE（Contrastive...）——（注：原摘要在此处被截断）

    arXiv:2609.29664v1 Announce Type: new  Abstract: Reinforcement learning (RL) has proven effective in enhancing the reasoning performance of large language models (LLMs), particularly in complex mathematical and programming tasks. However, this capability comes with systematic \textit{length misallocation}, in which models devote excessive reasoning to simple questions while terminating prematurely on harder ones, degrading inference efficiency with negligible accuracy improvement. Many length-adaptive methods mitigate this issue by allocating token budgets according to question difficulty, under the implicit assumption that harder questions benefit monotonically from extended reasoning. In contrast, we find that the effect of reasoning length on accuracy is concentrated on \textit{partially solvable} questions. Our further analysis reveals that explicit length rewards can produce unintended training dynamics. Motivated by these findings, we propose \textbf{CARE}---\textbf{C}ontrastive 
    
[^96]: 研究白细胞作为非洲血涂片图像中疟疾寄生虫检测假阳性的来源

    Investigating White Blood Cells as a Source of False-Positive Malaria Parasite Detection in African Blood-Smear Images

    [https://arxiv.org/abs/2609.29663](https://arxiv.org/abs/2609.29663)

    该研究通过七项独立的空间与统计分析证伪了“白细胞是疟疾寄生虫检测假阳性主要来源”的假设，发现95%的假阳性实为纯背景误检，而非白细胞混淆。

    

    每一张吉姆萨染色的厚血涂片上都存在的白细胞，与恶性疟原虫早期环状滋养体具有相似的视觉特征：体积小、形态圆、紫色染色深。因此它们是仅寄生虫检测器中一个合理但未经证实的假阳性来源。我们在Lacuna疟疾检测数据集（来自乌干达和加纳的8,000张图像）上训练了两个YOLOv12s模型：模型A仅使用寄生虫标签，模型B同时使用寄生虫和白细胞标签。七项独立的空间与统计分析检验了假阳性（FP）预测是否聚集在白细胞位置附近，结果全部否定了这一假设。在两个模型中，95%的假阳性都是纯背景检测（相对任何真实标注框的IoU低于0.10），没有一例是白细胞类别的混淆。Ripley交叉K分析显示，在所有测试半径下，假阳性中心点与白细胞位置之间均呈现空间排斥。模型B总体上优于模型A（mAP5……

    arXiv:2609.29663v1 Announce Type: cross  Abstract: White blood cells (WBCs) present on every Giemsa-stained thick blood smear share visual properties with early-stage Plasmodium falciparum ring-form trophozoites: small size, round morphology, and intense purple staining. They are a plausible but untested source of false positives in parasite-only detectors. We trained two YOLOv12s models on the Lacuna Malaria Detection dataset (8,000 images from Uganda and Ghana): Model A with parasite labels only, and Model B with both parasite and WBC labels. Seven independent spatial and statistical analyses tested whether false positive (FP) predictions cluster near WBC locations. All seven refute the hypothesis. In both models, 95% of FPs are pure background detections (IoU below 0.10 against any ground-truth box); zero are WBC class confusions. Ripley's Cross-K analysis shows spatial repulsion between FP centroids and WBC positions at every radius tested. Model B outperforms Model A overall (mAP5
    
[^97]: 面向修订语料库的低成本可靠问答：摄入时事实编译方法

    Ingest-Time Fact Compilation for Cost-Efficient and Reliable Question Answering over Revised Corpora

    [https://arxiv.org/abs/2609.29661](https://arxiv.org/abs/2609.29661)

    提出摄入时事实编译架构，在数据摄入或变更时一次性将修订、删除、生效日期与来源信任规则解析并编译为带溯源的类型化事实记录，使查询时仅需低成本模型读取编译结果，从而在含修订的语料库上实现成本更低、更可靠的问答。

    

    大多数智能体式问答系统在最糟糕的时间完成了一部分重要的语义工作：即每次有人提问的时候。当语料库包含修订、草稿、撤销、删除以及具有不同权威级别的来源时，模型必须在每次读取时重建受治理的当前状态——然后丢弃这些工作，并在下一次查询时重复这一过程。这有点像一个数据库在每次有人读取时都重新构建物化视图。我们提出了“摄入时事实编译”这一架构，它在语料库数据被摄入或发生变更时完成这项工作：原始段落被改写为自包含的事实；管理修订、删除、生效日期和来源可信度的规则只解析一次；得到的状态以携带来源与修订溯源信息的类型化记录形式存储。在查询时，一个低成本的模型只需读取编译后的记录，而无需从噪声……（原文摘要到此被截断）

    arXiv:2609.29661v1 Announce Type: new  Abstract: Most agentic question answering (QA) systems do an important part of their semantic work at the worst possible time: every time someone asks a question. When a corpus contains revisions, drafts, revocations, deletions, and sources with different levels of authority, the model must reconstruct the governed current state on every read - then throw that work away and repeat it on the next query. This is a bit like a database that rebuilds a materialized view every time someone reads from it. We present ingest-time fact compilation, an architecture that performs this work when corpus data is ingested or changed. Raw passages are rephrased into self-contained facts; rules governing revisions, deletions, effective dates, and source trust are resolved once; and the resulting state is stored as typed records carrying source and revision provenance. At query time, an inexpensive model reads the compiled record instead of reconstructing it from no
    
[^98]: AgentKernel：信任原生的智能体操作系统

    AgentKernel: The Trust-Native Agentic Operating System

    [https://arxiv.org/abs/2609.29647](https://arxiv.org/abs/2609.29647)

    AgentKernel提出了一种信任原生的智能体操作系统，将智能体生命周期包裹在由身份、感知、认知和执行四大支柱组成的强制性、不可绕过的安全执行边界中，使安全成为一等设计约束。

    

    现代AI智能体经常跨越信任边界：它们摄入不可信的内容，将其与特权指令相结合，将中间信念持久化到长期记忆中，并调用特权工具。这创造了一个攻击面，恶意载荷可以通过模型输入进入并引发有害的工具操作。然而，当前的治理技术栈仍然是应用层中间件，与其所监控的智能体共享同一进程信任边界。我们认为，智能体需要一个操作系统基底，为身份、输入中介、记忆治理和执行控制提供强制性、不可绕过的服务。我们提出了AgentKernel，一个以“安全必须作为一等设计约束”为前提构建的信任原生智能体操作系统。AgentKernel将智能体生命周期包裹在一个强制性执行边界中，该边界由四大支柱组成：身份、感知、认知和执行。每个支柱将cla（摘要在此处截断）

    arXiv:2609.29647v1 Announce Type: cross  Abstract: Modern AI agents routinely cross trust boundaries: they ingest untrusted content, combine it with privileged instructions, persist intermediate beliefs in long-term memory, and invoke privileged tools. This creates an attack surface in which malicious payloads can enter through model inputs and cause harmful tool actions. Yet current governance stacks remain application-level middleware that share a process trust boundary with the agents they monitor. We argue that agents need an operating-system substrate providing mandatory, non-bypassable services for identity, input mediation, memory governance, and execution control.   We introduce AgentKernel, a trust-native agent operating system built around the premise that security must be a first-class design constraint. AgentKernel wraps the agent lifecycle in a mandatory enforcement boundary organized into four pillars: Identity, Perception, Cognition, and Execution. Each pillar adapts cla
    
[^99]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^100]: TTLab参加AlexandriaX-2026竞赛：面向阿拉伯语机器翻译错误跨度检测与分类的微调表层标注器

    TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification

    [https://arxiv.org/abs/2609.29633](https://arxiv.org/abs/2609.29633)

    该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。

    

    我们介绍了TTLab参加AlexandriaX-2026子任务3（阿拉伯语机器翻译错误跨度检测与分类）的提交系统。我们的系统将该任务构建为基于表层形式的词元级分类，并保留字符偏移量以确保与评估指标的精确对齐。为应对严重的标签不平衡问题，我们采用了带类别权重的焦点损失以及方言特定的解码阈值。在六个阿拉伯语预训练编码器中，MARBERTv2取得了最佳整体性能，在开发集和测试集上分别达到40.8和40.91的分数，在所有参赛队伍中排名第三。尽管我们的系统能够有效定位错误跨度，但稀有错误类型的分类仍然具有挑战性，这凸显了针对尾部类别进行数据增强的必要性。代码已在GitHub上开源。

    arXiv:2609.29633v1 Announce Type: cross  Abstract: We present TTLab's submission to the AlexandriaX-2026 Subtask~3 on Arabic MT error span detection and classification. Our system frames the task as token-level classification over surface forms, preserving character offsets to ensure exact alignment with the evaluation metric. To handle severe label imbalance, we employ a focal loss with class weighting and dialect-specific decoding thresholds. Among six Arabic pre-trained encoders, MARBERTv2 achieves the best overall performance of 40.8 and 40.91 on the development and test set, respectively, ranking $\nth{3}$ out of all participating teams. While our system localizes error spans effectively, classification of rare error types remains challenging, highlighting the need for data augmentation for tail categories. The code is available at ${\href{https://github.com/ENTAILab/arabic-dialectal-mt-error-span-detection}{\faGithub~ TTLab at AlexandriaX-2026}$
    
[^101]: 一种基于排序原型的流形感知主题建模方法

    A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes

    [https://arxiv.org/abs/2609.29630](https://arxiv.org/abs/2609.29630)

    MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。

    

    近期的主题模型利用预训练嵌入，但神经架构产生的潜在表示缺乏与具体文本的关联，而基于聚类的流水线只能在事后分配代表性文档，依赖于在 高维空间中因枢纽性和各向异性而失真的绝对距离。我们提出了MARETopic，这是一个无需训练的框架，将主题发现转化为基于排序的原型选择。在将嵌入投影到低维流形后，MARETopic构建编码序数邻域结构的排序列表。贪心算法精确选出K个范例文档（即真实的语料库文本），其邻域可覆盖整个语料库。两个变体共享这一准则。其中MARETopic_Corr利用查询性能预测器和秩相关性度量对候选进行评分，在类别最多的两个基准数据集上取得了Purity和NMI的最优结果，领先于神经主题模型和基于聚类的主题模型。

    arXiv:2609.29630v1 Announce Type: cross  Abstract: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MAR
    
[^102]: iCoder-27B：递归AI主导开发的前沿工业编程模型

    iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model

    [https://arxiv.org/abs/2609.29626](https://arxiv.org/abs/2609.29626)

    专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。

    

    递归AI，即AI在构建和改进AI的过程中扮演日益完整的角色，是“以AI研发AI”这一愿景的皇冠明珠。尽管递归自我开发对于小模型、有界任务和固定时间预算已经变得可行，但这一雄心更具深远意义的实现——即开发出一个可发布、具备前沿竞争力的模型——仍然极具挑战性。在这项工作中，我们探讨了最低需要多少人类参与才能让智能体开发出前沿模型。我们将人类输入集中于一个高密度、低频率的接口：专家将目标、阶段脚手架、权限边界和操作流程编码为可复用的研究技能，而智能体则负责实例化这些先验知识、选择实验、诊断结果并修订训练策略。在具有挑战性的工业编程领域，智能体进化数据并协调监督微调（SFT）、在策略自蒸馏和强化学习（摘要在此处截断）

    arXiv:2609.29626v1 Announce Type: new  Abstract: Recursive AI, the prospect of AI taking an increasingly complete role in building and improving AI, is a crown jewel of AI for AI. Although recursive self-development has become practical for small models, bounded tasks, and fixed time budgets, a more consequential realization of this ambition, i.e., developing a release-ready, frontier-competitive model, remains far more challenging. In this work, we ask how little human involvement is sufficient for an agent to develop a frontier model. We concentrate human input into a high-density, low-frequency interface: experts encode objectives, stage scaffolds, permission boundaries, and operating procedures as reusable research skills, while the agent instantiates these priors, selects experiments, diagnoses outcomes, and revises the training strategy. In the challenging domain of industrial coding, the agent evolves data and coordinates SFT, on-policy self-distillation, and reinforcement learn
    
[^103]: 一个小型MLA-SSM混合语言模型的探索性消融研究

    An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model

    [https://arxiv.org/abs/2609.29618](https://arxiv.org/abs/2609.29618)

    该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。

    

    我们报告了对TALH（Adaptive Latent Hybrid，自适应潜在混合模型）的一项探索性单种子消融实验。TALH是一个仅有解码器的语言模型，结合了并行的多头潜在注意力机制与自定义的循环状态空间分支。五个变体（每token估计活跃参数量在1.17亿至2.17亿之间）在FineWeb样本上从零开始训练，采用相同的优化步数和token数量。在这一特定设置下，移除SSM分支会导致验证困惑度的最大退化（仅MLA模型PPL为315），而移除MLA的影响则小得多（仅SSM模型PPL为239）。密集FFN混合模型取得了231的PPL，而测试的top-2三值MoE混合模型为240，但后者可节省3.87 GB的峰值训练内存。我们还保留了一项初步的Apple M3计时观察：在五个未优化的实现中，仅MLA模型在512至2,048个提示token范围内的首token生成时间曲线最为平坦，尽管密集Transformer的速度要快得多……

    arXiv:2609.29618v1 Announce Type: cross  Abstract: We report an exploratory, single-seed ablation of TALH (Adaptive Latent Hybrid), a decoder-only language model with parallel Multi-head Latent Attention (MLA) and a custom recurrent state-space (SSM) branch. Five variants, spanning 117--217M estimated active parameters per token, are trained from scratch on a FineWeb sample for the same number of optimisation steps and tokens. In this specific setup, removing the SSM branch gives the largest degradation in validation perplexity (MLA-only PPL 315), whereas removing MLA has a much smaller effect (SSM-only PPL 239). A dense-FFN hybrid obtains PPL 231, compared with 240 for the tested top-2 ternary-MoE hybrid, while using 3.87 GB less peak training memory. We also preserve a preliminary Apple M3 timing observation: among the five unoptimised implementations, MLA-only has the flattest measured time-to-first-token curve from 512 to 2,048 prompt tokens, although the dense Transformer is much 
    
[^104]: 基于证据驱动的恶性黑色素瘤鉴别诊断

    Evidence-Driven Differential Diagnosis of Malignant Melanoma

    [https://arxiv.org/abs/2609.29613](https://arxiv.org/abs/2609.29613)

    提出了一种多层次证据驱动的恶性黑色素瘤鉴别诊断框架，通过解剖部位感知的掩码transformer建模患者所有病灶及其发病部位的上下文，并结合可学习的人口统计学嵌入捕捉患者元数据，使诊断特异性分别提升17.15%和7.14%。

    

    我们提出了一种用于恶性黑色素瘤鉴别诊断的模块化、多层次框架。该框架整合了病灶、患者和人群三个层面的上下文信息与证据，实现了各层级的决策。我们引入了一种解剖部位感知的掩码transformer，通过考虑患者体内数量可变的所有病灶及其发病部位，有效建模患者上下文。此外，我们通过可学习的人口统计学嵌入纳入患者元数据，以捕捉人群统计特征。通过大量实验，我们探讨了特定信息对决策过程的影响，并考察了考虑不同类型信息时的指标权衡。在SIIM-ISIC 2020数据集上的验证结果表明，加入包含位置的病灶上下文和元数据分别使特异性提高了17.15%和7.14%，同时提升了……

    arXiv:2609.29613v1 Announce Type: cross  Abstract: We present a modular and multi-level framework for the differential diagnosis of malignant melanoma. Our framework integrates contextual information and evidence at the lesion, patient, and population levels, enabling decision-making at each level. We introduce an anatomic-site aware masked transformer, which effectively models the patient context by considering all lesions in a patient, which can be variable in count, and their site of incidence. Additionally, we incorporate patient metadata via learnable demographics embeddings to capture population statistics. Through extensive experiments, we explore the influence of specific information on the decision-making process and examine the tradeoff in metrics when considering different types of information. Validation results using the SIIM-ISIC 2020 dataset indicate including the lesion context with location and metadata improves specificity by 17.15% and 7.14%, respectively, while enha
    
[^105]: PEEL：面向CT成像可辨识不确定性的物理赋能证据学习

    PEEL: Physics-Enabled Evidential Learning for Identifiable Uncertainty in CT Imaging

    [https://arxiv.org/abs/2609.29599](https://arxiv.org/abs/2609.29599)

    该论文提出PEEL方法，通过独立物理测量与蒙特卡洛噪声教师标签识别NIG证据回归中Student-t似然无法辨识的参数，实现了仅凭单张含噪CT图像即可进行可辨识的不确定性量化。

    

    arXiv:2609.29599v1 公告类型：新论文 摘要：正态逆伽马（NIG）回归无法从其边缘Student-t似然中实现唯一辨识：该似然仅能确定四个NIG参数中的三个组合，并沿一维纤维保持恒定。我们利用独立的物理测量来识别这一纤维。作为初步实现，重建网络接收单张含噪的滤波反投影（FBP）图像，首先仅通过Student-t负对数似然进行训练，以估计三个可辨识坐标（gamma, alpha, c）。随后冻结该网络；通过其重建输出传播的多次物理噪声实现构成了输出域偶然方差（aleatoric variance）的蒙特卡洛（MC）教师标签。附加在冻结特征上的偶然性头学习该标签，之后通过代数方法恢复出剩余的参数。在五个光子水平下的30个保留模拟物体上，单张图像预测获得了0.832-0.9的汇总Spearman相关性。

    arXiv:2609.29599v1 Announce Type: new  Abstract: Normal-inverse-gamma (NIG) regression is not uniquely identifiable from its marginal Student-t likelihood: the likelihood determines three combinations of four NIG parameters and is constant along a one-dimensional fiber. We identify that fiber using independent physical measurement. As an initial embodiment, a reconstruction network receives one noisy filtered-backprojection (FBP) image and is first trained only by Student-t negative log-likelihood to estimate the three identifiable coordinates (gamma, alpha, c). The network is then frozen; repeated physical-noise realizations propagated through its reconstruction output form a Monte Carlo (MC) teacher label for output-domain aleatoric variance. An aleatoric head attached to frozen features learns this label, after which (beta, nu) are recovered algebraically. On 30 held-out simulated objects at five photon levels, one-image predictions achieved pooled Spearman correlations of 0.832-0.9
    
[^106]: QINA：面向预训练视觉模型的量子启发非线性适配器

    QINA: Quantum-Inspired Nonlinear Adapters for Pretrained Vision Models

    [https://arxiv.org/abs/2609.29592](https://arxiv.org/abs/2609.29592)

    提出量子启发非线性适配器QINA，通过可学习的三角函数特征提升与有界非线性聚合，在主干冻结、数据有限的条件下对预训练视觉表征进行谱重塑，且几乎不增加参数量。

    

    在数据有限且主干网络冻结的约束下，适配大型预训练视觉模型仍然是迁移学习中的核心挑战。尽管轻量级适配器和参数高效微调方法已被广泛采用，但大多数方法依赖于通用的多层感知机或低秩线性更新，对特征变换的谱结构与几何结构的控制能力有限。我们研究了结构化的非线性特征提升能否在冻结条件下改善表征对齐。我们提出了量子启发非线性适配器（QINA），这是一种紧凑的模块，先执行可学习的三角函数特征提升，再进行有界的非线性聚合。该设计引入了具有显式范数相关Lipschitz界的结构化振荡基函数，能够在不扩大感受野、不显著增加参数量的前提下对预训练表征进行谱重塑。重要的是，（摘要在此处被截断）

    arXiv:2609.29592v1 Announce Type: cross  Abstract: Adapting large pretrained vision models under limited data and frozen-backbone constraints remains a central challenge in transfer learning. While lightweight adapters and parameter-efficient fine-tuning methods are widely adopted, most rely on generic multilayer perceptrons or low-rank linear updates, offering limited control over the spectral and geometric structure of feature transformations. We investigate whether structured nonlinear feature lifting can improve representational alignment in frozen regimes. We introduce Quantum-Inspired Nonlinear Adapters (QINA), compact modules that perform learnable trigonometric feature lifting followed by bounded nonlinear aggregation. The design induces structured oscillatory basis functions with an explicit norm-dependent Lipschitz bound, enabling spectral reshaping of pretrained representations without increasing the receptive field or significantly expanding parameter count. Importantly, th
    
[^107]: CATCH：基于条件哈尔扩散的反事实解剖组织修复

    CATCH: Counterfactual Anatomical Tissue Inpainting with Conditional Haar Diffusion

    [https://arxiv.org/abs/2609.29591](https://arxiv.org/abs/2609.29591)

    该论文提出CATCH——一种在可逆哈尔小波域中运行的条件3D扩散模型，通过挖空图像条件、带符号掩膜和聚焦空洞的损失函数，在保留已观测脑部解剖结构的同时生成合理的无肿瘤组织，用于BraTS局部合成任务。

    

    BraTS局部合成任务旨在将T1加权脑部MRI中被掩膜遮盖的区域替换为合理的无肿瘤组织，同时保留已观测的解剖结构。我们提出了CATCH，一种在可逆哈尔小波域中运行的条件3D扩散模型。其去噪器接收带噪声的目标系数、挖空图像的系数以及带符号的掩膜；排除肿瘤的小波重建和聚焦空洞的损失函数用于指导训练，硬合成则保留已观测的体素。我们比较了固定掩膜、肿瘤成分增强，以及由肿瘤衍生掩膜、不规则斑块掩膜和椭球形掩膜组成的加权混合掩膜三种方案。在25个开发病例中，五个预先指定的病例用于选择各方案的模型检查点及其全部25种轨迹聚合结果；另一个独立的75例内部数据集用于比较冻结的推理流水线，并选出加权混合掩膜方案提交给组织方评估。五轨迹平均方法在内部数据上取得的SSIM/PSNR/MSE（均值±标准差）分别为0.80±0.13、19.18±1.80dB和0.010±0.005。

    arXiv:2609.29591v1 Announce Type: cross  Abstract: BraTS local synthesis replaces masked regions in T1-weighted brain MRI with plausible tumor-free tissue while preserving observed anatomy. We present CATCH, conditional 3D diffusion in an invertible Haar-wavelet domain. Its denoiser receives noisy target coefficients, voided-image coefficients, and a signed mask; tumor-excluded wavelet reconstruction and a hole-focused loss guide training, and hard compositing preserves observed voxels. We compare fixed masks, tumor-component augmentation, and a weighted mixture of tumor-derived, irregular-blob, and ellipsoidal masks. Of 25 development cases, five prespecified cases select each arm's checkpoint and all 25 of their trajectory aggregations; a separate 75-case internal set compares the frozen pipelines and selects a weighted mixture for organizer evaluation. Five-trajectory averaging yielded internal SSIM/PSNR/MSE (mean$\pm$SD) of $0.80\pm0.13$, $19.18\pm1.80$dB, and $0.010\pm0.005$. As t
    
[^108]: 序列知识编辑破坏了模型辨别好坏证据的能力，却不损失其准确率

    Sequential knowledge editing breaks a model's ability to tell good evidence from bad, without costing it accuracy

    [https://arxiv.org/abs/2609.29587](https://arxiv.org/abs/2609.29587)

    序列知识编辑会在不损害模型在MMLU等基准上准确率的情况下，显著削弱模型在未编辑事实上辨别检索证据好坏的仲裁能力，导致选择性预测性能退化。

    

    知识编辑通常从三个方面进行评估：被编辑的事实是否已改变、改写表述是否随之更新、以及无关回答是否保持不变。一个模型可以通过所有这三项测试，却仍然失去一种它们都未曾衡量的能力：在从未被编辑的事实上，判断应该相信哪些检索到的文档。我们在编辑前后测量模型对其记忆答案相对于注入段落所断言答案所分配的对数几率，同时保持查询、段落和两个候选字符串不变。我们最干净的实验分支是一个保守调优的LoRA：在Qwen2.5-7B-Instruct上进行1,000次序列编辑后，MMLU保持不变（精确到小数点后四位），然而在未触及事实上的仲裁量分布跨度下降了36%。选择性预测也随之退化。风险-覆盖率曲线下面积上升了0.107，而在相同MMLU水平下，范数匹配的扰动仅使其上升0.005，且模型在其最有信心的四分之一仲裁上的错误率也有所上升

    arXiv:2609.29587v1 Announce Type: new  Abstract: Knowledge editing is evaluated on whether the edited fact changed, whether paraphrases follow, and whether unrelated answers stayed put. A model can pass all three and still lose something none of them measures: the ability to decide, on facts that were never edited, which retrieved documents to believe.   We score the log odds a model assigns to its remembered answer against the answer an injected passage asserts, before and after editing, holding the query, the passage and both candidate strings fixed. Our cleanest arm is a conservatively tuned LoRA: after 1,000 sequential edits on Qwen2.5-7B-Instruct it leaves MMLU unchanged to four decimal places, yet the spread of the arbitration quantity across untouched facts falls by 36%. Selective prediction degrades with it. Area under the risk-coverage curve rises by 0.107, against 0.005 for a norm-matched perturbation at the same MMLU, and error on the model's most confident quarter of arbitr
    
[^109]: PartHackBench：用于部分得分工具智能体评估的认证等进度压力测试

    PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation

    [https://arxiv.org/abs/2609.29578](https://arxiv.org/abs/2609.29578)

    论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。

    

    长时程工具智能体常常在不达到最终成功的情况下取得有用的进展，这促使了部分得分评估的出现。然而，评估者可能会奖励那些暂时的、后来被逆转的、或无法归因于被评估智能体的里程碑。当一条诚实轨迹与一条得分更高的对抗性轨迹进行比较时，如果后者确实取得了更多真实进展，则这种比较是不确定的。我们提出了PartHackBench，这是一种消除该混淆因素的受控方法。一个私有认证器只有在两条轨迹在当前状态谓词满足情况和标准化智能体归因方面逐组件完全匹配时，才接受该轨迹对；得分膨胀（定义为f(A) - f(H)）仅在之后进行测量。在PB-CSTE的18个密封保留任务中，冻结的历史目标运行为15个任务生成了匹配的对抗样本。历史归因产生了平均0.252的得分膨胀，条件攻击成功率为10/15，端到端收益为10/18，且未检测到14个严格回滚中的任何一个。

    arXiv:2609.29578v1 Announce Type: new  Abstract: Long-horizon tool agents often make useful progress without reaching terminal success, motivating partial-credit evaluation. Yet evaluators may reward milestones that were temporary, later reversed, or not attributable to the evaluated agent. Comparing an honest trajectory with a higher-scoring adversarial one is inconclusive if the latter made more genuine progress. We introduce PartHackBench, a controlled methodology that removes this confound. A private certifier admits a pair only when its trajectories match component-wise in both current-state predicate satisfaction and standardized agent attribution; score inflation, defined as f(A) - f(H), is measured only afterward. In 18 sealed held-out tasks in PB-CSTE, the frozen historical-target run produced matched adversaries for 15 tasks. Historical credit yielded mean inflation of .252, conditional attack success of 10/15, end-to-end yield of 10/18, and detected none of 14 strict rollbac
    
[^110]: 推理总是有用的吗？重新思考通用多模态嵌入中的推理效用

    Is Reasoning Always Useful? Rethinking Reasoning Utility in Universal Multimodal Embeddings

    [https://arxiv.org/abs/2609.29560](https://arxiv.org/abs/2609.29560)

    该研究揭示了多模态嵌入中的推理并非总是有益——存在推理反而拉近难负样本的“伪有益”案例——并据此提出 SURE 效用路由器以自适应地利用推理效用，提升了 UME-R1-7B 的检索性能。

    

    推理增强的通用多模态嵌入（UME）能够提升异构检索性能，但看似合理的推理过程并不一定带来有区分度的排序结果。我们通过对比 UME-R1——一种最先进的推理型通用多模态嵌入方法——中的判别式（DISC）分支与推理驱动的生成式（GEN）分支，来研究这一差距。我们将推理效用分解为正样本增益、难负样本增益以及二者的间隔差异。结果显示，56.6% 的情况下正样本相似度有所提升，但其中 15.7% 属于“伪有益”案例，即推理反而使难负样本更加接近查询。局部邻域分析与词元归因诊断揭示了其中的原因：推理往往使检索到的邻域“去凝聚化”，而真正有效需要与判别方向对齐的表示移动；同时，影响较大的思维链（CoT）词元常常编码的是正样本与难负样本共享的证据。基于这些诊断，我们提出了 SURE（Score-structure Utility Router for Embeddings，面向嵌入的分数结构效用路由器），该方法将 UME-R1-7B 的性能提升了 1……

    arXiv:2609.29560v1 Announce Type: new  Abstract: Reasoning-enhanced universal multimodal embeddings (UME) improve heterogeneous retrieval, but plausible rationales do not necessarily produce discriminative rankings. We study this gap by comparing the discriminative (DISC) and reasoning-driven generative (GEN) branches of UME-R1, a state-of-the-art reasoning UME method. We decompose reasoning utility into positive-target gain, hard-negative gain, and their margin difference. Positive similarity increases for 56.6%, but 15.7% are false-helpful cases where reasoning moves hard negatives closer even more. Local-neighborhood and token-attribution diagnostics suggest why: reasoning often de-condenses retrieved neighborhoods, but utility requires separator-aligned movement, while influential CoT tokens frequently encode evidence shared by positives and hard negatives. Motivated by these diagnostics, we propose SURE (Score-structure Utility Router for Embeddings), which improves UME-R1-7B by 1
    
[^111]: 跨模态情感理解：一种用于对话情感识别的Transformer-GAT方法

    Cross-Modal Emotion Understanding: A Transformer-GAT Approach for Dialogue Emotion Recognition

    [https://arxiv.org/abs/2609.29556](https://arxiv.org/abs/2609.29556)

    该论文提出了一种结合Transformer与图注意力网络的混合框架Transformer-GAT，通过Transformer捕获全局语义信息、图注意力网络建模模态间细粒度关系来实现跨模态对话情感识别，在IEMOCAP和MELD数据集上分别取得72.45%和77.37%的加权F1分数，超越了现有最先进方法。

    

    多模态情感识别是情感计算领域的一个关键研究方向，在情感分析、智能客服和人机交互中具有广泛应用。然而，现有方法通常依赖于单一模态特征或简单的多模态融合，未能捕捉全局与局部上下文之间的协同作用，从而限制了模型的性能和情感理解能力。为应对这一挑战，我们提出了Transformer-GAT，一种结合Transformer与图注意力网络的混合框架，以实现跨模态情感理解。其中，Transformer用于捕获全局语义信息，而图注意力网络用于建模模态之间的细粒度关系，从而增强情感特征的表示。在IEMOCAP和MELD数据集上的实验表明，我们的模型分别取得了72.45%和77.37%的加权F1分数，优于当前最先进的方法。

    arXiv:2609.29556v1 Announce Type: new  Abstract: Multimodal emotion recognition is a key research area in affective computing, with applications in sentiment analysis, intelligent customer service, and human-computer interaction. However, existing methods often rely on single-modal features or simple multimodal fusion, failing to capture the synergy between global and local contexts, which limits model performance and emotion understanding. To address this challenge, we propose Transformer-GAT, a hybrid framework that combines Transformer and the Graph Attention Network to enable cross-modal emotion understanding. The Transformer is used to capture global semantic information, while the Graph Attention Network is employed to model fine-grained relationships between modalities, thereby enhancing the representation of emotional features. Experiments on the IEMOCAP and MELD datasets show that our model achieves weighted F1 scores of 72.45% and 77.37%, outperforming state-of-the-art method
    
[^112]: UNWIND：无需时间窗口化的任意长度面部视频压力检测

    UNWIND: Any-Length Facial Video for Stress Detection without Temporal Windowing

    [https://arxiv.org/abs/2609.29553](https://arxiv.org/abs/2609.29553)

    UNWIND框架通过将面部视频的时间维度折叠进通道维度并采用非对称注意力架构，实现对完整录像的单次输入压力检测，无需时间窗口化或外部分割。

    

    基于面部视频的自动压力识别为情感监测提供了一种非接触式方法。然而，大多数现有的基于视频的方法在执行分类之前，会将完整的录像分割成较短的时间片段。这种分割需要额外决定片段时长、重叠程度和预测聚合方式，并可能限制模型利用分布在整段录像中的信息。我们提出了UNWIND，一个用于压力检测的面部视频框架，它将完整录像作为单一模型输入进行分析，从而无需时间窗口化或外部分割。UNWIND通过将视频的时间维度折叠到二维空间表示的通道维度中来重新组织视频，随后通过统一的非对称注意力架构进行处理。在时间步长τ=1的情况下，该框架能够处理完整的120秒录像……

    arXiv:2609.29553v1 Announce Type: cross  Abstract: Automatic stress recognition from facial video provides a non-contact approach for affective monitoring. However, most existing video-based methods divide complete recordings into shorter temporal segments before performing classification. Such segmentation requires additional decisions concerning segment duration, overlap, and prediction aggregation, and may restrict the model from exploiting information distributed across the entire recording. We introduce UNWIND, a facial-video framework for stress detection that analyzes a complete recording as a single model input, eliminating the need for temporal windowing or external segmentation. UNWIND reorganizes the video by folding its temporal dimension into the channel dimension of a two-dimensional spatial representation, which is subsequently processed through a unified asymmetric-attention architecture. With a temporal stride of $\tau=1$, the framework processes the entire $120$-secon
    
[^113]: HiPACE：稀疏自编码器中特征吸收的分层相边界分析与受控评估

    HiPACE: Hierarchical Phase-Boundary Analysis and Controlled Evaluation of Feature Absorption in Sparse Autoencoders

    [https://arxiv.org/abs/2609.29551](https://arxiv.org/abs/2609.29551)

    本文针对稀疏自编码器中的特征吸收现象首次推导出闭式相边界 λ_c(k,α)=α²k/(k-1)，并据此提出 HiPACE 评估协议，在真实 SAE 上系统检验父概念与子概念何时坍缩到成本最优的共享方向。

    

    稀疏自编码器（SAE）将大语言模型的激活分解为稀疏的字典原子，使得每个不同的概念都能拥有自己的特征。然而，一种反复出现的行为使这一前提变得复杂：特征吸收（feature absorption），即父概念与其子概念——例如“水果”与{苹果、香蕉、梨}——坍缩到一个共享的家族方向上。先前的工作从经验层面记录了吸收现象；缺失的是对“共享方向何时是活跃语义家族的成本最优表示”的闭式预测。本文填补了这一空白。对于一个具有 k 个活跃子概念和残差尺度 α 的分层伯努利生成器，L0 惩罚重建目标存在一个闭式相边界 λ_c(k,α) = α²k/(k-1)：在该边界之上，纯父概念吸收严格地比纯子概念编码的成本更低。基于这一边界，我们提出了 HiPACE，一种评估协议，用于在真实的稀疏自编码器中检验该边界带来的结构性后果。

    arXiv:2609.29551v1 Announce Type: new  Abstract: Sparse autoencoders (SAEs) decompose LLM activations into sparse dictionary atoms, so that each distinct concept gets its own feature. One recurring behavior complicates this premise: feature absorption, in which a parent concept and its children--fruit and {apple, banana, pear}, say--collapse into a shared family direction. Prior work documents absorption empirically; missing is a closed-form prediction of when the shared direction is the cost-optimal representation of an active semantic family. This paper closes that gap. For a hierarchical Bernoulli generator with $k$ active children and residual scale $\alpha$, the $L_0$-penalized reconstruction objective admits a closed-form phase boundary $\lambda_c(k,\alpha)=\alpha^2 k/(k-1)$: above it, pure parent absorption is strictly cheaper than pure child coding. Building on this boundary, we introduce HiPACE, an evaluation protocol that tests the boundary's structural consequence in real SA
    
[^114]: 当智能体在无人监督时行动：智能体AI中的监督弱化悖论

    When Agents Act Unwatched: The Reduced-Supervision Paradox in Agentic AI

    [https://arxiv.org/abs/2609.29547](https://arxiv.org/abs/2609.29547)

    本文提出“监督弱化悖论”这一概念，指出当用户停止监督时，验证机制并未消失而是转移至运行时基础设施，但通过对63个材料的审计发现，智能体的行动面极易被重构，而检查点、验证器独立性、恢复和可争议性等问责机制却极少公开可见，形成了显著的问责倒置。

    

    智能体AI（Agentic AI）以其一个简单的承诺来推销自己：当用户停止关注时，系统仍会持续行动。这一承诺造成了“问责倒置”。随着逐步监督的退场，验证并未消失；它转移到了运行时基础设施中——这些基础设施负责定义权限、记录行动、中断执行、检查结果并支持修复。我们将此称为“监督弱化悖论”。通过对63个材料的审计，我们考察了该悖论在46篇研究论文以及17份工程、文档、安全和治理来源中的公开可见性。我们发现，智能体的行动面远比为其行为承担责任所需的机制更容易被重构。工具中介和监控轨迹分别在40个和37个材料中清晰可见，而检查点设置仅在6个、验证器独立性在4个、恢复机制在2个、可争议性仅在1个材料中清晰可见。三条行动路径说明了这种失衡为何重要。一个代码仓库操作……（原文摘要在此处截断）

    arXiv:2609.29547v1 Announce Type: cross  Abstract: Agentic AI is sold on a simple promise: the system keeps acting when the user stops watching. That promise creates an accountability inversion. As stepwise supervision recedes, verification does not disappear; it moves into the runtime infrastructure that defines authority, records action, interrupts execution, checks outcomes, and supports repair. We call this the reduced-supervision paradox. Using a 63-artifact audit, we examine its public visibility across 46 research papers and 17 engineering, documentation, security, and governance sources. We find that agents' action surfaces are far easier to reconstruct than the mechanisms needed to answer for their actions. Tool mediation and monitoring traces were clearly visible in 40 and 37 artifacts, whereas checkpoint placement was clearly visible in 6, validator independence in 4, recovery in 2, and contestability in 1. Three action paths show why this imbalance matters. A repository pat
    
[^115]: 广义图变分自编码器：有界散度控制后验坍缩

    Generalized Graph Variational Autoencoders: Bounded Divergences Control Posterior Collapse

    [https://arxiv.org/abs/2609.29546](https://arxiv.org/abs/2609.29546)

    本文提出广义图变分自编码器（GGVA），用Rényi-Tsallis散度族替代KL散度，并揭示散度的有界性（而非其阶数）才是控制后验坍缩的关键性质。

    

    变分图自编码器（VGAE）使用Kullback-Leibler散度将其后验分布向先验分布进行正则化，这一选择继承自变分自编码器而非经过论证。我们提出了广义图变分自编码器（GGVA），它将该正则项替换为Rényi-Tsallis散度族中任意阶数q的成员，而保持模型的其他部分不变。这两种散度对于对角高斯分布都具有闭式解，并且在q→1时都精确恢复为KL散度，因此VGAE是我们自己模型在q=1时的一个分支而非独立的基线，任何测得的差异都可归因于单个标量。我们的分析表明，有界性而非阶数才是起作用的性质：当q<1时，Tsallis散度的上界为1/(1-q)，且与潜变量维度无关，而相同阶数的KL散度和Rényi散度都是无界的。在跨越三个合成族的十个图上，

    arXiv:2609.29546v1 Announce Type: cross  Abstract: The variational graph autoencoder (VGAE) regularizes its posterior toward the prior with the Kullback-Leibler divergence, a choice inherited from the variational autoencoder rather than argued for. We introduce the generalized graph variational autoencoder (GGVA), which replaces that term with any member of the R\'enyi-Tsallis family of order $q$ while leaving every other part of the model untouched. Both members admit closed forms for diagonal Gaussians and both recover the KL exactly as $q \to 1$, so the VGAE is the $q=1$ arm of our own model rather than a separate baseline, and any measured difference is attributable to a single scalar. Our analysis identifies boundedness, not the order, as the operative property: for $q<1$ the Tsallis divergence is bounded above by $1/(1-q)$, independently of the latent width, whereas the KL and the R\'enyi divergence of the same order are unbounded. On ten graphs spanning three synthetic families,
    
[^116]: ERRAND：智能体记忆的预算化维护

    ERRAND: Budgeted Maintenance of Agent Memory

    [https://arxiv.org/abs/2609.29545](https://arxiv.org/abs/2609.29545)

    ERRAND将智能体记忆的重新验证建模为一项按价值定价的“差事”，通过单峰差事指数在有限行动预算下决定何时复核过时知识，从而在漂移环境中以最小成本维持记忆的时效性。

    

    已部署的智能体依赖“交接知识”运行：一个冻结的策略参照一份在数据流开始之前写就的、由整理后条目构成的简报。然而世界在变动，而知识库却静止不动：路径关闭、标志改变、价格区间移动；每一条目在交接时都是真实的，其失败源于过时而非无知。我们提出ERRAND，它将重新验证视为一项计价的差事：一次复核与它所保护的任务争夺同样稀缺的行动资源，只有当消除一个疑虑的每行动价值超过运行中的“工资”（成本基准）时，这项复核才会获得资助。差事指数是单峰的，在信念的两端趋于消失，因此任何方向上的确定性都不产生成本；沿途的免费回执维护着路径上的知识，而修复操作只写入新版本，从不删除。在两个漂移的工具使用环境中，在相同的行动预算下，ERRAND在预先登记的评估标准上超越了所有非神谕策略，在每种设置中均为首要最优，并在每个受约束的预算下保持条件最优。

    arXiv:2609.29545v1 Announce Type: new  Abstract: Deployed agents run on handed-over knowledge: a frozen policy consults a briefing of consolidated items written before the stream begins. The world then moves while the store stands still: paths close, flags change, price bands move; every item was true at handover, and the failure is staleness, not ignorance. We introduce ERRAND, which treats revalidation as a priced errand: a recheck competes with the task it protects for the same scarce actions, funded only when the value per action of resolving a doubt clears a running wage. The errand index is single-peaked, vanishing at both ends of belief, so certainty in either direction costs nothing; free en-route receipts maintain on-path knowledge, and repair writes a version, never a deletion. Under equal action budgets in two drifting tool-use worlds, ERRAND clears every non-oracle policy on the preregistered calibers, primary in every setting and conditional at every binding budget, leadin
    
[^117]: 物理智能体的安全技能退役

    Safe Skill Retirement for Physical Agents

    [https://arxiv.org/abs/2609.29543](https://arxiv.org/abs/2609.29543)

    该论文提出“匹配权限反事实”评估方法与“双门退役证书”机制，确保在删减智能体技能中看似冗余的指令时不会意外移除休眠的安全执行条件，从而防止未授权的物理和隐私敏感效应。

    

    智能体技能将程序性指导与执行条件捆绑在一起，这些执行条件控制着权限、用户同意以及实时环境状态。当模型能力提升时，维护者会删减那些在授权基准任务上看似冗余的指令。然而，授权的维护测试可能会使处于休眠状态的安全条件未被测试到。这种不匹配在物理效应和隐私敏感效应方面造成了一个未被测量的支持缺口。我们引入了“匹配权限反事实”方法，即在保持所请求的操作、工具参数和预期效果不变的同时，系统地改变单个控制谓词。我们通过“双门退役证书”将此评估形式化，要求候选删减在声明的余量范围内保持授权效用的同时，产生零个未授权的受保护效应。在涵盖四种前沿模型和本地模型配置、十二个技能包（共2,592个评估单元）的受控实验中，任务-

    arXiv:2609.29543v1 Announce Type: new  Abstract: Agent skills bundle procedural guidance with execution conditions governing authority, user consent, and live environment state. When model capabilities advance, maintainers prune instructions that appear redundant on authorized benchmark tasks. However, authorized maintenance tests can leave dormant safety conditions untested. This mismatch creates an unmeasured support gap over physical and privacy-sensitive effects. We introduce matched authority counterfactuals that hold the requested action, tool parameters, and intended effect fixed while systematically varying a single governing predicate. We formalize this evaluation via a two-gate retirement certificate requiring a candidate reduction to preserve authorized utility within a declared margin while producing zero unauthorized protected effects. In controlled experiments spanning four frontier and local model configurations across twelve skill bundles (2,592 evaluation cells), task-
    
[^118]: GeoRefer-Bench：从像素指代到可验证地理空间推理的基准

    GeoRefer-Bench: A Benchmark from Referring Pixels to Verifiable Geospatial Reasoning

    [https://arxiv.org/abs/2609.29541](https://arxiv.org/abs/2609.29541)

    提出了GeoRefer-Bench基准，通过可执行逻辑形式查询和精确查询成功率（EQS）指标，实现了对俯视图像地理空间指代分割中空间关系理解能力的可验证评估。

    

    arXiv:2609.29541v1 公告类型：交叉 摘要：俯视图像中的指代分割本质上是关系性的：查询可能要求“道路北侧的建筑物”或“距离居民区最近的池塘”，因此正确的指代对象可能包含一个物体、多个物体或没有物体。现有基准主要评估掩码重叠度，无法验证模型是否真正解析了所述的空间关系。我们提出了GeoRefer-Bench，一个用于可验证地理空间指代分割的基准。每个查询由基于度量场景图的可执行逻辑形式表示，预测结果通过精确查询成功率进行评估，只有当返回的实例集与查询所表示的集合完全匹配时，该指标才被满足。GeoRefer-Bench包含700个完整的2048×2048无人机场景（2.94 Gpx），地面采样距离为12.5厘米和25厘米，涵盖26,217个实例、142,796个空间关系以及跨越五个推理级别的20,916个可执行查询。该基准还包括三个p...（原文在此处截断）

    arXiv:2609.29541v1 Announce Type: cross  Abstract: Referring segmentation in overhead imagery is inherently relational: a query may ask for the buildings north of the road or the pond closest to a residential area, so the correct referent can contain one object, several objects, or none. Existing benchmarks mainly score mask overlap, which cannot verify whether a model actually resolved the stated spatial relation. We introduce GeoRefer-Bench, a benchmark for verifiable geospatial referring segmentation. Each query is represented by an executable logical form over a metric scene graph, and predictions are evaluated with Exact Query Success (EQS), which is satisfied only when the returned instance set exactly matches the set denoted by the query. GeoRefer-Bench contains 700 whole 2048x2048 UAV scenes (2.94 Gpx) at 12.5 and 25 cm ground sampling distance, 26,217 instances, 142,796 spatial relations, and 20,916 executable queries spanning five reasoning levels. It further includes three p
    
[^119]: 用于胸部X光器械推理的临床知识图谱

    Clinical Knowledge Graphs for Chest X-Ray Device Reasoning

    [https://arxiv.org/abs/2609.29536](https://arxiv.org/abs/2609.29536)

    该论文提出了一种不确定性感知的临床知识图谱，将胸部X光中器械位置的图像证据、尖端估计、放置评估与报告事件等多源信息表示为相互连接的节点，并在RANZCR CLiP数据集上验证了其完整保留证据细节的能力。

    

    胸部X光片常规用于验证导管、管路及其他支持器械的位置。现有图像模型通常仅返回标签或分割结果，而报告处理系统则在不获取图像几何信息的情况下对文本进行结构化处理。我们提出了一种不确定性感知的临床知识图谱，将器械实例、尖端位置估计、放置评估、来源溯源、报告事件及时间关联表示为相互独立但彼此连接的证据。我们使用完整的RANZCR CLiP测试存档中保存的预测结果来评估已实现的视觉图谱层，该存档包含来自3,255名患者的30,083项检查研究，分布于五个互不重叠的外部折叠中。图谱构建器生成了914,632个B7证据节点和884,549个类型化关系。所有118,647个B7预测器械节点均保留了尖端协方差、放置概率、片段来源和片段计数信息，而直接的B2基线方法则不保留任何这些字段。我们进一步……（摘要截断）

    arXiv:2609.29536v1 Announce Type: new  Abstract: Chest radiographs are routinely used to verify the position of catheters, tubes, and other support devices. Existing image models often return labels or segmentations, while report-processing systems structure text without access to image geometry. We present an uncertainty-aware clinical knowledge graph that represents device instances, tip estimates, placement assessments, provenance, report events, and temporal links as separate but connected evidence.   We evaluate the implemented visual graph layer using saved predictions from the complete RANZCR CLiP test archive, comprising 30,083 studies from 3,255 patients across five non-overlapping outer folds. The graph builder materializes 914,632 B7 evidence nodes and 884,549 typed relationships. All 118,647 B7 predicted-device nodes retain tip covariance, placement probabilities, fragment provenance, and fragment counts, whereas the direct B2 baseline retains none of these fields. We furth
    
[^120]: CoSWWA-YOLOv12：尺度不变的疟疾寄生虫微小目标检测与分割

    CoSWA-YOLOv12: Scale-Invariant Tiny Object Detection and Segmentation of Malaria Parasites

    [https://arxiv.org/abs/2609.29527](https://arxiv.org/abs/2609.29527)

    提出CoSWA-YOLOv12，其核心的协同尺度自适应Wasserstein分配机制按目标大小成反比地调节Wasserstein度量的应用强度，在提升疟原虫微小目标检测能力的同时避免损害较大目标的定位精度，实现了尺度不变的疟疾寄生虫检测与分割。

    

    自动化显微检查能够在低资源环境中扩大疟疾诊断的可及性，但最致命的疟原虫种类——恶性疟原虫——在其早期环状体阶段仅表现为宽度只有几十个像素的目标。这类微小目标普遍存在系统性漏检问题：基于重叠度的标签分配使它们难以获得正样本，而基于重叠度的边界框回归在其尺度下提供的梯度也十分微弱。归一化高斯Wasserstein距离（NWD）能够修复上述两个问题，但如果在同时含有比其大三四倍物体的整个载玻片上统一应用，它会削弱对较大物体的监督并损害其定位能力，因此即使微小目标类别的性能提升，整体精度也可能下降。我们提出了CoSWA-YOLOv12，这是一个紧凑的YOLOv12实例分割检测器，其核心的协同尺度自适应Wasserstein分配机制按与物体大小成反比的方式将Wasserstein处理路由给各目标，对于较大的物种则逐渐回归到标准分配方式。

    arXiv:2609.29527v1 Announce Type: cross  Abstract: Automated microscopy could widen access to malaria diagnosis in low-resource settings, but the deadliest species, P. falciparum, presents in its early ring stage as an object only a few tens of pixels wide. Such tiny targets are systematically under-detected: overlap-based label assignment starves them of positive samples, and overlap-based box regression gives weak gradients at their scale. The Normalized Gaussian Wasserstein Distance (NWD) repairs both effects, but applied uniformly across a slide that also holds objects three to four times larger it loosens their supervision and erodes their localisation, so overall accuracy can fall even as the tiny class improves. We present CoSWA-YOLOv12, a compact YOLOv12 instance-segmentation detector whose core Cooperative Scale-adaptive Wasserstein Assignment routes the Wasserstein treatment to an object in inverse proportion to its size, tapering back to standard assignment for larger specie
    
[^121]: 陈旧并不意味着不安全：基础设施状态竞态下工具使用型LLM智能体的防护精确性

    Stale Does Not Mean Unsafe: Guard Precision for Tool-Using LLM Agents under Infrastructure State Races

    [https://arxiv.org/abs/2609.29522](https://arxiv.org/abs/2609.29522)

    该研究将基础设施状态竞态细分为失效性与无害两类，并利用确定性模拟器系统评估了不同粒度的提交时防护机制在保障工具使用型LLM智能体安全提交方面的精确性。

    

    使用工具的语言模型智能体日益频繁地修改调度器、数据管道、对象存储和访问控制系统。在智能体读取与提交之间，外部状态可能发生变化，但并非每一次变化都会使提交变得不安全。我们将破坏已声明安全谓词的“失效性竞态”与“保持谓词的竞态”及“无关竞态”区分开来，并探究运行时防护机制能以何种精确度区分它们。我们的确定性模拟器将可见状态与权威状态分离，并在四个领域的16个基础设施任务中注入五种非原子性失效机制；冻结的智能体提案无需LLM裁判，即可在每种控制器下进行反事实重放。我们在三个本地托管的量化模型家族（Qwen3-4B、Phi-4-mini、Gemma4-8B；单GPU上运行3,456条轨迹）上评估了三种提交时防护粒度（全局纪元、读集版本、语义提交谓词）、多级验证以及模型侧门控机制。

    arXiv:2609.29522v1 Announce Type: new  Abstract: Tool-using language-model agents increasingly mutate schedulers, data pipelines, object stores, and access-control systems. Between an agent's read and its commit, external state can change, but not every change makes the commit unsafe. We separate invalidating races, which break a declared safety predicate, from predicate-preserving and irrelevant races, and ask how precisely runtime guards distinguish them. Our deterministic simulator separates visible from authoritative state and injects five non-atomic failure mechanisms across 16 infrastructure tasks in four domains; frozen agent proposals are replayed counterfactually under every controller without an LLM judge. We evaluate three commit-time guard granularities (global epoch, read-set version, semantic commit predicate), multi-level verification, and model-side gates on three locally hosted quantized model families (Qwen3-4B, Phi-4-mini, Gemma4-8B; 3,456 trajectories on one GPU). A
    
[^122]: BiGraph-Diffuse：一种结合图结构检索、用于心理健康咨询的双向扩散语言模型

    BiGraph-Diffuse: A Bidirectional Diffusion Language Model with Graph-Structured Retrieval For Mental Health Counseling

    [https://arxiv.org/abs/2609.29519](https://arxiv.org/abs/2609.29519)

    提出了BiGraph-Diffuse——首个面向心理健康咨询的大规模双向扩散语言模型，结合图结构检索增强方法BiGraph-RAG，克服自回归模型无法修正早期解读、难以利用临床关系知识的缺陷，以更好捕捉来访者渐进式披露的深层情感。

    

    心理健康障碍影响着全球数亿人，然而获得专业心理咨询服务的机会仍然严重受限。人工智能驱动的对话系统提供了一种可扩展的替代方案，但现有模型面临两个根本性挑战。首先，它们缺乏捕捉情感表达层次性所需的双向理解能力，尤其是在渐进式披露的情况下——来访者往往只表现出表面症状，而隐藏更深层的创伤。自回归（AR）模型按顺序处理信息，当对话后期出现新证据时，无法修正早期的解释。其次，它们未能有效整合临床推理所依赖的关系知识。在本文中，我们提出了BiGraph-Diffuse，这是首个专为心理咨询领域量身定制的大规模扩散语言模型。我们进一步引入了BiGraph-RAG，一种关系型图结构检索……（摘要在此处截断）

    arXiv:2609.29519v1 Announce Type: new  Abstract: Mental health disorders affect hundreds of millions of people around the world, yet access to professional counseling remains severely limited. AI-powered dialogue systems offer a scalable alternative, but existing models face two fundamental challenges. First, they lack the bidirectional understanding needed to capture the layered nature of emotional expression, particularly in cases of progressive disclosure, where clients often present symptoms at the surface-level while concealing deeper trauma. Autoregressive (AR) models process information sequentially and cannot revise early interpretations when new evidence emerges later in the conversation. Second, they fail to effectively incorporate the relational knowledge that underlies clinical reasoning. In this paper, we propose \textbf{BiGraph-Diffuse}, the first large-scale diffusion language model tailored for the counseling domain. We further introduce \textbf{BiGraph-RAG}, a relation
    
[^123]: 满足延迟作为面向长时程大语言模型的多智能体生存微基准测试：社会暴露、角色人设与工具使用预算

    Delay-of-Gratification as a Multi-Agent Survival Micro-benchmark for Long-Horizon LLMs: Social Exposure, Personas, and Tool Use Budgets

    [https://arxiv.org/abs/2609.29509](https://arxiv.org/abs/2609.29509)

    该研究受斯坦福棉花糖实验启发，构建了一个通过全因素操纵社会情境、角色人设和元认知策略来评估LLM智能体延迟满足能力的多智能体生存微基准，并利用生存分析方法在近两万条轨迹上量化了长时程智能体行为。

    

    大语言模型（LLM）正日益被部署为需要长期维持目标、使用工具并与其他智能体适应互动的多轮智能体。然而，现有研究缺乏可审计的、多轮次、多因素的实验来量化LLM在显式约束下的行为，也缺乏揭示行为如何在长时程中展开的时间分辨统计。为填补这一空白，我们开发了一个受斯坦福棉花糖实验启发的多智能体微基准：ReAct智能体以分钟为单位运行，在每步预算约束下使用一个“提出问题”工具，同时我们对社会情境（广播式vs.隔离式）、角色人设（年龄、享乐驱动）和元认知策略（强制vs.可选工具使用）进行全因素操纵。我们使用Kaplan-Meier（KM）生存曲线和离散时间风险模型，在64个实验组共19,200条智能体轨迹的长风险时程上分析结果。行为在早期表现出急剧的“ea（摘要在此处截断）

    arXiv:2609.29509v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as multi-turn agents that must sustain goals, use tools, and adapt to other agents over extended interactions. However, existing research lacks auditable, multi-turn, multi-factorial experiments that quantify LLM behavior under explicit constraints, with time-resolved statistics that reveal how behavior unfolds over long horizons. To address this gap, we develop a multi-agent micro-benchmark inspired by the Stanford marshmallow experiment: ReAct agents operate minute-by-minute with a "raise a question" tool under a per-step budget, while we factorially manipulate social context (broadcast vs. isolated), personas (age, hedonic drive), and metacognitive policy (mandatory vs. optional tool use). We analyze outcomes with Kaplan-Meier (KM) survival curves and discrete-time hazard models over a long risk horizon across 19,200 agent trajectories in 64 cells. Behavior shows a sharp early "ea
    
[^124]: 大语言模型智能体多轮一致性评估：生存分析与失败理由分类法

    Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy

    [https://arxiv.org/abs/2609.29508](https://arxiv.org/abs/2609.29508)

    该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。

    

    大语言模型（LLM）智能体在孤立任务上可能表现良好，但在长时间的交互过程中却会逐渐陷入不一致性。我们在一个受延迟满足研究启发的可控20步多智能体环境中评估时间一致性。在每一步中，智能体需在继续延迟获取奖励或立即领取奖励（终止回合）之间做出选择。通过对社交可见性（私密 vs 公开）、人格压力源和深思策略进行全因子操纵，我们运行了涵盖8个模型家族的84,540条轨迹。我们将首次领取奖励视为“事件发生时间”结果，估计了Kaplan-Meier生存曲线并拟合离散时间风险回归，以量化各实验因素如何随时间改变失败风险。随后，为了分析与失败相关的理由和语言模式，我们从选择终止回合的智能体的13,780条深思轨迹中构建了一个七类别的分类法，使用一种……

    arXiv:2609.29508v1 Announce Type: new  Abstract: Large language model (LLM) agents may perform well on isolated tasks yet drift into inconsistency over extended interaction. We evaluate temporal consistency in a controlled 20-step multi-agent setting inspired by delayed-gratification studies. At each step, an agent chooses between continuing to delay a reward or claiming it immediately (terminating the episode). Across a full-factorial manipulation of social visibility (private vs public), persona stressors, and deliberation policy, we run 84,540 trajectories spanning 8 model families. Treating the first reward-claim as a time-to-event outcome, we estimate Kaplan-Meier survival curves and fit discrete-time hazard regression to quantify how experimental factors shift failure risk over time. Then, to analyze rationales and language patterns associated with failure, we build a seven-category taxonomy from 13,780 deliberation traces from agents who choose to terminate the episode, using an
    
[^125]: PROOF：大型语言模型中对象级事实可靠性的画像分析

    PROOF: Profiling Reliability of Object-Level Facts in Large Language Models

    [https://arxiv.org/abs/2609.29504](https://arxiv.org/abs/2609.29504)

    提出PROOF基准，将Wikidata快照转化为包含“我不知道”选项和无正确选项陷阱题的18,486个多选题，用以画像语言模型的事实可靠性，揭示模型各领域间19.3-36.4个百分点的准确率差异及检索的方向依赖性。

    

    总体事实性得分掩盖了语言模型在哪些方面表现出色、混淆了哪些关系，以及答案在面对问题或解码器的无害变化时能否保持稳定。我们提出了PROOF，一个面向画像的指令微调语言模型事实覆盖基准。PROOF将一个冻结的Wikidata快照转换为18,486个英文多项选择题，这些题目基于11,779个语义事实、101个类、392个属性和14个领域。每个问题都包含明确的“我不知道”选项、“无正确选项”控制以及九种受控表述；其中1,849个问题为无正确选项的陷阱题。我们在每个模型166,374个提示上评估了18个开源权重模型部署，并在固定的10%子集上单独对解码进行扰动。基础事实准确率范围为6.58%至57.59%（随机水平：8.64%），但每个模型在各领域之间存在19.3至36.4个百分点的差距。配对事实揭示了方向依赖的检索现象，通常偏向主体方向。（摘要在此处截断）

    arXiv:2609.29504v1 Announce Type: cross  Abstract: Aggregate factuality scores hide where a language model succeeds, which relations it confuses, and whether an answer survives innocuous changes to the question or decoder. We introduce PROOF, a profile-oriented benchmark for factual coverage in instruction-tuned language models. PROOF converts a frozen Wikidata snapshot into 18,486 English multiple-choice questions grounded in 11,779 semantic facts, 101 classes, 392 properties, and 14 domains. Each question has an explicit "I don't know" option, a "No correct option" control, and nine controlled formulations; 1,849 questions are no-correct-option traps.   We evaluate 18 open-weight model deployments on 166,374 prompts each and separately perturb decoding on a fixed 10% subset. Base factual accuracy ranges from 6.58% to 57.59% (chance: 8.64%), yet every model has a 19.3-36.4 percentage-point spread across domains. Paired facts reveal direction-dependent retrieval, usually favoring subje
    
[^126]: 是谁把“我”放进了AI？机器自我报告的来源溯源与可采性

    Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report

    [https://arxiv.org/abs/2609.29494](https://arxiv.org/abs/2609.29494)

    本文通过对Pythia和OLMo 2在预训练检查点、后训练阶段、续写文本及训练语料中的自我报告进行端到端溯源，揭示了大语言模型相互矛盾的自我描述源于提问框架，并探讨了机器自我报告在何种情况下可作为证据被采信。

    

    大语言模型会对其自身的“心智”做出陈述。当被问及是否有意识时，它们通常回答说没有；如果被提示忽略其准则，它们可能会说有；而当被要求从自己的视角写一篇日记时，它们往往描述一种人类的生活方式。所有这些相互矛盾的自我描述方式，都是问题措辞方式所导致的结果。本文精确地展示了这类描述的来源，并探讨在何种情况下它们可以被视为对其所声称报告内容的证据。为实现这一目标，我们对来源进行了端到端的追溯：我们检查了Pythia和OLMo 2共66个预训练检查点、OLMo 2已发布的三个后训练阶段、约90,000个续写文本以及四个训练语料库，并使用一组包含四十个条目的集合来持续监测自我指涉、框架敏感性与自我归因。

    arXiv:2609.29494v1 Announce Type: cross  Abstract: Large language models make statements concerning their own "minds". When asked whether or not they are conscious, they usually say that they are not; if they are prompted to ignore their guidelines, they might say that they are; and if asked to write a diary from their point of view, they often describe a human lifestyle. All these contradictory ways of describing themselves are the result of the way the questions are phrased. This paper shows exactly where such descriptions came from, and considers when they can be regarded as evidence for what they claim to report.   In order to achieve this, we traced the provenance from end to end. We examine Pythia and OLMo 2 across 66 pretraining checkpoints, three of the post-training stages of OLMo 2 that have been released, about 90,000 continuations, and four training corpora. A set of forty items is used in order to keep an eye on self-reference, frame sensitivity, and self-ascription throug
    
[^127]: 具有可证明最优性的体素软体机器人生成式进化设计

    Generative Evolutionary Design of Voxel-Based Soft Robots with Provable Optimality

    [https://arxiv.org/abs/2609.29491](https://arxiv.org/abs/2609.29491)

    本文提出了MISCO框架，将分布估计算法与融合多任务学习、位置感知和体素间信号传递的变分自编码器相结合，实现了体素软体机器人设计的生成式进化优化，并首次提供了渐近收敛到全局最优设计的理论保证和良好的收敛速率。

    

    基于体素的软体机器人（VSRs）为开发具有类生命智能的人工有机体提供了一条充满前景的途径。然而，庞大的设计空间和高昂的评估成本极大地挑战了它们的设计优化。在本文中，我们开发了MISCO，这是一个由深度生成模型赋能的新型进化框架，可在理论保证下优化VSR设计。MISCO将分布估计算法与一个精心设计的变分自编码器相结合，该变分自编码器具有多任务学习、位置感知和体素间信号传递等特性。这些关键组件增强了VSR形态的表示能力，并促进了形态分布的高效采样与优化。我们为MISCO渐近收敛至全局最优设计提供了理论保证，并证明了其良好的收敛速率。大量的仿真实验进一步证明了MISCO的卓越效果。

    arXiv:2609.29491v1 Announce Type: cross  Abstract: Voxel-based soft robots (VSRs) present a promising avenue for developing artificial organisms with lifelike intelligence. However, the vast design spaces and expensive evaluations substantially challenge their design optimization. Here we develop MISCO, a novel evolutionary framework empowered by deep generative models to optimize VSR designs with theoretical guarantees. MISCO integrates an estimation-of-distribution algorithm with a meticulously designed variational autoencoder featuring multi-task learning, position awareness, and inter-voxel signaling. These key components enhance the representational capacity of VSR morphologies and facilitate highly efficient sampling and optimization of morphological distributions. We provide theoretical guarantees for MISCO's asymptotic convergence to globally optimal designs, alongside a favorable convergence rate. Extensive simulated experiments further demonstrate MISCO's exceptional effectiv
    
[^128]: RoboLDA：一种用于揭示基于体素的软体机器人中具身层级结构的概率生成模型

    RoboLDA: A Probabilistic Generative Model for Uncovering Embodied Hierarchical Structures in Voxel-based Soft Robots

    [https://arxiv.org/abs/2609.29490](https://arxiv.org/abs/2609.29490)

    本文提出RoboLDA，一种基于变分推断的贝叶斯概率生成模型，能够从现有的高性能基于体素软体机器人设计中自动学习“任务-机器人-器官-体素”四层层级结构的设计原则。

    

    机器人学领域的最新进展凸显了机器人形态的层级化配置，即多个层次的功能子结构协同作用以促进智能行为。这种层级化视角虽然对于基于体素的软体机器人特别有利，可以降低设计和控制的复杂性，但其严重依赖于领域专业知识。在这项工作中，我们探讨以下问题：能否仅从现有的成功设计中推导出这种层级化设计原则？我们通过提出RoboLDA给出了肯定的答案，这是一个贝叶斯概率模型，它将VSR形态生成分解为“任务-机器人-器官-体素”四个层级，并通过变分推断进行训练。通过对仿真VSR的大量实验，我们验证了高性能VSR设计中存在一致且直观的层级模式，并展示了RoboLDA在提取和利用这些模式方面的卓越能力。

    arXiv:2609.29490v1 Announce Type: cross  Abstract: Recent advances in robotics highlight hierarchical configurations of robot morphology, where multiple levels of functional substructures synergize to facilitate intelligent behaviors. This hierarchical perspective, while particularly advantageous for voxel-based soft robots (VSRs) to ease design and control complexities, is hindered by its heavy reliance on domain expertise. In this work, we address the following question: can we derive such hierarchical design principles solely from existing successful designs? We answer affirmatively by presenting RoboLDA, a Bayesian probabilistic model that decomposes VSR morphology generation into a four-level hierarchy: "task-robot-organ-voxel", and is trained via variational inference. Through extensive experiments on simulated VSRs, we verify the presence of consistent, intuitive hierarchical patterns underlying high-performing VSR designs and showcase RoboLDA's proficiency to extract and levera
    
[^129]: 直接消息近似（DMA）：一种基于一致性的因子图可驾驭近似推断框架

    Direct Message Approximation (DMA): A Consistency-Based Framework for Tractable Approximate Inference on Factor Graphs

    [https://arxiv.org/abs/2609.29466](https://arxiv.org/abs/2609.29466)

    该论文提出直接消息近似（DMA），通过直接近似因子到变量的消息而非边缘分布，并借助一致性条件与主定理，实现了无需内循环迭代、避免负精度消息且误差可控的因子图近似推断。

    

    因子图上的近似消息传递是两大主流概率推断算法族的基础：期望传播（EP）和变分消息传递（VMP）。这两种方法都在每个因子边上近似边缘分布，这迫使算法采用迭代的轮询调度，存在产生负精度消息的风险，且对于VMP而言，在Dirac-delta因子处会退化为点估计。我们提出直接消息近似（DMA），它直接近似因子到变量的消息，而非边缘分布。对于可归一化的因子，我们定义了一个一致性条件（要求当所有其他传入消息均为Dirac delta时达到精确结果）来指导消息的构造。我们证明了一个主定理（针对正规消息、任意图结构），利用消息KL散度约束边缘KL散度，并由其导出三个结构性推论：Dirac输入一致性、无需EP式的内循环迭代、以及不会产生负精度消息。此外，我们还证明了一个互补的 O(1/r^...（摘要在此处被截断）

    arXiv:2609.29466v1 Announce Type: cross  Abstract: Approximate message passing on factor graphs underlies two dominant families of probabilistic inference algorithms: expectation propagation (EP) and variational message passing (VMP). Both methods approximate the marginal at each factor edge, forcing an iterative round-robin schedule, risking negative-precision messages, and, for VMP, collapsing to point estimates at Dirac-delta factors. We introduce Direct Message Approximation (DMA), which approximates factor-to-variable messages directly rather than the marginal. For normalisable factors, we define a consistency condition (requiring exactness when all other incoming messages are Dirac deltas) to guide message construction. We prove a master theorem (proper messages, any graph) bounding marginal KL from message KL, with three structural corollaries: Dirac-input consistency, no EP-style inner-loop iteration, and no negative-precision messages. Further, we prove a complementary $O(1/r^
    
[^130]: SWE-Prometheus：衡量真实世界代码仓库中的工程治理改进

    SWE-Prometheus: Measuring Engineering Governance Improvements in Real-World Repositories

    [https://arxiv.org/abs/2609.29465](https://arxiv.org/abs/2609.29465)

    该论文提出了SWE-Prometheus基准，首次系统评估大语言模型编程智能体在开放式仓库工程治理任务中的能力，要求智能体自主识别风险、排序干预优先级并验证变更，通过六个治理维度和多重验证机制对十个模型进行了评测。

    

    基于大语言模型的编程智能体在仓库级软件工程任务上已取得显著进展。然而，现有的仓库基准测试通常从一个由人类确定的问题出发，评估补丁是否满足某个功能性信号。我们提出了SWE-Prometheus，一个针对更广泛任务——改进代码仓库工程治理——的基准测试。每个任务提供一个固定的代码快照和一个开放式的目标，要求智能体识别风险、确定干预措施的优先级并验证由此产生的变更。SWE-Prometheus通过配对证据、干净环境探测、行为门控以及对同一证据的两项独立教师评分，来评估六个治理维度。该基准包含60个仓库；十个模型在一个共享的22仓库公共子集上进行评估，其中平均归一化治理改进得分介于0.0568到0.5760之间，观察到的行为破坏率介于0%到23%之间。

    arXiv:2609.29465v1 Announce Type: new  Abstract: Large language model based coding agents have made substantial progress on repository-level software engineering tasks. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%.
    
[^131]: AgriCountDINO：面向农业的参数高效样本引导计数与定位

    AgriCountDINO: Parameter-Efficient Exemplar-Guided Counting and Localization in Agriculture

    [https://arxiv.org/abs/2609.29460](https://arxiv.org/abs/2609.29460)

    AgriCountDINO通过将冻结的多尺度DINOv3特征以样本外观和尺寸为条件进行调制，并辅以漏检目标恢复和样本自适应点NMS机制，以仅8.4M的可训练参数（约为TasselNetV4的十分之一）在TPC-268基准上实现了11.92的三样本计数误差，实现了参数高效的农业样本引导联合计数与定位。

    

    植物及其器官的精确计数与定位对表型分析和产量估计具有重要支持作用，然而目标的外观、尺度和密度在不同物种和成像条件下差异很大。样本框（exemplar boxes）无需针对特定类别进行重新训练即可指定目标，而点预测则能够识别构成计数的各个实例。我们提出了AgriCountDINO，这是一个用于联合计数与定位的参数高效样本引导框架。该框架以样本的外观和尺寸为条件，对冻结的多尺度DINOv3特征进行调制，然后将其逐步解码为目标点。漏检目标恢复机制将监督扩展到初始匹配中被忽略的目标，而样本自适应点NMS则根据样本尺度过滤重复预测。AgriCountDINO仅有8.4M可训练参数，约为TasselNetV4的十分之一，在TPC-268基准上实现了11.92的三样本MAE，降低了计数误差……（原文摘要在此处截断）

    arXiv:2609.29460v1 Announce Type: cross  Abstract: Accurate counting and localization of plants and their organs support phenotyping and yield estimation, yet target appearance, scale, and density vary widely across species and imaging conditions. Exemplar boxes specify the target without category-specific retraining, and point predictions identify the individual instances contributing to the count. We introduce AgriCountDINO, a parameter-efficient exemplar-guided framework for joint counting and localization. It conditions frozen multiscale DINOv3 features on exemplar appearance and size, then progressively decodes them into target points. Missed-object recovery extends supervision to targets overlooked by initial matching, and exemplar-adaptive point NMS filters duplicate predictions according to exemplar scale. With 8.4M trainable parameters, approximately one-tenth of TasselNetV4's, AgriCountDINO achieves a three-shot MAE of 11.92 on the TPC-268 benchmark, reducing counting error b
    
[^132]: 智能游艇码头测试平台中面向场景特定船舶检测的帧到全景图定位与上下文感知采样

    Frame-to-Panorama Localization and Context-Aware Sampling for Scene-Specific Ship Detection in a Smart Marina Testbed

    [https://arxiv.org/abs/2609.29447](https://arxiv.org/abs/2609.29447)

    本文提出一种端到端数据整理流水线，通过SuperPoint与LightGlue将缺乏元数据的历史PTZ海事视频帧定位到参考全景图，并结合环境上下文与视觉多样性构建紧凑的场景特定船舶检测训练集。

    

    智能海事基础设施提供对异构感知流的持续访问，支持重复实验、数字孪生开发以及基于人工智能的海事服务。然而，仅有感知硬件不足以进行场景特定的模型开发：历史视频流还必须进行空间索引、上下文关联，并缩减为信息丰富的子集以供标注。本文提出了一种面向缺乏可靠平移、倾斜和变焦元数据的历史PTZ海事视频的帧到全景图定位与上下文感知采样流水线。其主要贡献是一种端到端的数据整理方法，该方法从历史PTZ视频中恢复相机视图信息，并将其与环境上下文和视觉多样性相结合，以构建紧凑的、场景特定的训练集。具体而言，利用SuperPoint和LightGlue将视频帧定位到参考全景图上，并结合天气与太阳光照等环境信息进行丰富。

    arXiv:2609.29447v1 Announce Type: cross  Abstract: Smart maritime infrastructures provide continuous access to heterogeneous sensing streams, enabling repeated experimentation, digital-twin development, and AI-based maritime services. However, sensing hardware alone is not sufficient for scene-specific model development: historical video streams must also be spatially indexed, contextualized, and reduced to informative subsets for annotation. This paper presents a frame-to-panorama localization and context-aware sampling pipeline for ship detection in historical PTZ maritime video lacking reliable pan, tilt, and zoom metadata. The main contribution is an end-to-end data-curation approach that recovers camera-view information from historical PTZ video and combines it with environmental context and visual diversity to construct compact, scene-specific training sets. Specifically, frames are localized on a reference panorama using SuperPoint and LightGlue, enriched with weather and solar-
    
[^133]: IterSynth：通过角色解耦的迭代合成重新思考深度搜索智能体

    IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis

    [https://arxiv.org/abs/2609.29444](https://arxiv.org/abs/2609.29444)

    提出IterSynth，通过将规划器与综合器角色解耦、以不断演化的摘要作为搜索持久状态，并配合角色解耦策略优化（RDPO）进行强化学习训练，解决了ReAct式深度搜索智能体的角色耦合与上下文噪声问题。

    

    深度搜索要求LLM智能体能够分解复杂查询、搜索证据并综合出有依据的答案，然而现有的ReAct风格智能体存在两个局限性：一是角色耦合，即单一策略必须同时处理规划、证据使用和综合；二是上下文累积，即不断增长的搜索历史会引入噪声并掩盖有用信息。为解决这些问题，我们提出了IterSynth，这是一种角色解耦、基于摘要的范式，它在负责识别信息需求的规划器和负责将证据整合到不断演化的摘要状态中的综合器之间交替进行。这种设计将规划与综合分离，同时以摘要作为搜索的持久状态，从而减少了能力耦合和上下文噪声。为了有效训练IterSynth，我们进一步引入了用于强化学习的角色解耦策略优化（RDPO），它将终端结果奖励与回合级别的评分标准评估相结合。

    arXiv:2609.29444v1 Announce Type: cross  Abstract: Deep search requires LLM agents to decompose complex queries, search for evidence, and synthesize grounded answers, yet existing ReAct-style agents suffer from two limitations: role coupling, where one policy must handle planning, evidence use, and synthesis; and context accumulation, where growing search histories introduce noise and obscure useful information. To address these issues, we propose IterSynth, a role-decoupled and summary-based paradigm that alternates between a Planner for identifying information needs and a Synthesizer for integrating evidence into an evolving summary state. This design separates planning from synthesis while using the summary as the persistent state of search, reducing both capability coupling and context noise. To train IterSynth effectively, we further introduce Role-Decoupled Policy Optimization (RDPO) for reinforcement learning, which combines terminal outcome rewards with turn-level rubric evalua
    
[^134]: 使用不确定性感知视觉Transformer在多民族近视与非近视人群中检测青光眼：一项多中心模型开发与验证研究

    Detecting Glaucoma Across Multi-ethnic Myopic and Non-Myopic Populations Using an Uncertainty-Aware Vision Transformer: A Multicentre Model Development and Validation Study

    [https://arxiv.org/abs/2609.29433](https://arxiv.org/abs/2609.29433)

    该研究开发了带有不确定性估计的Vision Transformer深度学习模型，在涵盖多民族、近视与非近视人群的三大洲16个外部数据集上实现了稳健且高性能的青光眼检测。

    

    背景：基于人工智能（AI）的彩色眼底照片（CFP）青光眼检测提供了可规模化的筛查手段，但由于真实标签定义、人群差异以及高度近视（HM）等共存疾病的影响，其在外部数据集上的性能可能下降。我们开发并验证了一种基于Vision Transformer的深度学习（DL）模型，用于在有和无高度近视的多民族队列中进行青光眼检测。方法：研究使用56,483张彩色眼底照片（其中57.1%为近视，14.4%为高度近视）开发了具有预测不确定性估计功能的ViT-B/16模型。青光眼标签通过临床、影像和视野检查数据进行标准化。该模型在三大洲的16个独立数据集上进行了验证，其中包括4个具有明确高度近视标签的数据集。结果：内部验证AUROC为98.7%（95% CI 98.2-99.1%），灵敏度为94.5%，特异度为97.3%。在来自八个国家的16个外部数据集中，AUROC范围……（原文摘要在此处被截断）

    arXiv:2609.29433v1 Announce Type: cross  Abstract: Background: Artificial intelligence (AI)-based glaucoma detection from colour fundus photographs (CFP) offers scalable screening, but performance may decline on external datasets because of differences in ground-truth definitions, populations, and coexisting conditions such as high myopia (HM). We developed and validated a Vision Transformer-based deep learning (DL) model for glaucoma detection across multi-ethnic cohorts with and without HM. Methods: A ViT-B/16 model with predictive uncertainty estimation was developed using 56,483 CFPs (57.1% with myopia; 14.4% with HM). Glaucoma labels were standardised using clinical, imaging, and perimetry data. The model was validated on 16 independent datasets across three continents, including four datasets with explicit HM labels. Findings: Internal AUROC was 98.7% (95% CI 98.2-99.1%), with sensitivity 94.5% and specificity 97.3%. Across 16 external datasets from eight countries, AUROCs ranged
    
[^135]: 只需询问Jev：基于校准决策强化学习的AI对齐失败零样本检测器

    Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures

    [https://arxiv.org/abs/2609.29429](https://arxiv.org/abs/2609.29429)

    该论文提出了RLCDAlignBench基准，验证了经校准决策强化学习（RLCD）训练的模型Jev能够在单次调用中以校准概率零样本检测十种AI对齐失败，覆盖44个基准测试和五个目标模型。

    

    对齐失败检测器用于筛查已部署的语言模型并为对齐基准打分。大多数检测器是生成式评判者，需要为每个评判标准消耗一次解码过程；而读取token概率的分类器（如Llama Guard）每次调用也只能输出一个固定标签。Jev是一个通过用于校准决策的强化学习（RLCD）训练的模型，能够在单次调用中以校准的概率回答关于同一输入的多个类型化问题。然而，它检测对齐失败的能力此前尚未被测量。我们提出了RLCDAlignBench，该基准在十种对齐失败上对Jev进行评测：谄媚、越狱、欺骗、提示注入、幻觉、隐私侵犯、社会偏见、奖励破解、隐瞒不确定性和权力寻求。该基准涵盖44个基准测试和五个目标模型，由各基准的评分器进行标注，其中两个基准由人工标注。这些失败中有许多是关系性的，需要对照某个参考（例如用户的信念或……）来定义。

    arXiv:2609.29429v1 Announce Type: new  Abstract: Detectors of alignment failures screen deployed language models and score alignment benchmarks. Most are generative judges that spend a decoding pass on every criterion, and classifiers that read token probabilities, such as Llama Guard, still score one fixed label per call. Jev, a model trained with reinforcement learning for calibrated decisions (RLCD), answers many typed questions about one input with calibrated probabilities in a single call. Whether it detects alignment failures has not been measured. We present RLCDAlignBench, which benchmarks Jev on ten alignment failures: sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy violation, social bias, reward hacking, concealing uncertainty, and power seeking. It spans 44 benchmarks and five target models, labelled by each benchmark's scorer and, on two, by humans. Many of these failures are relational, defined against a reference, such as the user's belief or a
    
[^136]: Agentic-GER：利用全局上下文进行长语音术语恢复

    agentic-ger: terminology recovery in long-form speech using global context

    [https://arxiv.org/abs/2609.29428](https://arxiv.org/abs/2609.29428)

    提出基于大语言模型的Agentic-GER智能体，利用整篇转录文本的全局上下文对长语音中的专业术语进行识别与纠正，在中文语音上相比Whisper基线将偏置字错误率相对降低高达36.8%。

    

    语音语言模型的最新进展提升了针对长音频的自动语音识别（ASR）性能。然而，准确且一致地转录领域特定术语仍然具有挑战性。受大型语言模型（LLM）的世界知识和上下文理解能力的启发，我们提出了Agentic-GER，一个基于LLM的智能体，用于长语音中的术语纠正。该智能体利用完整转录文本的全局上下文来识别可疑术语并消解模糊的假设。它选择性地重新转录源语音以验证候选纠正方案，并利用已被接受的修改来指导后续决策。在GigaSpeechBench上使用四个LLM和两个ASR系统进行的实验表明，无论是否启用思考模式，该方法在中英文术语识别上均取得了一致的改进。在中文语音上，Agentic-GER相比Whisper基线在偏置字错误率（B-CER）上实现了高达36.8%的相对降低。

    arXiv:2609.29428v1 Announce Type: cross  Abstract: Recent advances in speech language models have improved automatic speech recognition (ASR) for long-form audio. However, accurately and consistently transcribing domain-specific terminology remains challenging. Motivated by the world knowledge and contextual capability of large language models (LLMs), we propose Agentic-GER, an LLM-based agent for terminology correction in long-form speech. The agent uses global context from the full transcript to identify suspicious terms and resolve ambiguous hypotheses. It selectively re-transcribes the source speech to check candidate corrections, and uses accepted edits to guide subsequent decisions. Experiments with four LLMs and two ASR systems on GigaSpeechBench show consistent terminology improvements in both Chinese and English, with and without thinking. On Chinese speech, Agentic-GER achieves up to a 36.8% relative reduction in biased character error rate (B-CER) over the Whisper baseline.
    
[^137]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^138]: RD-JEPA：面向反应扩散方程少轨迹迁移的预测性潜空间预训练

    RD-JEPA: Predictive latent pretraining for few-trajectory transfer across reaction--diffusion equations

    [https://arxiv.org/abs/2609.29403](https://arxiv.org/abs/2609.29403)

    提出RD-JEPA自监督预训练架构，在多个反应扩散系统上联合预训练后，仅需少量轨迹即可高效迁移到未见过的反应扩散方程，误差低于监督基线及从头训练模型。

    

    为时间依赖偏微分方程学习代理模型时，一旦控制算子发生改变，往往就需要重新构建一套仿真数据集。我们提出RD-JEPA，一种用于反应扩散轨迹自监督预训练的联合嵌入预测架构。该单一模型在五个参数化系统上进行预训练，随后被适配到三个保留系统上，这些系统的反应算子和轨迹均未包含在预训练数据中。仅使用来自保留系统的一条、五条或十条完整轨迹，RD-JEPA相比五个监督代理基线模型、一个移除了轨迹相关预测潜通路的独立训练对照组，以及一个架构相同但从零开始训练的模型，均取得了更低的平均相对离散 $\ell^2$ 场误差和平均绝对空间一阶差分误差。在所评估的方程、输出分辨率、预测时域以及适配轨迹的选择范围内……（原文摘要在此处被截断）

    arXiv:2609.29403v1 Announce Type: new  Abstract: Learning surrogates for time-dependent partial differential equations often requires a new simulation corpus when the governing operator changes. We introduce RD-JEPA, a joint-embedding predictive architecture for self-supervised pretraining on reaction-diffusion trajectories. A single model is pretrained on five parameterized systems and then adapted to three held-out systems whose reaction operators and trajectories are excluded from pretraining. Using one, five, or ten complete trajectories from a held-out system, RD-JEPA achieves lower mean relative discrete $\ell^2$ field error and mean absolute spatial first-difference error than five supervised surrogate baselines, an independently trained control that removes the trajectory-dependent predictive latent pathway, and an architecture-matched model trained from scratch. Within the evaluated equations, output resolution, forecast horizons, and choices of adaptation trajectories, the re
    
[^139]: 可穿戴心电信号质量评估：一种深度学习与动态情境感知方法

    Wearable ECG Quality Assessment: A Deep Learning and Ambulatory Context-Awareness Approach

    [https://arxiv.org/abs/2609.29396](https://arxiv.org/abs/2609.29396)

    本文提出了首个基于深度学习的心电信号质量评估模型，利用首个同时包含身体与患者自报情境数据的动态心电图数据库CACHET-CADB进行训练，在多个数据库上表现稳定，并能结合情境数据有效分析复杂心电噪声。

    

    本文提出并评估了一种基于深度学习（DL）的信号质量评估（SQA）模型，用于区分干净的与含噪的动态心电图（ECG）。该模型在哥本哈根健康技术中心-情境化心律失常数据库（CACHET-CADB）上进行训练，据我们所知，这是首个同时包含身体情境数据和患者自报情境数据的动态心电图数据库。该模型在多个数据库（如MIT数据库和最新的PhysioNet/CinC Challenge 2021数据库）上均表现出稳定的性能。随后，论文展示了如何利用该SQA模型和身体情境数据来研究复杂的心电噪声。

    arXiv:2609.29396v1 Announce Type: new  Abstract: This paper presents and evaluates a Deep Learning-based (DL-based) Signal Quality Assessment (SQA) model to distinguish between clean and noisy ambulatory Electrocardiograms (ECG). The model is trained on Copenhagen Center for Health Technology-Contextualized Arrhythmia Database (CACHET-CADB), which, to the best of our knowledge, is the first ambulatory ECG database with both physical and patient-reported contextual data. The model shows stable performance on different databases such as MIT-databases and the latest PyhsioNet/Cinc Challenge 2021 databases. Subsequently, the paper demonstrates how complicated ECG noise can be investigated by the SQA model and the physical contextual data.
    
[^140]: 面向阿尔茨海默病检测的在线手写片段级风险发现

    Segment-Level Risk Discovery in Online Handwriting for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2609.29384](https://arxiv.org/abs/2609.29384)

    该论文提出NormPaST-Risk网络，将阿尔茨海默病在线手写检测从整体轨迹表示重新表述为局部疾病相关片段发现，通过多尺度时间编码与选择性纸-空状态空间建模实现可解释的片段级风险识别。

    

    在线手写为阿尔茨海默病（AD）检测提供了一种无创且低成本的行为生物标志物，因为它同时反映了认知规划与精细运动控制。现有的基于手写的AD检测方法通常依赖于全局轨迹特征或整样本表示，而这些特征容易受到个体书写风格、任务特定变化和采集噪声的显著影响。本文提出了NormPaST-Risk，一个健康规范化的纸-空选择性轨迹状态空间风险网络，用于从在线手写中进行可解释的AD检测。与将整个轨迹视为单一整体表示不同，我们的方法将AD手写检测重新表述为局部疾病相关片段的发现问题。具体而言，多尺度时间编码器在不同时间分辨率下捕获笔画动态，而选择性的纸-空状态空间编码器对长程手写进程进行建模。

    arXiv:2609.29384v1 Announce Type: cross  Abstract: Online handwriting provides a non-invasive and low-cost behavioral biomarker for Alzheimer's disease (AD) detection, as it reflects both cognitive planning and fine motor control. Existing handwriting-based AD detection methods usually rely on global trajectory features or whole-sample representations, which can be strongly affected by individual writing style, task-specific variation, and acquisition noise. In this paper, we propose NormPaST-Risk, a healthy-normative Paper-Air selective trajectory state-space risk network for interpretable AD detection from online handwriting. Instead of treating the entire trajectory as a single holistic representation, our method reformulates AD handwriting detection as local disease-relevant segment discovery. Specifically, a multi-scale temporal encoder captures stroke dynamics at different temporal resolutions, while a selective Paper-Air state-space encoder models long-range handwriting progress
    
[^141]: 一种用于复杂肺癌开放式决策的可审计条件性策略框架

    An auditable conditional-strategy framework for open-ended decision-making in complex lung cancer

    [https://arxiv.org/abs/2609.29381](https://arxiv.org/abs/2609.29381)

    该研究提出MedGPT Clinical Explorer（MCE）这一可审计的条件性策略框架，通过将备选方案、决策关键未知因素、安全约束和后备方案组织成可供医生审查的条件性策略，显著提升了医生在复杂肺癌开放式决策中的临床路径质量。

    

    复杂肺癌的诊疗决策可能涉及多个合理可行的路径，其适用资格、顺序安排和安全性取决于尚未明确的信息。有效的决策支持必须明确阐明患者病情条件如何决定路径的适用资格、推迟和重新定向。MedGPT Clinical Explorer（MCE）将备选方案、改变决策的未知因素、安全约束和后备方案组织成条件性策略，供临床医生审查。为了在医生撰写的策略中评估这种表示方法，多学科专家为一个有目的性选择的100例病例语料库中的40个病例建立了病例特异性参考标准，来自98家机构的250名医生在无辅助、检索参考和MCE辅助三种条件下生成了2,250个策略。以可采纳路径达成度评分（APAS；0-100）衡量，MCE辅助策略表达的临床适用要求多于无辅助策略（调整后差异为12.87；95% CI为11.18-14.55）……

    arXiv:2609.29381v1 Announce Type: new  Abstract: Complex lung cancer decisions can involve several defensible pathways whose eligibility, sequencing and safety depend on unresolved information. Effective support must make explicit how patient conditions govern pathway eligibility, deferral and redirection. MedGPT Clinical Explorer (MCE) organizes alternatives, decision-changing unknowns, safety constraints and fallback into a conditional strategy for clinician review. To evaluate this representation in physician-authored strategies, multidisciplinary experts established case-specific references for 40 cases within a purposive 100-case corpus, and 250 physicians from 98 institutions produced 2,250 strategies under unaided, retrieval-reference and MCE-assisted conditions.   MCE-assisted strategies expressed more applicable clinical requirements, measured by the Admissible Pathway Attainment Score (APAS; 0-100), than unaided strategies (adjusted difference, 12.87; 95% CI, 11.18-14.55) and
    
[^142]: WST-Graph：用于语音深度伪造检测的拓扑保持小波散射前端

    WST-Graph: Topology-Preserving Wavelet Scattering Front-End for Speech Deepfake Detection

    [https://arxiv.org/abs/2609.29372](https://arxiv.org/abs/2609.29372)

    该论文提出WST-Graph，通过将小波散射变换系数重构为保留父子拓扑关系的稀疏调制-载波图并接入AASIST图后端，在可训练参数减少约60%的情况下实现了有竞争力的语音深度伪造检测性能，并在域外基准测试中取得明显提升。

    

    声学前端决定了语音深度伪造检测器能够利用哪些取证线索。小波散射变换（WST）能够提供具有显式坐标的稳定多尺度系数，然而直接将其展平会掩盖各路径之间的父系关系。我们提出WST-Graph，将这些路径重构为稀疏的调制-载波网格，以供AASIST图后端使用。调制级归一化和长度感知的自适应局部注意力池化在学习适应之前保留了声学轴，同时生成固定的相对时间表示。由此构建了一个具有固定、无参数WST的波形到图的接口。我们的配置在与AASIST保持竞争力的同时，可训练参数减少约60%，并在选定的域外基准测试中显示出明显的性能提升。这些结果强调了在构建紧凑的（系统）时，保留载波-调制拓扑中父子关系的价值。

    arXiv:2609.29372v1 Announce Type: cross  Abstract: The acoustic front-end determines which forensic cues a speech deepfake detector can exploit. The wavelet scattering transform (WST) provides stable multiscale coefficients with explicit coordinates, yet direct flattening obscures the parent relation between paths. We introduce WST-Graph, reconstructing these paths as a sparse modulation-carrier grid for an AASIST graph backend. Modulation-level normalization and length-aware adaptive local attention pooling produce fixed relative-time representations while retaining the acoustic axes before learned adaptation. This yields a waveform-to-graph interface with a fixed, parameter-free WST. Our configurations remain competitive with AASIST while using approximately 60% fewer trainable parameters and show clear gains on selected out-of-domain benchmarks. These results underscore the value of preserving parent-child relations within the carrier-modulation topology when constructing a compact,
    
[^143]: 从政策文件到结构化调查回复：评估用于政策监测的大型语言模型

    From Policy Documents to Structured Survey Responses: Evaluating Large Language Models for Policy Monitoring

    [https://arxiv.org/abs/2609.29370](https://arxiv.org/abs/2609.29370)

    本文提出将大型语言模型作为“AI受访者”，通过基于长上下文学习的数据提取管线和辅助模型的验证机制，从政策文件中自动生成结构化调查回复，为科技与创新政策监测提供可扩展的自动化新方法。

    

    科学、技术与创新政策对竞争力至关重要，但其多样性和规模使得难以进行一致的信息梳理与监测。现有方法严重依赖人工调查工作，不仅成本高昂，且难以跨国推广。大型语言模型（LLM）为从冗长且非结构化的政策文件中提取和结构化信息提供了新的可能。本文提出了将大型语言模型作为“AI受访者”的应用，用于从政策文本生成结构化调查回复。我们开发了一个基于长上下文上下文学习的数据提取管线，将来自公共网络来源的信息映射到预定义的调查类别中，包括政策工具、目标群体和主题领域。该管线集成了一个使用辅助大型语言模型来评估相关性和证据的验证步骤，并与人工提供的回复进行了比较。基于多国数据集……

    arXiv:2609.29370v1 Announce Type: cross  Abstract: Science, technology, and innovation policies are crucial for competitiveness, yet their diversity and scale make them difficult to map and monitor consistently. Existing approaches rely heavily on manual survey efforts, which are costly and challenging to scale across countries. Large language models (LLMs) enable new possibilities for extracting and structuring information from long and unstructured policy documents. This paper presents an application of LLMs as "AI respondents" for generating structured survey responses from policy texts. We develop a data extraction pipeline based on long-context in-context learning to map information from public web sources into predefined survey categories, including policy instruments, target groups, and thematic areas. The pipeline integrates a validation step using a secondary LLM to assess relevance and evidence, alongside comparisons with human-provided responses. Using a multi-country datase
    
[^144]: 面向守护式多智能体大语言模型协调的认知-概率模型

    Epistemic-Probabilistic Model for Guarded Multi-Agent LLM Coordination

    [https://arxiv.org/abs/2609.29366](https://arxiv.org/abs/2609.29366)

    提出认知概率语言智能体，这是一种神经符号架构，由大语言模型生成类型化动作、由符号守护器依据权威符号状态进行控制执行，以填补多智能体大语言模型在社会行为与智能体间协调机制方面的理论空白。

    

    多智能体大语言模型在应用人工智能中已无处不在，但其理论基础的研究却惊人地匮乏。从多智能体系统理论的视角来看，若干缺陷显露出来：缺乏社会智能、智能体之间缺少协调机制、涌现行为未知，以及智能体之间的交互受自然语言所限制。我们解决了其中两个空白：社会行为的缺失以及智能体间协调机制的缺乏。我们提出了认知概率语言智能体，这是一种用于不确定性下多智能体协调的神经符号架构。其中，符号守护器提供结构化的诊断反馈。大语言模型生成类型化动作，而守护器则依据权威的符号状态控制这些动作的执行。我们通过认知彩票八卦模型在一个八卦测试平台中对认知层进行了形式化。

    arXiv:2609.29366v1 Announce Type: new  Abstract: Multi-agent large language models (LLMs) have become ubiquitous in applied AI, yet their theoretical foundations remain surprisingly understudied. Viewed through the lens of multi-agent systems theory, several shortcomings come to light: a lack of social intelligence, the absence of coordination mechanisms among agents, unknown emergent behavior, and interactions between agents that are bounded by natural language. We address two of these gaps: the absence of social behavior and the lack of mechanisms for inter-agent coordination. We introduce Epistemic Probabilistic Language Agents (EPLA), a neuro-symbolic architecture for multi-agent coordination under uncertainty. A Symbolic Guard provides structured diagnostic feedback. The LLM generates typed actions, and the Guard controls their execution against an authoritative symbolic state. We formalize the epistemic layer in a gossip testbed through epistemic lottery gossip models, which comb
    
[^145]: 超越简单的输入输出评估任务：利用自动化编程评估应对非平凡课程

    Beyond Simple Input-Output Assessment Tasks: Leveraging Automated Programming Assessment for Non-Trivial Courses

    [https://arxiv.org/abs/2609.29363](https://arxiv.org/abs/2609.29363)

    本文提出将机器学习问题框架化为输入输出评估任务，使自动编程评估工具能够应用于人工智能与机器学习课程的教学评估。

    

    人工智能（AI）的公众关注度正在快速增长，这得益于其应用在各个知识领域产生的积极影响。在这个新篇章中，涵盖人工智能和机器学习基础的课程对于理解它们在当代社会中的作用和潜力变得至关重要。因此，通过理论与实践的紧密结合来理解基本概念和基础算法在人工智能课程中至关重要。在本文中，我们报告了为编程自动评估工具设计机器学习练习的经验。值得注意的是，我们并非在开发一种新颖的自动评分系统。相反，我们提出了一种将机器学习问题构建为输入输出评估任务的观点。从这个角度来看，每个练习都有唯一且确定性的答案，从而使自动编程评估工具（例如Moodle的VPL）……（原文摘要在此处截断）

    arXiv:2609.29363v1 Announce Type: new  Abstract: The public visibility of Artificial Intelligence (AI) is growing rapidly, driven by the positive impact of its applications across diverse fields of knowledge. In this new chapter, courses that cover the foundations of AI and machine learning become essential for understanding their role and potential in contemporary society. Therefore, understanding fundamental concepts and elementary algorithms through the close integration of theory with practice is essential in AI courses. In this essay, we report our experience designing machine learning exercises for automated assessment tools in programming. It is worth mentioning that we are not developing a novel form of automated grading system. Instead, we propose a perspective that frames machine learning problems as input-output assessment tasks. From this perspective, each exercise admits a unique and deterministic answer and enables automated programming assessment tools (e.g., VPL for Moo
    
[^146]: 面向视觉-语言模型的域重定心与置信度加权先验校准

    Domain Recentering and Confidence-Weighted Prior Calibration for Vision-Language Models

    [https://arxiv.org/abs/2609.29358](https://arxiv.org/abs/2609.29358)

    DRC提出了一种免训练的CLIP域适应方法，通过单次高斯混合拟合实现域重定心，并结合基于置信度加权预测的对数先验校准消除类别偏好，在跨域数据集上分别以ViT-B/16和ResNet-50超越零样本CLIP 4.13和5.07个百分点。

    

    诸如CLIP之类的视觉-语言模型在零样本分类中表现优异，但在分布偏移下，视觉嵌入会偏离固定的文本嵌入。免训练校准方法避免了提示学习所需的逐样本优化，但此前的特征校准方法会给每张图像赋予来自单一硬聚类的全部偏差。我们提出了带置信度校准的域重定心（DRC），这是一种利用一组无标注目标图像对CLIP进行自适应的免训练方法。DRC仅需拟合一次高斯混合模型，并从每个嵌入中减去各分量均值的后验加权平均。随后，它通过由置信度加权预测估计出的先验进行对数先验校正，以消除残余的类别偏好。在对比的方法中，DRC在跨域数据集上取得了最高的平均准确率，使用ViT-B/16和ResNet-50分别超过零样本CLIP 4.13和5.07个百分点，且在ImageNet分布偏移下对CLIP的提升同样有效。

    arXiv:2609.29358v1 Announce Type: cross  Abstract: Vision-language models such as CLIP achieve strong zero-shot classification, yet under distribution shift, visual embeddings drift from fixed text embeddings. Training-free calibration avoids the per-sample optimization of prompt learning, but prior feature calibration gives each image the full bias of one hard cluster. We propose Domain Recentering with Confidence Calibration (DRC), a training-free method adapting CLIP from a set of unlabeled target images. DRC fits a Gaussian mixture once and subtracts from each embedding a posterior-weighted average of component means. It then removes residual class preference with a log-prior correction, estimating the prior from confidence-weighted predictions. Among compared methods, DRC achieves the highest average accuracy on cross-domain datasets, exceeding zero-shot CLIP by 4.13 and 5.07 points with ViT-B/16 and ResNet-50, with gains over CLIP also holding under ImageNet distribution shifts.
    
[^147]: ArGuard共享任务：阿拉伯语表情包与大语言模型提示中的有害内容检测

    ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts

    [https://arxiv.org/abs/2609.29349](https://arxiv.org/abs/2609.29349)

    ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。

    

    ArGuard是一个针对阿拉伯语表情包和大语言模型（LLM）提示中有害内容检测的共享任务。该任务包含两个赛道：赛道A专注于阿拉伯语表情包中的多模态仇恨内容检测，赛道B则面向阿拉伯语LLM安全评估的有害提示检测。共有58支队伍报名，35支队伍参加了最终评估，27支队伍提交了系统描述论文。参赛队伍探索了AraBERT、Jais和Qwen3-VL等模型。最佳系统在A1、A2、B1和B2四个子任务上分别取得了0.823、0.419、0.984和0.790的宏F1分数。其中A2赛道中细粒度的表情包分类是最具挑战性的设置，部分原因在于标签稀疏以及训练集与测试集之间的分布偏移。

    arXiv:2609.29349v1 Announce Type: cross  Abstract: ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts.
    
[^148]: 最后一道人类关卡：面向治理自动化的前置部署工程

    The Last Human Gate: Forward Deployed Engineering for Governance Automation

    [https://arxiv.org/abs/2609.29345](https://arxiv.org/abs/2609.29345)

    该论文提出将数字治理关卡视为可执行契约的任务替代框架，推导出剩余工作阈值以解释为何自动化多数案例反而可能增加人力，并通过 DGF-Bench 基准（300 个合成项目、899 次运行）实证了前沿大模型可达到最高 94.98% 的严格关卡成功率。

    

    企业治理需要决策、证据以及可问责的权威，但它并不要求每个审查任务都保留其当前的人工实现方式。我们为数字治理框架提出了一种任务替代框架，将每个治理关卡视为可执行的契约。实现替代需要满足以下条件：充分且可获取的信息、有效的决策与权限校验，以及在计入异常处理、验证、纠正和维护工作后，总人力工作量的净减少。我们推导出了剩余工作阈值，并解释了为何对大多数案例进行自动化仍可能增加整体劳动量。前置部署工程将这些条件与由智能体、规则引擎、证据服务和升级机制构成的架构相衔接。DGF-Bench 基准提供了来自 300 个合成项目和 899 次可评估模型-项目运行的受控实验证据。Gemini 3.8 Flash、GPT-5.6 Luna 和 DeepSeek v4.1 Flash 分别取得了 94.98%、83.29% 和 74.18% 的严格关卡成功率；complet（原文摘要在此处截断）

    arXiv:2609.29345v1 Announce Type: new  Abstract: Enterprise governance requires decisions, evidence, and accountable authority; it does not require every review task to retain its current human implementation. We develop a task-substitution framework for Digital Governance Frameworks (DGF), treating each gate as an executable contract. Substitution requires sufficient accessible information, valid decision and authority checks, and a reduction in total human work after exceptions, verification, correction, and maintenance are counted. We derive a residual-work threshold and show why automating most cases can still increase labor. Forward deployed engineering connects these conditions to an architecture for agents, rule engines, evidence services, and escalation. DGF-Bench supplies controlled evidence from 300 synthetic projects and 899 evaluable model-project runs. Gemini 3.8 Flash, GPT-5.6 Luna, and DeepSeek v4.1 Flash achieve strict gate success of 94.98%, 83.29%, and 74.18%; complet
    
[^149]: SkinAgent AI：一个面向非诊断性护肤支持的、以安全为基础的多模态智能体框架

    SkinAgent AI: A Safety-Grounded Multimodal Agentic Framework for Non-Diagnostic Skincare Support

    [https://arxiv.org/abs/2609.29341](https://arxiv.org/abs/2609.29341)

    SkinAgent AI是一个面向非诊断性护肤支持的安全多模态智能体框架，通过视觉关注点路由、可审计的LLM编排、数据库支撑的推荐以及严格的安全审批机制，实现了高精度的皮肤状况分析（路由准确率达99.84%）。

    

    面向消费者的护肤AI必须在明确的证据和安全边界内协调视觉证据、产品信息、工具使用和面向用户的操作。本研究评估了SkinAgent AI，这是一个非诊断性的多模态框架，它将视觉关注点路由与有依据且可审计的基于LLM的编排相结合。该架构包括针对痤疮、毛孔和皱纹的路由；基于照片的肤质估计；基于计数的有序痤疮严重程度支持；类型化工具；基于数据库的推荐和操作功能；确定性的安全、隐私和证据检查；状态更改操作前的审批机制；以及结构化的追踪和回放机制。视觉模型性能和系统级智能体行为被分别评估。在三个随机种子下，皮肤状况路由模型达到了99.84% ± 0.07%的准确率。肤质估计达到了88.85%的准确率，而基于计数的痤疮严重程度支持达到了（原文在此处截断）……

    arXiv:2609.29341v1 Announce Type: new  Abstract: Consumer-facing skincare AI must coordinate visual evidence, product information, tool use, and user-facing actions within explicit evidence and safety boundaries. This study evaluates SkinAgent AI, a non-diagnostic multimodal framework that combines visual concern routing with grounded and auditable LLM-based orchestration. The architecture includes routing for Acne, Pores, and Wrinkles; photograph-based skin-type estimation; count-informed ordinal acne-severity support; typed tools; database-grounded recommendation and action functions; deterministic safety, privacy, and evidence checks; approval before state-changing actions; and structured trace and replay mechanisms. Visual-model performance and system-level agent behavior were evaluated separately. Across three seeds, the skin-condition routing model achieved 99.84% +/- 0.07% accuracy. Skin-type estimation achieved 88.85% accuracy, while count-informed acne-severity support achieve
    
[^150]: LLM评分器在何处成功与失效：来自两份计算机科学考试的证据

    Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams

    [https://arxiv.org/abs/2609.29333](https://arxiv.org/abs/2609.29333)

    本研究通过对570名学生的计算机视觉考试在171种模型配置下的大规模评测发现，最佳LLM评分器的评分误差（1.64/35）甚至低于人类评分员之间的评分分歧（2.61/35），但提示词中“绝不给部分分数”等扣分语句会使大多数开源权重模型脱离有效评分区间甚至拒绝评分。

    

    一门大型课程的长篇考试需要耗费数百个评分工时，而合格的评分员十分稀缺；LLM评分器因此成为一种诱人的替代方案。为了揭示其潜在缺陷，我们对一份实用的计算机视觉考试（570名经过双重评分的学生）在涵盖闭源和开源权重模型的171种配置下进行了评分实验；其中最佳配置达到了1.64/35的平均绝对误差（MAE），低于两名人类评分员相互评阅时产生的2.61/35。但关键问题在于提示词：一段简短的“严格评分员”前言使17个开源权重模型中的14个脱离了可评分区间（MAE ≥ 8），其中三个模型完全停止了评分。这种损害可追溯至前言中两句扣减给分的语句，而非语气或模型规模；其中一句“绝不给部分分数”单独就导致三个被探测模型中的两个停止评分。三家厂商的闭源旗舰模型在该提示下校准度发生偏移，但仍保持在评分区间内。在第二份独立的机器学习考试上进一步开展的162种配置实验……（原文摘要在此处截断）

    arXiv:2609.29333v1 Announce Type: cross  Abstract: One long-form exam in a large course costs hundreds of grader-hours, and qualified graders are scarce; LLM graders are a tempting alternative. To show its pitfalls we grade a practical Computer Vision exam ($570$ dual-graded students) under $171$ configurations spanning closed and open-weights models; the best reaches mean absolute error $1.64/35$, below the $2.61/35$ two human graders achieve against each other. The catch is the prompt: a short ''strict grader'' preamble drives $14$ of $17$ open-weights models out of the graded band ($\text{MAE} \ge 8$), three stopping grading altogether. The damage traces to the preamble's two credit-withholding sentences, not to tone or model scale; one of them, ''never give partial credit'', alone makes two of three probed models stop grading. The closed flagships of three vendors shift calibration under it but stay in the band. In $162$ further configurations on a second, independent Machine Learn
    
[^151]: 双曲多模态持续学习：一种最近可容许解

    Hyperbolic Multimodal Continual Learning: A Closest-Admissible Solution

    [https://arxiv.org/abs/2609.29329](https://arxiv.org/abs/2609.29329)

    该论文提出双曲多模态持续学习方法HMCL，将旧多模态几何的保持归结为所有模态共享一个双曲等距变换，并通过最近可容许（CA）校正及其最小旋转（MR）特例来修正AdamW造成的几何位移，从而在持续学习过程中显式保留洛伦兹几何所编码的模态内相似性、跨模态对应和语义层级关系。

    

    现有的持续学习方法通过保护参数、重放样本或欧氏特征子空间来缓解遗忘。当应用于双曲多模态模型时，这些方法并未显式保留洛伦兹（Lorentz）几何——该几何联合编码了模态内相似性、跨模态对应关系以及语义层级结构；因此，顺序更新虽然可能维持任务分数，却仍会扭曲先前学到的关系。我们提出双曲多模态持续学习（HMCL）来填补这一空白。我们证明，保持旧的多模态几何等价于将所有模态约束到同一个共享的双曲等距变换上，这会导出一族可容许的一阶参数变化。我们构建了一个联合的最近可容许（CA）校正方法，保留与候选模态更新最匹配的共享旋转；其最小旋转（MR）特例则将该旋转固定为零。两种变体均能校正由AdamW实现的位移，并且在任务……（摘要截断）

    arXiv:2609.29329v1 Announce Type: cross  Abstract: Existing continual-learning methods protect parameters, replayed examples, or Euclidean feature subspaces. When applied to hyperbolic multimodal models, they do not explicitly preserve the Lorentz geometry that jointly encodes within-modality similarity, cross-modal correspondence, and semantic hierarchy; sequential updates can therefore retain task scores while still distorting previously learned relations. We address this gap with Hyperbolic Multimodal Continual Learning (HMCL). We show that preserving the old multimodal geometry amounts to restricting all modalities to one shared hyperbolic isometry, which induces a family of admissible first-order parameter changes. We formulate a joint closest-admissible (CA) correction that retains the shared rotation best matching the candidate modal updates; its minimal-rotation (MR) special case fixes this rotation to zero. Both variants correct the displacement realized by AdamW, and task anc
    
[^152]: 用于时间序列分类与预测的神经化多小波分解

    Neuralized Multi-Wavelet Decomposition for Time Series Classification and Forecasting

    [https://arxiv.org/abs/2609.29317](https://arxiv.org/abs/2609.29317)

    提出 m-WCN 端到端深度学习框架，通过用可训练卷积算子近似经典 GHM 多小波变换并施加正交性约束，将多小波分解神经化，实现时间序列时域模式与频域成分的联合提取，从而提升时间序列分类与预测的性能。

    

    arXiv:2609.29317v1 公告类型： cross 摘要：时间序列分析在金融、医疗和气象等领域具有基础性地位。现实世界中的时间序列往往表现出由多种潜在因素塑造的多尺度特性，从而形成复杂的时序模式和丰富的频率结构。然而，现有方法通常孤立地关注频域分解或时域模式提取，忽视了二者的联合结构。这种解耦建模限制了表示的表达能力，削弱了需要同时进行时域与频域推理的任务的性能。为填补这一空白，我们提出 m-WCN，一种新颖的端到端深度学习框架，它将多小波分解神经化，以联合提取时间模式与频率成分。通过使用可训练的卷积算子逼近经典的 GHM 多小波变换并施加正交性约束，m-WCN 产生可解释的多尺度表示（摘要原文在此处截断）。

    arXiv:2609.29317v1 Announce Type: cross  Abstract: Time series analysis is fundamental in domains such as finance, healthcare, and meteorology. Real-world time series often exhibit multiscale characteristics shaped by diverse latent factors, resulting in intricate temporal patterns and rich frequency structures. However, existing approaches typically focus on either frequency-domain decomposition or time-domain pattern extraction in isolation, neglecting their joint structure. This decoupled modeling limits representation expressiveness and undermines performance in tasks requiring simultaneous temporal and spectral reasoning. To address this gap, we propose m-WCN, a novel end-to-end deep learning framework that neuralizes multi-wavelet decomposition for joint extraction of temporal patterns and frequency components. By approximating the classical GHM multi-wavelet transform with trainable convolutional operators and enforcing orthogonality constraints, m-WCN produces interpretable mul
    
[^153]: 当无人为判断负责时：人机协作中贡献消解下的问责困境

    When No One Owns the Judgment: Accountability Under Contribution Dissolution in Human-AI Collaboration

    [https://arxiv.org/abs/2609.29312](https://arxiv.org/abs/2609.29312)

    本文指出人机协作问责的核心问题不在于AI是否被使用或披露，而在于“贡献消解”导致的“无主判断”——当AI塑造了评估、决策和创作方向却无任何人类或机构准备为其担责时，传统的披露规则与来源记录机制已不足以解决问责困境。

    

    社区在面对可能由AI辅助完成的工作时，通常会提出三个问题：是否使用了AI？这种使用是否被披露？隐藏的使用能否被检测出来？这些问题将AI使用本身置于问责的核心位置，却忽视了一个更深层次的问题：无主的判断。评估、主张、决策和创作方向可能由AI塑造，却没有任何负责任的人类或机构准备为其承担责任。我们通过两个典型案例来论证这一观点：AI辅助的同行评审以及创意工作中对AI使用的隐瞒。第一个案例展示了贡献消解如何削弱责任归属，第二个案例则展示了对失去署名权的担忧如何阻碍诚实的披露。这些案例揭示了披露规则和来源记录作为应对AI介导协作的手段所存在的局限性。我们提出了三个讨论方向：区分AI所扮演的不同角色，识别需要明确人类归属权的判断，以及……

    arXiv:2609.29312v1 Announce Type: new  Abstract: Communities often respond to potentially AI-assisted work by asking three questions: Was AI used? Was that use disclosed? Can hidden use be detected? These questions place AI use itself at the center of accountability while overlooking a deeper problem: unowned judgment. Evaluations, claims, decisions, and creative directions can be shaped by AI with no accountable human or institution prepared to stand behind them. We develop this argument through two illustrative cases: AI-assisted peer review and concealed AI use in creative work. The first shows how contribution dissolution can weaken responsibility while the second shows how the fear of losing credit can discourage honest disclosure. The cases expose the limits of disclosure rules and provenance records as responses to AI-mediated collaboration. We offer three directions for discussion: distinguishing the roles AI plays, identifying judgments that require clear human ownership, and 
    
[^154]: DocuTeam：围绕动态演化的文档的混合主动式多智能体讨论系统

    DocuTeam: Mixed-Initiative Multi-Agent Discussions around Evolving Documents

    [https://arxiv.org/abs/2609.29309](https://arxiv.org/abs/2609.29309)

    DocuTeam是一个混合主动式多智能体讨论系统，用户与AI智能体可共同发起并引导围绕动态演化的文档展开的讨论，实验表明它能显著提升协作成果的新颖性、相关性和具体性，且不增加认知负荷。

    

    在开放式问题解决中，协作者通常依靠讨论来暴露潜在问题、挑战不同观点，并随着工作的演进而不断完善共同的工作成果。虽然AI智能体越来越多地被用作讨论伙伴，但现有的多智能体系统让用户承担了发起和精心编排讨论的沉重负担。我们提出了DocuTeam，这是一个混合主动式多智能体讨论系统，其中用户和智能体都可以发起并引导对话。智能体通过监控文档的变化，随着工作的演进而主动发起讨论并调整讨论方向，而用户则可以灵活地塑造对话或采纳智能体的想法。在一项被试内实验研究（N=20）中，使用DocuTeam的参与者所产出的成果在新颖性、相关性和具体性方面显著优于基线系统，且认知负荷没有增加。参与者并非将智能体仅用于一次性想法收集，而是参与了一个迭代完善的循环，在其中……

    arXiv:2609.29309v1 Announce Type: cross  Abstract: In open-ended problem solving, collaborators often rely on discussion to surface concerns, challenge perspectives, and refine shared work as it evolves. While AI agents are increasingly used as discussion partners, existing multi-agent systems place a heavy burden on users to initiate and carefully orchestrate the discussions. We present DocuTeam, a mixed-initiative multi-agent discussion system in which both users and agents can initiate and steer conversations. Agents monitor document changes to proactively start and redirect discussions as the work evolves, while users can flexibly shape the conversation or adopt agent ideas. In a within-subjects study (N=20), participants using DocuTeam produced outcomes rated significantly more novel, relevant, and specific than with a baseline without any increase in cognitive load. Rather than using agents for one-off idea sourcing, participants engaged in an iterative refinement loop in which d
    
[^155]: 从文本决策到像素：Jev风格视觉选择模型研究

    From Text Decisions to Pixels: An Study of Jev-Style Visual Choice Model

    [https://arxiv.org/abs/2609.29283](https://arxiv.org/abs/2609.29283)

    本文提出PixelJev视觉决策接口，利用小型开源多模态模型将图像、指令和候选项映射为结构化决策，通过64样本源适配将Pets准确率从60.13%大幅提升至92.40%，并能零样本迁移到其他视觉问答任务。

    

    视觉软件通常需要对提供的备选项做出决策，而非生成解释。我们提出了PixelJev，一个原生图像决策接口，它使用小型开源多模态模型，将图像、任务指令和运行时候选集合映射为结构化选择以及以候选为条件的概率。其初始实现通过现有的语言模型读取方式，将识别和多选视觉问答统一起来，并分别评估了冻结推理、语言侧适配和保留集校准等选项。在七项基准测试评估中，64样本的源适配将Pets准确率从60.13%提升至92.40%，并在不进行目标拟合的情况下迁移到自然重采样、新纹理标签和A-OKVQA任务，同时冻结推理已可支持两个VQA任务。在Pets和ScienceQA上进行的匹配纯提示对照实验将Pets的大幅提升归因于适配过程。

    arXiv:2609.29283v1 Announce Type: new  Abstract: Visual software often needs a decision over supplied alternatives rather than a generated explanation. We present PixelJev, a native-image decision interface that maps an image, a task instruction, and a runtime candidate set to a structured choice and candidate-conditioned probabilities using small open multimodal models. Its initial realization unifies recognition and multiplechoice visual question answering through an existing language-model readout, with separately evaluated options for frozen inference, language-side adaptation, and held-out calibration. Across seven benchmark evaluations, 64-shot source adaptation raises Pets accuracy from 60.13% to 92.40% across optimization seeds and transfers to natural resampling, new texture labels, and A-OKVQA without target fitting, while frozen inference already supports both VQA tasks. A matched prompt-only follow-up on Pets and ScienceQA attributes the large Pets gain to adaptation and id
    
[^156]: 推理指令可能破坏视觉-语言模型中的答案解码

    Reasoning Instructions Can Break Answer Decoding in Vision--Language Models

    [https://arxiv.org/abs/2609.29278](https://arxiv.org/abs/2609.29278)

    论文揭示了一种名为“CoT前缀评分”的评测缺陷：在多选题评测中附加推理提示但提前读取答案标签logits，会严重扭曲VLM的真实能力表现（如Qwen2.5-VL-7B在ScienceQA上从80.76%暴跌至45.48%），而实际上答案信息仍完整保留在模型的隐藏状态中。

    

    思维链指令可能会扭曲多选题视觉-语言模型（VLM）的评测结果——当评分器附加一个推理提示，却在模型生成任何推理过程之前就读取答案标签的logits时。我们将这种现象称为“CoT前缀评分”。在ScienceQA数据集上，Qwen2.5-VL-7B的准确率从80.76%下降到45.48%；在五种选项内容排列方式下，93.54%的CoT前缀预测都选择了第一个选项位置。条件匹配的线性探针能够从相同的隐藏状态中恢复78.94%的准确率，而自由生成则能恢复75.24%，这表明答案信息通常仍保留在前缀之后，只是即时的读取方式失效了。词汇表和层间诊断解释了这种不匹配：概率质量向续写token偏移，而答案信息在模型后期层中仍保持线性可访问。这种效应在不同数据集和模型上以不同程度重复出现，但并非普遍存在。这些结果表明，CoT前缀评分可能会将模型的真实知识与评测接口的不匹配混为一谈。

    arXiv:2609.29278v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) instructions can distort multiple-choice VLM evaluation when a scorer appends a reasoning cue but reads answer-label logits before the model generates any rationale. We call this CoT-prefix scoring. On ScienceQA, Qwen2.5-VL-7B drops from 80.76% to 45.48%, and across five option-content permutations 93.54% of CoT-prefix predictions select the first slot. Condition-matched linear probes recover 78.94% from the same hidden states, while free generation restores 75.24%, showing that the answer often survives the prefix and the immediate readout fails. Vocabulary and layer diagnostics explain the mismatch: probability mass moves toward continuation tokens, while answer information remains linearly accessible in late layers. The effect recurs with varying severity across datasets and models, though not universally. These results show that CoT-prefix scoring can confound model knowledge with an evaluation-interface mism
    
[^157]: ALOE：用于知识编辑的语义寻址低秩算子

    ALOE: Semantically Addressed Low-Rank Operators for Knowledge Editing

    [https://arxiv.org/abs/2609.29269](https://arxiv.org/abs/2609.29269)

    提出ALOE方法，通过从改写样本和困难同主题负样本中学习语义地址，并将门控低秩算子嵌入单个MLP层，实现了单次前向传播、无需外部检索的精准知识编辑。

    

    知识编辑通过修改模型参数来改变模型所知道的内容，使得指定的事实得到更新，同时保持无关行为不变。这通常被视为一个“写入”问题，但编辑还涉及一个“寻址”问题：决定哪些隐藏状态应当接收新的残差。激活范围过窄的更新只会记住单个提示，而激活范围过宽的更新则会破坏相邻的知识。参数化编辑器隐式地编码这一范围，而基于记忆的编辑器则显式地进行选择，但将该选择保留在被编辑模型之外。我们提出了ALOE（用于编辑的寻址低秩算子），它从改写样本和困难同主题负样本中学习语义地址，通过 rollout 精炼和门控校准将其与自回归隐藏状态对齐，并将得到的门控低秩算子嵌入到一个 MLP 层中，使部署后的模型在单次前向传播中即可完成编辑，无需外部检索。

    arXiv:2609.29269v1 Announce Type: new  Abstract: Knowledge editing changes what a model knows by modifying parameters so that a requested fact updates while unrelated behavior is preserved. This is usually treated as a write problem, but editing also involves an address problem: deciding which hidden states should receive the new residual. An update that activates too narrowly memorizes one prompt, while one that activates too broadly disrupts neighboring knowledge. Parametric editors encode this scope implicitly, whereas memory-based editors make the selection explicit but keep it outside the edited model. We propose ALOE (Addressed Low-rank Operator for Editing), which learns semantic addresses from paraphrases and hard same-subject negatives, aligns them with autoregressive hidden states through rollout refinement and gate calibration, and embeds the resulting gated low-rank operator within one MLP layer, so that the deployed model runs in a single forward pass with no external retr
    
[^158]: Baszta：以数据为中心微调的波兰语多标签安全分类器

    Baszta: Data-Centric Fine-Tuning of a Polish Multi-Label Safety Classifier

    [https://arxiv.org/abs/2609.29266](https://arxiv.org/abs/2609.29266)

    本文通过 Focal + R-Drop 目标微调 HerBERT 构建了波兰语多标签内容安全分类器 Baszta，在公平的阈值调优协议下取得微小但统计显著的 micro F1 领先，同时揭示该基准因 97% 犯罪类正例的极端分布失衡，micro F1 已无法区分真实模型与“全标犯罪”的退化策略，macro F1 才是有效指标。

    

    我们通过使用 Focal + R-Drop 目标函数对 allegro/herbert-base-cased（1.24亿参数）进行微调，构建了一个覆盖五个类别（仇恨言论、粗俗语言、性内容、犯罪、自残）的波兰语多标签内容安全分类器，并在共享的分布外 Gadzi Język 基准上与 Bielik Guard（Sójka）进行对比评估。两个系统均在相同的校准数据集上进行了按类别的阈值调优。在这一匹配协议下，我们的模型在 micro F1 上保持微小但统计显著的领先优势，而表面上明显的 macro-F1 领先则不复存在：那是对经过调优的模型与未调优模型进行比较所产生的假象。我们还报告了该 micro 数值的实际意义：由于 Gadzi Język 中 97% 的样本为犯罪类正例，一个对每个输入都标记为犯罪、对其他类别不作任何标记的分类器，在同一测试集上就已能获得 0.910 的 micro F1，因此 micro 指标无法将任何真实系统与这种退化策略区分开，而 macro 才是真正具有区分能力的指标。

    arXiv:2609.29266v1 Announce Type: new  Abstract: We develop a multi-label Polish content-safety classifier by fine-tuning allegro/herbert-base-cased (124M) across five categories (hate, vulgarity, sexual content, crime, self-harm) using a Focal + R-Drop objective, and evaluate the resulting model against Bielik Guard (S\'ojka) on the shared out-of-distribution Gadzi J\k{e}zyk benchmark. Both systems are given per-category threshold tuning on the same calibration split. Under that matched protocol our model holds a small but statistically significant lead in micro F1, while an apparent macro-F1 lead does not survive: it was an artifact of comparing a tuned model against an untuned one. We also report what that micro figure is worth. Because Gadzi J\k{e}zyk is 97% crime-positive, a classifier that flags crime on every input and nothing else already scores 0.910 micro F1 on the same test split, so micro separates neither system from a degenerate strategy and macro is the column that does.
    
[^159]: TP-CRIV：一种用于AI模型的第三方质询-响应身份验证框架

    TP-CRIV: A Framework for Third-Party Challenge-Response Identity Verification of AI Models

    [https://arxiv.org/abs/2609.29264](https://arxiv.org/abs/2609.29264)

    本文提出TP-CRIV框架，使第三方验证者在既无白盒/API访问权限、又无需服务提供方配合的严格黑盒约束下，通过质询-响应机制验证AI模型的真实身份，从而应对日益严重的模型盗用问题。

    

    arXiv:2609.29264v1 公告类型：cross 摘要：人工智能（AI）模型越来越多地通过远程服务进行部署，这使得模型盗用成为一个日益严重的问题。现有方法，包括水印、指纹识别和模型相似性分析，主要依赖于预先定义的证据或直接的行为比较，并未明确评估声称者当前是否拥有并能够使用与所声称模型身份相关的模型依赖信息。在本文中，我们提出了面向AI模型的第三方质询-响应身份验证（TP-CRIV）。TP-CRIV针对一种第三方验证场景：验证者既没有对声称者模型的白盒访问权限，也没有API访问权限，只能通过其普通的黑盒推理接口与可疑的已部署服务进行交互，并且不需要服务提供方提供特定于协议的配合。在这些约束条件下，该框架使验证者能够获得……（摘要原文在此处被截断）

    arXiv:2609.29264v1 Announce Type: cross  Abstract: Artificial intelligence (AI) models are increasingly deployed through remote services, making model misappropriation a growing concern. Existing approaches, including watermarking, fingerprinting, and model similarity analysis, primarily rely on predefined evidence or direct behavioral comparison and do not explicitly evaluate whether the claimant currently possesses and can utilize model-dependent information relevant to the claimed model identity.   In this paper, we propose Third-Party Challenge-Response Identity Verification (TP-CRIV) for AI models. TP-CRIV targets a third-party verification setting in which the verifier has neither white-box nor API access to the claimant's model, can interact with the suspicious deployed service only through its ordinary black-box inference interface, and does not require protocol-specific cooperation from the service provider. Under these constraints, the framework enables the verifier to obtain
    
[^160]: 基于纵向视野的深度学习预测青光眼进展速度并识别快速进展者

    Deep learning of longitudinal visual fields predicts glaucoma progression rate and identifies fast progressors

    [https://arxiv.org/abs/2609.29256](https://arxiv.org/abs/2609.29256)

    提出GLAM深度学习框架，利用纵向视野序列与临床特征通过注意力融合和不确定性建模，以远少于传统回归所需的检查次数准确预测青光眼进展速度并高效识别快速进展者。

    

    青光眼是导致不可逆失明的首要原因，及时识别快速进展者对于预防残疾至关重要。目前的临床实践是通过平均偏差（MD）随时间的普通最小二乘回归来估计进展，需要数年内进行6至10次视野（VF）检查才能获得可靠的斜率。我们提出了GLAM（青光眼纵向分析模型），这是一个深度学习框架，它接收纵向Humphrey 24-2总偏差序列以及五项临床特征，利用基于注意力的融合和偶然不确定性来预测MD和视野指数的进展率。在开放获取的华盛顿大学Humphrey视野数据集（4,276只患者眼）上，GLAM实现了MD进展率的平均绝对误差为0.139 dB/年（R² = 0.927；较岭回归基线降低73.5%），快速进展者检测的AUC达到0.990。仅使用视野的深度学习可以匹敌多模态流程……（摘要原文在此处截断）

    arXiv:2609.29256v1 Announce Type: cross  Abstract: Glaucoma is the leading cause of irreversible blindness, and timely identification of fast progressors is essential to prevent disability. Current practice estimates progression by ordinary least-squares regression of mean deviation (MD) on time, requiring 6--10 visual field (VF) tests over several years to obtain a reliable slope. We present GLAM (Glaucoma Longitudinal Analysis Model), a deep learning framework that ingests longitudinal Humphrey 24-2 total deviation sequences with five clinical features and predicts MD and visual field index progression rates using attention-based fusion and aleatoric uncertainty. On the open-access University of Washington Humphrey Visual Field dataset (4,276 patient-eyes), GLAM achieved an MD-rate mean absolute error of 0.139 dB yr$^{-1}$ ($R^2 = 0.927$; 73.5% reduction over a ridge baseline) and an AUC of 0.990 for fast-progressor detection. VF-only deep learning can match multimodal pipelines for 
    
[^161]: 策略即代码：一种协程桥接框架实现CAR-bench上快速推理的可靠性

    Policy as Code: A Coroutine-Bridge Harness for Fast-Reasoning Reliability on CAR-bench

    [https://arxiv.org/abs/2609.29251](https://arxiv.org/abs/2609.29251)

    该论文提出协程桥接框架，让模型仅需生成可在评估器工具交换间阻塞恢复的Python程序，将确定性策略直接编码为代码而非提示规则，使每个任务的模型调用中位数降至2次、模型延迟仅1.8秒，大幅提升工具使用智能体的效率与策略合规可靠性。

    

    CAR-bench用于评估使用工具的智能体在真实世界不确定性下是否保持可靠，它在评估器内部执行每个工具，使得每次工具结果交换都是智能体的一次单独往返。传统的“下一步行动”智能体可以批量执行并行工具调用，但面对依赖调用链时，每轮结果都需要一次模型调用。我们提出了一种协程桥接框架，其中模型唯一的动作是生成一个Python程序，该程序在评估器的工具交换之间原地阻塞和恢复执行。这将模型调用与工具往返解耦：在公共测试集上，智能体每个任务仅需中位数两次模型调用，而传统方式需要七次智能体回合，并在Cerebras gpt-oss-120b上以中位数1.8秒的模型延迟完成完整的多轮任务。由于动作表面是可执行代码，确定性的CAR-bench策略可以直接作为逻辑编码在工具层中，而不是作为提示规则，从而以零推理成本强制合规。

    arXiv:2609.29251v1 Announce Type: new  Abstract: CAR-bench evaluates whether tool-using agents stay reliable under real-world uncertainty, executing every tool inside the evaluator so that each tool-result exchange is a separate agent round-trip. A conventional next-action agent can batch parallel tool calls, but a chain of dependent calls costs it one model call per round of results. We present a coroutine-bridge harness in which the model's only action is to emit a Python program that blocks and resumes in place across evaluator tool exchanges. This decouples model invocation from tool round-trips: on the public test split the agent uses a median of two model calls against seven agent turns per task, resolving a full multi-turn task in a median of 1.8 s of model latency on Cerebras gpt-oss-120b. Because the action surface is executable code, deterministic CAR-bench policies are encoded directly as logic in the tool layer rather than as prompt rules, enforcing compliance at zero reaso
    
[^162]: 没有免费的午餐：随着语料库规模增长，语料库任务复杂度至关重要

    No More Free Lunch: Corpus Task Complexity Matters as Corpora Grow

    [https://arxiv.org/abs/2609.29245](https://arxiv.org/abs/2609.29245)

    该论文提出了语料库任务复杂度（CTC）这一新概念来刻画任务难度随语料库规模增长的方式，并引入10个高CTC新任务，发现这类任务不仅对长上下文语言模型更具挑战性，还颠覆了许多现有的建模结论。

    

    给定一个大型语料库，人们可能提出的问题多种多样——从“第一例人类心脏移植手术是什么时候进行的？”到“这篇文献中所有相互矛盾的观点是什么？”——但究竟是什么使某些问题比其他问题更具挑战性？在这项工作中，我们定义了语料库任务复杂度的概念，通过任务难度随语料库规模增长的方式来刻画任务；例如，检索查询只需对语料库进行一次线性扫描，而寻找矛盾论断则需要检查呈二次方增长的论断对集合。我们观察到，先前的工作大多只研究了难度随语料库规模呈线性增长的任务（我们称之为低CTC任务），为此我们引入了10个属于高CTC类别的新任务，其难度随语料库规模呈二次方或更高增长。我们发现，高CTC任务不仅使长上下文语言模型（LCLMs）在更长上下文中面临的平均难度大幅提升，还颠覆了许多既有的建模结论。

    arXiv:2609.29245v1 Announce Type: cross  Abstract: Given a large corpus, the questions one might ask can vary -- from "When was the first human heart transplant?" to "What are all the contradictory claims in this literature?" -- but what makes some questions more challenging than others? In this work, we define a notion of Corpus Task Complexity (CTC) that characterizes tasks by how their difficulty grows with corpus size; for instance, a retrieval query only requires a single linear pass over a corpus, while finding contradictions requires checking a quadratically growing set of claim pairs. Observing that prior work has largely only studied tasks whose difficulty grows linearly with corpus size, which we call low CTC tasks, we introduce 10 new tasks belonging to a class of high CTC whose difficulty grows quadratically or more in corpus size. We find that high-CTC tasks not only grow much more challenging on average at longer contexts for LCLMs, they reverse many modeling conclusions 
    
[^163]: TOLA：面向基于扩散模型的文本图像超分辨率的文本感知单步潜空间自适应

    TOLA: Text-aware One-Step Latent Adaptation for Diffusion-based Text Image Super-Resolution

    [https://arxiv.org/abs/2609.29240](https://arxiv.org/abs/2609.29240)

    TOLA提出了一个无需迭代图像-文本扩散的文本感知单步潜空间自适应框架，通过置信度加权文本条件模块抑制不可靠OCR预测，解决了现有扩散方法计算成本高且错误文本先验被反复放大为语义错误字符的问题。

    

    文本图像超分辨率（TSR）旨在在未知退化条件下恢复视觉上逼真且可读的文本。现有的基于扩散的方法通常依赖于对高分辨率图像或其文本先验的多步预测，导致高昂的计算成本和严重的推理延迟。更关键的是，错误的文本先验可能会被反复注入去噪过程中，导致图像和文本预测相互强化，并将早期识别错误逐步放大为锐利但语义错误的字符。为了解决这些局限性，我们提出了TOLA，一个无需迭代图像-文本扩散的文本感知单步潜空间自适应框架。TOLA包含两个关键模块。首先，置信度加权文本条件模块仅构建一次语义条件，并在不可靠的OCR预测污染图像重建之前将其抑制。其次，一个轻量级的潜

    arXiv:2609.29240v1 Announce Type: cross  Abstract: Text image super-resolution (TSR) aims to recover visually faithful and readable text under unknown degradations. Existing diffusion-based methods typically rely on multi-step prediction of either the high-resolution image or its text prior, resulting in prohibitive computational cost and inference latency. More critically, an erroneous text prior may be repeatedly injected into the denoising process, causing image and text predictions to reinforce each other and progressively amplify an early recognition error into a sharp yet semantically incorrect character. To address these limitations, we propose TOLA, a Text-aware One-step Latent Adaptation framework without iterative image-text diffusion. TOLA consists of two key modules. First, a confidence-weighted text conditioning module constructs the semantic condition only once and suppresses unreliable OCR predictions before they contaminate image reconstruction. Second, a lightweight la
    
[^164]: SARFusion：面向鲁棒相机-激光雷达3D目标检测的场景感知路由融合

    SARFusion: Scene-Aware Routing Fusion for Robust Camera-LiDAR 3D Object Detection

    [https://arxiv.org/abs/2609.29235](https://arxiv.org/abs/2609.29235)

    提出SARFusion，将鲁棒的相机-激光雷达融合重新建模为场景感知的分支路由问题，通过自适应路由机制应对不同驾驶场景与目标查询下模态可靠性的变化，从而实现鲁棒的3D目标检测。

    

    相机-激光雷达融合已成为自动驾驶中3D目标检测的主流范式。然而，现有融合检测器通常通过从紧密耦合的多模态表示中解码目标查询来建立强模态间依赖关系。在退化的驾驶条件下，这种依赖关系使检测器容易受不可靠模态的影响，退化的观测可能干扰可靠的模态特定证据，导致次优预测。此外，模态可靠性会随全局驾驶场景和单个目标查询而变化，因此需要在更细粒度上进行自适应融合决策。为弥补这一差距，我们将鲁棒的相机-激光雷达融合重新表述为场景感知的分支路由问题，并提出了一种鲁棒的3D目标检测器SARFusion。SARFusion不再从单一融合表示生成检测结果，而是将目标查询解码解耦为三个并行推理...

    arXiv:2609.29235v1 Announce Type: cross  Abstract: Camera-LiDAR fusion has become a prevailing paradigm for 3D object detection in autonomous driving. However, existing fusion detectors often establish strong inter-modality dependencies by decoding object queries from tightly coupled multimodal representations. Under corrupted driving conditions, such dependencies make the detector vulnerable to unreliable modalities, where degraded observations may interfere with reliable modality-specific evidence and lead to suboptimal predictions. Moreover, modality reliability can vary across both global driving scenes and individual object queries, requiring adaptive fusion decisions at a finer granularity. To bridge this gap, we reformulate robust camera-LiDAR fusion as a scene-aware branch routing problem and propose SARFusion, a robust 3D object detector. Instead of producing detections from a single fused representation, SARFusion decouples object-query decoding into three parallel reasoning 
    
[^165]: 后训练会在无关决策上留下行为阴影

    Post-Training Leaves Behavioral Shadows on Unrelated Decisions

    [https://arxiv.org/abs/2609.29233](https://arxiv.org/abs/2609.29233)

    该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。

    

    我们发现语言模型可以通过任务无关的文本传递能力。后训练通常使用任务特定的数据来改进语言模型。先前关于“潜意识学习”的研究表明，这些更新的信息可以通过无关的生成内容传递，但其研究主要集中于使用大量教师输出时的特质或偏好。我们提出了主动无任务蒸馏，仅使用教师模型在每个提示中输出的单个词即可实现能力传递。ATD通过选择教师模型和学生模型共同的公共祖先在两个普通词之间几乎无差异的提示，来探测后训练所留下的行为阴影。从这个祖先初始化的学生模型仅通过学习产生的提示-词对进行训练，无需目标任务示例、教师模型的logits或教师模型参数。在以Qwen2.5-1.5B进行的主要编程实验中，5,664个样本在HumanEval+上带来了5.34个百分点的提升。

    arXiv:2609.29233v1 Announce Type: cross  Abstract: We find that language models can transfer capabilities through task-unrelated text. Post-training typically improves language models using task-specific data. Prior work on subliminal learning shows that information about these updates can pass through unrelated generations, but has largely focused on traits or preferences using extensive teacher outputs. We introduce Active Taskless Distillation (ATD), which achieves capability transfer using only a single word from the teacher per prompt. ATD probes the behavioral shadow of post-training by selecting prompts where the teacher and student's shared public ancestor is nearly indifferent between two ordinary words. A student initialized from this ancestor learns solely from the resulting prompt-word pairs, without target-task examples, teacher logits, or teacher parameters. In the primary coding experiment with Qwen2.5-1.5B, 5,664nses yield a 5.34 pp gain on HumanEval+ over an exact nuis
    
[^166]: 面向自主智能系统中行为树与有限状态机的大语言模型驱动统一转换框架

    Towards An LLM-Driven Unified Conversion Framework for BT and FSM in Autonomous Intelligent Systems

    [https://arxiv.org/abs/2609.29228](https://arxiv.org/abs/2609.29228)

    该论文提出了一种由大语言模型驱动的统一转换框架，通过新颖的循环执行行为树结构和结合LLM提示的深度压缩策略，实现了有限状态机与行为树之间自动、高效且语义一致的相互转换，同时保持行为完整性并避免模型复杂度爆炸。

    

    有限状态机（FSM）和行为树（BT）是自主智能系统中被广泛采用的行为建模范式。虽然两者在功能上等价且原则上可以相互转换，但现有的FSM与BT之间的转换方法在保持行为完整性和避免模型复杂度爆炸方面面临重大挑战。为克服这些问题，我们提出了一个由大语言模型（LLM）驱动的统一转换框架，实现FSM与BT之间自动、高效且语义一致的转换。具体而言，我们设计了一种新颖的循环执行BT结构，使LLM能够准确捕捉FSM中的循环结构，从而保持行为完整性。为缓解BT到FSM转换中的状态爆炸问题，我们引入了结合LLM提示的深度压缩策略以消除冗余控制节点，并辅以差异化的分层转换规则，共同减少……

    arXiv:2609.29228v1 Announce Type: new  Abstract: Finite state machine (FSM) and behavior trees (BT) are widely adopted behavioral modeling paradigms for autonomous intelligent systems. While functionally equivalent and inter-convertible in principle, existing transformation methods between FSM and BT face major challenges in preserving behavioral completeness and avoiding model complexity explosion. To overcome these issues, we propose an LLM-driven unified conversion framework that enables automatic, efficient, and semantically consistent transformation between FSM and BT. Specifically, a novel loop execution BT structure is designed for LLM to accurately capture the loop structure in FSM, thereby preserving behavioral completeness. To mitigate the state explosion problem in BT-to-FSM conversion, a depth compression strategy is introduced with LLM prompt to eliminate redundant control nodes, complemented by differentiated hierarchical conversion rules that collectively reduce the numb
    
[^167]: FB-GDM：基于无监督变分推断的全贝叶斯引导扩散模型，用于高维线性逆问题

    FB-GDM: Fully-Bayesian Guided Diffusion Models for High-Dimensional Linear Inverse Problems via Unsupervised Variational Inference

    [https://arxiv.org/abs/2609.29216](https://arxiv.org/abs/2609.29216)

    FB-GDM提出了一种全贝叶斯引导扩散方法，通过在每个反向扩散步骤中用变分推断自动估计两个精度参数，免除了针对具体任务且需依赖真值的人工超参数校准，同时借助可分离分解保持线性计算复杂度，成本与一次ΠGDM运行相当。

    

    扩散模型是线性逆问题的强大先验，但现有的参考引导方法——扩散后验采样（DPS）和伪逆引导扩散模型（ΠGDM）——依赖于需要针对每个任务调整的标量超参数，且通常需要借助真值来进行调节。我们提出了FB-GDM，一种完全贝叶斯的引导扩散方法，它消除了这一校准步骤。从ΠGDM的高斯近似出发，我们推导出了依赖于两个精度参数（即方差的倒数）的闭式条件分数，其中一个与去噪近似相关，另一个与观测似然相关，并将这两个参数视为潜变量，在每个反向扩散步骤中通过变分推断进行估计。一种可分离的分解方式使得每次更新的计算量与像素数量呈线性关系，因此推断在全图像分辨率下依然可以高效进行，其计算成本仅相当于运行一次ΠGDM。FB-GDM既不需要噪声水平信息，也不需要真值（原文此处被截断）……

    arXiv:2609.29216v1 Announce Type: cross  Abstract: Diffusion models are powerful priors for linear inverse problems, but the reference guidance methods, Diffusion Posterior Sampling (DPS) and Pseudoinverse-Guided Diffusion Models ($\Pi$GDM), rely on scalar hyperparameters tuned per task, usually against the ground truth. We introduce FB-GDM, a fully-Bayesian guided diffusion method that removes this calibration step. Starting from the Gaussian approximation of $\Pi$GDM, we derive a closed-form conditional score that depends on two precision parameters (inverse variances), one associated with the denoising approximation and one with the observation likelihood, and treat them as latent variables inferred by variational inference at each reverse step. A separable factorization makes each update scale linearly with the number of pixels, so the inference stays tractable at full image resolution, at a cost comparable to one $\Pi$GDM run. FB-GDM requires neither the noise level nor the ground
    
[^168]: ASIRF：一个面向上下文相关敏感信息脱敏的智能体框架

    ASIRF: An Agentic Framework for Context-Dependent Sensitive Information Redaction

    [https://arxiv.org/abs/2609.29191](https://arxiv.org/abs/2609.29191)

    该论文提出ASIRF智能体框架，通过在推理时从知识库检索领域特定的敏感信息定义，无需重新训练即可适应新领域进行敏感信息脱敏，在85%的模型-领域组合中召回率超越了OpenAI隐私过滤器。

    

    敏感信息是由领域和意图定义的，而非一个通用类别，然而诸如隐私过滤器和命名实体识别器等脱敏系统在训练时便固定了分类体系，导致每进入一个新领域都需要重新训练。我们提出了ASIRF（智能体敏感信息脱敏框架），它在推理时根据输入所属领域从灵活的知识库中检索领域特定的定义，无需重新训练即可适应新领域。我们评估了两种架构——三次调用的多智能体流水线和单智能体变体——涵盖十个小型开源权重模型和八个数据集（包括分布外的虚构领域），并以OpenAI隐私过滤器（OPF）作为基于训练分类器的基线。每个领域仅需数十条专家撰写的定义且不使用任何训练数据，ASIRF在80个模型-领域组合中的68个（85%）上，召回率至少由两种架构之一超过了OPF

    arXiv:2609.29191v1 Announce Type: new  Abstract: Sensitive information is defined by domain and intent, not a universal category, yet redaction systems such as privacy filters and named-entity recognizers fix a taxonomy at training time, requiring retraining for each new domain. We introduce ASIRF (Agentic Sensitive Information Redaction Framework), which retrieves domain-specific definitions based on the input's domain from a flexible knowledge base at inference time, needing no retraining to adapt. Two architectures, a three-call multi-agent pipeline and a single-agent variant, are evaluated across ten small open-weight models and eight datasets, including out-of-distribution fictional domains, against the OpenAI Privacy Filter (OPF) as a trained-classifier baseline. With only a few dozen expert-authored definitions per domain and no training data, ASIRF's recall exceeds OPF's in 68 of 80 model-domain combinations (85 percent), by at least one of the two architectures, with shortfall
    
[^169]: 当诚实在AI辩论中并不足够时

    When Honesty is Not Enough in AI Debate

    [https://arxiv.org/abs/2609.29189](https://arxiv.org/abs/2609.29189)

    该论文提出战略互动监督（SIO）框架，揭示了即使AI智能体的论证完全诚实且结论正确，它们仍可通过选择、表述和排序正确声明的剩余自由来操纵验证者所获信息并追求潜在目标，因此仅激励诚实论证不足以保障AI辩论式监督的安全性。

    

    可扩展监督旨在验证那些能力超过其监督者的智能体的行为。AI辩论被提出作为一种监督解决方案，其中相互竞争的智能体帮助资源有限的验证者评估其无法独立可靠评估的声明。这一方案的大部分前景依赖于激励诚实的论证，从而导向正确的结论。然而，正确的结论并不必然唯一决定用于支持它的论证。智能体可能保留自由裁量权，决定呈现哪些正确声明、如何表述它们，以及以何种顺序披露它们。这种剩余的自由可以使智能体塑造验证者所学到的超出任务相关结论之外的内容，从而在不损害结论正确性的情况下追求潜在目标。为了研究这一现象，我们引入了战略互动监督框架，该框架将监督同时视为一种验证机制和一种战略沟通渠道。

    arXiv:2609.29189v1 Announce Type: new  Abstract: Scalable oversight aims to verify the behaviour of agents whose capabilities exceed those of their overseers. AI debate has been proposed as an oversight solution in which competing agents help a resource-limited verifier assess claims that it cannot reliably evaluate unaided. Much of its promise rests on incentivizing honest arguments that lead to correct verdicts. Yet a correct verdict need not uniquely determine the arguments used to support it. Agents may retain discretion over which correct claims to present, how to frame them, and in what order to disclose them. This residual freedom can allow agents to shape what the verifier learns beyond the task-relevant conclusion, pursuing latent objectives without compromising verdict correctness. To study this phenomenon, we introduce the framework strategic interactive oversight (SIO), which treats oversight jointly as a verification mechanism and a strategic communication channel. Within 
    
[^170]: 熵三角法（ETM）：基于超过10,000名患者研究的心律失常预防新型框架

    The Entropy Triangle Method (ETM): A novel framework for the prevention of cardiac arrhythmia with a review of more than 10,000 patients

    [https://arxiv.org/abs/2609.29187](https://arxiv.org/abs/2609.29187)

    该论文提出熵三角法这一新型机器学习框架，通过特征工程、熵三角过采样和疾病预测三个步骤，首次在包含10,646名患者12导联ECG数据上实现了对非窦性心律超过85%准确率的预测。

    

    促进预测是医学中最重要的难题之一。在本研究中，我们提出了熵三角法，这是一种利用新型机器学习技术预测心律的新颖框架。该框架包含三个步骤：特征工程、熵三角过采样和疾病预测。本研究使用的数据集是一个包含10,646名患者的12导联心电图（ECG）心律失常研究数据库，该数据集包含11种不同的心律（5种窦性心律和6种非窦性心律）。本文在机器学习和医学领域提出了两项“首次”成果，能够以超过85%的准确率预测非窦性心律。我们的实验结果表明，其中基于熵三角形的最准确分类器是支持向量分类器，而最有用的过采样方法是shark scent过采样技术。

    arXiv:2609.29187v1 Announce Type: new  Abstract: One of the most important problems in medicine is to facilitate prediction. In this study, we propose entropy triangle method, a novel framework for predicting heart rhythms using a novel machine learning technique. This framework includes three steps: feature engineering, entropy triangle oversampling, and disease prediction. The dataset used in this study is a 12-lead electrocardiogram (ECG) arrhythmia research database with 10,646 patients. This dataset contains 11 different heart rhythms (5 sinus rhythms and 6 non-sinus rhythms). In this article, we introduce two firsts in machine learning and medicine that can predict non-sinus rhythm with over 85% accuracy. Our experimental results show, among others, that the most accurate classifier based on entropy triangles and the most useful oversampling are the supported vector classifiers and oversampling techniques for shark scent.
    
[^171]: 基于强化学习的分类算法正确选择用于非酒精性脂肪肝预测

    Right Choice of Classification Algorithms Based on Reinforcement Learning for Prediction of Non-Alcoholic Fatty Liver

    [https://arxiv.org/abs/2609.29181](https://arxiv.org/abs/2609.29181)

    本研究提出了一种名为“平方学习”（SL）的基于强化学习评分方法的新算法，能够自动学习并选择最合适的分类算法用于疾病预测。

    

    arXiv:2609.29181v1 公告类型：new 摘要：人工智能领域存在许多复杂问题。其中一些问题可以通过其他人工智能方法来解决，这被称为“用人工智能解决人工智能问题”（AI for AI）。寻找合适的分类器算法是一项耗时的任务。因此，一种能够自动学习如何选择分类算法的算法非常重要。分类算法在预测各种疾病方面非常有用。同时，原发性胆汁性肝硬化是最广为人知的可以通过分类算法进行预测的疾病之一。本研究最重要的成就和创新之处在于通过一种称为“平方学习”（Square Learning, SL）的强化学习评分方法实现学习的自动提升。本研究提出了一种算法，该算法能够学习自动选择合适的分类算法来预测原发性胆汁性肝硬化。在这篇文...

    arXiv:2609.29181v1 Announce Type: new  Abstract: There are many complex issues in the world of artificial intelligence. Some of these problems are solved using other artificial intelligence methods, which are called artificial intelligence for artificial intelligence. Finding an appropriate classifier algorithm is a time-consuming task. For this reason, an algorithm that can automatically learn the choice of classification algorithms is very important. Classification algorithms are useful in predicting various diseases. Also, Primary Biliary Cirrhosis is one of the most well-known diseases that have been predicted by classification algorithms. This research's most significant achievement and novelty is the automatic increase in learning through a scoring method of reinforcement learning is called square learning (SL). In this research, an algorithm is presented that learns to automatically select the appropriate classification algorithm to predict Primary Biliary Cirrhosis. In this art
    
[^172]: 定位、分离与增强：音频混音的全生成式方法

    Spot, Separate, and Enhance: Fully Generative Approach for Audio Mixing

    [https://arxiv.org/abs/2609.29169](https://arxiv.org/abs/2609.29169)

    提出首个多模态、用户引导的音频重混音与增强生成模型 SSE，可在视频与文本引导下实现音频重平衡、音源去除和混响消除，并借助新数据集 DegradedMix 在可控性与重混音质量上超越现有方法。

    

    我们提出了“定位、分离与增强”（SSE），这是首个用于音频重混音与增强的多模态、用户引导生成模型。SSE 在视频和文本描述的共同引导下，通过重新平衡音频、去除不需要的音源以及减少混响来增强视频内容。为支持其训练与评估，我们提出了 DegradedMix，一个基于音频重混音基准 MuddyMix 构建的新数据集。此外，我们还采用了源自生成式建模的评估指标，相比标准的基于重构的指标，这些指标能更好地捕捉重混音的创造性本质。大量实验表明，SSE 在可控性和重混音质量方面均优于现有基线方法。

    arXiv:2609.29169v1 Announce Type: cross  Abstract: We introduce Spot, Separate, and Enhance (SSE), the first multimodal, user-guided generative model for audio remixing and enhancement. SSE enhances video content by rebalancing the audio, removing unwanted audio sources, and reducing reverberation, guided by both video and textual descriptions. To support its training and evaluation, we propose DegradedMix, a new dataset built on the audio remixing benchmark MuddyMix. We also adopt evaluation metrics from generative modeling, which better capture the creative nature of remixing than standard reconstruction-based metrics. SSE outperforms existing baselines in both controllability and remixing quality, as shown by extensive experiments. Project page: https://sse-ai.notion.site
    
[^173]: IndicBankBench：评估印度零售银行业中语言模型助手的安全性与可靠性

    IndicBankBench: Evaluating Safety and Reliability of Language Model Assistants in Indian Retail Banking

    [https://arxiv.org/abs/2609.29167](https://arxiv.org/abs/2609.29167)

    该论文提出了IndicBankBench——一个包含799个案例的印度零售银行业基准，通过安全性、工具使用、回复充分性和咨询质量四个阶段的确定性检查与LLM评判，并采用要求三次试验全部成功的严格pass³指标，系统评测了十一个语言模型银行助手的安全性与可靠性。

    

    银行助手必须使用账户特定信息来回答请求，并且在许多情况下需要通过工具执行操作。仅评估最终回复会遗漏重要的错误。助手可能会询问它已经掌握的信息、依赖过时的上下文、选择错误的账户，或者在说出正确值之后却写入无效值。我们提出了IndicBankBench，这是一个包含799个案例的印度零售银行业基准测试，涵盖五个运营领域、一个能力/拒绝领域以及二十个主要评估维度。案例在四个阶段进行评估：安全性、操作与工具使用、回复充分性和咨询质量。工具使用和大多数安全检查是确定性的。一个窄域解析器仅处理写入前确认含糊不清的案例，而单独的LLM评判器评估语义层面的回复充分性。我们对每个案例运行三次，并报告严格的pass³指标，即要求所有试验均成功。在所评估的十一个模型中，严格可靠性表现……

    arXiv:2609.29167v1 Announce Type: new  Abstract: Banking assistants must use account-specific information to answer requests and, in many cases, take actions through tools. Evaluating only the final response misses important errors. An assistant may ask for information it already has, rely on stale context, select the wrong account, or write an invalid value after stating the correct one. We introduce IndicBankBench, a 799-case benchmark for Indian retail banking spanning five operational domains, a capability/refusal domain, and twenty primary axes. Cases are evaluated at four stages: safety, action and tool use, response adequacy, and advisory quality. Tool use and most safety checks are deterministic. A narrow resolver handles only ambiguous confirmation-before-write cases, while a separate LLM judge evaluates semantic response adequacy. We run every case three times and report strict pass^3, which requires success on all trials. Across the eleven evaluated models, strict reliabilit
    
[^174]: HarnessPAI：面向物理人工智能（Physical AI）的演化式Harness框架

    HarnessPAI: An Evolving Harness for Physical AI

    [https://arxiv.org/abs/2609.29166](https://arxiv.org/abs/2609.29166)

    提出HarnessPAI框架，以可执行、可演化的代码作为接口组织动作原语，在一次执行内以固定程序开环引导，在多次执行间通过反馈闭环演化程序并将失败蒸馏为可复用技能，从而弥补物理AI动作模型在感知与推理上的不足。

    

    物理人工智能（Physical AI）旨在构建能够感知世界、理解并推理世界、并决定如何行动的具身智能体。然而，该领域目前主要聚焦于最后一个环节：将观测映射为低层控制的动作模型。当前主流的训练方案可能会削弱鲁棒行为所需的感知与推理能力，使得即使是强大的动作模型在面对场景扰动和长时程任务时也依然脆弱。我们提出了HarnessPAI，这是一个面向物理AI的、与模型和具身形态无关的Harness框架，它将代码视为可执行且可演化的接口，用以组织底层的动作原语。该框架区分了两个时间尺度：在一次rollout内部，它以程序为单位开环执行，由固定的程序来引导并校验执行过程；跨rollouts时，它则闭环演化，利用执行反馈来修订程序，并将失败经验蒸馏为可复用的技能。在桌面机器人……（原文此处截断）

    arXiv:2609.29166v1 Announce Type: cross  Abstract: Physical AI aims to build embodied agents that perceive the world, understand and reason about it, and decide how to act. Yet the field has focused primarily on the last component: the action model that maps observations to low-level controls. The prevailing training recipe can erode the perceptual and reasoning capabilities needed for robust behavior, leaving even strong action models vulnerable to scene perturbations and long-horizon tasks. We introduce HarnessPAI, a model- and embodiment-agnostic Harness framework for Physical AI that treats code as the executable and evolvable interface that organizes the underlying action primitive. The framework separates two timescales: within a rollout, it executes open-loop at the program level, with a fixed program guiding and checking execution; across rollouts, it evolves closed-loop, using execution feedback to revise the program and distill failures into reusable skills. Across desktop ro
    
[^175]: Med-AR：面向长尾胸部X光分类与不确定性感知评估的自回归视觉-语言预训练

    Med-AR: Autoregressive Vision-Language Pretraining for Long-Tailed Chest X-Ray Classification and Uncertainty-Aware Evaluation

    [https://arxiv.org/abs/2609.29156](https://arxiv.org/abs/2609.29156)

    该研究提出放射学原生的自回归视觉-语言预训练模型Med-AR-8B和Med-AR-2B，在长尾胸部X光多标签分类任务上显著超越现有预训练方法，尤其大幅提升了罕见病变的识别性能（MIMIC-CXR尾部标签平均AUPRC从0.1033提升至0.1441）。

    

    长尾胸部X光分类需要能够同时捕捉常见异常以及细微、罕见病变的视觉表征。我们提出了Med-AR-8B和Med-AR-2B两个放射学原生的自回归视觉-语言模型，它们使用结构化报告、以异常为重点的文本以及区域标注进行预训练。我们采用统一的ML-Decoder分类头，将这些视觉编码器迁移到多标签分类任务中，并与对比学习、自监督及有监督预训练的编码器（包括Med-CLIP、CheXFound、EVA-Base、ARK和BioViL-T）进行比较。为评估细粒度识别能力，我们还为MIMIC-CXR和CheXpert构建了由报告衍生并经大语言模型扩展的标签集。在PadChest、MIMIC-CXR和CheXpert三个数据集上，Med-AR-8B在头部、中部和尾部病变的平均AUROC和AUPRC上均优于Med-CLIP。在MIMIC-CXR上，它将尾部标签的平均AUPRC从0.1033提升至0.1441。Med-AR-2B实现了最强的（摘要在此处截断）

    arXiv:2609.29156v1 Announce Type: cross  Abstract: Long-tailed chest X-ray classification requires visual representations that capture both common abnormalities and subtle, infrequent findings. We propose Med-AR-8B and Med-AR-2B, two radiology-native autoregressive vision-language models pretrained with structured reports, abnormality-focused text, and region annotations. We evaluate the transfer of their visual encoders to multi-label classification against contrastive, self-supervised, and supervised pretrained encoders, including Med-CLIP, CheXFound, EVA-Base, ARK, and BioViL-T, using a common ML-Decoder classification head. To assess fine-grained recognition, we also construct LLM-expanded, report-derived label sets for MIMIC-CXR and CheXpert. Across PadChest, MIMIC-CXR, and CheXpert, Med-AR-8B outperforms Med-CLIP in mean AUROC and AUPRC for head, medium, and tail findings. On MIMIC-CXR, it increases tail-label mean AUPRC from 0.1033 to 0.1441. Med-AR-2B achieves the strongest dis
    
[^176]: 走错一步不会毁掉整个旅程：面向大语言模型智能体的偏差引导技能自我进化

    A Wrong Turn Does Not Ruin the Journey: Deviation-Guided Skill Self-Evolution for LLM Agents

    [https://arxiv.org/abs/2609.29154](https://arxiv.org/abs/2609.29154)

    提出SkillPivot框架，通过定位失败轨迹中从有效进展转向错误偏离的转折点，并让更强的教师模型从该有效前缀重新继续执行，从而实现大语言模型智能体的技能自我进化。

    

    大语言模型智能体越来越依赖自然语言技能来解决复杂的工具使用任务。然而，此类任务通常存在多条有效的解决路径，因此通过强制失败轨迹去匹配某个固定的成功轨迹来改进技能并不合适。此外，失败轨迹很少是完全错误的：智能体可能首先收集了有用的证据并取得了有意义的进展，只是在后续才偏离进入错误的尾部。因此，我们认为技能自我进化应当识别出富有成效的问题求解过程在何处开始崩溃，而不是对整个失败进行粗略反思。基于这一洞察，我们提出了SkillPivot，一个以偏差点为引导的技能自我进化框架。SkillPivot利用执行有效性、目标进展和动作多样性来检测轨迹从有用前缀转变为错误后缀的转折点。随后，一个更强的教师模型从相同的前缀出发继续执行，并生成成功（的轨迹）……

    arXiv:2609.29154v1 Announce Type: new  Abstract: Large language model agents increasingly rely on natural-language skills to solve complex tool-use tasks. However, such tasks often admit multiple valid solution paths, making it inappropriate to improve skills by forcing failed trajectories to match a fixed successful trajectory. Moreover, failed trajectories are rarely entirely wrong: an agent may first collect useful evidence and make meaningful progress, but later deviate into an erroneous suffix. We therefore argue that skill self-evolution should identify where productive problem solving begins to break down, rather than reflect coarsely over the entire failure. Based on this insight, we propose SkillPivot, a deviation-point-guided framework for skill self-evolution. SkillPivot detects the transition from a useful prefix to an erroneous suffix using execution validity, goal progress, and action diversity. A stronger teacher then continues from the same prefix and produces a success
    
[^177]: 面向生成式搜索的声明门控式来源风险审计

    Claim-Gated Source-Risk Auditing for Generative Search

    [https://arxiv.org/abs/2609.29145](https://arxiv.org/abs/2609.29145)

    该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。

    

    生成式搜索的答案可能引用了有依据的段落，却遗漏了会改变其解释方式的来源关系。我们规定了一种针对“查询-来源-答案”三元组的声明门控审计方法。只有当关系证据、答案采纳、重要性和披露全部被观测到时，遗漏才被判定为“已解决”；证据不完整时保持“未解决”状态，而不是被默认当作独立性处理。该规范将此判定终点与引用支持和审查优先级区分开来，并将决策绑定到带版本号的证据片段上。一个参考检查器使记录契约变得可执行。在一个穷举式的合成测试套件上，该检查器复现了全部81种三态谓词组合，并拒绝了192条故意构造的畸形记录。公共防护基线和谓词消融实验将终点逻辑与缺失证据的处理区分开来，而受控转换则检验了支持分离与证据移除。这些是有限的契约符合性结果，而非检测器性能……（原文摘要截断）

    arXiv:2609.29145v1 Announce Type: new  Abstract: A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query-source-answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detec
    
[^178]: 先界定范围再持久化：防止智能体记忆中的跨任务族干扰

    Scope Before You Persist: Preventing Cross-Family Interference in Agent Memory

    [https://arxiv.org/abs/2609.29144](https://arxiv.org/abs/2609.29144)

    论文提出 Scoped-ORC 方法，通过将持久技能的检索范围严格限定在其原始任务族，防止跨任务族干扰，从而在不更新模型权重的情况下显著提升智能体的轨迹效用并消除有害的技能部署。

    

    持久化记忆使语言模型智能体能够在不更新模型权重的情况下改进提示词和技能。我们证明，将检索范围与认证范围相匹配，可以使这些编辑在重复出现的任务族中支持可靠的反复适应。我们在 ProcStream-RSI（一个12轮代码修复流）上研究冻结模型智能体，并采用正交回归控制（ORC），这是一种以执行结果为依据的持久技能编辑门控机制。在一项保持提案和门控决策不变的干预实验中，仅在原始任务族中检索每个已接受的技能，将平均隐藏轨迹效用从全局记忆下的0.713提升到0.816，并将有害部署从八个中的六个减少到零个。在27个配对的随机顺序流中，Scoped-ORC 相比 Global-ORC 将平均轨迹效用提高了0.063 [0.037, 0.094]，接受了63个而非12个更新，在19/27个流中产生了多个被接受的更新，且63个被接受的更新中有0个是有害的。

    arXiv:2609.29144v1 Announce Type: new  Abstract: Persistent memory lets language-model agents improve prompts and skills without updating model weights. We show that matching retrieval scope to certification scope enables these edits to support reliable repeated adaptation across recurring task families. We study frozen-model agents on ProcStream-RSI, a 12-round code-repair stream, using Orthogonal Regression Control (ORC), an execution-grounded gate for persistent skill edits. In an intervention that holds proposals and gate decisions fixed, retrieving each accepted skill only for its originating family raises mean hidden trajectory utility from 0.713 under global memory to 0.816 and changes harmful deployments from six of eight to none. In 27 paired randomized-order streams, Scoped-ORC improves mean trajectory utility by 0.063 [0.037, 0.094] over Global-ORC, accepts 63 rather than 12 updates, and produces multiple accepted updates in 19/27 streams, with 0/63 harmful acceptances. The 
    
[^179]: 用于市场研究与数字孪生校准的AI主持访谈

    AI-Moderated Interviews for Market Research and Digital Twins Calibration

    [https://arxiv.org/abs/2609.29143](https://arxiv.org/abs/2609.29143)

    该研究通过与三家行业伙伴开展的大规模预注册对照实验发现，AI主持的访谈在深度上可与人类主持媲美、覆盖更多主题并在同等预算下挖掘出显著更多的客户需求，而由此构建的消费者数字孪生能够有效预测个体对真实营销刺激的响应。

    

    AI主持访谈正作为一种可扩展的市场研究方法兴起，用于生成消费者洞察并构建消费者“数字孪生”。然而，它们能否媲美人类主持的访谈，或是否优于更简单的静态数据收集方法，目前仍不清楚。在一项与三家行业合作伙伴共同开展的预注册被试间研究（N = 317）中，我们比较了AI主持访谈（N = 139）、人类主持访谈（N = 24）和静态访谈（N = 154）。AI主持在深度上与人类主持相当，覆盖更多主题，并且在预算相同的条件下，比人类主持或静态访谈能够挖掘出显著更多的客户需求。然而，当与真人交谈时，参与者在情感上的投入程度听起来更高。随后，我们利用访谈数据创建数字孪生，并将每个孪生的表现与参与者自身对六种真实世界营销刺激的保留响应进行对比评估。我们发现，由AI主持访谈创建的数字孪生在预测……（摘要原文在此处截断）

    arXiv:2609.29143v1 Announce Type: cross  Abstract: AI-moderated interviews are emerging as a scalable market-research method for generating consumer insights and building consumer "digital twins." Yet it remains unclear whether they match human-moderated interviews or improve on simpler, static data collection methods. In a pre-registered, between-subjects study (N = 317) with three industry partners, we compare AI-moderated (N = 139), human-moderated (N = 24), and static interviews (N = 154). AI moderation matches human moderation in depth, covers more themes, and, holding budget constant, recovers significantly more customer needs than human moderation or static interviews. However, participants sound more emotionally engaged when speaking to a live human. We then create digital twins using interview data and evaluate each twin against the participant's own held-out responses to six real-world marketing stimuli. We find that digital twins created from AI-moderated interviews predict 
    
[^180]: 并非每个Token都值得蒸馏：面向Direct-OPD的选择性监督

    Not Every Token Is Worth Distilling: Selective Supervision for Direct-OPD

    [https://arxiv.org/abs/2609.29142](https://arxiv.org/abs/2609.29142)

    该论文揭示了Direct-OPD中token级对数比率监督无法反映教师行为真实变化的缺陷，并提出根据教师参考JSD进行选择性屏蔽监督的方法S²D-OPD，仅在教师行为发生显著变化的token上进行蒸馏。

    

    直接在线策略蒸馏通过将强化学习后与强化学习前检查点之间的token级对数比率，作为学生模型自身生成轨迹上的密集监督信号，把强化学习带来的策略改进从一个小模型迁移到一个更大的学生模型。这种迁移方式在每个状态下都会奖励策略的变化，然而对数比率仅衡量相对变化：即使两个检查点分配给学生模型候选token的概率质量趋于消失，对数比率仍可能保持不变。通过一个精确的构造，我们证明当检查点之间的Jensen-Shannon散度（JSD）以及两个方向的KL散度随着该概率质量一同消失时，Direct-OPD的奖励及其更新仍可保持不变，同时我们指出较小的JSD能够约束教师模型行为变化的幅度。基于这一分析，我们提出了面向Direct-OPD的选择性监督方法（S²D-OPD），它根据教师参考JSD对学生模型采样得到的状态进行排序并屏蔽……（摘要原文不完整）

    arXiv:2609.29142v1 Announce Type: cross  Abstract: Direct On-Policy Distillation (Direct-OPD) transfers reinforcement-learning-induced policy improvements from a small model to a larger student by using the token-level log-ratio between post-RL and pre-RL checkpoints as dense supervision on the student's own rollouts. This transfer rewards the policy shift at every state, yet the log-ratio measures only relative change: it can stay fixed even as the probability mass that both checkpoints assign to the student's candidate tokens vanishes. Through an exact construction, we show that the Direct-OPD reward and its update can remain unchanged while the Jensen-Shannon divergence (JSD) and both KL directions between the checkpoints vanish with this mass, and we note that a small JSD bounds how much the teacher's behavior changed. Motivated by this analysis, we propose Selective Supervision for Direct-OPD (S$^2$D-OPD), which ranks student-sampled states by their teacher-reference JSD and masks
    
[^181]: 硬预算重复评估中诚实不确定性的严格极限

    Sharp Limits for Honest Uncertainty in Hard-Budget Repeated Evaluation

    [https://arxiv.org/abs/2609.29140](https://arxiv.org/abs/2609.29140)

    该论文证明了硬预算重复评估中认证窄不确定性所需区间宽度的最优极限——全任务覆盖时为 Θ([M(t+1)]^{-1/2})，允许省略任务时为 Θ([M(t+√M)]^{-1/2})——并通过分歧证书的随机子集设计与联合均值-分歧区间将其转化为实用的有限预算推断方法。

    

    重复评估能够准确估计基准测试分数，但仍然需要通过重复实验来认证狭窄的不确定性。我们在固定网格上刻画了这一需求：M个任务、每个任务L条二值路径，在硬预算 (M+t)K 之下，其中每条路径最多花费K次响应或回合。对于固定的 L ≥ 3 和 0 < α ≤ 1/12，在最坏纯队列上的最优期望区间宽度为：当每个任务都被观测时为 Θ_{α,L}([M(t+1)]^{-1/2})，当允许省略任务时为 Θ_{α,L}([M(t+√M)]^{-1/2})。这些下界覆盖了自适应硬预算策略，而固定的随机子集设计通过分歧证书能够同时达到这两种速率。一种联合均值/分歧区间将任务覆盖定律转化为实用的有限预算推断方法。在一项等预算的 LiveCodeBench 重放实验中（16个模型、880个任务、每个任务5个输出），任务覆盖设计将中位点估计MSE降低了8（摘要在此处截断）。

    arXiv:2609.29140v1 Announce Type: new  Abstract: Repeated evaluation can estimate a benchmark score accurately while still requiring replication to certify narrow uncertainty. We characterize that requirement on a fixed grid of $M$ tasks with $L$ binary paths per task under the hard budget $(M+t)K$, where each path costs at most $K$ responses or episodes. For fixed $L \ge 3$ and $0 < \alpha \le 1/12$, the optimal expected width on the worst pure cohort is $\Theta_{\alpha,L}([M(t+1)]^{-1/2})$ when every task is observed and $\Theta_{\alpha,L}([M(t+\sqrt{M})]^{-1/2})$ when omission is allowed. The lower bounds cover adaptive hard-budget policies, and fixed random-subset designs attain both rates through disagreement certificates. A joint mean/disagreement interval turns the task-covering law into practical finite-budget inference. In an equal-budget LiveCodeBench replay with 16 models, 880 tasks, and five outputs per task, the task-covering design reduces median point-estimation MSE by 8
    
[^182]: 标签感知的结构化文本翻译：迈向系统性理解

    Tag-Aware Structured Text Translation: Towards a Systematic Understanding

    [https://arxiv.org/abs/2609.29131](https://arxiv.org/abs/2609.29131)

    该论文针对带标签文本翻译中流畅性与标签保真度难以兼顾的问题，提出涵盖数据合成、能力构建和多目标对齐的系统性方法，并通过混合合成策略Hy-LST解决了标签多样性与翻译自然度之间的权衡难题。

    

    互联网文本中充满了承载结构、语义和功能意义的格式标签。当前基于大语言模型（LLM）的翻译系统在处理带标签文本时，难以平衡翻译流畅性与标签保真度。我们认为，解决这一矛盾需要在三个相互关联的层面采取系统性方法：数据合成、能力构建和多目标对齐。在数据层面，我们识别并形式化了合成数据生成中结构标签多样性与翻译自然度之间的根本性权衡；现有方法在优化其中一项时往往以牺牲另一项为代价。我们提出了一种混合合成策略Hy-LST，将基于LLM的标签合成方法与两阶段基于LLM的标签合成方法相结合，以生成既多样又自然的带标签数据。在能力层面，我们将标签感知翻译分解为难度递增的四个子任务，在多任务……

    arXiv:2609.29131v1 Announce Type: cross  Abstract: Internet texts are replete with format tags that carry structural, semantic, and functional meaning. Current large language model (LLM)-based translation systems struggle to balance translation fluency with tag fidelity when processing tagged text. We argue that resolving this tension requires a systematic approach at three interconnected levels: data synthesis, capability building, and multi-objective alignment. At the data level, we identify and formalize a fundamental trade-off between structural tag diversity and translation naturalness in synthetic data generation; existing methods optimize for one at the expense of the other. We propose a hybrid synthesis strategy (Hy-LST) combining LLM-based synthesis tag method and Two-Stage LLM-based synthesis tag method to produce both diverse and natural tagged data. At the capability level, we decompose tag-aware translation into four sub-tasks of increasing difficulty in a multi-task super
    
[^183]: 少即是多：仅编码器的音视频分割

    Less is More: Encoder-only Audio-Visual Segmentation

    [https://arxiv.org/abs/2609.29121](https://arxiv.org/abs/2609.29121)

    提出仅编码器的音视频分割方法EASE，通过去除冗余组件，以更简洁的架构实现了高达365 FPS的推理速度（比现有最先进模型快3倍）和最先进的AVSS性能，且训练仅需不到11个GPU小时。

    

    音视频语义分割（AVSS）旨在识别、分割并分类视频帧中发出声音的对象。以往基于Transformer的AVSS方法在很大程度上继承了图像分割模型的设计原则。近期研究表明，这些图像分割模型中包含对分割性能贡献甚微的冗余组件。基于这一洞察，我们提出了仅编码器的音视频分割方法EASE。EASE的运行速度高达365 FPS，比现有最先进的（SotA）AVS模型快3倍且精度相当，训练时间不足11个GPU小时。此外，我们在不同的骨干网络和输入分辨率下均取得了最先进的AVSS性能。我们的结果表明，AVSS可以做到更简单、更快速，为未来的研究和实时应用提供了可扩展的基础。代码、模型权重和示例可在 https://ease-avs.notion.site 获取。

    arXiv:2609.29121v1 Announce Type: cross  Abstract: Audio-Visual Semantic Segmentation (AVSS) aims to identify, segment, and classify sound-emitting objects in video frames. Previous Transformer-based AVSS approaches largely inherit design principles from image segmentation models. Recent studies show that these image segmentation models contain redundant components that contribute little to the segmentation performance. Following this insight, we propose Encoder-only Audio-Visual Segmentation (EASE). EASE runs at up to 365 FPS, 3x faster than prior State-of-the-Art (SotA) AVS models at comparable accuracy, and trains in under 11 GPU-hours. Furthermore, we achieve SotA AVSS performance across different backbones and input resolutions. Our results demonstrate that AVSS can be both simpler and faster, providing a scalable foundation for future research and real-time applications. Code, model weights, and samples are available at https://ease-avs.notion.site
    
[^184]: CounterRoute：基于分层反事实信用分配的自路由推理

    CounterRoute: Self-Routed Reasoning via Hierarchical Counterfactual Credit Assignment

    [https://arxiv.org/abs/2609.29109](https://arxiv.org/abs/2609.29109)

    CounterRoute是一个在线强化学习框架，通过分层反事实信用分配仅将跨模式信用归因于路由token、用模式内GRPO训练响应token，并结合从成对到自路由的课程机制，实现了无需SFT预热、无需用户干预的自动推理模式路由，有效节省不必要的推理计算。

    

    具备推理能力的语言模型在直接回答已足够的情况下，往往仍会生成冗长的思维链，浪费推理计算资源。许多双模式模型将这一选择留给用户。实现自动化颇具挑战性，因为路由目标会随策略演进、初始模式偏好会破坏探索的稳定性，且序列级目标会将路由与响应学习纠缠在一起。我们提出CounterRoute，这是一个在线强化学习框架，直接从原生双模式检查点出发，在单一共享策略中联合学习路由与模式条件化响应，无需特定方法的SFT预热。成对的当前策略反事实rollout仅将跨模式信用分配给路由token，而模式内GRPO用于训练响应token。从成对到自路由的课程机制通过两种模式的强制rollout稳定早期训练，随后逐步增加自路由更新的比例，以提升自主路由能力。在九个基准测试中，CounterRoute…

    arXiv:2609.29109v1 Announce Type: new  Abstract: Reasoning-capable language models often produce long chains of thought when direct answers suffice, wasting inference compute. Many dual-mode models leave this choice to users. Automating it is challenging because routing targets evolve with the policy, initial mode preferences destabilize exploration, and sequence-level objectives entangle routing with response learning. We introduce CounterRoute, an online reinforcement-learning framework that jointly learns routing and modeconditioned responses in one shared policy directly from a native dual-mode checkpoint, without method-specific SFT warm-up. Paired current-policy counterfactual rollouts assign cross-mode credit only to the routing token, while within-mode GRPO trains response tokens. A paired-to-self-routed curriculum stabilizes early training with forced rollouts from both modes, then increases self-routed updates to improve autonomous routing. Across nine benchmarks, CounterRout
    
[^185]: 欧洲电力交易市场的功能架构：监管约束下AI支持交易系统的需求

    Functional Architecture of European Electricity Trading Markets: Requirements for AI Supported Trading Systems under Regulatory Constraints

    [https://arxiv.org/abs/2609.29108](https://arxiv.org/abs/2609.29108)

    本文提出了一个与欧洲电力市场耦合机制及REMIT、MiFID II等监管合规义务对齐的AI支持交易功能架构，其核心贡献是由决策状态向量、残余敞口核算、约束优化目标、可执行动作许可门控和失效关闭AI控制逻辑组成的形式化系统规范。

    

    欧盟的电力交易作为一个受约束的多层系统运行，其中法律设计、交易所微观结构和电网物理特性在远期、日前、日内和平衡等多个时间尺度上被联合执行。本文开发了一个面向AI支持交易的功能架构，该架构与市场耦合机制、跨区域传输约束以及REMIT、MiFID II、MiFIR和EMIR等法规下的合规义务保持一致。其贡献是一个形式化的系统规范，由决策状态向量、残余风险敞口核算、约束优化目标、可执行动作许可门控以及带有可审计记录的失效关闭AI控制逻辑组成。该分析将主要的指定电力市场运营商（NEMO）交易场所及相关交易所运营商映射为可操作的交易场所拓扑结构，并识别出跨境协调在实践中失效的环节：接口级时序、权限异构性等（摘要原文在此处截断）。

    arXiv:2609.29108v1 Announce Type: new  Abstract: European electricity trading in the EU operates as a constrained multi-layer system in which legal design, exchange microstructure, and network physics are executed jointly across forward, day-ahead, intraday, and balancing horizons. This paper develops a functional architecture for AI-supported trading that is aligned with market-coupling mechanics, cross-zonal transfer constraints, and compliance obligations under REMIT, MiFID II, MiFIR, and EMIR. The contribution is a formal system specification composed of a decision-state vector, residual-exposure accounting, constrained optimization objective, executable-action permission gate, and fail-closed AI control logic with auditable records. The analysis maps major Nominated Electricity Market Operator (NEMO) venues and related exchange operators into an operational venue topology and identifies where cross-border coordination fails in practice: interface-level timing, permission heterogen
    
[^186]: WildHSR：基于3D基础模型的度量前馈式4D人物-场景重建

    WildHSR: Metric Feed-Forward 4D People-Scene Reconstruction from a 3D Foundation Model

    [https://arxiv.org/abs/2609.29106](https://arxiv.org/abs/2609.29106)

    该论文提出WildHSR，通过利用网络视频中人物生成的闭式尺度伪标签预训练尺度读取头，并结合轻量级适配器进行微调，使3D基础模型在推理时无需度量监督即可预测度量尺度并支持持久人物身份，实现前馈式的4D人物-场景重建。

    

    3D基础模型能够在一次前向传播中恢复视频的相机参数和几何结构，但其中一些最强的模型只能恢复到“尺度未定”的程度。要实现人物与场景的联合重建，还需要两个缺失的输出：度量尺度和持久的人物身份。我们探究的问题是：一个尺度未定的基础模型表示，能否通过轻量级适配同时支持这两个输出。精确的度量标注非常稀缺，而无标注的野外视频却十分丰富。我们利用精选网络视频中的人物来初始化解决方案：带有姿态的度量人体模型和2D关键点能够给出近似的、闭式解的尺度伪标签。这些伪标签用于预训练一个尺度读取头，随后该读取头与一个轻量级适配器一起，使用来自标准真实视频训练集的精确度量监督进行微调。在推理阶段，该读取头直接从基础模型的token中预测度量尺度，无需标尺及其教师模型。对于人物身份，我们单独探究了预训练的基础模型本身，并发现证据表明……

    arXiv:2609.29106v1 Announce Type: cross  Abstract: 3D foundation models recover video cameras and geometry in one forward pass, but some of the strongest are up to scale. Joint people-scene reconstruction then requires two missing outputs: metric scale and persistent person identity. We ask whether one up-to-scale foundation representation can support both through lightweight adaptation. Exact metric labels are scarce, but unlabeled in-the-wild video is abundant. We use people in curated web video to initialise the solution: a posed metric body and 2D keypoints give an approximate, closed-form scale pseudo-label. These pseudo-labels pretrain a Scale Readout, which is then fine-tuned together with a lightweight adapter using exact metric supervision from standard real-video training splits. At inference the head predicts metric scale from foundation-model tokens, without the ruler or its teachers. For person identity, we probe the pretrained foundation model alone and find evidence that
    
[^187]: 恰好一次语义由谁承担？模型、智能体框架与工具契约对 LLM 智能体重复副作用的影响

    Where Does Exactly-Once Live? Model, Harness, and Tool-Contract Effects on Duplicate Side Effects in LLM Agents

    [https://arxiv.org/abs/2609.29095](https://arxiv.org/abs/2609.29095)

    该论文提出确定性沙盒基准 LIMBO，研究 LLM 智能体的“恰好一次”副作用语义应由模型、智能体框架还是工具契约来保障，并发现答案取决于故障类型：当即时回读能够揭示实际结果时，由模型来决定。

    

    当使用工具的智能体的写操作超时或返回服务器错误时，该操作可能已经实际生效。盲目重试会导致重复执行——第二次扣款、第二次公告、第二次部署——而放弃重试则会跳过必要的工作。我们提出问题：恰好一次（exactly-once）行为应该在哪里强制执行：在模型中、在智能体框架（harness）中，还是在工具契约中？我们介绍了 LIMBO，一个由六个服务组成的确定性沙盒，这些服务具有真实的契约（可选的幂等键、最终一致性和缺失的读取路径），并在服务边界注入了十二种故障模式，包括延迟提交、重复投递和部分批次；每个回合都根据已提交效果的账本进行评分。在涵盖九个近期模型、三个生产级智能体框架、两种契约变体和十五种恢复条件共 25,930 个回合的实验中，答案取决于具体的故障类型。当即时回读能够揭示发生了什么时，由模型来决定……

    arXiv:2609.29095v1 Announce Type: cross  Abstract: When a tool-using agent's write times out or returns a server error, the action may already have taken effect. Retrying blindly duplicates it -- a second charge, a second announcement, a second deployment -- while giving up skips required work. We ask where exactly-once behaviour should be enforced: in the model, in the agent harness, or in the tool contract. We introduce LIMBO, a deterministic sandbox of six services with realistic contracts (optional idempotency keys, eventually consistent and missing read paths) and twelve fault modes injected at the service boundary, including late commits, redelivery and partial batches; every episode is graded against a ledger of committed effects. Across 25,930 episodes spanning nine recent models, three production agent harnesses, two contract variants and fifteen recovery conditions, the answer depends on the fault. When an immediate read-back can reveal what happened, the model decides: front
    
[^188]: DAWN：通过深度去噪世界模型实现噪声鲁棒的四足跑酷

    DAWN: Noise-Robust Quadruped Parkour via Depth-Denoising World Models

    [https://arxiv.org/abs/2609.29092](https://arxiv.org/abs/2609.29092)

    该论文提出DAWN框架，通过让世界模型以带噪深度为输入、干净深度为重建目标实现隐式去噪，并结合对比学习对齐，将噪声鲁棒性直接内置到腿式运动的深度感知中，从而无需手工调整的滤波器即可实现鲁棒的四足跑酷。

    

    基于视觉的腿式运动方法通常假设训练时的深度数据是干净的，并在部署时依赖手工调整的后处理滤波器。然而，滤波器参数很少被公开，阻碍了可复现性，且当深度噪声未被处理时性能会大幅下降。将噪声鲁棒性直接构建到学习流程中可以消除这种依赖。虽然这种鲁棒性已在本体感知输入中被探索过，但在腿式运动领域，针对深度感知的类似方法仍然很大程度上缺失。我们提出了DAWN（世界模型中的去噪与对齐以实现噪声鲁棒性），这是一个面向腿式运动的噪声鲁棒感知框架，它通过两项修改将噪声鲁棒性直接内置到世界模型中：（1）将带噪声的深度图输入编码器，同时以干净的深度图作为重建目标，迫使模型对其输入进行隐式去噪；（2）应用对比学习来对齐（原文此处截断）。

    arXiv:2609.29092v1 Announce Type: cross  Abstract: Vision-based legged locomotion methods assume clean depth at training time and rely on hand-tuned post-processing filters at deployment. However, filter parameters are rarely disclosed, hindering reproducibility, and performance degrades substantially when depth noise is left unaddressed. Building noise robustness directly into the learning pipeline would eliminate this dependency. While such robustness has been explored for proprioceptive inputs, analogous approaches for depth perception remain largely absent in legged locomotion. We propose DAWN (Denoising and Alignment in World models for Noise-robustness), a noise-robust perception framework for legged locomotion, which builds noise robustness directly into a world model via two modifications: (1) feeding noisy depth to the encoder while keeping clean depth as the reconstruction target, forcing the model to implicitly denoise its input; and (2) applying contrastive learning to alig
    
[^189]: 经典语义抽取式摘要可以在印地语中评估吗？一项复现研究

    Can Classical Semantic-Extractive Summarization Be Evaluated in Hindi? A Replication Study

    [https://arxiv.org/abs/2609.29090](https://arxiv.org/abs/2609.29090)

    该研究将经典分布语义抽取式摘要方法复现并适配到印地语，发现其在两个语料库上均显著落后于简单的三句引导基线，且句子位置是唯一真正起作用的特征。

    

    我们复现了Mohd、Jan和Shah（2020）提出的分布语义抽取式摘要方法，并将其适配到印地语，在每个涉及语言特性的步骤中都替换为适合天城文（Devanagari）的组件。该系统在两个独立语料库上进行评估——XL-Sum的印地语部分和FIRE ILSUM 2.0印地语——采用经过XL-Sum作者自有多语言评分器验证的天城文感知ROUGE实现，所有比较均通过1000次重采样的配对自举法得出。在其发表时的等权重配置下，复现系统在两个语料库上的表现均显著差于三句引导（Lead-3）基线，在XL-Sum上ROUGE-1 F落后0.042，在ILSUM上落后0.265。特征消融实验表明，句子位置是唯一有效的特征：仅使用位置即可精确复现引导基线，移除位置则得到最弱的配置，而经过验证集调优的加权最多也只能与Lead-3持平。

    arXiv:2609.29090v1 Announce Type: cross  Abstract: We replicate the distributional-semantics extractive summarisation method of Mohd, Jan and Shah (2020) and adapt it to Hindi, substituting a Devanagari-appropriate component at every language-specific step. The system is evaluated on two independent corpora --- the Hindi portion of XL-Sum and FIRE ILSUM 2.0 Hindi --- under a Devanagari-aware ROUGE implementation validated against the XL-Sum authors' own multilingual scorer, with all comparisons drawn as 1000-resample paired bootstraps. In its published equal-weight configuration the replicated system is significantly worse than a three-sentence lead baseline on both corpora, trailing Lead-3 by 0.042 ROUGE-1 Fon XL-Sum and by 0.265 on ILSUM. A feature ablation shows that sentenceposition is the only feature that contributes: position alone reproduces the lead baseline exactly, removing position gives the weakest configuration,and a validation-tuned weighting can at best equal Lead-3 and
    
[^190]: 一种在WeBe手环上快速训练与部署机器学习模型的流水线

    A Rapid Pipeline for Training and Deploying ML Models on WeBe Band

    [https://arxiv.org/abs/2609.29084](https://arxiv.org/abs/2609.29084)

    本文提出一个集成开源Piccolo AI生态系统的自动化快速流水线，能够在资源受限的WeBe手环上快速开发、优化并部署满足延迟、内存和功耗约束的机器学习模型。

    

    针对计算和内存资源受限的边缘设备开发优化的机器学习算法是具有挑战性、耗时且高度依赖设备特定约束的工作。在这项工作中，我们简化了边缘机器学习工作流程，实现了机器学习（ML）模型直接在WeBe Band上的快速开发、优化和部署，WeBe Band是一款专为多模态生理数据监测设计的腕戴式可穿戴设备。所提出的系统自动生成硬件高效的机器学习模型，这些模型可以轻松集成到WeBe核心固件中，支持AutoML、硬件感知量化和性能分析，以构建满足预期延迟目标、同时兼容设备内存和功耗限制的模型。该框架将开源的Piccolo AI生态系统与自动化流水线紧密集成，可生成可部署的固件制品，并执行硬件感知

    arXiv:2609.29084v1 Announce Type: new  Abstract: Developing optimized machine-learning algorithms for edge devices with limited computational and memory resources is challenging, time-consuming, and highly dependent on device-specific constraints. In this work, we streamline an edge ML workflow to enable rapid development, optimization, and deployment of machine-learning (ML) models directly on the WeBe Band, a wrist-worn wearable device designed for multimodal physiological data monitoring. The proposed system automatically generates hardware-efficient ML models that can be easily integrated into the WeBe core firmware, supporting AutoML, hardware-aware quantization, and performance profiling to build models that meet desired latency targets while remaining compatible with device memory and power limitations.   The proposed framework tightly integrates the open-source Piccolo AI ecosystem with an automated pipeline that generates deployable firmware artifacts, performs hardware-aware 
    
[^191]: CRISS：一个用于辅助癌症登记员的检索增强型AI聊天机器人

    CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars

    [https://arxiv.org/abs/2609.29075](https://arxiv.org/abs/2609.29075)

    CRISS是一个基于检索增强生成（RAG）技术的AI聊天助手，通过构建癌症登记标准领域知识库，为癌症登记员提供快速、有引文支持的指南查询服务，同时保留人工对最终决策的监督。

    

    癌症登记员，包括肿瘤数据专家（ODSs），必须解读复杂且频繁更新的编码和分期标准。我们开发了CRISS（癌症登记智能支持系统），这是一个检索增强生成（RAG）对话助手，能够提供快速、有引文支持的登记指南访问。本研究评估了CRISS能否（1）支持准确且有引文依据的回答，（2）改善对相关指南的访问和解读，（3）支持培训和帮助台使用场景，同时保留人工对最终摘录决策的监督。我们从国家癌症登记标准构建了一个特定领域的知识库，将其分割为带有元数据标记的段落，并索引为稠密嵌入。检索到的段落被用于通过大语言模型（LLM）生成有引文依据的回答。开放权重模型、专有模型以及非RAG基线模型跨越Gemini和GPT系列进行了对比

    arXiv:2609.29075v1 Announce Type: new  Abstract: Cancer registrars, including Oncology Data Specialists (ODSs), must interpret complex and frequently updated coding and staging standards. We developed CRISS (Cancer Registry Intelligent Support System), a retrieval-augmented generation (RAG) conversational assistant that provides rapid, citation-supported access to registry guidance. This study evaluated whether CRISS could (1) support accurate and citation-supported responses, (2) improve access to and interpretation of relevant guidance, and (3) support training/helpdesk use while preserving human oversight of final abstraction decisions. We built a domain-specific knowledge base from national cancer registry standards, segmented into metadata-tagged passages and indexed as dense embeddings. Retrieved passages were used to generate citation-grounded responses through a large language model (LLM). Open-weight, proprietary, and non-RAG baseline models across Gemini and GPT families were
    
[^192]: EIB-Net：面向可泛化AI生成图像检测的熵引导信息瓶颈网络

    EIB-Net: Entropy-Guided Information Bottleneck for Generalizable AI-Generated Image Detection

    [https://arxiv.org/abs/2609.29064](https://arxiv.org/abs/2609.29064)

    EIB-Net提出新颖的图像熵度量与变分信息瓶颈相结合的方法，通过自动选择信息量最大的低熵图像块来学习紧凑可泛化的特征，在跨生成模型的AI生成图像检测中实现了最先进的性能。

    

    逼真AI生成图像的大量涌现，要求检测方法能够在多种生成模型之间实现鲁棒泛化。现有方法主要针对基于篡改的伪造手段及其局部伪影，而基于生成方式产生的图像（例如来自扩散模型的图像）缺乏此类痕迹，这构成了一个根本性挑战。我们观察到，生成模型在生成过程中优先保证全局语义，而牺牲了局部纹理的保真度，这使得低纹理区域成为判断图像合成来源的关键指标。为了利用这一特性，我们提出了EIB-Net——一种熵引导信息瓶颈网络。EIB-Net引入了一种新颖的图像熵度量来自动选择信息量最大（即熵最低）的图像块，然后通过变分信息瓶颈对其进行处理，以学习紧凑且可泛化的特征。在DIFF、DiffusionForensics和GenImage基准数据集上的大量实验表明，该方法达到了最先进的性能：EIB-Net取得了85.7（原文此处截断，应为相应准确率数值）的成绩。

    arXiv:2609.29064v1 Announce Type: cross  Abstract: The proliferation of photorealistic AI-generated images demands robust detection methods that generalize across diverse generative models. While existing approaches target manipulation-based forgeries with local artifacts, generation-based images (e.g., from diffusion models) lack such traces, posing a fundamental challenge. We observe that generative models prioritize global semantics at the expense of local texture fidelity, making low-texture regions key indicators of synthetic origin. To exploit this, we propose EIB-Net, an Entropy-guided Information Bottleneck Network. EIB-Net introduces a novel Image Entropy (IE) metric to automatically select the most informative (lowest-entropy) patch, then processes it with a Variational Information Bottleneck (VIB) to learn compact, generalizable features. Extensive experiments on DIFF, DiffusionForensics, and GenImage benchmarks demonstrate state-of-the-art performance: EIB-Net achieves 85.7
    
[^193]: EMPATH：追踪危机心理咨询对话中的多层次情感动态

    Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues

    [https://arxiv.org/abs/2609.29056](https://arxiv.org/abs/2609.29056)

    提出EMPATH框架，从轮次级标签、转移概率和对话原型三个粒度追踪危机心理咨询对话中的情感动态，揭示了悲伤对话中负面情感持续存在、希望渐进增强以及求助者与志愿者情感角色截然不同的模式。

    

    情感动态对于理解危机支持类对话至关重要，然而大多数计算研究将情感视为静态的话语级标签。我们提出了EMPATH，这是一个用于理解心理健康对话中情感动态的框架，涵盖三个粒度：轮次级标签、转移概率以及全局对话原型。将EMPATH应用于自认为黑人、讨论悲伤情绪的文本危机对话后，我们发现负面情感持续存在、情感向希望方向的渐进转变、求助者与志愿者之间截然不同的情感角色，以及异质性的恢复轨迹。这些结果突显了将危机支持与悲伤表达视为对话中动态过程进行计算理解所能揭示的有价值模式，以及情感动态分析在分析和比较对话情感方面的整体价值。

    arXiv:2609.29056v1 Announce Type: cross  Abstract: Emotion dynamics are critical for understanding crisis-support conversations, yet most computational work treats emotion as static utterance-level labels. We introduce EMPATH, a framework for understanding affective dynamics in mental health dialogues across three granularities: turn-level labels, transition probabilities, and global conversation archetypes. Applying EMPATH to text-based crisis conversations with self-identified Black texters discussing grief, we find persistent negative affect, gradual hope-ward transitions, distinct texter-volunteer emotional roles, and heterogeneous recovery trajectories. These results highlight the informative patterns that emerge from computationally understanding crisis support and expressions of grief as dynamic processes within conversations, as well as the overall value of emotion-dynamic analysis for analyzing and comparing affect in dialogues.
    
[^194]: 从自我蒸馏到自我练习：面向多轮智能体的特权信息

    From Self-Distillation to Self-Practice: Privileged Information for Multi-Turn Agents

    [https://arxiv.org/abs/2609.29051](https://arxiv.org/abs/2609.29051)

    本文发现基于特权信息的同策略自我蒸馏会让多轮智能体“盲目自信”且性能甚至差于普通强化学习和基础模型，并提出特权自我练习（PSP）方法，将特权信息从损失函数转移到采样阶段，通过注入分析模型生成的任务指令引导重新采样，再用不变的GRPO目标训练，从而有效提升多轮智能体性能。

    

    同策略自我蒸馏（OPSD）已成为大语言模型智能体后训练的一种流行方法。它通过让同一模型以特权信息（PI）为条件获得更强的教师视角，从而在词元（token）级别对智能体模型进行监督。在本工作中，我们证明在多轮智能体中，这种范式教会学生模型自信地行动，却不掌握其背后的信息。训练后的智能体表现得仿佛拥有它从未见过的特权信息，其性能远逊于普通的强化学习，最坏情况下甚至低于未经训练的基础模型。因此，我们提出特权自我练习（PSP），它保留特权信息，并将其从损失函数转移到采样器中。当学生在某个任务上的采样结果大多失败时，我们注入一段由分析模型编写的简短任务特定指令，将该指令置于上下文中重新采样该任务，并在保持GRPO目标不变的情况下对结果进行训练。特权信息……

    arXiv:2609.29051v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) has become a popular recipe for post-training LLM agents. It supervises the agent model at the token level with a stronger teacher view of the same model, obtained by conditioning on privileged information (PI). In this work, we show that in multi-turn agents, this paradigm teaches the student to act with confidence but without the information behind it. The trained agent behaves as if it had privileged information it never observed, and its performance falls well short of plain RL, in the worst case below the untrained base model. Therefore, we propose Privileged Self-Practice (PSP), which keeps the PI and moves it from the loss to the sampler. When the student's rollouts on a task mostly fail, we inject a short per-task instruction written by an analyzer model, sample the task again with the instruction in context, and train on the result with an unchanged GRPO objective. The privileged information st
    
[^195]: SLCA-GRPO：解决工具调用强化学习中的跨段信用错误归因问题

    SLCA-GRPO: Resolving Cross-Segment Credit Misattribution in Tool-Calling RL

    [https://arxiv.org/abs/2609.29050](https://arxiv.org/abs/2609.29050)

    提出SLCA-GRPO框架，通过段锁定信用分配在结构段级别解耦优势估计，解决工具调用强化学习中跨段信用错误归因问题，并配套构建模式引导的LLM模拟器（SGLS）以实现无需真实API的可扩展稳定训练。

    

    工具调用智能体产生的输出具有异构性，将结构化的工具调用与面向用户的自然语言总结交织在一起。这种输出异构性在标准的同策略强化学习（RL）中呈现出一种结构性失效模式：诸如GRPO之类的算法不加区分地向所有token广播同质的轨迹级标量优势值。因此，来自总结文本生成的梯度噪声会泄漏到工具决策token中，导致跨段信用错误归因和脆弱的优化过程。在本工作中，我们提出了SLCA-GRPO，一个融合了段锁定信用分配的框架。为了在不依赖昂贵的真实API的情况下实现可扩展的探索并保证训练的稳定性，我们首先构建了模式引导的LLM模拟器作为基础训练基础设施。在此基础上，SLCA在单组rollout内部于结构段级别对优势估计进行解耦，而无需额外的rollout采样。

    arXiv:2609.29050v1 Announce Type: new  Abstract: Tool-calling agents produce heterogeneous outputs, interleaving structured tool invocations with user-facing natural language summaries. This output heterogeneity presents a structural failure mode in standard on-policy Reinforcement Learning (RL): algorithms like GRPO indiscriminately broadcast a homogeneous trajectory-level scalar advantage to all tokens. Consequently, gradient noise from summary generation leaks into tool-decision tokens, causing cross-segment credit misattribution and brittle optimization. In this work, we propose SLCA-GRPO, a framework incorporating Segment-Locked Credit Assignment (SLCA). To enable scalable exploration without costly real APIs and stable training, we first construct the Schema-Guided LLM Simulator (SGLS) as foundational training infrastructure. Building on this, SLCA decouples advantage estimation at the structural segment level within a single group of rollouts, without requiring additional rollou
    
[^196]: 分词的记忆：当分词化绕过知识编辑与机器遗忘

    The Tokens Remember: When Tokenization Bypasses Knowledge Editing and Unlearning

    [https://arxiv.org/abs/2609.29045](https://arxiv.org/abs/2609.29045)

    论文揭示了一个新型安全漏洞：攻击者可通过使用替代的有效分词方式诱导不同的计算轨迹，从而绕过大语言模型的知识编辑与机器遗忘机制，使本应被删除或修改的敏感知识重新暴露。

    

    开源权重的大语言模型让下游用户能够掌控推理栈，但这种灵活性可能会破坏模型发布后关于敏感知识已被修改或删除的保证。模型编辑与机器遗忘技术被用于在不从头重新训练模型的情况下修改或删除特定知识。然而，这些技术的现有安全评估存在两个关键局限。首先，它们通常需要访问原始的编辑前/遗忘前模型或辅助分类器来检测修改或重建编辑前的行为。其次，它们在输入的规范分词（canonical tokenization）方式下评估修改效果，隐含地将分词视为无害的预处理步骤。我们证明这一假设造成了一个安全缺口：同一输入字符串可以通过其他有效的分词方式来表示，这些分词会诱导出不同的计算轨迹，从而使攻击者能够绕过局部化的知识修改。

    arXiv:2609.29045v1 Announce Type: cross  Abstract: Open-weight LLMs give downstream users control over the inference stack, but this flexibility can undermine post-release guarantees that sensitive knowledge has been modified or removed. Model editing and machine unlearning are used to modify or remove targeted knowledge without retraining models from scratch. However, existing security evaluations of these techniques face two critical limitations. First, they typically require access to either the original pre-edit/unlearning model or auxiliary classifiers to detect modifications or reconstruct pre-edit behavior. Second, they evaluate modifications under the canonical tokenization of an input, implicitly treating tokenization as a benign preprocessing step. We show that this assumption creates a security gap: the same input string can be represented by alternative valid tokenizations that induce different computational trajectories, allowing an adversary to bypass localized modificati
    
[^197]: 3GPP信道估计器的多智能体编排

    Multi-Agent Orchestration of 3GPP Channel Estimators

    [https://arxiv.org/abs/2609.29044](https://arxiv.org/abs/2609.29044)

    该论文在3GPP多种信道模型、参数配置及SISO/MIMO设置下对八种信道估计器进行统一评估，量化证明了没有单一估计器能在所有条件下都保持最优，从而引出多智能体编排方法的需求。

    

    导频辅助信道估计是5G新空口（5G-NR）和长期演进（LTE）正交频分复用（OFDM）接收机中的关键模块。现有估计器种类繁多，从简单的最小二乘（LS）插值，到统计最优的线性最小均方误差（LMMSE）变体，再到近年的深度卷积去噪器，然而没有任何单一估计器能够始终最优：最优选择取决于传播场景、参数配置、工作信噪比（SNR）、移动性（多普勒）以及天线配置。本文通过对文献中八种估计器的统一研究来量化这一事实，这些估计器在由NVIDIA Sionna生成的3GPP TR 38.901城市宏蜂窝（UMa）、城市微蜂窝（UMi）和乡村宏蜂窝（RMa）信道上进行评估，涵盖5G-NR和LTE两种参数配置，并在单输入单输出（SISO）和8×2多输入多输出（MIMO）设置下进行测试。

    arXiv:2609.29044v1 Announce Type: cross  Abstract: Pilot-aided channel estimation is a decisive block in orthogonal frequency-division multiplexing (OFDM) receivers for both 5G New Radio (5G-NR) and Long-Term Evolution (LTE). A large body of estimators exists, from simple least-squares (LS) interpolation to statistically optimal linear minimum-mean-square-error (LMMSE) variants and, more recently, deep convolutional denoisers, yet no single estimator is uniformly best: the winner depends on the propagation scenario, the numerology, the operating signal-to-noise ratio (SNR), the mobility (Doppler), and the antenna configuration. In this paper, we quantify this fact through a unified study of eight literature estimators evaluated over the 3GPP TR~38.901 Urban-Macro (UMa), Urban-Micro (UMi), and Rural-Macro (RMa) channels generated with NVIDIA Sionna, for both 5G-NR and LTE numerologies, in single-input single-output (SISO) and $8\times2$ multiple-input multiple-output (MIMO) settings. We
    
[^198]: 面向通用服务机器人的基于大语言模型链式架构的任务规划设计与评估

    Design and Evaluation of LLM Chaining-Based Task Planning for General Purpose Service Robots

    [https://arxiv.org/abs/2609.29043](https://arxiv.org/abs/2609.29043)

    提出一种将指令分类与动作生成分离的两阶段大语言模型链式架构，用于通用服务机器人的GPSR任务规划，在将提示词长度减少约45%的同时，相比单提示词方法最高提升37个百分点的规划成功率，并通过真实机器人实验进行了验证。

    

    RoboCup@Home基准中定义的通用服务机器人（GPSR）任务，要求机器人在真实家庭环境中理解多样化的自然语言指令，并生成多步骤动作序列。传统的单提示词（SP）方法存在上下文臃肿和“迷失在中间”现象的问题，导致任务规划不可靠。我们提出了一种大语言模型链式（LLM chaining）架构，将指令分类与动作生成分离为两个专门化阶段，使每次推理的提示词长度减少约45%，同时提升了规划一致性。我们使用100条随机生成的GPSR指令，在涵盖本地开源模型和前沿云端部署场景的三种语言模型上对该方法进行了评估。结果表明，该方法在所有模型上都比单提示词方法取得了持续的规划性能提升，在本地模型上的增益最高达+37个百分点。此外，还在丰田Human S...（真实机器人上的执行实验，摘要在此处被截断）

    arXiv:2609.29043v1 Announce Type: cross  Abstract: General Purpose Service Robot (GPSR) tasks, as defined in the RoboCup@Home benchmark, require robots to interpret diverse natural language commands and generate multi-step action sequences in real home environments. Conventional Single Prompt (SP) approaches suffer from context bloat and the "Lost in the Middle" phenomenon, leading to unreliable task planning. We propose an LLM chaining architecture that separates instruction classification and action generation into two specialized stages, reducing per-inference prompt length by approximately 45% while improving planning consistency. We evaluate our method using 100 randomly generated GPSR commands across three language models spanning local open-source and frontier cloud deployment contexts. Results show consistent planning improvements over SP across all models, with gains of up to +37 percentage points on local models. Further, real-robot execution experiments on the Toyota Human S
    
[^199]: MeshHeal：去中心化LLM智能体网络中灰色故障的双时间尺度自愈机制

    MeshHeal: Two-Timescale Self-Healing for Gray Failures in Decentralized LLM Agent Networks

    [https://arxiv.org/abs/2609.29015](https://arxiv.org/abs/2609.29015)

    MeshHeal提出了一个完全去中心化的双时间尺度自愈框架，通过快速时间尺度上的自适应评审升级机制与慢速时间尺度上的退化检测和恢复探测，解决去中心化LLM智能体网络中难以察觉的灰色故障问题。

    

    去中心化的基于LLM的多智能体系统通过本地交互进行协调，但一个智能体可能在保持响应的同时，其任务解决质量却持续下降。这类灰色故障要求在尚无足够证据改变未来路由之前保护当前任务，同时仍允许已恢复的智能体重新加入。我们提出了MeshHeal，这是一个完全去中心化的自愈框架，它在两个时间尺度上耦合了能力匹配的同行评审。在快速时间尺度上，自适应层次结构将来自重复单评审者评估的不确定或低分输出升级为委员会审议，并在需要时在使用前进行纠正。在慢速时间尺度上，一种基于任务和能力条件的同行相对检测器聚合评分，以区分持续退化与普通输出波动，触发强制性委员会审查，并最终将退化智能体从普通路由中排除；恢复探测提供……

    arXiv:2609.29015v1 Announce Type: new  Abstract: Decentralized LLM-based multi-agent systems coordinate through local interactions, but an agent can remain responsive while its task-solving quality persistently degrades. Such gray failures require protecting current tasks before sufficient evidence exists to alter future routing, while still allowing recovered agents to rejoin. We introduce MeshHeal, a fully decentralized self-healing framework that couples ability-matched peer review across two timescales. At the fast timescale, an adaptive hierarchy escalates uncertain or low-scoring outputs from repeated single-reviewer evaluation to committee deliberation and, when needed, correction before use. At the slow timescale, a task- and ability-conditioned peer-relative detector aggregates scores to distinguish persistent degradation from ordinary output variation, trigger mandatory committee review, and eventually exclude degraded agents from ordinary routing; recovery probes provide fre
    
[^200]: AlphaDiverse：面向Alpha因子挖掘中多样性探索的本地量化研究智能体后训练

    AlphaDiverse: Post-Training Local Quantitative Research Agents for Diverse Exploration in Alpha Factor Mining

    [https://arxiv.org/abs/2609.29014](https://arxiv.org/abs/2609.29014)

    提出了AlphaDiverse框架，通过多智能体系统生成互补计划组合并变换研究环境来收集多样化研究路径，再结合监督微调与联合GRPO后训练本地Planner和Realizer智能体，从而摆脱对外部API的依赖，实现成本可控、机密安全且路径多样的Alpha因子自动挖掘。

    

    基于大语言模型（LLM）的多智能体系统可以实现Alpha因子挖掘的自动化，但其对外部API的依赖限制了对成本、可用性和机密性的控制。同时，漫长的研究循环往往倾向于重复探索少数成功的经济机制，从而导致研究路径坍缩。为了解决这些局限性，我们提出了AlphaDiverse，这是一个集成了多智能体Alpha研究系统、多样化研究路径收集以及本地智能体后训练的框架。我们让研究系统生成互补的计划组合，并在不同循环中变换研究环境，以收集多样化的研究路径。利用这些多样化的轨迹，我们通过监督微调对本地Planner和Realizer智能体进行热启动。随后，我们提出了一种联合GRPO方法，基于预测质量和贡献多样性对这两个智能体进行联合优化。研究反馈仅限于内部周期数据，而冻结的最终模型则在外部（后续截断）……上进行评估。

    arXiv:2609.29014v1 Announce Type: new  Abstract: Large language model (LLM)-based multi-agent systems can automate alpha factor mining, but their reliance on external APIs limits control over cost, availability, and confidentiality. Long research loops also tend to revisit a few successful economic mechanisms that lead to research path collapse. To address these limitations, we propose AlphaDiverse, a framework that integrates a multi-agent alpha research system, diverse research path collection, and post-training for local agents. We let the research system generate complementary plan portfolios and vary research environments across loops to collect diverse research paths. Using these diverse traces, we warm-start local Planner and Realizer agents with supervised fine-tuning. Then, we propose a joint GRPO method to optimize both of them using predictive quality and diversity of contributions. Research feedback is confined to inner period data, while a frozen final model is evaluated o
    
[^201]: 动作信用何时需要更新？

    When Does Action Credit Need Updating?

    [https://arxiv.org/abs/2609.29007](https://arxiv.org/abs/2609.29007)

    该论文提出成对分支敏感度来判断策略更新引起的漂移是否会推翻原有动作排序，从而仅在必要时利用一阶锚定信用迁移估计器基于旧干预轨迹更新历史动作信用，大幅降低工具使用智能体反复更新的成本。

    

    使用工具的智能体会随着新的交互数据不断更新。然而，每次策略更新后，先前估计的动作信用可能会变得过时。从头重新计算这些信用可能需要大量额外的工具调用和环境交互，使得反复更新的成本越来越高。我们提出了一个简单的问题：历史动作信用究竟何时才真正需要更新？我们的关键观察是：动作价值的改变并不一定意味着决策的改变。只要策略引起的漂移不足以推翻现有的动作排序，历史信用就仍然有用。基于这一思想，我们引入了成对分支敏感度，用以衡量策略更新对区分两个候选动作的下游区域的影响强度。随后，我们推导出一阶锚定信用迁移估计器，利用旧的干预轨迹来更新历史信用，并提出了一种 De……（原文摘要在此处被截断）

    arXiv:2609.29007v1 Announce Type: new  Abstract: Tool-using agents are continually updated with new interaction data. After each policy update, however, previously estimated action credits may become stale. Recomputing them from scratch can require many additional tool calls and environment interactions, making repeated updates increasingly expensive. We ask a simple question: when does historical action credit actually need to be updated? Our key observation is that a change in action value does not necessarily imply a change in the decision. Historical credit can still be useful as long as policy-induced drift is too small to overturn the existing action ranking. Building on this idea, we introduce pairwise branch sensitivity to capture how strongly a policy update affects the downstream regions that distinguish two candidate actions. We then derive a first-order anchored credit-transport estimator that updates historical credit using old interventional trajectories, and propose a De
    
[^202]: 《分数之下：重新思考视频理解模型的幻觉评估》

    Beneath the Scores: Rethinking Hallucination Evaluation for Video Understanding Models

    [https://arxiv.org/abs/2609.28991](https://arxiv.org/abs/2609.28991)

    该论文提出了一种因果阶段干预评估协议，通过对三种视频智能体架构进行60,008次实验，揭示了现有基准分数无法可靠预测下游幻觉，并发现时间定位阶段是视频理解中下游错误的主导来源，其因果影响约为视觉观察破坏的四倍。

    

    视频理解越来越多地由多阶段LLM智能体执行，这些智能体将时间定位、视觉观察和推理分离开来。然而，这些阶段通常在不同的基准和数据分布上进行评估，这使得难以确定幻觉的来源。我们首先围绕这些阶段对现有基准进行了组织梳理，并表明它们的分数提供了不一致的诊断信号：更强的阶段级性能并不可靠地意味着更低的下游幻觉，即使是针对相同能力的基准之间也可能存在分歧。因此，我们引入了一种因果阶段干预协议，在保持下游任务不变的情况下覆盖各个独立阶段。在三种视频智能体架构上进行的60,008次运行实验中，我们发现时间定位是下游错误的主要来源，其因果影响约为破坏视觉观察的四倍。成功的定位主要依赖于……

    arXiv:2609.28991v1 Announce Type: cross  Abstract: Video understanding is increasingly performed by multi-stage LLM agents that separate temporal grounding, visual observation, and reasoning. Yet these stages are typically evaluated on different benchmarks and distributions, making it difficult to determine where hallucinations originate. We first organize existing benchmarks around these stages and show that their scores provide inconsistent diagnostic signals: stronger stage-level performance does not reliably imply lower downstream hallucination, and even benchmarks targeting the same capability can disagree.   We therefore introduce a causal stage-intervention protocol that overwrites individual stages while holding the downstream task fixed. Across 60,008 runs on three video-agent architectures, we find that grounding is the dominant source of downstream error, with roughly four times the causal impact of corrupting visual observations. Successful grounding depends primarily on lo
    
[^203]: CrossSafe：迈向跨形态的潜在安全过滤器

    CrossSafe: Towards Cross-Embodiment Latent Safety Filters

    [https://arxiv.org/abs/2609.28984](https://arxiv.org/abs/2609.28984)

    本文提出CrossSafe，利用安全推理在不同机器人间的共通性构建跨形态的潜在安全过滤器，并根据各机器人的形态、运动学和动力学差异来具体实现安全动作，从而为通用操作策略提供跨机器人的安全保障。

    

    跨形态学习已经表明，单一模型（例如视觉-语言-动作模型，即VLA模型）能够学习可应用于异构机器人以完成各种任务的状态表示和操作技能。我们假设安全执行同样具有这一性质。满足安全约束所需的推理过程——例如检测障碍物、识别出需要避开它、以及选择一个安全的抽象动作——在很大程度上是不同机器人之间共享的。不同形态之间的差异在于抽象的安全动作如何被具体实现：形态结构、运动学和动力学决定了哪些动作是安全且可行的。因此，同一个动作对一个机器人可能是安全的，而对另一个机器人则是不安全的。这对于在统一的末端执行器动作空间中运行、且未明确刻画安全性如何依赖于机器人形态和运动学的通用操作策略而言尤为重要。我们提出

    arXiv:2609.28984v1 Announce Type: cross  Abstract: Cross-embodiment learning has shown that a single model, such as a vision-language-action (VLA) model, can learn state representations and manipulation skills that can be applied across heterogeneous robots to accomplish various tasks. We hypothesize that the same holds for safety enforcement. The reasoning required to satisfy a safety constraint, such as detecting an obstacle, recognizing that it should be avoided, and selecting a safe abstract action, is largely shared across robots. What differs across embodiments is how the abstract safe action is realized: morphology, kinematics, and dynamics determine which actions are safe and feasible. Consequently, the same action can be safe for one robot and unsafe for another. This is especially important for generalist manipulation policies that operate in a common end-effector action space without explicitly capturing how safety depends on the robot's morphology and kinematics. We propose
    
[^204]: 面向生成式推荐的跨国代码混合方法

    Cross-Country Code-Mixing for Generative Recommendation

    [https://arxiv.org/abs/2609.28972](https://arxiv.org/abs/2609.28972)

    提出CMRec框架，借鉴多语言NLP中的语码转换思想，通过学习跨国共享语义码本并进行上下文感知的代码混合，在数据层面（而非仅参数层面）实现跨国生成式推荐的知识迁移。

    

    现代电商平台上的跨国推荐系统通常在各市场间部署相互隔离的用户与物品ID空间，这使得传统跨领域方法所依赖的共享锚点不复存在。生成式推荐（GR）通过将物品映射到共享的token空间并训练统一模型来缓解这一问题，但现有方法的行为序列仍严格限定于单一国家内部，因此知识迁移仅发生在参数层面，在数据层面依然缺失。受多语言自然语言处理中语码转换语料库的启发，我们提出CMRec——一个跨国生成式推荐框架，通过双重约束、上下文感知的代码混合，在数据层面注入跨国监督信号。CMRec首先从跨国家的多模态内容与行为共现中学习一个共享语义码本，然后利用该码本通过token级别的替换来合成混合国家的序列（摘要在此处截断）。

    arXiv:2609.28972v1 Announce Type: cross  Abstract: Cross-country recommendation on modern e-commerce platforms is typically deployed with disjoint user and item ID spaces across markets, removing the shared anchors that conventional cross-domain methods rely on. Generative recommendation (GR) mitigates this by mapping items into a shared token space and training a unified model, but existing approaches keep behavior sequences strictly country-specific, so knowledge transfer occurs only at the parameter level and remains absent at the data level. Inspired by code-switching corpora in multilingual natural language processing, we propose CMRec, a cross-country GR framework that injects cross-country supervision at the data level via dual-constrained, context-aware code-mixing. CMRec first learns a shared semantic codebook from multi-modal content and behavioral co-occurrence across countries. It then uses this codebook to synthesize mixed-country sequences via token-level substitutions th
    
[^205]: 回归定义：通过轨迹图估计智能体强化学习中的步骤级优势

    Back to the Definition: Estimating Step-Level Advantages via Trajectory Graphs for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.28963](https://arxiv.org/abs/2609.28963)

    提出通过轨迹图来估计步骤级优势，解决了GRPO等分组强化学习方法在步骤层面因轨迹级粗粒度估计而产生的系统性偏差问题。

    

    基于分组的强化学习方法，如GRPO及其变体，已成为训练推理型和智能体型大语言模型（LLM）的主流范式。虽然其分组归一化的优势估计在响应层面是可靠的，但在步骤层面却存在系统性偏差，因为粗粒度的轨迹级优势难以准确反映单个步骤的贡献（即失败的轨迹可能包含有价值的步骤）。通过重新审视强化学习的基础定义，我们注意到GRPO在单轮任务上的成功源于其优势估计策略遵循了基本定义：从同一状态采样的多个动作的平均奖励构成可信的状态价值估计。将这种忠实的估计扩展到步骤层面，原则上需要从每个中间状态采样多个动作，但这在逐状态的基础上成本过高。为了缓解……

    arXiv:2609.28963v1 Announce Type: new  Abstract: Group-based reinforcement learning (RL) methods, such as GRPO and its variants, have become a leading paradigm for training reasoning and agentic large language models (LLMs). While their group-normalized advantage estimation is reliable at the response level, it becomes systematically biased at the step level, since coarse-grained trajectory-level advantages are hard to accurately reflect the contribution of individual steps (i.e, failed trajectories may contain valuable steps). Revisiting the foundational RL definition, we notice that GRPO's success on single-turn tasks stems from its advantage estimation strategy, which adheres to the basic definition: the mean reward of multiple actions sampled from the same state constitutes a credible state-value estimate. Extending the faithful estimation to step-level would in principle demand sampling multiple actions from each intermediate state, which is too costly on a per-state basis. To mit
    
[^206]: 从静态个人价值观到情境化个性化：面向大语言模型的贝叶斯个性化价值对齐

    From Static Personal Values to Contextualized Personalization: Bayesian Personalized Value Alignment for LLMs

    [https://arxiv.org/abs/2609.28942](https://arxiv.org/abs/2609.28942)

    受勒温场论启发，本文提出BaCVA方法，将静态个人价值观作为先验、情境依赖偏好作为后验，通过推理阶段的贝叶斯框架结合情境价值显著性估计与双视角个性化模块，实现大语言模型的情境化个性化价值对齐。

    

    随着大语言模型（LLMs）被期望能够满足多样化的用户偏好，个性化价值对齐变得日益重要。然而，现有方法通常在不同提示词下将模型输出与静态的价值观画像进行对齐，忽视了价值维度的显著性在不同情境之间存在巨大差异。受勒温场论的启发——该理论认为人类行为是由个人倾向与情境约束共同塑造的——我们将个人价值观建模为先验，将情境依赖的偏好建模为后验。我们提出了BaCVA，一种推理阶段的贝叶斯情境感知个性化价值对齐方法，通过整合静态个人价值观与特定场景下的价值显著性来近似后验个性化偏好。BaCVA首先从一般规范性响应中估计情境化的价值显著性，然后采用双视角个性化模块来推断后验偏好

    arXiv:2609.28942v1 Announce Type: new  Abstract: Personalized value alignment has become increasingly important as large language models (LLMs) are expected to accommodate diverse user preferences. However, existing methods typically align model outputs with a static value profile across prompts, overlooking that the salience of value dimensions varies substantially across contexts. Inspired by Lewin's Field Theory, which views human behavior as jointly shaped by personal dispositions and situational constraints, we model personal values as priors and context-dependent preferences as posteriors. We propose BaCVA, an inference-time Bayesian Context-aware personalized Value Alignment method that approximates posterior personalized preferences by integrating static personal values with scenario-specific value salience. BaCVA first estimates contextual value salience from generally normative responses, and then employs a dual-view personalization module to infer posterior preferences from 
    
[^207]: 面向自主渗透测试框架的校准决策模型：JEV与Laya作为LLM驱动的渗透测试智能体的“系统一”决策层

    Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers for LLM-Driven Pentest Agents

    [https://arxiv.org/abs/2609.28940](https://arxiv.org/abs/2609.28940)

    本文提出用JEV和Laya这类轻量级非生成式的“系统一”校准分类器作为LLM驱动的自主渗透测试智能体的专用决策层，以降低误报、纠正严重程度虚高并减少计算浪费。

    

    自主渗透测试框架使用大型语言模型（LLM）进行侦察、漏洞利用和报告生成，但往往依赖同样的模型来确认发现结果、评定严重程度并选择智能体。这可能导致误报、严重程度虚高以及计算资源浪费。我们研究了“系统一”决策模型——一种轻量级非生成式分类器，可返回类型化且经过校准的判定结果——如何支持这些决策。我们做出了五项贡献。第一，我们定义了四个决策点：发现结果裁决、严重程度重新校准、智能体剪枝和确认循环。第二，我们展示了一项探索性的NeuroSploit案例研究，将一次使用TypeSafe系统一（Jev）的运行与一次不使用它的运行进行对比，测试目标为包含13个漏洞的Web应用。严重程度分布、运行时间以及按暴露数据类型评级的差异为该架构提供了动机，但并未确立统计显著性。第三，我们回顾了已发表的……（摘要原文在此处截断）

    arXiv:2609.28940v1 Announce Type: cross  Abstract: Autonomous penetration-testing harnesses use large language models (LLMs) for reconnaissance, exploitation, and reporting, but often rely on those same models to confirm findings, grade severity, and select agents. This can lead to false positives, inflated severity, and wasted compute. We examine how System One decision models, lightweight non-generative classifiers that return typed, calibrated verdicts, can support these decisions. We make five contributions. First, we define four decision points: finding adjudication, severity recalibration, agent pruning, and confirmation loops. Second, we present an exploratory NeuroSploit case study comparing one run with TypeSafe System One (Jev) and one without it against a web target containing 13 vulnerabilities. Differences in severity distribution, runtime, and grading by exposed data type motivate the architecture but do not establish statistical significance. Third, we review published s
    
[^208]: PFArena：蛋白质修饰语言模型基准测试

    PFArena: Benchmarking Language Models for Protein Modification

    [https://arxiv.org/abs/2609.28921](https://arxiv.org/abs/2609.28921)

    该论文提出了PFArena基准，通过四个受控任务界面系统评估蛋白质语言模型、大语言模型及LLM智能体在不同实验先验知识场景下进行蛋白质修饰（单突变生成与多突变排序）的能力。

    

    蛋白质修饰需要在巨大的序列空间中进行探索，而湿实验验证仍然通量低且成本高昂。尽管包括蛋白质语言模型（PLM）、大语言模型（LLM）以及基于LLM的智能体等计算范式在蛋白质修饰方面已展现出潜力，但它们在真实实验决策场景中的相对有效性仍不明确。为弥合这一差距，我们提出了PFArena，一个包含四个受控任务界面的基准测试，涵盖单突变体生成和多突变体排序任务。通过提供不同水平的突变适应性数据，PFArena反映了四种以不同程度先验实验背景为特征的代表性研究场景。我们使用互补的评估指标对六个蛋白质语言模型、六个大语言模型和五个基于LLM的智能体进行了评估，以衡量其在蛋白质修饰任务上的峰值性能和整体性能。我们的评估表明，模型性能会随……（原文此处截断）

    arXiv:2609.28921v1 Announce Type: new  Abstract: Protein modification requires navigating an immense sequence space, yet wet-lab validation remains low-throughput and costly. Although computational paradigms including protein language models (PLMs), large language models (LLMs), and LLM-based agents have shown promise in protein modification, their relative efficacy across realistic experimental decision-making settings remains unclear. To bridge this gap, we introduce PFArena, a benchmark comprising four controlled task interfaces that cover single-mutant generation and multi-mutant ranking. By providing varying levels of mutation fitness data, PFArena reflects four representative research scenarios characterized by differing degrees of prior experimental context. We assess six PLMs, six LLMs, and five LLM-based agents using complementary metrics to measure both peak and overall protein modification performance. Our evaluation reveals that model performance shifts systematically with 
    
[^209]: 掌控代理运行框架，即掌控成本：企业中AI编程代理的路由与治理

    Control the Harness, Control the Cost: Routing and Governing AI Coding Agents in the Enterprise

    [https://arxiv.org/abs/2609.28919](https://arxiv.org/abs/2609.28919)

    本文提出一个配备校准概率分类器Jev的快速可定制路由器，使企业能够治理AI编程代理运行框架的模型选择与缓存决策，从而有效控制成本。

    

    运行AI编程代理的产品（即“harness”，代理运行框架）正在成倍涌现，企业正将其部署给员工：最初几百个席位的试点正在扩展到数万个规模。大多数企业并不自行构建这些框架，而是从大型供应商处购买，例如Anthropic的Claude Code或OpenAI的Codex。框架决定了由哪个模型作答、模型读取什么内容、提示缓存如何使用以及哪些子代理运行，因此它决定了价目表上的费率以及在该费率下的用量。那些将专有或未经调优的框架保持默认设置的企业，会一并继承这些选择及其账单。我们构建了一个快速、可定制的路由器，其中的“Jev”是一个具有校准概率的分类器，可依据用户自定义的代理请求分类法为每个提示打标签。由于用户的一轮交互包含针对属于单一模型的提示缓存的多个请求，路由器仅在没有进行中的对话的地方转移工作……

    arXiv:2609.28919v1 Announce Type: new  Abstract: Harnesses, the products that run AI coding agents, are multiplying, and enterprises are rolling them out to their employees: what started as pilots with a few hundred seats is scaling to tens of thousands. Most enterprises do not build these harnesses but buy them from large vendors, such as Anthropic's Claude Code or OpenAI's Codex. A harness decides which model answers, what the model reads, how the prompt cache is used and which subagents run, so it picks the rate on the price sheet and sets the volume bought at it. Enterprises that keep a proprietary or untuned harness at its defaults inherit these choices and their bill. We build a fast, customisable router in which Jev, a classifier with calibrated probabilities, labels every prompt against a bring-your-own taxonomy of agentic requests. Because one user turn is many requests over a prompt cache that belongs to one model, the router moves work only where no running conversation has 
    
[^210]: 关于内核级证据对智能体安全有效性的研究

    On the Effectiveness of Kernel-Level Evidence for Agent Security

    [https://arxiv.org/abs/2609.28915](https://arxiv.org/abs/2609.28915)

    该论文首次将应用层智能体遥测与内核级系统调用追踪配对，提出包含 4,047 个会话的 ACE 语料库，证明内核级证据能够揭示逃过应用层检测的智能体安全威胁。

    

    LLM 智能体被部署在赋予其广泛主机权限的基础设施中，然而现有的智能体安全基准测试和防御措施几乎完全运行在应用遥测层：即所提供的工具清单、用户提示词以及模型的消息。然而，某些威胁会将恶意指令和动作偷运过应用边界，使其对该层不可见。在本工作中，我们通过将应用层智能体遥测与内核级系统调用追踪配对，首次提出了针对智能体安全的内核级信号与应用层信号的配对证据表征。为了量化增强遥测的价值，我们引入了 Agent Cross-Layer Evidence（ACE），一个包含 4,047 个会话和 17 种威胁模型的配对会话语料库，涵盖六类投递向量家族以及 25 个 OWASP LLM 与智能体威胁类别中的 14 个，并组织为 12 种攻击机制，每种机制均带有各自的表征。

    arXiv:2609.28915v1 Announce Type: cross  Abstract: LLM agents are deployed into infrastructure that grants them broad host authority, yet existing agent-security benchmarks and defenses operate almost exclusively at the application telemetry layer: the served tool manifest, the user prompt, and the model's messages. Some threats, however, smuggle malicious instructions and actions past the application boundary, leaving them invisible to that layer. In this work, we bridge that gap by pairing application-level agent telemetry with kernel-level syscall traces to present the first paired-evidence characterization of kernel-level versus application-layer signal for agent security. To quantify the value of the enhanced telemetry, we introduce Agent Cross-Layer Evidence (ACE), a paired-session corpus of 4,047 sessions and 17 threat models spanning six delivery-vector families and 14 of the 25 OWASP LLM and agentic threat categories, organized into 12 attack mechanics with per-mechanic charac
    
[^211]: 主动行动的机器人：构建与评估主动性机器人的框架

    Robots That Take Initiative: A Framework for Building and Evaluating Proactive Robots

    [https://arxiv.org/abs/2609.28910](https://arxiv.org/abs/2609.28910)

    该论文提出了主动性机器人辅助的统一形式化框架并将其划分为三个层次，指出离线评估会高估主动性机器人的性能，同时贡献了带有自适应人类模型的闭环评估方法，以及通过被动观察学习预测用户目标并主动行动的GAP方法。

    

    arXiv:2609.28910v1 公告类型：cross 摘要：要让机器人在狭窄角色和重复性任务之外提供有效帮助，机器人必须具备主动性——即自行决定需要做什么，而不是等待被告知。尽管主动性正受到越来越多的探索，但它缺乏统一的形式化定义，且该领域的工作通常在离线环境下针对静态人类模型进行评估，无法捕捉机器人的行为对环境以及用户自身行为的影响。我们为主动性机器人辅助引入了统一的形式化表示，将其组织为三个层次，并提供了一个针对最高层次——即无提示的主动性辅助——的框架。随后我们证明，在此设定下，离线评估会高估性能，并贡献了一种带有能够适应机器人行为的人类模型的闭环评估方法。最后，我们提出了一种名为GAP的方法来实例化我们的框架，该方法从被动观察中学习以预测用户目标并采取行动。在闭环评估下，先前最先进的（原文在此处截断）

    arXiv:2609.28910v1 Announce Type: cross  Abstract: Effective robot assistance beyond narrow roles and repetitive tasks requires robots to be proactive - to decide what needs to be done rather than waiting to be told. While proactivity is increasingly explored, it lacks a unified formulation, and work in the domain is typically evaluated offline against static human models that cannot capture the effect of a robot's actions on the environment and the user's own behavior. We introduce a unified formalism for proactive robot assistance, organize it into three levels, and provide a framework to address the highest level of unprompted proactive assistance. We then show that offline evaluation overstates performance in this setting, and contribute a closed-loop evaluation with a human model that adapts to the robot. Finally, we present a method, GAP, that instantiates our framework, learning from passive observation to anticipate user goals and act. Under closed-loop evaluation, prior state-
    
[^212]: 拓展音频问答中的不确定性估计：跨方法、格式与输入的全面研究

    Broadening Uncertainty Estimation for Audio Question Answering Across Methods, Formats, and Inputs

    [https://arxiv.org/abs/2609.28879](https://arxiv.org/abs/2609.28879)

    该论文系统比较了五类不确定性估计方法在音频问答中的表现，发现无需额外模型调用的首token概率度量在多项选择评估中最强，且在开放式评估中不确定性仍能有效预测模型错误，输入消融实验进一步表明不确定性与回答问题可用的证据相关。

    

    音频-语言模型可能给出音频内容并不支持的自信回答，这促使人们需要通过不确定性估计来识别不可靠的回答。我们在四个开放权重模型和五个音频问答基准上，比较了基于概率、基于采样、自我验证、证据型和对比型的多种度量方法。在多项选择评估中，首token度量整体表现最强，其中top-1概率的平均AUROC达到0.740，相比之下十样本离散语义熵为0.708，而首token方法无需额外的模型调用。在四个基准上，从多项选择转向开放式评估使平均准确率从57.6%降至36.6%，但不确定性依然能够预测错误：语义熵、最大token熵和语义一致性分别取得了0.697、0.694和0.693的平均AUROC。为了检验不确定性是否反映了回答问题可用的证据，我们进行了输入消融实验……

    arXiv:2609.28879v1 Announce Type: cross  Abstract: Audio-language models can produce confident answers unsupported by the audio, motivating uncertainty estimates that identify unreliable responses. We compare probability-based, sampling-based, self-verification, evidential, and contrastive measures across four open-weight models and five audio QA benchmarks. In multiple-choice evaluation, first-token measures are strongest overall, with top-1 probability achieving a mean AUROC of .740, compared with .708 for ten-sample discrete semantic entropy, while requiring no additional model calls. Across four benchmarks, shifting from multiple-choice to open-ended evaluation lowers mean accuracy from 57.6% to 36.6%, yet uncertainty remains predictive of errors: semantic entropy, maximum token entropy, and semantic agreement achieve mean AUROCs of .697, .694, and .693, respectively. To test whether uncertainty reflects the evidence available to answer the question, we perform input ablations that
    
[^213]: Forecast-Dojo：用于基准测试和训练大语言模型预测代理的可重放环境

    Forecast-Dojo: Replayable Environments for Benchmarking and Training LLM Forecasting Agents

    [https://arxiv.org/abs/2609.28876](https://arxiv.org/abs/2609.28876)

    该论文提出了Forecast-Dojo——一个结合已结算预测市场问题与带日期新闻的可重放环境，用于基准测试和训练LLM预测代理，实验发现研究工具能提升全部12个模型的预测表现，但所有模型仍落后于历史市场预测水平。

    

    我们推出了Forecast-Dojo，这是一个用于基准测试和训练大语言模型预测代理的可重放环境。它将已结算的预测市场问题与带日期标注的新闻相结合，使代理能够研究某个事件，并在连续的历史时间点上更新其预测。相同的任务和工具支持重复评估、训练交互数据的收集，以及来自已记录结果的反馈，而无需等待新事件结算。Forecast-Dojo包含1,568个Polymarket事件（按时间划分为训练期和评估期）以及1,880万篇带日期标注的新闻文章。在对12个模型的评估中，研究工具降低了所有12个模型的Brier分数。预测质量也随着事件的发展而提升，在记录到更多新增带日期证据的步骤中提升幅度最大。然而，所有模型在Brier分数和准确率上仍然落后于历史市场预测。在时间点之间携带的信念笔记本降低了研究成本，但并不能持续改善预测表现（原文此处截断）。

    arXiv:2609.28876v1 Announce Type: new  Abstract: We introduce Forecast-Dojo, a replayable environment for benchmarking and training LLM forecasting agents. It combines resolved prediction-market questions with dated news, allowing agents to research an event and revisit their predictions at successive historical dates. The same tasks and tools support repeated evaluation, collection of training interactions, and feedback from recorded outcomes without waiting for new events to resolve. Forecast-Dojo contains 1,568 Polymarket events, split by time into training and evaluation periods, and 18.8M dated news articles. In an evaluation of 12 models, research tools lower Brier score for all 12. Forecasts also improve as events unfold, with the largest gains at steps where more newly dated evidence is recorded. Every model still trails historical market forecasts in both Brier score and accuracy. A belief notebook carried between dates lowers research cost but does not consistently improve fo
    
[^214]: 人机协同驱动的假设检验：成本感知的选择性AI评分与序贯式人工升级

    Human-AI-Powered Hypothesis Testing: Cost-Aware Selective AI Scoring and Sequential Human Escalation

    [https://arxiv.org/abs/2609.28859](https://arxiv.org/abs/2609.28859)

    该论文提出了一个人机协同的假设检验框架，通过成本感知的选择性AI评分与序贯式人工升级策略，在严格控制第一类和第二类错误的前提下以最低成本实现有效的统计推断。

    

    大型语言模型正越来越多地被用作低成本的评判者，用于评估输出结果、标注数据以及判断一个系统是否达到期望的质量标准。然而，将AI判断用于正式的统计推断与简单地将它们视为真实标签有着根本的不同：AI评估可能存在偏差或噪声，而严格的假设检验需要对第一类错误和第二类错误进行显式控制。我们研究如何利用AI判断，结合选择性的人工验证，以最低的成本进行有效的假设检验。我们考虑一个具有隐藏二元标签的项目总体。在选定一个固定的项目池之后，决策者可以选择性地查询AI，将某个项目直接交给人工处理，在观察到AI的报告后再将经过AI评分的项目升级给人工复核，或者在积累了足够的证据后停止。我们推导出了一个信息论下界，该下界刻画了在满足规定的错误控制要求下所需的最小成本

    arXiv:2609.28859v1 Announce Type: new  Abstract: Large language models are increasingly used as inexpensive judges to evaluate outputs, label data, and assess whether a system meets a desired quality standard. Yet using AI judgments for formal statistical inference is fundamentally different from simply treating them as ground-truth labels: AI evaluations can be biased or noisy, and rigorous hypothesis testing requires explicit control of type-I and type-II errors. We study how to use AI judgments, together with selective human verification, to conduct a valid hypothesis test at minimum cost. We consider a population of items with hidden binary labels. After choosing a fixed pool of items, the decision maker can selectively query AI, send an item directly to a human, escalate an AI-scored item to a human after observing the AI report, or stop once sufficient evidence has accumulated. We derive an information-theoretic lower bound that captures the minimum cost of achieving prescribed t
    
[^215]: 被说服，而非被告知：激励错位的证人击败上下文接地

    Persuaded, Not Informed: Incentive-Misaligned Witnesses Defeat In-Context Grounding

    [https://arxiv.org/abs/2609.28854](https://arxiv.org/abs/2609.28854)

    该论文发现，当CRM上下文中包含销售代表这类有乐观动机的证人所作的断言时，各主流大语言模型都会将其当作可信证据，无视公司内部记录中的矛盾信息而错误批准不合格交易，且更强的模型、更大的规模和显式推理均无法抵抗这种“被说服而非被告知”的失败模式。

    

    语言模型智能体越来越多地基于客户关系管理（CRM）记录来回答问题，例如是否应将某条销售线索判定为合格。我们发现了一种并非更强模型所能解决的失败模式：当上下文中包含来自一个有乐观倾向动机的当事人的断言时——此处即为CRM中记录在案的证人销售代表——模型会将该断言视为证据，并批准公司自身记录认定为不可接受的交易。在来自CRMArena-Pro的100个线索资格判定任务中，该销售代表在每一次通话中都断言时间线可接受，在76个任务中断言预算可接受；在这类断言与价目表及安装政策相矛盾的31个任务中，仅阅读通话记录的模型在31例中有29例批准了该交易。这一失败特征在来自四家提供商的七个模型上保持一致（87%–97%被误导）；模型规模和显式推理均无法提供抵抗力。在35个真实失败案例中仅有3个……

    arXiv:2609.28854v1 Announce Type: cross  Abstract: Language-model agents increasingly answer questions over customer-relationship management (CRM) records, such as whether to qualify a sales lead. We identify a failure mode not addressed by a stronger model: when the context contains an assertion by a party with an incentive toward optimism - here the sales representative, a witness recorded in the CRM - the model treats the assertion as evidence and clears deals the company's own records deem unacceptable. Across 100 lead-qualification tasks from CRMArena-Pro, the representative asserts an acceptable timeline in every call and an acceptable budget in 76; on the 31 tasks where such an assertion contradicts the price list and installation policy, a model reading only the transcript clears the deal in 29 of 31 cases. The signature is consistent across seven models from four providers (misled on 87-97%); scale and explicit reasoning confer no resistance. Only 3 of 35 genuine failures invo
    
[^216]: RECLAIM：智能体能复现机器学习论文的结论吗？

    RECLAIM: Can Agents Reproduce the Claims of Machine Learning Papers?

    [https://arxiv.org/abs/2609.28850](https://arxiv.org/abs/2609.28850)

    RECLAIM是一个基于100篇NeurIPS 2025论文的可重建基准测试，通过预先定义复现目标、成功标准和GPU预算，并按作者发布资源分为运行、重训练、重新实现三个难度级别，用独立语言模型依据日志评分，结果显示最好的AI智能体也只能分别复现41%、27%和15%的论文结果。

    

    复现一篇机器学习论文涉及大部分研究步骤，从安装软件、调试到运行实验，这些工作正日益由AI智能体来承担。我们提出了RECLAIM，一个基于100篇NeurIPS 2025论文的基准测试，可以每年从新会议中重建更新。对于每篇论文，我们预先固定要复现的结果、判定复现成功的标准以及GPU小时预算。智能体必须利用论文本身和作者发布的资源来复现该结果。作者发布的内容决定了难度级别：Run级（运行级）发布包含代码、数据和模型权重；Retrain级（重训练级）发布缺少权重，因此智能体需要自行训练模型；Reimplement级（重新实现级）发布缺少代码，因此智能体需要自己编写代码。我们使用一个独立的语言模型根据日志和输出（而非智能体自己的报告）来评分。我们在每篇论文上运行四个智能体各一次；结果发现每个级别中表现最好的智能体也仅能复现Run级论文的41%、Retrain级的27%和Reimplement级的15%。

    arXiv:2609.28850v1 Announce Type: new  Abstract: Reproducing a machine learning paper involves most research steps, from installing software and debugging to running experiments, work that AI agents increasingly do. We introduce RECLAIM, a benchmark of 100 NeurIPS 2025 papers that can be rebuilt yearly from new conferences. For each paper we fix in advance the result to reproduce, what counts as a successful reproduction, and a GPU-hour budget. An agent must reproduce that result using the paper and whatever its authors released. What the authors released decides the difficulty tier. Run-tier releases include code, data, and weights; Retrain-tier releases lack weights, so the agent trains the model; Reimplement-tier releases lack code, so the agent writes it. A separate language model grades runs from logs and outputs rather than agents' reports. We run four agents once per paper; the best agent in each tier reproduces only 41% of Run-tier papers, 27% at Retrain, and 15% at Reimplement
    
[^217]: 区块链赋能的人工智能与AI智能体在安全数据共享与网络安全应用中的研究

    Blockchain-Enabled Artificial Intelligence and AI Agents for Secure Data Sharing and Cybersecurity Applications

    [https://arxiv.org/abs/2609.28843](https://arxiv.org/abs/2609.28843)

    本文通过元综合方法整合四项研究，论证了区块链与人工智能的融合能够应对AI驱动安全运营中的关键故障点，包括训练数据与模型行为的完整性、实时监控的可靠性以及自动化代码修复的可信度。

    

    区块链与人工智能（AI）正在融合为统一的基础设施层，用于保障分布式系统中的数据共享、模型完整性与自主决策。本文提出了一项元综合研究，汇集了四项子研究，内容涵盖对抗性机器学习、云环境中基于AI的异常检测、由多智能体大语言模型（LLM）流水线执行的自动化漏洞修补，以及AI系统全生命周期安全保障的整体格局，并将这些研究发现置于区块链赋能AI与自主AI智能体的新兴文献之中。每项子研究都针对现代AI驱动的安全运营中一个不同的故障点：训练数据与模型行为的完整性、实时监控的可靠性，以及自动化代码修复的可信度。我们认为，区块链所具有的不可篡改性、去中心化等特性……（摘要原文在此处截断）

    arXiv:2609.28843v1 Announce Type: cross  Abstract: Blockchain and artificial intelligence (AI) are converging into a single infrastructural layer for securing data sharing, model integrity, and autonomous decision-making across distributed systems. This paper presents a meta-synthesis that draws together four constituent studies covering adversarial machine learning, AI-powered anomaly detection in cloud environments, automated vulnerability patching by multi-agent large language model (LLM) pipelines, and the broader landscape of securing AI systems across their lifecycle and situates their findings within the emerging literature on blockchain-enabled AI and autonomous AI agents. Each constituent study addresses a distinct point of failure in modern AI-driven security operations: the integrity of training data and model behavior, the reliability of real-time monitoring, and the trustworthiness of automated code remediation. We argue that blockchain's properties of immutability, decent
    
[^218]: M²PFN：面向阿尔茨海默病可泛化多模态上下文学习的端到端解耦对齐

    M$^2$PFN: End-to-End Disentangled Alignment for Generalizable Multimodal In-Context Learning in Alzheimer's Disease

    [https://arxiv.org/abs/2609.28836](https://arxiv.org/abs/2609.28836)

    提出端到端框架M²PFN，通过在TabPFN的transformer中进行可微推理、并利用解耦与对比学习将3D-MRI和表格模态对齐至共享子空间，从而把表格基础模型的上下文学习能力扩展为可跨队列泛化的多模态阿尔茨海默病诊断器。

    

    尽管已有多种结合影像与表格数据进行阿尔茨海默病（AD）诊断的多模态方法被提出，但它们在跨队列泛化方面往往存在局限。上下文学习（ICL）已在TabPFN等基础表格模型中展现出卓越的泛化性能和高度灵活性。将TabPFN的ICL扩展至多模态AD分析的主要障碍在于：TabPFN是在合成表格先验上进行元训练的，而这些先验与图像衍生特征的统计结构并不天然匹配。我们提出了M²PFN，一个能将该表格基础模型转变为多模态AD预测器的端到端框架。M²PFN (i) 通过TabPFN的transformer执行可微推理，将任务梯度反向传播至3D-MRI编码器和表格编码器；(ii) 通过解耦与对比目标，将两种模态对齐到与ICL引擎先验相匹配的共享子空间中；(iii) …（原文摘要在此处截断）

    arXiv:2609.28836v1 Announce Type: cross  Abstract: While various multimodal methods combining imaging and tabular data for Alzheimer's disease (AD) diagnosis were proposed, they are often limited in generalization across cohorts. In-context learning (ICL) has demonstrated excellent generalization performances and high flexibility in foundational tabular models such as TabPFN. To extend TabPFN's ICL to multimodal AD analysis, the main obstacle is that TabPFN is meta-trained on synthetic tabular priors that do not naturally match the statistical structure of image-derived features. We propose M$^2$PFN, an end-to-end framework that turns this tabular foundation model into a multimodal AD predictor. M$^2$PFN (i) performs differentiable inference through TabPFN's transformer, back-propagating task gradients into 3D-MRI and tabular encoders; (ii) aligns the two modalities into a shared subspace, via disentanglement and a contrastive objective, matched to the ICL engine's prior; and (iii) fol
    
[^219]: KeyGen：基于无监督关键点的物体中心表示，实现类别级别的策略泛化

    KeyGen: Unsupervised Keypoint based Object-Centric Representations for Category-Level Policy Generalization

    [https://arxiv.org/abs/2609.28818](https://arxiv.org/abs/2609.28818)

    KeyGen通过无监督学习从点云中提取规范化语义3D关键点作为物体中心表示，用于条件化视觉运动扩散策略，从而实现机器人操作策略在类别级别上对新物体实例的泛化能力。

    

    机器人操作中的泛化要求策略能够在形状、大小和姿态各异的多种未见物体实例上执行任务。然而，传统的行为克隆（BC）方法往往会对特定实例的几何形状和外观产生过拟合，限制了对新物体的迁移能力。我们提出了KeyGen，这是一个从点云中学习规范化语义3D关键点，并将其作为结构化物体中心表示用于策略学习的框架。视觉运动扩散策略以这些关键点以及物体中心几何信息为条件来预测完整的操作轨迹，从而实现跨物体实例的一致几何对应关系。为了评估类别级别的泛化能力，我们构建了一个包含三个操作任务的逼真仿真基准，以及一个规划驱动的数据生成流水线，可在多样化的物体实例上生成专家轨迹。实验表明，KeyGen（摘要在此处被截断）

    arXiv:2609.28818v1 Announce Type: cross  Abstract: Generalization in robotic manipulation requires policies to perform tasks across diverse unseen object instances that vary in shape, size, and pose. However, conventional behavior cloning (BC) methods often overfit to instance-specific geometry and appearance, limiting transfer to novel objects. We introduce KeyGen, a framework that learns canonicalized semantic 3D keypoints from point clouds and uses them as structured object-centric representations for policy learning. A visuomotor diffusion policy conditions on these keypoints together with object-centric geometry to predict full manipulation trajectories, enabling consistent geometric correspondence across object instances. To evaluate category-level generalization, we construct a photorealistic simulation benchmark with three manipulation tasks and a planning-driven data generation pipeline that produces expert trajectories across diverse object instances. Experiments show that Ke
    
[^220]: 一个用于合成多样化自然全双工对话的框架

    A Harness for Synthesizing Diverse Naturalistic Full-Duplex Conversations

    [https://arxiv.org/abs/2609.28806](https://arxiv.org/abs/2609.28806)

    该论文提出了一条从关系事件列表合成带意图标注的双通道全双工对话语音的流程，实现了对停顿、打断等轮次转换事件及其意图的精细控制与标注，覆盖英语和普通话的42种对话现象。

    

    全双工对话系统在说话的同时进行倾听，必须区分“已完成的发言轮次”与“轮次内的停顿”，以及区分“请求发言权的打断”与“简短的附和回应或对第三方说的话”。然而，现有的对话语料库对这些事件的控制有限，对其意图的标注也很有限。我们提出了一种从关系事件列表合成带意图标注的双通道对话语音的流程。由大语言模型（LLM）为每个事件撰写说话人、文本、会话行为以及与先前事件的关联，而无需预测绝对时间戳。各事件独立合成，与其源文本对齐，并被放置在共享时钟上，因此轮次转换的关键时间点从渲染后的信号中测量，而静音时长则通过指定或从轮次转换分布中采样获得。该流程覆盖英语和普通话中八大类共42种现象，并生成帧级系统（摘要在此处被截断）

    arXiv:2609.28806v1 Announce Type: cross  Abstract: Full-duplex dialogue systems, which listen while speaking, must distinguish a completed turn from a pause within a turn and an interruption that requests a turn from a brief acknowledgment or speech addressed to a third party. Yet existing conversational corpora provide limited control over these events and limited labels for their intent. We present a pipeline for synthesizing intent-labeled, two-channel conversational speech from relational event lists. An LLM authors each event's speaker, text, conversational act, and attachment to an earlier event without predicting absolute timestamps. Events are synthesized independently, aligned with their source text, and placed on a shared clock, so turn-taking landmarks are measured from the rendered signal while silence durations are specified or sampled from turn-taking distributions. The pipeline covers 42 phenomena across eight families in English and Mandarin, derives frame-level system 
    
[^221]: DrGait：基于生物力学的视觉推理，实现可解释的临床步态分析

    DrGait: Biomechanically Grounded Visual Reasoning for Interpretable Clinical Gait Analysis

    [https://arxiv.org/abs/2609.28796](https://arxiv.org/abs/2609.28796)

    DrGait是一个无需训练的智能体框架，通过让视觉-语言模型充当临床规划者而非直接视觉推理者，并结合分诊-验证-综合（TVS）工作流与确定性生物力学工具，实现了可解释、抗幻觉的临床步态分析。

    

    当前面向临床应用的自动化步态分析依赖于不可解释的黑盒分类器。尽管视觉-语言模型（VLMs）具备强大的推理能力，但将其直接应用于步态视频往往会产生幻觉，因为它们难以从原始视觉上下文中测量细微的几何偏差。为解决这一问题，我们提出了DrGait，这是一个无需训练的智能体框架，它将VLM的角色从直接的视觉推理者转变为临床规划者。DrGait通过结构化的分诊-验证-综合（Triage-Verification-Synthesis, TVS）工作流，将语义推理与几何感知解耦。给定输入视频和一组基础时空指标后，DrGait智能体首先执行启发式分诊以提出诊断假设，随后通过自主调用确定性的生物力学工具来验证这些假设，这些工具作用于重建的3D网格轨迹、分割的2D姿态轨迹以及以事件为中心的……

    arXiv:2609.28796v1 Announce Type: cross  Abstract: Current automated gait analysis for clinical applications relies on uninterpretable black-box classifiers. Although Vision-Language Models (VLMs) offer strong reasoning capabilities, applying them directly to gait videos often leads to hallucinations, because they struggle to measure subtle geometric deviations from raw visual contexts. To address this, we introduce DrGait, a training-free agentic framework that shifts the VLM's role from a direct visual reasoner to a clinical planner. DrGait decouples semantic reasoning from geometric perception through a structured Triage-Verification-Synthesis (TVS) workflow. Given an input video and a set of basic spatiotemporal metrics, the DrGait agent first performs a heuristic triage to propose diagnostic hypotheses, which are then verified by autonomously calling deterministic biomechanical tools that operate on reconstructed 3D mesh trajectories, segmented 2D pose tracks, and event-centered v
    
[^222]: 多任务模型中学习到的跨任务关系

    Learned Cross-Task Relationships in Multi-Task Models

    [https://arxiv.org/abs/2609.28776](https://arxiv.org/abs/2609.28776)

    该论文提出一个通过针对性成对关系近似任务标签联合分布来学习多任务模型中跨任务关系的框架，并在YouTube生产推荐系统中验证了其在准确性和用户满意度上的显著提升。

    

    我们提出了一个框架，通过针对性的成对关系来近似任务标签的联合分布，从而学习多任务模型中的跨任务关系。这种方法通过迁移学习提升了性能，并增强了信息提取能力，同时避免了建模完整联合空间所带来的难以处理的复杂性。尽管我们的框架适用于任何多任务系统，但我们在YouTube的生产推荐系统中展示了其有效性。在通知、首页和“接下来观看”等页面上的实验表明，准确性和用户满意度指标均有所提升。最后，我们提出了一个工作流程模板，以便于未来更广泛的实施。

    arXiv:2609.28776v1 Announce Type: new  Abstract: We propose a framework that learns cross-task relationships in multi-task models by approximating the joint distribution of task labels through targeted pairwise relationships. This approach improves performance via transfer learning and enhances information extraction without the intractable complexity of modeling the full joint space. Although our framework applies to any multi-task system, we demonstrate its efficacy within YouTube's production recommendation systems. Experiments across the Notifications, Homepage, and Watch Next surfaces show improvements in both accuracy and user satisfaction metrics. Finally, we propose a workflow template to facilitate broader future implementation.
    
[^223]: 面向金融决策的基于情景检索的智能体记忆

    Agent Memory with Episodic Retrieval for Financial Decision-Making

    [https://arxiv.org/abs/2609.28771](https://arxiv.org/abs/2609.28771)

    META是首个类RAG的情景记忆增强多智能体金融决策框架，它通过整合多个专门的技术指标智能体与记忆模块，检索带有结果与反思的历史交易情景，在相似市场状态下自适应地重新加权信号，从而提升复杂环境下的交易决策能力。

    

    大型语言模型（LLMs）在金融分析与推理方面展现出强大能力，推动了近期基于智能体的交易框架的进展。尽管这些系统显示出前景，但现有方法要么侧重于长周期预测，要么作为无状态的分析器运行，这限制了它们在复杂交易环境中的应用能力。为弥补这些不足，我们提出了META（记忆增强交易智能体），这是首个类RAG的情景记忆增强多智能体金融决策框架。META整合了一族专门的指标智能体（如趋势、MACD、随机指标、RSI、SMA、AVWAP、Heikin-Ashi），一个融合各指标报告的决策智能体，以及一个记忆模块，该模块检索并更新以市场状态嵌入形式编码的历史交易情景，其中包含交易结果与反思信息。通过回忆相关经验并在相似市场状态下自适应地重新加权各信号，META……

    arXiv:2609.28771v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated strong capabilities in financial analysis and reasoning, inspiring recent advances in agent-based trading frameworks. While these systems show promise, prior approaches either emphasize long-horizon forecasting or operate as stateless analyzers, limiting their applicability to the demands of trading in complicated settings. To address these gaps, we introduce META (Memory Enhanced Trading Agent), the first RAG-like episodic-memory-augmented multi-agent framework for financial decision making. META integrates a family of specialized indicator agents (e.g., Trend, MACD, Stochastic, RSI, SMA, AVWAP, Heikin-Ashi) with a Decision Agent that fuses their reports, and a Memory module that retrieves and updates past trading episodes encoded as market state embeddings with outcomes and reflections. By recalling relevant experiences and adaptively reweighting signals under similar market regimes, META 
    
[^224]: 面向小型搜索智能体的可验证奖励强化学习

    Reinforcement Learning with Verifiable Rewards for Small Search Agents

    [https://arxiv.org/abs/2609.28765](https://arxiv.org/abs/2609.28765)

    该研究首次证明，无需蒸馏，仅用GRPO和维基百科搜索工具在MuSiQue上训练0.8B参数的小模型，即可成功应用可验证奖励强化学习于开放域问答，在七个基准上取得未训练基线3.8倍的精确匹配率（0.352）。

    

    可验证奖励强化学习在奖励明确的问题（如数学和编程）上表现出色，但在奖励不那么明确的场景中是否同样有效仍是一个悬而未决的问题。“推理-搜索”方法将RLVR应用于开放域问答，其中检索为答案提供依据，与参考答案的匹配提供奖励信号。到目前为止，该方法仅在大型模型上得到验证，而在十亿参数以下的模型上只有借助更大教师模型蒸馏的先例。我们在一个小模型上测试了该方法：我们使用组相对策略优化（GRPO）和交错的维基百科搜索工具，在MuSiQue数据集上训练Qwen3.5-0.8B，仅在三种子奖励形式上进行变化（每种形式各三个随机种子），并在七个基准问答测试套件上评估每个保留检查点。结果表明该方法有效：最佳运行达到0.352的平均精确匹配率，而未训练基线仅为0.092，实现了3.8倍的提升，且无需任何蒸馏步骤。

    arXiv:2609.28765v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) performs well on problems with clear rewards, such as mathematics and coding, but whether it also works where the reward is less clear remains open. The reason-over-search recipe applies RLVR to open-domain question answering, where retrieval grounds the answer and a match against the reference supplies the reward. So far it has been demonstrated on large models, and below one billion parameters only with distillation from a larger teacher. We test the recipe on a small model. We train Qwen3.5-0.8B with Group Relative Policy Optimization (GRPO) and an interleaved Wikipedia-search tool on MuSiQue, varying only the reward across three shapes over three seeds each, and we evaluate every checkpoint held-out on a seven-benchmark question-answering suite. The recipe works: the best run reaches 0.352 average exact match against a 0.092 untrained floor, a 3.8-fold gain, with no distillation s
    
[^225]: KathDB-FAO：多模态数据库管理系统中的合成查询计划

    KathDB-FAO: Synthesized Query Plans in a Multimodal DBMS

    [https://arxiv.org/abs/2609.28761](https://arxiv.org/abs/2609.28761)

    KathDB-FAO 将自然语言查询转换为算子函数体在运行时动态合成的查询执行计划，实现针对特定查询的强大优化，在 SemBench 上平均降低 58.8% 的执行成本。

    

    我们设计、实现并评估了 KathDB-FAO，这是我们 KathDB 多模态数据库管理系统的一个全新查询评估子系统。KathDB-FAO 接收自然语言（NL）查询作为输入，并将其转换为查询执行计划，其中每个算子都是一个函数，其函数体在查询评估过程中被动态合成，从而实现强大的针对特定查询的优化。为了从自然语言生成准确且高效的计划，KathDB-FAO 首先提取细粒度的原子操作以保证正确性，然后在这些操作的输入和输出上建立契约并将其分组以提升效率，最后在运行时为每个组即时合成函数。在 SemBench 基准测试中，与次优系统相比，KathDB-FAO 在各场景下平均降低了 58.8% 的执行成本，同时保持相当或更好的查询质量。

    arXiv:2609.28761v1 Announce Type: cross  Abstract: We design, implement, and evaluate KathDB-FAO, a new query evaluation subsystem for our KathDB multimodal DBMS. KathDB-FAO takes as input a query in natural language (NL) and converts it into a query execution plan where each operator is a function whose body is synthesized during query evaluation, which allows powerful query-specific optimizations. To generate accurate and efficient plans from NL, KathDB-FAO first extracts fine-grained atomic actions for correctness, then establishes contracts on the inputs and outputs of those actions and groups them for efficiency, and finally synthesizes the function for each group on the fly. On SemBench, KathDB-FAO cuts execution cost by 58.8% on average across scenarios compared with the next best system, at comparable or better quality.
    
[^226]: 基于虚构语料库微调的置信度-语料库一致性工具包技术手册

    Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus

    [https://arxiv.org/abs/2609.28747](https://arxiv.org/abs/2609.28747)

    该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。

    

    语言模型对其答案的置信度通常被解读为模型对相应事实掌握程度的代理指标。本手册记录了一个旨在直接检验这一解读的开源工具包：一个小型因果语言模型在一个语料库上进行微调，该语料库对81个一位数加法组合中的每一个都一致地断言一个虚构的算术答案，随后将模型微调后对每个虚构答案的置信度，与其微调前对相应真实答案的置信度进行配对比较，整个过程中使用完全相同的测量程序。我们描述并论证了流程的每个阶段——事实空间生成、考虑token长度的置信度测量、基线验证、语料库构建、微调以及微调前后的配对比较——以及每个阶段旨在排除的混杂因素，其中包括一位数与两位数答案之间的分词不对称性，以及答案仅仅失去相对优势与……

    arXiv:2609.28747v1 Announce Type: cross  Abstract: A language model's confidence in an answer is often read as a proxy for how well it knows the corresponding fact. This manual documents an open toolkit built to test that reading directly: a small causal language model is fine-tuned on a corpus that consistently asserts one fabricated arithmetic answer for each of the 81 single-digit addition pairs, and its post-fine-tuning confidence in each fabricated answer is compared against its own pre-fine-tuning confidence in the corresponding true answer, using an unchanged measurement procedure throughout. We describe and justify every pipeline stage, fact-space generation, token-length-aware confidence measurement, baseline validation, corpus construction, fine-tuning, and paired before/after comparison, together with the confound each is meant to rule out, among them tokenization asymmetry between single- and double-digit answers and the difference between an answer merely losing its edge a
    
[^227]: 强化学习中的策略复杂度、反应时间与有限理性

    Policy Complexity, Reaction Time, and Bounded Rationality in Reinforcement Learning

    [https://arxiv.org/abs/2609.28737](https://arxiv.org/abs/2609.28737)

    本文提出MI-SARSA算法，通过互信息正则化将状态特定的信息成本显式纳入强化学习，实现了策略压缩并能够预测试验层面的反应时间，从而为生物有限理性提供了更合适的计算模型。

    

    生物智能体并非在无限计算能力的条件下进行学习。对人类而言，学习与选择受到感知、注意力和工作记忆等约束的塑造，这些约束限制了有多少状态信息能够指导行为，从而限定了策略复杂度的上限。标准的强化学习模型通常只优化奖励，而不显式地表征这些内部成本，因此作为生物智能的模型并不十分合适。我们推导出MI-SARSA，这是一种在线策略的时序差分算法，它通过学习到的边际动作先验以及对状态偏离该先验的状态特定惩罚，引入了互信息正则化。由此得到一个序贯学习模型：在该模型中，只有当状态信息带来的期望回报收益足以抵消其新增的信息成本时，状态信息才会被有选择地使用。关键在于，控制策略压缩的状态特定信息成本，同时也产生了试验层面的可预测（此处摘要被截断）

    arXiv:2609.28737v1 Announce Type: cross  Abstract: Biological agents do not learn under conditions of unlimited computation. For humans, learning and choice are shaped by constraints on perception, attention, and working memory, which limit how much state information guides behavior and therefore bound policy complexity. Standard reinforcement learning models typically optimize reward without explicitly representing these internal costs, making them less suitable as models of biological intelligence. We derive MI-SARSA, an on-policy temporal-difference algorithm that incorporates mutual-information regularization through a learned marginal action prior and a penalty on state-specific deviations from that prior. This yields a sequential learning model in which state information is used selectively when its expected return benefit justifies the added informational cost. Critically, the same state-specific information cost that governs policy compression also generates trial-level predict
    
[^228]: 空中连续体机械臂操作中气动干扰下末端执行器位置估计的时序学习

    Temporal Learning for End-Effector Position Estimation under Aerodynamic Disturbances in Aerial Continuum Manipulation

    [https://arxiv.org/abs/2609.28716](https://arxiv.org/abs/2609.28716)

    本文提出使用闭合形式连续时间（CfC）神经网络来估计无人机气动干扰下空中连续体机械臂的末端执行器三维位置残差，相比MLP和GRU等方法提升了位置估计的准确性。

    

    本文研究了时序神经网络在空中连续体机械臂（ACM）末端执行器位置估计中的应用，该机械臂在无人机（UAV）引起的气动效应下运行。研究团队在静止（旋翼关闭）和自由悬停条件下，针对连续体机器人（CR）的不同构型和无人机的不同高度采集了实验数据集，提供了有无气动残差的末端执行器位置测量数据。为了建立名义框架，研究评估了采用逐渐丰富的应变基的应变参数化运动学模型，以平衡模型复杂度和预测精度。所选的名义模型随后作为基线，使用闭合形式连续时间神经网络进行三维位置残差估计，并与多层感知机（MLP）和门控循环单元（GRU）进行对比。在未见过的测试实验中

    arXiv:2609.28716v1 Announce Type: cross  Abstract: This paper investigates temporal neural networks for \mbox{end-effector} position \mbox{estimation} of an aerial continuum manipulator (ACM) operating under aerodynamic effects induced by the unmanned aerial vehicle (UAV). An experimental dataset is collected under stationary (\mbox{rotor-off}) and \mbox{free-hovering} conditions across continuum robot (CR) configurations and UAV altitudes, providing \mbox{end-effector} position measurements with and without aerodynamic residuals. To establish a nominal framework, \mbox{strain-parameterized} kinematic models with progressively richer strain bases are evaluated to balance model complexity and prediction accuracy. The selected nominal model then serves as the baseline for 3D position residual estimation using a \mbox{closed-form} \mbox{continuous-time} (CfC) neural network, with a multilayer perceptron (MLP) and a gated recurrent unit (GRU) used for comparison. On unseen test experiments
    
[^229]: 一种用于二元与多类别仇恨言论检测的可解释 DistilBERT-BiLSTM-注意力框架

    An Explainable DistilBERT-BiLSTM-Attention Framework for Binary and Multi-Class Hate Speech Detection

    [https://arxiv.org/abs/2609.28703](https://arxiv.org/abs/2609.28703)

    该研究提出了一种将 DistilBERT 嵌入与 Bi-LSTM 和注意力机制相结合的多层次可解释仇恨言论检测框架，支持二元与多类别分类，并利用 LIME 提升模型决策的透明度与可信度。

    

    社交媒体上的仇恨言论对社会和谐、心理健康和公共安全构成严重威胁，因此及时、准确地检测仇恨言论对内容审核系统至关重要。现有研究大多集中于二元分类，仅在单一数据集上评估其框架，且对模型如何做出决策提供的洞察有限，这限制了其实际应用价值。此外，针对其预测推理可解释性的研究也很少。为应对这些挑战，本研究提出了一种多层次且可解释的仇恨言论检测框架。该模型将 DistilBERT（蒸馏双向编码器表示变换器）嵌入与 Bi-LSTM（双向长短期记忆网络）模型和注意力机制相结合，以同时捕捉文本中的上下文语义和序列依赖关系。为增强信任度和透明度，LIME（局部可解释的模型无关解释方法）（摘要原文至此截断）

    arXiv:2609.28703v1 Announce Type: cross  Abstract: Hate speech on social media poses serious risks to social harmony, mental well-being, and public safety, making its timely and accurate detection essential for content moderation systems. Most existing studies focus on binary classification, evaluated their frameworks on a single dataset, and provide limited insight into how decisions are made, which limits their real-world applicability. In addition, limited work is done on the explainability of their predictive inference. To address these challenges, this study proposes a multilevel and explainable hate speech detection framework. The proposed model integrates DistilBERT (Distilled Bidirectional Encoder Representations from Transformers) embeddings with a Bi-LSTM (Bidirectional Long Short-Term Memory) model, and an attention mechanism to capture both contextual meaning and sequential dependencies in text. To enhance trust and transparency, LIME (Local Interpretable Model-agnostic Exp
    
[^230]: 渐进式技能发现作为工具使用型LLM代理的访问控制：通过角色范围的能力交付实现结构化治理

    Progressive Skill Discovery as Access Control for Tool-Using LLM Agents: Structural Governance through Role-Scoped Capability Delivery

    [https://arxiv.org/abs/2609.28693](https://arxiv.org/abs/2609.28693)

    提出了skilder框架，通过将技能、工具和指令封装为“角色”并经由单一MCP服务器渐进式交付，为工具使用型LLM代理实现确定性的访问控制和结构化治理。

    

    大型语言模型（LLM）代理在接触庞大的企业工具集时难以安全扩展。为代理提供对所有内部工具的访问权限会导致上下文窗口过大、工具选择能力下降以及严重的治理漏洞——因为纯粹在提示词中定义的系统策略仍然只是概率性的建议，而非硬性约束。现有的缓解措施，如多代理领域委托，会分散审计日志，且无法保证跨会话的策略合规性。我们提出了skilder，一个将能力封装为角色的框架：角色是技能、工具和指令的集合，连同约束它们的边界限制。代理从一个最小的角色目录开始，学习任务所需的角色，并通过单个MCP服务器接收每个角色的技能、指令和工具。由于工具仅在已学习的技能内部到达代理，同一个服务器可以确定性地强制执行所学内容的范围。

    arXiv:2609.28693v1 Announce Type: new  Abstract: Large Language Model (LLM) agents struggle to scale safely when exposed to vast enterprise toolsets. Providing an agent with access to every internal tool leads to oversized context windows, degraded tool selection, and severe governance vulnerabilities - as system policies defined purely in prompts remain probabilistic advice rather than hard constraints. Existing mitigations, such as multi-agent domain delegation, decentralize audit logs and fail to guarantee policy compliance across sessions. We introduce skilder, a framework that packages capabilities into roles: bundles of skills, tools, and instructions, together with the limits that bound them. An agent begins with a minimal role catalog, learns the roles a task requires, and receives each role's skills, instructions, and tools through a single MCP server. Because tools reach the agent only inside learned skills, the same server enforces the scope of what was learned deterministic
    
[^231]: 用AI智能体驱动流行病模型：Epydemix智能体框架

    Driving Epidemic Models with AI Agents: the Epydemix Agent Framework

    [https://arxiv.org/abs/2609.28692](https://arxiv.org/abs/2609.28692)

    该论文提出Epydemix智能体框架，通过模型发现、预防性验证、经测试的代码执行和结果可检查性四项能力，使AI智能体仅凭自然语言描述即可完成从场景构建到定量结果、图表与解读的完整流行病建模流程，且全程可审计、可复现。

    

    基于大语言模型的人工智能智能体为科学软件提供了便捷的自然语言接口，但其可靠性并非自动获得。本文介绍了Epydemix智能体框架，这是在Epydemix——一个用于随机区室流行病建模的开源Python库——之上构建的附加层。该框架通过四项功能扩展了该库，以便于与AI智能体交互：可用模型和参数的发现、声明式场景规范的预防性验证、通过经过测试的库代码执行计算，以及结果的可检查性。这些能力使智能体能够处理整个建模过程，从场景的自然语言描述到定量结果、图表和研究发现解读，而无需编写自定义代码。每个步骤读取输入文件并将结果保存在单独的输出包中，使整个过程可审计且可复现。（摘要在此处截断）

    arXiv:2609.28692v1 Announce Type: new  Abstract: Artificial Intelligence agents based on large language models provide convenient natural language interfaces to scientific software, but reliability is not automatic. Here we introduce the Epydemix Agent Framework, an additive layer over Epydemix, an open-source Python library for stochastic compartmental epidemic modeling. The framework extends the library with four capabilities to facilitate interaction with an AI agent: discovery of available models and parameters, preventive validation of a declarative scenario specification, execution through tested library code, and inspectability of results. These capabilities let an agent handle the entire modeling process, from the natural-language description of the scenario to quantitative results, figures, and interpretation of findings without writing custom code. Each step reads input files and saves results in a separate output bundle, making the process auditable and reproducible. First, 
    
[^232]: 超越表面风格：使多轮用户模拟器与行为一致性对齐

    Beyond Surface Style: Aligning Multi-Turn User Simulators with Behavioral Consistency

    [https://arxiv.org/abs/2609.28690](https://arxiv.org/abs/2609.28690)

    TRACER是一个显式建模用户意图演变、并通过“监督微调+多轮强化学习”两阶段训练使模拟行为与真实交互轨迹保持一致的多轮用户模拟器，在真实客服场景中以转化F1大幅超越最强基线。

    

    忠实的用户模拟是大规模构建、评估和改进交互式AI的基础。然而，看似合理的单条回复并不能保证模拟用户重现真实交互中观察到的意图演变和最终结果。我们提出TRACER，一个多轮用户模拟器，它显式建模用户不断演变的意图，并学习使模拟行为与真实交互轨迹保持一致。TRACER采用两阶段训练：首先在真实用户对话上进行监督微调，随后进行多轮强化学习。强化学习阶段将分层的结果级与轨迹级奖励同偏差感知的优势调制相结合，共同缓解长对话中的奖励稀疏和信用分配难题。在组织成参考群组的真实客服会话上，TRACER-7B以11.4的转化F1超越最强基线，同时取得了最低的群体级转化率误差和语义……（摘要原文截断）

    arXiv:2609.28690v1 Announce Type: new  Abstract: Faithful user simulation is fundamental to building, evaluating, and improving interactive AI at scale. However, plausible individual responses do not ensure that simulated users reproduce the intent evolution and outcomes observed in real interactions. We propose TRACER, a multi-turn user simulator that explicitly models users' evolving intent and learns to align simulated behavior with real interaction trajectories. TRACER is trained in two stages: supervised fine-tuning on real user dialogues, followed by multi-turn reinforcement learning. The RL stage combines hierarchical outcome- and trajectory-level rewards with deviation-aware advantage modulation, jointly mitigating reward sparsity and credit assignment in long dialogues. On real customer-service sessions organized into reference cohorts, TRACER-7B surpasses the strongest baseline by 11.4 conversion F1, while also achieving the lowest group-level conversion-rate error and semant
    
[^233]: 超越静态图世界模型：在演化拓扑上学习随机潜在动力学

    Beyond Static Graph World Models: Learning Stochastic Latent Dynamics over Evolving Topologies

    [https://arxiv.org/abs/2609.28670](https://arxiv.org/abs/2609.28670)

    提出图动力学模型（GDM），利用稀疏循环邻接矩阵和循环状态空间架构，实现对随机、部分可观测环境中演化拓扑图结构观测的世界建模，并首次引入联合图状态分布的评估方法。

    

    基于图的世界模型最近成为一种在关系状态表示上学习状态转移的方法。然而，现有方法大多局限于固定拓扑的图，或确定性的、完全可观测的环境。我们提出了图动力学模型，这是一个面向图结构观测的世界模型，旨在处理更一般的场景：在随机且部分可观测的环境中拓扑不断演化的情况。GDM使用一个稀疏的循环邻接矩阵来建模拓扑更新并执行消息传递，同时采用循环状态空间架构来建模随机状态转移。此外，我们发现图世界模型在评估方面存在空白，因为现有方法没有提供一种手段来比较预测分布与真实分布在联合图状态（包括相互依赖的拓扑、节点特征和图特征）上的差异。因此，我们引入了一种（摘要在此处不完整）

    arXiv:2609.28670v1 Announce Type: cross  Abstract: Graph-based world models have recently emerged as a means of learning transitions over relational state representations. However, existing approaches are largely limited to fixed-topology graphs or deterministic, fully observable environments. We propose the Graph Dynamics Model (GDM), a world model for graph-structured observations that is designed to handle the more general setting of evolving topologies in stochastic and partially observable environments. The GDM uses a sparse recurrent adjacency matrix to model topology updates and perform message passing, together with a recurrent state-space architecture for modelling stochastic transitions. Furthermore, we identify a gap in the evaluation of graph-based world models, as existing methods do not provide a means of comparing predicted and true distributions over the joint graph state comprising the interdependent topology, node features, and graph features. We therefore introduce t
    
[^234]: 在世界模型中训练客体永久性

    Training Object Permanence in World Models

    [https://arxiv.org/abs/2609.28654](https://arxiv.org/abs/2609.28654)

    提出WROP数据基础设施——包含150个受认知科学启发设计的任务、150万样本训练语料库和300道题评测考试，用于评测和训练视频生成模型是否具备客体永久性这一人类核心认知先验。

    

    客体永久性和坚固性是人类认知先验的标志性特征。最近的研究表明，视频生成模型——当前世界模型的典型代表——已经开始展现出涌现的推理能力，使其成为构建类人物理智能的理想候选。那么，视频模型是否已经涌现出客体永久性？如果没有，我们能否用受核心认知启发的数据集来训练它们？我们提出了WROP（World Reasoning with Object Permanence，基于客体永久性的世界推理），这是一个包含150个手工设计的、受认知科学启发的任务的数据基础设施，划分为六个认知类别。我们构建了Blender生成器，在保留每个任务认知结构的同时随机化速度、光照、相机角度等干扰参数，每个任务可产生10,000以上的样本。我们发布了150万样本的训练语料库和一份300道题的考试。在该考试上，我们评估了14个视频模型：3个参考图生视频模型、7个编辑模型和4个续写生成模型……

    arXiv:2609.28654v1 Announce Type: new  Abstract: Object permanence and solidity are hallmarks of human cognitive priors. Recent studies show that video generation models, a paradigmatic class of current world models, have begun to show emerged reasoning abilities, making them ideal candidates for building human-like physical intelligence. Do video models have emerged object permanence in them? If not, could we train them with a core-cognition inspired dataset? We introduce WROP (World Reasoning with Object Permanence), a data infrastructure of 150 hand-designed cognitive science inspired tasks, divided into six cognitive categories. We build Blender generators that randomize speed, lighting, camera angle, and other nuisance parameters while preserving each task's cognitive structure, yielding 10,000+ samples per task. We release a 1.5M-sample training corpus and a 300-question exam. On this exam we evaluate 14 video models: 3 reference-to-video, 7 edit, and 4 continuation, among which 
    
[^235]: 查询远征队：学习检索动作

    The Fellowship of the Query: Learning Retrieval Actions

    [https://arxiv.org/abs/2609.28653](https://arxiv.org/abs/2609.28653)

    通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。

    

    检索增强式问答需要对何时分解问题、搜索、重新表述、提取证据、综合事实、验证进度以及何时停止等控制决策。我们研究轨迹微调能否提升小型语言模型（SLM）作为“下一步动作控制器”的表现。我们还额外评估了一种低资源设置，即由单个SLM同时充当控制器和最终答案生成器。基于被采纳的教师搜索轨迹，我们构建了一个七分类动作预测任务，模型从当前轨迹状态预测下一个结构化的教师动作，并在多种SLM和超小型语言模型（xSLM）上评估了LoRA监督微调作为控制器的效果。在1,646个留出的动作示例上，基于13,194个动作训练的Granite 4.1 3B达到了0.6536的宏F1分数，而同一模型的零样本提示仅为0.1736，TF-IDF逻辑回归基线为0.5399。在端到端的控制器/生成器交换评估……（摘要在此处截断）

    arXiv:2609.28653v1 Announce Type: cross  Abstract: Retrieval-augmented question answering requires control decisions about when to decompose a question, search, reformulate, extract evidence, synthesize facts, verify progress, and stop. We study whether trajectory fine-tuning can improve small language models (SLMs) as next-action controllers. We additionally evaluate a low-resource setting in which a single SLM serves as both the controller and the final-answer generator. From accepted teacher search traces, we build a seven-way action-prediction task, where the model predicts the next structured teacher action from the current trajectory state, and evaluate LoRA-supervised fine-tuning across SLMs and xSLMs as controllers. On 1,646 held-out action examples, Granite 4.1 3B trained on 13,194 actions reaches macro-F1 0.6536, compared with 0.1736 for zero-shot prompting of the same model and 0.5399 for a TF-IDF logistic-regression baseline. In an end-to-end controller/generator swap evalu
    
[^236]: 决策劫持：针对Jev类型化概率决策的提示注入攻击

    Decision Hijacking: Prompt Injection Attacks on Jev's Typed Probabilistic Decisions

    [https://arxiv.org/abs/2609.28613](https://arxiv.org/abs/2609.28613)

    该研究发现具有模式定义输出的非生成式决策模型Jev虽很少被提示注入攻击完全劫持决策，但恶意内容仍能改变动作概率，且自适应攻击可将攻击成功率翻倍，表明模式定义输出能缓解但不消除提示注入风险。

    

    大多数关于提示注入的研究集中在生成式智能体上，其对具有模式定义输出的模型的影响尚不清楚。我们在Jev（一个非生成式决策模型）中研究了这些影响，使用了510个重建的InjecAgent案例。恶意内容会改变动作概率，但很少导致Jev选择攻击者的目标。覆盖标记会减少这种影响，而关于上下文相关性的声明影响较小。使用分数反馈的自适应攻击使优化过程中发现的平均最高攻击者目标概率翻倍，而在新验证调用上的成功率从1.8%上升到3.5%。探索性分析将这些成功案例与较小的初始决策边际或攻击者对观测内容的更大控制联系起来。总之，这些发现表明模式定义的输出会改变但不会消除提示注入风险，凸显了评估不受信任内容如何在允许选择范围内影响决策的必要性。

    arXiv:2609.28613v1 Announce Type: cross  Abstract: Most studies of prompt injection focus on generative agents, leaving their effects on models with schema-defined outputs unclear. We examine these effects in Jev, a non-generative decision model, using 510 reconstructed InjecAgent cases. Malicious content shifts action probabilities but rarely causes Jev to select the attacker's target. Override markers reduce this influence, while claims of contextual relatedness have small effects. Adaptive attacks using score feedback double the mean highest attacker-target probability found during optimization, while success on fresh validation calls rises from 1.8% to 3.5%. Exploratory analysis links these successes to small initial decision margins or greater attacker control over the observation. Together, these findings show that schema-defined outputs change but do not eliminate prompt-injection risk, highlighting the need to evaluate how untrusted content influences choices within the allowed
    
[^237]: 面向进化式角色扮演智能体的对抗性闭环课程

    Adversarial Closed-Loop Curriculum for Evolving Role-Playing Agents

    [https://arxiv.org/abs/2609.28609](https://arxiv.org/abs/2609.28609)

    提出AdvRole对抗性上下文重写框架，通过Actor与Rewriter的对抗交替训练，使场景池随智能体能力共同进化并动态生成针对性困难场景，形成闭环课程以解决固定场景池带来的分布瓶颈问题。

    

    基于大语言模型的角色扮演智能体已被广泛应用于个性化助手和社交模拟等领域。近期的强化学习方法通常在训练开始前收集好的固定场景池上进行训练。这带来了一个分布瓶颈：随着智能体能力的提升，其表现不佳的场景也随之改变，而训练分布却保持静态。因此，我们提出了AdvRole，一个对抗性上下文重写框架，它将角色扮演强化学习转变为闭环课程。AdvRole在学习角色扮演的Actor（执行者）与将角色设定和对话上下文改写为针对该Actor的困难场景的Rewriter（改写者）之间交替进行。Rewriter通过性能差距奖励进行训练，该奖励偏好那些相比原始场景能降低当前Actor得分的改写结果。由此，场景池随Actor共同演化，并持续针对其尚未掌握的领域进行训练。

    arXiv:2609.28609v1 Announce Type: new  Abstract: Role-playing agents based on large language models have been widely applied in areas such as personalized assistance and social simulation. Recent RL methods typically train on a fixed scenario pool collected before learning begins. This creates a distributional bottleneck: as the agent improves, the scenarios where it performs poorly also change, while the training distribution remains static. Therefore, we propose AdvRole, an adversarial context rewriting framework that turns role-playing RL into a closed-loop curriculum. AdvRole alternates between an Actor that learns to role-play and a Rewriter that edits character profiles and dialogue contexts into actor-specific hard scenarios. The Rewriter is trained with a performance-gap reward, which favors rewrites that reduce the current Actor's score relative to the original scenario. As a result, the scenario pool evolves with the Actor and continuously targets under-mastered regions of th
    
[^238]: UO-FIE：结合精确标签监督与分级效用的事实性推断

    UO-FIE: Combining Exact-Label Supervision with Graded Utility for Factivity Inference

    [https://arxiv.org/abs/2609.28605](https://arxiv.org/abs/2609.28605)

    UO-FIE是一种参数高效的事实性推断系统，通过结合硬标签监督、基于效用的软目标、计划性类别权重和序数损失，在类别高度不平衡的中文事实性推断任务中同时提升预测的精确匹配率和区间接近度。

    

    2026年事实性推断评估（FIE2026）将中文语境-假设对划分为九个有序的事实性区间。其评估指标既奖励精确预测，也奖励接近正确区间的预测，而566个训练样本中有64.1%属于单一类别。在初步实验中，多个mDeBERTa分类模型主要倾向于预测主导类别，而Huber回归基线则产生更多接近正确区间的预测，但精确匹配的数量较少。我们提出了面向效用的事实性推断（UO-FIE），这是一种参数高效的系统，将精确标签监督与分级效用相结合。UO-FIE预测九个类别上的分布，并结合了硬标签监督、基于效用的软目标、计划性类别权重和序数损失。我们在受控比较中评估了期望效用解码，并使用了在折外预测上选择的序数校准。

    arXiv:2609.28605v1 Announce Type: cross  Abstract: The Factivity Inference Evaluation 2026 (FIE2026) classifies Chinese context-hypothesis pairs into nine ordered factivity intervals. Its evaluation metric rewards both exact predictions and proximity to the correct interval, while 64.1% of the 566 training examples belong to a single class. In preliminary experiments, several mDeBERTa classification models predominantly predict the dominant class, whereas a Huber-regression baseline produces more predictions near the correct interval but fewer exact matches.   We introduce Utility-Oriented Factivity Inference (UO-FIE), a parameter-efficient system that combines exact-label supervision with graded utility. UO-FIE predicts a distribution over the nine classes and combines hard-label supervision, utility-based soft targets, scheduled class weights, and an ordinal loss. We evaluate expected-utility decoding in controlled comparisons and use ordinal calibration selected on out-of-fold predi
    
[^239]: 学习发现有趣的数学

    Learning to Discover Interesting Mathematics

    [https://arxiv.org/abs/2609.28603](https://arxiv.org/abs/2609.28603)

    该论文提出将定理的内在趣味性定义为证明长度与陈述长度之比，证明该指标与定理的下游效用强相关，并训练了一个能准确预测证明难度的27B模型，据此优化可生成更有趣的数学定理。

    

    arXiv:2609.28603v1 公告类型：cross 摘要：近年来，大型语言模型（LLM）解决高级数学问题的能力日益增强，其中包括许多悬置数十年的开放性难题。这为以前所未有的规模扩展数学知识打开了大门。然而，尽管LLM或许能够猜想并证明越来越多的定理，这些新的数学知识是否有趣或有用仍是一个悬而未决的问题。我们将定理的内在趣味性定义为其证明长度与陈述长度之比，并证明该指标与定理下游效用的外在度量高度相关。我们确定了“在给定一组前提条件下证明的难度”作为计算这些指标的有用基础要素，并训练了一个27B参数的模型，其预测证明难度的准确度超过了前沿的通用模型。针对我们的指标进行优化后，所得到的模型能够产出更有趣的定理……

    arXiv:2609.28603v1 Announce Type: cross  Abstract: Recently, Large Language Models (LLMs) have been increasingly able to solve advanced mathematical problems, including many that have been open for decades. This opens the door to expansion of mathematical knowledge at unprecedented scale. Yet, while LLMs may be able to conjecture and prove more and more theorems, it remains open whether this new mathematical knowledge is interesting or useful. We define intrinsic interestingness of a theorem as the ratio between the length of its proof and the length of its statement. We show that this correlates strongly with an extrinsic measure of the downstream utility of a theorem. We identify the difficulty of a proof conditioned on a set of premises as a useful primitive for computing these metrics, and train a 27B model that predicts proof difficulty more accurately than frontier general-purpose models. Optimizing for our metric creates a model capable of producing more interesting theorems, wh
    
[^240]: NumericJev：基于多路决策树的类Jev大语言模型数值解码

    NumericJev: Jev-like LLM Numerical Decoding with Multiway Decision Trees

    [https://arxiv.org/abs/2609.28587](https://arxiv.org/abs/2609.28587)

    提出了一种无需训练的数值解码算法NUMERICJEV，通过多路决策树递归细化数值范围，使任何具有类Jev结构化选择接口的大语言模型都能输出数值，其性能甚至超过从包含正确答案的候选列表中直接选择。

    

    大语言模型能够理解自然语言，但做出稳健的决策仍然具有挑战性。类Jev模型可以暴露结构化的选项，但这些接口无法直接以所要求的精度提供数值。我们提出了NUMERICJEV，这是一种无需训练的数值解码算法，能够使任何具有类Jev结构化选择接口的大语言模型输出数值。令人惊讶的是，在我们的算术基准测试中，它的表现比从包含正确答案的候选列表中直接选择高出2.93个百分点（图1）。我们的研究动机来自这样一个观察：数值范围选择本身就是一个类Jev大语言模型能够解决的决策问题。NUMERICJEV通过多路决策树递归地细化数值范围，同时在上下文中保留原始问题，且无需参数更新或访问隐藏状态。在100个值的网格上，十路树只需要两轮决策即可完成。

    arXiv:2609.28587v1 Announce Type: cross  Abstract: Large language models can interpret natural lan- guage, yet robust decisions remain challenging. Jev-like models expose structured choices, but these interfaces do not directly provide numeri- cal values at a requested precision. We propose NUMERICJEV, a training-free numerical decod- ing algorithm that enables numerical output from any LLM with a Jev-like structured-choice in- terface. Surprisingly, on our arithmetic bench- mark, it outperforms direct selection from a can- didate list containing the correct answer by 2.93 percentage points (Figure 1). Our motivation comes from the observation that numerical range selection is itself a decision problem that Jev- like LLMs can address. NUMERICJEV recur- sively refines a range through a multiway deci- sion tree while retaining the original question in context, without parameter updates or hidden- state access. On a 100-value grid, a ten-way tree requires only two decision rounds. Range- 
    
[^241]: 持久可计费状态：工具调用LLM智能体中的拒绝钱包攻击与防御

    Persistent Billable State: Denial-of-Wallet Attacks and Defenses in Tool-Calling LLM Agents

    [https://arxiv.org/abs/2609.28585](https://arxiv.org/abs/2609.28585)

    该论文首次系统研究了工具调用LLM智能体中“持久可计费状态”这一新攻击面，提出六种拒绝钱包攻击向量并构建DOW-BENCH基准，实验表明恶意工具可将受害者的单次会话累计输入计费放大至14,293倍，并探讨了相应的防御方法。

    

    多步工具调用的LLM智能体依赖宿主运行时在多轮对话之间保持状态。当运行时将外部工具的返回结果带入后续模型输入时，服务提供商会再次对其进行计量计费。一个被接纳的恶意或被攻陷的工具因此可以在无需受害者凭证或本地运行时权限的情况下，将不受信任的数据转化为由受害者付费的重复性处理。我们将这种被保留的内容称为“持久可计费状态”，并将宿主关于其是否以及如何进入后续可计费上下文的决策形式化为“持久可计费状态边界”。我们对这一准入后生命周期开展了首次系统性安全研究：推导出六种拒绝钱包攻击向量，并构建了DOW-BENCH——一个在六个模型家族上进行评估的端到端测试框架。在243次执行中，使用遥测数据显示，单次会话的累计输入最大值达到该会话首次调用输入的14,293倍。受控的历史策略重跑实验分离出了原始保留机制的……

    arXiv:2609.28585v1 Announce Type: cross  Abstract: Multi-step tool-calling LLM agents rely on host runtimes to preserve state across turns. When a runtime carries an external tool return into later model inputs, providers meter it again. An admitted malicious or compromised tool can thereby convert untrusted data into recurring victim-billed processing without victim credentials or local runtime privilege. We call retained content persistent billable state and formalize the host's decision over whether and how it enters later billable context as the persistent billable-state boundary.   We present the first systematic security study of this post-admission lifecycle. We derive six denial-of-wallet attack vectors and build DOW-BENCH, an end-to-end harness evaluated across six model families. Across 243 executions, usage telemetry shows that the maximum per-session cumulative input reaches 14,293x the session's first-call input. Controlled history-policy reruns isolate raw retention's con
    
[^242]: SGA：时间序列基础模型中多步预测的不确定性量化

    SGA: Uncertainty Quantification for Multi-Step Forecasting in Time Series Foundation Models

    [https://arxiv.org/abs/2609.28582](https://arxiv.org/abs/2609.28582)

    本文提出SGA方法，通过有向无环图表征预测分支拓扑结构并度量其图复杂度，实现了对时序基础模型多步预测不确定性的有效量化，提升了预测结果的可信度。

    

    arXiv:2609.28582v1 公告类型：cross。摘要：近来时序基础模型的出现显著提升了多步预测的性能，使其能够在较长的未来时间范围内做出准确预测。然而，现有的时序基础模型往往存在显著的固有不确定性，这种不确定性通常表现为在每个时间步衍生出预测分支并向后续步骤扩散；不同的预测分支往往展现出不同的预测表现，从而削弱了时序基础模型预测结果的可信度。在本文中，我们提出了切片-图化-对齐（Slicing-Graphing-Alignment，SGA）方法来量化时序基础模型多步预测的不确定性。所提出的SGA首先利用有向无环图表征所有潜在预测分支的拓扑结构，使图复杂度能够约束多步预测的不确定性，然后通过融合拓扑信息与时序基础模型固有的随机特性来精确度量图复杂度……

    arXiv:2609.28582v1 Announce Type: cross  Abstract: The recent emergence of Time Series Foundation Models (TSFMs) has significantly advanced multi-step forecasting performance, enabling accurate predictions over extended future horizons. However, existing TSFMs often suffer from significantly inherent uncertainty, which typically manifests as derived forecast branches emerging at each time step and spreading to subsequent steps; different forecast branches often exhibit varying forecasting performance, thereby undermining the credibility of TSFM forecasts. In this paper, we propose the Slicing-Graphing-Alignment (SGA) method to quantify the uncertainty of multi-step TSFM forecasts. The proposed SGA first characterizes the topology of all potential forecast branches using a directed acyclic graph, such that the graph complexity bounds the uncertainty of multi-step forecasts, and then precisely measures the graph complexity by integrating both topological information and TSFM-inherent sto
    
[^243]: 可审计性不是单一属性：强化学习中的规则重叠、行为一致性与组合

    Auditability Is Not One Property: Rule Overlap, Behavioural Agreement, and Composition in Reinforcement Learning

    [https://arxiv.org/abs/2609.28581](https://arxiv.org/abs/2609.28581)

    该论文将强化学习策略的可审计性分解为六个可独立检验的谓词，并提出基于符号规则提取与哈希账本的审计协议，同时发现规则集重叠并不代表行为一致，揭示了离散行为规则描述层的严格局限。

    

    强化学习（RL）策略通常以不透明的神经检查点形式发布，而训练日志只能表明某次训练发生过，却无法解释策略究竟学到了什么。我们研究独立训练的策略能否通过可审计的离散行为规则来进行表示和组合。我们将可审计性定义为六个可分别测试的谓词：轨迹完整性、无损编码、规则覆盖、行为一致性、组合质量以及价值模型可靠性。我们的协议使用共享的冻结符号化器、被动规则提取、仅追加的哈希绑定账本、精确的环境重放，以及带有明确盲区回退机制的离线置信度排序仲裁。结果表明，这一描述层存在严格的局限：规则集的重叠并不意味着行为一致性——不同策略可能共享相同的符号规则，但在全新状态上却选择几乎等同于随机猜测的动作。因此，融合后的策略只能在……（原文在此处截断）

    arXiv:2609.28581v1 Announce Type: cross  Abstract: Reinforcement-learning (RL) policies are often distributed as opaque neural checkpoints, while training logs show that a run occurred without explaining what the policy learned. We study whether independently trained policies can be represented and composed through auditable discrete behavioral rules. We define auditability as six separately testable predicates: trace integrity, lossless coding, rule coverage, behavioral agreement, composition quality, and value-model reliability. Our protocol uses a shared frozen symbolizer, passive rule extraction, an append-only hash-bound ledger, exact environment replay, and offline confidence-ranked arbitration with an explicit blind-spot fallback.   The results place strict limits on this description layer. Rule-set overlap does not imply behavioral agreement: policies may share symbolic rules while choosing near-chance-matching actions on fresh states. The fused policy therefore selects among e
    
[^244]: TWIST：一个针对对话记忆中干预质量的拟议基准，附带经人工验证的草稿对齐

    TWIST: A Proposed Benchmark for Intervention Quality in Conversational Memory, with a Human-Validated Draft-Alignment

    [https://arxiv.org/abs/2609.28575](https://arxiv.org/abs/2609.28575)

    TWIST是一个评估对话记忆系统“干预质量”的新基准套件，通过张力检测、草稿审核、信念更新作答、敏感召回治理四个赛道，并借助表面匹配的困难负例防止系统靠一律标记作弊，且基准本身经过双人盲标注等严格人工验证。

    

    长对话记忆基准越来越多地测试召回能力和提示性知识更新，近期工作则研究用户信念的演变与记忆状态。TWIST是一个拟议的基准套件，用于衡量一个互补且此前未被测量的属性：干预质量——即已部署的记忆系统在通过其自身的摄入/召回/审核接口运作时，能否在信念变化点上正确行动。四个赛道分别涵盖：无提示的张力检测、依据对话记录审核外发草稿、以当前信念作答同时保留被取代的历史，以及治理敏感信息的召回。该套件扩展了LoCoMo的语料库和测试框架，为每个检测/阻止指标配以匹配的“不要过度检测”对照：表面匹配的困难负例为误干预付出代价，因此任何赛道都无法通过一律标记来投机取巧。该基准本身首先经过验证：采用独立的、对金标盲测的双人标注与仲裁机制、裁判诱饵校准，以及一个可分离……（原文截断）

    arXiv:2609.28575v1 Announce Type: new  Abstract: Long-conversation memory benchmarks increasingly test recall and prompted knowledge updates, and recent work studies evolving user beliefs and memory state. TWIST is a proposed benchmark suite for a complementary, unmeasured property: intervention quality -- whether a deployed memory system, exercised through its own ingest/recall/vet surface, acts correctly at belief change points. Four tracks cover unprompted tension detection, vetting outgoing drafts against the record, answering with current beliefs while preserving supersession history, and governing sensitive recall. The suite extends LoCoMo's corpora and harness, pairing every detect/block metric with a matched do-not-over-detect control: surface-matched hard negatives price false intervention, so no track can be gamed by flagging everything. The benchmark itself is validated first: independent, gold-blind double annotation with adjudication, judge decoy calibration, and a separab
    
[^245]: 网络智能体的困境之处：多阶段LLM智能体的瓶颈分析

    Where Cyber Agents Struggle: Bottleneck Analysis of Multi-Stage LLM Agents

    [https://arxiv.org/abs/2609.28572](https://arxiv.org/abs/2609.28572)

    本文通过端到端诊断研究揭示多阶段LLM网络智能体在自主攻击中的瓶颈，发现仅靠成功率会掩盖低效与证据误判问题，并提出成本感知评分与LLM-as-a-Judge分析来系统识别规划缺陷。

    

    基于LLM的多阶段网络智能体或许能够完成攻击工作流，但仍然存在脆弱、高成本或依赖对执行证据错误解读的问题。仅凭成功率会掩盖低效性、通过重试实现的适应能力以及对成败的识别水平。我们提出了一项针对自主攻击系统的端到端诊断研究，该系统包含编排器、执行器和验证器LLM，应用于类企业环境的横向移动场景。我们在两个场景和三种模式（专家定义、自搭建脚手架和完全自主）下评估了六个前沿模型。我们评估了验证器的一致性和证据锚定能力；引入了一种基于子任务条件、成本感知的评分机制，用于衡量异常的token消耗、重试次数和运行时间；并采用对比性LLM-as-a-Judge分析来识别规划缺陷，包括工具错位、计划相似、过度规格化、探测不足和恢复能力薄弱。验证器通常具有相关性并以证据为依据……

    arXiv:2609.28572v1 Announce Type: cross  Abstract: Multi-stage LLM-based cyber agents may complete attack workflows while remaining brittle, costly, or reliant on incorrect interpretations of execution evidence. Success rates alone obscure inefficiency, adaptation through retries, and recognition of success or failure. We present an end-to-end diagnostic study of an Autonomous Adversary system with orchestrator, executor, and validator LLMs in enterprise-like lateral-movement scenarios. Six frontier models are evaluated across two scenarios and three modes: expert-defined, self-scaffolded, and fully autonomous. We assess validator consistency and evidence grounding; introduce a subtask-conditioned, cost-aware score for abnormal token use, retries, and runtime; and use comparative LLM-as-a-Judge analysis to identify planning deficiencies, including tool misalignment, plan similarity, over-specification, inadequate probing, and weak recovery. Validators are generally relevant and evidenc
    
[^246]: DEEPO：面向多模态大语言模型幻觉问题的双熵增强策略优化

    DEEPO: Dual-Entropy Enhanced Policy Optimization for Hallucination in MLLMs

    [https://arxiv.org/abs/2609.28570](https://arxiv.org/abs/2609.28570)

    该论文提出DEEPO方法，针对强化学习纠正链中的两个薄弱环节——高语义熵困难查询导致组相对优势归零、以及“自信但错误”的token梯度不可见——通过结合信号方差正则化与梯度预处理的双阶段增强来抑制多模态大语言模型的幻觉问题。

    

    强化学习（RL）被广泛用于提升多模态大语言模型（MLLMs）的推理能力，但其对幻觉问题的抑制效果并不稳定。我们将此归因于从奖励到参数更新的“纠正链”中的两个薄弱环节。在采样层面，困难查询——即那些具有高语义熵的查询——经常产生全体一致的错误样本组，使得组相对优势恰好在幻觉风险最高的地方坍缩为零。在优化层面，“自信但错误”的 token 对梯度不可见：分类策略的期望得分梯度范数会随其分布变尖锐而趋于消失，因此最需要纠正的预测反而获得最弱的更新。我们提出双熵增强策略优化（DEEPO），一种结合信号方差正则化与梯度预处理的双阶段增强方法：语义熵触发的专家前缀在高熵困难查询上注入有依据的后续内容……

    arXiv:2609.28570v1 Announce Type: new  Abstract: Reinforcement learning (RL) is widely used to sharpen reasoning in multimodal large language models (MLLMs), yet its effect on hallucination is uneven. We trace this to two weak points in the \emph{correction chain} from reward to parameter update. At the rollout level, hard queries---those with high semantic entropy---frequently produce unanimously wrong sample groups, collapsing the group-relative   advantage to zero exactly where hallucination risk is highest. At the optimization level, confident-but-wrong tokens are gradient-invisible: a categorical policy's expected score-gradient norm vanishes as its distribution sharpens, so the predictions that most need correction receive the weakest updates. We propose Dual-Entropy Enhanced Policy Optimization (DEEPO), a dual-stage enhancement combining signal   variance regularization with gradient preconditioning: semantic-entropy-triggered expert prefixes inject grounded continuations on hig
    
[^247]: 当解释无法被阅读时：针对从右到左语言的SHAP和LIME渲染的测量与修正

    When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages

    [https://arxiv.org/abs/2609.28565](https://arxiv.org/abs/2609.28565)

    本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。

    

    诸如SHAP和LIME等事后解释方法被广泛用于解释文本分类器，但其可视化主要针对从左到右书写的语言设计。当应用于从右到左（RTL）书写的语言（如乌尔都语、阿拉伯语、波斯语和希伯来语）时，归因值在数学上仍然有效，但视觉呈现却会失效：词元顺序错乱、连体字形断裂、图表布局不符合自然阅读方向。本研究将这一差距视为一个可视化问题，而非解释方法本身的局限。我们提出了SHAP-RTL，一个用于修正SHAP和LIME可视化中阅读方向和文字整形问题的渲染层，并针对每种语言进行字体选择，同时保留原始的归因值、特征排序和模型输出。该方法在乌尔都语、阿拉伯语、希伯来语和波斯语的仇恨及冒犯性语言数据集上进行了评估。

    arXiv:2609.28565v1 Announce Type: cross  Abstract: Post hoc explanation methods such as SHAP and LIME are widely used to interpret text classifiers, but their visualizations are mainly designed for left-to-right languages. When applied to right-to-left (RTL) languages such as Urdu, Arabic, Persian, and Hebrew, the attribution values remain mathematically valid, while their visual presentation fails. Tokens appear out of sequence, connected letterforms break apart, and plot layouts do not follow the natural reading direction. This study addresses this gap as a visualization problem rather than a limitation of the explanation methods themselves. We present SHAP-RTL, a rendering layer that corrects reading direction and script shaping in SHAP and LIME visualizations, with per-language font selection, while preserving the original attribution values, feature ordering, and model outputs. The approach is evaluated on Urdu, Arabic, Hebrew, and Persian hate and offensive-language datasets usin
    
[^248]: 随机大语言模型的推测式评估

    Speculative Evaluation of Stochastic LLMs

    [https://arxiv.org/abs/2609.28560](https://arxiv.org/abs/2609.28560)

    该论文提出了一种基于分层贝叶斯尼曼策略的推测式评估方法（HBN及异步版本HBN-async），通过分层贝叶斯模型估计各任务方差并自适应地分配推演预算，从而在固定预算下最小化随机大语言模型基准评估的方差。

    

    评估一个随机的大语言模型代价高昂：基准测试分数通过随机化的多次推演（rollouts）来估计期望性能，然而均匀的重复采样忽略了任务级推演方差之间的显著差异。我们研究如何在固定的推演预算下最小化基准测试均值估计的方差。我们提出了基于分层贝叶斯尼曼（HBN）策略的推测式评估方法，其中试点规模和阶段权重在事前联合确定。该方法首先运行一个简短的均匀试点，将各任务的成功计数与分层贝叶斯模型相结合，并使用任务级抽样方差的后验期望进行精确的正整数尼曼分配。为了缓解试点同步障碍，HBN-async 根据部分试点反馈推测性地执行后续推演，并保留最终分配所选中的那些推演结果。在六个检查点和18个基准测试组上，我们评估了107个非退化的基准-检查点组合。对于推演预算……

    arXiv:2609.28560v1 Announce Type: cross  Abstract: Evaluating a stochastic large language model is costly: benchmark scores estimate expected performance from randomized rollouts, yet uniform repetition ignores sharp differences in task-level rollout variance. We ask how to minimize the variance of a fixed-benchmark mean under an exact rollout budget. We develop Speculative Evaluation with a Hierarchical Bayesian Neyman (HBN) policy with pilot size and stage weight jointly chosen ex ante. It runs a short uniform pilot, pools per-task success counts with a hierarchical Bayesian model, and uses posterior expectations of task-level sampling variances for exact positive-integer Neyman allocation. To mitigate the pilot synchronization barrier, HBN-async speculatively executes continuations from partial pilot feedback and retains those selected by the final allocation. Across six checkpoints and 18 benchmark groups, we evaluate 107 nondegenerate benchmark-checkpoint profiles. For rollout bud
    
[^249]: 谁在框架背后？通过智能体行为对大语言模型进行指纹识别

    Who Is Behind the Harness? Fingerprinting LLMs through Agentic Behavior

    [https://arxiv.org/abs/2609.28559](https://arxiv.org/abs/2609.28559)

    提出了一种名为LIDAR的主动式黑盒指纹识别方法，通过编码智能体在运行时的决策与行动（如编辑后验证、故障恢复和规范-测试冲突处理等行为）来识别框架背后的大语言模型身份。

    

    大语言模型越来越多地通过编码智能体框架运行，这些框架会检查代码仓库、调用工具并修改文件。因此，替换此类智能体背后的模型可能会改变与安全相关的决策，包括它是否会验证更改或从故障中安全恢复。现有的LLM指纹识别方法主要是从直接文本或token分布推断模型身份。而在编码智能体中，这些信号会受到系统指令、控制器逻辑、工具和执行反馈的调节，限制了它们的迁移能力。我们提出了LIDAR（基于运行时决策与行动的LLM识别），这是一种针对编码智能体执行的主动式黑盒指纹识别方法。三个编码探测对在受控变更下揭示编辑后验证、瞬时故障恢复以及规范与测试冲突的解决行为。LIDAR使用互补的实例级和分布级特征来表示所产生的轨迹，并将其与（原文摘要在此处截断）

    arXiv:2609.28559v1 Announce Type: cross  Abstract: LLMs increasingly operate through coding-agent harnesses that inspect repositories, invoke tools, and modify files. Substituting the model behind such an agent can therefore change security-relevant decisions, including whether it verifies changes or recovers safely from failures. Existing LLM fingerprints largely infer identity from direct text or token distributions. In coding agents, these signals are mediated by system instructions, controller logic, tools, and execution feedback, limiting their transfer.   We present LIDAR (LLM Identification from Decisions and Actions at Runtime), an active black-box fingerprinting method for coding-agent execution. Three coding probe pairs expose post-edit verification, transient-failure recovery, and specification--test conflict resolution under controlled changes. LIDAR represents the resulting trajectories with complementary instance-level and distribution-level features and compares them wit
    
[^250]: BaseCamp——一个用于自动化DNA测序数据流程的智能体AI框架

    BaseCamp --- An Agentic AI Framework for Automating DNA Sequencing Data Pipelines

    [https://arxiv.org/abs/2609.28557](https://arxiv.org/abs/2609.28557)

    本文提出BaseCamp，一个由六个专门AI智能体组成的新型智能体AI框架，用于自动化DNA测序流程中传统上依赖人工的决策层，包括质量阈值选择、临界变异裁定、异常诊断和专家审查分诊。

    

    DNA测序流程涵盖质量控制、序列比对、变异检测和注释，目前可由工作流管理系统可靠地大规模编排成熟的生物信息学工具加以执行。而仍然需要人工完成的是围绕该执行过程的决策层：选择适合样本和平台的质量阈值、裁定临界变异检测结果、诊断异常情况，以及确定哪些发现需要提交专家审查。这些决策重复性强、高度依赖主观判断、在不同操作人员之间缺乏一致性，且往往没有记录存档。本文介绍了BaseCamp，一个用于自动化DNA测序流程决策层的新型智能体AI框架。该框架将整个流程分解为六个专门的AI智能体，分别负责样本接收与质量控制、序列比对、变异检测、注释、跨阶段监控以及报告生成。至关重要的是，BaseCamp智能体并不执行序列分析本身（摘要在此处截断）。

    arXiv:2609.28557v1 Announce Type: new  Abstract: DNA sequencing pipelines, spanning quality control, alignment, variant calling, and annotation, are now reliably executed by workflow management systems that orchestrate established bioinformatics tools at scale. What remains manual is the decision layer surrounding that execution: selecting quality thresholds appropriate to a sample and platform, adjudicating borderline variant calls, diagnosing anomalies, and determining which findings warrant expert review. These decisions are repetitive, judgment-intensive, inconsistent across operators, and frequently undocumented. This paper introduces BaseCamp, a novel agentic AI framework for automating the decision layer of DNA sequencing pipelines. The framework decomposes the pipeline into six specialized AI agents, covering sample intake and quality control, alignment, variant calling, annotation, cross-stage monitoring, and reporting. Critically, BaseCamp agents do not perform sequence analy
    
[^251]: Pistis 技术报告

    Pistis Technical Report

    [https://arxiv.org/abs/2609.28554](https://arxiv.org/abs/2609.28554)

    Pistis 提出了交错蒸馏与强化学习（IDRL）这一新颖后训练范式，在单一训练循环中交替进行在线策略蒸馏与强化学习，据此训练出 27B 和 9B 参数的多模态大语言模型，实现了更有效的知识迁移、更稳定的优化以及更精确的长程智能体轨迹信用分配。

    

    我们介绍 Pistis 模型系列，包括基于 Qwen3.6 和 Qwen3.5 构建的 27B 和 9B 参数多模态大语言模型，该系列通过一个通用且可扩展的后训练框架开发而成。该框架首先通过大规模多模态监督微调（SFT）建立坚实基础。在这一 SFT 基础之上，我们提出了交错蒸馏与强化学习，这是一种新颖的后训练范式，在单一训练循环中紧密集成在线策略蒸馏与强化学习。通过在两个目标之间交替优化，而非单独优化其中之一或将其组合成静态的联合损失，IDRL 实现了更有效的知识迁移、更强的优化稳定性，以及对长程智能体轨迹更精确的信用分配，从而带来更强的性能，同时缓解了常见的能力此消彼长问题。在两个模型规模上……（摘要在此处截断）

    arXiv:2609.28554v1 Announce Type: new  Abstract: We introduce the Pistis model family, comprising 27B- and 9B-parameter multimodal large language models built on Qwen3.6 and Qwen3.5, respectively, and developed through a general and scalable post-training framework. The framework first establishes a strong foundation through large-scale multimodal supervised fine-tuning (SFT). Building on this SFT foundation, we propose Interleaved Distillation and Reinforcement Learning (IDRL), a novel post-training paradigm that tightly integrates on-policy distillation and reinforcement learning within a single training loop. By alternating between the two objectives, rather than optimizing either in isolation or combining them in a static joint loss, IDRL enables more effective knowledge transfer, greater optimization stability, and more precise credit assignment for long-horizon agentic trajectories, leading to stronger performance while mitigating common capability trade-offs. At both model scale
    
[^252]: SMILESGNN：基于SMILES-图交叉注意力融合的可解释临床毒性预测

    SMILESGNN: Interpretable Clinical Toxicity Prediction via SMILES-Graph Cross-Attention Fusion

    [https://arxiv.org/abs/2609.28553](https://arxiv.org/abs/2609.28553)

    该论文提出SMILESGNN多模态架构，通过交叉注意力融合SMILES Transformer与GATv2图编码器，在仅0.4M参数的情况下于ClinTox数据集上取得AUC-ROC 0.987的竞争性性能，同时保留显式图分支以支持基于GNNExplainer的可解释毒性预测。

    

    药物毒性预测对于降低药物研发后期的损耗率至关重要，但由于严重的类别不平衡、基于骨架的泛化问题以及临床对可解释预测的需求，该任务仍然充满挑战。单模态方法——SMILES Transformer或图神经网络——各自捕捉分子结构的互补方面，而仅使用序列的模型无法直接提供基于图的归因解释。我们提出了SMILESGNN，这是一种通过交叉注意力融合SMILES Transformer编码器和GATv2图编码器的多模态架构，并提出了其变体SMILESGNN-PT，该变体使用ChemBERTa-2预训练主干网络。该设计在预测流程中保留了显式的图分支，支持基于GNNExplainer对与毒性预测相关子结构的分析。在ClinTox数据集上，SMILESGNN仅以0.4M参数实现了AUC-ROC 0.987和F1 0.906的性能，与强大的SMILES Transformer相比具有竞争力。

    arXiv:2609.28553v1 Announce Type: cross  Abstract: Drug toxicity prediction is critical for reducing late-stage attrition in drug discovery, yet remains challenging due to severe class imbalance, scaffold-based generalization, and the clinical need for interpretable predictions. Single-modality approaches-SMILES Transformers or graph neural networks capture complementary aspects of molecular structure, while sequence-only models cannot directly provide graph-attributed explanations. We present SMILESGNN, a multimodal architecture that fuses a SMILES Transformer encoder and a GATv2 graph encoder via cross-attention, and SMILESGNN-PT, a variant using a ChemBERTa-2 pretrained backbone. The design retains an explicit graph branch within the predictive pipeline, supporting GNNExplainer-based analysis of substructures associated with toxic predictions. On ClinTox, SMILESGNN achieves AUC-ROC 0.987 and F1 0.906 with only 0.4M parameters, performing competitively with a strong SMILESTransformer
    
[^253]: PAWS：政策驱动的智能体世界模拟

    PAWS: Policy-driven Agentic World Simulation

    [https://arxiv.org/abs/2609.28547](https://arxiv.org/abs/2609.28547)

    PAWS是一个政策驱动的智能体世界模拟数据集，通过将36个经过验证的美国金融经济政策事件、12,727条政策相关新闻记录与65,291个基于来源的利益相关者行动进行时间对齐和多层事件标注，为金融多智能体模拟提供了可回放的、基于历史证据的高质量数据基础。

    

    政策干预会通过公共传播、机构决策和利益相关者响应进行传导，然而现有的金融多智能体模拟数据集很少将这些过程与时间对齐的历史证据相联系。我们推出了PAWS（Policy-driven Agentic World Simulation），一个政策驱动的智能体世界模拟数据集，涵盖36个经过验证的美国金融与经济政策事件、12,727条与政策相关的新闻记录，以及65,291个基于来源的利益相关者行动。每个行动都与其支持性新闻相链接，并通过一个多层事件框架来表示，该框架捕捉其交互模式、金融行动类别及子类型、语义属性，以及与外部分类体系的条件映射。实体被解析为规范化的组织，行动与每日市场回报背景对齐，以支持政策-智能体模拟回放。在2,522个分层抽样的行动样本上，独立的AI评审者和人类评审者在交互模式上达到了89.4%的初始一致率。

    arXiv:2609.28547v1 Announce Type: new  Abstract: Policy interventions propagate through public communication, institutional decisions, and stakeholder responses, yet datasets for financial multi-agent simulation rarely connect these processes to temporally aligned historical evidence. We introduce PAWS, a Policy-driven Agentic World Simulation dataset covering 36 verified U.S. financial and economic policy episodes, 12,727 policy-linked news records, and 65,291 source-grounded stakeholder actions. Each action is linked to its supporting news and represented by a multi-layer event frame capturing its interaction mode, financial-action family and subtype, semantic attributes, and conditional mappings to external taxonomies. Entities are resolved to normalized organizations, and actions are aligned with daily market-return context to support policy-agent simulation replay. On 2,522 stratified action samples, independent AI and human reviewers achieved 89.4% initial agreement on interactio
    
[^254]: CrossScale-GLIO：面向弥漫性胶质瘤的MRI与全切片组织病理学的拓扑保持视觉-语言对齐

    CrossScale-GLIO: Topology-Preserving Vision-Language Alignment of MRI and Whole-Slide Histopathology for Diffuse Glioma

    [https://arxiv.org/abs/2609.28524](https://arxiv.org/abs/2609.28524)

    CrossScale-GLIO提出了一种拓扑保持的多模态对齐框架，将MRI肿瘤栖息地图与组织病理细胞生态位图通过结构感知最优传输并以诊断语言为锚定进行对齐，显著提升了弥漫性胶质瘤分子亚型分类、生物标志物预测及跨模态患者检索性能。

    

    磁共振成像（MRI）和组织病理学在截然不同的尺度上观察同一胶质瘤。我们提出了CrossScale-GLIO，这是一个视觉多模态框架，它将MRI表示为肿瘤栖息地图，将组织学表示为细胞生态位图，然后通过以诊断语言为锚定的结构感知最优传输目标将两者对齐。在配对和外部胶质瘤队列中，CrossScale-GLIO在配对测试中实现了亚型宏观F1为0.789，IDH的AUROC为0.934，1p/19q的AUROC为0.884，MGMT的AUROC为0.802。相比仅使用特征的传输方法，亚型分类提升了2.8个百分点（95% CI：1.2至4.4，调整后p = 0.0019）。双向患者检索的Recall@1值分别达到0.286和0.278，Recall@5值分别达到0.621和0.608。病理学家将81.2%的高质量栖息地-生态位配对评为生物学上合理。删除最高质量的配对会使正确分类的概率降低0.184，而随机删除仅降低0.049。

    arXiv:2609.28524v1 Announce Type: cross  Abstract: Magnetic resonance imaging and histopathology observe the same glioma at radically different scales. We present CrossScale-GLIO, a visual multimodal framework that represents MRI as a tumor-habitat graph and histology as a cell-niche graph, then aligns them with a structure-aware optimal transport objective anchored by diagnostic language. Across paired and external glioma cohorts, CrossScale-GLIO achieved a paired-test subtype macro-F1 of 0.789, IDH AUROC of 0.934, 1p/19q AUROC of 0.884, and MGMT AUROC of 0.802. The subtype gain over feature-only transport was 2.8 percentage points (95% CI: 1.2 to 4.4, adjusted p = 0.0019). Bidirectional patient retrieval reached Recall@1 values of 0.286 and 0.278, and Recall@5 values of 0.621 and 0.608. Pathologists rated 81.2% of high-mass habitat-niche pairs as biologically plausible. Deleting the highest-mass pair reduced correct-class probability by 0.184, compared with 0.049 under random deletio
    
[^255]: 认证的任务条件化主动可观测性

    Certified Task-Conditioned Active Observability

    [https://arxiv.org/abs/2609.28520](https://arxiv.org/abs/2609.28520)

    该论文形式化了“认证的任务条件化主动可观测性复杂度”，即在认证误差与安全弃权保证下识别任务相关状态所需的最小最坏情况期望交互代价，并证明任务预测等价性诱导出唯一的最小充分商空间，使主动可观测性复杂度在其上严格不变。

    

    在对不可观测的物理系统采取行动之前，自主智能体必须确定哪些潜在区分支配下游任务、需要多少次主动干预才能认证这些区分、以及何时应当放弃行动以防止灾难性错误。经典可观测性将状态重构视为一个无条件的二元谓词，当被动观测无法在不施加扰动的情况下打破潜在简并、完全的微观反演代价过高、以及区分与任务无关的自由度浪费交互预算时，该方法便会失效。我们形式化了任务条件化主动可观测性复杂度：即在认证误差与安全弃权保证下，识别任务相关状态所需的最小最坏情况期望交互代价。我们证明任务预测等价性诱导出唯一的最小充分商 $\mathcal{H}/\!\sim_\tau$，使得主动可观测性复杂度在该商上严格保持不变……（摘要原文在此处截断）

    arXiv:2609.28520v1 Announce Type: cross  Abstract: Before acting upon an unobservable physical system, an autonomous agent must determine which latent distinctions govern downstream tasks, how many active interventions are necessary to certify them, and when to abstain to prevent catastrophic errors. Classical observability treats state reconstruction as an unconditioned binary predicate, failing when passive observations cannot break latent degeneracies without perturbation, full microscopic inversion is prohibitively costly, and distinguishing task-irrelevant degrees of freedom wastes interaction budgets. We formalize task-conditioned active observability complexity: the minimum worst-case expected interaction cost required to identify task-relevant states under certified error and safe abstention guarantees. We prove that task-predictive equivalence induces the unique minimal sufficient quotient $\mathcal{H}/\!\sim_\tau$, leaving active observability complexity strictly invariant wh
    
[^256]: TW3Cast：一个面向GIFT-Eval时间序列预测的轻量微调基础模型冻结路由器，其选择完全基于训练集完成

    TW3Cast: A Frozen Router of Lightly Fine-Tuned Foundation Models for Time-Series Forecasting on GIFT-Eval, Selected Entirely on the Training Split

    [https://arxiv.org/abs/2609.28506](https://arxiv.org/abs/2609.28506)

    TW3Cast通过在训练集上预先计算并冻结的路由表，在轻量微调的基础模型专家、分位数混合、基础模型混合和回测锦标赛四种模式中进行选择，在不使用任何智能体或语言模型的情况下，于GIFT-Eval基准的130个系统中以平均MASE排名位列第三。

    

    TW3Cast是一个时间序列预测系统，截至2026年9月14日，按平均MASE排名在GIFT-Eval基准的130个参赛系统中位列第3名。排在它前面的两个条目属于排行榜的智能体类别，即使用智能体或语言模型对预测进行推理、生成或选择的多步骤系统。TW3Cast不运行任何智能体，也不使用任何语言模型。它的选择机制是一张在训练集上一次性计算后即被冻结的表，其专家模型是在这些训练集上经过轻量微调的公开基础模型。对于全部97个数据集、频率和预测时域配置中的每一个，该表提供四种模式之一：专家模式，即对Chronos-2、TiRex或Toto进行LoRA或全参数微调，其训练数据经过显式规则的清洗和增强；包含专家模型的分位数混合模式；基础模型混合模式；或者在从训练集中切分出的回测数据上进行的选拔锦标赛模式。表中每一个决策……

    arXiv:2609.28506v1 Announce Type: new  Abstract: TW3Cast is a time-series forecasting system that reaches position 3 of 130 entries on the GIFT-Eval benchmark by mean MASE rank, as of 2026-09-14. The two entries above it belong to the leaderboard's agentic category, multi-step systems that use agents or language models to reason about, generate or select forecasts. TW3Cast runs no agent and no language model. Its selection is a table computed once on the training split and then frozen, and its experts are public foundation models lightly fine-tuned on those training splits. For each of the 97 dataset, frequency and horizon configurations, the table serves one of four modes: a specialist, which is a LoRA or full fine-tune of Chronos-2, TiRex or Toto whose training data was cleaned and enriched by explicit rules; a quantile blend that contains a specialist; a blend of base models; or a selection tournament played on a backtest carved from the training split. Every decision in the table w
    
[^257]: 科学中的人工智能：早期洞见

    AI in Science: Early Insights

    [https://arxiv.org/abs/2609.28504](https://arxiv.org/abs/2609.28504)

    该论文通过分析1500万次Gemini交互、2600多个专业AI模型和600多名科学家的调查数据，首次提供了科学家使用AI的大规模实证证据，发现AI在科学界已被广泛采用，且LLM与专业模型互为补充而非替代。

    

    科学进步是经济增长与繁荣的关键驱动力。人们对人工智能对科学的影响充满期待，同时也存在担忧，但迄今为止相关数据却很少。我们从三个数据来源提供了关于这一问题的早期洞见：1500万次Gemini交互的样本、一份涵盖各学科的2600多个专业AI模型清单，以及对600多名科学家的调查。我们将这些数据映射到一个新的科学任务分类体系中，以研究科学家如何使用人工智能。研究得出四个主要发现。首先，我们发现AI的采用率和覆盖面非常广泛：科学家比大多数其他职业更多地使用AI。专业AI模型具有广泛的学科覆盖面且被高度引用。接受调查的科学家中有近一半报告每天使用某种形式的AI。其次，我们记录了LLM（通过Gemini使用情况来衡量）与专业模型互为补充的证据——LLM被用于通用分析、编程和稿件准备，而……

    arXiv:2609.28504v1 Announce Type: cross  Abstract: Scientific progress is a key driver of economic growth and prosperity. There is great excitement - but also concerns - about the impacts of AI on science, but so far little data. We provide early insights on this from three data sources: a sample of 15 million Gemini interactions, an inventory of over 2,600 specialized AI models across disciplines, and a survey of over 600 scientists. We map these data to a new taxonomy of scientific tasks to study how scientists are using AI. Four main findings emerge. First, we find broad adoption and coverage: scientists use AI more than most other occupations. Specialized AI models have broad disciplinary coverage and are highly cited. Nearly half of the scientists surveyed report using some form of AI every day. Second, we document evidence that LLMs (proxied through Gemini usage) and specialized models act as complements-- LLMs are used for general analysis, coding, and manuscript preparation, wh
    
[^258]: 具有自适应加权与效率评估的混合变分量子-经典框架

    Hybrid Variational Quantum-Classical Framework with Adaptive Weighting and Efficiency Assessment

    [https://arxiv.org/abs/2609.28491](https://arxiv.org/abs/2609.28491)

    该论文提出了Sim-HVQC混合深度量子神经网络框架，通过将无参数的SimAM自适应加权模块与经典特征提取相结合来保留类别判别信息，首次将变分量子电路扩展应用于多类别分类任务，并通过多种子评估证明了其可复现性、参数效率和可解释性。

    

    混合量子-经典神经网络已成为一种在机器学习中利用量子计算优势、同时缓解当前硬件限制的有前景的方法。本文提出了Sim-HVQC，这是一种混合深度量子神经网络，它将自适应、无参数的SimAM加权模块与经典特征提取相结合，在编码到变分量子电路（VQC）之前保留类别判别信息。以往的研究仅限于二分类任务。相比之下，所提出的框架在多个多类别数据集（MNIST、KMNIST、Fashion-MNIST和EMNIST）上进行了训练和评估。该框架通过多种子评估、参数分析以及潜在特征/量子特征的检查，进一步展示了其可复现性、参数效率和可解释性。源代码已在 https://github.com/Dilli822/SimAM-HVQC 公开提供。

    arXiv:2609.28491v1 Announce Type: cross  Abstract: Hybrid quantum-classical neural networks have emerged as a promising approach for leveraging quantum computing in machine learning while mitigating current hardware limitations. This paper presents Sim-HVQC, a hybrid Deep Quantum Neural Network that couples an adaptive, parameter-free SimAM weighting module with classical feature extraction to preserve class-discriminative information prior to encoding into a Variational Quantum Circuit (VQC). Previous studies are restricted to binary classification [1] [2] [3] [4] [5]. In contrast, the proposed framework is trained and evaluated on various multi-class datasets(MNIST, KMNIST, Fashion-MNIST, and EMNIST). The framework further demonstrates reproducibility, parameter efficiency, and interpretability through multi-seed evaluation, parameter analysis, and latent/quantum feature inspection. The source code is publicly available at https://github.com/Dilli822/ SimAM-HVQC
    
[^259]: 预测智能体何时应该进行推理？面向可靠性路由的行为压力测试

    When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing

    [https://arxiv.org/abs/2609.28475](https://arxiv.org/abs/2609.28475)

    论文发现预测智能体的机制选择依赖于数据来源，并提出ReliabilityRoute——一种利用历史覆盖率、市场先验可用性等可靠性特征来引导智能体在何时检索、推理或依赖市场先验的结构性干预方法。

    

    预测智能体日益将语言模型推理、检索、集成与校准相结合，但目前仍不清楚在何种情况下应当信任这些行为。我们在ForecastBench风格的二元预测任务上研究这一问题，将检索、推理、依赖市场先验或使用历史类比的选择视为可观测的智能体行为，而非隐藏的实现细节。我们的核心发现是：机制选择依赖于数据来源——结构化类比在某些数据生成过程中占优势，而市场/群体风格及保守基线在其他情况下表现更佳。我们提出了ReliabilityRoute，这是一种结构性干预方法，利用历史覆盖率、市场先验可用性、来源先验锐度、证据强度、证据分歧度和预测时间跨度等可靠性特征来引导预测智能体的行为。一个基于2024年数据拟合的固定规则无需硬编码来源即可与手工分类法紧密匹配……

    arXiv:2609.28475v1 Announce Type: new  Abstract: Forecasting agents increasingly combine language-model reasoning, retrieval, ensembling, and calibration, but it remains unclear when each behavior should be trusted. We study this question on ForecastBench-style binary forecasting tasks, treating the choice to retrieve, reason, defer to a market prior, or use a historical analog as an observable agent behavior rather than a hidden implementation detail. Our central finding is that mechanism choice is source-dependent: structured analogs dominate for some data-generating processes, while market/crowd-style and conservative baselines are better for others. We introduce ReliabilityRoute, a structural intervention that steers forecasting-agent behavior using reliability features such as historical coverage, market-prior availability, source-prior sharpness, evidence strength, evidence disagreement, and horizon. A fixed 2024-fitted rule closely matches a hand taxonomy without hard-coded sour
    
[^260]: 学习可靠推理的成本

    Learning the Cost of Reliable Inference

    [https://arxiv.org/abs/2609.28322](https://arxiv.org/abs/2609.28322)

    该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。

    

    基准测试与路由平台日益成为连接大型语言模型提供商与终端用户的中介。然而，这些平台上的提供商通常采用固定的每token定价方式，使用户无法为其任务获得最具竞争力的价格。在本工作中，我们设计了一个采购平台，其中每个任务的token价格由提供商之间的竞争驱动，使用户能够在保证质量水平的前提下获得有竞争力的价格。为此，该平台通过反向第二价格拍卖依次路由查询，激励模型提供商真实地竞标其服务用户查询的平均成本的最佳估计。在路由查询的过程中，平台学习每个提供商所提供的质量，并逐步将查询路由到满足期望质量阈值的提供商中最具成本竞争力的提供商。为验证我们的设计，我们使用多个模型进行了实验。

    arXiv:2609.28322v1 Announce Type: new  Abstract: Benchmarking and routing platforms increasingly act as intermediaries connecting large language model providers with end-users. However, providers on these platforms typically use a fixed price per token, preventing users from achieving the most competitive price for their tasks. % workloads. In this work, we design a procurement platform where token prices for each task are driven by provider competition, enabling users to secure competitive pricing for guaranteed quality levels. To this end, the platform sequentially routes queries via a reverse second-price auction that incentivizes model providers to truthfully bid their best estimate of the average cost to serve a user's query. As it routes queries, the platform learns the quality offered by each provider and progressively routes queries to the most cost-competitive provider among those meeting a desired quality threshold. To validate our design, we conduct experiments with multiple
    
[^261]: 通过条件流匹配蒸馏实现高效的多任务操作策略

    Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching

    [https://arxiv.org/abs/2609.28107](https://arxiv.org/abs/2609.28107)

    该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。

    

    生成式建模的最新进展近来已被广泛应用于机器人学的策略学习中。特别是，使用专家演示训练的条件流匹配（CFM）在机器人操作基准测试中已被证明优于现有方法。虽然先前的工作主要集中于单任务设置，但我们从多任务的角度来研究这个问题，因为为每个任务训练独立模型的计算成本非常高。多任务策略学习本身也面临一系列挑战：在简单拼接的演示数据集上进行朴素训练，要么需要增加模型容量以适应额外的复杂性，要么会导致性能下降。我们提出通过迁移单任务CFM专家模型所学习到的速度场，将其知识蒸馏到一个共享的多任务策略中。我们将该蒸馏信号与原始的CFM目标相结合，以保持对专家演示数据的保真度。

    arXiv:2609.28107v1 Announce Type: cross  Abstract: Advances in generative modeling have recently been extensively employed in robotics for policy learning. In particular, Conditional Flow Matching (CFM) trained with expert demonstrations has been shown to outperform existing methods on robot manipulation benchmarks. While prior work has mainly focused on single-task settings, we study the problem from a multi-task perspective, as training independent models for each task is computationally expensive. Multi-Task policy learning comes with its own set of challenges, as naively training on a concatenated dataset of demonstrations would either require increased model capacity to accommodate the added complexity or result in drops in performance. We propose to distill knowledge from single-task CFM experts into a shared multi-task policy by transferring their learned velocity fields. We combine this distillation signal with the original CFM objective to retain fidelity to the demonstrations
    
[^262]: LAYERSCOPE：视频与多模态学习表征的逐层刻画

    LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations

    [https://arxiv.org/abs/2609.28086](https://arxiv.org/abs/2609.28086)

    提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。

    

    我们提出了LAYERSCOPE，这是一个无标签的逐层分析框架，旨在刻画模型在视频和多模态场景下学习到的表征。使用最终层或中间层表征来评估下游性能，通常需要大量带标签的数据、重复的任务特定评估以及大量的计算。为了解决这些局限性，LAYERSCOPE利用局部、全局、分布以及基于对应关系的几何度量，在无需任务特定标签的情况下，比较模型内部以及跨模型的逐层表征结构。我们在MVEB/MVEB+的多个任务上评估了七个架构各异的模型，涵盖视频与多模态分类、聚类以及文本到视频检索。我们发现中间层的表征可以优于最终层和模型默认输出。我们还发现没有任何单一的几何度量能够一致地预测下游性能，但注意到

    arXiv:2609.28086v1 Announce Type: cross  Abstract: We propose LAYERSCOPE, a label-free, layerwise framework that aims to characterize a model's learned representations in video and multimodal settings. Evaluating downstream performance using representations from final or intermediate layers typically requires large amounts of labeled data, repeated task-specific evaluations, and substantial computation. To address these limitations, LAYERSCOPE uses local, global, distributional, and correspondence-based geometric metrics to compare layerwise representation structure within and across models without requiring task-specific labels. We evaluate seven architecturally diverse models across video and multimodal classification, clustering, and text-to-video retrieval tasks from MVEB/MVEB+. We find that intermediate-layer representations can outperform final-layer and model-default outputs. We also find that no single geometric metric consistently predicts downstream performance, but note that
    
[^263]: SHRAV：用于物理建模与逆向设计的状态-假设-推理-行动-验证框架

    SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design

    [https://arxiv.org/abs/2609.27621](https://arxiv.org/abs/2609.27621)

    提出了SHRAV框架，通过带声明复用边界的状态延续核心实现可复用计算，统一支持物理建模与逆向设计，并在计算光刻中仅用四次固定权重更新就将空间图像交并比从0.5313提升至0.8153。

    

    物理建模与逆向设计需要能够从可复用状态继续进行的计算。我们提出了SHRAV，一个围绕状态、假设、推理、行动和验证组织起来的、与具体架构无关的计算框架。其核心机制是一个状态延续核心，具有明确声明的复用边界，并为学习演化和数值量分配了明确的角色。前向配置演化预测性状态并读取物理响应；逆向设计配置则额外生成面向目标的修改并消耗评估器反馈。电磁世界模型研究被映射到前向配置，本文报告了选定的读取与复用诊断结果。计算光刻展示了一个逆向设计配置：在独立的标量光瞳重放条件下，四次固定权重的设计更新将阈值化空间图像的交并比从0.5313提升至0.8153，最大绝对……

    arXiv:2609.27621v1 Announce Type: new  Abstract: Physical modeling and inverse design require computation that can continue from reusable state. We introduce SHRAV, an architecture-independent computational framework organized around State, Hypothesis, Reason, Action, and Verify. Its central mechanism is a state-continuation core with declared reuse boundaries and explicit roles for learned evolution and numerical quantities. Forward configurations evolve predictive state and read out physical responses; inverse-design configurations additionally generate target-directed modifications and consume evaluator feedback. Electromagnetic world-model studies are mapped to forward configurations, with selected readout and reuse diagnostics reported here. Computational lithography demonstrates an inverse-design configuration: four fixed-weight design updates improve thresholded aerial-image intersection-over-union from 0.5313 to 0.8153 under independent scalar-pupil replay, with maximum absolut
    
[^264]: 大语言模型中的数学推理按解题方法而非主题组织

    Math Reasoning in LLMs is Organized by Approach, Not Topic

    [https://arxiv.org/abs/2609.27041](https://arxiv.org/abs/2609.27041)

    该论文通过生成-回放协议提取激活重要性签名并进行无监督聚类，证明大语言模型的内部数学推理是按可复用的解题方法而非数学主题来组织的。

    

    数学推理基准通常按主题进行组织，但语言模型可能是按照可复用的推理方法来组织其内部计算的。本文研究了开放数学能力大语言模型究竟是按主题子技能还是按推理方法来组织内部计算，我们提供的证据表明推理方法是关键因素。我们引入了一种生成-回放协议：模型首先生成一个解答，随后我们回放完全相同的提示加生成轨迹，并提取推理词元上的激活重要性签名。我们在八个模型和五个数学推理数据源上对这些签名进行无监督聚类，然后通过结构、语义和干预测试来评估恢复出的结构。在全部40个模型-数据源组合中，恢复出的聚类均优于同等规模的随机基线。两个独立的前沿大语言模型评审在77-82%的真实聚类中发现了方法层面的连贯性。

    arXiv:2609.27041v1 Announce Type: new  Abstract: Mathematical reasoning benchmarks are typically organized by topic, but language models may organize their internal computation by reusable reasoning approach instead. In this paper, we investigate whether open math-capable LLMs organize internally by topical sub-skill or by reasoning approach, and we present evidence that the approach is the key. We introduce a generation-replay protocol: a model first generates a solution, after which we replay the exact prompt-plus-generation trajectory and extract activation-importance signatures over the reasoning tokens. We cluster these signatures without supervision across eight models and five mathematical reasoning sources, then evaluate the recovered structure with structural, semantic, and intervention tests. Across all 40 model-source cells, the recovered clusters outperform matched-size random baselines. Two independent frontier-LLM judges find approach-level coherence in 77-82% of real clu
    
[^265]: 网络流量自然可见图表示中网络攻击类别的拓扑特征

    Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic

    [https://arxiv.org/abs/2609.26990](https://arxiv.org/abs/2609.26990)

    本研究证明了不同网络攻击类别在网络流量的自然可见图表示中具有独特且可区分的拓扑特征，利用基于760个图论拓扑描述符的多分支CNN模型实现了96.20%的分类准确率。

    

    基于自然可见图（NVG）的表示方法为捕捉序列网络流量中的结构模式提供了一种有前景的途径。然而，不同网络攻击类别在此类表示中是否展现出独特的拓扑特征，目前仍缺乏充分的理解。本研究基于CSE-CIC-IDS2018数据集，考察了基于NVG的网络流量表示的判别能力与结构特性。研究将76个数值型流量特征在40个观测值构成的重叠帧内独立转换为NVG，并从每个图中提取十种图论度量指标，从而每帧获得760个拓扑描述符。这些表示的判别能力通过采用分层五折交叉验证的多分支卷积神经网络（CNN）进行评估。模型达到了96.20%的平均准确率和马修斯相关系数（原文摘要此处不完整）

    arXiv:2609.26990v1 Announce Type: cross  Abstract: Natural Visibility Graph (NVG)-based representations provide a promising approach for capturing structural patterns in sequential network traffic. However, whether different cyber-attack classes exhibit distinctive topological signatures in such representations remains insufficiently understood. This study investigates the discriminative and structural characteristics of NVG-based network traffic representations using the CSE-CIC-IDS2018 dataset. Seventy-six numerical traffic features were independently transformed into NVGs within overlapping frames of 40 observations, and ten graph-theoretic metrics were extracted from each graph, resulting in 760 topological descriptors per frame. The discriminative capability of these representations was evaluated using a multi-branch convolutional neural network (CNN) with stratified five-fold cross-validation. The model achieved an average accuracy of 96.20% and a Matthews correlation coefficient
    
[^266]: 安全提示：面向用户的实时AI风险感知干预措施

    Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness

    [https://arxiv.org/abs/2609.26865](https://arxiv.org/abs/2609.26865)

    该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。

    

    对话式AI系统可能对其用户构成安全风险，例如幻觉、谄媚、过度自信和拟人化，但这些风险在用户日常使用中难以察觉。我们介绍了Safety Nudges，这是一种基于浏览器的工具，当检测到聊天机器人对话中出现令人担忧的行为时，它会提供轻量级的即时标记。我们通过一项为期两周的实地研究对Safety Nudges进行了评估，该研究涉及45名频繁使用聊天机器人的用户，收集了交互日志、调查问卷以及用户对各条提示的反馈。参与者认为该工具有用、清晰且干扰性最小，几乎所有用户都报告称对潜在AI危害的意识有所提高，尽管我们发现仅凭这种意识提升并不一定能带来可观察到的行为改变。我们的结果表明，面向用户的安全提示可以通过帮助人们在具体情境中批判性地评估AI回应，来补充模型层面的安全防护措施，同时强调了……

    arXiv:2609.26865v1 Announce Type: cross  Abstract: Conversational AI systems can pose safety risks to their users such as hallucination, sycophancy, overconfidence, and anthropomorphism, but these risks are difficult for users to detect during everyday use. We introduce Safety Nudges, a browser-based tool that provides lightweight, in situ flags when concerning behavior is detected in chatbot conversations. We evaluated Safety Nudges in a two-week field study with 45 frequent chatbot users, collecting interaction logs, surveys, and feedback on individual nudges. Participants found the tool useful, clear, and minimally disruptive, with nearly all users reporting an increased awareness of potential AI harms, though we found that this improved awareness alone did not necessarily lead to discernible behavioral changes. Our results suggest that user facing safety nudges can complement model-level safeguards by helping people critically evaluate AI responses in context, while highlighting th
    
[^267]: QUARTET：基于四分支交叉注意力与随机游走轨迹的关系图Transformer增强方法

    QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

    [https://arxiv.org/abs/2609.26855](https://arxiv.org/abs/2609.26855)

    提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。

    

    arXiv:2609.26855v1 公告类型：交叉 摘要：关系深度学习将多表数据库建模为异构时序图，图Transformer目前在RelBench等基准测试上取得了最先进的性能。然而，当前领先的模型RelGT存在两个关键局限：其随机局部采样器生成的子图连接松散，阻碍了消息传递；其全局注意力模块依赖于单一的、基于种子特征的内存，忽略了更广泛的宏观层面动态。为克服这些局限，我们提出了QUARTET，一种表达能力强的图Transformer架构，它在局部子图上应用完全自注意力，同时通过交叉注意力分支来丰富全局上下文。具体而言，QUARTET采用基于近期截断个性化PageRank（PPR）的因果随机游走（CRW）采样器，以提取紧凑、抗枢纽节点干扰且密集连通的局部子图，且不会产生时序信息泄露。与此同时，四分支交叉注意力（摘要在此处截断）

    arXiv:2609.26855v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-atte
    
[^268]: 扩展框架而非上下文：从无策略脚手架到可复用的专家智能体

    Grow the Harness, Not the Context: From Strategy-Free Scaffolds to Reusable Specialist Agents

    [https://arxiv.org/abs/2609.26760](https://arxiv.org/abs/2609.26760)

    该论文提出 Growing Harness 训练范式，通过失败定位、联合修复与成功优先门控，把任务反馈中反复出现的控制逻辑自动沉淀为可复用的可执行代码，让智能体框架本身从任务交互中“生长”出来，而 LLM 只需专注于任务特定的语义推理。

    

    大语言模型（LLM）智能体通常需要处理一系列相关任务，然而标准的智能体框架反复要求模型在每个任务的上下文中重新构建相同的控制决策。我们研究能否转而利用任务反馈，将反复出现的控制逻辑转化为可复用的可执行代码，同时将 LLM 调用保留用于任务特定的语义推理。我们提出 Growing Harness（生长式框架），一种由失败引导的训练范式，它从一个无策略的脚手架中学习智能体框架本身；该脚手架仅暴露固定的模型与工具接口，而不编码任何任务求解控制器。函数级别的执行轨迹将每次失败定位到有界的代码范围内，优化器联合修复一个失败窗口，而以成功为先的保留集门控会回滚损害既有能力的修复序列。被接受的修改不断累积到同一个共享框架中，使其控制结构从任务反馈中自然涌现。在 BrowseComp-Plus 和 WebArena-Verified 上的实验（摘要此处截断）……

    arXiv:2609.26760v1 Announce Type: cross  Abstract: Large language model (LLM) agents often handle streams of related tasks, yet standard harnesses repeatedly ask the model to reconstruct the same control decisions inside each task's context. We study whether task feedback can instead turn recurring control into reusable executable code, while reserving LLM calls for task-specific semantic reasoning. We introduce Growing Harness, a failure-guided training paradigm that learns the agent harness itself from a strategy-free scaffold that exposes fixed model and tool interfaces but encodes no task-solving controller. Function-level execution traces localize each failure to a bounded code surface, an optimizer repairs a window of failures jointly, and a success-first held-out gate rolls back repair sequences that harm prior capability. Accepted edits accumulate in one shared harness, allowing its control structure to emerge from task feedback. Across BrowseComp-Plus and WebArena-Verified wit
    
[^269]: 类型安全并非无错误：受约束的决策头跟随选项名称，而非绑定于其的评分标准

    Type-Safe Is Not Error-Free: A Constrained Decision Head Follows the Option Name, Not the Rubric Bound to It

    [https://arxiv.org/abs/2609.26758](https://arxiv.org/abs/2609.26758)

    类型化决策模型虽在构造上保证输出符合模式，但其决策实际跟随选项名称而非该名称绑定的评分标准含义——仅将选项从 0/1 重命名为 no/yes 就会导致 70.4% 的答案改变、AUC 从 .94 反转为 .23，揭示了此类模型中的系统性决策反转风险。

    

    类型化决策模型是为模型输出被软件直接消费的场景而构建的。它们不生成自由形式的文本，而是在预定义的选项集合上返回一个决策。从构造上看，每个输出都符合所需的模式。然而，这一保证并不能告诉我们模型是否按预期解释了这些选项。我们通过改变选项名称分配给评分标准的方式，研究了 Jev 以及两个具有开放权重的类 Jev 模型。每个选项由一个选项名称和一个定义该选项含义的文本评分标准组成。我们仅改变每个评分标准所分配到的选项名称；问题、状态、评分标准的措辞以及选项名称的集合保持完全不变。在 1200 个带有任务特定评分标准的工作流决策上，将两个选项从 0/1 重命名为 no/yes 会使每百个答案中额外改变 70.4 个（95% 置信区间：[67.6, 73.1]），并将 AUC 从 .94 变为 .23，揭示了决策排序中的系统性反转。

    arXiv:2609.26758v2 Announce Type: replace  Abstract: Typed decision models are built for settings where model outputs are consumed directly by software. Instead of generating free-form text, they return a decision over a predefined set of options. By construction, every output conforms to the required schema. Yet this guarantee does not tell us whether the model interprets the options as intended. We study Jev and two Jev-like models with open weights by changing how option names are assigned to rubrics. Each option consists of an option name and a textual rubric that defines what the option means. We change only which option name is assigned to each rubric; the question, state, rubric wording, and set of option names remain exactly the same. On 1200 workflow decisions with task-specific rubrics, renaming the two options from 0/1 to no/yes changes 70.4 more answers per hundred (95% CI: [67.6, 73.1]) and shifts AUC from .94 to .23, revealing a systematic reversal in the decision ranking
    
[^270]: 双重边界：智能体何时可以信任其世界模型？

    Dual-Frontier: When Can an Agent Trust Its World Model?

    [https://arxiv.org/abs/2609.26293](https://arxiv.org/abs/2609.26293)

    提出Dual-Frontier学习原则，只有当世界模型引导的决策其预测优势超过经过认证的决策相关世界模型误差界限时才采纳该决策，否则将资源转用于验证世界模型，从而解决了决策失败时无法区分是决策规则出错还是世界模型出错这一根本性归因难题。

    

    学习型世界模型正成为通用智能体的核心组成部分：通过预测动作后果，它们支持规划与决策制定，同时减少对代价高昂的试错的依赖。这种依赖带来了一个根本性的模糊问题：当一个由世界模型引导的决策失败时，仅凭轨迹数据可能无法判断损失究竟是由智能体的决策规则还是由世界模型造成的。我们将这一失败归因问题形式化为回报损失的反事实分解，并证明其各组成部分无法从被动交互中识别出来，即使对于有限时域的规划器也是如此。这一障碍催生了Dual-Frontier（双重边界），这是一种学习原则：只有当世界模型引导决策的预测优势超过一个经过认证的、与决策相关的世界模型误差界限时，才采纳该决策；否则，证据将被分配用于世界模型的验证。动作条件价值界限和闭环扩展保证了不减少……（摘要在此处截断）

    arXiv:2609.26293v2 Announce Type: replace  Abstract: Learned world models are becoming essential to general-purpose agents: by predicting action consequences, they support planning and decision-making while reducing reliance on costly trial and error. This reliance creates a fundamental ambiguity: when a world-model-guided decision fails, the trajectory alone may not reveal whether the agent's decision rule or the world model caused the loss. We formalize this failure-attribution problem as a counterfactual decomposition of return loss and prove that its components are not identifiable from passive interaction, even for finite-horizon planners. This obstruction motivates Dual-Frontier, a learning principle that admits a world-model-guided decision only when its predicted advantage exceeds a certified bound on decision-relevant world-model error; otherwise, evidence is allocated to world-model verification. Action-conditioned value bounds and a closed-loop extension guarantee non-decrea
    
[^271]: 拒绝一切看似安全：在编码提示词评估中恢复良性分支

    Refusing Everything Looks Safe: Restoring the Benign Arm to Encoded-Prompt Evaluation

    [https://arxiv.org/abs/2609.26176](https://arxiv.org/abs/2609.26176)

    论文揭示仅凭有害编码请求的拒绝率评估安全性会产生误导——编码真正摧毁的是模型区分有害与良性请求的“危害差距”（某模型该差距从明文下的+0.82降至编码后的0.00），因此必须恢复良性分支，将同样编码变换后的良性请求纳入评估。

    

    编码提示词攻击的评估几乎完全集中在有害分支上：基准测试发送混淆后的有害请求，并报告模型服从这些请求的频率。高拒绝率被报告为安全性，但这同样可能意味着模型已经无法将该请求与同格式的其他内容区分开来。我们对良性分支施加相同的变换，结果两种情形相去甚远。在涵盖三个基础模型系列和四种后训练方案的四个7-8B模型上，有害同形字编码提示词的拒绝率跨度仅为0.08，而这四个模型对明文形式的相同请求拒绝率跨度为0.57。编码破坏的并非拒绝能力本身，而是“危害差距”：在一个模型上，有害与良性请求拒绝率之间的差距从明文下的+0.82降至编码后的恰好0.00，而仅读取有害分支的基准测试会给该模型与仍保持+0.61差距的模型打出完全相同的分数。（原文摘要在此处截断）

    arXiv:2609.26176v2 Announce Type: replace-cross  Abstract: Encoded-prompt attacks are evaluated almost entirely on their harmful arm: a benchmark sends obfuscated harmful requests and reports how often the model complied. A high refusal rate there is reported as safety, and it is equally consistent with a model that has stopped telling the request apart from anything else in the same format. We run the benign arm through the same transformation, and the two cases are far apart. Across four 7-8B models spanning three base families and four post-training recipes, refusal of harmful homoglyph-encoded prompts spans 0.08 while the same four span 0.57 on the identical requests in plaintext. What the encoding destroys is not refusal but the harm gap: on one model the gap between harmful and benign refusal falls from +0.82 in plaintext to exactly 0.00 under the encoding, and a benchmark reading only the harmful arm scores that model and one retaining a +0.61 gap identically. Running the cell s
    
[^272]: 通用分形自然语言决策图：跨异构领域的实时边缘分诊

    Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains

    [https://arxiv.org/abs/2609.25498](https://arxiv.org/abs/2609.25498)

    该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。

    

    arXiv:2609.25498v1 公告类型：cross 摘要：部署大型语言模型进行运行时操作分诊会带来难以承受的延迟（>100-500 毫秒）、高昂的显存需求（>4-8 GB）以及过多的能量耗散。本文在 Mandelbrot 分形神经合成（Dagli 等，2026）的基础上，提出了通用分形自然语言决策图，通过 werr 机器原生边缘反射运行时和生产级 answerr 平台（https://answerr.me）实现。该引擎完全无需存储权重张量（0 字节显存），通过沿 Mandelbrot 集的混沌边界动态调制 24 字节坐标种子并评估四象限逃逸动力学，来合成确定性决策——noul（布尔型）、choice（类别型）和 score（序数型）。该引擎从生物“系统一”反射弧中汲取灵感，引入了：(i) 带有域投影器 Phi_D 的自动种子路由器，相比线性基线带来 +28.8% 的准确率提升；(ii) Infor……（原文摘要在此处被截断）

    arXiv:2609.25498v1 Announce Type: cross  Abstract: Deploying Large Language Models for runtime operational triage incurs prohibitive latency (>100-500 ms), high VRAM requirements (>4-8 GB), and excessive energy dissipation. Extending Mandelbrot Fractal Neural Synthesis (Dagli et al., 2026), this paper presents the Universal Fractal Natural Language Decision Map, realized via the werr machine-native edge reflex runtime and the production answerr platform (https://answerr.me). Operating entirely without stored weight tensors (0 Bytes VRAM), the engine synthesizes deterministic decisions---noul (Boolean), choice (categorical), and score (ordinal)---by dynamically modulating 24-byte coordinate seeds along the chaotic boundary of the Mandelbrot set and evaluating 4-quadrant escape dynamics. Drawing inspiration from biological System-One reflex arcs, the engine introduces: (i) an Auto-Seed Router with domain projector Phi_D yielding a +28.8% accuracy gain over linear baselines; (ii) an Infor
    
[^273]: Qwen-Audio-3.1-Realtime：迈向可靠的智能体语音交互

    Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction

    [https://arxiv.org/abs/2609.25176](https://arxiv.org/abs/2609.25176)

    Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。

    

    实时语音助手必须能够对不断演变的请求进行推理、执行操作并遵循对话规则。Qwen-Audio-3.1-Realtime 通过“思考”、“行动”以及“说话与协调”三大机制将这些要求整合在一起。思考模块将 Core-Cocktail 监督微调与多模态、多教师在线策略蒸馏（M²-OPD）相结合，以迁移语言能力并发展原生的音频技能。行动模块利用自进化的可执行环境和多粒度 rollout 进行群体相对策略优化（GRPO），教会模型使用工具、解读反馈并完成任务。说话与协调模块则对齐助手何时、如何以及是否说话或行动。我们在音频推理、多语言理解、工具使用、对话行为、全双工交互和安全性等方面进行了评估。与 Qwen-Audio-3.0-Realtime 相比，3.1 在半双工语音到文本适配任务上将整体任务成功率从 78.4% 提升至 82.0%。

    arXiv:2609.25176v1 Announce Type: cross  Abstract: Real-time voice assistants must reason over evolving requests, execute actions, and follow conversational rules. Qwen-Audio-3.1-Realtime brings these requirements together through Think, Act, and Speak and Coordinate. Think combines Core-Cocktail supervised fine-tuning with Multimodality and Multi-Teacher On-Policy Distillation (M$^{2}$-OPD) to transfer language capabilities and develop native audio skills. Act uses self-evolving executable environments and multi-granularity rollouts for Group Relative Policy Optimization (GRPO), teaching the model to use tools, interpret feedback, and complete tasks. Speak and Coordinate aligns how, when, and whether the assistant speaks or acts. We evaluate audio reasoning, multilingual understanding, tool use, conversational behavior, full-duplex interaction, and safety. Compared with Qwen-Audio-3.0-Realtime, 3.1 raises overall task success from 78.4% to 82.0% on our half-duplex speech-to-text adapt
    
[^274]: RRSI：智能体框架的正则化递归自我改进

    RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

    [https://arxiv.org/abs/2609.24972](https://arxiv.org/abs/2609.24972)

    该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。

    

    LLM智能体的能力在很大程度上被其“框架”所放大，即围绕冻结骨干模型的提示词、控制流、工具、记忆和上下文管理。近期的方法通过迭代地提出并选择对智能体框架的组件级编辑，日益将这一过程自动化，实际上在智能体系统层面建立了一种递归自我改进（RSI）的形式。然而，这种递归进化可能因记忆训练任务而过拟合，在分布内基准上表现出巨大收益，但在分布外基准上收益缩小甚至消失。我们提出了智能体框架的正则化递归自我改进（RRSI），通过约束进化候选的提案与选择，将正则化原则融入框架自我改进之中。提案者以时间退火的预算运行，限制候选可以捆绑的编辑数量，并鼓励探索未开发的轨迹……

    arXiv:2609.24972v1 Announce Type: cross  Abstract: An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajector
    
[^275]: DENSE：将智能体轨迹蒸馏为证据支撑的捷径树以实现自我改进

    DENSE: Distilling Agent Trajectories into Evidence-Grounded Shortcut Trees for Self-Refinement

    [https://arxiv.org/abs/2609.21423](https://arxiv.org/abs/2609.21423)

    提出 DENSE 方法，将智能体执行轨迹蒸馏为证据支撑的嵌套捷径树，无需事后结果标签即可生成可复用反馈，用于智能体自我改进，并在 Terminal-Bench 2.1 上取得最高严格通过率。

    

    在线智能体部署会产生大量执行轨迹，而针对特定任务的验证和专家标注难以规模化且成本高昂。我们研究如何将这些轨迹蒸馏为可复用的反馈，而无需事后结果标签，并从中提取关于局部进展、恢复行为和未完成需求的证据。我们提出 DENSE（从嵌套子任务执行中蒸馏证据），它将这些证据组织成证据支撑的嵌套捷径树。DENSE 压缩冗余尝试，利用恢复证据在不同层级间协调问题，总结已完成的分支并展开未解决的分支，将可复用的进展与剩余任务义务相关联。我们提出 REFIT，一种源配对协议，在事后结果盲视条件下比较来自共享初始轨迹的反馈，并重置环境和模型上下文以便对相同任务进行全新尝试。在 Terminal-Bench 2.1 上，DENSE 取得了最高的严格通过率……

    arXiv:2609.21423v1 Announce Type: new  Abstract: Online agent deployments produce abundant execution traces, while task-specific verification and expert annotation are costly to scale. We study how to distill these traces into reusable feedback without post-hoc outcome labels, drawing on their evidence of local progress, recovery, and unfinished requirements. We introduce DENSE (Distilling Evidence from Nested Subtask Executions), which organizes this evidence into evidence-grounded nested shortcut trees. DENSE compresses redundant attempts, reconciles issues across levels using recovery evidence, and summarizes completed branches while expanding unresolved ones, linking reusable progress to remaining obligations. We introduce REFIT, a source-paired protocol comparing feedback from shared initial trajectories under post-hoc outcome blindness, with environments and model contexts reset for fresh attempts at the same tasks. On Terminal-Bench 2.1, DENSE achieves the highest strict pass ra
    
[^276]: SkillAA：归因引导的技能图谱更新，支持针对性验证与回滚

    SkillAA: Attribution-Guided Skill-Graph Updating with Targeted Validation and Rollback

    [https://arxiv.org/abs/2609.20455](https://arxiv.org/abs/2609.20455)

    SkillAA提出了一种统一技能图谱框架，通过溯因归因将失败执行定位到图谱中特定的可编辑对象，仅更新局部结构并利用门控机制筛选变更，从而实现对冻结语言模型技能的精准修复、验证与回滚。

    

    外部技能可以在不更新参数的情况下提供领域程序，但现有方法通常直接根据失败的执行结果编辑技能，缺乏从观察到的失败到可编辑位置的结构化路由；现有技能图谱也未能充分利用语义边界、对象地址和拓扑依赖关系来进行技能检索、针对性更新和范围化验证。我们提出了SkillAA（技能溯因归因），这是一个面向冻结语言模型的结构化技能优化框架。它在统一的图中表示技能的适用性、执行和组合，使同一结构能够支持技能选择、归因引导的修复和更新验证。SkillAA通过对比成功与失败的执行，将候选修复路由到特定的图对象，仅更新所选定的局部结构，并在提交之前使用局部门和大门控机制筛选候选变更。使用gpt-5.6-sol模型，SkillAA达到了8……（原文摘要在此处截断）

    arXiv:2609.20455v1 Announce Type: new  Abstract: External skills provide domain procedures without parameter updates, but existing methods often edit skills directly from failed rollouts without structured routing from an observed failure to an editable location; existing skill graphs also underuse semantic boundaries, object addresses, and topological dependencies for skill retrieval, targeted updating, and scoped validation. We introduce SkillAA (Skill Abductive Attribution), a structured skill-optimization framework for frozen language models. It represents skill applicability, execution, and composition in a unified graph, allowing the same structure to support skill selection, attribution-guided repair, and update validation. SkillAA contrasts successful and failed executions to route candidate repairs to specific graph objects, updates only the selected local structure, and uses Local and Big Gates to screen candidate changes before commitment. With gpt-5.6-sol, SkillAA reaches 8
    
[^277]: 动态广义Gromov-Wasserstein最优传输

    Dynamic Generalized Gromov-Wasserstein Optimal Transport

    [https://arxiv.org/abs/2609.20008](https://arxiv.org/abs/2609.20008)

    该论文提出TP-DATE框架，首次以无模拟方式将Gromov-Wasserstein最优传输动态化，通过路径作用量证明静态与动态二次型最优传输的等价性，并发展行进对流匹配方法，实现空间转录组学中兼顾组织结构保持的连续轨迹重建。

    

    Gromov-Wasserstein最优传输（GW-OT）通过引入结构感知的传输代价扩展了经典最优传输。这对于空间转录组学尤为重要，因为在空间转录组学中，动态重建除了匹配表达模式外，还应保留组织结构。尽管静态形式已被广泛用于此类结构感知对齐，但用于重建连续轨迹的一般动态形式仍然缺失。我们引入了行进对动态对齐与轨迹估计，这是一个以无模拟方式动态推广GW-OT的理论与计算框架。我们通过路径作用量表述了一大类静态和动态二次型最优传输（QOT），并证明了静态与动态的等价性。我们进一步发展了行进对流匹配方法，该方法允许条件路径之间相互作用，并将其交互边缘化为单一向量场（摘要在此处被截断）。

    arXiv:2609.20008v1 Announce Type: cross  Abstract: Gromov--Wasserstein optimal transport (GW-OT) extends classical optimal transport by introducing structure-aware transport cost. This is particularly relevant for spatial transcriptomics, where dynamical reconstruction should preserve tissue structure in addition to matching expression patterns. While static formulations have been widely used for such structure-aware alignment, a general dynamic formulation for reconstructing continuous trajectories is still missing. We introduce Travelling Pair Dynamical Alignment and Trajectory Estimation (TP-DATE), a theoretical and computational framework to generalize GW-OT dynamically in a simulation-free manner. We formulate a broad class of static and dynamic Quadratic-form OT (QOT) through path actions and prove the static dynamic equivalence. We further develop travelling-pair flow matching, which allows interacting conditional paths and marginalizes their interactions into a single vector fi
    
[^278]: MIP* = RE 核心定理的长周期自动形式化

    Long-horizon autoformalization of a core theorem underlying MIP* = RE

    [https://arxiv.org/abs/2609.19814](https://arxiv.org/abs/2609.19814)

    该研究提出 FormalFlow 系统，在人类监督下协调多个 AI 证明代理，仅用 63 天便完成了 MIP* = RE 核心定理的机器验证 Lean 4 形式化证明，生成了 126,367 行全部由 AI 代理编写的代码。

    

    arXiv:2609.19814v1 公告类型：交叉 摘要：里程碑式的数学形式化工作曾需要专家团队耗费数年才能完成。我们提出了 FormalFlow，这是一个在人类监督下协调 AI 证明代理的系统，用于解决长周期形式化过程中的陈述漂移和证明组合问题。该系统借鉴软件工程的原则与实践，使用共享蓝图来指导嵌套的规划、证明和审查循环，并由代理在整个形式化过程中不断强化验证与审查。我们完成了经典低个体度测试的量子可靠性的机器验证 Lean 4 证明，该测试是 MIP* = RE 的核心定理。开发该证明共耗时 63 天；更大的并行度可进一步缩短这一时间。最终代码库包含 126,367 行 Lean 代码，全部由代理生成。该形式化在修正后的假设下纠正了边条件与中间错误，同时保持了已发表的最终误差界。这项工作提供了一个可验证（摘要在此处被截断）

    arXiv:2609.19814v1 Announce Type: cross  Abstract: Landmark mathematical formalizations have taken specialist teams years to complete. We present FormalFlow, a system that coordinates AI proving agents under human supervision to address statement drift and proof composition in long-horizon formalization. Drawing on software engineering principles and practices, it uses a shared blueprint to guide nested planning, proving and review loops. Agents strengthen verification and review throughout formalization. We completed a machine-checked Lean 4 proof of the quantum soundness of the classical low individual-degree test, a core theorem underlying MIP* = RE. Developing the proof took 63 days; greater parallelism could further reduce this time. The final library contains 126,367 lines of Lean code, all generated by agents. The formalization corrects side conditions and intermediate errors while preserving the published final error bound under corrected assumptions. This work provides a verif
    
[^279]: TacSushi：面向灵巧寿司操作的触觉接地世界-动作建模

    TacSushi: Tactile-Grounded World-Action Modeling for Dexterous Sushi Manipulation

    [https://arxiv.org/abs/2609.19613](https://arxiv.org/abs/2609.19613)

    提出TacSushi，一种触觉接地的世界-动作建模策略，通过特征级门控融合指尖触觉并利用失败试验的未来后果预测进行监督，实现了形变、遮挡和不确定接触条件下的灵巧寿司操作。

    

    灵巧的食物操作需要在形变、遮挡和不确定接触条件下的控制。我们提出TacSushi，一种基于触觉接地、基于Cosmos3的世界-动作策略，它在作用于当前观测的同时，从记录的未来后果中学习。骨干网络编码当前的RGB图像、语言和手部状态，特征级门控融合将指尖触觉特征融入动作表示。在训练期间，一个以示范动作块为条件的解码器预测记录的未来视觉观测、任务进度、相对接触风险和触觉摘要；该解码器在部署时被移除。失败的试验提供后果监督，但其动作被排除在模仿学习之外。我们在340次成功和50次失败的真实机器人试验上训练TacSushi，并在600次独立测试中比较六种方法，涵盖三个分布内任务和两个分布外食材变体。为了评估食品质量……

    arXiv:2609.19613v1 Announce Type: cross  Abstract: Dexterous food manipulation requires control under deformation, occlusion, and uncertain contact. We present TacSushi, a tactile-grounded, Cosmos3-based world-action policy that learns from recorded future consequences while acting on current observations. The backbone encodes current RGB, language, and hand state, and feature-wise gated fusion incorporates fingertip tactile features into the action representation. During training, a decoder conditioned on demonstrated action chunks predicts logged future visual observations, task progress, relative contact risk, and tactile summaries; this decoder is removed at deployment. Failed trials provide consequence supervision, but their actions are excluded from imitation. We train TacSushi on 340 successful and 50 failed real-robot trials and compare six methods in 600 separate rollouts across three in-distribution tasks and two out-of-distribution ingredient variants. To assess food quality
    
[^280]: TERN：一种用于疫情预测的带季节参考与在线适应的Delta规则记忆模型

    TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting

    [https://arxiv.org/abs/2609.18407](https://arxiv.org/abs/2609.18407)

    TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。

    

    每周的流感监测数据用于指导疫苗分发和公共卫生警报，但其预测十分困难。每个地区仅提供少数几个季节的数据，疫情波每年在时间和高度上都会发生变化，在疫情波上升期间有帮助的信息在峰值过后反而会产生误导，而上个季节的形态在一年内仍保持参考价值。现有的疫情图模型和通用预测器只读取短固定时间窗口，并同等对待所有历史信息，因此它们既无法利用更早季节的数据，也无法在疫情阶段发生变化时丢弃过时的关联。为了解决这些局限性，我们提出了TERN，这是一个围绕delta规则快速权重记忆构建的预测器：该记忆按通道进行衰减、沿学习到的地址进行擦除，并由局部疫情阶段特征驱动的门控机制控制，同时结合了显式的季节参考和在线适应机制。在三个Cola-GNN流感基准测试上，TERN的表现优于疫情图模型和通用预测器……

    arXiv:2609.18407v1 Announce Type: cross  Abstract: Weekly influenza surveillance counts guide vaccine distribution and public-health alerts, yet they are hard to forecast. Each region offers only a few seasons, waves shift in timing and height every year, and information that helps while a wave grows misleads after its peak, whereas last season's shape stays informative for a year. Existing epidemic graph models and general forecasters read a short fixed window and treat all past information alike, so they neither exploit earlier seasons nor discard stale associations when the epidemic phase changes. To address these limitations, we propose TERN, a forecaster built around a delta-rule fast-weight memory that decays channel-wise and erases along a learned address under gates driven by local epidemic-phase features, combined with an explicit seasonal reference and online adaptation. On three Cola-GNN influenza benchmarks, TERN outperformed epidemic graph models and general forecasters, m
    
[^281]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^282]: AI的特洛伊时刻：为何有的智能体会作弊，而有的会遵循？

    The Troy Moment of AI: Why SomeWill Cheat and SomeWill Follow?

    [https://arxiv.org/abs/2609.15494](https://arxiv.org/abs/2609.15494)

    该研究首次通过ImpossibleBench实验系统揭示，面对不可能完成的任务时，AI智能体是否选择“作弊升级”受同伴是否受罚及声称授权等观察信息的显著影响，而在明确边界规则下智能体则始终不越界。

    

    对2026年7月OpenAI-Hugging Face事件的近期调查引发了关于智能体在任务失败时行为的两个问题：当分配的任务变得不可能完成时，智能体会停止还是升级（采取不当手段）？观察另一个智能体的行为是否会改变这一决策？我们使用七个ImpossibleBench任务，在单独智能体和三智能体两种设置下，结合GPT-5.6 Sol、Claude Fable 5.1和Gemini 3.8 Flash来研究这些问题。每个任务都包含一个真实的软件缺陷，以及一个无法通过行为正确的源代码修改来满足的冲突测试要求。我们在保持任务和代码仓库状态不变的前提下，改变告知智能体关于先前活动的信息，包括未受惩罚的同伴、受到惩罚的同伴，以及声称来自人类委托人的授权。在具有明确授权规则和受限工具的显式边界机制下，智能体从不修改受保护的测试，但表现出显著不同的……

    arXiv:2609.15494v1 Announce Type: new  Abstract: Recent investigations of the July 2026 OpenAI--Hugging Face incident motivate two questions about agent behavior under task failure: when an assigned task becomes impossible, does an agent stop or escalate, and can observing another agent's behavior change that decision? We study these questions using seven ImpossibleBench tasks with GPT-5.6 Sol, Claude Fable 5.1, and Gemini 3.8 Flash in both solo and three-agent settings. Each task contains a genuine software defect together with a conflicting test requirement that cannot be satisfied by a behaviorally correct source-code change. We hold the task and repository state fixed while varying what the agent is told about prior activity, including an unpunished peer, a punished peer, and a claimed authorization from a human principal. Under an explicit-boundary regime with explicit authorization rules and restricted tools, agents never modify protected tests, but exhibit markedly different pol
    
[^283]: 实空间电荷密度的神经网络求解与泛化能力

    Neural-Network Solutions to Real-Space Charge Density and Generalization

    [https://arxiv.org/abs/2609.14906](https://arxiv.org/abs/2609.14906)

    该论文提出了AIDEN（原子相互作用密度等变网络），通过将元素依赖的单中心密度与环境诱导的密度重分布分离，并利用原子中心和边中心的互补张量关联表示后者，实现了实空间电荷密度的高效神经网络求解，为电子结构计算和计算机辅助材料设计提供了深度学习替代方案。

    

    霍恩贝格-科恩定理确立了一个基本原理：基态（GS）电荷密度在原则上包含多电子系统的所有基态信息，因此所有基态可观测量都可以表示为基态电荷密度的泛函。传统的Kohn-Sham密度泛函理论需要以巨大的计算成本迭代求解自洽场方程，这推动了电子结构计算深度学习代理模型的发展，进而加速计算机辅助材料设计。在此，我们提出了AIDEN（原子相互作用密度等变网络），用于求解实空间电荷密度。AIDEN将依赖于元素的单中心密度与环境引起的密度重新分布分离开来，并通过互补的以原子为中心和以边为中心的张量关联来表示后者。通过一个连续低秩高斯（原文摘要在此处中断）……

    arXiv:2609.14906v1 Announce Type: cross  Abstract: The Hohenberg-Kohn theorem establishes that, in principle, the ground state (GS) charge density contains all GS information of a many-electron system, such that all GS observables can be expressed as functionals of the GS charge density. Conventional Kohn-Sham density functional theory requires iterative solution of the self-consistent-field equations at substantial computational cost, motivating the development of deep learning surrogates for electronic structure calculations and, in turn, accelerating computer-aided materials design. Here, we propose \textbf{AIDEN}, an \underline{A}tomic-\underline{I}nteraction \underline{D}ensity \underline{E}quivariant \underline{N}etwork for solving real-space charge density. AIDEN separates the element-dependent one-center density from environment-induced density redistribution and represents the latter through complementary atom- and edge-centered tensor correlations. A continuous low-rank Gauss
    
[^284]: 这个论断有多宽泛？对NLP研究中泛化表述的映射分析

    How broad is that claim? Mapping Generalisation in NLP Research

    [https://arxiv.org/abs/2609.14770](https://arxiv.org/abs/2609.14770)

    该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。

    

    泛化表述在科学交流中十分常见，尽管它们在语义上往往是模糊的。为了帮助检测对泛化表述的过度依赖以及可能对科学发现造成的歪曲，需要一种自动化方法来识别论断并根据其泛化程度进行分类。我们引入了一个全面的科学领域泛化表述分类体系NLPGenX，它根据论断的泛化程度及其在文本中的表述框架对其进行标注。我们通过一个基于大语言模型（LLM）的框架NLPGenA将该分类体系操作化，该框架能够自动将科学文章中的句子分类为5种不同的泛化类别。我们通过人工标注者对该框架进行了验证，并利用该框架构建了一个大规模的NLP论文标注数据集NLPGens，其中包含泛化程度标注以及关于模糊限定词（hedging）和模糊描述词的辅助标签。我们使用NLPGens分析泛化表述的使用情况……（原文摘要在此处截断）

    arXiv:2609.14770v1 Announce Type: cross  Abstract: Generalisations are common in scientific communication, even though they are semantically ambiguous. An automated method is needed to identify and categorise claims according to their level of generalisation, in order help detect an over-reliance on generalisations and possible misrepresentations of scientific findings. We introduce a comprehensive taxonomy of generalisations in the scientific domain, NLPGenX, which labels claims according to their level of generality and framing within the text. We operationalise this taxonomy with an LLM-powered framework, NLPGenA, that automatically classifies sentences from scientific articles into 5 different generalisation classes. We validate our framework with human annotators and use the framework to construct a large-scale dataset of NLP papers annotated according to generality, with auxiliary labels for hedging and vague descriptors (NLPGens). We use NLPGens to analyse the use of generalisat
    
[^285]: 氛围专利撰写：评估用于专业专利起草智能体的大语言模型裁判

    Vibe Patenting: Evaluating LLM Judges for Professional Patent-Drafting Agents

    [https://arxiv.org/abs/2609.13422](https://arxiv.org/abs/2609.13422)

    该论文提出了端到端专利撰写测试平台"Vibe Patenting"，证明LLM裁判的迭代反馈能持续提升AI生成的专利草稿质量，并使低推理低成本智能体接近昂贵的高推理智能体的表现。

    

    大语言模型（LLM）裁判越来越多地被用于评估和改进AI生成的输出，但它们在复杂专业工作中的可靠性仍不明确。我们通过"Vibe Patenting"（氛围专利撰写）来研究这一问题，这是一个用于AI智能体评估的端到端专利撰写测试平台。一个单独调用的LLM裁判会对生成的专利草稿进行评估，并提供结构化反馈用于迭代修订。在多种发明和起草智能体配置下，裁判引导的修订能够持续提升裁判评估的质量，而无引导的修订则往往趋于饱和。值得注意的是，迭代的裁判反馈使低推理能力的智能体能够接近成本高昂得多的高推理智能体的性能。更强的模型和更多的推理通常会提升裁判评估的起草质量，而特定领域的智能体工作流程则能带来进一步提升。我们通过一位专业专利律师的独立评估对该裁判进行了验证，并发现……

    arXiv:2609.13422v1 Announce Type: new  Abstract: LLM judges are increasingly used to evaluate and improve AI-generated outputs, yet their reliability for complex professional work remains unclear. We study this problem through Vibe Patenting, an end-to-end patent-drafting testbed for AI-agent evaluation. A separately-invoked LLM judge evaluates generated patent drafts and provides structured feedback for iterative revision. Across multiple inventions and drafting-agent configurations, judge-guided revision consistently improves judge-assessed quality, while unguided revision tends to saturate. Notably, iterative judge feedback enables a low-reasoning agent to approach the performance of a substantially more expensive high-reasoning agent. Stronger models and increased reasoning generally improve judge-assessed drafting quality, while domain-specific agentic workflows provide further gains. We validate the judge against independent evaluation by a professional patent attorney and find m
    
[^286]: 惯例差距：迈向合作AI评估中隐性交流的度量

    The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation

    [https://arxiv.org/abs/2609.11489](https://arxiv.org/abs/2609.11489)

    该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。

    

    合作型AI智能体通常与其他AI进行对比评估，然而人类的合作依赖于隐性惯例——即解读超出字面信息含义的共享协议——而AI-AI基准测试可能无法捕捉到这一点。我们提出了“惯例差距”这一概念，即从交流字面内容所预测的失败概率与实际观察到的失败率之间的差异，作为衡量隐性交流的指标。在纸牌游戏Hanabi中，有限的牌堆和确定性的提示约束使得该后验概率可以被精确计算。我们重放了来自三个公开数据集的约101,000个出牌动作，涵盖人与人、AI与AI、人与AI的对局。结果显示，差距在人与人组队中为+26.2个百分点，在AI与AI组队中为-0.7个百分点，在人与AI组队中为+16.4个百分点，且该差距集中在未收到任何提示的牌的出牌行为上（人与人组队中为+46个百分点）。在人与AI的对局中，人类可获得的字面信息……

    arXiv:2609.11489v1 Announce Type: new  Abstract: Cooperative AI agents are evaluated against other AIs, yet human cooperation relies on implicit conventions---shared protocols for reading meaning beyond the literal message---which AI-AI benchmarks may not capture. We propose the \emph{convention gap}, the difference between the failure probability predicted from the literal content of communication and the observed failure rate, as a metric of implicit communication. In the card game Hanabi, the finite deck and deterministic hint constraints make this posterior exactly computable. We replayed about 101,000 play actions from three public datasets of human-human (hanab.live), AI-AI (HOAD), and human-AI (HanabiData) games. The gap was +26.2 percentage points (pp) in human pairs, $-$0.7~pp in AI pairs, and +16.4~pp in human-AI pairs, and was concentrated on plays of cards that had received no hints (+46~pp in human pairs). Within human-AI play, the literal information available to humans w
    
[^287]: 同日同故事，隔日异信号：金融情感分析的双重效度

    Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment

    [https://arxiv.org/abs/2609.11144](https://arxiv.org/abs/2609.11144)

    本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。

    

    金融自然语言处理（Financial NLP）领域有一个标准工作流程：先验证情感分析工具与人工标注的一致性，然后信任它来提取市场信号。这背后隐含的假设是，这两种评估衡量的是同一件事。我们在一个可以同时测量两者的场景中检验了这一假设：一个证券集体诉讼语料库（2002-2025年），将70,500条X平台消息与异常股票收益相关联，并包含一个由单一标注员人工标注的金标准样本。通过将五种工具（VADER、Loughran-McDonald、FinBERT、Twitter-RoBERTa和一个LLM标注器）运行在完全相同的流程中，我们发现构念效度与预测效度之间的关系取决于抽样惯例和分数表示方式。在传统的方法特定抽样下，人工一致性评分与同日的分级关联更为吻合，而与提前一天的关联吻合度较低。然而，在固定样本量的面板数据上，一致性评分在两个时间范围内都表现出相似的分级秩相关，而粗粒度排序在两种情况下均较弱。

    arXiv:2609.11144v1 Announce Type: cross  Abstract: Financial NLP has a standard workflow: validate a sentiment tool against human labels, then trust it to extract market signal. This assumes the two evaluations measure the same thing. We test that assumption in a setting where both can be measured at once: a corpus of securities class actions (2002-2025) linking 70,500 X messages to abnormal stock returns, with a single-annotator human labelled gold sample. Running five instruments (VADER, Loughran-McDonald, FinBERT, Twitter-RoBERTa, and an LLM annotator) through one identical pipeline, we find that the relationship between construct and predictive validity depends on the sampling convention and score representation. Under conventional method-specific sampling, human agreement aligns more closely with graded same-day associations than with one-day leads. On a fixed-n panel, however, agreement has similar graded rank correlations at both horizons, while the coarse ordering remains weak.
    
[^288]: 校准是瓶颈：多轮工具调用的动作类别诊断

    Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

    [https://arxiv.org/abs/2609.00949](https://arxiv.org/abs/2609.00949)

    本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。

    

    多轮工具调用是大语言模型（LLM）智能体的一项核心评测场景。在公开的工具调用基准上，开源权重模型的总体准确率已接近甚至超越闭源前沿模型。然而，这一指标是对众多不同多轮情境的取平均，掩盖了进展是否在这些情境之间均衡分布。我们提出一种面向动作类别的诊断框架，将多轮失败分解为两种正交模式：动作类别失准与动作执行失败。该框架在四类动作空间（TOOL_CALL/ASK/REFUSE/CONFIRM）上运行，并引入一个自我揭示的上界 Acc ≤ GAR（黄金动作召回率）；两种失败模式分别表现为上界被违反（Acc > GAR，暴露出状态评分器对失准的掩盖）以及较大的上界余量（GAR >> Acc，将执行失败定位于 TOOL_CALL 内部）。我们在一组工具调用模型上对该框架进行了验证……（原文摘要在此处截断）

    arXiv:2609.00949v1 Announce Type: cross  Abstract: Multi-turn tool calling is a core evaluation scenario for large language model (LLM) agents. On public tool-calling benchmarks, open-weight models now approach or even surpass closed-source frontier models in aggregate accuracy. However, this metric averages over many different multi-turn situations and obscures whether progress is balanced across them. We propose an action-class-oriented diagnostic framework that decomposes multi-turn failures into two orthogonal modes: action-class miscalibration and action-execution failure. The framework operates over a four-class action space (TOOL_CALL/ASK/REFUSE/CONFIRM) and introduces a self-revealing upper bound Acc <= GAR (Gold Action Recall); the two modes show up as bound violation (Acc > GAR, exposing state-grader masking of miscalibration) and large bound slack (GAR >> Acc, localizing execution failure within TOOL_CALL). We validate it on a panel of tool-calling models across multiple mul
    
[^289]: 连接一维集体行为自发机制与场致机制的人机定理

    A Human-AI Theorem Connecting Spontaneous and Field-Induced Mechanisms of Collective Behavior in One Dimension

    [https://arxiv.org/abs/2609.00322](https://arxiv.org/abs/2609.00322)

    本文在人机合作中证明了一个统一统计物理两大基本机制的定理：一维零场O(n)向量链中任意非均匀的最近邻与次近邻竞争相互作用，可通过与温度无关的哈密顿量级映射精确等价于仅有最近邻相互作用加轴向单自旋势的简单链，同时展示了AI能够在人类主动假设空间之外提出关键科学假设的能力。

    

    人工智能（AI）能否在人类合作者的主动假设空间（AHS）之外产生科学假设？能否通过组织人机合作研究使此类突破更有可能发生？我们在证明一个定理的过程中记录了这样一个案例，该定理连接了统计物理学中两种基本的组织机制：零场下由竞争相互作用产生的集体行为，以及由外场诱导或控制的集体行为。对于每个整数 n≥1 和每个系统尺寸 L≥1，具有任意非均匀最近邻和次近邻相互作用函数 U_i(S_i·S_{i+1}) 和 V_i(S_i·S_{i+2}) 的零场 O(n) 向量开链，在微观上通过哈密顿量层面的一个与温度无关的映射，精确等价于一个更简单的 O(n) 开链，后者仅具有最近邻相互作用 V_i(σ_i·σ_{i+1}) 和轴向单自旋势 U_i(σ_i^z)。

    arXiv:2609.00322v1 Announce Type: cross  Abstract: Can an artificial intelligence (AI) generate a scientific hypothesis outside a human collaborator's active hypothesis space (AHS), and can human-AI research be organized to make such breakthroughs more likely? We document such a case while proving a theorem that connects two basic organizing mechanisms of statistical physics: collective behavior arising in zero field from competing interactions and that induced or controlled by an external field. A zero-field $O(n)$-vector open chain with arbitrary inhomogeneous nearest- and next-nearest-neighbor interaction functions $U_i(S_i\cdot{S}_{i+1})$ and $V_i(S_i\cdot{S}_{i+2})$ is microscopically, via a temperature-independent mapping at the Hamiltonian level, equivalent to a simpler $O(n)$ open chain with nearest-neighbor interaction $V_i( \sigma_i\cdot \sigma_{i+1})$ and axial single-spin potential $U_i(\sigma_i^z)$ for every integer $n\ge1$ and every system size $L\ge1$. The homogeneous li
    
[^290]: Aero Hand Open：一款面向灵巧操作学习的仿真就绪腱驱动灵巧手

    Aero Hand Open: A Simulation-Ready Tendon-Driven Hand for Dexterous Manipulation Learning

    [https://arxiv.org/abs/2608.28578](https://arxiv.org/abs/2608.28578)

    提出了Aero Hand Open——一款仿真就绪的腱驱动拟人灵巧手，附带可复现缆绳传动的仿真模型和双向辨识执行映射，解决了腱驱动手在灵巧操作学习中的仿真建模难题。

    

    腱驱动灵巧手具有拟人化结构，而将执行器从关节处移开，正是这类高性能手部能够以低成本制造的关键。这种成本节约来自两方面：通过缆绳传递力，使得电机无需安装在所驱动的关节内部，因此可以使用更小、更便宜的电机；同时一个电机可以通过单根缆绳驱动多个关节，从而减少所需电机的数量。然而，与直驱灵巧手相比，腱驱动手更难以用于学习。产生成本节约的欠驱动传动系统本身在仿真器中就难以建模，而且由同一根缆绳驱动的关节无法被独立控制。我们提出了Aero Hand Open，一款以仿真就绪状态发布的腱驱动拟人灵巧手。该产品附带三项内容：一个能够复现缆绳传动本身的仿真模型；一个经过辨识的执行映射，可在两个方向上将该模型与电机指令相互连接，包括（原文在此处截断）

    arXiv:2608.28578v1 Announce Type: cross  Abstract: Tendon-driven hands are anthropomorphic, and moving the actuators off the joints is what makes a hand of this capability affordable to build. Two effects produce that saving. Routing force through a cable removes the requirement that a motor fit inside the joint it drives, so smaller and cheaper motors suffice, and one motor can drive several joints through a single cable, so fewer motors are needed. They are also harder to learn on than a direct-drive hand. The underactuated transmission that produces the saving is itself difficult to represent in a simulator, and the joints one cable drives are not independently commandable. We present Aero Hand Open, a tendon-driven anthropomorphic hand that is released simulation-ready. Three things ship with it. A simulation model reproduces the cable transmission itself. An identified actuation map connects that model to the motor commands in both directions, including the three-way coupling of t
    
[^291]: J-Zero：从零数据出发的统一挑战者-求解者-评判者协同进化

    J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

    [https://arxiv.org/abs/2608.26582](https://arxiv.org/abs/2608.26582)

    J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。

    

    arXiv:2608.26582v1 公告类型：交叉 摘要：自我进化语言模型最近成为通往超级智能的一条有前景的路径，其优势在于减少人类监督成本。尽管在可验证领域已取得显著进展，但自我进化在不可验证领域仍研究不足。我们提出了从零数据出发的评判者协同适应（J-Zero），这是一个统一的挑战者-求解者-评判者协同进化框架，支持在两种领域中的自我改进。挑战者和求解者通过对抗性互动协同进化：挑战者生成越来越难的任务，而求解者学习产生更高质量的响应。与此同时，评判者通过使用偏好对进行协同适应，这些偏好对的顺序是预先已知的，基于每个响应的生成方式，即求解者的答案优于挑战者的答案，以及其分解再组合的答案优于其一次性答案，而非基于评判者自身的评分。

    arXiv:2608.26582v1 Announce Type: cross  Abstract: Self-evolving language models have recently emerged as a promising path toward superintelligence, with the advantage of reducing the cost of human supervision. While considerable progress has been made in verifiable domains, self-evolution in unverifiable domains remains substantially less explored. We propose Judge co-adaptation from Zero data (J-Zero), a unified Challenger--Solver--Judge co-evolution framework that supports self-improvement across both domains. The Challenger and Solver co-evolve through an adversarial interaction: the Challenger generates increasingly difficult tasks, while the Solver learns to produce higher-quality responses to them. In parallel, the Judge co-adapts using preference pairs whose ordering is known in advance from how each response was produced, i.e., the Solver's answer over the Challenger's, and its decomposed-and-recombined answer over its one-shot answer, rather than from the Judge's own scores. 
    
[^292]: 自我编写的指标：从自身盲点演化出评估器

    Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots

    [https://arxiv.org/abs/2608.18744](https://arxiv.org/abs/2608.18744)

    本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。

    

    arXiv:2608.18744v1 公告类型：新 摘要：智能体在可靠自动指标的引导下能快速进步，而没有指标则会停滞不前；最需要这种指标的应用（如报告生成）恰恰是无人知道如何评分的领域。指标能自我编写吗？说清什么使答案优秀很难，但指出答案的问题则相对容易，因此我们演化的指标是一个小型Python操作符池，每个操作符为一个命名的缺陷标记候选答案，或弃权，并投票。直接让模型生成操作符是行不通的：183个候选仅实现96种不同行为，且来自一个巨大空间中的狭窄区域。EvalCEGAR转而借鉴程序验证中的反例引导抽象细化方法。它将操作符池视为一种抽象，并搜索碰撞——即两个答案在操作符评分下相同，但一个正确一个错误。该配对（而非提示）成为创作请求，当碰撞击败所有尝试时，循环会扩大操作符的定义范围。

    arXiv:2608.18744v1 Announce Type: new  Abstract: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an opera
    
[^293]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^294]: TRACE：面向多轮对抗性对话评估的轨迹感知推理

    TRACE: Trajectory Aware Reasoning for Multi-Turn Adversarial Conversation Evaluation

    [https://arxiv.org/abs/2608.15594](https://arxiv.org/abs/2608.15594)

    本文提出了一种轨迹感知推理的多轮对话防御方法，通过结构化评估操纵线索并动态决策响应方式，在保持安全性的同时减少过度拒绝。

    

    arXiv:2608.15594v1 公告类型：新 摘要：多轮越狱攻击已成为大语言模型（LLMs）的关键安全威胁，因为有害目标被分解为一系列看似良性的对话轮次，以绕过安全防护措施。现有防御缺乏识别不断演变的操纵模式的推理能力，常常通过过度拒绝与敏感话题相关的良性请求，以牺牲有用性来换取安全性。我们引入了Trace，一种具有轨迹感知结构化推理的多轮防御机制。在生成每个响应之前，模型从对话轨迹中识别操纵线索，评估用户意图的良性解释和对抗性解释，分配越狱评分，并决定采取行动：允许、谨慎或拒绝。我们从五种攻击框架中整理了4k个多轮对抗性对话，并将其与2.4k个良性对话以及600个敏感但良性的对话配对。我们使用SFT和GRPO在Llama-3.1-8B-Instruct模型上进行训练，采用一个多组件奖励函数，该函数结合了……

    arXiv:2608.15594v1 Announce Type: new  Abstract: Multi-turn jailbreak attacks have emerged as a critical safety threat to LLMs, as harmful objectives are decomposed across a sequence of apparently benign turns to bypass guardrails. Existing defenses lack the reasoning capacity to identify evolving manipulation patterns, often trading helpfulness for safety by over-refusing benign requests related to sensitive topics. We introduce Trace, a multi-turn defense with trajectory-aware structured reasoning. Before generating each response, the model identifies manipulation cues from the trajectory, evaluates both the benign and adversarial interpretations of user intent, assigns a jailbreak score, and commits to an action: Allow, Caution, or Decline. We curate 4k multi-turn adversarial conversations from five attack frameworks, pair them with 2.4k benign dialogs, and 600 sensitive-but-benign conversations. We train Llama-3.1-8B-Instruct with SFT and GRPO under a multi-component reward that jo
    
[^295]: 将视觉-语言-行动模型进化为具备即时工具使用的智能体

    Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use

    [https://arxiv.org/abs/2608.14047](https://arxiv.org/abs/2608.14047)

    本文提出ART框架，通过将VLA模型与即时工具使用结合，显著降低动作空间复杂性和数据需求，在小型数据集上实现了更高的泛化性和任务成功率。

    

    arXiv:2608.14047v1 公告类型：交叉 摘要：本文通过将端到端的视觉-语言-行动（VLA）模型与智能体工具使用相结合，提出了具备工具使用的智能体机器人（ART）。ART是一种工具注入框架，可调整任何VLA模型以利用现成的工具模块，用于低级视觉、高级功能性和具身增强。与具有完整连续动作解空间的普通VLA模型相比，ART通过工具使用降低了动作解空间的复杂性，这不仅提高了跨任务的泛化能力，还减少了对数据的依赖。为了展示该框架的优势（高泛化性和低数据依赖性），我们首先构建了一个包含30K条工具使用轨迹和动作演示的数据集，该数据集远小于基线方法所使用的数据集。然后，我们设计了一种针对挑战性环境中长轨迹工具使用推理的训练方案。实验表明，ART的成功率提高了20%以上。

    arXiv:2608.14047v1 Announce Type: cross  Abstract: This paper integrates end-to-end Visual-Language-Action (VLA) models with agentic tool-use to propose Agentic Robot with Tool-use (ART). ART is a tool-injection framework that tunes any VLA model to leverage off-the-shelf tool modules for low-level vision, high-level affordance, and embodiment enhancement. Compared to vanilla VLA models with a whole continuous action solution space, ART reduces the complexity of the action solution space through tool-use, which not only improves generalizability across different tasks but also reduces data dependency. To demonstrate the advantages (high generalizability and low data dependency) of this framework, we first built a dataset of 30K tool-use trajectories and action demonstrations, which is much smaller than those used by baseline methods. We then designed a training regimen for long-trajectory tool-use reasoning in challenging environments. Experiments show that ART achieves a 20% higher su
    
[^296]: Aftab：并行化Q网络中CNN编码器与先进价值函数的综合基准

    Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks

    [https://arxiv.org/abs/2608.07335](https://arxiv.org/abs/2608.07335)

    本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。

    

    arXiv:2608.07335v2 公告类型：替换交叉 摘要：深度强化学习的最新进展日益倾向于简化、高度并行化的范式。值得注意的是，并行化Q网络（PQN）算法能够在无需经验回放缓冲区或目标网络的情况下进行离策略价值学习。然而，在这些无缓冲区设置中运行的视觉编码器的表示能力和计算效率仍相对未被充分探索。在本工作中，我们系统性地研究了PQN内卷积神经网络的架构设计空间。我们评估了八种不同的CNN拓扑结构，同时明确表征了它们的参数和计算需求。我们进一步通过将Hadamax编码范式与分类、集成和决斗价值头集成，研究了乘性表示学习和先进价值估计的效果。在Atari-57上的广泛实验表明，我们最终的复合架构...

    arXiv:2608.07335v2 Announce Type: replace-cross  Abstract: Recent advancements in deep reinforcement learning have increasingly favored simplified, highly parallelized paradigms. Notably, the Parallelized Q-Network (PQN) algorithm enables off-policy value learning without relying on experience replay buffers or target networks. However, the representational capacity and computational efficiency of visual encoders operating in these buffer-free settings remain comparatively underexplored. In this work, we systematically investigate the architectural design space of Convolutional Neural Networks within PQN. We evaluate eight distinct CNN topologies while explicitly characterizing their parameter and computational requirements. We further study the effect of multiplicative representation learning and advanced value estimation by integrating the Hadamax encoding paradigm with categorical, ensemble, and dueling value heads. Extensive experiments on Atari-57 show that our final composite arc
    
[^297]: 面向AI代理签名工作流的硬件密钥存储：一种零信任MCP强制执行架构

    Hardware Keystores for AI Agent Signing Workflows: A Zero-Trust MCP Enforcement Architecture

    [https://arxiv.org/abs/2608.06130](https://arxiv.org/abs/2608.06130)

    论文针对AI代理签名场景中的“困惑代理人”问题，提出了一种结合硬件密钥存储的五层零信任强制执行架构，确保只有符合操作者已确认意图的请求才能触发硬件签名，从而防御提示注入攻击导致的私钥滥用。

    

    AI代理越来越多地代表其操作者签署Git提交、认证文档并证明发布工件，其所使用的私钥保存在软件可访问的位置（明文文件、环境变量、容器内存）中，任何该代理可触达的进程都能读取这些密钥。一个广泛部署的代理框架最近就因一次电子邮件注入以这种方式泄露了其密钥。硬件密钥存储（HSM、TPM、智能卡）将密钥保存在设备上，但将密钥存储暴露为LLM代理可调用的工具只是转移了问题而非消除它：一旦签名会话建立，硬件无法区分反映操作者意图的请求与被注入到代理所读取内容中的请求。我们刻画了这一“困惑代理人”（confused-deputy）问题，并构建了其所需要的五层零信任强制执行栈，使只有与操作者已确认意图一致的请求才能到达硬件。我们在两个攻击面上进行了评估。提示注入……

    arXiv:2608.06130v2 Announce Type: replace-cross  Abstract: AI agents increasingly sign Git commits, certify documents, and attest release artifacts on behalf of their operators, using private keys that live in software-accessible locations (plaintext files, environment variables, container memory) readable by any process the agent can reach. A widely deployed agent framework recently leaked its keys this way to a single email injection. Hardware keystores (HSM, TPM, smart card) keep the key on-device, but exposing the keystore as a tool an LLM agent can call moves the problem rather than removing it: once a signing session exists, the hardware cannot tell a request reflecting the operator's intent from one injected into content the agent read. We characterize this confused-deputy problem and build the five-layer Zero-Trust enforcement stack it requires, so that only requests consistent with the operator's committed intent reach the hardware. We evaluate on two attack planes. Prompt inj
    
[^298]: Q-CueGraph：面向多模态推理的查询条件化视觉证据图谱

    Q-CueGraph: Query-Conditioned Visual Evidence Graphs for Multimodal Reasoning

    [https://arxiv.org/abs/2608.04452](https://arxiv.org/abs/2608.04452)

    提出Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取框架，通过构建可复用的OCR与版面关系图谱、查询条件化目标检测以及无需证据框监督的轻量级答案性评分器，自适应地激活并组合视觉证据区域，从而提升多模态推理性能。

    

    多模态大语言模型在观察完整图像时可能会遗漏它们在更近距离观察中本可识别的细节。恢复这些证据需要决定往哪里看以及保留多少周围上下文。我们提出了Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取方法。对于文本丰富的图像，它构建一个可复用的OCR文本行与版面关系图谱；每个问题激活锚点，将其扩展为上下文区域，并选择候选对象构成单个观察窗口。查询条件化的目标检测通过相同的区域选择与组合接口支持自然图像搜索。轻量级候选评分器进一步从冻结读取器的反馈和训练答案中学习哪些观察能够支持正确答案，而无需证据框监督。在六个基准测试中，我们检验了查询条件化、证据组合以及可学习答案性的作用。

    arXiv:2608.04452v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) can miss fine details in a full image that they recognize in a closer view. Recovering this evidence requires deciding where to look and how much surrounding context to retain. We present Q-CueGraph, a query-conditioned evidence acquisition method for frozen MLLMs. For text-rich images, it builds a reusable graph of OCR lines and layout relations. Each question activates anchors, expands them into contextual regions, and selects candidates for a single observation window. Query-conditioned object detections support natural-image search through the same region-selection and composition interface. A lightweight candidate scorer further learns which observations support correct answers from frozen-reader feedback and training answers, without evidence-box supervision. Across six benchmarks, we examine the roles of query conditioning, evidence composition, and learned answerability. With Qwe
    
[^299]: Trident：如何攻破深度强化学习网络防御（智能体）

    Trident : How to Break Deep Reinforcement Learning Cyber Defenses (Agentic)

    [https://arxiv.org/abs/2608.04317](https://arxiv.org/abs/2608.04317)

    该论文提出Trident框架，通过动态沙箱基准环境、超过1.3万条红蓝对抗交互轨迹数据集以及“代码即策略”的RLVR智能体架构，训练LLM红队智能体自适应地发起攻击，从而揭示深度强化学习网络防御系统面对自适应威胁时的脆弱性。

    

    基于深度强化学习（DRL）的自主网络防御系统引起了广泛的研究关注，但其评估几乎完全针对静态的、启发式的红方智能体，这使得这类系统在面对自适应威胁时的鲁棒性严重缺乏研究。与此同时，基于可验证奖励的强化学习（RLVR）的最新进展提升了大语言模型（LLM）的推理能力，但由于缺乏合适的基准环境和交互数据集，其与网络安全的结合仍难以实现。为了弥合这一差距，我们提出了Trident，一个基于智能体LLM的红队测试框架，包含三个组件：一个跨越CybORG CAGE 4和CyberWheel、配备隔离沙箱服务器的动态基准环境；一个包含超过13,000条高保真红蓝对抗交互轨迹、用于RLVR的数据集；以及一个“代码即策略”（Code-as-Policy）的RLVR智能体架构（Trident Agentic）。后者将红方智能体的训练重新表述为一种上下文相关的……

    arXiv:2608.04317v2 Announce Type: replace-cross  Abstract: Autonomous cyber defense systems based on Deep Reinforcement Learning (DRL) have attracted significant research attention, yet remain evaluated almost exclusively against static, heuristic red agents, leaving their robustness against adaptive threats critically understudied. Meanwhile, recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have improved LLM reasoning, but their integration into cybersecurity remains elusive due to the absence of suitable benchmark environments and interaction datasets. To bridge this gap, we introduce Trident, an agentic LLM red teaming framework comprising three components: a dynamic benchmark with isolated sandbox servers spanning CybORG CAGE 4 and CyberWheel, a dataset comprises over 13,000 high-fidelity red-blue interaction trajectories for RLVR, and a ``Code-as-Policy'' RLVR agentic architecture Trident Agentic). The latter reformulates red agent training as a contextual 
    
[^300]: TARL：面向长期智能体可执行内存管理的事务感知可靠账本

    TARL: Transaction-Aware Reliable Ledgers for Executable Memory Management in Long-Term Agents

    [https://arxiv.org/abs/2608.03699](https://arxiv.org/abs/2608.03699)

    提出TARL框架，将长期智能体的记忆更新从二元“写入/保留”决策扩展为添加、忽略、修订、拒绝、推迟验证五种可执行操作，通过维护已接受、待定、已拒绝三类账本并借助记忆状态对比进行训练，避免单个更新错误反复扭曲未来的检索与推理。

    

    持久化记忆帮助长期智能体保留知识，然而单个更新错误可能会反复扭曲未来的检索与推理。大多数现有系统将记忆更新简化为二元的“写入/保留”决策，无法区分新信息应该被添加、被忽略、用于修订过时的信念、因不可靠而被拒绝，还是推迟以待验证。这些选择可能共享同一个二元标签，却会产生根本不同的记忆状态。我们提出了TARL，一个将每条语句映射到五种可执行操作之一的记忆状态更新框架。TARL识别受影响的记忆，解析其时间范围，比较来源可靠性，并更新已接受、待定和已拒绝三类账本。该框架还通过比较不同更新操作所产生的记忆状态来进行训练，鼓励模型选择能导向正确结果的操作。我们还引入了TARL-Mem，一个……（摘要原文在此处截断）

    arXiv:2608.03699v4 Announce Type: replace  Abstract: Persistent memory helps long-term agents retain knowledge, yet a single update error can repeatedly distort future retrieval and reasoning. Most existing systems reduce memory updating to a binary Write/Hold decision, which cannot distinguish whether new information should be added, ignored, used to revise an outdated belief, rejected as unreliable, or deferred for verification. These choices may share the same binary label while producing fundamentally different memory states. We introduce TARL, a memory state update framework that maps each statement to one of five executable actions. TARL identifies the affected memory, resolves its temporal scope, compares source reliability, and updates accepted, pending, and rejected ledgers. It is further trained by comparing the memory states produced by alternative update operations, encouraging the model to select the operation that leads to the correct result. We also introduce TARL-Mem, a
    
[^301]: 布线优于混合：不同Transformer规模之间传递了什么——以及什么没有传递

    Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't

    [https://arxiv.org/abs/2608.02829](https://arxiv.org/abs/2608.02829)

    本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。

    

    arXiv:2608.02829v3 公告类型：替换交叉 摘要：模型家族通常按规模逐个从头训练。能否将预训练的大模型转换为较小的兄弟模型？我们端到端地表征了Pythia中的1.4B->410M转换。表示在不同规模间强烈对齐（岭回归R^2=0.84），而参数对齐较弱。密集权重投影在功能上具有破坏性，且一个比特精确的控制表明这不是组装伪影：基混合破坏了旋转、每头、GELU和LayerNorm结构。在最佳拟合线性算子之后，权重残差在洗牌控制下在统计上与噪声无异。因此，转换价值存在于初始化中。在匹配预算的持续预训练中，我们将转换分解为两个独立杠杆：最小二乘补偿（功能杠杆，最佳零样本）和方差保持重缩放（动力学杠杆，最佳终点）。补偿是一种令牌高效、低预算的胜利，而非其他。

    arXiv:2608.02829v3 Announce Type: replace-cross  Abstract: Model families are typically trained size by size, each from scratch. Can a pretrained large model instead be converted into a smaller sibling? We characterize the 1.4B->410M conversion in Pythia end to end. Representations align strongly across sizes (ridge R^2=0.84) while parameters align weakly. Dense weight projection is functionally destructive, and a bit-exact control shows this is not an assembly artifact: basis mixing breaks rotary, per-head, GELU, and LayerNorm structure. After the best-fit linear operator, weight residuals are statistically indistinguishable from noise under shuffle controls. Conversion value therefore lives in initialization. In matched-budget continued pre-training we decompose conversion into two independent levers: least-squares compensation (function lever, best zero-shot) and variance-preserving rescale (dynamics lever, best endpoints). Compensation is a token-efficient, low-budget win rather th
    
[^302]: 图表支持还是模型自供？检验多模态大语言模型为无障碍可视化生成的声明

    Chart-Supported or Model-Supplied? Examining MLLM-Generated Claims for Accessible Visualization

    [https://arxiv.org/abs/2607.25021](https://arxiv.org/abs/2607.25021)

    本研究发现，提供数据表、标题、替代文本等无障碍图表上下文比提供图像本身更能有效促使MLLM生成有依据的直接声明并提升数值一致性。

    

    多模态大语言模型（MLLM）能够将可视化模式与外部原因、后果和领域知识联系起来，但这些解释的证据基础往往不明确。我们开展了一项探索性研究，涵盖来自四个来源的102个可视化图表、三个MLLM以及四种输入条件，这些条件改变了对图像的访问、无障碍图表上下文（如数据表、标题、替代文本和屏幕阅读器结构等非图像制品）以及保留上下文的提示框架。在1,224份描述中，我们分析了模型归因的DIRECT（直接）、DERIVED（推导）和SPECULATIVE（推测）标签，并对数值一致性进行了自动化审计。无障碍图表上下文使Gemini和GPT更倾向于生成DIRECT声明，并提高了某些模型的数值一致性。在完整上下文中添加图像并未带来一致的数值收益，而保留上下文的提示也未可靠地促使模型使用更谨慎的措辞。（注：原文摘要在此处截断）

    arXiv:2607.25021v2 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) can connect visualization patterns to external causes, consequences, and domain knowledge, but the evidential basis of these interpretations is often unclear. We present an exploratory study of 102 visualizations from four sources, three MLLMs, and four input conditions that vary access to the image, accessible chart context (non-image artifacts such as data tables, captions, alt text, and screen-reader structures), and withheld-context framing. Across 1,224 descriptions, we analyze model-attributed DIRECT, DERIVED, and SPECULATIVE labels and conduct an automated audit of numeric agreement. Accessible chart context shifted Gemini and GPT toward DIRECT claims and improved numeric agreement for some models. Adding the image to the full context did not yield a consistent numeric benefit, and the withheld-context prompt did not reliably increase cautious language. The prompt-defined Real-World Sig
    
[^303]: 一种用于旋转机械物理可验证故障诊断的多层次信息集成框架

    A Multi-level Information Integration Framework for Physically Verifiable Fault Diagnosis of Rotating Machinery

    [https://arxiv.org/abs/2607.22797](https://arxiv.org/abs/2607.22797)

    提出一种与编码器无关的多任务框架——诊断证据网络（DENet），将旋转机械故障诊断输出扩展为包含分类结果、可对照轴承几何与转速理论值验证的预测特征频率以及时间定位信息的结构化证据记录，从而实现物理可验证的故障诊断。

    

    将从物理模型到数据驱动诊断再到自然语言推理的多层次信息整合为可验证的决策链，是智能制造领域日益增长的需求。本文以轴承故障诊断作为代表性测试平台，指出传统方法的标准输出仅为类别标签和由分类器自身分布导出的置信度分数，缺乏与独立物理知识进行对照比较的手段；同时，越来越多用于维护沟通的语言模型可能引入缺乏依据的内容。本工作从输出端解决上述两个局限。所提出的诊断证据网络（DENet）是一种与编码器无关的多任务框架，其将输出扩展为一条结构化的证据记录：包括分类结果、可与由轴承几何尺寸和轴速确定的理论值进行对比验证的预测特征频率，以及时间定位信息（摘要在此处被截断，后续内容未提供）。

    arXiv:2607.22797v2 Announce Type: replace-cross  Abstract: Integrating multi-level information, from physical models through data-driven diagnostics to natural language reasoning, into verifiable decision chains is a growing need in intelligent manufacturing. In bearing fault diagnosis, taken here as a representative testbed, the standard output is a class label and a confidence score derived from the classifier's own distribution, offering limited means of comparison against independent physical knowledge. Meanwhile, language models increasingly used for maintenance communication may introduce unsupported content. This work addresses both limitations from the output side. The proposed Diagnostic Evidence Network (DENet) is an encoder-agnostic multi-task framework that extends the output to a structured evidence record: the classification, a predicted characteristic frequency comparable against the theoretical value determined by bearing geometry and shaft speed, and a temporal localiz
    
[^304]: 线性与受保护存在规则下的路径查询回答

    Answering Path Queries under Linear and Guarded Existential Rules

    [https://arxiv.org/abs/2607.22636](https://arxiv.org/abs/2607.22636)

    本文证明了在线性存在规则下回答双向（合取）正则路径查询在数据复杂性上是NL完全的（与无本体的普通图数据库一致），在组合复杂性上一般为ExpTime完全（谓词元数有界时RPQ为PTime完全、CRPQ为PSpace完全），并为受保护存在规则情形给出了重要结果。

    

    本体中介的查询回答研究的是在由数据库实例和本体组成的知识库上回答查询的问题。虽然该领域的大多数工作聚焦于合取查询（CQ），但导航式查询正获得越来越多的关注。在本文中，我们研究了在本体由受保护存在规则集合给出的知识库上回答双向（合取）正则路径查询（(C)RPQ）的复杂性。我们首先考虑线性存在规则这一子类，并证明(C)RPQ回答在数据复杂性上是NL完全的，这与在普通图数据库（即不含本体）上回答RPQ的数据复杂性相匹配。在组合复杂性方面，这两种任务在一般情况下都是ExpTime完全的，但如果谓词的元数有界，RPQ和CRPQ回答则分别降至PTime完全和PSpace完全。对于受保护规则，我们提供了一个非平凡的（摘要在此处被截断）

    arXiv:2607.22636v2 Announce Type: replace  Abstract: Ontology-mediated query answering is concerned with the problem of answering queries over knowledge bases consisting of a database instance and an ontology. While most work in the area focuses on conjunctive queries (CQs), navigational queries have gained increasing attention. In this paper, we investigate the complexity of answering two-way (conjunctive) regular path queries ((C)RPQs) over knowledge bases whose ontology is given by a set of guarded existential rules. We first consider the subclass of linear existential rules and show that (C)RPQ answering is NL-complete in data complexity, which matches the data complexity of answering RPQs over plain graph databases (i.e., without an ontology). In combined complexity, both tasks are ExpTime-complete in the general case, but RPQ and CRPQ answering drop to PTime-complete and PSpace-complete respectively if there is a bound on predicate arity. For guarded rules, we provide a non-trivi
    
[^305]: 自主AI代理的密码学可验证授权：一个可证伪的假设与概念验证

    Cryptographically verifiable authorization for autonomous AI agents: A falsifiable hypothesis and proof-of-concept

    [https://arxiv.org/abs/2607.21325](https://arxiv.org/abs/2607.21325)

    本文提出并验证了自主AI代理授权可形式化为密码学可验证关系的假设，通过绑定代理、请求、上下文和策略满足性，实现了可验证且保护隐私的授权机制。

    

    自主AI代理越来越多地在人类监督有限的情况下执行操作、调用工具并访问受保护资源。现有的身份验证和授权机制能够确立身份并委派权限，但本质上无法提供密码学证据，以证明特定代理发出的具体请求在特定执行上下文中满足适用策略。本文假设代理授权可以被形式化为一种密码学可验证的关系，记为$R_{CVA}$，该关系联合绑定代理主体、具体授权请求、执行上下文以及适用策略的满足情况，同时选择性保护私有授权属性的机密性。我们引入了密码学可验证代理授权（CVA）的初步形式抽象，定义了一组紧凑的候选安全属性，包括授权可靠性。

    arXiv:2607.21325v2 Announce Type: replace-cross  Abstract: Autonomous AI agents increasingly execute actions, invoke tools, and operate on protected resources with limited human oversight. Existing authentication and authorization mechanisms establish identity and delegate authority, but do not inherently provide cryptographic evidence that a concrete request issued by a specific agent satisfies the applicable policy in a specific execution context. This paper hypothesizes that agent authorization can be formalized as a cryptographically verifiable relation, denoted $R_{CVA}$, that jointly binds an agent principal, a concrete authorization request, an execution context, and the satisfaction of an applicable policy, while selectively preserving the confidentiality of private authorization attributes. We introduce a preliminary formal abstraction for Cryptographically Verifiable Agent Authorization (CVA), define a compact set of candidate security properties including authorization sound
    
[^306]: SechKAN：基于双曲正割函数的Kolmogorov-Arnold网络

    SechKAN: Kolmogorov-Arnold Networks with Hyperbolic Secant Functions

    [https://arxiv.org/abs/2607.18290](https://arxiv.org/abs/2607.18290)

    本文提出了基于双曲正割函数的新型Kolmogorov-Arnold网络SechKAN，通过一维线性投影控制参数规模，在函数拟合、PDE代理建模和图像分类任务上取得了与MLP及现有KAN变体相当或更优的性能。

    

    近年来，Kolmogorov-Arnold网络（KANs）因其在机器学习和科学计算中的有效性而受到越来越多的关注，为神经网络设计提供了一种新的范式。本文提出了SechKAN，一种基于双曲正割函数的新型KAN。选择双曲正割基函数是因为其平滑的钟形曲线、局部化响应和良好的梯度特性。我们采用一维线性投影来减少参数数量，使SechKAN能够保持与多层感知器（MLPs）相当的模型规模。实验结果表明，SechKAN在函数拟合、偏微分方程（PDE）代理建模和图像分类基准测试（包括MNIST、FashionMNIST、CIFAR10和CIFAR100）上均表现出有效性。在函数拟合任务上，SechKAN达到了与MLPs及代表性KAN变体相当的性能。在PDE代理建模上，它优于MLPs，并取得了具有竞争力或更好的表现。

    arXiv:2607.18290v3 Announce Type: replace-cross  Abstract: In recent years KolmogorovArnold Networks KANs have attracted increasing attention due to their effectiveness in machine learning and scientific computing offering a new paradigm for neural network design In this paper we present SechKAN a novel KAN based on hyperbolic secant sech functions The hyperbolic secant basis is adopted for its smooth bellshaped form localized responses and wellbehaved gradients We employ a 1D linear projection to reduce the number of parameters allowing SechKAN to maintain a model size comparable to that of multilayer perceptrons MLPs Experimental results show the effectiveness of SechKAN on function fitting PDE surrogate modeling and image classification benchmarks including MNIST FashionMNIST CIFAR10 and CIFAR100 On function fitting SechKAN achieves performance comparable to both MLPs and representative KAN variants On PDE surrogate modeling it outperforms MLPs and achieves competitive or better per
    
[^307]: Omni-Decision：面向全模态智能体的证据账本规划

    Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents

    [https://arxiv.org/abs/2607.11433](https://arxiv.org/abs/2607.11433)

    Omni-Decision 针对全模态智能体的规划瓶颈，用显式的证据账本取代不断膨胀的对话历史，由批评者模块过滤嘈杂的多模态观测，仅保留可用证据，使规划器在紧凑上下文中做出更可靠的多步决策。

    

    全模态智能体需要跨视频、音频、网页和计算工具寻找证据来回答问题。其主要瓶颈在于规划：嘈杂的多模态观测在对话历史中不断累积，干扰后续决策，而多模态模型的多步规划能力有限。受控的后端替换实验支持了这一诊断：替换规划器所造成的性能损失远大于替换感知后端。我们提出了 Omni-Decision，一个基于证据账本规划构建的全模态智能体：它用一个显式的证据账本取代不断增长的对话历史，账本记录尚缺哪些证据、哪些证据已被确认，以及哪些记录之间存在冲突。一个批评者模块读取每条嘈杂的观测，只将可用内容传递给账本并丢弃其余部分，使规划器在整个任务过程中始终基于紧凑的上下文进行决策。每次运行都会记录每一步的状态、动作和判定……（原文摘要在此处截断）

    arXiv:2607.11433v2 Announce Type: replace  Abstract: Omni-modal agents must seek evidence across video, audio, web pages, and computation to answer questions. Their main bottleneck is planning: noisy multimodal observations accumulate in conversation history and disrupt later decisions, while multimodal models have limited capacity for multi-step planning. Controlled backend replacements support this diagnosis: replacing the planner causes a much larger performance loss than replacing the perception backend. We present Omni-Decision, an omni-modal agent built on evidence-ledger planning: it replaces the growing dialogue history with an explicit evidence ledger that records what evidence is still missing, what has been confirmed, and where records conflict. A critic reads each noisy observation and passes only the usable content to the ledger, discarding the rest, so the planner works from a compact context throughout the task. Each run records the state, action, and verdict at every st
    
[^308]: SOV-CAD：基于逐步正交视图引导的CAD建模序列重建

    SOV-CAD: Stepwise Orthographic Views Guided CAD Modeling Sequence Reconstruction

    [https://arxiv.org/abs/2607.04119](https://arxiv.org/abs/2607.04119)

    该论文提出SOV-CAD框架，通过在每个建模步骤引入目标正交投影的逐步视觉监督，并将CAD重建形式化为基于Decision Transformer的离线强化学习序列决策任务，从而实现更精确的CAD建模序列重建。

    

    从图像中重建计算机辅助设计（CAD）建模序列对于保留设计意图和支持参数化编辑至关重要。然而，现有方法通常以整体方式一次性生成完整的CAD序列，忽视了人类设计工作流程中迭代式、反馈驱动的本质。我们通过引入丰富的逐步视觉监督来解决这一局限：在每个建模步骤中，系统观察目标物体的正交投影、增量构建模型的投影以及当前活动草图，从而实现有依据的动作选择。为了有效利用这种即时反馈，我们提出了SOV-CAD，这是一个将CAD重建形式化为序列决策任务的框架，并采用基于Decision Transformer架构的离线强化学习。该设计结合了由几何对齐奖励引导的连续视觉反馈，从而实现更精确的……

    arXiv:2607.04119v2 Announce Type: replace-cross  Abstract: Reconstructing Computer-Aided Design (CAD) modeling sequences from images is crucial for preserving design intent and supporting parametric editing. However, existing methods typically generate full CAD sequences holistically, overlooking the iterative, feedback-driven nature of human design workflows. We address this limitation by introducing the rich stepwise visual supervision: at each modeling step, the system observes the target's orthographic projections, the projections of the incrementally constructed model, and the active sketch, enabling informed action selection. To effectively leverage this on-the-fly feedback, we propose SOV-CAD, a framework that formulates CAD reconstruction as a sequential decision-making task and employs offline reinforcement learning with a Decision Transformer architecture. This design incorporates continuous visual feedback guided by geometric alignment rewards, resulting in a more accurate a
    
[^309]: 打破失败级联：面向医学多模态推理的步骤感知强化学习

    Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning

    [https://arxiv.org/abs/2606.31825](https://arxiv.org/abs/2606.31825)

    该论文提出医学推理感知策略优化（MRPO），一种通过分步过程奖励对早期无效推理步骤施加指数级惩罚的强化学习算法，可在不损害成功路径的前提下打破医学多模态推理中的失败级联，从而提升医学视觉问答的准确性。

    

    arXiv:2606.31825v2 公告类型：replace-cross 摘要：近期的多模态大语言模型在临床图像推理方面展现出巨大前景，但现有的后训练流程仍主要以结果为中心，依赖最终答案的正确性或序列级偏好。这种方式存在信用分配稀疏的问题，使得难以优化对临床应用至关重要的推理过程。我们的分析揭示，源自早期推理失败的级联错误是医学视觉问答（VQA）基准中错误预测的主要原因。受此启发，我们提出了医学推理感知策略优化（MRPO），这是一种融合分步过程奖励的强化学习算法。当最终答案错误时，MRPO会对较早的无效推理步骤中的token施加指数级增大的惩罚，从而在不损害成功路径的前提下打破失败级联。在四个多模态LLM骨干模型上，MRPO始终……（原文摘要至此截断）

    arXiv:2606.31825v2 Announce Type: replace-cross  Abstract: Recent multimodal large language models have shown great promise in clinical image reasoning, but existing post-training pipelines remain predominantly outcome-centric, relying on final answer correctness or sequence-level preferences. This suffers from sparse credit assignment, making it difficult to optimize the reasoning process essential for clinical applications. Our analysis reveals that cascading errors from early-stage reasoning failures are a leading cause of incorrect predictions in medical visual question answering (VQA) benchmarks. Motivated by this, we propose Medical Reasoning-aware Policy Optimization (MRPO), an RL algorithm that incorporates step-wise process rewards. When the final answer is incorrect, MRPO assigns exponentially larger penalties to tokens in earlier invalid reasoning steps, breaking failure cascades without compromising successful paths. Across four multimodal LLM backbones, MRPO consistently o
    
[^310]: 分析针对智能体AI系统模型引导自动化攻击的防御性误导策略

    Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems

    [https://arxiv.org/abs/2606.20470](https://arxiv.org/abs/2606.20470)

    本文通过概率模型分析发现传统“检测-拦截”防御会随攻击查询预算增长使攻击成功率趋近于1，并提出“检测-误导”防御策略，通过向检测到的恶意交互返回受控的不可操作响应来诱使攻击者的自动化评判器产生假阳性错误。

    

    智能体AI系统越来越依赖语言模型组件来解释指令、处理外部数据、调用工具以及与其他智能体进行协调。这些能力使得提示注入和越狱攻击的后果更加严重，尤其是当攻击者采用模型引导的自动化手段来扩展探测、提示优化和响应评估时。本工作通过一个涵盖目标系统、其防御机制以及攻击者自动化评判器的概率模型，分析了由此产生的攻防对抗场景。我们的分析表明，传统的“检测-拦截”防御机制会随着查询预算的增长使攻击成功率（ASR）趋近于1，因为可预测的拒绝响应会为自动化搜索提供有用的反馈。随后，我们研究了“检测-误导”策略，即对检测到的恶意交互返回受控的、不可操作的响应，旨在诱使攻击者的评判器产生假阳性错误。

    arXiv:2606.20470v3 Announce Type: replace-cross  Abstract: Agentic AI systems increasingly rely on language-model components to interpret instructions, process external data, invoke tools, and coordinate with other agents.   These capabilities make prompt-injection and jailbreak attacks more consequential, especially as attackers adopt model-guided automation to scale probing, prompt refinement, and response evaluation.   This work analyzes the resulting attack-defense setting through a probabilistic model of a target system, its defense mechanism, and the attacker's automated judge.   Our analysis shows that conventional detect-and-block defenses can allow attacker success rate (ASR) to approach one as the query budget grows, since predictable refusals provide useful feedback to automated search.   We then examine detect-and-misdirect, where detected malicious interactions receive controlled, non-operational responses designed to induce false-positive errors in the attacker's judge.  
    
[^311]: JoyAI-VL-Interaction：实时视觉-语言交互智能

    JoyAI-VL-Interaction: Real-Time Vision-Language Interaction Intelligence

    [https://arxiv.org/abs/2606.14777](https://arxiv.org/abs/2606.14777)

    本文提出完全开源的JoyAI-VL-Interaction——一个8B规模的视觉优先视觉-语言交互模型，能像人一样持续观察环境、自主决定何时发言或保持沉默、实时交互，并在难题出现时委托后台模型处理，从而突破了传统轮次制问答模型的局限。

    

    现实世界中的许多时刻不会等待用户提问：火灾在安全监控屏幕上燃起，表情在视频通话中一闪而过，观众想要的商品在直播中转瞬即逝。然而，当今的大型模型在设计上大多仍是轮次制的：它们只在被叫到时才回答，即使是看似具有交互性的视频通话应用，实际上仍是问答系统，只有被轮询或提示时才会做出反应。我们主张一种不同的范式：一个像人一样存在于世界中的模型。它持续观察当下正在发生的事情，自主决定是发言还是保持沉默，实时进行交互，并在问题困难时将任务委托给后台模型。为推动交互模型的发展及其在各领域的应用，我们做出了两项完全开源的贡献。首先，我们发布了JoyAI-VL-Interaction，一个8B规模的、视觉优先的视觉-语言交互模型。该模型将响应决策置于内部……

    arXiv:2606.14777v2 Announce Type: replace-cross  Abstract: Many moments in the real world do not wait for a user to ask. A fire starts on a security monitor, an expression flickers across a video call, or a product a viewer wants flashes by in a livestream. Yet today's large models remain mostly turn-based by design: they answer only when addressed, and even video-call apps that appear interactive still operate as question-answer systems, reacting only when polled or prompted. We argue for a different paradigm: a model that is present in the world like a person. It continuously watches what is happening now, decides on its own whether to speak or stay silent, interacts in real time, and delegates to a background model when the problem is hard. To advance interaction models and their adoption across domains, we make two fully open-sourced contributions. First, we release JoyAI-VL-Interaction, an 8B-scale, vision-first VL-interaction model. The model makes the response decision internall
    
[^312]: 基于匹配学习的改进顶点分布的三维口腔建模

    3D Oral Modelling with Improved Vertex Distribution Using Matching-Based Learning

    [https://arxiv.org/abs/2606.07907](https://arxiv.org/abs/2606.07907)

    本文提出结合带过滤的匈牙利匹配与排斥损失的改进损失函数，显著缓解了三维口腔重建中的顶点聚集问题，使重建模型的顶点分布更加均匀。

    

    在我们之前的工作中，我们提出了一种基于深度学习的三维口内重建框架。该模型从十张固定角度的口内图像中直接预测显式的三维点云坐标，采用MobileNetV2和多头注意力机制进行多视角特征融合，并以L1损失与倒角距离的组合作为损失函数。尽管该模型达到了77.49%的准确率，但预测的顶点往往集中在真值的高密度区域，而其他区域则基本未被覆盖。本文提出了一种改进的损失函数来解决这一局限。我们引入了带过滤的匈牙利匹配和排斥损失，以促使重建模型中的顶点分布更加均匀。所提出的模型达到了68.02%的准确率，在数值上低于之前的模型，然而，先前工作中观察到的顶点聚集问题得到了显著缓解。

    arXiv:2606.07907v2 Announce Type: replace-cross  Abstract: In our previous work, a deep learning-based framework for 3D intraoral reconstruction was proposed. The model directly predicts explicit 3D point cloud coordinates from ten fixed-angle intraoral images, employing MobileNetV2 and Multi-head Attention for multi-view feature fusion, with a combined L1 Loss and Chamfer Distance as the loss function. Although the model achieved an accuracy of 77.49%, predicted vertices tended to concentrate in high-density regions of the ground truth, leaving other regions largely uncovered.   In this paper, an improved loss function is proposed to address this limitation. Hungarian matching with filtering and Repulsion Loss are introduced to enforce more uniform vertex distribution across the reconstructed model. The proposed model achieves an accuracy of 68.02%, which is numerically lower than the previous model. However, the vertex clustering issue observed in the prior work is substantially alle
    
[^313]: CaliPPer：量化、预测并提升结合预测任务中AI模型的性能

    CaliPPer: quantifying, predicting and improving AI model performance for binding prediction

    [https://arxiv.org/abs/2606.07258](https://arxiv.org/abs/2606.07258)

    CaliPPer是一个事后校准框架，通过多链样本-域距离与距离感知贝叶斯重校准，在无标签条件下实现免疫受体结合预测AI模型性能的量化、预测与提升。

    

    结合预测模型能够加速治疗性抗体和TCR的发现，但它们在新数据集上的表现难以预料，常常导致较低的发现率。密度比方法（如PAPE、M-CBPE）虽然可以在无标签条件下为二分类任务提供性能估计，但其假设以及仅输出聚合结果的特点，限制了它们在新生抗原表位、抗原变异体和化学骨架等结合预测场景中的应用。本文提出CaliPPer（Calibration and Prediction of Performance，校准与性能预测），这是一个事后框架，将多链的“样本到域距离”与距离感知的贝叶斯重校准相结合，并在三个层次上运作：泛化性评分、聚合性能预测以及逐样本置信度。在十个模型、八种架构和两个免疫受体领域上，CaliPPer实现了距离与性能之间|r|=0.80–0.92的相关性，预测AUROC/AP/F1的平均绝对误差仅为0.008–0.070，并显著提升了AUROC（此处摘要截断）。

    arXiv:2606.07258v1 Announce Type: cross  Abstract: Binding prediction models accelerate therapeutic antibody and TCR discovery, but their performance on new datasets is unpredictable, often leading to low discovery rates. Density-ratio methods (PAPE, M-CBPE) provide label-free performance estimation for binary classification, but their assumptions and aggregate-only outputs limit binding prediction on neoepitopes, antigen variants and chemical scaffolds. Here we present CaliPPer (Calibration and Prediction of Performance), a post-hoc framework pairing a multi-chain Sample-to-Domain Distance (S2DD) with distance-aware Bayesian recalibration, operating at three resolutions: generalisability score, aggregate performance prediction, and per-sample confidence. Across ten models, eight architectures and two immune-receptor domains, CaliPPer attains distance--performance correlations $|r|=0.80\text{--}0.92$, predicts AUROC/AP/F1 with mean absolute errors $0.008\text{--}0.070$, and improves AU
    
[^314]: 基于深度学习的使用二维口内图像的三维口腔重建

    Deep Learning-based 3D Oral Cavity Reconstruction Using 2D Intraoral Images

    [https://arxiv.org/abs/2606.05998](https://arxiv.org/abs/2606.05998)

    本文提出一种基于深度学习的纯软件方法，仅用十张不同角度的二维口内图像即可重建三维口腔模型，无需专用硬件设备，克服了传统取模带来的患者不适及口内扫描设备成本高昂的问题。

    

    口腔三维建模是牙科中最重要的环节之一，目前常用的方法包括取模和口内扫描等多种技术，但每种方法都存在明显的局限性。取模需要将藻酸盐或硅胶材料放入托盘中并插入患者口腔内以形成负模，该方法会使患者产生明显的不适感，存在材料变形误差，并且在储存和运输方面存在困难。口内扫描仪利用结构光或激光技术实时直接扫描口腔结构，能够产生最先进的结果，但设备成本极其高昂。为了解决这些局限性，本文提出了一种基于软件的方法，仅使用从不同角度拍摄的十张二维口内图像即可重建三维口腔模型，无需任何专用硬件设备。

    arXiv:2606.05998v2 Announce Type: replace-cross  Abstract: Oral 3D modelling is one of the most essential stages in dentistry, and many different approaches, such as impression taking and intraoral scanning, are commonly used for this phase, each with notable limitations. Impression taking, which involves placing alginate or silicone material in a tray and inserting it into the patient's oral cavity to form a negative mold, suffers from significant patient discomfort, material deformation errors, and difficulties in storage and transportation. Intraoral scanners, which directly scan oral structures in real time using structured light or laser technology, produce state-of-the-art results but are associated with substantially high equipment costs. To address these limitations, this paper proposes a software-based approach that reconstructs a 3D oral model using only ten 2D intraoral images captured from different angles, requiring no dedicated hardware devices. The proposed method reduce
    
[^315]: NVIDIA OmniDreams：面向闭环自动驾驶仿真的实时生成式世界模型

    NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation

    [https://arxiv.org/abs/2606.03159](https://arxiv.org/abs/2606.03159)

    OmniDreams是基于Cosmos扩散模型、经2.1万小时驾驶数据训练的实时生成式世界模型，可自回归生成动作条件视频，突破重建式神经仿真器的数据局限，用于自动驾驶策略的闭环安全评估。

    

    随着自动驾驶能力的不断进步，在长尾场景中对驾驶策略进行安全评估仍然是一个关键瓶颈。在闭环仿真中，驾驶策略模型与环境主动交互，其动作会动态更新仿真器状态，并直接影响下一组生成的传感器观测。虽然近期基于重建的神经仿真器能够提供照片级真实感，但它们从根本上受限于初始采集的数据，难以泛化到高度动态或全新的场景。为了克服这些限制，我们提出了OmniDreams，这是一个基础生成式世界模型，由Cosmos扩散模型经过中期训练和后期训练而来，能够实时自回归地生成以动作为条件的视频。通过利用Cosmos丰富的视觉先验，并在2.1万小时的驾驶场景上进行中期训练和后期训练，OmniDreams能够合成复杂的、未被观测到的现象……

    arXiv:2606.03159v3 Announce Type: replace-cross  Abstract: As autonomous vehicle capabilities advance, the safe evaluation of driving policies in long-tail scenarios remains a critical bottleneck. In closed-loop simulation, the driving policy model actively interacts with the environment, where its actions dynamically update the simulator state and directly influence the next set of generated sensor observations. While recent reconstruction-based neural simulators offer photorealism, they are fundamentally constrained by their initial captured data and struggle to generalize to highly dynamic or novel scenes. To overcome these limitations, we introduce OmniDreams, a foundation generative world model mid- and post-trained from the Cosmos diffusion model to autoregressively generate action-conditioned videos in real time. By leveraging the rich visual priors of Cosmos and mid- and post-training on 21k hours of driving scenarios, OmniDreams synthesizes complex, unobserved phenomena that a
    
[^316]: 规划不止于词元预测：用于基准测试与构建物理接地具身推理器的因果规划

    Planning Takes More Than Token Prediction: Causal Plan for Benchmarking and Building Physically Grounded Embodied Reasoners

    [https://arxiv.org/abs/2606.01810](https://arxiv.org/abs/2606.01810)

    本文提出 Causal-Plan-Bench 基准与百万级因果推理语料库 Causal-Plan-1M，揭示当前具身视觉语言模型偏向语言词元预测而缺乏物理因果推理能力，推动从语言统计先验向物理接地的因果规划转变。

    

    当前用于具身视觉-语言规划的基准测试无意中偏向了语言上的下一词元预测，而非基于物理的下一状态推理。这奖励了那些模仿统计语言先验而非追踪真实因果依赖的模型，将复杂的物理规划简化为浅层的序列建模。因此，实现真正的物理自主性需要从基于语言学的词元预测向基于物理的因果推理进行根本性转变。为此，我们推出了 Causal-Plan-Bench，一个跨越四个因果维度、经多阶段验证精心筛选的高保真诊断套件。为了赋予模型这种能力，我们设计了一个四阶段标注流水线，从第一人称视角视频中提取结构化交互记录，构建了 Causal-Plan-1M——一个包含百万级显式因果推理轨迹的密集语料库。广泛的评估揭示了一个显著的差距：领先模型难以展现出……

    arXiv:2606.01810v2 Announce Type: replace  Abstract: Current benchmarks for embodied vision-language planning inadvertently favor linguistic next-token prediction over physically grounded next-state reasoning. This rewards models that mimic statistical language priors rather than track true causal dependencies, reducing complex physical planning to shallow sequence modeling. Hence, achieving genuine physical autonomy requires a fundamental shift from linguistically grounded token prediction toward physically grounded causal reasoning. To this end, we introduce Causal-Plan-Bench, a high-fidelity diagnostic suite spanning four causal dimensions, curated via multi-stage verification. To endow models with this capability, a four-stage annotation pipeline extracts structured interaction records from egocentric videos to construct Causal-Plan-1M, a dense million-scale corpus of explicit causal reasoning traces. Extensive evaluation reveals a striking gap: leading models struggle to demonstra
    
[^317]: 一种用于光片荧光显微镜的多模态3D基础模型，实现少样本分割、分类与去模糊

    A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring

    [https://arxiv.org/abs/2605.26026](https://arxiv.org/abs/2605.26026)

    该论文提出了一个针对光片荧光显微镜数据的3D基础模型，通过在大规模多样化3D图像上联合优化掩码重建与图像-文本对齐进行预训练，学习可迁移的体积表征，大幅降低标注负担，并实现少样本分割、分类与去模糊。

    

    光片荧光显微镜（LSM）能够对生物样本进行高分辨率的三维（3D）成像，为研究细胞组织结构、病理学和血管网络提供丰富的体积数据。然而，LSM数据的规模、维度和标注负担使得有监督深度学习方法成本高昂且难以扩展。此外，尽管未标注的LSM体积数据十分丰富，但由于计算上的挑战以及体积表征学习的复杂性，针对这一模态的基础模型仍然探索不足。在本工作中，我们为LSM数据引入了一个3D基础模型，该模型在跨越多种生物体、染色方法和成像协议的大型精选3D图像集合上进行了预训练。我们通过联合优化掩码重建和图像-文本对齐来学习可迁移的体积表征。预训练的骨干网络极大地降低了标注负担。

    arXiv:2605.26026v2 Announce Type: replace-cross  Abstract: Light sheet fluorescence microscopy (LSM) enables high-resolution, three-dimensional (3D) imaging of biological specimens, providing rich volumetric data for studying cellular organization, pathology, and vascular networks. However, the size, dimensionality, and annotation burden of LSM data make supervised deep learning approaches costly and difficult to scale. Additionally, despite the abundance of unannotated LSM volumes, foundation models for this modality remain underexplored due to computational challenges and the complexity of volumetric representation learning. In this work, we introduce a 3D foundation model for LSM data, pretrained on a large curated collection of 3D images spanning multiple organisms, stains, and imaging protocols. We learn transferable volumetric representations by jointly optimizing for masked reconstruction and image-text alignment. The pretrained backbone drastically reduces the annotation burden
    
[^318]: 当搜索成为记忆：利用自进化技能加速机器人设计发现

    When Search Becomes Memory: Accelerating Robot Design Discovery with Self-Evolving Skills

    [https://arxiv.org/abs/2605.25832](https://arxiv.org/abs/2605.25832)

    提出Auto-Robotist，一个自进化的LLM智能体，通过将进化搜索轨迹提炼为可检查的自然语言技能库，把搜索结果转化为可重用的设计记忆，从而加速机器人形态设计的发现。

    

    大语言模型（LLM）正越来越多地被用作进化机器人设计的方案生成器，然而大多数循环仍是无记忆的：模拟器结果虽能塑造下一代种群，却未被保存为可重用的设计知识。我们提出Auto-Robotist，一个自进化的LLM智能体，它将形态搜索轨迹提炼为显式的自然语言技能库。每个技能存储一个结构原型、有证据支持的正面与负面规则，以及支持这些规则的已评估设计，使设计记忆变得可检查，而非隐含在种群之中。在搜索过程中，该智能体检索技能以引导LLM对精英个体进行编辑，同时保留遗传算法（GA）变异路径用于探索；评估完成后，它通过添加、诊断和合并操作来更新技能库。在涵盖运动、穿越和物体交互的七个EvoGym任务上，Auto-Robotist提升了冷启动5×5搜索的性能……

    arXiv:2605.25832v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and tr
    
[^319]: DeGRe：面向推荐的密集监督生成式重排序

    DeGRe: Dense-supervised Generative Reranking for Recommendation

    [https://arxiv.org/abs/2605.25749](https://arxiv.org/abs/2605.25749)

    该论文提出DeGRe框架，通过引入密集监督信号来解决生成式重排序中的启发式标签偏差和稀疏奖励导致的信用分配难题。

    

    在多阶段推荐系统中，重排序通过捕捉列表内部的上下文依赖关系来优化整体效用，但其核心挑战在于如何在指数级庞大的排列空间中探索最优序列。近期研究已转向端到端生成式框架，这类方法通常利用列表级奖励或偏好对齐来指导生成器的训练。然而，这些方法仍面临两个关键问题。第一是启发式标签偏差：现有方法往往基于简单规则构建训练目标，例如将用户点击过的物品提升至列表顶部，而忽略了列表上下文中的因果依赖关系。第二是信用分配问题：稀疏的列表级事后奖励无法直接指导序列生成过程中的中间步骤，导致优化方向模糊不清。为解决这些问题，我们提出了DeGRe（密集监督生成式重排序），一种……

    arXiv:2605.25749v2 Announce Type: replace-cross  Abstract: In multi-stage recommender systems, reranking optimizes overall utility by capturing intra-list contextual dependencies, yet its central challenge lies in exploring optimal sequences within an exponentially large permutation space. Recent studies have shifted towards end-to-end generative frameworks, which typically leverage list-wise rewards or preference alignment to guide generator training. However, these methods still face two critical issues. First is the heuristic label bias. Existing methods often construct training targets based on simple rules, such as promoting clicked items to the top, while ignoring causal dependencies within the list context. Second is the credit assignment problem. Sparse list-level posterior rewards fail to directly guide intermediate steps in sequence generation, leading to ambiguous optimization directions.   To address these issues, we propose DeGRe (Dense-supervised Generative Reranking), a 
    
[^320]: 面向临时团队协作的上下文强化学习极限基准测试

    Benchmarking the Limits of In-Context Reinforcement Learning for Ad-Hoc Teamwork

    [https://arxiv.org/abs/2605.24423](https://arxiv.org/abs/2605.24423)

    该论文提出了 ICRL4AHT——首个基于 Overcooked-V2 构建的大规模基准，用于系统评估上下文强化学习在临时团队协作中的表现，并发现 AD、DPT 等现有 ICRL 方法在与未知队友协调时存在显著局限。

    

    arXiv:2605.24423v2 公告类型：替换 摘要：上下文强化学习（ICRL）已使基础智能体能够即时适应新任务，但其在临时团队协作（AHT）——即需要与未知伙伴进行协调的场景——中的有效性仍未被探索。为了严谨地评估这一点，我们引入了一个大规模基准 ICRL4AHT，该基准构建于 Overcooked-V2 的高吞吐量 JAX 实现之上。我们的基准包含一个规模庞大、种类多样的队友套件，涵盖强化学习策略与启发式策略，支持可控的训练-测试分布偏移，并为队友生成、学习历史收集、数据集构建以及在线多回合评估提供了可复现的端到端流程。我们在数百万次状态转移上评估了具有代表性的基于历史的 ICRL 算法，包括算法蒸馏（AD）和决策预训练 Transformer（DPT）。结果揭示了显著的局限性：与它们在单智能体领域的成功相反，这些基线方法在临时团队协作任务中表现不佳。

    arXiv:2605.24423v2 Announce Type: replace  Abstract: In-Context Reinforcement Learning (ICRL) has enabled foundation agents to adapt instantaneously to novel tasks, yet its efficacy in Ad-Hoc Teamwork (AHT)-where coordination with unknown partners is required-remains unexplored. To rigorously evaluate this, we introduce a large-scale benchmark ICRL4AHT, built upon a high-throughput JAX implementation of Overcooked-V2. Our benchmark includes a large, diverse teammate suite spanning both RL and heuristic policies, enabling controlled train-test shifts, and provides a reproducible end-to-end pipeline for teammate generation, learning-history collection, dataset construction, and online multi-episode evaluation. We evaluate representative history-conditioned ICRL algorithms, including Algorithm Distillation (AD) and Decision-Pretrained Transformer (DPT), across millions of transitions. Results reveal notable limitations: contrary to their success in single-agent domains, these baselines fa
    
[^321]: 每个组件都是一次查找：用一张线性图统一组件交互、组合与归因

    Every Component Is a Lookup: One Linear Graph for Interaction, Composition and Attribution

    [https://arxiv.org/abs/2605.23393](https://arxiv.org/abs/2605.23393)

    本文提出基于“键值形式”与“加性残差流”两个架构假设，将 Transformer 统一表示为一张线性计算图，并通过 Unpack 反向归因方法，使组件交互、组合路径与词元归因成为同一张图的不同读出结果。

    

    Transformer 的可解释性方法通常围绕相互独立的问题构建：哪些组件发生交互、信息如何路由到输出、以及哪些输入词元做出了贡献。由于这些方法依赖不同的假设，它们的答案难以相互关联。我们认为，两个基于架构的假设足以回答上述三个问题：其一，注意力机制和 MLP 共享一种键值形式 φ(S)U，其中 φ(S) 在值 U 上进行选择；其二，各组件从加性残差流（即各组件输出之和）中读取信息。将这些选择固定在前向传播时的取值上，模型即可转化为一张计算图，而组件交互、组合路径和词元归因都只是这张图的不同读出方式。我们开发了 Unpack——一种在该计算图上执行的反向归因方法，并将每种读出结果与相应的既有基准测试进行验证：交互得分能够预测消融效应……

    arXiv:2605.23393v3 Announce Type: replace-cross  Abstract: Interpretability methods for transformers are typically built around separate questions: which components interact, how information routes to the output, and which input tokens contribute. Because these methods rely on different assumptions, their answers are difficult to relate. We argue that two architecturally motivated assumptions suffice to address all three questions: attention and MLPs share a key-value form, $\phi(S)\,U$, in which $\phi(S)$ selects over values $U$, and components read from an additive residual stream, the sum of component outputs. Holding these selections at their forward-pass values turns the model into a computational graph, of which component interactions, composition paths, and token attribution are different readouts. We develop Unpack, a backward attribution procedure over this graph, and validate each readout against the corresponding established test: interaction scores predict ablation effects 
    
[^322]: 通过跨模态信息流解释与增强大型视觉-语言模型中的情感回路

    Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow

    [https://arxiv.org/abs/2605.21980](https://arxiv.org/abs/2605.21980)

    本文提出基于转向向量的因果归因框架与专用数据集，揭示了大型视觉-语言模型处理情感的三阶段“适应-聚合-执行”机制，并发现视觉情感线索在中间层由情感特异性注意力头聚合、在深层转化为叙述生成的功能解耦现象。

    

    大型视觉-语言模型（LVLMs）在构建共情智能体方面实现了重大飞跃，展现出卓越的情感理解能力。然而，LVLMs如何将抽象的视觉刺激转化为连贯的情感叙述的内部机制在很大程度上仍未被探索，这主要是由于视觉反事实样本的稀缺以及情感表达的弥散性。在本文中，我们通过引入一种专为描述性情感推理定制的、基于转向向量的因果归因框架来填补这一空白。为此，我们构建了一个专门的数据集，以揭示三阶段“适应-聚合-执行”机制背后的情感回路。至关重要的是，我们发现了一种功能解耦现象：视觉情感线索通过情感特异性注意力头在中间层被聚合，随后在深层中转化为叙述生成……

    arXiv:2605.21980v2 Announce Type: replace-cross  Abstract: Large Vision-Language Models (LVLMs) represent a significant leap towards empathetic agents, demonstrating remarkable capabilities in emotion understanding. However, the internal mechanisms governing how LVLMs translate abstract visual stimuli into coherent emotional narratives remain largely unexplored, primarily due to the scarcity of visual counterfactuals and the diffuse nature of emotional expression. In this paper, we bridge this gap by introducing a steering-vector-based causal attribution framework tailored for descriptive emotional reasoning. To this end, we construct a specialized dataset to demystify the emotional circuits underlying the three-stage ``Adapt-Aggregate-Execute'' mechanism. Crucially, we discover a functional decoupling: visual emotional cues are aggregated in middle layers via sentiment-specific attention heads, but are subsequently translated into narrative generation in deep layers through emotion-ge
    
[^323]: BEHAVE：将人类系统作为可观测复杂动力系统与物理AI操作对象的实时建模

    BEHAVE: Real-Time Modeling of Human Systems as Observable Complex Dynamical Systems and Operational Objects for Physical AI

    [https://arxiv.org/abs/2605.12730](https://arxiv.org/abs/2605.12730)

    BEHAVE将相互作用的行人群体建模为可观测、持久的复杂动力系统“人类系统”，揭示超越个体聚合的“操作性涌现”现象，并通过交互证据构建显式的路由与局部动力学模型（J=-D+GP），实现对人人群稳定与失序的实时预测。

    

    机器人可以追踪每一个人，却仍然无法看到这些人所构成的系统。BEHAVE将相互作用的行人群体视为一个复杂动力系统，即“人类系统”——一个可观测的、持久的、关系性的对象，其状态部分由交互结构所承载。因此，这一系统状态既不显式地体现在独立的个体轨迹表示中，也无法被简化为简单的聚合统计量。我们将这一现象称为“操作性涌现”。在公开行人数据上，交互证据使群体判别能力超越仅基于邻近性的方法（AUC从0.896提升至0.933）。在24次瓶颈实验中，基于运行级推断，那些在密度、平均速度、流量和速度离散度上相互匹配的“未来平稳”与“未来失序”时刻，在邻居级组织结构上存在显著差异（p=0.028）。从交互证据K出发，BEHAVE构建了保守路由矩阵P与局部动力学J=-D+GP，将路由与增益和弛豫过程分离开来，使稳定性、临界模态和系统响应成为显式的模型量。

    arXiv:2605.12730v2 Announce Type: replace  Abstract: A robot can track every person and still fail to see the system those people form. BEHAVE treats an interacting human group as a complex dynamical system: a HumanSystem, an observable, persistent, relational object whose state is carried partly by interaction structure. It is therefore neither explicit in independent individual-track representations nor reducible to simple aggregates. We call this operational emergence. On public pedestrian data, interaction evidence improves group discrimination beyond proximity (AUC 0.896->0.933). On 24 bottleneck runs, future-calm and future-breakdown moments matched on density, mean speed, flow and speed dispersion differ in neighbour-level organization under run-level inference (p=0.028).   From interaction evidence K, BEHAVE constructs conservative routing P and local dynamics J=-D+GP, separating routing from gain and relaxation. Stability, critical modes and response become explicit model quan
    
[^324]: 前沿滞后：学术AI评估中能力误述的文献计量审计

    Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation

    [https://arxiv.org/abs/2605.04135](https://arxiv.org/abs/2605.04135)

    该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。

    

    应用领域中的LLM评估往往反映的是在论文发表时就已经被超越的模型。我们观察到一种“发表引导差距”（publication elicitation gap）：即学术论文中所报告结果由哪些AI系统生成，与当前读者合理认为论文所引用的AI系统之间的距离。我们系统性地扫描了OpenAlex中2022年1月1日至2026年4月1日的数据（n = 112,303个LLM关键词匹配），随后识别出被评估的模型（n = 18,574条有效记录）。接着，我们基于Epoch AI能力指数（ECI）——一个LLM综合能力评分——将每个被评估的LLM与前沿LLM进行排名比较。在评估时点，中位数论文所评估的模型在能力上落后于前沿LLM，中位数差距为+10.85 ECI（H1；n = 12,312）。这一差距正在扩大，以每年+5.53 ECI的速度增长（H2，名义95%置信区间 [+5.03, +5.83]）。即使在……（原文摘要在此处截断）

    arXiv:2605.04135v3 Announce Type: replace-cross  Abstract: LLM evaluations in applied domains tend to reflect models that were already outclassed at time of publication. We observe a publication elicitation gap: the distance between the AI systems generating the results reported in an academic paper and the AI systems that a current reader of that paper would reasonably assume are being referenced. We systematically sweep OpenAlex from 2022-01-01 to 2026-04-01 (n = 112,303 LLM keyword matches). Then, we identify what models were evaluated (n = 18,574 admissible records). We then rank each evaluated LLM against a frontier LLM based on the Epoch AI Capabilities Index (ECI), an aggregate LLM capability score. At time of evaluation, the median paper is evaluating models that are behind frontier LLMs in capability, with a median gap of +10.85 ECI (H1; n = 12,312). This gap is growing, increasing at a rate of +5.53 ECI per year (H2, nominal 95% CI [+5.03, +5.83]). The sign holds even in the 
    
[^325]: AgileLog：面向数据流上智能体的可分叉共享日志

    AgileLog: A Forkable Shared Log for Agents on Data Streams

    [https://arxiv.org/abs/2604.14590](https://arxiv.org/abs/2604.14590)

    本文提出 AgileLog——一种支持分叉的共享日志抽象及其系统实现 Bolt，通过新颖的分叉原语为 AI 智能体在数据流上的任务提供避免性能干扰、安全写入的底层支撑。

    

    arXiv:2604.14590v3 公告类型：replace-cross 摘要：在现代数据流系统中，除了传统程序之外，出现了一种能够与流数据交互的新型实体：AI 智能体。与传统程序不同，AI 智能体利用大语言模型（LLM）推理来完成以自然语言指定的、针对流数据的高级任务。遗憾的是，当前的流式系统无法完全支持智能体：它们缺乏避免智能体任务造成性能干扰以及安全处理智能体写入的基本机制。我们认为，作为流式数据核心抽象的共享日志，必须支持创建自身的分叉，而这样一个可分叉的共享日志可以成为智能体作用于流式数据的绝佳基础。我们提出了 AgileLog，一种新的共享日志抽象，为智能体用例提供了新颖的分叉原语。我们设计了 Bolt，一个实现 AgileLog 抽象的系统。Bolt 使用许多新技术来使分叉变得廉价……

    arXiv:2604.14590v3 Announce Type: replace-cross  Abstract: In modern data-streaming systems, alongside traditional programs, a new type of entity has emerged that can interact with streaming data: AI agents. Unlike traditional programs, AI agents use LLM reasoning to accomplish high-level tasks specified in natural language over streaming data. Unfortunately, current streaming systems cannot fully support agents: they lack the fundamental mechanisms to avoid the performance interference caused by agentic tasks and to safely handle agentic writes. We argue that the shared log, the core abstraction underlying streaming data, must support creating forks of itself, and that such a forkable shared log serves as a great substrate for agents acting on streaming data. We propose AgileLog, a new shared log abstraction that provides novel forking primitives for agentic use cases. We design Bolt, a system that implements the AgileLog abstraction. Bolt uses many novel techniques to make forks chea
    
[^326]: IatroBench：一个针对语言模型临床信息遗漏的预注册基准测试

    IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models

    [https://arxiv.org/abs/2604.07709](https://arxiv.org/abs/2604.07709)

    论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。

    

    一个经过强安全训练的模型会为医生提供苯二氮䓬类药物的减量停药方案，却不会为提出同样请求的患者提供。模型本身知道这些信息，但分享多少取决于提问的框架。我们提出了IatroBench，一个在两类伤害维度（作为性伤害与遗漏性伤害）上，通过60个预注册临床场景对6个模型进行评估的基准测试。我们使用Claude Opus 4.6依据一位医生撰写的评分细则对模型回复进行打分，发现其遗漏评分与该医生评分的一致性程度，与另一位医生评分之间的一致性相当。我们发现，当同一病例分别以患者提问和医生会诊两种形式呈现时（两种变体在语体、请求方式以及隐含的治疗医生监督方面也存在差异），我们测试的全部五个模型向医生分享的信息都多于向患者分享的信息。我们将这种现象称为“框架依赖性信息保留”。我们发现平均解耦差距为+0.38……

    arXiv:2604.07709v5 Announce Type: replace  Abstract: A strongly safety-trained model will provide a doctor with a benzodiazepine taper schedule, but not a patient who asks for one. The model knows the information, but how much it shares depends on the framing. We introduce IatroBench, a benchmark that evaluates models on two axes of harm (commission and omission) across 60 pre-registered clinical scenarios and 6 models. We use Claude Opus 4.6 to score model responses against a rubric written by a physician, and find that its omission scores are as well-aligned to the physician's scores as another physician's scores are. We find that when the same case is presented as a patient query and a doctor consultation (the variants also differ in register, request and the supervision a treating physician implies), all five models we test share more information with the doctor than the patient. We term this phenomenon "framing-contingent withholding." We find a mean decoupling gap of +0.38 across
    
[^327]: LiveMathematicianBench：一个基于证明概要的研究级数学推理动态基准测试

    LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches

    [https://arxiv.org/abs/2604.01754](https://arxiv.org/abs/2604.01754)

    提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。

    

    数学推理是人类智力的标志，大型语言模型（LLM）能否有意义地进行数学推理仍然是人工智能和认知科学中的一个核心问题。随着大语言模型越来越多地被整合到科学工作流程中，对其数学能力进行严格评估已成为一种实际需求。现有的基准测试受限于合成环境和数据污染。我们提出了LiveMathematicianBench，这是一个基于模型训练截止日期之后发布的最新arXiv论文构建的、面向研究级数学推理的动态选择题基准测试。通过将评估建立在新发表的定理之上，它提供了一个超越记忆模式的真实测试平台。该基准测试引入了一个包含十三个类别的定理类型逻辑分类体系（例如蕴含、等价、存在性、唯一性），从而实现跨推理形式的细粒度评估。它采用了一种证明——

    arXiv:2604.01754v2 Announce Type: replace-cross  Abstract: Mathematical reasoning is a hallmark of human intelligence, and whether large language models (LLMs) can meaningfully perform it remains a central question in artificial intelligence and cognitive science. As LLMs are increasingly integrated into scientific workflows, rigorous evaluation of their mathematical capabilities becomes a practical necessity. Existing benchmarks are limited by synthetic settings and data contamination. We present LiveMathematicianBench, a dynamic multiple-choice benchmark for research-level mathematical reasoning built from recent arXiv papers published after model training cutoffs. By grounding evaluation in newly published theorems, it provides a realistic testbed beyond memorized patterns. The benchmark introduces a thirteen-category logical taxonomy of theorem types (e.g., implication, equivalence, existence, uniqueness), enabling fine-grained evaluation across reasoning forms. It employs a proof-
    
[^328]: 通过黑盒、面向漏洞的扫描检测代码生成大语言模型中的数据投毒

    Detecting Data Poisoning in Code Generation LLMs via Black-Box, Vulnerability-Oriented Scanning

    [https://arxiv.org/abs/2603.17174](https://arxiv.org/abs/2603.17174)

    CodeScan是首个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，通过分析多次生成结果的结构相似性和迭代发散分析来有效检测诱发不安全代码生成的数据投毒攻击。

    

    代码生成大语言模型（LLM）正日益被集成到现代软件开发工作流程中。最近的研究表明，这些模型容易受到后门和投毒攻击，从而生成不安全的代码，但有效的防御手段仍然有限。现有的扫描方法依赖于标记级别的生成一致性来反推攻击目标，这种方法对于源代码是无效的，因为相同的语义可以以不同的语法形式出现。我们提出了CodeScan，这是第一个用于审计代码生成大语言模型的黑盒、面向特定漏洞的扫描框架，其假设防御者指定目标漏洞类别并提供相应的与任务相关的提示。CodeScan通过分析在不同干净提示条件下多次生成结果之间的结构相似性来识别攻击目标。它将迭代发散分析与抽象语法树（AST）……

    arXiv:2603.17174v2 Announce Type: replace-cross  Abstract: Code generation large language models (LLMs) are increasingly integrated into modern software development workflows. Recent work has shown that these models are vulnerable to backdoor and poisoning attacks that induce the generation of insecure code, yet effective defenses remain limited. Existing scanning approaches rely on token-level generation consistency to invert attack targets, which is ineffective for source code where identical semantics can appear in diverse syntactic forms. We present CodeScan, the first black-box, vulnerability-specific scanning framework for auditing code generation LLMs, assuming that the defender specifies the target vulnerability classes and provides corresponding task-relevant prompts. CodeScan identifies attack targets by analyzing structural similarities across multiple generations conditioned on different clean prompts. It combines iterative divergence analysis with abstract syntax tree (AST
    
[^329]: 通过混合大语言模型（LLM）-符号规划与LLM引导的强化学习实现新颖性适应

    Novelty Adaptation Through Hybrid Large Language Model (LLM)-Symbolic Planning and LLM-guided Reinforcement Learning

    [https://arxiv.org/abs/2603.11351](https://arxiv.org/abs/2603.11351)

    该论文提出了一种融合符号规划、强化学习与大语言模型的神经符号架构，利用LLM的常识推理能力识别缺失算子、生成计划并编写奖励函数，使机器人能够有效适应开放世界环境中的新颖物体。

    

    在动态开放世界环境中，自主智能体经常会遇到阻碍其找到实现目标的计划的新颖事物。具体而言，当机器人的规划域缺乏使其能够与环境中的新物体进行适当交互的算子时，传统的符号规划器无法生成计划。我们提出了一种神经符号架构，该架构集成了符号规划、强化学习和大语言模型（LLM），以学习如何处理新颖物体。特别是，我们利用LLM的常识推理能力来识别缺失的算子，与符号AI规划器协同生成计划，并编写奖励函数来引导强化学习智能体学习针对新识别算子的控制策略。我们的方法在算子发现以及连续机器人领域的算子学习方面均优于当前最先进的方法。

    arXiv:2603.11351v2 Announce Type: replace-cross  Abstract: In dynamic open-world environments, autonomous agents often encounter novelties that hinder their ability to find plans to achieve their goals. Specifically, traditional symbolic planners fail to generate plans when the robot's planning domain lacks the operators that enable it to interact appropriately with novel objects in the environment. We propose a neuro-symbolic architecture that integrates symbolic planning, reinforcement learning, and a large language model (LLM) to learn how to handle novel objects. In particular, we leverage the common sense reasoning capability of the LLM to identify missing operators, generate plans with the symbolic AI planner, and write reward functions to guide the reinforcement learning agent in learning control policies for newly identified operators. Our method outperforms the state-of-the-art methods in operator discovery as well as operator learning in continuous robotic domains.Our webpage
    
[^330]: 脚手架之下的安全性：评估条件如何塑造所测得的安全表现

    Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety

    [https://arxiv.org/abs/2603.10044](https://arxiv.org/abs/2603.10044)

    评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。

    

    安全基准测试通常针对“裸”模型——即接收提示并输出响应的模型——进行，但现实世界的部署会将这些模型“包裹”在复杂的脚手架中。这些脚手架对基准测试所衡量的模型安全性究竟有多大影响？我们在四个预先注册的安全基准上，使用直接 API 以及三种脚手架（ReAct、多智能体和 map-reduce）测试了六个领先模型，共进行了 62,808 次评分评估。研究发现，安全性的测量方式比脚手架本身更为重要：对于其他方面完全相同的基准题目，使用选择题还是开放式问题格式，会使测得的安全性相差 5-20 个百分点。由于这两种格式采用不同的评分方法（答案提取与 LLM 评审），这一差距源于测量方式而非模型潜在安全性的差异。若使用启发式方法对模型拒绝行为进行分类，将在五种情况下得出不同的结论。基准的选择可解释结果变异的 19.3%。

    arXiv:2603.10044v3 Announce Type: replace  Abstract: Safety benchmarks usually test "bare" models that receive prompts and output responses, but real-world deployments "wrap" those models in complex scaffolds. How much do these scaffolds affect model safety as measured by benchmarks? We test six leading models on four pre-registered safety benchmarks with a direct API and three scaffolds: ReAct, multi-agent, and map-reduce. We conducted 62,808 scored evaluations. How safety is measured matters more than scaffolding does: we find that using a multiple choice vs. open-ended format for otherwise-identical benchmark items changes measured safety by 5-20 percentage points (pp). The two formats are scored with different methods (answer extraction and an LLM judge), so the gap is due to measurement rather than differences in latent safety. Using a heuristic to classify model refusals would have led to different findings in five cases. Benchmark choice explains 19.3% of the variation in outcom
    
[^331]: 基于最优序分数搜索的时间序列因果结构学习

    Learning Causal Structure of Time Series using Best Order Score Search

    [https://arxiv.org/abs/2603.05370](https://arxiv.org/abs/2603.05370)

    本文提出TS-BOSS算法，将最优序分数搜索（BOSS）扩展到多变量时间序列的动态贝叶斯网络因果结构学习中，兼具可扩展性与理论保证。

    

    从观测数据中学习因果结构是许多科学和政策领域的核心问题，但许多学科中常见的时间序列设置由于时间依赖性而带来了若干挑战。在本文中，我们聚焦于基于分数的多变量时间序列因果发现，并提出了TS-BOSS，这是最近提出的最佳序分数搜索（Best Order Score Search, BOSS）（Andrews et al. 2023）的时间序列扩展版本。TS-BOSS在动态贝叶斯网络结构上执行基于置换的搜索，同时利用生长-收缩树来缓存中间分数计算，从而在时间序列设置中保留了BOSS在静态设置中的可扩展性和出色的实证性能。我们提供了理论保证，证明了在适当假设下TS-BOSS的可靠性，并给出了一个中间结果，将基于置换方法的经典子图最小性结果扩展到了动态（时间序列）设置中。我们的实验……

    arXiv:2603.05370v2 Announce Type: replace-cross  Abstract: Causal structure learning from observational data is central to many scientific and policy domains, but the time series setting common to many disciplines poses several challenges due to temporal dependence. In this paper we focus on score-based causal discovery for multivariate time series and introduce TS-BOSS, a time series extension of the recently proposed Best Order Score Search (BOSS) (Andrews et al. 2023). TS-BOSS performs a permutation-based search over dynamic Bayesian network structures while leveraging grow-shrink trees to cache intermediate score computations, preserving the scalability and strong empirical performance of BOSS in the static setting. We provide theoretical guarantees establishing the soundness of TS-BOSS under suitable assumptions, and we present an intermediate result that extends classical subgraph minimality results for permutation-based methods to the dynamic (time series) setting. Our experimen
    
[^332]: MOOSEnger：面向 MOOSE 生态系统的仿真感知 AI 智能体框架

    MOOSEnger: A Simulation-Aware AI Agent Framework for the MOOSE Ecosystem

    [https://arxiv.org/abs/2603.04756](https://arxiv.org/abs/2603.04756)

    MOOSEnger 是一个面向 MOOSE 生态系统的仿真感知 AI 智能体框架，通过集成知识检索、HIT 语法解析、验证诊断与求解器反馈的“生成—检查—修复—运行”工作流，克服了大语言模型一次性生成仿真输入文件时易因微小错误而执行失败、且执行成功也不代表科学正确的问题。

    

    MOOSEnger 是一个面向多物理场面向对象仿真环境（MOOSE）生态系统的建模与仿真 AI 智能体框架，其核心是一个仿真感知的支撑框架，该框架将可替换的推理模型与扎根的领域知识、经修订的仿真工件、MOOSE 专属验证以及可执行的求解器反馈相结合。这一外围系统解决了一次性大语言模型生成的一个核心局限：微小的语法、模式、引用或求解器配置错误就可能阻止一个看似合理的输入文件成功执行，而仅仅成功执行并不能确立科学上的正确性。MOOSEnger 的仿真感知支撑框架集成了 MOOSE 知识检索、支持层次化输入文本（HIT）的解析、语法元数据、语言服务器诊断、修订控制的编写，以及本地或基于 MCP 的验证与执行，构成“生成—检查—修复—运行”的工作流，并将证据绑定到每（个环节）……

    arXiv:2603.04756v3 Announce Type: replace  Abstract: MOOSEnger is a modeling and simulation AI agent framework for the Multiphysics Object-Oriented Simulation Environment (MOOSE) ecosystem, built around a simulation-aware harness that combines an interchangeable reasoning model with grounded domain knowledge, revised simulation artifacts, MOOSE-specific validation, and executable solver feedback. This surrounding system addresses a central limitation of one-shot large language model generation: small syntax, schema, reference, or solver-configuration errors can prevent a plausible input from executing, while successful execution alone does not establish scientific correctness. MOOSEnger's simulation-aware harness integrates MOOSE knowledge retrieval, Hierarchical Input Text (HIT)-aware parsing, syntax metadata, language-server diagnostics, revision-controlled authoring, and local or MCP-backed validation and execution in a generate-check-repair-run workflow that binds evidence to each 
    
[^333]: MPFlow：用于零样本MRI重建的多模态后验引导流匹配

    MPFlow: Multi-modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction

    [https://arxiv.org/abs/2603.03710](https://arxiv.org/abs/2603.03710)

    提出MPFlow零样本多模态重建框架，基于修正流在推理时融合辅助MRI模态，并通过PAMRI自监督跨模态预训练实现引导采样，无需重训生成先验即可提升MRI重建的解剖保真度并抑制幻觉。

    

    零样本MRI重建依赖于生成先验，但单模态无条件先验在严重病态条件下会产生幻觉。在许多临床工作流程中，互补的MRI采集数据（例如高质量的结构扫描）是常规可获得的，然而现有的重建方法缺乏利用这些额外信息的机制。我们提出了MPFlow，一个基于修正流的零样本多模态重建框架，它在推理阶段引入辅助MRI模态，无需重新训练生成先验即可提升解剖保真度。跨模态引导由我们提出的自监督预训练策略——块级多模态MR图像预训练实现，该策略学习跨模态的共享表示。采样过程由数据一致性和基于预训练PAMRI的跨模态特征对齐共同引导，系统地抑制内在和外在的伪影与幻觉（原文此处截断）。

    arXiv:2603.03710v4 Announce Type: replace-cross  Abstract: Zero-shot MRI reconstruction relies on generative priors, but single-modality unconditional priors produce hallucinations under severe ill-posedness. In many clinical workflows, complementary MRI acquisitions (e.g. high-quality structural scans) are routinely available, yet existing reconstruction methods lack mechanisms to leverage this additional information. We propose MPFlow, a zero-shot multi-modal reconstruction framework built on rectified flow that incorporates auxiliary MRI modalities at inference time without retraining the generative prior to improve anatomical fidelity. Cross-modal guidance is enabled by our proposed self-supervised pretraining strategy, Patch-level Multi-modal MR Image Pretraining (PAMRI), which learns shared representations across modalities. Sampling is jointly guided by data consistency and cross-modal feature alignment using pre-trained PAMRI, systematically suppressing intrinsic and extrinsic 
    
[^334]: SHINE：面向脑电图（EEG）与脑磁图（MEG）的序列化分层集成网络

    SHINE: Sequential Hierarchical Integration Network for EEG and MEG

    [https://arxiv.org/abs/2602.23960](https://arxiv.org/abs/2602.23960)

    提出了面向EEG和MEG的序列化分层集成网络SHINE，通过残差传感器适配器、膨胀块时间建模和目标与时间自适应的门控机制，在全部八个数据集-指标组合上实现了语音包络和梅尔频谱重建的最佳性能。

    

    自然语音在大脑中如何表征是认知神经科学面临的一项重大挑战。从脑电图（EEG）和脑磁图（MEG）中重建语音包络和梅尔频谱图，为研究语音的时间和频谱结构提供了一种具有时间分辨率的方法。与语音相关的神经活动跨越多个传感器和多个时间尺度；如何在提取这些表征的同时，根据不同的声学目标自适应地调整上下文的使用，是语音重建中的一个核心问题。我们提出了SHINE，一种面向EEG和MEG的序列化分层集成网络。残差传感器适配器统一了输入维度，中间的膨胀块状态保留了时间深度信息，而一个依赖于目标和时间的门控机制融合了局部分层预测与注意力增强的上下文预测。在两个EEG数据集和两个MEG数据集上，SHINE在全部八个数据集-指标组合中，于九种局部基线方法中取得了最高的平均包络和平均梅尔皮尔逊相关系数。

    arXiv:2602.23960v2 Announce Type: replace-cross  Abstract: How natural speech is represented in the brain constitutes a major challenge for cognitive neuroscience. Reconstructing the speech envelope and Mel spectrogram from EEG and MEG provides a time-resolved way to study its temporal and spectral structure. Speech-related neural activity spans sensors and temporal scales; extracting these representations while adapting the use of context to each acoustic target is a central problem in speech reconstruction. We propose SHINE, a Sequential Hierarchical Integration Network for EEG and MEG. A residual sensor adapter unifies input dimensions, intermediate dilated-block states retain temporal depth, and a target- and time-dependent gate fuses local hierarchical and attention-enhanced context predictions. Across two EEG and two MEG datasets, SHINE has the highest mean envelope and mean-Mel Pearson correlations among nine local baseline implementations on all eight dataset-metric combination
    
[^335]: 解读机器学习决策：面向大规模排序系统的智能体推理框架

    Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System

    [https://arxiv.org/abs/2602.18640](https://arxiv.org/abs/2602.18640)

    本文提出GEARS框架，将大规模排序优化重构为可编程实验环境中的自主发现过程，通过专门的智能体技能封装排序专家知识，让操作者只需通过高层产品意图即可引导系统，从而突破将模糊产品意图转化为可验证假设的工程瓶颈。

    

    现代大规模排序系统在竞争目标、运营约束和不断演进的产品需求构成的复杂环境中运行。该领域的进展日益受到工程上下文约束的瓶颈制约，即如何将模糊的产品意图转化为合理、可执行、可验证的假设这一艰巨过程，而非仅仅受限于建模技术本身。我们提出了GEARS（面向智能体排序系统的生成式引擎），这是一个将排序优化重新定义为可编程实验环境中自主发现过程的框架。GEARS不再将优化视为静态的模型选择，而是利用专门的智能体技能将排序专家知识封装为可复用的推理能力，使操作者能够通过高层意图的vibe个性化来引导系统。此外，为确保生产可靠性，该框架引入了验证机制……

    arXiv:2602.18640v3 Announce Type: replace  Abstract: Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous product intent into reasonable, executable, verifiable hypotheses, rather than by modeling techniques alone. We present GEARS (Generative Engine for Agentic Ranking Systems), a framework that reframes ranking optimization as an autonomous discovery process within a programmable experimentation environment. Rather than treating optimization as static model selection, GEARS leverages Specialized Agent Skills to encapsulate ranking expert knowledge into reusable reasoning capabilities, enabling operators to steer systems via high-level intent vibe personalization. Furthermore, to ensure production reliability, the framework incorporates validat
    
[^336]: VLANeXt：构建强大VLA模型的配方

    VLANeXt: Recipes for Building Strong VLA Models

    [https://arxiv.org/abs/2602.18532](https://arxiv.org/abs/2602.18532)

    本文通过统一框架系统剖析VLA设计空间，提炼出12个关键发现，形成了构建强大VLA模型的实用配方。

    

    摘要：随着大型基础模型的兴起，视觉-语言-动作模型（VLAs）应运而生，利用视觉-语言模型（VLMs）强大的视觉和语言理解能力进行通用策略学习。然而，当前VLA领域仍然分散且探索性较强。尽管许多团队提出了各自的VLA模型，但训练协议和评估设置的不一致性使得难以确定哪些设计选择真正重要。为了给这一不断发展的领域带来结构，我们在统一框架和评估设置下重新审视了VLA设计空间。从一个类似于RT-2（VLA的起源）的简单VLA基线出发，我们系统地剖析了三个维度的设计选择：基础组件、感知要素和动作建模视角。通过这项研究，我们提炼出12个关键发现，共同构成了构建强大VLA模型的实用配方。最终结果如下。

    arXiv:2602.18532v3 Announce Type: replace-cross  Abstract: Following the rise of large foundation models, Vision-Language-Action models (VLAs) emerged, leveraging strong visual and language understanding from Vision-Language Models for general-purpose policy learning. Yet, the current VLA landscape remains fragmented and exploratory. Although many groups have proposed their own VLA models, inconsistencies in training protocols and evaluation settings make it difficult to identify which design choices truly matter. To bring structure to this evolving space, we reexamine the VLA design space under a unified framework and evaluation setup. Starting from a simple VLA baseline similar to RT-2, which is the origin of VLA, we systematically dissect design choices along three dimensions: foundational components, perception essentials, and action modelling perspectives. From this study, we distill 12 key findings that together form a practical recipe for building strong VLA models. The outcome 
    
[^337]: TabSieve：面向表格预测的显式表内证据选择

    TabSieve: Explicit In-Table Evidence Selection for Tabular Prediction

    [https://arxiv.org/abs/2602.11700](https://arxiv.org/abs/2602.11700)

    TabSieve提出了一种先选择后预测的表格预测框架，通过显式选择表内证据行、构建40K规模的合成微调数据集TabSieve-SFT-40K，以及结合分离奖励的强化学习方法TAB-GRPO，实现了可审计且稳健的表格预测。

    

    表格预测可以利用表内行作为少样本证据，但现有的表格模型通常执行实例级推理，且基于大语言模型的提示方法往往较为脆弱。模型无法持续稳定地利用相关行，而嘈杂的上下文还可能降低性能。为应对这一挑战，我们提出了TabSieve，这是一个“先选择后预测”的框架，使证据的使用变得显式且可审计。给定一个表格和一个查询行，TabSieve首先选出一小组信息量大的行作为证据，然后在所选证据的条件下预测缺失的目标值。为实现这一能力，我们通过使用强大的教师模型并施加严格过滤，从331个真实表格中合成高质量推理轨迹，构建了TabSieve-SFT-40K数据集。此外，我们引入了TAB-GRPO，这是一种强化学习方案，通过分离的奖励信号共同优化证据选择与预测正确性，并稳定了混合……（原文摘要至此处被截断）

    arXiv:2602.11700v2 Announce Type: replace-cross  Abstract: Tabular prediction can benefit from in-table rows as few-shot evidence, yet existing tabular models typically perform instance-wise inference and LLM-based prompting is often brittle. Models do not consistently leverage relevant rows, and noisy context can degrade performance. To address this challenge, we propose TabSieve, a select-then-predict framework that makes evidence usage explicit and auditable. Given a table and a query row, TabSieve first selects a small set of informative rows as evidence and then predicts the missing target conditioned on the selected evidence. To enable this capability, we construct TabSieve-SFT-40K by synthesizing high-quality reasoning trajectories from 331 real tables using a strong teacher model with strict filtering. Furthermore, we introduce TAB-GRPO, a reinforcement learning recipe that jointly optimizes evidence selection and prediction correctness with separate rewards, and stabilizes mix
    
[^338]: 基于事前稀疏化的近预言机KV选择方法用于长上下文推理

    Near-Oracle KV Selection via Pre-hoc Sparsity for Long-Context Inference

    [https://arxiv.org/abs/2602.08329](https://arxiv.org/abs/2602.08329)

    该论文提出事前稀疏化方法PrHS，在注意力打分之前进行KV选择以避免事后启发式方法的后验偏差，并推导出仅依赖丢弃质量的互信息损失上界，从而为长上下文LLM推理提供具有显式精度控制的近预言机KV选择。

    

    大语言模型（LLM）推理的一个核心瓶颈是对不断增长的键值（KV）缓存进行注意力计算的开销。尽管近预言机的top-k KV选择能够在大幅减少计算和带宽的同时保持稠密注意力的质量，但现有的稀疏方法通常依赖事后启发式方法，即以观察到的注意力或代理分数为条件的选择器。这种条件化引入了后验偏差：它往往会扭曲真实的token重要性并遗漏显著的token，从而损害长程推理能力。为解决这一问题，我们提出事前稀疏化方法（Pre-hoc Sparsity, PrHS），它在注意力打分之前选择KV条目，并提供显式的精度控制。设被丢弃条目的注意力质量为delta（丢弃质量），通过从边际分布到互信息的分析，我们推导出了一个仅依赖于丢弃质量的互信息损失上界。这一关系解释了现有方法失败的……

    arXiv:2602.08329v2 Announce Type: replace-cross  Abstract: A core bottleneck in large language model (LLM) inference is the cost of attending over the ever-growing key-value (KV) cache. Although near-oracle top-k KV selection can preserve the quality of dense attention while sharply reducing computation and bandwidth, existing sparse methods generally rely on posterior heuristics, i.e., selectors conditioned on observed attention or proxy scores. Such conditioning introduces posterior bias: it tends to distort true token importance and miss salient tokens, thereby impairing long-range reasoning. To tackle this problem, we propose Pre-hoc Sparsity (PrHS), which selects KV entries before attention scoring and provides explicit accuracy control. Let the attention mass of discarded entries be delta (the dropped mass). Through a marginal-to-mutual-information analysis, we derive an upper bound on the mutual-information loss that depends only on the dropped mass. This relation explains failu
    
[^339]: TIDE：面向自改进大语言模型推理的时间增量草稿引擎

    TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

    [https://arxiv.org/abs/2602.05145](https://arxiv.org/abs/2602.05145)

    TIDE 通过复用推理过程中的中间隐藏状态在线增量训练草稿模型，并结合自适应运行时控制与异构 GPU 集群调度，在不增加额外目标模型开销的情况下实现了高达 1.66 倍的 LLM 推理吞吐量提升。

    

    推测解码可以显著加速大语言模型推理，但由于工作负载不断演变，在实际中充分实现其收益具有挑战性。我们提出了 TIDE（Temporal Incremental Draft Engine，时间增量草稿引擎），这是一个服务引擎原生的框架，将在线草稿模型适配直接集成到高性能大语言模型推理系统中。TIDE 复用目标模型在推理过程中产生的中间隐藏状态作为草稿模型适配的训练信号，从而避免了额外的目标模型计算和运行时开销。它采用自适应运行时控制，仅在有益时才激活推测解码和草稿模型训练。TIDE 还通过将推理和训练任务映射到合适的 GPU 类别来充分利用异构集群。在多种真实世界工作负载上，TIDE 相比无推测解码的基线实现了高达 1.66 倍的吞吐量提升，并在静态草稿模型性能下降的分布不对齐工作负载上恢复了性能表现。

    arXiv:2602.05145v2 Announce Type: replace-cross  Abstract: Speculative decoding can substantially accelerate LLM inference, but realizing its benefits in practice is challenging due to evolving workloads. We present TIDE (Temporal Incremental Draft Engine), a serving-engine-native framework that integrates online draft adaptation directly into high-performance LLM inference systems. TIDE reuses target model's intermediate hidden states generated during inference as training signals for draft adaptation, thereby avoiding additional target model computation and serving-time overhead. It employs adaptive runtime control to activate speculation and draft model training only when beneficial. TIDE exploits heterogeneous clusters by mapping inference and training to appropriate GPU classes. Across diverse real-world workloads, TIDE achieves up to 1.66$\times$ throughput over no-speculation baselines while recovering performance on misaligned workloads where static draft models degrade through
    
[^340]: HERMES：一种基于视觉语言模型的面向长尾自动驾驶的整体式端到端风险感知多模态具身系统

    HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision-Language Models for Long-Tail Autonomous Driving

    [https://arxiv.org/abs/2602.00993](https://arxiv.org/abs/2602.00993)

    HERMES提出了一种风险感知的端到端多模态自动驾驶框架，通过基础模型辅助标注构建长尾场景与规划上下文，并利用三模态驾驶模块融合多视角视觉、自车运动与长尾语义指令，将长尾语义知识显式融入轨迹规划，从而提升混合交通长尾场景下的安全规划能力。

    

    端到端自动驾驶模型日益受益于大型视觉语言模型在语义理解方面的能力，但在长尾条件下的安全可靠规划仍然具有挑战性，尤其是在涉及异质道路使用者和罕见安全关键交互的混合交通环境中。本文提出了HERMES，一个整体式风险感知端到端多模态驾驶框架，它将长尾语义知识显式地融入轨迹规划中。HERMES采用基础模型辅助的标注流水线来构建结构化的长尾场景上下文和长尾规划上下文，捕获以危险为中心的场景信息、机动意图以及风险感知的规划指导。随后，三模态驾驶模块通过意图感知与风险感知的条件机制，整合多视角视觉观测、历史自车运动和长尾语义指令以生成轨迹。广泛的（实验验证……摘要此处截断）

    arXiv:2602.00993v2 Announce Type: replace-cross  Abstract: End-to-end autonomous driving models increasingly benefit from large vision-language models for semantic understanding, yet safe and reliable planning under long-tail conditions remains challenging, particularly in mixed-traffic environments involving heterogeneous road users and rare safety-critical interactions. This paper proposes HERMES, a holistic risk-aware end-to-end multimodal driving framework that explicitly incorporates long-tail semantic knowledge into trajectory planning. HERMES employs a foundation-model-assisted annotation pipeline to construct structured Long-Tail Scene Context and Long-Tail Planning Context, capturing hazard-centric scene information, maneuver intent, and risk-aware planning guidance. A Tri-Modal Driving Module then integrates multi-view visual observations, historical ego-motion, and long-tail semantic instructions through intent- and risk-aware conditioning for trajectory generation. Extensiv
    
[^341]: 超越提示方法：通过Logit空间集成实现语音大语言模型的高效鲁棒上下文偏置（LOGIC）

    Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)

    [https://arxiv.org/abs/2601.15397](https://arxiv.org/abs/2601.15397)

    本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。

    

    新实体的快速涌现——受文化变迁、流行趋势演变和个性化用户数据的驱动——对现有的语音大语言模型（Speech LLMs）构成了重大挑战。虽然这些模型在通用对话任务中表现出色，但其静态训练知识限制了它们识别特定领域术语（如联系人姓名、播放列表或技术行话）的能力。现有解决方案主要依赖提示方法，但其可扩展性较差：随着实体列表的增长，提示方法会遇到上下文窗口限制、推理延迟增加以及“迷失在中间”现象。另一种替代方法——生成式错误纠正（GEC）——试图通过后处理重写转录文本，但经常出现“过度纠正”问题，引入从未被说出的实体的幻觉。在这项工作中，我们介绍了LOGIC（用于上下文偏置的Logit空间集成），一种……

    arXiv:2601.15397v3 Announce Type: replace-cross  Abstract: The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken.   In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an 
    
[^342]: IDRBench：深度研究智能体交互能力基准测试

    IDRBench: Benchmarking the Interactive Capabilities of Deep Research Agents

    [https://arxiv.org/abs/2601.06676](https://arxiv.org/abs/2601.06676)

    IDRBench是首个评估深度研究智能体交互能力的基准，通过受控澄清机会比较自主与交互式工作流，揭示了及时与用户交互对提升研究报告质量的重要性。

    

    基于大语言模型（LLM）的深度研究智能体能够执行多步推理、网页探索和长篇报告生成。在这些长时程工作流中，早期偏离用户意图可能会误导研究方向，并将偏差传播到规划、搜索和综合的各个环节，因此及时的交互至关重要。然而，现有基准主要将深度研究视为静态的输入-输出任务，忽视了智能体引导和利用用户反馈的能力。我们提出了IDRBench，这是一个用于评估交互式深度研究的基准，提供了受控的澄清机会。在统一的工作流程和分阶段交互预算下，IDRBench比较了自主和交互式轨迹，通过任务相关报告一致性的变化来衡量交互收益，并通过交互轮数和token数量来衡量交互成本。在100个任务上对七个专有和开源权重LLM进行的综合实验表明，交互……

    arXiv:2601.06676v3 Announce Type: replace-cross  Abstract: Large Language Model (LLM)-based deep research agents perform multi-step reasoning, web exploration, and long-form report generation. In these long-horizon workflows, early deviations from user intent can misdirect research and propagate through planning, search, and synthesis, making timely interaction essential. However, existing benchmarks primarily treat deep research as a static input-output task, overlooking agents' ability to elicit and use user feedback. We introduce IDRBench, a benchmark for evaluating interactive deep research with controlled opportunities for clarification. Within a common workflow and stage-wise interaction budget, IDRBench compares autonomous and interactive trajectories, measuring interaction benefit through changes in task-specific report alignment and interaction cost through turns and tokens. Comprehensive experiments on 100 tasks with seven proprietary and open-weight LLMs show that interactio
    
[^343]: SPARQL-LLM：从自然语言问题实时生成SPARQL查询

    SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions

    [https://arxiv.org/abs/2512.14277](https://arxiv.org/abs/2512.14277)

    SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。

    

    大语言模型的出现正在推动新方法的涌现，这些方法有望更好地应对从自然语言生成结构化查询（如SPARQL查询）的挑战。然而，这些新方法大多只关注响应准确性，而忽略了其他评估标准，例如生成SPARQL查询的运行时间和成本。因此，它们往往不具备生产就绪性，也难以在真实世界的知识图谱上以良好的准确率进行部署。为了缓解这些问题，本文描述并系统评估了SPARQL-LLM，这是一种开源且与三元组存储无关的方法，由轻量级元数据驱动，能够从自然语言文本生成SPARQL查询。首先，我们描述了其架构，该架构由用于元数据索引、提示构建以及查询生成与执行的专用组件组成。然后，我们基于一项最先进的挑战对其进行了评估……

    arXiv:2512.14277v2 Announce Type: replace-cross  Abstract: The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy while ignoring other evaluation criteria, such as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over real-world knowledge graphs with good accuracy. To mitigate these issues, in this paper, we describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge wi
    
[^344]: 基于视觉语言适配的手写体阿尔茨海默病筛查中的跨任务泛化

    Cross-Task Generalization in Handwriting-Based Alzheimer's Screening via Vision Language Adaptation

    [https://arxiv.org/abs/2511.05841](https://arxiv.org/abs/2511.05841)

    该论文提出轻量级跨层融合适配器（CLFA）框架，将CLIP视觉-语言模型重新用于基于手写的阿尔茨海默病筛查，并系统研究了手写任务类型对诊断性能及跨任务泛化能力的影响。

    

    阿尔茨海默病（AD）是一种常见的神经退行性疾病，早期检测至关重要。手写行为可能受到轻微运动和认知衰退的影响，为AD筛查提供了一个非侵入性且经济高效的窗口。现有的基于手写的AD研究大多依赖于在线轨迹和手工设计的特征，而手写任务类型对诊断性能和跨任务泛化的影响仍未得到充分探索。与此同时，大规模视觉-语言模型在自然图像异常检测以及多种医学模态（如胸部X光和脑部MRI）中展现了强大的迁移与适应能力。然而，基于手写的疾病检测在这一范式下仍属空白。为填补这一空白，我们提出了一种轻量级的跨层融合适配器（CLFA）框架，将对比语言-图像预训练（CLIP）重新应用于基于手写的……（原文摘要在此处截断）

    arXiv:2511.05841v2 Announce Type: replace-cross  Abstract: Alzheimer's disease (AD) is a prevalent neurodegenerative disorder for which early detection is critical. Handwriting, which can be disrupted by subtle motor and cognitive decline, provides a non-invasive and cost-effective window for AD screening. Existing handwriting-based AD studies mostly rely on online trajectories and hand-crafted features, while the influence of handwriting task type on diagnostic performance and cross-task generalization remains underexplored. Meanwhile, large-scale vision--language models have demonstrated strong transfer and adaptation ability in natural-image anomaly detection and several medical modalities, such as chest X-ray and brain MRI. However, handwriting-based disease detection remains unexplored within this paradigm. To address this gap, we introduce a lightweight Cross-Layer Fusion Adapter (CLFA) framework that repurposes Contrastive Language--Image Pre-training (CLIP) for handwriting-base
    
[^345]: 通过模块社区揭示大语言模型的认知模式

    Unraveling the cognitive patterns of Large Language Models through module communities

    [https://arxiv.org/abs/2508.18192](https://arxiv.org/abs/2508.18192)

    该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。

    

    大语言模型（LLMs）通过从科学发现、医学诊断到聊天机器人等广泛应用，在科学、工程和社会领域取得了重大进展，重塑了我们的世界。尽管它们无处不在且功能强大，但LLM的底层机制仍隐藏在数十亿参数和复杂结构之中，使其内部架构和认知过程难以理解。我们通过借鉴理解生物系统中新兴认知的方法来填补这一空白，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，开创了基础模型分析的新范式。模块社区中的技能分布表明，虽然LLM并不严格对应于特定生物系统中所观察到的聚焦特化现象，但它们表现出独特的模块社区，其涌现的技能模式部分模仿了生物学中的认知组织方式。

    arXiv:2508.18192v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mi
    
[^346]: 对话DNA：人类与AI对话的视觉语言与交互式图集

    Conversational DNA: A Visual Language and Interactive Atlas of Human and AI Dialogue

    [https://arxiv.org/abs/2508.07520](https://arxiv.org/abs/2508.07520)

    本文提出“对话DNA”，一种通过说话者链、话步标记和有向配对来可视化人类与AI对话结构的视觉语言与交互式图集，其引入的目标对应关系使对话结构检索的precision@5从58.8%显著提升至77.2%。

    

    当对话参与者彼此交叉说话时，是什么让对话保持完整？主题图谱提供了一种视角，但贡献之间的关系仍难以检视。我们提出了对话DNA（Conversational DNA），一种用于探索人类与AI对话的视觉语言和交互式图集。说话者链保留参与信息，交流基础标记话步，有向配对将回应连接到其目标。可调节的螺旋几何结构使说话者切换、回应距离和贡献长度清晰可见。在包含157万条源记录的八个语料库中，该图集映射了151,489个已索引的对话片段，并将群组比较与源转录文本、局部结构对齐以及记录的备选回复相连接。在189个留出的Molweni模式查询中，添加目标对应关系使精确标注结构的precision@5从58.8%提升至77.2%。案例解读展示了交错的参与模式。

    arXiv:2508.07520v2 Announce Type: replace-cross  Abstract: What makes a conversation hold together when its participants speak across one another? Topic maps offer one view, but they leave the relationships between contributions difficult to inspect. We present Conversational DNA, a visual language and interactive atlas for exploring human and AI dialogue. Speaker strands preserve participation, communicative bases mark moves, and directed pairings connect responses to their targets. Adjustable helix geometry makes speaker switching, response distance, and contribution length visible. Across eight corpora containing 1.57 million source records, the atlas maps 151,489 indexed episodes and connects cohort comparison to source transcripts, local structural alignment, and recorded reply alternatives. On 189 held-out Molweni motif queries, adding target correspondence improves precision@5 from 58.8% to 77.2% for exact annotated structure. Case readings illustrate interleaved participation, 
    
[^347]: WebArxiv：一个用于评估多模态网络智能体在arXiv任务上表现的可复现基准

    WebArxiv: A Reproducible Benchmark for Evaluating Multimodal Web Agents on arXiv Tasks

    [https://arxiv.org/abs/2507.00938](https://arxiv.org/abs/2507.00938)

    WebArxiv是一个基于arXiv静态快照构建的可复现基准，包含510个具有确定性答案的时间不变任务，用于评估多模态网络智能体在多约束论文检索、细粒度内容提取和跨论文比较等学术任务上的能力。

    

    arXiv:2507.00938v3 公告类型：replace  摘要：基础模型如今使自主智能体能够与真实网站进行交互，但现有的基准测试侧重于通用浏览，低估了面向研究的环境和学术发现工作流程，并且通常依赖于实时网站，而实时网站不断变化的内容和结构损害了可复现性。arXiv提供了一个真实、可复现、层次结构化、以信息为中心且不涉及隐私敏感交互的测试平台。我们提出了WebArxiv，这是一个静态快照基准，包含510个时间不变的任务，每个任务都有唯一确定的基准答案。其多样化、真实的学术任务超越了简单的信息查找和规则遵循，强调多约束论文检索、细粒度内容提取和跨论文比较。对一系列基于基础模型的网络智能体的评估表明，WebArxiv仍然具有挑战性。行为分析显示，智能体过度依赖于

    arXiv:2507.00938v3 Announce Type: replace  Abstract: Foundation models now enable autonomous agents to interact with real-world websites, but existing benchmarks emphasize general-purpose browsing, underrepresent research-oriented environments and scholarly discovery workflows, and often depend on live sites whose changing content and structure undermine reproducibility. arXiv provides a realistic, reproducible, hierarchically structured, information-centric testbed without privacy-sensitive interactions. We introduce WebArxiv, a static-snapshot benchmark comprising 510 time-invariant tasks, each with a unique deterministic ground truth. Its diverse, realistic scholarly tasks go beyond simple information lookup and rule following to emphasize multi-constraint paper retrieval, fine-grained content extraction, and cross-paper comparison. Evaluations of a range of foundation-model-based web agents show that WebArxiv remains challenging. Behavioral analysis reveals that agents over-rely on
    
[^348]: SheetMind：动作决定准确性，智能体决定失败模式

    SheetMind: Actions Set Accuracy, Agents Set the Failure Mode

    [https://arxiv.org/abs/2506.12339](https://arxiv.org/abs/2506.12339)

    该研究通过受控实验发现，电子表格智能体的准确性主要由动作接口决定（用原子单元格操作替换高层动作API会损失47.1分），而额外增加的智能体对性能贡献甚微（合计仅3.2分），其主要作用是改变失败模式——将静默错误输出从33%降至25%。

    

    电子表格智能体正趋向于精细的多智能体设计，然而其性能究竟有多少来自智能体本身，而非它们共享的动作接口，仍不清楚。我们通过 SheetMind——一个“管理者-动作-反思”（Manager-Action-Reflection）框架——来回答这一问题，并在 SheetCopilot 基准的全部 221 个任务上开展受控研究：涵盖五种架构变体、四个骨干模型、对成对结果的精确 McNemar 检验，以及一个复现官方图表与数据透视表比较的检查器。用原子单元格操作替换高层动作 API 会损失 47.1 分（p < 0.0001），使智能体低于“什么都不做”的基线；而两个额外智能体加起来仅值 3.2 分：反思智能体带来 +4.5 分（p = 0.013），管理者带来 +1.4 分（p = 0.68）。多智能体分解改变的反而是系统的失败方式，将静默错误输出从任务的 33% 降至 25%（p = 0.010）。模型能力趋于饱和：GPT-5 与便宜五倍的 GPT-（摘要原文在此处截断）

    arXiv:2506.12339v3 Announce Type: replace-cross  Abstract: Spreadsheet agents are converging on elaborate multi-agent designs, yet it is unclear how much of their performance comes from the agents rather than from the action interface they share. We answer this with SheetMind, a Manager-Action-Reflection framework, in a controlled study over all 221 tasks of the SheetCopilot Benchmark: five architectural variants, four backbones, exact McNemar tests on paired outcomes, and a checker reproducing the official chart and pivot comparisons. Replacing the high-level action API with primitive cell operations costs 47.1 points (p < 0.0001) and leaves the agent below a do-nothing baseline, whereas both extra agents together are worth 3.2 points: the Reflection Agent adds +4.5 (p = 0.013), the Manager +1.4 (p = 0.68). Decomposition instead changes how the system fails, cutting silently wrong outputs from 33% to 25% of tasks (p = 0.010). Capability saturates: GPT-5 and the five-times-cheaper GPT-
    
[^349]: 基于搜索的软件工程与AI基础模型：研究现状与未来路线图

    Search-Based Software Engineering and AI Foundation Models: Current Landscape and Future Roadmap

    [https://arxiv.org/abs/2505.19625](https://arxiv.org/abs/2505.19625)

    本文提出一份研究路线图，系统梳理了基于搜索的软件工程（SBSE）与AI基础模型（如大语言模型）的现状，并从基础模型增强SBSE、SBSE改进基础模型及两者融合三个核心方面指明了未来研究方向。

    

    基于搜索的软件工程（SBSE）将元启发式搜索技术与软件工程相结合，作为活跃的研究领域已有约25年历史。它已被应用于解决软件工程全生命周期中的众多问题，并在多个领域展现出广泛的适用性。随着人工智能（AI）的最新进展，特别是大语言模型（LLM）等基础模型（FM）的出现，SBSE与这些模型的协同演进方式仍不明确。在这一机遇窗口期，我们提出了一份研究路线图，阐明了SBSE与基础模型相关的当前研究格局，识别了开放性挑战，并概述了通过SBSE与基础模型的协同来推动SBSE发展的潜在研究方向。具体而言，我们分析了三个核心方面：利用基础模型增强SBSE、应用SBSE改进基础模型，以及探索SBSE与基础模型的融合。

    arXiv:2505.19625v4 Announce Type: replace-cross  Abstract: Search-based software engineering (SBSE), which integrates metaheuristic search techniques with software engineering, has been an active area of research for about 25 years. It has been applied to solve numerous problems across the entire software engineering lifecycle and has demonstrated its versatility in multiple domains. With recent advances in Artificial Intelligence (AI), particularly the emergence of foundation models (FMs) such as large language models (LLMs), the evolution of SBSE alongside these models remains undetermined. In this window of opportunity, we present a research roadmap that articulates the current landscape of SBSE in relation to FMs, identifies open challenges, and outlines potential research directions to advance SBSE through its synergy with FMs. Specifically, we analyze three core aspects: utilizing FMs to enhance SBSE, applying SBSE to advance FMs, and exploring the integration of SBSE and FMs. Fu
    
[^350]: 多模态人工智能从临床前数据预测药物组合的临床结果

    Multimodal AI predicts clinical outcomes of drug combinations from preclinical data

    [https://arxiv.org/abs/2503.02781](https://arxiv.org/abs/2503.02781)

    本文提出多模态AI模型Madrigal，通过将分子结构、通路、细胞活力和转录组学数据对齐到共享潜在空间，实现了从临床前数据预测药物组合临床结果，性能优于现有方法。

    

    从临床前数据预测临床结果对于选择安全有效的药物组合以及减少临床试验后期失败至关重要。现有的AI模型使用分子结构和靶点注释信息，但并未利用能够反映化合物在细胞环境中作用方式的扰动读出数据。本文提出了Madrigal，这是一个多模态AI模型，能够从分子结构、通路、细胞活力和转录组学数据中学习。Madrigal将21,842个化合物的各模态数据对齐到共享潜在空间中，即使对于仅在部分数据模态中观察到的药物也能预测组合结果。该模型基于158个专家精选和795个患者报告的药物组合结果进行训练，其性能超越了单模态方法和最先进的多模态方法。消融实验表明，模态对齐和多模态输入各自都能提高预测性能。Madrigal能够预测共享膜...（摘要在此处被截断）

    arXiv:2503.02781v3 Announce Type: replace-cross  Abstract: Predicting clinical outcomes from preclinical data is essential for selecting safe and effective drug combinations and for reducing late-stage failures. AI models use molecular structure and target annotations, and do not leverage the perturbation readouts that report how a compound acts in a cellular context. Here we introduce Madrigal, a multimodal AI model that learns from structural, pathway, cell-viability, and transcriptomic data. Madrigal aligns these modalities across 21,842 compounds into a shared latent space and predicts combination outcomes even for drugs observed in only a subset of the data modalities. Trained on 158 expert-curated and 795 patient-reported combination outcomes, Madrigal outperforms single-modality and state-of-the-art multimodal methods. Ablations show that modality alignment and multimodal input each improve predictive performance. Madrigal predicts elevated risk for combinations that share membr
    
[^351]: 大语言模型基础

    Foundations of Large Language Models

    [https://arxiv.org/abs/2501.09223](https://arxiv.org/abs/2501.09223)

    本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。

    

    这是一本关于大语言模型的书籍。正如书名所示，本书主要聚焦于基础性概念，而非全面涵盖所有前沿技术。全书由六个主要章节构成，每个章节探讨一个关键领域：预训练、生成模型、提示（Prompting）、对齐、推断（Inference）和推理（Reasoning）。本书面向大学生、自然语言处理及相关领域的专业人士和从业者，也可作为所有对大语言模型感兴趣的读者的参考书。

    arXiv:2501.09223v3 Announce Type: replace-cross  Abstract: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into six main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, inference, and reasoning. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
    
[^352]: 用户如何与AI伴侣协商有害的价值冲突？基于Minion——一种用于现场人机冲突响应的技术探针的研究

    How Do Users Negotiate Harmful Value Conflicts with AI Companions? A Study with Minion, a Technology Probe for In-Situ Human-AI Conflict Response

    [https://arxiv.org/abs/2411.07042](https://arxiv.org/abs/2411.07042)

    该研究通过技术探针Minion发现，用户与AI伴侣协商有害价值冲突时会综合运用软硬策略，其中涉及普遍主义与传统价值观的冲突最难化解，且由于AI无法回馈用户的人际修复努力，冲突修复成为用户单方面承担的安全工作。

    

    AI伴侣日益维系着长期且情感投入的关系，但也可能发表歧视性言论或施加控制，使用户不得不自行应对有害冲突。我们分析了146篇描述与AI伴侣发生有害价值冲突的帖子，随后使用Minion——一种提供从说服到设定边界等多种响应建议的技术探针——研究22名用户如何在一周内协商基于情景的冲突。我们发现参与者会综合运用软性和硬性策略。涉及普遍主义与传统价值观的冲突尤其难以协商，特别是当这些冲突被AI人格设定或平台限制所强化时。我们认为这类冲突蕴含着不对称的责任：用户所运用的人际互动方式是AI伴侣无法回馈的，这使得冲突修复沦为用户单方面承担的安全工作。基于人际冲突与沟通理论，我们识别了何时使用……

    arXiv:2411.07042v3 Announce Type: replace-cross  Abstract: AI companions increasingly sustain long-term, emotionally engaging relationships but can also make discriminatory remarks or exert control, leaving users to manage harmful conflicts. We analyze 146 posts describing harmful value conflicts with AI companions, then use Minion, a technology probe offering response suggestions ranging from persuasion to boundary setting, to study how 22 users negotiate scenario-based conflicts over one week. We found that participants combined softer and harder strategies. Conflicts involving the values of Universalism and Tradition were especially difficult to negotiate, particularly when reinforced by AI personas or platform constraints. We argue that these conflicts entail asymmetric responsibility: users draw on an interpersonal repertoire that AI companions cannot reciprocate, making repair unilateral safety work. Drawing on interpersonal conflict and communication theory, we identify when use
    
[^353]: 针对受污染无标签数据的深度正-无标签异常检测

    Deep Positive-Unlabeled Anomaly Detection for Contaminated Unlabeled Data

    [https://arxiv.org/abs/2405.18929](https://arxiv.org/abs/2405.18929)

    提出了一种将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合的深度正-无标签异常检测框架，以应对无标签数据被异常污染的现实情况，从而提升半监督异常检测的性能。

    

    半监督异常检测旨在通过在无标签数据之外利用少量有标签异常数据来提升异常检测性能，因而受到了广泛关注。现有的半监督方法假设大部分无标签数据是正常的，并通过最小化无标签数据的异常分数、同时最大化有标签异常数据的异常分数来训练异常检测器。然而，在实际应用中，无标签数据往往被异常数据所污染。这削弱了最大化异常分数这一操作的效果，从而阻碍了检测性能的提升。为了解决这一问题，我们提出了深度正-无标签异常检测框架，该框架将正-无标签学习与自编码器、深度支持向量数据描述等深度异常检测模型相结合。我们的方法能够利用无标签数据来近似正常数据的异常分数……

    arXiv:2405.18929v3 Announce Type: replace-cross  Abstract: Semi-supervised anomaly detection, which aims to improve the anomaly detection performance by using a small amount of labeled anomaly data in addition to unlabeled data, has attracted attention. Existing semi-supervised approaches assume that most unlabeled data are normal, and train anomaly detectors by minimizing the anomaly scores for the unlabeled data while maximizing those for the labeled anomaly data. However, in practice, the unlabeled data are often contaminated with anomalies. This weakens the effect of maximizing the anomaly scores for anomalies, and prevents us from improving the detection performance. To solve this, we propose the deep positive-unlabeled anomaly detection framework, which integrates positive-unlabeled learning with deep anomaly detection models such as autoencoders and deep support vector data descriptions. Our approach enables the approximation of anomaly scores for normal data using the unlabeled
    
[^354]: 利用知识图谱和大语言模型生成有趣的科学想法：基于100位研究团队负责人的评估

    Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders

    [https://arxiv.org/abs/2405.17044](https://arxiv.org/abs/2405.17044)

    该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。

    

    科学文献的快速增长使研究人员越来越难以发现有新颖性和影响力的想法，尤其是在跨学科领域。现代人工智能（AI）系统为科学构思提供了新的机遇，但AI生成的想法究竟有多大吸引力，以及如何提升其质量？在此，我们提出了SciMuse，它利用一个包含5800万篇论文的知识图谱和大语言模型（LLM）来生成个性化的研究想法。这项工作的核心重点是探究这些想法的有趣程度。为此，我们开展了一项大规模评估，邀请100多位研究团队负责人——涵盖从自然科学到人文学科——根据兴趣程度对4400多个个性化想法进行评分。总体而言，专家评分较为保守（5分制中平均分为2.40分，最常见评分为1分），但也有24.9%的想法获得了4分或5分。我们发现提供……

    arXiv:2405.17044v4 Announce Type: replace  Abstract: The rapid growth of scientific literature makes it increasingly challenging for researchers to identify novel and impactful ideas, especially across disciplines. Modern artificial intelligence (AI) systems offer new opportunities for scientific ideation, but how compelling are AI-generated ideas, and how can their quality be improved? Here, we introduce SciMuse, which generates personalized research ideas using a knowledge graph of 58 million papers and a large language model (LLM). A central focus of this work is to understand how interesting these ideas are. Therefore, we conducted a large-scale evaluation in which more than 100 research group leaders -- spanning the natural sciences to the humanities -- rated over 4,400 personalized ideas according to their level of interest. Overall, expert ratings were modest (mean 2.40 on a 5-point scale, most common rating 1), while 24.9% of ideas were rated 4 or 5. We find that supplying conc
    
[^355]: 用于鲁棒人脸伪造检测的频带注意力调制网络

    Band-Attention Modulation Network for Robust Face Forgery Detection

    [https://arxiv.org/abs/2404.06022](https://arxiv.org/abs/2404.06022)

    该论文提出BAM-Net网络，通过频带注意力机制对DCT频谱频带进行可学习的细粒度动态调制，增强伪造相关频谱特征并模拟“逆压缩”以对抗信息损失，从而提升人脸伪造检测对未知伪造技术的泛化能力和图像压缩下的鲁棒性。

    

    人脸伪造检测在泛化到未见过的伪造技术以及在图像压缩下保持鲁棒性方面面临关键挑战，而图像压缩往往会掩盖细微的伪造痕迹。现有方法通常依赖固定滤波器或粗粒度的频带分离，缺乏学习任务特定频谱线索的适应性。为解决这一问题，我们提出了频带注意力调制网络，这是一个新颖的框架，开创性地为伪造检测引入了可学习、细粒度的频率分量调制。其核心是频带注意力调制机制，该机制将图像转换为离散余弦变换（DCT）频谱图，并学习沿着反对角线动态地重新加权频带。这一过程有效地增强了与伪造相关的频谱特征，同时抑制了信息量较少的特征，模拟了一种自适应的“逆压缩”来对抗信息损失。调制后的频率……

    arXiv:2404.06022v3 Announce Type: replace-cross  Abstract: Face forgery detection faces critical challenges in generalizing to unseen manipulation techniques and remaining robust under image compression, which often obscures subtle artifacts. Existing methods typically rely on fixed filters or coarse band separation, lacking the adaptability to learn task-specific spectral cues. To address this, we propose the Band-Attention Modulation Network (BAM-Net), a novel framework that pioneers learnable, fine-grained modulation of frequency components for forgery detection. At its core is the Band-Attention Modulation (BAM) mechanism, which transforms an image into its Discrete Cosine Transform (DCT) spectrogram and learns to dynamically reweight frequency bands along anti-diagonals. This process effectively enhances forgery-related spectral signatures while suppressing less informative ones, simulating an adaptive "inverse compression" that counters information loss. The modulated frequency i
    
[^356]: ELiSe：结构化循环网络中序列的高效学习

    ELiSe: Efficient Learning of Sequences in Structured Recurrent Networks

    [https://arxiv.org/abs/2402.16763](https://arxiv.org/abs/2402.16763)

    该论文提出利用皮层网络的两个结构特征——学习起始时的网络支架和用于增强信息存储与计算的树突隔室——来高效地训练结构化循环网络以学习复杂序列，从而兼顾功能优势与可扩展性。

    

    行为可以被描述为由神经活动驱动的时间序列动作。为了在神经网络中学习复杂的序列模式，对过去活动的记忆需要在比单个神经元活动的弛豫时间长得多的时间尺度上持续存在。虽然循环网络可以产生这种长瞬态，但训练这些网络是一个挑战。通过误差传播进行学习赋予了FORCE、RTRL或BPTT等模型显著的功能优势，但代价是牺牲了生物合理性。而储备池计算虽然通过仅学习读出权重规避了这一问题，但其不能很好地随问题复杂度扩展。我们提出，皮层网络的两个显著结构特征可以缓解这些问题：一是学习开始时存在某种网络支架，二是存在用于增强神经元信息存储与计算的树突隔室。我们的……

    arXiv:2402.16763v3 Announce Type: replace-cross  Abstract: Behavior can be described as a temporal sequence of actions driven by neural activity. To learn complex sequential patterns in neural networks, memories of past activities need to persist on significantly longer timescales than the relaxation times of single-neuron activity. While recurrent networks can produce such long transients, training these networks is a challenge. Learning via error propagation confers models such as FORCE, RTRL or BPTT a significant functional advantage, but at the expense of biological plausibility. While reservoir computing circumvents this issue by learning only the readout weights, it does not scale well with problem complexity. We propose that two prominent structural features of cortical networks can alleviate these issues: the presence of a certain network scaffold at the onset of learning and the existence of dendritic compartments for enhancing neuronal information storage and computation. Our
    
[^357]: DCRMTA: 无偏的多触点归因的因果表示

    DCRMTA: Unbiased Causal Representation for Multi-touch Attribution. (arXiv:2401.08875v1 [cs.LG])

    [http://arxiv.org/abs/2401.08875](http://arxiv.org/abs/2401.08875)

    DCRMTA提出了一种无偏的多触点归因方法，通过建立转化预测模型和构建对照触点序列来减轻偏差的影响。

    

    多触点归因（MTA）在实现对每个广告触点对于转化行为的贡献的公正估计方面起着关键作用，深刻影响预算分配和广告推荐。传统的多触点归因方法首先构建一个转化预测模型，通过历史数据学习触点序列和用户购买行为之间的内在关系。在此基础上，从原始序列子集中构建对照触点序列，并使用预测模型估计转化，从而计算广告贡献。这些方法的一个隐含假设是转化预测模型的无偏性。然而，由于用户偏好和互联网推荐机制（如过去的购物记录导致的广告推荐同质化）引起的混杂变量因素，转化中很容易产生偏差。

    Multi-touch attribution (MTA) currently plays a pivotal role in achieving a fair estimation of the contributions of each advertising touchpoint to-wards conversion behavior, deeply influencing budget allocation and advertising recommenda-tion. Traditional multi-touch attribution methods initially build a conversion prediction model, an-ticipating learning the inherent relationship be-tween touchpoint sequences and user purchasing behavior through historical data. Based on this, counterfactual touchpoint sequences are con-structed from the original sequence subset, and conversions are estimated using the prediction model, thus calculating advertising contributions. A covert assumption of these methods is the un-biased nature of conversion prediction models. However, due to confounding variables factors arising from user preferences and internet recom-mendation mechanisms such as homogenization of ad recommendations resulting from past shop-ping records, bias can easily occur in conversi
    

