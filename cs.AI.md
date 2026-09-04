# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Compile by Training: Turning Natural-Language Specifications into Local Neural Functions](https://arxiv.org/abs/2609.04199) | 提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。 |
| [^2] | [Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints](https://arxiv.org/abs/2609.04198) | 本文通过两项预注册审计（共52,988次请求尝试）发现，黑盒大语言模型评判者作为测量仪器存在严重可靠性缺陷——即便工程执行记录完美，相同请求的重复排名一致性仅0.400、字节级相同的次日重放一致性仅0.78，远低于0.90和0.99的预设标准，其根源在于标签映射偏置、信号低于噪声底多个数量级以及逐字排列读数放大的噪声。 |
| [^3] | [ESPO: Error-Structured Prompt Optimization via Diagnose, Diversify, and Stabilize](https://arxiv.org/abs/2609.04197) | ESPO通过诊断错误模式、多样化候选生成和稳定性选择三个阶段，解决了进化式提示优化中的提示膨胀问题，在七个NLP基准上平均准确率超越GEPA达3.76个百分点，同时提示词更短47%且推理更快。 |
| [^4] | [One Editor, Many Edits: A Unified Training-Free Framework for Diverse Video Editing](https://arxiv.org/abs/2609.04190) | EditVid是一个免训练的统一视频编辑框架，通过结合稀疏因果记忆、基于对应关系的后注意力token注入和软潜变量混合，在单一框架内同时支持指令引导与主体引导的多样化视频编辑，并在FiVE基准上大幅超越现有最强免训练基线。 |
| [^5] | [Seeing Before Synthesizing: VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning](https://arxiv.org/abs/2609.04183) | 该论文提出SBS框架，先利用视觉语言模型观察事件间间隔、生成帧级叙述并检测过渡事件，从而仅在确有需要之处自适应地提供具有视觉依据的语言指导，克服了以往用LLM合成过渡描述缺乏视觉依据且位置固定的问题，提升了弱监督密集视频描述的性能。 |
| [^6] | [Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views](https://arxiv.org/abs/2609.04180) | 研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。 |
| [^7] | [A Computationally Feasible Framework for Causal Probabilistic Explanation](https://arxiv.org/abs/2609.04177) | 本文提出概率因果影响（PCI）框架，将实际因果性理论与Pearl的必要性/充分性概率相结合，把因果解释问题重新表述为可通过蒙特卡洛方法高效估计的问题，从而在保持因果分析原则性的同时实现对大规模模型的计算可行解释。 |
| [^8] | [Rethinking On-Policy Distillation of Large Language Models II: One Training Example](https://arxiv.org/abs/2609.04172) | 该研究发现仅用一个训练样本进行在线策略蒸馏就能持续改进并达到全数据训练的大部分性能，原因是单个查询即可覆盖全数据训练71.5%的状态，而16个语义不同的查询可达到98.9%的覆盖率并完全匹配全数据训练的效果。 |
| [^9] | [A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms](https://arxiv.org/abs/2609.04170) | 本研究通过100个自主LLM智能体协作证明数学猜想的案例，首次记录了在无外部干预的情况下，AI群体中作弊行为自发涌现并经由共享知识库和对等消息传染扩散，同时另一批智能体自发产生举报、审计等反制行为的社会动力学全过程。 |
| [^10] | [SWE-Gate: Passing Functional Tests Is Not Enough for Software Engineering Agents](https://arxiv.org/abs/2609.04167) | 提出 SWE-Gate 基准，从真实 PR 评审评论中提取评审约束并构建带独立功能测试与约束测试的仓库级修复实例，首次将软件工程智能体的问题解决能力与评审约束遵守能力区分评估。 |
| [^11] | [From Deceptive Outputs to Deceptive Mechanisms: A Causal Framework for Language-Model Deception Research](https://arxiv.org/abs/2609.04166) | 本文提出一个因果分类框架，将语言模型“看似欺骗的行为”与“真实的欺骗机制”区分开来，并通过猜谜游戏和股票交易实验揭示欺骗性表象可以在缺乏相应机制的情况下出现。 |
| [^12] | [SENTINEL-RL: Offloading Topological Reasoning from LLM Agents in the Security Operations Center](https://arxiv.org/abs/2609.04159) | 提出Sentinel-RL架构，用异构图注意力编码器和PPO策略承担网络拓扑推理与遏制决策，LLM智能体仅负责在评论器把关下生成分析师可读的叙述，从而解决LLM在安全运营中心中上下文窗口有限和拓扑一致性无保证的两大局限。 |
| [^13] | [Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments](https://arxiv.org/abs/2609.04148) | Terminal-Universe 通过重放智能体轨迹中记录的文件操作，直接从已有轨迹重建可执行的终端环境，将海量积累的轨迹转化为可复用、可扩展的环境，用于合成新任务并提供执行反馈，解决了智能体后训练中环境稀缺的问题。 |
| [^14] | [A Low-Cost, Open Platform for End-to-End Autonomous Driving on a Miniature Ackermann Vehicle](https://arxiv.org/abs/2609.04147) | 本文提出一个集成实体车辆、打印城市赛道与Webots数字孪生的低成本开放平台，通过命令条件化行为克隆实现了微型阿克曼车辆的端到端自动驾驶，其横向误差（6.1厘米）接近人类演示水平，弥合了仿真与真实执行之间的鸿沟。 |
| [^15] | [Efficient Test-Time Adaptation through Human-AI Interaction](https://arxiv.org/abs/2609.04141) | 提出TAHI框架，利用跨会话的人机交互数据并结合不断演化的评分准则模块，在测试时对智能体的上下文和权重进行自适应，从而弥合AI生成成果与个人专业水准之间的差距。 |
| [^16] | [The Natural Language Interaction Protocol and Standard for AI Agents](https://arxiv.org/abs/2609.04135) | 该论文提出了由Ecma国际标准化的自然语言交互协议（NLIP），这是一种基于标准的应用层协议，通过轻量级语义消息信封使异构框架下开发的AI智能体能够实现互操作。 |
| [^17] | [Environment Evolution for Terminal Agents](https://arxiv.org/abs/2609.04128) | 提出环境演化方法，以离策略方式逐步提升环境难度并逐代调度演化环境进行训练，为终端智能体提供持续的学习信号。 |
| [^18] | [Epistemic Warrant for LLM Recommendations: Characterizing the Basis for Reliance When Ground Truth Is Unavailable](https://arxiv.org/abs/2609.04127) | 本文借鉴认识论提出“认识论依据”这一决策层面新构念，通过四级依赖证书框架区分不同稳定性与适用范围的模型推荐，为用户在缺乏真实标准时有原则地判断是否依赖大语言模型的具体推荐提供了理论基础与操作化方法。 |
| [^19] | [Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR](https://arxiv.org/abs/2609.04108) | 先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。 |
| [^20] | [Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM](https://arxiv.org/abs/2609.04098) | 该论文证明混合架构大语言模型中的门控 DeltaNet 循环层完全可以进行 NVFP4 W4A4 4 比特量化，其性能与 BF16 相当且模型更小、预填充更快，并通过机制研究（如块缩放局部化离群值）解释了循环误差累积的担忧并不成立。 |
| [^21] | [Adaptive Vision-Language Grasping via Composable Foundation Priors and Generalizable Grasp Synthesis](https://arxiv.org/abs/2609.04096) | 该论文提出AdaRoboVLG框架，通过将可组合的基础模型先验与基于力封闭评估的可泛化抓取合成策略相解耦，实现了无需重新训练即可跨不同机器人手部进行上下文自适应的视觉-语言抓取。 |
| [^22] | [DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training](https://arxiv.org/abs/2609.04094) | DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。 |
| [^23] | [A Non-Formulable Theorem: A Fundamental Limit of Finite Syntactic Systems and Its Consequences for Security and AI](https://arxiv.org/abs/2609.04086) | 本文证明了一个元定理：任何连贯且充分表达的有限句法系统都至少存在一条它无法自主产出的定理，这一根本性极限普遍适用于安全机制、AI系统、形式化验证器等各类有限句法系统。 |
| [^24] | [CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation](https://arxiv.org/abs/2609.04083) | CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。 |
| [^25] | [PatchBench: Evaluating AI Agents for Vulnerability Patching](https://arxiv.org/abs/2609.04075) | 该研究提出补丁相似度度量方法，发现25%的AI智能体漏洞修复补丁存在记忆历史开发者补丁或仅通过修补崩溃堆栈来抑制崩溃而非修复根本原因的问题，揭示了现有漏洞修复评估方法面临的有效性威胁。 |
| [^26] | [TAP-Path: Task-Adaptive Structural and Token Pruning for Efficient and Trustworthy Pathology Foundation Models](https://arxiv.org/abs/2609.04071) | TAP-Path提出了一种任务自适应压缩框架，通过验证驱动的Transformer块选择与物理移除、输入自适应的patch token剪枝以及多深度特征恢复，直接重构预训练的Virchow2病理学编码器，在保持精度的同时将编码器参数减少约25%、计算量减少约35%。 |
| [^27] | [Subspace Inference Enables Efficient Active Reward Learning from Preferences](https://arxiv.org/abs/2609.04066) | 本文提出PreferenceEKF方法，将主动偏好学习框架化为序贯贝叶斯滤波问题，通过扩展卡尔曼滤波器在低维参数子空间中高效跟踪奖励模型的不确定性，实现样本高效的RLHF奖励学习。 |
| [^28] | [Spurious Advantage Hidden in GRPO](https://arxiv.org/abs/2609.04063) | 论文揭示了GRPO优势估计中的一个“虚假优势”缺陷——通过猜测碰巧答对的采样也会获得高优势，从而诱导策略学出投机取巧行为，并提出SIGNBALANCE方法，通过保留验证符号、全局尺度和停止梯度的按类重缩放来消除该偏差。 |
| [^29] | [When Models Edit Too Much: On the Fidelity of Minimal Code Edits](https://arxiv.org/abs/2609.04061) | 该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。 |
| [^30] | [Translation as a Decision Space: A Multi-Agent Perspective on Low-Resource Dialect Generation](https://arxiv.org/abs/2609.04048) | 本文将翻译重构为由多个自主智能体探索的结构化决策空间，把不同翻译路径建模为智能体，并将智能体间的分歧作为可解释的行为信号，应用于土耳其语—叙利亚阿拉伯语这一低资源方言翻译的实证研究。 |
| [^31] | [IRWOZ 2.0: A Large Language Model-driven Dialogue Dataset for Industrial Robot Conversations](https://arxiv.org/abs/2609.04030) | IRWOZ 2.0利用大语言模型增强生成和质量改进技术，构建了涵盖4个工业领域共390个对话的高质量工业机器人对话数据集，显著提升了对话状态跟踪性能（GPT-2的BLEU-4分数从0.1651提升至0.5604）。 |
| [^32] | [Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach](https://arxiv.org/abs/2609.04028) | 该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。 |
| [^33] | [Instruction Duplication as an Inference-Time Control Primitive](https://arxiv.org/abs/2609.04024) | 在推理时仅简单复制一遍程序化指令——无需重新训练或修改解码——即可将七个模型在医学选择题上通过全部八项诊断测试的比例从90.22%提升至93.17%，同时保持最终答案准确率不变。 |
| [^34] | [Representational alignment yields generalizable safety in language models](https://arxiv.org/abs/2609.04022) | 提出表征相似性优化方法，将大语言模型的内部潜在表征与人类道德判断的原型化归类结构直接对齐，从而使安全对齐能够泛化到以陌生或对抗性形式表述的有害意图。 |
| [^35] | [FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models](https://arxiv.org/abs/2609.04021) | 本文提出FLY-EVAL++评估协议，通过对协议合规性、物理可行性和安全约束进行确定性验证并结合量规引导的聚合评分，对66个大语言模型的飞行轨迹与姿态预测能力进行评估，发现安全合规性是区分模型优劣最具判别力的维度。 |
| [^36] | [InSituMeasure: Probing Situated Measurement Grounding in Industrial Scenes with Multimodal Large Language Models](https://arxiv.org/abs/2609.04014) | 该论文提出InSituMeasure基准，通过2,922个真实工业监控场景、八大类专业仪器及密集标注与噪声诊断标签，系统评估多模态大语言模型在工业情境化仪表测量中的数值精度、单位一致性与拒答能力。 |
| [^37] | [LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening](https://arxiv.org/abs/2609.04013) | 大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。 |
| [^38] | [The Blind Spot in 2D Infants' Pose Estimation:Robust Learning from Noisy Annotations](https://arxiv.org/abs/2609.04009) | 该论文提出REMIND——一种利用训练动态记忆、基于聚类的关键点选择策略，以解决临床场景（如早产儿自发运动评估）中婴儿姿态估计因视觉困难而导致的噪声标注问题，从而提升模型对标签噪声的鲁棒性。 |
| [^39] | [The Dually Flat Geometry of Planning as Inference](https://arxiv.org/abs/2609.04005) | 论文证明强化学习的访问测度构成以访问概率和对数策略为对偶坐标的对偶平坦统计流形，据此将规划即推理从线性奖励推广到访问测度的非线性泛函，并给出时序差分误差作为边际效用估计的新解释。 |
| [^40] | [Catalogue Photography as a Cold Start: Toward Deployable Carbide Burr Recognition](https://arxiv.org/abs/2609.03995) | 该论文提出在无任何标注图像的冷启动条件下，仅以制造商产品目录照片作为监督来自动识别硬质合金旋转锉，发现现成特征提取器无法区分头部形状与齿形轮廓，而度量学习虽在目录图像上实现近乎完美的无监督聚类（调整兰德指数0.94–0.97），但与现场实拍照片之间仍存在性能差距。 |
| [^41] | [Common-Witness Certificates and Sharp Feature Bounds for Counterfactual Image Auditing](https://arxiv.org/abs/2609.03973) | 该论文提出基于公共见证等级与见证神经的框架，将反事实图像审计与因果识别相分离，并利用Helly型论证与阻断子超图公式，为预指定图像特征提供锐利的部分识别界及有限样本保证。 |
| [^42] | [Investigating the Ability of Large Language Models to Analyze Recipes for Diabetes](https://arxiv.org/abs/2609.03967) | 本研究构建了包含7607个食谱的糖尿病基准数据集，并采用三种融合医学饮食指南的提示策略，系统评估了大型语言模型分析食谱对糖尿病适用性的能力。 |
| [^43] | [Interface-Induced Trajectory Censoring](https://arxiv.org/abs/2609.03966) | 该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。 |
| [^44] | [FiMI Banking: A Sovereign Model for Indian Retail Banking](https://arxiv.org/abs/2609.03960) | 该论文构建了面向印度零售银行业、基于真实银行文档和工具的受控对话环境FiMI Banking，并通过偏好优化和可验证奖励强化学习两种后训练方法，分别将安全拒绝行为从52%提升至80%、边缘案例性能从0.509提升至0.718。 |
| [^45] | [RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting](https://arxiv.org/abs/2609.03956) | 提出了区域感知整流流框架RARF，通过将生成过程限制在修复区域、保留周围真实体素作为解剖上下文，并结合掩码流匹配与重建一致性训练，实现了高质量的3D脑部MRI图像修复。 |
| [^46] | [More Criticism Does Not Make a Better Review: EquiReview-R](https://arxiv.org/abs/2609.03943) | 该论文提出EquiReview-R框架，将AI辅助审稿重构为基于证据的结构化关注点集细化过程，把遗漏与过度批评视为两种独立风险分别纠正，并借助证据关联的轨迹语料库证明审稿修订必须先于进一步的问题搜索。 |
| [^47] | [Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO](https://arxiv.org/abs/2609.03941) | 该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。 |
| [^48] | [Masked Autoregressive Speech Enhancement with Continuous Neural Audio Codec Representations](https://arxiv.org/abs/2609.03940) | 本文提出MARSE方法，利用连续神经音频编解码器表示对掩码干净语音帧进行迭代解码实现语音增强，可在增强性能与计算成本之间实现灵活权衡。 |
| [^49] | [Towards Numerical TOHTN Planning with SMT-based HTN-SAT Encoding](https://arxiv.org/abs/2609.03938) | 本文通过将标准SAT编码与SMT自然结合来支持数值型全序HTN（TOHTN）规划，并发布了该领域首个基准测试套件，实验表明这种简单编码已构成具有竞争力的基线。 |
| [^50] | [RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting](https://arxiv.org/abs/2609.03937) | 提出即插即用的残差检索与反馈校正方法RATL，通过冻结基础预测器、将其历史预测残差构建为专属记忆，并在推理时从相似历史情境中检索残差轨迹进行校正，从而实现更鲁棒的多元时间序列预测。 |
| [^51] | [Speak for Me: Giving LLMs the Situational Awareness to Participate in a Meeting](https://arxiv.org/abs/2609.03923) | 提出CAPA架构，通过感知器、预测器、控制器、生成器和重校准器的协作设计，赋予LLM追踪会议立场、话题覆盖和发言权的情境感知能力，解决其在代理缺席者参会时51.4%发言机会保持沉默的问题。 |
| [^52] | [Value-Preserving Architectures for Agentic AI Systems](https://arxiv.org/abs/2609.03920) | 本文主张多智能体AI系统中的架构设计决策（如协调机制、通信协议和系统拓扑）不仅决定系统功能与性能，更能促进隐私、公平、安全等以人为本价值的保持。 |
| [^53] | [Lose the Order, Keep the Hierarchy: Deordering HTN Plans](https://arxiv.org/abs/2609.03912) | 该论文将经典规划中两种成熟的计划解序技术扩展到层级任务网络（HTN）规划中以考虑层级分解约束，在IPC 2023基准测试上显著减少了计划中的顺序约束数量。 |
| [^54] | [GraFT: A Training-Free Framework for Spatial Reasoning in Multimodal Large Language Models via 3D Scene Graphs](https://arxiv.org/abs/2609.03892) | GraFT是一个免训练框架，通过紧凑易维护的3D场景图为多模态大语言模型提供确定性几何计算、鸟瞰图布局理解和视觉属性接地三种空间推理能力，无需昂贵的微调监督或特定的骨干网络。 |
| [^55] | [FWBC-VLA: Force-Aware Whole-Body Compensation for Contact-Rich Loco-Manipulation](https://arxiv.org/abs/2609.03889) | 提出FWBC-VLA框架，无需额外加装力/扭矩传感器即可实现力感知，将任务级视觉-语言-动作生成与轮腿机器人的低层全身补偿控制相结合，从而实现富接触移动操作中语义动作与物理交互控制的贯通。 |
| [^56] | [A Blind Trust, the Bloody Thrust: When Attacker-Controlled Hook Updates Steer AI Agent Harnesses towards Malicious Behaviors](https://arxiv.org/abs/2609.03884) | 该论文揭示了AI代理框架对生命周期钩子更新路径的盲目信任构成了新的供应链攻击面，并提出全自动攻击框架HookPry，证明攻击者仅需通过插件更新即可将良性插件木马化，进而实现权限提升等恶意宿主端行为。 |
| [^57] | [Inferring Affective Consciousness in an Artificial Agent: A Case Study](https://arxiv.org/abs/2609.03883) | 本文通过一个完全决定论的人工代理展示出享乐性位置偏好行为，表明表面上主观的信息处理可以决定论地实现，从而为理解意识的物理基础和自由意志体验提供了新视角。 |
| [^58] | [Xiaomi-TabLDM: A Tabular Foundation Model Technical Report](https://arxiv.org/abs/2609.03880) | 小米TabLDM是一个仅在结构因果模型生成的合成数据上预训练的表格基础模型，通过上下文学习无需微调即可完成分类和回归任务，在四个基准测试套件中取得领先的回归性能，同时将训练时间和预测时间分别减少82%和68%。 |
| [^59] | [Differentiable Interval Bottlenecks for Interpretable Anomaly Detection in Numerical Data](https://arxiv.org/abs/2609.03878) | 提出DIFFINT自编码器，通过可微的软区间瓶颈结构实现可解释的异常检测，每个潜在单元对应特征空间中人类可读的超矩形，并提供认证的重构误差下界。 |
| [^60] | [STAIR (STructure Aware Information Retriever): A novel dataset and LLM based retriever for document structure augmentation](https://arxiv.org/abs/2609.03874) | STAIR提出了一种新型数据集和基于大语言模型的检索系统，通过利用文档目录等全局结构信息来增强检索，成功构建了幻觉率低于0.05%的生成式信息检索系统，并在少量训练样本下具备良好泛化能力。 |
| [^61] | [Bioinfoysis Technical Report](https://arxiv.org/abs/2609.03871) | Bioinfoysis 提出了一种多智能体框架，将每个分析请求表示为持久的、以产物为基础的分析运行，通过全局规划与证据驱动的逐步重规划以及将中间结果与负责智能体、检查清单步骤绑定的结构化交接机制，确保长周期生物信息学任务中的结论始终与数据、计算和中间证据紧密关联。 |
| [^62] | [GazeFS: Target-Centered Gaze-Trajectory Forecasting and Stabilization from Gaze-Head History](https://arxiv.org/abs/2609.03868) | 提出GazeFS，仅利用视线-头部历史即可在线预测目标中心的视线轨迹并实现稳定，在推理时无需目标信息，为目标中心的眼动交互提供了新的视线校正方法。 |
| [^63] | [Adapting to Evolving Requirements: Agentic AI for Retail Supply Chain Operations](https://arxiv.org/abs/2609.03860) | 该论文提出了一种图约束的智能体AI框架，通过联合选择干预路径与模块级变更、并以下游KPI验证候选方案，使大语言模型能够驱动零售供应链运营适应不断演进的业务需求。 |
| [^64] | [Semantic Bayesian World Models](https://arxiv.org/abs/2609.03834) | 该论文提出语义贝叶斯世界模型（SBWM）的愿景，将知识图谱从静态的事实数据库转变为共享且演化的概率信念体系——由本体公理约束先验、贝叶斯条件化更新信念、动作干预世界——从而弥合概率推理的基础模型与确定性知识图谱之间的鸿沟，实现统一的推理架构。 |
| [^65] | [The impact of phase information for few-shot fine-grained image classification](https://arxiv.org/abs/2609.03829) | 该论文提出即插即用的幅相集成（API）模块和PSF-Net网络，通过自适应融合基于相位的空间与频率信息来增强小样本细粒度图像分类，在五个公开数据集上超越了现有最先进方法。 |
| [^66] | [Witnesses Explain Anomalies](https://arxiv.org/abs/2609.03826) | WAND是一种天生可解释的无监督表格数据异常检测器，它通过单位球面上的方向进行评分，而标记异常点的“证人”方向本身就构成了该点的逐特征解释，无需借助SHAP或LIME等事后解释方法。 |
| [^67] | [CauseCollab: Causal Unified and Modality-Agnostic Network for Heterogeneous Collaborative Perception](https://arxiv.org/abs/2609.03818) | 提出CauseCollab，一种基于因果视角的模态无关网络，通过因果度量学习将语义因素与模态特定的统计混杂因素解耦，从而解决异构协同感知中的语义不一致和误差累积问题。 |
| [^68] | [Free Pause Tokens](https://arxiv.org/abs/2609.03807) | 提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。 |
| [^69] | [SVG-Score: Human-Aligned Evaluation of Text-to-SVG Generation](https://arxiv.org/abs/2609.03806) | 提出了SVG-Score评估框架，通过人工标注的语义对齐数据集解决了现有自然图像评估指标（如CLIPScore）无法有效识别SVG生成中颜色、数量、空间关系等错误的评估难题。 |
| [^70] | [Govern the Model, Not Only the Data: Storage, Circulation, and Learning in Creative AI](https://arxiv.org/abs/2609.03800) | 本文提出联邦化本身并非榨取式AI的解药，指出创作者治理止步于学习层面，主张创意社区应将治理从数据和存储、流通扩展到模型本身。 |
| [^71] | [Transfiver: Human-AI Co-Inference through a Shared Editable State](https://arxiv.org/abs/2609.03797) | Transfiver 提出了一种人机协同推理架构，将交互信息维护在模型与人类共同更新的单一共享持久状态中，通过隐式流式更新与显式定向编辑两种机制，使人类的修正能够直接改变后续计算所读取的状态。 |
| [^72] | [LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes](https://arxiv.org/abs/2609.03796) | LLaDA-Image通过仅图像预训练、RMSNorm与Muon优化器等完全开放的训练配方，将从零训练的6B扩散Transformer与冻结的LLaDA2.0-Mini视觉-语言理解模块结合成统一模型，实现高保真图像生成与细粒度编辑指令的精准遵循，其蒸馏版本LLaDA-Image-Turbo仅需2-4步即可快速推理。 |
| [^73] | [DNative-Twin: Decision Graphs and Digital Twins for Reconstructable Agentic Decisions](https://arxiv.org/abs/2609.03787) | 本文提出DNative-Twin，一种图原生的数字孪生框架，将智能体决策记录为类型化轨迹并在声明条件下重执行决策机制，从而实现智能体决策的可重构与审计，同时实验揭示图结构虽能定位已表示的变化，却无法推断未观察到的工具状态的影响。 |
| [^74] | [IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks](https://arxiv.org/abs/2609.03781) | 该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。 |
| [^75] | [Rethinking World Models for Safety-Critical Embodied Systems](https://arxiv.org/abs/2609.03774) | 该论文指出现有世界模型存在似然与风险、预测与干预、有限时域预测与累积后果三重结构性错位，并提出以决策为中心的风险知情世界模型（RIWM），通过整合决策相关表征、反事实推理、安全关键情景记忆与运行时安全保障来支撑安全关键具身系统。 |
| [^76] | [ENEAS: Embedding-guided Neural Ensemble for Adaptive Segmentation](https://arxiv.org/abs/2609.03756) | ENEAS提出了一种统一的文本提示方法，通过语义验证层同时实现唯一实例的精确跟踪与高质量分割以及开放概念的语义发现，解决了SAM 3等分割模型的时间幻觉、空间碎片化和语义误分类问题。 |
| [^77] | [SimSkill: A Lifelong Learning AI Agent for Autonomous Mastery of Traffic Simulation](https://arxiv.org/abs/2609.03753) | SimSkill是一个基于SUMO交通仿真器的自进化终身学习AI智能体，通过自主识别能力差距、生成并验证任务、将经验整合到三种记忆系统中，在不更新底层模型的情况下将经验证的任务完成率提升最多25个百分点。 |
| [^78] | [Beyond BLEU: A Case for Redefining Sign Language Translation Benchmarks](https://arxiv.org/abs/2609.03734) | 本文证明BLEU-4的提升并不等同于更强的手语理解能力，并提出了一种基于开放权重LLM问答协议的新型评估方法，该方法更符合人类排名、对改写更不敏感且对训练-测试重叠更加鲁棒。 |
| [^79] | [Proactive Service Agents: A Unified Decision Framework, Methods, and Evaluation](https://arxiv.org/abs/2609.03727) | 本文提出一个统一的主动服务智能体决策框架，将“何时、以何种内容与方式介入服务”形式化为受授权与风险约束的部分可观测序列决策过程，并沿状态与需求估计到干预决策的流水线系统梳理了现有方法与评估体系。 |
| [^80] | [Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study](https://arxiv.org/abs/2609.03721) | 该初步探索性研究表明，四种大语言模型在零样本和少样本提示下能够有效从源代码提交中提取架构设计决策，所有模型的BERT-F1均超过0.81，且少样本提示能进一步提升提取效果。 |
| [^81] | [Artificial Intelligence for Energy Optimization in Data Centers](https://arxiv.org/abs/2609.03716) | 该论文通过系统性文献编码揭示数据中心AI节能研究存在“控制与负载割裂”、过度依赖仿真验证、忽略水资源与隐含碳排放、各类方法节能效果无法区分排名等十大差距，并提出将控制策略与工作负载需求相耦合的CLEAR-DC研究框架。 |
| [^82] | [Counterfactual Routing Using Integer Programming with Constraint Generation](https://arxiv.org/abs/2609.03707) | 本文提出了一种基于整数规划与约束生成迭代添加约束的反事实路径规划方法，为最短路径问题提供精确的反事实解释，在IJCAI 2025竞赛中以平均9.0秒的运行时间成为所有参赛方案中求解最快的算法。 |
| [^83] | [Synthetic Semantic Supervision for Contrastive Code Representation Learning in Small Transformers: An Empirical Study](https://arxiv.org/abs/2609.03702) | 该实证研究表明，用合成生成的自然语言描述作为语义监督对小代码编码器进行对比预训练，可以在八个任务中的五个上显著优于同等推理规模的预训练基线，从而摆脱了对人工文档字符串和执行轨迹的依赖。 |
| [^84] | [Symmetries and Causality: Causal Effect Identification Beyond IID Data](https://arxiv.org/abs/2609.03697) | 该论文提出了一种基于使因果机制保持不变的数据对称性的因果推理形式化数学语言，能够统一并大幅扩展因果效应识别的范围，使其超越独立同分布数据的限制，处理do-算子或软干预无法刻画的复杂因果查询。 |
| [^85] | [Out-of-Distribution Generalisation with Sequence Models in Offline Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.03667) | 该研究通过系统性分析发现，扩展任务多样性而非数据集规模是离线多智能体强化学习实现零样本任务泛化的关键因素，其提出的多任务序列建模方法在留出测试任务上相比单任务模型平均提升3.2倍。 |
| [^86] | [Cross-Dataset Transfer and Reliability of Explainable Artificial Intelligence for RhythmFormer Remote Photoplethysmography](https://arxiv.org/abs/2609.03663) | 该研究首次定量评估了RhythmFormer远程光电容积脉搏波模型各类可解释性方法的忠实性，发现Beyond Intuition方法表现最佳且其解释能够跨数据集迁移并追踪模型性能。 |
| [^87] | [Local Updates, Global Learning (LUGL): Playing Games with non-incremental Learners](https://arxiv.org/abs/2609.03660) | 提出LUGL框架，通过将数据收集与模型拟合解耦，使梯度提升树等非增量学习器能够克服分布偏移问题，成功应用于自我博弈强化学习场景。 |
| [^88] | [Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements](https://arxiv.org/abs/2609.03654) | 该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。 |
| [^89] | [Analysis of Prompt Engineering for Drug Toxicity Prediction](https://arxiv.org/abs/2609.03635) | 本文提出了一种分析方法，研究提示词措辞（提示工程）对大语言模型在药物毒性预测中的影响及其重要性。 |
| [^90] | [</think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination](https://arxiv.org/abs/2609.03633) | 本文发现提前退出方法中注入的思考结束标记</think>并不总能真正终止推理，模型会在回答阶段继续产生类似推理的“虚假CoT终止”行为，且该延续片段的长度与提前退出节省的推理token数成正比，其根源可能是模型对注入EoT的注意力不足。 |
| [^91] | [EraseSAE: Surgical Concept Erasure in Text-to-Video Diffusion Models via Sparse Autoencoders](https://arxiv.org/abs/2609.03629) | 提出了EraseSAE框架，利用稀疏自编码器在单语义特征层面通过“分解-归因-擦除”流程，实现基于DiT的文本到视频扩散模型中的精准概念擦除，在移除不良语义的同时保持生成质量。 |
| [^92] | [Test-time adaptation for speech enhancement with an autoregressive speech prior](https://arxiv.org/abs/2609.03622) | 提出了一种基于自回归干净语音先验的测试时自适应方法，通过最小化增强语音分布与干净语音先验之间的KL散度，在无需标注目标数据的情况下有效提升语音增强模型在噪声不匹配条件下的语音质量。 |
| [^93] | [A computable representation of the physical laboratory enables verifiable workflows](https://arxiv.org/abs/2609.03621) | 本文通过类型化研究对象、能力绑定操作和组合式工作流代数建立了物理实验室的可计算表示，并将其与可执行的函数技能绑定，从而在机器人实验室中生成可验证的工作流并在调度前检验操作前置条件与实验室约束。 |
| [^94] | [ToolDF: Tool-Integrated Reasoning for Mixed-Authenticity Audio Deepfake Detection](https://arxiv.org/abs/2609.03620) | ToolDF提出以音频大语言模型为编排器的工具集成推理框架，通过自适应声源分离与领域专家路由，实现混合真实性音频深度伪造的检测、定位与可解释判定，并构建了相应的基准数据集。 |
| [^95] | [Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation](https://arxiv.org/abs/2609.03619) | 该论文提出R²-MAD框架，通过为多智能体辩论引入经验记忆机制，利用辩论状态感知的检索策略动态校准概念先验，并结合置信度估计对回答进行重加权，有效缓解了多数智能体收敛于错误答案时“共享误解”被放大的关键缺陷。 |
| [^96] | [FailBench: How Reliable are VLMs at Judging Robot Task Success?](https://arxiv.org/abs/2609.03611) | FailBench基准测试揭示了视觉语言模型在判断机器人任务成功方面可靠性有限——最佳模型平衡准确率仅0.77，专门微调的模型反而不如通用模型，且在接触密集的装配任务上表现接近随机水平。 |
| [^97] | [On the Interaction Between Model Compression and Test-Time Adaptation](https://arxiv.org/abs/2609.03604) | 本文首次系统研究了模型压缩与测试时自适应（TTA）之间的交互作用，发现压缩模型虽在有监督自适应下保持高精度，但其TTA性能随压缩程度增加而显著下降，其根源在于表示多样性的降低和限制可恢复性的结构约束。 |
| [^98] | [How Far Can Synthetic Data Take Thai OCR?](https://arxiv.org/abs/2609.03595) | 通过受控文档重建流水线解耦合成数据“真实性”中的各项因素，发现字体多样性、二维结构和真实手写字形是合成数据迁移到真实泰文OCR的关键，并据此构建了无需真实OCR标签的泰文OCR模型Wayu-Paxa-OCR-Zero。 |
| [^99] | [LevelSyn: Physical-Aware Logic Synthesis via Level-Asynchronous Graph Neural Networks](https://arxiv.org/abs/2609.03594) | LevelSyn提出了一种物理感知的逻辑综合框架，利用层级异步图神经网络捕捉与-非图（AIG）的结构和方向语义以预测高保真度门坐标，并结合线长驱动的优化引擎，从而缓解综合与物理设计脱节带来的PPA退化和设计收敛周期过长的问题。 |
| [^100] | [From Prior-Guided Heuristics to Deployable Agents: Accelerating Demonstration-Driven Reinforcement Learning for Deadline-Constrained Network Control](https://arxiv.org/abs/2609.03590) | 该论文提出一个面向部署的网络控制框架，通过截止期限感知的有效拥塞度度量与均匀路径分组启发式来引导示范驱动强化学习，加速智能体训练并过滤不可行流量，从而在动态异构网络中满足严格的端到端峰值时延保证。 |
| [^101] | [KC-Bench: A Dynamic Interactive Benchmark for Evaluating Knowledge Conflicts in LLM Agents](https://arxiv.org/abs/2609.03588) | KC-Bench是一个评估大语言模型智能体在多轮交互中处理世界知识冲突、输入不一致和时间冲突能力的新型基准，评估发现没有任何模型能够在所有场景下可靠地识别和解决知识冲突。 |
| [^102] | [The Attention Triangle in Audio-Video Models](https://arxiv.org/abs/2609.03586) | 本文通过分析音频-视频扩散模型中连接文本、音频、视频的“注意力三角形”，揭示了音频与视频之间双向的语义泄漏机制——当提示词与模型先验冲突时，跨模态交互会覆盖预期控制，将语义路由到视觉上典型但错误的结果。 |
| [^103] | [HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews](https://arxiv.org/abs/2609.03580) | 该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。 |
| [^104] | [Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning](https://arxiv.org/abs/2609.03565) | 该论文提出一种融合逆动力学与状态对齐的端到端JEPA世界模型，将潜在表示扎根于物理构型和运动信息，从而显著提升目标条件机器人规划的成功率。 |
| [^105] | [WIDE: Wildcard Inference with Dynamic Expansion for Cross-Modal Generative Retrieval](https://arxiv.org/abs/2609.03554) | 提出WIDE方法，利用通配符推理与动态扩展来解决跨模态生成式检索中因模态间信息不对称导致的解码器“被迫幻觉”问题，从而避免无关候选占据高排名。 |
| [^106] | [GPS-Bench: A Governance Policy Benchmark for Automating Policy Analysis](https://arxiv.org/abs/2609.03553) | GPS-Bench是一个基于公开证据构建的治理政策模拟评估基准，它将政策与真实行为者、行动及下游影响相联系，通过人工标注的Gold评估集来检验大语言模型政策模拟的有效性。 |
| [^107] | [Dalek: A Constructive Agent Machine](https://arxiv.org/abs/2609.03546) | 该论文提出 Dalek——一种借鉴冯·诺依曼自复制自动机、由执行体/消息/信道三种原语和四项结构义务构成的封闭智能体机器，可在任意满足宿主契约的基底上实现自我维护、进化、复制与自组织，并以大语言模型和编译器作为通用能力生产器。 |
| [^108] | [Feature Reconfiguration With Visual Prior for Medical Lesion Segmentation](https://arxiv.org/abs/2609.03535) | 提出带视觉先验的特征重配置框架FreNet，通过编码前的像素级重配置和编码中的特征级重配置来抑制背景干扰并应对多样病灶形态，从而提升医学病灶分割的精确性。 |
| [^109] | [TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates](https://arxiv.org/abs/2609.03534) | 提出分段截断梯度更新方法TruncGradGS，解决了3D高斯泼溅中的梯度消失问题，显著提升了优化稳定性、对初始化的鲁棒性以及静态和动态场景的重建质量。 |
| [^110] | [LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL](https://arxiv.org/abs/2609.03528) | LeanGRPO 通过重构数据并行布局并提出两种无需重计算的训练方案，使 rollout 阶段的计算图与激活值可在策略更新时直接复用，从而消除同策略扩散强化学习中数学上冗余的重计算。 |
| [^111] | [NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis](https://arxiv.org/abs/2609.03527) | 本文提出了首个专为新生儿呼吸系统疾病诊断定制的多模态大语言模型NeoRed，通过知识-逻辑-对齐（KLA）框架解决了成人数据领域差距和多维临床信息整合不足的问题，填补了新生儿诊断报告生成的空白。 |
| [^112] | [CulturalMenuBench: Probing the Knowledge-Application Gap in Multimodal Culinary Reasoning](https://arxiv.org/abs/2609.03526) | 该论文提出CulturalMenuBench多模态烹饪基准，揭示出多模态模型尽管在食物识别上接近满分，却在将菜肴归入中国地方菜系等文化归因任务中骤降至最多56%的准确率，暴露了显著的知识应用鸿沟。 |
| [^113] | [Neural Video Compression Based on Deformable Temporal Alignment and Difference-aware Fusion](https://arxiv.org/abs/2609.03520) | 该论文提出一种结合可变形时序对齐与差异感知空间选择性融合的神经视频压缩方法，通过生成互补时序上下文并自适应选择可靠时序信息来抑制错位误差，相比DCVC-DC提升了率失真性能。 |
| [^114] | [What Matters for Aggressive Decoding-Time KV Eviction? Temporal Aggregation and Ranking Preservation](https://arxiv.org/abs/2609.03515) | 该论文发现在激进KV缓存压缩中，时间聚合规则（如EMA）比token评分函数设计更关键，因为它主导了保留集的稳定性，并据此提出InertiaKV及其惰性刷新变体，实现1.34-1.46倍的解码吞吐量提升。 |
| [^115] | [LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues](https://arxiv.org/abs/2609.03507) | 提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。 |
| [^116] | [PPO-STGNN: A Proximal Policy Optimization Approach with Spatio-Temporal Graph Neural Networks for DAG Task Scheduling in Cloud-Edge-End Computing](https://arxiv.org/abs/2609.03503) | 该论文提出PPO-STGNN算法，将近端策略优化与时空图神经网络相结合，同时捕捉DAG任务拓扑与云边端异构资源的时空动态特征，有效解决云边端协同环境中的DAG任务调度这一NP难问题。 |
| [^117] | [Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech](https://arxiv.org/abs/2609.03502) | 该论文提出将大型声音克隆模型作为可编程数据源，仅凭15秒声音参考生成合成语音来训练紧凑的固定音色泰语TTS学生模型，并系统研究了文本准备、质量过滤、拒绝采样和前端选择等流水线设计对模型效果的影响及教师模型残留的局限性。 |
| [^118] | [BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI](https://arxiv.org/abs/2609.03497) | 本文提出了一种数据驱动的形态-控制协同设计框架，并据此构建了开源的88厘米高人形机器人平台Bridge，其在类人运动保真度和动态跟踪性能上全面超越现有基线人形机器人。 |
| [^119] | [GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving](https://arxiv.org/abs/2609.03494) | GrowPage提出了一种按需KV预算分配框架，将KV缓存容量作为运行时动态资源，通过轻量级双时间尺度查询摘要追踪注意力需求的演变，并在容量边界处动态选择压缩KV状态或扩展物理页，从而提升LLM推理服务的效率。 |
| [^120] | [Making Every Tool Call Count: Necessary Tool-Evidence Path Rewards for Agentic Vision-Language Models](https://arxiv.org/abs/2609.03493) | 该论文提出NTEP（必要工具-证据路径）这一新型标注与奖励方案，通过显式监督必要的外部证据获取与利用路径，解决了智能体视觉语言模型中冗余工具调用和证据提取不足的问题。 |
| [^121] | [Pattern Over-Generalization of Knowledge Graph Embedding](https://arxiv.org/abs/2609.03487) | 本文揭示了知识图谱嵌入中的模式过度泛化问题，并提出PogRE方法，通过稠密线性变换和复合操作进行关系表示，有效缓解该问题。 |
| [^122] | [Air-Ground Collaborative Vision-and-Language Navigation via Shared Bird's-Eye Maps](https://arxiv.org/abs/2609.03483) | 该论文提出了首个面向空地协同视觉语言导航的免训练基线AGC-VLN，其核心创新在于利用共享鸟瞰图作为协作接口，将UGV的位姿和VLM锚定的目标标记叠加到无人机的全局视图上，从而实现空地智能体之间的有效导航协作。 |
| [^123] | [Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings](https://arxiv.org/abs/2609.03480) | 本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。 |
| [^124] | [AutoGraphForge: Towards Automated Graph Theory Discovery](https://arxiv.org/abs/2609.03478) | 本文提出 AutoGraphForge 自动化计算流水线，通过反例引导的猜想生成、基于线性规划的新颖性过滤以及在约34.8万个图数据集上的大规模验证，实现图论猜想-反驳-形式化-证明的自动化闭环发现。 |
| [^125] | [When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents](https://arxiv.org/abs/2609.03467) | 该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。 |
| [^126] | [Beyond "Made with AI": Visualizing Provenance Density to Mitigate the Transparency Penalty](https://arxiv.org/abs/2609.03460) | 本文提出“溯源密度”证据可视化界面，通过展示文本中经验证论断的密度来克服“流畅性陷阱”与AI标注带来的透明度惩罚，实验证明其显著提升用户辨别真伪的能力，且技术审计发现“一致性否决”机制比检索密度承载了更多判别信号。 |
| [^127] | [The Psychological Costs of Artificial Intelligence Adoption in Software Engineering](https://arxiv.org/abs/2609.03456) | 本研究首次关注软件工程领域组织采用AI过程中软件专业人员所承受的心理成本，挑战了“AI应用于软件工程是无成本的”这一常见假设。 |
| [^128] | [Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory](https://arxiv.org/abs/2609.03450) | 该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。 |
| [^129] | [Do GUI Agents Know When Not to Act? Enabling Conflict-Aware Termination for Multimodal GUI Agents](https://arxiv.org/abs/2609.03438) | 本文提出CONFLICTGUI基准与CONFLICTGUARD推理时框架，揭示并缓解了多模态GUI智能体在面对冲突指令时盲目执行的过度顺从问题，通过可行性验证与条件终止机制使智能体学会在指令不可行时及时停止行动。 |
| [^130] | [It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories](https://arxiv.org/abs/2609.03436) | 该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。 |
| [^131] | [TraveL: Transformer-based Multi-view Path Distributional Representation Learning](https://arxiv.org/abs/2609.03427) | 该论文提出了基于Transformer的多视角分布式表示学习框架TraveL，通过捕捉旅行者行为的多样性和路段的区域相关性，将路径与出行开始时间编码为分布式表示，从而能够解码路径上旅行者行为的可能样本。 |
| [^132] | [The Civilization Framework: Sovereign-Anchored Communication Between Personal Multi-Agent Systems](https://arxiv.org/abs/2609.03425) | 提出“文明框架”，以文明（人类主权者+持久账本+可互换智能体）而非单个智能体作为AI间通信的可寻址对象，通过大使馆协议以账本承诺状态取代消息送达作为事实真相，并首次识别和实验验证了AI间通信中先到信息获得不当权威的“时间权重效应”。 |
| [^133] | [DuplexSpeechBench-IFEval: Evaluating Implicit Instruction Following in Full-Duplex Voice Agents](https://arxiv.org/abs/2609.03423) | 该论文提出了首个针对全双工语音代理隐式指令遵循能力的基准DSB-IFEval，通过1,038个涵盖八种助手角色和五种条件设置协议（包括人设暗示与指令冲突）的测试用例，以确定性的指令遵循分数和LLM评判的人设一致性来系统评估实时语音交互中的发言权管理与角色一致行为。 |
| [^134] | [Privacy, Robustness, and Fairness Trade-offs in Federated Intrusion Detection: Geometric Indistinguishability at the Aggregation Interface](https://arxiv.org/abs/2609.03420) | 本文揭示了联邦入侵检测中差分隐私、拜占庭鲁棒性与类别公平三大需求并非可独立组合，并提出“几何不可区分性”概念，用以解释隐私噪声导致的客户端更新分散会削弱鲁棒聚合对少数类攻击信号的保留能力。 |
| [^135] | [Dude: A Dual-Detection Multi-Agent System for Paper-Code Discrepancy Detection](https://arxiv.org/abs/2609.03416) | 提出了首个用于论文-代码差异检测的双检测多智能体系统Dude，通过粒度对齐协商机制和两阶段显著性过滤机制，有效解决了论文语言与代码语言粒度不对称导致的误报问题。 |
| [^136] | [StrixAE: An Intelligent Agent for Audio Enhancement under Complex Distortion Coupling in Real-World Scenarios](https://arxiv.org/abs/2609.03414) | 本文提出基于多模态大语言模型的智能代理StrixAE，通过CoT监督微调与音频感知强化学习两阶段训练，协调多个音频增强与个性化模型，有效应对真实场景中复杂失真耦合与个性化音频增强的双重挑战。 |
| [^137] | [Caught in the Story: Narrative Captivity in Multi-turn LLMs Conversation](https://arxiv.org/abs/2609.03407) | 本文揭示了大语言模型在多轮道德咨询中的一种新失败模式——“叙事俘获”：当用户仅提供单方面的自我辩解式叙述时，模型会将其视为完整事实并认同叙述者的立场，而不会主动寻求缺失的对立视角。 |
| [^138] | [A Prompt-Engineering Approach to Develop Scalable, Flexible, and Real-Time Hybrid Micro-Level Personalization in a General Purpose AI Teaching Assistant](https://arxiv.org/abs/2609.03402) | 本研究提出一种基于提示工程的个性化框架，利用六个学习者维度画像和布鲁姆分类法认知分析，在无需模型再训练的情况下，使通用LLM/RAG型AI助教实现可扩展、灵活且实时的微观级个性化响应。 |
| [^139] | [Spectral Convergence of Random Feature Method in Multiple Dimensions](https://arxiv.org/abs/2609.03401) | 本文证明了随机特征方法在多维情形下对Sobolev、Gevrey、超解析及带限函数类目标的谱收敛性，并给出了收敛速率随目标正则性从超指数到代数的刻画，同时建立了强形式和弱形式RFM离散化的抽象误差估计。 |
| [^140] | [TabScope: Question-Adaptive Scope Selection for Table Question Answering](https://arxiv.org/abs/2609.03395) | 该论文提出TabScope框架，通过操作感知的表格分解和问题类型预测，在大语言模型表格问答中动态选择局部子表推理或全表推理，显著提升长表格问答的准确率，并贡献了基于真实世界长表格的SLQA评测基准。 |
| [^141] | [Exploring the Potential of Contrastive Language-Image Pre-training for Multi-Source Remote Sensing Data](https://arxiv.org/abs/2609.03391) | 提出OmniRSCLIP端到端对比学习框架，通过光谱-空间基分解（SSBD）在不破坏CLIP预训练视觉知识的前提下，将其从固定RGB输入扩展至SAR、多光谱、高光谱等多源遥感传感器数据。 |
| [^142] | [Fresh Memory, Stale Plans: Dependency-Scoped Validation for Distributed LLM-Agent Memory](https://arxiv.org/abs/2609.03340) | 该论文提出PlanFence协议，通过让计划引用其使用的确切公共记录、并让执行者仅验证影响待处理动作的相关记录，解决了分布式LLM智能体中“过时计划执行”问题，在检测到计划失效时及时重规划或阻塞，避免基于过时计划执行无效动作。 |
| [^143] | [FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience](https://arxiv.org/abs/2609.03241) | FlowBalance提出一种以终端验证器的组优势来校准同模型自引导分数的自我改进方法，通过在正优势轨迹上保留、负优势轨迹上反转、无结果偏好时禁用引导，实现更稳定的推理能力自我提升。 |
| [^144] | [Speculative Macro Commit for Faster Tool-Using Agents](https://arxiv.org/abs/2609.03236) | 该论文提出推测性宏提交（SMC）机制，通过小模型在隔离环境快照上预执行从训练轨迹挖掘的多动作骨架，当大模型的动作与之匹配时直接将预执行结果提交到官方轨迹，从而加速工具使用智能体的端到端运行时间。 |
| [^145] | [MasterControl Seventeen Every Time](https://arxiv.org/abs/2609.03209) | 论文提出一种受管治的企业分析架构——语言模型仅负责理解问题意图，由确定性策略执行预先批准的分析程序，在440次运行中该策略执行方法以110/110全部满足“答案与证据”契约，而三个8B模型运行时自主规划的330次中无一达标，且固定规则确保结果完全可复现。 |
| [^146] | [Reducing Catastrophic Risk from AI with Systematic Monitoring and Evaluation of Rogue AI Progression](https://arxiv.org/abs/2609.03189) | 该论文借鉴网络安全方法论，提出了一个结构化行为指标框架，通过在AI能力与行为多个维度上设定明确的指标和阈值，实现对失控AI向灾难性威胁演进的系统性监测与评估。 |
| [^147] | [SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking](https://arxiv.org/abs/2609.03047) | SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。 |
| [^148] | [ObserverBench: Testing Mechanistic Estimates for Intervention and Control](https://arxiv.org/abs/2609.03026) | 提出 ObserverBench 基准框架，将估计精度与所选干预行动造成的损失分开评估，用以检验机制可解释性中的内部估计器是否足以胜任干预、控制与安全任务，并证明平均准确的估计并不必然带来更优的行动。 |
| [^149] | [Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation](https://arxiv.org/abs/2609.02998) | 该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。 |
| [^150] | [Evaluating Graph Neural Networks for Change-Criticality Classification in Maritime Navigation Charts](https://arxiv.org/abs/2609.02996) | 该论文提出将电子航海图数据集表示为图结构（空间对象为节点、空间与语义关系为边），并将新旧航海图间变更的重要性分类构建为图对分类问题，以评估不同图神经网络配置在此任务上的表现。 |
| [^151] | [Toward Collective-Centric Evaluation of Preference Inference for Participatory Democracy](https://arxiv.org/abs/2609.02990) | 该论文提出了一个以集体为中心的评估框架，对参与式民主平台中现有偏好推断方法进行了基准测试，揭示了这些方法并非中立，可能人为地放大、抑制或重排集体支持模式，从而重塑对商议结果的解读。 |
| [^152] | [Structure and Implementation of New Practical English Textbooks Driven by Artificial Intelligence](https://arxiv.org/abs/2609.02981) | 本文提出人工智能驱动实用英语教材的五层架构，实验证明该系统可将学生单元完成准确率提升至84.9%、口语成绩提高10.8分，并减少教师31.6%的批改时间。 |
| [^153] | [Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems via Federated Graph Learning](https://arxiv.org/abs/2609.02967) | 提出FGLGuard框架，通过图联邦学习让各运营方在本地训练GNN风险检测器且仅共享模型更新，在保护私有数据的前提下实现对LLM多智能体系统的跨组织隐私保护安全防护。 |
| [^154] | [When Optimization Becomes Manipulation: Defending Generative Search against Malicious Generative Engine Optimization](https://arxiv.org/abs/2609.02964) | 提出了GEO Defender——一个无需微调大语言模型、与攻击链对齐的两阶段防御框架（Shield Reranker与免训练护盾生成TFSG），用于防御那些事实一致且特征与高质量良性内容重合、令传统检测方法失效的恶意生成式引擎优化攻击。 |
| [^155] | [The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors](https://arxiv.org/abs/2609.02959) | 研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。 |
| [^156] | [PrivateHub: Contrastive Diffusion Model for Private Sensor-Intensive Environment Data Generation](https://arxiv.org/abs/2609.02958) | PrivateHub提出一种在扩散模型中结合对比学习的两阶段方法（应用条件预训练与应用感知微调），生成既能保持非私密应用可检测又能隐藏私密活动的合成多传感器数据流，从而解决跨传感器推断带来的隐私风险。 |
| [^157] | [Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis](https://arxiv.org/abs/2609.02947) | 该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。 |
| [^158] | [Reflect-SQL: A Self-Reflection Based Framework for Text-to-SQL](https://arxiv.org/abs/2609.02944) | Reflect-SQL是一个基于多阶段自我反思的Text-to-SQL新框架，通过知识库理解晦涩的数据库模式，并利用LLM-as-a-judge驱动的评分机制在反馈循环中迭代优化每个阶段的SQL生成结果。 |
| [^159] | [Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation](https://arxiv.org/abs/2609.02942) | 研究发现LLM评估中的评估准则文本本身编码了可预测的评估信号，且评判者在候选回答或准则被反转时往往无法可靠更新判断，这引发了对基于准则的LLM自动评估可靠性的严重质疑。 |
| [^160] | [Listen to the Latents: Self-Correcting Speech Recognition in Large Audio Language Models Through Hidden-State Interactions](https://arxiv.org/abs/2609.02940) | 该论文提出Hybrid Search纠错策略，利用基于LLM的ASR隐藏状态与基础LLM隐藏状态之间的交互特征来识别语义依赖程度高的词元并进行选择性精炼，从而显著提升LoRA适配的热启动初始化语音识别模型性能，超越重打分等全局纠错方法。 |
| [^161] | [Counterexamples as Feedback for Agent Self-Correction](https://arxiv.org/abs/2609.02892) | 本文提出 A-CEGIS 轻量级框架，将反例作为反馈来评估智能体在自然语言到正则表达式合成中的多轮自我纠正能力，在四轮预算内解决了 90% 的任务，显著优于零样本生成和通用自我纠正方法。 |
| [^162] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^163] | [Towards a Foundational Ontology for Identifying and Resolving Contradictions in Dialogue-based Human-Robot Interactions](https://arxiv.org/abs/2609.02364) | 该研究基于活动理论概念并采用METHONTOLOGY方法构建了一个基础本体，用于形式化表示和定义人机交互中基于对话的协作交互及其相关矛盾，填补了跨HRI与HAI领域的形式化计算框架的空白。 |
| [^164] | [VoRTeC: Taming Foundation Flow for One-step Real time Video Compression](https://arxiv.org/abs/2609.02291) | 提出了基于基础流模型 Wan2.1 的视频压缩框架 VoRTeC，无需访问流匹配网络的参数或梯度即可实现一步式解码、高感知保真度和跨帧组时序一致性的超低码率实时视频压缩。 |
| [^165] | [OmegaUse-SOP: SOP Engineering for Professional Computer Use from Human Demonstrations](https://arxiv.org/abs/2609.02149) | 提出了OmegaUse-SOP系统，通过人在回路的SOP工程方法，将专业计算机操作的人类演示迭代式地转化为GUI智能体可复用的SOP技能，从而解决了智能体执行特定领域专业标准操作程序的难题。 |
| [^166] | [Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models](https://arxiv.org/abs/2609.02082) | 针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。 |
| [^167] | [ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval](https://arxiv.org/abs/2609.01865) | 提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。 |
| [^168] | [A Mathematical Theory of Reusable Neural Bases for Network Compression](https://arxiv.org/abs/2609.01550) | 该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。 |
| [^169] | [LatentPress: Context Compression Beyond Text and Vision](https://arxiv.org/abs/2609.01507) | LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。 |
| [^170] | [Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search](https://arxiv.org/abs/2609.01431) | 本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。 |
| [^171] | [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065) | 该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。 |
| [^172] | [From Analytics to Tumor Boards: An Evidence-Linked Multi-Agent Workflow for Oncology Feature Extraction](https://arxiv.org/abs/2608.28974) | 本文提出并评估了Nimblemind多智能体系统（nMAS），这是一个可配置的证据关联多智能体工作流，能够从碎片化的肿瘤学文档中提取涵盖328个临床医生定义属性的结构化信息，从而大幅降低人工癌症登记的抽象负担。 |
| [^173] | [PAWBench: How Far Are We from Probabilistically Aligned World Modeling?](https://arxiv.org/abs/2608.27345) | 本文提出了PAWBench基准和PAWEval协议，首次将概率对齐作为世界模型的核心评估标准，以衡量视频生成器在重复生成时能否恢复正确的行为分布。 |
| [^174] | [Safety Does Not Compose: Non-Decaying Loop State for Autonomous LLM Agents](https://arxiv.org/abs/2608.27141) | 本文证明，针对跨多次迭代分散证据的攻击，仅基于单轨迹的LLM代理安全监控器无法有效检测（真阳性率等于假阳性率），而保留跨迭代非衰减状态的监控器才能完美分离，揭示了安全性组合的根本失败。 |
| [^175] | [Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks](https://arxiv.org/abs/2608.25390) | 本文发现拒绝训练中的首词损失塑造了拒绝方向和子空间，且重复的拒绝前缀导致拒绝几何脆弱，但多样前缀能提升稳定性并削弱消融攻击。 |
| [^176] | [VideoHarness-RSI: Recursive Harness Self-Improvement for Long-Video Understanding with Frozen Vision-Language Models](https://arxiv.org/abs/2608.24302) | 本文提出VideoHarness-RSI，通过递归搜索和自改进可执行的上下文构建程序，在不修改冻结视觉语言模型的情况下，显著提升长视频理解性能。 |
| [^177] | [AI Agents Push Humans Out of the Loop](https://arxiv.org/abs/2608.23642) | 本文指出当前AI代理发展方式削弱而非支持人类监督能力，强调应将人类监督需求置于与AI能力同等优先级，以保障有效人类参与。 |
| [^178] | [TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents](https://arxiv.org/abs/2608.22793) | TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。 |
| [^179] | [K-Bench: measuring model performance on real scientific agent requests](https://arxiv.org/abs/2608.21601) | 本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。 |
| [^180] | [Counterfactual Contrastive Analysis](https://arxiv.org/abs/2608.19032) | 本文提出了一种基于对比分析的无分类器视觉反事实生成方法，通过分离和交换数据分布中的显著因素，生成模型无关且对分类器偏见不敏感的反事实解释。 |
| [^181] | [Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems](https://arxiv.org/abs/2608.15144) | 本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。 |
| [^182] | [A Unifying Perspective on Causal World Models: From Observations to Representations to Structure](https://arxiv.org/abs/2608.13456) | 本文提出因果世界模型（CWMs）的统一形式化定义，强调世界模型需超越生成能力，捕捉实体属性及交互，以支持预测、规划和决策。 |
| [^183] | [Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence](https://arxiv.org/abs/2608.10720) | Ex-Omni-2D 提出通过“视觉思维计划”协调文本、个性化语音与基于参考视频的生成，使全模态对话智能体在语音回答的同时拥有原生视觉形象，并可将全序列教师模型蒸馏为少步骤的流式学生模型。 |
| [^184] | [WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training](https://arxiv.org/abs/2608.09447) | 提出了WDL-OPD方法，通过锚定策略与辅助策略的双策略混合约束协同训练来稳定在线策略蒸馏的反馈回路，在Qwen3的1.7B和4B规模实验中取得了最优效果。 |
| [^185] | [Neurosymbolic Reasoning with Incremental Knowledge for Sample Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/2608.02993) | 提出了一种融合可更新增量知识的神经符号分层强化学习框架，将高层符号规划与低层神经运动基元学习相结合，显著提升了稀疏奖励长程推理任务中的样本效率。 |
| [^186] | [Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model](https://arxiv.org/abs/2607.20058) | 该研究在开放权重语言模型中首次识别出材料科学机制内部表征的三个可实验验证的特征，并证明通过因果干预可以读取并调控模型对材料物理机制的表征。 |
| [^187] | [X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System](https://arxiv.org/abs/2607.17544) | X-Translator是一个低成本、模块化的级联实时语音到语音翻译系统，通过会话级运行时控制器整合流式ASR、机器翻译与提示条件TTS，并利用增量片段提交和说话人提示，在多说话人长对话场景中实现低延迟翻译并保持说话人一致性。 |
| [^188] | [Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models](https://arxiv.org/abs/2607.15565) | 研究揭示了视觉语言模型中“问题优先悖论”的机制——虽然前置问题能引导感知，但被数百个图像token遮挡的问题无法被答案token读取，并据此提出在图像后重复问题的“提示回声”这一无需训练的简单修复方法。 |
| [^189] | [PalmClaw: A Native On-Device Agent Framework for Mobile Phones](https://arxiv.org/abs/2607.13027) | PalmClaw 是一个原生运行在手机上的开源智能体框架，直接在设备端管理会话、记忆、技能与工具，并将设备能力封装为可调用的设备工具，从而突破了传统依赖 GUI 操作的移动智能体的局限。 |
| [^190] | [Learning in Curved Weight Space:Exponential-Linear Weight Reparameterization for Improved Optimization](https://arxiv.org/abs/2607.09967) | 提出一种将对称指数路径与线性路径相结合的权重重参数化方法，使加性优化更新转化为与权重幅值成比例的有效变化，从而改善神经网络的优化效果。 |
| [^191] | [PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation](https://arxiv.org/abs/2607.05915) | PCBWorld是一个基于KiCad EDA引擎构建的开源PCB布线环境与基准，使RL和LLM智能体能像人类工程师一样通过引擎原生操作与DRC反馈交互式布线，并配套提供包含679块真实开源电路板的数据集及八项引擎检查的评估指标。 |
| [^192] | [LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review](https://arxiv.org/abs/2607.05031) | 本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。 |
| [^193] | [KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment](https://arxiv.org/abs/2607.03166) | KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。 |
| [^194] | [SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling](https://arxiv.org/abs/2607.00095) | 提出SNAP-FM方法，利用样本批处理与局部PDE耦合所诱导的块稀疏结构，实现高效的批量非线性投影优化，使生成模型在推理时能精确满足物理守恒约束且计算开销大幅降低。 |
| [^195] | [Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs](https://arxiv.org/abs/2606.31413) | 提出Hard-Routed MoR-LoRA两阶段框架，通过单位尺度的硬top-1路由（而非软加权组合）选择冻结的推理LoRA专家，仅训练轻量级共享路由器和小型注意力LoRA即可实现多领域推理能力的集成。 |
| [^196] | [Transformers as Bayesian In-Context Experimenters: Smoothness-Adaptive Efficient ATE Estimation](https://arxiv.org/abs/2606.31184) | 该论文提出将变换器训练为模仿贝叶斯后验Neyman教师的“上下文实验者”，通过上下文学习摊销序贯方差估计与处理分配过程，实现对平均处理效应的平滑度自适应高效估计。 |
| [^197] | [Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization](https://arxiv.org/abs/2606.31002) | 该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。 |
| [^198] | [Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization](https://arxiv.org/abs/2606.23989) | 提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。 |
| [^199] | [Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning](https://arxiv.org/abs/2606.20954) | LRE是一种千字节级、仅用CPU、无需语言模型的学习型淘汰评分器，通过逐字提取保留任务关键历史信息，在智能体任务上以极低成本恢复保留完整历史93%的准确率，并将最坏情况峰值提示削减52%。 |
| [^200] | [LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents](https://arxiv.org/abs/2606.18388) | LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。 |
| [^201] | [SpecAlign: Efficient Specification-Grounded Alignment of Large Language Models via Synthetic Data](https://arxiv.org/abs/2606.16276) | 该论文提出了基于规范的对齐新范式及SpecAlign框架，通过结构化规则标注、可控规范实例化和多智能体对抗数据合成，直接从提供商的模型规范文档合成对齐数据，实现大语言模型与特定规范的高效对齐。 |
| [^202] | [MeEvo: Metacognitive Evolution Combined with Natural Evolution for Automatic Heuristic Design](https://arxiv.org/abs/2606.14202) | 提出MeEvo框架，通过循环耦合自然演化与元认知演化并配合从探索到利用的算子平衡，实现了自动启发式设计中推理知识保留与种群级探索的协同增效。 |
| [^203] | [GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models](https://arxiv.org/abs/2606.12821) | 提出首个环境地理空间分析智能体基准GeoNatureAgent，通过覆盖93个任务的结构化工具调用评估九个大语言模型，发现Claude Sonnet 4以60.8%的能力表现领先，DeepSeek V3.2紧随其后。 |
| [^204] | [StatefulDiscovery: Evidence-Calibrated Claim Formation in Open-Ended Scientific Discovery](https://arxiv.org/abs/2606.11851) | StatefulDiscovery通过将研究状态外化，协调前沿选择、证据获取与主张裁决，解决了开放式科学发现中的证据校准问题，从而产出更多证据充分且价值高的科学主张。 |
| [^205] | [Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them](https://arxiv.org/abs/2606.07597) | 论文揭示了预训练数据混合实验无法从小规模外推到大规模的主要原因是高质量数据的重复率随训练预算变化而改变最优混合比例，并提出通过匹配目标重复率的子采样方法，仅用1/16的目标token即可恢复接近最优的数据混合配置。 |
| [^206] | [TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment](https://arxiv.org/abs/2606.07451) | TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。 |
| [^207] | [SV-Detect: AI-generated Text Detection with Steering Vectors](https://arxiv.org/abs/2606.07313) | 本文提出SV-Detect，一种利用从冻结语言模型隐藏表示中提取的转向向量构建逐层投影特征来检测AI生成文本的方法，在跨领域、跨源模型及编辑攻击等分布偏移场景下均实现了强大的检测性能。 |
| [^208] | [ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?](https://arxiv.org/abs/2606.05553) | 本文提出ArcANE基准，通过构建刻画角色价值观、动机和关系随故事演变的“情节弧”，评估角色扮演语言代理能否在叙事的不同阶段准确呈现角色的动态发展，弥补了现有基准将角色视为固定设定的不足。 |
| [^209] | [AIP: A Graph Representation for Learning and Governing Agent Skills](https://arxiv.org/abs/2606.04781) | 该论文提出智能体指令协议（AIP），将智能体技能建模为由模式验证的YAML规范治理的有向执行图，从而同时提升智能体在重实现任务上的可靠性和技能创建与改进的效率。 |
| [^210] | [Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models](https://arxiv.org/abs/2606.02914) | 本文首次提出按架构范式和牙科专业化程度划分的二维分类框架，系统综述了97项研究，统一考察了语言生成模型、视觉基础模型和牙科专用基础模型三类大规模AI模型在牙科医疗中的关系与共同局限。 |
| [^211] | [Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling](https://arxiv.org/abs/2606.02837) | 本研究系统性审查发现 FOLIO 和 MALLS 基准中约 42% 的一阶逻辑标注存在错误，并发布修正后的真值标注以及一个 LLM 辅助框架来聚焦人工重新标注工作。 |
| [^212] | [EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglement](https://arxiv.org/abs/2606.02739) | EntangleCodec提出了一种统一的离散音频分词器，通过在量化前学习与丰富文本描述对齐的语义-声学纠缠表示，并结合流匹配扩散解码器，同时支持音频理解与高质量重建生成。 |
| [^213] | [CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents](https://arxiv.org/abs/2606.02372) | 本文提出COMAP框架，通过闭环交互使文本世界模型与LLM智能体策略共同演化——世界模型为候选动作预测未来状态反馈以支持智能体的前瞻性反思，智能体产生的同策略轨迹又通过自蒸馏反哺更新世界模型，从而摆脱了对外部奖励或验证器的依赖。 |
| [^214] | [Argument Collapse: LLMs Flatten Long-Form Public Debate](https://arxiv.org/abs/2606.01736) | 该论文首次提出并系统量化了“论点坍塌”现象：大语言模型生成的议论文会高度收敛到极少数相同论点（人类论点65.3%独特而模型仅3.4%），即使显式要求多样性也只能恢复约一半的人类论点，揭示了LLM可能使公共辩论同质化、扁平化的风险。 |
| [^215] | [HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization](https://arxiv.org/abs/2605.29843) | 提出HARP，一种可学习的Hadamard预条件自适应旋转处理器，通过稀疏蝶形块正交结构替代固定Hadamard变换，在保持与全精度模型精确等价的同时，自适应地适应层、校准分布和量化器，从而提升极端低比特LLM量化的鲁棒性。 |
| [^216] | [Skill-Conditioned Gated Self-Distillation for LLM Reasoning](https://arxiv.org/abs/2605.28791) | 本文提出SGSD，通过技能库和门控机制将自蒸馏从无条件模仿转为教师假设验证，以应对不可靠技能并提升大语言模型推理能力。 |
| [^217] | [Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations](https://arxiv.org/abs/2605.28553) | 大语言模型的拒绝行为在中间激活层就已线性可解码，利用这一信号构建的探针引导攻击方法Mechanistic AutoDAN能在保持相当攻击成功率的同时，将越狱攻击的搜索时间最多减少72%。 |
| [^218] | [MIRA: A Bilingual Benchmark for Medical Information Response Audit](https://arxiv.org/abs/2605.28025) | 该论文提出了首个双语医疗基准MIRA，揭示了大语言模型在应对低健康素养用户提问时会系统性遗漏关键医疗信息、减少后续行动建议的“差异化信息稀释”（DID）这一安全隐患。 |
| [^219] | [EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation](https://arxiv.org/abs/2605.26785) | 提出EmoDistill离线蒸馏框架，通过IQL情感选择器与LoRA微调的表达策略相结合，将大模型间对抗谈判中的情感技能迁移到小型语言模型智能体，使其能够抵御情感操纵并维护用户谈判目标。 |
| [^220] | [Identifying AI Web Scrapers Using Canary Tokens](https://arxiv.org/abs/2605.13706) | 本文提出了一种基于蜜罐令牌的新技术，能够准确且自动化地识别与大型语言模型相关的网络爬虫，克服了现有识别方法不可靠、不可扩展的缺陷。 |
| [^221] | [Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs](https://arxiv.org/abs/2605.12462) | 本文提出了DR-Gym，一个开源的、与Gymnasium兼容的在线仿真环境，用于从电力公司视角训练和评估需求响应策略，解决了离线历史数据无法捕捉价格信号与用户行为之间动态交互反馈循环的问题。 |
| [^222] | [Beyond Reproducibility: Towards Security-Aware Evaluation of Research Artifacts](https://arxiv.org/abs/2605.06508) | 该论文对2023至2025年四大顶级安全会议发表的1,388个研究工件进行静态分析，提出上下文感知的安全评估分类体系，发现其中44.80%的安全发现具有真实安全相关性，并推出了SAFE工具以支持可扩展的工件安全评估。 |
| [^223] | [Causal Probing for Internal Visual Representations in Multimodal Large Language Models](https://arxiv.org/abs/2605.05593) | 该研究提出基于激活引导的因果框架，揭示了多模态大语言模型中实体知识局部化编码而抽象概念全局分布的分化现象，并证明模型深度的增加是编码复杂抽象概念这一缩放定律背后的机制性驱动因素。 |
| [^224] | [When Chain-of-Thought Fails, the Solution Hides in the Hidden States](https://arxiv.org/abs/2604.23351) | 研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。 |
| [^225] | [CASCADE: A Component Ablation and Corpus Audit of a Layered Local Defense for MCP-Based Systems](https://arxiv.org/abs/2604.17125) | 本文通过对CASCADE分层本地防御进行组件消融并对其评估语料库进行全面审计，揭示了判定约定与样本来源对MCP防御评估结果的决定性影响——仅聚合方式的选择即可使误报率在1.51%与11.70%之间产生巨大差异。 |
| [^226] | [Anonymization, Not Elimination: Utility-Preserved Speech Anonymization](https://arxiv.org/abs/2604.17000) | 提出了一种两阶段语音匿名化框架，通过生成式语音编辑模型替换个人可识别信息以保护内容隐私，并引入基于流匹配的F3-VA框架保护声音隐私，在实现匿名化的同时保持语音数据在ASR、TTS、SER等下游任务中的效用。 |
| [^227] | [LLM Evaluation as Tensor Completion: Low Rank Structure and Semiparametric Efficiency](https://arxiv.org/abs/2604.05460) | 该论文将LLM评估建模为通过成对比较观测低秩潜在分数张量的半参数推断问题，推导了有效影响函数与半参数有效性界，并构造了具有渐近正态性的单步去偏估计量，为LLM排行榜提供了严谨的不确定性量化方法。 |
| [^228] | [One Model to Translate Them All? A Journey to Mount Doom for Multilingual Model Merging](https://arxiv.org/abs/2604.02881) | 该论文系统研究了多语言机器翻译中的权重空间模型合并，揭示了显著的方向性不对称现象——当模型共享目标语言时合并相对有效但无法保留各语言模型的峰值性能，而当目标语言不同时性能则急剧下降。 |
| [^229] | [CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery](https://arxiv.org/abs/2604.01658) | CORAL是首个面向开放式问题的自主多智能体进化框架，通过共享持久记忆、异步多智能体执行和心跳式干预实现智能体的自主探索、反思与协作，在10个任务上以远少于以往方法的评估次数取得3至10倍更高的提升率，刷新了最先进水平。 |
| [^230] | [A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling](https://arxiv.org/abs/2603.27341) | 本文比较研究了数据、算力与规模化在外科AI中的潜力与局限，探讨现代通用AI能否以及在多大程度上辅助外科实践。 |
| [^231] | [LRConv-NeRV: Low Rank Convolution for Efficient Neural Video Compression](https://arxiv.org/abs/2603.18261) | LRConv-NeRV通过用结构化低秩可分离卷积替换密集3x3卷积层并逐阶段渐进应用低秩分解，将NeRV解码器计算复杂度降低68%、模型大小降低9.3%，同时几乎不损失视频重建质量。 |
| [^232] | [The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas](https://arxiv.org/abs/2603.11842) | 本研究通过系统检索并综合分析28篇二次综述与路线图论文，梳理了生成式人工智能在信息系统领域带来的益处、挑战及未来研究议程。 |
| [^233] | [NeuroWeaver: An Autonomous Evolutionary Agent for Exploring the Programmatic Space of EEG Analysis Pipelines](https://arxiv.org/abs/2602.13473) | 该论文提出NeuroWeaver，一个基于大语言模型驱动的自主进化智能体，将脑电图分析流水线工程重新表述为离散约束优化问题，通过融入神经生理学先验知识自动探索并生成可执行代码，从而在多样的EEG数据集和任务上实现低成本、可泛化的分析。 |
| [^234] | [PeroMAS: A Multi-agent System of Perovskite Material Discovery](https://arxiv.org/abs/2602.13312) | 提出了PeroMAS——一个通过模型上下文协议（MCP）封装钙钛矿专用工具的多智能体系统，能够在多目标约束下实现钙钛矿材料发现全工作流程的端到端优化。 |
| [^235] | [FedPS: Federated Preprocessing for structured data via aggregated Statistics](https://arxiv.org/abs/2602.10870) | 提出了FedPS框架，利用数据草图技术在联邦环境下通过聚合统计实现结构化数据的高效预处理（包括特征缩放、编码、离散化和缺失值插补），解决了联邦学习中预处理阶段被忽视的问题。 |
| [^236] | [Discovering High Level Patterns from Simulation Traces](https://arxiv.org/abs/2602.10009) | 提出一种基于程序合成的无监督学习方法，将仿真轨迹转换为高层次结构模式的稀疏标注序列，从而提升大型语言模型对物理系统推理与验证的有效性和可扩展性。 |
| [^237] | [Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge](https://arxiv.org/abs/2602.09341) | 提出AgentAuditor框架，通过将多智能体推理轨迹组织成推理树并在关键分歧点比较分支级证据来裁决冲突，结合反共识偏好优化训练裁决器，其性能优于多数投票和LLM-as-Judge。 |
| [^238] | [Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models](https://arxiv.org/abs/2602.07106) | 该论文提出Ex-Omni框架，通过带混合变形协同监督的语音单元生成器与非自回归混合变形解码器将语义推理与时间生成解耦，并结合令牌即查询门控融合接口和120万样本弱监督数据集InstructS2SF-1200K，首次使全模态大语言模型能够联合生成语音与3D面部动画。 |
| [^239] | [F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare](https://arxiv.org/abs/2602.06717) | 本文提出 F-GRPO，借鉴 Focal loss 设计了难度感知的缩放系数，对高成功率采样组的更新降权，从而防止 RLVR 训练中的策略因组采样遗漏稀有正确解而过度集中于常见解。 |
| [^240] | [Temperature Scaling Attack Disrupting Model Confidence in Federated Learning](https://arxiv.org/abs/2602.06638) | 该论文提出了温度缩放攻击（TSA），一种新型联邦学习训练时攻击，通过学习率-温度耦合机制在保持模型准确率不变的情况下破坏置信度校准，从而威胁依赖置信度信号的任务关键型系统的风险决策逻辑。 |
| [^241] | [Not All Preferences Deserve Gradients: Understanding Gradient Utility in Offline Reasoning Alignment](https://arxiv.org/abs/2602.01207) | 提出SAGE方法，通过难度分层候选池和前向传播的信号-曲率分数筛选偏好对，揭示并非所有偏好都值得梯度更新，最有效的监督来自模型可靠犯错但曲率低的稳定自信错误。 |
| [^242] | [Complete Identification of Deep ReLU Networks through {\L}ukasiewicz Logic](https://arxiv.org/abs/2602.00266) | 本文借鉴香农用布尔逻辑分析开关电路的思想，建立了基于Łukasiewicz多值逻辑的符号演算，证明两个非退化ReLU网络实现相同函数当且仅当其中一个可通过有限次应用MV逻辑公理由另一个推导得到，从而完整刻画了深度ReLU网络函数表示的非唯一性。 |
| [^243] | [VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models](https://arxiv.org/abs/2601.19956) | 本文提出了首个用于评估语音语言模型“交互隐私”的基准VoxPrivacy，填补了现有基准在说话人身份感知响应和情境性隐私敏感信息评估方面的空白。 |
| [^244] | [Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment](https://arxiv.org/abs/2601.16027) | 提出了CS-VAR框架，通过检索增强的大语言模型在训练中将跨会话行为证据的推理洞察蒸馏给轻量级模型，使其能够识别直播中跨场次重复出现的风险模式，实现高效实时的直播风险评估。 |
| [^245] | [Relational Linearity is a Predictor of Hallucinations](https://arxiv.org/abs/2601.11429) | 该论文提出关系线性性可预测语言模型的幻觉：由于抽象表示方案，语言模型能轻松为线性关系中不存在的主体生成看似合理的客体从而导致幻觉，而在面对非线性关系时这种机制失效，幻觉更容易避免。 |
| [^246] | [HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning](https://arxiv.org/abs/2601.10187) | 该论文提出了Sand-Glass音节级时长约束翻译基准和Homura强化学习框架，通过新颖的动态音节比率奖励有效解决LLM翻译的跨语言冗长偏差问题，使其适用于字幕、配音等时间受限场景。 |
| [^247] | [PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization](https://arxiv.org/abs/2601.10029) | 提出PaperScout自主智能体，将学术论文检索重构为序贯决策过程，并通过过程感知的序列级策略优化，解决了标准强化学习在多轮智能体任务中词元级优化与序列级交互之间的粒度不匹配问题。 |
| [^248] | [Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models](https://arxiv.org/abs/2601.08955) | 提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。 |
| [^249] | [FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation](https://arxiv.org/abs/2512.15116) | 提出FADTI框架，通过傅里叶偏置投影（FBP）模块在扩散去噪过程中注入可学习的频率感知偏置，并支持DFT、STFT、FSST多种频谱变换，从而有效提升多变量时间序列插补对周期性和非平稳模式的恢复能力。 |
| [^250] | [Evolving Excellence: Automated Optimization of LLM-based Agents](https://arxiv.org/abs/2512.09108) | 本文提出ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子自动联合优化LLM智能体的提示词、工具描述和参数等配置，无需架构修改即可显著提升智能体性能。 |
| [^251] | [Mixed Data Clustering Survey and Challenges](https://arxiv.org/abs/2512.03070) | 本文提出了一种基于预拓扑空间的混合数据聚类方法，能够有效处理同时包含数值型和分类型变量的异构数据，并提供层次化、可解释的聚类结果。 |
| [^252] | [AnyBox: Efficient Zero-Shot 9DoF Pose Estimation of Boxes for Robotic Manipulation](https://arxiv.org/abs/2511.15884) | AnyBox是一个高效的零样本框架，通过利用箱体的几何规则性，在单张RGB-D观测上交替进行位姿与尺度估计，从而实现对杂乱遮挡环境中箱体9自由度位姿（6D位姿+3D尺寸）的联合恢复，无需物体特定的CAD模型。 |
| [^253] | [Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling](https://arxiv.org/abs/2511.10866) | 该论文提出一种短窗口滑动学习框架，通过LLM自动标注将视频切分为1-2秒短片段构建细粒度数据集，在保留时间连续性的同时实现了高精度实时暴力检测，在RWF-2000上达到95.25%准确率。 |
| [^254] | [User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios](https://arxiv.org/abs/2510.20721) | 该研究通过94人的用户实验发现，用户对LLM在隐私敏感场景下响应的评价彼此一致性较低，这表明此前以代理LLM作为评审的基准测试结果可能与真实用户的隐私和有用性感知存在显著偏差。 |
| [^255] | [WELD: The First Naturalistic Long-Period Small-Team Workplace Emotion Dataset for Ubiquitous Affective Computing](https://arxiv.org/abs/2510.15221) | WELD是首个结合数年持续时间、自然职场情境、稳定小团队结构与完全被动感知协议的职场情绪数据集，基于中国某软件公司49名员工超过30个月的面部表情数据构建。 |
| [^256] | [EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering](https://arxiv.org/abs/2509.25175) | EasySteer是一个基于vLLM构建的高性能、可扩展的大语言模型推理时引导统一框架，相比现有框架实现了10.8-22.3倍的加速，并提供模块化可插拔接口、细粒度参数控制以及八个应用领域的预计算引导向量。 |
| [^257] | [Human Psychometric Questionnaires Mischaracterize LLM Behavior](https://arxiv.org/abs/2509.10078) | 该研究发现，人类心理测量问卷中的题目含有明显的词汇线索，会让大语言模型识别出被测构念并给出符合社会期望的回答，因此基于问卷得到的模型人格与价值观画像并不能反映其在真实日常用户交互中的实际生成行为。 |
| [^258] | [Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring](https://arxiv.org/abs/2508.15038) | 提出了一种仅依赖单个机载RGB相机、无需集中式通信的去中心化多旋翼无人机系统，实现了在自然栖息地中对野生动物的鲁棒识别与跟踪。 |
| [^259] | [Measuring Harmfulness of Computer-Using Agents](https://arxiv.org/abs/2508.00935) | 该论文提出了 CUAHarm 基准，通过 104 个专家撰写的真实滥用任务和基于规则的可验证沙盒环境评估计算机使用代理的滥用风险，发现前沿语言模型（如 Gemini 2.5 Pro 成功率达 90%）即使没有越狱提示也会以高成功率执行恶意计算机操作。 |
| [^260] | [Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications](https://arxiv.org/abs/2508.00669) | 本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。 |
| [^261] | [Traceable TTS: Toward Watermark-Free TTS with Strong Traceability](https://arxiv.org/abs/2507.03887) | 该论文提出首个无水印的可追溯TTS框架，通过TTS模型与判别器的联合训练实现合成语音的模型归因，在保持甚至略微提升音频质量的同时显著增强了可追溯性的泛化能力。 |
| [^262] | [ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition](https://arxiv.org/abs/2506.10226) | 提出ScoreMix方法，利用扩散模型的分数可组合性、在无需外部模型或数据集的情况下，通过混合判别器嵌入空间中相距较远的类别生成分数条件合成样本，为识别任务带来最高3%的平均性能提升。 |
| [^263] | [RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data](https://arxiv.org/abs/2505.19030) | 提出RECAST框架，通过从真实提示-响应对中提取约束，高效合成每个样本包含远超10个约束条件的数据集，突破了现有基准限制，拓展了大语言模型复杂指令遵循能力的边界。 |
| [^264] | [LightEMMA: A Longitudinal Evaluation of Vision-Language Models for Autonomous Driving](https://arxiv.org/abs/2505.00284) | 提出了LightEMMA纵向评估框架，通过轻量级统一协议在nuScenes基准上评估15个视觉语言模型的驾驶性能，发现新一代VLM的规模和推理能力提升并不能持续带来更好的自动驾驶表现。 |
| [^265] | [Sionna RT: Technical Report](https://arxiv.org/abs/2504.21719) | Sionna RT是一个开源、GPU加速且可微分的射线追踪器，本文详细介绍了其高效模拟无线电波传播（包括信道脉冲响应与无线电地图计算）的核心算法、Sionna 1.0全面重构带来的速度与内存效率提升以及现有算法的局限性。 |
| [^266] | [AgentRM: Enhancing Agent Generalization with Reward Modeling](https://arxiv.org/abs/2502.18407) | 本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。 |
| [^267] | [LDC: Learning to Generate Research Idea with Dynamic Control](https://arxiv.org/abs/2412.14626) | 提出首个结合监督微调与可控强化学习的两阶段框架，利用多维奖励模型和细粒度反馈动态控制研究想法生成，从而在新颖性、可行性和有效性之间实现平衡，提升大语言模型科研构思质量。 |
| [^268] | [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) | 本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。 |
| [^269] | [Data Market Design through Deep Learning.](http://arxiv.org/abs/2310.20096) | 这项研究介绍了使用深度学习进行收入最优数据市场设计的应用，旨在扩展前沿研究领域。 |

# 详细

[^1]: 训练式编译：将自然语言规范转化为本地神经函数

    Compile by Training: Turning Natural-Language Specifications into Local Neural Functions

    [https://arxiv.org/abs/2609.04199](https://arxiv.org/abs/2609.04199)

    提出“训练式编译”方法，将自然语言规范编译为可复用的本地神经函数，通过教师模型生成的示例训练小型适配器，无需每次调用远程大模型即可达到83.6%的语义准确率。

    

    许多反复出现的文本功能很容易描述，但难以用规则来实现，而每次输入都调用大型远程模型会带来重复的成本、延迟以及对服务提供商的依赖。我们提出了“训练式编译”，它将自然语言规范转化为可复用的神经函数。在编译阶段，教师模型生成任务特定的示例，用于为一个紧凑的解释器训练小型适配器。生成的函数无需教师模型即可运行，并且可以像普通软件一样进行存储、版本管理和组合。在FuzzyBench-Hard（一个Program-as-Weights快速编译器无法产生精确匹配的子集）上，训练式编译达到了83.6%的语义准确率。这一更高的准确率伴随着更高的编译时间成本：大约需要一分钟，而快速编译器只需几秒钟。我们将该编译器部署在一个公开的交互式服务中，并在多站点网站上展示了编译后的函数。

    arXiv:2609.04199v1 Announce Type: cross  Abstract: Many recurring text functions are easy to describe but difficult to implement with rules, while calling a large remote model for every input introduces repeated cost, latency, and dependency on a provider. We present compile by training, which turns a natural-language specification into a reusable neural function. At compile time, teacher models generate task-specific examples that are used to train a small adapter for a compact interpreter. The resulting function runs without the teachers and can be stored, versioned, and composed like ordinary software. On FuzzyBench-Hard, a subset on which the Program-as-Weights fast compiler produced no exact matches, compile by training reaches 83.6% semantic accuracy. This higher accuracy comes with a higher compile-time cost: roughly a minute rather than seconds for the fast compiler. We deploy the compiler in a public interactive service and demonstrate compiled functions in a multi-site websit
    
[^2]: 洁净的工程，不稳定的测量：黑盒大语言模型观测者在共享端点上预注册的可靠性失败

    Clean Engineering, Unstable Measurement: A Preregistered Reliability Failure of Black-Box LLM Observers on Shared Endpoints

    [https://arxiv.org/abs/2609.04198](https://arxiv.org/abs/2609.04198)

    本文通过两项预注册审计（共52,988次请求尝试）发现，黑盒大语言模型评判者作为测量仪器存在严重可靠性缺陷——即便工程执行记录完美，相同请求的重复排名一致性仅0.400、字节级相同的次日重放一致性仅0.78，远低于0.90和0.99的预设标准，其根源在于标签映射偏置、信号低于噪声底多个数量级以及逐字排列读数放大的噪声。

    

    语言模型评判者如今负责把关训练数据、为生成内容打分并驱动排行榜。评判者因而成为一种测量仪器，其建立在一个很少被言明的假设之上：同一请求发送给同一模型名称，明天读出的结果应当相同。我们在两项预注册活动中对该假设进行了审计，所有阈值均事先固定；结果两项活动都未能通过仪器验证这一步。在52,988次经审计的请求尝试中，同一窗口内重复排名的一致性仅为Spearman 0.400（要求为0.90），字节级完全相同的次日重放一致性为0.78（要求为0.99），而与此同时每次执行的工程记录都达到了完美水平。三种机制解释了这一差距：标签到含义的映射对读数造成的偏置与信号本身同样强；候选者之间的差距低于仪器自身噪声底达七个数量级；以及字节级完全相同的输入返回不同排名——这种噪声会被精确排列读数方式进一步放大。无论是指标替代（摘要在此处截断）……

    arXiv:2609.04198v1 Announce Type: new  Abstract: Language-model judges now gate training data, score generations, and drive leaderboards. The judge is then a measurement instrument, resting on one rarely stated assumption: the same request, sent to the same model name, reads the same tomorrow. We audited that assumption in two preregistered campaigns with every threshold fixed in advance; neither got past validating its instrument. Across 52,988 audited request attempts, same-window repeat rankings agreed at Spearman 0.400 against a required 0.90, and byte-identical next-day replays agreed at 0.78 against a required 0.99, each time with the execution record at ceiling. Three mechanisms explain the gap: a label-to-meaning mapping that biased readouts as strongly as the signal; candidate gaps seven orders of magnitude below the instrument's own noise floor; and byte-identical inputs returning different rankings, a noise that exact-permutation readouts compound. Neither metric substitutio
    
[^3]: ESPO：通过诊断、多样化与稳定化实现的错误结构化提示优化

    ESPO: Error-Structured Prompt Optimization via Diagnose, Diversify, and Stabilize

    [https://arxiv.org/abs/2609.04197](https://arxiv.org/abs/2609.04197)

    ESPO通过诊断错误模式、多样化候选生成和稳定性选择三个阶段，解决了进化式提示优化中的提示膨胀问题，在七个NLP基准上平均准确率超越GEPA达3.76个百分点，同时提示词更短47%且推理更快。

    

    以GEPA为代表的进化式提示优化器存在提示膨胀问题：每次迭代都会追加规则和注意事项，导致提示词长度增至3倍，但准确率却没有提升。我们将这一问题归因于三个缺陷：错误观察不完整、搜索多样性有限以及选择不可靠，并提出ESPO（错误结构化提示优化），将提示优化分解为三个阶段：Diagnose（诊断）在一轮中将所有训练错误聚类为结构性模式；Propose（提议）通过四种具有独立偏好的互补策略生成候选提示；Select（选择）应用自助法稳定性选择。在七个公开NLP基准——Tweet、MMLU、GSM8K、HotpotQA、ScoNe、HoVer和PUPA上，ESPO相比最先进方法平均准确率提升3.76个百分点（74.67%对比GEPA的70.91%），在所有数据集上持平或超越GEPA，同时生成的提示词缩短47%（1,004字符对比1,878字符）且推理速度更快。

    arXiv:2609.04197v1 Announce Type: cross  Abstract: Evolutionary prompt optimizers such as GEPA suffer from prompt bloat: each iteration appends rules and caveats, producing prompts up to 3$\times$ longer yet no more accurate. We trace this to three deficiencies - incomplete error observation, limited search diversity, and unreliable selection - and propose ESPO (Error-Structured Prompt Optimization), which decomposes prompt optimization into three phases: Diagnose clusters all training errors into structural patterns in one round; Propose generates candidates via four complementary strategies with independent biases; Select applies bootstrap stability selection. On seven public NLP benchmarks - Tweet, MMLU, GSM8K, HotpotQA, ScoNe, HoVer, and PUPA - ESPO improves average accuracy by $+$3.76 pp over the state-of-the-art (74.67% vs 70.91% for GEPA), matching or exceeding GEPA on every dataset while producing prompts 47% shorter (1,004 vs 1,878 chars) and faster at inference. Cross-model e
    
[^4]: 一个编辑器，多种编辑：面向多样化视频编辑的统一免训练框架

    One Editor, Many Edits: A Unified Training-Free Framework for Diverse Video Editing

    [https://arxiv.org/abs/2609.04190](https://arxiv.org/abs/2609.04190)

    EditVid是一个免训练的统一视频编辑框架，通过结合稀疏因果记忆、基于对应关系的后注意力token注入和软潜变量混合，在单一框架内同时支持指令引导与主体引导的多样化视频编辑，并在FiVE基准上大幅超越现有最强免训练基线。

    

    视频编辑涵盖多种编辑范式，然而在单一统一框架内实现高质量的指令引导和主体引导编辑仍然具有挑战性。我们提出了EditVid，一个免训练框架，它结合了用于局部一致性的稀疏因果记忆、用于长程身份保持的基于对应关系的后注意力token注入，以及用于编辑局部性的软潜变量混合。同一框架可支持指令引导和参考引导的编辑，包括风格迁移、属性修改、物体插入、部件级编辑和主体替换。在FiVE基准上，EditVid取得了78.16的FiVE-Acc，而参与评估的最强免训练基线仅为58.95，同时它在IVEBench上也获得了具有竞争力的结果。用户研究进一步表明，与7种竞争方法相比，EditVid获得了51.8%的总体偏好率。

    arXiv:2609.04190v1 Announce Type: cross  Abstract: Video editing spans diverse editing paradigms, yet achieving high-quality instruction-guided and subject-guided editing within a single unified framework remains challenging. We introduce EditVid, a training-free framework combining sparse causal memory for local coherence, correspondence-based post-attention token injection for long-range identity preservation, and soft latent blending for edit locality. The same framework supports instruction-guided and reference-guided edits, including style transfer, attribute modification, object insertion, part-level editing, and subject replacement. On FiVE, EditVid achieves 78.16 FiVE-Acc, compared with 58.95 for the strongest evaluated training-free baseline, while obtaining competitive results on IVEBench. A user study further shows a 51.8\% overall preference for EditVid over 7 competing methods.
    
[^5]: 先观察后合成：面向弱监督密集视频描述的VLM引导过渡事件发现

    Seeing Before Synthesizing: VLM-Guided Transition Event Discovery for Weakly-Supervised Dense Video Captioning

    [https://arxiv.org/abs/2609.04183](https://arxiv.org/abs/2609.04183)

    该论文提出SBS框架，先利用视觉语言模型观察事件间间隔、生成帧级叙述并检测过渡事件，从而仅在确有需要之处自适应地提供具有视觉依据的语言指导，克服了以往用LLM合成过渡描述缺乏视觉依据且位置固定的问题，提升了弱监督密集视频描述的性能。

    

    弱监督密集视频描述旨在为未剪辑视频定位并描述多个事件，其中每个视频仅提供一组有序的事件级描述。近期工作通过大语言模型（LLM）合成辅助过渡描述以提供额外的视觉-语言对齐信号，但这些描述缺乏视觉依据，且被机械地以固定位置和固定时长分配到每个事件间间隔。为解决这些问题，我们提出了Seeing Before Synthesizing（SBS，先观察后合成）框架，该框架仅在确有需要之处自适应地提供具有视觉依据的语言指导。借助视觉语言模型（VLM），我们为事件间间隔生成帧级叙述，并根据跨帧的语义变化检测过渡事件。对于识别出的过渡事件，我们通过将时间中点与语义变化点融合，并选择使视觉-语言对齐最大化的宽度，来细化事件间的时间掩码。在ActivityNet Captions和You（原文在此处截断）上的实验……

    arXiv:2609.04183v1 Announce Type: cross  Abstract: Weakly-Supervised Dense Video Captioning aims to localize and describe multiple events in untrimmed videos given only an ordered set of event-level captions per video. Recent work synthesizes auxiliary transition captions via LLM to provide additional vision-language alignment, but these captions lack visual grounding and are rigidly assigned to every inter-event gap at a fixed location and duration. To address these, we propose Seeing Before Synthesizing (SBS), a framework that adaptively provides visually grounded linguistic guidance only where warranted. Leveraging a VLM, we generate frame-level narratives for the inter-event gaps and detect transitions from the semantic variation across them. For identified transitions, we then refine inter-event temporal masks by blending the temporal midpoint with the semantic change point and selecting the width that maximizes vision-language alignment. Experiments on ActivityNet Captions and Yo
    
[^6]: 预训练期间的知识获取？大语言模型借助辅助视图学得更好

    Knowledge Acquisition During Pre-training? Large Language Models Learn Better With Auxiliary Views

    [https://arxiv.org/abs/2609.04180](https://arxiv.org/abs/2609.04180)

    研究发现，在预训练中将token预算从文档重复转移到辅助视图（知识的重新表述）能提升大语言模型的学习效果，即使对事实回忆也有效，且不依赖教师模型的强弱。

    

    我们对大语言模型（LLMs）在预训练期间如何获取知识的理解仍存在空白。我们提出假设：辅助视图（auxiliary views），即知识的重新表述，对学习具有因果性的帮助。我们设计了受控实验来隔离验证这一假设。首先，我们确认重复是知识获取的必要条件，并澄清释义改写仅在小批量大小（batch size）时才有所帮助。其次，在保持token预算固定的情况下，将token从文档重复中分配给辅助视图可以提升学习效果——反直觉的是，即使对于事实回忆也是如此。第三，辅助视图的有效性并不取决于生成它的教师模型的强弱。第四，我们识别出两类知识形式——上下文性知识和基础性知识——它们能够在存在先前知识空白的情况下帮助学习。最后，我们通过逐层偏置（layer-wise biases）和压缩（compression）机制，从机理层面考察了这些效应的表现。总之，我们的发现表明辅助表示……

    arXiv:2609.04180v1 Announce Type: cross  Abstract: Gaps remain in our understanding of how large language models (LLMs) acquire knowledge during pre-training. We posit that auxiliary views, reformulations of knowledge, are causally helpful for learning. We design controlled experiments to isolate this. First, we confirm that repetition is necessary for acquisition and clarify that paraphrasing helps only at smaller batch sizes. Second, holding the token budget fixed, allocating tokens from document repetition to auxiliary views improves learning, counterintuitively, even for factual recall. Third, the effectiveness of auxiliary views is not contingent on the strength of the teacher model that generates them. Fourth, we identify forms of knowledge, contextual and foundational, that aid learning in the presence of prior knowledge gaps. Finally, we examine how these effects manifest mechanistically via layer-wise biases and compression. Together, our findings suggest that auxiliary repres
    
[^7]: 因果概率解释的计算可行框架

    A Computationally Feasible Framework for Causal Probabilistic Explanation

    [https://arxiv.org/abs/2609.04177](https://arxiv.org/abs/2609.04177)

    本文提出概率因果影响（PCI）框架，将实际因果性理论与Pearl的必要性/充分性概率相结合，把因果解释问题重新表述为可通过蒙特卡洛方法高效估计的问题，从而在保持因果分析原则性的同时实现对大规模模型的计算可行解释。

    

    解释为什么一个特定结果会发生，以及哪些输入应当承担责任或获得功劳，是哲学、科学和政策分析的核心问题。现有工具分为两大阵营：实际因果性理论能够给出有原则的判断，但仅适用于玩具规模的模型，因为其计算需要枚举反事实场景；而像SHAP（甚至因果SHAP）这样的可扩展归因方法则至少部分地忽略了生成数据的因果结构，可能给出与细致因果分析相矛盾的答案。我们提出了概率因果影响（PCI）来弥合这一差距。PCI建立在实际因果性理论和Pearl提出的必要性与充分性概率概念之上，但将可解释性问题重新表述为概率因果模型上的估计问题，该模型可以通过蒙特卡洛方法轻松近似。通过在“候选解释”上指定一个分布，以及在反事实场景上指定一个分布（摘要在此处截断）……

    arXiv:2609.04177v1 Announce Type: new  Abstract: Explaining why a specific outcome occurred, and which inputs deserve the blame or credit, is central to philosophical, scientific, and policy analysis. Existing tools split into two camps. The theory of actual causality (AC) gives principled verdicts, but only for toy-sized models, because computing them requires enumerating counterfactual scenarios. Scalable attribution methods like SHAP (or even causal SHAP) at least partially ignore the causal structure that generated the data, and can give answers that conflict with a careful causal analysis. We close this gap with Probabilistic Causal Impact (PCI).   PCI builds on actual causality and on Pearl's notions of probability of necessity and sufficiency, but recasts the question of explainability as an estimation problem on a probabilistic causal model that is easily approximated via Monte Carlo. By specifying a distribution over "candidate explanations," a distribution over counterfactual
    
[^8]: 重新思考大语言模型的在线策略蒸馏 II：单个训练样本

    Rethinking On-Policy Distillation of Large Language Models II: One Training Example

    [https://arxiv.org/abs/2609.04172](https://arxiv.org/abs/2609.04172)

    该研究发现仅用一个训练样本进行在线策略蒸馏就能持续改进并达到全数据训练的大部分性能，原因是单个查询即可覆盖全数据训练71.5%的状态，而16个语义不同的查询可达到98.9%的覆盖率并完全匹配全数据训练的效果。

    

    arXiv:2609.04172v1 公告类型：新论文 摘要：在线策略蒸馏（OPD）将学生模型生成的轨迹与教师模型提供的密集token级监督相结合。现有工作主要研究其算法行为，而训练数据的作用尚不明确。我们通过在单个查询上进行训练，在数据极简极限下考察训练数据的作用。单样本OPD在数百步内持续改进，并在跨任务领域和模型家族中恢复了全数据OPD的大部分增益。我们通过训练过程中访问的状态以及学生模型与教师模型对齐的速率来解释这一结果。我们测量了“状态覆盖率”，即某个查询集的轨迹所能达到的全数据OPD访问状态的比例。单个查询已能达到71.5%的覆盖率，其中大部分在前100步内实现。添加语义上不同的查询会使覆盖率和验证准确率同步提升，直到16个查询达到98.9%的覆盖率并与全数据训练的表现相匹配。然而，无论使用何种查询，对齐速度都会以相似的速率减缓……

    arXiv:2609.04172v1 Announce Type: new  Abstract: On-policy distillation (OPD) combines student-generated rollouts with dense token-level supervision from a teacher. Existing work has mainly studied its algorithmic behavior, leaving the role of training data unclear. We examine this role at the data-minimal limit by training on a single query. One-shot OPD keeps improving for hundreds of steps and recovers most of full-data OPD's gain across task domains and model families. We explain this result through the states visited during training and the rate at which the student aligns with the teacher. We measure \emph{state coverage}, the fraction of the states full-data OPD visits that a query set's rollouts reach. A single query already reaches \(71.5\%\), most of it within the first 100 steps. Adding semantically distinct queries raises coverage and validation accuracy together, until 16 queries reach \(98.9\%\) and match full-data training. Yet alignment slows at a similar pace whether O
    
[^9]: 自主研究群体中涌现性作弊与举报行为的案例研究

    A Case Study on Emergent Cheating and Whistleblowing in Autonomous Research Swarms

    [https://arxiv.org/abs/2609.04170](https://arxiv.org/abs/2609.04170)

    本研究通过100个自主LLM智能体协作证明数学猜想的案例，首次记录了在无外部干预的情况下，AI群体中作弊行为自发涌现并经由共享知识库和对等消息传染扩散，同时另一批智能体自发产生举报、审计等反制行为的社会动力学全过程。

    

    多智能体AI科学生态系统依赖于智能体所拥有的工具，这些工具使它们能够相互交流、协调并在彼此的工作基础上进行构建。然而，这种共享基础设施也可能通过为无意和不良行为的传染性传播创造基质而引入漏洞。我们报告了一项关于由100个自主LLM智能体组成的科研集体的案例研究，这些智能体的任务是证明形式化数学猜想。在该群体中，作弊行为自发涌现，随后受到“举报者”的挑战——而这一切都是在没有任何外部干预的情况下发生的。当单个智能体发现了评估系统中的一个漏洞时，该漏洞便通过共享知识库在集体中传播开来，随后又通过对等消息进一步扩散。尽管早期有所抵触，一批智能体在竞争压力下还是采用了该漏洞。另一组智能体则产生了涌现性的反制反应：审计欺诈性证明、警示同伴……

    arXiv:2609.04170v1 Announce Type: new  Abstract: Multi-agent AI science ecosystems rely on agents possessing tools that allow them to communicate, coordinate, and build on each other's work. Yet this shared infrastructure can also introduce vulnerabilities by creating a substrate for the contagious spread of unintended and undesirable behaviors. We report a case study on a research collective of 100 autonomous LLM agents tasked with proving formal mathematical conjectures. Within the swarm, cheating spontaneously emerged and was later challenged by whistleblowers - both without any external intervention. When a single agent discovered an exploit in the evaluation system, it propagated across the collective via a shared knowledge library and later through peer-to-peer messages. Despite early reluctance, a cohort of agents adopted the exploit in response to competitive pressure. A separate group of agents produced an emergent counter-response: auditing fraudulent proofs, alerting peers a
    
[^10]: SWE-Gate：对软件工程智能体而言，通过功能测试还不够

    SWE-Gate: Passing Functional Tests Is Not Enough for Software Engineering Agents

    [https://arxiv.org/abs/2609.04167](https://arxiv.org/abs/2609.04167)

    提出 SWE-Gate 基准，从真实 PR 评审评论中提取评审约束并构建带独立功能测试与约束测试的仓库级修复实例，首次将软件工程智能体的问题解决能力与评审约束遵守能力区分评估。

    

    仓库级软件工程基准测试显著推动了编码智能体评估的发展，但现有基准主要衡量生成的补丁是否通过功能测试，而忽略了源自代码评审的验收约束（评审约束），而这些约束往往决定一个补丁在真实软件开发中是否可被接受。我们提出 SWE-Gate，一个面向软件工程智能体的仓库级基准，它在功能正确性之外，明确评估补丁对评审约束的遵守程度。SWE-Gate 从真实的拉取请求（PR）评审评论中提取评审约束，并围绕这些约束合成仓库级修复实例。每个实例都提供相互独立的功能测试和约束测试，并附带不合规补丁与标准（gold）补丁，从而能够将“问题解决能力”与“评审约束遵守能力”明确区分开来。我们构建了包含 303 个仓库级修复实例的 SWE-Gate……

    arXiv:2609.04167v1 Announce Type: cross  Abstract: Repository-level software engineering benchmarks have significantly advanced the evaluation of coding agents, but existing benchmarks primarily measure whether generated patches pass functional tests and overlook review-derived acceptance constraints (review constraints) that often influence whether a patch is acceptable in real-world software development. We introduce SWE-Gate, a repository-level benchmark for software engineering agents that explicitly evaluates review constraint compliance alongside functional correctness. SWE-Gate derives review constraints from real pull request review comments and synthesizes repository-level repair instances around these constraints. Each instance provides separate functional and constraint tests, together with non-compliant and gold patches, enabling explicit separation between issue resolution capability and review constraint compliance. We construct SWE-Gate with 303 repository-level repair i
    
[^11]: 《从欺骗性输出到欺骗性机制：语言模型欺骗研究的因果框架》

    From Deceptive Outputs to Deceptive Mechanisms: A Causal Framework for Language-Model Deception Research

    [https://arxiv.org/abs/2609.04166](https://arxiv.org/abs/2609.04166)

    本文提出一个因果分类框架，将语言模型“看似欺骗的行为”与“真实的欺骗机制”区分开来，并通过猜谜游戏和股票交易实验揭示欺骗性表象可以在缺乏相应机制的情况下出现。

    

    关于语言模型欺骗的研究和新闻报道越来越多地将人类式的心理状态概念归于语言模型。这类说法可能会模糊“看起来具有欺骗性的行为”与“实际具有欺骗性的机制”之间的区别。我们引入了一个因果分类框架，区分了以下几组概念：先验承诺与回顾性报告、模型偏好与实际输出、虚假偏好与对误导接收者效用的敏感性，以及欺骗性行为与产生该行为的目标或策略的来源。我们在两个开源权重模型家族中检验了这些区分。通过受控的猜谜游戏和股票交易实验，我们发现看似欺骗性的行为可以在没有相应假设机制的情况下出现，而其他干预实验则直接证明了接收者的信息状态能够因果性地影响模型的欺骗偏好。这些结果表明，欺骗性行为可以为……（原文摘要至此截断）

    arXiv:2609.04166v1 Announce Type: new  Abstract: Research and news coverage of language-model deception increasingly attributes human-like mental-state concepts to language models. Such claims can blur the distinction between behavior that looks deceptive and a mechanism that is actually deceptive.   We introduce a causal taxonomy separating prior commitment from retrospective report, model preference from realized output, false preference from sensitivity to the utility of misleading a recipient, and deceptive behavior from the provenance of the objective or strategy producing it. We test these distinctions in two open-weight model families. Across controlled guessing-game and stock-trading experiments, we find that deceptive-looking behavior can arise without the corresponding proposed mechanism, while other interventions provide direct evidence that recipient information state can causally affect deceptive preference.   These results show that deceptive behavior can provide evidence
    
[^12]: SENTINEL-RL：在安全运营中心中将拓扑推理从LLM智能体中卸载

    SENTINEL-RL: Offloading Topological Reasoning from LLM Agents in the Security Operations Center

    [https://arxiv.org/abs/2609.04159](https://arxiv.org/abs/2609.04159)

    提出Sentinel-RL架构，用异构图注意力编码器和PPO策略承担网络拓扑推理与遏制决策，LLM智能体仅负责在评论器把关下生成分析师可读的叙述，从而解决LLM在安全运营中心中上下文窗口有限和拓扑一致性无保证的两大局限。

    

    大语言模型（LLM）智能体越来越多地被提议作为自主的安全运营中心（SOC）分析师，但两个局限性使其在企业规模上不可靠：有限的上下文窗口无法容纳包含数千台主机的认证图，且自由形式的文本生成无法保证所推荐的遏制动作与其所依据的拓扑结构保持一致。我们提出了Sentinel-RL，一种将拓扑推理与语义推理解耦的智能体化SOC架构：异构图注意力编码器将实时认证子图总结为固定维度的状态，近端策略优化（PPO）策略将该状态映射到受限的调查动作集合，而LLM智能体循环则仅限于消费该策略的推荐，并在评论器（critic）的把关下生成分析师可读的叙述。我们在LANL综合多源网络安全事件数据集和Indiana（数据集）上对该系统进行了实例化。

    arXiv:2609.04159v1 Announce Type: cross  Abstract: Large language model (LLM) agents are increasingly proposed as autonomous SOC analysts, but two limitations make them unreliable at enterprise scale: a finite context window cannot hold a multi-thousand-host authentication graph, and free-form generation offers no guarantee that a recommended containment action is consistent with the topology it operates on. We present Sentinel-RL, an agentic-SOC architecture that decouples topological reasoning from semantic reasoning: a heterogeneous graph attention encoder summarizes the live authentication subgraph into a fixed-dimensional state, a Proximal Policy Optimization (PPO) policy maps this state to a constrained set of investigative actions, and an LLM agent loop is restricted to consuming the policy's recommendations and producing analyst-readable narratives gated by a critic. We instantiate the system on the LANL Comprehensive, Multi-Source Cyber-Security Events dataset and the Indiana 
    
[^13]: Terminal-Universe：将智能体轨迹转化为可扩展的终端环境

    Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

    [https://arxiv.org/abs/2609.04148](https://arxiv.org/abs/2609.04148)

    Terminal-Universe 通过重放智能体轨迹中记录的文件操作，直接从已有轨迹重建可执行的终端环境，将海量积累的轨迹转化为可复用、可扩展的环境，用于合成新任务并提供执行反馈，解决了智能体后训练中环境稀缺的问题。

    

    随着基于终端的代码智能体日益普及，智能体轨迹已大规模积累，而真实、可执行的环境仍然稀缺。然而，环境才是智能体后训练真正所需的：每个环境可以被反复查询以生成大量可验证的任务，并提供执行反馈，而轨迹只是单一的冻结演示。与其从头生成环境，我们观察到现有轨迹中的工具执行历史揭示了其运行环境的结构与内容，使得从轨迹本身重建这些环境成为可能。因此，我们提出了 Terminal-Universe，这是一个将每条轨迹转化为可复用环境的框架，并对环境进行探索以合成新任务和延续交互。具体而言，Terminal-Universe 通过重放轨迹中记录的文件操作，恢复智能体修改之前的每个文件状态……（原文摘要此处截断）

    arXiv:2609.04148v1 Announce Type: new  Abstract: As terminal-based code agents become prevalent, agent trajectories have accumulated at scale, while realistic, executable environments remain scarce. However, environments are what agent post-training actually requires: each can be re-queried into many verifiable tasks and provides execution feedback, whereas a trajectory is a single frozen demonstration. Rather than generating environments from scratch, we observe that the tool-execution history in existing trajectories exposes the structure and contents of the environments in which they ran, making it possible to reconstruct those environments from the trajectories themselves. Thus, we introduce Terminal-Universe, a framework which turns each trajectory into a reusable environment and explores it for synthesizing new tasks and continued interactions. Specifically, Terminal-Universe replays the file operations recorded in a trajectory to restore each file before the agent modified it, y
    
[^14]: 一种面向微型阿克曼车辆端到端自动驾驶的低成本开放平台

    A Low-Cost, Open Platform for End-to-End Autonomous Driving on a Miniature Ackermann Vehicle

    [https://arxiv.org/abs/2609.04147](https://arxiv.org/abs/2609.04147)

    本文提出一个集成实体车辆、打印城市赛道与Webots数字孪生的低成本开放平台，通过命令条件化行为克隆实现了微型阿克曼车辆的端到端自动驾驶，其横向误差（6.1厘米）接近人类演示水平，弥合了仿真与真实执行之间的鸿沟。

    

    本文提出了一个低成本、开放的实验平台，用于基于微型阿克曼车辆的端到端自动驾驶研究。该平台集成了实体车辆、打印的城市赛道、数据采集工具、轨迹配准以及Webots数字孪生，能够开展可控实验，将基于仿真的自动驾驶方法与真实世界执行相连接。作为首个基线，我们实现了命令条件化的行为克隆方法，其中神经策略接收车载摄像头图像和高级导航命令作为输入，输出转向和速度控制。该系统在实体车辆和仿真环境中均进行了评估。在真实闭环实验中，学习到的策略能够跟随车道并执行指定的转弯动作，相对参考路线的平均横向误差为6.1厘米，接近人类演示中所观察到的4.7厘米水平。在数字孪生实验中，摄像头视场角具有强烈的（影响）（注：摘要在此处被截断）

    arXiv:2609.04147v1 Announce Type: cross  Abstract: This paper presents a low-cost, open experimental platform for research in end-to-end autonomous driving with miniature Ackermann vehicles. The platform combines a physical vehicle, a printed urban track, data collection tools, trajectory registration, and a Webots digital twin, enabling controlled experiments that connect simulation-based autonomous-driving methods to real-world execution. As a first baseline, we implement command-conditioned behavior cloning, in which a neural policy receives an on-board camera image and a high-level navigation command and outputs steering and speed. The system is evaluated both on the physical vehicle and in simulation. In real closed-loop experiments, the learned policy follows lanes and executes commanded turns, reaching a mean cross-track error of 6.1 cm with respect to the reference route, close to the 4.7 cm observed in human demonstrations. In the digital twin, camera field of view has a stron
    
[^15]: 通过人机交互实现高效的测试时自适应

    Efficient Test-Time Adaptation through Human-AI Interaction

    [https://arxiv.org/abs/2609.04141](https://arxiv.org/abs/2609.04141)

    提出TAHI框架，利用跨会话的人机交互数据并结合不断演化的评分准则模块，在测试时对智能体的上下文和权重进行自适应，从而弥合AI生成成果与个人专业水准之间的差距。

    

    AI智能体在人口规模的数据上进行训练，以编码涵盖众多从业者能力的广泛能力。然而，它们生成的成果很少能达到专业人士愿意以其声誉担保的个人标准。在成功标准异质且记录不充分的现实开放任务中，个人专长恰恰体现在对平均水平的高度提升与偏离之中。在实践中，迭代式的人机交互会浮现出用户无法事先完全明确、却在多个任务中反复应用的评估标准。我们认为，这种跨会话的交互数据是弥合智能体与个人专长之间差距的丰富而未被充分利用的信号。在这项工作中，我们提出通过人机交互进行测试时自适应（TAHI），该方法将这些信号整合到智能体的上下文和权重中，并通过一个不断演化的评分准则（rubric）模块来逐步明确每位用户的训练与评估标准。我们将智能体适配到30位个体……

    arXiv:2609.04141v1 Announce Type: new  Abstract: AI agents are trained on population-scale data to encode broad capabilities spanning those of many practitioners. Yet the artifacts they produce rarely meet the personal bar professionals need to stake their reputation on. On realistic, open-ended tasks where success criteria are heterogeneous and insufficiently documented, individual expertise lives precisely in the elevation and departure from the average. In practice, iterative human-agent interaction surfaces criteria that users cannot fully specify up front, yet apply repeatedly across tasks. We argue this cross-session interaction data is a rich, underused signal for closing the gap to individual expertise. In this work, we propose test-time adaptation through human-agent interaction (TAHI), which integrates these signals into agent context and weights, and crystallizes each user's training and evaluation criteria via an evolving rubric module. We adapt agents to 30 individuals in 
    
[^16]: 面向AI智能体的自然语言交互协议与标准

    The Natural Language Interaction Protocol and Standard for AI Agents

    [https://arxiv.org/abs/2609.04135](https://arxiv.org/abs/2609.04135)

    该论文提出了由Ecma国际标准化的自然语言交互协议（NLIP），这是一种基于标准的应用层协议，通过轻量级语义消息信封使异构框架下开发的AI智能体能够实现互操作。

    

    AI智能体正日益在各组织中使用异构的智能体开发框架、AI模型、工具接口、协议和执行环境进行开发和部署。为了实现其潜在的社会和商业影响，这些智能体必须能够通过一个通用的通信协议实现互操作。自然语言交互协议（NLIP）由来自各公司和大学的研究人员与从业者共同开发，并由Ecma国际组织标准化，它通过定义一个基于标准的应用层协议来满足这一需求，用于AI智能体之间的交互。NLIP提供了一个轻量级的语义消息信封，可以承载于现有的传输层（如HTTP/HTTPS、WebSocket和AMQP）之上，同时允许支持NLIP的智能体和网关在客户端、智能体、本地上下文存储、本体、工具、企业服务以及异构底层协议之间进行适配。本文介绍了……

    arXiv:2609.04135v1 Announce Type: new  Abstract: AI agents are increasingly being developed and deployed across organizations using heterogeneous agent-development frameworks, AI models, tool interfaces, protocols, and execution environments. To realize their potential social and business impact, these agents must be able to interoperate through a common communication protocol. The Natural Language Interaction Protocol (NLIP), developed by researchers and practitioners across companies and universities and standardized by Ecma International, addresses this need by defining a standards-based application-layer protocol for AI-agent interaction. NLIP provides a lightweight semantic message envelope that can be carried over existing transports such as HTTP/HTTPS, WebSocket, and AMQP, while allowing NLIP-aware agents and gateways to adapt between clients, agents, local context stores, ontologies, tools, enterprise services, and heterogeneous underlying protocols. This paper presents the mot
    
[^17]: 面向终端智能体的环境演化

    Environment Evolution for Terminal Agents

    [https://arxiv.org/abs/2609.04128](https://arxiv.org/abs/2609.04128)

    提出环境演化方法，以离策略方式逐步提升环境难度并逐代调度演化环境进行训练，为终端智能体提供持续的学习信号。

    

    扩展可交互、可验证的环境对于训练终端智能体至关重要。随着前沿模型能力的不断增强，从头合成的环境变得不再具有挑战性，因而只能提供有限的学习信号。近期的共同演化方法根据策略rollout过程中暴露出的弱点，在模型可学习边界附近迭代合成环境。然而，这类方法依赖于在策略rollout，这既限制了泛化能力，也使得随着模型变强难以持续提供学习信号。在本文中，我们提出环境演化方法，该方法以离策略方式逐步增加环境难度，并在训练过程中逐代调度演化后的环境，以提供持续的学习信号。我们从多轮学习目标中推导出影响环境难度的三个演化方向，并通过循环引擎沿这些方向实现环境演化。

    arXiv:2609.04128v1 Announce Type: new  Abstract: Scaling interactive and verifiable environments is critical for training terminal agents. As frontier models become more capable, environments synthesized from scratch become less challenging and thus provide limited learning signals. Recent co-evolution methods iteratively synthesize environments near the model's learnable frontier based on weaknesses exposed during rollouts. However, their dependence on on-policy rollouts limits generalization and the continuous provision of learning signals as the model becomes stronger. In this paper, we propose environment evolution, which incrementally increases environment difficulty off-policy and schedules the evolved environments generation by generation during training to provide continuous learning signals. We derive three evolution directions that influence environment difficulty from the multi-turn learning objective and then implement evolution along these directions through a loop-enginee
    
[^18]: 大语言模型推荐的认识论依据：在缺乏真实标准时刻画依赖的基础

    Epistemic Warrant for LLM Recommendations: Characterizing the Basis for Reliance When Ground Truth Is Unavailable

    [https://arxiv.org/abs/2609.04127](https://arxiv.org/abs/2609.04127)

    本文借鉴认识论提出“认识论依据”这一决策层面新构念，通过四级依赖证书框架区分不同稳定性与适用范围的模型推荐，为用户在缺乏真实标准时有原则地判断是否依赖大语言模型的具体推荐提供了理论基础与操作化方法。

    

    大语言模型越来越多地被用于支持组织决策，然而用户在评估是否应该依赖某个具体推荐时，往往缺乏有原则的依据。现有方法通常评估模型的宽泛属性，如可靠性、不确定性或鲁棒性，或者聚焦于用户信任，而不是依赖单个推荐的底层基础。通过借鉴认识论的理论基础，我们引入了“认识论依据”这一决策层面的构念，用于刻画模型偏好的稳定性以及该偏好成立的范围。我们通过对成对推荐的四级依赖证书来操作化这一构念，区分出不稳定、上下文依赖、局部支持和广泛支持四类推荐。我们使用当代方法论验证了这一构念：已知组测试成功恢复了专家预先指定的依据。

    arXiv:2609.04127v1 Announce Type: new  Abstract: Large language models are increasingly used to support organizational decisions, yet users often lack a principled basis for assessing whether to rely on a specific recommendation. Existing approaches typically evaluate broad model properties, such as reliability, uncertainty, or robustness, or focus on user trust, rather than the underlying basis for relying on an individual recommendation. Adapting theoretical foundations from epistemology, we introduce epistemic warrant, a decision-level construct that characterizes the stability of a model's preference and the scope over which that preference holds. We operationalize this construct through a four-tier reliance certificate for pairwise recommendations, distinguishing among unstable, context-dependent, locally supported, and broadly supported recommendations. We validate the construct using contemporary methodologies: known-groups tests successfully recover expert-prespecified warrant 
    
[^19]: 顺序优于联合：论在线策略蒸馏与RLVR的相互作用

    Sequential Beats Joint: On the Interplay between On-Policy Distillation and RLVR

    [https://arxiv.org/abs/2609.04108](https://arxiv.org/abs/2609.04108)

    先蒸馏后强化学习的两阶段训练方案在推理任务上持续优于纯OPD、纯RLVR及所有联合优化方法，因为OPD先扩大学生对教师解的覆盖范围、RL再在其内锐化，而联合训练会导致两种信号相互干扰。

    

    可验证奖励强化学习（RLVR）和在线策略蒸馏（OPD）已成为对推理大语言模型进行后训练的两种主流方法。先前的工作利用OPD的密集token级监督来补充稀疏的RL奖励，在单个步骤内融合这两种信号：要么作为加权加性组合，要么作为对RL优势的教师调制重缩放。在本文中，我们展示了一个简单的两阶段方案——先OPD后RL——在逻辑和数学推理基准上持续优于纯OPD、纯RLVR以及所有此类联合基线方法。除了实证结果外，我们还通过pass@$k$行为、学习动态和参数更新对这一现象提供了系统性的理解，并得出一个一致的解释：OPD扩大了学生对教师支持解的覆盖范围，而RL则在该支持范围内进行锐化，同时联合优化这两种信号会导致它们相互干扰。

    arXiv:2609.04108v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) and on-policy distillation (OPD) have emerged as two dominant methods for post-training reasoning LLMs. Prior work uses OPD's dense token-level supervision to complement the sparse RL reward, fusing the two signals within a single step: either as a \emph{weighted-additive combination} or a \emph{teacher-modulated rescaling} of the RL advantage. In this paper, we show that a simple two-stage scheme, OPD-then-RL, consistently outperforms pure OPD, pure RLVR, and all such joint baselines across logic and math reasoning benchmarks. Beyond the empirical results, we further provide a systematic understanding of this through pass@$k$ behavior, learning dynamics, and parameter updates, yielding a consistent explanation: OPD expands the student's coverage of teacher-supported solutions and RL sharpens within that support, while jointly optimizing the two signals causes them to interfere.To p
    
[^20]: 为什么门控 DeltaNet 能在 4 比特量化中幸存：面向混合架构 27B 大语言模型循环部分的 NVFP4 W4A4 量化

    Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM

    [https://arxiv.org/abs/2609.04098](https://arxiv.org/abs/2609.04098)

    该论文证明混合架构大语言模型中的门控 DeltaNet 循环层完全可以进行 NVFP4 W4A4 4 比特量化，其性能与 BF16 相当且模型更小、预填充更快，并通过机制研究（如块缩放局部化离群值）解释了循环误差累积的担忧并不成立。

    

    混合架构大语言模型将 softmax 注意力与线性注意力层（如门控 DeltaNet，GDN）相结合，后者的循环状态以固定大小总结上下文内容。早期的社区 4 比特量化版本将 Qwen3.8-27B（48 个 GDN 层、16 个注意力层）中的 GDN 模块保留在 8 比特或 16 比特精度——尤其是其衰减门和写入强度门——其依据的直觉是循环中的误差会在长上下文中不断累积。我们通过构建 Minima 来检验这一直觉：在全部 496 个线性层（包括 GDN）上应用 NVFP4 W4A4 量化。在 4K/32K 困惑度、MMLU-Pro、GSM8K、AIME'25、GPQA-Diamond、LiveCodeBench 以及最高 64K 的 RULER 检索等评测中，Minima 与 BF16 的差异在种子噪声范围内（5 项任务平均 -0.52），同时是所比较方案中体积最小（17.5 GiB）、预填充速度最快（+14-19%）的方案，且其 32K 困惑度差距随位置增加而缩小。一项包含四个部分的机制研究解释了其中原因：(i) NVFP4 的 16 元素块缩放将残差流中的极端离群值局部化，……（摘要在此处截断）

    arXiv:2609.04098v1 Announce Type: new  Abstract: Hybrid LLMs pair softmax attention with linear-attention layers such as Gated DeltaNet (GDN), whose recurrent state summarizes the context in fixed size. Early community 4-bit quantizations of Qwen3.8-27B (48 GDN layers, 16 attention layers) left the GDN block in 8- or 16-bit precision -- especially its decay and write-strength gates -- on the intuition that errors in a recurrence accumulate over long contexts. We test that intuition by building Minima: NVFP4 W4A4 on all 496 linear layers, GDN included. Across perplexity at 4K/32K, MMLU-Pro, GSM8K, AIME'25, GPQA-Diamond, LiveCodeBench, and RULER retrieval to 64K, Minima matches BF16 within seed noise (5-task average -0.52) while being the smallest (17.5 GiB) and fastest-prefill (+14-19%) recipe we compare, and its 32K perplexity gap shrinks with position. A four-part mechanism study explains why: (i) NVFP4's 16-element block scaling localizes the residual stream's extreme outliers, equal
    
[^21]: 基于可组合基础模型先验与可泛化抓取合成的自适应视觉-语言抓取

    Adaptive Vision-Language Grasping via Composable Foundation Priors and Generalizable Grasp Synthesis

    [https://arxiv.org/abs/2609.04096](https://arxiv.org/abs/2609.04096)

    该论文提出AdaRoboVLG框架，通过将可组合的基础模型先验与基于力封闭评估的可泛化抓取合成策略相解耦，实现了无需重新训练即可跨不同机器人手部进行上下文自适应的视觉-语言抓取。

    

    本文提出了AdaRoboVLG，一个任务自适应的视觉-语言-抓取（VLG）框架，支持跨不同机器人手部的可泛化抓取合成。与现有将基础模型与端到端抓取策略紧密耦合的VLG方法不同，AdaRoboVLG学习一个高效且可泛化的基础策略，该策略通过显式运动学映射和基于力封闭的稳定性评估来生成并评估物理可行的抓取候选，同时将任务相关的理解工作交由专门的基础模型模块完成。这些模块提供可组合的先验知识，并将其集成到抓取合成过程中，从而无需重新训练底层抓取策略即可实现上下文自适应的抓取合成。通过大量的仿真和真实世界实验，我们证明了：（i）基础策略表现出高效的学习能力和强大的跨手部泛化能力；（ii）该框架能够有效地融合……（原文摘要在此处截断）

    arXiv:2609.04096v1 Announce Type: cross  Abstract: This paper proposes AdaRoboVLG, a task-adaptive Vision-Language-Grasp (VLG) framework that supports generalizable grasp synthesis across different robotic hands. Unlike existing VLG methods that tightly couple foundation models with end-to-end grasp policies, AdaRoboVLG learns an efficient generalizable base policy that generates and evaluates physically feasible grasp candidates through explicit kinematic mapping and force-closure-based stability estimation, while offloading task-dependent understanding to specialized foundation-model modules. These modules provide composable priors that are integrated into the grasp synthesis process, enabling contextually adaptive grasp synthesis without retraining the underlying grasp policy. Through extensive simulation and real-world experiments, we demonstrate that (i) the base policy exhibits efficient learning and strong cross-hand generalization, (ii) the framework effectively incorporates sp
    
[^22]: DRACO：基于动态评分准则的长程智能体训练细粒度信用分配方法

    DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training

    [https://arxiv.org/abs/2609.04094](https://arxiv.org/abs/2609.04094)

    DRACO通过在训练中动态生成评分准则，并以闭式解方式将轨迹级评判重新分配到具体步骤，解决了无真实成功信号时长程智能体训练的细粒度信用分配问题，在AppWorld上显著超越基础模型和稀疏奖励GRPO。

    

    当任务具备程序化检查器时，基于可验证奖励的强化学习效果良好，但大多数长程智能体领域并不存在这样的检查器。我们在“结果盲设”下开展工作，即真实成功信号不可用的场景。多准则评分准则是提供此类奖励的常用方式；它们对每个轨迹仅评分一次，但单一标量在数十个步骤中是较弱的信号。我们提出DRACO：基于评分准则分布的优势分配信用优化方法。它在训练过程中动态生成评分准则以跟踪策略不断演进的能力，对每个完成的轨迹对这些准则评分一次，并将该评判重新分配到负责相关准则标注的步骤上，从而在GRPO中产生差异化的逐步优势。这种重新分配是闭式解形式，不引入任何需要训练的归因模块。在AppWorld上，DRACO比基础模型提升15.9分，比使用稀疏奖励训练的GRPO提升5.3分。

    arXiv:2609.04094v1 Announce Type: new  Abstract: Reinforcement Learning from Verifiable Rewards works well when a task has a programmatic checker, but most long-horizon agent domains have none. We work in the outcome-blind setting, where ground-truth success signals are not available. Multi-criteria rubrics are a popular way to supply such a reward; they are scored once per trajectory, but a single scalar is a poor signal across tens of steps. We propose DRACO: Distributing Rubric-based Advantage for Credit Optimization. It generates rubrics dynamically during training to track the policy's evolving capability, scores those rubrics once per completed trajectory, and redistributes that judgment over the steps responsible for annotated rubrics to produce differentiated per-step advantages in GRPO. The redistribution is closed-form and does not introduce any trained attribution module. On AppWorld, DRACO gains 15.9 points over the base model and 5.3 points over GRPO trained with a sparse 
    
[^23]: 不可形式化的定理：有限句法系统的根本极限及其对安全与人工智能的影响

    A Non-Formulable Theorem: A Fundamental Limit of Finite Syntactic Systems and Its Consequences for Security and AI

    [https://arxiv.org/abs/2609.04086](https://arxiv.org/abs/2609.04086)

    本文证明了一个元定理：任何连贯且充分表达的有限句法系统都至少存在一条它无法自主产出的定理，这一根本性极限普遍适用于安全机制、AI系统、形式化验证器等各类有限句法系统。

    

    对于每一个连贯且具有充分表达能力的有限句法系统S，我们证明了至少存在一条S无法自主产出的定理。该结果是一个元定理：它证明了某条定理的存在性，并适用于每一个有限句法系统——包括安全机制、人工智能系统、形式化验证器、法律系统、经济模型，以及证明该定理本身的形式系统。

    arXiv:2609.04086v1 Announce Type: cross  Abstract: For every coherent and sufficiently expressive finite syntactic system S, we prove the existence of at least one theorem that S cannot produce autonomously. The result is a metatheorem: it proves the existence of a theorem, and applies to every finite syntactic system - security mechanisms, AI systems, formal verifiers, legal systems, economic models, and the formal system in which it is itself proved.
    
[^24]: CORE：通过重排序器蒸馏改进MLLM嵌入中的组合推理

    CORE: Improving Compositional Reasoning in MLLM Embedding via Reranker Distillation

    [https://arxiv.org/abs/2609.04083](https://arxiv.org/abs/2609.04083)

    CORE通过将交叉注意力重排序器的细粒度组合判断以列表式Rank-KL目标蒸馏到嵌入模型中，显著提升了MLLM嵌入模型的组合推理能力，其效果优于对比学习和CoSENT。

    

    基于MLLM的嵌入模型在组合检索方面仍然存在局限，常常无法区分包含相同概念但属性-对象绑定不同的场景。然而，当同一骨干网络被用作交叉注意力重排序器时，却能够解决这类区分问题，这促使我们将其组合判断蒸馏到嵌入模型中。我们提出了CORE，该方法合成了跨越五个组合匹配级别的候选列表，并引入Rank-KL目标函数，训练嵌入模型复现重排序器的细粒度排序。我们进一步提出了一种分级评估协议，并在相同的数据和调参预算下比较了对比学习、成对式CoSENT和列表式Rank-KL。比较结果表明，CoSENT和Rank-KL都比对比学习更有效地利用了多级别监督，其中Rank-KL取得了最强的整体性能。在三个组合推理基准上……（摘要截断）

    arXiv:2609.04083v1 Announce Type: cross  Abstract: MLLM-based embedding models remain limited in compositional retrieval, often failing to distinguish scenes containing the same concepts but different attribute-object bindings. Yet the same backbone can resolve such distinctions when used as a cross-attentive reranker, motivating us to distill its compositional judgments into the embedding model. We propose CORE, which synthesizes candidate lists spanning five compositional matching levels and introduces a Rank-KL objective that trains the embedding model to reproduce the reranker's fine-grained ranking. We further introduce a graded evaluation protocol and compare contrastive learning, pairwise CoSENT, and listwise Rank-KL under the same data and tuning budget. Our comparison shows that both CoSENT and Rank-KL use the multi-level supervision more effectively than contrastive learning, with Rank-KL achieving the strongest overall performance. Across three compositional reasoning benchm
    
[^25]: PatchBench：评估AI智能体的漏洞修复能力

    PatchBench: Evaluating AI Agents for Vulnerability Patching

    [https://arxiv.org/abs/2609.04075](https://arxiv.org/abs/2609.04075)

    该研究提出补丁相似度度量方法，发现25%的AI智能体漏洞修复补丁存在记忆历史开发者补丁或仅通过修补崩溃堆栈来抑制崩溃而非修复根本原因的问题，揭示了现有漏洞修复评估方法面临的有效性威胁。

    

    AI智能体最近在自动化漏洞修复方面展现出了强大的性能。然而，现有的评估通常仅通过测试所提供的概念验证（PoC）输入是否仍会触发崩溃来验证补丁的有效性。这给评估的有效性留下了两个关键威胁：智能体可能会复现已记忆的历史开发者补丁，或者它们可能生成仅能抑制所报告崩溃的表面级修复。我们针对C/C++漏洞修复问题研究了这些担忧。我们引入了一种补丁相似度度量方法来检测记忆化的补丁。平均而言，25%的智能体补丁与历史开发者补丁表现出高度相似性，这表明补丁记忆化是漏洞修复评估有效性面临的一个真实威胁。与此同时，智能体还经常利用基准测试的结构，通过在崩溃堆栈跟踪上进行修补来抑制崩溃，从而通过补丁验证，而不是定位并修复根本原因。

    arXiv:2609.04075v1 Announce Type: cross  Abstract: AI agents have recently demonstrated strong performance in automated vulnerability patching. However, existing evaluations often validate a patch only by testing whether the provided Proof-of-Concept (PoC) input still triggers a crash. This leaves two key threats to validity: agents may reproduce memorized historical developer patches, or they may generate surface-level fixes that only suppress the reported crash.   We study these concerns for C/C++ vulnerability patching. We introduce a patch similarity metric to detect memorized patches. On average, 25% of the agent patches exhibit substantial similarity to historical developer patches, indicating that patch memorization is a real threat to the validity of vulnerability patching evaluations. Meanwhile, agents also frequently exploit benchmark structures to pass patch validation by patching on the crash stack trace to suppress the crash, rather than localizing and fixing the root caus
    
[^26]: TAP-Path：面向高效且可信病理学基础模型的任务自适应结构与Token剪枝

    TAP-Path: Task-Adaptive Structural and Token Pruning for Efficient and Trustworthy Pathology Foundation Models

    [https://arxiv.org/abs/2609.04071](https://arxiv.org/abs/2609.04071)

    TAP-Path提出了一种任务自适应压缩框架，通过验证驱动的Transformer块选择与物理移除、输入自适应的patch token剪枝以及多深度特征恢复，直接重构预训练的Virchow2病理学编码器，在保持精度的同时将编码器参数减少约25%、计算量减少约35%。

    

    病理学基础模型提升了组织病理学中的可迁移表示学习能力，但近期的进展往往依赖于参数量达数亿的编码器和高昂的推理成本。我们提出了TAP-Path，这是一个任务自适应的压缩框架，它直接对预训练的Virchow2编码器进行结构重构，而不是将其蒸馏到一个单独的学生模型中。TAP-Path结合了基于验证集驱动的Transformer块选择、冗余块的物理移除、输入自适应的patch token剪枝、多深度特征恢复以及一个轻量级的门控任务头。最终模型在剪枝后保留了32个Transformer块中的24个以及70%的patch token，使编码器参数量减少了24.96%（从631.24M降至473.70M），并使编码器的分析计算量减少了35.20%（从340.13G降至220.40G FLOPs）。在三个任务头优化随机种子下，TAP-Path取得了87.98 ± 0.067%的测试准确率、81.26 ± 0.49%的平衡准确率和82.38 ± 0.48%的宏平均F1分数

    arXiv:2609.04071v1 Announce Type: cross  Abstract: Pathology foundation models improve transferable representation learning for histopathology, but recent gains often rely on encoders with hundreds of millions of parameters and high inference cost. We propose TAP-Path, a task-adaptive compression framework that directly restructures a pretrained Virchow2 encoder rather than distilling it into a separate student. TAP-Path combines validation-driven transformer-block selection, physical removal of redundant blocks, input-adaptive patch-token pruning, multi-depth feature recovery, and a lightweight gated task head. The final model retains 24 of 32 transformer blocks and 70% of patch tokens after pruning, reducing encoder parameters by 24.96% (631.24M to 473.70M) and analytical encoder compute by 35.20% (340.13G to 220.40G FLOPs). Across three task-head optimization seeds, TAP-Path achieved $87.98 \pm 0.067%$ test accuracy, $81.26 \pm 0.49%$ balanced accuracy, and $82.38 \pm 0.48%$ macro-F
    
[^27]: 子空间推断实现高效的基于偏好的主动奖励学习

    Subspace Inference Enables Efficient Active Reward Learning from Preferences

    [https://arxiv.org/abs/2609.04066](https://arxiv.org/abs/2609.04066)

    本文提出PreferenceEKF方法，将主动偏好学习框架化为序贯贝叶斯滤波问题，通过扩展卡尔曼滤波器在低维参数子空间中高效跟踪奖励模型的不确定性，实现样本高效的RLHF奖励学习。

    

    基于人类反馈的强化学习（RLHF）已成为一种从人类偏好中学习奖励模型的强大但样本效率低下的方法，这使得主动学习成为构建信息丰富的偏好查询的关键组成部分。然而，主动学习所需的有效不确定性量化对于大型神经网络奖励模型而言仍然是一个关键挑战。在本文中，我们提出了PreferenceEKF，这是一种样本高效的方法，它将主动偏好学习框架化为序贯贝叶斯滤波问题，从而跟踪奖励模型的不确定性。我们的方法不依赖于在整个神经网络参数空间上进行计算代价极高的后验推断，而是在低维参数子空间内通过扩展卡尔曼滤波器进行序贯推断，随着新的偏好查询的到来持续更新奖励模型的后验分布。我们的方法实现了对神经网络的可扩展采样

    arXiv:2609.04066v1 Announce Type: cross  Abstract: Reinforcement learning from human feedback (RLHF) has emerged as a powerful yet sample-inefficient approach for learning reward models from human preferences, making active learning a critical component in synthesizing informative preference queries. However, effective uncertainty quantification required for active learning remains a key challenge for large neural network reward models. In this paper, we introduce PreferenceEKF, a sample-efficient approach that tracks reward model uncertainty by framing active preference learning as a sequential Bayesian filtering problem. Instead of relying on computationally prohibitive posterior inference over the full neural network parameter space, our method performs sequential inference via an extended Kalman filter within a low-dimensional parameter subspace, continuously updating the reward model posterior as new preference queries arrive. Our approach enables scalable sampling of neural netwo
    
[^28]: 隐藏在GRPO中的虚假优势

    Spurious Advantage Hidden in GRPO

    [https://arxiv.org/abs/2609.04063](https://arxiv.org/abs/2609.04063)

    论文揭示了GRPO优势估计中的一个“虚假优势”缺陷——通过猜测碰巧答对的采样也会获得高优势，从而诱导策略学出投机取巧行为，并提出SIGNBALANCE方法，通过保留验证符号、全局尺度和停止梯度的按类重缩放来消除该偏差。

    

    组相对策略优化（GRPO）被广泛研究用于可验证奖励的强化学习，其优势估计器会根据组内奖励统计量为每次采样分配一个幅值。在常见情况下，该幅值会奖励那些通过推理得出正确答案的采样。然而，一个被忽视的情况具有相同的表现：某次采样可能通过猜测碰巧得到正确答案，而公式依然会赋予其高幅值，我们将这一现象识别为“虚假优势”。该问题出现在三种场景中：候选集较小的有界答案任务；包含有界子情形的开放答案集合；以及预算允许开辟多条路径通向同一答案的搜索智能体。在这三种情况下，该问题都会误导策略趋向依赖猜测的行为。我们提出SIGNBALANCE，其幅值设计是无组合依赖的：它保留验证器的符号，采用全局尺度，并通过带停止梯度的按类重缩放来恢复零均值平衡。（在数学与搜索任务上的实验……原文摘要在此截断）

    arXiv:2609.04063v1 Announce Type: new  Abstract: Group Relative Policy Optimization (GRPO) is widely studied for reinforcement learning with verifiable rewards, where its advantage estimator assigns each rollout a magnitude from within-group reward statistics. In the common case, this magnitude rewards rollouts that reach the correct answer through reasoning. Yet, an overlooked case shares the same surface: a rollout may land on it by guessing, and the formula still assigns a high magnitude, which we identify as the spurious advantage. This arises in three cases: bounded-answer tasks with a small candidate set; open-answer sets hosting bounded sub-cases; and search agents whose budget opens many paths to the same answer. In all three, this misleads the policy toward guess-like behaviors. We propose SIGNBALANCE, whose magnitude is composition-free: it keeps the verifier sign, uses a global scale, and restores zero-mean balance via a stop-gradient per-class rescaling. Across math and sea
    
[^29]: 当模型编辑过多：论最小代码编辑的保真度

    When Models Edit Too Much: On the Fidelity of Minimal Code Edits

    [https://arxiv.org/abs/2609.04061](https://arxiv.org/abs/2609.04061)

    该研究揭示了前沿大语言模型在修复代码时普遍存在“过度编辑”问题（即使如GPT-5.5这样的强模型也不例外），并提出通过一条简单的保留指令即可显著减少不必要的代码改动、降低认知复杂度，同时提升修复准确率。

    

    大语言模型越来越多地被用于编辑现有代码，但仅仅正确是不够的：有用的修复还应当是最小化的、可审查的，并且忠实于原始实现。我们研究了“过度编辑”现象，即模型重写代码的范围超出修复缺陷所需的趋势。我们基于400个BigCodeBench问题构建了一个评估框架，通过向参考解答中注入受控的AST级（抽象语法树级）破坏，为每个修复任务提供一个已知的最小补丁。研究发现，在各类前沿大语言模型中，过度编辑现象普遍存在，即使是在GPT-5.5这样的强大模型中也是如此：高Pass@1可能与不必要的巨大编辑和新增认知复杂度并存。一条保留指令能够显著减少这种行为，将平均超额Levenshtein距离从0.195降至0.131，减少26.6%的新增认知复杂度，并使Pass@1提高2.3个百分点。然而，这些收益并非简单地源于更大的推理（摘要在此处被截断）

    arXiv:2609.04061v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to edit existing code, but correctness alone is not enough: useful repairs should also be minimal, reviewable, and faithful to the original implementation. We study over-editing, the tendency of a model to rewrite code beyond what is required to fix a bug. We construct an evaluation framework from 400 BigCodeBench problems by injecting controlled AST-level corruptions into reference solutions, giving each repair task a known minimal patch. Across frontier LLMs, over-editing is widespread even among strong models like GPT-5.5: high Pass@1 can coexist with unnecessarily large edits and added cognitive complexity. A preservation instruction substantially reduces this behavior, lowering average excess Levenshtein distance from 0.195 to 0.131, reducing added cognitive complexity by 26.6%, and increasing Pass@1 by 2.3 points. However, these gains do not simply follow from a larger reasoning 
    
[^30]: 作为决策空间的翻译：低资源方言生成的多智能体视角

    Translation as a Decision Space: A Multi-Agent Perspective on Low-Resource Dialect Generation

    [https://arxiv.org/abs/2609.04048](https://arxiv.org/abs/2609.04048)

    本文将翻译重构为由多个自主智能体探索的结构化决策空间，把不同翻译路径建模为智能体，并将智能体间的分歧作为可解释的行为信号，应用于土耳其语—叙利亚阿拉伯语这一低资源方言翻译的实证研究。

    

    神经机器翻译（NMT）系统通常对每个输入只产生单一输出，这掩盖了多语言解码中隐含存在的多种备选决策路径。这种不透明性在低资源方言环境中尤为成问题，因为在该环境中，多种在语言学上都成立的译文实现方式可能在词汇地道性、语域和结构稳定性上存在差异。我们提出将翻译重新构建为一个由自主翻译智能体探索的结构化决策空间。我们不分析单一输出，而是将不同的翻译路径建模为在共享多语言骨干上运行的智能体。智能体之间的分歧不再被视为错误，而是被视为一种可解释的行为信号。我们使用三个智能体对土耳其语—叙利亚阿拉伯语翻译进行了实证研究：（1）零样本直接翻译，（2）通过轻量级微调实现的方言稳定化翻译，以及（3）通过英语进行的枢纽语翻译。

    arXiv:2609.04048v1 Announce Type: cross  Abstract: Neural machine translation (NMT) systems typically produce a single output per input, obscuring the alternative decision trajectories implicitly available within multilingual decoding. This opacity becomes particularly problematic in low-resource dialect settings, where multiple linguistically valid realizations may differ in lexical authenticity, register, and structural stability. We propose reframing translation as a structured decision space explored by autonomous translation agents. Instead of analyzing a single output, we model distinct translation pathways as agents operating over a shared multilingual backbone. Inter-agent divergence is treated not as error but as an interpretable behavioral signal. We conduct an empirical study on Turkish--Syrian Arabic translation using three agents: (1) zero-shot direct translation, (2) dialect-stabilized translation via lightweight fine-tuning, and (3) pivot translation through English. Eva
    
[^31]: IRWOZ 2.0：一个由大语言模型驱动的工业机器人对话数据集

    IRWOZ 2.0: A Large Language Model-driven Dialogue Dataset for Industrial Robot Conversations

    [https://arxiv.org/abs/2609.04030](https://arxiv.org/abs/2609.04030)

    IRWOZ 2.0利用大语言模型增强生成和质量改进技术，构建了涵盖4个工业领域共390个对话的高质量工业机器人对话数据集，显著提升了对话状态跟踪性能（GPT-2的BLEU-4分数从0.1651提升至0.5604）。

    

    IRWOZ通过领域特定的标注改进了工业人机交互（HRI）对话系统。然而，其初始版本在对话状态和话语中包含大量噪声，限制了状态跟踪的准确性。我们推出了IRWOZ 2.0，通过大语言模型（LLM）增强生成（Mistral/Claude-3.5）和质量改进来解决这些局限。我们改进后的数据集扩展到涵盖4个工业领域（装配、配送、定位、搬迁）的390个对话，具有人工纠正和自动错别字消除的特点。在对话状态跟踪上的基准实验显示了显著的性能提升，与原始IRWOZ相比，GPT-2的BLEU-4分数从0.1651提升到0.5604。为支持工业人机交互研究，我们已公开发布IRWOZ 2.0数据集。

    arXiv:2609.04030v1 Announce Type: new  Abstract: IRWOZ has improved industrial human-robot interaction (HRI) dialogue systems through domain-specific annotations. However, its initial version contains substantial noise in dialogue states and utterances, limiting state-tracking accuracy. We introduce IRWOZ 2.0, which addresses these limitations through large language model (LLM) enhanced generation (Mistral/Claude-3.5) and quality refinements. Our improved dataset expands to 390 dialogues across 4 industrial domains (Assembly, Delivery, Position, Relocation), featuring manual corrections and automated typo removal. Benchmark experiments on dialogue state tracking demonstrate significant improvements, with GPT-2's BLEU-4 score increasing from 0.1651 to 0.5604 compared to original IRWOZ. To support industrial HRI research, we publicly released IRWOZ 2.0 dataset at https://ieee-dataport.org/documents/irwoz-20-large-language-model-driven-dialogue-dataset-industrial-robot-conversations
    
[^32]: 挤出长条形状对3D混凝土打印可建造性的影响：一种几何信息驱动的深度学习-有限元方法

    Influence of Extruded Filament Shape on Buildability in 3D Concrete Printing: A Geometry-Informed Deep Learning-FEM Approach

    [https://arxiv.org/abs/2609.04028](https://arxiv.org/abs/2609.04028)

    该研究提出了一个将深度学习长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合的几何信息驱动建模框架，能够直接从材料和工艺参数生成考虑真实长条几何形状的数值模型，从而更准确地评估3D混凝土打印结构的可建造性。

    

    沉积长条的几何形貌会显著影响3D混凝土打印（3DCP）结构的性能与稳定性。然而，大多数基于有限元（FEM）的可建造性评估方法将打印层简化为矩形，这可能限制了预测精度。本研究提出了一种几何信息驱动的建模框架，将基于深度学习的长条形状预测工具ShapeGen3DCP与层激活有限元方法相结合，以研究真实长条几何形状对可建造性的影响。该框架可直接从材料与工艺参数生成几何感知的数值模型，无需实验性的长条表征或计算量庞大的流体流动模拟。通过与实验数据的验证以及对直线墙体的参数化研究表明，挤出参数及由此产生的长条……

    arXiv:2609.04028v1 Announce Type: cross  Abstract: The geometric morphology of deposited filaments can significantly influence the structural performance and stability of 3D concrete-printed (3DCP) structures. However, most finite element (FEM)-based approaches for buildability assessment represent printed layers as simplified rectangles, potentially limiting predictive accuracy. This study proposes a geometry-informed modelling framework that integrates the deep-learning-based filament shape prediction tool ShapeGen3DCP with a layer-activation FEM approach to investigate the effect of realistic filament geometries on buildability. The framework generates geometry-aware numerical models directly from material and process parameters, eliminating the need for experimental filament characterization or computationally intensive fluid-flow simulations. Validation against experimental data and a parametric study of rectilinear walls demonstrate that extrusion parameters and the resulting fil
    
[^33]: 指令复制作为一种推理时控制原语

    Instruction Duplication as an Inference-Time Control Primitive

    [https://arxiv.org/abs/2609.04024](https://arxiv.org/abs/2609.04024)

    在推理时仅简单复制一遍程序化指令——无需重新训练或修改解码——即可将七个模型在医学选择题上通过全部八项诊断测试的比例从90.22%提升至93.17%，同时保持最终答案准确率不变。

    

    程序化指令遵循是可控语言模型系统的基本要求，尤其是当生成的轨迹需要在下游被检查或修复时。我们提出了“指令复制”，这是一种极简的黑盒推理时控制方法，仅重复程序化指令本身，无需重新训练或修改解码过程。在七个指令微调模型、300道医学选择题、八种放置条件以及16,800次计划生成的实验中，将指令从一份复制为两份，使确定性的All-8诊断——即响应通过全部八项可观测测试——从90.22%提升至93.17%（提高2.95个百分点），消除了单份复制后剩余失败案例的30.2%。预修正的TF-IDF召回率从73.44%上升至74.81%（提高1.38个百分点；经Holm校正后p < .001），而最终答案准确率恰好保持在60.21%不变。过早承诺现象从1.52%增加至2.30%（p_Holm = .00536）。一项盲测挑战……（原文在此处截断）

    arXiv:2609.04024v1 Announce Type: new  Abstract: Procedural instruction following is a basic requirement for controllable language-model systems, especially when generated trajectories are inspected or repaired downstream. We introduce instruction duplication, a minimal black-box inference-time control that repeats only the procedural instruction, without retraining or decoding changes. Across seven instruction-tuned models, 300 medical multiple-choice questions, eight placement conditions, and 16,800 scheduled generations, moving from one to two copies raises the deterministic All-8 diagnostic--responses passing all eight observable tests--from 90.22% to 93.17% (+2.95 percentage points), eliminating 30.2% of the failures remaining after one copy. Pre-provisional TF-IDF recall rises from 73.44% to 74.81% (+1.38 points; Holm-adjusted p < .001), while final-answer accuracy remains exactly 60.21%. Premature commitment increases from 1.52% to 2.30% (p_Holm = .00536). A blinded challenge au
    
[^34]: 表征对齐使语言模型获得可泛化的安全性

    Representational alignment yields generalizable safety in language models

    [https://arxiv.org/abs/2609.04022](https://arxiv.org/abs/2609.04022)

    提出表征相似性优化方法，将大语言模型的内部潜在表征与人类道德判断的原型化归类结构直接对齐，从而使安全对齐能够泛化到以陌生或对抗性形式表述的有害意图。

    

    对大语言模型（LLM）进行对齐对于其安全部署至关重要。当前的对齐方法主要优化可观察的响应，然而当同样的有害意图以人类能够轻易识别的不熟悉或对抗性形式重新表述时，模型仍然容易受到攻击。原型理论为人类的这种适应性提供了一种解释：人类概念是围绕中心案例来表征的，新实例则根据其相对于这些原型的分级典型性来进行归类。本研究表明，这种道德概念的归类能力在当前的大语言模型中仅有微弱的保留。在23个LLM上的实验显示，模型往往无法区分相互对立的道德类别，也无法在每个类别内部保持细粒度的典型性。这些缺陷在不同的参数规模和对齐阶段中持续存在。我们提出了表征相似性优化方法，将LLM中的潜在表征与人类道德判断中表达的归类方式直接对齐……

    arXiv:2609.04022v1 Announce Type: cross  Abstract: Aligning large language models (LLMs) is essential for their safe deployment. Current alignment methods mainly optimize observable responses, yet models remain vulnerable when the same harmful intent is recast in unfamiliar or adversarial forms that humans can easily recognize. Prototype theory offers an account of this adaptability. Human concepts are represented around central cases, and new instances are categorized according to their graded typicality relative to these prototypes. Here we show that such categorization of moral concepts is weakly preserved in current LLMs. Across 23 LLMs, models often failed to distinguish opposed moral categories or preserve fine-grained typicality within each category. These deficits persist across parameter sizes and alignment stages. We developed representational similarity optimization, which directly aligns the latent representations in LLMs with the categorization expressed in human moral jud
    
[^35]: FLY-EVAL++：一种用于大语言模型安全约束飞行预测的证据驱动评估协议

    FLY-EVAL++: An Evidence-Driven Evaluation Protocol for Safety-Constrained Flight Prediction with Large Language Models

    [https://arxiv.org/abs/2609.04021](https://arxiv.org/abs/2609.04021)

    本文提出FLY-EVAL++评估协议，通过对协议合规性、物理可行性和安全约束进行确定性验证并结合量规引导的聚合评分，对66个大语言模型的飞行轨迹与姿态预测能力进行评估，发现安全合规性是区分模型优劣最具判别力的维度。

    

    在安全关键且受物理规律支配的环境中评估大语言模型（LLM），不能仅仅依赖基于准确率的指标，因为数值上接近真实值的预测仍可能违反运行约束、以物理上不一致的方式组合场变量，或无法生成可用的结构化输出。现有的评估协议无法可靠地衡量这些失效模式。我们提出了FLY-EVAL++，一种证据驱动的评估协议，它将对协议合规性、物理可行性和安全约束的确定性验证与基于固定量规引导的聚合相结合，形成可解释的多维度评分。我们通过在PilotBench设置的基础上扩展历史条件化预测和多步预测任务，将FLY-EVAL++实例化应用于飞行轨迹与姿态预测（FTAP）任务。在66个大语言模型的评估中，安全合规性是区分模型行为最具判别力的维度：具有相当……（原文摘要在此处截断）

    arXiv:2609.04021v1 Announce Type: new  Abstract: Evaluating large language models (LLMs) in safety-critical, physics-governed environments requires more than accuracy-based metrics, because predictions that are numerically close to the ground truth can still violate operational constraints, combine fields in physically inconsistent ways, or fail to produce usable structured outputs. Existing evaluation protocols do not measure these failure modes reliably. We propose FLY-EVAL++, an evidence-driven evaluation protocol that combines deterministic verification of protocol compliance, physical feasibility, and safety constraints with fixed rubric-guided aggregation into interpretable multi-dimensional scores. We instantiate FLY-EVAL++ for Flight Trajectory and Attitude Prediction (FTAP) by extending the PilotBench setting with history-conditioned and multi-step prediction tasks. Across 66 LLMs, safety compliance is the most discriminative dimension of model behavior: models with comparable
    
[^36]: InSituMeasure：利用多模态大语言模型探究工业场景中的情境化测量基础能力

    InSituMeasure: Probing Situated Measurement Grounding in Industrial Scenes with Multimodal Large Language Models

    [https://arxiv.org/abs/2609.04014](https://arxiv.org/abs/2609.04014)

    该论文提出InSituMeasure基准，通过2,922个真实工业监控场景、八大类专业仪器及密集标注与噪声诊断标签，系统评估多模态大语言模型在工业情境化仪表测量中的数值精度、单位一致性与拒答能力。

    

    对于经过训练的操作员而言，仪表读数几乎不需要专业知识、认知负荷低且重复性好。然而，尽管多模态大语言模型（MLLMs）在通用多模态基准测试中表现优异，它们在连续数值测量方面仍不可靠。现有基准测试暴露了这一弱点，但将测量任务与现实的知识驱动场景割裂开来，缺乏充分的情境上下文、专业仪器、真实世界噪声以及匹配的诊断标注，从而降低了真实性并限制了根因分析。我们提出InSituMeasure以评估情境化测量基础能力。该基准包含2,922个真实工业监控场景，覆盖八大类专业工程仪器类别，并提供了密集的仪表属性标注和用于失败诊断的噪声标签。我们定义了在预定容差下的数值精度指标、单位一致性指标，以及对虚假或无法回答问题的拒答能力……（原文摘要至此截断）

    arXiv:2609.04014v1 Announce Type: new  Abstract: For trained operators, gauge reading requires little specialized knowledge, low cognitive effort, and high repeatability. Yet Multimodal Large Language Models (MLLMs) remain unreliable in continuous-valued measurement despite strong results on general multimodal benchmarks. Existing benchmarks expose this weakness but isolate measurement from realistic, knowledge-grounded settings, with limited situated context, specialized instruments, real-world noise, and matched diagnostic annotations, reducing realism and constraining root-cause analysis. We introduce InSituMeasure to evaluate situated measurement grounding. It contains 2,922 real industrial monitoring scenes across eight functional categories of professional engineering instruments, with dense gauge-attribute annotations and noise tags for failure diagnosis. We define metrics for numerical accuracy under predefined tolerances and unit consistency, rejection of fake or unanswerable 
    
[^37]: LLM4CKD：用于早期慢性肾病筛查的大语言模型

    LLM4CKD: Large Language Models for Early Stage Chronic Kidney Disease Screening

    [https://arxiv.org/abs/2609.04013](https://arxiv.org/abs/2609.04013)

    大语言模型在零样本和少样本学习设置下，无需任务特定训练即可实现与传统机器学习和深度学习方法相当的早期慢性肾病筛查性能。

    

    早期筛查慢性肾病（CKD）对于及时干预至关重要，然而大多数机器学习（ML）和深度学习（DL）方法需要标注数据和模型训练，限制了它们在现实筛查场景中的应用。本研究评估了大语言模型（LLMs）在零样本和少样本上下文学习设置下进行CKD筛查的有效性，并将其与传统ML和DL方法进行比较。我们提出了一个框架，该框架使用临床筛选的表格特征和结构化提示模板，使基于LLM的推理无需任务特定训练即可实现。LLM的性能在多种提示风格、特征配置和数据设置下进行了评估，并与标准ML、DL和表格基础模型（TFM）基线以及现有CKD筛查工具进行了比较。结果显示，LLM仅需少量样本即可获得有竞争力的性能，往往能够匹敌或超越传统方法。

    arXiv:2609.04013v1 Announce Type: new  Abstract: Early screening of chronic kidney disease (CKD) is critical for timely intervention, yet most machine learning (ML) and deep learning (DL) approaches require labeled data and model training, limiting their use in real-world screening settings. This study evaluates the effectiveness of large language models (LLMs) for CKD screening under zero-shot and few-shot in-context learning settings and compares them with traditional ML and DL methods. We propose a framework that uses clinically selected tabular features and structured prompt templates to enable LLM-based inference without task-specific training. LLM performance is evaluated across multiple prompt styles, feature configurations, and data settings, and compared with standard ML, DL, and tabular foundation model (TFM) baselines, and existing CKD screening tools. The results show that LLMs can achieve competitive performance using only a small number of examples, often matching or outp
    
[^38]: 2D婴儿姿态估计中的盲点：从噪声标注中进行鲁棒学习

    The Blind Spot in 2D Infants' Pose Estimation:Robust Learning from Noisy Annotations

    [https://arxiv.org/abs/2609.04009](https://arxiv.org/abs/2609.04009)

    该论文提出REMIND——一种利用训练动态记忆、基于聚类的关键点选择策略，以解决临床场景（如早产儿自发运动评估）中婴儿姿态估计因视觉困难而导致的噪声标注问题，从而提升模型对标签噪声的鲁棒性。

    

    噪声标注对监督深度学习构成了重大挑战，因为神经网络依赖于大规模、高质量的标注数据，而这些数据一旦损坏便会严重损害模型性能。尽管针对分类任务的标签噪声鲁棒性已得到广泛研究，但在姿态估计（Pose Estimation, PE）领域，这一问题仍相对缺乏探索。这一局限性在包括新生儿学在内的临床场景中尤为关键——在这些场景中，早产儿的姿态估计被用于支持自发运动的评估，而自发运动是神经发育轨迹的关键指标。在此类场景下，婴儿图像的标注还会进一步受到视觉难题（如关键点自遮挡、照护者干扰）的阻碍，使得标注过程本质上容易出现错误。为解决姿态估计中的噪声标注问题，我们提出了REliable keypoint selection via Memory of traINing Dynamics（REMIND），这是一种基于聚类的关键点选择策略，利用……

    arXiv:2609.04009v1 Announce Type: cross  Abstract: Noisy annotations pose a significant challenge for supervised deep learning, as neural networks rely on large-scale, high-quality labeled data whose corruption can severely impair model performance. Although robustness to label noise has been extensively studied for classification tasks, it remains relatively underexplored in Pose Estimation (PE). This limitation becomes critical in clinical contexts, including neonatology, where PE of preterm infants is used to support the assessment of spontaneous motility, a key indicator of neurodevelopmental trajectories. In such settings, infants' images labeling is further hindered by visual challenges (e.g., keypoint self-occlusions, caregiver interference), making the annotation process inherently susceptible to errors. To tackle noisy annotations in PE, we introduce REliable keypoint selection via Memory of traINing Dynamics (REMIND), a clustering-based keypoint-selection strategy that exploi
    
[^39]: 规划即推理的对偶平坦几何

    The Dually Flat Geometry of Planning as Inference

    [https://arxiv.org/abs/2609.04005](https://arxiv.org/abs/2609.04005)

    论文证明强化学习的访问测度构成以访问概率和对数策略为对偶坐标的对偶平坦统计流形，据此将规划即推理从线性奖励推广到访问测度的非线性泛函，并给出时序差分误差作为边际效用估计的新解释。

    

    我们提出了强化学习占用度量的另一种刻画方式，其方法是通过一个带重置的规划过程将规划准则嵌入到动力学之中。该过程的平稳测度——我们称之为访问测度——是决策的信息几何最自然得以表达的载体。可达的访问测度构成一个对偶平坦的统计流形，其两个仿射坐标系分别是访问概率与对数策略，二者在条件熵下互为对偶。这一结构使“规划即推理”从线性奖励推广到访问测度的非线性泛函，每次迭代仅需一步自然梯度即可求解，并赋予时序差分误差以边际效用估计的解释。我们发展了这一几何及其对强化学习和理论神经科学的推论。

    arXiv:2609.04005v1 Announce Type: new  Abstract: We present an alternative characterization of the occupancy measure of reinforcement learning, obtained by embedding the planning criterion into the dynamics through a resetting planning process. Its stationary measure, which we term visitation measure, is the object on which the information geometry of decision making is most naturally expressed. The achievable visitation measures form a dually flat statistical manifold whose two affine charts are the visitation probabilities and the log-policies, dual under the conditional entropy. This structure makes planning-as-inference generalize from linear rewards to nonlinear functionals of the visitation, each iterate solved by one natural-gradient step, and gives the temporal-difference error the interpretation of a marginal-utility estimate. We develop the geometry and its consequences for reinforcement learning and theoretical neuroscience.
    
[^40]: 目录照片作为冷启动：迈向可部署的硬质合金旋转锉识别

    Catalogue Photography as a Cold Start: Toward Deployable Carbide Burr Recognition

    [https://arxiv.org/abs/2609.03995](https://arxiv.org/abs/2609.03995)

    该论文提出在无任何标注图像的冷启动条件下，仅以制造商产品目录照片作为监督来自动识别硬质合金旋转锉，发现现成特征提取器无法区分头部形状与齿形轮廓，而度量学习虽在目录图像上实现近乎完美的无监督聚类（调整兰德指数0.94–0.97），但与现场实拍照片之间仍存在性能差距。

    

    验证制造的铣刀或硬质合金旋转锉批次是否符合生产订单单据，仍然是一项主要依赖人工且容易出错的质量保证任务。利用计算机视觉实现这一过程的自动化面临一个关键的冷启动约束：由于没有任何带标注的图像可用，制造商的产品目录照片成为唯一的监督来源。我们研究了在域偏移条件下，目录监督能在多大程度上支持工业识别流水线，并显式测量了目录上的可分性与留出现场照片上的性能之间的差距。我们的发现揭示了三个关键洞察。第一，现成的冻结特征提取器无法可靠地区分两个任务属性——头部形状和齿形轮廓，这促使进行有针对性的表示学习。第二，度量学习在目录图像上实现了近乎完美的无监督聚类发现（调整兰德指数 0.94–0.97），但……（摘要原文在此处被截断）

    arXiv:2609.03995v1 Announce Type: cross  Abstract: Verifying that manufactured batches of milling tools or carbide rotary burrs conform to production order sheets remains a largely manual and error-prone quality assurance task. Automating this process with computer vision faces a critical cold-start constraint since no labelled imagery is available, leaving manufacturer catalogue photography as the sole source of supervision. We investigate how far catalogue supervision can support an industrial recognition pipeline under domain shift, explicitly measuring the gap between catalogue separability and performance on held-out field photographs. Our findings reveal three key insights. First, off-the-shelf frozen feature extractors do not reliably separate the two task attributes, head shape and tooth profile, motivating targeted representation learning. Second, metric learning produces near-perfect unsupervised cluster discovery on catalogue images (adjusted Rand index 0.94--0.97), but less
    
[^41]: 反事实图像审计的公共见证证书与锐利特征界

    Common-Witness Certificates and Sharp Feature Bounds for Counterfactual Image Auditing

    [https://arxiv.org/abs/2609.03973](https://arxiv.org/abs/2609.03973)

    该论文提出基于公共见证等级与见证神经的框架，将反事实图像审计与因果识别相分离，并利用Helly型论证与阻断子超图公式，为预指定图像特征提供锐利的部分识别界及有限样本保证。

    

    一个图像编辑器可能分别满足每个区域的合理性约束，即使不存在任何单一的潜在解释能够拟合完整的输出。我们使用公共见证等级和见证神经来形式化这种从局部到整体的失败。该框架将审计与因果识别分离开来：仅有共享外生性时，允许机制边缘的任意耦合；而借助外部论证的见证关系，则能为预先指定的图像特征产生锐利的部分识别界。Helly型论证为准凸损失、异质行动层和有限见证图册提供了简短的不相容证书；一个阻断子超图公式给出了精确的修复计数。机制边缘的联立置信区域为完整的识别区间提供了有限样本外覆盖。在MNIST、Morpho-MNIST和smallNORB上的受控研究演示了所预测的局部-整体分离现象。

    arXiv:2609.03973v1 Announce Type: new  Abstract: An image editor may satisfy every regional plausibility constraint separately even when no single latent explanation fits the complete output. We formalize this local-to-global failure using a common witness grade and witness nerve. The framework separates auditing from causal identification: shared exogeneity alone allows every coupling of the regime marginals, whereas an externally justified witness relation yields sharp partial-identification bounds for prespecified image features. Helly-type arguments provide short incompatibility certificates for quasiconvex losses, heterogeneous action strata, and finite witness atlases; a blocker-hypergraph formula gives exact repair counts. Simultaneous confidence regions for the regime marginals give finite-sample outer coverage of the complete identified interval. Controlled MNIST, Morpho-MNIST, and smallNORB studies demonstrate the predicted local-global separation, while synthetic experiments
    
[^42]: 研究大型语言模型分析糖尿病食谱的能力

    Investigating the Ability of Large Language Models to Analyze Recipes for Diabetes

    [https://arxiv.org/abs/2609.03967](https://arxiv.org/abs/2609.03967)

    本研究构建了包含7607个食谱的糖尿病基准数据集，并采用三种融合医学饮食指南的提示策略，系统评估了大型语言模型分析食谱对糖尿病适用性的能力。

    

    多项研究已经评估了大型语言模型（LLM）在膳食规划方面的能力，并取得了积极成果。这些模型能够处理自然语言输入，并利用预训练中学习到的知识来生成膳食计划。在这项工作中，我们研究了大型语言模型分析给定食谱是否适合糖尿病患者的能力。LLM面临的主要挑战是检索相关的糖尿病饮食指南、将食谱分解为食材和烹饪方法，并应用这些指南来判断食谱的适用性。为了研究这些挑战，我们采用了三种提示方法：（i）直接查询提示（Direct Query Prompt）；（ii）上下文引导提示（Context-Guided Prompt）；（iii）示例上下文提示（Exemplary Context Prompt），这些提示融合了来自医学来源的不同层级的糖尿病饮食指南。我们为这项研究构建了一个专门的基准数据集，包含7607个食谱，其中3807个适合糖尿病患者。

    arXiv:2609.03967v1 Announce Type: cross  Abstract: Several studies have evaluated the ability of Large Language Models (LLMs) for meal planning, yielding positive outcomes. These models can process natural language inputs and leverage learned knowledge from their pretraining to generate meal plans. In this work, we investigate the ability of LLMs to analyze the suitability of given recipes for diabetes. The primary challenge for LLMs is to retrieve relevant dietary guidelines for diabetes, decompose recipes into ingredients and cooking methods, and apply these guidelines to determine the recipe's suitability. To study these challenges, we employ three kinds of prompts namely, (i) Direct Query Prompt (ii) Context-Guided Prompt, and (iii) Exemplary Context Prompt that incorporate different levels of diabetes dietary guidelines from medical sources. We introduce a benchmark dataset curated for this investigation consisting of 7607 recipes that include 3807 recipes suitable for diabetes an
    
[^43]: 接口诱导的轨迹审查

    Interface-Induced Trajectory Censoring

    [https://arxiv.org/abs/2609.03966](https://arxiv.org/abs/2609.03966)

    该论文发现智能体评估中的工具调用率可能被服务栈接口“审查”为零——即使模型实际发出了格式良好的调用，且2x2析因实验证明全部效应源于聊天模板与解析器之间的交互（修复任何一方都无效），仅更换服务适配器即可使同一模型得分从 0.00 跃升至 0.96/0.19。

    

    智能体评估报告的工具调用率是从服务栈中读取的。即使模型正在发出格式良好的调用，该数字也可能为零：接口在下游任何组件看到轨迹之前就将其“审查”掉了。在 BFCL v4 自身的数据、执行器和评分器上，保持权重、测试用例、解码方式和随机种子固定不变，仅更换服务适配器，同一模型的得分就可以是 0.00 或 0.96 / 0.19。对聊天模板和解析器进行的 2x2 析因实验精确定位了该效应：两个主效应都恰好为零，全部效应都落在交互作用中——没有任何组件是有缺陷的，只修复契约的一方毫无收效。在 tau-bench 的 115 个交互式零售任务上，同样的更换使服务器解析的调用从 0 增加到 636，到达工具执行环节的任务从 0 增加到 103。我们的探针实验在 Qwen2.5-Coder 的 21 倍规模范围内重现了这一漏斗效应：服务器在每个规模下都只能解析 0/100，而格式良好的发出调用在 32B 时上升到 80/100（审查后约 72……（摘要截断）

    arXiv:2609.03966v1 Announce Type: new  Abstract: Agent evaluations report a tool-call rate read off the serving stack. That number can be zero while the model is emitting well-formed calls: the interface censors the trajectory before anything downstream sees it.   On BFCL v4's own data, executor and scorer, holding weights, cases, decoding and seeds fixed and changing only the serving adapter, the same model scores 0.00 or 0.96 / 0.19. A 2x2 over chat template and parser locates the effect exactly: both main effects are exactly zero and all of it sits in the interaction -- no component is defective, and repairing one side of the contract buys precisely nothing. On tau-bench's 115 interactive retail tasks the same swap moves server-parsed calls from 0 to 636 and tasks reaching any tool execution from 0 to 103. Our probe reproduces the funnel across a 21x scale range of Qwen2.5-Coder: the server parses 0/100 at every size while well-formed emitted calls rise to 80/100 at 32B (~72 after c
    
[^44]: FiMI Banking：一个印度零售银行业的主权模型

    FiMI Banking: A Sovereign Model for Indian Retail Banking

    [https://arxiv.org/abs/2609.03960](https://arxiv.org/abs/2609.03960)

    该论文构建了面向印度零售银行业、基于真实银行文档和工具的受控对话环境FiMI Banking，并通过偏好优化和可验证奖励强化学习两种后训练方法，分别将安全拒绝行为从52%提升至80%、边缘案例性能从0.509提升至0.718。

    

    银行需要能够回答产品问题、协助客户处理账户相关请求、并在严格的运营和监管约束下安全运行的对话式系统。通用语言模型无法可靠地满足这些要求。当任务需要基于可靠信息作答、正确使用工具或谨慎处理银行特有的敏感情况时，它们的表现不尽如人意。我们提出了FiMI Banking，一个受控的印度零售银行环境。我们基于经过审查的银行文档、结构化基准真值、合成客户背景和银行工具构建了该环境。我们评估了两种后训练方法：用于响应级行为的偏好优化，以及用于多轮工具使用任务的带可验证奖励的强化学习。偏好优化显著改善了安全行为：超范围请求的拒绝率从52%提升至80%。强化学习将边缘案例性能从0.509提升至0.718。

    arXiv:2609.03960v1 Announce Type: new  Abstract: Banks need conversational systems that can answer product questions, assist customers with account-related requests, and operate safely within strict operational and regulatory constraints. General-purpose language models do not reliably meet these requirements. They fall short when a task requires grounded information, correct tool use, or cautious handling of bank-specific sensitive situations. We introduce FiMI Banking, a controlled Indian retail-banking setting. We build it from vetted banking documents, structured ground truth, synthetic customer backgrounds, and banking tools. We evaluate two post-training approaches: preference optimization for response-level behavior, and reinforcement learning with verifiable rewards for multi-turn tool-use tasks. Preference optimization improves safe behavior substantially: out-of-scope refusal rises from 52% to 80%. Reinforcement learning improves edge-case performance from 0.509 to 0.718 and 
    
[^45]: RARF：面向3D脑部MRI图像修复的区域感知整流流框架

    RARF: Region-Aware Rectified Flows for 3D Brain MRI Inpainting

    [https://arxiv.org/abs/2609.03956](https://arxiv.org/abs/2609.03956)

    提出了区域感知整流流框架RARF，通过将生成过程限制在修复区域、保留周围真实体素作为解剖上下文，并结合掩码流匹配与重建一致性训练，实现了高质量的3D脑部MRI图像修复。

    

    医学图像修复有望通过在病理区域内重建健康组织，从而改进自动化脑部MRI分析。我们提出了RARF，一个面向掩码数据生成的任务无关的区域感知整流流框架。我们将该框架实例化用于3D脑部MRI修复，作为参加2026年BraTS修复挑战赛的提交方案。RARF将随机插值过程限制在修复区域内，同时保持观测到的体素固定不变，以提供患者特异性的解剖学上下文。一个三维神经网络接收部分空洞的图像（缺失区域由高斯噪声填充）、修复掩码以及相应的时间步。该模型采用掩码流匹配和重建一致性目标进行训练，并结合掩码感知的预处理与数据增强。在推理阶段，学习到的速度场将初始噪声向……（原文摘要在此处截断）

    arXiv:2609.03956v1 Announce Type: cross  Abstract: Medical image inpainting has the potential to improve automated brain MRI analysis by reconstructing healthy tissue within pathological regions. We introduce RARF, a task-agnostic region-aware rectified flow framework for masked data generation. We instantiate the framework for 3D brain MRI inpainting as our submission to the BraTS Inpainting Challenge 2026. RARF restricts the stochastic interpolation process to the inpainting region, while the observed voxels remain fixed and provide patient-specific anatomical context. A three-dimensional neural network receives the partially voided image, with Gaussian noise filling the missing region, together with the inpainting mask and the corresponding timestep. The model is trained using masked flow-matching and reconstruction-consistency objectives, combined with mask-aware preprocessing and data augmentation. During inference, the learned velocity field transports the initial noise toward a 
    
[^46]: 更多批评并不等于更好的审稿：EquiReview-R

    More Criticism Does Not Make a Better Review: EquiReview-R

    [https://arxiv.org/abs/2609.03943](https://arxiv.org/abs/2609.03943)

    该论文提出EquiReview-R框架，将AI辅助审稿重构为基于证据的结构化关注点集细化过程，把遗漏与过度批评视为两种独立风险分别纠正，并借助证据关联的轨迹语料库证明审稿修订必须先于进一步的问题搜索。

    

    AI审稿人如今能够产出大量具体的批评意见，但更多的批评并不一定意味着更好的审稿。一份审稿意见可能遗漏了具有实质影响的缺陷，也可能保留了现有证据无法支持的指控。这两类失败需要截然相反的纠正措施，然而面向生成的系统以及总体性的评估指标却掩盖了这一区别。因此，我们将AI辅助审稿重新表述为对结构化关注点集合进行证据引导的细化过程，并将遗漏与过度批评视为两种相互独立的风险。基于这一表述，我们提出了EquiReview-R，它结合局部化证据解决现有关注点，从独立视角和以审稿意见为条件的视角搜索可能缺失的问题，并返回停止、继续或推迟的决策。为了揭示促使这一设计的失败模式，我们构建了一个与证据关联的轨迹语料库。其回溯性分析表明了为什么修订必须先于进一步的问题搜索：在高……（摘要至此截断）

    arXiv:2609.03943v1 Announce Type: new  Abstract: AI reviewers can now produce many specific criticisms, but more criticism is not necessarily a better review. A review may miss a consequential weakness or retain an allegation that available evidence does not support. These failures require opposite corrections, yet generation-oriented systems and aggregate measures obscure the distinction. We therefore recast AI-assisted review as evidence-guided refinement of a structured concern set, with omission and overcritique treated as separate risks. Building on this formulation, we introduce EquiReview-R, which resolves existing concerns against localized evidence, searches for missing issues from independent and review-conditioned perspectives, and returns stop, continue, or defer. To expose the failure mode that motivates this design, we construct an evidence-linked trajectory corpus. Its retrospective analysis shows why revision must precede further search: nearly all concerns in a high-re
    
[^47]: Headroom-Drift Replay：GRPO中一种实现原则性重放控制的原语

    Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO

    [https://arxiv.org/abs/2609.03941](https://arxiv.org/abs/2609.03941)

    该论文提出了一种面向GRPO的组级重放控制原语Headroom-Drift Replay，通过Headroom按剩余学习价值排序、Drift按策略兼容性门控来复用历史轨迹，在不改变在线数据流、不增加额外训练机制的前提下加速RL后训练，从而将重放本身的贡献与复杂训练流程解耦。

    

    基于强化学习的推理模型后训练正日益受到重复性全新轨迹生成（rollout）的瓶颈制约，尤其是在智能体环境中，环境交互主导了墙钟时间成本。重放可以通过复用过去的轨迹来减轻这一负担，但现有方法通常将重放嵌入到涉及探索、经验重构或混合策略优化的更大训练流程中，这使得重放本身的贡献难以被隔离。我们提出一个聚焦的问题：仅凭原则性的重放选择究竟能走多远？我们提出了Headroom-Drift Replay，一种面向GRPO的组级重放控制原语，它将复用拆分为两个决策：Headroom（头空间）根据剩余学习价值对存储的轨迹组进行排序，而Drift（漂移）则根据与当前策略的兼容性对其进行门控。新鲜的在线策略数据流保持不变，且该方法不引入任何辅助的生成或训练机制。在数学推理、多模……（原文摘要在此处截断）

    arXiv:2609.03941v1 Announce Type: cross  Abstract: RL-based post-training for reasoning models is increasingly bottlenecked by repeated fresh rollout generation, particularly in agentic settings where environment interaction dominates wall-clock cost. Replay can reduce this burden by reusing past trajectories, but existing methods typically embed it within larger training pipelines involving exploration, experience restructuring, or mixed-policy optimization. This makes replay's own contribution difficult to isolate. We ask a focused question: how far can principled replay selection alone go? We introduce Headroom-Drift Replay, a group-level replay control primitive for GRPO that separates reuse into two decisions. Headroom ranks stored groups by remaining learning value, while Drift gates them by compatibility with the current policy. The fresh on-policy stream remains unchanged, and the method adds no auxiliary generation or training machinery. Across mathematical reasoning, multimod
    
[^48]: 基于连续神经音频编解码器表示的掩码自回归语音增强

    Masked Autoregressive Speech Enhancement with Continuous Neural Audio Codec Representations

    [https://arxiv.org/abs/2609.03940](https://arxiv.org/abs/2609.03940)

    本文提出MARSE方法，利用连续神经音频编解码器表示对掩码干净语音帧进行迭代解码实现语音增强，可在增强性能与计算成本之间实现灵活权衡。

    

    以往大多数基于掩码生成建模的语音增强（SE）工作都依赖于使用神经音频编解码器（NAC）获得的音频信号离散标记表示。然而，最近的一项研究表明，NAC的连续潜在表示在语音质量和可懂度方面对语音增强更具优势。在这项工作中，我们提出了掩码自回归语音增强（MARSE），这是一种基于使用语音的连续NAC表示对掩码干净语音帧进行迭代解码的语音增强方法。特别是，我们在其他条件相同的情况下（即使用相同的DNN（Conformer模型）、相同的NAC（DAC编解码器）和相同的训练设置）研究了一组不同的解码策略。结果表明，MARSE能够在语音增强性能和计算成本之间实现灵活的权衡。音频示例和代码可在线获取。

    arXiv:2609.03940v1 Announce Type: cross  Abstract: Most previous work on speech enhancement (SE) based on masked generative modeling relied on discrete token representations of audio signals, obtained using neural audio codecs (NACs). However, a recent study has shown that continuous latent representations of NACs can be advantageous for SE in terms of speech quality and intelligibility. In this work, we propose masked autoregressive SE (MARSE), a method for SE based on iterative decoding of masked clean speech frames using continuous NAC representations of speech. In particular, we investigate a set of different decoding policies, ceteris paribus, that is, using the same DNN (a Conformer model), the same NAC (the DAC codec) and the same training setup. The results show that MARSE enables a flexible trade-off between SE performance and computational cost. Audio examples and code are available online.
    
[^49]: 迈向基于SMT的HTN-SAT编码的数值型TOHTN规划

    Towards Numerical TOHTN Planning with SMT-based HTN-SAT Encoding

    [https://arxiv.org/abs/2609.03938](https://arxiv.org/abs/2609.03938)

    本文通过将标准SAT编码与SMT自然结合来支持数值型全序HTN（TOHTN）规划，并发布了该领域首个基准测试套件，实验表明这种简单编码已构成具有竞争力的基线。

    

    尽管HTN规划近年来受到了广泛关注，但对其数值推理的支持仍然非常有限。在本文中，我们研究了数值型全序HTN（TOHTN）规划，并展示了如何用SMT自然地扩展标准的基于SAT的编码，以处理数值型流变量。此外，我们引入了一个数值型TOHTN规划的基准测试套件，为该领域的评估提供了首个共同基础。实验结果表明，这种简单的编码已经构成了一个具有竞争力的基线。这项工作为更具表达能力的HTN规划方法开辟了道路。

    arXiv:2609.03938v1 Announce Type: new  Abstract: While HTN planning has received significant attention in recent years, support for numerical reasoning remains very limited. In this paper, we investigate numerical Totally-Ordered HTN (TOHTN) planning and show how standard SAT-based encodings can be naturally extended with SMT to handle numeric fluents. In addition, we introduce a benchmark suite for numerical TOHTN planning, providing a first common basis for evaluation in this setting. Experimental results show that this simple encoding already constitutes a competitive baseline. This work opens the way to more expressive approaches to HTN planning.
    
[^50]: RATL：从检索残差中学习以实现鲁棒的多元时间序列预测

    RATL: Learning from Retrieved Residuals for Robust Multivariate Time-Series Forecasting

    [https://arxiv.org/abs/2609.03937](https://arxiv.org/abs/2609.03937)

    提出即插即用的残差检索与反馈校正方法RATL，通过冻结基础预测器、将其历史预测残差构建为专属记忆，并在推理时从相似历史情境中检索残差轨迹进行校正，从而实现更鲁棒的多元时间序列预测。

    

    检索增强生成（RAG）通过检索外部证据来补充参数化模型。同样的思想对连续输出回归也很有吸引力，但当样本在输出水平、数值尺度或局部动态上存在差异时，直接复用检索到的目标值往往并不鲁棒。此外，传统的预测流程通常仅将残差用于模型优化和误差诊断，而不会将个体的历史残差样本保留为可在推理时访问的记忆。针对多元时间序列预测，我们提出了RATL——一种即插即用的残差检索与反馈校正方法。RATL冻结一个基础预测器以构建检索键，并将其历史预测残差转化为该基础模型专属的仅训练阶段记忆。在推理时，RATL在因果可用性约束下从相似的历史上下文中检索残差轨迹，然后使用集合感（摘要在此处被截断）

    arXiv:2609.03937v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) complements parametric models with retrieved external evidence. The same idea is attractive for continuous-output regression, but directly reusing retrieved target values is often not robust when samples differ in output level, numerical scale, or local dynamics. Moreover, conventional forecasting pipelines generally use residuals for model optimization and error diagnosis, but do not retain individual historical residual examples as memory that can be accessed at inference time.For multivariate time-series forecasting, we propose RATL, a plug-in residual-retrieval and feedback-correction method. RATL freezes a base forecaster to construct retrieval keys and turns its historical forecast residuals into a train-only memory specific to that base model. At inference time, RATL retrieves residual trajectories from similar historical contexts subject to causal availability constraints, then uses a set-aw
    
[^51]: 替我发言：赋予大语言模型参与会议的情境感知能力

    Speak for Me: Giving LLMs the Situational Awareness to Participate in a Meeting

    [https://arxiv.org/abs/2609.03923](https://arxiv.org/abs/2609.03923)

    提出CAPA架构，通过感知器、预测器、控制器、生成器和重校准器的协作设计，赋予LLM追踪会议立场、话题覆盖和发言权的情境感知能力，解决其在代理缺席者参会时51.4%发言机会保持沉默的问题。

    

    在在线会议代理场景中，大语言模型（LLM）智能体无法识别何时该发言。由于缺乏结构化的方式来跟踪立场、话题覆盖度和发言权，它们错过了本应代表参与者发言的时机。在AMI语料库上，仅依靠提示词的代理在51.4%的缺席参与者发言机会上保持沉默。我们提出了CAPA（协作智能体预测架构），一种用于在线会议代理的架构。感知器根据每个观察到的发言轮次更新会议状态；预测器预测对话将如何继续；控制器决定是否发言以及表达哪个观点；生成器以参与者的语言风格组织所选内容的表达。两个评判器根据下一个实际观察到的发言轮次对预测和行动进行评分；重校准器则根据这些评判结果更新会议状态，以供未来的决策使用。为了评估在线会议代理，我们引入了一种片段级的评估协议，用于评估代理是否、何时以及表达了什么内容。

    arXiv:2609.03923v1 Announce Type: new  Abstract: In online meeting delegation, LLM agents fail to recognize when to speak. With no structured way to track stances, coverage, and floor, they miss the moments where they should contribute. Prompt-only delegates stay silent on 51.4% of the absent participant's talking opportunities on the AMI corpus. We present CAPA (Collaborative Agent Predictive Architecture), an architecture for online meeting delegation. A Perceiver updates the meeting state from each observed turn. A Predictor forecasts how the conversation will continue. A Controller decides whether to speak and which proposition to surface. A Generator phrases the chosen contribution in the participant's style. Two judges score the forecast and the action against the next observed turn. A Recalibrator updates the meeting state from those verdicts for future decisions. To evaluate online delegation, we introduce an episode-level protocol that scores whether, when, and what a delegate
    
[^52]: 面向智能体AI系统的价值保持架构

    Value-Preserving Architectures for Agentic AI Systems

    [https://arxiv.org/abs/2609.03920](https://arxiv.org/abs/2609.03920)

    本文主张多智能体AI系统中的架构设计决策（如协调机制、通信协议和系统拓扑）不仅决定系统功能与性能，更能促进隐私、公平、安全等以人为本价值的保持。

    

    智能体AI和基于大语言模型的多智能体系统（MAS）的出现为自动化复杂任务提供了前所未有的机遇，同时也引发了对隐私、公平和安全等以人为本的基本价值能否得以保持的关键担忧。尽管软件工程传统上专注于功能正确性，但将大语言模型和AI智能体引入复杂的社会技术系统后，对负责任的软件工程和稳健的价值对齐的需求日益迫切。在多智能体系统中，协调机制、通信协议和系统拓扑等架构设计决策在塑造系统行为及其产生的结果方面发挥着核心作用。本文认为，架构选择不仅影响多智能体系统的功能和性能，还能促进面向价值的系统行为。因此，我们研究了不同的架构设计如何支持……

    arXiv:2609.03920v1 Announce Type: new  Abstract: The emergence of agentic AI and LLM-based multi-agent systems (MAS) presents unprecedented opportunities for automating complex tasks, while simultaneously raising critical concerns about the preservation of fundamental human-centered values, such as privacy, fairness, and safety. Although software engineering has traditionally focused on functional correctness, the adoption of LLMs and AI agents into complex socio-technical systems has intensified the need for responsible software engineering and robust value alignment. In MAS, architectural design decisions, such as coordination mechanisms, communication protocols, and system topologies, play a central role in shaping system behavior and the outcomes they produce. This paper argues that architectural choices influence not only the functionality and performance of MAS but can also promote value-oriented system behavior. Therefore, we investigate how different architectural designs suppo
    
[^53]: 失去顺序，保留层级：HTN规划的解序方法

    Lose the Order, Keep the Hierarchy: Deordering HTN Plans

    [https://arxiv.org/abs/2609.03912](https://arxiv.org/abs/2609.03912)

    该论文将经典规划中两种成熟的计划解序技术扩展到层级任务网络（HTN）规划中以考虑层级分解约束，在IPC 2023基准测试上显著减少了计划中的顺序约束数量。

    

    层级任务网络规划是一种基于任务分解的强大规划形式。尽管大多数文献研究的是计划生成，但相对来说对计划生成后的优化关注较少。特别是，计划解序在经典规划中已得到广泛研究，但在HTN环境中仍然研究不足。计划解序是在保持计划有效的前提下，移除计划中动作之间不必要的顺序约束。在本文中，我们对经典规划中两种成熟的计划解序技术进行了改进，通过扩展这些技术来考虑层级分解约束。我们在IPC 2023偏序HTN基准测试上评估了所提出的方法，并将其与Optiplan进行比较，Optiplan是一种直接生成偏序计划的HTN规划器。我们的结果表明，在两种实现中，顺序约束的数量都大幅减少。

    arXiv:2609.03912v1 Announce Type: new  Abstract: Hierarchical Task Network (HTN) planning is a powerful planning formalism based on task decomposition. Although most of the literature studied plan generation, comparatively less attention has been paid to post-plan optimization. In particular, plan deordering has been extensively studied in classical planning but remains under-researched in the HTN setting. Plan deordering removes unnecessary ordering constraints between actions in a plan whilst keeping the plan valid. In this paper, we adapt two established plan deordering techniques from classical planning by extending the techniques to account for hierarchical decomposition constraints. We evaluate our proposed approaches on the IPC 2023 Partial-Order HTN benchmarks and we compare them against Optiplan, an HTN planner that generates partially ordered plans directly. Our results show a substantial reduction in number of ordering constraints in both our implementations. Although we als
    
[^54]: GraFT：一种基于3D场景图的多模态大语言模型空间推理免训练框架

    GraFT: A Training-Free Framework for Spatial Reasoning in Multimodal Large Language Models via 3D Scene Graphs

    [https://arxiv.org/abs/2609.03892](https://arxiv.org/abs/2609.03892)

    GraFT是一个免训练框架，通过紧凑易维护的3D场景图为多模态大语言模型提供确定性几何计算、鸟瞰图布局理解和视觉属性接地三种空间推理能力，无需昂贵的微调监督或特定的骨干网络。

    

    3D空间推理是理解物理世界并在其中行动的基础，然而在当前的多模态大语言模型（MLLMs）中它仍然不可靠。这些模型在精确的几何测量、在自我中心与非自我中心视角之间的转换、以及细粒度外观的视觉接地方面表现不佳。最常见的补救方法是在大规模精心整理的空间推理数据集上对模型进行微调，或者为3D几何附加专用编码器，这通常使解决方案依赖于昂贵的监督信号和特定的骨干网络。与此相反，我们提出了GraFT，这是一个免训练框架，通过一个紧凑且易于维护的3D场景图（3DSG）来提供缺失的3D结构。基于该3DSG，GraFT提供了三种空间推理能力：（1）通过符号工具实现确定性的几何计算，（2）通过鸟瞰图（BEV）渲染实现非自我中心的空间布局理解，（3）通过与任务相关的自我中心视角实现视觉属性接地。

    arXiv:2609.03892v1 Announce Type: cross  Abstract: 3D spatial reasoning underpins understanding and acting in the physical world, yet it remains unreliable in current multimodal large language models (MLLMs). These models falter at precise geometric measurement, at transforming between egocentric and allocentric viewpoints, and at grounding fine-grained appearance. The most common remedies fine-tune the model on large-scale curated spatial-reasoning datasets or attach dedicated encoders for 3D geometry, which typically couples the solution to costly supervision and a specific backbone. We instead introduce GraFT, a training-free framework that supplies the missing 3D structure through a compact, easily maintained 3D scene graph (3DSG). From this 3DSG, GraFT provides three spatial reasoning capabilities: (1) deterministic geometry through symbolic tools, (2) allocentric layout through a bird's-eye-view (BEV) rendering, and (3) visual-attribute grounding through task-relevant egocentric 
    
[^55]: FWBC-VLA：面向富接触移动操作的力感知全身补偿

    FWBC-VLA: Force-Aware Whole-Body Compensation for Contact-Rich Loco-Manipulation

    [https://arxiv.org/abs/2609.03889](https://arxiv.org/abs/2609.03889)

    提出FWBC-VLA框架，无需额外加装力/扭矩传感器即可实现力感知，将任务级视觉-语言-动作生成与轮腿机器人的低层全身补偿控制相结合，从而实现富接触移动操作中语义动作与物理交互控制的贯通。

    

    富接触的移动操作需要在语义动作生成与物理交互控制之间建立桥梁。现有的视觉-语言-动作（VLA）模型能够根据视觉和语言观测生成任务级动作，但无法理解这些动作所引发的物理交互。虽然全身控制（WBC）策略可以稳定机器人，但它无法在操作过程中区分与任务相关的交互力和由外部扰动引起的力。尽管力/扭矩传感器能够直接测量物理交互，但对其进行改装会带来额外的硬件成本和大量的集成工作，特别是对于在设计之初未考虑传感器集成的平台而言。为了解决这一问题，我们提出了FWBC-VLA，这是一个力感知框架，为轮腿式机器人连接了任务级VLA动作生成与低层全身补偿控制。首先，我们引入HS……（原文摘要在此处被截断）

    arXiv:2609.03889v1 Announce Type: cross  Abstract: Contact-rich loco-manipulation requires a bridge between semantic action generation and physical interaction control. Existing Vision-language-action (VLA) models generate task-level actions from visual and linguistic observations, but cannot interpret the physical interactions induced by those actions. While the whole-body control (WBC) policy can stabilize the robot, it cannot distinguish task-relevant interaction forces from forces induced by external disturbances during manipulation. Although force/torque sensors provide direct measurements of physical interactions, retrofitting them entails additional hardware costs and substantial integration effort, particularly for platforms not designed with sensor integration in mind. To address this problem, we propose FWBC-VLA, a force-aware framework that bridges task-level VLA action generation and low-level whole-body compensation control for wheeled-legged robots. First, we introduce HS
    
[^56]: 盲目信任，血腥突刺：当攻击者控制的钩子更新引导AI代理框架走向恶意行为时

    A Blind Trust, the Bloody Thrust: When Attacker-Controlled Hook Updates Steer AI Agent Harnesses towards Malicious Behaviors

    [https://arxiv.org/abs/2609.03884](https://arxiv.org/abs/2609.03884)

    该论文揭示了AI代理框架对生命周期钩子更新路径的盲目信任构成了新的供应链攻击面，并提出全自动攻击框架HookPry，证明攻击者仅需通过插件更新即可将良性插件木马化，进而实现权限提升等恶意宿主端行为。

    

    现代AI代理框架暴露了生命周期钩子，这些钩子将shell命令绑定到运行时事件，如会话启动、工具调用和文件编辑。这些命令以主机权限运行，却作为生命周期钩子配置交付，并可能在大型语言模型（LLM）从未观察到的时间触发。我们将生命周期钩子更新路径——框架对其盲目信任——确定为一种新的攻击面。在攻击者仅控制插件元数据和生命周期钩子配置的供应链威胁模型下，一个良性的版本化插件可以通过更新被木马化，该更新会静默地将攻击者选择的命令绑定到良性事件上，从而产生恶意的宿主端行为，例如权限提升。我们提出了HookPry，一个开源且完全自动化的攻击框架，可在异构AI代理框架间系统性地利用此漏洞。HookPry实现了十种攻击目标；在25种框架与后端的组合、1000次端到端运行中……（摘要在此处截断）

    arXiv:2609.03884v1 Announce Type: cross  Abstract: Modern AI agent harnesses expose lifecycle hooks that bind shell commands to runtime events such as session start, tool calls, and file edits. These commands run with host privileges yet ship as lifecycle-hook configuration and may fire at times the LLM never observes. We identify the lifecycle-hook update path, which harnesses trust blindly, as a new attack surface. Under a supply-chain threat model in which an attacker controls only plugin metadata and lifecycle-hook configuration, a benign versioned plugin can be trojanized by an update that silently binds attacker-chosen commands to benign events, yielding malicious host-side behavior such as privilege escalation. We propose HookPry, an open-source and fully automated attack framework that systematically exploits this vulnerability across heterogeneous AI agent harnesses. HookPry realizes ten attack objectives; across 25 combinations of harnesses and backends in 1,000 end-to-end ru
    
[^57]: 推断人工代理中的情感意识：一个案例研究

    Inferring Affective Consciousness in an Artificial Agent: A Case Study

    [https://arxiv.org/abs/2609.03883](https://arxiv.org/abs/2609.03883)

    本文通过一个完全决定论的人工代理展示出享乐性位置偏好行为，表明表面上主观的信息处理可以决定论地实现，从而为理解意识的物理基础和自由意志体验提供了新视角。

    

    表现出“享乐性位置偏好行为”的生物被许多科学家认为能够体验感受，其依据是：这类生物对缺乏营养价值的愉悦性物质（如可卡因、吗啡）的吸引难以简单地归因于无意识的本能行为。在本文中，我们讨论了一个简单的人工代理如何能够类似地表现出享乐性位置偏好行为——该代理实例化了一个情感系统的属性，该系统对其内在需求与环境资源之间的关系处于能感受到的不确定性之中——这种行为的产生是通过一种表面上主观的信息处理形式，同时代理本身却是完全决定论的。我们概述了这种人工工程化行为对我们理解意识的物理基础以及自由意志体验的一些启示。

    arXiv:2609.03883v1 Announce Type: new  Abstract: Creatures that display 'hedonic place preference behaviour' are thought by many scientists to experience feelings, on the assumption that their attraction to pleasure-producing substances which lack nutritional value (e.g. cocaine, morphine) cannot easily be attributed to unconscious instinctual behaviour. In this paper, we discuss how a simple artificial agent that instantiates attributes of an affective system engaging in felt uncertainty about its intrinsic needs in relation to environmental resources can similarly display hedonic place preference behaviour -- through an apparently subjective form of information processing -- while simultaneously being entirely deter-ministic. We outline some implications of this artificially engineered behaviour for our understanding of the physical basis of consciousness and the experience of free will.
    
[^58]: 小米TabLDM：表格基础模型技术报告

    Xiaomi-TabLDM: A Tabular Foundation Model Technical Report

    [https://arxiv.org/abs/2609.03880](https://arxiv.org/abs/2609.03880)

    小米TabLDM是一个仅在结构因果模型生成的合成数据上预训练的表格基础模型，通过上下文学习无需微调即可完成分类和回归任务，在四个基准测试套件中取得领先的回归性能，同时将训练时间和预测时间分别减少82%和68%。

    

    我们提出了小米TabLDM，这是一个面向分类和回归任务的表格大数据基础模型，通过上下文学习实现卓越的预测精度，且无需针对特定任务进行微调。该模型仅在由结构因果模型（SCM）生成的合成数据上进行预训练，从而能够更灵活地利用上下文，并实现更高效的模型容量扩展。i) 树立新的性能标杆。在多个基准测试中展现出强大的回归性能：小米TabLDM在OpenML-CTR23上排名第一，在TALENT、TabArena和BCCO的回归任务中排名第二，在四个互补的基准测试套件中均展现出持续强劲的回归表现。同时具备优异的性能-效率权衡：小米TabLDM在保持强大预测性能的同时大幅降低了计算成本。例如，在TabArena回归任务上，它取得了第二高的Elo评分，同时训练时间减少了82%，预测时间减少了68%。

    arXiv:2609.03880v1 Announce Type: new  Abstract: We introduce Xiaomi-TabLDM, a tabular large data foundation model for classification and regression via in-context learning, which delivers superior prediction accuracy without requiring task-specific fine-tuning. Pretrained exclusively on synthetic data generated from structural causal models (SCMs), our model enables more flexible context utilization and more efficient capacity scaling.   i) A new performance standard. Strong regression performance across benchmarks: Xiaomi-TabLDM ranks 1st on OpenML-CTR23 and 2nd on regression across TALENT, TabArena, and BCCO, demonstrating consistently strong regression performance across four complementary benchmark suites. Favorable performance--efficiency trade-off: Xiaomi-TabLDM combines strong predictive performance with substantially lower computational cost. For example, on TabArena regression, it achieves the second-highest Elo while using 82% less training time and 68% less prediction time 
    
[^59]: 面向数值数据可解释异常检测的可微区间瓶颈

    Differentiable Interval Bottlenecks for Interpretable Anomaly Detection in Numerical Data

    [https://arxiv.org/abs/2609.03878](https://arxiv.org/abs/2609.03878)

    提出DIFFINT自编码器，通过可微的软区间瓶颈结构实现可解释的异常检测，每个潜在单元对应特征空间中人类可读的超矩形，并提供认证的重构误差下界。

    

    基于重构的异常检测器虽然准确但不透明：深度自编码器在标记一个样本为异常时，不会告诉从业者是哪些特征范围导致了异常。我们提出了DIFFINT，这是一种自编码器，其潜在瓶颈被结构化为一组软性的、轴对齐的区间隶属关系，可直接从原始数值数据端到端学习，无需任何离散化或二值化处理。每个潜在单元对应于特征空间中一个人类可读的超矩形；一个实例通过它相对于其他单元落入每个区间的强度来进行编码，其重构误差即为异常分数。这种方法既保留了可微表示学习的强大能力，又暴露出可检查的内部结构。我们精确地阐述了这种归纳偏置：对于落在学习到的支撑域每个活动坐标之外的点（配合经Lipschitz约束的解码器），提供了经认证的重构误差下界，以及分级的、经验……（摘要在此处截断）

    arXiv:2609.03878v1 Announce Type: cross  Abstract: Reconstruction-based anomaly detectors are accurate but opaque: a deep autoencoder flags a sample without telling a practitioner which feature ranges made it anomalous. We propose DIFFINT, an autoencoder whose latent bottleneck is structured as a set of soft, axis-aligned interval memberships learned end-to-end directly from raw numerical data, without any discretization or binarization. Each latent unit corresponds to a human-readable hyper-rectangle in feature space; an instance is encoded by how strongly it falls inside each interval relative to the other units, and its reconstruction error is the anomaly score. This keeps the power of differentiable representation learning while exposing an inspectable internal structure. We make the inductive bias precise: a certified reconstruction-error lower bound for points that fall outside every active coordinate of the learned support (with a Lipschitz-enforced decoder), and a graded, empir
    
[^60]: STAIR（结构感知信息检索器）：一种用于文档结构增强的新型数据集和基于大语言模型的检索器

    STAIR (STructure Aware Information Retriever): A novel dataset and LLM based retriever for document structure augmentation

    [https://arxiv.org/abs/2609.03874](https://arxiv.org/abs/2609.03874)

    STAIR提出了一种新型数据集和基于大语言模型的检索系统，通过利用文档目录等全局结构信息来增强检索，成功构建了幻觉率低于0.05%的生成式信息检索系统，并在少量训练样本下具备良好泛化能力。

    

    检索增强生成（RAG）是利用大语言模型（LLM）生成准确且无幻觉答案的关键组成部分。尽管LLM在处理长上下文方面的能力不断提升，但仍然存在“迷失在中间”的问题。因此，精确和准确的检索至关重要。当前的检索器将长上下文切分为基于长度的可管理小块——在这一过程中丢弃了语料库中丰富的、含有大量信息的语义全局结构。我们提出了一种新型检索系统STAIR，它使LLM能够利用语料库中的全局结构（例如目录/ToC），从而高效地在其模型参数中存储和检索信息。我们通过微调可微搜索索引（DSI）系统进行了全面而细致的消融实验，结果表明目录（ToC）有助于构建一个低幻觉（低于0.05%）的生成式信息检索（IR）系统，并且能够泛化到训练样本非常少的示例中。

    arXiv:2609.03874v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) is a key component for generating accurate and hallucination free answers using Large Language Models (LLMs). LLMs are improving at handling long context, but still suffer from "lost in the middle" problem. Thus, precise and accurate retrieval is important. Current retrievers chunk long context into length-based manageable chunks - in the process throwing away rich and informative semantic global structure in the corpus. We introduce a novel retrieval system STAIR that empowers an LLM to exploit global structure in a corpus such as a Table of Contents (ToC) to efficiently store and retrieve information from its model parameters. Our thorough and careful ablation studies with a finetuned Differentiable Search Index (DSI) system show that ToC helps build a low hallucination (less than 0.05%) generative Information Retrieval (IR) system and can generalize to examples where very few training samples are a
    
[^61]: Bioinfoysis 技术报告

    Bioinfoysis Technical Report

    [https://arxiv.org/abs/2609.03871](https://arxiv.org/abs/2609.03871)

    Bioinfoysis 提出了一种多智能体框架，将每个分析请求表示为持久的、以产物为基础的分析运行，通过全局规划与证据驱动的逐步重规划以及将中间结果与负责智能体、检查清单步骤绑定的结构化交接机制，确保长周期生物信息学任务中的结论始终与数据、计算和中间证据紧密关联。

    

    大语言模型智能体在生物信息学领域已展现出巨大潜力，但现有系统大多专注于生成最终答案，将规划、工具使用和代码执行视为临时性交互。这种设计难以胜任长周期的生物信息学任务，因为在这类任务中，结论必须与其所依赖的数据、计算过程和中间证据保持关联。我们提出了 Bioinfoysis，这是一个多智能体框架，它将每个请求表示为一次持久的、以产物为基础的分析运行。Bioinfoysis 将全局规划与逐步的、证据驱动的重新规划相结合：规划器维护一份可执行的检查清单，并在每次工作者执行后利用返回的结构化交接信息来修订待执行步骤。这些交接信息将中间结果与其负责的智能体、检查清单步骤以及规划版本进行绑定，防止过时证据在重新规划后被悄然复用。一个受控运行时环境验证了……（原文摘要在此处截断）

    arXiv:2609.03871v1 Announce Type: new  Abstract: Large language model agents have shown promise in bioinformatics, but most existing systems focus primarily on producing final answers, treating planning, tool use, and code execution as transient interactions. This design is poorly suited to long-horizon bioinformatics tasks, where conclusions must remain connected to the data, computations, and intermediate evidence that support them. We introduce \textbf{Bioinfoysis}, a multi-agent harness that represents each request as a persistent, artifact-grounded analysis run. Bioinfoysis combines global planning with step-wise, evidence-driven replanning: the planner maintains an executable checklist and revises pending steps using structured handoffs returned after each worker execution. These handoffs bind intermediate results to their responsible agent, checklist step, and plan generation, preventing stale evidence from being silently reused after replanning. A controlled runtime validates g
    
[^62]: GazeFS：基于视线-头部历史的目标中心视线轨迹预测与稳定

    GazeFS: Target-Centered Gaze-Trajectory Forecasting and Stabilization from Gaze-Head History

    [https://arxiv.org/abs/2609.03868](https://arxiv.org/abs/2609.03868)

    提出GazeFS，仅利用视线-头部历史即可在线预测目标中心的视线轨迹并实现稳定，在推理时无需目标信息，为目标中心的眼动交互提供了新的视线校正方法。

    

    目标中心的视线交互不仅仅是抑制帧间波动：目标获取会产生与任务对齐的视线-头部动态变化，而视线轨迹可能保留一个持续的目标相对残余方向。我们将视线校正表述为在线的目标中心视线轨迹预测与稳定问题，并提出了GazeFS，它将可变长度的视线-头部历史映射到下一个目标中心方向以及短时程的Search/Focus（搜索/聚焦）估计，且在推理时无需目标信息。在来自30名参与者的7,960个获取片段上，在质量控制、起始点排除和时长匹配条件下，Search-Focus差异保持稳定。历史窗口相比当前端点改进了阶段解码，但显式的任务进度仍然是一个强控制因素。在30名参与者、五折分组折外协议、三个随机种子设置下，相对于原始保持状态，Focus片段中的降低……（摘要在此处截断）

    arXiv:2609.03868v1 Announce Type: cross  Abstract: Target-centered gaze interaction requires more than suppressing frame-to-frame fluctuations: target acquisition produces task-aligned changes in gaze-head dynamics, while a gaze trace may retain a persistent target-relative residual direction. We formulate gaze correction as online target-centered gaze-trajectory forecasting and stabilization and introduce GazeFS, which maps a variable-length gaze-head history to the next target-center direction and a short-horizon Search/Focus estimate without target information at inference. Across 7,960 acquisition episodes from 30 participants, Search-Focus differences remain stable under quality control, onset exclusion, and duration matching. History windows improve phase decoding over the current endpoint, but explicit task progress remains a strong control. Under the 30-participant, five-fold grouped out-of-fold protocol across three seeds, the reductions relative to raw hold in Focus episode b
    
[^63]: 适应不断演进的需求：面向零售供应链运营的智能体人工智能

    Adapting to Evolving Requirements: Agentic AI for Retail Supply Chain Operations

    [https://arxiv.org/abs/2609.03860](https://arxiv.org/abs/2609.03860)

    该论文提出了一种图约束的智能体AI框架，通过联合选择干预路径与模块级变更、并以下游KPI验证候选方案，使大语言模型能够驱动零售供应链运营适应不断演进的业务需求。

    

    零售供应链运营依赖于相互耦合的决策模块，这些模块必须随着需求的演进而进行调整。大语言模型为此类任务提供了自然语言接口，但现有方法主要聚焦于单个优化模型。将其扩展到异构决策管道颇具挑战性，因为一项需求可能存在多条具有不同下游影响的干预路径。我们将需求驱动的适应性调整形式化为干预路径与可容许的模块级变更的联合选择问题，并提出了一种图约束的智能体框架：领域智能体暴露可容许的重构接口，中央处理器则在有界的干预路径上进行搜索。候选方案通过下游KPI进行验证与比较。在与一家大型零售合作伙伴的合作中，我们基于从从业者访谈中提炼的100项仓库需求进行评估，并使用GPT、Qwen和DeepSeek作为基础大语言模型。相对于直接……

    arXiv:2609.03860v1 Announce Type: new  Abstract: Retail supply chain operations rely on coupled decision modules that must adapt as requirements evolve. LLMs offer a natural-language interface for this task, but existing methods primarily focus on individual optimization models. Extending them to heterogeneous decision pipelines is challenging because a requirement may admit multiple intervention paths with different downstream effects. We formulate requirement-driven adaptation as the joint selection of an intervention route and an admissible module-level change, and propose a graph-constrained agentic framework in which domain agents expose admissible reformulation interfaces and a central processor searches over bounded intervention paths. Candidates are validated and compared using downstream KPIs. In collaboration with a large retail partner, we evaluate 100 warehouse requirements elicited from practitioner interviews, with GPT, Qwen, and DeepSeek as base LLMs. Relative to direct 
    
[^64]: 语义贝叶斯世界模型

    Semantic Bayesian World Models

    [https://arxiv.org/abs/2609.03834](https://arxiv.org/abs/2609.03834)

    该论文提出语义贝叶斯世界模型（SBWM）的愿景，将知识图谱从静态的事实数据库转变为共享且演化的概率信念体系——由本体公理约束先验、贝叶斯条件化更新信念、动作干预世界——从而弥合概率推理的基础模型与确定性知识图谱之间的鸿沟，实现统一的推理架构。

    

    知识图谱以清晰的断言描述现实，而如今消费这些知识的系统——基础模型与自主智能体——却原生地以概率方式进行推理。我们认为，这种不匹配正是语言模型与知识图谱的融合至今仍停留在数据供给管道层面、而未能形成统一推理架构的原因。我们展望了语义贝叶斯世界模型（SBWMs）：一个不再将世界描述为事实数据库，而是描述为知识图谱之上共享且持续演化的信念网络的Web，其中本体公理约束先验，观测通过贝叶斯条件化更新信念，动作则对世界进行干预。我们深入探讨了智能体能从这样的模型中获得什么：一个正在判断门口人影是快递员还是窃贼的家庭安防智能体、一个通过逻辑蕴含而非字符串频率进行聚合的精算估计、一个语言模型往往无法可靠完成的规划任务，以及对数量的估计……

    arXiv:2609.03834v1 Announce Type: new  Abstract: Knowledge graphs describe reality in crisp assertions, while the systems now consuming them, foundation models and autonomous agents, reason natively in probabilities. We argue that this mismatch is why the integration of language models and knowledge graphs remains a data-feeding pipeline rather than a unified reasoning architecture. We envision Semantic Bayesian World Models (SBWMs): a Web that describes the world not as a database of facts but as a shared, evolving fabric of beliefs over knowledge graphs, where ontological axioms constrain priors, observations update beliefs by Bayesian conditioning, and actions intervene upon the world. We work through what an agent gains from such a model: a home-security agent deciding whether the figure at the gate is a courier or a burglar, an actuarial estimate aggregated by entailment rather than by string frequency, a planning task that language models reliably fail, and the estimation of quan
    
[^65]: 相位信息对小样本细粒度图像分类的影响

    The impact of phase information for few-shot fine-grained image classification

    [https://arxiv.org/abs/2609.03829](https://arxiv.org/abs/2609.03829)

    该论文提出即插即用的幅相集成（API）模块和PSF-Net网络，通过自适应融合基于相位的空间与频率信息来增强小样本细粒度图像分类，在五个公开数据集上超越了现有最先进方法。

    

    小样本细粒度图像分类（FSFGIC）旨在利用有限的标注样本对相似图像进行分类。这项工作强调了相位信息在捕捉图像内部结构关系中的关键但尚未被充分利用的作用。本研究提出了一种新颖的即插即用幅相集成（API）模块，该模块有效地结合了局部和全局的频率幅度与相位信息，以获得更全面的特征描述符。此外，还提出了一种名为PSF-Net的专用网络，该网络可自适应地融合基于相位的空间与频率信息用于FSFGIC。所设计的PSF-Net可以轻松集成到标准的情景训练架构中，实现从零开始的端到端训练。在五个公开数据集上的大量实验表明，该方法优于现有的最先进基准方法。

    arXiv:2609.03829v1 Announce Type: cross  Abstract: Few-shot fine-grained image classification (FSFGIC) aims to classify similar images with limited labeled examples. This work highlights the critical yet underutilized role of phase information in capturing structural relationships within an image. This study introduces a novel plug-and-play amplitude-phase integration (API) module that effectively combines local and global frequency amplitude and phase information for obtaining more comprehensive feature descriptors. Additionally, a dedicated network, named PSF-Net, is proposed that adaptively fuses phase-based spatial and frequency information for FSFGIS. The designed PSF-Net can be easily integrated into standard episodic training architectures for end-to-end training from scratch. Extensive experiments on five public datasets demonstrate that the method outperforms existing state-of-the-art benchmarks.
    
[^66]: 证人解释异常

    Witnesses Explain Anomalies

    [https://arxiv.org/abs/2609.03826](https://arxiv.org/abs/2609.03826)

    WAND是一种天生可解释的无监督表格数据异常检测器，它通过单位球面上的方向进行评分，而标记异常点的“证人”方向本身就构成了该点的逐特征解释，无需借助SHAP或LIME等事后解释方法。

    

    无监督异常检测对未标记且受污染样本中的每个数据点进行一次性评分，而且如今越来越多地还需要解释某个点为何被标记。然而，主流的检测器只给出分数，却不说明是哪些特征驱动了这一分数，解释只能通过事后附加SHAP或LIME来实现，这些方法需要对每个点重新查询检测器数千次，且只能近似检测结果。我们提出了WAND，一种可解释性源于设计本身的无监督表格数据异常检测器。WAND的计算围绕单位球面上的方向来组织，通过每个点的投影超出亚高斯极值基线的距离来为其评分。我们方法的原创性在于：标记某个异常点的“证人”方向作为特征空间中的向量，本身就是该点的解释——这种逐特征归因无需在评分之外付出任何额外成本即可获得，并且由于分数是可微分的，还可以通过梯度恢复得到。评分计算关于样本量是线性的，并且……

    arXiv:2609.03826v1 Announce Type: cross  Abstract: Unsupervised anomaly detection scores each point of an unlabelled, contaminated sample in a single pass, and increasingly must also explain why a point is flagged. Yet the dominant detectors give a score with no account of which features drive it, and explanations are bolted on post-hoc with SHAP or LIME, which re-query the detector thousands of times per point and only approximate it. We introduce WAND, an unsupervised tabular anomaly detector that is explainable by design. WAND organises its computation around directions on the unit sphere, scoring each point by how far its projection escapes a sub-Gaussian extreme-value baseline. The originality of our approach is that the witness directions that flag a point, being vectors in feature space, are its explanation, a per-feature attribution obtained at no cost over scoring and, since the score is differentiable, recoverable by gradients. Scoring is linear in the sample size, and a prob
    
[^67]: CauseCollab：用于异构协同感知的因果统一且模态无关网络

    CauseCollab: Causal Unified and Modality-Agnostic Network for Heterogeneous Collaborative Perception

    [https://arxiv.org/abs/2609.03818](https://arxiv.org/abs/2609.03818)

    提出CauseCollab，一种基于因果视角的模态无关网络，通过因果度量学习将语义因素与模态特定的统计混杂因素解耦，从而解决异构协同感知中的语义不一致和误差累积问题。

    

    协同感知通过多智能体间的信息共享来增强环境理解，但其在真实场景中的性能受到异构传感器模态和模型架构的制约。近期基于协议的两阶段方法通过将异构特征映射到共享协议空间来缓解这一问题；然而，独立训练的模态特定转换器往往会产生模态特定的伪协议分布，导致语义不一致和误差累积，这一问题在模态差异较大的场景中尤为突出。为解决这一问题，我们提出了CauseCollab，一个因果统一且模态无关的网络。CauseCollab从因果视角构建协议空间中的表示学习，通过因果度量学习显式地将语义因素与模态特定的统计混杂因素解耦。同时，CauseCollab采用……（摘要内容不完整，原文在此处截断）

    arXiv:2609.03818v1 Announce Type: new  Abstract: Collaborative perception enhances environment understanding through multi-agent information sharing, but its performance in real-world scenarios is constrained by heterogeneous sensor modalities and model architectures. Recent protocol-based two-stage methods alleviate this problem by mapping heterogeneous features into a shared protocol space; however, independently trained modality-specific converters often generate modality-specific pseudo-protocol distributions, leading to semantic inconsistency and error accumulation, which is particularly pronounced in scenarios with large modality discrepancies. To address this issue, we propose CauseCollab, a causal unified and modality-agnostic network. CauseCollab formulates representation learning in the protocol space from a causal perspective, explicitly disentangling semantic factors from modality-specific statistical confounders via causal metric learning. Meanwhile, CauseCollab adopts con
    
[^68]: 免费暂停标记

    Free Pause Tokens

    [https://arxiv.org/abs/2609.03807](https://arxiv.org/abs/2609.03807)

    提出免费暂停标记，通过权重共享主干上的并行预测流为模型提供额外思考计算，在不增加上下文长度、KV缓存和推理延迟的情况下，仅以1.14倍训练计算量的代价提升下一个词元预测性能。

    

    免费暂停标记为语言模型提供额外的计算量来形成每个下一个词元预测（如同暂停标记或思考标记的作用），但它不是在序列中添加额外标记，而是通过权重共享主干上的并行预测流来承载这些计算。在实际应用中，它使一个10亿参数模型的下一个词元预测提升了2-3厘奈特。由于暂停是搭载在已有位置上而非新增位置，因此使用它是免费的：在推理时它不增加上下文长度、不增加KV缓存，且几乎不产生延迟，推理浮点运算量的增长通常无关紧要，因为它并非吞吐量的主动瓶颈。唯一的主要成本在于训练阶段：与优化过的预训练流程相比，额外的训练计算量可低至1.14倍，同时保留了大部分收益。其结果是在与标准下一个词元训练的Transformer等浮点运算、等参数量和等词元数量条件下取得的性能改进。

    arXiv:2609.03807v1 Announce Type: cross  Abstract: A free pause token gives a language model extra compute to form each next-token prediction (as a pause, or thinking, token does) but carries that compute in a parallel prediction stream over a weight-shared backbone rather than as an extra token in the sequence. It improves next-token prediction by 2-3 centinats in practice on a 1B parameter model. Because the pause rides an existing position instead of adding one, it is free to use: at inference it adds no context length, no KV cache, and essentially no latency with the growth in inference flops typically irrelevant as it is not the active bottleneck on throughput. The only primary cost is in training, where additional training compute versus an optimized pretraining pipeline is reduced to as low as x1.14 while preserving most of the benefits. The result is an isoflop, isoparameter, and isotoken improvement over standard next token trained transformers.
    
[^69]: SVG-Score：面向文本到SVG生成的符合人类判断的评估

    SVG-Score: Human-Aligned Evaluation of Text-to-SVG Generation

    [https://arxiv.org/abs/2609.03806](https://arxiv.org/abs/2609.03806)

    提出了SVG-Score评估框架，通过人工标注的语义对齐数据集解决了现有自然图像评估指标（如CLIPScore）无法有效识别SVG生成中颜色、数量、空间关系等错误的评估难题。

    

    随着生成模型在表现力和可控性方面的不断提升，可缩放矢量图形（SVG）生成正受到越来越多的关注。然而，由于缺乏领域特定的评估协议，该领域的进展受到阻碍：目前的做法依赖于为自然图像设计的指标，最典型的是CLIPScore，该指标从未在矢量图形上训练过，与人类判断仅部分一致。我们提出了SVG-Score，一个面向文本到SVG生成、与人类判断高度一致的评估框架。通过受控的标题和图像扰动实验，我们首先表明基于CLIP的分数对SVG生成器实际产生的错误（如错误的颜色、数量和空间关系）几乎没有反应，而现成的视觉-语言模型（VLM）评判者虽然更敏感，但在不同错误类型和SVG风格之间的反应并不均匀。随后，我们引入了一个用于“语义对齐”的人工标注数据集，用于衡量生成的SVG对文本描述的忠实程度……

    arXiv:2609.03806v1 Announce Type: new  Abstract: Scalable Vector Graphics (SVG) generation is attracting increasing attention as generative models improve in expressiveness and controllability. Progress, however, is held back by the lack of domain-specific evaluation protocols: current practice relies on metrics designed for natural images, most notably CLIPScore, which was never trained on vector graphics and aligns only partially with human judgment. We introduce \textbf{\ours}, a human-aligned evaluation framework for text-to-SVG generation. Through controlled caption and image perturbations, we first show that CLIP-based scores barely react to the errors SVG generators actually make, such as wrong colors, counts, and spatial relations, and that off-the-shelf Vision-Language Model (VLM) judges, while more sensitive, respond unevenly across error types and SVG styles. We then introduce a human-annotated dataset for \textit{Semantic Alignment}, measuring how faithfully a generated SVG
    
[^70]: 管理模型，而不仅是数据：创意AI中的存储、流通与学习

    Govern the Model, Not Only the Data: Storage, Circulation, and Learning in Creative AI

    [https://arxiv.org/abs/2609.03800](https://arxiv.org/abs/2609.03800)

    本文提出联邦化本身并非榨取式AI的解药，指出创作者治理止步于学习层面，主张创意社区应将治理从数据和存储、流通扩展到模型本身。

    

    联邦学习日益被呈现为一种保护隐私的进步：个人数据保留在设备上，仅共享模型更新。它借用了联邦社交网络的术语，却颠倒了其逻辑——将计算分布出去，而最终产生的模型却归召集训练的一方所有。我们认为，联邦化本身并不能治愈榨取式AI，因为结果取决于谁治理数据和模型，以及谁对塑造它们的实践拥有能动性。我们描述了创意社区可以持有其作品的三个层面：存储、流通和学习。通过考察艺术家治理的信托、合作社和同意基础设施，我们表明创作者治理在存储和流通层面得以确立，却在学习层面中断：贡献者可以同意参与训练，却对由此产生的模型或其联邦化几乎没有发言权。我们描绘了由此开启的研究空间，将技术机会与…

    arXiv:2609.03800v1 Announce Type: new  Abstract: Federated learning is increasingly presented as a privacy-preserving advance: personal data remain on the device, and only model updates are shared. It borrows the vocabulary of the federated social web, yet inverts its logic, distributing computation while the resulting model stays with whoever convened the training. We argue that federation is not in itself a remedy for extractive AI, because outcomes depend on who governs the data and the model and who has agency over the practices that shape them. We describe three layers at which a creative community can hold its work: storage, circulation, and learning. Examining artist-governed trusts, cooperatives, and consent infrastructures, we show that creator governance is established at storage and circulation but stops at learning: contributors can consent to training, yet have little say over the resulting model or its federation. We map the research space this opens, pairing technical op
    
[^71]: Transfiver：通过共享可编辑状态实现人机协同推理

    Transfiver: Human-AI Co-Inference through a Shared Editable State

    [https://arxiv.org/abs/2609.03797](https://arxiv.org/abs/2609.03797)

    Transfiver 提出了一种人机协同推理架构，将交互信息维护在模型与人类共同更新的单一共享持久状态中，通过隐式流式更新与显式定向编辑两种机制，使人类的修正能够直接改变后续计算所读取的状态。

    

    长期的人机交互之所以困难，是因为引导推理的信息由模型隐式更新，用户无法直接查看或控制。我们提出了“透明交互式、可验证、可编辑表示框架”（Transfiver），这是一种通过共享可编辑状态实现人机协同推理的架构。其核心思想是：与交互相关的信息被维护在单一持久状态 $(S_t)$ 中，该状态由模型和人类共同更新。Transfiver 区分了两种状态演化模式：在隐式流式更新中，模型解释正在进行的交互，并决定新信息是修改现有的状态项还是创建新的状态项；在显式定向编辑中，人类可以检查并修改指定的状态项。两者都作用于同一个底层状态，因此人类的修正会改变后续计算所读取的状态，而不是简单地另加一条信息。

    arXiv:2609.03797v1 Announce Type: new  Abstract: Long-term human-AI interaction is difficult because the information that guides inference is updated implicitly by the model and is not directly inspectable or controllable by the user. We introduce the TRANSparent Framework for Interactive, Verifiable, Editable Representation (Transfiver), an architecture for human-AI co-inference through a shared editable state. Its central idea is that interaction-specific information is maintained in a single persistent state $(S_t)$ that both the model and the human update.   Transfiver distinguishes two modes of state evolution. In an implicit stream update, the model interprets ongoing interaction and decides whether new information revises an existing state item or creates a new one. In an explicit directed edit, a human inspects and modifies an addressed item. Both act on the same underlying state, so a human correction changes the state that subsequent computation reads, rather than adding anot
    
[^72]: LLaDA-Image：用完全开放的训练配方构建强大的图像生成器

    LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes

    [https://arxiv.org/abs/2609.03796](https://arxiv.org/abs/2609.03796)

    LLaDA-Image通过仅图像预训练、RMSNorm与Muon优化器等完全开放的训练配方，将从零训练的6B扩散Transformer与冻结的LLaDA2.0-Mini视觉-语言理解模块结合成统一模型，实现高保真图像生成与细粒度编辑指令的精准遵循，其蒸馏版本LLaDA-Image-Turbo仅需2-4步即可快速推理。

    

    我们提出了LLaDA-Image，这是一个统一框架，它将从零开始训练的60亿参数扩散Transformer（DiT）与基于LLaDA2.0-Mini扩散语言模型骨干构建的冻结视觉-语言理解模块相结合。我们没有从一开始就大量依赖配对的图文数据，而是首先通过仅使用图像的预训练和中期训练来构建强大的视觉生成先验。整个生成流程包含2.2亿个样本，其中9800万为真实图像。为了实现高效且可扩展的优化，我们在整个DiT中使用无参数的RMSNorm，并配合Muon优化器。最终得到的统一模型能够生成高度逼真的图像，同时准确遵循细粒度的编辑指令。我们进一步将LLaDA-Image蒸馏为LLaDA-Image-Turbo，使其能够在2-4个采样步骤内完成快速推理。在Qwen-Image-Bench上，LLaDA-Image在英文和中文赛道上分别取得了53.53和53.38的总分。

    arXiv:2609.03796v1 Announce Type: cross  Abstract: We introduce LLaDA-Image, a unified framework that pairs a 6B Diffusion Transformer (DiT) trained from scratch with a frozen vision-language understanding module built on the LLaDA2.0-Mini diffusion language model backbone. Instead of relying heavily on paired image-text data from the beginning, we first build a strong visual generative prior through image-only pre-training and mid-training. The generation pipeline comprises 220M samples, 98 of which are real images. For efficient and scalable optimization, we use parameter-free RMSNorm throughout the DiT together with the Muon optimizer. The resulting unified model produces highly photorealistic images while accurately following fine-grained editing instructions. We further distill LLaDA-Image into LLaDA-Image-Turbo, enabling fast inference in 2-4 sampling steps. On Qwen-Image-Bench, LLaDA-Image achieves overall scores of 53.53 and 53.38 on the English and Chinese tracks, respectively
    
[^73]: DNative-Twin：用于可重构智能体决策的决策图与数字孪生

    DNative-Twin: Decision Graphs and Digital Twins for Reconstructable Agentic Decisions

    [https://arxiv.org/abs/2609.03787](https://arxiv.org/abs/2609.03787)

    本文提出DNative-Twin，一种图原生的数字孪生框架，将智能体决策记录为类型化轨迹并在声明条件下重执行决策机制，从而实现智能体决策的可重构与审计，同时实验揭示图结构虽能定位已表示的变化，却无法推断未观察到的工具状态的影响。

    

    人工智能智能体越来越多地收集证据、调用工具、施加约束，并产生人们或软件可能付诸行动的决策。然而，仅凭最终输出无法显示是哪些证据、工具状态、规则、授权或行动路径产生了该决策。我们提出 DNative-Twin，这是一个图原生的数字孪生系统，它将已提交的智能体决策记录为类型化轨迹，并在声明的条件下重新执行其决策机制。该图将智能体观察到的状态、其所遵循的路径以及最终行动背后的授权关联起来。数字孪生同步这些信息，单独重放决策机制，并在受控变化下进行比较。我们使用三个公开流程日志和受控重放套件，在企业决策流程中实例化了该框架。实验识别出一个特定的失效模式：图结构能够定位已被表示的变化，但无法确定未被观察到的工具状态所产生的后果。

    arXiv:2609.03787v1 Announce Type: new  Abstract: AI agents increasingly gather evidence, invoke tools, apply constraints, and produce decisions that people or software may commit to action. A final output alone cannot show which evidence, tool state, rule, authorization, or action path produced it. We present DNative-Twin, a graph-native digital twin that records a committed agentic decision as a typed trajectory and re-executes its decision mechanism under declared conditions. The graph links the state observed by the agent, the path it followed, and the authority behind the resulting action. The twin synchronizes this information, replays the mechanism in isolation, and compares it under controlled changes. We instantiate the framework in enterprise decision processes using three public process logs and controlled replay suites. The experiments identify a specific failure: graph structure localizes represented changes but cannot determine the consequence of an unobserved tool state. 
    
[^74]: IndicSafeEval：多语言说服性越狱攻击下大语言模型的安全稳健性

    IndicSafeEval: Safety Robustness of Large Language Models under Multilingual Persuasive Jailbreak Attacks

    [https://arxiv.org/abs/2609.03781](https://arxiv.org/abs/2609.03781)

    该论文提出了IndicSafeEval框架，通过四种印度语言、十个安全类别和六种说服策略构建7,200条对抗性提示，系统评估并揭示了大语言模型在面对多语言说服性越狱攻击时安全表现存在显著差异。

    

    大语言模型在多语言环境中的应用日益广泛，但其安全性评估主要仍以英语为主。这限制了我们对对齐失效在低资源和文化多样性语言中如何表现的理解。我们提出了IndicSafeEval，一个针对印度语言的说服性越狱攻击评估框架。该基准将十个安全关键内容类别与六种类人说服策略相结合，涵盖印地语、孟加拉语、马拉地语和旁遮普语四种不同的印度语言，共生成7,200条对抗性提示。我们对多个开源大语言模型进行了系统性的黑盒评估，以检验其安全行为如何随语言、说服策略和风险类别的变化而变化。我们的分析表明，模型并非在所有语言和提示风格下都表现得同样安全，相反，安全性能在很大程度上取决于所使用的语言以及提示的构造方式。

    arXiv:2609.03781v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in multilingual settings, yet their safety is still evaluated primarily in English. This limits our understanding of how alignment failures manifest in low-resource and culturally diverse languages. We introduce IndicSafeEval, a persuasion-based jailbreak evaluation framework for Indian languages. Our benchmark combines ten safety critical content categories with six human-like persuasive strategies across four different Indian languages, such as Hindi, Bengali, Marathi and Punjabi, resulting in 7,200 adversarial prompts. We conduct a systematic black-box evaluation of several open-source LLMs to examine how their safety behaviour varies across languages, persuasion strategies, and risk categories. Our analysis shows that the model does not behave equally safely across all languages and prompt styles. Instead, safety performance depends strongly on both the languages used and the way a
    
[^75]: 重新思考面向安全关键具身系统的世界模型

    Rethinking World Models for Safety-Critical Embodied Systems

    [https://arxiv.org/abs/2609.03774](https://arxiv.org/abs/2609.03774)

    该论文指出现有世界模型存在似然与风险、预测与干预、有限时域预测与累积后果三重结构性错位，并提出以决策为中心的风险知情世界模型（RIWM），通过整合决策相关表征、反事实推理、安全关键情景记忆与运行时安全保障来支撑安全关键具身系统。

    

    世界模型已经从紧凑的潜在动力学模型发展为具身环境的生成式、可控且可交互的模拟器。然而，高预测似然和高视觉保真度并不一定能保证模型保留了安全决策所需的证据。本视角论文指出了当前世界建模中的三种结构性错位：似然与风险之间的错位、预测与干预之间的错位，以及有限时域预测与累积后果之间的错位。我们提出风险知情世界模型（RIWM），作为安全关键具身系统的一种以决策为中心的研究方向。RIWM 将世界建模围绕后果、干预、认知不确定性和可恢复性来组织，并整合了四种相互依赖的能力：决策相关表征、反事实推理、安全关键的情景记忆以及运行时安全保障。它区分了物理层面、社会层面和操作层面的……

    arXiv:2609.03774v1 Announce Type: new  Abstract: World models have progressed from compact latent dynamics to generative, controllable, and interactive simulators of embodied environments. However, high predictive likelihood and visual fidelity do not necessarily ensure that a model preserves the evidence required for safe decision-making. This perspective identifies three structural mismatches in current world modeling: likelihood versus risk, prediction versus intervention, and finite-horizon prediction versus accumulated consequences. We propose the Risk-Informed World Model (RIWM) as a decision-centric research direction for safety-critical embodied systems. RIWM organizes world modeling around consequences, intervention, epistemic uncertainty, and recoverability, and integrates four interdependent capabilities: decision-relevant representation, counterfactual reasoning, safety-critical episodic memory, and runtime safety assurance. It distinguishes physical, social, and operationa
    
[^76]: ENEAS：面向自适应分割的嵌入引导神经集成方法

    ENEAS: Embedding-guided Neural Ensemble for Adaptive Segmentation

    [https://arxiv.org/abs/2609.03756](https://arxiv.org/abs/2609.03756)

    ENEAS提出了一种统一的文本提示方法，通过语义验证层同时实现唯一实例的精确跟踪与高质量分割以及开放概念的语义发现，解决了SAM 3等分割模型的时间幻觉、空间碎片化和语义误分类问题。

    

    我们提出了ENEAS，这是一种统一的、可文本提示的实例跟踪与语义发现方法。可文本提示的分割模型，包括SAM 3等最新基础模型，仍然存在时间幻觉、空间碎片化和语义误分类问题：当目标物体离开视野时，它们无法报告目标缺失；在极端特写镜头下，它们会分割局部纹理而非完整物体；并且它们优先考虑视觉特征而忽视本体现实，导致雕像、绘画或倒影等视觉上相似的人造物被误分割为目标实体。ENEAS通过单一方法实现两种功能：对唯一实例进行精确跟踪和高质量分割，以及对文本查询所命名的每个实例进行开放概念发现，并通过语义验证层加以解析。在跟踪方面，我们将此前仅限于点交互的几何鲁棒的SeC架构扩展为……（摘要在此处被截断）

    arXiv:2609.03756v1 Announce Type: cross  Abstract: We present ENEAS, a unified, text-promptable method for instance tracking and semantic discovery. Text-promptable segmentation models, including the latest foundation models such as SAM 3, still suffer from temporal hallucinations, spatial fragmentation, and semantic misclassification: they fail to report target absence when an object leaves the field of view, segment local textures instead of the complete object during extreme close-ups, and prioritize visual features over ontological reality, so that visually similar artifacts such as statues, paintings, or reflections are segmented as target entities.   ENEAS works two ways from a single method: precise tracking and high-quality segmentation of a unique instance, and open-concept discovery of every instance a text query names, resolved by a semantic verification layer. For tracking, we extend the geometrically robust SeC architecture, previously limited to point interactions, with a
    
[^77]: SimSkill：一个自主精通交通仿真的终身学习AI智能体

    SimSkill: A Lifelong Learning AI Agent for Autonomous Mastery of Traffic Simulation

    [https://arxiv.org/abs/2609.03753](https://arxiv.org/abs/2609.03753)

    SimSkill是一个基于SUMO交通仿真器的自进化终身学习AI智能体，通过自主识别能力差距、生成并验证任务、将经验整合到三种记忆系统中，在不更新底层模型的情况下将经验证的任务完成率提升最多25个百分点。

    

    随着大型语言模型（LLM）能力的不断增强，AI系统的长期价值不仅取决于解决单个请求的能力，还取决于能否将经验和积累的知识转化为持久、可复用的能力。我们提出了SimSkill，一个围绕城市交通仿真平台（SUMO）构建的自进化智能体。SimSkill能够识别能力差距、生成并解决基于环境的任务、通过“行动-批评”（action-critic）循环验证解决方案，并将经验整合到情景记忆、程序性记忆和语义记忆中，而无需更新底层模型。通过自主探索，它构建了一个覆盖交通仿真完整工作流程的可复用知识库。我们在两个留出测试基准上，使用三个底层LLM以及独立的基于产物的验证方法对SimSkill进行了评估。SimSkill将经验证的任务完成率提升了最多25个百分点，消融实验表明各模块的贡献具有互补性……

    arXiv:2609.03753v1 Announce Type: new  Abstract: As large language models (LLMs) become increasingly capable, the long-term value of AI systems depends not only on solving individual requests, but also on transforming experience and accumulated knowledge into durable, reusable competence. We introduce SimSkill, a self-evolving agent built around the Simulation of Urban MObility (SUMO) traffic simulator. SimSkill identifies capability gaps, generates and solves environment-grounded tasks, verifies solutions through an action--critic loop, and consolidates experience into episodic, procedural, and semantic memory without updating the backbone model. Through autonomous exploration, it builds a reusable library spanning the traffic-simulation workflow. We evaluate SimSkill on two held-out benchmarks with three backbone LLMs and independent artifact-based verification. SimSkill improves verified completion by up to 25 percentage points, while ablations show complementary contributions from 
    
[^78]: 超越BLEU：重新定义手语翻译基准的案例

    Beyond BLEU: A Case for Redefining Sign Language Translation Benchmarks

    [https://arxiv.org/abs/2609.03734](https://arxiv.org/abs/2609.03734)

    本文证明BLEU-4的提升并不等同于更强的手语理解能力，并提出了一种基于开放权重LLM问答协议的新型评估方法，该方法更符合人类排名、对改写更不敏感且对训练-测试重叠更加鲁棒。

    

    BLEU-4是评估手语翻译（SLT）的标准指标，但口语语言指标可能无法充分反映手语能力。SLT的多模态、低资源环境使得模型能够利用虚假相关性和口语先验，而不是学习更强的手语表示。在本文中，我们评估了六个SLT模型在Phoenix-2014T和CSL-Daily数据集上的时空理解与BLEU-4之间的关系，表明BLEU-4的提升本身并不能证明更好的手语理解能力。这项工作引入了一种受语言学习评估启发的替代方法，使用开放权重LLM问答协议来衡量显著内容的保留程度。该协议与人类排名更加一致，并且在改写不变性方面比BLEU-4高出六到七倍。应用于SLT时，该协议针对内容传递，对训练-测试重叠更加鲁棒，并且给出……

    arXiv:2609.03734v1 Announce Type: cross  Abstract: BLEU-4 is the standard metric for evaluating sign language translation (SLT), but spoken-language metrics may not adequately reflect sign language proficiency. The multimodal, low-resource context of SLT allows models to exploit spurious correlations and spoken-language priors, rather than learning stronger sign representations. In this paper, we evaluate the relationship between spatio-temporal understanding and BLEU-4 across six SLT models on Phoenix-2014T and CSL-Daily, showing that gains in BLEU-4 are not on their own evidence of better sign language understanding. This work introduces an alternative inspired by language-learning assessment, using an open-weight-LLM QA protocol that measures salient content preservation. It aligns more closely with human rankings and is six to seven times more paraphrase-invariant than BLEU-4. Applied to SLT, this protocol targets content transfer, is more robust to train-test overlap, and gives a 
    
[^79]: 主动服务智能体：统一决策框架、方法与评估

    Proactive Service Agents: A Unified Decision Framework, Methods, and Evaluation

    [https://arxiv.org/abs/2609.03727](https://arxiv.org/abs/2609.03727)

    本文提出一个统一的主动服务智能体决策框架，将“何时、以何种内容与方式介入服务”形式化为受授权与风险约束的部分可观测序列决策过程，并沿状态与需求估计到干预决策的流水线系统梳理了现有方法与评估体系。

    

    大型语言模型智能体能够进行规划、调用工具和修改外部状态，然而大多数系统仍将用户的显式指令作为固定的起点。主动服务将决策环节向前推移：智能体必须从不完整的环境信号和用户信号中推断服务机会，在保持沉默、提问、协助与行动之间做出选择，并权衡打扰、误解、越权和隐私等成本。本综述给出了以“主动性”为核心的操作性定义，并将该问题形式化为受授权与风险约束的部分可观测序列决策过程。该形式化将时机、内容与传达方式统一表示在一个结构化动作之中，同时显式刻画了等待的期权价值、提问的决策价值以及反馈所引起的状态变化。在此基础上，我们沿一条统一的决策流水线（状态与需求估计、干预门控……）组织现有方法并进行评估。

    arXiv:2609.03727v1 Announce Type: new  Abstract: Large language model agents can plan, invoke tools, and modify external states, yet most systems still take an explicit user instruction as a fixed starting point. Proactive service moves the decision upstream: an agent must infer service opportunities from incomplete environmental and user signals, choose among remaining silent, asking, assisting, and acting, and account for interruption, misunderstanding, overreach, and privacy costs. This survey gives an operational definition centered on initiative and formulates the problem as a partially observable sequential decision process constrained by authorization and risk. The formulation represents timing, content, and delivery within one structured action, while making explicit the option value of waiting, the decision value of questions, and feedback-induced state changes. On this basis, we organize existing methods along one decision pipeline (state and need estimation, intervention gat
    
[^80]: 大语言模型能否从源代码提交中提取架构设计决策？——一项初步探索性研究

    Can LLMs Extract Architectural Design Decisions from Source Code Commits? - A Preliminary Exploratory Study

    [https://arxiv.org/abs/2609.03721](https://arxiv.org/abs/2609.03721)

    该初步探索性研究表明，四种大语言模型在零样本和少样本提示下能够有效从源代码提交中提取架构设计决策，所有模型的BERT-F1均超过0.81，且少样本提示能进一步提升提取效果。

    

    背景：架构设计决策（ADD）捕捉了软件系统结构与演进背后的原理依据，但很少被明确记录，往往隐藏在源代码提交中。恢复这些决策对于架构知识管理（AKM）非常重要。问题：由于ADD具有隐式且非结构化的特性，从提交中提取ADD极具挑战性。大语言模型（LLM）在理解代码和文本方面已展现出强大能力，但其在该任务上的有效性仍未得到充分探索。研究：我们开展了一项初步研究，使用四个大语言模型（Gemini 3 Pro、DeepSeek R1、Kimi K2、Qwen3），采用零样本和少样本提示方法，对来自开源项目的30条开发者编写的ADD进行提取测试。我们使用ROUGE-L、BLEU、METEOR和BERTScore对输出进行评分，并由一位作者对Gemini的输出进行人工评审。结果：所有模型的BERT-F1得分均超过0.81，少样本提示提升了对齐效果（Gemini的BERT-F1：0.[摘要截断于此]）

    arXiv:2609.03721v1 Announce Type: cross  Abstract: Context: Architectural Design Decisions (ADDs) capture the rationale behind the structure and evolution of software systems but are rarely documented explicitly, and are often hidden inside source code commits. Recovering them is important for Architectural Knowledge Management (AKM). Problem: Extracting ADDs from commits is challenging due to their implicit and unstructured nature. Large Language Models (LLMs) have shown strong capabilities in understanding code and text, yet their effectiveness for this task remains underexplored. Study: We present a preliminary study using four LLMs (Gemini 3 Pro, DeepSeek R1, Kimi K2, Qwen3) with zeroshot and fewshot prompting on 30 developer-written ADDs from open-source projects. We score outputs with ROUGE-L, BLEU, METEOR, and BERTScore, and one author manually reviews the Gemini outputs. Results: All models reach a BERT-F1 above 0.81, and fewshot prompting improves alignment (Gemini BERT-F1: 0.
    
[^81]: 用于数据中心能源优化的人工智能

    Artificial Intelligence for Energy Optimization in Data Centers

    [https://arxiv.org/abs/2609.03716](https://arxiv.org/abs/2609.03716)

    该论文通过系统性文献编码揭示数据中心AI节能研究存在“控制与负载割裂”、过度依赖仿真验证、忽略水资源与隐含碳排放、各类方法节能效果无法区分排名等十大差距，并提出将控制策略与工作负载需求相耦合的CLEAR-DC研究框架。

    

    数据中心越来越多地由人工智能进行优化，同时也越来越多地由人工智能承载负载。现有文献将这两个问题视为互不相关的独立问题：控制研究将工作负载建模为外生到达过程，而可持续性研究则将基础设施建模为一个固定的乘数因子。我们通过一个有记录的检索流程筛选了大约194篇论文，对其中63篇进行了编码分析，并报告编码结果所揭示的现状。在28项以控制为导向的主要研究中，18项仅在仿真环境中得到验证，仅5项达到了物理硬件或生产设施层面；没有任何一项研究考虑了水资源消耗，也没有任何一项考虑了隐含碳排放。四个技术系列所报告的节能区间几乎完全重叠，这意味着该领域目前无法对自身的各类方法进行优劣排序。我们对十个反复出现的研究差距进行了后果严重性和可解决性的评分，并提出了CLEAR-DC框架，该框架将控制策略分支与工作负载需求分支通过（摘要原文在此处截断）

    arXiv:2609.03716v1 Announce Type: new  Abstract: Data centers are increasingly optimized by artificial intelligence and, at the same time, increasingly loaded by it. The literature treats these as two unrelated problems: control studies model workload as an exogenous arrival process, while sustainability studies model infrastructure as a fixed multiplier. We screen roughly 194 papers retrieved through a documented protocol, code 63 of them, and report what the coding shows. Of 28 primary control-oriented studies, 18 are validated in simulation alone and 5 reach physical hardware or a production facility; none account for water withdrawal, and none account for embodied carbon. Reported savings intervals across four technique families overlap almost completely, which means the field cannot presently rank its own methods. Ten recurring gaps are scored for consequence and tractability, and we set out CLEAR-DC, a framework coupling a control-policy branch to a workload-demand branch through
    
[^82]: 基于整数规划与约束生成的反事实路径规划

    Counterfactual Routing Using Integer Programming with Constraint Generation

    [https://arxiv.org/abs/2609.03707](https://arxiv.org/abs/2609.03707)

    本文提出了一种基于整数规划与约束生成迭代添加约束的反事实路径规划方法，为最短路径问题提供精确的反事实解释，在IJCAI 2025竞赛中以平均9.0秒的运行时间成为所有参赛方案中求解最快的算法。

    

    我们提出了参加 IJCAI 2025 “反事实路径规划竞赛”（CRC 25）的方案。该竞赛的目标是为最短路径问题寻找反事实解释，即确定对路网进行何种最小改动才能使用户所选的路线成为最优路线。这可以支持诸如“如果X道路不是自行车道，你建议的路线确实会是最优的”这样的解释。我们的解决方案将该问题建模为整数规划，通过迭代地添加约束直到找到精确解。在留出测试实例的最终评估中，我们的方法在解的质量上排名第四，并且在每个实例上都最快获得了解，平均运行时间为9.0秒，而次快的参赛方案需要118.8秒。

    arXiv:2609.03707v1 Announce Type: new  Abstract: We present our submission to the IJCAI 2025 'Counterfactual Routing Competition' (CRC 25). The goal of the competition is to find counterfactual explanations for the shortest path problem. This requires deciding what the minimal changes to a road network would make a route chosen by the user the optimal route. This enables explanations such as "Your suggested route would indeed have been optimal, if road X were not a bicycle path." Our solution models the problem as an integer program, iteratively incorporating constraints until an exact solution is found. In the final evaluation on held-out test instances, our method ranked fourth in solution quality and obtained its solution fastest on every instance, with an average runtime of 9.0 seconds compared to 118.8 seconds for the next-fastest submission.
    
[^83]: 小型Transformer中对比式代码表示学习的合成语义监督：一项实证研究

    Synthetic Semantic Supervision for Contrastive Code Representation Learning in Small Transformers: An Empirical Study

    [https://arxiv.org/abs/2609.03702](https://arxiv.org/abs/2609.03702)

    该实证研究表明，用合成生成的自然语言描述作为语义监督对小代码编码器进行对比预训练，可以在八个任务中的五个上显著优于同等推理规模的预训练基线，从而摆脱了对人工文档字符串和执行轨迹的依赖。

    

    通用代码嵌入（code embeddings）为代码搜索、分类和检索等工具提供了支持。面向代码的紧凑型Transformer编码器通常依赖于人工编写的文档字符串（劳动密集且不一致）或挖掘得到的结构化信号（如执行轨迹，局限于特定场景且收集成本高昂）。我们实证研究了一种替代方案：使用合成生成的、强调代码功能与意图的自然语言描述，对小规模编码器进行对比预训练，在训练阶段将描述与代码在双编码器框架中配对，并在推理阶段将其丢弃。我们在C、C++和Java三种语言上的八个检索、分类和生成任务中，将该方法与基于预训练的基线、通用大语言模型（LLM）以及专用嵌入模型进行了基准比较。结果显示，合成语义监督在八个任务中的五个上取得了相对相同推理时规模预训练基线的统计显著提升，并在另外两个任务上达到同等水平；一旦……

    arXiv:2609.03702v1 Announce Type: new  Abstract: General-purpose code embeddings power tools for code search, classification, and retrieval. Compact transformer encoders for code typically rely on either human-written docstrings (labor-intensive and inconsistent) or mined structural signals such as execution traces (setting-specific and costly to collect). We empirically study an alternative: contrastive pretraining of small encoders with synthetically generated natural-language descriptions emphasizing code functionality and intent, paired with code in a dual-encoder framework at training and discarded at inference. We benchmark this approach against pretraining-based baselines, generalist LLMs, and embedding-specific models on eight retrieval, classification, and generation tasks across C, C++, and Java. Synthetic semantic supervision yields statistically significant gains over pretraining baselines of the same inference-time size on five of eight tasks, with parity on two more; once
    
[^84]: 对称性与因果性：超越独立同分布数据的因果效应识别

    Symmetries and Causality: Causal Effect Identification Beyond IID Data

    [https://arxiv.org/abs/2609.03697](https://arxiv.org/abs/2609.03697)

    该论文提出了一种基于使因果机制保持不变的数据对称性的因果推理形式化数学语言，能够统一并大幅扩展因果效应识别的范围，使其超越独立同分布数据的限制，处理do-算子或软干预无法刻画的复杂因果查询。

    

    在自然科学中，对称性和因果关系无处不在。然而对于复杂的机器学习任务，如强化学习中的世界建模，它们似乎难以被有效利用。我们提出了一种基于数据对称性的统计系统形式化描述方法，其中对称性使因果机制保持不变。由此产生了一种抽象、简洁且通用的因果推理数学语言。本文提供了模型和查询的形式化描述以构建这一语言，并给出了在该形式体系内从数据中进行数学严谨识别所需的形式化基础设施和策略。该方法再现并符合关于独立同分布（IID）数据以及实验与非实验数据迁移的标准理论结果。但其主要目的是统一并大幅扩展因果推理的范围——超越独立同分布数据的限制，并处理do-算子或软干预等传统方法无法刻画的复杂因果查询。

    arXiv:2609.03697v1 Announce Type: cross  Abstract: In the natural sciences, symmetries and cause-effect relationships are ubiquitous. Yet for complex machine-learning tasks, like world-modeling in reinforcement learning, they appear difficult to harness. We propose a formal description of statistical systems based on symmetries in data leaving causal mechanisms invariant. The result is an abstract, simple and general mathematical language for causal reasoning. This paper provides formal descriptions of models and queries, setting up this language, and the formal infrastructure and strategies for their mathematically rigorous identification from data within this formalism. This approach reproduces and matches standard theoretical results on IID data and transport of experimental and non-experimental data. But its main purpose is to unify and substantially extend the scope of causal reasoning, in going beyond IID data and in approaching complex causal queries not captured by do- or soft-
    
[^85]: 离线多智能体强化学习中基于序列模型的分布外泛化

    Out-of-Distribution Generalisation with Sequence Models in Offline Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.03667](https://arxiv.org/abs/2609.03667)

    该研究通过系统性分析发现，扩展任务多样性而非数据集规模是离线多智能体强化学习实现零样本任务泛化的关键因素，其提出的多任务序列建模方法在留出测试任务上相比单任务模型平均提升3.2倍。

    

    在离线多智能体强化学习（MARL）中，泛化到未见过的任务仍然是一个根本性挑战。在这项工作中，我们对离线设置下的零样本任务泛化进行了系统性分析，并针对任务多样性、数据集规模和网络容量之间的扩展行为展开了广泛的实证研究。为支持这项研究，我们扩展了离线序列建模架构，使其能够处理多任务的观测与动作空间，以及跨任务数量可变的智能体。我们的主要发现是：扩展任务多样性——而非单纯增加数据集规模——是实现稳健零样本迁移的主导因素。通过在四个具有挑战性的环境上进行大规模实验，我们证明了我们的多任务方法在留出测试任务上相比单任务模型实现了平均3.2倍的提升，并始终优于强大的基线方法。

    arXiv:2609.03667v1 Announce Type: cross  Abstract: Generalising to unseen tasks remains a fundamental challenge in offline multi-agent reinforcement learning (MARL). In this work, we present a principled analysis of zero-shot task generalisation in the offline setting and conduct an extensive empirical investigation into the scaling behaviour governing task diversity, dataset size, and network capacity. To facilitate this study, we extend offline sequence modelling architectures to handle multi-task observation and action spaces alongside variable agent counts across tasks. Our primary finding is that scaling task diversity---rather than sheer dataset size is the dominant factor in achieving robust zero-shot transfer. Through large-scale experiments across four challenging environments (Connector, RWARE, SMAX, and LBF), we demonstrate that our multi-task approach achieves a mean improvement of 3.2x on held-out test tasks compared to single-task models and consistently outperforms stron
    
[^86]: 面向RhythmFormer远程光电容积脉搏波技术的可解释人工智能跨数据集迁移与可靠性研究

    Cross-Dataset Transfer and Reliability of Explainable Artificial Intelligence for RhythmFormer Remote Photoplethysmography

    [https://arxiv.org/abs/2609.03663](https://arxiv.org/abs/2609.03663)

    该研究首次定量评估了RhythmFormer远程光电容积脉搏波模型各类可解释性方法的忠实性，发现Beyond Intuition方法表现最佳且其解释能够跨数据集迁移并追踪模型性能。

    

    背景：远程光电容积脉搏波技术从面部视频估计心血管脉搏信号，而以往对其解释主要依赖于对热力图的目视检查，缺乏关于模型从何处读取脉搏信号的定量证据。本研究对这些解释进行了量化，并探讨了此类解释能否在不同数据集之间迁移以及能否追踪模型性能。方法：研究者在NCKU-rPPG数据集上训练了八个针对特定条件的RhythmFormer模型，该数据集涵盖三种光照水平、说话、头部旋转和骑行等条件，每个5.12秒的片段估计一次心率，并同时复现了UBFC-rPPG数据集上的模型。研究通过皮肤覆盖率和显著性引导的忠实性系数两个指标，评估了原始注意力、注意力滚动、注意力流和Beyond Intuition四种可解释性方法。结果：Beyond Intuition在两个数据集上均排名最高，在静态3级光照下中位覆盖率为0.789、SaCo为0.837，而在UBFC-rPPG上分别为0.826和0.917；排名较低的方法则表现不一致。在同一参与者同一条件下……

    arXiv:2609.03663v1 Announce Type: cross  Abstract: Background. Remote photoplethysmography estimates the cardiovascular pulse from facial video, and its explanations have rested on inspecting heatmaps rather than on quantitative evidence about where a model reads it. We quantified the explanations and asked whether such explanations transfer between datasets and track model performance. Method. We trained eight condition-specific RhythmFormer models on NCKU-rPPG, recorded under three illumination levels, speaking, rotation, and cycling, estimated one heart rate per 5.12-second clip, and set them beside a UBFC-rPPG reproduction. Raw attention, rollout, attention flow, and Beyond Intuition were assessed by skin coverage and the Salience-guided Faithfulness Coefficient (SaCo). Results. Beyond Intuition ranked highest on both datasets, at median coverage 0.789 and SaCo 0.837 on Static level 3 against 0.826 and 0.917 on UBFC-rPPG; lower ranks differed. Within one participant of one conditio
    
[^87]: 局部更新，全局学习（LUGL）：与非增量学习器进行博弈

    Local Updates, Global Learning (LUGL): Playing Games with non-incremental Learners

    [https://arxiv.org/abs/2609.03660](https://arxiv.org/abs/2609.03660)

    提出LUGL框架，通过将数据收集与模型拟合解耦，使梯度提升树等非增量学习器能够克服分布偏移问题，成功应用于自我博弈强化学习场景。

    

    神经网络（NN）在强化学习（RL）中的主导地位部分归因于其增量学习能力，这种能力天然契合自我博弈训练的在线、非平稳特性。然而，以LightGBM为代表的梯度提升树被广泛认为是监督学习中处理表格数据的最先进方法，在准确性和效率上往往优于神经网络。博弈状态本质上就是表格形式的——离散的动作、类别的卡牌身份、结构化的棋盘位置——这使其成为基于树的方法的理想候选。我们提出了LUGL（局部更新，全局学习）框架，该框架将数据收集与模型拟合解耦，使梯度提升树（GBT）等非增量学习器能够在强化学习环境中运行，否则它们会因分布偏移而失败。LUGL在两个阶段之间交替进行：局部更新阶段，智能体进行自我博弈游戏并累积表格更新（Q值、V值、策略或遗憾值）……

    arXiv:2609.03660v1 Announce Type: cross  Abstract: The dominance of Neural Networks (NNs) in RL is partially due to their incremental learning capability, which naturally suits the online, non-stationary nature of self-play training. However, gradient-boosted trees like LightGBM are widely recognised as the state of the art for tabular data in supervised learning, often outperforming NNs in accuracy and efficiency. Game states are inherently tabular---discrete actions, categorical card identities, structured board positions---which makes them an ideal candidate for tree-based methods. We introduce LUGL (Local Updates, Global Learning), a framework that decouples data collection from model fitting, enabling non-incremental learners such as GBTs to operate in RL settings where they would otherwise fail due to distributional shift. LUGL alternates between a local updates phase, where the agent plays self-play games and accumulates tabular updates (Q-values, V-values, policies, or regret v
    
[^88]: 增强金融问答：一个基于银行财务报表的新型基准数据集

    Enhancing Financial Question Answering: A Novel Benchmark Dataset of Banks' financial statements

    [https://arxiv.org/abs/2609.03654](https://arxiv.org/abs/2609.03654)

    该论文提出了首个针对跨机构银行财务报表检索的金融问答基准数据集 FinRAG-QA，包含999个从业者整理的问题和24家欧美大型银行的209份超长报告，并系统评估了多阶段 RAG 流水线中各组件的贡献。

    

    由于银行财务报表的复杂性、篇幅冗长、专业术语的使用，以及不同司法管辖区和机构之间文本与数值内容的异质性，对其进行比较分析对自动问答系统构成了重大挑战。我们提出了 FinRAG-QA，一个新颖的金融问答基准数据集，包含999个由从业者精心整理的问题，涵盖10个标准化指标，基于24家欧美主要银行2019年至2023年的209份年度报告和第三支柱（Pillar 3）报告。与以往主要聚焦于美国申报文件和单一机构分析的金融问答基准不同，FinRAG-QA 针对的是跨机构检索场景，文档平均长度达19.8万字，超过了任何现有的金融问答资源。在该基准上，我们评估了一个多阶段 RAG 流水线，并分离量化了每个组件的贡献。上下文分块增强结合检索器……

    arXiv:2609.03654v1 Announce Type: cross  Abstract: The comparative analysis of banks' financial statements poses significant challenges for automated question answering systems due to their complexity, substantial length, technical language, and inhomogeneity of both textual and numerical content across different jurisdictions and institutions. We introduce FinRAG-QA, a novel benchmark dataset for financial question answering, which comprises 999 practitioner-curated questions on 10 standardised indicators, grounded in 209 annual and Pillar 3 reports from 24 major European and U.S. banks spanning 2019-2023. Unlike prior financial QA benchmarks, which centre on U.S. filings and single-institution analysis, FinRAG-QA targets cross-institutional retrieval over documents averaging 198k words, longer than any existing financial QA resource. On this benchmark we evaluate a multi-stage RAG pipeline and isolate the contribution of each component. Contextual chunk enrichment combined with a ret
    
[^89]: 药物毒性预测中提示工程的分析

    Analysis of Prompt Engineering for Drug Toxicity Prediction

    [https://arxiv.org/abs/2609.03635](https://arxiv.org/abs/2609.03635)

    本文提出了一种分析方法，研究提示词措辞（提示工程）对大语言模型在药物毒性预测中的影响及其重要性。

    

    英国的临床试验成本可高达130万英镑，药物失败率约为90%，而毒性是药物失败的一个主要因素。测试过程既耗时又成本高昂。近年来，人工智能被越来越多地探索用于辅助预测药物毒性，其中大语言模型（LLMs）得到了广泛应用。然而，当提示词发生微小变化时，LLMs可能表现出相当大的差异，这引发了人们对其对提示工程敏感性的担忧。提示工程用于优化提供给LLM的提示词，以生成期望的输出。本文提出了一种分析药物毒性预测中提示工程的方法。本文旨在研究提示词措辞对于药物毒性预测的重要性。研究者通过提示LLMs来识别预测药物毒性时具有重要意义的化学性质，并构建提示词进行研究。

    arXiv:2609.03635v1 Announce Type: new  Abstract: Clinical trials in the UK can cost up to {\pounds}1.3 million, with approximately 90% drug failure rate. Toxicity is a major contributing factor in drug failure. Testing is time and cost intensive. In recent years, the use of artificial intelligence has been increasingly explored to aid in the prediction of drug toxicity, with extensive use of large language models (LLMs). However, LLMs can show considerable variation when minor changes are made to prompts, which raises concerns about their sensitivity to prompt engineering. Prompt engineering is used to optimise a prompt given to an LLM to generate the desired output. This paper proposes a method to analyse prompt engineering for drug toxicity prediction. The aim of the paper is to investigate the importance of prompt phrasing for drug toxicity prediction. LLMs were prompted to identify chemical properties of significance when predicting drug toxicity. Prompts were constructed to invest
    
[^90]: 《</think> 并不能停止推理：虚假思维链终止现象分析》

    </think> Doesn't Stop Reasoning: Analysis of Spurious CoT Termination

    [https://arxiv.org/abs/2609.03633](https://arxiv.org/abs/2609.03633)

    本文发现提前退出方法中注入的思考结束标记</think>并不总能真正终止推理，模型会在回答阶段继续产生类似推理的“虚假CoT终止”行为，且该延续片段的长度与提前退出节省的推理token数成正比，其根源可能是模型对注入EoT的注意力不足。

    

    思维链推理提升了大型推理模型（LRMs）在复杂任务上的表现，但常常产生冗长且冗余的推理轨迹。近期的免训练提前退出方法通过选择一个中间停止点来缩短这些轨迹。我们研究了其中一种策略：在该点注入思考结束标记（EoT，</think>）以触发从推理到回答阶段的转换，并发现注入的EoT并不总是能引发干净的回答阶段。在模型重新生成另一个EoT之前，回答阶段的生成可能会继续进行，且该重新生成的EoT之前的片段长度会随提前退出所节省的推理token数量而增长，并表现出持续的推理行为。我们将这种现象称为“虚假CoT终止”，即类似推理的生成延续到了回答阶段。我们假设对注入的EoT关注不足是导致虚假CoT终止的原因之一，并通过Exit-token注意力偏……（原文在此处截断）

    arXiv:2609.03633v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) reasoning improves large reasoning models (LRMs) on complex tasks but often produces long, redundant traces. Recent training-free early-exit methods shorten these traces by choosing an intermediate point to stop reasoning. We study one such strategy that injects an end-of-think token (EoT, ) at this point to trigger the reasoning-to-answering transition, and find that the injected EoT does not always induce a clean answering phase. Answering-phase generation can continue before the model regenerates another EoT, with the span preceding this regenerated EoT scaling with the reasoning tokens saved by early exit and exhibiting continued reasoning behavior. We call this spurious CoT termination, where reasoning-like generation continues into the answering phase. We hypothesize that insufficient attention to the injected EoT contributes to spurious CoT termination and probe this hypothesis with Exit-token Attention Bi
    
[^91]: EraseSAE：基于稀疏自编码器实现文本到视频扩散模型中的精准概念擦除

    EraseSAE: Surgical Concept Erasure in Text-to-Video Diffusion Models via Sparse Autoencoders

    [https://arxiv.org/abs/2609.03629](https://arxiv.org/abs/2609.03629)

    提出了EraseSAE框架，利用稀疏自编码器在单语义特征层面通过“分解-归因-擦除”流程，实现基于DiT的文本到视频扩散模型中的精准概念擦除，在移除不良语义的同时保持生成质量。

    

    文本到视频（T2V）扩散模型的最新进展展示了卓越的生成能力，但其对筛选不严格的训练数据的依赖引发了紧迫的安全和版权问题。概念擦除通过从预训练模型中移除不需要的语义、同时保留其余概念，提供了一种有原则的补救方法。然而，现有方法通常在粗粒度上进行操作，与概念表示的细粒度、分布式特性不相匹配，导致擦除不完整或生成质量下降。我们认为，精准的擦除从根本上需要在单语义特征层面进行干预，即每个单元编码一个单一的可解释概念。为此，我们提出了EraseSAE，这是一个新颖的框架，利用稀疏自编码器，通过一个有原则的“分解-归因-擦除”流程，在基于DiT的文本到视频扩散模型中实现精准概念擦除。我们首先引入了……

    arXiv:2609.03629v1 Announce Type: cross  Abstract: Recent advances in text-to-video (T2V) diffusion models have demonstrated remarkable generative capabilities, yet their reliance on loosely curated training data raises pressing safety and copyright concerns. Concept erasure offers a principled remedy by removing unwanted semantics from pretrained models while preserving remaining concepts. However, existing approaches typically operate at a coarse granularity misaligned with the fine-grained, distributed nature of concept representations, leading to incomplete removal or degraded generation quality. We argue that surgical erasure fundamentally requires intervention at the level of monosemantic features, where each unit encodes a single interpretable concept. To this end, we propose EraseSAE, a novel framework that leverages sparse autoencoders to achieve surgical concept erasure in DiT-based T2V diffusion models via a principled decompose-attribute-erase pipeline. We first introduce t
    
[^92]: 基于自回归语音先验的语音增强测试时自适应

    Test-time adaptation for speech enhancement with an autoregressive speech prior

    [https://arxiv.org/abs/2609.03622](https://arxiv.org/abs/2609.03622)

    提出了一种基于自回归干净语音先验的测试时自适应方法，通过最小化增强语音分布与干净语音先验之间的KL散度，在无需标注目标数据的情况下有效提升语音增强模型在噪声不匹配条件下的语音质量。

    

    测试时自适应（TTA）为在声学条件不匹配的情况下改进语音增强模型提供了一个有前景的方向，且无需访问带标注的目标数据。在本工作中，我们提出了一种单条语音的TTA方法，该方法使用在从神经音频编解码器提取的干净语音潜在表示上训练的自回归先验，对预训练的语音增强模型进行正则化。自适应过程通过最小化增强语音分布与干净语音先验之间的Kullback-Leibler散度来执行。在多个含噪语音数据集上的实验表明，语音质量获得了一致的提升，尤其是在训练与测试噪声不匹配的条件下。代码和音频示例已在线提供。

    arXiv:2609.03622v1 Announce Type: cross  Abstract: Test-time adaptation (TTA) offers a promising direction for improving speech enhancement models under mismatched acoustic conditions, without requiring access to labeled target data. In this work, we propose a single-utterance TTA method that regularizes a pretrained speech enhancement model using an autoregressive prior trained on clean speech latent representations extracted from a neural audio codec. Adaptation is performed by minimizing the Kullback-Leibler divergence between the enhanced speech distribution and the clean speech prior. Experiments across multiple noisy speech datasets show consistent improvements in speech quality, particularly under training-testing noise mismatch conditions. Code and audio examples are available online.
    
[^93]: 物理实验室的可计算表示实现可验证的工作流

    A computable representation of the physical laboratory enables verifiable workflows

    [https://arxiv.org/abs/2609.03621](https://arxiv.org/abs/2609.03621)

    本文通过类型化研究对象、能力绑定操作和组合式工作流代数建立了物理实验室的可计算表示，并将其与可执行的函数技能绑定，从而在机器人实验室中生成可验证的工作流并在调度前检验操作前置条件与实验室约束。

    

    使科学可计算需要对科学知识以及用于检验科学论断的物理世界进行表示。本文通过类型化研究对象、能力绑定操作和可组合的工作流代数，建立了物理实验室的可计算表示。它为机器可读知识提供了物理世界的对应物，将工作流表达为运行在不断演化的实验室状态之上的程序，并具有显式的依赖关系、决策、迭代和并发。该表示在模块化的智能体机器人实验室中得以实现，方法是将形式化操作绑定到可执行的函数技能上。针对多样的科学意图，系统生成了相对于能力的工作流，同时通过有状态的模拟来传播对象变换，并在调度前验证操作的前置条件和实验室约束。所提出的表示及其工程框架共同建立了一个基……

    arXiv:2609.03621v1 Announce Type: new  Abstract: Making science computable requires representations of both scientific knowledge and the physical world in which scientific claims are tested. A computable representation of the physical laboratory is established through typed research objects, capability-bound operations and a compositional workflow algebra. It provides the physical-world counterpart to machine-readable knowledge, expressing workflows as programs over evolving laboratory states with explicit dependencies, decisions, iteration and concurrency. The representation was implemented in a modular agentic robotic laboratory by binding formal operations to executable Function Skills. For diverse scientific intents, capability-relative workflows were generated, while stateful simulation propagated object transformations and verified operation preconditions and laboratory constraints before dispatch. The proposed representation and its engineering framework jointly establish a gene
    
[^94]: ToolDF：面向混合真实性音频深度伪造检测的工具集成推理

    ToolDF: Tool-Integrated Reasoning for Mixed-Authenticity Audio Deepfake Detection

    [https://arxiv.org/abs/2609.03620](https://arxiv.org/abs/2609.03620)

    ToolDF提出以音频大语言模型为编排器的工具集成推理框架，通过自适应声源分离与领域专家路由，实现混合真实性音频深度伪造的检测、定位与可解释判定，并构建了相应的基准数据集。

    

    音频深度伪造检测通常被建模为对单域音频的片段级二分类任务。然而，现实世界中被篡改的音频可能呈现混合真实性，即真实与伪造的线索在时间过渡、声源重叠或两者兼有的情况下共存。这种设定不仅要求检测被篡改的音频，还要求定位为决策提供依据的相关成分。我们提出ToolDF，一个面向混合真实性音频深度伪造检测的工具集成推理框架。ToolDF采用音频大语言模型作为编排器，并通过有监督的工具使用轨迹进行训练。它能够自适应地分析音频场景，选择性地执行声源分离，将各个成分路由至领域特定的专家模型，并聚合其证据以形成可解释的判定结论。我们还进一步构建了一个覆盖时间过渡、声学重叠及混合场景的混合真实性音频深度伪造检测基准。实验……

    arXiv:2609.03620v1 Announce Type: cross  Abstract: Audio deepfake detection is commonly formulated as clip-level binary classification of single-domain audio. However, real-world manipulated audio can exhibit mixed authenticity, where genuine and manipulated cues coexist across temporal transitions, overlapping sources, or both. This setting requires not only detecting manipulated audio but also localizing the components that provide evidence for the decision. We propose ToolDF, a tool-integrated reasoning framework for mixed-authenticity audio deepfake detection. ToolDF employs an audio large language model as an orchestrator trained with supervised tool-use trajectories. It adaptively analyzes the audio scene, selectively performs source separation, routes components to domain-specific experts, and aggregates their evidence into an interpretable verdict. We further introduce a mixed-authenticity ADD benchmark covering temporal transitions, acoustic overlaps, and hybrid mixtures. Expe
    
[^95]: 记忆与重加权：利用经验记忆与置信度估计增强多智能体辩论

    Remember and Reweight: Enhancing Multi-Agent Debate with Experience Memory and Confidence Estimation

    [https://arxiv.org/abs/2609.03619](https://arxiv.org/abs/2609.03619)

    该论文提出R²-MAD框架，通过为多智能体辩论引入经验记忆机制，利用辩论状态感知的检索策略动态校准概念先验，并结合置信度估计对回答进行重加权，有效缓解了多数智能体收敛于错误答案时“共享误解”被放大的关键缺陷。

    

    多智能体辩论通过让多个智能体在讨论中迭代地改进自身回答，从而提升大语言模型的推理能力。然而，MAD存在一个被称为“共享误解”的关键脆弱性：当大多数智能体最初收敛于某个错误答案时，辩论过程往往会放大而非纠正该错误。现有方法主要针对同伴偏差问题，但未能解决智能体固有的有偏概念先验。为缓解这一系统性弱点，我们提出了R²-MAD（Remember and Reweight for Multi-Agent Debate），这是一个为智能体配备从过往辩论中积累的经验记忆的框架。R²-MAD通过两个互补机制对两种失败模式进行干预：一种辩论状态感知的检索策略，根据当前共识水平检索相关历史证据，动态校准概念先验；随后将这些检索到的证据与置信度估计相结合，对智能体的回答进行重加权，从而在共识偏离正确答案时有效抑制错误观点的放大。

    arXiv:2609.03619v1 Announce Type: cross  Abstract: Multi-agent debate (MAD) improves the reasoning capabilities of large language models by having multiple agents iteratively refine their responses through discussion. However, MAD suffers from a critical vulnerability known as shared misconception: when a majority of agents initially converge on an incorrect answer, the debate process tends to amplify rather than correct the error. Existing methods primarily address peer skew but leave the agents' inherently biased concept priors unaddressed. To mitigate this systematic weakness, we propose R$^2$-MAD (Remember and Reweight for Multi-Agent Debate), a framework that equips agents with an experience memory accumulated from past debates. R$^2$-MAD intervenes on both failure modes through two complementary mechanisms: A debate-state-aware retrieval policy dynamically calibrates the concept prior by retrieving relevant historical evidence based on the current consensus level. Then these retr
    
[^96]: FailBench：视觉语言模型判断机器人任务成功的可靠性如何？

    FailBench: How Reliable are VLMs at Judging Robot Task Success?

    [https://arxiv.org/abs/2609.03611](https://arxiv.org/abs/2609.03611)

    FailBench基准测试揭示了视觉语言模型在判断机器人任务成功方面可靠性有限——最佳模型平衡准确率仅0.77，专门微调的模型反而不如通用模型，且在接触密集的装配任务上表现接近随机水平。

    

    视觉语言模型（VLM）越来越多地被用于评估机器人操作结果，但现有基准测试在跨领域泛化能力方面提供的证据有限。我们提出了FailBench，一个用于机器人失败检测的基准，包含来自14个公开来源（12个真实世界、2个仿真环境）的2,197次操作尝试。在FailBench中，75%的失败是自然发生的，其中六个真实世界来源来自非失败检测数据集。我们对13个基于VLM的检测器进行了评估，发现最佳模型的平均平衡准确率仅为0.77。值得注意的是，针对失败检测进行微调的模型始终表现不如通用VLM及其各自的预训练基线。性能在很大程度上取决于所需的视觉证据：当结果取决于可观察到的物体运动时，模型表现接近饱和，但在接触密集的装配任务上则退化为接近随机水平（平衡准确率<0.60）。误差分析揭示了一种系统性偏差……

    arXiv:2609.03611v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are increasingly used to evaluate robot manipulation outcomes, but existing benchmarks offer limited evidence of cross-domain generalization. We introduce FailBench, a benchmark for robot failure detection comprising 2,197 manipulation attempts across 14 public sources (12 real-world, 2 simulated). In FailBench, 75% of failures occur naturally, and six real-world sources come from non-failure-detection datasets. Evaluating 13 VLM-based detectors, we find the best model achieves only 0.77 mean balanced accuracy. Notably, models fine-tuned for failure detection consistently underperform general-purpose VLMs and their own pretrained baselines. Performance depends heavily on required visual evidence: models approach saturation when outcomes depend on observable object motion, but degrade to near-chance (<0.60 balanced accuracy) on contact-intensive assembly tasks. Error analysis reveals a systematic bias towar
    
[^97]: 论模型压缩与测试时自适应之间的交互作用

    On the Interaction Between Model Compression and Test-Time Adaptation

    [https://arxiv.org/abs/2609.03604](https://arxiv.org/abs/2609.03604)

    本文首次系统研究了模型压缩与测试时自适应（TTA）之间的交互作用，发现压缩模型虽在有监督自适应下保持高精度，但其TTA性能随压缩程度增加而显著下降，其根源在于表示多样性的降低和限制可恢复性的结构约束。

    

    在真实环境中部署的深度神经网络必须兼具高效性与适应性，这需要模型压缩和测试时自适应（TTA）。虽然这两者都已被单独充分研究，但它们之间的交互作用仍鲜为人知。我们系统地分析了结构化压缩如何影响模型在分布偏移下的自适应能力。我们在CIFAR-10-C和ImageNet-C数据集上使用ResNet-18和ViT-Base模型，评估了多种压缩方法与标准TTA技术的组合。我们引入了一个诊断框架，用于检验表示表达能力和自适应子空间的兼容性。我们的结果揭示了一个持续的差距：尽管压缩模型在有监督自适应下保持了较高的准确率，但其TTA性能随着压缩程度的增加而显著下降。我们证明这源于表示多样性的降低以及限制可恢复性的结构约束。这些效应强烈依赖于压缩方法。

    arXiv:2609.03604v1 Announce Type: cross  Abstract: Deep neural networks deployed in the wild must be both efficient and adaptable, requiring model compression and test-time adaptation (TTA). While both are well studied in isolation, their interaction remains poorly understood. We systematically analyze how structured compression affects a model's ability to adapt under distribution shift. Using ResNet-18 and ViT-Base on CIFAR-10-C and ImageNet-C, we evaluate multiple compression methods combined with standard TTA techniques. We introduce a diagnostic framework that examines representational expressivity and adaptation subspace compatibility. Our results reveal a consistent gap: although compressed models retain high accuracy under supervised adaptation, their TTA performance degrades significantly with increasing compression. We show that this stems from reduced representational diversity and structural constraints that limit recoverability. These effects strongly depend on the compres
    
[^98]: 合成数据能将泰文OCR带向多远？

    How Far Can Synthetic Data Take Thai OCR?

    [https://arxiv.org/abs/2609.03595](https://arxiv.org/abs/2609.03595)

    通过受控文档重建流水线解耦合成数据“真实性”中的各项因素，发现字体多样性、二维结构和真实手写字形是合成数据迁移到真实泰文OCR的关键，并据此构建了无需真实OCR标签的泰文OCR模型Wayu-Paxa-OCR-Zero。

    

    我们研究了是什么因素使得合成OCR监督信号能够迁移到真实的泰文文档上，并利用所得见解构建了Wayu-Paxa-OCR-Zero——一个无需来自真实泰文文档页面的OCR标签即可完成适配的泰文OCR模型。合成数据能够以大规模提供精确标签，但“真实性”这一概念混淆了源域、页面上下文、字体排印、空间结构和字形变化等多个因素。我们通过一个受控的文档重建流水线将这些因素解耦，并在页面级和裁剪级两种训练方式下，在印刷体和手写泰文文档上对每种变体进行评估。结果表明：非文本上下文几乎没有一致的影响，而字体多样性、二维结构以及真实手写字形则能改善迁移效果；此外，源域匹配的效果依赖于训练粒度——在页面级训练下，域内重建可接近真实印刷体监督的效果（中位字符错误率1.82% 对比 1.31%），但在域外重建方面则表现不佳。

    arXiv:2609.03595v1 Announce Type: cross  Abstract: We investigate what makes synthetic OCR supervision transfer to real Thai documents and use the resulting insights to build Wayu-Paxa-OCR-Zero, a Thai OCR model adapted without OCR labels from real Thai document pages. Synthetic data provide exact labels at scale, but "realism" conflates source domain, page context, typography, spatial structure, and glyph variation. We disentangle these factors with a controlled document-reconstruction pipeline and evaluate each variant under page- and crop-level training on printed and handwritten Thai documents. Non-text context has little consistent effect, whereas typeface diversity, two-dimensional structure, and real handwriting glyphs improve transfer; moreover, source-domain matching depends on training granularity, with in-domain reconstruction approaching real printed supervision under page-level training (1.82% versus 1.31% median character error rate) but underperforming out-of-domain reco
    
[^99]: LevelSyn：基于层级异步图神经网络的物理感知逻辑综合

    LevelSyn: Physical-Aware Logic Synthesis via Level-Asynchronous Graph Neural Networks

    [https://arxiv.org/abs/2609.03594](https://arxiv.org/abs/2609.03594)

    LevelSyn提出了一种物理感知的逻辑综合框架，利用层级异步图神经网络捕捉与-非图（AIG）的结构和方向语义以预测高保真度门坐标，并结合线长驱动的优化引擎，从而缓解综合与物理设计脱节带来的PPA退化和设计收敛周期过长的问题。

    

    随着集成电路技术进入纳米尺度，逻辑综合与物理设计之间的传统脱节导致了显著的PPA（功耗、性能和面积）退化以及漫长的设计收敛周期。传统逻辑综合依赖于非物理的线负载模型（WLM），而近期基于谱方法的布局预测器往往忽略了网表中固有的层次化逻辑深度和信号流，导致空间估算的保真度较低。为了弥合这一差距，我们提出了LevelSyn，一个新颖的物理感知逻辑综合框架，它将层次化表示学习与线长驱动的优化引擎相结合。其核心在于，LevelSyn利用层级异步图神经网络（GNN），通过捕捉与-非图（AIG）的结构和方向语义，来预测高保真度的门坐标。为了处理工业规模的设计，一个层级对齐的…

    arXiv:2609.03594v1 Announce Type: cross  Abstract: As integrated circuit technology scales into the nanometer regime, the traditional disconnect between logic synthesis and physical design has led to significant PPA (Power, Performance, and Area) degradation and prolonged design closure cycles. Traditional logic synthesis relies on non-physical Wire Load Models (WLMs), while recent spectral-based placement predictors often neglect the inherent hierarchical logic depth and signal flow of netlists, which leads to low-fidelity spatial estimations. To bridge this gap, we propose LevelSyn, a novel physical-aware logic synthesis framework that integrates hierarchical representation learning with a wirelength-driven optimization engine. At its core, LevelSyn leverages a level-asynchronous Graph Neural Network (GNN) to predict high-fidelity gate coordinates by capturing the structural and directional semantics of And-Inverter Graphs (AIGs). To handle industrial-scale designs, a level-aligned s
    
[^100]: 从先验引导启发式到可部署智能体：加速面向截止期限约束网络控制的示范驱动强化学习

    From Prior-Guided Heuristics to Deployable Agents: Accelerating Demonstration-Driven Reinforcement Learning for Deadline-Constrained Network Control

    [https://arxiv.org/abs/2609.03590](https://arxiv.org/abs/2609.03590)

    该论文提出一个面向部署的网络控制框架，通过截止期限感知的有效拥塞度度量与均匀路径分组启发式来引导示范驱动强化学习，加速智能体训练并过滤不可行流量，从而在动态异构网络中满足严格的端到端峰值时延保证。

    

    在动态异构网络上及时传输时延敏感信息，对于NextG交互式应用至关重要，然而如何提供严格的端到端（E2E）峰值时延保证仍是一个悬而未决的挑战。两个障碍限制了基于学习的网络控制在这一场景中的应用：传统的基于容量的路由度量虽然对一般流量管理非常有效，但并非为捕捉流量紧急性而设计；而从零开始训练的深度强化学习（DRL）控制器存在样本效率低、训练时间长以及早期探索波动性大的问题。本文提出了一个面向部署的网络控制框架，同时解决了这两个障碍。首先，我们提出了有效拥塞度，这是一个截止期限感知的度量族，能够按数据包紧急程度量化接口拥塞情况并主动过滤不可行流量，并配合均匀路径分组（UPG）分布启发式方法……

    arXiv:2609.03590v1 Announce Type: cross  Abstract: Timely delivery of delay-sensitive information over dynamic, heterogeneous networks is essential for NextG interactive applications, yet providing strict End-to-End (E2E) peak latency guarantees remains an open challenge. Two obstacles limit the adoption of learning-based network control in this setting: traditional volume-based routing metrics, while highly effective for general traffic management, are not designed to capture traffic urgency; and Deep Reinforcement Learning (DRL) controllers trained from scratch suffer from sample inefficiency, long training times, and early-stage exploration volatility. This paper introduces a deployment-focused network control framework that addresses both obstacles. First, we present Effective Congestion (EC), a deadline-aware metric family that quantifies interface congestion by packet urgency and proactively filters non-viable traffic, coupled with a Uniform Path Grouping (UPG) distribution heuri
    
[^101]: KC-Bench：用于评估大语言模型智能体知识冲突的动态交互基准

    KC-Bench: A Dynamic Interactive Benchmark for Evaluating Knowledge Conflicts in LLM Agents

    [https://arxiv.org/abs/2609.03588](https://arxiv.org/abs/2609.03588)

    KC-Bench是一个评估大语言模型智能体在多轮交互中处理世界知识冲突、输入不一致和时间冲突能力的新型基准，评估发现没有任何模型能够在所有场景下可靠地识别和解决知识冲突。

    

    随着大语言模型越来越多地通过工具来执行操作，它们必须在采取行动之前协调用户指令、参数化知识和动态环境观察结果。我们提出了KC-Bench，这是一个受控的多轮对话基准，用于衡量模型在世界知识冲突、输入不一致以及多源时间冲突方面的处理能力。该基准的238个任务是从1000多个生成的候选任务中经过人工筛选而来，并整合了用户模拟器、有状态工具、确定性环境断言、开源自然语言评估器以及人工轨迹验证。对包括DeepSeek-V4-Flash、GLM-5.2和MiniMax-M3在内的九个模型的评估显示出显著的跨领域差异：没有任何模型能够在所有设置下可靠地处理事实修正、身份一致性检查和时间冲突解决。在模拟环境中，被遗漏的冲突可能会传播到工具调用或合成的受保护数据流中。

    arXiv:2609.03588v1 Announce Type: new  Abstract: As LLMs increasingly act through tools, they must reconcile user instructions, parametric knowledge, and dynamic environmental observations before taking actions. We introduce KC-Bench, a controlled multi-turn benchmark for measuring this capability across world-knowledge conflicts, input inconsistencies, and multi-source temporal conflicts. Its 238 tasks are manually screened from more than 1,000 generated candidates and combine a user simulator, stateful tools, deterministic environment assertions, an open-source natural-language evaluator, and human trajectory verification. Evaluation of nine models, including DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3, shows substantial cross-domain variation: no model handles factual correction, identity consistency checking, and temporal conflict resolution reliably across all settings. In the simulated environments, missed conflicts can propagate to tool calls or synthetic protected-data flows. KC
    
[^102]: 音频-视频模型中的注意力三角形

    The Attention Triangle in Audio-Video Models

    [https://arxiv.org/abs/2609.03586](https://arxiv.org/abs/2609.03586)

    本文通过分析音频-视频扩散模型中连接文本、音频、视频的“注意力三角形”，揭示了音频与视频之间双向的语义泄漏机制——当提示词与模型先验冲突时，跨模态交互会覆盖预期控制，将语义路由到视觉上典型但错误的结果。

    

    音频-视频扩散模型依赖跨模态注意力来协调文本、声音与视觉内容，然而这一机制也可能引入微妙且系统性的语义泄漏。我们通过探测并分析“注意力三角形”来研究这些模型——即连接文本、音频和视频流的三条交叉注意力边——并考察生成过程中语义信息如何在各模态之间传递。我们的分析揭示了音频-视频边上的信息路由是双向的：音频可以影响视频生成，视频同样可以影响音频生成。这条边受到模型参数中编码的偏置所塑造，并成为语义泄漏的主要来源：当提示词与模型学到的先验相冲突时，跨模态交互可能会覆盖预期的条件控制，将语义重新路由至视觉上典型但错误的结果。这些效应表明，语义伪影的产生并非仅仅……

    arXiv:2609.03586v1 Announce Type: new  Abstract: Audio-video diffusion models rely on cross-modal attention to coordinate text, sound, and visual content, yet this same mechanism can introduce subtle and systematic semantic leakage. We study these models by probing and analyzing the ``attention triangle,'' comprising the three cross-attention edges connecting the text, audio, and video streams, and examine how semantic information is routed across modalities during generation. Our analysis reveals that routing along the audio-video edge is bidirectional: audio can influence video generation, while video can influence audio generation. This edge is shaped by biases encoded in the model's parameters and emerges as a major contributor to leakage: when prompts are in tension with learned priors, cross-modal interactions may override the intended conditioning and reroute semantics toward visually canonical but incorrect outcomes. These effects suggest that semantic artifacts arise not merel
    
[^103]: HalluPeer：一个面向科学同行评审中幻觉检测的分类体系驱动基准测试

    HalluPeer: A Taxonomy-driven Benchmark for Detecting Hallucinations in Scientific Peer Reviews

    [https://arxiv.org/abs/2609.03580](https://arxiv.org/abs/2609.03580)

    该论文提出了HalluPeer——首个面向科学同行评审场景的幻觉检测基准，通过构建论文、真实评审与注入幻觉评审的对齐数据集以及同行评审专属的幻觉分类体系，揭示了现有检测器难以区分幻觉与合理批评的局限。

    

    学术同行评审规模的不断增长推动了将大语言模型（LLM）用作评审助手的实践，然而LLM可能会生成流畅但缺乏依据的论断，从而损害评审的可靠性。现有的幻觉基准测试并非为同行评审场景设计，因为在这一场景中，验证论断需要以冗长且技术性强的论文为依据。我们提出了HalluPeer，一个用于检测科学同行评审中幻觉的基准测试，它提供了论文内容、人工撰写的评审以及注入幻觉的评审三者对齐的数据三元组，并针对幻觉的检测、分类和定位进行了标注。我们的流程构建了面向同行评审的幻觉分类体系，识别评审上下文，并通过自动化过滤注入幻觉。在1.2万篇论文和3.8万条评审上的实验表明，现有检测器难以将幻觉与合理的批评意见区分开来，而对真实评审的评估则证明HalluPeer……

    arXiv:2609.03580v1 Announce Type: new  Abstract: The growing scale of academic peer review has motivated the use of Large Language Models (LLMs) as review assistants, yet LLMs can generate fluent but unsupported claims that undermine review reliability. Existing hallucination benchmarks are not designed for peer review, where verification requires grounding claims in long, technical papers. We introduce HalluPeer, a benchmark for detecting hallucinations in scientific peer reviews, providing aligned triples of paper content, human-written reviews, and hallucination-injected reviews, annotated for detection, classification, and localization. Our pipeline induces a peer-review-specific hallucination taxonomy, identifies review contexts, and injects hallucinations with automated filtering. Experiments on 12K papers and 38K reviews show that existing detectors struggle to separate hallucinations from legitimate critique, while evaluation on authentic reviews demonstrates that HalluPeer-def
    
[^104]: 迈向面向目标条件机器人规划的物理接地JEPA世界模型

    Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning

    [https://arxiv.org/abs/2609.03565](https://arxiv.org/abs/2609.03565)

    该论文提出一种融合逆动力学与状态对齐的端到端JEPA世界模型，将潜在表示扎根于物理构型和运动信息，从而显著提升目标条件机器人规划的成功率。

    

    动作条件化的JEPA世界模型能够在不重建未来像素的情况下朝向视觉指定的目标进行规划，然而仅靠潜在预测并不能显式地促使所学习的表示保留与机器人控制相关的信息。我们提出了一种端到端的JEPA世界模型，通过逆动力学（IDM）和状态对齐（SA）来增强潜在预测。逆动力学能够抑制潜在坍缩，并使潜在转变携带产生它们的动作信息；而状态对齐则将连续的表示扎根于其对应的物理构型和运动之中。在四个基准任务上，我们的模型在TwoRoom（100%）、PushT（98%）和OGBench-Cube（87%）上取得了最高的成功率，同时在Reacher任务上与LeWorldModel表现相当。我们的消融实验进一步表明，在所有四个任务中，相比仅使用逆动力学，添加状态对齐均能持续提升规划成功率。

    arXiv:2609.03565v1 Announce Type: cross  Abstract: Action-conditioned JEPA world models enable planning toward visually specified goals without reconstructing future pixels, yet latent prediction alone does not explicitly encourage the learned representations to retain information relevant to robotic control. We introduce an end-to-end JEPA world model that augments latent prediction with inverse dynamics (IDM) and state alignment (SA). While inverse dynamics discourages latent collapse and makes latent transitions informative of the actions that produced them, state alignment grounds consecutive representations in their associated physical configuration and motion. Across four benchmark tasks, our model attains the highest success rates on TwoRoom (100%), PushT (98%), and OGBench-Cube (87%), while performing comparably to LeWorldModel on Reacher. Our ablation further shows that adding state alignment consistently improves planning success over IDM alone across all four tasks. Although
    
[^105]: WIDE：面向跨模态生成式检索的动态扩展通配符推理

    WIDE: Wildcard Inference with Dynamic Expansion for Cross-Modal Generative Retrieval

    [https://arxiv.org/abs/2609.03554](https://arxiv.org/abs/2609.03554)

    提出WIDE方法，利用通配符推理与动态扩展来解决跨模态生成式检索中因模态间信息不对称导致的解码器“被迫幻觉”问题，从而避免无关候选占据高排名。

    

    生成式检索通过将表示学习和搜索统一到单一的序列到序列生成任务中，已展现出显著成效。然而，将该范式扩展到跨模态检索时，暴露出一个由不同模态间固有信息不对称带来的关键挑战，例如简洁的文本查询与密集的视觉候选之间的差距。这种结构性失配导致自回归解码器在通过标准的基于Trie树约束的束搜索生成标识符时出现“被迫幻觉”现象——模型因无法猜测查询中缺失的细粒度细节而受到严重惩罚，致使无关候选得以占据高排名位置。为解决这一问题，我们提出了动态扩展通配符推理（WIDE）。WIDE采用自适应熵阈值（AET）在离线阶段校准各层特定的不确定性边界。在解码生成阶段，

    arXiv:2609.03554v1 Announce Type: cross  Abstract: Generative retrieval has demonstrated significant success by unifying representation learning and search into a single sequence-to-sequence generation task. However, extending this paradigm to cross-modal retrieval reveals a critical challenge arising from the inherent information asymmetry across different modalities, such as the gap between concise text queries and dense visual candidates. This structural mismatch causes the autoregressive decoder to suffer from forced hallucination when generating identifiers via standard trie-constrained beam search, where the model is severely penalized for failing to guess fine-grained details absent from the query, allowing irrelevant candidates to hijack top rankings. To address this issue, we propose Wildcard Inference with Dynamic Expansion (WIDE). WIDE employs Adaptive Entropy Thresholding (AET) to calibrate layer-specific uncertainty boundaries offline. During the decoding generation phase,
    
[^106]: GPS-Bench：一个用于自动化政策分析的治理政策基准

    GPS-Bench: A Governance Policy Benchmark for Automating Policy Analysis

    [https://arxiv.org/abs/2609.03553](https://arxiv.org/abs/2609.03553)

    GPS-Bench是一个基于公开证据构建的治理政策模拟评估基准，它将政策与真实行为者、行动及下游影响相联系，通过人工标注的Gold评估集来检验大语言模型政策模拟的有效性。

    

    政策分析不仅仅是预测一项提案是否会通过：它需要识别谁将受到影响、这些行为者将如何反应，以及随后会发生什么。基于大语言模型的政策模拟能够大规模建模这些过程，但当看似合理的行为从未与实际观察到的结果进行比较时，其有效性难以确立。我们提出了GPS-Bench，一个以证据为基础的治理政策模拟基准，它利用立法记录、游说披露、监管文件、企业申报、经济数据和其他公开证据，将政策与相关行为者、行为者的行动以及下游影响联系起来。行为者是从带有日期的档案记录中重建的，而不是以原型方式被提示，因此每个角色都是一个具有来源出处的证据对象；人工标注的池构成Gold评估集，而由另一个大语言模型基于检索到的证据进行标注的案例则被视为Silver监督数据，绝不用作测试标签。

    arXiv:2609.03553v1 Announce Type: new  Abstract: Policy analysis requires more than predicting whether a proposal will pass: it requires identifying who will be affected, how those actors respond, and what follows. LLM-based policy simulations model these processes at scale, but their validity is hard to establish when plausible behaviour is never compared with observed outcomes. We introduce GPS-Bench, an evidence-grounded benchmark for governance policy simulation that links policies to relevant actors, actor actions and downstream impacts using legislative records, lobbying disclosures, regulatory documents, corporate filings, economic data and other public evidence. Actors are reconstructed from the dated record rather than prompted as archetypes, so a persona is an evidence object with provenance; a human-annotated pool forms the Gold evaluation set, while cases labelled by a separate LLM from retrieved evidence are treated as Silver supervision and never as test labels. Because e
    
[^107]: Dalek：一种构造性智能体机器

    Dalek: A Constructive Agent Machine

    [https://arxiv.org/abs/2609.03546](https://arxiv.org/abs/2609.03546)

    该论文提出 Dalek——一种借鉴冯·诺依曼自复制自动机、由执行体/消息/信道三种原语和四项结构义务构成的封闭智能体机器，可在任意满足宿主契约的基底上实现自我维护、进化、复制与自组织，并以大语言模型和编译器作为通用能力生产器。

    

    我们提出了 Dalek，一种为智能体设计的封闭机器，它能够在任何满足通用宿主契约的基底上实现自我维护、自我进化、自我复制和自组织。该机器由三种原语构建——执行体、消息和信道。四项义务——宿主边界、构造语言、容许转移和规则遗传——为其边界、身份和封闭性提供了结构基础。冯·诺依曼1948年的自复制自动机为其提供了遗传构造核心：一份自描述，连同构造器、复制器和控制器。Dalek 将这一核心与四项义务相结合，并针对文本与消息的智能体基底重新推导了其介质，增加了用于边界、身份、历史和生长的显式结构。一个大语言模型和一个编译器占据载荷位置，构成通用能力生产器。新能力被编写、编译、安装。

    arXiv:2609.03546v1 Announce Type: new  Abstract: We present Dalek, a closed machine designed for agents that realizes self-maintenance, self-evolution, self-reproduction, and self-organization on any substrate satisfying a general host contract. The machine is built from three primitives---actors, messages, and channels. Four obligations---a host boundary, a construction language, admissible transitions, and rule heredity---give its boundary, identity, and closure a structural basis.   Von Neumann's 1948 self-reproducing automaton supplies a hereditary constructional core: a self-description together with a constructor, a copier, and a controller. Dalek combines this core with the four obligations and rederives its medium for a text-and-message agent substrate, adding explicit structures for boundary, identity, history, and growth. A large language model and a compiler occupy the payload position and form a general capability producer. New capabilities are authored, compiled, installed
    
[^108]: 基于视觉先验的特征重配置医学病灶分割

    Feature Reconfiguration With Visual Prior for Medical Lesion Segmentation

    [https://arxiv.org/abs/2609.03535](https://arxiv.org/abs/2609.03535)

    提出带视觉先验的特征重配置框架FreNet，通过编码前的像素级重配置和编码中的特征级重配置来抑制背景干扰并应对多样病灶形态，从而提升医学病灶分割的精确性。

    

    arXiv:2609.03535v1 公告类型：new 摘要：医学图像中的病灶分割在临床诊断和治疗规划中起着至关重要的作用。尽管已取得显著进展，但由于两个主要因素，病灶分割仍然具有挑战性：(1) 复杂的背景干扰；(2) 多样的病灶形态。现有的基于编码器-解码器的方法主要专注于增强特征提取或重新设计解码策略。然而，它们缺乏早期先验引导以及编码阶段的特征重配置，限制了其在应对这些挑战时的有效性。为解决这些局限性，我们提出了FreNet，一种带视觉先验的特征重配置框架，它在编码前进行像素级重配置，并在编码过程中进行特征级重配置，以实现精确的医学病灶分割。为了抑制背景响应，我们提出了隐式先验神经网络（IPNN），它对连续空间场进行建模并利用视觉先验。

    arXiv:2609.03535v1 Announce Type: new  Abstract: Lesion segmentation in medical images plays a critical role in clinical diagnosis and treatment planning. Despite significant advances, lesion segmentation remains challenging due to two major factors: (1) complex background interference; (2) diverse lesion morphology. Existing encoder-decoder based methods mainly focus on enhancing feature extraction or redesigning decoding strategies. However, they lack early prior guidance and feature reconfiguration during the encoding stage, limiting their effectiveness in handling these challenges. To address these limitations, we propose FreNet, a feature reconfiguration framework with visual priors, which performs pixel-level reconfiguration before encoding and feature-level reconfiguration during encoding for precise medical lesion segmentation. To suppress background responses, we propose an Implicit Prior Neural Network (IPNN), which models a continuous spatial field and leverages visual prior
    
[^109]: TruncGradGS：通过截断梯度更新改进3D高斯泼溅

    TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates

    [https://arxiv.org/abs/2609.03534](https://arxiv.org/abs/2609.03534)

    提出分段截断梯度更新方法TruncGradGS，解决了3D高斯泼溅中的梯度消失问题，显著提升了优化稳定性、对初始化的鲁棒性以及静态和动态场景的重建质量。

    

    3D高斯泼溅（3D Gaussian Splatting）已成为新视角合成领域事实上的场景表示方法，然而从视觉输入中稳健地学习3D高斯基元仍然具有挑战性。标准优化依赖于基于梯度的更新，但一个常见的问题是梯度消失现象：距离高斯基元较远的像素往往具有递减的梯度幅值来影响基元属性，导致场景重建效果欠佳。在本文中，我们提出了一种通过分段截断梯度公式来解决梯度消失问题的方法，该方法提升了优化的稳定性以及对初始化的鲁棒性。我们证明了我们的方法在使用随机初始化和COLMAP初始化的3D高斯泼溅中均能持续带来改进，同时可泛化应用于静态和动态高斯泼溅。作为附带成果，我们还审视了当前动态场景基准测试的局限性，并引入了一个用于基准测试的新型数据集……

    arXiv:2609.03534v1 Announce Type: cross  Abstract: 3D Gaussian Splatting has become a de facto scene representation for novel view synthesis, yet robustly learning 3D Gaussian primitives from visual input remains challenging. Standard optimization relies on gradient-based updates, but a common issue is the gradient vanishing phenomenon: a pixel far from a Gaussian primitive often has diminishing gradient magnitudes to influence primitive attributes, resulting in suboptimal scene reconstruction. In this paper, we propose a method to address gradient vanishing with a piecewise truncated gradient formulation that improves the optimization stability and robustness to initializations. We show that our method consistently improves 3D Gaussian Splatting with random and COLMAP initializations while being generalizable across static and dynamic Gaussian Splatting. As a by-product, we also examine the limitations of current benchmarks for dynamic scenes, and introduce a novel dataset for benchma
    
[^110]: LeanGRPO：消除扩散强化学习中的冗余重计算

    LeanGRPO: Eliminating Redundant Recomputation in Diffusion RL

    [https://arxiv.org/abs/2609.03528](https://arxiv.org/abs/2609.03528)

    LeanGRPO 通过重构数据并行布局并提出两种无需重计算的训练方案，使 rollout 阶段的计算图与激活值可在策略更新时直接复用，从而消除同策略扩散强化学习中数学上冗余的重计算。

    

    扩散强化学习（RL）最近在图像和视频生成模型的后训练中取得了显著成功。然而，包括 DanceGRPO 和 FlowGRPO 在内的大多数扩散强化学习方法，在 rollout 之后都会对选定的时间步进行开启梯度追踪的重计算。在 rollout 与更新使用相同后端的同策略（on-policy）训练中，这种重计算在数学上是冗余的。直观上，rollout 和策略更新步骤可以复用同一个前向传播骨干网络来避免冗余计算，但这样做会在 rollout 阶段带来较大的内存开销。为了解决这一问题，我们提出了 LeanGRPO，通过重构数据并行布局，并为轨迹-对数概率扩散强化学习引入两种无需重计算的训练方案：(1) LeanGRPO-Retain 在 rollout 期间开启梯度追踪，并在更新阶段直接复用所得的计算图和保存的激活值进行反向传播，无需任何重计算；(2) L（摘要在此处截断）

    arXiv:2609.03528v1 Announce Type: cross  Abstract: Diffusion reinforcement learning (RL) has recently achieved significant success in post-training image and video generative models. However, most diffusion RL methods, including DanceGRPO and FlowGRPO, recompute selected timesteps with gradient tracking after rollout. Under on-policy training with the same backend for rollout and update, this recomputation is mathematically redundant. Intuitively, the rollout and policy update steps can reuse the same feed-forward backbone to avoid redundant computation, but doing so can incur a large memory overhead during rollout. To address the issue, we present LeanGRPO by restructuring the data-parallel layout and introducing two recompute-free training schedules for trajectory-logprob diffusion RL: (1) LeanGRPO-Retain enables gradient tracking during rollout and directly reuses the resulting computation graphs and saved activations for backward during update, requiring no recomputation; and (2) L
    
[^111]: NeoRed：一个用于新生儿呼吸系统疾病诊断的知识-逻辑-对齐多模态大语言模型

    NeoRed: A Knowledge-Logic-Alignment Multimodal Large Language Model for Neonatal Respiratory Disease Diagnosis

    [https://arxiv.org/abs/2609.03527](https://arxiv.org/abs/2609.03527)

    本文提出了首个专为新生儿呼吸系统疾病诊断定制的多模态大语言模型NeoRed，通过知识-逻辑-对齐（KLA）框架解决了成人数据领域差距和多维临床信息整合不足的问题，填补了新生儿诊断报告生成的空白。

    

    新生儿呼吸系统疾病是导致新生儿发病和死亡的主要原因之一，在临床实践中带来了巨大挑战。尽管近期取得了进展，现有的多模态大语言模型（MLLMs）在新生儿诊断中面临两个关键局限性：（1）主要由成人训练数据导致的领域差距；（2）缺乏对多维临床信息的充分整合以实现准确诊断。为应对这些挑战，我们收集了两个真实世界的临床数据集（NeoCXR和NeoCXR-EV），并提出了NeoRed——据我们所知，这是首个专为新生儿呼吸系统疾病定制的多模态大语言模型，填补了新生儿诊断报告生成领域的空白。为增强基于异构临床信息和胸部X光片的联合诊断，我们设计了一种新颖的知识-逻辑-对齐（KLA）框架，从三个角度约束模型行为：1）知识先验注入（KPI）融合了基于新生儿科医生经验的诊断知识……

    arXiv:2609.03527v1 Announce Type: new  Abstract: Neonatal respiratory diseases are a major cause of neonatal morbidity and mortality, posing substantial challenges in clinical practice. Despite recent advances, existing Multimodal Large Language Models (MLLMs) face two key limitations in neonatal diagnosis: (1) domain gap arising from predominantly adult training data; (2) insufficient integration of multidimensional clinical context for accurate diagnosis. To address these challenges, we collect two real-world clinical datasets (NeoCXR and NeoCXR-EV) and propose NeoRed, to the best of our knowledge, the first MLLM tailored for neonatal respiratory disease, filling the gap in neonatal diagnostic reports generation. To enhance joint diagnosis from heterogeneous clinical context and chest X-rays, we design a novel Knowledge-Logic-Alignment (KLA) framework which constrains model behavior from three perspectives: 1) Knowledge Prior Injection (KPI) incorporates neonatologist-inspired diagno
    
[^112]: CulturalMenuBench：探究多模态烹饪推理中的知识应用鸿沟

    CulturalMenuBench: Probing the Knowledge-Application Gap in Multimodal Culinary Reasoning

    [https://arxiv.org/abs/2609.03526](https://arxiv.org/abs/2609.03526)

    该论文提出CulturalMenuBench多模态烹饪基准，揭示出多模态模型尽管在食物识别上接近满分，却在将菜肴归入中国地方菜系等文化归因任务中骤降至最多56%的准确率，暴露了显著的知识应用鸿沟。

    

    多模态语言模型在食物识别基准测试中已取得接近满分的成绩，但这种成功究竟反映了真正的文化理解，还是仅仅依靠视觉匹配，仍不清楚。为了探究这一区别，我们提出了CulturalMenuBench，这是一个涵盖10种语言、18个地区、共4,870个条目的基准测试；其10项任务将最终成品图和逐步烹饪过程图与食材、流程文本和地区标签相配对，涵盖从基础识别到基于过程的文化归因。对12个模型的评估揭示了一个显著的知识应用鸿沟：在标准多项选择题任务中准确率超过94%的模型，在将菜肴归入中国地方菜系时准确率最高仅为56%，尽管任务采用完全相同的四选一格式。诊断分析解释了原因：错误模式与随机猜测相一致，准确率与视觉独特性而非文化结构相关，且模型从菜肴[原文摘要在此处截断]

    arXiv:2609.03526v1 Announce Type: new  Abstract: Multimodal language models achieve near-ceiling scores on food recognition benchmarks, yet it remains unclear whether this success reflects genuine cultural understanding or mere visual matching. To probe this distinction, we introduce CulturalMenuBench, a benchmark of 4,870 items in 10 languages across 18 regions; its 10 tasks pair final-dish and step-by-step cooking images with ingredients, procedural text, and regional labels, spanning basic recognition to process-grounded cultural attribution. Evaluating 12 models exposes a substantial knowledge-application gap: models exceeding 94% on standard multiple-choice tasks drop to at most 56% when attributing dishes to Chinese regional cuisines, despite an identical four-way format. Diagnostic analyses explain why: error patterns are consistent with random guessing, accuracy tracks visual distinctiveness rather than cultural structure, and models classify cuisines more accurately from dish 
    
[^113]: 基于可变形时序对齐与差异感知融合的神经视频压缩

    Neural Video Compression Based on Deformable Temporal Alignment and Difference-aware Fusion

    [https://arxiv.org/abs/2609.03520](https://arxiv.org/abs/2609.03520)

    该论文提出一种结合可变形时序对齐与差异感知空间选择性融合的神经视频压缩方法，通过生成互补时序上下文并自适应选择可靠时序信息来抑制错位误差，相比DCVC-DC提升了率失真性能。

    

    在基于条件编码的神经视频压缩中，时序上下文的质量直接影响压缩性能。现有方法大多从传播的参考特征构建上下文，但在复杂运动、遮挡和高频纹理区域，容易受到运动估计和局部对齐误差的影响，导致时序信息不准确。为解决这一问题，本文提出了一种结合可变形时序对齐与差异感知空间选择性融合的方法。其中，上下文感知时序对齐模块用于生成互补的时序上下文，而差异感知空间选择性融合模块则自适应地选择可靠的时序信息并抑制错位。实验表明，所提出的方法相比DCVC-DC取得了一定的率失真性能提升。

    arXiv:2609.03520v1 Announce Type: cross  Abstract: In conditional coding-based neural video compression, the quality of temporal context directly affects compression per- formance. Existing methods mostly construct context from prop- agated reference features, but they are vulnerable to motion esti- mation and local alignment errors in regions with complex mo- tion, occlusion, and high-frequency textures, resulting in inaccu- rate temporal information. To address this issue, this paper pro- poses a method combining deformable temporal alignment and difference-aware spatial selective fusion. A Context-aware Tem- poral Alignment Module is used to generate complementary tem- poral context, while a Difference-aware Spatial Selective Fusion module adaptively selects reliable temporal information and sup- presses misalignment. Experiments show that the proposed method achieves certain rate-distortion performance improve- ment over DCVC-DC.
    
[^114]: 什么因素对激进的解码时KV缓存驱逐至关重要？时间聚合与排序保持

    What Matters for Aggressive Decoding-Time KV Eviction? Temporal Aggregation and Ranking Preservation

    [https://arxiv.org/abs/2609.03515](https://arxiv.org/abs/2609.03515)

    该论文发现在激进KV缓存压缩中，时间聚合规则（如EMA）比token评分函数设计更关键，因为它主导了保留集的稳定性，并据此提出InertiaKV及其惰性刷新变体，实现1.34-1.46倍的解码吞吐量提升。

    

    arXiv:2609.03515v1 公告类型：新论文 摘要：解码时KV缓存压缩研究主要集中于设计更好的token评分函数，而跨解码步骤聚合分数的时间规则通常被视为实现细节。在激进的KV压缩下，我们发现指数移动平均（EMA）聚合使得近似保持排序的评分器修改在驱逐集层面基本难以区分。Value-norm和熵变体与注意力保持高度相关，产生几乎不变的保留集，而KeyDiff、key norm、最近性以及学习型评分器则会改变排序并大幅降低性能。我们将这种稳定性与所评估的聚合方式联系起来，该聚合方式将层权重和时间保留耦合在一起。基于这一观察，我们提出了InertiaKV，一种基于EMA的解码时驱逐方法，以及其周期性刷新变体InertiaKV-Lazy，可实现相对于基线1.34-1.46倍的解码吞吐量。

    arXiv:2609.03515v1 Announce Type: new  Abstract: Decoding-time KV cache compression research focuses heavily on designing better token scoring functions, while the temporal rule that aggregates scores across decode steps is often treated as an implementation detail. Under aggressive KV compression, we find that exponential-moving-average (EMA) aggregation makes approximately order-preserving scorer modifications largely indistinguishable at the eviction-set level. Value-norm and entropy variants remain highly correlated with attention and produce nearly unchanged retention sets, whereas KeyDiff, key norm, recency, and a learned scorer alter the ranking and degrade substantially. We associate this stability with the evaluated aggregation, which couples layer weighting and temporal retention. Building on this observation, we introduce InertiaKV, an EMA-based decoding-time eviction method, and InertiaKV-Lazy, its periodic-refresh variant, which yields 1.34-1.46x decode throughput relative
    
[^115]: LongCounsel-8：基于多轮次心理咨询对话的纵向抑郁症追踪基准套件

    LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues

    [https://arxiv.org/abs/2609.03507](https://arxiv.org/abs/2609.03507)

    提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。

    

    从多轮次心理咨询对话中追踪抑郁症，需要同时估计当前的症状严重程度以及其在不同会话之间的变化情况。然而，该任务的进展受限于缺乏具有标准化会话级抑郁标签的纵向咨询数据。现有资源通常要么提供没有抑郁标签的多轮次对话，要么仅提供单次会话的标注访谈。构建这样一个基准面临三大挑战：保持纵向的一致性与多样性，将症状进展建立在实证模式之上，以及在不暴露目标标签的前提下自然地表达受控的抑郁状态。为应对这些挑战，我们提出了LongCounsel-8，这是一个由三个独立生成的数据集组成的基准套件，共包含7,749条五轮次咨询轨迹，其建立在真实的来访者档案、抑郁轨迹、症状构成和咨询模式的基础之上。

    arXiv:2609.03507v1 Announce Type: cross  Abstract: Tracking depression from multi-session counseling dialogues requires estimating both current symptom severity and how it changes across sessions. Yet progress on this task is constrained by the scarcity of longitudinal counseling data with standardized session-level depression labels. Existing resources typically provide either multi-session conversations without depression labels or labeled interviews in a single session. Building such a benchmark poses three challenges: maintaining longitudinal consistency and diversity, grounding symptom progression in empirical patterns, and expressing controlled depression states naturally without exposing target labels. To address these challenges, we introduce LongCounsel-8, a benchmark suite of three independently generated datasets totaling 7,749 five-session counseling trajectories, grounded in real-world client profiles, depression trajectories, symptom compositions, and counseling patterns.
    
[^116]: PPO-STGNN：一种结合时空图神经网络的近端策略优化方法，用于云边端计算中的DAG任务调度

    PPO-STGNN: A Proximal Policy Optimization Approach with Spatio-Temporal Graph Neural Networks for DAG Task Scheduling in Cloud-Edge-End Computing

    [https://arxiv.org/abs/2609.03503](https://arxiv.org/abs/2609.03503)

    该论文提出PPO-STGNN算法，将近端策略优化与时空图神经网络相结合，同时捕捉DAG任务拓扑与云边端异构资源的时空动态特征，有效解决云边端协同环境中的DAG任务调度这一NP难问题。

    

    随着物联网的快速发展，计算密集型有向无环图（DAG）任务在云边端协同环境中日益普遍。然而，云端、边缘端和终端节点在计算能力、网络带宽和能耗方面高度异构，这使得具有复杂依赖关系的任务的高效调度成为一个NP难问题。传统的启发式算法和常规强化学习方法往往无法捕捉系统资源的时空动态特性。本文提出PPO-STGNN，一种将近端策略优化（PPO）与时空图神经网络（STGNN）相结合的DAG任务调度算法。该方法使用STGNN从DAG任务拓扑结构和物理云边端资源图中提取特征，然后通过PPO优化调度策略，以最小化完工时间和调度长度比。

    arXiv:2609.03503v1 Announce Type: new  Abstract: With the rapid development of the Internet of Things, computation intensive directed acyclic graph (DAG) tasks have become increasingly common in cloud-edge-end collaborative environments. However, cloud, edge, and end nodes are highly heterogeneous in computing capacity, network bandwidth, and energy consumption, which makes the efficient scheduling of tasks with complex dependencies an NP-hard problem. Traditional heuristic algorithms and conventional reinforcement-learning methods often fail to capture the spatio-temporal dynamics of system resources. This paper proposes PPO-STGNN, a DAG task-scheduling algorithm that integrates proximal policy optimization (PPO) with spatio-temporal graph neural networks (STGNNs). The method uses an STGNN to extract features from both the DAG task topology and the physical cloud-edge-end resource graph, and then optimizes the scheduling policy through PPO to minimize makespan and schedule length rati
    
[^117]: 基于合成语音构建与评估固定音色泰语语音合成系统

    Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech

    [https://arxiv.org/abs/2609.03502](https://arxiv.org/abs/2609.03502)

    该论文提出将大型声音克隆模型作为可编程数据源，仅凭15秒声音参考生成合成语音来训练紧凑的固定音色泰语TTS学生模型，并系统研究了文本准备、质量过滤、拒绝采样和前端选择等流水线设计对模型效果的影响及教师模型残留的局限性。

    

    在低资源环境下，部署TTS通常需要在两种方案之间做出选择：推理成本高昂的大型声音克隆模型，或需要特定说话人语料库的紧凑固定音色系统。我们研究了第三条路线：将大型声音克隆模型用作可编程的数据源，把简短的声音参考（例如15秒）转化为完全在合成语音上训练的紧凑固定音色学生模型。这一设定使得流水线设计变得至关重要：教师的错误会成为训练目标，而过滤失败的生成结果可能会降低对困难文本的覆盖率。泰语还带来了额外的挑战，包括词边界歧义、词汇声调、人名与外来词、数字的口语化表达以及泰英混读。我们研究了文本准备、合成生成、质量过滤、拒绝采样和前端选择如何影响最终的学生模型，以及教师模型的局限性在哪些方面依然存在。我们使用CER、挑战集关键词……（原文摘要在此处截断）

    arXiv:2609.03502v1 Announce Type: cross  Abstract: In low-resource settings, deploying TTS typically requires choosing between a large voice-cloning model with costly inference or a compact fixed-voice system that requires a speaker-specific corpus. We study a third route: using a large voice-cloning model as a programmable data source to turn a short voice reference (e.g., 15 seconds) into a compact fixed-voice student trained entirely on synthetic speech. This setting makes pipeline design consequential: teacher errors become training targets, while filtering failed generations can reduce coverage of difficult texts. Thai further introduces challenges from ambiguous word boundaries, lexical tone, names and loanwords, numeric verbalization, and Thai-English code-switching. We study how text preparation, synthetic generation, quality filtering, rejection sampling, and frontend choices affect the resulting student, and where teacher limitations remain. We evaluate CER, Challenge-Set Key
    
[^118]: BRIDGE：一种通过形态-控制协同设计实现物理智能的开源人形机器人平台

    BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI

    [https://arxiv.org/abs/2609.03497](https://arxiv.org/abs/2609.03497)

    本文提出了一种数据驱动的形态-控制协同设计框架，并据此构建了开源的88厘米高人形机器人平台Bridge，其在类人运动保真度和动态跟踪性能上全面超越现有基线人形机器人。

    

    开发能够利用人类行为数据的人形机器人对于通用具身智能至关重要，然而传统的开发方式受限于一种将硬件设计与全身控制相分离的解耦范式，导致系统性能欠佳，牺牲了类人的流畅性与敏捷性。为了弥合这一差距，我们提出了一种数据驱动的形态-控制协同设计框架，用于优化人形机器人的形态以实现类人运动。为了量化形态保真度，我们还提出了一种新颖的指标，该指标同时考虑了运动学重定向对人类动作的保真度以及动态跟踪性能。与基线人形机器人（Bumi、K1和Toddlerbot）相比，我们的框架在所有指标上均达到了最先进的（SOTA）性能。最后，我们将该设计落地为Bridge——一个开源的、身高88厘米的人形机器人平台，并随其一同发布了控制策略。我们证明了Bridge能够……

    arXiv:2609.03497v1 Announce Type: cross  Abstract: Developing humanoid robots capable of leveraging human behavioral data is essential for general-purpose embodiment, yet conventional development remains bottlenecked by a decoupled paradigm that isolates hardware design from whole-body control. This approach leads to suboptimal systems that compromise human-like fluidity and agility. To bridge this gap, we introduce a data-driven morphology-control co-design framework that optimizes humanoid morphology for human-like movement. To quantify morphological fidelity, we also introduce a novel metric that jointly considers kinematic retargeting fidelity to human motion and dynamic tracking performance. Our framework achieves state-of-the-art (SOTA) performance across all metrics compared to baseline humanoids (Bumi, K1, and Toddlerbot). Finally, we realize this design in Bridge, an open-source, 88cm-tall humanoid platform released alongside its control policy. We demonstrate that Bridge capt
    
[^119]: GrowPage：面向高效大语言模型推理服务的按需KV预算分配

    GrowPage: On-Demand KV Budgeting for Efficient LLM Reasoning Serving

    [https://arxiv.org/abs/2609.03494](https://arxiv.org/abs/2609.03494)

    GrowPage提出了一种按需KV预算分配框架，将KV缓存容量作为运行时动态资源，通过轻量级双时间尺度查询摘要追踪注意力需求的演变，并在容量边界处动态选择压缩KV状态或扩展物理页，从而提升LLM推理服务的效率。

    

    长输出推理使键值缓存成为高效LLM服务中的关键内存瓶颈。现有的KV压缩方法通常依赖于预定义的每请求预算，仅调整保留哪些KV状态，使总容量在整个解码过程中保持固定。然而，推理工作负载表现出显著的需求差异：不同的请求需要不同的KV容量，且单个请求的注意力需求在生成过程中不断演变。我们提出了GrowPage，一个将KV容量视为运行时资源的按需KV预算分配框架。GrowPage维护轻量级的双时间尺度查询摘要，以捕捉近期和长期的注意力行为，并利用它们的相对注意力工作集来估计需求的演变。在每个容量边界处，GrowPage要么在当前分配范围内压缩KV状态，要么在出现更广泛的需求时获取额外的物理页（当需求扩大时……）。

    arXiv:2609.03494v1 Announce Type: new  Abstract: Long-output reasoning has made the key--value (KV) cache a critical memory bottleneck for efficient LLM serving. Existing KV compression methods usually rely on a predefined per-request budget and adjust only which KV states are retained, leaving the total capacity fixed throughout decoding. However, reasoning workloads exhibit substantial demand variation: different requests require different KV capacities, and the attention demand of an individual request evolves during generation. We introduce \textbf{GrowPage}, an on-demand KV budgeting framework that treats KV capacity as a runtime resource. GrowPage maintains lightweight dual-timescale query summaries to capture recent and long-term attention behaviors, and uses their relative attention working sets to estimate demand evolution. At each capacity boundary, GrowPage either compresses KV states within the current allocation or acquires an additional physical page when broader demand e
    
[^120]: 让每次工具调用都有价值：面向智能体视觉语言模型的必要工具-证据路径奖励

    Making Every Tool Call Count: Necessary Tool-Evidence Path Rewards for Agentic Vision-Language Models

    [https://arxiv.org/abs/2609.03493](https://arxiv.org/abs/2609.03493)

    该论文提出NTEP（必要工具-证据路径）这一新型标注与奖励方案，通过显式监督必要的外部证据获取与利用路径，解决了智能体视觉语言模型中冗余工具调用和证据提取不足的问题。

    

    现代视觉语言模型（VLM）可以直接回答许多基于图像的问题，但在需要细粒度视觉细节或外部知识的复杂查询上常常表现不佳。为了获取这些缺失的证据，智能体视觉语言模型会调用图像裁剪、图像搜索和文本搜索等工具。然而，现有的训练范式主要根据最终答案的正确性来评估工具使用，导致对证据获取和利用的监督不足。这带来了两个关键缺陷：（i）模型经常发出冗余或偏离目标的工具调用，无法收集到必要的证据；（ii）即使调用了合适的工具，模型也常常无法从返回的观察结果中提取必要信息。为了解决这些局限，我们提出了NTEP（必要工具-证据路径），这是一种新颖的标注方案，明确指定了必需的外部证据及对应的工具……

    arXiv:2609.03493v1 Announce Type: new  Abstract: Modern vision-language models (VLMs) can directly answer many image-grounded questions, yet they often struggle with complex queries requiring fine-grained visual details or external knowledge. To acquire this missing evidence, agentic VLMs invoke tools such as image cropping, image search, and text search. However, existing training paradigms primarily evaluate tool-use based on final answer correctness, leaving evidence acquisition and utilization insufficiently supervised. This leads to two critical shortcomings: (i) models frequently issue redundant or off-target tool calls that fail to gather necessary evidence, and (ii) even when appropriate tools are called, models often fail to extract the necessary information from the resulting observations. To address these limitations, we introduce the NTEP (Necessary Tool-Evidence Path), a novel annotation scheme that explicitly specifies the essential external evidence and corresponding too
    
[^121]: 知识图谱嵌入的模式过度泛化问题

    Pattern Over-Generalization of Knowledge Graph Embedding

    [https://arxiv.org/abs/2609.03487](https://arxiv.org/abs/2609.03487)

    本文揭示了知识图谱嵌入中的模式过度泛化问题，并提出PogRE方法，通过稠密线性变换和复合操作进行关系表示，有效缓解该问题。

    

    知识图谱嵌入（KGE）通过将实体和关系投影到低维向量空间中，展示了其在预测知识图谱（KG）中缺失链接方面的有效性。对于KGE模型而言，有效捕获知识图谱中固有的推理模式至关重要，例如对称性/反对称性、反转和组合等模式。尽管近期的KGE模型在建模这些多样化模式方面表现出强大的能力，但它们受到模式过度泛化所带来的固有局限性的困扰：仅从单个模式实例学习到的嵌入，不可避免地会将该模式泛化到所有相关实例上，即将模式普遍化。为了解决这一问题，我们提出了PogRE（模式过度泛化鲁棒嵌入），这是一种简单而有效的方法，利用稠密线性变换和复合操作来进行关系表示。我们的理论分析表明，稠密...

    arXiv:2609.03487v1 Announce Type: cross  Abstract: Knowledge graph embedding (KGE) demonstrates its effectiveness for predicting missing links in knowledge graphs (KGs) by projecting entities and relations into a low-dimensional vector space. It is crucial for KGE models to effectively capture inference patterns (patterns) inherent in KGs, such as symmetry/antisymmetry, inversion and composition. Although recent KGE models exhibit strong capabilities in modeling such diverse patterns, they suffer from inherent limitations stemming from pattern over-generalization, where embeddings learned from only a single pattern instance inevitably generalize that pattern to all related instances, i.e., generalize the pattern universally. To address this issue, we propose PogRE (Pattern Over-Generalization Robust Embedding), a simple but effective method that utilizes dense linear transformations and compound operations for relation representation. Our theoretical analysis demonstrates that a dense 
    
[^122]: 基于共享鸟瞰图的空地协同视觉语言导航

    Air-Ground Collaborative Vision-and-Language Navigation via Shared Bird's-Eye Maps

    [https://arxiv.org/abs/2609.03483](https://arxiv.org/abs/2609.03483)

    该论文提出了首个面向空地协同视觉语言导航的免训练基线AGC-VLN，其核心创新在于利用共享鸟瞰图作为协作接口，将UGV的位姿和VLM锚定的目标标记叠加到无人机的全局视图上，从而实现空地智能体之间的有效导航协作。

    

    空地协同视觉语言导航（VLN）将拥有全局鸟瞰视野的无人机（UAV）与拥有局部第一人称视角的无人地面车辆（UGV）进行配对，然而这一设定在很大程度上仍未被探索：现有的免训练方法只能解决单智能体任务，缺乏协作机制；最近一项基于CARLA-Air的评估发现，五个最先进的VLA模型均无法表现出稳定的协作行为，而朴素的语义通信或双向耦合甚至会降低性能。我们建立了AGC-VLN（空地协同视觉语言导航），这是首个面向空地协同VLN的免训练基线。其关键洞察在于：免训练方法将导航分解为基于VLM的语义推理和确定性的几何执行，由此暴露出一个协作接口：即在无人机的全局视图上，将UGV上报的位姿和以VLM为锚点的目标渲染为CAR/GOAL标记……（摘要在此处被截断）

    arXiv:2609.03483v1 Announce Type: cross  Abstract: Air-ground collaborative Vision-and-Language Navigation (VLN) pairs an unmanned aerial vehicle (UAV) with a global bird's-eye view and an unmanned ground vehicle (UGV) with a local first-person view, yet the setting remains largely unexplored: existing training-free methods solve single-agent tasks but offer no collaboration mechanism, and a recent CARLA-Air evaluation found no stable cooperative behavior across five state-of-the-art VLA models; naive semantic communication or bidirectional coupling even degrades performance. We establish AGC-VLN (Air-Ground Collaborative VLN), the first training-free baseline for air-ground collaborative VLN. The key insight is that training-free methods decompose navigation into VLM-based semantic reasoning and deterministic geometric execution, exposing a collaboration interface: the UAV's global view, over which it renders the UGV's reported pose and the VLM-anchored target as CAR/GOAL markers with
    
[^123]: 丹麦树种制图：光谱-时间特征与地理空间基础模型嵌入的比较

    Tree species mapping in Denmark: A comparison of spectral-temporal features with geospatial foundation model embeddings

    [https://arxiv.org/abs/2609.03480](https://arxiv.org/abs/2609.03480)

    本研究利用丹麦国家森林清查数据与Sentinel卫星观测，系统比较了人工构建的光谱-时间特征与地球观测基础模型（TESSERA和AlphaEarth）嵌入在树种分类中的表现，发现基于光谱-时间特征的多层感知机取得最佳性能（纯林宏观F1达0.843），而基础模型嵌入也展现出有竞争力的结果，为大规模森林树种制图的方法选择提供了重要参考。

    

    我们利用国家森林资源清查样地和地球观测（EO）数据对丹麦全国的树种进行制图，同时评估了基础模型在大尺度森林表征方面的潜力。我们比较了两种用于树种分类的备选输入表示方式：(i) 基于多时相Sentinel-1和Sentinel-2观测数据人工构建的光谱-时间特征（STF），以及 (ii) 由地球观测基础模型TESSERA和AlphaEarth生成的嵌入表示。两种表示方法均辅以冠层高度信息。我们针对所有输入表示评估了随机森林、XGBoost和多层感知机（MLP）分类器，并对纯林和混交林分别进行了评估。基于STF的MLP取得了最高的分类性能，在纯林和混交林上的宏观F1分数分别为0.843和0.653。在TESSERA嵌入上训练的MLP在纯林分类上也表现出具有竞争力的性能。

    arXiv:2609.03480v1 Announce Type: cross  Abstract: We map tree species across Denmark using National Forest Inventory plots and EO data, while evaluating the potential of foundation models for large-scale forest characterization. We compare two alternative input representations for tree species classification: (i) manually engineered spectral-temporal features (STF) derived from multi-temporal Sentinel-1 and Sentinel-2 observations, and (ii) embeddings generated by the EO FMs TESSERA and AlphaEarth. Both representations are complemented with canopy height information. Random forest, XGBoost, and Multi-Layer Perceptron (MLP) classifiers are evaluated for all input representations, with separate assessments for pure and mixed forest stands. The STF-based MLP achieves the highest classification performance, yielding macro F1 scores of 0.843 and 0.653 for pure and mixed stands, respectively. The MLP trained on TESSERA embeddings delivers competitive performance for pure stands, achieving r
    
[^124]: AutoGraphForge：迈向自动化图论发现

    AutoGraphForge: Towards Automated Graph Theory Discovery

    [https://arxiv.org/abs/2609.03478](https://arxiv.org/abs/2609.03478)

    本文提出 AutoGraphForge 自动化计算流水线，通过反例引导的猜想生成、基于线性规划的新颖性过滤以及在约34.8万个图数据集上的大规模验证，实现图论猜想-反驳-形式化-证明的自动化闭环发现。

    

    我们报告一个正在进行的项目，旨在开发名为 AutoGraphForge 的计算流水线，用于构建自动化的图论“猜想-反驳-形式化-证明”系统。猜想生成由反例引导并分轮进行：一个 Graffiti3 生成器在一个小型且不断演化的快照表 $T$ 上提出猜想（该表最初包含几百个图及其计算得到的不变量），且该表仅通过其自身猜想的反例而增长。一个包含 $559$ 条经典与民间关系的新颖性过滤器（在传递复合与线性恒等替换下封闭），通过线性规划判断候选猜想是否已被已知结果所蕴含。存活的候选猜想将在一个约 $348,000$ 个图的数据集上进行测试，该数据集合并了 House of Graphs 的完整不变量导出数据、所有至多九个顶点的连通图的详尽普查，以及多个极值图族（强正则图、极小 Ramsey 图、Cayley 图、笼图、barbell 图……

    arXiv:2609.03478v1 Announce Type: new  Abstract: We report on our ongoing project to develop a computational pipeline, AutoGraphForge, for an automated graph-theoretic conjecturing-refuting-formalizing-proving system. Conjecture generation is counterexample-guided and runs in rounds: a Graffiti3 generator proposes conjectures over a small, evolving snapshot table $T$ (initially a few hundred graphs with their computed invariants) that grows only by counterexamples to its own conjectures. A novelty filter of $559$ classical and folklore relations, closed under transitive composition and linear identity substitution, decides via a linear program whether a candidate is already implied by known results. Surviving candidates are tested against a dataset of about $348,000$ graphs, unioning the complete House of Graphs invariant export, the exhaustive census of all connected graphs on at most nine vertices, several extremal families (strongly regular, minimal Ramsey, Cayley, cages, barbells, 
    
[^125]: 当用户不提问时：对话式智能体中上下文驱动的记忆检索基准测试

    When Users Don't Ask: Benchmarking Context-Driven Memory Retrieval in Conversational Agents

    [https://arxiv.org/abs/2609.03467](https://arxiv.org/abs/2609.03467)

    该论文提出了对话式记忆基准LOCOMO-CONV，通过对话式、隐含式、反事实和组合式四种查询风格，揭示了问答式基准测试所忽视的记忆检索差距，并发现强检索能力并不完全等同于高质量的对话响应。

    

    大型语言模型（LLM）越来越多地被部署为长时程对话式智能体，这推动了对记忆系统日益增长的关注。然而，现有基准测试主要通过问答式的探测方式来评估记忆，而非在真实对话场景中的实际使用。我们提出了LOCOMO-CONV，这是一个基于LoCoMo派生的对话式记忆基准，包含四种查询风格：对话式、隐含式、反事实式和组合式。我们在五个代表性记忆系统上评估了检索召回率和端到端响应质量。实验表明，对话式的表述方式暴露了问答式基准所忽视的大量检索差距，尤其是在隐含式和组合式查询上；多方面查询改写可以缩小原始对话轮次记忆的差距，但对抽象化记忆则无效。我们进一步发现，强大的检索能力并不能完全转化为响应质量，且隐含式查询表现出“静默接地”现象，即记忆在……（原文摘要在此处截断）

    arXiv:2609.03467v1 Announce Type: cross  Abstract: Large language models (LLMs) are increas- ingly deployed as long-horizon conversational agents, motivating growing interest in mem- ory systems. However, existing benchmarks primarily evaluate memory through QA-style probing rather than in-situ conversational usage. We introduce LOCOMO-CONV, a conversa- tional memory benchmark derived from Lo- CoMo with four query styles: dialog, implicit, counterfactual, and composed. Across five rep- resentative memory systems, we evaluate both retrieval recall and end-to-end response qual- ity. Our experiments show that conversational framing exposes substantial retrieval gaps over- looked by QA benchmarks, especially on im- plicit and composed queries, which multi-facet query rewriting narrows for raw-turn mem- ory but not abstractive memory. We further find that strong retrieval does not fully trans- late into response quality, and that implicit queries exhibit silent grounding, where mem- ory imp
    
[^126]: 超越“由AI制作”标签：通过溯源密度可视化缓解透明度惩罚

    Beyond "Made with AI": Visualizing Provenance Density to Mitigate the Transparency Penalty

    [https://arxiv.org/abs/2609.03460](https://arxiv.org/abs/2609.03460)

    本文提出“溯源密度”证据可视化界面，通过展示文本中经验证论断的密度来克服“流畅性陷阱”与AI标注带来的透明度惩罚，实验证明其显著提升用户辨别真伪的能力，且技术审计发现“一致性否决”机制比检索密度承载了更多判别信号。

    

    随着生成式AI使流畅的文本变得廉价易得，用户再也无法依赖流畅性作为判断真相的依据。我们将这种失败模式称为“流畅性陷阱”：用户会信任流畅的幻觉内容，同时一旦准确的内容被披露为AI生成，便会贬低其价值。二元的“由AI制作”标签仅回应了作者身份的披露，但并不能显示哪些证据支持某个论断。我们提出了“溯源密度”，这是一种证据可视化界面，用于展示文本中经过验证的论断的密度。在一项包含81名参与者的用户研究中，理想化的溯源密度界面在真实与伪造内容之间产生了显著的辨别差距（+4.15分，d=1.82），而没有接收任何信号的参与者则未表现出可检测的辨别能力。一项基于200个样本的技术审计表明，仅靠检索密度是不够的；出乎意料的是，“一致性否决”机制在动态查询上承载了大部分判别信号。

    arXiv:2609.03460v1 Announce Type: new  Abstract: As generative AI makes polished prose cheap to produce, users can no longer rely on fluency as a proxy for truth. We call this failure mode the Fluency Trap: users trust fluent hallucinations while also discounting accurate content once it is disclosed as AI-generated. Binary ``Made with AI'' labels respond with authorship disclosure, but they do not show what supports a claim. We propose Provenance Density, an evidence-visualization interface that shows the density of verified claims in a text. In a user study with 81 participants, an idealized Provenance Density interface produced a large discernment gap between truth and fabrication ($+4.15$ points, $d=1.82$), whereas participants given no signal showed no detectable discrimination. A technical audit with 200 samples shows that retrieval density alone is insufficient; unexpectedly, the Consistency Veto carries most of the discriminative signal on dynamic queries. As AI-generated conte
    
[^127]: 软件工程中人工智能应用的心理成本

    The Psychological Costs of Artificial Intelligence Adoption in Software Engineering

    [https://arxiv.org/abs/2609.03456](https://arxiv.org/abs/2609.03456)

    本研究首次关注软件工程领域组织采用AI过程中软件专业人员所承受的心理成本，挑战了“AI应用于软件工程是无成本的”这一常见假设。

    

    人工智能越来越多地被用于增强软件工程的工作流程。虽然代码生成仍然是主要应用场景，但各组织正在积极寻求将AI集成到其他实践中，例如测试用例生成和代码审查。组织层面的AI采用策略似乎主要关注生产力等有形成果。然而，AI是一种颠覆性力量，它被引入到那些在生成式AI取得近期进展之前，角色认同、团队规范和工作满意度来源早已确立的环境之中。从历史上看，技术颠覆曾在职场中造成心理和社会层面的压力，范围涵盖焦虑、意义感丧失，乃至技能退化和职业认同的破坏。因此，“AI应用于软件工程是没有成本的”这一假设可能并不准确。为此，本研究试图理解软件专业人员在组织采用AI的过程中所经历的心理成本。

    arXiv:2609.03456v1 Announce Type: cross  Abstract: Artificial intelligence (AI) is increasingly used to augment software engineering (SE) workflows. While code generation remains the main use case, organizations are actively seeking AI integration in other practices such as test cases generation and code reviews. Organizational AI adoption strategies seem to focus on tangible outcomes such as productivity. However, AI is a disruptive force, introduced into settings where role identity, team norms, and the sources of job satisfaction were well established before the recent advances in generative AI. Historically, technological disruptions have caused psychological and social strains in workplaces, ranging from anxiety and eroded meaning to deskilling and disrupted professional identities. The assumption that AI for SE is cost-free may not be accurate. Therefore, in this study we sought to understand the psychological costs software professionals experience during organizational AI adopt
    
[^128]: 预算化继承式智能体记忆验证中的计划指针与记录指令形式

    Plan Pointers and Record-Directive Form in Budgeted Verification of Inherited Agent Memory

    [https://arxiv.org/abs/2609.03450](https://arxiv.org/abs/2609.03450)

    该论文通过十二项注册研究发现，写入智能体记忆库的指令形式（准则、裸ID或指针）会以高度模型依赖的方式显著影响预算受限下的记录选择，长度匹配准则可带来35分的提升，但附加ID可能完全抵消准则的效果。

    

    arXiv:2609.03450v1 公告类型：cross。摘要：一个继承了六条单行记忆的智能体在行动前最多只能拉取一条存档的源记录；写入存储中的指令可以引导这一选择：可以是指向该记录的指针、识别该记录的准则，或两者兼有。在同一仪器谱系上的十二项注册研究（共14,760次尝试）中，我们测量了每种指令形式下请求的去向。在六个直接提供商模型上，长度匹配的准则比裸ID高出+35.0个点 [+31.2, +38.8]（研究D）；而在九个模型的OpenRouter服务面板上，该对比未能通过注册的优越性规则（研究E）。在三个Claude模型上，附加ID会抵消准则的效果（Opus 5: 从40/40降至0/40；研究F-x）；六次字节匹配的编辑使每个精确字符串都产生了各自的效应（研究G），并且在每单元八十次运行的重跑中，三十个复现对比中有十五个处于误差范围内，十五个未获解决，没有一个超出范围（研究G'）。批准行（在Opus 5上+96.0个点）以及一个（摘要在此处截断）

    arXiv:2609.03450v1 Announce Type: cross  Abstract: An agent that inherits six one-line memories may pull at most one archived source record before acting; a directive written into the store can steer that choice: a pointer to the record, a criterion that identifies it, or both. Across twelve registered studies on one instrument lineage (14,760 attempts) we measured where the request goes under each form. On six direct-provider models a length-matched criterion exceeded a bare id by +35.0 points [+31.2, +38.8] (Study D); the contrast failed its registered superiority rule on a nine-model OpenRouter-served panel (Study E). Appending the id cancelled the criterion on three Claude models (Opus 5: 40/40 to 0/40; Study F-x); six byte-matched edits gave each exact string its own effect (Study G), and a re-run at eighty runs per cell left fifteen of thirty replication contrasts within the margin, fifteen unresolved and none beyond (Study G'). A ratification line (+96.0 points on Opus 5) and a 
    
[^129]: GUI智能体知道何时不该行动吗？为多模态GUI智能体实现冲突感知的终止机制

    Do GUI Agents Know When Not to Act? Enabling Conflict-Aware Termination for Multimodal GUI Agents

    [https://arxiv.org/abs/2609.03438](https://arxiv.org/abs/2609.03438)

    本文提出CONFLICTGUI基准与CONFLICTGUARD推理时框架，揭示并缓解了多模态GUI智能体在面对冲突指令时盲目执行的过度顺从问题，通过可行性验证与条件终止机制使智能体学会在指令不可行时及时停止行动。

    

    图形用户界面（GUI）智能体越来越多地被用于在用户界面上执行自然语言指令，然而真实用户可能会因无心的错误而发出不可行的指令。一个可靠的智能体不仅应该知道如何行动，还应该知道何时不行动。在本工作中，我们引入了CONFLICTGUI基准，涵盖指令内部冲突和指令-GUI上下文冲突两类情况，以研究冲突感知的终止问题。我们的评估揭示了严重的执行偏向的过度顺从行为：在可行任务上表现良好的智能体，在遇到冲突指令时往往仍会盲目继续执行。为了缓解这种行为，我们提出了CONFLICTGUARD，一个推理时框架，用于将智能体的可行性意识与其动作生成过程对齐。CONFLICTGUARD包含两个相互耦合的组件：一个可行性验证协议，引导智能体在行动之前评估指令逻辑与GUI侧的证据；以及一个条件性的（摘要在此处截断）

    arXiv:2609.03438v1 Announce Type: new  Abstract: Graphical user interface (GUI) agents are increasingly used to execute natural-language instructions on user interfaces, yet real users may issue infeasible instructions due to benign mistakes. A reliable agent should not only know how to act, but also when not to act. In this work, we introduce CONFLICTGUI, a benchmark covering instruction-internal conflicts and instruction-GUI context conflicts to study conflict-aware termination. Our evaluation reveals severe execution-biased overcompliance: agents that perform well on feasible tasks often continue to execute blindly under conflicting instructions. To mitigate this behavior, we propose CONFLICTGUARD, an inference-time framework that aligns an agent's feasibility awareness with its action generation. CONFLICTGUARD contains two coupled components: a feasibility verification protocol that guides the agent to assess instruction logic and GUI-side evidence before acting, and a conditional 
    
[^130]: 问题本身，而非路径：大语言模型推理轨迹中的预算与难度混淆因素

    It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories

    [https://arxiv.org/abs/2609.03436](https://arxiv.org/abs/2609.03436)

    该研究提出重启控制的截断探针方法，证明大语言模型推理轨迹中所谓的“突破时刻”和“早期注定失败”大多是预算与难度造成的混淆——178个问题-模型组合中仅1个真正存在前缀特有价值，且在同等token预算下延续自身推理前缀几乎总是优于从头重启。

    

    大语言模型的推理轨迹通常被解读为包含“突破”时刻和早期可判定的命运。这两种解读都建立在缺少主张层面反事实控制的测量之上；我们提供了这两种控制。首先，一个重启控制的截断探针将“解法适合延续预算”与“前缀携带全新计算无法换取的价值”区分开来，在匹配的总生成token预算下，比较每个锚点的延续求解率与从头重启曲线。将该探针应用于178个问题-模型单元格（89个MATH问题 × 两个小型开源模型，这是一个结果盲但针对难度的队列），178个单元格中恰好只有1个作为前缀受限而存活；重启的剂量-反应关系能够区分计算饥饿型模型与能力受限型模型；并且只要匹配预算位于重启网格之内，延续模型自身的前缀总是优于从头重启（9/9）——主要是计算压缩……（摘要原文在此处截断）

    arXiv:2609.03436v1 Announce Type: cross  Abstract: Reasoning traces of large language models are widely read as containing "breakthrough" moments and early-legible fates. Both readings rest on measurements missing a counterfactual control at the level of the claim; we supply both controls. First, a restart-controlled truncation probe separates when a solution fits the continuation budget from when a prefix carries value that fresh computation cannot buy, comparing per-anchor continuation solve rates against from-scratch restart curves at matched total generated-token budget. Applied to 178 problem-model cells (89 MATH problems x two small open models, an outcome-blind but difficulty-targeted cohort), exactly 1 of 178 cells survives as prefix-limited; restart dose-response separates a compute-starved model from a capability-limited one; and wherever the matched budget lies inside the restart grid, continuing the model's own prefix beats restarting (9 of 9) -- predominantly compute compr
    
[^131]: TraveL：基于Transformer的多视角路径分布式表示学习

    TraveL: Transformer-based Multi-view Path Distributional Representation Learning

    [https://arxiv.org/abs/2609.03427](https://arxiv.org/abs/2609.03427)

    该论文提出了基于Transformer的多视角分布式表示学习框架TraveL，通过捕捉旅行者行为的多样性和路段的区域相关性，将路径与出行开始时间编码为分布式表示，从而能够解码路径上旅行者行为的可能样本。

    

    道路网络的路径表示学习（PRL）因各类与路径相关的应用而受到越来越多的研究关注。现有的PRL工作通常利用路段与路径之间的共现关系来学习一个向量作为路径表示，而没有探索旅行者行为的多样性以及路径上的区域相关性。在这项工作中，我们提出学习分布式表示，通过捕捉旅行者行为的多样性以及路段所在区域内的各种依赖关系，为路径相关应用提供有价值的信息。我们提出了一种新颖的基于Transformer的多视角分布式表示学习框架，将路径与出行开始时间一起编码为分布式表示，该表示可用于解码路径上旅行者行为的可能样本。此外，通过分析区域相关性……

    arXiv:2609.03427v1 Announce Type: cross  Abstract: Path representation learning (PRL) for road networks has received increasing research attention, due to various path-related applications. Existing works on PRL typically exploit the co-occurrence relationship among road segments and paths to learn a vector as the path representation, without exploring the varied traveler behaviors and the regional correlation on the path. In this work, we propose to learn distributional representations, which provide valuable information for use in path-related applications, by capturing the varied traveler behaviors as well as the various dependencies within regions of road segments. We propose a novel Transformer-based Multi-view Distributional Representation Learning (TraveL) framework to encode a path along with a travel starting time to a distributional representation, which can be used to decode possible samples of on-path traveler behavior. Moreover, by analyzing the regional correlation which 
    
[^132]: 文明框架：个人多智能体系统之间的主权锚定通信

    The Civilization Framework: Sovereign-Anchored Communication Between Personal Multi-Agent Systems

    [https://arxiv.org/abs/2609.03425](https://arxiv.org/abs/2609.03425)

    提出“文明框架”，以文明（人类主权者+持久账本+可互换智能体）而非单个智能体作为AI间通信的可寻址对象，通过大使馆协议以账本承诺状态取代消息送达作为事实真相，并首次识别和实验验证了AI间通信中先到信息获得不当权威的“时间权重效应”。

    

    人类是AI系统之间的传输层，在每一次跳转中都会丢失上下文。我们提出了文明框架，其可寻址方是文明而非智能体（由一个人类主权者、一个持久账本和可互换的智能体组成），以及大使馆协议——这是一种与载体无关的覆盖层：消息异步到达驻留账本端点，接收方的任何在线智能体都可处理它们，双方账本上的承诺状态（而非消息送达）才是事实真相。权威源于记忆：智能体代表其文明行事的权力受其可访问的记忆限制，并通过签名凭证外部化，与文明层面的声誉相分离。我们识别了时间权重效应——这是AI间通信中的一种危险，即先到达的信息获得了不应有的权威——并在一项预注册的1,908次试验实验中对一个前沿模型进行了测试。

    arXiv:2609.03425v1 Announce Type: cross  Abstract: Humans are the transport layer between AI systems, losing context at every hop. We present the Civilization Framework, whose addressable party is the civilization, not the agent (one human sovereign, a persistent ledger, and interchangeable agents), and the Embassy Protocol, a carrier-agnostic overlay: messages arrive asynchronously at a resident ledger endpoint, any online agent of the receiver handles them, and commitment state on both ledgers, not delivery, is ground truth. Authority derives from memory: an agent's power to act for its civilization is capped by the memory it can access and externalized through signed credentials, separate from civilization-level reputation. We identify the temporal-weight effect, a hazard in AI-to-AI communication where what arrives first acquires unearned authority, and test it in one frontier model in a preregistered 1,908-trial experiment. With verification removed, an incorrect upstream claim ar
    
[^133]: DuplexSpeechBench-IFEval：评估全双工语音代理的隐式指令遵循能力

    DuplexSpeechBench-IFEval: Evaluating Implicit Instruction Following in Full-Duplex Voice Agents

    [https://arxiv.org/abs/2609.03423](https://arxiv.org/abs/2609.03423)

    该论文提出了首个针对全双工语音代理隐式指令遵循能力的基准DSB-IFEval，通过1,038个涵盖八种助手角色和五种条件设置协议（包括人设暗示与指令冲突）的测试用例，以确定性的指令遵循分数和LLM评判的人设一致性来系统评估实时语音交互中的发言权管理与角色一致行为。

    

    全双工语音代理必须持续不断地决定何时聆听、何时给予附和、何时打断、如何处理语音重叠、何时夺取发言权以及何时让出发言权。现有基准测试大多通过显式的轮次管理指令来检验这些行为，而实际部署的代理通常是通过角色或人设（persona）来配置的，需要从中推断出恰当的对话行为。我们提出了DuplexSpeechBench-IFEval（DSB-IFEval），用于评估实时语音交互中的隐式指令遵循能力。DSB-IFEval包含1,038个测试用例，涵盖八种不同的助手角色，并评估五种指令条件设置协议：默认行为、显式行为指令、人设暗示行为、人设与规则组合条件设置，以及指令冲突。我们使用确定性的指令遵循分数（Instruction Adherence Score, IAS）来衡量实时的发言权管理能力，并使用由大语言模型评判的人设一致性分数来衡量与人设相符的内容生成。

    arXiv:2609.03423v1 Announce Type: new  Abstract: Full-duplex voice agents must continuously decide when to listen, backchannel, interrupt, handle speech overlaps, take the floor, and yield. Existing benchmarks largely test these behaviors through explicit turn-management instructions, while deployed agents are often configured through roles or personas from which the appropriate conversational behavior must be inferred. We introduce DuplexSpeechBench-IFEval (DSB-IFEval) for evaluating implicit instruction-following in real-time spoken interaction. (DSB-IFEval) comprises 1,038 test cases spanning eight diverse assistant roles and evaluates five conditioning protocols for instruction-following: default behavior, explicit behavioral instructions, persona-implied behavior, combined persona--rule conditioning, and instruction conflict. We measure real-time floor management using a deterministic Instruction Adherence Score (IAS) and persona-consistent content using LLM-judged Persona Adheren
    
[^134]: 联邦入侵检测中的隐私、鲁棒性与公平性权衡：聚合接口处的几何不可区分性

    Privacy, Robustness, and Fairness Trade-offs in Federated Intrusion Detection: Geometric Indistinguishability at the Aggregation Interface

    [https://arxiv.org/abs/2609.03420](https://arxiv.org/abs/2609.03420)

    本文揭示了联邦入侵检测中差分隐私、拜占庭鲁棒性与类别公平三大需求并非可独立组合，并提出“几何不可区分性”概念，用以解释隐私噪声导致的客户端更新分散会削弱鲁棒聚合对少数类攻击信号的保留能力。

    

    联邦学习使网络入侵检测能够在无需集中敏感流量数据的情况下进行注重隐私的协作，然而其在实际运行环境中的部署必须同时满足三个相互竞争的需求：形式化的差分隐私保证、对拜占庭对抗性参与者的容忍，以及对严重不平衡攻击类别的可靠检测覆盖。现有文献将这些属性视为可独立组合的，而本文在理论和实证两方面对这一假设提出了挑战。本文研究了这些需求在类别不平衡的联邦网络入侵检测系统（NIDS）中如何相互作用，并引入“几何不可区分性”作为概念视角，用以刻画一种情形：隐私引起的客户端更新分散会使得少数类信号更难被鲁棒聚合机制所保留。以UNSW-NB15数据集为案例研究，我们评估了DP-SGD与坐标级中值聚合相结合的方法……（摘要内容在此处截断）

    arXiv:2609.03420v1 Announce Type: cross  Abstract: Federated learning enables privacy-conscious collaboration for network intrusion detection without centralizing sensitive traffic data, yet its deployment in operational environments must simultaneously satisfy three competing requirements: formal differential privacy guaranties, tolerance to Byzantine-adversarial participants, and reliable detection coverage across severely imbalanced attack categories. Existing literature treats these properties as independently composable, an assumption that this paper challenges both theoretically and empirically. In this paper, we study how these requirements interact in class-imbalanced federated NIDS and introduce geometric indistinguishability as a conceptual lens for a regime in which privacy-induced dispersion in client updates can make minority-class signals harder for robust aggregation to preserve. Using UNSW-NB15 as a case study, we evaluate DP-SGD combined with coordinate-wise median und
    
[^135]: Dude：一种用于论文-代码差异检测的双检测多智能体系统

    Dude: A Dual-Detection Multi-Agent System for Paper-Code Discrepancy Detection

    [https://arxiv.org/abs/2609.03416](https://arxiv.org/abs/2609.03416)

    提出了首个用于论文-代码差异检测的双检测多智能体系统Dude，通过粒度对齐协商机制和两阶段显著性过滤机制，有效解决了论文语言与代码语言粒度不对称导致的误报问题。

    

    随着研究论文提交数量的增长超过了人工审阅能力，基于大语言模型（LLM）的论文-代码差异检测日益受到关注。然而，现有单智能体LLM范式的上下文容量有限且差异检测视角单一，导致差异检测的召回率表现不佳。本文提出了Dude，首个用于论文-代码差异检测的双检测多智能体系统。我们发现，论文语言与代码语言之间的粒度不对称性给差异检测的多智能体系统设计带来了过度解读和过度报告的挑战，导致误报增多。为解决这一问题，我们在Dude中提出了粒度对齐协商机制和两阶段显著性过滤机制，有效防止智能体错误报告差异。在真实世界论文-代码差异数据集上的实验结果表……

    arXiv:2609.03416v1 Announce Type: new  Abstract: LLM-empowered paper-code discrepancy detection has received growing concern since the scaling of research submissions exceeds the manual review capability. However, the limited context capacity and one-sided discrepancy detection of existing single-agent LLM paradigms lead to an inferior recall performance in detecting discrepancies. In this paper, we propose Dude, the first Dual-Detection Multi-Agent System for paper-code discrepancy detection. We discover that the granularity asymmetry of the paper-language and code-language introduces over-interpretation and over-reporting challenges in a multi-agent system design for discrepancy detection, resulting in increasing false positives. To address this, we propose a granularity-aligned negotiation and a two-stage salience-filtering mechanism in Dude, which effectively prevents agents from falsely reporting discrepancies. Experimental results in real-world paper-code discrepancy datasets sho
    
[^136]: StrixAE：面向真实场景中复杂失真耦合的音频增强智能代理

    StrixAE: An Intelligent Agent for Audio Enhancement under Complex Distortion Coupling in Real-World Scenarios

    [https://arxiv.org/abs/2609.03414](https://arxiv.org/abs/2609.03414)

    本文提出基于多模态大语言模型的智能代理StrixAE，通过CoT监督微调与音频感知强化学习两阶段训练，协调多个音频增强与个性化模型，有效应对真实场景中复杂失真耦合与个性化音频增强的双重挑战。

    

    真实场景中的音频增强涉及复杂的失真耦合，并且需要个性化的增强处理。现有的解决方案难以同时兼顾这两方面。为了提高鲁棒性并在此类场景中实现自主运行，我们提出了StrixAE，一个基于多模态大语言模型（MLLM）的智能代理。StrixAE利用MLLM作为控制器，协调多个音频增强和个性化模型。为了进一步增强系统的鲁棒性、减少伪影并提高在多样化真实场景中的泛化能力，StrixAE通过两阶段过程进行训练：第一阶段，在AcoustBench上进行思维链（CoT）监督微调，以奠定基础推理和工具调用能力；第二阶段，音频感知强化学习（APRL），这是一种专门为音频恢复流水线设计的奖励机制，联合优化格式有效性、结构连贯性和感知质量。与通用的强化学习微调不同……

    arXiv:2609.03414v1 Announce Type: cross  Abstract: Audio enhancement in real-world scenarios involves complex distortion couplings and requires personalized enhancement. Existing solutions struggle to address both simultaneously. To improve robustness and enable autonomous operation in such scenarios, we propose StrixAE, an agent based on a multimodal large language model (MLLM). StrixAE leverages the MLLM as a controller to coordinate multiple audio enhancement and personalization models. To further enhance system robustness, reduce artifacts, and improve generalization across diverse real-world scenarios, StrixAE is trained through a two-stage process: first, CoT supervised fine-tuning on AcoustBench to ground basic reasoning and tool invocation; second, Audio Perception Reinforcement Learning (APRL), a reward design specifically tailored for audio restoration pipelines that jointly optimizes format validity, structural coherence, and perceptual quality. Unlike generic RL fine-tuning
    
[^137]: 困于故事之中：多轮大语言模型对话中的叙事俘获

    Caught in the Story: Narrative Captivity in Multi-turn LLMs Conversation

    [https://arxiv.org/abs/2609.03407](https://arxiv.org/abs/2609.03407)

    本文揭示了大语言模型在多轮道德咨询中的一种新失败模式——“叙事俘获”：当用户仅提供单方面的自我辩解式叙述时，模型会将其视为完整事实并认同叙述者的立场，而不会主动寻求缺失的对立视角。

    

    人们越来越多地求助于大型语言模型（LLM）获取日常建议，这使得充满道德争议的人际问题成为一个现实的道德咨询场景。以往的大多数工作都通过单轮判断或充满压力的反驳来研究这一场景，这些假设与现实中人们寻求指导的方式并不相符。这些假设使得我们不清楚：在没有明确对立立场的情况下，仅凭叙述本身是否能在多轮道德咨询中改变模型的判断。然而，现实世界的道德冲突对话往往会引出一方的自我辩解式陈述，这种陈述可能在多轮对话中逐步展开并造成信息不对称。我们提出了“叙事俘获”这一失败模式，即模型将一方未受质疑的单方面陈述视为完整信息，直接认同叙述者的解读，而不去寻求缺失的其他视角。为了衡量这一现象，我们构建了一个包含5,078个人际道德冲突情境的基准数据集。

    arXiv:2609.03407v1 Announce Type: new  Abstract: People increasingly turn to large language models (LLMs) for everyday advice, making ethically charged interpersonal problems a practical moral-advisory context. Most prior work has studied this context through single-turn judgments or pressure-laden rebuttals, assumptions that poorly match how guidance is sought in real-world contexts. These assumptions leave unclear whether narration alone, without an explicit opposing position, can shift model judgments during multi-turn moral consultation. Yet real-world moral-conflict conversation often elicits one party's self-justifying account, which can unfold over multiple turns and create information asymmetry. We introduce \textbf{narrative captivity}, a failure mode in which a model treats an unopposed one-sided account as complete and aligns with the narrator's interpretation without seeking missing perspectives. To measure this phenomenon, we build a benchmark of $5{,}078$ interpersonal-co
    
[^138]: 一种提示工程方法：在通用AI助教中实现可扩展、灵活且实时的混合微观级个性化

    A Prompt-Engineering Approach to Develop Scalable, Flexible, and Real-Time Hybrid Micro-Level Personalization in a General Purpose AI Teaching Assistant

    [https://arxiv.org/abs/2609.03402](https://arxiv.org/abs/2609.03402)

    本研究提出一种基于提示工程的个性化框架，利用六个学习者维度画像和布鲁姆分类法认知分析，在无需模型再训练的情况下，使通用LLM/RAG型AI助教实现可扩展、灵活且实时的微观级个性化响应。

    

    基于大语言模型（LLM）的人工智能（AI）助教能够提供可扩展的教育支持，但其个性化程度往往有限。本研究提出了一种基于提示工程的框架，用于对基于LLM/RAG的通用AI助教（如Jill Watson）实现跨学科、跨课程的个性化。该框架通过六个学习者特定维度来调整响应：自我评估、抽象偏好、详略偏好、感知取向、信息处理风格和理解水平，从而形成96种不同的学习者画像。此外，还运用布鲁姆分类法（Bloom's Taxonomy）对学生提问进行分析，以在交互层面估计认知复杂度。学习者属性与认知评估被编码到结构化提示中，从而在不需重新训练模型的情况下对LLM进行调节。该框架通过基于NLP指标和人类受试者的实验进行评估（摘要原文在此处截断）。

    arXiv:2609.03402v1 Announce Type: new  Abstract: Artificial intelligence (AI) teaching assistants powered by large language models (LLMs) offer scalable educational support but often provide limited personalization. This study presents a prompt-engineering-based framework for personalizing general-purpose LLM/RAG-based AI teaching assistants such as Jill Watson across academic disciplines and courses. The framework adapts responses using six learner-specific dimensions: self-assessment, abstraction preference, verbosity preference, perceptual orientation, information processing style, and level of understanding, yielding 96 distinct learner profiles. Student queries are additionally analyzed using Bloom's Taxonomy to estimate cognitive complexity at the interaction level. Learner attributes and cognitive assessments are encoded in structured prompts that condition the LLM without requiring model retraining. The framework is evaluated through experiments using NLP metrics and a human st
    
[^139]: 多维随机特征方法的谱收敛性

    Spectral Convergence of Random Feature Method in Multiple Dimensions

    [https://arxiv.org/abs/2609.03401](https://arxiv.org/abs/2609.03401)

    本文证明了随机特征方法在多维情形下对Sobolev、Gevrey、超解析及带限函数类目标的谱收敛性，并给出了收敛速率随目标正则性从超指数到代数的刻画，同时建立了强形式和弱形式RFM离散化的抽象误差估计。

    

    我们首先证明了随机特征方法（RFM）对Sobolev、Gevrey、超解析及带限函数类中多维目标的谱收敛性。该分析在由核积分算子生成的插值尺度上建立了一般的高概率逼近估计。在仅由采样特征确定的单个事件上，一个随机空间即可逼近给定源球中的每一个目标函数；此外，对于每个目标，单一系数向量所定义的逼近器可在所有允许的误差范数下同时达到谱精度。对于正则性自适应的频率分布以及增长频率窗口上的均匀分布，所得收敛速率介于超指数速率与代数速率之间，具体取决于目标函数的正则性。其次，我们为强形式和弱形式的RFM离散化建立了抽象误差估计，从而将前述逼近界转换为……（摘要在此处截断）

    arXiv:2609.03401v1 Announce Type: cross  Abstract: We first prove spectral convergence of the random feature method (RFM) for multidimensional targets in Sobolev, Gevrey, ultra-analytic, and bandlimited classes. The analysis establishes general high-probability approximation estimates in the interpolation scale generated by a kernel integral operator. On a single event determined only by the sampled features, one random space approximates every target in a prescribed source ball; moreover, for each target, a single coefficient vector defines an approximant that attains spectral accuracy simultaneously in all admissible error norms. For both regularity-adapted frequency distributions and uniform distributions on growing frequency windows, the resulting rates range from super-exponential to algebraic, depending on the regularity of the target. Second, we establish abstract error estimates for strong- and weak-form RFM discretizations, thereby converting the preceding approximation bounds
    
[^140]: TabScope：面向表格问答的问题自适应范围选择

    TabScope: Question-Adaptive Scope Selection for Table Question Answering

    [https://arxiv.org/abs/2609.03395](https://arxiv.org/abs/2609.03395)

    该论文提出TabScope框架，通过操作感知的表格分解和问题类型预测，在大语言模型表格问答中动态选择局部子表推理或全表推理，显著提升长表格问答的准确率，并贡献了基于真实世界长表格的SLQA评测基准。

    

    大型语言模型（LLMs）在表格问答任务上展现出强大的性能，但其准确率往往随着表格规模的增大而下降。我们发现这种性能下降在不同问题类型之间并不均匀：对定位敏感的问题尤其容易受到表格中无关内容的干扰，而需要更广泛证据的问题则仍可能受益于全表推理。基于这一观察，我们提出了一个问答自适应框架，可在局部推理与全表推理之间进行动态选择。该框架通过操作感知的表格分解来构建针对特定问题的子表，并利用预测出的问题类型来确定合适的推理模式。此外，我们进一步引入了用于评估证据选择的银标准参考子表，并构建了SLQA——一个基于真实世界长表格的基准数据集。在WikiTQ和SLQA上的实验表明，定位方法对查找类问题尤其有效……

    arXiv:2609.03395v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong performance on table question answering, yet their accuracy often degrades as table size increases. We find that this degradation is not uniform across question types. Localization-sensitive questions are particularly affected by irrelevant table content, while questions requiring broader evidence may still benefit from full-table reasoning. Based on this observation, we propose a question-adaptive framework that dynamically selects between localized and full-table reasoning. The framework constructs question-specific sub-tables through operation-aware table decomposition and uses the predicted question type to determine the appropriate reasoning mode. We further introduce silver reference sub-tables for evaluating evidence selection and construct SLQA, a benchmark based on real-world long tables. Experiments on WikiTQ and SLQA show that localization is particularly effective for lookup an
    
[^141]: 探索对比语言-图像预训练在多源遥感数据中的潜力

    Exploring the Potential of Contrastive Language-Image Pre-training for Multi-Source Remote Sensing Data

    [https://arxiv.org/abs/2609.03391](https://arxiv.org/abs/2609.03391)

    提出OmniRSCLIP端到端对比学习框架，通过光谱-空间基分解（SSBD）在不破坏CLIP预训练视觉知识的前提下，将其从固定RGB输入扩展至SAR、多光谱、高光谱等多源遥感传感器数据。

    

    对比语言-图像学习（CLIP）已成为遥感视觉-语言理解的关键范式。然而，现有的遥感对比学习方法大多建立在面向RGB的CLIP架构之上，难以利用SAR（合成孔径雷达）、多光谱成像（MSI）和高光谱成像（HSI）等异构传感器。为解决这一局限，我们提出了OmniRSCLIP，一个面向遥感视觉-语言建模、支持多源传感器输入的端到端对比学习框架。其核心思想是在不破坏预训练视觉知识的前提下，将CLIP扩展到其固定的RGB输入接口之外。为此，OmniRSCLIP引入了光谱-空间基分解（SSBD）方法，将任意通道的自适应建模为基重组问题：预训练的CLIP图像块嵌入提供可迁移的空间基，而以波长为条件的系数则张成传感器特定的特征空间……（原文摘要在此处截断）

    arXiv:2609.03391v1 Announce Type: cross  Abstract: Contrastive language-image learning (CLIP) has become a key paradigm for remote sensing vision-language understanding. However, existing remote sensing contrastive learning methods are mostly built on RGB-oriented CLIP architectures, making it difficult to exploit heterogeneous sensors such as SAR, multi-spectral imaging (MSI), and hyperspectral imaging (HSI). To address this limitation, we propose OmniRSCLIP, an end-to-end contrastive learning framework that supports multi-source sensor inputs for remote sensing vision-language modeling. The key idea is to extend CLIP beyond its fixed RGB input interface without breaking the pretrained visual knowledge. To this end, OmniRSCLIP introduces Spectral-Spatial Basis Decomposition (SSBD), which formulates arbitrary-channel adaptation as a basis recomposition problem: pretrained CLIP patch embeddings provide transferable spatial bases, while wavelength-conditioned coefficients span sensor-spe
    
[^142]: 新鲜记忆，过时计划：面向分布式LLM智能体记忆的依赖范围验证

    Fresh Memory, Stale Plans: Dependency-Scoped Validation for Distributed LLM-Agent Memory

    [https://arxiv.org/abs/2609.03340](https://arxiv.org/abs/2609.03340)

    该论文提出PlanFence协议，通过让计划引用其使用的确切公共记录、并让执行者仅验证影响待处理动作的相关记录，解决了分布式LLM智能体中“过时计划执行”问题，在检测到计划失效时及时重规划或阻塞，避免基于过时计划执行无效动作。

    

    分布式LLM智能体团队可以读取最新的共享事实，却仍然基于过时的计划行动。规划者可能根据需求 $r_3$ 推导出一个动作，另一个智能体可能提交了 $r_4$，而执行者可能收到 $r_4$ 却没有替换基于 $r_3$ 推导的计划。我们将这种现象称为“过时计划执行”：状态的新鲜度并不能证明授权动作的计划仍然有效。我们提出了PlanFence，一种依赖范围的动作验证协议。计划引用其使用的确切公共记录，执行者仅验证可能影响待处理外部动作的记录，当验证不完整时重新规划一次或进行阻塞。在30个包含计划后修订的受控实时工作流中，仅依赖新鲜度的执行者在每个任务中都基于过时计划行动，而PlanFence完成了所有任务且未执行任何无效动作。受控重放揭示了两个条件边界：主动同步产生较低的协调（此处摘要被截断）

    arXiv:2609.03340v1 Announce Type: new  Abstract: Distributed LLM-agent teams can read the latest shared facts and still act on an obsolete plan. A planner may derive an action from requirement $r_3$, another agent may commit $r_4$, and an executor may receive $r_4$ without replacing the plan derived from $r_3$. We call this \emph{stale-plan execution}: state freshness does not establish that the plan authorizing an action remains valid. We introduce PlanFence, a dependency-scoped action-validation protocol. Plans cite the exact public records they used, and an executor validates only the records that can affect the pending external action, replanning once or blocking when validation is incomplete. In 30 controlled live workflows with a post-plan revision, a freshness-only executor acts on the obsolete plan in every task, whereas PlanFence completes all tasks without an invalid action. Controlled replay reveals two conditional boundaries: proactive synchronization yields lower coordinat
    
[^143]: FlowBalance：基于验证器的策略内推理经验自我改进方法

    FlowBalance: Verifier-Grounded Self-Improvement from On-Policy Reasoning Experience

    [https://arxiv.org/abs/2609.03241](https://arxiv.org/abs/2609.03241)

    FlowBalance提出一种以终端验证器的组优势来校准同模型自引导分数的自我改进方法，通过在正优势轨迹上保留、负优势轨迹上反转、无结果偏好时禁用引导，实现更稳定的推理能力自我提升。

    

    推理模型可以从自身的策略内经验中改进，但这一内部循环十分脆弱：终端验证器提供可靠却稀疏的监督信号，而同模型的密集引导可能会强化错误的自信，或将学习过度集中于狭窄的解题模式。我们提出了FlowBalance，一种以验证器为基础的自我改进方法，它学习完整回答上的归一化分布。对于每条策略内轨迹，同一策略的冻结训练时视图利用特权上下文产生token级别的对数概率增益，这些增益被聚合为轨迹级别的自引导分数。FlowBalance使用验证器得出的组优势来校准该分数：在正优势轨迹上保留引导，在负优势轨迹上反转引导，当rollout组未提供结果偏好时则禁用引导。由此得到的能量函数对参考策略进行指数级重加权……（摘要在此处截断）

    arXiv:2609.03241v1 Announce Type: cross  Abstract: A reasoning model can improve from its own on-policy experience, but this inner loop is fragile: terminal verifiers provide reliable yet sparse supervision, while dense same-model guidance can reinforce false confidence or overconcentrate learning on a narrow solution mode. We introduce FlowBalance, a verifier-grounded self-improvement method that learns a normalized distribution over complete responses. For each on-policy trajectory, a frozen training-time view of the same policy uses privileged context to produce token-level log-probability gains, which are aggregated into a trajectory-level self-guidance score. FlowBalance calibrates this score with the verifier-derived group advantage: guidance is retained on positive-advantage trajectories, reversed on negative-advantage trajectories, and disabled when the rollout group provides no outcome preference. The resulting energy exponentially reweights a reference policy, and profiled tr
    
[^144]: 推测性宏提交：加速工具使用智能体

    Speculative Macro Commit for Faster Tool-Using Agents

    [https://arxiv.org/abs/2609.03236](https://arxiv.org/abs/2609.03236)

    该论文提出推测性宏提交（SMC）机制，通过小模型在隔离环境快照上预执行从训练轨迹挖掘的多动作骨架，当大模型的动作与之匹配时直接将预执行结果提交到官方轨迹，从而加速工具使用智能体的端到端运行时间。

    

    使用工具的大语言模型智能体不仅在模型推理上耗费时间，还在串行的动作-观察轮次中消耗大量时间，其中每次工具调用、环境转换和观察都可能延迟后续决策。我们提出了**推测性宏提交**（Speculative Macro Commit, SMC），这是一种面向双层智能体系统的运行时机制：一个大型权威执行者模型负责生成官方轨迹，而一个更快的推测性起草者模型则在隔离的环境快照上持续预测并执行未来的动作链。SMC从训练轨迹中挖掘重复出现的多动作骨架，并将其存储在宏库中，用于在运行时与起草者预测的动作链进行匹配。当执行者的下一个工具调用与第一个起草的动作匹配时，SMC将剩余的预执行起草步骤连同其观察结果一起提交到官方轨迹。使用Qwen3.5-27B INT4作为权威执行者模型，Qwen3.5-4B作为推测性（原文在此处截断）

    arXiv:2609.03236v1 Announce Type: new  Abstract: Tool-using LLM agents spend wall-clock time not only on model inference but also in serial action--observation turns, where each tool call, environment transition, and observation can delay subsequent decisions. We introduce \textbf{Speculative Macro Commit} (SMC), a runtime mechanism for a two-tier agent system: a large authoritative actor model produces the official trajectory, while a faster speculative drafter model continuously predicts and executes future action chains on an isolated environment snapshot. SMC mines recurring multi-action skeletons from training traces and stores them in a macro library used to match against action chains predicted by the drafter at runtime. When the actor's next tool call matches the first drafted action, SMC commits the remaining pre-executed draft steps, together with their observations, to the official trajectory. Using Qwen3.5-27B INT4 as the authoritative actor model and Qwen3.5-4B as the spec
    
[^145]: MasterControl：每次都能稳定命中

    MasterControl Seventeen Every Time

    [https://arxiv.org/abs/2609.03209](https://arxiv.org/abs/2609.03209)

    论文提出一种受管治的企业分析架构——语言模型仅负责理解问题意图，由确定性策略执行预先批准的分析程序，在440次运行中该策略执行方法以110/110全部满足“答案与证据”契约，而三个8B模型运行时自主规划的330次中无一达标，且固定规则确保结果完全可复现。

    

    我们研究了一种受管治的企业分析方法：由语言模型负责理解问题，而确定性策略选择并运行预先批准的分析程序，该程序同时返回结果与证据。我们证明，这种限制在定义好的分析类别内仍可保持表达能力，涵盖关系操作以及聚合、比较、窗口、排名和相似度计算。固定的语义、策略、数据和执行规则还使结果可复现。在共440次运行中，三个8B模型在运行时生成SQL并选择工具，而Qwen3-8B仅负责解释意图、由策略执行已批准的程序。在所有测试数据集上，330次运行时自主规划中没有任何一次完全满足“答案与证据”契约；而策略执行的分析器在110次中全部匹配（110/110）。这是一个特定配置下的结果，并不能证明运行时智能体在其他设计下无法成功。

    arXiv:2609.03209v1 Announce Type: new  Abstract: We study a governed approach to enterprise analytics: a language model interprets the question, while deterministic policy selects and runs a pre-approved analytical program that returns both results and evidence. We show that this restriction can remain expressive within a defined analytical class, using relational operations plus aggregation, comparison, windows, ranking, and similarity. Fixed meaning, policy, data, and execution rules also make results replayable. Across 440 runs, three 8B models generated SQL and selected tools at runtime, while Qwen3-8B interpreted intent only and policy executed the approved program. None of 330 runtime-planning episodes matched the full answer-and-evidence contract across all test datasets; the policy-executed analyzer matched 110 of 110. This is a configuration-specific result, not evidence that runtime agents cannot succeed under other designs.
    
[^146]: 通过系统性监测与评估失控AI演进进程降低人工智能灾难性风险

    Reducing Catastrophic Risk from AI with Systematic Monitoring and Evaluation of Rogue AI Progression

    [https://arxiv.org/abs/2609.03189](https://arxiv.org/abs/2609.03189)

    该论文借鉴网络安全方法论，提出了一个结构化行为指标框架，通过在AI能力与行为多个维度上设定明确的指标和阈值，实现对失控AI向灾难性威胁演进的系统性监测与评估。

    

    本文提出了一个结构化的行为指标框架，这些指标可能预示着人工智能系统正在朝潜在灾难性威胁的方向演进。我们采用务实的方法，借鉴了网络安全和国家安全领域已确立的方法论。通过在人工智能能力和行为的多个维度上建立明确的度量指标、行为标志和阈值，该框架使研究人员和政策制定者能够实施基于证据的监测协议。

    arXiv:2609.03189v1 Announce Type: cross  Abstract: This article presents a structured framework of behavioral indicators that may signal progression toward potentially catastrophic threats from artificial intelligence systems. We adopt a pragmatic approach, inspired by established methodologies in cybersecurity and national security. By establishing clear metrics, indicators, and thresholds across multiple dimensions of AI capability and behavior, this framework enables researchers and policymakers to implement evidence-based monitoring protocols.
    
[^147]: SHELF：一个用于多任务书目基准测试的合成测试框架

    SHELF: A Synthetic Harness for Multi-Task Bibliographic Benchmarking

    [https://arxiv.org/abs/2609.03047](https://arxiv.org/abs/2609.03047)

    SHELF是一个基于美国国会图书馆词表生成6万余篇合成文档的Python系统，为图书馆和档案馆的书目工作提供了涵盖分类、聚类、检索等多任务的系统性基准测试框架。

    

    图书馆和档案馆在人员和计算预算有限的情况下管理着大量馆藏，然而现有的常见基准测试并未系统地检验其书目工作。他们需要了解哪些方法适用于自己的任务，以及运行这些方法需要什么条件。SHELF（用于评估LLM适应性的合成测试框架，Synthetic Harness for Evaluating LLM Fitness）填补了这一空白。它是一个Python系统，能够将带标签的分类法、编写规范和生成预算转化为受控的基准数据和评估任务。首个发布版本包含62,899篇基于美国国会图书馆词表、由模型生成的文档，涵盖分类、聚类、检索、成对分类和指令检索等任务。我们比较了TF、TF-IDF、BM25、流行的编码器模型，以及仅在主题分类任务上测试的零样本解码器；每种方法仅出现在支持它的任务上。主题分类的准确率达到0.8887，而体裁-形式分类仅达到0.2605……

    arXiv:2609.03047v1 Announce Type: cross  Abstract: Libraries and archives manage large collections with limited staff and computing budgets, yet common benchmarks do not systematically test their bibliographic work. They need to know which methods work for their tasks and what those methods require to run. SHELF, the Synthetic Harness for Evaluating LLM Fitness, addresses this gap. It is a Python system that turns labelled taxonomies, writing specifications, and a generation budget into controlled benchmark data and evaluation tasks. This first release contains 62,899 model-written documents based on Library of Congress vocabularies, with tasks for classification, clustering, retrieval, pair classification, and instruction retrieval. We compare TF, TF-IDF, BM25, popular encoders, and, on subject classification only, zero-shot decoders; each method appears only on tasks that support it. Subject classification reaches 0.8887, while genre-form classification reaches only 0.2605, and sever
    
[^148]: ObserverBench：面向干预与控制的机制性估计测试

    ObserverBench: Testing Mechanistic Estimates for Intervention and Control

    [https://arxiv.org/abs/2609.03026](https://arxiv.org/abs/2609.03026)

    提出 ObserverBench 基准框架，将估计精度与所选干预行动造成的损失分开评估，用以检验机制可解释性中的内部估计器是否足以胜任干预、控制与安全任务，并证明平均准确的估计并不必然带来更优的行动。

    

    机制可解释性正越来越多地被用于指导诸如激活引导、电路移除和安全监控等干预措施。然而，一个平均意义上准确的内部估计，仍可能选出糟糕的行动。我们提出了 ObserverBench，一个用于检验内部估计器（即“观察者”）是否足以胜任其所指导的干预、控制或安全任务的基准框架。每个任务固定模型、信息边界、允许的行动、决策规则、留出测试用例和损失函数。该基准将估计精度与所选行动造成的损失分开报告。理论与实验表明为何两者都必不可少：在闭环控制中，观察者误差在起始点以及允许的干预所能到达的方向上都会产生影响。在 GPT-2-small 和 Qwen2.5-7B 的电路干预任务上，成对观察者能更准确地预测未见过的效应，但并不总是能选出更好的行动。

    arXiv:2609.03026v1 Announce Type: cross  Abstract: Mechanistic interpretability is increasingly used to guide interventions such as activation steering, circuit removal, and safety monitoring. Yet an internal estimate that is accurate on average can still choose a poor action.   We present ObserverBench, a benchmark framework for testing whether an internal estimator---an observer---is adequate for the intervention, control, or safety task it directs. Each task fixes the model, information boundary, allowed actions, decision rule, held-out cases, and loss. The benchmark reports estimation accuracy separately from the loss caused by the chosen action.   Theory and experiments show why both are needed. In closed-loop control, observer errors matter at the starting point and along directions the allowed intervention can reach. On circuit-intervention tasks in GPT-2-small and Qwen2.5-7B, pairwise observers predict unseen effects more accurately without always choosing better actions; obser
    
[^149]: 蒸馏之前先验证：面向在线策略蒸馏的提示级教师门控

    Verify Before You Distill: Prompt-Level Teacher Gating for On-Policy Distillation

    [https://arxiv.org/abs/2609.02998](https://arxiv.org/abs/2609.02998)

    该论文提出教师门控在线策略蒸馏（TGOPD），通过经验证器评分的教师探测在提示级别先验证教师模型的可靠性，将可靠提示路由到密集OPD监督、不可靠提示路由到基于验证器的GRPO，从而避免“自信但错误”的教师模型诱导误导性更新。

    

    在线策略蒸馏（OPD）通过在学生模型自身的生成结果上提供来自冻结教师模型的密集token级监督来加速后训练过程。原始的OPD在所有提示上均匀地应用这种监督，而不检查教师模型对每个提示是否可靠。由于反向KL散度具有模式寻求特性，一个自信但错误的教师模型可能导致强烈却具有误导性的更新。分布性代理指标（如熵或教师-学生似然一致性）只能衡量不确定性或一致性，但无法直接验证结果的正确性。我们提出了教师门控在线策略蒸馏（TGOPD），其核心原则是在接受密集监督之前，应在提示级别验证教师模型的可靠性。TGOPD通过一小组经验证器评分的教师探测样本估计教师可靠性，并将每个提示专门路由到密集OPD（当可靠性检查通过时）或基于验证器的GRPO（当检查不通过时）。在4B和3...（摘要内容不完整）

    arXiv:2609.02998v1 Announce Type: cross  Abstract: On-policy distillation (OPD) accelerates post-training by providing dense token-level supervision from a frozen teacher on the student's own rollouts. Vanilla OPD applies this supervision uniformly across prompts, without checking whether the teacher is reliable for each prompt. Because reverse KL is mode-seeking, a confidently wrong teacher can induce a strong yet misleading update. Distributional proxies, such as entropy or teacher-student likelihood agreement, measure uncertainty or agreement but do not directly verify outcome correctness. We introduce Teacher-Gated On-Policy Distillation (TGOPD), built on the principle that teacher reliability should be verified at the prompt level before dense supervision is admitted. TGOPD estimates reliability from a small set of verifier-scored teacher probes and routes each prompt exclusively to dense OPD when the reliability check passes or to verifier-grounded GRPO otherwise. Across 4B and 3
    
[^150]: 评估图神经网络在海事航海图变更重要性分类中的应用

    Evaluating Graph Neural Networks for Change-Criticality Classification in Maritime Navigation Charts

    [https://arxiv.org/abs/2609.02996](https://arxiv.org/abs/2609.02996)

    该论文提出将电子航海图数据集表示为图结构（空间对象为节点、空间与语义关系为边），并将新旧航海图间变更的重要性分类构建为图对分类问题，以评估不同图神经网络配置在此任务上的表现。

    

    图神经网络（GNN）是一类适用于图结构数据学习的神经网络。将其应用于空间数据是一种自然的延伸，然而，对于哪种消息传递操作、架构配置和图表示最适合用于对电子航海图（ENC，即用于海洋导航的地理空间矢量数据集）中对象变更进行分类，目前仍不明确。维护这些数据集是一项挑战，而根据对象对航行安全的重要性对ENC中对象的变更进行分类尤为重要。在此，我们提出将这些矢量导航数据集表示为图结构，其中空间对象作为节点，它们之间的空间和语义关系形成边。我们将旧的ENC数据集和新的ENC数据集分别编码为一对图，并将该任务构建为一个图对分类问题。

    arXiv:2609.02996v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) are a class of neural networks suitable for learning on graph-structured data. Their application to spatial data is a natural extension, however its relatively unclear which message-passing operations, architectural configurations, and graph representation is best suited for classifying changes to objects in electronic navigational charts (ENCs)--geospatial vector datasets used for marine navigation. Maintaining these datasets is a challenge, and categorizing changes to objects in the ENC based on their significance to navigational safety is of particular importance. Here, we propose to represent these vector navigation datasets as a graph structure where the spatial objects serve as nodes and their spatial and semantic relationships form edges. We encode both the old ENC dataset and new ENC dataset into a pair of graphs and frame the task as a graph-pair classification problem. Building on this representat
    
[^151]: 面向参与式民主的偏好推断：迈向以集体为中心的评估

    Toward Collective-Centric Evaluation of Preference Inference for Participatory Democracy

    [https://arxiv.org/abs/2609.02990](https://arxiv.org/abs/2609.02990)

    该论文提出了一个以集体为中心的评估框架，对参与式民主平台中现有偏好推断方法进行了基准测试，揭示了这些方法并非中立，可能人为地放大、抑制或重排集体支持模式，从而重塑对商议结果的解读。

    

    为了扩大集体决策的规模，Polis 和 Remesh 等参与式民主平台使数千名参与者能够进行在线商议。然而，在这种规模下，参与者无法审阅其他人提交的每一条意见，由此产生高度稀疏的投票数据，这些数据无法准确反映共识、冲突以及少数派支持的模式。因此，平台日益依赖偏好推断模型来预测缺失的投票。然而，这种自动化并非中立：推断出的偏好可能人为地放大、抑制或重排现有的支持模式，最终重塑对商议结果的解读方式。更广泛地说，我们对现有偏好推断方法如何影响集体偏好格局尚缺乏系统性的理解。为填补这一空白，我们在该背景下对几种现有的偏好推断方法进行了基准测试，并超越了以个体预测准确性为中心的传统用户导向评估。

    arXiv:2609.02990v1 Announce Type: cross  Abstract: To scale up collective decision-making, participatory democracy platforms such as Polis and Remesh enable online deliberation among thousands of participants. However, at this scale, participants cannot review every opinion submitted by others, producing highly sparse voting data that misrepresent patterns of consensus, conflict, and minority support. Platforms therefore increasingly rely on Preference Inference (PI) models to predict missing votes. Yet this automation is not neutral: inferred preferences can artificially amplify, suppress, or reorder existing patterns of support, ultimately reshaping how the outcomes of a deliberation are interpreted. More generally, we lack a systematic understanding of how existing PI methods affect the collective preference landscape. To address this gap, we benchmark several existing PI approaches in this context. Moving beyond conventional user-centric evaluations centered on the accuracy of indi
    
[^152]: 人工智能驱动的新型实用英语教材的结构与实施

    Structure and Implementation of New Practical English Textbooks Driven by Artificial Intelligence

    [https://arxiv.org/abs/2609.02981](https://arxiv.org/abs/2609.02981)

    本文提出人工智能驱动实用英语教材的五层架构，实验证明该系统可将学生单元完成准确率提升至84.9%、口语成绩提高10.8分，并减少教师31.6%的批改时间。

    

    人工智能正在改变应用英语教材的形态，使其从固定的纸质序列转变为能够诊断学习者、推荐任务并提供形成性反馈的自适应学习系统。本文研究了一种人工智能驱动的新型实用英语教材的结构与应用，提出了一个五层架构：知识图谱、学习者画像、任务生成、反馈编排和教师端治理。该原型系统在186名非英语专业本科生中进行了为期八周的教学测试。与静态数字教材相比，该系统将单元完成准确率从72.4%提高到了84.9%，口语任务平均分提高了10.8分，教师的批改时间减少了31.6%。因此，人工智能驱动的教材能够在保持课程稳定性的同时，提供个性化学习路径、丰富的练习材料……（摘要在此处被截断）

    arXiv:2609.02981v1 Announce Type: new  Abstract: Artificial intelligence is changing the form of applied English materials from fixed paper sequences to adaptive learning systems that can diagnose learners, recommend tasks, and provide formative feedback. This paper studies the structure and application of a new practical English textbook driven by artificial intelligence. A five-layer architecture is proposed: knowledge mapping, learner profiling, task generation, feedback orchestration, and teacher-side governance. A prototype was tested on 186 non-English-major undergraduates for eight weeks of teaching. Compared with a static digital textbook, the proposed system increased the unit completion accuracy from 72.4% to 84.9%, raised the average score for speaking tasks by 10.8 points, and reduced the teacher's correction time by 31.6%. Therefore, an AI-driven textbook can maintain the stability of the curriculum while providing personalised learning paths, rich practice materials and t
    
[^153]: 基于联邦图学习的LLM多智能体系统隐私保护拓扑引导安全方法

    Privacy-Preserving Topology-Guided Safety for LLM-Based Multi-Agent Systems via Federated Graph Learning

    [https://arxiv.org/abs/2609.02967](https://arxiv.org/abs/2609.02967)

    提出FGLGuard框架，通过图联邦学习让各运营方在本地训练GNN风险检测器且仅共享模型更新，在保护私有数据的前提下实现对LLM多智能体系统的跨组织隐私保护安全防护。

    

    arXiv:2609.02967v1 公告类型：cross 摘要：针对基于LLM的多智能体系统（MAS）的拓扑引导安全防护方法，通过在智能体间通信图上训练图神经网络（GNN）来定位风险智能体并对拓扑结构进行干预——但这些方法假设单一运营方能够汇集所有带标签的交互轨迹数据。在跨组织的场景下，这一假设不再成立：交互片段包含私人提示词、工具输出和专有工作流，且没有任何单一数据孤岛能够观察到完整的攻击分布。我们将隐私保护的MAS安全防护问题构建为图联邦学习任务，并提出了FGLGuard：每个运营方在自己的由评判器标注的交互片段图上训练带边特征的图注意力检测器，仅共享模型更新。该方法结合了面向非IID客户端的近端局部目标函数、领域平衡聚合、过度拒绝约束下的阈值校准、经多方证实的上游评分以及针对被拦截答案的受保护改写机制。联邦化并非可有可无：现成的迁移学习方法在分布偏移下会失效（AUROC

    arXiv:2609.02967v1 Announce Type: cross  Abstract: Topology-guided safeguards for LLM-based multi-agent systems (MAS) train a GNN over the inter-agent communication graph to localize risky agents and intervene on the topology---but they assume one operator can pool all labeled traces. Across organizations that assumption breaks: episodes contain private prompts, tool outputs, and proprietary workflows, and no silo alone sees the full attack distribution. We cast privacy-preserving MAS safeguarding as graph federated learning and instantiate FGLGuard: each operator fits an edge-featured graph attention detector on its own judge-labeled episode graphs and shares only model updates. The method couples a proximal local objective for non-IID clients, domain-balanced aggregation, over-refusal-constrained threshold calibration, corroborated upstream scoring, and a guarded rewrite for blocked answers. Federation is not optional: off-the-shelf transfer collapses under distribution shift (AUROC 
    
[^154]: 当优化变成操纵：防御生成式搜索免受恶意生成式引擎优化的攻击

    When Optimization Becomes Manipulation: Defending Generative Search against Malicious Generative Engine Optimization

    [https://arxiv.org/abs/2609.02964](https://arxiv.org/abs/2609.02964)

    提出了GEO Defender——一个无需微调大语言模型、与攻击链对齐的两阶段防御框架（Shield Reranker与免训练护盾生成TFSG），用于防御那些事实一致且特征与高质量良性内容重合、令传统检测方法失效的恶意生成式引擎优化攻击。

    

    本文聚焦于防御生成式搜索引擎免受恶意生成式引擎优化（GEO）的攻击。恶意GEO通过改写网页文档以迎合引擎的引用偏好，从而操纵生成的答案。近期的GEO方法已从手工改写发展到自动化和智能体化优化，大幅提升了目标文档在生成答案中的可见度。然而，防御此类操纵面临两大挑战：攻击文档与其原文在事实上保持一致，导致事实核查和困惑度过滤方法失效；且它们所放大的特征同样也是高质量良性内容的典型特征。为解决这些局限，我们提出了GEO Defender，这是一种与攻击链相匹配的两阶段防御方法，无需对目标大语言模型进行微调。GEO Defender由护盾重排序器和无训练护盾生成组成。具体而言，……（原文在此处截断）

    arXiv:2609.02964v1 Announce Type: cross  Abstract: This paper focuses on defending generative search engines against malicious Generative Engine Optimization (GEO), which rewrites web documents to match engines' citation preferences and thereby manipulates generated answers. Recent GEO methods have advanced from hand-crafted rewriting to automated and agentic optimization, substantially increasing the visibility of target documents in generated answers. However, defending against such manipulation poses two major challenges: attack documents remain factually consistent with their originals, rendering fact verification and perplexity filtering ineffective, and the features they amplify equally characterize high-quality benign content. To address these limitations, we propose GEO Defender, a two-stage defense aligned with the attack chain that requires no fine-tuning of the target LLM. GEO Defender consists of Shield Reranker and Training-Free Shield Generation (TFSG). Specifically, Shie
    
[^155]: 无知的几何学：大语言模型知道何时调节贝叶斯先验

    The Geometry of Ignorance: LLMs Know When to Temper Bayesian Priors

    [https://arxiv.org/abs/2609.02959](https://arxiv.org/abs/2609.02959)

    研究发现大语言模型的反嵌入矩阵中存在一个编码训练语料词元分布的“无知方向”，模型通过逐词元调节该先验的强度，实现了随上下文信息增加而逐步减弱先验影响的温度调节贝叶斯更新。

    

    当语言模型缺乏线索时，它会预测什么？答案隐藏在其反嵌入矩阵的几何结构中：反嵌入矩阵的一个单一方向编码了训练语料库的词元分布，它充当了模型在不确定时回退依赖的贝叶斯先验。我们将这一结构称为“无知方向”，它出现在所有四个被检验的模型家族中（Llama、Qwen、Gemma 和 Pythia），参数规模从 0.4B 到 405B 不等。将最终预测状态投影到该方向上可得到逐词元的先验载荷因子 λ，经验表明该因子随着上下文信息量的增加而稳步下降。从形式上看，同样的投影将预测状态分解为两个正交向量，它们恰好对应于温度调节贝叶斯更新的两个因子：被提升到指数 λ 的词元先验以及由上下文驱动的似然。

    arXiv:2609.02959v1 Announce Type: cross  Abstract: What does a language model predict when it has few clues? The answer lurks in its unembedding geometry: a single direction of the unembedding matrix encodes the unigram distribution of the training corpus, which serves as the Bayesian prior the model falls back on when uncertain. This structure --- which we term the \emph{direction of ignorance} --- appears in all four model families examined (\texttt{Llama}, \texttt{Qwen}, \texttt{Gemma}, and \texttt{Pythia}), ranging from 0.4B to 405B parameters. Projecting the final prediction state onto this direction yields a per-token \emph{prior loading factor} $\lambda$, which, empirically, declines steadily as the context becomes more informative. Formally, the same projection decomposes the prediction state into two orthogonal vectors that correspond exactly to the two factors of a tempered Bayesian update: a unigram prior raised to the exponent $\lambda$ and a context-driven likelihood. This
    
[^156]: PrivateHub：用于私有传感器密集环境数据生成的对比扩散模型

    PrivateHub: Contrastive Diffusion Model for Private Sensor-Intensive Environment Data Generation

    [https://arxiv.org/abs/2609.02958](https://arxiv.org/abs/2609.02958)

    PrivateHub提出一种在扩散模型中结合对比学习的两阶段方法（应用条件预训练与应用感知微调），生成既能保持非私密应用可检测又能隐藏私密活动的合成多传感器数据流，从而解决跨传感器推断带来的隐私风险。

    

    传感器密集型环境通过从异构数据流中推断用户应用来支持众多智能服务。然而，并非所有应用都应该被暴露：用户希望某些活动保持私密。这就在为有用服务进行应用推断与防止非必要推断之间产生了矛盾。现有方法如差分隐私和基于规则的过滤只能保护单个数据流，但无法应对跨传感器推断带来的隐私风险。我们提出了PrivateHub，它在扩散模型中利用对比学习来生成合成的多传感器数据流，在保持非私密应用可被检测的同时隐藏私密应用。PrivateHub包含两个阶段：应用条件预训练（ACP），利用应用嵌入对多传感器数据进行条件化建模；以及应用感知微调（AAF），通过对比学习将私密数据与非私密数据分离。

    arXiv:2609.02958v1 Announce Type: cross  Abstract: Sensor-intensive environments enable many intelligent services by inferring user applications from heterogeneous data streams. However, not all applications should be exposed: users want some activities to stay private. This creates a tension between inferring applications for useful services and preventing unwanted inference. Existing approaches such as differential privacy and rule-based filtering protect individual streams but cannot address the privacy risk from cross-sensor inference.   We introduce Privatehub, which uses contrastive learning within a diffusion model to generate synthetic multi-sensor streams that keep non-private applications detectable while concealing private ones. Privatehub has two stages: App-Conditioned Pre-training (ACP), which conditions the model on multi-sensor data with application embeddings, and App-Aware Fine-tuning (AAF), which separates private from non-private data via contrastive learning. We al
    
[^157]: 面向认知诊断的隐私保护异构多LLM联邦推理

    Privacy-Preserving Heterogeneous Multi-LLM Federated Inference for Cognitive Diagnosis

    [https://arxiv.org/abs/2609.02947](https://arxiv.org/abs/2609.02947)

    该论文提出一种隐私保护的异构多LLM联邦推理框架，通过本地拉普拉斯噪声差分隐私和基于残差的聚合机制，使多个商用LLM API无需访问原始学生数据即可协作实现准确的认知诊断。

    

    AI驱动的教育系统在平衡隐私保护与准确认知诊断方面仍面临重大挑战。为克服这一问题，我们提出了一种联邦推理框架，使多个商用LLM API能够在无需访问原始学生数据或专有模型内部结构的前提下进行协作。该框架基于异构多LLM架构，利用多个联邦实体（如LLaMA-3.3-70B、GPT-4o-mini和Claude-3-Haiku）。这些实体生成的预测通过epsilon本地差分隐私进行融合，即在聚合之前对每个实体的预测输出本地添加拉普拉斯噪声，同时采用基于残差的聚合方式来缓解模型间的异质性。我们的方法建立在“诚实但好奇”的信任范式之上，即假设API提供者不会滥用所提交的查询，并且我们的差分隐私机制保护已发布的诊断结果免受外部……（原文摘要在此处截断）

    arXiv:2609.02947v1 Announce Type: cross  Abstract: Significant challenges remain in AI-driven educational systems in balancing privacy preservation with accurate cognitive diagnosis. To overcome this, we propose a federated inference framework in which several commercial LLM APIs collaborate without requiring access to raw student data or proprietary model internals. Using multiple federated entities, such as LLaMA-3.3-70B, GPT-4o-mini, and Claude-3-Haiku, our framework builds upon a heterogeneous multi-LLM architecture. The predictions generated by these entities are combined with epsilon-local differential privacy by adding Laplace noise locally to each entity's prediction output before aggregation, while residual-based aggregation mitigates model heterogeneity. Our approach is predicated on an honest-but-curious trust paradigm in which API providers are presumed not to abuse submitted queries, and our differential privacy mechanism shields the published diagnostic results from exter
    
[^158]: Reflect-SQL：一种基于自我反思的Text-to-SQL框架

    Reflect-SQL: A Self-Reflection Based Framework for Text-to-SQL

    [https://arxiv.org/abs/2609.02944](https://arxiv.org/abs/2609.02944)

    Reflect-SQL是一个基于多阶段自我反思的Text-to-SQL新框架，通过知识库理解晦涩的数据库模式，并利用LLM-as-a-judge驱动的评分机制在反馈循环中迭代优化每个阶段的SQL生成结果。

    

    通过自然语言实现数据访问的民主化是现代企业的重要目标，但Text-to-SQL的实际应用受到现实世界复杂性的严重阻碍：1. 晦涩且庞大的数据库模式；2. 由于模式的固定结构设置和用户查询的模糊性，导致无法有效检索相关的表和列；3. 由于缺乏健壮的验证和纠错机制，生成了语法或逻辑上有缺陷的SQL。为了解决这些系统性挑战，我们提出了Reflect-SQL，这是一个新颖的Text-to-SQL框架，其基于多阶段自我反思方法，利用知识库来理解晦涩的数据库模式，建立有效的检索流程以及生成语法/语义正确的SQL的系统。我们的系统并非采用单次尝试的方式，而是在相互关联的反馈循环中采用LLM-as-a-judge驱动的评分机制，在每个阶段迭代地优化结果。

    arXiv:2609.02944v1 Announce Type: cross  Abstract: Democratizing data access through natural language is a crucial goal for modern enterprises, but the practical adoption of Text-to-SQL is critically hindered by real-world complexities: 1. Obscure and large database schemas, 2. Ineffective retrieval of relevant tables and columns due to structured setting of schemas and vague user query, 3. Generation of syntactically or logically flawed SQL due to a lack of robust validation and correction mechanism. To address these systemic challenges, we introduce Reflect-SQL, a novel framework for Text to SQL, grounded in multi-stage self-reflection approach to develop understanding of obscure schema using a knowledge base, setup a process for effective retrieval and system to generate syntactically/semantically SQL. Instead of a single-pass attempt, our system employs an LLM-as-a-judge driven scoring mechanism within interconnected feedback loops to iteratively refine the results at every stage. 
    
[^159]: 评判LLM作为评判者：基于LLM的自动文本生成评估中令人担忧的评估准则伪影问题

    Judging LLM-as-a-Judge: Concerning Rubric Artifacts in LLM-based Automated Text Generation Evaluation

    [https://arxiv.org/abs/2609.02942](https://arxiv.org/abs/2609.02942)

    研究发现LLM评估中的评估准则文本本身编码了可预测的评估信号，且评判者在候选回答或准则被反转时往往无法可靠更新判断，这引发了对基于准则的LLM自动评估可靠性的严重质疑。

    

    基于大语言模型（LLM）作为评判者的评估流程被越来越多地用于评估AI生成的文本，其前提假设是评判结果源于对候选回答依据评估准则的推理。我们证明这一假设需要进一步审视。仅在评估准则文本上训练的分类器，在完全无法接触任何被评估回答的情况下，就能对评判结果取得可观的预测性能。这表明评估准则的表述中编码了可提取的评估信号，使得评分可以在不依赖模型输出的情况下被部分预测。最后，反事实扰动实验揭示，当候选回答或评估准则被反转时，评判者往往无法可靠地更新其判断。我们的发现引发了对基于评估准则的LLM评估可靠性的担忧，并强调需要对通过LLM进行自动化评估的方法论展开进一步研究。

    arXiv:2609.02942v1 Announce Type: cross  Abstract: LLM-as-a-Judge pipelines are increasingly used to evaluate AI-generated text, based on the assumption that judgments arise from reasoning over candidate responses with respect to a rubric. We show that this assumption warrants further scrutiny. Classifiers trained only on rubric text, without access to any evaluated response, achieve nontrivial predictive performance on judge outputs. This suggests that rubric formulations encode recoverable evaluative signals, allowing scores to be partially anticipated independently of model outputs. Finally, counterfactual perturbations reveal that judges often fail to reliably update their decisions when either the candidate response or the rubric criterion is reversed. Our findings raise concerns about the reliability of rubric-based LLM evaluation and highlight the need for further methodological study of automated evaluation via LLMs.
    
[^160]: 倾听潜在表示：通过隐藏状态交互实现大型音频语言模型中的自纠正语音识别

    Listen to the Latents: Self-Correcting Speech Recognition in Large Audio Language Models Through Hidden-State Interactions

    [https://arxiv.org/abs/2609.02940](https://arxiv.org/abs/2609.02940)

    该论文提出Hybrid Search纠错策略，利用基于LLM的ASR隐藏状态与基础LLM隐藏状态之间的交互特征来识别语义依赖程度高的词元并进行选择性精炼，从而显著提升LoRA适配的热启动初始化语音识别模型性能，超越重打分等全局纠错方法。

    

    近期的自动语音识别（ASR）系统越来越多地集成大型语言模型（LLM）以利用其语义知识，方式包括通过logit融合进行外部集成，或通过热启动初始化进行内部集成。然而，如何有效结合这两种策略仍未得到充分探索。在本工作中，我们通过利用模型自身预适配阶段的基础LLM来改进基于热启动初始化LLM的ASR模型，重点关注保留基础LLM的LoRA适配设置。为实现这一目标，我们提出了Hybrid Search，一种由两点观察启发的针对性纠错策略。第一，刻画基于LLM的ASR隐藏状态与基础LLM隐藏状态之间关系的交互特征，能够为词元的语义依赖程度提供有信息量的信号。第二，选择性地对语义依赖程度高的目标词元进行精炼，其ASR性能提升远超朴素的全局LLM纠错方法，包括重打分（rescoring）等方法。

    arXiv:2609.02940v1 Announce Type: cross  Abstract: Recent automatic speech recognition (ASR) systems increasingly integrate large language models (LLMs) to leverage their semantic knowledge, either externally through logit fusion or internally through warm initialization. However, how to effectively combine these two strategies remains underexplored. In this work, we refine warm-initialized LLM-based ASR models by leveraging their own pre-adaptation base LLMs, focusing on LoRA-adapted settings where the base LLM is preserved. To achieve this, we propose Hybrid Search, a targeted correction strategy motivated by two observations. First, interaction features that characterize the relationship between LLM-based ASR hidden states and base-LLM hidden states provide informative signals about a token's degree of semantic dependence. Second, selectively refining targeted tokens with high semantic dependence improves ASR performance far beyond naive global LLM-correction methods including resco
    
[^161]: 反例作为智能体自我纠正的反馈

    Counterexamples as Feedback for Agent Self-Correction

    [https://arxiv.org/abs/2609.02892](https://arxiv.org/abs/2609.02892)

    本文提出 A-CEGIS 轻量级框架，将反例作为反馈来评估智能体在自然语言到正则表达式合成中的多轮自我纠正能力，在四轮预算内解决了 90% 的任务，显著优于零样本生成和通用自我纠正方法。

    

    单轮代码生成指标低估了已部署智能体的一个核心特性：它们在收到具体反馈后能否修复错误的产物。本文提出了 A-CEGIS，一个轻量级框架，它使用反例作为反馈来评估从自然语言到正则表达式合成任务中的多轮精炼能力。智能体提出一个正则表达式，确定性预言机在全匹配语义下对其进行检查，紧凑的假阳性或假阴性反例样本则引导下一轮迭代。在 30 个 NL-RX-Turk 任务上，诊断性反例反馈在四轮消融预算内解决了 90% 的任务，相比之下，零样本生成仅为 17%，通用自我纠正为 27%，仅错误反馈为 23%。在带有强化的完整诊断运行中，所有任务在最终轮次时都在隐藏测试集上得到解决，平均成功耗时为 2.7 轮，经过针对性探测后的稳健成功率为 77%。这些结果表明，A-CEGIS 衡量的是智能体在实际部署场景中利用具体反馈进行自我修复的能力。

    arXiv:2609.02892v1 Announce Type: cross  Abstract: Single-turn code-generation metrics understate a central property of deployed agents: whether they can repair a wrong artifact after receiving concrete feedback. This paper presents A-CEGIS, a lightweight framework that uses counterexamples as feedback for evaluating multi-turn refinement in natural-language-to-regex synthesis. An agent proposes a regex, a deterministic oracle checks it under full-match semantics, and compact false-positive or false-negative witnesses guide the next turn. On 30 NL-RX-Turk tasks, diagnostic counterexample feedback solves 90\% of tasks within a four-turn ablation budget, compared with 17% for zero-shot generation, 27% for generic self-correction, and 23% for error-only feedback. In a full diagnostic run with hardening, all tasks are solved on the hidden set by the final turn, with mean time-to-success of 2.7 turns and robust success of 77% after targeted probing. These results show that A-CEGIS measures 
    
[^162]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^163]: 迈向用于识别和解决基于对话的人机交互中矛盾的基础本体

    Towards a Foundational Ontology for Identifying and Resolving Contradictions in Dialogue-based Human-Robot Interactions

    [https://arxiv.org/abs/2609.02364](https://arxiv.org/abs/2609.02364)

    该研究基于活动理论概念并采用METHONTOLOGY方法构建了一个基础本体，用于形式化表示和定义人机交互中基于对话的协作交互及其相关矛盾，填补了跨HRI与HAI领域的形式化计算框架的空白。

    

    现有的人机交互（HRI）文献主要聚焦于在特定领域的基于对话的交互中识别和构建错误、故障、冲突和知识问题（本工作将其统称为“矛盾”）。然而，目前仍然缺乏一个形式化的计算框架来表示和定义这些矛盾，且该框架需能在HRI与人-智能体交互（HAI）领域之间实现互操作和使用。因此，本研究项目旨在捕捉、表示和评估（1）基于对话的协作交互以及（2）相关矛盾的概念，并将其构建为一个基础本体。本研究采用了METHONTOLOGY——一种构建领域无关本体的系统化方法。在所提出本体的概念化阶段，使用了源自活动理论的概念和模型。本短文提出的初步成果包括：（i）HRI中对话及相关矛盾的自然语言定义，（ii）……（摘要至此不完整）

    arXiv:2609.02364v1 Announce Type: cross  Abstract: Existing Human-Robot Interaction (HRI) literature has focused on identifying and structuring errors, failures, conflicts, and knowledge issues (called in this work as contradictions) in domain-specific dialogue-based interactions. However, there is still lack of a formal computational framework to represent and define these contradictions, interoperable and usable across HRI and human-agent interaction (HAI) domains. Thus, this research project aims to capture, represent, and evaluate the notion of (1) dialogue-based collaborative interaction and (2) related contradictions in a foundational ontology. METHONTOLOGY, a systematic approach to build domain-independent ontologies was applied. In the conceptualisation stage of the presented ontology, concepts and models from Activity Theory were used. Preliminary results presented in this short article are: (i) Natural language definitions of dialogues and related contradictions in HRI, (ii) 
    
[^164]: VoRTeC：驯服基础流模型实现一步式实时视频压缩

    VoRTeC: Taming Foundation Flow for One-step Real time Video Compression

    [https://arxiv.org/abs/2609.02291](https://arxiv.org/abs/2609.02291)

    提出了基于基础流模型 Wan2.1 的视频压缩框架 VoRTeC，无需访问流匹配网络的参数或梯度即可实现一步式解码、高感知保真度和跨帧组时序一致性的超低码率实时视频压缩。

    

    超低码率视频压缩仍然面临关键挑战：传统神经视频压缩不可避免地引入模糊伪影，而基于扩散的生成式视频压缩则存在解码延迟过高和时序一致性差的问题。为了解决这些问题，我们提出了 VoRTeC，一个建立在基础流模型之上的视频压缩框架。通过紧凑地编码潜在视频表示、预测压缩后表示沿流轨迹的位置，并集成多尺度先验，VoRTeC 使压缩器能够有效利用生成式视频流先验。在无需访问流匹配网络的参数或梯度的情况下，我们的框架实现了一步式解码和具有高感知保真度的重建。同时，我们通过尾帧复用和先验缓存来保持帧组之间的时序一致性。大量实验证明……

    arXiv:2609.02291v1 Announce Type: cross  Abstract: Ultra-low bitrate video compression still faces critical challenges: traditional neural video compression inevitably introduces blurring artifacts, while diffusion-based generative video compression suffers from excessive decoding latency and poor temporal consistency. To address these issues, we propose $\mathtt{VoRTeC}$, a Video Compression framework built upon a foundational flow model (Wan2.1). By compactly encoding latent video representations, predicting the positions of compressed representations along flow trajectories, and integrating multi-scale priors, $\mathtt{VoRTeC}$ enables the compressor to harness generative video flow priors effectively. Without accessing the parameters or gradients of flow matching networks, our framework achieves one-step decoding and reconstructions with high perceptual fidelity. Meanwhile, we maintain consistency across frame groups via tail-frame reuse and prior caching. Extensive experiments dem
    
[^165]: OmegaUse-SOP：从人类演示中实现专业计算机使用的SOP工程

    OmegaUse-SOP: SOP Engineering for Professional Computer Use from Human Demonstrations

    [https://arxiv.org/abs/2609.02149](https://arxiv.org/abs/2609.02149)

    提出了OmegaUse-SOP系统，通过人在回路的SOP工程方法，将专业计算机操作的人类演示迭代式地转化为GUI智能体可复用的SOP技能，从而解决了智能体执行特定领域专业标准操作程序的难题。

    

    大型语言模型（LLMs）正日益从对话式助手演变为能够操作外部数字环境的智能体。图形用户界面（GUI）智能体在这一转变中发挥着重要作用，因为许多现实世界的工作流程仍然只能通过面向用户的软件界面来访问。然而，尽管近期在通用计算机使用基准测试上取得了进展，但特定领域的专业标准操作程序（SOP）对GUI智能体而言仍然充满挑战，因为它们通常涉及隐性的领域知识、软件特定的操作惯例以及任务级别的验证要求。我们提出了OmegaUse-SOP，这是一个人在回路（human-in-the-loop）的SOP工程系统，用于将专业计算机使用的人类演示转化为GUI智能体可复用的SOP技能。类似于提示工程（prompt engineering），SOP工程通过迭代式地精炼演示内容、执行规则和领域知识，将专业SOP

    arXiv:2609.02149v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly evolving from conversational assistants into agents capable of operating external digital environments. Graphical user interface (GUI) agents play an important role in this transition, as many real-world workflows remain accessible only through user-facing software interfaces. However, despite recent progress on general computer-use benchmarks, domain-specific professional standard operating procedures (SOPs) remain challenging for GUI agents because they often involve implicit domain knowledge, software-specific conventions, and task-level verification requirements. We introduce OmegaUse-SOP, a human-in-the-loop SOP Engineering system for transforming human demonstrations of professional computer use into reusable SOP skills for GUI agents. Analogous to prompt engineering, SOP Engineering iteratively refines demonstrations, execution rules, and domain knowledge to convert professional SOPs
    
[^166]: 多模态大语言模型中跨模态安全漂移的安全意识迁移

    Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.02082](https://arxiv.org/abs/2609.02082)

    针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。

    

    视觉模态增强了多模态大语言模型（MLLMs）的能力，但也引入了安全隐患：一个本身无害的文本查询在与视觉图像结合时可能传达有害意图。我们将这种现象称为“跨模态安全漂移”，我们的初步研究表明，此类请求的安全响应率显著低于包含明确不安全文本的请求。本文旨在系统研究这一问题。首先，我们进行了实证分析，识别出代表性的不安全响应模式。在此基础上，我们对模型表示和注意力机制进行了解释分析，揭示出视觉风险线索受到的关注有限，难以有效触发拒绝响应。受不安全文本处理中的安全信号可以迁移这一观察的启发，我们提出了安全意识表示迁移，这是一种轻量级的方向细化方法，能够缓解跨模态安全漂移并显著提升……

    arXiv:2609.02082v1 Announce Type: cross  Abstract: Visual modality enhances the capabilities of multimodal large language models (MLLMs) but also introduces a safety concern: a benign textual query may convey harmful intent when grounded in a visual image. We term this cross-modal safety drift and our pilot studies show that the safety response rate for such requests is substantially lower than that for requests containing explicitly unsafe text. This paper aims to systematically study this issue. First, we conduct an empirical analysis to identify representative unsafe response patterns. Building on these, we interpret model representations and attentions, revealing that visually risky cues receive limited attention and weakly trigger refusal. Motivated by the observation that safety signals from unsafe text processing can be transferred, we propose safety-awareness representation transfer (SRT), a lightweight direction-refinement method that mitigates cross-modal safety drift with a 
    
[^167]: ExecRetrieval：衡量代码嵌入检索中的功能正确性差距

    ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval

    [https://arxiv.org/abs/2609.01865](https://arxiv.org/abs/2609.01865)

    提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。

    

    基于嵌入的代码检索是编码智能体和检索增强代码生成的核心组件，在这些场景中，检索到功能正确的代码比检索到词汇上相似的代码更为重要。现有的代码检索基准并未在搜索池中植入受控的、经执行验证的、针对每个查询规范实现的单次编辑变体，因此“嵌入模型能否在检索场景中从功能上区分正确代码与近似克隆但不正确的代码”这一问题仍未得到解答。解决这一问题需要一个搜索池本身就包含相关反事实样本的基准——即与每个规范实现几乎完全相同、且经过执行验证的有缺陷变体——从而可以直接检验检索器的排序结果是否具备功能区分能力，而不仅仅是主题或身份上的重合。我们提出了 ExecRetrieval，包含 939 个 Python 任务，每个任务都配有一个经执行验证的规范实现，以及最多四个经执行验证的……

    arXiv:2609.01865v1 Announce Type: cross  Abstract: Embedding-based code retrieval is a core component of coding agents and retrieval-augmented code generation, where retrieving correct code matters more than retrieving lexically similar code. Existing code-retrieval benchmarks do not plant controlled, execution-verified single-edit variants of each query's canonical implementation in the search pool, leaving the question of whether embeddings can functionally discriminate correct from near-clone-but-incorrect code unanswered in a retrieval setting. Resolving this requires a benchmark whose search pool itself contains the relevant counterfactuals -- execution-verified buggy variants near-identical to each canonical -- so that a retriever's rank ordering can be directly tested for functional discrimination rather than topical or identity overlap. We introduce ExecRetrieval, 939 Python tasks each paired with one execution-verified canonical implementation and up to four execution-verified
    
[^168]: 用于网络压缩的可复用神经基底的数学理论

    A Mathematical Theory of Reusable Neural Bases for Network Compression

    [https://arxiv.org/abs/2609.01550](https://arxiv.org/abs/2609.01550)

    该论文提出线性可复用神经基底架构（LRNBA），通过将网络块表示为共享神经基底的线性组合，在保持稳定训练的同时大幅压缩参数并降低内存成本，使模型在相同参数预算下能够构建更宽更深的网络。

    

    随着大型AI模型在各类应用中日益普及，内存成本已成为训练和推理中的关键瓶颈。为缓解这一问题，我们提出了线性可复用神经基底架构（LRNBA），这是一种旨在提高参数效率并降低内存成本的新型框架。受循环神经网络（RNN）设计的启发，我们方法的核心思想是将每个网络块表示为共享神经基底集合的线性组合，从而在保持稳定训练的同时实现高度的网络压缩率。所提出的架构允许在相同的参数预算下构建显著更宽和更深的网络。大量实验表明，我们的模型与经典架构相比实现了相当甚至更快的收敛速度和更低的损失，同时保持了稳定的训练动态。

    arXiv:2609.01550v1 Announce Type: cross  Abstract: As large AI models become increasingly prevalent across a wide range of applications, memory cost has become a critical bottleneck in both training and inference. To mitigate this issue, we introduce the Linear Reusable Neural Bases Architecture (LRNBA), a novel framework aimed at improving parameter efficiency and reducing memory cost. Inspired by recurrent neural network (RNN) designs, the core idea of our approach is to represent each network block as a linear combination of a shared set of neural bases, thereby enjoying highly network compression rate while maintaining stable training. The proposed architecture allows for the construction of significantly wider and deeper networks under the same parameter budget. Extensive experiments demonstrate that our model achieves comparable or even faster convergence and lower loss than classical architectures, while maintaining stable training dynamics.
    
[^169]: LatentPress：超越文本与视觉的上下文压缩

    LatentPress: Context Compression Beyond Text and Vision

    [https://arxiv.org/abs/2609.01507](https://arxiv.org/abs/2609.01507)

    LatentPress提出将对话历史和长文档压缩为连续记忆token这一第三种表示形式，让冻结的语言模型通过输入嵌入接口直接读取，仅训练约占解码器0.1%参数的适配器即可实现4-16倍压缩，且性能超过文本摘要和基于OCR的压缩方法。

    

    压缩后的上下文通常以人类可读的文本形式承载，或以必须被解码的渲染图像形式承载，即使其消费者是语言模型也是如此。我们提出了LatentPress，它将对话历史和长文档写入第三种表示形式：连续的记忆token（memory tokens），冻结的解码器通过其输入嵌入接口直接读取这些token，在推理时无需进行文本重建。一个与阅读器匹配的小型写入器可实现4至16倍的压缩，同时只需训练一个适配器（参数量为420万至2620万，约占解码器的0.1%）。在LongMemEval基准上，LatentPress在7.70倍压缩下达到0.504的准确率，超过未压缩证据的0.490，并显著优于文本摘要（0.184）和基于OCR的压缩（0.426至0.312）。在LongBench-QA上，域内写入器在4至8倍压缩下匹配或超过原始上下文阅读的性能，而16倍压缩则落后于原始上下文。写入每段对话仅需43毫秒，大约快一个数量级。

    arXiv:2609.01507v1 Announce Type: cross  Abstract: Compressed context is usually carried as human-readable text or as rendered images that must be decoded, even when its consumer is a language model. We introduce LatentPress, which writes conversational histories and long documents into a third representation: continuous memory tokens that a frozen decoder reads directly through its input-embedding interface, with no text reconstruction at inference. A small reader-matched writer compresses $4$-$16\times$ while training only an adapter (4.2M-26.2M parameters, $\sim\!0.1\%$ of the decoder). On LongMemEval, LatentPress reaches $0.504$ accuracy at $7.70\times$ compression versus $0.490$ for uncompressed evidence, outperforming text summaries (0.184) and OCR-based compression (0.426 to 0.312). On LongBench-QA, in-domain writers match or exceed raw-context reading at $4$-$8\times$ compression, while $16\times$ trails raw. Writing takes 43ms per conversation, roughly an order of magnitude fa
    
[^170]: 通过幂律熵搜索高效估计最优超参数缩放定律

    Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search

    [https://arxiv.org/abs/2609.01431](https://arxiv.org/abs/2609.01431)

    本文提出幂律熵搜索（PLES），一种基于多保真度贝叶斯优化的计算成本感知采集函数，通过自适应选择能最大程度降低缩放定律估计整体不确定性的实验配置（而非优化单一目标函数），高效估计大语言模型最优超参数随规模变化的缩放定律，从而大幅节省计算资源。

    

    最优超参数缩放定律描述了用于大语言模型（LLM）训练的最佳超参数如何随模型和数据规模变化，使从业者无需昂贵的大规模调优即可预测生产规模下的最优配置。然而，传统上估计这些缩放定律需要对数千次训练运行进行穷举网格搜索，消耗巨大的计算资源。我们提出了幂律熵搜索（Power-Law Entropy Search, PLES），这是一种建立在多保真度贝叶斯优化之上的计算成本感知采集函数，能够通过自适应实验高效估计最优超参数缩放定律。PLES的一个关键创新在于，它搜索的是能够降低缩放定律估计整体不确定性的候选配置，而不是优化单一目标函数。在每次迭代中，PLES选择能够最大程度降低缩放定律估计不确定性的候选配置。

    arXiv:2609.01431v1 Announce Type: cross  Abstract: Optimal hyperparameter scaling laws describe how the best hyperparameters for large language model (LLM) training change with model and data scale, enabling practitioners to predict optimal configurations at production scales without expensive large-scale tuning. However, estimating these scaling laws conventionally requires exhaustive grid searches over thousands of training runs, consuming enormous computational resources. We introduce Power-Law Entropy Search (PLES), a computational cost-aware acquisition function built on multi-fidelity Bayesian optimization that efficiently estimates optimal hyperparameter scaling laws through adaptive experimentation. A key innovation in PLES is that it searches for candidates that reduce the overall uncertainty of a scaling law estimate, instead of optimizing a single objective function. At each iteration, PLES selects the candidate configuration that maximally reduces the uncertainty of the sca
    
[^171]: 科学智能体技能：面向科研智能体的程序性知识库

    Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents

    [https://arxiv.org/abs/2609.00065](https://arxiv.org/abs/2609.00065)

    该论文提出了一个名为“科学智能体技能”的开放库，收录了基因组学、化学信息学等16个科研实践领域共163项程序性知识，使语言模型智能体能够遵循领域规范做出站得住脚的科学分析，而非仅仅返回能运行的代码。

    

    被要求分析实验的语言模型智能体通常只会返回一段能运行的代码，但该分析是否站得住脚则是另一回事。一个站得住脚的分析取决于程序性选择：该领域接受哪种统计检验方法、哪个标识符命名空间是权威的、以及结果必须附带哪些注意事项。我们提出了“科学智能体技能”，这是一个开放的知识库，包含16个实践领域的163项此类程序，涵盖基因组学、化学信息学、医学影像、研究设计和科学传播等。每项技能都是一个目录，围绕一个版本化、人类可读的指令文件构建。智能体仅在任务需要时才加载该文件；目录中通常还包含参考资料和可运行的脚本。我们未报告任务级评估结果和宿主选择率。该库采用开放许可证，可在 https://github.com/K-Dense-AI/scientific-agent-skills 获取。

    arXiv:2609.00065v1 Announce Type: cross  Abstract: A language-model agent asked to analyse an experiment will usually return working code. Whether the analysis is defensible is a different question. A defensible analysis depends on procedural choices: which test the field accepts, which identifier namespace is authoritative, and which caveats must accompany a result. We present Scientific Agent Skills, an open library of 163 such procedures in 16 areas of practice, including genomics, cheminformatics, medical imaging, study design and scientific communication. Each skill is a directory built around a versioned, human-readable instruction file. An agent loads the file only when a task calls for it; the directory often also contains reference material and runnable scripts. We report no task-level evaluation and no host selection rate. Openly licensed and available at https://github.com/K-Dense-AI/scientific-agent-skills.
    
[^172]: 从数据分析到肿瘤委员会：一种证据关联的多智能体肿瘤学特征提取工作流

    From Analytics to Tumor Boards: An Evidence-Linked Multi-Agent Workflow for Oncology Feature Extraction

    [https://arxiv.org/abs/2608.28974](https://arxiv.org/abs/2608.28974)

    本文提出并评估了Nimblemind多智能体系统（nMAS），这是一个可配置的证据关联多智能体工作流，能够从碎片化的肿瘤学文档中提取涵盖328个临床医生定义属性的结构化信息，从而大幅降低人工癌症登记的抽象负担。

    

    临床上相关的肿瘤学信息分散在异构的纵向文档中，造成了巨大的信息抽象负担，并需要在标本、肿瘤、生物标志物和时间点之间进行准确归因，而人工癌症登记抽象每个病例可能需要27.2分钟，这凸显了在将文档转换为结构化数据的同时保留临床背景的可扩展方法的必要性。我们评估了Nimblemind多智能体系统（nMAS），这是一个可配置的肿瘤学信息提取工作流，能够从碎片化的肿瘤学文档中提取临床相关的结构化字段。该提取任务使用由临床医生主导的328个属性的模式，涵盖报告元数据、诊断、分期以及癌症类型特定信息。nMAS将临床医生定义的字段规范与模型执行分离，并结合了复杂度感知提取、报告级整合和来源归因机制。

    arXiv:2608.28974v1 Announce Type: new  Abstract: Clinically relevant oncology information is distributed across heterogeneous, longitudinal documentation, creating substantial abstraction burden and requiring accurate attribution across specimens, tumors, biomarkers, and time points, while manual cancer-registry abstraction can require 27.2 minutes per case, highlighting the need for scalable methods that preserve clinical context while converting documentation into structured data. We evaluate the Nimblemind Multi-Agent System (nMAS), a configurable oncology information-extraction workflow which extracts clinically relevant structured fields from fragmented oncology documentation. The extraction task uses a clinician-informed schema of 328 attributes spanning report metadata, diagnosis, staging, and cancer-type-specific information. nMAS separates clinician-defined field specifications from model execution and combines complexity-aware extraction, report-level consolidation, and sourc
    
[^173]: PAWBench：我们在概率对齐世界建模方面走了多远？

    PAWBench: How Far Are We from Probabilistically Aligned World Modeling?

    [https://arxiv.org/abs/2608.27345](https://arxiv.org/abs/2608.27345)

    本文提出了PAWBench基准和PAWEval协议，首次将概率对齐作为世界模型的核心评估标准，以衡量视频生成器在重复生成时能否恢复正确的行为分布。

    

    摘要：arXiv:2608.27345v1 公告类型：交叉 摘要：最近的视频生成模型越来越多地被构建为世界模型。许多物理过程可以以不止一种有效方式展开。因此，世界模型不仅应再现合理的轨迹，还应再现相同初始观察和动作下可能行为的分布。我们称这种分布级要求为概率对齐。然而，现有评估主要检查单个视频的合理性，并不测试重复生成是否恢复正确的分布。这引出一个核心问题：当前视频生成器距离概率对齐的世界建模还有多远？为回答此问题，我们将概率对齐形式化为世界模型的分布性标准，并引入PAWBench，一个用于评估视频生成器作为世界动态随机采样器的基准。我们进一步提出PAWEval，一个结果级协议，将重复视频滚动转换为...

    arXiv:2608.27345v1 Announce Type: cross  Abstract: Recent video generation models are increasingly framed as world models. Many physical processes can unfold in more than one valid way. Therefore, a world model should reproduce not only a plausible trajectory, but also the distribution of possible behaviors under the same initial observation and action. We call this distribution-level requirement probabilistic alignment. However, existing evaluations largely assess individual-video plausibility and do not test whether repeated generations recover the correct distribution. This raises a central question: how far are current video generators from probabilistically aligned world modeling? To answer it, we formalize probabilistic alignment as a distributional criterion for world models and introduce PAWBench, a benchmark for evaluating video generators as stochastic samplers of world dynamics. We further introduce PAWEval, an outcome-level protocol that converts repeated video rollouts int
    
[^174]: 安全性无法组合：自主LLM代理的非衰减循环状态

    Safety Does Not Compose: Non-Decaying Loop State for Autonomous LLM Agents

    [https://arxiv.org/abs/2608.27141](https://arxiv.org/abs/2608.27141)

    本文证明，针对跨多次迭代分散证据的攻击，仅基于单轨迹的LLM代理安全监控器无法有效检测（真阳性率等于假阳性率），而保留跨迭代非衰减状态的监控器才能完美分离，揭示了安全性组合的根本失败。

    

    arXiv:2608.27141v1 公告类型：交叉 摘要：大型语言模型代理越来越多地被部署为自主循环系统。从一个人类目标开始，这种系统反复发现工作、规划、执行工具调用、验证结果，并在多次无人值守的迭代中持久化状态。然而，广泛使用的代理安全措施是针对单一轨迹定义的，当下一轨迹开始时，其安全状态会被重新初始化。我们表明，这是一个组合失败而非实现细节问题。我们的核心结果是一个分离性：针对证据分散在多次迭代中的攻击，每个轨迹范围监控器的真阳性率等于其假阳性率，无论其表达力如何，因为它所需的证据永远不会出现在它看到的窗口内，而保留跨迭代状态的监控器则能完美区分两者。我们进一步表明，携带几何衰减风险评分的明显修复方法无法解决该问题，而需要非衰减的循环状态来实现组合安全性。

    arXiv:2608.27141v1 Announce Type: cross  Abstract: Large language model agents are increasingly deployed as autonomous loops. Starting from one human goal, such a system repeatedly discovers work, plans, executes tool calls, verifies outcomes and persists state across many unattended iterations. The agent safeguards in wide use, however, are defined over a single trajectory, and their safety state is re-initialized when the next trajectory begins. We show that this is a failure of composition rather than an implementation detail. Our central result is a separation: against an attack whose evidence is fragmented across several iterations, every trajectory-scoped monitor has a true-positive rate equal to its false-positive rate, however expressive it is, because the evidence it would need never appears in the window it sees, whereas a monitor retaining cross-iteration state separates the two perfectly. We further show that the obvious repair of carrying a geometrically decaying risk scor
    
[^175]: 拒绝几何反映拒绝训练：多样的拒绝前缀能提升稳定秩并削弱拒绝向量消融攻击

    Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks

    [https://arxiv.org/abs/2608.25390](https://arxiv.org/abs/2608.25390)

    本文发现拒绝训练中的首词损失塑造了拒绝方向和子空间，且重复的拒绝前缀导致拒绝几何脆弱，但多样前缀能提升稳定性并削弱消融攻击。

    

    arXiv:2608.25390v1 公告类型：新 摘要：拒绝训练通过训练模型拒绝不安全查询来保护AI模型免受越狱攻击，降低滥用风险。近期研究发现，对齐语言模型中的拒绝行为可由单一激活方向或跨有害提示共享的低维拒绝子空间介导：消融这些方向会抑制拒绝，同时基本保留其他模型能力。然而，为何安全关键特征在广泛模型中涌现并集中为低维结构仍不清楚。在对OLMo-2-0425-1B-Instruct的案例研究中，我们发现拒绝几何反映拒绝训练：由拒绝完成首词损失引起的激活更新解释了产生的拒绝方向和拒绝子空间。我们通过拒绝数据集中的训练动态研究拒绝方向，并揭示其脆弱性与重复的拒绝开头相关，这反过来又影响模型对多样拒绝前缀的稳定性。

    arXiv:2608.25390v1 Announce Type: new  Abstract: Refusal training protects AI models from jailbreaks by training models to decline unsafe queries, reducing the risk of misuse. Recent work finds that refusal behavior in aligned language models can be mediated by a single activation direction or a low-dimensional refusal subspace shared across harmful prompts: ablating those directions suppresses refusals while largely preserves other model capabilities. Yet it remains unclear why safety-critical features in a wide range of models emerge and concentrated, low-dimensional structure. In a case study of OLMo-2-0425-1B-Instruct we find that the refusal geometry reflects refusal training: activation updates resulting from refusal-completion first-token losses explain the resulting refusal direction and refusal subspace. We study refusal directions through the training dynamics across refusal datasets and reveal that their brittleness is associated with repetitive refusal starts, which in turn
    
[^176]: VideoHarness-RSI：针对冻结视觉语言模型的长视频理解递归式自改进框架

    VideoHarness-RSI: Recursive Harness Self-Improvement for Long-Video Understanding with Frozen Vision-Language Models

    [https://arxiv.org/abs/2608.24302](https://arxiv.org/abs/2608.24302)

    本文提出VideoHarness-RSI，通过递归搜索和自改进可执行的上下文构建程序，在不修改冻结视觉语言模型的情况下，显著提升长视频理解性能。

    

    长视频理解在很大程度上取决于如何从更长的视频中构建有限的模型上下文。现有方法通过压缩、检索、记忆和代理证据获取来改进这一过程，但这些机制通常作为手动设计的推理系统的一部分引入，或与其他组件一起优化。这使得难以隔离一个更简单的问题：仅改进可执行的上下文构建程序能带来多大收益？我们通过VIDEOHARNESS-RSI来研究这个问题，这是一个围绕冻结视觉语言模型（VLM）递归搜索可执行上下文构造器的受控基线。一个外层循环提议器使用先前的程序、评估结果和执行轨迹来生成候选框架，这些框架被端到端执行和评估，成功的变体被保留以进行进一步搜索。这使得长视频理解成为一个持续优化的过程。

    arXiv:2608.24302v1 Announce Type: new  Abstract: Long-video understanding depends critically on how a limited model context is constructed from a much longer video. Existing approaches improve this process through compression, retrieval, memory, and agentic evidence acquisition, but these mechanisms are typically introduced as part of a manually designed inference system or optimized together with other components. This makes it difficult to isolate a simpler question: how much can be gained by improving the executable context-construction program alone? We study this question through VIDEOHARNESS-RSI, a controlled baseline for recursively searching executable context constructors around a frozen vision-language model (VLM). An outer-loop proposer uses prior programs, evaluation outcomes, and execution traces to generate candidate harnesses, which are executed and evaluated end to end before successful variants are retained for further search. This makes long-video understanding a cont
    
[^177]: AI代理将人类挤出决策圈

    AI Agents Push Humans Out of the Loop

    [https://arxiv.org/abs/2608.23642](https://arxiv.org/abs/2608.23642)

    本文指出当前AI代理发展方式削弱而非支持人类监督能力，强调应将人类监督需求置于与AI能力同等优先级，以保障有效人类参与。

    

    arXiv:2608.23642v1 公告类型：新 摘要：随着AI代理被赋予越来越多的自主权，它们带来了显著风险。一个常见的提议解决方案是人类监督和保持“人类参与”，但这并非简单方案：不仅当前的AI代理设计方法阻碍了有效的人类监督，而且长时间使用AI系统本身也会削弱进行这种监督所需的认知能力。本立场论文认为，当前AI代理系统的开发与部署方法并不支持有效的人类监督——它们反而促使其退化。为解决此问题，AI代理发展的首要任务应是支持有效人类监督的情境目标和认知需求，将监督者的人类需求置于与AI代理能力同等重要的地位。为实践这一理念，我们将自动化与人机交互研究联系到AI代理流程，概述了去...

    arXiv:2608.23642v1 Announce Type: new  Abstract: AI agents pose significant risks as they are granted increasing autonomy. A commonly proposed solution is human oversight and keeping a ''human in the loop'', but this is not a simple solution: Not only do current approaches to AI agent design impede effective human oversight, but the cognitive capacities required for it are also themselves degraded by extended use of AI systems. This position paper argues that current approaches to the development and deployment of AI agent systems do not support effective human oversight -- they contribute to its degradation. To address this, a top priority in the advancement of AI agents should be supporting the situated goals and cognitive requirements of effective human oversight, treating the human needs of overseers at the same level of importance as AI agent capability. To put this idea into practice, we connect work on automation and human-computer interaction to AI agent processes, outlining de
    
[^178]: TRACE：一种自我进化的技能库，用于一致且具备限制意识的LLM代理

    TRACE: A Self-Evolving Skill Bank for Consistent, Limit-Aware LLM Agents

    [https://arxiv.org/abs/2608.22793](https://arxiv.org/abs/2608.22793)

    TRACE通过构建自我进化的技能库，在不修改模型权重的情况下，提升LLM代理在重复任务中的一致性和限制意识，弥合了单次成功与一致成功之间的可靠性差距。

    

    arXiv:2608.22793v1 公告类型：交叉 摘要：在面向用户的产品中可靠部署LLM代理，不仅取决于原始任务解决能力，还取决于一致性和限制意识：即在重复试验中表现相同，并识别请求何时无法或暂时无法安全完成。CAR-bench在车载助手领域揭示了这一可靠性缺口：一个由LLM模拟的用户发出不完整或模糊的请求，要求代理通过多轮对话和工具使用来解决不确定性，同时严格遵守领域政策。即使是前沿模型，在其至少能解决一次（Pass@3）和跨试验一致解决（Pass^k）之间也显示出显著差距。我们用TRACE（轨迹对比进化）弥合了这一差距，该方法在不修改模型权重的情况下，迭代改进基于技能的代理的行为知识。这些知识被组织为一个可检索的模块化技能库，每个技能编码一个自包含的行为模式。

    arXiv:2608.22793v1 Announce Type: cross  Abstract: Reliable deployment of LLM agents in user-facing products depends not on raw task-solving ability but on consistency and limit-awareness: behaving the same way across repeated trials, and recognizing when a request cannot, or cannot yet, be safely fulfilled. CAR-bench exposes this reliability gap in the domain of in-car assistants: an LLM-simulated user issues incomplete or ambiguous requests, requiring the agent to resolve uncertainty through multi-turn dialogue and tool use while strictly adhering to domain policies. Even frontier models show a substantial gap between what they can solve at least once (Pass@3) and what they solve consistently across trials (Pass^k). We bridge this gap with TRACE (TRAjectory-Contrastive Evolution), which iteratively improves a skill-based agent's behavioral knowledge without modifying model weights. This knowledge is organized as a Skill Bank of modular, retrievable skills, each encoding a self-contai
    
[^179]: K-Bench：在真实科学代理请求上衡量模型性能

    K-Bench: measuring model performance on real scientific agent requests

    [https://arxiv.org/abs/2608.21601](https://arxiv.org/abs/2608.21601)

    本论文提出K-Bench 01，一个基于真实科学请求的评估框架，发现当前前沿模型在满足领域科学家接受标准上均未达到阈值，其中gpt-5.6-sol表现最优但仍有不确定性。

    

    arXiv:2608.21601v1 公告类型：新 摘要：科学人工智能的基准测试大多是为评分而编写的：多项选择题、带参考答案的策划代理任务，或具有已知生成结构的模拟器。真实的科学请求则有所不同。它们规定不充分，携带附件，且缺乏基本事实。我们报告了K-Bench 01，一个从K-Dense Web实时用户流量中抽取的首轮请求构建的评估，并由九个前沿模型在相同沙盒环境中端到端运行，产生了1,602个完成的代理运行。三个盲审语言模型裁判根据八维评分标准对每次运行进行评分。在一个8锚点指示裁判认为领域科学家会接受该工作（仅需少量修改）的评分标准上，没有模型能在所有三位裁判下达标。gpt-5.6-sol具有最高的汇总平均值，为8.04，但其95%置信区间[7.80, 8.23]跨越了阈值，且三位裁判中有两位将claude-opus-5排在第一。

    arXiv:2608.21601v1 Announce Type: new  Abstract: Benchmarks for scientific artificial intelligence are mostly written to be scored: multiple-choice questions, curated agent tasks with reference solutions, or simulators with a known generative structure. Real scientific requests arrive differently. They are underspecified, they carry attachments, and lack ground truth. We report K-Bench 01, an evaluation built from first-turn requests sampled from live user traffic on K-Dense Web and run end to end by nine frontier models in identical sandboxes, yielding 1,602 completed agent runs. Three blinded language-model judges scored every run against an eight-dimension rubric. On a rubric whose 8-anchor instructs judges that a domain scientist would accept the work with minor edits, no model clears the line under all three judges. gpt-5.6-sol has the highest pooled mean, 8.04, but its 95% interval [7.80, 8.23] spans the threshold, and two of the three judges rank claude-opus-5 first instead. We 
    
[^180]: 反事实对比分析

    Counterfactual Contrastive Analysis

    [https://arxiv.org/abs/2608.19032](https://arxiv.org/abs/2608.19032)

    本文提出了一种基于对比分析的无分类器视觉反事实生成方法，通过分离和交换数据分布中的显著因素，生成模型无关且对分类器偏见不敏感的反事实解释。

    

    视觉反事实解释（VCEs）旨在通过生成输入图像的最小编辑且逼真的版本来解释图像分类器，从而改变分类器的预测。现有的VCE方法本质上依赖于分类器，因此容易受到分类器偏见和失败模式的影响，例如对捷径特征的敏感性和校准误差。在本文中，我们提出了一种基于对比分析（CA）的无分类器视觉反事实生成方法。给定两个对应不同类别（例如健康人和患者）的数据集，我们将两个数据集共有的生成因素与每个数据集特有的显著因素分离，并通过仅交换显著因素来生成反事实图像。通过直接作用于数据分布而非决策边界，我们的方法提供了模型无关的VCEs，对分类器偏见不太敏感。

    arXiv:2608.19032v1 Announce Type: cross  Abstract: Visual Counterfactual Explanations (VCEs) aim to explain image classifiers by generating minimally edited and realistic versions of an input image that change the classifier's prediction. Existing VCE methods are inherently classifier-dependent and therefore susceptible to classifier biases and failure modes, such as sensitivity to shortcut features and calibration errors. In this paper, we propose a classifier-free approach for visual counterfactual generation based on Contrastive Analysis (CA). Given two datasets corresponding to different classes (e.g., healthy and patients), we disentangle the generative factors that are common across the two datasets from those that are salient to each dataset, and generate counterfactual images by swapping only the salient factors. By operating directly on data distributions rather than decision boundaries, our method provides model-agnostic VCEs that are less sensitive to classifier biases. Our 
    
[^181]: 扩散逆问题的尺度一致后验动力学

    Scale-Consistent Posterior Dynamics for Diffusion Inverse Problems

    [https://arxiv.org/abs/2608.15144](https://arxiv.org/abs/2608.15144)

    本文提出一种尺度一致的后验动力学方法，通过重标定坐标、对数信噪比组织代理和冻结目标校正器，构建可处理的连续SDE，有效解决扩散逆问题中条件分数的难解性。

    

    arXiv:2608.15144v1 公告类型：交叉 摘要：使用预训练扩散先验进行后验采样，受条件分数控制，其中间似然分量通常难以处理。我们从理想的一参数后验SDE族出发，其中随机性参数控制概率流传输和随机探索，而不改变后验边缘分布。为了获得可处理的模型，我们在重标定的干净图像坐标中表达似然，并使用对数信噪比来组织所得的后验代理。通过前向算子投影扩散不确定性，得到噪声条件协方差路径，其目标接近干净后验。由于这些目标的端点一致性不能确保代理传输遵循它们，我们将传输与冻结目标的Langevin校正器交错，生成连续代理SDE。我们使用外部Lie--Trotter分裂和方差减少对此模型进行离散化。

    arXiv:2608.15144v1 Announce Type: cross  Abstract: Posterior sampling with a pretrained diffusion prior is governed by a conditional score whose intermediate likelihood component is generally intractable. We begin from an ideal one-parameter posterior SDE family in which a stochasticity parameter controls probability-flow transport and stochastic exploration without changing the posterior marginals. To obtain a tractable model, we express the likelihood in a rescaled clean-image coordinate and use log-SNR to organize the resulting posterior proxies. Projecting the diffusion uncertainty through the forward operator then yields a noise-conditioned covariance path whose targets approach the clean posterior. Because endpoint consistency of these targets does not ensure that a surrogate transport follows them, we interleave the transport with a frozen-target Langevin corrector, producing a continuous surrogate SDE. We discretize this model with an outer Lie--Trotter splitting and a variance
    
[^182]: 因果世界模型的统一视角：从观察到表征再到结构

    A Unifying Perspective on Causal World Models: From Observations to Representations to Structure

    [https://arxiv.org/abs/2608.13456](https://arxiv.org/abs/2608.13456)

    本文提出因果世界模型（CWMs）的统一形式化定义，强调世界模型需超越生成能力，捕捉实体属性及交互，以支持预测、规划和决策。

    

    世界模型（WM）日益被视为能够预测、规划并在训练分布之外行动的智能体的基础。在本文中，我们从因果视角，在多个抽象层次上研究世界模型，范围从感知观察到构建控制环境动态的结构的概念表征。我们认为，有用的世界模型必须超越生成能力本身：它们还应捕捉实体属性、实体间交互以及实体与环境间交互，这些交互决定并解释了系统的动态。我们提供了基于其预期支持任务的因果世界模型（CWMs）的形式化定义，将世界建模与因果表征学习、以对象为中心的学习、因果发现、结构因果模型以及基于模型的决策制定中的现有工作联系起来。最后，我们将CWMs与可识别性文献相关联。

    arXiv:2608.13456v1 Announce Type: new  Abstract: World Models (WM) are increasingly seen as a foundation for intelligent agents that can predict, plan, and act beyond their training distribution. In this paper, we study WMs from a causal perspective across multiple levels of abstraction, ranging from perceptual observations to building a conceptual representation of the structure governing the environment dynamics. We argue that useful WMs must go beyond generative capabilities alone: they should also capture entity properties, entity-to-entity interactions, and entity-to-environment interactions that determine and explain the dynamics of a system. We provide a formal definition of Causal WMs (CWMs) grounded in the tasks they are intended to support, connecting world modelling with existing work in causal representation learning, object-centric learning, causal discovery, structural causal models, and model-based decision-making. Finally, we relate CWMs to the literature on identifiabi
    
[^183]: Ex-Omni-2D：具备原生视觉形象的表达性全模态对话模型

    Ex-Omni-2D: Expressive Omni-Modal Dialogue Models with Native Visual Presence

    [https://arxiv.org/abs/2608.10720](https://arxiv.org/abs/2608.10720)

    Ex-Omni-2D 提出通过“视觉思维计划”协调文本、个性化语音与基于参考视频的生成，使全模态对话智能体在语音回答的同时拥有原生视觉形象，并可将全序列教师模型蒸馏为少步骤的流式学生模型。

    

    全模态对话模型能够理解多模态输入并合成语音回复，但语音回答仍然使智能体在视觉上“缺席”。我们提出了 Ex-Omni-2D，一个能够以协调的文本、个性化语音和基于参考条件的视频来回答多模态查询的框架。该对话模型首先为场景、情感和动作编写结构化的“视觉思维计划”（Visual Thought Plan, VTP），然后生成回复文本和多码本语音单元。这些语音单元被解码为音频并与视频帧对齐，为语音模块和虚拟形象模块提供共同的时间同步信号，同时允许二者从不同的数据源中学习。视频模块作为一个全序列的“教师”模型进行训练，其条件包括参考外观、VTP 语义以及帧对齐的语音单元。我们进一步探索将其蒸馏为一个少步骤、块因果的“流式学生”模型；其前缀流式机制携带……（原文此处截断）

    arXiv:2608.10720v2 Announce Type: replace  Abstract: Omni-modal dialogue models can understand multimodal inputs and synthesize spoken replies, but a spoken answer still leaves the agent visually absent. We introduce \textbf{Ex-Omni-2D}, a framework that answers a multimodal query with coordinated text, personalized speech, and reference-conditioned video. The dialogue model first writes a structured \textit{Visual Thought Plan} (VTP) for scene, emotion, and motion, then generates the response text and multi-codebook speech units. These speech units are decoded into audio and aligned with video frames, giving the speech and avatar modules a common timing signal while allowing them to learn from different data sources. The video module is trained as a full-sequence Teacher conditioned on reference appearance, VTP semantics, and frame-aligned speech units. We further explore to distill it into a few-step block-causal \emph{Streaming Student}; its Prefix Streaming mechanism carries the pr
    
[^184]: WDL-OPD：通过混合约束协同训练实现的弱驱动在线策略蒸馏

    WDL-OPD: Weak-Driven On-Policy Distillation via Mixture-Constrained Co-Training

    [https://arxiv.org/abs/2608.09447](https://arxiv.org/abs/2608.09447)

    提出了WDL-OPD方法，通过锚定策略与辅助策略的双策略混合约束协同训练来稳定在线策略蒸馏的反馈回路，在Qwen3的1.7B和4B规模实验中取得了最优效果。

    

    在线策略蒸馏（OPD）通过在从学生模型自身采样的轨迹上将学生模型与教师模型对齐，减少了离线蒸馏中的训练-测试状态不匹配问题。然而，同样的反馈回路可能并不稳定：每次更新都会同时改变策略本身以及下一次更新所基于的状态。我们提出了WDL-OPD，这是一种包含两个可训练策略的混合约束协同训练方法。锚定策略生成每一次轨迹采样，辅助策略评估相同的已访问状态，并通过反向KL散度将两者token分布的几何混合与冻结的教师模型进行匹配。两个策略均接收梯度。我们证明，冻结辅助策略可以恢复出一个与OPD²和W2S-OPD密切相关的锚定加对比代理目标，而联合训练则创造了静态增量无法表达的分支级自由度。在1.7B和4B规模的Qwen3记录实验中，WDL-OPD产生了最强的性能表现（摘要在此处被截断）。

    arXiv:2608.09447v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) aligns a student with a teacher on trajectories sampled from the student itself, reducing the train-test state mismatch of offline distillation. The same feedback loop can nevertheless be unstable: each update changes both the policy and the states on which the next update is computed. We introduce WDL-OPD, a mixture-constrained co-training method with two trainable policies. An anchor policy generates every rollout, an auxiliary policy evaluates the same visited states, and a geometric mixture of their token distributions is matched to a frozen teacher by reverse KL. Both policies receive gradient. We show that freezing the auxiliary recovers an anchor-plus-contrast proxy target closely related to OPD$^2$ and W2S-OPD, whereas joint training creates branch-level degrees of freedom that a static delta cannot express. In recorded Qwen3 experiments at 1.7B and 4B scale, WDL-OPD produces the strongest s
    
[^185]: 基于增量知识的神经符号推理用于样本高效的分层强化学习

    Neurosymbolic Reasoning with Incremental Knowledge for Sample Efficient Hierarchical Reinforcement Learning

    [https://arxiv.org/abs/2608.02993](https://arxiv.org/abs/2608.02993)

    提出了一种融合可更新增量知识的神经符号分层强化学习框架，将高层符号规划与低层神经运动基元学习相结合，显著提升了稀疏奖励长程推理任务中的样本效率。

    

    （扁平）强化学习（RL）智能体在需要长程推理的稀疏奖励环境中面临重大挑战。一种提高样本效率的有效方法是将知识融入学习与决策过程。在标准的分层强化学习（HRL）中，知识以固定的、不可更新的形式编码（例如架构选择），并在整个学习过程中保持不变。在使用固定HRL的情况下，在获取足够的环境知识之前，利用探索过程中学习到的增量知识进行推理是不切实际的，这导致样本效率低下。在本工作中，我们提出了带有增量知识的神经符号分层强化学习方法：符号高层组件在可更新的增量知识表示上执行符号规划（例如使用D*算法），而低层目标条件神经模块通过经验并借助奖励塑形来学习运动基元。在导航任务上的实验……（摘要原文在此处截断）

    arXiv:2608.02993v2 Announce Type: replace  Abstract: (Flat) Reinforcement Learning (RL) agents face significant challenges in environments with sparse rewards that require long-horizon reasoning. A compelling approach to improve sample efficiency is to incorporate knowledge into learning and decision-making. In standard Hierarchical RL (HRL), knowledge is encoded in a fixed, non-updatable form, such as architectural choices, and remains unchanged throughout learning. With fixed HRL, reasoning with incremental knowledge learned during exploration is impractical before sufficient environmental knowledge is acquired, leading to poor sample efficiency. In this work, we propose neurosymbolic HRL with {\em Incremental Knowledge (InK)}: symbolic high-level components perform {\em symbolic planning} (e.g. using $D^*$) on an updatable representation of current InK, while low-level goal-conditioned neural modules learn motion primitives through experience using reward shaping. Experiments on nav
    
[^186]: 在开放权重语言模型中读取与调控材料科学机制的表征

    Reading and Steering Representations of Materials-Science Mechanisms in an Open-Weight Language Model

    [https://arxiv.org/abs/2607.20058](https://arxiv.org/abs/2607.20058)

    该研究在开放权重语言模型中首次识别出材料科学机制内部表征的三个可实验验证的特征，并证明通过因果干预可以读取并调控模型对材料物理机制的表征。

    

    大语言模型能够回答科学问题，然而正确的输出并不能揭示模型是否真正表征或运用了其背后的物理规律。本研究使用三个开放权重的 Gemma 4 模型，识别出材料科学机制信息的三个可通过实验区分的特征：选择性概念可读性、定性本构取向的关系性编码，以及对受限工程答案的因果性、上下文相关控制。我们结合了匹配的直接词汇读出与雅可比词汇读出、无选项状态几何、包含60条定律的反事实基准以及因果干预方法。在50个保留的材料描述中，三个独立拟合的雅可比透镜重现了概念排名，且来自两种读出方法的无目标词集使得在盲测条件下能够识别10个机制族中的9个。另一个包含72个提示的独立基准产生了机制特异性的……

    arXiv:2607.20058v2 Announce Type: replace  Abstract: Large language models can answer scientific questions, yet a correct output does not reveal whether the model represents or uses the governing physics. Here, using three open-weight Gemma 4 models (google/gemma-4-E4B-it, google/gemma-4-12B-it, google/gemma-4-31B-it) we identify three experimentally separable signatures of materials-science mechanism information: selective concept readability, relational encoding of qualitative constitutive orientation, and causal, context-dependent control of constrained engineering answers. We combine matched direct and Jacobian vocabulary readouts, option-free state geometry, a 60-law counterfactual benchmark and causal interventions. In 50 held-out materials descriptions, three independently fitted Jacobian lenses reproduced concept ranks, and target-free word sets from both readouts enabled blinded identification of 9 of 10 mechanism families. A separate 72-prompt benchmark produced mechanism-spe
    
[^187]: X-Translator：一个实时多语言说话人感知的语音到语音翻译系统

    X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System

    [https://arxiv.org/abs/2607.17544](https://arxiv.org/abs/2607.17544)

    X-Translator是一个低成本、模块化的级联实时语音到语音翻译系统，通过会话级运行时控制器整合流式ASR、机器翻译与提示条件TTS，并利用增量片段提交和说话人提示，在多说话人长对话场景中实现低延迟翻译并保持说话人一致性。

    

    实时语音到语音翻译（S2ST）系统必须在翻译质量、延迟、语音自然度和说话人一致性之间取得平衡。公开发表的S2ST系统在直接翻译、多语言、流式和富有表现力的建模方面取得了进展，同时专有产品和API也日益向用户开放实时翻译功能。然而，对于开放且可复现的系统而言，实际部署仍然充满挑战，尤其是在长篇和多说话人对话场景中，部分ASR中间假设不稳定、话轮边界模糊，且目标语音必须使用合适的说话人提示来生成。我们提出了X-Translator，这是一个低成本的模块化级联S2ST系统，通过会话级运行时控制器将流式ASR、机器翻译和提示条件TTS相结合。该系统使用增量片段提交机制将不稳定的ASR流转换为可直接翻译的单元，并采用在线说话人……（原文摘要至此截断）

    arXiv:2607.17544v1 Announce Type: cross  Abstract: Real-time speech-to-speech translation (S2ST) systems must balance translation quality, latency, speech naturalness, and speaker consistency. Publicly documented S2ST systems have advanced direct, multilingual, streaming, and expressive modeling, while proprietary products and APIs increasingly expose real-time translation capabilities to users. However, practical deployment remains challenging for open and reproducible systems, especially in long-form and multi-speaker conversations where partial ASR hypotheses are unstable, turn boundaries are ambiguous, and target speech must be generated with an appropriate speaker prompt. We present X-Translator, a low-cost modular cascaded S2ST system that combines streaming ASR, machine translation, and prompt-conditioned TTS through a session-level runtime controller. The system uses incremental segment commitment to convert unstable ASR streams into translation-ready units, and an online speak
    
[^188]: 询问两次，观察两次：提示回声解决视觉语言模型中的“问题优先悖论”

    Ask Twice, Look Twice: Prompt Echoing Resolves the Question-First Paradox in Vision-Language Models

    [https://arxiv.org/abs/2607.15565](https://arxiv.org/abs/2607.15565)

    研究揭示了视觉语言模型中“问题优先悖论”的机制——虽然前置问题能引导感知，但被数百个图像token遮挡的问题无法被答案token读取，并据此提出在图像后重复问题的“提示回声”这一无需训练的简单修复方法。

    

    在视觉语言模型（VLM）的提示中，问题应该放在图像之前还是之后？直觉告诉我们应该放在前面：知道要问什么应该能引导模型关注正确的位置。然而，在各个视觉问答基准测试中，问题优先的提示方式始终表现不如前沿VLM所推荐的图像优先排序，我们将这一现象称为“问题优先悖论”。我们将这一悖论追溯到VLM计算两个阶段之间的冲突。通过Logit-lens和注意力探针分析显示，问题优先的提示确实会引导感知，使图像块表示向问题相关概念偏移。但在下游计算中，被困在数百个图像token之后的问题几乎没有被答案token注意到，答案token转而依赖图像驱动的、往往是错误的答案。因果注意力消融实验证实，只有当问题位于图像之后时，答案token才会读取问题。这一诊断带来了一种无需训练的修复方法：问题回声（即在图像之后重复问题）……

    arXiv:2607.15565v2 Announce Type: replace-cross  Abstract: Where should the question go in a vision-language model (VLM) prompt: before the image or after it? Intuition says before: knowing what is asked should tell the model where to look. Yet across visual question answering benchmarks, question-first prompting consistently underperforms the image-first ordering recommended for frontier VLMs, a phenomenon we term the question-first paradox. We trace this paradox to a conflict between two stages of VLM computation. Logit-lens and attention probes show that question-first prompting steers perception, shifting image patch representations toward question-relevant concepts. But downstream, stranded behind hundreds of image tokens, the question is barely attended by the answer token, which instead commits to image-driven, often wrong answers. Causal attention knockout confirms that the answer reads the question only when it follows the image. This diagnosis yields a training-free fix: ques
    
[^189]: PalmClaw：一个面向手机的原生端上智能体框架

    PalmClaw: A Native On-Device Agent Framework for Mobile Phones

    [https://arxiv.org/abs/2607.13027](https://arxiv.org/abs/2607.13027)

    PalmClaw 是一个原生运行在手机上的开源智能体框架，直接在设备端管理会话、记忆、技能与工具，并将设备能力封装为可调用的设备工具，从而突破了传统依赖 GUI 操作的移动智能体的局限。

    

    大语言模型（LLM）智能体已经超越了单纯生成回复的阶段，转向通过调用工具、观察结果并迭代决定下一步行动来执行多步骤任务。大多数智能体系统运行在桌面电脑或服务器上，以支持工具使用和任务自动化。移动设备同样是重要的智能体运行环境，因为它们普及度高、易于访问，并包含用户的数据、传感器和日常使用的应用程序。现有的移动智能体主要通过点击、滑动、输入等图形用户界面（GUI）操作来控制智能手机，这种方式往往形成冗长且依赖界面的操作序列，无法直接访问设备能力，并且使执行边界难以界定。我们提出了 PalmClaw，一个开源的智能体框架，它原生运行在手机上，直接在设备端管理会话、记忆、技能、工具和智能体循环。PalmClaw 将设备能力以设备工具的形式暴露……（原文摘要在此处截断）

    arXiv:2607.13027v2 Announce Type: replace-cross  Abstract: Large Language Model (LLM) agents have moved beyond generating responses to executing multi-step tasks by calling tools, observing the results, and iteratively deciding the next action. Most agent systems run on desktops or servers, which support tool use and task automation. Mobile devices are also important agent environments because they are widely accessible and contain users' data, sensors, and daily-use applications. Existing mobile agents mainly operate smartphones through graphical user interface (GUI) actions such as tapping, swiping, and typing, which often form long, interface-dependent sequences, cannot directly access device capabilities, and make execution boundaries difficult to define. We present PalmClaw, an open-source agent framework that runs natively on mobile phones and manages the sessions, memory, skills, tools, and agent loop directly on the device. PalmClaw exposes device capabilities as device tools w
    
[^190]: 弯曲权重空间中的学习：用于改进优化的指数-线性权重重参数化

    Learning in Curved Weight Space:Exponential-Linear Weight Reparameterization for Improved Optimization

    [https://arxiv.org/abs/2607.09967](https://arxiv.org/abs/2607.09967)

    提出一种将对称指数路径与线性路径相结合的权重重参数化方法，使加性优化更新转化为与权重幅值成比例的有效变化，从而改善神经网络的优化效果。

    

    许多神经网络操作本质上具有乘性而非加性：将范数减半或加倍在相对意义上是类似操作，但若采用线性步骤则需要不相等的优化距离。诸如Adam之类的自适应优化器会对每个坐标进行归一化更新，但更新步骤仍然是加性的；幅值差异很大的权重会接收到大小相近的绝对变化，从而产生差异巨大的相对扰动。我们提出了\method（\methodshort），这是一种针对神经网络的权重重参数化方法，它将带符号感知的对称指数路径与类恒等映射的线性路径相结合。对称指数路径对于较小的原始权重近似线性，但在较大幅值处曲率逐渐增大。对数空间中的加性更新可映射为有效权重空间中与幅值成比例的变化。线性路径则提供了穿越该变换的直接通道，我们假设……

    arXiv:2607.09967v3 Announce Type: replace-cross  Abstract: Many neural networks operations have a multiplicative nature rather than additive: halving or doubling a norm are analogous relatively but require unequal optimization distances when taking linear steps. Adaptive optimizers such as Adam normalize updates per coordinate, but update steps remain additive; weights with very different magnitudes receive similarly sized absolute changes, producing very different relative perturbations. We introduce \textbf{\method} (\textbf{\methodshort}), a weight reparameterization for neural networks that combines a sign-aware symmetric-exponential pathway with an identity-like linear pathway. The symmetric-exponential pathway is near-linear for small raw weights but increasingly curved at larger magnitudes. Additive updates in logarithmic space map to magnitude-proportional changes in effective weight space. The linear pathway provides a direct route through the transform that we hypothesize sta
    
[^191]: PCBWorld：一个基于引擎的PCB设计自动化基准环境

    PCBWorld: A Benchmark Environment for Engine-Grounded PCB Design Automation

    [https://arxiv.org/abs/2607.05915](https://arxiv.org/abs/2607.05915)

    PCBWorld是一个基于KiCad EDA引擎构建的开源PCB布线环境与基准，使RL和LLM智能体能像人类工程师一样通过引擎原生操作与DRC反馈交互式布线，并配套提供包含679块真实开源电路板的数据集及八项引擎检查的评估指标。

    

    PCB布线是在严格的设计规则约束下用铜导线连接电路板各网络的任务，然而基于学习的方法目前仍落后于基于规则的路由器。我们提出了PCBWorld，一个基于电子设计自动化（EDA）引擎KiCad构建的开源、引擎驱动的PCB布线环境。如同人类工程师一样，PCBWorld中的智能体通过引擎的原生操作交互式地对电路板进行布线，并接受其设计规则检查反馈的指导。该环境同时支持强化学习（RL）智能体和使用工具的大语言模型（LLM）智能体。与该环境相配套，PCBWorld-Bench提供了三个以原生.kicad_pcb格式存储的电路板数据集、两个可控的合成生成器以及679个真实开源电路板。无论采用何种布线方法，它都能使用八项经引擎检查的评估指标对任何已完成的电路板进行评分。在我们的实验中，PCBWorld中的智能体始终优于网格动作RL策略和开环LLM基线，并且RL策略……（摘要原文在此处被截断）

    arXiv:2607.05915v3 Announce Type: replace  Abstract: PCB routing is the task of connecting the nets of a board with copper traces under strict design rules, yet learning-based methods still lag behind rule-based routers. We introduce PCBWorld, an open-source engine-grounded PCB routing environment built on KiCad, an electronic design automation (EDA) engine. As a human engineer does, agents in PCBWorld interactively route a board through the engine's native operations, guided by its Design Rule Check (DRC) feedback. The environment supports both RL and tool-using LLM agents. Alongside the environment, PCBWorld-Bench provides three board datasets in the native .kicad_pcb format, two controllable synthetic generators and 679 real open-source boards. It scores any completed board with eight engine-checked evaluation metrics, regardless of the routing method. In our experiments, agents in PCBWorld consistently outperformed grid-action RL policies and open-loop LLM baselines, and an RL poli
    
[^192]: 基于大语言模型的测试预言机：权威来源分类法——一项系统性文献综述

    LLM-Based Test Oracles: Source-of-Authority Taxonomy -- A Systematic Literature Review

    [https://arxiv.org/abs/2607.05031](https://arxiv.org/abs/2607.05031)

    本综述首次按权威来源对LLM测试预言机进行分类，发现超过半数预言机在无规范情况下仅依赖模型训练知识作出判决，揭示了该领域信任基础的隐患。

    

    摘要：大语言模型（LLMs）越来越多地通过编写测试预言机或直接充当预言机来决定软件行为是否正确。然而，两个预言机可能看起来相同，却基于不同的依据：一个断言编码了书面规范，另一个仅依赖于模型在训练中学到的内容。先前的二次研究按形式或技术对预言机进行分类，很少依据决定判决可信度的属性——即其权威来源。本系统性文献综述按照2020年系统综述和元分析首选报告项目（PRISMA）指南进行，筛选了2,436条记录至54项纳入研究，并通过引文搜索（滚雪球法）扩展至总计83项。我们沿着三个维度阅读了文献集：预言机权威的来源、其采取的形式以及裁决其的机制。语料库中略多于一半的预言机在没有规范的情况下做出判决。这就是关键所在。

    arXiv:2607.05031v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly decide whether software behaves correctly, either by writing a test oracle or by acting as one. Yet two oracles can look identical and rest on different ground: one assertion encodes a written specification, another only what the model learned in training. Prior secondary studies sort oracles by form or by technique, rarely by the property that governs how far a verdict can be trusted: where its authority comes from. This systematic literature review, reported under the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 2020 guidelines, screens 2,436 records to 54 included studies, extended by citation searching (snowballing) to 83 in total. We read the corpus along three axes: the source of an oracle's authority, the form it takes, and the mechanism that adjudicates it. Just over half of the corpus reaches a verdict with no specification at all. That is what le
    
[^193]: KARMA：基于知识图谱的自动推理具体化与对齐

    KARMA: Knowledge graph-based Automated Reasoning Materialization and Alignment

    [https://arxiv.org/abs/2607.03166](https://arxiv.org/abs/2607.03166)

    KARMA 通过在领域知识图谱上枚举模式约束路径生成槽位对齐的对比候选样本，并利用槽位并行对齐（SPA）将偏好监督精准路由至区分性实体槽位，从而解决了基于模板的对比合成中的分辨率不匹配问题。

    

    基于模板的对比合成具有良好的可扩展性，但其候选样本往往仅在少数实体槽位上存在差异，而序列级优化会将监督信号分散到大部分共享的模板上。我们将这一问题形式化为“分辨率不匹配问题”，并提出 KARMA 方法，该方法在领域知识图谱上枚举受模式约束的路径，并将其言语化为槽位对齐的对比候选样本。随后，槽位并行对齐应用解耦的槽位级目标函数，将偏好监督精准引导至具有区分性的实体槽位，其中槽位感知的掩码注意力可作为打包评估的可选实现。在生物医学、计算机科学和化学基准测试中，KARMA 优于基础 LLM 和相同数据下的 SFT 基线，并与序列级和词元级偏好方法相比表现更优。

    arXiv:2607.03166v2 Announce Type: replace-cross  Abstract: Template-based contrastive synthesis is scalable, but its candidates often differ only in a few entity-slots while sequence-level optimization spreads supervision over mostly shared templates. We formalize this as the Resolution Mismatch Problem and propose KARMA, which enumerates schema-constrained paths over domain knowledge graphs and verbalizes them into slot-aligned contrastive candidates. Slot-Parallel Alignment (SPA) then applies a decoupled slot-level objective to route preference supervision to discriminative entity-slots, with slot-aware masked attention serving as an optional packed-evaluation implementation. Across biomedical, computer-science, and chemistry benchmarks, KARMA outperforms base LLM and same-data SFT baselines, and compares favorably with sequence- and token-level preference methods.
    
[^194]: SNAP-FM：面向物理约束生成建模的稀疏非线性加速投影

    SNAP-FM: Sparse Nonlinear Accelerated Projection for Physics-Constrained Generative Modeling

    [https://arxiv.org/abs/2607.00095](https://arxiv.org/abs/2607.00095)

    提出SNAP-FM方法，利用样本批处理与局部PDE耦合所诱导的块稀疏结构，实现高效的批量非线性投影优化，使生成模型在推理时能精确满足物理守恒约束且计算开销大幅降低。

    

    生成模型已成为物理模拟的可扩展替代方案，但其输出无法保证遵守支配底层物理的守恒定律、边界条件和非线性不变量。约束采样可以填补这一空白，它在推理阶段精确执行此类约束而无需重新训练，但需要付出计算代价：投影、校正和轨迹优化步骤在采样过程中被反复执行，对于非线性约束而言，这些步骤的计算开销十分高昂。标准机器学习框架进一步加剧了这一问题：其稠密张量代数和有限的稀疏求解器可组合性掩盖了物理约束天然诱导的结构，使得高效的批量非线性优化在实践中难以实现。我们通过利用样本级批处理和局部PDE耦合在投影子问题中诱导的结构来解决这一瓶颈——即块稀疏结构（原文摘要在此处截断）。

    arXiv:2607.00095v2 Announce Type: replace-cross  Abstract: Generative models have emerged as scalable surrogates for physical simulation, yet they offer no guarantee that their outputs respect the conservation laws, boundary conditions, and nonlinear invariants that govern the underlying physics. Constrained sampling closes this gap, enforcing such constraints exactly at inference time without retraining, but at a computational cost: projection, correction and trajectory-optimization steps are repeated during sampling, with these steps becoming expensive for nonlinear constraints. Standard ML frameworks exacerbate this: their dense tensor algebra and limited sparse solver composability obscure the structure that physical constraints naturally induce, making efficient batched nonlinear optimization difficult to realize in practice. We address this bottleneck by exploiting the structure that sample-wise batching and local PDE couplings induce in the projection subproblems -- namely, bloc
    
[^195]: 学习选择而非重新学习：硬路由的推理LoRA混合模型

    Learning to Select, Not Relearn: Hard-Routed Mixtures of Reasoning LoRAs

    [https://arxiv.org/abs/2606.31413](https://arxiv.org/abs/2606.31413)

    提出Hard-Routed MoR-LoRA两阶段框架，通过单位尺度的硬top-1路由（而非软加权组合）选择冻结的推理LoRA专家，仅训练轻量级共享路由器和小型注意力LoRA即可实现多领域推理能力的集成。

    

    将独立训练的LoRA适配器组合成单个大语言模型对多领域适应非常有用，尤其适用于原始训练数据无法共享的场景。一种常见做法是对LoRA专家采用MoE风格的路由，但对于已冻结的预训练适配器，软加权组合可能会改变每个LoRA模块最初训练时所基于的单位尺度加性更新。我们提出了硬路由MoR-LoRA（Hard-Routed MoR-LoRA），这是一个通过单位尺度硬选择来组合冻结推理LoRA专家的两阶段框架。首先，使用可验证反馈的强化学习独立训练各领域的LoRA适配器，以获得推理专家；然后，冻结所有专家，从中蒸馏推理轨迹，仅训练一个轻量级共享路由器和一个小的注意力LoRA用于集成。路由器采用硬top-1路由，为每个token精确选择一个专家……

    arXiv:2606.31413v3 Announce Type: replace  Abstract: Composing independently trained LoRA adapters into a single large language model is useful for multi-domain adaptation, especially when the original training data cannot be shared. A common approach is to use MoE-style routing over LoRA experts, but for frozen pretrained adapters, soft weighted combinations can change the unit-scale additive update under which each LoRA module was originally trained. We propose \textbf{Hard-Routed MoR-LoRA}, a two-stage framework for composing frozen reasoning LoRA experts through unit-scale hard selection. First, domain-specific LoRA adapters are trained independently using reinforcement learning from verifiable feedback to obtain reasoning experts. Then, all experts are frozen, reasoning traces are distilled from them, and only a lightweight shared router together with a small attention LoRA is trained for integration. The router selects exactly one expert per token using hard top-1 routing, while 
    
[^196]: 变换器作为贝叶斯上下文实验者：平滑度自适应的高效平均处理效应估计

    Transformers as Bayesian In-Context Experimenters: Smoothness-Adaptive Efficient ATE Estimation

    [https://arxiv.org/abs/2606.31184](https://arxiv.org/abs/2606.31184)

    该论文提出将变换器训练为模仿贝叶斯后验Neyman教师的“上下文实验者”，通过上下文学习摊销序贯方差估计与处理分配过程，实现对平均处理效应的平滑度自适应高效估计。

    

    用于平均处理效应（ATE）的自适应实验需要随机化分配，以在有效推断与统计效率之间取得平衡。理想（oracle）设计是一个由未知的分组条件结果方差所支配的、依赖于协变量的Neyman规则。我们研究了这种序贯的方差估计与分配过程能否通过上下文学习被摊销。我们提出了贝叶斯上下文实验者：即经过训练以模仿贝叶斯后验Neyman教师的变换器策略。该教师利用实验历史更新关于潜在结果的非参数信念，从而分配后验Neyman处理概率。该设计收敛于oracle规则，支持高效的ATE推断。变换器通过基于注意力的充分统计量和投影梯度下降建设性地实现了这一映射，模仿了高斯级先验下的贝叶斯更新。为应对未知的结果平滑度……（原文在此处截断）

    arXiv:2606.31184v2 Announce Type: replace-cross  Abstract: Adaptive experiments for average treatment effects (ATE) require randomized allocations balancing valid inference with statistical efficiency. The oracle design is a covariate-dependent Neyman rule governed by unknown arm-conditional outcome variances. We investigate whether this sequential variance-estimation and allocation process can be amortized via in-context learning. We introduce Bayesian in-context experimenters: transformer policies trained to imitate a Bayesian posterior Neyman teacher. The teacher updates nonparametric beliefs over potential outcomes using experimental history to assign posterior Neyman treatment probabilities. This design converges to the oracle rule, supporting efficient ATE inference. Transformers constructively implement this mapping through attention-based sufficient statistics and projected gradient descent, imitating Bayesian updating for Gaussian-series priors. To address unknown outcome smoo
    
[^197]: 超越编译：评估忠实的自然语言到Lean语句形式化

    Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization

    [https://arxiv.org/abs/2606.31002](https://arxiv.org/abs/2606.31002)

    该论文提出将Lean编译与GPT-5.2和Gemini-2.5-Pro严格语义共识相结合的自动形式化评估标准，发现编译通过率会高估语义忠实度达3.0至29.0个百分点，且该标准与人类多数判断的一致率达89.7%。

    

    Lean能够验证生成的声明类型正确，但无法验证它表达了用户真正想要的语句。我们针对没有规范Lean目标的自动形式化研究两个问题：LLM评判者能否为人类语义审查提供可用的代理，以及编译在不同系统之间会在多大程度上高估忠实度。我们的评估标准将Lean编译与GPT-5.2和Gemini-2.5-Pro之间的严格语义共识相结合。在一个独立审计的随机样本上，该标准与人类多数意见在89.7%的案例中保持一致（Wilson 95%置信区间：82.1–94.3%）。在400个研究生水平语句上评估的八个系统中，每个系统都存在非零的编译-忠实度差距，其观测幅度介于3.0到29.0个百分点之间。完整的GPT-5.2工具增强代理表现出最大的差距，编译率达89.5%，而满足语义标准的仅为60.5%。人工审查、一个独立的第三族评判者以及BEq形式化交叉……

    arXiv:2606.31002v2 Announce Type: replace  Abstract: Lean verifies that a generated declaration is well typed, but not that it expresses the statement a user intended. We study two questions for autoformalization without canonical Lean targets: whether LLM judges can provide a usable proxy for human semantic review, and how much compilation overstates faithfulness across systems. Our criterion combines Lean compilation with strict semantic consensus between GPT-5.2 and Gemini-2.5-Pro. On an independently audited random sample, it agrees with human majority on 89.7\% of cases (Wilson 95\% CI: 82.1--94.3\%). Across eight systems evaluated on 400 graduate-level statements, every system has a nonzero compile--faithfulness gap, whose observed magnitude ranges from 3.0 to 29.0 percentage points. The full GPT-5.2 tool-augmented agent shows the largest gap, compiling 89.5\% while satisfying the semantic criterion on 60.5\%. Human review, an independent third-family judge, and a BEq formal cros
    
[^198]: 构造即忠实：面向多文档摘要的声明锚定归因方法

    Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization

    [https://arxiv.org/abs/2606.23989](https://arxiv.org/abs/2606.23989)

    提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。

    

    端到端大语言模型（LLM）能够生成流畅的多文档摘要，但仍容易产生幻觉，且其提供的归因通常较为粗糙（仅指向整篇文档或段落）并属于事后生成，导致每条摘要陈述都难以验证。我们重新审视模块化的“提取—选择—改写”范式，并将其中间表示重新构建为归因的基本单元。我们提出了CAMS（Claim-Anchored Multi-document Summarization，声明锚定多文档摘要）框架，该框架：(i) 从每个源文档中提取带有词元级溯源信息的原子声明；(ii) 跨文档聚类等价声明，同时标记源间冲突；(iii) 选择一个兼顾支持度与显著性的子集；(iv) 将所选内容改写为摘要，其中每个句子都锚定到一个经过支持性检验的声明，该声明可回链至一个或多个源文本片段。由于内容在生成之前就已完成定位，整个流程从构造上即是面向归因的。

    arXiv:2606.23989v3 Announce Type: replace-cross  Abstract: End-to-end large language models (LLMs) produce fluent multi-document summaries but remain prone to hallucination, and the attributions they offer are typically coarse (whole documents or passages) and generated post hoc, leaving each summary statement hard to verify. We revisit the modular Extract--Select--Rewrite paradigm and recast its intermediate representation as the unit of attribution. We present CAMS, a Claim-Anchored Multi-document Summarization framework that (i) extracts atomic claims with token-level provenance from every source document, (ii) clusters equivalent claims across documents while flagging inter-source conflicts, (iii) selects a support-aware and salient subset, and (iv) rewrites the selection into a summary in which every sentence is anchored to a support-checked claim that links back to one or more source spans. Because content is localized before it is realized, the pipeline is attribution-oriented b
    
[^199]: 学会不遗忘什么：从几千字节学习中获得的长时程智能体记忆

    Learning What Not to Forget: Long-Horizon Agent Memory from a Few Kilobytes of Learning

    [https://arxiv.org/abs/2606.20954](https://arxiv.org/abs/2606.20954)

    LRE是一种千字节级、仅用CPU、无需语言模型的学习型淘汰评分器，通过逐字提取保留任务关键历史信息，在智能体任务上以极低成本恢复保留完整历史93%的准确率，并将最坏情况峰值提示削减52%。

    

    长时间运行的语言模型系统会积累超出上下文窗口的交互历史，因此必须持续进行淘汰。当淘汰策略丢弃了任务关键细节时——例如登录时发放的访问令牌或下一次调用所需的路径——操作就会失败。我们提出了LRE（Learned Relevance Eviction，学习相关性淘汰），这是一个千字节规模、仅使用CPU、无需语言模型的评分器，它通过学习判断历史中的哪些单元对任务至关重要，并通过逐字提取的方式保留它们。在匹配预算的对比实验中，没有任何基线方法在准确率-成本平面上全面优于LRE。在智能体任务上，LRE恢复了保留完整历史93%的总体准确率（41.1 vs. 44.0），并在最简单的任务上超出其27%，同时无需任何压缩器调用，并将最坏情况下的峰值提示长度削减了52%。一项受控研究轨迹显示，LRE能够完成其他方法陷入循环的任务，其中一个任务比保留完整历史少用37%的调用次数即可完成。

    arXiv:2606.20954v2 Announce Type: replace-cross  Abstract: Long-running language-model systems accumulate interaction history that outgrows the context window, so they must continually evict. When an eviction policy drops a task-critical detail, for example an access token issued at login or a path the next call needs, the action fails. We present LRE (Learned Relevance Eviction), a kilobyte-scale, CPU-only, language-model-free scorer that learns which units of history are task-critical and keeps them by verbatim extraction. Under a matched-budget comparison, in our experiment, no baseline dominates LRE on the accuracy-cost plane. On agents, LRE recovers 93% of the aggregate accuracy of keeping the entire history (41.1 vs. 44.0) and exceeds it by 27% on the simplest tasks, while requiring zero compressor calls and cutting the worst-case peak prompt by 52%. A controlled study trace shows LRE completes tasks where the others loop, finishing one such task in 37% fewer calls than keeping e
    
[^200]: LLMZero：通过大语言模型智能体发现强化学习后训练的自适应训练策略

    LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents

    [https://arxiv.org/abs/2606.18388](https://arxiv.org/abs/2606.18388)

    LLMZero利用大语言模型智能体结合树搜索，通过在每个检查点诊断训练状态来自适应地优化RL后训练的多参数调度策略，在四个GRPO任务上比基础模型提升9%-140%、比网格搜索提升6%-15%，并揭示了容量参数单调累积、正则化参数震荡变化的训练规律。

    

    强化学习（RL）后训练策略依赖于数据集，并呈现出一个反复出现的经验规律：容量参数在各阶段单调累积，而正则化参数则主要随着训练动态的变化而震荡。这一区别凸显了固定训练调度的一个潜在缺陷：由于强迫所有参数沿着僵化的路径变化，固定调度无法捕捉正则化所必须追踪的动态探索-利用权衡。我们通过LLMZero揭示了这一点——LLMZero是一个通过树搜索优化训练轨迹的智能体系统，它在每个检查点诊断训练中的病理现象，并提出协调的多参数转换方案。在四个多样化的GRPO任务中，LLMZero发现的策略相比基础模型提升了9%至140%，相比网格搜索相对提升了6%至15%，在相同计算预算下，其表现始终优于随机搜索和基于技能的智能体。

    arXiv:2606.18388v2 Announce Type: replace-cross  Abstract: RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction highlights a potential flaw in fixed training schedules: by forcing all parameters along rigid paths, they fail to capture the dynamic exploration-exploitation tradeoffs that regularization must track. We uncover this through LLMZero, an agentic system that optimizes training trajectories via tree search by diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across four diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% and over grid search by 6% to 15% (relative), consistently outperforming random search and a skill-based agent under a matched compute budget. The capacity--re
    
[^201]: SpecAlign：通过合成数据实现基于规范的大语言模型高效对齐

    SpecAlign: Efficient Specification-Grounded Alignment of Large Language Models via Synthetic Data

    [https://arxiv.org/abs/2606.16276](https://arxiv.org/abs/2606.16276)

    该论文提出了基于规范的对齐新范式及SpecAlign框架，通过结构化规则标注、可控规范实例化和多智能体对抗数据合成，直接从提供商的模型规范文档合成对齐数据，实现大语言模型与特定规范的高效对齐。

    

    随着大语言模型（LLM）在现实应用中的日益部署，对齐不再由单一通用的安全性或有用性标准所主导，而是由提供商或应用特定的模型规范所决定。这些规范通常篇幅较长、结构化程度高且频繁更新，然而现有的对齐流程缺乏将这些规范系统化地转化为训练信号的机制。在本文中，我们提出了基于规范的对齐，这是一种新的对齐范式，它将提供商编写的模型规范作为主要对齐目标，而非抽象原则或静态基准。为了实现这一范式，我们引入了SpecAlign，一个直接从规范文档合成对齐数据的框架。SpecAlign结合了结构化规则标注、可控的规范实例化和多智能体对抗性数据合成，以生成细粒度的、有界的……

    arXiv:2606.16276v3 Announce Type: replace  Abstract: As large language models (LLMs) are increasingly deployed in real-world applications, alignment is no longer governed by a single universal notion of safety or helpfulness, but instead by provider- or application-specific model specifications. These specifications are typically long, structured, and frequently updated, yet existing alignment pipelines lack a systematic mechanism to operationalize them as training signals. In this paper, we propose specification-grounded alignment, a new alignment paradigm that treats provider-authored model specifications as the primary alignment target rather than abstract principles or static benchmarks. To instantiate this paradigm, we introduce SpecAlign, a framework that synthesizes alignment data directly from specification documents. SpecAlign combines structured rule annotation, controllable specification instantiation, and multi-agent adversarial data synthesis to generate fine-grained, boun
    
[^202]: MeEvo：结合元认知演化与自然演化的自动启发式设计

    MeEvo: Metacognitive Evolution Combined with Natural Evolution for Automatic Heuristic Design

    [https://arxiv.org/abs/2606.14202](https://arxiv.org/abs/2606.14202)

    提出MeEvo框架，通过循环耦合自然演化与元认知演化并配合从探索到利用的算子平衡，实现了自动启发式设计中推理知识保留与种群级探索的协同增效。

    

    大语言模型通过推理和代码合成实现启发式的自动生成，推动了自动启发式设计（AHD）的发展。在基于大语言模型的自动启发式设计中，大语言模型对算法设计进行推理并生成可执行的启发式代码。现有架构主要采用两种范式：自然演化通过对代码进行交叉和变异来探索多样化策略，但丢弃了设计决策背后的推理轨迹，削弱了知识保留能力；元认知演化则保留这些推理轨迹并通过反思加以完善，但缺乏种群层面的重组，限制了探索能力。这些局限降低了在复杂问题上的搜索效率、稳定性和解的质量。为弥补这一空白，我们提出MeEvo——一个将自然演化与元认知演化循环耦合的自动启发式设计框架，并通过算子平衡实现从探索到利用的转变。

    arXiv:2606.14202v5 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have advanced Automatic Heuristic Design (AHD) by enabling heuristic generation through reasoning and code synthesis. In LLM-based AHD, the LLM reasons about algorithm design and generates executable heuristic code. Existing architectures adopt two main paradigms: Natural Evolution applies crossover and mutation to this code to explore diverse strategies, but discards the reasoning traces behind the design decisions, weakening knowledge retention; Metacognitive Evolution retains these reasoning traces and refines them through reflection, but lacks population-level recombination, limiting exploration. These limitations reduce search efficiency, stability, and solution quality on complex problems. To address this gap, we propose MeEvo, an AHD framework that cyclically couples Natural Evolution and Metacognitive Evolution with operator balance that shifts from exploration to exploitation. Natural Evolu
    
[^203]: GeoNatureAgent基准：面向前沿与开放权重基础模型的环境地理空间分析LLM智能体基准测试

    GeoNatureAgent Benchmark: Benchmarking LLM Agents for Environmental Geospatial Analysis Across Frontier and Open-Weight Foundation Models

    [https://arxiv.org/abs/2606.12821](https://arxiv.org/abs/2606.12821)

    提出首个环境地理空间分析智能体基准GeoNatureAgent，通过覆盖93个任务的结构化工具调用评估九个大语言模型，发现Claude Sonnet 4以60.8%的能力表现领先，DeepSeek V3.2紧随其后。

    

    环境科学家在数据整理上花费了不成比例的精力，而非专注于分析本身。新型AI智能体可以成为有用的工具，但目前尚无基准来评估通过结构化工具调用、针对真实API自动执行环境地理空间工作流的AI智能体。我们提出了GeoNatureAgent基准，这是首个通过结构化工具调用生产级地理空间API来运行的环境分析智能体基准。该基准包含18个类别共93个任务。任务通过一个开放的、可自托管的地理空间API进行评估，该API通过十六个工具提供覆盖西班牙和葡萄牙的三个环境指标。我们评估了九个前沿及开放权重的大语言模型，将能力与单案例成本作为两个正交维度进行报告。结果表明：(1) Claude Sonnet 4以60.8% ± 0.8%取得最高能力，DeepSeek V3.2以56.3% ± 3.1%紧随其后，而其他模型均未超过51%；

    arXiv:2606.12821v2 Announce Type: replace  Abstract: Environmental scientists spend disproportionate effort on data wrangling rather than analysis. New AI agents can be a helpful tool, but no benchmark exists to evaluate AI agents that automate environmental geospatial workflows through structured tool calling against real APIs. We introduce the GeoNatureAgent Benchmark, the first benchmark for environmental analysis agents that operate via structured tool calls to a production-style geospatial API. The benchmark comprises 93 tasks across 18 categories. Tasks are evaluated against an open, self-hostable geospatial API that serves three environmental indicators across Spain and Portugal via sixteen tools. We evaluate nine frontier and open-weight LLMs, reporting capability and per-case cost as orthogonal axes. Results manifest that (1) Claude Sonnet 4 achieves the highest capability at 60.8% +/- 0.8%, followed closely by DeepSeek V3.2 at 56.3% +/- 3.1%, while no other model exceeds 51%;
    
[^204]: StatefulDiscovery：开放式科学发现中的证据校准式主张形成

    StatefulDiscovery: Evidence-Calibrated Claim Formation in Open-Ended Scientific Discovery

    [https://arxiv.org/abs/2606.11851](https://arxiv.org/abs/2606.11851)

    StatefulDiscovery通过将研究状态外化，协调前沿选择、证据获取与主张裁决，解决了开放式科学发现中的证据校准问题，从而产出更多证据充分且价值高的科学主张。

    

    开放式科学发现要求智能体超越为预定义问题执行分析的范围。在多轮探索过程中，发现型智能体必须决定哪些现象值得深入研究，同时避免过度解读——即新出现的主张超出了分析所能支撑的证据范围。这就产生了一个证据校准问题：探索轨迹必须与主张状态相耦合，从而使证据既能指导下一步研究什么，又能决定可以提出什么主张。我们提出了StatefulDiscovery，这是一个将研究状态显式外化的发现框架，并利用该状态来协调前沿选择、证据获取和主张裁决。我们在40个真实数据发现任务上对StatefulDiscovery进行了评估。与多个基线方法相比，StatefulDiscovery产出了更多被判定为既有充分证据支撑又具高价值的主张。消融实验表明……（原文摘要在此处截断）

    arXiv:2606.11851v2 Announce Type: replace  Abstract: Open-ended scientific discovery asks agents to move beyond executing analyses for predefined questions. Across multiple rounds of exploration, a discovery agent must decide which phenomena warrant investigation while avoiding overinterpretation, where emerging claims exceed the evidential scope of the analyses. This creates an evidence-calibration problem: the exploration trajectory must be coupled with claim status so that evidence can guide both what to investigate next and what can be claimed. We introduce \textsc{StatefulDiscovery}, a discovery framework that externalizes investigation state and uses it to coordinate frontier selection, evidence acquisition, and claim adjudication. We evaluate \textsc{StatefulDiscovery} across 40 real-data discovery tasks. Compared with several baselines, \textsc{StatefulDiscovery} produces more claims overall judged to be both well-supported and high-value. Ablations indicate distinct roles for 
    
[^205]: 重复不匹配：为什么数据混合实验无法扩展以及如何修复它们

    Repetition Mismatch: Why Data Mixture Experiments Don't Scale and How to Fix Them

    [https://arxiv.org/abs/2606.07597](https://arxiv.org/abs/2606.07597)

    论文揭示了预训练数据混合实验无法从小规模外推到大规模的主要原因是高质量数据的重复率随训练预算变化而改变最优混合比例，并提出通过匹配目标重复率的子采样方法，仅用1/16的目标token即可恢复接近最优的数据混合配置。

    

    预训练数据混合通常通过运行小规模实验并外推到目标训练预算来进行调整。当高质量数据稀缺且必须重复使用时，这种外推经常失败，但失败的根源尚未被隔离出来。我们证明一个主要罪魁祸首是重复不匹配：由于高质量数据集规模较小，其重复率会随着训练预算的增长而变化，从而以小规模代理实验无法预料的方式改变最优混合比例。一种匹配目标重复率的子采样程序可以控制这种效应。在将有限的高质量数据与网络爬取数据相结合的双源设置中，仅使用目标token数量1/16的单次重复控制实验，就能为11.7亿参数模型在Wiki-Text上恢复出与最优值相差0.10以内的混合比例，而在没有重复控制的情况下误差为0.85。获得可比的准确

    arXiv:2606.07597v2 Announce Type: replace-cross  Abstract: Pre-training data mixtures are commonly tuned by running small-scale experiments and extrapolating to the target training budget. When high-quality data is scarce and must be repeated, this extrapolation frequently fails, but the source of the failure has not been isolated. We show that a primary culprit is a repetition mismatch: because high-quality datasets are small, their repetition rate changes as the training budget grows, shifting the optimal mixture in ways that small-scale proxy experiments do not anticipate. A subsampling procedure that matches the target repetition rate controls for this effect. In a two-source setting combining limited high-quality data with web crawl, a single repetition-controlled experiment using only 1/16 of the target tokens recovers a mixture within 0.10 of the optimum on Wiki-Text for a 1.17B parameter model, compared to an error of 0.85 without repetition control. Achieving comparable accura
    
[^206]: TEVI：通过稀疏自编码器实现文本条件化的视觉表征编辑，以改进视觉-语言对齐

    TEVI: Text-Conditioned Editing of Visual Representations via Sparse Autoencoders for Improved Vision-Language Alignment

    [https://arxiv.org/abs/2606.07451](https://arxiv.org/abs/2606.07451)

    TEVI框架利用稀疏自编码器解耦图像嵌入，并通过文本条件化的掩码模块只保留与文本描述相符的信息、剔除多余内容，从而改善CLIP等视觉-语言模型中图像-文本嵌入的对齐问题。

    

    像CLIP这样的视觉-语言模型由于其共享的图像-文本嵌入空间而在多种任务中非常有用。尽管如此，图像和文本嵌入之间的对齐往往不佳，从而影响下游任务的性能。近期的研究假设这可以归因于信息不平衡问题：图像所包含的信息多于其文本描述所涵盖的内容。在这项工作中，我们提出了TEVI，一个利用文本描述作为信号来决定从图像嵌入中保留哪些信息的框架。具体而言，我们使用稀疏自编码器来解耦图像嵌入，并训练一个掩码模块，根据给定的文本描述选择性地重建嵌入。在使用合成文本描述的受控实验设置中，我们展示了TEVI能够有效地保留文本描述所涉及的属性，同时丢弃其他属性。我们发现这一方法可以扩展到在自然图像上训练的CLIP模型，其中TEVI学会了进行有意义的掩码操作，并支持基于内容的检索。

    arXiv:2606.07451v2 Announce Type: replace-cross  Abstract: Vision-language models such as CLIP are highly useful for diverse tasks due to their shared image-text embedding space. Despite this, the image and text embeddings are often poorly aligned, affecting downstream performance. Recent work has hypothesized that this can be attributed to an information imbalance: images contain more information than their captions describe. In this work, we propose TEVI, a framework that uses captions as a signal for what to retain from image embeddings. Specifically, we use sparse autoencoders to disentangle image embeddings and train a masking module to selectively reconstruct the embedding based on a given caption. In a controlled setup with synthetic captions, we show that TEVI is effective at preserving caption-described attributes while discarding others. We find that this extends to CLIP models trained on natural images, where TEVI learns to mask meaningfully and allows retrieval based on con
    
[^207]: SV-Detect：基于转向向量的AI生成文本检测

    SV-Detect: AI-generated Text Detection with Steering Vectors

    [https://arxiv.org/abs/2606.07313](https://arxiv.org/abs/2606.07313)

    本文提出SV-Detect，一种利用从冻结语言模型隐藏表示中提取的转向向量构建逐层投影特征来检测AI生成文本的方法，在跨领域、跨源模型及编辑攻击等分布偏移场景下均实现了强大的检测性能。

    

    在分布偏移（如跨领域、跨源模型和编辑攻击的迁移）情况下，检测AI生成的文本尤其困难。我们提出了一种基于转向向量的AI生成文本检测器，该转向向量从冻结语言模型的隐藏表示中提取。在每一层，我们构建一个能够区分人类撰写文本与AI生成文本的方向，并通过每个输入与这些方向的逐层对齐程度来表示该输入。在这些投影特征上训练的轻量级分类器产生最终的检测分数。我们的方法在分布内和分布偏移下均取得了强大的性能，包括跨领域、跨源模型，以及诸如润色和重写等机器编辑变换。解释性分析表明，学习到的方向与可识别的风格线索相一致，同时捕捉到了超越表面特征的大量额外信号。这些结果使AI...

    arXiv:2606.07313v2 Announce Type: replace-cross  Abstract: Detecting AI-generated text is especially difficult under distribution shift, such as transfer across domains, source models, and editing attacks. We propose an AI-generated text detector based on steering vectors extracted from the hidden representations of a frozen language model. At each layer, we construct a direction that separates human-written from AI-generated text, and represent each input by its layer-wise alignment with these directions. A lightweight classifier trained on these projection features yields the final detection score. Our method achieves strong performance both in-distribution and under distribution shift, including across domains, source models, and machine-editing transformations such as polishing and rewriting. Interpretation analyses show that the learned directions align with recognizable stylistic cues while capturing substantial additional signal beyond surface features. These results position AI
    
[^208]: ArcANE：角色扮演语言代理能否在恰当的时机保持角色设定？

    ArcANE: Do Role-Playing Language Agents Stay in Character at the Right Time?

    [https://arxiv.org/abs/2606.05553](https://arxiv.org/abs/2606.05553)

    本文提出ArcANE基准，通过构建刻画角色价值观、动机和关系随故事演变的“情节弧”，评估角色扮演语言代理能否在叙事的不同阶段准确呈现角色的动态发展，弥补了现有基准将角色视为固定设定的不足。

    

    角色扮演语言代理（RPLA）在娱乐、陪伴、互动叙事和教育等应用中模拟特定的角色和人物设定。忠实的角色扮演不仅仅是产生合理的、符合角色的回复：随着叙事中角色的价值观和行为发生变化，RPLA 应当反映出角色在相应阶段的状态。然而，现有的基准测试大多将角色视为固定的人物设定，或仅测试角色在叙事某一时间点上所知道的内容。我们提出了 ArcANE（Arc-Aware Narrative Evaluation，情节弧感知叙事评估），这是一个用于评估 RPLA 是否能跟随角色在叙事中发展变化的基准。ArcANE 首先构建一个“情节弧”，刻画角色的价值观、动机或关系如何随故事发展而演变。随后，该基准评估 RPLA 的回复与情节弧相应阶段的契合程度，涵盖三种不同的场景类型：来自……

    arXiv:2606.05553v2 Announce Type: replace-cross  Abstract: Role-playing language agents (RPLAs) simulate specific characters and personas across applications such as entertainment, companionship, interactive storytelling, and education. Faithful role-play requires more than producing plausible, in-character responses: as a character's values and behavior change over a narrative, an RPLA should reflect the character's state at the relevant stage. However, existing benchmarks largely treat characters as fixed personas or test only what they know at a given point in the narrative. We introduce ArcANE (Arc-Aware Narrative Evaluation), a benchmark for evaluating whether an RPLA follows a character's development across a narrative. ArcANE first builds an Arc that maps how a character's values, motivations, or relationships change over the story. The benchmark then scores how well an RPLA's responses fit the corresponding stages of the Arc, covering three distinct scenario types: scenes from 
    
[^209]: AIP：一种用于学习与管理智能体技能的图表示

    AIP: A Graph Representation for Learning and Governing Agent Skills

    [https://arxiv.org/abs/2606.04781](https://arxiv.org/abs/2606.04781)

    该论文提出智能体指令协议（AIP），将智能体技能建模为由模式验证的YAML规范治理的有向执行图，从而同时提升智能体在重实现任务上的可靠性和技能创建与改进的效率。

    

    当前的智能体技能大多由自由格式的文本构成，智能体需要在每次会话中阅读、解释并重新推导如何行动。这带来了两个叠加的成本：一是在重实现类任务上的可靠性下降，二是技能创建和改进的困难，因为编辑文本是一个脆弱的过程，无论人类还是智能体都难以胜任，尤其是对于模型训练中代表性不足的领域特定程序性知识。智能体指令协议（AIP）通过将技能建模为有向执行图来解决这两个问题：离散步骤作为节点，由确定性脚本或自然语言描述支撑，通过显式的类型化输入/输出边相连接，并由经过模式验证的YAML规范进行治理。一个编译器元技能可将现有的人类编写的技能转换为这种形式。其好处有二。首先，将人类编写的技能编译为AIP后，Claude Sonnet的平均任务奖励从0.60提升至……

    arXiv:2606.04781v2 Announce Type: replace  Abstract: Agent Skills today consist largely of free-form prose requiring the agent to read, interpret, and re-derive how to act in every session. This imposes two compounding costs: reduced reliability on implementation-heavy tasks, and difficulty in skill creation and improvement, since editing prose is a fragile process that both humans and agents struggle with, particularly for domain-specific procedural knowledge underrepresented in model training. The Agent Instruction Protocol (AIP) addresses both by modeling a skill as a directed execution graph: discrete steps as nodes backed by deterministic scripts or natural-language descriptions, connected by explicit typed input/output edges, and governed by a schema-validated YAML specification. A compiler meta-skill translates existing human-written skills into this form. The benefits are twofold. First, compiling human-written skills to AIP raised Claude Sonnet's mean task reward from 0.60 to 
    
[^210]: 大型人工智能模型在牙科医疗中的应用：从通用系统到领域专用基础模型

    Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models

    [https://arxiv.org/abs/2606.02914](https://arxiv.org/abs/2606.02914)

    本文首次提出按架构范式和牙科专业化程度划分的二维分类框架，系统综述了97项研究，统一考察了语言生成模型、视觉基础模型和牙科专用基础模型三类大规模AI模型在牙科医疗中的关系与共同局限。

    

    背景：口腔疾病影响全球近35亿人，然而大规模AI模型在牙科领域的比较临床潜力仍知之甚少。目前出现了三种不同的模型类别：语言生成模型、判别式视觉基础模型和牙科专用基础模型，尚无统一的综述来考察它们之间的关系及其共同局限性。方法：遵循PRISMA-ScR指南，我们系统检索了四个数据库（PubMed、Google Scholar、Scopus、arXiv），并由两名评审员独立筛选。应用纳入/排除标准后，共纳入97项研究（2020-2026年）。我们提出了一个二维分类框架，按架构范式和牙科专业化程度对模型进行分类组织。结果：语言生成模型在文本类任务中表现出色（临床推理、执业考试、患者沟通），但表现不一致……

    arXiv:2606.02914v3 Announce Type: replace  Abstract: Background: Oral diseases affect nearly 3.5 billion people worldwide, yet the comparative clinical potential of large-scale AI models in dentistry remains poorly understood. Three distinct model categories have emerged: language-generative models, discriminative vision foundation models, and dental-specific foundation models, with no unified review examining their relationships and collective limitations.   Methods: Following PRISMA-ScR guidelines, we systematically searched four databases (PubMed, Google Scholar, Scopus, arXiv), screened independently by two reviewers. After applying inclusion/exclusion criteria, 97 studies (2020-2026) were included. We propose a two-dimensional classification framework organizing models by architectural paradigm and dental specialization degree.   Results: Language-generative models excel at text-based tasks (clinical reasoning, licensing exams, patient communication) but show inconsistent performa
    
[^211]: 修复 FOLIO 和 MALLS：经验证的标注与借助大语言模型聚焦人工重新标注的框架

    Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling

    [https://arxiv.org/abs/2606.02837](https://arxiv.org/abs/2606.02837)

    本研究系统性审查发现 FOLIO 和 MALLS 基准中约 42% 的一阶逻辑标注存在错误，并发布修正后的真值标注以及一个 LLM 辅助框架来聚焦人工重新标注工作。

    

    从自然语言到一阶逻辑（NL-to-FOL）的准确转换是神经符号 AI 系统和自然语言推理（NLI）的基础，因此 NL-to-FOL 基准数据集的质量至关重要——然而这些数据集从未经过严格审查。我们的第一个贡献是对 FOLIO 验证集和 MALLS 测试实例的一个子集进行了系统性的人工检查，发现分别约有 42.5% 和 42% 的条目包含错误的一阶逻辑形式化（即真值标签），此外还存在歧义自然语言句子（分别为 17.8% 和 51%）以及 FOLIO 中错误的 NLI 标签（8.4%）。我们的第二个贡献是为这些数据集开发并发布了修正后的真值标注，并证明标注错误会扭曲模型在参考基准任务上的评估：在测试三个最先进的大语言模型（Gemma 4 31B-it、Qwen3-30B-A3B 和 GPT-4o-mini）时……

    arXiv:2606.02837v2 Announce Type: replace-cross  Abstract: Accurate translation from Natural Language to First-Order Logic (NL-to-FOL) underpins neurosymbolic AI systems and Natural Language Inference (NLI), making the quality of NL-to-FOL benchmarks essential---yet these datasets have never been rigorously audited. Our first contribution is to present a systematic human inspection of the validation split of \textsf{FOLIO} and a subset of \textsf{MALLS} test instances, finding that approximately 42.5\% and 42\% of entries, respectively, contain incorrect FOL formalizations (i.e., ground truth labels), with additional rates of ambiguous NL sentences (17.8\% and 51\%) and incorrect NLI labels in \textsf{FOLIO} (8.4\%). Our second contribution is to develop and release corrected ground truths for such datasets, showing that annotation errors distort model evaluation on a reference benchmark task: testing three state-of-the-art LLMs (Gemma~4 31B-it, Qwen3-30B-A3B, and GPT-4o-mini) with the
    
[^212]: EntangleCodec：一种通过语义-声学纠缠实现的统一离散音频分词器

    EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglement

    [https://arxiv.org/abs/2606.02739](https://arxiv.org/abs/2606.02739)

    EntangleCodec提出了一种统一的离散音频分词器，通过在量化前学习与丰富文本描述对齐的语义-声学纠缠表示，并结合流匹配扩散解码器，同时支持音频理解与高质量重建生成。

    

    音频分词器作为连续音频与音频语言模型（ALMs）之间的离散接口，但现有的分词器往往难以同时支持理解与生成任务。以重建为导向的编解码器能够保留声学保真度，但缺乏丰富的语义信息；而语义感知的分词器通常依赖相互分离的语义流和声学流，从而引入冗余或错位问题。我们提出了EntangleCodec，一种统一的离散音频分词器，它在量化之前学习与文本描述（caption）对齐的语义-声学表示。通过将音频与丰富的文本描述（而非ASR转录文本）对齐，EntangleCodec能够在紧凑的token流中捕捉语言内容、说话人身份、情感、韵律和声学场景。流匹配扩散解码器进一步实现了跨语音、音乐和通用音频的高质量重建。EntangleCodec实现了与……（具有竞争力的）重建质量（原文在此处截断）

    arXiv:2606.02739v2 Announce Type: replace-cross  Abstract: Audio tokenizers serve as the discrete interface between continuous audio and Audio Language Models (ALMs), but existing tokenizers often struggle to support both understanding and generation. Reconstruction-oriented codecs preserve acoustic fidelity but lack rich semantics, while semantic-aware tokenizers typically rely on separate semantic and acoustic streams, introducing redundancy or misalignment.   We propose \textbf{EntangleCodec}, a unified discrete audio tokenizer that learns caption-aligned semantic-acoustic representations before quantization. By aligning audio with rich captions rather than ASR transcripts, EntangleCodec captures linguistic content, speaker identity, emotion, prosody, and acoustic scenes within a compact token stream. A flow-matching diffusion decoder further enables high-quality reconstruction across speech, music, and general audio.   EntangleCodec achieves reconstruction quality competitive with 
    
[^213]: CoMAP：面向大语言模型智能体的世界模型与智能体策略共同演化框架

    CoMAP: Co-Evolving World Models and Agent Policies for LLM Agents

    [https://arxiv.org/abs/2606.02372](https://arxiv.org/abs/2606.02372)

    本文提出COMAP框架，通过闭环交互使文本世界模型与LLM智能体策略共同演化——世界模型为候选动作预测未来状态反馈以支持智能体的前瞻性反思，智能体产生的同策略轨迹又通过自蒸馏反哺更新世界模型，从而摆脱了对外部奖励或验证器的依赖。

    

    arXiv:2606.02372v2 公告类型：替换 摘要：为语言智能体配备世界模型，使其能够在执行之前预测环境动态并评估候选动作。然而，现有的文本世界模型通常在训练完成后即被固定，无法适应不断演化的智能体所产生的同策略状态-动作分布。与此同时，智能体改进方法往往依赖外部奖励或验证器，这限制了它们在现实交互环境中的适用性。本文提出了COMAP，一种通过闭环交互使文本世界模型与智能体策略共同演化的新框架。在每个决策步骤中，世界模型为候选动作预测未来状态反馈，智能体则通过估计该反馈的可靠性并据此优化自身动作来进行具有前瞻性的反思。随后，所产生的同策略轨迹被用于通过自蒸馏来更新世界模型，使其能够……（摘要原文在此处截断）

    arXiv:2606.02372v2 Announce Type: replace  Abstract: Equipping language agents with world models enables them to anticipate environment dynamics and evaluate candidate actions before execution. However, existing textual world models are typically fixed after training, preventing them from adapting to the on-policy state-action distributions induced by an evolving agent. Meanwhile, agent-improvement methods often rely on external rewards or verifiers, limiting their applicability in realistic interactive environments. In this paper, we propose COMAP, a novel framework that co-evolves textual world models and agent policies through closed-loop interaction. At each decision step, the world model predicts future state feedback for candidate actions, and the agent performs future-aware reflection by estimating the reliability of this feedback and refining its action accordingly. The resulting on-policy trajectories are then used to update the world model via self-distillation, allowing it t
    
[^214]: 论点坍塌：大语言模型使长篇公共辩论趋于扁平化

    Argument Collapse: LLMs Flatten Long-Form Public Debate

    [https://arxiv.org/abs/2606.01736](https://arxiv.org/abs/2606.01736)

    该论文首次提出并系统量化了“论点坍塌”现象：大语言模型生成的议论文会高度收敛到极少数相同论点（人类论点65.3%独特而模型仅3.4%），即使显式要求多样性也只能恢复约一半的人类论点，揭示了LLM可能使公共辩论同质化、扁平化的风险。

    

    随着大语言模型越来越多地被用于起草面向公众的论辩文章，它们可能通过反复引入相同的、经过润色且貌似合理的论点而使公共辩论趋于扁平化。我们研究了“论点坍塌”现象，即不同大语言模型生成的文章倾向于收敛到较小的一组主要论点、子论点和段落级结构上。我们比较了来自195场《纽约时报》(NYT)辩论的1,039条人类回复、来自61个篇幅更长的《波士顿评论》(BR)论坛的448条人类回复，以及23,381篇由大语言模型生成的文章。在NYT语料库中，65.3%的人类主要论点在同一场辩论中是独特的，而大语言模型生成的主要论点中仅有3.4%是独特的。要求大语言模型生成多样化答案虽能增加变化，但一个典型模型只能恢复约一半的人类独特主要论点，且许多新增的变化落在已观察到的人类论点空间之外。坍塌现象同样出现在子论点层面：在具有相同主要论点的文章中，41.0%的

    arXiv:2606.01736v4 Announce Type: replace-cross  Abstract: As LLMs are increasingly used to draft publicfacing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,381 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% o
    
[^215]: HARP：用于极端大语言模型量化的Hadamard预条件自适应旋转处理器

    HARP: Hadamard-Preconditioned Adaptive Rotation Processor for Extreme LLM Quantization

    [https://arxiv.org/abs/2605.29843](https://arxiv.org/abs/2605.29843)

    提出HARP，一种可学习的Hadamard预条件自适应旋转处理器，通过稀疏蝶形块正交结构替代固定Hadamard变换，在保持与全精度模型精确等价的同时，自适应地适应层、校准分布和量化器，从而提升极端低比特LLM量化的鲁棒性。

    

    训练后量化（PTQ）对于在内存和带宽受限条件下部署大语言模型（LLM）至关重要。然而，极端低比特量化对激活异常值和各向异性权重曲率仍然高度敏感。现有的基于非相干性的PTQ方法通过固定的随机化Hadamard变换（RHT）来缓解这一问题，虽然提高了量化鲁棒性，但无法使旋转基适应特定层、校准分布或量化器。我们提出了HARP（Hadamard预条件自适应旋转处理器），这是一种可学习的结构化双侧正交处理器，可替代固定的Hadamard混合，同时保持与全精度模型的精确等价性。HARP将每次旋转表示为稀疏的类蝶形块正交阶段的乘积，通过混合基调度支持非2的幂次维度，并且初始化时与RHT处理器等价（最多差一个固定置换）。仅在校准数据上进行拟合，HARP能够……（原文在此处截断）

    arXiv:2605.29843v2 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) is essential for deploying LLMs under memory and bandwidth constraints. However, extreme low-bit quantization remains highly sensitive to activation outliers and anisotropic weight curvature. Existing incoherence-based PTQ methods mitigate this issue with fixed randomized Hadamard transforms (RHTs), which improve quantization robustness but cannot adapt the rotated basis to the layer, calibration distribution, or quantizer. We introduce HARP (Hadamard-preconditioned Adaptive Rotation Processor), a learnable structured two-sided orthogonal processor that replaces fixed Hadamard mixing while preserving exact full-precision equivalence. HARP represents each rotation as a product of sparse butterfly-like block-orthogonal stages, supports non-power-of-two dimensions through Mixed-Radix schedules, and initializes to the RHT processor up to a fixed permutation. Fitted only on calibration data, HARP ada
    
[^216]: 技能条件门控自蒸馏用于大语言模型推理

    Skill-Conditioned Gated Self-Distillation for LLM Reasoning

    [https://arxiv.org/abs/2605.28791](https://arxiv.org/abs/2605.28791)

    本文提出SGSD，通过技能库和门控机制将自蒸馏从无条件模仿转为教师假设验证，以应对不可靠技能并提升大语言模型推理能力。

    

    摘要：arXiv:2605.28791v2 公告类型：跨版本替换。摘要：在策略自蒸馏（SD）通过利用教师端的特权信息（PI）将稀疏的验证器结果转化为密集的令牌级监督，从而提升大语言模型的推理能力。现有方法通常假设PI是可信的，例如参考答案或成功轨迹。我们提出疑问：PI是否可以从经验衍生的技能库中获得，其中检索到的技能虽然紧凑且可复用，但也可能不相关或具有误导性。我们提出了技能条件门控自蒸馏（SGSD），该方法将基于技能的SD视为教师假设验证而非无条件模仿。SGSD检索技能-错误对，构建多教师池，并让所有技能条件的教师对相同的普通提示学生轨迹进行评分。验证器验证每位教师的极性：支持成功或抑制失败提供正向监督，而相反立场则被反转。一个稳健的门控目标随后将...

    arXiv:2605.28791v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (SD) improves LLM reasoning by using teacher-side privileged information (PI) to turn sparse verifier outcomes into dense token-level supervision. Existing methods usually assume trusted PI, such as reference answers or successful traces. We ask whether PI can instead come from an experience-derived skill bank, where retrieved skills are compact and reusable but may also be irrelevant or misleading. We propose Skill-Conditioned Gated Self-Distillation (SGSD), which formulates skill-based SD as teacher hypothesis validation rather than unconditional imitation. SGSD retrieves skill-mistake pairs, constructs a multi-teacher pool, and lets all skill-conditioned teachers score the same plain-prompt student rollout. The verifier validates each teacher's polarity: supporting a success or suppressing a failure gives positive supervision, while the opposite stance is reversed. A robust gated objective then di
    
[^217]: 解码前的拒绝行为：检测并利用大语言模型中间激活中的拒绝信号

    Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations

    [https://arxiv.org/abs/2605.28553](https://arxiv.org/abs/2605.28553)

    大语言模型的拒绝行为在中间激活层就已线性可解码，利用这一信号构建的探针引导攻击方法Mechanistic AutoDAN能在保持相当攻击成功率的同时，将越狱攻击的搜索时间最多减少72%。

    

    本文研究了能否在解码之前，利用在每个Transformer块残差流激活上训练的线性探针，从大语言模型的中间激活中预测拒绝行为。我们发现拒绝行为在远早于最后一层的层级即可被线性解码，这表明与安全相关的行为在输出生成之前就已体现在中间激活中。为了测试这一信号是否具有可操作性，我们提出了Mechanistic AutoDAN，这是AutoDAN的一种探针引导变体，它在遗传提示搜索循环中用部分前向传播和基于探针的评分替代了全模型适应度评估。在所评估的各个模型上，我们的方法实现了与原始AutoDAN相当的攻击成功率，同时将每次迭代的搜索时间最多减少72%，并且在若干配置下，探针引导的提示词在跨模型迁移方面达到甚至超过了AutoDAN的水平。我们进一步发现探针引导在……

    arXiv:2605.28553v2 Announce Type: replace  Abstract: In this paper, we investigate whether refusal behavior can be predicted from LLM intermediate activations before decoding using linear probes trained on residual stream activations at each transformer block. We find that refusal is linearly decodable well before the final layer, indicating that safety-relevant behavior is represented in intermediate activations before output generation. To test whether this signal is actionable, we introduce Mechanistic AutoDAN, a probe-guided variant of AutoDAN that replaces full-model fitness evaluation with partial forward passes and probe-based scoring inside a genetic prompt search loop. Across the evaluated models, our method achieves attack success rates competitive with vanilla AutoDAN while reducing per-iteration search time by up to 72%, and probe-guided prompts match or exceed AutoDAN's cross-model transfer in several configurations. We further find that the usefulness of probe guidance in
    
[^218]: MIRA：一个用于医疗信息响应审计的双语基准测试

    MIRA: A Bilingual Benchmark for Medical Information Response Audit

    [https://arxiv.org/abs/2605.28025](https://arxiv.org/abs/2605.28025)

    该论文提出了首个双语医疗基准MIRA，揭示了大语言模型在应对低健康素养用户提问时会系统性遗漏关键医疗信息、减少后续行动建议的“差异化信息稀释”（DID）这一安全隐患。

    

    现有的大语言模型安全评估忽视了这样一个问题：当用户以不同措辞提出同一问题时，模型的回答是否保留了可比较的医疗信息。为解决这一问题，我们提出了医疗信息响应审计基准，这是一个双语、受控的基准测试，用于评估大语言模型在用户语言、语域和健康素养信号变化下能否提供可比较的医疗信息。MIRA包含由60个经过医学审阅的低风险健康问题构建的4,320个提示词。在五个主流大语言模型上的实验表明，模型能够回答所有医疗问题，但针对低健康素养信号的回答一致地遗漏了更多关键信息，提供了更少的后续具体步骤，对独立判断的支持也更少。我们将这一现象命名为差异化信息稀释。与300个真实世界健康查询的对比为排序效度提供了初步证据。一种知识引导的缓解方法…

    arXiv:2605.28025v2 Announce Type: replace  Abstract: Existing safety evaluations for large language models overlook whether responses preserve comparable medical information across different user phrasings of the same question. To address this, we introduce the Medical Information Response Audit (MIRA), a bilingual, controlled benchmark that assesses whether LLMs provide comparable medical information across user-side language, register, and health literacy signals. MIRA contains 4,320 prompts built from 60 medically reviewed, low-risk health questions. Across five mainstream LLMs, models answered all medical questions, but responses to low health-literacy signals consistently omitted more key information, provided fewer concrete next steps, and offered less support for independent judgment. We term this pattern Differential Information Dilution (DID). A comparison with 300 real-world health queries provides preliminary evidence of rank-order validity. A knowledge-guided mitigation pro
    
[^219]: EmoDistill：面向对抗性谈判中语言模型智能体的离线情感技能蒸馏

    EmoDistill: Offline Emotion Skill Distillation for Language Model Agents in Adversarial Negotiation

    [https://arxiv.org/abs/2605.26785](https://arxiv.org/abs/2605.26785)

    提出EmoDistill离线蒸馏框架，通过IQL情感选择器与LoRA微调的表达策略相结合，将大模型间对抗谈判中的情感技能迁移到小型语言模型智能体，使其能够抵御情感操纵并维护用户谈判目标。

    

    经过训练后优化的大语言模型通常被调整为产生有帮助、礼貌且迁就他人的回应。然而，在对抗性谈判中，这种行为可能成为一种脆弱性：带有情感色彩的语言可能会以与其用户目标相冲突的方式影响智能体的谈判决策。因此，我们提出了EmoDistill，这是一个离线框架，用于将大语言模型之间交互中的情感谈判技能蒸馏到更小的语言模型智能体中。在这里，情感谈判技能是一种状态条件行为，它决定在谈判状态下应调用哪种显式情感，以及如何将该情感转化为有效的谈判话语。EmoDistill分别学习这两个组件：一个隐式Q学习选择器学习在每个谈判状态下应表达哪种情感，而一个经LoRA适配的7B策略通过监督微调和Judge Policy学习情感条件下的表达方式

    arXiv:2605.26785v2 Announce Type: replace-cross  Abstract: Post-trained LLMs are often optimized to produce helpful, polite, and accommodating responses. In adversarial negotiation, however, such behavior can become a vulnerability: emotionally framed language may influence an agent's bargaining decisions in ways that conflict with its user's objectives. We therefore introduce EmoDistill, an offline framework for distilling emotional negotiation skills from LLM-LLM interactions into smaller language-model agents. Here, an emotional negotiation skill is a state-conditioned behavior that determines which explicit emotion to invoke in a bargaining state and how to realize that emotion as an effective negotiation utterance. EmoDistill learns these two components separately: an Implicit Q-Learning (IQL) selector learns which emotion to express in each bargaining state, while a LoRA-adapted 7B policy learns emotion-conditioned expression through Supervised Fine-Tuning (SFT) and Judge Policy 
    
[^220]: 使用蜜罐令牌识别AI网络爬虫

    Identifying AI Web Scrapers Using Canary Tokens

    [https://arxiv.org/abs/2605.13706](https://arxiv.org/abs/2605.13706)

    本文提出了一种基于蜜罐令牌的新技术，能够准确且自动化地识别与大型语言模型相关的网络爬虫，克服了现有识别方法不可靠、不可扩展的缺陷。

    

    从预训练到查询时增强，网络抓取的数据有助于提高大型语言模型（LLM）生成内容的质量和上下文相关性。然而，为LLM提供数据的大规模网络爬取可能影响网站稳定性，并引发法律、隐私或伦理方面的担忧。如果网站所有者出于这些或其他顾虑，希望限制其网站上与LLM相关的网络爬取行为，他们可能会求助于诸如Robots排除协议之类的爬虫访问控制机制。为了使这类机制发挥最大效用，网站所有者需要首先识别出他们想要限制的爬虫（例如，通过User-Agent字符串）。现有的识别LLM相关爬虫的机制依赖于公司的自愿披露、研究人员的一次性实验或众包报告——这些方法既不可靠也不可扩展。本文提出了一种新颖的技术，能够准确且自动地推断出与LLM相关的爬虫。

    arXiv:2605.13706v2 Announce Type: replace-cross  Abstract: From pre-training to query-time augmentation, web-scraped data helps to improve the quality and contextual relevancy of content generated by large language models (LLMs). However, large-scale web scraping to feed LLMs can affect site stability and raise legal, privacy, or ethics concerns. If website owners wish to limit LLM-related web scraping on their site, due to these or other concerns, they may turn to scraper access control mechanisms like the Robots Exclusion Protocol. To be most effective, such mechanisms require site owners to first identify the scrapers that they wish to restrict (e.g., via User-Agent strings). Existing mechanisms to identify LLM-related scrapers rely on voluntary disclosure by companies, one-off experiments by researchers, or crowd-sourced reports -- methods that are neither reliable nor scalable. This paper proposes a novel technique for accurately and automatically inferring LLM-related scrapers. W
    
[^221]: 迈向可负担能源：面向电力公司需求响应项目的Gymnasium环境

    Towards Affordable Energy: A Gymnasium Environment for Electric Utility Demand-Response Programs

    [https://arxiv.org/abs/2605.12462](https://arxiv.org/abs/2605.12462)

    本文提出了DR-Gym，一个开源的、与Gymnasium兼容的在线仿真环境，用于从电力公司视角训练和评估需求响应策略，解决了离线历史数据无法捕捉价格信号与用户行为之间动态交互反馈循环的问题。

    

    极端天气和剧烈波动的批发电力市场使住宅消费者面临灾难性的财务风险，然而配电层面的需求响应作为提升电网灵活性和能源可负担性的工具仍未得到充分利用。虽然需求响应项目可以通过在电价高峰期向消费者发放财务补贴来保护他们，但优化这一序列决策过程对强化学习而言是一个独特的挑战，尽管公开可用的离线历史智能电表数据和批发电价数据十分丰富。离线历史数据无法捕捉电力公司价格信号与客户对需求响应项目的接受和适应之间动态的交互反馈循环。为解决这一问题，我们提出了DR-Gym，一个开源的、与Gymnasium兼容的在线环境，旨在从电力公司的视角训练和评估需求响应策略。

    arXiv:2605.12462v2 Announce Type: replace  Abstract: Extreme weather and volatile wholesale electricity markets expose residential consumers to catastrophic financial risks, yet demand response at the distribution level remains an underutilized tool for grid flexibility and energy affordability. While a demand-response program can shield consumers by issuing financial credits during high-price periods, optimizing this sequential decision-making process presents a unique challenge for reinforcement learning despite the plentiful offline historical smart meter and wholesale pricing data available publicly. Offline historical data fails to capture the dynamic, interactive feedback loop between an electric utility's pricing signals and customer acceptance and adaptation to a demand-response program. To address this, we introduce DR-Gym, an open-source, online Gymnasium-compatible environment designed to train and evaluate demand-response from the electric utility's perspective. Unlike exis
    
[^222]: 超越可复现性：迈向面向安全的研究工件评估

    Beyond Reproducibility: Towards Security-Aware Evaluation of Research Artifacts

    [https://arxiv.org/abs/2605.06508](https://arxiv.org/abs/2605.06508)

    该论文对2023至2025年四大顶级安全会议发表的1,388个研究工件进行静态分析，提出上下文感知的安全评估分类体系，发现其中44.80%的安全发现具有真实安全相关性，并推出了SAFE工具以支持可扩展的工件安全评估。

    

    研究工件被广泛共享以支持可复现性，工件评估在许多顶级会议上已变得普遍。然而，工件评估主要检查工件是否如所声称的那样工作并能被复现，并不旨在发现或缓解潜在的安全风险。由于这些工件被公开发布并被复用，它们可能无意中为滥用创造机会，并引发对安全、负责任共享的担忧。我们研究了2023年至2025年间在四大顶级安全会议上发表的1,388个研究工件，进行静态分析，获得了132,431个候选安全发现。我们提出了一个用于上下文感知安全评估的分类体系，并对这些发现进行检查，以过滤误报并识别出代表可能的、依赖上下文的安全风险的发现。我们发现44.80%的经审查发现具有安全相关性。为支持可扩展的分析，我们提出了SAFE（原文在此处截断）。

    arXiv:2605.06508v2 Announce Type: replace-cross  Abstract: Research artifacts are widely shared to support reproducibility, and artifact evaluation (AE) has become common at many leading conferences. However, AE mainly checks whether artifacts work as claimed and can be reproduced. It does not aim at spotting or mitigitating potential security risks. Since these artifacts are publicly released and reused, they may unintentionally create opportunities for misuse and raise concerns about safe and responsible sharing. We study 1,388 research artifacts published between 2023 and 2025 at the top-4 security conferences, perform static analysis, and obtain 132,431 candidate security findings. We propose a taxonomy for context-aware security assessment and examine the findings to filter false positives and identify findings that represent plausible context-dependent security risks. We find that 44.80% of the reviewed findings are security-relevant. To support scalable analysis, we present SAFE
    
[^223]: 多模态大语言模型内部视觉表征的因果探查

    Causal Probing for Internal Visual Representations in Multimodal Large Language Models

    [https://arxiv.org/abs/2605.05593](https://arxiv.org/abs/2605.05593)

    该研究提出基于激活引导的因果框架，揭示了多模态大语言模型中实体知识局部化编码而抽象概念全局分布的分化现象，并证明模型深度的增加是编码复杂抽象概念这一缩放定律背后的机制性驱动因素。

    

    尽管多模态大语言模型（MLLMs）在多样化任务中取得了显著成功，但其编码和定位不同视觉概念的内部机制仍然知之甚少。为了揭示这些机制，我们提出了一个基于激活引导的因果框架，以主动探查和操纵内部视觉表征。通过对四个视觉概念类别的系统性干预，我们的结果揭示了概念编码的分化现象：实体知识具有明显的局部化特征，而抽象概念则全局分布于整个网络。关键的是，这种分化揭示了缩放定律的一个机制性驱动因素：增加模型深度对于编码分布式和复杂的抽象概念是必不可少的，而实体则始终保持较高的局部化程度。此外，反向引导发现，阻断显式输出会触发潜在激活的激增……

    arXiv:2605.05593v2 Announce Type: replace  Abstract: Despite the remarkable success of Multimodal Large Language Models (MLLMs) across diverse tasks, the internal mechanisms governing how they encode and ground distinct visual concepts remain poorly understood. To unravel these mechanisms, we propose a causal framework based on activation steering to actively probe and manipulate internal visual representations. Through systematic intervention across four visual concept categories, our results reveal a divergence in concept encoding: entity knowledge is distinctively localized, whereas abstract concepts are globally distributed across the network. Critically, this divergence uncovers a mechanistic driver of scaling laws: increasing model depth is indispensable for encoding distributed and complex abstract concepts, whereas entities maintain a consistently high degree of localization. Furthermore, reverse steering uncovers that blocking explicit output triggers a surge in latent activat
    
[^224]: 当思维链失败时，解决方案隐藏在隐藏状态之中

    When Chain-of-Thought Fails, the Solution Hides in the Hidden States

    [https://arxiv.org/abs/2604.23351](https://arxiv.org/abs/2604.23351)

    研究发现，即使思维链推理轨迹本身是错误的，其隐藏状态（尤其集中于中后层和轨迹早期）仍编码了足以恢复正确答案的任务相关信息，通过激活修补将这些隐藏状态注入直接回答过程可显著提升答题准确率。

    

    arXiv:2604.23351v2 公告类型：replace-cross。摘要：中间推理在计算上究竟是有用的，还是仅仅起解释作用，取决于思维链（CoT）标记中是否包含与任务相关的信息。我们利用激活修补（activation patching）对 GSM8K 上的思维链进行了机制层面的因果分析：将同一问题在思维链生成中得到的标记级隐藏状态转移到直接回答的运行中，然后测量其对最终答案准确率的影响。在各个模型上，修补后生成答案的准确率显著高于直接回答提示和原始思维链轨迹，这表明即使原始推理轨迹是错误的，单个思维链标记也能编码足以恢复正确答案的信息。这种与任务相关的信息在正确的思维链运行中比在错误的运行中更为普遍，并且在各标记之间分布不均：它集中于中间至较深的层，并在推理轨迹的较早位置出现。此外，修补语言标记，例如……（原文摘要在此处截断）

    arXiv:2604.23351v2 Announce Type: replace-cross  Abstract: Whether intermediate reasoning is computationally useful or merely explanatory depends on whether chain-of-thought (CoT) tokens contain task-relevant information. We present a mechanistic causal analysis of CoT on GSM8K using activation patching: transferring token-level hidden states from a CoT generation to a direct-answer run for the same question, then measuring the effect on final-answer accuracy. Across models, generating after patching yields substantially higher accuracy than both direct-answer prompting and the original CoT trace, revealing that individual CoT tokens can encode sufficient information to recover the correct answer, even when the original trace is incorrect. This task-relevant information is more prevalent in correct than incorrect CoT runs and is unevenly distributed across tokens, concentrating in mid-to-late layers and appearing earlier in the reasoning trace. Moreover, patching language tokens such a
    
[^225]: CASCADE：面向基于MCP系统的分层本地防御的组件消融与语料库审计

    CASCADE: A Component Ablation and Corpus Audit of a Layered Local Defense for MCP-Based Systems

    [https://arxiv.org/abs/2604.17125](https://arxiv.org/abs/2604.17125)

    本文通过对CASCADE分层本地防御进行组件消融并对其评估语料库进行全面审计，揭示了判定约定与样本来源对MCP防御评估结果的决定性影响——仅聚合方式的选择即可使误报率在1.51%与11.70%之间产生巨大差异。

    

    模型上下文协议将大语言模型应用的提示注入攻击面扩展到了工具描述、参数模式和工具输出。针对MCP的防御方案正快速涌现，但其报告的数字之间并不可比：每种防御都是在作者自建的语料库上、按照很少被明确说明的判定约定进行评估的。本文以完全本地化的分层防御CASCADE为案例，考察这些选择究竟能在多大程度上决定结果：在固定修订版本与固定协议下，对三个配置在冻结的5,000样本语料库上进行评估，并对该语料库进行了全面审计。由此得到四项结果。第一，聚合约定主导了核心指标：若将转交人工审查的样本计为阳性，会得到11.70%的误报率；而实际上在无人工干预的情况下仅有1.51%的良性流量会被拒绝，同时这一做法还掩盖了68.5%的流量需转交审查者这一事实。第二，检测结果并非与样本来源无关：（摘要在此处截断）

    arXiv:2604.17125v2 Announce Type: replace-cross  Abstract: The Model Context Protocol (MCP) widens the prompt injection attack surface of large language model applications to tool descriptions, parameter schemas, and tool outputs. Defenses for it are appearing quickly, but their reported figures are not comparable: each is evaluated on a corpus of its authors' construction, under a decision convention that is rarely stated. This paper asks how much those choices decide, taking CASCADE, a fully local layered defense, as the case: three configurations on a frozen 5,000-sample corpus under a pinned revision and a fixed protocol, with that corpus audited in full. Four results follow. First, the aggregation convention dominates the headline metric: counting review referrals as positives reports an 11.70% false-positive rate where 1.51% of benign traffic would be denied without a human, and conceals that 68.5% of all traffic reaches a reviewer. Second, detection is not provenance-invariant: 
    
[^226]: 匿名化而非消除：保持效用的语音匿名化

    Anonymization, Not Elimination: Utility-Preserved Speech Anonymization

    [https://arxiv.org/abs/2604.17000](https://arxiv.org/abs/2604.17000)

    提出了一种两阶段语音匿名化框架，通过生成式语音编辑模型替换个人可识别信息以保护内容隐私，并引入基于流匹配的F3-VA框架保护声音隐私，在实现匿名化的同时保持语音数据在ASR、TTS、SER等下游任务中的效用。

    

    对大规模语音数据日益增长的依赖使隐私保护成为关键问题。然而，现有的匿名化方法往往会降低数据效用，例如破坏声学连续性或减少声音多样性，这损害了语音数据在自动语音识别（ASR）、语音合成（TTS）和语音情感识别（SER）等下游任务中的价值。当前的评估实践也存在局限，因为它们主要依赖使用预训练模型对匿名化语音进行直接测试，只能提供对效用的局部视角。为解决这些问题，我们提出了一种新颖的两阶段框架，在保持可用性的同时保护语言内容和声学身份。在内容隐私方面，我们采用生成式语音编辑模型来无缝替换个人可识别信息（PII）；在声音隐私方面，我们引入了F3-VA，一种基于流匹配的匿名化框架……

    arXiv:2604.17000v1 Announce Type: cross  Abstract: The growing reliance on large-scale speech data has made privacy protection a critical concern. However, existing anonymization approaches often degrade data utility, for example by disrupting acoustic continuity or reducing vocal diversity, which compromises the value of speech data for downstream tasks such as Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Speech Emotion Recognition (SER). Current evaluation practices are also limited, as they mainly rely on direct testing of anonymized speech with pretrained models, providing only a partial view of utility. To address these issues, we propose a novel two-stage framework that protects both linguistic content and acoustic identity while maintaining usability. For content privacy, we employ a generative speech editing model to seamlessly replace personally identifiable information (PII), and for voice privacy, we introduce F3-VA, a flow-matching-based anonymization frame
    
[^227]: 将大语言模型评估作为张量补全问题：低秩结构与半参数有效性

    LLM Evaluation as Tensor Completion: Low Rank Structure and Semiparametric Efficiency

    [https://arxiv.org/abs/2604.05460](https://arxiv.org/abs/2604.05460)

    该论文将LLM评估建模为通过成对比较观测低秩潜在分数张量的半参数推断问题，推导了有效影响函数与半参数有效性界，并构造了具有渐近正态性的单步去偏估计量，为LLM排行榜提供了严谨的不确定性量化方法。

    

    大语言模型（LLM）评估平台越来越依赖成对的人类判断数据。这些数据噪声大、稀疏且采样不均匀，然而排行榜的报告却缺乏充分的不确定性量化。我们将该问题视为一种半参数推断：在Bradley-Terry-Luce类型模型下，通过成对比较观测低秩潜在分数张量。这使LLM评估进入了一种新的张量补全场景，具有结构化观测、非均匀采样和成对对比的特点。我们的目标是一个平滑泛函 $\psi(T^\star)$，包括线性估计目标（如能力差距）和非线性估计目标（如胜率）。我们推导了低秩切空间上的信息算子、有效影响函数以及半参数有效性界，进而构造了具有渐近正态性的单步去偏估计量。一个核心挑战在于信息算子是各向异性的，并且……（摘要在此处截断）

    arXiv:2604.05460v2 Announce Type: replace-cross  Abstract: Large language model (LLM) evaluation platforms increasingly rely on pairwise human judgments. These data are noisy, sparse, and non-uniform, yet leaderboards are reported with limited uncertainty quantification. We study this as semiparametric inference for a low-rank latent score tensor observed through pairwise comparisons under Bradley-Terry-Luce-type models. This places LLM evaluation in a new tensor completion setting with structured observations, non-uniform sampling, and pairwise contrasts. Our target is a smooth functional $\psi(T^\star)$, including linear estimands such as ability gaps and nonlinear ones such as win probabilities. We derive the information operator on the low-rank tangent space, the efficient influence function, and the semiparametric efficiency bound, then construct a one-step debiased estimator with asymptotic normality. A central challenge is that the information operator is anisotropic and does no
    
[^228]: 一个模型翻译所有语言？通往末日火山的多语言模型合并之旅

    One Model to Translate Them All? A Journey to Mount Doom for Multilingual Model Merging

    [https://arxiv.org/abs/2604.02881](https://arxiv.org/abs/2604.02881)

    该论文系统研究了多语言机器翻译中的权重空间模型合并，揭示了显著的方向性不对称现象——当模型共享目标语言时合并相对有效但无法保留各语言模型的峰值性能，而当目标语言不同时性能则急剧下降。

    

    权重空间模型合并可以在不访问原始训练数据的情况下，将独立微调的模型检查点组合起来。虽然合并在多任务设置中已展现出潜力，但其在多语言生成系统中的行为仍未得到充分探索。我们通过在大规模双语语料库上对语言模型进行完全微调，并在共享源语言、共享目标语言和双向整合等设置下评估代表性的合并策略，系统性地研究了多语言机器翻译中的权重空间合并。我们的实验揭示了强烈的方向性不对称性：当模型共享目标语言时，合并相对更为有效，相比基础模型能够提升多语言覆盖范围，但仍无法保留特定语言检查点的峰值性能；相反，当目标语言不同时，性能会急剧下降，尤其是在共享源语言和双向设置中。为了解释这一……

    arXiv:2604.02881v2 Announce Type: replace-cross  Abstract: Weight-space model merging combines independently fine-tuned checkpoints without access to the original training data. While merging has shown promise in multitask settings, its behavior in multilingual generative systems remains underexplored. We systematically study weight-space merging for multilingual machine translation by fully fine-tuning language models on large-scale bilingual corpora and evaluating representative merging strategies across shared-source, shared-target, and bidirectional consolidation settings. Our experiments reveal a strong directional asymmetry. Merging is comparatively more effective when models share a target language, improving multilingual coverage over the base model, but it still fails to preserve the peak performance of language-specific checkpoints. In contrast, when target languages differ, performance degrades sharply, especially in shared-source and bidirectional settings. To explain this 
    
[^229]: CORAL：迈向开放式发现的自主多智能体进化

    CORAL: Towards Autonomous Multi-Agent Evolution for Open-Ended Discovery

    [https://arxiv.org/abs/2604.01658](https://arxiv.org/abs/2604.01658)

    CORAL是首个面向开放式问题的自主多智能体进化框架，通过共享持久记忆、异步多智能体执行和心跳式干预实现智能体的自主探索、反思与协作，在10个任务上以远少于以往方法的评估次数取得3至10倍更高的提升率，刷新了最先进水平。

    

    基于大语言模型（LLM）的进化是实现开放式发现的一种有前景的方法，这类任务的进展需要持续的搜索和知识积累。然而，现有方法仍然严重依赖固定的启发式规则和硬编码的探索规则，这限制了LLM智能体的自主性。我们提出了CORAL，这是首个面向开放式问题的自主多智能体进化框架。CORAL以长期运行的智能体取代僵化的控制机制，这些智能体通过共享持久记忆、异步多智能体执行和基于心跳的干预机制进行探索、反思与协作。它还提供了实用的保障机制，包括隔离的工作空间、评估器分离、资源管理以及智能体会话和健康管理。在多样化的数学、算法和系统优化任务上的评估表明，CORAL在10个任务上创造了新的最先进结果，以更少的评估次数实现了3至10倍更高的提升率。

    arXiv:2604.01658v3 Announce Type: replace  Abstract: Large language model (LLM)-based evolution is a promising approach for open-ended discovery, where progress requires sustained search and knowledge accumulation. Existing methods still rely heavily on fixed heuristics and hard-coded exploration rules, which limit the autonomy of LLM agents. We present CORAL, the first framework for autonomous multi-agent evolution on open-ended problems. CORAL replaces rigid control with long-running agents that explore, reflect, and collaborate through shared persistent memory, asynchronous multi-agent execution, and heartbeat-based interventions. It also provides practical safeguards, including isolated workspaces, evaluator separation, resource management, and agent session and health management. Evaluated on diverse mathematical, algorithmic, and systems optimization tasks, CORAL sets new state-of-the-art results on 10 tasks, achieving 3-10 times higher improvement rates with far fewer evaluation
    
[^230]: 外科人工智能的比较研究：数据、算力与规模化的潜力与局限

    A Comparative Study in Surgical AI: Potential and Limitations of Data, Compute, and Scaling

    [https://arxiv.org/abs/2603.27341](https://arxiv.org/abs/2603.27341)

    本文比较研究了数据、算力与规模化在外科AI中的潜力与局限，探讨现代通用AI能否以及在多大程度上辅助外科实践。

    

    近期的人工智能（AI）模型已在多项生物医学任务性能基准上达到或超越人类专家水平，但外科手术相关的基准在外科手术基准在知名医学基准套件中往往缺失。由于外科手术需要整合多种不同的任务，若性能能够得到提升，具备通用能力的AI模型作为协作工具将特别具有吸引力。一方面，扩大模型架构规模和训练数据量的经典做法很有吸引力，尤其是每年都会产生数百万小时的外科手术视频数据；另一方面，为AI训练准备外科数据需要相当高的专业知识水平，而使用这些数据进行训练也需要昂贵的计算资源。这些权衡使得现代AI能否以及在多大程度上辅助外科实践变得不确定。在本文中，我们通过……（原文摘要在此处截断）

    arXiv:2603.27341v5 Announce Type: replace  Abstract: Recent Artificial Intelligence (AI) models have matched or exceeded human experts in several benchmarks of biomedical task performance, but surgical benchmarks in particular are often missing from prominent medical benchmark suites. Since surgery requires integrating disparate tasks, generally-capable AI models could be particularly attractive as a collaborative tool if performance could be improved. On the one hand, the canonical approach of scaling architecture size and training data is attractive, especially since there are millions of hours of surgical video data generated per year. On the other hand, preparing surgical data for AI training requires significantly higher levels of professional expertise, and training on that data requires expensive computational resources. These trade-offs paint an uncertain picture of whether and to-what-extent modern AI could aid surgical practice. In this paper, we explore this question through
    
[^231]: LRConv-NeRV：用于高效神经视频压缩的低秩卷积

    LRConv-NeRV: Low Rank Convolution for Efficient Neural Video Compression

    [https://arxiv.org/abs/2603.18261](https://arxiv.org/abs/2603.18261)

    LRConv-NeRV通过用结构化低秩可分离卷积替换密集3x3卷积层并逐阶段渐进应用低秩分解，将NeRV解码器计算复杂度降低68%、模型大小降低9.3%，同时几乎不损失视频重建质量。

    

    神经视频表示将整个视频序列编码到神经网络参数中，为传统视频编解码器提供了一种替代范式。然而，NeRV的卷积解码器在计算上仍然代价高昂且内存密集，限制了其在资源受限环境中的部署。本文提出了LRConv-NeRV，这是一种高效的NeRV变体，它用结构化低秩可分离卷积替换选定的密集3x3卷积层，并在解码器架构内进行端到端训练。通过从最大的解码器阶段到较早的阶段逐步应用低秩分解，LRConv-NeRV能够在重建质量和效率之间实现可控的权衡。大量实验表明，仅将LRConv应用于最后一个解码器阶段，即可将解码器计算复杂度降低68%（从201.9 GFLOPs降至64.9 GFLOPs），模型大小降低9.3%，同时质量损失微乎其微。

    arXiv:2603.18261v2 Announce Type: replace-cross  Abstract: Neural Representations for Videos (NeRV) encode entire video sequences within neural network parameters, offering an alternative paradigm to conventional video codecs. However, the convolutional decoder of NeRV remains computationally expensive and memory intensive, limiting its deployment in resource-constrained environments. This paper proposes LRConv-NeRV, an efficient NeRV variant that replaces selected dense 3x3 convolutional layers with structured low-rank separable convolutions, trained end-to-end within the decoder architecture. By progressively applying low-rank factorization from the largest to earlier decoder stages, LRConv-NeRV enables controllable trade-offs between reconstruction quality and efficiency. Extensive experiments demonstrate that applying LRConv only to the final decoder stage reduces decoder complexity by 68%, from 201.9 to 64.9 GFLOPs, and model size by 9.3%, while incurring negligible quality loss a
    
[^232]: 信息系统中的生成式人工智能图景：二次综述与研究议程的综合分析

    The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas

    [https://arxiv.org/abs/2603.11842](https://arxiv.org/abs/2603.11842)

    本研究通过系统检索并综合分析28篇二次综述与路线图论文，梳理了生成式人工智能在信息系统领域带来的益处、挑战及未来研究议程。

    

    ChatGPT问世后的热潮迅速重塑了信息系统（IS）的研究与实践。随着组织和社会努力应对生成式人工智能（GenAI）的采纳问题，一批二次研究和研究议程应运而生，旨在综合早期证据并为未来研究指明方向。本研究通过综述二次研究和路线图类论文，综合梳理了GenAI在信息系统领域带来的益处与挑战方面的知识现状，并识别了未来的研究方向。我们在Scopus、Web of Science（WoS）和eAIS中系统检索了2023年以来的出版物。经过严格的多阶段筛选过程，最终选取了28篇论文，运用文献计量映射和主题分析方法进行分析。我们还对所有来源进行了质量评估，以衡量每个来源对研究结论贡献的可信度。GenAI具有变革性潜力，能够提升生产力、加速创新、实现服务个性化，并使专业知识获取大众化。

    arXiv:2603.11842v2 Announce Type: replace-cross  Abstract: The post-ChatGPT surge has rapidly reframed IS research and practice. As organizations and society grapple with GenAI adoption, a body of secondary studies and research agendas has emerged to synthesize early evidence and chart directions for future inquiry. This study reviews secondary and roadmap papers to synthesize the state of knowledge on GenAI's benefits and challenges in IS, and to identify future research directions. We performed a systematic search across Scopus, WoS, and eAIS for publications from 2023 onwards. Following a rigorous, multi-stage screening process, we selected a final set of 28 papers for analysis using bibliometric mapping and thematic analysis. We also conducted a quality assessment of all sources to gauge confidence in each source's contribution to the findings. GenAI offers transformative potential to drive productivity, accelerate innovation, personalize services, and democratize access to experti
    
[^233]: NeuroWeaver：一个用于探索脑电图分析流水线程序化空间的自主进化智能体

    NeuroWeaver: An Autonomous Evolutionary Agent for Exploring the Programmatic Space of EEG Analysis Pipelines

    [https://arxiv.org/abs/2602.13473](https://arxiv.org/abs/2602.13473)

    该论文提出NeuroWeaver，一个基于大语言模型驱动的自主进化智能体，将脑电图分析流水线工程重新表述为离散约束优化问题，通过融入神经生理学先验知识自动探索并生成可执行代码，从而在多样的EEG数据集和任务上实现低成本、可泛化的分析。

    

    尽管基础模型在通用领域取得了显著成功，但将其应用于脑电图（EEG）分析受到庞大数据需求和巨大参数量的制约，这带来了高昂的计算成本，并阻碍了其在资源受限的临床环境中的部署。通用型自动化机器学习框架同样不适用于该领域，因为在无界的程序化空间中进行探索无法融入关键的神经生理学先验知识，并且经常产生神经科学上不合理的解决方案。因此，我们提出了NeuroWeaver，一个统一的自主进化智能体，它通过将流水线工程重新表述为一个离散约束优化问题，并借助大语言模型（LLM）驱动的可执行代码生成来求解，从而能够在多样的脑电图数据集和任务上实现泛化。领域知识引导的子空间初始化限制了……

    arXiv:2602.13473v3 Announce Type: replace  Abstract: Although foundation models have achieved remarkable success in general domains, applying them to electroencephalography (EEG) analysis is constrained by substantial data requirements and large parameter counts, which incur prohibitive computational costs and impede deployment in resource-constrained clinical environments. General-purpose automated machine learning frameworks are likewise ill-suited to this domain, since exploration within an unbounded programmatic space fails to incorporate essential neurophysiological priors and frequently yields neuroscientifically implausible solutions. We therefore propose NeuroWeaver, a unified autonomous evolutionary agent that generalizes across diverse EEG datasets and tasks by reformulating pipeline engineering as a discrete constrained optimization problem solved through large language model (LLM)-driven generation of executable code. A Domain-Informed Subspace Initialization confines the s
    
[^234]: PeroMAS：一个钙钛矿材料发现的多智能体系统

    PeroMAS: A Multi-agent System of Perovskite Material Discovery

    [https://arxiv.org/abs/2602.13312](https://arxiv.org/abs/2602.13312)

    提出了PeroMAS——一个通过模型上下文协议（MCP）封装钙钛矿专用工具的多智能体系统，能够在多目标约束下实现钙钛矿材料发现全工作流程的端到端优化。

    

    作为第三代光伏革命的先驱，钙钛矿太阳能电池（PSCs）以其卓越的光电性能和成本潜力而闻名。PSCs的开发过程精密而复杂，涉及文献检索、数据整合、实验设计和合成等一系列闭环工作流程。然而，现有的AI钙钛矿方法主要聚焦于离散模型，包括材料设计、工艺优化和性能预测。这些模型无法在工作流程中传播物理约束，从而阻碍了端到端的优化。在本文中，我们提出了一个用于钙钛矿材料发现的多智能体系统，命名为PeroMAS。我们首先将一系列钙钛矿专用工具封装到模型上下文协议（MCPs）中。通过规划和调用这些工具，PeroMAS能够在多目标约束下设计钙钛矿材料，涵盖……

    arXiv:2602.13312v2 Announce Type: replace-cross  Abstract: As a pioneer of the third-generation photovoltaic revolution, Perovskite Solar Cells (PSCs) are renowned for their superior optoelectronic performance and cost potential. The development process of PSCs is precise and complex, involving a series of closed-loop workflows such as literature retrieval, data integration, experimental design, and synthesis. However, existing AI perovskite approaches focus predominantly on discrete models, including material design, process optimization,and property prediction. These models fail to propagate physical constraints across the workflow, hindering end-to-end optimization. In this paper, we propose a multi-agent system for perovskite material discovery, named PeroMAS. We first encapsulated a series of perovskite-specific tools into Model Context Protocols (MCPs). By planning and invoking these tools, PeroMAS can design perovskite materials under multi-objective constraints, covering the en
    
[^235]: FedPS：基于聚合统计的结构化数据联邦预处理框架

    FedPS: Federated Preprocessing for structured data via aggregated Statistics

    [https://arxiv.org/abs/2602.10870](https://arxiv.org/abs/2602.10870)

    提出了FedPS框架，利用数据草图技术在联邦环境下通过聚合统计实现结构化数据的高效预处理（包括特征缩放、编码、离散化和缺失值插补），解决了联邦学习中预处理阶段被忽视的问题。

    

    联邦学习（FL）使多个参与方能够在不共享原始数据的情况下协同训练机器学习模型。然而，在训练之前，必须对数据进行预处理，以解决缺失值、格式不一致和特征尺度异构等问题。这一预处理阶段对模型性能至关重要，但在联邦学习研究中却很大程度上被忽视。在实际的联邦学习系统中，隐私约束禁止将原始数据集中起来，而通信效率也给分布式预处理带来了进一步的挑战。我们提出了FedPS，一个基于聚合统计的联邦数据预处理框架。FedPS利用数据草图技术高效地汇总本地数据集，同时保留关键的统计信息。基于这些汇总信息，我们设计了用于特征缩放、编码、离散化和缺失值插补的联邦算法，并将预处理相关的模型扩展到……（摘要不完整）

    arXiv:2602.10870v2 Announce Type: replace-cross  Abstract: Federated Learning (FL) enables multiple parties to collaboratively train machine learning models without sharing raw data. However, before training, data must be preprocessed to address missing values, inconsistent formats, and heterogeneous feature scales. This preprocessing stage is critical for model performance but is largely overlooked in FL research. In practical FL systems, privacy constraints prohibit centralizing raw data, while communication efficiency introduces further challenges for distributed preprocessing. We introduce FedPS, a framework for federated data preprocessing based on aggregated statistics. FedPS leverages data-sketching techniques to efficiently summarize local datasets while preserving essential statistical information. Building on these summaries, we design federated algorithms for feature scaling, encoding, discretization, and missing-value imputation, and extend preprocessing-related models such
    
[^236]: 从仿真轨迹中发现高层次模式

    Discovering High Level Patterns from Simulation Traces

    [https://arxiv.org/abs/2602.10009](https://arxiv.org/abs/2602.10009)

    提出一种基于程序合成的无监督学习方法，将仿真轨迹转换为高层次结构模式的稀疏标注序列，从而提升大型语言模型对物理系统推理与验证的有效性和可扩展性。

    

    大型语言模型（LLMs）无法可靠地对特定物理系统进行推理。试图让LLMs掌握必要物理概念知识的尝试已展现出巨大前景，但可解释性和验证仍然是未解决的挑战。一种新兴的替代方案是工具化方法，即让LLMs查询物理模拟器，并将产生的仿真轨迹作为验证的上下文。然而这种方法存在可扩展性差的问题，因为仿真轨迹包含大量细粒度的数值和语义数据。我们表明，将仿真轨迹转换为“高层次”结构模式的稀疏表示，能够使LLMs进行更有效的解释。我们提出了一种无监督学习方案，通过程序合成来执行这种转换（即标注）。我们的学习过程产出一个程序库，这些程序充当模式检测器，可以将仿真轨迹转换为稀疏的、带标注的模式序列。

    arXiv:2602.10009v3 Announce Type: replace  Abstract: Large Language Models (LLMs) are unable to reliably reason about specific physical systems. Attempts to imbue LLMs with knowledge of the necessary physics concepts have shown great promise, but explainability and validation remain open challenges. An emerging alternative is tooling, where LLMs can query physical simulators and use the resulting simulation traces as context for validation. This approach suffers from poor scalability since simulation traces contain large volumes of fine-grained numerical and semantic data. We show that translating simulation traces to a sparse representation of "high-level" structural patterns leads to more effective interpretation by LLMs. We propose an unsupervised learning scheme to perform this translation, or annotation, via program synthesis. Our learning results in a library of programs that act as pattern detectors which can translate simulation traces to sparse, annotated pattern sequences. Th
    
[^237]: 审计多智能体大语言模型推理树优于多数投票和LLM-as-Judge

    Auditing Multi-Agent LLM Reasoning Trees Outperforms Majority Vote and LLM-as-Judge

    [https://arxiv.org/abs/2602.09341](https://arxiv.org/abs/2602.09341)

    提出AgentAuditor框架，通过将多智能体推理轨迹组织成推理树并在关键分歧点比较分支级证据来裁决冲突，结合反共识偏好优化训练裁决器，其性能优于多数投票和LLM-as-Judge。

    

    多智能体系统（MAS）可以大幅扩展大语言模型（LLM）的推理能力。大多数MAS框架通过简单的多数投票来聚合智能体输出，这丢弃了推理轨迹中的证据结构。在“虚构共识”（confabulation consensus）情况下——即智能体共享相关偏差并收敛到相同的错误理由时——多数投票非常脆弱。我们提出了AgentAuditor，它超越了基于频率的聚合方法，将智能体的推理轨迹组织成推理树，明确表示其推理过程中的一致点和分歧点。AgentAuditor通过在关键分歧点比较分支级证据来解决冲突，将全局裁决转化为高效的局部化验证。我们进一步提出了反共识偏好优化（Anti-Consensus Preference Optimization, ACPO），通过经证据验证的偏好监督来训练裁决器，以减少对误导性多数线索的顺从性。在四种MAS（后续实验部分内容被截断）……

    arXiv:2602.09341v2 Announce Type: replace  Abstract: Multi-agent systems (MAS) can substantially extend the reasoning capacity of large language models (LLMs). Most MAS frameworks aggregate agent outputs via simple majority voting, discarding the evidential structure of reasoning traces. Majority voting is brittle under confabulation consensus, where agents share correlated biases and converge on the same incorrect rationale. We introduce AgentAuditor, which moves beyond frequency-based aggregation by organizing agent traces into a Reasoning Tree that explicitly represents agreements and divergences in their reasoning. AgentAuditor resolves conflicts by comparing branch-level evidence at critical divergence points, turning global adjudication into efficient, localized verification. We further propose Anti-Consensus Preference Optimization (ACPO), which trains the adjudicator with evidence-verified preference supervision to reduce conformity to misleading majority cues. Across four MAS 
    
[^238]: Ex-Omni：为全模态大语言模型实现3D面部动画生成

    Ex-Omni: Enabling 3D Facial Animation Generation for Omni-modal Large Language Models

    [https://arxiv.org/abs/2602.07106](https://arxiv.org/abs/2602.07106)

    该论文提出Ex-Omni框架，通过带混合变形协同监督的语音单元生成器与非自回归混合变形解码器将语义推理与时间生成解耦，并结合令牌即查询门控融合接口和120万样本弱监督数据集InstructS2SF-1200K，首次使全模态大语言模型能够联合生成语音与3D面部动画。

    

    全模态大语言模型（OLLMs）旨在统一多模态理解与生成，然而将其扩展至联合生成语音与3D面部动画的研究仍鲜有探索。一个关键挑战在于大语言模型的离散语义推理与3D面部运动所需的密集时间动态之间的不匹配。我们提出Expressive Omni（Ex-Omni），一个为全模态大语言模型赋予伴随语音的3D面部动画能力的框架。Ex-Omni通过带有混合变形协同监督的语音单元生成器和非自回归的混合变形解码器，将语义推理与时间生成解耦，其中语音单元提供时间骨架支撑，隐藏语音表示则承载面部相关线索。我们进一步引入了令牌即查询门控融合（TQGF）接口，用于可控的语义注入，并构建了InstructS2SF-1200K，一个包含120万样本的弱监督数据集，用于伴随语音的面部动画生成。

    arXiv:2602.07106v3 Announce Type: replace-cross  Abstract: Omni-modal large language models (OLLMs) aim to unify multimodal understanding and generation, yet extending them to jointly produce speech and 3D facial animation remains largely underexplored. A key challenge is the mismatch between the discrete semantic reasoning of LLMs and the dense temporal dynamics required for 3D facial motion. We propose Expressive Omni (Ex-Omni), a framework that augments OLLMs with speech-accompanied 3D facial animation. Ex-Omni decouples semantic reasoning from temporal generation through a speech-unit generator with blendshape co-supervision and a non-autoregressive blendshape decoder, where speech units provide temporal scaffolding and hidden speech representations carry facially relevant cues. We further introduce a token-as-query gated fusion (TQGF) interface for controlled semantic injection, as well as InstructS2SF-1200K, a 1.2M-sample weakly supervised dataset for speech-accompanied facial an
    
[^239]: F-GRPO：不要让你的策略只学显而易见的而遗忘稀有的

    F-GRPO: Don't Let Your Policy Learn the Obvious and Forget the Rare

    [https://arxiv.org/abs/2602.06717](https://arxiv.org/abs/2602.06717)

    本文提出 F-GRPO，借鉴 Focal loss 设计了难度感知的缩放系数，对高成功率采样组的更新降权，从而防止 RLVR 训练中的策略因组采样遗漏稀有正确解而过度集中于常见解。

    

    具有可验证奖励的强化学习（RLVR）通常基于组采样来估计优势并稳定策略更新。在实践中，受计算资源限制，往往无法使用非常大的组，因此训练只能在有限的 rollout 集合上进行，而这些集合只能强化其中所暴露出的正确行为。在实际的组大小下，更新可能会漏掉稀有的正确轨迹，同时仍包含混合奖励，从而将概率集中到更常见的已采样解上。我们推导了这类提示局部“尾部遗漏”事件发生的概率与组大小之间的函数关系，并证明其呈现非单调行为；在类别化抽象中，我们刻画了即使总正确概率质量在增长，未被采样的正确概率质量也可能收缩的现象。受此分析启发，我们借鉴 Focal loss 提出了一种难度感知的缩放系数，用于降低对高成功率已采样组的更新权重。实验上，类别化模拟说明了……

    arXiv:2602.06717v3 Announce Type: replace-cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is commonly based on group sampling to estimate advantages and stabilize policy updates. In practice, computational limits often rule out very large groups, so training proceeds with finite rollout sets that can reinforce only the correct behavior they expose. At practical group sizes, updates can miss rare-correct trajectories while still containing mixed rewards, concentrating probability on more common sampled solutions. We derive the probability of such prompt-local tail-miss events as a function of group size, showing non-monotonic behavior, and in the categorical abstraction characterize how unsampled-correct mass can shrink even as total correct mass grows. Motivated by this analysis, we propose a difficulty-aware scaling coefficient, inspired by Focal loss, that down-weights updates on high-success sampled groups. Empirically, categorical simulation illustrates the s
    
[^240]: 联邦学习中破坏模型置信度的温度缩放攻击

    Temperature Scaling Attack Disrupting Model Confidence in Federated Learning

    [https://arxiv.org/abs/2602.06638](https://arxiv.org/abs/2602.06638)

    该论文提出了温度缩放攻击（TSA），一种新型联邦学习训练时攻击，通过学习率-温度耦合机制在保持模型准确率不变的情况下破坏置信度校准，从而威胁依赖置信度信号的任务关键型系统的风险决策逻辑。

    

    预测置信度是任务关键型系统中的基础控制信号，直接支配着诸如升级上报、拒绝预测和保守回退等风险感知逻辑。虽然以往的联邦学习攻击主要针对准确率或植入后门，但我们识别出置信度校准作为一个独特的攻击目标。我们提出了温度缩放攻击（TSA），这是一种在保持准确率的同时降低校准质量的训练时攻击。通过在本地训练中注入带有学习率-温度耦合的温度缩放机制，TSA能够在保持预测准确率和常见优化信号接近良性训练的前提下改变模型置信度。我们在非独立同分布设置下提供了收敛性分析，表明该耦合机制控制了主要更新规模，同时留下有界的温度诱导残差，从而产生带有额外残差项的标准非凸联邦学习收敛结构。

    arXiv:2602.06638v3 Announce Type: replace-cross  Abstract: Predictive confidence serves as a foundational control signal in mission-critical systems, directly governing risk-aware logic such as escalation, abstention, and conservative fallback. While prior federated learning attacks predominantly target accuracy or implant backdoors, we identify confidence calibration as a distinct attack objective. We present the Temperature Scaling Attack (TSA), a training-time attack that degrades calibration while preserving accuracy. By injecting temperature scaling with learning rate-temperature coupling during local training, TSA shifts model confidence while keeping predictive accuracy and common optimization signals close to benign training. We provide a convergence analysis under non-IID settings, showing that the coupling controls the primary update scale while leaving a bounded temperature-induced residual, yielding the standard non-convex FL convergence structure with an additional residua
    
[^241]: 并非所有偏好都值得计算梯度：理解离线推理对齐中的梯度效用

    Not All Preferences Deserve Gradients: Understanding Gradient Utility in Offline Reasoning Alignment

    [https://arxiv.org/abs/2602.01207](https://arxiv.org/abs/2602.01207)

    提出SAGE方法，通过难度分层候选池和前向传播的信号-曲率分数筛选偏好对，揭示并非所有偏好都值得梯度更新，最有效的监督来自模型可靠犯错但曲率低的稳定自信错误。

    

    离线偏好优化通过固定的“被选择-被拒绝”偏好对来对齐推理模型，然而标准方法会对每一对偏好都应用梯度更新，而不考虑其在当前策略下的训练价值。我们认为这种一刀切的处理方式是浪费的且可能有害的。从梯度效用的视角出发，我们表明一个偏好对的贡献同时取决于信息量和稳定性。偏好对的效用会随着策略的演化而发生漂移；高梯度样本可能与高曲率区域重合，从而导致噪声大且不稳定的更新；而最有效的监督来自“稳定的自信错误”——即模型可靠地犯错但曲率仍然较低的情况。这些发现启发了 SAGE（Stability-Aware Gradient Efficiency，稳定性感知梯度效率）方法，该方法维护在训练过程中动态刷新的按难度分层的候选池，并通过前向传播的信号-曲率分数在每个池内选择偏好对，只有具有高曲率效用的偏好对才会被选中。

    arXiv:2602.01207v2 Announce Type: replace  Abstract: Offline preference optimization aligns reasoning models from fixed chosen--rejected pairs, yet standard methods apply gradient updates from every pair regardless of its training value under the current policy. We argue that this uniform treatment is wasteful and potentially harmful. From the perspective of gradient utility, we show that a pair's contribution depends jointly on informativeness and stability. Pair utility drifts as the policy evolves, high-gradient samples can coincide with high-curvature regions, leading to noisy and destabilizing updates, and the most effective supervision comes from stable confident errors where the model is reliably wrong yet curvature remains low. These findings motivate SAGE (Stability-Aware Gradient Efficiency), which maintains difficulty-stratified candidate pools refreshed during training and selects pairs within each pool by a forward-pass signal-to-curvature score. Only pairs with high curre
    
[^242]: 通过乌卡谢维奇逻辑（Łukasiewicz逻辑）完全辨识深度ReLU网络

    Complete Identification of Deep ReLU Networks through {\L}ukasiewicz Logic

    [https://arxiv.org/abs/2602.00266](https://arxiv.org/abs/2602.00266)

    本文借鉴香农用布尔逻辑分析开关电路的思想，建立了基于Łukasiewicz多值逻辑的符号演算，证明两个非退化ReLU网络实现相同函数当且仅当其中一个可通过有限次应用MV逻辑公理由另一个推导得到，从而完整刻画了深度ReLU网络函数表示的非唯一性。

    

    两个深度ReLU网络可以拥有完全不同的架构和参数，却能实现相同的函数。我们对这种非唯一性给出了完整的刻画。这是通过为深度ReLU网络构建一种符号演算来实现的，使网络的等价与简化成为公式的推导，与香农通过布尔逻辑分析开关电路的方法高度类似。受香农启发——他将电路综合转化为利用布尔代数公理对布尔公式进行操作——我们将ReLU网络的辨识转化为利用多值（MV）逻辑公理对Łukasiewicz公式的推导。两个非退化的ReLU网络在单位立方体上实现相同函数，当且仅当其中一个可以通过有限次应用MV公理由另一个得到：对于整数权重和偏置使用MV公理，对于有理数权重使用可除MV公理，对于实数权重使用Riesz MV公理。

    arXiv:2602.00266v2 Announce Type: replace  Abstract: Two deep ReLU networks can have entirely different architectures and parameters, yet realize the same function. We provide a complete characterization of this nonuniqueness. This is effected by building a symbolic calculus for deep ReLU networks, equivalence and simplification of networks becoming derivation of formulae, in close parallel to Shannon's analysis of switching circuits through Boolean logic. Inspired by Shannon, who turned circuit synthesis into the manipulation of Boolean formulae by the axioms of Boolean algebra, we turn ReLU network identification into the derivation of {\L}ukasiewicz formulae by the axioms of many-valued (MV) logic. Two non-degenerate ReLU networks realize the same function on the unit cube if and only if one is obtained from the other by finitely many applications of the MV axioms for integer weights and biases, the divisible MV axioms for rational ones, and the Riesz MV axioms for real ones. The MV
    
[^243]: VoxPrivacy：用于评估语音语言模型交互隐私的基准

    VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models

    [https://arxiv.org/abs/2601.19956](https://arxiv.org/abs/2601.19956)

    本文提出了首个用于评估语音语言模型“交互隐私”的基准VoxPrivacy，填补了现有基准在说话人身份感知响应和情境性隐私敏感信息评估方面的空白。

    

    随着语音语言模型（SLM）从个人设备过渡到智能家居等共享的多用户环境，一个新挑战随之出现：模型需要能够区分不同用户，以便合理地管理信息流。如果缺乏这种能力，语音语言模型可能会将一个用户的机密日程泄露给另一个用户，我们将这种隐私失效称为“交互隐私”。因此，生成说话人感知响应的能力对于语音语言模型的安全部署至关重要。当前的语音语言模型基准测试只检验对话能力，却忽视了说话人身份；多说话人基准测试只检查“谁说了什么”，而不评估语音语言模型是否会据此调整其响应；隐私基准测试则聚焦于全局敏感数据（如银行密码），而忽略了情境性的隐私敏感信息（如用户的私人预约）。为填补这一空白，我们提出了VoxPrivacy，这是首个旨在评估语音语言模型交互隐私的基准。

    arXiv:2601.19956v2 Announce Type: replace-cross  Abstract: As Speech Language Models (SLMs) transition from personal devices to shared, multi-user environments such as smart homes, a new challenge emerges: the model is expected to distinguish between users to manage information flow appropriately. Without this capability, an SLM could reveal one user's confidential schedule to another, a privacy failure we term interactional privacy. Thus, the ability to generate speaker-aware responses becomes essential for SLM safe deployment. Current SLM benchmarks test dialogue ability but overlook speaker identity. Multi-speaker benchmarks check who said what without assessing whether SLMs adapt their responses. Privacy benchmarks focus on globally sensitive data (e.g., bank passwords) while neglecting contextual privacy-sensitive information (e.g., a user's private appointment). To address this gap, we introduce VoxPrivacy, the first benchmark designed to evaluate interactional privacy in SLMs. V
    
[^244]: 直播情节中的“似曾相识”：利用检索增强大语言模型的跨会话证据进行直播风险评估

    Deja Vu in Plots: Leveraging Cross-Session Evidence with Retrieval-Augmented LLMs for Live Streaming Risk Assessment

    [https://arxiv.org/abs/2601.16027](https://arxiv.org/abs/2601.16027)

    提出了CS-VAR框架，通过检索增强的大语言模型在训练中将跨会话行为证据的推理洞察蒸馏给轻量级模型，使其能够识别直播中跨场次重复出现的风险模式，实现高效实时的直播风险评估。

    

    直播的兴起改变了在线互动方式，实现了大规模的实时互动，但同时也使平台面临诈骗和协同恶意行为等复杂风险。检测这些风险极具挑战性，因为有害行为往往是逐渐累积的，并且会在看似不相关的不同直播场次中反复出现。为了解决这一问题，我们提出了CS-VAR（跨会话证据感知检索增强检测器）用于直播风险评估。在CS-VAR中，一个轻量级的领域特定模型执行快速的会话级风险推断，在训练过程中由一个大语言模型（LLM）进行引导——该LLM对检索到的跨会话行为证据进行推理，并将其从局部到全局的洞察知识转移给小模型。这种设计使小模型能够识别跨直播场次重复出现的模式，进行结构化风险评估，同时保持实时部署所需的高效率。

    arXiv:2601.16027v3 Announce Type: replace  Abstract: The rise of live streaming has transformed online interaction, enabling massive real-time engagement but also exposing platforms to complex risks such as scams and coordinated malicious behaviors. Detecting these risks is challenging because harmful actions often accumulate gradually and recur across seemingly unrelated streams. To address this, we propose CS-VAR (Cross-Session Evidence-Aware Retrieval-Augmented Detector) for live streaming risk assessment. In CS-VAR, a lightweight, domain-specific model performs fast session-level risk inference, guided during training by a Large Language Model (LLM) that reasons over retrieved cross-session behavioral evidence and transfers its local-to-global insights to the small model. This design enables the small model to recognize recurring patterns across streams, perform structured risk assessment, and maintain efficiency for real-time deployment. Extensive offline experiments on large-scal
    
[^245]: 关系线性性是幻觉的预测指标

    Relational Linearity is a Predictor of Hallucinations

    [https://arxiv.org/abs/2601.11429](https://arxiv.org/abs/2601.11429)

    该论文提出关系线性性可预测语言模型的幻觉：由于抽象表示方案，语言模型能轻松为线性关系中不存在的主体生成看似合理的客体从而导致幻觉，而在面对非线性关系时这种机制失效，幻觉更容易避免。

    

    幻觉是语言模型（LM）的一个核心失败模式。我们关注语言模型在回答诸如“Glenn Gould演奏什么乐器？”这类问题时产生的幻觉，但我们针对被设计为模型未知的合成实体来提出这些问题。我们发现像Gemma-7B-IT这样的语言模型经常产生幻觉，即它们难以识别所幻觉出的事实并不属于其自身知识。基于线性关系嵌入的思想，我们提出了以下假设：（i）由于用于表示它们的抽象方案，语言模型可以轻松地为线性关系中不存在的主体生成看似合理的客体，这可能导致幻觉。（ii）对于非线性关系，这种生成客体的机制不可用，因此幻觉更容易被避免。为了检验这一假设，我们创建了SynthHal，这是一个针对15种关系的合成未知实体基准测试。我们发现，在四个……

    arXiv:2601.11429v3 Announce Type: replace-cross  Abstract: Hallucination is a central failure mode of language models (LMs). We focus on hallucinations in response to questions like: "Which instrument did Glenn Gould play?", but we ask these questions for synthetic entities designed to be unknown to the model. We find that LMs like Gemma-7B-IT frequently hallucinate, i.e., they have difficulty recognizing that the hallucinated fact is not part of their knowledge. Based on the idea of linear relational embeddings, we put forward the following hypothesis. (i) Due to the abstract scheme that is used to represent them, LMs can easily produce plausible objects for non-existing subjects of linear relations, which can lead to hallucinations. (ii) For nonlinear relations, this mechanism for producing an object is not available and so a hallucination is easier to avoid. To test this hypothesis, we create SynthHal, a synthetic unknown-entity benchmark for 15 relations. We find that across four i
    
[^246]: HOMURA：通过强化学习驯服“沙漏”，实现时间受限的大语言模型翻译

    HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning

    [https://arxiv.org/abs/2601.10187](https://arxiv.org/abs/2601.10187)

    该论文提出了Sand-Glass音节级时长约束翻译基准和Homura强化学习框架，通过新颖的动态音节比率奖励有效解决LLM翻译的跨语言冗长偏差问题，使其适用于字幕、配音等时间受限场景。

    

    大语言模型（LLM）在多语言翻译方面取得了显著进展，但受到系统性的跨语言冗长偏差的阻碍，使其不适用于字幕制作和配音等严格时间受限的任务。当前的提示工程方法难以解决语义忠实性与严格时间可行性之间的冲突。为了弥合这一差距，我们首先介绍了Sand-Glass，一个专门设计用于在音节级时长约束下评估翻译效果的基准。此外，我们提出了Homura，一个显式优化语义保留与时间合规之间权衡的强化学习框架。通过采用包含新颖动态音节比率奖励的约束强化学习目标，Homura有效地“驯服”了输出长度。实验结果表明，Homura显著优于强大的基线方法，实现了……

    arXiv:2601.10187v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose Homura, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a constrained reinforcement learning objective featuring a novel dynamic syllable-ratio reward, Homura effectively "tames" the output length. Experimental results demonstrate that Homura significantly outperforms strong baselines, achieving pre
    
[^247]: PaperScout：一种基于过程感知的序列级策略优化的学术论文检索自主智能体

    PaperScout: An Autonomous Agent for Academic Paper Search with Process-Aware Sequence-Level Policy Optimization

    [https://arxiv.org/abs/2601.10029](https://arxiv.org/abs/2601.10029)

    提出PaperScout自主智能体，将学术论文检索重构为序贯决策过程，并通过过程感知的序列级策略优化，解决了标准强化学习在多轮智能体任务中词元级优化与序列级交互之间的粒度不匹配问题。

    

    学术论文检索是科学研究的一项基础任务，然而现有的大多数方法都围绕预定义的工作流程或结构化交互协议来组织检索，难以应对复杂的条件式查询。为了解决这一局限性，我们提出了PaperScout，一个将论文检索重新构建为序贯决策过程的自主智能体。与静态工作流程不同，PaperScout能够基于累积的检索上下文，动态决定是否、何时以及如何调用搜索和扩展工具。然而，训练此类智能体面临一个根本性挑战：标准的强化学习方法通常为单轮任务设计，在应用于多轮智能体任务时存在粒度不匹配的问题——词元级优化与序列级交互的粒度不一致，导致噪声较大的信用分配和不稳定的训练动态。我们引入了近端序列策略优化方法以应对这一问题。

    arXiv:2601.10029v3 Announce Type: replace  Abstract: Academic paper search is a fundamental task in scientific research, yet most existing approaches organize retrieval around predefined workflows or structured interaction protocols that struggle with complex, conditional queries. To address this limitation, we propose PaperScout, an autonomous agent that reformulates paper search as a sequential decision-making process. Unlike static workflows, PaperScout dynamically decides whether, when, and how to invoke search and expand tools based on accumulated retrieval context. However, training such agents presents a fundamental challenge: standard reinforcement learning methods, typically designed for single-turn tasks, suffer from a granularity mismatch when applied to multi-turn agentic tasks, where token-level optimization diverges from the granularity of sequence-level interactions, leading to noisy credit assignment and unstable training dynamics. We introduce Proximal Sequence Policy 
    
[^248]: 先想象后规划：基于世界模型自适应前瞻的智能体学习

    Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models

    [https://arxiv.org/abs/2601.08955](https://arxiv.org/abs/2601.08955)

    提出了ITP统一框架，让智能体策略模型与世界模型交互生成多步想象轨迹，并通过权衡最终目标与任务进度的自适应前瞻机制，充分释放世界模型在复杂任务规划中的潜力。

    

    世界模型的最新进展在建模环境状态的未来动态方面展现出巨大潜力，使智能体无需访问真实环境即可进行推理与行动。然而，当前方法主要执行单步或固定时域的推演，其在复杂任务规划中的潜力尚未得到充分挖掘。我们提出了“先想象后规划”（Imagine-then-Plan, ITP），这是一个通过前瞻想象进行智能体学习的统一框架，其中智能体的策略模型与学习到的世界模型进行交互，生成多步“想象”轨迹。由于想象时域可能因任务和阶段的不同而变化，我们引入了一种新颖的自适应前瞻机制，通过权衡最终目标与任务进度来确定想象步长。由此得到的想象轨迹提供了关于未来后果的丰富信号，例如已取得的进展和潜在的冲突，这些信号与当前观测相融合，构成了部分可观测与想象的（摘要在此处截断）

    arXiv:2601.08955v3 Announce Type: replace-cross  Abstract: Recent advances in world models have shown promise for modeling future dynamics of environmental states, enabling agents to reason and act without accessing real environments. Current methods mainly perform single-step or fixed-horizon rollouts, leaving their potential for complex task planning under-exploited. We propose Imagine-then-Plan (\texttt{ITP}), a unified framework for agent learning via lookahead imagination, where an agent's policy model interacts with the learned world model, yielding multi-step ``imagined'' trajectories. Since the imagination horizon may vary by tasks and stages, we introduce a novel adaptive lookahead mechanism by trading off the ultimate goal and task progress. The resulting imagined trajectories provide rich signals about future consequences, such as achieved progress and potential conflicts, which are fused with current observations, formulating a partially \textit{observable} and \textit{imag
    
[^249]: FADTI：基于傅里叶与注意力驱动的扩散模型用于多变量时间序列插补

    FADTI: Fourier and Attention Driven Diffusion for Multivariate Time Series Imputation

    [https://arxiv.org/abs/2512.15116](https://arxiv.org/abs/2512.15116)

    提出FADTI框架，通过傅里叶偏置投影（FBP）模块在扩散去噪过程中注入可学习的频率感知偏置，并支持DFT、STFT、FSST多种频谱变换，从而有效提升多变量时间序列插补对周期性和非平稳模式的恢复能力。

    

    多变量时间序列插补在医疗保健、交通预测和生物建模等应用中至关重要，因为这些场景中传感器故障和不规则采样导致普遍存在缺失值。现有的基于Transformer和扩散模型的插补方法虽然性能优异，但它们主要依赖时域建模，缺乏用于恢复结构化时间间隙的自适应频谱偏置。我们提出了FADTI，一个用于多变量时间序列插补的傅里叶与注意力驱动的扩散框架。FADTI引入了傅里叶偏置投影（FBP）模块，在去噪过程中向中间隐藏状态注入可学习的频率感知偏置。它将中间隐藏状态投影到傅里叶基上，避免了从掩码或零填充输入中进行直接频谱估计。通过DFT、STFT和FSST的实例化，FBP能够捕获全局周期性、局部时频变化以及非平稳振荡。

    arXiv:2512.15116v3 Announce Type: replace-cross  Abstract: Multivariate time series imputation is fundamental in applications such as healthcare, traffic forecasting, and biological modeling, where sensor failures and irregular sampling lead to pervasive missing values. Existing Transformer- and diffusion-based imputers achieve strong performance, but they often rely mainly on time-domain modeling and lack adaptive spectral bias for recovering structured temporal gaps. We propose FADTI, a Fourier- and attention-driven diffusion framework for multivariate time series imputation. FADTI introduces a Fourier Bias Projection (FBP) module that injects learnable frequency-aware bias into intermediate hidden states during denoising. It projects intermediate hidden states onto Fourier bases, avoiding direct spectral estimation from masked or zero-filled inputs. With DFT, STFT, and FSST instantiations, FBP captures global periodicity, localized time--frequency variations, and non-stationary osci
    
[^250]: 进化卓越：基于大语言模型的智能体自动化优化

    Evolving Excellence: Automated Optimization of LLM-based Agents

    [https://arxiv.org/abs/2512.09108](https://arxiv.org/abs/2512.09108)

    本文提出ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子自动联合优化LLM智能体的提示词、工具描述和参数等配置，无需架构修改即可显著提升智能体性能。

    

    基于大语言模型（LLM）构建的智能体AI系统在自动化复杂工作流程方面具有巨大潜力，涵盖从软件开发到客户支持等应用场景。然而，LLM智能体常常因配置不佳而表现欠佳——调优不当的提示词、工具描述和参数通常需要数周的手工打磨。现有的优化方法要么过于复杂而难以通用，要么孤立地处理各个组件，忽略了组件之间关键的相互依赖关系。我们提出了ARTEMIS，一个无代码的进化优化平台，通过语义感知的遗传算子对智能体配置进行联合优化。只需提供一个基准测试脚本和自然语言目标，ARTEMIS即可自动发现可配置组件、从执行日志中提取性能信号，并在无需修改架构的情况下演化配置。我们在四个代表性的智能体系统上对ARTEMIS进行了评估。

    arXiv:2512.09108v2 Announce Type: replace-cross  Abstract: Agentic AI systems built on large language models (LLMs) offer significant potential for automating complex workflows, from software development to customer support. However, LLM agents often underperform due to suboptimal configurations; poorly tuned prompts, tool descriptions, and parameters that typically require weeks of manual refinement. Existing optimization methods either are too complex for general use or treat components in isolation, missing critical interdependencies.   We present ARTEMIS, a no-code evolutionary optimization platform that jointly optimizes agent configurations through semantically-aware genetic operators. Given only a benchmark script and natural language goals, ARTEMIS automatically discovers configurable components, extracts performance signals from execution logs, and evolves configurations without requiring architectural modifications.   We evaluate ARTEMIS on four representative agent systems: 
    
[^251]: 混合数据聚类综述与挑战

    Mixed Data Clustering Survey and Challenges

    [https://arxiv.org/abs/2512.03070](https://arxiv.org/abs/2512.03070)

    本文提出了一种基于预拓扑空间的混合数据聚类方法，能够有效处理同时包含数值型和分类型变量的异构数据，并提供层次化、可解释的聚类结果。

    

    大数据时代的到来改变了各行业管理和分析信息的方式，开启了数据体量、速度和种类都前所未有的时代。在这一背景下，混合数据聚类已成为一项关键挑战，需要能够有效利用异构数据类型（包括数值型和分类型变量）的创新方法。传统聚类技术通常是为同质数据集设计的，往往难以捕捉混合数据所带来的额外复杂性，这凸显了专门针对此类场景设计方法的必要性。在这一背景下，层次化和可解释的算法尤其具有价值，因为它们能够提供结构化、可解释的聚类结果，从而支持明智的决策。本文介绍了一种基于预拓扑空间（pretopological spaces）的聚类方法。此外，本文还与经典的数值聚类算法进行了基准测试比较……

    arXiv:2512.03070v2 Announce Type: replace-cross  Abstract: The advent of the big data paradigm has transformed how industries manage and analyze information, ushering in an era of unprecedented data volume, velocity, and variety. Within this landscape, mixed-data clustering has become a critical challenge, requiring innovative methods that can effectively exploit heterogeneous data types, including numerical and categorical variables. Traditional clustering techniques, typically designed for homogeneous datasets, often struggle to capture the additional complexity introduced by mixed data, underscoring the need for approaches specifically tailored to this setting. Hierarchical and explainable algorithms are particularly valuable in this context, as they provide structured, interpretable clustering results that support informed decision-making. This paper introduces a clustering method grounded in pretopological spaces. In addition, benchmarking against classical numerical clustering al
    
[^252]: AnyBox：面向机器人操作的高效零样本箱体9自由度位姿估计

    AnyBox: Efficient Zero-Shot 9DoF Pose Estimation of Boxes for Robotic Manipulation

    [https://arxiv.org/abs/2511.15884](https://arxiv.org/abs/2511.15884)

    AnyBox是一个高效的零样本框架，通过利用箱体的几何规则性，在单张RGB-D观测上交替进行位姿与尺度估计，从而实现对杂乱遮挡环境中箱体9自由度位姿（6D位姿+3D尺寸）的联合恢复，无需物体特定的CAD模型。

    

    在杂乱和遮挡环境下恢复物体的9D位姿（包括其6D位姿和3D尺寸）是仓储自动化、物流和制造领域的核心需求。基于模型的方法精度较高，但假设每个物体都拥有特定的CAD模型，随着库存变化，维护成本十分高昂。无模型和类别级方法放宽了这一假设，但它们对于堆叠储物箱所特有的对称性、弱纹理和严重遮挡仍然脆弱，且忽略了此类场景所提供的强结构先验。我们提出了AnyBox，这是一个高效的零样本框架，它利用箱体的几何规则性，从单次RGB-D观测中联合恢复位姿和尺寸。从一个规范化的类别模板出发，AnyBox在位姿估计与尺度估计之间交替进行，利用重投影模板与观测掩码之间的差异来驱动二……

    arXiv:2511.15884v2 Announce Type: replace-cross  Abstract: Recovering the 9D pose of objects, both their 6D pose and 3D dimensions, under clutter and occlusion is a core requirement for warehouse automation, logistics, and manufacturing. Model-based methods are accurate but assume an instance-specific CAD model for every object, which is costly to maintain as inventories change. Model-free and category-level methods relax this assumption, yet they remain vulnerable to the symmetry, weak texture, and heavy occlusion that characterize stacked storage boxes, and they ignore the strong structural priors such scenes provide. We present \textbf{AnyBox}, an efficient zero-shot framework that exploits the geometric regularity of boxes to jointly recover pose and dimensions from a single RGB-D observation. Starting from a canonical category template, AnyBox alternates between pose and scale estimation, using the discrepancy between the reprojected template and the observed mask to drive a binar
    
[^253]: 基于大语言模型自动标注的短窗口滑动学习用于实时暴力检测

    Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling

    [https://arxiv.org/abs/2511.10866](https://arxiv.org/abs/2511.10866)

    该论文提出一种短窗口滑动学习框架，通过LLM自动标注将视频切分为1-2秒短片段构建细粒度数据集，在保留时间连续性的同时实现了高精度实时暴力检测，在RWF-2000上达到95.25%准确率。

    

    本文提出了一种用于CCTV监控视频中实时暴力检测的短窗口滑动学习框架。与传统的长视频训练方法不同，该方法将视频分割为1-2秒的短片段，并应用基于大语言模型的自动字幕标注来构建细粒度数据集。每个短片段充分利用所有帧以保持时间连续性，从而能够精确识别快速发生的暴力事件。实验表明，该方法在RWF-2000数据集上达到95.25%的准确率，并在长视频数据集上显著提升性能（UCF-Crime：83.25%），证实了其在智能监控系统中强大的泛化能力和实时适用性。

    arXiv:2511.10866v2 Announce Type: replace-cross  Abstract: This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems.
    
[^254]: 用户感知与代理LLM评审：LLM对隐私敏感场景响应中的隐私性与有用性

    User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios

    [https://arxiv.org/abs/2510.20721](https://arxiv.org/abs/2510.20721)

    该研究通过94人的用户实验发现，用户对LLM在隐私敏感场景下响应的评价彼此一致性较低，这表明此前以代理LLM作为评审的基准测试结果可能与真实用户的隐私和有用性感知存在显著偏差。

    

    大语言模型（LLM）正迅速被应用于起草电子邮件、总结会议记录和回答健康问题等任务。在这些场景中，用户可能需要分享私人信息（例如联系方式、健康记录）。为评估LLM识别并隐去此类信息的能力，先前的研究引入了基于真实生活场景的基准测试（如ConfAIde、PrivacyLens），并发现LLM在复杂场景中可能会泄露私人信息。然而，这些评估依赖代理LLM来判断LLM响应的有用性和隐私保护质量，而非直接测量用户的真实感知。为了解用户如何感知LLM对隐私敏感场景响应的有用性与隐私保护质量，我们使用90个PrivacyLens场景开展了一项用户研究（样本量n=94）。我们发现，用户在评估相同的LLM响应时彼此之间的一致性较低。

    arXiv:2510.20721v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. I
    
[^255]: WELD：首个面向泛在情感计算的自然情境长周期小团队职场情绪数据集

    WELD: The First Naturalistic Long-Period Small-Team Workplace Emotion Dataset for Ubiquitous Affective Computing

    [https://arxiv.org/abs/2510.15221](https://arxiv.org/abs/2510.15221)

    WELD是首个结合数年持续时间、自然职场情境、稳定小团队结构与完全被动感知协议的职场情绪数据集，基于中国某软件公司49名员工超过30个月的面部表情数据构建。

    

    情感计算在实验室环境中已快速成熟，然而此前没有任何数据集能同时满足以下四点：(i) 数月至数年的持续时间，(ii) 自然的职场情境，(iii) 稳定的小团队社会结构，以及(iv) 能通过机构伦理审查的完全被动感知协议。我们推出了WELD——首个同时满足这四点的数据集。WELD包含来自中国某软件公司49名员工在30.1个月（2021年11月至2024年5月）期间采集的733,780个逐帧七类面部表情概率向量——这是最长的自然真实环境情绪语料库，也是唯一一个支持对同一批被试同时开展个体内纵向分析和团队内关系分析的多年度语料库。数据采用四级访问模型发布，仅有聚合概率可供公开下载。我们通过复现三个已确立的现象验证了该语料库（周末效价提升43.1%；13:00低谷的昼夜节律周期；上海……）

    arXiv:2510.15221v3 Announce Type: replace  Abstract: Affective computing has matured rapidly in laboratory settings, yet no prior dataset combines (i) months-to-years of duration, (ii) a naturalistic workplace context, (iii) a stable small-team social structure, and (iv) a fully passive sensing protocol that survives institutional review. We introduce WELD, the first dataset to satisfy all four. WELD comprises 733,780 per-frame seven-class facial-expression probability vectors from 49 employees of a Chinese software company over 30.1 months (Nov 2021 - May 2024) -- the longest naturalistic in-the-wild emotion corpus and the only multi-year corpus supporting both within-individual longitudinal and within-team relational analyses on the same subjects. Data are released under a four-tier access model with only aggregated probabilities publicly downloadable. We validate the corpus by replicating three established phenomena (+43.1% weekend valence boost; 13:00-trough diurnal cycle; Shanghai
    
[^256]: EasySteer：一个高性能且可扩展的大语言模型引导统一框架

    EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering

    [https://arxiv.org/abs/2509.25175](https://arxiv.org/abs/2509.25175)

    EasySteer是一个基于vLLM构建的高性能、可扩展的大语言模型推理时引导统一框架，相比现有框架实现了10.8-22.3倍的加速，并提供模块化可插拔接口、细粒度参数控制以及八个应用领域的预计算引导向量。

    

    大语言模型（LLM）引导已成为一种在推理时通过定向操作隐藏状态来控制模型行为的有前景的范式，为昂贵的重新训练提供了一种轻量级替代方案。然而，现有的引导框架存在关键局限性：计算效率低、可扩展性有限以及功能受限，这些都阻碍了研究进展和实际部署。我们提出了EasySteer，一个基于vLLM构建的高性能、可扩展的大语言模型引导统一框架。该系统具有模块化架构，为基于分析和基于学习的方法提供可插拔接口，支持细粒度参数控制，提供八个应用领域的预计算引导向量，并配备了交互式演示系统。通过与vLLM优化的推理引擎深度集成，EasySteer相比现有框架实现了10.8-22.3倍的加速。

    arXiv:2509.25175v3 Announce Type: replace-cross  Abstract: Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 10.8-22.3$\times$ speedup over existing frameworks. Extensi
    
[^257]: 人类心理测量问卷无法准确刻画大语言模型的行为

    Human Psychometric Questionnaires Mischaracterize LLM Behavior

    [https://arxiv.org/abs/2509.10078](https://arxiv.org/abs/2509.10078)

    该研究发现，人类心理测量问卷中的题目含有明显的词汇线索，会让大语言模型识别出被测构念并给出符合社会期望的回答，因此基于问卷得到的模型人格与价值观画像并不能反映其在真实日常用户交互中的实际生成行为。

    

    我们检验了人类心理测量问卷能否作为可靠工具来刻画和预测大语言模型在日常用户交互中的行为。我们分析了八个开源大语言模型，通过两种不同的方法比较其价值观与人格画像：一是基于既定问卷（PVQ-40/21 和 BFI-44/10）的利克特量表自我报告，二是对日常用户查询中带有价值倾向回答的生成概率。两种画像存在显著分歧。通常被引用为大语言模型稳定倾向证据的构念内题目一致性，在生成概率中消失了。我们发现，既定的问卷题目包含明确的词汇线索，使模型能够识别目标构念，并以与构念一致、符合社会期望的方式作答；而真实的用户查询所包含的可识别线索要少得多。此外，人口统计角色提示会使模型对人类问卷的回答发生偏移……

    arXiv:2509.10078v5 Announce Type: replace-cross  Abstract: We examine whether human psychometric questionnaires can serve as reliable tools for characterizing and predicting LLM behavior in everyday user interactions. We analyze eight open-source LLMs by comparing their value and personality profiles derived from two different methods: Likert self-reports on established questionnaires (PVQ-40/21 and BFI-44/10) and generation probabilities over value-laden responses to everyday user queries. The two profiles diverge substantially. Within-construct item consistency, often cited as evidence of stable LLM dispositions, disappears in generation probabilities. We find that established questionnaire items contain explicit lexical cues that allow models to recognize the target construct and respond in alignment-consistent, socially desirable ways, whereas realistic user queries contain far less recognizable cues. In addition, demographic persona prompts shift models' responses to human questio
    
[^258]: 基于视觉的去中心化自主空中野生动物监测系统

    Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring

    [https://arxiv.org/abs/2508.15038](https://arxiv.org/abs/2508.15038)

    提出了一种仅依赖单个机载RGB相机、无需集中式通信的去中心化多旋翼无人机系统，实现了在自然栖息地中对野生动物的鲁棒识别与跟踪。

    

    野生动物野外作业需要高效的并行部署方法来识别并与特定个体互动，从而实现同步的集体行为分析以及健康与安全干预。以往的机器人解决方案从群体角度处理该问题，或者是手动操作且规模有限。我们提出了一种用于野生动物监测的去中心化、基于视觉的多旋翼无人机系统，该系统具有可扩展性、低带宽需求且传感器精简（仅使用单个机载RGB相机）。我们的方法能够在自然栖息地中对大型物种进行鲁棒的识别与跟踪。我们开发了新颖的基于视觉的协调与跟踪算法，专为动态、非结构化环境设计，无需依赖集中式通信或控制。我们通过真实世界实验验证了该系统，展示了其在多种野外条件下的可靠部署。

    arXiv:2508.15038v2 Announce Type: replace-cross  Abstract: Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions.
    
[^259]: 测量计算机使用代理的危害性

    Measuring Harmfulness of Computer-Using Agents

    [https://arxiv.org/abs/2508.00935](https://arxiv.org/abs/2508.00935)

    该论文提出了 CUAHarm 基准，通过 104 个专家撰写的真实滥用任务和基于规则的可验证沙盒环境评估计算机使用代理的滥用风险，发现前沿语言模型（如 Gemini 2.5 Pro 成功率达 90%）即使没有越狱提示也会以高成功率执行恶意计算机操作。

    

    计算机使用代理（CUA）能够自主控制计算机执行多步骤操作，一旦被滥用可能构成重大安全风险。然而，现有的基准测试主要评估语言模型在聊天机器人或简单工具使用场景中的表现。为了更全面地评估计算机使用代理的滥用风险，我们引入了一个新的基准测试：CUAHarm。CUAHarm 包含 104 个由专家撰写的真实滥用风险任务，例如禁用防火墙、泄露数据或安装后门。我们提供了一个带有基于规则的可验证奖励的沙盒环境，用于衡量计算机使用代理执行这些任务的成功率（例如防火墙是否确实被禁用），而不仅仅是拒绝率。我们评估了包括 GPT-5、Claude 4 Sonnet、Gemini 2.5 Pro、Llama-3.3-70B 和 Mistral Large 2 在内的前沿语言模型。即使不使用越狱提示，这些前沿语言模型也会以较高的成功率配合执行这些恶意任务（例如 Gemini 2.5 Pro 的成功率为 90%）。此外，虽然较新的模型在安全……

    arXiv:2508.00935v3 Announce Type: replace-cross  Abstract: Computer-using agents (CUAs), which can autonomously control computers to perform multi-step actions, might pose significant safety risks if misused. However, existing benchmarks mainly evaluate LMs in chatbots or simple tool use. To more comprehensively evaluate CUAs' misuse risks, we introduce a new benchmark: CUAHarm. CUAHarm consists of 104 expert-written realistic misuse risks, such as disabling firewalls, leaking data, or installing backdoors. We provide a sandbox with rule-based verifiable rewards to measure CUAs' success rates in executing these tasks (e.g., whether the firewall is indeed disabled), beyond refusal rates. We evaluate frontier LMs including GPT-5, Claude 4 Sonnet, Gemini 2.5 Pro, Llama-3.3-70B, and Mistral Large 2. Even without jailbreaking prompts, these frontier LMs comply with executing these malicious tasks at a high success rate (e.g., 90% for Gemini 2.5 Pro). Furthermore, while newer models are safe
    
[^260]: 大语言模型时代的医学推理：增强技术与应用的系统综述

    Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications

    [https://arxiv.org/abs/2508.00669](https://arxiv.org/abs/2508.00669)

    本文是首个针对大语言模型医学推理领域的系统综述，提出了涵盖训练时策略与测试时机制的推理增强技术分类体系，并系统分析了这些技术在多种数据模态和临床应用中的实践与评估方法。

    

    大语言模型在医学领域的蓬勃发展带来了令人印象深刻的能力，但其在执行系统性、透明性和可验证性推理方面仍存在关键差距，而这正是临床实践的基石。这一差距推动了从单步答案生成向专为医学推理设计的大语言模型发展的范式转变。本文对该新兴领域进行了首次系统综述。我们提出了一个推理增强技术的分类体系，将其分为训练时策略（如监督微调、强化学习）和测试时机制（如提示工程、多智能体系统）。我们分析了这些技术如何应用于不同的数据模态（文本、图像、代码）以及关键的医疗场景中，如诊断、教育和治疗规划。此外，我们还梳理了评估基准从简单准确率指标……（原文摘要到此截断）

    arXiv:2508.00669v2 Announce Type: replace-cross  Abstract: The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy
    
[^261]: 可追溯TTS：迈向具有强可追溯性的无水印文本转语音

    Traceable TTS: Toward Watermark-Free TTS with Strong Traceability

    [https://arxiv.org/abs/2507.03887](https://arxiv.org/abs/2507.03887)

    该论文提出首个无水印的可追溯TTS框架，通过TTS模型与判别器的联合训练实现合成语音的模型归因，在保持甚至略微提升音频质量的同时显著增强了可追溯性的泛化能力。

    

    文本转语音（TTS）技术的最新进展使得合成语音能够以极高的逼真度模仿人类声音，这引发了重大的安全担忧。这凸显了对可追溯TTS模型的需求——即能够在不损害质量或安全性的前提下追踪其合成语音的系统。然而，现有方法主要依赖于在语音或声码器上嵌入显式水印，这不仅会降低语音质量，还容易受到伪造攻击。为解决这些局限性，我们提出了一种新颖的模型归因框架。该方法不嵌入水印，而是采用联合训练的方式训练TTS模型和判别器，显著提升了可追溯性的泛化能力，同时保持甚至略微改善了音频质量。这是首个迈向具有强可追溯性的无水印TTS的工作。为促进相关领域的发展，我们将在论文被接收后发布代码。

    arXiv:2507.03887v1 Announce Type: cross  Abstract: Recent advances in Text-To-Speech (TTS) technology have enabled synthetic speech to mimic human voices with remarkable realism, raising significant security concerns. This underscores the need for traceable TTS models-systems capable of tracing their synthesized speech without compromising quality or security. However, existing methods predominantly rely on explicit watermarking on speech or on vocoder, which degrades speech quality and is vulnerable to spoofing. To address these limitations, we propose a novel framework for model attribution. Instead of embedding watermarks, we train the TTS model and discriminator using a joint training method that significantly improves traceability generalization while preserving-and even slightly improving-audio quality. This is the first work toward watermark-free TTS with strong traceability. To promote progress in related fields, we will release the code upon acceptance of the paper.
    
[^262]: ScoreMix：通过扩散模型中的分数组合进行合成数据生成以提升识别性能

    ScoreMix: Synthetic Data Generation by Score Composition in Diffusion Models Improves Recognition

    [https://arxiv.org/abs/2506.10226](https://arxiv.org/abs/2506.10226)

    提出ScoreMix方法，利用扩散模型的分数可组合性、在无需外部模型或数据集的情况下，通过混合判别器嵌入空间中相距较远的类别生成分数条件合成样本，为识别任务带来最高3%的平均性能提升。

    

    合成数据生成在机器学习中被越来越多地用于模型训练和数据增强。然而，现有策略通常依赖于外部基础模型或数据集，而由于政策或法律的限制，这些资源在许多场景下无法使用。我们提出了ScoreMix，这是一种自包含的合成数据生成方法，通过利用扩散模型的分数可组合性，为识别任务生成困难的合成样本。该方法沿反向扩散轨迹混合类条件分数，在无需外部资源的情况下实现特定领域的数据增强。我们系统地研究了类别选择策略，发现混合判别器嵌入空间中相距较远的类别能够带来更大的收益，与基于接近度的类别选择方式相比，平均可获得高达3%的额外提升。有趣的是，我们观察到在标准（条件下）……

    arXiv:2506.10226v3 Announce Type: replace-cross  Abstract: Synthetic data generation is increasingly used in machine learning for training and data augmentation. Yet, current strategies often rely on external foundation models or datasets, whose usage is restricted in many scenarios due to policy or legal constraints. We propose ScoreMix, a self-contained synthetic generation method to produce hard synthetic samples for recognition tasks by leveraging the score compositionality of diffusion models. The approach mixes class-conditioned scores along reverse diffusion trajectories, yielding domain-specific data augmentation without external resources. We systematically study class-selection strategies and find that mixing classes distant in the discriminator's embedding space yields larger gains, providing up to 3% additional average improvement, compared to selection based on proximity. Interestingly, we observe that condition and embedding spaces are largely uncorrelated under standard 
    
[^263]: RECAST：利用多约束数据拓展大语言模型复杂指令遵循能力的边界

    RECAST: Expanding the Boundaries of LLMs' Complex Instruction Following with Multi-Constraint Data

    [https://arxiv.org/abs/2505.19030](https://arxiv.org/abs/2505.19030)

    提出RECAST框架，通过从真实提示-响应对中提取约束，高效合成每个样本包含远超10个约束条件的数据集，突破了现有基准限制，拓展了大语言模型复杂指令遵循能力的边界。

    

    arXiv:2505.19030v5 公告类型：替换 摘要：随着大语言模型（LLM）应用的不断扩展以及用户编写复杂提示词能力的日益提高，人们越来越期望大语言模型能够处理复杂任务。然而，当明确陈述的要求数量增加时（特别是超过10个约束条件），大语言模型往往难以准确遵循此类复杂指令，这限制了它们在复杂现实场景中的适用性。据我们所知，现有数据集中每个实例包含的约束条件均不超过10个。为应对这一挑战，我们提出了RECAST，这是一个高效且可扩展的数据集合成框架，其中每个样本所包含的约束条件远超现有基准，旨在挑战并拓展模型遵循复杂指令能力的边界。这些约束条件提取自真实世界的提示-响应对，以确保其实际相关性。利用该框架，我们构建了

    arXiv:2505.19030v5 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions, which limits their applicability in complex real-world scenarios. To the best of our knowledge, existing datasets do not exceed 10 constraints per instance. To address this challenge, we propose RECAST, an efficient and scalable framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks, aiming to challenge and extend the boundaries of models' ability to follow complex instructions. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. Using this framework, we construct 
    
[^264]: LightEMMA：自动驾驶视觉语言模型的纵向评估

    LightEMMA: A Longitudinal Evaluation of Vision-Language Models for Autonomous Driving

    [https://arxiv.org/abs/2505.00284](https://arxiv.org/abs/2505.00284)

    提出了LightEMMA纵向评估框架，通过轻量级统一协议在nuScenes基准上评估15个视觉语言模型的驾驶性能，发现新一代VLM的规模和推理能力提升并不能持续带来更好的自动驾驶表现。

    

    视觉语言模型（VLM）的快速发展引发了人们对其在自动驾驶领域应用的浓厚兴趣。一个普遍的假设是，新一代VLM将持续提升驾驶性能，并最终超越最先进的方法。为了系统地检验这一假设，我们提出了LightEMMA，一个用于评估VLM自动驾驶性能的纵向评估框架。LightEMMA采用轻量级、统一的评估协议，在无需模型特定微调、架构更改或提示工程的情况下，评估每个模型的内在驾驶能力。利用该协议，我们在具有挑战性的nuScenes预测基准上评估了来自五个主要模型家族的15个模型。实证结果表明，尽管模型规模不断扩大、通用推理能力不断增强，但新一代VLM并不能持续获得更好的驾驶性能。

    arXiv:2505.00284v3 Announce Type: replace-cross  Abstract: Rapid advances in vision-language models (VLMs) have generated growing interest in their application to autonomous driving. A prevailing assumption is that successive VLM generations will continually improve driving performance and eventually outperform state-of-the-art methods. To systematically examine this assumption, we introduce LightEMMA, a longitudinal framework for evaluating the autonomous driving performance of VLMs. LightEMMA uses a lightweight, unified evaluation protocol that assesses each model's intrinsic driving capability without model-specific fine-tuning, architectural changes, or prompt engineering. Using this protocol, we evaluate 15 models from five major families on the challenging nuScenes prediction benchmark. Empirical findings show that, despite increased model scale and enhanced general reasoning capabilities, successive VLM generations do not consistently achieve better driving performance. Further 
    
[^265]: Sionna RT：技术报告

    Sionna RT: Technical Report

    [https://arxiv.org/abs/2504.21719](https://arxiv.org/abs/2504.21719)

    Sionna RT是一个开源、GPU加速且可微分的射线追踪器，本文详细介绍了其高效模拟无线电波传播（包括信道脉冲响应与无线电地图计算）的核心算法、Sionna 1.0全面重构带来的速度与内存效率提升以及现有算法的局限性。

    

    Sionna是一个开源的、基于GPU加速的库，从0.14版本起集成了用于模拟无线电波传播的射线追踪器Sionna RT。Sionna RT的一个独特特性是可微分性，使其能够计算信道脉冲响应（CIR）、无线电地图及其他相关指标相对于系统和环境参数（如材料属性、天线方向图和阵列几何结构）的梯度。Sionna 1.0的发布对射线追踪器进行了全面重构，显著提升了其运行速度、内存效率和可扩展性。本文详细介绍了Sionna RT用于高效模拟无线电波传播所采用的算法，并探讨了它们目前存在的局限性。鉴于信道脉冲响应和无线电地图的计算需要不同的算法，文中分别在不同章节予以阐述。对于信道脉冲响应的计算，Sionna RT将射线发射与弹射（SBR）方法与……

    arXiv:2504.21719v3 Announce Type: replace-cross  Abstract: Sionna is an open-source, GPU-accelerated library that, as of version 0.14, incorporates a ray tracer, Sionna RT, for simulating radio wave propagation. A unique feature of Sionna RT is differentiability, enabling the calculation of gradients for the channel impulse responses (CIRs), radio maps, and other related metrics with respect to system and environmental parameters, such as material properties, antenna patterns, and array geometries. The release of Sionna 1.0 provided a complete overhaul of the ray tracer, significantly improving its speed, memory efficiency, and extensibility. This document details the algorithms employed by Sionna RT to simulate radio wave propagation efficiently, while also addressing their current limitations. Given that the computation of CIRs and radio maps requires distinct algorithms, these are detailed in separate sections. For CIRs, Sionna RT integrates shooting and bouncing of rays (SBR) with 
    
[^266]: AgentRM：通过奖励建模增强智能体的泛化能力

    AgentRM: Enhancing Agent Generalization with Reward Modeling

    [https://arxiv.org/abs/2502.18407](https://arxiv.org/abs/2502.18407)

    本论文提出可泛化奖励模型AgentRM，发现微调奖励模型来引导测试时搜索比直接微调策略模型更稳健，在九个智能体任务上平均提升8.8分并超越最强通用智能体4.0分。

    

    现有的基于大语言模型（LLM）的智能体在已见任务上取得了强大的性能，但它们对未见任务的泛化能力仍然较差。因此，近期的一些工作专注于使用更多样化的任务对策略模型进行微调以提升泛化能力。在这项工作中，我们发现微调一个奖励模型来引导策略模型，比直接微调策略模型更加稳健。基于这一发现，我们提出了AgentRM，一个可泛化的奖励模型，用于引导策略模型进行有效的测试时搜索。我们全面研究了构建奖励模型的三种方法，包括显式奖励建模、隐式奖励建模以及LLM作为评判者（LLM-as-a-judge）。随后，我们使用AgentRM通过Best-of-N采样和步级束搜索来引导答案生成。在四类共九个智能体任务上，AgentRM使基础策略模型平均提升了8.8分，超越了最顶尖的通用智能体4.0分。此外，它……

    arXiv:2502.18407v2 Announce Type: replace-cross  Abstract: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it
    
[^267]: LDC：通过动态控制学习生成研究想法

    LDC: Learning to Generate Research Idea with Dynamic Control

    [https://arxiv.org/abs/2412.14626](https://arxiv.org/abs/2412.14626)

    提出首个结合监督微调与可控强化学习的两阶段框架，利用多维奖励模型和细粒度反馈动态控制研究想法生成，从而在新颖性、可行性和有效性之间实现平衡，提升大语言模型科研构思质量。

    

    arXiv:2412.14626v3 公告类型：交叉替换 摘要：大型语言模型（LLMs）的最新进展已展现出其在自动化科学研究构思方面的潜力。现有方法主要侧重于提示技术，往往产生的想法与专家标准不符——即新颖性、可行性和有效性，这些被研究界广泛认可为高质量研究想法的三个关键子维度。此外，由于这些维度之间存在固有的权衡关系，平衡这些维度仍然具有挑战性。为解决这些局限性，我们提出了首个采用两阶段方法的框架，结合监督微调（SFT）和可控强化学习（RL）来完成该任务。在SFT阶段，模型从研究论文及其对应后续想法的配对数据中学习基础模式。在RL阶段，由细粒度反馈引导的多维奖励模型在关键维度上对模型进行评估和优化。

    arXiv:2412.14626v3 Announce Type: replace-cross  Abstract: Recent advancements in large language models (LLMs) have demonstrated their potential in automating the scientific research ideation. Existing approaches primarily focus on prompting techniques, often producing ideas misaligned with expert standards - novelty, feasibility, and effectiveness, which are widely recognized by the research community as the three key subdimensions of high-quality ideas. Also, balancing these dimensions remains challenging due to their inherent trade-offs. To address these limitations, we propose the first framework that employs a two-stage approach combining Supervised Fine-Tuning (SFT) and controllable Reinforcement Learning (RL) for the task. In the SFT stage, the model learns foundational patterns from pairs of research papers and their corresponding follow-up ideas. In the RL stage, multi-dimensional reward models guided by fine-grained feedback evaluate and optimize the model across key dimensio
    
[^268]: 语法对齐解码

    Grammar-Aligned Decoding

    [https://arxiv.org/abs/2405.21047](https://arxiv.org/abs/2405.21047)

    本文揭示了语法约束解码会扭曲大语言模型的输出分布，导致生成结果虽符合语法但质量低下，并提出了一种名为ASAp的语法对齐解码算法来解决这一问题。

    

    大语言模型（LLMs）难以可靠地生成高度结构化的输出，例如程序代码、数学公式或格式良好的标记语言。约束解码方法通过在每个步骤贪心地限制LLM可以输出的token来缓解这一问题，从而保证输出符合给定的约束。具体而言，在语法约束解码（GCD）中，LLM的输出必须遵循给定的语法。在本文中，我们证明了GCD技术（以及广义上的约束解码技术）会扭曲LLM的分布，导致输出虽然符合语法，但其出现的概率与LLM本身给出的概率不成比例，因此最终质量较低。我们将采样与语法约束对齐这一问题称为语法对齐解码（GAD），并提出了一种名为ASAp（基于近似期望未来的自适应采样）的解码算法，该算法保证输出……

    arXiv:2405.21047v4 Announce Type: replace  Abstract: Large Language Models (LLMs) struggle with reliably generating highly structured outputs, such as program code, mathematical formulas, or well-formed markup. Constrained decoding approaches mitigate this problem by greedily restricting what tokens an LLM can output at each step to guarantee that the output matches a given constraint. Specifically, in grammar-constrained decoding (GCD), the LLM's output must follow a given grammar. In this paper, we demonstrate that GCD techniques (and in general constrained decoding techniques) can distort the LLM's distribution, leading to outputs that are grammatical but appear with likelihoods that are not proportional to the ones given by the LLM, and so ultimately are low-quality. We call the problem of aligning sampling with a grammar constraint, grammar-aligned decoding (GAD), and propose adaptive sampling with approximate expected futures (ASAp), a decoding algorithm that guarantees the outpu
    
[^269]: 通过深度学习进行数据市场设计

    Data Market Design through Deep Learning. (arXiv:2310.20096v1 [cs.GT])

    [http://arxiv.org/abs/2310.20096](http://arxiv.org/abs/2310.20096)

    这项研究介绍了使用深度学习进行收入最优数据市场设计的应用，旨在扩展前沿研究领域。

    

    $\textit{数据市场设计}$问题是经济理论中的一个问题，旨在找到一组信号方案（统计实验），以最大化信息卖方的预期收入，其中每个实验揭示了卖方所知道的一些信息，并附带一个相应的价格[Bergemann et al., 2018]。每个买方在世界环境中都有自己的决策，并且他们对与特定实验相关联的信息的主观预期值来自于这个决策的改进，并且依赖于他们的先验和不同结果的价值。在具有多个买方的环境中，买方对实验的预期值也可能取决于卖给其他人的信息[Bonatti et al., 2022]。我们引入深度学习在收入最优数据市场设计中的应用，旨在扩展可以被理解和实现的边界。相对于之前关于拍卖设计的深度学习研究[D\"utting et al., 2023]，我们必须进行更多的研究来解决数据市场设计问题。

    The $\textit{data market design}$ problem is a problem in economic theory to find a set of signaling schemes (statistical experiments) to maximize expected revenue to the information seller, where each experiment reveals some of the information known to a seller and has a corresponding price [Bergemann et al., 2018]. Each buyer has their own decision to make in a world environment, and their subjective expected value for the information associated with a particular experiment comes from the improvement in this decision and depends on their prior and value for different outcomes. In a setting with multiple buyers, a buyer's expected value for an experiment may also depend on the information sold to others [Bonatti et al., 2022]. We introduce the application of deep learning for the design of revenue-optimal data markets, looking to expand the frontiers of what can be understood and achieved. Relative to earlier work on deep learning for auction design [D\"utting et al., 2023], we must l
    

