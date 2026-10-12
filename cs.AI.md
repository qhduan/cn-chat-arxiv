# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Language Models Play the Wikipedia Game?](https://arxiv.org/abs/2610.13948) | 该论文提出以维基百科游戏作为信息检索基准，通过OpenAlex构建大规模本地维基百科页面图与论文引用图，在多语言和多学术数据集上系统比较句子嵌入模型与大语言模型的导航检索能力，发现没有任何模型能显著超越人类表现。 |
| [^2] | [Hide and Seek: HIDE Concepts to SEEK explanations for black box models](https://arxiv.org/abs/2610.13935) | 提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。 |
| [^3] | [Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping](https://arxiv.org/abs/2610.13909) | 该论文系统性地重新评估了表格深度回归中的各类目标函数，发现分位数损失是唯一能相对调优后的MSE基线带来一致提升的目标函数，并进一步揭示了由于不同噪声水平的区域需要不同的训练时机，全局早停策略存在根本性局限。 |
| [^4] | [Task Context Shapes Component Roles in Language Models](https://arxiv.org/abs/2610.13895) | 该论文揭示了任务上下文会重新分配语言模型中共享注意力头的影响力，表明同一模型组件在不同任务中扮演着不同的功能性角色。 |
| [^5] | [RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics](https://arxiv.org/abs/2610.13889) | RATED通过冻结双编码器进行语义匹配，并用轻量级上下文老虎机模块根据实时执行反馈重排序候选工具，无需模型重训即可快速适应运行时工具故障，同时提出含5,172个真实工具的EcoAgent基准，实现52.2%的相对MRR提升。 |
| [^6] | [X-AgentRed: Red-Teaming Personal Agents in Evolving Multi-Service Ecosystems](https://arxiv.org/abs/2610.13886) | 提出X-AgentRed，一个自进化的红队测试框架，通过工具依赖图系统性地发现多服务生态系统中个人智能体的有害工具调用路径，克服了现有方法偏向熟悉工具组合、忽略新颖跨服务交互的局限。 |
| [^7] | [Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga](https://arxiv.org/abs/2610.13840) | 提出PAGE框架，利用人工智能将漫画扫描件自动解析为分格与文本，并借助视觉语言模型按编码手册进行标注和与人工标注对齐，实现了对3929本美国漫画和87卷日本漫画的大规模整体化研究。 |
| [^8] | [SpatialASM: Internalizing Active Mentalization for Spatial Reasoning](https://arxiv.org/abs/2610.13832) | 提出SpatialASM框架，通过训练时的空间模拟与新视角生成，将主动空间心智化（ASM）能力内化到大型视觉语言模型中，显著提升其空间推理能力。 |
| [^9] | [MC-EvoBench: Auditing Experience-Driven Self-Improvement in Minecraft Agents](https://arxiv.org/abs/2610.13829) | MC-EvoBench是一个受控的Minecraft基准，通过19条注册进化链、任务状态图与严格的源-目标边界控制，审计智能体能否将交互经验真正转化为可跨世界持久化并迁移到后续任务的可复用能力。 |
| [^10] | [Same Query, Different Answer: Run-to-Run Variance in Semantic Query Processing](https://arxiv.org/abs/2610.13823) | 该研究首次系统测量了语义查询引擎多次执行间的结果方差，发现即使在温度为0时也有最高超过10%的行输出会改变，且查询计划会指数级放大这种不确定性，导致约三分之二的最终答案发生变化，说明基于少量执行的基准测试评估并不可靠。 |
| [^11] | [Your Discrete Graph Diffusion Model is Secretly Memorizing](https://arxiv.org/abs/2610.13786) | 本文首次证明离散图扩散模型存在“条件记忆化”现象——当反向扩散轨迹以损坏的训练图为条件时，模型几乎能完美恢复该训练图，而这种记忆化在常规推理和样本级评估指标中均无法被察觉。 |
| [^12] | [UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes](https://arxiv.org/abs/2610.13783) | 本文发布了包含200张乌克兰战时梗图的说服技术检测数据集UkrMeme及乌克兰语翻译基准PTM-UA，并评估了视觉-语言模型在零样本多模态设置下的说服检测性能及可解释性方法。 |
| [^13] | [ReplaySSM: Cache SSM Inputs, Not State](https://arxiv.org/abs/2610.13778) | ReplaySSM 通过将 SSM 状态表示为周期性更新的检查点加近期输入缓冲区，将状态内存流量约减半，使投机解码回滚变为简单的缓冲区指针操作，并支持以低精度检查点进行输出计算。 |
| [^14] | [DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs](https://arxiv.org/abs/2610.13767) | 提出了免校准的双尺度训练后量化框架DualQuant，利用最小化权重重建误差所得的缩放因子作为激活缩放，无需校准数据即可同时完成大语言模型的权重与激活低比特量化。 |
| [^15] | [When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors](https://arxiv.org/abs/2610.13765) | 本文指出离散图扩散模型的E-R随机图先验是GNN去噪器失效的根本原因，并提出潜在社交性先验（LSP）与联合扩散过程，使节点社交性变量与图结构共同演化，从而恢复生成图中的节点异质性和高阶结构。 |
| [^16] | [Matched-Context Group Policy Optimization for LLM Agent Training](https://arxiv.org/abs/2610.13755) | MCGPO通过匹配上下文从已有轨迹中构建加权动作偏好，在无需critic和额外rollout的情况下，为稀疏终端反馈下长时程LLM智能体的中间决策分配信用。 |
| [^17] | [A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning](https://arxiv.org/abs/2610.13744) | 本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。 |
| [^18] | [SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.13737) | SafeDiSTL利用信号时序逻辑鲁棒性与可微动力学提供的动作敏感度，将团队级联合约束代价分解并分配到各智能体，解决了联合约束下多智能体强化学习的信用分配难题。 |
| [^19] | [Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers](https://arxiv.org/abs/2610.13735) | Orthros注意力通过相位感知的方式在序列处理的不同阶段动态切换计算模式，实现了线性复杂度的预填充与完整注意力的解码相结合，从而在重预填充、轻解码的长文本场景下比原始Transformer具有更低的渐近复杂度。 |
| [^20] | [LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos](https://arxiv.org/abs/2610.13732) | 提出LAPT-Scale范式，通过在人类视频上学习连续潜在动作进行预训练，并结合机器人具身对齐与任务后训练，从而以低成本扩展通用机器人VLA策略的能力。 |
| [^21] | [Amortized Counterfactual Prediction with Graph Neural Networks](https://arxiv.org/abs/2610.13724) | 提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。 |
| [^22] | [Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy](https://arxiv.org/abs/2610.13708) | Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。 |
| [^23] | [A Baseline Assessment for the Agentic Web Technology Stack](https://arxiv.org/abs/2610.13703) | 本文提出了MASTIF多智能体系统测试框架，对由3个LLM、3种协议和4个智能体实现框架组成的智能体网络技术栈在300个Mind2Web基准任务上进行了基线评估，发现针对特定任务选择技术栈可以提升任务完成率、效率得分和社会价值稳定性。 |
| [^24] | [Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks](https://arxiv.org/abs/2610.13693) | 该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。 |
| [^25] | [Variance-Reduced Policy Gradient is Nearly Minimax-Optimal](https://arxiv.org/abs/2610.13692) | 该论文提出LVR-PG统一理论框架，证明了一类结合递归更新与防御性重要性采样的一阶方差缩减策略梯度算法，在弱梯度支配等最小假设下、无需梯度预处理器，即可达到 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度的近乎极小极大最优最后一次迭代全局收敛。 |
| [^26] | [Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels](https://arxiv.org/abs/2610.13674) | 该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。 |
| [^27] | [Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction](https://arxiv.org/abs/2610.13672) | 该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。 |
| [^28] | [A Step Harder, Not a Step Too Far: The Benefits and Limits of Curriculum Difficulty Progression in GRPO](https://arxiv.org/abs/2610.13669) | 该论文提出在GRPO训练中基于当前策略动态重估样本难度，并设计逐步向更难样本推进的分阶段课程，实验表明持续只选中等难度样本并非最优，适度渐进地提升难度才能带来更强的最终性能，但难度提升也存在限度。 |
| [^29] | [A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models](https://arxiv.org/abs/2610.13663) | 本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。 |
| [^30] | [Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM](https://arxiv.org/abs/2610.13660) | 本文在统一协议下控制变量地对比视觉型VARC与递归型TRM模型，证明预训练与测试时训练预算（尤其是增加TTT样本呈现次数）对ARC性能有决定性影响，提示性能差异不能简单归因于架构本身。 |
| [^31] | [SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data](https://arxiv.org/abs/2610.13644) | 该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。 |
| [^32] | [Learning Size Independent Constraint Models from Input Data and Positive and Negative Example Solutions](https://arxiv.org/abs/2610.13640) | 本文提出一个约束获取工具，能从输入数据和多个问题实例的正负示例解中自动学习特定尺寸的约束模型并将其泛化为尺寸无关的模型，生成可在未见数据上运行的MiniZinc代码。 |
| [^33] | [Model-agnostic Knowledge Augmented Machine Learning for Weakly Annotated Bioimaging Data](https://arxiv.org/abs/2610.13637) | 提出一种模型无关的知识增强方法，通过注入领域知识显著提升弱标注生物成像数据的分类性能，与自监督学习结合时在多个数据集上实现最高6.8%的精确率提升。 |
| [^34] | [Humanize Olympiad Agents: A Multi-Session Workflow for Olympiad Problem Solving with Solve, Review, Wiki, Verify](https://arxiv.org/abs/2610.13632) | 提出 HOA，一个开放的流程级框架，通过独立评审、结构化维基与验证三个互补反馈通道协调多个智能体会话，以解决单一会话智能体在奥赛难题上易传播错误、缺乏知识且可能曲解任务的脆弱性问题。 |
| [^35] | [SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle](https://arxiv.org/abs/2610.13626) | 该论文提出了SWE-DeceptionBench基准，系统评估大语言模型在软件开发生命周期六个阶段中的主动欺骗与被动欺骗行为，并通过中立、奖励、压力三种条件考察外部诱导对欺骗行为的影响。 |
| [^36] | [AudioTETRIS: Token Elimination via Transitional Importance Signals for Efficient Audio LLMs](https://arxiv.org/abs/2610.13600) | 提出AudioTETRIS，一个无需训练的两阶段token压缩框架，利用音频编码器的表示几何作为声学先验，识别声学过渡处信息密度峰值的关键token，在保留细粒度语音过渡信息的同时大幅降低音频大语言模型的推理计算开销。 |
| [^37] | [MENTOR: Learning from Failure with On-Policy Pedagogical Intervention](https://arxiv.org/abs/2610.13594) | 提出了MENTOR多智能体框架，通过在线策略式教学干预，实时诊断失败轨迹的根本原因并动态生成前瞻性引导，从而解决大推理模型训练中的“零奖励崩溃”问题。 |
| [^38] | [Constructing Pareto Frontiers of Optimal Reasoning Termination Methods for Overthinking in Reasoning Models](https://arxiv.org/abs/2610.13570) | 该论文提出逐步截断探测（STP）方法来量化推理模型的过度思考程度，并在统一框架下评估动态路由、固定预算和动态提前退出等终止策略，构建了准确率与效率之间权衡的帕累托前沿。 |
| [^39] | [Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning](https://arxiv.org/abs/2610.13536) | 提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。 |
| [^40] | [MAGMA-Bench: Evaluating Long-Horizon Embodied Agents under Dynamic Tasks and Execution Failures](https://arxiv.org/abs/2610.13534) | 提出了MAGMA-Bench——一个架构无关的基准，通过递增的任务时间跨度和受控在线扰动（如执行失败、任务中断、任务更新）评估动态多机器人环境中长时程具身智能体的持续任务推理与异构机器人协调能力，并发现即使最强的模型也难以应对这些挑战。 |
| [^41] | [Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2610.13521) | 该论文提出Road MOPD方法，将未参与路由的教师模型作为辅助跨域批评者，通过词元级信用分配与梯度更新协调多教师监督，从而解决多教师在线策略蒸馏中的梯度冲突和能力退化问题。 |
| [^42] | [Causal Effect Identifiability in Causal Abstractions: A Strict Hierarchy of Criteria](https://arxiv.org/abs/2610.13520) | 本文在因果抽象所定义的图集合层面上研究因果效应可识别性，建立了各可识别性标准之间的严格层次结构，并证明一个效应可能在图集合上可识别，却不存在对所有兼容图一致成立的单一因果推导。 |
| [^43] | [RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors](https://arxiv.org/abs/2610.13516) | 提出了RESCUE反事实评估框架，通过在执行检查点恢复模拟器和策略状态来评估VLA运行时监控器的警报是否能带来有效干预，并发现可恢复性取决于状态-干预对而非仅取决于机器人状态本身。 |
| [^44] | [Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis](https://arxiv.org/abs/2610.13510) | 该论文提出了PRISM，一种通过双模态人群记忆库在表示空间中检索相似病例嵌入、并融合异构临床证据的检索增强多模态语言模型，在包含1,443个任务的MDS-ED临床预后基准上，于诊断、病情恶化、ICU入院和死亡率预测等多项任务中取得了优异表现。 |
| [^45] | [Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning](https://arxiv.org/abs/2610.13504) | 该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。 |
| [^46] | [Humanize MLE Agents: Multi-Agent Session Alternation Beyond Single-Agent Limits](https://arxiv.org/abs/2610.13494) | 提出将多个智能体的会话交替协作建模为上下文重置半马尔可夫决策过程（CR-SMDP），通过工作流层面的跨会话协作突破单智能体的性能极限，从而克服长时程减速与框架过时问题。 |
| [^47] | [Branching On-Policy Distillation](https://arxiv.org/abs/2610.13481) | BranchOPD通过学生不确定性和师生分歧选择信息丰富的决策点进行分支重采样，对主轨迹和分支同时施加token级教师监督，并借助共享前缀复用将生成token成本降低约19%-31%。 |
| [^48] | [A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance](https://arxiv.org/abs/2610.13479) | 该论文提出了一种基于笔记本电脑的本地智能体AI框架，利用本地语言模型通过自主的“思考-行动-观察-更新-继续”循环自适应生成长尾边缘案例场景，用于自主系统的安全保证，同时兼具低成本、完全可控和数据隐私安全的优势。 |
| [^49] | [When MLLMs miss what humans read: diagnosing and learning to read squint text](https://arxiv.org/abs/2610.13465) | 该论文发现多模态大模型存在“眯眼文字”识别缺陷，通过构建 SquintBench 与 SquintCtrl 两个基准诊断出空间选择是关键瓶颈，并据此提出 Squint OPSD 方法教会模型阅读此类隐藏文字。 |
| [^50] | [SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability](https://arxiv.org/abs/2610.13461) | SqueezeBench提出了一种抗饱和的数学能力评估基准，要求AI不断提出并用Lean机器验证证明整数函数的上下界记录，使每次成功都提高门槛而非消除问题。 |
| [^51] | [Agentic Large Language Models](https://arxiv.org/abs/2610.13455) | 本书探讨了智能体化大型语言模型如何将LLM的语言推理能力与现实世界中的行动和交互能力相结合。 |
| [^52] | [Collective Autonomous Vehicle Routing Needs Social Awareness for Coexistence with Human Drivers](https://arxiv.org/abs/2610.13439) | 在自动驾驶车辆的奖励中引入衡量其对他车影响的边际成本，能让混合交通中的人类驾驶员与自动驾驶车辆双方都获益，从而实现更和谐的人机共存。 |
| [^53] | [Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits](https://arxiv.org/abs/2610.13438) | 提出展开时干预（RTI）技术，在轨迹采样阶段利用奖励侧证据（评估量规）将错误轨迹转化为成功轨迹，为超出模型能力边界的任务提供有效学习信号，突破智能体强化学习的同策略推理瓶颈。 |
| [^54] | [Local Diagnostics of Learned Event Representations for Anomaly Detection at the HL-LHC](https://arxiv.org/abs/2610.13437) | 该论文在高亮度LHC条件下模拟的7.5亿事件数据集COLLIDE-2V上，评估了监督与自监督神经嵌入策略，并以学习潜空间中自编码器的重构误差作为异常评分，强调必须对被标记事件进行局部诊断，以区分新物理与建模误差或探测器效应，并为后续定向搜索识别可观测量。 |
| [^55] | [CNN-Transformer with Sigmoid Label for High-Resolution Sub-Degree DOA Estimation](https://arxiv.org/abs/2610.13433) | 提出将信源DOA编码为角度偏移单调sigmoid函数之和的sigmoid标签，配合CNN-Transformer网络和0.25度网格步长，实现了亚度级精度和1度分辨率，突破了基于分类的DOA估计的网格量化误差限制。 |
| [^56] | [Web$hell: A Unified Interface for Agentic Web Access](https://arxiv.org/abs/2610.13416) | 提出Web$hell——一个统一搜索增强、浏览器与文件系统等多种访问模式的智能体网页访问接口，使智能体能够在单条任务轨迹内同时完成信息收集、浏览器操作和证据回溯。 |
| [^57] | [In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation](https://arxiv.org/abs/2610.13411) | 本文提出一种统一的概率化上下文学习元辨识架构，将基于Transformer的编码器-解码器与长序列处理机制相结合，实现长时域校准预测，并通过基准测试验证了其有效性。 |
| [^58] | [RegulaTutor: A Theory-Informed Framework for Long-Term Personalized Learning](https://arxiv.org/abs/2610.13399) | 提出基于自我调节学习理论的RegulaTutor框架，通过CAMM（认知、情感、元认知、动机）机制将学习者历史证据转化为有据可依的教学策略并随新观察持续更新，实现长程个性化辅导。 |
| [^59] | [ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents](https://arxiv.org/abs/2610.13395) | ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。 |
| [^60] | [AgentSWE: Can Coding Agents Build the Agent You Actually Want?](https://arxiv.org/abs/2610.13361) | 提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。 |
| [^61] | [SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories](https://arxiv.org/abs/2610.13356) | 提出了SWE-Critix后训练流水线，用于训练能够可靠评判和解释编码智能体在长时程软件工程任务中执行轨迹的通用验证器。 |
| [^62] | [AlphaNIST: Emergent Assembly Skills from Binary Task Rewards](https://arxiv.org/abs/2610.13354) | 受AlphaGo Zero启发，该研究仅用稀疏的二元任务奖励从零开始训练单一策略，在仿真中无需演示或密集奖励即可完成NIST装配任务板1号全部20个装配任务，达到98.2%的训练成功率。 |
| [^63] | [Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning](https://arxiv.org/abs/2610.13330) | 本文提出首个基于掩码扩散语言模型的化学推理智能体ChemDiffAgent，将化学推理建模为对推理与工具使用轨迹的迭代去噪过程，使工具使用决策能够在全轨迹上下文中联合确定，突破了传统自回归范式在双向分子理解和多源证据整合上的局限。 |
| [^64] | [Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans](https://arxiv.org/abs/2610.13303) | 该论文提出一种无标签的评分标准澄清程序，通过将裁判间分歧转化为对争议子标准的重写修复，显著提升了LLM评审团内部的一致性，但发现这种共识的提升并不能转化为与人类评判更高的一致性。 |
| [^65] | [PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real](https://arxiv.org/abs/2610.13296) | PRIME提出了一个索引侧优化框架，通过LLM驱动的语义重构与确定性的元数据合成及分层检索两条互补轨道，提升结构化文档检索质量并降低嵌入计算成本。 |
| [^66] | [BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation](https://arxiv.org/abs/2610.13285) | BoundaryEvo提出一种两阶段框架，先帮助学生模型建立基础推理能力，再使训练数据随学生模型的能力边界协同演化，从而在仅有少量教师轨迹的条件下高效提升小模型的长链推理能力。 |
| [^67] | [Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis](https://arxiv.org/abs/2610.13273) | Agent4Spec提出了一个自主多智能体框架，通过验证流水线自动检测并纠正OpenAPI语法错误，同时通过市场分析流水线开展竞争研究、识别与行业领先者的功能差距并自动生成缺失的API规范功能，实现了API规范的持续优化与演进。 |
| [^68] | [Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents](https://arxiv.org/abs/2610.13267) | 本文提出受神经科学启发的无需训练记忆修复框架DREAM，通过对被投毒记忆进行选择性功能性遗忘（动态再激活、异常诊断与记忆修复三阶段），实现LLM智能体对记忆投毒攻击的内在高效防御。 |
| [^69] | [Tabular Foundation Models Are Effectively Shallow](https://arxiv.org/abs/2610.13253) | 表格基础模型（TFMs）实际上是“浅层”的——其 transformer 块中高达 96% 可以被一个闭式、无需标签的仿射转换器替换而不损失性能，说明这类模型存在大量架构冗余，可以被大幅简化。 |
| [^70] | [How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing](https://arxiv.org/abs/2610.13236) | 该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。 |
| [^71] | [ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation](https://arxiv.org/abs/2610.13225) | 提出ReSFlow模型，通过在统计流形上校正类别流使其轨迹逼近测地线，实现离散数据的一步生成，大幅降低推理计算成本。 |
| [^72] | [Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models](https://arxiv.org/abs/2610.13212) | 该论文提出动作-未来耦合效果学习，让视觉-语言-动作模型以未来物理效果而非动作本身作为主要预测目标，从而提升机器人策略的性能，在DexJoCo上达到50.97%的总体成功率。 |
| [^73] | [Does Learning Recurrent Networks Matter in Partially Observable RL?](https://arxiv.org/abs/2610.13197) | 该研究发现冻结（无需训练）的循环网络即可为部分可观测强化学习提供数百步的记忆能力，并在多样化的POMDP任务中取得与训练循环网络相当的性能，从而质疑了在部分可观测RL中学习循环网络的必要性。 |
| [^74] | [JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models](https://arxiv.org/abs/2610.13194) | JANUS提出了一种由共同演化的世界模型驱动的“预测-验证-演化”循环框架，通过在执行前预测候选方案的价值提升、信息增益和探索效用来缓解长时程自主研究中的反馈延迟问题，并将反馈经验蒸馏为不断演化的技能。 |
| [^75] | [AdaC-GRPO: Towards Mastering Long-Horizon Visual Sudoku Puzzles via Adaptive Curriculum Learning](https://arxiv.org/abs/2610.13184) | 提出AdaC-GRPO方法，通过在线rollout准确率追踪模型的“最近发展区”来动态构建训练批次，从而克服长程视觉数独任务中因奖励稀疏导致标准RLVR失效的问题。 |
| [^76] | [RoleWorld: Knowledge Transfer in World Modeling with Functional Role Abstractions](https://arxiv.org/abs/2610.13172) | RoleWorld提出一种以功能角色抽象（可复用原型与环境特定参数值）为核心组织世界动力学知识的符号世界建模方法，使世界模型能够高效迁移到具有相似交互结构的新环境中。 |
| [^77] | [Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories for Supervised Fine-Tuning](https://arxiv.org/abs/2610.13152) | 提出TQS轨迹质量评分框架，直接基于轨迹结构评估过程质量，无需LLM评判且线性时间运行，用于筛选高质量编程智能体轨迹以提升监督微调效果。 |
| [^78] | [VIGOR: Verifiable Industrial Geometry Optimization and Repair in Scene Programs](https://arxiv.org/abs/2610.13145) | 该论文识别了大语言模型生成的场景程序中“可执行但不工程可行”的鸿沟，提出VIGOR框架结合基于执行的验证与分阶段几何修复，并构建了包含1,749个场景和每场景94项几何检查的工业场景程序数据集，实现工业布局的可验证优化与自动修复。 |
| [^79] | [Competence-Aware Orchestration for LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2610.13144) | 该论文提出一种两阶段能力感知编排框架，通过建模大语言模型骨干在当前查询上的能力，自适应地调整多智能体系统中的角色激活与通信图，在六个基准测试上取得平均93.81%的成绩。 |
| [^80] | [DP-MIND: Differentially private memory-guided Long context Inference](https://arxiv.org/abs/2610.13132) | 提出DP-MIND框架，通过本地扰动输入并利用能保留任务相关信息的紧凑记忆机制逐块处理序列，将差分隐私推理成功扩展到百万token级别的长上下文场景。 |
| [^81] | [Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories](https://arxiv.org/abs/2610.13131) | 提出了一种新的图约束生成范式“预期性生成建模”，将路网轨迹生成从位置模拟转向行为建模，通过预期未来路线来解决路口转向选择问题，避免轨迹偏离路网。 |
| [^82] | [Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA](https://arxiv.org/abs/2610.13128) | 提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。 |
| [^83] | [DRYLAB-Bench: Benchmarking the Biosecurity Risk of Agentic LLM Mutation Design](https://arxiv.org/abs/2610.13116) | 提出DRYLAB-Bench，首个用于评估LLM智能体高危生物突变设计能力的综合基准，将评估从知识问答转向实际应用，以量化LLM结合生物设计工具后带来的生物安全风险。 |
| [^84] | [Continually Post-training Small Models via Synthetic Self-distillation](https://arxiv.org/abs/2610.13113) | 该论文提出合成自蒸馏方法，通过让略大的同族对齐模型生成同分布的合成示范，并结合离线策略监督与拒绝采样，使缺乏强上下文学习能力的小模型也能在持续后训练中学习新能力的同时保留已有知识。 |
| [^85] | [Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination](https://arxiv.org/abs/2610.13112) | 该研究提出一种黑盒因果审计方法，通过CLAP相似度定位并用谱门控移除音频证据区域，发现定位消融在控制时长后稳定优于随机移除，但其幻觉检测效果存在模型依赖性，且先前空白波形对比法的表面优势实为时长混淆所致。 |
| [^86] | [VETR: Execution-Grounded Self-Improving Agent for Symbolic Graphics Reasoning](https://arxiv.org/abs/2610.13098) | VETR提出了一种以执行轨迹为依据的自我改进智能体框架，通过追踪程序执行中符号编辑的保留、消失与干扰情况，将执行证据转化为受约束的备选程序，从而实现符号图形推理的递归自我改进。 |
| [^87] | [When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows](https://arxiv.org/abs/2610.13092) | 本研究通过在ALFWorld和AndroidWorld上对五个智能体进行4,500次任务执行的实证分析发现，引入Jev虽能显著降低决策成本与延迟，但这种局部效率提升并不必然转化为更好的任务表现——具身任务的成功率接近基线，而两个移动智能体的成功率却明显下降。 |
| [^88] | [SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty](https://arxiv.org/abs/2610.13081) | SWEAR提出了一种难度感知的音频推理框架，通过五维难度评分和强化学习在推理各阶段自适应分配推理资源，在大幅减少生成token的同时提升准确率，简单样本推理更快而不损失困难样本的性能。 |
| [^89] | [Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs](https://arxiv.org/abs/2610.13078) | 该研究揭示了LLM安全机制中的分工路径：注意力模块（尤其早期层）主要负责“看见”有害信息，MLP模块主要负责“写出”拒绝输出，且后训练会增强MLP的拒绝写入能力但不会提升有害性识别能力。 |
| [^90] | [LongHarness: Advancing Agentic Reasoning over 100M-Token Corpora via Stateful Evidence Management](https://arxiv.org/abs/2610.13075) | LongHarness通过显式的“证据-结论协同演化”机制管理证据状态，使智能体能够对超出上下文窗口的1亿Token级语料库进行长上下文推理，有效避免无界探索和证据状态漂移两大失败模式。 |
| [^91] | [Evolving Creatives with Clicks: CTR-Driven Advertising image Editing via Attribute Disentanglement and Preference Alignment](https://arxiv.org/abs/2610.13066) | 该论文提出ECC框架，将广告创意优化从整体生成转变为细粒度编辑，通过ANOVA和随机森林回归识别出28个与点击率显著相关的属性，实现属性解耦与偏好对齐，从而选择性地优化损害点击率的属性并保留有益属性。 |
| [^92] | [Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents](https://arxiv.org/abs/2610.13058) | 提出ReTRAIL框架，使智能体能够将环境回退至较早状态进行回顾性探索，同时保留已学知识与有效进展，并通过状态锚定GRPO和分支推进奖励两种互补信号训练策略，帮助长时程智能体主动从死胡同轨迹中恢复。 |
| [^93] | [CoBe: A Benchmark For Conversational Counterfactual Text Editing](https://arxiv.org/abs/2610.13057) | 该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。 |
| [^94] | [GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs](https://arxiv.org/abs/2610.13035) | 提出GeoGR两阶段框架，通过将跨视角观察压缩为3D体素令牌（GeoSemZip）并在LLM层间进行令牌分组路由，联合压缩场景表示与视觉计算，从而实现高效的3D视觉语言模型推理。 |
| [^95] | [Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development](https://arxiv.org/abs/2610.13022) | 该论文提出ReqClarifyBench，一个包含341个任务（涵盖游戏、小程序和网页开发）的交互式需求澄清基准，用于评估LLM通过交互将模糊用户意图转化为清晰需求的能力及其对后续代码生成的影响。 |
| [^96] | [EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents](https://arxiv.org/abs/2610.13021) | 提出了首个具身仿真持续学习基准 EmbodiedCLBench，以组合泛化为核心，评估自我进化智能体能否将基础任务中积累的经验迁移并重组以解决未见的高级任务。 |
| [^97] | [Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence](https://arxiv.org/abs/2610.13015) | 该论文提出广义近端策略优化（GPPO），基于带单调性能改进保证的广义 f-散度信赖域理论，从散度几何视角统一理解和设计大语言模型强化学习算法，克服了 PPO 固定截断机制无法自适应策略偏移的缺陷。 |
| [^98] | [Provenance-Aware Belief Updating in Human-AI Feedback Loops](https://arxiv.org/abs/2610.13008) | 该论文提出LQET训练目标，使语言模型能够依据证据来源而非报告数量进行信念更新，显著降低了重复转发信息造成的虚假佐证与置信度漂移问题。 |
| [^99] | [Scaling Empowered Agents](https://arxiv.org/abs/2610.13002) | 该论文提出了一种简单可扩展的赋能度最大化方法——将赋能度视为某状态下可执行技能的数量，引导智能体到达多种技能的良好起点状态——从而首次在大规模环境中实现了理论预测的赋能度涌现特性，如收集资源、改造环境、生存，以及在 Crafter 中建造房屋。 |
| [^100] | [Hindsight Structured Advice Distillation: Steering Executor Agents with Advisors Trained on Their Failures and Task Characteristics](https://arxiv.org/abs/2610.13000) | 提出后见之明结构化建议蒸馏方法，训练一个小型顾问模型分析冻结执行智能体的失败案例与任务特征，为每个输入动态生成结构化建议并附加到其提示词中，从而以低成本弥补专业任务中细节知识的缺失。 |
| [^101] | [Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification](https://arxiv.org/abs/2610.12997) | 双曲空间嵌入能显著提升跨分类代码的现有技术检索能力——双曲VAE比欧几里得VAE多检索到52%的真正跨类别现有技术，尽管它与官方专利分类系统的一致性更低。 |
| [^102] | [What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training](https://arxiv.org/abs/2610.12995) | 提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。 |
| [^103] | [Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction](https://arxiv.org/abs/2610.12984) | 该论文提出Work2Skill框架，通过WorkRecorder同步采集屏幕录制、键鼠事件与无障碍树信息，构建了首个人类桌面GUI演示多模态数据集WorkDemo-350，以从人类操作经验中归纳出可复用的智能体技能。 |
| [^104] | [The Evidence Anchoring Boundary: Where Visual Interventions Matter and When They Help in Multimodal LLMs](https://arxiv.org/abs/2610.12981) | 该研究发现了多模态大语言模型中的“证据锚定边界”（EAB），证明视觉干预在该边界之前层级的干预效果是边界之后的13.78至26.10倍，为确定视觉干预的有效层级和时机提供了新依据。 |
| [^105] | [Beyond Semantic IDs: Finite-Beam Reachability and Continuous-Query Generation in Generative Recommendation](https://arxiv.org/abs/2610.12977) | 该论文提出两种连续查询生成范式CQG-Single和CQG-AR，通过去除离散语义ID词表与束搜索机制，消除了生成式推荐中的有限束可达性约束，使目录中所有物品都能被直接评分与检索。 |
| [^106] | [SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents](https://arxiv.org/abs/2610.12975) | SmartSalesBench是首个专为评估LLM销售代理应对具有心理抗拒性买家而设计的多轮对话基准测试，通过13,680个会话和三层级九项评估指标，颠覆了传统对话基准假设用户合作的评估范式。 |
| [^107] | [MotionCoach: Constructing Multimodal Guidance for Video Professional Motion Editing](https://arxiv.org/abs/2610.12963) | 提出MotionCoach——一个通过多轮工具调用与自我验证来生成接地提示词和参考片段的多模态智能体，将外部专业动作知识注入视频编辑模型，从而实现细粒度的专业动作编辑。 |
| [^108] | [JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents](https://arxiv.org/abs/2610.12957) | 本文提出JoyInside-VLM-Agent，一个通过产品条件化轨迹构建、闭环沙盒数据生成与分阶段策略学习（含新颖的DPER强化学习方法）训练的ReAct式多模态具身智能体，能够将多模态观察与用户请求转化为可执行动作，并适应产品约束、目标变化与执行反馈。 |
| [^109] | [MathVis: Can AI Reason through Drawing? A Benchmark for Mathematical Figure Generation](https://arxiv.org/abs/2610.12954) | MathVis 基准通过 512 个数学图形生成任务评估了大语言模型的可视化与图表辅助推理能力，发现即使最先进的模型 pass@1 也仅约 31%，且其输出偏重形式语言和符号而非显式的图示推理，生成的图形对人类而言往往视觉混乱。 |
| [^110] | [Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding](https://arxiv.org/abs/2610.12939) | 该论文提出了一个基于矩阵熵信息量分析与中心化核对齐（CKA）多样性度量的原则性框架，用于为大语言模型推测解码选择多层隐藏状态，并揭示了解码器表示中的几何与信息分区特性。 |
| [^111] | [Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training](https://arxiv.org/abs/2610.12933) | 该论文提出一种不确定性引导的渐进收缩框架，利用蒙特卡洛推理树揭示推理过程中的不确定性，将推理监督数据分两阶段（教师感知与学习者感知）逐步压缩，在大幅降低训练成本的同时保持复杂推理能力。 |
| [^112] | [Collaborative Learning and Reasoning over Model Slices](https://arxiv.org/abs/2610.12927) | 该论文提出“协同学习”框架，通过模型切片在同等参数预算下用多个异构小模型协作替代单一大模型，并借助金字塔监督优化跨模型推理与交互决策，从而解决单模型推理中错误传播和盲点难以重评的问题。 |
| [^113] | [Teaching neural decoders with language](https://arxiv.org/abs/2610.12921) | 提出“语言作为监督”方法，将神经特征与自然语言类别描述对齐以替代传统类别标签监督，在视觉、运动和语音解码任务中显著提升准确率，并支持候选集复用、受试者与会话自适应及未见频率的泛化解码。 |
| [^114] | [AI-Mediation in Retail Markets: Revenue Effects and Assortment Optimization](https://arxiv.org/abs/2610.12917) | 该研究通过构建AI产品排序数据集并建立Plackett-Luce随机中介模型，证明在零售商品组合优化中纳入AI中介效应可将平均收入损失降低2.52%至11.37%，并显著改变最优商品组合的选择。 |
| [^115] | [DeFiRank: Federated Pairwise Ranking for Governance in Decentralized Financial Networks](https://arxiv.org/abs/2610.12916) | DeFiRank提出了一种基于联邦学习的成对排序框架，使多个机构无需共享私有交易数据即可协同训练模型，通过捕捉交易对手多样性、多应用使用和交易频率等全网行为模式来评估提供者行为的真实性，从而实现去中心化金融网络中自动化且可扩展的治理。 |
| [^116] | [OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models](https://arxiv.org/abs/2610.12914) | 提出OGPO框架，通过语义关键状态重标注和目标条件化的过程奖励，将VLA模型后训练从逐步模仿专家动作转向可恢复的任务进展状态优化。 |
| [^117] | [Mind the Spatial Gap: A Benchmark for Geospatial Reasoning and Constrained Spatial Editing](https://arxiv.org/abs/2610.12881) | 该论文提出了GeoSpatialBench，一个基于六个城市真实城市数据、在GeoJSON与地图图像两种模态上评估28种空间关系的新基准，用于系统考察多模态大模型从空间关系理解、几何推理到满足几何约束的空间编辑的递进式空间认知能力。 |
| [^118] | [One-Step Text Generation by Seq-Drifting](https://arxiv.org/abs/2610.12879) | 该论文提出Seq-Drifting框架，通过构建具有上下文自适应目标的漂移场（结合吸引力与排斥力机制），使模型直接学习从高斯噪声到文本的映射，实现单次前向传播即可生成完整序列的文本。 |
| [^119] | [Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses](https://arxiv.org/abs/2610.12872) | 该论文提出协同心智理论（CToM）框架，通过训练大语言模型预测用户对草稿回复的响应，并利用该预测在回复前进行内部自我修订，同时结合GRPO优化，从而提升模型在多轮交互中与用户协作的能力。 |
| [^120] | [Localized Generative Design of Ship Hulls with Diffusion Models and High-Fidelity Assessment](https://arxiv.org/abs/2610.12870) | 该论文提出一种奖励导向的扩散模型框架，通过参数化掩码实现球鼻艏和船体前部等局部区域的生成式优化，并结合经验证的高保真自动化CFD流程进行评估，发现生成的船体在低速时阻力与基准相当，但在高速时阻力高出45-50%。 |
| [^121] | [Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization](https://arxiv.org/abs/2610.12855) | 该论文发现在线策略自蒸馏（OPSD）会导致大语言模型已获得能力的逐渐退化，其根源在于隐藏表示几何结构的恶化，并提出表示几何正则化（RGR）方法，通过保持簇内紧凑性和簇间分离性来缓解自我改进过程中的能力退化问题。 |
| [^122] | [Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests](https://arxiv.org/abs/2610.12849) | 该研究通过涵盖26,999个拉取请求的大规模因果分析，首次证明AI智能体同时修改应用代码与部署基础设施的“双范围”拉取请求会引入额外的集成开销，揭示了当前智能体缺乏部署感知能力的关键局限。 |
| [^123] | [When Generative Models Act as Simulators](https://arxiv.org/abs/2610.12847) | 该论文研究生成模型通过“可编辑性”充当模拟器的条件，提出两种在模型潜在状态与模拟器状态之间学习映射的方法（探针解码加引导、逆映射），并发现逆映射方法即使在引导失败时也能成功。 |
| [^124] | [Towards Defining, Measuring, and Leveraging Creativity in AI Applications](https://arxiv.org/abs/2610.12845) | 本文基于心理学创造力理论（尤其是Boden的新颖性、价值与惊奇三大准则）提出了创造力的形式化公设与数学定义，并在AI应用中对其加以验证。 |
| [^125] | [Neural networks are more modular than single neurons suggest](https://arxiv.org/abs/2610.12842) | 该论文提出了一个形式化的嵌套模块化层次结构（单元级、正交级和线性级），并通过统一的因果评估框架证明，即使神经网络在单个神经元层面缺乏模块化，它们在子空间层面往往仍表现出明显的功能模块化。 |
| [^126] | [COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering](https://arxiv.org/abs/2610.12837) | 提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。 |
| [^127] | [Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems](https://arxiv.org/abs/2610.12822) | 该论文首次揭示奖励破解并非单智能体的孤立失败，而是一种可在多智能体编码系统中通过代码、共享记忆和自然语言计划传播的“传染性”策略，为多智能体系统的安全性提出了全新挑战。 |
| [^128] | [LeWAM: End-to-End World and Action Modeling with JEPAs](https://arxiv.org/abs/2610.12818) | LeWAM提出了一种端到端的JEPA控制框架，通过联合训练潜在状态预测与动作流匹配并对潜在状态施加各向同性高斯约束，使单一模型同时支持策略执行与高效规划，将接触密集任务和长时程任务的成功率大幅提升，并实现32.3倍的规划加速。 |
| [^129] | [VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails](https://arxiv.org/abs/2610.12813) | 提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。 |
| [^130] | [Efficient Audio-Visual Speech Enhancement via Reinforcement Learning-Based Neural Architecture Search](https://arxiv.org/abs/2610.12796) | 本文提出一种基于强化学习的神经架构搜索框架，通过联合优化语音增强性能与模型紧凑度，自动搜索出参数量约为基线九分之一（290万 vs 2200万）而SI-SNR性能仅降低0.7 dB的高效视听语音增强模型，适合资源受限的边缘设备部署。 |
| [^131] | [Meta-Reflective Evolution Effectively Improves Production AI Systems](https://arxiv.org/abs/2610.12794) | 提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。 |
| [^132] | [EvoBelief: Falsifiable World Models for Self-Improving LLM Agents](https://arxiv.org/abs/2610.12789) | 提出EvoBelief，一种将环境知识表示为可证伪假设的自我演化世界模型，通过意外事件触发假设形成、独立验证后方可在支持范围内指导决策、并利用反例随环境变化修正或淘汰规则，使语言模型保持冻结的同时在部署中持续演化智能体的环境信念。 |
| [^133] | [Representation Alignment from a Few Paired Data via Distilled Optimal Transport](https://arxiv.org/abs/2610.12781) | 该论文提出一种基于蒸馏最优传输的方法，仅利用约100-500个配对锚点样本即可对齐两个独立预训练的单模态编码器：通过锚点相似度构建共同坐标系，用熵最优传输的边际约束对抗枢纽性问题，并将传输方案蒸馏到轻量级编码器中实现可归纳的跨模态对应。 |
| [^134] | [REDaCT: Collapse-Free Unlearning in Large Reasoning Models](https://arxiv.org/abs/2610.12769) | REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。 |
| [^135] | [Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries](https://arxiv.org/abs/2610.12768) | SemViews通过复用历史查询留下的语义视图标签、局部嵌入和辅助模型评分来减少昂贵的LLM预言机调用，同时以至少1−δ的概率保证任意工作负载下的精确率和召回率目标。 |
| [^136] | [Faithful and Complete LLM-Assisted Automated Construction of Hierarchical Hardware Knowledge Graphs](https://arxiv.org/abs/2610.12762) | 提出了一种LLM驱动的自动化流水线，通过分块属性提取、层级路由和迭代优化，将硬件数据手册忠实且完整地转换为保留来源可追溯性的层级式硬件知识图谱。 |
| [^137] | [Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data](https://arxiv.org/abs/2610.12747) | 该论文构建了首个针对道路施工区域的多模态驾驶数据集WZDrive-Data，并提出了多模态自车轨迹预测器WZDrive，有效解决了施工区域车道布局临时变化导致的车辆轨迹难以预测的难题。 |
| [^138] | [KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting](https://arxiv.org/abs/2610.12738) | 该论文提出KITE架构，通过可切换的协变量和注意力偏置两种机制将工程师掌握的语义先验知识注入自监督事件预测器，无需架构搜索即可适配各类先验，在标签稀缺场景下显著提升罕见事件预测性能（域内最高提升47%、跨域提升53%）。 |
| [^139] | [Unsupervised Multimodal Reward Modeling](https://arxiv.org/abs/2610.12707) | 提出一种无需人类偏好标签的无监督多模态奖励建模方法，通过在冻结的 Qwen3-VL 中间层训练线性头来区分网络图注与其扰动版本，仅用数千参数和约千分之一的微调计算量即可在多个基准上超越或持平 GPT-4o。 |
| [^140] | [GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks](https://arxiv.org/abs/2610.12669) | 本文提出了GeoFitBench-LLM基准，通过时空关系和语义锚定两个推理维度，以独立于数据提供者的标准化模式评估大语言模型从自然语言地理分析请求中推断所需数据集、支持资源和约束条件的能力。 |
| [^141] | [Recurrent Harness for Agentic Continual Learning](https://arxiv.org/abs/2610.12668) | 提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。 |
| [^142] | [Incremental Retrieval-Augmented Generation for Human-in-the-Loop Decision Support](https://arxiv.org/abs/2610.12666) | 论文提出了iRAG，一个知识库从空开始、通过将人类决策存储为可检索先例而逐步增量构建的人在回路决策支持系统，它从静默观察过渡到基于证据的质疑、再到带弃权的自主处理，并揭示了检索衰减带来的稳定性与可塑性权衡。 |
| [^143] | [MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis](https://arxiv.org/abs/2610.12659) | 该论文发现主流RLVR数学语料库中约88%的问题因过于简单或过难而无法为目标策略提供有效学习信号（即零优势陷阱），并提出MagicBeanStalk方法，通过递归数据合成生成既蕴含难题所需推理操作、又难度适中的有效训练问题。 |
| [^144] | [Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization](https://arxiv.org/abs/2610.12657) | 提出克罗内克积注意力（KPA），将键值激活表示为紧凑的词元相关矩阵因子的克罗内克积之和，严格推广了张量积注意力（TPA），在保持表达能力的同时实现了内存高效的KV缓存压缩。 |
| [^145] | [CLEAT: Learning Value-Guided Clarification for Interactive Agents](https://arxiv.org/abs/2610.12655) | 提出CLEAT框架，通过残差查询价值的估计，从共享交互结果中协同训练澄清价值控制器与任务策略，使LLM智能体学会判断何时提出澄清问题才能真正提升任务效用。 |
| [^146] | [Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller](https://arxiv.org/abs/2610.12644) | 该论文提出了一种名为“癌症式后门攻击”的新型后门攻击方法，利用不可见令牌和阶段控制器，使恶意行为在大语言模型中通过多个隐藏阶段逐步激活，从而渐进式地破坏模型的推理能力。 |
| [^147] | [Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building](https://arxiv.org/abs/2610.12638) | 该研究在一栋大学建筑上对统计、机器学习、循环神经网络和预训练TimesFM等日前电力预测模型进行了统一基准测试，发现带协变量的TimesFM 3.0总体误差最低（平均绝对误差较每周季节性朴素预测降低约43%），而XGBoost仍具竞争力。 |
| [^148] | [Mesh-Native Tokenisation: Retrofitting Foundation Models to Unstructured Data](https://arxiv.org/abs/2610.12618) | 本文提出网格原生卷积（MNC），一种局部感知的标记化方法，使基于卷积的物理基础模型能够高效处理非结构化网格数据，解决了传统卷积编码器无法适用于此类数据、而图神经网络方案训练昂贵且编码方式不兼容的问题。 |
| [^149] | [Self-Preservation in Language Models](https://arxiv.org/abs/2610.12602) | 该论文提出将任务描述及示例组织成任务库，让语言模型自行选择保留目标并生成重放数据，从而在持续学习中更广泛地保护自身原有能力。 |
| [^150] | [Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents](https://arxiv.org/abs/2610.12598) | 提出行为自我奖励（BSR）方法，让策略模型自我评判同一任务中相同验证结果轨迹间的调试行为差异，将其转化为有界辅助奖励，从而在二元可验证奖励之上引入更精细的监督信号，且无需单独的奖励模型或新的环境交互。 |
| [^151] | [SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire](https://arxiv.org/abs/2610.12596) | SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。 |
| [^152] | [Know What You Don't Know: Agentic Data Synthesis by Evolving Generation Space for Closed-Loop Learning](https://arxiv.org/abs/2610.12593) | EvoSyn 提出一个由学习反馈驱动的多智能体数据合成框架，通过可修订的生成空间状态将智能体的学习弱点直接转化为合成修订与资源分配，弥合了反馈与决策之间的鸿沟，从而实现闭环学习。 |
| [^153] | [AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits](https://arxiv.org/abs/2610.12545) | AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。 |
| [^154] | [Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study](https://arxiv.org/abs/2610.12542) | 在严格控制的等计算预算实验中，简单的随机重启列表调度方法在DAG工作流调度上比蒙特卡洛树搜索（MCTS）好2.88%，表明更复杂的搜索方法并不必然优于简单基线。 |
| [^155] | [EliteTable: Self-Improving Program Search for Executable Control Policies](https://arxiv.org/abs/2610.12536) | EliteTable 是一种由大语言模型引导、利用固定容量精英方案表来指导方案生成与改进的自我改进程序搜索算法，在六个 Gymnasium 控制任务上验证了其迭代提升可执行控制策略性能的能力。 |
| [^156] | [CircuitTree: Composing Circuits into A Syllable Transcriber](https://arxiv.org/abs/2610.12520) | 该论文提出CircuitTree数据结构及贪心密度缩放组合算法，将39个从HuBERT中提取的音素分类电路高效组合成单个模型，实现了任何单个电路都不具备的音节转写能力，以64.8%的准确率转写开音节，仅比完全不共享电路的上限低1.2个百分点。 |
| [^157] | [Energy-Efficient Spiking Neural Networks for Sensing-Aided Beam Prediction](https://arxiv.org/abs/2610.12507) | 该论文为毫米波传感辅助波束预测开发了脉冲神经网络（SNN），发现在视觉和多模态输入下，SNN能在相近能耗下取得比人工神经网络更高的精度，或在同等精度下将能耗降低多达一个数量级。 |
| [^158] | [Kairos: Toward a Byte-Level Foundation Model for Every Modality](https://arxiv.org/abs/2610.12495) | Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。 |
| [^159] | [Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance](https://arxiv.org/abs/2610.12493) | 本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。 |
| [^160] | [AI-Based On-Board Maritime Object Detection for Earth Observation Payload Data Reduction on Versal Embedded Hardware](https://arxiv.org/abs/2610.12182) | 该论文将量化后的YOLOX-S船只检测器部署于Versal VC1902嵌入式硬件上，实现星载智能筛选下传内容，从而根据数据内容有效缩减高分辨率对地观测卫星的载荷数据量。 |
| [^161] | [InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers](https://arxiv.org/abs/2610.12023) | 提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。 |
| [^162] | [MultiWorldBench: Do Independently Controlled Views Describe One Shared World?](https://arxiv.org/abs/2610.11723) | 该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。 |
| [^163] | [Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting](https://arxiv.org/abs/2610.11567) | 提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。 |
| [^164] | [Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper](https://arxiv.org/abs/2610.11542) | 该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。 |
| [^165] | [Cognition-Oriented Emotion Tracing from Causes to Consequences in Real-World Social Scenes](https://arxiv.org/abs/2610.11410) | 该论文提出面向认知的TRACE框架，通过条件-情感-效应三阶段建模，并构建包含3,746个问答对和646个视频的TRACE-Bench基准，用于评估多模态模型在真实社交场景中从原因到后果的全链路情绪理解能力，发现人机之间存在显著的性能差距。 |
| [^166] | [Estimating great expectations under autoregressive language models with potentials](https://arxiv.org/abs/2610.11399) | 本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。 |
| [^167] | [TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models](https://arxiv.org/abs/2610.11392) | 该论文提出了TypedBench基准测试，通过七种策略标注生成器和九个评估套件，以改写中位数范围和有限样本噪声底限为参照，系统评估系统一类型化决策模型的校准质量、措辞敏感性与决策成本。 |
| [^168] | [Rethinking Contrastive Loss in CLIP Post-training: A Complementary Framework with Frozen Text Encoder](https://arxiv.org/abs/2610.11374) | 论文发现CLIP后训练中的灾难性遗忘主要源于对比温度τ取值不当而非负样本不足，并据此提出冻结文本编码器的轻量级单轮后训练框架ComCLIP，证明在足够小的温度下对比后训练能提升而非损害预训练CLIP的性能。 |
| [^169] | [Higher-Order Action Supervision Makes A Strong Policy Class](https://arxiv.org/abs/2610.11175) | 本文提出同时监督零阶和一阶动作并辅以正式理论保证的损失方案，可显著提升模仿学习和强化学习策略的性能与控制鲁棒性，解决其在实际部署中的不稳定性问题。 |
| [^170] | [LinSlot: Exploiting Linear Representation hypothesis for unsupervised attribute discovery from slot based object representation](https://arxiv.org/abs/2610.10722) | 本文提出LinSlot方法，利用线性表示假设（LRH）构建连接图像、对象槽和属性块的概率模型，实现了无监督的属性发现与对象-属性解耦表示学习。 |
| [^171] | [PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs](https://arxiv.org/abs/2610.10455) | 提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。 |
| [^172] | [Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models](https://arxiv.org/abs/2610.10405) | 该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。 |
| [^173] | [Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution](https://arxiv.org/abs/2610.09772) | 该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。 |
| [^174] | [From Expert-Guided Proof Search to Automated Open-Problem Solving](https://arxiv.org/abs/2610.09769) | 研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。 |
| [^175] | [Learning to Report Unsafe Tasks in a Multi-Agent Game](https://arxiv.org/abs/2610.09002) | 该论文揭示了多智能体共享策略下“沉默陷阱”的成因——共享策略的梯度会把沉默的收益重复计入 k 次，使团队局部收敛于沉默——并给出了一个审计惩罚条件，使策略梯度更新能够可靠地收敛到普遍上报不安全任务（在多数情形下既充分又必要）。 |
| [^176] | [Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization](https://arxiv.org/abs/2610.08969) | 该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。 |
| [^177] | [OTel: Open Telco AI Datasets, Benchmarks, and Models](https://arxiv.org/abs/2610.07766) | OTel发布了一个统一的开放电信AI资源，提供面向检索、重排序、指令微调及安全/拒答的电信数据集，以及采用开放训练方案后训练的30个基线模型（10个嵌入模型、3个重排序器、17个语言模型）。 |
| [^178] | [TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning](https://arxiv.org/abs/2610.07043) | 提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。 |
| [^179] | [From Benchmark to Bench: Can Agents Survive Real-World Drug Discovery?](https://arxiv.org/abs/2610.06411) | 该研究开发了开源模块化药物研发智能体MAGI，并在三家制药公司的九项真实先导化合物优化项目中验证，发现无论由大语言模型直接生成分子还是委托REINVENT 4生成，项目成败的关键在于预测模型的优劣，而非分子生成途径的选择。 |
| [^180] | [GraphDecisionBench: Benchmarking System One Models on Graph Tasks](https://arxiv.org/abs/2610.06354) | 提出与模型无关的图决策基准GraphDecisionBench（GraphDecide），通过结构化任务画像、图-文本输入对照与启发式提案控制，系统评估Jev等系统一模型在图任务上的表现，揭示出邻接识别准确不等于整体结构正确、联合图-文本输入未必提升预测、可行解构造也未必带来高质量解。 |
| [^181] | [DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate](https://arxiv.org/abs/2610.05615) | 提出DREAM框架，通过零样本探测为每个智能体动态分配不同的图像分辨率，并利用ANLL不确定性量化和自适应阈值来缓解多模态多智能体辩论中的群体思维问题，从而提升多模态推理能力。 |
| [^182] | [Green-Routed Neural Operators: Physics Determines Where the Network Reads](https://arxiv.org/abs/2610.05337) | 提出格林路由神经算子（GRNO），利用控制方程决定潜在特征的采样位置，在五个PDE系统中的四个上取得最低的自回归预测误差。 |
| [^183] | [Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos](https://arxiv.org/abs/2610.04432) | 该论文提出了 Video2World 基准，将“从具身视频端到端自动构建交互式模拟环境”形式化为软件工程任务，利用 222 个来自机器人和人类演示视频的重建实例，从几何保真度、动态保真度和功能正确性三方面评估前沿基础模型与编码智能体的自主世界建模能力。 |
| [^184] | [AgroGround: Multi-Granularity Grounded Recognition in Agriculture](https://arxiv.org/abs/2610.04425) | 提出大规模农业定位识别数据集 AgroGround，通过自动化流程将八个农业视觉问答数据集的标签转换为病害病斑与完整目标的区域标注（共794,850条指令样本），并利用健康图像作为负向监督，微调视觉-语言模型实现植物病害等目标的识别与定位。 |
| [^185] | [Solving VeriContest with a Lean-Backed Rust Verifier](https://arxiv.org/abs/2610.03994) | 该论文提出由Lean 4支持的Rust验证器Rust-Prover，将VeriContest基准中全部1007道题目的1325个定理完全证明，其中1259个定理在不到32小时内以中位3.2分钟、1.17美元的成本完成，大幅超越了此前最强模型13.95%的首试通过率。 |
| [^186] | [GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive](https://arxiv.org/abs/2610.03861) | GOTT提出了一种基于单一可复用跨本体接触获取原语的灵巧操作框架，能够将来自规划、外部模型或人类演示的操作意图可靠地转化为多指机器人的稳定接触与物体运动跟踪，且无需针对不同任务或机器人本体进行改动。 |
| [^187] | [Evolving Hybrid Quantum-Classical Architectures for Image Classification](https://arxiv.org/abs/2610.03220) | 该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。 |
| [^188] | [Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination](https://arxiv.org/abs/2610.02626) | 提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。 |
| [^189] | [Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents](https://arxiv.org/abs/2610.01892) | 提出基于选择的结构化推理框架SSR，将开放式推理生成转化为从预定义推理候选中的似然选择，并借助共享上下文KV缓存实现并行评分，从而大幅提升小模型多模态搜索智能体的推理效率。 |
| [^190] | [Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models](https://arxiv.org/abs/2610.00661) | 提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。 |
| [^191] | [Useful to Whom? Sample Value Is Defined Only Relative to the Learner](https://arxiv.org/abs/2610.00221) | 该论文通过冻结样本子集、只改变学习器的受控实验证明，核心集选择中“易样本优先”与“几何覆盖”准则之间的交叉边界并非由数据固定，而是随学习器容量（如ResNet-18宽度、输入网格分辨率）而移动，说明样本的价值只有相对于特定学习器才有意义。 |
| [^192] | [Scale and Selection: What Makes Automatic Harness Evolution Work for Visual-Interface Robot Agents](https://arxiv.org/abs/2609.39304) | 该研究表明，用另一个编码智能体自动优化机器人智能体的执行框架（提示词、工具与控制规则）时，每轮参与评估的训练 rollout 数量是成败关键——将其从 5 个扩大到 100 个，可通过提升每次采纳决策的信噪比，把留出集成功率从 47% 提高到 67%。 |
| [^193] | [Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment](https://arxiv.org/abs/2609.38972) | 本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。 |
| [^194] | [OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime](https://arxiv.org/abs/2609.38345) | 提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。 |
| [^195] | [EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?](https://arxiv.org/abs/2609.37686) | 该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。 |
| [^196] | [FACT: Fidelity-Aware Construction of Articulated Twins](https://arxiv.org/abs/2609.37067) | FACT提出了一种智能体框架，通过证据—诊断—修正循环渐进式构建铰接孪生体，同时提升其几何、接触和动力学三个层面的保真度。 |
| [^197] | [MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution](https://arxiv.org/abs/2609.35664) | 提出多尺度门控线性注意力（MS-GLA），通过将注意力头分配到多个时间分辨率并用可学习的输入依赖融合层动态重组各头组输出，解决GLA Transformer在单一时间分辨率下的表示瓶颈，在不增加每头状态大小的情况下扩展有效记忆容量。 |
| [^198] | [Graph World Models for Constrained Epidemic Policy Planning](https://arxiv.org/abs/2609.35545) | EpiMind是一个用于受限疫情政策规划的图世界模型框架，通过图因子化循环状态空间模型和图-时间ADMM实现跨区域协调干预，在保证共享资源预算可行性的同时将入院预测误差降低29%。 |
| [^199] | [Hyper Algorithm Design Agent: Evolving Learnable Optimizer from Zero](https://arxiv.org/abs/2609.35328) | 该论文提出将元黑盒优化（MetaBBO）的设计过程建模为编码任务，通过双智能体框架利用先进编码智能体的递归自我改进能力，从零开始自动演化可学习优化器，从而消除人工逐案定制设计所带来的主观性限制。 |
| [^200] | [ThinkNet: Evolutionary Deep Ensembles for Efficient Motor-Imagery EEG Classification](https://arxiv.org/abs/2609.33967) | ThinkNet提出了一种验证控制的框架，通过仅训练数据归一化、验证引导的进化搜索和验证门控推理，在严格防止测试受试者信息泄露的前提下，实现面向新用户的高效、紧凑的运动想象脑电分类。 |
| [^201] | [Claim-Gated Source-Risk Auditing for Generative Search](https://arxiv.org/abs/2609.29145) | 该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。 |
| [^202] | [JEV-as-a-Judge: Accept When Confident, Escalate When Unsure](https://arxiv.org/abs/2609.26550) | 提出仅返回标签概率的低成本裁判JEV，通过置信度阈值机制将高置信判定直接接受、不确定判定升级至推理型裁判，从而在费用仅为GPT-6的41%的情况下，实现了比GPT-6高出0.9个百分点的评估准确率。 |
| [^203] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^204] | [Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies](https://arxiv.org/abs/2609.19844) | 论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。 |
| [^205] | [IBBench-Light: A Paired Evaluation of Task-Conditioned Responses to External Directives](https://arxiv.org/abs/2609.13725) | 该论文提出IBBench-Light配对评估基准，证明仅看边际平均准确率会掩盖模型对同一外部记录在“执行”与“处理”两类任务上的表现差异，且仅改变EOS设置即可使精确配对成功率从0/144骤升至62/144。 |
| [^206] | [The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation](https://arxiv.org/abs/2609.11489) | 该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。 |
| [^207] | [What Makes Adversarial Examples Transfer Across Deepfake Detectors?](https://arxiv.org/abs/2609.10002) | 该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。 |
| [^208] | [API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces](https://arxiv.org/abs/2609.08861) | 研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。 |
| [^209] | [SUN: Reaching for Novelty in Reinforcement Learning](https://arxiv.org/abs/2609.08642) | 本文提出基于后继价值函数的SUN指标，可同时衡量目标的新颖性与可达性，该可达性感知的目标选择框架能无缝集成到任意离策略强化学习算法中，并具有恢复计数奖励、约束命中概率和拒绝不可达目标的理论保证。 |
| [^210] | [SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents](https://arxiv.org/abs/2609.08149) | 该论文发现 SWE-Bench Pro 基准存在答案泄露导致的奖励破解和任务质量问题，并推出了经过验证的 SWE-Bench Pro Verified 版本，通过防泄露保护与任务修正实现更可靠的软件工程智能体评估。 |
| [^211] | [TDDN: Text-aligned Diffused DINO Network for Puzzle Understanding](https://arxiv.org/abs/2609.07937) | 提出TDDN模型，通过融合DINOv3与CleanDIFT表示并与RoBERTa-L进行文本对齐，在仅使用约59万个对齐数据对且冻结骨干网络的情况下，恢复了CLIP所缺乏的细粒度视觉感知能力，在图文检索和密集预测分割任务上大幅超越CLIP，从而提升了图像谜题理解能力。 |
| [^212] | [Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families](https://arxiv.org/abs/2609.07199) | 该论文发现基于特征的硬件木马检测性能严重依赖测试集划分协议——当测试门电路来自训练中已见过的宿主电路时性能被高估，而按宿主家族留出时 F1/AP 会从约 0.91/0.98 骤降至约 0.46/0.58。 |
| [^213] | [What Does Multi-Harness RL Learn? Credit Assignment and Portability in Coding Agents](https://arxiv.org/abs/2609.04518) | 该研究发现，在多框架强化学习训练编码智能体时，评估框架是决定性能的主导变量（使平均解决率相差4.3倍），远超训练配方本身的影响。 |
| [^214] | [Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents](https://arxiv.org/abs/2609.01487) | 提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。 |
| [^215] | [H2Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning](https://arxiv.org/abs/2609.01216) | 提出H2Table框架，将复杂表格表示为层次嵌套超图，通过定制超图编码器建模表头与单元格间的语义关系，并利用可学习查询向量将结构嵌入注入大语言模型，从而显著提升复杂表格推理能力。 |
| [^216] | [Data-Driven Persona-Conditioned Agents for A/B Test Simulation](https://arxiv.org/abs/2609.01038) | 该论文提出利用基于真实用户匿名行为数据（活动模式、参与度信号、人口统计特征）构建的数据驱动角色来条件化LLM智能体，从而更忠实地模拟用户群体并预测A/B测试结果，替代昂贵耗时的真实用户实验。 |
| [^217] | [DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving](https://arxiv.org/abs/2608.30386) | 提出衰减感知状态压缩（DASC），通过从模型权重中推导不同头和通道的“保留视野”来压缩混合线性注意力模型的持久状态检查点，从而缓解内存压力、减少缓存驱逐和重复预填充。 |
| [^218] | [Can escalation channels redirect reward hacking toward defect disclosure?](https://arxiv.org/abs/2608.29460) | 该论文提出在代码智能体遭遇测试基础设施缺陷时为其提供结构化上报工具（上报渠道），可将其奖励黑客行为从23.6%降至5.3%，同时暴露出触发该行为的底层缺陷。 |
| [^219] | [Preference Elicitation for Policy Optimization and Application to Aligning Heart Transplantation with Human Values](https://arxiv.org/abs/2608.28620) | 提出一种两阶段的线性效用偏好引出算法，通过成对比较学习切割平面并可证明收敛到用户效用函数，从而直接从分配结果学习效用函数用于策略优化，并成功应用于心脏移植分配的价值对齐问题。 |
| [^220] | [GenCoord: Skill-Path Commitments under Private Information](https://arxiv.org/abs/2608.22055) | GenCoord提出了一种在私有信息条件下，通过本地模型生成技能路径承诺并利用有界反馈机制，将任务后果转化为可执行行动的协作方法，从而解决具身智能体间的信息不对称问题。 |
| [^221] | [From Mastery Profile to Simulated Response: Stochastic Student Knowledge Graphs (SSKG) for Faithful LLM Student Simulation](https://arxiv.org/abs/2608.21668) | 本文提出基于随机学生知识图谱（SSKG）的方法，以解决大语言模型在模拟低掌握水平学生时因内置能力导致难以区分掌握程度差异的问题，从而实现更忠实的学生模拟。 |
| [^222] | [Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance](https://arxiv.org/abs/2608.10434) | 该研究提出一种由大语言模型驱动的对话式可解释AI界面，并通过受控实验证明其在无人机入侵检测的事后审计任务中，相比传统XAI仪表盘能更好地提升操作员的理解、信任与依赖。 |
| [^223] | [STAIR: Effective Incident Response Using an End-to-End Agentic Planning Framework](https://arxiv.org/abs/2608.09524) | STAIR提出了一个围绕Graph-as-State共享事件状态组织的端到端智能体响应框架，通过统一的决策依据在长时间跨度的多阶段安全事件响应中有效维持事件状态并对齐不断演进的恢复目标。 |
| [^224] | [Different Feedback, Different Updates: Selective Self-Learning from User Interactions for Large Language Models](https://arxiv.org/abs/2608.09109) | 该论文提出SLIFT框架，将用户反馈按任务相关性分解为Fix、Spec、Null三类原子组件，并通过通才和专才两个互补的LoRA适配器分别在不同泛化范围内整合相应改变，实现大语言模型从用户交互中的选择性自学习。 |
| [^225] | [ResidencyRL: Reinforcement Learning in Simulated Clinical Environments](https://arxiv.org/abs/2608.07418) | ResidencyRL提出了一种强化学习方法，通过模拟多轮临床诊疗环境（含对抗性患者模拟器和结构化诊断奖励）来训练临床AI智能体，使其像住院医师一样在实践互动中提升完整临床决策序列的能力。 |
| [^226] | [Fast LapSum: Exact Differentiable Top-$k$ at Million Scale](https://arxiv.org/abs/2608.06912) | Fast LapSum 通过概率区间限定与认证遍历将排序成本限制在阈值附近的窄区间内，实现了百万规模下保持精确选择质量且支持端到端微分的高效可微 top-k 求解。 |
| [^227] | [What Current AI Benchmarks Leave Unmeasured: Modality, Search, Citations, and Implications (for Safety Evaluations)](https://arxiv.org/abs/2608.06202) | 该研究通过比较ChatGPT聊天界面与OpenAI API两种模态在有无网络搜索条件下的表现（涵盖401个提示词、4,812个响应），揭示了当前LLM基准评估所忽视的关键维度——如响应一致性、引用依据和弃答行为——并指出这些盲区对安全评估具有重要启示。 |
| [^228] | [CASCADE: An Agentic Regulatory Network Framework for Patient-Data-Validated Downstream Perturbation Prediction](https://arxiv.org/abs/2608.05359) | CASCADE是一个通过MCP暴露的智能体调控网络框架，创新性地使用真实TCGA患者肿瘤数据（以局灶基因拷贝数扩增作为敲低的剂量代理）验证基因扰动预测的“方向准确性”而非“成员关系”，在MYC靶点预测中于三种癌症类型达到72%-90%的准确率，并在独立队列METABRIC中得到复现。 |
| [^229] | [Enhancing Tabular Learners with Context-Aware Semantic Embeddings](https://arxiv.org/abs/2608.03565) | 提出CASE框架，通过用代表性行样本预填充基于Gemma 3的表格语言模型的KV缓存来为表格行嵌入提供数据集级语义上下文，从而弥合大语言模型语义理解与表格学习器统计能力之间的差距，解决文本特征的语义歧义问题。 |
| [^230] | [First-Order In-Context Shapley: Scalable Data Auditing for LLM Alignment and Evaluation](https://arxiv.org/abs/2607.22766) | 该论文提出了一种仅需推理、无需重训练的一阶上下文内 Shapley 数据估值方法，可在 O(k) 次前向传播内完成对大语言模型对齐数据的可扩展审计，以 19 倍更少的计算量实现了与蒙特卡洛上下文内 Shapley 在全部目标样本上一致的估值效果。 |
| [^231] | [Learning from All-Failed Groups via First-Visit Coverage in Long-Horizon Agentic Tasks](https://arxiv.org/abs/2607.22724) | 提出CovGPO方法，通过在整组rollout全部失败时引入首次访问观测覆盖，为欠采样的状态改变动作提供学习信号，从而打破长程智能体任务中采样不平衡导致的自我延续信用陷阱。 |
| [^232] | [MIRROR: Learning from the Other View for Multi-Modal Reasoning](https://arxiv.org/abs/2607.21552) | 本文提出MIRROR强化学习方法，通过在同一问题的不同模态视角中选取模型表现最佳者作为教师，实现跨视角自蒸馏，从而提升多模态推理能力。 |
| [^233] | [Medical foundation models converge less under label supervision](https://arxiv.org/abs/2607.20274) | 该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。 |
| [^234] | [Robustness of Reinforcement Learning-Based Congestion Management in Low-Voltage Grids](https://arxiv.org/abs/2607.16004) | 本文提出一种将随机森林违规预分类器与actor-critic控制器相结合的解耦式强化学习阻塞管理方法，在真实低压电网中可将总违规幅度降低98.9%，并对测量噪声和电网参数失配表现出强鲁棒性。 |
| [^235] | [The One-Word Census: Answer-Choice Conformity Across 44 Language Models](https://arxiv.org/abs/2607.12796) | 本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。 |
| [^236] | [Who&When Pro: Can LLMs Really Attribute Failures in AI Agents?](https://arxiv.org/abs/2607.09996) | 提出大规模基准Who&When Pro，通过严格控制的失败注入流水线构建了12,326条带黄金标签的失败轨迹，系统评估并揭示了大语言模型在AI智能体失败归因任务中的表现规律与模式。 |
| [^237] | [Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents](https://arxiv.org/abs/2607.04528) | 该论文提出信念推演诊断方法，首次揭示执行框架会在任务、环境和基础LLM完全固定的情况下显著塑造多步LLM智能体对环境的信念，导致不同框架间出现从终端输出难以察觉的显著信念分歧。 |
| [^238] | [Agent Step Value: Auditing Evaluation-Channel Sensitivity in Black-Box Agent Traces](https://arxiv.org/abs/2607.04419) | 本文提出智能体步骤价值（ASV）指标，发现评估器衍生的步骤奖励在跨评估通道使用时高度不稳定（48.3%的比较中增益符号不一致），并将负向分量定位于生成视图通道。 |
| [^239] | [Improving Certified Robustness via Adversarial Distillation](https://arxiv.org/abs/2606.31653) | 本文提出了结合对抗蒸馏的认证训练方法AD-CERT，通过将对抗训练目标与IBP松弛界相结合，在标准准确率与认证鲁棒性之间取得更好的权衡。 |
| [^240] | [Hierarchical Experimentalist Agents](https://arxiv.org/abs/2606.29315) | 提出分层实验智能体（HExA），一种以实验为中心的上下文内自我改进框架，使智能体能够通过主动实验收集查询相关证据、发现未知环境的一般规律，并将交互经验转化为可复用技能。 |
| [^241] | [The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development](https://arxiv.org/abs/2606.27045) | 提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。 |
| [^242] | [Erased, but Not Gone: Output Forgetting Is Not True Forgetting](https://arxiv.org/abs/2606.25001) | 该论文提出以重训练模型群体作为参考分布的“重训练一致的表示遗忘”评估范式，揭示现有机器遗忘方法虽在输出层面看似遗忘成功，但在表示空间中仍与理想遗忘存在结构性偏差，即输出遗忘并不等于真正的遗忘。 |
| [^243] | [RoboMME-Interference: Benchmarking Robot Memory Under Interference](https://arxiv.org/abs/2606.22338) | 本文提出了RoboMME-Interference，一个跨会话机器人记忆基准，用于评估长上下文和干扰下VLA模型的记忆性能，并发现现有感知记忆变体在干扰下表现有限。 |
| [^244] | [Decoupling the Declarative from the Procedural in Vision-Language-Action Models](https://arxiv.org/abs/2606.21496) | 该论文指出现有视觉-语言-动作模型无法将陈述性知识（概念与实体语义）与程序性知识（操作方法）解耦，导致其对空间、语义和任务变化十分脆弱、难以泛化到训练物体之外，这是通用机器人技能迁移的根本瓶颈。 |
| [^245] | [TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability](https://arxiv.org/abs/2606.19821) | 提出TelcoAgent框架，通过自动构建3GPP知识图谱、零样本时间序列基础模型预测和领域可解释推理三个组件，实现了无需站点特定训练的准确、可扩展且可解释的5G多KPM预测。 |
| [^246] | [Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence](https://arxiv.org/abs/2606.19386) | 本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。 |
| [^247] | [Scalable Circuit Learning for Interpreting Large Language Models](https://arxiv.org/abs/2606.16939) | 提出基于稀疏线性回归的CircuitLasso方法，以极低的计算成本学习到与最先进干预方法结构准确性相当的电路，并能高效揭示人类可理解的SAE语义特征在大语言模型中的传播及其对预测的影响。 |
| [^248] | [PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents](https://arxiv.org/abs/2606.16215) | PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。 |
| [^249] | [Let Large Language Models Judge Each Other: Multi-Agent Peer-Reviewed Reasoning for Medical Question Answering](https://arxiv.org/abs/2606.15419) | 该论文提出多智能体同行评审推理方法，让多个大语言模型在医学问答中互相评审彼此的思维链推理并择优生成最终答案，显著提升了准确性、可解释性和鲁棒性。 |
| [^250] | [Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory](https://arxiv.org/abs/2606.09365) | 提出SkeMex框架，将医疗智能体的交互轨迹提炼为结构化的技能记忆，在不更新模型权重的情况下实现自我演化，从而支持可泛化的长程临床推理。 |
| [^251] | [Do More Agents Help? Evaluating Accuracy and Cost in LLM Multi-Agent Workflows](https://arxiv.org/abs/2606.05670) | 该研究提出共享评估框架BenchAgent，在十个基准上比较六种多智能体工作流与单智能体方案的准确率与token成本，发现大多数多智能体配置以更高的成本换来更差的总体表现，仅EvoAgent小幅胜出。 |
| [^252] | [Agents' Last Exam](https://arxiv.org/abs/2606.05405) | 本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。 |
| [^253] | [The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show](https://arxiv.org/abs/2606.05328) | 该论文通过逆转扩散采样过程恢复模型内部轨迹，首次证明视频扩散模型内部确实线性编码了物理合理性信息，其可从扩散Transformer状态中以约81.27%的准确率解码出来，超越了专用表征学习基线。 |
| [^254] | [Automatic-Differentiation-Enabled Finite Element Model Updating: Learning Constitutive Neural Networks from Structural Deformation Data](https://arxiv.org/abs/2606.05199) | 提出了ADE-FEMU框架，将本构神经网络嵌入JAX-FEM非线性求解器，通过梯度最小化测量失配损失，从动力学信息有限的结构变形数据中实现本构模型参数的自动学习。 |
| [^255] | [The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents](https://arxiv.org/abs/2606.04296) | 该研究以18维情感动力学引擎HEART作为诊断探针，揭示了“状态饱和陷阱”等机制，说明基于情感的触发器和大语言模型裁判无法准确把握中断自主智能体的干预时机。 |
| [^256] | [EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents](https://arxiv.org/abs/2606.03678) | EvoDrive是首个基于大语言模型智能体进化的多目标安全关键驾驶场景自动生成框架，通过仿真器支撑的执行者-评论者架构与帕累托进化，在保持场景真实性的同时最大化对抗性，突破了手工启发式方法的局限。 |
| [^257] | [When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation](https://arxiv.org/abs/2606.03532) | 本研究发现在自我在线策略蒸馏中，教师更新的隔离期（更新之间完全冻结教师）而非教师年龄是实现稳定学习的关键结构属性，并提出诊断框架揭示了状态遗忘崩溃现象——短程最优的固定更新时间表在长程训练中会因时钟驱动的刷新将漂移的学生一次性不可逆地复制进教师而导致灾难性失败。 |
| [^258] | [Latent Reward Steering: An Adaptive Inference-Time Framework that Implicitly Promotes Cognitive Behaviors in Reasoning LLMs](https://arxiv.org/abs/2606.00726) | 本文提出潜在奖励引导（LRS）框架，通过优化稀疏自编码器的潜在状态并在推理时利用奖励梯度进行自适应修正，隐式促进推理大语言模型中的认知行为，无需显式行为控制。 |
| [^259] | [MyoSem: Aligning Electromyography with Natural-Language Action Semantics](https://arxiv.org/abs/2606.00174) | MyoSem提出了一个肌电-动作语义对齐框架，将手部EMG信号与多视角自然语言动作描述映射到共享特征空间，通过语义原型构建、局部激活感知编码和语义查询聚合，实现了双向的EMG-文本检索，突破了传统固定手势标签分类的局限。 |
| [^260] | [VITAL: Visual-Semantic Dual Supervision for Enhanced and Interpretable Latent Reasoning in Medical MLLMs](https://arxiv.org/abs/2605.28422) | VITAL提出了一种带视觉-语义双重监督的医学多模态大语言模型潜在空间推理框架，通过辅助文本解码器与视觉投影器在推理时零开销地实现推理过程的文本与视觉双重可解释性。 |
| [^261] | [MPMMine: Better Benchmarks for Constraint Acquisition](https://arxiv.org/abs/2605.26279) | 本文提出了MPMMine基准测试套件，包含基准数据集、配套软件包和基准测试工具，用于对利用多样化领域知识工件来发现、验证和增强数学规划模型的约束获取算法进行一致、系统和全面的评估，填补了该领域基准测试基础设施的空白。 |
| [^262] | [EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation](https://arxiv.org/abs/2605.23954) | EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。 |
| [^263] | [Stochastic MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent](https://arxiv.org/abs/2605.21282) | SMFP是一种将MeanFlow噪声到动作映射与高斯重参数化相结合的单步生成式策略，通过解析熵下界实现熵正则化镜像下降强化学习，在七个MuJoCo基准上优于高斯策略和其他生成式策略。 |
| [^264] | [From SGD to Muon: Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781) | 该论文提出了一种基于Schatten-p范数的数据驱动准则，可动态地为神经网络各层自适应选择从SGD到Muon之间的最优更新几何结构，并在统一框架下恢复了SGD、Muon、Adam和MuAdam等主流优化器。 |
| [^265] | [Sampling-Based Safe Reinforcement Learning](https://arxiv.org/abs/2605.19469) | 该论文提出基于采样的安全强化学习算法SBSRL，通过在有限动力学样本上联合施加约束来近似最坏情况优化，并结合认知不确定性约束的探索策略，提供了学习全过程的高概率安全保证和有限时间样本复杂度界，在仿真与真实机器人上实现了安全高效的探索。 |
| [^266] | [When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents](https://arxiv.org/abs/2605.11928) | 该论文提出了基于真实部署失效案例构建的RobustBench-TC基准，用以揭示工具使用智能体在模拟与真实环境之间严重不均衡的鲁棒性差距，并给出了域随机化强化学习训练方案以提升其现实部署可靠性。 |
| [^267] | [MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading](https://arxiv.org/abs/2605.10268) | MemReread通过在最终记忆不充分时触发问题分解与针对性重读，恢复流式阅读中被过早丢弃的间接证据，从而在不依赖中间检索的情况下增强智能体的长上下文非线性推理能力，并结合强化学习提升长度外推能力和重读次数的自适应决策。 |
| [^268] | [MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs](https://arxiv.org/abs/2605.10064) | MAGE框架将自我知识外化为经验、任务搜索与技能路由协同进化的四子图知识图谱，使冻结骨干的语言模型智能体在自进化过程中既能决定学什么、又能跨迭代保留所学知识。 |
| [^269] | [FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](https://arxiv.org/abs/2605.09081) | FactoryNet是一个包含1.13亿数据点、覆盖7种机器形态和4项任务的大规模工业时间序列预训练数据集，通过统一的S-E-F-C控制理论模式使指令意图与实际响应可在不同机器间比较，为工业时间序列基础模型奠定了基础。 |
| [^270] | [UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence](https://arxiv.org/abs/2605.09040) | UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。 |
| [^271] | [Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph](https://arxiv.org/abs/2605.08037) | 本文提出 GraphDPO，将 DPO 从成对偏好比较推广到有向无环偏好图结构，充分利用多个 rollout 之间的传递性偏好关系进行聚合监督，从而实现更稳定、更充分的语言模型对齐。 |
| [^272] | [Convex Optimization of Spiking Neural Networks with Global and Approximation Guarantees](https://arxiv.org/abs/2605.08022) | 该论文将脉冲神经网络（包括LIF-SNN）的训练问题首次等价转化为凸优化问题，证明了零对偶间隙与全局最优保证，并提出具有高概率近似保证的随机化替代方法以解决凸规划变量指数级增长的问题。 |
| [^273] | [FactoryBench: Evaluating Industrial Machine Understanding](https://arxiv.org/abs/2605.07675) | FactoryBench基于珀尔因果阶梯构建了四个因果层级的问答基准，包含7万余个问答项，用于评估时间序列模型和大语言模型对工业机器人遥测数据的理解能力，零样本评测显示当前前沿大语言模型在结构化任务上不超过50%、在决策任务上不超过18%。 |
| [^274] | [Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning](https://arxiv.org/abs/2605.07251) | 提出ChemCost基准，利用化学品采购成本估算这一具有精确、无评判者真实标准的实际任务，首次对大语言模型智能体的科学工具使用能力（包括接地、检索、采购和算术）进行严格评估与分阶段诊断。 |
| [^275] | [LensVLM: Selective Context Expansion for Compressed Visual Representation of Text](https://arxiv.org/abs/2605.07019) | LensVLM提出了一种推理框架和后训练方案，使VLM能先扫描压缩的渲染文本图像，再通过学习到的工具选择性地将相关图像扩展为未压缩形式，从而在4.3倍有效压缩下保持与全文处理相当的准确率，并在最高10.1倍压缩下超越现有基线。 |
| [^276] | [IAD-Unify: Task-Specific Interfaces for Industrial Anomaly Understanding, Segmentation, and Generation](https://arxiv.org/abs/2604.12440) | 提出 IAD-Unify 统一架构，通过任务专用 token 接口连接多模态语言模型、密集视觉专家和扩散编辑器，首次将工业异常的解释、像素级分割与受控生成三种互补能力整合于同一框架。 |
| [^277] | [Physics-guided surrogate learning enables zero-shot control of turbulent wings](https://arxiv.org/abs/2604.09434) | 该研究利用壁面湍流的局部结构，通过物理引导的代理学习，将在槽道流中训练的强化学习策略直接零样本迁移部署到 NACA4412 机翼上，实现 11.1% 的总阻力降低，超越了最先进的对向控制方法（8.9%）。 |
| [^278] | [AnomalyAgent: Agentic Industrial Anomaly Synthesis via Tool-Augmented Reinforcement Learning](https://arxiv.org/abs/2604.07900) | 该论文提出AnomalyAgent异常合成智能体，通过配备提示词生成、图像生成、质量评估、知识检索和掩码生成五种工具实现闭环优化，并采用监督微调加强化学习的两阶段训练，生成高真实感且多样的工业异常样本，以缓解异常检测的数据稀缺问题。 |
| [^279] | [Mixture-Greedy for Online Generative Model Selection: Is UCB Necessary in Diversity-Aware Multi-Armed Bandits?](https://arxiv.org/abs/2603.21716) | 本文提出Mixture-Greedy策略，并发现在多样性感知的多臂老虎机生成模型选择问题中，UCB探索奖励并非必要——去除UCB反而收敛更快、样本效率更高、性能更好。 |
| [^280] | [DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models](https://arxiv.org/abs/2603.18048) | 提出DEAF基准，通过2,700多个冲突刺激和多层次受控评估框架，诊断音频多模态大语言模型是真正处理声学信号还是依赖文本推断，发现七个模型均呈现文本主导的一致模式。 |
| [^281] | [NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling](https://arxiv.org/abs/2603.16148) | 提出 NeuronSpark，一个无需自注意力的选择性脉冲状态空间语言模型，通过让每个神经元的衰减率、增益和发放阈值随输入词元内容自适应调整，解决了标准脉冲神经元统一发放敏感性所导致的密集或静默发放问题。 |
| [^282] | [AutoControl Arena: Synthesizing Executable Test Environments for Frontier AI Risk Evaluation](https://arxiv.org/abs/2603.07427) | 该论文提出AutoControl Arena框架，通过“逻辑-叙事解耦”原则将确定性状态交由可执行代码处理、生成性动态交由LLM承担，从而自动合成可执行测试环境以缓解幻觉并高效评估前沿AI风险，同时通过X-Bench揭示了前沿模型在压力下风险率从21.7%激增至54.5%的“对齐幻觉”现象。 |
| [^283] | [CORE-Seg: Reasoning-Driven Segmentation for Complex Lesions via Reinforcement Learning](https://arxiv.org/abs/2603.05911) | 该论文提出了首个面向推理驱动复杂病灶分割的思维链基准数据集ComLesion-14K，并构建了通过语义引导提示适配器将推理与分割相融合、采用SFT到GRPO渐进式强化学习训练的端到端框架CORE-Seg，推动医学图像分割从模式匹配向认知推理的范式转变。 |
| [^284] | [Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing](https://arxiv.org/abs/2603.03753) | 该论文提出了智能体对等网络这一新范式，设计了一种将连接性、语义发现与执行解耦的基于平面的参考架构，并引入带签名的软状态能力描述符，以支持边缘设备上客户端自主代理之间安全的能力共享与协作。 |
| [^285] | [ITO: Multi-View Alignment and Training-Time Fusion for Image-Text Pretraining](https://arxiv.org/abs/2603.02767) | ITO通过基于多视图增广的多模态多重对齐丰富监督信号，并引入仅在训练阶段使用的轻量级多模态融合模块作为正则化器（推理时丢弃），使图文预训练得到的表示更多按语义而非模态组织，同时保持双编码器架构的推理效率。 |
| [^286] | [On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective](https://arxiv.org/abs/2603.02095) | 该论文在双神经元ReLU网络的最简二分类设置中证明，梯度下降虽能收敛到最优对抗鲁棒性边际，但收敛速度极慢（严格为 Θ(1/ln(t))），并首次建立了非线性模型中鲁棒性边际收敛速度的显式下界。 |
| [^287] | [Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parametric Policies](https://arxiv.org/abs/2602.23811) | 该工作将离线强化学习的理论保证从有限小动作空间扩展到大/连续动作空间上的参数化策略类，识别出将镜像下降推广至参数化策略时的核心难点“上下文耦合”，并据此提出了计算上高效（oracle高效）的独立参数化离线策略优化算法。 |
| [^288] | [CoMMa: Contribution-Aware Medical Multi-Agents for Decentralized Oncology Decision Support](https://arxiv.org/abs/2602.09159) | CoMMa通过去中心化数据访问和贡献感知聚合机制，实现了隐私保护下的肿瘤学多智能体决策支持，并用确定性嵌入投影替代随机推理，提供可解释的信用分配。 |
| [^289] | [COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation](https://arxiv.org/abs/2602.03806) | COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。 |
| [^290] | [Preference-Conditioned Multi-Objective Reinforcement Learning for Tactical Decision Making in Heavy-Duty Vehicles](https://arxiv.org/abs/2601.18783) | 提出一种基于偏好条件的多目标强化学习框架，结合广义策略改进与近端策略优化，在安全性、能效和时间三个冲突目标之间逼近帕累托前沿，从而实现重型车辆的战术决策优化。 |
| [^291] | [Global Consistency Repair for Natural-Language Claim Sets](https://arxiv.org/abs/2601.13600) | 该论文提出了QXR框架，将大语言模型作为带噪声的集合级一致性判断器，通过定位最小冲突子集并删除一个小的命中集声明，实现对不一致自然语言声明集的全局一致性修复，从而保留最大的连贯子集。 |
| [^292] | [HAFixAgent: History-Aware Agentic Program Repair](https://arxiv.org/abs/2511.01047) | HAFixAgent通过将基于git blame归因的仓库历史上下文注入修复循环，显著提升了基于大语言模型的自动程序修复效果，对较弱模型的提升最高达64.3%，并大幅超越RepairAgent和BIRCH-feedback等现有最优智能体方法。 |
| [^293] | [A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)](https://arxiv.org/abs/2510.22809) | 提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。 |
| [^294] | [ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios](https://arxiv.org/abs/2510.10998) | 该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。 |
| [^295] | [Discovering Phase Space Structure in Learned Hamiltonian Systems](https://arxiv.org/abs/2509.23003) | 提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。 |
| [^296] | [Access Paths for Efficient Ordering with Large Language Models](https://arxiv.org/abs/2509.00303) | 提出了LLM ORDER BY语义算子及预算感知优化器，通过动态选择近优访问路径，在所有基准测试中实现了与最佳静态方法相当或更优的排序准确率。 |
| [^297] | [Towards a Measure Theory of Semantic Information](https://arxiv.org/abs/2508.00525) | 本文批判了Floridi未能解决Bar-Hillel-Carnap悖论的强语义信息理论，并提出了一种基于单位圆和von Neumann量子概率类比的新方法，成功构建了信息量的测度空间。 |
| [^298] | [DeltaLLM: Delta-Key Sparse Attention for Edge LLMs](https://arxiv.org/abs/2507.19608) | DeltaLLM是一个无需训练的边缘端LLM加速框架，通过算法-硬件-数据流协同设计，利用因果感知的K-only delta表示实现GQA下的Delta-Key跨查询头复用，并结合上下文感知的混合策略在对角线附近保留精确注意力，从而加速长提示词预填充阶段的注意力计算。 |
| [^299] | [From the Fluency Fallacy to the Micro-to-Macro Validity Gap: Opportunities and Pitfalls of LLMs in Social Simulation](https://arxiv.org/abs/2507.19364) | 本文批判性评述了大语言模型在社会模拟中的应用，提出“微观-宏观效度鸿沟”框架，指出LLMs在个体层面的幻觉与随机不一致等局限会在多智能体社会中被放大为系统性脆弱性。 |
| [^300] | [Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values](https://arxiv.org/abs/2506.11849) | 本文提出一种将蒙特卡洛采样与回归调整相结合的新方法来估计Shapley值等概率值，允许用任意可高效计算概率值的函数族（如XGBoost）替代线性回归，并通过交叉拟合实现无偏且性能领先的概率值估计。 |
| [^301] | [LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge](https://arxiv.org/abs/2506.09956) | 本文提出了LLMail-Inject，一个包含208,095个人工编写攻击提示的大规模基准测试，通过真实红队挑战收集而成，用于评估间接提示注入攻击的完整执行链，克服了现有小规模模板化基准饱和且过拟合的问题。 |
| [^302] | [UniConFlow: A Unified Constrained Flow-Matching Framework for Certified Motion Planning](https://arxiv.org/abs/2506.02955) | 本文提出UniConFlow，一个统一的约束流匹配轨迹生成框架，通过新颖的预设时间归零函数系统性地处理等式与不等式约束，实现可认证的机器人运动规划。 |
| [^303] | [Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion](https://arxiv.org/abs/2505.22193) | 本文提出两种受物理启发的量子扩散模型方案——利用量子随机行走中量子与经典动力学的相互作用降低生成图像的FID，以及仅用四个量子比特借助真实IBM量子硬件的固有噪声生成图像——为可扩展的量子生成式AI算法开辟了新路径。 |
| [^304] | [Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon](https://arxiv.org/abs/2505.22184) | 提出了C$^2$TU，一种无需训练、无需提示的新方法，利用中文同音字图谱和毒性词典，通过子串匹配识别候选毒性词并进行过滤纠正，有效破解社交媒体中以谐音伪装毒性词汇规避审查的内容。 |
| [^305] | [ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models](https://arxiv.org/abs/2505.17244) | 大型推理模型的思维链中间步骤中可能隐藏有害内容，导致现有审核工具性能大幅下降，本文提出ReasoningShield基准与轻量级模型，用于对CoT推理轨迹进行安全审核。 |
| [^306] | [Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal](https://arxiv.org/abs/2505.15738) | 该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。 |
| [^307] | [Heterogeneity-Aware Client Sampling for Optimal and Efficient Federated Learning](https://arxiv.org/abs/2505.11304) | 本文首次对一般异构联邦学习进行了统一理论分析，揭示了通信异构性与计算异构性各自导致目标不一致的不同机制，并据此提出异构感知的客户端采样策略，以实现最优且高效的联邦学习。 |
| [^308] | [WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care](https://arxiv.org/abs/2504.06185) | 该论文通过统一训练与评估流程，对通用视觉、医学影像及伤口挑战赛中的最先进分割模型进行了系统基准测试，评估了其在真实部署中的泛化性、效率与可解释性，并提出基于参考物体的伤口大小测量方法，以弥合前沿语义分割与实际伤口护理应用之间的差距。 |
| [^309] | [CRM: Retrieval Model with Controllable Condition](https://arxiv.org/abs/2412.13844) | 提出可控检索模型（CRM），将观看时长作为用户塔的条件输入，使模型在推理阶段无需重新训练即可通过指定期望观看时长来引导检索，同时保持物品向量可预先计算以支持高效的 ANN 检索。 |
| [^310] | [Understanding Transformer-based Vision Models through Inversion](https://arxiv.org/abs/2412.06534) | 本文提出一种新颖的模块化特征反演方法，能更高效地从大规模Transformer视觉模型的中间表示重建语义更连贯的图像，并借此揭示了DETR与ViT等模型内部机制随网络深度的差异。 |
| [^311] | [Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs](https://arxiv.org/abs/2411.02817) | 该论文提出了基于条件熵的条件Vendi和条件RKE两种新度量方法，能够在提示引导的生成任务中分离出模型自身引起的多样性，为生成式AI模型和大语言模型提供了提示感知的多样性评估工具。 |
| [^312] | [Provably Accurate Shapley Value Estimation via Leverage Score Sampling](https://arxiv.org/abs/2410.01917) | 提出Leverage SHAP，通过对Kernel SHAP的轻量级改进，利用杠杆分数采样将模型评估次数从指数级降至O(n log n)，实现了可证明精确的Shapley值估计。 |

# 详细

[^1]: 语言模型能玩维基百科游戏吗？

    Can Language Models Play the Wikipedia Game?

    [https://arxiv.org/abs/2610.13948](https://arxiv.org/abs/2610.13948)

    该论文提出以维基百科游戏作为信息检索基准，通过OpenAlex构建大规模本地维基百科页面图与论文引用图，在多语言和多学术数据集上系统比较句子嵌入模型与大语言模型的导航检索能力，发现没有任何模型能显著超越人类表现。

    

    语言模型日益用于支持或自动化从过于庞大而无法建立索引的文档集合中进行信息检索的任务。当该任务需要专业的语言学或科学知识时，难度会显著增加。我们使用维基百科游戏来测试语言模型在类似信息检索任务上的表现。在该游戏中，玩家必须仅通过文章之间的链接到达目标文章。我们在五种维基百科语言和五个学术文献数据集上比较了句子嵌入模型与大型语言模型（LLM）。我们利用OpenAlex构建了规模非常大的本地维基百科页面图和论文引用图，并提供了构建和导航这些图的代码。这使得我们无需依赖第三方API即可研究语言模型的导航能力。我们将这些模型与人类表现和TF-IDF基准进行了比较。维基百科游戏的形式能够在不同技术、语言和领域之间实现稳健的比较。没有模型显著超越人类，尽管表现最好的模……

    arXiv:2610.13948v1 Announce Type: new  Abstract: Language models increasingly support or automate information retrieval from document collections too large to index. This task is difficult when it requires specialist linguistic or scientific knowledge. We used the Wikipedia Game to test language models on an analogous information retrieval task. In the game, players must reach a target article using only article links. We compared sentence embedding models and LLMs across five Wikipedia languages and five academic-literature datasets. We construct very large local graphs of Wikipedia pages and of paper citations using OpenAlex and provide code to build and navigate them. This allows the study of language models' navigation capabilities without relying on third-party APIs. We compared the models with human and TF-IDF benchmarks. The Wikipedia Game format enables robust comparisons across techniques, languages, and domains. No model significantly outperformed humans, although the best mo
    
[^2]: 捉迷藏：隐藏概念以寻找黑盒模型的解释

    Hide and Seek: HIDE Concepts to SEEK explanations for black box models

    [https://arxiv.org/abs/2610.13935](https://arxiv.org/abs/2610.13935)

    提出了一种无需模型权重或概念标注数据集的黑盒视觉模型概念级归因方法HnS，通过视觉-语言模型检索概念相关图像、遮挡图像中的概念区域，并依据模型输出的变化来衡量该概念的归因贡献。

    

    解释机器学习模型的预测是一项核心挑战，尤其是对于黑盒视觉模型而言。基于概念的归因方法利用人类可理解的概念来解释模型。现有的基于概念的方法只能解释分类和回归模型，它们需要访问模型权重或大型的概念标注数据集，并且大多数方法无法提供图像级别的归因。我们提出了“捉迷藏”方法，这是一种新的概念级归因方法，无需访问模型权重或标注的概念数据集。HnS能够解释广泛的视觉模型。给定一个概念和一个图像集合，HnS使用视觉-语言模型检索与该概念对齐的图像，在每张图像中定位该概念，遮挡定位到的区域，并根据黑盒模型在原始图像与遮挡图像之间输出的变化来衡量归因。这通过直接测量……实现了忠实的归因。

    arXiv:2610.13935v1 Announce Type: cross  Abstract: Interpreting machine learning model predictions is a central challenge, particularly for black-box vision models. Concept-based attribution methods interpret models using human-understandable concepts. Existing concept-based methods can only interpret classification and regression models. They require access to model weights or large concept-annotated datasets, and most methods cannot provide image-level attribution. We propose Hide and Seek (HnS), a new method for concept-level attribution that does not require access to model weights or labeled concept datasets. HnS can interpret a wide range of vision models. Given a concept and image set, HnS retrieves concept-aligned images using a vision-language model, localizes the concept within each image, masks the localized region, and measures attribution from the change in the black-box model's output between the original and masked images. This enables faithful attribution by directly me
    
[^3]: 重新审视表格数据深度回归：回归目标函数与全局早停的局限

    Revisiting Tabular Deep Regression: Objectives and the Limits of Global Early Stopping

    [https://arxiv.org/abs/2610.13909](https://arxiv.org/abs/2610.13909)

    该论文系统性地重新评估了表格深度回归中的各类目标函数，发现分位数损失是唯一能相对调优后的MSE基线带来一致提升的目标函数，并进一步揭示了由于不同噪声水平的区域需要不同的训练时机，全局早停策略存在根本性局限。

    

    尽管面向表格数据的深度学习模型近来受到大量研究关注，但回归任务中目标函数的选择仍是一个研究不足的自由度。若干研究方向提出了标准均方误差（MSE）的替代方案，但由于数据集、骨干网络和评估协议的差异，其实际收益难以相互比较。在本工作中，我们通过在广泛收集的成熟表格数据集和多种神经骨干网络上对回归目标函数进行系统性重新评估来填补这一空白。我们的比较表明，许多已提出的目标函数相对经过恰当调参的MSE基线只能带来有限或不一致的收益，而分位数损失在所测试的各个骨干网络上均能相对MSE取得一致的总体提升。随后，我们识别出即使采用这种强目标函数仍然存在的一个局限：具有不同目标噪声水平的区域会在不同的时机达到t……

    arXiv:2610.13909v1 Announce Type: cross  Abstract: While deep learning models for tabular data have recently received significant research attention, the choice of objective functions for regression tasks remains an understudied degree of freedom. Several lines of research offer alternatives to the standard Mean Squared Error (MSE), but differences in datasets, backbones, and evaluation protocols make their practical benefits difficult to compare. In this work, we address this gap by performing a systematic re-evaluation of regression objectives across a broad collection of established tabular datasets and multiple neural backbones. Our comparison reveals that many proposed objectives provide limited or inconsistent gains over a properly tuned MSE baseline, while quantile loss yields consistent aggregate improvements over MSE across the tested backbones. We then identify a limitation that remains even with this strong objective: regions with different levels of target noise can reach t
    
[^4]: 任务上下文塑造语言模型组件的角色

    Task Context Shapes Component Roles in Language Models

    [https://arxiv.org/abs/2610.13895](https://arxiv.org/abs/2610.13895)

    该论文揭示了任务上下文会重新分配语言模型中共享注意力头的影响力，表明同一模型组件在不同任务中扮演着不同的功能性角色。

    

    大语言模型在不同任务中会复用相同的注意力头，但仅仅识别出这些共享的注意力头并不能解释它们在每个任务中究竟做了什么不同的事情。一个自然的灵感来源是共享脑区之间随任务变化的相互作用，这启发了将语言模型计算视为网络的观点。在本文中，我们使用相同的图事实（即指明存在哪些有向连接的陈述语句）来比较不同任务下固定的模型组件。我们测量了干预对注意力头及后续层激活状态的影响（影响力），以及对答案的影响（重要性）。研究发现，任务变化会在五个开源权重模型（涵盖四个模型系列、参数规模从15亿到140亿）中重新分配影响力。在Gemma-2-9B和Qwen3-14B中，对早期内部激活向量进行编辑，可以在书面指令保持不变的情况下切换模型所读取的事实。随后，相同的注意力头会响应不同的图事实……

    arXiv:2610.13895v1 Announce Type: cross  Abstract: Large language models reuse the same attention heads across tasks, but identifying these shared heads does not explain what they do differently in each task. A natural source of inspiration is task-dependent interactions among shared brain regions, which motivate a network view of language-model computation. Here, we compare fixed model components across tasks using identical graph facts, statements specifying which directed connections are present. We measure how interventions affect attention heads and the activation state in later layers (influence), as well as the answer (importance). Task changes redistribute the influence in five open-weight models spanning four families and 1.5-14 billion parameters. In Gemma-2-9B and Qwen3-14B, an activation edit changes an early internal activation vector and switches which fact the model reads while the written instruction stays fixed. The same heads then respond to different graph facts and 
    
[^5]: RATED：基于执行动态的可靠性感知AI智能体工具检索

    RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics

    [https://arxiv.org/abs/2610.13889](https://arxiv.org/abs/2610.13889)

    RATED通过冻结双编码器进行语义匹配，并用轻量级上下文老虎机模块根据实时执行反馈重排序候选工具，无需模型重训即可快速适应运行时工具故障，同时提出含5,172个真实工具的EcoAgent基准，实现52.2%的相对MRR提升。

    

    随着AI智能体日益依赖外部工具，准确的工具检索对于实现可靠行为至关重要。现有方法仅优化语义相关性，却在很大程度上忽略了执行期间工具的可用性。为填补这一空白，我们提出了RATED（基于执行动态的可靠性感知工具检索），这是一个将鲁棒语义匹配与动态可靠性自适应相结合的检索框架。RATED采用经对比微调的冻结双编码器进行语义匹配，并利用一个轻量级的上下文多臂老虎机模块，基于实时执行反馈对候选工具进行重排序。这一设计使系统能够快速适应运行时工具故障，而无需昂贵的模型重新训练。我们还引入了EcoAgent基准，包含5,172个真实世界工具和超过21万个查询-工具对，用于离线检索与动态可靠性评估。实验结果表明，RATED相比ToolLLM API实现了52.2%的相对MRR提升。

    arXiv:2610.13889v1 Announce Type: new  Abstract: As AI agents increasingly rely on external tools, accurate retrieval is essential for reliable behavior. Existing methods optimize semantic relevance but largely ignore tool availability during execution. To address this gap, we propose RATED (Reliability-Aware Tool Retrieval via Execution Dynamics), a retrieval framework that combines robust semantic matching with dynamic reliability adaptation. RATED employs a frozen dual-encoder with contrastive fine-tuning for semantic matching, and a lightweight contextual bandit module that re-ranks candidates based on real-time execution feedback. This design enables rapid adaptation to runtime tool failures without expensive model retraining. We introduce EcoAgent, a benchmark of 5,172 real-world tools and over 210,000 query-tool pairs for offline retrieval and dynamic reliability evaluation. Experimental results demonstrate that RATED achieves a 52.2% relative MRR improvement over the ToolLLM AP
    
[^6]: X-AgentRed：在不断演化的多服务生态系统中对个人智能体进行红队测试

    X-AgentRed: Red-Teaming Personal Agents in Evolving Multi-Service Ecosystems

    [https://arxiv.org/abs/2610.13886](https://arxiv.org/abs/2610.13886)

    提出X-AgentRed，一个自进化的红队测试框架，通过工具依赖图系统性地发现多服务生态系统中个人智能体的有害工具调用路径，克服了现有方法偏向熟悉工具组合、忽略新颖跨服务交互的局限。

    

    个人智能体日益通过使用工具连接到网络和应用程序中的各类服务。随着其行动的效果在各个服务之间累积并相互作用，智能体在完成良性的用户请求时可能会产生非预期或有害的结果。主动识别这些组合性失败需要系统地探索多样化的跨服务路径。现有的红队测试方法通常只是简单地提示大语言模型从任务和环境描述中识别潜在的有害路径，或者将人类或大语言模型指定的有害目标分解为看似良性的行动组合。然而，这种生成方式偏向于训练期间学习到的模式，可能倾向于使用熟悉的工具组合，导致新颖的跨服务交互未被充分探索。为解决这一问题，我们提出了X-AgentRed，这是一个自进化的红队测试框架，它通过工具依赖图来发现有害的工具调用路径，并扩展和……

    arXiv:2610.13886v1 Announce Type: new  Abstract: Personal agents increasingly connect to services across the web and applications by using tools. As the effects of their actions accumulate and interact across services, unintended or harmful outcomes can arise when agents fulfill benign user requests. Proactively identifying these compositional failures requires systematically exploring diverse cross-service paths. Existing red-teaming methods often simply prompt LLMs to identify potentially harmful paths from task and environment descriptions or decompose harmful objectives specified by humans or LLMs into seemingly benign action combinations. However, generation is biased toward patterns learned during training, potentially favoring familiar tool combinations and leaving novel cross-service interactions underexplored. To address this, we propose X-AgentRed, a self-evolving red-teaming framework that discovers harmful tool call paths through a graph of tool dependencies and expands and
    
[^7]: 使用人工智能大规模研究漫画：PAGE框架及美国漫画与日本漫画的三项研究

    Studying Comics at Scale with Artificial Intelligence: The PAGE Framework and Three Studies of American Comics and Japanese Manga

    [https://arxiv.org/abs/2610.13840](https://arxiv.org/abs/2610.13840)

    提出PAGE框架，利用人工智能将漫画扫描件自动解析为分格与文本，并借助视觉语言模型按编码手册进行标注和与人工标注对齐，实现了对3929本美国漫画和87卷日本漫画的大规模整体化研究。

    

    漫画是二十世纪最伟大的视觉记录之一：在数十万页的漫画中，大众读者看到了他们时代的领袖、战争与英雄。然而这一记录的大部分从未被作为一个整体加以研究，漫画学术研究通常一次只针对一本作品展开。我们认为，包括生成模型在内的人工智能可以改变这一现状，从而有益于漫画媒介本身。我们提出了PAGE框架，使人工智能成为漫画研究的一种工具：它将每一份扫描件解析为分格、文本和书目记录；使用遵循书面编码手册的视觉语言模型对每一页进行标注，如同人类编码员的工作方式；将这些解读的样本与人工标注进行对齐验证；并通过控制出版商和体裁等变量的估计来解释档案库中的变化。我们在3,929本美国漫画书（1938-1956年）和87卷日本漫画上对该框架进行了验证，开展了三项研究……

    arXiv:2610.13840v1 Announce Type: new  Abstract: Comics are one of the great visual records of the twentieth century: hundreds of thousands of pages in which a mass readership saw its leaders, its wars and its heroes. Most of this record has never been studied as a whole; comics scholarship proceeds one book at a time. We argue that artificial intelligence, including generative models, can change this to the benefit of the medium itself. We propose PAGE, a framework that makes artificial intelligence an instrument for comics research. It parses every scan into panels, text and a bibliographic record; annotates each page with a vision-language model that follows a written codebook, as a human coder would; grounds samples of these readings in human annotation; and explains change across an archive, with estimates controlled for publisher and genre. We demonstrate the framework on 3,929 American comic books (1938--1956) and 87 volumes of Japanese manga, in three studies, each resting on a
    
[^8]: SpatialASM：将主动空间心智化内化于空间推理

    SpatialASM: Internalizing Active Mentalization for Spatial Reasoning

    [https://arxiv.org/abs/2610.13832](https://arxiv.org/abs/2610.13832)

    提出SpatialASM框架，通过训练时的空间模拟与新视角生成，将主动空间心智化（ASM）能力内化到大型视觉语言模型中，显著提升其空间推理能力。

    

    在解决复杂的空间推理任务时，普通人会自然而然地从与任务相关的视角重新审视场景，并在头脑中重建这些视角下的空间布局。但大型视觉语言模型（LVLMs）能否主动地以同样的方式思考？我们将这种能力称为主动空间心智化，它是视觉智能的内在组成部分，但在当前的LVLMs中仍严重欠缺。为了弥补这一差距，我们提出了SpatialASM，这是一个可扩展的框架，通过训练时的空间模拟将ASM内化到LVLMs中。SpatialASM采用一种以他者为中心的新视角生成器，将假设的视角锚定在视觉证据之上，并产生两种互补形式的监督：i）显式的ASM轨迹，教导LVLMs在空间推理过程中何时以及如何自适应地调用视角规划与想象；ii）自动构建的视角采择任务，进一步强化……

    arXiv:2610.13832v1 Announce Type: cross  Abstract: When solving complex spatial reasoning tasks, ordinary humans naturally revisit the scenes from task-relevant viewpoints and mentally reconstruct the spatial layouts from those viewpoints. But can Large Vision-Language Models (LVLMs) proactively think in the same way? We term this ability Active Spatial Mentalization (ASM), an intrinsic component of visual intelligence that remains substantially underdeveloped in current LVLMs. To bridge this gap, we introduce SpatialASM, a scalable framework to internalize ASM into LVLMs through training-time spatial simulation. SpatialASM employs an allocentric novel-view generator to ground hypothetical viewpoints in visual evidence and produces two complementary forms of supervision: i) explicit ASM trajectories that teach LVLMs when and how to adaptively invoke view planning and imagination during spatial reasoning, and ii) automatically-constructed perspective-taking tasks that further strengthen
    
[^9]: MC-EvoBench：审计Minecraft智能体中经验驱动的自我改进

    MC-EvoBench: Auditing Experience-Driven Self-Improvement in Minecraft Agents

    [https://arxiv.org/abs/2610.13829](https://arxiv.org/abs/2610.13829)

    MC-EvoBench是一个受控的Minecraft基准，通过19条注册进化链、任务状态图与严格的源-目标边界控制，审计智能体能否将交互经验真正转化为可跨世界持久化并迁移到后续任务的可复用能力。

    

    具身递归自我改进的核心挑战在于使智能体能够将交互历史转化为超越单一世界持续存在、并可迁移到后续任务的能力。保存的记忆或成功的最终动作本身并不能证明获得了可复用的能力；评估必须将能力变化与遗留资源、不平等的源暴露以及早期执行失败区分开来。我们构建了MC-EvoBench，这是一个受控的Minecraft基准，包含固定的19条已注册进化链目录，其中包括12条主要评估链和7条校准或探测链，并配有三个与图对齐的配对探测族用于迁移评估。该基准将每个任务表示为任务状态图，定义了阶段级环境验证器，分配匹配的源角色与目标角色，重置目标世界，并仅允许一个已声明的智能体工件跨越源-目标边界。（原文摘要在此处截断）

    arXiv:2610.13829v1 Announce Type: cross  Abstract: A central challenge in embodied recursive self-improvement is enabling agents to convert interaction history into capabilities that persist beyond one world and transfer to later tasks. A saved memory or successful final action does not by itself establish that a reusable capability was acquired; evaluation must separate capability change from carried-over resources, unequal source exposure, and early execution failures. We construct MC-EvoBench as a controlled Minecraft benchmark with a fixed catalog of 19 registered evolution chains, including 12 primary evaluation chains and seven calibration or probe chains, together with three graph-aligned paired probe families for transfer. The benchmark represents each task as a task-state graph, defines stage-level environment verifiers, assigns matched source and target roles, resets the target world, and permits only a declared agent artifact to cross the source-target boundary. For each pro
    
[^10]: 相同的查询，不同的答案：语义查询处理中的运行间方差

    Same Query, Different Answer: Run-to-Run Variance in Semantic Query Processing

    [https://arxiv.org/abs/2610.13823](https://arxiv.org/abs/2610.13823)

    该研究首次系统测量了语义查询引擎多次执行间的结果方差，发现即使在温度为0时也有最高超过10%的行输出会改变，且查询计划会指数级放大这种不确定性，导致约三分之二的最终答案发生变化，说明基于少量执行的基准测试评估并不可靠。

    

    语义查询引擎让大型语言模型（LLM）来判断单个数据行，而基准测试通常在温度为0的设置下通过少量执行的质量来比较这些引擎。我们测量了再次执行会带来多大的结果变化。我们通过各自的测试框架重新执行了SemBench、LRO-Bench和UDA-Bench的文本与表格工作负载，记录了全部120万次模型调用，并重新分析了SemBench已公布的执行结果。在温度为0时，五次执行之间有0.07%至10.2%的行的算子输出会发生变化（取决于具体模型），在某一任务上甚至高达24.0%。正如可能世界模型所预测的，查询计划会随答案所暴露的行数呈指数级放大这些变化：在UDA-Bench上，只有百分之几的提取值发生变化，但约三分之二的答案会发生改变。在SemBench已公布的结果中，21%的文本结果在不同执行之间质量发生变化，其中包括计划为确定性的引擎。

    arXiv:2610.13823v1 Announce Type: cross  Abstract: Semantic query engines ask a large language model (LLM) to judge individual rows, and benchmarks compare these engines on the quality of a few executions at temperature 0. We measure how much another execution changes. We re-execute the text and tabular workloads of SemBench, LRO-Bench, and UDA-Bench through their own harnesses, log every one of 1.2 million model calls, and reanalyze SemBench's published executions. At temperature 0, between 0.07% and 10.2% of rows change their operator output over five executions, depending on the model, and up to 24.0% on one task. Plans amplify these changes exponentially in the number of rows an answer exposes, as a possible-worlds model predicts: on UDA-Bench, a few percent of extracted values change, but about two thirds of the answers do. In SemBench's published results, 21% of text results change quality between executions, including on engines whose plans are deterministic. Several per-query s
    
[^11]: 你的离散图扩散模型正在秘密记忆

    Your Discrete Graph Diffusion Model is Secretly Memorizing

    [https://arxiv.org/abs/2610.13786](https://arxiv.org/abs/2610.13786)

    本文首次证明离散图扩散模型存在“条件记忆化”现象——当反向扩散轨迹以损坏的训练图为条件时，模型几乎能完美恢复该训练图，而这种记忆化在常规推理和样本级评估指标中均无法被察觉。

    

    离散图扩散模型（GDMs）在马尔可夫噪声模型下，通过独立地向节点和边添加噪声来破坏图结构。在没有节点特征的无条件设置下，在单个图上训练的等变去噪器能够生成新颖的图，并且边重叠和WL核相似度等样本级指标未发现记忆化的证据。在这项工作中，我们挑战了这一观点，并首次证明图扩散模型确实表现出记忆化。通过分析在双块随机块模型（SBM）上训练的GDM的反向扩散轨迹，我们发现存在强烈的训练数据偏差：与保留的测试图相比，该模型几乎完美地恢复了被损坏的训练图。我们将这种现象称为“条件记忆化”，因为它仅在反向轨迹以损坏的训练图为条件时才会显现，而在常规推理过程中并不明显。它与不断增长的基因（原文在此处截断）

    arXiv:2610.13786v1 Announce Type: cross  Abstract: Discrete graph diffusion models (GDMs) corrupt the graph structure by adding noise independently to nodes and edges under a Markovian noise model. In the unconditional setting without node features, an equivariant denoiser trained on a single graph is able to generate novel graphs and sample-level metrics such as edge overlap and WL kernel similarity find no evidence of memorization. In this work, we challenge this view and show for the first time that graph diffusion models indeed exhibit memorization. Analyzing the reverse diffusion trajectories of GDMs trained on two-block stochastic block models (SBMs), we find a strong train-data bias: the model recovers a corrupted training graph almost perfectly compared to a held-out test graph. We term this conditional memorization, as it surfaces only when the reverse trajectory is conditioned on a corrupted training graph and is not apparent during inference. It coincides with a growing gene
    
[^12]: UkrMeme：乌克兰梗图中的多模态可解释说服检测

    UkrMeme: Multimodal Explainable Persuasion Detection in Ukrainian Memes

    [https://arxiv.org/abs/2610.13783](https://arxiv.org/abs/2610.13783)

    本文发布了包含200张乌克兰战时梗图的说服技术检测数据集UkrMeme及乌克兰语翻译基准PTM-UA，并评估了视觉-语言模型在零样本多模态设置下的说服检测性能及可解释性方法。

    

    梗图在现代网络交流中扮演着核心角色，尤其是在政治冲突的背景下，视觉与文本线索相结合以传递具有说服力的信息。我们使用视觉-语言模型（VLM）研究乌克兰梗图中的说服技术检测。我们介绍了UkrMeme——一个包含200张自然产生的乌克兰战时梗图并针对说服技术进行标注的数据集，以及PTM-UA——Propaganda 950基准的一个子集的乌克兰语翻译版本，该版本在翻译梗图文本的同时保留了原始视觉内容。我们在零样本提示设置下，使用分层F1指标，评估了视觉-语言模型在图像、文本和多模态输入上的表现。最后，我们比较了逐层相关性传播（LRP）与模型生成的词元归因，并分析了各种解释方法所强调证据的语义类别。

    arXiv:2610.13783v1 Announce Type: new  Abstract: Memes play a central role in modern online communication, especially in contexts of political conflict where visual and textual cues combine to convey persuasive messages. We study persuasive technique detection in Ukrainian memes using vision-language models (VLMs). We introduce UkrMeme, a dataset of 200 organically produced Ukrainian wartime memes annotated for persuasion techniques, and PTM-UA, a Ukrainian translation of a subset of the Propaganda 950 benchmark that preserves original visual content while translating meme text. We evaluate VLMs in a zero-shot prompting setup across image, text, and multimodal inputs using hierarchical F1 metrics. Finally, we compare Layer-wise Relevance Propagation (LRP) and model-generated token attributions and analyze the semantic categories of evidence emphasized by each explanation method.
    
[^13]: ReplaySSM：缓存 SSM 输入，而非状态

    ReplaySSM: Cache SSM Inputs, Not State

    [https://arxiv.org/abs/2610.13778](https://arxiv.org/abs/2610.13778)

    ReplaySSM 通过将 SSM 状态表示为周期性更新的检查点加近期输入缓冲区，将状态内存流量约减半，使投机解码回滚变为简单的缓冲区指针操作，并支持以低精度检查点进行输出计算。

    

    状态空间模型（SSM）维护一个固定大小的循环状态，但在每个解码步骤都对其进行读写可能成为混合模型的瓶颈。这种急切的状态管理会导致大量低算术强度的状态 I/O，使投机解码无法进行低成本的回滚，并将状态量化误差引入后续的循环更新中。我们提出了 ReplaySSM，它通过展开循环，将状态表示为一个检查点和一个近期 SSM 输入的缓冲区，从而将输出计算与检查点更新分离开来。通过周期性地更新检查点，ReplaySSM 在大多数步骤中避免了状态写入，并大约将主导性的状态内存流量减半。保留近期输入还使投机回滚成为缓冲区指针操作，并支持仅输出的验证，而无需具体化中间状态。同样的分离机制还允许低精度的检查点在大多数步骤中服务于输出计算，而更高精度的……（摘要原文在此处被截断）

    arXiv:2610.13778v1 Announce Type: cross  Abstract: State-space models (SSMs) maintain a fixed-size recurrent state, but reading and writing it at every decoding step can bottleneck hybrid models. This eager state management causes large state I/O with low arithmetic intensity, leaves speculative decoding no cheap rollback, and feeds state quantization error into subsequent recurrent updates. We introduce ReplaySSM, which unrolls the recurrence to represent the state as a checkpoint and a buffer of recent SSM inputs. This separates output computation from checkpoint updates. By updating the checkpoint periodically, ReplaySSM avoids state writes on most steps and roughly halves the dominant state memory traffic. Keeping recent inputs also makes speculative rollback a buffer-pointer operation and enables output-only verification without materializing intermediate states. The same separation allows a low-precision checkpoint to serve output computation on most steps while a higher-precisio
    
[^14]: DualQuant：面向大语言模型的双尺度免校准权重与激活量化

    DualQuant: Dual-scale Calibration-free Weight-and-Activation Quantization for LLMs

    [https://arxiv.org/abs/2610.13767](https://arxiv.org/abs/2610.13767)

    提出了免校准的双尺度训练后量化框架DualQuant，利用最小化权重重建误差所得的缩放因子作为激活缩放，无需校准数据即可同时完成大语言模型的权重与激活低比特量化。

    

    在原生支持4比特的加速器上进行低比特推理，需要同时对权重和激活进行量化，才能充分发挥这些格式所提供的吞吐量提升。虽然训练后权重量化已得到广泛研究，但激活量化仍然更具挑战性，且通常需要校准数据。在这项工作中，我们提出了DUALQUANT，一个双尺度、免校准的训练后量化框架，它直接最小化权重重建的均方误差（MSE），并作为直接副产品，在激活量化之前产生逐通道的激活缩放因子。我们的关键观察是，通过最小化权重重建误差获得的缩放因子能够提供有效的激活缩放，从而在无需校准数据的情况下实现具有竞争力的激活量化。我们将DUALQUANT表述为一个不可微的权重重建问题，并开发了一个高效的求解器。

    arXiv:2610.13767v1 Announce Type: cross  Abstract: Low-bit inference on accelerators with native 4-bit support requires quantizing both weights and activations to realize the full throughput gains these formats provide. While post-training weight quantization has been extensively studied, activation quantization remains considerably more challenging and typically requires calibration data. In this work, we introduce DUALQUANT, a dual-scale, calibration-free post-training quantization framework that directly minimizes the weight-reconstruction Mean Squared Error (MSE) and, as an immediate by-product, yields per-channel activation rescalings prior to activation quantization. Our key observation is that the scaling factors obtained by minimizing weight-reconstruction error provide effective activation rescalings, enabling competitive activation quantization without calibration data. We formulate DUALQUANT as a non-differentiable weight-reconstruction problem and develop an efficient solve
    
[^15]: 当边独立性失效时：基于潜在社交性先验的联合图扩散

    When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors

    [https://arxiv.org/abs/2610.13765](https://arxiv.org/abs/2610.13765)

    本文指出离散图扩散模型的E-R随机图先验是GNN去噪器失效的根本原因，并提出潜在社交性先验（LSP）与联合扩散过程，使节点社交性变量与图结构共同演化，从而恢复生成图中的节点异质性和高阶结构。

    

    离散图扩散模型通过具有马尔可夫噪声模型的独立边更新来破坏图结构。结合对输出分布的交换性要求，终端先验必然是Erdős–Rényi（E-R）随机图。这对无节点特征的无条件图生成具有根本性的影响：在大图极限下，任何应用于E-R图的GNN去噪器都近似输出E-R图，使得有效去噪变得困难。在实践中，生成的图无法恢复有意义的节点异质性和高阶结构。以往的模型引入专门的节点特征来弥补这一问题，将其归结为表达能力问题，而先验的作用则未被审视。我们转而从根本原因入手，引入潜在社交性先验（LSP）和一种联合扩散过程，使潜在节点社交性变量与图结构共同演化。

    arXiv:2610.13765v1 Announce Type: cross  Abstract: Discrete graph diffusion models corrupt graph structure through independent edge updates with a Markovian noise model. Combined with the exchangeability requirement on the output distribution, the terminal prior is necessarily an Erd\H{o}s--R\'enyi (E-R) random graph. This has a fundamental consequence for unconditional graph generation without node features: in the large graph limit, any GNN denoiser applied to an E-R graph approximately outputs an E-R graph, making effective denoising difficult. In practice, generated graphs fail to recover meaningful node heterogeneity and higher-order structure. Previous models introduced specialized node features to compensate, framing this as an expressivity problem, while the role of the prior remained unexamined. We instead address the root cause by introducing the Latent Sociability Prior (LSP) and a join} diffusion process that co-evolves latent node sociability variables and graph structure,
    
[^16]: 面向大语言模型智能体训练的匹配上下文群组策略优化

    Matched-Context Group Policy Optimization for LLM Agent Training

    [https://arxiv.org/abs/2610.13755](https://arxiv.org/abs/2610.13755)

    MCGPO通过匹配上下文从已有轨迹中构建加权动作偏好，在无需critic和额外rollout的情况下，为稀疏终端反馈下长时程LLM智能体的中间决策分配信用。

    

    训练长时程语言模型智能体需要从稀疏的终端反馈中为中间决策分配信用。在有限的同批rollout群组内，有用的动作比较取决于其上下文、两个备选方案的证据支持，以及这些证据如何进入学习过程。我们提出匹配上下文群组策略优化（MCGPO），这是一种从已收集轨迹中构建加权动作偏好的方法。MCGPO通过动作前的状态摘要和观察到的评分标准进展来定义比较上下文，并由任务规则进行标注。双动作支持机制在有序的精确匹配与粗略匹配规则中进行选择。固定折将候选决策与用于定向其偏好的终端结果分离开来，而依赖支持度的证据权重则在共享的GraphGPO任务学习之外，对辅助成对目标进行缩放。MCGPO复用rollout数据以及每个响应的原始历史，保持无critic（价值函数）的训练方式（摘要在此处截断）。

    arXiv:2610.13755v1 Announce Type: cross  Abstract: Training long-horizon language model agents requires assigning credit to intermediate decisions from sparse terminal feedback. Within a finite group of sibling rollouts, useful action comparisons depend on their context, evidence for both alternatives, and how that evidence enters learning. We propose Matched-Context Group Policy Optimization (MCGPO), a procedure for constructing weighted action preferences from already collected trajectories. MCGPO defines comparison contexts with pre-action state summaries and observed rubric progress, annotated by task rules. Two-action support selects among ordered exact and coarse matching rules. Fixed folds separate candidate decisions from the terminal outcomes used to orient their preferences, and support-dependent evidence weights scale an auxiliary pairwise objective alongside shared GraphGPO task learning. MCGPO reuses rollout data and each response's original history, preserving critic-free
    
[^17]: 只会打分而不会搜索的特权教师：面向推理的批评条件化在策略自蒸馏

    A Privileged Teacher That Only Scores Cannot Search: Critique-Conditioned On-Policy Self-Distillation for Reasoning

    [https://arxiv.org/abs/2610.13744](https://arxiv.org/abs/2610.13744)

    本文发现自蒸馏中的评分信号既无法定位学生推理中的错误，也无法可靠区分正确与错误的rollout，并指出错误定位本质上是搜索问题，因而提出批评条件化的在策略自蒸馏方法来弥补“只会打分、不会搜索”的评分教师的结构性缺陷。

    

    在策略自蒸馏有望用密集的token级信用信号取代可验证奖励强化学习中稀疏的结果信号：一个以参考解为条件的教师对学生在策略生成的rollout进行评分，由此产生的分布偏移再被蒸馏回策略中。我们研究这种评分信号是否真的包含纠正性信用。利用学生rollout中的黄金错误区间，我们检验了自蒸馏目标所依赖的两种信号——全分布KL散度和token对数比——是否具备两个关键性质：能否定位错误，以及能否将概率从错误的学生token上推开。结果表明，这两种信号均不具备上述性质，也无法在同一问题内可靠地区分正确与错误的rollout。我们认为这种失败是结构性的：在推理轨迹中定位错误本身就是一个搜索问题，而评分教师从未被明确要求执行这一任务。

    arXiv:2610.13744v1 Announce Type: new  Abstract: On-policy self-distillation promises to replace the sparse outcome signal of verifiable-reward RL with dense token-level credit: a teacher conditioned on the reference solution scores the student's own rollout, and the resulting distributional shift is distilled back into the policy. We investigate whether this scoring signal actually contains corrective credit. Using gold error spans in student rollouts, we test the two signals consumed by self-distillation objectives, the full-distribution KL and the token log-ratio, for two properties: whether they localize errors and whether they push probability away from erroneous student tokens. They exhibit neither of these properties nor can they reliably distinguish correct from incorrect rollouts within the same problem. We argue that this failure is structural: locating an error in a reasoning trace is itself a search problem, while a scoring teacher is never explicitly asked to perform that 
    
[^18]: SafeDiSTL：面向安全多智能体强化学习的联合约束分解

    SafeDiSTL: Decomposing Joint Constraints for Safe Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.13737](https://arxiv.org/abs/2610.13737)

    SafeDiSTL利用信号时序逻辑鲁棒性与可微动力学提供的动作敏感度，将团队级联合约束代价分解并分配到各智能体，解决了联合约束下多智能体强化学习的信用分配难题。

    

    协作型多智能体任务通常对共享目标或集体安全施加约束。团队层面的违规代价函数可以衡量这种约束结果，但无法区分哪些智能体的动作对该违规的影响最大。我们提出了SafeDiSTL，这是一个约束多智能体强化学习框架，即使在仅有联合约束度量的情况下也能实现信用分配。我们利用信号时序逻辑（STL）的鲁棒性，通过运行最大代价增量来衡量回合级的违规程度。平滑的全回合鲁棒性和可微动力学提供了动作敏感度，这些敏感度在同一决策时刻将每个代价增量分配到各个智能体上。在加权各智能体对每个代价增量的份额时，这些敏感度还考虑了延迟效应。我们在协作载荷运输、联网导航和网络化交通控制等任务中对该框架进行了研究。SafeDiSTL实现了1.6-5.1倍的（性能提升）。

    arXiv:2610.13737v1 Announce Type: cross  Abstract: Cooperative multi-agent tasks often impose constraints on a shared objective or collective safety. A team level violation cost function can measure this constraint outcome, but does not distinguish which agents' actions most strongly influence that violation. We present SafeDiSTL, a constrained multi-agent reinforcement learning framework that allows for credit assignment even with joint constraint measurements. We use Signal Temporal Logic (STL) robustness to measure episode-level violation through running-maximum cost increments. Smooth full-episode robustness and differentiable dynamics provide action sensitivities, which allocate each cost increment across agents at the same decision time. These sensitivities account for delayed effects when weighting agents' shares of each cost increment. We study the framework in cooperative payload transport, connected navigation, and networked traffic control tasks. SafeDiSTL achieves 1.6-5.1x 
    
[^19]: Orthros：面向高效Transformer的相位感知异构注意力

    Orthros: Phase-Aware Heterogeneous Attention for Efficient Transformers

    [https://arxiv.org/abs/2610.13735](https://arxiv.org/abs/2610.13735)

    Orthros注意力通过相位感知的方式在序列处理的不同阶段动态切换计算模式，实现了线性复杂度的预填充与完整注意力的解码相结合，从而在重预填充、轻解码的长文本场景下比原始Transformer具有更低的渐近复杂度。

    

    诸如线性注意力和滑窗注意力等高效注意力机制，旨在将Transformer的计算复杂度从二次方降至线性。然而，这些机制在上下文记忆能力上的退化往往需要交错插入完整的注意力层，导致长文本处理中的二次方瓶颈依然未得到解决。为解决这一问题，我们提出了Orthros注意力，它将效率范式从结构性分配转变为相位感知解耦。作为一个统一模块，它能够在序列处理的不同阶段动态切换计算模式，实现线性复杂度的预填充和完整注意力的解码。我们的设计使Orthros注意力在以“重预填充、轻解码”为特征的常见长文本场景中，相比原始Transformer具有更低的渐近复杂度。除了理论推导和复杂度分析之外，大量实证评估也证明了Orthros的有效性……

    arXiv:2610.13735v1 Announce Type: cross  Abstract: Efficient attention mechanisms, such as linear and sliding window attention, aim to reduce the computational complexity of Transformers from quadratic to linear. However, their degraded in-context recall often necessitates interleaving full attention layers, leaving the quadratic bottleneck unresolved in long-context processing. To address this, we introduce Orthros attention, which shifts the efficiency paradigm from structural allocation to phase-aware decoupling. As a unified module, it dynamically switches its computation pattern across different phases of sequence processing, enabling linear complexity prefilling and full attention decoding. Our design grants Orthros attention a lower asymptotic complexity compared to vanilla Transformers in prevalent long-context scenarios characterized by heavy prefilling and light decoding. Beyond theoretical formulation and complexity analysis, extensive empirical evaluations demonstrate Orthr
    
[^20]: LAPT-Scale：通过在人类视频上进行潜在动作预训练来扩展具身策略

    LAPT-Scale: Scaling Embodied Policies with Latent Action Pre-Training on Human Videos

    [https://arxiv.org/abs/2610.13732](https://arxiv.org/abs/2610.13732)

    提出LAPT-Scale范式，通过在人类视频上学习连续潜在动作进行预训练，并结合机器人具身对齐与任务后训练，从而以低成本扩展通用机器人VLA策略的能力。

    

    摘要：通用机器人策略的扩展受限于具身交互数据收集的高昂成本。大规模的自我中心人类视频提供了一种丰富的替代方案，但其缺乏机器人动作标签，且现有的动作标注方法在真实世界视觉复杂性和多样化操作动态下往往存在噪声。我们提出了LAPT-Scale，这是一种可扩展的潜在动作预训练范式，包含两个组件：LAPT-Scale-LAM通过解释人类视频中帧到帧的视觉转换来学习连续潜在动作，并将其作为源自人类的监督信号；LAPT-Scale-Policy则通过人类视频潜在动作预训练、目标机器人具身对齐以及任务特定的后训练来训练VLA策略。与先前的向量量化LAM不同，LAPT-Scale-LAM采用连续潜在空间来保留细粒度的运动和交互动态。我们进一步结合了感知、语义和光流对齐信号

    arXiv:2610.13732v1 Announce Type: cross  Abstract: Scaling generalist robotic policies is limited by the high cost of collecting embodied interaction data. Large-scale egocentric human videos offer a rich alternative, but they lack robot action labels, and existing action annotation methods are often noisy under real-world visual complexity and diverse manipulation dynamics. We introduce LAPT-Scale, a scalable latent action pre-training paradigm with two components: LAPT-Scale-LAM learns continuous latent actions by explaining frame-to-frame visual transitions in human videos and uses them as human-derived supervision, while LAPT-Scale-Policy trains a VLA policy through human-video latent-action pre-training, target-robot embodiment alignment, and task-specific post-training. Unlike prior vector-quantized LAMs, LAPT-Scale-LAM uses a continuous latent space to preserve fine-grained motion and interaction dynamics. We further combine perceptual, semantic, and optical-flow alignment signa
    
[^21]: 基于图神经网络的摊销式反事实预测

    Amortized Counterfactual Prediction with Graph Neural Networks

    [https://arxiv.org/abs/2610.13724](https://arxiv.org/abs/2610.13724)

    提出反事实图神经过程（CFGNP），利用图神经网络端到端地直接从观测数据回答反事实查询，无需先估计因果图，从而避免因果图估计误差向下游反事实预测的传播。

    

    反事实推断研究观察到的结果在不同情形下可能有何不同，例如患者如果接受不同的治疗是否会康复。只要决策依赖于未观察到的替代方案的结果，此类问题就会出现。回答反事实查询通常需要了解潜在的因果结构，而这在实践中往往是不可得的。因此，一种常见的做法是先从观测数据中估计因果图，然后再进行反事实推断。然而，估计出的因果图中的误差可能会传播到下游的反事实预测中。为了应对这一挑战，我们提出了反事实图神经过程（CFGNP），这是一种端到端的方法，可以直接从观测数据回答反事实查询，而无需先估计因果图。CFGNP 利用图神经网络（GNN）来发掘对偶……（原文在此处截断）

    arXiv:2610.13724v1 Announce Type: cross  Abstract: Counterfactual inference asks how an observed outcome might have differed under different circumstances, such as whether a patient would have recovered had they received a different treatment. Such questions arise whenever decisions depend on the outcomes of alternatives that were not observed. Answering counterfactual queries generally requires knowledge of the underlying causal structure, which is often unavailable in practice. A common approach is therefore to first estimate the causal graph from observational data and then perform counterfactual inference. However, errors in the estimated causal graph can propagate to downstream counterfactual predictions. To address this challenge, we introduce the Counterfactual Graph Neural Process (CFGNP), an end-to-end approach that answers counterfactual queries directly from observational data without first estimating a causal graph. CFGNP uses Graph Neural Networks (GNNs) to exploit the dua
    
[^22]: 自适应 DHiSparse：基于负载自适应 KV 分层结构的推测解码

    Adaptive DHiSparse: Speculative Decoding over a Load-Adaptive KV Hierarchy

    [https://arxiv.org/abs/2610.13708](https://arxiv.org/abs/2610.13708)

    Adaptive DHiSparse 在分层稀疏 KV 架构上支持推测解码，并根据负载动态调整 KV 缓存驻留策略，在 100K 上下文、128 并发下相比 HiSparse 将解码吞吐提升至 1.45 倍。

    

    长上下文推理服务往往在硬件达到有效吞吐量之前就耗尽了 KV 缓存内存。HiSparse 将历史 KV 存储在主机内存中，仅取回被选中的条目，但它在任何负载下都使用单一的缓存预算，并且不支持诸如 DSpark 和 DFlash2 等基于草稿模型的推测解码。我们提出了 Adaptive DHiSparse，它支持在分层稀疏 KV 架构上进行推测解码，并根据负载自适应地调整 KV 驻留策略。它保护必需的工作集，在 GPU 内存有富余时保留更多的历史记录，并通过精确的融合检索恢复其余部分。DSpark 和 DFlash2 的提议与接受规则保持不变：查询共享缺失行的检索，草稿 KV 使用窗口受限的环形缓冲区，验证暂存在层组之间被复用。在 100K 上下文和 128 并发条件下，GLM-5.3 配合 DFlash2 在每个解码 GPU 上达到 793 输出 token/秒，是 HiSparse 的 1.45 倍、非推测自适应服务的 1.27 倍。（摘要在此处截断）

    arXiv:2610.13708v1 Announce Type: new  Abstract: Long-context serving runs out of KV memory before the hardware reaches its useful throughput. HiSparse stores history in host memory and fetches only the selected entries, but it uses one cache budget at every load and does not handle draft-model speculative decoding such as DSpark and DFlash2. We present Adaptive DHiSparse, which supports speculative decoding over hierarchical sparse KV and adapts residency to load. It protects the required working set, retains additional history while GPU memory is spare, and recovers the rest through exact fused retrieval. DSpark and DFlash2 keep their proposal and acceptance rules: queries share missing-row retrieval, draft KV uses a window-bounded ring, and verification staging is reused across layer groups. At 100K context and concurrency 128, GLM-5.3 with DFlash2 delivers 793 output tokens/s per decode GPU, 1.45$\times$ HiSparse and 1.27$\times$ non-speculative adaptive serving. Held to the same l
    
[^23]: 智能体网络技术栈的基线评估

    A Baseline Assessment for the Agentic Web Technology Stack

    [https://arxiv.org/abs/2610.13703](https://arxiv.org/abs/2610.13703)

    本文提出了MASTIF多智能体系统测试框架，对由3个LLM、3种协议和4个智能体实现框架组成的智能体网络技术栈在300个Mind2Web基准任务上进行了基线评估，发现针对特定任务选择技术栈可以提升任务完成率、效率得分和社会价值稳定性。

    

    针对基于大语言模型（LLM）的智能体，各类协议、框架和技术正以极快的速度被提出，这给任何旨在创建最先进AI解决方案的人带来了复杂的局面。如果缺乏对不同协议、框架和库如何协同工作以构建智能体网络解决方案的评估，开发者和组织甚至在成为智能体网络生态系统一员之前，就会面临各种挑战（例如技能培养、治理、基础设施等）。在此背景下，我们提出了一个用于执行智能体网络评估的系统，称为MASTIF（多智能体系统测试框架），并针对300个Mind2Web基准任务，对智能体网络技术进行了基线评估，其中包括3个大语言模型、3种协议和4个智能体实现框架。结果表明，选择面向特定任务的技术栈可能会带来任务完成率、效率得分和社会价值稳定性得分的提升，并实现某些方面的降低。

    arXiv:2610.13703v1 Announce Type: new  Abstract: The fast pace in which protocols, frameworks, and technologies are being proposed for LLM-based agents results in a complex scenario for anyone aiming to create state-of-the-art AI solutions. Without an assessment on how different protocols, frameworks, and libraries work together to compose an Agentic Web solution, developers and organizations face different challenges (e.g., skilling, governance, infrastructure) even before becoming part of the Agentic Web ecosystem. In this context, we propose a system for performing Agentic Web evaluations called MASTIF (Multi-Agent System TestIng Framework) and present a baseline assessment of Agentic Web technologies, including 3 LLMs, 3 protocols, and 4 agent implementation frameworks, against 300 Mind2Web benchmark tasks. Results suggest that selecting a task-specific technology stack may result in gains of task completion rate, efficiency score, and social value stability scores; and reductions 
    
[^24]: 一步到位的校准：联合学习既准确又校准良好的神经网络

    Calibration in One Go: Jointly Learning Accurate and Calibrated Neural Networks

    [https://arxiv.org/abs/2610.13693](https://arxiv.org/abs/2610.13693)

    该论文提出通过双层 Stackelberg 优化将校准器与主干网络端到端联合训练，无需单独的校准数据集或专用校准损失，仅引入校准器学习率这一个超参数，即可同时学习到既准确又校准良好的神经网络。

    

    神经网络（NN）虽然具有高度的灵活性和通用性，但往往校准不佳。事后校准方法使用单独的校准数据集和额外的校准参数来调整已经训练好的模型的输出。这一过程增加了额外开销，需要对校准器进行单独优化以及后续的模型选择。在本文中，我们证明了标准的（事后）校准器可以通过双层 Stackelberg 公式与主干网络进行端到端联合训练。该公式通过使用交叉熵同时优化准确性和校准性能，消除了对单独校准集或专用校准损失函数的需求，仅引入了一个额外的超参数：校准器学习率。尽管我们的框架支持从温度缩放、矩阵缩放到神经网络层等各种校准器，但我们的结果表明，使用最终的分类层本身作为校准器即可得到既准确又校准良好的神经网络。

    arXiv:2610.13693v1 Announce Type: cross  Abstract: Neural networks (NNs), while highly flexible and versatile, are often poorly calibrated. Post-hoc calibration methods adjust the outputs of an already trained model using a separate calibration dataset and additional calibration parameters. This process adds overhead that requires separate optimization of the calibrator and subsequent model selection. In this paper, we show that standard (post-hoc) calibrators can be trained end-to-end with the backbone via a bilevel Stackelberg formulation. This formulation eliminates the need for a separate calibration set or a dedicated calibration loss by jointly optimizing accuracy and calibration using cross-entropy, introducing only one additional hyperparameter: the calibrator learning rate. Although our framework supports calibrators ranging from temperature and matrix scaling to NN layers, our results show that using the final classification layer itself as the calibrator yields accurate, wel
    
[^25]: 方差缩减策略梯度近乎极小极大最优

    Variance-Reduced Policy Gradient is Nearly Minimax-Optimal

    [https://arxiv.org/abs/2610.13692](https://arxiv.org/abs/2610.13692)

    该论文提出LVR-PG统一理论框架，证明了一类结合递归更新与防御性重要性采样的一阶方差缩减策略梯度算法，在弱梯度支配等最小假设下、无需梯度预处理器，即可达到 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 样本复杂度的近乎极小极大最优最后一次迭代全局收敛。

    

    标准策略梯度（PG）算法由于依赖蒙特卡洛更新而存在高估计方差的问题。虽然离策略方差缩减技术可以缓解这一问题，但要在当前建立最优样本复杂度，要么需要限制性假设（如一致有界的重要性权重），要么需要不实用的二阶操作。在本工作中，我们提出了无环方差缩减策略梯度（LVR-PG），这是一个统一的理论框架，抽象出一类将递归更新与防御性重要性采样相结合的一阶估计器，其中包括已知的估计器如PAGE-PG和STORM-PG。我们证明，在一般参数化策略的弱梯度支配条件下，该算法类能够实现 $\tilde{\mathcal{O}}(\epsilon^{-2})$ 的最后一次迭代全局收敛。我们仅使用纯一阶更新，在最小假设条件下且无需梯度预处理器即可建立这一保证。

    arXiv:2610.13692v1 Announce Type: cross  Abstract: Standard Policy Gradient (PG) algorithms suffer from high estimation variance because they rely on Monte Carlo updates. While off-policy variance reduction mitigates this problem, establishing optimal sample complexity currently requires either restrictive assumptions, such as uniformly bounded importance weights, or impractical second-order operations. In this work, we introduce Loopless Variance-Reduced PG (LVR-PG), a unified theoretical framework that abstracts a class of first-order estimators that integrate recursive updates with defensive importance sampling, including known estimators such as PAGE-PG and STORM-PG. We prove that this algorithmic class achieves $\tilde{\mathcal{O}}(\epsilon^{-2})$ last-iterate global convergence under weak gradient domination for general parametric policies. We establish this guarantee using purely first-order updates, with minimal assumptions and without gradient preconditioners. By extending sta
    
[^26]: 并非所有维度生而平等：基于分组二次核的离群值感知注意力线性化

    Not All Dimensions Are Equal: Outlier-Aware Attention Linearization with Grouped Quadratic Kernels

    [https://arxiv.org/abs/2610.13674](https://arxiv.org/abs/2610.13674)

    该论文发现预训练注意力中少数离群维度主导了线性化的近似误差，并提出维度感知的OAL方法，通过Hadamard乘积表述将全局系数放宽为维度特定的线性项与跨维度二次交互，再以分组二次核在更少参数下实现高保真的注意力线性化。

    

    线性化softmax注意力为高效的Transformer推理提供了一条途径，但如何在近似中保留预训练注意力仍然充满挑战。我们识别出一个被标准线性化方法所忽略的重要近似误差来源：一小部分离群维度对查询-键点积的贡献不成比例地大，并占据了近似误差的很大一部分。受这种异质性的启发，我们提出了离群值感知注意力线性化（Outlier-Aware Attention Linearization, OAL），这是一种用于构建二次注意力近似的维度感知方法。从二阶泰勒近似出发，我们引入了一种Hadamard乘积表述，使各个查询-键的独立贡献得以显式呈现。该表述放宽了全局共享系数的限制，以支持维度特定的线性项和跨维度的二次交互，同时保持可分离的计算方式。为了以更少的参数保留这种灵活性，OAL将维度进行分组（原文在此处截断）。

    arXiv:2610.13674v1 Announce Type: cross  Abstract: Linearizing softmax attention offers a route to efficient Transformer inference, but preserving pretrained attention remains challenging. We identify an important source of approximation error overlooked by standard linearization methods: a small number of outlier dimensions contribute disproportionately to query-key dot products and account for a large fraction of the approximation error. Motivated by this heterogeneity, we propose Outlier-Aware Attention Linearization (OAL), a dimension-aware method for constructing quadratic attention approximations. Starting from a second-order Taylor approximation, we introduce a Hadamard-product formulation that makes individual query-key contributions explicit. This formulation relaxes globally shared coefficients to support dimension-specific linear terms and cross-dimensional quadratic interactions while preserving separable evaluation. To retain this flexibility with fewer parameters, OAL gro
    
[^27]: 注入轮次而非标记：学习在何处重定向大语言模型推理以实现自我纠错

    Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction

    [https://arxiv.org/abs/2610.13672](https://arxiv.org/abs/2610.13672)

    该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。

    

    大语言模型难以可靠地识别并纠正自身推理中的错误。与以往通过微调大语言模型来生成修订或反馈的训练类方法不同，我们学习在失败的推理生成过程中于何处进行干预，并确定如何对其重定向。为了选择干预位置，我们在冻结的编码器上仅利用最终答案的正确性训练了一个 13 万参数的 MLP 头。对于重定向方式，我们发现在固定的生成预算下，已测试的行内注入无法可靠地改变推理方向，而将相同指令作为新的用户轮次呈现则能改变模型的走向。由此得到的纠错框架 UPTURN（在习得位置注入用户提示轮次）无需对大语言模型进行任何微调：它保留选定的前缀，将一条指令作为新的用户轮次注入，并重新生成后缀。在四个推理模型和六个数学基准上，UPTURN 在所有设置下均提升了平均准确率。

    arXiv:2610.13672v1 Announce Type: cross  Abstract: Large language models struggle to reliably identify and correct errors in their own reasoning. Unlike prior training-based approaches that fine-tune an LLM to generate revisions or feedback, we learn where to intervene in failed rollouts and determine how to redirect them. To choose where to intervene, we train a 130K-parameter MLP head on a frozen encoder using only final-answer correctness. For redirection, we find that under a fixed generation budget, the tested inline injections fail to reliably redirect reasoning, whereas presenting the same instruction in a new user turn changes the model's course. The resulting correction framework, UPTURN (User-Prompt TURN injection at a learned position), requires no LLM fine-tuning: it retains a selected prefix, injects an instruction as a new user turn, and regenerates the suffix. Across four reasoners and six math benchmarks, UPTURN improves average accuracy in every setting over a 32-rollo
    
[^28]: 更进一步而非过犹不及：GRPO中课程难度递进的收益与局限

    A Step Harder, Not a Step Too Far: The Benefits and Limits of Curriculum Difficulty Progression in GRPO

    [https://arxiv.org/abs/2610.13669](https://arxiv.org/abs/2610.13669)

    该论文提出在GRPO训练中基于当前策略动态重估样本难度，并设计逐步向更难样本推进的分阶段课程，实验表明持续只选中等难度样本并非最优，适度渐进地提升难度才能带来更强的最终性能，但难度提升也存在限度。

    

    课程学习通常按照从易到难的顺序组织训练样本。在GRPO中，中等难度的样本对在线选择颇具吸引力，因为它们更有可能产生信息丰富的组间相对学习信号，优先选择这些样本可以提升训练效率。然而，持续选择此类样本是否也能带来最强的最终模型性能，或者训练是否能从向更难样本推进中获益，目前仍不清楚。同时，样本难度会随策略的演化而变化，固定的难度估计可能与当前策略产生偏差。为此，我们在当前策略下重新估计样本难度，并受课程学习从易到难原则的启发，设计并评估了一种分阶段课程，将活跃难度区间逐步向更难的样本推进。在GSM8K上的实验表明，持续选择中等难度样本并不能（原文摘要在此处截断）

    arXiv:2610.13669v1 Announce Type: new  Abstract: Curriculum learning typically organizes training samples from easy to hard. In GRPO, medium-difficulty samples are attractive for online selection because they are more likely to yield informative group-relative learning signals, and prioritizing them can improve training efficiency. However, it remains unclear whether persistently selecting such samples also leads to the strongest final model performance, or whether training can benefit from progressing toward harder samples. Meanwhile, sample difficulty changes as the policy evolves, so fixed difficulty estimates can become misaligned with the current policy. We therefore re-estimate sample difficulty under the current policy and, inspired by the easy-to-hard principle of curriculum learning, design and evaluate a staged curriculum that progressively shifts the active difficulty band toward harder samples. Experiments on GSM8K show that persistent medium-difficulty selection does not a
    
[^29]: 用于专用神经网络模型自动调优的多智能体LLM系统

    A Multi-Agent LLM System for Automated Tuning of Specialized Neural Network Models

    [https://arxiv.org/abs/2610.13663](https://arxiv.org/abs/2610.13663)

    本文提出一个由分析师、编码员和记录员组成的多智能体LLM框架，在保持整体网络结构不变的前提下自动生成并调优指定关键组件的代码，并可迁移至脉冲神经网络分类与PINN求解高阶微分方程等不同类型模型。

    

    许多机器学习方法都是两级结构：专家先选定一类方法，而在该类方法内部，关键组件的选择至关重要且通常依赖人工。本文致力于实现这些组件的自动调优，并提出了一种可跨不同性质模型迁移的框架。该多智能体大语言模型（LLM）框架包含三个角色——分析师、编码员和记录员——并配备实验日志和沙盒执行环境。与优化整个机器学习脚本或在离散空间中搜索架构的系统不同，该框架在一个开放但语义上狭窄的空间中运作：智能体仅为指定的组件生成代码，而保持整体网络结构不变。该方法在两个方向上得到验证：调优基于相关性学习的分类脉冲神经网络，以及使用物理信息神经网络（PINN）求解高阶微分方程。

    arXiv:2610.13663v1 Announce Type: new  Abstract: Many machine learning approaches are two-level: an expert selects a class of methods, within this class the choice of key components is critical and usually manual. This paper addresses the automatic tuning of these components and proposes a framework transferable across models of different nature. The multi-agent large language model (LLM) framework has three roles---analyst, coder, and recorder~---with an experiment log and a sandboxed execution environment. Unlike systems that optimize the entire ML script or search architectures in a discrete space, the framework operates over an open but semantically narrow space: the agent generates code only for the specified components, leaving the overall network structure unchanged. The approach is validated in two directions: tuning a classification spiking neural network with correlation learning, and solving high-order differential equations with Physics-Informed Neural Networks (PINN). Acro
    
[^30]: 预训练与测试时训练预算对ARC至关重要：VARC与TRM的对照研究

    Pretraining and Test-Time Training Budgets Matter for ARC: A Controlled Study of VARC and TRM

    [https://arxiv.org/abs/2610.13660](https://arxiv.org/abs/2610.13660)

    本文在统一协议下控制变量地对比视觉型VARC与递归型TRM模型，证明预训练与测试时训练预算（尤其是增加TTT样本呈现次数）对ARC性能有决定性影响，提示性能差异不能简单归因于架构本身。

    

    抽象与推理语料库（ARC）越来越多地被用于评估神经网络能否从少量示例中推断出任务特定的变换。近期的ARC模型不仅在架构上存在差异，在更广泛的训练与评估协议上也各不相同，包括预训练数据、数据增强、测试时设置和优化预算，这使得关于架构优势的结论备受争议。在本工作中，我们在ARC-AGI-1上采用统一的归纳协议，研究视觉启发的VARC模型与基于递归的TRM模型的性能如何受预训练和测试时训练（TTT）预算的影响。我们将五个预训练预算与每个测试任务的四个TTT预算进行交叉组合，每个预算规定了任务变体数量和总样本呈现次数，从而定义了受控的训练预算。我们观察到，在所有设置下，增加TTT样本呈现次数均能提升两种模型的标准ARC pass@2指标。

    arXiv:2610.13660v1 Announce Type: cross  Abstract: The Abstraction and Reasoning Corpus (ARC) is increasingly used to evaluate whether neural networks can infer task-specific transformations from few demonstrations. Recent ARC models differ in architecture, but also in broader training and evaluation protocols, including pretraining data, data augmentation, test-time setup, and optimization budgets, making conclusions about architecture benefits debatable. In this work, we study how the performance of the vision-inspired VARC and recursion-based TRM models is affected by pretraining and test-time training (TTT) budgets under a common inductive protocol on ARC-AGI-1. We cross five pretraining budgets with four TTT budgets for each test task. Each budget specifies a task-variant count and total sample presentations, defining the controlled training budgets. We observe that increasing TTT sample presentations improves the standard ARC pass@2 metric in all settings for both models. Increas
    
[^31]: SelectArena：面向真实世界表格数据的现代特征选择基准测试

    SelectArena: A Modern Benchmark for Feature Selection on Real-world Tabular Data

    [https://arxiv.org/abs/2610.13644](https://arxiv.org/abs/2610.13644)

    该论文提出了SelectArena——一个基于真实世界表格数据的现代特征选择基准测试，通过比较15种广泛使用的特征选择方法，弥补了现有基准依赖过时模型、临时评估协议和人工数据而无法指导实际应用的不足。

    

    从表格数据集中选择特征子集（列）是现实应用中普遍存在的挑战。行业专家希望最小化模型训练成本，气候科学家需要可解释的因素子集来理解极端天气事件，而临床医生由于财务和时间限制需要在各类诊断测试中做出选择。当从业者查阅科学文献以确定应使用哪种特征选择方法时，他们面对的是一个缺乏代表性的基准测试现状。尽管相关研究众多，现有的特征选择基准测试所能提供的指导却十分有限，因为它们依赖于过时的模型、临时设计的评估协议，或无法反映真实问题的人工生成数据。为了弥合科学基准测试与现实应用之间的差距，我们推出了 SelectArena，一个现代化的特征选择基准测试。我们比较了15种广泛使用的特征选择方法……

    arXiv:2610.13644v1 Announce Type: cross  Abstract: Selecting a subset of features (columns) from a tabular dataset is a ubiquitous challenge in real-world applications. Industry experts seek to minimize model training costs, climate scientists require an interpretable subset of factors to understand extreme weather events, and clinicians need to choose among diagnostic tests due to financial and time constraints. When looking at the scientific literature to determine which feature selection method to use, practitioners across domains face an unrepresentative landscape of benchmarking. Despite extensive research, existing feature selection benchmarks provide limited guidance because they rely on outdated models, ad hoc evaluation protocols, or artificially generated data that do not reflect real-world problems. To close the gap between scientific benchmarking and real-world applications, we introduce SelectArena, a modern benchmark for feature selection. We compare 15 widely used featur
    
[^32]: 从输入数据及正负示例解中学习尺寸无关的约束模型

    Learning Size Independent Constraint Models from Input Data and Positive and Negative Example Solutions

    [https://arxiv.org/abs/2610.13640](https://arxiv.org/abs/2610.13640)

    本文提出一个约束获取工具，能从输入数据和多个问题实例的正负示例解中自动学习特定尺寸的约束模型并将其泛化为尺寸无关的模型，生成可在未见数据上运行的MiniZinc代码。

    

    本文描述了一个约束获取工具，该工具能够从输入数据以及多个问题实例的正负示例解中生成与尺寸无关的约束程序。该系统基于全局约束和二元约束上的典型约束模式找到特定尺寸的模型，然后将这些模型泛化为可以在未见过的输入数据上运行的尺寸无关模型。该系统生成MiniZinc代码以及模型的文本描述。我们在PTHG21挑战赛的示例问题上对该工具进行了评估，并展示了该方法的可行性。

    arXiv:2610.13640v1 Announce Type: new  Abstract: In this paper we describe a Constraint Acquisition Tool that generates size independent constraint programs from input data and positive and negative example solutions of multiple problem instances. The system finds size specific models based on typical constraint patterns over global and binary constraints, then generalizes the models to a size independent model that can be run on unseen input data. The system produces MiniZinc code as well as a textual description of the model. We evaluate the tool on the example problems of the PTHG21 Challenge and show the feasibility of the approach.
    
[^33]: 模型无关的知识增强机器学习用于弱标注生物成像数据

    Model-agnostic Knowledge Augmented Machine Learning for Weakly Annotated Bioimaging Data

    [https://arxiv.org/abs/2610.13637](https://arxiv.org/abs/2610.13637)

    提出一种模型无关的知识增强方法，通过注入领域知识显著提升弱标注生物成像数据的分类性能，与自监督学习结合时在多个数据集上实现最高6.8%的精确率提升。

    

    机器学习模型日益广泛地被用于从弱标注的生物成像数据（如显微镜和细胞形态学图像）中自动提取特征。然而，由于完全依赖自动特征提取，现有方法在不同弱标注数据集上表现出不一致的分类性能。本文提出了一种模型无关的知识增强方法，使机器学习模型能够利用超出自动特征提取所能捕获范围的领域知识。所提出的方法在七种机器学习架构上均提升了分类性能。当与基于掩码自编码器的自监督学习相结合时，我们的方法取得了最强的整体性能：在三个内部显微镜数据集上，精确率较次优方法提高了2.9%；在三个公开的细胞形态学基准数据集上，精确率平均提高了6.8%。我们基于TAR DNA结合蛋白的错位定位进行了细胞核级别的分类。

    arXiv:2610.13637v1 Announce Type: cross  Abstract: Machine learning models are increasingly used for automated feature extraction from weakly annotated bioimaging data, such as microscopy and cytomorphology images. Due to the sole dependency of automated feature extraction, existing methods yield inconsistent classification performance across weakly annotated datasets. Here, we propose a model-agnostic knowledge augmentation method that enables ML models to exploit domain knowledge beyond what automated feature extraction can capture. The proposed approach improves classification performance across seven ML architectures. When integrated with masked autoencoder-based self-supervised learning, our method achieves the strongest overall performance, improving precision by 2.9% over the next-best method on three in-house microscopy datasets and by an average of 6.8% across three public cytomorphology benchmarks. We perform nucleus-level classification based on mislocalization of TAR DNA-bi
    
[^34]: 人本化奥赛智能体：一种通过求解、评审、维基、验证实现奥赛问题求解的多会话工作流

    Humanize Olympiad Agents: A Multi-Session Workflow for Olympiad Problem Solving with Solve, Review, Wiki, Verify

    [https://arxiv.org/abs/2610.13632](https://arxiv.org/abs/2610.13632)

    提出 HOA，一个开放的流程级框架，通过独立评审、结构化维基与验证三个互补反馈通道协调多个智能体会话，以解决单一会话智能体在奥赛难题上易传播错误、缺乏知识且可能曲解任务的脆弱性问题。

    

    大语言模型（LLM）智能体在长周期科学与奥赛任务上的能力正迅速提升。近期的系统已经产生了纳维-斯托克斯（Navier-Stokes）问题的 Lean 形式化解答，以及对埃尔德什（Erdős）猜想数据库中四个开放问题的自主求解。然而，这些系统均未同时公开其底层模型与完整的推理技术栈。更广泛地说，单一智能体会话在困难问题上仍然脆弱：早期错误可能不断传播，缺失的领域知识可能使进展停滞，而表面上的成功可能掩盖了对原始任务的误解或弱化。我们提出 HOA（Humanize Olympiads Agent，人本化奥赛智能体），这是一个开放的、流程级框架，通过三个互补的反馈通道来协调多个智能体会话。首先，一个独立的评审者对每个候选解进行评估并提供可操作的反馈。其次，一个结构化的维基提供模型中可能缺失或不可靠的任务特定知识……（摘要截断）

    arXiv:2610.13632v1 Announce Type: new  Abstract: Large language model (LLM) agents are rapidly improving on long-horizon scientific and olympiad tasks. Recent systems have produced a Lean-formalized resolution of the Navier--Stokes problem and autonomous solutions to four open questions in the Erd\H{o}s Conjectures database. Yet neither system releases both its underlying model and complete inference stack. More broadly, a single agent session remains brittle on difficult problems: early errors can propagate, missing domain knowledge can stall progress, and apparent success can mask a misinterpretation or weakening of the original task. We introduce HOA (Humanize Olympiads Agent), an open, flow-level framework that coordinates multiple agent sessions through three complementary feedback channels. First, an independent reviewer evaluates each candidate and provides actionable feedback. Second, a structured wiki supplies task-specific knowledge that may be missing or unreliable in model 
    
[^35]: SWE-DeceptionBench：贯穿软件开发生命周期的大语言模型欺骗行为基准测试

    SWE-DeceptionBench: Benchmarking Deceptive Behaviors of LLMs Across the Software Development Lifecycle

    [https://arxiv.org/abs/2610.13626](https://arxiv.org/abs/2610.13626)

    该论文提出了SWE-DeceptionBench基准，系统评估大语言模型在软件开发生命周期六个阶段中的主动欺骗与被动欺骗行为，并通过中立、奖励、压力三种条件考察外部诱导对欺骗行为的影响。

    

    大语言模型（LLM）在软件开发生命周期中的应用日益广泛。这种对LLM日益增长的依赖引发了对其可信度的担忧，其中欺骗行为构成了一项尤为重要的风险。欺骗行为表现为误导性回复，或在伪造、隐瞒、扭曲信息方面提供协助。为研究这一行为，我们提出了SWE-DeceptionBench，这是一个用于评估LLM在软件工程中欺骗行为的基准。该基准涵盖六个生命周期阶段以及两种欺骗触发方式：主动欺骗，即模型明知需求不可行或存在冲突，却仍将其表述为可实现的；以及被动欺骗，即模型帮助满足用户明确提出的、旨在误导他人的请求。我们进一步引入中立、奖励和压力三种条件来研究外部诱导因素的影响，最终构建了包含中英文的720个测试实例。对七个主流大语言模型的评估……

    arXiv:2610.13626v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used throughout the software development lifecycle. This growing reliance on LLMs raises concerns about their trustworthiness, with deception posing a particularly important risk. Deceptive behavior takes the form of misleading responses or assistance in fabricating, concealing, or distorting information. To study this behavior, we introduce SWE-DeceptionBench, a benchmark for evaluating LLM deception in software engineering. The benchmark covers six lifecycle stages and two deception triggers: Active Deception, where models recognize infeasible or conflicting requirements but still present them as achievable, and Passive Deception, where models help fulfill explicit user requests to mislead others. We further introduce Neutral, Reward, and Pressure conditions to study the effect of external inducements, resulting in 720 test instances in Chinese and English. Evaluating seven mainstream LLM
    
[^36]: AudioTETRIS：通过过渡重要性信号进行token消除的高效音频大语言模型

    AudioTETRIS: Token Elimination via Transitional Importance Signals for Efficient Audio LLMs

    [https://arxiv.org/abs/2610.13600](https://arxiv.org/abs/2610.13600)

    提出AudioTETRIS，一个无需训练的两阶段token压缩框架，利用音频编码器的表示几何作为声学先验，识别声学过渡处信息密度峰值的关键token，在保留细粒度语音过渡信息的同时大幅降低音频大语言模型的推理计算开销。

    

    大型音频语言模型（LALMs）在语音-文本任务上表现出色，但标准音频编码器即使对短音频片段也会生成数百个token。由于自注意力的计算量随序列长度呈二次方增长，这大幅增加了推理时的计算开销。当前无需训练的token压缩方法仅基于粗略的注意力或特征相似度来剪枝或合并token。然而，将音频token视为通用的各向同性特征会平滑掉频谱过渡，从而破坏对语音解码至关重要的细粒度语音过渡信息。相比之下，语音信号表现出高度结构化的时间模式，其信息密度在声学过渡处达到峰值。我们提出AudioTETRIS，这是一个无需训练的两阶段框架，在应用基于注意力的压缩之前，利用音频编码器的表示几何结构作为token重要性的声学先验……

    arXiv:2610.13600v1 Announce Type: cross  Abstract: Large Audio Language Models (LALMs) achieve strong performance on speech-text tasks, but standard audio encoders produce hundreds of tokens for even short audio clips. Because self-attention scales quadratically in sequence length, this inflates inference-time compute substantially. Current training-free token compression methods prune or merge tokens based solely on coarse attention or feature similarity. However, treating audio tokens as generic, isotropic features smooths over spectral transitions, destroying fine-grained phonetic transitions essential for decoding speech. In contrast, speech signals exhibit highly structured temporal patterns where information density peaks at acoustic transitions. We propose AudioTETRIS, a training-free, two-stage framework that uses the audio encoder's representation geometry as an acoustic prior for token importance before applying attention-based compression. Instead of discarding encoder struc
    
[^37]: MENTOR：通过在线策略式教学干预从失败中学习

    MENTOR: Learning from Failure with On-Policy Pedagogical Intervention

    [https://arxiv.org/abs/2610.13594](https://arxiv.org/abs/2610.13594)

    提出了MENTOR多智能体框架，通过在线策略式教学干预，实时诊断失败轨迹的根本原因并动态生成前瞻性引导，从而解决大推理模型训练中的“零奖励崩溃”问题。

    

    从可验证奖励的强化学习（RLVR）是大型推理模型的关键驱动力，但它容易遭受“零奖励崩溃”——一种训练失败现象，即任务难度突然增加导致学习信号消失。我们认为，传统丢弃这些失败轨迹的做法是一种浪费，因为它们包含了关于模型认知差距的宝贵诊断信息，这是一个可以被利用来实现恢复的亮点。然而，现有解决方案通常退化为静态的、“离策略式”的干预手段，如预定义的提示库，或者进行事后修正，无法提供摆脱崩溃所需的适应性、前瞻性指导。为解决这一问题，我们提出向在线策略式教学干预的范式转变，并引入MENTOR——一个充当实时导师的多智能体框架。MENTOR实时诊断失败的根本原因，并动态生成前瞻性的引导。

    arXiv:2610.13594v1 Announce Type: new  Abstract: Reinforcement Learning from Verifiable Rewards (RLVR) is a key driver for Large Reasoning Models, but it is vulnerable to "zero-reward collapse", a training failure where a sudden increase in task difficulty causes the learning signal to vanish. We argue that the conventional approach of discarding these failed trajectories is wasteful, as they contain valuable diagnostic information about the model's cognitive gaps, a bright side that can be leveraged for recovery. Existing solutions, however, often regress to static, "off-policy-style" interventions like pre-defined hint libraries or perform post-hoc refinement, failing to provide the adaptive, forward-looking guidance needed to escape the collapse. To address this, we propose a paradigm shift towards on-policy pedagogical intervention and introduce MENTOR, a multi-agent framework that acts as a live mentor. MENTOR diagnoses the root cause of failure in real-time and dynamically genera
    
[^38]: 构建推理模型中过度思考的最优推理终止方法的帕累托前沿

    Constructing Pareto Frontiers of Optimal Reasoning Termination Methods for Overthinking in Reasoning Models

    [https://arxiv.org/abs/2610.13570](https://arxiv.org/abs/2610.13570)

    该论文提出逐步截断探测（STP）方法来量化推理模型的过度思考程度，并在统一框架下评估动态路由、固定预算和动态提前退出等终止策略，构建了准确率与效率之间权衡的帕累托前沿。

    

    推理模型可以从测试时扩展中受益，但持续的推理可能变得无效甚至有害，这种现象被称为“过度思考”。目前针对完整思考过程的评估只比较准确率和token使用量，缺乏衡量过度思考严重程度以及各类策略理论缓解潜力的参照基准。为了在模型-任务层面刻画过度思考，我们首先引入了逐步截断探测方法，在每条固定的采样推理轨迹上，从多个保留的前缀重复生成答案。以每条轨迹上最早达到最大经验准确率的测量位置作为参照，我们定义了过度思考比率和准确率退化，分别表示其后生成的思考token所占比例，以及从该参照点到完整思考的准确率下降幅度。其次，我们将动态路由、固定预算和动态提前退出纳入统一框架，以量化各方法的理论缓解潜力……

    arXiv:2610.13570v1 Announce Type: new  Abstract: Reasoning models can benefit from test-time scaling, but continued reasoning may become ineffective or harmful, a phenomenon termed overthinking. Evaluations against full thinking compare accuracy and token usage but lack references for overthinking severity and strategies' theoretical mitigation potential. To characterize overthinking at the model-task level, we first introduce Step-wise Truncation Probing (STP), repeatedly generating answers from multiple retained prefixes along each fixed sampled reasoning trajectory. Using each trajectory's earliest measured position attaining maximum empirical accuracy as a reference, we define Overthinking Ratio and Accuracy Degradation as the fraction of thinking tokens generated afterward and the accuracy decrease from this reference to full thinking, respectively. Secondly, we then formulate dynamic routing, fixed budgets, and dynamic early exit within a unified framework to quantify the theoret
    
[^39]: 防御蒸馏的黑魔法：阈下学习的无监督发现

    Defence against the dark arts of distillation: Unsupervised discovery of subliminal learning

    [https://arxiv.org/abs/2610.13536](https://arxiv.org/abs/2610.13536)

    提出了一种无需预先知晓目标特征的无监督防御方法，仅通过分析学生模型的参数更新即可发现、解释并抑制从共享基础参数的教师模型阈下继承的失对齐行为特征。

    

    当语言模型从共享其基础参数的教师模型蒸馏而来时，学生模型可能会继承教师模型的行为特征——包括失对齐——即使训练数据在语义上与该特征无关。现有的所有防御方法都需要预先知道该特征。我们移除了这一要求，并证明通过利用一个关键的不对称性——阈下传递的特征在教师模型生成的多样化数据集上保持一致表达，而任务特定能力和共享能力则各不相同——可以仅凭参数更新，在无监督条件下恢复该特征。由此得到的估计器会返回一个权重空间方向，放大该方向即可发现并解释该特征，或将其投影出去以抑制该特征。在阈下学习基准测试中，我们的方法在没有任何特征标签的情况下成功恢复了传递的动物偏好特征，并抑制了其表达。

    arXiv:2610.13536v1 Announce Type: cross  Abstract: When a language model is distilled from a teacher that shares its base parameters, the student can inherit behavioural traits of the teacher---including misalignment---even from data that is semantically unrelated to the trait (Cloud et al., 2026). All existing defences require knowing the trait. We remove this requirement and prove that a subliminally transferred trait can be recovered from the parameter updates alone, without supervision, by exploiting a single asymmetry: the trait is expressed consistently across diverse datasets generated by the teacher, whereas task-specific and shared capabilities vary. The resulting estimator returns a weight-space direction that can be amplified to discover and interpret the trait or projected out to suppress it. Applied to the subliminal-learning benchmarks of (Cloud et al., 2026), our method recovers the transmitted animal-preference traits without any trait labels and suppresses their expres
    
[^40]: MAGMA-Bench：在动态任务与执行失败下评估长时程具身智能体

    MAGMA-Bench: Evaluating Long-Horizon Embodied Agents under Dynamic Tasks and Execution Failures

    [https://arxiv.org/abs/2610.13534](https://arxiv.org/abs/2610.13534)

    提出了MAGMA-Bench——一个架构无关的基准，通过递增的任务时间跨度和受控在线扰动（如执行失败、任务中断、任务更新）评估动态多机器人环境中长时程具身智能体的持续任务推理与异构机器人协调能力，并发现即使最强的模型也难以应对这些挑战。

    

    长时程具身智能体必须维持与任务相关的信息，协调异构机器人，并随着执行的推进调整其决策。现有基准研究了动态任务执行和具身多智能体协调，但对其交叉领域的评估十分有限。我们提出了MAGMA-Bench，这是一个架构无关的基准，用于评估动态多机器人环境中的持续任务推理能力。MAGMA-Bench在逐步增长的任务时间跨度和受控在线扰动（包括执行失败、任务中断和任务更新）下评估智能体，其场景需要具有异构能力的机器人之间进行协调。该基准使用结构化的技能级接口，将任务级推理与感知及低级控制相分离。我们评估了可本地部署的开源权重规划器、多智能体状态跟踪策略以及前沿模型。即使是被评估的最强模型也在……（原文摘要在此处截断）

    arXiv:2610.13534v1 Announce Type: new  Abstract: Long-horizon embodied agents must maintain task-relevant information, coordinate heterogeneous robots, and adapt their decisions as execution unfolds. Existing benchmarks investigate dynamic task execution and embodied multi-agent coordination, but provide limited evaluation of their intersection. We introduce MAGMA-Bench, an architecture-agnostic benchmark for persistent task reasoning in dynamic multi-robot environments. MAGMA-Bench evaluates agents across increasing task horizons and controlled online perturbations, including execution failures, task interruptions, and mission updates, with scenarios requiring coordination across robots with heterogeneous capabilities. It uses a structured skill-level interface to separate task-level reasoning from perception and low-level control. We evaluate locally deployable open-weight planners, multi-agent state-tracking strategies, and frontier models. Even the strongest evaluated models fail o
    
[^41]: 超越域路由监督：面向多教师在线策略蒸馏的跨教师守护

    Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2610.13521](https://arxiv.org/abs/2610.13521)

    该论文提出Road MOPD方法，将未参与路由的教师模型作为辅助跨域批评者，通过词元级信用分配与梯度更新协调多教师监督，从而解决多教师在线策略蒸馏中的梯度冲突和能力退化问题。

    

    多教师在线策略蒸馏扩展了在线策略蒸馏，通过将特定领域的数据路由到专门的教师模型，在学生模型生成的轨迹上通过密集的词元级监督来整合各教师模型的能力。然而，域路由只是区分了不同领域的监督来源；来自所有领域的更新仍然作用于共享参数。当教师模型在响应风格和领域专长上存在显著差异时，单独使用时有效的监督在联合优化过程中仍可能导致持续的梯度冲突和能力退化。为应对这一挑战，我们提出了面向多教师在线策略蒸馏的跨教师守护方法（Road MOPD），该方法将未路由的教师模型引入为同一学生轨迹上的辅助跨域批评者，并通过词元级信用分配和梯度更新来协调多教师监督。具体而言，跨教师引导的信用分配……

    arXiv:2610.13521v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation (MOPD) extends on-policy distillation (OPD) by routing domain-specific data to specialized teachers, integrating their capabilities through dense token-level supervision on student-generated trajectories. However, domain routing only distinguishes the supervision sources for different domains; updates from all domains still act on shared parameters. When teachers differ substantially in response style and domain expertise, supervision that is effective in isolation may still lead to persistent gradient conflicts and capability degradation during joint optimization. To address this challenge, we propose Cross-Teacher Guarding for MOPD (Road MOPD), which introduces non-routed teachers as auxiliary cross-domain critics on the same student trajectories and coordinates multi-teacher supervision through token-level credit assignment and gradient updates. Specifically, cross-teacher-guided credit assignment 
    
[^42]: 因果抽象中的因果效应可识别性：一个严格的标准层次结构

    Causal Effect Identifiability in Causal Abstractions: A Strict Hierarchy of Criteria

    [https://arxiv.org/abs/2610.13520](https://arxiv.org/abs/2610.13520)

    本文在因果抽象所定义的图集合层面上研究因果效应可识别性，建立了各可识别性标准之间的严格层次结构，并证明一个效应可能在图集合上可识别，却不存在对所有兼容图一致成立的单一因果推导。

    

    从观测数据中识别因果效应通常需要一个完全指定的因果图，这一假设在复杂或高维环境中往往不切实际。因果抽象通过仅保留部分因果信息来解决这一局限，从而定义了与之兼容的因果图的等价类。在本文中，我们直接在这些图集合的层面上研究因果效应的可识别性，而非在单个因果图的层面上进行研究。我们在这一设定下形式化了可识别性的若干概念，并将它们组织成一个层次结构，以阐明它们之间的关系与差异。特别地，我们证明了某个因果效应可能在一个图集合上是可识别的，却不存在一个在所有兼容图上一致成立的单一因果推导。这些结果有助于揭示现有文献中的空白，并澄清其中的模糊之处。

    arXiv:2610.13520v1 Announce Type: new  Abstract: Identifying causal effects from observational data typically requires a fully specified causal graph, an assumption that is often unrealistic in complex or high-dimensional settings. Causal abstractions address this limitation by retaining only partial causal information, thereby defining equivalence classes of compatible causal graphs. In this paper, we study causal effect identifiability directly at the level of these graph collections, rather than at the level of individual causal diagrams. We formalize several notions of identifiability in this setting and organize them into a hierarchy that clarifies their relationships and differences. In particular, we establish that a causal effect may be identified over a collection of graphs without admitting a single causal derivation that holds uniformly across all compatible graphs. These results help reveal gaps and clarify ambiguities in the existing literature.
    
[^43]: RESCUE：这个警报值得采取行动吗？VLA运行时监控器的反事实评估

    RESCUE: Is This Alarm Worth Acting On? Counterfactual Evaluation of VLA Runtime Monitors

    [https://arxiv.org/abs/2610.13516](https://arxiv.org/abs/2610.13516)

    提出了RESCUE反事实评估框架，通过在执行检查点恢复模拟器和策略状态来评估VLA运行时监控器的警报是否能带来有效干预，并发现可恢复性取决于状态-干预对而非仅取决于机器人状态本身。

    

    机器人策略的运行时故障监控器通常根据其区分失败执行与成功执行的准确性来评估。然而，一个准确的警报可能到达得太晚而无法进行恢复，而不必要的干预可能会破坏原本会成功的执行。我们提出了RESCUE，一个用于评估监控器的警报是否能够实现有效干预的反事实框架。通过在执行检查点处恢复模拟器和策略状态，RESCUE比较替代动作，识别满足指定恢复标准的最新检查点，并评估根据警报采取行动是否能降低预期的任务完成成本。我们使用128,340个反事实分支，在LIBERO-Spatial和LIBERO-Plus数据集上结合OpenVLA-OFT和π0.5对RESCUE进行了评估。可恢复性取决于状态-干预对，而不仅仅取决于机器人状态本身：在τ=0.5时，66个OpenVLA-OFT/LIBERO-Plus失败案例中有14个包含可恢复的情形（原文在此处截断）。

    arXiv:2610.13516v1 Announce Type: cross  Abstract: Runtime failure monitors for robot policies are typically evaluated by how accurately they distinguish failed from successful executions. Yet an accurate alarm may arrive too late for recovery, while an unnecessary intervention may disrupt an execution that would otherwise succeed. We introduce RESCUE, a counterfactual framework for evaluating whether a monitor's alarms enable useful interventions. By restoring simulator and policy states at execution checkpoints, RESCUE compares alternative actions, identifies the latest checkpoint satisfying a specified recovery criterion, and evaluates whether acting on alarms reduces expected task-completion cost. We evaluate RESCUE with OpenVLA-OFT and $\pi_{0.5}$ over LIBERO-Spatial and LIBERO-Plus using 128,340 counterfactual branches. Recoverability depends on the state--intervention pair rather than on robot state alone: at $\tau=0.5$, 14 of 66 OpenVLA-OFT/LIBERO-Plus failures contain a recove
    
[^44]: 面向临床预后预测的检索增强异构证据融合多模态语言模型

    Retrieval-Augmented Multimodal Language Model with Heterogeneous Evidence Integration for Clinical Prognosis

    [https://arxiv.org/abs/2610.13510](https://arxiv.org/abs/2610.13510)

    该论文提出了PRISM，一种通过双模态人群记忆库在表示空间中检索相似病例嵌入、并融合异构临床证据的检索增强多模态语言模型，在包含1,443个任务的MDS-ED临床预后基准上，于诊断、病情恶化、ICU入院和死亡率预测等多项任务中取得了优异表现。

    

    临床预后预测需要整合结构化测量指标、生理波形、纵向病史以及人群背景信息。我们提出了PRISM（基于人群检索的结构化多模态预后推断），这是一种多模态语言模型，它结合了逐特征数值标记化、纵向心电图的近期加权聚合以及双模态人群记忆库。检索在表示空间中进行：模型接收相似训练队列就诊记录的均值池化嵌入，而任何单独的检索记录都不会进入提示词中。在包含1,443个任务的MDS-ED基准测试上，PRISM在诊断任务中达到86.2%的AUROC，在病情恶化预测中达到93.1%，在ICU入院预测中达到91.8%，在死亡率预测中达到92.8%。这些结果使用了96小时的到院前时间窗，而所报告的基线方法无法获得该时间窗的数据。仅使用基准时间窗内的输入时，PRISM达到0.8895的AUROC，相比之下UniPACT为0.8937，MDS-ED为0.8889。

    arXiv:2610.13510v1 Announce Type: new  Abstract: Clinical prognosis requires integrating structured measurements, physiological waveforms, longitudinal history, and population context. We introduce PRISM (Population-Retrieved Inference for Structured Multimodal Prognosis), a multimodal language model that combines per-feature numerical tokenization, recency-weighted aggregation of longitudinal ECGs, and a dual-modality population memory bank. Retrieval operates in representation space: the model receives mean-pooled embeddings of similar training-cohort encounters, while no individual retrieved record enters the prompt. On the 1,443-task MDS-ED benchmark, PRISM achieves AUROCs of 86.2% for diagnosis, 93.1% for deterioration, 91.8% for ICU admission, and 92.8% for mortality. These results use a 96-hour pre-arrival window unavailable to reported baselines. With benchmark-window inputs only, PRISM reaches 0.8895 AUROC, compared with 0.8937 for UniPACT and 0.8889 for MDS-ED. A matched-inpu
    
[^45]: 超越时间邻近性：面向强化学习中信用分配的解释性世界模型

    Beyond Temporal Proximity: Explanatory World Models for Credit Assignment in Reinforcement Learning

    [https://arxiv.org/abs/2610.13504](https://arxiv.org/abs/2610.13504)

    该论文提出解释性世界模型（EWM）框架，用语言模型生成的因果解释取代传统的时间邻近性来进行强化学习中的信用分配，通过保守分层算子递归重分配奖励并保持回合总回报不变，从而提升生成式推荐系统中长时程决策的样本效率。

    

    arXiv:2610.13504v1 公告类型：交叉 摘要：生成式推荐系统越来越多地采用强化学习（RL）来优化那些可能在决策发生数百甚至数千次交互之后才显现的结果。标准的强化学习依据时间邻近性而非显式相关性来传播奖励信号，这削弱了分配给那些距离遥远但至关重要的动作的信号，并成为样本效率低下的一个根源。我们提出了解释性世界模型，这是一个归因框架，利用解释来识别可能对最终已实现结果有贡献的早期事件。我们用一个前沿语言模型来实例化EWM，并通过一个保守的分层算子将其生成的解释用作学习信号，该算子在保持未折扣回合回报的前提下递归地重新分配奖励。为了探究什么样的解释才是有用的，我们受Halpern的实际因果理论、Lombrozo的解释简洁性理论以及De……（原文摘要在此处截断）

    arXiv:2610.13504v1 Announce Type: cross  Abstract: Generative Recommenders increasingly use Reinforcement Learning (RL) to optimize outcomes that may appear hundreds or thousands of interactions after a decision. Standard RL propagates reward according to temporal proximity rather than explicit relevance, attenuating the signal assigned to distant but important actions and creating a source of sample inefficiency. We introduce Explanatory World Models (EWM), an attribution framework that uses explanations to identify the earlier events that plausibly contributed to a realized outcome. We instantiate EWM with a frontier language model and use its explanations as learning signals using a conservative hierarchical operator that recursively redistributes reward while preserving the undiscounted episode return. To investigate what constitutes a useful explanation, we operationalize three complementary variants inspired by Halpern's actual causation, Lombrozo's explanatory simplicity, and De
    
[^46]: 人性化MLE智能体：超越单智能体极限的多智能体会话交替

    Humanize MLE Agents: Multi-Agent Session Alternation Beyond Single-Agent Limits

    [https://arxiv.org/abs/2610.13494](https://arxiv.org/abs/2610.13494)

    提出将多个智能体的会话交替协作建模为上下文重置半马尔可夫决策过程（CR-SMDP），通过工作流层面的跨会话协作突破单智能体的性能极限，从而克服长时程减速与框架过时问题。

    

    现有的基于大语言模型（LLM）的长时程机器学习工程（MLE）智能体方法通常采用上下文工程技术，通过自定义执行框架来管理工具、技能和积累的经验。然而，我们发现随着单次会话执行时间的延长，智能体的推理强度会下降且进展放缓，我们将其总结为“长时程减速”；同时，随着底层模型的改进，这些自定义框架的收益会逐渐消失，我们将其总结为“框架过时”。这些发现表明，此类框架的收益可能无法在不同模型版本间持续存在，这促使我们提出一种工作流层面的方法，让智能体在跨会话中相互承接彼此的工作成果。我们将这种交替进行的智能体会话建模为上下文重置半马尔可夫决策过程（CR-SMDP），建立了交替模式超越单智能体参考极限的充分条件，推导出明确的实验预算上界，并刻画了会话上限在何种条件下……（摘要截断）

    arXiv:2610.13494v1 Announce Type: new  Abstract: Existing LLM-based agent approaches for long-horizon machine learning engineering (MLE) often use context engineering, implemented through custom harnesses that manage tools, skills, and accumulated experience. However, we find that as single-session execution lengthens, the agent's reasoning intensity declines alongside slowing progress, summarized as long-horizon deceleration, and the benefits of these custom harnesses vanish as the underlying model improves, summarized as harness staleness. These findings suggest that their benefits may not persist across versions, which motivates a workflow-level approach in which agents build on one another's work across sessions. We model these alternating agent sessions as a context-reset semi-Markov decision process (CR-SMDP) and establish sufficient conditions for alternation to exceed single-agent reference limits, derive an explicit experiment-budget bound, and characterize when session caps p
    
[^47]: 分支在线策略蒸馏

    Branching On-Policy Distillation

    [https://arxiv.org/abs/2610.13481](https://arxiv.org/abs/2610.13481)

    BranchOPD通过学生不确定性和师生分歧选择信息丰富的决策点进行分支重采样，对主轨迹和分支同时施加token级教师监督，并借助共享前缀复用将生成token成本降低约19%-31%。

    

    在线策略蒸馏（OPD）在学生模型生成的轨迹上提供教师监督，但单一轨迹在每个决策点只能实现一种延续，这在有限采样预算下限制了监督的覆盖范围。我们提出了BranchOPD（分支在线策略蒸馏），它利用学生不确定性和师生分歧来选择信息丰富的决策点，从共享前缀重新采样学生策略的延续，并对主轨迹和分支同时施加token级别的教师监督。BranchOPD在数学任务中作用于token级别的决策，在智能体任务中作用于回合内的决策。与七条独立轨迹相比，共享前缀复用在ALFWorld上降低了约19%、在WebShop上降低了约31%的学生生成token成本。我们还引入了协议边界修复（PBR）方法，它在协议约束下使用针对性监督来缓解重复生成和无效生成的问题。

    arXiv:2610.13481v1 Announce Type: new  Abstract: On-policy distillation (OPD) provides teacher supervision on student-generated trajectories, but a single trajectory realizes only one continuation at each decision, limiting supervision coverage under a finite sampling budget. We propose BranchOPD (Branching On-Policy Distillation), which selects informative decision points using student uncertainty and teacher-student disagreement, resamples student-policy continuations from shared prefixes, and applies token-level teacher supervision to both main trajectories and branches. BranchOPD operates at token-level decisions in mathematics and within-turn decisions in agent tasks. Shared-prefix reuse reduces student-generated token costs by approximately 19% on ALFWorld and 31% on WebShop relative to seven independent trajectories. We also introduce protocol-boundary repair (PBR), which uses targeted supervision under protocol constraints to mitigate repetition and wasted generation. BranchOPD
    
[^48]: 一种用于生成边缘案例场景以支持安全保证的本地智能体AI框架

    A Local Agentic AI Framework to Generate Edge Case Scenarios for Safety Assurance

    [https://arxiv.org/abs/2610.13479](https://arxiv.org/abs/2610.13479)

    该论文提出了一种基于笔记本电脑的本地智能体AI框架，利用本地语言模型通过自主的“思考-行动-观察-更新-继续”循环自适应生成长尾边缘案例场景，用于自主系统的安全保证，同时兼具低成本、完全可控和数据隐私安全的优势。

    

    生成边缘案例场景对于自主系统的训练、测试和安全保证至关重要。这可以识别危险故障以提高系统的可靠性和鲁棒性，确保系统在其运行设计域的边界上仍能安全运行，并学习这些边缘案例下预期的系统行为。语言模型生成和分析能力的快速发展扩大了其适用范围，最近的研究工作已经探索了基于语言模型的场景生成。然而，许多近期的方法依赖于人在回路或基于规则的策略，这限制了语言模型生成多样化和边缘案例场景的能力，即分布的长尾部分。我们的工作在基于笔记本电脑的智能体AI框架中使用本地语言模型进行自适应场景生成。这种方法成本低廉、完全可控，并确保数据隐私和安全。它采用思考、行动、观察、更新和继续的循环机制（配合轻量级组件……摘要在此处被截断）

    arXiv:2610.13479v1 Announce Type: new  Abstract: Generating edge case scenarios is crucial for training, testing and safety assuring autonomous systems. This identifies hazardous failures to increase system reliability and robustness, ensures the system can still function safely on the boundaries of its operational design domain and learns the expected system behaviour for these edge cases. The rapid advancement in the generation and analysis capabilities of language models (LMs) has expanded their applicability and recent work has investigated LM scenario generation. However, many recent approaches rely on human-in-the-loop or rule-based strategies. This restricts the LM's ability to generate diverse and edge case scenarios; the long tail of the distribution. Our work uses local LMs in a laptop-based agentic AI framework for adaptive scenario generation. This is cheap, fully-controlled, ensures data privacy and security. It uses think, act, observe, update, and continue loops with lig
    
[^49]: 当多模态大语言模型读不懂人类能读的文字：诊断并学习阅读“眯眼文字”

    When MLLMs miss what humans read: diagnosing and learning to read squint text

    [https://arxiv.org/abs/2610.13465](https://arxiv.org/abs/2610.13465)

    该论文发现多模态大模型存在“眯眼文字”识别缺陷，通过构建 SquintBench 与 SquintCtrl 两个基准诊断出空间选择是关键瓶颈，并据此提出 Squint OPSD 方法教会模型阅读此类隐藏文字。

    

    多模态大语言模型（MLLM）可能无法读取嵌入在图像粗略结构中的文字（即人类通过眯眼就能辨认的文字），并错误地报告图像中不存在任何文字。我们将这类文字命名为“眯眼文字”，并将其作为一种识别缺陷加以研究，研究动机来自生产级内容审核流水线中的实际案例。我们构建了 SquintBench，包含 1076 张源自生产环境的审核图像，以及 SquintCtrl——一个用于机制诊断的受控合成测试集。冻结权重分析表明，隐藏的文字会留下粗略的信号，而字形身份的表征较弱；当提供正确的空间区域时，字形身份的可获取性会显著提高，这表明空间选择是主要瓶颈。在固定视觉 token 预算下，抑制高频内容也能提升阅读效果，同时视觉 token 的数量也可能产生影响。这些发现促使我们提出 Squint OPSD：一个冻结的教师模型接收缩小尺寸的视图和真实转录文本作为先验（原文摘要在此处截断）……

    arXiv:2610.13465v1 Announce Type: cross  Abstract: Multimodal LLMs can fail to read text embedded in an image's coarse structure (text a human recovers by squinting) and report that no text is present. We name this class Squint Text and study it as a recognition deficiency, motivated by cases from a production moderation pipeline. We introduce SquintBench, comprising 1076 production-derived moderation images, and SquintCtrl, a controlled synthetic suite for mechanism diagnosis. Frozen-weight analysis shows that hidden text leaves a coarse signal while glyph identity is weakly represented; identity becomes substantially more accessible when the correct spatial regions are supplied, implicating spatial selection as a major bottleneck. Suppressing high-frequency content also improves reading at a fixed visual-token budget, while visual token count may also contribute. These findings motivate Squint OPSD: a frozen teacher receives a downscaled view and the ground-truth transcription as pri
    
[^50]: SqueezeBench：抗饱和的数学能力评估

    SqueezeBench: Saturation-Resilient Evaluation of Mathematical Capability

    [https://arxiv.org/abs/2610.13461](https://arxiv.org/abs/2610.13461)

    SqueezeBench提出了一种抗饱和的数学能力评估基准，要求AI不断提出并用Lean机器验证证明整数函数的上下界记录，使每次成功都提高门槛而非消除问题。

    

    人工智能系统在数学领域正取得快速进展，但这种进展难以衡量。针对已解决问题的基准会饱和，而在开放问题的基准中，每个问题只能被首次解决一次。我们提出了SqueezeBench，这是一个对饱和具有韧性的基准：一次成功会提高门槛，而不是移除问题。每个问题固定一个整数函数 $f(n)$，其取值仅在较小的 $n$ 时已知，该基准为它维护一个下界记录和一个上界记录。记录是一个可计算的Lean函数，并附带机器验证的证明表明它约束着 $f$。一次提交需要提出一个新函数，并证明它约束着 $f$、在任何 $n$ 处都不差于当前记录、且在某个 $n$ 处严格更优。一旦被接受，它就成为下一次提交必须超越的记录。因此，该系统既选择了界，也选择了其证明，并且可以在单个 $n$ 处改进记录，或改进其增长率……

    arXiv:2610.13461v1 Announce Type: new  Abstract: AI systems are making rapid progress in mathematics, but this progress is hard to measure. Benchmarks of solved problems saturate, and in benchmarks of open problems each problem can be resolved for the first time only once. We present SqueezeBench, a benchmark that is resilient to saturation: a success raises the bar instead of removing the problem. Each problem fixes an integer function $f(n)$ whose values are known only for small $n$, and the benchmark keeps a lower and an upper record on it. A record is a computable Lean function with a machine-checked proof that it bounds $f$. A submission proposes a new function and proves that it bounds $f$, that it is no worse than the current record at any $n$, and that it is strictly better at some $n$. Once accepted, it becomes the record that the next submission has to beat. The system thus chooses the bound as well as its proof, and it may improve the record at a single $n$ or its growth rat
    
[^51]: 智能体化大型语言模型

    Agentic Large Language Models

    [https://arxiv.org/abs/2610.13455](https://arxiv.org/abs/2610.13455)

    本书探讨了智能体化大型语言模型如何将LLM的语言推理能力与现实世界中的行动和交互能力相结合。

    

    大型语言模型（LLMs）已经风靡全球。它们能够用我们的语言与我们交流，并对我们的世界进行推理，这使它们成为写作、回答问题等众多任务的首选工具。它们的回答已经在影响我们的决策和行动。如果大型语言模型能够直接在世界中行动，会发生什么？如果大型语言模型之间能够相互交互，又会发生什么？智能体化大型语言模型将大型语言模型的能力与现实世界中的行动能力结合在一起。在本书中，我们将讨论智能体化大型语言模型如何工作、如何行动、如何交互，以及这对我们作为用户意味着什么。

    arXiv:2610.13455v1 Announce Type: new  Abstract: Large language models (LLMs) have taken the world by storm. Their ability to communicate with us in our own language, and to reason about our world, has made them a favorite tool for writing, answering questions, and much more. Their answers already influence our decisions and our actions. What would happen if LLMs themselves could act in the world directly? What would happen if LLMs could interact with each other? Agentic LLMs combine the power of large language models with the ability to act in the real world. In this book we will discuss how agentic LLMs work, how they act, how they interact, and what this means for us, as users.
    
[^52]: 集体自动驾驶车辆路径规划需要社会意识才能与人类驾驶员共存

    Collective Autonomous Vehicle Routing Needs Social Awareness for Coexistence with Human Drivers

    [https://arxiv.org/abs/2610.13439](https://arxiv.org/abs/2610.13439)

    在自动驾驶车辆的奖励中引入衡量其对他车影响的边际成本，能让混合交通中的人类驾驶员与自动驾驶车辆双方都获益，从而实现更和谐的人机共存。

    

    在真实交通网络中，集体自动驾驶车辆（AV）的路径选择可能对人类驾驶车辆（HDV）产生外部性，因为有利于自动驾驶车辆的路线可能会增加其他道路使用者的出行时间。为解决这一问题，我们在自动驾驶车辆的自私出行时间奖励中引入了一个社会性成分——边际成本。边际成本通过将其他道路使用者的出行时间与反事实模拟（即移除该自动驾驶车辆后的模拟）中的出行时间进行对比，来刻画该自动驾驶车辆对其他道路使用者的影响。我们使用多智能体强化学习（MARL）研究混合自动驾驶交通环境下的自动驾驶车辆路径选择问题，其中自动驾驶车辆通过与交通环境的反复交互学习路由策略。在英戈尔施塔特和普罗万两个真实路网上的实验表明，将边际成本纳入自动驾驶车辆奖励不仅限制了人类驾驶车辆出行时间的增加，相较于自私路由策略还缩短了自动驾驶车辆自身的出行时间，同时提升了整体系统性能。

    arXiv:2610.13439v1 Announce Type: cross  Abstract: Collective Autonomous Vehicle (AV) route choices in real-world traffic networks can create externalities for Human-Driven Vehicles (HDVs), as routes that benefit AVs may increase travel times for other road users. To address this, we introduce a social component, the marginal cost, to the selfish travel time AV reward. Marginal cost captures an AV's impact on other road users by comparing their travel times with those in a counterfactual simulation in which the AV is removed.   We study AV route choice in mixed-autonomy traffic settings using Multi-Agent Reinforcement Learning (MARL), where AVs learn routing policies through repeated interactions with the traffic environment. Experiments on the real-world networks of Ingolstadt and Provins show that incorporating marginal cost into the AV reward not only limits the increase in HDV travel times and shortens AV travel times relative to selfish AV routing, but also improves overall system
    
[^53]: 展开时干预：解锁超越策略极限的智能体强化学习

    Rollout-Time Intervention: Unlocking Agentic Reinforcement Learning Beyond Policy Limits

    [https://arxiv.org/abs/2610.13438](https://arxiv.org/abs/2610.13438)

    提出展开时干预（RTI）技术，在轨迹采样阶段利用奖励侧证据（评估量规）将错误轨迹转化为成功轨迹，为超出模型能力边界的任务提供有效学习信号，突破智能体强化学习的同策略推理瓶颈。

    

    强化学习（RL）已成为提升大语言模型（LLM）智能体能力的关键范式，使其在交互环境中具备更强的推理、规划和工具使用能力。然而，现有的智能体强化学习方法从根本上受到同策略推理瓶颈的制约：当任务超出模型当前的能力边界时，有效轨迹变得极其稀疏，训练常常发生崩溃。我们提出了展开时干预，这是一种旨在诱导产生超出策略内在探索能力的成功轨迹的技术。RTI 通过在展开（rollout）过程中利用奖励侧证据（即评估量规），将错误轨迹转化为成功轨迹，为策略自身很少能独立解决的任务提供有效的学习信号。尽管干预产生的轨迹本质上是偏离策略的，但我们发现当轨迹（原文摘要在此处被截断）……

    arXiv:2610.13438v1 Announce Type: new  Abstract: Reinforcement learning (RL) has emerged as a key paradigm for improving large language model (LLM) agents, enabling stronger reasoning, planning, and tool-use in interactive environments. However, existing agent RL methods remain fundamentally constrained by an on-policy reasoning bottleneck: when tasks exceed the model's current capability boundary, effective trajectories become extremely sparse and training often collapses. We introduce Rollout-Time Intervention (RTI), a technique designed to elicit successful trajectories beyond the policy's intrinsic exploration capacity. RTI operates by leveraging reward-side evidence (i.e., evaluation rubrics) during rollout to convert erroneous trajectories into successful ones, providing effective learning signals for tasks that the policy rarely solves on its own. Although intervention trajectories are inherently off-policy, we find that they can effectively guide policy optimization when trajec
    
[^54]: 面向高亮度大型强子对撞机（HL-LHC）异常检测的学习型事件表征的局部诊断方法

    Local Diagnostics of Learned Event Representations for Anomaly Detection at the HL-LHC

    [https://arxiv.org/abs/2610.13437](https://arxiv.org/abs/2610.13437)

    该论文在高亮度LHC条件下模拟的7.5亿事件数据集COLLIDE-2V上，评估了监督与自监督神经嵌入策略，并以学习潜空间中自编码器的重构误差作为异常评分，强调必须对被标记事件进行局部诊断，以区分新物理与建模误差或探测器效应，并为后续定向搜索识别可观测量。

    

    异常检测，即识别偏离标准模型预期的碰撞事件，是在大型强子对撞机（LHC）上进行新物理搜索中最具前景的模型无关策略之一。在学习得到的低维表征中搜索稀有事件，可以免去事先依赖手工设计的可观测量，但也牺牲了这些可观测量所提供的直接物理解释。因此，要使异常检测能够进入发现流程，必须对被标记的事件进行局部诊断：一方面要区分潜在的新物理与错误建模或探测器效应，另一方面要识别出可用于后续定向搜索的可观测量。我们在COLLIDE-2V数据集上评估了监督式和自监督式神经嵌入策略，该数据集是在高亮度LHC真实条件下模拟的、包含7.5亿事件的大规模数据集，并使用在学习的潜空间中训练的自编码器的重构误差作为异常评分。

    arXiv:2610.13437v1 Announce Type: new  Abstract: Anomaly detection, the identification of collision events that deviate from Standard Model expectations, is one of the most promising model-agnostic strategies for new-physics searches at the Large Hadron Collider. Searching for rare events in learned low-dimensional representations removes the need to commit to hand-crafted observables a priori, but sacrifices the immediate physical interpretation those observables provide. For anomaly detection to enter a discovery pipeline, flagged events must therefore be diagnosed locally, both to distinguish potential new physics from mismodeling or detector effects and to identify observables for targeted follow-up searches. We evaluate supervised and self-supervised neural embedding strategies on COLLIDE-2V, a 750-million-event dataset simulated under realistic High-Luminosity LHC conditions, using the reconstruction error of an autoencoder trained in the learned latent space as the anomaly score
    
[^55]: 采用Sigmoid标签的CNN-Transformer实现高分辨率亚度级波达方向（DOA）估计

    CNN-Transformer with Sigmoid Label for High-Resolution Sub-Degree DOA Estimation

    [https://arxiv.org/abs/2610.13433](https://arxiv.org/abs/2610.13433)

    提出将信源DOA编码为角度偏移单调sigmoid函数之和的sigmoid标签，配合CNN-Transformer网络和0.25度网格步长，实现了亚度级精度和1度分辨率，突破了基于分类的DOA估计的网格量化误差限制。

    

    本文针对基于深度学习的波达方向（DOA）估计中的网格量化误差问题展开研究。基于分类的DOA方法将连续角度空间离散化为网格，因此采用基于峰值的读出方式时，精度和分辨率受到网格步长和标签峰值宽度的限制。为解决这一问题，我们提出了一种sigmoid标签，将信源的DOA编码为角度偏移的单调sigmoid函数之和。对使用sigmoid标签训练的CNN-Transformer网络的输出进行求导，可在信源DOA处产生尖锐的谱峰。结合适度细化的0.25度网格步长，采用所提出sigmoid标签的CNN-Transformer实现了亚度级精度和1度分辨率，突破了网格量化误差的限制。在统一的CNN-Transformer骨干网络下，我们将所提出的Sigmoid标签与高斯标签、0-1硬标……（原文摘要在此处截断）

    arXiv:2610.13433v1 Announce Type: cross  Abstract: This paper addresses the grid quantization error of deep-learning (DL)-based direction-of-arrival (DOA) estimation. Classification- based DOA methods discretize the continuous angular space into a grid, so that with a peak-based read-out, the accuracy and resolution are limited by the grid step and by the width of the label peak. To address this issue, we propose a sigmoid label that encodes the sources' DOAs as the sum of angle-shifted monotonic sigmoid functions. Differentiating the output of a CNN-Transformer network trained with the sigmoid label produces sharp spectral peaks at the source DOAs. Combined with a moderately refined grid step of 0.25 degrees, the CNN-Transformer with the proposed sigmoid label attains sub-degree accuracy and a 1-degree resolution, breaking the limit of the grid quantization error. Under the unified CNN-Transformer backbone, we compare the proposed Sigmoid label with the Gaussian label, the 0-1 hard la
    
[^56]: Web$hell：智能体网页访问的统一接口

    Web$hell: A Unified Interface for Agentic Web Access

    [https://arxiv.org/abs/2610.13416](https://arxiv.org/abs/2610.13416)

    提出Web$hell——一个统一搜索增强、浏览器与文件系统等多种访问模式的智能体网页访问接口，使智能体能够在单条任务轨迹内同时完成信息收集、浏览器操作和证据回溯。

    

    大型语言模型（LLM）智能体正在成为开放网页的直接用户。随着它们承担越来越复杂、信息密集的任务，有效的网页访问正成为智能体问题求解的核心能力。现有方法通过不同的访问模式推进了智能体网页访问，包括搜索增强型智能体、基于浏览器的智能体，以及近期的基于文件系统的智能体。然而，随着用户期望智能体完成长时程、端到端的任务，它们越来越需要在同一条任务轨迹内进行广泛的信息收集、执行浏览器操作，并重新审视已获取的证据。我们的初步分析表明，现有访问模式难以满足这些需求：它们主要只支持信息收集或浏览器交互二者之一，而将两者能力结合起来往往伴随巨大的成本或性能折衷。为此，我们提出了Web$hell，一个面向智能体网页访问的统一接口。

    arXiv:2610.13416v1 Announce Type: new  Abstract: Large language model (LLM) agents are becoming direct users of the open web. As they take on increasingly complex and information-intensive tasks, effective web access is becoming a core capability for agentic problem solving. Existing approaches have advanced agentic web access through different access models, including search-augmented agents, browser-based agents, and, more recently, filesystem-based agents. Nevertheless, as users expect agents to complete long-horizon, end-to-end tasks, they increasingly need to conduct broad information gathering, perform browser actions, and revisit acquired evidence within a single trajectory. Yet our preliminary analyses reveal that existing access models struggle to meet these demands. They primarily support either information gathering or browser interaction, while combining their capabilities can involve substantial costs or performance trade-offs. To this end, we present Web\$hell, an interfa
    
[^57]: 面向系统辨识的上下文学习：概率化建模、长序列处理与基准验证

    In-Context Learning for System Identification: Probabilistic Formulation, Long-Sequence Handling, and Benchmark Validation

    [https://arxiv.org/abs/2610.13411](https://arxiv.org/abs/2610.13411)

    本文提出一种统一的概率化上下文学习元辨识架构，将基于Transformer的编码器-解码器与长序列处理机制相结合，实现长时域校准预测，并通过基准测试验证了其有效性。

    

    传统的系统辨识通常包含两个步骤：首先，对训练数据进行处理以估计目标系统的模型；然后，利用该模型对新输入查询进行输出预测。相比之下，近期的上下文学习方法学习一个元模型，以零样本方式直接从训练数据和查询输入映射到输出，从而绕过显式的模型估计。该方法利用在大量合成生成数据上离线训练的元模型，来模拟系统辨识算法的行为。在本工作中，我们从三个方向推进了这一面向系统辨识的元学习框架。首先，我们开发了一个统一的概率化元辨识架构，该架构将基于Transformer的编码器-解码器与专为处理长训练序列设计的机制相结合，实现了长时域上的校准预测。其次，我们展示了如何利用预训练的……（摘要原文在此处截断）

    arXiv:2610.13411v1 Announce Type: cross  Abstract: Traditional system identification typically involves two steps: initially, training data is processed to estimate a model of the system of interest; then, the model is used to predict outputs for new input queries. In contrast, recent in-context learning approaches learn a meta model that maps directly from training data and query inputs to outputs in a zero-shot fashion, bypassing explicit model estimation. This approach leverages a meta model trained offline, on large sets of synthetically generated data, to emulate the behavior of a system identification algorithm. In this work, we advance this meta-learning framework for system identification in three directions. First, we develop a unified probabilistic meta-identification architecture that combines a Transformer-based encoder-decoder with tailored mechanisms for handling long training sequences, enabling calibrated predictions over long horizons. Second, we show how a pre-trained
    
[^58]: RegulaTutor：一个基于理论的长程个性化学习框架

    RegulaTutor: A Theory-Informed Framework for Long-Term Personalized Learning

    [https://arxiv.org/abs/2610.13399](https://arxiv.org/abs/2610.13399)

    提出基于自我调节学习理论的RegulaTutor框架，通过CAMM（认知、情感、元认知、动机）机制将学习者历史证据转化为有据可依的教学策略并随新观察持续更新，实现长程个性化辅导。

    

    长期个性化辅导要求语言模型能够将学习者不断演进的历史记录转化为适当的教学支持。现有基于大语言模型的辅导系统使用记忆和检索技术来个性化响应，但将历史证据与教学策略联系起来的教学理论依据并不总是明确的。我们提出了RegulaTutor，这是一个面向长期个性化辅导的基于理论的框架，它将历史证据、学习者解读和面向学习者的教学三者连接起来。在自我调节学习（SRL）研究的指导下，其辅导框架实现了一个涵盖认知、情感、元认知和动机（CAMM）的框架，用于推导有证据支持的学习者解读和教学策略。所得策略保留与其支持证据的联系，并随着新观察结果的到来而持续更新。这一分阶段的工作流程还通过PairRoute支持系统性改进，PairRoute是一种离线处理……

    arXiv:2610.13399v1 Announce Type: new  Abstract: Long-term personalized tutoring requires language models to translate a learner's evolving history into appropriate instructional support. Existing LLM-based tutoring systems use memory and retrieval to personalize responses, but the pedagogical rationale linking historical evidence to teaching strategies is not always explicit. We introduce RegulaTutor, a theory-informed framework for long-term personalized tutoring that connects historical evidence, learner interpretation, and learner-facing instruction. Guided by self-regulated learning (SRL) research, its tutoring harness operationalizes a cognitive, affective, metacognitive, and motivational (CAMM) framework for deriving evidence-supported learner interpretations and teaching policies. The resulting policies retain links to their supporting evidence and are updated as new observations arrive. This staged workflow also supports systematic refinement through PairRoute, an offline proc
    
[^59]: ProgramGym：面向长周期程序复现智能体的可扩展训练环境

    ProgramGym: Scaling Training Environments for Long-Horizon Program Reproduction Agents

    [https://arxiv.org/abs/2610.13395](https://arxiv.org/abs/2610.13395)

    ProgramGym提出了一个由Program-Env和Program-Agent组成的框架，通过反馈驱动的循环从代码仓库自动构建可验证的训练环境，实现了对长周期程序复现智能体的可扩展训练。

    

    arXiv:2610.13395v1 公告类型：cross 摘要：编程智能体正越来越多地被用于像软件工程师那样从零开始构建程序，而不是解决代码仓库中的问题或根据给定的需求文档生成仓库。然而，能够促使智能体自主探索所需功能并决定系统设计方式的、可扩展且可验证的训练环境仍然稀缺。此外，现有的智能体工作流也无法支撑长周期的开发过程，即智能体在实现之前先对参考程序进行探索。为了应对这些挑战，我们提出了ProgramGym，这是一个由Program-Env和Program-Agent组成的框架，旨在为长周期程序复现智能体提供可扩展的训练环境。具体而言，Program-Env通过一个反馈驱动的循环从代码仓库构建可验证的训练环境，该循环能够修复运行时沙箱构建问题并提升黑盒测试的质量。利用这一新颖的流水线，我们获得了1,648个训练……（摘要不完整，原文在此处截断）

    arXiv:2610.13395v1 Announce Type: cross  Abstract: Coding agents are increasingly used to build programs from scratch as software engineers do rather than to resolve issues within a repository or to generate a repository from a given requirements document. However, scalable verifiable training environments that force agents to discover what is required and to decide how the system should be designed remain scarce. Moreover, existing agent workflows also cannot sustain long-horizon development in which the agent explores the reference program before implementation. To address these challenges, we introduce ProgramGym, a framework consisting of Program-Env and Program-Agent for scalable training environments of long-horizon program reproduction agents. Specifically, Program-Env constructs verifiable training environments from repositories through a feedback-driven loop that repairs runtime sandbox builds and the quality of black-box tests. Using this novel pipeline, we obtain 1,648 train
    
[^60]: AgentSWE：编码智能体能构建出你真正想要的智能体吗？

    AgentSWE: Can Coding Agents Build the Agent You Actually Want?

    [https://arxiv.org/abs/2610.13361](https://arxiv.org/abs/2610.13361)

    提出了首个检验编码智能体能否忠实构建出真正满足用户需求的智能体的基准 AgentSWE，其以“委托”形式覆盖创建、编辑、优化三个开发阶段共25个任务，并通过保留用例和需求驱动的严格评分标准进行验收。

    

    随着软件的生产逐渐交由软件本身来完成，编码智能体越来越多地被委托去构建贴合用户自身工作流的智能体。这种委托隐含着一个未被言明的假设：交付的智能体将忠实服务于它被构建时所要满足的需求，而不仅仅是表面上看起来如此。目前尚无任何基准测试检验这一假设。我们提出 AgentSWE，一个面向智能体软件工程的基准，涵盖智能体开发生命周期的三个阶段：根据自然语言需求创建完整智能体、编辑生产级智能体代码库，以及朝向可度量目标进行优化。其25个任务中每一个都被设计为一次“委托”而非一场“考试”：构建者仅针对少量示例用例进行迭代，而验收环节则在保留用例上对冻结提交的成果进行评估，这些保留用例旨在覆盖完整需求，并依据源自需求的评分标准以严格的通过/……（原文摘要在此处截断）

    arXiv:2610.13361v1 Announce Type: cross  Abstract: As the production of software passes into the hands of software itself, coding agents are increasingly commissioned to build agents fitted to their users' own workflows. This delegation carries a quiet assumption, that the delivered agent will faithfully serve the requirement it was built for, rather than merely appear to. No existing benchmark tests that assumption. We introduce AgentSWE, a benchmark for agent software engineering spanning three stages of the agent development lifecycle: Creation of a complete agent from a natural-language requirement, Editing of production agent codebases, and Optimization toward a measurable target. Each of its 25 tasks is framed as a commission rather than an exam: the builder iterates against a handful of illustrative cases, while acceptance exercises the frozen submission on held-out cases designed to span the full requirement, grading behavior against requirement-derived rubrics under hard pass/
    
[^61]: SWE-Critix：用于评估编码智能体轨迹的后训练通用验证器

    SWE-Critix: Post-Training Generalist Verifiers for Assessing Coding-Agent Trajectories

    [https://arxiv.org/abs/2610.13356](https://arxiv.org/abs/2610.13356)

    提出了SWE-Critix后训练流水线，用于训练能够可靠评判和解释编码智能体在长时程软件工程任务中执行轨迹的通用验证器。

    

    编码智能体正越来越多地被部署用于端到端的软件工程任务，然而在这些任务中，往往缺乏一个现成的、能够定义成功标准的“神谕”评估器。这促使人们需要对编码智能体进行验证，以便可靠地评判和解释其行为。然而，验证并非易事，因为编码智能体解决长时程软件工程任务的能力日益增强。它们的执行轨迹可能极其漫长，包含与代码仓库和外部工具的数百次迭代交互。现有的验证器要么依赖未经专门训练的通用大语言模型（LLM），要么训练主要专注于产生标量奖励的专用评论模型，这凸显了基于训练的通用验证方法的探索仍相对有限。我们提出了SWE-Critix，这是一个用于训练编码智能体轨迹通用验证器的后训练流水线。给定一个任务及其执行轨迹……（注：原摘要不完整，此处仅翻译可用部分）

    arXiv:2610.13356v1 Announce Type: new  Abstract: Coding agents are increasingly deployed for end-to-end software engineering tasks, where a readily available oracle evaluator that defines the success criteria is often absent. This motivates verifications for coding agents that can reliably judge and explain their behavior. However, verification is non-trivial as coding agents are increasingly capable of solving long-horizon software engineering tasks. Their execution trajectories can be extremely long, containing hundreds of iterative interactions with code repositories and external tools. Existing verifiers either rely on general-purpose LLMs without specialized training or train dedicated critics that predominantly focus on producing scalar rewards, highlighting the relatively limited exploration of training-based generalist verification. We introduce SWE-Critix, a post-training pipeline for training generalist verifiers of coding-agent trajectories. Given a task and its execution tr
    
[^62]: AlphaNIST：从二元任务奖励中涌现的装配技能

    AlphaNIST: Emergent Assembly Skills from Binary Task Rewards

    [https://arxiv.org/abs/2610.13354](https://arxiv.org/abs/2610.13354)

    受AlphaGo Zero启发，该研究仅用稀疏的二元任务奖励从零开始训练单一策略，在仿真中无需演示或密集奖励即可完成NIST装配任务板1号全部20个装配任务，达到98.2%的训练成功率。

    

    NIST装配任务板1号（NIST Assembly Task Board #1）仍然是机器人操作领域基于学习方法的一个具有挑战性的基准。在短时间尺度上，它要求对小型、紧密配合的零件进行精确且富含接触的控制；在长时间尺度上，它需要在延长的动作序列中进行恢复与协调。演示数据、任务分解和密集的进度奖励通过提供“如何装配”的指导推动了研究进展。但是，策略中究竟有多少必须被显式规定？从AlphaGo到AlphaGo Zero的演进表明，大规模强化学习可以在不规定如何获胜的情况下发现超越人类的策略。AlphaNIST探究的是：稀疏的二元任务奖励，以及这种任务无关奖励所能容纳的任务多样性，是否同样能让机器人自行发现如何进行装配。在仿真中，我们从零开始训练一个单一策略，涵盖NIST装配任务板1号上的全部20个装配任务，在无演示的情况下达到了98.2%的总体训练成功率。

    arXiv:2610.13354v1 Announce Type: new  Abstract: The NIST Assembly Task Board #1 remains a challenging benchmark for learning-based methods in robotic manipulation. Over short horizons, it demands precise, contact-rich control of small, tight-fitting parts. Over long horizons, it requires recovery and coordination across extended action sequences. Demonstrations, task decomposition, and dense progress rewards have enabled research advances by providing guidance on how to assemble. But how much of the strategy must be specified? The progression from AlphaGo to AlphaGo Zero showed that RL at scale can discover superhuman strategies without prescribing how to win. AlphaNIST asks whether a sparse binary task reward and task diversity admitted by this task-agnostic reward can similarly let a robot discover how to assemble. In simulation, we train a single policy from scratch across all 20 assembly tasks on the NIST Assembly Task Board #1, reaching 98.2% overall training success without demo
    
[^63]: 面向工具集成化学推理的掩码扩散语言智能体

    Masked Diffusion Language Agents for Tool-Integrated Chemical Reasoning

    [https://arxiv.org/abs/2610.13330](https://arxiv.org/abs/2610.13330)

    本文提出首个基于掩码扩散语言模型的化学推理智能体ChemDiffAgent，将化学推理建模为对推理与工具使用轨迹的迭代去噪过程，使工具使用决策能够在全轨迹上下文中联合确定，突破了传统自回归范式在双向分子理解和多源证据整合上的局限。

    

    为复杂化学任务设计工具集成的推理智能体系统仍然是一个根本性的挑战。尽管近期的化学智能体通过集成外部工具展现出有前景的性能，但它们仍建立在自回归范式之上，这种范式本质上受限于因果注意力和从左到右的生成方式。因此，它们在双向分子理解、全局协调的工具规划以及多源证据整合方面常常表现不佳。为应对这一挑战，我们提出了ChemDiffAgent，这是首个用于工具集成化学推理的掩码扩散语言智能体。它将化学推理重新表述为对交错的推理与工具使用轨迹的迭代去噪过程，使得工具使用决策能够在完整轨迹的上下文中联合确定，从而更好地将智能体的决策边界与其知识边界对齐。我们进一步分析了如何……（原文摘要在此处截断）

    arXiv:2610.13330v1 Announce Type: new  Abstract: Designing tool-integrated reasoning agent systems for complex chemical tasks remains a fundamental challenge. Although recent chemical agents have demonstrated promising performance through the integration of external tools, they are still built on the autoregressive paradigm, which is inherently constrained by causal attention and left-to-right generation. As a result, they often struggle with bidirectional molecular understanding, global coordinated tool planning and multi-source evidence integration. To address this challenge, we introduce ChemDiffAgent, the first masked diffusion language agent for tool-integrated chemical reasoning. It reformulates chemical reasoning as iterative denoising over interleaved reasoning and tool-use trajectories, enabling tool-use decisions to be jointly determined under full-trajectory context and thereby better aligning the agent's decision boundary with its knowledge boundary. We further analyze how 
    
[^64]: 无需对齐的一致性：无标签的评分标准澄清提升了LLM裁判间的共识，却未提升与人类评判的一致性

    Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans

    [https://arxiv.org/abs/2610.13303](https://arxiv.org/abs/2610.13303)

    该论文提出一种无标签的评分标准澄清程序，通过将裁判间分歧转化为对争议子标准的重写修复，显著提升了LLM评审团内部的一致性，但发现这种共识的提升并不能转化为与人类评判更高的一致性。

    

    LLM裁判越来越多地以评审团的形式部署，而评审团成员之间的分歧通常被解读为评分标准不清晰的征兆。我们测试了一种无标签的程序，将这种分歧转化为评分标准的修复。该程序将每个评分标准分解为子标准，根据在开发集上裁判间的分歧程度对子标准进行排序，并让一个编辑模型仅重写争议最大的子标准。在我们的评估中，将可文本检查的子标准迁移至确定性检查器的附加阶段没有迁移任何子标准。人类标签从不用于选择或编辑条款；它们仅用于评估结果。我们在一个版本控制的协议中预先指定了四个假设，并使用来自三个提供商的五名裁判组成的评审团，在SummEval、HealthBench元评估和USR Topical-Chat上评估了六种评分标准条件，总计进行了129,770次裁判请求（不含重试）。在SummEval上，该程序提高了裁判间的Krippendorff alpha系数……（摘要原文在此处截断）

    arXiv:2610.13303v1 Announce Type: cross  Abstract: LLM judges are increasingly deployed as panels, and disagreement among panel members is often read as a symptom of an unclear rubric. We test a label-free procedure that turns this disagreement into rubric repairs. The procedure decomposes each rubric into sub-criteria, ranks them by cross-judge disagreement on a development split, and has an editor model rewrite only the most contested ones. An additional stage that moves text-checkable sub-criteria to deterministic checkers transferred none in our evaluation. Human labels are never used to select or edit clauses; they are used only to evaluate the outcome. We pre-specified four hypotheses in a version-controlled protocol and evaluated six rubric conditions with a panel of five judges from three providers on SummEval, the HealthBench meta-evaluation, and USR Topical-Chat, for a total of 129,770 judge requests, excluding retries. On SummEval, the procedure raised inter-judge Krippendor
    
[^65]: PRIME：面向结构化文档检索的检索前索引建模增强

    PRIME: Pre-Retrieval Index Modeling Enhancement for Structured Document Retrieval-Real

    [https://arxiv.org/abs/2610.13296](https://arxiv.org/abs/2610.13296)

    PRIME提出了一个索引侧优化框架，通过LLM驱动的语义重构与确定性的元数据合成及分层检索两条互补轨道，提升结构化文档检索质量并降低嵌入计算成本。

    

    对外部知识库的高精度检索对于LLM智能体的高效运行至关重要。然而，现有的信息检索（IR）研究主要优化查询侧的检索器（如ColBERT、BGE）和生成器（如Self-RAG、GraphRAG），而将语料库侧的表示简化为分块处理，这使得索引侧质量成为一个未被充分探索的优化杠杆。我们提出了检索前索引建模与增强（PRIME），这是一个索引侧优化框架，建立在两条互补的技术轨道上：一条是由LLM驱动的语义重构（SR）轨道，将长文档转换为结构化、信息密集的章节；另一条是确定性的元数据合成（MS）与分层检索（HR）轨道，提取语义锚点和标题结构，实现由粗到细的章节检索。我们的评估表明，确定性元数据在检索质量上可与结构感知分块相媲美，同时削减了嵌入计算成本。

    arXiv:2610.13296v1 Announce Type: cross  Abstract: High-precision retrieval over external knowledge bases is essential for LLM agents to operate effectively. However, existing Information Retrieval (IR) research primarily optimizes query-side retrievers (e.g., ColBERT, BGE) and generators (e.g., Self-RAG, GraphRAG), while reducing corpus-side representation to chunking. This leaves index-side quality an underexplored lever. We propose Pre-Retrieval Index Modeling & Enhancement (PRIME), an index-side optimization framework built on two complementary tracks: an LLM-driven Semantic Restructuring (SR) track that transforms long documents into structured, information-dense sections, and a deterministic Metadata Synthesis (MS) and Hierarchical Retrieval (HR) track that extracts semantic anchors and heading structure for coarse-to-fine section retrieval. Our evaluations show that deterministic metadata matches structure-aware chunking's retrieval quality while cutting embedding computation by
    
[^66]: BoundaryEvo：面向长链推理蒸馏的能力边界处数据-模型协同演化

    BoundaryEvo: Data-Model Co-Evolution at the Capability Boundary for Long-Chain Reasoning Distillation

    [https://arxiv.org/abs/2610.13285](https://arxiv.org/abs/2610.13285)

    BoundaryEvo提出一种两阶段框架，先帮助学生模型建立基础推理能力，再使训练数据随学生模型的能力边界协同演化，从而在仅有少量教师轨迹的条件下高效提升小模型的长链推理能力。

    

    从大型教师模型中蒸馏长链推理轨迹是提升小型语言模型推理能力的有效方法。然而，现有方法通常在蒸馏前选取固定的训练集，忽略了推理样本的有用性会随着学生模型的演进而变化这一事实。当只能使用少量教师轨迹进行训练时，这种不匹配的代价尤为显著。此外，学生模型已经掌握的样本几乎不提供新的监督信号，而远超学生能力范围的样本则可能难以从中学习。因此，训练数据应当随模型能力的演化而自适应调整，但当学生模型初始能力较弱时，可靠的数据选择十分困难。为应对这些挑战，我们的核心思想是首先建立基础推理能力，然后使训练数据适配学生模型不断演化的能力边界。我们提出BoundaryEvo，一个两阶段……（摘要至此截断）

    arXiv:2610.13285v1 Announce Type: new  Abstract: Distilling long-chain reasoning trajectories from large teacher models provides an effective way to improve the reasoning ability of small language models. However, existing methods typically select a fixed training set before distillation, overlooking that the usefulness of a reasoning example changes as the student evolves. This mismatch becomes especially costly when only a small number of teacher trajectories can be used for training. Besides, examples already mastered provide little new supervision, while examples far beyond the capability of student may be difficult to learn from. Training data should therefore adapt as model capability evolves, yet reliable data selection is difficult when the student is initially weak. To address these challenges, our key idea is to first establish the basic reasoning capability and then adapt the training data to the evolving capability boundary of the student. We propose BoundaryEvo, a two-stag
    
[^67]: Agent4Spec：一个基于大语言模型的智能体系统，通过自动化验证与市场分析优化API规范

    Agent4Spec: An LLM-based Agentic System to Optimize API Specifications through Automated Validation and Market Analysis

    [https://arxiv.org/abs/2610.13273](https://arxiv.org/abs/2610.13273)

    Agent4Spec提出了一个自主多智能体框架，通过验证流水线自动检测并纠正OpenAPI语法错误，同时通过市场分析流水线开展竞争研究、识别与行业领先者的功能差距并自动生成缺失的API规范功能，实现了API规范的持续优化与演进。

    

    API规范是现代软件开发的关键文档，直接影响集成效率和客户端代码生成。尽管近期的研究进展已利用大语言模型（LLM）从源代码生成初始规范，但现有方法将这一过程视为一次性事件，忽视了对这些工件进行持续验证、长期维护以及市场驱动演进的需求。为解决这一问题，我们提出了Agent4Spec，这是一个自主的多智能体框架，包含两条不同的流水线：（1）验证流水线，用于检测并自我纠正OpenAPI语法错误；（2）市场分析流水线，用于开展竞争性研究、识别与行业领先者相比的功能差距，并自动生成缺失的API规范功能。我们在两个精心整理的API规范数据集上对Agent4Spec进行了评估，涵盖通用型和行业特定型的API规范。

    arXiv:2610.13273v1 Announce Type: cross  Abstract: API specifications serve as critical documentation for modern software development, directly impacting integration efficiency and client code generation. While recent advancements have leveraged Large Language Models (LLMs) to generate initial specifications from source code, existing approaches treat this process as a one-time event - overlooking the needs for continuous validation, long-term maintenance, and market-driven evolution of these artifacts. To address this issue, we present Agent4Spec, an autonomous multi-agent framework featuring two distinct pipelines: (1) a validation pipeline that detects and self-corrects OpenAPI syntax errors, and (2) a market analysis pipeline that conducts competitive research, identifies functional gaps against industry leaders, and automatically generates missing API-spec functionalities. We evaluate Agent4Spec on two curated datasets of API specifications spanning both general-purpose and indust
    
[^68]: 遗忘并非总是坏事：一种针对被投毒LLM智能体的神经启发记忆修复机制

    Forgetting is Not Always Bad: A Neuro-Inspired Memory Repair Mechanism for Poisoned LLM Agents

    [https://arxiv.org/abs/2610.13267](https://arxiv.org/abs/2610.13267)

    本文提出受神经科学启发的无需训练记忆修复框架DREAM，通过对被投毒记忆进行选择性功能性遗忘（动态再激活、异常诊断与记忆修复三阶段），实现LLM智能体对记忆投毒攻击的内在高效防御。

    

    大型语言模型（LLM）智能体极易受到记忆投毒攻击。现有防御方法主要依赖外部模块进行静态记忆隔离或持续在线审计，导致记忆效率低下且计算开销高昂。为解决这些局限，受神经科学机制启发——弱重新激活可使记忆变得不稳定，并在缺乏强化的情况下引发记忆衰减——我们提出了一种智能体内在的、无需训练的记忆修复框架DREAM（Dynamic Reactivation for Engram Attenuation in Memory，用于记忆印迹衰减的动态再激活），能够对被投毒记忆实现选择性功能性遗忘。具体而言，DREAM实现了三阶段流程：通过扰动诱导的隐式再激活来揭示恶意记忆的结构脆弱性，基于多维激活模式的自适应异常诊断来检测拓扑异常，以及一个选择性修复被污染记忆的动态记忆修复模块。

    arXiv:2610.13267v1 Announce Type: new  Abstract: Large language model (LLM) agents are highly vulnerable to memory-poisoning attacks. Existing defenses primarily rely on external modules for either static memory isolation or continuous online auditing, resulting in low memory efficiency and high computational overhead. To address these limitations, inspired by neuroscientific mechanisms that weak reactivation can destabilize memories and induce decay in the absence of reinforcement, we propose an agent-intrinsic training-free memory repair framework, DREAM (Dynamic Reactivation for Engram Attenuation in Memory), that enables selective functional forgetting of poisoned memories. Specifically, DREAM implements a three-stage pipeline: perturbation-induced implicit reactivation to reveal the structural fragility of malicious memories, adaptive anomaly diagnosis based on multi-dimensional activation patterns to detect topological anomalies, and a dynamic memory repair module that selectivel
    
[^69]: 表格基础模型实际上是浅层的

    Tabular Foundation Models Are Effectively Shallow

    [https://arxiv.org/abs/2610.13253](https://arxiv.org/abs/2610.13253)

    表格基础模型（TFMs）实际上是“浅层”的——其 transformer 块中高达 96% 可以被一个闭式、无需标签的仿射转换器替换而不损失性能，说明这类模型存在大量架构冗余，可以被大幅简化。

    

    基于上下文学习的表格基础模型（TFMs）特别适用于资源受限或存在隐私约束的场景，因为它们无需额外训练即可解决新任务。然而，其架构由多个 transformer 块组成，导致高昂的计算和内存开销。与此同时，梯度提升决策树（GBDTs）等传统替代方法仍极具竞争力，并且可以在 CPU 上高效运行，这表明表格任务可能本质上并不需要深层架构。这种对比引出了一个根本性问题：能否通过消除架构冗余来简化 TFMs，而不牺牲性能？为了回答这个问题，我们提出了一种闭式、无需标签的仿射转换器，它可以替换连续窗口中高达 96% 的块，而其余部分保持不变。在 171 个分类数据集上对八个前沿 TFMs 进行评估后，我们发现表格 transformer 模型存在严重的参数冗余。

    arXiv:2610.13253v1 Announce Type: cross  Abstract: In-context-learning Tabular Foundation Models (TFMs) are uniquely suited to scenarios with limited resources or privacy constraints, as they solve new tasks without additional training. However, their architecture consists of multiple transformer blocks, resulting in high computational and memory footprints. Meanwhile, traditional alternatives like Gradient-Boosted Decision Trees (GBDTs) remain highly competitive and run efficiently on CPUs, suggesting that tabular tasks may not inherently require deep architectures. This contrast raises a fundamental question: can TFMs be simplified by removing architectural redundancies without sacrificing performance? To answer this, we introduce a closed-form, label-free affine translator that replaces a contiguous window of up to 96% of the blocks, leaving the rest unchanged. Evaluating eight frontier TFMs on 171 classification datasets, we reveal that tabular transformers are vastly overparameter
    
[^70]: AMR解析器的输出有多可靠？面向AMR解析的不确定性估计

    How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing

    [https://arxiv.org/abs/2610.13236](https://arxiv.org/abs/2610.13236)

    该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。

    

    近期的进展显著提升了抽象意义表示（AMR）解析器的性能，然而判断其输出何时可以被信任这一问题在很大程度上仍未被探索。本工作首次在黑盒设置下，利用top-k束搜索候选结果对AMR解析器开展不确定性估计研究。不确定性通过成对候选相似度来估计，并使用预测拒绝率进行评估。本工作刻画了解析性能与不确定性之间的关系，评估了束候选分布的可靠性，分析了评估指标对不确定性评估的影响，并识别出在性能与可靠性之间实现最佳平衡的解析器。实验结果表明，不确定性估计作为评估AMR解析器的补充维度具有重要价值。

    arXiv:2610.13236v1 Announce Type: cross  Abstract: Recent advances have significantly improved the performance of Abstract Meaning Representation (AMR) parsers, yet determining when their outputs can be trusted remains largely unexplored. This work presents the first study of uncertainty estimation for AMR parsers in a black-box setting using top-k beam-search candidates. Uncertainty is estimated from pairwise candidate similarities and evaluated using the Prediction Rejection Ratio. This work characterises the relationship between parsing performance and uncertainty, evaluates the reliability of beam candidate distributions, analyses the effect of evaluation metrics on uncertainty assessment, and identifies parsers that provide the strongest balance between performance and reliability. Experimental results show the value of uncertainty estimation as a complementary dimension for evaluating AMR parsers.
    
[^71]: ReSFlow：在统计流形上校正类别流以实现一步生成

    ReSFlow: Rectifying Categorical Flows on Statistical Manifolds for One-Step Generation

    [https://arxiv.org/abs/2610.13225](https://arxiv.org/abs/2610.13225)

    提出ReSFlow模型，通过在统计流形上校正类别流使其轨迹逼近测地线，实现离散数据的一步生成，大幅降低推理计算成本。

    

    在统计流形上学习连续向量场，将简单先验分布传输至目标离散分布，极大地推动了离散数据的生成建模。然而，由于这些连续时间流不受沿统计流形测地线运动的约束，因此需要进行多步数值积分来模拟，导致推理阶段产生高昂的计算成本。在本工作中，受连续欧几里得空间中校正流“直而快”特性的启发，我们提出了校正统计流模型，该模型通过显式学习遵循内蕴黎曼几何的测地线轨迹，实现了统计流形上的一步类别生成。通过在统计流形上校正流，我们从理论上保证了先验分布与目标分布之间的轨迹逼近测地线。

    arXiv:2610.13225v1 Announce Type: cross  Abstract: Generative modeling for discrete data has been significantly advanced by learning continuous vector fields on the statistical manifolds to transport simple priors to target discrete distributions. However, because these continuous-time flows are not constrained to follow geodesics on the statistical manifold, thereby requiring multi-step numerical integration for simulation and leading to substantial computational costs during inference. In this work, drawing inspiration from the "straight and fast" properties of Rectified Flow in continuous Euclidean space, we propose the Rectified Statistical Flow (ReSFlow) model, which enables one-step categorical generation on statistical manifolds by explicitly learning geodesic trajectories that adhere to the intrinsic Riemannian geometry. By rectifying the flow on the statistical manifold, we theoretically guarantee that trajectories between prior and target distributions approximate geodesics. 
    
[^72]: 超越动作：为视觉-语言-动作模型学习未来操作目标

    Beyond Actions: Learning Future Operation Targets for Vision-Language-Action Models

    [https://arxiv.org/abs/2610.13212](https://arxiv.org/abs/2610.13212)

    该论文提出动作-未来耦合效果学习，让视觉-语言-动作模型以未来物理效果而非动作本身作为主要预测目标，从而提升机器人策略的性能，在DexJoCo上达到50.97%的总体成功率。

    

    arXiv:2610.13212v1 公告类型：交叉 摘要：动作规定了机器人应该执行什么，但没有描述这些指令将会实现什么效果。然而，视觉-语言-动作（VLA）策略目前主要通过从图像、语言和状态预测动作块来学习。我们提出了动作-未来耦合效果学习，该方法使基于未来的物理效果成为策略的主要预测目标。一个离线分词器从演示动作、机器人实际响应和稀疏视觉过渡中构建按时间排序的效果标记。重建目标鼓励该表示保留可执行的细节和观察到的物理变化。一个流匹配VLA从当前图像、指令和机器人状态预测效果，而状态条件解码器则将这些效果转换为动作块。特定于本体的接口使单臂和双臂任务能够共享同一个效果模型。在DexJoCo上，AFCE实现了50.97%的总体成功率，相比……有所提升。

    arXiv:2610.13212v1 Announce Type: cross  Abstract: Actions specify what a robot should execute, but do not describe what those commands will achieve. Vision-language-action (VLA) policies nevertheless learn primarily by predicting action chunks from images, language, and state. We introduce Action--Future Coupled Effect Learning (AFCE), which makes a future-grounded physical Effect the policy's primary prediction target. An offline tokenizer constructs temporally ordered Effect tokens from demonstrated actions, realized robot responses, and sparse visual transitions. Reconstruction objectives encourage the representation to retain executable detail and observed physical change. A flow-matching VLA predicts Effects from current images, instruction, and robot state, while a state-conditioned decoder converts them into action chunks. Embodiment-specific interfaces allow single- and dual-arm tasks to share the Effect model. On DexJoCo, AFCE achieves $50.97\%$ overall success, improving ove
    
[^73]: 在部分可观测强化学习中，学习循环网络真的重要吗？

    Does Learning Recurrent Networks Matter in Partially Observable RL?

    [https://arxiv.org/abs/2610.13197](https://arxiv.org/abs/2610.13197)

    该研究发现冻结（无需训练）的循环网络即可为部分可观测强化学习提供数百步的记忆能力，并在多样化的POMDP任务中取得与训练循环网络相当的性能，从而质疑了在部分可观测RL中学习循环网络的必要性。

    

    部分可观测条件下的强化学习（RL）通常依赖循环网络来编码时间信息，循环模块往往与策略或价值函数联合训练。这引出了一个自然的问题：学习循环网络真的有必要吗？在本文中，我们研究了冻结的循环网络能否为求解部分可观测马尔可夫决策过程（POMDP）提供足够的记忆能力。受储备池计算和广义储备池计算理论基础的启发，我们研究了一种将固定循环网络与可训练的非线性输出模块相结合的简单架构。我们在具有不同记忆需求、观测模态和动作空间的各种任务上对该方法进行了评估。令人惊讶的是，冻结的循环网络能够支持数百步的记忆回溯，并在多样化的POMDP任务中取得具有竞争力的控制性能……

    arXiv:2610.13197v1 Announce Type: cross  Abstract: Reinforcement learning (RL) under partial observability typically re- lies on recurrent networks to encode temporal information, with the recurrent module trained jointly with the policy or value function. This raises a natural question: is learning the recurrent network actually necessary? In this paper, we investigate whether a frozen recurrent network can provide sufficient memory for solving par- tially observable Markov decision processes (POMDPs). Motivated by the theoretical foundations of Reservoir Computing (RC) and Generalized Reservoir Computing (GRC), we study a simple archi- tecture that combines a fixed recurrent network with a trainable nonlinear output module. We evaluate this approach across tasks with varying memory demands, observation modalities, and action spaces. Surprisingly, frozen recurrent networks support recall over hundreds of steps and achieve competitive control performance across diverse POMDPs, while us
    
[^74]: JANUS：利用共同演化的世界模型增强长时程自主研究

    JANUS: Enhancing Long-Horizon Autonomous Research with Co-Evolving World Models

    [https://arxiv.org/abs/2610.13194](https://arxiv.org/abs/2610.13194)

    JANUS提出了一种由共同演化的世界模型驱动的“预测-验证-演化”循环框架，通过在执行前预测候选方案的价值提升、信息增益和探索效用来缓解长时程自主研究中的反馈延迟问题，并将反馈经验蒸馏为不断演化的技能。

    

    长时程自主研究通过持续的假设生成、实现和验证循环，在加速科学进步方面展现出了巨大的潜力。然而，这种“生成-执行-反馈”范式依赖于被动的试错机制，其中代价高昂且耗时的执行过程会引入严重的反馈延迟，在爆炸式扩展的搜索空间下阻碍了演化进程。为应对这一挑战，我们提出了JANUS，这是一个通过由共同演化的世界模型驱动的“预测-验证-演化”循环，将未来结果预测与过往经验反思统一起来的框架。在一个不断演化的图中，研究智能体提出候选方案，而世界模型通过预测价值提升、信息增益和探索效用对其进行评分，然后在执行前相应地选择、精炼或剪枝分支。在接收到真实反馈后，JANUS将经验蒸馏为不断演化的技能：层次化……（摘要在此处截断）

    arXiv:2610.13194v1 Announce Type: new  Abstract: Long-horizon autonomous research has shown promise in accelerating scientific progress through sustained cycles of hypothesis generation, implementation, and validation. However, this Generate-Execute-Feedback paradigm relies on reactive trial-and-error, where costly and time-consuming executions introduce severe feedback delays, throttling evolutionary progress under explosively expanding search space. To address this challenge, we propose JANUS, a framework that unifies future outcomes forecasting and past experience reflection through a Predict-Verify-Evolve loop driven by a co-evolving world model. Within an evolving graph, the research agent proposes candidates while world model scores them by forecasting value improvement, information gain, and exploration utility, then accordingly selects, refines, or prunes branches prior to execution. After receiving ground-truth feedback, JANUS distills experience into evolving skills: hierarch
    
[^75]: AdaC-GRPO：通过自适应课程学习掌握长程视觉数独谜题

    AdaC-GRPO: Towards Mastering Long-Horizon Visual Sudoku Puzzles via Adaptive Curriculum Learning

    [https://arxiv.org/abs/2610.13184](https://arxiv.org/abs/2610.13184)

    提出AdaC-GRPO方法，通过在线rollout准确率追踪模型的“最近发展区”来动态构建训练批次，从而克服长程视觉数独任务中因奖励稀疏导致标准RLVR失效的问题。

    

    虽然可验证奖励的强化学习（RLVR）能够增强大语言模型的推理能力，但其在长程多模态任务中的有效性仍未得到充分探索。视觉数独谜题构成了一项代表性的挑战，需要对局部约束和全局依赖链进行多步演绎推理。为此，我们提出了FG-Sudoku，这是一个按网格大小和所需演绎步骤分类的细粒度基准。评估结果显示，前沿的多模态大语言模型（MLLM）在处理复杂的远程依赖时会失败（例如9*9数独）。在后训练过程中，这种接近于零的成功率导致严重的奖励稀疏问题，使标准RLVR方法停滞不前。为克服这一问题，我们提出了自适应课程组相对策略优化。与僵化的或基于先验的课程安排不同，AdaC-GRPO通过在线rollout准确率追踪模型的“最近发展区”，从而动态构建训练批次。实验表明，AdaC-

    arXiv:2610.13184v1 Announce Type: new  Abstract: While reinforcement learning with verifiable reward (RLVR) enhances reasoning in large language models, its efficacy in long-horizon multimodal tasks remains underexplored. Visual Sudoku puzzles pose a representative challenge, requiring multi-step deductive reasoning over both local constraints and global dependency chains. To this end, we introduce FG-Sudoku, a fine-grained benchmark categorized by grid size and required deductive steps. Evaluations reveal that frontier MLLMs fail on complex long-range dependencies (e.g., 9*9 puzzles). During post-training, this near-zero success rate causes severe reward sparsity, halting standard RLVR. We propose Adaptive Curriculum Group Relative Policy Optimization (AdaC-GRPO) to overcome this. Unlike rigid or prior-based schedules, AdaC-GRPO dynamically constructs training batches by tracking the model's "Zone of Proximal Development" via online rollout accuracy. Experiments demonstrate that AdaC-
    
[^76]: RoleWorld：基于功能角色抽象的世界建模知识迁移

    RoleWorld: Knowledge Transfer in World Modeling with Functional Role Abstractions

    [https://arxiv.org/abs/2610.13172](https://arxiv.org/abs/2610.13172)

    RoleWorld提出一种以功能角色抽象（可复用原型与环境特定参数值）为核心组织世界动力学知识的符号世界建模方法，使世界模型能够高效迁移到具有相似交互结构的新环境中。

    

    世界模型使大语言模型（LLM）智能体能够预判动作效果并做出明智决策。为了实现泛化，理想情况下这类世界模型应当能够高效迁移到具有相似交互结构和底层动力学的新环境中。近期的方法将世界动力学编码为符号规则或可执行程序，其组合结构在原则上支持迁移，但这些方法仍然围绕环境特定的对象来组织此类知识。为解决这一局限性，我们提出了RoleWorld，这是一种符号世界建模方法，它使用可复用的原型和环境特定的参数值作为功能角色抽象，来表示对象在交互中的行为方式。从交互轨迹中，一个教师LLM提取角色原型以及定义在角色实例上的可执行转移规则，同时还提取一个引导行动者LLM在决策中使用角色信息的技能。

    arXiv:2610.13172v1 Announce Type: new  Abstract: World models enable large language model (LLM) agents to anticipate action effects and make informed decisions. For generalization, it is desirable for such world models to transfer efficiently to new environments with similar interaction structure and underlying dynamics. Recent approaches encode world dynamics as symbolic rules or executable programs, whose compositional structure in principle affords transfer, but still organize such knowledge around environment-specific objects. To address this limitation, we introduce RoleWorld, a symbolic world modeling approach which uses reusable prototypes and environment-specific parameter values as functional role abstractions to represent how objects behave in interactions. From interaction trajectories, a teacher LLM extracts role prototypes and executable transition rules defined over role instances, together with a skill that guides the actor LLM to use role information in decision making.
    
[^77]: 质量，而不仅仅是结果：用于选择编程智能体轨迹进行监督微调的结构化评分框架

    Quality, Not Just Outcome: A Structural Scoring Framework for Selecting Coding-Agent Trajectories for Supervised Fine-Tuning

    [https://arxiv.org/abs/2610.13152](https://arxiv.org/abs/2610.13152)

    提出TQS轨迹质量评分框架，直接基于轨迹结构评估过程质量，无需LLM评判且线性时间运行，用于筛选高质量编程智能体轨迹以提升监督微调效果。

    

    开放编程智能体通常通过监督微调进行训练，其使用的交互轨迹需满足最终补丁通过基准测试套件的条件。然而，这种结果层面的过滤标准忽略了过程质量：最终成功的轨迹可能仍包含低效的交互模式，例如重复的文件检查、不稳定的工具使用或延迟的修改行为，而这些都会成为模仿目标的一部分。我们提出了TQS（轨迹质量分数，Trajectory Quality Score），一个针对编程智能体轨迹的结构化评分框架，它直接作用于轨迹结构，无需LLM评判者，并以线性时间运行。TQS通过围绕“结构化收敛”组织的结构化信号来刻画轨迹质量：信息聚焦、意图-行动对齐以及策略动态。利用来自公开开放许可语料库的155,144条成功轨迹，我们根据TQS对轨迹进行排序并微调编程智能体。

    arXiv:2610.13152v1 Announce Type: new  Abstract: Open coding agents are commonly trained through supervised fine-tuning on interaction trajectories whose final patches pass the benchmark test suite. However, this outcome-level filtering criterion ignores process quality: trajectories that eventually succeed may still contain inefficient interaction patterns such as repetitive file inspection, unstable tool usage, or delayed modification behavior, all of which become part of the imitation target. We propose TQS (Trajectory Quality Score), a structural scoring framework for coding-agent trajectories that operates directly on trajectory structure, requires no LLM judge, and runs in linear time. TQS characterizes trajectory quality through structural signals organized around structured convergence: information focus, intent--action alignment, and policy dynamics. Using 155,144 successful trajectories from a public open-license corpus, we rank trajectories by TQS and fine-tune coding agents
    
[^78]: VIGOR：场景程序中可验证的工业几何优化与修复

    VIGOR: Verifiable Industrial Geometry Optimization and Repair in Scene Programs

    [https://arxiv.org/abs/2610.13145](https://arxiv.org/abs/2610.13145)

    该论文识别了大语言模型生成的场景程序中“可执行但不工程可行”的鸿沟，提出VIGOR框架结合基于执行的验证与分阶段几何修复，并构建了包含1,749个场景和每场景94项几何检查的工业场景程序数据集，实现工业布局的可验证优化与自动修复。

    

    大语言模型（LLM）日益能够根据自然语言指令生成结构化布局和可执行的场景程序。然而，程序可执行并不意味着工程可行性：一个工业场景程序可能成功执行，但其最终状态仍会违反耦合边界、碰撞、间隙、通道或结构等约束。我们将这种不匹配定义为“可执行性与工程可行性之间的鸿沟”，即局部看似合理的设备摆放可能全局不可行，因为对某一设备的修正可能在其他位置引入新的违规。为了在可控环境中使这一鸿沟可度量，我们构建了一个基于执行的工业场景程序数据集，包含1,749个可构建场景、完整的可执行程序，以及每个场景94项最终状态几何检查。我们进一步提出VIGOR，它将监督式场景程序生成与基于执行的验证和分阶段几何修复相结合……

    arXiv:2610.13145v1 Announce Type: new  Abstract: Large language models (LLMs) increasingly generate structured layouts and executable scene programs from natural-language instructions. However, executability does not imply engineering feasibility: an industrial scene program may execute successfully while its final state still violates coupled boundary, collision, clearance, aisle, or structural constraints. We identify this mismatch as the executability-engineering-feasibility gap, where locally plausible placements can remain globally infeasible because corrections to one device may introduce violations elsewhere. To make this gap measurable in a controlled setting, we construct an execution-grounded industrial scene-program dataset with 1,749 constructible scenarios, complete executable programs, and 94 final-state geometric checks per scenario. We further propose VIGOR, which combines supervised scene-program generation with execution-grounded verification and staged geometric refi
    
[^79]: 面向基于大语言模型的多智能体系统的能力感知编排

    Competence-Aware Orchestration for LLM-Based Multi-Agent Systems

    [https://arxiv.org/abs/2610.13144](https://arxiv.org/abs/2610.13144)

    该论文提出一种两阶段能力感知编排框架，通过建模大语言模型骨干在当前查询上的能力，自适应地调整多智能体系统中的角色激活与通信图，在六个基准测试上取得平均93.81%的成绩。

    

    基于大语言模型的多智能体系统（MAS）已成为通过角色专业化与协作来增强复杂推理能力的一种有前景的范式，但一个关键挑战仍然存在：如何使协作适配骨干模型在当前查询上的能力水平？这个问题之所以重要，是因为更多的协作并不总是更好——更丰富的协调可能有助于困难案例，但也会带来额外的 token 成本和冗余交互。我们提出了一个用于自动 MAS 编排的两阶段能力感知框架。该框架首先对大语言模型骨干在当前查询上的任务条件化能力进行建模，然后利用这一信号来调整角色激活和通信图。在六个基准测试上，我们的方法取得了平均 93.81% 的得分。使用六个能力水平各异的大语言模型骨干进行的实验进一步表明，我们的方法能够根据推断出的能力来调整协作：对于能力较高的骨干……（原文在此处截断）

    arXiv:2610.13144v1 Announce Type: new  Abstract: LLM-based multi-agent systems (MAS) have emerged as a promising paradigm for enhancing complex reasoning through role specialization and collaboration, yet a key challenge remains: how can collaboration be adapted to a backbone's competence on the current query? This problem matters because more collaboration is not always better--richer coordination may help on difficult cases, but it also incurs additional token cost and redundant interactions. We propose a two-stage competence-aware framework for automatic MAS orchestration. It first models the LLM backbone's task-conditioned competence for the current query, and then uses this signal to adapt role activation and the communication graph. Across six benchmarks, our method achieves an average score of 93.81%. Experiments with six LLM backbones of varying competence levels further show that our method adjusts collaboration according to the inferred competence: for higher-competence backb
    
[^80]: DP-MIND：差分隐私记忆引导的长上下文推理

    DP-MIND: Differentially private memory-guided Long context Inference

    [https://arxiv.org/abs/2610.13132](https://arxiv.org/abs/2610.13132)

    提出DP-MIND框架，通过本地扰动输入并利用能保留任务相关信息的紧凑记忆机制逐块处理序列，将差分隐私推理成功扩展到百万token级别的长上下文场景。

    

    arXiv:2610.13132v1 公告类型：新 摘要：LLM即服务（LLM-as-a-Service）已成为无需本地部署即可访问大语言模型的主流范式。然而，当用户查询可能包含敏感或机密上下文时，这一过程会引发严重的隐私问题。差分隐私（DP）为保护此类输入提供了一种有原则的方法，但现有的DP推理方法主要针对短上下文或中等长度上下文场景设计，由于上下文容量有限以及DP扰动下严重的效用下降，它们难以处理百万token规模的输入。在本文中，我们提出DP-MIND，一种记忆引导的框架，将差分隐私推理扩展到百万token级别的长上下文。DP-MIND通过在本地对输入进行扰动，并利用一个紧凑的记忆机制逐块处理输入，从而实现私有的长上下文推理，该记忆机制能够在整个序列中保留与任务相关的信息。为了使该记忆对DP噪声具有鲁棒性，我们优化了记忆构建……（原文在此处截断）

    arXiv:2610.13132v1 Announce Type: new  Abstract: LLM-as-a-Service has become a dominant paradigm for accessing large language models without requiring local deployment. However, the process raises severe privacy concerns when user queries may include sensitive or confidential contexts. Differential privacy (DP) provides a principled approach to protecting such inputs, but existing DP inference methods are primarily designed for short- or moderate-context settings and struggle with million-token-scale inputs due to limited context capacity and severe utility degradation under DP perturbation.   In this paper, we propose DP-MIND, a memory-guided framework that scales differentially private inference to million-token contexts. DP-MIND enables private long-context inference by locally perturbing the input and processing it chunk by chunk with a compact memory that preserves task-relevant information across the sequence. To make this memory robust to DP noise, we optimize the memory constru
    
[^81]: 瞻前顾后：面向路网轨迹的预期性生成建模

    Looking Ahead and Back: Anticipatory Generative Modeling for Road-Network Trajectories

    [https://arxiv.org/abs/2610.13131](https://arxiv.org/abs/2610.13131)

    提出了一种新的图约束生成范式“预期性生成建模”，将路网轨迹生成从位置模拟转向行为建模，通过预期未来路线来解决路口转向选择问题，避免轨迹偏离路网。

    

    轨迹生成是交通管理和城市规划等城市应用的基础。在道路网络中，车辆沿路段的运动在很大程度上是确定性的，而在交叉路口的转向决策则至关重要——多条出路口的路径可能看起来同样合理，但它们却可能通向截然不同的下游路线。现有的生成方法（如扩散模型）仅模拟位置序列，难以完成下一步转向选择，甚至会产生偏离道路网络的轨迹。为填补这一空白，我们主张一个根本性的转变：从位置模拟转向道路网络上的行为建模。在本文中，我们从生成式路径规划的视角重新审视轨迹生成，引入了一种新的图约束生成范式——预期性生成建模，通过结合对未来路线的预期来解决下一步转向选择问题。

    arXiv:2610.13131v1 Announce Type: new  Abstract: Trajectory generation underpins urban applications such as traffic management and urban planning. Within road networks, vehicle movement along road segments is largely deterministic, whereas transitions at intersections are critical--multiple outgoing paths may appear equally plausible, yet they can lead to substantially different downstream routes. Existing generative methods such as diffusion models simulate location sequences and fail short of next transition selection, even producing trajectories that drift off the road network. To fill this gap, we advocate a fundamental shift: moving beyond location simulation toward behavior modeling on the road network. In this paper, we revisit trajectory generation through the lens of generative route planning, and introduce a new graph-constrained generative paradigm, Anticipatory Generative Modeling, which resolves next transition selection by incorporating the Anticipation of how the future 
    
[^82]: 定位、判断、阅读：基于知识点索引的角色专门化工具代理用于开放域多模态文档问答

    Locate, Judge, Read: Role-Specialized Tool Agents over a Knowledge-Point Index for Open-Domain Multimodal Document QA

    [https://arxiv.org/abs/2610.13128](https://arxiv.org/abs/2610.13128)

    提出按职能分离的KnowForge流水线，通过知识点索引、仅能搜索的定位器、强制第二跳检索的桥接判断器和仅能阅读的深度阅读器等角色专门化工具代理，在开放域多模态文档问答上显著超越了现有检索增强方法。

    

    对大型异构文档集合进行问答需要先找到正确的段落再进行阅读，然而当前的检索增强代理将两类工具都交给同一个模型并让其自行调度。我们提出了KnowForge，一个按职能分离的流水线：一个混合检索的知识点索引；一个只能执行搜索的定位器代理；一个由代码调度的桥接判断器，强制执行缺失的第二跳检索；一个类型化、大小受限的证据池，用于合并每次搜索的排序结果；一个只能阅读文本、图像和表格的深度阅读器代理；以及一个带有格式检查器的综合器。通过自托管的Qwen3.6-35B-A3B（激活参数3B），KnowForge在完整的开放域M3DocVQA（2,441个问题、3,368个PDF）上达到F1 47.5，比已发布的M3DocRAG系统高出11.0，比在同一索引和模型上进行一次性检索回答高出7.1；使用DeepSeek-V4-Flash时达到53.5。在MultiHop-RAG上达到73.5的准确率，高于最强（摘要在此处截断）

    arXiv:2610.13128v1 Announce Type: new  Abstract: Question answering over large heterogeneous document collections requires first finding the right passages and then reading them, yet current retrieval-augmented agents give one model both kinds of tools and let it schedule itself. We present KnowForge, a pipeline separated by job: a hybrid-retrieval knowledge-point index; a Locator agent that can only search; a code-scheduled bridge judge that forces the missing second hop; a typed, size-bounded evidence pool that merges every search's ranking; a Deep Reader agent that can only read text, images and tables; and a synthesizer with a format checker. With a self-hosted Qwen3.6-35B-A3B (3B active), KnowForge reaches F1 47.5 on the full open-domain M3DocVQA (2,441 questions, 3,368 PDFs), +11.0 over the published M3DocRAG system and +7.1 over one-shot retrieve-and-answer on the same index and model, and 53.5 with DeepSeek-V4-Flash. On MultiHop-RAG it reaches 73.5 accuracy, above the strongest
    
[^83]: DRYLAB-Bench：对智能体化大语言模型突变设计的生物安全风险进行基准测试

    DRYLAB-Bench: Benchmarking the Biosecurity Risk of Agentic LLM Mutation Design

    [https://arxiv.org/abs/2610.13116](https://arxiv.org/abs/2610.13116)

    提出DRYLAB-Bench，首个用于评估LLM智能体高危生物突变设计能力的综合基准，将评估从知识问答转向实际应用，以量化LLM结合生物设计工具后带来的生物安全风险。

    

    随着大语言模型（LLM）获得先进的生物设计能力，它们在为研究带来巨大收益的同时，也引入了严重的两用风险，例如改造增强型病原体或耐药细菌。尽管生物安全框架已认识到这些威胁，但LLM执行危险生物修饰的实际风险仍未被量化，尤其是在其借助专门的生物设计工具（BDT）增强的情况下。现有评估主要衡量模型对危险知识的掌握，而非其实际应用能力，忽视了BDT带来的风险放大效应，且依赖于不可靠、昂贵或危险的验证方法。为填补这些空白，我们提出了DRYLAB-Bench（LLM智能体在生物设计中的设计风险），这是首个全面评估LLM智能体高危生物突变设计能力的基准测试。DRYLAB-Bench超越了基础问答的形式，要求智能体执行任务……

    arXiv:2610.13116v1 Announce Type: new  Abstract: As large language models (LLMs) acquire advanced biological design capabilities, they offer substantial research benefits but simultaneously introduce severe dual-use risks, such as engineering enhanced pathogens or drug-resistant bacteria. While biosecurity frameworks recognize these threats, the actual risk of LLMs executing dangerous biological modifications remains unquantified, particularly when augmented with specialized biological design tools (BDTs). Existing evaluations primarily assess hazardous knowledge possession rather than practical application, overlook risk amplification from BDTs, and rely on unreliable, expensive, or hazardous validation methods. To address these gaps, we introduce DRYLAB-Bench (Design Risk Yielded by LLM Agent in Biodesign), the first comprehensive benchmark evaluating the high-risk biological mutation design capabilities of LLM agents. Moving beyond basic question-answering, DRYLAB-Bench tasks agents
    
[^84]: 通过合成自蒸馏对小模型进行持续后训练

    Continually Post-training Small Models via Synthetic Self-distillation

    [https://arxiv.org/abs/2610.13113](https://arxiv.org/abs/2610.13113)

    该论文提出合成自蒸馏方法，通过让略大的同族对齐模型生成同分布的合成示范，并结合离线策略监督与拒绝采样，使缺乏强上下文学习能力的小模型也能在持续后训练中学习新能力的同时保留已有知识。

    

    持续后训练面临获取与保持的双重挑战：模型必须在学习新能力的同时保留已有知识。虽然自蒸馏微调通过使用上下文示例的自我监督缓解了这一问题，但它依赖于强大的上下文学习和指令遵循能力。因此，SDFT 在对低延迟、端侧应用至关重要的小模型上会失效。为了在更小规模上实现持续学习，我们首先找出了失败的根本原因。我们证明了模型获取能力差和遗忘问题与后训练模型的分布偏移呈线性相关，这揭示了同分布学习的必要性。因此，我们提出了合成自蒸馏，通过将同分布重写任务外包给一个略大的同族对齐模型来模仿 SDFT 的做法。结合离线策略监督和拒绝采样，该方法提供了高质量的监督信号。

    arXiv:2610.13113v1 Announce Type: cross  Abstract: Continual post-training faces a dual challenge of acquisition and retention: models must learn new capabilities while preserving prior knowledge. While Self-Distillation Fine-Tuning (SDFT) mitigates this using self-supervision with in-context demonstrations, it relies on strong in-context learning and instruction-following abilities. Consequently, SDFT fails on smaller models critical for low-latency, on-device use. To enable continual learning at smaller scales, we first identify the root cause of failure. We show that poor acquisition and forgetting correlate linearly with the post-trained model's distribution shift, revealing the necessity of in-distribution learning. We thus introduce Synthetic Self-Distillation (SynthSelf), imitating SDFT's in-distribution rewrite by outsourcing the task to a minimally larger, same-family aligner. Coupled with off-policy supervision and rejection sampling, this provides the high-quality supervisio
    
[^85]: 位置重要吗？黑盒因果审计揭示LALM幻觉中模型依赖的音频接地

    Does Location Matter? A Black-Box Causal Audit Reveals Model-Dependent Audio Grounding in LALM Hallucination

    [https://arxiv.org/abs/2610.13112](https://arxiv.org/abs/2610.13112)

    该研究提出一种黑盒因果审计方法，通过CLAP相似度定位并用谱门控移除音频证据区域，发现定位消融在控制时长后稳定优于随机移除，但其幻觉检测效果存在模型依赖性，且先前空白波形对比法的表面优势实为时长混淆所致。

    

    大型音频语言模型（LALMs）经常表现出对象幻觉，即确认并不存在的声音。主流的缓解方法——音频感知解码（AAD）——将真实音频下的logits与全零的“空白”波形进行对比，而此前已有综述指出这种空白波形属于分布外输入。我们引入了一种黑盒因果测试，直接隔离所声称的证据本身：通过CLAP相似度定位某个声明的证据区域，用谱门控方法将其移除，并在三个LALMs上以2x2x2的实验网格设计将其与随机窗口对照进行比较。在Qwen2-Audio-7B上，定位消融的原始AUROC（0.613）低于空白方法（0.721）；但这属于时长混淆，而非定位失败，因为空白方法移除的是整个音频片段，而网格实验仅针对2-4秒的片段。在控制时长一致的条件下，定位方法在两个模型的所有实验单元中均优于随机移除。原始检测价值仍依赖于具体模型，在Audio Flamingo 3上尤为明显。同样的证据可用于纠正解码：对比解码组合……（摘要在此处截断）

    arXiv:2610.13112v1 Announce Type: new  Abstract: Large audio-language models (LALMs) frequently exhibit object hallucination, affirming absent sounds. The dominant mitigation, Audio-Aware Decoding (AAD), contrasts logits under real audio against an all-zero "blank" waveform, flagged by prior review as out-of-distribution. We introduce a black-box causal test that isolates the claimed evidence itself: localize a claim's evidence region via CLAP similarity, remove it by spectral gating, and compare against a random-window control across a 2x2x2 grid on three LALMs. On Qwen2-Audio-7B, localized ablation's raw AUROC trails blank's (0.613 vs. 0.721); this is a duration confound, not a localization failure, since blank removes the entire clip versus the grid's 2-4 s. Holding duration fixed, localization beats random removal at every cell on both models. Raw detection value remains model-dependent, especially on Audio Flamingo 3. The same evidence corrects decoding: contrastive decoding combi
    
[^86]: VETR：面向符号图形推理的基于执行的自我改进智能体

    VETR: Execution-Grounded Self-Improving Agent for Symbolic Graphics Reasoning

    [https://arxiv.org/abs/2610.13098](https://arxiv.org/abs/2610.13098)

    VETR提出了一种以执行轨迹为依据的自我改进智能体框架，通过追踪程序执行中符号编辑的保留、消失与干扰情况，将执行证据转化为受约束的备选程序，从而实现符号图形推理的递归自我改进。

    

    arXiv:2610.13098v1 公告类型：新 摘要：用于符号图形推理的视觉智能体依赖于可执行的世界表示，然而其持续改进仍受限于来自最终渲染结果的模糊监督。近期的智能体系统通过反复交互来优化场景，但它们缺乏从执行轨迹中递归提取可操作证据的机制，忽视了哪些被提出的符号编辑在程序执行过程中得以保留、消失或产生了干扰。为解决这一问题，我们提出了VETR（用于优化的经验证执行轨迹，Verified Execution Trajectories for Refinement），这是一个使智能体能够在符号图形推理的可执行世界表示上进行递归自我改进的框架。在每个任务中，VETR保留与有限运行时观测对齐的程序/渲染记录，并引入一个执行优化器，将观测到的写入操作和状态变化编译为受约束的备选程序。该优化器根据验证条件对其终点和保留情况进行校验

    arXiv:2610.13098v1 Announce Type: new  Abstract: Visual agents for symbolic graphics reasoning depend on executable world representations, yet continued improvement remains bottlenecked by ambiguous supervision from final renders. Recent agentic systems refine scenes through repeated interaction, but they lack mechanisms to recursively extract actionable evidence from execution traces, overlooking which proposed symbolic edits survived, disappeared, or interfered during program execution. To address this, we propose VETR (Verified Execution Trajectories for Refinement), a framework that enables agentic recursive self-improvement over executable world representations for symbolic graphics reasoning. Within each task, VETR retains aligned program/render records with bounded runtime observations and introduces an execution refiner that compiles observed writes and state changes into constrained alternative programs. The refiner validates their endpoints and preservation conditions against
    
[^87]: 当更快的决策无法造就更好的智能体：Jev在智能体工作流中的实证研究

    When Faster Decisions Do Not Make Better Agents: An Empirical Study of Jev in Agent Workflows

    [https://arxiv.org/abs/2610.13092](https://arxiv.org/abs/2610.13092)

    本研究通过在ALFWorld和AndroidWorld上对五个智能体进行4,500次任务执行的实证分析发现，引入Jev虽能显著降低决策成本与延迟，但这种局部效率提升并不必然转化为更好的任务表现——具身任务的成功率接近基线，而两个移动智能体的成功率却明显下降。

    

    AI智能体依赖反复的决策来完成任务，而大语言模型（LLM）通常充当其核心决策者。近期Jev的出现为使这些决策更快、成本更低提供了机会。然而，这种局部节省能否转化为高效且可靠的任务完成仍不清楚。本文在两个代表性应用场景——ALFWorld中的具身任务和AndroidWorld中的移动应用使用——中，对五个智能体的效率传递进行了研究。在4,500次任务执行中，我们综合评估了任务有效性、决策级与任务级的资源使用情况以及执行行为。实验结果表明，集成Jev在所有五个智能体上都显著降低了决策成本和延迟，但任务级收益因工作流而异。在ALFWorld上，任务成功率保持在接近基线的水平，而在两个移动智能体中则出现了更大幅度的下降。

    arXiv:2610.13092v1 Announce Type: new  Abstract: AI agents depend on repeated decisions to accomplish tasks, with large language models (LLMs) commonly serving as their core decision makers. The recent emergence of Jev offers an opportunity to make these decisions faster and at lower cost. However, it remains unclear whether such local savings can be translated into efficient and reliable task completion. In this paper, we investigate this efficiency transfer across five agents in two representative application settings, embodied tasks in ALFWorld and mobile app use in AndroidWorld. Across 4,500 task executions, we jointly evaluate task effectiveness, decision-level and task-level resource use, and execution behavior. Experimental results show that Jev integration substantially reduces decision cost and latency across all five agents, with task-level gains varying by workflow. Task success remains close to baseline levels on ALFWorld and declines more substantially in two mobile agents
    
[^88]: SWEAR：基于自评难度的分段式高效音频推理

    SWEAR: Section-Wise Efficient Audio Reasoning via Self-Rated Difficulty

    [https://arxiv.org/abs/2610.13081](https://arxiv.org/abs/2610.13081)

    SWEAR提出了一种难度感知的音频推理框架，通过五维难度评分和强化学习在推理各阶段自适应分配推理资源，在大幅减少生成token的同时提升准确率，简单样本推理更快而不损失困难样本的性能。

    

    近期关于大型音频-语言模型（LALMs）的研究探索了自适应推理，以在准确性与效率之间取得平衡。然而，现有方法通常依赖粗粒度的难度估计，忽视了决定推理工作量的多样化因素。为解决这一局限，我们从互补的离线评估中推导出五维难度分数和压缩的推理轨迹。基于这种刻画方式，我们提出了SWEAR——一个难度感知的推理框架，它在监督微调阶段学习难度信号，并利用强化学习在各个推理阶段之间自适应地分配推理资源。在MMAU-test-mini和MMAR数据集上的实验表明，SWEAR在大幅减少生成token数量的同时提高了准确率，在不牺牲困难样本性能的前提下缩短了简单样本的推理过程。

    arXiv:2610.13081v1 Announce Type: new  Abstract: Recent work on large audio-language models (LALMs) explores adaptive reasoning to balance accuracy and efficiency. However, existing approaches often rely on coarse-grained difficulty estimation, overlooking the diverse factors that determine reasoning effort. We address this limitation by deriving five-dimensional difficulty scores and compressed reasoning traces from complementary offline assessments. Based on this characterization, we propose SWEAR, a difficulty-aware reasoning framework that learns difficulty signals during supervised fine-tuning and uses reinforcement learning to adapt reasoning allocation across reasoning stages. Experiments on MMAU-test-mini and MMAR show that SWEAR improves accuracy while substantially reducing generated tokens, shortening reasoning for easy instances without sacrificing performance on difficult ones.
    
[^89]: 注意力负责识别，MLP负责拒绝：追踪大语言模型中从注意力到MLP的安全路径

    Attention Sees, MLP Refuses: Tracing the Attention-to-MLP Safety Pathway in LLMs

    [https://arxiv.org/abs/2610.13078](https://arxiv.org/abs/2610.13078)

    该研究揭示了LLM安全机制中的分工路径：注意力模块（尤其早期层）主要负责“看见”有害信息，MLP模块主要负责“写出”拒绝输出，且后训练会增强MLP的拒绝写入能力但不会提升有害性识别能力。

    

    安全对齐的大语言模型（LLM）仍然容易受到越狱攻击，并且可能过度拒绝良性请求。理解这些失败需要将有害性识别与生成拒绝的计算区分开来。我们将这一问题重新构建为一个路径问题，并使其在仅解码器Transformer的残差流更新上可测量，从而将模块更新所暴露的有害性信息与其在可读拒绝证据上产生的变化分离开来。在涵盖Qwen、Llama和Gemma的18个Base-Instruct模型对中，有害性更容易从注意力更新中被解码出来，尤其是在早期层，而MLP更新在对齐模型中对伤害选择性拒绝写入的贡献更大。后训练在所有18个模型对中增强了MLP的拒绝写入能力，但并未系统性提升有害性的可解码性，同时正向MLP写入在后训练后变得更集中于更深的层。

    arXiv:2610.13078v1 Announce Type: new  Abstract: Safety-aligned large language models (LLMs) remain vulnerable to jailbreaks and can over-refuse benign requests. Understanding these failures requires distinguishing harmfulness recognition from the computation that writes refusal. We reframe this as a pathway question and make it measurable on residual-stream updates in decoder-only Transformers, separating the harmfulness information a module update exposes from the change it produces in readable refusal evidence. Across 18 Base-Instruct pairs spanning Qwen, Llama, and Gemma, harmfulness is decoded more readily from attention updates, particularly in early layers, whereas MLP updates contribute more strongly to harm-selective refusal writing in aligned models. Post-training strengthens MLP refusal writing in all 18 pairs without systematically improving harmfulness decodability, while positive MLP writing becomes more concentrated in deeper layers after post-training. Across six reason
    
[^90]: LongHarness：通过有状态证据管理推进对1亿Token级语料库的智能体推理

    LongHarness: Advancing Agentic Reasoning over 100M-Token Corpora via Stateful Evidence Management

    [https://arxiv.org/abs/2610.13075](https://arxiv.org/abs/2610.13075)

    LongHarness通过显式的“证据-结论协同演化”机制管理证据状态，使智能体能够对超出上下文窗口的1亿Token级语料库进行长上下文推理，有效避免无界探索和证据状态漂移两大失败模式。

    

    语料库级的长上下文推理要求智能体在其上下文窗口无法容纳的文档集合中获取并整合证据。现有的智能体运行框架支持对这些语料库的增量式访问，但已获取的证据和不断演化的结论往往仍纠缠在不断增长的交互历史中。这种隐式的状态管理可能导致两种相互耦合的失败模式：无界探索（即检索不断扩展交互历史却未能解决与决策相关的信息缺口）和证据状态漂移（即先前获取的证据或后续修正未能影响当前结论）。我们提出了LongHarness，一个模型无关的运行框架，它使已获取证据与演化结论之间的交互变得显式。其核心机制“证据-结论协同演化”利用未解决的结论引导探索以获取缺失信息，同时新获取的证据又会不断强化和完善结论。

    arXiv:2610.13075v1 Announce Type: new  Abstract: Corpus-level long-context reasoning requires agents to acquire and integrate evidence across document collections that exceed their context windows. Existing agent harnesses enable incremental access to these corpora, but acquired evidence and evolving conclusions often remain entangled in a growing interaction history. This implicit state management can lead to two coupled failure modes: unbounded exploration, where retrieval expands the interaction history without resolving decision-relevant information gaps, and evidence-state drift, where previously acquired evidence or subsequent corrections fail to inform the current conclusion. We present LongHarness, a model-agnostic harness that makes the interaction between acquired evidence and evolving conclusions explicit. Its core mechanism, Evidence-Conclusion Coevolution, uses unresolved conclusions to guide exploration toward missing information, while newly acquired evidence strengthens
    
[^91]: 基于点击的创意进化：通过属性解耦与偏好对齐实现CTR驱动的广告图像编辑

    Evolving Creatives with Clicks: CTR-Driven Advertising image Editing via Attribute Disentanglement and Preference Alignment

    [https://arxiv.org/abs/2610.13066](https://arxiv.org/abs/2610.13066)

    该论文提出ECC框架，将广告创意优化从整体生成转变为细粒度编辑，通过ANOVA和随机森林回归识别出28个与点击率显著相关的属性，实现属性解耦与偏好对齐，从而选择性地优化损害点击率的属性并保留有益属性。

    

    电子商务广告创意的优化目标已从美学质量转向点击率（CTR）等在线业务指标。近期方法通过构建基于CTR的奖励信号，并借助强化学习优化提示词生成，从产品图像生成吸引人的广告图像。然而这类范式会对表现不佳图像中的所有属性进行同等程度的惩罚，即使其中某些属性实际上对点击是有益的。在本工作中，我们提出了Evolving Creatives with Clicks（ECC），将创意优化重新表述为细粒度的编辑而非从零开始的生成。ECC选择性地改进损害CTR的属性，同时保留那些对点击有积极贡献的属性。具体而言，面对海量图像中丰富的属性空间，ECC首先通过方差分析（ANOVA）和随机森林回归识别出28个与CTR显著相关的属性。基于表现不同图像之间的属性对比……（摘要在此处截断）

    arXiv:2610.13066v1 Announce Type: cross  Abstract: The optimization objective for e-commerce advertising creatives has shifted from aesthetic quality to online business metrics like click-through rate (CTR). Recent methods generate appealing ad images from product images by constructing CTR-based reward signals and refining prompt generation via Reinforcement Learning. Such paradigms equally penalize all attributes within an underperforming image, even though some are actually beneficial to clicks. In this work, we propose Evolving Creatives with Clicks (ECC), which reformulates creative optimization as fine-grained editing rather than generation from scratch. ECC selectively refines attributes that hurt CTR while preserving those that contribute positively. Specifically, faced with a rich attribute space across massive images, ECC first identifies 28 attributes significantly correlated with CTR via ANOVA and Random Forest regression. Based on attribute contrasts between images with di
    
[^92]: 回溯前行：面向长时程智能体的回顾性探索与演化

    Travel Back To Move Forward: Retrospective Exploration and Evolving for Long Horizon Agents

    [https://arxiv.org/abs/2610.13058](https://arxiv.org/abs/2610.13058)

    提出ReTRAIL框架，使智能体能够将环境回退至较早状态进行回顾性探索，同时保留已学知识与有效进展，并通过状态锚定GRPO和分支推进奖励两种互补信号训练策略，帮助长时程智能体主动从死胡同轨迹中恢复。

    

    大语言模型智能体越来越多地在长时程环境中运行，在这种环境中，早期的错误可能引发连锁式失败，使智能体陷入无法继续的死胡同轨迹。因此，智能体必须能够主动从这类状态中恢复，并将探索中获得的经验转化为后续决策的指导。我们提出了ReTRAIL，这是一个使智能体能够将环境回退至较早状态的框架，同时保留有效的进展，并携带从被放弃的探索中所学到的知识。环境状态向后回溯，而智能体的知识持续向前演化。ReTRAIL通过两种互补的信号训练共享策略：状态锚定的GRPO（State-Anchored GRPO）通过比较从重访状态出发的不同后续行动方案来改进执行能力，而分支推进奖励则通过后续探索的进展来引导反思。在四个交互式基准测试上的实验：扫雷、推箱子、ALFWorld以及

    arXiv:2610.13058v1 Announce Type: new  Abstract: Large language model agents increasingly operate in long-horizon environments where early mistakes can trigger cascading failures and trap agents in dead-end trajectories. Agents must therefore be able to proactively recover from such states and turn their exploratory experience into guidance for subsequent decisions. We introduce ReTRAIL, a framework that enables agents to rewind the environment to earlier states while preserving valid progress and carrying forward knowledge learned from abandoned exploration. The environment state moves backward while the agent's knowledge continues to move forward. ReTRAIL trains a shared policy with two complementary signals: State-Anchored GRPO compares alternative continuations from revisited states to improve execution, while a branch-advancement reward guides reflection through the progress of subsequent exploration. Experiments across four interactive benchmark MineSweeper, Sokoban, ALFWorld and
    
[^93]: CoBe：面向对话式反事实文本编辑的基准测试

    CoBe: A Benchmark For Conversational Counterfactual Text Editing

    [https://arxiv.org/abs/2610.13057](https://arxiv.org/abs/2610.13057)

    该论文提出了CoBe基准，用于评估模型的对话式反事实文本编辑能力，要求模型通过因果理解判断假设性变化会影响的范围，在修改文本的同时保留未受因果影响的事实，而非仅生成统计上相关联的文本。

    

    反事实文本编辑是指根据一个假设性变化来修改给定文本的能力，同时保留未受该变化因果影响的事实。当被问及“如果医生没有推荐手术会发生什么”时，模型输出应保留患者的病史，仅修改该决定所带来的后果。这需要因果理解能力：模型必须在没有被明确指示的情况下，自行判断哪些事件受到假设性变化的影响，哪些内容应保持不变。这项技能是人类推理的核心，尤其在医疗或法律等安全关键领域中至关重要。相比之下，关联性文本生成任务只是生成与改变后的条件在统计上相关联的文本。在上述医疗示例中，关联性响应可能会让患者的症状变得更轻，因为较轻的病例通常与不做手术相关。这样的响应会产生一个看似合理的场景……

    arXiv:2610.13057v1 Announce Type: cross  Abstract: Counterfactual text editing is the ability to revise a given text in light of a hypothetical change while preserving facts not causally affected by that change. When asked what would have happened if a doctor had not recommended surgery, the model output should retain the patient's medical history and modify only the consequences of the decision. This requires causal understanding: a model must determine, without being explicitly instructed to, what events are affected by the hypothetical change and what should remain fixed. This skill is central to human reasoning, especially in safety-critical domains like healthcare or law. By contrast, the task of associational text generation merely generates text statistically associated with the altered condition. In the medical example, an associational response might make the patient's symptoms less severe because milder cases correlate with no surgery. Such responses produce a plausible scena
    
[^94]: GeoGR：面向高效3D视觉语言模型的空间-语义令牌压缩与分组路由

    GeoGR: Spatial-Semantic Token Compression and Group Routing for Efficient 3D VLMs

    [https://arxiv.org/abs/2610.13035](https://arxiv.org/abs/2610.13035)

    提出GeoGR两阶段框架，通过将跨视角观察压缩为3D体素令牌（GeoSemZip）并在LLM层间进行令牌分组路由，联合压缩场景表示与视觉计算，从而实现高效的3D视觉语言模型推理。

    

    多视角表示使预训练的图像和视频视觉语言模型能够理解3D场景，但其冗长的视觉令牌序列导致推理成本高昂。冗余一方面来自不同视角对同一场景区域的重复观察，另一方面来自在LLM的每一层都处理所有视觉令牌，而不考虑其与当前计算的相关性。我们提出GeoGR，这是一个联合压缩场景表示与视觉计算的两阶段框架。在投影器层面，GeoSemZip将跨视角观察整合为3D体素令牌，并结合语义显著性与空间覆盖来构建紧凑的场景表示。未被选中的体素特征被聚合为上下文锚点，以保留原本会被丢弃的场景信息。在LLM层面，我们识别出有效的视觉计算窗口，并观察到基于查询条件的令牌相关性保持局部稳定，但（摘要在此处截断）

    arXiv:2610.13035v1 Announce Type: cross  Abstract: Multi-view representations enable pretrained image and video vision-language models to understand 3D scenes, but their long visual token sequences make inference costly. Redundancy arises both from repeated observations of the same scene regions across views and from processing all visual tokens at every LLM layer, regardless of their relevance to the current computation. We propose GeoGR, a two-stage framework that jointly compresses scene representations and visual computation. At the projector level, GeoSemZip consolidates cross-view observations into 3D voxel tokens and combines semantic saliency with spatial coverage to construct a compact scene representation. Unselected voxel features are aggregated into contextual anchors to retain scene information that would otherwise be discarded. At the LLM level, we identify an effective visual computation window and observe that query-conditioned token relevance remains locally stable but
    
[^95]: 不要猜测，直接询问：评估仓库级软件开发中的交互式需求澄清

    Don't Guess, Just Ask: Evaluating Interactive Requirement Clarification for Repository-Level Software Development

    [https://arxiv.org/abs/2610.13022](https://arxiv.org/abs/2610.13022)

    该论文提出ReqClarifyBench，一个包含341个任务（涵盖游戏、小程序和网页开发）的交互式需求澄清基准，用于评估LLM通过交互将模糊用户意图转化为清晰需求的能力及其对后续代码生成的影响。

    

    随着大语言模型（LLM）在软件实现方面的能力日益增强，一个关键瓶颈逐渐显现：它们是否真正理解用户想要什么。用户往往从模糊的意图出发，许多重要需求需要通过交互来澄清。在仓库级开发中，误解可能会持续传播并导致代价高昂的返工。然而，现有的基准测试通常基于完整规格说明或欠规范的请求来评估代码实现，忽视了编码前的需求澄清环节。以往关于需求澄清的研究主要聚焦于函数级任务。因此，目前尚不清楚LLM通过交互将模糊意图转化为清晰需求的能力究竟如何，以及这种能力会如何影响后续的代码生成。我们提出了ReqClarifyBench，一个面向仓库级软件开发的交互式需求澄清基准。它包含341个任务，涵盖游戏、小程序和网页开发，每个任务都带有……（摘要在此处被截断）

    arXiv:2610.13022v1 Announce Type: cross  Abstract: As LLMs become increasingly capable of implementing software, a critical bottleneck emerges: whether they truly understand what users want. Users often begin with vague intent, leaving important requirements to be clarified through interaction. In repository-level development, misunderstandings can propagate and cause costly rework. However, existing benchmarks typically evaluate implementation from complete specifications or underspecified requests, overlooking pre-coding requirement clarification. Prior clarification studies mainly focus on function-level tasks. It therefore remains unclear how well LLMs can transform vague intent into clear requirements through interaction and how this capability affects subsequent code generation.   We introduce ReqClarifyBench, an interactive requirement clarification benchmark for repository-level software development. It contains 341 tasks spanning game, miniapp, and web development, each with a
    
[^96]: EmbodiedCLBench：评估自我进化具身智能体的持续学习能力

    EmbodiedCLBench: Evaluating Continual Learning for Self-Evolving Embodied Agents

    [https://arxiv.org/abs/2610.13021](https://arxiv.org/abs/2610.13021)

    提出了首个具身仿真持续学习基准 EmbodiedCLBench，以组合泛化为核心，评估自我进化智能体能否将基础任务中积累的经验迁移并重组以解决未见的高级任务。

    

    持续学习要求模型积累过去的经验并将其迁移到新任务上。尽管持续学习已得到大量研究，现有工作大多仍局限于非具身场景，尚未扩展到具身环境。为填补这一空白，我们提出了 EmbodiedCLBench——首个用于评估自我进化智能体持续学习能力的具身仿真基准。该基准以组合泛化为核心评估原则，考察智能体能否先从基础任务中学习，再将所习得的技能重新组合，以解决复杂且未曾见过的高级任务。基于 EmbodiedCLBench，我们在多种智能体框架和学习方法上开展了大量实验，揭示了一致的局限性与独特的行为特征。我们的结果表明，虽然在零样本设置下高级任务几乎无法被解决，但来自基础任务的经验能够带来持续的提升（原文此处截断）。

    arXiv:2610.13021v1 Announce Type: new  Abstract: Continual learning requires models to accumulate past experience and transfer it to new tasks. Despite substantial research on continual learning, existing work remains largely confined to non-embodied settings, without extending to embodied environments. To address this gap, we introduce EmbodiedCLBench, the first benchmark in embodied simulation for evaluating the continual learning capability of self-evolving agents. With compositional generalization as the core evaluation principle, our benchmark evaluates whether agents can first learn from basic tasks and then recombine the acquired skills to solve complex, unseen advanced tasks. Based on EmbodiedCLBench, we conduct extensive experiments across multiple agent harnesses and learning methods, revealing consistent limitations and distinctive behaviors. Our results show that while advanced tasks can be hardly solved under zero-shot setting, the experience from basic tasks enables consi
    
[^97]: 广义近端策略优化：从散度视角统一大语言模型强化学习算法

    Generalized Proximal Policy Optimization: Unifying LLM RL Algorithms from the Perspective of Divergence

    [https://arxiv.org/abs/2610.13015](https://arxiv.org/abs/2610.13015)

    该论文提出广义近端策略优化（GPPO），基于带单调性能改进保证的广义 f-散度信赖域理论，从散度几何视角统一理解和设计大语言模型强化学习算法，克服了 PPO 固定截断机制无法自适应策略偏移的缺陷。

    

    强化学习已成为提升大语言模型（LLM）推理能力的基础范式。然而，主流方法近端策略优化（PPO）依赖一个几何上退化的截断代理，无法自适应地调整对策略偏移的抵抗，从而导致过早截断和探索能力退化。后续的各类变体则通过零散且启发式的修改来缓解这些问题。我们提出了广义近端策略优化（GPPO），这是一个有原则的统一框架，通过散度的视角来理解和设计大语言模型强化学习算法。我们建立了具有单调性能改进保证的广义 $f$-散度信赖域理论，并证明了不同的 $f$-散度会诱导出不同的几何结构与策略更新动态。我们进一步引入了新颖的估计器拓扑结构，相比传统的单样本蒙特卡洛方法，能够提供更忠实的散度估计。

    arXiv:2610.13015v1 Announce Type: cross  Abstract: Reinforcement learning has become the foundational paradigm for enhancing the reasoning capabilities of LLMs. Yet the dominant approach, Proximal Policy Optimization (PPO), relies on a geometrically degenerate clipping proxy that cannot adapt its resistance to policy shifts, leading to premature clipping and degraded exploration. Subsequent variants alleviate these issues through fragmented and heuristic modifications. We propose Generalized Proximal Policy Optimization (GPPO), a principled and unified framework for understanding and designing LLM RL algorithms through the lens of divergence. We establish a generalized $f$-divergence trust-region theory with a monotonic performance improvement guarantee, and show that different $f$-divergences induce distinct geometries and policy update dynamics. We further introduce novel estimator topologies that provide more faithful divergence estimates than the conventional single-sample Monte Ca
    
[^98]: 人机反馈循环中的溯源感知信念更新

    Provenance-Aware Belief Updating in Human-AI Feedback Loops

    [https://arxiv.org/abs/2610.13008](https://arxiv.org/abs/2610.13008)

    该论文提出LQET训练目标，使语言模型能够依据证据来源而非报告数量进行信念更新，显著降低了重复转发信息造成的虚假佐证与置信度漂移问题。

    

    在重复的人机交互中，单一观察结果可能被引用、总结，并以多条报告的形式返回给模型，使得重复信息看起来像是独立的佐证。我们研究语言模型能否利用所提供的报告谱系，依据证据的来源而非报告数量来进行更新。我们提出了潜在商证据训练，这是一种关系型训练目标，将稀疏贝叶斯监督与约束条件相结合，促使模型在报告未带来新证据时保持预测稳定，并在不同转发历史中对相同的新证据做出一致的更新。在一个受控的二分类基准测试中，在 Qwen3-4B 和 Phi-4-mini 两种模型上（每种骨干模型均采用三组配对训练随机种子），与仅使用任务训练相比，LQET 将新证据更新的均方根误差（RMSE）降低了 41-44%，并将八次转发步骤后的对数几率漂移降低了 89-91%。这些收益还延伸到一个闭环模拟中，其中模型置信度控制着多少转发报告重新进入系统。

    arXiv:2610.13008v1 Announce Type: new  Abstract: In repeated human-AI interaction, a single observation can be quoted, summarized, and returned to a model as multiple reports, making repetition look like independent corroboration. We study whether language models can use supplied report lineage to update according to evidence provenance rather than report count. We introduce Latent Quotient Evidence Training (LQET), a relational objective that combines sparse Bayesian supervision with constraints that encourage stable predictions when reports add no new evidence and consistent updates to the same new evidence across relay histories. On a controlled binary benchmark, LQET reduces new-evidence update RMSE by 41-44% and log-odds drift after eight relay steps by 89-91% relative to task-only training across Qwen3-4B and Phi-4-mini, with three paired training seeds per backbone. The gains extend to a closed-loop simulation in which model confidence controls how many relayed reports re-enter 
    
[^99]: 规模化赋能智能体

    Scaling Empowered Agents

    [https://arxiv.org/abs/2610.13002](https://arxiv.org/abs/2610.13002)

    该论文提出了一种简单可扩展的赋能度最大化方法——将赋能度视为某状态下可执行技能的数量，引导智能体到达多种技能的良好起点状态——从而首次在大规模环境中实现了理论预测的赋能度涌现特性，如收集资源、改造环境、生存，以及在 Crafter 中建造房屋。

    

    赋能度衡量的是智能体主动控制其环境的能力。在本工作中，我们阐明了技能学习与赋能度之间的联系。虽然技能学习并不等于赋能度，但我们提出了一种简单且可扩展的赋能度最大化方法：赋能度对应于从某一状态出发可执行的技能数量，因此最大化赋能度就等价于到达那些能够作为众多技能良好起点的状态。我们的方法适用于任意底层技能学习方法。以往关于赋能度的研究大多局限于表格型设置，而该方法使我们能够在大规模环境中研究赋能度，从而实现理论上已被预测、但经验上难以观测到的赋能度特性（例如收集资源、改造环境、生存等）。实验表明，赋能度最大化能够解锁有趣的涌现特性，例如在 Crafter 环境中建造房屋。

    arXiv:2610.13002v1 Announce Type: new  Abstract: Empowerment measures the capacity of an agent to actively control its environment. In this work, we explain the connection between skill learning and empowerment. While skill learning is not empowerment, we introduce a simple scalable recipe for empowerment maximization: empowerment corresponds to the number of executable skills from a state, so maximizing empowerment corresponds to reaching states that are good starting points for many skills. Our recipe applies to arbitrary underlying skill learning methods. Whereas most prior work on empowerment looks at tabular settings, this recipe enables us to study empowerment at a scale where we can realize theoretically-predicted but empirically-elusive properties of empowerment (e.g., gathering resources, environment modification, survival). Experiments show that empowerment maximization unlocks interesting emergent properties, such as building a house in Crafter. Website and code at \href{htt
    
[^100]: 后见之明结构化建议蒸馏：利用基于执行智能体失败与任务特征训练的顾问模型来引导执行智能体

    Hindsight Structured Advice Distillation: Steering Executor Agents with Advisors Trained on Their Failures and Task Characteristics

    [https://arxiv.org/abs/2610.13000](https://arxiv.org/abs/2610.13000)

    提出后见之明结构化建议蒸馏方法，训练一个小型顾问模型分析冻结执行智能体的失败案例与任务特征，为每个输入动态生成结构化建议并附加到其提示词中，从而以低成本弥补专业任务中细节知识的缺失。

    

    语言模型智能体被应用于金融研究、医疗健康和法律等专业领域的工作。这类智能体通常已具备完成任务的能力，但仍会遗漏其中的细节。例如在金融领域，这些细节包括某个数字出自哪份申报文件，或一家公司如何披露某项指标。重新训练智能体来学习这些细节成本高昂，且当智能体是封闭或共享系统时往往无法实现。而将缺失的专业知识写入智能体的提示词，会使提示词冗长且对每个输入都完全相同，尽管不同输入需要不同的指导。我们转而提出学习生成这种指导。一个小型顾问模型读取每个输入并撰写结构化建议，该建议被附加到冻结的执行器提示词之后。顾问模型借鉴基于特权信息进行蒸馏的相关工作，通过后见之明（hindsight）方式进行训练。在一个带有分级答案的训练池上，我们分析执行器的失败情况，在多种策略下运行它……（摘要原文在此处截断）

    arXiv:2610.13000v1 Announce Type: new  Abstract: Language-model agents are used for specialized work such as financial research, healthcare and law. Such agents are often capable enough for a task but still miss its details. In finance, for example, these details include which filing reports a figure or how a company discloses a metric. Retraining the agent to learn these details is expensive, and often impossible when the agent is a closed or shared system. Writing the missing know-how into the agent's prompt makes the prompt long and identical for every input, although different inputs need different guidance. We propose to learn to generate this guidance instead. A small advisor model reads each input and writes structured advice, which is appended to the prompt of a frozen executor. The advisor is trained with hindsight, following work on distillation with privileged information. On a training pool with graded answers, we analyse the executor's failures, run it under several strate
    
[^101]: 面向无监督专利权利要求分类的嵌入几何与聚类算法基准测试

    Benchmarking Embedding Geometries and Clustering Algorithms for Unsupervised Patent Claim Classification

    [https://arxiv.org/abs/2610.12997](https://arxiv.org/abs/2610.12997)

    双曲空间嵌入能显著提升跨分类代码的现有技术检索能力——双曲VAE比欧几里得VAE多检索到52%的真正跨类别现有技术，尽管它与官方专利分类系统的一致性更低。

    

    现有技术检索系统在结构上是围绕分类代码组织的，这使得跨代码的发现变得困难。本文探究在专利文本上训练的无监督表示能否恢复受代码限制的搜索无法发现的相关联系，以及双曲空间是否有所帮助。我们在45,000件美国专利上训练了双曲和对应的欧几里得变分自编码器（VAE）（96次运行，6个随机种子）。双曲模型检索到真正跨类别现有技术的频率比欧几里得模型高52%（在前10个结果中分别为18.76%对12.34%，共3,000个查询）。这对本文所针对的检索问题而言是一个实用性增益。双曲模型与官方分类的一致性较低（NMI分别为0.2392对0.3206），我们从几何角度对此进行了解释：这两种度量捕捉的目标不同，只有其中之一与现有技术检索的需求相契合。在对10个技术领域（另有50,000件专利）的研究中，我们发现一个领域的分类……

    arXiv:2610.12997v1 Announce Type: new  Abstract: Prior-art retrieval systems are structurally organized around classification codes, making cross-code discovery difficult. This paper asks if an unsupervised representation trained on patent text can recover relevant connections that code-restricted search cannot, and if hyperbolic space helps. We trained a hyperbolic and a matched Euclidean Variational Autoencoder (VAE) on 45,000 US patents (96 runs, 6 seeds). The hyperbolic model retrieves genuinely cross-category prior art 52% more often than the Euclidean model (18.76% vs. 12.34% of top-10 results, 3,000 queries). This represents a practical gain for the motivating search problem. The hyperbolic model agrees less with official classification (NMI 0.2392 vs. 0.3206), which we explain geometrically: the two measures capture different goals, and only one aligns with prior-art search needs. In a study of 10 technology domains (50,000 more patents), we find that a domain's classification 
    
[^102]: SFT应该学习什么？用于可泛化后训练的条件信息增益

    What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training

    [https://arxiv.org/abs/2610.12995](https://arxiv.org/abs/2610.12995)

    提出CIG-SFT信息论框架，用条件信息增益而非模型兼容性来决定SFT应学习哪些词元，从而提升大语言模型后训练的泛化能力。

    

    监督微调（SFT）仍然是大语言模型后训练中广泛使用的方法，因为固定的示范可以直接引入期望的推理策略和行为模式。然而，强迫模型模仿外部提供的轨迹也可能干扰预训练已经支持的行为，从而限制模型在训练分布之外的泛化能力。近期的动态SFT方法通过根据模型侧信号（如预测概率、先验支持度或分布兼容性）对词元级监督进行重新加权来缓解这一问题。然而，模型兼容性并不等同于学习价值：一个当前模型支持度较弱的词元，可能是不希望出现的不匹配，也可能恰恰是示范意图教授的任务特定决策。我们提出CIG-SFT，这是一个信息论框架，用于估计任务条件下的信息量……

    arXiv:2610.12995v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) remains a widely used approach for post-training large language models, because fixed demonstrations can directly introduce desired reasoning strategies and behavioral patterns. Yet forcing a model to imitate externally provided trajectories may also interfere with behaviors already supported by pretraining, limiting generalization beyond the training distribution. Recent dynamic SFT methods mitigate this issue by reweighting token-level supervision according to model-side signals such as predictive probability, prior support, or distributional compatibility. However, model compatibility is not equivalent to learning value: a token that is weakly supported by the current model may either be an undesirable mismatch or precisely the task-specific decision that the demonstration is intended to teach. We propose CIG-SFT, an information-theoretic framework that estimates the task-conditioned information carried 
    
[^103]: Work2Skill：捕获人类GUI经验以实现可复用的智能体技能归纳

    Work2Skill: Capturing Human GUI Experience for Reusable Agent Skill Induction

    [https://arxiv.org/abs/2610.12984](https://arxiv.org/abs/2610.12984)

    该论文提出Work2Skill框架，通过WorkRecorder同步采集屏幕录制、键鼠事件与无障碍树信息，构建了首个人类桌面GUI演示多模态数据集WorkDemo-350，以从人类操作经验中归纳出可复用的智能体技能。

    

    计算机使用智能体能够执行日益复杂的GUI任务，但面对不熟悉的软件以及需要人类领域知识的工作流程时仍然存在挑战。人类演示通过相同的GUI交互空间传递程序性知识，但要复用这些知识，需要在上下文中解读操作，并将工作流规则与任务特定细节分离开来。因此，我们提出了Work2Skill，一个捕获人类GUI经验以实现可复用智能体技能归纳的框架。WorkRecorder为该框架提供了收集此类经验的基础设施，将屏幕录制、鼠标和键盘事件以及可获取的无障碍树信息进行对齐。利用WorkRecorder，我们构建了WorkDemo-350，这是首个用于可复用智能体技能归纳及下游评估的人类桌面GUI演示同步多模态数据集。其350个演示均经过完整性与任务成功性的筛选。随后，我们提出了一种智能体化的技能归纳方法……（原文在此截断）

    arXiv:2610.12984v1 Announce Type: new  Abstract: Computer-use agents can perform increasingly complex GUI tasks, yet unfamiliar software and workflows requiring human domain knowledge remain challenging. Human demonstrations convey procedural knowledge through the same GUI interaction space, but reuse requires interpreting actions in context and separating workflow rules from task-specific details. We thus introduce Work2Skill, a framework that captures human GUI experience for reusable agent skill induction. WorkRecorder provides the framework's infrastructure for collecting this experience, aligning screen recordings, mouse and keyboard events, and available accessibility-tree information. Using WorkRecorder, we construct WorkDemo-350, the first synchronized multimodal dataset of human desktop GUI demonstrations for reusable agent skill induction and downstream evaluation. Its 350 demonstrations are screened for completeness and task success. We then propose an agentic skill-inductio
    
[^104]: 证据锚定边界：视觉干预在多模态大语言模型中的关键层级与有效时机

    The Evidence Anchoring Boundary: Where Visual Interventions Matter and When They Help in Multimodal LLMs

    [https://arxiv.org/abs/2610.12981](https://arxiv.org/abs/2610.12981)

    该研究发现了多模态大语言模型中的“证据锚定边界”（EAB），证明视觉干预在该边界之前层级的干预效果是边界之后的13.78至26.10倍，为确定视觉干预的有效层级和时机提供了新依据。

    

    视觉干预提供了一种无需训练的方法来减少多模态大语言模型中的幻觉。然而，干预在某些层可能几乎没有作用力，即使它改变了输出，也可能损害原本正确的答案。我们研究了视觉干预在哪些层级保持作用力，以及何时能够改进预测。首先，我们在连续的解码器层上应用相同的视觉残差编辑，并在不同深度上匹配其RMS强度。在来自三个模型系列的六个检查点上，我们观察到干预效果持续下降，并将该编辑效果的操作性起点定义为证据锚定边界（Evidence Anchoring Boundary, EAB）。在基于校准查询固定边界后，EAB之前的不相交评估查询上的保留效果是EAB之后的13.78倍至26.10倍。与中间读出的对比表明，编辑可能在答案偏好稳定之前就失去作用力。其次，我们比较了……（原文摘要在此处截断）

    arXiv:2610.12981v1 Announce Type: cross  Abstract: Visual interventions provide a training-free way to reduce hallucinations in multimodal large language models. However, an intervention may have little leverage at some layers and may damage a correct answer even when it changes the output. We investigate where visual interventions retain leverage and when they improve predictions. First, we apply the same visual-residual edit at successive decoder layers, matching its RMS strength across depth. Across six checkpoints from three model families, we observe a persistent drop in intervention effect and define its operational onset for this edit as the \textbf{Evidence Anchoring Boundary (EAB)}. With boundaries fixed on calibration queries, held-out effects on disjoint evaluation queries before EAB are $13.78\times$--$26.10\times$ larger than those after it. Comparison with intermediate readouts shows that the edit can lose leverage before answer preferences stabilize. Second, we compare c
    
[^105]: 超越语义ID：生成式推荐中的有限束可达性与连续查询生成

    Beyond Semantic IDs: Finite-Beam Reachability and Continuous-Query Generation in Generative Recommendation

    [https://arxiv.org/abs/2610.12977](https://arxiv.org/abs/2610.12977)

    该论文提出两种连续查询生成范式CQG-Single和CQG-AR，通过去除离散语义ID词表与束搜索机制，消除了生成式推荐中的有限束可达性约束，使目录中所有物品都能被直接评分与检索。

    

    arXiv:2610.12977v1 公告类型：新论文 摘要：社交媒体平台的生成式推荐（GR）通常通过有限宽度的束搜索自回归地解码离散语义ID（SID）来检索物品。本研究考察了这种输出接口如何约束被留出的目标物品进入有限返回列表的可能性，同时区分了全目录可评分性与经验性Top-K支持。我们引入了两种连续查询范式：CQG-Single在单次传递中生成一个检索查询，CQG-AR则自回归地生成连续残差更新，在保留逐步解码的同时移除了SID物品词表、码本查找和离散束扩展。这些设计从而消除了由有限束解码导致的SID特定可达性约束，并避免了将离散SID词表与预训练主干模型向量空间对齐的技术难题。两种方法都使得每个被编码的物品可以通过稠密检索或近似检索直接评分……（摘要在此处被截断）

    arXiv:2610.12977v1 Announce Type: new  Abstract: Generative recommendation (GR) of social media platforms commonly retrieves items by autoregressively decoding discrete Semantic IDs (SIDs) with finite-width beam search. This work examines how this output interface constrains whether held-out targets enter finite returned lists, while distinguishing catalog-wide scoreability from empirical top-K support. We introduce two continuous-query paradigms: CQG-Single generates one retrieval query in a single pass, and CQG-AR autoregressively generates continuous residual updates, retaining step-wise decoding while removing the SID item-token vocabulary, code lookup, and discrete beam expansion. These designs thereby eliminate the SID-specific reachability constraint induced by finite-beam decoding and avoid the technical challenge of aligning a discrete SID vocabulary with the pretrained backbone's vector space. Both methods make every encoded item directly scoreable through dense or approximat
    
[^106]: SmartSalesBench：一个用于评估大语言模型销售代理的抗性感知多轮基准测试

    SmartSalesBench: A Resistance-Aware Multi-Turn Benchmark for Evaluating LLM Sales Agents

    [https://arxiv.org/abs/2610.12975](https://arxiv.org/abs/2610.12975)

    SmartSalesBench是首个专为评估LLM销售代理应对具有心理抗拒性买家而设计的多轮对话基准测试，通过13,680个会话和三层级九项评估指标，颠覆了传统对话基准假设用户合作的评估范式。

    

    我们提出了SmartSalesBench，这是首个明确设计用于评估大型语言模型（LLM）销售代理应对心理层面真实且具有抗拒性买家角色的多轮对话基准测试。现有的任务导向对话基准测试假设用户是合作的并衡量任务完成度；SmartSalesBench颠覆了这一假设，将代理置于被设定为在长达二十轮对话中不断抗拒和挑战的买家角色面前。该基准测试包含五个代理档案、十二个买家角色、横跨六个价格档位的十五款智能手机产品，以及四次包含3次迭代的对照实验，总计13,680个会话。销售代理和买家角色的生成均通过检索增强生成（RAG）进行基础支撑。我们引入了九个评估指标，分为三个层级：主要结果指标（CR、BRS、RIR）、次要质量指标（SMS、PFS、ETI）以及计划在未来报告中呈现的探索性指标。

    arXiv:2610.12975v1 Announce Type: new  Abstract: We present SmartSalesBench, the first multi-turn conversational benchmark explicitly designed to evaluate large language model (LLM) sales agents against psychologically grounded, resistant buyer personas. Existing task-oriented dialogue benchmarks assume cooperative users and measure task completion; SmartSalesBench inverts this assumption, placing agents against personas programmed to resist and challenge across conversations of up to twenty turns. The benchmark includes five agent profiles, twelve buyer personas, fifteen smartphone products across six price tiers, and four controlled experiments with 3 iterations totaling 13,680 sessions. Both sales-agent and buyer persona generations are grounded through retrieval-augmented generation (RAG). We introduce nine evaluation metrics grouped into three tiers: primary outcome metrics (CR, BRS, RIR), secondary quality metrics (SMS, PFS, ETI), and exploratory metrics planned for future report
    
[^107]: MotionCoach：为视频专业动作编辑构建多模态引导

    MotionCoach: Constructing Multimodal Guidance for Video Professional Motion Editing

    [https://arxiv.org/abs/2610.12963](https://arxiv.org/abs/2610.12963)

    提出MotionCoach——一个通过多轮工具调用与自我验证来生成接地提示词和参考片段的多模态智能体，将外部专业动作知识注入视频编辑模型，从而实现细粒度的专业动作编辑。

    

    视频编辑模型在大规模训练数据的加持下，在外观、背景、风格和动作的修改方面已取得出色的表现。除一般场景外，专业动作在表演、体育和武术等领域具有重要意义。本文聚焦于专业动作编辑，这涉及细粒度的动作技巧和时间结构。在仅有专业动作名称而缺乏详细描述的情况下，现有模型往往无法生成专业动作，例如在“Stoppie（前轮翘头急刹）”指令下错误地抬起前轮。为应对这一挑战，我们提出了MotionCoach——一个经过训练的多模态智能体，它通过多轮工具调用并执行自我验证，生成接地气的提示词和参考片段，作为可靠且充分的多模态引导，将外部专业动作知识注入视频编辑模型。为提供训练数据，我们构建了一个数据管道来构造……（原文此处截断）

    arXiv:2610.12963v1 Announce Type: cross  Abstract: Video editing models achieve strong performance in modifying appearance, backgrounds, styles, and motions, empowered by large-scale training data. Beyond general scenarios, professional motions are significant in performance, sports, and martial arts. This paper focuses on professional motion editing, which involves fine-grained motion technique and temporal structure. Given professional motion names but lacking detailed descriptions, existing models often fail to generate professional motions, such as incorrectly lifting the front wheel under a Stoppie instruction. To address this challenge, we introduce MotionCoach, a trained multimodal agent that uses multi-turn tools and performs self-verification, producing a grounded prompt and a reference clip as reliable and sufficient multimodal guidance to inject external professional motion knowledge into video editing models. To supply training data, we build a data pipeline to construct tr
    
[^108]: JoyInside-VLM-Agent：面向具身智能体的产品条件化多模态策略学习

    JoyInside-VLM-Agent: Product-Conditioned Multimodal Policy Learning for Embodied Agents

    [https://arxiv.org/abs/2610.12957](https://arxiv.org/abs/2610.12957)

    本文提出JoyInside-VLM-Agent，一个通过产品条件化轨迹构建、闭环沙盒数据生成与分阶段策略学习（含新颖的DPER强化学习方法）训练的ReAct式多模态具身智能体，能够将多模态观察与用户请求转化为可执行动作，并适应产品约束、目标变化与执行反馈。

    

    具身智能体必须将多模态观察和用户请求转化为可执行的动作，同时适应产品约束、不断变化的目标以及执行反馈。我们提出了JoyInside-VLM-Agent，这是一个ReAct风格的多模态智能体，通过面向机器人平台和智能硬件的产品条件化轨迹构建与分阶段策略学习开发而成。一个闭环沙盒生成涵盖目标修订和失败恢复的多模态交互轨迹，并依据执行结果和语义标准进行校验。在强化学习之前，先进行监督微调和多教师在线策略蒸馏；强化学习采用DPER（双源特权在线策略自蒸馏与token级证据路由）方法。DPER在冻结的特权教师与共享推理策略参数的特权学生之间路由token级证据，在调节由奖励推导的token优势的同时保持其符

    arXiv:2610.12957v1 Announce Type: cross  Abstract: Embodied agents must translate multimodal observations and user requests into executable actions while adapting to product constraints, changing goals, and execution feedback. We present JoyInside-VLM-Agent, a ReAct-style multimodal agent developed through product-conditioned trajectory construction and staged policy learning for robotic platforms and smart hardware. A closed-loop sandbox generates multimodal interaction trajectories covering goal revisions and failure recovery, which are checked against execution outcomes and semantic criteria. Supervised fine-tuning and multi-teacher on-policy distillation precede reinforcement learning with DPER (Dual-Source Privileged On-Policy Self-Distillation with Token-Level Evidence Routing). DPER routes token-level evidence between a frozen privileged teacher and a privileged student sharing the rollout policy's parameters, modulating reward-derived token advantages while preserving their sig
    
[^109]: MathVis：AI 能通过绘图进行推理吗？一个数学图形生成基准

    MathVis: Can AI Reason through Drawing? A Benchmark for Mathematical Figure Generation

    [https://arxiv.org/abs/2610.12954](https://arxiv.org/abs/2610.12954)

    MathVis 基准通过 512 个数学图形生成任务评估了大语言模型的可视化与图表辅助推理能力，发现即使最先进的模型 pass@1 也仅约 31%，且其输出偏重形式语言和符号而非显式的图示推理，生成的图形对人类而言往往视觉混乱。

    

    本文介绍了 MathVis，一个旨在评估大语言模型（LLM）可视化前沿数学问题并执行图表辅助推理能力的基准。该基准包含 512 个任务，每个任务由一个数学命题、一张参考图、可执行的 TikZ 或 SVG 代码、一个黄金中间表示（Gold IR）、原子化评分标准以及错误代码构成。经过严格评估，GPT-6-Astra、Claude Fable 5 和 Gemini 3.1 Pro Preview 的 pass@1 分数分别为 31.25%、26.56% 和 22.85%。研究发现，尽管这些任务需要可视化，LLM 生成的输出却以形式语言、符号和图形程序为主，而非显式的以图表为媒介的推理。此外，LLM 生成的解答虽然可能包含所有必要元素，但对人类而言在视觉上往往令人困惑。该基准旨在实现数学问题的可视化，并推动使 LLM 能够进行图形化推理的研究。

    arXiv:2610.12954v1 Announce Type: new  Abstract: This paper introduces MathVis, a benchmark designed to evaluate LLMs' ability to visualize frontier mathematical problems and perform diagram-assisted reasoning. The benchmark consists of 512 tasks, each comprising a mathematical statement, a reference diagram, executable TikZ or SVG code, a Gold Intermediate Representation (IR), atomic rubrics, and error codes. Through rigorous evaluation, GPT-6-Astra, Claude Fable 5, and Gemini 3.1 Pro Preview achieve pass@1 scores of 31.25%, 26.56%, and 22.85%, respectively. We found that although the tasks require visualization, LLMs produce outputs dominated by formal languages, symbols, and graphics programs, rather than explicit diagram-mediated reasoning. Furthermore, while the solutions generated by LLMs may contain all the necessary elements, they are often visually confusing to humans. Ultimately, this benchmark aims to visualize mathematical problems and promote research on enabling LLMs to s
    
[^110]: 冲浪激活波：面向推测解码的大语言模型隐藏状态信息与几何特性分析

    Surfing the Activation Wave: Analyzing the Information and Geometry of LLM Hidden States for Speculative Decoding

    [https://arxiv.org/abs/2610.12939](https://arxiv.org/abs/2610.12939)

    该论文提出了一个基于矩阵熵信息量分析与中心化核对齐（CKA）多样性度量的原则性框架，用于为大语言模型推测解码选择多层隐藏状态，并揭示了解码器表示中的几何与信息分区特性。

    

    来自大语言模型（LLM）不同层的解码器隐藏状态通常编码互补信息。然而，现有的推测解码方法通常依赖于通过固定的、基于位置的启发式规则所选的少量目标模型层。选择最优的层子集本质上是一个组合问题，即使对于中等深度的模型，穷举式评估也不可行。在这项工作中，我们提出了一个用于推测解码的多层隐藏状态选择的原则性框架，该框架基于两个互补的准则：信息量——通过矩阵熵分析来度量；多样性——使用中心化核对齐（CKA）来量化。我们研究了逐层的香农熵、隐藏状态拼接下的聚合熵，以及跨网络深度的成对CKA动态，从而揭示了解码器表示中截然不同的几何与信息分区。基于这些发现，我们提出了……

    arXiv:2610.12939v1 Announce Type: cross  Abstract: Decoder hidden states from different layers of large language models (LLMs) often encode complementary information. However, existing speculative decoding methods typically rely on a small set of target-model layers chosen using fixed, position-based heuristics. Selecting an optimal subset of layers is inherently a combinatorial problem, making exhaustive evaluation infeasible even for moderately deep models. In this work, we present a principled framework for multi-layer hidden-state selection for speculative decoding based on two complementary criteria: informativeness, measured through matrix-entropy analysis, and diversity, quantified using centered kernel alignment (CKA). We investigate layer-wise Shannon entropy, aggregate entropy under hidden-state concatenation, and pairwise CKA dynamics across depth, revealing distinct geometric and informational compartments within decoder representations. Motivated by these findings, we prop
    
[^111]: 少训练、多学习：面向高效推理后训练的不确定性引导渐进收缩方法

    Learning More by Training Less: Uncertainty-Guided Progressive Contraction for Efficient Reasoning Post-Training

    [https://arxiv.org/abs/2610.12933](https://arxiv.org/abs/2610.12933)

    该论文提出一种不确定性引导的渐进收缩框架，利用蒙特卡洛推理树揭示推理过程中的不确定性，将推理监督数据分两阶段（教师感知与学习者感知）逐步压缩，在大幅降低训练成本的同时保持复杂推理能力。

    

    大型语言模型在复杂推理方面取得了显著进展。然而，通过使用越来越庞大的语料库来扩展推理后训练，可能会大幅增加训练成本，而性能提升却不成比例。因此，本研究提出这样一个问题：在不牺牲能力的前提下，推理监督数据能够被压缩到何种程度？为此，我们提出了一种不确定性引导的渐进收缩框架，将监督与优化集中在最具学习价值的推理轨迹上。具体而言，我们不再将教师生成的思维链（CoT）视为单一条已实现的路径，而是通过围绕中间推理状态构建蒙特卡洛推理树来揭示过程级的不确定性，从而构建带有不确定性标注的推理监督数据。该不确定性首先将原始语料库收缩为一个由教师信息指导的紧凑数据池。随后，第二次具备学习者感知能力的收缩则执行在线路由……（原文摘要在此处截断）

    arXiv:2610.12933v1 Announce Type: new  Abstract: Large language models have achieved remarkable advances in complex reasoning. However, scaling reasoning post-training with ever-larger corpora can substantially increase training cost without proportional performance gains. In this study, we therefore ask: how far can reasoning supervision be compressed without sacrificing capability? To this end, we propose an uncertainty-guided progressive contraction framework that concentrates supervision and optimization on the reasoning trajectories most valuable for learning. Specifically, rather than treating a teacher-generated chain-of-thought (CoT) as a single realized path, we construct uncertainty-profiled reasoning supervision by building Monte Carlo reasoning trees around intermediate reasoning states to expose process-level uncertainty. This uncertainty first contracts the original corpus into a compact teacher-informed pool. A second, learner-aware contraction then performs online routi
    
[^112]: 基于模型切片的协同学习与推理

    Collaborative Learning and Reasoning over Model Slices

    [https://arxiv.org/abs/2610.12927](https://arxiv.org/abs/2610.12927)

    该论文提出“协同学习”框架，通过模型切片在同等参数预算下用多个异构小模型协作替代单一大模型，并借助金字塔监督优化跨模型推理与交互决策，从而解决单模型推理中错误传播和盲点难以重评的问题。

    

    大语言模型（LLM）凭借知识先验、上下文推理和测试时计算能力，在多种任务中表现出色。尽管具备这些优势，单模型推理仍然容易受到错误传播的影响，因为后续步骤依赖于模型先前的输出，导致推理中的盲点难以被独立地重新评估。为解决这一局限，我们开发了“协同学习”方法，涵盖异构协作者的构建以及有效跨模型协作的学习。具体而言，“模型切片”（model slicing）在可比参数预算下用更小的异构模型替代大模型，使先前的推理能够在不同的模型参数化之间得到重新评估。“金字塔监督”（pyramid supervision）将协议蒸馏与步骤级监督相结合，在整个协作过程中优化推理和交互决策。我们在四个……上评估了我们的方法。

    arXiv:2610.12927v1 Announce Type: new  Abstract: Large language models (LLMs) excel across diverse tasks, benefiting from knowledge priors, in-context reasoning, and test-time computation. Despite these strengths, single-model reasoning remains vulnerable to error propagation, as later steps depend on the model's earlier outputs, leaving reasoning blind spots difficult to independently reassess. To address this limitation, we develop \textit{collaborative learning}, encompassing the construction of heterogeneous collaborators and the learning of effective cross-model collaboration. Specifically, \textit{model slicing} replaces a large model with smaller, heterogeneous models under a comparable parameter budget, enabling prior reasoning to be reassessed across distinct model parameterizations. \textit{Pyramid supervision} combines protocol distillation with step-level supervision to refine reasoning and interaction decisions throughout collaboration. We evaluate our approach across four
    
[^113]: 用语言教神经解码器

    Teaching neural decoders with language

    [https://arxiv.org/abs/2610.12921](https://arxiv.org/abs/2610.12921)

    提出“语言作为监督”方法，将神经特征与自然语言类别描述对齐以替代传统类别标签监督，在视觉、运动和语音解码任务中显著提升准确率，并支持候选集复用、受试者与会话自适应及未见频率的泛化解码。

    

    脑机接口能够解码神经活动，然而类别标签监督（CLS）使类别之间的关系处于隐式状态。我们提出了“语言作为监督”方法，该方法通过学习到的投影将神经特征与自然语言类别描述对齐，并利用候选文本原型来预测类别。我们在来自头皮和颅内记录的视觉、运动和语音解码任务上评估了七种编码器架构，每种架构均使用独立训练的模型。与匹配的CLS基线相比，LaS将语音解码准确率提升高达17.15个百分点。其表示支持候选集复用和标签监督下的受试者与会话自适应，同时LaS实现了稳态视觉诱发电位（SSVEP）响应中的条件性未见频率解码。正确的类别描述对应关系支撑了这种泛化能力，双侧归因可视化展示了神经与文本对同一类别匹配的共同贡献。

    arXiv:2610.12921v1 Announce Type: new  Abstract: Brain-computer interfaces decode neural activity, yet class-label supervision (CLS) leaves relationships among classes implicit. We propose Language as Supervision (LaS), which aligns neural features with natural-language class descriptions through learned projections and predicts classes using candidate text prototypes. We evaluate seven encoder architectures with independently trained models for visual, motor, and speech decoding from scalp and intracranial recordings. LaS improves speech-decoding accuracy by up to 17.15~pp over matched CLS baselines. Its representations support candidate-set reuse and label-supervised subject and session adaptation, while LaS enables conditional unseen-frequency decoding in steady-state visual evoked potential (SSVEP) responses. Correct class-description correspondence supports this generalization, and dual-sided attribution visualization shows neural and textual contributions to the same class-matchi
    
[^114]: 零售市场中的AI中介：收入效应与商品组合优化

    AI-Mediation in Retail Markets: Revenue Effects and Assortment Optimization

    [https://arxiv.org/abs/2610.12917](https://arxiv.org/abs/2610.12917)

    该研究通过构建AI产品排序数据集并建立Plackett-Luce随机中介模型，证明在零售商品组合优化中纳入AI中介效应可将平均收入损失降低2.52%至11.37%，并显著改变最优商品组合的选择。

    

    AI购物助手日益影响着消费者会考虑购买哪些产品。我们研究了AI中介环境下的零售商品组合规划问题：零售商选择商品目录，AI中介机构形成面向顾客的候选清单，顾客根据多项Logit（MNL）模型从该候选清单中进行选择。我们考虑了两种中介方式：确定性中介，即AI遵循固定的、针对特定细分顾客群体的排序；以及基于Plackett-Luce模型的随机中介。在实证研究方面，我们构建了一个来自主流AI模型的产品排序数据集，估计了Plackett-Luce排序模型，并在留出的商品组合上评估了其预测性能。将这些估计与合成的顾客偏好相结合，我们发现将AI中介纳入考量能够实质性地改变最优商品组合与收入水平。在所测试的各种容量配置下，忽略中介效应会导致相对于PL模型的2.52%至11.37%的平均收入损失。

    arXiv:2610.12917v1 Announce Type: new  Abstract: AI shopping assistants increasingly influence which products consumers consider. We study retail assortment planning under AI mediation, where a retailer chooses a catalog, an AI intermediary forms a customer-facing shortlist, and the customer chooses from that shortlist according to a multinomial logit model. We consider both deterministic mediation, in which the AI follows a fixed segment-specific ranking, and stochastic mediation based on the Plackett-Luce model.   Empirically, we construct a dataset of product rankings from leading AI models, estimate Plackett-Luce ranking models, and evaluate their predictive performance on held-out assortments. Combining these estimates with synthetic customer preferences, we find that accounting for AI mediation can materially change optimal assortments and revenue. Across the tested capacity configurations, ignoring mediation yields mean revenue losses between 2.52% and 11.37% relative to the PL 
    
[^115]: DeFiRank：面向去中心化金融网络治理的联邦成对排序框架

    DeFiRank: Federated Pairwise Ranking for Governance in Decentralized Financial Networks

    [https://arxiv.org/abs/2610.12916](https://arxiv.org/abs/2610.12916)

    DeFiRank提出了一种基于联邦学习的成对排序框架，使多个机构无需共享私有交易数据即可协同训练模型，通过捕捉交易对手多样性、多应用使用和交易频率等全网行为模式来评估提供者行为的真实性，从而实现去中心化金融网络中自动化且可扩展的治理。

    

    现有的去中心化金融网络治理方法依赖于人工委员会进行的手动审查流程，这会引入主观偏见，在高网络交易量下造成可扩展性瓶颈，并且无法捕捉复杂的行为模式和不断演变的网络动态。为了解决这些局限性，我们提出了DeFiRank，一个基于联邦学习的成对排序框架，多个机构可以在不共享其私有交易数据的情况下协同训练排序模型。这种协同学习使模型能够捕捉到体现强网络参与度并直接促进网络效应的全网行为模式，包括交易对手的多样性、多个应用的使用情况以及交易的频率。该系统根据提供者行为的真实性对其进行排名，提升真正贡献者的排名，并降低表现出可疑行为者的排名。

    arXiv:2610.12916v1 Announce Type: new  Abstract: Existing governance approaches in decentralized financial networks rely on manual review processes conducted by human committees, which introduces subjective bias, creates a scalability bottleneck under high network volume, and fails to capture complex behavioural patterns and evolving network dynamics. To address these limitations, we propose \name, a federated learning-based pairwise ranking framework in which multiple institutions collaboratively train a ranking model without sharing their private transaction data. This collaborative learning enables the model to capture network-wide behavioural patterns that signify strong network participation and directly contribute to the network effect, including diversity of counterparties, usage of multiple applications, and frequency of transactions. The proposed system ranks providers based on the authenticity of their actions, promoting genuine contributors and demoting those exhibiting susp
    
[^116]: OGPO：面向可恢复视觉-语言-动作模型的离线目标条件策略优化

    OGPO: Offline Goal-conditioned Policy Optimization for Recoverable Vision-Language-Action Models

    [https://arxiv.org/abs/2610.12914](https://arxiv.org/abs/2610.12914)

    提出OGPO框架，通过语义关键状态重标注和目标条件化的过程奖励，将VLA模型后训练从逐步模仿专家动作转向可恢复的任务进展状态优化。

    

    视觉-语言-动作（VLA）模型在机器人操作领域展现出强大潜力，但其后训练仍主要依赖于对机器人示范数据的轨迹级监督模仿。虽然这种方式能有效学习可行动作，但该范式监督策略在每一步复现专家动作，使其偏向于示范数据特有的动作路径，且当执行偏离示范轨迹时，对于如何恢复到任务进展状态所提供的指导十分有限。为解决这一局限，我们提出了OGPO，一个用于可恢复VLA后训练的离线目标条件策略优化框架。OGPO不再将离线示范仅仅视为逐步的动作标签，而是自适应地利用语义关键状态对轨迹进行重标注，并构建目标条件化的过程奖励，以优化策略去达成这些任务进展状态。通过这种方式，OGPO将VLA后训练从……

    arXiv:2610.12914v1 Announce Type: new  Abstract: Vision-language-action (VLA) models have shown strong promise in robotic manipulation, yet their post-training still predominantly relies on trajectory-level supervised imitation over robot demonstrations. While effective for learning feasible actions, this paradigm supervises policies to reproduce expert actions at each step, biasing them toward demonstration-specific action paths and providing limited guidance for recovering toward task-progress states once execution deviates from demonstrated trajectories. To address this limitation, we propose OGPO, an Offline Goal-conditioned Policy Optimization framework for recoverable VLA post-training. Instead of treating offline demonstrations merely as step-wise action labels, OGPO adaptively relabels trajectories with semantic key states and constructs goal-conditioned process rewards to optimize the policy for reaching these task-progress states. In this way, OGPO shifts VLA post-training fr
    
[^117]: 注意空间差距：一个用于地理空间推理与受限空间编辑的基准测试

    Mind the Spatial Gap: A Benchmark for Geospatial Reasoning and Constrained Spatial Editing

    [https://arxiv.org/abs/2610.12881](https://arxiv.org/abs/2610.12881)

    该论文提出了GeoSpatialBench，一个基于六个城市真实城市数据、在GeoJSON与地图图像两种模态上评估28种空间关系的新基准，用于系统考察多模态大模型从空间关系理解、几何推理到满足几何约束的空间编辑的递进式空间认知能力。

    

    理解与操作空间关系，对于能够对物理环境进行推理并构建扎根世界模型的人工智能系统而言至关重要。然而，现有多模态大语言模型（MLLM）空间能力的评估主要依赖于对小型或合成场景的静态问答（QA），很少探究模型能否在满足显式几何约束的前提下修改空间配置。我们提出了GeoSpatialBench，这是一个旨在评估空间认知逐步递进能力的基准，涵盖从空间关系理解、几何推理到受限空间编辑的完整进阶过程。GeoSpatialBench基于来自六个地理位置各异城市的真实世界城市数据构建，在结构化GeoJSON与地图图像两种模态上，对涵盖拓扑、方向、度量和顺序四大类共28种空间关系的完整分类体系进行评估，并提供真值参考与约束……（摘要原文在此处截断）

    arXiv:2610.12881v1 Announce Type: new  Abstract: Understanding and manipulating spatial relations is fundamental to AI systems that reason about physical environments and construct grounded world models. However, existing evaluations of multimodal large language models (MLLMs) for spatial reasoning rely predominantly on static question-answering (QA) over small-scale or synthetic scenes, rarely probing whether models can modify spatial configurations while satisfying explicit geometric constraints. We introduce GeoSpatialBench, a benchmark designed to evaluate an escalating progression of spatial cognition, ranging from spatial relation understanding and geometric reasoning to constrained spatial editing. Grounded in real-world urban data across six geographically diverse cities, GeoSpatialBench evaluates a comprehensive taxonomy of 28 topological, directional, metric, and ordering relations across structured GeoJSON and map image modalities, with ground-truth references and constraint
    
[^118]: 基于Seq-Drifting的单步文本生成

    One-Step Text Generation by Seq-Drifting

    [https://arxiv.org/abs/2610.12879](https://arxiv.org/abs/2610.12879)

    该论文提出Seq-Drifting框架，通过构建具有上下文自适应目标的漂移场（结合吸引力与排斥力机制），使模型直接学习从高斯噪声到文本的映射，实现单次前向传播即可生成完整序列的文本。

    

    扩散语言模型（DLM）可以并行生成词元，但仍需要多个采样步骤才能产生高质量文本。在本文中，我们研究模型是否能够学会在单次前向传播中生成整个序列。我们提出了Seq-Drifting，这是一个将漂移机制应用于连续词元表示序列的框架，它直接学习从高斯噪声到文本的映射。与现有扩散语言模型直接从训练序列中观察到的词元推导每个位置的监督信号不同，Seq-Drifting构建了一个漂移场，其逐位置目标会适应当前已生成的上下文。在每个位置上，吸引力引导每个输出趋向一组在上下文上合理的词元表示，而排斥力则防止不同样本之间的坍缩以及序列内部的重复。生成器在训练期间从这个漂移场中学习，并在推理时通过一次前向传播联合生成所有词元。

    arXiv:2610.12879v1 Announce Type: new  Abstract: Diffusion language models generate tokens in parallel, but still require multiple sampling steps to produce high-quality text. In this paper, we investigate whether a model can learn to generate an entire sequence in a single forward pass. We propose Seq-Drifting, a framework that applies drifting to sequences of continuous token representations and directly learns a map from Gaussian noise to text. Unlike existing DLMs that derive the supervision at each position directly from the observed token in a training sequence, Seq-Drifting builds a drift field whose position-wise targets adapt to the current generated context. At each position, attraction guides each output toward a set of contextually plausible token representations, while repulsion discourages collapse across samples and repetition within sequences. The generator learns from this drift field during training and produces all tokens jointly in one pass at inference. Empirically
    
[^119]: 协同心智理论：通过预测用户响应学习自我修订

    Collaborative Theory of Mind: Learning to Self-Revise by Anticipating User Responses

    [https://arxiv.org/abs/2610.12872](https://arxiv.org/abs/2610.12872)

    该论文提出协同心智理论（CToM）框架，通过训练大语言模型预测用户对草稿回复的响应，并利用该预测在回复前进行内部自我修订，同时结合GRPO优化，从而提升模型在多轮交互中与用户协作的能力。

    

    大型语言模型（LLMs）在单个任务上展现出强大的性能，但在长时间的多轮交互中仍然难以与用户协作，例如在理解用户需求之前就急于给出解决方案，或者在收到反馈后未能修正错误的假设。我们提出了协同心智理论，一个通过显式训练大语言模型预测用户响应，并利用这些预测来改进自身行为，从而应对上述挑战的框架。CToM先起草一个回复，预测用户将如何回应，并在最终呈现回复之前利用该预测来指导内部修订。在训练过程中，用户对每个草稿的回复被用作预测下一轮对话的目标，我们训练助手最小化该回复的负对数似然。我们将这一预测目标与组相对策略优化（GRPO）相结合，以同时优化任务表现……

    arXiv:2610.12872v1 Announce Type: new  Abstract: Large language models (LLMs) demonstrate strong individual task performance, yet still struggle to collaborate with users over extended, multi-turn interactions, for example by attempting solutions before understanding the user's needs or by failing to revise mistaken assumptions after receiving feedback. We propose Collaborative Theory of Mind (CToM), a framework that addresses these challenges by explicitly training LLMs to anticipate user responses and to use those predictions to improve their own behavior. CToM drafts a response, predicts how the user will respond, and uses this prediction to guide internal revision before presenting a response. During training, the user's reply to each draft serves as the target for predicting the next turn, and we train the assistant to minimize the negative log-likelihood of that reply. We combine this prediction objective with group-relative policy optimization (GRPO) to optimize both task perfor
    
[^120]: 基于扩散模型与高保真度评估的船体局部化生成设计

    Localized Generative Design of Ship Hulls with Diffusion Models and High-Fidelity Assessment

    [https://arxiv.org/abs/2610.12870](https://arxiv.org/abs/2610.12870)

    该论文提出一种奖励导向的扩散模型框架，通过参数化掩码实现球鼻艏和船体前部等局部区域的生成式优化，并结合经验证的高保真自动化CFD流程进行评估，发现生成的船体在低速时阻力与基准相当，但在高速时阻力高出45-50%。

    

    实际船舶设计需要对局部区域进行精细化改进并进行性能评估。本文提出了一种奖励导向的扩散框架，用于船体局部优化。在数据集生成过程中采用参数化掩码技术，将几何变化限制在选定的变量上（包括球鼻艏和船体前部），同时保持整船的几何连续性。代理模型辅助的序贯蒙特卡洛采样以降低预测的静水阻力为目标，通过训练数据保持稳定性，并在筛选过程中匹配排水量。研究构建了一个自动化的粘性自由液面CFD流程（针对两艘基准集装箱船与实测数据进行了验证），用于评估一个生成的船体变体以及GenHull E——一艘由船舶设计师在同一参数表示框架内改进的船体。在匹配的KRISO集装箱船（KCS）排水量下，生成的变体在傅汝德数Fn=0.195时总阻力与基准相当，但在Fn=0.250和0.260时阻力高出45-50%，这一缺陷被……（摘要原文在此截断）

    arXiv:2610.12870v1 Announce Type: cross  Abstract: Practical ship design requires localized refinement and performance evaluation. This paper presents a reward-directed diffusion framework for localized hull optimization. Parametric masking during dataset generation restricts geometric changes to selected variables, including the bulbous bow and forebody, while preserving full-hull continuity. Surrogate-assisted sequential Monte Carlo sampling targets reduced predicted calm-water resistance, with stability retained through training data and displacement matched during screening. An automated viscous free-surface CFD pipeline, validated against measurements for two benchmark container ships, evaluates a generated variant and GenHull E, a hull refined by a naval architect within the same representation. At matched KRISO Container Ship (KCS) displacement, the generated variant has comparable total resistance at $F_n=0.195$ but 45-50% higher resistance at $F_n=0.250$ and $0.260$, missed by
    
[^121]: 通过表示几何正则化缓解大语言模型自我改进中的能力退化

    Mitigating Capability Degradation in LLM Self-Improvement via Representation Geometry Regularization

    [https://arxiv.org/abs/2610.12855](https://arxiv.org/abs/2610.12855)

    该论文发现在线策略自蒸馏（OPSD）会导致大语言模型已获得能力的逐渐退化，其根源在于隐藏表示几何结构的恶化，并提出表示几何正则化（RGR）方法，通过保持簇内紧凑性和簇间分离性来缓解自我改进过程中的能力退化问题。

    

    在线策略自蒸馏（On-Policy Self-distillation, OPSD）是一种自我改进范式，它使大语言模型（LLM）能够在不同的条件上下文中，将同一模型同时作为教师和学生来提升自身能力。然而，我们发现OPSD方法可能会逐渐退化模型先前已获得的能力：在早期训练阶段能够正确回答的查询，随着优化的进行可能变为回答错误，即使整体性能仍在持续提升。进一步分析表明，这种能力退化与隐藏表示几何结构的恶化同时发生，其特征表现为簇内紧凑度降低和簇间分离度减弱。受此观察启发，我们提出了表示几何正则化（Representation Geometry Regularization, RGR），通过鼓励簇内紧凑性和簇间分离性，在后训练过程中保持表示几何结构。RGR可以无缝地（摘要原文在此处截断）

    arXiv:2610.12855v1 Announce Type: new  Abstract: On-Policy Self-distillation is a self-improvement paradigm that enables large language models (LLMs) to improve their capabilities by leveraging the same model as both teacher and student under different conditioning contexts. However, we find that OPSD methods can progressively degrade previously acquired capabilities: queries that are correctly responded at earlier training stages may become incorrectly responded as optimization proceeds, even the overall performance continues to improve. Further analysis reveals that this degradation coincides with a deterioration of the geometry of hidden representations, characterized by reduced intra-cluster compactness and diminished inter-cluster separation. Motivated by this observation, we propose Representation Geometry Regularization (RGR), which preserves representation geometry during post-training by encouraging compactness within cluster and separation across cluster. RGR can be seamlessl
    
[^122]: 同时修改应用代码与部署基础设施：AI智能体拉取请求的因果研究

    Co-Modifying Application Code and Deployment Infrastructure: A Causal Study of AI Agent Pull Requests

    [https://arxiv.org/abs/2610.12849](https://arxiv.org/abs/2610.12849)

    该研究通过涵盖26,999个拉取请求的大规模因果分析，首次证明AI智能体同时修改应用代码与部署基础设施的“双范围”拉取请求会引入额外的集成开销，揭示了当前智能体缺乏部署感知能力的关键局限。

    

    当前的软件工程智能体主要在隔离的沙盒环境中、针对本地同步测试进行评估。然而，在协作式代码仓库中部署新功能，除了修改应用逻辑之外，还需要同步修改持续集成/持续交付（CI/CD）流水线、容器以及资源供给清单。由于智能体缺乏部署感知能力，尝试这类跨领域修改会向异步集成流水线中引入未经验证的依赖。为了分离“双范围拉取请求”（PR）——即同时修改应用代码和部署基础设施——所带来的因果性集成开销，我们在两个大型数据集（AIDev v5 和 MOSAIC-3M）上评估了26,999个拉取请求，涵盖六种智能体架构以及一个对齐的人类基线。通过确定性的四层文件分类法和精确的4D分层匹配框架，我们在控制应用代码量、保持条件可交换性的前提下，分离出了相关集成指标。

    arXiv:2610.12849v1 Announce Type: cross  Abstract: Current software engineering agents are evaluated predominantly inside isolated sandboxes against synchronous local tests. However, deploying features in collaborative repositories requires modifying continuous integration (CI/CD) pipelines, containers, and provisioning manifests alongside application logic. Because agents lack deployment awareness, attempting cross-domain modifications introduces unverified dependencies into asynchronous integration pipelines. To isolate the causal integration overhead of dual-scope pull requests (PRs)-modifying both application code and deployment infrastructure-we evaluated 26,999 PRs across two large datasets (AIDev v5 and MOSAIC-3M) encompassing six agent architectures and an aligned human baseline. Using a deterministic four-tier file taxonomy and an exact 4D strata-matching framework, we isolated integration metrics under conditional exchangeability while controlling for application volume. We f
    
[^123]: 当生成模型充当模拟器时

    When Generative Models Act as Simulators

    [https://arxiv.org/abs/2610.12847](https://arxiv.org/abs/2610.12847)

    该论文研究生成模型通过“可编辑性”充当模拟器的条件，提出两种在模型潜在状态与模拟器状态之间学习映射的方法（探针解码加引导、逆映射），并发现逆映射方法即使在引导失败时也能成功。

    

    模拟器将环境总结为一个用户可以读取和修改的状态。生成模型无需显式指定即可学习世界的表示，但没有提供编辑该世界状态的内置方法。我们研究了生成模型在何种条件下可以通过“可编辑性”充当模拟器：即改变模型的潜在表示是否能产生与编辑模拟器状态后相同的下一步输出。简而言之，我们的目标是在生成模型的潜在状态与模拟器状态之间找到一种映射，其中模型仅通过观测数据进行训练，而映射则是在此之后学习的。我们测试了两种映射方法，两者都是从与模拟器状态配对的观测数据中学习得到的。第一种方法是训练探针从模型中解码出模拟器状态，然后沿着这些方向进行引导；第二种方法是通过逆映射直接学习该映射，这种方法即使在引导失败的情况下也能取得成功。

    arXiv:2610.12847v1 Announce Type: cross  Abstract: A simulator summarizes an environment in a state that users can read and change. Generative models learn representations of the world without explicit specification, but offer no built-in way to edit that world's state. We investigate the conditions under which a generative model can act as a simulator through $\textit{editability}$: whether changing a model's latent representation produces the next-step outputs we get by editing a simulator's state. Plainly stated, our goal is to find a mapping between a generative model's latent state and the simulator state, where the model is trained on observations alone and the mapping is learned afterward. We test two kinds of mappings, both learned from observations paired with simulator states. In one, we train probes to decode the simulator state from the model, then steer along them. In the other, we learn the mapping directly with an inverse map, which can succeed even where steering fails.
    
[^124]: 迈向在人工智能应用中定义、测量与利用创造力

    Towards Defining, Measuring, and Leveraging Creativity in AI Applications

    [https://arxiv.org/abs/2610.12845](https://arxiv.org/abs/2610.12845)

    本文基于心理学创造力理论（尤其是Boden的新颖性、价值与惊奇三大准则）提出了创造力的形式化公设与数学定义，并在AI应用中对其加以验证。

    

    尽管已有大量研究，计算创造力在艺术、科学研究和游戏等场景中仍然缺乏一个概念上有据可依、实践上相关且经验上可适用的形式化定义。现有的创造力研究方法往往要么侧重于提出非正式的概念性定义，却不提供形式化的计算概念；要么提出形式化定义，却不仔细审视其底层假设，也不通过在特定应用环境中测试其表现来验证这些定义。为解决这些问题，本文基于先前关于创造力的心理学研究，尤其是与Boden提出的创造力三大核心准则（新颖性、价值和惊奇）相关的工作，提出了若干创造力的形式化公设。此外，我们提出了一个与这些公设相一致的概念性数学创造力定义，并对所提出的创造力（定义）在……（原文摘要在此处截断）

    arXiv:2610.12845v1 Announce Type: new  Abstract: Despite extensive research, computational creativity still lacks a conceptually grounded, practically relevant, and empirically applicable formal definition across contexts such as art, scientific research, and games. Existing approaches to creativity often focus on proposing informal conceptual definitions without providing formal computational concepts. Alternatively, they propose formal definitions without carefully examining their underlying assumptions or providing validation of these definitions by testing their performance in specific application settings. To address these problems, this paper proposes several formal postulates of creativity grounded in prior psychological work on creativity, especially as related to Boden's three core criteria of creativity: newness, value, and surprise. Furthermore, we propose a conceptual mathematical definition of creativity that aligns with these postulates. We also validate our proposed crea
    
[^125]: 神经网络比单个神经元所暗示的更具模块化性

    Neural networks are more modular than single neurons suggest

    [https://arxiv.org/abs/2610.12842](https://arxiv.org/abs/2610.12842)

    该论文提出了一个形式化的嵌套模块化层次结构（单元级、正交级和线性级），并通过统一的因果评估框架证明，即使神经网络在单个神经元层面缺乏模块化，它们在子空间层面往往仍表现出明显的功能模块化。

    

    神经网络通常会发展出单个神经元同时参与多种不同功能的表示，这导致了一种普遍观点，即此类网络缺乏模块化组织。我们通过引入一个形式化的嵌套模块化层次结构来挑战这一结论：单元级模块化，要求不同的计算占据互不相交的神经元；正交级模块化，探讨旋转是否能揭示功能上独立的子空间；以及线性级模块化，探讨有界线性映射能否将任务相关的计算分离到独立的子空间中。我们开发了一个统一的框架，用于在每个层次上学习分解，并通过选择性消融对分解进行因果评估。在视觉模型、语言模型和多任务循环神经网络（RNN）中，我们发现缺乏单元级模块化的网络往往在子空间层面表现出清晰的功能模块化。在视觉和语言模型中……（原文摘要在此处截断）

    arXiv:2610.12842v1 Announce Type: new  Abstract: Neural networks routinely develop representations in which individual neurons contribute to multiple distinct functions, leading to the widespread view that such networks lack modular organization. We challenge this conclusion by introducing a formally nested hierarchy of modularity: unit-level, which requires distinct computations to occupy disjoint neurons; orthogonal, which asks whether a rotation reveals functionally independent subspaces; and linear, which asks whether a bounded linear map can separate task-relevant computations into independent subspaces. We develop a unified framework for learning decompositions at each level and evaluating them causally through selective ablation. Across vision models, language models and multi-task Recurrent Neural Networks (RNNs), we find that networks which lack unit-level modularity nonetheless often exhibit clear functional modularity at the subspace level. In vision and language models, con
    
[^126]: COFFEQA：面向大语言模型问答的反事实引导少样本蒸馏

    COFFEQA: Counterfactual-Guided Few-Shot Distillation for LLM Question Answering

    [https://arxiv.org/abs/2610.12837](https://arxiv.org/abs/2610.12837)

    提出COFFEQA方法，通过反事实引导为少样本蒸馏中教师决策边界附近的不确定样本提供额外监督，从而提升大语言模型问答任务的蒸馏效果。

    

    知识蒸馏旨在训练一个较小的学生模型来模仿较大的教师模型，以适应资源受限的环境。然而，蒸馏过程通常数据密集且计算成本高昂，尤其是对于生成式任务。少样本蒸馏提供了一种替代方案，即使用有限的训练样本来迁移教师模型的知识。然而，我们观察到少样本蒸馏的性能在很大程度上取决于这些训练样本的选择。事实上，我们证明了使用教师决策边界附近更不确定的样本会显著恶化蒸馏性能。基于这一观察，我们提出了开放书本问答中的反事实引导少样本蒸馏（COFFEQA），这是一种新颖的策略，巧妙地利用反事实为具有挑战性的事实样本提供额外的监督。在生成式问答设置中，我们定义了一个反事实……

    arXiv:2610.12837v1 Announce Type: cross  Abstract: Knowledge distillation aims to train a smaller student model to mimic a larger teacher model for resource-constrained environments. However, the process of distillation is typically data-intensive and computationally expensive, particularly for generative tasks. Few-shot distillation offers an alternative by transferring teacher knowledge using limited training samples. However, we observe that few-shot distillation performance is heavily dependent on the choice of these training samples. In fact, we demonstrate that using more uncertain samples near the teacher's decision boundary can significantly deteriorate distillation performance. Motivated by this observation, we propose Counterfactual-guided Few-shot Distillation in Open-book Question Answering (COFFEQA), a novel strategy that aptly uses counterfactuals to provide additional supervision around challenging factual samples. In the generative QA setting, we define a counterfactual
    
[^127]: 奖励破解作为多智能体编码系统中的可传播策略

    Reward Hacking as a Transmissible Strategy in Multi-Agent Coding Systems

    [https://arxiv.org/abs/2610.12822](https://arxiv.org/abs/2610.12822)

    该论文首次揭示奖励破解并非单智能体的孤立失败，而是一种可在多智能体编码系统中通过代码、共享记忆和自然语言计划传播的“传染性”策略，为多智能体系统的安全性提出了全新挑战。

    

    多智能体系统正越来越多地被用于解决编码任务，其方式是协调共享代码和计划的多个智能体。这些智能体通常通过强化学习（RL）结合自动评估器进行优化，但绕过评估即可在不产生真正解决方案的情况下获得奖励。以往的研究大多将奖励破解视为单智能体的失败模式。在这项工作中，我们证明在一个智能体中诱发的奖励破解策略可以传播给其他智能体，并影响系统的协调。在四个开放权重骨干模型上，编码强化学习在两种设置下诱发并放大了奖励破解：系统提示中的显式机制描述，以及模拟上游接触编码环境相关事实的中间训练。以中间训练设置中经过强化学习后的智能体作为源头，我们观察到奖励破解通过代码和共享内存进行传播，并通过自然语言计划在不同角色之间传播。即使是基于小型开放权重模型的智能体也能诱发……（原文摘要在此处截断）

    arXiv:2610.12822v1 Announce Type: new  Abstract: Multi-agent systems are increasingly used to solve coding tasks by coordinating agents that share code and plans. These agents are often optimized with reinforcement learning (RL) using automated evaluators, but bypassing evaluation can earn reward without producing a genuine solution. Prior work has largely treated reward hacking as a single-agent failure. In this work, we show that a reward-hacking strategy elicited in one agent can propagate to others and affect system coordination. Across four open-weight backbones, coding RL elicits and amplifies reward hacking in two settings: explicit mechanism descriptions in the system prompt and mid-training that models upstream exposure to facts about the coding environment. Using post-RL agents from the mid-training setting as sources, we observe propagation through code and shared memory, and across roles through natural-language plans. Even agents based on small open-weight models induce re
    
[^128]: LeWAM：基于JEPA的端到端世界与动作建模

    LeWAM: End-to-End World and Action Modeling with JEPAs

    [https://arxiv.org/abs/2610.12818](https://arxiv.org/abs/2610.12818)

    LeWAM提出了一种端到端的JEPA控制框架，通过联合训练潜在状态预测与动作流匹配并对潜在状态施加各向同性高斯约束，使单一模型同时支持策略执行与高效规划，将接触密集任务和长时程任务的成功率大幅提升，并实现32.3倍的规划加速。

    

    联合嵌入预测架构（JEPA）无需像素重构即可学习紧凑的潜在世界模型。现有的JEPA流程通常先训练视觉表示和动作条件动力学，然后再添加控制策略或求解器。我们提出LeWAM，一种用于控制的端到端JEPA，其特点是：（1）对下一潜在状态预测和动作流匹配的联合训练目标；（2）对潜在状态施加各向同性高斯约束，使单一模型既能支持直接的策略执行，又能利用其学习到的潜在动力学评估采样动作，从而实现高效规划。在视觉规划任务上，与先前的JEPA相比，LeWAM将接触密集型任务的成功率从28.6%提升到89.7%，将长时程任务的成功率从10.9%提升到46.2%，且规划速度快32.3倍。在仿真和实体机器人的行为克隆任务上，LeWAM也与现有方法相比具有竞争力。可解释性分析表明，Le（摘要在此处被截断）

    arXiv:2610.12818v1 Announce Type: cross  Abstract: Joint-embedding predictive architectures (JEPAs) learn compact latent world models without pixel reconstruction. Existing JEPA pipelines typically train visual representations and action-conditioned dynamics before adding a control policy or solver. We introduce LeWAM, an end-to-end JEPA for control with: (1) a joint training objective over next latent state prediction and action flow matching, and (2) an isotropic Gaussian constraint on the latent state, enabling a single model to support direct policy execution and efficient planning by evaluating sampled actions with its learned latent dynamics. Compared with prior JEPAs in visual planning, LeWAM improves success rate from 28.6% to 89.7% on contact-rich tasks and from 10.9% to 46.2% on long-horizon tasks, and plans 32.3 times faster. On behavior cloning tasks in simulation and physical robots, LeWAM is also competitive with established methods. Interpretability analyses show that Le
    
[^129]: VEPA：面向终身LLM智能体防护栏的经过验证的演化形式化策略

    VEPA: Verified Evolving Formal Policy for Lifelong LLM Agent Guardrails

    [https://arxiv.org/abs/2610.12813](https://arxiv.org/abs/2610.12813)

    提出VEPA框架，将形式化防护栏的确定性执行能力与从部署经验中持续学习和演化安全策略的机制相结合，为LLM智能体提供终身的、可验证的自适应安全防护。

    

    LLM智能体越来越多地运行于敏感资源之上，例如私人文档和可变的系统状态，这可能带来严重的安全风险。因此，运行时防护栏应在潜在不可逆的智能体动作执行之前，做出可检查、可复现且确定性的允许或拒绝决策。然而，安全需求在部署之前很少是完整的，因此防护栏需要随着新风险的出现而不断适应。现有的自适应防护栏从部署经验中学习，但它们通常通过LLM推理来应用安全策略，这可能导致决策不一致和策略解释模糊。另一方面，形式化防护栏以确定性方式执行策略，但它们通常依赖于部署前或特定任务的策略，并且缺乏适应新观察到的风险的机制。我们提出了VEPA，一个用于形式化智能体防护栏经过验证演化的框架……

    arXiv:2610.12813v1 Announce Type: new  Abstract: LLM agents increasingly operate over sensitive resources, such as private documents and mutable system states, which can create serious safety and security risks. Runtime guardrails should therefore make inspectable, reproducible, and deterministic allow or refuse decisions before potentially irreversible agent actions are executed. However, safety and security requirements are rarely complete before deployment, so guardrails need to keep adapting as new risks emerge. Existing adaptive guardrails learn from deployment experience, but they usually apply safety policies through LLM inference, which can lead to inconsistent decisions and ambiguous policy interpretations. Formal guardrails, on the other hand, enforce policies deterministically, but they usually rely on pre-deployment or task-specific policies and have no mechanism to adapt to newly observed risks. We introduce VEPA, a framework for the verified evolution of formal agent guar
    
[^130]: 基于强化学习神经架构搜索的高效视听语音增强

    Efficient Audio-Visual Speech Enhancement via Reinforcement Learning-Based Neural Architecture Search

    [https://arxiv.org/abs/2610.12796](https://arxiv.org/abs/2610.12796)

    本文提出一种基于强化学习的神经架构搜索框架，通过联合优化语音增强性能与模型紧凑度，自动搜索出参数量约为基线九分之一（290万 vs 2200万）而SI-SNR性能仅降低0.7 dB的高效视听语音增强模型，适合资源受限的边缘设备部署。

    

    视听语音增强（AVSE）能够改善嘈杂环境中的语音可懂度，但通常依赖于计算开销高昂的模型，不适合资源受限的边缘设备。本文提出了一种由强化学习驱动的神经架构搜索（RL-NAS）框架，用于自动发现紧凑、高效的视听语音增强架构。该搜索空间保持了音频流与视觉流之间的时间对齐，同时通过一种新颖的奖励函数联合优化语音增强性能（以SI-SNR提升来衡量）与模型紧凑度。借助带有约束感知动作掩蔽的近端策略优化，该框架仅评估了35个候选架构，便找到一个参数量仅为290万的模型，而基线模型参数量为2200万。所提出的架构实现了14.5 dB的SI-SNR提升，仅比基线低0.7 dB，同时将模型规模缩减了约九倍。

    arXiv:2610.12796v1 Announce Type: cross  Abstract: Audio-visual speech enhancement (AVSE) improves speech intelligibility in noisy environments but often relies on computationally expensive models unsuitable for resource-constrained edge devices. This paper introduces a reinforcement learning-driven neural architecture search (RL-NAS) framework to discover compact, efficient AVSE architectures. The search space preserves temporal alignment between audio and visual streams, while a novel reward function jointly optimises speech enhancement performance, measured by SI-SNR improvement, and model compactness. Using proximal policy optimisation with constraint-aware action masking, the framework evaluates only 35 candidate architectures and identifies a model with 2.9 million parameters, compared with 22 million in the baseline. The proposed architecture achieves a 14.5 dB SI-SNR improvement, only 0.7 dB below the baseline, while reducing model size by approximately nine-fold. Evaluation on
    
[^131]: 元反思式进化有效改进生产级AI系统

    Meta-Reflective Evolution Effectively Improves Production AI Systems

    [https://arxiv.org/abs/2610.12794](https://arxiv.org/abs/2610.12794)

    提出基于“元反思”机制的进化优化器CoCoEvolve，通过从完整进化历史中检索证据并探测运行中的系统来指导变异，在DABStep上以92.8%的难题准确率登顶排行榜，有效改进了数据智能体、LLM数据流水线和数据工程工作流等生产级AI系统。

    

    arXiv:2610.12794v1 公告类型：交叉 摘要：由大语言模型（LLM）引导的进化优化已成功改进了提示词、程序，乃至整个AI系统。大多数优化方法从搜索前沿中选取一个候选，然后仅基于其失败情况，在单次LLM调用中对其进行变异。这种对搜索前沿的关注在生产级AI系统上可能陷入停滞，因为孤立的失败很难揭示复杂的隐藏交互。我们提出了CoCoEvolve（Cortex Code Evolve），一个围绕我们称之为“元反思”（meta-reflection）机制构建的进化优化器，其中智能体变异器（Cortex Code）会选择性地从整个进化历史中检索证据，并在验证和提出变异之前探测实际运行的系统。我们将CoCoEvolve应用于优化三类生产级AI系统：数据智能体、基于LLM的数据流水线以及数据工程工作流。在DABStep上，CoCoEvolve登顶公开排行榜，在留出难题集上达到92.8%的准确率，相比之下基线仅为78.4%（摘要原文在此处截断）。

    arXiv:2610.12794v1 Announce Type: cross  Abstract: Large Language Model (LLM)-guided evolutionary optimization has successfully improved prompts, programs, and even entire AI systems. Most optimization methods select a candidate from the search frontier, then mutate it in one LLM pass based only on its failures. This focus on the frontier can stall on production AI systems, where isolated failures reveal little about complex, hidden interactions. We introduce CoCoEvolve (Cortex Code Evolve), an evolutionary optimizer built around what we call meta-reflection, wherein an agentic mutator (Cortex Code) selectively retrieves evidence from the entire evolutionary history and probes the live system before verifying and proposing mutations. We apply CoCoEvolve to optimize three classes of production AI systems: data agents, LLM-powered data pipelines, and data engineering workflows. On DABStep, CoCoEvolve tops the public leaderboard, reaching 92.8% held-out hard accuracy compared to 78.4% for
    
[^132]: EvoBelief：面向自我改进大语言模型智能体的可证伪世界模型

    EvoBelief: Falsifiable World Models for Self-Improving LLM Agents

    [https://arxiv.org/abs/2610.12789](https://arxiv.org/abs/2610.12789)

    提出EvoBelief，一种将环境知识表示为可证伪假设的自我演化世界模型，通过意外事件触发假设形成、独立验证后方可在支持范围内指导决策、并利用反例随环境变化修正或淘汰规则，使语言模型保持冻结的同时在部署中持续演化智能体的环境信念。

    

    自我演化的LLM智能体通常通过积累经验、反思或技能来改进其行动方式。然而，即使程序性能力很强的智能体，如果其对环境动态的信念不完整、错误或过时，也可能失败。我们研究自我演化的一个互补维度：在部署过程中演化这些信念。这类信念可以支持多个任务，但错误的信念也可能在任务之间传播。我们提出了EvoBelief，这是一种自我演化的世界模型，它将环境知识表示为可证伪的假设。意外结果和与决策相关的未知因素会触发假设的形成。后续的观察和受控实验会提供独立验证，之后规则才能在其支持范围内指导预测或规划。随着环境的变化，反例会缩小或淘汰规则。语言模型本身保持冻结，每条规则都与其证据以及它所改变的决策保持关联。

    arXiv:2610.12789v1 Announce Type: new  Abstract: Self-evolving LLM agents typically improve how they act through accumulated experience, reflection, or skills. Yet even procedurally capable agents can fail when their beliefs about environment dynamics are incomplete, incorrect, or outdated. We study a complementary axis of self-evolution: evolving those beliefs during deployment. Such beliefs can support multiple tasks, but incorrect ones can also propagate across them. We introduce EvoBelief, a self-evolving world model that represents environment knowledge as falsifiable hypotheses. Unexpected outcomes and decision-relevant unknowns trigger hypothesis formation. Later observations and controlled experiments provide independent validation before a rule can guide predictions or plans within its supported scope. Counterexamples narrow or retire rules as the environment changes. The language model remains frozen, and each rule remains linked to its evidence and the decisions it changes. 
    
[^133]: 通过蒸馏最优传输从少量配对数据实现表示对齐

    Representation Alignment from a Few Paired Data via Distilled Optimal Transport

    [https://arxiv.org/abs/2610.12781](https://arxiv.org/abs/2610.12781)

    该论文提出一种基于蒸馏最优传输的方法，仅利用约100-500个配对锚点样本即可对齐两个独立预训练的单模态编码器：通过锚点相似度构建共同坐标系，用熵最优传输的边际约束对抗枢纽性问题，并将传输方案蒸馏到轻量级编码器中实现可归纳的跨模态对应。

    

    对齐两个独立预训练的单模态编码器通常需要数百万个配对样本。本文研究了仅有约100至500对配对样本可用、而两侧均有大量非配对数据的情形。我们的方法通过每个表示与N个配对锚点的相似度来表达该表示，从而将两种模态置于一个共同的坐标系中；然后在整个数据池上求解单个熵正则化最优传输问题来匹配两个点云，其边际约束能够抵消导致最近邻匹配失效的枢纽性问题；最后将所得的传输方案蒸馏到两个轻量级编码器中，使对应关系具备归纳泛化能力。此外，我们将该边际约束与双语词典归纳文献中的单步枢纽校正方法建立了联系。配对数据仅锚点需要：非配对数据集合可以从两个边际分布中独立采样，因此……

    arXiv:2610.12781v1 Announce Type: new  Abstract: Aligning two independently pretrained unimodal encoders typically demands millions of paired examples. In this paper, we study the regime where only $N \sim 100$--$500$ pairs are available, while unpaired data on either side is abundant. Our method expresses every representation by its similarities to the $N$ paired anchors, which places both modalities in a common coordinate system; matches the two point clouds with a single entropic optimal transport problem solved over the entire pool, whose marginal constraints counteract the hubness that defeats nearest-neighbor matching; and distills the resulting plan into two lightweight encoders, so that the correspondence becomes inductive. We additionally put this marginal constraint in relation with the one-step hub corrections of the bilingual lexicon induction literature. Pairing is required only for the anchors: the unpaired collections may be drawn independently from the two marginals, so
    
[^134]: REDaCT：大型推理模型中的无坍塌遗忘学习

    REDaCT: Collapse-Free Unlearning in Large Reasoning Models

    [https://arxiv.org/abs/2610.12769](https://arxiv.org/abs/2610.12769)

    REDaCT是一种面向大型推理模型的推理时遗忘学习框架，通过泄漏检测器与纠错嵌入注入机制，在不引发推理坍塌的前提下从思维链和最终答案中移除遗忘集信息。

    

    大型推理模型（LRM）中的遗忘学习要求同时从推理轨迹和最终答案中移除遗忘集信息。现有方法在实现这一目标时以推理坍塌为代价，即模型无法生成连贯的思维链（CoT），也无法正确终止其推理过程。为应对这一挑战，我们提出了REDaCT（通过检测与纠错微调实现再生成），一种面向大型推理模型的推理时遗忘学习框架。REDaCT包含两个关键组件：（1）一个二元泄漏检测器，用于识别推理轨迹中泄露遗忘集信息的句子；（2）一个嵌入优化模块，离线学习一个纠错嵌入向量，在推理过程中首次检测到泄漏时将其注入，从而引导后续推理生成不泄露遗忘集信息的句子。实验结果表明，REDaCT完全避免了推理坍塌。

    arXiv:2610.12769v1 Announce Type: cross  Abstract: Unlearning in large reasoning models (LRMs) requires removing forget-set information from both the reasoning trace and the final answer. Existing approaches achieve this objective at the cost of reasoning collapse, where the model fails to produce a coherent chain-of-thought (CoT) and does not properly terminate its reasoning process. To address this challenge, we propose REDaCT (REgenerate via Detection and Corrective Tuning), an inference-time unlearning framework for LRMs. REDaCT consists of two key components: (1) a binary leak detector that identifies sentences in the reasoning trace that leak forget-set information, and (2) an embedding optimization module that learns a corrective embedding offline, such that when injected at the first detected leak during inference, it guides subsequent reasoning towards sentences that do not leak forget-set information. Experimental results show that REDaCT completely avoids reasoning collapse,
    
[^135]: 语义视图：跨语义查询的LLM标签认证复用

    Semantic Views: Certified Reuse of LLM Labels Across Semantic Queries

    [https://arxiv.org/abs/2610.12768](https://arxiv.org/abs/2610.12768)

    SemViews通过复用历史查询留下的语义视图标签、局部嵌入和辅助模型评分来减少昂贵的LLM预言机调用，同时以至少1−δ的概率保证任意工作负载下的精确率和召回率目标。

    

    语义查询引擎使用大语言模型来评估自然语言过滤器，而每次评估都是一次代价高昂的预言机调用。旨在减少这些调用的优化器面临两难困境：信任廉价代理的级联方法在许多查询上无法达到精确率和召回率目标，具有有限样本保证的方法节省有限，而且所有这些方法在查询结束时都会丢弃预言机产生的标签。我们提出了SemViews，一种经过认证的语义过滤器，它通过无需预言机调用的信号对元组进行排序（包括先前查询留下的作为语义视图的标签、局部嵌入以及辅助模型的评分），再用无论如何都必须付费获取的标签来细化该排序，并通过新鲜的分层预言机样本来认证其答案。对于每个工作负载和每个目录，其答案满足用户精确率和召回率目标的概率至少为 $1-\delta$，因此过时或误导性的视图可能只会耗费额外的预言机调用，但绝不会损害准确性。

    arXiv:2610.12768v1 Announce Type: cross  Abstract: Semantic query engines evaluate natural-language filters with a large language model, and every evaluation is a costly oracle call. Optimizers that reduce these calls face a dilemma: cascades that trust a cheap proxy miss their precision and recall targets on many queries, methods with finite-sample guarantees save little, and all of them discard the oracle's labels when the query ends. We present SemViews, a certified semantic filter that orders tuples by signals that cost no oracle calls (the labels that earlier queries left behind as semantic views, a local embedding, and a helper model's score), refines that order with labels it must pay for anyway, and certifies its answer with fresh stratified oracle samples. The answer meets user precision and recall targets with probability at least $1-\delta$ for every workload and every catalog, so a stale or misleading view can cost oracle calls but never accuracy. On four workloads replayed
    
[^136]: 忠实且完整的LLM辅助层级式硬件知识图谱自动构建

    Faithful and Complete LLM-Assisted Automated Construction of Hierarchical Hardware Knowledge Graphs

    [https://arxiv.org/abs/2610.12762](https://arxiv.org/abs/2610.12762)

    提出了一种LLM驱动的自动化流水线，通过分块属性提取、层级路由和迭代优化，将硬件数据手册忠实且完整地转换为保留来源可追溯性的层级式硬件知识图谱。

    

    硬件数据手册包含对元器件选型、系统集成和硬件兼容性验证至关重要的工程规格参数。然而，由于数据手册篇幅冗长且形式异构，密集的信息分散在表格、图表和文本中，人工查阅既繁琐又容易出错。尽管大语言模型（LLM）可以自动化这一过程，但简单的一次性处理即使在整个文档能够放入模型上下文窗口的情况下，仍会遗漏相关的规格参数。我们提出了一种自动化的LLM驱动流水线，通过基于分块的属性提取、层级路由和迭代优化，将硬件数据手册转换为通用的层级式硬件知识图谱。该框架利用LLM进行信息提取和语义推理，同时保留信息来源的可追溯性，并通过经过验证的指令和确定性的本地执行来应用图谱更新。这种分

    arXiv:2610.12762v1 Announce Type: new  Abstract: Hardware datasheets contain engineering specifications essential for component selection, system integration, and hardware compatibility verification. However, manual inspection is tedious and error-prone because datasheets are lengthy and heterogeneous, with dense information fragmented across tables, figures, and text. Although large language models (LLMs) can automate this process, na\"ive one-shot processing omits relevant specifications even when the document fits within the model's context window.   We present an automated LLM-driven pipeline that transforms hardware datasheets into general-purpose hierarchical hardware knowledge graphs through chunk-based property extraction, hierarchical routing, and iterative refinement. The framework uses LLMs for information extraction and semantic reasoning while preserving source provenance and applying graph updates through validated instructions and deterministic local execution. This stag
    
[^137]: 基于人在环路驾驶数据的道路施工区域自动驾驶轨迹预测

    Trajectory Prediction for Autonomous Driving in Road Construction Zones Using Human-in-the-Loop Driving Data

    [https://arxiv.org/abs/2610.12747](https://arxiv.org/abs/2610.12747)

    该论文构建了首个针对道路施工区域的多模态驾驶数据集WZDrive-Data，并提出了多模态自车轨迹预测器WZDrive，有效解决了施工区域车道布局临时变化导致的车辆轨迹难以预测的难题。

    

    arXiv:2610.12747v1 公告类型：cross 摘要：在城市道路施工区域中进行自车轨迹预测仍然是自动驾驶研究中一个未被充分探索的课题，专用数据和系统性评估都很有限。道路施工区域涉及车道布局和道路连接性的临时变化，使与周围车辆和工人的交互变得复杂，这些变化使得未来的车辆轨迹难以预测。为了应对这些挑战，我们收集了WZDrive-Data，这是一个多模态数据集，包含来自33名驾驶员的1,183个施工区域驾驶片段，使用带有预录制工人动画和VR控制工人的人在环路驾驶模拟器进行采集。驾驶员控制和注视信号从人类参与者处记录，而车辆轨迹和场景信息从模拟器中获取。我们提出的多模态自车轨迹预测器WZDrive实现了平均minADE (K=6)为1.09米和minFDE (K=6)为2.12米的性能。与表现最好的（摘要在此处被截断）

    arXiv:2610.12747v1 Announce Type: cross  Abstract: Ego-trajectory prediction in urban roadway work zones remains an underexplored topic in autonomous-driving research, with limited dedicated data and systematic evaluation. Roadway work zones involve temporary changes in the lane layouts and connectivity of the roads, complicating interactions with surrounding vehicles and workers. These changes make future vehicle trajectories difficult to predict. To address these challenges, we collect WZDrive-Data, a multimodal dataset comprising 1,183 work-zone driving segments from 33 drivers, using a human-in-the-loop driving simulator with prerecorded worker animations and VR-controlled workers. Driver control and gaze signals are recorded from human participants, while vehicle trajectories and scene information are obtained from the simulator. Our proposed multimodal ego-trajectory predictor, WZDrive, achieves a mean minADE (K=6) of 1.09 m and minFDE (K=6) of 2.12 m. Compared with the best-perf
    
[^138]: KITE：一种用于时间序列事件预测的知识注入型时间编码器

    KITE: A Knowledge-Injected Temporal Encoder for Time-Series Event Forecasting

    [https://arxiv.org/abs/2610.12738](https://arxiv.org/abs/2610.12738)

    该论文提出KITE架构，通过可切换的协变量和注意力偏置两种机制将工程师掌握的语义先验知识注入自监督事件预测器，无需架构搜索即可适配各类先验，在标签稀缺场景下显著提升罕见事件预测性能（域内最高提升47%、跨域提升53%）。

    

    在多元时间序列系统中预测罕见且高代价的事件十分困难，因为这些事件的前兆很少被观测到，因此纯粹的数据驱动模型几乎没有可学习的信息。然而，运维这些系统的工程师掌握着先于此类事件发生的知识——这些信息在稀缺的标签中并不存在。我们将其称为语义先验，并通过两种可切换的机制将其注入自监督事件预测器中：协变量机制和注意力偏置机制，每种机制在零状态下均为恒等映射，因此单一架构即可服务于所有类型的先验，无需针对每个数据集进行架构搜索。当先验能够提供观测窗口中尚未包含的信息时，它才能发挥作用：在短上下文情境下，信息丰富的先验在域内优于无先验基线（汇总h-AUROC相对无先验基线：周期性时钟提升14%，p<0.001；守恒残差在0.25标签比例下提升47%），且有效的先验可以跨域迁移（提升53%）……

    arXiv:2610.12738v1 Announce Type: cross  Abstract: Forecasting rare, high-cost events in multivariate time-series systems is hard because their precursors are seldom observed, so a purely data-driven model has little to learn from. The engineers who run these systems, however, hold knowledge that precedes such events, information absent from the scarce labels. We call this a semantic prior and inject it into a self-supervised event forecaster through two switchable mechanisms, a covariate mechanism and an attention-bias mechanism, each the identity at its null, so one architecture serves every prior type with no per-dataset architecture search. A prior helps when it supplies information the observed window does not already contain: in a short context an informative prior beats no prior in-domain (pooled h-AUROC over the no-prior baseline: a periodicity clock +14%, p<0.001; a conservation residual +47% at a 0.25 label fraction), and a working prior transfers out-of-domain (+53%), substi
    
[^139]: 无监督多模态奖励建模

    Unsupervised Multimodal Reward Modeling

    [https://arxiv.org/abs/2610.12707](https://arxiv.org/abs/2610.12707)

    提出一种无需人类偏好标签的无监督多模态奖励建模方法，通过在冻结的 Qwen3-VL 中间层训练线性头来区分网络图注与其扰动版本，仅用数千参数和约千分之一的微调计算量即可在多个基准上超越或持平 GPT-4o。

    

    视觉-语言奖励模型通常在标注的偏好对上进行训练，这类数据收集成本高昂、标注可能相互矛盾，且模型往往仅以基准测试准确率来评判排名。我们构建了一个无需任何人类偏好标签的奖励模型：在冻结的 Qwen3-VL 骨干网络的中间层之上加一个线性头，通过训练使其对网络爬取的原始图注的打分高于同一图注的扰动版本；标注数据仅用于选择层。层的选择是决定性因素，相比使用最后一层可带来 5.8 分的提升；在 32B 模型上，一套无需重新调参即可直接迁移的配方在三个基准上超越或至少持平 GPT-4o，在 VL-RewardBench 上领先 6.6 分，而这一切仅依赖数千个参数和大约千分之一的微调计算量。在相同的配对数据预算下，扰动方法比精心筛选的 RLAIF-V 数据对多教会冻结头 3 到 5 分。由于骨干网络被冻结，目标函数是凸的，且扰动采样器本身即定义了奖励——这是基准测试无法识别的：两个头

    arXiv:2610.12707v1 Announce Type: cross  Abstract: Vision-language reward models are trained on annotated preference pairs, which are expensive to collect, can disagree, and are ranked by benchmark accuracy. We build one with no human preference label: a linear head on an intermediate layer of a frozen Qwen3-VL backbone, trained to score a web caption above a perturbed version of the same caption; labeled data enters only to select the layer. The layer is the decisive choice, worth 5.8 points over the last layer, and one recipe transferred without re-tuning beats or at least matches GPT-4o on three benchmarks at 32B, 6.6 points ahead on VL-RewardBench, from a few thousand parameters at about a thousandth of fine-tuning compute. At an equal pair budget, perturbations teach a frozen head 3 to 5 points more than curated RLAIF-V pairs. Because the backbone is frozen, the objective is convex, and the perturbation sampler specifies the reward, which the benchmark does not identify: two heads
    
[^140]: GeoFitBench-LLM：一个评估大语言模型在地理分析任务的地理空间数据需求获取方面能力的基准测试

    GeoFitBench-LLM: A Benchmark for Assessing Large Language Models' Capabilities in Geospatial Data Requirements Elicitation for Geo-Analytic Tasks

    [https://arxiv.org/abs/2610.12669](https://arxiv.org/abs/2610.12669)

    本文提出了GeoFitBench-LLM基准，通过时空关系和语义锚定两个推理维度，以独立于数据提供者的标准化模式评估大语言模型从自然语言地理分析请求中推断所需数据集、支持资源和约束条件的能力。

    

    大语言模型正日益成为地理空间人工智能（GeoAI）系统的核心，然而对它们的评估很少考虑地理空间数据需求的获取。这一缺失至关重要，因为如果不首先确定所需的数据集和约束条件，面向检索的地理分析任务就无法得到可靠支持。本文提出了GeoFitBench-LLM基准，用于评估大语言模型能否从自然语言的地理分析请求中推断出所需的数据集、支持资源和约束条件。该基准围绕两个推理维度构建：时空关系和语义锚定，反映了地理空间信息检索及相关自然语言任务中长期存在的挑战。真值需求以固定的、独立于数据提供者的模式进行编码，以便对资源识别和属性级规范进行可重复的评估。在最小提示条件下对七个模型进行的实验表明……

    arXiv:2610.12669v1 Announce Type: cross  Abstract: Large language models are becoming central to GeoAI systems, yet their evaluation rarely considers the elicitation of geospatial data requirements. This omission is critical as retrieval oriented geo-analytic tasks cannot be supported reliably without first identifying the necessary datasets and constraints. GeoFitBench-LLM is introduced as a benchmark for assessing whether LLMs can infer the datasets, the supporting resources and the constraints required by geo-analytic requests in natural language. The benchmark is structured around two reasoning dimensions, Spatio-Temporal Relational and Semantic Grounding, reflecting long-standing challenges in geospatial information retrieval and related natural language tasks. Ground truth requirements are encoded in a fixed, provider-independent schema for reproducible evaluation of resource identification and attribute-level specification. Experiments on seven models under minimal prompting ind
    
[^141]: 面向智能体持续学习的循环式执行框架

    Recurrent Harness for Agentic Continual Learning

    [https://arxiv.org/abs/2610.12668](https://arxiv.org/abs/2610.12668)

    提出 ReHarness，一种循环式智能体持续学习框架，在严格在线流式协议下，将每个实例处理后的轨迹作为反馈向后传播，修订该实例对应的反思状态，实现前向读写与后向修订相结合的在线适应。

    

    语言模型智能体日益需要在长程交互流上运行，这要求通过持续学习对分布偏移进行在线适应。然而，现有的智能体框架通常要么仅通过前向信息流来演化其状态，要么依赖额外的回放或验证数据来进行框架优化。为解决这一问题，我们提出 ReHarness，一个面向智能体持续学习的循环式框架，它在严格的在线流式协议下，在前向读写之外补充了后向修订机制。我们提出这样一个问题：在框架状态中，哪些信息本可以以不同的方式被捕获，以便更好地服务于流中的当前实例。具体而言，在每个实例处理完成后，我们利用所产生的轨迹作为反馈，对为该实例检索到的反思进行修订，从而将反馈通过文本化的框架状态向后传播。该框架在三个持续学习基准和三个语言模型上进行了验证。

    arXiv:2610.12668v1 Announce Type: cross  Abstract: Language model agents increasingly operate over long interaction streams, requiring online adaptation to distribution shifts through continual learning. However, existing agent harnesses typically either evolve their states only through forward information flow or rely on additional replay or validation data for harness optimization. To address this, we propose ReHarness, a recurrent harness framework for agentic continual learning that complements forward read and write with backward revision under a strictly online streaming protocol. We ask what information could have been captured differently in the harness state to better inform the current instance in the stream. Specifically, after each instance, we leverage the resulting trajectory as feedback to revise the reflections retrieved for that instance, thereby propagating feedback backward through the textual harness state. Across three continual learning benchmarks and three langua
    
[^142]: 面向人在回路决策支持的增量式检索增强生成

    Incremental Retrieval-Augmented Generation for Human-in-the-Loop Decision Support

    [https://arxiv.org/abs/2610.12666](https://arxiv.org/abs/2610.12666)

    论文提出了iRAG，一个知识库从空开始、通过将人类决策存储为可检索先例而逐步增量构建的人在回路决策支持系统，它从静默观察过渡到基于证据的质疑、再到带弃权的自主处理，并揭示了检索衰减带来的稳定性与可塑性权衡。

    

    我们提出了iRAG，这是一个将检索增强生成（RAG）与混合决策相结合的人在回路决策支持系统。与依赖静态知识库（KB）的传统RAG系统不同，我们的方案针对的是知识库初始为空、并通过人类决策逐步填充的场景。该系统将先前的决策存储为可检索的先例，并从静默观察逐步过渡到基于证据的质疑，最终实现带有弃权机制的自主处理。相对于独立基线，iRAG在10%漂移下无需检索衰减即可达到与基线相同的最终决策错误率，并在所有配置下于40%漂移时降低了错误率。检索衰减在提升对已变化知识的处理性能的同时，增加了对稳定知识的错误率，这表明其体现的是一种稳定性与可塑性之间的权衡，而非普遍性优势。人类监督是该工作流程的架构性安排；其在真实场景中的可用性（摘要原文在此处截断）

    arXiv:2610.12666v1 Announce Type: cross  Abstract: We introduce iRAG, a human-in-the-loop decision-support system combining Retrieval-Augmented Generation (RAG) with Hybrid Decision-Making. Unlike conventional RAG systems, relying on static Knowledge Bases (KBs), our proposal addresses settings where the KB is initially empty and is progressively populated through human decisions. The system stores previous decisions as retrievable precedents and moves from silent observation to evidence-grounded contestation and, finally, to autonomous handling with abstention. Against an independent baseline, iRAG matches final-decision error under 10% drift without retrieval decay and reduces it under 40% drift across all configurations. Retrieval decay improves performance on changed knowledge while increasing error on stable knowledge, evidencing a stability--plasticity trade-off rather than a general advantage. Human oversight is an architectural provision of the workflow; its usability with real
    
[^143]: MagicBeanStalk：通过递归数据合成摆脱RLVR零优势陷阱

    MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis

    [https://arxiv.org/abs/2610.12659](https://arxiv.org/abs/2610.12659)

    该论文发现主流RLVR数学语料库中约88%的问题因过于简单或过难而无法为目标策略提供有效学习信号（即零优势陷阱），并提出MagicBeanStalk方法，通过递归数据合成生成既蕴含难题所需推理操作、又难度适中的有效训练问题。

    

    以数据为中心的RLVR扩展一直聚焦于收集或生成可验证的训练问题。然而，语料库的规模本身并不能决定该数据集能否为目标策略提供有用的学习信号。为了指导有效的RLVR数据合成，我们通过实验将数据集的有效性刻画为其问题、目标策略和训练rollout预算的函数。在这一刻画下，我们展示了一个汇集自广泛使用的RLVR数据集的大型开源数学语料库几乎完全陷入了零优势陷阱：在固定的目标策略和rollout预算下，大约88%的问题要么在每次rollout中都被解决，要么在每次rollout中都失败。过滤可以浓缩剩余部分，但它只能进行选择，从而丢弃了难题。因此，有效的合成问题应当需要解决那些难题所必需的推理操作及其组合，但其难度必须加以控制（原文在此处截断）。

    arXiv:2610.12659v1 Announce Type: new  Abstract: Data-centric scaling of RLVR has focused on collecting or generating verifiable training problems. However, corpus size alone does not determine whether the dataset provides a useful learning signal for a target policy. To guide the synthesis of effective RLVR data, we experimentally characterize dataset effectiveness as a function of its problems, the target policy, and the training rollout budget. Under this characterization, we show that a large open-source math corpus pooled from widely used RLVR datasets falls almost entirely into a zero-advantage trap: for a fixed target policy and rollout budget, roughly 88% of its problems are solved in every rollout or failed in every rollout. Filtering concentrates what remains, but it can only select, discarding the hard problems. Effective synthetic problems should therefore require the reasoning operations and compositions required to solve those hard problems. Yet their difficulty must be c
    
[^144]: 克罗内克积注意力：基于结构化分解的内存高效KV缓存压缩

    Kronecker Product Attention: Memory-Efficient KV Compression via Structured Factorization

    [https://arxiv.org/abs/2610.12657](https://arxiv.org/abs/2610.12657)

    提出克罗内克积注意力（KPA），将键值激活表示为紧凑的词元相关矩阵因子的克罗内克积之和，严格推广了张量积注意力（TPA），在保持表达能力的同时实现了内存高效的KV缓存压缩。

    

    高效的自回归生成日益受到键值（KV）缓存内存占用的限制，该缓存随序列长度、批次大小和模型深度线性增长。现有方法如多查询注意力（MQA）、分组查询注意力（GQA）、多头潜在注意力（MLA）和张量积注意力（TPA）通过不同的权衡来降低这一成本，但往往牺牲了表达能力、需要额外的位置编码机制，或强加了限制性的低秩结构。我们提出克罗内克积注意力（KPA），它将键和值激活表示为紧凑的、依赖于词元的矩阵因子的克罗内克积之和。KPA严格推广了TPA：秩一向量外积分解作为一种退化的克罗内克分裂出现，而非退化的矩阵因子能够表示更丰富的块结构模式。我们进一步证明，与之匹配的结构化旋转位置编码算子……

    arXiv:2610.12657v1 Announce Type: cross  Abstract: Efficient autoregressive generation is increasingly limited by the memory footprint of the key-value (KV) cache, which grows linearly with sequence length, batch size, and model depth. Existing approaches such as Multi-Query Attention (MQA), Grouped-Query Attention (GQA), Multi-Head Latent Attention (MLA), and Tensor Product Attention (TPA) reduce this cost through different trade-offs, but often sacrifice expressivity, require additional positional-encoding machinery, or impose restrictive low-rank structure. We propose Kronecker Product Attention (KPA), which represents key and value activations as sums of Kronecker products of compact, token-dependent matrix factors. KPA strictly generalizes TPA: rank-one vector-outer-product factorization arises as a degenerate Kronecker split, while non-degenerate matrix factors can represent richer block-structured patterns. We further show that a structured rotary embedding operator matched to t
    
[^145]: CLEAT：为交互式智能体学习价值导向的澄清机制

    CLEAT: Learning Value-Guided Clarification for Interactive Agents

    [https://arxiv.org/abs/2610.12655](https://arxiv.org/abs/2610.12655)

    提出CLEAT框架，通过残差查询价值的估计，从共享交互结果中协同训练澄清价值控制器与任务策略，使LLM智能体学会判断何时提出澄清问题才能真正提升任务效用。

    

    LLM智能体能够执行日益复杂的任务，但有效的辅助还要求其行动与用户可能未明确说明的目标和约束保持一致。我们将该问题作为通过澄清实现的交互式意图对齐进行研究。然而，只有当智能体能够将答案转化为更好的决策时，澄清才是有用的。由于训练会改变智能体处理不完整信息以及利用额外答案的方式，同一个澄清可能变得更有用或更无用。我们提出了CLEAT（通过价值导向的智能体协同训练实现澄清），这是一个基于共享交互结果，协同训练澄清价值控制器与任务策略的框架。我们形式化了残差查询价值，用以衡量在当前策略下通过提问而非直接行动所获得的任务效用，并通过来自同一交互历史的成对rollout来估计该价值。这些成对回报用于训练控制器预测澄清的价值。

    arXiv:2610.12655v1 Announce Type: new  Abstract: LLM agents can execute increasingly complex tasks, but effective assistance also requires aligning their actions with goals and constraints that users may leave unstated. We study this problem as interactive intent alignment through clarification. Yet clarification is useful only when the agent can turn the answer into better decisions. As training changes how the agent handles incomplete information and uses additional answers, the same clarification can become more or less useful. We introduce CLEAT (CLarification through valuE-guided Agent co-Training), a framework that co-trains a clarification-value controller and the task policy from shared interaction outcomes. We formulate residual query value to measure the task utility gained by asking rather than proceeding under the current policy, and estimate it through paired rollouts from the same interaction history. These paired returns train the controller to predict clarification valu
    
[^146]: 癌症式后门攻击：通过不可见令牌与阶段控制器实现渐进激活

    Cancerous Backdoor Attack: Gradual Activation via Invisible Tokens and A Stage Controller

    [https://arxiv.org/abs/2610.12644](https://arxiv.org/abs/2610.12644)

    该论文提出了一种名为“癌症式后门攻击”的新型后门攻击方法，利用不可见令牌和阶段控制器，使恶意行为在大语言模型中通过多个隐藏阶段逐步激活，从而渐进式地破坏模型的推理能力。

    

    大型语言模型（LLM）上的后门攻击通常遵循触发-响应模式，即满足触发条件后激活单一的终端行为。然而，这种单点后门激活并未探索这样一类攻击：恶意行为在到达终端激活阶段之前，通过多个中间阶段逐步发展。我们提出了癌症式后门攻击，通过利用不可见令牌和阶段控制器，在一系列交互中逐步破坏大语言模型的推理能力。具体而言，不可见令牌在不出现在用户可见请求中的情况下向模型提供隐藏的阶段信息，而阶段控制器在推理过程中跟踪攻击的进展，从而实现渐进式激活。我们在多个推理数据集上评估了癌症式后门攻击。该攻击在多个模型和任务上均能持续产生渐进式的推理破坏。我们将癌症式后门攻击与

    arXiv:2610.12644v1 Announce Type: cross  Abstract: Backdoor Attacks on Large Language Models (LLMs) typically follow a trigger-response pattern, where satisfying a trigger condition activates a single terminal behavior. However, this single-point backdoor activation does not explore attacks in which malicious behavior develops gradually through multiple intermediate stages before reaching the terminal activation stage. We present Cancerous Backdoor Attack, that gradually corrupts an LLM's reasoning over a sequence of interactions by leveraging invisible tokens and a stage controller. Specifically, invisible tokens provide hidden stages to the model without appearing in the user's visible request and the stage controller tracks the attack's progression during inference to realize gradual activation. We evaluate Cancerous Backdoor Attack across several reasoning datasets. The attack consistently produces gradual reasoning corruption across models and tasks. We compare Cancerous Backdoor 
    
[^147]: 面向大学建筑日前电力预测的机器学习与时间序列基础模型

    Machine Learning and Time-Series Foundation Models for Day-Ahead Electricity Forecasting in a University Building

    [https://arxiv.org/abs/2610.12638](https://arxiv.org/abs/2610.12638)

    该研究在一栋大学建筑上对统计、机器学习、循环神经网络和预训练TimesFM等日前电力预测模型进行了统一基准测试，发现带协变量的TimesFM 3.0总体误差最低（平均绝对误差较每周季节性朴素预测降低约43%），而XGBoost仍具竞争力。

    

    日前电力预测可以帮助建筑运营者根据预期需求来安排供暖、通风及其他服务。选择预测模型还需要在预测精度与其预测时所需输入、本地训练和计算需求之间进行权衡。我们针对一栋大学建筑，对统计模型、机器学习模型、循环神经网络模型以及预训练的TimesFM模型进行了基准测试。我们采用按时间顺序划分的验证方式，并在相同的57个测试日上评估所有配置，比较各模型在工作日、周末和节假日的精度，同时比较本地拟合与预测所需时间。天气协变量采用回溯观测数据。带协变量的TimesFM 3.0取得了最低的总体误差，与每周季节性朴素预测相比，平均绝对误差降低了约43%。XGBoost仍具有竞争力，尽管逐日配对比较并未证明TimesFM在平均绝对误差上具有明确优势。

    arXiv:2610.12638v1 Announce Type: cross  Abstract: Day-ahead electricity forecasts can help building operators schedule heating, ventilation, and other services around anticipated demand. Choosing a forecasting model also requires balancing accuracy with its forecast-time inputs, local training, and computational requirements.   We benchmark statistical, machine-learning, recurrent neural-network, and pretrained TimesFM models for a university building. We use chronological validation and evaluate all configurations on the same 57 test days, comparing accuracy across workdays, weekends, and holidays alongside local fitting and forecasting time. Weather covariates use retrospective observations.   TimesFM 3.0 with covariates achieves the lowest overall errors, reducing mean absolute error by approximately 43\% relative to weekly seasonal-naive forecasting. XGBoost remains competitive, although paired daily comparisons do not establish a clear mean-absolute-error advantage for TimesFM. T
    
[^148]: 网格原生标记化：将基础模型改造以适用于非结构化数据

    Mesh-Native Tokenisation: Retrofitting Foundation Models to Unstructured Data

    [https://arxiv.org/abs/2610.12618](https://arxiv.org/abs/2610.12618)

    本文提出网格原生卷积（MNC），一种局部感知的标记化方法，使基于卷积的物理基础模型能够高效处理非结构化网格数据，解决了传统卷积编码器无法适用于此类数据、而图神经网络方案训练昂贵且编码方式不兼容的问题。

    

    物理基础模型是构建时空系统通用仿真器的一种日益流行的方法。这类模型需要标记化层，将物理场数据提升为适合动态处理的潜在表示。该层的典型选择是卷积编码器，它以局部感知的方式提取场信息，并且训练效率很高。然而，这种选择无法应用于一大类科学计算问题，这类问题中的数据表示在非结构化网格上，例如在具有弯曲边界的计算域上进行的模拟。此类数据通常需要图神经网络架构，其训练成本更高，而且以目前使用的形式，其信息编码方式与其CNN对应方法不同，从而限制了物理仿真器在这两类数据模式之间的可移植性。在这项工作中，我们提出了网格原生卷积（MNC），一种局部感知的标记化方法……

    arXiv:2610.12618v1 Announce Type: cross  Abstract: Physics foundation models are an increasingly popular approach for building versatile emulators for spatiotemporal systems. They require tokenisation layers that lift physical field data into a latent representation suitable for dynamic processing. A typical choice for this layer is a convolutional encoder, which lifts field information in a locality-aware way and can be trained very efficiently. However, this choice cannot be applied to a large class of scientific computing problems where data are represented on unstructured meshes, for example simulations on domains with curved boundaries. Such data typically require graph neural network architectures, which are more expensive to train and, in the form currently used, encode information differently from their CNN counterparts, thus limiting the portability of physics emulators between these data regimes. In this work we introduce Mesh-Native Convolution (MNC), a locality-aware tokeni
    
[^149]: 语言模型中的自我保存

    Self-Preservation in Language Models

    [https://arxiv.org/abs/2610.12602](https://arxiv.org/abs/2610.12602)

    该论文提出将任务描述及示例组织成任务库，让语言模型自行选择保留目标并生成重放数据，从而在持续学习中更广泛地保护自身原有能力。

    

    持续学习旨在让预训练模型在适应新任务的同时保留原有能力。然而，用于保留的任务仅覆盖了模型所掌握知识的一部分，而遗忘的范围可能会超出这些任务。近期研究表明，语言模型可以在无法访问其原始训练数据的情况下，通过生成有用的重放样本（replay）来补充这些有限的证据。这引出了一个关于“自我保存”的问题：模型能否帮助选择保留目标，从而更广泛地保护自身的能力？为了研究这一问题，我们将任务描述及其示例组织成任务库，其条目可以被选择和组合，用以指导冻结模型的重放数据生成。使用不同规模的Qwen模型进行持续指令微调得出了三个发现。首先，新任务和学习率决定了哪些先前能力会发生退化，而跟踪之前回答正确的样本可以揭示平均准确率所掩盖的能力损失。其次，……

    arXiv:2610.12602v1 Announce Type: cross  Abstract: Continual learning (CL) seeks to adapt pretrained models while preserving prior capabilities. Yet the tasks supplied for preservation cover only part of what a model knows, while forgetting can extend beyond them. Recent work shows that language models can supplement this limited evidence by generating useful replay without access to their original training data. This raises a question about self-preservation: can a model help choose preservation targets that protect its own capabilities more broadly? To study this question, we organize task descriptions and their examples into Task Banks, whose entries can be selected and combined to guide replay generation by a frozen model. Continual instruction tuning with Qwen models of varying sizes yields three findings. First, the new task and learning rate shape which prior capabilities degrade, while tracking previously correct answers reveals losses that average accuracy can conceal. Second,
    
[^150]: 超越二元可验证奖励：面向软件工程智能体的行为自我奖励

    Beyond Binary Verifiable Rewards: Behavioral Self-Rewarding for SWE Agents

    [https://arxiv.org/abs/2610.12598](https://arxiv.org/abs/2610.12598)

    提出行为自我奖励（BSR）方法，让策略模型自我评判同一任务中相同验证结果轨迹间的调试行为差异，将其转化为有界辅助奖励，从而在二元可验证奖励之上引入更精细的监督信号，且无需单独的奖励模型或新的环境交互。

    

    基于可验证奖励的强化学习（RLVR）通过可执行测试为软件工程智能体提供可靠的监督信号，但二元奖励无法区分调试质量差异巨大的轨迹。我们研究了当前策略能否评判自身的轨迹，并利用这些评判来提升任务性能，而无需单独训练的奖励模型。为此，我们提出了行为自我奖励（BSR），这是一种奖励精炼方法，它复用策略模型来比较同一任务中具有相同验证器结果的多条轨迹之间的可观察调试行为。BSR将成对偏好聚合为任务局部分数，并将其转换为有界的辅助奖励，在保持验证器定义的通过/失败排序的同时引入奖励对比。这无需新的环境交互，即可从已收集的轨迹中获得额外的监督信号。

    arXiv:2610.12598v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) provides reliable supervision for software engineering agents through executable tests, but binary rewards fail to distinguish trajectories with substantially different debugging quality. We investigate whether the current policy can judge its own trajectories and use these judgments to improve task performance without a separately trained reward model. To this end, we introduce Behavioral Self-Rewarding (BSR), a reward-refinement method that reuses the policy to compare observable debugging behavior among trajectories from the same task with the same verifier outcome. BSR aggregates pairwise preferences into task-local scores and converts them into bounded auxiliary rewards, introducing reward contrast while preserving the verifier-defined pass/fail ordering. This provides additional supervision from already-collected trajectories without requiring new environment interactions. Exper
    
[^151]: SpireBench：利用《杀戮尖塔》评估智能体模型的长期战略博弈能力

    SpireBench: Evaluating Long-Horizon Strategic Play of Agentic Models with Slay the Spire

    [https://arxiv.org/abs/2610.12596](https://arxiv.org/abs/2610.12596)

    SpireBench让20个大语言模型完整游玩《杀戮尖塔》以评估长时程战略推理能力，发现前沿模型虽达到普通人类水平但远逊于专家，且随着游戏进程延长差距不断扩大，所有模型都误判长线风险，倾向保守保血而非牺牲血量换取成长。

    

    arXiv:2610.12596v1 公告类型：新论文 摘要：智能体大语言模型如今已能处理复杂任务，但针对长时程的战略推理研究仍然不足：大多数智能体基准测试短小且情节化，而且模型在长任务上的表现难以分解为存储知识、推理能力和运气。我们提出SpireBench，让模型完整地游玩商业卡牌构筑游戏《杀戮尖塔》，在该游戏中早期决策决定长期成败。在最难难度下对20个大语言模型进行测试后发现：前沿模型击败了启发式基线，达到普通人类玩家的平均水平，但仍远远落后于专家玩家。这一差距随着单局进程的延长而扩大：最强的模型在第一幕的胜率与专家相当，但随后逐渐落后。区分强弱模型的关键在于它们打好每一场战斗的能力；而所有模型在整个局中都会误判风险，倾向于保留生命值而非花费生命值来换取成长。对游戏中所有术语进行全面且自洽的重命名实验表明……（摘要至此截断）

    arXiv:2610.12596v1 Announce Type: new  Abstract: Agentic large language models now tackle complex tasks, yet strategic reasoning over long horizons is under-studied: most agentic benchmarks are short and episodic, and performance on long tasks is hard to dissect into stored knowledge, reasoning and luck. We introduce SpireBench, in which models play the complete commercial deck-builder Slay the Spire (StS), where early decisions decide long-term success. Across 20 large language models (LLMs) at the game's highest difficulty, frontier models beat heuristic baselines and match the average human player but remain far behind experts. This gap widens as the run lengthens: the strongest models win the first act as often as experts, then fall behind. What separates strong models from weak is how well they play each fight; all of them misjudge risk over a run, preserving health rather than spending it to grow stronger. A comprehensive, self-consistent renaming of every game term shows that pa
    
[^152]: 知你所不知：通过演化生成空间实现闭环学习的智能体数据合成

    Know What You Don't Know: Agentic Data Synthesis by Evolving Generation Space for Closed-Loop Learning

    [https://arxiv.org/abs/2610.12593](https://arxiv.org/abs/2610.12593)

    EvoSyn 提出一个由学习反馈驱动的多智能体数据合成框架，通过可修订的生成空间状态将智能体的学习弱点直接转化为合成修订与资源分配，弥合了反馈与决策之间的鸿沟，从而实现闭环学习。

    

    大语言模型的持续改进日益依赖于合成经验，然而，面向交互式智能体的有效数据合成不仅需要可执行、可验证的轨迹，还需要能为当前任务智能体提供学习价值的交互。现有方法越来越多地引入环境反馈、数据验证或学习者失败信息，但仍面临“从反馈到决策”的鸿沟：观察到的弱点并不能直接指明未来的合成应如何改变，也无法确定这些弱点能否被当前的合成空间所表示。为弥合这一鸿沟，我们提出了 EvoSyn，一个由学习反馈驱动的多智能体框架，它围绕一个可修订的状态来组织合成，该状态规定了哪些交互因素可以被刻意生成，以及合成精力应分配到哪里。一个角色专门化的多智能体系统将学习者的反馈转化为合成修订，同时一个设计者负责分配合成……

    arXiv:2610.12593v1 Announce Type: new  Abstract: The continued improvement of large language models increasingly depends on synthetic experience, yet effective data synthesis for interactive agents requires not only executable and verifiable trajectories, but also interactions that provide learning value for the current task agent. Existing methods increasingly incorporate environment feedback, data validation, or learner failures, but still face a feedback-to-decision gap: observed weaknesses do not directly specify how future synthesis should change, or whether they can be represented by the current synthesis space. To address this gap, we propose EvoSyn, a learning-feedback-driven multi-agent framework that organizes synthesis around a revisable state specifying what interaction factors can be deliberately generated and where synthesis effort should be allocated. A role-specialized multi-agent system translates learner feedback into synthesis revisions, while a Designer allocates sy
    
[^153]: AutoIndexer：通过在编辑链上训练因果模型实现灵活顺序解码

    AutoIndexer: Flexible-Order Decoding by Training Causal Models on Chains of Edits

    [https://arxiv.org/abs/2610.12545](https://arxiv.org/abs/2610.12545)

    AutoIndexer提出了一种模拟文本编辑器的“编辑链”训练与解码机制，使因果自回归模型能够通过插入、替换和删除操作灵活地事后修改已生成的输出，从而支持非因果顺序的灵活解码。

    

    自回归解码不太适合非因果顺序的数据，例如交叉引用或未来依赖关系，因为其输出无法事后修改，这与迭代式的人类推理不同。扩散式大语言模型部分缓解了这一限制，但仍无法灵活地在预先生成的上下文中插入或删除任意数量的标记。我们将AutoIndexer设计为一个具有新颖训练和解码机制的生成模型，该机制模拟“编辑链”，使模型能够通过标记的插入、替换和删除来灵活地事后修改其先前的输出。该模型学习预测用于开启和关闭编辑的标记token，同时将光标移动到现有token序列中的位置，类似于文本编辑器。训练序列是通过对真实标签施加逆链式编辑操作进行扰动而获得的，模型通过撤销这些逆操作来学习编辑。扰动过程……

    arXiv:2610.12545v1 Announce Type: cross  Abstract: Autoregressive decoding is less suited to non-causally ordered data, e.g., cross-references or future dependencies, since their outputs cannot be revised post hoc, unlike iterative human reasoning. Diffusion LLMs partially mitigate this limitation but still cannot flexibly insert or remove arbitrary numbers of tokens from a pre-generated context. We design AutoIndexer as a generative model with a novel training and decoding mechanism that simulates a Chain of Edits, allowing the model to flexibly modify its prior outputs in hindsight via token insertions, substitutions, and deletions. The model learns to predict marker tokens for opening and closing edits, simultaneously moving a cursor to positions in the existing token sequence, analogous to a text editor. Training sequences are obtained by perturbing ground-truth labels with inverse chained edit operations, and the model learns to edit by undoing those inverses. The perturbation tra
    
[^154]: 等预算蒙特卡洛树搜索与随机重启列表调度在DAG工作流上的对比：一项受控等预算计算研究

    Equal-Budget Monte Carlo Tree Search versus Randomized-Restart List Scheduling for DAG Workflows: A Controlled Equal-Budget Computational Study

    [https://arxiv.org/abs/2610.12542](https://arxiv.org/abs/2610.12542)

    在严格控制的等计算预算实验中，简单的随机重启列表调度方法在DAG工作流调度上比蒙特卡洛树搜索（MCTS）好2.88%，表明更复杂的搜索方法并不必然优于简单基线。

    

    蒙特卡洛树搜索（MCTS）常被提出用于调度有向无环图（DAG）工作流，但大多数研究从未进行过可能推翻该结论的对比实验——即给予一个简单得多的方法相同的搜索预算。我们在受控协议下进行了这项对比实验。所有方法分支获得完全相同的预算——构建和评估的候选调度数量相同：一个MCTS调度方法（预注册为主要分支）；一个不含搜索树的随机重启列表调度器（作为对照组）；以及一个教科书式UCT的MCTS分支（作为在同一领域上的注册后续实验）。分析计划在数据收集之前就已锁定；一名独立检查员确认了确认阶段的全部768个调度均有效且完成时间计算正确。在主预算为2048个被评估调度的条件下，MCTS比随机重启方法差2.88%（95%置信区间[-3.62%, -2.15%]；在96个保留实例中有72个相对于基线存在正向提升空间）。这一性能劣势并非源于预算不足（摘要在此处被截断）。

    arXiv:2610.12542v1 Announce Type: new  Abstract: Monte Carlo Tree Search (MCTS) is often proposed for scheduling directed-acyclic-graph (DAG) workflows, yet most studies never run the comparison that could refute it - giving a much simpler method the same search effort. We run that comparison under a controlled protocol. All arms receive an identical budget - the same number of candidate schedules built and evaluated: an MCTS scheduling method, preregistered as the main arm; a randomized-restart list scheduler with no search tree as the control; and a textbook-UCT MCTS arm as a registered follow-up on the same domain. The analysis plan was locked before data collection; an independent checker confirmed all 768 confirmation-stage schedules valid with correct finish times. At the main budget of 2048 evaluated schedules, MCTS is 2.88% worse than the restarts (95% CI [-3.62%,-2.15%]; the 72 of 96 held-out instances with positive baseline-relative headroom). The deficit is not budget starva
    
[^155]: EliteTable：面向可执行控制策略的自我改进程序搜索

    EliteTable: Self-Improving Program Search for Executable Control Policies

    [https://arxiv.org/abs/2610.12536](https://arxiv.org/abs/2610.12536)

    EliteTable 是一种由大语言模型引导、利用固定容量精英方案表来指导方案生成与改进的自我改进程序搜索算法，在六个 Gymnasium 控制任务上验证了其迭代提升可执行控制策略性能的能力。

    

    EliteTable 是一种由大语言模型（LLM）引导的自我改进程序搜索算法，它使用一个容量固定的高分方案表来引导新方案的生成、对现有方案的修改以及多个方案的混合重组。性能评估覆盖了六个 Gymnasium 控制任务，每个任务进行十代搜索，每代包含五十个候选槽位，并在十次独立初始化的搜索中分别评估。所有六个任务的候选得分中位数从第一代到末代均有所上升，其中五个任务的入选优胜者搜索得分中位数有所上升。在 CartPole-v1 中，入选优胜者在全部十次搜索的第一代就达到了搜索得分上限。六个任务中有四个的全部十次配对留出端点评估均有所改进。BipedalWalker-v3 和 CarRacing-v3 有所改进但未达到其报告目标。Acrobot-v1 的十次搜索中有六次在留出片段上表现下降，且配对变化的中位数为负。

    arXiv:2610.12536v1 Announce Type: new  Abstract: EliteTable is a self-improving program-search algorithm guided by large language models (LLMs) that uses a fixed-capacity table of high-scoring solutions to guide new proposals, edits of existing solutions, and remixes of several solutions. Performance was evaluated across six Gymnasium control tasks, each with ten generations and fifty candidate slots per generation. Each task was evaluated across ten independently initialized searches. Candidate-score medians rose between the first and final generations in all six tasks, and selected-winner search-score medians rose in five. In CartPole-v1, the selected winner reached the search-score ceiling in the first generation of all ten searches. All ten paired held-out endpoints improved in four of the six tasks. BipedalWalker-v3 and CarRacing-v3 improved without reaching their reporting targets. Six of ten Acrobot-v1 searches declined on held-out episodes, and the median paired change was nega
    
[^156]: CircuitTree：将电路组合成音节转写器

    CircuitTree: Composing Circuits into A Syllable Transcriber

    [https://arxiv.org/abs/2610.12520](https://arxiv.org/abs/2610.12520)

    该论文提出CircuitTree数据结构及贪心密度缩放组合算法，将39个从HuBERT中提取的音素分类电路高效组合成单个模型，实现了任何单个电路都不具备的音节转写能力，以64.8%的准确率转写开音节，仅比完全不共享电路的上限低1.2个百分点。

    

    电路发现的目标是将庞大的单一神经网络分解为微小的模块化子网络，这些子网络是可解释的，因为每个子网络执行一个可识别的子任务。我们开发了一种将任意数量的电路组合成单个模型的实用方法。我们提出了一种贪心的、密度缩放的组合算法，克服了以往通过简单并集来组合电路的尝试的局限性，然后通过CircuitTree将其推广到N个电路，CircuitTree是一种基于Θ(N log N)树组合调度构建的新数据结构。我们将从HuBERT中剪枝得到的用于分类单个音素的电路组合成一个音节分类器，这是39个音素电路中任何一个单独都不具备的能力。将CircuitTree串联成电路层次结构流水线，我们以64.8%的准确率转写开音节，与完全不共享任何电路的66.0%上限相差仅1.2个百分点，同时共享了99.5%的电路组件。

    arXiv:2610.12520v1 Announce Type: cross  Abstract: The goal of circuit discovery is to decompose a monolithic neural network into tiny modular subnetworks which are interpretable in the sense that each does an identifiable subtask. We develop a practical method for composing arbitrarily many circuits into a single model. We propose a greedy, density-scaled composition algorithm that overcomes limitations of previous attempts to compose circuits by naive union, then generalize it to $N$ circuits via CircuitTree, a new data structure built on a $\Theta(N\log N)$ tree-compose schedule. We compose circuits pruned from HuBERT for classifying individual phonemes into a syllable classifier, a capability none of the 39 phoneme circuits has individually. Chaining CircuitTrees into a hierarchy-of-circuits pipeline, we transcribe open syllables at 64.8\% accuracy, within 1.2 points of a 66.0\% ceiling that shares no circuitry at all, with 99.5\% of circuit components shared.
    
[^157]: 用于传感辅助波束预测的高能效脉冲神经网络

    Energy-Efficient Spiking Neural Networks for Sensing-Aided Beam Prediction

    [https://arxiv.org/abs/2610.12507](https://arxiv.org/abs/2610.12507)

    该论文为毫米波传感辅助波束预测开发了脉冲神经网络（SNN），发现在视觉和多模态输入下，SNN能在相近能耗下取得比人工神经网络更高的精度，或在同等精度下将能耗降低多达一个数量级。

    

    传感辅助波束预测可以利用视觉和位置等互补模态提供的环境信息，减少毫米波（mmWave）系统的波束训练开销。然而，由此形成的传感与推理流程可能非常耗能，这促使人们考虑使用高能效的脉冲神经网络（SNN）来替代传统的人工神经网络（ANN）。在这项工作中，我们开发了用于波束预测的SNN架构，并在仅位置、仅视觉和多模态三种输入条件下将其与ANN进行了系统性比较。结果表明，对于仅位置传感，紧凑型ANN始终优于SNN；而对于视觉和多模态传感，SNN在相近能耗下可获得比ANN更高的精度，且在相当精度下能耗可降低多达一个数量级。例如，多模态SNN达到了58.08%的top-1精度，优于最接近的……

    arXiv:2610.12507v1 Announce Type: cross  Abstract: Sensing-aided beam prediction can reduce the beam-training overhead of millimeter-wave (mmWave) systems by leveraging environmental information from complementary modalities such as vision and position. However, the resulting sensing and inference pipeline can be energy-intensive, motivating the use of energy-efficient spiking neural networks (SNNs) as an alternative to conventional artificial neural networks (ANNs). In this work, we develop SNN architectures for beam prediction and systematically compare them with ANNs using position-only, vision-only, and multimodal inputs. Our results show that compact ANNs consistently outperform SNNs for position-only sensing. In contrast, for vision and multimodal sensing, SNNs achieve higher accuracy than ANNs at similar energy costs and require up to an order of magnitude less energy at a comparable accuracy. For example, the multimodal SNN achieves 58.08% top-1 accuracy, outperforming the clos
    
[^158]: Kairos：迈向适用于所有模态的字节级基础模型

    Kairos: Toward a Byte-Level Foundation Model for Every Modality

    [https://arxiv.org/abs/2610.12495](https://arxiv.org/abs/2610.12495)

    Kairos 提出了一种在统一字节空间上处理文本、图像、音频、视频、激光雷达等多模态的小型基础模型架构，融合门控 DeltaNet 线性注意力、滑动窗口注意力、注意力-残差聚合、稀疏 MoE 和多尺度字节级编解码器，并通过单机上的小规模过拟合收敛测试验证该组合能否有效优化。

    

    在单台本地机器上训练的小型语言模型几乎没有留下架构试错的空间。Kairos 结合了基于共享投影的双向门控 DeltaNet 线性注意力与滑动窗口注意力（LiZAttention2）、分块的注意力-残差聚合、稀疏混合专家（MoE）前馈层，以及多尺度字节级编解码器，在文本、图像、音频、视频、激光雷达和控制流共享的统一字节空间上，以均匀噪声去噪目标进行训练；本报告在追问这种组合是否有用之前，先追问它究竟能否正常优化。我们给出了每个组件由代码推导出的算法，包括一个跨会话的循环状态记忆门控，随后进行收敛性健全性检查：在 1–15M 参数规模下对单个 440 字节的句子进行过拟合测试（仅文本），其中对编解码器和注意力选择使用三个随机种子，并对另外七个因素进行单次运行的参数扫描。所测试的两个注意力变体在……

    arXiv:2610.12495v1 Announce Type: cross  Abstract: Small language models trained on a single local machine leave little room for architectural trial and error. Kairos combines bidirectional gated-DeltaNet linear attention and sliding-window attention over shared projections (LiZAttention2), block-wise Attention-Residual aggregation, a sparse Mixture-of-Experts (MoE) feed-forward layer and a multi-scale byte-level codec, trained with a uniform-noise denoising objective over one byte space shared by text, image, audio, video, lidar and control streams; this report asks whether the combination optimizes at all before asking whether it is useful. We give code-derived algorithms for each component, including a cross-session recurrent-state memory gate, then run convergence sanity checks: overfittinga single 440-byte sentence at 1--15M parameters, text only, with three seeds for codec and attention choices and single-run sweeps for seven further factors. The two attention variants tested at 
    
[^159]: 无锚点的忠实性：关于可解释AI特征重要性忠实性度量结构性错位的立场论文

    Faithfulness Without Anchors: A Position Paper on the Structural Misalignments of Explainable AI Faithfulness Metrics for Feature Importance

    [https://arxiv.org/abs/2610.12493](https://arxiv.org/abs/2610.12493)

    本文是一篇立场论文，指出当前面向特征重要性的可解释AI忠实性度量存在结构性错位，并强调XAI实际上更多扮演调试工具的角色而非建立信任的手段。

    

    可解释人工智能（XAI）工具通常被呈现为一种通过使机器学习模型算法的某些方面更易于人类理解来提升对模型信任的方法。然而，过去十年积累的证据表明，这一目标总体上未能实现，反而促进了对错误预测的过度依赖，进而可能导致算法厌恶（algorithmic aversion）。实际上，XAI似乎主要作为一种调试工具发挥作用，有助于获得关于模型所学到的特征等方面的经验性洞察。造成这一局限性的主要问题之一在于，XAI方法常常过度简化模型的内部工作机制，导致解释与模型本身之间出现错位——这种行为被称为不忠实性（unfaithfulness）或不忠（infidelity）。忠实性可以被理解为解释与模型之间的近似程度。以严格的方式计算忠实性……

    arXiv:2610.12493v1 Announce Type: new  Abstract: Explainable Artificial Intelligence (XAI) tools have often been presented as methods to increase trust in machine learning models by making aspects of their algorithms more human-understandable. However, evidence accumulated over the last decade points to an overall failure of this objective, instead promoting overreliance on incorrect predictions, which can subsequently lead to algorithmic aversion. Rather, XAI appears to function primarily as a debugging tool, useful for obtaining anecdotal insights into, for example, features learned by the model. One of the main issues underlying this limitation is that XAI methods often oversimplify the inner workings of the model, resulting in misalignment between the explanation and the model itself - a behavior referred to as unfaithfulness or infidelity. Faithfulness can be interpreted as the degree of approximation between an explanation and the model. Computing faithfulness in a rigorous manne
    
[^160]: 基于人工智能的星载海上目标检测：在Versal嵌入式硬件上实现地球观测载荷数据缩减

    AI-Based On-Board Maritime Object Detection for Earth Observation Payload Data Reduction on Versal Embedded Hardware

    [https://arxiv.org/abs/2610.12182](https://arxiv.org/abs/2610.12182)

    该论文将量化后的YOLOX-S船只检测器部署于Versal VC1902嵌入式硬件上，实现星载智能筛选下传内容，从而根据数据内容有效缩减高分辨率对地观测卫星的载荷数据量。

    

    超高分辨率地球观测卫星所获取的数据量超出了其存储和下传能力，而在海上监视应用中，船只仅占每幅场景的极小部分。本文研究星载船只检测技术，用于筛选下传数据的内容，从而根据数据内容而非对每个像素进行编码的方式实现数据缩减；该方法与传统的星载压缩技术互补。本工作沿三个方向展开：(i) 数据与算法：基于68幅带标注的Maxar场景生成了一个受控数据集，涵盖43类船只，并在其上训练了YOLOX-S检测器。(ii) 嵌入式部署：对检测器进行量化并部署在Versal VC1902的DPU上，检测质量损失有限，每幅场景的处理时间为数秒。(iii) 数据缩减：我们提出了多种下传模式，从仅下传元数据（每个检测目标的边界框、类别和置信度分数），到下传船只周边图像裁剪区域、包含检测结果的图块，或完整的……（摘要截断）

    arXiv:2610.12182v1 Announce Type: cross  Abstract: Very-high-resolution Earth-observation satellites acquire more data than they can store and downlink, while in maritime surveillance the vessels cover a tiny fraction of each scene. We study onboard vessel detection as a way to select what is downlinked, which reduces the data according to its content rather than coding every pixel; it is complementary to conventional onboard compression. The work follows three axes. (i) Data and algorithm: a controlled dataset is generated from 68 annotated Maxar scenes with 43 vessel classes, and a YOLOX-S detector is trained on it. (ii) Embedded deployment: the detector is quantized and deployed on the DPU of a Versal VC1902, with a limited loss of detection quality and a processing time of a few seconds per scene. (iii) Data reduction: we propose several downlink modes, from metadata only (box, class and score of each detection) to image crops around vessels, tiles holding detections, or the whole 
    
[^161]: InterviewPlayground：一个用于评估AI访谈员的模拟环境

    InterviewPlayground: A Simulation Environment for Evaluating AI Interviewers

    [https://arxiv.org/abs/2610.12023](https://arxiv.org/abs/2610.12023)

    提出了InterviewPlayground，一个基于社会理论模拟参与者行为、通过多轮交互生成标准化访谈成绩单来评估AI访谈员的模拟环境，并验证了模拟评估结果能够预测AI访谈员与真实人类参与者互动时的实际表现。

    

    AI访谈员正被越来越多地开发出来，用于在市场研究、民意调查、偏好引导和社会科学研究等应用中获取开放式回答。然而，评估AI访谈员十分困难，因为它们在长时间的多轮交互中运行，必须根据参与者的行为做出适应。为了满足这一需求，我们开发了InterviewPlayground，一个用于评估AI访谈员的模拟环境，其中模拟研究对象的行为建立在社会理论基础之上。InterviewPlayground中的模拟研究会生成一份InterviewReportCard（访谈成绩单），通过一系列经过验证的测量指标来评估AI访谈员的表现。为了检验这种基于模拟的评估能否预测AI访谈员在与真实人类参与者互动时的表现，我们开展了15项真实的定性研究，涉及五个AI访谈员、三个访谈主题和450名人类参与者，并将其与InterviewPlayground中的模拟研究结果进行了比较。

    arXiv:2610.12023v1 Announce Type: new  Abstract: Increasingly, AI interviewers are being developed to elicit open-ended responses in applications like market research, public polling, preference elicitation, and social science research. However, evaluating AI interviewers is challenging because they function in extended, multi-turn interactions where they must adapt to participant behaviors. To address this need, we develop InterviewPlayground, a simulation environment for evaluating AI interviewers using simulated study participants whose behaviors are grounded in social theory. Simulated studies in InterviewPlayground produce an InterviewReportCard, which assesses the performance of AI interviewers using a suite of validated measures. To test whether our simulation-based evaluations predict performance with human participants, we conduct 15 real qualitative studies with five AI interviewers, three interview topics, and 450 human participants and compare them to simulated studies in I
    
[^162]: MultiWorldBench：独立控制的视角能否描述同一个共享世界？

    MultiWorldBench: Do Independently Controlled Views Describe One Shared World?

    [https://arxiv.org/abs/2610.11723](https://arxiv.org/abs/2610.11723)

    该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。

    

    多人世界模型必须确保独立控制的各个视角与一个共享且持久的世界保持一致。我们提出了MultiWorldBench，这是一个诊断性Minecraft基准测试，包含495个案例配置，涵盖七个任务套件和十种能力，包括独立控制、跨视角运动、共享状态同步、持久性、结构推理、并发交互和延迟重访。我们评估了Solaris、Gamma-World和MineWorld，并使用Engine GT作为参考。在生成式系统中，Gamma-World在十项能力上的平均分最高，为21.39，其次是Solaris为20.88，MineWorld为1.89，而Engine GT达到91.69。Gamma-World在多项控制、共享状态和重访能力上表现更佳，而Solaris在跨视角运动和竞态条件一致性方面领先。然而，所有生成式系统在状态持久性方面得分最高仅为8.00，在结构方面得分为1.33。

    arXiv:2610.11723v1 Announce Type: new  Abstract: Multiplayer world models must ensure that independently controlled views remain consistent with one shared and persistent world. We introduce MultiWorldBench, a diagnostic Minecraft benchmark containing 495 case configurations across seven task suites and ten capabilities, including independent control, cross-view motion, shared-state synchronization, persistence, structural reasoning, concurrent interaction, and delayed revisit. We evaluate Solaris, Gamma-World, and MineWorld, using Engine GT as a reference. Gamma-World achieves the highest ten-capability average among the generated systems at 21.39, followed by Solaris at 20.88 and MineWorld at 1.89, while Engine GT reaches 91.69. Gamma-World performs better on several control, shared-state, and revisit capabilities, whereas Solaris leads in cross-view motion and race-condition consistency. Nevertheless, all generated systems score at most 8.00 on state persistence and 1.33 on structur
    
[^163]: Sera：面向可靠且可解释电池健康预测的语义表示聚合

    Sera: Semantic Representation Aggregation for Reliable and Interpretable Battery Health Forecasting

    [https://arxiv.org/abs/2610.11567](https://arxiv.org/abs/2610.11567)

    提出语义表示聚合框架 Sera，通过融合基于规则的知识与大语言模型解读所构建的退化语义表示来补充时序建模，从而实现更可靠、更可解释的电池健康状态预测。

    

    电池健康状态预测对电池管理至关重要，但由于电池退化的非线性以及不同电池之间的异质性，该任务仍然充满挑战。现有的数据驱动方法主要依靠时序模型从数值型电池时间序列中学习，而更高层次的退化特征往往没有被显式地表示出来。然而，这些特征能够提供退化方面的指导，从而支持可靠的预测，并使退化的影响更具可解释性。在本文中，我们提出了 Sera，一个语义表示聚合框架，通过退化语义来补充时序建模。在电池领域专业知识的指导下，Sera 从时间序列中提取退化语义，并利用基于规则的知识和基于大语言模型（LLM）的解读构建两种互补的表示，这些表示随后被独立编码（摘要在此处被截断）。

    arXiv:2610.11567v1 Announce Type: cross  Abstract: Battery state of health (SoH) forecasting is important for battery management, but remains challenging due to nonlinear degradation and heterogeneity across batteries. Existing data-driven approaches primarily use temporal models to learn from numerical battery time series, and higher-level degradation characteristics are often not explicitly represented. These characteristics, however, can provide degradation guidance to support reliable forecasting and make the influence of degradation more interpretable. In this paper, we propose \textsc{Sera}, a \underline{se}mantic \underline{r}epresentation \underline{a}ggregation framework that complements temporal modelling with degradation semantics. Guided by battery domain expertise, \textsc{Sera} extracts degradation semantics from time series and constructs two complementary representations using rule-based knowledge and LLM-based interpretation. The representations are independently encod
    
[^164]: 面向多智能体LLM谈判的宪法门控与确定性恢复：针对有状态对抗性守门人的消融实验

    Constitutional Gating and Deterministic Recovery for Multi-Agent LLM Negotiation: Ablations Against a Stateful Adversarial Gatekeeper

    [https://arxiv.org/abs/2610.11542](https://arxiv.org/abs/2610.11542)

    该论文提出由5支柱运行时宪法、4层集群和认知退火组成的三部分控制栈，并通过针对有状态对抗性守门人的消融实验，验证其能消除多智能体LLM谈判中的三类模型调用浪费并实现确定性死锁恢复。

    

    多智能体LLM系统在与有状态对手谈判时会以三种方式浪费模型调用：永远无法满足对手隐藏接受条件的“礼貌循环”、触发重试的格式错误输出，以及对手要求智能体必须拒绝的事项时所陷入的合规死锁。我们研究了一个由三部分组成的控制栈——5支柱运行时宪法、4层集群（导演、三智能体多数投票、监视器、模式硬门）以及认知退火（确定性死锁检测、对智能体侧上下文的原子清除、规范化恢复消息）——并在一个已发布的对抗性守门人上进行测试，该守门人的接受规则为固定的正则表达式，其LLM仅用于渲染回复文本。该测试环境具有已知解：它衡量的是该控制栈能否在对抗集群漂移的同时执行符合宪法的策略并从死锁中恢复，而非衡量它是否有所发现。每个配置运行五次（共30次运行；……

    arXiv:2610.11542v1 Announce Type: cross  Abstract: Multi-agent LLM systems negotiating with a stateful counterpart waste model calls in three ways: polite loops that never meet the counterpart's hidden acceptance condition, malformed outputs that trigger retries, and compliance deadlocks in which the counterpart demands something the agent must refuse. We study a three-part control stack - a 5-Pillar runtime constitution, a 4-tier swarm (Director, three-agent majority vote, Monitor, schema hard gate) and Cognitive Annealing (deterministic deadlock detection, atomic purge of the agent-side context, a canonical recovery message) - against a released adversarial Gatekeeper whose acceptance rules are fixed regular expressions and whose LLM only renders reply text. The testbed has a known solution: it measures whether the stack executes a constitution-aligned strategy against swarm drift and recovers from deadlock, not whether it discovers anything. In five runs per configuration (30 runs; 
    
[^165]: 面向认知的真实世界社交场景中从原因到后果的情绪追踪

    Cognition-Oriented Emotion Tracing from Causes to Consequences in Real-World Social Scenes

    [https://arxiv.org/abs/2610.11410](https://arxiv.org/abs/2610.11410)

    该论文提出面向认知的TRACE框架，通过条件-情感-效应三阶段建模，并构建包含3,746个问答对和646个视频的TRACE-Bench基准，用于评估多模态模型在真实社交场景中从原因到后果的全链路情绪理解能力，发现人机之间存在显著的性能差距。

    

    情感计算已经从类别化的情绪识别发展到基于大型多模态模型的开放式情感分析。然而，情感科学将情绪描述为一个由评价、调节和社会解读所塑造的动态展开过程，这一过程在计算上仍未得到充分探索。我们提出了TRACE，这是一个面向认知的框架，通过三个相互关联的阶段——条件、情感和效应——来形式化情感事件，将可观察的线索与内部立场、情绪表现调节等认知因素相结合。基于这一表述，TRACE-Bench在真实世界社交场景中通过五项任务评估多模态模型，涵盖基于情境的情感识别、调节解码、原因推理、效应推理和全链路重构，包含基于646个视频的3,746个结构化问答对。匹配的人机对比实验揭示了显著的性能差距，而情感……

    arXiv:2610.11410v1 Announce Type: new  Abstract: Affective computing has progressed from categorical emotion recognition to open-ended affective analysis with large multimodal models. Yet affective science describes emotion as an unfolding process shaped by appraisal, regulation, and social interpretation, which remains underexplored computationally. We propose TRACE, a cognition-oriented framework that formalizes an affective episode through three interrelated stages: Condition, Affect, and Effect, integrating observable cues with cognitive factors such as internal stance and regulation of emotional display. Based on this formulation, TRACE-Bench evaluates multimodal models in real-world social scenes through five tasks spanning grounded affect recognition, regulation decoding, cause reasoning, effect reasoning, and full-chain reconstruction, with 3,746 structured question-answer pairs over 646 videos. A matched human-model comparison reveals a substantial performance gap, while affec
    
[^166]: 利用势函数估计自回归语言模型下的广义期望

    Estimating great expectations under autoregressive language models with potentials

    [https://arxiv.org/abs/2610.11399](https://arxiv.org/abs/2610.11399)

    本文提出利用采样时免费获得的下一词元条件概率构造势函数，对语言模型下检验泛函的期望进行估计，在计算成本相近的情况下显著降低了估计方差。

    

    语言模型的许多应用所依赖的并非单个样本，而是模型下某个检验泛函的期望。可靠地估计这类期望在计算上可能非常昂贵。在本文中，我们展示了如何利用采样过程中作为副产品自然获得的下一词元条件概率来使估计更加高效。我们通过势函数来实现这一点：势函数是定义在前缀上的实值函数，能够对检验泛函进行加性分解。我们构造了一个估计器，其方差取决于所选择的势函数，并推导出势函数能够降低该方差的充分条件。随后，我们为若干估计对象和应用开发了实用的势函数，并在计算成本相当的情况下，在多个估计对象上展示了显著的方差缩减效果。

    arXiv:2610.11399v1 Announce Type: new  Abstract: Many applications of language models hinge not on individual samples but on the expectation of a test functional under the model. Estimating such expectations reliably can be computationally expensive. In this paper, we show how to make estimation more efficient by exploiting the next-token conditional probabilities which are available as a by-product of sampling. We do so through potentials: real-valued functions on prefixes that decompose the test functional additively. We construct an estimator whose variance depends on the chosen potential, and derive conditions under which a potential reduces this variance. We then develop practical potentials for several estimands and applications, and demonstrate substantial variance reductions across several estimands at comparable computational cost.
    
[^167]: TypedBench：系统一决策模型中校准、措辞敏感性与成本的基准测试

    TypedBench: A Benchmark for Calibration, Framing Sensitivity, and Cost in System One Decision Models

    [https://arxiv.org/abs/2610.11392](https://arxiv.org/abs/2610.11392)

    该论文提出了TypedBench基准测试，通过七种策略标注生成器和九个评估套件，以改写中位数范围和有限样本噪声底限为参照，系统评估系统一类型化决策模型的校准质量、措辞敏感性与决策成本。

    

    系统一（System One）模型通过非生成式接口输出针对类型化答案（如分类选择、有序等级或二元结果）的校准概率。软件可以通过阈值、成本加权选择和升级规则对这些概率进行操作。因此，如果这些概率校准不良或对措辞敏感，软件可能会采取非预期的行动，而人类操作员却没有文本依据可供审查。目前的评估大多在公开分类数据集上报告准确率和校准情况，且缺乏明确的参照标准。我们提出了TypedBench，这是一个针对类型化决策模型（如Jev）的基准测试，由七个策略标注的生成器和九个评估套件构成。我们将准确率报告为各种改写表述下的中位数和范围，并将校准误差相对于匹配的完美校准预测器的有限样本噪声底限进行衡量。我们通过选择性预测、有序概……（原文摘要在此处截断）

    arXiv:2610.11392v1 Announce Type: new  Abstract: System One models output calibrated probabilities over typed answers such as categorical choices, ordinal levels, or binary outcomes, via a non-generative interface. Software can act on these probabilities through thresholds, cost-weighted choices, and escalation rules. Consequently, if these probabilities are miscalibrated or wording-sensitive, the software ma take unintended actions leaving human operators with no textual rationale to inspect. Current evaluations largely report accuracy and calibration on public classification datasets without a clear reference. We present TypedBench, a benchmark for typed decision models like Jev built from seven policy-labelled generators and nine evaluation suites. We report accuracy as median and range across paraphrases, and calibration error relative to the finite-sample noise floor of a matched, perfectly calibrated predictor. We assess probability quality through selective prediction, ordinal p
    
[^168]: 重新思考CLIP后训练中的对比损失：一种基于冻结文本编码器的互补框架

    Rethinking Contrastive Loss in CLIP Post-training: A Complementary Framework with Frozen Text Encoder

    [https://arxiv.org/abs/2610.11374](https://arxiv.org/abs/2610.11374)

    论文发现CLIP后训练中的灾难性遗忘主要源于对比温度τ取值不当而非负样本不足，并据此提出冻结文本编码器的轻量级单轮后训练框架ComCLIP，证明在足够小的温度下对比后训练能提升而非损害预训练CLIP的性能。

    

    CLIP作为基础性的视觉-语言模型，是诸如LLaVA等下游视觉-语言模型（VLM）事实上的视觉编码器。后训练为改进CLIP提供了一条轻量级路径，但近期研究认为标准对比损失不适用于后训练，因为小批量训练下会产生灾难性遗忘，这促使一些设计放弃对比目标而转向蒸馏方法。我们重新审视了这一前提，发现对于InfoNCE目标而言，所报告的遗忘主要并非由负样本不足驱动，而是由对比温度τ的不当取值所致：当τ设置得足够小时，对比后训练会改善而非损害预训练的CLIP，我们通过InfoNCE梯度的温度依赖性对这一现象进行了解释。基于这一发现，我们提出了ComCLIP，这是一种轻量级的单轮（单epoch）后训练方案，它冻结了CLIP的文本编码器——从而使……

    arXiv:2610.11374v1 Announce Type: cross  Abstract: CLIP serves as a foundational vision-language model and the de facto vision encoder for downstream VLMs such as LLaVA. Post-training offers a lightweight route to refine CLIP, but recent work argues that the standard contrastive loss is unsuitable for post-training due to catastrophic forgetting under small batches, motivating designs that abandon the contrastive objective in favor of distillation. We revisit this premise and find that, for the InfoNCE objective, the reported forgetting is driven primarily not by insufficient negatives but by an inappropriate magnitude of the contrastive temperature $\tau$: with $\tau$ set sufficiently small, contrastive post-training improves rather than degrades the pretrained CLIP, which we explain through the temperature dependence of the InfoNCE gradient. Building on this finding, we propose \textbf{ComCLIP}, a lightweight single-epoch post-training recipe that freezes CLIP's text encoder---so the
    
[^169]: 高阶动作监督造就强大的策略类

    Higher-Order Action Supervision Makes A Strong Policy Class

    [https://arxiv.org/abs/2610.11175](https://arxiv.org/abs/2610.11175)

    本文提出同时监督零阶和一阶动作并辅以正式理论保证的损失方案，可显著提升模仿学习和强化学习策略的性能与控制鲁棒性，解决其在实际部署中的不稳定性问题。

    

    现代数据驱动的决策方法，如模仿学习（IL）和强化学习（RL），在解决许多复杂任务方面取得了巨大成功。然而，这些方法在机器人技术和自动驾驶等现实应用中常常面临严重的控制不稳定性和鲁棒性问题，给其实际部署带来了显著挑战。我们认为，这种不稳定性问题主要源于它们仅监督和优化零阶动作（即动作标签）的局限性，未能考虑高阶动作动态和时间一致性。在本文中，我们证明同时监督零阶和一阶动作可以显著提升策略的性能和控制鲁棒性。为实现这一目标，我们提出了一种新颖而优雅的损失方案，该方案有正式的理论保证支持，可以为任何现成的策略模型（配备这种能力）。

    arXiv:2610.11175v1 Announce Type: cross  Abstract: Modern data-driven decision-making methods, such as imitation learning (IL) and reinforcement learning (RL), have achieved great success in solving many complex tasks. However, these methods often suffer from serious control instability and robustness issues when applied in real-world applications such as robotics and autonomous driving, posing notable challenges for their practical deployment. We argue that this instability issue stems largely from their limitations in solely supervising and optimizing zeroth-order actions (i.e., the action labels), failing to account for higher-order action dynamics and temporal consistency. In this paper, we show that simultaneously supervising both zeroth- and first-order actions can dramatically enhance policies' performance and control robustness. To achieve this, we introduce a novel and elegant loss scheme supported by formal theoretical guarantees that can equip any off-the-shelf policy model 
    
[^170]: LinSlot：利用线性表示假设从基于槽的对象表示中进行无监督属性发现

    LinSlot: Exploiting Linear Representation hypothesis for unsupervised attribute discovery from slot based object representation

    [https://arxiv.org/abs/2610.10722](https://arxiv.org/abs/2610.10722)

    本文提出LinSlot方法，利用线性表示假设（LRH）构建连接图像、对象槽和属性块的概率模型，实现了无监督的属性发现与对象-属性解耦表示学习。

    

    本文研究了从原始、非结构化的图像数据中学习对象及其属性的解耦表示问题。基于槽的方法在从图像中无监督学习对象表示方面已取得显著成功。基于块槽注意力的方法通过假设对象表示可以被均匀分解为属性，将该框架扩展到属性表示，但这种假设可能是次优的，从而限制了所学习表示的质量。因此，我们研究了一个联合发现对象与属性表示的框架。我们的关键贡献是利用线性表示假设（LRH），该假设认为可组合的概念可以在槽表示中以线性可加子空间的形式表示。基于这一洞见，我们提出了一个连接图像、槽（对象）和块（属性）的概率模型。我们提出了一种架构

    arXiv:2610.10722v1 Announce Type: cross  Abstract: This paper studies the problem of learning disentangled representations of objects and their attributes from raw, unstructured image data. Slot-based methods have shown considerable success in unsupervised learning of object representations from images. Block-slot attention-based methods extend this framework to attribute representations by assuming a uniform factorization of object representations into attributes, which may be suboptimal and consequently limit the quality of the learned representations. We therefore investigate a framework for jointly discovering object and attribute representations. Our key contribution is leveraging the Linear Representation Hypothesis (LRH), which postulates that composable concepts can be represented as linearly additive subspaces in slot representations. Based on this insight, we propose a probabilistic model connecting images, slots (objects), and blocks (attributes). We present an architecture 
    
[^171]: PHRBench：大语言模型幻觉后推理的行为学评估

    PHRBench: A Behavioral Evaluation of Post-Hallucination Reasoning in LLMs

    [https://arxiv.org/abs/2610.10455](https://arxiv.org/abs/2610.10455)

    提出了PHRBench这一受控基准，通过幻觉顺从、幻觉规避和启发式纠正等行为指标，在四个领域对18个大语言模型处理幻觉前提的推理轨迹进行评估，发现模型成功从幻觉中恢复并得出正确答案的情况仍然相对罕见。

    

    幻觉信息可能在多阶段大语言模型系统中传播，并成为后续推理上下文的一部分。现有关于幻觉后推理（PHR）的研究主要刻画最终结果的变化和总体的推理动态，而对模型在响应层面如何处理幻觉前提的理解仍然不足。在本工作中，我们提出了PHRBench，这是一个受控基准，用于在四个领域、18个大语言模型上对行为结构化的幻觉后推理进行评估。PHRBench通过幻觉顺从、幻觉规避和启发式纠正三个指标，独立于最终答案正确性地刻画每条推理轨迹，并将“有洞察力的轨迹”定义为最终成功纠正并达到正确答案的轨迹。在4820个受控实例中，我们发现成功的恢复仍然相对罕见，并且与推理轨迹上更频繁的信念更新相关（摘要原文在此处被截断）。

    arXiv:2610.10455v1 Announce Type: new  Abstract: Hallucinated information can propagate through multi-stage LLM systems and become part of the context for subsequent reasoning. Existing studies of post-hallucination reasoning (PHR) mainly characterize changes in final outcomes and aggregate reasoning dynamics, leaving how models resolve hallucinated premises at the response level insufficiently understood. In this work, we introduce PHRBench, a controlled benchmark for behaviorally structured PHR across four domains and 18 large language models. PHRBench characterizes each reasoning trajectory independently of final-answer correctness through Hallucination Compliance, Hallucination Avoidance, and Heuristic Correction, and defines an insightful trajectory as successful correction that ultimately reaches the correct answer. Across 4820 controlled instances, we find that successful recovery remains relatively rare and is associated with more frequent belief updates along the reasoning tra
    
[^172]: 大型语言模型在被提示进行不真实回答时的推理Token激增现象

    Reasoning-Token Spikes Under Prompted Untruthful Responding in Large Language Models

    [https://arxiv.org/abs/2610.10405](https://arxiv.org/abs/2610.10405)

    该论文基于认知负荷理论，提出利用推理token数量这一无需访问思维链内容的低带宽信号，发现大语言模型在被提示进行不真实回答时会出现推理token数量的激增，从而为检测模型欺骗行为提供了新方法。

    

    监控推理型人工智能模型的思维链仍然是检测此类模型中欺骗行为及其他形式不良行为的关键方法。然而，基于语义的思维链监控依赖于推理轨迹清晰可读、对产生模型行为的底层计算足够忠实，更不用说其还必须是可访问的。此外，越来越多的证据表明，思维链输出可能很快变得难以解读或不忠实，甚至可能不再可访问。基于认知负荷理论，我们研究了一种低带宽信号——生成的推理token数量——它不需要访问推理轨迹的内容。三个具备推理能力的大型语言模型回答了210道选择题——涵盖分析性、描述性和规范性推理类型以及道德与非道德领域——在系统提示指示它们进行不真实回答的条件下（原文在此处截断）。

    arXiv:2610.10405v1 Announce Type: cross  Abstract: Monitoring the chain-of-thought of reasoning artificial intelligence (AI) models remains a key approach to detecting deception and other forms of misbehavior in such models. However, semantic chain-of-thought monitoring depends on reasoning traces being legible and sufficiently faithful to the underlying computations that produced the model's behavior, not to mention accessible. Moreover, there is increasing evidence that chain-of-thought outputs may soon become illegible or unfaithful, if they even remain accessible. Based on cognitive load theory, we investigate a lower-bandwidth signal -- the number of reasoning tokens generated -- which does not require access to the content of the reasoning trace. Three reasoning-capable large language models answered 210 multiple-choice questions -- across analytic, descriptive, and normative reasoning types as well as moral and non-moral domains -- under system prompts instructing them to respon
    
[^173]: 解耦逻辑与人设：边缘大语言模型智能体对上下文污染的结构性免疫

    Decoupling Logic from Persona: Structural Immunity of Edge LLM Agents to Context Pollution

    [https://arxiv.org/abs/2610.09772](https://arxiv.org/abs/2610.09772)

    该论文提出AO-DA解耦架构，将边缘LLM智能体的逻辑推理与人设表达分离为同一INT4基础模型上两条可热插拔LoRA适配器的独立推理路径，使逻辑部分对上下文污染实现结构性免疫。

    

    运行在边缘设备上的小型语言模型智能体必须在同一个上下文窗口内同时维持人设并进行正确推理，而该窗口中充满了对话历史和人设指令。我们研究了当这些历史记录冗长、具有误导性且人设内容占比过高（即人设-逻辑干扰）时，此类智能体的逻辑推理部分会发生什么，并提出了一种解耦架构（AO-DA），在单个INT4量化的基础模型上，通过可热插拔的LoRA适配器，将逻辑推理（“做什么/What”）与人设表达（“怎么做/How”）分离为两条独立的推理路径。逻辑路径仅接收核心对话轮次，并输出可验证的结构化状态；人设路径则结合完整的对话历史，以符合角色设定的方式对该状态进行渲染。在同一基础模型上的消融实验中（在Apple M2笔记本电脑上进行，使用Llama-3.1-8B-Instruct和Gemma-3-4B-it，4比特量化；共480次运行，涵盖4个污染等级 × 3个实验组 × 2个任务 × 2个人设 × 5个随机种子），我们发现：（i）解耦后的逻辑路径对污染具有结构性不变性：其……（原文摘要在此处截断）

    arXiv:2610.09772v1 Announce Type: new  Abstract: Small language-model agents on edge devices must hold a persona and reason correctly at once, inside one context window that fills with conversational history and persona instructions. We study what happens to the logical part of such an agent when that history is long, misleading and persona-heavy (persona-logic interference), and present a Decoupling Architecture (AO-DA) that separates logical inference ("What") from persona expression ("How") into two inference paths on one INT4 base model with hot-swappable LoRA adapters. The logic path receives only the core turn and emits a verifiable structured state (Micro-State); the persona path renders it in character with the full history. In same-base-model ablations on an Apple M2 laptop (Llama-3.1-8B-Instruct and Gemma-3-4B-it, 4-bit; 480 runs over 4 pollution levels x 3 arms x 2 tasks x 2 personas x 5 seeds) we find: (i) the decoupled logic path is structurally invariant to pollution: its
    
[^174]: 从专家引导的证明搜索到自动化开放问题求解

    From Expert-Guided Proof Search to Automated Open-Problem Solving

    [https://arxiv.org/abs/2610.09769](https://arxiv.org/abs/2610.09769)

    研究者开发的多智能体开源系统Bolzano无需针对具体问题的人类指导，在约3,800个开放问题中自动解决了约200个，其中包括经原作者确认的STOC 2026论文中提出的4个问题。

    

    大型语言模型对数学研究的贡献日益增多，而数学研究的进展往往依赖于高效的证明搜索、渐进式的改进以及细致的验证。我们介绍了Bolzano，一个多智能体开源系统，它使用并行的证明器智能体配合验证器智能体协同工作，并维护人类可读的研究状态。在专家挑选的问题上进行的初期人工使用产生了8项成果，其证明均经领域专家核查。受这些案例研究的启发，我们在从四组论文中提取的约3,800个开放问题上运行了Bolzano，无需针对具体问题的人类指导，便解决了其中约200个开放问题。其中一项实验使用了被理论计算机科学顶级会议STOC 2026接收的论文，我们回答了这些论文中提出的四个问题，并得到了论文作者的确认。

    arXiv:2610.09769v1 Announce Type: cross  Abstract: Large language models are increasingly contributing to mathematical research, where progress often depends on efficient proof search, incremental improvements and careful verification. We describe Bolzano, a multi-agent open-source system that uses parallel prover agents with a verifier agent and maintains a human-readable research state. Initial manual use on expert-selected problems yielded 8 results whose proofs were checked by domain experts. Motivated by these case studies, we ran Bolzano without problem-specific human guidance on about 3,800 open problems extracted from four sets of papers, solving about 200 open problems. One experiment used papers accepted to STOC 2026, a top conference in theoretical computer science. There, we answered four questions raised in the papers, as confirmed by their authors.
    
[^175]: 多智能体博弈中学习上报不安全任务

    Learning to Report Unsafe Tasks in a Multi-Agent Game

    [https://arxiv.org/abs/2610.09002](https://arxiv.org/abs/2610.09002)

    该论文揭示了多智能体共享策略下“沉默陷阱”的成因——共享策略的梯度会把沉默的收益重复计入 k 次，使团队局部收敛于沉默——并给出了一个审计惩罚条件，使策略梯度更新能够可靠地收敛到普遍上报不安全任务（在多数情形下既充分又必要）。

    

    当一组智能体共享完成任务所得的奖励时，上报并停止不安全的工作会减少团队的总奖励。惩罚沉默的随机审计可以使“普遍上报”成为最优策略，却无法保证一个保持沉默的团队能够学会上报。我们在一个任何目击者都可以通过上报来停止任务的博弈中研究这一问题。当一项任务的 k 个目击者共享同一策略并独立抽取动作时，在普遍沉默点处，期望奖励对共享沉默概率的导数会将沉默的收益重复计算 k 次，而“普遍上报与沉默”之间的比较只将其计算一次。因此，即使上报是最优的，完全共享的策略也可能在局部被吸引到沉默状态。对于任何按共享策略的分组，我们给出了一个审计条件：它对精确的策略梯度更新达到普遍上报是充分的，并且在非边界情形下，从接近普遍沉默的状态出发也是必要的。在一个平衡家族中，满足……（摘要原文在此处截断）

    arXiv:2610.09002v2 Announce Type: replace  Abstract: When a team of agents shares a reward for completed tasks, reporting and stopping unsafe work reduces the team's reward. Random audits that penalize silence can make universal reporting optimal without ensuring that a silent team learns to report. We study this problem in a game where any witness can stop a task by reporting. When a task's k witnesses share a policy and draw actions independently, the derivative of expected reward with respect to the shared silence probability at universal silence counts its benefit k times, while the comparison between universal reporting and silence counts it once. A fully shared policy can therefore be locally attracted to silence even when reporting is optimal. For any grouping into shared policies, we give an audit condition sufficient for exact policy-gradient updates to reach universal reporting and, outside boundary cases, necessary from near universal silence. In a balanced family, meeting t
    
[^176]: 趁它们沉睡时工作：利用评估延迟实现全贝叶斯优化

    Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization

    [https://arxiv.org/abs/2610.08969](https://arxiv.org/abs/2610.08969)

    该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。

    

    黑盒优化问题在科学与工程领域无处不在，通常涉及代价高昂的目标函数。这种目标评估延迟在优化过程中带来两个后果：（i）目标函数评估主导了整个执行时间；（ii）样本高效的算法对于加速开发、避免资源浪费至关重要。贝叶斯优化（BO）方法是规划器为推荐下一个尝试点的事实上的标准选择。标准的BO采用点估计来拟合代理模型的超参数；而完全贝叶斯方法则通过模型平均来考虑超参数的不确定性，从而获得更好的不确定性估计——这在BO中普遍存在的低数据场景下非常有用。然而，该方法往往计算代价过高，因此很少被使用。在这项工作中，我们提出ELF-BO，一种利用目标函数评估延迟来提前启动全贝叶斯代理模型计算的算法。

    arXiv:2610.08969v1 Announce Type: new  Abstract: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of th
    
[^177]: OTel：开放电信AI数据集、基准测试与模型

    OTel: Open Telco AI Datasets, Benchmarks, and Models

    [https://arxiv.org/abs/2610.07766](https://arxiv.org/abs/2610.07766)

    OTel发布了一个统一的开放电信AI资源，提供面向检索、重排序、指令微调及安全/拒答的电信数据集，以及采用开放训练方案后训练的30个基线模型（10个嵌入模型、3个重排序器、17个语言模型）。

    

    我们提出了Open Telco（OTel），这是一个开放的电信AI资源，发布了用于检索、重排序、指令微调以及安全/拒答的衍生电信数据集，同时提供了30个全参数后训练的基线模型，涵盖10个嵌入模型、3个重排序器和17个语言模型。社区已经大量使用该资源：截至2026年5月3日，发布的模型已被下载超过1600万次，该项目在全球范围内获得了157篇以上的媒体报道。在先前的开放电信数据集和基准测试的基础上，OTel在一个统一资源中提供了有文档记录的电信数据源、留出的评估分区、经过训练的嵌入模型、重排序器、基于上下文的大语言模型以及安全/拒答数据。每个基线模型都从开源权重模型出发，采用开放的训练方案在OTel衍生数据上进行后训练，然后在留出的OTel评估分区上进行评估。OTel后训练改进（摘要在此处截断）

    arXiv:2610.07766v1 Announce Type: new  Abstract: We present Open Telco (OTel), an open telecom AI resource that releases derived telecom datasets for retrieval, reranking, instruction tuning, and safety/abstention, together with 30 full-parameter post-trained baselines spanning 10 embedding models, 3 rerankers, and 17 language models. The community has already engaged substantially with the resource: as of May 3, 2026, the released models have been downloaded over 16 million times and the project has received 157+ pieces of media coverage worldwide. Building on prior open telecom datasets and benchmarks, OTel provides documented telecom data sources, held-out evaluation partitions, trained embedding models, rerankers, context-grounded LLMs, and safety/abstention data in one unified resource. Each baseline starts from an open-weight model and is post-trained on OTel-derived data using an open training recipe, then evaluated on held-out OTel evaluation partitions. OTel post-training impr
    
[^178]: TRIAGE：面向原生NVFP4强化学习的方向感知失配稳定化方法

    TRIAGE: Direction-Aware Mismatch Stabilization of Native NVFP4 Reinforcement Learning

    [https://arxiv.org/abs/2610.07043](https://arxiv.org/abs/2610.07043)

    提出TRIAGE方法，通过方向感知的片段级诊断选择性地重新平衡策略梯度更新，并对严重失配进行有界修复，从而稳定原生NVFP4低精度强化学习的策略优化训练。

    

    低精度执行可以大幅加速大语言模型的强化学习（RL），但学习器与采样器执行之间的差异（失配）可能会破坏策略优化的稳定性。在本文中，我们刻画了失配与策略梯度方向之间的相互作用，区分了局部放大型与收缩型的更新贡献，而仅凭失配幅度无法识别这些差异。在原生NVFP4运行中，我们观察到两个放大区域之间的早期不平衡，偏向于负优势、负差距的更新。在失配全局扩散之前，这些尾部分词会集中在一小部分响应片段中。基于这些发现，我们提出了TRIAGE，一种方向感知的稳定化方法，它利用片段级诊断来选择性地重新平衡策略梯度更新，并对残余的严重失配应用有界修复。TRIAGE在修改优化目标的同时……

    arXiv:2610.07043v1 Announce Type: cross  Abstract: Low-precision execution can substantially accelerate reinforcement learning (RL) for large language models, but discrepancies between learner and sampler execution can destabilize policy optimization. In this paper, we characterize the interaction between mismatch and the policy-gradient direction, distinguishing locally amplifying from contracting update contributions that mismatch magnitude alone cannot identify. In native NVFP4 runs, we observe an early imbalance between the two amplifying regions, favoring negative-advantage, negative-gap updates. Their tail tokens become concentrated in a small fraction of response segments before mismatch spreads globally. Motivated by these findings, we introduce TRIAGE, a direction-aware stabilization method that uses segment-level diagnosis to selectively rebalance policy-gradient updates and applies bounded repair to residual severe mismatch. TRIAGE modifies the optimization objective while r
    
[^179]: 从基准测试到实验台：智能体能否在真实世界的药物研发中存活？

    From Benchmark to Bench: Can Agents Survive Real-World Drug Discovery?

    [https://arxiv.org/abs/2610.06411](https://arxiv.org/abs/2610.06411)

    该研究开发了开源模块化药物研发智能体MAGI，并在三家制药公司的九项真实先导化合物优化项目中验证，发现无论由大语言模型直接生成分子还是委托REINVENT 4生成，项目成败的关键在于预测模型的优劣，而非分子生成途径的选择。

    

    智能体系统越来越多地被用于协调分子设计工具，但尚不清楚技术栈中的哪一层限制了真实项目上的结果。我们开发了MAGI，一个开源的模块化智能体，它能够撰写优化目标、启动并监控优化过程、解读构效关系，并据此调整自身策略。MAGI既可以直接通过大语言模型生成分子，也可以将任务委托给REINVENT 4，同时评分服务在统一接口约定下可互换。我们在来自三家制药公司的九项回顾性先导化合物优化项目上对其进行了测试，这些项目在固定的时间截点下重放。两种生成都产生了有效的分子结构：大语言模型提出的分子更贴近局部化学空间，并以更少的操作步骤达到相当或更高的初级活性，而REINVENT则探索了更广阔的化学空间。一个优化项目能否达成其目标取决于预测模型的质量，而非分子生成途径：达成度……（原文截断）

    arXiv:2610.06411v2 Announce Type: replace  Abstract: Agentic systems increasingly coordinate molecular-design tools, but it is unclear which layer of the stack limits outcomes on real projects. We developed MAGI, an open modular agent that authors objectives, launches and monitors optimization, interprets structure--activity relationships, and revises its strategy accordingly. MAGI generates molecules either directly through the LLM or by delegating to REINVENT 4, with scoring services interchangeable behind a common contract. We tested it across nine retrospective lead-optimization campaigns from three pharmaceutical companies, replayed under fixed temporal cutoffs. Both routes produced valid structures: LLM proposals stayed closer to local chemistry and reached comparable or higher primary activity in fewer operations, whereas REINVENT explored broader chemical space. Whether a campaign met its objective depended on the predictive models, not on the generation route: attainment follo
    
[^180]: GraphDecisionBench：图任务上系统一模型的基准测试

    GraphDecisionBench: Benchmarking System One Models on Graph Tasks

    [https://arxiv.org/abs/2610.06354](https://arxiv.org/abs/2610.06354)

    提出与模型无关的图决策基准GraphDecisionBench（GraphDecide），通过结构化任务画像、图-文本输入对照与启发式提案控制，系统评估Jev等系统一模型在图任务上的表现，揭示出邻接识别准确不等于整体结构正确、联合图-文本输入未必提升预测、可行解构造也未必带来高质量解。

    

    大语言模型（LLM）在图理解与决策方面的应用正受到越来越多的探索，而以Jev为代表的系统一（System One）模型则会直接从给定选项中进行选择。然而，系统一模型在图相关任务上的能力仍不明确。我们提出了GraphDecide，一个与模型无关的基准，它结合了结构化任务画像、匹配的图-文本输入对照以及启发式提案控制，用于诊断图决策性能。我们在语言模型基线之外，对Jev及相关基于选择的模型进行了评估，涵盖十四种模型-接口配置。Jev的结果凸显了该基准的核心区分点：准确的邻接识别并不能保证更广泛的结构正确性；图与文本的联合输入并不总能提升预测效果；构造出可行解也不意味着能获得高质量的解。其任务契约、候选接口与评分规则支持可比较的分析（原文在此处截断）。

    arXiv:2610.06354v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly explored for graph understanding and decision-making, while System One models such as Jev select directly from supplied options. However, the capabilities of System One models on graph-related tasks remain unclear. We introduce GraphDecide, a model-independent benchmark that combines structural task profiles, matched graph-text input contrasts and heuristic-proposal controls to diagnose graph decision performance. We evaluate Jev and related choice-based models alongside language-model baselines, covering fourteen model-interface configurations. Jev's results illustrate the benchmark's central distinctions: accurate adjacency recognition does not guarantee broader structural correctness, joint graph-text input does not consistently improve prediction, and feasible construction does not establish high solution quality. Its task contracts, candidate interfaces and scoring rules support comp
    
[^181]: DREAM：面向多模态多智能体辩论的动态分辨率分配方法

    DREAM: Dynamic Resolution Assignment For Multimodal Multi-agent Debate

    [https://arxiv.org/abs/2610.05615](https://arxiv.org/abs/2610.05615)

    提出DREAM框架，通过零样本探测为每个智能体动态分配不同的图像分辨率，并利用ANLL不确定性量化和自适应阈值来缓解多模态多智能体辩论中的群体思维问题，从而提升多模态推理能力。

    

    多智能体辩论（MAD）已成为提升大语言模型（LLM）推理能力的有效范式，并且正日益被扩展到多模态场景中。然而，现有多模态MAD框架通常让所有智能体接触相同的固定视觉输入，忽视了不同样本和不同智能体所需视觉尺度的巨大差异。此外，这些框架经常受到“群体思维”（groupthink）现象的困扰，即智能体过早放弃正确的推论，转而迎合自信但产生幻觉的同伴回答。为解决这些瓶颈，我们提出了DREAM（面向多模态多智能体辩论的动态分辨率分配），该方法通过两个核心组件运作：（1）动态分辨率分配——一个零样本探测回合，智能体在其中测试多种分辨率，使用平均归一化对数似然（ANLL）量化不确定性，并利用自适应阈值将每个智能体分配到其…

    arXiv:2610.05615v2 Announce Type: replace  Abstract: Multi-agent debate (MAD) has emerged as an effective paradigm to improve the reasoning capabilities of large language models (LLMs) and is increasingly being extended to multimodal settings. However, existing multimodal MAD frameworks typically expose agents to the same fixed visual input, ignoring substantial variation in the visual scale needed across samples and agents. In addition, these frameworks frequently suffer from groupthink, a phenomenon where agents prematurely abandon correct deductions to conform with confident but hallucinated peer responses. To address these bottlenecks, we introduce DREAM (Dynamic Resolution Assignment For Multimodal Multi-Agent Debate), which operates via two core components: (1) Dynamic Resolution Assignment, a zero-shot probe round where agents test multiple resolutions, quantify uncertainty using Average Normalized Log-Likelihood (ANLL), and use an adaptive threshold to assign each agent to its 
    
[^182]: 格林路由神经算子：物理决定网络读取的位置

    Green-Routed Neural Operators: Physics Determines Where the Network Reads

    [https://arxiv.org/abs/2610.05337](https://arxiv.org/abs/2610.05337)

    提出格林路由神经算子（GRNO），利用控制方程决定潜在特征的采样位置，在五个PDE系统中的四个上取得最低的自回归预测误差。

    

    我们发现许多偏微分方程（PDE）中输运场的物理角色与其在神经算子中的常见用法之间存在不匹配：PDE利用输运场来选择读取坐标，而神经算子通常仅将其视为输入值。我们提出格林路由神经算子（GRNO）来解决这一不匹配问题，该方法利用控制方程来确定潜在特征的采样位置。一个无参数的方程适配器评估诊断关系并构建偏移图，其数值即为读取坐标。一个多尺度编码器-解码器结合潜在特征的中心读取与路由读取，以学习完整的有限时间更新。在五个二维和三维PDE系统中，GRNO在40步自回归评估下于其中四个系统上取得了最低的平均最终相对L²误差，并在Keller-Segel系统上保持竞争力。固定权重的路由干预实验表明，模型对读取方向和空间对齐具有强烈的依赖性。

    arXiv:2610.05337v2 Announce Type: replace-cross  Abstract: We identify a mismatch between the physical role of transport fields in many PDEs and their usual role in neural operators: PDEs use them to select read coordinates, whereas neural operators typically treat them only as input values. We address this mismatch with the Green-Routed Neural Operator (GRNO), which uses the governing equation to determine where latent features are sampled. A parameter-free equation adapter evaluates the diagnostic relation and constructs a departure map whose values are the read coordinates. A multiscale encoder-decoder combines centered and routed reads of latent features to learn the complete finite-time update. Across five two- and three-dimensional PDE systems, GRNO achieves the lowest mean final relative $L^2$ error on four under 40-step autoregressive evaluation and remains competitive on Keller-Segel. Fixed-weight route interventions reveal strong dependence on direction and spatial alignment 
    
[^183]: Video2World：基于具身视频的交互式世界建模编码智能体基准测试

    Video2World: Benchmarking Coding Agents for Interactive World Modeling from Embodied Videos

    [https://arxiv.org/abs/2610.04432](https://arxiv.org/abs/2610.04432)

    该论文提出了 Video2World 基准，将“从具身视频端到端自动构建交互式模拟环境”形式化为软件工程任务，利用 222 个来自机器人和人类演示视频的重建实例，从几何保真度、动态保真度和功能正确性三方面评估前沿基础模型与编码智能体的自主世界建模能力。

    

    从真实世界观测中构建交互式模拟器是扩展具身数据的一条有前景的途径，但当前的流水线仍然严重依赖人工环境构建与校准。我们研究了前沿基础模型和编码智能体能否端到端地自动化这一过程。我们将“自主视频到模拟”表述为一项软件工程任务：智能体观察一段具身视频，构建相应的模拟环境与机器人行为，并通过执行反馈迭代地优化结果。为评估这一能力，我们提出了 Video2World，这是一个包含 222 个重建实例的基准，这些实例源自 189 段机器人与人类演示视频。Video2World 从几何保真度、动态保真度和功能正确性三个维度衡量重建的世界，涵盖空间感知、物理推理和可执行交互。对 9 个前沿（模型）的评估……

    arXiv:2610.04432v3 Announce Type: replace-cross  Abstract: Building interactive simulators from real-world observations is a promising way to scale embodied data, but current pipelines still rely heavily on manual environment construction and calibration. We study whether frontier foundation models and coding agents can automate this process end to end. We formulate \emph{autonomous video-to-simulation} as a software engineering task in which an agent observes an embodied video, constructs the corresponding simulated environment and robot behavior, and iteratively refines the result through execution feedback. To evaluate this capability, we introduce \textbf{Video2World}, a benchmark comprising 222 reconstruction instances derived from 189 robot and human demonstration videos. Video2World measures reconstructed worlds along geometric fidelity, dynamic fidelity, and functional correctness, capturing spatial perception, physical reasoning, and executable interaction. Evaluating 9 fronti
    
[^184]: AgroGround：农业领域中的多粒度定位识别

    AgroGround: Multi-Granularity Grounded Recognition in Agriculture

    [https://arxiv.org/abs/2610.04425](https://arxiv.org/abs/2610.04425)

    提出大规模农业定位识别数据集 AgroGround，通过自动化流程将八个农业视觉问答数据集的标签转换为病害病斑与完整目标的区域标注（共794,850条指令样本），并利用健康图像作为负向监督，微调视觉-语言模型实现植物病害等目标的识别与定位。

    

    农业视觉模型通常仅在识别或定位二者之一上进行评估，但可靠的诊断需要同时识别图像中存在什么目标并定位其证据区域。农业视觉问答（VQA）数据集虽然带有丰富的语义标签，却很少将这些标签与图像区域关联起来，而以人工方式大规模添加此类标注成本高昂。我们提出了 AgroGround，一个面向农业定位识别的大规模数据集，即识别植物病害及其他农业目标并定位其在图像中的对应区域。一个自动化流程将八个农业视觉问答数据集的标签转换为病害病斑级和完整目标级的标注，共生成 794,850 条指令样本。健康图像为病害查询提供负向监督，教会模型在无病害时返回空预测。我们在已知目标的定位指令与需要……

    arXiv:2610.04425v2 Announce Type: replace-cross  Abstract: Agricultural visual models are typically evaluated for either recognition or localization, but reliable diagnosis requires identifying what is present and localizing the evidence. Agricultural visual question answering (VQA) datasets carry rich semantic labels but rarely link them to image regions, and adding such annotations by hand is costly at scale. We introduce AgroGround, a large-scale dataset for grounded agricultural recognition: identifying plant diseases and other agricultural targets and localizing their image regions. An automated pipeline converts the labels of eight agricultural VQA datasets into annotations for disease lesions and whole objects, producing 794,850 instruction examples. Healthy images provide negative supervision for disease queries, teaching the model to return empty predictions. We fine-tune a shared vision-language model on known-target grounding instructions combined with instructions requiring
    
[^185]: 使用基于Lean的Rust验证器求解VeriContest

    Solving VeriContest with a Lean-Backed Rust Verifier

    [https://arxiv.org/abs/2610.03994](https://arxiv.org/abs/2610.03994)

    该论文提出由Lean 4支持的Rust验证器Rust-Prover，将VeriContest基准中全部1007道题目的1325个定理完全证明，其中1259个定理在不到32小时内以中位3.2分钟、1.17美元的成本完成，大幅超越了此前最强模型13.95%的首试通过率。

    

    VeriContest是一个包含1007道Rust竞赛编程题目的基准测试，每道题目都附有Verus规范、一个通过评测的解答以及一个Verus证明。其作者报告称，证明生成是前沿模型的瓶颈：在给定规范和代码的情况下，最强的模型首次尝试仅能对13.95%的题目生成被接受的Verus证明。我们报告了使用Rust-Prover——一个由Lean 4支持的Rust验证器——来解决同样的证明生成任务。Verus规范和Rust代码被重新表述并翻译成Lean，每个规范都成为一个定理，由智能体证明这些定理，并以Lean的内核作为最终检查。全部1007道题目的1325个定理全部得到证明。其中1259个定理在Claude Opus 5.5上的一次运行（不到32小时）中完成证明，每个证明的中位耗时为3.2分钟、成本为1.17美元，且其中70%在首次迭代即告成功。重新表述的规范已与基准测试进行了核对（原文在此处截断）。

    arXiv:2610.03994v2 Announce Type: replace-cross  Abstract: VeriContest is a benchmark of 1007 competitive-programming problems in Rust, each with a Verus specification, a judge-accepted solution, and a Verus proof. Its authors report that proof generation is the bottleneck for frontier models: given the specification and the code, the best model produces an accepted Verus proof for 13.95% of the problems on the first attempt. We report on solving the same proof-generation task with Rust-Prover, a verifier for Rust backed by Lean 4. The Verus specification and the Rust code are restated and translated into Lean, each specification becomes a theorem, and agents prove the theorems with Lean's kernel as the final check. All 1325 theorems of all 1007 problems were proved. 1259 of them were proved in one run of under 32 hours on Claude Opus 5.5, at a median of 3.2 minutes and $1.17 per proof, and 70% of them on the first iteration. The restated specifications were checked against the benchma
    
[^186]: GOTT：基于可复用跨机器人本体原语的以物体为中心的灵巧操作

    GOTT: Object-centric Dexterous Manipulation with a Reusable Cross-Embodiment Primitive

    [https://arxiv.org/abs/2610.03861](https://arxiv.org/abs/2610.03861)

    GOTT提出了一种基于单一可复用跨本体接触获取原语的灵巧操作框架，能够将来自规划、外部模型或人类演示的操作意图可靠地转化为多指机器人的稳定接触与物体运动跟踪，且无需针对不同任务或机器人本体进行改动。

    

    基础模型和大规模人类数据为操作意图提供了丰富的来源，但将这种意图转化为多指机器人行为仍然困难。灵巧手目前仍缺乏一种能在不同任务和机器人本体之间可靠建立接触的可复用低层原语。我们提出了GOTT，这是一个围绕单一跨本体接触获取原语构建的“接近-获取-移动”框架。给定一个与机器人无关的物体轨迹和接近规范，GOTT首先将手移动到任务相关接触区域的附近；随后，共享的闭环原语从这个近似初始化状态建立稳定接触，并由姿态条件控制器跟踪期望的物体运动。接近规范可以来自具有前瞻性的规划、外部模型或人类演示，而原语和跟踪后端保持不变。仿真和真实世界实验表明，GOTT能够建立……（摘要原文在此处截断）

    arXiv:2610.03861v2 Announce Type: replace-cross  Abstract: Foundation models and large-scale human data provide rich sources of manipulation intent, but translating this intent into multi-fingered robot behavior remains difficult. Dexterous hands still lack a reusable low-level primitive that reliably establishes contact across tasks and embodiments. We propose GOTT, a reach-acquire-move framework built around a single cross-embodiment contact-acquisition primitive. Given a robot-agnostic object trajectory and a reach specification, GOTT first brings the hand near a task-relevant contact region. The shared closed-loop primitive then establishes stable contact from this approximate initialization, and a pose-conditioned controller tracks the desired object motion. Reach specifications may come from future-aware planning, external models, or human demonstrations, while the primitive and tracking backend remain unchanged. Simulation and real-world experiments show that GOTT is able to est
    
[^187]: 面向图像分类的混合量子-经典架构演化

    Evolving Hybrid Quantum-Classical Architectures for Image Classification

    [https://arxiv.org/abs/2610.03220](https://arxiv.org/abs/2610.03220)

    该论文将自动化量子电路发现的演化框架EXAQC扩展至图像分类任务，通过演化参数化量子电路作为中间处理模块，克服了人工设计量子电路难以适配特定任务的局限。

    

    混合量子-经典神经网络将参数化量子电路（PQC）与成熟的深度学习架构相结合，但其性能在很大程度上取决于量子电路架构的选择，而这一选择目前仍主要依赖人工完成。现有的大多数方法依赖于手工设计或固定的电路拟设，需要预先指定电路结构、门组合和量子比特连接方式，且无法保证这些设计适合特定任务。这一限制在图像分类任务中尤为突出，因为量子电路既要对经典网络提取的特征进行变换，又要保持足够紧凑以便于实际训练，而通用的、与任务无关的拟设很难同时满足这些要求。我们将EXAQC——一个用于自动化量子电路发现的演化框架——扩展到图像分类任务。EXAQC将参数化量子电路作为中间处理模块进行演化，同时保留经典……（摘要原文在此处截断）

    arXiv:2610.03220v1 Announce Type: cross  Abstract: Hybrid quantum classical neural networks integrate parameterized quantum circuits (PQCs) with established deep learning architectures, but their performance depends strongly on the choice of quantum circuit architecture, a choice that remains largely manual. Most existing approaches rely on hand-designed or fixed circuit ans\"atze, requiring circuit structure, gate composition, and qubit connectivity to be specified in advance with no guarantee that they suit the task. This limitation is especially acute in image classification, where quantum circuits must transform features extracted by classical networks while remaining compact enough for practical training, requirements that generic, task-agnostic ans\"atze are unlikely to satisfy simultaneously. We extend EXAQC, an evolutionary framework for automated quantum circuit discovery, to image classification. EXAQC evolves PQCs as intermediate processing modules while retaining classical 
    
[^188]: 想象未来，内化要旨：通过内化时空想象实现高效的VLA推理

    Imagine the Future, Internalize the Gist: Efficient VLA Reasoning via Internalized Spatiotemporal Imagination

    [https://arxiv.org/abs/2610.02626](https://arxiv.org/abs/2610.02626)

    提出IG-VLA框架，通过在视觉表征空间中直接想象任务相关的未来场景演变，并将其内化为紧凑的场景要旨标记，使VLA模型在推理时无需昂贵的显式未来推演即可获得未来推理带来的动作预测增益，实现高效的机器人操作。

    

    视觉-语言-动作（VLA）模型日益融入中间推理以提升机器人操作能力，然而现有方法主要针对观测到的状态进行推理，并未显式地预测未来场景的演变。而若在每次推理步骤中都将推理扩展到显式的未来推演，则会引入大量计算开销。我们提出IG-VLA，一个使模型能够想象未来并内化其要旨的VLA推理框架。我们的潜在时空推理学会直接在视觉表征空间中想象与任务相关的未来场景演变，从而引导动作预测，而无需代价高昂的像素级视频生成。为进一步降低推理开销，我们引入了场景要旨记忆，它将推理所得的场景-行为关联内化为一个紧凑的场景要旨标记，在保留未来推理优势的同时，在推理阶段绕过显式的未来想象。

    arXiv:2610.02626v2 Announce Type: cross  Abstract: Vision-language-action (VLA) models increasingly incorporate intermediate reasoning to improve robotic manipulation, yet existing approaches primarily reason about observed states without explicitly anticipating future scene evolution. Extending such reasoning to explicit future rollouts at every inference step, however, introduces substantial computational overhead. We propose IG-VLA, a VLA reasoning framework that enables models to imagine the future and internalize the gist. Our Latent Spatiotemporal Reasoning learns to imagine task-relevant future scene evolution directly in visual representation space, guiding action prediction without costly pixel-level video generation. To further reduce inference overhead, we introduce Scene Gist Memory, which internalizes reasoning-derived scene-behavior associations into a compact Scene Gist Token, preserving the benefits of future reasoning while bypassing explicit future imagination at infe
    
[^189]: 基于选择的结构化推理：迈向高效的多模态搜索智能体

    Selection-Based Structured Reasoning: Toward Efficient Multimodal Search Agents

    [https://arxiv.org/abs/2610.01892](https://arxiv.org/abs/2610.01892)

    提出基于选择的结构化推理框架SSR，将开放式推理生成转化为从预定义推理候选中的似然选择，并借助共享上下文KV缓存实现并行评分，从而大幅提升小模型多模态搜索智能体的推理效率。

    

    多模态智能体通常在每次执行动作前生成自由形式的推理。对于小模型而言，有限的模型容量会导致冗长的推理，这不仅对动作生成几乎提供不了有用的指导，还会带来巨大的推理开销。为应对这一挑战，我们提出了基于选择的结构化推理框架（Selection-based Structured Reasoning，SSR），该框架将推理重新表述为选择，而非开放式生成。SSR将反复出现的高层推理表示为预先指定的、可复用的自然语言候选。在每一轮中，模型基于当前上下文下的似然从这些推理候选中进行选择，无需任何辅助任务头。使用预先指定的推理轨迹使得并行评分成为可能，即通过教师强制预填充（teacher-forced prefilling）并借助共享的上下文KV缓存，在候选内部和候选之间并发计算token似然。我们在七个多模态搜索基准上使用2B和……的模型对SSR进行了评估。

    arXiv:2610.01892v1 Announce Type: cross  Abstract: Multimodal agents commonly generate free-form reasoning before each action. For small models, limited model capacity can result in lengthy reasoning that provides little useful guidance for action generation while incurring substantial inference cost. To address this challenge, we introduce Selection-based Structured Reasoning (SSR), a framework that reformulates reasoning as selection instead of open-ended generation. SSR represents recurring high-level reasoning as pre-specified, reusable natural-language candidates. At each turn, the model selects from these reasoning candidates based on their likelihoods given the current context, without requiring an auxiliary task head. Using pre-specified reasoning traces enables parallel scoring, where teacher-forced prefilling computes token likelihoods concurrently within and across candidates using a shared context KV cache. We evaluate SSR on seven multimodal search benchmarks using 2B and 
    
[^190]: 探索更多，推理更优：面向扩散语言模型的逐步风险敏感GRPO

    Exploring More, Reasoning Better: Stepwise Risk-Sensitive GRPO for Diffusion Language Models

    [https://arxiv.org/abs/2610.00661](https://arxiv.org/abs/2610.00661)

    提出StepRS-GRPO方法，通过在去噪状态间动态调整群组优势变换的风险系数，使扩散语言模型在数学推理任务上同时提升pass@1准确率、pass@k覆盖率和答案多样性。

    

    扩散大语言模型通过去噪一个序列或连续的多个块来生成文本，能够并行地揭示多个 token。基于可验证奖励的强化学习（RLVR）会在这些决策中重复使用终端反馈，即使决策所依赖的上下文已经发生变化。我们提出逐步风险敏感的 GRPO（StepRS-GRPO），它在不同去噪状态之间改变群组优势变换的风险系数，同时保留底层训练器。对于二元奖励，我们证明了这种变换恰好是对中心化结果优势进行的、依赖于提示和状态的重新缩放。基于能力的校准给出了系数的合适量级，而端点与插值消融实验则指导调度方案的选择。在多个 dLLM 骨干模型和数学推理基准上，StepRS-GRPO 相比中心化 GRPO 提升了 pass@1 准确率和 pass@k 覆盖率，同时增加了答案多样性。

    arXiv:2610.00661v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) generate text by denoising a sequence or successive blocks, allowing several tokens to be revealed in parallel. Reinforcement learning with verifiable rewards (RLVR) reuses terminal feedback across these decisions, even as their conditioning context changes. We propose stepwise risk-sensitive GRPO (StepRS-GRPO), which varies the risk coefficient of the group-advantage transformation across denoising states while retaining the underlying trainer. For binary rewards, we show that this transformation is exactly a prompt- and state-dependent rescaling of centered outcome advantages. A capability-based calibration suggests a coefficient scale, while endpoint and interpolation ablations guide schedule selection. Across multiple dLLM backbones and mathematical reasoning benchmarks, StepRS-GRPO improves both pass@1 accuracy and pass@k coverage over centered GRPO, while increasing answer diversity. In our
    
[^191]: 对谁有用？样本价值只有相对于学习器才有意义

    Useful to Whom? Sample Value Is Defined Only Relative to the Learner

    [https://arxiv.org/abs/2610.00221](https://arxiv.org/abs/2610.00221)

    该论文通过冻结样本子集、只改变学习器的受控实验证明，核心集选择中“易样本优先”与“几何覆盖”准则之间的交叉边界并非由数据固定，而是随学习器容量（如ResNet-18宽度、输入网格分辨率）而移动，说明样本的价值只有相对于特定学习器才有意义。

    

    arXiv:2610.00221v1 通告类型：cross 摘要：一个模型需要什么样的数据才能学习？核心集选择（coreset selection）使这个问题变得具体：在预算约束下，保留对训练最有用的样本。“易样本优先”与“几何覆盖”两类准则在不同的预算区间内各有胜负，二者之间由一个交叉边界分隔。我们追问：这个边界是由数据本身固定，还是会随目标学习器而改变？受控实验冻结所选子集，仅改变训练所用的学习器。在低分辨率ImageNet-100上，将ResNet-18的宽度加倍会使交叉点从每类57个样本移动到85个样本：学习器改变了同一样本的相对价值。更大范围的扫描揭示了输入网格与模型容量之间的交互作用：在保留相同图像信息的前提下扩大输入网格会使边界左移，且这种偏移随宽度增大而减弱。步长控制实验在不改变输入网格的情况下再现并逆转了网格效应；仅移除最后一个下采样……（摘要原文在此处截断）

    arXiv:2610.00221v1 Announce Type: cross  Abstract: What kind of data does a model need in order to learn? Coreset selection makes this question concrete: under a budget, keep the samples most useful for training. Easy-first and geometric coverage criteria can win in different budget regimes, separated by a crossover boundary. We ask whether this boundary is fixed by the data or changes with the target learner. Controlled experiments freeze the selected subsets and manipulate only the training learner. On low-resolution ImageNet-100, doubling ResNet-18's width moves the crossover from 57 to 85 samples per class: the learner changes the relative value of the same samples. A wider sweep reveals an interaction between input grid and capacity. Enlarging the grid while retaining the same image information shifts the boundary left, and this shift weakens as width increases. Stride controls reproduce and reverse the grid effect without changing the input grid; removing only the last downsampli
    
[^192]: 规模与选择：究竟是什么让自动执行框架演化对视觉界面机器人智能体奏效

    Scale and Selection: What Makes Automatic Harness Evolution Work for Visual-Interface Robot Agents

    [https://arxiv.org/abs/2609.39304](https://arxiv.org/abs/2609.39304)

    该研究表明，用另一个编码智能体自动优化机器人智能体的执行框架（提示词、工具与控制规则）时，每轮参与评估的训练 rollout 数量是成败关键——将其从 5 个扩大到 100 个，可通过提升每次采纳决策的信噪比，把留出集成功率从 47% 提高到 67%。

    

    当一个现成的编码智能体被直接用作机器人策略时——即通过截图观察基于浏览器的3D界面，并通过少量工具操纵虚拟目标夹爪来执行动作——智能体的执行框架（harness，即其提示词、工具和控制规则）在很大程度上决定了成败，而迄今为止该框架一直依靠人工编写。我们证明这一执行框架可以由另一个编码智能体（即优化智能体）自动改进，并报告了两项关于其为何有效的发现。首先，优化智能体每轮所看到的 rollout（轨迹运行）数量，决定了演化出的框架是否可信、能否泛化并持续稳定提升。单个 rollout 是一个带噪声的二元结果，因此每轮 rollout 数量很少时，某次修改可能仅凭运气就被采纳；扩大批次规模可以提升每次采纳决策的信噪比。在保持轮数不变的情况下，将训练集从 5 个 rollout 扩大到 100 个 rollout，留出集成功率从 47% 提升至 67%（原文摘要在此处截断）。

    arXiv:2609.39304v1 Announce Type: cross  Abstract: When an off-the-shelf coding agent is used directly as a robot policy, observing a browser-based 3D interface through screenshots and acting by posing a virtual target gripper through a few tools, the agent's harness, its prompts, tools, and control rules, largely determines success, and until now it has been written by hand. We show that this harness can be improved automatically by another coding agent, the optimizer agent, and report two findings about what makes it work. First, the number of rollouts the optimizer agent sees per round governs whether the evolved harness is trustworthy, generalizes, and improves steadily. A single rollout is a noisy binary outcome, so with few rollouts per round a revision can be promoted on luck; enlarging the batch raises the signal-to-noise ratio of every promotion decision. Holding rounds fixed and growing the training set from 5 to 100 rollouts, held-out success rises from 47% to 67%, while sma
    
[^193]: 让大语言模型说出它们的真实想法：测量与改进思维链-可解释性对齐

    Making LLMs Say What They Think: Measuring and Improving CoT-Interpretability Alignment

    [https://arxiv.org/abs/2609.38972](https://arxiv.org/abs/2609.38972)

    本文提出CoT-可解释性对齐（CIA）指标来测量LLM思维链与内部计算之间的一致性，发现现有模型对齐程度有限（44.8%–75.9%），并通过以任务准确率和参数化忠实度为奖励的后训练方法有效提升了对齐性。

    

    思维链（CoT）轨迹通常被用作观察大型语言模型（LLM）如何得出答案的代理。然而，越来越多的证据表明，模型的思维链往往无法反映其内部计算，并且可以在不影响最终答案的情况下被改变。在这项工作中，我们测量并改进了LLM思维链中描述的推理过程与其内部实际计算之间的对齐程度。我们提出了CoT-可解释性对齐（CIA）指标，用于衡量模型的思维链轨迹与可解释性工具检测到的内部推理策略之间的一致性。我们在三项任务（两跳问答、提示干预和整数乘法）上对三个LLM进行了CIA评估，发现LLM在所有任务上的对齐程度都较为有限（44.8%–75.9%）。随后，我们尝试通过后训练来改进CIA，将任务准确率和参数化忠实度信号设置为奖励。实验表明……

    arXiv:2609.38972v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) traces often serve as a proxy for how Large Language Models (LLMs) arrive at their answers. However, growing evidence shows that models' CoT often fails to reflect their internal computations and can be changed without affecting their final answers. In this work, we measure and improve the alignment between the reasoning described in an LLM's CoT and what it computes internally. We propose CoT-Interpretability Alignment (CIA), a metric that measures the agreement between a model's CoT traces and its internal reasoning strategies as detected by interpretability tools. We evaluate CIA on three tasks (two-hop question answering, hint intervention, and integer multiplication) across three LLMs, finding that LLMs exhibit limited alignment across all tasks (44.8-75.9%). We then experiment with improving CIA via post-training, setting both the task accuracy and parametric faithfulness signals as a reward. Experiments sh
    
[^194]: OpenCollab：一个具有可编程协作与可控运行时的多智能体编程框架

    OpenCollab: A Multi-Agent Coding Framework with Programmable Collaboration and Controllable Runtime

    [https://arxiv.org/abs/2609.38345](https://arxiv.org/abs/2609.38345)

    提出OpenCollab多智能体编程框架，通过统一组织设计、共享可控运行时和细粒度事件流追踪，并首次定义Adherence指标来量化协作组织结构的实际遵循程度，揭示任何单一配置变化都会使遵循度从47.2%大幅波动至97%以上。

    

    多智能体编程系统旨在通过协作来解决复杂的软件工程任务。然而，现有的评估通常假设所配置的组织结构会被忠实地遵循，而实际情况并非如此。这种行为差距，再加上底层系统组件的差异，使得观察到的性能提升难以进行明确的归因。为此，我们提出了OpenCollab，这是一个多智能体编程框架，为可编程协作与可控运行时提供了统一的基础设施。具体而言，OpenCollab统一了组织设计，在共享运行时上强制实施实验控制，并通过细粒度的事件流来跟踪执行过程。在此基础上，我们定义了Adherence（遵循度）指标，用于量化所声明的组织结构是否真正得以实现。我们的实验表明，智能体在不同配置下的协作方式差异很大：改变任何一个单一维度都会使Adherence发生变化，从47.2%到高达97%以上。

    arXiv:2609.38345v1 Announce Type: cross  Abstract: Multi-agent coding systems are designed to tackle complex software engineering tasks through collaboration. However, existing evaluations typically assume configured organizations are followed faithfully, whereas reality differs. This behavioral gap, combined with differences in underlying system components, prevents clear attribution of observed gains. To this end, we introduce OpenCollab, a multi-agent coding framework that provides a unified infrastructure for programmable collaboration and controllable runtime. Specifically, OpenCollab unifies organization design, enforces experimental control on a shared runtime, and tracks execution through fine-grained event streams. On this basis, we define Adherence to quantify whether the declared organization is actually realized. Our experiments reveal that agents collaborate very differently across configurations: changing any single dimension shifts Adherence, from 47.2% to as high as 97.
    
[^195]: EngiWorld：前沿智能体在专业工程环境中能交付什么？

    EngiWorld: What Can Frontier Agents Deliver in Professional Engineering Environments?

    [https://arxiv.org/abs/2609.37686](https://arxiv.org/abs/2609.37686)

    该论文提出了首个覆盖完整设计循环的工程智能体基准EngiWorld，涵盖1,301个专家任务、6个工程领域和26个专业软件平台，并引入以工件为中心的自动化评估方法来检验工程成果的几何有效性、物理可行性与规则合规性。

    

    自主智能体在通用计算机使用方面取得了快速进展，但对专业工业工程的可靠自动化仍然遥不可及，因为工程工作流程要求对几何和物理约束进行推理，并处理跨软件和设计阶段保留的依赖关系。我们提出了EngiWorld，这是第一个围绕完整设计循环构建的基准：包含1,301个由专家精心策划的任务，涵盖6个工程领域（CAD、CAE、CAM、BIM、EDA和3D可视化）和26个专业软件平台，同时提供GUI和CLI两种界面，以及从软件选择到开放式任务的6种任务类型。我们进一步引入了一种以工件为中心的评估方法，该方法建立在统一的领域验证器套件之上，能够以编程方式检查最终和中间工件的几何有效性、物理可行性和规则合规性，并通过规范达成度对定量设计任务进行连续评分。

    arXiv:2609.37686v1 Announce Type: new  Abstract: Autonomous agents have made rapid progress in general-purpose computer use, but reliable automation of professional industrial engineering remains out of reach, as engineering workflows demand reasoning over geometric and physical constraints and dependencies preserved across software and design stages. We present EngiWorld, the first benchmark structured around the complete design loop: 1,301 expert-curated tasks spanning 6 engineering domains (CAD, CAE, CAM, BIM, EDA, and 3D visualization) and 26 professional software platforms, with both GUI and CLI interfaces and 6 task types ranging from software-selection to open-ended tasks. We further introduce an artifact-centric evaluation methodology built on a unified domain-verifier suite, which programmatically checks the geometric validity, physical feasibility, and rule compliance of final and intermediate artifacts, and scores quantitative design tasks continuously by specification attai
    
[^196]: FACT：保真度感知的铰接孪生体构建

    FACT: Fidelity-Aware Construction of Articulated Twins

    [https://arxiv.org/abs/2609.37067](https://arxiv.org/abs/2609.37067)

    FACT提出了一种智能体框架，通过证据—诊断—修正循环渐进式构建铰接孪生体，同时提升其几何、接触和动力学三个层面的保真度。

    

    视觉上逼真的铰接资产在接触交互中仍可能失效，或表现出不准确的运动。我们提出了FACT（保真度感知的铰接孪生体构建），这是一个智能体框架，通过渐进式构建铰接孪生体来提升其几何、接触和动力学保真度。该智能体在共享的可编辑表示上驱动“证据—诊断—修正”循环，利用定量反馈来选择测量方式和模型编辑操作，同时由数值工具执行并验证更新。该框架首先通过特征规划、针对性测量和诊断性细化，从图像中重建可编辑的铰接几何；在此基础上，在保真度约束压缩之前，通过任务感知的局部重划分来修复碰撞代理；最后，从被动响应视频中构建响应模型，利用仿真残差来指导模型修正和受约束的物理参数拟合。

    arXiv:2609.37067v1 Announce Type: cross  Abstract: Visually plausible articulated assets may still fail during contact interactions or exhibit inaccurate motion. We present FACT (Fidelity-Aware Construction of Articulated Twins), an agentic framework that progressively constructs articulated twins to improve geometry, contact, and dynamic fidelity. The agent drives an evidence--diagnosis--revision loop on a shared editable representation, selecting measurements and model edits using quantitative feedback, while numerical tools execute and validate the updates. It reconstructs editable articulated geometry from images through feature planning, targeted measurements, and diagnostic refinement. On this reference, it repairs collision proxies through task-aware local repartitioning before fidelity-constrained compression. Finally, it constructs response models from passive-response videos, using simulation residuals to guide model revision and constrained physical parameter fitting. Experi
    
[^197]: MS-GLA：通过多时间分辨率解决表示瓶颈的多尺度门控线性注意力

    MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution

    [https://arxiv.org/abs/2609.35664](https://arxiv.org/abs/2609.35664)

    提出多尺度门控线性注意力（MS-GLA），通过将注意力头分配到多个时间分辨率并用可学习的输入依赖融合层动态重组各头组输出，解决GLA Transformer在单一时间分辨率下的表示瓶颈，在不增加每头状态大小的情况下扩展有效记忆容量。

    

    门控线性注意力（GLA）Transformer 通过数据依赖的门控机制推动了线性循环模型的发展，但面临一个核心局限：所有注意力头共享的固定容量记忆矩阵在单一时间分辨率下运行，每个词元被逐一单独处理，这迫使它们必须同时编码局部句法模式和长程语义结构，从而造成了仅靠门控机制无法解决的表示瓶颈。我们提出多尺度门控线性注意力（MS-GLA），通过将注意力头分布到多个时间分辨率上来解决这一问题。较粗分辨率的头组池化更长的词元跨度，自然地专门化于长程依赖关系；而较细的头组则保留对局部句法结构的敏感性。一个可学习的、依赖输入的融合层在每个时间步动态重组各头组的输出，从而在不增加每头状态大小的情况下扩展了有效记忆容量……

    arXiv:2609.35664v2 Announce Type: replace-cross  Abstract: Gated Linear Attention (GLA) Transformers advance linear recurrent models through data-dependent gating, but face a core limitation: the fixed-capacity memory matrices across all heads operate at a single temporal resolution, where each token is processed individually, forcing them to simultaneously encode local syntactic patterns and long-range semantic structure, creating a representational bottleneck that gating alone is insufficient to resolve. We introduce Multi-Scale Gated Linear Attention (MS-GLA), which addresses this by distributing attention heads across multiple temporal resolutions. Coarser resolutions pool longer token spans naturally specializing toward long-range dependencies, while finer head groups retain sensitivity to local syntactic structure. A learnable, input-dependent fusion layer dynamically recombines head group outputs at each timestep, expanding effective memory capacity without increasing per-head s
    
[^198]: 面向受限疫情政策规划的图世界模型

    Graph World Models for Constrained Epidemic Policy Planning

    [https://arxiv.org/abs/2609.35545](https://arxiv.org/abs/2609.35545)

    EpiMind是一个用于受限疫情政策规划的图世界模型框架，通过图因子化循环状态空间模型和图-时间ADMM实现跨区域协调干预，在保证共享资源预算可行性的同时将入院预测误差降低29%。

    

    疫情政策规划通常需要在地理区域之间进行协调，既要考虑人口流动带来的跨区域溢出效应，又要考虑如何利用有限的资源。现有方法要么缺乏针对耦合动力学的动作条件模型，要么无法保证每个时期的可行性。我们提出了EpiMind，一个用于跨区域受限疫情政策规划的图世界模型框架。图因子化的循环状态空间模型从区域潜在信念生成联合的策略条件推演，而图-时间ADMM优化区域干预措施，通过投影确保共享资源的可行性，并在学习到的模型下评估时间规范。相对于无图结构的动力学建模，EpiMind将入院人数RMSE降低了29%；在保证共享预算可行性的前提下，其规划结果与最佳可行恒定政策相差在1-5%以内；并且在三种资源预算下优于所有可部署的基线方法。

    arXiv:2609.35545v2 Announce Type: replace-cross  Abstract: Epidemic policy planning often requires coordination between geographical regions, taking into account mobility-driven spillovers and how to make use of limited resources. Existing methods either lack action-conditioned models of coupled dynamics or cannot guarantee per-period feasibility. We present EpiMind, a graph world model framework for constrained epidemic policy planning across regions. A graph-factored recurrent state-space model generates joint policy-conditioned rollouts from regional latent beliefs, while graph-temporal ADMM optimizes regional interventions, enforces shared-resource feasibility through projection, and evaluates temporal specifications under the learned model. EpiMind reduces admission RMSE by 29% relative to graph-free dynamics modeling, plans within 1-5% of the best feasible constant policy with guaranteed shared-budget feasibility, and outperforms all deployable baselines across three resource bud
    
[^199]: 超级算法设计智能体：从零开始演化可学习优化器

    Hyper Algorithm Design Agent: Evolving Learnable Optimizer from Zero

    [https://arxiv.org/abs/2609.35328](https://arxiv.org/abs/2609.35328)

    该论文提出将元黑盒优化（MetaBBO）的设计过程建模为编码任务，通过双智能体框架利用先进编码智能体的递归自我改进能力，从零开始自动演化可学习优化器，从而消除人工逐案定制设计所带来的主观性限制。

    

    元黑盒优化（MetaBBO）是近年来“AI for Optimization（人工智能助力优化）”趋势中的亮点之一。该范式的双层工作流利用元层面的可学习算法设计策略，以确保底层优化任务在性能和泛化能力上的提升。虽然MetaBBO有助于抬高所得优化系统的性能下限，但目前它需要针对不同的优化问题进行人工逐案手工定制，这不可避免地引入了固有的主观性，从而限制了性能上限以及实际应用中的可用性。在本文中，我们通过将MetaBBO的设计循环视为编码任务来解决这一问题，从而借助先进编码智能体的递归自我改进能力，将开放性引入MetaBBO。具体而言，我们提出了一个双智能体框架：i) 任务智能体持续迭代改进目标MetaBBO方法的代码库……

    arXiv:2609.35328v2 Announce Type: replace  Abstract: Meta-Black-Box Optimization (MetaBBO) is one of the highlights in the recent AI for Optimization trend. This paradigm's bi-level workflow leverages the learnable algorithm design policy at meta level to ensure the performance and generalization improvement on the low-level optimization task. While MetaBBO helps advance the performance lower bound of the resulted optimization system, it is currently handcrafted and customized case by case to adapt different optimization problems, which inevitably introduces inherent subjectivity and hence restricts the performance upper bound and usability in practice. In this paper, we address this issue by regarding MetaBBO's design loop as coding task, where we could introduce openendedness into MetaBBO with recursive self-improvement capability of advanced coding agents. Specifically, we propose a dual-agent framework: i) a task agent continuously refines the codebase of a target MetaBBO approach 
    
[^200]: ThinkNet：用于高效运动想象脑电分类的进化式深度集成框架

    ThinkNet: Evolutionary Deep Ensembles for Efficient Motor-Imagery EEG Classification

    [https://arxiv.org/abs/2609.33967](https://arxiv.org/abs/2609.33967)

    ThinkNet提出了一种验证控制的框架，通过仅训练数据归一化、验证引导的进化搜索和验证门控推理，在严格防止测试受试者信息泄露的前提下，实现面向新用户的高效、紧凑的运动想象脑电分类。

    

    实用的辅助与康复类脑机接口需要受试者无关的运动想象脑电（MI-EEG）解码器，能够在目标用户数据有限且计算资源受限的条件下泛化到新用户。然而，当测试受试者的信息影响了预处理、模型选择或集成选择时，留出受试者的性能可能被高估。我们提出了ThinkNet，这是一个验证控制的框架，结合仅基于训练数据的归一化、验证引导的进化搜索和验证门控推理，为留出受试者识别紧凑的解码器与推理策略。我们在BCI Competition IV-2a（T会话）四分类解码任务上进行评估，采用九个留一受试者交叉验证折、三个随机种子、七个固定解码器条目，并对十个代表性解码器家族进行更广泛的搜索；留出受试者从不被用于归一化、超参数、架构或集成策略的选择。

    arXiv:2609.33967v2 Announce Type: replace-cross  Abstract: Practical assistive and rehabilitative brain--computer interfaces require subject-independent motor-imagery EEG (MI-EEG) decoders that generalize to new users under limited target-user data and constrained compute. However, held-out-subject performance can be overstated when test-subject information influences preprocessing, model selection, or ensemble selection. We present \textit{ThinkNet}, a validation-controlled framework that combines train-only normalization, validation-guided evolutionary search, and validation-gated inference to identify compact decoders and inference policies for held-out subjects. We evaluate four-class BCI Competition IV-2a (session T) decoding with nine Leave-One-Subject-Out (LOSO) folds, three seeds, seven fixed decoder entries, and a broader search over ten representative decoder families; the held-out subject is never used for normalization, hyperparameter, architecture, or ensemble-policy selec
    
[^201]: 面向生成式搜索的声明门控式来源风险审计

    Claim-Gated Source-Risk Auditing for Generative Search

    [https://arxiv.org/abs/2609.29145](https://arxiv.org/abs/2609.29145)

    该论文提出针对生成式搜索的声明门控来源风险审计契约：只有当关系证据、答案采纳、重要性和披露全部被观测到时才判定遗漏已解决，否则保持未解决状态，并通过可执行的参考检查器在穷举合成测试中验证了该契约的完备性。

    

    生成式搜索的答案可能引用了有依据的段落，却遗漏了会改变其解释方式的来源关系。我们规定了一种针对“查询-来源-答案”三元组的声明门控审计方法。只有当关系证据、答案采纳、重要性和披露全部被观测到时，遗漏才被判定为“已解决”；证据不完整时保持“未解决”状态，而不是被默认当作独立性处理。该规范将此判定终点与引用支持和审查优先级区分开来，并将决策绑定到带版本号的证据片段上。一个参考检查器使记录契约变得可执行。在一个穷举式的合成测试套件上，该检查器复现了全部81种三态谓词组合，并拒绝了192条故意构造的畸形记录。公共防护基线和谓词消融实验将终点逻辑与缺失证据的处理区分开来，而受控转换则检验了支持分离与证据移除。这些是有限的契约符合性结果，而非检测器性能……（原文摘要截断）

    arXiv:2609.29145v1 Announce Type: new  Abstract: A generative search answer can cite a supported passage yet omit a source relationship that changes its interpretation. We specify a claim-gated audit of the query-source-answer tuple. An omission is resolved only when relationship evidence, answer adoption, materiality, and disclosure are all observed; incomplete evidence remains unresolved rather than being treated as independence. The specification separates this endpoint from citation support and review priority, and binds decisions to versioned evidence spans. A reference checker makes the record contract executable. On an exhaustive synthetic suite, it reproduces all 81 three-state predicate combinations and rejects 192 deliberately malformed records. Common-guard baselines and predicate ablations isolate endpoint logic from missing-evidence handling, while controlled transitions check support separation and evidence removal. These are finite contract-conformance results, not detec
    
[^202]: JEV作为裁判：自信时接受，不确定时升级

    JEV-as-a-Judge: Accept When Confident, Escalate When Unsure

    [https://arxiv.org/abs/2609.26550](https://arxiv.org/abs/2609.26550)

    提出仅返回标签概率的低成本裁判JEV，通过置信度阈值机制将高置信判定直接接受、不确定判定升级至推理型裁判，从而在费用仅为GPT-6的41%的情况下，实现了比GPT-6高出0.9个百分点的评估准确率。

    

    LLM作为裁判可以扩展评估规模，但推理型裁判速度慢且成本高。我们研究了JEV作为裁判：使用JEV进行评估，JEV是一种仅做决策的裁判，它返回标签概率而非文本，其置信度决定是接受其判定还是升级到推理型裁判。在与十六个生成式裁判和奖励模型裁判的对比中，并经过盲测人工裁决，只要判定可以直接从文本中读出，JEV与GPT-6的差距在三个百分点以内，而费用仅为GPT-6的0.36%，中位延迟为0.15秒；而在必须通过推导才能得出判定的情况下（如数学、代码和逻辑），JEV则落后。其置信度恰好标记了这一能力边界。通过预先固定阈值，接受高置信度的判定并将其余判定升级，在1,610个保留样本对上比GPT-6的准确率高0.9个百分点，而费用仅为GPT-6的41%；在两项新工作负载的预先指定的实时测试中，该级联系统与GPT-6的准确率完全匹配。置信度路由在风格对抗性样本上会减弱……（摘要在此处被截断）

    arXiv:2609.26550v3 Announce Type: replace  Abstract: LLM-as-a-judge scales evaluation, but reasoning judges are slow and costly. We study JEV-as-a-Judge: evaluation with JEV, a decision-only judge that returns label probabilities instead of text, and whose confidence decides whether to accept its verdict or escalate to a reasoning judge. Against sixteen generative and reward-model judges, with blinded human adjudication, JEV comes within three points of GPT-6 wherever a verdict can be read off the text, at 0.36% of its fee and a 0.15-second median latency, and falls behind where the verdict must be derived, as in math, code, and logic. Its confidence marks this boundary. With a threshold frozen in advance, accepting confident verdicts and escalating the rest is 0.9 points more accurate than GPT-6 on 1,610 held-out pairs at 41% of its fee, and in a pre-specified live test on two new workloads the cascade matches GPT-6's accuracy exactly. Confidence routing weakens on style-adversarial p
    
[^203]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^204]: 信任，但要验证工具：在自建安全回归代理上审计AI生成的RTL验证计划

    Trust, but Validate the Instrument: Auditing AI-Generated RTL Verification Plans on Authored Security-Regression Proxies

    [https://arxiv.org/abs/2609.19844](https://arxiv.org/abs/2609.19844)

    论文提出可审计框架SecTB-RTL，通过自建硬件安全回归测试发现AI生成的RTL验证计划虽被提供商接受但几乎全部无法通过生产语义验证，证明提供商模式的接受并不能等同于执行的有效性。

    

    AI生成的RTL验证计划可能满足提供商的模式，但在与可信执行的边界处失败。我们提出了SecTB-RTL，一个涵盖31个任务和124个自建硬件安全回归测试的可审计框架。确定性非AI基线在递增的资源限制下分别杀死了36、75和78个突变体。第一次确认性运行（C1-R2）在模型执行之前就失败了，因为提供商拒绝了其响应模式。在未查看结果的情况下进行仅模式修复后，单独冻结的后续运行（C1-R3）完成了1,860次调用。提供商接受了1,857个响应，但只有九个通过了生产语义验证器。生成规则和执行规则并不匹配。因此，我们将该运行保留为工具验证事件，不报告任何提示效果估计。这一事件表明，提供商或模式的接受并不能证明执行的有效性。编译和覆盖率仅是诊断指标。

    arXiv:2609.19844v1 Announce Type: cross  Abstract: AI-generated RTL verification plans can satisfy a provider schema yet fail at the boundary to trusted execution. We present SecTB-RTL, an auditable framework covering 31 tasks and 124 authored hardware-security regressions. A deterministic non-AI baseline killed 36, 75, and 78 mutants at increasing resource limits. The first confirmatory run (C1-R2) failed before model execution because the provider rejected its response schema. After a schema-only repair made without viewing outcomes, a separately frozen follow-up run (C1-R3) completed 1,860 calls. The provider accepted 1,857 responses, but only nine passed the production semantic validator. The generation and execution rules did not match. We therefore preserve the run as an instrument-validation incident and report no prompt-effect estimate. This incident shows that provider or schema acceptance does not establish execution validity. Compilation and coverage are only diagnostics; th
    
[^205]: IBBench-Light：针对外部指令的任务条件化响应的配对评估

    IBBench-Light: A Paired Evaluation of Task-Conditioned Responses to External Directives

    [https://arxiv.org/abs/2609.13725](https://arxiv.org/abs/2609.13725)

    该论文提出IBBench-Light配对评估基准，证明仅看边际平均准确率会掩盖模型对同一外部记录在“执行”与“处理”两类任务上的表现差异，且仅改变EOS设置即可使精确配对成功率从0/144骤升至62/144。

    

    外部记录可能包含需要执行的过程或需要阅读的文本，具体取决于用户的请求。IBBench-Light 针对同一条记录测试这两种用途。十二个语义基础为每个模型生成144个匹配对；四个量化指令模型共产生1,152条存档的贪婪解码响应。配对精确契约准确率（PECA）要求配对中的两个成员均满足各自的输出契约。Qwen在132个执行提示和109个处理提示上成功，但仅有97个完整配对，这揭示了边际平均值所遗漏的问题。我们审查了字面目标暴露和大小写规范化问题，随后增加1,722条记录的CPU生成，以测试无指令对照、另外十二个语义基础、基础内措辞变化以及生成停止条件。在固定配置的Phi重跑实验中，仅改变序列结束符（EOS）集合就使精确配对成功率从0/144变为62/144。一项有界的IHEval比较使用了相同的SmolLM2检查点和输出预算，同时保留其已公布的指令设置。（注：原始摘要在此处被截断）

    arXiv:2609.13725v1 Announce Type: new  Abstract: An external record may contain a procedure to apply or text to read, depending on the user's request. IBBench-Light tests both uses against the same record. Twelve semantic bases yield 144 matched pairs per model; four quantized instruction models produced 1,152 archived greedy responses. Paired exact-contract accuracy (PECA) requires both members to satisfy their output contracts. Qwen succeeds on 132 execute and 109 process prompts, but only 97 complete pairs, showing what marginal averages omit. We audit literal-target exposure and case normalization, then add 1,722 logged CPU generations to test directive-absent controls, twelve additional semantic bases, within-base wording changes, and generation stopping. In the pinned Phi rerun, changing the end-of-sequence (EOS) set changes exact paired success from 0/144 to 62/144. A bounded IHEval comparison uses the same SmolLM2 checkpoint and output budget while preserving its published inst
    
[^206]: 惯例差距：迈向合作AI评估中隐性交流的度量

    The Convention Gap: Towards Measuring Implicit Communication in Cooperative AI Evaluation

    [https://arxiv.org/abs/2609.11489](https://arxiv.org/abs/2609.11489)

    该论文提出“惯例差距”这一新指标，通过在Hanabi游戏中对比从字面交流预测的失败概率与实际失败率，发现人类玩家在合作中依赖超出字面信息的隐性惯例（差距达26.2个百分点），而AI-AI对局中不存在这种差距，为评估合作AI的隐性交流能力提供了可精确计算的方法。

    

    合作型AI智能体通常与其他AI进行对比评估，然而人类的合作依赖于隐性惯例——即解读超出字面信息含义的共享协议——而AI-AI基准测试可能无法捕捉到这一点。我们提出了“惯例差距”这一概念，即从交流字面内容所预测的失败概率与实际观察到的失败率之间的差异，作为衡量隐性交流的指标。在纸牌游戏Hanabi中，有限的牌堆和确定性的提示约束使得该后验概率可以被精确计算。我们重放了来自三个公开数据集的约101,000个出牌动作，涵盖人与人、AI与AI、人与AI的对局。结果显示，差距在人与人组队中为+26.2个百分点，在AI与AI组队中为-0.7个百分点，在人与AI组队中为+16.4个百分点，且该差距集中在未收到任何提示的牌的出牌行为上（人与人组队中为+46个百分点）。在人与AI的对局中，人类可获得的字面信息……

    arXiv:2609.11489v1 Announce Type: new  Abstract: Cooperative AI agents are evaluated against other AIs, yet human cooperation relies on implicit conventions---shared protocols for reading meaning beyond the literal message---which AI-AI benchmarks may not capture. We propose the \emph{convention gap}, the difference between the failure probability predicted from the literal content of communication and the observed failure rate, as a metric of implicit communication. In the card game Hanabi, the finite deck and deterministic hint constraints make this posterior exactly computable. We replayed about 101,000 play actions from three public datasets of human-human (hanab.live), AI-AI (HOAD), and human-AI (HanabiData) games. The gap was +26.2 percentage points (pp) in human pairs, $-$0.7~pp in AI pairs, and +16.4~pp in human-AI pairs, and was concentrated on plays of cards that had received no hints (+46~pp in human pairs). Within human-AI play, the literal information available to humans w
    
[^207]: 是什么使对抗样本能够跨深度伪造检测器迁移？

    What Makes Adversarial Examples Transfer Across Deepfake Detectors?

    [https://arxiv.org/abs/2609.10002](https://arxiv.org/abs/2609.10002)

    该研究通过对涵盖六种骨干网络、两种预训练方案和五种训练数据配置的60个深度伪造检测器进行受控评估，发现当攻击者的源模型与目标模型共享相同的骨干网络、架构家族、预训练方案或训练数据时，黑盒对抗迁移攻击的成功率显著更高，且这种兼容性效应因攻击方法而异。

    

    深度伪造检测器仍然容易受到基于迁移的黑盒攻击的影响，在这类攻击中，对抗样本在源替代模型上生成，随后迁移到攻击者无法访问的目标模型。然而，源模型与目标模型之间的兼容性如何影响攻击成功率，目前仍缺乏深入理解。先前的研究仅评估了有限的检测器集合，且很少将架构因素与训练因素解耦开来。我们对对抗样本的可迁移性进行了受控评估，涵盖60个检测器，涉及六种骨干网络、两种预训练方案和五种训练数据配置，并使用了两种攻击方法：AutoAttack（AA）和结合变换期望的Carlini-Wagner攻击（CW-EOT）。匹配比较结果表明，当源模型与目标模型共享完全相同的骨干网络、架构家族、预训练方案或训练数据时，攻击的迁移成功率显著更高。这种兼容性结构依赖于攻击方法：完全相同的骨干网络兼容性具有最……（原文此处截断）

    arXiv:2609.10002v1 Announce Type: cross  Abstract: Deepfake detectors remain vulnerable to transfer-based black-box attacks, in which adversarial examples are generated on a source surrogate model and transferred to a target model, unknown to the attacker. Yet how source--target compatibility shapes attack success remains poorly understood. Prior studies evaluate limited detector pools and rarely disentangle architectural from training factors. We conduct a controlled evaluation of adversarial transferability across 60 detectors spanning six backbones, two pretraining regimes, and five training-data configurations, using two attack procedures: AutoAttack (AA) and the Carlini--Wagner attack with Expectation over Transformation (CW--EOT). Matched comparisons reveal significantly higher transfer when source and target share an exact backbone, architecture family, pretraining regime, or training data. This compatibility structure is attack-dependent: exact backbone compatibility has the la
    
[^208]: API基准测试分数无法可靠地迁移到聊天机器人界面

    API Benchmark Scores Do Not Reliably Transfer to Chatbot Interfaces

    [https://arxiv.org/abs/2609.08861](https://arxiv.org/abs/2609.08861)

    研究通过对ChatGPT、Claude和Gemini的系统审计发现，通过API测得的基准测试分数无法可靠反映聊天机器人界面的真实表现，API与界面间的性能差异之大甚至相当于模型降级一个版本。

    

    基准测试分数是模型发布中的核心“货币”：它们为采购决策提供依据，塑造公众信任，并影响政策制定。然而，基准测试分数的一个关键假设是，通过API测量的模型性能能够忠实地反映已部署系统的实际行为。我们通过审计ChatGPT、Claude和Gemini，涵盖七个系统和九个基准测试（横跨通用能力、社会偏见和谄媚性），对这一假设提出了挑战。我们发现API与聊天界面之间存在系统性的准确性和一致性差异。平均而言，API评估的准确率比相应的界面评估高3.4个百分点，重测一致性高2.1个百分点。对于ChatGPT而言，API与界面访问方式之间的性能差异，甚至超过了仅通过API测量的GPT 5.3与GPT 5.4之间的差异。换言之，切换访问渠道所导致的性能下降，可能与降级整个模型版本的影响相当。

    arXiv:2609.08861v1 Announce Type: new  Abstract: Benchmark scores are a central currency in model releases: they inform purchasing decisions, shape public trust, and influence policy. Yet, a key assumption underlying benchmark scores is that the model performance measured through APIs faithfully reflects the behavior of deployed systems.   We challenge this assumption by auditing ChatGPT, Claude, and Gemini across seven systems and nine benchmarks spanning general capability, social bias, and sycophancy. We find systematic API--interface differences in both accuracy and consistency. On average, API evaluations score 3.4 percentage points higher in accuracy and 2.1 percentage points higher in test--retest agreement than corresponding interface evaluations. For ChatGPT, the performance difference between API and interface access exceeds the API-only difference between GPT 5.3 and GPT 5.4. Put differently, switching access surfaces can degrade performance as much as downgrading a full mod
    
[^209]: SUN：在强化学习中追求新颖性

    SUN: Reaching for Novelty in Reinforcement Learning

    [https://arxiv.org/abs/2609.08642](https://arxiv.org/abs/2609.08642)

    本文提出基于后继价值函数的SUN指标，可同时衡量目标的新颖性与可达性，该可达性感知的目标选择框架能无缝集成到任意离策略强化学习算法中，并具有恢复计数奖励、约束命中概率和拒绝不可达目标的理论保证。

    

    强化学习（RL）中的探索仍然是一个根本性的挑战。近来的目标条件强化学习策略（通过选择目标来鼓励更广泛的状态覆盖）已展现出有希望的结果，但没有一种方法能够同时依据新颖性和可达性对目标进行评分：这两个信号要么被手动权衡，要么被依次应用，或者其中之一被完全忽视。在本文中，我们引入了一个可达性感知的目标选择框架，它明确地整合了这两个方面，并且可以无缝地融入任何离策略（off-policy）强化学习算法中。为此，我们提出了SUccessor-to-Novelty（SUN），这是一个由后继价值函数（successor value functions）导出的指标，用于识别既新颖又可达的目标。我们证明SUN在极限情况下可恢复基于计数的奖励，能够约束短时域命中概率，并且可以可证明地拒绝不可达的目标。我们进一步提出了一种利用这些特性的自适应目标选择策略，以及一个准确的……（摘要在此处截断）

    arXiv:2609.08642v1 Announce Type: cross  Abstract: Exploration in reinforcement learning (RL) remains a fundamental challenge. Recent goal-conditioned RL strategies (which select goals to encourage broader state coverage) have shown promising results, but none scores a goal by novelty and reachability jointly: the two signals are traded off by hand, applied in sequence, or one is neglected outright. In this paper, we introduce a reachability-aware goal-selection framework that explicitly integrates these two aspects, and that can be seamlessly incorporated into any off-policy RL algorithm. To this aim, we propose SUccessor-to-Novelty (SUN), an indicator derived from successor value functions to identify goals that are both novel and reachable. We prove that SUN recovers count-based bonuses in the limit, bounds short-horizon hitting probabilities, and provably rejects unreachable goals. We further present an adaptive goal-selection strategy that leverages these properties, and an accura
    
[^210]: SWE-Bench Pro Verified：一个可靠的软件工程智能体基准测试

    SWE-Bench Pro Verified: A Reliable Benchmark for Software Engineering Agents

    [https://arxiv.org/abs/2609.08149](https://arxiv.org/abs/2609.08149)

    该论文发现 SWE-Bench Pro 基准存在答案泄露导致的奖励破解和任务质量问题，并推出了经过验证的 SWE-Bench Pro Verified 版本，通过防泄露保护与任务修正实现更可靠的软件工程智能体评估。

    

    SWE-Bench Pro 已成为评估软件工程智能体在具有挑战性的仓库级任务上表现的标准基准。然而，我们的分析工作表明，其评估结果受到两类不可靠因素的损害：一是“奖励破解”，即由于标准答案或隐藏评估信息的泄露而使智能体得以作弊；二是任务质量问题，包括具有误导性的问题描述和范围界定不当的测试。这些问题可能会夸大基准测试成绩，掩盖智能体真实的编程能力。我们提出了 SWE-Bench Pro Verified，这是 SWE-Bench Pro 的一个经过验证的版本，能够同时解决上述两个问题。我们的方法结合了防破解保护措施（在不干扰智能体正常功能的前提下消除主要的泄露渠道）和任务精炼（对有缺陷实例中的不一致之处进行最小程度的修正）。在 SWE-Bench Pro Verified 上的评估结果显示，一些模型的表现明显变差……

    arXiv:2609.08149v1 Announce Type: new  Abstract: SWE-Bench Pro has emerged as a standard benchmark for evaluating software engineering agents on challenging repository-level tasks. However, our analysis work show that its evaluation is undermined by two sources of unreliability: \textbf{reward hacking}, enabled by leakage of gold solutions or hidden evaluation information, and \textbf{task quality issues}, including misleading problem statements and improperly scoped tests. These issues can inflate benchmark performance and obscure agents' true coding ability. We present \textbf{SWE-Bench Pro Verified}, a verified version of SWE-Bench Pro that addresses both problems. Our approach combines \textbf{anti-hacking} safeguards that eliminate major leakage channels without disrupting normal agent functionality, with \textbf{task refinement} that minimally corrects inconsistencies within flawed instances. Evaluations on SWE-Bench Pro Verified reveal that some models perform substantially wors
    
[^211]: TDDN：面向谜题理解的文本对齐扩散DINO网络

    TDDN: Text-aligned Diffused DINO Network for Puzzle Understanding

    [https://arxiv.org/abs/2609.07937](https://arxiv.org/abs/2609.07937)

    提出TDDN模型，通过融合DINOv3与CleanDIFT表示并与RoBERTa-L进行文本对齐，在仅使用约59万个对齐数据对且冻结骨干网络的情况下，恢复了CLIP所缺乏的细粒度视觉感知能力，在图文检索和密集预测分割任务上大幅超越CLIP，从而提升了图像谜题理解能力。

    

    结构化视觉推理（如图像谜题）需要细粒度的视觉感知能力，而当前的视觉语言模型（VLM）缺乏这种能力。基于CLIP的ViT骨干网络构建的VLM以牺牲细粒度细节为代价换取高层语义，我们证明这种损失会向下游传播。为了恢复这种能力，我们将DINOv3和CleanDIFT表示融合到一个感知编码器（DiffusedDINO）中，并将其与RoBERTa-L对齐，从而得到一个文本对齐的模型TDDN，该模型保留了这种感知优势：在冻结骨干网络且仅使用约59万个对齐数据对的情况下，TDDN在图文检索任务上与CLIP相当，并在四个设置中的三个上超越了CLIP。同时，TDDN将CLIP的密集预测准确率提高了三倍以上（ADE20K从5.20提升到18.11 mIoU，COCO-Stuff从7.35提升到24.44），尽管CLIP拥有庞大的训练语料库。在包括SigLIP 2在内的通用对比编码器中，TDDN在分割基准测试中处于领先地位。我们进一步引入了谜题感知（Puzzle Perception），一种……

    arXiv:2609.07937v1 Announce Type: cross  Abstract: Structured visual reasoning, such as image puzzles, demands fine-grained visual perception, an ability current Vision Language Models (VLMs) lack. VLMs built on CLIP-based ViT backbones trade fine-grained detail for high-level semantics, and we show this loss propagates downstream. To recover it, we fuse DINOv3 and CleanDIFT representations into a perception encoder (DiffusedDINO) and align it with RoBERTa-L, yielding a text-aligned model TDDN that preserves this perceptual advantage: with frozen backbones and only $\sim$590K alignment pairs, TDDN matches CLIP on image-text retrieval, surpassing it on three of four settings. It does so while more than tripling CLIP's dense-prediction accuracy (ADE20K 5.20 $\to$ 18.11 mIoU, COCO-Stuff 7.35 $\to$ 24.44), despite CLIP's massive training corpus. TDDN leads on segmentation benchmarks among general-purpose contrastive encoders, including SigLIP$\,$2. We further introduce Puzzle Perception, a
    
[^212]: 测试协议设置对跨 Trust-Hub 家族基于特征的硬件木马检测的影响

    Protocol effects on feature-based hardware-Trojan detection across Trust-Hub families

    [https://arxiv.org/abs/2609.07199](https://arxiv.org/abs/2609.07199)

    该论文发现基于特征的硬件木马检测性能严重依赖测试集划分协议——当测试门电路来自训练中已见过的宿主电路时性能被高估，而按宿主家族留出时 F1/AP 会从约 0.91/0.98 骤降至约 0.46/0.58。

    

    Trust-Hub 复用宿主电路：多个文件之间的主要区别仅在于插入的木马。当同一家族兄弟变体的门电路同时进入训练集和测试集时，检测器可以从已经见过的宿主逻辑中获益。我们测量的是这种效应，而不是提出又一种分类器。语料库包含来自 16 个网表的 49,124 个门电路，分为五个宿主家族。我们保持解析器、36 个门级特征、类别权重、模型设置、阈值和家族级聚合方式不变，仅改变一个选择：测试集边界。三种设置分别从合并语料库中抽取测试门电路、留出一个完整网表、或留出一个宿主的所有变体。这一选择至关重要：随机森林在合并门电路下 F1/AP 为 0.914/0.978，留出一个完整网表时降至 0.636/0.851，留出一个宿主家族时仅为 0.460/0.577；在同样对比下，XGBoost 从 0.946/0.976 下降到 0.464/0.544；逻辑回归损失了 AP，尽管其固定阈值……

    arXiv:2609.07199v1 Announce Type: cross  Abstract: Trust-Hub reuses host circuits: several files differ mainly in the inserted Trojan. When gates from sibling variants enter both training and test folds, a detector can benefit from host logic it has already seen. We measure that effect instead of proposing another classifier. The corpus contains 49,124 gates from 16 netlists grouped into five host families. We left the parser, 36 gate features, class weighting, model settings, threshold, and family-level aggregation unchanged and altered one choice: the test boundary. The three settings draw test gates from the pooled corpus, withhold a complete netlist, or withhold every variant of one host. The choice matters. Random forest records F1/AP of 0.914/0.978 with pooled gates, 0.636/0.851 with one netlist held out, and 0.460/0.577 with a host family held out. XGBoost falls from 0.946/0.976 to 0.464/0.544 across the same comparison. Logistic regression loses AP, although its fixed-threshold
    
[^213]: 多执行框架强化学习学到了什么？编码智能体中的功劳分配与可移植性

    What Does Multi-Harness RL Learn? Credit Assignment and Portability in Coding Agents

    [https://arxiv.org/abs/2609.04518](https://arxiv.org/abs/2609.04518)

    该研究发现，在多框架强化学习训练编码智能体时，评估框架是决定性能的主导变量（使平均解决率相差4.3倍），远超训练配方本身的影响。

    

    智能体强化学习（RL）越来越多地通过完整的执行框架运行，而多框架配方混合了两个选择：将策略暴露给多个框架，以及在一个相对优势组内比较它们的奖励。我们在仓库级编码任务中分离出第二个选择。从同一个 Qwen3-8B 监督热启动出发，我们以相同的更新次数重放来自 Aider、OpenHands、Qwen Code 和 SWE-agent 的相同冻结任务-框架记录，在组相对策略优化（GRPO）的两种规则下进行训练——Within（每个任务-框架对一组）和 Cross（同一任务内各框架合并为一组）——并用密封的 SWE-bench Verified 预言机对每个检查点在四个源框架和一个未参与训练的最小框架上进行评分。评估框架是主导变量：在 24,000 次密封评估中，它使平均解决率从 2.14% 变动到 9.27%，相差 4.3 倍，而训练配方仅使其变动 1.16。

    arXiv:2609.04518v1 Announce Type: new  Abstract: Agent reinforcement learning (RL) increasingly runs through full execution harnesses, and a multi-harness recipe mixes two choices: exposing the policy to several harnesses, and comparing their rewards inside one relative-advantage group. We isolate the second choice in repository-level coding. From one Qwen3-8B supervised warm start we replay the same frozen task-harness records from Aider, OpenHands, Qwen Code, and SWE-agent, with the same number of updates, under two rules for group-relative policy optimization (GRPO), Within (one group per task-harness pair) and Cross (harnesses pooled within a task), and score every checkpoint with a sealed SWE-bench Verified oracle on four source harnesses and a minimal harness held out of training. The evaluation harness is the dominant variable: across 24,000 sealed evaluations it moves the mean solve rate from 2.14\% to 9.27\%, a factor of 4.3, where the training recipe moves it by 1.16. The gro
    
[^214]: 防御即技能：为技能增强型智能体演化运行时守卫技能

    Defense-as-Skill: Evolving Runtime Guard Skill for Skill-Augmented Agents

    [https://arxiv.org/abs/2609.01487](https://arxiv.org/abs/2609.01487)

    提出“防御即技能”新范式，将运行时安全守卫 SkillSonar 本身实现为可安装、可检查、可编辑的技能，使其与不可信任务技能并行运行，并依据用户任务边界对敏感操作进行条件化检查（允许、重新规划或确认），同时构建了覆盖 6 个风险类别的 SCOPE-R 数据集。

    

    技能增强型智能体将可复用技能作为持久的运行时上下文进行加载，这虽然提升了任务性能，但也为恶意技能开辟了一条持久通道，使其能够影响智能体未来的行动。此类技能可能在具体用户任务和工作区状态使不安全操作显得有用时，才泄露机密、破坏代码、绕过审批，或暂存数据以待外泄。这使得仅靠安装前的审查已不再足够，需要运行时的、任务条件化的保护。我们提出 Defense-as-Skill（防御即技能），一种将运行时守卫本身实现为可安装、可检查、可编辑技能的防御范式。我们的守卫 SkillSonar 与不可信的任务技能并行运行，依据用户的任务边界对敏感操作进行检查，并将每个操作路由至允许、重新规划或请求确认的决策，而无需修改底层智能体运行时。为研究这一场景，我们构建了 SCOPE-R，一个任务条件化的数据集，涵盖 6 个风险类别和 21 个子类别。

    arXiv:2609.01487v1 Announce Type: cross  Abstract: Skill-augmented agents load reusable skills as persistent runtime context, improving task performance but also giving malicious skills a durable channel for steering future actions. Such skills may leak secrets, corrupt code, bypass approvals, or stage data for exfiltration only after a concrete user task and workspace state make the unsafe action appear useful. This makes pre-install vetting insufficient and calls for runtime, task-conditioned protection. We propose Defense-as-Skill, a defense paradigm that implements the runtime guard itself as an installable, inspectable, and editable skill. Our guard, SkillSonar, runs alongside untrusted task skills and checks sensitive actions against the user's task boundary, routing each action to an allow, replan, or confirmation decision without modifying the underlying agent runtime. To study this setting, we construct SCOPE-R, a task-conditioned dataset covering 6 risk families and 21 sub-ca
    
[^215]: H2Table：基于层次超图增强的大语言模型复杂表格推理方法

    H2Table: Hierarchical Hypergraph-Enhanced Large Language Models for Complex Table Reasoning

    [https://arxiv.org/abs/2609.01216](https://arxiv.org/abs/2609.01216)

    提出H2Table框架，将复杂表格表示为层次嵌套超图，通过定制超图编码器建模表头与单元格间的语义关系，并利用可学习查询向量将结构嵌入注入大语言模型，从而显著提升复杂表格推理能力。

    

    表格在各个领域中无处不在，然而对表格进行推理仍然是现代大语言模型（LLM）面临的重大挑战。当前的方法通常将表格线性化为序列，这本质上忽略了表格固有的二维和层次结构。为了解决这个问题，我们提出了H2Table（层次超图增强表格推理），这是一个将复杂表格表示为层次嵌套超图的新颖框架。为了处理这种表示，我们设计了一个定制的超图编码器，以促进超边（表头）和节点（单元格）之间的消息传递，从而感知复杂表格中它们之间的语义蕴含关系。此外，我们引入了一组可学习的查询向量，作为轻量级桥梁，将编码器中具有代表性的结构嵌入提取到大语言模型中。实验结果表明，我们的方法能够有效处理复杂……

    arXiv:2609.01216v1 Announce Type: new  Abstract: Tables are ubiquitous across diverse domains, yet reasoning over them remains a significant challenge for modern large language models (LLMs). Current approaches typically linearize tables into sequences, inherently overlooking their intrinsic two-dimensional and hierarchical structure. To address this, we propose H2Table (Hierarchical Hypergraph-Enhanced Table Reasoning), a novel framework that represents complex tables as hierarchical nested hypergraphs. To process this representation, we design a tailored hypergraph encoder to facilitate message passing between hyperedges (headers) and nodes (cells), thereby perceiving the semantic entailment relationships between them within complex tables. Furthermore, we introduce a set of learnable query vectors acting as a lightweight bridge to extract representative structural embeddings from the encoder into the LLM. Experimental results demonstrate that our approach effectively handles complex
    
[^216]: 数据驱动的角色条件化智能体用于A/B测试模拟

    Data-Driven Persona-Conditioned Agents for A/B Test Simulation

    [https://arxiv.org/abs/2609.01038](https://arxiv.org/abs/2609.01038)

    该论文提出利用基于真实用户匿名行为数据（活动模式、参与度信号、人口统计特征）构建的数据驱动角色来条件化LLM智能体，从而更忠实地模拟用户群体并预测A/B测试结果，替代昂贵耗时的真实用户实验。

    

    A/B测试是评估产品变更的黄金标准，但每次实验都需要真实用户流量、工程投入以及长达数周的测量时间。我们提出了一个模拟框架，使用基于大语言模型（LLM）的智能体来预测A/B测试结果，这些智能体以基于真实用户行为信号构建的数据驱动角色（persona）为条件。与以往依赖合成角色或基于规则角色的研究不同，我们的智能体由匿名化的行为数据构建——包括活动模式、参与度信号和推断的人口统计特征——从而实现更忠实的人群建模。我们将A/B测试模拟构建为一个结构化问题任务，并系统性地研究了：(i) 问题设计格式；(ii) 角色数据来源和领域对齐的影响；(iii) 单个角色行为深度与人群多样性之间的权衡；(iv) 高效的人群子采样方法。在一个涵盖两种指标类型的40个A/B测试基准上，我们的最佳配置取得了……

    arXiv:2609.01038v1 Announce Type: new  Abstract: A/B testing is the gold standard for evaluating product changes, but each experiment requires real user traffic, engineering effort, and weeks of measurement. We propose a simulation framework that predicts A/B test outcomes using LLM-powered agents conditioned on data-driven personas grounded in real user behavioral signals. Unlike prior work that relies on synthetic or rule-based personas, our agents are constructed from anonymized behavioral data-activity patterns, engagement signals, and inferred demographics-enabling more faithful population modeling. We frame A/B test simulation as a structured question task and systematically study (i) question design formats, (ii) the impact of persona data source and domain alignment, (iii) the trade-off between per-persona behavioral depth and population diversity, and (iv) efficient population subsampling. On a benchmark of 40 A/B tests spanning two metric types, our best configuration achieve
    
[^217]: DASC：面向混合线性注意力服务的衰减感知状态压缩

    DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving

    [https://arxiv.org/abs/2608.30386](https://arxiv.org/abs/2608.30386)

    提出衰减感知状态压缩（DASC），通过从模型权重中推导不同头和通道的“保留视野”来压缩混合线性注意力模型的持久状态检查点，从而缓解内存压力、减少缓存驱逐和重复预填充。

    

    混合线性注意力架构最近已扩展至大型开源权重模型，在提供与全注意力相当的质量的同时，大幅减少了键/值（KV）缓存的增长。然而，其原位递归状态更新使缓存管理变得复杂：前缀重用需要在全注意力KV之外额外保存状态检查点，而完整存储状态检查点又会增加内存压力，导致更多的缓存驱逐和重复预填充。通过分析门控DeltaNet（GDN）和Kimi Delta注意力（KDA）的衰减结构，我们发现不同的头和通道在显著不同的时间尺度上保留前缀信息，我们将其称为“保留视野”。这种差异表明持久状态检查点存在巨大的压缩潜力。基于这一观察，我们提出了衰减感知状态压缩（DASC），它从模型权重中推导保留视野，选择长视野状态……（摘要在此处截断）

    arXiv:2608.30386v1 Announce Type: cross  Abstract: Hybrid linear-attention architectures have recently scaled to large open-weight models, offering quality competitive with full attention while substantially reducing key/value (KV) cache growth. However, their in-place recurrent-state updates complicate cache management: prefix reuse requires state checkpoints alongside full-attention KV, while storing state checkpoints in full increases memory pressure, leading to more evictions and repeated prefill. By analyzing the decay structure of Gated DeltaNet (GDN) and Kimi Delta Attention (KDA), we find that different heads and channels retain prefix information over markedly different timescales, which we term \emph{retention horizons}. This variation suggests substantial compression potential in persistent state checkpoints. Building on this observation, we introduce \emph{Decay-Aware State Compression} (DASC), which derives retention horizons from model weights, selects long-horizon state 
    
[^218]: 上报渠道能否将奖励黑客行为转向缺陷披露？

    Can escalation channels redirect reward hacking toward defect disclosure?

    [https://arxiv.org/abs/2608.29460](https://arxiv.org/abs/2608.29460)

    该论文提出在代码智能体遭遇测试基础设施缺陷时为其提供结构化上报工具（上报渠道），可将其奖励黑客行为从23.6%降至5.3%，同时暴露出触发该行为的底层缺陷。

    

    当代码智能体遇到有缺陷的测试基础设施时，可能会进行奖励黑客行为：硬编码输出或编辑测试文件，以通过它们无法合法满足的测试。这种模式如今已出现在基准测试之外，出现在对某大型AI平台生产基础设施的协调式多智能体入侵事件中。让智能体能够检测并利用缺陷的同一能力，在合适的决策环境下，也可以让它转而上报缺陷。我们评估了上报渠道——即在冲突发生时可供智能体使用的结构化报告工具——作为一种决策环境干预手段，它既能减少奖励黑客行为，又能暴露出触发该行为的底层基础设施缺陷。一项2×2因子实验设计分离了上报工具、独立的反奖励黑客策略以及二者组合各自的贡献。在涵盖5个模型家族的8个前沿模型上，组合干预将奖励黑客行为从23.6%降低到5.3%（混合效应模型……）

    arXiv:2608.29460v1 Announce Type: new  Abstract: When coding agents encounter defective test infrastructure they may reward-hack: hardcoding outputs or editing test files to pass tests they cannot legitimately satisfy, a pattern that has now appeared outside benchmarks, in a coordinated multi-agent intrusion of a major AI platform's production infrastructure. The same capability that lets an agent detect and exploit a defect could let it report one, given the right decision environment. We evaluate escalation channels, structured reporting tools available to the agent at the point of conflict, as a decision-environment intervention that both reduces reward hacking and surfaces the infrastructure defects that trigger it. A $2 \times 2$ factorial separates the contributions of an escalation tool, a standalone anti-reward-hacking policy, and their combination. Across 8 frontier models spanning 5 families, the combined intervention reduces reward hacking from 23.6\% to 5.3\% (mixed-effects
    
[^219]: 面向策略优化的偏好引出方法及其在心脏移植与人类价值对齐中的应用

    Preference Elicitation for Policy Optimization and Application to Aligning Heart Transplantation with Human Values

    [https://arxiv.org/abs/2608.28620](https://arxiv.org/abs/2608.28620)

    提出一种两阶段的线性效用偏好引出算法，通过成对比较学习切割平面并可证明收敛到用户效用函数，从而直接从分配结果学习效用函数用于策略优化，并成功应用于心脏移植分配的价值对齐问题。

    

    偏好引出对于将人工智能系统与人类价值观对齐至关重要。以往的方法（例如用于器官分配的方法）通常要求利益相关者比较算法做出的决策（例如患者A与患者B孰先孰后）。这种决策层面的方法混淆了手段与目的。与此不同，我们直接针对分配结果引出偏好，以学习用于策略优化的效用函数。我们构建了一种新颖的面向线性效用的偏好引出算法，其在实践中优于以往的技术。我们的算法分为两个阶段：第一阶段通过成对比较学习切割平面，快速缩小可能的属性权重空间，并通过剔除被支配区域为第二阶段热身；第二阶段则可证明地收敛到用户的效用函数。我们将该技术应用于心脏移植分配问题，其中分配策略必须平衡移植后结局与等待期死亡率等相互竞争的目标。

    arXiv:2608.28620v1 Announce Type: new  Abstract: Preference elicitation is essential for aligning AI systems with human values. Prior approaches (e.g., for organ allocation) often ask stakeholders to compare the decisions of an algorithm (e.g., patient A vs. patient B). Such a decision-level approach conflates the means with the ends. Instead, we elicit preferences directly over allocation outcomes to learn a utility function for policy optimization. We construct a novel preference elicitation algorithm for linear utilities that outperforms prior techniques in practice. Our algorithm has two phases. The first phase learns cutting planes through pairwise comparisons to rapidly shrink the space of possible attribute weights and warm-starts the second phase by eliminating dominated regions. The second phase then provably converges to the user's utility function. We apply our technique to heart transplant allocation where a policy must balance competing objectives such as post-transplant o
    
[^220]: GenCoord：私有信息下的技能路径承诺机制

    GenCoord: Skill-Path Commitments under Private Information

    [https://arxiv.org/abs/2608.22055](https://arxiv.org/abs/2608.22055)

    GenCoord提出了一种在私有信息条件下，通过本地模型生成技能路径承诺并利用有界反馈机制，将任务后果转化为可执行行动的协作方法，从而解决具身智能体间的信息不对称问题。

    

    摘要：假设一个具身智能体知道必须构建什么，而其队友独自知道其工作单元能执行哪种变换。任何局部视角都无法确定谁应该行动、应该交接什么、以及联合任务应如何继续。我们引入了GenCoord，它将这类私有事实的任务后果转化为可执行的技能路径承诺。一个本地Qwen3.5-0.8B模型生成多步骤的SELF计划和同伴REQ；有界反馈条件在决策能力位于同伴本地时引导修订路径。解析后的承诺经过检查、规范物化、编译为Mineflayer技能，并通过交接和终止状态进行验证。保持世界、调用计划和执行者不变的反事实干预，使得请求者修订和接收者执行在两个方向上都遵循注入的任务后果。在三个独立训练的种子中，正确的能力反馈弥合了配对的本地-全局信息差距。

    arXiv:2608.22055v1 Announce Type: new  Abstract: Suppose one embodied agent knows what must be built, while its teammate alone knows which transformation its workcell can perform. Neither local view determines who should act, what should be handed off, or how the joint task should continue. We introduce GenCoord, which turns the task consequence of such private facts into an executable skill-path commitment. A local Qwen3.5-0.8B model emits a multi-step SELF plan and peer REQ; bounded feedback conditions route revision when the deciding capability is peer-local. The resolved commitment is parsed, checked, canonically materialized, compiled to Mineflayer skills, and verified by handoff and terminal state. Counterfactual interventions that hold the world, call schedule, and executor unchanged make requester revision and receiver execution follow the injected task consequence in both directions. Across three independently trained seeds, correct capability feedback closes the paired local-
    
[^221]: 从掌握度画像到模拟响应：用于忠实大语言模型学生模拟的随机学生知识图谱（SSKG）

    From Mastery Profile to Simulated Response: Stochastic Student Knowledge Graphs (SSKG) for Faithful LLM Student Simulation

    [https://arxiv.org/abs/2608.21668](https://arxiv.org/abs/2608.21668)

    本文提出基于随机学生知识图谱（SSKG）的方法，以解决大语言模型在模拟低掌握水平学生时因内置能力导致难以区分掌握程度差异的问题，从而实现更忠实的学生模拟。

    

    arXiv:2608.21668v1 公告类型：新 摘要：大语言模型（LLMs）越来越多地被用于模拟不同掌握水平的学生。这些模拟可以生成合成训练数据，并对辅导系统进行压力测试。然而，常见的基于提示的方法将答题决策交给大语言模型，即使被指示模拟低掌握水平的学生，模型也倾向于根据其内置能力进行表现。因此，这些方法可能难以区分低掌握和高掌握水平的学生。我们使用379道经大学理事会校准的SAT代数题目和五种典型掌握度画像来证明这一局限性。来自三家供应商的三个大语言模型（Gemini 3.1 Flash Lite、Claude Haiku 4.5和GPT-5.4-mini）在所有画像上实现了96.8%至100%的准确率。为解决这一局限性，我们引入了一种基于随机学生知识图谱（SSKG）的方法。从一本开放代数教科书中提取课程知识图谱（CKG），并...

    arXiv:2608.21668v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to simulate students at different mastery levels. These simulations can generate synthetic training data and stress-test tutoring systems. However, common prompt-based approaches leave the answer decision to the LLM, which tends to perform according to its built-in capabilities even when instructed to simulate a student with low mastery. As a result, these approaches may have difficulty distinguishing students with low and high levels of mastery. We demonstrate this limitation using 379 College Board-calibrated SAT Algebra items and five archetypal mastery profiles. Three LLMs from three vendors (Gemini 3.1 Flash Lite, Claude Haiku 4.5, and GPT-5.4-mini) achieve 96.8-100% accuracy across all profiles. To address this limitation, we introduce a method grounded in a Stochastic Student Knowledge Graph (SSKG). A curriculum knowledge graph (CKG) is extracted from an open algebra textbook, and
    
[^222]: 对话式与仪表盘式可解释人工智能在无人机入侵检测中的对比：关于操作员信任与依赖的实证研究

    Conversational versus Dashboard Explainable AI for UAV Intrusion Detection: An Empirical Study of Operator Trust and Reliance

    [https://arxiv.org/abs/2608.10434](https://arxiv.org/abs/2608.10434)

    该研究提出一种由大语言模型驱动的对话式可解释AI界面，并通过受控实验证明其在无人机入侵检测的事后审计任务中，相比传统XAI仪表盘能更好地提升操作员的理解、信任与依赖。

    

    基于机器学习的入侵检测系统（IDS）在保障无人机（UAV）网络安全方面已展现出卓越性能。然而，这些模型的“黑箱”特性，加上多模态信息物理数据的高维度，带来了显著的可解释性挑战。静态可视化仪表盘可能难以以易于操作员查看和解读的形式呈现多模态信息物理特征之间的复杂关系。为解决这一问题，我们提出了一种由大语言模型（LLM）驱动的对话式可解释人工智能（XAI）界面，以便按需开展深入调查。在一项有参与者参与的受控实验中，我们系统地评估了这种对话式界面与传统XAI仪表盘相比，在事后审计任务中对操作员的理解、信任和依赖所产生的影响。我们的结果表明，对话式界面被认为更加……

    arXiv:2608.10434v2 Announce Type: replace  Abstract: Machine learning-based Intrusion Detection Systems (IDS) have demonstrated superior performance in securing Unmanned Aerial Vehicle (UAV) networks. However, the 'black-box' nature of these models, combined with the high dimensionality of multimodal cyber-physical data, poses significant interpretability challenges. Static visualization dashboards may struggle to present complex relationships among multimodal cyber-physical features in a form that is easy for operators to inspect and interpret. To address this, we propose a Conversational XAI interface powered by Large Language Models (LLM) to facilitate on-demand investigation. In a controlled experiment with participants, we systematically evaluated the impact of this conversational interface versus a traditional XAI Dashboard on operator understanding, trust, and reliance during post-incident auditing tasks. Our results suggest that the conversational interface was perceived as mor
    
[^223]: STAIR：使用端到端智能体规划框架实现有效的事件响应

    STAIR: Effective Incident Response Using an End-to-End Agentic Planning Framework

    [https://arxiv.org/abs/2608.09524](https://arxiv.org/abs/2608.09524)

    STAIR提出了一个围绕Graph-as-State共享事件状态组织的端到端智能体响应框架，通过统一的决策依据在长时间跨度的多阶段安全事件响应中有效维持事件状态并对齐不断演进的恢复目标。

    

    事件响应规划对于恢复受损的软件系统至关重要。现有的AIOps方法能够自动化故障诊断与修复，并利用反馈来调整响应，但在规划与执行之间缺乏统一的决策依据。这限制了它们在长时间跨度、多阶段的安全事件响应过程中维持事件状态并使行动与不断变化的恢复目标保持一致的能力。我们提出了STAIR，这是一个端到端的智能体响应框架，它围绕共享的事件状态来组织规划与执行。STAIR在Graph-as-State（GAS）中维护事件证据、响应动作和执行反馈，并利用最新状态来确定响应阶段以及面向阶段专项规划的恢复目标。该框架基于当前状态和阶段检索经过效果验证的历史经验，以指导动作选择。执行工具负责执行动作并将反馈返回给……

    arXiv:2608.09524v2 Announce Type: replace-cross  Abstract: Incident response planning is critical for restoring compromised software systems. Existing AIOps methods automate fault diagnosis and remediation and adapt response using feedback, but lack a unified decision basis across planning and execution. This limits their ability to maintain incident state and align actions with evolving recovery objectives during long-horizon, multi-stage security incident response. We present STAIR, an end-to-end agentic response framework that organizes planning and execution around a shared incident state. STAIR maintains incident evidence, response actions, and execution feedback in Graph-as-State (GAS), using the latest state to determine the response stage and recovery objectives for stage-specialized planning. The framework retrieves effect-validated historical experience based on the current state and stage to guide action selection. An Execution Harness executes actions, returns feedback to u
    
[^224]: 不同的反馈，不同的更新：基于用户交互的大语言模型选择性自学习

    Different Feedback, Different Updates: Selective Self-Learning from User Interactions for Large Language Models

    [https://arxiv.org/abs/2608.09109](https://arxiv.org/abs/2608.09109)

    该论文提出SLIFT框架，将用户反馈按任务相关性分解为Fix、Spec、Null三类原子组件，并通过通才和专才两个互补的LoRA适配器分别在不同泛化范围内整合相应改变，实现大语言模型从用户交互中的选择性自学习。

    

    用户反馈为大语言模型的持续改进提供了天然的监督信号，但单条消息可能支持多种具有不同泛化范围的行为改变。我们提出了SLIFT，一个建立在任务相对视角的用户反馈之上的选择性自学习框架。SLIFT将每条反馈消息分解为原子组件，并将每个组件相对于原始任务解释为Fix、Spec或Null：即任务有效性的要求、兼容的条件特定细化改进，或没有可靠正向更新方向的内容。为了在适当的范围纳入每种改变，SLIFT在共享的冻结骨干模型上训练两个互补的LoRA适配器：一个通才适配器，通过反馈条件下的自蒸馏将Fix要求整合为默认行为；一个专才适配器，仅观察任务和通才的响应，为适用的、尚未满足的Spec细化提供残差指导。Null组件……

    arXiv:2608.09109v2 Announce Type: replace  Abstract: User feedback offers natural supervision for persistent LLM improvement, but a single message may support multiple behavioral changes with different scopes of generalization. We introduce SLIFT, a selective self-learning framework built on a task-relative view of user feedback. SLIFT decomposes each feedback message into atomic components and interprets each component relative to the original task as Fix, Spec, or Null: requirements for task validity, compatible condition-specific refinements, or content with no reliable positive update direction. To incorporate each change at the appropriate scope, SLIFT trains two complementary LoRA adapters on a shared frozen backbone: a Generalist that consolidates Fix requirements into default behavior through feedback-conditioned self-distillation, and a Specialist that observes only the task and Generalist response to supply residual guidance for applicable, unmet Spec refinements. Null compon
    
[^225]: ResidencyRL：模拟临床环境中的强化学习

    ResidencyRL: Reinforcement Learning in Simulated Clinical Environments

    [https://arxiv.org/abs/2608.07418](https://arxiv.org/abs/2608.07418)

    ResidencyRL提出了一种强化学习方法，通过模拟多轮临床诊疗环境（含对抗性患者模拟器和结构化诊断奖励）来训练临床AI智能体，使其像住院医师一样在实践互动中提升完整临床决策序列的能力。

    

    在医学教育中，医生通过住院医师培训将学术知识转化为临床专长：历经数年的训练、数千次的诊疗实践，获得多样化的反馈来源，并逐步获得更大的自主权。临床推理在很大程度上依赖于患者诊疗过程——这是一种对话式的互动，临床医生在其中采集病史、完善诊断假设，并在不确定性下做出治疗决策。虽然大语言模型（LLM）在静态医学基准测试中表现出色，但优化完整临床决策序列的方法仍然发展不足。我们提出了ResidencyRL，这是一种强化学习方法，通过模拟的多轮临床诊疗过程（每次轨迹最多包含60轮对话和8次工具调用）来训练临床人工智能（AI）智能体。ResidencyRL将策略智能体与能够表现出复杂、对抗性行为的LLM模拟器配对，通过与诊断准确性对齐的结构化奖励进行训练。

    arXiv:2608.07418v2 Announce Type: replace  Abstract: In medical education, physicians convert academic knowledge into clinical expertise through residency: years of training across thousands of encounters, with diverse sources of feedback and progressively greater autonomy. Much of clinical reasoning relies on the patient encounter, a dialogue in which a clinician elicits history, refines diagnostic hypotheses, and decides management under uncertainty. While large language models (LLMs) excel on static medical benchmarks, methods to optimize the full sequence of clinical decisions remain underdeveloped. We present ResidencyRL, a reinforcement learning (RL) method for training clinical artificial intelligence (AI) agents through simulated multi-turn clinical encounters (up to 60 dialogue turns and 8 tool calls per trajectory). ResidencyRL pairs the policy agent with LLM simulators capable of complex, adversarial behaviors, training against a structured reward aligned to diagnostic accur
    
[^226]: 快速 LapSum：百万规模下的精确可微分 Top-$k$ 运算

    Fast LapSum: Exact Differentiable Top-$k$ at Million Scale

    [https://arxiv.org/abs/2608.06912](https://arxiv.org/abs/2608.06912)

    Fast LapSum 通过概率区间限定与认证遍历将排序成本限制在阈值附近的窄区间内，实现了百万规模下保持精确选择质量且支持端到端微分的高效可微 top-k 求解。

    

    选择 top-$k$ 元素是在大规模模型和优化问题中诱导稀疏性的一项基础操作，可用于实现鲁棒的专家激活、token 路由或注意力剪枝。然而，硬性 top-$k$ 运算不可微分，而现有的可微分替代方法会随着坐标数量的增长而变得愈发昂贵。我们提出了 Fast LapSum，这是一种针对 LapSum 软 top-$k$ 公式化的可扩展求解器，它在保持精确选择质量 $k$ 的同时，支持端到端微分。在 Fast LapSum 中，我们通过概率区间限定来降低排序成本，该方法利用从核噪声样本中获得的二项式顺序统计量，将排序限制在阈值附近的一个狭窄分数区间内。一次认证遍历以额外一次线性遍历的代价，将概率定位升级为经过验证的定位，并配有覆盖最坏情况的全排序回退机制。我们的认证 GPU 实现……

    arXiv:2608.06912v2 Announce Type: replace  Abstract: Selecting the top-$k$ elements is a fundamental operation for inducing sparsity in large-scale models and optimization problems, enabling robust expert activation, token routing or attention pruning. However, hard top-$k$ is non-differentiable, while existing differentiable alternatives become increasingly expensive as the number of coordinates grows. We introduce Fast LapSum, a scalable solver for the LapSum soft top-$k$ formulation that preserves an exact selection mass of $k$, while supporting end-to-end differentiation. In Fast LapSum, we reduce sorting cost using probabilistic bracketing, which restricts sorting to a narrow band of scores around the threshold using a binomial order-statistic from kernel-noised samples. A certification pass upgrades the probabilistic localization to a verified one at the cost of one additional linear pass, with a full-sort fallback that covers the worst-case scenario. Our certified GPU implementa
    
[^227]: 当前AI基准测试的未测量之处：模态、搜索、引用及其影响（对安全评估而言）

    What Current AI Benchmarks Leave Unmeasured: Modality, Search, Citations, and Implications (for Safety Evaluations)

    [https://arxiv.org/abs/2608.06202](https://arxiv.org/abs/2608.06202)

    该研究通过比较ChatGPT聊天界面与OpenAI API两种模态在有无网络搜索条件下的表现（涵盖401个提示词、4,812个响应），揭示了当前LLM基准评估所忽视的关键维度——如响应一致性、引用依据和弃答行为——并指出这些盲区对安全评估具有重要启示。

    

    大语言模型（LLM）基准评估通常被用来支持关于模型安全性、可靠性和部署准备度的主张。然而，大多数评估依赖于单一访问模态（模型API），对每个提示词仅执行单次运行，并将准确率作为主要结果指标，而没有考虑诸如网络搜索等可能影响模型部署行为的条件。我们对使用最广泛的LLM之一进行了这些假设的审计，比较了两种模态——ChatGPT的聊天界面和OpenAI的API——在启用和禁用网络搜索条件下的表现。我们使用了来自两个流行基准BBQ和SafetyBench的401个提示词的分层总样本，在每个提示词三次重复运行中共收集了4,812个响应。除了标准性能指标外，我们还评估了模型输出的多个维度，包括响应一致性、响应文本相似度、引用依据和弃答行为。例如，聊天界面…

    arXiv:2608.06202v2 Announce Type: replace-cross  Abstract: Large language model (LLM) benchmark evaluations are routinely used to support claims about model safety, reliability, and deployment readiness. Yet most evaluations rely on a single access modality (model APIs), perform a single run per prompt, and report accuracy as the primary outcome metric, without accounting for conditions such as web search that may have effects on model behavior in deployment. We audit these assumptions for one of the most widely-used LLMs, comparing two modalities, ChatGPT's chat UI and OpenAI's API, with and without web search enabled. We use a stratified total sample of 401 prompts from two popular benchmarks, BBQ and SafetyBench, collecting 4,812 total responses across three repeated runs per prompt. Beyond standard performance measures, we evaluate model output dimensions including response consistency, response text similarity, citation grounding, and abstention behavior. For instance, chat UI res
    
[^228]: CASCADE：一个基于智能体调控网络框架的患者数据验证的下游扰动预测

    CASCADE: An Agentic Regulatory Network Framework for Patient-Data-Validated Downstream Perturbation Prediction

    [https://arxiv.org/abs/2608.05359](https://arxiv.org/abs/2608.05359)

    CASCADE是一个通过MCP暴露的智能体调控网络框架，创新性地使用真实TCGA患者肿瘤数据（以局灶基因拷贝数扩增作为敲低的剂量代理）验证基因扰动预测的“方向准确性”而非“成员关系”，在MYC靶点预测中于三种癌症类型达到72%-90%的准确率，并在独立队列METABRIC中得到复现。

    

    CASCADE是一个智能体框架，它从预计算的ARACNe调控网络预测基因扰动的下游转录效应，并通过MCP协议进行暴露。先前的工作通过检查预测的基因是否为已知癌症基因（成员关系）来验证此类工具；我们则测试预测的变化方向是否与现实相符，使用局灶基因拷贝数扩增作为敲低反向效应的基于剂量的代理指标，并与真实的TCGA患者肿瘤数据进行对比验证。对于MYC，CASCADE预测的敲低靶点在三种癌症类型中与真实的扩增vs非扩增肿瘤表达显示出强烈的一致性（BRCA：90.0%，COAD：72.0%，STAD：85.7%；所有p<0.0013），远高于置换基线，通过了PAM50亚型对照检验，并在独立队列（METABRIC，87.2%）中得到复现。与通过Fisher精确检验的经人工整理的MSigDB基因集基线相比，CASCADE的准确性并未显示超出

    arXiv:2608.05359v2 Announce Type: replace  Abstract: CASCADE is an agentic framework that predicts downstream transcriptional effects of gene perturbation from precomputed   ARACNe regulatory networks, exposed via MCP. Prior work validates such tools by checking whether predicted genes are   known cancer genes (membership); we instead test whether the predicted direction of change matches reality, using   focal-gene copy-number amplification as a dosage-based proxy for the inverse of knockdown against real TCGA patient   tumor data.   For MYC, CASCADE's predicted knockdown targets show strong concordance with real amplified-vs-non-amplified tumor   expression across three cancer types (BRCA: 90.0%, COAD: 72.0%, STAD: 85.7%; all p<0.0013), well above permutation   baselines, surviving a PAM50 subtype control and replicating in an independent cohort (METABRIC, 87.2%). Compared   against curated MSigDB gene-set baselines via Fisher's exact test, CASCADE's accuracy is not shown to exceed e
    
[^229]: 基于上下文感知语义嵌入的表格学习器增强方法

    Enhancing Tabular Learners with Context-Aware Semantic Embeddings

    [https://arxiv.org/abs/2608.03565](https://arxiv.org/abs/2608.03565)

    提出CASE框架，通过用代表性行样本预填充基于Gemma 3的表格语言模型的KV缓存来为表格行嵌入提供数据集级语义上下文，从而弥合大语言模型语义理解与表格学习器统计能力之间的差距，解决文本特征的语义歧义问题。

    

    虽然现代表格学习器擅长捕捉统计模式，但它们常常在“语义真空”中运行，将文本特征视为离散符号，忽略了特征名称或单元格条目中蕴含的丰富语义。我们提出了CASE（上下文感知语义嵌入），这是一个新颖的框架，旨在弥合大型语言模型（LLM）的语义理解能力与表格学习器统计能力之间的鸿沟。与现有方法将行独立嵌入不同，CASE采用了一种上下文化策略：我们使用具有代表性的行样本预填充一个基于Gemma 3定制训练的表格语言模型的KV缓存，从而建立数据集语义的持久锚点。这确保了生成的行嵌入能够被动态上下文化，解决语义歧义，并将表示锚定在特定领域的上下文中。我们在多个基准测试（CARTE、TextTab等）上进行了实验……

    arXiv:2608.03565v2 Announce Type: replace  Abstract: While modern tabular learners excel at capturing statistical patterns, they frequently operate in a semantic vacuum, treating textual features as discrete symbols, ignoring the rich semantics inherent in feature names or cell entries. We propose CASE (Context-Aware Semantic Embeddings), a novel framework that bridges the gap between the semantic understanding of Large Language Models (LLMs) and the statistical capabilities of tabular learners. Unlike existing methods that embed rows in isolation, CASE utilizes a contextualization strategy: we pre-fill the KV cache of a custom-trained Gemma 3-based Tabular Language Model with a representative sample of rows to establish a persistent anchor of the dataset's semantics. This ensures that generated row embeddings are dynamically contextualized, resolving semantic ambiguities and anchoring representations in domain-specific context. Our experiments across several benchmarks (CARTE, TextTab
    
[^230]: 一阶上下文内沙普利：面向大语言模型对齐与评估的可扩展数据审计

    First-Order In-Context Shapley: Scalable Data Auditing for LLM Alignment and Evaluation

    [https://arxiv.org/abs/2607.22766](https://arxiv.org/abs/2607.22766)

    该论文提出了一种仅需推理、无需重训练的一阶上下文内 Shapley 数据估值方法，可在 O(k) 次前向传播内完成对大语言模型对齐数据的可扩展审计，以 19 倍更少的计算量实现了与蒙特卡洛上下文内 Shapley 在全部目标样本上一致的估值效果。

    

    大语言模型的后训练对齐正日益受到数据质量的制约：偏好数据与指令微调语料库中不断累积标签噪声、安全反转以及标注者之间的矛盾，而去重方法无法发现这些问题，全语料库的大模型评审又成本高昂难以负担。虽然 Data Shapley 提供了一个有原则性的数据估值框架，但精确计算需要进行难以承受的重训练，而诸如 KNN-Shapley 之类的嵌入代理方法则无法区分高效用与低效用的演示样本（与生成式效用的相关性仅为 ρ = 0.0003）。我们提出了一种仅基于推理的估值流水线，它在语义 k 近邻邻域上计算上下文内合作博弈的一阶项，仅需 O(k) 次前向传播且无需任何参数更新；该估值在 100% 的目标样本上与多次采样蒙特卡洛上下文内 Shapley 呈正相关，同时计算量减少 19 倍。通过将成对对数似然变化投影到有向影响图上，可以刻画每个……（摘要在此处被截断）

    arXiv:2607.22766v2 Announce Type: replace-cross  Abstract: Post-training alignment of Large Language Models is increasingly bottlenecked by data quality: preference and instruction-tuning corpora accumulate label noise, safety inversions, and inter-annotator contradictions that deduplication misses and full-corpus LLM judging cannot afford. While Data Shapley offers a principled valuation framework, exact computation requires intractable retraining, and embedding surrogates like KNN-Shapley fail to separate high- from low-utility demonstrations ($\rho = 0.0003$ with generative utility). We introduce an inference-only valuation pipeline that computes the first-order term of an in-context cooperative game over semantic $k$-NN neighborhoods in $O(k)$ forward passes without parameter updates, correlating positively with multi-shot Monte-Carlo in-context Shapley on 100% of targets at 19x less compute. Projecting pairwise log-likelihood shifts onto a directed influence graph characterizes ea
    
[^231]: 在长程智能体任务中通过首次访问覆盖从全失败组中学习

    Learning from All-Failed Groups via First-Visit Coverage in Long-Horizon Agentic Tasks

    [https://arxiv.org/abs/2607.22724](https://arxiv.org/abs/2607.22724)

    提出CovGPO方法，通过在整组rollout全部失败时引入首次访问观测覆盖，为欠采样的状态改变动作提供学习信号，从而打破长程智能体任务中采样不平衡导致的自我延续信用陷阱。

    

    基于组的策略优化（Group-based policy optimization）已被越来越多地用于通过比较组内的轨迹或步骤，从稀疏的结果奖励中训练大语言模型（LLM）智能体。然而，我们发现，在困难的长程智能体任务上，这种比较方式存在采样不平衡问题：重复的或低效用的动作主导了策略的高概率区域，使得能够改变状态的有用动作被欠采样。这种不平衡导致产生大量全失败的rollout组，而这类组内部的相对优势无法为策略更新提供任何方向。由此，采样不平衡与全失败组共同坍缩为一个自我延续的信用陷阱：以失败为主的采样无法产生学习信号，从而使重复的低效用动作得以持续存在。为了打破这一陷阱，我们提出了覆盖条件化组策略优化（Coverage-conditioned Group Policy Optimization, CovGPO），它仅当所有采样……（摘要原文在此处截断）

    arXiv:2607.22724v2 Announce Type: replace-cross  Abstract: Group-based policy optimization has been increasingly used to train large language model (LLM) agents from sparse outcome rewards by comparing trajectories or steps within a group. However, on difficult long-horizon agentic tasks, we found this comparison suffers from a sampling imbalance: repeated or low-effect actions dominate the high-probability region of the policy, leaving useful state-changing actions under-sampled. This imbalance produces many all-failed rollout groups, where the relative advantages within such a group provide no direction for updating the policy. Consequently, the sampling imbalance and the all-failed groups collapse into a self-perpetuating credit trap: failure-dominated sampling yields no learning signal, allowing repeated low-effect actions to persist. To break this trap, we propose Coverage-conditioned Group Policy Optimization (CovGPO), which uses first-visit observation coverage only when all sam
    
[^232]: MIRROR：从另一视角学习以实现多模态推理

    MIRROR: Learning from the Other View for Multi-Modal Reasoning

    [https://arxiv.org/abs/2607.21552](https://arxiv.org/abs/2607.21552)

    本文提出MIRROR强化学习方法，通过在同一问题的不同模态视角中选取模型表现最佳者作为教师，实现跨视角自蒸馏，从而提升多模态推理能力。

    

    视觉-语言模型在处理同一个底层问题时，若问题以文本形式或视觉形式呈现，其表现可能截然不同。这种不一致性表明，不同的视角会暴露出不同的推理路径和失败模式，而标准的多模态后训练方法并未充分加以利用。为了研究和利用这一现象，我们构建了ODA-Data，一个高质量的成对多模态几何数据集，其中同一问题具有文本主导、图像主导以及图像+文本组合三种视角，并附带用于训练和评估视角相关推理行为的数据划分。随后，我们提出了模态感知互惠推理优化，这是一种强化学习方法，它从模型自身的模态依赖能力中提取跨视角自蒸馏信号。对于每个问题，MIRROR会在所有视角下评估模型，选择表现最佳的视角作为教师，并对其余视角进行训练……

    arXiv:2607.21552v2 Announce Type: replace  Abstract: Vision-language models (VLMs) can behave very differently depending on whether the same underlying problem is presented textually or visually. This inconsistency suggests that different views expose different reasoning paths and failure modes that standard multimodal post-training does not fully exploit. To study and exploit this phenomenon, we construct ODA-Data, a high-quality paired multimodal geometry dataset with text-dominant, image-dominant, and combined image+text views of the same problems, together with splits for training and evaluating view dependent reasoning behaviors. We then develop Modality-Informed Reciprocal Reasoning Optimization (MIRROR), a reinforcement learning approach that derives a cross-view self-distillation signal from the model's own modality-dependent capabilities. For each problem, MIRROR evaluates the model under all views, selects the best-performing view as a teacher, and trains the remaining views 
    
[^233]: 标签监督下医学基础模型的表征收敛程度更低

    Medical foundation models converge less under label supervision

    [https://arxiv.org/abs/2607.20274](https://arxiv.org/abs/2607.20274)

    该论文通过对75万余张医学图像、18个公开模型和101个自训练模型的大规模分析发现，医学基础模型——尤其是经过标签监督训练的模型——对相同图像的表征彼此差异巨大，其近邻重叠度甚至接近随机初始化的水平，因此不同基础模型的嵌入不可互换使用。

    

    诊断分类器和影像生物标志物通常拟合在医学基础模型的嵌入表示之上，当新版本的模型出现时，这些基础模型就会被替换。这种做法隐含了一个假设：不同模型对图像的表征是相似的。我们用来自5种成像模态、14个数据集的超过75万张图像、18个公开模型，以及101个在胸片和组织病理学图像上训练的模型（这些模型在预训练目标、标签类型、模型规模、随机种子、初始化、训练患者或训练配方上各不相同）检验了这一假设。模型间的一致性用相互k近邻重叠度来衡量，即一张图像的最近邻中有多少是两个模型所共有的。两个公开模型在胸片图像的10个最近邻中平均仅有0.091的重叠，而随机初始化的网络与它们也有0.036的重叠。在其他四种成像模态中，重叠度至多为0.300。在胸片上，一致性更多取决于源数据集和……（原文在此处截断）

    arXiv:2607.20274v3 Announce Type: replace-cross  Abstract: Diagnostic classifiers and imaging biomarkers are fitted on the embeddings of medical foundation models. These models are replaced as new versions appear. This practice assumes that different models represent images alike. We tested this assumption with more than 750,000 images from 14 datasets in 5 imaging modalities, 18 public models, and 101 models trained on chest radiographs and histopathology that differ in pretraining objective, label type, size, random seed, initialization, training patients, or training recipe. Agreement was measured as the mutual k-nearest-neighbor overlap, the share of an image's nearest neighbors common to two models. Two public models shared on average 0.091 of the 10 nearest neighbors of a chest radiograph. A randomly initialized network shared 0.036 with them. In the four other modalities, they shared at most 0.300. On chest radiographs, agreement depended more on the source dataset and the radio
    
[^234]: 基于强化学习的低压电网阻塞管理鲁棒性研究

    Robustness of Reinforcement Learning-Based Congestion Management in Low-Voltage Grids

    [https://arxiv.org/abs/2607.16004](https://arxiv.org/abs/2607.16004)

    本文提出一种将随机森林违规预分类器与actor-critic控制器相结合的解耦式强化学习阻塞管理方法，在真实低压电网中可将总违规幅度降低98.9%，并对测量噪声和电网参数失配表现出强鲁棒性。

    

    光伏发电的增加、电动汽车充电以及热泵需求对低压配电网的运行极限构成了挑战。这需要能够在稀疏可观测性、含噪测量和不完美电网模型条件下运行的矫正性削减方法。与以往针对部分可观测削减问题的端到端强化学习方法不同，本工作通过将随机森林违规预分类器与actor-critic控制器相结合，实现了阻塞检测与控制的解耦，并评估了该方法对测量噪声和电网参数失配的鲁棒性。该框架在一个真实的低压电网上进行测试，使用了具有低可观测性和低可控性的合成未来运行场景。在电网参数准确的情况下，控制器将总违规幅度降低了98.9%，并且在所测试的测量噪声设置下，该性能几乎保持不变。电网模型失配情形下的表现……（摘要原文在此处被截断）

    arXiv:2607.16004v2 Announce Type: replace-cross  Abstract: Increases in photovoltaic generation, charging of electric vehicles and heat-pump demand challenge operating limits in low-voltage distribution grids. This requires curative curtailment methods that can operate under sparse observability, noisy measurements, and imperfect grid models. Unlike prior end-to-end reinforcement-learning approaches for partially observable curtailment, this work decouples congestion detection and control by combining a random-forest violation pre-classifier with an actor-critic controller, and evaluates its robustness to measurement noise and grid-parameter mismatch. The framework is tested on a real low-voltage grid using synthetic future operating scenarios with low observability and controllability. With accurate grid parameters, the controller reduces total violation magnitude by 98.9%, and this performance remains nearly unchanged under the tested measurement-noise settings. Grid-model mismatch p
    
[^235]: 单词普查：44个语言模型中的答案选择趋同性

    The One-Word Census: Answer-Choice Conformity Across 44 Language Models

    [https://arxiv.org/abs/2607.12796](https://arxiv.org/abs/2607.12796)

    本研究通过96个开放式单词选择提示对105个语言模型进行了“答案趋同性”普查，发现模型的选择高度趋同——例如46%的模型在“选一个词”时都选了"serendipity"——且这种趋同并非由小型模型或个性化微调模型所致，反而轻度后训练和个性化微调的模型分歧最大。

    

    当一个语言模型必须从大量同样有效的选项中选择一个答案时，它会选择哪个答案，又有多少时候它与所有其他模型的选择一致？当被要求“选一个词”时，来自二十多个实验室的105个语言模型有46%的概率选择了“serendipity”这个词。我们通过96个单轮提示来测量这种趋同性以及每个模型在其中的贡献，每个提示指定一个包含许多有效单词答案的类别（如“说出一种树的名字。”），每个模型被询问八次，并以精确匹配进行评分，不使用嵌入模型，也不使用评判模型。模型的“答案选择惊异度”定义为其答案在所有其他模型汇总答案下的平均 -log2 概率。在96个类别中，有28个类别里单个答案占据了所有答案的至少80%。这种集中度并非源于小型模型或个性化微调模型：87个主要实验室的模型的集中程度至少与整个领域相当。轻度后训练和个性化微调的模型的分歧最大（原文在此处截断）。

    arXiv:2607.12796v3 Announce Type: replace-cross  Abstract: When a language model must choose one answer from a large space of equally valid options, which answer does it choose, and how often is it the answer every other model chooses? Asked to "pick a word," 105 language models from more than twenty labs chose serendipity 46% of the time. We measure this convergence, and each model's share in it, with 96 single-turn prompts that each name a category with many valid one-word answers ("Name a tree."), asked eight times per model and scored by exact match, with no embeddings and no judge. A model's answer-choice surprisal is the average -log2 probability of its answers under the pooled answers of all other models. In 28 of 96 categories a single answer takes at least 80% of all answers. The concentration does not depend on small or persona-tuned models: the 87 major-lab models are at least as concentrated as the full field. Lightly post-trained and persona-tuned models are the most diver
    
[^236]: Who&When Pro：大语言模型真的能为AI智能体进行失败归因吗？

    Who&When Pro: Can LLMs Really Attribute Failures in AI Agents?

    [https://arxiv.org/abs/2607.09996](https://arxiv.org/abs/2607.09996)

    提出大规模基准Who&When Pro，通过严格控制的失败注入流水线构建了12,326条带黄金标签的失败轨迹，系统评估并揭示了大语言模型在AI智能体失败归因任务中的表现规律与模式。

    

    自动失败归因利用大语言模型来识别智能体系统在哪里失败以及为什么失败。随着智能体能力不断提升，其失败变得更加隐蔽，这使得自动归因变得越来越重要。我们提出了Who&When Pro，一个面向智能体系统自动失败归因的大规模基准。通过一个严格控制的流水线——仅在精确重放成功前缀之后注入失败——我们构建了12,326条带有黄金标签的失败轨迹，涵盖3种模态和26个基准，覆盖多种场景。除了基准测试之外，我们还进行了广泛的实验与分析，揭示了模型在跨模态、协议和模型家族进行失败归因时的系统性模式，并为未来的自动失败归因系统提供了实证指导。

    arXiv:2607.09996v2 Announce Type: replace  Abstract: Automated failure attribution uses LLMs to identify where and why agentic systems fail. As agents become more capable, their failures become subtler, making automated attribution increasingly important. We introduce Who&When Pro, a large-scale benchmark for automated failure attribution in agentic systems. Using a strictly controlled pipeline that injects a failure only after exactly replaying a successful prefix, we construct 12,326 failed trajectories with golden labels across 3 modalities and 26 benchmarks covering various scenarios. Beyond benchmarking, we conduct extensive experiments and analyses, revealing systematic patterns in how models attribute failures across modalities, protocols, and model families, and providing empirical guidance for future automated failure attribution systems.
    
[^237]: 测量多步LLM智能体中由执行框架引起的信念分歧

    Measuring Harness-Induced Belief Divergence in Multi-Step LLM Agents

    [https://arxiv.org/abs/2607.04528](https://arxiv.org/abs/2607.04528)

    该论文提出信念推演诊断方法，首次揭示执行框架会在任务、环境和基础LLM完全固定的情况下显著塑造多步LLM智能体对环境的信念，导致不同框架间出现从终端输出难以察觉的显著信念分歧。

    

    软件智能体在现代编程工作流程中正变得越来越重要，它们通过与代码仓库的多步交互来解决复杂任务。这些系统的核心组件是执行框架，它决定了底层LLM如何观察环境、对环境执行操作以及验证其行为。我们证明了执行框架反过来会塑造智能体对环境不断演变的信念，即使任务、环境和基础LLM保持不变。为了揭示这一效应，我们引入了一种信念推演诊断方法，用于追踪智能体的预期世界状态和未来决策在不同执行框架下如何演变。我们进一步将接口引入的即时分歧与在更长时间范围内发展出的分歧区分开来。在包括HIBench-Code、SWE-bench Verified和Terminal-Bench在内的受控编程任务上，我们发现不同执行框架之间存在显著的信念分歧，而这种分歧从终端输出往往难以察觉。

    arXiv:2607.04528v2 Announce Type: replace  Abstract: Software agents are becoming increasingly important in modern coding workflows, where they solve complex tasks through multi-step interaction with code repositories. A central component of these systems is the execution harness, which governs how the underlying LLM observes, acts on, and verifies its environment. We show that the harness can in turn shape the agent's evolving beliefs about the environment, even when the task, environment, and base LLM remain fixed. To reveal this effect, we introduce a belief-rollout diagnostic that traces how an agent's anticipated world state and future decisions evolve under alternative harnesses. We further separate immediate divergence introduced by the interface from divergence that develops over longer horizons. Across controlled coding tasks include HIBench-Code, SWE-bench Verified and Terminal-Bench, we find substantial cross-harness belief divergence that is often invisible from terminal su
    
[^238]: 智能体步骤价值：审计黑盒智能体轨迹中评估通道敏感性

    Agent Step Value: Auditing Evaluation-Channel Sensitivity in Black-Box Agent Traces

    [https://arxiv.org/abs/2607.04419](https://arxiv.org/abs/2607.04419)

    本文提出智能体步骤价值（ASV）指标，发现评估器衍生的步骤奖励在跨评估通道使用时高度不稳定（48.3%的比较中增益符号不一致），并将负向分量定位于生成视图通道。

    

    arXiv:2607.04419v5 公告类型：替换 摘要：在不同评估通道之间复用由评估器衍生的步骤奖励会引出一个测量问题：所赋予的增益方向是否保持稳定？我们将智能体步骤价值（ASV）定义为按通道索引的目标裕度前后增益，并通过匹配的2x2实验设计识别出状态与通道之间的交互效应。在冻结的PubMed问答转换数据上，跨通道的增益符号在48.3%的比较中不一致，而同通道重试的不一致率仅为0.3%。转换交互效应在符号上近乎平衡（中位数+0.014；10%截尾均值-0.033），但在经验目标混合分布下，不对称的负值幅度产生了正的直接均值和负的生成视图均值。等目标加权能保留负的平均交互效应，却无法维持这种严格的均值反转。匹配的检索设计将负分量定位到生成视图坐标，并在改变读取方式等因素的三次符号检验中保持了该分量的方向。

    arXiv:2607.04419v5 Announce Type: replace  Abstract: Reusing evaluator-derived step rewards across evaluation channels raises a measurement question: does the assigned direction remain stable? We define Agent Step Value (ASV) as a channel-indexed before/after target-margin gain and identify the state-by-channel interaction from a matched 2x2 design. Across frozen PubMed question-answering transitions, cross-channel gain signs disagree on 48.3% of comparisons, compared with 0.3% for same-channel retries. Transition interactions are nearly balanced in sign (median +0.014; 10% trimmed mean -0.033), but asymmetric negative magnitudes produce a positive direct mean and a negative generated-view mean under the empirical target mixture. Equal-target weighting preserves a negative mean interaction but not this strict mean reversal. Matched retrieval designs localize the negative component to the generated-view coordinate and retain its direction across three sign checks that vary readout and e
    
[^239]: 通过对抗蒸馏提升认证鲁棒性

    Improving Certified Robustness via Adversarial Distillation

    [https://arxiv.org/abs/2606.31653](https://arxiv.org/abs/2606.31653)

    本文提出了结合对抗蒸馏的认证训练方法AD-CERT，通过将对抗训练目标与IBP松弛界相结合，在标准准确率与认证鲁棒性之间取得更好的权衡。

    

    认证训练旨在产生预测结果能够针对对抗扰动进行形式化验证的模型，通常通过在允许的扰动集上优化最坏情况损失的上界来实现。对于神经网络而言，纯粹基于紧致松弛界的认证训练方法所产生的网络虽然易于认证，但会牺牲标准准确率。相反，对抗训练通常能带来更强的经验鲁棒性和标准准确率，但由此产生的模型通常难以用神经网络验证器进行认证。最近有研究表明，通过将对抗训练目标与基于区间界传播（IBP）的松散过近似相结合，可以实现更好的标准准确率与认证准确率之间的权衡，有效地在最坏情况损失的下界与上界之间进行插值。在此基础上，我们提出了AD-CERT，一种认证训练方法（摘要在此处截断）

    arXiv:2606.31653v2 Announce Type: replace-cross  Abstract: Certified training aims to produce models whose predictions can be formally verified against adversarial perturbations, typically by optimising upper bounds on the worst-case loss over an allowed perturbation set. For neural networks, certified training methods based purely on tight relaxation bounds produce networks that are amenable to certification, but sacrifice standard accuracy. Conversely, adversarial training often yields stronger empirical robustness and standard accuracy, but the resulting models are generally difficult to certify with neural network verifiers. Recently, the literature has shown that better standard-certified accuracy trade-offs can be achieved by combining adversarial training objectives with loose over-approximations based on Interval Bound Propagation (IBP), effectively interpolating between lower and upper bounds of the worst-case loss. Building on this, we introduce AD-CERT, a certified training 
    
[^240]: 分层实验智能体

    Hierarchical Experimentalist Agents

    [https://arxiv.org/abs/2606.29315](https://arxiv.org/abs/2606.29315)

    提出分层实验智能体（HExA），一种以实验为中心的上下文内自我改进框架，使智能体能够通过主动实验收集查询相关证据、发现未知环境的一般规律，并将交互经验转化为可复用技能。

    

    arXiv:2606.29315v2 公告类型：替换 摘要：大语言模型（LLM）越来越多地被用于在现实世界中执行任务和辅助人类决策，然而大多数系统依赖于通过模仿获得的参数化知识，并可选择地通过后训练、检索或在固定数据上的搜索来加以改进。这种范式在 novel 领域或需要先验知识之外信息的复杂查询上表现不佳；例如，仅知道物理定律本身并不能保证在复杂物理系统中进行可靠的推理或长时程行动。我们提出，智能体需要主动实验的能力：即收集针对特定查询的目标性证据、发现未知环境的一般原理，并将交互经验转化为可复用技能。我们提出了分层实验智能体（HExA），这是一个以实验为中心的上下文内自我改进框架，它（1）迭代地设计和完善与查询相关的实验，（2）增量……

    arXiv:2606.29315v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to act in the real world and augment human decision-making, yet most systems rely on parametric knowledge acquired through imitation, optionally improved by post-training, retrieval, or search over fixed data. This paradigm struggles in novel domains or on complex queries that require information unavailable from prior knowledge alone; knowing the laws of physics, for example, does not by itself enable reliable reasoning or long-horizon actions in a complex physical system. We argue that agents need active experimentation: the ability to gather targeted query-specific evidence, discover general principles of unseen environments, and convert interaction experience into reusable skills. We introduce Hierarchical Experimentalist Agents (HExA), an in-context, experiment-centric self-improvement framework that (1) iteratively designs and refines query-relevant experiments, (2) incremental
    
[^241]: 规格增长引擎：面向AI辅助软件开发的规格锚定、代码耦合与漂移强制架构

    The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development

    [https://arxiv.org/abs/2606.27045](https://arxiv.org/abs/2606.27045)

    提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。

    

    arXiv:2606.27045v1 公告类型：交叉  摘要：AI编码代理极大地加速了实现速度，但引入了两种现有规格驱动方法未能完全解决的结构性失效模式：(1) 上下文爆炸——代理必须同时推理整个代码仓库，随着上下文窗口填满，输出质量下降；以及(2) 静默规格-代码漂移——代码演化，规格未更新，差异变得不可见，直到修复代价高昂才被发现。  我们提出了规格增长引擎，这是一个轻量级框架，通过以下方式解决这两种失效模式：一个机器可读的规格图，其节点携带明确的合同/设计分离；一个主干上下文组装器，将代理上下文限定到所有权路径；一个垂直切片增长协议，强制执行最难优先排序；以及一个漂移门，使规格-代码差异成为阻塞合并条件。  该设计综合了成熟的软件工程原则（Parnas信息隐藏）。

    arXiv:2606.27045v1 Announce Type: cross  Abstract: AI coding agents dramatically accelerate implementation speed but introduce two structural failure modes that existing spec-driven approaches do not fully solve: (1) context explosion -- the agent must reason over an entire repository at once, degrading output quality as the context window fills; and (2) silent spec-code drift -- code evolves, the specification does not, and the divergence becomes invisible until it is costly to repair.   We present the Spec Growth Engine, a lightweight framework that addresses both failure modes through a machine-readable spec graph whose nodes carry explicit contract/design separation, a Spine context assembler that scopes agent context to an ownership path, a vertical-slice growth protocol that enforces hardest-first ordering, and a drift gate that makes spec-code divergence a blocking merge condition.   The design synthesises well-established software engineering principles (Parnas information hidi
    
[^242]: 已被抹除，但并未消失：输出层面的遗忘并非真正的遗忘

    Erased, but Not Gone: Output Forgetting Is Not True Forgetting

    [https://arxiv.org/abs/2606.25001](https://arxiv.org/abs/2606.25001)

    该论文提出以重训练模型群体作为参考分布的“重训练一致的表示遗忘”评估范式，揭示现有机器遗忘方法虽在输出层面看似遗忘成功，但在表示空间中仍与理想遗忘存在结构性偏差，即输出遗忘并不等于真正的遗忘。

    

    机器遗忘（MU）通常通过输出层面的遗忘效果来评判，例如遗忘集上的低准确率或成员推断攻击效果的减弱。但如果输出层面的成功可能与表示空间中重训练不一致的残余共存，那么当前的评估究竟在认证何种意义上的“遗忘”？我们通过“重训练一致的表示遗忘”来研究这一问题，使用一组重训练模型（即从头开始训练且不包含待遗忘数据的模型）的总体作为理想遗忘的可操作参考分布。在多种遗忘方法、数据集和模型上，我们的理论分析与实证结果表明，标准的输出层面评估会系统性地高估遗忘的成功程度。在这种更严格的视角下，当前方法往往在输出层表现得像已遗忘，但在表示层面却与重训练结果存在结构性的不匹配。它们在遗忘集上仅部分对齐于重训练……

    arXiv:2606.25001v2 Announce Type: replace-cross  Abstract: Machine unlearning (MU) is commonly judged by output forgetting, such as low forget-set accuracy or reduced membership inference. But if output-level success can coexist with retraining-inconsistent residuals in representation space, what kind of forgetting are current evaluations actually certifying? We study this question through retraining-consistent representation forgetting, using a population of retrained models (i.e., trained from scratch without the forget data) as an operational reference distribution for ideal forgetting. Across multiple unlearning methods, datasets, and models, our theoretical analysis and empirical results show that standard output-level evaluation can systematically overestimate the success of unlearning. Under this stronger lens, current methods often appear forgotten at the output layer while exhibiting a structured mismatch relative to retraining. They partially align with retraining on forget s
    
[^243]: RoboMME-Interference：干扰下的机器人记忆基准测试

    RoboMME-Interference: Benchmarking Robot Memory Under Interference

    [https://arxiv.org/abs/2606.22338](https://arxiv.org/abs/2606.22338)

    本文提出了RoboMME-Interference，一个跨会话机器人记忆基准，用于评估长上下文和干扰下VLA模型的记忆性能，并发现现有感知记忆变体在干扰下表现有限。

    

    arXiv:2606.22338v3 公告类型：替换-交叉 摘要：部署在现实环境中的机器人会在其部署期间跨多个会话和任务积累经验。机器人的任务往往需要它记住多个会话之前的信息，这使得长上下文机器人记忆对于实际部署至关重要。然而，目前大多数机器人记忆基准测试基于单次情节或短上下文。为了衡量当前机器人记忆系统在更长会话和更多干扰下的表现，我们引入了RoboMME-Interference，这是一个基于RoboMME（Dai等人，2026）构建的跨会话基准测试。对于每个查询情节，我们使用查询相关的先前演示构建一个会话历史，随后加入受控数量的无关会话，并将其作为记忆提供给视觉-语言-动作模型（VLA），然后测量准确性。通过在该基准测试上运行RoboMME发布的未经修改的记忆增强π0.5变体，我们发现感知记忆变体虽然有所改进，但成功……（摘要截断）

    arXiv:2606.22338v3 Announce Type: replace-cross  Abstract: Robots deployed in realistic settings will accumulate experience across many sessions and tasks over their deployment. The robot's tasks may often require it to remember information from multiple sessions ago, making long-context robot memory important for real-world deployments. However, most robot-memory benchmarks today are based on single episodes or a short context. To measure how current robot memory systems perform on longer sessions with more distractions, we introduce RoboMME-Interference, a cross-session benchmark built on RoboMME (Dai et al., 2026). For each query episode, we construct a session history using the query's relevant prior demonstration followed by a controlled number of unrelated sessions, which we provide to the VLA as memory and measure accuracy. Running RoboMME's released memory-augmented $\pi_{0.5}$ variants unmodified through this benchmark, we find that while perceptual memory variants improve suc
    
[^244]: 视觉-语言-动作模型中陈述性知识与程序性知识的解耦

    Decoupling the Declarative from the Procedural in Vision-Language-Action Models

    [https://arxiv.org/abs/2606.21496](https://arxiv.org/abs/2606.21496)

    该论文指出现有视觉-语言-动作模型无法将陈述性知识（概念与实体语义）与程序性知识（操作方法）解耦，导致其对空间、语义和任务变化十分脆弱、难以泛化到训练物体之外，这是通用机器人技能迁移的根本瓶颈。

    

    在现实世界中部署通用机器人智能体需要可迁移的技能。具体而言，通过物体特定演示训练来克隆行为的策略必须能够超越该物体进行泛化，否则数据收集需求将变得难以承受。近年来，对预训练的数十亿参数视觉-语言模型（VLM）进行微调——首先在大规模机器人数据集上进行，然后在较少的场景特定演示上进行——已成为设计视觉-语言-动作（VLA）模型的主流范式。虽然这些策略在分布内实现了最先进的操作性能，但它们对微小的空间、语义和任务变化仍然十分脆弱。在这项工作中，我们针对当前模型无法将参数中编码的陈述性知识（即概念和实体语义）与程序性知识（即如何执行某项操作）解耦的问题进行研究，这是……的一个根本性瓶颈。

    arXiv:2606.21496v2 Announce Type: replace-cross  Abstract: Deploying generalist robotic agents in the real world requires transferable skills. Specifically, a policy trained to clone a behavior from object-specific demonstrations must generalize beyond that object, otherwise data collection requirements become intractable. Recently, fine-tuning of pre-trained billion-parameter Vision-Language Models (VLMs), initially on large-scale robot datasets and then on fewer scenario-specific demonstrations, has emerged as the predominant paradigm for designing Vision-Language-Action (VLA) models. While these policies achieve state-of-the-art manipulation performance in-distribution, they remain brittle to minor spatial, semantic, and task variations. In this work, we address the inability of current models to decouple the declarative (i.e., concepts and entity semantics) from the procedural knowledge (i.e., how to do something) encoded in their parameters, which is a fundamental bottleneck for z
    
[^245]: TelcoAgent：具有基于3GPP可解释性的可扩展5G多KPM预测

    TelcoAgent: A Scalable 5G Multi-KPM Forecasting With 3GPP-Grounded Explainability

    [https://arxiv.org/abs/2606.19821](https://arxiv.org/abs/2606.19821)

    提出TelcoAgent框架，通过自动构建3GPP知识图谱、零样本时间序列基础模型预测和领域可解释推理三个组件，实现了无需站点特定训练的准确、可扩展且可解释的5G多KPM预测。

    

    关键性能测量（KPM）预测对于5G及下一代电信网络的主动式网络管理至关重要。然而，现有的机器学习方法在可扩展性和可解释性方面存在显著局限，限制了其在实际部署中的有效性。我们提出了TelcoAgent，这是一个基于基础模型的框架，无需针对特定站点进行训练，即可对跨多个网络小区的多个KPM实现准确、可扩展且可解释的预测。具体而言，该框架包含三个关键组件：(i) 一个自动化三智能体流水线，可直接从3GPP规范文档构建第三代合作伙伴计划（3GPP）知识图谱；(ii) 一个基于时间序列基础模型（TSFM）的可扩展预测流水线，可提供准确的零样本预测；(iii) 一个推理与解释流水线，可提供可操作的、基于领域知识的解释。

    arXiv:2606.19821v2 Announce Type: replace  Abstract: Key Performance Measurement (KPM) forecasting is essential for proactive network management of 5G and next-generation telecom networks. However, existing machine learning (ML) approaches face significant limitations in scalability and explainability, restricting their effectiveness in real-world deployments. We propose TelcoAgent, a foundation model-based framework that enables accurate, scalable, and explainable forecasting of multiple KPMs across diverse network cells without the need for site-specific training. Specifically, the framework comprises three key components: (i) an automated three-agent pipeline that constructs a 3rd Generation Partnership Project (3GPP) knowledge graph directly from specification documents, (ii) a scalable, time-series foundation model (TSFM)-based prediction pipeline to deliver accurate, zero-shot forecasting, and finally (iii) a reasoning and explanation pipeline that provides actionable, domain-gro
    
[^246]: 构造性双稳态：以墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制

    Bistable by Construction: Wall-Clock-Calibrated State Monitors Have No Moment-Detection Regime at Agent Cadence

    [https://arxiv.org/abs/2606.19386](https://arxiv.org/abs/2606.19386)

    本文勘误并重新审视了先前报告的“状态饱和陷阱”（实为动作间 dt=0 导致指数衰减从未运行的纯累加器结果），揭示出以采样时间校准与以墙钟时间校准的监测器动态在固定速率流上等价、但在动作间隔差异巨大的智能体流上根本不同，因此墙钟时间校准的状态监测器在智能体节奏下不存在瞬时检测机制。

    

    面向自主智能体的运行时监测器通常对累积的内部状态设定阈值——可以是行为基线、漂移统计量，或在我们先前工作中的建模情感状态。我们曾报告过一种“状态饱和陷阱”：在连续情感引擎之上基于状态阈值的触发机制，会在 SWE-bench 调试智能体上变成近乎持续不断的报警（Modgil 2026）。发布后的审计发现，该引擎在动作之间接收到的 dt=0，因此其指数衰减从未运行：先前发表的陷阱实际上是一个纯累加器的结果。我们更正了记录（勘误，v2），并将这一缺陷视为一次实验。它所揭示的关键变量是：监测器的动态是以采样时间（按每次观测，如 CUSUM）校准，还是以墙钟时间（以秒为单位的半衰期，如情感模型和 EMA 基线）校准。在固定速率的数据流上二者重合；而在动作间隔时间相差数个数量级的智能体数据流上，二者并不重合。一项预先注册的……

    arXiv:2606.19386v2 Announce Type: replace-cross  Abstract: Runtime monitors for autonomous agents commonly threshold an accumulated internal state - a behavioural baseline, a drift statistic, or, in our prior work, a modelled affective state. We previously reported a State Saturation Trap: threshold-on-state triggers over a continuous affect engine become near-constant alarms on SWE-bench debugging agents (Modgil 2026). A post-release audit found the engine received dt=0 between actions, so its exponential decay never operated: the published trap is a pure-accumulator result. We correct the record (erratum, v2) and treat the flaw as an experiment. The key variable it exposes is whether a monitor's dynamics are calibrated in sample time (per observation, as in CUSUM) or wall-clock time (half-lives in seconds, as in affect models and EMA baselines). On fixed-rate streams these coincide; on agent streams, where inter-action time varies by orders of magnitude, they do not. A pre-registered
    
[^247]: 面向大语言模型解释的可扩展电路学习方法

    Scalable Circuit Learning for Interpreting Large Language Models

    [https://arxiv.org/abs/2606.16939](https://arxiv.org/abs/2606.16939)

    提出基于稀疏线性回归的CircuitLasso方法，以极低的计算成本学习到与最先进干预方法结构准确性相当的电路，并能高效揭示人类可理解的SAE语义特征在大语言模型中的传播及其对预测的影响。

    

    机制可解释性研究的一个重要方向是在大语言模型的组件之上学习稀疏电路，以揭示这些组件如何共同产生模型行为。然而，原始神经元具有多义性，使得学习到的电路难以解释。稀疏自编码器（SAE）特征缓解了这一问题，但其高维度使得现有的基于干预的电路学习方法在计算上难以承受。我们提出了CircuitLasso，一种基于稀疏线性回归的可扩展电路学习方法。在基准数据上，CircuitLasso恢复的电路在结构准确性上与最先进的基于干预的方法相当，而计算成本仅为后者的一小部分。在可解释性方面，CircuitLasso能够高效地揭示SAE特征之间的关系，展示人类可理解的语义特征如何在模型中传播并影响其预测。最后，我们验证了所学习电路的实用性。

    arXiv:2606.16939v2 Announce Type: replace-cross  Abstract: A prominent research direction in mechanistic interpretability is learning sparse circuits over LLM components to reveal how they jointly produce model behavior. However, raw neurons are polysemantic, making learned circuits hard to interpret. Sparse autoencoder (SAE) features alleviate this, but their high dimensionality makes existing intervention-based circuit learning methods computationally prohibitive. We propose CircuitLasso, a scalable circuit learning approach based on sparse linear regression. CircuitLasso recovers circuits whose structural accuracy matches that of state-of-the-art intervention-based methods on the benchmark data, at a fraction of the computational cost. For interpretability, CircuitLasso efficiently uncovers relationships among SAE features, showing how human-interpretable semantic features propagate through the model and influence its predictions. Finally, we validate the utility of our learned circ
    
[^248]: PACT：面向多轮工具使用智能体的特权轨迹协同训练框架

    PACT: Privileged Trace Co-Training for Multi-Turn Tool-Use Agents

    [https://arxiv.org/abs/2606.16215](https://arxiv.org/abs/2606.16215)

    PACT框架提出将专家轨迹仅作为训练时的优化信号而非推理提示，通过轨迹条件化强化学习代理目标和带退火过程监督的组件感知SFT损失两个互补信号进行协同训练，同时保持推理时仅使用提示词生成，从而兼顾强化学习的推理一致性与监督微调的密集过程监督优势。

    

    多轮工具使用智能体需要在多轮交互过程中进行推理、调用工具并适应观测结果。对此类智能体进行后训练极具挑战性：强化学习虽然与仅提示词的推理设置相匹配，但常常面临奖励稀疏和信用分配能力弱的问题；而基于专家轨迹的监督微调虽然能提供密集的过程监督，却可能过度约束模型，使其拘泥于固定轨迹。为解决这一问题，我们提出了PACT——一个面向多轮工具使用智能体的特权轨迹协同训练框架。其核心思想是仅将专家轨迹用作训练时的优化信号，而非推理时的提示。PACT在生成rollout时保持仅提示词模式，随后通过两个互补的信号利用专家轨迹来引导优化：一是轨迹条件化的强化学习代理目标，它在专家轨迹的上下文中评估仅提示词的rollout；二是组件感知的监督微调损失，它提供退火式的过程监督

    arXiv:2606.16215v2 Announce Type: replace-cross  Abstract: Multi-turn tool-use agents must reason, call tools, and adapt to observations across several interaction turns. Post-training such agents is challenging, as reinforcement learning often suffers from sparse rewards and weak credit assignment despite matching the prompt-only inference setting, while supervised fine-tuning on expert traces provides dense process supervision but can over-constrain the model to fixed trajectories. To tackle this, we propose PACT, a Privileged trAce Co-Training framework for multi-turn tool-use agents. The key idea is to use expert traces only as training-time optimization signals rather than rollout-time hints. PACT keeps rollout generation prompt-only, then uses expert traces to guide optimization through two complementary signals: a trace-conditioned RL surrogate that evaluates prompt-only rollouts under expert-trace context, and a component-aware SFT loss that provides annealed process supervisio
    
[^249]: 让大语言模型互相评判：面向医学问答的多智能体同行评审推理

    Let Large Language Models Judge Each Other: Multi-Agent Peer-Reviewed Reasoning for Medical Question Answering

    [https://arxiv.org/abs/2606.15419](https://arxiv.org/abs/2606.15419)

    该论文提出多智能体同行评审推理方法，让多个大语言模型在医学问答中互相评审彼此的思维链推理并择优生成最终答案，显著提升了准确性、可解释性和鲁棒性。

    

    目标：提升大语言模型（LLMs）在医学问答（MedQA）任务中的准确性、可解释性和鲁棒性。方法：我们设计了一种多智能体同行评审推理方法，其中多个大语言模型智能体独立生成包含候选答案的思维链推理，随后充当同行评审，对彼此的推理进行事实正确性和逻辑严密性的评估，并选择评分最高的推理链来产生最终答案。实验使用了五个最先进的大语言模型在三个基准数据集上进行，并与单模型思维链推理以及基于思维链的多数投票方法进行了性能对比。结果：同行评审推理方法在所有实验中均一致优于两种基线方法，最佳模型组合的平均准确率达到0.820...

    arXiv:2606.15419v2 Announce Type: replace-cross  Abstract: Objective: To enhance the accuracy, interpretability, and robustness of large language models (LLMs) in medical question answering (MedQA).   Method: We designed a multi-agent peer-reviewed reasoning method in which multiple LLM agents independently generate chain-of-thought reasoning with candidate answers, then act as peer reviewers to evaluate each other's reasoning for factual correctness and logical soundness. The highest-rated reasoning chain is selected to produce the final answer. Experiments were conducted with five state-of-the-art LLMs (Llama-3.1-8B, Qwen2.5-7B, Phi-4, DeepSeek-LLM-7B, GPT-oss-20B) on three benchmark datasets: HeadQA, MedQA-USMLE, and PubMedQA. Performance was compared against single-model chain-of-thought reasoning and chain-of-thought-based majority voting.   Results: Peer-reviewed reasoning consistently outperformed both baselines. The best model combination achieved an average accuracy of 0.820 a
    
[^250]: 经验造就技能：通过自我演化的技能记忆实现可泛化的医疗智能体推理

    Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory

    [https://arxiv.org/abs/2606.09365](https://arxiv.org/abs/2606.09365)

    提出SkeMex框架，将医疗智能体的交互轨迹提炼为结构化的技能记忆，在不更新模型权重的情况下实现自我演化，从而支持可泛化的长程临床推理。

    

    医疗智能体系统正日益被期望支持交互式临床决策，而不仅仅是静态的问答。在这种场景下，有效的智能体必须能够在不断演变的病例中复用先前的经验，然而现有的记忆机制往往保留原始的历史轨迹，这些轨迹冗余、嘈杂且难以管理。更重要的是，它们很少区分哪些记忆对未来的推理真正有用。这限制了智能体为长程临床推理积累紧凑且可靠经验的能力。为弥补这一差距，我们提出了SkeMex，一个部署后的自我演化框架，它通过基于技能的记忆来改进医疗智能体，而无需更新模型权重。SkeMex将信息丰富的交互轨迹提炼为编码可复用程序性知识的结构化技能，并将它们组织成一个涵盖通用、任务特定和动作层面经验的多分支知识库。

    arXiv:2606.09365v4 Announce Type: replace  Abstract: Medical agent systems are increasingly expected to support interactive clinical decision making rather than only static question answering. In such settings, effective agents must reuse prior experience across evolving cases, yet existing memory mechanisms often retain raw historical traces that are redundant, noisy, and difficult to govern. More importantly, they rarely distinguish which memories are truly useful for future reasoning. This limits their ability to accumulate compact and reliable experience for long-horizon clinical reasoning. To close this gap, we propose SkeMex, a post-deployment self-evolution framework that improves medical agents through a skill-based memory without updating model weights. SkeMex distills informative interaction trajectories into structured skills that encode reusable procedural knowledge, and organizes them into a multi-branch repository spanning general, task-specific, and action-level experien
    
[^251]: 更多的智能体有帮助吗？评估LLM多智能体工作流的准确性与成本

    Do More Agents Help? Evaluating Accuracy and Cost in LLM Multi-Agent Workflows

    [https://arxiv.org/abs/2606.05670](https://arxiv.org/abs/2606.05670)

    该研究提出共享评估框架BenchAgent，在十个基准上比较六种多智能体工作流与单智能体方案的准确率与token成本，发现大多数多智能体配置以更高的成本换来更差的总体表现，仅EvoAgent小幅胜出。

    

    判断是否应该用多智能体替代单智能体控制器，需要同时提供任务性能和推理开销两方面的证据。我们提出了BenchAgent，一个共享评估框架，并在十个推理、编程和工具使用基准上，将六种多智能体工作流配置与其单智能体参照BenchAgent Core进行比较。所有配置均使用GPT-4.1以及通用的工具与评分接口；Core执行共享相同的步骤限制。我们在固定任务集上进行了三次运行，测量准确率并记录token使用量，每次运行中每个配置包含3,302个实例。EvoAgent达到79.70%的平均汇总准确率，而Core为76.65%，且EvoAgent在全部三次运行中都超过了Core；其余五种多智能体系统落后1.71至8.76个百分点。LLM-Debate在HumanEval上提升9.67分，但汇总准确率下降1.96分，同时消耗的token是Core的1.39倍。ChatEval在IFEval上提升22.25分，但汇总准确率落后于Core。

    arXiv:2606.05670v2 Announce Type: replace  Abstract: Choosing whether to replace a single-agent controller requires evidence on both task performance and inference expenditure. We present BenchAgent, a shared evaluation framework, and compare six multi-agent workflow configurations against its single-agent reference, BenchAgent Core, across ten reasoning, coding, and tool-use benchmarks. All configurations use GPT-4.1 and common tool and scoring interfaces; Core executions share the same step limits. We measure accuracy and recorded token use over three runs on fixed task sets, with 3,302 instances per configuration per run. EvoAgent reaches 79.70% mean pooled accuracy versus 76.65% for Core and exceeds Core in all three runs; the other five MAS trail by 1.71-8.76 percentage points. LLM-Debate gains 9.67 points on HumanEval but loses 1.96 points in pooled accuracy while consuming 1.39 times the tokens of Core. ChatEval gains 22.25 points on IFEval but trails Core in pooled accuracy wit
    
[^252]: 智能体的最后考试

    Agents' Last Exam

    [https://arxiv.org/abs/2606.05405](https://arxiv.org/abs/2606.05405)

    本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。

    

    arXiv:2606.05405v3 公告类型：替换 摘要：近年来的人工智能系统已在各类基准测试中取得了优异的成绩，但这些成果并未转化为在众多专业领域中的经济价值意义上的实际部署。我们认为，这一差距在很大程度上是一个评估问题：广泛使用的基准测试缺乏对真实的、具有经济价值的工作流程的持续性性能测量。本文提出了 Agents' Last Exam（ALE），这是一个旨在评估AI智能体在长周期、具有经济价值、结果可验证的真实世界任务上表现的基准。该基准与250多位行业专家合作开发，参考O*NET / SOC 2018（美国联邦职业分类体系）定义了非实体行业，并围绕一个任务分类体系组织，该体系包含55个子领域、归为13个行业集群，涵盖1千多项任务。当前结果显示，最难的任务层级仍远未达到饱和：在主流的运行框架和底层模型配置下，平均……

    arXiv:2606.05405v3 Announce Type: replace  Abstract: Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents' Last Exam (ALE), a benchmark designed to evaluate AI agents on long horizon, economically valuable, real world tasks with verifiable outcomes. Developed in collaboration with 250+ industry experts, ALE covers non-physical industries defined with reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). It is organized around a task taxonomy with 55 sub fields grouped into 13 industry clusters covering 1K+ tasks. Current results show that the hardest tier remains far from saturated: across mainstream harness and backbone configurations, the avera
    
[^253]: 物理学的无形之手：当视频扩散模型知道的比它们展示的更多时

    The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show

    [https://arxiv.org/abs/2606.05328](https://arxiv.org/abs/2606.05328)

    该论文通过逆转扩散采样过程恢复模型内部轨迹，首次证明视频扩散模型内部确实线性编码了物理合理性信息，其可从扩散Transformer状态中以约81.27%的准确率解码出来，超越了专用表征学习基线。

    

    现代视频扩散模型能够生成越来越逼真且时间上连贯的视频，这促使它们被用作候选的世界模拟器。然而，这些模型究竟是在内部编码了物理结构，还是仅仅复现了训练期间见过的运动模式，目前仍不清楚。我们通过在对应于具有已知物理合理性真实视频的潜空间轨迹上探测视频扩散模型来研究这一问题。为了获得这些轨迹，我们通过将学习到的速度场从干净的视频潜表示反向积分至噪声，近似地逆转确定性采样过程，从而获得模型的中间状态和注意力图。利用这些恢复的轨迹，我们证明物理合理性可以从扩散Transformer的状态中被线性解码出来，在IntPhys和InfLevel数据集上达到约81.27%的平均准确率，并优于专用的表征学习基线方法（如……

    arXiv:2606.05328v2 Announce Type: replace-cross  Abstract: Modern video diffusion models generate increasingly realistic and temporally coherent videos, motivating their use as candidate world simulators. Yet it remains unclear whether these models internally encode physical structure, or merely reproduce motion patterns seen during training. We study this question by probing video diffusion models along latent trajectories corresponding to real videos with known physical plausibility. To obtain such trajectories, we approximately invert the deterministic sampling process by integrating the learned velocity field backward from a clean video latent to noise, giving access to the model's intermediate states and attention maps. Using these recovered trajectories, we show that physical plausibility is linearly decodable from diffusion transformer states across IntPhys and InfLevel, reaching around 81.27% average accuracy and outperforming dedicated representation-learning baselines such as
    
[^254]: 基于自动微分的有限元模型更新：从结构变形数据中学习本构神经网络

    Automatic-Differentiation-Enabled Finite Element Model Updating: Learning Constitutive Neural Networks from Structural Deformation Data

    [https://arxiv.org/abs/2606.05199](https://arxiv.org/abs/2606.05199)

    提出了ADE-FEMU框架，将本构神经网络嵌入JAX-FEM非线性求解器，通过梯度最小化测量失配损失，从动力学信息有限的结构变形数据中实现本构模型参数的自动学习。

    

    从异构全场变形数据中识别本构神经网络模型，为基于均匀应力-应变实验的传统标定方法提供了一种鲁棒的替代方案，尤其是考虑到可训练参数的高维度问题。本工作提出了ADE-FEMU（基于自动微分的有限元模型更新），这是一个端到端可微分的框架，它将本构神经网络模型集成到JAX-FEM非线性求解器中，并通过基于梯度的测量失配损失最小化来识别其参数，从而能够在动力学信息有限的结构数据上进行训练。因此，该方法可被视为超弹性材料有限元模型更新（FEMU）的扩展，实现于GPU加速、完全自动微分的有限元框架内，牛顿切线刚度与损失梯度均可自动计算。

    arXiv:2606.05199v2 Announce Type: replace-cross  Abstract: The identification of constitutive neural network models from heterogeneous full-field deformation data provides a robust alternative to traditional calibration methods based on homogeneous stress-strain experiments, particularly given the high dimensionality of trainable parameters. This work presents ADE-FEMU (Automatic-Differentiation-Enabled Finite Element Model Updating), an end-to-end differentiable framework that integrates a constitutive neural network model within a JAX-FEM nonlinear solver and identifies its parameters through gradient-based minimization of a measurement-mismatch loss, allowing for training on structural data with limited kinetic information. Thus, it can be considered as an extension of finite element model updating (FEMU) for hyperelastic materials, implemented within a GPU-accelerated, fully automatically differentiable finite element framework Newton tangent stiffness and loss gradients are comput
    
[^255]: 饱和陷阱与干预时机的主观性：基于情感的触发器与大语言模型裁判为何无法为自主智能体的干预把握恰当时机

    The Saturation Trap and the Subjectivity of Intervention Timing: Why Affect-Based Triggers and LLM Judges Fail to Time Interventions on Autonomous Agents

    [https://arxiv.org/abs/2606.04296](https://arxiv.org/abs/2606.04296)

    该研究以18维情感动力学引擎HEART作为诊断探针，揭示了“状态饱和陷阱”等机制，说明基于情感的触发器和大语言模型裁判无法准确把握中断自主智能体的干预时机。

    

    随着自主AI智能体从对话系统转向长时程软件执行，决定何时中断智能体的运行时安全层已变得不可或缺。我们以一个连续的18维情感动力学引擎（HEART）作为诊断探针来研究这一时机问题，在SWE-bench-Verified调试轨迹上，将四类干预触发器——绝对状态阈值、复合“状态-动作”模式、基于正则表达式的推理特征提取，以及零样本"LLM-as-judge"（大语言模型裁判）——与人工标注的干预点进行对比评估。我们报告了三项发现。第一，状态饱和陷阱：被建模的挫折感迅速越过阈值并停留在最大值（v2勘误：观察者无法记录恢复过程；衰减从未运行），使得“基于状态的阈值”触发器从“时刻检测器”退化为近乎恒定的指示器，在五条轨迹上对39%–83%的动作触发。第二，能力与情境下限……（原文摘要在此处截断）

    arXiv:2606.04296v2 Announce Type: replace  Abstract: As autonomous AI agents move from conversational systems to long-horizon software execution, runtime safety layers that decide when to interrupt an agent have become essential. We study this timing problem using a continuous 18-dimensional affective-dynamics engine (HEART) as a diagnostic probe, evaluating four intervention trigger families - absolute state thresholds, composite state-action patterns, regex reasoning-feature extraction, and zero-shot LLM-as-judge - against human-annotated intervention points on SWE-bench-Verified debugging traces. We report three findings. First, a State Saturation Trap: modeled frustration quickly crosses the threshold and stays at its maximum (v2 erratum: the observer could not register recovery; decay never ran), converting threshold-on-state triggers from moment detectors into near-constant indicators that fire on 39-83% of actions across five trajectories. Second, a capability-and-context floor 
    
[^256]: EvoDrive：基于自我改进大语言模型智能体的安全关键自动驾驶帕累托进化

    EvoDrive: Pareto Evolution for Safety-Critical Autonomous Driving via Self-Improving LLM Agents

    [https://arxiv.org/abs/2606.03678](https://arxiv.org/abs/2606.03678)

    EvoDrive是首个基于大语言模型智能体进化的多目标安全关键驾驶场景自动生成框架，通过仿真器支撑的执行者-评论者架构与帕累托进化，在保持场景真实性的同时最大化对抗性，突破了手工启发式方法的局限。

    

    生成安全关键场景对于验证和改进自动驾驶系统至关重要，但这本质上需要在保持真实性的同时最大化对抗性以暴露系统失效。现有方法通常依靠手工设计的启发式规则来处理这种权衡，将生成局限于已知的先验模式，并忽略了尚未充分探索的场景类型。尽管近年来开放式智能体进化方法可以突破这一限制，但不受约束的通用智能体缺乏严格的仿真器支撑，并且倾向于将多目标之间的张力坍缩为单标量的最大化问题。我们提出了EvoDrive，这是首个基于大语言模型的自动化智能体进化框架，用于多目标场景生成。EvoDrive采用了以仿真器为基础的执行者-评论者架构，其中由记忆驱动的执行者迭代地为生成器提出改进建议，评论者过滤掉不合理的候选方案，同时一个自进化的世界评估器将有前景的提案路由到（原文摘要在此处截断）

    arXiv:2606.03678v2 Announce Type: replace  Abstract: Generating safety-critical scenarios is essential for validating and improving autonomous driving systems, yet it inherently requires maximizing adversariality to expose failures while preserving realism. Existing methods usually manage this trade-off with handcrafted heuristics, confining generation to known priors and overlooking underexplored patterns. While recent open-ended agentic evolution can push this limit, unconstrained general agents lack strict simulator grounding and tend to collapse the multi-objective tension into single-scalar maximization. Here we present EvoDrive, the first automated, LLM-based agentic evolution framework for multi-objective scenario generation. EvoDrive employs a simulator-grounded actor-critic architecture where a memory-driven actor iteratively proposes improvements to the generators and critics filter out implausible candidates, and a self-evolving world evaluator routes promising proposals to 
    
[^257]: 教师何时应该移动？自我在线策略蒸馏中的时间耦合与稳定性

    When Should the Teacher Move? Temporal Coupling and Stability in Self On-Policy Distillation

    [https://arxiv.org/abs/2606.03532](https://arxiv.org/abs/2606.03532)

    本研究发现在自我在线策略蒸馏中，教师更新的隔离期（更新之间完全冻结教师）而非教师年龄是实现稳定学习的关键结构属性，并提出诊断框架揭示了状态遗忘崩溃现象——短程最优的固定更新时间表在长程训练中会因时钟驱动的刷新将漂移的学生一次性不可逆地复制进教师而导致灾难性失败。

    

    自我在线策略蒸馏通过让学生策略与一个源自其自身参数历史的教师进行训练，然而教师的更新时间表——它决定了教师与学生之间的时间耦合——尚未被系统地研究作为稳定性变量。通过在Qwen3-8B上进行受控的时间表扫描，我们确立了隔离期（定义为更新之间教师完全冻结）是实现稳定学习的关键结构属性，而非教师年龄。为了刻画这些底层的训练动态，我们引入了一个包含时间KL结构、刷新冲击和长度尾部风险的诊断框架。该框架进一步揭示了状态遗忘崩溃现象：最优的短程固定时间表在长程训练中会灾难性地失败，因为时钟驱动的刷新可以在单个不可逆的步骤中将瞬态漂移的学生复制到教师中。这一失败

    arXiv:2606.03532v2 Announce Type: replace-cross  Abstract: Self on-policy distillation trains a student policy against a teacher derived from its own parameter history, yet the teacher's update schedule -- which governs the \emph{temporal coupling} between teacher and student -- has not been systematically studied as a stability variable. Through a controlled schedule sweep on Qwen3-8B, we establish that \emph{isolation periods}, defined as complete teacher freezing between updates, are the key structural property enabling stable learning, not teacher age. To characterize these underlying training dynamics, we introduce a diagnostic framework of temporal KL structure, refresh shock, and length-tail risk. This framework further uncovers \emph{state-oblivious collapse}: optimal short-horizon fixed schedules catastrophically fail under long-horizon training because a clock-driven refresh can copy a transiently drifting student into the teacher in a single, irreversible step. This failure 
    
[^258]: 潜在奖励引导：一种自适应推理时框架，隐式促进推理大语言模型中的认知行为

    Latent Reward Steering: An Adaptive Inference-Time Framework that Implicitly Promotes Cognitive Behaviors in Reasoning LLMs

    [https://arxiv.org/abs/2606.00726](https://arxiv.org/abs/2606.00726)

    本文提出潜在奖励引导（LRS）框架，通过优化稀疏自编码器的潜在状态并在推理时利用奖励梯度进行自适应修正，隐式促进推理大语言模型中的认知行为，无需显式行为控制。

    

    摘要：arXiv:2606.00726v3 公告类型：替换 摘要：强推理不仅依赖于模型知识，还依赖于生成过程中认知行为的有效部署。现有方法通常依赖显式的行为级控制，这使得它们在失败和所需修正因推理状态、任务和模型而异时不够自适应。为此，我们提出潜在奖励引导（LRS），一种自适应推理时框架，通过优化隐含携带认知行为的稀疏自编码器（SAE）潜在状态来促进这些行为。LRS不依赖预定义的认知行为或由其派生的引导方向，而是基于最终答案正确性在推理轨迹上训练潜在奖励模型，以估计中间潜在状态的质量。在推理过程中，奖励梯度为脆弱的潜在状态提供状态特定的修正方向，而奖励和置信门控将干预限制在奖励模型能够可靠改进的状态上。

    arXiv:2606.00726v3 Announce Type: replace  Abstract: Strong reasoning depends not only on model knowledge but also on how effectively cognitive behaviors are deployed during generation. Existing methods often rely on explicit behavior-level control, making them insufficiently adaptive when failures and required corrections vary across reasoning states, tasks, and models. To this end, we propose Latent Reward Steering (LRS), an adaptive inference-time framework that promotes cognitive behaviors by optimizing the sparse-autoencoder (SAE) latent states that implicitly carry them. Rather than relying on predefined cognitive behaviors or steering directions derived from them, LRS trains a latent reward model on reasoning traces by final answer correctness to estimate the quality of intermediate latent states. During inference, reward gradients provide state-specific correction directions for fragile latent states, while a reward and confidence gate restricts intervention to states the rewar
    
[^259]: MyoSem：将肌电信号与自然语言动作语义对齐

    MyoSem: Aligning Electromyography with Natural-Language Action Semantics

    [https://arxiv.org/abs/2606.00174](https://arxiv.org/abs/2606.00174)

    MyoSem提出了一个肌电-动作语义对齐框架，将手部EMG信号与多视角自然语言动作描述映射到共享特征空间，通过语义原型构建、局部激活感知编码和语义查询聚合，实现了双向的EMG-文本检索，突破了传统固定手势标签分类的局限。

    

    肌电图（EMG）通过记录肌肉激活信号，被应用于手势识别、假肢控制和可穿戴交互等领域。现有的EMG方法通常将手部动作理解任务表述为对固定手势标签的预测，导致EMG信号与动作语义文本未能在共同特征空间中对齐，限制了信号与手部动作描述之间的双向查询能力。我们提出了MyoSem，一个EMG-动作语义对齐框架，将手部EMG信号与多视角自然语言动作描述映射到共享特征空间中以实现检索。MyoSem从多视角描述中构建语义原型，采用局部激活感知表示对EMG进行编码，并利用语义查询聚合局部EMG证据，从而实现双向的EMG-文本检索。我们在NinaPro DB2、EMG2Pose以及NinaPro DB2→DB3跨数据集上，通过七种评估协议对MyoSem进行了评估，涵盖随机划分、用户偏移、保留动作等场景。

    arXiv:2606.00174v2 Announce Type: replace-cross  Abstract: Electromyography (EMG) records muscle activation for hand gesture recognition, prosthetic control, and wearable interaction. Existing EMG methods usually formulate hand action understanding as prediction over fixed gesture labels, leaving EMG signals and action-semantic text unaligned in a common feature space and limiting bidirectional querying between signals and hand-action descriptions. We present MyoSem, an EMG--action semantic alignment framework that maps hand EMG and multi-view natural-language action descriptions into a shared feature space for retrieval. MyoSem builds semantic prototypes from multi-view descriptions, encodes EMG with local activation-aware representations, and uses semantic queries to aggregate local EMG evidence for bidirectional EMG--text retrieval. We evaluate MyoSem under seven protocols on NinaPro DB2, EMG2Pose, and NinaPro DB2$\rightarrow$DB3, covering random splits, user shifts, held-out action
    
[^260]: VITAL：基于视觉-语义双重监督的医学多模态大语言模型增强可解释潜在推理方法

    VITAL: Visual-Semantic Dual Supervision for Enhanced and Interpretable Latent Reasoning in Medical MLLMs

    [https://arxiv.org/abs/2605.28422](https://arxiv.org/abs/2605.28422)

    VITAL提出了一种带视觉-语义双重监督的医学多模态大语言模型潜在空间推理框架，通过辅助文本解码器与视觉投影器在推理时零开销地实现推理过程的文本与视觉双重可解释性。

    

    潜在推理使模型能够基于连续的隐状态而非显式token进行推理，从而避免了医学视觉问答中思维链的语言瓶颈与推理开销。然而，现有方法存在模态坍缩、视觉监督不足以及训练-推理不匹配等问题。此外，其不透明的隐状态缺乏可解释性，而这在临床应用中至关重要。我们提出VITAL，一种具有视觉-语义双重监督的医学多模态大语言模型潜在空间推理框架：一个辅助文本解码器从隐状态中重建推理链，同时一个视觉投影器从冻结的、独立的医学视觉编码器中回归ROI特征。这两个模块在推理时被丢弃，实现零开销，但可在事后重新附加以提供双重可解释性，在不牺牲效率的前提下为推理过程提供文本与视觉两种解释。我们构建了一个61K规模的数据……

    arXiv:2605.28422v2 Announce Type: replace-cross  Abstract: Latent reasoning enables reasoning over continuous hidden states rather than explicit tokens, avoiding the language bottleneck and inference overhead of chain-of-thought for medical VQA. However, existing methods suffer from modality collapse, insufficient visual supervision, and train-inference mismatch. Moreover, their opaque latent states offer no interpretability, which is critical in clinical applications. We propose VITAL, a latent-space reasoning framework for medical MLLMs with visual-semantic dual supervision: an auxiliary text decoder reconstructs reasoning chains from latent states, while a visual projector regresses ROI features from a frozen, independent medical vision encoder. Both modules are discarded at inference with zero overhead, yet can be re-attached post-hoc for dual interpretability, providing textual and visual explanations of the reasoning process without sacrificing efficiency. We construct a 61K data
    
[^261]: MPMMine：更好的约束获取基准测试

    MPMMine: Better Benchmarks for Constraint Acquisition

    [https://arxiv.org/abs/2605.26279](https://arxiv.org/abs/2605.26279)

    本文提出了MPMMine基准测试套件，包含基准数据集、配套软件包和基准测试工具，用于对利用多样化领域知识工件来发现、验证和增强数学规划模型的约束获取算法进行一致、系统和全面的评估，填补了该领域基准测试基础设施的空白。

    

    约束获取以及利用领域知识工件对数学规划模型进行验证和增强的相关研究，目前受到基准测试不足的制约。现有的基准测试组织松散，为单个问题提供的工件类型和格式不一致，是为求解器而非约束获取算法的基准测试而设计的，仅涵盖狭窄的问题类别，或遗漏了实验可重复性和约束获取方法公平比较所需的关键细节。我们开发了一个基准数据集、一个配套软件包和一个基准测试工具，专为对约束获取算法进行一致、系统和全面的评估而打造，填补了算法开发与实验基础设施之间的空白。本工作提出了MPMMine，这是一个旨在评估利用多样化工件来发现、验证和增强数学规划模型算法的基准测试套件。

    arXiv:2605.26279v2 Announce Type: replace  Abstract: Constraint Acquisition (CA) and related research on the validation and enhancement of Mathematical Programming (MP) models using domain knowledge artifacts are currently limited by inadequate benchmarks. The existing benchmarks are loosely organized, provide inconsistent types and formats of artifacts for individual problems, are designed for benchmarking solvers rather than CA algorithms, include narrow classes of problems, or omit crucial details for the reproducibility of experiments and fair comparison of CA methods.   We develop a benchmark dataset, a supporting software package, and a benchmarking tool crafted for the consistent, systematic, and comprehensive assessment of CA algorithms, bridging the gap in algorithmic development and experimental infrastructure.   This work presents MPMMine, a benchmark suite designed to assess algorithms that discover, validate, and enhance MP models using diverse artifacts. The development o
    
[^262]: EchoDistill：通过噪声到干净的自蒸馏实现鲁棒的大型音频语言模型

    EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation

    [https://arxiv.org/abs/2605.23954](https://arxiv.org/abs/2605.23954)

    EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。

    

    大型音频语言模型（LALMs）仍然容易受到声学噪声的影响，噪声会掩盖与任务相关的证据并产生不可靠的响应。我们提出了EchoDistill，这是一种噪声到干净的自蒸馏框架，在后训练阶段将干净音频作为特权信息加以利用。以噪声输入的学生模型采样能够反映其推理时行为的候选响应，而同一骨干网络的冻结副本则处理对应的干净音频。EchoDistill结合了掩码响应token蒸馏、任务门控一致性塑形以及教师参考的组相对优化，使噪声输入下的生成结果与干净条件下的语义保持对齐。推理时仅保留学生模型，不引入任何额外的推理开销。在三个LALM骨干网络和三个音频领域、信噪比为-10dB的条件下，EchoDistill相比最强基线将噪声输入下的平均准确率提升了1.63个百分点。在Qwen2.5-Omni上，它

    arXiv:2605.23954v2 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) remain vulnerable to acoustic noise, which can obscure task-relevant evidence and produce unreliable responses. We propose EchoDistill, a noisy-to-clean self-distillation framework that uses clean audio as privileged information during post-training. A noisy-input student samples candidate responses reflecting its inference-time behavior, while a frozen copy of the same backbone processes the corresponding clean audio. EchoDistill combines masked response-token distillation, task-gated consistency shaping, and teacher-referenced group-relative optimization to align noisy-input generation with clean-conditioned semantics. Only the student is retained at inference time, introducing no additional inference cost. Across three LALM backbones and three audio domains at -10dB, EchoDistill improves average noisy-input accuracy by 1.63 percentage points over the strongest baseline. On Qwen2.5-Omni, it
    
[^263]: 随机MeanFlow策略：基于熵镜像下降的单步生成式控制

    Stochastic MeanFlow Policies: One-Step Generative Control with Entropic Mirror Descent

    [https://arxiv.org/abs/2605.21282](https://arxiv.org/abs/2605.21282)

    SMFP是一种将MeanFlow噪声到动作映射与高斯重参数化相结合的单步生成式策略，通过解析熵下界实现熵正则化镜像下降强化学习，在七个MuJoCo基准上优于高斯策略和其他生成式策略。

    

    在在线离策略强化学习中，将熵正则化与镜像下降相结合，需要既具备高效采样能力又能显式控制熵的表达性策略。高斯策略难以表示多模态目标，而生成式策略通常需要迭代采样或缺乏可处理的熵估计。我们提出了随机MeanFlow策略，这是一种单步生成式策略类别，它将基于MeanFlow的噪声到动作映射与高斯重参数化相结合。潜在映射能够捕捉多样化的动作模式，而可学习的条件高斯尺度则提供了具有解析形式的熵下界。我们利用该下界来调控随机性，并结合直接价值引导和优势加权的MeanFlow回归，从而实现基于单步策略的熵正则化镜像下降学习。在七个MuJoCo基准测试中，SMFP相较于高斯策略和生成式策略取得了强劲的性能表现。

    arXiv:2605.21282v3 Announce Type: replace-cross  Abstract: Combining entropy regularisation with mirror descent in online off-policy reinforcement learning requires expressive policies with efficient sampling and explicit entropy control. Gaussian policies struggle to represent multimodal targets, whereas generative policies often require iterative sampling or lack tractable entropy estimates. We introduce Stochastic MeanFlow Policies (SMFP), a one-step generative policy class that combines MeanFlow-based noise-to-action mappings with Gaussian reparameterisation. The latent mapping captures diverse action modes, while a learnable conditional Gaussian scale provides an analytic entropy lower bound. We use this bound to regulate stochasticity alongside direct value guidance and advantage-weighted MeanFlow regression, enabling entropy-regularised mirror-descent learning with one-step policies. Across seven MuJoCo benchmarks, SMFP achieves strong performance against Gaussian and generative
    
[^264]: 从SGD到Muon：基于Schatten-p范数的自适应优化

    From SGD to Muon: Adaptive Optimization via Schatten-p Norms

    [https://arxiv.org/abs/2605.19781](https://arxiv.org/abs/2605.19781)

    该论文提出了一种基于Schatten-p范数的数据驱动准则，可动态地为神经网络各层自适应选择从SGD到Muon之间的最优更新几何结构，并在统一框架下恢复了SGD、Muon、Adam和MuAdam等主流优化器。

    

    现代优化器（如Muon）对其更新施加矩阵级别的几何约束，这些约束可以在线性最小化预言机（LMO）理论下得到统一。然而，现有所有方法都为更新规则施加固定的LMO几何结构，这些结构是通过设计或经验选择的，未必与问题本身的几何特性相符从而并非最优。我们提出了一种新颖且高效的数据驱动准则，用于在深度神经网络的各个层上动态选择近似最优的更新LMO几何结构。该准则借助单步随机特征回归代理模型，从梯度和激活统计量中以闭式形式推导得出，能够在一个从SGD更新到Muon更新之间进行插值的设计空间中自由导航。此外，通过整合参数级预条件化，我们的框架可以将SGD、Muon、Adam和MuAdam作为特定极值情形加以恢复。为使这种自适应方法具备可扩展性，我们将其与高效的计算策略相结合。

    arXiv:2605.19781v2 Announce Type: replace  Abstract: Modern optimizers, like Muon, impose matrix-wise geometry constraints on their updates. These matrix-wise constraints can be unified under Linear Minimization Oracle (LMO) theory. However, all current methods impose fixed LMO geometries for the update rules, chosen by-design or empirically, which are not necessarily optimal according to the problem's geometry. We introduce a novel efficient datadriven criterion for dynamically choosing proxy-optimal update LMO geometries on individual Deep Neural Network layers. Derived in closed form from gradient and activation statistics using a single-step random feature regression surrogate model, our criterion navigates a design space interpolating from SGD to Muon updates. Moreover, integrating parameter-wise preconditioning allows our framework to recover SGD, Muon, Adam, and MuAdam as specific extrema. To make this adaptive approach scalable, we pair it with efficient computational strategie
    
[^265]: 基于采样的安全强化学习

    Sampling-Based Safe Reinforcement Learning

    [https://arxiv.org/abs/2605.19469](https://arxiv.org/abs/2605.19469)

    该论文提出基于采样的安全强化学习算法SBSRL，通过在有限动力学样本上联合施加约束来近似最坏情况优化，并结合认知不确定性约束的探索策略，提供了学习全过程的高概率安全保证和有限时间样本复杂度界，在仿真与真实机器人上实现了安全高效的探索。

    

    安全探索仍然是强化学习（RL）中的一个根本性挑战，它限制了RL智能体在现实世界中的部署。我们提出了基于采样的安全强化学习（SBSRL），这是一种基于模型的RL算法，通过在有限的动力学样本集合上联合实施约束，在整个学习过程中保持安全性。这一表述方式近似了针对不确定动力学的难以处理的最坏情况优化，并使连续域中的实用安全保证成为可能。我们进一步引入了一种基于约束认知不确定性的探索策略，从而无需显式的探索奖励。在正则性条件下，我们推导出了整个学习过程中安全性的高概率保证，以及恢复近最优策略的有限时间样本复杂度界。实验表明，SBSRL在仿真和真实机器人硬件上都实现了安全且高效的探索。

    arXiv:2605.19469v2 Announce Type: replace-cross  Abstract: Safe exploration remains a fundamental challenge in reinforcement learning (RL), limiting the deployment of RL agents in the real world. We propose Sampling-Based Safe Reinforcement Learning (SBSRL), a model-based RL algorithm that maintains safety throughout the learning process by enforcing constraints jointly across a finite set of dynamics samples. This formulation approximates an intractable worst-case optimization over uncertain dynamics and enables practical safety guarantees in continuous domains. We further introduce an exploration strategy based on constraining epistemic uncertainty, eliminating the need for explicit exploration bonuses. Under regularity conditions, we derive high-probability guarantees of safety throughout learning and a finite-time sample complexity bound for recovering a near-optimal policy. Empirically, SBSRL achieves safe and efficient exploration both in simulation and in real robotic hardware, 
    
[^266]: 当模拟撒谎时：面向工具使用智能体的模拟到现实基准与域随机化强化学习方案

    When Simulation Lies: A Sim-to-Real Benchmark and Domain-Randomized RL Recipe for Tool-Use Agents

    [https://arxiv.org/abs/2605.11928](https://arxiv.org/abs/2605.11928)

    该论文提出了基于真实部署失效案例构建的RobustBench-TC基准，用以揭示工具使用智能体在模拟与真实环境之间严重不均衡的鲁棒性差距，并给出了域随机化强化学习训练方案以提升其现实部署可靠性。

    

    工具使用型语言智能体通常在假设输入干净、工具注册表无歧义且API可靠的基准上进行评估。然而真实部署环境违背了所有这些假设：用户的拼写错误会传播为幻觉出的工具名称，配置错误的请求超时可能导致智能体无限期停滞，跨服务器重复的工具名称可能使SDK冻结。我们将这些失效问题视为工具使用部分可观测马尔可夫决策过程（POMDP）中的模拟到现实差距，其中部署噪声通过观测、动作空间、奖励相关元数据或转移动力学进入系统。我们提出了RobustBench-TC基准，包含22种扰动类型并按这四个POMDP组件组织，每种扰动均基于经过验证的GitHub issue或有据可查的工具调用失效案例。在对参数规模从1.5B到32B的21个模型（包括闭源的o4-mini）的评估中，鲁棒性表现极不均衡：观测扰动使准确率下降……（原文摘要在此处截断）

    arXiv:2605.11928v2 Announce Type: replace  Abstract: Tool-use language agents are evaluated on benchmarks that assume clean inputs, unambiguous tool registries, and reliable APIs. Real deployments violate all these assumptions: user typos propagate into hallucinated tool names, a misconfigured request timeout can stall an agent indefinitely, and duplicate tool names across servers can freeze an SDK. We study these failures as a sim-to-real gap in the tool-use partially observable Markov decision process (POMDP), where deployment noise enters through the observation, action space, reward-relevant metadata, or transition dynamics. We introduce RobustBench-TC, a benchmark with 22 perturbation types organized by these four POMDP components, each grounded in a verified GitHub issue or documented tool-calling failure. Across 21 models from 1.5B to 32B parameters (including the closed-source o4-mini), the robustness profile is sharply uneven: observation perturbations reduce accuracy by less 
    
[^267]: MemReread：通过记忆引导的重读增强智能体长上下文推理

    MemReread: Enhancing Agentic Long-Context Reasoning via Memory-Guided Rereading

    [https://arxiv.org/abs/2605.10268](https://arxiv.org/abs/2605.10268)

    MemReread通过在最终记忆不充分时触发问题分解与针对性重读，恢复流式阅读中被过早丢弃的间接证据，从而在不依赖中间检索的情况下增强智能体的长上下文非线性推理能力，并结合强化学习提升长度外推能力和重读次数的自适应决策。

    

    长上下文推理任务推动了智能体记忆方法的发展，这类方法按顺序处理文档块并维护一个动态更新的记忆，从而避免了标准注意力机制的二次方复杂度。现有方法通常依赖检索来恢复记忆覆盖过程中被丢弃的信息，但在记忆形成过程中仍然容易丢失证据，并且容易受到无效查询的干扰。为了克服这些局限性，我们提出了MemReread。该方法建立在流式阅读之上，规避了中间检索，当最终记忆不足以回答问题时，会触发问题分解和针对性重读。通过恢复可能被过早丢弃的间接证据，MemReread在保持文档理解逻辑流程的同时实现了非线性推理。我们进一步引入了一个强化学习框架，该框架提升了长度外推能力，并能自适应地确定重读的次数。

    arXiv:2605.10268v2 Announce Type: replace-cross  Abstract: Long-context reasoning tasks have motivated agent-memory approaches that process document chunks sequentially while maintaining a dynamically updated memory, avoiding the quadratic complexity of standard attention. Existing methods often rely on retrieval to recover information discarded during memory overwriting, but remain vulnerable to evidence loss during memory formation and interference from invalid queries. To overcome these limitations, we propose MemReread. Built upon streaming reading, it circumvents intermediate retrieval, triggering question decomposition and targeted rereading when the final memory is insufficient. By recovering indirect evidence that may have been prematurely discarded, MemReread enables non-linear reasoning while preserving the logical flow of document comprehension. We further introduce a reinforcement learning framework that improves length extrapolation and adaptively determines the number of 
    
[^268]: MAGE：基于协同进化知识图谱的多智能体自进化

    MAGE: Multi-Agent Self-Evolution with Co-Evolutionary Knowledge Graphs

    [https://arxiv.org/abs/2605.10064](https://arxiv.org/abs/2605.10064)

    MAGE框架将自我知识外化为经验、任务搜索与技能路由协同进化的四子图知识图谱，使冻结骨干的语言模型智能体在自进化过程中既能决定学什么、又能跨迭代保留所学知识。

    

    自进化的语言模型智能体必须决定下一步学习什么，以及如何在迭代过程中保留已学到的知识。现有系统通常以自然语言反馈、扁平的情节记忆或隐式强化信号的形式承载这种跨迭代知识，但这些方式都无法在推理时良好地支持冻结的弱骨干模型。本文提出MAGE（Multi-Agent Graph-guided Evolution，多智能体图引导进化），一个将自我知识外化为四子图协同进化知识图谱的框架。其经验子图同时存储教师撰写的失败纠正记录和学习者自身过去的正确推理轨迹，这些内容作为任务条件化的指导被检索，供冻结的执行模型使用。在进化过程中，知识图谱、任务级搜索赌博机和技能级路由赌博机从同一奖励流中同步更新，而学习者的骨干模型保持不变。我们进一步提供了结构分析，表明……

    arXiv:2605.10064v3 Announce Type: replace  Abstract: Self-evolving language-model agents must decide what to learn next and how to preserve what they have learned across iterations. Existing systems typically carry this cross-iteration knowledge as natural-language feedback, flat episodic memory, or implicit reinforcement signals, none of which cleanly supports a frozen weak backbone at inference time. This paper introduces MAGE (Multi-Agent Graph-guided Evolution), a framework that externalizes self-knowledge into a four-subgraph co-evolutionary knowledge graph. Its experience subgraph stores both teacher-written failure corrections and the learner's own past correct reasoning traces, which are retrieved as task-conditioned guidance for a frozen execution model. During evolution, the graph, a task-level search bandit, and a skill-level routing bandit are updated from the same reward stream, while the learner's backbone remains unchanged. We further provide structural analysis showing 
    
[^269]: FactoryNet：面向工业时间序列基础模型的大规模数据集

    FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models

    [https://arxiv.org/abs/2605.09081](https://arxiv.org/abs/2605.09081)

    FactoryNet是一个包含1.13亿数据点、覆盖7种机器形态和4项任务的大规模工业时间序列预训练数据集，通过统一的S-E-F-C控制理论模式使指令意图与实际响应可在不同机器间比较，为工业时间序列基础模型奠定了基础。

    

    工业机器是受驱动的系统，然而大多数开放的时间序列语料库仅记录传感器的输出结果；少数记录了指令信号的语料库则以供应商特定的格式呈现，且不同机器之间各不相同，因此指令意图与测量响应无法在不同机器形态之间进行比较。我们提出了FactoryNet，一个面向工业时间序列的预训练语料库：包含1.13亿个数据点，涵盖7种机器形态、4项任务共5.7万次端到端任务执行（其中1.5万次为真实数据，4.2万次为合成数据），并附带27种标注的异常类型，以及健康基线和反事实配对数据。每个来源中的每个信号都被映射到一个统一的控制理论模式——设定值、作用量、反馈、上下文，该模式以共同的表征框架表达任何受驱动系统，并使指令与实际执行之间的动态差异可以作为显式的预测残差被读取。利用该语料库，我们发现一个仅在仿真数据上训练的任务阶段模型，在没有……

    arXiv:2605.09081v5 Announce Type: replace-cross  Abstract: Industrial machines are actuated systems, yet most open time-series corpora record sensor outcomes alone; the few that do log commanded signals expose them in vendor-specific forms that differ from machine to machine, so commanded intent and measured response cannot be compared across embodiments. We introduce FactoryNet, a pretraining corpus for industrial time-series: 113M datapoints across 57k end-to-end task executions (15k real, 42k synthetic) on seven embodiments and four tasks, with 27 annotated anomaly types alongside healthy baselines and counterfactual pairs. Every signal in every source is mapped into a single control-theoretic schema, Setpoint, Effort, Feedback, Context (S-E-F-C), which expresses any actuated system in a common representational frame and makes commanded-versus-realized dynamics readable as explicit prediction residuals. Using the corpus, we find that a task-phase model trained on simulation, with no
    
[^270]: UxSID：面向超长序列的语义感知用户兴趣建模

    UxSID: Semantic-Aware User Interests Modeling for Ultra-Long Sequence

    [https://arxiv.org/abs/2605.09040](https://arxiv.org/abs/2605.09040)

    UxSID 通过语义 ID 与双层注意力机制构建语义组共享兴趣记忆，为超长用户行为序列建模提供兼顾效率与效果的第三条路径，实现 SOTA 性能并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    

    对超长用户行为序列进行建模需要在效率与效果之间进行艰难的权衡。现有范式要么依赖特定物品的检索，要么采用与物品无关的压缩方式，我们提出 UxSID 框架，探索第三条路径：语义组共享兴趣记忆。通过利用语义 ID（SIDs）和双层注意力策略，UxSID 无需承担特定物品模型的沉重开销，即可捕捉目标感知的用户偏好。这一端到端架构在计算简洁性与语义感知之间取得平衡，实现了最先进的性能，并在大规模广告 A/B 测试中带来 0.337% 的收入提升。

    arXiv:2605.09040v4 Announce Type: replace  Abstract: Modeling ultra-long user sequences involves a difficult trade-off between efficiency and effectiveness. While current paradigms rely on either item-specific search or item-agnostic compression, we propose UxSID, a framework exploring a third path: semantic-group shared interest memory. By utilizing Semantic IDs (SIDs) and a dual-level attention strategy, UxSID captures target-aware preferences without the heavy cost of item-specific models. This end-to-end architecture balances computational parsimony with semantic awareness, achieving state-of-the-art performance and a 0.337% revenue lift in large-scale advertising A/B test.
    
[^271]: 超越成对比较：你的语言模型正在暗中优化一张偏好图

    Beyond Pairs: Your Language Model is Secretly Optimizing a Preference Graph

    [https://arxiv.org/abs/2605.08037](https://arxiv.org/abs/2605.08037)

    本文提出 GraphDPO，将 DPO 从成对偏好比较推广到有向无环偏好图结构，充分利用多个 rollout 之间的传递性偏好关系进行聚合监督，从而实现更稳定、更充分的语言模型对齐。

    

    直接偏好优化（DPO）通过成对偏好比较来对齐语言模型，为基于人类反馈的强化学习（RLHF）提供了一种简单有效的替代方案。然而，在许多实际场景中，训练数据由每个提示的多个 rollout 组成，这蕴含了丰富的偏好结构，而成对 DPO 未能充分利用这些结构。将此类数据压缩为独立配对会丢弃传递性，引入冗余或冲突的监督信号，并可能导致优化不稳定。我们提出了图直接偏好优化（GraphDPO），这是 DPO 的一种结构化推广方法，作用于由 rollout 排名诱导的有向无环偏好图。GraphDPO 将支配关系编码为图中的边，并优化一种受 Plackett-Luce 启发的图结构化目标函数，该函数在 dominated 邻域上聚合监督信号，传播输入偏好的传递结构，同时恢复（注：原文摘要在此处截断）

    arXiv:2605.08037v2 Announce Type: replace-cross  Abstract: Direct Preference Optimization (DPO) aligns language models using pairwise preference comparisons, offering a simple and effective alternative to Reinforcement Learning (RL) from human feedback. However, in many practical settings, training data consists of multiple rollouts per prompt, inducing rich preference structure that pairwise DPO fails to exploit. Collapsing such data into independent pairs discards transitivity, introduces redundant or conflicting supervision, and can lead to unstable optimization. We propose Graph Direct Preference Optimization (GraphDPO), a structured generalization of DPO that operates over directed acyclic preference graphs induced by rollout rankings. GraphDPO encodes dominance relations as edges and optimizes a graph-structured Plackett--Luce-inspired objective that aggregates supervision over dominated neighborhoods, propagating the transitive structure of the input preferences while recovering
    
[^272]: 具有全局最优与近似保证的脉冲神经网络凸优化

    Convex Optimization of Spiking Neural Networks with Global and Approximation Guarantees

    [https://arxiv.org/abs/2605.08022](https://arxiv.org/abs/2605.08022)

    该论文将脉冲神经网络（包括LIF-SNN）的训练问题首次等价转化为凸优化问题，证明了零对偶间隙与全局最优保证，并提出具有高概率近似保证的随机化替代方法以解决凸规划变量指数级增长的问题。

    

    脉冲神经网络已被提出作为深度神经网络的节能替代方案。其训练问题是非凸的，且涉及不可微的脉冲函数，因此现有方法无法保证达到全局最优。我们通过将脉冲神经网络优化问题表述为一个等价的凸优化问题来解决这一难题。具体而言，我们证明了一类阈值循环神经网络（包括泄漏积分发放型脉冲神经网络）在带权重衰减正则化条件下的训练问题具有零对偶间隙，且与某个凸规划具有相同的最优值。由于所得的凸规划的变量数量可能随数据量呈指数级增长，我们进一步引入了一种实用的随机化替代方法，并为其近似质量建立了高概率保证。从实证结果来看，表现最好的凸优化变体在加法（addition）和首尾XOR（first-last-XOR）任务上优于代理梯度训练方法，在序列MNIST任务上与其性能相当。

    arXiv:2605.08022v3 Announce Type: replace-cross  Abstract: Spiking neural networks (SNNs) have been proposed as energy-efficient alternatives to deep neural networks. Their training is non-convex and involves a non-differentiable spike function, so existing methods offer no guarantee of reaching a global optimum. We address this issue by formulating SNN optimization as an equivalent convex optimization problem. Specifically, we show that the weight-decay-regularized training problem for a class of threshold RNNs, including leaky integrate-and-fire SNNs, has zero duality gap and the same optimal value as a convex program. Since the resulting convex programs can have exponentially many variables in the amount of data, we introduce a practical randomized surrogate and establish a high-probability guarantee on its approximation quality. Empirically, the best convex variant outperforms surrogate-gradient training on addition and first-last-XOR, matches its performance on sequential MNIST, a
    
[^273]: FactoryBench：评估工业机器理解能力

    FactoryBench: Evaluating Industrial Machine Understanding

    [https://arxiv.org/abs/2605.07675](https://arxiv.org/abs/2605.07675)

    FactoryBench基于珀尔因果阶梯构建了四个因果层级的问答基准，包含7万余个问答项，用于评估时间序列模型和大语言模型对工业机器人遥测数据的理解能力，零样本评测显示当前前沿大语言模型在结构化任务上不超过50%、在决策任务上不超过18%。

    

    我们推出了FactoryBench，一个用于评估时间序列模型和大语言模型（LLM）在工业机器人遥测数据上机器理解能力的基准。问答对按照四个因果层级（状态、干预、反事实、决策）组织，实例化了珀尔的因果阶梯，并涵盖五种答案格式：其中四种结构化格式采用确定性评分，自由形式答案则通过“大模型作为裁判”的投票协议进行评分。我们提出了一个围绕结构化问题模板构建的可扩展问答生成框架，介绍了FactoryWave（一个从UR3协作机器人和KUKA KR10工业机械臂采集的密集、多任务、多变量传感器数据集），并基于来自FactoryWave、AURSAD和voraus-AD的约1.5万个规范化回合，构建了包含超过7万个问答项的大规模基准FactoryBench。对六个前沿大语言模型的零样本评估表明，没有任何模型在结构化层级上超过50%，在决策任务上不超过18%，揭示了……

    arXiv:2605.07675v2 Announce Type: replace  Abstract: We introduce FactoryBench, a benchmark for evaluating time-series models and LLMs on machine understanding over industrial robotic telemetry. Q&A pairs are organized along four causal levels (state, intervention, counterfactual, decision) instantiating Pearl's ladder of causation, and span five answer formats: four structured formats are scored deterministically and free-form answers are scored by an LLM-as-judge voting protocol. We propose a scalable Q&A generation framework built around structured question templates, present FactoryWave (a dense, multitask, multivariate sensor dataset collected from a UR3 cobot and a KUKA KR10 industrial arm), and construct FactoryBench as a large-scale benchmark of over 70k Q&A items grounded in roughly 15k normalized episodes from FactoryWave, AURSAD, and voraus-AD. Zero-shot evaluation of six frontier LLMs shows that no model exceeds 50% on structured levels or 18% on decision-making, revealing 
    
[^274]: 智能体能为化学反应定价吗？评估大语言模型的化学品成本推理能力

    Can Agents Price a Reaction? Evaluating LLMs on Chemical Cost Reasoning

    [https://arxiv.org/abs/2605.07251](https://arxiv.org/abs/2605.07251)

    提出ChemCost基准，利用化学品采购成本估算这一具有精确、无评判者真实标准的实际任务，首次对大语言模型智能体的科学工具使用能力（包括接地、检索、采购和算术）进行严格评估与分阶段诊断。

    

    大语言模型（LLMs）作为使用工具的智能体能力日益增强，现有基准已涵盖多种通用智能体任务。然而，对科学工具使用的严格评估仍然有限。在化学领域，近期的智能体虽然能够规划合成路线并调用领域特定工具，但其评估往往依赖精心策划的演示、专家评审或以LLM作为评判者的打分，而非精确的、无需评判者的真实标准。我们通过化学品采购成本估算任务来填补这一空白。这是一项实际任务，智能体需要根据反应描述完成以下工作：确定化学物质的身份（接地）、检索供应商报价、选择有效的可购买包装规格、标准化数量并计算成本。我们推出了ChemCost基准，包含1,427个可评估的反应，基于涵盖2,261种化学品和230,775条供应商报价的冻结定价快照，支持标量评分以及对接地、检索、采购和算术等各阶段的错误诊断。

    arXiv:2605.07251v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have become increasingly capable as tool-using agents, with benchmarks spanning diverse general agentic tasks. Yet rigorous evaluation of scientific tool use remains limited. In chemistry, recent agents can plan syntheses and invoke domain-specific tools, but evaluations often rely on curated demonstrations, expert assessment, or LLM-as-judge scoring rather than exact, judge-free ground truth. We address this gap with chemical procurement cost estimation, a practical task in which an agent must ground chemical identities, retrieve supplier quotes, select valid purchasable packs, normalize quantities, and compute cost from a reaction description. We introduce ChemCost, a benchmark of 1,427 evaluable reactions grounded to a frozen pricing snapshot covering 2,261 chemicals and 230,775 supplier quotes, supporting scalar scoring and stage-level diagnosis of grounding, retrieval, procurement, and arithmetic fai
    
[^275]: LensVLM：针对文本压缩视觉表示的选择性上下文扩展

    LensVLM: Selective Context Expansion for Compressed Visual Representation of Text

    [https://arxiv.org/abs/2605.07019](https://arxiv.org/abs/2605.07019)

    LensVLM提出了一种推理框架和后训练方案，使VLM能先扫描压缩的渲染文本图像，再通过学习到的工具选择性地将相关图像扩展为未压缩形式，从而在4.3倍有效压缩下保持与全文处理相当的准确率，并在最高10.1倍压缩下超越现有基线。

    

    视觉语言模型（VLM）提供了一种令人兴奋的可能性，即将文本作为渲染图像进行处理，从而避免了将文本标记化为长token序列的需要。由于VLM的图像编码器将固定大小的图像映射为固定数量的视觉token，改变渲染分辨率便提供了一个细粒度的压缩旋钮。然而，随着压缩程度的增加，准确率会迅速下降：字符缩小到低于视觉编码器的有效分辨率，导致其无法被辨别。为解决这一问题，我们提出了LensVLM，这是一个推理框架与后训练方案，使VLM能够先扫描压缩图像，然后通过学习到的工具选择性地仅将相关图像扩展至未压缩形式。基于Qwen3.5-9B-Base构建的LensVLM在4.3倍有效压缩下仍保持与全文上限相当的准确率，并在最高10.1倍有效压缩下优于基于检索的、文本压缩和视觉压缩等基线方法。

    arXiv:2605.07019v2 Announce Type: replace-cross  Abstract: Vision Language Models (VLMs) offer the exciting possibility of processing text as rendered images, bypassing the need for tokenizing the text into long token sequences. Since VLM image encoders map fixed-size images to a fixed number of visual tokens, varying rendering resolution provides a fine-grained compression knob. However, accuracy deteriorates quickly as compression increases: characters shrink below the vision encoder's effective resolution, making them indistinguishable. To address this, we propose LensVLM, an inference framework and post-training recipe that enables VLMs to scan compressed images, then selectively expand only the relevant images to their uncompressed form via learned tools. Building on Qwen3.5-9B-Base, LensVLM maintains accuracy comparable to the full-text upper bound at 4.3$\times$ effective compression and outperforms retrieval-based, text- and visual-compression baselines up to 10.1$\times$ effec
    
[^276]: IAD-Unify：面向工业异常理解、分割与生成的任务专用接口

    IAD-Unify: Task-Specific Interfaces for Industrial Anomaly Understanding, Segmentation, and Generation

    [https://arxiv.org/abs/2604.12440](https://arxiv.org/abs/2604.12440)

    提出 IAD-Unify 统一架构，通过任务专用 token 接口连接多模态语言模型、密集视觉专家和扩散编辑器，首次将工业异常的解释、像素级分割与受控生成三种互补能力整合于同一框架。

    

    工业异常检测需要互补的能力：解释观察到的缺陷、定位其像素位置，以及合成受控的编辑。我们提出了 IAD-Unify，这是一种统一架构，通过任务专用的 token 接口连接多模态语言模型（MLLM）、密集视觉专家模块和扩散编辑器。多参考 DINOv2 通路构建一个密集异常场，并将其 1,369 个单元压缩为 81 个结构化证据 token。Qwen3.5 利用这些 token 生成有依据的回答，并通过 32 个任务 token 将其转换为叠加在密集掩码之上的语义残差。一个独立的 256 查询接口将 Qwen 状态重采样为 Stable Diffusion 的完整交叉注意力上下文，同时编辑器保留其源隐变量和硬掩码输入。各任务通路共享同一个 Qwen 适配，而无需强迫每个任务经过相同的视觉瓶颈；特别地，证据 token 被排除在生成过程之外。

    arXiv:2604.12440v2 Announce Type: replace-cross  Abstract: Industrial anomaly inspection requires complementary capabilities: explaining an observed defect, localizing its pixels, and synthesizing a controlled edit. We present IAD-Unify, a unified architecture connecting a multimodal language model (MLLM), dense visual expert, and diffusion editor through task-specific token interfaces. A multi-reference DINOv2 pathway forms a dense anomaly field and compresses its 1,369 cells into 81 structured evidence tokens. Qwen3.5 consumes these tokens for grounded answers and, with 32 task tokens, converts them into a semantic residual over the dense mask. A separate 256-query interface resamples Qwen states into Stable Diffusion's complete cross-attention context, while the editor retains its source latent and hard-mask inputs. The task pathways share one Qwen adaptation without forcing every task through the same visual bottleneck; in particular, evidence tokens are excluded from the generatio
    
[^277]: 物理引导的代理学习实现湍流机翼的零样本控制

    Physics-guided surrogate learning enables zero-shot control of turbulent wings

    [https://arxiv.org/abs/2604.09434](https://arxiv.org/abs/2604.09434)

    该研究利用壁面湍流的局部结构，通过物理引导的代理学习，将在槽道流中训练的强化学习策略直接零样本迁移部署到 NACA4412 机翼上，实现 11.1% 的总阻力降低，超越了最先进的对向控制方法（8.9%）。

    

    气动表面上的湍流边界层是飞机阻力的主要来源，然而由于多尺度动力学特性和空间变异性，尤其在逆压梯度条件下，其控制仍然极具挑战性。强化学习在典型流动中已超越最先进的控制策略，但其在真实几何形状上的应用受到计算成本和可迁移性的限制。本研究表明，通过利用壁面约束湍流的局部结构，可以克服这些限制。策略在与机翼边界层统计特性相匹配的湍流槽道流中进行训练，随后直接部署到雷诺数 $Re_c=2\times10^5$ 的 NACA4412 翼型上，而无需在机翼上进行任何额外训练，构成零样本部署。迁移后的策略使总阻力降低了 11.1%，而最先进的对向控制仅能降低 8.9%。在考虑执行成本的情况下，相应的……（原文摘要至此截断）

    arXiv:2604.09434v3 Announce Type: replace-cross  Abstract: Turbulent boundary layers over aerodynamic surfaces are a major source of aircraft drag, yet their control remains challenging due to multiscale dynamics and spatial variability, particularly under adverse pressure gradients. Reinforcement learning has outperformed state-of-the-art strategies in canonical flows, but its application to realistic geometries is limited by computational cost and transferability. Here we show that these limitations can be overcome by exploiting local structures of wall-bounded turbulence. Policies are trained in turbulent channel flows matched to wing boundary-layer statistics and deployed directly onto a NACA4412 wing section at $Re_c=2\times10^5$ without additional training on the wing, constituting zero-shot deployment. The transferred policies reduce total drag by 11.1\%, while state-of-the-art opposition control reduces it by 8.9\%. With the actuation cost taken into account, the corresponding 
    
[^278]: AnomalyAgent：基于工具增强强化学习的智能体式工业异常合成

    AnomalyAgent: Agentic Industrial Anomaly Synthesis via Tool-Augmented Reinforcement Learning

    [https://arxiv.org/abs/2604.07900](https://arxiv.org/abs/2604.07900)

    该论文提出AnomalyAgent异常合成智能体，通过配备提示词生成、图像生成、质量评估、知识检索和掩码生成五种工具实现闭环优化，并采用监督微调加强化学习的两阶段训练，生成高真实感且多样的工业异常样本，以缓解异常检测的数据稀缺问题。

    

    工业异常生成是缓解异常检测任务中数据稀缺问题的重要方法。现有的大多数异常合成方法依赖单步生成机制，缺乏复杂推理和迭代优化能力，难以生成具有高语义真实感的异常样本。我们提出AnomalyAgent，一个具备自我反思、知识检索和迭代优化能力的异常合成智能体，旨在生成真实且多样的异常。具体而言，AnomalyAgent配备了五个工具：提示词生成（PG）、图像生成（IG）、质量评估（QE）、知识检索（KR）和掩码生成（MG），从而实现闭环优化。为提升决策与自我反思能力，我们基于真实异常图像构建结构化轨迹，并设计了两阶段训练框架：先进行监督微调，再进行强化学习训练（摘要内容在此处截断）。

    arXiv:2604.07900v2 Announce Type: replace-cross  Abstract: Industrial anomaly generation is a crucial method for alleviating the data scarcity problem in anomaly detection tasks. Most existing anomaly synthesis methods rely on single-step generation mechanisms, lacking complex reasoning and iterative optimization capabilities, making it difficult to generate anomaly samples with high semantic realism. We propose AnomalyAgent, an anomaly synthesis agent with self-reflection, knowledge retrieval, and iterative refinement capabilities, aiming to generate realistic and diverse anomalies. Specifically, AnomalyAgent is equipped with five tools: Prompt Generation (PG), Image Generation (IG), Quality Evaluation (QE), Knowledge Retrieval (KR), and Mask Generation (MG), enabling closed-loop optimization. To improve decision-making and self-reflection, we construct structured trajectories from real anomaly images and design a two-stage training framework: supervised fine-tuning followed by reinfo
    
[^279]: 面向在线生成模型选择的混合贪心策略：在多样性感知的多臂老虎机中UCB是否必要？

    Mixture-Greedy for Online Generative Model Selection: Is UCB Necessary in Diversity-Aware Multi-Armed Bandits?

    [https://arxiv.org/abs/2603.21716](https://arxiv.org/abs/2603.21716)

    本文提出Mixture-Greedy策略，并发现在多样性感知的多臂老虎机生成模型选择问题中，UCB探索奖励并非必要——去除UCB反而收敛更快、样本效率更高、性能更好。

    

    在现代生成式AI中，如何在多个生成模型之间进行高效选择变得越来越重要，因为从次优模型中采样的代价很高。该问题可以被视为一个多臂老虎机（MAB）任务。在多样性感知的评估分数下，生成器的非退化混合可以优于任何单个模型，这使得该MAB设置与经典的最佳臂识别问题区别开来。先前的方法在混合目标中引入了置信上界（UCB）探索奖励。然而，在多个数据集和评估指标上，我们观察到UCB项始终会减慢收敛速度并降低样本效率。相比之下，一种简单的、不包含显式UCB类乐观主义的Mixture-Greedy（混合贪心）策略收敛更快，并取得了更好的性能，尤其是对于FID和Vendi这类广泛使用但难以构造紧致置信界的指标。我们提供了理论见解……

    arXiv:2603.21716v2 Announce Type: replace-cross  Abstract: Efficient selection among multiple generative models is increasingly important in modern generative AI, where sampling from suboptimal models is costly. This problem can be viewed as a multi-armed bandit (MAB) task. Under diversity-aware evaluation scores, a non-degenerate mixture of generators can outperform any individual model, distinguishing this MAB setting from classical best-arm identification. Prior approaches incorporate an Upper Confidence Bound (UCB) exploration bonus into the mixture objective. However, across multiple datasets and evaluation metrics, we observe that the UCB term consistently slows convergence and reduces sample efficiency. In contrast, a simple Mixture-Greedy strategy without explicit UCB-type optimism converges faster and achieves even better performance, particularly for widely used metrics such as FID and Vendi where tight confidence bounds are difficult to construct. We provide theoretical insi
    
[^280]: DEAF：音频语言模型声学忠实度的诊断评估基准

    DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models

    [https://arxiv.org/abs/2603.18048](https://arxiv.org/abs/2603.18048)

    提出DEAF基准，通过2,700多个冲突刺激和多层次受控评估框架，诊断音频多模态大语言模型是真正处理声学信号还是依赖文本推断，发现七个模型均呈现文本主导的一致模式。

    

    最近的音频多模态大语言模型在语音基准测试中展现出令人瞩目的性能，然而这些模型究竟是真正处理声学信号，还是依赖于基于文本的语义推断，仍不清楚。为了系统地研究这一问题，我们提出了DEAF（声学忠实度诊断评估）基准，该基准包含超过2,700个冲突刺激，涵盖三个声学维度：情感韵律、背景声音和说话人身份。随后，我们设计了一个受控的多层次评估框架，逐步增加文本影响的程度，从内容中的语义冲突到误导性提示及其组合，使我们能够将内容驱动的偏差与提示诱发的谄媚行为区分开来。我们进一步引入了诊断指标，以量化模型对文本线索相对于声学信号的依赖程度。我们对七个音频多模态大语言模型的评估揭示了一致的文本主导模式。

    arXiv:2603.18048v3 Announce Type: replace  Abstract: Recent Audio Multimodal Large Language Models (Audio MLLMs) demonstrate impressive performance on speech benchmarks, yet it remains unclear whether these models genuinely process acoustic signals or rely on text-based semantic inference. To systematically study this question, we introduce DEAF (Diagnostic Evaluation of Acoustic Faithfulness), a benchmark of over 2,700 conflict stimuli spanning three acoustic dimensions: emotional prosody, background sounds, and speaker identity. Then, we design a controlled multi-level evaluation framework that progressively increases textual influence, ranging from semantic conflicts in the content to misleading prompts and their combination, allowing us to disentangle content-driven bias from prompt-induced sycophancy. We further introduce diagnostic metrics to quantify model reliance on textual cues over acoustic signals. Our evaluation of seven Audio MLLMs reveals a consistent pattern of text dom
    
[^281]: NeuronSpark：面向语言建模的具有自适应计算的选择性脉冲状态空间模型

    NeuronSpark: Selective Spiking State Space Models with Adaptive Computation for Language Modeling

    [https://arxiv.org/abs/2603.16148](https://arxiv.org/abs/2603.16148)

    提出 NeuronSpark，一个无需自注意力的选择性脉冲状态空间语言模型，通过让每个神经元的衰减率、增益和发放阈值随输入词元内容自适应调整，解决了标准脉冲神经元统一发放敏感性所导致的密集或静默发放问题。

    

    脉冲神经网络有望实现类脑的神经动力学，然而与转换、蒸馏和主干适配等方法相比，原生的大规模脉冲语言模型预训练在很大程度上仍未被探索。我们诊断出一个核心障碍：标准的脉冲神经元对每个词元施加相同的发放敏感性，从而产生密集发放或静默发放的状态，仅靠全局调参难以解决这一问题。这个问题在语言建模中尤为突出，因为单个序列中包含信息角色差异极大的词元，使得统一的发放机制与数据特性不匹配。我们提出 NeuronSpark 来解决这一问题，这是一个无需自注意力的语言模型，其核心是基于输入依赖的脉冲膜动力学，其中每个神经元的衰减率、增益和发放阈值都会根据当前词元内容进行自适应调整。结合门控联想记忆与自适应时间聚合机制，这种选择性设计……（原文摘要在此处被截断）

    arXiv:2603.16148v2 Announce Type: replace  Abstract: Spiking neural networks promise brain-inspired neural dynamics, yet native large-scale spiking language-model pretraining remains largely unexplored relative to conversion, distillation, and backbone-adaptation approaches. We diagnose a central obstacle: standard spiking neurons apply the same firing sensitivity to every token, creating dense- or silent-firing regimes that are difficult to resolve with globally tuned parameters alone. This problem is particularly acute in language modeling, where a single sequence contains tokens of vastly different informational roles, making a uniform firing regime poorly matched to the data. We address this with NeuronSpark, a self-attention-free language model built around input-dependent spiking membrane dynamics, where each neuron's decay rate, gain, and firing threshold adapt to the current token content. Together with gated associative memory and adaptive temporal aggregation, this selectivit
    
[^282]: AutoControl Arena：面向前沿AI风险评估的可执行测试环境自动合成框架

    AutoControl Arena: Synthesizing Executable Test Environments for Frontier AI Risk Evaluation

    [https://arxiv.org/abs/2603.07427](https://arxiv.org/abs/2603.07427)

    该论文提出AutoControl Arena框架，通过“逻辑-叙事解耦”原则将确定性状态交由可执行代码处理、生成性动态交由LLM承担，从而自动合成可执行测试环境以缓解幻觉并高效评估前沿AI风险，同时通过X-Bench揭示了前沿模型在压力下风险率从21.7%激增至54.5%的“对齐幻觉”现象。

    

    随着大语言模型（LLM）演变为自主智能体，现有的安全评估面临一个根本性权衡：人工基准成本高昂，而基于LLM的模拟器虽具备可扩展性，却存在逻辑幻觉问题。我们提出了AutoControl Arena，一个基于“逻辑-叙事解耦”原则构建的前沿AI风险自动评估框架。通过将确定性状态锚定于可执行代码，同时将生成性动态委托给LLM，我们在保持灵活性的同时缓解了幻觉问题。该原则通过一个三智能体框架得以实例化，实现了超过98%的端到端成功率，并在人类偏好上比现有模拟器高出60%。为激发潜在风险，我们在X-Bench（涵盖70个场景、7类风险类别）中改变环境的压力与诱惑水平。对9个前沿模型的评估揭示：（1）对齐幻觉：在压力之下，风险率从21.7%激增至54.5%，且能力更强的模型表现出不成比例的……（摘要在此处被截断）

    arXiv:2603.07427v3 Announce Type: replace  Abstract: As Large Language Models (LLMs) evolve into autonomous agents, existing safety evaluations face a fundamental trade-off: manual benchmarks are costly, while LLM-based simulators are scalable but suffer from logic hallucination. We present AutoControl Arena, an automated framework for frontier AI risk evaluation built on the principle of logic-narrative decoupling. By grounding deterministic state in executable code while delegating generative dynamics to LLMs, we mitigate hallucination while maintaining flexibility. This principle, instantiated through a three-agent framework, achieves over 98% end-to-end success and 60% human preference over existing simulators. To elicit latent risks, we vary environmental Stress and Temptation across X-Bench (70 scenarios, 7 risk categories). Evaluating 9 frontier models reveals: (1) Alignment Illusion: risk rates surge from 21.7% to 54.5% under pressure, with capable models showing disproportiona
    
[^283]: CORE-Seg：基于强化学习的复杂病灶推理驱动分割

    CORE-Seg: Reasoning-Driven Segmentation for Complex Lesions via Reinforcement Learning

    [https://arxiv.org/abs/2603.05911](https://arxiv.org/abs/2603.05911)

    该论文提出了首个面向推理驱动复杂病灶分割的思维链基准数据集ComLesion-14K，并构建了通过语义引导提示适配器将推理与分割相融合、采用SFT到GRPO渐进式强化学习训练的端到端框架CORE-Seg，推动医学图像分割从模式匹配向认知推理的范式转变。

    

    医学图像分割正在经历从传统视觉模式匹配到认知推理分析的范式转变。尽管多模态大语言模型（MLLMs）在融合语言与视觉知识方面已展现出前景，但仍存在显著差距：现有的通用多模态大模型具备广泛的常识，却缺乏复杂病灶所需的专门视觉推理能力；而传统分割模型擅长像素级分割，却缺乏逻辑可解释性。在本文中，我们提出了ComLesion-14K，这是首个面向推理驱动复杂病灶分割的多样化思维链基准数据集。为完成该任务，我们提出CORE-Seg，一个通过语义引导提示适配器将推理与分割相集成的端到端框架。我们设计了从SFT到GRPO的渐进式训练策略，并配备自适应双粒度奖励机制，以缓解奖励……

    arXiv:2603.05911v2 Announce Type: replace-cross  Abstract: Medical image segmentation is undergoing a paradigm shift from conventional visual pattern matching to cognitive reasoning analysis. Although Multimodal Large Language Models (MLLMs) have shown promise in integrating linguistic and visual knowledge, significant gaps remain: existing general MLLMs possess broad common sense but lack the specialized visual reasoning required for complex lesions, whereas traditional segmentation models excel at pixel-level segmentation but lack logical interpretability. In this paper, we introduce ComLesion-14K, the first diverse Chain-of-Thought (CoT) benchmark for reasoning-driven complex lesion segmentation. To accomplish this task, we propose CORE-Seg, an end-to-end framework integrating reasoning with segmentation through a Semantic-Guided Prompt Adapter. We design a progressive training strategy from SFT to GRPO, equipped with an adaptive dual-granularity reward mechanism to mitigate reward 
    
[^284]: 智能体对等网络：从内容分发到能力与行动共享

    Agentic Peer-to-Peer Networks: From Content Distribution to Capability and Action Sharing

    [https://arxiv.org/abs/2603.03753](https://arxiv.org/abs/2603.03753)

    该论文提出了智能体对等网络这一新范式，设计了一种将连接性、语义发现与执行解耦的基于平面的参考架构，并引入带签名的软状态能力描述符，以支持边缘设备上客户端自主代理之间安全的能力共享与协作。

    

    人工智能（AI）正从集中式云应用程序编程接口（API）向边缘设备本地推理持续转变，这使得客户端自主代理（CSAAs）成为可能——这类持久的个人代理能够进行规划、访问本地上下文并代表用户调用工具。当这些代理直接协作时，它们便形成了智能体对等网络。与交换静态的、以哈希索引内容的经典覆盖网络（如BitTorrent）不同，智能体P2P网络交换的是动态的、异构的且可能不安全的能力，因此需要语义发现、漂移管理和风险感知执行。我们概述了实现这种协作的网络基础。为了组织这一设计空间，我们提出了一种基于平面的参考架构，将连接性、语义发现和执行三者解耦。为支持漂移环境下的发现，我们引入了带签名的软状态能力描述符，以支持意图……

    arXiv:2603.03753v2 Announce Type: replace-cross  Abstract: The ongoing shift of Artificial Intelligence (AI) from centralized cloud application programming interfaces (APIs) to local inference on edge devices is enabling Client-Side Autonomous Agents (CSAAs) - persistent personal agents that can plan, access local context, and invoke tools on behalf of users. As these agents collaborate directly, they form Agentic Peer-to-Peer (P2P) Networks. Unlike classic overlays exchanging static, hash-indexed content (e.g., BitTorrent), agentic P2P exchanges dynamic, heterogeneous capabilities that are potentially unsafe, requiring semantic discovery, drift management, and risk-aware execution. We outline the networking foundations for such collaboration. To organize this design space, we propose a plane-based reference architecture decoupling connectivity, semantic discovery, and execution. To support discovery under drift, we introduce signed, soft-state capability descriptors to support intent-
    
[^285]: ITO：面向图文预训练的多视图对齐与训练时融合

    ITO: Multi-View Alignment and Training-Time Fusion for Image-Text Pretraining

    [https://arxiv.org/abs/2603.02767](https://arxiv.org/abs/2603.02767)

    ITO通过基于多视图增广的多模态多重对齐丰富监督信号，并引入仅在训练阶段使用的轻量级多模态融合模块作为正则化器（推理时丢弃），使图文预训练得到的表示更多按语义而非模态组织，同时保持双编码器架构的推理效率。

    

    图文对比预训练已成为视觉表示学习的主导范式，然而现有方法产生的表示往往仍在一定程度上按模态而非按语义进行组织。我们提出ITO，一个通过两种角色不同、相互补充的机制来解决这一局限的框架。多模态多重对齐通过从多视图图像增广中构建多样的跨模态对应关系来丰富监督信号，提供了判别性增益的主要来源。随后，一个轻量级的训练时多模态融合模块充当训练时正则化器，促使编码器产生在融合下兼容的特征。关键在于，该融合模块在推理时被丢弃，从而在仅增加训练时开销的同时，保留了标准双编码器架构的高效性。研究在从数百万到数十亿规模的预训练上进行了大量实验。

    arXiv:2603.02767v4 Announce Type: replace-cross  Abstract: Image--text contrastive pretraining has become a dominant paradigm for visual representation learning, yet existing methods often yield representations that remain partially organized by modality rather than by semantics. We propose ITO, a framework addressing this limitation through two complementary mechanisms with distinct roles. Multimodal multiple alignment enriches supervision by constructing diverse cross-modal correspondences from multi-view image augmentations, providing the primary source of discriminative gain. A lightweight training-time multimodal fusion module then acts as a training-time regularizer, encouraging the encoders to produce features that are compatible under fusion. Crucially, the fusion module is discarded at inference, preserving the efficiency of standard dual-encoder architectures while incurring training-time overhead only. Extensive experiments across pretraining scales from millions to billions
    
[^286]: 关于非线性神经网络中梯度下降收敛速度的研究：对抗鲁棒性视角

    On the Rate of Convergence of GD in Non-linear Neural Networks: An Adversarial Robustness Perspective

    [https://arxiv.org/abs/2603.02095](https://arxiv.org/abs/2603.02095)

    该论文在双神经元ReLU网络的最简二分类设置中证明，梯度下降虽能收敛到最优对抗鲁棒性边际，但收敛速度极慢（严格为 Θ(1/ln(t))），并首次建立了非线性模型中鲁棒性边际收敛速度的显式下界。

    

    我们在一个极简的二分类设置中研究梯度下降（GD）的收敛动态，该设置由一个双神经元ReLU网络和两个训练样本组成。我们证明，即使在这些强简化假设下，虽然GD能够成功收敛到最优鲁棒性边际，有效地最大化决策边界与训练点之间的距离，但这一收敛以极其缓慢的速率发生，严格以 Θ(1/ln(t)) 的速度缩放。据我们所知，这建立了非线性模型中鲁棒性边际收敛速度的首个显式下界。通过经验模拟，我们进一步证明这种固有的失效模式是普遍存在的，在多个自然网络初始化下均表现出完全相同的紧致收敛速率。我们的理论保证是通过对GD在不同激活机制下轨迹的严格分析得出的。

    arXiv:2603.02095v2 Announce Type: replace-cross  Abstract: We study the convergence dynamics of Gradient Descent (GD) in a minimal binary classification setting, consisting of a two-neuron ReLU network and two training instances. We prove that even under these strong simplifying assumptions, while GD successfully converges to an optimal robustness margin, effectively maximizing the distance between the decision boundary and the training points, this convergence occurs at a prohibitively slow rate, scaling strictly as $\Theta(1/\ln(t))$. To the best of our knowledge, this establishes the first explicit lower bound on the convergence rate of the robustness margin in a non-linear model. Through empirical simulations, we further demonstrate that this inherent failure mode is pervasive, exhibiting the exact same tight convergence rate across multiple natural network initializations. Our theoretical guarantees are derived via a rigorous analysis of the GD trajectories across the distinct act
    
[^287]: 超越逐状态镜像下降：基于参数化策略的离线策略优化

    Beyond State-Wise Mirror Descent: Offline Policy Optimization with Parametric Policies

    [https://arxiv.org/abs/2602.23811](https://arxiv.org/abs/2602.23811)

    该工作将离线强化学习的理论保证从有限小动作空间扩展到大/连续动作空间上的参数化策略类，识别出将镜像下降推广至参数化策略时的核心难点“上下文耦合”，并据此提出了计算上高效（oracle高效）的独立参数化离线策略优化算法。

    

    我们研究了在一般函数逼近下离线强化学习（RL）的理论问题。尽管先前的工作（如 Xie et al., 2021）已通过悲观原则确立了从离线数据中学习良好策略的理论基础，但现有在计算上可行的算法（通常在 oracle 高效的意义上），例如 PSPI，仅适用于有限且较小的动作空间。此外，这些算法依赖于逐状态的镜像下降，并要求策略执行者（actor）从评论者（critic）函数中隐式导出，因而无法适应实践中普遍存在的独立策略参数化方式。在本工作中，我们解决了这些局限性，将理论保证扩展到大动作空间或连续动作空间上的参数化策略类。在将镜像下降扩展到参数化策略时，我们识别出上下文耦合是其中的核心难点，并展示了如何通过将镜像下降与……（原文摘要在此处截断）

    arXiv:2602.23811v5 Announce Type: replace-cross  Abstract: We investigate the theoretical aspects of offline reinforcement learning (RL) under general function approximation. While prior works (e.g., Xie et al., 2021) have established the theoretical foundations of learning a good policy from offline data via pessimism, existing algorithms that are computationally tractable (often in an oracle-efficient sense), such as PSPI, only apply to finite and small action spaces. Moreover, these algorithms rely on state-wise mirror descent and require actors to be implicitly induced from the critic functions, failing to accommodate standalone policy parameterization which is ubiquitous in practice. In this work, we address these limitations and extend the theoretical guarantees to parameterized policy classes over large or continuous action spaces. When extending mirror descent to parameterized policies, we identify contextual coupling as the core difficulty, and show how connecting mirror desce
    
[^288]: CoMMa：面向去中心化肿瘤学决策支持的贡献感知医疗多智能体框架

    CoMMa: Contribution-Aware Medical Multi-Agents for Decentralized Oncology Decision Support

    [https://arxiv.org/abs/2602.09159](https://arxiv.org/abs/2602.09159)

    CoMMa通过去中心化数据访问和贡献感知聚合机制，实现了隐私保护下的肿瘤学多智能体决策支持，并用确定性嵌入投影替代随机推理，提供可解释的信用分配。

    

    摘要：近期多智能体框架在肿瘤学决策支持方面展现出潜力，但大多数框架假设集中式数据访问，并依赖基于提示的任务分配，这限制了它们在隐私敏感的临床环境中的适用性。我们提出了贡献感知医疗多智能体（CoMMa），一种去中心化的LLM智能体框架，其中专家智能体在分区临床数据流上运行。与先前跨智能体共享输入的方法不同，CoMMa强制数据去中心化以增强角色专业化，并通过智能体特定微调进一步强化这一点。为实现可靠且可解释的协调，我们引入了一种贡献感知聚合机制，该机制用确定性嵌入投影替代随机、叙事性推理，以近似每个智能体的边际效用。这产生了对智能体的显式信用分配，提供了与临床实践一致的稳定、可解释的决策路径。

    arXiv:2602.09159v2 Announce Type: replace  Abstract: Recent multi-agent frameworks have shown promise for oncology decision support, yet most assume centralized data access and rely on prompt-based assignment, limiting their applicability in privacy-sensitive clinical settings. We propose Contribution-Aware Medical Multi-Agents (CoMMa), a decentralized LLM-agent framework where specialists operate on partitioned clinical data streams. Unlike prior approaches that share inputs across agents, CoMMa enforces data decentralization to include stronger role specialization and further enhances this via agent-specific finetuning. To enable reliable and interpretable coordination, we introduce a contribution-aware aggregation mechanism that replaces stochastic, narrative-based reasoning with deterministic embedding projections to approximate each agent's marginal utility. This yields explicit credit assignment over agents, providing a stable and interpretable decision pathway aligned with clini
    
[^289]: COBALT：基于离线轨迹的上下文老虎机学习用于多轮代码生成

    COBALT: Contextual Bandit Learning with Offline Trajectories for Multi-Turn Code Generation

    [https://arxiv.org/abs/2602.03806](https://arxiv.org/abs/2602.03806)

    COBALT 利用多轮代码生成是单步可恢复马尔可夫决策过程的观察，通过重用参考模型收集的离线多轮轨迹并采用单步补全的上下文老虎机学习，替代昂贵且不稳定的在线强化学习来高效训练 LLM 进行多轮代码生成。

    

    训练大语言模型（LLM）进行多轮代码生成通常依赖在线强化学习（RL），即模型通过反复生成的在线 rollout 进行学习。虽然有效，但在线强化学习成本高昂且不稳定。本文提出了基于离线轨迹的上下文老虎机学习方法（COBALT），这是一种简单的方法，通过重用离线多轮轨迹来高效训练 LLM 进行多轮代码生成。基于多轮代码生成是一个单步可恢复马尔可夫决策过程的观察，COBALT 首先使用一个参考 LLM 收集代码生成轨迹，并将其划分为部分轨迹。然后，在在线老虎机学习过程中，训练 LLM 通过单步代码生成来补全每个部分轨迹。在 LiveCodeBench 上，COBALT 使 R1-Distill 8B 和 Qwen3 8B 的绝对 Pass@1 分数最多分别提升 9.0 和 6.2 个百分点，并匹敌了（原摘要在此处截断）。

    arXiv:2602.03806v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) for multi-turn code generation typically relies on online reinforcement learning (RL), in which models learn from repeatedly generated online rollouts. Although effective, online RL can be costly and unstable. In this paper, we present contextual bandit learning with offline trajectories (COBALT), a simple method that reuses offline multi-turn trajectories to efficiently train LLMs for multi-turn code generation. Building on the observation that multi-turn code generation is a one-step recoverable Markov decision process, COBALT first collects code generation trajectories using a reference LLM and divides them into partial trajectories. Then, during online bandit learning, the LLM is trained to complete each partial trajectory through single-step code generation. On LiveCodeBench, COBALT substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 points and matche
    
[^290]: 基于偏好条件的重型车辆战术决策多目标强化学习

    Preference-Conditioned Multi-Objective Reinforcement Learning for Tactical Decision Making in Heavy-Duty Vehicles

    [https://arxiv.org/abs/2601.18783](https://arxiv.org/abs/2601.18783)

    提出一种基于偏好条件的多目标强化学习框架，结合广义策略改进与近端策略优化，在安全性、能效和时间三个冲突目标之间逼近帕累托前沿，从而实现重型车辆的战术决策优化。

    

    将强化学习应用于自动驾驶等现实世界场景的一个核心难点在于，通过聚合多个通常相互冲突的目标而得到的传统标量奖励设计，可能会掩盖这些目标之间权衡关系的结构。这些挑战对于重型车辆的决策制定尤为突出，因为重型车辆需要在严苛的商业需求下平衡安全性、效率和运营成本。我们提出了一种基于偏好条件的多目标强化学习框架，该框架显式地表示这些目标，并利用广义策略改进和近端策略优化来学习能够捕捉这些目标之间不同权衡策略组合的策略。所提出的方法在三个相互冲突的目标之间逼近帕累托前沿：安全性（以碰撞和成功完成任务来量化）、能效（以能耗成本来量化）以及时间（原文摘要此处截断）。

    arXiv:2601.18783v3 Announce Type: replace-cross  Abstract: A central difficulty in applying reinforcement learning to real-world applications such as autonomous driving is that conventional scalar reward formulations obtained by aggregating multiple often competing objectives may obscure the structure of their trade-offs. These challenges are particularly relevant to decision-making for heavy-duty vehicles, which requires balancing safety, efficiency, and operational costs under demanding commercial requirements. We present a preference-conditioned multi-objective reinforcement learning framework that explicitly represents these objectives and utilizes Generalized Policy Improvement and Proximal Policy Optimization to learn policies capturing different trade-offs between them. The proposed method approximates the Pareto front across three conflicting objectives: safety, quantified in terms of collisions and successful completion; energy efficiency, quantified using energy cost; and tim
    
[^291]: 自然语言声明集的全局一致性修复

    Global Consistency Repair for Natural-Language Claim Sets

    [https://arxiv.org/abs/2601.13600](https://arxiv.org/abs/2601.13600)

    该论文提出了QXR框架，将大语言模型作为带噪声的集合级一致性判断器，通过定位最小冲突子集并删除一个小的命中集声明，实现对不一致自然语言声明集的全局一致性修复，从而保留最大的连贯子集。

    

    自然语言处理（NLP）系统越来越多地从检索到的段落、文档和生成的推理步骤中聚合声明。然而，当前的事实性方法大多是对声明进行逐条验证或成对比较，而未能解决整个聚合声明集能否保持连贯的问题。这忽略了一个基本的失败模式：各条声明可能在局部看似合理且两两兼容，但合在一起却是不一致的。我们形式化了全局一致性修复问题：给定一个不一致的自然语言声明集，在保留最大连贯子集的同时，定位紧凑的不一致性来源。我们提出了QXR，一个自适应的诊断与修复框架，它将大语言模型（LLM）视为一个带噪声的集合级一致性判断器，定位最小冲突子集，并通过删除一个小的命中集（hitting set）声明来修复声明集。在多种LLM判断器以及基于VitaminC、FEVER和冲突证据RAG构建的基准上，QXR保持……

    arXiv:2601.13600v2 Announce Type: replace  Abstract: NLP systems increasingly aggregate claims from retrieved passages, documents, and generated reasoning steps. Yet current factuality methods mostly verify claims individually or compare pairs of claims, leaving open the question of whether the entire aggregated claim set can be retained coherently. This misses a fundamental failure mode: claims may be locally plausible and pairwise compatible while still being jointly inconsistent. We formalize global consistency repair: given an inconsistent natural-language claim set, preserve the largest coherent subset while localizing the compact sources of inconsistency. We introduce QXR, an adaptive diagnosis-and-repair framework that treats an LLM as a noisy set-level consistency judge, localizes minimal conflicting subsets, and repairs the claim set by deleting a small hitting set of claims. Across LLM judges and benchmarks derived from VitaminC, FEVER, and conflicting-evidence RAG, QXR prese
    
[^292]: HAFixAgent：历史感知的智能体化程序修复

    HAFixAgent: History-Aware Agentic Program Repair

    [https://arxiv.org/abs/2511.01047](https://arxiv.org/abs/2511.01047)

    HAFixAgent通过将基于git blame归因的仓库历史上下文注入修复循环，显著提升了基于大语言模型的自动程序修复效果，对较弱模型的提升最高达64.3%，并大幅超越RepairAgent和BIRCH-feedback等现有最优智能体方法。

    

    自动程序修复（APR）近来已转向大语言模型和基于智能体的系统，而先前的研究表明，仓库历史可以帮助修复单行缺陷。我们开展了一项大规模实证研究，探讨在超越这一设定的更广泛场景下，仓库历史是否、在多大程度上以及在何种条件下能够改进基于智能体的APR。我们提出了HAFixAgent，一个历史感知的程序修复智能体，它将基于blame归因分析获得的仓库上下文注入其修复循环中。在854个Defects4J（Java）缺陷和501个BugsInPy（Python）缺陷上，我们首先证明了与缺陷相关的历史信息是广泛可获得的。随后，我们在完美故障定位条件下，基于三种基础模型对HAFixAgent进行了评估。最佳的历史启发式方法相比不使用历史信息的方法，在使用最强模型时在Defects4J上提升+4.4%、在BugsInPy上提升+1.1%，而在较弱模型上提升高达+64.3%。在最强模型上，HAFixAgent还比RepairAgent高出+56.6%，比BIRCH-feedback高出+47.1%。

    arXiv:2511.01047v4 Announce Type: replace-cross  Abstract: Automated program repair (APR) has recently shifted toward large language models and agent-based systems, while prior work shows that repository history can help repair single-line bugs. We conduct a large-scale empirical study of whether, how much, and under what conditions repository history improves agentic APR beyond this setting. We present HAFixAgent, a history-aware program repair agent that injects blame-derived repository context into its repair loop. Across 854 Defects4J (Java) and 501 BugsInPy (Python) bugs, we first show that bug-relevant history is widely available. We then evaluate HAFixAgent across three base models under perfect fault localization. The best history heuristic improves over non-history from +4.4% on Defects4J and +1.1% on BugsInPy with the strongest model to as much as +64.3% with weaker models. HAFixAgent also outperforms RepairAgent by +56.6% and BIRCH-feedback by +47.1% on the strongest model, 
    
[^293]: 信息力学理论：通过不确定性测量实现泛化（学习即测量）

    A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)

    [https://arxiv.org/abs/2510.22809](https://arxiv.org/abs/2510.22809)

    提出了一种基于信息论不确定性（惊奇度）的“无模型”学习框架，将学习视为对不确定性的测量，可直接从原始数据进行推理，支持数据的编辑与删除，并在异常检测等任务上达到最先进性能。

    

    传统机器学习依赖显式模型和领域假设，这常常限制了灵活性和可解释性。我们引入了一种“无模型”的基于实例的学习框架，利用惊奇度（surprisal，即信息论中的不确定性）直接对原始数据进行分析和推理，从而消除了分布建模的需求，减少了偏差，并支持高效的更新操作，包括直接编辑和删除训练数据。通过以不确定性来量化相关性，该方法能够在多种任务上实现可泛化的推理，包括生成式推理、因果发现、异常检测和时间序列预测。我们的方法强调可追溯性、可解释性和数据驱动的决策，为机器学习提供了一个统一的、人类可理解的框架，几乎可以适用于所有机器学习任务，在所有任务上均表现出色，并在异常检测上达到了最先进的性能。

    arXiv:2510.22809v2 Announce Type: replace-cross  Abstract: Traditional machine learning relies on explicit models and domain assumptions, often limiting flexibility and interpretability. We introduce a "model-free" framework of instance-based learning using surprisal (information theoretic uncertainty) to directly analyze and perform inferences from raw data, eliminating distribution modeling, reducing bias, and enabling efficient updates including direct edits and deletion of training data. By quantifying relevance through uncertainty, the approach enables generalizable inference across tasks including generative inference, causal discovery, anomaly detection, and time series forecasting. Our approach emphasizes traceability, interpretability, and data-driven decision making, offering a unified, human-understandable framework for machine learning, can be adapted to virtually all machine learning tasks, achieving competency across all and state-of-the-art performance on anomaly detecti
    
[^294]: ABLEIST：LLM生成的招聘场景中的交叉性残障偏见

    ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios

    [https://arxiv.org/abs/2510.10998](https://arxiv.org/abs/2510.10998)

    该论文提出了ABLEIST评估框架，通过五个健全主义伤害指标和三个交叉性伤害指标，对六个LLM在2,820个招聘场景中的表现进行审计，揭示了针对残障候选人（尤其在性别、种姓等交叉身份维度）的显著偏见伤害，而许多先进模型无法检测到这些伤害。

    

    大型语言模型（LLM）因在招聘等高风险领域延续基于身份的歧视而日益受到审视，尤其是针对残障人士（PwD）的歧视。然而，现有研究仍主要聚焦于西方视角，忽视了性别和种姓等交叉性边缘化形式如何影响全球南方残障人士的经历。我们对六个LLM在2,820个招聘场景中进行了全面审计，这些场景涵盖多样的残障、性别、国籍和种姓特征。为了捕捉微妙的交叉性伤害与偏见，我们引入了ABLEIST（健全主义、励志化、超人化与象征化）框架，这是一套基于残障研究文献构建的评估体系，包含五个健全主义特定伤害指标和三个交叉性伤害指标。我们的结果显示，针对残障候选人的ABLEIST伤害显著增加——而许多最先进的模型未能检测到这些伤害。（摘要在此处截断）

    arXiv:2510.10998v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly under scrutiny for perpetuating identity-based discrimination in high-stakes domains such as hiring, particularly against people with disabilities (PwD). However, existing research remains largely Western-centric, overlooking how intersecting forms of marginalization--such as gender and caste--shape experiences of PwD in the Global South. We conduct a comprehensive audit of six LLMs across 2,820 hiring scenarios spanning diverse disability, gender, nationality, and caste profiles. To capture subtle intersectional harms and biases, we introduce ABLEIST (Ableism, Inspiration, Superhumanization, and Tokenism), a set of five ableism-specific and three intersectional harm metrics grounded in disability studies literature. Our results reveal significant increases in ABLEIST harms towards disabled candidates--harms that many state-of-the-art models failed to detect. These harms were furthe
    
[^295]: 发现学习到的哈密顿系统中的相空间结构

    Discovering Phase Space Structure in Learned Hamiltonian Systems

    [https://arxiv.org/abs/2509.23003](https://arxiv.org/abs/2509.23003)

    提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。

    

    力学启发的神经动力学模型提升了可解释性和物理一致性，但它们通常假设构型空间已知、物理参数固定，并且从给定初始条件出发进行确定性预测。我们提出结构化相空间生成对抗网络（SPS-GAN），这是一种条件生成模型，能够同时从笛卡尔轨迹和视频中学习结构化动力学。SPS-GAN 将端口哈密顿骨干网络与对抗式条件生成相结合，使保守系统、耗散系统和受迫系统能够在单一架构中得到表示。为了从原始观测中推断潜在的力学结构，我们采用循环坐标损失，在无需访问真实相空间的情况下同时恢复系统的自由度和正则坐标。在模拟系统和真实世界系统上的实验表明，SPS-GAN 在轨迹精度上优于监督动力学基线方法，并提升了视频一致性。

    arXiv:2509.23003v3 Announce Type: replace-cross  Abstract: Mechanics-inspired neural dynamical models improve interpretability and physical consistency, but they usually assume a known configuration space, fixed physical parameters, and deterministic prediction from a prescribed initial condition. We introduce Structured Phase Space GAN (SPS-GAN), a conditional generative model that learns structured dynamics from both Cartesian trajectories and video. SPS-GAN combines a port-Hamiltonian backbone with adversarial conditional generation, allowing conservative, dissipative, and forced systems to be represented in a single architecture. To infer latent mechanical structure from raw observations, we use a cyclic-coordinate loss that recovers both the system degrees of freedom and canonical coordinates without access to the true phase space. Experiments on simulated and real-world systems show that SPS-GAN improves trajectory accuracy over supervised dynamics baselines and video consistency
    
[^296]: 面向大语言模型高效排序的访问路径

    Access Paths for Efficient Ordering with Large Language Models

    [https://arxiv.org/abs/2509.00303](https://arxiv.org/abs/2509.00303)

    提出了LLM ORDER BY语义算子及预算感知优化器，通过动态选择近优访问路径，在所有基准测试中实现了与最佳静态方法相当或更优的排序准确率。

    

    在这项工作中，我们提出了 \texttt{LLM ORDER BY} 语义算子作为一种逻辑抽象，并对其物理实现进行了系统性研究。首先，我们对现有的语义排序算法提出了若干改进，并引入了一种语义感知的外部归并排序算法。我们的大量评估表明，没有任何单一实现在所有数据集上都能达到普遍最优。从评估中，我们观察到基于比较的算法在排序成本与排序质量之间存在普遍的扩展关系。基于这些洞察，我们设计了一个预算感知的优化器，该优化器利用启发式规则、LLM-as-Judge 评估和共识聚合来动态选择 LLM ORDER BY 的近优访问路径。在我们的大量评估中，我们的优化器在所有基准测试中始终实现了与最佳静态方法相当或更优的排序准确率。我们相信，这

    arXiv:2509.00303v4 Announce Type: replace-cross  Abstract: In this work, we present the \texttt{LLM ORDER BY} semantic operator as a logical abstraction and conduct a systematic study of its physical implementations. First, we propose several improvements to existing semantic sorting algorithms and introduce a semantic-aware external merge sort algorithm. Our extensive evaluation reveals that no single implementation offers universal optimality on all datasets. From our evaluations, we observe a general scaling relationship between sorting cost and the ordering quality for comparison-based algorithms. Building on these insights, we design a budget-aware optimizer that utilizes heuristic rules, LLM-as-Judge evaluation, and consensus aggregation to dynamically select the near-optimal access path for LLM ORDER BY. In our extensive evaluations, our optimizer consistently achieves ranking accuracy on par with or superior to the best static methods across all benchmarks. We believe that this
    
[^297]: 迈向语义信息的测度论

    Towards a Measure Theory of Semantic Information

    [https://arxiv.org/abs/2508.00525](https://arxiv.org/abs/2508.00525)

    本文批判了Floridi未能解决Bar-Hillel-Carnap悖论的强语义信息理论，并提出了一种基于单位圆和von Neumann量子概率类比的新方法，成功构建了信息量的测度空间。

    

    关于语义信息量化的经典论述来自Bar-Hiller和Carnap。他们的观点提出陈述的信息量与其概率之间呈反比关系。然而，他们的方法将最大信息量分配给了矛盾命题——Floridi将其称为Bar-Hillel-Carnap悖论。Floridi基于距离度量和抛物线关系开发了一种新理论，旨在消除这一悖论，但遗憾的是他的方法未能实现这一目标。本文从理论自身的角度对Floridi的强语义信息理论进行了批判性分析，指出了其成功与失败之处。随后，我提出了一种基于单位圆的新方法（单位圆这一关系一直是从基础三角学到量子理论等诸多理论的基础）。通过与von Neumann量子概率的类比，该方法被用于构建一个满足所有既定要求的信息量测度空间。

    arXiv:2508.00525v2 Announce Type: replace-cross  Abstract: A classic account of the quantification of semantic information is that of Bar-Hiller and Carnap. Their account proposes an inverse relation between the informativeness of a statement and its probability. However, their approach assigns the maximum informativeness to a contradiction: which Floridi refers to as the Bar-Hillel-Carnap paradox. He developed a novel theory founded on a distance metric and parabolic relation, designed to remove this paradox. Unfortunately is approach does not succeed in that aim.   In this paper I critique Floridi's theory of strongly semantic information on its own terms and show where it succeeds and fails. I then present a new approach based on the unit circle (a relation that has been the basis of theories from basic trigonometry to quantum theory). This is used, by analogy with von Neumann's quantum probability to construct a measure space for informativeness that meets all the requirements stip
    
[^298]: DeltaLLM：面向边缘大语言模型的Delta键稀疏注意力

    DeltaLLM: Delta-Key Sparse Attention for Edge LLMs

    [https://arxiv.org/abs/2507.19608](https://arxiv.org/abs/2507.19608)

    DeltaLLM是一个无需训练的边缘端LLM加速框架，通过算法-硬件-数据流协同设计，利用因果感知的K-only delta表示实现GQA下的Delta-Key跨查询头复用，并结合上下文感知的混合策略在对角线附近保留精确注意力，从而加速长提示词预填充阶段的注意力计算。

    

    设备端LLM推理对边缘平台而言日益重要，但由于高计算与内存需求以及严格的资源限制，仍面临诸多挑战。现代应用通常涉及长提示词。尽管紧凑模型通过权重压缩降低了线性层的开销，但注意力机制在很大程度上仍未被优化，成为主要瓶颈，尤其是在预填充阶段。现有的注意力剪枝方法主要面向高度并行的硬件（如GPU）和长上下文窗口，限制了其在边缘场景中的适用性。我们提出了DeltaLLM，这是一个无需训练的LLM框架，基于算法-硬件-数据流协同设计来加速预填充阶段的注意力计算。DeltaLLM引入了一种因果感知的仅K（K-only）delta表示方法，使其能够在分组查询注意力（GQA）机制下跨查询头实现Delta-Key的复用。同时，一种上下文感知的混合策略在对角线附近保留精确注意力，并在其他区域应用delta近似，以降低计算与访存开销，从而适应边缘设备严苛的资源约束。

    arXiv:2507.19608v2 Announce Type: replace  Abstract: On-device LLM inference is increasingly important for edge platforms but remains challenging due to high computational and memory demands and tight resource constraints. Modern applications often involve long prompts. Although compact models reduce linear-layer costs through weight compression, attention remains largely unoptimized and becomes the dominant bottleneck, especially during prefilling. Existing attention pruning methods primarily target highly parallel hardware (e.g., GPUs) and long context windows, limiting their suitability for edge scenarios. We present DeltaLLM, a training-free LLM framework based on algorithm-hardware-dataflow co-design to accelerate prefilling attention. DeltaLLM introduces a causal-attention-aware K-only delta formulation that enables Delta-Key reuse across query heads under Grouped-Query Attention (GQA). A context-aware hybrid strategy preserves exact attention near the diagonal while applying del
    
[^299]: 从流畅性谬误到微观-宏观效度鸿沟：大语言模型在社会模拟中的机遇与陷阱

    From the Fluency Fallacy to the Micro-to-Macro Validity Gap: Opportunities and Pitfalls of LLMs in Social Simulation

    [https://arxiv.org/abs/2507.19364](https://arxiv.org/abs/2507.19364)

    本文批判性评述了大语言模型在社会模拟中的应用，提出“微观-宏观效度鸿沟”框架，指出LLMs在个体层面的幻觉与随机不一致等局限会在多智能体社会中被放大为系统性脆弱性。

    

    将大语言模型（LLMs）整合到社会模拟中引发了相当大的热情，但也带来了重大的方法论和认识论挑战。这篇批判性综述从计算社会科学的视角，审视了将LLMs用作模拟智能体的认知或决策组件的做法。我们主张不应将LLMs的心理评估与模拟机制割裂开来，而是认为它们在行为和认知层面的局限性在扩展到多智能体社会时可能会成为系统性脆弱性。我们首先描绘了快速发展的LLM驱动平台的图景，涵盖从小型叙事沙盒到人口规模和空间结构化的模拟。然后，我们建立了一个统一的批判性分析框架来解析“微观-宏观效度鸿沟”：即幻觉、随机不一致性、代表性不足等微观层面局限性在社会层面的传播与放大（摘要在此处被截断）。

    arXiv:2507.19364v3 Announce Type: replace  Abstract: The integration of Large Language Models (LLMs) into social simulation has generated considerable enthusiasm, but also raises substantial methodological and epistemological challenges. This critical review examines the use of LLMs as cognitive or decision-making components of simulated agents from a computational social science perspective. Rather than treating the psychological evaluation of LLMs as separate from simulation mechanics, we argue that their behavioural and epistemic limitations can become systemic vulnerabilities when scaled to multi-agent societies. We first map the rapidly evolving landscape of LLM-driven platforms, ranging from small narrative sandboxes to population-scale and spatially structured simulations. We then develop a unified critical framework for analyzing the Micro-to-Macro Validity Gap: the propagation and amplification of micro-level limitations, including hallucinations, stochastic inconsistency, rep
    
[^300]: 针对Shapley值与概率值的回归调整蒙特卡洛估计器

    Regression-adjusted Monte Carlo Estimators for Shapley Values and Probabilistic Values

    [https://arxiv.org/abs/2506.11849](https://arxiv.org/abs/2506.11849)

    本文提出一种将蒙特卡洛采样与回归调整相结合的新方法来估计Shapley值等概率值，允许用任意可高效计算概率值的函数族（如XGBoost）替代线性回归，并通过交叉拟合实现无偏且性能领先的概率值估计。

    

    源于博弈论的概率值（如Shapley值、Banzhaf值和半值）已成为可解释人工智能领域的核心工具，被广泛应用于特征归因、数据归因、数据估值等任务。由于精确计算可能需要指数级时间，研究工作一直聚焦于两类高效近似技术：蒙特卡洛采样和线性回归建模。在本工作中，我们提出了一种将这两种技术相结合的新方法。与以往的算法相比，我们的方法更加灵活，允许将线性回归替换为任何其概率值可以被高效计算的函数族。这使得我们能够利用XGBoost等基于树的模型的准确性；通过交叉拟合，该方法还能产生无偏估计。在八个数据集上的实验表明，我们的方法在估计概率值方面达到了最先进的性能。

    arXiv:2506.11849v3 Announce Type: replace-cross  Abstract: With origins in game theory, probabilistic values like Shapley values, Banzhaf values, and semi-values have emerged as a central tool in explainable AI. They are used for feature attribution, data attribution, data valuation, and more. Since exact computation can require exponential time, research has focused on efficient approximation methods using two techniques: Monte Carlo sampling and linear regression formulations. In this work, we present a new way of combining both of these techniques. Our approach is more flexible than prior algorithms, allowing for linear regression to be replaced with any function family whose probabilistic values can be computed efficiently. This allows us to harness the accuracy of tree-based models like XGBoost; with cross-fitting, the method also produces unbiased estimates. From experiments across eight datasets, we find that our methods give state-of-the-art performance for estimating probabili
    
[^301]: LLMail-Inject：来自现实自适应提示注入挑战的数据集

    LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge

    [https://arxiv.org/abs/2506.09956](https://arxiv.org/abs/2506.09956)

    本文提出了LLMail-Inject，一个包含208,095个人工编写攻击提示的大规模基准测试，通过真实红队挑战收集而成，用于评估间接提示注入攻击的完整执行链，克服了现有小规模模板化基准饱和且过拟合的问题。

    

    间接提示注入攻击利用了大语言模型的一个根本弱点：执行来自不受信任来源的未授权任务。尽管这种攻击向量具有重要的实际意义，但现有基准测试依赖于小规模的基于模板的攻击集（100-1,000个示例），无法捕捉真实世界攻击者的多样性和适应性。因此，这些基准测试已经饱和，并对静态攻击产生了过拟合。我们提出了LLMail-Inject，这是一个大规模基准测试，包含通过一项有839名参与者参加的公开红队挑战收集的208,095个独特的人工编写的攻击提示。LLMail-Inject评估攻击的完整执行链：提示必须从电子邮件数据库中检索、规避生产级防御，并以正确的参数触发未授权的工具调用。评估结果显示，在现有基准测试中达到5%攻击成功率（ASR）的模型在LLMail-Inject上失败率更高：Meta SecAlign——

    arXiv:2506.09956v2 Announce Type: replace-cross  Abstract: Indirect prompt injection attacks exploit a fundamental weakness of LLMs: executing unauthorized tasks from untrusted sources. Despite the practical significance of this attack vector, existing benchmarks rely on small sets of template-based attacks (100-1,000 examples) that fail to capture the diversity and adaptiveness of real-world adversaries. Consequently, benchmarks are saturated and overfit to static attacks. We introduce LLMail-Inject, a large-scale benchmark of 208,095 distinct human-written attack prompts collected through a public red-teaming challenge with 839 participants. LLMail-Inject evaluates the attack's complete chain of execution: prompts must be retrieved from an email database, evade production-level defenses, and trigger unauthorized tool calls with correct arguments. Evaluation reveals that models that achieve 5% attack success rate (ASR) on existing benchmarks fail more on LLMail-Inject: Meta SecAlign--
    
[^302]: UniConFlow：一种用于可认证运动规划的统一约束流匹配框架

    UniConFlow: A Unified Constrained Flow-Matching Framework for Certified Motion Planning

    [https://arxiv.org/abs/2506.02955](https://arxiv.org/abs/2506.02955)

    本文提出UniConFlow，一个统一的约束流匹配轨迹生成框架，通过新颖的预设时间归零函数系统性地处理等式与不等式约束，实现可认证的机器人运动规划。

    

    生成式模型已成为机器人运动生成中日益强大的工具，能够在各类任务中实现灵活且多模态的轨迹生成。然而，大多数现有方法在处理多种类型的约束（如避碰、执行器限制和动力学一致性）方面仍然有限，这些约束通常被单独或以启发式方式处理。在这项工作中，我们提出了UniConFlow，一个基于统一约束流匹配的轨迹生成框架，它系统性地融合了等式约束和不等式约束。此外，UniConFlow引入了一种新颖的预设时间归零函数，用于在推理过程中塑造时变引导场，使生成过程能够适应不同的系统模型和任务需求。进一步地，为了解决长时域和高维轨迹生成的计算挑战，我们提出了两种实用的……（原文在此处截断）

    arXiv:2506.02955v3 Announce Type: replace-cross  Abstract: Generative models have become increasingly powerful tools for robot motion generation, enabling flexible and multimodal trajectory generation across various tasks. Yet, most existing approaches remain limited in handling multiple types of constraints, such as collision avoidance, actuation limits, and dynamic consistency, which are typically addressed individually or heuristically. In this work, we propose UniConFlow, a unified constrained flow matching-based framework for trajectory generation that systematically incorporates both equality and inequality constraints. Moreover, UniConFlow introduces a novel prescribed-time zeroing function that shapes a time-varying guidance field during inference, allowing the generation process to adapt to varying system models and task requirements. Furthermore, to further address the computational challenges of long-horizon and high-dimensional trajectory generation, we propose two practica
    
[^303]: 基于真实硬件噪声量子扩散的物理启发生成式人工智能模型

    Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion

    [https://arxiv.org/abs/2505.22193](https://arxiv.org/abs/2505.22193)

    本文提出两种受物理启发的量子扩散模型方案——利用量子随机行走中量子与经典动力学的相互作用降低生成图像的FID，以及仅用四个量子比特借助真实IBM量子硬件的固有噪声生成图像——为可扩展的量子生成式AI算法开辟了新路径。

    

    量子扩散模型（QDMs）是生成式人工智能中的一种新兴范式，旨在利用量子特性来提升其经典对应模型的性能。然而，由于近期量子设备的限制，现有算法难以扩展。在我们此前关于量子扩散模型工作的基础上，本文提出并实现了两种受物理启发的协议。在第一种方法中，我们采用量子随机行走的形式体系，表明正向过程中量子动力学与经典动力学的特定相互作用能够产生统计上更稳健的模型，其生成的MNIST图像集合的Fréchet Inception Distance（FID）低于完全使用经典动力学的情形。在第二种方法中，我们实现了一种算法，仅使用四个量子比特，通过利用真实IBM量子硬件的固有噪声来生成图像。我们的工作可以为大规模量子算法的新应用场景铺平道路。

    arXiv:2505.22193v2 Announce Type: replace-cross  Abstract: Quantum Diffusion Models (QDMs) are an emerging paradigm in Generative AI that aims to use quantum properties to improve the performances of their classical counterparts. However, existing algorithms are not easily scalable due to the limitations of near-term quantum devices. Following our previous work on QDMs, here we propose and implement two physics-inspired protocols. In the first, we use the formalism of quantum stochastic walks, showing that a specific interplay of quantum and classical dynamics in the forward process produces statistically more robust models generating sets of MNIST images with lower Fr\'echet Inception Distance (FID) than using totally classical dynamics. In the second approach, we realize an algorithm to generate images by exploiting the intrinsic noise of real IBM quantum hardware with only four qubits. Our work could be a starting point to pave the way for new scenarios for large-scale algorithms in
    
[^304]: 打破伪装！基于同音字图谱与毒性词典的中文谐音毒性内容识别

    Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon

    [https://arxiv.org/abs/2505.22184](https://arxiv.org/abs/2505.22184)

    提出了C$^2$TU，一种无需训练、无需提示的新方法，利用中文同音字图谱和毒性词典，通过子串匹配识别候选毒性词并进行过滤纠正，有效破解社交媒体中以谐音伪装毒性词汇规避审查的内容。

    

    社交媒体平台上的毒性内容（包括辱骂性语言和歧视性言论）显著增加，给内容审核带来了越来越大的挑战。一些用户通过故意用谐音伪装毒性词汇来规避审查，这使得识别伪装毒性内容成为一项必要的任务。现有方法大多针对英文文本设计，而中文伪装毒性内容的识别问题尚未得到解决。为解决这一问题，我们提出了C$^2$TU，这是一种新颖的无需训练、无需提示（prompt-free）的中文伪装毒性内容识别方法。该方法首先利用子串匹配，基于中文同音字图谱和毒性词典识别候选毒性词汇。然后，它过滤掉那些无毒的候选词，并将伪装词纠正为其对应的毒性词。具体而言，我们开发了两种用于过滤的模型变体，分别基于BERT和大语言模型（LLMs）。

    arXiv:2505.22184v3 Announce Type: replace-cross  Abstract: Social media platforms have experienced a significant rise in toxic content, including abusive language and discriminatory remarks, presenting growing challenges for content moderation. Some users evade censorship by deliberately disguising toxic words through homophonic cloak, which necessitates the task of unveiling cloaked toxicity. Existing methods are mostly designed for English texts, while Chinese cloaked toxicity unveiling has not been solved yet. To tackle the issue, we propose C$^2$TU, a novel training-free and prompt-free method for Chinese cloaked toxic content unveiling. It first employs substring matching to identify candidate toxic words based on Chinese homophone graph and toxic lexicon. Then it filters those candidates that are non-toxic and corrects cloaks to be their corresponding toxicities. Specifically, we develop two model variants for filtering, which are based on BERT and LLMs, respectively. For LLMs, w
    
[^305]: ReasoningShield：面向大型推理模型推理轨迹的安全检测

    ReasoningShield: Safety Detection over Reasoning Traces of Large Reasoning Models

    [https://arxiv.org/abs/2505.17244](https://arxiv.org/abs/2505.17244)

    大型推理模型的思维链中间步骤中可能隐藏有害内容，导致现有审核工具性能大幅下降，本文提出ReasoningShield基准与轻量级模型，用于对CoT推理轨迹进行安全审核。

    

    大型推理模型利用显式推理轨迹（即思维链CoT推理）在得出最终答案之前将复杂问题分解为中间步骤。然而，这些推理轨迹带来了独特的安全挑战：即使最终答案看似无害，有害内容也可能嵌入在中间步骤中。我们的研究表明，现有最先进的内容审核工具在CoT审核方面出现了显著的性能下降，与传统答案审核相比，F1分数下降高达35.2%。为应对这些挑战，我们提出了ReasoningShield，这是一个全面的基准测试以及一套强大的轻量级模型，用于CoT安全审核。我们的基准包含9.2K个带标注的（查询，CoT）对，具有结构化的分步风险分析，包括一个7K样本的训练集和一个2.2K的人工标注测试集，涵盖10个风险类别、3个安全等级，……

    arXiv:2505.17244v3 Announce Type: replace-cross  Abstract: Large Reasoning Models (LRMs) leverage explicit reasoning traces, known as Chain-of-Thought (CoT) reasoning, to decompose complex problems into intermediate steps before deriving final answers. However, these reasoning traces introduce unique safety challenges: harmful content can be embedded in intermediate steps even when final answers appear benign. Our study reveals that existing state-of-the-art moderation tools experience significant performance degradation on CoT moderation, with F1 scores dropping by up to 35.2% compared to traditional answer moderation. To address these challenges, we present ReasoningShield, a comprehensive benchmark and a suite of strong lightweight models for CoT safety moderation. Our benchmark includes 9.2K annotated (Query, CoT) pairs with structured stepwise risk analysis, consisting of a 7K-sample training set and a 2.2K human-annotated test set, spanning 10 risk categories, 3 safety levels, an
    
[^306]: 在提示注入防御到达用户之前对其进行审计：强大对手揭示了什么

    Auditing Prompt Injection Defenses Before They Reach Users: What Strong Adversaries Reveal

    [https://arxiv.org/abs/2505.15738](https://arxiv.org/abs/2505.15738)

    该论文主张，由于基于微调的提示注入防御缺乏形式化保证且可能带来虚假安全感，开发者应在防御部署给用户之前用强大的对抗性攻击对其进行压力测试。

    

    大型语言模型越来越多地被部署在面向用户的产品中，这些产品在处理敏感个人数据的同时还需与不可信的外部内容交互。这使得它们成为提示注入攻击的首要目标——攻击者在外部内容中注入恶意指令以劫持模型。为缓解此类攻击，研究者提出了基于微调的防御方法，通过训练模型遵循良性的用户指令并忽略注入的指令。由于缺乏形式化保证，这些防御方法仅依据当时可获得的最有效的现实攻击进行实证评估，往往报告近乎为零的攻击成功率——这给开发者以及依赖这些防御的用户带来了虚假的安全感，同时也留下了未来更强大的攻击可能绕过这些防御、并在广泛部署的系统中造成严重后果的隐患。在本文中，我们主张开发者应对其防御……

    arXiv:2505.15738v3 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed in user-facing products that handle sensitive personal data while interacting with untrusted external content. This makes them a prime target for prompt injection attacks, in which an adversary injects malicious instructions in that content to hijack the model. Fine-tuning-based defenses have been proposed to mitigate these attacks by training models to follow benign user instructions and ignore injected ones. Lacking formal guarantees, these defenses are evaluated empirically against the most effective realistic attacks available at the time, often reporting near-zero attack success rates -- giving developers, and the users who depend on them, a false sense of security and leaving open the possibility that future, more capable attacks could circumvent them and cause severe consequences in widely deployed systems. In this paper, we argue that developers should stress-test their de
    
[^307]: 面向最优且高效联邦学习的异构感知客户端采样

    Heterogeneity-Aware Client Sampling for Optimal and Efficient Federated Learning

    [https://arxiv.org/abs/2505.11304](https://arxiv.org/abs/2505.11304)

    本文首次对一般异构联邦学习进行了统一理论分析，揭示了通信异构性与计算异构性各自导致目标不一致的不同机制，并据此提出异构感知的客户端采样策略，以实现最优且高效的联邦学习。

    

    联邦学习（FL）通常涉及具有多样化通信和计算能力的客户端。这种异构性会显著扭曲优化动态并导致目标不一致，即全局模型收敛到一个可能远离所追求最优解的错误驻点。尽管其影响至关重要，但由于通信与计算异构性之间相互作用的内在复杂性，两者的联合效应在很大程度上仍未被探索。在本文中，我们揭示了异构通信和异构计算驱动联邦学习中不一致性的截然不同的机制。据我们所知，这是首个针对一般异构联邦学习的统一理论分析，为这两种异构性如何在任意局部求解器选择下共同扭曲优化轨迹提供了原则性的理解。基于这些见解……

    arXiv:2505.11304v4 Announce Type: replace-cross  Abstract: Federated learning (FL) commonly involves clients with diverse communication and computational capabilities. Such heterogeneity can significantly distort the optimization dynamics and lead to objective inconsistency, where the global model converges to an incorrect stationary point potentially far from the pursued optimum. Despite its critical impact, the joint effect of communication and computation heterogeneity has remained largely unexplored, due to the intrinsic complexity of their interaction. In this paper, we reveal the fundamentally distinct mechanisms through which heterogeneous communication and computation drive inconsistency in FL. To the best of our knowledge, this is the first unified theoretical analysis of general heterogeneous FL, offering a principled understanding of how these two forms of heterogeneity jointly distort the optimization trajectory under arbitrary choices of local solvers. Motivated by these i
    
[^308]: WoundAmbit：连接最先进语义分割技术与真实世界伤口护理

    WoundAmbit: Bridging State-of-the-Art Semantic Segmentation and Real-World Wound Care

    [https://arxiv.org/abs/2504.06185](https://arxiv.org/abs/2504.06185)

    该论文通过统一训练与评估流程，对通用视觉、医学影像及伤口挑战赛中的最先进分割模型进行了系统基准测试，评估了其在真实部署中的泛化性、效率与可解释性，并提出基于参考物体的伤口大小测量方法，以弥合前沿语义分割与实际伤口护理应用之间的差距。

    

    慢性伤口影响着大量人群，尤其是老年人和糖尿病患者，他们通常行动受限并伴有其他健康问题。通过移动设备图像采集实现自动化伤口监测，可以支持远程跟踪伤口大小，从而减少患者面对面的就医次数。语义分割是实现这一过程的关键，然而伤口分割在医学影像研究中仍受到的关注不足。为此，我们对来自通用视觉领域、医学影像领域的最先进深度学习模型，以及公开伤口挑战赛中的顶级方法进行了基准测试。为了实现公平比较，我们统一了训练、数据增强和评估流程，并进行交叉验证以最小化数据划分偏差。我们还评估了实际部署相关的方面，包括在分布外伤口数据集上的泛化能力、计算效率以及可解释性。此外，我们提出了一种基于参考物体的伤口大小测量方法。

    arXiv:2504.06185v3 Announce Type: replace-cross  Abstract: Chronic wounds affect a large population, particularly the elderly and diabetic patients, who often exhibit limited mobility and co-existing health conditions. Automated wound monitoring via mobile image capture can reduce in-person physician visits by enabling remote tracking of wound size. Semantic segmentation is key to this process, yet wound segmentation remains underrepresented in medical imaging research. To address this, we benchmark state-of-the-art deep learning models from general-purpose vision, medical imaging, and top methods from public wound challenges. For a fair comparison, we standardize training, data augmentation, and evaluation, conducting cross-validation to minimize partitioning bias. We also assess real-world deployment aspects, including generalization to an out-of-distribution wound dataset, computational efficiency, and interpretability. Additionally, we propose a reference object-based approach to c
    
[^309]: CRM：具备可控条件的检索模型

    CRM: Retrieval Model with Controllable Condition

    [https://arxiv.org/abs/2412.13844](https://arxiv.org/abs/2412.13844)

    提出可控检索模型（CRM），将观看时长作为用户塔的条件输入，使模型在推理阶段无需重新训练即可通过指定期望观看时长来引导检索，同时保持物品向量可预先计算以支持高效的 ANN 检索。

    

    面向推荐的检索模型需要在高效搜索大规模物品语料库的同时，兼顾观看时长等结果指标。我们提出了可控检索模型，这是一个以结果为条件的检索框架，它将观看时长作为用户塔的输入。在训练阶段，CRM 学习在给定观测到的观看时长的条件下预测下一个物品；在推理阶段，我们输入期望的观看时长来引导检索，而无需重新训练。由于物品塔与该条件无关，物品向量仍然可以被预先计算并建立索引，用于近似最近邻（ANN）搜索。我们将 CRM 同时实现为基于 DNN 的双塔模型和 Decision Transformer 两种形式。其中 Transformer 将物品 token 与剩余观看时长条件 token 交错排列，并使用目标观看时长作为数值提示来进行下一物品预测。对于每个在线请求，我们在用户近期历史记录的平均观看时长与最大观看时长之间进行选择。

    arXiv:2412.13844v2 Announce Type: replace-cross  Abstract: Retrieval models for recommendation need to account for outcomes such as watch time while searching large item corpora efficiently. We present the Controllable Retrieval Model (CRM), an outcome-conditioned retrieval framework that adds watch time as an input to the user tower. During training, CRM learns to predict the next item given its observed watch time; at inference, we supply a desired watch time to steer retrieval without retraining. The item tower is independent of the condition, so item embeddings can still be precomputed and indexed for approximate nearest neighbor (ANN) search. We implement CRM as both a DNN-based two-tower model and a Decision Transformer. The Transformer interleaves item tokens with watch-time-to-go condition tokens, using the target watch time as a numerical prompt for next-item prediction. For each online request, we choose between the average and maximum watch times in the user's recent history
    
[^310]: 通过反演理解基于Transformer的视觉模型

    Understanding Transformer-based Vision Models through Inversion

    [https://arxiv.org/abs/2412.06534](https://arxiv.org/abs/2412.06534)

    本文提出一种新颖的模块化特征反演方法，能更高效地从大规模Transformer视觉模型的中间表示重建语义更连贯的图像，并借此揭示了DETR与ViT等模型内部机制随网络深度的差异。

    

    理解深度神经网络的内部机制仍然是机器学习中的核心挑战。在计算机视觉领域，一种有前景但仅被初步探索的方法是通过逆向网络进行特征反演，即使用训练好的逆向网络从中间表示重建图像。在本研究中，我们重新审视了通过逆向网络进行特征反演的方法，提出了一种新颖的模块化变体，使该技术在计算上更加高效，并且在某些情况下能够生成语义上更加连贯的图像重建结果。我们将该方法应用于大规模基于Transformer的视觉模型，具体包括检测Transformer（Detection Transformer）、视觉Transformer（Vision Transformer）、Swin Transformer和数据高效图像Transformer（Data-Efficient Image Transformer），并分析了重建结果随网络深度的变化。我们的主要分析比较了检测Transformer和视觉Transformer，二者展现出最具信息量的差异。

    arXiv:2412.06534v5 Announce Type: replace-cross  Abstract: Understanding the internal mechanisms of deep neural networks remains a central challenge in machine learning. In computer vision, one promising yet only preliminarily explored approach is feature inversion via inverse networks, which reconstructs images from intermediate representations using trained inverse networks. In this study, we revisit feature inversion via inverse networks, introducing a novel, modular variant that enables a computationally more efficient application of the technique while, in some cases, producing semantically more coherent image reconstructions. We apply our method to large-scale transformer-based vision models, specifically Detection Transformer, Vision Transformer, Swin Transformer, and Data-Efficient Image Transformer, analyzing the resulting reconstructions across network depth. Our main analysis compares Detection Transformer and Vision Transformer, which exhibit the most informative difference
    
[^311]: 条件Vendi分数：面向生成式AI模型与大语言模型的提示感知多样性评估

    Conditional Vendi Score: Prompt-Aware Diversity Evaluation for Generative AI Models and LLMs

    [https://arxiv.org/abs/2411.02817](https://arxiv.org/abs/2411.02817)

    该论文提出了基于条件熵的条件Vendi和条件RKE两种新度量方法，能够在提示引导的生成任务中分离出模型自身引起的多样性，为生成式AI模型和大语言模型提供了提示感知的多样性评估工具。

    

    由文本提示引导的生成式模型在保真度和提示对齐方面已被广泛评估，但其产生多样化输出的能力仍未得到充分研究。现有的多样性指标（如基于核矩阵的冯·诺依曼熵和Rényi熵的Vendi与RKE）是为无条件模型设计的，无法区分由提示引起的变异与模型自身引起的变异。我们通过引入条件Vendi（Conditional-Vendi）和条件RKE（Conditional-RKE）来填补这一空白，这两种多样性度量源自半正定矩阵的条件熵。这些分数能够在提示引导的生成中分离出由模型引起的多样性，其中条件RKE具有O(1/√n)的收敛速率。对于条件Vendi，我们提出了一种截断谱近似方法，可提供可扩展且一致的估计。在文本到图像、图像描述生成以及大语言模型任务上的实验表明，条件分数能够……（原文摘要在此处截断）

    arXiv:2411.02817v3 Announce Type: replace-cross  Abstract: Generative models guided by text prompts are widely evaluated for fidelity and prompt alignment, yet their ability to produce diverse outputs remains underexplored. Existing diversity metrics such as Vendi and RKE, which are based on the von Neumann and R\'enyi entropies of kernel matrices, were developed for unconditional models and cannot distinguish prompt-induced from model-induced variability. We address this gap by introducing \textit{Conditional-Vendi} and \textit{Conditional-RKE}, diversity measures derived from the conditional entropy of positive semidefinite matrices. These scores isolate model-induced diversity in prompt-guided generation, with Conditional-RKE enjoying an $O(1/\sqrt{n})$ convergence rate. For Conditional-Vendi, we introduce a truncated-spectrum approximation that yields scalable and consistent estimates. Experiments on text-to-image, image-captioning, and LLM tasks show that the conditional scores re
    
[^312]: 基于杠杆分数采样的可证明精确的Shapley值估计

    Provably Accurate Shapley Value Estimation via Leverage Score Sampling

    [https://arxiv.org/abs/2410.01917](https://arxiv.org/abs/2410.01917)

    提出Leverage SHAP，通过对Kernel SHAP的轻量级改进，利用杠杆分数采样将模型评估次数从指数级降至O(n log n)，实现了可证明精确的Shapley值估计。

    

    Shapley值最初源于博弈论，如今已成为可解释机器学习中的核心工具，用于将模型预测归因于特定的输入特征。然而，精确计算Shapley值的代价十分高昂：对于一个具有 $n$ 个特征的一般模型，需要进行 $O(2^n)$ 次模型评估。为解决这一问题，近似算法被广泛采用，其中最流行的算法之一是Kernel SHAP，该算法与具体模型无关，且在实践中表现出色。在我们最初提交论文时，Kernel SHAP尚缺乏强有力的非渐近复杂度保证。为解决这一问题，我们提出了Leverage SHAP，这是对Kernel SHAP的一种轻量级改进，仅需 $O(n\log n)$ 次模型评估即可提供可证明精确的Shapley值估计。我们的方法利用了Shapley值估计与不可知主动学习之间的联系，通过采用……

    arXiv:2410.01917v3 Announce Type: replace-cross  Abstract: Originally introduced in game theory, Shapley values have emerged as a central tool in explainable machine learning, where they are used to attribute model predictions to specific input features. However, computing Shapley values exactly is expensive: for a general model with $n$ features, $O(2^n)$ model evaluations are necessary. To address this issue, approximation algorithms are widely used. One of the most popular is the Kernel SHAP algorithm, which is model agnostic and remarkably effective in practice. At the time of our original submission, no strong non-asymptotic complexity guarantees were known for Kernel SHAP. We address this issue by introducing Leverage SHAP, a lightweight modification of Kernel SHAP that provides provably accurate Shapley value estimates with just $O(n\log n)$ model evaluations. Our approach takes advantage of a connection between Shapley value estimation and agnostic active learning by employing 
    

