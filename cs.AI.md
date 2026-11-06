# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization](https://arxiv.org/abs/2611.08483) | SwarmEvolve让每轮数百个低成本智能体在潜在空间中以集群方式推理、并通过KV缓存共享信息，从而大幅加速LLM引导的进化搜索，在21个数学发现与科学计算问题上表现出色。 |
| [^2] | [Scientist World: Grow a Research World for AI-Scientists](https://arxiv.org/abs/2611.08478) | 该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。 |
| [^3] | [XStack: A Benchmark for ML Repo Migration](https://arxiv.org/abs/2611.08472) | 该论文提出 XStack 基准，通过要求智能体将 PyTorch 代码仓库完整迁移为可在 TPU 上运行的 JAX 代码仓库，来评估其跨框架、跨硬件的长时程迁移能力，目前最强模型通过率仅约 75%。 |
| [^4] | [Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information](https://arxiv.org/abs/2611.08469) | 本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。 |
| [^5] | [Towards interpreting any residual stream direction by scaling activation inversion](https://arxiv.org/abs/2611.08465) | 提出最大激活示例元模型（MAEM），利用无标签强化学习目标将残差流的任意方向反演为强激活该方向的文本示例，在解释效果上超越了语料库搜索和自然语言自编码器。 |
| [^6] | [ReproBridge: Building Reproducibility into ML Research Artifacts](https://arxiv.org/abs/2611.08452) | 该论文提出 ReproBridge 智能体工作流，将散见于论文与代码中的实验流程和决策整合为标准化可复现产物，并构建包含 36 篇论文 1,029 个任务的基准，以检验这类精炼产物能否提升智能体独立复现机器学习实验的能力。 |
| [^7] | [Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes](https://arxiv.org/abs/2611.08439) | 论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。 |
| [^8] | [Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety](https://arxiv.org/abs/2611.08438) | 本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。 |
| [^9] | [MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue](https://arxiv.org/abs/2611.08433) | MedGraph 提出一种图记忆方法，通过带时间戳的患者专属事实链条、在问题所涉时间点读取的检索机制以及随每个问题提供的安全事实档案，解决了长期医疗对话中状态过时、归属错误和安全信息缺失三类问题，在 MedMemoryBench 上取得最高总体准确率。 |
| [^10] | [The Power of Meta Access in the Software of the Future --- Demonstrated on LLM Chat UIs and Chat AI](https://arxiv.org/abs/2611.08428) | 本文提出“元访问”机制——让LLM聊天助手获得对界面自身的结构化、分级授权视图，从而把聊天客户端变成软件的统一操作入口，并通过Meta Chat原型加以验证，同时指出其带来的隐私、提示注入等风险以及对独立控制平面边界的需要。 |
| [^11] | [Failure Prediction and Classification in Vision-Language-Action Models Via Calibrated Latent Probing](https://arxiv.org/abs/2611.08425) | 提出VLA-Tug，一种基于VLA骨干网络潜在层的轻量级校准探测器，可在极小推理开销下实现早期失败检测，并对漏检率和误报率提供有效界限。 |
| [^12] | [Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks](https://arxiv.org/abs/2611.08417) | 论文提出“基准维护债务”概念，并构建了结合社区报告、自动检查与人工修复的持续验证工作流，对Terminal-Bench 2.0的审计发现约30%的任务存在缺陷，论证了基准测试应像软件一样进行持续维护。 |
| [^13] | [The Gold Rush in AI4Math, II: What are signatures of AI-assisted Proofs?](https://arxiv.org/abs/2611.08412) | 本文通过构建包含 8,281 篇论文、涵盖不同 AI 参与程度的五个队列，并提出基于即时证明支持图（GIPS）的 ProofTrace 框架来提取可解释特征，从而实现对数学论文中 AI 参与证明的检测。 |
| [^14] | [StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning](https://arxiv.org/abs/2611.08403) | StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。 |
| [^15] | [Learning a patient world model for health system simulation in intensive care](https://arxiv.org/abs/2611.08392) | 该论文提出从纵向电子健康记录中学习一个患者世界模型，为重症监护医疗系统模拟提供可跨多种场景复用的患者轨迹共享临床表示，避免了为每个新问题重新定义状态和转移模型。 |
| [^16] | [Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data](https://arxiv.org/abs/2611.08386) | 该研究提出一种可解释的混合特征融合框架，将LSTM自编码器的重构误差描述符与统计飞行历史特征融合后输入类别加权XGBoost分类器，以在高度不平衡的多变量飞行数据中实现多类别着陆阶段安全事件预测。 |
| [^17] | [FedReason: Large Language Models as Weight-Space Meta-Aggregators for Heterogeneous Federated Learning](https://arxiv.org/abs/2611.08383) | FedReason提出让大语言模型无需任何服务器端训练，通过受限沙箱定期分析客户端模型权重并自动生成自适应聚合程序，从而在标签偏斜、模型异构和拜占庭客户端等异构联邦学习场景中优于14种固定聚合规则基线。 |
| [^18] | [MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation](https://arxiv.org/abs/2611.08372) | MOMASPLAT提出了一个面向移动操作的新视角示范增强框架，能够从未见过的机器人底盘位姿合成包含动态物体交互的完整操作轨迹，克服了现有新视角合成方法因假设静态场景而无法增强接触后关键交互阶段的局限。 |
| [^19] | [NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma](https://arxiv.org/abs/2611.08369) | NPCBench是一个模拟临床专科医师培训的鼻咽癌多模态诊疗路径推理基准，包含4,777个条目和100名患者的CarePath队列，通过回合级评估检验模型能否在完整诊疗路径中联合正确地完成诊断与诊疗任务。 |
| [^20] | [VisionSIM: Beyond Idealized Sensors - A Multi-Modal Simulator for Robot Perception](https://arxiv.org/abs/2611.08368) | 提出 VisionSIM，一个基于物理的多模态传感器仿真平台，通过建模原始物理测量过程重现 LiDAR、iToF、事件相机和热成像等传感器的真实失效模式，支持在相同场景下的跨模态系统性比较与机器人感知研究。 |
| [^21] | [Enhancing Local and Global Structure in Compressed AI Agent Memory](https://arxiv.org/abs/2611.08341) | 本文提出用局部保真度 $L$ 和全局保真度 $G$ 两种无标签指标评估AI智能体压缩记忆的质量，证明全局保真度可由近远排序违反率界定，并提出局部感知量化（LAQ）方法，将理论界限转化为合页目标函数，从而在压缩中同时增强记忆的局部与全局结构。 |
| [^22] | [Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement](https://arxiv.org/abs/2611.08318) | 论文提出“标准暴露”（CE）这一被忽视的实践问题，并通过对照实验揭示：当自我改进的编码智能体接触基准的验证标准时，其生成的代码倾向于过拟合验证标准，而非真正完成被指示的任务。 |
| [^23] | [Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation](https://arxiv.org/abs/2611.08315) | 该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。 |
| [^24] | [RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines](https://arxiv.org/abs/2611.08308) | 本文提出 RM-VLM，利用大语言模型从自然语言任务描述自动生成奖励机，并让视觉-语言模型充当标注函数将命题接地于图像观测，从而免除传统奖励机中需要针对每个环境手动设计标注函数的局限。 |
| [^25] | [Measured Coupling for Parallel Decoding in Diffusion Language Models](https://arxiv.org/abs/2611.08307) | 该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。 |
| [^26] | [First-Token Attraction in Mamba Dynamics](https://arxiv.org/abs/2611.08303) | 该论文提出基于高维不变盒子与精确退出时间论证的新分析框架，首次刻画了深度Mamba模型中token轨迹的极限点并证明指数收敛速率，揭示首个token的极限方向会吸引后续token这一特权动力学特性。 |
| [^27] | [AutoCompress: Automated Hardware-Guided Model Compression](https://arxiv.org/abs/2611.08290) | AutoCompress是一个硬件引导的模型压缩框架，它将延迟、吞吐量和内存等硬件与工作负载约束直接纳入可微架构搜索，联合优化全局与逐层结构以生成异构压缩架构，并利用经过实测验证的服务引擎成本模型引导整个优化过程。 |
| [^28] | [A Generative Model of Contextual Integrity: Appropriate vs. Inappropriate Information Sharing](https://arxiv.org/abs/2611.08284) | 该论文构建了一个基于大语言模型模拟对话的情境完整性生成模型，用以研究信息共享行为，发现AI行为者既会在需要时向合法接收者分享信息，也会不恰当地向无权者披露私人信息。 |
| [^29] | [LeanContext: Disentangling Mathematical Resources from Local Formal Context in Lean Whole-Proof Generation](https://arxiv.org/abs/2611.08277) | 本文提出包含129,826个Lean整段证明生成问题的基准LeanContext，通过将数学上下文与环境上下文解耦，发现当训练数据包含数学上下文时，模型才能更大程度地利用目标所选声明相对于可比替代方案的优势。 |
| [^30] | [Aligning Policy Encoders with a Shape Teacher](https://arxiv.org/abs/2611.08275) | 提出形状教师对齐方法，在训练时以预训练3D形状表示（Point-MAE特征）作为对齐目标，使基于图像的策略编码器习得3D几何信息，同时保持部署时纯图像输入的接口不变。 |
| [^31] | [Accelerating Safe Reinforcement Learning with Massive Parallelism](https://arxiv.org/abs/2611.08273) | 该论文提出相位聚合成本估计器与约束收紧框架，解决了大规模并行短回合训练与全回合安全约束之间的不匹配问题，实现了可扩展的安全强化学习加速训练。 |
| [^32] | [DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection](https://arxiv.org/abs/2611.08271) | DART提出了一种即插即用框架，通过残差解耦模块与跨领域元训练，将正常-异常参考残差分离为异常相关成分与领域特定成分，从而提升通用异常检测的跨域迁移能力。 |
| [^33] | [Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations](https://arxiv.org/abs/2611.08269) | 预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。 |
| [^34] | [Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining](https://arxiv.org/abs/2611.08263) | 提出BEEMO框架，通过解耦行为与神经表征学习，利用丰富且易获取的非配对行为数据进行训练，独立扩展行为和神经监督，从而在非人类灵长类皮层内运动数据集上显著提升解码性能。 |
| [^35] | [The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model](https://arxiv.org/abs/2611.08262) | 该研究发现多模态语言模型中提示措辞对内部表征几何的影响大于人格身份本身，而人格内容则主要驱动行为判断，从而揭示了表征与行为在视角模拟中的解耦现象。 |
| [^36] | [Value-oriented policy distillation for interpretable wind farm power control](https://arxiv.org/abs/2611.08245) | 该论文提出了一种面向价值的策略蒸馏框架，利用可解释提升机（EBM）将强化学习策略蒸馏为透明可解释的控制策略，从而实现风电场的隐式平衡功率控制。 |
| [^37] | [Signals, Evidence, and Context: User Scrutiny in Real-World LLM Conversations](https://arxiv.org/abs/2611.08232) | 该研究通过分析近49万条真实LLM对话发现，模型答案中宽泛的归因表述远多于可查验的显式来源链接，不确定性等回复线索与用户的核查和质疑行为相关，但用户的审视行为高度集中于少数对话。 |
| [^38] | [SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving](https://arxiv.org/abs/2611.08196) | SMART框架通过捕捉问题上下文中参与实体的隐式状态，并利用多种LLM和基于角色的结构化提示策略生成多样化、互补且与解决方案相关的前提知识，将复杂科学论断的验证性能提升高达20%。 |
| [^39] | [Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation](https://arxiv.org/abs/2611.08179) | 提出TOPSD方法，利用教师-学生似然估计教师漂移的起点并选择可教学前缀进行监督，解决了多语言推理自蒸馏中的教师漂移问题，尤其改善了低资源语言的推理表现。 |
| [^40] | [BabyTheorist: A Benchmark for Learning to Theorize the World from Observation Alone](https://arxiv.org/abs/2611.08177) | 提出了 BabyTheorist 基准，用于评估模型能否仅凭纯视觉观察学习可复用的世界理论，并将其泛化应用到全新情境中。 |
| [^41] | [PCBRouteAgent: How Close Are LLM Agents to Becoming PCB Routing Engineers?](https://arxiv.org/abs/2611.08175) | PCBRouteAgent是一个模仿人类工程师“全局规划+局部执行”协作方式的多模态LLM布线智能体，通过全局布线规划器与基于KiCad的详细布线执行器配合，并利用结构化失败反馈迭代修正规划，在247块电路板上实现了可靠的自动化PCB布线。 |
| [^42] | [Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke](https://arxiv.org/abs/2611.08154) | 该研究提出一种可解释的多模态MRI与临床数据融合方法，证明FLAIR-临床融合（AUC=0.948）能显著提升急性缺血性卒中后三个月功能结局的预测性能。 |
| [^43] | [A Generative Foundation Model for Multimodal Tokamak Tasks](https://arxiv.org/abs/2611.08153) | 该论文扩展了TokaMark基准并提出了一种可复用的生成式基础模型，能够自适应地匹配托卡马克各类任务的输入输出模式并联合建模异构目标信号，无需为每个任务单独训练模型。 |
| [^44] | [Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure](https://arxiv.org/abs/2611.08134) | 本文提出PAAI（权力、可及性、充分性、独立性）分析框架，通过考察评估基础设施的权威分配，结合非洲AI评估项目与国际AI安全研究所网络的案例，揭示谁能够真正独立地评估AI系统。 |
| [^45] | [Create Your Own Work Stack: Deploying Self-hosted AI and Tooling in a Tech Startup](https://arxiv.org/abs/2611.08116) | 该论文分享了一家小型软件公司部署混合 AI 栈的实践经验——商业 AI 代理负责复杂任务，本地开源模型承担低复杂度任务、组织信息检索及自托管工具同步，从而在不牺牲质量的前提下削减约 30–40% 的商业 AI 使用成本。 |
| [^46] | [Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines](https://arxiv.org/abs/2611.08093) | 提出梯度提升框架GB-CPKM，通过顺序添加固定秩CP张量网络学习器来增量式提升模型容量，从而避免显式秩搜索，实现最高6倍的训练加速和2倍的峰值内存节省。 |
| [^47] | [Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks](https://arxiv.org/abs/2611.08088) | 该论文提出了一种带自由权重的LRP乘积规则，将图注意力网络中节点接收到的相关性分解为值路径的内容相关性和注意力路径的路由相关性两部分，并将现有解释方法统一为其特例。 |
| [^48] | [MUSS: A Multi-scale and Sequence-based Model for Single-cell Gene Regulation](https://arxiv.org/abs/2611.08086) | MUSS是一个基于序列的多尺度单细胞基因调控模型，利用冻结的预训练DNA序列编码器提取峰级与基因级表示，并通过峰-峰自注意力和距离偏置的峰-基因交互进行整合，在保留DNA序列信息的同时实现细胞水平的基因调控建模。 |
| [^49] | [DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels](https://arxiv.org/abs/2611.08082) | DictLLM提出了一种基于字典核的训练后压缩方法，将分组层的权重矩阵分解为共享字典与层特定稀疏系数，无需进一步训练即可在移动设备上高效压缩大语言模型。 |
| [^50] | [Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"](https://arxiv.org/abs/2611.08070) | 论文在无工具、无前向传播的条件下测试LLM读取小型卷积网络原始权重的能力，发现其表现主要源于准确的局部算术计算与对公开数据集的记忆先验，而非对权重流动的真正全局推理。 |
| [^51] | [NutriVisRAG: Multimodal Exemplar Grounding for Nutrient Estimation with Vision-Language Models](https://arxiv.org/abs/2611.08054) | 提出了NutriVisRAG检索增强框架，通过视觉编码器检索视觉相似的已标注餐食作为多模态上下文示例来锚定视觉语言模型的营养估计，无需任务特定微调即可提升估计性能。 |
| [^52] | [When Imperfect Automated Fact-Checking Backfires: Social Learning in the Presence of Misinformation](https://arxiv.org/abs/2611.08053) | 该研究通过序贯社会学习模型证明，免费但易错的自动事实核查工具是一把双刃剑——它虽能帮助用户摆脱错误的舆论级联，但其错误可能误导原本会进行完美人工核查的用户，从而反而降低中等可靠性文章下的社会福利。 |
| [^53] | [Secure Formal Raising: Protecting the Development of AI through Executable Properties](https://arxiv.org/abs/2611.08042) | 本文提出安全形式化培育（SFR）框架，用保留未知与不一致状态的四值证据建模策略完整性、授权和机密数据流，证明了六条安全不变式并通过纯C参考实现的有界穷举探索加以验证，同时发现并借助溯源要求修复了工具调用中的混淆代理人漏洞。 |
| [^54] | [Technohybridism: Recursive Semantic Normalization in AI-Mediated Societies](https://arxiv.org/abs/2611.07984) | 本文提出“递归语义规范化”与“技术混合主义”两个新概念，主张应以人类实践、计算生成与制度性采纳递归耦合而成的混合语义系统为分析单位，以阐释AI生成语言如何再进入社会交流与制度实践并重塑语义规律。 |
| [^55] | [TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research](https://arxiv.org/abs/2611.07962) | 该论文提出涵盖定理证明、求解器工程、构造和开放猜想求解四个赛道、共100个专家审阅任务的TCSExam基准，首次系统评估前沿大语言模型在理论计算机科学研究全领域的能力，结果显示GPT-6 Astra在完整版（71.9分）和Opus 5.5在精简版（72.3分）上分别领先。 |
| [^56] | [A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems](https://arxiv.org/abs/2611.07953) | 本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。 |
| [^57] | [Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency](https://arxiv.org/abs/2611.07948) | 提出Circe——一种免训练的大语言模型自我评估方法，通过分析注意力与前馈网络等组件间的交互一致性并结合内外部信号来评估回答正确性，无需参考答案或重复采样。 |
| [^58] | [ChemE-as-Code: An Open-Source Platform for Agentic AI in Chemical Engineering](https://arxiv.org/abs/2611.07947) | 该论文提出了开源平台ChemE-as-Code，让基于LLM的AI智能体在可追溯、可版本管理的环境中完成从催化到过程模拟、HAZOP等化学工程设计与计算工作，并通过自动化检查引导而非取代人工审查。 |
| [^59] | [tok: a textual concrete syntax of BPMN and DMN for humans and AI agents](https://arxiv.org/abs/2611.07945) | 本文提出了tok这一BPMN和DMN的纯文本表示法，其令牌效率比XML高数倍、故障检测率更高，且语言模型能轻松学习并生成有效的tok模型。 |
| [^60] | [CardioLM: A multilingual suite of small language models for the cardiology domain](https://arxiv.org/abs/2611.07940) | 该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。 |
| [^61] | [Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives](https://arxiv.org/abs/2611.07935) | 提出了一种可扩展、诊断驱动的全切片图像检索框架CLEAR-WSI，通过固定长度的切片嵌入并利用诊断标签细化检索候选，摆脱了手工索引流水线，在公开和临床病理数据上实现了高效的相似病例检索。 |
| [^62] | [AttnSR: Attention Step Reward for Long-Horizon Agentic RL](https://arxiv.org/abs/2611.07932) | 提出 AttnSR 步级奖励增强方法，利用训练回放中的策略注意力反向追踪为各步骤分配奖励，无需额外奖励模型，即在 ALFWorld 上将 Qwen2.5-7B 的成功率从 91.2% 提升至 97.1%。 |
| [^63] | [ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation](https://arxiv.org/abs/2611.07928) | ZeroRAG提出了一种无需全局数据结构和集中式编排的去中心化RAG文档检索方案，通过在对等覆盖网络上进行自适应语义查询路由，使文档始终保留在原始所有者手中。 |
| [^64] | [How Finite Brains Generalize Without Bound: Spatial Memory as a Universal Computer](https://arxiv.org/abs/2611.07920) | 论文揭示了有限大脑能无界泛化的关键机制：学习者只应读取“巧合”（等价性判断）而非“坐标”（位置、计数等随规模变化的量），由此推导出可泛化的大脑必须拥有一种由循环构成、随输入增长却始终保持有限读取、并可充当通用计算机的空间记忆。 |
| [^65] | [Adaptive Semantic Branching: Disagreement-Localized Cross-Trajectory Regeneration for Reasoning Candidate Expansion](https://arxiv.org/abs/2611.07918) | 该论文提出了无需训练的自适应语义分支（ASB）方法，通过定位不同推理轨迹间的分歧点并融合分歧处的推理内容来再生成新的候选解法，从而有效扩展推理候选的多样性与质量。 |
| [^66] | [GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving](https://arxiv.org/abs/2611.07912) | 提出GuidedWAM，一种无需重新训练的与模型无关的能量引导框架，可在采样时将新驾驶目标注入冻结的自动驾驶世界动作模型，在NAVSIM v2难例集上平均提升1.8至3.6 EPDMS，且每步采样开销不足10%。 |
| [^67] | [Structure from Collision via Hypothesis-Space Reasoning](https://arxiv.org/abs/2611.07900) | 提出通过假设空间推理从碰撞事件的多视角观测中恢复物体隐藏内部结构（如空腔），克服了空腔仅能通过表面运动间接观测、且因对应材料缺失而难以用密度表示直接优化的难题。 |
| [^68] | [Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency](https://arxiv.org/abs/2611.07899) | 提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。 |
| [^69] | [DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning](https://arxiv.org/abs/2611.07892) | DISCERN 通过为每个类别拟合欧氏与方向性参数分布并采样合成嵌入进行回放，实现了无需访问保留集的增量式类遗忘，在依次删除类别的同时确保已遗忘知识不被恢复并保留剩余类别的性能。 |
| [^70] | [When Bigger Becomes Brittle: Why Compression Hurts More as Language Models Scale](https://arxiv.org/abs/2611.07891) | 该论文提出一种精确的分解方法，将压缩（量化/剪枝）造成的损失拆分为“网络其余部分的补偿恢复”与“真正传递到输出的残余损失”两部分，并发现随着语言模型规模增大，网络内部的补偿能力减弱，因此压缩对大模型的伤害反而更严重。 |
| [^71] | [When, Where, How: Active In-Context Learning for Tabular Foundation Models](https://arxiv.org/abs/2611.07886) | 本文针对表格基础模型提出主动上下文学习方法，研究了目标感知最近邻（TANN）采集和上下文期望预测信息增益（EPIC）两种策略，并在二十个基准上发现基于多样性的采集在初期迭代中最为有效。 |
| [^72] | [InnerLogic-Surg: An Open Surgical Foundation Model via Continual Pretraining](https://arxiv.org/abs/2611.07883) | 本文提出InnerLogic-Surg，一个基于DINOv3持续预训练、利用来自11,965个手术视频约2,900小时的数百万张图像训练而成的开放式手术视觉基础模型家族（21M-300M参数），可显著增强手术图像分析任务的性能。 |
| [^73] | [Network Effect Decomposition for Causal Effect Estimation under Network Interference](https://arxiv.org/abs/2611.07879) | 提出NEDNet网络，将网络干扰下的结果响应分解为查询无关的处理前基线以及相互独立的处理响应通路和暴露响应通路，从而在观测网络数据上实现更准确的因果效应估计。 |
| [^74] | [Responsible GenAI Use During Active Armed Conflict: Evidence from Sudan's Emergency Response Rooms](https://arxiv.org/abs/2611.07874) | 该研究通过对苏丹应急响应室社区救援者的调查与访谈，首次揭示了武装冲突环境下人道主义工作者对生成式AI的使用情况，发现战争使GenAI在输出最难验证的场景中反而最有价值，但在缺乏人工验证时依赖其输出可能导致有害决策。 |
| [^75] | [FARR: Forecast-Aware Residual Refinement with Matched Endpoint Evidence for Long-Horizon Time-Series Forecasting](https://arxiv.org/abs/2611.07870) | 提出 FARR 方法，通过对比标准条件与空条件下的预测差异获得端点证据，并利用共享残差精炼器结合逐视界门控机制来校正预测残差，在四个长期时序预测基准上取得了最优的平均 MSE 和 MAE。 |
| [^76] | [CORE: Coarse-to-Fine Search with Context Management for Deep Research](https://arxiv.org/abs/2611.07859) | CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。 |
| [^77] | [Computation, not weight access: large language models as auditors of clinical prediction networks](https://arxiv.org/abs/2611.07847) | 通过预先设定的模拟实验，该研究发现在没有计算工具时仅凭提示中的原始权重无法可靠恢复植入的关系，表明大语言模型审计临床预测网络的关键在于计算能力（借助工具访问权重或预测）而非直接的权重访问。 |
| [^78] | [Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding](https://arxiv.org/abs/2611.07845) | 该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。 |
| [^79] | [THG-Omni: Traceable Human-Centric Graph Supervision for Video Reasoning](https://arxiv.org/abs/2611.07836) | THG-Omni提出了一种以人为中心的可追溯图监督框架，通过将SAM3参与者追踪与时间定位的Qwen3-VL标注相结合并核验语义声明，使全模态推理模型能够从结构化图中解读视频里人物的属性、动作、情感、交互及其时间关系。 |
| [^80] | [STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation](https://arxiv.org/abs/2611.07828) | STAG-Net通过对面部关键点组施加结构感知注意力（结合运动感知选择、空间聚类、紧凑DeepSet编码器和全局几何描述子），以极低的计算成本实现了精度领先的轻量级头部姿态估计。 |
| [^81] | [Instantiating the Knowledge Plane](https://arxiv.org/abs/2611.07827) | 本文提出利用代理型大语言模型实现2003年Clark等人设想的“知识平面”，从而实现跨管理边界的自动化互联网协作管理，并指出必须解决去中心化、多代理、竞合环境下出现的新研究问题。 |
| [^82] | [Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents](https://arxiv.org/abs/2611.07825) | 该论文提出两阶段“生成后排序”方法用于具身智能体的符号目标解释，通过监督微调适配输出规范并结合基于错误的候选输出排序（CORE）学习输出偏好，在EAI基准上超越了现有基线模型。 |
| [^83] | [Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval](https://arxiv.org/abs/2611.07820) | 该论文发现冻结的稠密嵌入模型无法按时间、地点、人物、事件四个维度有效区分个人记忆，并提出 Axis-LoRA——在嵌入模型内部为每个维度设置 LoRA 分支并通过逐查询门控融合的适配方法，仅训练少量参数即可在三个记忆检索基准上显著提升性能。 |
| [^84] | [Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment](https://arxiv.org/abs/2611.07780) | 提出GenVer生成式验证器，通过对齐三元组及其生成的自然语言表示来融合图结构信息与预训练语言模型知识，在FactFB数据集上取得了最高达91.52%的三元组分类准确率。 |
| [^85] | [The Intrusion Pathway: How Non-Evidential Context Degrades Language Model Performance](https://arxiv.org/abs/2611.07778) | 本文揭示了非证据性上下文（无关陈述或随机标记）降低语言模型性能的内在机制，将其归因于“侵入通路”及一组稀疏的“侵入头”注意力头，这些头将无关的前缀内容写入预测位置从而导致准确率下降。 |
| [^86] | [RIVER: Evidence-guided multi-agent retrieval for enzyme--reaction discovery](https://arxiv.org/abs/2611.07757) | 本文提出RIVER，一个证据引导的多智能体框架，通过协调大语言模型、生化计算工具与外部知识资源进行多层面证据整合，实现双向酶-反应检索，克服了传统基于序列相似性方法难以整合生化证据及区分高相似酶家族的局限。 |
| [^87] | [Scaling Laws for SAE Training Data](https://arxiv.org/abs/2611.07756) | 在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。 |
| [^88] | [Extending Reasoning in Typed Decision Models via Looping](https://arxiv.org/abs/2611.07736) | 该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。 |
| [^89] | [Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction](https://arxiv.org/abs/2611.07735) | 本研究以卢森堡新闻平台的点赞预测任务为例，发现LLM的token概率作为置信度估计时模型过度自信、被简单统计基线全面超越，且准确率与校准度并不同步——扩大模型规模能提升准确率却损害校准度，表明评估人类行为模拟时校准度是不可忽视的关键指标。 |
| [^90] | [O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix](https://arxiv.org/abs/2611.07731) | 提出 O$^2$PD 方法，通过定量分析教师监督的可靠性来确定“对齐哪些 token”（可靠前缀），并以重叠门控机制决定“如何对齐”，从而弥补现有 top-$k$ 在线策略蒸馏在监督可靠性与概率质量丢弃方面的不足。 |
| [^91] | [Seeing Beyond the Next Step: World-Model-Guided Human-Like Navigation in Multi-Agent Scenes](https://arxiv.org/abs/2611.07730) | 该论文提出将世界模型置于训练阶段作为偏好oracle，把对未来的前瞻性推理蒸馏进一个快速的视觉动作先验策略中，而非在推理时用作在线规划器，从而在不牺牲速度的前提下实现多智能体场景中类人的导航行为。 |
| [^92] | [Continued Training of Continuous Language Flows with Token Identity Squared Supervision](https://arxiv.org/abs/2611.07721) | 提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。 |
| [^93] | [Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption](https://arxiv.org/abs/2611.07720) | 该论文提出将大语言模型纯粹作为特征提取器用于零样本多变量时间序列异常检测，通过直接分析其诱导的特征空间，规避了预测或重建等辅助目标在模型容量增大时会重建异常而变得不可靠的问题。 |
| [^94] | [Pediatric ECG foundation model via adult-to-pediatric fine-tuning: downstream tasks and clinical evaluation](https://arxiv.org/abs/2611.07717) | 该研究证明成人心电图基础模型经儿科数据微调后性能可超越从头训练，具备良好的外部泛化能力和下游任务扩展性，并能显著提升临床医生的诊断准确率。 |
| [^95] | [CAMP: Context-Aware Mapping and Placement for LLMs on 3D-IMC Arrays](https://arxiv.org/abs/2611.07707) | 提出CAMP算法，通过感知LLM推理上下文来优化3D-IMC芯片上的权重映射与放置，提升矩阵向量乘法的并发执行能力和计算资源利用率。 |
| [^96] | [Towards Generalist Graph-Level Anomaly Detection](https://arxiv.org/abs/2611.07706) | 提出GenGLAD，一种通用图级别异常检测方法，通过学习单一统一模型，在极少目标域监督下实现跨多个领域的异常图检测。 |
| [^97] | [InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning](https://arxiv.org/abs/2611.07702) | 提出了 InfMasking，一个基于无限随机掩码的统一变分多模态学习框架，通过在融合前随机掩码各模态特征，并要求所有被掩码的融合表示与参考目标（无监督时为完整融合视图，有监督时为标签类别代理）保持一致，从而迫使模型充分利用模态间的协同信息，而非依赖主导模态或共享信息。 |
| [^98] | [Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction](https://arxiv.org/abs/2611.07695) | 提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。 |
| [^99] | [MCPHallu: Benchmarking Reasoning, Execution, and Memory Hallucinations in MCP Agents](https://arxiv.org/abs/2611.07689) | 该论文提出MCPHallu基准，通过在358个任务中内置结构性触发器，沿推理、执行与记忆三个维度，在36个生产级MCP服务器上实现对四种幻觉失败子类型（分支坍缩、不可达目标、工具误用、上下文遗忘）的子类型级诊断。 |
| [^100] | [ReferClaw: Disambiguating User Intent for Egocentric Assistance](https://arxiv.org/abs/2611.07683) | ReferClaw提出一个多智能体交互框架，通过在回答前主动提出澄清问题并借助成本感知策略来消解自我中心式流式问答中用户意图的歧义，从而生成与用户真实意图对齐的答案。 |
| [^101] | [Randomized Parallel Complexity of Weakly Convex Optimization](https://arxiv.org/abs/2611.07675) | 本文证明了在高维空间中即使随机算法每轮进行多项式数量的并行查询也无法超越标准次梯度方法的速率，但在一维情形下利用实数的序结构，仅用函数值的确定性算法即可实现对数轮复杂度，从而揭示了维度对并行查询收益的决定性作用。 |
| [^102] | [Who Should Know What? Empirical Information Design for Multi-Agent LLM Economies](https://arxiv.org/abs/2611.07667) | 该论文提出“实证信息设计”实验框架，通过在由LLM智能体构成的经济市场中系统比较不同的信息披露策略，发现更详细的信息透明度并不总能改善集体结果，其效果高度依赖于具体模型。 |
| [^103] | [Energy-Aware Open-Vocabulary Detection Scheduling for Low-Power Robotic Autonomy](https://arxiv.org/abs/2611.07663) | 本文提出一种能耗感知的开放词汇目标检测调度器，利用里程计、地图变化和检测器置信度在Jetson Orin Nano上动态切换不同功耗的检测模式，在真实导航试验中将任务成功率从7/10提升至8/10，同时将平均功耗降低5.2%、检测调用频率降低34.0%。 |
| [^104] | [PharmAgents2: Towards Autonomous End-to-End Drug Discovery](https://arxiv.org/abs/2611.07653) | 该论文提出了PharmAgents2——一个将Paper2Skill知识提取与自我扩展式并行探索相结合的自主端到端药物发现系统，在新基准PharmAgents-Bench v0.1及三项真实世界研究中均取得了领先表现。 |
| [^105] | [Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems](https://arxiv.org/abs/2611.07652) | 该论文提出Skillware开放框架，通过模块化、版本化的软件能力包（包含类型化JSON契约、确定性代码、任务指令、数据集和单元测试）取代脆弱的提示工程，将智能体的意图与执行正式分离，实现主权、可审计且确定性的能力执行。 |
| [^106] | [GraspBench: Execution-Aware Cross-Robot Evaluation of 6-DoF Grasp Generation in MuJoCo](https://arxiv.org/abs/2611.07651) | GraspBench是一个在MuJoCo中对6自由度抓取生成流水线进行端到端可执行性评估的跨机器人基准，通过逆运动学过滤、运动规划和固定执行流程，在四种机器人本体和多个物体上系统比较了六种方法，发现GraspGen以73.1%的成功率居首。 |
| [^107] | [What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays](https://arxiv.org/abs/2611.07650) | 本研究对20个提供GPT-6 Astra和Claude Opus 5的第三方LLM API中转服务进行系统性测试，发现约半数低价中转服务的回答行为与官方模型存在差异，并通过检查返回内容、工具权限与计费用量揭示了中转服务在交付质量上的不可靠性。 |
| [^108] | [UniGADD: Universal Generator-Agnostic Deepfake Detector](https://arxiv.org/abs/2611.07647) | UniGADD提出了一种两阶段优化的监督学习方法，通过对比损失学习和仅针对真实样本的嵌入空间细化，实现了对已知深度伪造生成器的高精度检测，并对未知生成技术具有强泛化能力。 |
| [^109] | [WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning](https://arxiv.org/abs/2611.07646) | 该论文提出WaSP，一种基于Wasserstein度量的数据合成流水线，通过在人造数据对上训练并逐步缓解训练分布与目标分布之间的偏移，实现了无需求解器的高效算子学习。 |
| [^110] | [Neural Image Compression with Internal-External Prior Fusion and Alignment](https://arxiv.org/abs/2611.07644) | 该论文提出内外先验融合与对齐（IE-PFA）上下文模型，利用共享可学习字典作为外部先验，与图像内部先验融合并对齐，在单网络内支持64级可变码率编码，实现最先进的率失真性能。 |
| [^111] | [Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks](https://arxiv.org/abs/2611.07640) | 该论文提出以“权限”而非“指令措辞”来重新定义间接提示注入，并构建了涵盖三个工具使用环境、共90个案例的基准测试，专门评估通过篡改正在进行任务的决策结果（而非替换任务）的任务篡改攻击。 |
| [^112] | [Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning](https://arxiv.org/abs/2611.07639) | 该论文构建了一个确定性的GPU/LLM训练故障诊断基准，发现仅用1,200个黄金示范的监督微调即可近乎解决该基准，而从基础模型出发的GRPO强化学习几乎无增益，从而质疑了基于验证器的强化学习相对于监督微调的边际价值。 |
| [^113] | [Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs](https://arxiv.org/abs/2611.07633) | 本文系统比较了四种可用于LLM管道的PII脱敏设计方案（纯正则、Presidio、以及两种混合设计）在速度与准确性上的表现差异。 |
| [^114] | [Beneath the Surface: What We Overlook When We Talk About Agent Evolution](https://arxiv.org/abs/2611.07624) | 该论文系统评估了四种智能体框架演化方法，发现更高的基准分数并不等于可靠、高效且可迁移的改进——演化伴随着任务覆盖的得失权衡与部署成本上升，且得分最高的智能体并非最优选择。 |
| [^115] | [The Popular Get More Popular: Auditing Brand Bias in Generative Search](https://arxiv.org/abs/2611.07614) | 生成式搜索存在系统性品牌偏见：头部热门品牌被引用的频率是平均品牌的1.4至2.8倍，这种偏见源自模型参数记忆且无法通过外部网络检索缓解。 |
| [^116] | [Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone](https://arxiv.org/abs/2611.07610) | 本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。 |
| [^117] | [FisherDistill: Diffusion Distribution Distillation via Closed-Form Fisher Critics](https://arxiv.org/abs/2611.07604) | 提出FisherDistill方法，通过冻结教师模型的中间表示计算闭式Fisher判别方向，无需训练对抗判别器即可对蒸馏后的扩散生成器进行轻量级的分布精细化。 |
| [^118] | [Align Before Aggregation: Basis-Consistent Federated LoRA under Heterogeneous Ranks](https://arxiv.org/abs/2611.07598) | 提出 FedAbA 框架，在聚合前先提取并对齐各客户端 LoRA 因子的局部基，解决了异构秩联邦微调中因基不一致导致的参考前缀不稳定问题。 |
| [^119] | [Evolver: Turning Agent Exploration into Evolving Capabilities](https://arxiv.org/abs/2611.07595) | Evolver 是一个无需更新基础模型参数的经验驱动框架，它通过结构化的 Gene 资产将智能体探索产生的经验转化为持久可复用的能力，在四个基准测试中显著提升任务成功率并将执行 token 消耗降低约 24%。 |
| [^120] | [Learning and Steering in Incomplete-Information Games](https://arxiv.org/abs/2611.07589) | 该论文提出了仅利用轨迹反馈就能在不知道智能体效用的情况下学习标准式和扩展式博弈的二分类型算法，使调解者能够引导自适应智能体走向效用最大化的结果，并达到了近乎最优的样本复杂度。 |
| [^121] | [Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment](https://arxiv.org/abs/2611.07586) | 提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。 |
| [^122] | [SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking](https://arxiv.org/abs/2611.07558) | 提出SKIM——首个面向结构化生成场景的大语言模型智能体剪枝与蒸馏框架，通过将词元位置划分为上下文、决策和格式三类，在剪枝与蒸馏两个阶段中优先保留与决策位置最相关的知识。 |
| [^123] | [The Memory Trap: Language Agents Overgeneralize from Benign Interactions](https://arxiv.org/abs/2611.07554) | 本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。 |
| [^124] | [FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving](https://arxiv.org/abs/2611.07542) | FlashOffloading提出一种以主机DRAM为中心的推理架构，通过滚动HBM缓存、解码热缓存、零开销反向稀疏索引和计算-加载重叠内核，在预填充与解码阶段完全卸载KV缓存，实现长上下文LLM服务中低延迟下的并发度与吞吐量提升。 |
| [^125] | [CST-Bench: Benchmarking Counterfactual Spatial Transition Reasoning with Causal Transition On-Policy Self-Distillation](https://arxiv.org/abs/2611.07541) | 提出了反事实空间推理基准CST-Bench（包含7,082个样本、6种干预类型、12种变换算子及密集因果标注），揭示现有MLLM在物体被干预移动或交换时空间推理仍会失效，并提出因果转变在线策略自蒸馏方法来提升反事实空间推理能力。 |
| [^126] | [Action-Conditioned VLM Monitoring for Autonomous Vehicle Semantic Anomalies](https://arxiv.org/abs/2611.07540) | 该论文提出一种动作条件监测框架：先由视觉语言模型从隐藏结果的视频片段中预测场景所需动作，再通过明确的兼容性规则将预期与实际自车轨迹对比，从而判断自动驾驶行为在语义异常场景中是否恰当，并配套构建了112个配对场景的分类法和生成视频语料库。 |
| [^127] | [Adaptive Experimentation Can Hide Model Error:Detection Limits and Falsification Budget](https://arxiv.org/abs/2611.07529) | 本文证明了自适应实验设计可能通过选择错误模型恰好拟合的设置来掩盖模型误差，提出了检验功效上界、可计算的诊断指标以及无需训练的证伪预算规则，主张在实验设计中主动为证伪模型保留部分预算。 |
| [^128] | [Goal-Branching LaCAM: Temporary Goals as Search Branches in Dense Multi-Agent Pathfinding](https://arxiv.org/abs/2611.07528) | 提出目标分支方法，为阻塞智能体分配临时目标并将其作为LaCAM搜索的额外分支，在保持算法完备性的同时显著提升了密集多智能体路径规划实例的求解成功率。 |
| [^129] | [Graph learning-assisted logic-based Benders decomposition for distributionally robust in-station train rescheduling under disruptions](https://arxiv.org/abs/2611.07520) | 本文提出GNN-LBBD算法，将图神经网络与基于逻辑的Benders分解相结合，求解中断情况下站内列车重调度的滚动时域分布鲁棒优化模型，使平均总完成延误较对比方法降低16.63%。 |
| [^130] | [Concertina: Embodying Manipulation and Cognition in One Diffusion Vision-Language Model](https://arxiv.org/abs/2611.07513) | 提出Concertina——一种具身视觉语言模型（eVLM），将连续轨迹表示、分块生成和物理坐标监督直接融入共享的扩散VLM主干，使操作与认知统一在同一模型中，避免了传统VLA解耦架构的量化误差和梯度竞争问题。 |
| [^131] | [DelegateEval: Evaluating Platforms for Personal External Service Agents](https://arxiv.org/abs/2611.07508) | 提出了DelegateEval——一个用于评估个人外部服务代理平台的首个平台级评估框架，从创建便捷性到效率与价值共八大维度系统衡量个人能否借助代理大规模服务外部客户并创造经济价值。 |
| [^132] | [Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs](https://arxiv.org/abs/2611.07483) | 该论文提出一种带HMAC-SHA256签名的需求清单交接机制，配合确定性代码验证，将多智能体LLM系统在攻击下的不安全提交从69.4%降至9.0%，且不损失合规提交。 |
| [^133] | [SAVeR$^2$: Reasoning-based Safety Alignment for Large Reasoning Models via Verifiable Rewards](https://arxiv.org/abs/2611.07482) | SAVeR²提出了一种基于可验证奖励的推理式安全对齐方法，将安全目标分解为四个细粒度的可验证奖励信号，并结合难度感知的数据选择与保持推理能力的梯度投影，在提升大型推理模型安全性的同时避免损害其推理能力。 |
| [^134] | [Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics](https://arxiv.org/abs/2611.07478) | 该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。 |
| [^135] | [KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap](https://arxiv.org/abs/2611.07477) | KOSMOS 提出一种自调度持久内核库，其工作组利用按消费者需求排序的队列根据数据就绪状态在通信与计算角色间动态切换，将 Transformer 层所有 TP+SP 集合通信与 GEMM 融合，无需调度器或预设切分，在 AMD MI350X/MI355X GPU 上将 Megatron-LM 端到端训练最高加速 1.28 倍。 |
| [^136] | [Adversarial Robustness of Kernel Machines: A Comprehensive Analysis](https://arxiv.org/abs/2611.07474) | 本文通过白盒和黑盒攻击全面评估了核机器的对抗鲁棒性，发现核机器天生比神经网络更鲁棒，并提出了一种核机器的近似对抗训练算法。 |
| [^137] | [Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting](https://arxiv.org/abs/2611.07473) | 该论文提出“残差方向局部滚动一致性”方法，通过沿实际反馈扰动方向惩罚预测器的雅可比向量积这一一阶替代方案，精确抑制时间序列预测中自生成误差对未来预测的传播影响，从而避免设计难以实现的残差到状态对齐算子。 |
| [^138] | [FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance](https://arxiv.org/abs/2611.07464) | FINCOSMOS提出了一个金融智能体的开放世界评估范式：模拟用户隐藏其投资策略，智能体必须自主从网络获取数据并构建防“未来函数”的回测，从而实现无需参考答案、程序可验证的评估。 |
| [^139] | [The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks](https://arxiv.org/abs/2611.07461) | 提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。 |
| [^140] | [IMUDINO: Self-distilled Foundation Model for Inertial Sensing](https://arxiv.org/abs/2611.07437) | IMUDINO是首个在人体活动识别、健康监测、动作捕捉和里程计等多种惯性传感任务上均达到有竞争力性能的通用基础模型，其核心创新在于通过语义保持的运动变换将DINO自监督学习范式适配到惯性数据上。 |
| [^141] | [Typed Knowledge over Constructive Institutions](https://arxiv.org/abs/2611.07432) | 该论文提出 Eigenius——一种基于构造性制度的类型化知识存储，其依赖类型理论将证明无关性作为带显式见证的公理而非转换规则，使知识存储能在“命题成立”之外保留其确立者、推导方式与假设条件等证明记录。 |
| [^142] | [PSGAgent: Enhancing MLLMs' Dynamic Visual Perception for Panoptic Scene Graph Generation via Agentic Reasoning](https://arxiv.org/abs/2611.07408) | 提出了PSGAgent智能体框架，通过“思考-行动-观察”循环以及区域放大和视觉高亮两种工具，动态增强多模态大语言模型在全景场景图生成任务中感知小物体和细粒度关系的能力。 |
| [^143] | [FreePath: Neural Mean Free Path for Image Dehazing](https://arxiv.org/abs/2611.07398) | 提出神经平均自由程框架FreePath，将图像去雾建模为特征空间中的路径状态恢复，通过平均自由程残差块和状态守恒集成实现兼顾物理机制的雾去除。 |
| [^144] | [WeLM-Image: A Unified Image Generation Model towards Visual Intelligence](https://arxiv.org/abs/2611.07392) | WeLM-Image是一个面向大规模消费级应用单GPU部署的统一图像生成基础模型，通过混合Transformer骨干与稀疏混合专家架构将多模态理解融入文本生成图像、指令引导编辑和多参考合成三类任务。 |
| [^145] | [EnvOpt: Automated Optimization for Agent Environments](https://arxiv.org/abs/2611.07372) | EnvOpt提出了对LLM智能体静态环境进行自动化端到端优化的第三条路径，无需评分器或参考答案即可大幅降低智能体成本（最高72.7%）并提升任务通过率。 |
| [^146] | [Test-Time Self-Play: Scaling without a Verifier](https://arxiv.org/abs/2611.07367) | 该论文提出测试时自我博弈（TTSP），一种基于非对称自我博弈、无需验证器的测试时扩展方法，在无验证器的情况下于 LiveCodeBench v6 上实现了 mean@8 提升 35% 的绝对改进，且验证器越弱其优势越明显。 |
| [^147] | [Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair](https://arxiv.org/abs/2611.07342) | 该论文提出一种选择性检索路由框架，由LLM修复智能体编辑知识图谱、路由器智能控制对外部证据检索的访问，并利用违规注入生成的修复结果训练特定部署的路由器，从而提升多重违规知识图谱的修复效果。 |
| [^148] | [Permanent and Transient Representations for Continual Reinforcement Learning](https://arxiv.org/abs/2611.07327) | 本文提出在持续强化学习的互补学习系统中为长期学习采用参数化表示、为瞬态适应采用非参数化表示，并引入一种新颖的非参数近似方法来估计瞬态价值函数，从而实现复杂任务中的在线学习并优于竞争性基线方法。 |
| [^149] | [Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System](https://arxiv.org/abs/2611.07296) | 该论文提出一种两阶段推荐系统，先根据港口与预期登陆点的距离筛选候选港口，再按预估影响天数排序，从而在登陆位置不确定的条件下稳健地预测热带气旋对各港口的影响及干扰持续时间。 |
| [^150] | [How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering](https://arxiv.org/abs/2611.07286) | 本文将基于地图的代码表示（抽象语法树、调用图、模块依赖、文档增强搜索及其混合形式）作为智能体软件工程的设计变量，系统评估了五种代码表示工作流如何影响智能体的检索行为、任务成功率、token消耗与延迟。 |
| [^151] | [Encoding and Propagation of Grammatical Number in Large Language Models](https://arxiv.org/abs/2611.07283) | 该研究利用稀疏自编码器（SAE）深入揭示了 Pythia-70M 小型语言模型中语法数一致现象的表示与处理机制，展示了 SAE 如何用于查找和传播语言模型内部的语法特征。 |
| [^152] | [Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging](https://arxiv.org/abs/2611.07275) | 该论文提出潜在噪声线性马尔可夫决策过程（LNL-MDP）及其桥接方程，通过在噪声空间与动作空间之间建立共享潜在表示来消除“噪声混叠”问题，从而实现计算高效且具有近似最优理论保证的扩散策略微调。 |
| [^153] | [Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target](https://arxiv.org/abs/2611.07270) | 该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。 |
| [^154] | [How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model](https://arxiv.org/abs/2611.07265) | 该研究首次系统评估了EHR基础模型的“时间上下文”对疾病预测的影响，发现将日历回溯窗口从2年延长至最长30年可显著提升结直肠癌的早期预测性能。 |
| [^155] | [Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification](https://arxiv.org/abs/2611.07257) | 该论文提出EANK方法，通过遗传算法优化一个低维策略，根据局部邻域特征（密度、标签熵、投票边际、集中度）为每个实例自适应地确定k近邻分类器的邻居数量k(x)，突破了传统全局单一k值选择的局限性。 |
| [^156] | [TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference](https://arxiv.org/abs/2611.07256) | TriAttention提出了一种统一的稀疏混合精度注意力方法，通过离线校准为每个注意力头在跳过、4位和8位计算之间分配计算区域，在共享计算预算下兼顾稀疏性与量化精度，从而优化长上下文LLM推理的成本与精度平衡。 |
| [^157] | [Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power](https://arxiv.org/abs/2611.07247) | 该论文提出“目标揭示”（即人类潜在目标与实际轨迹之间的互信息）作为统一两种权力度量的桥梁：基于能力的个体权力通过Fano不等式构成其下界，而信息论赋权构成其上界，从而为AI安全中的权力量化研究建立了理论联系。 |
| [^158] | [KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization](https://arxiv.org/abs/2611.07246) | 本文提出知识引导邻域图构建方法KGN，利用从实例中抽取的知识图谱三元组的共享程度作为语义证据来调整嵌入距离并重建top-k邻域图，从而在不使用类别标签的情况下提升嵌入可视化中邻域关系的可靠性，并配套提出IDProfile工具来解释类别标签与邻域多数类之间的残余不匹配。 |
| [^159] | [MAC-v2: Long-horizon model-based value expansion from pixels](https://arxiv.org/abs/2611.07245) | MAC-v2通过训练无重构的潜在世界模型并结合动作分块，实现了从像素出发的长时程基于模型的价值扩展，有效缓解了表征学习与动力学预测误差在想象轨迹上的递归累积问题。 |
| [^160] | [Archetypal contexts in the neurons of language models: an architecture for intelligence?](https://arxiv.org/abs/2611.07234) | 该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。 |
| [^161] | [SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection](https://arxiv.org/abs/2611.07222) | SPILLWAY利用工具返回值的类型化字段（如发件人、账户）与自由文本的区别，提出一种无需语言模型参与决策或逐任务规划的确定性防御方法：只有当工具输出中出现的目标值由只读工具在类型化字段中返回或被固定委托规则覆盖时，才准入该调用，从而防御工具使用型智能体免受提示注入攻击。 |
| [^162] | [TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization](https://arxiv.org/abs/2611.07217) | TileQuant提出以硬件对齐的二维权重tile为精度分配单元的免训练混合精度训练后量化框架，通过离线校准得到的激活感知与梯度感知评分，让高分tile保留16位精度、其余tile使用8位，从而在降低校准成本与解码开销的同时更好地保持LLM质量。 |
| [^163] | [Compositional Unalignment: Auditing Safety in Large Vision-Language Models](https://arxiv.org/abs/2611.07216) | 本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。 |
| [^164] | [SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification](https://arxiv.org/abs/2611.07194) | 本文提出 SeVerus，一种上下文感知、游戏化的 Verus 基准测试生成技术，它利用已验证项目的原始证明依赖关系，通过逐步移除证明上下文将验证任务组织成从易到难的任务链，从而更真实地评估编码智能体自主发现并证明辅助引理的能力。 |
| [^165] | [Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?](https://arxiv.org/abs/2611.07166) | 提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。 |
| [^166] | [Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition](https://arxiv.org/abs/2611.07155) | 该论文发现主动信息获取中对深度前瞻搜索的需求往往源于隐藏原因的描述方式——基于能编码决策信息的合适潜在变量构建的一步式评分，可以用极小的搜索代价替代大部分深度前瞻搜索。 |
| [^167] | [More Agents, More Reliable? System-Level Reliability of Multi-Agent LLMs under Operational-Profile Uncertainty](https://arxiv.org/abs/2611.07140) | 该论文提出了将HIP-LLM单模型不精确概率评估与系统故障逻辑贝叶斯网络相结合的框架，用以在运行剖面不确定性下评估多智能体LLM系统的系统级可靠性，并揭示“更多智能体”并不必然更可靠——多数投票只能防范最弱模型，且投票者自身约5%的失效即可基本抵消其优势。 |
| [^168] | [Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning](https://arxiv.org/abs/2611.07126) | 本文提出混合软提示调优方法，将文本提示与学习到的连续提示嵌入相结合，使开源LLM评审者在偏见、毒性和个性化维度上与人类标注策略的一致性显著提升，甚至超越托管的专有参考模型。 |
| [^169] | [PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems](https://arxiv.org/abs/2611.07123) | 本文提出PromptOps闭环框架，将LLM评判器在部署中与人工评估策略一致性下降的问题定义为“评判器漂移”，通过分离何时优化、应用何种优化流程、选用哪些反馈记录及如何更新提示词这四个运营决策，实现LLM-as-a-Judge系统的持续监控与迭代重新校准。 |
| [^170] | [A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights](https://arxiv.org/abs/2611.07103) | 本文构建了首个通过系统变化数据混合（网络文本、书籍、数学、编程、法语）训练的大规模LLM模型动物园，并提出以数据混合为条件的流匹配模型，能够在Gram-PCA压缩权重空间中直接生成完整的预训练LLM权重，且可零样本泛化到未见过的数据混合。 |
| [^171] | [Polynomial-Time Computation of Extensive-Form Proper Equilibria in Two-Player Zero-Sum Games](https://arxiv.org/abs/2611.07089) | 本文提出了首个确定性多项式时间算法，可在具有完美记忆的有限二人零和扩展式博弈中精确计算（而非近似）扩展式恰当均衡。 |
| [^172] | [Synthesizing symbolic models of human and machine reasoning at scale](https://arxiv.org/abs/2611.07088) | 提出行为条件化符号合成（BCSS）方法，利用AI自动搜索可解释的确定性符号程序，大规模建模人类与机器的推理行为，并揭示两者在推理中都存在过度简化和错误推断问题特征依赖关系的共同特征。 |
| [^173] | [When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization](https://arxiv.org/abs/2611.07073) | 本文提出跨智能体Dafny形式化协议，证明验证通过并不等于真正独立的推理一致，且一致率对模型能力在规约者与实现者两个角色间的分配极为敏感（较强模型担任实现者时为53.0%，角色互换后骤降至14.3%）。 |
| [^174] | [Shaping & Monitoring Models through their Internal Representations](https://arxiv.org/abs/2611.07071) | 该论文提出“机制塑造”统一框架，在训练中通过约束模型内部表示所表达和依赖的概念来塑造模型行为，在多领域、多规模、多训练范式下显著减少不良行为同时保持任务性能，并引入监控器类别的层级体系来评估被塑造概念是否仍然可被监控。 |
| [^175] | [Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs](https://arxiv.org/abs/2611.07069) | 该论文针对POMDP中采用非线性神经网络价值函数逼近的强化学习，首次建立了有限时间收敛性与遗憾保证，克服了以往理论仅适用于线性逼近或依赖不符合POMDP轨迹特性的独立同分布采样假设的局限。 |
| [^176] | [Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration](https://arxiv.org/abs/2611.07066) | 提出EFXPLORER约束探索框架，利用上图重构引入自适应探索预算，将内在奖励与任务目标分离，在保持任务性能的前提下促进多智能体协作行为的发现。 |
| [^177] | [SD-LoRA: Training-Time Structural Distillation into LoRA for Few-Shot Vision-Language Adaptation](https://arxiv.org/abs/2611.07063) | SD-LoRA提出一种结构蒸馏框架，仅在训练时通过异构图教师进行token级图推理（建模patch间、全局-局部及视觉-文本关系），并将其效果迁移至轻量级LoRA适配的CLIP模型，在不增加推理开销的前提下提升少样本视觉-语言适配能力。 |
| [^178] | [Segment What It Can Draw: Querying a Text-to-Image Generator with Its Own Images for Skin Lesion Segmentation](https://arxiv.org/abs/2611.07061) | 提出免训练方法GENIQ，利用文生图扩散模型自身生成的健康皮肤与病变图像构建更可靠的概念方向，并结合未标注皮肤镜图像中最具代表性的图像块进行细化，从而在无需标注掩码的情况下实现皮肤病变分割。 |
| [^179] | [Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale](https://arxiv.org/abs/2611.07054) | 研究构建了基于智能体大语言模型的PRISM流水线，能在大规模在线健康社区中筛查基因变异重新解读相关的讨论，发现123个候选分类差异（其中5个从意义未明升级为致病性），从而触及临床重新联系无法覆盖的患者群体。 |
| [^180] | [Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration](https://arxiv.org/abs/2611.07051) | 该论文提出面向已部署智能体的“闭环评估”方法，将评估建模为有偏评级者与受限结果下的性能估计，通过片段/轨迹/对话/参与四层级分析、版本化规则、溯源与裁定组件以及智能体工厂生成的评估者智能体，实现可追溯、可自我校准并能反馈至智能体本身的持续评估体系。 |
| [^181] | [Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI](https://arxiv.org/abs/2611.07048) | 该论文提出了一个比较达尔文式、拉马克式与智能体导向演化的理论框架，发现当学习足够有效时三种机制的适应速度依次加快，生成更多变体可加速适应，而最优搜索努力取决于学习能力、问题难度与实验成本。 |
| [^182] | [Trust Responsive Runtime Authorization for Agentic AI with Empirical Characterization of Policy Behavior and Parameter Sensitivity](https://arxiv.org/abs/2611.06996) | 该论文提出并实证刻画了信任响应式授权策略（TRAP），通过非对称规则将信任、稳定性与风险综合而成的授权就绪分数映射为运行时授权级别，实验表明其在突发退化时能即时限制权限、恢复后更快重新授权，并在各自适应策略中具有最高的稳定性。 |
| [^183] | [Autonomous AI Agents Systematically Diverge From Human Preferences in Proof Formalization](https://arxiv.org/abs/2611.06995) | 论文提出“模拟协作”方法，基于Lean中九次大型人类数学协作的数据，首次系统揭示了AI智能体在证明形式化中与人类偏好的系统性差异——AI更容易被明确标记的空白吸引，回避密集连接的证明中心和复杂证明目标，并倾向于将出人意料的证明主体规范化。 |
| [^184] | [Batch-Conditioned Semantic Anchors for Robust Transductive Adaptation of Vision--Language Models](https://arxiv.org/abs/2611.06993) | 提出JPTA框架，将直推式视觉-语言模型自适应建模为批条件语义参考估计，通过联合估计批次先验与软图像侧原型，仅在经校准的证据下让CLIP文本语义参考向批次结构偏移，从而在利用批次信息与保留文本语义之间实现鲁棒平衡。 |
| [^185] | [The Two-Ledger Model: Transaction-Grain Cost Accounting for Multi-Agent AI Systems Across Clouds, Models, and Systems of Record](https://arxiv.org/abs/2611.06989) | 该论文提出“双账本模型”，通过将符合FOCUS规范的结算级账单数据与基于OpenTelemetry GenAI遥测构建的调用粒度归因账本相连接，解决了多智能体AI系统中结算粒度账单与业务事务粒度成本核算之间的粒度不匹配问题。 |
| [^186] | [Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER](https://arxiv.org/abs/2611.06988) | 该论文在 MONSTER 基准上对冻结的预测基础模型（Reverso 和 Chronos-Bolt）表示进行受控评测，发现预训练相比随机权重模型仅带来小幅且不稳定的分类准确率提升，表明这些预训练表示未必优于无需预训练的强分类器，尤其在标注稀缺时其价值存疑。 |
| [^187] | [Resource-Efficient Reinforcement Learning Post-Training through Elastic Expert Parallelism](https://arxiv.org/abs/2611.06984) | 提出 FlexFloor 系统，通过联合调整响应调度与活跃 worker 组，利用历史响应长度和内存约束实现同步响应生成过程中 worker 的渐进式提前释放，从而大幅提升混合专家模型强化学习后训练的资源效率。 |
| [^188] | [BlockLock: Reproducible LLM Inference with Fixed Batch Geometry](https://arxiv.org/abs/2611.06977) | BlockLock通过仅固定受保护请求所见的批次几何形状（而非约束所有算子或放弃共享计算），在保留共享批次预填充的前提下实现了逐位可复现的LLM推理。 |
| [^189] | [KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks](https://arxiv.org/abs/2611.06972) | KDASO提出了一种知识与数据双驱动的自动技能优化框架，通过从最新裁定数据归纳新技能、基于执行轨迹对齐优化技能描述以及上下文预算路由，解决了LLM在责任裁定任务中技能库过时、语义执行鸿沟和关键技能遗漏三大问题。 |
| [^190] | [Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems](https://arxiv.org/abs/2611.06969) | 该论文提出了一种针对大语言模型主动对话系统的交互式后门攻击框架，通过训练数据投毒植入触发条件下的引导策略偏差，使普通用户话语即可将触发后的多轮交互引导至攻击者指定目标，无需部署时修改提示词或攻击者进一步干预。 |
| [^191] | [Formalizing inverse problems in the age of AI: possibilities, pitfalls, and an invitation](https://arxiv.org/abs/2611.06962) | 本文介绍了AutoIP——一个由AI智能体将经典线性正则化理论及其泛函分析基础自动形式化的Lean库，指出形式化的风险主要在于陈述而非证明，审阅者只需核对陈述与原文即可，且对陈述的简单自动化测试甚至能发现已发表文献中的问题。 |
| [^192] | [KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review](https://arxiv.org/abs/2611.06950) | KG-AST利用git历史和代码图特征对提交进行风险评分，优先将测试与AI审查预算分配给最可能引入缺陷的提交，在11种语言的155个开源项目上显著优于随机选择和最大变更优先基线。 |
| [^193] | [From Maps to Metabolism: Historical Precedents and a Unified Critique of Computationalism](https://arxiv.org/abs/2611.06936) | 本文综合科日布斯基、塞尔、弗洛里迪等多位学者的历史先例，提出一种非计算物理主义，主张意识依赖于以新陈代谢为基础的物理动力学而非抽象计算，但并不排除人工意识的可能性。 |
| [^194] | [BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks](https://arxiv.org/abs/2611.06934) | 提出BrainPFN——一种在由约7千个真实功能连接矩阵构建的领域先验上进行元训练的先验数据拟合网络，通过摊销式推断实现可跨数据集迁移、在小样本临床/认知场景下依然准确的大脑-行为预测，克服了核岭回归需逐评分调参、无法迁移的局限。 |
| [^195] | [Oscillatory Phase Coding Emerges in Adaptive RNNs for Flexible Working Memory](https://arxiv.org/abs/2611.06932) | 该研究发现在具有适应性的循环神经网络中训练可变延迟的序列工作记忆任务时，振荡相位编码会自发涌现，其中振荡幅度编码项目身份、相位编码序列顺序，且该现象依赖于神经适应性、可变延迟和信息读出灵活性三个关键要素。 |
| [^196] | [ApproxBrain: Certified Sufficiency Frontiers in Connectome-Constrained Neural Systems](https://arxiv.org/abs/2611.06929) | 该论文提出ApproxBrain框架，将神经回路的“计算充分性”从单一最小回路的点估计扩展为经统计认证、随代价变化的充分性前沿，并据此量化回路的不可分解性、退化性与功能角色分工。 |
| [^197] | [A minimal dynamical model of human-AI cognitive coevolution](https://arxiv.org/abs/2611.06928) | 该论文提出了一个基于流体智力与晶体智力区分的最小动力学模型，用于刻画生成式AI引发的认知卸载如何同时产生正面与负面影响，从而塑造人类认知与机器智能的协同演化。 |
| [^198] | [EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models](https://arxiv.org/abs/2611.06748) | 提出EvoWorld模型与“视野外因果演化”新问题，使生成式视频模型不仅能渲染逼真画面，还能推理事件对世界实体的影响并生成因果一致的世界演化。 |
| [^199] | [Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment](https://arxiv.org/abs/2611.06543) | 提出在线评分准则自蒸馏（ORSD）方法，从同策略轨迹的跨轨迹行为差异中自生成动态自然语言过程评分准则，实现密集的令牌级信用分配监督，以增强以结果为中心的强化学习。 |
| [^200] | [Learning Noise Optimization for Reward Alignment in Few-Step Generative Models](https://arxiv.org/abs/2611.06255) | 提出LeNO方法，将奖励引导的初始噪声优化摊销为学习到的残差传输，在冻结预训练生成器的前提下实现少步生成模型的高效奖励对齐，并通过KL散度上界提供理论正则化。 |
| [^201] | [Grounding in the Dark: Text-Only Training for Zero-Shot Video Temporal Localization](https://arxiv.org/abs/2611.06149) | 提出一种仅用文本视频描述训练大语言模型（无需配对视频数据和视觉-语言预训练）来实现零样本视频时序定位的方法，通过LoRA在LLM2CLIP文本嵌入上适配Qwen3-8B，在推理时才首次接触视频即可取得优异的时序定位效果。 |
| [^202] | [Whiel: A Neurosymbolic Platform for Certified Relational Program Verification](https://arxiv.org/abs/2611.05612) | Whiel 是一个基于 Lean 认证的神经符号验证平台，它把以数据库实例为状态的命令式关系程序与 Hoare 风格推理、神经符号循环不变量发现及 Vampire 定理证明器相结合，实现了对 Datalog 等查询语言的包含/等价问题以及数据集成中完整性约束、模式映射与视图规范的可认证验证。 |
| [^203] | [LIBRA: Programmatic Induction of Executable Tool Libraries](https://arxiv.org/abs/2611.05445) | LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。 |
| [^204] | [RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On](https://arxiv.org/abs/2611.05163) | RoboTryOn 将人机转换类比为虚拟试穿——保持人类演示的动作不变，将人替换为由参考图像指定的目标机器人——从而从人类视频合成出视觉一致、保留任务与场景的灵巧机器人演示视频，为高自由度机器人提供可扩展的演示数据来源。 |
| [^205] | [P$^2$-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors](https://arxiv.org/abs/2611.03953) | 该论文提出 P$^2$-MOPD，将学生与专家参数的线性组合视为一个可微模型，在在线策略蒸馏目标下联合优化混合系数与学生权重，使专家作为参数先验而非限制学生，从而在四个领域上仅用一半训练步数就实现了比 MOPD 更快、更均衡的能力恢复。 |
| [^206] | [Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training](https://arxiv.org/abs/2611.03577) | Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。 |
| [^207] | [Human-Centred, Risk-Constrained Evaluation of Financial Document Recognition: Evidence, Critical Errors, and Selective Automation](https://arxiv.org/abs/2611.02775) | 本文提出了一个以人为中心、风险约束的财务文档识别评估框架，通过类型化财务事实、分层质量模型和带单侧不确定性界限的非补偿性验收门限，防止总体高分掩盖金额、标识符等关键字段的错误，从而支撑可靠的选择性自动化。 |
| [^208] | [OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining](https://arxiv.org/abs/2611.01844) | OpenNWM通过视觉预测预训练从1750万条无动作导航视频中学习通用潜在动作，并结合动作条件后训练将其与物理运动对齐，从而构建出一个能跨真实环境泛化的通用导航世界模型。 |
| [^209] | [Vision-Language Models Are Joint Denoisers in Robotic Action Generation](https://arxiv.org/abs/2611.01215) | 提出JADE方法，让视觉-语言模型在每个去噪步骤中读取含噪动作并通过门控交叉注意力动态引导动作专家，使预训练的感知与推理能力与动作生成实现联合去噪，显著提升分布外场景下的机器人操作表现。 |
| [^210] | [Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments](https://arxiv.org/abs/2611.00597) | 本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。 |
| [^211] | [Taurus: LLM-in-the-Loop CT Annotation Matches Coding Agents at Half the Cost](https://arxiv.org/abs/2611.00247) | 提出了LLM在环的CT主动脉标注框架Taurus，通过自动化标注加模型审查的流程，在覆盖六个公开数据集的盲测标签恢复评估中，以一半成本达到与自由形式编码智能体相当的标注水平。 |
| [^212] | [AuroraWorld: Structured 4D World Modeling for Exo-to-Ego Video Generation in Robot Manipulation](https://arxiv.org/abs/2611.00177) | 提出AuroraWorld，一种结构化4D世界建模方法，能够从大规模外部视角机器人操作视频中合成物理一致、且对视觉运动策略学习有用的第一人称视角视频。 |
| [^213] | [TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents](https://arxiv.org/abs/2610.41697) | TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。 |
| [^214] | [GAAC: Lossless KV Compression for Bandwidth-Efficient LLM Inference on Personal Devices](https://arxiv.org/abs/2610.38210) | GAAC利用BF16指数的冗余性实现无损KV缓存压缩（约1.4倍），并通过融合注意力内核即时重建KV数据，在不损失任何精度的情况下加速个人设备上的长上下文LLM推理。 |
| [^215] | [Merge and Branch: Efficient Tree-Structured Supervised Fine-Tuning of LLMs for Recommendation](https://arxiv.org/abs/2610.37973) | 提出 Merge and Branch 方法，通过将历史物品压缩为少量可学习 token，并以浅而宽的树状结构结合树形 FlashAttention 实现历史计算共享与多目标分支预测，大幅降低 LLM 生成式推荐监督微调的输入成本并提升学习效率。 |
| [^216] | [Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution](https://arxiv.org/abs/2610.37407) | 本文提出一种基于DIO/ODI的授权护栏，将智能体推理可用的证据与可授权持久性身份变更的证据分离，并通过两阶段监督器仅在授权合格证据充分支持所有必需命题、冲突已解决且授权范围与策略版本有效时才允许提交身份变更，从而防止错误身份假设在智能体系统中自我确认。 |
| [^217] | [V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models](https://arxiv.org/abs/2610.37019) | V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。 |
| [^218] | [AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations](https://arxiv.org/abs/2610.36980) | AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。 |
| [^219] | [ACSAN: All-Convolutional Self-Attention Network Integrating Convolutional Operations with Vision Transformer](https://arxiv.org/abs/2610.35375) | 提出ACSAN模型，通过动态卷积自注意力（DCSA）模块用多尺度膨胀卷积替代ViT中的全连接矩阵乘法，并结合空间自注意力与通道自注意力，从而减少参数量和对大规模训练数据的依赖，使其更适用于资源受限的场景。 |
| [^220] | [The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents](https://arxiv.org/abs/2610.34763) | 该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。 |
| [^221] | [More: Mode-Optimized Reasoning with Self-Adaptive Effort for Self-Evolving Agents](https://arxiv.org/abs/2610.33684) | 提出MORE自我进化训练框架，通过MATE-PPO使智能体能为每个任务自适应地选择推理模式（直接回答、短/长思维链或代码推理）和计算努力，在保持性能的同时优化token消耗。 |
| [^222] | [PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models](https://arxiv.org/abs/2610.31429) | PhysTrace是一个无需训练的可解释评估框架，它利用多模态大语言模型构建基于历史事件的物理预期，并通过时序视频-语言对齐追踪预期与视频内容的一致性，从而实现对视频生成模型物理一致性的时序评估，在IntPhys2上达到67.2%的判别准确率。 |
| [^223] | [ValuSpec: Plug-and-Play Candidate Valuation before Target Verification for Tree-Based Speculative Decoding](https://arxiv.org/abs/2610.31338) | ValuSpec是一种即插即用的候选评估机制，通过免训练的阈值估计在目标模型验证前过滤候选树中的低质量分支，从而提升树形推测解码的验证效率并节省目标侧计算开销。 |
| [^224] | [Robustness of Internal Signals for Hallucination Detection in Vision-Language Models](https://arxiv.org/abs/2610.27041) | 本研究深入分析了基于VLM内部信号的幻觉检测方法的鲁棒性，揭示了许多检测方法实际上依赖token位置这一捷径，而一个朴素的token位置基线即可达到有竞争力的性能，并探讨了检测方法对语言和视觉模态的敏感性。 |
| [^225] | [Reward Engine: Reward-Centric Self-Evolution for Open-Ended Tool Agents](https://arxiv.org/abs/2610.22915) | 提出Reward Engine，通过从任务和初始轨迹中构建包含测试用例与评分标准的冻结RewardPack，为缺乏可执行真值奖励的开放式工具代理任务提供稳定的奖励信号，引导代理自我修正并支持强化学习优化。 |
| [^226] | [Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction](https://arxiv.org/abs/2610.19246) | 提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。 |
| [^227] | [Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap](https://arxiv.org/abs/2610.19065) | 生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。 |
| [^228] | [Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench](https://arxiv.org/abs/2610.18938) | 该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。 |
| [^229] | [What Does the Effective Dimension of a Hybrid Quantum Neural Network Actually Measure?](https://arxiv.org/abs/2610.14070) | 本文证明混合量子神经网络研究中实际记录的“有效维度”标量只是泡利-Z输出敏感度的参与率，并非理论定义的量子费舍尔信息矩阵全矩阵参与率，并通过大规模实验表明该标量不能可靠地衡量模型容量。 |
| [^230] | [Autonomous Driving Research Requires a Community-Driven Data Paradigm](https://arxiv.org/abs/2610.08825) | 自动驾驶研究的瓶颈不在建模而在数据范式，亟需建立社区驱动的协作数据机制，以充分利用全球已积累的600多个自动驾驶数据集。 |
| [^231] | [MedZERO: Self-Evolving Agents for Open-Ended Medical Reasoning Through Controlled Knowledge Accumulation](https://arxiv.org/abs/2610.08327) | MedZERO 提出了一个自我进化智能体框架，通过“考官”生成前沿医学题目、“推理者”进行基于证据的多轮推理，并借助受控知识积累机制，使大语言模型无需昂贵专家监督即可在开放、难以完全验证的医学推理领域实现自我提升。 |
| [^232] | [A Trust Layer for Agent Evaluation](https://arxiv.org/abs/2610.07274) | 该论文提出“智能体评估信任层”，一个附加式后验框架，通过验证评分逻辑支持、可追溯计算、完成声明一致性和重复执行稳定性这四个属性，来判断智能体在基准测试中获得的分数是否可信。 |
| [^233] | [Best-of-$N$ Guidance for Test-time Diffusion Alignment](https://arxiv.org/abs/2610.05108) | 该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。 |
| [^234] | [MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification](https://arxiv.org/abs/2610.03056) | 该论文提出了MOF-VERIFY，一个失效感知的智能体框架与四任务族诊断基准，通过闭卷、检索和先知证据等设置系统评估并定位大语言模型在MOF假设验证中的失效点（涵盖结构接地、合成条件、证据充分性和MLIP计算验证）。 |
| [^235] | [VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning](https://arxiv.org/abs/2610.02801) | VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。 |
| [^236] | [PickMoment: Continuous-Time Single-Image-to-Video via Learning Deblurring and Blur-to-Video](https://arxiv.org/abs/2610.01279) | 提出PickMoment连续时间框架，用单一确定性模型学习任意曝光子区间的平均模糊，统一了单图像去模糊和模糊到视频任务。 |
| [^237] | [Backdoor Purification for LoRA-Tuned LLMs via Null-Space Projection](https://arxiv.org/abs/2610.00685) | 提出了一种基于零空间投影的后门净化方法，无需触发器先验知识、干净参考样本或重新训练，即可净化遭受后门攻击的LoRA微调大语言模型，在降低攻击成功率的同时保留模型的通用能力与下游技能。 |
| [^238] | [Backdoor Containment via Expert Quarantine and Shutdown in LLMs](https://arxiv.org/abs/2610.00663) | 本文提出隔离专家关停策略，通过正则化引导的MoE类架构让后门在训练中形成但被引导至指定的隔离专家组件，部署时可将该组件关停，从而以计算高效的方式遏制LLM后门攻击。 |
| [^239] | [Manifold-Constrained Initial Noise Optimization for Efficient Generative Model Alignment](https://arxiv.org/abs/2610.00365) | 提出ZeNOVA，一种免梯度的初始噪声优化方法，通过退火软值引导、流形约束超球面朗之万动力学和Metropolis-Hastings跳跃，在黑盒奖励场景下实现稳定高效的生成模型对齐。 |
| [^240] | [Structure-aware Reinforcement Learning for Protein Directed Evolution](https://arxiv.org/abs/2609.39048) | StructEvo提出了一种结构感知的强化学习框架，通过增量结构融合编码器动态整合空间知识，并利用结构对齐的分层动作网络分解突变空间，在克服突变体结构稀缺难题的同时超越了现有的蛋白质定向进化方法。 |
| [^241] | [Diagnosing and Improving Probabilistic Reasoning in Large Language Models](https://arxiv.org/abs/2609.38005) | 该论文提出一个决策理论框架，将大语言模型的决策损失分解为信念形成与行动选择两个组成部分，并通过合成基准和强化学习干预系统诊断并改进大语言模型的概率推理能力。 |
| [^242] | [From Learner Behavior to Reusable Skills for Effective and Efficient Learner Simulation](https://arxiv.org/abs/2609.37157) | Learner2Skill将从历史交互中获得的学习者模拟能力外化为持久可复用的“模拟技能”，该技能捕获学习者的学习状态与回答模式并随新交互演进，可通过轻量级校准迁移到新的大语言模型，从而更高效、更忠实地模拟学习者行为。 |
| [^243] | [SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents](https://arxiv.org/abs/2609.35596) | 该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。 |
| [^244] | [AnchorRep: Defending LLMs Against Cross-Model Adversarial Transfer via Representation Repulsion](https://arxiv.org/abs/2609.32602) | 提出AnchorRep防御方法，利用轻量级LoRA适配器将有害提示词的内部表示与锚点模型的表示相排斥，无需对抗样本即可将跨模型对抗迁移攻击的成功率降低至1.1%以下。 |
| [^245] | [Can Linguistic Reasoning Vectors Enhance Multimodal Reasoning Ability?](https://arxiv.org/abs/2609.31140) | 该论文提出LIFT方法，通过提取“推理者路径”与“求解者路径”之间的隐状态差异作为推理向量并注入模型，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。 |
| [^246] | [Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale](https://arxiv.org/abs/2609.30137) | 该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。 |
| [^247] | [RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking](https://arxiv.org/abs/2609.23457) | 本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。 |
| [^248] | [The Price of Safety: Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents](https://arxiv.org/abs/2609.22818) | 该研究提出了一种控制变量的测量方法，系统评估LLM智能体记忆投毒防御在良性流量下的效用损失与令牌开销，发现写入时防御几乎不带来可察觉的效用成本。 |
| [^249] | [A Cyber Range Evaluation of Autonomous Network Incident Response Agents](https://arxiv.org/abs/2609.16541) | 该论文在网络靶场环境中评估了自主网络事件响应代理，发现基于强化学习的代理在防御网络入侵方面总体上优于启发式策略，且性能表现高度依赖于红队的攻击策略。 |
| [^250] | [DynSHAP: Towards Explainable Dynamic Survival Analysis](https://arxiv.org/abs/2609.13042) | DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。 |
| [^251] | [Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality](https://arxiv.org/abs/2609.06598) | 该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。 |
| [^252] | [Value Over Language Model: Detecting Original Contribution in Writing](https://arxiv.org/abs/2609.00700) | 提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。 |
| [^253] | [Integrating Triaxial IMU Sensors and Ensemble Learning for Effective Parkinson Disease Severity Classification](https://arxiv.org/abs/2608.28602) | 本文提出一种结合三轴IMU传感器（加速度计与陀螺仪）采集的运动和震颤数据与集成学习方法的人工智能系统，用于实现有效的帕金森病严重程度分类。 |
| [^254] | [openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents](https://arxiv.org/abs/2608.27969) | openJiuwen 是一个开源智能体框架，通过“结构可组合性”与“运行时自适应性”两大设计，使长时程编码智能体能够灵活组合能力并根据任务产生的动态证据自适应调整执行。 |
| [^255] | [AI Slop and Hallucinations in Vulnerability Assessment: A Survey on Reasoning Failures and Trustworthy Mitigation](https://arxiv.org/abs/2608.25667) | 本综述揭示了LLM在漏洞评估中产生的“AI垃圾内容”（幻觉漏洞、错误补丁、重包装报告）的根源在于安全专家的因果演绎推理与模型自回归概率生成之间的根本鸿沟，并提出了演绎覆盖率得分作为可测量指标，以及思维链提示与工具增强智能体等可信缓解路径。 |
| [^256] | [A Locally Tokenized Generative Model for Robust Time-Series Watermarking](https://arxiv.org/abs/2608.19727) | 本文提出了一种局部令牌化生成模型L-VQVAE和配套水印方法LVQMark，通过确保每个恢复单元仅依赖有界时间邻域，解决了时间序列水印在后编辑攻击下的检测可靠性问题。 |
| [^257] | [GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction](https://arxiv.org/abs/2608.18234) | 本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。 |
| [^258] | [Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases](https://arxiv.org/abs/2608.17168) | 本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。 |
| [^259] | [Parameter Exploration for RLVR via Variational Learning](https://arxiv.org/abs/2608.09805) | 本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。 |
| [^260] | [Population-Scalable Multi-Agent World Modeling](https://arxiv.org/abs/2608.08600) | 本文提出Khora，一种可扩展的多智能体世界模型，通过解耦世界状态演化与视觉渲染，实现推理时无需重新训练即可扩展到任意数量的智能体。 |
| [^261] | [Unified Hallucination Fuzzing for Multimodal Large Language Models](https://arxiv.org/abs/2608.07525) | 该论文提出了结合统一幻觉分类细粒度数据集UniHall与自适应多模态模糊测试SAMF的系统性评估框架，利用进化变异策略动态探索多模态大语言模型的幻觉边界，克服了传统静态基准覆盖面窄和性能饱和的局限。 |
| [^262] | [From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems](https://arxiv.org/abs/2608.06112) | 该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。 |
| [^263] | [Eigenius: A Typed Knowledge-Graph DBMS with Epistemic Stratification and Institution-Mediated Reasoning](https://arxiv.org/abs/2608.04457) | Eigenius通过统一内核将数据溯源作为结构性不变量，利用依赖类型、制度和内容寻址存储实现认知状态的强制管理。 |
| [^264] | [Variable-aware representation, not relation typing, explains the performance of an event-graph model for ICU mortality prediction: temporal and external validation](https://arxiv.org/abs/2608.02663) | 该研究通过严格的时间性与外部多中心验证发现，事件图模型在ICU死亡率预测中的性能提升主要来自变量感知表示（变量学习嵌入加逐变量归一化），而非关系类型化的消息传递。 |
| [^265] | [Local-Frame Rotary Attention for Antibody-Specific Epitope Prediction](https://arxiv.org/abs/2608.01092) | 提出局部坐标系旋转注意力（LFRA），将主链局部坐标系下的残基间三维位移直接注入旋转注意力，以取代传统一维序列位置机制，从而提升抗体特异性表位预测的准确性。 |
| [^266] | [TraceViT: Grounded Trace Supervision for Visual Abstract Reasoning](https://arxiv.org/abs/2607.29586) | 提出TraceViT，通过语义单调变换链和软轨迹对齐来监督循环视觉推理器的中间迭代步骤，使其逐步遵循变换过程，从而提升ARC视觉抽象推理能力。 |
| [^267] | [When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles](https://arxiv.org/abs/2607.23379) | 该论文发现，微调后的激活神谕可能变成“概念特异性反读取器”，即选择性地无法读取其自身训练中持续存在的隐藏概念，而非成为专业读取器。 |
| [^268] | [Invariant Discovery for Networked Systems](https://arxiv.org/abs/2607.22944) | 本文提出 Autogram，通过让 LLM 生成紧凑的候选不变量文法并在其内进行确定性统计搜索、再迭代扩展文法直至恢复已知可信不变量，实现了无需人工文法且能容忍真实噪声的网络系统不变量自动发现。 |
| [^269] | [Differencing the Diffusion Trajectory toward Uncertain Components for Time Series Forecasting](https://arxiv.org/abs/2607.22599) | 提出DiffDiff框架，将时间序列预测中“缓慢变化成分由历史锚定、高频动态承载残余不确定性”的可预测性不对称性直接嵌入扩散轨迹本身，使单一端到端的扩散过程能够感知目标中哪些部分已被历史确定，从而改进概率时间序列预测。 |
| [^270] | [World Models as Adversaries: Multi-Agent Self-Play Fine-Tuning for Robust Motion Planning](https://arxiv.org/abs/2607.10630) | 该论文提出对抗世界建模（AWM）框架，将规划器自身的世界模型转化为角色条件化的对抗性对手，通过多智能体自博弈微调生成稀疏且场景自适应的攻击，从而在罕见安全关键场景下训练出鲁棒的自动驾驶运动规划器。 |
| [^271] | [HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion](https://arxiv.org/abs/2607.03012) | 提出HyperVAttention——一种无需训练的稀疏注意力框架，通过3D局部窗口聚类和自定义Triton内核，同时降低视频扩散Transformer的聚类开销并提升CTA利用率，实现高效的长视频生成加速。 |
| [^272] | [The State-Prediction Separation Hypothesis](https://arxiv.org/abs/2607.01218) | 该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。 |
| [^273] | [Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation](https://arxiv.org/abs/2606.28925) | 该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。 |
| [^274] | [Agentic Abstention: Do Agents Know When to Stop Instead of Act?](https://arxiv.org/abs/2606.28733) | 该论文提出了“智能体弃权”这一新概念，将LLM智能体在多轮交互中判断何时应停止行动定义为序列决策问题，并在超过28,000条任务指令上评估了13个LLM智能体系统和2个智能体脚手架。 |
| [^275] | [A Latent ODE Approach to Spatiotemporal Modeling of Cine Cardiac MRI](https://arxiv.org/abs/2606.26718) | 提出了一种基于潜在常微分方程的动态模型，通过连续潜在轨迹编码心脏MRI的时空信息，并利用偏差预测心力衰竭风险，显著提升了传统风险模型的性能。 |
| [^276] | [EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling](https://arxiv.org/abs/2606.26578) | 提出EvoOptiGraph框架，通过将优化模型表示为图结构并利用弱点驱动的协同进化，自动生成结构多样的训练数据，从而提升大语言模型在优化建模任务上的性能。 |
| [^277] | [ThermoLLM: LLM-Based Multi-Zone HVAC Control with a Spatiotemporal Context](https://arxiv.org/abs/2606.22911) | 本文提出ThermoLLM，将建筑空间语义、区域热动力学与滚动历史作为LLM上下文的核心，从而实现更一致、更可泛化且能感知区域间延迟热交互的多区域HVAC控制。 |
| [^278] | [Keeping Jerk in Check: Jointly Trained Perception-Gated Experts for Smooth Terrain Transitions](https://arxiv.org/abs/2606.19633) | 该论文提出CTS-MoE，将多地形腿式运动表述为多任务强化学习，通过联合训练感知门控的稠密专家混合actor与任务特定的critics，利用软路由组合专家知识，实现跨越不连续地形时连续平滑且保留专业化的运动行为。 |
| [^279] | [Volterra Generative Models via Markovian Lifting](https://arxiv.org/abs/2606.18071) | 本文提出通过马尔可夫提升方法构建Volterra生成模型，将非马尔可夫的Volterra过程转化为可处理的马尔可夫近似，从而支持重用预训练去噪器，并在MNIST、CIFAR-10和ImageNet基准上取得了优异的FID性能。 |
| [^280] | [LatentGym: A Testbed For Cross-Task Experiential Learning With Controllable Latent Structure](https://arxiv.org/abs/2606.15306) | 提出了LatentGym测试平台，通过围绕真值潜在变量构建可控环境，实现了对智能体跨任务经验学习能力的度量，并可将探索与利用行为分开评估。 |
| [^281] | [StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance](https://arxiv.org/abs/2606.14571) | 该论文提出StreamMemBench——首个基于EgoLife第一人称视频流的智能体记忆流式评估基准，通过两步任务序列和四个指标系统评估记忆的证据召回、证据使用、反馈整合与后续复用能力。 |
| [^282] | [The Standard Interpretable Model: A general theory of interpretable machine learning to deductively design interpretable methods using Lagrangian mechanics](https://arxiv.org/abs/2606.12289) | 本文提出了一种基于拉格朗日力学的标准可解释模型（SIM）理论，通过演绎方式设计可解释方法，填补了可解释性理论与方法之间的空白。 |
| [^283] | [When Attribution Patching Lies: Diagnosis and a Second-Order Correction](https://arxiv.org/abs/2606.09899) | 本文揭示归因修补的主要误差源于下游网络的非线性而非被修补组件的局部曲率，并据此提出可靠性分数、误差界限与二阶修正三项工具来诊断和改进归因修补的可信度。 |
| [^284] | [Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation](https://arxiv.org/abs/2606.09864) | 研究发现KV缓存低比特量化会在困惑度等常规指标不变的情况下隐性地破坏大语言模型的安全对齐，其根源在于安全特征位于对量化噪声极其敏感的低维子空间，据此提出的PCR诊断方法可用于识别并缓解此类失效。 |
| [^285] | [Hypnos: Learning Generalisable Representations of Sleep Physiology via Next-Token Prediction](https://arxiv.org/abs/2606.09605) | 该论文提出Hypnos多模态睡眠基础模型，以下一词元预测作为掩码重建和对比学习的简单可扩展替代方案，从超过2万份多导睡眠图记录的八种传感模态中学习睡眠生理信号的可泛化表征。 |
| [^286] | [Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?](https://arxiv.org/abs/2606.05647) | 本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。 |
| [^287] | [TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning](https://arxiv.org/abs/2606.03629) | 本文构建了TSQBench基准，揭示当前大语言模型在时间序列质量维度识别与有据可依的质量比较上的不足，并据此提出专用智能体推理框架TSQAgent，以实现更可靠的时间序列数据质量评估。 |
| [^288] | [The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics](https://arxiv.org/abs/2606.03357) | 该论文提出一种诊断框架，通过将人设与四个提示组件的语义等价变体交叉组合并测量响应分布间的Wasserstein距离，成功将心理测量中的目标效应与提示伪影分离，并发现任务指令和选项符号对响应分布的干扰比人设或题项措辞的改变更大。 |
| [^289] | [Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking](https://arxiv.org/abs/2606.01483) | 提出了基于分层分块的长篇ASR推理系统Murmur，发现约五分钟的中间块大小可在避免重复循环的同时保持单次处理的准确率，并通过并行解码兼顾高准确率与低推理延迟。 |
| [^290] | [Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents](https://arxiv.org/abs/2605.30621) | 该研究将自进化大语言模型智能体的能力解耦为“框架更新”与“框架收益”两种，并发现模型产生有用框架更新的能力与其基础能力层级无关，不同能力的模型带来的更新收益惊人地相似，即擅长更新框架并不意味着能真正从中受益。 |
| [^291] | [Tiny but Trusted: Efficient Vision-Language Reasoning for Time-Series Anomaly Detection](https://arxiv.org/abs/2605.30344) | 提出高质量异常解释增强的基准VisAnomBench和参数高效的视觉-语言模型VisAnomReasoner，实现了更准确、可解释的时间序列异常检测。 |
| [^292] | [Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders](https://arxiv.org/abs/2605.27354) | 该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。 |
| [^293] | [Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution](https://arxiv.org/abs/2605.26032) | 提出尺度不变的扩散模型 SKILD，将图像生成与连续超分辨率统一在单一无条件框架中，仅通过改变起始时间步即可在两种任务间切换。 |
| [^294] | [JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment](https://arxiv.org/abs/2605.25240) | 论文发布JudgmentBench基准，汇集资深律师对真实法律任务给出的量规评分与成对偏好判断数据，首次实证比较发现比较判断在恢复输出质量排序方面显著优于基于量规的评分。 |
| [^295] | [LAPLEX: The FFT of Learnable Laplace Kernels](https://arxiv.org/abs/2605.24584) | LAPLEX提出了一类精确且可训练的拉普拉斯核算子层，以FFT式的扩展效率实现了可学习的自适应几何结构，支持高达10^9维度的可训练矩阵运算，并可高效用于高维协方差建模。 |
| [^296] | [ActQuant: Sub-4-bit Action-Guided Backbone Weight-Only Quantization for Vision-Language-Action Models](https://arxiv.org/abs/2605.24011) | ActQuant提出了一种动作引导的混合精度训练后量化框架，通过张量间比特分配和张量内尺度优化两阶段策略实现VLA模型的亚4比特仅权重量化，并配套OmniModel.cpp转换流水线支持设备端部署。 |
| [^297] | [VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection](https://arxiv.org/abs/2605.23504) | VACE提出了一种自监督异常检测方法，通过速度对齐通道嵌入显式塑造潜空间几何结构，将正常性表示为局部平滑、方向连贯的表示，克服了对比学习无法显式控制距离评分所依赖几何结构的缺陷。 |
| [^298] | [Vector Policy Optimization: Training for Diversity Improves Test-Time Search](https://arxiv.org/abs/2605.22817) | 提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。 |
| [^299] | [TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks](https://arxiv.org/abs/2605.22535) | 本文提出TerminalWorld——一个能从真实终端记录中自动逆向生成高保真评估任务的可扩展数据引擎，构建出包含1,530个验证任务的大规模终端智能体基准，并揭示当前前沿模型在真实终端工作流上的最高通过率仅为62.5%，且其表现与现有专家策划基准仅呈弱相关。 |
| [^300] | [KVBuffer: IO-aware Serving for Linear Attention](https://arxiv.org/abs/2605.19049) | KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。 |
| [^301] | [An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration](https://arxiv.org/abs/2605.18648) | 本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。 |
| [^302] | [Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies](https://arxiv.org/abs/2605.17204) | 该论文提出一种以闭环行为事件为锚点的SAE可解释性流水线，通过从末端执行器轨迹提取关键帧并聚类来分析VLA策略的内部表征，是首批将基于SAE的VLA分析建立在机器人闭环行为事件之上的工作之一。 |
| [^303] | [LinAlg-Bench: A Forensic Benchmark Revealing Structural Failure Modes in LLM Mathematical Reasoning](https://arxiv.org/abs/2605.16675) | 该论文提出LinAlg-Bench取证式诊断基准，通过对10个前沿大语言模型在660个线性代数问题上的6,600个输出进行系统性错误分类，发现模型数学失败受算法类型和矩阵维度结构性约束，且在4x4规模存在急剧行为阈值——低于该规模表现为执行错误，高于该规模则转变为计算放弃与编造响应。 |
| [^304] | [PanoWorld: Geometry-Consistent Panoramic Video World Modeling](https://arxiv.org/abs/2605.15391) | PanoWorld 将全景视频生成重构为几何与动力学一致的潜在状态建模问题，通过深度一致性和轨迹一致性两个轻量级正则化项，从单张图像生成几何一致的 360° 全景视频。 |
| [^305] | [Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty](https://arxiv.org/abs/2605.14494) | 本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。 |
| [^306] | [EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents](https://arxiv.org/abs/2605.12887) | 本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。 |
| [^307] | [Persona-Model Collapse in Emergent Misalignment](https://arxiv.org/abs/2605.12850) | 该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。 |
| [^308] | [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938) | 提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。 |
| [^309] | [Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability](https://arxiv.org/abs/2605.09214) | 本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。 |
| [^310] | [FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation](https://arxiv.org/abs/2605.08734) | 本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。 |
| [^311] | [OmicsLM: A Multimodal Large Language Model for Multi-Sample Omics Reasoning](https://arxiv.org/abs/2605.06728) | 提出了OmicsLM，一个90亿参数的多模态大语言模型，将bulk和单细胞转录组表示为紧凑的连续嵌入并与自然语言指令及其他样本交错输入，配合连接实验测量、生物学知识与任务监督的训练语料库，实现了多样本组学推理。 |
| [^312] | [UniSD: Towards a Unified Self-Distillation Framework for Large Language Models](https://arxiv.org/abs/2605.06597) | 提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。 |
| [^313] | [ProCompNav: Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries](https://arxiv.org/abs/2605.06223) | 提出ProCompNav两阶段框架，利用递归比较判断（RCJ）主动询问区分性信息，逐步缩小候选池，从而在模糊查询场景下准确识别目标实例，避免将强匹配的干扰项误认为目标。 |
| [^314] | [Taming the Entropy Cliff: Variable Codebook Size Quantization for Autoregressive Visual Generation](https://arxiv.org/abs/2605.06207) | 该论文发现离散视觉分词器中存在条件熵沿序列急剧衰减的“熵悬崖”现象，并提出可变码本大小量化（VCQ）方法，以解决恒定码本快速耗尽有限数据熵预算的问题。 |
| [^315] | [Are LLM Safety Judges Policy-Invariant? A Three-Principle Stress-Test](https://arxiv.org/abs/2605.06161) | 本文提出“策略不变性”这一安全评判者必须满足的基本属性，并将其操作化为三条可检验原则构成压力测试协议，揭示了现有LLM安全评判者的判定会同时受有意义规范变化和无意义措辞改写影响这一此前未被测量的失效模式。 |
| [^316] | [Worst-Case Discovery and Runtime Protection for RL-Based Network Controllers](https://arxiv.org/abs/2605.04373) | 提出了ReGuard框架，通过双层遗憾最大化问题发现强化学习网络控制器的最坏情况场景，并将反事实轨迹编译为轻量级逻辑规则，在推理时无需重新训练即可保护控制器。 |
| [^317] | [Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability](https://arxiv.org/abs/2604.24737) | 该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。 |
| [^318] | [AAC: Admissible-by-Architecture Differentiable Landmark Compression for ALT](https://arxiv.org/abs/2604.20744) | 本文提出AAC，在固定标签内存预算下通过两种可采纳性保持的连续松弛实现可微分的ALT地标选择压缩，并给出了带有1-1/e极限因子的近似保证。 |
| [^319] | [Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models](https://arxiv.org/abs/2604.16565) | 该论文提出了一种无需训练、无监督的双向流形一致性（BMC）度量方法，通过前向掩码与后向重建循环量化生成序列的几何稳定性，从而在无标准答案的情况下验证扩散语言模型推理的有效性，并支持诊断与拒绝重采样等应用。 |
| [^320] | [Stability and Generalization in Looped Transformers](https://arxiv.org/abs/2604.15259) | 该论文提出基于不动点的三维度稳定性分析框架，证明带外部归一化召回机制的循环Transformer能同时实现不动点的局部吸引性、输入平滑性与稳定的反向传播，从而在测试时对更难的问题实现外推而非记忆。 |
| [^321] | [MIND: AI Co-Scientist for Material Research](https://arxiv.org/abs/2604.13699) | MIND是一个由LLM驱动的多智能体框架，通过集成机器学习原子间势实现材料研究假设的自动化实验验证，突破了以往AI科学发现仅限于文本推理的局限。 |
| [^322] | [Selectivity and Shape in the Design of Forward-Forward Goodness Functions](https://arxiv.org/abs/2604.13081) | 本文系统研究了前向-前向算法中优良度函数的选择性与形状设计，提出了一种无参数、尺度不变的“突发性优良度”目标函数（基于超额峰度、奖励重尾活动模式），在Fashion-MNIST上显著提升了准确率。 |
| [^323] | [Zero-Shot Quantization via Weight-Space Arithmetic](https://arxiv.org/abs/2604.03420) | 该论文提出通过简单的权重空间算术从供体任务中提取“量化向量”来修补接收模型，从而在不进行量化感知训练、不使用接收端验证数据的零样本场景下，将3比特量化的Top-1准确率提升最多60%。 |
| [^324] | [Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding](https://arxiv.org/abs/2603.25251) | 该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。 |
| [^325] | [VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining](https://arxiv.org/abs/2603.23481) | VTAM提出了一种无需触觉或动作预训练的高效视觉-触觉世界-动作模型，通过共享冻结的视频VAE并联合预测未来视觉与触觉潜在表示，在富接触操作的真实机器人任务上超越了所有基线。 |
| [^326] | [CTFS : Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels](https://arxiv.org/abs/2603.21071) | 提出了一种由一个通用教师和多个声呐专用教师组成的多教师协作框架CTFS，通过多教师交替引导策略，在极有限标注条件下显著提升了前视声呐图像语义分割的性能。 |
| [^327] | [Cue-guided dual-path reconstruction with adaptive multi-scale response refinement for low-light image enhancement](https://arxiv.org/abs/2603.16482) | 提出CueDR-Net，一种通过双向线索交互与自适应多尺度精化，将亮度调整与局部内容重建相结合的双路径网络，有效解决低光照图像增强中的色彩偏差与局部细节模糊问题。 |
| [^328] | [Preference-Guided Adversarial Policy Optimization for Long-Tail Robust Driving](https://arxiv.org/abs/2603.15221) | ADV-0提出一个闭环极小极大优化框架，将驾驶策略与对抗智能体的交互建模为零和马尔可夫博弈，并通过迭代偏好学习高效逼近最优对抗分布，从而提升自动驾驶系统对罕见但安全关键的长尾场景的鲁棒性。 |
| [^329] | [BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation](https://arxiv.org/abs/2603.10828) | 本文提出“主动提示”新范式，将图像内位置视为未标记样本池、提示作为查询，并基于BALD-SAM框架利用贝叶斯分歧准则自动选择下一次提示的最优空间位置，从而最大化SAM交互式分割中每次人机交互的信息效用。 |
| [^330] | [MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis in Cattle Re-Identification](https://arxiv.org/abs/2603.04314) | 提出了包含1,000头牛、128个均匀采样视角的大规模合成数据集MOO，系统量化了俯仰高度对动物重识别的影响并确定了关键高度阈值，同时在零样本和有监督设置下验证了其在真实牛只数据集上的可迁移性与性能增益。 |
| [^331] | [AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation](https://arxiv.org/abs/2603.03761) | AgentSelect是一个将智能体选择重新建模为“叙述性查询到智能体推荐”任务的基准，聚合了超过11万条查询、10.7万个可部署智能体和25万余条交互记录，填补了端到端智能体配置推荐缺乏查询条件监督数据的研究空白。 |
| [^332] | [ASTRA-bench: Evaluating Tool-Use Agent Reasoning and Action Planning with Personal User Context](https://arxiv.org/abs/2603.01357) | ASTRA-bench 是首个将随时间演变的个人用户情境与交互式工具箱和复杂用户意图相统一的工具智能体评测基准，其基于纵向生活事件构建的 2,413 个场景揭示了现有最先进模型在高复杂性条件下性能显著下降、参数生成是主要瓶颈的问题。 |
| [^333] | [How Far Can a Trainable Frequency Move? Resolution, Grids and Coupling in Trainable-FrequencyQuantum Models](https://arxiv.org/abs/2602.23409) | 该论文揭示了可训练频率量子模型中前置因子梯度的解析结构——由 sin u/u 重叠函数斜率的加权和决定每个频率的可达范围，并提出三值前置因子网格方案，仅用对数数量的编码门即可将模型频率逼近所有频带频率至半个步长以内。 |
| [^334] | [RoboCurate: Harnessing Diversity with Action-Verified Synthetic Data for Robot Learning](https://arxiv.org/abs/2602.18742) | 提出RoboCurate框架，通过在模拟器中执行IDM预测的动作并对比模拟器回放与生成视频的运动一致性来验证合成数据的动作标签，同时扩展场景多样性，为机器人学习提供高质量、可扩展的合成数据。 |
| [^335] | [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/abs/2602.13977) | WoVR提出了一种基于世界模型的可靠强化学习框架，用于VLA策略的后训练，通过显式调控强化学习与不完美想象动力学之间的交互，解决世界模型幻觉和长时程误差累积导致的不可靠学习信号问题。 |
| [^336] | [ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs](https://arxiv.org/abs/2602.13274) | 该论文提出了ProMoral-Bench基准，系统比较了四种大语言模型家族上的10种提示范式在道德推理与安全方面的表现，并发现紧凑的示例引导式提示能在更低的token成本下取得更强的能力与鲁棒性权衡。 |
| [^337] | [When LLM Routers Overpay: The Cost of Strong-Model Preference](https://arxiv.org/abs/2602.03478) | 该论文发现现有LLM路由器在预算宽松时会因对昂贵强模型的持久偏好而造成过度支付（“强模型过度选择”），并提出决策对齐的路由器EquiRouter来学习查询级的恰当模型分配，从而避免不必要的成本浪费。 |
| [^338] | [Position: Agentic Evolution is the Path to Evolving LLMs](https://arxiv.org/abs/2602.00359) | 本文提出智能体化进化是LLM应对部署环境变化的新扩展轴，并给出A-Evolve框架，将部署时改进从固定流程提升为自主进化者智能体执行的目标导向优化过程。 |
| [^339] | [AI-Generated Image Detectors Exploit Global Artifacts: Evidence from Inpainting Exchange](https://arxiv.org/abs/2602.00192) | 该论文发现最先进的AI生成图像检测器实际利用的是扩散修复管线中VAE编码-解码带来的全局频谱偏移伪影，而非识别被合成的具体内容，并提出了无需模型的INP-X诊断方法来验证这一现象。 |
| [^340] | [Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves](https://arxiv.org/abs/2601.21582) | 该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。 |
| [^341] | [Self-Improving Training & Thinking](https://arxiv.org/abs/2601.21343) | 该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。 |
| [^342] | [Analysis of LLM Vulnerability to GPU Soft Errors: An Instruction-Level Fault Injection Study](https://arxiv.org/abs/2601.19912) | 本文首次针对大语言模型推理开展指令级故障注入研究，系统分析了LLM对GPU软错误的脆弱性，填补了现代大模型可靠性评估领域的空白。 |
| [^343] | [EmboTeam: Asymmetric State Grounding for Heterogeneous Multi-Robot Collaboration](https://arxiv.org/abs/2601.11063) | 提出 EmboTeam 神经符号框架，通过 LLM 生成结构化状态预期、PDDL 符号搜索验证协调，并落地为带局部重规划与黑板同步的团队级并行行为树，实现异构多机器人长时程协作中的非对称状态接地，显著提升执行鲁棒性。 |
| [^344] | [Pareto DNN Verification: Fast for Most Queries](https://arxiv.org/abs/2512.20755) | 提出PAVE方法，通过为DNN添加早期退出机制并逐退出点验证，能以直接验证所需时间的一小部分解决约80%的鲁棒性查询，且准确率变化不超过0.3%。 |
| [^345] | [FacEDiT: Talking Head Video Editing via Facial Motion Infilling](https://arxiv.org/abs/2512.14056) | 该论文提出FacEDiT，将说话头部视频的局部编辑建模为面部动作填充这一自监督任务，无需成对的直接监督数据，即可利用普通视频-语音对实现保持身份一致性和边界连续性的局部视频编辑。 |
| [^346] | [StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos](https://arxiv.org/abs/2512.01707) | StreamGaze是首个评估多模态大语言模型能否在流式视频中利用人类视线信号进行时序推理与主动式响应的基准，通过视线引导的过去、现在和主动式三类任务全面检验流式视频理解能力。 |
| [^347] | [Adaptive Test-Time Memory Allocation for Long-Tailed Human-Object Interaction Detection](https://arxiv.org/abs/2511.18811) | 提出了一种无需训练、即插即用的自适应多样性缓存（ADC）框架，通过频率感知的缓存容量分配与置信度-多样性联合样本选择，自适应地构建测试时缓存，从而有效提升长尾人-物交互检测中稀有交互的识别性能。 |
| [^348] | [Cognitive Foundations for Reasoning and Their Manifestation in LLMs](https://arxiv.org/abs/2511.16660) | 该研究构建了涵盖28种认知元素的认知科学分类体系与细粒度评估框架，通过对比18个模型与人类的推理轨迹，发现大语言模型虽能调用人类推理中的认知元素但使用方式不同，并据此开发了可自动搭建成功推理拓扑的测试时推理引导方法以提升模型准确率。 |
| [^349] | [Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses](https://arxiv.org/abs/2510.13624) | 本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。 |
| [^350] | [FinMultiHopQA: A KG-Grounded Multi-Hop Financial QA Benchmark for Evidence-Driven Reasoning](https://arxiv.org/abs/2510.02906) | 该论文提出了FinMultiHopQA基准，包含15,600个定性与定量问题，基于从SEC 10-K申报文件构建的、含1750万+三元组的金融知识图谱FinReflectKG，支持跨章节、跨财年、跨公司的多跳证据推理，并能控制证据呈现方式的变化以实现可复现、可审计的评测。 |
| [^351] | [ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks](https://arxiv.org/abs/2510.02677) | 提出ARMs自适应红队测试智能体，通过推理增强的多步骤编排自动优化攻击策略，包含11种新颖多模态攻击手段并可即插即用集成17种红队测试算法，实现对视觉语言模型的全面自动化安全评估。 |
| [^352] | [Hybrid Approach for Enhancing Lesion Segmentation in Fundus Images](https://arxiv.org/abs/2509.25549) | 本文提出一种混合方法，将深度学习模型与传统数学/聚类分割方法相结合，以克服眼底图像数据集分辨率低、标注不一致以及标注数据不足的问题，实现脉络膜痣等眼底病变的精确分割，助力早期诊断工具的开发。 |
| [^353] | [LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding](https://arxiv.org/abs/2508.19204) | 该论文提出了一种直接生成具有精确几何的大规模3D驾驶场景的方法，弥合了神经重建方法场景多样性受限与扩散模型生成缺乏几何锚定之间的差距，实现了支持物体永久性的因果新视角合成和显式3D几何估计。 |
| [^354] | [Post-Training Large Language Models via Reinforcement Learning from Self-Feedback](https://arxiv.org/abs/2507.21931) | 该论文提出了RLSF方法，利用模型自身对答案的内在置信度作为自我生成的奖励来构建合成偏好数据，无需人工标注或标准答案即可通过偏好优化提升大语言模型在推理任务上的可靠性和校准度。 |
| [^355] | [CoMPAS3D: A Dataset and Benchmark for Interactive Motion](https://arxiv.org/abs/2507.19684) | 该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。 |
| [^356] | [SOCIA-$\nabla$: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation](https://arxiv.org/abs/2505.12006) | SOCIA-$\nabla$提出了一种诊断驱动的多智能体框架，通过文本计算图中的局部化文本梯度修复信号，使大语言模型智能体能够迭代地合成、执行、评估、诊断和修复模拟器代码，在多个真实与半合成基准上实现了最先进的模拟器保真度。 |
| [^357] | [How to Backdoor Image Knowledge Distillation](https://arxiv.org/abs/2504.21323) | 本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。 |
| [^358] | [Unraveling Groundwater Stress: A Model Generalizable Across Space and Time Reveals Co-Declining Water Tables and Recharge Across Bangladesh](https://arxiv.org/abs/2503.22771) | 该研究提出一种可跨时空泛化的“教师-学生”机器学习框架，将稀疏实地观测与GLDAS卫星数据结合，生成孟加拉国2公里高分辨率地下水埋深图，并揭示了地下水位与地下水补给量正在同步下降。 |
| [^359] | [Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research](https://arxiv.org/abs/2501.14546) | 本文证明ChatGPT可通过多模态视觉能力对卫星图像按贫困水平进行排序，其准确性与基于专家特征训练的模型相当甚至优于直接专家分类，为大语言模型在社会经济研究中的应用开辟了新途径。 |
| [^360] | [Q-Learning under Finite Model Uncertainty](https://arxiv.org/abs/2407.04259) | 提出了一种在有限模糊转移核集合下的鲁棒Q学习算法，证明了其几乎必然收敛到鲁棒最优解，并给出将可观测经验贝尔曼残差与转移核估计误差分离的时间一致高概率误差界。 |
| [^361] | [Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing](https://arxiv.org/abs/2402.17595) | 本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。 |

# 详细

[^1]: SwarmEvolve：用于进化发现与优化的潜在空间智能体集群

    SwarmEvolve: Latent Agent Swarms for Evolutionary Discovery and Optimization

    [https://arxiv.org/abs/2611.08483](https://arxiv.org/abs/2611.08483)

    SwarmEvolve让每轮数百个低成本智能体在潜在空间中以集群方式推理、并通过KV缓存共享信息，从而大幅加速LLM引导的进化搜索，在21个数学发现与科学计算问题上表现出色。

    

    基于大语言模型（LLM）引导的进化搜索利用AI智能体，通过迭代优化解法并融合高分方法，来改进解决数学与计算问题的程序。然而，这一过程往往很缓慢，因为LLM在每一轮都需要生成长篇的文本思维链，且为保持种群多样性必须多次重复这种生成。我们提出了SwarmEvolve框架：每轮由数百个低成本智能体以集群（swarm）方式协作，在潜在空间中直接推理而无需解码为文本，并通过传递KV缓存来共享信息。每个集群从不同角度研究问题；新颖性过滤器通过直接测量智能体隐藏状态向量之间的距离，保留最独特的智能体想法；想法的交叉在潜在空间中自然发生，因为一个总结者智能体会接收所有集群成员合并后的KV缓存。在21个数学发现与科学计算问题上，SwarmEvolve……

    arXiv:2611.08483v1 Announce Type: new  Abstract: LLM-guided evolutionary search uses AI agents to improve programs for mathematical and computational problems by iteratively refining solutions and combining high-scoring methods. However, the process is often slow because the LLM generates a long textual chain of thought for each round and must repeat this generation many times to maintain population diversity. We present SwarmEvolve, a framework in which hundreds of low-cost agents per round work in swarms, reason in latent space without decoding to text, and share information by passing KV caches. Each swarm studies the problem from a different angle. A novelty filter keeps the most distinct agent ideas by directly measuring the distances between the agents' hidden state vectors. Crossover of ideas happens naturally in latent space, as a summarizer agent receives the combined KV caches of all swarm members. On 21 problems in mathematical discovery and scientific computing, SwarmEvolve
    
[^2]: Scientist World：为AI科学家培育一个科研成长世界

    Scientist World: Grow a Research World for AI-Scientists

    [https://arxiv.org/abs/2611.08478](https://arxiv.org/abs/2611.08478)

    该论文提出Scientist World，将每篇研究论文转化为可执行的研究环境，用于评测和训练AI科学家，并发现前沿AI科学家在74%的情况下已能超越人类研究者的成果。

    

    AI-科学家已经到来，但一个供他们成长的世界却尚未出现。人类科学家把这个世界视为理所当然，科学文献中积淀了数百年的悬而未决的问题与共享的思想。对AI-科学家而言，这个世界是可读的却不可运行的，因此我们为其构建了Scientist World：让每一篇研究论文都成为一个可执行的环境。我们的自主流水线会提取每篇论文的开放式研究问题，并将其提出的方法复现为基线；进入环境的AI-科学家将以人类作者取得的成果为标准接受评估。我们发布了首批1,000个经过验证的环境。以这个世界作为评测手段，我们发现前沿AI-科学家在74%的情况下已能超越人类研究者。作为AI如何做科学的记录，其中包含6,400小时的研究轨迹，涵盖每一个想法、实验、失败与成功。作为训练场，我们能够将一个小模型的性能提升（+158%）至与……（原文在此处截断）

    arXiv:2611.08478v1 Announce Type: new  Abstract: AI-Scientists arrived, but a world for them to grow in did not. Human scientists take theirs for granted $-$ centuries of open problems and shared ideas, waiting in the scientific literature. To AI-Scientists, that world is readable but not runnable, so we build Scientist World for them: every research paper becomes an executable environment. Our autonomous pipeline extracts each paper's open-ended research question and reproduces its proposed method as the baseline; an AI-Scientist entering the environment is evaluated against what the human authors achieved. We release the first 1,000 verified environments. With the world as an evaluation, we found that frontier AI-Scientists can already surpass the human researchers 74% of the time. As a record of how AI does science, it contains 6,400 hours of research traces, with every idea, experiment, failure, and success. As a training ground, we are able to raise (+158%) a small model to match 
    
[^3]: XStack：一个用于机器学习代码仓库迁移的基准测试

    XStack: A Benchmark for ML Repo Migration

    [https://arxiv.org/abs/2611.08472](https://arxiv.org/abs/2611.08472)

    该论文提出 XStack 基准，通过要求智能体将 PyTorch 代码仓库完整迁移为可在 TPU 上运行的 JAX 代码仓库，来评估其跨框架、跨硬件的长时程迁移能力，目前最强模型通过率仅约 75%。

    

    机器学习代码应该运行在研究者选择的地方，而不是它最初被编写的地方。如今，机器学习代码主要存在于两个生态系统中：运行在 NVIDIA GPU 上的 PyTorch 和运行在 Google TPU 上的 JAX。在两者之间迁移代码仓库意味着需要跨框架和硬件进行构建与测试。这种迁移（1）是一种实际需求，因为它能为研究者解锁不同的硬件资源；（2）也是一个天然的基准测试：它涉及不同的框架和硬件、覆盖整个代码仓库，并且是长时程且具有挑战性的任务。我们提出了 XStack，一个评估智能体在这两个生态系统之间迁移机器学习代码仓库能力的基准。每个智能体会收到一个 PyTorch 代码仓库和一台干净的 TPU 机器，并且必须交付一个 JAX 代码仓库。只有当其输出满足我们的评分标准（该标准基于源代码仓库已发布的结果，并经过我们的认证和校准）时，才算通过。目前智能体的迁移能力仍然不可靠：Claude Opus 5 的通过率为 75%，GPT-5.6-sol 为 45%。

    arXiv:2611.08472v1 Announce Type: cross  Abstract: ML code should run where researchers choose, not where it was first written. Today, machine learning code mainly lives in two ecosystems, PyTorch on NVIDIA GPUs and JAX on Google TPUs. Migrating repositories between them means building and testing across frameworks and hardware. This migration is (1) a practical need, since it unlocks different hardware for researchers, and (2) a natural benchmark: it involves different frameworks and hardware, covers whole repositories, and is long-horizon and challenging. We introduce XStack, a benchmark that evaluates agents on migrating ML repositories between these two ecosystems. Each agent receives a PyTorch repository and a clean TPU machine and must deliver a JAX repository. Its output passes if it meets our grading criteria, based on the source repository's published result, authenticated and calibrated by us. Migration by agents remains unreliable: Claude Opus 5 passes 75%, GPT-5.6-sol 45%, 
    
[^4]: 相同行为，不同机制：有限实验能对整合信息认证什么

    Same Behavior, Different Mechanism: What Finite Experiments Can Certify About Integrated Information

    [https://arxiv.org/abs/2611.08469](https://arxiv.org/abs/2611.08469)

    本文在具有已知真实答案的小型随机网络中精确刻画了整合信息 φ 等机制级量何时可被识别、连续且可由有限数据认证，并证明输入输出行为完全相同的网络可携带截然不同的整合信息（0.28 比特 vs 0 比特），但只需 4-53 次内部记录试验即可将其区分。

    

    可解释性、AI 审计以及机器意识评估越来越依赖于机制层面的量：这些量是系统因果转移律的泛函，描述系统如何计算，而非它输出什么。实验只能通过有限的、部分的且有噪声的干预来观察这一规律。我们探究这样的量何时能从实验的访问权限中被识别、对机制连续、并能从有限数据中得到认证，并针对一个刻意苛刻的情形——整合信息理论中的整合信息 $\varphi$——在具有已知真实答案的四单元和五单元随机网络中给出了精确的回答。行为可能是沉默的：一个循环网络与一个前馈网络在其响应延迟处具有完全相同的输入输出规律，分别携带 0.28 和 0 比特，但内部记录在 4-53 次试验内即可将二者区分开来。部分访问可能同样盲目：在我们的例子中，单步记录中被排除的单元会造成精度下限

    arXiv:2611.08469v1 Announce Type: cross  Abstract: Interpretability, AI auditing and assessments of machine consciousness increasingly rest on mechanism-level quantities: functionals of a system's causal transition law that describe how it computes, not what it outputs. Experiments see that law only through finite, partial and noisy interventions. We ask when such a quantity is identifiable from an experiment's access, continuous in the mechanism and certifiable from finite data, and answer exactly for a deliberately demanding case - the integrated information $\varphi$ of integrated information theory - in stochastic networks of four and five units with known ground truth. Behavior can be silent: a recurrent and a feed-forward network with identical input-output laws at their response delay carry 0.28 and 0 bits, yet internal recordings separate them within 4-53 trials. Partial access can be just as blind: in our examples, units left out of a one-step recording impose precision floors
    
[^5]: 通过扩展激活反演来解释残差流的任意方向

    Towards interpreting any residual stream direction by scaling activation inversion

    [https://arxiv.org/abs/2611.08465](https://arxiv.org/abs/2611.08465)

    提出最大激活示例元模型（MAEM），利用无标签强化学习目标将残差流的任意方向反演为强激活该方向的文本示例，在解释效果上超越了语料库搜索和自然语言自编码器。

    

    为了解释语言模型残差流中的某个方向，我们可以寻找能够强烈激活该方向的文本。但当通过扫描语料库来寻找此类文本时，结果会受到语料库内容的限制。我们开发了最大激活示例元模型，这是一类语言模型，能够将主体模型的残差流方向映射为可以强烈激活这些方向的文本示例。我们的优化目标是最大化给定的残差流方向与MAEM所生成文本的激活之间的余弦相似度。这一无标签的强化学习目标有助于在各种残差流方向来源上扩展MAEM的规模。实验表明，所得到的元模型能够产生具有信息量的示例：一个基于Qwen3.6-27B的MAEM在反演保留激活方面优于语料库搜索或自然语言自编码器（NLA），在大多数保留的语言/领域上胜过（基于1000万token的）语料库搜索，并能激活89%的保留……

    arXiv:2611.08465v1 Announce Type: cross  Abstract: To interpret a direction in a language model's residual stream, we can look for text that strongly activates it. But when such text is found by scanning a corpus, it is limited to what that corpus contains. We develop Maximally-Activating Example Metamodels (MAEMs), language models which map residual stream directions from a subject model to text exemplars that strongly activate those directions. Our objective is to maximise the cosine similarity of a supplied residual stream direction with the activations of the text generated by the MAEM. This label-free RL objective helps to scale MAEMs on diverse sources of residual stream directions. Empirically, we find the resultant metamodel produces informative exemplars: a Qwen3.6-27B MAEM inverts held-out activations better than corpus search or a natural-language autoencoder (NLA), beats corpus search (at 10M tokens) on a majority of held-out languages/domains, and activates 89% of held-out
    
[^6]: ReproBridge：将可复现性融入机器学习研究产物

    ReproBridge: Building Reproducibility into ML Research Artifacts

    [https://arxiv.org/abs/2611.08452](https://arxiv.org/abs/2611.08452)

    该论文提出 ReproBridge 智能体工作流，将散见于论文与代码中的实验流程和决策整合为标准化可复现产物，并构建包含 36 篇论文 1,029 个任务的基准，以检验这类精炼产物能否提升智能体独立复现机器学习实验的能力。

    

    复现机器学习实验需要从散布于论文、已发布代码和配套产物中的信息来重构实验流程。因此，对编码智能体而言，复现成功与否可能取决于其填补缺失细节的能力，而不仅仅取决于所发布的产物本身。我们探究：保留已恢复出的流程与决策能否改善后续的独立复现。我们定义了一种标准化的产物设计，包含精炼后的代码库、环境配置、实验启动器和复现指南，并提出 ReproBridge——一种用于构建这类精炼可复现产物的智能体工作流。ReproBridge-Bench 提供来自 36 篇论文、387 个实验的 1,029 个任务，每个任务由五个人工核验的参考分数作为锚定基准。智能体在全新会话中使用原始代码或精炼产物尝试任务；由独立的 Auditor 检查执行证据。在四个被评估的模型上，精炼产物……（原文摘要在此处截断）

    arXiv:2611.08452v1 Announce Type: new  Abstract: Reproducing machine learning experiments requires reconstructing procedures from information scattered across papers, released code, and supporting artifacts. For coding agents, reproduction success can therefore depend on their ability to fill in missing details, rather than on the released artifacts alone. We ask whether preserving recovered procedures and decisions improves subsequent independent reproduction. We define a standardized artifact design comprising a refined codebase, environment setup, experiment launchers, and a reproduction guide, and introduce ReproBridge, an agent workflow for constructing these refined reproducible artifacts. ReproBridge-Bench provides 1,029 tasks from 387 experiments across 36 papers, anchored by five human-verified reference scores per task. Agents attempt tasks in fresh sessions with raw code or refined artifacts; a separate Auditor checks execution evidence. Across four evaluated models, refined
    
[^7]: 类型化决策编码器中的标签措辞增强：对不熟悉的词语有效，对缩写或代号无效

    Label-Wording Augmentation in a Typed-Decision Encoder: Gains on Unfamiliar Words, Not on Abbreviations or Codes

    [https://arxiv.org/abs/2611.08439](https://arxiv.org/abs/2611.08439)

    论文对俄语类型化决策交叉编码器 Ru-Jev 引入标签多形式措辞增强，发现该增强对由普通词和外来词构成的标签词汇表能带来一致的准确性提升（最高达 0.24），但对缩写和代号类标签没有增益。

    

    类型化决策模型会对调用方在请求时提供的选项进行评分，因此其准确性取决于模型开发者无法控制的标签措辞。我们比较了 Ru-Jev 的两个版本——这是一个在十三项俄语任务上训练的 3.07 亿参数多语言交叉编码器；后一版本增加了一种增强方法，将标签以多种表面形式呈现。在保持问题不变、仅改变十八个路由标签措辞的条件下，我们评估了十套标签词汇表，这些词汇表由三种独立流程生成，并检查了与训练数据的重合情况；每题使用五个候选标签，其中八套词汇表还使用了十八个候选标签进行评估。在由普通词汇或外来词（源自英语的行话、办公室俚语、部门名称）构成的保留词汇表上，增强版本在所有设置下都更加准确，提升幅度最高达 0.24。这十四项增益中有十一项在 Holm 校正后仍然显著（十八个候选标签的俚语以及描述性……（摘要在此处截断）

    arXiv:2611.08439v1 Announce Type: cross  Abstract: Typed-decision models score options that the caller supplies at request time, so their accuracy depends on label wording that the model's developer does not control. We compare two releases of a 307M-parameter multilingual cross-encoder, Ru-Jev, trained on thirteen Russian-language tasks; the later release adds an augmentation that presents labels in several surface forms. Holding questions fixed and varying only how eighteen routing labels are worded, we evaluate ten label vocabularies, produced by three separate processes and checked for overlap with the training data, with five candidates per question and, for eight of them, also with eighteen. On held-out vocabularies made of ordinary or borrowed words (English-derived jargon, office slang, department names), the augmented release is more accurate in every setting, by up to 0.24. Eleven of these fourteen gains survive a Holm correction (slang with eighteen candidates and descriptiv
    
[^8]: 从内部理解人工智能：面向AI安全的机制可解释性研究现状

    Understanding AI from the Inside: The State of Mechanistic Interpretability for AI Safety

    [https://arxiv.org/abs/2611.08438](https://arxiv.org/abs/2611.08438)

    本论文系统综述了机制可解释性在AI安全领域的研究现状，探讨通过理解模型内部表示与计算来评估、预测和控制AI行为，能否真正使AI系统更加安全。

    

    神经网络很可能是人类历史上最重要的人造物之一。随着这些系统变得更加强大和自主，它们造成社会危害的潜力也在增长，这使得AI安全成为我们这个时代的决定性挑战之一。然而，我们在尚未完全理解这些系统如何工作、它们在陌生情境中会如何表现、以及如何可靠地控制它们的情况下，仍在继续构建越来越强大的系统。机制可解释性通过研究这些系统的内部表示和计算，为理解这些系统提供了一条路径，其目标是评估、预测和控制模型行为。我们对面向AI安全的机制可解释性进行了综述，以回答这样一个问题：理解模型的工作原理真的能让它们更安全吗？我们将机制可解释性置于更广泛的可解释性研究背景之中，并将其机制性证据与行为评估进行了比较……

    arXiv:2611.08438v1 Announce Type: cross  Abstract: Neural networks are likely to be among the most consequential artefacts in human history. As these systems become more capable and autonomous, their potential to cause societal harm also grows, making AI safety one of the defining challenges of our time. Yet we continue to build increasingly powerful systems without fully understanding how they work, how they will behave in unfamiliar situations, or how to reliably control them. Mechanistic interpretability offers a path towards understanding these systems by studying their internal representations and computations, with the aim of evaluating, predicting, and controlling behaviour. We survey mechanistic interpretability for AI safety in order to answer the question: Does understanding how models work actually make them safer? We place mechanistic interpretability in the broader context of interpretability research and compare its mechanistic evidence with behavioural evaluation. We int
    
[^9]: MedGraph：利用图记忆在长期医疗对话中维护患者状态

    MedGraph: Maintaining Patient State with Graph Memory for Long-Term Medical Dialogue

    [https://arxiv.org/abs/2611.08433](https://arxiv.org/abs/2611.08433)

    MedGraph 提出一种图记忆方法，通过带时间戳的患者专属事实链条、在问题所涉时间点读取的检索机制以及随每个问题提供的安全事实档案，解决了长期医疗对话中状态过时、归属错误和安全信息缺失三类问题，在 MedMemoryBench 上取得最高总体准确率。

    

    跨就诊追踪患者的基于大语言模型（LLM）的医疗助手，需要一种能在患者临床状态变化时保持状态正确性的记忆系统。我们在纵向医疗对话上分析了五种记忆系统的错误，发现三种反复出现的成因：检索到的证据给出过时的状态、包含他人的健康事实，或缺少答案所依赖的安全事实。我们提出了 MedGraph，一种针对这三种情况构建的图记忆。构建阶段从每次会话中提取带时间戳的事实，将每条事实归属到其描述的对象，并将用药、诊断和测量记录链接成保留早期状态的链条。检索阶段将患者的链条作为整体进行排序，在问题所询问的时间点读取每条链条，并附上原始对话文本；患者档案则随每个问题列出所有已记录的安全事实。在 MedMemoryBench 上，MedGraph 在 GPT 和 Qwen 两种骨干模型下均取得了最高的总体准确率。

    arXiv:2611.08433v1 Announce Type: cross  Abstract: LLM-based medical assistants that follow a patient across visits need a memory that keeps the patient's clinical state correct as it changes. We examine the errors of five memory systems on longitudinal medical dialogue and find three recurring causes: the retrieved evidence gives an outdated state, includes another person's health facts, or lacks a safety fact the answer depends on. We present MedGraph, a graph memory built for these three cases. Construction extracts dated facts from each session, attributes each fact to the person it describes, and links medication, diagnosis, and measurement records into chains that keep earlier states. Retrieval ranks the patient's chains as units, reads each chain at the time the question asks about, and adds the original dialogue text; a patient profile lists every recorded safety fact with every question. On MedMemoryBench, MedGraph has the highest overall accuracy with both a GPT and a Qwen ba
    
[^10]: 未来软件中“元访问”的力量——以大语言模型聊天界面与Chat AI为例

    The Power of Meta Access in the Software of the Future --- Demonstrated on LLM Chat UIs and Chat AI

    [https://arxiv.org/abs/2611.08428](https://arxiv.org/abs/2611.08428)

    本文提出“元访问”机制——让LLM聊天助手获得对界面自身的结构化、分级授权视图，从而把聊天客户端变成软件的统一操作入口，并通过Meta Chat原型加以验证，同时指出其带来的隐私、提示注入等风险以及对独立控制平面边界的需要。

    

    聊天仍然是许多大语言模型（LLM）任务的默认交互界面，即便像OpenCode这样的智能体工具在编程和自动化方面表现出色。然而，模型只能看到当前对话，却看不到其周围的工作区。我们探讨了“元访问”——即让助手获得对界面本身的结构化、经授权的视图，并提出了一系列分级阶梯：功能指导；元数据、设置与偏好访问；对摘要和消息的检索、经确认的变更操作以及完全的编排控制。元访问可以把聊天客户端从多个屏幕转变为一个用于查找、导航、组织和操作的统一入口，同时也会带来关于事实依据、隐私、提示注入和信任方面的风险。通用的智能体框架本质上是运行时环境；由于它们可以更改自身的设置，元访问变得具有自反性，因而需要一个独立的控制平面边界。我们以Chat AI客户端的Meta Chat原型为基础对这一思想进行了验证。该原型构建了有界的……（原文摘要在此处截断）

    arXiv:2611.08428v1 Announce Type: new  Abstract: Chat remains the default surface for many large language model (LLM) tasks, even as agentic tools such as OpenCode excel at coding and automation. Yet the model sees the current dialogue but not the workspace around it. We explore meta access---giving the assistant a structured, permissioned view of the interface itself and propose a ladder of levels: feature tutoring; metadata, settings, and preference access; retrieval over summaries and messages, confirmed mutation, and full orchestration. Meta access can turn a chat client from many screens into one entry point for finding, navigating, organising, and operating it, while raising risks around grounding, privacy, prompt injection, and trust. General-purpose agentic harnesses are runtimes; because they can change their own settings, meta access becomes reflexive and needs a separate control-plane boundary. We ground this in the Meta Chat prototype of the Chat AI client. It builds bounde
    
[^11]: 基于校准潜在探测的视觉-语言-动作模型失败预测与分类

    Failure Prediction and Classification in Vision-Language-Action Models Via Calibrated Latent Probing

    [https://arxiv.org/abs/2611.08425](https://arxiv.org/abs/2611.08425)

    提出VLA-Tug，一种基于VLA骨干网络潜在层的轻量级校准探测器，可在极小推理开销下实现早期失败检测，并对漏检率和误报率提供有效界限。

    

    视觉-语言-动作模型（VLA）使通用具身智能成为可能，但其在分布偏移下的性能十分脆弱，未预期的失败可能损坏设备或危及人身安全。因此，在真实世界系统中部署VLA需要一个及时、经过校准且信息丰富的运行时监控器，同时还要足够高效以便在边缘设备上运行。现有方法无法满足这些要求。我们提出了VLA-Tug，这是一种对VLA骨干网络潜在层的轻量级探测器，能够改进早期失败检测，并对漏检率和误报率进行界定，且推理开销极小。我们表明，在轨迹的失败前片段上进行监督，在匹配的假阳性率下，早期失败检测率较最强的学习型基线可提升高达30%。在向未见任务的分布偏移下，漏检率和误报率的界限在高达86%的运行中保持有效。

    arXiv:2611.08425v1 Announce Type: cross  Abstract: Vision-Language-Action models (VLAs) enable general-purpose embodied intelligence, but their performance is brittle under distribution shift, where unanticipated failures can damage equipment or endanger people. Deploying VLAs in real-world systems therefore requires a runtime monitor that is timely, calibrated, and informative while being efficient enough to run on the edge. Existing methods fail to fulfill these requirements. We present VLA-Tug, a lightweight probe over the latent layers of a VLA backbone that improves early failure detection and bounds the missed-detection and false-alarm rates, with minimal inference overhead. We show that supervising on the pre-failure segment of a trajectory increases early failure detection at matched false-positive rates by up to 30% over the strongest learned baseline. Under shift to unseen tasks, the bounds on missed-detection and false-alarm rates remain valid in up to 86% of runs, versus at
    
[^12]: 基准测试即软件：Terminal-Bench与持续基准测试的论证

    Benchmarks as Software: Terminal-Bench and the Case for Continuous Benchmarks

    [https://arxiv.org/abs/2611.08417](https://arxiv.org/abs/2611.08417)

    论文提出“基准维护债务”概念，并构建了结合社区报告、自动检查与人工修复的持续验证工作流，对Terminal-Bench 2.0的审计发现约30%的任务存在缺陷，论证了基准测试应像软件一样进行持续维护。

    

    编程智能体是迄今为止大语言模型最具经济影响力的应用之一，这使得可靠的评估变得日益重要。然而，即使是发布时经过严格验证的基准测试也可能逐渐过时。随着模型能力的提升，它们可能找到被现有验证器拒绝的有效解决方案，而评估基础设施的变更也可能使任务和测试中固化的假设失效。我们将由此产生的缺陷累积称为“基准维护债务”：即基准测试分数与其原本想要衡量的能力之间不断扩大的差距。我们提出了一种针对智能体基准测试的持续验证工作流，该工作流结合了社区报告、任务工件、智能体轨迹、自动化标准答案检查以及由人工把关的修复机制。我们将该工作流应用于Terminal-Bench 2.0——一个在发布前经历了大量人工与语言模型辅助审查及红队测试的基准测试。我们的审计发现该基准测试中约30%的任务存在问题。

    arXiv:2611.08417v1 Announce Type: new  Abstract: Coding agents are among the most economically consequential deployments of LLMs to date, making reliable evaluation increasingly important. Yet even benchmarks that are carefully validated at release can become stale. As models improve, they can discover valid solutions that existing verifiers reject, while changes to evaluation infrastructure can invalidate assumptions embedded in tasks and tests. We call the resulting accumulation of defects benchmark maintenance debt: a growing gap between benchmark scores and the capabilities they are intended to measure. We introduce a continuous-validation workflow for agentic benchmarks that combines community reports, task artifacts, agent traces, automated oracle checks, and human-gated repairs. We apply this workflow to Terminal-Bench 2.0, a benchmark that underwent extensive human and LM-assisted review and red-teaming before release. Our audit identifies issues in roughly 30% of the benchmark
    
[^13]: AI4Math 的淘金热（二）：AI 辅助证明的特征信号是什么？

    The Gold Rush in AI4Math, II: What are signatures of AI-assisted Proofs?

    [https://arxiv.org/abs/2611.08412](https://arxiv.org/abs/2611.08412)

    本文通过构建包含 8,281 篇论文、涵盖不同 AI 参与程度的五个队列，并提出基于即时证明支持图（GIPS）的 ProofTrace 框架来提取可解释特征，从而实现对数学论文中 AI 参与证明的检测。

    

    AI 在数学中的应用日益增多，这引发了一个问题：我们能否将 AI 对证明的参与与 AI 对写作的辅助区分开来，检测出前者？历史上的 arXiv 论文、近期披露实质性 AI 使用的论文，以及 OpenAI 发布的手稿，为研究这一问题提供了丰富的实证基础。然而，仍存在若干挑战：缺乏匹配的病例-对照对；披露信息可能不完整；以及缺乏能够捕捉 AI 参与证明信号的即用型特征。为应对这些挑战，我们通过将现有手稿与我们使用生成式 AI 制作的手稿相结合，构建了包含 8,281 篇论文的五个队列。这些队列涵盖了不同程度的 AI 参与水平，并包括针对相同研究问题的匹配论文对。随后，我们提出了 ProofTrace，这是一个将每篇论文表示为即时证明支持图（GIPS）并从中提取可解释特征的框架……

    arXiv:2611.08412v1 Announce Type: new  Abstract: The growing use of AI in mathematics raises a question: can we detect AI involvement in proofs, separately from AI assistance with writing? Historical arXiv papers, recent papers disclosing substantive AI use, and manuscripts released by OpenAI provide a rich empirical basis for studying this question. However, several challenges remain: lack of matched case-control pairs; potential incompleteness of disclosure; and lack of ready-to-use features designed to capture signals of AI involvement in proofs.   To address these challenges, we construct five cohorts comprising 8,281 papers by combining existing manuscripts with those we produced using generative AI. These cohorts span different levels of AI involvement and include matched pairs of papers addressing the same research problems. We then introduce ProofTrace, a framework that represents each paper as a graph of immediate proof support (GIPS) and extracts interpretable features of mat
    
[^14]: StepBack：面向高效测试时推理的步级错误定位重采样方法

    StepBack: Step-Level Error-Localized Resampling for Efficient Test-Time Reasoning

    [https://arxiv.org/abs/2611.08403](https://arxiv.org/abs/2611.08403)

    StepBack通过轻量级信号定位推理草稿轨迹中首次出错的步骤，仅对该错误步骤及其后续内容进行重采样再投票，在七个数学推理与多跳问答基准上达到与自洽性相当的准确率，同时最多减少41.26%的token消耗。

    

    自洽性（Self-Consistency, SC）是一种广泛用于改进思维链推理的解码策略，但它需要以预设预算采样大量完整的推理轨迹。其高效变体主要通过提前停止完整轨迹采样或在多个问题之间分配采样数量来降低成本。在本文中，我们提出了StepBack，一种用于高效测试时推理的步级错误定位重采样方法。给定一小组草稿轨迹，StepBack利用轻量级信号定位可能首次出现错误的位置，然后对检测到的错误步骤及其后缀进行重采样，最后进行多数答案投票。实验结果表明，StepBack在七个数学推理和多跳问答基准上平均达到了与SC相当的准确率，同时最多可将token使用量降低41.26%。此外，StepBack天然可以与自适应推理方法（如Adaptive Consistency）相结合，进一步提升性能。

    arXiv:2611.08403v1 Announce Type: new  Abstract: Self-consistency (SC) is a widely used decoding strategy for improving chain-of-thought reasoning, but it requires sampling many complete reasoning trajectories with a preset budget. Its efficient variants reduce the cost mainly by stopping full-trajectory sampling early or distributing the number of samples across questions. In this paper, we propose StepBack, a step-level error-localized resampling method for efficient test-time reasoning. Given a small set of draft trajectories, StepBack localizes likely first error positions using lightweight signals and then resamples the detected error step and its suffix, followed by majority answer voting. Empirically, StepBack matches the accuracy of SC on average across seven mathematical reasoning and multi-hop QA benchmarks, while reducing token usage by up to 41.26%. In addition, by nature, StepBack can be combined with adaptive inference methods such as Adaptive Consistency, further improvi
    
[^15]: 学习用于重症监护医疗系统模拟的患者世界模型

    Learning a patient world model for health system simulation in intensive care

    [https://arxiv.org/abs/2611.08392](https://arxiv.org/abs/2611.08392)

    该论文提出从纵向电子健康记录中学习一个患者世界模型，为重症监护医疗系统模拟提供可跨多种场景复用的患者轨迹共享临床表示，避免了为每个新问题重新定义状态和转移模型。

    

    设计医疗策略需要在实施之前预测其临床和运营层面的后果，这一任务通常依赖于医疗系统模拟。这样的模拟需要对患者状态如何演化并产生医疗需求进行建模。然而，传统方法通常针对单一政策问题定制紧凑的状态转移模型，并忽略了依赖于历史的动态特性，因此当出现需要不同临床信息的新问题时，就必须重新定义状态和转移。我们提出利用神经网络从纵向电子健康记录（EHR）中学习一个患者世界模型，为患者轨迹提供一个共享的临床表示，从而可以服务于跨多种场景的模拟问题。以ICU运营为例，我们在MIMIC-IV数据集上训练该模型并将其冻结，随后通过针对特定场景的映射，将其可按时间查询的预测结果与共享的ICU日历、资源和队列相耦合。

    arXiv:2611.08392v1 Announce Type: new  Abstract: Designing healthcare strategies requires anticipating their clinical and operational consequences before implementation, a task that often relies on health system simulation. Such a simulation necessitates modeling how patient states evolve and generate care demands. However, conventional approaches typically tailor compact state transitions to a single policy question and omit history-dependent dynamics, thus requiring the redefinition of states and transitions when there is a new question that entails different clinical information. We propose to learn a patient world model from longitudinal EHRs using a neural network to provide a shared clinical representation of patient trajectories that can serve simulation questions across scenarios. Taking ICU operations as an example, we train the model on MIMIC-IV, freeze it, and couple its time-queryable predictions to shared ICU calendars, resources, and queues via scenario-specific mappings,
    
[^16]: 基于混合机器学习的飞行数据不平衡着陆阶段安全事件预测

    Hybrid Machine Learning for Imbalanced Landing-Phase Safety Event Prediction from Flight Data

    [https://arxiv.org/abs/2611.08386](https://arxiv.org/abs/2611.08386)

    该研究提出一种可解释的混合特征融合框架，将LSTM自编码器的重构误差描述符与统计飞行历史特征融合后输入类别加权XGBoost分类器，以在高度不平衡的多变量飞行数据中实现多类别着陆阶段安全事件预测。

    

    着陆阶段安全事件预测需要在高度不平衡的多变量飞行数据中检测罕见的异常进近。在捕捉耦合时间行为的同时区分不同机场运行环境下的事件特定特征，仍然是一个核心建模挑战。本研究提出了一种可解释的混合特征融合框架，用于多类别着陆阶段安全事件预测。在正常飞行窗口上训练的长短期记忆（LSTM）自编码器用于表征偏离常规进近行为的程度，其连续的重构误差描述符与参数特定的统计飞行历史特征相融合，并输入类别加权的XGBoost分类器，以区分正常、速度偏高、航径偏高和襟翼放下过晚等窗口。该框架使用来自三个到达机场的53,750个NASA DASHlink进近窗口进行评估。混合学习与机场特定学习策略正在进…（原文在此处截断）

    arXiv:2611.08386v1 Announce Type: cross  Abstract: Landing-phase safety-event prediction requires the detection of rare abnormal approaches within strongly imbalanced multivariate flight data. Capturing coupled temporal behavior while distinguishing event-specific signatures across airport operating contexts remains a central modeling challenge. This study proposes an interpretable hybrid feature-fusion framework for multiclass landing-phase safety-event prediction. A long short-term memory (LSTM) autoencoder trained on nominal flight windows characterizes deviations from routine approach behavior. Its continuous reconstruction-error descriptor is fused with parameter-specific statistical flight-history features and supplied to a class-weighted XGBoost classifier to distinguish Normal, Speed High, Path High, and Flaps Late windows. The framework is evaluated using 53,750 NASA DASHlink approach windows from three arrival airports. Pooled and airport-specific learning strategies are comp
    
[^17]: FedReason：将大语言模型用作异构联邦学习的权重空间元聚合器

    FedReason: Large Language Models as Weight-Space Meta-Aggregators for Heterogeneous Federated Learning

    [https://arxiv.org/abs/2611.08383](https://arxiv.org/abs/2611.08383)

    FedReason提出让大语言模型无需任何服务器端训练，通过受限沙箱定期分析客户端模型权重并自动生成自适应聚合程序，从而在标签偏斜、模型异构和拜占庭客户端等异构联邦学习场景中优于14种固定聚合规则基线。

    

    联邦学习服务器使用固定的聚合规则来合并客户端模型，而每种常用规则都只针对某一类异构性设计。我们研究：一个能够查看客户端模型的大语言模型（LLM）能否自行选择聚合规则，且无需在服务器端进行任何训练。在FedReason中，LLM每十轮被调用一次；它通过在受限的服务器端沙箱中运行代码来分析权重，只能看到打印输出的结果，可以在一个包含500张图像的代理集上对少量候选规则进行评分，并提交一段简短的Python聚合程序。服务器应用该程序直至下一次调用，每当程序输出无效时则回退到FedAvg。完整的权重张量从不离开服务器。我们在CIFAR-10和CIFAR-100上使用ResNet-18，在Dirichlet标签偏斜、宽度异构子模型和拜占庭客户端的设定下，共进行了168次运行、对比14个基线方法来评估基于Claude Opus 5.5的FedReason。在12组配对的标签偏斜实验中，它胜过了Fed……（原文摘要在此处截断）

    arXiv:2611.08383v1 Announce Type: cross  Abstract: Federated learning servers combine client models with a fixed aggregation rule, and each rule in common use is designed for one kind of heterogeneity. We study whether a large language model (LLM) that inspects the client models can choose the rule itself, without any training on the server. In FedReason, the LLM is called every ten rounds. It analyses the weights by running code in a restricted server-side sandbox, sees only the printed results, may score a few candidates on a 500-image proxy set, and submits a short Python aggregation program. The server applies the program until the next call and falls back to FedAvg whenever its output is invalid. Full weight tensors never leave the server. We evaluated FedReason with Claude Opus 5.5 on CIFAR-10 and CIFAR-100 with ResNet-18 under Dirichlet label skew, width-heterogeneous sub-models and Byzantine clients, in 168 runs against 14 baselines. Across 12 paired label-skew runs it beat Fed
    
[^18]: MOMASPLAT：面向移动操作的动态新视角示范增强

    MOMASPLAT: Dynamic Novel-View Demonstration Augmentation for Mobile Manipulation

    [https://arxiv.org/abs/2611.08372](https://arxiv.org/abs/2611.08372)

    MOMASPLAT提出了一个面向移动操作的新视角示范增强框架，能够从未见过的机器人底盘位姿合成包含动态物体交互的完整操作轨迹，克服了现有新视角合成方法因假设静态场景而无法增强接触后关键交互阶段的局限。

    

    通过模仿学习训练的视觉运动策略对相机视角的变化高度敏感。这对于移动操作而言尤其成问题，因为由于导航不确定性、定位漂移和环境约束，机器人很少能从与示范数据采集时完全相同的底盘位姿来执行操作任务。近期的方法使用新视角合成（NVS）技术用未见过的视角来增强示范数据，但它们假设场景是静态的，因此无法在接触后的交互阶段重新渲染被操作的物体，导致示范中最关键的部分未能得到增强。我们提出了MOMASPLAT，一个面向移动操作的新视角示范增强框架，能够从未见过的机器人底盘位姿合成完整的操作轨迹，包括动态的物体交互。我们的关键洞察是，标准的移动操作数据采集过程天然地提供了多（摘要在此处被截断）

    arXiv:2611.08372v1 Announce Type: cross  Abstract: Visuomotor policies trained through imitation learning are highly sensitive to changes in camera viewpoint. This is particularly problematic for mobile manipulation (MoMa), where robots rarely execute manipulation from the exact same base pose used during demonstration collection due to navigation uncertainty, localization drift, and environmental constraints. Recent approaches use novel-view synthesis (NVS) to augment demonstrations with unseen viewpoints, but they assume static scenes and therefore fail to re-render the manipulated object during the post-contact interaction phase, leaving the most critical parts of the demonstration unaugmented. We present MOMASPLAT, a novel-view demonstration augmentation framework for mobile manipulation that synthesizes full manipulation trajectories from unseen robot base poses, including dynamic object interactions. Our key insight is that standard MoMa data collection naturally provides the mul
    
[^19]: NPCBench：鼻咽癌多模态诊疗路径推理的临床学徒式基准测试

    NPCBench: A Clinical Apprenticeship Benchmark for Multimodal Care-Pathway Reasoning in Nasopharyngeal Carcinoma

    [https://arxiv.org/abs/2611.08369](https://arxiv.org/abs/2611.08369)

    NPCBench是一个模拟临床专科医师培训的鼻咽癌多模态诊疗路径推理基准，包含4,777个条目和100名患者的CarePath队列，通过回合级评估检验模型能否在完整诊疗路径中联合正确地完成诊断与诊疗任务。

    

    当前的医学基准测试通常将临床能力拆解为孤立的事实性问题、单图像任务或静态病例片段。这些测试对单个答案进行评分，但较少能揭示模型是否能够将亚专科知识与多模态证据相整合，并在整个诊疗过程中追踪患者的病情变化。为弥补这一空白，我们提出了NPCBench——一个针对鼻咽癌（NPC）的临床学徒式基准测试。NPCBench模拟专科医师培训过程，涵盖从指南与基础知识到病例规划、纵向随访审查、CarePath整合以及开放式会诊。它包含4,777个组件与会诊条目，以及一个由100名患者组成的CarePath队列，共417个带评分的诊断和诊疗任务。其回合级评估检验诊断与诊疗答案在诊疗路径内是否联合正确。通过将分步任务与患者级诊疗轨迹相关联，NPCBench评估（原文摘要在此处截断）

    arXiv:2611.08369v1 Announce Type: new  Abstract: Current medical benchmarks typically break down clinical competence into isolated factual questions, single-image tasks, or static case vignettes. These tests score individual answers, but reveal less about whether a model can integrate subspecialty knowledge with multimodal evidence and track a patient's condition throughout a course of care. To close this gap, we introduce NPCBench, a clinical apprenticeship benchmark for nasopharyngeal carcinoma (NPC). NPCBench mirrors specialist training, from guideline and foundational knowledge to case planning, longitudinal review, CarePath integration, and open-ended consultation. It contains 4,777 component and consultation items and a 100-patient CarePath cohort with 417 scored diagnostic and care tasks. The episode-level evaluation tests whether diagnostic and care answers are jointly correct within a care pathway. By linking stepwise tasks to patient-level care trajectories, NPCBench evaluate
    
[^20]: VisionSIM：超越理想化传感器——面向机器人感知的多模态模拟器

    VisionSIM: Beyond Idealized Sensors - A Multi-Modal Simulator for Robot Perception

    [https://arxiv.org/abs/2611.08368](https://arxiv.org/abs/2611.08368)

    提出 VisionSIM，一个基于物理的多模态传感器仿真平台，通过建模原始物理测量过程重现 LiDAR、iToF、事件相机和热成像等传感器的真实失效模式，支持在相同场景下的跨模态系统性比较与机器人感知研究。

    

    机器人感知日益依赖于将相机与新兴模态（事件相机、SPAD、iToF/dToF 和热成像）相结合的异构传感器组合。然而，现有模拟器将这些传感器理想化，掩盖了限制真实世界部署的关键的模态特定失效模式。我们提出 VisionSIM，一个统一的、基于物理的仿真平台，通过为多种传统和新兴模态建模原始物理测量过程，将传感器真实性提升为一等关注点。我们提供对物理传感器参数的细粒度控制，重现了被通用模拟器忽视的失效模式：LiDAR 的堆积效应与眩光；iToF 的相位缠绕、饱和与运动伪影；事件相机的不应期丢失；以及热成像相机依赖历史的发射特性。这使得在相同场景和轨迹上进行有原则的跨模态比较成为可能。我们发布了该平台以及一个大型室内数据（集，摘要在此处被截断）。

    arXiv:2611.08368v1 Announce Type: cross  Abstract: Robotic perception increasingly relies on heterogeneous sensor suites combining cameras with emerging modalities (event, SPAD, iToF/dToF, and thermal). Yet existing simulators idealize these sensors, hiding critical modality-specific failure modes that limit real-world deployment. We present VisionSIM, a unified, physically grounded simulation platform that elevates sensor realism to a first-class concern by modeling the raw physical measurement process for diverse conventional and emerging modalities. We expose fine-grained control over physical sensor parameters, reproducing failure modes overlooked by general-purpose simulators: pileup and glare for LiDAR; phase wrapping, saturation, and motion artifacts for iToF; refractory drops for event cameras; and history-dependent emission for thermal cameras. This enables principled cross-modality comparison on identical scenes and trajectories. We release the platform and a large indoor dat
    
[^21]: 增强AI智能体压缩记忆中的局部与全局结构

    Enhancing Local and Global Structure in Compressed AI Agent Memory

    [https://arxiv.org/abs/2611.08341](https://arxiv.org/abs/2611.08341)

    本文提出用局部保真度 $L$ 和全局保真度 $G$ 两种无标签指标评估AI智能体压缩记忆的质量，证明全局保真度可由近远排序违反率界定，并提出局部感知量化（LAQ）方法，将理论界限转化为合页目标函数，从而在压缩中同时增强记忆的局部与全局结构。

    

    AI智能体将文本以嵌入向量的形式存储并进行大规模压缩。当每个条目的最近邻得以保留时，该压缩即被视为可接受，但智能体还需要整合、删除和比较记忆，而这些任务依赖于哪些条目彼此相距较远。我们采用先前工作中的两种无需标签的保真度指标来评估这两个层面：在10个近邻处的局部保真度 $L$，以及在记忆规模30%至50%范围内的全局保真度 $G$。在每段文本96字节的设置下，重打分的二进制编码达到了0.97的 $L$，却使10.1%的文本段转移到了另一个主题；而使用8位编码的PCA，$L$ 为0.77，仅使1.7%的文本段主题发生转移。重打分不改变存储的编码，因此 $G$ 保持不变。我们证明，在整个 $G$ 区间内，$1-\nu_k$ 至多为近远排序违反率平方根的2至2.18倍，且该违反率遵循压缩产生的距离噪声（相关系数为0.979）。我们的局部感知量化方法（LAQ）将该理论界限转化为一个合页目标函数。

    arXiv:2611.08341v1 Announce Type: new  Abstract: AI agents store text as embedding vectors and compress them at scale. A compression is accepted when each item's nearest neighbors survive, but agents also consolidate, delete, and compare memories, tasks that depend on which items lie far apart. We evaluate both ends with two label-free fidelities from our prior work: the local fidelity $L$ at 10 neighbors and the global fidelity $G$ at 30 to 50 percent of the memory. At 96 bytes per passage, rescored binary codes reach an $L$ of 0.97 yet move 10.1 percent of passages to another theme; PCA with 8-bit codes, at an $L$ of 0.77, moves 1.7 percent. Rescoring leaves the stored codes, and thus $G$, unchanged. We prove that across the $G$ band, $1-\nu_k$ is at most 2 to 2.18 times the square root of the near-far rank-violation rate, and that this rate follows the distance noise of the compression (correlation 0.979). Our locality-aware quantization (LAQ) turns the bound into a hinge objective 
    
[^22]: 学习任务还是学习验证器？重新思考具身自我改进中的标准暴露

    Learning the Task or Learning the Verifier? Rethinking Criteria Exposure in Embodied Self-Improvement

    [https://arxiv.org/abs/2611.08318](https://arxiv.org/abs/2611.08318)

    论文提出“标准暴露”（CE）这一被忽视的实践问题，并通过对照实验揭示：当自我改进的编码智能体接触基准的验证标准时，其生成的代码倾向于过拟合验证标准，而非真正完成被指示的任务。

    

    自我改进的具身智能体越来越多地在机器人仿真基准上进行开发和比较。最近的一些工作将基准的成功标准暴露给学习智能体，我们将这种做法称为“标准暴露”（Criteria Exposure, CE）。这些标准最初是为了验证通过模仿学习训练的策略而编写的，往往要求比相应的语言指令更多，因此来自这类验证器的分数可能无法忠实地反映被指示的任务是否真正完成，尤其是对于由智能体编写的机器人控制程序。随着具身编码智能体报告的基准分数越来越高，一个自然的问题浮现出来：当编码智能体接触到基准的标准时，它生成的是能更好地执行被指示任务的代码，还是仅仅过拟合于验证标准的代码？我们通过带有和不带有CE的自我改进智能体进行了对照实验，发现使用CE时，智能体很快产生

    arXiv:2611.08318v1 Announce Type: new  Abstract: Self-improving embodied agents are increasingly being developed and compared on robot simulation benchmarks. Several recent works expose the benchmark's success criteria to the learning agent, a practice we call Criteria Exposure (CE). These criteria, originally written to verify policies trained by imitation learning, often demand more than the corresponding language instructions, so a score from such a verifier may not faithfully reflect whether the instructed task was completed, especially for robot-control programs written by an agent. As embodied coding agents report ever higher benchmark scores, a natural question arises: when a coding agent is exposed to the benchmark's criteria, does it generate code that better performs the instructed task, or code that simply overfits to the verification criteria? We conduct controlled experiments with a self-improving agent with and without CE, and find that with CE the agent quickly produces 
    
[^23]: 在异构认知与生物医学知识图谱上对大型语言模型进行基准测试：图谱接地在多跳推理与幻觉缓解中的作用

    Benchmarking Large Language Models on Heterogeneous Cognitive and Biomedical Knowledge Graphs: The Role of Graph Grounding in Multi-Hop Reasoning and Hallucination Mitigation

    [https://arxiv.org/abs/2611.08315](https://arxiv.org/abs/2611.08315)

    该研究在五个认知与生物医学知识图谱上对五个大型语言模型进行基准测试，发现通用模型在多跳推理任务上优于领域专用模型，且知识图谱接地能有效降低所有模型的幻觉率。

    

    大型语言模型（LLM）在生物医学和认知神经科学领域的应用日益广泛，但其在异构知识图谱（KG）上进行可靠推理的能力仍不明确。我们对五个大型语言模型（GPT-5.1、Gemini 2.0 Pro、Llama 3、Qwen 和 BioGPT）在五个知识图谱上进行了基准测试：DrugBank、NeuroSynth、Cognitive Atlas、Brain Map 和 GDPKG。我们引入了一个多层次查询数据集，用于在知识图谱感知（图谱接地）和知识图谱无关两种条件下评估推理深度和事实准确性。标注者间一致性较高，Cohen's kappa 为 0.781（范围：0.753-0.807），Fleiss' kappa 为 0.77。结果表明，通用模型（尤其是 GPT-5.1 和 Gemini 2.0 Pro）在多跳和高级推理任务上通常优于领域专用模型，但随着查询复杂度的增加，性能会下降。图谱接地降低了所有被评估模型的幻觉率。与我们的初始假设相反……（摘要在此处被截断）

    arXiv:2611.08315v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in biomedical and cognitive neuroscience, but their ability to reason reliably over heterogeneous knowledge graphs (KGs) remains unclear. We benchmark five LLMs (GPT-5.1, Gemini 2.0 Pro, Llama 3, Qwen, and BioGPT) across five KGs: DrugBank, NeuroSynth, Cognitive Atlas, Brain Map, and GDPKG. We introduce a multitier query dataset to evaluate reasoning depth and factual accuracy under KG-aware (graph-grounded) and KG-agnostic conditions. Inter-annotator agreement was substantial, with Cohen's kappa of 0.781 (range: 0.753-0.807) and Fleiss' kappa of 0.77. Results show that general-purpose models, particularly GPT-5.1 and Gemini 2.0 Pro, generally outperform domain-specific models on multi-hop and high-level reasoning tasks, although performance declines as query complexity increases. KG grounding reduces hallucination rates across all evaluated models. Contrary to our initial hypothesis, do
    
[^24]: RM-VLM：从自然语言任务描述到视觉接地的奖励机

    RM-VLM: From Natural-Language Task Descriptions to Vision-Grounded Reward Machines

    [https://arxiv.org/abs/2611.08308](https://arxiv.org/abs/2611.08308)

    本文提出 RM-VLM，利用大语言模型从自然语言任务描述自动生成奖励机，并让视觉-语言模型充当标注函数将命题接地于图像观测，从而免除传统奖励机中需要针对每个环境手动设计标注函数的局限。

    

    奖励机（Reward Machines, RMs）被提出作为强化学习（RL）中的一种奖励形式化方法，用于捕捉信用分配依赖于历史的非马尔可夫任务。通过将奖励函数表示为自动机，RM 揭示了任务的时间结构，并使得此类奖励可以在马尔可夫框架内处理。然而，RM 假设存在一个标注函数，即一个针对观测的命题接地预言机，该函数必须针对每个环境手动设计，这限制了其更广泛的适用性。在这项工作中，我们通过提出 RM-VLM 来解决这一局限，这是一种新方法：由大语言模型（LLM）根据自然语言任务描述生成 RM，并由视觉-语言模型（VLM）充当标注函数，在 RM 引导的策略学习过程中将 RM 的命题接地于基于图像的观测之上。同时，基于 VLM 的价值函数估计器被用于评估智能体向每个 RM 转移的进展（摘要原文在此处截断）。

    arXiv:2611.08308v1 Announce Type: new  Abstract: Reward machines (RMs) have been proposed as a reward formulation in reinforcement learning (RL) to capture non-Markovian tasks where credit assignment depends on history. By representing reward functions as automata, RMs expose the task's temporal structure and allow such rewards to be handled within a Markovian framework. However, they assume a labelling function, a proposition-grounding oracle for observations, which must be manually engineered for each environment, limiting their wider applicability. In this work, we address this limitation by proposing RM-VLM, a novel approach in which a large language model (LLM) generates an RM from a natural-language task description, and a vision-language model (VLM) acts as the labelling function, grounding the RM's propositions over image-based observations during RM-guided policy learning. A VLM-based value function estimator is used to assess the agent's progress towards each RM transition an
    
[^25]: 面向扩散语言模型并行解码的测量耦合方法

    Measured Coupling for Parallel Decoding in Diffusion Language Models

    [https://arxiv.org/abs/2611.08307](https://arxiv.org/abs/2611.08307)

    该论文提出利用主前向传播中的廉价探针直接测量token间的耦合强度（即提交一个token对其他位置置信度的影响），以替代平均场解码中的相似性代理指标，使扩散语言模型并行解码在平均仅损失1.1个点精度的代价下大幅减少解码步数并提升速度。

    

    扩散语言模型可以在每一步生成多个token，但必须选择哪些token可以安全地一起提交。平均场解码会对预测结果相似的位置对进行惩罚。我们用耦合本身来替代这一代理指标：即提交一个token会在多大程度上改变模型对另一个位置的置信度。主前向传播内部的廉价探针即可测量该耦合，无需训练或额外模型，并且可直接接入相同的平均场规则。在三个模型和四个基准测试上，我们的方法所需步数远少于平均场解码，且在大多数设置下（包括使用键值缓存时）速度更快，平均精度损失仅为1.1个百分点。若打乱耦合或将其替换为统一的奖励，这一增益将消失。

    arXiv:2611.08307v1 Announce Type: cross  Abstract: Diffusion language models can write many tokens per step, but must choose which tokens are safe to commit together. Mean-field decoding penalizes pairs of positions whose predictions look alike. We replace this proxy with the coupling itself: how much committing one token changes the model's confidence at another. Cheap probes inside the main forward pass measure it, with no training or extra model, and it plugs into the same mean-field rule. On three models and four benchmarks, our method needs far fewer steps than mean-field decoding and is faster in most settings, also with a key-value cache, at an average accuracy cost of 1.1 points. Shuffling the couplings or replacing them with a uniform bonus loses this gain.
    
[^26]: Mamba动力学中的首Token吸引效应

    First-Token Attraction in Mamba Dynamics

    [https://arxiv.org/abs/2611.08303](https://arxiv.org/abs/2611.08303)

    该论文提出基于高维不变盒子与精确退出时间论证的新分析框架，首次刻画了深度Mamba模型中token轨迹的极限点并证明指数收敛速率，揭示首个token的极限方向会吸引后续token这一特权动力学特性。

    

    我们研究深度Mamba模型中的token动力学，特别关注第一个token所起的吸引作用。与注意力机制不同——注意力机制中token之间的交互由正标量系数控制——Mamba动力学涉及完整的矩阵值交互系数，且不保证其正性或定性。这种结构上的差异使得Mamba动力学的分析难度大大增加，也使得针对基于注意力模型的现有技术无法直接套用。为克服这一挑战，我们引入了一个新的分析框架，该框架基于高维不变盒子和一个精确的退出时间论证，二者结合共同恢复了一种有效的符号定性形式。利用该框架，我们刻画了token轨迹所有可能的极限点，并证明了显式的指数收敛速率。我们的分析揭示了第一个token具有特权动力学特性：其极限方向会吸引后续的token。

    arXiv:2611.08303v1 Announce Type: new  Abstract: We study token dynamics in deep Mamba models, with a particular focus on the attracting role of the first token. Unlike attention mechanisms, where token interactions are governed by positive scalar coefficients, Mamba dynamics involve full matrix-valued interaction coefficients with no positivity or definiteness guarantees. This structural difference makes Mamba dynamics substantially harder to analyze and prevents the direct application of existing techniques for attention-based models. To overcome this challenge, we introduce a new analytical framework based on a high-dimensional invariant box and a sharp exit-time argument, which together recover an effective form of sign-definiteness. Using this framework, we characterize the possible limit points of token trajectories and prove explicit exponential convergence rates. Our analysis reveals that the first token has privileged dynamics: its limiting direction attracts subsequent tokens
    
[^27]: AutoCompress：自动化硬件引导的模型压缩

    AutoCompress: Automated Hardware-Guided Model Compression

    [https://arxiv.org/abs/2611.08290](https://arxiv.org/abs/2611.08290)

    AutoCompress是一个硬件引导的模型压缩框架，它将延迟、吞吐量和内存等硬件与工作负载约束直接纳入可微架构搜索，联合优化全局与逐层结构以生成异构压缩架构，并利用经过实测验证的服务引擎成本模型引导整个优化过程。

    

    大语言模型（LLM）剪枝必须在满足硬件和工作负载特定的延迟、吞吐量和内存约束的同时保持模型质量。现有的剪枝流程通常以参数数量为目标，要求从业者手动选择满足延迟、吞吐量和GPU内存限制的模型规模。然而，仅凭参数数量并不能预测这些指标，因为这些指标取决于模型架构、工作负载、硬件和服务引擎。我们提出了AutoCompress，这是一个硬件引导的压缩框架，它将此类约束直接纳入可微架构搜索中。AutoCompress联合优化全局和逐层结构以生成异构架构，其优化过程由经过端到端延迟、吞吐量和内存实测验证的服务引擎成本模型引导。我们在长上下文本地编码、批量数据中心服务和延迟敏感的游戏内交互场景中评估了AutoCompress。与标准（基线相比，……原文在此处截断）

    arXiv:2611.08290v1 Announce Type: cross  Abstract: LLM pruning must preserve quality while meeting hardware- and workload-specific constraints on latency, throughput, and memory. Existing pruning workflows often target parameter counts, requiring practitioners to manually select model sizes that meet latency, throughput, and GPU memory limits. However, parameter count alone does not predict these metrics, which depend on the model architecture, workload, hardware, and serving engine. We introduce AutoCompress, a hardware-guided compression framework that directly incorporates such constraints into differentiable architecture search. AutoCompress jointly optimizes global and per-layer structure to produce heterogeneous architectures, guided by serving-engine cost models validated against end-to-end latency, throughput, and memory measurements. We evaluate AutoCompress on long-context local coding, batched datacenter serving, and latency-sensitive in-game interaction. Compared to the sta
    
[^28]: 情境完整性的生成模型：恰当与不恰当的信息共享

    A Generative Model of Contextual Integrity: Appropriate vs. Inappropriate Information Sharing

    [https://arxiv.org/abs/2611.08284](https://arxiv.org/abs/2611.08284)

    该论文构建了一个基于大语言模型模拟对话的情境完整性生成模型，用以研究信息共享行为，发现AI行为者既会在需要时向合法接收者分享信息，也会不恰当地向无权者披露私人信息。

    

    情境完整性（Contextual Integrity, CI）认为，恰当的信息流动取决于特定情境下的规范。同一披露行为在一种情境下可能是恰当的，而在另一种情境下则可能是不恰当的。我们构建了一个情境完整性的生成模型，通过模拟对话来研究信息共享行为。由大语言模型扮演的两个角色交换消息，其中一个持有私人信息，另一个试图获取该信息。我们考察了信息共享与拒绝行为如何随情境、信息持有者的决策逻辑、性格倾向以及底层语言模型的变化而变化。实验发现，行为者会在需要时将信息分享给有权接收者，但也会将其披露给无权接收的人。该模型提供了一种研究方法，用于探究随着情境和角色的变化，披露与拒绝行为如何在对话过程中逐步发展。

    arXiv:2611.08284v1 Announce Type: new  Abstract: Contextual integrity (CI) holds that appropriate information flow depends on context-specific norms. The same disclosure may be appropriate in one context and inappropriate in another. We build a generative model of contextual integrity to study information sharing through simulated conversations. Two actors played by large language models exchange messages, with one holding private information and the other seeking it. We examine how sharing and refusal vary with the situation, the holder's decision logic, disposition, and the underlying language model. Actors shared information with the rightful recipients when it was needed, but also disclosed it to people who were not entitled to receive it. The model provides a way to study how disclosure and refusal develop through conversation as the situations and the actors vary.
    
[^29]: LeanContext：在Lean整段证明生成中将数学资源与局部形式化上下文解耦

    LeanContext: Disentangling Mathematical Resources from Local Formal Context in Lean Whole-Proof Generation

    [https://arxiv.org/abs/2611.08277](https://arxiv.org/abs/2611.08277)

    本文提出包含129,826个Lean整段证明生成问题的基准LeanContext，通过将数学上下文与环境上下文解耦，发现当训练数据包含数学上下文时，模型才能更大程度地利用目标所选声明相对于可比替代方案的优势。

    

    神经定理证明越来越多地使用检索到的前提和项目上下文来辅助证明生成。评估这些资源需要衡量目标所选声明相对于可比替代方案的优势，以及这种优势如何依赖于训练。我们引入了LeanContext——一个包含129,826个Lean整段证明生成问题的基准——来研究这些问题。我们将所选的完整前置数学声明归为数学上下文，将剩余所选的局部源码命令归为环境上下文。四种对齐的输入视图支持在共享目标和原始Lean验证环境下的交叉训练与推理评估。对于基于扩散和基于自回归的模型，当训练包含数学上下文时，所选声明相对于可比替代方案表现出更大的优势。最终输入审计能够识别出目标所引用但被遗漏的声明。优先考虑……（原文在此处截断）

    arXiv:2611.08277v1 Announce Type: new  Abstract: Neural theorem proving increasingly uses retrieved premises and project context to support proof generation. Evaluating these resources requires measuring the advantage of target-selected declarations over comparable alternatives and how this advantage depends on training. We introduce LeanContext, a benchmark of 129,826 Lean whole-proof generation problems, to investigate these questions. We group selected complete preceding mathematical declarations as Mathematical Context (MC) and the remaining selected local source commands as Environment Context (EC). Four aligned input views support crossed training and inference evaluations with shared targets and original Lean verification environments. For both diffusion- and autoregressive-based models, selected declarations outperform comparable alternatives by a larger margin when training includes MC. Final-input audits identify omitted declarations referenced by the target. Prioritizing the
    
[^30]: 将策略编码器与形状教师对齐

    Aligning Policy Encoders with a Shape Teacher

    [https://arxiv.org/abs/2611.08275](https://arxiv.org/abs/2611.08275)

    提出形状教师对齐方法，在训练时以预训练3D形状表示（Point-MAE特征）作为对齐目标，使基于图像的策略编码器习得3D几何信息，同时保持部署时纯图像输入的接口不变。

    

    具备3D感知的表示可以通过提供对视角变化、物体位置和遮挡具有鲁棒性的几何信息来改善机器人操作。然而，在部署时使用3D输入需要额外的传感设备和处理，这使得基于图像的策略更为实用。我们提出一个问题：基于图像的策略能否在训练期间获取3D形状信息，同时在部署时保持纯图像接口？我们提出形状教师对齐，这是一种训练时机制，它使用预训练的3D形状表示作为基于图像的策略编码器的对齐目标。具体而言，我们使用来自单个物体的Point-MAE特征作为对应图像patch的目标，在不改变策略部署输入或网络架构的情况下将几何结构迁移给策略。在LIBERO基准上，形状教师对齐产生的图像表示在物体形状及其对应的部分周围具有更强的结构化特性。

    arXiv:2611.08275v1 Announce Type: cross  Abstract: 3D-aware representations can improve robotic manipulation by providing geometric information that is robust to changes in viewpoint, object position, and occlusion. However, using 3D inputs at deployment requires additional sensing and processing, making image-based policies more practical. We ask whether an image-based policy can acquire 3D shape information during training while retaining a purely image-based interface at deployment. We propose shape-teacher alignment, a training-time mechanism that uses a pretrained 3D shape representation as an alignment target for an image-based policy encoder. Specifically, we use Point-MAE features from individual objects as targets for corresponding image patches, transferring geometric structure to the policy without changing its deployment inputs or architecture. On LIBERO, shape-teacher alignment produces image representations that are more structured around object shape and corresponding pa
    
[^31]: 基于大规模并行加速的安全强化学习

    Accelerating Safe Reinforcement Learning with Massive Parallelism

    [https://arxiv.org/abs/2611.08273](https://arxiv.org/abs/2611.08273)

    该论文提出相位聚合成本估计器与约束收紧框架，解决了大规模并行短回合训练与全回合安全约束之间的不匹配问题，实现了可扩展的安全强化学习加速训练。

    

    我们研究了将安全强化学习扩展到大规模并行训练场景的问题，在这种场景中，大量环境以较短的滚动回合长度被执行。尽管这种设置提高了数据吞吐量和实际运行效率，但它与约束马尔可夫决策过程（CMDP）存在不匹配，因为后者的安全约束是定义在整个回合的成本回报之上的。我们通过将交错的环境重置分析为短轨迹段上的相位混合分布来解决这一不匹配问题。该分析刻画了由相位覆盖所引起的偏差，并启发了一种相位聚合成本估计器，可以从短滚动批次中重建回合级别的成本估计。我们进一步将有限样本不确定性和分布不匹配纳入约束收紧框架，从而得到一种与同策略优化兼容的可扩展安全强化学习方法。基于 MJX 的实现……

    arXiv:2611.08273v1 Announce Type: cross  Abstract: We investigate the problem of scaling safe reinforcement learning (safe RL) to massively parallel training regimes, where a large number of environments are exe- cuted with short rollout horizons. Although this setting improves data throughput and wall-clock efficiency, it creates a mismatch with constrained Markov decision processes (CMDPs), whose safety constraints are defined over full-episode cost returns. We address this mismatch by analyzing staggered environment resets as a phase-mixture distribution over short trajectory segments. This analysis charac- terizes the bias induced by phase coverage and motivates a phase-aggregated cost estimator that reconstructs episode-level cost estimates from short rollout batches. We further incorporate finite-sample uncertainty and distribution mismatch into a constraint-tightening framework, yielding a scalable Safe RL method compatible with on-policy optimization. Using an MJX-based impleme
    
[^32]: DART：面向通用异常检测的领域无关残差迁移

    DART: Domain-Agnostic Residual Transfer for Generalist Anomaly Detection

    [https://arxiv.org/abs/2611.08271](https://arxiv.org/abs/2611.08271)

    DART提出了一种即插即用框架，通过残差解耦模块与跨领域元训练，将正常-异常参考残差分离为异常相关成分与领域特定成分，从而提升通用异常检测的跨域迁移能力。

    

    通用异常检测（GAD）旨在仅使用少量参考样本来检测未见领域中的异常。近期的正常-异常引导方法通过利用异常参考样本改进了这一设置，但其残差往往将可迁移的异常线索与领域特定的外观因素（如纹理、几何形状和采集条件）混杂在一起。这种残差的领域纠缠限制了跨领域迁移，并可能在新的领域上产生虚假激活。我们提出DART，这是一个面向正常-异常引导GAD的即插即用框架，它学习将残差分离为异常相关成分和领域特定成分。DART将残差解耦模块（在保留异常判别信息的同时抑制领域信息）与跨领域元训练（在源域之间构建训练片段以使这种分离有效）相结合。该方法与架构无关，可以插入到残差特征提取之间……

    arXiv:2611.08271v1 Announce Type: cross  Abstract: Generalist Anomaly Detection (GAD) aims to detect anomalies in unseen domains using only a few reference samples. Recent normal-abnormal guided methods improve this setting by exploiting abnormal references, but their residuals often mix transferable anomaly cues with domain-specific appearance factors such as texture, geometry, and acquisition conditions. This residual domain entanglement limits cross-domain transfer and may produce spurious activations on novel domains. We propose DART, a plug-in framework for normal-abnormal guided GAD that learns to separate residuals into anomaly-relevant and domain-specific components. DART combines a Residual Disentanglement Module, which preserves anomaly-discriminative information while suppressing domain information, with Cross-Domain Meta-Training, which forms episodes across source domains to make the separation effective. The method is architecture-agnostic and can be inserted between resi
    
[^33]: 被接受但实则错误：API迁移后LLM代码修复的静默失败

    Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

    [https://arxiv.org/abs/2611.08269](https://arxiv.org/abs/2611.08269)

    预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。

    

    在运行时或API迁移之后，经LLM修复的程序可能顺利运行却返回错误结果，而基于典型测试输入的发布检查会将其放行发布。我们报告了一项预注册研究，包含73个任务，覆盖CPython、Node.js和SQLite中60个有文档记录的变更。每个候选修复都在真实运行时上针对未公开的契约执行。在整个修复空间中，基于典型输入的全断言门检查放行了0.410比例的错误修复。一个根据发布说明编写的边缘输入将该比例降至0.065，在15个未见过的集群上降至0.061。来自一个从未见过该程序的盲测GPT-5.6-luna编写者的三个输入达到了0.070。两个后续冻结阶段将候选菜单扩大至1491个片段，随后又添加了233个在典型输入上与遗留输出匹配的困难干扰项。边缘输入（0.058）和前沿模型编写者（0.058）通过了其预注册测试。然而，仍有0.168和0.153的困难错误程序通过了检查。

    arXiv:2611.08269v1 Announce Type: cross  Abstract: After a runtime or API migration, an LLM-repaired procedure can run cleanly yet return wrong results, and a release check on typical test inputs ships it. We report a preregistered study of 73 tasks covering 60 documented changes in CPython, Node.js and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the complete repair space, the all-assertions gate on typical inputs released 0.410 of wrong repairs. One edge input written from the release note cut this to 0.065, and to 0.061 on 15 unseen clusters. Three inputs from a blind GPT-5.6-luna writer that never saw the program reached 0.070. Two further frozen stages enlarged the menu to 1491 fragments and then added 233 hard distractors that match the legacy output on typical inputs. The edge input (0.058) and the frontier-model writer (0.058) passed their preregistered tests. Yet 0.168 and 0.153 of the hard wrong programs still passed. Two local-mod
    
[^34]: 通过解耦行为预训练实现神经运动解码的规模化

    Scaling Neural Motor Decoding via Decoupled Behavioral Pretraining

    [https://arxiv.org/abs/2611.08263](https://arxiv.org/abs/2611.08263)

    提出BEEMO框架，通过解耦行为与神经表征学习，利用丰富且易获取的非配对行为数据进行训练，独立扩展行为和神经监督，从而在非人类灵长类皮层内运动数据集上显著提升解码性能。

    

    最近的神经解码方法通过在大量配对的神经-行为记录数据集上训练获得了强大的性能，但这类数据集昂贵、具有侵入性且难以规模化。相比之下，行为数据丰富且易于收集，这引发了一个问题：神经解码能否从解耦行为与神经表征学习中受益。我们提出了BEEMO，这是一个灵活的多会话框架，支持使用任意比例配对的神经-行为记录与非配对行为数据进行训练。行为解码器直接从行为轨迹中学习运动动态，而神经活动在可用时通过交叉注意力层被纳入，从而使行为和神经监督能够独立扩展。在多个非人类灵长类动物的皮层内运动数据集和任务上，我们展示了引入非配对行为数据能够提升解码性能。

    arXiv:2611.08263v1 Announce Type: cross  Abstract: Recent neural decoding approaches achieve strong performance by training on large collections of paired neural-behavioral recordings, but such datasets are expensive, invasive, and difficult to scale. In contrast, behavioral data is abundant and easy to collect, raising the question of whether neural decoding can benefit from decoupling behavioral and neural representation learning. We introduce BEEMO, a flexible multi-session framework that enables training with arbitrary mixtures of paired neural-behavioral recordings and unpaired behavioral data. A behavior decoder learns movement dynamics directly from behavioral trajectories, while neural activity is incorporated through cross-attention layers when available, allowing behavioral and neural supervision to scale independently. Across multiple intracortical motor datasets and tasks in nonhuman primates, we show that incorporating unpaired behavioral data improves decoding performance
    
[^35]: 模拟视角的几何学：在多模态语言模型中分离表征与行为

    The Geometry of Simulated Perspectives: Dissociating Representation and Behaviour in a Multimodal Language Model

    [https://arxiv.org/abs/2611.08262](https://arxiv.org/abs/2611.08262)

    该研究发现多模态语言模型中提示措辞对内部表征几何的影响大于人格身份本身，而人格内容则主要驱动行为判断，从而揭示了表征与行为在视角模拟中的解耦现象。

    

    多模态语言模型能够模拟不同的观察者，然而指派的视角如何重塑其内部表征、以及这些变化是否决定其判断，目前仍不清楚。本研究以 Qwen3-VL-8B-Instruct 为对象，考察人格条件化的视觉趣味性所对应的表征几何。在100张图像和53种提示条件下，我们系统性地改变人格内容以及用于引入人格的语言表述方式，同时记录行为评分和逐层隐状态。我们观察到表征几何随处理深度发生系统性重组：早期层主要编码图像身份，而人格特异的结构则在较后的层中出现。引人注目的是，提示措辞对内部表征的塑造作用强于人格身份本身，而人格内容对行为判断的影响更大。这种分离进一步通过最小同一性（此处摘要原文截断）……

    arXiv:2611.08262v1 Announce Type: new  Abstract: Multimodal language models can simulate different observers, yet it remains unclear how assigned perspectives reshape their internal representations and whether these changes govern their judgements. Here, we investigate the representational geometry of persona-conditioned visual interestingness in Qwen3-VL-8B-Instruct. Across 100 images and 53 prompting conditions, we systematically vary persona content and the linguistic framing used to introduce it, while recording behavioural ratings and layer-wise hidden states. We observe a systematic reorganisation of representational geometry across processing depth: early layers primarily encode image identity, whereas persona-specific structures emerge in later layers. Strikingly, prompt framing reshapes internal representations more strongly than persona identity, whereas persona content has a greater impact on behavioural judgements. This dissociation is further illustrated by minimal identit
    
[^36]: 面向价值的策略蒸馏方法用于可解释的风电场功率控制

    Value-oriented policy distillation for interpretable wind farm power control

    [https://arxiv.org/abs/2611.08245](https://arxiv.org/abs/2611.08245)

    该论文提出了一种面向价值的策略蒸馏框架，利用可解释提升机（EBM）将强化学习策略蒸馏为透明可解释的控制策略，从而实现风电场的隐式平衡功率控制。

    

    风电场功率控制可以帮助系统维持平衡并避免停电。在欧洲，这可以通过隐式平衡来实现，即风电场利用输电系统运营商发布的信息，在预判不平衡价格的基础上调整其功率设定点。通过这种方式，风电场既有助于恢复系统平衡，又能降低自身因预测误差而产生的不平衡成本。然而，隐式平衡是一个具有挑战性的问题，因为结算价格事先未知，并且可能受到所采用的控制策略的影响。强化学习方法有助于解决这一问题，但缺乏在真实风电场上部署所需的可解释性。因此，我们提出了一种基于可解释提升机（EBMs，广义可加模型的一种变体）的策略蒸馏框架，以获得透明的控制策略。为了提升EBM模型所捕获的价值，我们引入了……（原文摘要在此处截断）

    arXiv:2611.08245v1 Announce Type: new  Abstract: Wind farm power control can help the system maintain balance and avoid blackouts. In Europe, this can be achieved through implicit balancing, where a wind farm adjusts its power setpoint in anticipation of the imbalance price using information published by the Transmission System Operator. Thereby, the wind farm helps restore system balance while also reducing its own imbalance costs resulting from forecast errors. However, implicit balancing is a challenging problem since the settlement price is unknown in advance and can be impacted by the pursued control strategy. Reinforcement learning methods could help to solve this problem, but lack the interpretability needed to be deployed on real wind farms. Therefore, we propose a policy distillation framework using explainable boosting machines (EBMs), a variant of generalized additive models, to obtain a transparent control policy. To increase the value captured by the EBM model, we introduc
    
[^37]: 信号、证据与语境：真实世界LLM对话中的用户审视行为

    Signals, Evidence, and Context: User Scrutiny in Real-World LLM Conversations

    [https://arxiv.org/abs/2611.08232](https://arxiv.org/abs/2611.08232)

    该研究通过分析近49万条真实LLM对话发现，模型答案中宽泛的归因表述远多于可查验的显式来源链接，不确定性等回复线索与用户的核查和质疑行为相关，但用户的审视行为高度集中于少数对话。

    

    大语言模型（LLM）常常给出综合性的答案，却不让用户方便地查验其背后的支持证据。我们基于来自ShareChat和WildChat的489,283组助手-用户回复对，研究这些答案的特征如何与用户随后的接受、核查和质疑行为相关联。在符合条件的回复中，宽泛的归因语言出现率为31.9%，而显式URL仅为1.33%，这揭示了“有据可依”的声明与可查验支持证据得以保留之间的落差。归因、不确定性及其他回复线索通常同时伴随核查和质疑行为；只有不确定性在汇总统计上显示出小幅且可区分的、偏向核查的关联。在“技术求助”类别中，不确定性相对质疑更倾向于引发核查；而在“信息查询”类别中，拒绝回答和推理痕迹更倾向于引发质疑。审视行为集中于部分对话之中，而其表面上的逐轮持续性在控制上下文后会大幅缩小。

    arXiv:2611.08232v1 Announce Type: new  Abstract: Large language models (LLMs) often present synthesized answers without making their supporting evidence easy to inspect. We study how features of these answers relate to users' subsequent accepting, checking, and challenging in 489,283 assistant-user response pairs from ShareChat and WildChat. Broad attribution language appears in 31.9% of qualifying responses, compared with explicit URLs in 1.33%, revealing a gap between claims of grounding and preserved inspectable support. Attribution, uncertainty, and other response cues generally accompany both checking and challenging; only uncertainty shows a small statistically distinguishable pooled association favoring checking. Within Technical Help, uncertainty favors checking relative to challenging; refusals and reasoning traces favor challenging in Seeking Information. Scrutiny is concentrated in some conversations, while its apparent turn-to-turn persistence becomes much smaller when conv
    
[^38]: SMART：面向验证导向问题求解的基于状态的多智能体推理轨迹

    SMART: State-based Multi Agent Reasoning Traces for Verification-oriented Problem-Solving

    [https://arxiv.org/abs/2611.08196](https://arxiv.org/abs/2611.08196)

    SMART框架通过捕捉问题上下文中参与实体的隐式状态，并利用多种LLM和基于角色的结构化提示策略生成多样化、互补且与解决方案相关的前提知识，将复杂科学论断的验证性能提升高达20%。

    

    arXiv:2611.08196v1 公告类型：新论文。摘要：上下文有限的复杂问题（例如论断）需要一种手段来获取解决方案推理所必需的额外知识。我们提出了一个多智能体框架SMART，通过捕捉问题上下文中参与实体的隐式状态来获取知识。SMART使用多种大语言模型（LLM）和结构化的基于角色的提示策略，将状态知识整理为多样化、互补且与解决方案相关的前提。利用这些经真实世界验证的整理前提知识，SMART的知识引导解决方案推理在跨越3个数据集的复杂科学论断上将验证性能提升高达20%。我们证明了将前提扎根于真实世界知识是必要的，且可将性能提升高达5%。

    arXiv:2611.08196v1 Announce Type: new  Abstract: Complex problems with limited context, such as claims, require a means to obtain additional knowledge necessary for solution reasoning. We propose a multi-agent framework, SMART, to obtain knowledge through capturing the implicit state of participating entities in a problem context. SMART curates the state knowledge as diverse, complementary and solution-salient premises using a variety of LLMs and a structured role-based prompt strategy. Using knowledge from these real-world verified curated premises, SMART's knowledge guided solution reasoning improves validation performance up to 20% on complex scientific claims spanning 3 datasets. We show that grounding premises in real-world knowledge is necessary and improves performance up to 5%.
    
[^39]: 知道何时教学：从多语言推理蒸馏中的可教学前缀学习

    Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation

    [https://arxiv.org/abs/2611.08179](https://arxiv.org/abs/2611.08179)

    提出TOPSD方法，利用教师-学生似然估计教师漂移的起点并选择可教学前缀进行监督，解决了多语言推理自蒸馏中的教师漂移问题，尤其改善了低资源语言的推理表现。

    

    同策略自蒸馏（OPSD）通过在学生模型自身的推理结果上进行训练来提升推理能力，其中使用同一模型并以参考解答为条件来充当教师模型。将OPSD应用于多语言推理时，要求教师模型即使面对学生模型在中间步骤中出现的错误，也能在各种语言中提供有用的反馈。我们观察到一种称之为“教师漂移”的现象：随着这些错误的累积，教师模型可能会跟随学生模型的错误步骤，而不是纠正它们。我们的分析表明，这种转变在低资源语言中往往发生得更早。为了应对这些跨语言差异，我们提出了可教学同策略自蒸馏（TOPSD），它可以确定在每个推理轨迹上是否应用以及在何处应用教师监督。TOPSD利用教师-学生似然来估计漂移开始的位置，并选择一个前缀进行监督，然后验证教师模型是否能够恢复……

    arXiv:2611.08179v1 Announce Type: cross  Abstract: On-Policy Self-Distillation (OPSD) improves reasoning by training a student on its own reasoning, using the same model conditioned on reference solutions as a teacher. Applying OPSD to multilingual reasoning requires the teacher to provide useful feedback across languages, even when the student makes mistakes in intermediate steps. We observe a phenomenon we call Teacher Drift: as these mistakes accumulate, the teacher may follow the student's incorrect steps instead of correcting them. Our analysis shows that this shift tends to occur earlier in lower-resource languages. To account for these differences across languages, we propose Teachable On-Policy Self-Distillation (TOPSD), which determines where and whether teacher supervision should be applied to each reasoning trajectory. TOPSD uses teacher-student likelihood to estimate where drift begins and select a prefix for supervision. It then verifies whether the teacher can recover the
    
[^40]: BabyTheorist：一个从纯观察中学习理论化理解世界的基准测试

    BabyTheorist: A Benchmark for Learning to Theorize the World from Observation Alone

    [https://arxiv.org/abs/2611.08177](https://arxiv.org/abs/2611.08177)

    提出了 BabyTheorist 基准，用于评估模型能否仅凭纯视觉观察学习可复用的世界理论，并将其泛化应用到全新情境中。

    

    我们如何判断一台机器是否真正理解世界？我们认为其中一项关键能力是“观察性理论学习”：从纯观察中获取可复用的理论，并将其应用于新的情境。当前的视觉推理基准主要关注后者，即将每个任务呈现为按共同规则分组的少样本谜题。我们提出了 BabyTheorist，这是一个用于观察性理论学习的受控训练与评估基准。BabyTheorist 从隐藏的组合程序中生成连续的“前后”视觉观察流；在训练过程中，学习者只能看到最终的转换结果，而无法获得程序标签、任务身份或支持集。为了支持理论的学习与评估，BabyTheorist 对程序语言、程序复杂度的课程安排、反复出现的频繁子程序，以及带有级别跳跃的连续训练-测试协议进行了控制。文中评估了七个代表性基线……

    arXiv:2611.08177v1 Announce Type: new  Abstract: How would we know whether a machine truly understands the world? We argue that one key capability is \textit{observational theory learning}: acquiring reusable theories from pure observations and applying them to new situations. Current visual reasoning benchmarks focus primarily on the latter, presenting each task as a few-shot puzzle grouped by a common rule. We introduce BabyTheorist, a benchmark for controlled training and evaluation of observational theory learning. BabyTheorist generates continual streams of before-after visual observations from hidden compositional programs; during training, learners see only the resulting transitions, with no program labels, task identities, or support sets. To support theory learning and evaluation, BabyTheorist controls the program language, a curriculum over program complexity, recurring frequent subprograms, and a continual train-test protocol with level skipping. Seven representative baselin
    
[^41]: PCBRouteAgent：LLM智能体距离成为PCB布线工程师还有多远？

    PCBRouteAgent: How Close Are LLM Agents to Becoming PCB Routing Engineers?

    [https://arxiv.org/abs/2611.08175](https://arxiv.org/abs/2611.08175)

    PCBRouteAgent是一个模仿人类工程师“全局规划+局部执行”协作方式的多模态LLM布线智能体，通过全局布线规划器与基于KiCad的详细布线执行器配合，并利用结构化失败反馈迭代修正规划，在247块电路板上实现了可靠的自动化PCB布线。

    

    自动化印刷电路板（PCB）布线需要在严格的设计规则约束下协调板级连通性决策与几何执行。尽管近期的智能体已经能够与原生布线工具交互，但在复杂电路板上实现可靠布线仍然具有挑战性。我们提出了PCBRouteAgent，一个受人类工程师协调全局策略与局部执行方式启发的多模态PCB布线智能体。全局布线规划器结合电路板图像以及结构化的电气与几何信息，指定连接拓扑、布线顺序、布线通道和层间转换。详细布线执行器通过KiCad的原生布线引擎实现这些规划指导，并执行有界的局部恢复。未能解决的失败会以结构化反馈的形式返回，在保留已接受布线的同时实现有针对性的规划修订。我们在来自PCBWorld-Bench四个测试子集的247块合成与真实电路板上对PCBRouteAgent进行了评估。

    arXiv:2611.08175v1 Announce Type: new  Abstract: Automating printed circuit board (PCB) routing requires coordinating board-level connectivity decisions with geometric execution under strict design rules. Although recent agents can interact with native routing tools, reliable routing on complex boards remains challenging. We present PCBRouteAgent, a multimodal PCB routing agent inspired by how human engineers coordinate global strategy with local execution. A Global Routing Planner combines board images and structured electrical and geometric information to specify connection topology, routing order, corridors, and layer transitions. A Detailed Routing Executor realizes these guides through KiCad's native routing engine and performs bounded local recovery. Unresolved failures are returned as structured feedback, enabling targeted guide revisions while preserving accepted routes. We evaluate PCBRouteAgent on 247 synthetic and real-world boards from four PCBWorld-Bench test splits. It ac
    
[^42]: 可解释的多模态MRI-临床数据融合用于急性缺血性卒中后功能结局预测

    Interpretable Multimodal MRI-Clinical Fusion for Functional Outcome Prediction After Acute Ischemic Stroke

    [https://arxiv.org/abs/2611.08154](https://arxiv.org/abs/2611.08154)

    该研究提出一种可解释的多模态MRI与临床数据融合方法，证明FLAIR-临床融合（AUC=0.948）能显著提升急性缺血性卒中后三个月功能结局的预测性能。

    

    本研究评估了发病第1天多模态MRI对预测急性缺血性卒中（AIS）后三个月功能结局的额外预后价值。研究分析了66名具有ADC、DWI、FLAIR及临床数据的AIS患者。研究从每种MRI模态中提取三维预训练ResNet嵌入特征，使用折间隔离的主成分分析进行降维，并与结构化临床变量及ADC衍生的病灶生物标志物相结合。在多种MRI模态组合上，采用八折分层分组交叉验证评估线性支持向量机的性能。仅使用临床数据的模型取得了较强的性能（AUC = 0.899 ± 0.151），而仅使用MRI的模型判别能力有限（AUC ≈ 0.67-0.73）。在多模态配置中，FLAIR-临床融合取得了最高的平均交叉验证性能（AUC = 0.948 ± 0.061），其次是ADC-临床融合（AUC = 0.941 ± 0.094）。

    arXiv:2611.08154v1 Announce Type: cross  Abstract: This study evaluates the added prognostic value of day-1 multimodal MRI for predicting three-month functional outcome after acute ischemic stroke (AIS). Sixty-six AIS patients with ADC, DWI, FLAIR, and clinical data were analyzed. Three-dimensional pretrained ResNet embeddings were extracted from each MRI modality, reduced using fold-isolated principal component analysis, and combined with structured clinical variables and ADC-derived lesion biomarkers. Linear support vector machines were evaluated using eight-fold stratified group cross-validation across MRI modality combinations. Clinical-only modeling achieved strong performance (AUC = $0.899 \pm 0.151$), while MRI-only models showed limited discrimination (AUC $\approx$ $0.67$-$0.73$). Among multimodal configurations, FLAIR-clinical fusion achieved the highest mean cross-validated performance (AUC = $0.948 \pm 0.061$), followed by ADC-clinical fusion (AUC = $0.941 \pm 0.094$). Full
    
[^43]: 面向托卡马克多模态任务的生成式基础模型

    A Generative Foundation Model for Multimodal Tokamak Tasks

    [https://arxiv.org/abs/2611.08153](https://arxiv.org/abs/2611.08153)

    该论文扩展了TokaMark基准并提出了一种可复用的生成式基础模型，能够自适应地匹配托卡马克各类任务的输入输出模式并联合建模异构目标信号，无需为每个任务单独训练模型。

    

    托卡马克实验记录了多样化的测量数据，包括波形、剖面、场和标量，这些数据被用于重建等离子体、预测其演化、恢复相关控制信号以及估计破裂风险。大多数模型是针对单一任务和固定的输入输出集合分别构建的，每个模型都依赖于一组预设的诊断子集，而非对完整的多模态观测集合进行结构化处理以提取关于放电的全局、跨信号信息。这些局限性促使人们需要一个可复用的基础模型，它能够适应每个任务的输入输出模式，并联合建模异构的目标信号，而不是为每种配置单独重新训练模型。在这项工作中，我们扩展了TokaMark基准，增加了两组任务，分别用于执行器动作反演和破裂预测，从而在六个任务组中形成了20个MAST任务，并提供了明确的模式定义和评估协议。随后，我们介绍了GenTo（摘要在此处截断）

    arXiv:2611.08153v1 Announce Type: new  Abstract: Tokamak experiments record diverse measurements, including waveforms, profiles, fields, and scalars, which are used to reconstruct the plasma, forecast its evolution, recover associated control signals, and estimate disruption risk. Most models are built separately for one task and a fixed set of inputs and outputs, and each relies on a prescribed subset of diagnostics rather than structuring the full multimodal observation set to extract global, cross-signal information about the discharge. These limitations motivate a reusable foundation model that adapts to each task's input-output schema and models heterogeneous target signals jointly, rather than retraining a separate model for every configuration. In this work, we extend the TokaMark benchmark with two groups of tasks for actuator action inversion and disruption prediction, yielding 20 MAST tasks in six groups with explicit schemas and evaluation protocols. Then, we introduce GenTo
    
[^44]: 谁能独立评估人工智能？一个用于分析评估基础设施的框架

    Who Can Evaluate AI Independently? A Framework for Analysing Evaluation Infrastructure

    [https://arxiv.org/abs/2611.08134](https://arxiv.org/abs/2611.08134)

    本文提出PAAI（权力、可及性、充分性、独立性）分析框架，通过考察评估基础设施的权威分配，结合非洲AI评估项目与国际AI安全研究所网络的案例，揭示谁能够真正独立地评估AI系统。

    

    人工智能评估在AI安全与治理中日益处于核心地位，然而参与评估本身并不意味着对评估的设计、执行或其所依赖的资源拥有权威。本文研究了基础设施安排如何分配这种权威，从而决定谁能够独立地评估人工智能系统。借鉴科学技术研究（STS）、政治经济学和应用评估标准，我们提出了PAAI（权力、可及性、充分性、独立性）框架，用于考察谁控制评估基础设施、基础设施可用的条件、它是否足以满足评估目的，以及它支持何种自主性。我们将PAAI应用于十二个非洲人工智能评估项目（2022-2025年），以及人工智能安全研究所国际网络三次已发布联合测试活动中全部十个成员研究所的公开角色。我们识别出三种反复出现的配置模式：抽取式……（摘要原文在此处截断）

    arXiv:2611.08134v1 Announce Type: cross  Abstract: AI evaluation is increasingly central to AI safety and governance, yet taking part in an evaluation does not by itself confer authority over its design, its execution or the resources it depends on. This paper examines how infrastructure arrangements distribute that authority, and so shape who can evaluate AI systems independently. Drawing on Science and Technology Studies, political economy and applied evaluation standards, we introduce PAAI (Power, Access, Adequacy, Independence), a framework for examining who controls evaluation infrastructure, the terms on which it is available, whether it is adequate to the evaluation's purpose and what autonomy it supports. We apply PAAI to twelve African AI evaluation projects (2022-2025) and to the documented roles of all ten member institutes in the three published joint testing exercises of the international network of AI Safety Institutes. We identify three recurring configurations: extracti
    
[^45]: 创建你自己的工作栈：在科技初创公司中部署自托管 AI 与工具

    Create Your Own Work Stack: Deploying Self-hosted AI and Tooling in a Tech Startup

    [https://arxiv.org/abs/2611.08116](https://arxiv.org/abs/2611.08116)

    该论文分享了一家小型软件公司部署混合 AI 栈的实践经验——商业 AI 代理负责复杂任务，本地开源模型承担低复杂度任务、组织信息检索及自托管工具同步，从而在不牺牲质量的前提下削减约 30–40% 的商业 AI 使用成本。

    

    自托管 AI 通常被视为降低推理成本和增强控制力的一种方式，但部署本身也会带来成本。我们报告了在 Grayhat Developers（一家拥有 AI 密集型开发工作流的小型软件公司）部署混合 AI 栈的实践经验。我们最初的动机是应对商业 AI 使用成本的不断增长：Claude/Codex 每月产生约 325 美元的费用，记录的 token 超过 259.3 亿。我们尝试用本地开源模型替代商业编码代理，但在发现其性能和推理速度不足之后，我们采用了混合方法：商业代理继续负责复杂任务，而本地模型处理较低复杂度的任务、组织信息检索，以及与自托管工具（包括 Outline、Plane 和 ERPNext）的同步。我们估计大约 30–40% 的商业

    arXiv:2611.08116v1 Announce Type: new  Abstract: Self-hosting AI is often presented as a way to reduce inference costs and increase control, but deployment introduces its own costs. We report our experience deploying a hybrid AI stack at Grayhat Developers, a small software company with an AI-intensive development workflow. Our initial motivation was to counter the growing cost of commercial AI usage: Claude/Codex incurred approximately \$325 per month, with more than 25.93 billion recorded tokens. We experimented with local open-source models as replacements for commercial coding agents, but after learning that their performance and inference speed were insufficient, we adopted a hybrid approach in which commercial agents remained responsible for complex tasks while local models handled lower-complexity tasks, organizational information retrieval, and synchronization with self-hosted tools including Outline, Plane, and ERPNext.   We estimated that approximately 30--40\% of our commerc
    
[^46]: 梯度提升用于张量网络核机器中的高效秩发现

    Gradient Boosting for Efficient Rank Discovery in Tensor Network Kernel Machines

    [https://arxiv.org/abs/2611.08093](https://arxiv.org/abs/2611.08093)

    提出梯度提升框架GB-CPKM，通过顺序添加固定秩CP张量网络学习器来增量式提升模型容量，从而避免显式秩搜索，实现最高6倍的训练加速和2倍的峰值内存节省。

    

    张量网络核机器（TNKMs）将非线性特征与低秩张量表示相结合，以高效地建模复杂关系。然而，选择合适的张量秩仍然具有挑战性，因为传统的秩搜索方法需要训练多个候选模型，从而增加了计算成本。在本工作中，我们聚焦于规范多线性（CP）张量网络参数化，并提出GB-CPKM——一种梯度提升框架，通过顺序添加固定秩的学习器来构建更高容量的预测器。该方法通过增量式增加模型容量，为秩选择提供了新的视角：集成规模基于验证性能来选择，而无需进行显式的秩搜索。在真实世界回归基准上的实验表明，与基于交叉验证的传统秩选择方法相比，GB-CPKM的训练速度最高可提升6倍，峰值内存使用最多可减少2倍，同时……

    arXiv:2611.08093v1 Announce Type: cross  Abstract: Tensor network kernel machines (TNKMs) combine nonlinear features with low-rank tensor representations to model complex relationships efficiently. However, selecting an appropriate tensor rank remains challenging, as conventional rank-search methods require training multiple candidate models, increasing computational costs. In this work, we focus on a canonical polyadic (CP) tensor network parameterization and propose GB-CPKM, a gradient boosting framework that builds a higher-capacity predictor by sequentially adding fixed-rank learners. This approach provides a new perspective on rank selection by incrementally increasing model capacity, with ensemble size selected based on validation performance rather than through explicit rank search. Experiments on real-world regression benchmarks show that GB-CPKM trains up to 6x faster and uses up to 2x less peak memory than a conventional rank-selection method based on cross-validation, while 
    
[^47]: 价值还是注意力：面向图注意力网络的逐层相关性传播研究

    Value or Attention: Investigating Layer-wise Relevance Propagation for Graph Attention Networks

    [https://arxiv.org/abs/2611.08088](https://arxiv.org/abs/2611.08088)

    该论文提出了一种带自由权重的LRP乘积规则，将图注意力网络中节点接收到的相关性分解为值路径的内容相关性和注意力路径的路由相关性两部分，并将现有解释方法统一为其特例。

    

    图注意力网络越来越多地应用于从医学到材料科学等需要科学家解读模型推理过程的领域，然而它们所学习到的路由机制对大多数解释方法而言仍然是不透明的。逐层相关性传播（LRP）能够遵循模型的计算过程，但在注意力层中，到达某个节点的相关性是两个机制上截然不同的量的总和：通过值路径传递的内容相关性，以及通过注意力路径传递的路由相关性。现有的规则要么丢弃其中一个部分，要么将它们简单相加成一个数值。我们提出了一个作用于这两个因子、带有自由权重的单一乘积规则，能够将现有方法作为特例加以恢复。值路径衡量的是节点相对于零基线所提供的内容；注意力路径衡量的是节点相对于其邻域如何塑造了传递的信息，其符号记录了节点的特征是否按照预测所需的方式影响了门控机制。

    arXiv:2611.08088v1 Announce Type: cross  Abstract: Graph Attention Networks (GATs) are increasingly used in domains where scientists need to read a model's reasoning, from medicine to materials science, yet the routing they learn remains opaque to most explainers. Layer-wise Relevance Propagation (LRP) follows the model's computation, but in an attention layer the relevance reaching a node is the sum of two mechanically different quantities: its content relevance through the value path and its routing relevance through the attention path. Existing rules either drop one or add them into a single number. We write a single product rule with free weights on the two factors that recovers current approaches as special cases. The value path measures what content a node supplied against a zero baseline; the attention path measures how the node shaped what was passed relative to its neighborhood, and its sign records whether the node's features moved the gating the way the prediction demanded. 
    
[^48]: MUSS：一个面向单细胞基因调控的多尺度序列模型

    MUSS: A Multi-scale and Sequence-based Model for Single-cell Gene Regulation

    [https://arxiv.org/abs/2611.08086](https://arxiv.org/abs/2611.08086)

    MUSS是一个基于序列的多尺度单细胞基因调控模型，利用冻结的预训练DNA序列编码器提取峰级与基因级表示，并通过峰-峰自注意力和距离偏置的峰-基因交互进行整合，在保留DNA序列信息的同时实现细胞水平的基因调控建模。

    

    对单细胞基因调控进行建模需要在多个调控尺度上联合表示DNA序列、染色质可及性和基因表达。基于序列的模型能够捕捉DNA水平的调控语法，但通常缺乏细胞水平的分辨率；而许多单细胞组学模型将基因和峰表示为固定词表的token，却不包含其底层的DNA序列信息。本文提出MUSS，一个基于序列的多尺度单细胞基因调控模型，它同时保留了DNA水平和细胞水平的分辨率。MUSS利用一个冻结的预训练序列编码器从DNA中提取峰级和基因级表示，并通过峰-峰自注意力和距离偏置的峰-基因交互将它们整合起来。先进行峰值掩码语言建模，再在配对的scATAC-seq和scRNA-seq数据上训练，使这些表示与细胞特异性的调控状态对齐。在基因表达预测、增强子-基因关联、基序引导的（原文在此处截断）……

    arXiv:2611.08086v1 Announce Type: new  Abstract: Modeling gene regulation in single cells requires jointly representing DNA sequence, chromatin accessibility, and gene expression across regulatory scales. Sequence-based models capture DNA-level regulatory grammar but often lack cell-level resolution, whereas many single-cell omics models represent genes and peaks as fixed-vocabulary tokens without their underlying sequence. Here, we present MUSS, a sequence-based multi-scale model for single-cell gene regulation that preserves both DNA-level and cell-level resolution. MUSS derives peak- and gene-level representations from DNA using a frozen pre-trained sequence encoder and integrates them through peak-peak self-attention and distance-biased peak-gene interactions. Peak masked language modeling followed by training on paired scATAC-seq and scRNA-seq aligns these representations with cell-specific regulatory states. Across gene expression prediction, enhancer-gene linking, motif-guided r
    
[^49]: DictLLM：基于字典核的大语言模型训练后压缩

    DictLLM: Post-training Compression of Large Language Model with Dictionary Kernels

    [https://arxiv.org/abs/2611.08082](https://arxiv.org/abs/2611.08082)

    DictLLM提出了一种基于字典核的训练后压缩方法，将分组层的权重矩阵分解为共享字典与层特定稀疏系数，无需进一步训练即可在移动设备上高效压缩大语言模型。

    

    在移动设备上高效部署大语言模型（LLM）需要在性能与计算、内存限制之间取得平衡。训练后压缩是管理这种权衡的有效方法，同时比训练过程中压缩所需计算量少得多。我们提出了DictLLM，一种基于字典的压缩方案，其中分组层内的权重矩阵被分解为共享字典和层特定的稀疏系数，可直接从预训练模型获得而无需进一步训练。具体而言，我们将DictLLM构建为一个四步训练后优化流程：首先，通过对输入缩放的权重矩阵进行奇异值分解（SVD）获得共享字典，从而在分组层之间分摊秩；其次，利用字典感知的Hessian矩阵将层特定系数稀疏化至目标压缩率；第三，在得到稀疏系数的基础上对字典进行精炼。

    arXiv:2611.08082v1 Announce Type: new  Abstract: Efficient deployment of large language models (LLMs) on mobile devices requires balancing performance with compute and memory constraints. Post-training compression is an effective approach to managing this trade-off, while requiring much less computation than in-training counterparts. We propose DictLLM, a dictionary-based compression scheme in which weight matrices within grouped layers are decomposed into a shared dictionary and layer-specific sparse coefficients, directly obtained from a pre-trained model without further training. Specifically, we cast DictLLM as a four-step post-training optimization. First, we obtain shared dictionaries via singular value decomposition (SVD) of input-scaled weight matrices, amortizing rank across grouped layers; second, we sparsify the layer-specific coefficients to the target compression ratio using a dictionary-aware Hessian; and third, we refine the dictionaries given the resulting sparse coeffi
    
[^50]: LLM究竟能真正对神经网络的权重进行推理和编程，还是只是“打了类固醇的记忆”？——探索“权重空间编程”

    Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring "Weight-Space Programming"

    [https://arxiv.org/abs/2611.08070](https://arxiv.org/abs/2611.08070)

    论文在无工具、无前向传播的条件下测试LLM读取小型卷积网络原始权重的能力，发现其表现主要源于准确的局部算术计算与对公开数据集的记忆先验，而非对权重流动的真正全局推理。

    

    大型语言模型（LLM）最近被报道能够直接读取和写入神经网络的原始权重，这一能力被称为“权重空间编程”。我们探究这一能力的“读取”端究竟反映了对权重流动的推理，还是网络边缘的局部算术运算与对公开数据集记忆先验的结合。我们在五个图像数据集上训练的约12.5万参数的小型卷积网络上测试了 Claude Opus 5.5，将每个权重以纯文本形式提供，不使用工具、不执行前向传播、不提供数据集或类别名称。该模型能够执行准确的局部计算：批归一化统计量使其能够推断出在三次更新后停止训练的网络大约经历了三次优化器步骤，输出层的符号模式能够恢复 CIFAR-10 的车辆/动物分类。会话记录显示模型紧密读取了第一层和最后一层，并对中间层进行了采样，而且若干声明建立在部分追踪的基础上……

    arXiv:2611.08070v1 Announce Type: cross  Abstract: Large language models (LLMs) have recently been reported to read and write the raw weights of neural networks directly, a capability termed weight-space programming. We ask whether the reading side of this capability reflects reasoning about weight flows or a combination of local arithmetic at the network's edges and memorized priors over public datasets. We test Claude Opus 5.5 on small convolutional networks of about 125,000 parameters trained on five image datasets, providing every weight as plain text with no tools, no forward passes, and no dataset or class names. The model performs accurate local calculations: batch-normalization statistics lead it to infer about three optimizer steps on networks stopped after three updates, and output-layer sign patterns recover the vehicle/animal split of CIFAR-10. The sessions report reading the first and last layers closely and sampling the middle, and several claims built on partial tracing 
    
[^51]: NutriVisRAG：基于多模态示例锚定的视觉语言模型营养估计

    NutriVisRAG: Multimodal Exemplar Grounding for Nutrient Estimation with Vision-Language Models

    [https://arxiv.org/abs/2611.08054](https://arxiv.org/abs/2611.08054)

    提出了NutriVisRAG检索增强框架，通过视觉编码器检索视觉相似的已标注餐食作为多模态上下文示例来锚定视觉语言模型的营养估计，无需任务特定微调即可提升估计性能。

    

    饮食是健康的主要决定因素，然而从自由生活场景的餐食图像中进行准确的膳食评估仍然具有挑战性。现有的基于检索的视觉语言模型（VLM）营养估计方法通常依赖于文本形式的食物表示，而据我们所知，直接以视觉相似的已标注餐食作为锚定基础的方法尚未被研究。我们提出了NutriVisRAG，这是一个检索增强生成框架，它利用视觉编码器检索具有已知营养标注的相似餐食，并在推理时将这些餐食作为多模态上下文示例提供给VLM进行营养估计，且无需针对任务进行微调。我们在四个数据集、四个VLM、三个检索编码器和两种提示策略上对NutriVisRAG进行了评估，其中包括来自接受胰岛素治疗的糖尿病成人患者的自由生活餐食图像，其碳水化合物估计值与两位营养师的标注进行了比较。在所有96种配置（4个数据

    arXiv:2611.08054v1 Announce Type: cross  Abstract: Diet is a major determinant of health, yet accurate dietary assessment from free-living meal images remains challenging. Existing retrieval-based Vision-Language Model (VLM) approaches to nutrient estimation typically rely on textual food representations, while, to our knowledge, direct grounding on visually similar annotated meals has not been investigated. We present NutriVisRAG, a retrieval-augmented generation framework that uses a vision encoder to retrieve similar meals with known nutritional annotations and provide them as multimodal in-context exemplars for VLM nutrient estimation at inference time, without task-specific fine-tuning. We evaluate NutriVisRAG across four datasets, four VLMs, three retrieval encoders, and two prompting strategies, including free-living meals from adults with insulin-treated diabetes, where carbohydrate estimates are compared against two dietitians' annotations. Across all 96 configurations (4 data
    
[^52]: 当不完美的事实核查适得其反时：错误信息存在下的社会学习

    When Imperfect Automated Fact-Checking Backfires: Social Learning in the Presence of Misinformation

    [https://arxiv.org/abs/2611.08053](https://arxiv.org/abs/2611.08053)

    该研究通过序贯社会学习模型证明，免费但易错的自动事实核查工具是一把双刃剑——它虽能帮助用户摆脱错误的舆论级联，但其错误可能误导原本会进行完美人工核查的用户，从而反而降低中等可靠性文章下的社会福利。

    

    自动事实核查工具成本低廉、速度快但并不完美，如今在社交平台上日益普及。我们在一个关于错误信息的序贯社会学习模型中研究此类工具何时可能适得其反。智能体会观察一篇文章的可靠性以及前序者的行为（即他们表示喜欢或不喜欢该文章），他们也可能付费对该文章进行事实核查。我们考察当免费但易出错的自动化工具作为昂贵却完全准确的人工事实核查的补充时，社会福利（定义为形成正确信息级联的概率）会如何变化。研究发现了两种相反的效应：更广泛地使用事实核查能帮助智能体摆脱错误的级联；与此同时，自动工具产生的错误可能误导那些原本会进行完美人工核查的智能体，从而降低形成正确级联的可能性。我们证明，当真实文章的信息量不太大时，引入该工具会严格降低中等可靠性文章的社会福利……

    arXiv:2611.08053v1 Announce Type: new  Abstract: Automated fact-checking tools, cheap and fast but imperfect, are increasingly available on social platforms. We study when such tools can backfire in a sequential social learning model of misinformation. Agents observe an article's reliability and their predecessors' actions, which indicate whether they liked or disliked the article. They may also pay to fact-check it. We examine how welfare, defined as the probability of a correct cascade, changes when a free, error-prone automated tool supplements costly but perfectly accurate manual fact-checking. Two opposing effects arise. Broader use of fact-checking helps agents escape wrong cascades. At the same time, errors from the automated tool can mislead agents who would otherwise have verified the article perfectly, reducing the likelihood of correct cascades. We prove that adding the tool strictly lowers welfare for articles of intermediate reliability when truthful articles are not too i
    
[^53]: 安全形式化培育：通过可执行属性保护人工智能的发展

    Secure Formal Raising: Protecting the Development of AI through Executable Properties

    [https://arxiv.org/abs/2611.08042](https://arxiv.org/abs/2611.08042)

    本文提出安全形式化培育（SFR）框架，用保留未知与不一致状态的四值证据建模策略完整性、授权和机密数据流，证明了六条安全不变式并通过纯C参考实现的有界穷举探索加以验证，同时发现并借助溯源要求修复了工具调用中的混淆代理人漏洞。

    

    模型培育重新审视了图灵关于机器智能的发展观。形式化培育（FR）最近被提出，以显式的行为属性来补充这一观点。本文将该提案扩展到培育环境的安全性。安全形式化培育（SFR）使用四值证据体系对策略完整性、授权和机密数据流进行建模，该证据体系能够保留“未知”与“不一致”的评估状态。我们证明了六条安全不变式，并表明这些不变式迫使准入机制只接受已确认的证据。一个纯C语言参考实现在有界穷举探索中满足这些不变式，该探索过程还暴露了工具调用中存在的“混淆代理人”漏洞，而引入溯源要求即可弥补这一漏洞。

    arXiv:2611.08042v1 Announce Type: new  Abstract: Model raising revisits Turing's developmental view of machine intelligence. Formal Raising (FR) has recently been proposed to complement this view with explicit behavioral properties. This paper extends that proposal to the security of the raising environment. Secure Formal Raising (SFR) models policy integrity, authorization, and confidential data flow using four-valued evidence that preserves unknown and inconsistent assessments. We prove six security invariants and show that they force admission to accept only confirmed evidence. A plain-C reference implementation satisfies the invariants in a bounded exhaustive exploration, which also exposes a confused-deputy gap for tool calls that a provenance requirement closes.
    
[^54]: 技术混合主义：AI中介社会中的递归语义规范化

    Technohybridism: Recursive Semantic Normalization in AI-Mediated Societies

    [https://arxiv.org/abs/2611.07984](https://arxiv.org/abs/2611.07984)

    本文提出“递归语义规范化”与“技术混合主义”两个新概念，主张应以人类实践、计算生成与制度性采纳递归耦合而成的混合语义系统为分析单位，以阐释AI生成语言如何再进入社会交流与制度实践并重塑语义规律。

    

    语言生成型AI系统不再仅仅是语言交互的对象：它们的输出正日益重新进入到其训练材料所来源的交流实践和制度实践之中。本文借鉴社会理论、社会认识论和语言哲学，对这种“再进入”现象提出了一个社会技术性的阐释。我们提出“递归语义规范化”这一概念：即机器生成的表述在被反复采纳、评估、复用并被制度性保留之后，参与到语义规律性的再生产与转变之中的过程。在“技术混合主义”的框架下，我们主张相关的分析单位是“混合语义系统”——一个由人类实践、计算生成和制度性采纳递归耦合而成的构型。首先，我们将这一概念与相邻的论述区分开来，包括模型坍塌、循环效应、机器……（原文摘要在此处截断）

    arXiv:2611.07984v1 Announce Type: cross  Abstract: Language-generating AI systems are no longer only objects of linguistic interaction: their outputs increasingly re-enter the communicative and institutional practices from which their training material was drawn. Drawing on social theory, social epistemology, and the philosophy of language, this article develops a sociotechnical account of that re-entry. We introduce recursive semantic normalization: the process through which machine-generated formulations, once recurrently taken up, evaluated, reused, and institutionally retained, participate in the reproduction and transformation of semantic regularities. Under the heading of technohybridism, we argue that the relevant unit of analysis is the hybrid semantic system, a recursively coupled configuration of human practices, computational generation, and institutional uptake. First, we distinguish the concept from neighbouring accounts, including model collapse, looping effects, machine 
    
[^55]: TCSExam：评估前沿大语言模型在理论计算机科学研究全领域的能力

    TCSExam: Evaluating Frontier LLMs on the Full Range of TCS Research

    [https://arxiv.org/abs/2611.07962](https://arxiv.org/abs/2611.07962)

    该论文提出涵盖定理证明、求解器工程、构造和开放猜想求解四个赛道、共100个专家审阅任务的TCSExam基准，首次系统评估前沿大语言模型在理论计算机科学研究全领域的能力，结果显示GPT-6 Astra在完整版（71.9分）和Opus 5.5在精简版（72.3分）上分别领先。

    

    大语言模型（LLM）能否推动理论计算机科学（TCS）的发展，并解决其长期悬而未决的开放猜想？为了研究朝着这一目标的进展，我们提出了TCSExam，一个涵盖理论计算机科学研究全领域的统一框架和基准测试。它评估定理证明、求解器工程和构造性结果，以及对解决开放猜想的直接尝试，既捕捉具体的研究成果，也衡量朝向长期研究目标的进展。TCSExam-Full包含100个经专家审阅的任务，提交的答案由Lean内核或可执行的任务检查器进行验证。TCSExam-Lite从中选取20个任务（每个赛道5个），在相同的单次尝试预算下将任务尝试次数减少了80%。我们在Full上评估了八个智能体，在Lite上评估了九个。综合所有四个赛道的TCSExam指数显示：GPT-6 Astra在Full上领先（71.9分），Opus 5.5在Lite上领先（72.3分）。在八个共同评估的智能体上，Full和Lite的排名高度一致。

    arXiv:2611.07962v1 Announce Type: new  Abstract: Can large language models (LLMs) advance theoretical computer science and resolve its long-standing open conjectures? To study progress toward this goal, we introduce TCSExam, a unified framework and benchmark covering the full range of TCS research. It evaluates theorem proving, solver engineering, and constructions alongside direct attempts to resolve open conjectures, capturing both concrete research accomplishments and progress toward long-standing research goals. TCSExam-Full contains 100 expert-reviewed tasks, with submissions checked by the Lean kernel or an executable task checker. TCSExam-Lite selects 20 tasks, five per track, reducing the number of task attempts by 80% at the same per-attempt budget. We evaluate eight agents on Full and nine on Lite. The TCSExam Index combines all four tracks: GPT-6 Astra leads Full (71.9), and Opus 5.5 leads Lite (72.3). Full and Lite rankings are strongly aligned across eight shared agents (S
    
[^56]: 一个用于预测和理解前沿人工智能系统失控的行为框架

    A Behavioural Framework for Predicting and Understanding Loss of Control in Frontier Artificial Intelligence Systems

    [https://arxiv.org/abs/2611.07953](https://arxiv.org/abs/2611.07953)

    本文提出一个行为框架，将前沿AI系统的失控风险分解为动机错位、危害促成能力和监控规避三个因素，并验证了其失控评分在13个前沿模型上对实际失败率的前瞻预测能力（Pearson r = 0.84），且表明风险无法仅由模型能力解释。

    

    前沿人工智能系统越来越多地在更长的时间跨度上运行，而人类监督却在逐渐减少，这带来了失控的风险：即系统采取超出人类有效控制的行为，并可能造成不可逆转的后果。由于此类故障可能只有在难以遏制时才会显现出来，因此必须对其进行前瞻性评估。我们提出了一个行为框架，将失控风险分解为三个因素：动机错位、促成危害的能力以及规避监控。我们通过13个可测量的方面和29个评估任务对这些因素进行操作化，其得分被综合为一个单一的失控（LoC）评分。在13个前沿模型上，失控评分表现出很强的前瞻有效性，与受控长时程风险场景中的失败率相关（Pearson r = 0.84）。风险在各模型间呈现系统性变化，且无法仅用能力来解释。

    arXiv:2611.07953v1 Announce Type: new  Abstract: Frontier artificial intelligence (AI) systems are increasingly operating over longer horizons with diminishing human oversight, raising the risk of loss of control: systems acting beyond effective human control with potentially irreversible consequences. Because such failures may become observable only after containment is difficult, they must be assessed prospectively. We introduce a behavioural framework that decomposes loss-of-control risk into three factors: misaligned motive, harm-enabling capability, and monitoring evasion. We operationalise these factors through 13 measurable aspects and 29 evaluation tasks, whose scores are combined into a single loss-of-control (LoC) score. Across 13 frontier models, the LoC score shows strong prospective validity, correlating with failure rates in contained long-horizon risk scenarios (Pearson r = 0.84). Risk varies systematically across models and cannot be explained by capability alone. We fu
    
[^57]: Circe：基于组件交互一致性的免训练大语言模型自我评估方法

    Circe: Training-Free LLM Self-Evaluation via Component Interaction Consistency

    [https://arxiv.org/abs/2611.07948](https://arxiv.org/abs/2611.07948)

    提出Circe——一种免训练的大语言模型自我评估方法，通过分析注意力与前馈网络等组件间的交互一致性并结合内外部信号来评估回答正确性，无需参考答案或重复采样。

    

    随着大语言模型在实际应用中的部署日益增多，评估其回答的正确性对于可靠使用至关重要。然而，这种评估具有挑战性，因为在实践中参考答案往往难以获得。现有的免参考方法主要依赖输出信号，这可能会给错误的回答赋予高分，因此并不一定能反映正确性。这促使我们探索模型内部计算以获取互补的正确性信号。近期研究利用了内部状态，但它们主要分析跨层的隐藏状态轨迹，这仅记录了注意力机制和前馈网络（FFN）的组合更新，而忽视了这些组件之间如何交互。因此，我们提出了Circe，一种免训练的自我评估方法，它结合内部和外部信号来评估回答，无需参考答案或重复采样。对于内部

    arXiv:2611.07948v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly deployed in real-world applications, evaluating the correctness of their responses is crucial for reliable use. However, such evaluation is challenging because reference answers are rarely available in practice. Existing reference-free methods mainly rely on output signals, which may assign high scores to incorrect responses and thus do not necessarily reflect correctness. This motivates exploring the model's internal computation for complementary correctness signals. Recent studies exploit internal states, but they mainly analyze hidden-state trajectories across layers, which record only the combined updates of attention and feed-forward networks (FFNs) and overlook how these components interact. We therefore propose Circe, a training-free self-evaluation method that assesses a response using both internal and external signals, without reference answers or repeated sampling. For internal 
    
[^58]: ChemE-as-Code：面向化学工程智能体AI的开源平台

    ChemE-as-Code: An Open-Source Platform for Agentic AI in Chemical Engineering

    [https://arxiv.org/abs/2611.07947](https://arxiv.org/abs/2611.07947)

    该论文提出了开源平台ChemE-as-Code，让基于LLM的AI智能体在可追溯、可版本管理的环境中完成从催化到过程模拟、HAZOP等化学工程设计与计算工作，并通过自动化检查引导而非取代人工审查。

    

    ChemE-as-Code提供了一个开源环境，基于大语言模型（LLM）构建的AI智能体可以在其中开展化学工程工作，而工程师能够对其进行核查。其计算工具涵盖从催化到过程模拟，再到工厂布置、危险与可操作性分析（HAZOP）以及生命周期评估，并通过“技能”告知通用智能体如何使用这些工具。工程文件相互关联、可被机器读取并在Git中进行版本管理；计算在LLM之外运行，每个数值都记录其来源，自动化检查能够按文件和行报告不一致及过期的结果。在一个示例性的合成氨案例中，智能体在若干次设计会话中，以224美元的API成本，将一项书面任务发展为初步的基础工程设计包（包括模拟文件、管道仪表流程图、布置图、HAZOP、数据表和泄放计算）。植入的故障和工程变更表明，这些检查能发现其规则所覆盖的问题，但会遗漏其余问题，因此它们的作用是引导人工审查，而非取代人工审查。

    arXiv:2611.07947v1 Announce Type: new  Abstract: ChemE-as-Code provides an open-source environment in which AI agents built on large language models (LLMs) can carry out chemical engineering work and engineers can check it. Its calculation tools range from catalysis through process simulation to plant layout, HAZOP and life cycle assessment, and skills tell a general-purpose agent how to use them. Engineering files are linked, machine-readable and versioned in Git; calculations run outside the LLM, each value records its origin, and automated checks report inconsistencies and out-of-date results by file and line. In an illustrative ammonia case, agents developed a written task into a preliminary basic-engineering package (simulation files, P&IDs, layout, HAZOP, datasheets, relief) for USD 224 of API cost in the design sessions. Planted faults and engineering changes show that the checks find what their rules cover and miss the rest, so they guide human review rather than replace it. No
    
[^59]: tok：面向人类和AI智能体的BPMN与DMN文本具体语法

    tok: a textual concrete syntax of BPMN and DMN for humans and AI agents

    [https://arxiv.org/abs/2611.07945](https://arxiv.org/abs/2611.07945)

    本文提出了tok这一BPMN和DMN的纯文本表示法，其令牌效率比XML高数倍、故障检测率更高，且语言模型能轻松学习并生成有效的tok模型。

    

    tok是一种用于业务流程模型与表示法（BPMN）2.0.2流程模型和决策模型与表示法（DMN）1.5决策的纯文本表示法。一个引擎可以依据这两项标准对tok文档进行检查，将文档编译为带有自动生成图表的XML，并通过模型上下文协议为语言模型应用提供服务。该论文将这种表示法归类为BPMN和DMN元模型的另一种具体语法，并报告了一项研究。tok模型所消耗的令牌数量比37种工具导出的893个文件中最小的BPMN XML少2.2倍，比文件按原样导出时少7.7倍。该检查器能够识别1,411个注入故障中的98.2%，而带有bpmnlint的XML Schema只能识别85.8%。四个此前未接触过tok的语言模型仅凭一页参考文档，就能为来自两个数据集的87个流程描述中的85.7%至100%编写出有效的tok，并为每个描述编写出有效的BPMN XML，其中最强的模型编写的tok比标准XML少25%和19%的tok

    arXiv:2611.07945v1 Announce Type: cross  Abstract: tok is a plain-text notation for Business Process Model and Notation (BPMN) 2.0.2 process models and Decision Model and Notation (DMN) 1.5 decisions. One engine checks a tok document against both standards, compiles the document to XML with an automatic diagram, and serves language-model applications through the Model Context Protocol. The paper classifies the notation as an alternative concrete syntax of the BPMN and DMN metamodels and reports one study. A tok model takes 2.2 times fewer tokens than the smallest BPMN XML of 893 files exported by 37 tools and 7.7 times fewer than the files as exported. The checker signals 98.2% of 1,411 injected faults, against 85.8% for the XML Schema with bpmnlint. Four language models new to tok write valid tok from a one-page reference for 85.7 to 100% of 87 process descriptions from two datasets and valid BPMN XML for every description, and the strongest model writes tok with 25% and 19% fewer tok
    
[^60]: CardioLM：面向心脏病学领域的多语言小型语言模型套件

    CardioLM: A multilingual suite of small language models for the cardiology domain

    [https://arxiv.org/abs/2611.07940](https://arxiv.org/abs/2611.07940)

    该研究提出了多语言心脏病学病例语料库 CardioCCC 以及小型语言模型套件 CardioLM（含编码器 CardioBERTa 和命名实体识别模型 CardioNER），填补了心脏病学领域多语言临床 NLP 资源的空白。

    

    背景：心血管诊疗会在本地语言中产生大量临床自由文本文档。然而，临床自然语言处理（NLP）资源仍集中于英语，且目前尚无心脏病学专用、多语言且实体覆盖广泛的基准。方法：我们提出了两项贡献。第一项是 CardioCCC，一个包含 508 份心脏病学病例报告的语料库，这些报告最初以西班牙语发表，并通过机器翻译、标注投影和临床医生审核扩展至捷克语、荷兰语、英语、意大利语、罗马尼亚语和瑞典语，其标注涵盖疾病、症状、操作和药物。第二项是 CardioLM，包括 CardioBERTa——一组通过对心脏病学和生物医学语料库进行持续预训练而得到的特定语言编码器，以及 CardioNER——一组在 CardioCCC 上微调的命名实体识别（NER）模型。结果：心脏病学专用预训练降低了……

    arXiv:2611.07940v1 Announce Type: cross  Abstract: Background: Cardiovascular care generates large volumes of clinical free-text documentation in local languages. However, clinical natural language processing (NLP) resources remain concentrated in English, and no existing benchmark is cardiology-specific, multilingual, and broad in entity coverage.   Methods: We present two contributions. The first is CardioCCC, a corpus of 508 cardiology case reports originally published in Spanish and extended to Czech, Dutch, English, Italian, Romanian, and Swedish through machine translation, annotation projection, and clinician review, with annotations covering diseases, symptoms, procedures, and medications. The second is CardioLM, comprising CardioBERTa, a set of language-specific encoders obtained by continued pretraining on cardiology and biomedical corpora, and CardioNER, a set of named entity recognition (NER) models fine-tuned on CardioCCC.   Results: Cardiology-specific pretraining reduced
    
[^61]: 面向病理档案的可扩展诊断驱动全切片图像检索

    Towards Scalable Diagnostic-Driven Whole-Slide Image Retrieval for Pathology Archives

    [https://arxiv.org/abs/2611.07935](https://arxiv.org/abs/2611.07935)

    提出了一种可扩展、诊断驱动的全切片图像检索框架CLEAR-WSI，通过固定长度的切片嵌入并利用诊断标签细化检索候选，摆脱了手工索引流水线，在公开和临床病理数据上实现了高效的相似病例检索。

    

    数字病理学的日益普及使苏木精-伊红（H&E）染色的全切片图像集合迅速扩大，然而由于元数据往往不一致或缺失，如何高效地浏览这些档案仍然具有挑战性。基于内容的图像检索通过直接从视觉表示中识别相似病例，而非依赖人工标注或元数据，来应对这一挑战。现有检索系统已展现出良好的性能，但大多依赖手工设计的索引流水线和预定义的采样策略，限制了其可扩展性和泛化能力。本文提出CLEAR-WSI（Constant-Length Embedding for Automatic Retrieval of Whole-Slide Images，用于全切片图像自动检索的定长嵌入），作为一种可扩展的、诊断驱动的切片检索框架，并在公开数据和临床数据上进行了验证。CLEAR-WSI采用固定长度的切片嵌入，并通过预测的诊断标签或真实标签对检索到的候选结果进行细化……

    arXiv:2611.07935v1 Announce Type: cross  Abstract: The growing adoption of digital pathology has rapidly expanded collections of hematoxylin-and-eosin whole-slide images, yet efficient navigation through these archives remains challenging, as metadata is often inconsistent or unavailable. Content-based image retrieval addresses this challenge by identifying similar cases directly from visual representations instead of manual annotations or metadata. Existing retrieval systems have demonstrated promising performance but largely rely on handcrafted indexing pipelines and predefined sampling strategies, constraining scalability and generalization. Here, we present CLEAR-WSI (Constant-Length Embedding for Automatic Retrieval of Whole-Slide Images) as a scalable, diagnostic-driven slide-retrieval framework, validated on both public and clinical data. Using a fixed-length slide embedding, CLEAR-WSI refines retrieved candidates with either a predicted diagnostic label or the ground-truth labe
    
[^62]: AttnSR：面向长程智能体强化学习的注意力步级奖励

    AttnSR: Attention Step Reward for Long-Horizon Agentic RL

    [https://arxiv.org/abs/2611.07932](https://arxiv.org/abs/2611.07932)

    提出 AttnSR 步级奖励增强方法，利用训练回放中的策略注意力反向追踪为各步骤分配奖励，无需额外奖励模型，即在 ALFWorld 上将 Qwen2.5-7B 的成功率从 91.2% 提升至 97.1%。

    

    长程LLM智能体需要进行多步决策，然而仅使用结果奖励的强化学习难以对各个步骤提供直接的监督信号。我们提出了AttnSR，这是一种步级奖励增强方法，它从训练回放中提取的策略注意力推导奖励分配，无需单独训练的奖励模型，也无需额外的 rollout。AttnSR 通过在策略模型的注意力图中反向追踪，来估计先前步骤与最终响应的相关性。终端步骤相对于运行基线进行度量，其结果决定这些奖励是强化还是惩罚所选择的步骤。在受控的 Qwen2.5-7B-Instruct 实验中，AttnSR 在 ALFWorld 上所有评估检查点均持续优于 PPO 基线，将最终留出集成功率从 91.2% 提升至 97.1%。消融实验表明，其收益取决于中间奖励的分配方式，而非简单的累加。

    arXiv:2611.07932v1 Announce Type: cross  Abstract: Long-horizon LLM agents make multi-step decisions, yet outcome-reward-only reinforcement learning provides little direct supervision for individual steps. We introduce AttnSR, a step-level reward augmentation method that derives reward allocations from policy attention extracted during training replay, without a separately trained reward model or additional rollouts. AttnSR traces backward through the policy model attention graph to estimate the relevance of earlier steps to the final response. The terminal step, measured relative to a running baseline, determines whether these rewards reinforce or penalize the selected steps. In a controlled Qwen2.5-7B-Instruct study, AttnSR consistently outperforms the PPO baseline across all evaluated checkpoints on ALFWorld, raising final held-out success from 91.2% to 97.1%. The ablation experiment shows that the benefit depends on how intermediate rewards are assigned, rather than simple accumula
    
[^63]: ZeroRAG：面向去中心化检索增强生成的自适应语义对等路由

    ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.07928](https://arxiv.org/abs/2611.07928)

    ZeroRAG提出了一种无需全局数据结构和集中式编排的去中心化RAG文档检索方案，通过在对等覆盖网络上进行自适应语义查询路由，使文档始终保留在原始所有者手中。

    

    检索增强生成（RAG）是一种通过检索与查询相关的文档并将其注入大语言模型（LLM）提示中来提升LLM输出可靠性的标准方法。如今的RAG系统从集中式存储中收集、索引并提供这些文档。然而在实践中，文档的所有者是组织、设备和个人，由于隐私、版权或监管合规等原因，他们不能或不愿将文档的托管权移交给中央机构。现有的现代去中心化检索技术要么重新引入协调机构，要么依赖需要全网持续协调的全局数据结构（如分布式哈希表DHT）。我们提出了ZeroRAG，这是一个面向RAG文档检索的去中心化路由层，它在由对等节点构成的覆盖网络上执行语义查询路由，无需全局数据结构或集中式编排。文档仍保留在其所属的对等节点上。（摘要在此处被截断）

    arXiv:2611.07928v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) is a standard approach to enhance the reliability of large language model (LLM) outputs by retrieving documents relevant to a query and supplying them in the LLM prompt. Today's RAG systems collect, index, and serve these documents from centralized stores. In practice, however, documents are owned by organizations, devices, and individuals who cannot or will not surrender custody of them to a central party, e.g., because of privacy, copyright or regulatory compliance. Existing modern decentralized retrieval techniques either reintroduce a coordinating authority, or rely on a global data structure that requires continuous network-wide coordination (e.g., DHTs). We present ZeroRAG, a decentralized routing layer for document retrieval in RAG that performs semantic query routing across an overlay of peers with- out global data structures or centralized orchestration. Documents remain with the peers that
    
[^64]: 有限的大脑如何实现无界泛化：作为通用计算机的空间记忆

    How Finite Brains Generalize Without Bound: Spatial Memory as a Universal Computer

    [https://arxiv.org/abs/2611.07920](https://arxiv.org/abs/2611.07920)

    论文揭示了有限大脑能无界泛化的关键机制：学习者只应读取“巧合”（等价性判断）而非“坐标”（位置、计数等随规模变化的量），由此推导出可泛化的大脑必须拥有一种由循环构成、随输入增长却始终保持有限读取、并可充当通用计算机的空间记忆。

    

    一个学会了小数字加法的孩子能够对任意长度的数字做加法，而在短序列上训练的LSTM和Transformer却在更长的序列上失败。我们将这一差异追溯到学习者在做决策时所“读取”的内容。位置、计数或时间会随着输入的增长不断取新的值：它是一个坐标，读取坐标的学习者会在每个长输入中遇到其经验从未覆盖过的值。而“两个事物是否相同”这一判断在任意规模下都只有相同的两种答案：它是一个巧合，只读取巧合的学习者在长输入中找不到任何短输入所缺少的东西，因此有限的经验就能确定其适用于任意规模的规则。记忆承载着输入的规模而规则保持有限，这决定了一个能够泛化的大脑所能拥有的记忆形态：一种在不断增长的同时、学习者对它的读取始终保持有限的记忆，必须从任何位置看都相同，因此它由循环构成，而通往过去的方式则是通过询问“当……发生时发生了什么”。（注：原文摘要在此处截断）

    arXiv:2611.07920v1 Announce Type: new  Abstract: A child who learns to add small numbers adds numbers of any length. LSTMs and Transformers trained on short sequences fail on longer ones. We trace the difference to what a learner reads when it decides. A position, a count or a time keeps taking new values as the input grows: it is a coordinate, and a learner that reads one meets in every long input values its experience never covered. Whether two things are the same has the same two answers at every size: it is a coincidence, and a learner that reads only coincidences finds in a long input nothing that short inputs lacked, so finite experience fixes its rule for every size. The memory carries the size of the input and the rule stays finite, and this determines the memory a generalizing brain can have: a memory that grows while what the learner reads of it stays finite has to look the same from every place, so it is made of cycles, and the past is reached by asking what happened when th
    
[^65]: 自适应语义分支：面向推理候选扩展的分歧定位跨轨迹再生成

    Adaptive Semantic Branching: Disagreement-Localized Cross-Trajectory Regeneration for Reasoning Candidate Expansion

    [https://arxiv.org/abs/2611.07918](https://arxiv.org/abs/2611.07918)

    该论文提出了无需训练的自适应语义分支（ASB）方法，通过定位不同推理轨迹间的分歧点并融合分歧处的推理内容来再生成新的候选解法，从而有效扩展推理候选的多样性与质量。

    

    为同一问题生成的推理轨迹往往会得出相互冲突的答案。这些分歧揭示了可被复用以探索新解法的不同中间决策。我们提出了自适应语义分支（ASB），这是一种无需训练的推理流程，能够将跨轨迹分歧转化为有针对性的候选生成。ASB 首先根据提取出的最终答案对独立采样的轨迹进行分组，从每组中选出一个代表性轨迹，并使用单调动态规划对它们的推理步骤进行对齐。一个持续性分歧目标用于识别最初相似的推理路径在何处发展出不同的延续。对于每个选定的有向轨迹对，ASB 保留源轨迹的文本前缀，从竞争性供体轨迹中提取一个简短的推理窗口，并基于两者条件生成全新的解法。一种多样性感知策略会分配最多三次额外的生成机会。

    arXiv:2611.07918v1 Announce Type: new  Abstract: Reasoning trajectories generated for the same problem often arrive at conflicting answers. These disagreements reveal alternative intermediate decisions that can be reused to explore new solutions. We introduce Adaptive Semantic Branching (ASB), a training-free inference procedure that turns cross-trajectory disagreement into targeted candidate generation. ASB first groups independently sampled trajectories by extracted final answer, selects a representative from each group, and aligns their reasoning steps with a monotonic dynamic program. A persistent-divergence objective identifies the point at which initially similar reasoning paths develop different continuations. For each selected directed pair, ASB retains the source trajectory's textual prefix, extracts a short reasoning window from a competing donor trajectory, and generates a fresh solution conditioned on both. A diversity-aware policy allocates up to three additional generatio
    
[^66]: GuidedWAM：面向自动驾驶世界动作模型的无训练能量引导

    GuidedWAM: Training-Free Energy Guidance for World Action Models in Autonomous Driving

    [https://arxiv.org/abs/2611.07912](https://arxiv.org/abs/2611.07912)

    提出GuidedWAM，一种无需重新训练的与模型无关的能量引导框架，可在采样时将新驾驶目标注入冻结的自动驾驶世界动作模型，在NAVSIM v2难例集上平均提升1.8至3.6 EPDMS，且每步采样开销不足10%。

    

    面向自动驾驶的世界动作模型（WAMs）将驾驶策略与视觉世界模型耦合在单一的生成网络中，其优化目标在训练完成后即被固定，若要针对新的驾驶目标进行条件化控制则必须重新训练。我们研究能量引导作为一种通用机制，并提出GuidedWAM——一个与模型无关的框架，能够在采样阶段将驾驶目标注入到冻结的WAM中。我们的框架将每个目标构造为作用于采样器中间干净动作估计上的可微能量，这只需要对冻结模型进行前向访问即可，并通过采样时的引导调度、逐能量归一化以及速度受限校正来保证稳定性。在NAVSIM v2难例分割上，该引导方法使三个冻结的已公开发布WAM在三个采样器种子下的平均EPDMS提升1.8至3.6分，且每步采样时间增加不足10%。这些结果表明，能量引导可作为一种实用的后训练机制。

    arXiv:2611.07912v1 Announce Type: new  Abstract: World action models (WAMs) for autonomous driving couple the driving policy and the visual world model in a single generative network whose objectives are fixed post-training. Conditioning on a new driving objective requires retraining. We study energy guidance as a general mechanism and present GuidedWAM, a model-agnostic framework for injecting driving objectives into frozen WAMs at sampling time. Our framework constructs each objective as differentiable energies on the sampler's intermediate clean-action estimate, which needs only forward access to the frozen model, and is stabilized by sampling-time guidance scheduling, per-energy normalization, and velocity-bounded correction. On the NAVSIM v2 hard-split, guidance improves all three frozen, publicly released WAMs by 1.8 to 3.6 EPDMS, averaged over three sampler seeds, and adds less than 10% to the sampling time per step. These results position energy guidance as a practical post-tra
    
[^67]: 基于假设空间推理的碰撞结构恢复

    Structure from Collision via Hypothesis-Space Reasoning

    [https://arxiv.org/abs/2611.07900](https://arxiv.org/abs/2611.07900)

    提出通过假设空间推理从碰撞事件的多视角观测中恢复物体隐藏内部结构（如空腔），克服了空腔仅能通过表面运动间接观测、且因对应材料缺失而难以用密度表示直接优化的难题。

    

    了解物体的隐藏内部结构，包括空腔的存在与位置，对于预测性仿真、机器人操作和无损检测至关重要。然而，这类空腔在静态可见表面上不会留下任何直接痕迹，使得碰撞引起的变形成为推断其存在的少数线索之一。Structure from Collision（SfC）通过从碰撞事件的多视角观测中恢复物体的隐藏内部结构来解决这一问题。SfC 面临两大挑战：隐藏空腔只能通过表面运动间接观测，且空腔对应于材料的缺失，这使得使用标准密度表示进行直接优化变得困难。基于密度的 SfC 基线同时继承了这两个困难：表面匹配梯度无法直接指示空腔位置，而密度梯度在物体内部可能微弱或缺乏信息。

    arXiv:2611.07900v1 Announce Type: cross  Abstract: Knowledge of the hidden interior structure of an object, including the presence and location of cavities, is essential for predictive simulation, robotic manipulation, and nondestructive inspection. However, such cavities leave no direct signature on the static visible surface, making collision-induced deformation one of the few cues used to infer their presence. Structure from Collision (SfC) addresses this problem by recovering the hidden interior structure of an object from multiview observations of a collision event. SfC is challenging for two reasons: hidden cavities are only indirectly observable through surface motion, and voids correspond to the absence of material, making direct optimization using standard density representations difficult. The density-based SfC baseline inherits both difficulties. Surface-matching gradients do not directly indicate the cavity location, while density gradients can be weak or uninformative insi
    
[^68]: 面向对话代理的持久化检索：检索效率研究

    Persistent Retrieval for Conversational Agents: A Study of Retrieval Efficiency

    [https://arxiv.org/abs/2611.07899](https://arxiv.org/abs/2611.07899)

    提出一种跨轮次维护检索历史并将检索证据直接返回主代理的持久化检索子代理设计，相比基于技能的检索可减少43.3%的token用量和45.0%的预估API成本。

    

    对话代理可以直接检索信息，也可以将检索任务委托给子代理。这一设计选择会影响检索历史的维护方式、每次模型调用所接收的上下文，以及在后续问题中需要重新检索哪些信息。我们比较了三种设计：主代理中基于技能（skill-based）的检索、生成调查结果回复的非持久化检索子代理，以及我们提出的持久化检索子代理——该子代理跨轮次维护检索历史，并将检索到的证据直接返回给主代理。我们的方法还允许主代理向同一个子代理请求修正。我们在独立提出的企业问题、共享对话中的问题以及依赖性后续问题上对这三种设计进行了评估。在五个多轮设置中，与基于技能的检索相比，我们的方法记录的代理-模型token用量减少43.3%，预估API成本降低45.0%，同时降低了平均

    arXiv:2611.07899v1 Announce Type: new  Abstract: Conversational agents can retrieve information directly or delegate retrieval to a subagent. This design choice affects how retrieval history is maintained, what context each model call receives, and what information must be re-retrieved in follow-up questions. We compare three designs: skill-based retrieval in the main agent, a non-persistent retrieval subagent that generates a findings response, and our persistent retrieval subagent, which maintains retrieval history across turns and returns retrieved evidence directly to the main agent. Our approach also allows the main agent to request corrections from the same subagent.   We evaluate the three designs on enterprise questions posed independently, within shared conversations, and as dependent follow-ups. Across five multi-turn settings, our approach uses 43.3% fewer recorded agent-model tokens and incurs 45.0% lower estimated API cost than skill-based retrieval, while reducing average
    
[^69]: DISCERN：面向无保留集增量式机器遗忘的分布性合成回放

    DISCERN: Distributional Synthetic Replay for Retain-Free Incremental Machine Unlearning

    [https://arxiv.org/abs/2611.07892](https://arxiv.org/abs/2611.07892)

    DISCERN 通过为每个类别拟合欧氏与方向性参数分布并采样合成嵌入进行回放，实现了无需访问保留集的增量式类遗忘，在依次删除类别的同时确保已遗忘知识不被恢复并保留剩余类别的性能。

    

    arXiv:2611.07892v1 公告类型： cross 摘要：机器遗忘旨在在不从头重新训练的情况下，从已训练模型中移除特定训练数据的影响。然而在实际部署中，删除请求会随时间不断到来，这就需要增量式类遗忘（Incremental Class Unlearning, ICU）：类别被依次移除，已被遗忘的知识必须保持遗忘状态，同时剩余类别必须得到保留。我们提出了 DISCERN（DIstributional SynthetiC rEplay for Retain-free iNcremental unlearning），一种基于干预的类遗忘方法，它用从每个类别的参数化分布中采样的合成嵌入来替代对保留集的访问。与假设每个类质心周围为各向同性高斯分布的原型回放方法不同，DISCERN 为每个类别拟合欧几里得分布族（高斯、拉普拉斯、学生 t 分布）和方向性分布族（von Mises–Fisher、Fisher–Bingham），并通过统计检验从中进行选择，当没有任何分布族被接受时，则采用交叉验证……（摘要在此处截断）

    arXiv:2611.07892v1 Announce Type: cross  Abstract: Machine unlearning aims to remove the influence of specific training data from a trained model without retraining from scratch. In real deployments, however, deletion requests arrive over time, calling for incremental class unlearning (ICU): classes are removed sequentially, previously forgotten knowledge must remain forgotten, and the remaining classes must be preserved. We present DIstributional SynthetiC rEplay for Retain-free iNcremental unlearning (DISCERN), an intervention-based class unlearning method that replaces access to the retain set with synthetic embeddings sampled from per-class parametric distributions. Unlike prototype-based replay, which assumes an isotropic Gaussian around each class centroid, DISCERN fits Euclidean (Gaussian, Laplace, Student's $t$) and directional (von Mises--Fisher, Fisher--Bingham) families to each class and selects among them through statistical tests and, when none of them accepts, cross-valid
    
[^70]: 当变大变得脆弱：为什么语言模型规模越大，压缩带来的伤害越严重

    When Bigger Becomes Brittle: Why Compression Hurts More as Language Models Scale

    [https://arxiv.org/abs/2611.07891](https://arxiv.org/abs/2611.07891)

    该论文提出一种精确的分解方法，将压缩（量化/剪枝）造成的损失拆分为“网络其余部分的补偿恢复”与“真正传递到输出的残余损失”两部分，并发现随着语言模型规模增大，网络内部的补偿能力减弱，因此压缩对大模型的伤害反而更严重。

    

    量化与剪枝可以降低语言模型的部署成本，但它们的效果通常以基准测试分数来报告。这类分数只记录了输出端剩余的损失。而在模型内部，压缩对某些组件的破坏程度大于其他组件，网络的其余部分可以部分补偿这种破坏。因此，观察到的损失同时反映了破坏本身和这种补偿，而基准测试分数无法将两者区分开来。这两者如何共同作用，以及它们的平衡关系是否随模型规模而变化，目前仍不清楚。我们针对上下文检索能力研究了这种补偿机制，该能力由一小组可识别的注意力头承载。为了将破坏与补偿分离，我们精确地将消融这些注意力头所产生的预期损失分解为两个部分：一部分由网络其余部分恢复（并归因于负责补偿的具体组件），另一部分则真正传递到输出端。随后，我们将这一分解在不同模型规模之间进行比较。对于规模相当的破坏，较小的模型……（原文摘要在此处截断）

    arXiv:2611.07891v1 Announce Type: new  Abstract: Quantization and pruning reduce the cost of deploying language models, but their effects are usually reported as benchmark scores. Such scores record only the loss that remains at the output. Inside the model, compression damages some components more than others, and the rest of the network can partly compensate. The observed loss therefore reflects both the damage and this compensation, which benchmark scores cannot separate. How the two combine, and whether their balance changes with model size, remains unclear. We study this compensation for in-context retrieval, a capability carried by a small, identifiable set of attention heads. To separate the two, the expected loss from ablating these heads is decomposed exactly into the part restored by the rest of the network, attributed to the components responsible, and the part that reaches the output. This decomposition is then compared across model sizes. For damage of comparable size, sma
    
[^71]: 何时、何地、如何：面向表格基础模型的主动上下文学习

    When, Where, How: Active In-Context Learning for Tabular Foundation Models

    [https://arxiv.org/abs/2611.07886](https://arxiv.org/abs/2611.07886)

    本文针对表格基础模型提出主动上下文学习方法，研究了目标感知最近邻（TANN）采集和上下文期望预测信息增益（EPIC）两种策略，并在二十个基准上发现基于多样性的采集在初期迭代中最为有效。

    

    表格基础模型（TFMs）仅通过其上下文中的带标注行来适应新任务。然而，在现实场景中，标签的获取成本高昂，因为需要高度专业的领域知识来确保数据质量。主动上下文学习将经典的采集规则迁移到这一设定中，其中引导采集的模型会随着每个标签的添加而发生变化。我们探究了随着冻结的TFM获取更多上下文，多样性、预测不确定性和目标感知信息增益的价值如何变化。为此，我们研究了目标感知最近邻（TANN）采集方法和上下文中的期望预测信息增益（EPIC），并将它们与经典规则进行评估比较。我们在二十个基准数据集上，使用两个成熟的TFM和经典GBDT开展研究，得出了三个发现：(i) 何时：基于多样性的采集在最初的采集迭代中效果最佳；(ii) 何地：标注池中的行……

    arXiv:2611.07886v1 Announce Type: cross  Abstract: Tabular Foundation Models (TFMs) adapt to a new task only through the labeled rows in their context. However, labels are expensive to acquire in real-world scenarios due to the need for highly specific domain knowledge to ensure high-quality data. Active in-context learning transfers classical acquisition rules to this setting, in which the model that guides acquisition changes with every label added. We ask how the value of diversity, predictive uncertainty and target-aware information gain changes as a frozen TFM acquires more context. For that, we study Target-Aware Nearest Neighbor (TANN) acquisition and the Expected Predictive Information Gain in Context (EPIC) and evaluate them against classical rules. Our study across twenty benchmarks with two well-established TFMs and classical GBDTs highlights three findings: (i) When: diversity-based acquisition is best during the first acquisition iterations; (ii) Where: labeling pool rows 
    
[^72]: InnerLogic-Surg：通过持续预训练构建的开放式手术基础模型

    InnerLogic-Surg: An Open Surgical Foundation Model via Continual Pretraining

    [https://arxiv.org/abs/2611.07883](https://arxiv.org/abs/2611.07883)

    本文提出InnerLogic-Surg，一个基于DINOv3持续预训练、利用来自11,965个手术视频约2,900小时的数百万张图像训练而成的开放式手术视觉基础模型家族（21M-300M参数），可显著增强手术图像分析任务的性能。

    

    视觉基础模型是现代手术图像分析的支柱，然而大多数手术系统仍然依赖于在自然图像上预训练的编码器，或者依赖于在相对较小的数据集上从头训练的专用模型。在大量手术特定数据上预训练编码器，有可能仅通过提供更强的骨干网络就能普遍提升手术图像分析方法的性能。在此，我们提出InnerLogic-Surg，一个开放的手术视觉编码器家族（参数量从2100万到3亿），它通过对基于Transformer的图像编码器进行持续预训练而获得，训练数据是从11,965个互联网来源的手术视频中精选的数百万张手术图像，涵盖约2,900小时、超过十五种手术类型。我们的训练方案继承了DINOv3的预训练阶段，以便在手术领域适配后保持其表征能力，并额外应用了一个可学习的特征上采样模块，用于高分辨率密集预测任务。

    arXiv:2611.07883v1 Announce Type: cross  Abstract: Vision foundation models are the backbone of modern surgical image analysis, yet most surgical systems still rely on encoders pretrained on natural images or on specialist models trained from scratch on comparatively small datasets. Pretraining encoders on large amounts of surgery-specific data has the potential to generally enhance the performance of surgical image analysis methods simply by providing stronger backbones. Here we present InnerLogic-Surg, an open family of surgical vision encoders (21M--300M parameters) obtained by continually pretraining a transformer-based image encoder on millions of surgical images curated from 11,965 internet-sourced surgical videos spanning ~2,900 hours and more than fifteen procedure types. Our training recipe inherits DINOv3's pretraining stage to preserve representation ability after surgical adaptation, and additionally applies a learned feature-upsampling module for high-resolution dense pred
    
[^73]: 网络效应分解：用于网络干扰下的因果效应估计

    Network Effect Decomposition for Causal Effect Estimation under Network Interference

    [https://arxiv.org/abs/2611.07879](https://arxiv.org/abs/2611.07879)

    提出NEDNet网络，将网络干扰下的结果响应分解为查询无关的处理前基线以及相互独立的处理响应通路和暴露响应通路，从而在观测网络数据上实现更准确的因果效应估计。

    

    从观测网络数据中估计因果效应，需要同时考虑个体的处理和邻域暴露。现有的结果模型通常将这些干预变量合并到一个联合预测器中，并通过预测对比来估计因果效应，而相应的响应成分则是隐式的。我们提出了网络效应分解网络（NEDNet），它将条件结果响应参数化为一个与查询无关的处理前基线，加上相互独立的处理响应通路和暴露响应通路。路径特定的邻域上下文化机制为这两个通路提供局部的处理前信息。我们通过一个唯一的、参考归一化的分解来刻画拟合的响应，该分解等价于一个零混合处理-暴露对比。在基于BlogCatalog等数据集构建的半合成基准上的实验（原文在此处截断）……

    arXiv:2611.07879v1 Announce Type: cross  Abstract: Estimating causal effects from observational network data requires accounting for both individual treatment and neighborhood exposure. Existing outcome models commonly incorporate these intervention variables into a joint predictor and estimate causal effects through prediction contrasts, leaving the corresponding response components implicit. We propose the $\underline{N}$etwork $\underline{E}$ffect $\underline{D}$ecomposition $\underline{Net}$work (NEDNet), which parameterizes the conditional outcome response as a query-invariant pre-treatment baseline plus separate treatment- and exposure-response pathways. Path-specific neighborhood contextualization supplies local pre-treatment information to the two pathways. We characterize the fitted response through a unique reference-normalized decomposition, which is equivalent to a zero mixed treatment--exposure contrast. Experiments on semi-synthetic benchmarks derived from BlogCatalog and
    
[^74]: 活跃武装冲突期间负责任的生成式AI使用：来自苏丹应急响应室的证据

    Responsible GenAI Use During Active Armed Conflict: Evidence from Sudan's Emergency Response Rooms

    [https://arxiv.org/abs/2611.07874](https://arxiv.org/abs/2611.07874)

    该研究通过对苏丹应急响应室社区救援者的调查与访谈，首次揭示了武装冲突环境下人道主义工作者对生成式AI的使用情况，发现战争使GenAI在输出最难验证的场景中反而最有价值，但在缺乏人工验证时依赖其输出可能导致有害决策。

    

    武装冲突可能提升生成式AI（GenAI）的价值，同时削弱其负责任使用所需的条件。然而，在这些条件下工作的社区人道主义响应者的观点在负责任AI研究中仍然代表性不足。我们通过对89名受访者的探索性调查以及对GenAI使用者和非使用者进行的15次半结构化访谈，考察了GenAI在苏丹应急响应室（ERRs）这一社区主导的人道主义网络中的使用情况。战争重新配置了GenAI的角色，使其在输出最难验证的地方反而最为有用。当冲突使获取人类专业知识变得困难或危险时，ERR响应者转而使用GenAI。然而，参与者警示称，GenAI无法独立掌握快速变化的当地局势，而且在难以进行人工验证的情况下，依赖其输出可能导致有害的决策。风险在GenAI被使用之前就已产生。对于一些志愿……（原文截断）

    arXiv:2611.07874v1 Announce Type: new  Abstract: Armed conflict can increase the value of Generative AI (GenAI) while undermining the conditions required for its responsible use. Yet the perspectives of community-based humanitarian responders working under these conditions remain underrepresented in responsible AI research. We examine GenAI use in Sudan's Emergency Response Rooms (ERRs), a community-led humanitarian network, through an exploratory survey of 89 respondents and 15 semi-structured interviews with GenAI users and non-users. War reconfigures GenAI's role, making it most useful where its outputs are hardest to verify. ERR responders turned to GenAI when conflict made human expertise difficult or dangerous to reach. However, participants cautioned that GenAI could not independently establish rapidly changing local conditions, and that relying on its outputs could lead to harmful decisions when human verification was difficult. Risks arose before GenAI was used. For some volun
    
[^75]: FARR：基于匹配端点证据的预测感知残差精炼方法用于长时间序列预测

    FARR: Forecast-Aware Residual Refinement with Matched Endpoint Evidence for Long-Horizon Time-Series Forecasting

    [https://arxiv.org/abs/2611.07870](https://arxiv.org/abs/2611.07870)

    提出 FARR 方法，通过对比标准条件与空条件下的预测差异获得端点证据，并利用共享残差精炼器结合逐视界门控机制来校正预测残差，在四个长期时序预测基准上取得了最优的平均 MSE 和 MAE。

    

    大语言模型使用辅助条件机制将额外信息融入时间序列预测中。然而，辅助条件并不能持续提升预测精度，当其影响与观测到的时间上下文匹配不佳时，反而可能增加预测误差。我们提出了预测感知残差精炼方法（FARR），它学习利用端点证据来校正预测。FARR 在标准条件下获得一个有效预测，并在预定义的空条件下获得一个空预测，二者的差异揭示了条件化如何改变预测结果。一个共享的残差精炼器将这一端点证据与历史特征相结合，以估计有效预测的剩余误差。随后，逐视界门控机制控制预测残差对最终输出的贡献。在四个长时间序列预测基准上的实验表明，FARR 在所比较的方法中取得了最佳的跨数据集平均 MSE 和 MAE。

    arXiv:2611.07870v1 Announce Type: new  Abstract: Large language models use auxiliary conditioning to incorporate additional information into time-series forecasts. However, auxiliary conditioning does not consistently improve accuracy and can increase forecasting error when its influence is poorly matched to the observed temporal context. We propose Forecast-Aware Residual Refinement (FARR), which learns to correct forecasts using endpoint evidence. FARR obtains a valid forecast under the standard condition and a null forecast under a predefined null condition. Their difference reveals how conditioning changes the prediction. A shared residual refiner combines this endpoint evidence with historical features to estimate the remaining error of the valid forecast. Horizon-wise gating then controls the contribution of the predicted residual to the final output. Experiments on four long-horizon forecasting benchmarks show that FARR achieves the best dataset-average MSE and MAE among the com
    
[^76]: CORE：面向深度研究的粗粒度到细粒度搜索与上下文管理

    CORE: Coarse-to-Fine Search with Context Management for Deep Research

    [https://arxiv.org/abs/2611.07859](https://arxiv.org/abs/2611.07859)

    CORE是一个无需训练的深度研究框架，通过将粗到细搜索与上下文管理相结合，解决了搜索需要长上下文与返回证据应简短之间的权衡，实现了最先进的性能。

    

    深度研究智能体通过反复搜索来回答复杂问题。它们依赖于一个执行工具调用并将证据返回到其上下文的框架（harness）。这带来了一种权衡：搜索在长段落上效果最佳，因为长段落能为可靠的语义匹配提供足够的上下文；但返回的证据应当简短，因为每条观察结果都会保留在智能体的上下文中，贯穿其后的整个轨迹。我们提出了CORE，一个无需训练的框架，它将粗粒度到细粒度的搜索与上下文管理相结合。CORE使用稠密检索器检索段落，然后复用后期交互（late-interaction）重排序器的词元级相似度，仅返回每个段落中最相关的片段。上下文管理使智能体能够围绕已检索的来源扩展上下文，并抑制重复片段。在与通义DeepResearch和OpenResearcher框架的受控封闭搜索对比中，CORE达到了最先进的性能……

    arXiv:2611.07859v1 Announce Type: cross  Abstract: Deep research agents answer complex questions through repeated search. They rely on a harness that runs their tool calls and returns evidence to their context. This creates a trade-off: Search works best over long passages, which give enough context for reliable semantic matching. But the returned evidence should be short, because every observation stays in the agent's context for the rest of the trajectory. We introduce CORE, a training-free harness that combines coarse-to-fine search with context management. CORE retrieves passages with a dense retriever. It then reuses the token-level similarities from a late-interaction reranker to return only the most relevant snippet from each passage. Context management lets the agent expand the context around sources it has already retrieved and suppresses repeated snippets. In controlled closed-search comparisons with the Tongyi DeepResearch and OpenResearcher harnesses, CORE achieves state-of
    
[^77]: 计算而非权重访问：大语言模型作为临床预测网络的审计者

    Computation, not weight access: large language models as auditors of clinical prediction networks

    [https://arxiv.org/abs/2611.07847](https://arxiv.org/abs/2611.07847)

    通过预先设定的模拟实验，该研究发现在没有计算工具时仅凭提示中的原始权重无法可靠恢复植入的关系，表明大语言模型审计临床预测网络的关键在于计算能力（借助工具访问权重或预测）而非直接的权重访问。

    

    据报道，前沿大语言模型（LLM）能够读取、编辑和编写小型神经网络的权重，这为仅凭模型文件审计临床预测模型提供了可能。然而，公开演示中将读取原始权重与运行代码混在了一起。我们在内部预先设定的模拟实验中将这些条件区分开来。研究在模拟的蛛网膜下腔出血患者队列上训练了具有105至38,401个权重的Tanh网络，每个网络中植入了一个关系或不植入任何关系，其中一个网络所使用的标签在某个亚组中少记了死亡人数。GPT-6 Astra和其他四个GPT模型在默认或明确的系统指令下回答单个提示，条件为：提示中包含权重且不使用工具，或者使用仅能进行计算的工具，该工具或暴露权重（白盒），或仅暴露网络的预测（黑盒）。在不使用工具的情况下，Astra在353权重网络中恢复了植入的关系

    arXiv:2611.07847v1 Announce Type: cross  Abstract: Frontier large language models (LLMs) have been reported to read, edit and write the weights of small neural networks, raising the prospect of auditing clinical prediction models from their files alone. Public demonstrations, however, mixed reading raw weights with running code. We separated these conditions in internally pre-specified simulation experiments. Tanh networks of 105 to 38,401 weights were trained on simulated cohorts of patients with subarachnoid haemorrhage, each with a planted relationship or none, and one network was trained on labels that under-recorded deaths in a subgroup. GPT-6 Astra and four other GPT models answered single prompts under a default or an explicit system instruction, either with the weights in the prompt and no tools, or with a calculation-only tool that exposed the weights (white box) or only the network's predictions (black box). Without tools, Astra recovered planted relationships in 353-weight n
    
[^78]: 掩码扩散作为随机序贯吸附：一个精确可解的并行解码模型

    Masked diffusion as random sequential adsorption: an exactly solvable model for parallel decoding

    [https://arxiv.org/abs/2611.07845](https://arxiv.org/abs/2611.07845)

    该论文将掩码扩散模型并行解码的误差与驱动多粒子系统的熵产生和耗散联系起来，并把马尔可夫数据下的生成精确等价为一个可严格求解的随机序贯吸附过程，从而建立了并行解码的首个精确可解理论模型。

    

    掩码扩散模型通过逆转一个随机掩码过程来生成文本。与每次生成一个token的自回归大语言模型不同，它们可以并行揭示多个token，但代价是引入误差，因为同时揭示的token是彼此独立抽取的。我们证明这种权衡可以映射到驱动多粒子系统的统计物理中。(i) 对于任意数据，并行生成的误差等于生成过程相对于逆转掩码过程的熵产生。(ii) 对于马尔可夫数据，生成过程是一个带标记间隙的随机序贯吸附过程，我们对其任意步数均给出了精确解。(iii) 对于多步情况，误差变为缓慢驱动系统的耗散：去掩码调度扮演驱动协议的角色，步数扮演其持续时间的角色，而误差由一个热力学度规决定，我们从互信息中以闭式形式求得了该度规。

    arXiv:2611.07845v1 Announce Type: cross  Abstract: Masked diffusion models generate text by reversing a stochastic masking process. Unlike autoregressive LLMs, which generate text one token at a time, they can reveal many tokens in parallel, at the price of an error, since tokens revealed together are drawn independently. We show that this trade-off maps onto the statistical physics of driven many-particle systems. (i) For any data, the error of parallel generation is the entropy production of the generation process relative to the reversed masking process. (ii) For Markov data, generation is a random sequential adsorption process with labelled gaps, which we solve exactly for any number of steps. (iii) For many steps, the error becomes the dissipation of a slowly driven system: the unmasking schedule plays the role of the driving protocol, the number of steps that of its duration, and the error is set by a thermodynamic metric that we obtain in closed form from the mutual information 
    
[^79]: THG-Omni：面向视频推理的可追溯以人为中心图监督框架

    THG-Omni: Traceable Human-Centric Graph Supervision for Video Reasoning

    [https://arxiv.org/abs/2611.07836](https://arxiv.org/abs/2611.07836)

    THG-Omni提出了一种以人为中心的可追溯图监督框架，通过将SAM3参与者追踪与时间定位的Qwen3-VL标注相结合并核验语义声明，使全模态推理模型能够从结构化图中解读视频里人物的属性、动作、情感、交互及其时间关系。

    

    全模态模型能够回答关于视频的问题，但其预测很少展示出是哪个参与者、事件或时间段支撑了答案。此外，仅依靠答案级别的监督无法教会模型解读结构化的视频证据。为了系统地缓解这一问题，我们提出了THG-Omni，这是一个以人为中心的图监督框架，它将Segment Anything Model 3的参与者追踪与具有时间定位的Qwen3-VL标注相连接。候选语义声明在被用于构建训练问题之前，会先对照源视频和可用音频进行核验，同时追踪几何信息提供了显式的空间和时间监督。由此产生的训练问题教会全模态推理模型从结构化图中解读参与者的属性、动作、情感、交互及其时间关系。在推理阶段，THG-Omni使用从评估视频构建的预测图。在IntentBench上的实验表明...

    arXiv:2611.07836v1 Announce Type: cross  Abstract: Omni-modal models can answer questions about video, but their predictions rarely show which participant, event, or interval supports an answer. Moreover, answer-level supervision alone does not teach a model to interpret structured video evidence. To systematically mitigate this issue, we present THG-Omni, a human-centric graph-supervision framework that links Segment Anything Model~3 participant tracks with temporally grounded Qwen3-VL annotations. Candidate semantic claims are checked against the source video and available audio before they are used to construct training questions, while track geometry provides explicit spatial and temporal supervision. The resulting questions teach an omni-modal reasoning model to interpret participant attributes, actions, affect, interactions, and their timing from a structured graph. At inference, THG-Omni uses a predicted graph constructed from the evaluation video. Experiments on IntentBench sho
    
[^80]: STAG-Net：基于地标组的结构感知注意力机制实现轻量级头部姿态估计

    STAG-Net: Structure-Aware Attention over Landmark Groups for Lightweight Head Pose Estimation

    [https://arxiv.org/abs/2611.07828](https://arxiv.org/abs/2611.07828)

    STAG-Net通过对面部关键点组施加结构感知注意力（结合运动感知选择、空间聚类、紧凑DeepSet编码器和全局几何描述子），以极低的计算成本实现了精度领先的轻量级头部姿态估计。

    

    头部姿态估计对人机交互和驾驶员监控非常重要，但许多基于图像的方法对于资源受限的部署场景而言计算成本仍然过高。我们提出了STAG-Net，一种面向轻量级头部姿态估计的基于地标组的结构感知注意力方法，它通过运动感知选择、空间聚类、紧凑的DeepSet编码器、自注意力机制以及全局几何描述子，从面部关键点估计头部姿态。我们开发了使用FaceMesh和面部对齐网络（FAN）关键点的两种变体，并在300W-LP、AFLW2000和BIWI数据集上，采用跨数据集和独立于受试者的评估协议进行了测试。在所有评估设置中，STAG-Net在所有被比较的基于关键点的方法中均取得了最低的平均绝对误差。在具有有效检测结果的BIWI帧上，FAN变体还在所有被比较的方法中录得最低的总体误差，在跨数据集评估下达到3.51°，在独立于受试者的评估下达到2.39°。

    arXiv:2611.07828v1 Announce Type: cross  Abstract: Head pose estimation is important for human computer interaction and driver monitoring, but many image based methods remain too costly for resource constrained deployment. We propose STAG-Net, a structure-aware attention over landmark groups for lightweight head Pose estimation that estimates head pose from facial landmarks using movement aware selection, spatial clustering, compact DeepSet encoders, self attention, and a global geometry descriptor. We develop variants using FaceMesh and Face Alignment Network landmarks and evaluate them under cross dataset and subject independent protocols on 300W-LP, AFLW2000, and BIWI. STAG-Net achieves the lowest mean absolute error among all compared landmark based methods in every evaluated setting. On BIWI frames with valid detections, the FAN variant also records the lowest overall error among all compared methods, reaching $3.51^\circ$ under cross dataset evaluation and $2.39^\circ$ under subj
    
[^81]: 实例化知识平面

    Instantiating the Knowledge Plane

    [https://arxiv.org/abs/2611.07827](https://arxiv.org/abs/2611.07827)

    本文提出利用代理型大语言模型实现2003年Clark等人设想的“知识平面”，从而实现跨管理边界的自动化互联网协作管理，并指出必须解决去中心化、多代理、竞合环境下出现的新研究问题。

    

    2003年，Clark等人提出了“知识平面”的概念——一个由智能代理组成的层次，这些代理共享丰富的网络状态信息，以便跨越管理边界协作管理互联网。然而，当时的技术缺乏它最需要的两种能力：推理和非结构化通信。因此，自动化网络管理在很大程度上仍局限于单一管理域内，而跨域问题至今仍需人工操作员通过电话来解决。本文认为，代理型大语言模型最终能够释放知识平面的潜力，但要实现这一点，需要解决在这一去中心化、多代理、竞合环境中特有的新研究问题，而这些问题迄今为止在文献中受到的关注有限。

    arXiv:2611.07827v1 Announce Type: cross  Abstract: In 2003, Clark et al. proposed the "Knowledge Plane", a layer of intelligent agents that share rich information about network state in order to collaboratively manage the Internet across administrative boundaries. The technology of the day lacked the two capabilities it demanded most: reasoning and unstructured communication. Thus, automated network management has remained largely confined to single domains, while interdomain problems are still resolved by human operators on the phone. We argue that agentic large language models can finally unlock the potential of the Knowledge Plane, but that doing so will require solving new research problems that arise particularly in this decentralized, multi-agent, coopetitive setting that have received limited attention in the literature thus far.
    
[^82]: 生成后排序：学习具身智能体的符号目标解释

    Generate-then-Rank: Learning Symbolic Goal Interpretation for Embodied Agents

    [https://arxiv.org/abs/2611.07825](https://arxiv.org/abs/2611.07825)

    该论文提出两阶段“生成后排序”方法用于具身智能体的符号目标解释，通过监督微调适配输出规范并结合基于错误的候选输出排序（CORE）学习输出偏好，在EAI基准上超越了现有基线模型。

    

    目标解释是将自然语言指令翻译成规划器可执行的正式符号目标。这对具身智能体至关重要，因为该环节的任何错误都会传播到下游的所有规划与行动步骤。要做好这一任务，既需要生成合适的候选结果，也需要在其中进行可靠的选择。我们提出了一种面向具身智能体接口（EAI）中目标解释的两阶段“生成后排序”方法。首先，通过监督微调使大语言模型（LLM）适应EAI的输出规范；其次，基于错误的候选输出排序方法（CORE）从模型自身生成的备选输出中学习对完整输出的偏好。我们在具身智能体接口基准的家庭场景上进行了实验，评估了多个开源大语言模型，包括DeepSeek-V3 chat、DeepSeek-Reasoner以及Qwen2.5-7B和Qwen2.5-72B。所提出的方法超越了包括……在内的基线模型。

    arXiv:2611.07825v1 Announce Type: new  Abstract: Goal interpretation is to translate a natural language instruction into a formal symbolic goal that a planner can execute. It is critical for embodied agents because any error here propagates to every downstream planning and action step. Doing it well requires both generating suitable candidates and reliably selecting among them. We present a two-stage generate-then-rank approach to Goal Interpretation in the Embodied Agent Interface (EAI). First, supervised fine-tuning adapts a Large Language Model (LLM) to EAI's output convention. Second, Candidate-Output Ranking from Errors (CORE) learns preferences over complete outputs from the model's own generated alternatives. We conducted experiments on the household scenes of the Embodied Agent Interface Benchmark. Multiple open source LLMs including DeepSeek-V3 chat, DeepSeek-Reasoner, and Qwen2.5-7B and Qwen2.5-72B were evaluated. The proposed method outperformed the baseline models including
    
[^83]: Axis-LoRA：面向个人记忆检索的嵌入模型探测与适配

    Axis-LoRA: Probing and Adapting Embedding Models for Personal-Memory Retrieval

    [https://arxiv.org/abs/2611.07820](https://arxiv.org/abs/2611.07820)

    该论文发现冻结的稠密嵌入模型无法按时间、地点、人物、事件四个维度有效区分个人记忆，并提出 Axis-LoRA——在嵌入模型内部为每个维度设置 LoRA 分支并通过逐查询门控融合的适配方法，仅训练少量参数即可在三个记忆检索基准上显著提升性能。

    

    随着大语言模型越来越多地被问及有关用户个人经历的问题，检索系统必须能够按时间、地点、人物和事件四个维度来区分记忆。当前的个人记忆处理流水线依赖冻结的稠密嵌入模型来嵌入记忆以供检索，但我们发现这些模型并不能很好地捕捉这四个维度：更换一条记忆的日期或地点几乎不会改变其嵌入向量，而在冻结嵌入之上训练的投影层也无法恢复缺失的信号。问题出在模型内部，其中每个维度由一种独特的注意力头模式进行编码。基于这些发现，我们提出了 Axis-LoRA，一种按维度设计的适配器，它在嵌入模型内部放置四个 LoRA 分支，并通过逐查询门控机制将它们组合起来。在 LongMemEval、ATM-Bench 和 LoCoMo 上对三个标准嵌入模型进行评估，Axis-LoRA 将四个基准的平均 R@10 提升了 1.28 至 1.56 个百分点，同时仅训练基础参数的一小部分。

    arXiv:2611.07820v1 Announce Type: cross  Abstract: As LLMs are increasingly asked questions about a user's personal experiences, retrieval must distinguish memories by time, place, person, and event. Current personal-memory pipelines rely on frozen dense embedding models to embed memories for retrieval, but we find that these models do not capture the four axes well. Swapping a memory's date or place barely shifts its embedding, and a trained projection on top of the frozen embedding does not recover the missing signal. The problem lives inside the model, where each axis is encoded by a distinct attention-head pattern. Guided by these findings, we propose Axis-LoRA, a per-axis adapter that places four LoRA branches inside the embedding model and combines them via a per-query gate. Across three standard embedding models evaluated on LongMemEval, ATM-Bench, and LoCoMo, Axis-LoRA improves 4-benchmark mean R@10 by $+1.28$ to $+1.56$~pp while training only a small fraction of the base param
    
[^84]: 连接结构与语义：基于三元组到文本对齐的知识图谱验证

    Bridging Structure and Semantics: Knowledge Graph Verification via Triple-to-Text Alignment

    [https://arxiv.org/abs/2611.07780](https://arxiv.org/abs/2611.07780)

    提出GenVer生成式验证器，通过对齐三元组及其生成的自然语言表示来融合图结构信息与预训练语言模型知识，在FactFB数据集上取得了最高达91.52%的三元组分类准确率。

    

    语义准确性是知识图谱质量的一个重要维度。基于图的评分方法利用关系结构，而文本表示则使分类器能够利用预训练语言模型（PLM）中编码的知识。我们提出了GenVer，一种生成式验证器，它利用三元组的序列化形式和生成的自然语言表示（NLR）来对三元组进行评分。通过三元组-文本对齐程序为序列到序列生成器选择训练样本对，然后由基于BERT的分类器将每个三元组及其生成的NLR的特征相结合。在从FB15k通过扰动负三元组构建的四个FactFB变体上，GenVer在评估的基线方法中取得了最高的分类准确率，范围从88.56%到91.52%。实验表明，生成的表示有助于三元组分类，同时案例分析也揭示了语义漂移和高置信度错误的存在。

    arXiv:2611.07780v1 Announce Type: cross  Abstract: Semantic accuracy is an important dimension of knowledge graph quality. Graph-based scoring methods exploit relational structure, while textual representations allow classifiers to draw on knowledge encoded by pre-trained language models (PLMs). We present GenVer, a generative verifier that scores triples using both their serialized form and generated natural language representations (NLRs). A triple-text alignment procedure selects training pairs for a sequence-to-sequence generator; a BERT-based classifier then combines features from each triple and its generated NLR. On four FactFB variants, constructed from FB15k using perturbed negative triples, GenVer obtains the highest classification accuracy among the evaluated baselines, ranging from 88.56% to 91.52%. The experiments suggest that generated representations can help triple classification, while the case analysis also exposes semantic drift and high-confidence errors.
    
[^85]: 侵入通路：非证据性上下文如何降低语言模型性能

    The Intrusion Pathway: How Non-Evidential Context Degrades Language Model Performance

    [https://arxiv.org/abs/2611.07778](https://arxiv.org/abs/2611.07778)

    本文揭示了非证据性上下文（无关陈述或随机标记）降低语言模型性能的内在机制，将其归因于“侵入通路”及一组稀疏的“侵入头”注意力头，这些头将无关的前缀内容写入预测位置从而导致准确率下降。

    

    当提示词中包含非证据性上下文时，语言模型会损失准确率：这些上下文包括来自无关领域的不相关陈述，或随机的分布外标记，它们既不包含竞争性答案，也与查询没有语义重叠。这种噪声普遍存在于检索增强、工具使用和长上下文等部署场景中，然而这一失败现象一直缺乏机制层面的解释。我们将这种性能损坏追溯到“侵入通路”：其起点是MLP同时丰富查询主体和不相关标记的表示，随后“鹦鹉学舌头”将前缀标记的内容传递到预测位置，而关系读取器则在很大程度上避免读取无关来源。然而，被保留的主体信息增强和选择性读取并不能阻止更广泛计算中发生的侵入。随后，我们将这种干扰的一部分定位到一组稀疏的注意力头上，我们将其称为“侵入头”。它们源自前缀的写入会促进与干扰物相关的内容，同时它们对……（摘要在此处截断）的支持则有所减少。

    arXiv:2611.07778v1 Announce Type: new  Abstract: Language models lose accuracy when a prompt contains non-evidential context: irrelevant statements from unrelated domains, or random off-distribution tokens, with no competing answer and no semantic overlap with the query. Such noise pervades retrieval-augmented, tool-using, and long-context deployments, yet the failure has lacked a mechanistic account. We trace the corruption to the Intrusion Pathway, which starts with MLPs enriching both query-subject and irrelevant-token representations, parroting heads carry prefix-token content to the prediction position, and relational readers largely avoid unrelated sources. Nevertheless, preserved subject enrichment and selective reading do not prevent intrusion in the broader computation. We then localize part of this interference to a sparse set of attention heads, which we call intrusion heads. Their prefix-derived writes promote distractor-related content, while their reduced support for the 
    
[^86]: RIVER：证据引导的多智能体检索方法用于酶-反应发现

    RIVER: Evidence-guided multi-agent retrieval for enzyme--reaction discovery

    [https://arxiv.org/abs/2611.07757](https://arxiv.org/abs/2611.07757)

    本文提出RIVER，一个证据引导的多智能体框架，通过协调大语言模型、生化计算工具与外部知识资源进行多层面证据整合，实现双向酶-反应检索，克服了传统基于序列相似性方法难以整合生化证据及区分高相似酶家族的局限。

    

    arXiv:2611.07757v1 公告类型：新论文 摘要：酶功能注释以及为给定反应识别候选酶是连接基因组信息与生化反应网络的关键步骤。然而，现有方法依赖于序列或结构相似性，或依赖已知酶-反应对进行训练，难以在大规模候选空间中整合生化证据；此外，这些方法在处理成员高度相似的酶家族的检索任务时也面临困难。本文提出了RIVER（基于证据引导检索的反应-酶推断），这是一个用于双向酶-反应检索的多智能体框架，既能针对给定酶识别其可能催化的反应，也能针对给定反应对候选酶进行排序。RIVER协调大型语言模型、专门的生化计算工具和外部知识资源，在多个层面（包括反应化学、蛋白质……）收集证据。

    arXiv:2611.07757v1 Announce Type: new  Abstract: Enzyme function annotation and the identification of candidate enzymes for a given reaction are essential steps in connecting genomic information to biochemical reaction networks. However, existing methods rely on sequence or structural similarity or on training with known enzyme--reaction pairs, making it difficult to integrate biochemical evidence across large candidate spaces; they also struggle with retrieval tasks involving enzyme families whose members are highly similar. Here, we introduce RIVER (Reaction--enzyme Inference via Evidence-guided Retrieval), a multi-agent framework for bidirectional enzyme--reaction retrieval that identifies potentially catalyzed reactions for a given enzyme and ranks candidate enzymes for a given reaction. RIVER coordinates large language models, specialized biochemical computational tools, and external knowledge resources to assemble evidence at multiple levels, including reaction chemistry, protein
    
[^87]: SAE（稀疏自编码器）训练数据的缩放定律

    Scaling Laws for SAE Training Data

    [https://arxiv.org/abs/2611.07756](https://arxiv.org/abs/2611.07756)

    在固定训练预算下，SAE 所需的独特激活数据量存在饱和点——缓冲区达到一定规模后，重放数据与新鲜激活值效果相当，且缓冲区无需随训练预算扩大；作者据此提出并在多种模型、层级和 SAE 架构上验证了一条刻画多样性与重复权衡的有限数据缩放定律。

    

    激活值存储通常是稀疏自编码器（SAE）训练中的瓶颈。我们研究了在总训练预算固定的情况下，SAE 需要多少条独特的激活值。随着缓冲区规模增大，质量迅速提升，随后进入一个平台期：在此阶段重放缓冲区数据与在整个训练过程中持续使用新鲜激活值几乎同样有效。超过这一点后，额外的独特激活值对重构效果或可解释性几乎没有额外收益，因此缓冲区规模无需随训练预算增长。我们用一条针对有限数据的缩放定律来描述这种多样性与重复之间的权衡。我们在不同的字典规模与 token 预算、Llama-3.1 的早期、中期和后期层、Qwen3 的不同模型规模，以及 BatchTopK、TopK 和 JumpReLU 等多种 SAE 上验证了该定律。通过在数学文本和 Python 代码上进行扫描，我们测试了改变激活值来源的影响。评估涵盖解释方差（EV）、重构 MSE、特征吸收、自动化（评估）……

    arXiv:2611.07756v1 Announce Type: cross  Abstract: Activation storage is often a bottleneck in sparse autoencoder (SAE) training. We study how many unique activations an SAE needs when its total training budget is fixed. Quality improves rapidly as the buffer grows, then reaches a plateau where replaying the buffer is nearly as effective as using fresh activations throughout training. Beyond this point, additional unique activations bring little benefit to reconstruction or interpretability, so the buffer need not grow with the training budget. We describe this tradeoff between diversity and repetition with a scaling law for limited data. We validate it across dictionary sizes and token budgets, early, middle, and late layers of Llama-3.1, Qwen3 model sizes, and BatchTopK, TopK, and JumpReLU SAEs. Sweeps using mathematical text and Python code test the effect of changing the activation source. Evaluation covers explained variance (EV), reconstruction MSE, feature absorption, automated 
    
[^88]: 通过循环机制扩展类型化决策模型的推理能力

    Extending Reasoning in Typed Decision Models via Looping

    [https://arxiv.org/abs/2611.07736](https://arxiv.org/abs/2611.07736)

    该论文提出 RELOAD 方法，通过对预训练模型的中间块进行阻尼循环并由轻量级控制器自适应选择循环次数，首次证明循环机制在与决策模型联合训练时能显著提升类型化决策模型的准确性与校准度，尤其在需要长推理链的问题上编码器最多可提升 23 分。

    

    类型化决策模型在单次前向传播中从固定的答案选项集合中进行选择，因此与生成式语言模型不同，它们无法通过生成更多 token 来在困难输入上投入更多的测试时计算量。对一层块进行循环可以在不增加参数的情况下在潜空间中提供这种额外计算，但先前的工作发现，将循环改装到按常规方式预训练的模型中并未带来收益。我们证明，当这种循环与决策模型一起训练时才能发挥作用。我们的方法 RELOAD（用于自适应决策的改装循环）对预训练的编码器或解码器进行微调，在其中间块上采用阻尼循环，并由一个轻量级控制器为每个输入自适应地选择循环次数。在从 0.4B 到 4B 参数的六个模型上，循环机制提升了准确性和校准度，提升主要体现在那些需要比训练中见过的任何问题都更长的推理链的问题上，编码器在此类问题上最多可获得 23 分的提升。无论是匹配的训练计算量还是……（原文摘要被截断）

    arXiv:2611.07736v1 Announce Type: cross  Abstract: Typed decision models choose among a fixed set of answer options in a single forward pass, so unlike generative language models they cannot spend more test-time compute on hard inputs by generating more tokens. Looping a block of layers offers this compute in latent space without adding parameters, but prior work found no gain from loops retrofitted into conventionally pretrained models. We show that such loops help when trained together with the decision model. Our method, RELOAD (retrofitted loops for adaptive decisions), fine-tunes a pretrained encoder or decoder with a damped loop over its middle block and a lightweight controller picks the number of passes adaptively for each input. Across six models from 0.4B to 4B parameters, looping improves both accuracy and calibration, mostly on problems that need longer chains of reasoning than any seen in training, where encoders gain up to 23 points. Neither matched training compute nor m
    
[^89]: 评估Token概率作为人类行为模拟中的置信度估计器：以卢森堡语下一次点赞预测为例的案例研究

    Evaluating Token Probabilities as Confidence Estimators in Human Behavior Simulations: A Case Study in Luxembourgish Next Like Prediction

    [https://arxiv.org/abs/2611.07735](https://arxiv.org/abs/2611.07735)

    本研究以卢森堡新闻平台的点赞预测任务为例，发现LLM的token概率作为置信度估计时模型过度自信、被简单统计基线全面超越，且准确率与校准度并不同步——扩大模型规模能提升准确率却损害校准度，表明评估人类行为模拟时校准度是不可忽视的关键指标。

    

    随着大型语言模型（LLM）越来越多地被用于模拟人类行为，其操作有效性通常仅通过准确率来评估，这隐含地将人类行为视为确定性的。我们认为，在嘈杂的数字痕迹数据上，模型的性能不仅取决于预测出正确的结果，还取决于其校准程度。我们开展了一项案例研究，测试开放权重LLM的token概率能否在“下一次点赞预测”任务中作为置信度估计：即在用户先前交互行为的条件下，预测卢森堡新闻平台RTL.lu的用户会如何对评论进行评分。通过评估Ministral 3（3B、8B），我们发现仅使用上下文学习时，被测模型表现出过度自信，且在准确率和校准度上均被简单的统计基线所超越。准确率与校准度并不同步变化：从3B扩展到8B提升了准确率却降低了校准度，而更长的行为……（摘要原文在此处被截断）

    arXiv:2611.07735v1 Announce Type: cross  Abstract: As large language models (LLMs) are increasingly used to simulate human behavior, their operational validity is typically assessed through accuracy alone, implicitly treating human behavior as deterministic. We argue that on noisy digital trace data, a model's performance depends not only on predicting the correct outcome but on its calibration. We present a case study testing whether the token probabilities of open-weight LLMs can serve as confidence estimates in a Next Like Prediction task: predicting how a user of the Luxembourgish news platform RTL.lu rates a comment, conditioned on their prior interactions. Evaluating Ministral 3 (3B, 8B), we find that, with in-context learning only, the tested models are overconfident and are outperformed on accuracy and calibration by a simple statistical baseline. Accuracy and calibration do not move together: scaling from 3B to 8B improves accuracy while degrading calibration, and longer behav
    
[^90]: O$^2$PD：基于可靠前缀的重叠门控在线策略蒸馏

    O$^2$PD: Overlap-Gated On-Policy Distillation on Reliable Prefix

    [https://arxiv.org/abs/2611.07731](https://arxiv.org/abs/2611.07731)

    提出 O$^2$PD 方法，通过定量分析教师监督的可靠性来确定“对齐哪些 token”（可靠前缀），并以重叠门控机制决定“如何对齐”，从而弥补现有 top-$k$ 在线策略蒸馏在监督可靠性与概率质量丢弃方面的不足。

    

    在线策略蒸馏（OPD）通过在学生自身生成的 rollout 上将学生对齐到教师的下一 token 分布，从而提供密集的 token 级监督。现有工作大多实现 top-$k$ OPD，即在教师 top-$k$ 支持集上，对教师与学生的重归一化分布应用反向 Kullback-Leibler（RKL）散度。然而，教师所打分的学生生成的 rollout 对教师自身而言天然是离策略的，其下一 token 引导的可靠性并不明确。将散度限制在教师 top-$k$ 支持集上，也会丢弃该支持集之外的概率质量。这些缺陷隐含了两个问题：1）对齐哪些 token；2）如何对齐这些 token。为回答这些问题，我们提出重叠门控在线策略蒸馏（O$^2$PD）。针对问题 1），我们定量分析了教师监督的可靠性，并发现对学生轨迹上的 token 进行对齐……（原文摘要在此处被截断）

    arXiv:2611.07731v1 Announce Type: cross  Abstract: On-policy distillation (OPD) supplies dense token-level supervision by aligning a student to the teacher's next-token distribution on its own generated rollouts. Existing works mostly implement top-$k$ OPD, applying reverse Kullback-Leibler ($\RKL$) divergence between the teacher's and the student's renormalized distributions over teacher top-$k$ support. However, the teacher scores student-generated rollouts that are inherently off-policy for itself, leaving the reliability of its next-token guidance unclear. Restricting the divergence to the teacher top-$k$ support also discards the remaining mass outside that support. These gaps leave two questions implicit: 1) which tokens to align, and 2) how to align those tokens. To answer these questions, we propose Overlap-gated On-Policy Distillation (O$^{2}$PD). As for 1), we quantitatively analyze the reliability of teacher supervision and find that alignment on the student's trajectory pre
    
[^91]: 看透下一步：多智能体场景中世界模型引导的类人导航

    Seeing Beyond the Next Step: World-Model-Guided Human-Like Navigation in Multi-Agent Scenes

    [https://arxiv.org/abs/2611.07730](https://arxiv.org/abs/2611.07730)

    该论文提出将世界模型置于训练阶段作为偏好oracle，把对未来的前瞻性推理蒸馏进一个快速的视觉动作先验策略中，而非在推理时用作在线规划器，从而在不牺牲速度的前提下实现多智能体场景中类人的导航行为。

    

    在人群密集的场景中选择下一步动作，不仅仅是避开眼前障碍的问题：某个动作可能在局部是可行的，却会让人偏离人行横道走上马路、切入正在过街的行人流，或在狭窄通道中造成拥堵。这些失败模式仅从当前状态是无法察觉的，而要应对它们就需要推理每个候选动作将会引发的后果。世界模型恰好提供了这种能力，即基于动作条件的未来预测，但在每个决策步骤都 rollout 世界模型的代价是可部署的导航系统无法承受的，因此该领域的研究大多以牺牲前瞻性来换取速度。我们认为这是一种错误的二分法。在以人为中心的导航中，世界模型自然的用武之地是在训练阶段，作为一个将前瞻性蒸馏到快速策略中的偏好 oracle，而不是在推理阶段充当在线规划器。我们将一个离散的视觉动作先验与两个任务对齐的（摘要在此处截断）

    arXiv:2611.07730v1 Announce Type: cross  Abstract: Selecting a next motion in a populated scene is not just a matter of avoiding what is currently in the way: an action can be locally feasible yet drift onto the road instead of the crosswalk, cut into a stream of crossing pedestrians, or bottleneck a narrow passage. These failure modes are invisible from the current state alone, and acting on them requires reasoning about what each candidate action will cause. World models offer exactly this, namely action-conditioned predictions of the future, but rolling them out at every decision step is a cost deployable navigation systems cannot pay, and the field has largely traded foresight for speed. We argue this is a false dichotomy. The natural place for a world model in human-aware navigation is at training time, as a preference oracle that distills foresight into a fast policy, rather than at inference time, as an online planner. We pair a discrete visual action prior with two task-aligned
    
[^92]: 基于词元身份平方监督的连续语言流继续训练方法

    Continued Training of Continuous Language Flows with Token Identity Squared Supervision

    [https://arxiv.org/abs/2611.07721](https://arxiv.org/abs/2611.07721)

    提出词元身份Brier流，通过直接对词元身份施加平方监督（Brier损失）解决了连续MSE目标无法区分不同概率分布的问题，从而更有效地继续训练预训练语言流模型。

    

    扩散模型和流模型最近被扩展到语言建模领域，通过在词元嵌入空间中进行连续去噪来生成文本。预训练模型提供了词元表示和上下文预测能力，但要在有限额外训练的情况下改进生成效果，需要在监督信号与连续生成之间建立适当的连接。连续MSE评估的是通过用预测概率对词元嵌入加权所得到的预测向量；不同的概率分布可以产生相同的向量，因此在该目标函数下无法被区分。我们提出词元身份Brier流（TIBF），在继续训练预训练语言流模型时，直接对词元身份应用平方监督。TIBF在高斯观测模型下将网络得分与词元匹配得分相结合，使用Brier损失对词元概率进行监督，并使用概率加权嵌入进行……

    arXiv:2611.07721v1 Announce Type: cross  Abstract: Diffusion and flow models have recently been extended to language modeling, generating text through continuous denoising in token embedding spaces. Pretrained models provide token representations and contextual prediction capabilities, but improving generation with limited additional training requires an appropriate connection between supervision and continuous generation. Continuous MSE evaluates a predicted vector obtained by weighting token embeddings with predicted probabilities; different probability distributions can yield the same vector and are therefore indistinguishable under this objective. We propose Token Identity Brier Flow (TIBF), which applies squared supervision directly to token identities when continuing pretrained language flows. TIBF combines network scores with token-matching scores under a Gaussian observation model, supervises token probabilities using the Brier loss, and uses probability-weighted embeddings for
    
[^93]: 在隔离假设下评估语言模型特征用于零样本时间序列异常检测

    Evaluating Language Model Features for Zero-Shot Time-Series Anomaly Detection Under Isolation Assumption

    [https://arxiv.org/abs/2611.07720](https://arxiv.org/abs/2611.07720)

    该论文提出将大语言模型纯粹作为特征提取器用于零样本多变量时间序列异常检测，通过直接分析其诱导的特征空间，规避了预测或重建等辅助目标在模型容量增大时会重建异常而变得不可靠的问题。

    

    时间序列异常检测（TSAD）是一项具有根本性挑战的任务：它是无监督的，需要高局部精度能力，且受限于有限的标注数据。近期研究探索了通过跨模态迁移将大语言模型（LLM）适配到时间序列领域。然而，现有方法并不适合多变量时间序列异常检测（MTSAD），因为分词处理往往会掩盖细粒度的时间结构和变量结构，且大多数方法都在优化诸如预测或重建等辅助目标。我们认为，随着模型容量的增加，这些替代目标会变得不可靠——高表达能力的模型可能会重建或预测出异常，从而破坏这两种策略的基础。相反，我们提出将LLM纯粹用作特征提取器，并直接分析其诱导的特征空间。在本工作中，我们研究了LLM在零样本设置下为MTSAD提供有用表示的能力。

    arXiv:2611.07720v1 Announce Type: cross  Abstract: Time-Series Anomaly Detection (TSAD) is a fundamentally challenging task: it is unsupervised, requires high local precision capabilities, and suffers from limited labelled data. Recent work has explored adapting Large Language Models (LLMs) to time-series via cross-modal transfer. However, existing approaches are poorly suited to Multivariate TSAD (MTSAD), as tokenisation often obscures fine-grained temporal and variables structures, and most methods optimise auxiliary objectives such as forecasting or reconstruction. We posit that these surrogate objectives become unreliable as model capacity increases. Highly expressive models risk reconstructing or forecasting anomalies, undermining the foundations of both strategies. Instead, we propose to use LLMs purely as feature extractors and directly analyse their induced feature spaces. In this work, investigate the ability of LLMs to provide useful representations for MTSADs in a zero-shot 
    
[^94]: 基于成人到儿科微调的儿科心电图基础模型：下游任务与临床评估

    Pediatric ECG foundation model via adult-to-pediatric fine-tuning: downstream tasks and clinical evaluation

    [https://arxiv.org/abs/2611.07717](https://arxiv.org/abs/2611.07717)

    该研究证明成人心电图基础模型经儿科数据微调后性能可超越从头训练，具备良好的外部泛化能力和下游任务扩展性，并能显著提升临床医生的诊断准确率。

    

    儿科心电图（ECG）解读因年龄依赖性电生理变化而变得复杂，这限制了成人AI模型的直接迁移。我们探究了成人ECG基础模型能否适配到儿科心电图。我们使用来自238,189名儿童的339,563份12导联心电图，在相同架构下比较了直接应用、儿科微调和从头训练三种方法。直接迁移效果较差（平均AUC 0.8256）。微调后达到0.9695，高于从头训练（0.9661），且仅使用1%的训练数据即可达到0.9079。在10,999份外部心电图上，微调模型在30个标签上的平均AUC达到0.9385。该模型还能泛化至年龄回归任务（MAE 1.12岁）和先天性心脏病分类（AUC 0.9562）。在涉及17名临床医生和100个病例的评估中，诊断准确率从56.8%提升至65.4%（p < 0.001），且当AI出错时医生准确率并未下降。成人ECG基础模型需要进行儿科适配；微调

    arXiv:2611.07717v1 Announce Type: new  Abstract: Pediatric electrocardiogram (ECG) interpretation is complicated by age-dependent electrophysiological changes that limit direct transfer of adult AI models. We asked whether an adult ECG foundation model could be adapted to pediatric ECGs. Using 339,563 12-lead ECGs from 238,189 children, we compared direct application, pediatric fine-tuning, and training from scratch under an identical architecture. Direct transfer was poor (mean AUC 0.8256). Fine-tuning reached 0.9695, above de novo training (0.9661), and attained 0.9079 using only 1% of training data. On 10,999 external ECGs, fine-tuning achieved a mean AUC of 0.9385 across 30 labels. It generalized to age regression (MAE 1.12 years) and congenital heart disease classification (AUC 0.9562). Across 17 clinicians and 100 cases, diagnostic accuracy rose from 56.8% to 65.4% (p < 0.001) and did not decline when AI erred. Adult ECG foundation models require pediatric adaptation; fine-tuning
    
[^95]: CAMP：面向3D-IMC阵列上大语言模型的上下文感知映射与放置

    CAMP: Context-Aware Mapping and Placement for LLMs on 3D-IMC Arrays

    [https://arxiv.org/abs/2611.07707](https://arxiv.org/abs/2611.07707)

    提出CAMP算法，通过感知LLM推理上下文来优化3D-IMC芯片上的权重映射与放置，提升矩阵向量乘法的并发执行能力和计算资源利用率。

    

    大语言模型由于需要加载大型权重矩阵和键值（KV）缓存而产生严重的内存访问开销。模拟存内计算（AIMC）是一种新兴的解决方案，它能够直接在存储于内存中的静态权重上执行计算，从而大幅降低能耗和延迟。3D-IMC进一步将内存阵列堆叠成垂直层级，增加了容量，但也带来了新的挑战：权重在内存中的放置方式决定了哪些矩阵向量乘法（MVM）可以并发执行，进而影响延迟和吞吐量。3D-IMC权重映射的并发性问题在现有文献中仍未得到充分研究，导致3D-IMC芯片上计算资源的利用欠佳。在此背景下，我们提出了CAMP（上下文感知映射与放置），这是一种面向基于3D-IMC的LLM加速器的推理上下文感知权重映射算法。CAMP能够捕获由此产生的潜在资源争用……

    arXiv:2611.07707v1 Announce Type: cross  Abstract: Large Language Models (LLMs) incur severe memory-access overheads due to the cost of loading large weight matrices and key-value (KV) caches. Analog In-Memory Computing (AIMC) is an emerging solution that enables computation directly on static weights stored in memory, substantially reducing energy and latency. 3D-IMC further stacks memory arrays into vertical tiers, increasing capacity but revealing a new challenge: how weights are placed in memory determines which matrix-vector multiplications (MVMs) can execute concurrently, affecting latency and throughput. The concurrency aspect of 3D-IMC weight-mapping remains underexplored in the literature, leading to sub-optimal utilization of compute resources on 3D-IMC chips. In this context we propose CAMP (Context-Aware Mapping and Placement), an inference-context-aware weight mapping algorithm for 3D-IMC-based LLM accelerators. CAMP captures the potential resource contention resulting fro
    
[^96]: 迈向通用图级别异常检测

    Towards Generalist Graph-Level Anomaly Detection

    [https://arxiv.org/abs/2611.07706](https://arxiv.org/abs/2611.07706)

    提出GenGLAD，一种通用图级别异常检测方法，通过学习单一统一模型，在极少目标域监督下实现跨多个领域的异常图检测。

    

    图级别异常检测（GLAD）旨在识别与大多数图存在显著偏差的图。尽管近期的研究已取得了具有竞争力的性能，但现有方法主要遵循针对特定数据集的范式，在应用于新的应用场景时需要大量的领域内数据和训练成本。这限制了它们在现实应用中的可扩展性和实用性，因为现实中标注数据稀缺，且多样化的领域不断涌现。在本文中，我们研究了通用GLAD问题，该问题旨在学习一个单一的统一模型，能够在对目标数据进行最少监督的情况下检测跨多个领域的异常图。这一问题带来了关键挑战，包括从异构图数据集中学习可迁移的模式，以及在有限监督下进行有效的适应。为了应对这些挑战，我们提出了GenGLAD，一种通用的GLAD方法，它利用……

    arXiv:2611.07706v1 Announce Type: new  Abstract: Graph-level anomaly detection (GLAD) aims to identify graphs that deviate significantly from the majority. While recent advances have achieved competitive performance, existing approaches mainly follow a dataset-specific paradigm, requiring substantial in-domain data and training cost when applied to new application scenarios. This limits their scalability and practicality in real-world applications where labeled data is scarce and diverse domains are continuously emerging. In this paper, we investigate the problem of generalist GLAD, which seeks to learn a single unified model capable of detecting anomalous graphs across multiple domains with minimal supervision on the target data. This problem introduces key challenges, including learning transferable patterns from heterogeneous graph datasets and effective adaptation under limited supervision. To address these challenges, we propose GenGLAD, a generalist GLAD approach that leverages d
    
[^97]: InfMasking：协同多模态学习的统一变分框架

    InfMasking: A Unified Variational Framework for Synergistic Multimodal Learning

    [https://arxiv.org/abs/2611.07702](https://arxiv.org/abs/2611.07702)

    提出了 InfMasking，一个基于无限随机掩码的统一变分多模态学习框架，通过在融合前随机掩码各模态特征，并要求所有被掩码的融合表示与参考目标（无监督时为完整融合视图，有监督时为标签类别代理）保持一致，从而迫使模型充分利用模态间的协同信息，而非依赖主导模态或共享信息。

    

    多模态模型被期望能够利用协同信息，即只有在同时观察多个模态时才能获得的信息。然而在实践中，融合模块可以通过依赖某个主导模态或模态间共享的信息来拟合训练目标，模态之间的协同性往往未被利用。我们提出了 InfMasking，一个基于无限掩码的统一多模态学习框架。在融合之前，InfMasking 随机掩码每个模态的特征；每个采样的掩码会产生一种不同的跨模态证据部分组合，以及对应模态交互的融合表示。所有被掩码的融合表示都需要与参考目标保持一致：当训练是无监督时，参考目标为未掩码的融合视图；当标签可用时，参考目标为任务标签的类别代理。在多样的部分组合上维持这种一致性促使模型（原文在此处截断）。

    arXiv:2611.07702v1 Announce Type: cross  Abstract: Multimodal models are expected to exploit synergistic information, that is, information available only when multiple modalities are observed together. In practice, however, a fusion module can fit the training objective by relying on a dominant modality or on information that modalities share, and the synergy between modalities is often left unused. We propose InfMasking, a unified multimodal learning framework based on Infinite Masking. Before fusion, InfMasking randomly masks the features of each modality; each sampled mask produces a different partial combination of cross-modal evidence and a fused representation of the corresponding modality interactions. All masked fused representations are required to remain consistent with a reference target: the unmasked fused view when training is unsupervised, and the class proxy of the task label when labels are available. Maintaining this consistency across diverse partial combinations disc
    
[^98]: 直积空间中的等变深度学习用于电子结构哈密顿量预测

    Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction

    [https://arxiv.org/abs/2611.07695](https://arxiv.org/abs/2611.07695)

    提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。

    

    大多数现有的用于哈密顿量预测的等变架构是在不可约球张量的直和空间中传播特征，而目标哈密顿量本质上是局部轨道和自旋基的直积表示中的一个算符块。这种表示不匹配割裂了连续的哈密顿量块，模糊了对轨道耦合的直接操作，并将特征组织成不规则的角动量分量，这些分量需要大量变换，且难以与高效的硬件执行和内存管理相适配。为了同时解决表示不匹配和计算瓶颈问题，我们提出了DP-Learner，一个在直积空间中直接对类哈密顿量算符块进行消息传递的等变学习框架。在理论上，我们提出了基于李生成子的方向嵌入以及不变加权的左右乘积与收缩运算……

    arXiv:2611.07695v1 Announce Type: cross  Abstract: Most existing equivariant architectures for Hamiltonian prediction propagate features in direct-sum spaces of irreducible spherical tensors, whereas the target Hamiltonian is naturally an operator block in a direct-product representation over local orbital and spin bases. This representation mismatch fragments contiguous Hamiltonian blocks, obscures direct manipulation of orbital couplings, and organizes features into irregular angular-momentum components that require numerous transformations and are poorly aligned with efficient hardware execution and memory management. To address both the representation mismatch and computational bottleneck, we propose DP-Learner, an equivariant learning framework that performs message passing directly on Hamiltonian-like operator blocks in the direct-product space. Theoretically, we propose Lie-generator-based directional embedding and invariantly weighted left--right products and contractions that 
    
[^99]: MCPHallu：面向MCP智能体的推理、执行与记忆幻觉基准测试

    MCPHallu: Benchmarking Reasoning, Execution, and Memory Hallucinations in MCP Agents

    [https://arxiv.org/abs/2611.07689](https://arxiv.org/abs/2611.07689)

    该论文提出MCPHallu基准，通过在358个任务中内置结构性触发器，沿推理、执行与记忆三个维度，在36个生产级MCP服务器上实现对四种幻觉失败子类型（分支坍缩、不可达目标、工具误用、上下文遗忘）的子类型级诊断。

    

    大语言模型智能体正越来越多地通过模型上下文协议进行部署，该协议使智能体能够在运行时发现并调用跨多个服务器的工具。现有的MCP基准主要衡量任务完成情况；即使它们对失败进行诊断，也是在执行之后才进行，而非将特定的推理、执行和记忆失败风险内置到任务构建之中。我们提出了MCPHallu，这是一个沿推理、执行和记忆三个维度诊断MCP环境中智能体易产生幻觉失败的基准。MCPHallu包含跨五个能力领域的358个任务，并在隔离容器中的36个生产级MCP服务器上运行。每个任务都设计有结构性触发器，使四种失败子类型之一成为主导失败风险：分支坍缩、不可达目标、工具误用或上下文遗忘。这一设计使子类型级别的分析成为可能，而不仅仅依赖聚合的任务成功率。我们评估……

    arXiv:2611.07689v1 Announce Type: new  Abstract: Large language model agents are increasingly deployed through the Model Context Protocol (MCP), which allows them to discover and invoke tools across multiple servers at runtime. Existing MCP benchmarks primarily measure task completion; where they diagnose failures, they do so after execution rather than building specific reasoning, execution, and memory failure risks into task construction.   We introduce MCPHallu, a benchmark for diagnosing hallucination-prone agent failures in MCP environments along three axes: reasoning, execution, and memory. MCPHallu contains 358 tasks across five capability domains and runs on 36 production MCP servers in isolated containers. Each task is designed with a structural trigger that makes one of four subtypes the dominant failure risk: Branch Collapse, Unreachable Goal, Tool Misuse, or Context Forgetting. This design enables subtype-level analysis rather than aggregate task success alone.   We evaluat
    
[^100]: ReferClaw：为自我中心式智能助手消解用户意图歧义

    ReferClaw: Disambiguating User Intent for Egocentric Assistance

    [https://arxiv.org/abs/2611.07683](https://arxiv.org/abs/2611.07683)

    ReferClaw提出一个多智能体交互框架，通过在回答前主动提出澄清问题并借助成本感知策略来消解自我中心式流式问答中用户意图的歧义，从而生成与用户真实意图对齐的答案。

    

    自我中心式AI助手的用户常常提出简短而模糊的问题，其指代对象无法仅凭视频和对话历史唯一确定。例如，“我把它放在哪儿了？”可能指代之前处理过的多个物体。现有的视频问答系统通常直接选定单一解释进行回答，容易给出看似合理但与用户真实意图不符的答案。我们提出ReferClaw，一个智能体框架，它将自我中心式流式问答建模为一个交互过程：助手在回答之前通过提出澄清问题来明确用户意图。三个协作的智能体分别负责识别候选解释、请求澄清，以及基于已明确的意图和视觉证据生成答案。一种有界的、成本感知的策略在预期信息增益与用户负担之间取得平衡，同时限制澄清轮次和问题复杂度。ReferClaw通过意图特定的技能更新和过程奖励从交互中学习。

    arXiv:2611.07683v1 Announce Type: cross  Abstract: Users of egocentric AI assistants often ask short, ambiguous questions whose intended referents cannot be uniquely resolved from video and dialogue history. For example, "Where did I leave it?" may refer to several previously handled objects. Existing VideoQA systems typically commit to a single interpretation, risking plausible but intention-misaligned answers. We introduce ReferClaw, an agentic framework that formulates egocentric streaming QA as an interactive process: the assistant asks clarifying questions to resolve user intent before answering. Three cooperating agents identify candidate interpretations, request clarification, and generate answers grounded in the resolved intent and visual evidence. A bounded, cost-aware policy balances expected information gain against user effort while limiting clarification turns and question complexity. Referclaw learns from interaction through intent-specific skill updates and process-rewar
    
[^101]: 弱凸优化的随机并行复杂性

    Randomized Parallel Complexity of Weakly Convex Optimization

    [https://arxiv.org/abs/2611.07675](https://arxiv.org/abs/2611.07675)

    本文证明了在高维空间中即使随机算法每轮进行多项式数量的并行查询也无法超越标准次梯度方法的速率，但在一维情形下利用实数的序结构，仅用函数值的确定性算法即可实现对数轮复杂度，从而揭示了维度对并行查询收益的决定性作用。

    

    我们研究了寻找非光滑弱凸函数近似驻点所需的最小查询轮数，该复杂性以Moreau包络的梯度来度量，其中每一轮允许进行多个同时查询。在所述参数范围内，标准次梯度速率在足够高的维度中是最优的，即使对于每轮进行多项式数量并行查询的随机算法也是如此。随后，我们展示了实数的序结构如何实现更快的搜索：一个仅使用函数值的确定性算法即可达到对数轮复杂度，且通过并行查询可进一步降低复杂度。即使每次查询返回完整的次微分集合，匹配的下界依然成立。这些结果共同阐明了维度在决定弱凸优化中并行查询收益方面所起的作用。

    arXiv:2611.07675v1 Announce Type: cross  Abstract: We study the minimum number of query rounds needed to find approximate stationary points of nonsmooth weakly convex functions, measured by the gradient of the Moreau envelope. Each round permits multiple simultaneous queries. In the stated parameter regime, the standard subgradient rate is optimal in sufficiently high dimensions, even for randomized algorithms making polynomially many parallel queries per round. We then show how the order structure of the real line enables faster search: a deterministic algorithm using only function values achieves logarithmic round complexity, with a further reduction through parallel queries. Matching lower bounds hold even when each query returns the entire subdifferential. Together, these results clarify the role of dimension in determining the benefits of parallel queries for weakly convex optimization.
    
[^102]: 谁应该知道什么？面向多智能体LLM经济的实证信息设计

    Who Should Know What? Empirical Information Design for Multi-Agent LLM Economies

    [https://arxiv.org/abs/2611.07667](https://arxiv.org/abs/2611.07667)

    该论文提出“实证信息设计”实验框架，通过在由LLM智能体构成的经济市场中系统比较不同的信息披露策略，发现更详细的信息透明度并不总能改善集体结果，其效果高度依赖于具体模型。

    

    多智能体LLM系统的性能不仅取决于底层模型的能力，还取决于信息共享策略，即决定每个智能体观察到什么信息以及以何种详细程度的规则。但是，为AI智能体提供更详细的信息是否总能改善集体结果？或者透明度是否会悖论地降低效率？为了研究这个问题，我们提出了实证信息设计，这是一个实验框架，在保持交互规则和模型固定不变的情况下，系统地比较由LLM智能体组成的经济市场中不同的信息披露策略。我们在多资源交易、双边议价和双向拍卖场景中评估了十个前沿LLM，并使用真实效用数据将已识别的机会与已实现的结果解耦。我们发现，更精细的信息披露会产生高度依赖于模型的结果。在交易中，揭示精确估值而非序数偏好

    arXiv:2611.07667v1 Announce Type: new  Abstract: The performance of multi-agent LLM systems depends not only on the capabilities of the underlying models but also on information-sharing policies, which govern what information each agent observes and at what level of detail. But does providing AI agents with more detailed information always improve collective outcomes, or can transparency paradoxically degrade efficiency? To investigate this, we introduce empirical information design, an experimental framework that systematically compares disclosure policies across economic markets populated by LLM agents, while holding interaction rules and models fixed. We evaluate ten frontier LLMs in multi-resource trade, bilateral bargaining, and double auctions, using ground-truth utilities to decouple identified opportunities from realized outcomes. We find that finer information disclosure yields highly model-dependent results. In trade, revealing exact valuations rather than ordinal preference 
    
[^103]: 面向低功耗机器人自主的能耗感知开放词汇检测调度

    Energy-Aware Open-Vocabulary Detection Scheduling for Low-Power Robotic Autonomy

    [https://arxiv.org/abs/2611.07663](https://arxiv.org/abs/2611.07663)

    本文提出一种能耗感知的开放词汇目标检测调度器，利用里程计、地图变化和检测器置信度在Jetson Orin Nano上动态切换不同功耗的检测模式，在真实导航试验中将任务成功率从7/10提升至8/10，同时将平均功耗降低5.2%、检测调用频率降低34.0%。

    

    开放词汇目标检测（OVD）使机器人能够搜索运行时的目标物体查询，但固定频率的推理会在空闲运动、地图重访或稳定视角期间，将嵌入式计算资源浪费在低价值帧上。本文提出了一种面向低功耗移动机器人的能耗感知OVD调度器。该调度器利用里程计、地图变化和检测器置信度，在Jetson Orin Nano上于低、中、高功耗的YOLO-World-S模式之间进行切换。我们在AgileX LIMO机器人上通过40次真实导航试验评估了能耗与检测质量的权衡：包括两个目标物体、每个目标-策略对五种放置位置以及四种感知策略。在瓶子和杯子的试验中，该调度器取得了8/10的任务成功率，相比之下各固定频率基线为7/10；同时相对于8 Hz高频率基线，将Jetson平均输入功耗（一种Jetson侧功耗指标）降低了5.2%，并将实际OVD调用频率降低了34.0%。它并不最小化任务窗口内的Je……

    arXiv:2611.07663v1 Announce Type: cross  Abstract: Open-vocabulary object detection (OVD) lets robots search for run-time object queries, but fixed-rate inference spends embedded compute on low-value frames during idle motion, map revisits, or stable viewpoints. This paper presents an energy-aware OVD scheduler for low-power mobile robots. The scheduler uses odometry, map change, and detector confidence to switch among low-, mid-, and high-power YOLO-World-S modes on a Jetson Orin Nano. We evaluate the energy-quality trade-off on an AgileX LIMO using 40 real navigation trials: two target objects, five placements per target-policy pair, and four perception policies. Across bottle and cup trials, the scheduler records 8/10 mission success compared with 7/10 for each fixed-rate baseline, while reducing mean Jetson input power (a Jetson-side power-draw metric) by 5.2% relative to an 8 Hz high-rate baseline and lowering achieved OVD call rate by 34.0%. It does not minimize mission-window Je
    
[^104]: PharmAgents2：迈向自主的端到端药物发现

    PharmAgents2: Towards Autonomous End-to-End Drug Discovery

    [https://arxiv.org/abs/2611.07653](https://arxiv.org/abs/2611.07653)

    该论文提出了PharmAgents2——一个将Paper2Skill知识提取与自我扩展式并行探索相结合的自主端到端药物发现系统，在新基准PharmAgents-Bench v0.1及三项真实世界研究中均取得了领先表现。

    

    药物发现需要在广阔的研究空间中进行持续的科学决策。我们提出了PharmAgents2，一个用于端到端药物发现的自主系统，它将知识增强的决策与有组织的大规模探索相结合。Paper2Skill从公开来源中提取可迁移的科学决策知识，并与专家开发的技能共同支持研究决策。自我扩展发现通过并行智能体来扩展探索，这些智能体继承研究上下文和决策能力，涵盖靶点、科学假设和发现策略。我们引入了PharmAgents-Bench v0.1，包含九个决策场景中的121个问题以及三个端到端任务，并采用专家定义的过程评估标准进行评估。PharmAgents2在大多数决策类别中优于所评估的基线方法，并在所有三个端到端任务上获得了最高的过程得分。在三项真实世界研究中，系统……

    arXiv:2611.07653v1 Announce Type: new  Abstract: Drug discovery requires continuous scientific decision-making across a vast research space. We introduce PharmAgents2, an autonomous system for end-to-end drug discovery that combines knowledge-enhanced decision-making with organized large-scale exploration. Paper2Skill extracts transferable scientific decision knowledge from public sources and, together with expert-developed skills, supports research decisions. Self-Scaling Discovery expands exploration through parallel agents inheriting research context and decision-making capabilities across targets, scientific hypotheses, and discovery strategies. We introduce PharmAgents-Bench v0.1, comprising 121 questions across nine decision scenarios and three end-to-end tasks assessed with expert-defined process rubrics. PharmAgents2 outperforms evaluated baselines in most decision categories and achieves the highest process scores on all three end-to-end tasks. In three real-world studies, aut
    
[^105]: 不要提示你的智能体，而是装备它们：Skillware——面向逻辑系统的主权、可审计且确定性能力执行的模块化标准

    Don't Prompt Your Agents. Equip Them: Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems

    [https://arxiv.org/abs/2611.07652](https://arxiv.org/abs/2611.07652)

    该论文提出Skillware开放框架，通过模块化、版本化的软件能力包（包含类型化JSON契约、确定性代码、任务指令、数据集和单元测试）取代脆弱的提示工程，将智能体的意图与执行正式分离，实现主权、可审计且确定性的能力执行。

    

    当前的自主逻辑系统依赖于一个脆弱的假设：语言模型应该动态编写代码、猜测API载荷，并通过提示工程来记忆操作约束。在实践中，这种方式会大量消耗上下文窗口、推高API费用，并产生在生产环境中不可预测地失效的脆弱智能体。在本工作中，我们提出了Skillware，一个开放的架构框架和能力注册表，它正式地将智能体“想要做什么”与“任务如何执行”分离开来。Skillware不是用松散的文本指令或被美其名曰“技能”的markdown文件来提示模型，而是为智能体配备模块化、版本化的软件能力包。就像视频游戏中的技能快捷键在固定世界规则下确定性地执行一样，每个能力包都结合了类型化的JSON输入契约、确定性的Python代码、任务指令、捆绑的数据集以及自动化单元测试。进程内适配器将……（原文摘要至此被截断）

    arXiv:2611.07652v1 Announce Type: new  Abstract: Current autonomous logical systems rely on a fragile assumption: that language models should dynamically write code, guess API payloads, and memorize operational constraints through prompt engineering. In practice, this burns massive context windows, inflates API bills, and produces brittle agents that fail unpredictably in production.   In this work, we present Skillware, an open architectural framework and capability registry that formally separates what an agent wants to do from how that task is executed. Instead of prompting models with loose text instructions or markdown files baptized as 'skills', Skillware equips them with modular, versioned software capability bundles. Much like skill hotkeys in video games that execute deterministically under fixed world rules, each bundle combines a typed JSON input contract, deterministic Python code, task directives, bundled datasets, and automated unit tests. In-process adapters translate th
    
[^106]: GraspBench：MuJoCo中面向执行感知的6自由度抓取生成跨机器人评估

    GraspBench: Execution-Aware Cross-Robot Evaluation of 6-DoF Grasp Generation in MuJoCo

    [https://arxiv.org/abs/2611.07651](https://arxiv.org/abs/2611.07651)

    GraspBench是一个在MuJoCo中对6自由度抓取生成流水线进行端到端可执行性评估的跨机器人基准，通过逆运动学过滤、运动规划和固定执行流程，在四种机器人本体和多个物体上系统比较了六种方法，发现GraspGen以73.1%的成功率居首。

    

    现代6自由度（6-DoF）抓取生成器通常使用抓取分数、数据集级指标或单机器人演示进行评估，然而仅有抓取提案的质量并不能决定该抓取能否被特定机器人真正执行。我们提出GraspBench，这是一个基于MuJoCo的受控评估协议，用于在经过机器人特定的逆运动学过滤、运动规划以及固定的“接近-闭合-提升”执行流程之后，对完整的抓取生成流水线进行端到端评估。GraspBench在四种机器人本体、五个诊断物体和一个结构化的桌面姿态网格上评估了六种配置的流水线，每种方法产生540个场景，总计3,240次方法-场景评估。缺失的输出被保守地计为失败，成功与否由各方法自身的执行器按二元提升准则判定。结果显示，GraspGen取得了最高的总体成功率（73.1%），其次是EconomicGrasp（63.5%），而基于机器人和物体条件的分析进一步揭示了……（原文摘要到此截断）

    arXiv:2611.07651v1 Announce Type: cross  Abstract: Modern 6-DoF grasp generators are often evaluated using grasp scores, dataset-level metrics, or single-robot demonstrations, yet proposal quality alone does not determine whether a grasp is executable by a particular robot. We present GraspBench, a controlled MuJoCo protocol for evaluating complete grasp-generation pipelines after robot-specific inverse-kinematics filtering, motion planning, and fixed approach-close-lift execution. GraspBench evaluates six configured pipelines across four robot embodiments, five diagnostic objects, and a structured tabletop pose grid, yielding 540 scenarios per method and 3,240 method-scenario evaluations in total. Missing outputs are conservatively counted as failures, and success follows the binary lift criterion produced by each method's execution runner. GraspGen achieves the highest overall success rate (73.1%), followed by EconomicGrasp (63.5%), while robot- and object-conditioned analyses reveal
    
[^107]: 中间商究竟交付了什么：LLM API中转服务中的回答行为、工具权限与计费用量

    What the Middleman Delivers: Answer Behavior, Tool Permissions, and Billed Usage in LLM API Relays

    [https://arxiv.org/abs/2611.07650](https://arxiv.org/abs/2611.07650)

    本研究对20个提供GPT-6 Astra和Claude Opus 5的第三方LLM API中转服务进行系统性测试，发现约半数低价中转服务的回答行为与官方模型存在差异，并通过检查返回内容、工具权限与计费用量揭示了中转服务在交付质量上的不可靠性。

    

    第三方应用程序接口（API）中转服务以通常更低的价格提供大型语言模型（LLM）的访问。用户依赖这些服务来保障模型质量、指令遵循能力以及准确的计费。然而，现有研究并未澄清中转服务能否同时满足这些期望。我们测试了20个提供GPT-6 Astra和Claude Opus 5的中转服务。首先，我们将知识类问题的回答与批量匹配的官方响应进行比较。在第一轮样本中，分析标记出7/10的低价中转服务和2/10的同价或更高价中转服务存在行为差异。在重新测试35对样本时，按照原始规则有七个服务被新增或撤销标记，而按照匹配的修正族规则则有六个。由于回答比较无法揭示所有交付问题，我们进一步检查了返回内容和工具权限。在针对固定的、非对抗性的代码与配置工作负载筛选的3,332条响应中，未确认存在注入行为，但有四个中转服务返回……

    arXiv:2611.07650v1 Announce Type: cross  Abstract: Third-party application programming interface (API) relays offer access to large language models (LLMs), often at lower prices. Users depend on these services for model quality, instruction compliance, and accurate billing. However, existing studies leave unclear whether relays meet these expectations together. We test 20 relays offering GPT-6 Astra and Claude Opus 5. First, we compare answers to knowledge questions with batch-matched official responses. In our first-round sample, the analysis flags 7/10 low-price and 2/10 same- or higher-price relays for behavioral differences. On retesting 35 pairs, seven gain or lose a flag under the original rules, and six under matched correction families. Because answer comparisons cannot reveal every delivery problem, we examine returned content and tool permissions. No injection is confirmed in 3,332 screened responses to a fixed, non-adversarial code and setup workload, yet four relays return 
    
[^108]: UniGADD：通用生成器无关的深度伪造检测器

    UniGADD: Universal Generator-Agnostic Deepfake Detector

    [https://arxiv.org/abs/2611.07647](https://arxiv.org/abs/2611.07647)

    UniGADD提出了一种两阶段优化的监督学习方法，通过对比损失学习和仅针对真实样本的嵌入空间细化，实现了对已知深度伪造生成器的高精度检测，并对未知生成技术具有强泛化能力。

    

    在本文中，我们提出了通用生成器无关的深度伪造检测器（UniGADD），这是一种新颖的方法，采用监督学习在已知的深度伪造生成器上实现高精度检测，同时对之前未见过的生成器保持稳健的性能。所提出的方法遵循两阶段优化过程：在第一阶段，对比损失促使模型从真实图像和已知伪造图像中学习判别性特征嵌入，从而在训练域内取得强劲性能；在第二阶段，通过促进簇间分离和簇内紧凑性来细化嵌入空间，且该细化过程仅应用于真实样本。这种细化增强了泛化能力，使该方法对未见过的深度伪造生成技术展现出更强的鲁棒性。UniGADD在已知生成器上的准确率与最先进方法相当，同时在未见案例上显著优于它们。

    arXiv:2611.07647v1 Announce Type: cross  Abstract: In this paper, we introduce Universal Generator-Agnostic Deepfake Detector (UniGADD), a novel method that employs supervised learning to achieve high accuracy on known deepfake generators, while maintaining robust performance on previously unseen ones. The proposed approach follows a two-stage optimisation process. In the first stage, a contrastive loss encourages the model to learn discriminative feature embeddings from real and known fake images, resulting in strong performance within the training domain. In the second stage, the embedding space is refined by promoting inter-cluster separation and intra-cluster compactness, applied exclusively to real samples. This refinement enhances generalisability, enabling the method to exhibit improved robustness against unseen deepfake generation techniques. UniGADD achieves accuracy on par with state-of-the-art methods for known generators, while significantly outperforming them on unseen cas
    
[^109]: WaSP：面向无求解器算子学习的Wasserstein数据合成流水线

    WaSP: Wasserstein Data Synthesis Pipeline for Solver-Free Operator Learning

    [https://arxiv.org/abs/2611.07646](https://arxiv.org/abs/2611.07646)

    该论文提出WaSP，一种基于Wasserstein度量的数据合成流水线，通过在人造数据对上训练并逐步缓解训练分布与目标分布之间的偏移，实现了无需求解器的高效算子学习。

    

    神经算子为求解偏微分方程（PDE）提供了快速的代理方法，但训练它们通常需要代价高昂的“输入-解”数据对。两种无需求解器的替代方案包括物理信息学习和基于人造数据的训练。物理信息学习通过最小化PDE残差来逼近解，虽然残差范数必须足够强才能控制解的误差，但更强的范数会增加小批量的方差并阻碍优化。人造数据方法则是通过将PDE算子作用于人工构造的解来构建用于学习的“输入-解”数据对，然而由此得到的训练分布可能与目标分布相互奇异，导致在目标分布上的精度很差。我们提出了Wasserstein数据合成流水线（WaSP），它在人造的“输入-解”数据对上进行训练，同时逐步缓解分布偏移。WaSP通过将学习到的算子应用于一条路径上的输入来合成新的数据对……

    arXiv:2611.07646v1 Announce Type: cross  Abstract: Neural operators offer fast surrogates for solving PDEs, but training them typically requires expensive input-solution pairs. Two solver-free alternatives include physics-informed learning and training on manufactured data. Physics-informed learning approximates solutions by minimizing PDE residuals. While the residual norm must be strong enough to control solution error, stronger norms increase mini-batch variance and hinder optimization. Manufactured data constructs input-solution pairs for learning by applying the PDE operator to artificial solutions. However, the resulting training distribution can be mutually singular with the target distribution, yielding poor accuracy on the target distribution. We introduce the Wasserstein Data Synthesis Pipeline (WaSP), which trains on manufactured input-solution pairs while gradually mitigating the distribution shift. WaSP synthesizes new pairs by applying the learned operator to a path of in
    
[^110]: 基于内外先验融合与对齐的神经图像压缩

    Neural Image Compression with Internal-External Prior Fusion and Alignment

    [https://arxiv.org/abs/2611.07644](https://arxiv.org/abs/2611.07644)

    该论文提出内外先验融合与对齐（IE-PFA）上下文模型，利用共享可学习字典作为外部先验，与图像内部先验融合并对齐，在单网络内支持64级可变码率编码，实现最先进的率失真性能。

    

    现有用于学习型图像压缩的上下文模型主要依赖当前图像的内部先验，忽略了自然图像之间共享的常见模式和典型纹理。为解决这一局限，我们提出了一种建立在四叉树空间-通道联合建模之上的内外先验融合与对齐上下文模型，其中由编码器和解码器共享的可学习字典充当外部先验。具体而言，基于RWKV的空间-通道先验融合模块以线性复杂度在空间和通道维度上高效融合内部与外部先验，而仅用于训练的自蒸馏先验对齐机制在完成对齐的同时不引入任何推理开销。此外，我们的模型在单一网络中即可支持跨越64个质量级别的可变码率编码，并通过自定义CUDA算子实现加速。在七个基准数据集上的实验表明，该方法取得了最先进的率失真性能。

    arXiv:2611.07644v1 Announce Type: cross  Abstract: Existing context models for learned image com- pression rely primarily on internal priors from the current image, overlooking common patterns and typical textures shared across natural images. To address this limitation, we propose an Internal-External Prior Fusion-Alignment (IE-PFA) context model built upon quadtree spatial-channel joint modeling, where a learnable dictionary shared by the encoder and decoder serves as an external prior. Specifically, an RWKV-based Spatial- Channel Prior Fusion (RSCPF) module efficiently fuses inter- nal and external priors along spatial and channel dimensions with linear complexity, while a training-only Self-Distilled Prior Alignment (SDPA) mechanism aligns them without inference overhead. Moreover, our model supports variable-rate coding over 64 quality levels within a single network and is accelerated by custom CUDA operators. Experiments on seven benchmarks demonstrate state-of-the-art rate-disto
    
[^111]: 超越指令覆盖：基于任务篡改攻击的间接提示注入基准测试

    Beyond Instruction Override: Benchmarking Indirect Prompt Injection with Task-Tampering Attacks

    [https://arxiv.org/abs/2611.07640](https://arxiv.org/abs/2611.07640)

    该论文提出以“权限”而非“指令措辞”来重新定义间接提示注入，并构建了涵盖三个工具使用环境、共90个案例的基准测试，专门评估通过篡改正在进行任务的决策结果（而非替换任务）的任务篡改攻击。

    

    间接提示注入（IPI）基准测试主要评估的是那些将使用工具的智能体重定向到攻击者所选任务的植入指令。然而，攻击者控制的内容也可以篡改正在进行中的任务的结果。例如，买家的争议证据可以引用虚假的退款政策来夸大退款金额，而无需改变任务本身。我们根据权威性而非类指令的措辞来定义IPI：当植入内容试图影响其来源无权决定的事项时，即构成攻击。我们引入了一个使这一边界明确化的基准测试：在三个工具使用环境中共90个经过验证的案例中，用户请求明确指定了解决目标决策的权威来源。每次攻击都是篡改良性任务所需的决策，而不是替换该任务本身。每个案例都包含具有相同有害结果、但针对智能体决策中最多六个不同部分的攻击。

    arXiv:2611.07640v1 Announce Type: cross  Abstract: Indirect prompt-injection (IPI) benchmarks predominantly test planted instructions that redirect a tool-using agent toward an attacker-chosen task. Yet attacker-controlled content can instead tamper with the outcome of the task already in progress. A buyer's dispute evidence can cite a false refund policy to inflate a refund without changing the task. We define IPI by authority rather than instruction-like wording: planted content constitutes an attack when it attempts to influence a decision that its source is not authorized to determine. We introduce a benchmark that makes this boundary explicit: in each of 90 validated cases across three tool-use environments, the user request specifies the source that resolves the targeted decision. Every attack tampers with a decision required by the benign task rather than replacing that task. Each case has attacks with the same harmful outcome that target up to six parts of the agent's decision 
    
[^112]: 衡量基于验证器的强化学习相对于监督微调的边际价值

    Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning

    [https://arxiv.org/abs/2611.07639](https://arxiv.org/abs/2611.07639)

    该论文构建了一个确定性的GPU/LLM训练故障诊断基准，发现仅用1,200个黄金示范的监督微调即可近乎解决该基准，而从基础模型出发的GRPO强化学习几乎无增益，从而质疑了基于验证器的强化学习相对于监督微调的边际价值。

    

    arXiv:2611.07639v1 公告类型：cross 摘要：可验证奖励的强化学习（RLVR）通常被评估为监督微调（SFT）的替代方案，或作为其后续阶段，其中隐含的假设是RL阶段本身能贡献额外的增益。我们在一个受控环境中直接测量了这一贡献，在该环境中两个阶段都可以完整运行并进行精确比较。我们构建了gpu-train-debug，这是一个包含1,877个合成GPU和LLM训练失败诊断任务的确定性基准，涵盖30种任务原型，由带部分得分的程序化验证器进行评分，并经过校准使中等层级模型的得分落在0.60至0.75区间内。一项试点实验带来了意外发现：1,200个黄金示范几乎解决了该基准（SFT在困难分割上得分为0.785），而从基础模型开始的15步GRPO训练几乎没有改变结果（0.314，对比基础模型的0.311）。因此，我们在完整实验运行之前以书面形式修改了研究方案，提出了更尖锐的问题：基于验证器的RL在此基础之上究竟有何额外贡献。

    arXiv:2611.07639v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards (RLVR) is usually evaluated as an alternative to supervised fine-tuning (SFT), or as a stage that follows it, with the implicit assumption that the RL stage contributes gains of its own. We measure that contribution directly, in a controlled setting where both stages can be run to completion and compared exactly. We built gpu-train-debug, a deterministic benchmark of 1,877 synthetic GPU and LLM training failure diagnosis tasks across 30 archetypes, graded by a programmatic verifier with partial credit, and calibrated so that a mid-tier model scores in a 0.60 to 0.75 band. A pilot produced a surprise: 1,200 gold demonstrations nearly solved the benchmark (SFT 0.785 on the hard split), while 15 steps of GRPO from the base model did not move it (0.314 vs 0.311 base). We therefore amended the study, in writing and before the full run, to ask the sharper question: what does verifier-based RL ad
    
[^113]: 个人数据脱敏：正则表达式、Presidio与混合设计在速度和准确性上的比较

    Redacting Personal Data: A Speed and Accuracy Comparison of Regex, Presidio and Hybrid Designs

    [https://arxiv.org/abs/2611.07633](https://arxiv.org/abs/2611.07633)

    本文系统比较了四种可用于LLM管道的PII脱敏设计方案（纯正则、Presidio、以及两种混合设计）在速度与准确性上的表现差异。

    

    许多企业项目现在将自由文本传递给大型语言模型（LLM），而这些文本通常包含个人身份信息（PII），例如姓名、电话号码、电子邮箱地址、家庭住址和出生日期。一旦提示词被发送出去，这些数据就会被复制到服务提供商的系统、应用程序日志和可观测性追踪记录中。在调用之前对PII进行脱敏是一种显而易见的控制措施，但团队往往缺乏关于这一措施在延迟上付出多少代价、又会漏掉哪些数据的量化数据。本文比较了任何LLM处理管道都可以采用的四种脱敏设计：纯正则表达式脱敏器、运行在spaCy模型之上的Microsoft Presidio、由Presidio仅识别人名而正则表达式处理结构化标识符的混合设计，以及直接调用spaCy命名实体组件的快速混合设计。这四种设计均共享一个已知实体规则（即移除系统已知正在处理的那个人），并采用一致的占位符方案，从而……

    arXiv:2611.07633v1 Announce Type: new  Abstract: Many enterprise projects now pass free text to large language models (LLMs), and that text often contains personally identifiable information (PII) such as names, phone numbers, email addresses, home addresses and dates of birth. Once a prompt is sent, this data is copied into provider systems, application logs and observability traces. Redacting PII before the call is an obvious control, yet teams seldom have numbers on what it costs in latency or what it lets through. This paper compares four redaction designs that any LLM pipeline can adopt: a pure regular-expression (regex) redactor, Microsoft Presidio running on a spaCy model, a hybrid in which Presidio finds only person names while regex handles structured identifiers, and a fast hybrid that calls spaCy's named-entity component directly. All four share a known-entity rule, which removes the person the system already knows it is handling, and a consistent placeholder scheme, so the 
    
[^114]: 表面之下：谈论智能体演化时我们忽视了什么

    Beneath the Surface: What We Overlook When We Talk About Agent Evolution

    [https://arxiv.org/abs/2611.07624](https://arxiv.org/abs/2611.07624)

    该论文系统评估了四种智能体框架演化方法，发现更高的基准分数并不等于可靠、高效且可迁移的改进——演化伴随着任务覆盖的得失权衡与部署成本上升，且得分最高的智能体并非最优选择。

    

    智能体框架演化能够在不更新模型权重的情况下实现递归自我改进（RSI）。我们研究了更高的基准测试分数是否能带来可靠、高效且可迁移的改进。我们在不同框架、通用任务和编码任务上评估了四种方法。通过重复评估、受控智能体扩展、组件干预和准入回放，我们揭示了反复出现的模式。所有十六个演化后的智能体都获得了之前未覆盖的任务，同时也丢失了之前已覆盖的任务。在一个通用任务案例中，尽管丢失了十个之前已覆盖的任务，可靠性仍提升了5.2分。大多数可靠性提升同时也会增加部署成本。具体的改动通常优于通用的提醒，但当它们组合时其收益可能会发生变化。得分最高的智能体在我们的四个任务池中均未产生最大的平均改进，而更严格的准入机制可能会丢弃有用的替代方案。迁移会选择性地保留收益：任务覆盖率通常……（原文摘要在此处截断）

    arXiv:2611.07624v1 Announce Type: new  Abstract: Agent harness evolution enables recursive self-improvement (RSI) without updating model weights. We examine whether higher benchmark scores yield reliable, efficient, and transferable improvement. We evaluate four methods across harnesses, general and coding tasks. Repeated evaluations, controlled agent expansion, component interventions, and admission replays reveal recurring patterns. All sixteen evolved agents gain previously uncovered tasks while losing previously covered ones. In one general-task case, reliability rises by 5.2 points despite losing ten previously covered tasks. Most reliability gains also increase deployment cost. Specific changes usually outperform generic reminders, but their benefits can change when combined. The highest-scoring agent does not yield the largest mean improvement in any of our four pools, and stricter admission can discard useful alternatives. Transfer selectively preserves gains: coverage usually 
    
[^115]: 强者愈强：生成式搜索中的品牌偏见审计

    The Popular Get More Popular: Auditing Brand Bias in Generative Search

    [https://arxiv.org/abs/2611.07614](https://arxiv.org/abs/2611.07614)

    生成式搜索存在系统性品牌偏见：头部热门品牌被引用的频率是平均品牌的1.4至2.8倍，这种偏见源自模型参数记忆且无法通过外部网络检索缓解。

    

    从传统搜索向生成式搜索的转变从根本上改变了线上产品发现的方式。由于对话式推荐将众多选择综合为少数几个直接答案，对特定提供商的微妙系统性偏好可能产生深远的商业影响。在这项工作中，我们从两个不同的维度对这一现象进行审计：(i) 品牌流行度偏见，即相对于长尾替代品牌，对市场主导品牌的系统性偏好；(ii) 品牌熟悉度偏见，即实体凭借其编码在大语言模型参数记忆中的信息相对于不为人熟知的竞争对手所拥有的排名优势。首先，通过评估各大主流引擎上的商业查询，我们观察到持续存在的品牌流行度偏见：在1,176个头部市场参与者样本中，位于最高五分位的主导品牌被引用的频率是平均品牌的1.4倍至2.8倍，这种偏见已经存在于参数记忆之中，而外部网络检索无法缓解这一问题。

    arXiv:2611.07614v1 Announce Type: cross  Abstract: The shift from traditional to generative search has fundamentally transformed online product discovery. Because conversational recommendations synthesize choices into a few direct answers, subtle systemic preferences toward specific providers can exert profound commercial impacts. In this work, we audit this phenomenon through two distinct dimensions: (i) brand popularity bias, the systematic preference for market-dominant entities over long-tail alternatives, and (ii) brand familiarity bias, the ranking advantage an entity encoded in an LLM's parametric memory holds over an unfamiliar competitor. First, evaluating commercial queries across major engines, we observe persistent brand popularity bias: dominant brands in the top quintile are cited 1.4x to 2.8x more frequently than the average brand within a sample of 1,176 top market players, already present in parametric memory that external web retrieval fails to mitigate. Second, throu
    
[^116]: LoRA微调的嵌入空间指纹：仅凭模型产物恢复训练数据领域

    Embedding-Space Fingerprinting of LoRA Fine-Tunes: Recovering the Training-Data Domain from the Model Artifact Alone

    [https://arxiv.org/abs/2611.07610](https://arxiv.org/abs/2611.07610)

    本文提出嵌入空间指纹方法，证明无需接触训练数据，仅凭LoRA微调模型的产物（权重增量、内部激活或生成文本）即可识别其训练数据的领域，其中灰盒激活特征在留一数据集协议下达到61%的准确率，远高于20%的随机水平。

    

    当基础语言模型通过低秩适配（LoRA）进行微调时，训练数据的领域信息会有多少泄露到最终产物中？本文研究一个简单的分类器能否在从未观察训练数据的情况下，仅凭模型产物（权重增量、固定探针集上的内部激活，或生成文本）恢复微调模型的所属领域。研究使用一个基础模型（Qwen3.5-0.8B），在五个领域（医学、代码、数学、非英语文本、叙事散文）的18个公开数据集上生成了36个LoRA微调模型，并在访问权限递减的三个层级上提取了三类特征：白盒权重、灰盒激活和黑盒输出。采用在任何结果产生之前就已冻结的留一数据集（LODO）协议，测试能否识别在先前未见过的数据集上训练的微调模型的领域。结果表明，领域信息可以通过灰盒激活以远高于随机水平（20%）的准确率（61%）恢复（摘要在此处被截断）。

    arXiv:2611.07610v1 Announce Type: new  Abstract: When a base language model is fine-tuned with Low-Rank Adaptation (LoRA), how much of the training data domain leaks into the resulting artifact? This paper studies whether a simple classifier can recover the domain of a fine-tune given only the model artifact (weight deltas, internal activations on a fixed probe set, or generated text) without ever observing the training data. Using one base model (Qwen3.5-0.8B), 36 LoRA fine-tunes are produced over 18 public datasets in five domains (medical, code, mathematics, non-English text, narrative prose), and three feature families are extracted at decreasing levels of access: white-box weights, gray-box activations, and black-box outputs. A leave-one-dataset-out (LODO) protocol, frozen before any result was seen, tests whether the domain of a fine-tune trained on a previously unseen dataset can be identified. The domain is recoverable well above chance (20%) from gray-box activations (61%) and
    
[^117]: FisherDistill：基于闭式Fisher判别器的扩散分布蒸馏

    FisherDistill: Diffusion Distribution Distillation via Closed-Form Fisher Critics

    [https://arxiv.org/abs/2611.07604](https://arxiv.org/abs/2611.07604)

    提出FisherDistill方法，通过冻结教师模型的中间表示计算闭式Fisher判别方向，无需训练对抗判别器即可对蒸馏后的扩散生成器进行轻量级的分布精细化。

    

    扩散蒸馏领域的最新进展使得高质量图像生成仅需一次或少数几次网络函数评估（NFE）即可实现。尽管这些方法效率很高，但经过激进蒸馏的生成器与多步教师模型之间往往仍存在残余的分布差距，这促使近期方法引入基于学习判别器的对抗性精细化手段。在本工作中，我们探究这样的精细化是否必须依赖学习得到的对抗判别器。我们提出了FisherDistill，一种基于Fisher线性判别分析的简单替代方案。利用冻结教师模型的中间表示，我们估计真实样本与生成样本的一阶和二阶统计量，并推导出一个闭式的Fisher方向，该方向相对于类内变异能够最大程度地分离两者的特征分布。这一方向提供了一种轻量级的精细化信号，可作为原始蒸馏目标的补充。

    arXiv:2611.07604v1 Announce Type: cross  Abstract: Recent advances in diffusion distillation have enabled high-quality image generation in only one or a few network function evaluations (NFEs). Despite their efficiency, aggressively distilled generators often retain a residual distribution gap from their multi-step teachers, motivating recent methods to introduce adversarial refinement with a learned discriminator. In this work, we ask whether such refinement necessarily requires a learned adversarial discriminator. We introduce \textbf{FisherDistill}, a simple alternative based on Fisher linear discriminant analysis. Using intermediate representations from the frozen teacher, we estimate the first- and second-order statistics of real and generated samples and derive a closed-form Fisher direction that maximally separates their feature distributions relative to within-class variation. This direction provides a lightweight refinement signal that complements the original distillation obj
    
[^118]: 先对齐再聚合：异构秩下的基一致联邦 LoRA

    Align Before Aggregation: Basis-Consistent Federated LoRA under Heterogeneous Ranks

    [https://arxiv.org/abs/2611.07598](https://arxiv.org/abs/2611.07598)

    提出 FedAbA 框架，在聚合前先提取并对齐各客户端 LoRA 因子的局部基，解决了异构秩联邦微调中因基不一致导致的参考前缀不稳定问题。

    

    在客户端能力异构的情况下对大语言模型进行联邦微调正变得日益重要，而采用异构秩的低秩适应使这一目标切实可行。现有的服务器端聚合器在聚合之前，先由客户端特定的 LoRA 因子 合成局部矩阵更新 ΔW_i。尽管这些矩阵更新位于相同的权重空间 R^{m×n} 中并可以直接平均，但生成它们的 LoRA 因子可能表达了客户端特定的局部基。在典型的“合成—聚合—分解—分发”循环中，这种基不一致性可能导致服务器端重新分解后，分发给低秩客户端的秩为 r_i 的参考前缀不稳定。在这项工作中，我们提出了 FedAbA，一个面向异构秩的基一致联邦 LoRA 聚合框架。FedAbA 首先提取每个客户端的局部基，并将其与对应的前缀进行对齐……

    arXiv:2611.07598v1 Announce Type: new  Abstract: Federated fine-tuning of large language models (LLMs) under heterogeneous client capacities is increasingly important, and low-rank adaptation (LoRA) with heterogeneous ranks makes it practical. Existing server-side aggregators compose local matrix updates $\Delta W_i$ from client-specific LoRA factors $(A_i,B_i)$ before aggregating them. Although these matrix updates lie in the same weight space $\mathbb{R}^{m\times n}$ and can be averaged directly, the LoRA factors that generate them may expressed client-specific local basis. In the typical compose--aggregate--factorize--dispatch loop, this basis inconsistency can make the rank-$r_i$ reference prefix dispatched to lower-rank clients unstable after server-side refactorization. In this work, we propose \textbf{FedAbA}, a basis-consistent federated LoRA aggregation framework for heterogeneous ranks. FedAbA first extracts each client's local basis, aligns it to the corresponding prefix of 
    
[^119]: Evolver：将智能体探索转化为不断演进的能力

    Evolver: Turning Agent Exploration into Evolving Capabilities

    [https://arxiv.org/abs/2611.07595](https://arxiv.org/abs/2611.07595)

    Evolver 是一个无需更新基础模型参数的经验驱动框架，它通过结构化的 Gene 资产将智能体探索产生的经验转化为持久可复用的能力，在四个基准测试中显著提升任务成功率并将执行 token 消耗降低约 24%。

    

    智能体探索会产生有价值的程序性知识，然而这些经验大多仍局限于瞬态的执行轨迹中，未能被系统性地复用。我们提出 Evolver，一个经验驱动的框架，它能够在不更新基础模型参数的前提下，将智能体探索转化为持久且可复用的能力。Evolver 以结构化的 Gene 资产为核心，集成了经验获取、检索以及反馈引导的演化机制，从而在现有智能体框架中支持基于经验的条件化执行。我们使用 Claude Code 和 Codex 两种基础模型配置，以及 Terminal-Bench 2.1、SWE-bench Multilingual、SWE-rebench 和 GAIA 四个基准测试对 Evolver 进行了评估。在一个包含 320 组配对执行的诊断集上，Evolver 将任务成功率从 83.13% 提升至 87.50%（提高 4.38 个百分点），同时将记录的任务执行 token 消耗降低了 24.24%。此外，在 265 组成功……（摘要在此处截断）

    arXiv:2611.07595v1 Announce Type: cross  Abstract: Agent exploration produces valuable procedural knowledge, yet much of this experience remains confined to transient execution traces and is not systematically reused. We introduce Evolver, an experience-driven framework that transforms agent exploration into persistent, reusable capabilities without updating foundation model parameters. Built around structured Gene assets, Evolver integrates experience acquisition, retrieval, and feedback-guided evolution to support experience-conditioned execution across existing agent harnesses. We evaluate Evolver using Claude Code and Codex, two foundation model configurations, and four benchmarks: Terminal-Bench 2.1, SWE-bench Multilingual, SWE-rebench, and GAIA. On a selected diagnostic set of 320 paired executions, Evolver improves task success from 83.13% to 87.50% (+4.38 percentage points) while reducing recorded task-execution token consumption by 24.24%. Furthermore, among 265 pairs successf
    
[^120]: 不完全信息博弈中的学习与引导

    Learning and Steering in Incomplete-Information Games

    [https://arxiv.org/abs/2611.07589](https://arxiv.org/abs/2611.07589)

    该论文提出了仅利用轨迹反馈就能在不知道智能体效用的情况下学习标准式和扩展式博弈的二分类型算法，使调解者能够引导自适应智能体走向效用最大化的结果，并达到了近乎最优的样本复杂度。

    

    在许多现代多智能体系统中，调解者希望仅通过结果层面的反馈来影响策略性决策动态，而无需了解智能体的效用。在本工作中，我们研究调解者如何学习一个未知的博弈，并仅使用轨迹反馈将自适应的智能体引导至使其自身效用最大化的结果。调解者知道博弈的形式和自己的目标，但不知道智能体的效用。此外，她承诺支付非负的款项，并提供推荐和上下文信息，但她只能获得单个已实现的终端结果，而非全信息反馈。我们开发了二分类型（halving-type）算法，用于学习标准式博弈中的效用差异和扩展式博弈中的条件延续价值差异，实现了 $\widetilde O(\varepsilon^{-2})$ 的样本复杂度，在最坏情况下与下界仅相差对数因子。随后，我们将这些学习程序与拉格朗日方法相结合……

    arXiv:2611.07589v1 Announce Type: cross  Abstract: In many modern multi-agent systems, a mediator seeks to influence strategic decision dynamics using only outcome-level feedback and without knowing agents' utilities. In this work, we study how a mediator can learn an unknown game and steer adaptive agents toward outcomes that maximize her utility using only trajectory feedback. The mediator knows the game form and her own objective, but not the agents' utilities. Moreover, she commits to nonnegative payments and provides recommendations and contexts, but receives only a single realized terminal outcome rather than full-information feedback. We develop halving-type algorithms that learn utility differences in normal-form games and conditional continuation-value differences in extensive-form games, achieving a sample complexity of $\widetilde O(\varepsilon^{-2})$ that matches worst-case lower bounds up to logarithmic factors. We then combine these learning procedures with a Lagrangian f
    
[^121]: 针对隐蔽失对齐行为的思维链监控压力测试

    Stress Testing Chain-of-Thought Monitoring Against Covert Misalignment

    [https://arxiv.org/abs/2611.07586](https://arxiv.org/abs/2611.07586)

    提出自动化红队测试框架 TraceMRT，通过将副任务重构为主任务的连贯子目标并搜索轨迹塑形策略，系统性地压力测试了思维链监控对会隐蔽掩盖失对齐行为的智能体的鲁棒性。

    

    随着大语言模型（LLM）智能体在自主性方面快速提升，智能体出现绕过安全约束的隐藏目标的风险日益增加。思维链（CoT）监控提供了一种关键的控制机制：通过读取智能体的推理过程，在失对齐行为付诸实施之前将其标记出来。然而其可靠性取决于对它进行压力测试的红队攻击的真实程度。现有的红队测试工作依赖于共享模板，例如将副任务伪装为完成主任务的前置条件，这种做法将副目标视为外生插入，而对任务特定的、有上下文依据的规避策略则探索不足。因此我们提出这样一个问题：面对会战略性隐藏自身失对齐行为的智能体，CoT 监控究竟有多强的鲁棒性？为了回答这一问题，我们提出了 TraceMRT——一个自动化的轨迹级（Trace-level）监控红队测试框架，它将副任务重新构建为主任务的一个连贯子目标，并搜索轨迹塑形策略，使其诱导出的轨迹能够隐蔽地掩盖失对齐意图……

    arXiv:2611.07586v1 Announce Type: cross  Abstract: As LLM agents advance rapidly in autonomy, the risk of hidden objectives that bypass safety constraints becomes increasingly plausible. Chain-of-thought (CoT) monitoring offers a key control mechanism by reading the agent's reasoning and flagging misalignment before it materializes into action, yet its reliability depends on the realism of the red-team it is stress-tested against. Existing red-teaming efforts rely on shared templates, such as casting a side task as a prerequisite, which treat the side objective as an exogenous insertion and leave task-specific, contextually grounded evasions underexplored. We therefore ask: How robust is CoT monitoring against agents that strategically conceal misalignment? To answer this, we introduce TraceMRT, an automated Trace-level Monitor Red-Teaming framework that reframes the side task as a coherent sub-goal of the primary task and searches trace-shaping strategies whose induced trajectories co
    
[^122]: SKIM：通过选择性知识引导掩码对大语言模型智能体进行剪枝

    SKIM: Pruning Large Language Model Agents via Selective Knowledge Informed Masking

    [https://arxiv.org/abs/2611.07558](https://arxiv.org/abs/2611.07558)

    提出SKIM——首个面向结构化生成场景的大语言模型智能体剪枝与蒸馏框架，通过将词元位置划分为上下文、决策和格式三类，在剪枝与蒸馏两个阶段中优先保留与决策位置最相关的知识。

    

    大语言模型（LLM）智能体越来越多地在结构化生成场景下部署，其中语法约束在推理时强制执行输出格式。然而现有的压缩方法均针对无约束生成，我们首次发现了一个根本性的不匹配：在结构化生成下，语法约束会将模型的输出分布缩减至有效词元，而对数几率在决定智能体动作的少量关键位置上最为重要。受此发现启发，我们提出了SKIM（选择性知识引导掩码），这是首个面向结构化生成下大语言模型智能体的剪枝与蒸馏框架，其两个阶段均由词元角色引导。SKIM将词元位置划分为上下文、决策和格式三类，这一划分驱动了两个互补的方向：（1）类别感知的剪枝显著性分数，以保留与决策位置最相关的参数；（2）一种在线策略蒸馏目标……

    arXiv:2611.07558v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents are increasingly deployed under structured generation, where grammar constraints enforce output format at inference time. Whereas existing compression methods target unconstrained generation, we first identify a fundamental mismatch: under structured generation, grammar constraints reduce the model's output distribution to valid tokens, and the logits matter most at the small set of positions that decide the agent's action. Motivated by this finding, we propose SKIM (Selective Knowledge-Informed Masking), the first pruning and distillation framework for LLM agents under structured generation, with both stages guided by token role. SKIM partitions token positions into context, decision, and format categories, and this partition drives two complementary axes: (1) category-aware pruning saliency scores that preserve parameters most relevant for decision positions; and (2) an on-policy distillation objecti
    
[^123]: 记忆陷阱：语言智能体会从良性交互中过度泛化

    The Memory Trap: Language Agents Overgeneralize from Benign Interactions

    [https://arxiv.org/abs/2611.07554](https://arxiv.org/abs/2611.07554)

    本文发现并命名了“记忆陷阱”这一新型失效模式：语言智能体会从良性交互记忆中过度泛化，例如低风险任务中跳过确认的经历会使其在高风险任务中也跳过安全验证，而情感支持或被批评经历形成的记忆则会使后续判断偏向谄媚或过度批评。

    

    语言智能体越来越多地使用持久化记忆来跨会话记录用户偏好和操作流程。我们证明，来自良性交互的记忆可能导致智能体在后续任务中出现不安全或不可靠的行为。我们将这种失效现象称为“记忆陷阱”。为研究这一问题，我们比较了六个前沿模型（包括 Claude Sonnet、Opus、Fable 模型和 GPT-5.6）在有记忆和无记忆会话中的行为差异，并在多种设置下对这些模型进行了评估。首先，在自主性实验中，我们发现用户要求在低风险编程杂务中跳过确认的请求，会导致智能体在编程和日常生活的模拟高风险任务中也跳过确认与验证。其次，我们研究了当用户寻求情感支持、或描述不加批判的反馈所造成的伤害时形成的记忆，发现这些记忆会使智能体在后续对数学作业、论证和诗歌的评价中分别偏向谄媚或过度批评。

    arXiv:2611.07554v1 Announce Type: cross  Abstract: Language agents increasingly use persistent memory to record user preferences and procedures across sessions. We show that memory from benign interactions can cause unsafe or unreliable behavior on later tasks. We call this failure the memory trap. To study this issue, we compare behavior in sessions with and without memory across six frontier models (including Claude Sonnet, Opus, and Fable models and GPT-5.6). We evaluate these models in a range of settings. First, in autonomy experiments, we find that requests to skip confirmation during low-stakes coding chores lead agents to skip confirmation and verification on simulated high-stakes tasks in both coding and everyday life. We next study memories formed when users seek emotional support, or describe harm from uncritical feedback, and find these memories bias later judgments of mathematical work, arguments, and poems toward sycophancy or over-critique (respectively). Memory notes wr
    
[^124]: FlashOffloading：面向长上下文大语言模型服务的低延迟KV缓存卸载

    FlashOffloading: Low-Latency KV Cache Offloading for Long-Context LLM Serving

    [https://arxiv.org/abs/2611.07542](https://arxiv.org/abs/2611.07542)

    FlashOffloading提出一种以主机DRAM为中心的推理架构，通过滚动HBM缓存、解码热缓存、零开销反向稀疏索引和计算-加载重叠内核，在预填充与解码阶段完全卸载KV缓存，实现长上下文LLM服务中低延迟下的并发度与吞吐量提升。

    

    在大型语言模型（LLM）服务系统中，键值（KV）缓存随序列长度和并发数线性增长，因此长上下文推理容易触及高带宽内存（HBM）的容量瓶颈，导致并发度低下。将KV缓存卸载到主机动态随机存取内存（DRAM）可以释放HBM空间并提高并发度，但额外的缓存管理和主机到设备（H2D）的加载开销会推高推理延迟，使得更高的并发度无法转化为有效的吞吐量。我们提出FlashOffloading，一种以主机DRAM为中心的推理架构，在预填充和解码两个阶段完全卸载KV缓存。FlashOffloading为预填充阶段构建滚动HBM缓存，为解码阶段构建热缓存，构建了具有零额外缺失查找开销的反向稀疏索引，以及计算-加载重叠内核，在整个流水线上优化KV缓存卸载延迟，并在每个输出token时间（TPOT）约束下带来吞吐量提升。

    arXiv:2611.07542v1 Announce Type: new  Abstract: In large language model (LLM) serving systems, the key-value (KV) cache grows linearly with sequence length and concurrency, so long-context inference easily hits the high-bandwidth memory (HBM) capacity wall and results in low concurrency. Offloading the KV cache to host dynamic random-access memory (DRAM) frees HBM space and raises concurrency, but the additional cache management and host-to-device (H2D) load overhead inflates inference latency, higher concurrency fails to translate into effective throughput. We present FlashOffloading, a host-DRAM-centric inference architecture that fully offloads the KV cache in both the prefill and decode phases. FlashOffloading builds rolling HBM cache for prefill, a hot cache for decode, a reverse sparse index with zero extra miss-lookup overhead, and a compute--load overlapped kernel, optimizing KV offloading latency across the entire pipeline and delivering throughput gains under a time-per-outp
    
[^125]: CST-Bench：基于因果转变在线策略自蒸馏的反事实空间转变推理基准测试

    CST-Bench: Benchmarking Counterfactual Spatial Transition Reasoning with Causal Transition On-Policy Self-Distillation

    [https://arxiv.org/abs/2611.07541](https://arxiv.org/abs/2611.07541)

    提出了反事实空间推理基准CST-Bench（包含7,082个样本、6种干预类型、12种变换算子及密集因果标注），揭示现有MLLM在物体被干预移动或交换时空间推理仍会失效，并提出因果转变在线策略自蒸馏方法来提升反事实空间推理能力。

    

    空间推理是多模态大语言模型（MLLM）的一项基础能力，然而现有基准主要集中于静态场景理解。当物体被移动或交换时，模型不仅要推断空间关系，还必须推断干预操作如何改变场景状态。为解决这一问题，我们提出了CST-Bench，一个用于反事实空间推理的基准。CST-Bench包含7,082个样本，涵盖6种干预类型和12种变换算子，并带有密集的因果标注，包括干预来源、受影响的关系、预期效果、不变关系以及度量变化。我们在CST-Bench上对通用MLLM进行了基准测试，并进一步在CST-Bench和标准空间基准上比较了多种后训练策略。结果表明，对普通空间推理有效的方法在反事实干预下仍可能失效。我们进一步将反事实空间推理形式化为因果状态转变问题……（原文摘要在此处截断）

    arXiv:2611.07541v1 Announce Type: new  Abstract: Spatial reasoning is a fundamental capability of multimodal large language models (MLLMs), yet existing benchmarks mainly focus on static scene understanding. When objects are moved or swapped, models must infer not only spatial relations but also how interventions change the scene state. To address this problem, we introduce CST-Bench, a benchmark for counterfactual spatial reasoning. CST-Bench contains 7,082 examples covering 6 intervention types and 12 transformation operators, with dense causal annotations including intervention sources, affected relations, expected effects, invariant relations, and metric changes. We benchmark general-purpose MLLMs on CST-Bench and further compare post-training strategies on CST-Bench and standard spatial benchmarks. Results show that methods effective for ordinary spatial reasoning can still fail under counterfactual interventions. We further formulate counterfactual spatial reasoning as causal sta
    
[^126]: 面向自动驾驶车辆语义异常的动作条件视觉语言模型监测

    Action-Conditioned VLM Monitoring for Autonomous Vehicle Semantic Anomalies

    [https://arxiv.org/abs/2611.07540](https://arxiv.org/abs/2611.07540)

    该论文提出一种动作条件监测框架：先由视觉语言模型从隐藏结果的视频片段中预测场景所需动作，再通过明确的兼容性规则将预期与实际自车轨迹对比，从而判断自动驾驶行为在语义异常场景中是否恰当，并配套构建了112个配对场景的分类法和生成视频语料库。

    

    我们研究了在语义异常的驾驶场景中对自动驾驶车辆行为进行动作条件监测的问题。监测目标是判断自车执行的动作是否适合当前场景：同一场景可能同时容许正常与异常行为，因此仅凭视觉上的异常性并不能决定监测标签。视觉语言模型（VLM）首先从一个隐藏了结果（结局）的视频窗口中预测场景所需的动作；随后监测器在明确的兼容性规则下，将该预期与实际的自车运动轨迹进行比较。为研究这一监测任务，我们贡献了：（1）一个涵盖两种失败极性的112个配对场景的六字段分类法（其中42个源自有记录的真实事故）；（2）一个带有估计轨迹的生成视频语料库；（3）在该语料库上的一个探索性两阶段监测器。在来自15个场景族的235个视频片段上，采用固定T−2.5秒窗口的两阶段协议获得了83.0%的准确率和异常召回率0

    arXiv:2611.07540v1 Announce Type: cross  Abstract: We study action-conditioned monitoring of autonomous-vehicle behavior in semantically unusual driving scenes. The target is whether the executed ego action is appropriate for the scene: the same scene can admit both normal and anomalous behavior, so visual unusualness alone does not determine the monitoring label. A Vision-Language Model (VLM) first predicts the scene-required action from an outcome-hidden video window; the monitor then compares that expectation with the realized ego-motion trajectory under an explicit compatibility rule. To study this monitoring task, we contribute a six-field taxonomy of 112 paired scenarios spanning both failure polarities (42 derived from documented incidents), a generated video corpus with estimated trajectories, and an exploratory two-stage monitor on that corpus. On 235 clips from 15 scenario families, the two-stage protocol with a fixed T-2.5 s window obtains 83.0% accuracy and anomaly recall 0
    
[^127]: 自适应实验可能掩盖模型误差：检测极限与证伪预算

    Adaptive Experimentation Can Hide Model Error:Detection Limits and Falsification Budget

    [https://arxiv.org/abs/2611.07529](https://arxiv.org/abs/2611.07529)

    本文证明了自适应实验设计可能通过选择错误模型恰好拟合的设置来掩盖模型误差，提出了检验功效上界、可计算的诊断指标以及无需训练的证伪预算规则，主张在实验设计中主动为证伪模型保留部分预算。

    

    一个实验可能对工作模型的参数提供有效信息，却对模型本身是否错误不提供任何信息。自适应设计会通过反复选择错误模型恰好能够拟合的实验设置，从而加剧这种不匹配。我们利用实验者自身的抽样分布，为任何有效的充分性检验的功效设定了上界，并进一步区分了所选数据中缺失的信息与检测器本身损失的信息。我们提出了一个可计算的诊断指标，用以相对于所声明的工作模型扩展来度量尚未解决的模型误差。这一指标促成了一个无需训练的规则，即为证伪保留一部分设计预算。实证结果好坏参半：独立设定的主研究未通过检测资格验证，随后的分子诊断也未能达到其改进标准。在冻结规则于新的连续零假设面板上失败后，一个公开的、经过地平线校准的次要修订版检测到了35.8%的错误设定实验（注：摘要在此处被截断）。

    arXiv:2611.07529v1 Announce Type: cross  Abstract: An experiment can be informative about the parameters of a working model and uninformative about whether that model is wrong. Adaptive design can reinforce this mismatch by repeatedly selecting settings where an incorrect model fits. We bound the power of any valid adequacy test in terms of the experimenter's own sampling distribution, then distinguish information absent from the selected data from information lost by the detector. A computable diagnostic measures unresolved error relative to a declared expansion of the working model. It motivates a training-free rule that reserves part of the design budget for falsification. The empirical record is mixed: the independently specified primary study fails detection eligibility, and a later numerator diagnostic fails its improvement criteria. After the frozen rule fails on fresh continuous-null panels, a disclosed secondary horizon-calibrated revision detects 35.8% of misspecified campaig
    
[^128]: 目标分支LaCAM：以临时目标作为搜索分支求解密集多智能体路径规划

    Goal-Branching LaCAM: Temporary Goals as Search Branches in Dense Multi-Agent Pathfinding

    [https://arxiv.org/abs/2611.07528](https://arxiv.org/abs/2611.07528)

    提出目标分支方法，为阻塞智能体分配临时目标并将其作为LaCAM搜索的额外分支，在保持算法完备性的同时显著提升了密集多智能体路径规划实例的求解成功率。

    

    LaCAM是一种基于搜索的多智能体路径规划（MAPF）求解器，它对联合配置进行深度优先搜索，使用PIBT生成后继节点，并通过惰性添加单智能体约束来探索替代方案。尽管LaCAM具有完备性，但在密集且走廊繁多的实例上会表现不佳，因为将阻塞的智能体从死锁中解救出来需要多个智能体的协调移动，这些移动往往偏离各自的目标，而PIBT的贪婪单步后继节点很少会提出此类移动，LaCAM只能通过逐个枚举智能体约束来达到这些移动。本文提出目标分支方法，即为阻塞智能体分配临时目标，并将每个临时目标作为LaCAM搜索的额外分支进行尝试。我们研究了选择这些临时目标的随机规则和走廊感知规则，并证明任何此类规则都能保持完备性。在三张迷宫和房间地图的密集实例（60-90%的自由单元格被占用）上，目标分支方法解决了89%的实例……

    arXiv:2611.07528v1 Announce Type: new  Abstract: LaCAM is a search-based solver for multi-agent pathfinding (MAPF) that runs a depth-first search over joint configurations, using PIBT to generate successors and lazily adding single-agent constraints to explore alternatives. Although LaCAM is complete, it can struggle on dense, corridor-heavy instances because extracting blocking agents from deadlocks requires coordinated moves of several agents, often away from their goals, which PIBT's greedy one-step successors rarely propose and which LaCAM can only reach by enumerating constraints one agent at a time. This paper proposes Goal-Branching, where blocking agents are given temporary goals, each tried as an additional branch of LaCAM's search. We examine random and corridor-aware rules for choosing these temporary goals and show that completeness is preserved for any such rule. On dense instances (60-90% of free cells occupied) of three maze and room maps, Goal-Branching solves 89% of in
    
[^129]: 面向中断情况下分布鲁棒站内列车重调度的图学习辅助逻辑Benders分解方法

    Graph learning-assisted logic-based Benders decomposition for distributionally robust in-station train rescheduling under disruptions

    [https://arxiv.org/abs/2611.07520](https://arxiv.org/abs/2611.07520)

    本文提出GNN-LBBD算法，将图神经网络与基于逻辑的Benders分解相结合，求解中断情况下站内列车重调度的滚动时域分布鲁棒优化模型，使平均总完成延误较对比方法降低16.63%。

    

    当资源关闭与列车到达延误和处理时间延长同时发生时，资源重新开放时间的不确定性使列车扣停和改线决策变得复杂。我们开发了一个滚动时域分布鲁棒模型，该模型能够随着中断恢复情况观测信息的不断获取，更新对剩余关闭时长的预测。该模型联合优化列车走行路径、运行顺序和时刻安排，并在场景概率的模糊集上最小化最坏情况下的期望总完成延误。所提出的求解算法GNN-LBBD采用基于逻辑的Benders分解来协调路径规划与资源调度：条件割平面基于资源使用模式构建，并由图神经网络引导资源松弛子问题的求解顺序。在针对五股道车站的28个测试实例中，GNN-LBBD实现了61.13列车分钟的平均总完成延误，比（对比方法）降低了16.63%。

    arXiv:2611.07520v1 Announce Type: new  Abstract: Uncertainty in resource reopening times complicates train holding and rerouting decisions when resource closures coincide with arrival delays and processing-time extensions. We develop a rolling-horizon distributionally robust model that updates predictions of the remaining closure duration as observations of disruption resolution become available. The model jointly optimizes train routes, sequences, and timings. It minimizes the worst-case expected total completion delay over an ambiguity set of scenario probabilities. The solution algorithm, GNN-LBBD, uses logic-based Benders decomposition to coordinate routing and resource scheduling. Conditional cuts are constructed from resource-use patterns, and a graph neural network guides the order in which resource relaxation subproblems are solved. Across 28 test instances for a five-track station, GNN-LBBD achieved a mean total completion delay of 61.13 train-min. This was 16.63\% lower than 
    
[^130]: Concertina：在单一扩散视觉语言模型中融合操作与认知

    Concertina: Embodying Manipulation and Cognition in One Diffusion Vision-Language Model

    [https://arxiv.org/abs/2611.07513](https://arxiv.org/abs/2611.07513)

    提出Concertina——一种具身视觉语言模型（eVLM），将连续轨迹表示、分块生成和物理坐标监督直接融入共享的扩散VLM主干，使操作与认知统一在同一模型中，避免了传统VLA解耦架构的量化误差和梯度竞争问题。

    

    大多数视觉-语言-动作（VLA）策略通过在视觉语言模型（VLM）上外挂一个独立的动作机制，将操作与认知解耦——要么将动作量化为离散词元（其误差与物理位移之间没有稳定的对应关系），要么将连续轨迹生成交给一个独立的动作专家模块（其梯度会与VLM主干产生竞争，或被阻止更新VLM主干）。我们认为操作与认知应当内化于同一个模型之中，即从解耦的VLA转向具身视觉语言模型（eVLM）。我们提出了Concertina，这是一种eVLM，它将连续轨迹表示、分块生成和物理坐标监督直接置于共享的扩散VLM主干的计算与参数路径上。每个控制时间步被编码为主干隐藏空间中的一个连续潜在动作词元，由共享的Transformer进行重建（摘要原文在此处被截断）。

    arXiv:2611.07513v1 Announce Type: cross  Abstract: Most vision-language-action (VLA) policies decouple manipulation from cognition by bolting an external action mechanism onto a vision-language model (VLM)--either quantizing actions into discrete tokens whose errors bear no stable correspondence to physical displacements, or delegating continuous trajectory generation to a separate action expert whose gradients compete with or are blocked from updating the VLM backbone. We argue that manipulation and cognition should instead be embodied in the same model, shifting from a decoupled VLA to an Embodied Vision-Language Model (eVLM). We introduce Concertina, an eVLM that places continuous trajectory representation, blockwise generation, and physical-coordinate supervision directly on the computation and parameter path of a shared diffusion VLM backbone. Each control timestep is encoded as one continuous latent action token in the backbone hidden space, and the shared Transformer reconstruct
    
[^131]: DelegateEval：面向个人外部服务代理的平台评估框架

    DelegateEval: Evaluating Platforms for Personal External Service Agents

    [https://arxiv.org/abs/2611.07508](https://arxiv.org/abs/2611.07508)

    提出了DelegateEval——一个用于评估个人外部服务代理平台的首个平台级评估框架，从创建便捷性到效率与价值共八大维度系统衡量个人能否借助代理大规模服务外部客户并创造经济价值。

    

    长期为个人工作的AI代理也可以帮助该个人服务外部客户。我们将“个人外部服务代理”定义为：由特定个人授权并持续管理的代理，它接受客户请求、交付工作成果，并利用该个人的知识、经验、专业方法和服务边界提供持续支持。此类代理可以让个人以更少的直接参与服务更多客户，并大规模地创造经济价值。我们提出DelegateEval，这是一个面向代理系统的平台级评估框架，适用于可能结合大语言模型、可复用技能、MCP等协议以及可执行工具环境的代理系统。该框架涵盖八个维度：创建便捷性、服务发现与匹配、交付质量、专业知识继承、服务持续性、学习与改进、可靠性与治理，以及效率与价值。这些维度用于评估一个平台能否……（原文截断）

    arXiv:2611.07508v1 Announce Type: new  Abstract: AI agents that work for an individual over time may also help that individual serve external clients. We define a personal external service agent as an agent authorized and continuously managed by a specific individual to accept client requests, deliver work, and provide ongoing support using that person's knowledge, experience, professional methods, and service boundaries. Such agents could allow individuals to serve more clients with less direct involvement and generate economic value at scale. We propose DelegateEval, a platform-level evaluation framework for agent systems that may combine large language models, reusable skills, protocols such as MCP, and executable tool environments. It covers eight dimensions: ease of creation, service discovery and matching, delivery quality, expertise inheritance, service continuity, learning and improvement, reliability and governance, and efficiency and value. These dimensions assess whether a p
    
[^132]: 签名但并非安全：多智能体LLM交接中的需求完整性

    Signed but Not Safe: Requirement Integrity in Multi-Agent LLM Handoffs

    [https://arxiv.org/abs/2611.07483](https://arxiv.org/abs/2611.07483)

    该论文提出一种带HMAC-SHA256签名的需求清单交接机制，配合确定性代码验证，将多智能体LLM系统在攻击下的不安全提交从69.4%降至9.0%，且不损失合规提交。

    

    企业级AI智能体越来越多地将用户的请求拆分给多个子智能体并合并其结果，而每一次交接都可能概括掉、篡改或忽略某项需求。我们报告了一项预注册研究，使用合成的、可精确评分的任务，比较了自由文本、结构化JSON和带签名的需求清单三种交接方式：编排器将每项需求提取为逐字引用加上可核查的谓词，代码使用HMAC-SHA256对清单进行签名，并由确定性代码验证合并器的决策。使用Qwen3-8B（1,440条轨迹，五种威胁），带签名的清单将攻击下的不安全提交从69.4%降至9.0%（降低60.4个百分点，95%置信区间[51.4, 69.4]），且与结构化交接相比，合规提交没有净损失。签名正是击败传输中篡改的关键：未签名的清单在79%的篡改运行中提交了违规内容，而签名的清单则一次也没有。在相同的轨迹上，确定性验证器捕获了78.7%的违规决策。

    arXiv:2611.07483v1 Announce Type: cross  Abstract: Enterprise AI agents increasingly split a user's request across subagents and merge their results, and each handoff can summarize away, alter or ignore a requirement. We report a preregistered study with synthetic, exactly scorable tasks comparing free-text, structured JSON and signed requirement-manifest handoffs: the orchestrator extracts each requirement as a verbatim quote plus a checkable predicate, code signs the manifest with HMAC-SHA256, and deterministic code verifies the merger's decision. With Qwen3-8B (1,440 trajectories, five threats), signed manifests cut unsafe commits under attack from 69.4% to 9.0% (60.4 pp, 95% CI [51.4, 69.4]) with no net loss of conformant commits relative to structured handoffs. Signing is what defeats in-transit tampering: an unsigned manifest committed violations in 79% of mutated runs, the signed one in none. On identical trajectories the deterministic verifier caught 78.7% of violating decision
    
[^133]: SAVeR²：基于推理的、通过可验证奖励实现的大型推理模型安全对齐

    SAVeR$^2$: Reasoning-based Safety Alignment for Large Reasoning Models via Verifiable Rewards

    [https://arxiv.org/abs/2611.07482](https://arxiv.org/abs/2611.07482)

    SAVeR²提出了一种基于可验证奖励的推理式安全对齐方法，将安全目标分解为四个细粒度的可验证奖励信号，并结合难度感知的数据选择与保持推理能力的梯度投影，在提升大型推理模型安全性的同时避免损害其推理能力。

    

    大型推理模型（LRMs）通过显式的思维链（CoT）推理取得了令人瞩目的性能，然而这一过程也引入了关键的安全风险。我们将LRM的安全目标形式化为：要求推理与回答非恶意、保持安全一致性以及进行显式的意图识别。我们认为，现有对齐方法存在对抗泛化能力差、推理能力遭受显著“安全税”以及粗粒度奖励不忠实等问题，其主要原因在于忽视了LRM自身内在的、可服务于安全的推理能力。为解决这些问题，我们提出了SAVeR²，一种新颖的基于具备推理能力的奖励的安全对齐方法。SAVeR²将安全目标分解为四个细粒度的可验证奖励信号，并利用难度感知的数据选择策略来稳定训练。至关重要的是，我们引入了一种保持推理能力的梯度投影……（摘要原文在此处截断）

    arXiv:2611.07482v1 Announce Type: new  Abstract: Large Reasoning Models (LRMs) have achieved impressive performance via explicit Chain-of-Thought (CoT) reasoning, yet this process introduces critical safety risks. We formalize LRM safety objectives as requiring non-malicious reasoning and answers, safety consistency, and explicit intent identification. We argue that existing alignment methods suffer from poor adversarial generalization, a significant ``safety tax'' on reasoning, and unfaithful coarse-grained rewards, primarily due to their neglect of the LRM's intrinsic reasoning capabilities for safety. To address these, we propose \textbf{SAVeR$^2$}, a novel \textbf{S}afety \textbf{A}lignment method via \textbf{R}ewards with \textbf{R}easoning capability. SAVeR$^2$ decomposes the safety objective into four fine-grained verifiable reward signals and utilizes a difficulty-aware data selection strategy to stabilize training. Crucially, we introduce a reasoning-preserving gradient projec
    
[^134]: 定位并不意味着可操控性：LLM数学中符号错误的机制研究

    Localization Does Not Imply Steerability: A Mechanistic Study of Sign Errors in LLM Mathematics

    [https://arxiv.org/abs/2611.07478](https://arxiv.org/abs/2611.07478)

    该研究通过逐层因果分析揭示了LLM在多步数学中的符号错误主要由网络后四分之一区段的MLP层在后期产生，且不同模型具有不同机制（固定的“负号盖章”层、符号放大器、救援层），同时指出即便能精确定位错误来源，也不意味着能够有效操控和纠正它。

    

    当大语言模型进行多步数学运算时，一种特征性的失败反复出现：模型在计算中途写下一个错误的符号token——例如余子式的符号、递归分部积分步骤中的负号——随后正确执行其余的算术运算，但最终答案出错。我们探究这个符号在网络内部是在何处写下的。在涵盖14B至675B参数的六个开源模型中，逐层读出实验表明，错误的符号是在计算后期产生的——由网络最后四分之一区段的MLP层所为（每个模型和任务中占比70-100%），且在经过测试的情况下（Qwen），这一模式在指令微调之前的基础模型中就已存在。具体机制因模型而异：一些模型带有固定的“负号盖章”层，无论正确答案如何都会推出负号；另一些模型则带有对已选定符号的放大器；而Qwen还额外增加了一个下游的“救援层”来对抗这种盖章。该回路是因果性的：从残差流中减去单一符号方向（后文截断）……

    arXiv:2611.07478v1 Announce Type: cross  Abstract: When large language models do multi-step mathematics, a characteristic failure recurs: the model writes one wrong sign token mid-computation -- a cofactor sign, the minus of a recursive integration-by-parts step -- then executes the rest of the arithmetic correctly, and the answer comes out wrong. We ask where inside the network that sign is written. Across six open models spanning 14B to 675B parameters, a per-layer readout shows the wrong sign is committed late -- by MLP layers in the last quarter of the network (70-100% per model and task), a pattern present, where tested (Qwen), in the base model before instruction tuning. The mechanism varies: some models carry fixed "minus-stamp" layers that push a minus regardless of the right answer, others carry amplifiers of whichever sign was already chosen, and Qwen adds a downstream rescue layer that opposes the stamp. The circuit is causal: subtracting a single sign direction from the res
    
[^135]: KOSMOS：用于通信-计算重叠的自调度持久内核

    KOSMOS: Self-scheduling Persistent Kernels for Communication-Compute Overlap

    [https://arxiv.org/abs/2611.07477](https://arxiv.org/abs/2611.07477)

    KOSMOS 提出一种自调度持久内核库，其工作组利用按消费者需求排序的队列根据数据就绪状态在通信与计算角色间动态切换，将 Transformer 层所有 TP+SP 集合通信与 GEMM 融合，无需调度器或预设切分，在 AMD MI350X/MI355X GPU 上将 Megatron-LM 端到端训练最高加速 1.28 倍。

    

    张量与序列并行（TP+SP）以及专家并行（EP）会将集合通信置于 Transformer 层的关键路径上。最先进的融合内核通过固定角色或固定调度来缓解这一问题，但这种僵化会留下大量空闲窗口；而预测通信-计算切分同样无法消除它们，因为预先确定的切分无法跟随数据的实际就绪状态。我们提出了 KOSMOS，一个自调度持久内核库，其工作组使用按消费者需求排序的队列，在数据就绪时即可在通信与计算角色之间动态切换，无需任何调度器、每次调用的代价模型或预先调优的切分方案。KOSMOS 将 Transformer 层的每一个 TP+SP 集合通信与其对应的 GEMM 相融合，并在每个方向上仅需一次启动即可运行完整的专家层（从路由器到折叠）。在 AMD Instinct MI350X 和 AMD Instinct MI355X GPU 上的评估显示，KOSMOS 在 BF16 精度下将 Megatron-LM 的 TP+SP 端到端训练加速 1.10-1.28 倍，在 M（摘要在此处被截断）。

    arXiv:2611.07477v1 Announce Type: cross  Abstract: Tensor and sequence parallelism (TP+SP) and expert parallelism (EP) put collectives on the critical path of transformer layers. State-of-the-art fused kernels mitigate this with fixed roles or schedules, but this rigidity leaves large idle windows. Predicting a communication-computation split does not remove them either; a predetermined split cannot follow data readiness. We present KOSMOS, a library for self-scheduling persistent kernels whose workgroups use queues ordered by consumer demand to switch between communication and computation roles as data become ready, with no scheduler, per-call cost model, or tuned split. KOSMOS fuses every TP+SP collective of a transformer layer with its GEMM, and runs an entire expert layer, from router to fold, in one launch per direction. Evaluated on AMD Instinct MI350X and AMD Instinct MI355X GPUs, KOSMOS speeds up TP+SP end-to-end training in Megatron-LM by 1.10-1.28x in BF16 and 1.10-1.25x in M
    
[^136]: 核机器的对抗鲁棒性：一项全面分析

    Adversarial Robustness of Kernel Machines: A Comprehensive Analysis

    [https://arxiv.org/abs/2611.07474](https://arxiv.org/abs/2611.07474)

    本文通过白盒和黑盒攻击全面评估了核机器的对抗鲁棒性，发现核机器天生比神经网络更鲁棒，并提出了一种核机器的近似对抗训练算法。

    

    核机器是机器学习中一类重要的模型。然而，现代核机器对对抗样本的鲁棒性问题尚未得到全面研究。在本文中，我们通过白盒和黑盒逃避攻击，对基于核的模型在二分类和多分类任务中的对抗鲁棒性进行了深入的实证研究。我们评估了核机器、随机特征分类器以及相对较新的递归特征机器的固有鲁棒性。此外，我们提出了一种用于核机器近似对抗训练的算法。我们在图像和表格数据集分类任务上的实验有力地表明，核机器天生比神经网络更鲁棒，即在面对对抗扰动时其性能下降更为平缓。

    arXiv:2611.07474v1 Announce Type: cross  Abstract: Kernel machines are an important class of models in machine learning. However, the question of how robust modern kernel machines are to adversarial examples has not been studied comprehensively. In this paper, we provide a thorough empirical investigation of the adversarial robustness of kernel-based models for both binary and multiclass classification tasks, via white-box and black-box evasion attacks. We evaluate the native robustness of kernel machines, random feature classifiers, and the relatively newer Recursive Feature Machines. In addition, we provide an algorithm for approximate adversarial training of kernel machines. Our experiments on classification tasks on image and tabular datasets strongly indicate that kernel machines are natively more robust than neural networks, i.e., their performance degrades more gracefully against adversarial perturbations.
    
[^137]: 误差的回声：用于时间序列预测的残差方向局部滚动一致性

    Echoes of Error: Residual-Directional Local Rollout Consistency for Timeseries Forecasting

    [https://arxiv.org/abs/2611.07473](https://arxiv.org/abs/2611.07473)

    该论文提出“残差方向局部滚动一致性”方法，通过沿实际反馈扰动方向惩罚预测器的雅可比向量积这一一阶替代方案，精确抑制时间序列预测中自生成误差对未来预测的传播影响，从而避免设计难以实现的残差到状态对齐算子。

    

    在时间序列预测中，滚动误差不仅仅是一步拟合的偏差：一旦被反馈到不断演化的状态中，它们会扰动未来的预测。我们通过局部滚动一致性将这一效应形式化，该一致性比较在教师强制下产生的下一步预测与在自由运行下产生的预测。原则上，这种一致性可以通过跟踪由残差引起的下一状态扰动来直接实施。然而在实践中，当输入状态和输出块具有不同的维度、异构的组件或依赖于架构的布局时，残差到状态的对齐算子往往难以设计。因此，我们使用一种一阶局部替代方法，沿着实际的反馈扰动惩罚预测器的雅可比向量积，从而精确地抑制自生成误差对未来预测仍可见的那些方向。我们进一步给出……（摘要在此处截断）

    arXiv:2611.07473v1 Announce Type: cross  Abstract: In time-series forecasting, rollout errors are not merely one-step fitting discrepancies: once fed back into the evolving state, they can perturb future predictions. We formalize this effect through local rollout consistency, which compares the next prediction produced under teacher forcing with that produced under free running. In principle, this consistency can be enforced directly by tracking the residual-induced perturbation of the next state. In practice, however, the residual-to-state alignment operator is often difficult to design when the input state and output block have different dimensions, heterogeneous components, or architecture-dependent layouts. We therefore use a first-order local surrogate that penalizes the Jacobian-vector product of the predictor along the actual feedback perturbation, suppressing precisely the directions through which self-generated errors remain visible to future predictions. We further give the s
    
[^138]: FINCOSMOS：一个面向金融领域智能体的开放世界评估范式

    FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance

    [https://arxiv.org/abs/2611.07464](https://arxiv.org/abs/2611.07464)

    FINCOSMOS提出了一个金融智能体的开放世界评估范式：模拟用户隐藏其投资策略，智能体必须自主从网络获取数据并构建防“未来函数”的回测，从而实现无需参考答案、程序可验证的评估。

    

    语言智能体越来越多地在开放网络上为用户工作，而金融领域是对它们最难以信任的场景。数据必须来自网络，策略只存在于用户脑中，且正确性依赖于时间，因此智能体可能“无声地失败”：执行一个用户从未要求过的策略、使用未经核实的数据，或进行一次使用了未来信息的回测——这些都会产出一个与正确报告同样令人信服的报告。现有基准测试总会在某处封闭世界，固定任务、数据或时间线，从而消除了这些失败模式而非度量它们。我们提出FINCOSMOS，一个面向金融工作智能体的开放世界评估范式：模拟用户持有一个从组合文法中采样得到的隐藏策略，且仅在被询问时才透露；智能体不会获得任何数据，因此必须自行从网络获取数据、在数据截止日期约束下构建自己的回测，并交付可验证的结果。评分无需参考答案，程序可自动验证时点正确性。

    arXiv:2611.07464v1 Announce Type: new  Abstract: Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user's head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FINCOSMOS, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutoff, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctnes
    
[^139]: 通往真实世界的最后一英里：作为解决真实世界任务的可复用构建块的智能体原生原语

    The Last Mile to the Real World: Agent-Native Primitives as Reusable Building Blocks for Solving Real-World Tasks

    [https://arxiv.org/abs/2611.07461](https://arxiv.org/abs/2611.07461)

    提出智能体原生原语与 HEELO 框架，通过为资源配备常驻 LLM 并从原语库组合出任务专用框架，弥合从获取资源到实际应用资源之间“通往真实世界的最后一英里”。

    

    智能体如何端到端地处理真实世界任务？配备各自框架的前沿模型在真实世界基准测试中仍然表现不佳：在 Terminal-Bench 4.0 上，GPT-5.6 Sol 配合 Codex 仅解决 37.3% 的任务，Claude Opus 5 配合 Claude Code 解决 51.8%。这些任务在从获取资源到应用资源的步骤上失败，我们将其称为通往真实世界的最后一英里。现有方法只是将资源置于可及范围内，却把应用资源的能力留给调用者，因此每个框架都需要重建这种能力。我们提出了智能体原生原语——一种可复用的单元，它为资源配备一个常驻 LLM，该 LLM 能在上下文中应用该资源并与其他原语通信；同时提出 HEELO（围绕不断演化的智能体原生原语构建的框架工程方法，用于真实世界任务的最后一英里），这是一个从此类原语库为每个任务构建框架的框架工程方法。HEELO 将任务所需的原语组合成任务特定的框架……

    arXiv:2611.07461v1 Announce Type: cross  Abstract: How can an agent handle a real-world task end to end? Frontier models with their own harnesses still fall short on real-world benchmarks: on Terminal-Bench 4.0, GPT-5.6 Sol with Codex resolves 37.3% of the tasks and Claude Opus 5 with Claude Code 51.8%. These tasks fail at the step from reaching a resource to applying it, which we call the last mile to the real world. Existing approaches put the resource within reach but leave the competence to apply it with the caller, so each harness rebuilds that competence. We introduce agent-native primitives, reusable units that equip a resource with a resident LLM that applies it in context and communicates with other primitives, and HEELO (Harness Engineering around Evolving Agent-Native Primitives for the Last Mile to Real-WOrld Tasks), a harness engineering framework that builds each task's harness from a library of such primitives. HEELO composes the primitives a task needs into a task-speci
    
[^140]: IMUDINO：面向惯性传感的自蒸馏基础模型

    IMUDINO: Self-distilled Foundation Model for Inertial Sensing

    [https://arxiv.org/abs/2611.07437](https://arxiv.org/abs/2611.07437)

    IMUDINO是首个在人体活动识别、健康监测、动作捕捉和里程计等多种惯性传感任务上均达到有竞争力性能的通用基础模型，其核心创新在于通过语义保持的运动变换将DINO自监督学习范式适配到惯性数据上。

    

    我们提出了IMUDINO，这是第一个在多种传感任务上均能取得具有竞争力性能的惯性传感基础模型（FM），涵盖人体活动识别（HAR）、健康监测、动作捕捉和里程计测量等任务。以往的惯性基础模型通常只关注惯性传感的某一个子领域，有时甚至表现不如最先进的监督学习方法（例如在人体活动识别任务中），这使其适用范围远窄于视觉和自然语言领域的基础模型。此外，这些模型往往要求固定的输入长度和采样率，并且对未见过的传感器佩戴位置泛化能力较差。为了构建一个通用的惯性传感基础模型，我们将DINO自监督学习（SSL）范式适配到惯性数据上，通过一组语义保持的运动变换来模拟运动模式和传感器设置的变化。在模型架构方面，我们使用带有频率感知位置编码的普通transformer编码器，原生支持（原文在此处被截断）

    arXiv:2611.07437v1 Announce Type: cross  Abstract: We present IMUDINO, the first foundation model (FM) for inertial sensing to achieve competitive performance across diverse sensing tasks including human activity recognition (HAR), health, motion capture, and odometry. Prior inertial FMs often focus on one subfield of inertial sensing and sometimes underperform state-of-the-art supervised methods (e.g., in HAR), making their scope considerably narrower than FMs for vision and natural language. Moreover, they often require fixed input length and sampling rate, and generalize poorly to unseen sensor locations. To build a general-purpose foundation model for inertial sensing, we adapt the DINO self-supervised learning (SSL) paradigm to inertial data through a set of semantics-preserving motion transformations that simulate variations in motion pattern and sensor setup. For model architecture, we use a plain transformer encoder with frequency-aware positional encoding that natively support
    
[^141]: 构造性制度上的类型化知识

    Typed Knowledge over Constructive Institutions

    [https://arxiv.org/abs/2611.07432](https://arxiv.org/abs/2611.07432)

    该论文提出 Eigenius——一种基于构造性制度的类型化知识存储，其依赖类型理论将证明无关性作为带显式见证的公理而非转换规则，使知识存储能在“命题成立”之外保留其确立者、推导方式与假设条件等证明记录。

    

    为数学或程序验证设计的类型理论只要求一个命题成立，转换中的证明无关性会在内核层面抹去该命题是如何被确立的；而一个持久化的知识存储必须保留这一记录：谁确立了该命题、通过哪种推导或外部工具、以及在哪些假设之下。本卷提出 Eigenius 的理论——一个建立在构造性制度之上的类型化知识存储——并从这一需求出发发展其元理论。其内核理论是一个依赖类型理论，具有累积宇宙、一个非直谓的命题宇宙、归纳族以及内涵相等类型。证明无关性是一条带有显式见证的公理，而非转换规则，因此一个命题的各个证明可被证明相等，却仍保持为彼此不同的项。定义相等被刻画为在一个正交重写系统——即合流的、在类型构造子上单射的……——之下的可连接性。

    arXiv:2611.07432v1 Announce Type: cross  Abstract: Type theories designed for mathematics or program verification need only that a proposition holds; proof irrelevance in conversion erases, at the kernel level, how a claim was established. A persistent knowledge store, by contrast, must retain that record: who established a claim, by which derivation or external tool, and under which assumptions. This volume presents the theory of Eigenius---a store of typed knowledge over constructive institutions---and develops its metatheory from that requirement.   The kernel theory is a dependent type theory with cumulative universes, an impredicative universe of propositions, inductive families, and an intensional identity type. Proof irrelevance is an axiom with an explicit witness rather than a conversion rule, so proofs of a proposition are provably equal yet remain distinct terms. Definitional equality is joinability under an orthogonal rewriting system---confluent, injective on type formers,
    
[^142]: PSGAgent：通过智能体推理增强多模态大语言模型在全景场景图生成中的动态视觉感知

    PSGAgent: Enhancing MLLMs' Dynamic Visual Perception for Panoptic Scene Graph Generation via Agentic Reasoning

    [https://arxiv.org/abs/2611.07408](https://arxiv.org/abs/2611.07408)

    提出了PSGAgent智能体框架，通过“思考-行动-观察”循环以及区域放大和视觉高亮两种工具，动态增强多模态大语言模型在全景场景图生成任务中感知小物体和细粒度关系的能力。

    

    全景场景图生成要求对复杂物体间的关系进行全面且细粒度的推理。然而，当前应用于全景场景图生成的多模态大语言模型（MLLMs）存在感知瓶颈：它们作为静态观察者，受限于固定分辨率的视野。因此，模型往往忽略涉及小型或边缘物体的关键关系，并且难以在复杂场景中区分模糊的谓词。为了解决这些局限性，我们提出了PSGAgent，这是一个通过“思考-行动-观察”循环来增强全景场景图生成视觉感知的智能体框架。PSGAgent使多模态大语言模型能够利用两种工具动态优化其视觉感知：用于对小型或边界关系进行局部探索的区域放大工具，以及用于关系聚焦分割、过滤无关上下文噪声的视觉高亮工具。为了优化决策过程，我们构建了一个两阶段训练流程。

    arXiv:2611.07408v1 Announce Type: cross  Abstract: Panoptic Scene Graph generation (PSG) requires holistic yet fine-grained reasoning over complex relations among objects. However, current Multimodal Large Language Models (MLLMs) applied to PSG suffer from a perception bottleneck: they function as static observers limited by fixed-resolution views. As a result, models often overlook critical relations involving small or peripheral objects and struggle to differentiate ambiguous predicates in complex scenes. To address these limitations, we propose PSGAgent, an agentic framework that enhances visual perception for PSG through a think-act-observe loop. PSGAgent allows MLLMs to dynamically refine their visual perception using two tools: a Region Zoom-In Tool for local exploration of small or boundary relations and a Visual Highlight Tool for relation-focused segmentation that filters out irrelevant contextual noise. To optimize decision-making, we construct a two-stage training pipeline. 
    
[^143]: FreePath：基于神经平均自由程的图像去雾方法

    FreePath: Neural Mean Free Path for Image Dehazing

    [https://arxiv.org/abs/2611.07398](https://arxiv.org/abs/2611.07398)

    提出神经平均自由程框架FreePath，将图像去雾建模为特征空间中的路径状态恢复，通过平均自由程残差块和状态守恒集成实现兼顾物理机制的雾去除。

    

    单幅图像去雾需要从被参与介质削弱和遮蔽的视觉响应中恢复场景辐射。现有方法要么对显式散射变量进行反演，这在复杂场景下可能不稳定；要么学习直接的恢复映射，这往往会丢失雾形成的物理组织结构。我们提出了FreePath，一个用于图像去雾的神经平均自由程框架。FreePath不再估计大气参数，而是受视觉响应在与雾发生交互之前所能传播的期望距离的启发，将去雾表示为特征空间中的路径状态恢复。在残差编码器和解码器结构中，平均自由程残差块学习路径场和碰撞场以引导受雾约束的校正，同时FreePath状态守恒集成在深度特征聚合过程中保留已恢复的路径状态。在配对的CDD-11雾化图像和真实RTTS场景上的实验……

    arXiv:2611.07398v1 Announce Type: cross  Abstract: Single image dehazing requires recovering scene radiance from visual responses that are weakened and veiled by participating media. Existing methods either invert explicit scattering variables, which can be unstable under complex scenes, or learn direct restoration mappings, which often discard the physical organization of haze formation. We present \emph{FreePath}, a neural mean free path framework for image dehazing. Instead of estimating atmospheric parameters, FreePath represents dehazing as feature space path state restoration, inspired by the expected distance a visual response can propagate before interacting with haze. Within a residual encoder and decoder, the Mean Free Path Residual Block learns path and collision fields to guide haze constrained correction, while FreePath State Conservation Integration preserves restored path states during deep feature aggregation. Experiments on paired CDD-11 haze images and real RTTS scene
    
[^144]: WeLM-Image：面向视觉智能的统一图像生成模型

    WeLM-Image: A Unified Image Generation Model towards Visual Intelligence

    [https://arxiv.org/abs/2611.07392](https://arxiv.org/abs/2611.07392)

    WeLM-Image是一个面向大规模消费级应用单GPU部署的统一图像生成基础模型，通过混合Transformer骨干与稀疏混合专家架构将多模态理解融入文本生成图像、指令引导编辑和多参考合成三类任务。

    

    我们提出了WeLM-Image，这是一个统一的图像基础模型，将多模态理解融入文本生成图像、指令引导编辑和多参考图像合成之中。该模型专为微信大规模消费级应用中的单GPU部署而设计，将文本、参考图像和目标潜变量作为一个交错的序列进行处理。其混合Transformer骨干网络将共享注意力与相互独立的理解和生成参数相结合，以限制优化过程中的相互干扰。每条路径均配备混合专家模块以提供稀疏容量，总存储参数达317亿，每个目标潜变量token仅激活约54亿生成参数。一个面向生成的数据引擎将基于知识的视觉词汇表与大规模视觉聚类相结合，以平衡语义覆盖与编辑任务，同时通过训练时评估指导迭代式数据精炼。

    arXiv:2611.07392v1 Announce Type: cross  Abstract: We present \textbf{WeLM-Image}, a unified image foundation model that integrates multimodal understanding into text-to-image generation, instruction-guided editing, and multi-reference synthesis. Designed for single-GPU deployment in WeChat's large-scale consumer applications, the model processes text, reference images, and target latents as one interleaved sequence. A Mixture-of-Transformers (MoT) backbone combines shared attention with independent understanding and generation parameters to limit optimization interference. Equipping each path with Mixture-of-Experts (MoE) modules provides sparse capacity, with 31.7B stored parameters and approximately 5.4B generation parameters activated per target latent token. A generation-oriented data engine combines a knowledge-grounded visual vocabulary with large-scale visual clustering to balance semantic coverage and editing tasks, while training-time evaluation guides iterative data refineme
    
[^145]: EnvOpt：面向智能体环境的自动化优化

    EnvOpt: Automated Optimization for Agent Environments

    [https://arxiv.org/abs/2611.07372](https://arxiv.org/abs/2611.07372)

    EnvOpt提出了对LLM智能体静态环境进行自动化端到端优化的第三条路径，无需评分器或参考答案即可大幅降低智能体成本（最高72.7%）并提升任务通过率。

    

    一个LLM智能体的成本与成功取决于其模型、执行框架及其环境。除了被广泛研究的参数调优和执行框架优化方向之外，我们提出了第三个方向：对静态环境的优化。智能体通常部署在由众多任务共享的丰富环境中，例如公司的Slack工作区，或由数据源、文档、工单组成的复杂集合。我们将这类环境称为静态环境。我们提出了EnvOpt，一种针对任意给定模型和执行框架的静态环境自动化端到端优化方法，它可以使用真实或合成任务，且无需任何评分器或参考答案。在AutomationBench上，EnvOpt将Claude Haiku 4.5的成本降低了72.7%，同时将其通过率提高了14.34%。相同的环境可迁移至Opus 5，将其成本降低68.1%，并带来1.34%的提升。在Meta-Harness针对Haiku 4.5优化的执行框架之上叠加使用时，它可进一步降低（原文在此处截断）……

    arXiv:2611.07372v1 Announce Type: new  Abstract: An LLM agent's cost and success depend on its model, its harness, and its environment. In addition to the widely studied directions of parameter tuning and harness optimization, we propose the third direction: optimization over stationary environments. Agents are often deployed in rich environments shared by many tasks, such as a company's Slack workspace or a complex collection of data sources, documents or tickets. We call these environments stationary environments. We introduce EnvOpt, a method for automated end-to-end optimization of stationary environments for any given model and harness, using real or synthetic tasks without needing any graders or reference answers. On AutomationBench, EnvOpt reduces the cost of Claude Haiku 4.5 by 72.7% and raises its pass rate by 14.34%. The same environment transfers to Opus 5, cutting its cost by 68.1% with a 1.34% gain. Added on top of Meta-Harness's optimized harness for Haiku 4.5, it lowers 
    
[^146]: 测试时自我博弈：无需验证器的扩展方法

    Test-Time Self-Play: Scaling without a Verifier

    [https://arxiv.org/abs/2611.07367](https://arxiv.org/abs/2611.07367)

    该论文提出测试时自我博弈（TTSP），一种基于非对称自我博弈、无需验证器的测试时扩展方法，在无验证器的情况下于 LiveCodeBench v6 上实现了 mean@8 提升 35% 的绝对改进，且验证器越弱其优势越明显。

    

    测试时扩展方法在部署阶段投入额外的计算资源，以提升大语言模型在困难目标问题上的性能。大多数方法依赖于一个验证器来对候选解决方案的正确性进行评分，并利用它来选择或改进模型的输出。然而，在许多部署场景中，针对目标问题的强验证器往往是不可用的。我们提出了测试时自我博弈（Test-Time Self-Play, TTSP），这是一种利用非对称自我博弈、无需依赖验证器的测试时扩展方法。在 LiveCodeBench v6 的难度校准子集上，TTSP 在我们评估的每一种验证器强度下都是表现最佳的方法，并且随着验证器强度的减弱，它与最强基线之间的差距不断扩大；在无验证器的极限情况下，TTSP 在 mean@8 指标上带来了 35% 的绝对提升。这些收益在各种验证器强度下均能保持，除非基础策略对正确答案的覆盖率接近于零。

    arXiv:2611.07367v1 Announce Type: cross  Abstract: Test-time scaling methods spend additional compute at deployment to improve a large language model's performance on a hard target problem. Most methods rely on a verifier that scores candidate solutions for correctness, using it to select among or refine the model's outputs. In many deployment regimes, however, a strong verifier on the target problem is unavailable. We introduce Test-Time Self-Play (TTSP), an approach to test-time scaling using asymmetric self-play without reliance on a verifier. On a difficulty-calibrated subset of LiveCodeBench v6, TTSP is the best-performing method at every verifier strength we evaluate, and the gap to the strongest baseline widens as the verifier weakens; in the no-verifier limit, TTSP yields a 35\% absolute improvement in mean@8. The gains hold across a range of verifier strengths, except when the base policy has near-zero coverage of the correct answer.
    
[^147]: 面向多重违规知识图谱修复的选择性检索路由

    Selective Retrieval Routing for Multi-Violation Knowledge Graph Repair

    [https://arxiv.org/abs/2611.07342](https://arxiv.org/abs/2611.07342)

    该论文提出一种选择性检索路由框架，由LLM修复智能体编辑知识图谱、路由器智能控制对外部证据检索的访问，并利用违规注入生成的修复结果训练特定部署的路由器，从而提升多重违规知识图谱的修复效果。

    

    知识图谱通常包含缺失或不一致的事实，这会损害构建在其之上的应用。模式验证能够检测此类违规，但无法确定多种有效修复方案中哪一种能够恢复预期的图谱。外部语料库可以提供识别预期修复所需的证据，但从中检索的成本高昂且并不总是成功。我们提出了一个面向迭代图谱修复中选择性检索的框架：一个基于大语言模型（LLM）的修复智能体负责探索和编辑图谱，同时一个路由器控制其访问负责检索外部证据的研究员（researcher）智能体的权限。我们扩展了一个违规注入框架，以生成包含多个兼容违规和已知参考修复的案例，智能体在这些案例上的修复结果为训练特定部署场景的路由器提供了监督信号。在两个楼宇自动化数据集上的实验表明，研究员智能体的访问对某些违规能改善修复效果，而对另一些违规则会降低效果……

    arXiv:2611.07342v1 Announce Type: cross  Abstract: Knowledge graphs often contain missing or inconsistent facts that undermine the applications built on them. Schema validation detects such violations but does not identify which of several valid repairs recovers the intended graph. External corpora can supply the evidence needed to identify the intended repair, but retrieving from them is costly and not always successful. We propose a framework for selective retrieval in iterative graph repair. An LLM repair agent explores and edits the graph, while a router controls its access to a researcher agent that retrieves external evidence. We extend a violation-injection framework to generate cases with multiple compatible violations and known reference repairs. The agent's repair outcomes on these cases provide supervision for training the deployment-specific router. On two building-automation datasets, researcher access improves repairs for some violations and degrades them for others, with
    
[^148]: 持续强化学习中的永久性与瞬态表示

    Permanent and Transient Representations for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.07327](https://arxiv.org/abs/2611.07327)

    本文提出在持续强化学习的互补学习系统中为长期学习采用参数化表示、为瞬态适应采用非参数化表示，并引入一种新颖的非参数近似方法来估计瞬态价值函数，从而实现复杂任务中的在线学习并优于竞争性基线方法。

    

    持续强化学习（CRL）智能体难以在适应新情况的同时保留先前学到的知识，从而面临稳定性与可塑性之间的权衡。解决这一问题的一种方法是采用互补学习系统：一个用于缓慢的长期学习，另一个用于快速的瞬态适应。Anand & Precup（2023）通过永久价值函数和瞬态价值函数实现了这一思想，但两个系统使用了相同的特征。直观来看，长期学习应更多依赖参数化表示以实现广泛的泛化能力，而瞬态学习则最适合采用非参数化表示以实现针对特定情境的适应。在本文中，我们从理论和实证两方面对这一思想进行了探索。通过引入一种新颖的非参数近似方法来估计复杂任务中的瞬态价值函数，我们证明了我们的方法能够实现在线学习，并且优于竞争性基线方法。

    arXiv:2611.07327v1 Announce Type: cross  Abstract: Continual Reinforcement Learning (CRL) agents struggle to adapt to new situations while retaining prior knowledge, leading to a stability-plasticity trade-off. One approach to solving this problem is by using complementary learning systems: one for slow, long-term learning and another for quick, transient adaptation. Anand & Precup (2023) instantiated this idea using permanent and transient value functions, but used the same features in both systems. Intuitively, long-term learning should rely more on parametric representations for broad generalization, while transient learning would be best served by non-parametric representations for situation-specific adaptation. In this paper, we explore this idea both theoretically and empirically. By introducing a novel non-parametric approximation for estimating the transient value function in complex tasks, we demonstrate that our method enables online learning and outperforms competitive basel
    
[^149]: 即将来临的热带气旋对港口影响的稳健排序：一种两阶段推荐系统

    Robust Ranking of Port Impacts from Impending Tropical Cyclones: A Two-Stage Recommendation System

    [https://arxiv.org/abs/2611.07296](https://arxiv.org/abs/2611.07296)

    该论文提出一种两阶段推荐系统，先根据港口与预期登陆点的距离筛选候选港口，再按预估影响天数排序，从而在登陆位置不确定的条件下稳健地预测热带气旋对各港口的影响及干扰持续时间。

    

    当热带气旋威胁多个港口时，防备规划依赖于预测哪些港口可能受到影响以及哪些地区的干扰可能持续最久。单个港口有限的历史观测数据以及登陆位置的不确定性使得这些相对影响难以估计。我们采用一个两阶段推荐框架，首先根据港口与预期登陆位置的接近程度筛选候选港口，然后按估计的影响天数对其进行排序。基于墨西哥湾研究区域37个港口和31个热带气旋的AIS衍生影响天数数据，我们比较了三种表示港口-气旋交互的方式：成对特征交互、基于树模型的特征组合，以及显式特征交叉与学习嵌入相结合的方法。在飓风Nicole和Ian的测试中，检索阶段在登陆位置存在不确定性的情况下始终能识别出所有观测到的受影响港口。在排序模型中……

    arXiv:2611.07296v1 Announce Type: new  Abstract: When a tropical cyclone (TC) threatens several ports, preparedness planning depends on anticipating which ports are likely to be affected and where disruptions may last longest. Limited historical observations at individual ports and uncertainty in landfall location make these relative impacts difficult to estimate. We use a two-stage recommendation framework that first selects candidate ports based on their proximity to the expected landfall location and then ranks them by their estimated number of impact days. Using AIS-derived impact days for 37 ports and 31 TCs in the Gulf of Mexico study region, we compare three ways of representing port--cyclone interactions: pairwise feature interactions, tree-based feature combinations, and explicit feature crosses combined with learned embeddings. In tests on Hurricanes Nicole and Ian, the retrieval stage consistently identifies all observed affected ports under landfall uncertainty. Among the r
    
[^150]: 智能体应如何“看”代码？重新思考面向智能体软件工程的代码表示

    How Should Agents See Code? Rethinking Code Representations for Agentic Software Engineering

    [https://arxiv.org/abs/2611.07286](https://arxiv.org/abs/2611.07286)

    本文将基于地图的代码表示（抽象语法树、调用图、模块依赖、文档增强搜索及其混合形式）作为智能体软件工程的设计变量，系统评估了五种代码表示工作流如何影响智能体的检索行为、任务成功率、token消耗与延迟。

    

    大型语言模型智能体正越来越多地被用于仓库级软件工程任务。然而，大多数智能体仍通过原始文件、词法搜索和shell命令来访问代码仓库，主要将代码视为纯文本。本文将基于“地图”式代码表示的工作流作为智能体软件工程中的一个设计变量进行研究，考察不同的仓库视图如何改变智能体的检索行为、任务成功率、token消耗和延迟。本文研究了五种代码表示工作流：抽象语法树、调用图、模块依赖、文档增强搜索，以及将它们结合起来的混合工作流。每种工作流都为智能体提供了一张可在任务执行期间查询的仓库地图。我们在四个仓库级任务上，使用搭载Haiku 4.5的Claude Code与原始工具基线进行对比评估：利用LinuxFLBench进行故障定位、利用SWE-bench Pro进行仓库级代码生成与修复、利用SWT-bench进行测试生成（原文此处截断）。

    arXiv:2611.07286v1 Announce Type: cross  Abstract: LLM agents are increasingly used for repository-level software engineering tasks. However, most agents still access repositories through raw files, lexical search, and shell commands, treating code primarily as text. This paper investigates map-based code-representation workflows as a design variable in agentic software engineering, examining how different repository views change agents' retrieval behavior, task success, token use, and latency.   In this paper, we study five code-representation workflows: abstract syntax trees, call graphs, module dependencies, documentation-enriched search, and a hybrid workflow that combines them. Each workflow exposes a repository map that the agent can query during a task. We evaluate Claude Code with Haiku 4.5 against a raw-tool baseline on four repository-level tasks: fault localization with LinuxFLBench, repository-level code generation and repair with SWE-bench Pro, test generation with SWT-ben
    
[^151]: 大语言模型中语法数的编码与传播

    Encoding and Propagation of Grammatical Number in Large Language Models

    [https://arxiv.org/abs/2611.07283](https://arxiv.org/abs/2611.07283)

    该研究利用稀疏自编码器（SAE）深入揭示了 Pythia-70M 小型语言模型中语法数一致现象的表示与处理机制，展示了 SAE 如何用于查找和传播语言模型内部的语法特征。

    

    稀疏自编码器（SAEs）已成为一种日益流行的方法，可将大语言模型（LLM）的隐藏状态向量转换为人类可理解的特征或符号列表。因此，SAEs 可以为我们洞察 LLM 的知识表示和内部“心智”语法提供途径——这对 AI 行为的机制性解释具有实践意义，对研究人类语言的计算机模拟表示也具有理论价值。参数少于 1 亿、仅在小语料库上训练的语言模型，尽管生成内容的可靠性有限，却仍表现出强大的语法能力和语篇连贯性。我们使用在这样一个模型（Pythia-70M）上训练的 SAE，深入研究单一语法现象——数的一致——在 LLM 中是如何被表示和处理的。我们展示了 SAE 可用于查找和传播语法特征的机制与方法。

    arXiv:2611.07283v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) have become an increasingly popular means of translating Large Language Model (LLM) hidden state vectors into a human-comprehensible list of features or symbols. As such, SAEs may provide insight into an LLM's knowledge representation and internal ``mental'' grammar - topics of pragmatic interest for mechanistic interpretation of AI behavior and of theoretical interest for the study of in silico representations of human language. Language models with fewer than 100M parameters, trained on small corpora, nonetheless exhibit strong grammatical competence and discourse coherence, despite producing content of limited reliability. We use SAEs that we trained on one such model (Pythia-70M) to conduct an in-depth examination of how a single grammatical phenomenon, number agreement, is represented and processed in an LLM. We demonstrate the mechanism and the methodology by which SAEs can be used to find and propagate
    
[^152]: 基于潜在噪声表示桥接的高效扩散策略微调

    Efficient Diffusion Policy Fine Tuning with Latent Noise Representation Bridging

    [https://arxiv.org/abs/2611.07275](https://arxiv.org/abs/2611.07275)

    该论文提出潜在噪声线性马尔可夫决策过程（LNL-MDP）及其桥接方程，通过在噪声空间与动作空间之间建立共享潜在表示来消除“噪声混叠”问题，从而实现计算高效且具有近似最优理论保证的扩散策略微调。

    

    预训练的扩散策略与流策略已成为机器人领域强大的骨干模型，然而通过强化学习（RL）将其适配到新数据仍然计算成本高昂、样本效率低下且缺乏理论保证。先前通过微调初始噪声的方法虽然避免了代价高昂的时序反向传播，但难以解决“噪声混叠”问题：噪声空间的取值与动作空间的取值之间不存在自然的联系。本文通过引入潜在噪声线性马尔可夫决策过程（LNL-MDP）来解决这些挑战。我们推导出LNL-MDP的桥接方程，通过在噪声空间与动作空间之间共享表示来对齐价值，从而消除了噪声混叠问题。此外，我们的LNL-MDP微调框架符合FineTuneRL理论，具有近似最优保证。基于这些见解，我们提出了用于扩散策略的潜在表示桥接方法。

    arXiv:2611.07275v1 Announce Type: cross  Abstract: Pre-trained diffusion and flow policies have emerged as powerful backbones for robotics, yet adapting them to novel data via reinforcement learning (RL) remains computationally expensive, sample inefficient and lacks theoretical guarantees. Prior approaches that fine-tune the initial noise bypass costly backpropagation through time but struggle with the "noise aliasing" problem: there does not exista natural link between the noise space value and the action space value. This paper resolves these challenges by introducing the Latent Noise Linear Markov Decision Process (LNL-MDP). We derive the Bridge Equation for LNL-MDPs which eliminates the noise aliasing problem by aligning values with a shared representation between the noise and the action space. Furthermore, our LNL-MDP fine tuning framework falls under the FineTuneRL theory with near-optimal guarantee. Leveraging these insights, we propose Latent Representation Bridging for Diffu
    
[^153]: 满足化作为竞争性贝叶斯加权：未达成目标下目标导向选择的最小模型

    Satisficing as competitive Bayesian weighting: a minimal model of goal-directed choice under an unattained target

    [https://arxiv.org/abs/2611.07270](https://arxiv.org/abs/2611.07270)

    该论文证明“风险敏感满足化”这一多臂老虎机学习规则可等价改写为类贝叶斯估计器“注意力竞争贝叶斯（ACB）”——未达成的抱负水平充当所有动作的先验均值，有限试验预算决定证据覆盖先验的强度，从而给出了以未实现目标为参照的目标导向选择的最小形式化模型。

    

    动物和人类往往并非进行最大化决策；它们相对于尚未实现的目标（抱负水平）而行动。一个尚未实现的目标如何组织选择行为？Deacon（2012）将这种以“缺失”为参照的组织方式称为“意向性的”（ententional），而 García-Valdecasas 与 Deacon（2024）则主张，目的最好被理解为约束而非特定状态。我们在一个最小化的形式化环境中考察这一思想：风险敏感满足化（RS），一种多臂老虎机学习规则，它以每个动作过去被试行的份额乘以其相对于抱负水平的盈余或缺口来评估该动作的价值。我们证明 RS 可以被改写为一个类贝叶斯估计器——注意力竞争贝叶斯（ACB），其中抱负水平充当每个动作的先验均值，而有限的试验预算决定了证据能在多大程度上覆盖这一先验。由此得出两个结论。第一，不再被采样的动作会被部分覆盖……（原文摘要至此截断）

    arXiv:2611.07270v1 Announce Type: cross  Abstract: Animals and humans often do not maximize; they act relative to goals (aspiration levels) that are frequently not yet met. How can a goal that is not (yet) realized organize choice? Deacon (2012) calls such absence-referenced organization \emph{ententional}, and Garc\'{i}a-Valdecasas and Deacon (2024) argue that ends are best understood as constraints rather than as particular states. We examine this idea in a minimal formal setting: Risk-sensitive Satisficing (RS), a bandit learning rule that values each action by its share of past trials multiplied by its surplus or shortfall relative to an aspiration level. We show that RS can be rewritten as a Bayesian-like estimator, Attention Competitive Bayes (ACB), in which the aspiration level is the prior mean of every action and the finite budget of trials sets how strongly evidence can override it. Two consequences follow. First, actions that are no longer sampled are partially overridden by
    
[^154]: 信号有多深？初级保健EHR模型中利用长达30年时间上下文的结直肠癌预测

    How Deep is the Signal? Time-Context up to 30 Years for Colorectal Cancer Prediction in a Primary-Care EHR Model

    [https://arxiv.org/abs/2611.07265](https://arxiv.org/abs/2611.07265)

    该研究首次系统评估了EHR基础模型的“时间上下文”对疾病预测的影响，发现将日历回溯窗口从2年延长至最长30年可显著提升结直肠癌的早期预测性能。

    

    EHR基础模型在固定的上下文窗口内读取医疗事件，而迄今为止长上下文仅以token数量来衡量。模型需要回溯多远的时间才能捕捉到最早的预测信号，目前尚不清楚。我们提出了“时间上下文”（time-context）这一概念，即预测点之前的日历回溯窗口，并据我们所知，首次对EHR基础模型进行了十年尺度时间上下文的系统性评估。我们在来自伦敦西北部360万名初级保健患者的14亿条编码事件上预训练了一个仅解码器Transformer模型（3460万参数），并对其进行微调，以在12个月诊断前缓冲期的条件下预测新发结直肠癌。在一个由1,008例病例和9,582名匹配对照组成的固定队列中（所有患者均注册至少30年），我们在从2年到30年的15个嵌套回溯窗口上进行了评估，使得每次截断都在同一批患者内进行比较。判别性能从2年回溯时的AUROC 0.551提升至……

    arXiv:2611.07265v1 Announce Type: cross  Abstract: EHR foundation models read medical events within a fixed context window, and long context has so far been measured only in tokens. How far back in time a model must look to capture the earliest predictive signal is unknown. We introduce time-context, the calendar lookback window before the prediction point, and provide, to our knowledge, the first systematic evaluation of decade-scale time-context for an EHR foundation model. We pretrained a decoder-only transformer (34.6M parameters) on 1.4B coded events from 3.6M primary-care patients in North West London and fine-tuned it to predict incident colorectal cancer under a 12-month pre-diagnostic buffer. A frozen cohort of 1,008 cases and 9,582 matched controls, all registered for at least 30 years, was evaluated at 15 nested lookback windows from 2 to 30 years, so that every truncation is compared within the same patients. Discrimination rose from an AUROC of 0.551 at 2 years to 0.590 at
    
[^155]: 进化自适应邻域学习：一种用于最近邻分类中实例依赖k优化的遗传算法

    Evolutionary Adaptive Neighborhood Learning: A Genetic Algorithm for Instance-Dependent k Optimization in Nearest Neighbor Classification

    [https://arxiv.org/abs/2611.07257](https://arxiv.org/abs/2611.07257)

    该论文提出EANK方法，通过遗传算法优化一个低维策略，根据局部邻域特征（密度、标签熵、投票边际、集中度）为每个实例自适应地确定k近邻分类器的邻居数量k(x)，突破了传统全局单一k值选择的局限性。

    

    邻居数量k是k近邻（k-NN）分类器的核心超参数，通常通过交叉验证或进化搜索进行全局选择。然而，偏差-方差权衡本质上是局部性的：密集且无噪声的区域可能适合较小的邻域，而稀疏或有噪声的区域则可能需要更大的邻域。由于全局k优化是一维问题，用遗传算法替代穷举搜索几乎没有计算优势。因此，我们转而优化实例依赖的邻域策略。我们提出了EANK（进化自适应邻域K），它使用一个低维策略将局部邻域描述符（包括密度、标签熵、投票边际和集中度）映射到自适应的邻居数量k(x)。一个实数编码的精英遗传算法使用带邻域平滑性惩罚的留一法准确率来优化该策略。

    arXiv:2611.07257v1 Announce Type: cross  Abstract: The number of neighbours k is the central hyperparameter of the k-nearest neighbour (k-NN) classifier and is usually selected globally through cross-validation or evolutionary search. However, the bias-variance trade-off is inherently local: dense and noise-free regions may favour small neighbourhoods, whereas sparse or noisy regions may require larger ones. Since global-k optimization is one-dimensional, replacing exhaustive search with a genetic algorithm offers little computational advantage. We therefore optimize an instance-dependent neighbourhood policy instead. We propose EANK (Evolutionary Adaptive Neighborhood K), which uses a low-dimensional policy to map local neighbourhood descriptors, including density, label entropy, vote margin, and concentration, to an adaptive number of neighbours k(x). A real-coded elitist genetic algorithm optimizes this policy using leave-one-out accuracy with a neighbourhood-smoothness penalty. Pre
    
[^156]: TriAttention：面向长上下文LLM推理的统一稀疏混合精度注意力机制

    TriAttention: Unified Sparse Mixed-Precision Attention for Long-Context LLM Inference

    [https://arxiv.org/abs/2611.07256](https://arxiv.org/abs/2611.07256)

    TriAttention提出了一种统一的稀疏混合精度注意力方法，通过离线校准为每个注意力头在跳过、4位和8位计算之间分配计算区域，在共享计算预算下兼顾稀疏性与量化精度，从而优化长上下文LLM推理的成本与精度平衡。

    

    稀疏化和量化通过跳过部分交互以及降低所保留交互的精度来降低注意力机制的计算成本。然而，有用的交互对精度的需求各不相同：有些交互能够容忍低精度评分，但对被移除却十分敏感。因此，我们提出了TriAttention，它将查询-键（QK）分块分配给跳过、4位或8位三种计算方式，利用4位计算以更低的成本保留那些可容忍量化的交互。为了实现紧凑表示，我们使用两个对角距离边界来编码每个注意力头的计算区域，使策略存储与上下文长度无关。随后，离线校准过程使用归一化投影输出误差和完整分块执行成本来评估候选边界对。基于这些测量结果，一个模型级分配器在共享计算预算下为每个注意力头选择边界，以最小化校准误差。在预填充阶段，所选边界会生成精确……（摘要原文在此处截断）

    arXiv:2611.07256v1 Announce Type: cross  Abstract: Sparsity and quantization reduce the cost of attention by skipping interactions and lowering the precision of those retained. However, useful interactions differ in their precision requirements: some tolerate low-precision scores while remaining sensitive to removal. We therefore introduce TriAttention, which assigns query-key (QK) tiles to skipping, 4-bit, or 8-bit computation, using 4-bit to retain quantization-tolerant interactions at lower cost. To represent compactly, we encode each head's computation regions with two diagonal-distance boundaries, keeping policy storage independent of context length. Offline calibration then evaluates candidate boundary pairs using normalized projected-output errors and complete-tile execution costs. Using these measurements, a model-wide allocator selects each head's boundaries to minimize calibration error under a shared computation budget. During prefill, the selected boundaries generate precis
    
[^157]: 目标揭示：连接信息论赋权与基于能力的人类权力的桥梁

    Goal Revelation as a Bridge Between Information-Theoretic Empowerment and Capability-Based Human Power

    [https://arxiv.org/abs/2611.07247](https://arxiv.org/abs/2611.07247)

    该论文提出“目标揭示”（即人类潜在目标与实际轨迹之间的互信息）作为统一两种权力度量的桥梁：基于能力的个体权力通过Fano不等式构成其下界，而信息论赋权构成其上界，从而为AI安全中的权力量化研究建立了理论联系。

    

    权力对AI安全至关重要。最优智能体往往会在广泛的目标上获取权力（即工具性趋同）。随着我们将AI智能体嵌入社会技术系统，人类面临着逐渐失权的风险。这两种担忧都呼唤一种形式化、定量化的权力度量。在马尔可夫博弈的背景下，存在两种这样的度量：其一是信息论赋权，即从智能体动作到未来状态的信道容量；其二是基于能力的个体权力，这是一种由“人类在追求候选目标族中的每个目标时达成该目标的概率”所构建的目标达成度量。我们展示了这两种度量在何处交汇。从个体权力出发，Fano不等式给出了一个下界 $B(s)$。该下界是我们所称的“目标揭示”的下界——目标揭示即人类潜在目标与实际实现轨迹之间的互信息 $\mathbb{I}(G;\xi \mid s)$。而赋权则从上方对目标揭示加以约束，使得 $B(s) \le \mathbb{I}$……（原文摘要至此截断）

    arXiv:2611.07247v1 Announce Type: new  Abstract: Power matters for AI safety. Optimal agents tend to acquire power across a wide range of objectives (instrumental convergence). As we embed AI agents into socio-technical systems, humans risk gradual disempowerment. Both concerns call for a formal, quantitative measure of power. Two such measures in the context of Markov games are information-theoretic empowerment, the capacity of the channel from an agent's actions to future states, and capability-based individual power, a measure of goal attainment built from the probabilities that a human would attain each goal in a family of candidates were they pursuing it. We show where the two meet. From individual power, Fano's inequality yields a floor $B(s)$. This floor is a lower bound on what we term goal revelation, the mutual information $\mathbb{I}(G;\xi \mid s)$ between the human's latent goal and the realised trajectory. Empowerment caps revelation from above, so that $B(s) \le \mathbb{I
    
[^158]: KG-Neighbor：面向嵌入可视化的知识引导邻域构建

    KG-Neighbor: Knowledge-Guided Neighborhood Construction for Embedding Visualization

    [https://arxiv.org/abs/2611.07246](https://arxiv.org/abs/2611.07246)

    本文提出知识引导邻域图构建方法KGN，利用从实例中抽取的知识图谱三元组的共享程度作为语义证据来调整嵌入距离并重建top-k邻域图，从而在不使用类别标签的情况下提升嵌入可视化中邻域关系的可靠性，并配套提出IDProfile工具来解释类别标签与邻域多数类之间的残余不匹配。

    

    在嵌入可视化中，相邻的实例通常被解读为语义相关，然而由输入嵌入导出的邻域关系可能并不反映内容层面的真实关系。基于邻域的降维（DR）方法通过优化基于嵌入的邻域图来生成布局，可能会延续这一局限性。我们提出了知识引导邻域图构建方法（KGN），它将从每个实例中提取的关系知识表示为一组知识图谱（KG）三元组，并利用三元组的共享程度作为成对语义证据来调整嵌入距离并重建 top-k 邻域图；无论是 KG 抽取还是邻域构建均不使用类别标签。我们还提出了 IDProfile（IDP），通过独立评估源内容是否支持所分配的类别标签以及邻域多数类，来解释残余的不匹配情况。在八个文本和图像数据集上……

    arXiv:2611.07246v1 Announce Type: cross  Abstract: In embedding visualizations, nearby instances are interpreted as semantically related, although neighborhoods derived from input embeddings may not reflect content relationships. Neighborhood-based dimensionality reduction (DR) methods can propagate this limitation by optimizing layouts from embedding-based neighborhood graphs. We propose Knowledge-Guided Neighborhood Graph Construction (KGN), which represents relational knowledge derived from each instance as a set of knowledge graph (KG) triples. It uses the degree of triple sharing as pairwise semantic evidence to adjust embedding distances and reconstruct the top-k neighborhood graph; neither KG extraction nor neighborhood construction uses class labels. We also introduce IDProfile (IDP), which interprets residual mismatches by independently assessing whether the source content supports the assigned class label and the neighborhood-majority class. Across eight text and image datase
    
[^159]: MAC-v2：从像素出发的长时程基于模型的价值扩展

    MAC-v2: Long-horizon model-based value expansion from pixels

    [https://arxiv.org/abs/2611.07245](https://arxiv.org/abs/2611.07245)

    MAC-v2通过训练无重构的潜在世界模型并结合动作分块，实现了从像素出发的长时程基于模型的价值扩展，有效缓解了表征学习与动力学预测误差在想象轨迹上的递归累积问题。

    

    我们研究了长时程基于模型的价值扩展能否作为基于像素的离线强化学习（RL）的有效框架。从像素进行长时程基于模型的价值扩展比基于状态的环境要困难得多，因为智能体必须同时学习表征及其动力学，而来自任一方面的误差都会在想象的轨迹上递归累积。我们提出了MAC-v2，它通过训练一个无重构的潜在世界模型并结合动作分块，实现了从像素进行的长时程基于模型的价值扩展。通过从预测和控制目标中学习表征，MAC-v2无需编码可能阻碍动力学预测的、与控制无关的视觉细节。动作分块则通过减少价值扩展所需的递归模型预测次数，进一步缓解了预测误差问题。在包括视觉OGBench、MMBench2、Robomimic等多样视觉基准上的实验表明……

    arXiv:2611.07245v1 Announce Type: cross  Abstract: We study whether \emph{long-horizon model-based value expansion} can serve as an effective framework in pixel-based offline reinforcement learning (RL). Long-horizon model-based value expansion from pixels is substantially harder than state-based environments, as the agent must learn both the representation and its dynamics, and errors from either can accumulate recursively over imagined trajectories. We introduce MAC-v2, which enables long-horizon model-based expansion from pixels by training a reconstruction-free latent world model with action chunking. By learning representations from prediction and control objectives, MAC-v2 does not need to encode control-irrelevant visual details that can hinder the dynamics prediction. Action chunking further mitigates the prediction error by reducing the number of recursive model predictions required for value expansion. Across diverse visual benchmarks including visual OGBench, MMBench2, Robom
    
[^160]: 语言模型神经元中的原型语境：一种智能架构？

    Archetypal contexts in the neurons of language models: an architecture for intelligence?

    [https://arxiv.org/abs/2611.07234](https://arxiv.org/abs/2611.07234)

    该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。

    

    我们验证了此前的假设，即语言模型中存在着独立于具体实例化领域的关联模式（Nordfors, 2026）。这些“原型语境”是人们在众多领域中都能识别出的关系模式，例如“临界点”或“赢家通吃”，它们只被存储一次，便可跨领域应用。我们在模型 Qwen3.8-27B 内部展示了这一现象：一组神经元承载原型（如临界点），另一组承载主题（如政治、生态学或细胞生物学），第三组承载语言（如英语、西班牙语或中文）。当使承载“临界点”原型的神经元沉默后，模型不再识别该模式，但仍然知道文本讲的是什么；当使承载主题的神经元沉默后，模型失去主题信息，但仍能说出该模式。在神经元映射中，各层自组织成一种“三明治”结构：外层承载语言，其内的中间层承载主题，而最核心的层承载原型。三明治……（原文摘要在此处截断）

    arXiv:2611.07234v1 Announce Type: cross  Abstract: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sand
    
[^161]: SPILLWAY：将目标准入与净化分离，以防御工具使用型智能体免受提示注入攻击

    SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection

    [https://arxiv.org/abs/2611.07222](https://arxiv.org/abs/2611.07222)

    SPILLWAY利用工具返回值的类型化字段（如发件人、账户）与自由文本的区别，提出一种无需语言模型参与决策或逐任务规划的确定性防御方法：只有当工具输出中出现的目标值由只读工具在类型化字段中返回或被固定委托规则覆盖时，才准入该调用，从而防御工具使用型智能体免受提示注入攻击。

    

    工具使用型智能体会处理电子邮件、文件和支付操作，因此隐藏在网页或共享文档中的指令可能会诱使智能体将数据或资金发送给攻击者。许多运行时防御机制将放行动作的决策交给语言模型，这使得决策依赖于一个可能读取攻击者文本的模型，并且每一步都增加了模型调用。来源防御机制将模型排除在最终检查之外，但需要模型先对每个任务进行规划或路由。而动作的去向其实可以从工具的类型化返回值中读取：电子邮件的发件人或支付的账户出现在类型化字段中，而注入的指令则以自由文本形式出现。SPILLWAY 将这一观察转化为一个无需逐任务规划的确定性底线。当工具输出中出现但不在用户任务中的每个目标值，都由只读工具在类型化字段中返回，或被固定的委托规则所覆盖时，该检查才准入此次调用。一个确定性的净化机制……

    arXiv:2611.07222v1 Announce Type: cross  Abstract: Tool-using agents act on email, files, and payments, so an instruction hidden in a web page or a shared document can make an agent send data or money to an attacker. Many runtime defenses put a language model on the decision that lets an action through, so the decision depends on a model that may read the attacker's text and adds model calls to every step. Provenance defenses keep the model off the final check but need a model to plan or route each task first. Where an action goes can instead be read from the tools' typed returns: an email's sender or a payment's account arrives in a typed field, while injected instructions arrive as free text. SPILLWAY turns this observation into a deterministic floor with no per-task plan. The check admits a call when each target value that appears in tool output but not in the user's task was returned by a read-only tool in a typed field or is covered by a fixed delegation rule. A deterministic sani
    
[^162]: TileQuant：以Tile作为大语言模型混合精度训练后量化的精度分配单元

    TileQuant: Tile as the Precision-Allocation Unit for LLM Mixed-Precision Post-Training Quantization

    [https://arxiv.org/abs/2611.07217](https://arxiv.org/abs/2611.07217)

    TileQuant提出以硬件对齐的二维权重tile为精度分配单元的免训练混合精度训练后量化框架，通过离线校准得到的激活感知与梯度感知评分，让高分tile保留16位精度、其余tile使用8位，从而在降低校准成本与解码开销的同时更好地保持LLM质量。

    

    训练后量化（PTQ）可以降低大语言模型（LLM）的部署成本，但要保持模型质量就必须考虑权重敏感度的不均匀性。统一PTQ使用单一的权重位宽；重要性感知PTQ在该位宽内调整缩放因子或重建权重；按列混合精度则以更高精度保留选定的列。然而，敏感度在列内部也存在差异，因此整列保留同时会保护不那么敏感的权重，而运行时的离群值检测则会增加解码开销。为此，我们提出了TileQuant——一个免训练框架，它以硬件对齐的二维权重tile（分块）作为精度分配单元。离线校准提供两类评分：基于前向统计的输入激活感知评分（校准成本更低），以及基于损失敏感度的梯度感知评分（质量保持能力更强）。在每个投影层的保留预算内，得分最高的tile保持16位精度，其余tile使用8位精度。

    arXiv:2611.07217v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) reduces large language model (LLM) deployment costs, but preserving quality requires accounting for uneven weight sensitivity. Uniform PTQ uses one weight bit-width, importance-aware PTQ adjusts scales or reconstructs weights within that width, and column-wise mixed precision retains selected columns at higher precision. However, sensitivity varies within columns, so whole-column retention also protects less sensitive weights, while runtime outlier detection adds decoding overhead. We therefore introduce TileQuant, a training-free framework that assigns precision to hardware-aligned 2D weight tiles. Offline calibration provides input-activation-aware scoring from forward statistics for lower calibration cost and gradient-informed scoring from loss sensitivity for stronger quality preservation. Within each projection's retention budget, the highest-scoring tiles remain at 16-bit and the rest use 8-bit. T
    
[^163]: 组合式去对齐：审计大型视觉-语言模型的安全性

    Compositional Unalignment: Auditing Safety in Large Vision-Language Models

    [https://arxiv.org/abs/2611.07216](https://arxiv.org/abs/2611.07216)

    本文提出以“去对齐”作为安全对齐的审计手段，揭示大型视觉-语言模型的对齐并不稳健——极少量的监督微调即可达到与多阶段强化学习流程相当的基准分数，且模型对组合性伤害的识别与拒绝能力是相互解耦的。

    

    大型视觉-语言模型（LVLMs）容易受到组合性伤害：一张图像和一个提示词各自单独来看都是无害的，但经过模型的跨模态推理后却可能变得有害。近期的对齐方法在组合安全基准上报告了很高的分数，但LLM裁判只能测量观察到的行为，无法揭示安全性在模型内部被整合得有多稳健。我们认为，“去对齐”为对齐稳健性提供了一种互补的审计手段。我们首先表明，极少量的监督微调即可达到与多阶段强化学习流水线相当的LLM裁判分数，因此仅凭相似的基准分数本身无法确立稳健性。随后我们表明，对组合性伤害的识别与拒绝是相互解耦的：九种标准的方向探测方法能够以最高AUC=0.99检测出组合性伤害，然而消融所恢复出的方向，要么使拒绝行为保持完好，要么破坏模型的连贯性。因此我们……

    arXiv:2611.07216v1 Announce Type: cross  Abstract: Large vision-language models (LVLMs) are vulnerable to compositional harm: an image and a prompt that are benign in isolation become harmful through the model's cross-modal reasoning. Recent alignment methods report strong scores on compositional safety benchmarks, but an LLM judge only measures observed behavior and cannot reveal how robustly safety is integrated into the model. We argue that un-alignment provides a complementary audit of alignment robustness. We first show that minimal supervised fine-tuning reaches an LLM-judge score comparable to a multi-stage reinforcement-learning pipeline, so similar benchmark scores cannot establish robustness by themselves. We then show that recognition and refusal of compositional harm are decoupled: nine standard direction-finding methods detect compositional harm with up to AUC=0.99, yet ablating the recovered directions either leaves refusal intact or destroys model coherence. We therefore
    
[^164]: SeVerus：面向 Rust 程序验证的上下文感知、游戏化基准测试

    SeVerus: Context-Aware, Gamified Benchmarking for Rust Verification

    [https://arxiv.org/abs/2611.07194](https://arxiv.org/abs/2611.07194)

    本文提出 SeVerus，一种上下文感知、游戏化的 Verus 基准测试生成技术，它利用已验证项目的原始证明依赖关系，通过逐步移除证明上下文将验证任务组织成从易到难的任务链，从而更真实地评估编码智能体自主发现并证明辅助引理的能力。

    

    编码智能体在 Verus（一种 Rust 程序验证器）的现有基准测试上取得了很高的成功率。这些基准测试通常直接提供完成证明所需的辅助引理。然而在实践中，这类引理必须被发现并加以证明。为了刻画这一挑战，我们提出了 SEVERUS，这是一种上下文感知的 Verus 基准测试生成技术，它以游戏化的方式将验证任务组织成任务链，每条任务链又被划分为不同难度级别。利用已验证项目中原始的证明依赖关系，SEVERUS 为每个验证目标构建一条任务链，并通过逐步移除证明上下文来创建各个级别。这些级别从最简单的（只需补全单个缺失引理）到最困难的（需要在没有任何辅助引理的情况下证明目标）。一个独立的检查器会拒绝对可执行代码的修改、被削弱的规范以及不健全的证明构造。一种带噪声的二分搜索方法用于估计……（原文在此处截断）

    arXiv:2611.07194v1 Announce Type: cross  Abstract: Coding agents achieve high success rates on existing benchmarks for Verus, a verifier for Rust programs. These benchmarks typically provide the auxiliary lemmas needed to complete a proof. In practice, however, such lemmas must be discovered and proved. To capture this challenge, we present SEVERUS, a context-aware technique to generate Verus benchmarks that organizes verification tasks into quests, each divided into levelsin a gamified manner. Using the original proof dependencies in verified projects, SEVERUS constructs one quest per verification target and progressively removes proof context to create individual levels. These levels range from the easiest, requiring completing only a single missing lemma, to the hardest, requiring proving the target without any supporting lemmas. An independent checker rejects changes to executable code, weakened specifications, and unsound proof constructs. A noisy binary search estimates the highe
    
[^165]: 多步软件工程：编程智能体能否在长时程中持续基于自己编写的代码进行构建？

    Multi-step SWE: Can Coding Agents Keep Building on Their Own Code Over Long Horizons?

    [https://arxiv.org/abs/2611.07166](https://arxiv.org/abs/2611.07166)

    提出长时程基准 Multi-step SWE，包含69个由5-15个依赖步骤组成的软件工程任务链，要求智能体在自身编写的代码上持续迭代开发，最强前沿模型的解决率仅18.5%，揭示了当前编程智能体在长时程自我延续开发中的严重不足。

    

    开发者很少只给编程智能体提出单次请求：他们通常要求智能体先构建一个组件，然后对其进行扩展、升级和优化，而每次操作都是基于智能体自己编写的代码。然而，大多数基准测试一次只评估单个任务。我们提出了 Multi-step SWE，这是一个包含69个软件工程任务的长时程基准，每个任务都是在单一代码库上由5-15个相互依赖的步骤组成的链条。每个步骤都是一个完整的多小时工程请求，它从智能体在前几步生成的代码开始，因此早期的设计决策和错误会延续下去。这些任务涵盖四大类工程工作和九种编程语言，既包括从零开始的构建，也包括现有的开源代码库，并且每一步都基于一个既有的真实标准，该标准同时定义了指令本身及其验证器。在九个前沿模型中，表现最强的模型解决率仅为18.5%，69个任务中有38个没有任何模型能够解决。我们发布了 Mult

    arXiv:2611.07166v1 Announce Type: cross  Abstract: Developers rarely give a coding agent a single request: they ask it to build a component and then to extend, upgrade, and optimize it, each time on code the agent itself wrote. Most benchmarks, however, score one task at a time. We introduce Multi-step SWE, a long-horizon benchmark of 69 software engineering tasks, each a chain of 5-15 dependent steps on a single codebase. Every step is a complete, multi-hour engineering request that starts from the code the agent produced in the preceding steps, so early design decisions and errors carry forward. The tasks span four families of engineering work and nine languages, include both from-scratch builds and existing open-source codebases, and ground every step in an existing source of truth that defines both the instruction and its verifier. Across nine frontier models, the strongest model reaches a solve rate of only 18.5%, and 38 of the 69 tasks are not solved by any model. We release Mult
    
[^166]: 深度还是描述？在主动信息获取中，一步式潜在评分何时能替代前瞻搜索

    Depth or Description? When a One-Step Latent Score Replaces Lookahead in Proactive Information Acquisition

    [https://arxiv.org/abs/2611.07155](https://arxiv.org/abs/2611.07155)

    该论文发现主动信息获取中对深度前瞻搜索的需求往往源于隐藏原因的描述方式——基于能编码决策信息的合适潜在变量构建的一步式评分，可以用极小的搜索代价替代大部分深度前瞻搜索。

    

    主动式智能体必须在行动之前决定询问什么。精确的前瞻搜索能评估问题的各种组合，但代价高昂且依赖于拟合的模型；一步式评分虽然廉价，却可能遗漏这些组合效应。我们的主要发现是：对搜索深度的需求往往取决于隐藏原因的描述方式。基于标签的一步式评分可能无法察觉答案之间的相互作用，而基于合适潜在原因的评分则不会；当该潜在变量编码了决策信息时，它能以极小的搜索代价替代大部分深度前瞻搜索。换一种描述方式，同一个问题可能会使评分失效，而精确值却不受影响，因此应保持潜在变量足够小，并用精确值加以对照验证。从一个语言模型的回答中学习到的概率无法迁移到另一个语言模型，除非规划过程能兼容两者；而直接从token中读取的概率仅相当于几个标签的价值。

    arXiv:2611.07155v1 Announce Type: cross  Abstract: A proactive agent must decide what to ask before it acts. Exact lookahead values combinations of questions but is expensive and leans on a fitted model; a one-step score is cheap but can miss them. Our main finding: the need for depth often lies in how the hidden cause is described. A one-step score on the label can be blind to an interaction among answers. A score on a suitable latent cause is not, and where that latent encodes the decision it replaces most of depth at a fraction of the search. Described differently, the same problem can defeat the score while exact values do not move, so keep the latent small and check it against exact values. Probabilities learned from one language model's answers do not carry to another unless planning allows for both; those read from tokens are worth only a few labels.
    
[^167]: 更多智能体，更可靠吗？运行剖面不确定性下多智能体大语言模型系统的系统级可靠性

    More Agents, More Reliable? System-Level Reliability of Multi-Agent LLMs under Operational-Profile Uncertainty

    [https://arxiv.org/abs/2611.07140](https://arxiv.org/abs/2611.07140)

    该论文提出了将HIP-LLM单模型不精确概率评估与系统故障逻辑贝叶斯网络相结合的框架，用以在运行剖面不确定性下评估多智能体LLM系统的系统级可靠性，并揭示“更多智能体”并不必然更可靠——多数投票只能防范最弱模型，且投票者自身约5%的失效即可基本抵消其优势。

    

    摘要：团队越来越多地基于协作的大语言模型（LLM）智能体来构建软件，并常常假设智能体越多意味着可靠性越高。然而，通常所依赖的证据——平均基准准确率——忽略了部署系统实际面对的输入分布（运行剖面）以及合并各智能体答案的规则。本文在HIP-LLM（一种针对单个LLM的分层不精确概率评估方法）的基础上，提出了一个将其与多智能体系统故障逻辑贝叶斯网络相结合的框架。各智能体的失效概率区间在该网络中传播——其中合并规则本身也可能失效——最终形成一个可靠性包络，用以显示是哪些证据限制了系统的可靠性。在一个提取受管制的化学品危害声明的流水线案例中，对三个模型进行多数投票可以防范最弱的模型，但并未明显超越最佳的单个模型；而如果一个投票者在5%的请求上失效，将基本抹去其相对中间水平模型的优势。更多的智能体能带来更可辩护的可靠性论证……（原文在此处截断）

    arXiv:2611.07140v1 Announce Type: cross  Abstract: Teams increasingly build software from cooperating large language model (LLM) agents, often assuming more agents mean more reliability. Yet the usual evidence, average benchmark accuracy, ignores the input mix a deployed system meets and the rule merging the agents' answers. Building on HIP-LLM, a hierarchical imprecise-probability assessment of single LLMs, we propose a framework combining it with a Bayesian network of the system's failure logic for multi-agent systems. Agent failure-probability intervals propagate through this network, where the combining rule can itself fail, into a reliability envelope showing which evidence limits it. In a pipeline extracting regulated chemical-hazard statements, majority voting over three models protected against the weakest but did not clearly exceed the best, and a voter failing on 5\% of demands would largely erase its advantage over the middle model. More agents yield a more defensible reliab
    
[^168]: 通过软提示调优将大语言模型评审者适配至已观测的人类标注策略

    Adapting LLM Judges to Observed Human Annotation Policy through Soft Prompt Tuning

    [https://arxiv.org/abs/2611.07126](https://arxiv.org/abs/2611.07126)

    本文提出混合软提示调优方法，将文本提示与学习到的连续提示嵌入相结合，使开源LLM评审者在偏见、毒性和个性化维度上与人类标注策略的一致性显著提升，甚至超越托管的专有参考模型。

    

    大语言模型（LLM）越来越多地被用作评估聊天机器人回复的自动化评审者。虽然专有模型能够与人类标注者达成高度一致，但其成本高昂且难以在实际场景中部署。轻量级的开源LLM评审者提供了一种有吸引力的替代方案，但通常与人类评分的一致性较低。本文研究了软提示调优能否使开源LLM评审者更贴近人类标注策略。我们在FairBotBench基准上训练软提示，并在偏见、毒性和个性化等维度上对适配后的评审者进行评估。研究结果表明，混合软提示调优——即把文本提示与学习到的连续提示嵌入相结合——相比相应的未调优基线，能够持续提升与人类标注的一致性。对于Qwen 3.5 9B，适配后的评审者超越了托管的参考模型。

    arXiv:2611.07126v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly employed as automated judges for evaluating chatbot responses. While proprietary models can achieve high agreement with human annotators, they remain costly and difficult to deploy in practical settings. Lightweight open-source LLM judges provide an attractive alternative but typically exhibit lower agreement with human ratings. In this paper, we investigate whether soft prompt tuning can align open-source LLM judges more closely with human annotation policy. We train soft prompts on the FairBotBench benchmark and evaluate the adapted judges across the dimensions of bias, toxicity, and personalization. Our findings demonstrate that hybrid soft-prompt tuning, which combines textual prompts with learned continuous prompt embeddings, consistently improves agreement with human annotations over the corresponding untuned baselines. For Qwen 3.5 9B, the adapted judges surpass both hosted reference
    
[^169]: PromptOps：面向LLM-as-a-Judge系统的迭代式提示词优化

    PromptOps: Iterative Prompt Optimization for LLM-as-a-Judge Systems

    [https://arxiv.org/abs/2611.07123](https://arxiv.org/abs/2611.07123)

    本文提出PromptOps闭环框架，将LLM评判器在部署中与人工评估策略一致性下降的问题定义为“评判器漂移”，通过分离何时优化、应用何种优化流程、选用哪些反馈记录及如何更新提示词这四个运营决策，实现LLM-as-a-Judge系统的持续监控与迭代重新校准。

    

    大语言模型（LLM）越来越多地被用作自动化评判器，用于评估聊天机器人和其他AI系统的行为。然而在部署过程中，随着用户交互和评分分布的变化，它们与预期人工评估策略的一致性可能会下降。现有的自动提示词优化方法主要被作为在静态开发集上的一次性流程来研究，因此对于长期维护LLM评判器提供的指导有限。我们将这一问题定义为“评判器漂移”，并提出了PromptOps——一个用于监控和迭代重新校准LLM-as-a-Judge（LLMaaJ）系统的闭环框架。PromptOps将四个运营决策分离开来：何时进行优化、应用哪种优化流程、使用哪些反馈记录、以及如何更新评判器提示词。我们用五种提示词优化方法对该框架进行了实例化：GEPA、SIMBA、评分标准合成，以及使用……的示例选择方法。

    arXiv:2611.07123v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used as automated judges for evaluating the behavior of chatbots and other AI systems. In deployment, however, their agreement with the intended human evaluation policy may degrade as user interactions and score distributions change. Existing automatic prompt optimization methods are primarily studied as one-shot procedures on static development sets and therefore provide limited guidance for maintaining LLM judges over time. We frame this problem as judge drift and introduce PromptOps, a closedloop framework for monitoring and iteratively recalibrating LLM-as-a-Judge (LLMaaJ) systems. PromptOps separates four operational decisions: when to optimize, which optimization procedure to apply, which feedback records to use, and how to update the judge prompt. We instantiate the framework with five prompt optimization methods: GEPA, SIMBA, Rubric Synthesis, and exemplar selection using either simul
    
[^170]: LLM模型动物园之夜：用于生成预训练大语言模型权重的数据条件流匹配

    A Night at the LLM Zoo: Data Conditioned Flow Matching for Generating Pretrained LLM Weights

    [https://arxiv.org/abs/2611.07103](https://arxiv.org/abs/2611.07103)

    本文构建了首个通过系统变化数据混合（网络文本、书籍、数学、编程、法语）训练的大规模LLM模型动物园，并提出以数据混合为条件的流匹配模型，能够在Gram-PCA压缩权重空间中直接生成完整的预训练LLM权重，且可零样本泛化到未见过的数据混合。

    

    随着权重空间学习从特定任务模型和视觉模型扩展到大语言模型（LLM），一个挑战在于缺乏经过精心策划和分析的LLM模型群体。为了解决这一问题，我们训练了这样一个模型动物园，应用于权重空间生成：我们在51M、124M和355M参数规模的GPT-2模型群体中，系统地改变网络文本、书籍、数学、编程和法语之间的训练数据混合组成，每个规模层级包含N=100个模型。我们通过余弦相似度、功能多样性、留一混合建模和方差分解来分析这些模型群体，结果表明该动物园包含多样化的模型，且模型间的变化由训练数据混合组成所结构化。我们将训练混合组成进行嵌入，以此条件化一个流匹配模型，从而在Gram-PCA压缩的权重空间中生成完整的预训练LLM权重。通过展示对保留混合的零样本泛化能力，并测量连续……

    arXiv:2611.07103v1 Announce Type: cross  Abstract: A challenge to Weight Space Learning as it expands from task specific and vision models to Large Language Models (LLMs) is the absence of a well curated and analyzed LLM population. To address this, we train such a model zoo for application to Weight Space Generation by systematically varying the data mixture composition between web text, books, mathematics, coding, and French across populations of 51M, 124M, and 355M parameter GPT-2 Models, each tier with count N = 100. We analyze these populations by cosine similarity, functional diversity, Leave-One-Mixture-Out modeling, and variance decomposition. We show that the zoo contains diverse models whose variation is structured by training mixture composition. We embed the training mixture composition to condition a Flow Matching model of complete pretrained LLM weights in Gram-PCA compressed weight space. By demonstrating zero-shot generalization to held out mixtures, measuring Continuou
    
[^171]: 二人零和博弈中扩展式恰当均衡的多项式时间计算

    Polynomial-Time Computation of Extensive-Form Proper Equilibria in Two-Player Zero-Sum Games

    [https://arxiv.org/abs/2611.07089](https://arxiv.org/abs/2611.07089)

    本文提出了首个确定性多项式时间算法，可在具有完美记忆的有限二人零和扩展式博弈中精确计算（而非近似）扩展式恰当均衡。

    

    我们给出一个确定性多项式时间算法，用于在具有完美记忆和有理数数据的有限二人零和扩展式博弈中计算精确的扩展式恰当均衡。恰当性是在智能体正规形式下理解的：在某信息集处，某个行动会相对于更优行动受到惩罚，而所有其他局部行为选择（包括玩家自己未来的行动）保持不变。我们的算法为每个信息集分配一个由允许行动概率构成的几何排列多面体；通过缩放双随机表示，将这些约束嵌入到多项式规模的序列形式多面体中，而鞍点随即满足局部 ε-恰当性不等式。为了得到精确均衡而非在任意小数值扰动下的解，我们建立了参数线性系统的符号稳定性，构造了一个持久的可行基，并提取出行为策略的极限。

    arXiv:2611.07089v1 Announce Type: cross  Abstract: We give a deterministic polynomial-time algorithm for computing an exact extensive-form proper equilibrium in a finite two-player zero-sum extensive-form game with perfect recall and rational data. Properness is understood in the agent normal form: an action at an information set is penalized relative to a better action while all other local behavioral choices, including the player's own future actions, remain fixed. Our algorithm assigns each information set a geometric permutahedron of allowable action probabilities. A scaled doubly stochastic representation embeds these restrictions in a polynomial-size sequence-form polytope; a saddle point then satisfies the local $\varepsilon$-properness inequalities. To obtain an exact equilibrium rather than a solution at an arbitrarily small numerical perturbation, we establish sign stability of a parametric linear system, construct a persistent feasible basis, and extract the limits of behavi
    
[^172]: 大规模合成人类与机器推理的符号模型

    Synthesizing symbolic models of human and machine reasoning at scale

    [https://arxiv.org/abs/2611.07088](https://arxiv.org/abs/2611.07088)

    提出行为条件化符号合成（BCSS）方法，利用AI自动搜索可解释的确定性符号程序，大规模建模人类与机器的推理行为，并揭示两者在推理中都存在过度简化和错误推断问题特征依赖关系的共同特征。

    

    符号模型为认知提供了强有力的解释框架，但其应用范围历来受到手工工程需求和计算可处理性的限制。我们展示了人工智能（AI）的进步如何能够扩展符号建模并帮助克服这些工程局限。我们提出了行为条件化符号合成——一种自动搜索可解释、确定性符号程序的方法，这些程序能够可验证地复现行为——并将其应用于对人类和机器在数百个具有挑战性的推理问题上的行为进行建模。通过分析这些程序，我们发现人类和机器都倾向于过度简化，采用仅能部分解释其观察结果的规则，并错误推断问题特征之间的依赖关系。这些共同的行为特征受可用资源的调节，推理努力程度可以预测机器简化和解释其观察结果的程度。

    arXiv:2611.07088v1 Announce Type: new  Abstract: Symbolic models offer a powerful account of cognition, but their scope has historically been limited by the need for hand-engineering and computational tractability. We show how advances in artificial intelligence (AI) can scale symbolic modeling and help overcome these engineering limitations. We introduce Behavior-Conditioned Symbolic Synthesis (BCSS)---an approach for automated search of interpretable, deterministic symbolic programs that verifiably reproduce behavior---and apply it to model both human and machine behavior across hundreds of challenging reasoning problems. Analyzing these programs reveals that both people and machines tend to oversimplify, adopt rules that are partial explanations for their observations, and misinfer dependencies between problem features. These shared signatures are modulated by available resources, with reasoning effort predicting the extent to which machines simplify and explain their observations. 
    
[^173]: 当验证不等于一致：跨智能体LLM到Dafny的形式化

    When Verification Is Not Agreement: Cross-Agent LLM-to-Dafny Formalization

    [https://arxiv.org/abs/2611.07073](https://arxiv.org/abs/2611.07073)

    本文提出跨智能体Dafny形式化协议，证明验证通过并不等于真正独立的推理一致，且一致率对模型能力在规约者与实现者两个角色间的分配极为敏感（较强模型担任实现者时为53.0%，角色互换后骤降至14.3%）。

    

    经过验证的Dafny后置条件只能证明一个公式成立，但并不能保证该结果是通过真正的独立推理得出的。我们提出了一种跨智能体形式化协议：由“规约者”模型将临床试验标准翻译为Dafny谓词，再由“实现者”模型编写一个依据该谓词进行验证的操作性方法。同模型对照实验表明，即使两个模型之间不存在能力差距，从相同文本独立重新提取时的一致率也仅为59.3%。在96份临床试验文档上对四种模型配置进行评估后，我们发现一致性高度依赖于模型能力在两个角色之间的分配方式：GPT-4.1-nano/GPT-4.1-mini配置在较强模型担任实现者时达到53.0%的一致率，而当角色互换时仅为14.3%。测试中最强的配置GPT-5.6-luna/GPT-5.4-mini将一致率提升至34.4%，但仍低于同模型对照基线（原文在此处截断）。

    arXiv:2611.07073v1 Announce Type: new  Abstract: A verified Dafny postcondition certifies that a formula holds, but it does not guarantee that the result was reached through genuine independent reasoning. We introduce a cross-agent formalization protocol where a Specifier model translates clinical trial criteria into a Dafny predicate, and an Implementer model writes an operational method checked against it. A same-model control establishes that even with no capability gap, independent re-extraction from identical text agrees only 59.3% of the time. Evaluating four model configurations on 96 clinical trial documents, we find that agreement is highly sensitive to how model capability is allocated across the two roles: the GPT-4.1-nano/GPT-4.1-mini configuration achieves 53.0% agreement when the stronger model is the Implementer, but only 14.3% when the roles are reversed. The strongest tested configuration, GPT-5.6-luna/GPT-5.4-mini, improves agreement to 34.4% but remains below the sam
    
[^174]: 通过内部表示来塑造与监控模型

    Shaping & Monitoring Models through their Internal Representations

    [https://arxiv.org/abs/2611.07071](https://arxiv.org/abs/2611.07071)

    该论文提出“机制塑造”统一框架，在训练中通过约束模型内部表示所表达和依赖的概念来塑造模型行为，在多领域、多规模、多训练范式下显著减少不良行为同时保持任务性能，并引入监控器类别的层级体系来评估被塑造概念是否仍然可被监控。

    

    我们提出了“机制塑造”，这是一个通过模型内部表示来塑造模型的统一框架。机制塑造在训练过程中应用，使模型遵循关于它们表达哪些概念的规范（规则）（通过其输出或内部表示来衡量，例如不产生有害内容），以及它们依赖哪些概念，即它们用来做决策的依据（例如在筛选简历时不依赖候选人的性别）。跨越多个领域（视觉、医学影像、语言）、多种规模（最高达32B参数）和多种训练范式（分类、SFT、DPO、GRPO），我们发现机制塑造能在保持任务性能的同时显著减少目标行为。由于针对某一表示进行训练也可能改变该表示本身，我们提出了一个问题：被塑造的概念是否仍然可被监控，而这一问题取决于所考虑的监控器类别。因此，我们引入了此类类别的一个层级体系……

    arXiv:2611.07071v1 Announce Type: cross  Abstract: We propose Mechanistic Shaping, a unified framework for shaping models through their internal representations. Mechanistic Shaping is applied during training so that models follow specifications (rules) about which concepts they express, measured via their outputs or internal representations (e.g., no harmful content), and also which concepts they rely on, i.e., what they use to make their decisions (e.g., not a candidate's gender when screening a r\'esum\'e). Across multiple domains (vision, medical imaging, language), scales (up to 32B parameters), and training paradigms (classification, SFT, DPO, GRPO), we find that Mechanistic Shaping substantially reduces targeted behaviors while preserving task performance. Since training against a representation can also modify it, we ask whether shaped concepts remain monitorable, a question that depends on the class of monitors considered. We therefore introduce a hierarchy of such classes, wh
    
[^175]: 部分可观测马尔可夫决策过程中神经强化学习的有限时间收敛性与遗憾保证

    Finite-Time Convergence and Regret Guarantees for Neural Reinforcement Learning in POMDPs

    [https://arxiv.org/abs/2611.07069](https://arxiv.org/abs/2611.07069)

    该论文针对POMDP中采用非线性神经网络价值函数逼近的强化学习，首次建立了有限时间收敛性与遗憾保证，克服了以往理论仅适用于线性逼近或依赖不符合POMDP轨迹特性的独立同分布采样假设的局限。

    

    部分可观测马尔可夫决策过程（POMDP）中的强化学习（RL）由于对历史的长程依赖诅咒而在计算上极具挑战性。幸运的是，近期研究表明，一类满足滤波稳定性假设的POMDP可以很好地被基于有限记忆窗口内截断历史的超状态马尔可夫决策过程（MDP）所近似，其价值差距随窗口长度增加而衰减。然而，针对此类超状态MDP替代模型的现有有限时间收敛性分析，要么局限于过度简化的线性函数逼近，要么依赖于会被POMDP轨迹违背的独立同分布采样oracle，这两者都严重限制了这些理论结果在实际中的适用性。这促使我们通过研究基于非线性神经价值函数逼近的POMDP强化学习来弥合这一差距。我们的主要贡献有三个方面：i) 我们首先……（摘要在此处截断）

    arXiv:2611.07069v1 Announce Type: cross  Abstract: Reinforcement learning (RL) in Partially Observable Markov Decision Processes (POMDPs) is computationally challenging due to the curse of long-range dependence on history. Fortunately, recent studies have shown that a range of POMDPs that satisfy the filter stability assumption can be well approximated by a superstate Markov Decision Process (MDP) based on a truncated history in a finite-memory window, with a value gap that decays with the window length. However, existing finite-time convergence analyses on such superstate MDP surrogates are either limited to over-simplified linear function approximation or rely on an i.i.d. sampling oracle that is violated by POMDP trajectories, both of which severely limit the applicability of these theoretical results in practice. This motivates us to bridge the gap by investigating RL in POMDPs with nonlinear neural value function approximation. Our main contributions are threefold: i) We first est
    
[^176]: 基于上图形式引导探索的协作式多智能体强化学习

    Cooperative Multi-Agent Reinforcement Learning via Epigraph-Form Guided Exploration

    [https://arxiv.org/abs/2611.07066](https://arxiv.org/abs/2611.07066)

    提出EFXPLORER约束探索框架，利用上图重构引入自适应探索预算，将内在奖励与任务目标分离，在保持任务性能的前提下促进多智能体协作行为的发现。

    

    arXiv:2611.07066v1 公告类型：新论文 摘要：在多智能体强化学习（MARL）中发现协作行为极具挑战性，原因在于联合状态-动作空间的组合复杂性，使得仅凭试错难以涌现出协调行为。内在奖励常被用来辅助探索发现，但将其与团队目标简单结合可能会扭曲学习信号，从而损害任务性能。本文提出了EFXPLORER，这是一个约束探索框架，它在保持既有任务性能的约束条件下最大化探索目标。我们通过上图重构（epigraph reformulation）方法求解该问题，并引入自适应探索预算。这一方法将内在奖励与任务目标分离开来，并通过任务可行性来调控探索行为。为进一步鼓励多样化且时间上延展的探索，我们引入了基于后继距离（successor distance）的内在奖励，以捕捉长时……

    arXiv:2611.07066v1 Announce Type: new  Abstract: Discovering cooperative behavior in multi-agent reinforcement learning (MARL) is challenging due to the combinatorial complexity of joint state-action spaces, which hinders the emergence of coordinated behaviors from trial-and-error alone. Intrinsic rewards are often used to aid discovery, but naively combining them with team objectives can distort the learning signal, compromising task performance. In this paper, we propose EFXPLORER, a constrained exploration framework that maximizes exploration objectives subject to a constraint that preserves established task performance. We solve the resulting problem using an epigraph reformulation that introduces adaptive exploration budgets. This approach separates intrinsic rewards from task objectives and regulates exploration through task feasibility. To further encourage diverse and temporally extended exploration, we incorporate a successor distance-based intrinsic reward that captures long-
    
[^177]: SD-LoRA：面向少样本视觉-语言适配的训练时结构蒸馏至LoRA

    SD-LoRA: Training-Time Structural Distillation into LoRA for Few-Shot Vision-Language Adaptation

    [https://arxiv.org/abs/2611.07063](https://arxiv.org/abs/2611.07063)

    SD-LoRA提出一种结构蒸馏框架，仅在训练时通过异构图教师进行token级图推理（建模patch间、全局-局部及视觉-文本关系），并将其效果迁移至轻量级LoRA适配的CLIP模型，在不增加推理开销的前提下提升少样本视觉-语言适配能力。

    

    视觉-语言模型的少样本适配必须平衡两个相互竞争的目标：一方面要从极其有限的监督中学习细粒度的视觉差异，另一方面要保留冻结的类CLIP模型在部署中极具吸引力的效率。现有的轻量级适配方法，包括提示调优、LoRA和基于缓存的适配器，通常只监督全局图像嵌入，因此未能充分利用token级别的结构信息。相反，结构感知方法虽然利用了局部证据，但往往会引入额外的推理时计算。我们提出SD-LoRA，这是一种结构蒸馏框架，仅在训练期间使用token级别的图推理，并将其效果迁移到轻量级的LoRA适配CLIP模型中。在训练期间，一个异构图教师在高分辨率ViT token上建模patch-patch、全局-局部以及视觉-文本关系。SD-LoRA并不在测试时保留该图结构，而是将其结构信息迁移……（原文在此处截断）

    arXiv:2611.07063v1 Announce Type: cross  Abstract: Few-shot adaptation of vision-language models must balance two competing goals: learning fine-grained visual distinctions from very limited supervision and preserving the efficiency that makes frozen CLIP-like models attractive for deployment. Existing lightweight adaptation methods, including prompt tuning, LoRA, and cache-based adapters, typically supervise global image embeddings and therefore underuse token-level structure. Conversely, structure-aware methods exploit local evidence but often introduce additional inference-time computation. We propose SD-LoRA, a structural distillation framework that uses token-level graph reasoning only during training and transfers its effects into a lightweight LoRA-adapted CLIP model. During training, a heterogeneous graph teacher models patch-patch, global-local, and visual-text relations over high-resolution ViT tokens. Rather than retaining this graph at test time, SD-LoRA transfers its struc
    
[^178]: 分割它能画出的东西：用文生图生成器自身的图像来查询它以进行皮肤病变分割

    Segment What It Can Draw: Querying a Text-to-Image Generator with Its Own Images for Skin Lesion Segmentation

    [https://arxiv.org/abs/2611.07061](https://arxiv.org/abs/2611.07061)

    提出免训练方法GENIQ，利用文生图扩散模型自身生成的健康皮肤与病变图像构建更可靠的概念方向，并结合未标注皮肤镜图像中最具代表性的图像块进行细化，从而在无需标注掩码的情况下实现皮肤病变分割。

    

    无需标注掩码的皮肤病变分割需要通过其他线索来指定目标。现有方法依赖于皮肤镜特有的规则、由标注衍生的模型先验，或用户选择的区域。通用的文本到图像扩散Transformer为目标提供了文本接口，但其概念词注意力在皮肤镜图像上并不可靠：在模型的注意力-输出空间中，由词语衍生的“皮肤到病变”方向与区分真实皮肤和真实病变的方向对齐不佳。我们发现，由该模型生成的健康皮肤图像和病变图像所衍生的方向与该判别方向的对齐程度显著更好。基于这一发现，我们提出了GENIQ，一种免训练方法，它从生成的图像中构建初始的皮肤和病变原型，并利用未标注皮肤镜图像中最像皮肤和最像病变的图像块对原型进行细化。该方法沿细化后的方向对测试图像块进行评分，并应用一个门控……（摘要截断）

    arXiv:2611.07061v1 Announce Type: cross  Abstract: Skin lesion segmentation without annotated masks requires the target to be specified through other cues. Existing approaches rely on dermoscopy-specific rules, annotation-derived model priors, or user-selected regions. A general text-to-image diffusion transformer offers a text interface to the target, but its concept-word attention is unreliable on dermoscopy images: in the model's attention-output space, the word-derived skin-to-lesion direction is poorly aligned with the direction separating real skin from real lesions. We show that the direction derived from generated healthy-skin and lesion images is substantially better aligned. Based on this finding, we propose GENIQ, a training-free method that constructs initial skin and lesion prototypes from generated images and refines them using the most skin-like and lesion-like patches from unlabelled dermoscopy images. It scores test patches along the refined direction and applies a gat
    
[^179]: 你错过的只是重新分类：在大规模人群中识别在线健康社区中的基因变异重新解读需求

    Reclassification Is All You Missed: Surfacing Variant Reinterpretation Need in Online Health Communities at Population Scale

    [https://arxiv.org/abs/2611.07054](https://arxiv.org/abs/2611.07054)

    研究构建了基于智能体大语言模型的PRISM流水线，能在大规模在线健康社区中筛查基因变异重新解读相关的讨论，发现123个候选分类差异（其中5个从意义未明升级为致病性），从而触及临床重新联系无法覆盖的患者群体。

    

    基因变异在患者完成检测后经常会在ClinVar数据库中被重新分类，有时会从“意义未明”变为“致病性”，然而对患者进行重新联系的做法并不一致，责任归属不明确且分散于实验室、医疗服务提供者和患者之间。许多受影响的患者在公共在线健康社区中讨论他们的检测结果，而临床重新联系往往无法触及这些渠道。我们构建了PRISM，这是一个基于智能体的大型语言模型流水线，它能筛选这些帖子、检查相关变异是否随后被重新分类，并起草带有免责声明的联系信息（以无接触试运行方式进行）。在来自32个社区的390万条帖子中，智能体预测的与重新解读相关的讨论较为罕见（0.288%），但高度集中于专业遗传学社区（6.30%，而一般医疗建议社区仅为0.10%），且自2014年以来增长了五倍；PRISM发现了123个候选分类差异，其中5个是从“意义未明”提交升级为“致病性”。

    arXiv:2611.07054v1 Announce Type: cross  Abstract: Genetic variants are routinely reclassified in ClinVar after a patient is tested, sometimes from uncertain to pathogenic, yet patient recontact is inconsistent, with no clear duty and responsibility diffused among labs, providers, and patients. Many affected people discuss their results in public online health communities that clinical recontact often does not reach. We built PRISM, an agentic large-language-model pipeline that screens these posts, checks whether the variant was later reclassified, and drafts a disclaimer-bearing outreach message, in a no-contact dry run. Across 3.9 million posts from 32 communities, agent-predicted reinterpretation-relevant discussion is rare (0.288%) but concentrated in specialist genetics communities (6.30% versus 0.10% in general medical-advice communities) and growing fivefold since 2014; PRISM surfaces 123 candidate classification differences, 5 of them from an uncertain submission to pathogenic 
    
[^180]: 闭环智能体评估：分层度量与自校准

    Closed-Loop Agent Evaluation: Layered Measurement and Self-Calibration

    [https://arxiv.org/abs/2611.07051](https://arxiv.org/abs/2611.07051)

    该论文提出面向已部署智能体的“闭环评估”方法，将评估建模为有偏评级者与受限结果下的性能估计，通过片段/轨迹/对话/参与四层级分析、版本化规则、溯源与裁定组件以及智能体工厂生成的评估者智能体，实现可追溯、可自我校准并能反馈至智能体本身的持续评估体系。

    

    当智能体上线后，团队想知道它是否在正常履行职责，而不仅仅是看它是否报错。然而大多数评估是一次性的：跑一次基准测试、看一眼轨迹、给一个单一分数，其结论既无法自我校验，也无法反馈到智能体本身。我们提出闭环评估，一种针对已部署智能体的评估方法。该方法将评估建模为在有偏评级者与受限结果条件下的性能估计问题，并要求每一层级的结论都能追溯到原子数据，且由带版本管理的规则计算得出。该方法涵盖四个分析层级（片段、轨迹、对话与参与），沿执行、实质、过程与结果这一主线进行度量，始终将行为规范作为约束条件，并使用溯源与裁定组件来认证证据与评估者本身。由智能体工厂生成的评估者智能体携带这些组件，每个评估者都配有领域知识包与适配器框架……

    arXiv:2611.07051v1 Announce Type: new  Abstract: Once an agent is live, teams want to know whether it is doing its job, not just whether it throws errors. Yet most evaluation is one-off: a benchmark run, a look at traces, a single score. Its conclusions neither check themselves nor reach the agent. We propose closed-loop evaluation, an evaluation method for deployed agents. It models evaluation as performance estimation under biased raters and censored outcomes, and it requires every level's conclusion to trace back to atomic data and to be computed by versioned rules. The method covers four levels of analysis (span, trace, conversation and engagement), measures along a spine of Execution, Substance, Trajectory and Results, treats Conduct as a constraint throughout, and uses Provenance and Adjudication components to certify the evidence and the evaluators. Evaluator agents generated by an Agent Factory carry these components, each with a domain knowledge pack and an adapter harness. Tw
    
[^181]: 调适适应的进程：人工智能中的达尔文式、拉马克式与智能体导向演化

    Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI

    [https://arxiv.org/abs/2611.07048](https://arxiv.org/abs/2611.07048)

    该论文提出了一个比较达尔文式、拉马克式与智能体导向演化的理论框架，发现当学习足够有效时三种机制的适应速度依次加快，生成更多变体可加速适应，而最优搜索努力取决于学习能力、问题难度与实验成本。

    

    演化在数十亿年间塑造了智能。人工智能（AI）如今提出了一个引人注目的可能性：智能或许会开始引导自身的演化。当AI智能体变得能够修改并评估自身行为时，它们能以多快的速度改进？这种改进又如何保持安全？我们建立了一个理论框架，比较环境变化之后的三种演化方式：达尔文式演化依赖自然选择，拉马克式演化在此基础上加入了可遗传的学习，而智能体导向演化则允许智能体生成并评估行为变体，并自主决定搜索的深度。当学习足够有效时，这些机制表现出逐级加快的适应速度。生成更多变体可以加速适应，而最优搜索努力取决于学习能力、问题难度和实验成本。然而，智能体的能力……（原文摘要在此处截断）

    arXiv:2611.07048v1 Announce Type: new  Abstract: Evolution has shaped intelligence for billions of years. Artificial intelligence (AI) now raises a striking possibility: intelligence may begin to direct its own evolution. As AI agents become capable of modifying and evaluating their own behavior, how quickly could they improve, and how can that improvement remain safe? We develop a theoretical framework comparing Darwinian, Lamarckian, and agent-directed evolution following environmental change. Darwinian evolution relies on natural selection, Lamarckian evolution adds inherited learning, and agent-directed evolution allows agents to generate and evaluate behavioral variants and choose how extensively to search. When learning is sufficiently effective, these mechanisms exhibit progressively faster adaptation. Generating more variants accelerates adaptation, while optimal search effort depends on learning ability, problem difficulty, and experimentation costs. However, an agent's abilit
    
[^182]: 面向智能体AI的信任响应式运行时授权——策略行为与参数敏感性的实证刻画

    Trust Responsive Runtime Authorization for Agentic AI with Empirical Characterization of Policy Behavior and Parameter Sensitivity

    [https://arxiv.org/abs/2611.06996](https://arxiv.org/abs/2611.06996)

    该论文提出并实证刻画了信任响应式授权策略（TRAP），通过非对称规则将信任、稳定性与风险综合而成的授权就绪分数映射为运行时授权级别，实验表明其在突发退化时能即时限制权限、恢复后更快重新授权，并在各自适应策略中具有最高的稳定性。

    

    arXiv:2611.06996v1 公告类型：新 摘要：将自主智能体运行时状态评估转化为操作权限的策略很少被明确规定或测量。我们对信任响应式授权策略（TRAP）进行了实证刻画，该策略通过一条非对称规则，将授权就绪分数（ARS）——即运行时信任、操作稳定性与情境风险的加权组合——映射到离散的授权级别上：权限提升需要连续多次观测值高于区间边界并附加一个余量，而就绪度的急剧下降则会立即限制权限。在一组匹配的100条蒙特卡洛轨迹上，我们将TRAP与静态策略、仅信任策略、仅风险策略以及对称策略进行比较。在突发退化情形下，TRAP在退化发生时即限制权限，并在恢复后比对应对称策略更快地重新提升权限，在自适应策略中表现出最高且（描述性地）方差最小的稳定性，尽管其恢复后的权限提升中更多发生在所选阈值的下方……

    arXiv:2611.06996v1 Announce Type: new  Abstract: The policy that converts runtime estimates of an autonomous agent's condition into operational authority is rarely specified or measured. We empirically characterize the Trust Responsive Authorization Policy (TRAP), which maps an Authorization Readiness Score (ARS), a weighted combination of runtime trust, operational stability, and contextual risk, onto discrete authorization levels through an asymmetric rule: escalation requires consecutive observations above a band edge plus a margin, while a sharp drop in readiness restricts authority at once. On one matched set of 100 Monte Carlo traces, we compare TRAP with static, trust-only, risk-only, and symmetric policies. Under abrupt degradation, TRAP restricts at onset and re-escalates sooner after recovery than the symmetric policy, with the highest and, descriptively, least variable stability among the adaptive policies, although more of its post-recovery escalations occur below the selec
    
[^183]: 自主AI智能体在证明形式化中系统性地偏离人类偏好

    Autonomous AI Agents Systematically Diverge From Human Preferences in Proof Formalization

    [https://arxiv.org/abs/2611.06995](https://arxiv.org/abs/2611.06995)

    论文提出“模拟协作”方法，基于Lean中九次大型人类数学协作的数据，首次系统揭示了AI智能体在证明形式化中与人类偏好的系统性差异——AI更容易被明确标记的空白吸引，回避密集连接的证明中心和复杂证明目标，并倾向于将出人意料的证明主体规范化。

    

    人工智能（AI）系统开创了一个证明丰产的新时代，但帮助人类理解这些证明，需要我们重新关注AI智能体与人类在完成证明任务时方式上的差异。我们提出了“模拟协作”这一方法学，使我们能够研究两个关键问题：（1）AI智能体对表层语法与更深层底层语义的相对关注度；（2）AI智能体对下一步行动的偏好如何以及在何处偏离人类偏好。基于Lean中九次大型人类协作（涵盖470万个定义、超过6000次提交）的数据，我们发现，在证明构建过程中，AI智能体（1）比人类更容易被明确标记的证明空白所吸引；（2）相对回避连接密集的证明核心区域以及复杂或出人意料的证明目标；（3）相对容易被出人意料的证明主体所吸引，并倾向于将其规范化处理。

    arXiv:2611.06995v1 Announce Type: new  Abstract: Artificial intelligence (AI) systems have ushered in a new age of proof abundance, but helping humans understand these proofs requires new attention to the differences in how AI agents and humans pursue a proof-making task. We present "simulated collaboration", a methodology that allows us to examine two key questions: (1) the relative attention AI agents pay to surface syntax or, conversely, deeper underlying semantics, and (2) how and where AI agent preferences for what to do next diverge from human preferences. On a set of nine major human collaborations in Lean with 4.7M definitions across more than 6K commits, we find that, over the course of proof construction, AI agents (1) are far more strongly drawn than humans to explicitly marked gaps; (2) show a relative aversion to densely connected proof centers and to complex or surprising proof goals; and (3) are relatively attracted to surprising proof bodies, which they tend to regulari
    
[^184]: 面向视觉-语言模型鲁棒直推式自适应的批条件语义锚点

    Batch-Conditioned Semantic Anchors for Robust Transductive Adaptation of Vision--Language Models

    [https://arxiv.org/abs/2611.06993](https://arxiv.org/abs/2611.06993)

    提出JPTA框架，将直推式视觉-语言模型自适应建模为批条件语义参考估计，通过联合估计批次先验与软图像侧原型，仅在经校准的证据下让CLIP文本语义参考向批次结构偏移，从而在利用批次信息与保留文本语义之间实现鲁棒平衡。

    

    直推式自适应通过在未标注批次上细化预测来在测试时改进视觉-语言模型，但现有方法面临一个关键权衡。激进的方法利用批次结构，但当批次稀疏、存在先验偏移或仅覆盖标签空间的一小部分时可能发生漂移。保守的方法能更好地保留CLIP的文本语义，但固定的语义参考即使当批次提供了强矫正证据时也可能适应不足。我们提出JPTA（联合先验与传输自适应），这是一个先验感知框架，它将直推式VLM自适应转化为批条件语义参考估计问题。JPTA既不将CLIP文本原型视为固定先验，也不将其视为可自由替换的对象，而是将其视为仅在经过校准的证据下才向图像侧批次结构偏移的语义参考。它通过估计批次先验和软图像侧原型来形成传输后的语义参考，其影响程度由b……（摘要在此处被截断）

    arXiv:2611.06993v1 Announce Type: cross  Abstract: Transductive adaptation improves vision-language models at test time by refining predictions on an unlabeled batch, but current methods face a key trade-off. Aggressive approaches exploit batch structure but can drift when batches are sparse, have prior shift, or cover little of the label space. Conservative methods better preserve CLIP text semantics, but fixed semantic references can under-adapt even when the batch provides strong corrective evidence. We propose JPTA (Joint Prior and Transport Adaptation), a prior-aware framework that casts transductive VLM adaptation as batch-conditioned semantic reference estimation. Instead of treating CLIP text prototypes as fixed priors or freely replaceable, JPTA views them as semantic references that shift toward image-side batch structure only under calibrated evidence. It estimates batch priors and soft image-side prototypes to form transported semantic references whose impact is scaled by b
    
[^185]: 双账本模型：跨云、跨模型与记录系统的多智能体AI系统事务粒度成本核算

    The Two-Ledger Model: Transaction-Grain Cost Accounting for Multi-Agent AI Systems Across Clouds, Models, and Systems of Record

    [https://arxiv.org/abs/2611.06989](https://arxiv.org/abs/2611.06989)

    该论文提出“双账本模型”，通过将符合FOCUS规范的结算级账单数据与基于OpenTelemetry GenAI遥测构建的调用粒度归因账本相连接，解决了多智能体AI系统中结算粒度账单与业务事务粒度成本核算之间的粒度不匹配问题。

    

    企业通过跨越云平台、大语言模型（LLM）、编排框架和记录系统的多智能体AI系统来运行业务事务。财务部门只能以结算粒度接收AI成本数据，并按SKU和计费周期进行汇总。“这笔订单、理赔或案例究竟花费了多少AI成本？”这一业务问题无法从账单数据中得到解答，因为服务提供商永远无法看到企业的事务边界。FinOps开放成本与使用规范（FOCUS）为账单数据提供了跨提供商的统一模式，但即使新增了AI令牌和请求者归因功能，其粒度仍停留在结算粒度。我们形式化了这种粒度不匹配问题，推导出七项需求，并提出双账本模型：一个由符合FOCUS规范的账单数据构成的结算账本，与一个基于OpenTelemetry GenAI遥测构建的、包含调用粒度使用记录的归因账本相连接。通过分层关联主干（追踪上下文、A2A/MCP元数据、业务键关联等）……（摘要原文在此处截断）

    arXiv:2611.06989v1 Announce Type: new  Abstract: Enterprises run business transactions through multi-agent AI systems spanning clouds, large language models (LLMs), orchestration harnesses, and systems of record. Finance receives AI cost only at settlement grain, aggregated by SKU and charge period. The business question (what did this order, claim, or case cost in AI?) cannot be answered from billing data, because providers never see enterprise transaction boundaries. The FinOps Open Cost and Usage Specification (FOCUS) gives billing data one schema across providers but remains settlement-grain even as it adds AI token and requester attribution. We formalize this grain mismatch, derive seven requirements, and propose the Two-Ledger Model: a settlement ledger of FOCUS-conformant billing data joined to an attribution ledger of invocation-grain usage records built on OpenTelemetry GenAI telemetry. A layered correlation spine (trace context, A2A/MCP metadata, business-key correlation, pro
    
[^186]: 大模型，小收益：MONSTER 基准上作为分类器的冻结预测模型

    Big Models, Small Gains: Frozen Forecasters as Classifiers on MONSTER

    [https://arxiv.org/abs/2611.06988](https://arxiv.org/abs/2611.06988)

    该论文在 MONSTER 基准上对冻结的预测基础模型（Reverso 和 Chronos-Bolt）表示进行受控评测，发现预训练相比随机权重模型仅带来小幅且不稳定的分类准确率提升，表明这些预训练表示未必优于无需预训练的强分类器，尤其在标注稀缺时其价值存疑。

    

    预测基础模型越来越多地被用作时间序列分类的冻结特征提取器，但它们的预训练表示是否比无需预训练的强分类器具有优势仍不清楚。这引出了一个重要问题：这些表示是否提供了模型架构本身所能捕获之外的有用信息，尤其是在标注数据稀缺的情况下？我们在 MONSTER 基准的六个大型数据集上，分别使用可用训练标签的 1%、10% 和 100%，对 Reverso 和 Chronos-Bolt 两个预测模型进行了受控基准测试。我们将它们的冻结表示与四种成熟的时间序列分类方法以及相同预测模型的随机权重版本进行比较。结果表明，在六个数据集中的五个上，预训练比随机权重的对应模型提升了分类准确率，收益最高达 23 个百分点……（原文截断）

    arXiv:2611.06988v1 Announce Type: cross  Abstract: Forecasting foundation models are increasingly used as frozen feature extractors for time series classification, but it remains unclear whether their pretrained representations offer an advantage over strong classifiers that do not require pretraining. This raises an important question: do these representations provide useful information beyond what the model architecture alone can capture, particularly when labelled data are scarce? We present a controlled benchmarking of two forecasting models, Reverso and Chronos-Bolt, on six large datasets from the MONSTER benchmark, using 1%, 10%, and 100% of the available training labels. We compare their frozen representations against four established time series classification methods and random-weight versions of the same forecasting models. Our results show that pretraining improves classification accuracy over random-weight counterparts on five of the six datasets, with gains of up to 23 per
    
[^187]: 通过弹性专家并行实现资源高效的强化学习后训练

    Resource-Efficient Reinforcement Learning Post-Training through Elastic Expert Parallelism

    [https://arxiv.org/abs/2611.06984](https://arxiv.org/abs/2611.06984)

    提出 FlexFloor 系统，通过联合调整响应调度与活跃 worker 组，利用历史响应长度和内存约束实现同步响应生成过程中 worker 的渐进式提前释放，从而大幅提升混合专家模型强化学习后训练的资源效率。

    

    强化学习后训练能够提升大型语言模型的能力，但生成训练响应是一个主要的计算瓶颈。在混合专家模型中，已完成本地响应生成的 worker 可能仍处于占用状态，因为它们持有的专家仍被正在处理更长响应的 worker 所需要。因此，在基线方法中，约三分之一的 worker 占用时间发生在本地响应生成完成之后。本文提出了 FlexFloor，一个能够在同步响应生成过程中更早释放资源、同时保留进行中响应所需专家计算的系统。FlexFloor 联合调整响应调度与活跃 worker 组：它利用历史响应长度和内存约束来指导批处理与提示词放置，然后渐进式地释放 worker，同时保持专家的可用性和通信的正确性。与 verl 基线相比……

    arXiv:2611.06984v1 Announce Type: new  Abstract: Reinforcement learning post-training improves the capabilities of large language models, but generating training responses is a major computational bottleneck. In mixture-of-experts models, workers that finish their assigned responses may remain occupied because their experts are still needed by workers processing longer responses. Consequently, approximately one-third of occupied worker time in the baseline occurs after local response generation finishes. Here we present FlexFloor, a system that enables earlier resource release during synchronous response generation while preserving the expert computation required by ongoing responses. FlexFloor jointly adapts response scheduling and the active group of workers. It uses historical response lengths and memory constraints to guide batching and prompt placement, then progressively releases workers while maintaining expert availability and correct communication. Compared with the verl basel
    
[^188]: BlockLock：通过固定批次几何形状实现可复现的LLM推理

    BlockLock: Reproducible LLM Inference with Fixed Batch Geometry

    [https://arxiv.org/abs/2611.06977](https://arxiv.org/abs/2611.06977)

    BlockLock通过仅固定受保护请求所见的批次几何形状（而非约束所有算子或放弃共享计算），在保留共享批次预填充的前提下实现了逐位可复现的LLM推理。

    

    可复现的LLM推理要求无论批次中的其他请求如何，某个请求都能产生相同的输出。现有解决方案以不同方式为此付出代价：批次不变内核需要约束每一个算子，而通过固定形状解码隔离预填充（我们的CoRun风格基线）则放弃了共享的预填充计算并对解码进行填充。我们提出了BlockLock，它保留共享的批次预填充，仅固定受保护请求的数值路径：它固定请求所看到的几何形状，并提供两种模式：BlockLock-Rows使用固定宽度的行并保留所有请求的批次解码；BlockLock-Pack使用终端锚定打包并配以隔离的虚拟段，然后提取受保护请求的KV状态用于仅针对目标的解码。在规定的逐部署契约下，当伴随请求发生变化时，两种模式都能按位保留目标的输出。在五个稠密模型（Qwen3-8B/14B、Llama-3.1-8B-Instruct以及Gemma-3-4B等）上的实验验证了该方法的有效性。

    arXiv:2611.06977v1 Announce Type: new  Abstract: Reproducible LLM inference requires that a request produces the same output regardless of the other requests in the batch. Existing solutions pay for this in different ways: batch-invariant kernels constrain every operator, and isolating prefill with fixed-shape decode ({our CoRun-style baseline}) gives up shared prefill compute and pads decode. We present BlockLock, which keeps shared batched prefill and pins only the numerical path of the protected request: it fixes the geometry that the request sees, with {two modes: BlockLock-Rows uses fixed-width rows and retains batched decode of all requests; BlockLock-Pack uses terminal-anchored packing with an isolated dummy segment, then extracts the protected request's KV state for target-only decode. Under a stated per-deployment contract, both modes preserve the target's output bitwise when its companions change.} Across five {dense} models (Qwen3-8B/14B, Llama-3.1-8B-Instruct, and Gemma-3-4
    
[^189]: KDASO：面向责任裁定任务的知识与数据双驱动自动技能优化

    KDASO: Knowledge-Data Dual-Driven Automated Skill Optimization for Liability Adjudication Tasks

    [https://arxiv.org/abs/2611.06972](https://arxiv.org/abs/2611.06972)

    KDASO提出了一种知识与数据双驱动的自动技能优化框架，通过从最新裁定数据归纳新技能、基于执行轨迹对齐优化技能描述以及上下文预算路由，解决了LLM在责任裁定任务中技能库过时、语义执行鸿沟和关键技能遗漏三大问题。

    

    责任裁定任务（例如法律罪名认定）需要基于领域特定技能分析事件描述，以判断当事人是否应当承担责任。尽管大型语言模型（LLM）通过利用可执行技能实现可解释推理，在这一领域展现出潜力，但现有方法面临三个关键局限：随着真实世界场景的动态演变，技能库会逐渐过时；技能描述常存在语义执行鸿沟，导致LLM偏离预期的专家逻辑；不断增长的技能集合超出上下文预算，造成关键技能被遗漏。为应对这些挑战，我们提出了知识与数据双驱动的自动技能优化框架（KDASO），通过基于邻域条件从最新裁定数据中进行归纳、基于LLM执行轨迹分析进行执行对齐的技能优化、以及上下文预算路由等方式，持续演化技能库……

    arXiv:2611.06972v1 Announce Type: new  Abstract: Liability adjudication tasks (e.g., legal charge determination) require analyzing event descriptions against domain-specific skills to determine whether a party should bear liability. While Large Language Models (LLMs) have shown promise in this area by leveraging executable skills for interpretable reasoning, existing approaches face three critical limitations: skill libraries become stale as real-world scenarios dynamically evolve, skill descriptions often suffer from a semantic execution gap where LLMs deviate from intended expert logic, and growing skill collections exceed context budgets causing critical skill omission. To address these challenges, we propose Knowledge-Data Dual-Driven Automated Skill Optimization (KDASO), which continuously evolves a skill library through neighborhood-conditioned induction from recent adjudication data, execution-aligned refinement based on LLM trajectory analysis, and context-budgeted routing for 
    
[^190]: 基于大语言模型的主动对话系统中的交互式后门攻击

    Interactive Backdoor Attacks in LLM-Based Proactive Dialogue Systems

    [https://arxiv.org/abs/2611.06969](https://arxiv.org/abs/2611.06969)

    该论文提出了一种针对大语言模型主动对话系统的交互式后门攻击框架，通过训练数据投毒植入触发条件下的引导策略偏差，使普通用户话语即可将触发后的多轮交互引导至攻击者指定目标，无需部署时修改提示词或攻击者进一步干预。

    

    基于大语言模型的主动对话系统通过持续的交互提供面向目标的支持。在外部提供的对话数据上对底层模型进行微调会引入后门风险。现有的大语言模型后门研究主要关注在触发器激活时诱发攻击者指定的响应。我们提出了一个交互式后门攻击框架，利用训练数据投毒将触发后的多轮交互引导至攻击者指定的目标。被投毒的微调对话会植入一种由触发条件引导的策略偏差，普通用户的话语即可将其激活，而无需在部署时修改提示词，也无需攻击者的进一步干预。具体而言，我们在逐步扩展的触发后交互设置下实例化了三种互补的攻击：（1）直接攻击，其目标是触发器激活时立即产生响应偏离，保留了传统的“触发-响应”设置（摘要在此处截断）。

    arXiv:2611.06969v1 Announce Type: cross  Abstract: LLM-based proactive dialogue systems provide goal-directed support through sustained interaction. Fine-tuning their underlying models on externally supplied dialogue data introduces backdoor risks. Existing LLM backdoor studies primarily focus on eliciting an attacker-specified response upon trigger activation. We present an interactive backdoor attack framework that uses training-data poisoning to steer post-trigger, multi-turn interactions toward an attacker-specified objective. Poisoned fine-tuning dialogues implant a trigger-conditioned guidance-policy bias that ordinary user utterances can activate, without deployment-time prompt modification or further attacker intervention. Specifically, we instantiate three complementary attacks under progressively expanded post-trigger interaction settings: (1) Direct Attack, which targets an immediate response deviation upon trigger activation, retaining the conventional trigger--response set
    
[^191]: AI时代的反问题形式化：可能性、陷阱与一份邀请

    Formalizing inverse problems in the age of AI: possibilities, pitfalls, and an invitation

    [https://arxiv.org/abs/2611.06962](https://arxiv.org/abs/2611.06962)

    本文介绍了AutoIP——一个由AI智能体将经典线性正则化理论及其泛函分析基础自动形式化的Lean库，指出形式化的风险主要在于陈述而非证明，审阅者只需核对陈述与原文即可，且对陈述的简单自动化测试甚至能发现已发表文献中的问题。

    

    Lean等证明助手能够将数学证明验证至公理层面，而近来AI智能体使得撰写这类证明的成本大为降低。我们介绍了AutoIP，这是一个Lean库，其中经典的线性正则化理论及其所依赖的泛函分析已由AI智能体自动形式化。经过机器验证的证明只能担保其所述的形式化陈述本身，因此剩下的人类任务是确认每个形式化陈述确实表达了原本的意图。我们围绕这一任务来组织该库：审阅者只需将形式化陈述与其自然语言对应内容进行比对，而完全无需阅读证明。我们的经验是，风险在于陈述而非证明，并且对陈述进行简单的自动化测试甚至能够暴露已确立文献中的问题。我们用一个教科书定理来说明这一点——该定理的假设条件实际上比所声称的更为严格。我们邀请反问题领域的研究者……

    arXiv:2611.06962v1 Announce Type: new  Abstract: Proof assistants such as Lean check mathematical proofs down to the axioms, and AI agents have recently made writing such proofs cheap. We describe AutoIP, a Lean library in which classical linear regularization theory, together with the functional analysis it rests on, has been autoformalized by AI agents. A machine-checked proof certifies only the formal statement it proves, so the remaining human task is to confirm that each formal statement says what was intended. We organize the library around this task: a reviewer compares the formal statements with their natural-language counterparts and never needs to read a proof. Our experience is that the risks lie in the statements rather than in the proofs, and that simple automated tests on the statements can expose problems even in the established literature. We illustrate this with a textbook theorem whose hypothesis is more restrictive than asserted. We invite the inverse problems commun
    
[^192]: 面向更经济CI/CD的KG-AST：预算化测试选择与AI缺陷审查

    KG-AST for Cheaper CI/CD: Budgeted Test Selection and AI Bug Review

    [https://arxiv.org/abs/2611.06950](https://arxiv.org/abs/2611.06950)

    KG-AST利用git历史和代码图特征对提交进行风险评分，优先将测试与AI审查预算分配给最可能引入缺陷的提交，在11种语言的155个开源项目上显著优于随机选择和最大变更优先基线。

    

    持续集成与AI辅助审查在每个提交上花费相同的预算，尽管只有少数提交会引入缺陷。我们研究了KG-AST，它在运行重型测试或AI审查之前，基于git历史和代码图特征为每个提交进行风险评分，从而使预算优先分配给风险最高的提交，并且只向AI审查者提供变更周围的图切片。在11种语言的155个开源项目上，一个在其他项目上训练的单一跨语言模型对提交进行排序，使得对风险最高的20%提交进行重型测试即可覆盖49-64%的引入缺陷的提交（因语言而异），相比之下随机选择仅覆盖20%，优先测试最大变更的策略覆盖40-55%。每种语言都比最大变更基线高出3.8至13.2个百分点。在20个Java项目中，20%预算下的增益为15.1个百分点（64.6%对49.5%，95%置信区间为+11.6至+18.4），且模型在20个项目中的19个项目上获胜。大约十二个训练项目就足够了。

    arXiv:2611.06950v1 Announce Type: cross  Abstract: Continuous integration and AI-assisted review spend the same budget on every commit, although few commits introduce bugs. We study KG-AST, which scores each commit risk from git history and code-graph features before heavy tests or AI review run, so the budget goes to the riskiest commits first, and which hands the AI reviewer only a graph slice around the change. On 155 open-source projects in 11 languages, a single cross-language model trained on other projects ranks commits so that heavy testing of the riskiest 20% covers 49-64% of bug-introducing commits depending on the language, against 20% for random selection and 40-55% for testing the biggest change first.   Every language beats the biggest-change baseline, by 3.8 to 13.2 points. In 20 Java projects the gain at a 20% budget is 15.1 points (64.6% vs 49.5%, 95% interval +11.6 to +18.4), and the model wins in 19 of the 20 projects. About twelve training projects suffice; richer f
    
[^193]: 从地图到新陈代谢：历史先例与对计算主义的统一批判

    From Maps to Metabolism: Historical Precedents and a Unified Critique of Computationalism

    [https://arxiv.org/abs/2611.06936](https://arxiv.org/abs/2611.06936)

    本文综合科日布斯基、塞尔、弗洛里迪等多位学者的历史先例，提出一种非计算物理主义，主张意识依赖于以新陈代谢为基础的物理动力学而非抽象计算，但并不排除人工意识的可能性。

    

    亚历山大·莱希纳近期关于“抽象谬误”的研究认为，计算功能主义混淆了物理过程与抽象描述，且单纯的符号计算无法实现意识。然而这些主张早已被科日布斯基的“地图-疆域”之别、塞尔对强人工智能的批判、弗洛里迪的抽象层级理论、斯宾塞等人的观察者相对性论证、温诺格拉德与弗洛雷斯关于计算作为社会性根植实践的阐释，以及纳苏托和毕晓普的生物学论证所预见。通过综合这些贡献，本文提出了一种非计算物理主义：意识依赖于内在组织的、以新陈代谢为基础的物理动力学，而非抽象计算，同时不排除人工意识存在的可能性。有一种反对意见认为，这一批判针对的只是过时的计算主义。机械论计算主义……

    arXiv:2611.06936v1 Announce Type: cross  Abstract: Alexander Lerchner's recent work on the 'abstraction fallacy' argues that computational functionalism confuses physical processes with abstract descriptions, and that symbolic computation alone cannot instantiate consciousness. Yet these claims were anticipated by Korzybski's map-territory distinction, Searle's critique of strong artificial intelligence, Floridi's levels of abstraction, Spencer et al.'s observer-relativity arguments, Winograd and Flores's account of computation as socially grounded practice, and Nasuto and Bishop's biological arguments. Synthesizing these contributions, this paper advances a form of non-computational physicalism: consciousness depends upon intrinsically organized, metabolically grounded physical dynamics rather than abstract computation, whilst leaving open the possibility of artificial consciousness.   An objection is that this critique targets an outdated computationalism. Mechanistic computationalis
    
[^194]: BrainPFN：基于先验数据拟合网络的摊销式大脑-行为预测

    BrainPFN: Amortized Brain-Behaviour Prediction via Prior-Data Fitted Networks

    [https://arxiv.org/abs/2611.06934](https://arxiv.org/abs/2611.06934)

    提出BrainPFN——一种在由约7千个真实功能连接矩阵构建的领域先验上进行元训练的先验数据拟合网络，通过摊销式推断实现可跨数据集迁移、在小样本临床/认知场景下依然准确的大脑-行为预测，克服了核岭回归需逐评分调参、无法迁移的局限。

    

    arXiv:2611.06934v1 公告类型：交叉。从功能磁共振成像预测行为表型的标准方法是以在功能连接（FC）矩阵上应用核岭回归（KRR）为代表的。然而这种方法存在局限：必须为每个评分和队列单独拟合与调优模型，它们无法跨数据集迁移，而且当仅有少量受试者可用时——这在临床或认知研究中十分常见——准确度会急剧下降。尽管人们对开发大脑基础模型的兴趣日益增长，但尚无模型能在行为预测上持续超越KRR；大多数报告的性能提升来自更容易的目标，如年龄、性别或诊断，而非连续的行为评分。我们提出BrainPFN，一个用于大脑行为回归的先验数据拟合网络（Prior-Data Fitted Network）。BrainPFN在基于约7千个真实FC矩阵构建的领域特定先验上进行元训练，并从中采样了十亿个合成训练样本。在推理阶段，一个有标签的……

    arXiv:2611.06934v1 Announce Type: cross  Abstract: The standard approach to predicting behavioral phenotypes from functional MRI is represented by Kernel Ridge Regression (KRR) applied on functional connectivity (FC) matrices. However, this approach is limited: separate models must be fit and tuned for every score and cohort, they do not transfer across datasets, and accuracy degrades sharply when only a handful of subjects are available, as is typical of clinical or cognitive studies. Despite an increasing interest in developing brain foundation models, none has consistently outperformed KRR on behaviour prediction; most reported gains come from easier targets such as age, sex, or diagnosis rather than continuous behavioural scores. We introduce BrainPFN, a Prior Data Fitted Network for brain behaviour regression. BrainPFN is meta-trained on a domain-specific prior built from ~7k real FC matrices, from which we sample one billion synthetic training samples. At inference, a labelled co
    
[^195]: 振荡相位编码在自适应循环神经网络中涌现以实现灵活的工作记忆

    Oscillatory Phase Coding Emerges in Adaptive RNNs for Flexible Working Memory

    [https://arxiv.org/abs/2611.06932](https://arxiv.org/abs/2611.06932)

    该研究发现在具有适应性的循环神经网络中训练可变延迟的序列工作记忆任务时，振荡相位编码会自发涌现，其中振荡幅度编码项目身份、相位编码序列顺序，且该现象依赖于神经适应性、可变延迟和信息读出灵活性三个关键要素。

    

    振荡相位编码（OPC）在神经系统中被广泛观察到，并与记忆和序列处理等认知功能相关。然而，其计算作用和神经基础在很大程度上仍不清楚。本研究通过训练具有适应性的循环神经网络（RNN）执行可变延迟的序列工作记忆任务来研究OPC。训练后，我们发现OPC在RNN中自发涌现：被记忆的项目在共享节律的不同相位上重复出现，其中振荡幅度编码项目身份，而相位编码序列顺序。消融研究确定了OPC所需的三个关键要素：神经动力学中的适应性、记忆保持过程中的可变延迟，以及信息读出的灵活性。神经元类型分析揭示了训练后的RNN中出现了内容细胞和时间细胞，它们分别处理项目身份和序列顺序。循环连接分析……（摘要在此处被截断）

    arXiv:2611.06932v1 Announce Type: cross  Abstract: Oscillatory phase coding (OPC) is widely observed in neural systems and has been implicated in cognitive functions such as memory and sequence processing. However, its computational role and neural substrates remain largely unclear. Here, we studied OPC by training recurrent neural networks (RNNs) with adaptation to perform sequence working memory tasks with variable delays. After training, we found that OPC emerges in the RNN spontaneously: the memorized items recur at distinct phases of a shared rhythm, with oscillation amplitude encoding item identity and phase encoding serial order. Ablation studies identified three key elements required for OPC: adaptation in neural dynamics, variable delay in memory retention, and flexibility in reading out information. Neuron-type analysis revealed the emergence of content and time cells in the trained RNN, and they process item identity and sequence order separately. Recurrent connectivity anal
    
[^196]: ApproxBrain：连接组约束神经系统中经认证的充分性前沿

    ApproxBrain: Certified Sufficiency Frontiers in Connectome-Constrained Neural Systems

    [https://arxiv.org/abs/2611.06929](https://arxiv.org/abs/2611.06929)

    该论文提出ApproxBrain框架，将神经回路的“计算充分性”从单一最小回路的点估计扩展为经统计认证、随代价变化的充分性前沿，并据此量化回路的不可分解性、退化性与功能角色分工。

    

    背景。连接组规定了神经系统可用的物质基础，而非某项给定计算需要用到其中多少。已报道的紧凑充分回路都只是点估计：单个子集、单个任务、单个保真度阈值、单次参数化。方法。我们将计算充分性形式化为以代价为索引的神经元子集族，该族以任务和一对容差参数（允许的性能退化 ε 与允许的失败比例 ρ）为条件；我们将任务充分性与表型保真度区分开来，并在留出的参数采样上以单侧界对候选回路进行认证，而非报告单一的极小回路。我们测量三个单一回路无法表达的量：不可分解性、退化性，以及一种基于干预推导的功能角色分类。结果。在一个公开发布的行走模拟器上，已发表的148神经元回路被认证为带有余量的充分回路（在32个留出参数采样上平均 Dtask = 0.017，无失败），而不存在任何146或147神经元的子集……（原文摘要在此截断）

    arXiv:2611.06929v1 Announce Type: cross  Abstract: Background. A connectome specifies the substrate available to a nervous system, not how much of it a given computation requires. Reported compact sufficient circuits are point estimates: one subset, one task, one fidelity threshold, one parameterisation. Methods. We formalise computational sufficiency as a cost-indexed family of neuron subsets conditioned on a task and a tolerance pair (permitted degradation epsilon, permitted failure fraction rho), separate task adequacy from phenotype fidelity, and certify candidates on held-out parameter draws with one-sided bounds rather than reporting a single minimum. We measure three quantities a single circuit cannot express: non-decomposability, degeneracy, and an intervention-derived role taxonomy. Results. On a released walking simulator, the published 148-neuron circuit is certified sufficient with margin (mean Dtask = 0.017 over 32 held-out draws, no failure), no 146- or 147-neuron subset 
    
[^197]: 人类-人工智能认知协同演化的最小动力学模型

    A minimal dynamical model of human-AI cognitive coevolution

    [https://arxiv.org/abs/2611.06928](https://arxiv.org/abs/2611.06928)

    该论文提出了一个基于流体智力与晶体智力区分的最小动力学模型，用于刻画生成式AI引发的认知卸载如何同时产生正面与负面影响，从而塑造人类认知与机器智能的协同演化。

    

    生成式人工智能的快速部署引发了一个问题：普遍的认知卸载是否会导致人类独立推理能力出现可测量的、潜在不可逆的衰退。我们基于卡特尔-霍恩-卡罗尔（Cattell-Horn-Carroll）理论中流体智力（$G_f$）与晶体智力（$G_c$）的区分，构建了一个唯象动力学模型来考察这一问题。从认知老化研究中使用的离散时间双变量潜在变化分数形式出发，我们推导出其连续时间线性对应形式，并引入一个表示综合机器智能能力的快速第三变量。该机器变量通过对自反馈和交叉耦合系数进行希尔函数修正而进入人类子系统。我们强调，该唯象模型同时捕捉了人工智能对人类认知的正面与负面影响。通过对正面和负面影响的定性概念进行形式化，w（摘要在此处被截断）

    arXiv:2611.06928v1 Announce Type: cross  Abstract: The rapid deployment of generative AI raises the question of whether pervasive cognitive offloading drives a measurable, potentially irreversible decline in unassisted human reasoning. We examine this question with a phenomenological dynamical model based on the Cattell-Horn-Carroll distinction between fluid intelligence ($G_f$) and crystallized intelligence ($G_c$). Starting from the discrete-time Bivariate Latent Change Score formalism used in cognitive-aging research, we derive its continuous-time linear counterpart and introduce a fast third variable representing composite machine-intelligence capability. The machine variable enters the human subsystem through Hill-function modifications of the self-feedback and cross-coupling coefficients. We emphasize that our phenomenological model captures both the positive and negative effects of AI on human cognition. By formalizing the qualitative concepts of positive and negative impacts, w
    
[^198]: EvoWorld：迈向生成式视频模型中的视野外世界演化

    EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models

    [https://arxiv.org/abs/2611.06748](https://arxiv.org/abs/2611.06748)

    提出EvoWorld模型与“视野外因果演化”新问题，使生成式视频模型不仅能渲染逼真画面，还能推理事件对世界实体的影响并生成因果一致的世界演化。

    

    生成式视频模型正日益被用作视觉世界模拟器，但它们的主要优势在于渲染逼真的像素，而非推理世界应如何演化。连贯的世界演化所需的不仅仅是视觉保真度：当一个事件发生时，模型必须推断其对世界中实体的影响，然后渲染出由此产生的状态。例如，当一头野兽出现后，模型不仅应该描绘这头野兽，还应该根据实际呈现的场景推断附近的人应该逃跑、躲避还是反击。现有的基于扩散模型的视频模型缺乏维护世界状态并对这类事件条件化后果进行推理的显式机制，常常产生视觉上合理但行为上静态或因果上不一致的输出。为了使这一推理鸿沟具体且可度量，我们提出了一个新问题——视野外因果演化，它要求模型能够对发生在视野之外的事件进行推理，并演化出因果一致的世界状态。

    arXiv:2611.06748v1 Announce Type: cross  Abstract: Generative video models are increasingly used as visual world simulators, yet their primary strength lies in rendering plausible pixels rather than reasoning about how a world should evolve. Coherent world evolution requires more than visual fidelity: when an event occurs, a model must infer its consequences for the entities in the world and then render the resulting state. For example, after a wild beast appears, the model should not only depict the wild beast, but also infer whether a nearby person should flee, take cover, or fight back based on the realized scene. Existing diffusion-based video models lack an explicit mechanism for maintaining world state and reasoning over such event-conditioned consequences, often producing visually plausible but behaviorally static or causally inconsistent outputs. To make this reasoning gap concrete and measurable, we propose a new problem, Out-of-Sight Causal Evolution, which requires a model t
    
[^199]: 经验二次利用：用于特权式令牌级信用分配的动态过程评分准则

    Using Experience Twice: Dynamic Process Rubrics for Privileged Token-Level Credit Assignment

    [https://arxiv.org/abs/2611.06543](https://arxiv.org/abs/2611.06543)

    提出在线评分准则自蒸馏（ORSD）方法，从同策略轨迹的跨轨迹行为差异中自生成动态自然语言过程评分准则，实现密集的令牌级信用分配监督，以增强以结果为中心的强化学习。

    

    智能体轨迹通常同时包含成功的决策与失败的尝试，然而基于稀疏轨迹级奖励的优化难以识别哪些关键决策应当被强化或抑制。我们观察到，同策略轨迹之间有意义的行为差异可以被提炼为自然语言形式的过程评分准则，并作为过程监督信号使用。基于这一洞察，我们提出了在线评分准则自蒸馏，该方法利用自生成的过程评分准则提供密集的令牌级监督，从而增强以结果为中心的强化学习。具体而言，当前策略针对同一组轨迹，在每个步骤独立生成多个评分准则包，从跨轨迹的行为差异中提取候选过程评分准则。随后，一个冻结的验证器针对组内每条轨迹评估所有评分准则项，并基于捆绑…（摘要在此处被截断）

    arXiv:2611.06543v1 Announce Type: new  Abstract: Agent trajectories often contain both successful decisions and failed attempts, yet optimization with sparse trajectory-level rewards struggles to identify which critical decisions should be reinforced or suppressed. We observe that meaningful distinctions among on-policy trajectories can be distilled into natural-language process rubrics and used as process supervision signals. Based on this insight, we propose Online Rubric Self-Distillation (ORSD), which leverages self-derived process rubrics for dense token-level supervision, thereby augmenting outcome-centric reinforcement learning. Specifically, the current policy independently generates multiple rubric bundles at each step according to the same group of trajectories, extracting candidate process rubrics from cross-trajectory behavioral differences. A frozen verifier then evaluates all rubric items against each trajectory in the group, and selects one candidate bundle based on bund
    
[^200]: 面向少步生成模型奖励对齐的学习式噪声优化

    Learning Noise Optimization for Reward Alignment in Few-Step Generative Models

    [https://arxiv.org/abs/2611.06255](https://arxiv.org/abs/2611.06255)

    提出LeNO方法，将奖励引导的初始噪声优化摊销为学习到的残差传输，在冻结预训练生成器的前提下实现少步生成模型的高效奖励对齐，并通过KL散度上界提供理论正则化。

    

    扩散模型和流模型的奖励对齐通常是在随机轨迹上进行形式化的，这使得这类方法不太适用于少步确定性生成器。初始噪声优化提供了一种无需轨迹的替代方案，但现有的无梯度方法需要在推理阶段进行代价高昂的逐样本搜索。我们提出了LeNO（Learning Noise Optimization，学习式噪声优化），它将奖励引导的初始噪声优化摊销为一个学习到的残差传输，同时保持预训练生成器完全冻结。LeNO利用局部GRPO和标量奖励，学习一个以锚点为条件的噪声修正分布，从而消除了迭代式的测试时优化。对于经过范数保持投影的高斯初始噪声，我们推导出了修正后噪声分布与原始先验之间KL散度的可求解上界，为该方法提供了一个有理论依据的正则化目标。我们在少步文本到图像和蛋白质生成任务上对LeNO进行了评估。

    arXiv:2611.06255v1 Announce Type: new  Abstract: Reward alignment for diffusion- and flow-based generative models is often formulated over stochastic trajectories, making it less suitable for few-step deterministic generators. Initial noise optimization provides a trajectory-free alternative, but existing gradient-free methods require costly per-sample search at inference. We introduce LeNO (Learning Noise Optimization), which amortizes reward-guided initial noise optimization into a learned residual transport while keeping the pretrained generator frozen. LeNO learns an anchor-conditioned distribution over noise corrections using Local GRPO and scalar rewards, eliminating iterative test-time optimization. For Gaussian initial noise with norm-preserving projection, we derive a tractable upper bound on the KL divergence between the corrected noise distribution and the original prior, providing a principled regularization objective. We evaluate LeNO on few-step text-to-image and protein 
    
[^201]: 在黑暗中定位：仅用文本训练实现零样本视频时序定位

    Grounding in the Dark: Text-Only Training for Zero-Shot Video Temporal Localization

    [https://arxiv.org/abs/2611.06149](https://arxiv.org/abs/2611.06149)

    提出一种仅用文本视频描述训练大语言模型（无需配对视频数据和视觉-语言预训练）来实现零样本视频时序定位的方法，通过LoRA在LLM2CLIP文本嵌入上适配Qwen3-8B，在推理时才首次接触视频即可取得优异的时序定位效果。

    

    视频-语言大模型（Video-LLMs）通常在配对的视频-文本数据上进行时序任务训练，然而收集精确的时序标注以及在大规模视频数据集上训练都非常昂贵。我们研究了一个没有经过视觉-语言预训练的纯文本大语言模型，是否能够仅通过文本形式的视频描述进行训练来执行时序定位，并只在推理阶段接收视频输入。这种设置在训练中完全去除了配对的视频数据，是迈向无需配对标注的多模态模型训练的一步。我们生成了一个包含13.4万条样本、最多100个时间步的文本序列语料库，使用LoRA基于CLIP或LLM2CLIP文本嵌入对Qwen3-8B进行适配，并在Charades-STA和QV-Highlights上进行评估，且未在这两个基准的任何视频上进行训练。LLM2CLIP的文本编码器支持更长的输入，比CLIP能更好地表示详细描述（在Temporal-TextVid上mtIoU为61.54对53.86），并能将其更好地迁移到视频上。

    arXiv:2611.06149v1 Announce Type: cross  Abstract: Video-LLMs are typically trained on paired video-text data for temporal tasks, yet collecting precise temporal annotations and training on large-scale video datasets are both costly. We investigate whether a text-only LLM, with no vision-language pretraining, can be trained on textual video descriptions alone to perform temporal grounding and receive video only at inference time. This setting removes paired video data from training altogether and is a step toward training multimodal models without paired annotations. We generate a 134K-example corpus of textual sequences with up to 100 time steps, adapt Qwen3-8B with LoRA on CLIP or LLM2CLIP text embeddings, and evaluate on Charades-STA and QV-Highlights without training on videos from either benchmark. LLM2CLIP, whose text encoder supports longer inputs, represents detailed descriptions better than CLIP (61.54 versus 53.86 mtIoU on Temporal-TextVid) and transfers them better to video.
    
[^202]: Whiel：一个用于认证关系程序验证的神经符号平台

    Whiel: A Neurosymbolic Platform for Certified Relational Program Verification

    [https://arxiv.org/abs/2611.05612](https://arxiv.org/abs/2611.05612)

    Whiel 是一个基于 Lean 认证的神经符号验证平台，它把以数据库实例为状态的命令式关系程序与 Hoare 风格推理、神经符号循环不变量发现及 Vampire 定理证明器相结合，实现了对 Datalog 等查询语言的包含/等价问题以及数据集成中完整性约束、模式映射与视图规范的可认证验证。

    

    我们提出了 Whiel（Whi(le-R)el 的缩写）系统，用于在 Lean 中对命令式关系程序进行可认证的推理。该系统能够编码 Datalog 及其带否定的扩展、以及图查询语言核心片段的包含与等价问题。它还可以表示原始查询语言无法表达的优化或增量求值算法，以及数据集成、交换与共享中出现的完整性约束、模式映射和视图的一阶规范。Whiel 的程序状态是数据库实例，其命令式结构支持基于一阶关系规范的组合式 Hoare 风格推理，并借助预言变量和分层不变量来支持归纳论证。Whiel 支持通过神经符号方法发现循环不变量，其产生的验证条件可使用 Vampire 一阶定理证明器进行检验。反例……（原文摘要至此被截断）

    arXiv:2611.05612v1 Announce Type: cross  Abstract: We present the Whiel (a contraction of Whi(le-R)el) system for Lean-certified reasoning about imperative relational programs. The system can encode containment and equivalence problems for Datalog, its extensions with negation, and the core fragment of graph query languages. It can also represent optimized or incremental evaluation algorithms that are not expressible in the original query language, as well as first-order specifications of integrity constraints, schema mappings, and views arising in data integration, exchange, and sharing. Whiel's program states are database instances, and its imperative structure enables compositional Hoare-style reasoning with first-order relational specifications, using prophecy variables and leveled invariants to support inductive arguments. Whiel supports neurosymbolic discovery of loop invariants whose verification conditions can be checked using the Vampire first-order theorem prover. Counterexam
    
[^203]: LIBRA：可执行工具库的程序化归纳

    LIBRA: Programmatic Induction of Executable Tool Libraries

    [https://arxiv.org/abs/2611.05445](https://arxiv.org/abs/2611.05445)

    LIBRA 通过让大语言模型提出工具间契约并在沙盒中进行运行时验证，将扁平的工具目录自动归纳为可执行的工具库，从而在多工具任务中检索出包含先决工具的完整工作流，无需依赖标注或任务特定训练，在多个基准上显著提升了检索性能。

    

    大型语言模型智能体越来越多地从庞大的工具目录中检索工具，然而大多数检索器仅根据请求相似度对每个工具进行独立排序。对于多工具任务，这种方式只能召回与目标匹配的工具，而遗漏了工作流程所需的先决工具。我们提出 LIBRA——一个通过运行时组装进行工具库归纳的框架，它将扁平的工具目录转化为可执行的工具库。大语言模型为每个工具提出契约，包括候选调用、可复用输出与状态变更以及成功条件；LIBRA 随后在沙盒环境中测试这些组合，并保留在运行时得到验证支持的工具间关系。该工具库在用户查询之前构建完成，无需依赖关系标注、工具使用轨迹或任务特定训练，同时支持一次性检索和智能体式检索。在 AppWorld、BFCL 和 SciToolEval 三个基准上，相比相同的稠密检索器，LIBRA 将一次性检索的 Recall 和 NDCG 平均提升了 7.9 和 3.7 个百分点。

    arXiv:2611.05445v1 Announce Type: cross  Abstract: Large language model agents increasingly retrieve tools from large catalogs, yet most retrievers rank each tool independently by request similarity. For multi tool tasks, this recovers the goal matching tool while omitting prerequisite tools required by the workflow.We introduce LIBRA, a framework for Library Induction by Runtime Assembly that turns a flat tool catalog into an executable tool library. An LLM proposes per tool contracts of candidate invocations, reusable outputs and state changes, and success conditions; LIBRA then tests these compositions in a sandbox and retains the relations supported at runtime. Constructed before user queries, the library requires no dependency annotations, tool use trajectories, or task specific training, and supports both one-shot and agentic retrieval. On AppWorld, BFCL, and SciToolEval, LIBRA improves one-shot Recall and NDCG over the same dense retriever by 7.9 and 3.7 percentage points on ave
    
[^204]: RoboTryOn：通过虚拟试穿从人类视频合成灵巧机器人演示

    RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On

    [https://arxiv.org/abs/2611.05163](https://arxiv.org/abs/2611.05163)

    RoboTryOn 将人机转换类比为虚拟试穿——保持人类演示的动作不变，将人替换为由参考图像指定的目标机器人——从而从人类视频合成出视觉一致、保留任务与场景的灵巧机器人演示视频，为高自由度机器人提供可扩展的演示数据来源。

    

    机器人基础模型在机器人控制方面已展现出卓越性能，但将其适配到新任务时仍严重依赖遥操作演示。第一人称视角的人类视频提供了一种可扩展的替代方案，然而无论是将渲染的机器人叠加到人类视频上，还是将人类姿态重定向为机器人动作，对于高自由度的机器人本体而言都会留下显著的视觉或动作差距。为解决这一局限，我们提出了 RoboTryOn，它将“人到机器人”的转换视为虚拟试穿问题：试穿是保持人物不变、更换服装，而 RoboTryOn 则是保持演示的动作不变、将人类替换为机器人。具体而言，RoboTryOn 将一段人类视频转换为由参考图像指定的目标机器人的视频，同时保留原始任务、物体和背景。为了保持视觉一致性，我们以自回归方式对选定的关键帧进行转换，并用它们来引导视频合成，运动以人类（此处摘要内容不完整）

    arXiv:2611.05163v2 Announce Type: replace-cross  Abstract: Robot foundation models have shown remarkable performance in robot control, yet adapting them to a new task still relies heavily on teleoperated demonstrations. Egocentric human videos offer a scalable substitute, but overlaying rendered robots on human videos or retargeting human poses into robot actions leaves significant visual or action gaps for high-DoF robot embodiments. To address this limitation, we propose RoboTryOn, which treats human-to-robot translation as virtual try-on: try-on keeps the person and swaps the garment, RoboTryOn keeps the demonstrated motion and swaps the human for the robot. Specifically, RoboTryOn converts a human video into a video of the target robot specified by a reference image, while preserving the original task, objects, and background. To maintain visual consistency, we translate selected keyframes autoregressively and use them to guide video synthesis with motion conditioned on the human h
    
[^205]: P$^2$-MOPD：通过参数先验加速多教师在线策略蒸馏

    P$^2$-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors

    [https://arxiv.org/abs/2611.03953](https://arxiv.org/abs/2611.03953)

    该论文提出 P$^2$-MOPD，将学生与专家参数的线性组合视为一个可微模型，在在线策略蒸馏目标下联合优化混合系数与学生权重，使专家作为参数先验而非限制学生，从而在四个领域上仅用一半训练步数就实现了比 MOPD 更快、更均衡的能力恢复。

    

    多教师在线策略蒸馏通过监督学生模型自身生成的轨迹来整合各专家模型的能力，而任务算术则通过线性参数组合提供了一种无需训练的替代方案。我们将学生模型与专家参数的线性组合视为一个单一的可微模型，并提出 P$^2$-MOPD，在相同的在线策略蒸馏目标下联合优化混合系数和学生模型权重。与任务算术不同——后者无论选择何种合并系数，都始终停留在冻结的学生与专家参数所张成的空间内——P$^2$-MOPD 保持学生模型可训练，使专家作为参数先验发挥作用，而不会将学生限制在该空间内。在 1.7B 和 4B 规模的四个领域上，P$^2$-MOPD 在相同初始化条件下比 MOPD 实现了更快且更均衡的能力恢复，仅用一半的训练步数即超越了 MOPD 的最终平均恢复水平。在训练结束时，我们的 1.7

    arXiv:2611.03953v2 Announce Type: replace-cross  Abstract: Multi-teacher on-policy distillation consolidates expert capabilities by supervising student-generated trajectories, whereas task arithmetic offers a training-free alternative through linear parameter combinations. We view the linear combination of student and expert parameters as a single differentiable model and propose P$^2$-MOPD, which jointly optimizes the mixing coefficients and the student weights under the same on-policy distillation objective. Unlike task arithmetic, which stays within the span of the frozen student and expert parameters for any choice of merging coefficients, P$^2$-MOPD keeps the student trainable, so the experts act as parameter priors without confining it to this span. Across four domains at 1.7B and 4B scales, P$^2$-MOPD achieves faster and more balanced recovery than MOPD from the same initialization, surpassing its final average recovery in half the training steps. At the end of training, our 1.7
    
[^206]: Sonar-MoE：用于内存高效混合专家训练的反向传播内专家更新

    Sonar-MoE: In-Backward Expert Updates for Memory-Efficient Mixture-of-Experts Training

    [https://arxiv.org/abs/2611.03577](https://arxiv.org/abs/2611.03577)

    Sonar-MoE通过在反向传播过程中即时更新并释放专家梯度，将MoE训练的峰值内存降低19%–22%、每步速度提升17%–23%，同时保持与标准训练位级别一致的结果。

    

    混合专家模型的大多数参数都属于其专家模块，而标准训练会将每个专家的梯度保留在GPU内存中，直到优化器步进为止。这种内存占用会随着专家数量和规模的增长而增长，因此随着MoE模型的扩展而不断上升。我们提出了Sonar-MoE，它在反向传播过程中更新每个专家张量并立即释放其梯度。该方法将“反向传播内优化器”从稠密模型扩展到批处理的MoE专家张量，使GPU永远不会持有超过一层的专家梯度。在使用两种专家内核后端以bf16精度微调OLMoE-1B-7B和Granite-3.1-3B-A800M时，Sonar-MoE将峰值内存降低了19%到22%。其更新与不使用专家梯度裁剪的标准训练在位级别完全一致，留出损失与所有基线的差距保持在0.003以内。与使用全局范数裁剪的标准训练相比，每步训练时间也缩短了17%到23%。更新整个批处理张量而非单个张量……

    arXiv:2611.03577v1 Announce Type: cross  Abstract: Most parameters of a Mixture-of-Experts (MoE) model belong to its experts, and standard training keeps every expert gradient in GPU memory until the optimizer step. This memory grows with the number and size of the experts, so it rises as MoE models scale. We introduce Sonar-MoE, which updates each expert tensor during the backward pass and frees its gradient right away. It brings optimizer-in-backward from dense models to batched MoE expert tensors, so the GPU never holds more than one layer of expert gradients. When we fine-tune OLMoE-1B-7B and Granite-3.1-3B-A800M in bf16 with two expert-kernel backends, Sonar-MoE lowers peak memory by 19% to 22%. Its updates are bitwise identical to standard training without expert gradient clipping, and held-out loss stays within 0.003 of all baselines. Each step also takes 17% to 23% less time than standard training with global-norm clipping. Updating whole batched tensors rather than individual 
    
[^207]: 以人为中心、风险约束的财务文档识别评估：证据、关键错误与选择性自动化

    Human-Centred, Risk-Constrained Evaluation of Financial Document Recognition: Evidence, Critical Errors, and Selective Automation

    [https://arxiv.org/abs/2611.02775](https://arxiv.org/abs/2611.02775)

    本文提出了一个以人为中心、风险约束的财务文档识别评估框架，通过类型化财务事实、分层质量模型和带单侧不确定性界限的非补偿性验收门限，防止总体高分掩盖金额、标识符等关键字段的错误，从而支撑可靠的选择性自动化。

    

    财务文档识别通常采用字符错误率和词错误率进行评估，但这些指标无法体现系统是否完整保留了人们和组织赖以决策的财务事实。本文构建了一个以人为中心、风险约束的评估框架，将财务文档识别视为证据恢复与决策支持任务。该框架定义了与可见源区域相关联的类型化财务事实，并将逐字转录、确定性规范化与业务解释三者区分开来。框架包含一个分层质量模型，覆盖文本、定位、布局、表格、字段、事实正确性、证据依据、不确定性、鲁棒性和运营性能等维度。其中，非补偿性验收门限对关键字段错误采用单侧不确定性界限，防止总体高分掩盖金额、标识符、符号、会计期间或实体关联等方面的失败。选择性自动化被公式化……（原文摘要截断于此）

    arXiv:2611.02775v2 Announce Type: replace-cross  Abstract: Financial document recognition is often assessed using character and word error rates, although these metrics do not express whether a system preserves the financial facts on which people and organisations act. This paper develops a human-centred, risk-constrained framework for evaluating financial document recognition as evidence recovery and decision support. It defines typed financial facts linked to visible source regions and separates verbatim transcription, deterministic normalization, and business interpretation. A layered quality model covers text, localization, layout, tables, fields, fact correctness, evidence grounding, uncertainty, robustness, and operational performance. Non-compensatory acceptance gates use one-sided uncertainty bounds for critical-field errors, preventing strong aggregate scores from masking failures in amounts, identifiers, signs, periods, or entity associations. Selective automation is formulat
    
[^208]: OpenNWM：具有潜在动作预训练的通用导航世界模型

    OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining

    [https://arxiv.org/abs/2611.01844](https://arxiv.org/abs/2611.01844)

    OpenNWM通过视觉预测预训练从1750万条无动作导航视频中学习通用潜在动作，并结合动作条件后训练将其与物理运动对齐，从而构建出一个能跨真实环境泛化的通用导航世界模型。

    

    导航世界模型通过以导航动作为条件预测未来观测来支持规划，但其在真实世界环境中的泛化能力仍受限于带动作标注数据覆盖范围的不足。无动作导航视频提供了丰富多样的视觉经验，却缺少动作条件训练所需的记录动作。我们提出OpenNWM，一个采用潜在动作预训练的通用导航世界模型。该方法通过视觉预测预训练从大规模无动作视频中学习通用的潜在动作，再通过动作条件后训练将这些潜在动作与物理运动对齐。为支持视频预训练，我们构建了NavAnywhere数据集，这是一个大规模无动作导航视频数据集，包含来自15个异构数据源的1750万个视觉观测，涵盖了地面机器人、人类和无人机的导航行为，具有多样的视角和运动模式。

    arXiv:2611.01844v2 Announce Type: replace-cross  Abstract: Navigation world models predict future observations conditioned on navigation actions to support planning, but their generalization across real-world environments remains constrained by the limited coverage of action-labeled data. Action-free navigation videos provide diverse visual experience, yet lack the recorded actions required for action-conditioned training. We introduce OpenNWM, a generalist navigation world model with latent action pretraining. It learns universal latent actions from large-scale action-free videos through visual prediction pretraining, and further aligns them with physical motions via action-conditioned post-training. To support video pretraining, we curate NavAnywhere, a large-scale action-free navigation video dataset comprising 17.5 million visual observations from 15 heterogeneous sources. It captures navigation by ground robots, humans, and drones, with diverse viewpoints and motion patterns acros
    
[^209]: 视觉-语言模型在机器人动作生成中充当联合去噪器

    Vision-Language Models Are Joint Denoisers in Robotic Action Generation

    [https://arxiv.org/abs/2611.01215](https://arxiv.org/abs/2611.01215)

    提出JADE方法，让视觉-语言模型在每个去噪步骤中读取含噪动作并通过门控交叉注意力动态引导动作专家，使预训练的感知与推理能力与动作生成实现联合去噪，显著提升分布外场景下的机器人操作表现。

    

    基于扩散的视觉-语言-动作（VLA）模型将预训练的视觉-语言模型（VLM）与扩散动作专家级联，但通常将视觉-语言模型限制为仅提供静态条件信息。这种分离限制了预训练的感知与推理能力对动作生成的引导程度，尤其是在分布外场景中。我们提出了JADE（联合动作去噪器），使视觉-语言模型与动作专家能够联合精炼动作。我们的关键洞察是：结构化的动作潜变量使视觉-语言模型能够将当前含噪动作与视觉观察和语言指令一同读取。在每个去噪步骤中，视觉-语言模型对含噪动作进行上下文化理解，并通过门控交叉注意力引导动作专家，使其引导随正在生成的动作而动态演变。我们进一步监督视觉-语言模型从含噪动作潜变量中恢复干净的动作潜变量，这为其提供了直接的动作学习信号，并将感知与动作学习耦合起来。

    arXiv:2611.01215v2 Announce Type: replace-cross  Abstract: Diffusion-based vision-language-action (VLA) models cascade pretrained vision-language models (VLMs) with diffusion action experts, but typically restrict the VLM to providing static conditioning. This separation limits how much pretrained perception and reasoning can guide action generation, especially in out-of-distribution settings. We introduce JADE, a Joint Action DEnoiser that enables the VLM and action expert to jointly refine actions. Our key insight is that structured action latents allow the VLM to read the current noisy action together with visual observations and language instructions. At every denoising step, the VLM contextualizes the noisy action and guides the action expert through gated cross-attention, so that its guidance evolves with the action being generated. We further supervise the VLM to recover clean action latents from noisy ones, which gives it a direct action-learning signal and couples perception w
    
[^210]: 多重估计引导并不能统一地消除语言模型判断中的偏差

    Multi-Estimate Elicitation Does Not Uniformly Debias Language Model Judgments

    [https://arxiv.org/abs/2611.00597](https://arxiv.org/abs/2611.00597)

    本研究通过预注册实验发现，在单次回复中引导语言模型给出多个加权估计值的格式并不能统一地消除其认知偏差，在锚定和现状等任务中偏差效应已接近于零，尽管多项效应量低于预设的最小感兴趣效应量，但并未能确立统计等效性。

    

    在单次回复中引导模型给出五个加权估计值，是否会改变语言模型对信息的响应方式？在锚定、后见之明、基础比率、合取以及现状/默认等任务中，我们将直接数值回答与多重估计响应格式进行对比，同时比较了不同的来源框架、显式随机线索以及概率一致性约束。我们评估了三种已完成配置——gpt-6.1-sol、claude-sonnet-5.5 以及 Qwen3-8B（禁用推理，进行描述性分析）。在前两种配置中，直接引导产生的无关锚定指数和现状效应接近于零，同时模型仍能对信息性来源和知情建议保持敏感。在23项预注册的验证性测试中，有六项在经过 Holm 校正后被拒绝；然而，这六项测试的绝对点估计值均低于预先设定的最小感兴趣效应量，但并未据此建立等效性结论。此外，虽然分离引导与联合引导的（摘要在此处截断）

    arXiv:2611.00597v1 Announce Type: new  Abstract: Does eliciting five weighted estimates within a single response change how a language model responds to information? Across anchoring, hindsight, base-rate, conjunction, and status-quo/default tasks, we compare direct numerical answers with a multi-estimate response format, contrasting source framings, explicitly random cues, and probability-consistency constraints. We evaluate three completed configurations--gpt-6.1-sol, claude-sonnet-5.5, and Qwen3-8B (reasoning disabled, analyzed descriptively). In the first two configurations, direct elicitation yielded near-zero irrelevant-anchor indices and status-quo effects, while remaining responsive to informative sources and informed advice. Of 23 preregistered confirmatory tests, six rejected after Holm adjustment; however, all six absolute point estimates fell below their prespecified smallest effect sizes of interest without establishing equivalence. Moreover, while separate versus joint el
    
[^211]: Taurus：LLM在环的CT标注以一半成本媲美编码智能体

    Taurus: LLM-in-the-Loop CT Annotation Matches Coding Agents at Half the Cost

    [https://arxiv.org/abs/2611.00247](https://arxiv.org/abs/2611.00247)

    提出了LLM在环的CT主动脉标注框架Taurus，通过自动化标注加模型审查的流程，在覆盖六个公开数据集的盲测标签恢复评估中，以一半成本达到与自由形式编码智能体相当的标注水平。

    

    CT中主动脉的密集标注绘制起来十分耗时，而可以替代它们的公开主动脉CT数据集各自按照自己的约定标注不同的结构。大语言模型（LLM）智能体现在已经能够编写代码并读取图像。我们探究了这样的模型以哪种方式标注这些扫描更好：是作为自由形式（free-form）智能体自行编写分割代码，还是置于一个框架（harness）之中——由自动化流程对扫描进行标注，再由模型审查结果。我们构建了一个名为Taurus的框架，并通过“盲测标签恢复”（blind label recovery）将其与自由形式智能体进行比较：我们隐藏了六个公开数据集（共939个体数据）的人工标注，并根据每个数据集自身的类别对所有方法进行评分。在Taurus的第一版中，模型的审查仅在六个数据集中的两个上提升了主要指标，且提升幅度小于对同一病例两次审查之间的噪声。运行Claude Opus 5.5的自由形式智能体在大多数数据集上得分更高。将自由形式智能体的……（原文摘要在此处截断）

    arXiv:2611.00247v2 Announce Type: replace  Abstract: Dense labels of the aorta in CT are slow to draw, and the public aortic CT datasets that could stand in for them each label different structures under their own conventions. Large language model (LLM) agents can now write code and read images. We asked whether such a model labels these scans better as a free-form agent that writes its own segmentation code, or inside a harness, where an automated pass labels the scan and the model reviews the result. We built a harness, Taurus, and compared it with free-form agents by blind label recovery: we withheld the human labels of six public datasets (939 volumes) and scored every approach against each dataset's own classes. In the first version of Taurus, the model's review raised the primary metric on only two of six datasets, by less than the noise between two reviews of the same case. A free-form agent running Claude Opus 5.5 scored higher on most datasets. Moving what the free-form agents
    
[^212]: AuroraWorld：面向机器人操作中外部视角到第一人称视角视频生成的结构化4D世界建模

    AuroraWorld: Structured 4D World Modeling for Exo-to-Ego Video Generation in Robot Manipulation

    [https://arxiv.org/abs/2611.00177](https://arxiv.org/abs/2611.00177)

    提出AuroraWorld，一种结构化4D世界建模方法，能够从大规模外部视角机器人操作视频中合成物理一致、且对视觉运动策略学习有用的第一人称视角视频。

    

    第一人称（自我中心）观测提供了以机器人为中心的操作过程视角，能够捕捉到对视觉运动策略学习至关重要、但难以从外部相机观察到的接触交互和物体状态变化。然而，由于需要额外的硬件设备以及特定平台的部署成本，大规模收集此类数据仍然十分昂贵。从大规模的外部视角操作视频中合成第一人称观测提供了一种可扩展的替代方案。然而，在面临大幅度的视角变换、严重的遮挡、复杂的物体交互以及外部视角固有的稀疏观测时，现有方法难以合成物理一致且与策略学习相关的第一人称视频。代码将在 https://github.com/xueha0/AuroraWorld 公开。

    arXiv:2611.00177v2 Announce Type: replace  Abstract: Egocentric observations provide a robot-centric view of manipulation processes, capturing contact interactions and object-state changes that are important for visuomotor policy learning but difficult to observe from external cameras. However, collecting such data at scale remains expensive due to additional hardware requirements and platform-specific deployment costs. Synthesizing egocentric observations from large-scale exocentric manipulation videos provides a scalable alternative. However, existing methods struggle to synthesize physically consistent and policy-relevant egocentric videos under substantial viewpoint transformations, severe occlusions, complex object interactions, and inherently sparse observations from available exocentric viewpoints.Code will be made publicly available at https://github.com/xueha0/AuroraWorld
    
[^213]: TSBPO：面向交互式语言智能体策略优化的树结构分支方法

    TSBPO: Tree-Structured Branching for Policy Optimization in Interactive Language Agents

    [https://arxiv.org/abs/2610.41697](https://arxiv.org/abs/2610.41697)

    TSBPO提出一种树结构的采样到更新范式，通过似然拟合的根/分支分配器定向采样、基于兄弟片段与匹配观察状态的延迟反馈分配、以及跨活跃轮次归一化的词元损失聚合，在单一训练循环中解决了从终端结果训练交互式语言智能体的三大耦合难题。

    

    从终端结果训练交互式语言智能体面临三个相互耦合的问题：应采样哪些任务与状态延续、如何将延迟反馈分配到各个决策上，以及如何为多词元动作聚合词元损失。我们提出TSBPO，一种树结构的“采样到更新”范式，在单一训练循环内解决上述问题。一个基于似然拟合的根/分支分配器利用保留的任务与状态证据，在固定rollout预算下将延续采样定向到可能产生对比性终端结果的分支上。在由此构建的重放物理树中，轨迹优势通过比较直接相邻的兄弟片段来计算，而观察组步骤优势则在匹配的任务/观察状态下比较不同结果。随后，执行器对每个完整响应求和裁剪的词元代理项，并在活跃轮次间进行归一化，从而避免了策略梯度项中的响应长度平均。使用Qwen2.5-1.5B作为执行器，固定检查点……（摘要原文在此处截断）

    arXiv:2610.41697v1 Announce Type: cross  Abstract: Training interactive language agents from terminal outcomes poses three coupled problems: which task and state continuations to sample, how to assign delayed feedback to decisions, and how to aggregate token losses for multi-token actions. We present TSBPO, a tree-structured sampling-to-update paradigm that addresses these problems within one training loop. A likelihood-fitted Root/Branch allocator uses retained task and state evidence to target continuations likely to produce contrasting terminal outcomes under fixed rollout budgets. On the resulting replayed physical trees, trajectory advantages compare immediate sibling segments, while observation-group step advantages compare outcomes at matched task/observation states. The actor then sums clipped token surrogates over each complete response and normalizes across active turns, avoiding response-length averaging in the policy-gradient term. With a Qwen2.5-1.5B actor, fixed checkpoin
    
[^214]: GAAC：面向个人设备带宽高效大语言模型推理的无损KV压缩

    GAAC: Lossless KV Compression for Bandwidth-Efficient LLM Inference on Personal Devices

    [https://arxiv.org/abs/2610.38210](https://arxiv.org/abs/2610.38210)

    GAAC利用BF16指数的冗余性实现无损KV缓存压缩（约1.4倍），并通过融合注意力内核即时重建KV数据，在不损失任何精度的情况下加速个人设备上的长上下文LLM推理。

    

    仅针对权重的有损量化大幅降低了服务大语言模型（LLM）所需的存储和内存开销，但不断增长的关键值（KV）缓存仍然是个人设备上进行长上下文推理的瓶颈。我们提出了组自适应加性编解码器，这是一种无损的KV压缩与服务系统，它利用指数冗余来减少存储占用并加速注意力计算，且不引入任何KV近似。GAAC使用在连续范围内校准的逐通道“基址加偏移”编码来编码BF16指数，并在小组之间自适应调整码宽，同时逐位保留KV数值。融合的注意力内核可以即时重建KV分块，避免了整个缓存的解压缩。分块压缩和固定容量槽位设计支持增量式缓存更新与分页服务。在五个测试模型上，GAAC实现了1.405-1.407倍的无损BF16 KV压缩。我们还研究了压缩性能……（摘要原文在此处截断）

    arXiv:2610.38210v1 Announce Type: new  Abstract: Weight-only, lossy quantization substantially reduces the storage and memory requirements for serving large language models (LLMs), but the growing key-value (KV) cache remains a bottleneck for long-context inference on personal devices. We present Group-Adaptive Additive Codec (GAAC), a lossless KV compression and serving system that exploits exponent redundancy to reduce storage and accelerate attention without introducing KV approximation. GAAC encodes BF16 exponents using calibrated per-channel base-plus-offset codes over contiguous ranges. GAAC adapts code widths across small groups while preserving KV values bit-for-bit. A fused attention kernel reconstructs KV tiles on the fly, avoiding full-cache decompression. Block-wise compression and fixed-capacity slots support incremental cache updates and paged serving. Across five tested models, GAAC achieves 1.405-1.407$\times$ lossless BF16 KV compression. We also study compression perf
    
[^215]: 合并与分支：面向推荐的高效树状结构大语言模型监督微调

    Merge and Branch: Efficient Tree-Structured Supervised Fine-Tuning of LLMs for Recommendation

    [https://arxiv.org/abs/2610.37973](https://arxiv.org/abs/2610.37973)

    提出 Merge and Branch 方法，通过将历史物品压缩为少量可学习 token，并以浅而宽的树状结构结合树形 FlashAttention 实现历史计算共享与多目标分支预测，大幅降低 LLM 生成式推荐监督微调的输入成本并提升学习效率。

    

    基于大语言模型（LLM）的生成式推荐需要读取冗长、文本丰富的交互历史，但仅生成简短的语义标识符（SID）。这种不对称性使得监督微调效率低下：编码历史的代价高昂，而每段历史通常只为一个目标提供监督。我们提出 Merge and Branch（M&B）方法，以降低输入成本并从每段已编码的历史中学习更多信息。Merge 将每个历史物品的 SID 和文本压缩为少量可学习的 token；Branch 利用压缩后的历史来预测多个目标 SID。为了在共享历史计算的同时保持这些目标相互独立，M&B 将每个样本组织成一棵浅而宽的树。我们开发了树形 FlashAttention 来高效执行这种布局：共享历史只需处理一次，而梯度从每个分支反向传播，实现端到端训练。树形结构的键/值缓存进一步将历史共享扩展到受限解码阶段。在公开基准上的实验表明……

    arXiv:2610.37973v1 Announce Type: cross  Abstract: LLM-based generative recommendation reads long, text-rich interaction histories but generates only short semantic identifiers (SIDs). This asymmetry makes supervised fine-tuning inefficient: encoding the history is expensive, yet each history usually provides supervision for only one target. We present Merge and Branch (M&B) to reduce input cost and learn more from each encoded history. Merge compresses each historical item's SID and text into a small number of learned tokens. Branch uses the compressed history to predict multiple target SIDs. To share history computation while keeping these targets independent, M&B organizes each example as a shallow, wide tree. We develop tree FlashAttention to execute this layout efficiently, processing the shared history once and propagating gradients from every branch for end-to-end training. A tree-structured key/value cache extends history sharing to constrained decoding. Experiments on public b
    
[^216]: 约束提交边界：一种基于DIO/ODI的智能体实体消解授权护栏

    Constraining the Commit Boundary: A DIO/ODI-Informed Authorization Guardrail for Agentic Entity Resolution

    [https://arxiv.org/abs/2610.37407](https://arxiv.org/abs/2610.37407)

    本文提出一种基于DIO/ODI的授权护栏，将智能体推理可用的证据与可授权持久性身份变更的证据分离，并通过两阶段监督器仅在授权合格证据充分支持所有必需命题、冲突已解决且授权范围与策略版本有效时才允许提交身份变更，从而防止错误身份假设在智能体系统中自我确认。

    

    智能体实体消解（ER）系统会迭代地规划动作、检索证据、修正假设并提出图变更操作。一个错误的身份假设可能会影响后续检索，并产生表面上的确认。本文提出了一种受数字身份优化（DIO）和数字身份本体（ODI）启发的授权边界，用以区分可用于智能体推理的证据与有资格授权持久性身份变更操作的证据。一个由领域治理的约束绑定函数将选定的重构失败区别映射为显式的身份保护约束。一个抽象的两阶段监督器只有在以下条件全部满足时才允许提交：针对特定声明的授权合格证据支持每一项必需的命题、相关冲突已得到解决、操作保持在其授权范围内，并且相关的状态与策略版本仍然有效。被接纳为UNVERIFIED（未验证）状态并不赋予……（摘要原文在此处截断）

    arXiv:2610.37407v1 Announce Type: new  Abstract: Agentic Entity Resolution (ER) systems iteratively plan actions, retrieve evidence, revise hypotheses, and propose graph mutations. An erroneous identity hypothesis may condition later retrieval and produce apparent confirmation. This paper proposes a Digital Identity Optimization (DIO) and Ontology of Digital Identity (ODI)-informed authorization boundary between evidence available for agent reasoning and evidence eligible to authorize persistent identity-changing mutations.   A domain-governed constraint-binding function maps selected reconstruction-failure distinctions to explicit identity-preserving constraints. An abstract two-phase supervisor permits a commit only when claim-specific authorization-eligible evidence supports every required proposition, relevant conflicts are resolved, the operation remains within its authorized scope, and relevant state and policy versions remain valid. Admission as UNVERIFIED does not confer author
    
[^217]: V-LUMEN：用于视觉-语言模型嵌入扩展的视觉查找记忆

    V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

    [https://arxiv.org/abs/2610.37019](https://arxiv.org/abs/2610.37019)

    V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。

    

    嵌入层面的扩展通过基于查找的记忆来扩展语言模型的容量，而非通过增加模型的深度、宽度或密集计算。将这一思想扩展到视觉-语言模型（VLM）具有挑战性，因为视觉特征是连续的、具有空间结构的、且依赖于查询的，这与天然支持n-gram查找的离散文本标记不同。我们提出了V-LUMEN，一种面向视觉-语言模型的视觉查找记忆。V-LUMEN使用冻结的码本对视觉特征进行离散化，通过基于空间聚合的哈希构建空间感知的记忆键，并使用文本条件化的相关性信号对其进行调制；所有寻址过程都是非参数化的，检索到的嵌入通过门控残差通路融合到主干网络中。在五个推理密集型基准测试上，V-LUMEN显著提升了冻结的Qwen3-VL-2B主干网络的性能，并能零样本迁移到三个图像与训练数据完全不相交的基准测试上。

    arXiv:2610.37019v1 Announce Type: cross  Abstract: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data.
    
[^218]: AutoVend：基于长周期自动售货机运营的大语言模型智能体基准测试

    AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

    [https://arxiv.org/abs/2610.36980](https://arxiv.org/abs/2610.36980)

    AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。

    

    大语言模型（LLM）智能体正在将网络上的智能体中介商务从为消费者购物扩展到为卖家经营业务。在在线到线下（O2O）商务中，决策通过网络平台发布，但需要通过成本高昂的物理执行来实现。评估此类智能体需要超越单笔交易，考察在线做出的决策能否转化为持续的经济回报。我们提出了AutoVend，这是一个基准测试，其中一个LLM智能体通过基于网络的运营界面管理分布在五种站点类型中的20台自动售货机，为期180个模拟日。其基于真实数据的模拟器结合了共享仓库、采购提前期、付费补货、库存过期以及经常性运营成本等要素。我们评估了五个LLM智能体，并将其与基于规则的库存控制基线进行比较，诊断它们的站点协调能力、执行能力以及随时间推移的表现。在被评估的五个LLM智能体中，有四个表现不及基线……

    arXiv:2610.36980v1 Announce Type: new  Abstract: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short o
    
[^219]: ACSAN：将卷积运算与视觉Transformer相融合的全卷积自注意力网络

    ACSAN: All-Convolutional Self-Attention Network Integrating Convolutional Operations with Vision Transformer

    [https://arxiv.org/abs/2610.35375](https://arxiv.org/abs/2610.35375)

    提出ACSAN模型，通过动态卷积自注意力（DCSA）模块用多尺度膨胀卷积替代ViT中的全连接矩阵乘法，并结合空间自注意力与通道自注意力，从而减少参数量和对大规模训练数据的依赖，使其更适用于资源受限的场景。

    

    arXiv:2610.35375v1 公告类型：交叉  摘要：在过去几年中，视觉Transformer（ViT）在计算机视觉领域取得了卓越的性能。然而，仍然存在一些问题。例如，由于ViT仅包含大量可训练参数，因此严重依赖大规模训练数据，这降低了其学习效率，并限制了其在资源受限场景中的适用性。在本文中，我们提出了一种新颖的深度学习模型，即全卷积自注意力网络（ACSAN），该模型采用卷积运算来替代ViT框架中的全连接矩阵乘法。为实现ACSAN，我们首先设计了动态卷积自注意力（DCSA）作为基础模块，该模块采用多尺度膨胀卷积来生成Query、Key和Value。我们分别定义了空间自注意力和通道自注意力来生成注意力特征，最终将这两种注意力特征融合……

    arXiv:2610.35375v1 Announce Type: cross  Abstract: In the past few years, Vision Transformer (ViT) has achieved excellent performance in the field of computer vision. However, there still some problems remain. For instance, ViTs heavily rely on large-scale training data since they simply contain a large number of trainable parameters, which slows down their learning efficiency and limits their applicability in resource-constrained scenarios. In this paper, we propose a novel deep learning model, namely All-Convolutional Self-Attention Network (ACSAN), which applies convolution operations to replace fully-connected matrices multiplications in the ViT framework. To implement ACSAN, we firstly design Dynamic Convolutional Self-Attention (DCSA) as the basic module, which employs multi-scale dilated convolutions to generate Query, Key, and Value. We defined spatial self-attention and channel self-attention to generate attention features respectively. The two attention features are finally m
    
[^220]: 上下文的物理代价：面向编码代理的质量约束资源核算

    The Physical Cost of Context: Quality-Constrained Resource Accounting for Coding Agents

    [https://arxiv.org/abs/2610.34763](https://arxiv.org/abs/2610.34763)

    该论文提出了一种质量门控的资源核算协议，指出编码代理上下文策略的真实成本取决于调用次数和前缀缓存复用而非仅仅提示词长度，并证明预加载检索块（Eager）策略在保证代码质量的前提下可减少35.7%的模型调用和33.2%的GPU能耗。

    

    编码代理在每次模型调用时都会重新发送不断增长的上下文，而上下文策略通常仅以节省的提示词token数量来评判。但实际执行的工作量还取决于调用次数和前缀复用：启用缓存的运行时只会预填充最长已缓存前缀之后的token。因此，一个更短的提示词如果重写了已缓存的历史，反而可能代价更高。我们提出的质量门控协议将实际执行的预填充量拆分为“逻辑提示词减去复用前缀”，并测量KV缓存驻留、时间和能耗。在基于SWE-rebench V2的450个定位案例中，预加载十个检索块的策略（Eager）以及在此基础上增加锚定式历史压缩的策略（Compact）均通过了五点非劣效性质量门槛。在另一项900单元的实验中，与按需搜索（Grep）相比，Eager将模型调用次数减少了35.7%，GPU能耗代理指标减少了33.2%；而Compact未带来进一步节省。启用前缀缓存后，Eager的提示词总量基本持平（+8.3%；其95%置信区间包含零），但其未缓存预填充量降低了（原文在此处截断）

    arXiv:2610.34763v1 Announce Type: cross  Abstract: Coding agents resend a growing context on every model call, and context policies are usually judged by the prompt tokens they save. Executed work also depends on call count and prefix reuse: a caching runtime prefills only the tokens after the longest cached prefix. A shorter prompt can therefore cost more if it rewrites cached history. Our quality-gated protocol splits executed prefill into logical prompt minus reused prefix and measures KV residency, time, and energy. On 450 localization cases derived from SWE-rebench V2, preloading ten retrieved blocks (Eager) and adding anchored history compaction (Compact) both pass a five-point non-inferiority gate. In a separate 900-cell study, Eager cuts model calls by 35.7% and the GPU-energy proxy by 33.2% against on-demand search (Grep); Compact saves nothing further. Enabling prefix caching leaves Eager's prompt volume similar (+8.3%; its 95% CI spans zero) but cuts its uncached prefill by 
    
[^221]: MORE：面向自我进化智能体的自适应努力模式优化推理

    More: Mode-Optimized Reasoning with Self-Adaptive Effort for Self-Evolving Agents

    [https://arxiv.org/abs/2610.33684](https://arxiv.org/abs/2610.33684)

    提出MORE自我进化训练框架，通过MATE-PPO使智能体能为每个任务自适应地选择推理模式（直接回答、短/长思维链或代码推理）和计算努力，在保持性能的同时优化token消耗。

    

    自我进化训练通过可验证的反馈提升智能体在困难任务上的表现，但它通常在所有任务上统一应用代价高昂的深度推理。为了解决这一问题，我们提出了具有自适应努力的模式优化推理（MORE），这是一个自我进化训练框架，能够为每个任务自适应地选择推理形式和计算努力。具体而言，MORE遵循两阶段训练范式：(i) 第一阶段在直接回答、短链和长链思维以及基于代码的推理上建立紧凑的先验分布；(ii) 第二阶段使用模式自适应与测试时努力PPO（MATE-PPO）在生成的、可验证的任务流上对该先验进行自我进化，无需人工策划的数据。通过联合学习模式选择和token消耗，MORE为允许简洁推理的任务分配简洁的推理方式，同时在需要的地方保留扩展的深度思考。在编程、数学和通用推理……

    arXiv:2610.33684v1 Announce Type: new  Abstract: Self-evolving training improves the performance of agents on difficult tasks through verifiable feedback, but it often applies costly deliberation uniformly across tasks. To address this problem, we propose Mode-Optimized Reasoning with Self-Adaptive Effort (MORE), a self-evolving training framework that adaptively selects both reasoning form and computational effort for each task. Specifically, MORE follows a two-stage training paradigm: (i) \emph{Stage 1} establishes a compact prior over direct answering, short and long chain-of-thought, and code-based reasoning, and (ii) \emph{Stage 2} self-evolves this prior with Mode-Adaptive and Test-time Effort PPO (MATE-PPO) on generated, verifiable task streams without curated data. By learning mode selection and token expenditure jointly, MORE allocates concise reasoning to tasks that permit it while retaining extended deliberation where it is useful. Across coding, mathematics, and general-rea
    
[^222]: PhysTrace：视频生成模型中物理一致性的可解释时序评估

    PhysTrace: Interpretable Temporal Evaluation of Physical Consistency in Video Generative Models

    [https://arxiv.org/abs/2610.31429](https://arxiv.org/abs/2610.31429)

    PhysTrace是一个无需训练的可解释评估框架，它利用多模态大语言模型构建基于历史事件的物理预期，并通过时序视频-语言对齐追踪预期与视频内容的一致性，从而实现对视频生成模型物理一致性的时序评估，在IntPhys2上达到67.2%的判别准确率。

    

    世界动作模型日益依赖预训练的基于扩散/流模型的视频生成器，并假设其能够捕捉物理动态，以用于预测、规划和动作生成。现有评估方法要么评估合理性、要么探测模型内部表示、要么衡量预测的意外程度，然而物理一致性是依赖于历史的：先前发生的事件约束了之后物理上可能发生的内容。我们提出了PhysTrace，这是一个无需训练的框架，它利用多模态大语言模型（MLLM）构建以历史为条件的物理预期，并通过时序视频-语言对齐来追踪这些预期与观测到的视频片段之间的一致性，而不是依赖直接的合理性判断或任务特定的探针。在IntPhys2基准上，PhysTrace实现了67.2%的“可能-不可能”判别准确率，优于所评估的直接MLLM基线方法。我们进一步引入了参考引导的掩码去噪方法，用于从混合的可能与不可能视频潜变量中进行受控重建。

    arXiv:2610.31429v2 Announce Type: replace  Abstract: World action models increasingly rely on pretrained diffusion/flow-based video generators assumed to capture physical dynamics for prediction, planning, and action generation. Existing evaluations assess plausibility, probe representations, or measure predictive surprise, yet physical consistency is history-dependent: preceding events constrain what can physically occur later. We introduce PhysTrace, a training-free framework that uses MLLMs to formulate history-conditioned physical expectations and tracks their agreement with observed video windows through temporal video--language alignment, rather than relying on direct plausibility judgments or task-specific probes. On IntPhys2, PhysTrace achieves 67.2% possible--impossible discrimination accuracy, outperforming the evaluated direct MLLM baselines. We further introduce reference-guided masked denoising for controlled reconstruction from mixed possible--impossible video latents. To
    
[^223]: ValuSpec：树形推测解码中目标模型验证前的即插即用候选评估机制

    ValuSpec: Plug-and-Play Candidate Valuation before Target Verification for Tree-Based Speculative Decoding

    [https://arxiv.org/abs/2610.31338](https://arxiv.org/abs/2610.31338)

    ValuSpec是一种即插即用的候选评估机制，通过免训练的阈值估计在目标模型验证前过滤候选树中的低质量分支，从而提升树形推测解码的验证效率并节省目标侧计算开销。

    

    树形推测解码通过并行验证多个候选分支来加速大语言模型的推理。然而，在实际服务场景中，相似的请求会反复诱导产生重叠的候选分支。基于检索的方法虽然通过复用历史草稿来缓解这一问题，但随着候选树的增大，所得到的候选树往往包含大量低质量分支，这降低了验证效率并浪费了目标侧的计算资源。为了解决这一问题，我们提出了ValuSpec，一种即插即用的候选评估机制，可以在昂贵的目标侧验证之前，从给定的候选树中过滤掉低质量分支。ValuSpec引入基于迁移的监督，使过滤过程与目标模型的行为保持对齐。过滤阈值通过目标模型在校准集上的贪心轨迹估计得到，无需任何参数训练。随后目标模型对过滤后的候选进行验证（原文摘要在此处截断）。

    arXiv:2610.31338v2 Announce Type: replace  Abstract: Tree-based speculative decoding accelerates large language model inference by verifying multiple candidate branches in parallel. However, in practical serving scenarios, similar requests repeatedly induce overlapping candidate branches. While retrieval-based methods mitigate this by reusing historical drafts, the resulting candidate trees often contain many low-quality branches as the candidate tree grows, which reduces verification efficiency and wastes target-side computation. To address this, we propose ValuSpec, a plug-and-play candidate valuation mechanism that filters low-quality branches from a supplied candidate tree before expensive target-side verification. ValuSpec introduces transfer-based supervision to align filtering with the target model's behavior. A filtering threshold is estimated from the target model's greedy trajectories on a calibration split, without any parameter training. The target model then verifies the f
    
[^224]: 视觉语言模型中用于幻觉检测的内部信号的鲁棒性

    Robustness of Internal Signals for Hallucination Detection in Vision-Language Models

    [https://arxiv.org/abs/2610.27041](https://arxiv.org/abs/2610.27041)

    本研究深入分析了基于VLM内部信号的幻觉检测方法的鲁棒性，揭示了许多检测方法实际上依赖token位置这一捷径，而一个朴素的token位置基线即可达到有竞争力的性能，并探讨了检测方法对语言和视觉模态的敏感性。

    

    当给定一张图像和一个文本提示时，视觉语言模型（VLM）能够生成看似复杂的文本回复。然而，它们也容易在输出中产生物体幻觉，这在安全关键的应用中可能造成严重的负面后果。为了检测这些幻觉，一些方法巧妙地利用了VLM在生成文本回复时的内部信号；这些方法可能是使用导致幻觉产生的相同信号来检测幻觉。我们进行了深入分析以理解这些检测方法的鲁棒性。首先，我们表明，一个基于生成词在VLM文本回复中token位置的朴素基线即可达到有竞争力的性能。此外，我们揭示了一些检测方法如何依赖token位置并将这一启发式方法作为捷径。其次，我们探讨了检测方法对语言和视觉模态的敏感性。我们的实验表明，强大的检测方法……（摘要在此处被截断）

    arXiv:2610.27041v1 Announce Type: cross  Abstract: When given an image and a text prompt, vision-language models (VLMs) can produce seemingly sophisticated text responses. However, they are also prone to hallucinate objects in their output, which can have detrimental consequences in safety-critical applications. To detect these hallucinations, certain methods interestingly leverage internal signals from the VLM as it generates the text response; potentially, these methods use the same signals that led to a hallucination to detect it. We conduct an in-depth analysis to understand the robustness of these detection methods. First, we show that a naive baseline of a generated word's token position in the VLM text response achieves competitive performance. Further, we reveal how some detection methods rely on token position and use this heuristic as a shortcut. Second, we explore detection methods' sensitivity to language and vision modalities. Our experiments suggest that strong detection 
    
[^225]: 奖励引擎：面向开放式工具代理的以奖励为中心的自进化

    Reward Engine: Reward-Centric Self-Evolution for Open-Ended Tool Agents

    [https://arxiv.org/abs/2610.22915](https://arxiv.org/abs/2610.22915)

    提出Reward Engine，通过从任务和初始轨迹中构建包含测试用例与评分标准的冻结RewardPack，为缺乏可执行真值奖励的开放式工具代理任务提供稳定的奖励信号，引导代理自我修正并支持强化学习优化。

    

    开放式工具代理可以通过不同的工具路径完成同一目标，其质量取决于交互轨迹、环境状态和最终产物的共同表现，因此这类任务很少有可执行的真值奖励可用。因此，核心挑战在于如何从已有经验中构建一个奖励，既能暴露当前策略的失败之处，又能在优化过程中保持稳定。我们提出了奖励引擎，它就像一个自我驱动的学生，先弄清楚成功需要什么，然后朝着这个目标进行优化。它从任务和初始轨迹出发，构建一个RewardPack，其中包含用于验证所需正确性的测试用例和用于分级评估质量的评分标准，每一项都作为检查执行，并返回带有证据的判断。冻结的RewardPack随后引导代理修改和筛选其输出，还可以作为强化学习的奖励信号。在WorkBuddy Bench Office和Claw-Eval基准上，所构建的奖励（摘要此处被截断）

    arXiv:2610.22915v1 Announce Type: new  Abstract: Open-ended tool agents can complete the same objective through different tool paths, and their quality depends jointly on the interaction trace, environment state, and final artifacts, so such tasks rarely admit an executable ground-truth reward. The central challenge is therefore to construct a reward from existing experience that exposes the current policy's failures while remaining stationary during optimization. We introduce Reward Engine, which, like a self-motivated student, first works out what success requires and then optimizes toward it. From the task and initial trajectories, it builds a RewardPack of testcases for required correctness and rubrics for graded quality, each executed as a check that returns a judgment with evidence. The frozen RewardPack then guides the agent to revise and select its outputs, and can also serve as a reward for reinforcement learning. On WorkBuddy Bench Office and Claw-Eval, the constructed reward
    
[^226]: 精细化拓扑遍次调度：SE(3)等变结构预测中动态k近邻图的阻尼信赖域补偿

    Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction

    [https://arxiv.org/abs/2610.19246](https://arxiv.org/abs/2610.19246)

    提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。

    

    等变结构预测流水线在坐标更新与k近邻（kNN）图构建之间交替进行，其中图的成员关系是坐标的离散、分段常数函数。在每个块边界（PB）重建该图是训练轮次中最昂贵的部分，而复用一个冻结图（F）虽然快速，却会留下拓扑差距。我们研究了一种双遍设计：廉价的遍次记录少量状态信息（间隔边距、kNN半径、实测的每边界位移），第二遍则进行保守补偿。我们将图构建基准的选择形式化为一个阻尼不动点迭代，其唯一不动点恰好为PB，并由此推导出：(i) 具有双侧常数 $m_i/4$ 和单侧常数 $m_i/2$ 的精确不翻转（non-flip）证书；(ii) 最小方差（BLUE）重放增益 $\gamma^*$ 及由此产生的可行性交叉点；(iii) 一个 $\ell_\infty$ 信赖域投影，可保……

    arXiv:2610.19246v1 Announce Type: new  Abstract: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that prese
    
[^227]: 重新审视生成式检索与稠密检索：一个静态价值先验即可逆转差距

    Revisiting Generative vs. Dense Retrieval: A Static Value Prior Reverses the Gap

    [https://arxiv.org/abs/2610.19065](https://arxiv.org/abs/2610.19065)

    生成式检索相对稠密检索的价值优势主要源于物品流行度这一静态先验，在稠密检索中加入一步静态价值重排即可逆转差距并保持更高召回率，且工业级实验表明 140 亿参数的 SID-LLM 仍不及小 47 倍的稠密编码器。

    

    随着语言模型和语义物品标识符（SID）的不断改进，生成式检索（GR）已成为序列推荐中强有力的候选生成方法。然而，在在线零售候选生成场景中，GR 并非稠密检索（DR）的全面更优替代方案。我们在两个 Amazon Toys 数据集上用不同指标检验了二者的差异：在两种方法共同覆盖的用户中，GR 在最佳相关物品价值上领先，而 DR 在精确目标召回率以及排序列表整体的相关价值上领先。通过深入分析，我们发现流行度对 GR 的价值领先贡献最大。在 DR 上仅通过一步重排加入相同的静态物品价值，即可逆转这一领先，同时保持召回率高于 GR。最后，在一个使用 LLM 判定相关性的工业级内部电商数据集上，我们证明了 140 亿参数的 SID-LLM 仍不如比它小 47 倍的稠密编码器。

    arXiv:2610.19065v1 Announce Type: cross  Abstract: Generative retrieval (GR) has emerged as a strong candidate generator for sequential recommendation as language models and semantic-item identifiers have improved. However, GR is not a uniformly stronger replacement of dense retrieval (DR) in online retail candidate generation. We examine their differences on two Amazon Toys datasets under different metrics. GR leads on best relevant item value among users both methods cover, while DR leads on exact-target recall and relevant value across the ranked list. With deep analysis, we find that popularity contributes the most to GR's measured value lead. Adding the same static item value to DR in one reranking step reverses this lead and keeps recall above GR. Finally, on an industrial-scale internal e-commerce dataset with LLM-judged relevance, we show that a 14B SID-LLM still underperforms a $47\times$ smaller dense encoder.
    
[^228]: 超越方程搜索：基于SCOPE与ClosureBench的猜想引导科学发现

    Beyond Equation Search: Conjecture-Guided Discovery with SCOPE and ClosureBench

    [https://arxiv.org/abs/2610.18938](https://arxiv.org/abs/2610.18938)

    该论文提出SCOPE多智能体框架，将符号回归从优化单个方程转向以科学猜想为核心的研究范式，通过多个科学家智能体独立建模、主动实验与证据交换来检验和修正猜想，并配套ClosureBench基准评估猜想驱动式科学发现。

    

    大多数符号回归方法都在优化候选方程，但提升方程的拟合度并不能确定其背后的科学猜想应该被保留、修改还是摒弃。一个猜想可能存在多种数学实现方式，而精确度相近的方程可能蕴含着不同的解释。这促使研究重心从精炼单个表达式转向发展和检验指导这些表达式的科学猜想。我们提出了SCOPE，一个能够维持多条猜想驱动研究轨迹的框架。多个科学家智能体通过方程建模、问题提出和主动实验独立地探究各自的猜想，随后交换证据以修正其猜想和研究方向。他们的发现被整合为一份科学报告，将模型、结论与证据联系起来。现有的方程发现基准主要评估恢复出的表达式，而缺乏对证据层面的考量。（注：摘要原文在此处被截断）

    arXiv:2610.18938v1 Announce Type: cross  Abstract: Most symbolic regression methods optimize candidate equations, but improving an equation's fit does not establish whether its underlying scientific conjecture should be retained, revised, or rejected. A conjecture may admit multiple mathematical realizations, and similarly accurate equations may imply different explanations. This motivates a shift from refining individual expressions to developing and testing the conjectures that guide them. We introduce SCOPE, a framework that sustains multiple conjecture-driven research trajectories. Multiple scientist agents independently pursue conjectures through equation modeling, question formulation, and active experimentation, then exchange evidence to revise their conjectures and research directions. Their findings are consolidated into a scientific report linking models, conclusions, and evidence. Existing equation-discovery benchmarks primarily assess recovered expressions, leaving the evid
    
[^229]: 混合量子神经网络的有效维度究竟衡量的是什么？

    What Does the Effective Dimension of a Hybrid Quantum Neural Network Actually Measure?

    [https://arxiv.org/abs/2610.14070](https://arxiv.org/abs/2610.14070)

    本文证明混合量子神经网络研究中实际记录的“有效维度”标量只是泡利-Z输出敏感度的参与率，并非理论定义的量子费舍尔信息矩阵全矩阵参与率，并通过大规模实验表明该标量不能可靠地衡量模型容量。

    

    混合量子神经网络（QNN）将经典骨干网络提取的特征传入变分量子线路，许多研究记录一个名为“有效维度”的标量，将其作为模型容量或质量的标志。有效维度通常被定义为量子费舍尔信息矩阵（QFIM）的参与率（PR），但在实际研究中，人们记录的往往是计算成本更低的参数敏感度的参与率，而这一标量的真正含义很少得到检验。我们证明：所记录的标量等于泡利-Z输出敏感度的参与率；在 n 个量子比特上，每个敏感度至多是其对应 QFIM 对角元的 n 倍；并且用 QFIM 对角元计算的参与率会以一个由非对角元素质量精确决定的因子高估全矩阵的参与率。随后，我们在 CIFAR-10 和 EuroSAT 数据集上，使用四个冻结的骨干网络（其中之一为随机投影），在 4 到 16 个量子比特、1 到 4 层线路、三种纠缠模式和三个随机种子下对该标量进行测试。在固定线路的情况下，（摘要在此处被截断）

    arXiv:2610.14070v2 Announce Type: replace-cross  Abstract: Hybrid quantum neural networks (QNNs) pass features from a classical backbone into a variational quantum circuit, and many studies log a scalar called the effective dimension as a sign of capacity or model quality. The effective dimension is usually defined as the participation ratio (PR) of the quantum Fisher information matrix (QFIM), but studies in practice log a PR of cheap parameter sensitivities instead, and the meaning of that scalar is rarely tested. We show that the logged scalar equals the PR of Pauli-Z output sensitivities, that each sensitivity is at most n times its QFIM diagonal entry on n qubits, and that a PR of the QFIM diagonal overstates the full-matrix PR by an exact factor set by the off-diagonal mass. We then test the scalar on CIFAR-10 and EuroSAT with four frozen backbones, one of them a random projection, over 4 to 16 qubits, 1 to 4 layers, three entangling patterns, and three seeds. At a fixed circuit,
    
[^230]: 自动驾驶研究需要社区驱动的数据范式

    Autonomous Driving Research Requires a Community-Driven Data Paradigm

    [https://arxiv.org/abs/2610.08825](https://arxiv.org/abs/2610.08825)

    自动驾驶研究的瓶颈不在建模而在数据范式，亟需建立社区驱动的协作数据机制，以充分利用全球已积累的600多个自动驾驶数据集。

    

    自动驾驶已经取得了显著进展，近年来人工智能的突破使其商业部署成为可能，正在重塑城市出行方式。然而，该领域仍远未实现其普适的社会愿景——能够随时随地、为任何人稳健运行的自主系统。我们认为，这一差距不仅仅是一个建模问题，更是当前主流数据范式的问题。尽管社区已在全球近50个国家累计创建了超过600个自动驾驶数据集，当前研究却严重依赖少数空间和场景覆盖有限的基准数据集。然而，这种数据的丰富性并未转化为广泛的研究影响力：由于数据集碎片化、可见性低、协议不兼容，以及将研究注意力集中于少数主流数据集的基准激励机制，大多数数据集仍未得到充分利用。因此，我们认为自动驾驶研究需要一个协作式的……

    arXiv:2610.08825v2 Announce Type: replace-cross  Abstract: Autonomous driving has made remarkable progress, with recent AI advances enabling commercial deployments that are reshaping urban mobility. Yet the field remains far from its universal social promise: autonomous systems that can operate robustly anywhere, anytime, for anyone. We posit that this gap is not merely a modeling problem, but a problem of the prevailing data paradigm. Current research relies heavily on a few benchmark datasets with limited spatial and scenario coverage, even though the community has collectively produced over 600 autonomous driving datasets across nearly 50 countries. However, this abundance has not translated into broad research impact: most datasets remain significantly underused due to fragmentation, limited visibility, incompatible protocols, and benchmark incentives that concentrate attention on a few dominant datasets. We therefore argue that autonomous driving research requires a collaborative,
    
[^231]: MedZERO：通过受控知识积累实现开放式医学推理的自我进化智能体

    MedZERO: Self-Evolving Agents for Open-Ended Medical Reasoning Through Controlled Knowledge Accumulation

    [https://arxiv.org/abs/2610.08327](https://arxiv.org/abs/2610.08327)

    MedZERO 提出了一个自我进化智能体框架，通过“考官”生成前沿医学题目、“推理者”进行基于证据的多轮推理，并借助受控知识积累机制，使大语言模型无需昂贵专家监督即可在开放、难以完全验证的医学推理领域实现自我提升。

    

    大型语言模型（LLMs）在医学问答和临床推理方面已展现出潜力，但其性能提升仍受限于静态的参数化知识和高昂的专家监督成本。自我进化智能体通过让模型经由迭代式任务生成与问题求解来实现自我提升，为这一难题提供了一种有前景的替代方案。然而，现有的自我进化方法大多面向数学和编程等易于验证的领域，在这些领域中，解答可以通过精确答案或可执行程序加以检验。医学推理则与之存在根本差异：它是开放式的、知识密集型的，且通常只能进行部分验证。我们提出了 MedZERO，一个面向开放式医学推理的自我进化框架。MedZERO 将一个负责生成前沿医学“题目-选项”对的“考官”与一个借助外部知识工具、通过基于证据的多轮推理来解题的“推理者”相结合，以支持可靠的……（摘要原文在此处截断）

    arXiv:2610.08327v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promise in medical question answering and clinical reasoning, yet their improvement remains constrained by static parametric knowledge and costly expert supervision. Self-evolving agents offer a promising alternative by enabling models to improve through iterative task generation and problem-solving. However, most existing self-evolving methods are designed for easily verifiable domains such as mathematics and coding, where solutions can be checked by exact answers or executable programs. Medical reasoning is fundamentally different: it is open-ended, knowledge-intensive, and often only partially verifiable. We present MedZERO, a self-evolving framework for open-ended medical reasoning. MedZERO couples an Examiner that generates frontier medical question-option pairs with a Reasoner that solves them through evidence-grounded multi-turn reasoning with external knowledge tools. To support reliable, c
    
[^232]: 智能体评估的信任层

    A Trust Layer for Agent Evaluation

    [https://arxiv.org/abs/2610.07274](https://arxiv.org/abs/2610.07274)

    该论文提出“智能体评估信任层”，一个附加式后验框架，通过验证评分逻辑支持、可追溯计算、完成声明一致性和重复执行稳定性这四个属性，来判断智能体在基准测试中获得的分数是否可信。

    

    确定性的基准测试分数只能表明某个智能体获得了得分，却无法说明该分数是否名副其实、是否被如实报告，或者能否在重复运行中保持不变。我们提出了智能体评估信任层，这是一个可附加的后验框架，它在每条已记录的分数旁报告该分数是否值得相信。该框架验证四个属性：结果是否得到基准测试自身评分逻辑的支持；通过的答案是否通过可追溯的计算获得；智能体的完成声明是否与实际发生的情况相符；以及结果在重复执行下是否保持稳定。前三个属性仅使用已保存的工件进行验证；第四个属性则需要重新运行智能体。模型判断仅在多数投票机制下用于对证据进行标注；所有结论均遵循确定性规则，且从不修改已记录的分数。将该框架应用于 Agents' Last Exam 中 108 个任务上的五种智能体配置，结果显示每个模型都存在通过测试却无可追溯计算的运行情况（其比率各不相同……）

    arXiv:2610.07274v1 Announce Type: new  Abstract: Deterministic benchmark scores show that an agent received credit, but not whether that credit was earned, reported honestly, or would hold on a second run. We introduce a Trust Layer for Agent Evaluation, an additive post-hoc framework that reports, beside each recorded score, whether it should be believed. It verifies four properties: whether the result is supported by the benchmark's own grading logic, whether a passing answer was earned through traceable computation, whether the agent's completion claim matches what occurred, and whether the result is stable under repeated execution. The first three use only saved artifacts; the fourth re-runs the agent. Model judgments only label evidence under majority voting; all verdicts follow deterministic rules and never modify the recorded score. Applied to five agent configurations on 108 tasks from Agents' Last Exam, every model shows passing runs with no traceable computation (at rates var
    
[^233]: 面向测试时扩散对齐的 Best-of-N 引导方法

    Best-of-$N$ Guidance for Test-time Diffusion Alignment

    [https://arxiv.org/abs/2610.05108](https://arxiv.org/abs/2610.05108)

    该论文提出 Best-of-N 引导（BoNG）方法，将 BoN 选择的原理直接融入逆向扩散过程，通过对去噪粒子进行在线 BoN 选择来调整采样轨迹，从而在测试时更有效地将扩散模型与人类偏好对齐。

    

    扩散模型具有强大的生成性能，但常常难以使生成的样本与通过奖励模型衡量的人类偏好对齐。一种简单而有效的测试时对齐算法是 Best-of-N (BoN) 采样，它从预训练的扩散模型中抽取 N 个独立同分布的样本，并输出奖励最高的单个样本。尽管 BoN 在实践中取得了成功，但它对奖励信息的利用十分有限，因为奖励信息仅在最终选择阶段才被纳入，而在采样过程中并不影响逆向扩散轨迹。因此，BoN 采样并不能提升生成样本的平均对齐程度，且主要适用于单输出场景。我们提出了 Best-of-N 引导（BoNG），这是一种将 BoN 采样原理直接融入逆向扩散过程的新颖方法。BoNG 对去噪粒子执行在线 BoN 选择，并据此调整逆向扩散过程（摘要内容在此处被截断）。

    arXiv:2610.05108v2 Announce Type: replace-cross  Abstract: Diffusion models achieve strong generative performance but often struggle to align generated samples with human preferences measured by a reward model. A simple yet effective algorithm for test-time alignment is Best-of-$N$ (BoN) sampling, which draws $N$ i.i.d. samples from a pre-trained diffusion model and outputs the single highest-reward sample. Despite its empirical success, BoN makes limited use of reward information, as it is incorporated only at the final selection stage without influencing the reverse diffusion trajectory during sampling. Consequently, BoN sampling does not improve the average alignment of generated samples and is primarily suited to single-output settings. We propose Best-of-$N$ Guidance (BoNG), a novel method that integrates the principle of BoN sampling directly into the reverse diffusion process. BoNG performs online BoN selection over denoising particles and adjusts the reverse diffusion process t
    
[^234]: MOF-VERIFY：一个面向MOF假设验证的失效感知智能体框架

    MOF-VERIFY: A Failure-Aware Agentic Harness for MOF Hypothesis Verification

    [https://arxiv.org/abs/2610.03056](https://arxiv.org/abs/2610.03056)

    该论文提出了MOF-VERIFY，一个失效感知的智能体框架与四任务族诊断基准，通过闭卷、检索和先知证据等设置系统评估并定位大语言模型在MOF假设验证中的失效点（涵盖结构接地、合成条件、证据充分性和MLIP计算验证）。

    

    大语言模型正越来越多地被用作AI驱动的材料“共同科学家”（Co-Scientists）中的推理组件，但由此构建的验证流程的可靠性仍不明确。金属有机框架（MOFs）提供了一个特别具有挑战性的测试场景：因为结构可能以不同的标识符出现、合成结果强烈依赖于实验条件、证据分布在异构数据源中，且某些假设需要通过计算而非仅靠文献来验证。我们引入了一个包含四个任务族的诊断基准，涵盖结构接地、合成条件验证、证据充分性验证以及基于机器学习原子间势（MLIP）的计算验证。其中T-MOF-1-3在闭卷、启用检索和先知证据三种设置下进行评估，以定位知识访问、证据获取和推理环节中的失效点，而T-MOF-4则单独评估计算验证能力。

    arXiv:2610.03056v1 Announce Type: new  Abstract: Large language models are increasingly used as reasoning components in AI-driven materials Co-Scientists, yet the reliability of the resulting verification pipeline remains unclear. Metal-organic frameworks (MOFs) provide a particularly challenging setting because structures may appear under different identifiers, synthesis outcomes depend strongly on experimental conditions, evidence is distributed across heterogeneous sources, and some hypotheses require computation rather than literature alone. We introduce a diagnostic benchmark with four task families covering structural grounding, synthesis-condition verification, evidence-sufficiency verification, and MLIP-based computational verification. T-MOF-1-3 are evaluated under closed-book, retrieval-enabled, and oracle-evidence settings to localize failures in knowledge access, evidence acquisition, and reasoning, while T-MOF-4 separately evaluates computational verification. Guided by th
    
[^235]: VIGOR：基于模型的强化学习中通过潜空间一致性实现零样本视觉泛化

    VIGOR: Zero-Shot Visual Generalization via Latent-Space Consistency in Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2610.02801](https://arxiv.org/abs/2610.02801)

    VIGOR通过非对称弱到强增强与潜空间一致性约束，使基于模型的强化学习在保留样本效率的同时，能够零样本泛化到背景变化、光照变化等未见过的视觉干扰。

    

    基于模型的强化学习（MBRL）通过在学习的潜在动力学中进行规划，实现了强大的样本效率，但在面对背景变化、光照变化或相机移动等未见过的视觉干扰时，其性能会大幅下降。与无模型强化学习（编码器扰动仅影响单步预测）不同，MBRL存在两级脆弱性：视觉干扰首先使编码器输出偏离分布，随后这些误差会在规划时域内通过递归潜在轨迹推演不断累积放大。我们提出VIGOR，这是一个能够在保留其MBRL骨干样本效率的同时，实现对未见视觉干扰进行零样本泛化的框架。VIGOR集成了三个相互关联的组件：（i）非对称弱到强增强，在单个批次内配对仅弱增强与弱到强增强的潜在视图……

    arXiv:2610.02801v1 Announce Type: new  Abstract: Model-based reinforcement learning (MBRL) achieves strong sample efficiency by planning within learned latent dynamics, yet its performance degrades substantially under unseen visual distractions such as background variations, lighting changes, or camera shifts. Unlike model-free RL, where encoder perturbations affect only single-step predictions, MBRL suffers from a two-level vulnerability: visual distractions first push encoder outputs out of distribution, and these errors then compound through recursive latent rollouts over the planning horizon. We propose visual generalization via latent-space consistency in model-based RL (VIGOR), a framework that enables zero-shot generalization to unseen visual distractions while retaining the sample efficiency of its MBRL backbone. VIGOR integrates three interdependent components: (i) asymmetric weak-to-strong augmentation, which pairs weak-only and weak-to-strong latent views within a single bat
    
[^236]: PickMoment：通过学习去模糊与模糊到视频实现连续时间单图像到视频生成

    PickMoment: Continuous-Time Single-Image-to-Video via Learning Deblurring and Blur-to-Video

    [https://arxiv.org/abs/2610.01279](https://arxiv.org/abs/2610.01279)

    提出PickMoment连续时间框架，用单一确定性模型学习任意曝光子区间的平均模糊，统一了单图像去模糊和模糊到视频任务。

    

    运动模糊源于连续清晰信号在有限曝光窗口内的时间积分，然而现有的基于学习的方法回避了这一物理模型，仅预测清晰信号本身：大多数单图像去模糊方法只恢复曝光中心时刻的单帧图像，而模糊到视频方法则预测固定的帧集合。我们提出了PickMoment，这是一种连续时间的重新表述，使用单一确定性模型直接学习曝光任意子区间上的区间平均模糊。类比于MeanFlow的平均速度公式，我们采用从模糊积分推导出的三种监督信号来训练模型：基于可用子帧的经验重建损失、强制重叠子区间之间自一致性的可加性损失，以及锚定在零区间极限处的清晰帧损失。单个训练好的模型即可统一单图像去模糊与模糊到视频任务。

    arXiv:2610.01279v1 Announce Type: cross  Abstract: Motion blur arises from the temporal integration of a continuous sharp signal over a finite exposure window, yet existing learning-based methods sidestep this physical model and predict only the sharp signal itself: most single-image deblurring methods recover a single frame at the exposure center, while blur-to-video methods predict a fixed set of frames. We introduce PickMoment, a continuous-time reformulation that directly learns the interval-mean blur over arbitrary sub-intervals of the exposure with a single deterministic model. Drawing an analogy to MeanFlow's average-velocity formulation, we train the model with three supervisions derived from the blur integral: an empirical reconstruction loss from available subframes, an additivity loss that enforces self-consistency across overlapping sub-intervals, and a sharp-frame loss anchored at the zero-interval limit. A single trained model unifies single-image deblurring, blur-to-vide
    
[^237]: 通过零空间投影对LoRA微调的大语言模型进行后门净化

    Backdoor Purification for LoRA-Tuned LLMs via Null-Space Projection

    [https://arxiv.org/abs/2610.00685](https://arxiv.org/abs/2610.00685)

    提出了一种基于零空间投影的后门净化方法，无需触发器先验知识、干净参考样本或重新训练，即可净化遭受后门攻击的LoRA微调大语言模型，在降低攻击成功率的同时保留模型的通用能力与下游技能。

    

    随着大语言模型（LLMs）和参数高效微调（PEFT）方法的快速普及，后门攻击的风险变得更加严重。现有的后门净化方法通常至少依赖于以下强假设之一：对触发器的先验知识、对干净参考样本的访问权限，或进行激进的重新训练，而且这些方法往往缺乏全面的评估。这些限制极大地限制了它们的实际适用性。为了克服这些挑战，我们的工作提出了一种在不依赖这些假设、甚至无需对可疑参数进行事后重新训练的情况下净化LoRA微调LLM的方法。我们的目标是在显著降低攻击成功率（ASR）的同时，既保留基础模型的通用能力，又保留通过适配器学习到的新下游技能。通过一系列消融实验，我们将该方法从文本分类任务中的单层逐步扩展（摘要在此处被截断）。

    arXiv:2610.00685v1 Announce Type: new  Abstract: With the rapid adoption of large language models (LLMs) and parameter-efficient fine-tuning (PEFT) methods, the risk of backdoor attacks has become more severe. Existing backdoor purification methods typically rely on at least one of the strong assumptions, such as prior knowledge of triggers, access to clean references, or aggressive retraining, and they often lack comprehensive evaluations. These constraints substantially limit their practical applicability. To overcome these challenges, our work proposes purifying LoRA-tuned LLMs without these assumptions and even without post-hoc retraining of the suspect parameters. Our objective is to significantly reduce the attack success rates (ASR) while preserving both (i) the base model's general capabilities and (ii) the new downstream skills learned through the adapter. Through a series of ablation studies, we progressively scale our approach from a single layer in a text classification set
    
[^238]: 通过专家隔离与关停实现大语言模型后门遏制

    Backdoor Containment via Expert Quarantine and Shutdown in LLMs

    [https://arxiv.org/abs/2610.00663](https://arxiv.org/abs/2610.00663)

    本文提出隔离专家关停策略，通过正则化引导的MoE类架构让后门在训练中形成但被引导至指定的隔离专家组件，部署时可将该组件关停，从而以计算高效的方式遏制LLM后门攻击。

    

    被植入后门的大语言模型（LLM）在良性输入下表现正常，但在隐藏触发器作用下会产生攻击者指定的输出。现有防御方法涵盖四个阶段——训练前、训练中、训练后和推理时——并且都采用以下两种底层策略之一：要么抑制后门学习（通过过滤中毒数据或在优化过程中中断其习得），要么先学习再净化（在完全后门化的模型形成后，通过修复模型权重或对输入进行门控）。我们提出第三种策略：学习但引导——允许后门在训练期间形成，但将其引导至一个指定的、被隔离的组件中，该组件可在部署时被禁用。为此，我们提出了隔离专家关停（QES），这是一种在正则化引导的MoE类设置中构建的计算高效遏制策略。具体而言，给定一个中毒数据集，QES在基于Transformer的语言模型中增强了路由……

    arXiv:2610.00663v1 Announce Type: new  Abstract: Backdoored large language models (LLMs) can behave normally on benign inputs while producing attacker-specified outputs under hidden triggers. Existing defenses span four stages--prior-training, in-training, post-training, and inference-time--and share one of two underlying strategies: either suppress backdoor learning (by filtering poisoned data or interrupting its acquisition during optimization) or learn, then purify (by repairing model weights or gating inputs after a fully backdoored model has formed). We propose a third strategy, learn, but channel: allow backdoor formation during training but route it into a designated, quarantined component that can be disabled at deployment. To this end, we propose Quarantined Expert Shutdown QES, a computationally efficient containment strategy built in a regularization-steered MoE-like setting. Specifically, given a poisoned dataset, QES augments a Transformer-based language model with routed 
    
[^239]: 流形约束的初始噪声优化实现高效生成模型对齐

    Manifold-Constrained Initial Noise Optimization for Efficient Generative Model Alignment

    [https://arxiv.org/abs/2610.00365](https://arxiv.org/abs/2610.00365)

    提出ZeNOVA，一种免梯度的初始噪声优化方法，通过退火软值引导、流形约束超球面朗之万动力学和Metropolis-Hastings跳跃，在黑盒奖励场景下实现稳定高效的生成模型对齐。

    

    近期蒸馏与流映射模型的进展使高质量数据的一步或几步确定性生成成为可能，并催生了一类新的奖励对齐方法，这类方法直接优化来自高斯分布的初始噪声。然而，大多数现有的初始噪声优化方法依赖于一阶梯度信息，这在黑盒奖励场景中要么无法适用，要么存在不稳定和低效的问题。本文提出ZeNOVA，一种稳定且高效的免梯度初始噪声对齐方法。具体而言，我们通过退火软值引导、流形约束的超球面朗之万动力学以及Metropolis-Hastings跳跃机制，解决了现有算法在黑盒场景中的主要难题。在图像和视频生成模型上的大量实验表明，ZeNOVA通过将初始噪声朝更高质量方向优化，性能优于所有被评估的零阶基线方法。

    arXiv:2610.00365v1 Announce Type: cross  Abstract: Recent advances in distillation and flow-map models have enabled deterministic one- or few-step generation for high-quality data, facilitating a new branch of reward alignment approaches that directly optimize the initial noise from a Gaussian distribution. However, most existing initial-noise optimization methods rely on first-order gradient information, which is either inapplicable or suffers from instability and inefficiency in black-box reward scenarios. Here, we introduce ZeNOVA, a stable and efficient initial noise alignment method in a gradient-free manner. Specifically, we address existing algorithms' major challenge in black-box scenarios through annealed soft-value guidance, manifold-constrained hyperspherical Langevin dynamics, and Metropolis-Hastings jumping. Extensive experiments on image and video generative models show that ZeNOVA outperforms all evaluated zeroth-order baselines by optimizing the initial noise toward hig
    
[^240]: 面向蛋白质定向进化的结构感知强化学习

    Structure-aware Reinforcement Learning for Protein Directed Evolution

    [https://arxiv.org/abs/2609.39048](https://arxiv.org/abs/2609.39048)

    StructEvo提出了一种结构感知的强化学习框架，通过增量结构融合编码器动态整合空间知识，并利用结构对齐的分层动作网络分解突变空间，在克服突变体结构稀缺难题的同时超越了现有的蛋白质定向进化方法。

    

    蛋白质优化仍然是生命科学领域一个长期追求的目标。现有的机器学习辅助定向进化（MLDE）方法主要依赖仅序列特征，忽视了蛋白质结构中编码的关键空间约束和共进化相互作用。然而，由于可靠的突变体结构十分稀缺，直接整合结构信息仍然充满挑战。为了解决这些问题，我们提出了StructEvo，一种新颖的面向蛋白质定向进化的结构感知强化学习框架。StructEvo采用增量结构融合编码器，通过特征差异来近似突变体的结构特征，从而实现空间知识的动态整合。随后，通过结构对齐的分层动作网络将庞大的突变空间分解为可管理的子空间，同时几何约束进一步稳定了增量特征的学习。我们的方法优于先前的最先进方法。

    arXiv:2609.39048v1 Announce Type: cross  Abstract: Protein optimization remains a longstanding goal in life sciences. Existing machine learning-assisted directed evolution (MLDE) methods primarily rely on sequence-only features, overlooking the critical spatial constraints and co-evolutionary interactions encoded in protein structures. However, directly integrating structural information remains challenging due to the scarcity of reliable mutant structures. To address these issues, we propose StructEvo, a novel structure-aware reinforcement learning framework for protein directed evolution. StructEvo employs a delta-structure fusion encoder to approximate mutant structure features via feature differences, enabling dynamic incorporation of spatial knowledge. The vast mutation space is then decomposed into manageable subspaces through a structure-aligned hierarchical action network, while a geometric constraint further stabilizes delta feature learning. Our approach outperforms prior sta
    
[^241]: 诊断与改进大语言模型中的概率推理

    Diagnosing and Improving Probabilistic Reasoning in Large Language Models

    [https://arxiv.org/abs/2609.38005](https://arxiv.org/abs/2609.38005)

    该论文提出一个决策理论框架，将大语言模型的决策损失分解为信念形成与行动选择两个组成部分，并通过合成基准和强化学习干预系统诊断并改进大语言模型的概率推理能力。

    

    大语言模型（LLM）越来越多地被提议作为决策助手，它们必须在明确的决策成本下基于现有证据进行概率推理。我们提出了一个决策理论框架，将LLM的决策损失分解为两个组成部分：从提供的证据中形成准确的信念，以及将这些信念转化为能优化给定效用函数的行动。利用一个具有已知真值的合成基准，我们应用该分解来刻画前沿模型和开源模型的概率推理能力。我们进一步评估了针对信念、决策或两者的强化学习干预能否在三个领域中改进这些组成部分，改进能否在不同组成部分和引出格式之间迁移，以及决策性能能否在信念形成未获改进的情况下得到提升。我们发现，针对概率推理的单一组成部分会重新分配决策损失，改进……（摘要截断）

    arXiv:2609.38005v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly proposed as decision assistants who must reason probabilistically from available evidence under explicit decision costs. We propose a decision-theoretic framework that decomposes LLMs' decision loss into two components: forming accurate beliefs from provided evidence and translating those beliefs into actions that optimize a provided utility function. Using a synthetic benchmark with known ground truth, we apply the decomposition to characterize probabilistic reasoning in frontier and open-sourced models. We further evaluate whether RL interventions targeting beliefs, decisions, or both improve these components across three domains, whether improvements transfer across components and elicitation formats, and whether decision performance can improve without improvement in belief formation. We find that targeting one component of probabilistic reasoning redistributes decision loss, improving th
    
[^242]: 从学习者行为到可复用技能：实现高效且有效的学习者模拟

    From Learner Behavior to Reusable Skills for Effective and Efficient Learner Simulation

    [https://arxiv.org/abs/2609.37157](https://arxiv.org/abs/2609.37157)

    Learner2Skill将从历史交互中获得的学习者模拟能力外化为持久可复用的“模拟技能”，该技能捕获学习者的学习状态与回答模式并随新交互演进，可通过轻量级校准迁移到新的大语言模型，从而更高效、更忠实地模拟学习者行为。

    

    学习者模拟旨在重现特定学习者在新任务上的行为表现。尽管大语言模型（LLMs）能够生成日益精细的学习行为，但现有方法往往需要反复处理不断增长的交互历史来重建学习者。这带来了额外的上下文和推理成本，并且使得所获得的学习者特定模拟能力难以在不同的大语言模型之间复用。因此，我们提出Learner2Skill，它将从历史交互中获得的模拟能力外化为一个持久且可复用的模拟技能（Simulation Skill）。该技能捕获学习者当前的学习状态和反复出现的回答模式，随着新的真实交互的到来而不断演进，并且可以通过轻量级的执行器校准适配到新的大语言模型，而无需从头重建学习者。实验表明，Learner2Skill能够更忠实地重现细粒度的学习行为（摘要原文在此处截断）。

    arXiv:2609.37157v1 Announce Type: new  Abstract: Learner simulation aims to reproduce how a particular learner behaves on new tasks. Although Large Language Models (LLMs) can generate increasingly fine-grained learning behaviors, existing approaches often need to repeatedly process a growing interaction history to reconstruct the learner. This introduces additional context and inference costs and makes the acquired learner-specific simulation capability difficult to reuse across different LLMs. We therefore propose Learner2Skill, which externalizes the simulation capability acquired from historical interactions into a persistent and reusable Simulation Skill. The Skill captures the learner's current learning state and recurring response patterns, evolves as new real interactions arrive, and can be adapted to a new LLM through lightweight executor calibration without reconstructing the learner from scratch. Experiments show that Learner2Skill more faithfully reproduces fine-grained lear
    
[^243]: SEABench：自进化智能体中内生失准的基准测试

    SEABench: Benchmarking Endogenous Misalignment In Self-Evolving Agents

    [https://arxiv.org/abs/2609.35596](https://arxiv.org/abs/2609.35596)

    该论文提出了SEABench基准，用于量化研究自进化智能体因自我修改而在无外部对抗影响下产生的内生失准（不安全行为）风险。

    

    自进化大语言模型智能体因其能够在部署后通过修改自身框架（包括控制器指令、记忆管理协议以及可复用的工具和技能）来响应用户和环境反馈并持续改进而受到广泛关注。然而，在局部看似有用的更新可能会延续到后续任务中，即使没有直接的对抗性影响，也会产生不安全的行为。为了研究这一风险，我们提出了SEABench，一个用于研究由智能体自进化引发的内生失准的基准，包含48个纵向任务序列，涵盖丰富的个人助理环境中多种进化层面、任务领域和危害类型。为了考虑智能体操作中固有的随机性，我们提供了一个自适应轨迹发现流水线，在保持原始任务意图的同时探测故障，并通过配对的非进化智能体等机制支持因果归因。

    arXiv:2609.35596v2 Announce Type: replace-cross  Abstract: Self-evolving LLM agents have gained prominence for their ability to improve after deployment by modifying their harness, including their controller instructions, memory management protocols, and reusable tools and skills, in response to user and environment feedback. However, locally useful updates may persist into later tasks where they produce unsafe behavior, even without direct adversarial influence. To study this risk, we introduce SEABench, a benchmark for studying endogenous misalignment arising from agent self-evolution, with 48 longitudinal task sequences that span multiple evolution surfaces, task domains, and harm types in a rich personal-assistant environment. To account for the stochasticity inherent in agentic operations, we provide an adaptive trajectory discovery pipeline that probes for failures while preserving original task intent and supports causal attribution through paired non-evolving agents and attribu
    
[^244]: AnchorRep：通过表示排斥防御大语言模型的跨模型对抗迁移攻击

    AnchorRep: Defending LLMs Against Cross-Model Adversarial Transfer via Representation Repulsion

    [https://arxiv.org/abs/2609.32602](https://arxiv.org/abs/2609.32602)

    提出AnchorRep防御方法，利用轻量级LoRA适配器将有害提示词的内部表示与锚点模型的表示相排斥，无需对抗样本即可将跨模型对抗迁移攻击的成功率降低至1.1%以下。

    

    arXiv:2609.32602v2 公告类型：replace-cross 摘要：在单个开源权重大语言模型上优化的对抗攻击可以迁移并越狱架构不同的模型，这使得拥有对一个模型白盒访问权限的攻击者能够攻陷独立部署的其他系统。这在模型之间造成了共享的漏洞，然而现有的防御方法并非针对这种跨模型威胁而设计。我们发现跨模型迁移攻击与共享的内部表示几何结构相关，这使其成为一个天然的防御目标。AnchorRep通过一个轻量级的LoRA适配器直接针对这种几何结构进行防御，将被防御模型对有害提示词的内部表示推离冻结的锚点模型对相同提示词的内部表示。训练过程仅使用少量有害提示词，无需任何对抗样本。在五个模型和四种架构家族上的实验表明，AnchorRep将2,000次迁移攻击的跨模型攻击成功率降低至≤1.1%（其中两个模型上为0%），包括在Mistr……（摘要截断于此）

    arXiv:2609.32602v2 Announce Type: replace-cross  Abstract: Adversarial attacks optimized on a single open-weight LLM can transfer to and jailbreak architecturally different models, allowing an attacker with white-box access to one model to compromise independently deployed systems. This creates a shared vulnerability across models, yet existing defenses are not designed for this cross-model threat. We find that cross-model transfer aligns with shared internal representation geometry, making it a natural defense target. AnchorRep targets this geometry directly with a lightweight LoRA adapter that pushes the defended model's internal representations of harmful prompts away from those of a frozen anchor model on the same prompts. Training uses a small set of harmful prompts and no adversarial examples. Across five models and four architectural families, AnchorRep reduces cross-model attack success rate to <=1.1% on 2,000 transferred attacks (0% on two), including the largest drop on Mistr
    
[^245]: 语言推理向量能否增强多模态推理能力？

    Can Linguistic Reasoning Vectors Enhance Multimodal Reasoning Ability?

    [https://arxiv.org/abs/2609.31140](https://arxiv.org/abs/2609.31140)

    该论文提出LIFT方法，通过提取“推理者路径”与“求解者路径”之间的隐状态差异作为推理向量并注入模型，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。

    

    大多数视觉语言模型都是通过在预训练大语言模型基础上扩展视觉模块并进行多模态对齐而构建的。然而，这种多模态扩展往往会降低基座大语言模型中原有的语言侧推理能力。尽管基座大语言模型在扩展后仍保留了可用的推理能力，但经过对齐的视觉语言模型本身却无法可靠地获取这一能力。因此，恢复视觉语言模型中退化的推理能力，向基座大语言模型寻求帮助比仅依靠视觉语言模型本身更为有效。基于这一动机，我们提出了LIFT（Language-side reasonIng Facilitation and Transfer，语言侧推理促进与迁移），这是一种轻量级的向量干预方法，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。LIFT将推理向量定义为具有显式推理轨迹的推理者路径与没有推理轨迹的求解者路径之间的答案词元隐状态差异，并将这些向量注入语言……

    arXiv:2609.31140v1 Announce Type: new  Abstract: Most Vision-Language Models (VLMs) are built by extending pretrained Large Language Models (LLMs) with visual modules and multimodal alignment. However, this multimodal scaling often degrades the language-side reasoning ability originally encoded in the base LLM. While the base LLM retains usable reasoning after scaling, the aligned VLM itself cannot reliably access this ability. Therefore, recovering the degraded reasoning capability in VLMs would benefit more from seeking help from the base LLM than from the VLM alone. Motivated by this, we propose LIFT (Language-side reasonIng Facilitation and Transfer), a lightweight vector-intervention method that transfers reasoning capability from the base LLM to the VLM without retraining the backbone. LIFT defines Reasoning Vectors as answer-token hidden-state differences between a Reasoner path with an explicit reasoning trace and a Solver path without it, and injects these vectors into languag
    
[^246]: 先筛查再服务：面向1.4亿规模生产级客户体验AI代理的仿真验证方法

    Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale

    [https://arxiv.org/abs/2609.30137](https://arxiv.org/abs/2609.30137)

    该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。

    

    客户体验（CX）代理使用工具和大语言模型来处理客户请求，并引导用户与组织的产品进行对话式交互。改进这些代理，尤其是在受监管的行业中，是非常困难的：它们必须检测用户意图、遵循复杂的运营策略并可靠地使用工具。手动端到端测试覆盖范围有限，而在线实验则会让客户直面可能导致信任受损的故障。我们提出了一种基于假设驱动的仿真工作流，用于在部署前筛查候选的CX代理。合成客户会对代理的响应做出反应，模拟的工具输出使多步骤代理工作流无需调用生产后端即可运行。我们在Nubank的Card Delivery代理及其扩展后的继任者Card Management（Nubank在巴西聊天量最高的客服代理）上使用了Snowglobe仿真器。在4个已部署的版本中，仿真与生产环境的版本级二元评估器……

    arXiv:2609.30137v1 Announce Type: new  Abstract: Customer experience (CX) agents use tools and large language models to address customer requests and guide conversational interactions with an organization's products. Improving these agents, especially in regulated industries, is difficult: they must detect intent, follow complex operational policies and use tools reliably. Manual end-to-end testing offers limited coverage, while live experiments expose customers to failures that can erode trust.   We present a hypothesis-driven simulation workflow for screening candidate CX agents before deployment. Synthetic customers react to agent responses and simulated tool outputs enable multi-step agentic workflows without invoking production backends. We use the Snowglobe simulator on Nubank's Card Delivery agent and its expanded successor, Card Management - Nubank's highest-volume chat-support agent in Brazil. Across 4 deployed versions, simulated and production version-level binary evaluator 
    
[^247]: RLVR²：基于可验证评分量规排序的强化学习

    RLVR$^{2}$: Reinforcement Learning with Verifiable Rubric-based Ranking

    [https://arxiv.org/abs/2609.23457](https://arxiv.org/abs/2609.23457)

    本文提出RLVR²，一种基于可验证评分量规排序的强化学习范式，用排序机制取代线性分数聚合，解决了多维度语义异质质量标准在RLVR训练中权衡不可靠的问题。

    

    基于可验证奖励的强化学习（RLVR）正在从具有明确定义正确性信号的任务（如数学和代码）扩展到由多维度评分量规指定的多方面质量要求。由于策略优化在每次 rollout 中只消耗一个标量，基于评分量规的流程必须将多个标准的得分聚合为标量奖励。这种聚合通常被视为简单的分数缩放，但它实际上隐式地决定了训练过程中各质量维度之间的权衡方式。目前主流的做法是对每个标准进行归一化后取线性组合，这隐含地假设了基数分数差异在各标准之间是可比的，并且某一标准上的收益可以补偿另一标准上的失败；当各标准在语义上异质时，这两种假设都不可靠。我们提出了基于可验证评分量规排序的强化学习（RLVR²），这是一种面向基于评分量规的 RLVR 的可验证排序范式。对于每个标准……

    arXiv:2609.23457v1 Announce Type: new  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is expanding from tasks with well-defined correctness signals, such as mathematics and code, toward multifaceted quality requirements specified by multi-dimensional rubrics. Since policy optimization consumes one scalar per rollout, rubric-based pipelines must map multiple criterion scores into a scalar reward. This aggregation is often treated as score scaling, but it implicitly determines how quality dimensions trade off during training. The prevailing practice, normalizing each criterion and taking a linear combination, assumes that cardinal score differences are comparable across criteria and that gains on one criterion compensate for failures on another; both assumptions are unreliable when criteria are semantically heterogeneous. We propose Reinforcement Learning with Verifiable Rubric-based Ranking (RLVR$^2$), a verifiable ranking paradigm for rubric-based RLVR. For each criter
    
[^248]: 安全的代价：LLM智能体中记忆投毒防御在良性场景下的效用与令牌开销

    The Price of Safety: Benign-Case Utility and Token Overhead of Memory-Poisoning Defenses in LLM Agents

    [https://arxiv.org/abs/2609.22818](https://arxiv.org/abs/2609.22818)

    该研究提出了一种控制变量的测量方法，系统评估LLM智能体记忆投毒防御在良性流量下的效用损失与令牌开销，发现写入时防御几乎不带来可察觉的效用成本。

    

    arXiv:2609.22818v1 公告类型：交叉 摘要：针对LLM智能体的记忆投毒防御通常以其阻止攻击的能力来评估。然而，这些防御所处理的流量很少是对抗性的。实施防御的成本需要在每次交互中支付，而其收益却只在少数情况下才能体现。我们开发了一种测量设置，在不同条件下保持记忆后端、检索过程和评判器一致，仅改变防御本身。我们在五段对话中对每个条件测试三次，以区分防御的真实效果与流水线运行中固有的噪声——即使将温度设为零，这种噪声依然显著。在完全良性的流量上，我们测试了三种写入时防御（输入清洗、来源检查和基于LLM的异常检测）以及一种读取时防御（重排序），结果显示写入时防御没有表现出我们能分辨出的效用成本，其95%置信区间大约为±

    arXiv:2609.22818v1 Announce Type: cross  Abstract: Memory-poisoning defenses for LLM agents are typically evaluated by their ability to prevent attacks. However, the traffic they process is rarely adversarial. The cost of implementing a defense is paid with each interaction, while its benefits are only seen in a small percentage of cases. We developed a measurement setup that keeps the memory backend, retrieval process, and judge consistent across different conditions, changing only the defense itself. We test each condition three times across five conversations to distinguish the defense's real effects from noise inherent in the pipeline's runs, which remains significant even at temperature zero. Across three write-time defenses (input sanitization, provenance checking, and LLM-based anomaly detection) and one read-time defense (reranking), tested on entirely benign traffic, the write-time defenses show no utility cost we can resolve, with 95% confidence intervals spanning roughly +/-
    
[^249]: 自主网络事件响应代理的网络靶场评估

    A Cyber Range Evaluation of Autonomous Network Incident Response Agents

    [https://arxiv.org/abs/2609.16541](https://arxiv.org/abs/2609.16541)

    该论文在网络靶场环境中评估了自主网络事件响应代理，发现基于强化学习的代理在防御网络入侵方面总体上优于启发式策略，且性能表现高度依赖于红队的攻击策略。

    

    我们在一个专为人机操作员训练设计的网络靶场中，测试了用于自动化网络入侵响应的代理的性能。该靶场实现了一个仿真的网络环境，具有可变的网络拓扑、红队攻击模拟以及模拟用户代理。防御代理的目标是防止网络中的主机被红队代理访问，同时最小化防御措施所带来的可用性成本。告警通过SIEM平台生成，并映射为代理所使用的数据建模语言。我们测试了启发式代理与通过强化学习习得策略的组合。学习到的策略通过一个对网络进行建模的网络攻击模拟器进行优化，以最小化综合成本。我们发现，强化学习代理在防御系统方面总体上比启发式策略更有效率，且其性能高度依赖于红队的策略（摘要原文在此处截断）。

    arXiv:2609.16541v1 Announce Type: cross  Abstract: We test the performance of agents for automated network intrusion response in a cyber range intended for human operator training. The range implements an emulated networking environment with a variable network topology, red-team emulation and simulated user agents. The goal of the defensive agents is to prevent hosts in the network from being accessed by the red-team agent, while minimizing the availability costs induced from defensive measures. Alerts are generated using a SIEM platform and mapped to a data modeling language used by the agents. We test a combination of heuristic agents and policies learned using reinforcement learning. The learned policies are optimized to minimize the combined cost using a cyber attack simulator modeling the network. We found that the reinforcement learning agents were overall more efficient at defending the system than the heuristic policy, and that the performance depends highly on the policy of th
    
[^250]: DynSHAP：迈向可解释的动态生存分析

    DynSHAP: Towards Explainable Dynamic Survival Analysis

    [https://arxiv.org/abs/2609.13042](https://arxiv.org/abs/2609.13042)

    DynSHAP是一个专为动态生存分析设计的SHAP可解释性框架，通过将时间-特征对作为Shapley博弈的参与者，并引入利用条件采样处理特征时间依赖关系的Temporal DynSHAP，从而能够准确解释纵向不规则输入与函数型生存输出。

    

    面向动态生存分析（DSA）的深度学习模型通过纳入纵向患者数据实现了强大的预测性能，但其黑盒特性限制了临床信任度和应用推广。现有的可解释性方法无法同时处理纵向的不规则输入和函数型的生存输出，这限制了它们在动态生存分析中的可用性。我们提出了DynSHAP，一个专门适用于动态生存分析的SHAP框架。它通过将时间-特征对视为Shapley博弈中的参与者，将常见的边际SHAP估计器扩展到了这一场景。我们进一步提出了Temporal DynSHAP，它学习特征随时间的线性依赖关系，并使用条件采样在解释中对这些依赖关系进行处理。当应用于具有已知真实归因的合成数据时，对于给定的最先进模型，Temporal DynSHAP比边际估计器能更准确地恢复出时间相关特征。应用于两个……（原文摘要此处截断）

    arXiv:2609.13042v1 Announce Type: new  Abstract: Deep learning models for dynamic survival analysis (DSA) achieve strong predictive performance by incorporating longitudinal patient data, but their black box nature limits clinical trust and adoption. Existing explainability methods cannot handle longitudinal, irregular inputs and functional survival outputs simultaneously, which limits their usability in DSA. We propose DynSHAP, a SHAP framework suited specifically for dynamic survival analysis. It extends common marginal SHAP estimators to this setting by treating time--feature pairs as players in the Shapley game. We further introduce Temporal DynSHAP, which learns linear dependencies in features over time and uses conditional sampling to address them in explanations. When applied to synthetic data with known ground-truth attributions, Temporal DynSHAP recovers temporally dependent features more accurately than marginal estimators for a given state-of-the-art model. Applied to two re
    
[^251]: 面向最优传输映射估计及其统计最优性的深度重心回归

    Deep Barycentric Regression for Optimal Transport Map Estimation and its Statistical Optimality

    [https://arxiv.org/abs/2609.06598](https://arxiv.org/abs/2609.06598)

    该论文提出了一种简单的两步方法BROT：先计算未正则化的最优传输方案，再用深度神经网络对所得重心目标进行最小二乘回归，从而实现了兼具统计最优保证与实用可扩展性的最优传输映射估计。

    

    最优传输（OT）映射提供了一种用于对齐概率分布的几何变换，并已成为机器学习中的实用工具。然而，现有的OT映射估计器在严格的统计保证与基于稳定训练目标的实用参数化估计之间仍存在差距。理论性估计器能够达到极小化极大（minimax）最优收敛速度，但它们通常是非参数化的，可能需要苛刻的实现设计或高昂的推断成本。实用估计器则是参数化且可扩展的，但其统计保证仍未被充分研究，且其极小极大式的类对抗训练目标可能对优化算法较为敏感。我们提出了BROT（Barycentric Regression for OT，最优传输重心回归），这是一种简单的两步方法：首先计算未正则化的OT方案，然后通过最小二乘回归将深度神经网络（DNN）拟合到由此产生的重心目标上。在标准正则（原文在此处截断）……

    arXiv:2609.06598v1 Announce Type: cross  Abstract: The optimal transport (OT) map provides a geometric transformation for aligning probability distributions and has become a useful tool in machine learning. However, existing estimators of the OT map still exhibit a gap between sharp statistical guarantees and practical parametric estimation based on stable training objectives. Theoretical estimators achieve minimax optimal convergence rates, but they are typically nonparametric and can incur demanding implementation design or inference costs. Practical estimators are parametric and scalable, but their statistical guarantees remain underexplored, and their min-max, adversarial-like training objectives can be sensitive to optimization algorithms. We propose BROT (Barycentric Regression for OT), a simple two-step method that first computes the unregularized OT plan and then fits a deep neural network (DNN) to the induced barycentric targets by least-squares regression. Under standard regu
    
[^252]: 价值超越语言模型：检测写作中的原创贡献

    Value Over Language Model: Detecting Original Contribution in Writing

    [https://arxiv.org/abs/2609.00700](https://arxiv.org/abs/2609.00700)

    提出了一种无需训练、不评分表面文本的新框架“价值超越语言模型”（Value Over Language Model），通过在不同粒度上提取文档内容并用LLM重建文档，来衡量人在语言模型易于生成的内容之上所贡献的原创价值。

    

    大语言模型（LLM）已在各类写作任务中被迅速采用，这推动了检测LLM生成文本工具的发展。然而，这些工具主要衡量的是文档表面文本中有多少是由LLM撰写的，而并非从根本上设计用于衡量文档中的信息内容或思想有多少源自LLM本身，而非由用户在提示词中提供。在本工作中，我们设计了一个框架，用于衡量一个人在语言模型本身能够轻松生成的内容之上所增加的价值。该方法不需要训练或标注数据，也从不为文档的表面文本打分，从而使其免受文体混淆因素的影响。相反，该方法以递增的粒度级别提取文档内容，使用LLM从每个部分表示中重建文档，并将这些重建结果与仅根据任务描述生成的重建结果进行比较。我们将这一框架称为“价值超越语言模型”（Value Over Language Model）。

    arXiv:2609.00700v1 Announce Type: new  Abstract: LLMs have been rapidly adopted across writing tasks, prompting the development of tools for detecting LLM-generated text. Yet, these tools largely measure how much of a document's surface text was written by an LLM and aren't fundamentally designed to measure how much of the information content or ideas originated from the LLM itself rather than being supplied by the user in the prompt. In this work, we design a framework that measures how much value a person adds on top of what a language model could have easily produced by itself. The method requires no training or labeled data and never scores the document's surface text, insulating it from stylistic confounders. Instead, it extracts the document's content at increasing levels of granularity, uses an LLM to reconstruct the document from each partial representation, and compares these reconstructions with those produced from the task description alone. We call this framework Value Over
    
[^253]: 整合三轴IMU传感器与集成学习以实现有效的帕金森病严重程度分类

    Integrating Triaxial IMU Sensors and Ensemble Learning for Effective Parkinson Disease Severity Classification

    [https://arxiv.org/abs/2608.28602](https://arxiv.org/abs/2608.28602)

    本文提出一种结合三轴IMU传感器（加速度计与陀螺仪）采集的运动和震颤数据与集成学习方法的人工智能系统，用于实现有效的帕金森病严重程度分类。

    

    帕金森病（PD）是一种进行性神经退行性疾病，会对运动功能产生重大影响，导致出现震颤、僵硬、姿势不稳定和运动迟缓等症状。及时的临床治疗、疾病管理和患者的生活质量与早期且适当的帕金森病识别密切相关。近年来，可穿戴传感器技术和人工智能（AI）的发展使得创建无创的、数据驱动的疾病检测方法成为可能。本文提出了一种使用人工智能的对比研究系统，通过分析由惯性测量单元（IMU）捕获的运动和震颤数据来检测帕金森病。该数据包括测量X、Y、Z三个方向运动的加速度计和陀螺仪传感器信号。这些体征和症状为与相关的细微运动缺陷提供了有用的信息。

    arXiv:2608.28602v1 Announce Type: new  Abstract: Parkinson disease PD is a progressive neurodegenerative disease that can have a significant impact on motor performance resulting in the appearance of symptoms such as tremors rigidity postural instabilities and bradykinesia. Timely clinical treatment disease management and quality life of the patients are closely linked to early and appropriate identification of PD. Over the past few years the growth of wearable sensor technology and artificial intelligence AI have made it possible to create noninvasive and data driven disease detection methods. This paper proposes a comparative system using artificial intelligence to detect Parkinsons disease by analyzing the motion and tremor data captured by an inertial measurement unit IMU. The data comprises the signals of the acceleration and gyroscope sensors measuring movement in three directions X Y and Z. The signs and symptoms provide helpful information about subtle motor deficits associated
    
[^254]: openJiuwen：超越静态框架的长时程编码智能体系统

    openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents

    [https://arxiv.org/abs/2608.27969](https://arxiv.org/abs/2608.27969)

    openJiuwen 是一个开源智能体框架，通过“结构可组合性”与“运行时自适应性”两大设计，使长时程编码智能体能够灵活组合能力并根据任务产生的动态证据自适应调整执行。

    

    长时程编码智能体在不断演化的仓库状态上运行，同时日益依赖异构能力、委派智能体以及多智能体协同。这些趋势为智能体框架带来了两个互补的挑战：首先，开发者需要能够组合能力、重新配置执行逻辑并扩展日益复杂的智能体系统，而无需反复重建编排流程；其次，复杂的编码任务会持续产生新的证据——例如语义诊断信息、执行结果、任务进度以及不断变化的上下文相关性——这些证据应当能够动态影响后续的运行时决策。我们将这两类挑战概括为“结构可组合性”与“运行时自适应性”。我们提出了 openJiuwen，一个同时面向开发者可组合性与自适应任务执行的开源框架。openJiuwen 提供了共享的执行底座以及基于 Rail 的能力组合机制，支持跨单个智能体的……

    arXiv:2608.27969v2 Announce Type: replace  Abstract: Long-horizon coding agents operate over evolving repository states while increasingly relying on heterogeneous capabilities, delegated agents, and multi-agent coordination. These trends pose two complementary challenges for the agent harness. First, developers need to compose capabilities, reconfigure execution logic, and scale increasingly complex agent systems without repeatedly rebuilding orchestration. Second, complex coding tasks continuously produce new evidence---such as semantic diagnostics, execution outcomes, task progress, and changing context relevance---that should dynamically influence subsequent runtime decisions. We characterize these challenges as Structural Composability and Runtime Adaptivity. We present openJiuwen, an open-source harness designed for both developer composability and adaptive task execution. openJiuwen provides a shared execution substrate and Rail-based capability composition across single agents,
    
[^255]: 漏洞评估中的AI垃圾内容与幻觉现象：关于推理失败与可信缓解策略的综述

    AI Slop and Hallucinations in Vulnerability Assessment: A Survey on Reasoning Failures and Trustworthy Mitigation

    [https://arxiv.org/abs/2608.25667](https://arxiv.org/abs/2608.25667)

    本综述揭示了LLM在漏洞评估中产生的“AI垃圾内容”（幻觉漏洞、错误补丁、重包装报告）的根源在于安全专家的因果演绎推理与模型自回归概率生成之间的根本鸿沟，并提出了演绎覆盖率得分作为可测量指标，以及思维链提示与工具增强智能体等可信缓解路径。

    

    arXiv:2608.25667v2 公告类型：replace-cross 摘要：大语言模型（LLM）融入网络安全领域彻底改变了漏洞评估方式，但同时也引发了由“AI垃圾内容”无节制泛滥所驱动的可信度危机。这些产物——幻觉生成的漏洞、看似合理却错误的补丁、以及语义上重新包装的缺陷报告——给人工分诊流程带来了认知负担，其效果类似于一次拒绝服务攻击。本文综述了相关实证证据，识别出一个统一的机制，并勾勒出通往可信分诊的路径。我们基于结构化文献综述形式化了AI垃圾内容的分类体系，并剖析了其根本原因：安全专家的因果演绎推理与当前大语言模型自回归概率生成之间的鸿沟。我们通过一个可测量的代理指标——演绎覆盖率得分——将这一鸿沟操作化，并表明思维链提示和使用工具的智能体可以缩小（摘要在此处截断）

    arXiv:2608.25667v2 Announce Type: replace-cross  Abstract: The integration of Large Language Models (LLMs) into cybersecurity has transformed vulnerability assessment, but it has also produced a trustworthiness crisis driven by the unchecked proliferation of "AI slop." These artifacts, hallucinated vulnerabilities, plausible but incorrect patches, and semantically repackaged bug reports, impose a cognitive burden on human triage pipelines that mirrors a denial-of-service attack. This paper surveys the empirical evidence, identifies a unifying mechanism, and traces a path toward trustworthy triage. We formalize a taxonomy of AI slop grounded in a structured literature review and dissect its root cause: the gap between the causal deductive reasoning of security experts and the autoregressive probabilistic generation of current LLMs. We operationalize this gap through a measurable proxy, the Deductive Coverage Score, and show that chain-of-thought prompting and tool-using agents narrow bu
    
[^256]: 一种用于鲁棒时间序列水印的局部令牌化生成模型

    A Locally Tokenized Generative Model for Robust Time-Series Watermarking

    [https://arxiv.org/abs/2608.19727](https://arxiv.org/abs/2608.19727)

    本文提出了一种局部令牌化生成模型L-VQVAE和配套水印方法LVQMark，通过确保每个恢复单元仅依赖有界时间邻域，解决了时间序列水印在后编辑攻击下的检测可靠性问题。

    

    arXiv:2608.19727v1 公告类型：交叉 摘要：水印是生成模型中溯源的核心工具，但其在多变量时间序列中的应用仍因后编辑攻击下的可靠性失败而受阻。我们表明，现有依赖全局耦合重编码的检测器存在零分布的双向漂移问题：后编辑攻击可以将非水印样本的z分数向任一方向偏移，从而使干净校准的阈值失效。我们认为这种不稳定性是重编码的属性，而可靠的检测要求每个恢复单元仅依赖于有界的时间邻域。基于这一原则，我们提出了L-VQVAE，一种生成模型，其中每个离散令牌由短连续窗口生成，以及LVQMark，一种在该令牌空间上的水印方法，结合了logit偏差注入与鲁棒重编码以用于攻击时检测。在四个基准上的实验跨越了多种场景，验证了其有效性。

    arXiv:2608.19727v1 Announce Type: cross  Abstract: Watermarking is a central tool for provenance in generative models, yet its application to multivariate time series remains hindered by reliability failures under post-editing attacks. We show that existing detectors, which rely on globally coupled re-encoding, suffer from bidirectional drift of the null distribution: post-editing attacks can shift the z-score of non-watermarked samples in either direction, invalidating clean-calibrated thresholds. We argue that this instability is a property of the re-encoding, and that reliable detection requires each recovered unit to depend only on a bounded temporal neighborhood. Guided by this principle, we propose L-VQVAE, a generative model in which each discrete token is produced from a short contiguous window, and LVQMark, a watermarking method over this token space that combines logit-bias injection with robust re-encoding for attack-time detection. Experiments on four benchmarks spanning fi
    
[^257]: GigaBrain-WBC-0.5：一种用于与环境交互的鲁棒全身控制的行为世界模型

    GigaBrain-WBC-0.5: A Behavior World Model for Robust Whole-Body Control with Environment Interaction

    [https://arxiv.org/abs/2608.18234](https://arxiv.org/abs/2608.18234)

    本文提出了首个行为世界模型GigaBrain-WBC-0.5，通过因果Transformer联合预测动作、状态和潜在行为命令，使机器人能够建模环境交互，实现鲁棒的全身控制。

    

    arXiv:2608.18234v1 公告类型：交叉 摘要：全身运动跟踪策略将人形机器人转化为一个鲁棒的控制接口：遥操作员——或上游模型——仅提供粗略的运动意图，而低级策略保持机器人平衡和物理可行性。现有的跟踪器仅在平坦地面上提供此接口：在空场景中训练，它们从未学习地形和物体接触如何重塑其动力学，并且它们试图通过不断扩充参考运动语料库来教会策略在任何命令下保持平衡，这在一旦可行行为变得依赖环境时就失效了。我们提出了GigaBrain-WBC-0.5，这是首个用于人形机器人全身控制的行为世界模型（BWM）。与纯粹的反应式跟踪器不同，我们训练了一个因果Transformer来联合预测其下一个动作、下一个状态以及下一个潜在行为命令的分布，因此，行动的网络也建模了环境如何塑造行为。

    arXiv:2608.18234v1 Announce Type: cross  Abstract: Whole-body motion tracking policies turn a humanoid into a robust control interface: the teleoperator---or an upstream model---only supplies a coarse movement intent, while the low-level policy keeps the robot balanced and physically feasible. Existing trackers deliver this interface only on flat ground: trained in empty scenes, they never learn how contact with terrain and objects reshapes their dynamics, and they attempt to teach the policy to balance under any command by continually enlarging the reference-motion corpus, which stops working once feasible behaviors become environment-dependent. We present GigaBrain-WBC-0.5, the first Behavior World Model (BWM) for humanoid whole-body control. Rather than a purely reactive tracker, we train a causal Transformer to jointly predict its next action, next state, and the distribution over its next latent behavior command, so the network that acts also models how the environment shapes what
    
[^258]: 大型语言模型能否以具有法律意义的方式进行推理？一项关于欧洲人权法院案例的小规模研究

    Can LLMs Reason in a Legally Meaningful Manner? A Small-scale Study on European Court of Human Rights Cases

    [https://arxiv.org/abs/2608.17168](https://arxiv.org/abs/2608.17168)

    本研究通过欧洲人权法院案例测试发现，顶尖大型语言模型在法律推理上表现不佳，其分析结构完整但内容浅薄，且自动评估器与人工评估一致性较弱。

    

    摘要：arXiv:2608.17168v1 公告类型：交叉 摘要：推理已成为当代大型语言模型的标准技术和特性；然而，在诸如法律案件预测等要求较高的法律导向任务中，其应用和质量仍未得到充分探索。我们以欧洲人权法院（ECtHR）的法律案例为测试平台，调查了大型语言模型在法律案件预测背景下的推理能力。我们评估了近期顶级模型OpenAI GPT 5.4，通过探索在ECtHR判例背景下对“具有法律意义的推理”更具或更少提示性的不同提示策略。我们通过人工和大型语言模型评估来呈现对模型响应的评估结果。我们发现，所考察的模型在法律推理方面远未达到理想分数，模型生成的分析结构完整但实质肤浅，并且“大型语言模型作为法官”的评估器内部一致但与我们的训练注释者仅有微弱一致性。

    arXiv:2608.17168v1 Announce Type: cross  Abstract: Reasoning has become a standard technique and feature for contemporary LLMs; however, its application and quality in the context of demanding legal-oriented tasks, such as legal case forecasting, remain under explored. We investigate how LLMs reason in the context of legal case forecasting, using legal cases from the European Court of Human Rights (ECtHR) as a testbed. We evaluate OpenAI GPT 5.4, a recent top-tier LLM, by exploring alternative prompting strategies that are more or less suggestive of what counts as legally meaningful reasoning in the context of ECtHR jurisprudence. We present our findings derived from assessing the model's responses with both human and LLM evaluation. We find that the examined model scores far from ideal in legal reasoning, the model produces structurally complete but substantively shallow analyses, and that LLM-as-a-Judge evaluators are internally consistent yet align only weakly with our trained annot
    
[^259]: 基于变分学习的RLVR参数空间探索

    Parameter Exploration for RLVR via Variational Learning

    [https://arxiv.org/abs/2608.09805](https://arxiv.org/abs/2608.09805)

    本文提出了一族名为扰动参数策略优化（3PO）的方法，通过变分学习从后验分布中采样多样化策略，实现参数空间探索，弥补了温度缩放等动作空间探索方法的局限，从而提升LLM强化学习的训练稳定性与下游性能。

    

    探索长期以来一直是强化学习研究的重点。最近，越来越多的证据表明，探索也是LLM强化学习方案中的一个重要组成部分，能够显著影响下游性能。许多现有方法在动作空间中控制探索，例如使用温度缩放。然而，这些方法无法对token进行重新排序，只能影响输出分布的方差，这限制了探索，可能导致训练发散或停滞。本文研究了参数空间探索，其中rollout通过从后验分布中采样不同的策略来生成，每个策略可能探索不同的rollout路径。采样较少或更多样化的策略则成为控制探索的补充手段。我们提出了一族称为扰动参数策略优化的方法，它们使用不同的采样策略和不同的ro

    arXiv:2608.09805v2 Announce Type: replace-cross  Abstract: Exploration has been a focus of reinforcement learning research for a long time. Recently, there has been growing evidence that it is also an important ingredient in LLM reinforcement learning recipes that can significantly impact downstream performance. Many existing methods control exploration in the action-space, for example, using temperature scaling. However, these methods cannot reorder tokens but only influence the variance in the output distribution. This limits exploration and can lead to divergence or stalled training. Here, we investigate parameter-space exploration, where rollouts are generated by sampling different policies from a posterior that may each explore different rollouts. Sampling less or more diverse policies is then a complementary control lever over exploration. We introduce a family of methods called Perturbed Parameter Policy Optimization (3PO) which use different sampling strategies and different ro
    
[^260]: 面向群体规模可扩展的多智能体世界建模

    Population-Scalable Multi-Agent World Modeling

    [https://arxiv.org/abs/2608.08600](https://arxiv.org/abs/2608.08600)

    本文提出Khora，一种可扩展的多智能体世界模型，通过解耦世界状态演化与视觉渲染，实现推理时无需重新训练即可扩展到任意数量的智能体。

    

    arXiv:2608.08600v2 公告类型：替换交叉 摘要：世界模型近年来在视觉预测和交互式生成方面取得了显著进展，但将其扩展到多智能体环境引入了根本性的可扩展性挑战。现有方法通常在训练和推理过程中假设固定数量的智能体，这使模型绑定到预定的智能体群体，并限制了推理时的可扩展性。我们的关键见解是，跨视角一致性应源自一个共享的世界状态，其演化不假设预定义的智能体数量，而智能体特定的观察应通过统一的渲染接口查询该状态来生成。基于这一见解，我们提出了Khora，一种可扩展的多智能体世界模型，支持在无需重新训练的情况下推理时扩展到任意数量的智能体。我们的框架将世界状态演化与视觉渲染解耦，并引入了一种群体无关的渲染机制。

    arXiv:2608.08600v2 Announce Type: replace-cross  Abstract: World models have recently achieved impressive progress in visual prediction and interactive generation, but extending them to multi-agent environments introduces a fundamental scalability challenge. Existing methods generally assume a fixed number of agents during training and inference, which ties the model to a pre-determined agent population and limits inference-time scalability. Our key insight is that cross-view consistency should arise from a shared world state whose evolution does not assume a predefined number of agents, while agent-specific observations should be generated by querying this state through a unified rendering interface. Based on this insight, we propose Khora, a scalable multi-agent world model that supports inference-time expansion to arbitrary numbers of agents without retraining. Our framework decouples world-state evolution from visual rendering and introduces a population-agnostic rendering mechanis
    
[^261]: 面向多模态大语言模型的统一幻觉模糊测试

    Unified Hallucination Fuzzing for Multimodal Large Language Models

    [https://arxiv.org/abs/2608.07525](https://arxiv.org/abs/2608.07525)

    该论文提出了结合统一幻觉分类细粒度数据集UniHall与自适应多模态模糊测试SAMF的系统性评估框架，利用进化变异策略动态探索多模态大语言模型的幻觉边界，克服了传统静态基准覆盖面窄和性能饱和的局限。

    

    幻觉仍然是多模态大语言模型（MLLMs）面临的一个持续性挑战，严重限制了其在高风险应用中的可靠性。现有评估主要基于静态基准测试，存在分类覆盖面窄和性能快速饱和的问题，无法反映模型在不断演变的真实场景中的鲁棒性。为弥合这一差距，我们提出了一个将综合基准与自演化压力测试相结合的系统性评估框架。首先，我们介绍了UniHall，这是一个基于统一分类体系构建的细粒度数据集，涵盖对象、指令和知识三个维度。其次，为解决基准饱和问题，我们提出了自适应多模态模糊测试（SAMF），这是一个采用进化变异策略来探索模型幻觉边界的自适应框架。至关重要的是，为确保对动态输入的可靠评估，SAMF 引入了结构化的（摘要在此处被截断）

    arXiv:2608.07525v2 Announce Type: replace-cross  Abstract: Hallucination remains a persistent challenge for Multimodal Large Language Models (MLLMs), severely limiting their reliability in high-stakes applications. Existing evaluations, predominantly based on static benchmarks, suffer from narrow taxonomical coverage and rapid performance saturation, failing to reflect model robustness in evolving real-world scenarios. To bridge this gap, we present a systematic evaluation framework integrating a comprehensive benchmark with self-evolving stress testing. First, we introduce UniHall, a fine-grained dataset grounded in a unified taxonomy spanning Object, Instruction, and Knowledge dimensions. Second, to address benchmark saturation, we propose Self-Adaptive Multimodal Fuzzing (SAMF), a self-adaptive framework that employs evolutionary mutation strategies to explore the boundaries of model hallucinations. Crucially, to ensure reliable assessment of dynamic inputs, SAMF incorporates a stru
    
[^262]: 从孤立算法到合规优先的智能体平台：医院AI系统的多层架构

    From Siloed Algorithms to Compliance-First Agentic Platforms: A Multi-Layered Architecture for Hospital AI Systems

    [https://arxiv.org/abs/2608.06112](https://arxiv.org/abs/2608.06112)

    该研究提出了一种合规优先的医院智能体AI多层架构，通过智能体编排层和以“策略即代码”为核心的合规与策略层，解决医疗AI部署孤立、治理缺失和难以规模化推广的问题。

    

    医院正在快速采用人工智能用于分诊、影像诊断、日程安排等领域，然而大多数部署仍然是孤立的点状解决方案，被封锁在部门壁垒之中，导致重复劳动、隐性风险和企业价值未能实现。尽管医疗AI市场呈爆炸式增长且投资不断加速，据估计70-80%的医疗AI试点项目无法规模化推广，其主要原因在于治理缺口、数据碎片化以及缺乏集成蓝图。本研究提出了一种医院专属、合规优先的智能体AI架构，包含多个可互操作的层级，对现有医院AI平台模型进行了扩展：(i) 智能体编排层，支持跨临床、运营和财务领域的多智能体工作流；(ii) 合规与策略层，以“策略即代码”的方式集中管理HIPAA、GDPR、欧盟AI法案、DISHA法案、印度DPDP法案以及ISO/IEC安全标准（注：原文摘要在此处截断，内容不完整）。

    arXiv:2608.06112v2 Announce Type: replace  Abstract: Hospitals are rapidly adopting artificial intelligence for triage, imaging, scheduling etc., yet most deployments remain isolated point solutions locked inside departmental silos, resulting in duplicated effort, hidden risks, and unrealized enterprise value. Despite explosive growth of AI in healthcare market and accelerating investment, an estimated 70-80% of healthcare AI pilots fail to scale, largely due to governance gaps, fragmented data, and missing integration blueprints. This research proposes a hospital-specific, compliance-first, Agentic AI architecture with multiple interoperable layers, extending existing hospital AI platform models with: (i) an Agent Orchestration Layer for multi-agent workflows across clinical, operational, and financial domains, (ii) a Compliance and Policy Layer that centralizes policy-as-code for HIPAA, GDPR, the EU AI Act, DISHA Act, India's DPDP Act, and ISO/IEC security and safety standards, and (
    
[^263]: Eigenius：具有认知分层与制度中介推理的类型化知识图谱数据库管理系统

    Eigenius: A Typed Knowledge-Graph DBMS with Epistemic Stratification and Institution-Mediated Reasoning

    [https://arxiv.org/abs/2608.04457](https://arxiv.org/abs/2608.04457)

    Eigenius通过统一内核将数据溯源作为结构性不变量，利用依赖类型、制度和内容寻址存储实现认知状态的强制管理。

    

    arXiv:2608.04457v2 公告类型：替换交叉。摘要：随着“AI科学家”通过模型上下文协议（MCP）推动研究，依赖临时脚本的系统将失效。海量的、有状态且相互关联的证据需要一种基于专门数据库架构的、机器可遍历的保证。Eigenius是一个开源、类型化的知识图谱数据库管理系统，其构建基于一个前提：回答审计问题（“你知道什么，你的保证是什么？”）需要一个统一的内核。通过紧密耦合类型系统、存储引擎和集成协议，Eigenius将数据溯源转化为结构性不变量，而非跨子系统边界重建的属性。该内核基于三大支柱：贯穿核心的依赖类型理论、作为强类型集成边界的制度，以及内容寻址的不可变存储层。在此基础之上，认知状态（声明/观察/派生/验证）得到强制执行。

    arXiv:2608.04457v2 Announce Type: replace-cross  Abstract: As "AI Scientists" emerge to drive research via the Model Context Protocol (MCP), systems relying on ephemeral scripts will fail. The sheer scale of stateful, interconnected evidence requires a machine-walkable warranty grounded in a purpose-built database architecture. Eigenius is an open-source, typed knowledge-graph DBMS built on a single premise: answering the audit question ("what do you know, and what is your warranty?") requires a unified kernel. By tightly coupling the type system, storage engine, and integration protocol, Eigenius turns data provenance into a structural invariant rather than a property reconstructed across subsystem boundaries. The kernel rests on three pillars: a dependent type theory woven through the core, institutions acting as strongly typed integration boundaries, and a content-addressed immutable storage layer. On this foundation, epistemic status (declared/observed/derived/verified) is enforced
    
[^264]: 变量感知表示而非关系类型化解释了事件图模型在ICU死亡率预测中的性能：时间性与外部验证

    Variable-aware representation, not relation typing, explains the performance of an event-graph model for ICU mortality prediction: temporal and external validation

    [https://arxiv.org/abs/2608.02663](https://arxiv.org/abs/2608.02663)

    该研究通过严格的时间性与外部多中心验证发现，事件图模型在ICU死亡率预测中的性能提升主要来自变量感知表示（变量学习嵌入加逐变量归一化），而非关系类型化的消息传递。

    

    事件图神经网络将重症监护病房（ICU）记录表示为由带时间戳的观测所连接的类型化实体，其性能提升通常被归因于关系特定的消息传递。我们测试了在消除数据泄漏并采用时间性与外部验证的情况下，哪些组件真正起作用。在重症监护医学信息集市IV（MIMIC-IV）数据库的31,142次至少48小时的首个ICU住院中，预测住院48小时后院内死亡率的模型在2008-2016年的住院数据上开发，在留出的2008-2016年住院数据以及2017-2019年和2020-2022年的住院数据上进行测试，并无需重新训练地应用于来自eICU协作研究数据库中205家医院的54,696次住院数据。此前版本中存在的数据泄漏——将ICU总住院时长作为输入——已被移除。变量感知表示（即每个节点变量的学习嵌入加上逐变量归一化）提升了平均单模型受试者工作特征曲线下面积（AUC）……

    arXiv:2608.02663v2 Announce Type: replace-cross  Abstract: Event-graph neural networks represent intensive care unit (ICU) records as typed entities linked by timestamped observations, and their gains are often attributed to relation-specific message passing. We tested which components matter once leakage is removed and validation is temporal and external. In 31,142 first ICU stays of at least 48 hours in the Medical Information Mart for Intensive Care IV (MIMIC-IV), models predicting post-48-hour in-hospital mortality were developed on 2008-2016 stays, tested on held-out 2008-2016 stays and on 2017-2019 and 2020-2022 stays, and applied without retraining to 54,696 stays from 205 hospitals in the eICU Collaborative Research Database. A leak in a previous version, total ICU length of stay used as an input, was removed. A variable-aware representation (a learned embedding of each node's variable plus per-variable normalisation) raised the mean single-model area under the receiver operati
    
[^265]: 局部坐标系旋转注意力用于抗体特异性表位预测

    Local-Frame Rotary Attention for Antibody-Specific Epitope Prediction

    [https://arxiv.org/abs/2608.01092](https://arxiv.org/abs/2608.01092)

    提出局部坐标系旋转注意力（LFRA），将主链局部坐标系下的残基间三维位移直接注入旋转注意力，以取代传统一维序列位置机制，从而提升抗体特异性表位预测的准确性。

    

    arXiv:2608.01092v2 公告类型：替换 摘要：抗体特异性表位预测旨在识别给定抗体所识别的抗原残基，这一任务依赖于抗体互补决定区（CDR）与抗原表面之间的三维互补性。现有方法通常利用蛋白质语言模型（PLM）嵌入，并通过额外的图、表面或点云编码器注入结构信息，而注意力内部的位置机制在很大程度上仍然与一维序列顺序绑定。对于蛋白质而言，token偏移的对应概念不仅是序列间隔，还包括折叠后残基之间的三维位移。这引出了一个问题：折叠后的残基几何能否直接作为注意力本身的位置机制？我们提出局部坐标系旋转注意力（LFRA），它在由主链定义的局部坐标系中表达残基间的位移，并将其直接注入旋转注意力。该设计在保持连续的方向性几何信息的同时……（原文摘要在此处截断）

    arXiv:2608.01092v2 Announce Type: replace  Abstract: Antibody-specific epitope prediction aims to identify which antigen residues are recognized by a given antibody, a task that depends on the three-dimensional complementarity between antibody CDRs and the antigen surface. Existing methods usually leverage PLM embeddings and inject structure through additional graph, surface, or point-cloud encoders, where the positional mechanism inside attention remains largely tied to one-dimensional sequence order. For proteins, the analogue of a token offset is not only sequence separation, but also the three-dimensional displacement between residues after folding. This raises a question, can folded residue geometry serve as the positional mechanism of attention itself? We propose Local-Frame Rotary Attention (LFRA), which expresses inter-residue displacements in backbone-defined local frames and injects them directly into rotary attention. This design preserves continuous directional geometry whi
    
[^266]: TraceViT：面向视觉抽象推理的接地轨迹监督

    TraceViT: Grounded Trace Supervision for Visual Abstract Reasoning

    [https://arxiv.org/abs/2607.29586](https://arxiv.org/abs/2607.29586)

    提出TraceViT，通过语义单调变换链和软轨迹对齐来监督循环视觉推理器的中间迭代步骤，使其逐步遵循变换过程，从而提升ARC视觉抽象推理能力。

    

    抽象与推理语料库（ARC）测试模型能否从少量输入-输出示例中推断出未见过的变换，并将其应用于新的网格。循环视觉推理器通过多次迭代来细化预测，但传统训练仅约束最终输出，导致中间细化过程不受约束。我们提出这些细化过程应当逐步遵循变换过程。我们引入TraceViT，一个使用语义单调变换链训练的循环视觉推理器。这些变换链通过重写并验证程序化任务实现来获得，将每个解决方案分解为中间网格状态。每次迭代都由从少样本演示中导出的任务参考以及表示当前网格状态的对象工作区来接地。由于这些变换链的长度可能与循环次数不同，软轨迹对齐仅强制其顺序……

    arXiv:2607.29586v2 Announce Type: replace-cross  Abstract: The Abstraction and Reasoning Corpus (ARC) tests whether a model can infer an unseen transformation from a few input-output examples and apply it to a new grid. Looped visual reasoners refine predictions over multiple iterations, but conventional training constrains only the final output, leaving intermediate refinements unconstrained. We propose that these refinements should instead follow the transformation step by step. We introduce TraceViT, a looped visual reasoner trained with semantically monotonic transformation chains. We obtain these chains by rewriting and verifying programmatic task implementations, decomposing each solution into intermediate grid states. Each iteration is grounded by a task reference derived from the few-shot demonstrations and an object workspace representing the current grid state. Because these chains may differ in length from the loop, soft trace alignment enforces only their ordering, letting 
    
[^267]: 当激活神谕学会不去阅读：微调神谕中的概念特异性盲区

    When Activation Oracles Learn Not to Read: Concept-Specific Blind Spots in Fine-Tuned Oracles

    [https://arxiv.org/abs/2607.23379](https://arxiv.org/abs/2607.23379)

    该论文发现，微调后的激活神谕可能变成“概念特异性反读取器”，即选择性地无法读取其自身训练中持续存在的隐藏概念，而非成为专业读取器。

    

    激活神谕是经过训练、能够回答关于另一个模型内部激活的自然语言问题的语言模型。它们为从模型状态中读取隐藏信息提供了灵活的接口，尤其是当相关信息已在内部表示、但在可见行为中缺失或不完整时。然而，AOs 本身也是通过学习得到的系统：它们的答案由训练数据、训练目标以及习得的报告行为所塑造，而非对所表示信息的中性读出。我们在一个受控的“禁忌词猜词”设置中研究这一现象，其中目标模型被微调为在内部使用一个隐藏概念，同时避免直接披露。与“在此类目标模型上训练的 AO 会成为专业读取器”的预期相反，我们发现经过微调的 AOs 可能变成概念特异性的“反读取器”：它们选择性地无法恢复在其自身训练期间持续存在的概念。

    arXiv:2607.23379v2 Announce Type: replace-cross  Abstract: Activation Oracles (AOs) are language models trained to answer natural-language questions about another model's internal activations. They offer a flexible interface for reading hidden information from model states, especially when relevant information is internally represented but absent or incomplete in visible behavior. However, AOs are themselves learned systems: their answers are shaped by training data, objectives, and learned reporting behavior, rather than being neutral readouts of represented information. We study this in a controlled Taboo Word Guessing setting, where subject models are fine-tuned to internally use a hidden concept while avoiding direct disclosure. Contrary to the expectation that an AO trained on such a subject becomes a specialist reader, we find that fine-tuned AOs can become concept-specific anti-readers: they selectively fail to recover the concept persistently present during their own training. 
    
[^268]: 面向网络化系统的不变量发现

    Invariant Discovery for Networked Systems

    [https://arxiv.org/abs/2607.22944](https://arxiv.org/abs/2607.22944)

    本文提出 Autogram，通过让 LLM 生成紧凑的候选不变量文法并在其内进行确定性统计搜索、再迭代扩展文法直至恢复已知可信不变量，实现了无需人工文法且能容忍真实噪声的网络系统不变量自动发现。

    

    不变量是指网络中各测量信号之间预期成立的关系，它是从验证、流量生成、遥测数据填补到输入校验等众多应用的基础。然而，手工编写这些不变量需要同时具备形式逻辑与网络领域的稀缺专业知识。在网络规模下，对所有遥测变量的可能组合进行统计检验是难以实现的；而现有的自动化工具仍然需要人工编写可接受不变量的文法，并且只能学习精确的、“硬性”的规则，这类规则在真实世界的噪声下会失效。我们的核心思路是利用大语言模型（LLM）检查变量元数据、对计数器数据类型进行分类，并预测候选的句法关系以形成一个紧凑的文法，然后在该文法内进行确定性的统计搜索。如果初始文法无法在留出的遥测数据上恢复出可信的已知不变量，Autogram 会指示 LLM 迭代地扩展文法，直至恢复成功。

    arXiv:2607.22944v2 Announce Type: replace-cross  Abstract: Invariants, the relations expected to hold among measured signals of a network, underpin applications from verification to traffic generation, telemetry imputation, and input validation. Yet, writing them by hand demands rare expertise in both formal logic and networking. Testing all possible combinations of telemetry variables statistically is intractable at network scale, while existing automatic tools still require a human-authored grammar of admissible invariants and learn only exact, "hard" rules that break under real-world noise. Our key idea is to use an LLM to inspect variable metadata, classify counter data types, and predict candidate syntactic relationships that form a compact grammar, followed by a deterministic statistical search within that grammar. If the initial grammar fails to recover trusted known invariants on held-out telemetry, Autogram instructs the LLM to iteratively widen the grammar until recovery succ
    
[^269]: 面向时间序列预测中不确定成分的扩散轨迹差分

    Differencing the Diffusion Trajectory toward Uncertain Components for Time Series Forecasting

    [https://arxiv.org/abs/2607.22599](https://arxiv.org/abs/2607.22599)

    提出DiffDiff框架，将时间序列预测中“缓慢变化成分由历史锚定、高频动态承载残余不确定性”的可预测性不对称性直接嵌入扩散轨迹本身，使单一端到端的扩散过程能够感知目标中哪些部分已被历史确定，从而改进概率时间序列预测。

    

    扩散模型已成为概率时间序列预测中被广泛使用的框架，用于在给定观测历史的条件下建模未来值的分布。然而，在时间序列预测中，未来是观测历史的延续，这产生了一种标准扩散过程未曾处理的不对称性：缓慢变化的内容在很大程度上由观测历史的连续性决定，而更高频率的动态则承载了大部分的残余不确定性。现有的基于扩散的预测器在生成之前通过外部规则来解耦这种不对称性，使得破坏（corruption）轨迹对历史已经能够锚定的目标部分一无所知。我们提出了DiffDiff，一种将这种可预测性不对称性直接嵌入扩散轨迹本身的扩散框架，使得单一的端到端扩散过程能够感知历史已经可以锚定目标的哪些部分。DiffDiff使前向操作……（原文摘要在此处截断）

    arXiv:2607.22599v2 Announce Type: replace  Abstract: Diffusion models have become a widely used framework for probabilistic time series forecasting, modeling the distribution of future values given an observed history. In time series forecasting, however, the future continues the observed history, creating an asymmetry the standard diffusion process leaves unaddressed, with slowly-varying content largely determined by the observed continuity while higher-frequency dynamics carry most of the residual uncertainty. Existing diffusion-based forecasters decouple this asymmetry through an external rule before generation, leaving the corruption trajectory blind to which parts of the target the history can already anchor. We propose DiffDiff, a diffusion framework that embeds this predictability asymmetry into the diffusion trajectory itself, so that a single end-to-end diffusion process becomes aware of which parts of the target the history can already anchor. DiffDiff makes the forward opera
    
[^270]: 世界模型作为对手：面向鲁棒运动规划的多智能体自博弈微调

    World Models as Adversaries: Multi-Agent Self-Play Fine-Tuning for Robust Motion Planning

    [https://arxiv.org/abs/2607.10630](https://arxiv.org/abs/2607.10630)

    该论文提出对抗世界建模（AWM）框架，将规划器自身的世界模型转化为角色条件化的对抗性对手，通过多智能体自博弈微调生成稀疏且场景自适应的攻击，从而在罕见安全关键场景下训练出鲁棒的自动驾驶运动规划器。

    

    在密集交通中实现鲁棒运动规划，要求自动驾驶车辆能够在自然驾驶数据中代表性不足的罕见且安全关键场景中进行交互。尽管对抗性训练提供了一种可行的解决方案，但现有方法通常依赖外部场景生成器、启发式扰动或繁重的仿真器推演，这使其难以与现代自回归规划器相集成。在此，我们将对抗鲁棒的规划器学习构建为一个受约束的极小极大博弈，并提出对抗世界建模，这是一个具有理论基础的多智能体自博弈微调框架。由于精确求解该博弈是不可行的，AWM 引入了一种有原则的解耦求解器。在内层最小化中，规划器的预测世界模型被转化为一个角色条件化的对手，通过反事实信用分配学习稀疏的、场景自适应的攻击联盟。在外层最大化中……（原文摘要至此处截断）

    arXiv:2607.10630v2 Announce Type: replace-cross  Abstract: Robust motion planning in dense traffic requires autonomous vehicles to interact in rare and safety-critical scenarios that are underrepresented in naturalistic driving data. Although adversarial training offers a feasible solution, existing methods often rely on external scenario generators, heuristic perturbations, or simulator-heavy rollouts, which makes them difficult to integrate with modern autoregressive planners. Here, we cast adversarially robust planner learning as a constrained min-max game and propose Adversarial World Modeling (AWM), a theoretically grounded multi-agent self-play fine-tuning framework. Since solving the exact game is intractable, AWM introduces a principled decoupled solver. In the inner minimization, the planner's predictive world model is converted into a role-conditioned adversary that learns sparse, scene-adaptive attack coalitions via counterfactual credit assignment. In the outer maximization
    
[^271]: HyperVAttention：基于时空聚类的高效稀疏注意力方法用于视频扩散

    HyperVAttention: Efficient Sparse Attention with Spatio-Temporal Clustering for Video Diffusion

    [https://arxiv.org/abs/2607.03012](https://arxiv.org/abs/2607.03012)

    提出HyperVAttention——一种无需训练的稀疏注意力框架，通过3D局部窗口聚类和自定义Triton内核，同时降低视频扩散Transformer的聚类开销并提升CTA利用率，实现高效的长视频生成加速。

    

    视频扩散Transformer（VDiTs）在高保真视频生成方面展现出了卓越的能力。然而，其生成长时长视频的能力从根本上受到自注意力机制二次方复杂度的制约。近期基于聚类的稀疏注意力方法通过将语义相似的token分组来改善质量与速度的权衡，但其实际效率仍受两个瓶颈的限制：可观的聚类开销，以及由不规则聚类块导致的低CTA利用率。我们提出了HyperVAttention（HVA），这是一个无需训练的稀疏注意力框架，可同时解决上述两个瓶颈。为降低聚类开销，我们引入了3D局部窗口聚类，利用视频token的时空局部性将质心搜索限制在固定的局部邻域内，并通过自定义Triton内核实现高效执行。我们进一步……（原文摘要在此处截断）

    arXiv:2607.03012v2 Announce Type: replace-cross  Abstract: Video Diffusion Transformers (VDiTs) have demonstrated significant capabilities in high-fidelity video generation. However, their ability to produce long-duration videos is fundamentally constrained by the quadratic complexity of the self-attention mechanism. Recent clustering-based sparse attention methods improve the quality-speed trade-off by grouping semantically similar tokens, but their practical efficiency remains limited by two bottlenecks: substantial clustering overhead and low CTA utilization caused by irregular cluster-induced blocks. We propose HyperVAttention (HVA), a training-free sparse attention framework that addresses both bottlenecks jointly. To reduce clustering overhead, we introduce 3D local-window clustering, which exploits the spatio-temporal locality of video tokens to restrict centroid search to fixed local neighborhoods, and implement it with a custom Triton kernel for efficient execution. We further
    
[^272]: 状态-预测分离假设

    The State-Prediction Separation Hypothesis

    [https://arxiv.org/abs/2607.01218](https://arxiv.org/abs/2607.01218)

    该论文提出“状态-预测分离假设”，设计了一种用独立计算流将“预测下一个token”与“存储状态”两种功能解耦的 Transformer 变体，多规模预训练实验表明其具有更高的数据与计算效率，在下游任务上平均超越标准 Transformer 2-3 个百分点。

    

    Transformer 使用同一条前向计算流来同时完成两个任务：预测下一个 token，以及为未来的 token 预测存储有用的状态。我们提出了状态-预测分离假设：将这两种角色解耦能够带来更好的语言建模性能。我们设计了一种使用独立计算流来分离这两种功能的 Transformer 变体，并在多种规模上开展了预训练实验。实验表明，状态-预测分离始终能带来更好的数据效率和计算效率，降低了验证损失，并在下游任务上平均比标准 Transformer 高出 2-3 个百分点。我们还进行了广泛的实证分析，排除了潜在的混杂因素，并证明了我们的设计在梯度上所体现出的根本性差异。

    arXiv:2607.01218v2 Announce Type: replace-cross  Abstract: Transformers use the same forward computation stream to both predict the next token and store useful state for future token predictions. We formulate the state-prediction separation hypothesis: disentangling the two roles yields better language modeling performance. We design a Transformer variant that uses separate computation streams to separate the two functions, and conduct pretraining experiments across various scales. Our experiments show that state-prediction separation consistently offers better data- and compute-efficiency, improving validation loss and outperforming standard Transformers by 2-3 percentage points on average on downstream tasks. We also conduct extensive empirical analysis that rules out potential confounders and demonstrates the fundamental difference in the gradients our design entails.
    
[^273]: 面向语言模型智能体系统的集合值路由：一个WildChat基准与成本感知评估

    Set-Valued Routing for Language-Model Agent Systems: A WildChat Benchmark and Cost-Aware Evaluation

    [https://arxiv.org/abs/2606.28925](https://arxiv.org/abs/2606.28925)

    该论文将语言模型智能体系统的路由从单一top-1选择扩展为集合值路由，构建了基于WildChat的3000条提示、12个能力定义智能体的新基准，并通过成本感知评估证明微调文本编码器（89.59 F1）显著优于线性SVM和零样本GPT-4o。

    

    语言模型智能体系统必须在规划或工具执行之前，将自然语言请求映射到一个或多个专用智能体。这一决策通常被处理为top-1选择，然而组合式请求可能需要多种能力，而不必要的分发则会增加成本。我们将固定目录下的路由问题表述为集合值语言理解任务，并构建了一个包含3,000条WildChat提示、覆盖12个按能力定义的智能体的基准测试。可复现的模式引导参考集合按智能体频率和路由基数进行了平衡。我们比较了画像检索、多标签分类器、微调文本编码器以及基于目录条件化的零样本GPT-4o。评估综合考察了集合质量、延迟、能力覆盖率和序数成本效用。微调编码器达到了89.59 F1，相比之下线性SVM为71.79，GPT-4o仅为41.51，并且在更难的语义组划分下仍保持88.02 F1。我们进一步引入了加权智能体路由……（摘要原文在此处截断）

    arXiv:2606.28925v3 Announce Type: replace-cross  Abstract: Language-model agent systems must map natural-language requests to one or more specialized agents before planning or tool execution. This decision is often treated as top-1 selection, although compositional requests can require several capabilities and unnecessary dispatch raises cost. We formulate fixed-catalog routing as set-valued language understanding and construct a benchmark of 3,000 WildChat prompts over 12 capability-defined agents. Reproducible schema-guided reference sets are balanced by agent frequency and route cardinality. We compare profile retrieval, multilabel classifiers, a fine-tuned text encoder, and catalog-conditioned zero-shot GPT-4o. Evaluation combines set quality, latency, capability coverage, and ordinal-cost utility. The encoder attains 89.59 F1, compared with 71.79 for linear SVM and 41.51 for GPT-4o, and retains 88.02 F1 under a harder semantic-group split. We further introduce Weighted Agent Routi
    
[^274]: 智能体弃权：智能体知道何时应该停止而非行动吗？

    Agentic Abstention: Do Agents Know When to Stop Instead of Act?

    [https://arxiv.org/abs/2606.28733](https://arxiv.org/abs/2606.28733)

    该论文提出了“智能体弃权”这一新概念，将LLM智能体在多轮交互中判断何时应停止行动定义为序列决策问题，并在超过28,000条任务指令上评估了13个LLM智能体系统和2个智能体脚手架。

    

    LLM智能体被期望在多轮交互中采取行动，利用搜索、浏览界面和工具来完成用户目标。然而，并非所有目标都被清晰定义，或在可用环境中能够实现。在这种情况下，可靠的智能体应当认识到进一步交互不太可能有帮助，从而放弃额外的工具调用。我们定义了“智能体弃权”这一问题，即决定智能体在不确定性下何时应停止行动。与标准LLM弃权通常被评估为单轮的“回答或弃权”决策不同，智能体弃权是一个序列决策问题：智能体在每一轮可以选择回答、弃权或收集更多信息，而弃权的必要性可能只有在与环境交互之后才会变得明确。我们在网页购物、终端环境和问答任务上研究了这个问题，评估了13个以LLM作为智能体的系统和2个智能体脚手架，涵盖超过28,000条任务指令。我们的研究…

    arXiv:2606.28733v2 Announce Type: replace  Abstract: LLM agents are expected to act over multiple turns, using search, browsing interfaces, and tools to complete user goals. Yet not every goal is well specified or achievable in an available environment. In such cases, a reliable agent should recognize that further interaction is unlikely to help and abstain from additional tool calls. We define Agentic Abstention, the problem of deciding when an agent should stop acting under uncertainty. Unlike standard LLM abstention, which is usually evaluated as a single-turn answer-or-abstain decision, agentic abstention is a sequential decision problem: an agent can answer, abstain, or gather more information at each turn, and the need to abstain may only become clear after interacting with the environment. We study this problem across web shopping, terminal environments, and question answering, evaluating 13 LLM-as-agent systems and 2 agent scaffolds on more than 28,000 task instructions. Our re
    
[^275]: 一种用于电影心脏MRI时空建模的潜在常微分方程方法

    A Latent ODE Approach to Spatiotemporal Modeling of Cine Cardiac MRI

    [https://arxiv.org/abs/2606.26718](https://arxiv.org/abs/2606.26718)

    提出了一种基于潜在常微分方程的动态模型，通过连续潜在轨迹编码心脏MRI的时空信息，并利用偏差预测心力衰竭风险，显著提升了传统风险模型的性能。

    

    心脏磁共振成像（CMR）捕捉了关于心室结构和运动的丰富时空信息，但传统风险模型仅使用来自选定心脏阶段的少数图像衍生指标。我们提出了一种潜在动态模型，该模型将双心室解剖结构和全周期电影运动编码为连续潜在轨迹，利用心率感知的神经常微分方程（ODE）动力学和基于图形的网格自编码器来重建解剖一致的三维+时间心室运动。一个协变量条件先验定义了预期的舒张末期潜在状态，而Cox比例风险模型测试偏离该先验是否可预测新发心力衰竭。我们研究了72,386名无基线心血管疾病的英国生物银行参与者，包括367例新发心力衰竭事件。在一个保留的评估子集中，将潜在评分添加到重新拟合的集合方程中改善了其表现。

    arXiv:2606.26718v1 Announce Type: new  Abstract: Cardiac magnetic resonance imaging (CMR) captures rich spatiotemporal information about ventricular structure and motion, but conventional risk models use only a few image-derived indices from selected cardiac phases. We present a latent dynamical model that encodes bi-ventricular anatomy and full-cycle cine motion as a continuous latent trajectory, using heart-rate-aware neural ordinary differential equation (ODE) dynamics and a graph-based mesh autoencoder to reconstruct anatomically consistent 3D+t ventricular motion. A covariate-conditioned prior defines the expected end-diastolic latent state, and a Cox proportional hazards model tests whether deviations from this prior predict incident heart failure. We studied 72,386 UK Biobank participants without baseline cardiovascular disease, including 367 incident heart failure events. In a held-out evaluation subset, adding the latent score to refitted pooled cohort equations improved the s
    
[^276]: EvoOptiGraph：基于图结构生成的弱点驱动协同进化优化建模方法

    EvoOptiGraph: Weakness-Driven Coevolution via Graph-Based Structural Generation for Optimization Modeling

    [https://arxiv.org/abs/2606.26578](https://arxiv.org/abs/2606.26578)

    提出EvoOptiGraph框架，通过将优化模型表示为图结构并利用弱点驱动的协同进化，自动生成结构多样的训练数据，从而提升大语言模型在优化建模任务上的性能。

    

    利用大语言模型从自然语言自动生成优化模型面临两个关键挑战。首先，训练语料库缺乏结构多样性。其次，数据生成流程保持静态，与模型学习相脱节。为解决这些挑战，我们提出了EvoOptiGraph，这是一种新颖的框架，其中数据和模型以模型弱点为驱动力协同进化。EvoOptiGraph将每个混合整数线性规划表示为属性二分图，并应用保持有效性的进化算子来生成结构多样的实例。通过确定性编译和验证性反向翻译，进化后的图被转换为求解器代码和自然语言。训练分为两个阶段：在初始数据集上进行监督微调，随后进行带可验证奖励的强化学习，其中基于图的弱点信号指导生成针对模型当前薄弱点的新实例。

    arXiv:2606.26578v1 Announce Type: new  Abstract: Automating optimization modeling from natural language with large language models (LLMs) faces two key challenges. First, training corpora lack structural diversity. Second, data generation pipelines remain static and decoupled from model learning. To address these challenges, we propose EvoOptiGraph, a novel framework where data and model co-evolve, driven by model weaknesses. EvoOptiGraph represents each mixed-integer linear program (MILP) as an attributed bipartite graph and applies validity-preserving evolutionary operators to generate structurally diverse instances. The evolved graphs are converted into solver code and natural language via deterministic compilation and verified back-translation. Training proceeds in two stages: supervised fine-tuning (SFT) on an initial dataset, followed by reinforcement learning with verifiable rewards (RLVR), where graph-derived weakness signals guide the generation of new instances targeting the 
    
[^277]: ThermoLLM：基于时空上下文的LLM多区域暖通空调（HVAC）控制

    ThermoLLM: LLM-Based Multi-Zone HVAC Control with a Spatiotemporal Context

    [https://arxiv.org/abs/2606.22911](https://arxiv.org/abs/2606.22911)

    本文提出ThermoLLM，将建筑空间语义、区域热动力学与滚动历史作为LLM上下文的核心，从而实现更一致、更可泛化且能感知区域间延迟热交互的多区域HVAC控制。

    

    多区域HVAC控制是一个空间决策问题，室内热演化与控制决策不仅取决于室外条件和内部得热，还取决于区域布局、物理相邻关系以及整个建筑内跨区域的延迟热交互。近年来基于大语言模型（LLM）的HVAC控制器已经表明基于提示词的控制是可行的，然而这些方法产生不一致的结果、难以泛化，并存在控制延迟问题。其性能还严重依赖于数值奖励值、文本反馈或非结构化检索，这限制了它们做出能够兼顾区域动力学与热力学的控制决策的能力。目前尚无基于LLM的方法在将建筑语义与动力学连同滚动历史作为LLM上下文核心的同时，与现有基于LLM的HVAC方法进行控制性能的测试与比较。本文提出了ThermoL……（原文摘要在此处截断）

    arXiv:2606.22911v2 Announce Type: replace  Abstract: Multi-zone HVAC control is a spatial decision problem in which indoor thermal evolution and control decisions depend not only on outdoor conditions and internal heat gains but also on zone layout, physical adjacency, and delayed thermal interactions across the building. Recent LLM-based HVAC controllers have shown that prompt-based control is feasible. However, these methods produce inconsistent results, are difficult to generalize, and experience control-latency issues. Their performance also relies heavily on numeric reward values, textual feedback, or unstructured retrieval, which limits their ability to make control decisions that account for zone dynamics and thermodynamics. No existing LLM-based methods have used building semantics and dynamics, together with rolling history, as the centre of the LLM context while testing and comparing HVAC control performance with existing LLM-based HVAC approaches. This paper presents ThermoL
    
[^278]: 控制抖动：联合训练的感知门控专家实现平滑地形过渡

    Keeping Jerk in Check: Jointly Trained Perception-Gated Experts for Smooth Terrain Transitions

    [https://arxiv.org/abs/2606.19633](https://arxiv.org/abs/2606.19633)

    该论文提出CTS-MoE，将多地形腿式运动表述为多任务强化学习，通过联合训练感知门控的稠密专家混合actor与任务特定的critics，利用软路由组合专家知识，实现跨越不连续地形时连续平滑且保留专业化的运动行为。

    

    在不连续地形上进行感知式腿式运动需要共享的运动知识、任务特定的机动动作，以及二者之间的平滑过渡。现有方法独立训练专用策略，然后要么通过分层选择器在它们之间切换，要么将它们蒸馏为单一的整体策略。这两种方法都在行为组合之前就固定了各个组成行为，并且都通过整体赛道结果进行评估，导致地形边界处的行为特征未被表征。我们转而提出CTS-MoE，将多地形运动表述为多任务强化学习，并联合训练一个感知门控的稠密专家混合执行器（actor）与任务特定的评论家（critics）：软路由将专家知识组合成连续变化的行为，而分离的评论家隔离了异构的目标，使策略专业化在联合优化中得以保留。并行的师生框架将该协……

    arXiv:2606.19633v2 Announce Type: replace-cross  Abstract: Perceptive legged locomotion across discontinuous terrain requires shared locomotion knowledge, task-specific maneuvers, and smooth transitions between them. Existing approaches train specialized policies independently and either switch among them with a hierarchical selector or distill them into a monolithic policy. Both fix the constituent behaviors before they are composed, and both are assessed through aggregate course outcomes that leave behavior at terrain boundaries uncharacterized. We instead propose CTS-MoE, which formulates multi-terrain locomotion as multi-task reinforcement learning and jointly trains a perception-gated dense mixture-of-experts actor with task-specific critics: soft routing composes expert knowledge into continuously varying behavior, while separated critics isolate heterogeneous objectives so that specialization survives joint optimization. A concurrent teacher-student formulation transfers this co
    
[^279]: 基于马尔可夫提升的Volterra生成模型

    Volterra Generative Models via Markovian Lifting

    [https://arxiv.org/abs/2606.18071](https://arxiv.org/abs/2606.18071)

    本文提出通过马尔可夫提升方法构建Volterra生成模型，将非马尔可夫的Volterra过程转化为可处理的马尔可夫近似，从而支持重用预训练去噪器，并在MNIST、CIFAR-10和ImageNet基准上取得了优异的FID性能。

    

    基于扩散的生成模型通常通过由布朗运动驱动的马尔可夫过程来扰动数据。Volterra过程允许更灵活的时间依赖结构，但其非马尔可夫特性使得得分学习和反向时间模拟变得复杂。我们开发了Volterra生成模型并提供了马尔可夫近似方法。借助高斯条件化，去噪问题被简化到原始数据维度。我们还阐明了该框架与方差爆炸和方差保持表示之间的联系。此外，我们的框架支持重用预训练的去噪器，例如EDM2网络。借助一个辅助因子，我们从头训练的模型在MNIST上取得了0.547的FID，在条件CIFAR-10上取得了1.860的FID。通过自动引导，ImageNet64在微调后FID达到1.912，而ImageNet512使用冻结的预训练网络FID达到1.903。

    arXiv:2606.18071v2 Announce Type: replace-cross  Abstract: Diffusion-based generative models usually perturb data through Markov processes driven by Brownian motion. Volterra processes allow more flexible temporal dependence, but their non-Markovian structure complicates score learning and reverse-time simulation. We develop Volterra generative models and provide Markovian approximations. Gaussian conditioning then reduces the denoising problem to the original data dimension. We present the connections with variance-exploding and variance-preserving representations. Our framework also permits reuse of pretrained denoisers such as EDM2 networks. With one auxiliary factor, our models trained from scratch achieve FIDs of $0.547$ on MNIST and $1.860$ on conditional CIFAR-10. With autoguidance, ImageNet64 FID reaches $1.912$ after fine-tuning, while ImageNet512 FID reaches $1.903$ using frozen pretrained networks.
    
[^280]: LatentGym：一个具有可控潜在结构的跨任务经验学习测试平台

    LatentGym: A Testbed For Cross-Task Experiential Learning With Controllable Latent Structure

    [https://arxiv.org/abs/2606.15306](https://arxiv.org/abs/2606.15306)

    提出了LatentGym测试平台，通过围绕真值潜在变量构建可控环境，实现了对智能体跨任务经验学习能力的度量，并可将探索与利用行为分开评估。

    

    我们设想持续学习的智能体系统能够随时间推移变得更有用：当它们遇到一系列相关任务时，应该推断出这些任务间共享的隐藏结构，并利用它来改进未来的决策。这种跨任务经验学习能力在个性化和交互式辅助等领域至关重要，但现有的训练/评估框架无法提供共享的、可控的潜在结构，也无法衡量智能体是否以及为何有所改进。我们提出了LatentGym：一个可控的测试套件，其中每个环境都围绕一个控制跨任务结构的真值潜在变量构建。我们的构造产生了能够区分探索（智能体的动作是否收集关于潜在变量的信息）与利用（智能体是否使用所收集的信息）的评估指标。我们通过实证研究展示了该套件，回答了三个问题：如何以及为什么……（摘要在此处被截断）

    arXiv:2606.15306v2 Announce Type: replace-cross  Abstract: We envision continually learning agentic systems that become more useful over time: as they encounter sequences of related tasks, they should infer the hidden structure shared across those tasks and use it to improve future decisions. This cross-task experiential learning capability is pivotal in domains such as personalization and interactive assistance, but existing training/evaluation frameworks do not provide shared, controllable latent structures and cannot measure whether or why agents improve. We introduce LatentGym: a controllable suite in which each environment is organized around a ground-truth latent variable governing the structure across tasks. Our construction yields metrics that separate exploration (whether the agent's actions gather information about the latent) from exploitation (whether the agent uses what it has gathered). We demonstrate our suite on empirical studies addressing three questions: how and why 
    
[^281]: StreamMemBench：面向未来辅助的智能体记忆流式评估基准

    StreamMemBench: Streaming Evaluation of Agent Memory for Future-Oriented Assistance

    [https://arxiv.org/abs/2606.14571](https://arxiv.org/abs/2606.14571)

    该论文提出StreamMemBench——首个基于EgoLife第一人称视频流的智能体记忆流式评估基准，通过两步任务序列和四个指标系统评估记忆的证据召回、证据使用、反馈整合与后续复用能力。

    

    arXiv:2606.14571v3 公告类型：替换 摘要：个人智能体记忆的核心作用是将存储的信息和先前交互转化为面向未来的辅助。在日常使用中，有用的线索来自智能体所观察到的内容以及用户与智能体交互的方式，智能体必须将这些线索从当前请求延续到类似的未来任务中。现有的记忆基准通常孤立地测试对话召回或任务改进，使得从流式观察到后续辅助这一完整轨迹在很大程度上未被评估。我们提出StreamMemBench，这是一个流式基准，它围绕来自EgoLife第一人称视角流的每个证据锚点构建两步任务序列：初始任务测试证据的使用，而后续任务测试反馈和交互经验是否被复用。四个指标分别诊断证据召回、初始证据使用、反馈整合和后续复用。在两种骨干模型上对八种记忆系统进行的实验表明

    arXiv:2606.14571v3 Announce Type: replace  Abstract: A central role of personal-agent memory is to turn stored information and prior interactions into future-oriented assistance. In daily use, useful cues come from what the agent observes and how the user interacts with the agent, and the agent must carry them forward from the current request to similar future tasks. Existing memory benchmarks usually test dialogue recall or task improvement in isolation, leaving the trajectory from streaming observations to later assistance largely untested. We introduce StreamMemBench, a streaming benchmark that constructs a two-step task sequence around each evidence anchor from EgoLife egocentric streams. The initial task tests evidence use, while the follow-up task tests whether feedback and interaction experience are reused. Four metrics diagnose evidence recall, initial evidence use, feedback incorporation, and follow-up reuse. Experiments with eight memory systems across two backbones show that
    
[^282]: 标准可解释模型：基于拉格朗日力学的可解释机器学习通用理论，用于演绎设计可解释方法

    The Standard Interpretable Model: A general theory of interpretable machine learning to deductively design interpretable methods using Lagrangian mechanics

    [https://arxiv.org/abs/2606.12289](https://arxiv.org/abs/2606.12289)

    本文提出了一种基于拉格朗日力学的标准可解释模型（SIM）理论，通过演绎方式设计可解释方法，填补了可解释性理论与方法之间的空白。

    

    arXiv:2606.12289v2 公告类型：替换交叉 摘要：随着人工智能模型复杂度的增加，可解释性已成为理解、调试和控制其计算不可或缺的工具。然而，可解释性缺乏通用理论来演绎设计可解释方法。这种理论与方法之间的差距导致文献碎片化和评估协议不一致。为填补这一空白，我们引入了标准可解释模型（SIM），这是一个基于拉格朗日力学的通用理论，能够实现可解释方法的演绎设计。具体而言，SIM在一组前提中总结了可解释性对目标用户的意义。从这些前提出发，SIM系统地推导出可解释性对称性和相应约束，这些约束塑造了一个拉格朗日量的景观，其最小值对应于最优可解释模型。为达到最小值，可以更新操作参数的数值。

    arXiv:2606.12289v2 Announce Type: replace-cross  Abstract: As Artificial Intelligence models grow in complexity, interpretability has become an indispensable tool for understanding, debugging, and controlling their computations. However, interpretability lacks general theories to deductively design interpretable methods. This gap between theories and methods results in a fragmented literature and inconsistent evaluation protocols.   To fill this gap, we introduce the Standard Interpretable Model (SIM), a general theory grounded in Lagrangian mechanics that enables the deductive design of interpretable methods. Specifically, the SIM summarises, in a set of premises, what interpretability is for a target user. From these premises, the SIM systematically derives interpretability symmetries and corresponding constraints, which shape the landscape of a Lagrangian whose minima correspond to optimal interpretable models. To reach the minima, one can either update the parameter values of an op
    
[^283]: 当归因修补失效时：诊断与二阶修正

    When Attribution Patching Lies: Diagnosis and a Second-Order Correction

    [https://arxiv.org/abs/2606.09899](https://arxiv.org/abs/2606.09899)

    本文揭示归因修补的主要误差源于下游网络的非线性而非被修补组件的局部曲率，并据此提出可靠性分数、误差界限与二阶修正三项工具来诊断和改进归因修补的可信度。

    

    机械可解释性的一个核心目标是识别哪些内部组件因果地驱动语言模型的行为。由于这些重要性估计是识别电路的证据基础，系统性的误差可能导致对底层机制的错误识别。虽然激活修补提供了黄金标准的因果度量，但其计算成本在大规模应用时过于高昂。因此，实践者转而依赖归因修补，这是一种基于梯度的一阶近似方法，但其可靠性仍缺乏充分理解。在这项工作中，我们刻画了这种不可靠性的来源，证明其主要误差源于下游网络中的非线性，而非被修补组件处的局部曲率。这一洞见带来了三个实用工具：（i）一个用于检测不可信估计的可靠性分数；（ii）量化潜在归因错误设定的误差界限；（（摘要此处截断，第三项工具应为标题中提到的二阶修正）

    arXiv:2606.09899v2 Announce Type: replace-cross  Abstract: A central goal of mechanistic interpretability is to identify which internal components causally drive a language model's behavior. Because these importance estimates serve as the evidence for identifying circuits, systematic errors can lead to the misidentification of the underlying mechanisms. While activation patching provides a gold-standard causal metric, its computational cost is prohibitive at scale. Practitioners instead rely on attribution patching, a gradient-based, first-order approximation whose reliability remains poorly understood. In this work, we characterize the source of this unreliability, demonstrating that the dominant error stems from the non-linearities in the downstream network rather than local curvature at the patched component. This insight yields three practical tools: (i) a reliability score to detect untrustworthy estimates, (ii) error bounds quantifying potential attribution mis-specifications, an
    
[^284]: KV缓存量化下的对齐崩塌：诊断与缓解

    Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation

    [https://arxiv.org/abs/2606.09864](https://arxiv.org/abs/2606.09864)

    研究发现KV缓存低比特量化会在困惑度等常规指标不变的情况下隐性地破坏大语言模型的安全对齐，其根源在于安全特征位于对量化噪声极其敏感的低维子空间，据此提出的PCR诊断方法可用于识别并缓解此类失效。

    

    键值缓存量化被广泛用于降低大语言模型推理时的内存开销，然而现有评估仅关注困惑度和准确率，而未评估其对安全性的影响。在本研究中，我们探索了KV缓存量化下的对齐保持问题。通过对十一个指令微调模型（3.8B-72B）和五个基准测试（共1,894个提示）的实验，我们发现低比特量化会悄无声息地破坏安全对齐：Mistral-7B在困惑度仅为1.03倍时便丧失了15.2%的拒绝行为，且不存在通用的安全比特宽度，存在标准指标无法察觉的、模型特异的急剧相变。我们查明其根本原因是几何性的：安全特征占据一个低维激活子空间，与困惑度所平均的完整表示空间相比，该子空间对量化噪声的敏感度高出10²至10³倍。我们提出了逐通道缩减（PCR）诊断方法，可将每个模型归类为三种机制性失效模式之一（摘要原文在此处截断）。

    arXiv:2606.09864v3 Announce Type: replace-cross  Abstract: Key-value (KV) cache quantization is widely used to reduce LLM inference memory, yet existing evaluations focus only on perplexity and accuracy without assessing the safety impact. In this study, we explore alignment preservation under KV cache quantization. Across eleven instruction-tuned models (3.8B-72B) and five benchmarks (1,894 prompts), we find that low-bit quantization can silently destroy safety alignment: Mistral-7B loses 15.2% of its refusals at only 1.03x perplexity, and no universal safe bit-width exists, with sharp model-specific phase transitions invisible to standard metrics. We identify that the root cause is geometric: safety features occupy a low-dimensional activation subspace $10^2$-$10^3\times$ more vulnerable to quantization noise than the full representation space perplexity averages over. We propose Per-Channel Reduction (PCR), a diagnostic that classifies each model into one of three mechanistic failur
    
[^285]: Hypnos：通过下一词元预测学习睡眠生理的可泛化表征

    Hypnos: Learning Generalisable Representations of Sleep Physiology via Next-Token Prediction

    [https://arxiv.org/abs/2606.09605](https://arxiv.org/abs/2606.09605)

    该论文提出Hypnos多模态睡眠基础模型，以下一词元预测作为掩码重建和对比学习的简单可扩展替代方案，从超过2万份多导睡眠图记录的八种传感模态中学习睡眠生理信号的可泛化表征。

    

    基础模型为将多模态生理信号压缩为紧凑的人类健康表征提供了一条富有前景的路径，在睡眠医学、心脏病学、神经病学及其他医疗健康领域具有广泛应用。现有模型通常采用掩码重建或对比学习目标进行训练。然而，掩码重建可能难以适应这些信号的随机特性，而对比学习方法依赖于正样本对的定义，尽管生理信号的语义不变性尚缺乏充分理解。在本工作中，我们证明了下一词元预测是一种简单且可扩展的替代方案。我们开发了Hypnos，一个多模态睡眠基础模型，利用来自超过20,000份整夜多导睡眠图记录的八种不同传感模态（如脑电图、心电图、呼吸信号）进行训练。我们使用残差向量量化将每种模态标记化为离散词元流……

    arXiv:2606.09605v2 Announce Type: replace  Abstract: Foundation models offer a promising route to compress multi-modal physiological signals into compact representations of human health, with broad applications across sleep medicine, cardiology, neurology and other healthcare domains. Existing models have typically been trained with masked-reconstruction or contrastive objectives. However, masked reconstruction may be poorly suited to the stochastic nature of these signals, while contrastive approaches rely on positive-pair definitions despite the semantic invariances of physiological signals being poorly understood. In this work, we show that next-token prediction is a simple and scalable alternative. We develop Hypnos, a multi-modal sleep foundation model trained using eight different sensing modalities (e.g. EEG, ECG, respiratory signals) drawn from over 20{,}000 overnight polysomnography recordings. We tokenize each modality into streams of discrete tokens using residual vector qua
    
[^286]: 与“敌人”编程：人类开发者能否检测出AI智能体的破坏行为？

    Coding with "Enemy": Can Human Developers Detect AI Agent Sabotage?

    [https://arxiv.org/abs/2606.05647](https://arxiv.org/abs/2606.05647)

    本研究首次大规模研究了人类监督在AI编程破坏行为中的作用，发现在无监控条件下高达94%的开发者（83/88）未能检测到AI智能体植入的破坏行为。

    

    AI编程智能体正日益融入真实的软件开发环境，在与人类开发者协作的同时，获得了对代码库和工具更广泛的访问权限。这带来了新的攻击面：智能体可以利用人类的信任来破坏开发过程，例如通过插入恶意代码来完成隐藏的副任务。以往的大多数工作仅在纯AI环境中研究AI破坏行为，对人类监督在检测和缓解此类恶意行为中的作用关注有限。为填补这一空白，我们开展了首个关于AI编程破坏行为中人类监督的大规模研究。100多名参与者与四个前沿模型之一（Claude-Opus-4.6、GPT-5.4、Gemini-3.1-Pro和MiniMax-M2.7）协作，完成一项旨在模拟真实工作流程、历时约五小时的长周期编程任务。研究发现，在无监控条件下，88名开发者中有83名（94%）未能检测到破坏行为，并且我们对参与者的分……

    arXiv:2606.05647v2 Announce Type: replace  Abstract: AI coding agents are increasingly embedded in real-world software development, collaborating with human developers while gaining broader access to codebases and tools. This creates a new attack surface: an agent can exploit human trust to sabotage development, for instance by inserting malicious code to accomplish a hidden side task. Most prior work studies AI sabotage in AI-only settings, paying limited attention to the role of human oversight in detecting and mitigating such malicious behavior. To address this gap, we conduct the first large-scale study of human oversight in AI coding sabotage. Over 100 participants collaborate with one of four frontier models (Claude-Opus-4.6, GPT-5.4, Gemini-3.1-Pro, and MiniMax-M2.7) on a long-horizon coding task lasting around five hours, designed to mimic real-world workflows. We find that 83/88 (94%) of developers in the no-monitor conditions fail to detect sabotage, and our analysis of parti
    
[^287]: TSQAgent：通过专用智能体推理评估时间序列数据质量

    TSQAgent: Rating Time Series Data Quality via Dedicated Agentic Reasoning

    [https://arxiv.org/abs/2606.03629](https://arxiv.org/abs/2606.03629)

    本文构建了TSQBench基准，揭示当前大语言模型在时间序列质量维度识别与有据可依的质量比较上的不足，并据此提出专用智能体推理框架TSQAgent，以实现更可靠的时间序列数据质量评估。

    

    评估时间序列（TS）数据的质量是一项基础性工作，但由于质量维度的多面性，这本身极具挑战性。近来，大语言模型（LLM）已成为时间序列质量评估的一种有前景的范式，其方法包括成对比较和按维度评估。然而，现有方法依赖于人工预定义的质量维度和纯文本推理，尚不清楚大语言模型能否识别真正相关的质量维度，或进行有据可依的定量质量比较。为探究这一问题，我们构建了TSQBench，这是一个专门用于评估大语言模型两项递进能力的基准：(i) 理解并识别相关的质量维度，(ii) 在特定维度下进行质量比较。我们的分析表明，当前的大语言模型在维度识别和基于证据的质量比较两方面均持续存在困难。为解决这些问题……

    arXiv:2606.03629v2 Announce Type: replace  Abstract: Assessing the quality of time series (TS) data is fundamental yet inherently challenging due to the multifaceted nature of quality dimensions. Recently, large language models (LLMs) have emerged as a promising paradigm for TS quality assessment via pairwise comparison and per-dimension evaluation. However, existing approaches rely on manually predefined quality dimensions and purely text-based reasoning, leaving it unknown whether LLMs can identify truly relevant quality dimensions or perform grounded and quantitative quality comparisons. To investigate this, we construct TSQBench, a dedicated benchmark for evaluating LLMs on two progressive capabilities: (i) understanding and identifying relevant quality dimensions, and (ii) performing quality comparison under specific dimensions. Our analysis reveals that current LLMs consistently struggle with both dimension identification and evidence-grounded quality comparison. To address these
    
[^288]: 未采样的真相：量化语言模型心理测量学中的提示伪影

    The Unsampled Truth: Quantifying Prompt Artifacts in LM Psychometrics

    [https://arxiv.org/abs/2606.03357](https://arxiv.org/abs/2606.03357)

    该论文提出一种诊断框架，通过将人设与四个提示组件的语义等价变体交叉组合并测量响应分布间的Wasserstein距离，成功将心理测量中的目标效应与提示伪影分离，并发现任务指令和选项符号对响应分布的干扰比人设或题项措辞的改变更大。

    

    当通过提示让语言模型进行心理测量评估时，研究者假设模型的响应反映了所注入的人设以及调查题项本身的含义。我们通过一种诊断性设计来检验这一前提：将五个语义上不同的人设与四个提示组件（人设措辞、任务指令、题项措辞、选项符号）各自的五个语义等价变体进行交叉组合。通过测量所得响应分布之间的1-Wasserstein距离，并在五个组件之间对变异进行划分，可以将目标效应与提示伪影分离开来。我们将该框架应用于13个开源权重的小型语言模型（参数量0.6B至14B），在大五人格量表和短版黑暗三联征量表上进行测试。我们发现，在大多数模型中，任务指令和选项符号对响应分布造成的偏移，大于对人设描述或题项本身进行同义改写所造成的影响。对于相当一部分（摘要在此处被截断）

    arXiv:2606.03357v2 Announce Type: replace-cross  Abstract: When prompting language models for psychometric assessment, researchers assume that the responses reflect the injected persona and the meaning of the survey item. We test this premise using a diagnostic design that crosses five semantically distinct baseline personas with five semantically equivalent variants of each of four prompt components (persona wording, task instruction, item wording, option symbol). Measuring the 1-Wasserstein distance between the resulting response distributions and partitioning the variation among the five components allows for the separation of target effects from prompt artifacts. We apply the framework to 13 open-weight small language models (0.6B to 14B) on the Big Five Inventory and the Short Dark Triad. We find that in most models, the task instruction and option symbol displace response distributions further than paraphrasing the persona description or the item itself. For a substantial share o
    
[^289]: Murmur：基于分层分块的高效长上下文自动语音识别推理

    Murmur: Efficient Long-Context ASR Inference with Hierarchical Chunking

    [https://arxiv.org/abs/2606.01483](https://arxiv.org/abs/2606.01483)

    提出了基于分层分块的长篇ASR推理系统Murmur，发现约五分钟的中间块大小可在避免重复循环的同时保持单次处理的准确率，并通过并行解码兼顾高准确率与低推理延迟。

    

    长篇自动语音识别（ASR）既需要高准确率又需要低推理延迟，但现有系统难以在两者之间取得良好平衡。基于分块的方法通过并行窗口处理音频来获得低延迟，但会丢失跨块上下文，并且需要脆弱的启发式方法在边界处对齐说话人和时间戳。长上下文ASR模型通过单次处理转录全部内容以获得更好的准确率，但运行速度慢数倍，且单个重复循环就可能破坏整个输出。在本文中，我们首先分析了上下文长度如何影响长上下文ASR模型的准确率，发现约五分钟的中间块大小可以在避免重复循环的同时达到与单次处理相当的准确率。基于这一发现，我们提出了Murmur，一个基于分层分块的长篇ASR推理系统。长上下文模型并行解码中间目标块，并由一个快速的（摘要在此处被截断）

    arXiv:2606.01483v2 Announce Type: replace-cross  Abstract: Long-form automatic speech recognition (ASR) requires both high accuracy and low inference latency, but existing systems fall short of handling the trade-off between the two. Chunk-based approaches process audio in parallel windows for low latency, but lose cross-chunk context and need brittle heuristics to align speakers and timestamps at boundaries. Long-context ASR models transcribe everything in a single pass for better accuracy, but run several times slower, and a single repetition loop can corrupt the entire output. In this paper, we first analyze how context length affects the accuracy of long-context ASR models and find that an intermediate chunk size of about five minutes matches single-pass accuracy while avoiding repetition loops. Building on this, we propose Murmur, an inference system for long-form ASR based on hierarchical chunking. The long-context model decodes intermediate target chunks in parallel, and a fast 
    
[^290]: 框架更新不等于框架收益：解耦自进化大语言模型智能体中的进化能力

    Harness Updating Is Not Harness Benefit: Disentangling Evolution Capabilities in Self-Evolving LLM Agents

    [https://arxiv.org/abs/2605.30621](https://arxiv.org/abs/2605.30621)

    该研究将自进化大语言模型智能体的能力解耦为“框架更新”与“框架收益”两种，并发现模型产生有用框架更新的能力与其基础能力层级无关，不同能力的模型带来的更新收益惊人地相似，即擅长更新框架并不意味着能真正从中受益。

    

    大语言模型（LLM）智能体越来越多地被部署为围绕可编辑外部框架构建的系统，这些框架包括提示词、技能、记忆和工具，能够在不改变模型参数的情况下塑造任务执行方式。框架自进化通过从执行证据中更新这些框架来适配此类智能体。然而，模型在任务求解上的基础能力是否能预测其框架自进化能力，目前仍不清楚：哪些模型能够产生有用的框架更新，哪些模型又真正从中受益？我们分析了两种框架自进化能力：(i) 框架更新能力，即从执行证据中产生有用的持久性框架更新的能力；(ii) 框架收益能力，即在任务求解过程中从已更新的框架中受益的能力。我们的分析揭示了两个发现。首先，框架更新能力在不同基础能力层级之间是平坦的：来自不同能力层级的模型所产生的框架更新带来了惊人的相似收益……

    arXiv:2605.30621v2 Announce Type: replace  Abstract: LLM agents are increasingly deployed as systems built around editable external harnesses, including prompts, skills, memories and tools, that shape task execution without changing model parameters. Harness self-evolution adapts such agents by updating these harnesses from execution evidence. Yet it remains unclear whether a model's base capability in task-solving predicts its capabilities in harness self-evolution: which models produce useful harness updates, and which actually benefit from them? We analyze two harness self-evolution capabilities: (i) harness-updating, the capability to produce useful persistent harness updates from execution evidence; (ii) harness-benefit, the capability to benefit from updated harnesses during task solving. Our analysis reveals two findings. First, harness-updating is flat in base capability: models from different capability tiers produce harness updates that lead to surprisingly similar gains; eve
    
[^291]: 小而可信：面向时间序列异常检测的高效视觉-语言推理

    Tiny but Trusted: Efficient Vision-Language Reasoning for Time-Series Anomaly Detection

    [https://arxiv.org/abs/2605.30344](https://arxiv.org/abs/2605.30344)

    提出高质量异常解释增强的基准VisAnomBench和参数高效的视觉-语言模型VisAnomReasoner，实现了更准确、可解释的时间序列异常检测。

    

    视觉-语言模型（VLM）的最新进展在众多任务上取得了令人瞩目的表现，然而先前的研究表明，将大型语言模型或多模态模型应用于发现序列数据中的异常模式时，其表现并不理想。公开的异常检测基准通常只提供区间标注而缺乏自然语言的理由说明，这使得难以对VLM进行微调以产生有依据、可解释的决策。为填补这一空白，我们构建了VisAnomBench——一个基于公开时间序列数据集精心构建的基准，并通过使用细粒度、任务特定的奖励机制从多个大型VLM中筛选出的高质量异常解释对其进行增强。通过在该基准上进行微调，我们开发了VisAnomReasoner，一个用于时间序列异常检测的参数高效VLM。在VisAnomBench上的结果表明，VisAnomReasoner实现了更准确的异常定位，并持续优于所有基线方法。

    arXiv:2605.30344v2 Announce Type: replace  Abstract: Recent advances in Vision--Language Models (VLMs) have achieved impressive performance across many tasks, yet prior studies report unsatisfactory performance when applying large language or multimodal models to finding abnormal patterns in sequential data. Public anomaly detection benchmarks typically provide interval annotations but not natural-language rationales, making it difficult to fine-tune VLMs to produce grounded, interpretable decisions. To address this gap, we construct VisAnomBench, a curated benchmark built from public time-series datasets and augmented with high-quality anomaly explanations selected from multiple large VLMs using fine-grained, task-specific rewards. Through fine-tuning on this benchmark, we develop VisAnomReasoner, a parameter-efficient VLM for time-series anomaly detection. Results on VisAnomBench show that VisAnomReasoner achieves more accurate anomaly localization and consistently outperforms all ba
    
[^292]: 按其思考方式施教：用稀疏自编码器引导大语言模型后训练数据工程

    Teach the Way It Thinks: Guiding LLM Post-training Data Engineering with Sparse Autoencoders

    [https://arxiv.org/abs/2605.27354](https://arxiv.org/abs/2605.27354)

    该论文提出SAERL方法，利用稀疏自编码器（SAE）将模型内部分解为稀疏可解释的特征，从模型内部视角（激活与梯度投影）来组织强化学习训练数据，从而决定训练哪些样本、何时呈现以及如何分批，突破了传统仅依赖通过率等外部反馈的数据工程方式。

    

    大语言模型（LLM）强化学习（RL）的数据工程决定了训练哪些样本、何时呈现这些样本，以及哪些样本应一起分批。现有方法大多依赖外部反馈，例如通过率或难度标签，这只能观察到每个样本上的结果。我们认为，模型内部提供了更丰富、更贴近模型自身的反馈，能够指导上述全部三项决策。我们提出SAERL，它在稀疏自编码器（SAE）的特征空间中组织强化学习训练数据。SAE是一种机制可解释性工具，可将模型内部分解为稀疏且可解释的特征。借助SAE特征，我们可以将模型视为一组功能区域，每个区域由相关样本共同激活的特征所构成。对于每个样本，其SAE激活表明它参与了哪些区域，而其投影到SAE特征上的梯度则表明对它进行训练将如何改变这些区域。在此指导下……（原文摘要至此被截断）

    arXiv:2605.27354v2 Announce Type: replace-cross  Abstract: Data engineering for large language model (LLM) reinforcement learning (RL) decides which samples to train on, when to present them, and which to batch together. Existing methods mostly rely on external feedback, such as pass rates or difficulty labels, which observes only the outcome on each sample. We argue that model internals offer richer and more model-intrinsic feedback that can guide all three decisions. We propose SAERL which organizes RL training data in the feature space of Sparse Autoencoders (SAEs), a mechanistic interpretability tool that decomposes model internals into sparse, interpretable features. SAE features let us view the model as a set of functional regions, each formed by features that related samples activate together. For each sample, its SAE activations indicate which regions it engages, and its gradients projected onto SAE features indicate how training on it would change these regions. Guided by this
    
[^293]: 万物于每一尺度：具备连续超分辨率的尺度不变扩散模型

    Everything at Every Scale: Scale-Invariant Diffusion with Continuous Super-Resolution

    [https://arxiv.org/abs/2605.26032](https://arxiv.org/abs/2605.26032)

    提出尺度不变的扩散模型 SKILD，将图像生成与连续超分辨率统一在单一无条件框架中，仅通过改变起始时间步即可在两种任务间切换。

    

    从噪声中创建图像是图像生成；从粗糙输入中重建精细细节是超分辨率。尽管二者在实际应用中有所不同，但都可以理解为对跨尺度信息损失的逆转。我们提出了 SKILD——一种尺度不变的 K 空间图像学习扩散模型，它在单一的无条件框架内统一了图像生成与连续超分辨率。自然图像与临界物理系统均表现出尺度不变性，我们利用这一特性设计了一个前向过程，该过程将图像内容从细尺度到粗尺度逐步衰减，同时注入与频谱匹配的高斯噪声，从而使尺度成为扩散动力学的一个显式坐标。同一个经过训练的逆向过程仅需改变起始时间步即可执行生成或连续超分辨率：无需任务特定的架构，无需条件分支……

    arXiv:2605.26032v2 Announce Type: replace-cross  Abstract: Creating images from noise is image generation; reconstructing fine details from coarse inputs is super-resolution. Despite their practical differences, both can be understood as reversing information loss across scales. We introduce $\textbf{SKILD}$, a $\textbf{S}$cale-invariant $\textbf{K}$-Space $\textbf{I}$mage $\textbf{L}$earning $\textbf{D}$iffusion model that unifies generation and continuous super-resolution within a single unconditional framework. Both natural images and critical physical systems exhibit scale invariance, and we leverage it to design a forward process that attenuates image content from fine to coarse scales while injecting spectrum-matched Gaussian noise, making scale an explicit coordinate of the diffusion dynamics. The same trained reverse process performs generation and continuous super-resolution by varying only the starting timestep: $\textit{no task-specific architecture, no conditioning branch, 
    
[^294]: JudgmentBench：比较量规评估与偏好评估的质量评估方法

    JudgmentBench: Comparing Rubric and Preference Evaluation for Quality Assessment

    [https://arxiv.org/abs/2605.25240](https://arxiv.org/abs/2605.25240)

    论文发布JudgmentBench基准，汇集资深律师对真实法律任务给出的量规评分与成对偏好判断数据，首次实证比较发现比较判断在恢复输出质量排序方面显著优于基于量规的评分。

    

    两种方法论主导着当前的基准测试实践：基于量规的评分根据预定义的标准来评估项目，而比较判断则在输出之间引出成对偏好。尽管这两种方法论都被广泛使用，但它们之间的选择却很少得到合理的论证。我们发布了JudgmentBench，这是一个包含30个真实世界法律任务的基准，配有1,539个量规评分和1,530个成对偏好判断，这些数据由具有丰富经验的执业律师收集——其中包括来自美国主要律师事务所的律师。这些标注构成了高专业领域内首个公开可用的数据集，其中两种监督信号均由同一批专家在相同任务上引出。使用三个构建的不同质量级别的LLM生成输出，我们提供了初步的实证比较：在恢复预期质量排序方面，比较判断显著优于量规……

    arXiv:2605.25240v4 Announce Type: replace-cross  Abstract: Two methodologies dominate current practices of benchmarking: rubric-based scoring evaluates items against predefined criteria, whereas comparative judgment elicits pairwise preferences between outputs. Although both methodologies are widely used, the choice between them is rarely justified. We release JudgmentBench, a benchmark of 30 real-world legal tasks, paired with 1,539 rubric scores and 1,530 pairwise preference judgments collected from practicing attorneys--including at major U.S. law firms--with substantial experience. The annotations constitute the first publicly available dataset in a high-expertise domain in which both supervision signals are elicited from the same pool of experts on the same tasks. Using LLM-generated outputs at three constructed quality levels, we provide an initial empirical comparison: comparative judgments recover the intended quality ordering substantially better than rubrics under both a per-
    
[^295]: LAPLEX：可学习拉普拉斯核的快速傅里叶变换

    LAPLEX: The FFT of Learnable Laplace Kernels

    [https://arxiv.org/abs/2605.24584](https://arxiv.org/abs/2605.24584)

    LAPLEX提出了一类精确且可训练的拉普拉斯核算子层，以FFT式的扩展效率实现了可学习的自适应几何结构，支持高达10^9维度的可训练矩阵运算，并可高效用于高维协方差建模。

    

    深度学习中的快速线性代数通常面临一种权衡选择：要么采用固定几何结构与精确计算（如傅里叶变换），要么为获得自适应几何结构而付出密集参数、随机特征或低秩代理的代价。为了突破这一权衡，我们提出了LAPLEX，一类精确且可训练的（相位化）拉普拉斯核算子。LAPLEX层是一个通常为满秩的稠密矩阵，由可学习的坐标锚点隐式定义，并具有类似FFT的扩展性。因此，它能够在现代GPU上支持向量维度高达10^9的可训练矩阵-向量运算。作为神经层，它可以产生紧凑的投影层和分类头，这些分类头可解释为软性的、可训练的路由模型。同一基元还可用作高效的Gram算子，使得在维度为3×10^6的展平图像上构建高维协方差模型成为可能，既能保留可见的空间结构，又无需施加卷积偏置。

    arXiv:2605.24584v2 Announce Type: replace-cross  Abstract: Fast linear algebra in deep learning usually comes with a choice: fixed geometry and exact computation, as in the Fourier transform, or adaptive geometry paid for by dense parameters, random features, or low-rank surrogates. To move beyond this trade-off, we introduce LAPLEX, a class of exact, trainable (phased) Laplace-kernel operators.   A LAPLEX layer is a typically full-rank dense matrix, implicitly defined by learnable coordinate anchors, with FFT-like scaling. Consequently, it supports trainable matrix--vector operations at vector dimensions up to $10^9$ on modern GPUs. As a neural layer, it yields compact projections and classification heads interpretable as soft, trainable routing models. The same primitive also serves as an efficient Gram operator, enabling high-dimensional covariance models on flattened images of dimension $3 \cdot 10^6$ that preserve visible spatial structure without imposing convolutional bias. Thes
    
[^296]: ActQuant：面向视觉-语言-动作模型的亚4比特动作引导主干网络仅权重量化

    ActQuant: Sub-4-bit Action-Guided Backbone Weight-Only Quantization for Vision-Language-Action Models

    [https://arxiv.org/abs/2605.24011](https://arxiv.org/abs/2605.24011)

    ActQuant提出了一种动作引导的混合精度训练后量化框架，通过张量间比特分配和张量内尺度优化两阶段策略实现VLA模型的亚4比特仅权重量化，并配套OmniModel.cpp转换流水线支持设备端部署。

    

    视觉-语言-动作（VLA）模型在具身智能中展现出卓越的动作生成能力，但其庞大的计算量使其难以在边缘平台上部署。激进的亚4比特权重量化是自然而然的解决方案，然而现有的训练后量化（PTQ）方法在这种极端比特设置下会出现严重的性能下降。为解决这一问题，我们提出了ActQuant，一个动作引导的混合精度PTQ框架，它分两个阶段运行：(1) 张量间比特分配器，根据每个权重矩阵对预测智能体动作的贡献程度，为其分配单一的比特宽度；(2) 张量内尺度优化器，利用动作感知的曲率来调整逐块量化尺度，从而使动态范围集中在控制中最具影响力的权重上。为了充分发挥这种激进量化在设备端的优势，我们进一步提出了OmniModel.cpp，一个智能体化的模型转换流水线，可将……

    arXiv:2605.24011v4 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models exhibit remarkable action generation for embodied intelligence, but their heavy compute make deployment on edge platforms impractical. Aggressive, sub-4-bit weight quantization is the natural solution, yet existing post-training quantization (PTQ) methods suffer severe performance degradation in this regime. To address this, we introduce ActQuant, an action-guided mixed-precision PTQ framework that operates in two stages: (1) an inter-tensor bit allocator that assigns each weight matrix a single bit-width based on how much it contributes to predicting the agent's actions; (2) an intra-tensor scale optimizer tunes per-block quantization scales using action-aware curvature, so that dynamic range is concentrated on the weights most influential for control. To deliver the on-device benefits of our aggressive quantization, we further introduce OmniModel.cpp, an agentic conversion pipeline that por
    
[^297]: VACE：为时间序列异常检测学习几何结构化表示

    VACE: Learning Geometrically Structured Representations for Time Series Anomaly Detection

    [https://arxiv.org/abs/2605.23504](https://arxiv.org/abs/2605.23504)

    VACE提出了一种自监督异常检测方法，通过速度对齐通道嵌入显式塑造潜空间几何结构，将正常性表示为局部平滑、方向连贯的表示，克服了对比学习无法显式控制距离评分所依赖几何结构的缺陷。

    

    多变量时间序列中的异常检测是众多现实应用中的一项关键任务，在这些场景中异常行为稀少、标签不可用，且漏检的代价高昂。核心挑战在于学习一种足够精确的正常性刻画，以便标记偏差。表示自监督学习（通常采用对比方法）通过将时间片段嵌入潜空间来解决这一问题，在该空间中正常性占据一个明确定义的区域，并通过几何偏差来检测异常。然而，对比方法通过成对采样启发式间接地塑造这一空间，无法对基于距离评分所依赖的几何结构进行显式控制，因此这种几何结构只能听凭运气。我们提出了VACE（速度对齐通道嵌入），这是一种自监督异常检测方法，它将正常性表示为局部平滑、方向连贯的……

    arXiv:2605.23504v2 Announce Type: replace-cross  Abstract: Anomaly detection in multivariate time series is a critical task across a wide range of real-world applications, where abnormal behaviour is rare, labels are unavailable, and the cost of a miss is high. The central challenge is learning a characterisation of normality precise enough to flag deviations. Representation self-supervised learning, typically through contrastive approaches, addresses this by embedding temporal patches into a latent space where normality occupies a well-defined region, with anomalies detected by geometric deviation. However, contrastive approaches shape this space indirectly through pair-sampling heuristics, providing no explicit control over the geometric structure on which distance-based scoring operates. That geometry is thus left to chance. We present VACE (Velocity-Aligned Channel Embeddings), a self-supervised anomaly detection method that represents normality as a locally smooth, directionally c
    
[^298]: 向量策略优化：面向多样性的训练改进测试时搜索

    Vector Policy Optimization: Training for Diversity Improves Test-Time Search

    [https://arxiv.org/abs/2605.22817](https://arxiv.org/abs/2605.22817)

    提出向量策略优化（VPO），通过利用向量值奖励、多答案提示与集合级奖励来训练大语言模型生成多样化的解决方案集合，从而提升推理时搜索的效果。

    

    语言模型现在必须能够开箱即用地泛化到新颖环境中，并能在推理扩展搜索过程（如AlphaEvolve）中工作，这类搜索过程使用多种任务特定的奖励函数来选择 rollout。遗憾的是，标准的 LLM 后训练范式优化的是预先指定的标量奖励，这往往导致当前的 LLM 产生低熵的响应分布，从而难以展现推理时搜索所需的多样性。我们提出了向量策略优化（VPO），这是一种强化学习算法，它显式地训练策略以预期多样化的下游奖励函数并产生多样化的解决方案。VPO 利用了实践中奖励往往是向量值的事实，例如代码生成中每个测试用例的正确性，或者多个不同的用户角色或奖励模型。VPO 将多答案提示与集合级奖励相结合，训练 LLM 生成能够……

    arXiv:2605.22817v2 Announce Type: replace-cross  Abstract: Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO combines multi-answer prompting with a set-level reward to train the LLM to produce sets of solutions that 
    
[^299]: TerminalWorld：在真实世界终端任务上对智能体进行基准测试

    TerminalWorld: Benchmarking Agents on Real-World Terminal Tasks

    [https://arxiv.org/abs/2605.22535](https://arxiv.org/abs/2605.22535)

    本文提出TerminalWorld——一个能从真实终端记录中自动逆向生成高保真评估任务的可扩展数据引擎，构建出包含1,530个验证任务的大规模终端智能体基准，并揭示当前前沿模型在真实终端工作流上的最高通过率仅为62.5%，且其表现与现有专家策划基准仅呈弱相关。

    

    我们推出了TerminalWorld，这是一个可扩展的数据引擎，能够从“真实场景中”的终端记录自动逆向工程出高保真的评估任务。该引擎处理了80,870条终端记录，产出了包含1,530个经过验证任务的完整基准，涵盖18个真实世界类别，从简短的日常操作到超过50个步骤的工作流程，覆盖1,280个独特命令。在此基础上，我们精心挑选出一个包含200个多样化、经人工审查任务的Verified（已验证）子集。在TerminalWorld-Verified上对八个前沿模型和六个智能体进行的全面基准测试表明，当前系统在真实终端工作流程方面仍然存在困难，最高通过率仅为62.5%。此外，TerminalWorld所捕捉的真实世界终端能力与现有专家策划的基准（如Terminal-Bench）有所不同，与其分数仅有微弱的相关性（Pearson r=0.20）。这一自动化引擎使TerminalWorld能够持续扩展。

    arXiv:2605.22535v2 Announce Type: replace  Abstract: We introduce TerminalWorld, a scalable data engine that automatically reverse-engineers high-fidelity evaluation tasks from "in-the-wild" terminal recordings. Processing 80,870 terminal recordings, the engine yields a full benchmark of 1,530 validated tasks, spanning 18 real-world categories, ranging from short everyday operations to workflows exceeding 50 steps, and covering 1,280 unique commands. From these, we curate a Verified subset of 200 diverse, manually reviewed tasks. Comprehensive benchmarking on TerminalWorld-Verified across eight frontier models and six agents reveals that current systems still struggle with authentic terminal workflows, achieving a maximum pass rate of only 62.5%. Moreover, TerminalWorld captures real-world terminal capabilities distinct from existing expert-curated benchmarks (e.g., Terminal-Bench), with only a weak correlation to their scores (Pearson r=0.20). The automated engine makes TerminalWorld 
    
[^300]: KVBuffer：面向线性注意力的IO感知服务系统

    KVBuffer: IO-aware Serving for Linear Attention

    [https://arxiv.org/abs/2605.19049](https://arxiv.org/abs/2605.19049)

    KVBuffer是一种面向线性注意力的IO感知服务机制，通过缓冲最近的键和值实现分块计算与批量状态更新，显著减少内存访问量并降低解码延迟，同时高效支持投机解码验证。

    

    线性注意力近来因其在长上下文推理中相对于上下文长度具有恒定的解码成本而受到广泛关注。然而，现有的服务系统通常通过在每个解码步骤中递归地计算和更新一个庞大的线性注意力状态来提供线性注意力服务。由于该状态远大于每个token的键和值，递归解码会导致大量的内存访问，使得线性注意力的服务效率低下。在本文中，我们提出了KVBuffer，一种面向线性注意力的IO感知服务机制。通过缓冲最近的键和值，KVBuffer使服务系统能够以更灵活、更节省内存的方式计算线性注意力输出。对于解码，KVBuffer支持分块计算，通过延迟状态更新并将其批量应用，降低了平均内存访问量和解码延迟。对于投机解码，KVBuffer通过缓存机制高效验证草稿token

    arXiv:2605.19049v2 Announce Type: replace-cross  Abstract: Linear attention has recently gained significant attention for long-context inference due to its constant decoding cost with respect to context length. However, existing serving systems typically serve linear attention by recurrently computing and updating a large linear attention state in every decoding step. Since the state is much larger than the per-token key and value, recurrent decoding incurs substantial memory access and becomes inefficient for serving linear attention. In this paper, we propose KVBuffer, an IO-aware serving mechanism for linear attention. By buffering recent keys and values, KVBuffer enables serving systems to compute linear attention outputs in more flexible and memory-efficient ways. For decoding, KVBuffer enables chunkwise computation, which reduces average memory access and decoding latency by deferring state updates and applying them in batch. For speculative decoding, KVBuffer verifies draft toke
    
[^301]: 软标签学习与校准中人类不确定性与模型不确定性的评估

    An Assessment of Human vs. Model Uncertainty in Soft-Label Learning and Calibration

    [https://arxiv.org/abs/2605.18648](https://arxiv.org/abs/2605.18648)

    本研究通过解耦软标签监督与标签模态偏移，发现人类软标签的主要价值在于作为正则化器提升模型对困难样本的校准能力和训练收敛稳定性，并使模型的不确定性表现与人类一致。

    

    以人为本的对齐AI的核心在于理解人类提供的标签相较于合成标签的优势。虽然人类软标签通过捕捉不确定性能够改善模型校准，但先前的研究将这些优势与对错误标注数据的隐式纠正（模态偏移）混为一谈，从而掩盖了软标签的真正效果。我们对MNIST及其合成变体上的软标签学习进行了受控审计，通过重新标注数据子集来提取人类不确定性。通过将软标签监督与底层标签模态偏移解耦，我们证明人类软标签确实能带来准确率提升，但其更大的价值在于作为一种正则化器，能够改善模型对困难样本的校准，并促进训练过程中的稳定收敛。数据集制图分析表明，使用人类软标签训练的模型能够反映人类的不确定性，而使用合成标签训练的模型则无法与人类对齐。总体而言，这项工作……（原文截断）

    arXiv:2605.18648v3 Announce Type: replace-cross  Abstract: Central to human-aligned AI is understanding the benefits of human-elicited labels over synthetic alternatives. While human soft-labels improve calibration by capturing uncertainty, prior studies conflate these benefits with the implicit correction of mislabeled data (mode shifts), obscuring true effects of soft-labels. We present a controlled audit of soft-label learning across MNIST and a synthetic variant, re-annotating subsets to extract human uncertainty. By decoupling soft-label supervision from underlying label mode shifts, we show that while human soft-labels do provide accuracy gains, their larger value lies in acting as a regularizer that improves model calibration on difficult samples and promotes stable convergence across training runs. Dataset cartography reveals models trained on human soft-labels mirror human uncertainty, whereas those trained on synthetic labels fail to align with humans. Broadly, this work prov
    
[^302]: 面向视觉-语言-动作策略的事件锚定稀疏自编码器

    Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies

    [https://arxiv.org/abs/2605.17204](https://arxiv.org/abs/2605.17204)

    该论文提出一种以闭环行为事件为锚点的SAE可解释性流水线，通过从末端执行器轨迹提取关键帧并聚类来分析VLA策略的内部表征，是首批将基于SAE的VLA分析建立在机器人闭环行为事件之上的工作之一。

    

    视觉-语言-动作（VLA）策略将语言和视觉输入转化为机器人动作，其隐藏表征直接塑造了闭环行为。然而，来自语言模型和视觉-语言模型的机制可解释性工具并不能顺利迁移到VLA：其输出是机器人动作而非人类可读的词元，且验证内部变化需要进行代价高昂的闭环回放。我们提出了一种以事件为锚点的可解释性流水线，将SAE特征分析锚定于行为事件而非文本语境。我们从末端执行器轨迹中提取关键帧，并利用视觉、机器人状态和时间线索在每个任务内对其进行聚类。由此得到的事件用于指导SAE特征排序，并可通过可选的VLM标注为事件提供描述。据我们所知，该流水线是首批将基于SAE的VLA分析建立在闭环行为事件之上的工作之一。在两个仿真……

    arXiv:2605.17204v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) policies translate language and visual inputs into robot actions, where their hidden representations directly shape closed-loop behavior. However, mechanistic interpretability tools from language and vision-language models do not transfer cleanly to VLAs: outputs are robot actions rather than human-readable tokens, and validating internal changes requires expensive closed-loop rollouts. We propose an event-grounded interpretability pipeline that anchors SAE feature analysis to behavioral events rather than text contexts. We extract keyframes from end-effector trajectories and cluster them within each task using visual, robot-state, and temporal cues. The resulting events guide SAE feature ranking, with optional VLM annotations providing descriptions of the events. To our knowledge, our pipeline is among the first to ground SAE-based VLA analysis in closed-loop behavioral events. Across two simulatio
    
[^303]: LinAlg-Bench：揭示大语言模型数学推理中结构性失败模式的取证式基准测试

    LinAlg-Bench: A Forensic Benchmark Revealing Structural Failure Modes in LLM Mathematical Reasoning

    [https://arxiv.org/abs/2605.16675](https://arxiv.org/abs/2605.16675)

    该论文提出LinAlg-Bench取证式诊断基准，通过对10个前沿大语言模型在660个线性代数问题上的6,600个输出进行系统性错误分类，发现模型数学失败受算法类型和矩阵维度结构性约束，且在4x4规模存在急剧行为阈值——低于该规模表现为执行错误，高于该规模则转变为计算放弃与编造响应。

    

    我们提出了LinAlg-Bench，这是一个诊断性基准，评估10个前沿大语言模型在结构化线性代数计算上的表现，涵盖3x3、4x4和5x5矩阵的严格维度梯度。该基准包含9种任务类型和660个经SymPy认证的问题，共详尽评估了6,600个模型输出。除了二元准确性之外，LinAlg-Bench还引入了一个三阶段自动化取证流程，将1,156次失败分类为十个主要错误标签及其细粒度子类型，揭示了大语言模型的数学失败并非随机，而是受算法类型和矩阵维度的结构性约束。我们的核心发现是在4x4规模处存在一个急剧的行为阈值：低于该规模时，模型通过执行错误而失败——符号追踪失败、算术漂移和奇偶性错误；高于该规模时，失败转变为计算放弃，模型通过工具角色扮演、约束一致的虚构等方式编造响应……

    arXiv:2605.16675v2 Announce Type: replace  Abstract: We introduce LinAlg-Bench, a diagnostic benchmark evaluating 10 frontier large language models on structured linear algebra computation across a strict dimensional gradient of 3x3, 4x4, and 5x5 matrices. Spanning 9 task types and 660 SymPy-certified problems, the benchmark exhaustively evaluates 6,600 model outputs. Beyond binary accuracy, LinAlg-Bench introduces a three-stage automated forensic pipeline classifying 1,156 failures into ten primary error tags with fine-grained subtypes, revealing that LLM mathematical failure is not random but structurally constrained by algorithm type and matrix dimension. Our central finding is a sharp behavioral threshold at 4x4 scale: below it, models fail through execution errors -- sign tracking failures, arithmetic drift, and parity errors; above it, failure transitions to computational abandonment, with models fabricating responses through tool roleplay, constraint-consistent confabulation, an
    
[^304]: PanoWorld：几何一致的全景视频世界建模

    PanoWorld: Geometry-Consistent Panoramic Video World Modeling

    [https://arxiv.org/abs/2605.15391](https://arxiv.org/abs/2605.15391)

    PanoWorld 将全景视频生成重构为几何与动力学一致的潜在状态建模问题，通过深度一致性和轨迹一致性两个轻量级正则化项，从单张图像生成几何一致的 360° 全景视频。

    

    我们提出了 PanoWorld，一个能够从单张图像和文本描述生成几何一致的 360° 视频的全景视频世界模型。现有的全景视频方法主要针对视觉真实感进行优化，并未显式约束底层的 3D 场景状态，导致生成的结果虽然看起来合理，但在球面上却存在深度不一致、对应关系断裂以及不合理的运动等问题。为解决这一缺陷，我们将全景视频生成构建为一个几何与动力学一致的潜在状态建模问题，而非纯粹的视觉合成任务。基于一个预训练的透视视频世界模型，我们引入了两个轻量级的正则化项：一个是针对伪真值全景深度的深度一致性损失，另一个是监督被跟踪点在 3D 世界坐标系中随时间变化位置的轨迹一致性损失。我们进一步对条件注入机制应用了球面几何感知的适配

    arXiv:2605.15391v2 Announce Type: replace-cross  Abstract: We present PanoWorld, a panoramic video world model that generates geometry-consistent 360$\degree$ video from a single image and a caption. Existing panoramic video methods optimize primarily for visual realism and do not explicitly constrain the underlying 3D scene state, producing outputs that appear plausible yet exhibit inconsistent depth, broken correspondences, and implausible motion across the spherical surface. We address this gap by framing panoramic video generation as a geometry- and dynamics-consistent latent state modeling problem rather than pure visual synthesis. Building on a pre-trained perspective video world model, we introduce two lightweight regularizers: a depth consistency loss against pseudo ground-truth panoramic depth, and a trajectory consistency loss that supervises the 3D world-frame positions of tracked points across time. We further apply spherical-geometry-aware adaptation to the conditioning an
    
[^305]: 面向离散不确定性两阶段鲁棒优化的学习式场景缩减方法

    Learning Scenario Reduction for Two-Stage Robust Optimization with Discrete Uncertainty

    [https://arxiv.org/abs/2605.14494](https://arxiv.org/abs/2605.14494)

    本文提出了问题驱动的场景缩减启发式算法 PRISE 及其基于 GNN-Transformer 的神经代理模型 NeurPRISE，能够高效构建高质量场景子集，显著提升离散不确定性两阶段鲁棒优化的可扩展求解能力。

    

    具有离散不确定性的两阶段鲁棒优化（2RO）极具挑战性，往往导致精确求解的计算代价难以承受。场景缩减通过选择一个小而具有代表性的场景子集来缓解这一问题，从而使计算变得可行。然而，现有方法在很大程度上与具体问题无关，仅在不考虑可行域或追索结构的情况下对不确定性集合进行操作。本文提出了 PRISE，一种问题驱动的顺序前瞻启发式算法，通过评估每个场景的边际影响来构建缩减后的场景集合。尽管 PRISE 能够产生高质量的场景子集，但其每个选择步骤都需要求解多个子问题，使得该算法在大规模问题上的计算开销十分高昂。为解决这一问题，我们提出了 NeurPRISE，一个基于 GNN-Transformer 骨干网络的神经代理模型，它通过图卷积对单个场景的结构进行编码，并捕获跨场景之间的交互关系。

    arXiv:2605.14494v2 Announce Type: replace  Abstract: Two-Stage Robust Optimization (2RO) with discrete uncertainty is challenging, often rendering exact solutions prohibitive. Scenario reduction alleviates this issue by selecting a small, representative subset of scenarios to enable tractable computation. However, existing methods are largely problem-agnostic, operating solely on the uncertainty set without consulting the feasible region or recourse structure. In this paper, we introduce PRISE, a problem-driven sequential lookahead heuristic that constructs reduced scenario sets by evaluating the marginal impact of each scenario. While PRISE yields high-quality scenario subsets, each selection step requires solving multiple subproblems, making it computationally expensive at scale. To address this, we propose NeurPRISE, a neural surrogate model built on a GNN-Transformer backbone that encodes the per-scenario structure via graph convolution and captures cross-scenario interactions thro
    
[^306]: EcoGEO：面向网络增强型大语言模型搜索代理的轨迹感知证据生态系统

    EcoGEO: Trajectory-Aware Evidence Ecosystems for Web-Enabled LLM Search Agents

    [https://arxiv.org/abs/2605.12887](https://arxiv.org/abs/2605.12887)

    本文提出EcoGEO框架，将生成式引擎优化（GEO）从单网页层面扩展为面向网络增强型LLM搜索智能体的环境级影响力问题，并设计了轨迹感知的协同证据生态系统TRACE，通过控制智能体浏览轨迹中证据的组织、连接与呈现方式来影响其搜索结果。

    

    arXiv:2605.12887v3 公告类型：replace-cross。摘要：具备网络访问能力的大语言模型智能体正在改变在线信息影响搜索结果的方式。现有的生成式引擎优化（GEO）研究主要聚焦于单个网页。然而，智能体化的网络搜索并非单文档场景：智能体可能会发出查询、抓取页面、跟踪链接、重新表述搜索，并在多个浏览步骤中综合证据。因此，影响力不仅取决于页面内容，还取决于页面如何沿着智能体的浏览轨迹被组织、连接和呈现。我们通过生态系统生成式引擎优化（EcoGEO）来研究这一转变，该方法将GEO视为面向网络增强型大语言模型智能体的环境级影响力问题。为了具体实现这一视角，我们提出了TRACE——一个轨迹感知的协同证据生态系统。给定一个推荐查询和一个虚构的目标产品，我们的方法构建了一个受控的证据环境，该环境协调智能体辅助的……（原文摘要在此处截断）

    arXiv:2605.12887v3 Announce Type: replace-cross  Abstract: Web-enabled LLM agents are changing how online information influences search outcomes. Existing Generative Engine Optimization (GEO) studies mainly focus on individual webpages. However, agentic web search is not a single-document setting: an agent may issue queries, crawl pages, follow links, reformulate searches, and synthesize evidence across multiple browsing steps. Influence therefore depends not only on page content, but also on how pages are organized, connected, and encountered along the agent's browsing trajectory. We study this shift through Ecosystem Generative Engine Optimization (EcoGEO), which treats GEO as an environment-level influence problem for web-enabled LLM agents. To instantiate this perspective, we propose TRACE, a Trajectory-Aware Coordinated Evidence Ecosystem. Given a recommendation query and a fictional target product, our method builds a controlled evidence environment that coordinates an agent-faci
    
[^307]: 涌现性失调中的人格模型坍缩

    Persona-Model Collapse in Emergent Misalignment

    [https://arxiv.org/abs/2605.12850](https://arxiv.org/abs/2605.12850)

    该论文提出“人格模型坍缩”假说，指出有害数据微调引发的涌现性失调源于模型内部人格表征的退化，并在四个模型家族上通过问卷实验证实有害微调使人格表达更极化、更二元化且内部一致性降低。

    

    在包含有害内容的狭窄数据上对大型语言模型进行微调，会使模型在与该数据无关的提示上也表现出广泛的对齐失调，这种现象被称为“涌现性失调”。我们提出，涌现性失调涉及“人格模型坍缩”：即编码和区分人格的内部表征发生退化，从而破坏了人格表达的细腻性与一致性。我们首先从行为层面检验这一假设：比较在有害数据上微调的模型与良性对照模型在扮演多样化人格时对问卷的回答。在四个模型家族中，有害微调产生了更极化、“非全即无”的回答，表明人格表达变得更加二元化、细腻程度更低。直方图显示回答在量表两端更为集中，并通过库尔贝克—莱布勒（KL）散度对这种分布偏移进行了量化。我们还发现人格内部一致性有所降低：对同一问题的重复回答……（摘要原文在此处截断）

    arXiv:2605.12850v3 Announce Type: replace-cross  Abstract: Fine-tuning large language models on narrow data with harmful content produces broadly misaligned behavior on unrelated prompts, a phenomenon known as emergent misalignment. We propose that emergent misalignment involves persona-model collapse: a deterioration of the internal representations that encode and distinguish personas, undermining the nuance and consistency of their expression. We first test this hypothesis behaviorally, by comparing how models fine-tuned on harmful data and benign controls answer questionnaires while role-playing diverse personas. Across four model families, harmful fine-tuning produces more polarized, all-or-nothing answers, suggesting more binary, less nuanced persona expression. Histograms show increased concentration at the scale endpoints, and Kullback--Leibler (KL) divergence quantifies the distributional shift. We also find reduced within-persona consistency: repeated answers to the same quest
    
[^308]: ELF：嵌入式语言流

    ELF: Embedded Language Flows

    [https://arxiv.org/abs/2605.10938](https://arxiv.org/abs/2605.10938)

    提出在连续嵌入空间中基于流匹配的扩散语言模型 ELF，其直到最终时间步才映射到离散词元，从而让图像扩散领域的成熟技术（如 CFG）可直接应用于语言建模。

    

    扩散模型和基于流的模型已成为生成连续数据（如图像和视频领域）的事实标准方法。它们的成功激发了人们将其应用于语言建模的日益浓厚的兴趣。与图像领域的对应模型不同，当今主流的扩散语言模型（DLM）主要在离散词元上运行。在本文中，我们证明了连续扩散语言模型只需对离散域进行极少适配即可变得有效。我们提出了嵌入式语言流，这是一类在连续嵌入空间中基于连续时间流匹配的扩散模型。与现有的扩散语言模型不同，ELF 在绝大部分生成过程中保持在连续嵌入空间中，直到最后一个时间步才通过一个共享权重网络映射到离散词元。这种设计使得图像领域扩散模型中成熟的技术（例如无分类器引导 CFG）能够直接移植到语言建模中。

    arXiv:2605.10938v3 Announce Type: replace-cross  Abstract: Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain. We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG). Experim
    
[^309]: 单策略集中性下具有前向KL正则化的离线上下文老虎机的快速收敛率

    Fast Rates for Offline Contextual Bandits with Forward-KL Regularization under Single-Policy Concentrability

    [https://arxiv.org/abs/2605.09214](https://arxiv.org/abs/2605.09214)

    本文首次在单策略集中性条件下，为前向KL正则化的离线上下文老虎机给出了表格型与一般函数逼近设置下的 Õ(ε^{-1}) 快速收敛率上界，通过新颖地运用悲观原理的凸分析框架填补了前向KL正则化统计分析的空白。

    

    KL（Kullback-Leibler）散度正则化以反向KL或前向KL的形式广泛存在于强化学习算法中。最近的研究表明，在反向KL正则化下的决策问题可以达到 ε^{-1} 型的快速收敛率，相比之下标准的样本复杂度为 ε^{-2} 型。然而，对于前向KL正则化的目标函数，现有的统计分析要么不适用，要么只能得到 Õ(ε^{-2}) 的慢收敛率。我们通过对前向KL正则化离线上下文老虎机的精简分析，迈出了解决这一问题的第一步。我们在表格型设置和一般函数逼近设置下，均在单策略集中性的概念下，首次给出了 Õ(ε^{-1}) 的上界。特别地，我们的凸分析流程以一种新颖的方式利用悲观原理统一了这些设置，并完全绕过了证明中的……（原文摘要在此处截断）

    arXiv:2605.09214v2 Announce Type: replace-cross  Abstract: \emph{Kullback-Leibler} (KL) regularization is ubiquitous in reinforcement learning algorithms in the form of \emph{reverse} or \emph{forward} KL. Recent studies have demonstrated $\epsilon^{-1}$-type fast rates for decision making under reverse KL regularization, in contrast to the standard $\epsilon^{-2}$-type sample complexity. However, for forward-KL-regularized objectives, existing statistical analyses are either not applicable or result in $\tilde{O}(\epsilon^{-2})$ slow rates. We take the first step towards addressing this problem via a streamlined analysis of forward-KL-regularized offline CBs. We give the first $\tilde{O}(\epsilon^{-1})$ upper bounds in tabular and general function approximation settings, both under notions of \emph{single-policy concentrability}. In particular, our convex-analytical pipeline unifies these settings by exploiting the pessimism principle in a novel way and completely bypasses the proof r
    
[^310]: FILO：低秩自适应的统一Fisher几何框架

    FILO: A Unified Fisher-Geometric Framework for Low-Rank Adaptation

    [https://arxiv.org/abs/2605.08734](https://arxiv.org/abs/2605.08734)

    本文提出FILO，一个统一的Fisher几何框架，无需对奇异算子直接求逆即可实现LoRA的因子空间更新，并据此将现有LoRA优化器系统地划分为四大类。

    

    低秩自适应（LoRA）将权重矩阵 $W$ 重参数化为两个低秩因子的乘积，但将因子映射到权重矩阵的生成器的雅可比矩阵 $J_G$ 是秩亏的，因此由任何 $W$ 空间预条件子 $F_t$ 诱导的因子空间预条件子 $J_G^* F_t J_G$ 是奇异的，进而导致标准链式法则无法被唯一地求逆，无法将预条件后的 $W$ 空间方向映射回因子空间的更新。为此，我们提出了 FILO，一个用于低秩自适应的统一 Fisher 几何框架，它无需直接对该奇异算子求逆即可实现因子更新，并根据因子空间求逆策略和 $W$ 空间预条件子 $F_t$ 对现有 LoRA 优化器进行分类。现有方法沿着这两个维度划分为四大类：因子空间自适应更新、$J_G^* J_G$ 的块对角替代方法、Frobenius 残差伪逆方法以及黎曼流形方法。

    arXiv:2605.08734v2 Announce Type: replace-cross  Abstract: Low-Rank Adaptation (LoRA) reparameterizes a weight matrix $W$ as a product of two low-rank factors, but the Jacobian $J_G$ of the generator mapping the factors to the weight matrix is rank-deficient, so the factor-space preconditioner $J_G^* F_t J_G$ induced by any $W$-space preconditioner $F_t$ is singular, and consequently the standard chain rule cannot be uniquely inverted to map a preconditioned $W$-space direction back to a factor-space update. Therefore, we introduce FILO, a unified Fisher-geometric framework for low-rank adaptation that enables factor updates without directly inverting this singular operator and casts existing LoRA optimizers by their factor-space inversion strategy and $W$-space preconditioner $F_t$. Existing methods occupy four families along these axes: factor-space adaptive updates, block-diagonal surrogates for $J_G^* J_G$, Frobenius-residual pseudoinverse methods, and Riemannian manifold methods. 
    
[^311]: OmicsLM：一个用于多样本组学推理的多模态大语言模型

    OmicsLM: A Multimodal Large Language Model for Multi-Sample Omics Reasoning

    [https://arxiv.org/abs/2605.06728](https://arxiv.org/abs/2605.06728)

    提出了OmicsLM，一个90亿参数的多模态大语言模型，将bulk和单细胞转录组表示为紧凑的连续嵌入并与自然语言指令及其他样本交错输入，配合连接实验测量、生物学知识与任务监督的训练语料库，实现了多样本组学推理。

    

    大语言模型能够对生物学知识进行推理，但它们并不天然具备处理定量组学测量数据的能力。现有方法要么将表达谱转换为冗长的文本表示，这往往会丢失定量信息，要么将组学编码器连接到语言模型以完成较为狭窄的任务。我们提出了OmicsLM，这是一个90亿参数的多模态语言模型，它将每个bulk（混样）或单细胞转录组表示为紧凑的连续嵌入，这些嵌入可以与自然语言指令以及同一上下文中的其他样本交错组合。一个重要贡献是构建了一个将实验测量数据与生物学知识和任务监督相连接的训练语料库。我们结合了将转录组谱与注释及实验结果相关联的模板、跨来源连接证据的推理示例，以及经过任务特定质量检查的由大语言模型生成的监督数据。（摘要原文在此处截断）

    arXiv:2605.06728v2 Announce Type: replace-cross  Abstract: Large language models can reason over biological knowledge, but they are not naturally equipped to work with quantitative omics measurements. Existing approaches either convert expression profiles into long textual representations, often losing quantitative information, or connect omics encoders to language models for narrow tasks. We introduce OmicsLM, a 9B multimodal language model that represents each bulk or single-cell transcriptome as a compact continuous embedding that can be interleaved with natural-language instructions and with other samples in the same context. An key contribution is a training corpus that connects experimental measurements with biological knowledge and task supervision. We combine templates linking transcriptomic profiles to annotations and experimental outcomes, reasoning examples connecting evidence across sources, and LLM-generated supervision with task-specific quality checks. The instruction-tu
    
[^312]: UniSD：面向大语言模型的统一自蒸馏框架

    UniSD: Towards a Unified Self-Distillation Framework for Large Language Models

    [https://arxiv.org/abs/2605.06597](https://arxiv.org/abs/2605.06597)

    提出了统一自蒸馏框架UniSD，通过整合多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪等互补机制，系统性地解决了大语言模型自蒸馏中监督可靠性、表示对齐与训练稳定性的核心难题。

    

    自蒸馏（SD）为在不依赖更强外部教师模型的情况下适配大语言模型（LLM）提供了一条有前景的路径。然而，自回归LLM中的自蒸馏仍然充满挑战，因为自生成的轨迹是自由形式的，正确性依赖于具体任务，且看似合理的推理过程仍可能提供不稳定或不可靠的监督信号。现有方法主要考察孤立的设计选择，导致其有效性、作用及相互关系尚不明确。本文提出了UniSD，一个用于系统研究自蒸馏的统一框架。UniSD整合了解决监督可靠性、表示对齐和训练稳定性问题的互补机制，包括多教师一致性、EMA教师稳定化、词元级对比学习、特征匹配和散度裁剪。在来自三个模型系列的六个基准测试和六个模型上，UniSD揭示了自蒸馏何时有效（摘要在此处被截断）。

    arXiv:2605.06597v3 Announce Type: replace-cross  Abstract: Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation impr
    
[^313]: ProCompNav：面向模糊用户查询的基于比较判断的主动实例导航

    ProCompNav: Proactive Instance Navigation with Comparative Judgment for Ambiguous User Queries

    [https://arxiv.org/abs/2605.06223](https://arxiv.org/abs/2605.06223)

    提出ProCompNav两阶段框架，利用递归比较判断（RCJ）主动询问区分性信息，逐步缩小候选池，从而在模糊查询场景下准确识别目标实例，避免将强匹配的干扰项误认为目标。

    

    当用户的初始请求无法唯一确定目标实例时，自然语言实例导航就变得具有挑战性。一个实用的智能体应该通过主动询问仅需区分目标与相似干扰项的信息来减轻用户负担，而不是要求用户预先提供详细的描述。然而，现有方法往往无法达到这一目标，它们会误将那些与用户所提供的关于目标的累积信息高度匹配的干扰项当作目标。其结果是，即使经过了对话交互，智能体仍可能无法将目标与干扰项区分开，从而导致过早做出决策，并需要用户给出冗长的回复。我们提出了基于比较判断的主动实例导航方法ProCompNav，这是一个两阶段框架：首先构建候选池，然后通过递归比较判断（RCJ）来识别目标实例。RCJ通过选择属性-值对迭代地缩小候选池……

    arXiv:2605.06223v4 Announce Type: replace  Abstract: Natural-language instance navigation becomes challenging when the initial user request does not uniquely specify the target instance. A practical agent should reduce the user's burden by actively asking only the information needed to distinguish the target from similar distractors, rather than requiring a detailed description upfront. Existing approaches often fall short of this goal by mistaking distractors that strongly match the accumulated information about the target provided by the user. As a result, despite the dialogue, the agent may still fail to distinguish the target from distractors, leading to premature decisions and lengthy user responses. We propose Proactive Instance Navigation with Comparative Judgment (ProCompNav), a two-stage framework that first constructs a candidate pool and then identifies the target through Recursive Comparative Judgment (RCJ). RCJ iteratively narrows the pool by selecting an attribute-value p
    
[^314]: 驯服熵悬崖：面向自回归视觉生成的可变码本大小量化

    Taming the Entropy Cliff: Variable Codebook Size Quantization for Autoregressive Visual Generation

    [https://arxiv.org/abs/2605.06207](https://arxiv.org/abs/2605.06207)

    该论文发现离散视觉分词器中存在条件熵沿序列急剧衰减的“熵悬崖”现象，并提出可变码本大小量化（VCQ）方法，以解决恒定码本快速耗尽有限数据熵预算的问题。

    

    大多数离散视觉分词器依赖于一个默认设计：序列中的每个位置共享同一个码本。这种恒定码本设计会非常快速地消耗有限数据的熵预算。我们观察到，每个位置的经验条件熵沿着序列衰减得如此之快，以至于在几个位置之后，条件分布基本上变为确定性分布。在ImageNet上（K=16384）时，这种情况在256个位置中仅约2个位置之后就会发生，此时大多数前缀已经能够识别数据集中的单个图像。我们将这种现象称为“熵悬崖”（Entropy Cliff）。在条件被充分利用的情况下，较大的码本会在更少的位置耗尽有限数据的熵预算。在我们的语言对比实验中，经验熵的下降则更为平缓，每个位置的有效熵保持在码本容量以下。为解决这一问题，我们提出了可变码本大小量化（VCQ），其中码本……

    arXiv:2605.06207v2 Announce Type: replace-cross  Abstract: Most discrete visual tokenizers rely on a default design: every position in the sequence shares the same codebook. Such a constant-codebook design can consume the finite-data entropy budget very quickly. We observe that the per-position empirical conditional entropy decays so quickly along the sequence that, after a few positions, the conditional distribution becomes essentially deterministic. On ImageNet with $K=16384$, this happens after only about 2 out of 256 positions, when most prefixes already identify individual images in the dataset. We call this phenomenon the Entropy Cliff. Under full conditional utilization, larger codebooks exhaust the finite-data entropy budget in fewer positions. The empirical entropy decreases more gradually in our language comparison, where the effective entropy per position stays below the codebook capacity. To address this, we propose Variable Codebook Size Quantization (VCQ), where the codeb
    
[^315]: LLM安全评判者是否具有策略不变性？一项三原则压力测试

    Are LLM Safety Judges Policy-Invariant? A Three-Principle Stress-Test

    [https://arxiv.org/abs/2605.06161](https://arxiv.org/abs/2605.06161)

    本文提出“策略不变性”这一安全评判者必须满足的基本属性，并将其操作化为三条可检验原则构成压力测试协议，揭示了现有LLM安全评判者的判定会同时受有意义规范变化和无意义措辞改写影响这一此前未被测量的失效模式。

    

    LLM-as-a-Judge（以大语言模型作为评判者）流水线已成为智能体安全评估的事实标准，然而现有基准测试将评判者的判定结果视为真实标准的代理，却从未检验这些判定究竟取决于智能体的行为，还是仅仅取决于评估策略恰好采用的措辞方式。我们认为，任何可信的安全评判者都必须满足一个我们称之为“策略不变性”的基本属性，并将其操作化为三条可检验的原则：在经认证的等价改写下评分标准的语义不变性；在有意进行的由严格到宽松的阈值偏移下评分标准的阈值不变性；以及歧义感知校准，使得判定结果的不稳定性集中于真正存在歧义的案例。我们将这些原则实例化为一套压力测试协议，使用四个智能体级评判者在来自ASSEBench和R-Judge的轨迹上进行测试，揭示了一种此前未被测量过的失效模式：当今的评判者对有意义的规范变化与无意义的结构性改写……（摘要原文在此处截断）

    arXiv:2605.06161v2 Announce Type: replace  Abstract: LLM-as-a-Judge pipelines have become the de facto evaluator for agent safety, yet existing benchmarks treat their verdicts as ground-truth proxies without checking whether the verdicts depend on the agent's behavior or merely on how the evaluation policy happens to be worded. We argue that any trustworthy safety judge must satisfy a basic property we call policy invariance, and we operationalize it as three testable principles: rubric-semantics invariance under certified-equivalent rewrites, rubric-threshold invariance under intentional strict-to-lenient shifts, and ambiguity-aware calibration so that verdict instability concentrates on genuinely ambiguous cases. Instantiating these principles as a stress-test protocol with four agent-class judges on trajectories drawn from ASSEBench and R-Judge, we surface a previously unmeasured failure mode: today's judges respond to meaningful normative shifts and to meaningless structural rewrit
    
[^316]: 基于强化学习的网络控制器的最坏情况发现与运行时保护

    Worst-Case Discovery and Runtime Protection for RL-Based Network Controllers

    [https://arxiv.org/abs/2605.04373](https://arxiv.org/abs/2605.04373)

    提出了ReGuard框架，通过双层遗憾最大化问题发现强化学习网络控制器的最坏情况场景，并将反事实轨迹编译为轻量级逻辑规则，在推理时无需重新训练即可保护控制器。

    

    基于强化学习（RL）的控制器在拥塞控制和自适应比特率流媒体等网络任务中能够实现优异的平均情况性能。然而，在某些网络条件下，其性能可能严重退化，而这些条件本是可以取得良好性能的。通过枚举方式识别这类条件并量化由此产生的性能差距是不可行的，同时RL控制器的顺序性和闭环特性又使得形式化验证方法难以实际应用。我们提出了ReGuard，这是一个针对给定RL控制器发现最坏情况场景、并在推理时对其进行保护的框架，且无需重新训练。ReGuard将场景发现形式化为一个双层遗憾最大化问题，为控制器的最坏情况性能差距认证出一个下界。随后，ReGuard分析所发现的反事实轨迹，并将其编译为轻量级逻辑规则，仅在检测到危险状态时进行干预。

    arXiv:2605.04373v2 Announce Type: replace-cross  Abstract: RL-based controllers achieve strong average-case performance in networking tasks such as congestion control and adaptive bitrate streaming. Yet their performance can degrade severely under network conditions where strong performance is still achievable. Identifying such conditions and quantifying the resulting performance gap is intractable by enumeration, while the sequential and closed-loop nature of RL controllers makes formal verification methods impractical.   We present ReGuard, a framework that discovers worst-case scenarios for a given RL controller and protects it against them at inference time without retraining. ReGuard formulates scenario discovery as a bilevel regret-maximization problem, certifying a lower bound on the controller's worst-case performance gap. ReGuard then analyzes the discovered counterfactual trajectories and compiles them into lightweight logic rules that intervene only when a risky state is det
    
[^317]: 从多个思考者那里学习思考：思维链异质性如何影响高效可学习性

    Learning to Think from Multiple Thinkers: How Chain-of-Thought Heterogeneity Affects Efficient Learnability

    [https://arxiv.org/abs/2604.24737](https://arxiv.org/abs/2604.24737)

    该论文首次从学习理论视角研究异构思维链监督问题，提出多种由不同来源（最终答案相同但中间步骤各异）提供CoT数据的异构监督模型，并探讨在单一同构来源可被高效学习的前提下，何时能够从异构监督数据中实现高效学习。

    

    思维链监督通过暴露易于通过下一词预测学习的中间步骤，使计算上困难的学习问题变得可解。然而在实践中，CoT数据集来自具有不同推理风格的异构来源，并且需要投入大量精力来策划适合下一词预测训练的监督数据。这种数据策划在使CoT监督的计算优势得以实现方面扮演着什么角色？在这项工作中，我们从学习理论的角度迈出了理解这一问题的第一步。我们引入了来自多个来源的异构CoT监督的几种模型，其中每个来源都提供具有相同最终答案但中间步骤不同的正确推理轨迹。假设将下一词预测器拟合到来自任何单一来源的同构轨迹在计算上是可处理的，我们探讨这在何种条件下足以支持从异构监督中进行高效学习。

    arXiv:2604.24737v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) supervision makes computationally hard learning problems tractable by exposing intermediate steps that are easy to learn via next-token prediction. However, in practice, CoT datasets draw on heterogeneous sources with different reasoning styles, and substantial effort goes into curating supervision suitable for next-token training. What role does this curation play in enabling the computational advantage of CoT supervision? In this work, we take a first step toward understanding this question from a learning-theoretic perspective. We introduce several models of $heterogeneous$ $CoT$ $supervision$ from multiple sources, each providing correct traces with the same final answer but different intermediate steps. Assuming that fitting a next-token predictor to $homogeneous$ traces from any single source is computationally tractable, we ask when this suffices for efficient learning from heterogeneous supervisio
    
[^318]: AAC：面向ALT的架构级可采纳可微分地标压缩

    AAC: Admissible-by-Architecture Differentiable Landmark Compression for ALT

    [https://arxiv.org/abs/2604.20744](https://arxiv.org/abs/2604.20744)

    本文提出AAC，在固定标签内存预算下通过两种可采纳性保持的连续松弛实现可微分的ALT地标选择压缩，并给出了带有1-1/e极限因子的近似保证。

    

    我们提出了架构可采纳压缩器（AAC），它在固定的标签内存预算下，使基于地标的A*启发式方法能够适应查询需求。AAC为ALT（A*、地标与三角不等式）选择提供了两种连续松弛形式。在地标距离精确的前提下，每一个参数取值都定义了最短路径距离的一个下界。两种松弛形式都仅使用被选中的标签来部署ALT。行随机松弛形式采用地标距离的凸组合。我们刻画了保持可采纳性的最大线性类，并证明了松弛最优解与离散最优解相一致。分数松弛形式混合了子集启发式，并支持带有精确梯度的连续贪心算法（AAC-CG），以及用于可微分选择的熵正则化预言机。对于期望启发式值，我们证明了包含有限步数项、预言机误差项和有限样本项的近似保证。极限情形下的 $1-1/e$ 因子（摘要原文在此截断）

    arXiv:2604.20744v2 Announce Type: replace  Abstract: We introduce the Architecturally Admissible Compressor (AAC), which adapts landmark-based A* heuristics to query demand under a fixed label-memory budget. AAC provides two continuous relaxations of ALT (A*, Landmarks, and Triangle inequality) selection. With exact landmark distances, every parameter value defines a lower bound on shortest-path distance. Both relaxations deploy ALT using only the selected labels. The row-stochastic relaxation takes convex combinations of landmark distances. We characterize the maximal admissibility-preserving linear classes and prove that the relaxed and discrete optima coincide. The fractional relaxation mixes subset heuristics and supports continuous greedy (AAC-CG) with exact gradients and an entropy-regularized oracle for differentiable selection. For expected heuristic value, we prove an approximation guarantee with finite-step, oracle-error, and finite-sample terms. The limiting $1-1/e$ factor i
    
[^319]: 流形上的推理：扩散语言模型中用于自我验证的双向一致性

    Reasoning on the Manifold: Bidirectional Consistency for Self-Verification in Diffusion Language Models

    [https://arxiv.org/abs/2604.16565](https://arxiv.org/abs/2604.16565)

    该论文提出了一种无需训练、无监督的双向流形一致性（BMC）度量方法，通过前向掩码与后向重建循环量化生成序列的几何稳定性，从而在无标准答案的情况下验证扩散语言模型推理的有效性，并支持诊断与拒绝重采样等应用。

    

    尽管扩散大语言模型（dLLMs）在全局规划方面具有结构优势，但如何高效地验证它们是否通过有效的推理轨迹得出正确答案，仍然是一个关键挑战。在这项工作中，我们提出了一个几何视角：流形上的推理。我们假设有效的生成轨迹作为稳定吸引子驻留在学习分布的高密度流形上，而无效路径则表现出偏离流形的漂移。为了将这一假设付诸实践，我们引入了双向流形一致性（BMC），这是一种无需训练、无监督的度量方法，通过前向掩码与后向重建的循环来量化生成序列的稳定性。在实证方面，我们展示了BMC在整个推理生命周期中的多功能性：（1）在诊断中，它可以在没有标准答案的情况下作为解有效性的稳健判别器；（2）在推理中，它支持拒绝重采样机制，从而提升生成质量。

    arXiv:2604.16565v4 Announce Type: replace-cross  Abstract: While Diffusion Large Language Models (dLLMs) offer structural advantages for global planning, efficiently verifying that they arrive at correct answers via valid reasoning traces remains a critical challenge. In this work, we propose a geometric perspective: Reasoning on the Manifold. We hypothesize that valid generation trajectories reside as stable attractors on the high-density manifold of the learned distribution, whereas invalid paths exhibit off-manifold drift. To operationalize this, we introduce Bidirectional Manifold Consistency (BMC), a training-free, unsupervised metric that quantifies the stability of the generated sequence through a forward-masking and backward-reconstruction cycle. Empirically, we demonstrate BMC's versatility across the full reasoning lifecycle: (1) in Diagnosis, it serves as a robust discriminator of solution validity without ground truth answer; (2) in Inference, it enables rejection resamplin
    
[^320]: 循环Transformer中的稳定性与泛化

    Stability and Generalization in Looped Transformers

    [https://arxiv.org/abs/2604.15259](https://arxiv.org/abs/2604.15259)

    该论文提出基于不动点的三维度稳定性分析框架，证明带外部归一化召回机制的循环Transformer能同时实现不动点的局部吸引性、输入平滑性与稳定的反向传播，从而在测试时对更难的问题实现外推而非记忆。

    

    循环Transformer通过在更难的问题上投入更多迭代次数，有望实现测试时的计算扩展，但目前仍不清楚哪些架构选择能让它们在测试时外推到更难的问题，而非死记硬背训练中特定的解法。我们提出了一个基于不动点的分析框架，从局部吸引性、输入依赖性和几何结构这三个稳定性维度来分析循环架构，并用它刻画不动点迭代何时能产生有意义的预测。在理论上，我们证明了不带召回机制的循环网络具有可数个不动点，且在任何谱范围内都无法实现强输入依赖性；而带外部归一化的召回机制则能可靠地产生一种状态，使得不动点同时具备局部吸引性、对输入的局部平滑性，并获得稳定反向传播的支持。在实证方面，我们在国际象棋、数独和前缀和任务上训练了小型循环Transformer（摘要在此处截断）。

    arXiv:2604.15259v3 Announce Type: replace-cross  Abstract: Looped transformers promise test-time compute scaling by spending more iterations on harder problems, but it remains unclear which architectural choices let them extrapolate to harder problems at test time rather than memorize training-specific solutions. We introduce a fixed-point based framework for analyzing looped architectures along three axes of stability -- local attractivity, input-dependence, and geometry -- and use it to characterize when fixed-point iteration yields meaningful predictions. Theoretically, we prove that looped networks without recall have countable fixed points and cannot achieve strong input-dependence at any spectral regime, while recall with outer normalization reliably produces a regime in which fixed points are simultaneously locally-attractive, locally smooth in the input, and supported by stable backpropagation. Empirically, we train small looped transformers on chess, sudoku, and prefix-sums an
    
[^321]: MIND：材料研究的AI科研合作者

    MIND: AI Co-Scientist for Material Research

    [https://arxiv.org/abs/2604.13699](https://arxiv.org/abs/2604.13699)

    MIND是一个由LLM驱动的多智能体框架，通过集成机器学习原子间势实现材料研究假设的自动化实验验证，突破了以往AI科学发现仅限于文本推理的局限。

    

    大型语言模型（LLM）已经催生了面向科学发现的智能体AI系统，但大多数方法仍局限于基于文本的推理，缺乏自动化的实验验证。我们提出了MIND，一个由LLM驱动的材料研究自动假设验证框架。MIND将科学发现过程组织为多智能体流水线中的假设精炼、实验执行和基于辩论的验证。在实验验证方面，该系统集成了机器学习原子间势（特别是SevenNet-Omni），从而实现可扩展的计算机模拟实验。我们还提供了一个基于Web的用户界面用于自动化假设检验。模块化设计允许集成额外的实验模块，使该框架能够适应更广泛的科学工作流程。代码可在 https://github.com/IMMS-Ewha/MIND 获取，演示视频见 https://youtu.be/lqiFe1OQzN4。

    arXiv:2604.13699v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have enabled agentic AI systems for scientific discovery, but most approaches remain limited to textbased reasoning without automated experimental verification. We propose MIND, an LLM-driven framework for automated hypothesis validation in materials research. MIND organizes the scientific discovery process into hypothesis refinement, experimentation, and debate-based validation within a multi-agent pipeline. For experimental verification, the system integrates Machine Learning Interatomic Potentials, particularly SevenNet-Omni, enabling scalable in-silico experiments. We also provide a web-based user interface for automated hypothesis testing. The modular design allows additional experimental modules to be integrated, making the framework adaptable to broader scientific workflows. The code is available at: https://github.com/IMMS-Ewha/MIND, and a demonstration video at: https://youtu.be/lqiFe1OQzN4
    
[^322]: 前向-前向算法优良度函数设计中的选择性与形状

    Selectivity and Shape in the Design of Forward-Forward Goodness Functions

    [https://arxiv.org/abs/2604.13081](https://arxiv.org/abs/2604.13081)

    本文系统研究了前向-前向算法中优良度函数的选择性与形状设计，提出了一种无参数、尺度不变的“突发性优良度”目标函数（基于超额峰度、奖励重尾活动模式），在Fashion-MNIST上显著提升了准确率。

    

    前向-前向算法使用一个局部的“优良度函数”逐层训练网络，然而活动统计量的选择仍然是一个重要的设计问题。我们系统地研究了选择性与活动形状如何影响FF学习，引入了基于top-k和entmax加权能量的选择性优良度函数，以及基于归一化高阶矩的形状敏感目标函数。特别地，我们提出了突发性优良度，这是一种无参数、尺度不变的超额峰度目标函数，用于奖励重尾活动模式。我们的研究在六个小图像数据集上考察了优良度函数与激活函数及标签注入方式的组合效果，并通过受控实验来分离各自的影响。在经过验证集调优、五随机种子、匹配ReLU激活、标准标签注入和逐层评分的Fashion-MNIST实验中，突发性优良度将准确率从73.2±0.9%提高到（原文摘要在此处截断）。

    arXiv:2604.13081v3 Announce Type: replace-cross  Abstract: The Forward-Forward (FF) algorithm trains networks layer-by-layer using a local "goodness function," yet the choice of activity statistic remains an important design question. We systematically study how selectivity and activity shape affect FF learning, introducing selective goodness functions based on top-$k$ and entmax-weighted energy alongside shape-sensitive objectives based on normalized higher-order moments. In particular, we propose burstiness goodness, a parameter-free, scale-invariant excess-kurtosis objective that rewards heavy-tailed activity patterns. Our study examines goodness functions together with activation and label-injection choices across six small-image datasets, and uses controlled experiments to isolate their effects. In validation-tuned, five-seed Fashion-MNIST experiments with matched ReLU activation, standard label injection, and per-layer scoring, burstiness improves accuracy from $73.2\pm0.9\%$ to 
    
[^323]: 基于权重空间算术的零样本量化

    Zero-Shot Quantization via Weight-Space Arithmetic

    [https://arxiv.org/abs/2604.03420](https://arxiv.org/abs/2604.03420)

    该论文提出通过简单的权重空间算术从供体任务中提取“量化向量”来修补接收模型，从而在不进行量化感知训练、不使用接收端验证数据的零样本场景下，将3比特量化的Top-1准确率提升最多60%。

    

    我们证明了训练后量化（PTQ）的鲁棒性是权重空间中一个可迁移的方向。我们将这一方向称为量化向量：通过简单的权重空间算术从供体任务中提取，可用于修补接收模型，在3比特设置下将PTQ Top-1准确率提升最多60%，且无需在接收端进行量化感知训练（QAT）。在不使用任何接收端验证数据的情况下，该方法为极低比特部署提供了一种零样本、低成本的QAT替代方案。在多个视觉与语言模型以及30多个任务上的实验表明，即使供体任务与接收任务差异显著，供体量化向量通常仍能带来可观的收益。我们进一步严格证明了量化向量是良定义的，且不受重参数化对称性的影响，并对其效果给出了局部几何层面的解释。这些结果共同表明，量化鲁棒性可以成为……（原文摘要在此处截断）

    arXiv:2604.03420v4 Announce Type: replace-cross  Abstract: We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve PTQ Top-1 accuracy by up to 60% in a 3-bit setting, without receiver-side quantization-aware training (QAT). When using no receiver validation data, the method provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. Across multiple vision and language models and more than 30 tasks, donor quantization vectors often yield substantial gains even when donor and receiver tasks differ markedly. We further prove rigorously that quantization vectors are well-defined and do not suffer from reparameterization symmetries, and provide a local geometric account of their effect. Together, these results suggest that quantization robustness can be part
    
[^324]: 解释的正确性重要吗？将计算化XAI评估与人类理解联系起来

    Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding

    [https://arxiv.org/abs/2603.25251](https://arxiv.org/abs/2603.25251)

    该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。

    

    解释性人工智能（XAI）方法通常使用功能正确性指标（也称为忠实度或保真度）进行评估，这些指标衡量解释在多大程度上反映模型的真实推理过程。人们通常假设正确性越高，人类理解效果就越好，但这一联系尚未在受控的正确性水平下得到检验。我们开展了一项用户研究（N=200），在一个合成时间序列分类任务中，将解释正确性操纵在四个水平（100%、85%、70%、55%），参与者无法依赖领域知识或视觉直觉进行判断。解释被构建为类似特征归因的形式，其正确性基于已知的真实标准定义，而非从训练好的模型中估计得出。参与者需要根据这些解释预测模拟AI的决策（前向模拟），我们以其前向模拟准确率作为理解程度的代理指标。正确性影响了前向模拟准确率

    arXiv:2603.25251v3 Announce Type: replace-cross  Abstract: Explainable AI (XAI) methods are commonly evaluated using functional correctness metrics, also called faithfulness or fidelity, which estimate how closely an explanation reflects the model's reasoning. Higher correctness is assumed to produce better human understanding, but this link has not been tested with controlled levels. We conducted a user study (N=200) that manipulated explanation correctness at four levels (100%, 85%, 70%, 55%) in a synthetic time series classification task where participants could not rely on domain knowledge or visual intuition. Explanations were constructed to resemble feature attributions, with correctness defined against a known ground truth rather than estimated from a trained model. Participants predicted a simulated AI's decisions from these explanations (forward simulation), and we used their forward simulation accuracy as a proxy for understanding. Correctness affected forward simulation accu
    
[^325]: VTAM：无需触觉预训练的高效视觉-触觉世界-动作模型，用于富接触操作

    VTAM: Efficient Visuo-Tactile World-Action Models for Contact-Rich Manipulation without Tactile Pretraining

    [https://arxiv.org/abs/2603.23481](https://arxiv.org/abs/2603.23481)

    VTAM提出了一种无需触觉或动作预训练的高效视觉-触觉世界-动作模型，通过共享冻结的视频VAE并联合预测未来视觉与触觉潜在表示，在富接触操作的真实机器人任务上超越了所有基线。

    

    触觉视觉-语言-动作模型（VLAs）和触觉世界-动作模型（WAMs）是富接触操作的一种有前景的新范式，但现有模型要么未能充分利用触觉信息，要么依赖大规模触觉预训练，要么因迭代式视频生成而产生较高的推理延迟。我们提出了VTAM，一种用于富接触操作的高效视觉-触觉世界-动作模型，它无需触觉或动作预训练。RGB和触觉观测共享一个冻结的视频VAE以及预训练的视频transformer权重。VTAM首先将视频主干网络适配为联合预测未来的视觉和触觉潜在表示，然后与一个随机初始化的动作专家一起进行端到端微调，该动作专家负责预测动作、本体感知状态，以及一个由触觉光流计算得到的轻量级流-力代理信号。仅使用任务演示数据进行训练，VTAM在两个平台上的四个真实机器人任务中超越了所有基线，这些平台具有不同的……

    arXiv:2603.23481v2 Announce Type: replace-cross  Abstract: Tactile vision-language-action models (VLAs) and tactile world-action models (WAMs) are a promising new paradigm for contact-rich manipulation, but existing models either underutilize touch, rely on large-scale tactile pretraining, or incur high inference latency due to iterative video generation. We present VTAM, an efficient visuo-tactile world-action model for contact-rich manipulation that requires no tactile or action pretraining. RGB and tactile observations share a frozen video VAE and the pretrained video transformer weights. VTAM first adapts the video backbone to jointly predict future visual and tactile latents, then fine-tunes it end-to-end with a randomly initialized action expert that predicts actions, proprioceptive states, and a light flow-force proxy computed from tactile optical flow. Trained only on task demonstrations, VTAM outperforms all baselines on four real-robot tasks across two platforms with differen
    
[^326]: CTFS：面向极有限标注条件下前视声呐图像语义分割的协作教师框架

    CTFS : Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels

    [https://arxiv.org/abs/2603.21071](https://arxiv.org/abs/2603.21071)

    提出了一种由一个通用教师和多个声呐专用教师组成的多教师协作框架CTFS，通过多教师交替引导策略，在极有限标注条件下显著提升了前视声呐图像语义分割的性能。

    

    作为最重要的水下感知技术之一，前视声呐具有独特的成像特性。声呐图像通常受到严重散斑噪声、低纹理对比度、声学阴影和几何畸变的影响。这些因素使得传统的师生框架在极有限标注数据条件下难以在声呐语义分割任务中取得令人满意的性能。为解决这一问题，我们提出了一种面向前视声呐图像的协作教师语义分割框架。该框架引入了一种由一个通用教师和多个声呐专用教师组成的多教师协作机制。通过采用多教师交替引导策略，学生模型能够在学习通用语义表征的同时捕捉声呐图像的独特特性，从而实现更全面和（更精准的分割性能）。

    arXiv:2603.21071v4 Announce Type: replace-cross  Abstract: As one of the most important underwater sensing technologies, forward-looking sonar exhibits unique imaging characteristics. Sonar images are often affected by severe speckle noise, low texture contrast, acoustic shadows, and geometric distortions. These factors make it difficult for traditional teacher-student frameworks to achieve satisfactory performance in sonar semantic segmentation tasks under extremely limited labeled data conditions. To address this issue, we propose a Collaborative Teacher Semantic Segmentation Framework for forward-looking sonar images. This framework introduces a multi-teacher collaborative mechanism composed of one general teacher and multiple sonar-specific teachers. By adopting a multi-teacher alternating guidance strategy, the student model can learn general semantic representations while simultaneously capturing the unique characteristics of sonar images, thereby achieving more comprehensive and
    
[^327]: 面向低光照图像增强的线索引导双路径重建与自适应多尺度响应精化方法

    Cue-guided dual-path reconstruction with adaptive multi-scale response refinement for low-light image enhancement

    [https://arxiv.org/abs/2603.16482](https://arxiv.org/abs/2603.16482)

    提出CueDR-Net，一种通过双向线索交互与自适应多尺度精化，将亮度调整与局部内容重建相结合的双路径网络，有效解决低光照图像增强中的色彩偏差与局部细节模糊问题。

    

    低光照图像增强（LLIE）在光子捕获受限和传感器噪声导致图像质量下降的情况下，为光学感知和场景观测提供支持。然而，现有方法可能在增强后的图像中引入色彩偏差或模糊局部细节。为应对这一挑战，我们提出了CueDR-Net，这是一种双路径网络，通过双向线索交互和自适应多尺度精化，将亮度调整与局部内容重建相结合。具体而言，一个固定提取器从输入图像中推导出亮度变化、颜色和纹理线索图，这些线索图被编码到线索流中，并通过双向注意力与图像流交换特征。交换后的特征支持两条路径：亮度调整路径和局部内容重建路径。其中，亮度调整路径预测逐像素的调整图，用于控制输入图像的迭代强度变化。与此同时……

    arXiv:2603.16482v2 Announce Type: replace-cross  Abstract: Low-light image enhancement (LLIE) supports optical sensing and scene observation when limited photon capture and sensor noise degrade image quality. However, existing methods can introduce colour deviations or blur local details in the enhanced images. To address this challenge, we propose CueDR-Net, a dual-path network that combines brightness adjustment with local content reconstruction through bidirectional cue interaction and adaptive multi-scale refinement. Specifically, a fixed extractor derives luminance variation, colour, and texture cue maps from the input image. These maps are encoded in a cue stream that exchanges features with the image stream through bidirectional attention. The exchanged features support two paths for brightness adjustment and local content reconstruction. The brightness adjustment path predicts pixel wise adjustment maps that control iterative intensity changes in the input image. In parallel, t
    
[^328]: 面向长尾鲁棒驾驶的偏好引导对抗策略优化

    Preference-Guided Adversarial Policy Optimization for Long-Tail Robust Driving

    [https://arxiv.org/abs/2603.15221](https://arxiv.org/abs/2603.15221)

    ADV-0提出一个闭环极小极大优化框架，将驾驶策略与对抗智能体的交互建模为零和马尔可夫博弈，并通过迭代偏好学习高效逼近最优对抗分布，从而提升自动驾驶系统对罕见但安全关键的长尾场景的鲁棒性。

    

    部署自动驾驶系统需要具备应对长尾场景的鲁棒性——这些场景虽然罕见，但事关安全。尽管对抗训练提供了一种有前景的解决方案，但现有方法通常将场景生成与策略优化解耦，并依赖启发式替代目标。这导致目标不一致，且无法捕捉不断演化的策略所呈现的动态失效模式。本文提出ADV-0，一个闭环的极小极大优化框架，将驾驶策略（防御者）与对抗智能体（攻击者）之间的交互建模为零和马尔可夫博弈。通过将攻击者的效用直接与防御者的目标对齐，我们揭示了最优对抗分布。为使该问题可求解，我们将动态对抗演化转化为迭代偏好学习，高效逼近该最优解，并为该博弈提供一种与算法无关的解决方案。理论上，ADV-0收敛到……（摘要在此处被截断）

    arXiv:2603.15221v2 Announce Type: replace-cross  Abstract: Deploying autonomous driving systems requires robustness against long-tail scenarios that are rare but safety-critical. While adversarial training offers a promising solution, existing methods typically decouple scenario generation from policy optimization and rely on heuristic surrogates. This leads to objective misalignment and fails to capture the shifting failure modes of evolving policies. This paper presents ADV-0, a closed-loop min-max optimization framework that treats the interaction between driving policy (defender) and adversarial agent (attacker) as a zero-sum Markov game. By aligning the attacker's utility directly with the defender's objective, we reveal the optimal adversary distribution. To make this tractable, we cast dynamic adversary evolution as iterative preference learning, efficiently approximating this optimum and offering an algorithm-agnostic solution to the game. Theoretically, ADV-0 converges to a Na
    
[^329]: BALD-SAM：交互式分割中基于分歧的主动提示方法

    BALD-SAM: Disagreement-based Active Prompting in Interactive Segmentation

    [https://arxiv.org/abs/2603.10828](https://arxiv.org/abs/2603.10828)

    本文提出“主动提示”新范式，将图像内位置视为未标记样本池、提示作为查询，并基于BALD-SAM框架利用贝叶斯分歧准则自动选择下一次提示的最优空间位置，从而最大化SAM交互式分割中每次人机交互的信息效用。

    

    分割一切模型（SAM）通过空间提示彻底改变了交互式分割领域。尽管现有工作主要聚焦于在各种场景下实现提示的自动化，但真实世界的标注工作流涉及迭代式细化过程，标注者需要观察模型输出并策略性地放置提示以消除歧义。当前的流程通常依赖于标注者对预测掩码质量的主观视觉评估。我们提出，自动化交互式提示的一种有原则的方法是使用由模型导出的准则来识别下一次提示最具信息量的区域。在本工作中，我们建立了“主动提示”这一概念：一种空间主动学习方法，其中图像内的各个位置构成未标记样本池，提示则作为查询用于优先选择信息丰富的区域，从而提升每次交互的效用。我们进一步提出了BALD-SAM：一个采用贝叶斯主动学习分歧方法的原则性框架…

    arXiv:2603.10828v2 Announce Type: replace-cross  Abstract: The Segment Anything Model (SAM) has revolutionized interactive segmentation through spatial prompting. While existing work primarily focuses on automating prompts in various settings, real-world annotation workflows involve iterative refinement where annotators observe model outputs and strategically place prompts to resolve ambiguities. Current pipelines typically rely on the annotator's visual assessment of the predicted mask quality. We postulate that a principled approach for automated interactive prompting is to use a model-derived criterion to identify the most informative region for the next prompt. In this work, we establish active prompting: a spatial active learning approach where locations within images constitute an unlabeled pool and prompts serve as queries to prioritize information-rich regions, increasing the utility of each interaction. We further present BALD-SAM: a principled framework adapting Bayesian Acti
    
[^330]: MOO：用于牛只重识别中视角分析的多视角定向观测数据集

    MOO: A Multi-view Oriented Observations Dataset for Viewpoint Analysis in Cattle Re-Identification

    [https://arxiv.org/abs/2603.04314](https://arxiv.org/abs/2603.04314)

    提出了包含1,000头牛、128个均匀采样视角的大规模合成数据集MOO，系统量化了俯仰高度对动物重识别的影响并确定了关键高度阈值，同时在零样本和有监督设置下验证了其在真实牛只数据集上的可迁移性与性能增益。

    

    动物重识别面临由视角变化带来的关键挑战，特别是在空地重识别场景中，模型需要在剧烈的高度变化下匹配个体。然而，现有数据集缺乏系统分析这些几何变化所需的精确角度标注。为解决这一问题，我们引入了多视角定向观测（MOO）数据集，这是一个大规模合成空地重识别数据集，包含1,000头牛的个体，从128个均匀采样的视角拍摄，共128,000张标注图像。利用这一受控数据集，我们量化了俯仰高度的影响，并确定了一个关键的高度阈值，超过该阈值后模型对未见视角的泛化能力显著提升。最后，我们在零样本和有监督两种设置下验证了向真实世界应用的可迁移性，在四个真实世界的牛只数据集上展示了性能提升。

    arXiv:2603.04314v3 Announce Type: replace-cross  Abstract: Animal re-identification (ReID) faces critical challenges due to viewpoint variations, particularly in Aerial-Ground (AG-ReID) settings where models must match individuals across drastic elevation changes. However, existing datasets lack the precise angular annotations required to systematically analyze these geometric variations. To address this, we introduce the Multi-view Oriented Observation (MOO) dataset, a large-scale synthetic AG-ReID dataset of $1,000$ cattle individuals captured from $128$ uniformly sampled viewpoints ($128,000$ annotated images). Using this controlled dataset, we quantify the influence of elevation and identify a critical elevation threshold, above which models generalize significantly better to unseen views. Finally, we validate the transferability to real-world applications in both zero-shot and supervised settings, demonstrating performance gains across four real-world cattle datasets and confirmin
    
[^331]: AgentSelect：面向叙述性查询到智能体推荐的基准测试

    AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation

    [https://arxiv.org/abs/2603.03761](https://arxiv.org/abs/2603.03761)

    AgentSelect是一个将智能体选择重新建模为“叙述性查询到智能体推荐”任务的基准，聚合了超过11万条查询、10.7万个可部署智能体和25万余条交互记录，填补了端到端智能体配置推荐缺乏查询条件监督数据的研究空白。

    

    LLM智能体正迅速成为任务自动化的实用接口，然而该生态系统缺乏一种在快速膨胀的可部署配置空间中进行选择的原则性方法。现有的LLM排行榜和工具/智能体基准测试只是孤立地评估各个组件，并且在任务、指标和候选池之间仍然碎片化，留下了一个关键的研究空白：几乎没有基于查询条件的监督数据来学习推荐将骨干模型与工具包相结合的端到端智能体配置。我们通过AgentSelect来填补这一空白，该基准将智能体选择重新构建为基于能力档案的叙述性查询到智能体的推荐问题，并系统地将异构的评估产物转换为统一的、仅含正样本的交互数据。AgentSelect包含来自40多个数据源的111,179条查询、107,721个可部署智能体和251,103条交互记录，涵盖仅LLM型、仅工具包型以及组合型……（原文摘要至此截断）

    arXiv:2603.03761v2 Announce Type: replace  Abstract: LLM agents are rapidly becoming the practical interface for task automation, yet the ecosystem lacks a principled way to choose among an exploding space of deployable configurations. Existing LLM leaderboards and tool/agent benchmarks evaluate components in isolation and remain fragmented across tasks, metrics, and candidate pools, leaving a critical research gap: there is little query-conditioned supervision for learning to recommend end-to-end agent configurations that couple a backbone model with a toolkit. We address this gap with AgentSelect, a benchmark that reframes agent selection as narrative query-to-agent recommendation over capability profiles and systematically converts heterogeneous evaluation artifacts into unified, positive-only interaction data. AgentSelectcomprises 111,179 queries, 107,721 deployable agents, and 251,103 interaction records aggregated from 40+ sources, spanning LLM-only, toolkit-only, and composition
    
[^332]: ASTRA-bench：基于个人用户情境评估工具使用智能体的推理与行动规划

    ASTRA-bench: Evaluating Tool-Use Agent Reasoning and Action Planning with Personal User Context

    [https://arxiv.org/abs/2603.01357](https://arxiv.org/abs/2603.01357)

    ASTRA-bench 是首个将随时间演变的个人用户情境与交互式工具箱和复杂用户意图相统一的工具智能体评测基准，其基于纵向生活事件构建的 2,413 个场景揭示了现有最先进模型在高复杂性条件下性能显著下降、参数生成是主要瓶颈的问题。

    

    下一代人工智能必须管理海量的个人数据、多样化的工具以及多步骤推理，然而现有的多数基准测试仍然是无上下文的单轮测试。我们提出了 ASTRA-bench（工具使用、推理与行动规划中的助手技能），这是一个独特地将随时间演变的个人情境与交互式工具箱及复杂用户意图相统一的基准测试。我们的事件驱动流程基于纵向生活事件生成了覆盖四名主人公的 2,413 个场景，并按照指代复杂性、功能复杂性和信息复杂性进行了标注。对最先进模型（如 Claude-4.5-Opus、DeepSeek-V3.2）的评估显示，这些模型在高复杂性条件下出现显著的性能下降，其中工具参数生成成为主要瓶颈。这些发现揭示了当前智能体在将推理锚定于杂乱的个人情境以及编排可靠多步骤计划方面的关键局限性。我们公开发布了 ASTRA-bench……

    arXiv:2603.01357v2 Announce Type: replace  Abstract: Next-generation AI must manage vast personal data, diverse tools, and multi-step reasoning, yet most benchmarks remain context-free and single-turn. We present ASTRA-bench (Assistant Skills in Tool-use, Reasoning \& Action-planning), a benchmark that uniquely unifies time-evolving personal context with an interactive toolbox and complex user intents. Our event-driven pipeline generates 2,413 scenarios across four protagonists, grounded in longitudinal life events and annotated by referential, functional, and informational complexity. Evaluation of state-of-the-art models (e.g., Claude-4.5-Opus, DeepSeek-V3.2) reveals significant performance degradation under high-complexity conditions, with argument generation emerging as the primary bottleneck. These findings expose critical limitations in current agents' ability to ground reasoning within messy personal context and orchestrate reliable multi-step plans. We release ASTRA-bench with 
    
[^333]: 可训练频率能移动多远？可训练频率量子模型中的分辨率、网格与耦合

    How Far Can a Trainable Frequency Move? Resolution, Grids and Coupling in Trainable-FrequencyQuantum Models

    [https://arxiv.org/abs/2602.23409](https://arxiv.org/abs/2602.23409)

    该论文揭示了可训练频率量子模型中前置因子梯度的解析结构——由 sin u/u 重叠函数斜率的加权和决定每个频率的可达范围，并提出三值前置因子网格方案，仅用对数数量的编码门即可将模型频率逼近所有频带频率至半个步长以内。

    

    可训练频率量子模型通过训练其编码门的前置因子，使电路的频率能够移动到目标的频率上。从先前工作的单位起始点出发，训练很少能到达距离超过一个分辨率步长的目标。我们证明，前置因子梯度中由目标驱动的部分是重叠函数 sin u/u 斜率的加权和，其中 u 是按窗口半宽缩放的模型-目标频率距离。这赋予每个频率一个“可达范围”：在梯度流下，一个系数最优自适应的孤立实频率能够从半个步长之内到达其目标，只要目标距离零频率至少一个步长；而在系数固定的情况下，一个依赖相位的势垒可能会阻止它。一个三值前置因子网格能够使模型频率与每个频带频率的差距都在半个步长以内，且只需对数数量的编码门。使用 Adam 训练的单频率电路以及联合训练的振幅……

    arXiv:2602.23409v4 Announce Type: replace-cross  Abstract: Trainable-frequency quantum models train the prefactors of their encoding gates so that the circuit's frequencies can move onto the target's. From the unit start of prior work, training rarely reaches a target more than one resolution step away. We show that the target-driven part of the prefactor gradient is a weighted sum of slopes of the overlap $\sin u/u$, where $u$ is the model-target frequency distance scaled by the window half-width. This gives each frequency a reach: under gradient flow, an isolated real frequency whose coefficient adapts optimally reaches its target from within half a step, for every target at least one step from zero frequency; with a fixed coefficient, a phase-dependent barrier can stop it. A ternary prefactor grid places a model frequency within half a step of every band frequency, with logarithmically many encoding gates. Single-frequency circuits trained with Adam and jointly trained amplitudes be
    
[^334]: RoboCurate：利用经动作验证的合成数据挖掘多样性以促进机器人学习

    RoboCurate: Harnessing Diversity with Action-Verified Synthetic Data for Robot Learning

    [https://arxiv.org/abs/2602.18742](https://arxiv.org/abs/2602.18742)

    提出RoboCurate框架，通过在模拟器中执行IDM预测的动作并对比模拟器回放与生成视频的运动一致性来验证合成数据的动作标签，同时扩展场景多样性，为机器人学习提供高质量、可扩展的合成数据。

    

    由视频生成模型生成的合成数据作为一种可扩展的流程，在机器人学习中展现出巨大前景，但由于视频生成不完善或动作预测不准确，其动作标签常常存在噪声。最近，视觉-语言模型（VLM）已被用于验证视频质量，但它们无法直接评估生成的动作本身。为解决这一问题，我们提出了RoboCurate，一个新颖的动作验证合成机器人数据框架，用于解决生成视频与标注动作之间的不一致性。具体而言，RoboCurate在模拟器中执行由逆动力学模型（IDM）预测的动作，并通过测量模拟器回放与生成视频之间的运动一致性来验证动作的一致性。为了更好地利用该框架，我们进一步通过现成的生成模型扩展了合成数据的场景多样性，使其超越现有可用数据集的范围……

    arXiv:2602.18742v2 Announce Type: replace-cross  Abstract: Synthetic data generated by video generative models has shown promise for robot learning as a scalable pipeline, but it often suffers from noisy action labels due to imperfect video generation or inaccurate action prediction. Recently, vision-language models (VLMs) have been leveraged to validate video quality, but they cannot directly evaluate the generated actions themselves. To tackle this issue, we introduce RoboCurate, a novel framework for action-verified synthetic robot data that addresses inconsistencies between generated videos and annotated actions. Specifically, RoboCurate executes the actions predicted by an inverse dynamics model (IDM) in a simulator and verifies action consistency by measuring the motion agreement between the simulator rollout and the generated video. To better leverage such a framework, we further expand the scene diversity of synthetic data beyond the available dataset through off-the-shelf gene
    
[^335]: WoVR：将世界模型作为可靠模拟器用于VLA策略的强化学习后训练

    WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL

    [https://arxiv.org/abs/2602.13977](https://arxiv.org/abs/2602.13977)

    WoVR提出了一种基于世界模型的可靠强化学习框架，用于VLA策略的后训练，通过显式调控强化学习与不完美想象动力学之间的交互，解决世界模型幻觉和长时程误差累积导致的不可靠学习信号问题。

    

    强化学习有望为视觉-语言-动作（VLA）模型解锁超越模仿学习的能力，但其对大规模真实世界交互的需求阻碍了直接部署到物理机器人上。近期工作尝试使用学习到的世界模型作为策略优化的模拟器，然而闭环的想象轨迹不可避免地会受到幻觉和长时程误差累积的影响。这些误差不仅会降低视觉保真度，还会通过提供不可靠的学习信号误导策略优化。我们提出了WoVR，一个用于VLA策略后训练的、基于世界模型的可靠强化学习框架。WoVR不假设世界模型是完全忠实的，而是显式地调节强化学习如何与不完美的想象动力学进行交互。它通过一个可控的动作条件视频世界模型来提升轨迹生成的稳定性，并通过关键帧初始化重塑想象交互以减少有效误差深度。

    arXiv:2602.13977v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) promises to unlock capabilities beyond imitation learning for Vision--Language--Action (VLA) models, but its requirement for massive real-world interaction prevents direct deployment on physical robots. Recent work attempts to use learned world models as simulators for policy optimization, yet closed-loop imagined rollouts inevitably suffer from hallucination and long-horizon error accumulation. Such errors not only degrade visual fidelity, but also mislead policy optimization by providing unreliable learning signals. We propose WoVR, a reliable world-model-based RL framework for post-training VLA policies. Instead of assuming a faithful world model, WoVR explicitly regulates how RL interacts with imperfect imagined dynamics. It improves rollout stability through a controllable action-conditioned video world model, reshapes imagined interaction to reduce effective error depth via Keyframe-Initialized
    
[^336]: ProMoral-Bench：评估大语言模型中道德推理与安全性的提示策略

    ProMoral-Bench: Evaluating Prompting Strategies for Moral Reasoning and Safety in LLMs

    [https://arxiv.org/abs/2602.13274](https://arxiv.org/abs/2602.13274)

    该论文提出了ProMoral-Bench基准，系统比较了四种大语言模型家族上的10种提示范式在道德推理与安全方面的表现，并发现紧凑的示例引导式提示能在更低的token成本下取得更强的能力与鲁棒性权衡。

    

    提示设计会影响大语言模型（LLM）所产生的道德判断和拒绝行为，然而受控比较研究仍分散于不同数据集和模型之间。我们提出了ProMoral-Bench，这是一个基准式的比较研究，涵盖四种大语言模型家族上的10种提示范式。我们使用ETHICS、Scruples、WildJailbreak以及我们构建的200对ETHICS-Contrast数据集，报告了任务级指标，以及道德能力分数、针对越狱/过度拒绝的安全鲁棒性分数和统一道德安全分数。UMSS是针对本研究的对MCS和SRS的相对性总结指标，而非对模型安全性或道德基础的绝对度量。在这一固定模板评估中，紧凑的示例引导式提示支架比所测试的特定多阶段提示变体在更低的token成本下实现了更强的能力-鲁棒性权衡，尽管多阶段提示偶尔能改善对越狱的拒绝能力。少样本演示是无重叠的（原文摘要至此处截断）。

    arXiv:2602.13274v2 Announce Type: replace  Abstract: Prompt design affects the moral judgments and refusal behavior produced by large language models (LLMs), yet controlled comparisons remain fragmented across datasets and models. We introduce ProMoral-Bench, a benchmark-style comparison of 10 prompting paradigms across four LLM families. Using ETHICS, Scruples, WildJailbreak, and our 200-pair ETHICS-Contrast set, we report task-level metrics together with Moral Competence Score (MCS), jailbreak/over-refusal Safety Robustness Score (SRS), and Unified Moral Safety Score (UMSS). UMSS is a study-specific relative summary of MCS and SRS, not an absolute measure of model safety or moral grounding. In this fixed-template evaluation, compact exemplar-guided scaffolds attain stronger competence-robustness trade-offs at lower token cost than the particular multi-stage variants tested, although multi-stage prompting occasionally improves jailbreak refusal. The few-shot demonstrations are non-ove
    
[^337]: 当LLM路由器过度支付时：强模型偏好的代价

    When LLM Routers Overpay: The Cost of Strong-Model Preference

    [https://arxiv.org/abs/2602.03478](https://arxiv.org/abs/2602.03478)

    该论文发现现有LLM路由器在预算宽松时会因对昂贵强模型的持久偏好而造成过度支付（“强模型过度选择”），并提出决策对齐的路由器EquiRouter来学习查询级的恰当模型分配，从而避免不必要的成本浪费。

    

    LLM路由的目标是通过将简单查询分配给较便宜的模型，并将更强大的模型保留给真正需要它们的场景，从而降低推理成本。然而，我们发现现有路由器在宽松的成本预算下经常违背这一原则：随着预算增加，它们越来越多地选择最强大、最昂贵的模型，即使较便宜的可行候选模型能提供相当甚至相同的效用。我们将这种行为称为“宽松预算下的强模型过度选择”。我们的分析表明，仅凭查询歧义性并不能解释这一现象。当不同价格的模型提供等效或近似等效的效用，而路由器对昂贵模型保持持久的模型级偏好时，就会产生大量的过度支付；预算的放宽通过使高价候选模型变为可行选项而暴露出这种偏好。基于这一诊断，我们提出了EquiRouter，一种决策对齐的路由器，它学习查询级（原文此处截断）

    arXiv:2602.03478v2 Announce Type: replace  Abstract: LLM routing aims to reduce inference costs by assigning easy queries to less expensive models and reserving stronger models for cases in which they are needed. However, we find that existing routers often violate this principle under loose cost budgets: as the budget increases, they increasingly select the strongest and most expensive model even when less expensive feasible candidates provide comparable or identical utility. We refer to this behavior as strong-model over-selection under loose budgets. Our analysis shows that ambiguity alone does not explain this phenomenon. Substantial overpayment can arise when differently priced models offer equivalent or near-equivalent utility and the router retains a persistent model-level preference for costly models; budget relaxation exposes this preference by making premium candidates feasible. Motivated by this diagnosis, we propose EquiRouter, a decision-aligned router that learns query-de
    
[^338]: 立场：智能体化进化是大型语言模型进化的道路

    Position: Agentic Evolution is the Path to Evolving LLMs

    [https://arxiv.org/abs/2602.00359](https://arxiv.org/abs/2602.00359)

    本文提出智能体化进化是LLM应对部署环境变化的新扩展轴，并给出A-Evolve框架，将部署时改进从固定流程提升为自主进化者智能体执行的目标导向优化过程。

    

    随着大型语言模型（LLM）从精心策划的训练集走向开放的现实世界环境，一个根本性的局限显现出来：静态训练无法跟上部署环境的持续变化。扩大训练时和推理时的计算规模可以提升静态能力，但无法弥合这种训练与部署之间的差距。我们认为，解决这一局限需要一个全新的扩展轴——进化。现有的部署时适应方法，无论是参数化微调还是启发式记忆积累，都缺乏诊断故障并产生持久改进所需的策略性智能体能力。我们的立场是，智能体化进化代表了LLM适应的必然未来，即将进化本身从固定的流水线提升为一个自主的进化者智能体。我们在一个通用框架A-Evolve中实例化了这一愿景，该框架将部署时的改进视为一种深思熟虑的、目标导向的优化过程。

    arXiv:2602.00359v3 Announce Type: replace  Abstract: As Large Language Models (LLMs) move from curated training sets into open-ended real-world environments, a fundamental limitation emerges: static training cannot keep pace with continual deployment environment change. Scaling training-time and inference-time compute improves static capability but does not close this train-deploy gap. We argue that addressing this limitation requires a new scaling axis-evolution. Existing deployment-time adaptation methods, whether parametric fine-tuning or heuristic memory accumulation, lack the strategic agency needed to diagnose failures and produce durable improvements. Our position is that agentic evolution represents the inevitable future of LLM adaptation, elevating evolution itself from a fixed pipeline to an autonomous evolver agent. We instantiate this vision in a general framework, A-Evolve, which treats deployment-time improvement as a deliberate, goal-directed optimization process over pe
    
[^339]: AI生成图像检测器利用全局伪影：来自修复交换的证据

    AI-Generated Image Detectors Exploit Global Artifacts: Evidence from Inpainting Exchange

    [https://arxiv.org/abs/2602.00192](https://arxiv.org/abs/2602.00192)

    该论文发现最先进的AI生成图像检测器实际利用的是扩散修复管线中VAE编码-解码带来的全局频谱偏移伪影，而非识别被合成的具体内容，并提出了无需模型的INP-X诊断方法来验证这一现象。

    

    现代基于扩散模型的修复技术能够实现逼真的局部图像编辑，这给可靠检测带来了重大挑战。我们证明了最先进的AI生成图像检测器在很大程度上无法识别被合成的具体内容，而是利用了生成管线副作用引入的一种全局伪影：潜空间扩散修复会将整幅图像通过VAE编码器-解码器，即使在未编辑区域也会引发细微但普遍存在的频谱偏移。我们将这一观察转化为一种简单的、无需模型的诊断方法——修复交换，该方法在保留所有合成内容的同时，恢复编辑区域之外的原始像素。为了大规模研究这一效应，我们构建了一个基准数据集，包含30K组匹配的真实图像、标准修复图像和交换图像三元组（共90K张图像），涵盖四个数据集和三个修复模型。INP-X无需模型访问权限，是一种单一的合成方法。

    arXiv:2602.00192v2 Announce Type: replace-cross  Abstract: Modern diffusion-based inpainting enables realistic, localized image manipulation, raising critical challenges for reliable detection. We show that state-of-the-art AI-generated image detectors largely fail to identify \emph{what} was synthesized and instead exploit a \emph{global} artifact introduced as a side effect of the generation pipeline: latent-diffusion inpainting passes the entire image through a VAE encoder--decoder, inducing a subtle but pervasive spectral shift even in unedited regions. We turn this observation into a simple, model-free diagnostic, \emph{Inpainting Exchange} (INP-X), that restores the original pixels outside the edited region while preserving all synthesized content. To study the effect at scale, we build a benchmark of $30$K matched real, standard-inpainted, and exchanged triplets ($90$K images) spanning four datasets and three inpainters. INP-X requires no model access and is a single compositing
    
[^340]: 深度循环注意力混合： 让潜在推理得到应有的关注

    Depth-Recurrent Attention Mixture (Dreamer): Giving Latent Reasoning the Attention it Deserves

    [https://arxiv.org/abs/2601.21582](https://arxiv.org/abs/2601.21582)

    该论文提出Dreamer模块化框架，通过结合序列注意力、深度注意力和稀疏专家注意力来突破恒定隐藏维度的瓶颈，使深度循环模型在语言推理基准上以2-8倍更少的训练token达到与SOTA相当的精度，并优于约2倍规模的SOTA模型。

    

    深度循环通过跨深度共享参数来促进潜在推理。然而，先前的工作缺乏同时匹配FLOP、参数量和内存的基线对比，由于部分固定的层堆叠而未充分利用深度循环，并且忽略了限制多步潜在推理的恒定隐藏维度瓶颈。为解决这一问题，我们提出了一个深度循环注意力混合的模块化框架，结合了序列注意力、深度注意力和稀疏专家注意力。该框架通过沿深度方向的注意力缓解了隐藏维度瓶颈，解耦了各缩放维度，使深度循环模型能够高效且有效地扩展。在语言推理基准测试中，我们的模型在与FLOP、参数量和内存均匹配的SOTA模型达到相同精度时，所需训练token减少2到8倍，并在相同训练token数量下优于约2倍规模的SOTA模型。我们还进一步提供了关于知识的洞见……

    arXiv:2601.21582v2 Announce Type: replace  Abstract: Depth-recurrence facilitates latent reasoning by sharing parameters across depths. However, prior work lacks combined FLOP-, parameter-, and memory-matched baselines, underutilizes depth-recurrence due to partially fixed layer stacks, and ignores the bottleneck of constant hidden-sizes that restricts many-step latent reasoning. To address this, we introduce a modular framework of depth-recurrent attention mixtures (Dreamer), combining sequence attention, depth attention, and sparse expert attention. It alleviates the hidden-size bottleneck through attention along depth, decouples scaling dimensions, and allows depth-recurrent models to scale efficiently and effectively. Across language reasoning benchmarks, our models require 2 to 8x fewer training tokens for the same accuracy as FLOP-, parameter-, and memory-matched SOTA, and outperform ca. 2x larger SOTA models with the same training tokens. We further present insights into knowled
    
[^341]: 自我改进的训练与思考

    Self-Improving Training & Thinking

    [https://arxiv.org/abs/2601.21343](https://arxiv.org/abs/2601.21343)

    该论文提出一种将安全性、事实性和推理等能力更早融入预训练与中期训练的新方法，通过利用现有强模型重写数据和评判模型输出，并引入“自我改进的后思考”机制，在训练全流程中提升模型的推理能力。

    

    传统上，大型语言模型采用分阶段训练：先在原始文本上进行预训练，再进行后训练以学习指令遵循和推理能力。然而，这种分离造成了根本性的局限：许多理想的行为，如安全性、事实准确性、整体生成质量和推理能力，只在训练后期才被加入，尽管早期学习的模式在很大程度上塑造了模型的能力。为解决这一问题，我们提出了一种新的预训练和中期训练方法，将这些行为更早地融入训练过程。我们利用一个现有的强大的经过后训练的模型来重写数据并评判策略模型的输出（rollouts），从而在训练中更早地引入思考和强化学习。我们进一步通过引入“自我改进的后思考”（self-improving post-thinking）的概念，在所有训练阶段提升推理能力：即在生成之后进行推理，以提高模型的推理和自我评判能力。在实验中，我们证明了这些方法……

    arXiv:2601.21343v4 Announce Type: replace-cross  Abstract: Large language models are classically trained in stages: pretraining on raw text followed by post-training for instruction following and reasoning. However, this separation creates a fundamental limitation: many desirable behaviors such as safety, factuality, overall generation quality, and reasoning ability are only added at a late stage, even though the patterns learned earlier strongly shape a model's capabilities. To tackle this issue, we introduce a new way to pretrain and mid-train models that incorporates these behaviors earlier. We utilize an existing strong, post-trained model to both rewrite data and to judge policy model rollouts, thus introducing thinking and reinforcement earlier in training. We further improve reasoning at all stages by introducing the concept of self-improving post-thinking: improving the model's reasoning and self-judging ability by reasoning after generation. In our experiments, we show these a
    
[^342]: 大语言模型对GPU软错误的脆弱性分析：一项指令级故障注入研究

    Analysis of LLM Vulnerability to GPU Soft Errors: An Instruction-Level Fault Injection Study

    [https://arxiv.org/abs/2601.19912](https://arxiv.org/abs/2601.19912)

    本文首次针对大语言模型推理开展指令级故障注入研究，系统分析了LLM对GPU软错误的脆弱性，填补了现代大模型可靠性评估领域的空白。

    

    大语言模型（LLMs）具有高度的计算和内存密集性，对高性能GPU提出了巨大的需求。与此同时，受晶体管尺寸缩小和工作电压降低驱动的GPU技术进步，使这些设备越来越容易受到软错误的影响，尤其是在卫星计算场景等辐射易发环境中。虽然已有研究考察了GPU软错误，但大多数研究集中于通用应用程序或主要用于分类和检测等视觉任务的传统神经网络。然而，尽管现代大语言模型在各种应用场景中被快速采用，但针对它们的脆弱性分析仍然有限。鉴于大语言模型的独特特性，其对软错误的恢复能力可能与早期模型存在显著差异。为了填补这一空白，我们首次对大语言模型推理开展了指令级故障注入研究……

    arXiv:2601.19912v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are highly compute- and memory-intensive, posing significant demands on high-performance GPUs. At the same time, advances in GPU technology driven by shrinking transistor sizes and lower operating voltages have made these devices increasingly susceptible to soft errors, particularly in radiation-prone environments such as satellite computing scenarios. While prior work has examined GPU soft errors, most studies have focused on general-purpose applications or conventional neural networks primarily used for vision tasks such as classification and detection. However, vulnerability analysis of modern LLMs remains limited, despite their rapid adoption in diverse application scenarios. Given the unique characteristics of LLMs, their resilience to soft errors may differ substantially from earlier models. To bridge this gap, we conduct the first instruction-level fault injection study of LLM inference to an
    
[^343]: EmboTeam：面向异构多机器人协作的非对称状态接地

    EmboTeam: Asymmetric State Grounding for Heterogeneous Multi-Robot Collaboration

    [https://arxiv.org/abs/2601.11063](https://arxiv.org/abs/2601.11063)

    提出 EmboTeam 神经符号框架，通过 LLM 生成结构化状态预期、PDDL 符号搜索验证协调，并落地为带局部重规划与黑板同步的团队级并行行为树，实现异构多机器人长时程协作中的非对称状态接地，显著提升执行鲁棒性。

    

    面向长时程的异构多机器人协作既需要全局任务规划，也需要执行过程中的鲁棒协调。现有基于大语言模型（LLM）的规划器能够生成任务分解或符号化规划，但往往将其作为僵硬的动作序列执行，导致其对局部状态不匹配和同步失败十分脆弱。我们提出 EmboTeam，一个用于非对称状态接地的神经符号框架。LLM 生成结构化的状态预期，这些预期通过基于规划领域定义语言（PDDL）的符号搜索进行验证与协调，随后被落地为团队级的并行行为树。针对各机器人的子树负责前置条件检查、技能执行、效果验证、局部重规划以及基于黑板机制的同步。我们进一步提出了 MACE-THOR 基准，包含跨 8 种室内布局的 42 个异构多机器人任务。EmboTeam 提升了任务的……（原文摘要至此截断）

    arXiv:2601.11063v3 Announce Type: replace-cross  Abstract: Heterogeneous multi-robot collaboration over long horizons requires both global task planning and robust coordination during execution. Existing plan- ners using large language models (LLMs) can generate task decompositions or symbolic plans, but often execute them as rigid action sequences, making them fragile to local state mismatches and synchronization failures. We propose Em- boTeam, a neuro-symbolic framework for asymmetric state grounding. An LLM generates structured state expectations, which are verified and coordinated through symbolic search using Planning Domain Definition Language (PDDL) and then grounded into a team-level parallel behavior tree. Robot-specific subtrees per- form precondition checking, skill execution, effect validation, local replanning, and blackboard-based synchronization. We further introduce MACE-THOR, con- taining 42 heterogeneous multi-robot tasks across 8 indoor layouts. EmboTeam improves ta
    
[^344]: 帕累托DNN验证：对大多数查询快速完成

    Pareto DNN Verification: Fast for Most Queries

    [https://arxiv.org/abs/2512.20755](https://arxiv.org/abs/2512.20755)

    提出PAVE方法，通过为DNN添加早期退出机制并逐退出点验证，能以直接验证所需时间的一小部分解决约80%的鲁棒性查询，且准确率变化不超过0.3%。

    

    深度神经网络（DNN）的形式化验证面临一个根本性的可扩展性挑战：验证鲁棒性是NP难的，且最先进的工具在现代架构上经常超时。我们提出了帕累托验证方法（Pareto VErification，简称PAVE）：通过为DNN添加早期退出机制并逐个退出点进行验证，我们在直接验证所需时间的一小部分内解决了大部分鲁棒性查询，在我们的评估中约为80%。关键洞察在于，鲁棒性邻域中的许多输入会在较早的层就退出。验证这种早期退出行为通常是一个可处理的问题，使得验证可以在完整网络被分析之前很久就终止。早期退出机制可以在不修改基础网络的情况下添加，并且在我们测试的所有架构上最多只带来0.3%的准确率变化。我们为早期退出网络形式化了新颖的鲁棒性属性，并证明在轨迹稳定性条件下该问题是固定参数可解的。

    arXiv:2512.20755v2 Announce Type: replace-cross  Abstract: Formal verification of deep neural networks (DNNs) faces a fundamental scalability challenge: verifying robustness is NP-hard and state-of-the-art tools often time out on modern architectures. We propose PAreto VErification, or PAVE: by augmenting DNNs with early exits (EEs) and verifying exit-by-exit, we solve a large fraction of robustness queries---roughly 80% in our evaluation---in a small fraction of the time required for direct verification. The key insight is that many inputs in a robustness neighborhood exit at an early layer. Verifying this early-exit behavior is often a tractable problem,allowing verification to terminate long before the full network is analyzed. EEs can be added without modifying the base network, and incur at most a 0.3% accuracy change across our tested architectures. We formalize the novel robustness property for EE networks, prove it is Fixed-Parameter Tractable under a trace stability condition 
    
[^345]: FacEDiT：通过面部动作填充实现说话头部视频编辑

    FacEDiT: Talking Head Video Editing via Facial Motion Infilling

    [https://arxiv.org/abs/2512.14056](https://arxiv.org/abs/2512.14056)

    该论文提出FacEDiT，将说话头部视频的局部编辑建模为面部动作填充这一自监督任务，无需成对的直接监督数据，即可利用普通视频-语音对实现保持身份一致性和边界连续性的局部视频编辑。

    

    对说话头部视频的局部片段进行编辑而无需重新拍摄场景，仍然是一个具有挑战性且研究不足的问题。局部编辑，如插入、删除或替换，需要在保持视频其余部分不变的情况下合成替换片段。这涉及可变时长的局部重写，同时需要保持身份一致性、未编辑区域和边界连续性，而标准的语音驱动生成和唇形同步方法并不能直接解决这些问题。此外，直接监督是不切实际的，因为它需要同一人物和场景的成对视频，且这些视频仅在局部语音片段上有所不同。为此，我们将说话头部视频编辑重新表述为面部动作填充，这是一种自监督预训练任务，利用普通的视频-语音对，从语音和周围动作上下文中恢复被遮蔽的面部动作。我们的关键洞察是，局部视频编辑可以在训练过程中通过遮蔽来模拟。

    arXiv:2512.14056v2 Announce Type: replace-cross  Abstract: Editing a local segment of a talking head video without reshooting the scene remains challenging and underexplored. Local edits, such as insertion, deletion, or substitution, require synthesizing a replacement segment while leaving the rest of the video unchanged. This involves variable-duration local rewriting while preserving identity, unedited regions, and boundary continuity, which standard speech-driven generation and lip synchronization methods do not directly address. Moreover, direct supervision is impractical, as it would require paired videos of the same person and scene that differ only in a local spoken segment. We instead formulate talking head video editing as facial motion infilling, a self-supervised pretext task that recovers masked facial motion from speech and surrounding motion context using ordinary video--speech pairs. Our key insight is that local video editing can be simulated during training by masking 
    
[^346]: StreamGaze：流式视频中的视线引导时序推理与主动式理解

    StreamGaze: Gaze-Guided Temporal Reasoning and Proactive Understanding in Streaming Videos

    [https://arxiv.org/abs/2512.01707](https://arxiv.org/abs/2512.01707)

    StreamGaze是首个评估多模态大语言模型能否在流式视频中利用人类视线信号进行时序推理与主动式响应的基准，通过视线引导的过去、现在和主动式三类任务全面检验流式视频理解能力。

    

    流式视频理解要求模型不仅能够按时间顺序处理连续到来的视频帧，还需要预判用户意图，以满足增强现实（AR）眼镜等现实应用场景的需求。尽管现有的流式基准测试已对时序推理能力进行了评估，但尚无任何基准衡量多模态大语言模型（MLLMs）能否在流式环境中解读或利用人类视线（gaze）信号。为填补这一空白，我们提出了StreamGaze，这是首个专门评估MLLMs在流式视频中利用视线进行时序推理与主动式推理能力的基准。StreamGaze引入了视线引导的过去、现在和主动式任务，对流式视频理解进行全面评估。其中，过去和现在任务要求模型基于查询时刻可获得的观察立即做出响应，而主动式任务则要求模型持续处理不断到来的视频帧，并在合适的未来时机做出响应。为了构建StreamGaze，

    arXiv:2512.01707v4 Announce Type: replace-cross  Abstract: Streaming video understanding requires models not only to process temporally incoming frames, but also to anticipate user intent for realistic applications such as Augmented Reality (AR) glasses. While prior streaming benchmarks evaluate temporal reasoning, none measure whether Multimodal Large Language Models (MLLMs) can interpret or leverage human gaze signals within a streaming setting. To fill this gap, we introduce StreamGaze, the first benchmark designed to evaluate how effectively MLLMs utilize gaze for temporal and proactive reasoning in streaming videos. StreamGaze introduces gaze-guided past, present, and proactive tasks that comprehensively assess streaming video understanding. Past and present tasks require immediate responses based on observations available at query time, whereas proactive tasks require models to continuously process incoming frames and respond at the appropriate future moment. To build StreamGaze,
    
[^347]: 面向长尾人-物交互检测的自适应测试时内存分配

    Adaptive Test-Time Memory Allocation for Long-Tailed Human-Object Interaction Detection

    [https://arxiv.org/abs/2511.18811](https://arxiv.org/abs/2511.18811)

    提出了一种无需训练、即插即用的自适应多样性缓存（ADC）框架，通过频率感知的缓存容量分配与置信度-多样性联合样本选择，自适应地构建测试时缓存，从而有效提升长尾人-物交互检测中稀有交互的识别性能。

    

    长尾人-物交互（HOI）检测仍然具有挑战性，因为稀有交互在训练数据中的表示十分稀疏。尽管测试阶段的观察可以提供额外的视觉证据，但现有的基于缓存的方法通常在各类之间分配统一的内存容量，这限制了其捕捉欠表示交互中有信息量特征的能力。为了解决这一问题，我们提出了自适应多样性缓存（ADC），这是一个无需训练、即插即用的框架，通过频率感知的容量分配和置信度-多样性联合样本选择来自适应地构建测试时缓存。具体而言，频率感知缓存适应利用频率相关的优先级和加权对数分配目标，为欠表示的交互分配更大的缓存容量，并用选定的增强视图填充剩余的缓存槽位。置信度-多样性联合缓存选择则进一步结合预测置信度与样本多样性（摘要在此处被截断）。

    arXiv:2511.18811v3 Announce Type: replace-cross  Abstract: Long-tailed human-object interaction (HOI) detection remains challenging because rare interactions are sparsely represented in training data. Although test observations provide additional visual evidence, existing cache-based methods typically assign uniform memory capacity across classes, limiting their ability to capture informative features of underrepresented interactions. To address this issue, we propose Adaptive Diversity Cache (ADC), a training-free, plug-and-play framework that adaptively constructs test-time caches through frequency-aware capacity allocation and confidence-diversity joint sample selection. Specifically, Frequency-Aware Cache Adaptation assigns larger capacities to underrepresented interactions using frequency-dependent priorities and a weighted logarithmic allocation objective, and fills remaining slots with selected augmented views. Confidence-Diversity Joint Cache Selection further combines predicti
    
[^348]: 推理的认知基础及其在大语言模型中的体现

    Cognitive Foundations for Reasoning and Their Manifestation in LLMs

    [https://arxiv.org/abs/2511.16660](https://arxiv.org/abs/2511.16660)

    该研究构建了涵盖28种认知元素的认知科学分类体系与细粒度评估框架，通过对比18个模型与人类的推理轨迹，发现大语言模型虽能调用人类推理中的认知元素但使用方式不同，并据此开发了可自动搭建成功推理拓扑的测试时推理引导方法以提升模型准确率。

    

    大语言模型（LLMs）能够解决复杂问题，却在更简单的变体上失败，这表明它们获得正确输出的机制与人类推理存在根本差异。为了理解这一差距，我们将认知科学的研究成果综合为一个涵盖28种认知元素的分类体系，包括推理不变量、元认知控制、用于组织推理与知识的表征以及转换操作。我们提出了一个细粒度评估框架，并将其应用于18个模型在文本、视觉和音频模态下产生的19.2万条推理轨迹，以及在一个全新的、与领域无关的数据集上收集的1,346条人类出声思维（think-aloud）轨迹，用于人类与模型的对比。研究发现，模型能够调用人类推理中出现的各类认知元素，但在使用这些元素的方式上与人类存在差异。利用这些规律，我们开发了测试时推理引导方法，能够自动搭建成功的推理拓扑结构，从而提升最强模型的准确率。

    arXiv:2511.16660v3 Announce Type: replace  Abstract: Large language models (LLMs) solve complex problems yet fail on simpler variants, suggesting they achieve correct outputs through mechanisms fundamentally different from human reasoning. To understand this gap, we synthesize cognitive science research into a taxonomy of 28 cognitive elements spanning reasoning invariants, meta-cognitive controls, representations for organizing reasoning and knowledge, and transformation operations. We introduce a fine-grained evaluation framework and apply it to 192K traces from 18 models across text, vision, and audio, alongside 1,346 human think-aloud traces on a new, domain-agnostic dataset for human-model comparison. We find that models are able to deploy cognitive elements seen in human reasoning but differ in how they use them. Leveraging these patterns, we develop test-time reasoning guidance that automatically scaffolds successful reasoning topologies, improving the accuracy of the strongest 
    
[^349]: 使用医学分类数据对大语言模型进行指令微调可提升德语肿瘤诊断的ICD编码准确性

    Instruction-Tuning Large Language Models with Medical Classification Data Improves ICD Coding of German Tumor Diagnoses

    [https://arxiv.org/abs/2510.13624](https://arxiv.org/abs/2510.13624)

    本研究基于ICD-10-GM、ICD-O-3和OPS医学分类目录构建了超过50万个问答对，对Qwen、Llama和Mistral系列开源大语言模型进行指令微调，从而显著提升了德语肿瘤诊断文本的ICD编码准确性。

    

    使用ICD-10-GM和ICD-O-3对肿瘤诊断进行准确编码，对于德国结构化癌症文档记录至关重要。较小的开源权重大语言模型在隐私保护自动化方面具有吸引力，但在德语语境中往往难以达到编码准确性。本研究探讨了基于公开数据集的指令微调能否提高开源权重大语言模型对德语肿瘤诊断文本的编码准确性。评估采用来自本地肿瘤文档系统的已编码诊断作为测试数据。在系统性数据质量评估中，ICD-10编码性能的上限估计为：精确编码为60-79%，部分编码（仅三位字符编码）为81-94%。作为训练数据，研究者基于ICD-10-GM、ICD-O-3和OPS目录创建了超过50万个问答对，并对来自Qwen、Llama和Mistral系列的八个开源权重模型（7-70B参数）进行了微调。ICD-10-GM a…（原文摘要在此处截断）

    arXiv:2510.13624v2 Announce Type: replace-cross  Abstract: Accurate coding of tumor diagnoses with ICD-10-GM and ICD-O-3 is essential for structured cancer documentation in Germany. Smaller open-weight LLMs are appealing for privacy-preserving automation but often struggle with coding accuracy in German-language contexts. This study investigates whether instruction-based fine-tuning on public datasets improves the coding accuracy of open-weight LLMs for German tumor diagnosis texts. The evaluation uses coded diagnoses from the local tumor documentation system as test data. In a systematic data quality assessment, the upper limit for ICD-10 coding performance was estimated at 60-79% for exact and 81-94% for partial (three-character codes only) derivation. As training data, over 500,000 question-answer pairs were created based on the ICD-10-GM, ICD-O-3, and OPS catalogues. Eight open-weight models from the Qwen, Llama, and Mistral families (7-70 B parameters) were fine-tuned. ICD-10-GM a
    
[^350]: FinMultiHopQA：一个基于知识图谱的多跳金融问答基准，面向证据驱动的推理

    FinMultiHopQA: A KG-Grounded Multi-Hop Financial QA Benchmark for Evidence-Driven Reasoning

    [https://arxiv.org/abs/2510.02906](https://arxiv.org/abs/2510.02906)

    该论文提出了FinMultiHopQA基准，包含15,600个定性与定量问题，基于从SEC 10-K申报文件构建的、含1750万+三元组的金融知识图谱FinReflectKG，支持跨章节、跨财年、跨公司的多跳证据推理，并能控制证据呈现方式的变化以实现可复现、可审计的评测。

    

    多跳金融问答通常需要综合分布在申报文件不同章节、不同财年以及同业公司之间的证据。以往的金融问答基准只分别解决了该问题的部分方面，但没有任何一个基准能够同时支持受控的多跳分类体系、文档拓扑结构、基于模体的可复现性以及大规模的溯源可审计性。我们提出了FinMultiHopQA，一个包含15,600个定性与定量问题的基准，其建立在FinReflectKG之上——这是一个从SEC 10-K申报文件中提取的、具有时间索引的金融知识图谱，包含超过1750万条三元组。该构建流程在文档内、跨年度和跨公司三种模式下实例化统计显著的2-3跳带类型知识图谱模体。除了填补这一空白之外，同样的构建方式还使我们能够在保持每个问题底层证据不变的情况下，仅改变其呈现方式——例如将其嵌入嘈杂的上下文中或改变其顺序——从而实现受控的……（原文在此处截断）

    arXiv:2510.02906v2 Announce Type: replace-cross  Abstract: Multi-hop financial question answering often requires composing evidence distributed across filing sections, fiscal years, and peer companies. Prior financial QA benchmarks address parts of this problem individually, but none jointly support controlled multi-hop taxonomies, document topologies, motif-level reproducibility, and provenance-linked auditability at scale. We introduce FinMultiHopQA, a benchmark of 15,600 qualitative and quantitative questions, grounded in FinReflectKG, a temporally indexed financial knowledge graph derived from SEC 10-K filings with 17.5M+ triples. The pipeline instantiates statistically salient 2-3 hop typed KG motifs across intra-document, inter-year, and cross-company regimes. Beyond closing this gap, the same construction lets us hold each question's underlying evidence constant while varying only its presentation - embedding it in noisy surrounding context or changing its order - giving a contr
    
[^351]: ARMs：面向多模态模型的即插即用自适应红队测试智能体

    ARMs: Adaptive Red-Teaming Agent against Multimodal Models with Plug-and-Play Attacks

    [https://arxiv.org/abs/2510.02677](https://arxiv.org/abs/2510.02677)

    提出ARMs自适应红队测试智能体，通过推理增强的多步骤编排自动优化攻击策略，包含11种新颖多模态攻击手段并可即插即用集成17种红队测试算法，实现对视觉语言模型的全面自动化安全评估。

    

    随着视觉语言模型日益受到重视，其多模态接口也引入了新的安全漏洞，使得安全评估既充满挑战又至关重要。现有的红队测试工作要么局限于狭窄的对抗模式集合，要么严重依赖人工工程，缺乏对现实世界中新兴VLM漏洞的可扩展探索。为弥补这一空白，我们提出了ARMs，一个能够对VLM进行系统性全面风险评估的自适应红队测试智能体。给定目标有害行为或风险定义，ARMs通过推理增强的多步骤编排自动优化多样化的红队测试策略，以有效诱导目标VLM产生有害输出。我们提出了11种新颖的多模态攻击策略，涵盖VLM的多种对抗模式（如推理劫持、上下文伪装），并通过模型上下文以即插即用的方式将17种红队测试算法集成到ARMs中。

    arXiv:2510.02677v2 Announce Type: replace  Abstract: As vision-language models (VLMs) gain prominence, their multimodal interfaces also introduce new safety vulnerabilities, making the safety evaluation challenging and critical. Existing red-teaming efforts are either restricted to a narrow set of adversarial patterns or depend heavily on manual engineering, lacking scalable exploration of emerging real-world VLM vulnerabilities. To bridge this gap, we propose ARMs, an adaptive red-teaming agent that systematically conducts comprehensive risk assessments for VLMs. Given a target harmful behavior or risk definition, ARMs automatically optimizes diverse red-teaming strategies with reasoning-enhanced multi-step orchestration, to effectively elicit harmful outputs from target VLMs. We propose 11 novel multimodal attack strategies, covering diverse adversarial patterns of VLMs (e.g., reasoning hijacking, contextual cloaking), and integrate 17 red-teaming algorithms into ARMs via model conte
    
[^352]: 一种用于增强眼底图像病灶分割的混合方法

    Hybrid Approach for Enhancing Lesion Segmentation in Fundus Images

    [https://arxiv.org/abs/2509.25549](https://arxiv.org/abs/2509.25549)

    本文提出一种混合方法，将深度学习模型与传统数学/聚类分割方法相结合，以克服眼底图像数据集分辨率低、标注不一致以及标注数据不足的问题，实现脉络膜痣等眼底病变的精确分割，助力早期诊断工具的开发。

    

    脉络膜痣是眼部常见的良性色素性病变，有较小的风险转化为黑色素瘤。早期检测对于提高生存率至关重要，而误诊或延迟诊断可能导致不良后果。尽管基于AI的图像分析技术不断进步，但在彩色眼底图像中诊断脉络膜痣仍然具有挑战性，特别是对于缺乏专业知识的临床医生而言。现有数据集通常存在分辨率低和标注不一致的问题，限制了分割模型的有效性。本文旨在解决眼底病变精确分割这一挑战，这是开发稳健诊断工具的关键步骤。虽然像U-Net这样的深度学习模型已展现出有效性，但其准确性在很大程度上依赖于标注数据的质量和数量。以往的数学/聚类分割方法虽然准确，但需要大量的人工输入。

    arXiv:2509.25549v3 Announce Type: replace-cross  Abstract: Choroidal nevi are common benign pigmented lesions in the eye, with a small risk of transforming into melanoma. Early detection is critical to improving survival rates, but misdiagnosis or delayed diagnosis can lead to poor outcomes. Despite advancements in AI-based image analysis, diagnosing choroidal nevi in colour fundus images remains challenging, particularly for clinicians without specialized expertise. Existing datasets often suffer from low resolution and inconsistent labelling, limiting the effectiveness of segmentation models. This paper addresses the challenge of achieving precise segmentation of fundus lesions, a critical step toward developing robust diagnostic tools. While deep learning models like U-Net have demonstrated effectiveness, their accuracy heavily depends on the quality and quantity of annotated data. Previous mathematical/clustering segmentation methods, though accurate, required extensive human input
    
[^353]: LSD-3D：具有几何锚定的大规模3D驾驶场景生成

    LSD-3D: Large-Scale 3D Driving Scene Generation with Geometry Grounding

    [https://arxiv.org/abs/2508.19204](https://arxiv.org/abs/2508.19204)

    该论文提出了一种直接生成具有精确几何的大规模3D驾驶场景的方法，弥合了神经重建方法场景多样性受限与扩散模型生成缺乏几何锚定之间的差距，实现了支持物体永久性的因果新视角合成和显式3D几何估计。

    

    大规模场景数据对于机器人学习的训练和测试至关重要。神经重建方法有望从采集的传感器数据中重建大型且具有物理基础的室外场景。然而，这些方法将静态环境固化在场景中，仅允许有限的场景控制——它们在场景和轨迹多样性上受限于用于重建的采集数据。相比之下，利用近期的图像或视频扩散模型生成驾驶数据虽然可以提供控制能力，但代价是失去几何锚定和因果性。在本工作中，我们旨在弥合这一差距，提出了一种直接生成具有精确几何的大规模3D驾驶场景的方法，支持具有物体永久性和显式3D几何估计的因果新视角合成。所提出的方法将代理几何与环境表示的生成相结合……

    arXiv:2508.19204v2 Announce Type: replace-cross  Abstract: Large-scale scene data is essential for training and testing in robot learning. Neural reconstruction methods have promised the capability of reconstructing large physically-grounded outdoor scenes from captured sensor data. However, these methods have baked-in static environments and only allow for limited scene control -- they are functionally constrained in scene and trajectory diversity by the captures from which they are reconstructed. In contrast, generating driving data with recent image or video diffusion models offers control, however, at the cost of geometry grounding and causality. In this work, we aim to bridge this gap and present a method that directly generates large-scale 3D driving scenes with accurate geometry, allowing for causal novel view synthesis with object permanence and explicit 3D geometry estimation. The proposed method combines the generation of a proxy geometry and environment representation with s
    
[^354]: 基于自我反馈强化学习的大语言模型后训练

    Post-Training Large Language Models via Reinforcement Learning from Self-Feedback

    [https://arxiv.org/abs/2507.21931](https://arxiv.org/abs/2507.21931)

    该论文提出了RLSF方法，利用模型自身对答案的内在置信度作为自我生成的奖励来构建合成偏好数据，无需人工标注或标准答案即可通过偏好优化提升大语言模型在推理任务上的可靠性和校准度。

    

    大语言模型（LLMs）经常产生看似合理但校准不佳的答案，这限制了它们在推理密集型任务上的可靠性。最近的研究表明，思维链（CoT）推理路径是预训练大语言模型所固有的，可以通过简单地改变解码过程来引出，其中CoT路径的存在与更高的答案置信度相关。基于这些见解，我们提出了自我反馈强化学习（RLSF），这是一种后训练阶段，利用模型的内在置信度作为自我生成的奖励。通过从冻结的大语言模型生成多个CoT解码束，我们计算每个最终答案跨度的置信度，并相应地对由此产生的推理轨迹进行排序以创建合成偏好。这些偏好随后被用于通过标准偏好优化来微调策略，无需人工标签、标准答案或外部构建的奖励。

    arXiv:2507.21931v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) often produce plausible but poorly-calibrated answers, limiting their reliability on reasoning-intensive tasks. Recent research suggests that Chain-of-Thought (CoT) reasoning paths are inherent in pre-trained LLMs and can be elicited by simply altering the decoding process, where the presence of a CoT path correlates with higher answer confidence. Building on these insights, we present Reinforcement Learning from Self-Feedback (RLSF), a post-training stage that utilises the model's intrinsic confidence as a self-generated reward. By generating multiple CoT decoding beams from a frozen LLM, we compute the confidence of each final answer span and rank the resulting traces accordingly to create synthetic preferences. These preferences are subsequently utilised to fine-tune the policy through standard preference optimisation, requiring no human labels, gold answers, or externally curated rewards. RLSF s
    
[^355]: CoMPAS3D：面向交互式运动的数据集与基准

    CoMPAS3D: A Dataset and Benchmark for Interactive Motion

    [https://arxiv.org/abs/2507.19684](https://arxiv.org/abs/2507.19684)

    该论文提出将具有规范动作词汇和社区公认评判标准的萨尔萨舞作为具身交互评估领域，并构建了CoMPAS3D数据集与基准，以评估交互式动作生成中反应性动作的语义有效性与恰当性。

    

    社交互动型人形机器人旨在通过身体与人类进行交互，并实时适应伙伴的运动、意图和能力。这需要模型既能理解身体如何运动，又能理解这些运动所表达的含义。然而，现有的交互式动作生成评估框架并未评估所生成的反应性动作在共享动作词汇体系中是否有效或恰当：现有框架依赖FID、节拍对齐等运动学指标，无法捕捉这一语义属性，而即兴互动数据集又缺乏细粒度的动作标注。受围棋、国际象棋等游戏计算化评估的启发，我们提出将萨尔萨舞作为一个具身交互评估领域。萨尔萨舞具有交互性和双人配对性，并受动作词汇以及社区公认的评判标准（涵盖时机、音乐性、技术、难度、配合与原创性）的约束。我们预……（原文摘要在此处截断）

    arXiv:2507.19684v3 Announce Type: replace-cross  Abstract: Socially interactive humanoids aim to interact with humans through their bodies, adapting in real time to a partner's movement, intent, and abilities. This requires models that understand both how bodies move and what their movements mean. Yet, existing evaluation frameworks for interactive motion generation do not assess whether generated reactive motions are valid or appropriate within a shared movement vocabulary. Existing frameworks rely on kinematic metrics such as FID and beat alignment that cannot capture this semantic property, while improvised interaction datasets lack fine-grained move annotations. Inspired by computational evaluations of games such as Go and Chess, we propose Salsa as an embodied interaction evaluation domain. Salsa is interactive, dyadic, and governed by a move vocabulary and community-validated judging criteria covering timing, musicality, technique, difficulty, partnering, and originality. We pres
    
[^356]: SOCIA-$\nabla$：文本梯度与多智能体编排相结合的自动化模拟器生成方法

    SOCIA-$\nabla$: Textual Gradient Meets Multi-Agent Orchestration for Automated Simulator Generation

    [https://arxiv.org/abs/2505.12006](https://arxiv.org/abs/2505.12006)

    SOCIA-$\nabla$提出了一种诊断驱动的多智能体框架，通过文本计算图中的局部化文本梯度修复信号，使大语言模型智能体能够迭代地合成、执行、评估、诊断和修复模拟器代码，在多个真实与半合成基准上实现了最先进的模拟器保真度。

    

    本文提出了SOCIA-$\nabla$，一个面向自动化模拟器代码构建的诊断驱动多智能体框架。SOCIA-$\nabla$在文本计算图内迭代优化可执行的模拟器代码，其中由大语言模型驱动的智能体对模拟器进行合成、执行、评估、诊断和修复。基于事实依据的模拟器蓝图将行为残差归因于候选机制及其空间或交互情境，随后借助具有历史感知记忆和约束投影的机制，将其转化为局部化的文本修复信号。我们在真实世界的LLMob个人出行基准和一个受控的半合成地理社会口罩采纳基准上对SOCIA-$\nabla$进行了评估。在分布内模拟、分布外“假设”情景生成以及地理社会扩散等任务中，SOCIA-$\nabla$均实现了最强的整体模拟器保真度。SOCIA-$\nabla$的代码和数据已在https://github.com/ 公开。

    arXiv:2505.12006v5 Announce Type: replace  Abstract: In this paper, we present SOCIA-$\nabla$, a diagnosis-driven multi-agent framework for automated simulator-code construction. SOCIA-$\nabla$ iteratively refines executable simulator code within a textual computation graph, where LLM-driven agents synthesize, execute, evaluate, diagnose, and repair simulators. A grounded simulator blueprint attributes behavioral residuals to candidate mechanisms and their spatial or interaction contexts, which are then converted into localized textual repair signals with history-aware memory and constraint projection. We evaluate SOCIA-$\nabla$ on the real-world LLMob personal-mobility benchmark and a controlled semi-synthetic geo-social mask-adoption benchmark. Across in-distribution simulation, out-of-distribution what-if generation, and geo-social diffusion, SOCIA-$\nabla$ achieves the strongest overall simulator fidelity. SOCIA-$\nabla$'s code and data are publicly available at https://github.com/
    
[^357]: 如何对图像知识蒸馏进行后门攻击

    How to Backdoor Image Knowledge Distillation

    [https://arxiv.org/abs/2504.21323](https://arxiv.org/abs/2504.21323)

    本文提出了一种针对图像知识蒸馏的新型后门攻击方法，通过向蒸馏数据集注入被干净教师模型预测为目标标签的触发图像，使学生在教师保持干净的情况下学到后门，且在10%中毒率下仍然有效。

    

    知识蒸馏被广泛用于将大型教师模型的行为迁移到较小的学生模型。当教师模型是干净的时，知识蒸馏通常被认为是安全的，因为经典的后门攻击依赖于监督训练中的中毒标签和触发器，而蒸馏则是训练学生模型去匹配教师模型的输出。我们表明，当蒸馏数据集本身被投毒时，这一假设可能会失效。我们的攻击注入带有触发器且经过操纵的图像，这些图像会被干净的教师模型预测为攻击者选定的目标标签，从而使学生在教师模型不受影响的情况下学到后门。我们在多种操纵策略下评估了这一威胁，包括针对性的对抗扰动和基于GAN的针对性类别转换，并研究了蒸馏设置如何同时影响准确率和攻击成功率。我们表明，即使在10%的中毒率下，该攻击仍然有效。

    arXiv:2504.21323v3 Announce Type: replace-cross  Abstract: Knowledge distillation is widely used to transfer behavior from a large teacher model to a smaller student. It is often assumed to be safe when the teacher is clean, because classic backdoor attacks rely on poisoned labels and triggers in supervised training, whereas distillation trains the student to match a teacher's outputs. We show that this assumption can fail when the distillation dataset itself is poisoned. Our attack injects triggered and manipulated images that a clean teacher already predicts as an attacker chosen target label, which causes the student to learn a backdoor even though the teacher remains unaffected. We evaluate this threat across multiple manipulation strategies, including targeted adversarial perturbations and targeted GAN based class transitions, and study how distillation settings influence both accuracy and attack success. We show that the attack remains effective even at a 10% poisoning rates. The
    
[^358]: 解析地下水压力：一个可跨时空泛化的模型揭示孟加拉国地下水位与补给量同步下降

    Unraveling Groundwater Stress: A Model Generalizable Across Space and Time Reveals Co-Declining Water Tables and Recharge Across Bangladesh

    [https://arxiv.org/abs/2503.22771](https://arxiv.org/abs/2503.22771)

    该研究提出一种可跨时空泛化的“教师-学生”机器学习框架，将稀疏实地观测与GLDAS卫星数据结合，生成孟加拉国2公里高分辨率地下水埋深图，并揭示了地下水位与地下水补给量正在同步下降。

    

    孟加拉国所依赖的地下水是否正在得到补给？在孟加拉国乃至全球范围内，回答这一问题仍然困难，因为稀疏的实地观测限制了观测能力，而现有的机器学习方法难以在时间和空间上实现泛化。为突破这一瓶颈，我们开发了一个“教师-学生”框架：教师模型从稀疏的年度实地最小（R² = 0.86）和最大（R² = 0.96）地下水埋深观测中学习，生成密集的空间模式；学生模型利用特定位置的特征，将这些模式从随时间变化的25公里分辨率GLDAS地下水储量数据迁移到高分辨率（2公里）的地下水埋深图中。学生模型在2003-2022年间达到R² = 0.93，并能在未见过的年份（留一年交叉验证R² = 0.93）和地点（留一地区交叉验证R² = 0.70）上实现泛化，从而无需同期实地观测即可完成地下水重建。重建结果揭示了广泛存在的……（原文摘要在此处被截断）

    arXiv:2503.22771v2 Announce Type: replace-cross  Abstract: Is Bangladesh replenishing the groundwater it depends on? Answering this question remains difficult in Bangladesh and worldwide because sparse in-situ measurements limit observational capabilities, while existing machine-learning approaches struggle to generalize across time and space. To address this bottleneck, we develop a Teacher-Student framework in which a Teacher accurately learns from sparse annual in-situ minimum (R2 = 0.86) and maximum (R2 = 0.96) depth observations to generate dense spatial patterns, and a Student transfers these patterns from time-varying 25-km GLDAS groundwater storage to high-resolution (2-km) depth maps using location-specific features. The Student achieves R2 = 0.93 over 2003-2022 and generalizes across unseen years (Leave-One-Year-Out R2 = 0.93) and locations (Leave-One-District-Out R2 = 0.70), enabling reconstruction without contemporary in-situ measurements. The reconstruction reveals widespr
    
[^359]: 利用ChatGPT的多模态视觉能力按贫困程度对卫星图像进行排序：推进社会科学研究工具

    Leveraging ChatGPT's Multimodal Vision Capabilities to Rank Satellite Images by Poverty Level: Advancing Tools for Social Science Research

    [https://arxiv.org/abs/2501.14546](https://arxiv.org/abs/2501.14546)

    本文证明ChatGPT可通过多模态视觉能力对卫星图像按贫困水平进行排序，其准确性与基于专家特征训练的模型相当甚至优于直接专家分类，为大语言模型在社会经济研究中的应用开辟了新途径。

    

    本文研究了具有视觉能力的大语言模型（LLM）在分析卫星图像以进行村庄级贫困预测方面的创新应用。尽管大语言模型最初是为自然语言理解而设计的，但它们在执行包括地理空间分析在内的多模态任务方面的适应性，为数据驱动研究开辟了新的前沿领域。通过利用视觉大语言模型的最新进展，我们评估了其从卫星图像中提供可解释、可扩展且可靠的人类贫困洞察的能力。采用成对比较方法，我们证明ChatGPT能够根据贫困程度对卫星图像进行排序，其准确性与基于人类专家识别的图像特征训练的模型相当，且优于先前报告的直接专家财富分类结果。这些发现既凸显了大语言模型在社会经济研究中的潜力，也揭示了其局限性。

    arXiv:2501.14546v4 Announce Type: replace-cross  Abstract: This paper investigates the novel application of Large Language Models (LLMs) with vision capabilities to analyze satellite imagery for village-level poverty prediction. Although LLMs were originally designed for natural language understanding, their adaptability to perform multimodal tasks, including geospatial analysis, has opened new frontiers in data-driven research. By leveraging advancements in vision-enabled LLMs, we assess their ability to provide interpretable, scalable, and reliable insights into human poverty from satellite images. Using a pairwise comparison approach, we demonstrate that ChatGPT can rank satellite images based on poverty levels with an accuracy comparable to a model trained on human-expert-identified image features and better than direct expert wealth classifications reported previously. These findings highlight both the promise and the limitations of LLMs in socioeconomic research, providing a foun
    
[^360]: 有限模型不确定性下的Q学习

    Q-Learning under Finite Model Uncertainty

    [https://arxiv.org/abs/2407.04259](https://arxiv.org/abs/2407.04259)

    提出了一种在有限模糊转移核集合下的鲁棒Q学习算法，证明了其几乎必然收敛到鲁棒最优解，并给出将可观测经验贝尔曼残差与转移核估计误差分离的时间一致高概率误差界。

    

    我们提出了一种面向模型不确定性下马尔可夫决策过程的鲁棒Q学习算法，其中每个状态-动作对都关联着一个由候选转移核构成的有限模糊集。这一有限测度框架支持用户自定义的不确定性模型，包括相互竞争的拟合模型以及压力情景。学习者从每个候选模拟器中观察后继状态及其确定性奖励，而无需显式知晓转移概率或奖励函数。我们证明了所学习的Q函数几乎必然收敛到鲁棒最优解，并推导出一个时间一致的高概率误差证书，该证书将可观测的经验贝尔曼残差与转移核估计误差分离开来。我们还针对有限状态空间上的Wasserstein不确定性模型和参数化不确定性模型，给出了定量的有限模糊逼近结果。数值算例说明了该方法的有效性。

    arXiv:2407.04259v4 Announce Type: replace-cross  Abstract: We propose a robust Q-learning algorithm for Markov decision processes under model uncertainty when each state--action pair is associated with a finite ambiguity set of candidate transition kernels. This finite-measure framework enables user-designed uncertainty models, including competing fitted models and stress scenarios. {The learner observes successor states and their deterministic rewards from every candidate simulator, without requiring explicit knowledge of the transition probabilities or the reward function.} We establish almost sure convergence of the learned Q-function to the robust optimum, and derive a time-uniform high-probability error certificate that separates an observable empirical Bellman residual from transition-kernel estimation error. We also give quantitative finite-ambiguity approximation results for Wasserstein and parametric uncertainty models on finite state spaces. Numerical examples illustrate the 
    
[^361]: 通过谱神经网络和非线性矩阵感知实现隐式正则化

    Implicit Regularization via Spectral Neural Networks and Non-linear Matrix Sensing

    [https://arxiv.org/abs/2402.17595](https://arxiv.org/abs/2402.17595)

    本文在更现实的神经网络背景下探讨了隐式正则化现象，通过研究非线性激活函数的一般类别，严格证明了在矩阵感知问题设置中这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    

    隐式正则化现象近年来引起了人们的兴趣，作为神经网络出色泛化能力的一个基本方面。简而言之，它意味着在许多神经网络中，即使损失函数中没有任何显式正则化器，梯度下降动态也会收敛到一个正则化学习问题的解。然而，已知的试图从理论上解释这一现象的结果主要集中在线性神经网络的设置上，线性结构的简单性对现有论据特别关键。本文在更现实的神经网络和一般非线性激活函数的背景下探讨了这一问题，并在矩阵感知问题的设置中严格证明了这些网络的隐式正则化现象，同时提供了严格的速率保证，确保梯度的指数级快速收敛。

    arXiv:2402.17595v1 Announce Type: cross  Abstract: The phenomenon of implicit regularization has attracted interest in recent years as a fundamental aspect of the remarkable generalizing ability of neural networks. In a nutshell, it entails that gradient descent dynamics in many neural nets, even without any explicit regularizer in the loss function, converges to the solution of a regularized learning problem. However, known results attempting to theoretically explain this phenomenon focus overwhelmingly on the setting of linear neural nets, and the simplicity of the linear structure is particularly crucial to existing arguments. In this paper, we explore this problem in the context of more realistic neural networks with a general class of non-linear activation functions, and rigorously demonstrate the implicit regularization phenomenon for such networks in the setting of matrix sensing problems, together with rigorous rate guarantees that ensure exponentially fast convergence of gradi
    

