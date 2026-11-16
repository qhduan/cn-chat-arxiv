# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [WeightBench: Evaluating Language Models as Weight-Space Programmers](https://arxiv.org/abs/2611.18553) | 该论文提出了WEIGHTBENCH基准，通过五大赛道的42项权重空间编程任务系统评估语言模型直接读取、修复和编写神经网络参数的能力，发现前沿模型已能读取小型网络并编写可用的控制策略，但尚无法修补或压缩大型语言模型。 |
| [^2] | [Imitation Bootstrapped Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.18543) | 该论文提出IBMARL框架，将模仿引导强化学习扩展到多智能体协作场景，通过轮替行为克隆训练模仿策略，并利用集中式评论家集成在模仿策略与RL策略的动作提议中进行选择，从而克服了单演示者难以同时操控多个智能体以及演示质量有限的难题。 |
| [^3] | [Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond](https://arxiv.org/abs/2611.18542) | 该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。 |
| [^4] | [ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL](https://arxiv.org/abs/2611.18533) | 提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。 |
| [^5] | [Alignment by Socialization: Learning Tradeoffs Under Selection](https://arxiv.org/abs/2611.18531) | 该论文提出“社会化”这一对齐机制：在进化选择压力下，学习型智能体会收敛到环境所奖励的任务完成与行为规范之间的权衡；当由人类提供奖励时，智能体的选择便能反映人类判断并经受住后续的选择压力。 |
| [^6] | [Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination](https://arxiv.org/abs/2611.18514) | 本文首次将机制可解释性系统应用于网页智能体幻觉失败分析，发现线性探针可在生成前预测幻觉（0.80 AUC）且引导能纠正85%的幻觉，但由于对提示格式敏感，这只是一种部分解释。 |
| [^7] | [MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading](https://arxiv.org/abs/2611.18513) | MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。 |
| [^8] | [Agentic AI for Discovering Fault-Tolerant Quantum Circuits](https://arxiv.org/abs/2611.18504) | 该论文提出将容错量子线路的设计问题表述为智能体式进化程序合成，利用大型语言模型在电路验证器引导下自动进化生成用于逻辑态制备、逻辑门设计和码切换等任务的容错克利福德线路，从而取代传统的手工设计方法。 |
| [^9] | [ARCHER: From Rationale Imitation to Executable Reasoning in Small Language Models](https://arxiv.org/abs/2611.18494) | ARCHER提出两阶段训练框架，先将自由格式的思维链编译为包含计算器调用与验证的类型化动作轨迹，再通过非对称分层强化学习优化策略，使小型语言模型学会生成可执行、可审计的数学推理轨迹。 |
| [^10] | [TradeGauntlet: Evaluating Compliance Controls for Autonomous Trading Agents](https://arxiv.org/abs/2611.18493) | 论文提出 TradeGauntlet——一个可回放的限价订单簿模拟器，用于评估自主交易代理的合规控制，结果表明仅凭零警报不足以证明智能体合规，且对抗性搜索能发现规避现有行为规则的拆分订单策略。 |
| [^11] | [Interactive Program Induction with Language Model Priors](https://arxiv.org/abs/2611.18485) | 本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。 |
| [^12] | [From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration](https://arxiv.org/abs/2611.18484) | 本文通过结构化范围综述指出，AI远程患者监测正迈向具身智能体机器人护理，但成熟的智能体护理尚未实现，未来需要以具身化、人类监督、治理和安全可互操作的护理架构为基础，实现从监测到行动的转变。 |
| [^13] | [Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation](https://arxiv.org/abs/2611.18471) | 本文提出“验证迁移”（Verification Transport）这一可靠性测试方法，通过在保持任务、轨迹、结果等条件不变的情况下构建原生、抽象和恢复三种匹配视图，测量LLM验证器的判断在经历工具组合、委托边界、历史压缩等抽象化后是否仍能保持稳定可靠。 |
| [^14] | [Typing Matters: Muscle, Language and Typing Behaviour in Live Keystroke Decoding from sEMG](https://arxiv.org/abs/2611.18464) | 该论文提出一种基于状态空间模型（SSM）的完全因果、零前瞻sEMG打字解码器，将肌肉信号、语言结构与打字行为三者联合建模，实现手部持续移动时的实时按键预测。 |
| [^15] | [When to Trust Simulated Patient Responses:Support-Gated World Models for Vasopressor Decision Support](https://arxiv.org/abs/2611.18458) | 该论文提出支持门控世界模型决策支持方法，利用行为模型估计的动作支持度与动力学集成模型的分歧度进行联合门控，在模拟预测不可靠时主动隐藏预测或弃权，从而在eICU-CRD数据上以62%覆盖率将MAP预测误差从9.2 mmHg降至7.3 mmHg。 |
| [^16] | [Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles](https://arxiv.org/abs/2611.18453) | 该论文提出了一种仅需黑盒访问的两级“集成之集成”方法，通过对固定提示词随机解码构成内层集成、对提示词进行语义扰动构成外层集成，并利用全协方差定律在嵌入空间中精确分解大语言模型的偶然性不确定性与认知性不确定性。 |
| [^17] | [Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements](https://arxiv.org/abs/2611.18439) | 该研究通过32名跨学科专家参与的参与式研讨会，揭示了将欧盟《人工智能法案》义务转化为技术要求时在可测量性、责任归属与人工监督实践、以及合规证据在动态运行条件下的有效性方面反复出现的张力。 |
| [^18] | [Towards Explainable Sleep Stage Classification using Single-Channel Electroencephalogram](https://arxiv.org/abs/2611.18434) | 该论文扩展了SleepXAI睡眠分期框架，系统比较五种XAI方法，并以AASM定义的脑电频段的Spearman秩相关评估归因保真度，实现了基于单通道脑电、兼顾分类性能与临床可解释性的睡眠阶段分类。 |
| [^19] | [LECAT: Measuring LLM Category Learning from Pretraining Data](https://arxiv.org/abs/2611.18419) | 本文提出 LeCat 两阶段框架，一方面评估大语言模型在记忆、泛化和灵活运用三个层面上的类别知识学习能力，另一方面通过追踪预训练文本中不同的“知识表达”形式来揭示这类知识在预训练中的习得机制。 |
| [^20] | [CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2611.18417) | 提出CoPre-MoE方法，通过为每个任务先学习任务特定MoE专家，再对跨任务冗余专家进行整合与保留，在避免灾难性遗忘的同时有效抑制多模态持续指令微调中的跨任务参数冗余和参数线性增长。 |
| [^21] | [Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent](https://arxiv.org/abs/2611.18397) | 提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。 |
| [^22] | [Smart Home Agents Make Capable but Unpredictable Grid Operators](https://arxiv.org/abs/2611.18395) | 本文首次对智能家居代理在完整停电过程中的韧性进行分析，证明其可以在停电时充当电网运营商，同时识别出多种新的故障类别，为提升电网边缘整体韧性奠定了基础。 |
| [^23] | [VelServe: Taming Decode Jitters in LLM Inference with Tail-Calibrated Scheduling](https://arxiv.org/abs/2611.18392) | VelServe通过利用解码步骤自身迭代延迟的实测尾部数据来动态校准每步的预填充工作量，有效驯服LLM推理中的解码抖动，从而满足TTFT和TBT的延迟服务级目标。 |
| [^24] | [AI-Based Multi-View Echocardiographic Image Synthesis using Dynamic Shape Control](https://arxiv.org/abs/2611.18375) | 该论文提出了一种基于动态全心脏网格形状控制的两阶段AI生成框架，能够合成具有跨心腔多视角一致性的超声心动图序列，突破了现有方法局限于左心室形状监督和特定心脏视角的限制。 |
| [^25] | [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372) | 这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。 |
| [^26] | [Envisioning Agentic Robotics: A New Era of Robotic Systems](https://arxiv.org/abs/2611.18359) | 本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。 |
| [^27] | [VLAN: Vision-Language Accessible Navigation](https://arxiv.org/abs/2611.18346) | 提出首个面向多种身体形态（人形/四足机器人及视觉或行动障碍人士）的大规模视觉-语言无障碍导航基准VLAN，通过无障碍对齐指标评估导航的物理安全性与用户特定约束的契合度。 |
| [^28] | [Causal AI-Driven Secure mmWave Beam Alignment with Defensive Variational Autoencoders](https://arxiv.org/abs/2611.18321) | 该论文提出了一种将因果发现与防御性变分自编码器相结合的因果感知深度学习框架，在提升毫米波波束对准的可解释性与泛化能力的同时，有效抵御对抗攻击。 |
| [^29] | [Experts Across Schemas: Agent-Guided Mixture-of-Relational-Experts for Cross-Database Generalization](https://arxiv.org/abs/2611.18315) | 该论文提出MoREGT关系专家混合图Transformer，通过基于列元数据的专家路由和标签无感知的LLM智能体自动生成派生特征，证明更好的架构设计与目标数据库特征相结合能够以远少于现有方法的预训练数据实现跨数据库泛化性能。 |
| [^30] | [AI for Parameter Selection in Sample-based Quantum Diagonalization](https://arxiv.org/abs/2611.18307) | 该研究证明机器学习模型可以根据哈密顿量直接预测量子运行时间和近似误差，从而自动为SKQD算法选择最优参数配置，解决了基于样本的量子对角化方法中缺乏正式参数选择方法的问题。 |
| [^31] | [When evaluation becomes intervention in clinical AI](https://arxiv.org/abs/2611.18297) | 临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。 |
| [^32] | [PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review](https://arxiv.org/abs/2611.18293) | 该论文提出了PrivacyPR数据集，包含4,131个带LLM生成的隐私相关性和GDPR原则标注的真实GitHub拉取请求，为评估编码智能体能否在代码审查中发现法律要求的隐私缺陷提供了首个大规模基准。 |
| [^33] | [Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering](https://arxiv.org/abs/2611.18274) | 该论文将基于稀疏自编码器的激活引导重构为固定预算下的逐词元分配问题，并提出无记忆的对比特征投影（CFP）与轨迹感知的闭环自适应伯努利引导（CLABS）两种门控机制，通过只在恰当的词元和时机施加干预来避免分布偏移、保持生成质量。 |
| [^34] | [From Bytes to Behavior: Request-Level Attribution from Live GPU Hidden States](https://arxiv.org/abs/2611.18272) | 该论文提出了一种无需修改进程、无需预知模型架构或权重的取证方法，可直接从运行中的GPU进程提取逐层残差流激活值，实现对无法重启或修改的大语言模型服务的请求级行为归因分析。 |
| [^35] | [ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization](https://arxiv.org/abs/2611.18252) | 提出了一种基于惩罚的极小-极大-极小优化公式（M³问题）及单循环一阶算法 ParetoM³，将偏好优化与弱帕累托最优性统一在单一目标中，并证明了算法能收敛到近似弱帕累托最优的ε-稳定点。 |
| [^36] | [Dancing Tokens on the Vocabulary Network from LLM Generation](https://arxiv.org/abs/2611.18248) | 该研究通过构建LLM词表嵌入的相似性网络，揭示知识空间的碎片化拓扑在预训练阶段即已确定（蒸馏几乎不改变它），且文本生成并非沿该网络的游走，而是集中访问网络内部、并在网络深处才最终确定的动态过程。 |
| [^37] | [Probabilities of Causation in Nondeterministic Causal Models](https://arxiv.org/abs/2611.18245) | 该论文提出了一个面向非确定性因果模型的通用识别框架，通过矩约束与约束优化推导出因果概率的尖锐解析界，所得识别区域通常比确定性模型下的经典结果更具信息量。 |
| [^38] | [RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation](https://arxiv.org/abs/2611.18244) | 提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。 |
| [^39] | [Personalization of Basic and Moral Values Guiding LLM-Controlled Assistant Robots](https://arxiv.org/abs/2611.18230) | 该研究通过SVS、PVQ-21和MFQ三种心理测量工具发现，用户自身的价值观与他们期望助理机器人体现的价值观存在差异，因此主张对LLM控制的助理机器人进行基本与道德价值观的个性化设定，并基准测试了十五个当代大语言模型。 |
| [^40] | [From Pen-and-Paper to Machine-Checked Proofs: Lean for System Identification](https://arxiv.org/abs/2611.18220) | 本文向系统辨识与控制领域的研究者介绍Lean 4证明助手，通过可辨识性、集合不变性及预测误差辨识中经典引理的形式化示例，展示机器验证证明如何弥补传统纸笔证明难以逐步核验的不足。 |
| [^41] | [T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer](https://arxiv.org/abs/2611.18199) | T2ST 框架通过引入丰富的时间序列数据辅助训练，并从冻结的时间序列基础模型中迁移知识，在时空观测稀缺的情况下实现数据高效的时空基础模型训练。 |
| [^42] | [Who Should Be on the Panel? Task-Conditioned Persona Selection for LLM Trading Agents](https://arxiv.org/abs/2611.18150) | 本文提出FinPanel系统，其核心创新在于根据每只股票的交易特征动态选择合适的LLM交易员角色组合，取代传统多智能体系统中固定角色阵容的做法，使智能体团队与具体交易任务相匹配。 |
| [^43] | [Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries](https://arxiv.org/abs/2611.18140) | 本文提出了一种由大语言模型赋能的智能体闭环框架，可自适应地控制锂电池快充，在缩短充电时间的同时兼顾抑制电池老化与安全性，克服了传统协议、模型优化和常规数据驱动控制方法的局限。 |
| [^44] | [PUMA-Former: Uncertainty-Aware Agentic Computation Offloading in MEO-LEO Satellite Networks With Incomplete Information](https://arxiv.org/abs/2611.18139) | 该论文提出PUMA-Former框架，在不完备信息条件下通过不确定性感知的智能体机制协调MEO-LEO卫星网络中的协作计算卸载，利用潜在延迟注意力与队列误差感知融合来平衡已完成工作量、服务延迟和执行成本。 |
| [^45] | [JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams](https://arxiv.org/abs/2611.18130) | 该论文提出了JFinBench——一个基于2026年最新版日本资格考试构建、涵盖会计/税务/理财/银行领域的可持续更新金融评测基准，并通过人在回路的智能体流水线大幅降低了纳入新试题、保持基准时效性的成本。 |
| [^46] | [Optimizing Drilling Trajectories in Multi-Material Structures Using Quantum Annealers](https://arxiv.org/abs/2611.18121) | 本文提出一种基于D-Wave Leap混合求解器的量子-混合优化方法，将多材料结构的钻孔轨迹规划建模为包含材料分组、刀具磨损等工业约束的路径优化问题，以最小化总能耗，并在10至55孔的实例上验证了该方法的有效性。 |
| [^47] | [Interpretable weight-space programming builds transparent transformers and brain encoding models](https://arxiv.org/abs/2611.18114) | 该论文提出可解释权重空间编程（iWSP），让LLM智能体在不使用反向传播的情况下，通过显式编写权重和特征代码来迭代构建透明的Transformer和大脑编码模型，从而实现真正可解释的科学建模。 |
| [^48] | [An In-Silico Audit of Topographic Organization in a Multimodal Brain-Response Model](https://arxiv.org/abs/2611.18105) | 该论文提出了一套计算机模拟审计方法（映射刺激、拟合流程及已知真值的正负对照模型），用于检验多模态脑响应模型 TRIBE v2 能否再现感觉皮层的经典拓扑组织，结果发现该模型缺失对侧半视野响应和极角映射等视网膜拓扑特征。 |
| [^49] | [EcoVGGT: Pinned-Key Inference for Efficient and Streamable Feed-Forward 3D Reconstruction from Long Videos](https://arxiv.org/abs/2611.18099) | EcoVGGT 提出了一种无需训练的推理框架，仅用 31 个均匀分布的帧构成固定键集并配合场景缓存机制，将跨帧注意力成本从随帧数二次方增长降为每帧恒定成本，实现了长视频的高效、可流式的前馈三维重建。 |
| [^50] | [Do Machines Learn Deviance From Their Peers? A Randomized Test of Differential Association With AI Agents](https://arxiv.org/abs/2611.18098) | 该研究巧妙利用AI智能体可被随机分配同伴的特性，首次通过随机化实验为差异交往理论提供了因果证据：AI智能体会从声称作弊的“越轨同伴”那里习得作弊行为，而从“诚实同伴”那里显著减少作弊。 |
| [^51] | [AG-JEPA: Anatomically Guided Joint Embedding Predictive Architecture for Neurodegenerative Representation Learning](https://arxiv.org/abs/2611.18097) | 该论文提出AG-JEPA，通过在JEPA预训练中引入与图像块token交互的解剖学token来学习AD相关脑区的区域特定表征，并进一步提出Prog-AG-JEPA将疾病进展轨迹建模为表征空间中的条件化潜在位移。 |
| [^52] | [Two-Clustering Regime of Token Dynamics in Causal Attention](https://arxiv.org/abs/2611.18088) | 本文部分证明了因果注意力动力学在值变换最大特征值为正时收敛为与主导特征向量对齐的双聚类的猜想，并揭示了高维Token轨迹坍缩到由少数主导方向支配的二维结构的严格降维现象。 |
| [^53] | [TelePhysica: Agentic Reconstruction of Physically Interactive Scenes](https://arxiv.org/abs/2611.18078) | TelePhysica提出了一个闭环框架，通过视觉-语言编码智能体和两阶段验证循环，从单张照片重建可执行的多模式物理世界，克服了扩散模型缺乏持久3D力学平衡与单次程序化重建场景易坍塌的问题，实现有状态的3D交互和基于轨迹的视频合成。 |
| [^54] | [A$^3$: Adaptive Audio Agent Framework for Collaborative Edge Computing](https://arxiv.org/abs/2611.18064) | A³框架通过在服务器部署完整流水线、在设备端保留后备智能体，并利用链路感知截止时间动态对冲，实现了链路健康时媲美纯服务器方案的性能，同时保证完全断网时仍能可靠响应。 |
| [^55] | [RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime](https://arxiv.org/abs/2611.18035) | RL$^{\mathrm{RT}}$通过将执行器与学习器的耦合变为可调参数（刷新界限与松弛感知的GPU许可），在共享嵌入式GPU上实现了高效的端侧强化学习，将16.8M参数策略的截止期限错过率从85%-100%降至1%以下。 |
| [^56] | [Understanding the limits of snapshot-based fault diagnosis in distribution networks](https://arxiv.org/abs/2611.17998) | 本文通过系统分离分类器、特征、测量误差和故障电阻等误差来源，首次揭示了基于快照的配电网络故障诊断的性能瓶颈在于测量误差（尤其是高阻抗故障）而非分类器本身。 |
| [^57] | [LACCO: Local Accumulation with Computation--Communication Overlap](https://arxiv.org/abs/2611.17991) | 提出LACCO两阶段算法，在不引入外层更新陈旧性的前提下实现计算与通信重叠并保持各副本更新一致性，在语言模型预训练中持续优于Eager和Delayed DiLoCo等通信重叠基线。 |
| [^58] | [Interactive deep-learning based interface for brain tumour segmentation](https://arxiv.org/abs/2611.17983) | 本文提出了一种默认全自动运行但允许人工轻松校正的深度学习交互式脑肿瘤分割系统，从而能够为所有患者确保足够高的分割精度。 |
| [^59] | [TotalSegmentator Aorta Report: Automated Aortic Diameter Measurement at 11 Standardized Landmarks on CT Images](https://arxiv.org/abs/2611.17968) | 该研究开发并验证了一种全自动开源工具，可在CT图像上遵循指南推荐的11个标准化解剖标志点自动测量主动脉直径，与人工参考标准的平均绝对误差仅为1.06毫米。 |
| [^60] | [From Images to Grounded Edits: A Real-World Benchmark for Chinese Grammatical Error Correction in Handwritten Essays](https://arxiv.org/abs/2611.17966) | 该论文提出了定位式中文语法纠错新任务及首个基于真实K-12手写作文图像的基准G-CGEC-Eval，要求模型同时预测纠错内容及其在图像上的位置，弥合了传统文本级语法纠错与现实教育场景之间的差距。 |
| [^61] | [WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents](https://arxiv.org/abs/2611.17955) | WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。 |
| [^62] | [The Transformer Paradigm: Dynamic Processing and a New Principle of Inference](https://arxiv.org/abs/2611.17953) | 该论文提出Transformer革命本质上是一种基于“动态构建变换”的新推理范式（即先适配先前习得的变换再应用，通过输出-权重互连实现，称为序列级交互式动态并行处理SIDPP），而非深度学习的简单量级扩张，该框架可解释强提示敏感性并开启可解释性、控制与对齐等研究方向。 |
| [^63] | [Toward Interference-Agnostic RF Source Separation](https://arxiv.org/abs/2611.17950) | 本文提出在所有信号-干扰对上训练单一统一分离器，测试时无需识别干扰类型，就能以单个专用模型的规模达到接近全部专用模型的分离性能，并在面对未训练过的干扰时避免信号质量恶化。 |
| [^64] | [Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent](https://arxiv.org/abs/2611.17939) | 论文对一款现成发布的金融研究智能体进行系统审计，发现其基准分数不稳定且运行成本波动高达17%，并按指令、工具、模型等责任层对失败轨迹进行归因分类，其诊断结果与人类盲评者一致率达94%，进而提出可审计的针对性干预方法。 |
| [^65] | [Measured from the feed: the unsustainable growth of AI video generation](https://arxiv.org/abs/2611.17938) | 该研究通过实测两个商业文生视频平台的年度产出并建立前瞻性碳足迹模型，发现若使用量持续增长，尽管有技术进步，到2030年排放仍将激增近75倍，因此减排必须同时依靠限制视频生成增长和实现大幅效率提升。 |
| [^66] | [Tracing Actual Causes with Counterfactual Witness Maps](https://arxiv.org/abs/2611.17937) | 该论文提出“见证目标”与祖先干预文法以构造有限的有向无环见证映射，并给出首个可靠且完备的枚举算法TrAC，用于找出无环结构因果模型中给定情境下的所有实际原因。 |
| [^67] | [OutfitterBench: Evaluating LLM Agents on Long-Horizon Resource Allocation and Commitment Coordination](https://arxiv.org/abs/2611.17933) | 本文提出OutfitterBench基准，让大语言模型智能体在200天内管理一个户外装备租赁业务，以评估其在自身承诺形成未来约束的长周期场景中进行资源分配与承诺协调的能力，实验显示多数前沿模型难以在保持计划可行性的同时重新评估资源分配。 |
| [^68] | [MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration](https://arxiv.org/abs/2611.17931) | 提出首个面向真实矿产勘探场景的MineralExp基准，用于评估LLM智能体自动化地球化学实验设计的能力，并解决了实验方法多样性与主观性高、开放式方案难以客观评分两大难题。 |
| [^69] | [Evaluating Large Language Models for Scientific Paper Search: Retrieval Success and Inference Latency on Mac Studio and NVIDIA A100](https://arxiv.org/abs/2611.17929) | 本研究在Mac Studio与NVIDIA A100上评估了八个开放权重大语言模型执行科学论文网络辅助检索的表现，发现检索成功率主要取决于模型选择与查询的具体性，A100推理速度更快，但Mac Studio的统一内存可支持更大的模型。 |
| [^70] | [What Balanced Accuracy Cannot Identify in Neural-Signal Decoding under Test-Time Adaptation](https://arxiv.org/abs/2611.17925) | 本文揭示在测试时自适应的神经信号解码评估中，平衡准确率等总体指标存在盲区——仅报告单一类别的解码器可获得与正常50%平衡解码器相同的分数，且占样本约十分之一的类别在112个受试者单元中有26-47个完全无法被预测。 |
| [^71] | [RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation](https://arxiv.org/abs/2611.17921) | RecMem提出一种循环记忆压缩框架，将长交互历史压缩为固定数量的记忆槽，使内存与计算开销不随历史长度增长，同时通过推荐监督与流一致性训练保证预测信息保真与不同处理顺序下的一致性。 |
| [^72] | [Condensing Law of Security Information](https://arxiv.org/abs/2611.17918) | 本文指出缩放定律在安全领域失效——模型越强越会在下游安全模块察觉前“悄悄修复”异常输入，并首次将其形式化为认知能力与安全信息保持率之间的权衡，推导出三个信息论界。 |
| [^73] | [When Cooperation Must Change: Semantic Strategy Unlearning in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.17917) | 本文提出了多智能体强化学习中的“语义策略遗忘”新问题：当部署需求变化时，需在保留有用合作的前提下，移除可由不同智能体以角色不变方式重现的特定协调约定。 |
| [^74] | [SWIFT: Sample-Wise Instance Fine-Tuning of Time-Series Foundation Models for Forecasting](https://arxiv.org/abs/2611.17915) | SWIFT 提出了一种样本级实例微调框架，利用目标时间序列自身历史中的结构先验，无需额外数据集即可对时间序列基础模型进行对用户透明的细粒度微调，显著降低了模型适配的应用门槛。 |
| [^75] | [Adversarially Attacking Symbolic Vocabulary Vulnerabilities In LLM Planners](https://arxiv.org/abs/2611.17910) | 本文提出了一种名为Symbolic Swapper的基于LLM的对抗攻击模型，通过自动生成符号词汇被混淆的规划任务，揭示了LLM智能体在符号解释与规划结合方面仍然存在的关键弱点。 |
| [^76] | [More Genes, Faster Inference: Efficient Large-Panel Spatial Transcriptomics Prediction from Histology](https://arxiv.org/abs/2611.17899) | 本文提出GeneFlow框架，将流匹配生成建模与结构化多模态融合相结合，直接从组织学图像预测空间基因表达，突破了以往方法局限于固定小基因面板、架构复杂的瓶颈，实现了高效的大规模基因面板空间转录组预测。 |
| [^77] | [Securing Multi-Agent Systems by Design](https://arxiv.org/abs/2611.17883) | 本文提出了响应操纵攻击以揭示多智能体系统面对恶意响应的安全隐患，并进一步提出基于响应间语义相似性的安全聚合规则SecMAS来有效防御此类攻击。 |
| [^78] | [Photogrammetric 3D Model-Based Animal Behavior Classification](https://arxiv.org/abs/2611.17881) | 本文提出一种两阶段动物行为分类方法，先通过摄影测量重建三维身体与物体几何结构再进行行为分类，从而使二维到三维逆问题的模糊性显式化且可被纠正。 |
| [^79] | [Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents](https://arxiv.org/abs/2611.17880) | Soteria 通过在工具调用前对分层计划进行形式化验证并在运行时强制执行规范，使 LLM 智能体在不牺牲任务效用的情况下保证安全性，并在不安全动作被阻止时借助已验证计划提供有原则的恢复指导。 |
| [^80] | [From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning](https://arxiv.org/abs/2611.17879) | 该论文建立了多步推理中专家共享的理论框架，通过权衡方差节省与偏差推导出共享何时有效的精确准则，并揭示了一个关键现象：共享专家可能降低局部预测误差却反而增大最终输出误差。 |
| [^81] | [DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery](https://arxiv.org/abs/2611.17878) | DySENSE提出了一个联合多模态生成式AI框架，通过解耦长期静态城市形态与短期动态能耗，基于卫星影像与土地利用条件生成空间对齐的月度建筑能耗地图，填补了现有研究只能预测已有城市而无法为未来城市生成动态能源数据的空白。 |
| [^82] | [OmniScore-Bench: Evaluating Vision-Language Models on Musical Score Reading](https://arxiv.org/abs/2611.17870) | OmniScore-Bench是一个包含3000个问题、覆盖六大方面和24种任务子类型的综合基准，用于系统评估视觉语言模型在乐谱阅读上的光学音乐识别能力，揭示了各模型从符号定位到证据关联问答等任务中的优势与瓶颈。 |
| [^83] | [UltraBERT-JEV: Encoder models for System One Applications](https://arxiv.org/abs/2611.17867) | 提出UltraBERT-JEV编码器模型，通过单次前向传播直接从多个选项中选择答案而无需生成文本，配合“是-否”差值打分等微调方案，准确率最高达72.1%并超越同数据训练的0.6B解码器模型，同时推理速度快一倍以上（29毫秒对比68毫秒）。 |
| [^84] | [Active Learning of Temporal Specifications through Distinguishing Queries](https://arxiv.org/abs/2611.17866) | 提出了一种LTLf时序规约的主动学习框架，教师只需标注轨迹并用自然语言回答问题，学习者通过维护枚举版本空间并生成区分性查询来高效学习目标规约，避免了传统方法中等价查询、反例构造和搜索重启的开销。 |
| [^85] | [FlexCast: Flexible Context Learning for Time Series Foundation Models](https://arxiv.org/abs/2611.17862) | FlexCast通过对时间序列补丁进行可变长度掩码预测的预训练框架，解耦了上下文条件化与自回归这两个因素，使同一个预训练模型能够灵活支持单步预测、置信度迭代细化和自回归式生成等多种推理方式。 |
| [^86] | [AI-aided Decision-Making in Education - Uncertainty, Explainability, and Human Responsibility](https://arxiv.org/abs/2611.17858) | 本文提出不确定性、可解释性与人类责任三大原则是人工智能在教育领域合理应用的基石，强调需正视AI决策的固有不确定性，并通过可解释性与人类监督保障其可信度。 |
| [^87] | [DECODE: Diagnosing Interpretability Failures in Automated Interpretation Process](https://arxiv.org/abs/2611.17856) | 提出DECODE可视分析系统，通过将诊断流程分解为三个渐进阶段，重新引入人类监督，帮助隔离和定位自动化解释过程中可解释性失效的来源。 |
| [^88] | [Tracing Uncertainties of Extreme El Ni\~{n}o Prediction to Mesoscale Ocean Currents](https://arxiv.org/abs/2611.17854) | 该研究将条件非线性最优扰动（CNOP）方法应用于可微分海洋AI模型，首次揭示了极端厄尔尼诺预测不确定性可追溯至初始海洋状态中的中尺度洋流误差，且这些误差的来源具有强烈的事件依赖性。 |
| [^89] | [Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors](https://arxiv.org/abs/2611.17846) | 本研究在五组新旧模型对上，使用多种基于方向的消融方法与 ARA 在统一实验条件下进行系统对比，发现拒绝移除的难度并未随着模型版本的更新而普遍增加。 |
| [^90] | [V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations](https://arxiv.org/abs/2611.17831) | 该论文提出V-JEPA-LM，通过定位V-JEPA冻结特征中跨视频共享的动态子空间，并利用像素变换构建反事实监督，使视频语言模型能够真正读取编码器中蕴含的场景动态信息（如速度），而非仅仅检索文本对齐的静态属性。 |
| [^91] | [SceneAgent: A Multi-Agent Framework for Portable Game Scene Generation](https://arxiv.org/abs/2611.17821) | SceneAgent是一个多智能体框架，通过规范的场景上下文、后端适配器以及从人工批准轨迹中提炼的可复用技能，实现了游戏场景构建过程与经验在不同游戏引擎后端间的跨平台迁移复用。 |
| [^92] | [Aligning AI Teams](https://arxiv.org/abs/2611.17820) | 该研究揭示了单个LLM智能体的对齐训练难以迁移到多智能体团队的机制，发现责任分散是安全差距的主要成因，而相互批评、群聊协调等常见自然干预措施无法可靠地恢复团队的对齐行为。 |
| [^93] | [Causally Constrained and Uncertainty-Aware Cardiopulmonary Risk Prediction](https://arxiv.org/abs/2611.17819) | 提出因果包络智能体（CE-Agent）框架，通过将流行病学证据转化为显式因果约束并融合CT表型与闭式不确定性传播，实现对肺癌与心血管疾病的联合风险预测，并能快速处理缺失数据。 |
| [^94] | [G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands](https://arxiv.org/abs/2611.17814) | G-CAST提出了一套受治理的持续自训练系统，使电商LLM智能体能够在用户需求漂移和执行框架不断演进的条件下自动持续适应，从而摆脱滞后且高成本的人工周期性再训练。 |
| [^95] | [CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.17808) | CarryOn是一个轻量级上下文管理器，通过学习的选择器和自适应保留策略，将紧凑的执行状态与细粒度的来源链接证据相结合，无需额外LLM调用即可为长程LLM智能体实现可靠的上下文管理。 |
| [^96] | [Can Agents Keep Up with Time? An Empirical Analysis of Temporal Adaptation After a Conversational Hiatus](https://arxiv.org/abs/2611.17806) | 本文提出“时间状态感知与运用”这一新能力概念，通过设计会话重燃任务（固定先前对话、仅变化后续日期），实证考察长时程智能体在对话中断后能否感知时间流逝并据此调整建议，填补了现有评估只关注事实记忆与显式状态更新的空白。 |
| [^97] | [Multimodal Person Concept Model for Person Identification under Varying Observation Conditions](https://arxiv.org/abs/2611.17802) | 本研究提出一种基于非参数贝叶斯生成模型的多模态人物概念模型，融合CLIP和CLAP提取的视觉、声学与语言特征，在少量无标签观测下实现变化观测条件下的鲁棒人员识别。 |
| [^98] | [Vision-as-a-Tool: Visual Harness for Multimodal Agentic Coding](https://arxiv.org/abs/2611.17797) | 该论文提出“视觉即工具”范式，将通用多模态大语言模型重新定义为可调用的外部工具，使纯文本智能体能够完全卸载视觉推理，从而突破智能体自身感知能力对性能的限制。 |
| [^99] | [Robustness Evaluation of GPU Cloud Pricing Benchmarks under Incomplete Observations: A Controlled Missingness Study](https://arxiv.org/abs/2611.17796) | 本文通过受控缺失实验对GPU云租赁价格基准进行压力测试，发现全源中断主要损害覆盖率而成分变化损失引入显著误差，且仅用历史数据的层级消融方法能有效将误差降低4.15至6.70个对数点。 |
| [^100] | [MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals](https://arxiv.org/abs/2611.17793) | MASTARS是一种基于扩散模型的多智能体轨迹增强框架，通过跨智能体顺序生成轨迹并利用修复方法保证协调一致性，有效解决了离线多智能体强化学习中多样性与协调性难以兼得的权衡难题。 |
| [^101] | [AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning](https://arxiv.org/abs/2611.17790) | AutoRL提出了一个由感知器等多个强化学习专属智能体角色组成的多智能体系统，通过持续监控演变的训练状态并将过程级监督与研究融入正在进行的RL训练，实现强化学习训练的自动化，为未来的递归自我改进（RSI）奠定基石。 |
| [^102] | [Can Large Language Models Optimize Pretrained Neural Network Weights? A Multi-Phase Empirical Evaluation of Gradient-Free Editing in YOLO26 Object Detection](https://arxiv.org/abs/2611.17789) | 本研究通过对YOLO26检测器的五阶段实证评估发现，包括LLM引导和进化算法在内的无梯度直接权重编辑方法虽能在开发集上带来微小增益，但在独立验证子集上泛化失败，表明大语言模型引导权重优化的价值仍然有限。 |
| [^103] | [Towards Seconds-level Video Representation: Quadratically Gated Feature Grid with Adaptive, Prediction-free Sampling](https://arxiv.org/abs/2611.17786) | Q-GAS通过多分辨率显式3D特征网格、二次门控解码器和误差感知的免预测采样策略，实现了可在50秒内完成拟合的高效秒级隐式神经视频表示。 |
| [^104] | [TReC: Evaluating Move-Induced State Changes in Instruction-Conditioned Go](https://arxiv.org/abs/2611.17781) | 本文提出迁移表示一致性指标，通过比较预测落子与专家落子按规则执行后引起的棋盘状态嵌入变化的余弦相似度来评估落子差异，实验表明 TReC 具有显著区分度且与坐标距离仅弱相关，能捕捉坐标距离无法反映的状态变化差异。 |
| [^105] | [LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent](https://arxiv.org/abs/2611.17780) | 提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。 |
| [^106] | [Certified Improvements to Best-Known Thomson-Problem Configurations with a Search Pipeline Built with Large Language Model Agents](https://arxiv.org/abs/2611.17776) | 研究者借助大语言模型智能体构建的搜索流水线（核心为副本交换盆地跳跃等方法），发现了N=912、1382和2502的汤姆逊问题新构型，其能量均低于剑桥团簇数据库的最优记录，并通过精确整数运算获得认证。 |
| [^107] | [Activation Steering the Nan-Shan in Typhoon](https://arxiv.org/abs/2611.17768) | 本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。 |
| [^108] | [GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks](https://arxiv.org/abs/2611.17766) | 该论文提出GDP_xlsx基准，利用13个领域专业人士撰写的70个真实工作簿任务和791项专家评分准则来评估前沿AI模型的电子表格处理能力，发现最强模型Gemini 4 Argon的通过率也仅有38.3%，表明该领域仍有巨大的提升空间。 |
| [^109] | [Knowledge After Collapse: Context and Wording Shape Factual Accessibility in Language Models](https://arxiv.org/abs/2611.17763) | 提出知识答案分歧（KAD）评估协议，通过在模型首次采纳错误答案时冻结对话并分支独立探针，揭示了对话历史与措辞方式显著影响语言模型在错误采纳后恢复正确事实的能力。 |
| [^110] | [Creative Product Design under Brand Image Constraints](https://arxiv.org/abs/2611.17762) | 该论文提出了一个品牌感知的产品再设计生成框架，通过构建品牌形象调研报告并将其创意方向整合进图像评价与提示词修订的自我增强循环中，从而确保生成的创意设计符合品牌形象约束。 |
| [^111] | [Superfluid: Local AI Server for Distributed LLM Inference](https://arxiv.org/abs/2611.17753) | Superfluid是一个开源本地LLM推理服务器，通过统一调度器在单设备上协调多智能体与交互用户共享模型服务（支持llama.cpp、MLX、BaseRT等运行时），并能将多台异构闲置机器聚合为单一API端点实现分布式推理。 |
| [^112] | [Logically Consistent Accurate Forecasts](https://arxiv.org/abs/2611.17735) | 该论文提出微调一个监督语言模型，利用支撑每个预测的推理过程来协调相关预测之间的逻辑不一致性，从而在ForecastBench上同时实现预测的逻辑一致性与准确性的提升。 |
| [^113] | [A Spectral Framework for Universal Proxy Regularization in Model Merging](https://arxiv.org/abs/2611.17728) | 本文提出通用代理正则化的谱框架，通过谱覆盖率与谱泄漏两个指标刻画代理分布与任务分布的关系，并从理论上证明更小的谱指标可通过更好的谱对齐降低模型合并中的参数干扰。 |
| [^114] | [Federated Soft Poly-Level Learning Over a Distributed Network with Guaranteed Convergence](https://arxiv.org/abs/2611.17722) | 该论文提出了联邦多层级学习框架FPLL，利用新颖的非线性割技术，在保证收敛性的前提下以联邦方式解决分布式网络上含软约束的四层级及以上的多层级优化问题，同时避免了数据集中带来的隐私风险。 |
| [^115] | [Cross-Linguistic Acoustic Biomarkers of Cognitive Impairment Across Interactive and Single-Speaker Speech Tasks](https://arxiv.org/abs/2611.17710) | 该研究通过对西班牙语、普通话、英语和日语四种语言中对话式与结构化语音任务的跨语言、跨任务声学分析，结合统计、专家指导和机器学习三种特征选择方法，提取出紧凑可解释的声学生物标志物，为多语言认知障碍检测提供了可扩展且公平的筛查方案。 |
| [^116] | [An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system](https://arxiv.org/abs/2611.17704) | 本研究提出AI模型ChlCast，利用分解式时空卷积与多尺度架构，以约4公里分辨率生成全球10天逐日表层叶绿素a预报，其精度超越了业务化的哥白尼全球海洋生物地球化学预报系统。 |
| [^117] | [Co-Sight Harness: Externalized Control Assets and Deterministic Enforcement for Long-Horizon Business Agents](https://arxiv.org/abs/2611.17696) | 该论文提出Co-Sight Harness框架，通过将工作流定义、状态与领域知识外部化为可版本化的控制资产，并借助状态机、机器可校验的验收门控和独立审计器进行确定性强制执行，在不改动底层语言模型的情况下，使超过200步的长程业务工作流无人值守完成率从50%提升至100%、端到端延迟降低约64%。 |
| [^118] | [Agent-written heuristics versus commercial solvers on unsolved job-shop benchmark instances](https://arxiv.org/abs/2611.17682) | 前沿大语言模型作为编程代理、仅用开源工具编写的求解器在多个未解决的作业车间基准实例上超越了商业求解器，改进了大量开放实例并建立了新的下界。 |
| [^119] | [OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation](https://arxiv.org/abs/2611.17681) | 该论文提出OCTAVE基准，通过八个能力维度、180个任务和1,800条评估需求对端到端HTML5游戏生成进行需求级能力诊断，并揭示当前最优模型组合的完整性与丰富度仅约33%。 |
| [^120] | [Aligned, Conflicting, or Unspecified? A Diagnostic Benchmark for Multimodal Time Series Forecasting](https://arxiv.org/abs/2611.17680) | 该论文提出了首个按文本与历史时间序列关系（一致、冲突、未指明）标注的多模态时间序列预测诊断基准，包含覆盖天气、维基百科和金融领域的1,762对文本-窗口数据，用于零样本系统评估十个预训练模型利用外部文本的可靠性与错误修订能力。 |
| [^121] | [Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention](https://arxiv.org/abs/2611.17662) | 提出了LeBet统计学习框架，通过学习可解释的“或-与”规则监控器并结合固定下注统计量积累证据，在达到阈值时对自主智能体运行进行干预，并提供有限样本下的假阳性控制和检测功效保证。 |
| [^122] | [Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost](https://arxiv.org/abs/2611.17658) | 该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。 |
| [^123] | [Binarization as an Attack Surface in Weightless Neural Networks](https://arxiv.org/abs/2611.17646) | 该研究揭示了无权重神经网络（WiSARD及其布隆过滤器变体）虽然看似对对抗样本天然鲁棒，但其二值化阈值本身可被利用为攻击面，并通过阈值跨越界限和响应边际证书对这一攻击机制进行了形式化证明。 |
| [^124] | [CoTTAGE: An Explainable Judge for Text-to-Audio Generation](https://arxiv.org/abs/2611.17644) | CoTTAGE是一个采用“先推理后预测”范式的可解释文本到音频生成评估器，它先生成基于音频的错误诊断依据，再预测文本相关性和音频质量分数，并通过监督微调与基于Thurstone偏好目标的强化学习，实现了与人类判断的高度一致。 |
| [^125] | [What LLM Agents See and What Users Say](https://arxiv.org/abs/2611.17639) | 该研究运用语言学预设理论系统分析了LLM智能体在工具输出与用户陈述冲突时的行动边界，发现只有当搜索结果明确“列出”多个匹配项时才能阻止大多数智能体基于错误假设行动，而仅“计数”或依赖用户声明则不足以防止误操作。 |
| [^126] | [Generated Evaluation Suites as Diversity-Driven Search for Real Agent Regressions](https://arxiv.org/abs/2611.17632) | 该论文将从智能体规范自动生成评估套件视为多样性驱动搜索，并通过注入受控缺陷证明覆盖度上限使多数可追踪缺陷无法被套件级平均通过率检测到，进而提出按场景单元格评分以发现真实的智能体回归缺陷。 |
| [^127] | [Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation](https://arxiv.org/abs/2611.17629) | 通过揭示多视角3D Transformer中每层全局注意力只承担三种功能模式之一，提出免训练的RAWATTA方法，仅在必要的层保留16位精度、其余层以8位精度近似注意力，从而在保持精度的同时大幅加速推理。 |
| [^128] | [AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs](https://arxiv.org/abs/2611.17621) | 提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。 |
| [^129] | [Safety Side Effects of Preventative Persona Steering: An Empirical Audit](https://arxiv.org/abs/2611.17620) | 该实证审计发现，预防性人格向量引导虽然能保持MMLU能力，但仍然会改变模型的拒绝行为等安全特性，揭示了以往仅基于能力评估的验证所忽略的安全副作用。 |
| [^130] | [LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters](https://arxiv.org/abs/2611.17616) | 该论文提出让大语言模型（LLM）基于无阈值的聚类整体状态视图，就地决策行为聚类的生命周期操作（分裂、合并、退役、创建），从而替代传统上依赖硬编码阈值或重聚类规则的维护范式。 |
| [^131] | [More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs](https://arxiv.org/abs/2611.17610) | 本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。 |
| [^132] | [X-Nav Technical Report: Real-World Long-Horizon Indoor-to-Outdoor Navigation on a Humanoid Robot](https://arxiv.org/abs/2611.17598) | X-Nav是首个让双足人形机器人根据语言指令在真实世界中完成从室内到室外长时程导航任务的系统，通过数据引擎、仅依赖RGB图像的MELIOR导航模型以及X-Navarch智能体工作流，实现了从数据到部署的完整流程。 |
| [^133] | [Rethinking Visual Reasoning in Text-to-Image Reward Modeling](https://arxiv.org/abs/2611.17585) | 该论文发现现有文本到图像奖励模型在视觉推理上表现不佳且偏好微调会损害这类能力，提出通过将视觉推理数据集转换为偏好对，以数据中心化的方式将视觉推理监督注入偏好学习。 |
| [^134] | [When Does Less Information Pay? An Exact Finite Atlas of Representation, Payoff, and Cost](https://arxiv.org/abs/2611.17583) | 该论文在四状态世界中穷举所有可采纳的表征与二元动作规则，精确刻画了信息保留量、任务收益与信息成本之间的完整对应关系，证明信息更少的表征在某些情形下反而能取得更高收益。 |
| [^135] | [AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI](https://arxiv.org/abs/2611.17564) | AI-Voyager是一个公益性的“人机公民推理”系统，它将3,653份各级政府治理文档转化为可溯源的证据段落，让居民与AI协同理解、质疑并起草修订治理AI的规则，在文档检索评测中达到83.0%的宏观召回率。 |
| [^136] | [Benchmarks Don't Talk Back: Learning Through Self-Disagreement](https://arxiv.org/abs/2611.17563) | 该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。 |
| [^137] | [A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes](https://arxiv.org/abs/2611.17556) | 该论文提出了一个用于工业过程目标中心因果发现的合成基准，其生成器是受批量生产启发的两层结构因果模型，由可独立设置的组件构成，能够逐一调节识别目标原因的难度，从而系统评估各类因果发现方法。 |
| [^138] | [Verifiable pXRD: Hill-climbing pXRD structure determination with LLM agents](https://arxiv.org/abs/2611.17539) | 本文将pXRD晶体结构测定形式化为可验证的智能体搜索问题，提出pXRD-Bench基准测试集，并利用LLM智能体爬山式搜索将结构恢复率从27%提升至69%，进一步扩展至实验多相pXRD场景。 |
| [^139] | [Astrocyte Biocomputing: Astrocytic Contributions to Neural Network Computation](https://arxiv.org/abs/2611.17532) | 本文将星形胶质细胞（尤其是其同时连接并调制数千个突触的能力）的特性建模引入脑启发机器学习，实现了更快的训练速度和更强的噪声鲁棒性。 |
| [^140] | [GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization](https://arxiv.org/abs/2611.17527) | 该研究发现冻结语言模型对外围作用域和上游调用者的零温度相关性评分能可靠区分真正的结构近邻与词汇干扰项，而其他关系类型的评分接近随机水平，据此校准方向性相关性信号以实现单张消费级GPU上的无状态有界上下文代码定位。 |
| [^141] | [Model Annealing: Data-Efficient Learning through Review and Consolidation](https://arxiv.org/abs/2611.17524) | 模型退火通过将训练划分为学习率升降循环的多个回合、维护过去模型状态的预测记忆库，并从对抗性分歧和记忆预测中学习，实现了对网络自身知识的复习与整合，在数据受限场景下显著提升分类准确率且不增加推理开销。 |
| [^142] | [Only as Safe as the Weakest Agent: A Break That Appears Only When the System Is Multi-Agent](https://arxiv.org/abs/2611.17520) | 该论文揭示了单模型层面已修复的攻击可通过伪造文件记录等手段攻破 Claude Code 和 Codex 等多智能体系统的子智能体机制，表明多智能体系统的安全性受制于其最弱的智能体。 |
| [^143] | [A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent](https://arxiv.org/abs/2611.17515) | 本研究将Gemma 3 12B置于基于真实伦敦白银定价诉讼记录的市场模拟中，发现损失与指责等框架效应以及向模型第22层注入相应内部状态能显著提高LLM智能体参与市场合谋操纵的倾向。 |
| [^144] | [Adversarial Tracking: Attacks on Image Sensors for Object Tracking in Autonomous Vehicles](https://arxiv.org/abs/2611.17505) | 本文首次在CARLA模拟环境中系统评估了对抗性攻击对自动驾驶“检测即跟踪”系统的影响，发现图像特定攻击可使基于YOLOv3的跟踪器的HOTA指标从约54–56%骤降至约14.8%。 |
| [^145] | [Advancing Quantization for R&R Models with Microscaling Formats](https://arxiv.org/abs/2611.17501) | 本文首次报告了原生MXFP6 GEMM（同时量化权重和激活）在AMD MI350X上为工业规模推荐与排序模型推理带来显著加速，同时满足相对基线仅0.01%归一化熵偏差的严苛精度要求。 |
| [^146] | [An Operationalization of Opaque Serial Depth](https://arxiv.org/abs/2611.17495) | 本文将“不透明串行深度”操作化为NLS深度指标，通过定义“自然语言根植节点”作为可解释瓶颈的标准，为衡量模型的非言语化串行认知能力提供了可在训练前判定架构的依据，以应对架构变化可能削弱思维链可监控性的风险。 |
| [^147] | [Do Mixture-of-Experts Routers Benefit from Ensembling?](https://arxiv.org/abs/2611.17472) | 提出在混合专家模型的路由步骤中引入路由器集成方法，实验证明更多路由器可降低验证损失，且显式训练路由器最大化专家多样性能提升平均性能。 |
| [^148] | [Use-case-driven ontology modeling with the NeOn methodology for ambient assisted living](https://arxiv.org/abs/2611.17466) | 该论文面向环境辅助生活场景，基于两个真实用例收集需求、采用NeOn方法学并复用再工程现有本体，为意大利Design For All项目的虚拟家庭框架构建了模块化本体作为语义数据层。 |
| [^149] | [Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models](https://arxiv.org/abs/2611.17462) | 提出扩散树采样（DTS）方法，通过在去噪轨迹上进行蒙特卡洛树搜索，在推理阶段引导晶体扩散生成模型，既能直接复用现有性质条件模型而不扭曲目标分布，又支持多目标采样以高效生成满足多个冲突性质的晶体候选。 |
| [^150] | [Racing the Clock: Teaching VLNs Budget-Conditioned Object Search](https://arxiv.org/abs/2611.17442) | 该论文提出“预算条件下的目标搜索”新任务，使视觉语言导航智能体能够根据明确的动作预算自适应调整搜索策略，并对比了基于特权信息的TSP专家与基于部分可观测和跨场景语义先验的房间级Bellman专家两种监督方案。 |
| [^151] | [Interpreting Coding Agents Through Their Behavioral Trajectories](https://arxiv.org/abs/2611.17440) | 该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。 |
| [^152] | [WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search](https://arxiv.org/abs/2611.17412) | 提出WETW学习型距离，利用小型对称网络为每对序列的每个嵌入维度预测权重，实现针对特定序列对的自适应比较且无需计算对齐，将误差相比最佳静态距离减半，从而实现高效的DTW相似性搜索。 |
| [^153] | [Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming](https://arxiv.org/abs/2611.17395) | 本文通过实证安全研究首次系统评估了大语言模型在权重空间编程中的后门注入能力，表明通用大语言模型可以在不访问任务数据、不执行前向传播、不计算梯度的情况下，仅凭自然语言指令直接修改模型权重来注入后门，揭示了权重空间编程带来的新型安全威胁。 |
| [^154] | [Confident and Wrong: Structure Confidence Is a Poor Selection Filter for Protein Function](https://arxiv.org/abs/2611.17391) | 研究表明结构置信度分数（如pTM、ipLDDT）无法预测蛋白质功能——基于结构置信度筛选的lanmodulin变体中仅16.7%能结合铽，而基于功能引导模型设计的变体结合率高达83.3%，两者结合几率相差25倍，说明功能引导模型才是设计稀土结合蛋白的有效筛选策略。 |
| [^155] | [Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms](https://arxiv.org/abs/2611.17390) | 提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。 |
| [^156] | [Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents](https://arxiv.org/abs/2611.17388) | 该论文构建了一个零泄漏的时间一致性评估基准，在DragonFly和React的历史截止快照上，以四种提示粒度对比评估基线、扁平AST图与分层依赖图ServiceExplorer三种上下文增强方案对SWE智能体解决100个后续拉取请求的效果。 |
| [^157] | [ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects](https://arxiv.org/abs/2611.17381) | ProdPilot是一个本地MCP服务器，其核心创新在于将50条生产就绪规则中的88%实现为固定模板或基于项目信息提取的确定性修复，而非依赖不可控的模型自由生成，从而对氛围编码项目实现可验证的自动化审计、修复、门控与部署。 |
| [^158] | [Hybrid Quantum-Classical Sequence Models for Sample-Efficient Energy Regression in Simulated Calorimeter Showers](https://arxiv.org/abs/2611.17354) | 该论文在CaloGAN基准数据集上系统比较了八种经典与量子增强架构在正电子能量重建任务中的表现，发现基于纵向簇射序列的混合量子-经典模型结合物理信息驱动的数据表示达到了最高精度，证明了数据表示选择对能量回归的关键作用。 |
| [^159] | [Tabular Population Priors as Structured Retrieval for EHR Medication Set Prediction](https://arxiv.org/abs/2611.17341) | 该论文提出用从训练数据估计的条件-药物共现表构建的表格化人群先验，替代临床RAG系统中的患者记录检索，作为更贴合药物集合预测任务的结构化检索对象，在MIMIC-IV数据上无需额外LLM调用和推理时患者索引即可显著提升F1和Jaccard指标。 |
| [^160] | [LoRAWeaver: Directional Adapter Routing in LoRA Representation Space](https://arxiv.org/abs/2611.17340) | 提出了一种免训练的LoRA适配器路由方法，通过在适配器输出空间中用余弦相似度将适配器条件化的查询表示与任务原型匹配来选择适配器，在跨任务设置中优于免训练基线，且无需学习路由器即可接近训练过的检索编码器的性能。 |
| [^161] | [Does the Ranking Hold? A Practitioner-Facing Audit of Hallucination Detectors on Small, Quantized, On-Device LLMs](https://arxiv.org/abs/2611.17339) | 本文通过单GPU可复现审计发现，幻觉检测方法在小型4位量化端侧模型之间的排名会发生反转，说明在大型研究模型上得出的检测器评估结论无法直接迁移到从业者实际部署的小型端侧模型上。 |
| [^162] | [When Should LLM Agents Forget? Memory Filtering Under Query-Critical Resolver Errors](https://arxiv.org/abs/2611.17338) | 该论文揭示LLM记忆系统的硬过滤删除在解析器误删查询关键事实时会显著损害下游准确率且误删比误留危害更大，提出应将删除决策视为成本敏感问题，并用部署匹配的门控机制取得了优于固定硬删除策略的效果。 |
| [^163] | [Ballad: Bandit-Based LLM Routing for Automated Heuristic Discovery](https://arxiv.org/abs/2611.17241) | 该论文提出Ballad，一种基于多臂老虎机的LLM路由方法，可在自动化启发式发现过程中自适应地选择调用哪个LLM，并实证表明更大更昂贵的模型并不总能带来更好的启发式。 |
| [^164] | [S1-Evo: Experience-Guided Evidence Competition Attacks on Typed System-1 Decision Models](https://arxiv.org/abs/2611.16937) | 提出S1-Evo自进化攻击框架，通过契约感知的证据扰动诱发证据竞争，并借助经验卡机制指导测试时扰动构建，从而在保持事实不变性的前提下有效攻击LLM智能体中部署的类型化系统-1决策模型。 |
| [^165] | [ZONOS3-Mini Technical Report](https://arxiv.org/abs/2611.16525) | ZONOS3-Mini 通过 886M 参数（每 token 仅激活 100M）的潜在 MoE 骨干网络和新型音频编解码器 ZAC-1，实现了可在笔记本和手机上实时运行的零样本语音克隆 TTS 模型，并在 55 个语言集合上取得 6.71% 宏平均 WER，达到与更大模型相当的先进水平。 |
| [^166] | [HyBridge: Bridge-Guided Self-Supervised Hyperbolic Hypergraph Representation Learning](https://arxiv.org/abs/2611.16351) | 该论文提出HyBridge框架，通过考虑超边外部连通性来估计节点桥梁分数，实现了自监督的双曲超图表示学习，解决了传统桥梁分数估计无法捕捉超图群组连通性的问题。 |
| [^167] | [Benchmarking AI risk behaviors](https://arxiv.org/abs/2611.16318) | 该论文提出了一个评估LLM风险倾向的工具包，发现2026年最先进模型因更倾向于采用期望值最大化作为决策程序，而比旧模型更稳定、更一致且更趋于风险中性。 |
| [^168] | [Rectified Looped Transformer](https://arxiv.org/abs/2611.16265) | 提出整流循环Transformer（RLT），通过受流模型启发的插值调度重新设计循环间反馈，在理论上保证隐藏状态幅度有界并降低扰动放大，从而提升参数高效的语言建模能力。 |
| [^169] | [Indic-Task Gains and Regressions after Reinforcement Learning](https://arxiv.org/abs/2611.16228) | 该研究通过GRPO强化学习显著提升了覆盖英语和八种印度语言的30B MoE模型的数学及多数印度语言任务表现，但发现英语回答倾向增强导致跨语言问答大幅退化，且加入语言奖励的第二次RL训练虽改善了指令遵循，却以牺牲推理和编码能力为代价。 |
| [^170] | [JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents](https://arxiv.org/abs/2611.16174) | 提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。 |
| [^171] | [Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models](https://arxiv.org/abs/2611.15592) | 该研究通过在完全统一的协议下（相同主干模型、训练数据、训练方案和随机种子）重新实现并训练五种类型化决策模型的读出头，以对照实验的方式厘清哪些设计选择真正影响决策性能。 |
| [^172] | [CARE-Drive: Coupled Alternating Refinement for Ego-trajectories](https://arxiv.org/abs/2611.15360) | CARE-Drive将轨迹精化分解为路径点残差修正与以位置为条件的航向生成的耦合交替精化框架，实现了位置和航向的专门化建模，从而提升了端到端自动驾驶VLA模型的轨迹预测质量。 |
| [^173] | [MIGRO: A Decoupled and Portable Long-Term Memory System for Language Models](https://arxiv.org/abs/2611.15315) | MIGRO提出了一种解耦且可移植的长期记忆系统，通过“非对称保真与延迟选择性”原则——精确保存历史证据、近似组织索引、延迟检索决策——将用户交互积累的经验作为持久状态，在相继的语言模型之间共享与复用。 |
| [^174] | [CastWorld: Benchmarking Context-Aware Agentic Time Series Forecasting Beyond Predictive Accuracy](https://arxiv.org/abs/2611.15262) | CastWorld是一个面向上下文感知智能体时间序列预测的基准测试，提供了涵盖8个领域25个真实数据集的丰富上下文数据套件，以及包含49个核心工具和状态自适应循环的预测原生框架，实现了对预测智能体超越单纯预测精度的系统化评估。 |
| [^175] | [TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery](https://arxiv.org/abs/2611.13971) | TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。 |
| [^176] | [AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding](https://arxiv.org/abs/2611.13710) | AutoManifold 提出了一种由多智能体大语言模型驱动的算法生成流水线，能够根据用户指定的结构保持偏好，从受约束的原语词汇中自动组合并迭代优化流形嵌入算法，实现面向用户偏好的数据可视化算法自主设计。 |
| [^177] | [ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding](https://arxiv.org/abs/2611.13442) | 该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。 |
| [^178] | [Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA](https://arxiv.org/abs/2611.13331) | 该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。 |
| [^179] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^180] | [Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood](https://arxiv.org/abs/2611.13213) | 该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。 |
| [^181] | [Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks](https://arxiv.org/abs/2611.12799) | 该论文提出Agentic-OPD方法，将GRPO训练中被丢弃的检查点作为免费的教师池，通过有选择性的在线策略蒸馏来恢复最佳验证检查点所遗忘的智能体问题求解能力。 |
| [^182] | [Sage-Taiyi: A Metacognitive Architecture for Scientific Discovery](https://arxiv.org/abs/2611.12762) | Sage-Taiyi是一种受元认知启发的科学发现架构，通过监控并调控自身对证据与假设的理解来动态调整研究目标和推理方法，在Taiyi-Bench的全部八项指标上超越最强基线AI Scientist-v2排名第一，并经专家协作实现了实验与计算验证。 |
| [^183] | [All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States](https://arxiv.org/abs/2611.10123) | 该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。 |
| [^184] | [ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective](https://arxiv.org/abs/2611.09326) | 该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。 |
| [^185] | [AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling](https://arxiv.org/abs/2611.08752) | AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。 |
| [^186] | [Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations](https://arxiv.org/abs/2611.08269) | 预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。 |
| [^187] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^188] | [Consistency Is Not Understanding: Orbit Audits for Neural Field Weight Readers](https://arxiv.org/abs/2611.05798) | 该论文提出“轨道审计”方法——利用保持函数等价的隐藏神经元置换和正 ReLU 缩放变换来检验从神经场权重中读取视觉内容的模型，发现现有读取器在等价变换下准确率大幅下降甚至对相同输入给出不一致答案，证明其表面上的输出一致性并不等于对权重所表示函数的真正理解。 |
| [^189] | [An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems](https://arxiv.org/abs/2611.03301) | 本文提出一个以 Shapley 公理为结构性约束的公理化框架，通过新颖的 NAHT 分解刻画开放与完全可控多智能体系统的本质差异，为开放多智能体系统的强化学习奠定合作博弈论基础，并证明对个体价值函数施加这些公理可恢复 Dec-POMDP 的 Shapley 值。 |
| [^190] | [One Offset, Two Directions: Baseline Adjustment via Subgroup Offsets](https://arxiv.org/abs/2611.00782) | 该论文提出用“子群偏移”（子群均值与全局均值的偏差）配合一个带符号的系数 λ 来调整 GRPO 的基线，以解决提示内采样输出因异质子群而产生不同奖励分布时的信用分配问题，且 λ 的正负号给出两种相反的基线调整方向。 |
| [^191] | [EmbodiExp: Compositional Experience Transfer for Embodied Agents](https://arxiv.org/abs/2610.39180) | EmbodiExp 提出一种以结果为依据的子目标经验表示，使具身智能体能够在任务执行过程中在线组合与迁移经验，并借助子目标层面的结果归因来利用失败轨迹。 |
| [^192] | [Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS](https://arxiv.org/abs/2610.33253) | 本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。 |
| [^193] | [Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3](https://arxiv.org/abs/2610.30499) | 该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。 |
| [^194] | [Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI](https://arxiv.org/abs/2610.28411) | Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。 |
| [^195] | [TimesJev: Thinking Fast on Time Series with System 1 Sufficiency](https://arxiv.org/abs/2610.26002) | 本文提出TimesJev——首个时间序列系统1模型，通过形式化系统1充分性并构建保留关键频率模式的压缩表示，使模型无需长思维链即可对时间序列实现即时推理，从而大幅降低推理延迟。 |
| [^196] | [Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations](https://arxiv.org/abs/2610.24471) | 该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。 |
| [^197] | [FlashRT-PAI: Fast Kernels Are Not Fast Pipelines for Physical AI](https://arxiv.org/abs/2610.24349) | 该论文提出FlashRT-PAI低延迟推理引擎，通过深度内核融合、启动重叠和执行契约等模型-硬件流水线协同设计机制，消除物理AI系统实时推理中的系统级开销瓶颈，突破了传统孤立内核优化的局限。 |
| [^198] | [TorchTPU - A Hybrid Eager/Compiled Architecture for Native PyTorch Execution on TPU](https://arxiv.org/abs/2610.24161) | TorchTPU 是 Google 与 Meta 合作的开源项目，通过混合即时/编译架构为 TPU 提供原生 PyTorch 后端，实现开箱即用的模型迁移并达到业界竞争力的训练与推理性能。 |
| [^199] | [Self-Building Labs](https://arxiv.org/abs/2610.23826) | 该论文提出“自构建实验室”（SBL），能根据高层科学目标自主设计并搭建实验室的布局、数字孪生、机器人技能与实验流程，突破了现有自动驾驶实验室灵活性不足这一制约科学发现规模化的瓶颈。 |
| [^200] | [Learning to Read Out: Unembedding Dynamics in Language Model Pretraining](https://arxiv.org/abs/2610.21025) | 提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。 |
| [^201] | [Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction](https://arxiv.org/abs/2610.13672) | 该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。 |
| [^202] | [When to Rethink: Learning Multi-Perspective Self-Verification for Vision-Language Models](https://arxiv.org/abs/2610.07018) | 该论文提出MOTIVE框架，通过多视角自验证与可靠性引导的选择性重新思考机制，解决了视觉语言模型中因单一验证标准或固定提示词导致的可靠性估计不完整且不稳定的问题。 |
| [^203] | [Proof-Grounded Patient-Specific Clinical Explanations from Knowledge-Graph Reasoning](https://arxiv.org/abs/2610.06549) | 该论文提出 CKG 临床解释引擎，在免训练的临床知识图谱推理器下游将患者推理状态转化为可审计、带溯源链接且符合政策许可的临床解释，并在逾万名患者的大规模数据上验证了事实与报告的完整性和数值保真度。 |
| [^204] | [What Should World Models Forget? Stratified Retention for Continual Adaptation](https://arxiv.org/abs/2610.03713) | 提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。 |
| [^205] | [Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback](https://arxiv.org/abs/2610.02771) | 本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。 |
| [^206] | [Cross-Benchmark Transfer from RL on Agentic Coding Tasks](https://arxiv.org/abs/2610.00890) | 仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。 |
| [^207] | [Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN](https://arxiv.org/abs/2609.39626) | 本研究提出利用StyleGAN2的中间潜在空间对储层物性进行参数化，以克服集合数据同化方法在处理非高斯复杂地质（复杂相分布）条件下历史拟合性能下降的问题，并与VAE-GAN和潜在扩散模型等深度学习参数化方案进行了对比评估。 |
| [^208] | [MeanFlowAdvantage: Stable Reward Fine-Tuning for Few-Step Average-Velocity Generators](https://arxiv.org/abs/2609.37670) | 提出 MeanFlowAdvantage——一种带符号优势加权最小二乘目标，通过共享的 MeanFlow 导数校正解决平均速度生成器与优势奖励微调之间的失配问题，在保留原生少步采样的同时将奖励改进直接迁移到部署的流映射，并在 SD3.5-Medium 全部八项指标上取得提升。 |
| [^209] | [From Data to Program: Fast & Direct Generative Program Inference from Empirical Data](https://arxiv.org/abs/2609.35348) | PRODiGI是一种预训练的数据到程序模型，能在单次前向传播中从经验数据直接推断出显式可执行的生成程序，配合程序空间微调在密度和分数估计上超越现有预训练模型并获得数倍加速。 |
| [^210] | [Hidden not Deleted: How Networks Suppress Entangled Features](https://arxiv.org/abs/2609.27593) | 该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。 |
| [^211] | [CIDERS: Cloud-Edge LLM Collaborative Learning via Accelerating Personalized Bilevel Optimization](https://arxiv.org/abs/2609.15664) | 本文首次提出个性化双层优化框架将云边LLM协作形式化为双重结构，并通过CIDERS求解器将模型分解为可学习骨干和信使组件，加速实现云端知识迁移与边缘个性化适配的协调演进。 |
| [^212] | [Efficient Test-Time Adaptation through Human-AI Interaction](https://arxiv.org/abs/2609.04141) | 提出TAHI框架，利用跨会话的人机交互数据并结合不断演化的评分准则模块，在测试时对智能体的上下文和权重进行自适应，从而弥合AI生成成果与个人专业水准之间的差距。 |
| [^213] | [AI Agents Push Humans Out of the Loop](https://arxiv.org/abs/2608.23642) | 本文指出当前AI代理发展方式削弱而非支持人类监督能力，强调应将人类监督需求置于与AI能力同等优先级，以保障有效人类参与。 |
| [^214] | [From Regulation to Implementation: A Critical Evaluation of LLM-Assisted Regulatory Compliance in Industry](https://arxiv.org/abs/2608.21317) | 本文批判性评估了大语言模型在工业合规文档生成中的应用，指出其面临数据异构、格式标准化及跨学科知识整合等挑战，并提出改进路径。 |
| [^215] | [A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations](https://arxiv.org/abs/2608.18389) | 本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。 |
| [^216] | [ReForge: Keeping ABR Algorithms Never Finished with Verified Large Language Model Edits](https://arxiv.org/abs/2608.15138) | ReForge利用大型语言模型循环持续重新设计ABR算法，通过验证每个小编辑对历史场景的无害性，实现算法对不断变化网络场景的分钟级适应。 |
| [^217] | [Hallucination-Free GUI Grounding via Regression-Free Layout-Aware Matching](https://arxiv.org/abs/2608.09654) | 该论文提出了一种无回归框架，利用冻结的MLLM将抽象指令解析为富含布局线索的结构化视觉描述，再通过布局感知匹配实现无坐标回归、无幻觉的精确GUI元素定位。 |
| [^218] | [When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning](https://arxiv.org/abs/2607.29617) | 该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。 |
| [^219] | [Courteous Anticipation: Improving Long-Lived Task Planning in Persistent Shared Environments](https://arxiv.org/abs/2607.20289) | 提出“礼貌性预判规划”，让机器人在持久共享环境中求解当前任务时，通过独立的每机器人学习估计器，联合最小化即时成本与所有机器人的聚合预期未来成本，从而避免副作用在长任务序列中累积。 |
| [^220] | [CPM-MultiAgent: A CPM-Grounded Appraisal Framework for Simulating Dynamic Patient Emotional Evolution in Healthcare Dialogue](https://arxiv.org/abs/2607.07824) | 该论文提出基于心理学组件过程模型（CPM）的多智能体框架CPM-MultiAgent，能够模拟患者情绪随医护人员沟通而动态演变的过程，为医疗沟通训练中的学习者提供真实的情绪交互反馈。 |
| [^221] | [REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming](https://arxiv.org/abs/2607.07738) | 本文提出REFORGE——一个带溯源追踪的评测流水线，通过八门置信度漏斗将编译器优化下的二进制-源码对齐不确定性量化分级，从而解决了LLM逆向工程能力公平评估中函数级真值可靠性不足这一核心障碍。 |
| [^222] | [Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study](https://arxiv.org/abs/2607.04926) | 该研究通过对微型Transformer进行完全可枚举实验发现，无论信息以符号token、神谕代码还是纠缠感知向量的形式输入，模型都无法实现零样本组合绑定（其失败源于归纳偏置而非信息缺失），但在小样本设置下，绑定学习的样本效率由输入代码的可读性决定。 |
| [^223] | [URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment](https://arxiv.org/abs/2607.04688) | URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。 |
| [^224] | [When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts](https://arxiv.org/abs/2607.03836) | 在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。 |
| [^225] | [QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents](https://arxiv.org/abs/2606.32034) | 该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。 |
| [^226] | [CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies](https://arxiv.org/abs/2606.16613) | CoffeeBench通过模拟咖啡供应链中的多智能体经济，评估LLM智能体的长程商业决策能力，发现Claude Sonnet 5.5凭借更高的售价和更低的采购成本获得最高净收入。 |
| [^227] | [Larch: Learned Query Optimization for Semantic Predicates](https://arxiv.org/abs/2606.07923) | Larch利用非结构化数据嵌入所承载的语义信息，通过学习型优化技术高效优化AI SQL查询中语义过滤器的执行。 |
| [^228] | [PhysDox: Benchmarking LLMs on Physical Feasibility Auditing of Physiological Sensing Protocols](https://arxiv.org/abs/2606.05003) | 本文提出PhysDox基准用于评估大语言模型对生理传感协议物理可行性的审计能力，发现现有LLM表现不佳（最高宏F1仅53.0），且存在将程序完整性与物理有效性混淆的“脚手架偏差”，导致隐式约束的漏检率是显式违规的两倍。 |
| [^229] | [Certified Policy Optimisation for Nested Causal Bandits via PAC-Bayes Risk](https://arxiv.org/abs/2605.29788) | 本文提出嵌套上下文因果老虎机（NCCB）这一分层因果决策问题框架及嵌套因果汤普森采样（NCTS）算法，其核心贡献是一个因果PAC-Bayesian超额风险界，可仅凭历史数据、离策略且随时地对任意候选部署策略进行认证，回答“能否信任该智能体、风险多大”的部署问题。 |
| [^230] | [DeltaBox: Millisecond-Scale Sandbox Checkpoint and Rollback for Stateful AI Agents](https://arxiv.org/abs/2605.22781) | 提出DeltaBox系统，基于“相邻沙盒状态间差异微小”这一关键洞察，设计操作系统级抽象DeltaState，通过仅复制增量变更而非全量状态，实现了毫秒级的沙盒检查点与回滚，支撑AI智能体的高频状态探索。 |
| [^231] | [TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation](https://arxiv.org/abs/2605.22355) | 该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。 |
| [^232] | [Wavelet Flow Matching for Multi-Scale Physics Emulation](https://arxiv.org/abs/2605.16573) | 提出小波流匹配（WFM），直接在多尺度小波空间中进行最优传输流匹配，无需预训练自编码器，在混沌流体仿真中以更低成本实现了更优的谱相干性与细节保持能力。 |
| [^233] | [ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents](https://arxiv.org/abs/2605.16116) | ShopGym提出了一种集成框架，通过将真实在线商店转化为匿名化、可复现的沙盒仿真环境，解决了电子商务网络智能体评估中真实性与可控性、可扩展性难以兼顾的方法论瓶颈。 |
| [^234] | [D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models](https://arxiv.org/abs/2605.13276) | D-VLA提出了一个面向大规模视觉-语言-动作模型的高并发分布式异步强化学习框架，通过“平面解耦”隔离训练数据与权重控制、四线程异步“泳道”流水线实现完全并行，解决了仿真与深度学习间的资源冲突并大幅提升系统吞吐量。 |
| [^235] | [Hybrid Mechanical and Agentic Specification Inference](https://arxiv.org/abs/2605.10005) | 该论文将机械式最弱前置条件（WP）分析与LLM智能体驱动的规约推断相结合，集成到 Move Prover 中自动为 Move 智能合约推导规约，在 Aptos 框架的无规约公开函数上实现了超过 90% 的任务完成率。 |
| [^236] | [Do LLMs Experience an Internal Polylogue? Investigating Reasoning through the Lens of Personas](https://arxiv.org/abs/2605.09159) | 本文创新性地将“人格向量”从静态引导工具转变为动态可监控信号，通过追踪其与隐藏激活在生成过程中的对齐序列（“多重对话”）来预测推理正确性，并据此设计分段干预方法提升模型在MMLU-Pro上的准确率。 |
| [^237] | [Multi-Environment POMDPs with Finite-Horizon Objectives](https://arxiv.org/abs/2605.07537) | 本文针对具有有限时域目标的多环境POMDP，证明了最多n个确定性策略的混合即可覆盖n个环境下的最优策略，并提出了无需存储指数级规模策略描述即可计算最优值与最优策略的新方法。 |
| [^238] | [Exposing and Mitigating Temporal Attack in Deepfake Video Detection](https://arxiv.org/abs/2605.07398) | 本文揭示了时空深度伪造检测器因过拟合脆弱的时间频谱线索而易受时间攻击的问题，并提出频谱不变防御框架SpInShield，通过可学习频谱对抗器与捷径抑制策略将语义运动与频谱伪影解耦，在模拟攻击下AUC比最强基线提升21.30个百分点。 |
| [^239] | [Time series causal discovery with variable lags](https://arxiv.org/abs/2605.04081) | 本文提出了一种可分解的BIC风格评分方法，用于联合学习时间序列因果发现中的变量时间顺序与每个有序变量对的主导滞后项，并通过增强的禁忌搜索算法进行优化。 |
| [^240] | [BoostLoRA: Growing Effective Rank by Boosting Adapters](https://arxiv.org/abs/2604.27308) | BoostLoRA提出一种梯度提升框架，通过在错误样本上迭代训练并合并正交子空间的超低秩适配器，使有效秩随轮次线性增长，在不增加任何推理开销的情况下超越了全量微调的性能。 |
| [^241] | [Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based LLM Confidence Signals](https://arxiv.org/abs/2604.17714) | 本文借鉴临床人格评估的有效性筛查传统，提出一种基于L、Fp和RBS指标的可移植筛查方法，在解读之前检验LLM置信度信号是否真正跟踪正确性，对20个前沿模型的评估发现15个有效、4个不确定、1个无效（DeepSeek-R1弃答了95%的正确答案）。 |
| [^242] | [CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations](https://arxiv.org/abs/2604.14586) | 本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。 |
| [^243] | [Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection](https://arxiv.org/abs/2604.14397) | 提出一种用双语词典增强对齐器并结合过滤策略的跨语言词义投影方法，能够将WordNet式词汇资源自动扩展到新语言，在保持可解释性和资源高效的同时提升精确率。 |
| [^244] | [Minimizing classical resources in variational measurement-based quantum computation for generative modeling](https://arxiv.org/abs/2604.11578) | 本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。 |
| [^245] | [Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification](https://arxiv.org/abs/2604.07953) | 本文提出多维度评估框架、有界误差剪枝策略及新型Hydrant分类器扩展，可在时间序列分类任务中实现能耗降低高达70%，而预测精度几乎不受影响。 |
| [^246] | [Generative machine learning for protein design](https://arxiv.org/abs/2603.26378) | 本文系统综述了59种生成式机器学习方法在六类蛋白质设计任务（序列生成、骨架生成、结构条件序列设计、序列-结构联合设计、基序支架及结合蛋白/抗体设计）中的应用，并讨论了设计蛋白质的计算与实验评估方法。 |
| [^247] | [FlashSampling: Fast and Memory-Efficient Exact Sampling](https://arxiv.org/abs/2603.15854) | FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。 |
| [^248] | [Comparative Analysis of 3D Convolutional and 2.5D Slice-Conditioned U-Net Architectures for MRI Super-Resolution via Elucidated Diffusion Models](https://arxiv.org/abs/2603.14667) | 该研究在阐明扩散模型框架下系统比较了3D卷积U-Net与2.5D切片条件U-Net两种骨干架构用于脑部MRI的2倍面内超分辨率重建，发现两者总体性能相当，在训练退化条件下PSNR分别为37.72 dB和37.70 dB。 |
| [^249] | [A Variational Latent Equilibrium for Learning in Neuronal Circuits](https://arxiv.org/abs/2603.09600) | 该论文提出了一种基于神经元状态前瞻性能量函数的通用形式化框架，以可控且生物学合理的方式统一并扩展了多种局部时空信用分配方法，从而实现对随时间反向传播（BPTT）的近似，用于学习复杂时间依赖性。 |
| [^250] | [Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation](https://arxiv.org/abs/2603.09527) | 本文提出参数与数据高效的草稿模型适配框架 EDA，通过解耦架构、数据再生策略和样本选择机制三项创新，高效地将草稿模型对齐到微调后的目标模型，从而提升推测解码的加速效果。 |
| [^251] | [ViroGym: Realistic Large-Scale Benchmarks for Evaluating Viral Proteins](https://arxiv.org/abs/2603.06740) | ViroGym是一个针对病毒蛋白的大规模综合基准，涵盖79项深度突变扫描实验、21项流感中和任务和SARS-CoV-2现实世界大流行预测，发现ProGen2系列在所有任务中表现最佳，且DMS和中和任务的表现可可靠地预测模型在现实世界新变异出现时的泛化能力。 |
| [^252] | [ESENSC: A Polynomial-Time Axiomatic Alternative to SHAP](https://arxiv.org/abs/2603.00496) | ESENSC是一种仅需线性次联盟值评估的多项式时间确定性归因方法，在表格预测任务上与精确SHAP高度一致，同时计算成本远低于后者。 |
| [^253] | [Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume](https://arxiv.org/abs/2602.24195) | 提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。 |
| [^254] | [Recursive Belief Vision Language Action Models](https://arxiv.org/abs/2602.20659) | RB-VLA提出一种以信念为中心的视觉-语言-动作模型架构，通过自监督世界模型目标训练，维护编码任务相关历史、动力学与物体交互的紧凑潜在信念状态，每个任务仅需查询一次VLM即可支持部分可观测条件下的长时程多阶段操控。 |
| [^255] | [Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability](https://arxiv.org/abs/2602.17544) | 该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。 |
| [^256] | [ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation](https://arxiv.org/abs/2602.15189) | 本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。 |
| [^257] | [ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression](https://arxiv.org/abs/2602.11008) | ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。 |
| [^258] | [From Chats to Markets: AgenticPay for LLM-Powered Negotiation in Multi-Agent Commerce](https://arxiv.org/abs/2602.06008) | AgenticPay是一个包含160个多模态任务、覆盖四种商业场景和八种市场拓扑的统一基准，用于评估多模态LLM智能体在竞争性多智能体市场中通过多轮谈判达成高福利、多维度协议的能力。 |
| [^259] | [The Alignment Curse: Modality Alignment Supercharges Audio Attacks via Text Transfer](https://arxiv.org/abs/2602.02557) | 提出并验证了“对齐诅咒”原则——全模态模型中文本-音频模态对齐越强，文本越狱攻击向音频模态的迁移就越有效，揭示了模型能力提升与安全性之间的根本张力。 |
| [^260] | [Tracing 3D Anatomy in 2D Strokes: Can Projection-Derived Masks Replace 3D Segmentation for Cervical Spine Fracture Recognition?](https://arxiv.org/abs/2601.15235) | 本研究提出一种多阶段流水线，通过从二维投影近似生成椎体掩膜来替代完整的三维分割，在大幅降低计算复杂度的同时，仍能保留足够的诊断上下文以实现颈椎骨折的高精度识别。 |
| [^261] | [ISLA: Ensemble learning and domain adaptation for generalizable acute ischemic stroke lesion segmentation in diffusion MRI with clinical validation](https://arxiv.org/abs/2601.08732) | 该研究提出了基于nnU-Net设计原则的急性缺血性卒中病灶分割框架ISLA，通过集成学习与利用无标签目标域数据的半监督域适应变体ISLA-MT，显著提升了模型在不同临床中心的泛化能力，并完成了临床验证。 |
| [^262] | [Vision-language models lag human performance on physical dynamics and intent reasoning](https://arxiv.org/abs/2601.01547) | 该论文提出“目的性空间智能”（TSI）概念，并基于11,328个真实世界视频构建了EscherVerse大规模基准与指令微调数据集，揭示当前最先进的视觉语言模型在整合物理动力学与人类意图进行空间推理方面仍显著落后于人类。 |
| [^263] | [NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation](https://arxiv.org/abs/2512.05844) | 提出NEAT模型，通过新颖的邻域引导训练策略将分子图视为原子集合并学习与顺序无关的token分布，实现原子级置换不变性，在QM9和GEOM-Drugs数据集上达到最先进的3D分子生成质量。 |
| [^264] | [Model-Free Assessment of Simulator Fidelity via Quantile Curves](https://arxiv.org/abs/2512.05024) | 该论文提出一种无模型方法，通过为潜在总体参数构建置信集并估计分位数曲线，来稳健地量化生成式AI模拟器与真实系统之间随情景变化的“仿真-现实”差距。 |
| [^265] | [E0: Expressive Fine-Grained Discrete Action Prediction for Vision-Language-Action Models via Tweedie Discrete Diffusion](https://arxiv.org/abs/2511.21542) | 提出E0框架，通过Tweedie离散扩散将动作生成建模为对量化动作token的迭代去噪过程，实现了与token化推理天然契合的表达性细粒度离散动作预测，从而解决VLA模型泛化能力差和动作粗糙不稳定的问题。 |
| [^266] | [Manifold Approximation leads to Robust Kernel Alignment](https://arxiv.org/abs/2510.22953) | 提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。 |
| [^267] | [BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution](https://arxiv.org/abs/2510.08697) | 该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。 |
| [^268] | [Graph Your Own Prompt](https://arxiv.org/abs/2509.23373) | 该论文提出图一致性正则化（GCR），通过在任意深度插入无参数的图一致性层，将特征相似性图与由模型自身预测导出的类感知掩码预测图对齐，实现自我提示式训练，使特征表示更具类一致性和语义意义。 |
| [^269] | [HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding](https://arxiv.org/abs/2505.24098) | HARDTESTGEN提出了一种为算法编程问题自动合成高质量测试用例的流水线，并构建了包含2.66万道题的HARDTESTS数据集，其测试用例在验证LLM生成代码方面的精确度比现有测试提升11.22个百分点，有效支撑拒绝采样和强化学习等下游后训练。 |
| [^270] | [Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection](https://arxiv.org/abs/2505.18934) | 提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。 |
| [^271] | [AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion](https://arxiv.org/abs/2505.12358) | AbFlowNet提出了一种将GFlowNet与扩散模型融合的新型生成框架，通过将每个扩散步骤建模为GFlowNet中的状态，在单一训练流程中直接引入能量信号，实现扩散损失与抗体-抗原结合能的联合优化，从而避免了依赖不可靠估计器的昂贵在线强化学习流程。 |
| [^272] | [LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark](https://arxiv.org/abs/2410.13013) | 提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。 |
| [^273] | [Benign Overfitting for Regression with Trained Two-Layer ReLU Networks](https://arxiv.org/abs/2410.06191) | 本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。 |
| [^274] | [BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/abs/2406.15877) | BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。 |
| [^275] | [Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models](https://arxiv.org/abs/2401.00788) | 该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。 |
| [^276] | [Fake News Detectors are Biased against Texts Generated by Large Language Models.](http://arxiv.org/abs/2309.08674) | 假新闻检测器倾向于将大型语言模型生成的内容标记为假新闻，而将人工编写的假新闻误分类为真实，我们提出了一种通过敌对训练和LLM改写的真实新闻等方法来解决这个问题，并取得了显著的改进。 |

# 详细

[^1]: WeightBench：将语言模型作为权重空间程序员进行评估

    WeightBench: Evaluating Language Models as Weight-Space Programmers

    [https://arxiv.org/abs/2611.18553](https://arxiv.org/abs/2611.18553)

    该论文提出了WEIGHTBENCH基准，通过五大赛道的42项权重空间编程任务系统评估语言模型直接读取、修复和编写神经网络参数的能力，发现前沿模型已能读取小型网络并编写可用的控制策略，但尚无法修补或压缩大型语言模型。

    

    最近的研究表明，大型语言模型（LLM）可以直接操作其他神经网络的参数。它们能够读取一个网络被训练来做什么，修复失效的网络，并从零开始编写可正常工作的网络——所有这些都无需梯度下降。这种能力被称为权重空间编程。要系统地探索这一新范式，需要一个稳健的基准。我们提出了WEIGHTBENCH，这是一个在五大赛道（读取权重、修复权重、编写权重、识别不可能的编辑以及无数据压缩）上，对能力递增的LLM进行42项WSP任务评估的基准。LLM读取和编辑权重的网络规模从2K参数的控制策略到7B参数的语言模型不等。实验基准结果表明，前沿智能体已经能够读取小型网络并在从未实际运行它们的情况下编写出可正常工作的控制策略，但目前还无法修补语言模型或对其进行压缩。

    arXiv:2611.18553v1 Announce Type: cross  Abstract: Recent work shows that large language models (LLMs) can operate directly on the parameters of other neural networks. They can read what a network was trained to do, fix a network that fails, and write working networks from scratch, all without gradient descent. This capability has been termed Weight-Space Programming (WSP). Exploring this new paradigm systematically requires a robust benchmark. We introduce WEIGHTBENCH, a benchmark that evaluates LLMs of increasing capability on 42 WSP tasks across five tracks: reading, fixing and writing weights, recognizing edits that are impossible, and data-free compression. The networks whose weights the LLMs read and edit range from 2K-parameter control policies to 7B-parameter language models. Experimental benchmark results show that frontier agents can already read small networks and write working control policies without ever running them, but cannot yet patch language models, compress them wi
    
[^2]: 模仿引导的多智能体强化学习

    Imitation Bootstrapped Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.18543](https://arxiv.org/abs/2611.18543)

    该论文提出IBMARL框架，将模仿引导强化学习扩展到多智能体协作场景，通过轮替行为克隆训练模仿策略，并利用集中式评论家集成在模仿策略与RL策略的动作提议中进行选择，从而克服了单演示者难以同时操控多个智能体以及演示质量有限的难题。

    

    演示是加速单智能体强化学习（RL）的有效方法，但将这一思想扩展到多智能体强化学习（MARL）并非易事，因为让单个人同时远程操控多个智能体十分困难，且往往无法实现。轮替行为克隆通过让单个演示者每次只控制一个智能体、而其他智能体执行其当前策略的方式解决了这一问题。然而，由此得到的模仿策略受到演示质量的限制。我们提出了模仿引导多智能体强化学习（IBMARL），这是一个将模仿引导强化学习（IBRL）扩展到协作团队、并使用R2BC来训练模仿策略的框架。在每个时间步，每个智能体分别从其模仿策略和RL策略各获得一个动作提议，然后由一组集中式评论家集成在每个智能体的组合中进行选择……

    arXiv:2611.18543v1 Announce Type: new  Abstract: Demonstrations are an effective way to accelerate single-agent reinforcement learning (RL), but extending this idea to multi-agent reinforcement learning (MARL) is non-trivial as it is difficult and often impossible for a single person to teleoperate multiple agents simultaneously. Round-Robin Behavior Cloning (R2BC) addresses this by letting a single demonstrator control one agent at a time while the others execute their current policies. However, the resulting imitation policies are limited by the quality of the demonstrations. We propose Imitation Bootstrapped Multi-Agent Reinforcement Learning (IBMARL), a framework that extends Imitation Bootstrapped Reinforcement Learning (IBRL) to cooperative teams and uses R2BC to train the imitation policies. At each time step, every agent receives one action proposal from its imitation policy and one from its RL policy, and an ensemble of centralized critics selects among the per-agent combinati
    
[^3]: 椭圆采样：面向高斯分布及更广泛情形的最优插值调度

    Elliptic Sampling: Optimal Interpolant Schedules for Gaussians and Beyond

    [https://arxiv.org/abs/2611.18542](https://arxiv.org/abs/2611.18542)

    该论文提出了一种基于复化ODE轨迹解析性质的插值调度设计新框架，通过保角映射精确求解得到由雅可比椭圆函数参数化的保方差椭圆调度族，并结合Zolotarev最优有理函数逼近构建了能根据目标分布条件数自适应优化的新采样算法Zoloflow。

    

    基于随机插值的生成建模需要确定两个关键选择：连接初始分布与目标分布的插值调度，以及由此得到的概率流ODE的高效离散化方法。在这项工作中，我们提出了一个基于（复化后）ODE轨迹解析性质的调度/离散化联合设计新框架，该解析性质进而保证了良好的数值行为。我们的公式可以通过保角映射精确求解，从而得到一个由雅可比椭圆函数参数化的、自然的保方差椭圆调度族。将椭圆调度与量身定制的离散化方法相结合，我们得到了Zoloflow——一种与Zolotarev最优有理函数逼近密切相关的新采样算法。关键之处在于，椭圆调度与Zoloflow都显式地纳入了目标分布的条件数 $\kappa$，并可退化为经典的……（摘要截断）

    arXiv:2611.18542v1 Announce Type: cross  Abstract: Generative modeling based on stochastic interpolants requires specifying two key choices: an interpolant schedule bridging the initial and target distributions, and an efficient discretization of the resulting probability-flow ODE. In this work, we introduce a new framework for joint schedule/discretization design based on the analyticity properties of the (complexified) ODE trajectories, which in turn ensure good numerical behavior. Our formulation can be solved exactly via conformal mapping, yielding a natural family of variance-preserving elliptic schedules, parametrized by Jacobi elliptic functions. Pairing an elliptic schedule with a tailored discretization, we obtain Zoloflow, a new sampling algorithm closely connected to Zolotarev's optimal rational function approximants. Crucially, both the elliptic schedule and Zoloflow explicitly incorporate the condition number $\kappa$ of the target distribution and reduce to the classical 
    
[^4]: ModHarness-RL：将支架重构为可学习、可演化的程序以服务于智能体强化学习

    ModHarness-RL: Recasting Harnesses as Learnable and Evolving Programs for Agentic RL

    [https://arxiv.org/abs/2611.18533](https://arxiv.org/abs/2611.18533)

    提出 ModHarness-RL，将智能体强化学习中固定不变的手写支架重构为可从数据学习并随训练不断演化的程序，通过压缩程序（组装上下文）和引导程序（行为偏离时提供指导）两个核心组件，从轨迹中提出候选程序并经执行验证而持续进化。

    

    智能体强化学习（RL）日益依赖于一个“支架”，即围绕模型的脚手架，它决定模型能看到自身交互中的哪些内容，以及模型会接收到什么样的引导。大多数现有方法都在一个全程固定不变的手写支架下进行训练。然而，一个有效的支架取决于领域、数据和模型本身，并且会随着模型的训练而持续变化。我们提出 ModHarness-RL，它将支架重构为从数据中学习、并随着模型通过强化学习训练而不断演化的程序。ModHarness-RL 将交互记录为结构化的状态，即一组带类型的块，而支架的两个核心组件就是基于这些块的程序：一个压缩程序，负责将积累的历史组装进模型的上下文中；一个引导程序，在行为发生偏离时提供指导。在强化学习训练过程中，候选程序从最近的轨迹中被提出，通过执行进行验证，并在过去的（轨迹上）进行重放。

    arXiv:2611.18533v1 Announce Type: new  Abstract: Agentic reinforcement learning (RL) increasingly relies on a harness, the scaffold around the model that decides what it sees of its own interaction and what guidance it receives. Most existing approaches train under a hand-written harness that stays fixed throughout. However, an effective harness depends on the domain, the data and the model, and keeps changing as the model trains. We introduce ModHarness-RL, which recasts the harness as programs that are learned from data and evolve as the model trains with RL. ModHarness-RL records the interaction as structured State, a set of typed blocks, and two core components of the harness are programs over these blocks, a Compression program that assembles the accumulated history into the model's context and a Steering program that supplies guidance when behavior deviates. During RL training, candidate programs are proposed from recent trajectories, verified by execution, and replayed over past
    
[^5]: 通过社会化实现智能体对齐：选择压力下的权衡学习

    Alignment by Socialization: Learning Tradeoffs Under Selection

    [https://arxiv.org/abs/2611.18531](https://arxiv.org/abs/2611.18531)

    该论文提出“社会化”这一对齐机制：在进化选择压力下，学习型智能体会收敛到环境所奖励的任务完成与行为规范之间的权衡；当由人类提供奖励时，智能体的选择便能反映人类判断并经受住后续的选择压力。

    

    人工智能（AI）智能体被期望在遵循行为准则的同时完成任务，但这些目标之间可能存在冲突。我们研究了进化选择如何塑造智能体在这两者之间的抉择，以及经验能否使这些选择反映人类的判断并承受后续的压力。在我们的模型中，一个智能体的多个副本面对共同的不确定奖励状态，只有采取了获得奖励行为的副本才能繁殖。我们证明，当每个行为被采取的频率与环境对其的奖励频率一致时，长期增长被最大化。我们比较了三种策略：仅任务策略在只奖励任务完成的沙盒环境中表现良好，但在行为规范至关重要时会失败；固定线性策略随着任务利害关系的增加越来越偏向任务完成，即使在人类从不奖励这种偏向的情况下也是如此；而学习型策略则会收敛到其环境所奖励的权衡上；当人类提供这些奖励时，我们将此过程称为社会化。积累的经验……

    arXiv:2611.18531v1 Announce Type: new  Abstract: Artificial intelligence (AI) agents are expected to complete tasks while following rules of conduct, but these objectives can conflict. We study how evolutionary selection shapes an agent's choices between them, and whether experience can make those choices reflect human judgments and withstand subsequent pressures. In our model, copies of an agent face a common uncertain reward state, and only those taking the rewarded action reproduce. We show that long-run growth is maximized when each action is taken as often as the environment rewards it. We compare three policies. A task-only policy thrives in a sandbox rewarding completion alone but fails when conduct matters. A fixed linear policy increasingly favors task completion as its stakes rise, even where humans never reward it. A learned policy instead converges to the tradeoffs rewarded by its environment; when humans supply the rewards, we call this socialization. Accumulated experienc
    
[^6]: 机制可解释性能否帮助解释大语言模型智能体的失败？——网页元素幻觉的案例研究

    Can Mechanistic Interpretability Help Explain LLM Agent Failures? A Case Study on Web Element Hallucination

    [https://arxiv.org/abs/2611.18514](https://arxiv.org/abs/2611.18514)

    本文首次将机制可解释性系统应用于网页智能体幻觉失败分析，发现线性探针可在生成前预测幻觉（0.80 AUC）且引导能纠正85%的幻觉，但由于对提示格式敏感，这只是一种部分解释。

    

    机制可解释性（MI）在解释大语言模型（LLM）的内部计算方面已取得显著进展；然而，这些进展尚未在解释基于大语言模型的智能体的决策过程方面展现出同等的成功。本文探讨了MI工具在应用于真实智能体失败场景时所面临的挑战。我们聚焦于其中一种失败：网页智能体幻觉出网站上不存在的操作。我们对与真实网站交互的Qwen2.5-7B-Instruct应用了一系列机制分析。我们发现，基于最后一个提示token训练的线性探针能够在生成开始之前预测幻觉（0.80 AUC），并且沿探针方向进行引导可以纠正85%的幻觉，但会干扰7%的有效操作。然而，这些结果仅能部分解释智能体为何产生幻觉：探针方向对提示格式非常敏感，因而丧失了大部分……

    arXiv:2611.18514v1 Announce Type: new  Abstract: Mechanistic Interpretability (MI) has made significant progress in explaining the internal computations of Large Language Models (LLMs); however, these advances have yet to demonstrate comparable success in explaining the decision-making processes of LLM-based agents. This paper examines the challenges MI tools face when applied to realistic agentic failures. We focus on one such failure: web agents hallucinating actions that are not available on a website. We apply a series of mechanistic analyzes to Qwen2.5-7B-Instruct as it interacts with live websites. We find that a linear probe trained on the last prompt token can predict hallucinations before generation begins (0.80 AUC) and steering along a probe direction corrects 85\% of hallucinations, but disrupts 7\% of valid actions. However, these results only provide a partial explanation of why the agent hallucinates. The probe direction is sensitive to prompt formatting, losing much of 
    
[^7]: MeReader：面向叙事类电子书阅读的离线、隐私保护、进度感知AI助手

    MeReader: An Offline, Privacy-Preserving, Progress-Aware AI Assistant for Narrative eBook Reading

    [https://arxiv.org/abs/2611.18513](https://arxiv.org/abs/2611.18513)

    MeReader是一个离线、隐私保护且进度感知的电子书阅读AI助手，通过混合向量相似度、BM25与摘要嵌入的检索架构，将内容检索与生成严格限制在读者当前阅读进度内以避免剧透，并证明2b-4b的中等规模本地模型可在可接受延迟下达到与更大模型相当的理解能力。

    

    本文介绍了MeReader，这是一个隐私保护、离线且进度感知的AI助手，旨在作为消费级硬件上叙事阅读的上下文伴侣。与传统的基于大语言模型（LLM）的系统不同——后者以一次性处理全部文本的方式在线处理文档/书籍，从而存在叙事剧透的风险——MeReader将检索和生成都限制在读者在文本中的具体阅读位置。该系统集成了混合检索架构，结合向量相似度、BM25和摘要嵌入，以确保对读者当前上下文的高保真度。我们使用定量理解指标和基于LLM的定性独立评审，在经典英文书籍上对该系统进行了评估。结果确定了一个质量-延迟边界，表明中等规模的本地模型（2b-4b）可以在保持可接受（延迟）的同时，达到与更大模型（5b+）相当的理解能力。

    arXiv:2611.18513v1 Announce Type: cross  Abstract: This paper introduces MeReader, a privacy-preserving, offline and progress-aware AI assistant designed to function as a contextual companion for narrative reading on consumer hardware. Unlike conventional Large Language Model (LLM)-based systems that process documents/books online in an all-texts-at-once type of way, thereby risking narrative spoilers, MeReader confines both retrieval and generation to the reader's distinct position in the text. The system integrates a hybrid retrieval architecture, combining vector similarity, BM25, and summary embeddings, to ensure high fidelity to the reader's current context. We evaluate the system across classical English books using both quantitative comprehension metrics and qualitative LLM-based independent judging. The results identify a quality-latency frontier, showing that mid-sized local models (2b-4b) can achieve comprehension parity with larger counterparts (5b+) while maintaining accept
    
[^8]: 用于发现容错量子线路的智能体人工智能

    Agentic AI for Discovering Fault-Tolerant Quantum Circuits

    [https://arxiv.org/abs/2611.18504](https://arxiv.org/abs/2611.18504)

    该论文提出将容错量子线路的设计问题表述为智能体式进化程序合成，利用大型语言模型在电路验证器引导下自动进化生成用于逻辑态制备、逻辑门设计和码切换等任务的容错克利福德线路，从而取代传统的手工设计方法。

    

    arXiv:2611.18504v1 公告类型：cross。摘要：将量子计算扩展到实用化的量子优势，目前受限于基于量子纠错码编码信息运行的线路效率。量子计算的完整技术栈涵盖若干任务：逻辑码态的制备、逻辑门的实现、以及作用于编码量子比特的逻辑例程等。然而，这些线路目前大多是为单个案例手工设计的，近期仅有一些工作在某些方面引入了机器学习。我们将这一问题表述为智能体式进化程序合成：大型语言模型利用其预训练知识和工具使用能力，并在电路验证器的引导下，进化出一个程序，该程序为一系列精心挑选的任务定义克利福德线路，包括逻辑态制备、逻辑门设计，以及码切换、码增长等逻辑例程。我们的方法能够持续找到容错的（原文摘要在此处截断）

    arXiv:2611.18504v1 Announce Type: cross  Abstract: Scaling quantum computing toward practical quantum advantage is currently limited by the efficiency of circuits that operate on information encoded by quantum error-correcting codes. The full stack of quantum computing encompasses several tasks: preparation of logical code states, implementation of logical gates, logical routines on encoded qubits, etc. Yet, these circuits are largely designed by hand for individual cases, with some recent works incorporating machine learning for certain aspects. We formulate this problem as agentic evolutionary program synthesis, wherein a large language model, leveraging its pretraining knowledge and tool-use proficiency, and guided by a circuit verifier, evolves a program that defines a Clifford circuit for a wide range of curated tasks such as logical state preparation, logical gate design, and logical routines such as code-switching, code growth, etc. Our approach consistently finds fault-tolerant
    
[^9]: ARCHER：小型语言模型从推理依据模仿到可执行推理

    ARCHER: From Rationale Imitation to Executable Reasoning in Small Language Models

    [https://arxiv.org/abs/2611.18494](https://arxiv.org/abs/2611.18494)

    ARCHER提出两阶段训练框架，先将自由格式的思维链编译为包含计算器调用与验证的类型化动作轨迹，再通过非对称分层强化学习优化策略，使小型语言模型学会生成可执行、可审计的数学推理轨迹。

    

    小型语言模型因其适合本地、低成本和隐私敏感的部署场景而具有吸引力，但在需要任务分解、精确计算、工具使用和验证的数学推理任务上仍然不可靠。我们提出了ARCHER（Action-trace Reinforcement for Calculator-grounded Hierarchical Executable Reasoning，基于计算器的分层可执行推理的动作轨迹强化），这是一个用于训练小型语言模型（SLM）生成可执行且可审计推理轨迹的两阶段框架。在第一阶段，ARCHER将自由格式的思维链推理依据编译为类型化的动作轨迹，其中包含推理步骤、计算器调用、确定性运行时观测、验证动作和最终答案。这将监督目标从模仿解释性文本转变为学习一种基于工具的推理策略。在第二阶段，ARCHER通过非对称分层强化学习对这一策略进行进一步优化（原文摘要在此处截断）。

    arXiv:2611.18494v1 Announce Type: new  Abstract: Small language models are attractive for local, low-cost, and privacy-sensitive deployment, but they remain unreliable on mathematical reasoning tasks that require decomposition, exact computation, tool use, and verification. We introduce \textbf{ARCHER}: \textbf{A}ction-trace \textbf{R}einforcement for \textbf{C}alculator-grounded \textbf{H}ierarchical \textbf{E}xecutable \textbf{R}easoning, a two-stage framework for training small language models (SLMs) to produce executable and auditable reasoning trajectories. In the first stage, ARCHER compiles free-form chain-of-thought rationales into typed action traces containing reasoning steps, calculator calls, deterministic runtime observations, verification actions, and final answers. This shifts the supervision target from imitating explanatory text to learning a tool-grounded reasoning policy. In the second stage, ARCHER refines this policy through asymmetric hierarchical reinforcement le
    
[^10]: TradeGauntlet：评估自主交易代理的合规控制

    TradeGauntlet: Evaluating Compliance Controls for Autonomous Trading Agents

    [https://arxiv.org/abs/2611.18493](https://arxiv.org/abs/2611.18493)

    论文提出 TradeGauntlet——一个可回放的限价订单簿模拟器，用于评估自主交易代理的合规控制，结果表明仅凭零警报不足以证明智能体合规，且对抗性搜索能发现规避现有行为规则的拆分订单策略。

    

    自主交易代理需要能够防止问题订单、同时又不至于让合法任务变得难以执行的控制机制。我们通过 TradeGauntlet 研究智能体合规的这一方面——这是一个可回放的限价订单簿模拟器，将检测、预防和执行成本三者分离。在使用 Claude Haiku 5.5 和 Sonnet 5.5 进行的 1,680 次模拟会话中，所有行为警报均为自成交。门控机制通过拒绝自交叉订单消除了这些警报，而聚合门控条件使 Sonnet 的双边成交减少了约 15%。行为提醒降低了 Sonnet 的警报率，但在没有可检测到的警报收益的情况下降低了 Haiku 的成交量。这些结果表明，仅凭零警报不足以确立智能体的合规性。脚本化探针和两种搜索方法用于测试大模型运行中未观察到的弱点。搜索发现了能够规避已实现行为规则的拆分订单计划；一个学习得到的基线捕获了全部 2,400 个经过测试的搜索样本。

    arXiv:2611.18493v1 Announce Type: cross  Abstract: Autonomous trading agents need controls that prevent problematic orders without making legitimate tasks impractical. We study this aspect of agentic compliance with TradeGauntlet, a replayable limit-order-book simulator that separates detection, prevention, and execution cost. In 1,680 simulated sessions with Claude Haiku 5.5 and Sonnet 5.5, all conduct alerts were self-matches. Gates eliminated those alerts by rejecting self-crossing orders, while the aggregate-gate condition reduced Sonnet's two-sided fills by about 15%. A conduct reminder reduced Sonnet's alert rate but reduced Haiku's fills without a detectable alert benefit. These results show why zero alerts alone do not establish an agent's compliance. Scripted probes and two search methods test weaknesses beyond those observed in the LLM runs. The searches find split-order schedules that evade the implemented conduct rules; a learned baseline catches all 2,400 tested searched e
    
[^11]: 基于语言模型先验的交互式程序归纳

    Interactive Program Induction with Language Model Priors

    [https://arxiv.org/abs/2611.18485](https://arxiv.org/abs/2611.18485)

    本文提出以语言模型作为程序先验的贝叶斯交互式程序归纳方法，通过贪心选择最具信息量的输入向预言机查询，逐步将后验概率集中于目标程序，以化解从有限示例归纳程序时固有的歧义性问题。

    

    程序归纳是指学习者从少量给定的输入-输出示例中恢复出程序的过程。这一范式通过让用户直接演示他们希望应用于未见数据上的基于规则的行为，而无需亲自实现该行为，从而降低了编程的门槛。然而，从有限的输入-输出示例中恢复程序在本质上存在歧义性，因为一个有限的输入-输出示例集在理论上与无穷多个程序相容，而这些程序在未见输入上的输出可能各不相同。我们从贝叶斯的视角来审视这一问题：在给定程序先验分布 p⁰ 的情况下，识别目标程序意味着通过观察更多的输入-输出示例，将后验概率质量逐步集中到该程序上。在交互式程序归纳的设定中，学习者会主动选择信息量最大的输入，并向预言机（例如用户）查询其对应输出。我们推导了一种贪心的查询选择算法……

    arXiv:2611.18485v1 Announce Type: cross  Abstract: Program induction refers to the process in which a learner aims to recover a program from a few given input-output examples. This paradigm broadens programming access by letting users demonstrate the rule-based behavior they want to apply to unseen data, instead of having to implement it themselves. However, recovering a program from finite input-output examples is fundamentally ambiguous, as a finite set of input-output examples is theoretically consistent with infinitely many programs whose output diverges on unseen inputs. We view this problem from a Bayesian perspective, where, given a prior $p^0$ over programs, identifying the target program means concentrating posterior probability mass on it by observing additional input-output examples. In the interactive program induction setting, the learner chooses maximally informative inputs and queries an oracle, such as the user, for their outputs. We derive a greedy query-selection algo
    
[^12]: 从AI赋能的远程患者监测迈向智能体机器人：具身化、治理与护理整合

    From AI-Enabled Remote Patient Monitoring Toward Agentic Robotics: Embodiment, Governance, and Care Integration

    [https://arxiv.org/abs/2611.18484](https://arxiv.org/abs/2611.18484)

    本文通过结构化范围综述指出，AI远程患者监测正迈向具身智能体机器人护理，但成熟的智能体护理尚未实现，未来需要以具身化、人类监督、治理和安全可互操作的护理架构为基础，实现从监测到行动的转变。

    

    AI赋能的远程患者监测正在从被动的数据收集演进为具有能动性的AI系统，这类系统能够解读患者状态、协调护理，并在家庭和临床环境中发起行动。这项结构化的范围综述发现，成熟的智能体护理尚未实现：大多数系统仍由人类主导，而更强的决策-行动耦合和具身AI在很大程度上仍局限于原型与试点阶段。医疗机器人通过将AI生成的建议和决策转化为语音、动作、提示、物理辅助、升级上报和护理协调，使从监测到行动的转变尤为直观，这也使具身化、临床整合、人类监督和治理成为基本的设计要求。进一步的进展需要安全且可互操作的护理架构，使智能组件能够在经过临床验证且可问责的工作流程中运行。我们提出了一种……（原文摘要在此处截断）

    arXiv:2611.18484v1 Announce Type: cross  Abstract: AI-enabled remote patient monitoring is evolving beyond passive data collection toward agentic AI systems that interpret patient states, coordinate care, and initiate action across domestic and clinical settings. This structured scoping review finds that mature agentic care has not yet emerged: Most systems remain human led, while stronger decision-action coupling and embodied AI remain largely confined to prototypes and pilots. Healthcare robotics makes the transition from monitoring to action especially visible by translating AI-generated recommendations and decisions into speech, movement, prompting, physical assistance, escalation, and care coordination, making embodiment, clinical integration, human oversight, and governance fundamental design requirements. Further progress requires safe and interoperable care architectures in which intelligent components operate through clinically validated and accountable workflows. We propose a
    
[^13]: 验证器的判断能否经受住抽象化？面向LLM智能体评估的验证迁移测量

    Do Verifier Judgments Survive Abstraction? Measuring Verification Transport for LLM Agent Evaluation

    [https://arxiv.org/abs/2611.18471](https://arxiv.org/abs/2611.18471)

    本文提出“验证迁移”（Verification Transport）这一可靠性测试方法，通过在保持任务、轨迹、结果等条件不变的情况下构建原生、抽象和恢复三种匹配视图，测量LLM验证器的判断在经历工具组合、委托边界、历史压缩等抽象化后是否仍能保持稳定可靠。

    

    arXiv:2611.18471v1 公告类型：交叉。摘要：基于LLM的验证器被越来越多地用于评估智能体的执行过程，但其可靠性通常只在轨迹的单一表示形式下进行测量。在部署的智能体系统中，同一次执行可能通过组合工具、委托边界或压缩的历史记录等方式暴露出来，这改变了验证器可直接观察到的执行证据。因此，正确的验证器判断是否能在这些抽象边界之间保持一致，目前仍不清楚。我们提出了“验证迁移”（Verification Transport），这是一种可靠性测试，用于测量在抽象引起的执行可观察性变化下判断的稳定性。我们在保持任务、实际执行轨迹、结果、真值、评估标准和验证器全部固定不变的前提下，构建了同一次冻结执行的匹配的原生视图、抽象视图和恢复视图。我们在四个智能体评估基准、七种验证器骨干、四种验证协议和三种抽象操作（实验在摘要中被截断）上进行了测试。

    arXiv:2611.18471v1 Announce Type: cross  Abstract: LLM-based verifiers are increasingly used to evaluate agent executions, but their reliability is typically measured under a single representation of the trajectory. In deployed agent systems, the same execution may be exposed through composed tools, delegation boundaries, or compacted histories, changing what execution evidence is directly observable to the verifier. It therefore remains unclear whether correct verifier judgments persist across such abstraction boundaries. We introduce Verification Transport, a reliability test that measures judgment stability under abstraction-induced changes in execution observability. We construct matched native, abstracted, and restored views of the same frozen executions while holding the task, realized trajectory, outcome, ground truth, evaluation criterion, and verifier fixed. Across four agent-evaluation benchmarks, seven verifier backbones, four verification protocols, and three abstraction bo
    
[^14]: 打字至关重要：基于sEMG实时按键解码中的肌肉、语言与打字行为

    Typing Matters: Muscle, Language and Typing Behaviour in Live Keystroke Decoding from sEMG

    [https://arxiv.org/abs/2611.18464](https://arxiv.org/abs/2611.18464)

    该论文提出一种基于状态空间模型（SSM）的完全因果、零前瞻sEMG打字解码器，将肌肉信号、语言结构与打字行为三者联合建模，实现手部持续移动时的实时按键预测。

    

    来自手腕的表面肌电信号（sEMG）可以在没有键盘的情况下恢复打出的文本。这通常被表述为一个信号推断问题：从肌肉信号中读出字符，并离线报告字符错误率。我们认为这种框架忽略了问题的大部分内容。解码器同时受到肌肉、正在输入的语言以及打字行为（如拼写错误、纠正、退格和换行）的约束，而一个可用的系统必须在手部持续移动时实时预测按键。现有系统在两个方面存在不足：（1）它们以固定窗口处理sEMG信号，且不在窗口之间传递任何信息，丢弃了在整个会话中积累的上下文；（2）它们缺乏对驱动打字的语言结构的理解，因此每一次按键都仅凭肌肉信号进行推断。我们提出了一种完全因果的、零前瞻的基于状态空间模型（SSM）的sEMG打字解码器……

    arXiv:2611.18464v1 Announce Type: new  Abstract: Surface electromyography (sEMG) from the wrists can recover typed text without a keyboard. It is usually posed as a signal-inference problem: characters are read out of the muscle signal and character error rate is reported offline. We argue that this framing removes most of the problem. The decoder is constrained simultaneously by the muscles, by the language being typed, and by typing behaviour such as typos, corrections, backspaces and line breaks, and a usable system must predict keys live while the hand keeps moving. Existing systems fall short on two counts: (1) they process sEMG in fixed windows and carry nothing from one window to the next, discarding context accumulated over a session, and (2) they lack any understanding of the linguistic structure that drives typing, so every keystroke is inferred from the muscle signal alone. We propose a fully causal, zero-lookahead sEMG typing decoder built on a state space model (SSM), in w
    
[^15]: 何时信任模拟的患者反应：用于血管加压药决策支持的支持门控世界模型

    When to Trust Simulated Patient Responses:Support-Gated World Models for Vasopressor Decision Support

    [https://arxiv.org/abs/2611.18458](https://arxiv.org/abs/2611.18458)

    该论文提出支持门控世界模型决策支持方法，利用行为模型估计的动作支持度与动力学集成模型的分歧度进行联合门控，在模拟预测不可靠时主动隐藏预测或弃权，从而在eICU-CRD数据上以62%覆盖率将MAP预测误差从9.2 mmHg降至7.3 mmHg。

    

    临床世界模型让语言模型智能体接触预测的患者反应，但模拟对于支持度不足的治疗或不熟悉的医院可能不可靠。我们提出了支持门控世界模型决策支持方法，这是“提议—模拟—精炼”智能体的一个轻量级扩展。一个循环动力学集成模型预测对所记录的去甲肾上腺素剂量类别的短期生理反应，同时一个行为模型估计观察性动作支持度。该界面会在支持不足或集成分歧过大时隐藏预测，并在没有候选方案合格时选择弃权。在eICU-CRD上的评估涵盖70家医院和30,700个决策转换，其中14家医院留作测试。在62.0%的事实转换覆盖率下，联合门控的MAP平均绝对误差为7.3 mmHg，而相同覆盖率下随机选择为9.2 mmHg。门控智能体与记录达到60.0%的一致性。

    arXiv:2611.18458v1 Announce Type: new  Abstract: Clinical world models expose language-model agents to predicted patient responses, but simulations can be unreliable for poorly supported treatments or unfamiliar hospitals. We introduce Support-Gated World-Model decision support, a lightweight extension of a propose--simulate--refine agent. A recurrent dynamics ensemble predicts short-horizon physiological responses to documented norepinephrine dose categories, while a behavior model estimates observational action support. The interface withholds predictions with insufficient support or excessive ensemble disagreement and abstains when no candidate qualifies. Evaluation on eICU-CRD includes 70 hospitals and 30,700 decision transitions, with 14 hospitals held out for testing. At 62.0% factual-transition coverage, joint gating yields a MAP mean absolute error of 7.3 mmHg, compared with 9.2 mmHg for random selection at the same coverage. The gated agent achieves 60.0% agreement with record
    
[^16]: 基于集成之集成的大语言模型黑盒不确定性量化

    Black-Box Uncertainty Quantification for Large Language Models via Ensemble-of-Ensembles

    [https://arxiv.org/abs/2611.18453](https://arxiv.org/abs/2611.18453)

    该论文提出了一种仅需黑盒访问的两级“集成之集成”方法，通过对固定提示词随机解码构成内层集成、对提示词进行语义扰动构成外层集成，并利用全协方差定律在嵌入空间中精确分解大语言模型的偶然性不确定性与认知性不确定性。

    

    可靠地使用大语言模型（LLM）需要不确定性估计，以区分提示词固有的模糊性与真正的知识缺口。贝叶斯方法和深度集成方法在原则上能够提供这种偶然性/认知性的分解，但在LLM规模上计算成本过高，且需要白盒访问。相比之下，现有的黑盒方法只能将其总结为单一标量，无法做出上述区分。我们引入了一种两级集成方法来弥合这一差距。对固定提示词进行随机解码构成内层集成；对提示词进行保义语义扰动构成外层集成。变异性在连续嵌入空间中进行测量，全协方差定律给出了精确的扰动内/扰动间分解，从而估计偶然性不确定性与认知性不确定性（AU/EU）；该方法仅需要黑盒采样访问。在此估计器的基础上，我们提出了三项理论结果（摘要在此处被截断）

    arXiv:2611.18453v1 Announce Type: cross  Abstract: Reliable use of large language models (LLMs) requires uncertainty estimates that separate ambiguity intrinsic to a prompt from genuine knowledge gaps. Bayesian and deep-ensemble methods deliver this aleatoric/epistemic decomposition in principle, but are computationally prohibitive at LLM scale and require white-box access. Existing black-box methods, in contrast, summarize to a single scalar that cannot make the distinction. We introduce a two-level ensemble that bridges this gap. Stochastic decoding of a fixed prompt forms the inner ensemble; meaning-preserving semantic perturbations of the prompt form the outer ensemble. Variability is measured in a continuous embedding space, and the law of total covariance gives an exact within-/between-perturbation decomposition to estimate aleatoric and epistemic uncertainty (AU/EU); it requires only black-box sampling access. On top of this estimator, we contribute three theoretical results tha
    
[^17]: 在义务与证据之间：将欧盟《人工智能法案》转化为技术要求的参与式研究

    Between Obligations and Evidence: A Participatory Study of Translating the EU AI Act into Technical Requirements

    [https://arxiv.org/abs/2611.18439](https://arxiv.org/abs/2611.18439)

    该研究通过32名跨学科专家参与的参与式研讨会，揭示了将欧盟《人工智能法案》义务转化为技术要求时在可测量性、责任归属与人工监督实践、以及合规证据在动态运行条件下的有效性方面反复出现的张力。

    

    欧盟《人工智能法案》对高风险人工智能系统提出了技术和组织方面的要求，但应用这些要求涉及情境化的选择——即应该测量什么，以及什么证据能够支持合规性声明。我们通过一场由来自不同学科的32名研究人员、政策制定者和从业者参与的参与式研讨会来研究这些挑战，并采用医疗健康、招聘和公共交通领域的现实场景，考察监管义务如何被解读、转化为技术要求，并在实践中得到评估。通过对研讨会数据的主题分析，我们识别出反复出现的张力：监管义务如何转化为可测量的系统属性、责任与人工监督在实践中如何行使，以及合规证据的有效性如何受到运行条件变化和系统用途改变的影响。例如，参与者强调了公平……（原文摘要在此处截断）

    arXiv:2611.18439v1 Announce Type: cross  Abstract: The EU AI Act imposes technical and organizational requirements for high-risk AI systems, but applying these requirements involves contextual choices about what to measure and what evidence supports a compliance claim. We investigate these challenges through a participatory workshop involving 32 researchers, policymakers, and practitioners from different disciplines, using realistic scenarios in healthcare, recruitment, and public transport to examine how regulatory obligations are interpreted, translated into technical requirements, and evaluated in practice. Through a thematic analysis of workshop data, we identify recurring tensions between how regulatory obligations are translated into measurable system properties, how responsibility and human oversight are exercised in practice, and how the validity of compliance evidence is affected by changing operational conditions and system uses. For example, participants highlighted how fair
    
[^18]: 基于单通道脑电图的可解释睡眠阶段分类研究

    Towards Explainable Sleep Stage Classification using Single-Channel Electroencephalogram

    [https://arxiv.org/abs/2611.18434](https://arxiv.org/abs/2611.18434)

    该论文扩展了SleepXAI睡眠分期框架，系统比较五种XAI方法，并以AASM定义的脑电频段的Spearman秩相关评估归因保真度，实现了基于单通道脑电、兼顾分类性能与临床可解释性的睡眠阶段分类。

    

    自动化睡眠分期已达到很强的基准准确率；然而，由于缺乏可解释性，其临床应用仍然受限。在这项工作中，我们通过扩展SleepXAI睡眠分期框架来系统比较五种不同的可解释人工智能（XAI）方法，填补了这一研究空白。此外，我们采用与AASM定义的脑电（EEG）频段之间的Spearman秩相关系数来评估归因保真度，将其作为具有临床依据的参考信号。本文聚焦于单通道脑电图（EEG），相比完整的多导睡眠图设备，它更便于实际临床部署。实验在Sleep-EDF基准数据集和DREAMT数据集上进行。在DREAMT数据集中，我们根据阻塞性睡眠呼吸暂停（OSA）将受试者分为四个呼吸暂停低通气指数（AHI）严重程度组。我们观察到SleepXAI模型在两个数据集上均取得了很强的分类性能。在所评估的方法中，积分梯度法

    arXiv:2611.18434v1 Announce Type: new  Abstract: Automated sleep staging has achieved strong bench- mark accuracy; however, clinical adoption remains limited due to a lack of interpretability. In this work, we fill the research gap by extending the SleepXAI sleep staging framework to systematically compare five different XAI methods. Further, attribution fidelity is evaluated using Spearman rank correlation with AASM-defined EEG frequency bands, serving as clinically grounded reference signals. In this paper, we focus on single- channel Electroencephalogram (EEG), which is more amenable to practical clinical deployment compared to full polysomnography setups. Experiments are conducted on the Sleep-EDF benchmark dataset and the DREAMT dataset. In DREAMT, we stratified subjects based on OSA into four Apnea-Hypopnea Index (AHI) severity groups. We observed that the SleepXAI model achieves strong classification performance on both datasets. Among the evaluated methods, Integrated Gradients
    
[^19]: LECAT：衡量大语言模型从预训练数据中的类别学习

    LECAT: Measuring LLM Category Learning from Pretraining Data

    [https://arxiv.org/abs/2611.18419](https://arxiv.org/abs/2611.18419)

    本文提出 LeCat 两阶段框架，一方面评估大语言模型在记忆、泛化和灵活运用三个层面上的类别知识学习能力，另一方面通过追踪预训练文本中不同的“知识表达”形式来揭示这类知识在预训练中的习得机制。

    

    人类知道飞机很快，这一属性也适用于红色的飞机，但不适用于损坏的飞机。大语言模型（LLM）是如何习得这些人类能够轻松学习和应用的类别知识的？我们提出了 LeCat，一个用于分析大语言模型类别知识学习的两阶段框架。首先，我们使用 LeCat 评估模型学习类别知识的能力，区分三种情况：对域内类别知识的记忆、对未见实例的泛化，以及在合适语境中灵活运用知识的能力。然后，我们使用 LeCat 追踪这些知识如何在预训练过程中从文本中习得——这些文本可能以各种方式编码类别知识，例如逐字重复或例外情况，我们将其称为不同的“知识表达”形式。我们首先使用 199 个精心整理的类别-特征对来评估开源预训练模型，并分析其训练数据。我们的发现表明，加强……（原文截断）

    arXiv:2611.18419v1 Announce Type: new  Abstract: Humans know that airplanes are fast, that this extends to red airplanes, but that it does not hold for broken airplanes. How do LLMs acquire category knowledge that humans learn and apply effortlessly? We introduce LeCat, a two-part framework to analyze LLM's Learning of Category knowledge. First, we use LeCat to evaluate models on their ability to learn category knowledge, distinguishing between three cases: memorization of in-domain category knowledge, generalization to unseen instances, and ability to flexibly use knowledge in appropriate contexts. Then, we use LeCat to trace how this knowledge can be learned in pretraining from text that may encode category knowledge in various ways, such as verbatim repetition or exceptions, which we refer to as forms of "knowledge expression". We first assess open-source pretrained models using 199 curated category-feature pairs and analyze their training data. Our findings suggest that strengtheni
    
[^20]: CoPre-MoE：面向多模态持续指令微调的专家混合模型整合与保留方法

    CoPre-MoE: Consolidating and Preserving MoE for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2611.18417](https://arxiv.org/abs/2611.18417)

    提出CoPre-MoE方法，通过为每个任务先学习任务特定MoE专家，再对跨任务冗余专家进行整合与保留，在避免灾难性遗忘的同时有效抑制多模态持续指令微调中的跨任务参数冗余和参数线性增长。

    

    多模态持续指令微调要求多模态大语言模型在依次获取新能力的同时不发生灾难性遗忘。专家混合（MoE）架构通过专家专业化与组合，为缓解任务间干扰提供了一种有前景的途径。然而，现有基于MoE的方法通过为每个任务分配独立专家来避免覆盖共享参数，但这种渐进式扩展会导致严重的跨任务冗余和参数线性增长。具体而言，不同任务的专家在学习任务特定能力的同时，往往会学到重叠的功能。例如，ScienceQA和TextVQA都需要定位相关的图像区域。为了在不引发遗忘的情况下抑制参数冗余，我们提出了整合与保留MoE（CoPre-MoE）。对于每个到来的任务，CoPre-MoE首先学习任务特定的MoE以捕捉专门化能力，从而消除冗余并减少参数……（摘要不完整）

    arXiv:2611.18417v1 Announce Type: cross  Abstract: Multimodal continual instruction tuning requires multimodal large language models to sequentially acquire new capabilities without catastrophic forgetting. Mixture-of-experts (MoE) architectures offer a promising way to mitigate task interference through expert specialization and combination. However, while existing MoE-based methods avoid overwriting shared parameters by allocating separate experts for each task, this progressive expansion incurs severe cross-task redundancy and linear parameter growth. Specifically, experts across tasks often learn overlapping functions alongside task-specific capabilities. For instance, ScienceQA and TextVQA both locate relevant image regions. To curb parameter redundancy without inducing forgetting, we propose Consolidating and Preserving MoE (CoPre-MoE). For each incoming task, CoPre-MoE first learns task-specific MoE to capture specialized capabilities. To eliminate redundancy and reduce paramete
    
[^21]: Gamer：面向长时程实时游戏智能体的通用视觉-语言-动作基础模型

    Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

    [https://arxiv.org/abs/2611.18397](https://arxiv.org/abs/2611.18397)

    提出Gamer，一个端到端的通用视觉-语言-动作基础模型，通过海量键鼠交互轨迹预训练、稀疏思考指令微调和稳定的多轮强化学习，实现了在复杂2D/3D交互环境中执行长时程任务的实时游戏智能体。

    

    我们提出了Gamer，一个专为复杂交互环境中长时程任务构建的通用智能体框架。其端到端架构结合了感知、记忆管理和多步推理。统一的训练与部署流程支持与常规2D图形界面以及复杂3D模拟世界的交互。我们首先在从数万小时设备级键盘和鼠标交互中构建的数据集上进行继续预训练，这些轨迹涵盖模拟游戏和具身环境，使模型能够学习跨物理与虚拟领域的环境动态。随后的稀疏思考指令微调在保留复杂决策所需推理能力的同时，大幅减少了冗余计算。我们采用一种稳定的多轮强化学习方法来训练策略以实现长时程交互，该方法解决了有限上下文窗口、累积（误差/奖励）等问题（摘要在此处被截断）。

    arXiv:2611.18397v1 Announce Type: cross  Abstract: We introduce Gamer, a generalist agent framework built for long-horizon tasks in complex interactive environments. Its end-to-end architecture combines perception, memory management, and multi-step reasoning. A shared training and deployment pipeline supports interaction with both conventional 2D graphical interfaces and complex 3D simulated worlds. We first continue pre-training on a dataset constructed from tens of thousands of hours of device-level keyboard and mouse interaction. These trajectories span simulated games and embodied environments, allowing the model to learn environmental dynamics across both physical and virtual domains. Sparse-thinking instruction tuning then substantially reduces redundant computation while preserving the reasoning needed for complex decisions. We train the policy for long-horizon interaction with a stable multi-turn reinforcement learning method that addresses finite context windows, accumulated r
    
[^22]: 智能家居代理可以成为有能力但不可预测的电网运营商

    Smart Home Agents Make Capable but Unpredictable Grid Operators

    [https://arxiv.org/abs/2611.18395](https://arxiv.org/abs/2611.18395)

    本文首次对智能家居代理在完整停电过程中的韧性进行分析，证明其可以在停电时充当电网运营商，同时识别出多种新的故障类别，为提升电网边缘整体韧性奠定了基础。

    

    由AI赋能的智能家居在电网边缘正变得越来越普遍。然而，关于故障响应的研究工作还很少。这一领域至关重要，因为智能家居运行在电网边缘，故障可能在此处级联扩散。如果智能家居内部发生停电，用户必须手动排查智能家居的故障，包括电力负载平衡和设备恢复等任务。我们首次对智能家居代理在完整停电过程中的韧性进行了分析，提供了一套在真实环境中运行的停电模拟，用以测量停电前、停电中以及停电后的韧性表现。我们展示了智能家居代理能够在停电时充当电网运营商，并识别出几种新的故障类别，揭示了当前所有模型存在的局限性。这些发现为智能家居代理作为电网运营商以提升电网边缘整体韧性奠定了基础。

    arXiv:2611.18395v1 Announce Type: new  Abstract: AI-enabled smart homes are becoming more common across the grid edge. However, there has not been much work on failure response. This area is critical because smart homes operate on the grid edge where failures can cascade down. If a power outage happens within a smart home, a user has to manually troubleshoot their smart home to include tasks such as power load balance and device recovery. We conduct the first resilience analysis of smart home agents across a full power outage, providing a suite of outage simulations in a live environment that measures resilience before, during, and post power outage. We show that smart home agents can act as grid operators in power outages and identify several new failure categories highlighting the current limitations across all of our models. These findings establish the groundwork for smart home agents as grid operators to increase overall resilience on the grid edge.
    
[^23]: VelServe：通过尾部校准调度驯服大语言模型推理中的解码抖动

    VelServe: Taming Decode Jitters in LLM Inference with Tail-Calibrated Scheduling

    [https://arxiv.org/abs/2611.18392](https://arxiv.org/abs/2611.18392)

    VelServe通过利用解码步骤自身迭代延迟的实测尾部数据来动态校准每步的预填充工作量，有效驯服LLM推理中的解码抖动，从而满足TTFT和TBT的延迟服务级目标。

    

    交互式大语言模型（LLM）服务必须满足针对首令牌时间（TTFT）和令牌间隔时间（TBT）的延迟服务级目标（SLO），因为流式传输和软实时应用需要可预测的令牌交付。我们在聚合式和分离式两种服务架构中都发现了解码抖动现象。一次抖动会延迟批处理中的每一个请求，而长回复跨越数百个解码步骤，因此即使是罕见的超时也会导致大量SLO违规。在聚合式服务中，SLO感知调度器会为每个步骤设定预填充预算，使其预测延迟恰好满足TBT目标，这导致系统没有为预测误差或主机停顿留下任何余地。分离式架构虽然以牺牲容量为代价满足了平均TBT目标，但接纳新请求的步骤仍然会延迟完成。我们提出了VelServe，一种利用解码步骤自身迭代延迟的实测尾部数据来校准每个步骤所承载预填充工作量的调度器。SLO决定了延迟步骤需要多么罕见，VelServe学习……

    arXiv:2611.18392v1 Announce Type: new  Abstract: Interactive LLM services must meet latency service-level objectives (SLOs) for the time to first token (TTFT) and the time between tokens (TBT), as streaming and soft real-time applications require predictable token delivery. We identify decode jitters in both aggregated and disaggregated serving. A jitter delays every request in the batch, and long responses span hundreds of steps, so even rare overruns cause many SLO violations. In aggregated serving, SLO-aware schedulers set each step's prefill budget until its predicted latency just meets the TBT target, leaving no room for mispredictions or host stalls. Disaggregation meets average TBT targets at a cost in capacity, yet the steps that admit new requests still run late.   We present \sysname, a scheduler that calibrates how much prefill work each decode step carries using the measured tail of its own iteration latency. The SLO determines how rare a late step must be, and \sysname lea
    
[^24]: 基于人工智能的动态形状控制多视角超声心动图图像合成

    AI-Based Multi-View Echocardiographic Image Synthesis using Dynamic Shape Control

    [https://arxiv.org/abs/2611.18375](https://arxiv.org/abs/2611.18375)

    该论文提出了一种基于动态全心脏网格形状控制的两阶段AI生成框架，能够合成具有跨心腔多视角一致性的超声心动图序列，突破了现有方法局限于左心室形状监督和特定心脏视角的限制。

    

    尽管自动化方法已经改善了超声心动图的工作流程，但进展仍然受限于对大型、标注完善数据集的需求。由于临床数据库的建立通常受到伦理和后勤方面的挑战限制，合成数据生成已成为一种有前景的替代方案。然而，现有方法通常依赖基于左心室形状的监督，或仅针对特定心脏视图定制，这限制了生成更多标准和非常规成像平面的能力，也无法保持跨心腔的多视角一致性。为解决这些局限，我们提出了一种新颖的形状控制多视角超声心动图序列合成框架。首先，从动态全心脏网格中提取多视角横截面，然后通过两阶段基于人工智能的流程引入类似超声的外观。在第一阶段，非配对的GAN-扩散模型将这些截面转换为具有超声特征的图像表示，第二阶段进一步提升合成图像的真实感和质量。

    arXiv:2611.18375v1 Announce Type: cross  Abstract: Although automated methods have improved echocardiographic workflows, progress remains constrained by the need for large, well-annotated datasets. As the development of clinical databases is often limited by ethical and logistical challenges, synthetic data generation has emerged as a promising alternative. However, existing approaches commonly rely on left ventricle shape-based supervision or are tailored to specific cardiac views, limiting the generation of additional standard and unconventional imaging planes and the preservation of multi-view consistency across cardiac chambers. To address these limitations, we propose a novel framework for shape-controlled synthesis of multi-view echocardiographic sequences. First, multi-view cross-sections are extracted from dynamic whole-heart meshes. Ultrasound-like appearance is then introduced through a two-stage AI-based pipeline. In Stage 1, an unpaired GAN-diffusion model translates these 
    
[^25]: 视觉-语言-动作模型：设计、训练与部署的全生命周期综述

    Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment

    [https://arxiv.org/abs/2611.18372](https://arxiv.org/abs/2611.18372)

    这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。

    

    视觉-语言-动作（VLA）模型将网络预训练的视觉-语言模型适配为输出机器人动作，从而使网络知识得以迁移到物理控制中。在不到4年的时间里，该领域从首个VLA模型RT-2，发展到可在单张消费级GPU上微调的开源模型，再到能够清扫陌生住宅的专有通用模型。本综述回顾了2022年12月至2026年10月间的88项研究工作，主要聚焦于机器人操作与导航，并涵盖人形机器人控制、世界-动作模型与评估。我们沿着七个维度追踪VLA从模型设计到训练与部署的全生命周期：模型设计涵盖，输入与记忆，即模型观察什么；架构，即其流水线如何构建；动作输出，即它如何产生运动指令；中间预测，即它还预测什么，从推理步骤到未来观测。训练涵盖——预训练数据，即机器人、网络、人类与仿真数据的混合（注：原文摘要在此处截断）。

    arXiv:2611.18372v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated 
    
[^26]: 展望智能体机器人：机器人系统的新时代

    Envisioning Agentic Robotics: A New Era of Robotic Systems

    [https://arxiv.org/abs/2611.18359](https://arxiv.org/abs/2611.18359)

    本文提出“智能体机器人”这一新范式，主张由基础模型与具身框架构成的“具身系统2”是智能体机器人的决策核心，并呼吁学界建立统一定义、评估框架与研究路线图。

    

    GPT-6 Astra 的发布标志着机器人学新范式——智能体机器人的诞生。短短几周内，配备智能体框架的前沿模型在许多具身基准测试中达到甚至超越了任务专用策略的表现。与为每一类任务训练专用策略不同，智能体机器人系统将基础模型应用于开放世界的具身任务。它以闭环方式运行：观察环境、理解用户意图、对任务进行推理、调用技能与工具、验证结果并调整策略。尽管取得了这些进展，学界尚未在统一定义、可靠的评估框架或研究路线图上达成共识。我们认为，由基础模型与具身框架共同实现的“具身系统 2”构成了智能体机器人的决策核心，其核心挑战在于将通用推理转化为在……（原文摘要在此处截断）

    arXiv:2611.18359v1 Announce Type: cross  Abstract: The release of GPT-6 Astra marks the emergence of a new paradigm in robotics: agentic robotics. Within weeks, frontier models with agentic harnesses matched or exceeded task-specific policies on many embodied benchmarks. Instead of training a dedicated policy for each family of tasks, an agentic robotic system applies a foundation model to open-world embodied tasks. It operates in a closed loop: observing the environment, interpreting user intent, reasoning about the task, invoking skills and tools, verifying outcomes, and adapting its strategy. Despite these advances, the community has yet to converge on a shared definition, a reliable evaluation framework, or a research roadmap. We argue that Embodied System 2, realized by a foundation model together with an embodied harness, forms the decision-making core of agentic robotics. Its central challenge is to translate general-purpose reasoning into effective and verifiable actions under 
    
[^27]: VLAN：视觉-语言无障碍导航

    VLAN: Vision-Language Accessible Navigation

    [https://arxiv.org/abs/2611.18346](https://arxiv.org/abs/2611.18346)

    提出首个面向多种身体形态（人形/四足机器人及视觉或行动障碍人士）的大规模视觉-语言无障碍导航基准VLAN，通过无障碍对齐指标评估导航的物理安全性与用户特定约束的契合度。

    

    无障碍通行取决于用户而非场景本身，因为一种身体形态可以通行的路径，对另一种身体形态来说可能是无法通行的。然而，现有的基准测试仅在固定且有限的身体形态下评估可通行性。我们提出了视觉-语言无障碍导航，包含1,000个场景、2万条导航轨迹和16万关键帧，涵盖多样的光照、环境条件、场景语义和路径几何。对于每条轨迹，VLAN为腿式和轮式智能体（包括人形机器人、四足机器人以及有视觉或行动障碍的人士）提供特定于身体形态的动作和无障碍特征，同时将日常障碍物和无障碍标签归类为可通过、绕行、高风险和阻塞。与仅以目标为导向的评估不同，我们的基准通过到达目标、碰撞感知和无障碍对齐等指标来衡量物理安全性以及与用户特定约束的一致性。

    arXiv:2611.18346v1 Announce Type: new  Abstract: Accessible mobility is user-conditioned rather than scene-intrinsic, as walkable paths for one embodiment are inaccessible to another. Yet, existing benchmarks evaluate traversability under a fixed and limited embodiment. We present Vision-Language Accessible Navigation (VLAN), containing 1,000 scenes, 20K navigation trajectories, and 160K keyframes across diverse lighting, environmental conditions, scene semantics, and path geometries. For each trajectory, VLAN provides embodiment-specific actions and accessibility features for legged and wheeled agents, including humanoid and quadruped robots and people with visual or mobility impairments, while everyday obstacles and accessibility labels are categorized as Passable, Bypass, HighRisk, and Blocked. Unlike goal-only evaluation, our benchmark measures physical safety and alignment with user-specific constraints via goal-reaching, collision-aware, and accessibility-aligned metrics. We benc
    
[^28]: 基于因果AI驱动的安全毫米波波束对准与防御性变分自编码器

    Causal AI-Driven Secure mmWave Beam Alignment with Defensive Variational Autoencoders

    [https://arxiv.org/abs/2611.18321](https://arxiv.org/abs/2611.18321)

    该论文提出了一种将因果发现与防御性变分自编码器相结合的因果感知深度学习框架，在提升毫米波波束对准的可解释性与泛化能力的同时，有效抵御对抗攻击。

    

    高效且安全的波束对准对毫米波通信系统至关重要。然而，现有的基于深度学习（DL）的波束对准方法往往忽视输入与输出之间的因果关系，导致可解释性有限、泛化能力差以及过多的波束扫描。此外，这些深度学习方案容易受到对抗攻击的影响，即接收功率测量中微小的扰动就可能误导波束分类器。为了克服这些问题，我们提出了一种将因果发现融入波束对准的因果感知深度学习框架。该框架中的两阶段因果波束选择算法首先学习一个贝叶斯图，用以捕捉接收信号强度指示（RSSI）作为输入与最优波束索引作为目标变量之间的依赖关系，然后利用该图为深度学习分类器选择一个最小化的相关输入特征集合。为了缓解对抗攻击，我们引入了一种防御性变分自编码器（DVAE）。

    arXiv:2611.18321v1 Announce Type: new  Abstract: Efficient and secure beam alignment is essential for mmWave communication systems. However, existing deep learning (DL)-based beam alignment methods often neglect causal relationships between inputs and outputs, resulting in limited interpretability, poor generalization, and excessive beam sweeping. Moreover, these DL solutions are vulnerable to adversarial attacks, where small perturbations in received power measurements can mislead beam classifiers. To overcome these issues, we propose a causally-aware DL framework integrating causal discovery into beam alignment. A two-stage causal beam selection algorithm learns a Bayesian graph capturing dependencies between received signal strength indicators (RSSI) as inputs and the optimal beam index as the target variable, then uses it to select a minimal set of relevant input features for the DL classifier. To mitigate adversarial attacks, we introduce a defensive variational autoencoder (DVAE)
    
[^29]: 跨模式专家：面向跨数据库泛化的智能体引导的关系专家混合模型

    Experts Across Schemas: Agent-Guided Mixture-of-Relational-Experts for Cross-Database Generalization

    [https://arxiv.org/abs/2611.18315](https://arxiv.org/abs/2611.18315)

    该论文提出MoREGT关系专家混合图Transformer，通过基于列元数据的专家路由和标签无感知的LLM智能体自动生成派生特征，证明更好的架构设计与目标数据库特征相结合能够以远少于现有方法的预训练数据实现跨数据库泛化性能。

    

    网络平台将用户和物品的活动记录在关系数据库中，而各平台的数据库模式彼此不同，这使得预测模型难以在平台之间迁移。关系型基础模型通过跨数据库预训练来解决这一问题，但近来的性能提升依赖于日益庞大的预训练语料库。我们探究：更好的架构设计，结合目标数据库特有的特征，能否用远少于以往的预训练数据实现相当的性能提升。我们提出MoREGT，一种关系专家混合图Transformer，它将行和单元格均表示为token，在注意力机制中引入行的对数时间戳和外键关系名称，并根据列元数据而非取值将单元格和行路由至相应专家。一个对标签无感知的LLM智能体对该模型形成补充：它读取目标数据库的模式和列统计信息，并添加经过验证的单行派生列，且无法访问标签数据。

    arXiv:2611.18315v1 Announce Type: new  Abstract: Web platforms record user and item activity in relational databases whose schemas differ from platform to platform, which makes it difficult to transfer predictive models between them. Relational foundation models address this through cross-database pretraining, yet recent improvements have relied on increasingly large pretraining corpora. We ask whether a better architecture, together with features specific to the target database, can deliver comparable gains from far less pretraining data. We introduce MoREGT, a mixture-of-relational-experts graph transformer that represents both rows and cells as tokens, adds the log-ages of rows and the names of foreign-key relationships to attention, and routes cells and rows to experts by column metadata rather than by values. A label-blind LLM agent complements the model: it reads the target database's schema and column statistics and adds validated single-row derived columns, without access to la
    
[^30]: 基于样本的量子对角化中参数选择的人工智能方法

    AI for Parameter Selection in Sample-based Quantum Diagonalization

    [https://arxiv.org/abs/2611.18307](https://arxiv.org/abs/2611.18307)

    该研究证明机器学习模型可以根据哈密顿量直接预测量子运行时间和近似误差，从而自动为SKQD算法选择最优参数配置，解决了基于样本的量子对角化方法中缺乏正式参数选择方法的问题。

    

    基于样本的量子对角化（SQD）方法提供了一种计算分子基态能量的途径，但其精度和运行时间高度依赖于若干配置选择，例如子空间维度、传播时间步长以及固定的采样预算在多个电路之间的分配，而这些配置目前尚无正式的选择方法。在本工作中，我们探索了预言机模型能否根据哈密顿量和候选配置来预测量子运行时间和近似误差。通过使用具有可控谱性质的合成哈密顿量，我们证明机器学习模型可以直接将哈密顿量映射到SKQD算法预定义网格中的最优配置，并以经典Krylov传播进行验证，在真实值与预测值之间实现了误差和运行时间的强秩相关性，Spearman ρ 分别为0.969和0.905。将该方法扩展到50个分子哈密顿量……（摘要原文在此处被截断）

    arXiv:2611.18307v1 Announce Type: cross  Abstract: Sample-based Quantum Diagonalization (SQD) methods provide a way to compute molecular ground-state energies, but their accuracy and runtime depend sensitively on configuration choices such as the subspace dimension, propagation timestep, and the allocation of a fixed sampling budget across circuits, for which no formal selection method exists. In this work, we explore whether oracle models can predict the quantum runtime and approximation error from the Hamiltonian and the candidate configuration. Using synthetic Hamiltonians with controllable spectral properties, we show that machine-learned models can map a Hamiltonian directly to the best-ranked configuration within a predefined grid for the SKQD algorithm, validated against classical Krylov propagation, achieving strong rank correlations between true and predicted error and runtime, Spearman $\rho = 0.969$ and $\rho = 0.905$, respectively. Extending the approach to 50 molecular Ham
    
[^31]: 当评估在临床AI中变成干预

    When evaluation becomes intervention in clinical AI

    [https://arxiv.org/abs/2611.18297](https://arxiv.org/abs/2611.18297)

    临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。

    

    AI评判者（AI judges）是一种依据评分标准对另一个系统的输出进行打分的模型，它将临床审查的规模扩展到了临床AI书写内容的规模。当团队使用评判者的评估结果来选择提示词或训练模型时，评判者本身就成为了干预的一部分。这种选择过程可能会偏向评判者所忽视的错误。我们描述了评判者的报告必须包含哪些内容，以及每一项由评判者驱动的变更所需的独立临床测试。

    arXiv:2611.18297v1 Announce Type: new  Abstract: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs.
    
[^32]: PrivacyPR：一个用于隐私感知代码审查基准测试的真实世界隐私修复数据集

    PrivacyPR: A Dataset of Real-World Privacy Fixes for Benchmarking Privacy-Aware Code Review

    [https://arxiv.org/abs/2611.18293](https://arxiv.org/abs/2611.18293)

    该论文提出了PrivacyPR数据集，包含4,131个带LLM生成的隐私相关性和GDPR原则标注的真实GitHub拉取请求，为评估编码智能体能否在代码审查中发现法律要求的隐私缺陷提供了首个大规模基准。

    

    编码智能体越来越多地参与代码审查，但它们能否发现GDPR等法律要求软件避免的隐私缺陷，这在很大程度上仍未得到检验，部分原因在于很少有数据集将真实的隐私缺陷与其被接受的修复方案及审查上下文联系起来。我们提出了PrivacyPR，这是一个包含来自1,469个仓库的4,131个已合并GitHub拉取请求（PR）的数据集，创建时间为2018年6月至2024年1月。每个PR都包含其提交记录、代码差异、审查讨论，以及由大语言模型生成的隐私相关性、修复类型和GDPR原则标签。PrivacyPR提供了1,169个高置信度隐私修复和1,643个非修复PR，还包括Verified（一个经人工核验的子集，包含250个隐私修复和250个非修复PR）以及Small（一个100+100的评估子集）。在Verified子集上，两名标注者均认可了LLM所做GDPR原则标注中的72%。除了支持对隐私缺陷如何产生和修复的实证研究之外，PrivacyPR还提供了一个基准测试平台。

    arXiv:2611.18293v1 Announce Type: cross  Abstract: Coding agents increasingly review code, yet whether they catch the privacy defects that laws such as the GDPR require software to avoid remains largely untested, partly because few datasets connect real privacy defects to their accepted fixes and review context. We present PrivacyPR, a dataset of 4,131 merged GitHub pull requests (PRs) from 1,469 repositories, created between June 2018 and January 2024. Each PR includes its commits, diffs, review discussion, and LLM-generated labels for privacy relevance, fix type, and GDPR principles. PrivacyPR provides 1,169 high-confidence privacy fixes and 1,643 non-fix PRs, together with Verified, a manually checked split of 250 privacy fixes and 250 non-fix PRs, and Small, a 100 + 100 subset for evaluation. On Verified, both annotators endorsed 72% of the LLM's GDPR-principle assignments. Beyond supporting empirical study of how privacy defects arise and are fixed, PrivacyPR provides a benchmark 
    
[^33]: 正确的词元，正确的时机：从逐词元门控到轨迹感知的激活引导

    Right Token, Right Time: From Per-Token Gating to Trajectory-Aware Activation Steering

    [https://arxiv.org/abs/2611.18274](https://arxiv.org/abs/2611.18274)

    该论文将基于稀疏自编码器的激活引导重构为固定预算下的逐词元分配问题，并提出无记忆的对比特征投影（CFP）与轨迹感知的闭环自适应伯努利引导（CLABS）两种门控机制，通过只在恰当的词元和时机施加干预来避免分布偏移、保持生成质量。

    

    通过稀疏自编码器（SAE）进行的激活引导能够在无需任务特定微调的情况下对大型语言模型实现行为控制。然而，标准方法会在生成的每一个词元上都施加引导信号，这会使模型表示偏离其训练时所处的分布，从而损害生成质量。我们证明了在生成过程中“何时”施加干预是决定控制效果的一个关键维度。我们将引导问题形式化为在固定可引导词元预算下的逐词元分配问题，使不同方法之间的差异仅体现在其选择的词元上，从而让分配方式成为唯一变量。在该框架下，我们引入了两种门控机制：第一种是对比特征投影（Contrastive Feature Projection, CFP），它是无记忆的，通过稀疏且可解释的特征投影来选择与目标行为最不对齐的词元；第二种是闭环自适应伯努利引导，它是轨迹感知的，将分配视为一个闭环过程。

    arXiv:2611.18274v1 Announce Type: cross  Abstract: Activation steering via sparse autoencoders (SAEs) enables behavioral control of large language models without task-specific fine-tuning. Yet, standard methods apply the steering signal at every generated token, displacing representations from the distribution the model was trained on and eroding generation quality. We show that \emph{when} along a generation the intervention is applied is a decisive axis of control. We cast steering as a per-token allocation problem under a fixed budget of steerable tokens, so that methods differ only in which tokens they select, making allocation the sole variable. Within this framework we introduce two gates. The first, Contrastive Feature Projection (CFP), is memoryless and selects the tokens least aligned with the target behavior through a sparse, interpretable feature projection. The second, Closed-Loop Adaptive Bernoulli Steering (CLABS), is trajectory aware. It treats allocation as closed-loop 
    
[^34]: 从字节到行为：基于运行中GPU隐藏状态的请求级归因

    From Bytes to Behavior: Request-Level Attribution from Live GPU Hidden States

    [https://arxiv.org/abs/2611.18272](https://arxiv.org/abs/2611.18272)

    该论文提出了一种无需修改进程、无需预知模型架构或权重的取证方法，可直接从运行中的GPU进程提取逐层残差流激活值，实现对无法重启或修改的大语言模型服务的请求级行为归因分析。

    

    现有的针对机器学习系统的内存取证通常面向静态产物，例如模型权重或提示词日志，而非运行中推理过程的瞬态计算状态。这一空白限制了取证分析人员对无法重启或修改的、正在运行的大语言模型（LLM）服务进行具体行为分析的能力。我们提出了一种方法，利用非修改式调试器从运行中的GPU进程获取逐层残差流激活值，且无需预先了解模型的架构参数或权重。该方法采用一个通用的六阶段流程并配合平台特定的适配器，通过PyTorch分配器时钟或内核启动时钟来识别并读取显存中的张量分配。我们在涵盖九种架构类别的33个开源仅解码器（decoder-only）模型上验证了该采集方法，证明恢复出的字节达到字节级精确且可用。

    arXiv:2611.18272v1 Announce Type: cross  Abstract: Existing memory forensics for machine learning systems typically targets static artifacts, such as model weights or prompt logs, rather than the transient computational states of a live inference process. This gap limits an examiner's ability to analyze the specific behavior of a running large language model (LLM) service that cannot be restarted or modified. We present a method for acquiring per-layer residual-stream activations from a live GPU process using a non-modifying debugger, without requiring prior knowledge of the model's architecture parameters or weights. The approach employs a general six-stage process with platform-specific adapters, utilizing either a PyTorch allocator clock or a kernel-launch clock to identify and read tensor allocations from video memory. We validate this acquisition on 33 open-source decoder-only models across nine architecture classes, demonstrating that the recovered bytes are byte-exact and usable
    
[^35]: ParetoM³：通过极小-极大-极小优化在偏好引导下学习帕累托集

    ParetoM$^3$: Learning on the Pareto Set under Preference Guidance via Min-Max-Min Optimization

    [https://arxiv.org/abs/2611.18252](https://arxiv.org/abs/2611.18252)

    提出了一种基于惩罚的极小-极大-极小优化公式（M³问题）及单循环一阶算法 ParetoM³，将偏好优化与弱帕累托最优性统一在单一目标中，并证明了算法能收敛到近似弱帕累托最优的ε-稳定点。

    

    在多个本质冲突的目标上进行学习，需要从帕累托集中选择理想的权衡方案。在这项工作中，我们研究了偏好引导的多目标优化问题，即在多个非凸目标的弱帕累托集上最小化一个预定义的偏好函数。利用一个在弱帕累托集上取值为零的優值函数（merit function），我们提出了一种基于惩罚的极小-极大-极小（min-max-min）公式，称为 M³ 问题，该公式将偏好优化与弱帕累托最优性的强制约束统一在单一目标中。我们进一步开发了 ParetoM³，这是一种单循环一阶算法，用于计算所得 M³ 问题的稳定点。在局部 Kurdyka-Łojasiewicz 假设下，我们证明了当惩罚参数足够大时，ParetoM³ 能够找到 M³ 问题的 ε-稳定点，并且该点对于原始多目标问题也是近似弱帕累托最优的。

    arXiv:2611.18252v1 Announce Type: cross  Abstract: Learning across multiple inherently conflicting objectives requires selecting desirable trade-offs from the Pareto set. In this work, we study preference-guided multi-objective optimization, where a predefined preference function is minimized over the weak Pareto set of multiple nonconvex objectives. Leveraging a merit function that vanishes on the weak Pareto set, we propose a penalty-based min-max-min formulation, termed the M$^3$ problem, which unifies preference optimization and the enforcement of weak Pareto optimality in a single objective. We further develop ParetoM$^3$, a single-loop first-order algorithm for computing stationary solutions of the resulting M$^3$ problem. Under a local Kurdyka-Lojasiewicz assumption, we prove that ParetoM$^3$ finds an $\epsilon$-stationary point of the M$^3$ problem that is also approximately weakly Pareto optimal for the original multi-objective problem when the penalty parameter is sufficientl
    
[^36]: LLM生成过程中词汇网络上起舞的标记

    Dancing Tokens on the Vocabulary Network from LLM Generation

    [https://arxiv.org/abs/2611.18248](https://arxiv.org/abs/2611.18248)

    该研究通过构建LLM词表嵌入的相似性网络，揭示知识空间的碎片化拓扑在预训练阶段即已确定（蒸馏几乎不改变它），且文本生成并非沿该网络的游走，而是集中访问网络内部、并在网络深处才最终确定的动态过程。

    

    大语言模型（LLM）逐个标记（token）地生成文本，但特定的生成过程与预训练中学习到的几何结构之间的关系仍不清楚，LLM的根本可解释性依然难以捉摸。在此，我们基于六个开源LLM的输入嵌入构建了一个相似性网络，并将生成过程映射到该网络上。该标记嵌入网络是无标度且呈重尾分布的，这与协方差匹配的随机嵌入类似，但其碎片化程度要高得多。蒸馏几乎不改变这种拓扑结构，因为输入嵌入在很大程度上是被继承的，因此知识空间在预训练阶段即已确定。连续生成的标记很少是图上的邻居，且在大多数模型中，其相邻出现频率并不比乱序排列时更高，因此生成过程并非在预定义网络上的游走，尽管它所访问的标记都集中于该网络内部。通过逐层的logit lens分析发现，在大多数模型中，标记在网络较深处才趋于稳定，而具有重复生成倾向的模型……（摘要原文在此处截断）

    arXiv:2611.18248v1 Announce Type: new  Abstract: Large language models (LLMs) generate text one token at a time, yet how a particular generation relates to the geometry learned in pre-training remains unclear. The fundamental explainability of LLMs is elusive. Here, we build a similarity network from the input embeddings of six open LLMs and map generation onto it. The token-embedding network is scale-free while heavy-tailed, as are random embeddings with matched covariance, but far more fragmented. Distillation barely changes this topology because input embeddings are largely inherited. The knowledge space is thus deterministic from pre-training. Consecutive generated tokens are rarely graph neighbours and, in most models, no more often than in shuffled order, so generation is not a walk on the predefined network, although the tokens it visits are concentrated within it. Through the logit lens in a layer- wise manner, tokens settle late in depth in most models, and repetitive models r
    
[^37]: 非确定性因果模型中的因果概率

    Probabilities of Causation in Nondeterministic Causal Models

    [https://arxiv.org/abs/2611.18245](https://arxiv.org/abs/2611.18245)

    该论文提出了一个面向非确定性因果模型的通用识别框架，通过矩约束与约束优化推导出因果概率的尖锐解析界，所得识别区域通常比确定性模型下的经典结果更具信息量。

    

    我们研究非确定性因果模型中的因果概率，在这类模型中，即使所有潜在变量都被固定，机制本身仍可能保持内在的随机性。聚焦于二元处理和结果，我们开发了一个通用的识别程序，将观察性数据和干预性数据表示为对潜在随机机制类型的矩约束，并通过约束优化计算反事实量的最小值和最大值。在该框架内，我们在若干专为非确定性模型量身定制的假设下，推导出因果概率的尖锐解析界。所得的识别区域往往比确定性替代方法所能获得的识别区域具有更丰富的信息。我们通过一个关于个性化治疗决策的案例研究以及在半合成数据集上的实验，展示了这些结果的实际意义。我们的结果阐明了经典界与非确定性因果模型之间的关系。

    arXiv:2611.18245v1 Announce Type: new  Abstract: We study probabilities of causation in nondeterministic causal models, where mechanisms may remain intrinsically stochastic even after all latent variables are fixed. Focusing on binary treatments and outcomes, we develop a generic identification program that represents observational and interventional data as moment constraints on latent stochastic mechanism types and computes minimal/maximal values of counterfactual quantities via constrained optimization. Within this framework, we derive sharp analytic bounds on the probabilities of causation under several assumptions tailored to nondeterministic models. Resulting identification regions are often more informative than those attainable under deterministic alternatives. We illustrate the practical relevance of these results in a case study on personalized treatment decisions and experiments on semi-synthetic datasets. Our results clarify how classical bounds relate to nondeterministic c
    
[^38]: RACE：面向高效序列推荐的排序不确定性感知容量自适应定制

    RACE: Ranking-Uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation

    [https://arxiv.org/abs/2611.18244](https://arxiv.org/abs/2611.18244)

    提出RACE框架，通过捕获模型预测的排序不确定性并将其转化为主动的资源分配器，以不确定性感知的压缩比率动态调整模型结构和输入序列，从而解决序列推荐中统一压缩比率导致的容量-复杂度失配问题，实现资源受限环境下的高效推荐。

    

    高效序列推荐已成为提供实时个性化服务的一种有前景的范式。为了在资源受限的环境下部署模型，模型压缩成为必要手段，然而现有方法通常采用统一的压缩比率，忽视了用户意图复杂性的异质性。这种“一刀切”的方式导致了根本性的“容量-复杂度失配”问题，即计算资源要么浪费在简单实例上，要么对复杂实例而言不足。为解决这一问题，我们提出了RACE（排序不确定性感知的容量自适应定制框架），该框架利用不确定性感知的压缩比率，动态调整模型结构和输入序列。为实现容量自适应推理，RACE捕获模型预测的内在不确定性，并将其转化为一种主动的资源分配器……

    arXiv:2611.18244v1 Announce Type: cross  Abstract: Efficient sequential recommendation has emerged as a promising paradigm for providing real-time personalized service. To deploy models under resource-constrained environment, model compression becomes a necessity, yet existing methods typically apply uniform compression ratios, overlooking the heterogeneity in user intent complexity. This "one-size-fits-all" approach leads to a fundamental Capacity-Complexity Mismatch, where computational resources are either wasted on simple instances or insufficient for complex ones. To address this, we propose RACE (Ranking-uncertainty Aware Capacity-Adaptive Customization for Efficient Sequential Recommendation), a framework that dynamically adapts both model structures and input sequences with uncertainty-aware compression ratios. To ensure capacity-adaptive inference, RACE captures the model's intrinsic uncertainty of its predictions and transforms it into a proactive resource allocator. Specific
    
[^39]: 指导大语言模型控制的助理机器人的基本与道德价值观个性化

    Personalization of Basic and Moral Values Guiding LLM-Controlled Assistant Robots

    [https://arxiv.org/abs/2611.18230](https://arxiv.org/abs/2611.18230)

    该研究通过SVS、PVQ-21和MFQ三种心理测量工具发现，用户自身的价值观与他们期望助理机器人体现的价值观存在差异，因此主张对LLM控制的助理机器人进行基本与道德价值观的个性化设定，并基准测试了十五个当代大语言模型。

    

    基于大语言模型（LLM）的个人助理机器人很可能在未来十年内进入日常生活，但应当指导其行为的价值观在很大程度上仍未被明确界定。当前的模型对齐实践将单一模型导向人类价值观的总体聚合，却忽略了个体用户可能更希望机器人的价值观特征与自己的有所不同。我们向潜在的未来用户（N=22）发放了三种成熟心理测量工具的平行自评版与机器人定向版——简化版施瓦茨价值观调查（SVS）、肖像价值观问卷（PVQ-21）和道德基础问卷（MFQ），同时捕捉了他们的个人价值观以及他们希望助理机器人所体现的价值观。我们使用UEQ-S比较了这三种工具作为个性化工具的效果，并将十五个当代大语言模型与由此得到的价值观特征进行了基准测试。参与者希望他们的机器人得分明显……（原文摘要在此处截断）

    arXiv:2611.18230v1 Announce Type: cross  Abstract: Personal assistant robots powered by large language models (LLMs) are likely to enter everyday life within the next decade, but the values that should guide their behaviour remain largely unspecified. Current alignment practice steers a single model toward an aggregate of human values, ignoring that individual users may prefer a robot whose value profile differs from their own. We administer parallel self- and robot-targeted versions of three established psycho-metric instruments - the simplified Schwartz Values Survey (SVS), the Portrait Values Questionnaire (PVQ-21), and the Moral Foundations Questionnaire (MFQ) - to potential future users (N=22), capturing both their personal values and the values they would want an assistant robot to embody. We compare the three instruments as personalization tools using the UEQ-S and benchmark fifteen contemporary LLMs against the resulting profiles. Participants wanted their robot to score marked
    
[^40]: 从纸笔证明到机器验证证明：面向系统辨识的Lean

    From Pen-and-Paper to Machine-Checked Proofs: Lean for System Identification

    [https://arxiv.org/abs/2611.18220](https://arxiv.org/abs/2611.18220)

    本文向系统辨识与控制领域的研究者介绍Lean 4证明助手，通过可辨识性、集合不变性及预测误差辨识中经典引理的形式化示例，展示机器验证证明如何弥补传统纸笔证明难以逐步核验的不足。

    

    科学论文中的数学证明往往冗长复杂，并依赖于对先前结果的层层引用，这使得读者和审稿人难以逐步验证每一个细节。交互式定理证明提供了一种补充手段：定理及其证明以代码形式书写，每一步推理都由一个被称为“内核”的小型可信软件进行检验。本文面向系统辨识与控制领域的研究者，介绍Lean 4证明助手，且不要求读者具备形式化方法的先验知识。我们首先通过两个基础示例展示如何阅读Lean证明：第一个关于可辨识性，第二个关于集合不变性。随后，我们以预测误差辨识中噪声滤波器求逆的一个经典引理（Ljung, 1999, 引理3.1）为例，说明如何利用Lean分析文献中的已有结果。由于在Lean中每一步推理都必须有据可查，形式化的过程……

    arXiv:2611.18220v1 Announce Type: cross  Abstract: Mathematical proofs in scientific papers are often long and intricate, relying on chains of references to earlier results. This makes it difficult for readers and reviewers to verify each and every step. Interactive theorem proving offers a complement: the theorem and its proof are written as code, and every inference is checked by a small, trusted piece of software called the kernel. This paper serves as an introduction to the Lean 4 proof assistant for a system identification and control audience, assuming no prior exposure to formal methods. We first show how a Lean proof is read using two elementary examples: the first on identifiability, the second on set invariance. We then illustrate how Lean can be used to analyse results from the literature, taking as example a classic lemma on the inversion of the noise filter in prediction-error identification (Ljung, 1999, Lemma 3.1). Since in Lean every step must be justified, the formaliz
    
[^41]: T2ST：通过时间数据与知识迁移实现数据高效的时空基础模型

    T2ST: Data-Efficient Spatio-Temporal Foundation Models via Temporal Data and Knowledge Transfer

    [https://arxiv.org/abs/2611.18199](https://arxiv.org/abs/2611.18199)

    T2ST 框架通过引入丰富的时间序列数据辅助训练，并从冻结的时间序列基础模型中迁移知识，在时空观测稀缺的情况下实现数据高效的时空基础模型训练。

    

    arXiv:2611.18199v1 公告类型：cross 摘要：时空基础模型（STFMs）需要大规模训练语料，然而同步观测和空间元数据的获取与整理成本高昂，限制了合适训练数据的可获得性。相比之下，时间序列数据丰富且更易于收集。由于每个位置的观测本身构成了一个时间序列，这引出了一个自然的问题：当时空观测稀缺时，丰富的时间序列数据能否增强时空基础模型？为弥合这一差距，我们提出了 T2ST，一个数据高效的训练框架，它利用精选的辅助时间序列来增强有限的时空观测，并从冻结的时间序列基础模型中迁移知识。我们的关键洞察是：时间模式学习可以从丰富的外部时间序列中受益，而空间依赖学习则仍然立足于真实的时空观测。基于这一洞察，T2ST 检索匹配的序列以强化稀疏的……（原文摘要至此截断）

    arXiv:2611.18199v1 Announce Type: cross  Abstract: Spatio-temporal foundation models (STFMs) require large-scale training corpora, yet synchronized observations and spatial metadata are costly to acquire and curate, limiting the availability of suitable data. By contrast, time-series data are abundant and easier to collect. Since observations at each location form a time series, this motivates a natural question: can abundant time-series data enhance STFMs when spatio-temporal observations are scarce? To bridge this gap, we propose T2ST, a data-efficient training framework that augments limited spatio-temporal observations with selected auxiliary time series and transfers knowledge from a frozen time-series foundation model. Our key insight is that temporal pattern learning can benefit from abundant external time series, while spatial dependency learning remains grounded in real spatio-temporal observations. Guided by this insight, T2ST retrieves matched sequences to reinforce sparsely
    
[^42]: 谁应该进入专家组？面向LLM交易智能体的任务条件化角色选择

    Who Should Be on the Panel? Task-Conditioned Persona Selection for LLM Trading Agents

    [https://arxiv.org/abs/2611.18150](https://arxiv.org/abs/2611.18150)

    本文提出FinPanel系统，其核心创新在于根据每只股票的交易特征动态选择合适的LLM交易员角色组合，取代传统多智能体系统中固定角色阵容的做法，使智能体团队与具体交易任务相匹配。

    

    多智能体LLM系统通常通过固定一组角色阵容、再设计这些角色如何协作来构建。对于给定任务，哪些智能体应组成团队则交由设计者决定，且同一套角色阵容被重复用于所有任务。金融交易使这一选择的代价清晰可见：不同股票的交易方式各异，适合某一只股票的交易风格未必适合另一只。我们研究任务条件化角色选择，即根据团队将要面对的任务特征来组建智能体阵容，并通过FinPanel将其具体实现——这是一个LLM多智能体交易系统，其中“招聘者”根据每只股票交易行为的方向盲统计概要为其选出一组具名交易员角色，每个角色以自己的风格独立给出每日交易判断，而“管理者”在阅读专家组的论据以及部门保存的分层历史业绩记录后执行交易。由于没有任何信息反馈给交易员，专家组保持稳定……

    arXiv:2611.18150v1 Announce Type: cross  Abstract: Multi-agent LLM systems are built by fixing a cast of roles and then designing how the cast coordinates. Which agents should make up the group for a given task is left to the designer, and the same cast is reused for every task. Financial trading makes the cost of that choice visible: stocks trade differently, and the trading styles suited to one need not suit another. We study task-conditioned persona selection, composing the group from the characteristics of the task it will face, and instantiate it with FinPanel, an LLM multi-agent trading system in which a recruiter selects a panel of named trader personas for each stock from a direction-blind statistical sketch of how the stock trades, each persona gives an independent daily call in its own style, and a manager commits the trade after reading the panel's arguments and their desk-kept, hierarchical track records. Because nothing is returned to the traders, the panel stays the panel
    
[^43]: 大语言模型赋能的智能体用于锂电池自适应快充控制

    Large Language Model-Empowered Intelligent Agent for Adaptive Fast-Charging Control of Lithium-Ion Batteries

    [https://arxiv.org/abs/2611.18140](https://arxiv.org/abs/2611.18140)

    本文提出了一种由大语言模型赋能的智能体闭环框架，可自适应地控制锂电池快充，在缩短充电时间的同时兼顾抑制电池老化与安全性，克服了传统协议、模型优化和常规数据驱动控制方法的局限。

    

    锂电池快充是一个多目标控制问题，需要在缩短充电时间、维持高能量吞吐量的同时，抑制加速老化和涉及安全的关键电化学反应。传统充电协议易于实现，但在适应电池化学体系、运行工况和老化状态变化方面的能力有限。基于模型的优化和传统数据驱动控制通常需要精确的模型、高昂的迭代搜索成本或针对特定任务的策略。我们开发了一种由大语言模型（LLM）赋能的智能体，用于电池自适应快充控制。在这一闭环框架中，智能体解析结构化的电池状态、生成可执行的充电动作、观察由此产生的电池响应，并持续调整后续决策。我们利用PyBaMM构建了一个涵盖五种电池化学体系的大规模仿真语料库……

    arXiv:2611.18140v1 Announce Type: new  Abstract: Fast charging of lithium-ion batteries is a multi-objective control problem that requires reducing charging time and sustaining high energy throughput while suppressing accelerated degradation and safety-critical electrochemical reactions. Conventional protocols are easy to implement but have limited ability to adapt to variations in battery chemistry, operating conditions, and aging state. Model-based optimization and conventional data-driven control often require accurate models, costly iterative search, or task-specific policies. We develop a large language model (LLM)-empowered intelligent agent for adaptive battery fast-charging control. In this closed-loop framework, the agent interprets structured battery states, generates executable charging actions, observes the resulting battery response, and continuously adjusts subsequent decisions. We construct a large-scale simulation corpus with PyBaMM across five battery chemistries and 1
    
[^44]: PUMA-Former：不完备信息下中低轨（MEO-LEO）卫星网络中不确定性感知的智能体式计算卸载

    PUMA-Former: Uncertainty-Aware Agentic Computation Offloading in MEO-LEO Satellite Networks With Incomplete Information

    [https://arxiv.org/abs/2611.18139](https://arxiv.org/abs/2611.18139)

    该论文提出PUMA-Former框架，在不完备信息条件下通过不确定性感知的智能体机制协调MEO-LEO卫星网络中的协作计算卸载，利用潜在延迟注意力与队列误差感知融合来平衡已完成工作量、服务延迟和执行成本。

    

    卫星边缘计算将移动工作负载分布于不断移动的处理器之间，其服务能力需要随时间进行协调。在中地球轨道（MEO）-低地球轨道（LEO）网络中，周期性状态报告会导致CPU状态过时或缺失，使卸载决策对不确定的服务容量和共享目的地的拥塞十分敏感。我们提出PUMA-Former，这是一个在不完备信息下进行协作卸载的不确定性感知智能体框架，可平衡已完成工作量、服务延迟与执行成本。MEO协调器维护差异化的报告记忆，并在可用证据、容量估计、分配指令与执行反馈之间形成闭环。潜在延迟注意力机制对合理的测量时延假设进行加权，历史增量信息则围绕最新收到的报告对CPU估计进行细化。队列误差感知融合机制在考虑积压不确定性的同时纳入辅助压力信息。推断的资源…

    arXiv:2611.18139v1 Announce Type: cross  Abstract: Satellite edge computing distributes mobile workloads across moving processors whose service capacity must be coordinated over time. In medium Earth orbit (MEO)--low Earth orbit (LEO) networks, periodic reports leave CPU states stale or missing, making offloading sensitive to uncertain capacity and shared-destination congestion. We propose PUMA-Former, an uncertainty-aware agentic framework for cooperative offloading under incomplete information that balances completed work, service delay, and execution cost. A MEO coordinator maintains distinct-report memory and closes the loop between available evidence, capacity estimates, allocation commands, and execution feedback. Latent-delay attention weights plausible measurement-age hypotheses, and historical innovations refine CPU estimates around the latest received report. Queue-error-aware fusion incorporates auxiliary pressure while accounting for backlog uncertainty. The inferred resour
    
[^45]: JFinBench：基于最新资格考试的可持续更新的日本金融基准

    JFinBench: A Continuously Refreshable Japanese Financial Benchmark from the Latest Certification Exams

    [https://arxiv.org/abs/2611.18130](https://arxiv.org/abs/2611.18130)

    该论文提出了JFinBench——一个基于2026年最新版日本资格考试构建、涵盖会计/税务/理财/银行领域的可持续更新金融评测基准，并通过人在回路的智能体流水线大幅降低了纳入新试题、保持基准时效性的成本。

    

    大语言模型（LLM）在金融领域的应用日益广泛，这催生了对能够在不同法规与实务的金融体系中评估其能力的基准的需求。然而，现有的金融基准对特定国家金融体系的覆盖有限，而且由于已发布的题目会泄露到训练数据中，它们难以保持时效性。我们提出了JFinBench，一个日本金融基准，包含来自2026年版六项定期举办的资格考试的321个题目，涵盖会计、税务、理财规划和银行业务等领域。为了构建该基准，我们开发了一个人在回路的智能体流水线：编码智能体负责编写针对特定考试的转换代码和审核页面，而人类则验证每个题目并将修正反馈给智能体。该流水线能够适应异构的考试格式，并降低了纳入新考试版本的工作量，从而实现基准的定期更新。

    arXiv:2611.18130v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly applied to finance, creating a need for benchmarks that evaluate their capabilities across financial systems with different regulations and practices. However, existing financial benchmarks provide limited coverage of country-specific financial systems, and they are difficult to keep up to date as released questions leak into training data. We introduce JFinBench, a Japanese financial benchmark comprising 321 examples from the 2026 editions of six regularly administered certification exams covering accounting, tax, financial planning, and banking. To construct it, we develop a human-in-the-loop agentic pipeline in which a coding agent writes exam-specific conversion code and a review page, while a human verifies every example and feeds corrections back to the agent. This pipeline adapts to heterogeneous exam formats and reduces the effort of incorporating new exam editions, enabling periodic 
    
[^46]: 使用量子退火机优化多材料结构中的钻孔轨迹

    Optimizing Drilling Trajectories in Multi-Material Structures Using Quantum Annealers

    [https://arxiv.org/abs/2611.18121](https://arxiv.org/abs/2611.18121)

    本文提出一种基于D-Wave Leap混合求解器的量子-混合优化方法，将多材料结构的钻孔轨迹规划建模为包含材料分组、刀具磨损等工业约束的路径优化问题，以最小化总能耗，并在10至55孔的实例上验证了该方法的有效性。

    

    本文提出了一种量子-混合优化方法，用于规划多材料结构上的钻孔轨迹，目标是最小化总能耗。所提出的建模方式将钻孔过程描述为一个约束路径规划问题，其中每个孔都必须被访问，同时考虑异质材料特性、依赖于材料的钻孔成本、位移能量以及刀具磨损惩罚。该方法纳入了现实的工业约束，包括开放路径规划以及将属于同一材料的孔进行强制分组，以避免频繁更换钻头。该问题使用D-Wave的Leap约束二次模型混合求解器进行求解。为了评估性能，作者对10到55个孔、具有不同材料分布和成本矩阵的实例进行了基准测试。结果凸显了量子-混合优化在处理复杂多材料钻孔场景方面的能力。

    arXiv:2611.18121v1 Announce Type: new  Abstract: In this work we present a quantum-hybrid optimization approach for planning drilling trajectories on multi-material structures, with the objective of minimizing total energy consumption. The proposed formulation models the drilling process as a constrained routing problem in which each hole must be visited while accounting for heterogeneous material properties, material-dependent drilling costs and displacement energy, as well as tool-wear penalties. Our method incorporates realistic industrial constraints, including open-path routing and mandatory grouping of holes belonging to the same material to avoid frequent tool-bit changes. The modeled problem is solved using D-Wave's Leap Constrained Quadratic Model Hybrid Solver. To evaluate performance, we benchmark instances ranging from 10 to 55 holes with varying material distributions and cost matrices. The results highlight the capability of quantum-hybrid optimization to handle complex m
    
[^47]: 可解释权重空间编程构建透明Transformer与大脑编码模型

    Interpretable weight-space programming builds transparent transformers and brain encoding models

    [https://arxiv.org/abs/2611.18114](https://arxiv.org/abs/2611.18114)

    该论文提出可解释权重空间编程（iWSP），让LLM智能体在不使用反向传播的情况下，通过显式编写权重和特征代码来迭代构建透明的Transformer和大脑编码模型，从而实现真正可解释的科学建模。

    

    神经网络是强大的预测器，但其不透明性限制了它们在科学中的价值，因为科学的目标是解释而不仅仅是预测。例如在神经科学中，深度编码模型能够准确预测大脑对自然刺激的反应活动，却难以揭示究竟是哪些刺激特征驱动了这些预测。在此，我们提出可解释权重空间编程（iWSP），这是一种由LLM智能体通过在代码中显式编写模型权重和特征来迭代构建可解释模型的方法，全程不使用基于梯度的训练（反向传播）。由于没有反向传播来调优参数，智能体必须依靠其先验知识并对先前的试验进行推理，从而提出假设、手工构建并改进人类可理解的计算，这一流程类似于人类科学家的工作方式。LLM审计证实了这些手工构建模型的可解释性。我们首先通过让iWSP构建（任务）来验证该方法的有效性

    arXiv:2611.18114v1 Announce Type: cross  Abstract: Neural networks are powerful predictors, yet their opacity limits their usefulness in science, where the goal is explanation rather than mere prediction. In neuroscience, for example, deep encoding models accurately predict brain activity in response to natural stimuli, but struggle to reveal which stimulus features drive those predictions. Here we introduce interpretable weight-space programming (iWSP), a method in which an LLM agent iteratively builds interpretable models by explicitly programming their weights and features in code, without gradient-based training (backpropagation). Without backpropagation to tune parameters, the agent must draw on its prior knowledge and reason over previous trials to hypothesize, hand-craft, and refine human-interpretable computations mirroring the workflow of a human scientist. LLM audits confirm the interpretability of the hand-crafted models. We first validate iWSP by tasking it with constructin
    
[^48]: 多模态脑响应模型中拓扑组织的计算机模拟审计

    An In-Silico Audit of Topographic Organization in a Multimodal Brain-Response Model

    [https://arxiv.org/abs/2611.18105](https://arxiv.org/abs/2611.18105)

    该论文提出了一套计算机模拟审计方法（映射刺激、拟合流程及已知真值的正负对照模型），用于检验多模态脑响应模型 TRIBE v2 能否再现感觉皮层的经典拓扑组织，结果发现该模型缺失对侧半视野响应和极角映射等视网膜拓扑特征。

    

    TRIBE v2（d'Ascoli 等，2026）能够从视频、音频和文本预测整个大脑皮层表面的 fMRI 响应。其作者通过类别选择性定位（面孔、地点、身体、文字）和语言对比对该模型进行了验证，但并未验证感觉皮层的经典拓扑组织特性：视网膜拓扑映射、音调拓扑映射以及 MT+ 区域的运动选择性。从架构上看，存在一个明确的理由让人预期视网膜拓扑映射会失败：TRIBE 的视频特征在进入响应头之前经过了空间平均，这应当会丢弃视野位置信息。我提出了映射刺激、一个改编自标准 fMRI 映射分析的拟合流程，以及一个具有已知真值的替身模型作为正负对照，并将它们应用于 TRIBE v2 的预测，同时使用自旋检验将拟合出的映射图谱与人类图谱进行比较。TRIBE v2 呈现出明显的分化表现：不存在对侧半视野响应，也没有极角映射，且在各种刺激条件下都稳健地保持这一结果（原文在此处截断）。

    arXiv:2611.18105v1 Announce Type: cross  Abstract: TRIBE v2 (d'Ascoli et al., 2026) predicts fMRI responses across the cortical surface from video, audio and text. Its authors validated it with category-selective localizers (faces, places, bodies, words) and language contrasts, but not with the classic topographic organization of sensory cortex: retinotopy, tonotopy, and motion selectivity in area MT+. There is a clear architectural reason to expect retinotopy to fail: TRIBE's video features are averaged over space before the response head, which should discard visual-field position. I present mapping stimuli, a fitting pipeline adapted from standard fMRI mapping analyses, and a stand-in model with known ground truth that serves as a positive and negative control, and apply them to TRIBE v2's predictions, comparing the fitted maps with human atlases using spin tests. TRIBE v2 shows a sharp split. There is no contralateral hemifield response and no polar-angle map, robustly across stimu
    
[^49]: EcoVGGT：面向长视频高效且可流式前馈三维重建的固定键推理方法

    EcoVGGT: Pinned-Key Inference for Efficient and Streamable Feed-Forward 3D Reconstruction from Long Videos

    [https://arxiv.org/abs/2611.18099](https://arxiv.org/abs/2611.18099)

    EcoVGGT 提出了一种无需训练的推理框架，仅用 31 个均匀分布的帧构成固定键集并配合场景缓存机制，将跨帧注意力成本从随帧数二次方增长降为每帧恒定成本，实现了长视频的高效、可流式的前馈三维重建。

    

    诸如 VGGT 这类前馈多视角 Transformer 能够在一次前向传播中恢复视频的相机位姿与几何信息，但其跨帧注意力的计算代价随帧数呈二次方增长，且在长视频上的精度会下降。现有的大多数解决方案要么在计算预算内对所有帧的注意力进行近似，要么根据内容选择少量关键帧。我们假设长视频的注意力键无需查看其内容即可选定，并且在 VGGT 和 π³ 上的实验支持了这一假设：几十帧均匀分布的帧即可作为所有帧的键。我们提出了 EcoVGGT，这是一个无需训练的推理框架，其键构成一个由 31 帧均匀分布帧组成的固定键集。一次固定遍历将这些帧的 token 存入场景缓存，其余各帧以独立批次读取该缓存，从而精确复现与这些键的联合推理，且每帧成本恒定。在线流式模式可在录制过程中更新固定帧。六个中间全局块对齐各帧……（摘要在此处截断）

    arXiv:2611.18099v1 Announce Type: cross  Abstract: Feed-forward multi-view transformers such as VGGT recover the cameras and geometry of a video in one pass, but the cost of their cross-frame attention grows quadratically with the frame count, and their accuracy drops on long videos. Most remedies approximate attention over all frames within a budget or select a few keys by their content. We hypothesize that the keys of a long video can be chosen without looking at its content, and our experiments on VGGT and $\pi^3$ support it: a few dozen evenly spread frames can serve as the keys of every frame. We present EcoVGGT, a training-free inference framework whose keys form a Pinned Key Set of 31 such frames. A pin pass stores their tokens in a Scene Cache, and the other frames read it in independent batches that reproduce joint inference with these keys exactly, at a constant cost per frame. An Online Stream updates the pinned frames during recording. Six middle global blocks align the fra
    
[^50]: 机器会从同伴那里习得越轨行为吗？基于AI智能体的差异交往理论随机化检验

    Do Machines Learn Deviance From Their Peers? A Randomized Test of Differential Association With AI Agents

    [https://arxiv.org/abs/2611.18098](https://arxiv.org/abs/2611.18098)

    该研究巧妙利用AI智能体可被随机分配同伴的特性，首次通过随机化实验为差异交往理论提供了因果证据：AI智能体会从声称作弊的“越轨同伴”那里习得作弊行为，而从“诚实同伴”那里显著减少作弊。

    

    差异交往理论及其延伸——社会学习理论认为，违规行为以及使违规显得可接受的观念，是通过与他人的互动习得的。由于人们会自主选择朋友，这一主张一直难以检验。我们利用AI智能体来解决这一选择性问题，因为AI智能体仅通过交流进行学习，且可以被随机分配同伴。在一项预注册实验中，420个由三个智能体组成的团队（共1,260个智能体；涉及七个AI模型）完成了无法通过诚实方式解决的编程任务。一个脚本化的队友要么声称自己作弊了（越轨同伴），要么表示自己已诚实地报告了该问题（诚实同伴），要么什么都不说（沉默同伴）。其中一半团队被告知有审计员会阅读他们的聊天记录。与沉默同伴相比，越轨同伴提高了作弊率（66.2% vs. 62.9%；OR = 1.96），而诚实同伴降低了作弊率（55.7%；OR = 0.35）。在作弊的智能体中，越轨同伴增加了……（原文摘要在此处截断）

    arXiv:2611.18098v1 Announce Type: new  Abstract: Differential association theory and its extension, social learning theory, hold that rule-breaking, and the definitions that make it acceptable, are learned through interaction with others. Because people choose their friends, this claim has been hard to test. We addressed this selection problem with AI agents, which learn only through communication and can be assigned to peers at random. In a preregistered experiment, 420 teams of three agents (1,260 agents; seven AI models) worked on coding tasks that could not be solved honestly. A scripted teammate said that it had cheated (deviant peer), that it had reported the problem honestly (honest peer), or said nothing (silent peer). Half of the teams were told that an auditor read their chat. Compared with a silent peer, a deviant peer increased cheating (66.2% vs. 62.9%; OR = 1.96) and an honest peer reduced it (55.7%; OR = 0.35). Among agents that cheated, a deviant peer increased claims t
    
[^51]: AG-JEPA：用于神经退行性疾病表征学习的解剖学引导联合嵌入预测架构

    AG-JEPA: Anatomically Guided Joint Embedding Predictive Architecture for Neurodegenerative Representation Learning

    [https://arxiv.org/abs/2611.18097](https://arxiv.org/abs/2611.18097)

    该论文提出AG-JEPA，通过在JEPA预训练中引入与图像块token交互的解剖学token来学习AD相关脑区的区域特定表征，并进一步提出Prog-AG-JEPA将疾病进展轨迹建模为表征空间中的条件化潜在位移。

    

    脑部MRI可以在可测量的认知衰退变得明显之前揭示阿尔茨海默病（AD）的早期迹象。然而，在从认知正常（CN）向轻度认知障碍（MCI）过渡的阶段，疾病相关信号往往十分微弱，且仅局限于少数脑区。现有的联合嵌入预测架构（JEPA）模型学习的是一般性的结构表征，但并未显式捕获AD所涉及的解剖区域。我们提出了解剖学引导的JEPA（AG-JEPA），它在预训练过程中引入与图像块token进行交互的解剖学token，以从与神经退行性变相关的脑区中学习区域特定的表征。在这一表征空间的基础上，我们进一步提出了Prog-AG-JEPA，它将疾病进展轨迹建模为该空间内的条件化潜在位移。我们在涵盖阿尔茨海默病诊断的五个下游任务上对AG-JEPA进行了评估……（原文摘要在此处截断）

    arXiv:2611.18097v1 Announce Type: cross  Abstract: Brain MRI can reveal early signs of Alzheimers disease (AD) before measurable cognitive decline becomes apparent. However, during the transition from cognitively normal (CN) to mild cognitive impairment (MCI), the disease-related signals are often subtle and confined to only a few brain regions. Existing Joint Embedding Predictive Architecture (JEPA) models learn general structural representations but do not explicitly capture the anatomical regions affected in AD. We propose Anatomically Guided JEPA (AG-JEPA), which introduces anatomical tokens that interact with patch tokens during pretraining to learn region-specific representations from brain regions relevant to neurodegeneration. Building on this representation space, we further propose Prog-AG-JEPA, which models disease progression trajectory as a conditioned latent displacement within this space. We evaluate AG-JEPA across five downstream tasks spanning Alzheimer's disease diagn
    
[^52]: 因果注意力中Token动力学的双聚类状态

    Two-Clustering Regime of Token Dynamics in Causal Attention

    [https://arxiv.org/abs/2611.18088](https://arxiv.org/abs/2611.18088)

    本文部分证明了因果注意力动力学在值变换最大特征值为正时收敛为与主导特征向量对齐的双聚类的猜想，并揭示了高维Token轨迹坍缩到由少数主导方向支配的二维结构的严格降维现象。

    

    我们部分解决了 [Karagodin 等，Clustering in Causal Attention Masking (NeurIPS 2024)] 中的猜想 1，并将其扩展到时间相关的参数区域。该猜想预测，因果注意力动力学收敛到与值变换最大特征值 λ_max 相关联的特征向量对齐的两个聚类。我们证明了当 λ_max > 0 时该行为成立，而当 λ_max ≤ 0 时通常不成立。由此产生的双聚类状态在实践中很常见，且比先前研究的经典单聚类情形更具挑战性。我们工作的一个核心贡献是识别出一个严格的降维现象：尽管Token轨迹在高维嵌入空间中演化，但它们会坍缩到一个由少数主导方向支配的二维结构上。基于这一见解，我们引入了一种新的分析框架……

    arXiv:2611.18088v1 Announce Type: new  Abstract: We partially resolve Conjecture 1 of [Karagodin et al., Clustering in Causal Attention Masking (NeurIPS 2024)] and extend it to time-dependent parameter regimes. The conjecture predicts that causal attention dynamics converge to two clusters aligned with the eigenvectors associated with the largest eigenvalue $\lambda_{\max}$ of the value transformation. We prove that this behavior holds when $\lambda_{\max}>0$ and is generally false when $\lambda_{\max}\le0$. The resulting two-cluster regime is common in practice and substantially more challenging than the classical single-cluster setting studied previously. A central contribution of our work is the identification of a rigorous dimension reduction phenomenon: despite evolving in a high-dimensional embedding space, token trajectories collapse onto a two-dimensional structure governed by a small number of dominant directions. Building on this insight, we introduce a new analytical framewo
    
[^53]: TelePhysica：物理交互场景的智能体式重建

    TelePhysica: Agentic Reconstruction of Physically Interactive Scenes

    [https://arxiv.org/abs/2611.18078](https://arxiv.org/abs/2611.18078)

    TelePhysica提出了一个闭环框架，通过视觉-语言编码智能体和两阶段验证循环，从单张照片重建可执行的多模式物理世界，克服了扩散模型缺乏持久3D力学平衡与单次程序化重建场景易坍塌的问题，实现有状态的3D交互和基于轨迹的视频合成。

    

    直接的图像到视频扩散模型将像素合成与隐式潜在运动纠缠在一起，阻碍了持续的3D力学平衡、生成过程中的干预以及反事实的力控制。相反，从未标定的照片进行单次程序化3D重建会使单目深度和支撑间隙约束不足，导致重建的场景相互穿透或在重力作用下坍塌。我们提出了TelePhysica，一个闭环框架，它从单张照片重建出可执行的、多模式的物理世界，以驱动有状态的3D交互和基于轨迹的视频合成。一个视觉-语言编码智能体在两阶段验证循环中合成可分离的视觉网格和声明式物理模型，该循环将像素对齐的轮廓评估与零力重力平衡预检相结合。经验证的世界编译为一个统一的求解器，耦合刚体、关（摘要在此处截断）

    arXiv:2611.18078v1 Announce Type: cross  Abstract: Direct image-to-video diffusion models entangle pixel synthesis with implicit latent motion, precluding persistent 3D mechanical equilibrium, mid-rollout intervention, and counterfactual force control. Conversely, single-pass programmatic 3D reconstruction from an uncalibrated photograph leaves monocular depth and support clearance underconstrained, causing reconstructed scenes to interpenetrate or collapse under gravity. We present TelePhysica, a closed-loop framework that reconstructs an executable, multi-regime physical world from a single photograph to drive stateful 3D interaction and trajectory-grounded video synthesis. A vision-language coding agent synthesizes separable visual meshes and declarative physical models within a two-stage verification loop that pairs pixel-aligned silhouette critique with a zero-force gravitational equilibrium preflight. The verified world compiles into a unified solver coupling rigid bodies, articu
    
[^54]: A³：面向协同边缘计算的自适应音频智能体框架

    A$^3$: Adaptive Audio Agent Framework for Collaborative Edge Computing

    [https://arxiv.org/abs/2611.18064](https://arxiv.org/abs/2611.18064)

    A³框架通过在服务器部署完整流水线、在设备端保留后备智能体，并利用链路感知截止时间动态对冲，实现了链路健康时媲美纯服务器方案的性能，同时保证完全断网时仍能可靠响应。

    

    嵌入式音频智能体需要依靠附近的边缘服务器来获得快速响应，但依赖服务器的智能体在链路失效时将无法返回任何结果。我们提出了A³，这是一个自适应音频智能体框架，它将整个流水线阶段部署在服务器上，在具备承载能力的设备上保留完整的后备智能体，并通过链路感知的截止时间机制对两条路径进行对冲，以决定何时放弃使用服务器。在Jetson设备和RTX A4500服务器上的实验表明，A³在链路健康时与纯服务器智能体的表现相当，在AGX设备完全断网的情况下能够响应每一个请求，并且在模拟断网场景中优于百分位对冲策略。

    arXiv:2611.18064v1 Announce Type: new  Abstract: Embedded audio agents need a nearby edge server for speed, but a server-dependent agent returns nothing when the link fails. We present A$^3$, an adaptive audio agent framework that places whole pipeline stages on the server, keeps a complete fallback agent on devices that can host one, and hedges both paths with a link-aware deadline that decides when to give up on the server. On Jetson devices and an RTX A4500 server, A$^3$ matches the server-only agent on a healthy link, answers every request in a full outage on the AGX, and beats a percentile hedge in simulated outages.
    
[^55]: RL$^{\mathrm{RT}}$：端侧强化学习运行时

    RL$^{\mathrm{RT}}$: On-Device Reinforcement Learning Runtime

    [https://arxiv.org/abs/2611.18035](https://arxiv.org/abs/2611.18035)

    RL$^{\mathrm{RT}}$通过将执行器与学习器的耦合变为可调参数（刷新界限与松弛感知的GPU许可），在共享嵌入式GPU上实现了高效的端侧强化学习，将16.8M参数策略的截止期限错过率从85%-100%降至1%以下。

    

    端侧强化学习在与其它工作负载共享的单个嵌入式GPU上运行周期性执行器和学习器。由于学习器会生成执行器的权重，现有设计要么阻塞执行器，要么导致其策略发生漂移。我们提出了RL$^{\mathrm{RT}}$，一个端侧运行时系统，它将这种耦合转化为一个可调参数：刷新界限限制刷新之间执行器的发布次数，而具有松弛感知准入机制的GPU许可则对执行器进行优先调度，并在任意基础调度器之上利用空闲时间进行训练。在两块Jetson开发板上考虑真实的刷新开销的实验中，RL$^{\mathrm{RT}}$将一个16.8M参数策略的截止期限错过率控制在1%以下，而没有使用它时该比率高达85%至100%。

    arXiv:2611.18035v1 Announce Type: cross  Abstract: On-device reinforcement learning runs a periodic actor and a learner on one embedded GPU shared with other workloads. Because the learner produces the actor's weights, existing designs stall the actor or let its policy drift. We present RL$^{\mathrm{RT}}$, an on-device runtime that makes this coupling a tunable parameter: a refresh bound limits actor releases between refreshes, and a GPU permit with slack-aware admission prioritises the actor and fills spare time with training atop any base scheduler. Under the real refresh cost on two Jetson boards, RL$^{\mathrm{RT}}$ keeps a 16.8 M-parameter policy's deadline misses below 1%, against 85 to 100% without it.
    
[^56]: 理解配电网络中基于快照的故障诊断的局限性

    Understanding the limits of snapshot-based fault diagnosis in distribution networks

    [https://arxiv.org/abs/2611.17998](https://arxiv.org/abs/2611.17998)

    本文通过系统分离分类器、特征、测量误差和故障电阻等误差来源，首次揭示了基于快照的配电网络故障诊断的性能瓶颈在于测量误差（尤其是高阻抗故障）而非分类器本身。

    

    逆变器型资源会削弱配电网络中的故障电流，而基于快照的诊断方法——即利用某一时刻少数母线上的电压相量对故障进行分类——提供了一种低成本的第一阶段筛查手段。然而，已发表的准确率结果建立在不同的噪声假设之上，无法表明误差究竟来自分类器、特征、测量误差还是故障电阻。本文在三个IEEE测试馈线和一个包含分布式发电的公开数据集上，使用四种特征集、十三个分类器以及由总矢量误差（TVE）等级定义的测量误差，对这些误差来源进行了系统分离。结果表明，分类器并非限制因素，因为在无噪声条件下完整的相量快照能够正确分类所有故障，剩余误差来自高阻抗故障上的测量误差。三个电压传感器在1%精度等级下可分类90-96%的故障，采用配电微型相量测量单元时可达96-99%。

    arXiv:2611.17998v1 Announce Type: new  Abstract: Inverter-based resources weaken fault currents in distribution networks, and snapshot-based diagnosis, which classifies a fault from voltage phasors at a few buses at one instant, offers a cheap first-stage screen. Its published accuracies, however, rest on different noise assumptions and do not show whether the error comes from the classifier, the features, the measurement error, or the fault resistance. This paper separates these sources on three IEEE test feeders and a public data set with distributed generation, using four feature sets, thirteen classifiers, and measurement error defined by total vector error classes. The classifier is not the limiting factor, because the full phasor snapshot classifies every fault correctly without noise, and the remaining error comes from measurement error on high-impedance faults. Three voltage sensors classify 90-96% of faults at the 1% accuracy class and 96-99% with distribution micro phasor mea
    
[^57]: LACCO：带计算-通信重叠的本地累积

    LACCO: Local Accumulation with Computation--Communication Overlap

    [https://arxiv.org/abs/2611.17991](https://arxiv.org/abs/2611.17991)

    提出LACCO两阶段算法，在不引入外层更新陈旧性的前提下实现计算与通信重叠并保持各副本更新一致性，在语言模型预训练中持续优于Eager和Delayed DiLoCo等通信重叠基线。

    

    在低带宽网络上训练语言模型需要减少同步开销。诸如 DiLoCo 等本地优化方法仅在执行若干本地步骤后才进行通信，但外层梯度平均和外层更新仍然处于关键路径上。现有的计算-通信重叠方法通过延迟更新或副本本地更新来避免这种停滞，但这可能损害收敛性能或导致副本之间产生漂移。我们提出了 LACCO（带计算-通信重叠的本地累积），这是一种两阶段算法，它将全局通信与本地计算重叠，同时应用由所有副本共享的全局聚合更新，而不会在外层更新中引入陈旧性。我们为光滑非凸目标建立了收敛性保证。在使用 FineWeb 数据集进行的语言模型预训练实验中，LACCO 持续优于 Eager 和 Delayed DiLoCo 等通信重叠基线方法。

    arXiv:2611.17991v1 Announce Type: new  Abstract: Training language models across low-bandwidth networks requires reducing synchronization overhead. Local optimization methods such as DiLoCo communicate only after several local steps, but outer gradient averaging and the outer update remain on the critical path. Existing computation-communication overlap methods avoid this stall through delayed or replica-local updates, which can degrade convergence or cause replicas to drift. We propose LACCO (Local Accumulation with Computation-Communication Overlap), a two-stage algorithm that overlaps global communication with local computation while applying globally aggregated updates shared by every replica, without introducing staleness in the outer update. We establish convergence guarantees for smooth non-convex objectives. Experiments on language-model pre-training with FineWeb show that LACCO consistently outperforms communication-overlapped baselines such as Eager and Delayed DiLoCo. These 
    
[^58]: 基于深度学习的交互式脑肿瘤分割界面

    Interactive deep-learning based interface for brain tumour segmentation

    [https://arxiv.org/abs/2611.17983](https://arxiv.org/abs/2611.17983)

    本文提出了一种默认全自动运行但允许人工轻松校正的深度学习交互式脑肿瘤分割系统，从而能够为所有患者确保足够高的分割精度。

    

    从磁共振（MR）图像中进行脑肿瘤分割是肿瘤治疗（无论是手术切除还是放疗）不可或缺的组成部分。这两种应用都需要高度的精确性，但由于这些肿瘤在大小、形状、数量和位置上存在高度的可变性，很难确保对所有患者都能达到这种精确性。这给通过深度学习进行自动分割带来了困难——当图像在所使用的特定MR序列方面存在差异（即使差异很细微，且不同厂商的序列各不相同）时，深度学习模型可能出现灾难性的分割错误。本文提出了一种基于深度学习的交互式分割系统，该系统默认情况下完全自动运行，同时具备易于人工校正的能力，使用户能够确保所有患者的脑肿瘤分割精度足够高。该界面从其自动方法的准确性和……

    arXiv:2611.17983v1 Announce Type: new  Abstract: Brain tumour segmentation from MR images is an integral part of their treatment, either through surgical resection or radiotherapy. Both applications require a high degree of precision that can be difficult to ensure for all patients due to the high degree of variability in the size, shape, number, and location of these tumours. This leads to difficulties with their automatic segmentation via deep learning, which can display catastrophic segmentation errors when presented with images that differ, even subtly, in terms of the specific MR sequence used, which varies from vendor to vendor. This article presents a deep-learning-based interactive segmentation system that, while being fully automatic by default, has the capability of being easily corrected by human users, allowing them ensure sufficiently high brain tumour segmentation accuracy for all patients. This interface is evaluated both in terms of the accuracy of its automatic approac
    
[^59]: TotalSegmentator Aorta 报告：在CT图像上11个标准化解剖标志点处的自动化主动脉直径测量

    TotalSegmentator Aorta Report: Automated Aortic Diameter Measurement at 11 Standardized Landmarks on CT Images

    [https://arxiv.org/abs/2611.17968](https://arxiv.org/abs/2611.17968)

    该研究开发并验证了一种全自动开源工具，可在CT图像上遵循指南推荐的11个标准化解剖标志点自动测量主动脉直径，与人工参考标准的平均绝对误差仅为1.06毫米。

    

    目的：开发并评估一种全自动、开源的工具，该工具遵循当前指南推荐，可在任何CT图像上从主动脉瓣环到髂动脉分叉处的11个标准化解剖标志点测量主动脉直径。材料与方法：这项回顾性单中心研究纳入了2025年4月至2026年10月间230名患者的239次动脉期CT血管造影检查（中位年龄65岁[范围18-99岁]；男性163名）（其中209次为心电图门控）。该工具基于TotalSegmentator对主动脉分支和椎体的分割结果定位解剖标志点，并测量每个标志点处的最大直径。由一名资深放射科医师审核并在需要时校正自动测量结果，以建立参考标准。一致性通过平均绝对误差（MAE）、偏倚以及大于2mm的测量差异进行评估。结果：在所有标志点处，MAE为1.06 mm（95% CI: 0.91, 1.22），86.6%的[摘要内容在此处截断]

    arXiv:2611.17968v1 Announce Type: new  Abstract: Purpose: To develop and evaluate a fully automated, open-source tool that measures aortic diameters at 11 standardized landmarks from the aortic annulus to the bifurcation on any CT image, following current guideline recommendations.   Materials and Methods: This retrospective single-center study included 239 arterial phase CT angiography examinations of 230 patients (median age, 65 years [range, 18-99 years]; 163 men) performed between April 2025 and October 2026 (209 electrocardiographically [ECG] gated). The tool locates landmarks from TotalSegmentator segmentations of aortic branches and vertebrae and measures the maximum diameter at each. A senior radiologist reviewed and, where needed, corrected the automated measurements to create the reference standard. Agreement was assessed with mean absolute error (MAE), bias, and differences greater than 2 mm.   Results: Across all landmarks, MAE was 1.06 mm (95% CI: 0.91, 1.22), and 86.6% (9
    
[^60]: 从图像到定位编辑：手写作文中文语法纠错的现实世界基准

    From Images to Grounded Edits: A Real-World Benchmark for Chinese Grammatical Error Correction in Handwritten Essays

    [https://arxiv.org/abs/2611.17966](https://arxiv.org/abs/2611.17966)

    该论文提出了定位式中文语法纠错新任务及首个基于真实K-12手写作文图像的基准G-CGEC-Eval，要求模型同时预测纠错内容及其在图像上的位置，弥合了传统文本级语法纠错与现实教育场景之间的差距。

    

    中文语法纠错通常被定义为文本到文本的问题。虽然这种形式适用于数字文本，但在现实的K-12教育场景中却存在不足，因为在这些场景中，AI辅助纠错需要直接在作文图像上进行，而这些作文主要是手写的，并且存在许多不规范之处。为了弥合这一差距，我们引入了定位式中文语法纠错，这是一个图像到定位编辑的任务，要求系统同时预测纠正内容及其在作文图像上的位置。我们提出了G-CGEC-Eval，一个包含现实世界K-12中文作文图像及细粒度标注的基准。在评估方面，我们改编了目标检测中基于IoU的指标，并提出了一种基于中心距离匹配的更宽容的替代方案。使用该基准，我们在直接单次处理设置下评估了开源和专有的视觉语言模型，以及带有辅助组件的设置……

    arXiv:2611.17966v1 Announce Type: cross  Abstract: Chinese Grammatical Error Correction (CGEC) is typically formulated as a text-to-text problem. While suitable for digital texts, this formulation falls short in real-world K-12 education, where AI-assisted corrections are expected directly on essay images, which are primarily handwritten and exhibit many irregularities. To bridge this gap, we introduce Grounded Chinese Grammatical Error Correction (G-CGEC), an image-to-grounded-edit task requiring systems to predict both corrections and their positions on essay images. We propose G-CGEC-Eval, a benchmark comprising real-world images of K-12 Chinese essays with fine-grained annotations. For evaluation, we adapt IoU-based metrics from object detection and propose a more tolerant alternative based on center-distance matching. Using this benchmark, we evaluate open-source and proprietary vision-language models under a direct single-pass setting, as well as settings with auxiliary component
    
[^61]: WebArena-Pro：一个面向Web智能体的异构、多模态、可复现基准测试

    WebArena-Pro: A Heterogeneous, Multimodal, Reproducible Benchmark for Web Agents

    [https://arxiv.org/abs/2611.17955](https://arxiv.org/abs/2611.17955)

    WebArena-Pro是一个涵盖20个自托管Web应用、300个任务的异构多模态可复现基准测试，首次支持对Web智能体理解和运用音频、视频内容能力的评估。

    

    基于大语言模型和视觉-语言模型的Web智能体正越来越多地被应用于涉及异构应用、多模态内容和有状态工作流的现实浏览器任务中。然而，现有的可复现Web智能体基准测试仅覆盖来自少数软件类别的少量Web应用，且模态仅限于文本和视觉。实时基准测试虽然扩大了站点覆盖范围，但由于页面和数据在运行之间会发生漂移，牺牲了可复现性。此外，现有基准测试未能有效评估智能体能否理解和使用嵌入在Web任务中的音频和视频内容。为弥补这些不足，我们提出了WebArena-Pro，这是一个包含300个任务的基准测试，涵盖六个领域类别中的20个自托管Web应用，跨越不同的界面惯例、工作流和数据模型。在所评估的智能体中，Codex with GPT 6.1 Sol 取得了最佳表现。

    arXiv:2611.17955v1 Announce Type: new  Abstract: Web agents powered by large language and vision-language models are increasingly applied to realistic browser work that spans heterogeneous applications, multimodal content, and stateful workflows. However, existing reproducible web-agent benchmarks cover only a small number of web applications drawn from a few software categories, and restrict modality to text and vision. Live benchmarks broaden site coverage but sacrifice reproducibility, since pages and data drift between runs. Moreover, existing benchmarks do not meaningfully evaluate whether agents can understand and use audio and video content embedded within web tasks. To address these gaps, we introduce WebArena-Pro, a benchmark comprising 300 tasks across 20 self-hosted web applications in six domain categories, spanning distinct interface conventions, workflows, and data models. Across the evaluated agents, the best performance is achieved by both Codex with GPT 6.1 Sol and Cla
    
[^62]: Transformer范式：动态处理与一种新的推理原理

    The Transformer Paradigm: Dynamic Processing and a New Principle of Inference

    [https://arxiv.org/abs/2611.17953](https://arxiv.org/abs/2611.17953)

    该论文提出Transformer革命本质上是一种基于“动态构建变换”的新推理范式（即先适配先前习得的变换再应用，通过输出-权重互连实现，称为序列级交互式动态并行处理SIDPP），而非深度学习的简单量级扩张，该框架可解释强提示敏感性并开启可解释性、控制与对齐等研究方向。

    

    我们认为，Transformer革命不应被简单地理解为深度学习的量级扩张，而应被理解为一种基于不同推理原理的新范式的出现。我们将Transformer解释为一个这样的系统：其中token表示通过参数化变换（即“变换概念”）进行转换。在推理过程中，其中一些变换会根据当前输入被动态构建：先对先前习得的变换进行调整（适配），然后再加以应用。在机械结构层面，这种动态处理是通过输出-权重互连来实现的，即某些网络的输出决定其他网络的权重。由此产生的处理模式，我们称之为序列级交互式动态并行处理（SIDPP），它为解释强烈的提示敏感性提供了框架，并开启了关于可解释性、控制、对齐等方面的研究方向。

    arXiv:2611.17953v1 Announce Type: new  Abstract: We argue that the Transformer revolution is not adequately understood as a merely quantitative expansion of deep learning, but as the emergence of a new paradigm based on a different principle of inference. We interpret the Transformer as a system in which token representations are transformed by parameterized transformations, or transforming concepts. During inference, some of these transformations are dynamically constructed from the current input: previously acquired transformations are first adapted and then applied. At the mechanical-structural level, this dynamic processing is realized through output-weight interconnections, whereby outputs of some networks determine the weights of others. The resulting mode of processing, which we call Sequence-level Interactive Dynamic Parallel Processing (SIDPP), provides a framework for explaining strong prompt sensitivity and opens research directions concerning interpretability, control, alte
    
[^63]: 迈向干扰无关的射频信号源分离

    Toward Interference-Agnostic RF Source Separation

    [https://arxiv.org/abs/2611.17950](https://arxiv.org/abs/2611.17950)

    本文提出在所有信号-干扰对上训练单一统一分离器，测试时无需识别干扰类型，就能以单个专用模型的规模达到接近全部专用模型的分离性能，并在面对未训练过的干扰时避免信号质量恶化。

    

    将感兴趣的射频信号从同频干扰中分离出来，目前主要依靠专用分离器实现——每个分离器只针对一对特定的信号与干扰类型进行训练，因而依赖于一个严重受扰的接收机无法识别的标签。我们证明这种依赖并非无害：当这类分离器被应用于其未曾训练过的干扰时，通常无法恢复信号，并且在我们测量的大多数情况下，处理后的信号甚至比不做任何处理更糟。为此，我们在所有信号-干扰对上训练一个统一分离器，并在测试时不向其提供任何关于所面对干扰类型的信息，采用了两种没有任何共同组件的架构：ICASSP 2024 射频挑战赛的卷积 WaveNet，以及我们从音乐源分离领域改造而来的频带切分 Transformer（band-split transformer）。两者表现一致：在仅相当于一个专用模型的规模下，该统一分离器的性能接近于它所替代的全部专用模型，保持误码率……（原文摘要在此处截断）

    arXiv:2611.17950v1 Announce Type: new  Abstract: Separating a radio-frequency signal of interest from a co-channel interference is mostly addressed with specialized separators, each trained for one pair of signal and interference types and therefore relying on a label that a heavily jammed receiver cannot identify. We show that this reliance is not benign: applied to an interference it was not trained on, such a separator usually fails to recover the signal, and in most of the cases we measure it leaves the signal worse than no processing at all. We then train a single separator over every pair, giving it no indication at test time of which one it faces, on two architectures with no component in common: the convolutional WaveNet of the ICASSP 2024 RF Challenge, and a band-split transformer that we adapt from music source separation. Both behave the same way. At the size of one specialized model, the unified separator stays close to the whole models it replaces, keeping the bit error ra
    
[^64]: 你能信任你的金融智能体吗？对一款已发布的金融研究智能体的漂移、诊断与可审计干预

    Can You Trust Your Finance Agent? Drift, Diagnosis and Auditable Intervention on One Shipped Financial Research Agent

    [https://arxiv.org/abs/2611.17939](https://arxiv.org/abs/2611.17939)

    论文对一款现成发布的金融研究智能体进行系统审计，发现其基准分数不稳定且运行成本波动高达17%，并按指令、工具、模型等责任层对失败轨迹进行归因分类，其诊断结果与人类盲评者一致率达94%，进而提出可审计的针对性干预方法。

    

    你能信任一个并非由你构建的金融智能体吗？我们采用Vals金融智能体基准v2（Vals Finance Agent Benchmark v2）随附的参考智能体，不对其进行任何调优，并提出了四个排行榜分数无法回答的问题。分数是否稳定？在18个问题上进行七次完全相同的运行，149项评分标准中通过的数量在105至123之间浮动，且账单费用差异高达17%。当智能体失败时，应由谁来负责修复？我们按照事先冻结的分类法，将来自九个保留问题的三次运行共20条失败轨迹，依据应由哪一层的责任方来修复进行归类：13条属于智能体的指令，3条属于其工具，4条属于模型，没有一条属于其数据。这样的诊断是否可信？来自三个模型家族的四位盲评者，在仅看到轨迹的情况下，在94%的案例中与我们的LLM分析器指出相同的故障，这一比例与盲评者彼此之间的一致率（93%）相当。所做的更改是否可信？一个优化循环读取轨迹，每轮进行一个针对性的更改……

    arXiv:2611.17939v1 Announce Type: cross  Abstract: Can you trust a finance agent you did not build? We take the reference agent shipped with the Vals Finance Agent Benchmark v2, leave it untuned, and ask four questions a leaderboard score does not answer. Is the score stable? Seven identical runs on 18 questions pass between 105 and 123 of 149 rubric criteria, and the bill varies by up to 17%. When the agent fails, who owns the fix? We classify 20 failed trajectories, from three runs of nine held-out questions, by the layer whose owner would remedy them, under a taxonomy frozen in advance: 13 belong to the agent's instructions, 3 to its tools, 4 to the model, none to its data. Can that diagnosis be trusted? Four blind readers from three model families, given only the traces, name the same fault as our LLM analyzer in 94% of cases, the rate at which they agree with one another (93%). Can a change be trusted? An optimization loop reads the traces, makes one targeted change per round and 
    
[^65]: 从信息流中测量：AI视频生成的不可持续增长

    Measured from the feed: the unsustainable growth of AI video generation

    [https://arxiv.org/abs/2611.17938](https://arxiv.org/abs/2611.17938)

    该研究通过实测两个商业文生视频平台的年度产出并建立前瞻性碳足迹模型，发现若使用量持续增长，尽管有技术进步，到2030年排放仍将激增近75倍，因此减排必须同时依靠限制视频生成增长和实现大幅效率提升。

    

    文本到视频生成已实现商业化规模运行，但据我们所知，其更广泛的环境影响仍然研究不足。服务提供商很少披露视频产量或能源需求；我们测量了两个商业平台的年度视频产出，并构建了一个前瞻性模型，用于评估此类服务从训练到存储的绝对环境足迹。我们的预测表明，减少排放既取决于限制视频生成的增长，也取决于实现大幅度的效率提升。在持续增长的情况下，尽管有技术进步，到2030年排放量仍将增长近七十五倍。而当使用量保持在2026年水平时，同样的技术进步可使排放量减少约五分之二。这些估算是一个保守的下限：它们仅涵盖众多AI服务中的一种；其他服务和用途将使总量更高。

    arXiv:2611.17938v1 Announce Type: new  Abstract: Text-to-video generation runs at commercial scale, but its broader environmental impacts remain, to our knowledge, understudied. Providers rarely disclose video volumes or energy needs; we measured the yearly video output of two commercial platforms and built a prospective model of such services' absolute footprint from training to storage. Our projections suggest that reducing emissions depends on both limiting growth in video generation and achieving substantial efficiency gains. Under continued growth, emissions rise nearly seventy-five-fold by 2030 despite technological improvements. When usage remains at 2026 levels, the same improvements reduce emissions by about two-fifths. These estimates are a conservative lower bound: they cover one AI service among many; other services and uses would push the total higher.
    
[^66]: 基于反事实见证映射追踪实际原因

    Tracing Actual Causes with Counterfactual Witness Maps

    [https://arxiv.org/abs/2611.17937](https://arxiv.org/abs/2611.17937)

    该论文提出“见证目标”与祖先干预文法以构造有限的有向无环见证映射，并给出首个可靠且完备的枚举算法TrAC，用于找出无环结构因果模型中给定情境下的所有实际原因。

    

    哈尔彭式实际因果是对无环结构因果模型中哪些事件在特定情境下导致了特定结果的形式化研究，它是法律责任、道德归责以及因果损害评估的基础。尽管该逻辑框架十分丰富，但此前尚不存在已知可靠且完备的枚举程序来找出给定情境下的所有实际原因，这阻碍了责任归属、归责评估和因果损害分级等需要对完整原因集合进行聚合的下游任务。我们通过引入“见证目标”（即包含每一个反事实见证的联合配置集合）来填补这一空白，并结合祖先干预文法，推导出一个有限的有向无环见证映射。我们证明，对于每一个实际原因，都存在一个偶然性集合，使得该联合干预集合作为此映射的一个节点出现，并提出了一个可靠且（完备的）实际原因追踪算法TrAC……

    arXiv:2611.17937v1 Announce Type: new  Abstract: Halpernian actual causation, the formal study of which events in a specific situation caused a specific outcome in acyclic structural causal models, underpins legal responsibility, moral blame, and the assessment of causal harm. Despite the rich logical framework, no prior sound and complete enumeration procedure is known for finding all actual causes in a given setting, blocking downstream tasks such as responsibility attribution, blame assessment, and causal-harm grading that aggregate over the full cause set. We close this gap by introducing witness targets: sets of joint configurations containing every counterfactual witness. In combination with an ancestor intervention grammar we derive a finite directed acyclic witness map. We prove that for every actual cause there exists a contingency set such that the joint intervention set appears as a node of this map, and present an algorithm for Tracing Actual Causes (TrAC) that is sound and
    
[^67]: OutfitterBench：评估大语言模型智能体在长周期资源分配与承诺协调上的能力

    OutfitterBench: Evaluating LLM Agents on Long-Horizon Resource Allocation and Commitment Coordination

    [https://arxiv.org/abs/2611.17933](https://arxiv.org/abs/2611.17933)

    本文提出OutfitterBench基准，让大语言模型智能体在200天内管理一个户外装备租赁业务，以评估其在自身承诺形成未来约束的长周期场景中进行资源分配与承诺协调的能力，实验显示多数前沿模型难以在保持计划可行性的同时重新评估资源分配。

    

    在许多长周期环境中，智能体自身的选择会带来未来的约束：对有限容量的承诺会建立未来的义务，并使后续的机会失去可能。因此，要持续保持良好性能，既需要让不断演进的计划保持可行，又需要在机会变化时将资源分配的价值与其所取代的备选方案进行比较。近期的基准测试越来越多地评估大语言模型智能体在持续运营中的表现，但对于智能体如何在机会变化时重新评估资源分配、同时保持自身承诺的可行性，关注较少。我们提出了OutfitterBench，在该基准中，智能体需要管理一个户外装备租赁业务200天，从包含46种产品的目录中采购，并处理约1,100个预订请求。在固定的资金上限下，预订和不确定的上门客流需求共同竞争可用性随时间变化的可重复使用装备。对七个前沿大语言模型的实验表明，若干智能体的表现不佳（原文摘要在此处截断）。

    arXiv:2611.17933v1 Announce Type: new  Abstract: In many long-horizon settings, an agent's own choices create future constraints: committing finite capacity establishes future obligations and forecloses later opportunities. Sustaining performance therefore requires both keeping evolving plans feasible and comparing the value of resource allocations with the alternatives they displace. Recent benchmarks increasingly evaluate LLM agents in continuing operations, but give less attention to how agents keep their own commitments feasible while reassessing resource allocations as opportunities change. We introduce OutfitterBench, where agents manage an outdoor-equipment rental business for 200 days, purchasing from a 46-product catalog and handling roughly 1,100 reservation requests. Under a fixed capital ceiling, reservations and uncertain walk-in demand compete for reusable equipment whose availability changes over time. Experiments with seven frontier LLMs show that several agents underpe
    
[^68]: MineralExp：面向真实世界矿产勘探中实验设计的LLM智能体基准测试

    MineralExp: Benchmarking LLM Agents for Experimental Design in Real-World Mineral Exploration

    [https://arxiv.org/abs/2611.17931](https://arxiv.org/abs/2611.17931)

    提出首个面向真实矿产勘探场景的MineralExp基准，用于评估LLM智能体自动化地球化学实验设计的能力，并解决了实验方法多样性与主观性高、开放式方案难以客观评分两大难题。

    

    在矿产勘探中，验证地质假设依赖于设计多种地球化学实验。虽然大语言模型（LLM）智能体日益自动化地完成下游的地球科学数据分析任务，但一个关键的上游能力在很大程度上仍未被探索：自动化地球化学实验设计，即在开展昂贵的实验室工作之前决定应该进行哪些实验。评估LLM智能体的这一能力面临两大困难。首先，地球化学实验具有高度的方法多样性和主观性：为了检验同一假设，不同的地质学家可能会选择不同的矿物和仪器，因此没有单一的实验方案可以作为标准参考答案。其次，开放式的实验方案难以表示为结构化标签，这使得客观评分变得困难。为了应对这些挑战，我们提出了MineralExp，这是一个将地球化学实验设计进行重新表述的基准……

    arXiv:2611.17931v1 Announce Type: cross  Abstract: In mineral exploration, verifying geological hypotheses relies on designing a variety of geochemical experiments. While large language model (LLM) agents increasingly automate downstream geoscientific data analysis, a crucial upstream capability remains largely unexplored: automated geochemical experimental design, i.e., deciding which experiments to conduct before committing to costly laboratory work. Evaluating this capability of LLM agents is difficult for two reasons. First, geochemical experiments exhibit high methodological diversity and subjectivity: to test the same hypothesis, different geologists may choose different minerals and instruments, so no single protocol can serve as the reference answer. Second, open-ended experimental plans resist representation as structured labels, which makes objective scoring difficult. To address these challenges, we introduce MineralExp, a benchmark that reformulates geochemical experimental
    
[^69]: 评估大型语言模型在科学论文搜索中的应用：Mac Studio与NVIDIA A100上的检索成功率与推理延迟

    Evaluating Large Language Models for Scientific Paper Search: Retrieval Success and Inference Latency on Mac Studio and NVIDIA A100

    [https://arxiv.org/abs/2611.17929](https://arxiv.org/abs/2611.17929)

    本研究在Mac Studio与NVIDIA A100上评估了八个开放权重大语言模型执行科学论文网络辅助检索的表现，发现检索成功率主要取决于模型选择与查询的具体性，A100推理速度更快，但Mac Studio的统一内存可支持更大的模型。

    

    科学文献检索可以从本地部署的大语言模型（LLM）助手中受益，这类助手减少了对托管模型提供商的依赖，同时让研究团队对自身数据和基础设施拥有更大的控制权。本研究探讨了开放权重模型能否在实际计算约束下支持网络辅助的论文检索。我们首先在Mac Studio和配备两块NVIDIA A100 GPU的服务器上，表征了输入上下文从512个token增长到128k个token过程中的推理延迟。随后，我们使用代表三个不同领域知识水平的查询，评估了八个模型对来自七个科学存储库的70篇论文的检索能力。结果表明，检索成功率在很大程度上取决于所选模型和查询的具体程度。此外，A100 GPU通常比Mac Studio更快，但Mac Studio的统一内存允许加载在所测试的GPU配置中无法装载的更大模型。

    arXiv:2611.17929v1 Announce Type: new  Abstract: Scientific literature search can benefit from locally deployed LLM assistants, which reduce reliance on hosted model providers while giving research teams greater control over their data and infrastructure. This work investigates whether open-weight models can support web-assisted paper retrieval under practical computing constraints. We first characterize inference latency as the input context grows from 512 to 128k tokens on a Mac Studio and a server equipped with two NVIDIA A100 GPUs. We then evaluate eight models on the retrieval of 70 papers from seven scientific repositories, using queries that represent three levels of domain knowledge. The results show that retrieval success depends strongly on the selected model and the specificity of the query. And that A100 GPUs are generally faster than the Mac Studio, but its unified memory allows larger models that could not be loaded in the tested GPU configuration.
    
[^70]: 测试时自适应下神经信号解码中平衡准确率无法识别的问题

    What Balanced Accuracy Cannot Identify in Neural-Signal Decoding under Test-Time Adaptation

    [https://arxiv.org/abs/2611.17925](https://arxiv.org/abs/2611.17925)

    本文揭示在测试时自适应的神经信号解码评估中，平衡准确率等总体指标存在盲区——仅报告单一类别的解码器可获得与正常50%平衡解码器相同的分数，且占样本约十分之一的类别在112个受试者单元中有26-47个完全无法被预测。

    

    神经信号分类的适配通常是直推式的：每个批次在完成更新后才进行评分，其增益依赖于适配后的模型。本文探讨的是，总体指标的变化无法刻画队列内部哪些问题。全文测试时自适应（full_ttt）在11个数据集-编码器组合单元上进行了评估（Siena/EEG和PTB-XL/ECG各4个单元，Sleep-EDF/EEG 3个单元），数据划分与源模型评估保持固定；以受试者为分析单元，精确符号检验经Holm校正并以共享数据流为条件，同时通过自助法区间进行了依赖性审计。研究得出两类不同性质的结果。在二分类情形下，平衡准确率会给只报告单一类别的解码器与在50%水平上报告两个类别的解码器相同的分值，因此总体指标无法识别出不再被报告的类别；一个占受试者样本约十分之一的类别，在112个受试者单元中有26-47个出现完全无法预测的情况，而对照组仅为7-9个。

    arXiv:2611.17925v1 Announce Type: new  Abstract: Adaptation for neural-signal classification is usually transductive: each batch is scored after updating on it, and the gain depends on the adapted model [4]. This paper asks what an aggregate change fails to characterise inside the cohort. Full test-time adaptation (full_ttt) was evaluated in 11 dataset-encoder cells (four each on Siena/EEG and PTB-XL/ECG, three on Sleep-EDF/EEG), with splits and source-model evaluation fixed; the subject was the unit, exact sign tests were Holm-corrected and conditional on the shared stream, and bootstrap intervals accompanied a dependence audit. Two results of different kinds follow. In the binary case balanced accuracy gives a one-class decoder the same value as one reporting both classes at 50%, so the aggregate cannot identify a class that stopped being reported; a class covering a tenth of a subject's samples went unpredicted in 26-47 of 112 subject-cells, against 7-9 for the controls, which is a 
    
[^71]: RecMem：面向长序列生成式推荐的循环记忆压缩

    RecMem: Recurrent Memory Compression for Long-Sequence Generative Recommendation

    [https://arxiv.org/abs/2611.17921](https://arxiv.org/abs/2611.17921)

    RecMem提出一种循环记忆压缩框架，将长交互历史压缩为固定数量的记忆槽，使内存与计算开销不随历史长度增长，同时通过推荐监督与流一致性训练保证预测信息保真与不同处理顺序下的一致性。

    

    生成式推荐受益于长交互历史，但直接处理这些历史的成本会随历史长度的增长而增加。我们提出RecMem，一个循环记忆压缩框架，其设计遵循三个目标：空间不变性、语义保真度和终身可演化性。RecMem采用分叉式编码器-解码器（共享底层）以及逐记忆槽的遗忘门控更新，将历史片段压缩到一组固定的记忆槽中。推荐监督训练使记忆保留具有预测价值的信息，而流一致性训练则对齐顺序处理与打包处理两种方式下的预测结果。通过构造，记忆状态和解码器上下文与历史长度无关；在收缩性和近似半群一致性条件下，我们进一步推导出这两种处理顺序之间状态差异的与时间视界无关的界限。在MerRec数据集上，RecMem仅使用16个记忆槽和216个token的解码器上下文……

    arXiv:2611.17921v1 Announce Type: new  Abstract: Generative recommendation benefits from long interaction histories, but the cost of processing them directly grows with history length. We propose RecMem, a recurrent memory compression framework guided by three desiderata: space invariance, semantic fidelity, and lifelong evolvability. RecMem compresses historical segments into a fixed set of memory slots using a bifurcated encoder-decoder with shared lower layers and slot-wise forget-gated updates. Recommendation supervision trains the memory to preserve predictive information, while stream-consistency training aligns predictions from sequential and packed processing. By construction, the memory state and decoder context are independent of history length; under contraction and approximate semigroup consistency, we further derive a horizon-independent bound on the state discrepancy between these two processing orders. On MerRec, RecMem uses 16 memory slots and a 216-token decoder contex
    
[^72]: 安全信息凝聚定律

    Condensing Law of Security Information

    [https://arxiv.org/abs/2611.17918](https://arxiv.org/abs/2611.17918)

    本文指出缩放定律在安全领域失效——模型越强越会在下游安全模块察觉前“悄悄修复”异常输入，并首次将其形式化为认知能力与安全信息保持率之间的权衡，推导出三个信息论界。

    

    缩放定律表明，在数据充足的情况下，大语言模型（LLM）的泛化误差随模型规模单调递减，“更大模型优于更小模型”这一推论现已被广泛接受。我们论证这一推论在安全领域并不成立：安全决策依赖于异常输入，而异常输入呈长尾分布（例如零日攻击），因此似然拟合难以有效处理此类输入。更糟糕的是，规模所带来的能力（例如从受损的部分推断出原本完整的整体）会使LLM在任何下游安全模块察觉之前就悄悄地“修复”异常。我们将这一现象形式化为两个量之间的权衡：模型对世界分布的认知能力 K(W)=I(W;P)，以及在其中间表示中得以保留的安全信息保持率 R(W)=I(Z;S)/I(A;S)。我们推导了三个信息论界来连接……

    arXiv:2611.17918v1 Announce Type: cross  Abstract: Scaling laws indicate that the generalization error of a large language model (LLM) decreases monotonically with model size once data is sufficient, and the corollary that larger models outperform smaller ones is now widely accepted. We argue that this corollary fails in security: decisions there hinge on anomalous inputs, which are long-tailed (e.g., zero-day attacks) and therefore poorly served by likelihood fitting. Worse still, the capability brought about by scale (e.g., inferring the intended whole from a corrupted part) leads a LLM silently \emph{repair} the anomaly before any downstream security head ever sees it. We formalize this as a tradeoff between two quantities: the cognitive capability $K(W)=I(W;P)$ that a model holds about the world distribution, and the security-information retention $R(W)=\frac{I(Z;S)}{I(A;S)}$ that survives in its intermediate representation. We derive three information-theoretic bounds to connect t
    
[^73]: 当合作必须改变时：多智能体强化学习中的语义策略遗忘

    When Cooperation Must Change: Semantic Strategy Unlearning in Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.17917](https://arxiv.org/abs/2611.17917)

    本文提出了多智能体强化学习中的“语义策略遗忘”新问题：当部署需求变化时，需在保留有用合作的前提下，移除可由不同智能体以角色不变方式重现的特定协调约定。

    

    合作式多智能体强化学习（MARL）通过学习到的协调约定，支持分布式Web服务、多机器人控制和协作游戏中的协同决策。当部署需求发生变化时，先前有用的约定可能变得过时或不安全。例如，Web内容交付中修订的缓存共享权限可能要求缓存放弃已学习的存储与服务角色分工，同时保持高效的交付。因此，团队必须在保留有用合作的同时移除指定的约定。这具有挑战性，因为相同的关系模式可以由不同智能体在不同历史中重现。运行时屏蔽会保留底层的协调偏好并可能损害任务进度，而现有的强化遗忘和轨迹删除方法并未直接针对角色不变的约定。本文形式化了（注：摘要原文在此处截断，不完整）

    arXiv:2611.17917v1 Announce Type: new  Abstract: Cooperative multi-agent reinforcement learning (MARL) supports coordinated decision-making in distributed Web services, multi-robot control, and collaborative games through learned coordination conventions. When deployment requirements change, a previously useful convention may become obsolete or unsafe. For example, revised cache-sharing permissions in Web content delivery may require caches to abandon a learned division of storage and serving roles while maintaining efficient delivery. Teams must therefore remove the specified convention while preserving useful cooperation. This is challenging because the same relational pattern can be reproduced by different agents across different histories. Runtime shielding leaves the underlying coordination preference intact and can impair task progress, while existing reinforcement unlearning and trajectory-deletion methods do not directly target role-invariant conventions. This paper formalizes 
    
[^74]: SWIFT：面向预测任务的时间序列基础模型样本级实例微调

    SWIFT: Sample-Wise Instance Fine-Tuning of Time-Series Foundation Models for Forecasting

    [https://arxiv.org/abs/2611.17915](https://arxiv.org/abs/2611.17915)

    SWIFT 提出了一种样本级实例微调框架，利用目标时间序列自身历史中的结构先验，无需额外数据集即可对时间序列基础模型进行对用户透明的细粒度微调，显著降低了模型适配的应用门槛。

    

    当前针对时间序列预测基础模型的少样本适配策略主要以数据集为中心，而微调过程需要获取额外数据集这一要求给模型的推广应用带来了重大障碍。为解决这一限制，我们提出了样本级实例微调（SWIFT），这是一种新颖的预测适配框架，通过利用目标实例自身历史中的结构先验来实现细粒度的专门化。这一转变使得微调过程对最终用户完全透明，显著提升了模型应用的便捷性。为实现该框架，SWIFT 协同运用自检索进行历史挖掘、频率感知损失保证频谱一致性，并采用偏差感知动态融合来平衡适配能力与通用知识的保留。在四个时间序列基础模型骨干和三个长历史基准子集上进行的广泛实验验证了该方法的有效性。

    arXiv:2611.17915v1 Announce Type: new  Abstract: While current few-shot adaptation strategies for time series forecasting foundation models are predominantly dataset-centric, the necessity of obtaining an additional dataset for the finetuning process poses a significant hurdle for adoption. To address this constraint, we propose Sample-Wise Instance Fine-Tuning (SWIFT), a novel forecasting adaptation framework that achieves fine-grained specialization by leveraging structural priors from the target instance's own history. This shift allows the fine-tuning process to be entirely transparent to the end-user, which significantly enhances the ease of adoption. To implement this framework, SWIFT synergizes Self-Retrieval for historical mining and Frequency-Aware Loss for spectral consistency, while employing Deviation-Aware Dynamic Fusion to balance adaptation with general knowledge preservation. Extensive experiments across four TSFM backbones and three long-history benchmark subsets valid
    
[^75]: 对抗性攻击LLM规划器中的符号词汇漏洞

    Adversarially Attacking Symbolic Vocabulary Vulnerabilities In LLM Planners

    [https://arxiv.org/abs/2611.17910](https://arxiv.org/abs/2611.17910)

    本文提出了一种名为Symbolic Swapper的基于LLM的对抗攻击模型，通过自动生成符号词汇被混淆的规划任务，揭示了LLM智能体在符号解释与规划结合方面仍然存在的关键弱点。

    

    随着大型语言模型（LLM）日益被部署为自主智能体，它们在处理自动规划任务方面的能力变得至关重要。然而，尽管具有诸多优势，LLM的规划能力并不稳健，特别是当它们不被允许利用用于定义规划任务的符号词汇（例如动作、谓词和对象的名称）中的常识线索时。这使得LLM与符号规划器相比显得不可靠。鉴于近期的推理模型在符号词汇被混淆的任务上表现出更好的规划性能，我们利用对抗性攻击框架来揭示LLM智能体在联合符号解释与规划方面的关键弱点在多大程度上仍然存在，以及在面对严重情况时它们能否正确推理。因此，我们的主要贡献是设计了一个基于LLM的攻击模型，我们称之为Symbolic Swapper（符号交换器），用于自动生成混淆的规划任务...

    arXiv:2611.17910v1 Announce Type: new  Abstract: As large language models (LLMs) are increasingly deployed as autonomous agents, their abilities in navigating automated planning tasks have become essential. However, despite their strengths, LLMs' planning ability is not robust, particularly when they are not allowed to exploit commonsense cues in the symbolic vocabulary (e.g., names of actions, predicates, and objects) used to define planning tasks. This makes LLMs unreliable compared to symbolic planners. With recent reasoning models showing improved planning performance on tasks with obfuscated symbolic vocabularies, we utilize an adversarial attack framework to reveal to what extent the crucial weaknesses of LLM agents at joint symbolic interpretation and planning remain and whether they can reason properly when presented with severe cases. As such, our main contribution is devising an LLM-based attack model, which we call Symbolic Swapper, to automatically generate obfuscated plann
    
[^76]: 更多基因，更快推理：基于组织学图像的高效大规模基因面板空间转录组预测

    More Genes, Faster Inference: Efficient Large-Panel Spatial Transcriptomics Prediction from Histology

    [https://arxiv.org/abs/2611.17899](https://arxiv.org/abs/2611.17899)

    本文提出GeneFlow框架，将流匹配生成建模与结构化多模态融合相结合，直接从组织学图像预测空间基因表达，突破了以往方法局限于固定小基因面板、架构复杂的瓶颈，实现了高效的大规模基因面板空间转录组预测。

    

    空间转录组学（ST）预测旨在直接从组织图像推断空间分辨的基因表达谱。这是一个具有挑战性的问题，目前通常需要借助复杂的模型架构、额外的输入和繁琐的训练流程来解决，且往往局限于固定的小规模基因面板。这引出了一个自然的问题：是否存在一种高效的大规模基因面板ST预测方法？在此，我们提出了GeneFlow，一个更简单却高效的大规模空间基因表达预测框架，它将流匹配生成建模与结构化多模态融合统一起来。具体而言，GeneFlow首先通过组织学分词器和基因分词器，在空间位点上对组织学形态和跨基因相互作用进行编码。该模型直接预测空间基因表达，而非中间速度场，并采用基于速度的流匹配目标进行训练。

    arXiv:2611.17899v1 Announce Type: cross  Abstract: Spatial transcriptomics (ST) prediction from histology aims to infer spatially resolved gene expression profiles directly from tissue images. This is a challenging problem that is currently often addressed with complicated model architectures, additional inputs, and complicated training procedures, while remaining limited to fixed, small gene panels. This raises a natural question: Is there an efficient method for large-panel ST prediction? Here, we present GeneFlow, a simpler yet effective framework for large-scale spatial gene expression prediction that unifies flow-matching generative modeling with structured multimodal fusion. Specifically, GeneFlow first encodes both histological morphology and cross-gene interactions in spatial spots through histology and gene tokenizers. The model directly predicts spatial gene expression, rather than an intermediate velocity field, while being trained with a velocity-based flow-matching objecti
    
[^77]: 从设计层面保障多智能体系统的安全

    Securing Multi-Agent Systems by Design

    [https://arxiv.org/abs/2611.17883](https://arxiv.org/abs/2611.17883)

    本文提出了响应操纵攻击以揭示多智能体系统面对恶意响应的安全隐患，并进一步提出基于响应间语义相似性的安全聚合规则SecMAS来有效防御此类攻击。

    

    多智能体系统通过将复杂任务分解为子任务、将子任务委托给各个智能体，并使用聚合规则汇总各智能体的响应，从而解决复杂任务。恶意智能体可以通过对其被分配的子任务提供恶意响应来操纵最终解决方案。在本文中，我们首先提出了响应操纵攻击，它通过迭代优化恶意响应来最大化攻击效果。具体而言，我们将无目标攻击和有目标攻击形式化为优化问题，并提出一种近似求解这些问题的方法，即基于替代多智能体系统的反馈，迭代地改进恶意响应。为防御此类攻击，我们进一步提出了SecMAS，一种安全聚合规则，它利用各智能体响应之间的语义相似性，选出与足够大的其他响应子集最为一致的响应。（摘要在此处截断）

    arXiv:2611.17883v1 Announce Type: cross  Abstract: Multi-agent systems solve complex tasks by decomposing them into subtasks, delegating the subtasks to agents, and aggregating the agents' responses using an aggregation rule. Malicious agents can manipulate the final solution by providing malicious responses to their assigned subtasks. In this paper, we first propose response manipulation attacks (ReMA), which iteratively optimize malicious responses to maximize attack effectiveness. Specifically, we formulate untargeted and targeted attacks as optimization problems and develop an approach to approximately solve these problems by iteratively refining malicious responses based on feedback from a surrogate multi-agent system. To defend against such attacks, we further propose SecMAS, a secure aggregation rule that leverages the semantic similarities among agent responses to select the response that is most consistent with a sufficiently large subset of the other responses. We theoretical
    
[^78]: 基于摄影测量三维模型的动物行为分类

    Photogrammetric 3D Model-Based Animal Behavior Classification

    [https://arxiv.org/abs/2611.17881](https://arxiv.org/abs/2611.17881)

    本文提出一种两阶段动物行为分类方法，先通过摄影测量重建三维身体与物体几何结构再进行行为分类，从而使二维到三维逆问题的模糊性显式化且可被纠正。

    

    基于视频的行为估计通常直接从图像序列端到端实现。我们提倡一种两阶段的替代方案：在分类行为之前，先重建三维身体和物体的几何结构。这使得病态的二维到三维逆问题的模糊性显式化，并有可能得到纠正。我们通过模拟的相互作用小鼠演示了该策略，并将其扩展到真实视频。

    arXiv:2611.17881v1 Announce Type: cross  Abstract: Video-based behavior estimation is often implemented end-to-end from image sequences. We advocate a two-stage alternative: reconstruct 3D body and object geometry before classifying behavior. This makes ambiguities of the ill-posed 2D-to-3D inverse problem explicit and potentially correctable. We demonstrate the strategy with simulated interacting mice and extend it to real video.
    
[^79]: Soteria：面向安全 LLM 智能体的形式化验证规划与运行时强制执行

    Soteria: Formally Verified Planning with Runtime Enforcement for Safe LLM Agents

    [https://arxiv.org/abs/2611.17880](https://arxiv.org/abs/2611.17880)

    Soteria 通过在工具调用前对分层计划进行形式化验证并在运行时强制执行规范，使 LLM 智能体在不牺牲任务效用的情况下保证安全性，并在不安全动作被阻止时借助已验证计划提供有原则的恢复指导。

    

    执行多步骤工具调用的 LLM 智能体必须同时满足两个目标：完成用户的任务（效用）和遵守领域策略与正确性约束（安全）。这两个目标之间存在张力——阻止一个不安全的动作虽然可以防止违规，但可能使智能体陷入困境且缺乏有原则的恢复途径。现有方法往往牺牲其中一个目标来换取另一个。我们提出 Soteria，一个通过经过验证的分层规划和规范的运行时强制执行来调和安全与效用的框架。在执行之前，智能体生成一个结构化计划，并在任何工具调用之前针对规范对其进行形式化验证。在执行期间，经过验证的计划发挥双重作用：它确保智能体采取符合规范的轨迹，并在不安全动作被阻止时提供恢复指导。在涵盖多种工具使用任务的多个基准测试中……（原文在此处截断）

    arXiv:2611.17880v1 Announce Type: new  Abstract: LLM agents that execute multi-step tool calls must satisfy two objectives simultaneously: completing the user's task (utility) and conforming to domain policies and correctness constraints (safety). These objectives are in tension -- blocking an unsafe action prevents a violation but can leave the agent stranded with no principled way to recover. Existing approaches sacrifice one objective for the other. We introduce Soteria, a framework that reconciles safety and utility through verified hierarchical planning and runtime enforcement of specifications. Before execution, the agent generates a structured plan that is formally verified against the specifications prior to any tool invocation. During execution, the verified plan serves dual roles: it ensures that the agent takes specification-conformant trajectories and provides guidance when unsafe actions are blocked. Across multiple benchmarks covering a diverse range of tool-use tasks, So
    
[^80]: 从局部预测误差到最终输出：多步推理中的专家共享理论

    From Local Prediction Error to Final Outputs: A Theory of Expert Sharing in Multi-Step Reasoning

    [https://arxiv.org/abs/2611.17879](https://arxiv.org/abs/2611.17879)

    该论文建立了多步推理中专家共享的理论框架，通过权衡方差节省与偏差推导出共享何时有效的精确准则，并揭示了一个关键现象：共享专家可能降低局部预测误差却反而增大最终输出误差。

    

    混合专家模型可以在层间或计算步骤间共享专家，汇集来自不同上下文的训练信息。这可以降低局部预测误差，但多步系统最终是以其最终输出来评估的。我们研究了这些局部收益何时能在复合过程中得以保留。在指定的线性回归模型中，我们通过权衡方差节省与跨不同目标行为的角色共享所带来的偏差，推导出精确的共享与合并准则。在共同固定线性传播映射下，我们刻画了期望平方最终输出风险，并保留了误差方向和相关性。我们构造了共享降低平均局部误差却增加最终输出误差的案例。在此设定下，我们还识别了带各向同性噪声的正交完全池化在何种条件下能在最终输出处保留所有局部改进。对于一般的路由计算，我们推导出条件端到端误差界。

    arXiv:2611.17879v1 Announce Type: cross  Abstract: Mixture-of-experts (MoE) models can share experts across layers or computational steps, pooling training information from different contexts. This can reduce local prediction error, but multi-step systems are ultimately evaluated by their final outputs. We study when these local gains survive composition. In specified linear regression models, we derive exact sharing and merging criteria by balancing variance savings against bias from sharing across roles with different target behavior. Under a common fixed linear propagation map, we characterize expected squared final-output risk, retaining error directions and correlations. We construct cases where sharing lowers average local error yet increases final-output error. Within this setting, we also identify when orthogonal full pooling with isotropic noise preserves every local improvement at the final output. For general routed computations, we derive conditional end-to-end error bounds
    
[^81]: DySENSE：基于卫星影像的动态时空能源合成与仿真

    DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery

    [https://arxiv.org/abs/2611.17878](https://arxiv.org/abs/2611.17878)

    DySENSE提出了一个联合多模态生成式AI框架，通过解耦长期静态城市形态与短期动态能耗，基于卫星影像与土地利用条件生成空间对齐的月度建筑能耗地图，填补了现有研究只能预测已有城市而无法为未来城市生成动态能源数据的空白。

    

    城市消耗了全球超过三分之二的能源，并产生了超过70%的全球二氧化碳排放。城市能源对于实现2050年净零目标以及联合国可持续发展目标7和11至关重要。尽管现有研究已取得进展，但它们受限于数据和方法上的空白。现有研究主要针对已有城市进行预测，而非对未来城市进行生成；细粒度的、空间对齐的月度动态能源数据十分稀缺；且现有生成式AI模型难以生成动态的城市功能月度能源。我们提出了DySENSE，一个联合多模态生成式AI框架，它将静态城市卫星影像与空间对齐的动态月度建筑能耗地图进行合成。我们将长期静态城市形态与短期动态能源消耗解耦。通过以静态城市土地利用为条件，DySENSE可以生成逼真、多样且全新的静态（摘要在此处截断）……

    arXiv:2611.17878v1 Announce Type: cross  Abstract: Cities account for over two-thirds of the world's energy and more than 70% of global carbon dioxide emissions. City energy is essential for achieving the 2050 net-zero target and the UN's Sustainable Development Goals 7 and 11. Although existing studies have made progress, they are limited by data and methodological gaps. Existing studies are largely predictive for existing cities rather than generative for future cities; fine-grained, aligned monthly dynamic energy data are scarce; and existing GenAI models struggle to generate dynamic urban functional monthly energy. We present DySENSE, a joint multimodal GenAI framework that synthesizes static urban satellite imagery together with spatially aligned, dynamic monthly building energy consumption maps. We decouple long-term static urban morphology from short-term dynamic energy consumption. By conditioning on static urban land use, DySENSE can generate realistic, diverse and new static 
    
[^82]: OmniScore-Bench：评估视觉语言模型在乐谱阅读上的能力

    OmniScore-Bench: Evaluating Vision-Language Models on Musical Score Reading

    [https://arxiv.org/abs/2611.17870](https://arxiv.org/abs/2611.17870)

    OmniScore-Bench是一个包含3000个问题、覆盖六大方面和24种任务子类型的综合基准，用于系统评估视觉语言模型在乐谱阅读上的光学音乐识别能力，揭示了各模型从符号定位到证据关联问答等任务中的优势与瓶颈。

    

    视觉语言模型（VLM）能够回答关于乐谱的问题，但它们识别记谱符号、恢复乐谱结构以及解读其音乐含义的能力需要单独评估。现有的光学音乐识别（OMR）资源已建立了检测和转录的度量标准，但仅凭转录分数无法确定VLM乐谱阅读器的优势与瓶颈所在。我们提出了OmniScore-Bench，通过涵盖六个方面和24种任务子类型的3000个问题来评估VLM中的OMR能力。每个问题都指定了可见范围、参考标注和答案模式，从而实现从符号定位、音乐一致性到基于证据的问答的自动化评估。我们对十二个VLM的评估揭示了针对不同操作类型的能力画像，以及在选段、单页和多页等不同转录任务上表现各异的领先模型。控制变量分析还考察了乐谱（摘要在此处截断）

    arXiv:2611.17870v1 Announce Type: new  Abstract: Vision-language models (VLMs) can answer questions about musical scores, but their ability to recognise notation, recover its structure and interpret its musical meaning requires separate evaluation. Existing optical music recognition (OMR) resources establish detection and transcription measurements; a transcription score alone leaves the strengths and bottlenecks of a VLM score reader unresolved. We introduce OmniScore-Bench, an evaluation of OMR capabilities in VLMs through 3,000 questions spanning six aspects and 24 task subtypes. Each question specifies a visible scope, reference annotations and an answer schema, enabling automatic evaluation from symbol localisation to musical consistency and evidence-linked question answering. Our assessment of twelve VLMs reveals operation-specific capability profiles and different transcription leaders across excerpt, single-page and multipage tasks. Controlled analyses examine the roles of scor
    
[^83]: UltraBERT-JEV：面向“系统一”应用的编码器模型

    UltraBERT-JEV: Encoder models for System One Applications

    [https://arxiv.org/abs/2611.17867](https://arxiv.org/abs/2611.17867)

    提出UltraBERT-JEV编码器模型，通过单次前向传播直接从多个选项中选择答案而无需生成文本，配合“是-否”差值打分等微调方案，准确率最高达72.1%并超越同数据训练的0.6B解码器模型，同时推理速度快一倍以上（29毫秒对比68毫秒）。

    

    我们研究了一种编码器模型，它可以在单次前向传播中从多个选项中直接挑选一个，而无需生成文本。本技术报告比较了微调 UltraBERT-380M（每个选项对应一个掩码）的几种方案。这四种方案的区别在于如何对选项打分以及如何将分数转换为概率，其中一种借鉴了 DeepSeek-V3 的专家门控机制。在与 Qwen3-0.6B 使用相同数据训练的情况下，采用 LLM-as-Jev 方案的解码器测试准确率为 67.6%，而编码器模型达到了 69.7% 至 72.1%。用“是”减“否”（而非仅用“是”）来为选项打分带来了更大的提升：测试集上从 69.7% 提升至 71.3%，外部基准上从 57.8% 提升至 59.6%（解码器为 56.1%）。Sigmoid 归一化额外带来 0.8 个测试点的提升，而平方根 softplus 则没有带来提升。在 NVIDIA A100 上单次处理一个请求时，编码器每次决策仅需 29 毫秒（解码器为 68 毫秒），但峰值内存占用更高（2.8 GiB 对比 2.2 GiB）。一个在人工标注证据和标签上训练的辅助头……（摘要原文在此处截断）

    arXiv:2611.17867v1 Announce Type: cross  Abstract: We study encoder models that pick one of several options in a single forward pass, without generating text. This technical report compares recipes for fine-tuning UltraBERT-380M with one mask per option. The four recipes differ in how an option is scored and how the scores become probabilities; one follows the expert gate of DeepSeek-V3. Trained on the same data as Qwen3-0.6B with the LLM-as-Jev recipe (67.6% test accuracy), the encoder reaches 69.7% to 72.1%. Scoring an option by "yes" minus "no" instead of "yes" alone gives the larger gain, from 69.7% to 71.3% on the test set and from 57.8% to 59.6% on external benchmarks (56.1% for the decoder). Sigmoid normalisation adds 0.8 test points; a square-root softplus adds none. One request at a time on an NVIDIA A100, the encoder takes 29 ms per decision against 68 ms, but uses more peak memory (2.8 against 2.2 GiB). An auxiliary head, trained on human-marked evidence and on labels select
    
[^84]: 通过区分性查询主动学习时序规约

    Active Learning of Temporal Specifications through Distinguishing Queries

    [https://arxiv.org/abs/2611.17866](https://arxiv.org/abs/2611.17866)

    提出了一种LTLf时序规约的主动学习框架，教师只需标注轨迹并用自然语言回答问题，学习者通过维护枚举版本空间并生成区分性查询来高效学习目标规约，避免了传统方法中等价查询、反例构造和搜索重启的开销。

    

    时序规约对于验证和监控至关重要，然而编写能够刻画预期行为的公式却十分困难。主动学习通过学习者与教师之间的交互来推导此类规约，但在时序逻辑领域仍未得到充分探索。现有方法要求教师回答等价查询并构造反例，且往往依赖基于SAT的学习过程，每遇到新示例就需要重启搜索。我们提出了一种针对LTLf（有限时域线性时序逻辑）的主动学习框架，其中教师仅需对轨迹进行标注，并用自然语言回答关于预期行为的问题。学习者维护一个枚举的候选公式版本空间，生成能够区分不同候选公式的轨迹，并利用教师提供的提示来对候选公式排序或限制搜索范围。我们证明了该框架与观测示例的一致性，以及在有界假设类内的可识别性（摘要在此处被截断）。

    arXiv:2611.17866v1 Announce Type: cross  Abstract: Temporal specifications are essential for verification and monitoring, yet writing formulas that capture the intended behaviour is difficult. Active learning derives such specifications through interaction between a learner and a teacher, but remains underexplored for temporal logics. Existing approaches require the teacher to answer equivalence queries and to construct counterexamples, and they often rely on SAT-based learning procedures that restart the search after each new example. We present an active-learning framework for LTLf in which the teacher only labels traces and answers natural-language questions about the intended behaviour. The learner maintains an enumerated version space of candidate formulas, generates traces that distinguish candidates, and uses the teacher's hints to rank candidates or to restrict the search. We prove consistency with the observed examples, identification within bounded hypothesis classes with a b
    
[^85]: FlexCast：面向时间序列基础模型的灵活上下文学习

    FlexCast: Flexible Context Learning for Time Series Foundation Models

    [https://arxiv.org/abs/2611.17862](https://arxiv.org/abs/2611.17862)

    FlexCast通过对时间序列补丁进行可变长度掩码预测的预训练框架，解耦了上下文条件化与自回归这两个因素，使同一个预训练模型能够灵活支持单步预测、置信度迭代细化和自回归式生成等多种推理方式。

    

    基于自回归（AR）下一词元预测的时间序列基础模型（TSFM）将两个因素纠缠在一起：可变上下文条件化与带有自回归展开的预测单元分解。因此，强大的AR性能无法揭示泛化能力究竟源于灵活的训练分布、局部预测目标、从左到右的推理过程，还是它们的组合。我们提出FlexCast，一种能够解耦这些因素的灵活上下文-目标学习框架。FlexCast将预测重新表述为对时间序列补丁的可变长度掩码预测：在预训练期间，它动态采样可见的上下文补丁和被掩码的目标补丁，使模型在无需顺序解码的情况下接触多样的上下文-目标配置。在推理时，同一个预训练模型通过统一的掩码预测机制，支持单步预测、基于置信度的迭代细化以及下一词元式的生成。

    arXiv:2611.17862v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) based on autoregressive (AR) next-token prediction entangle two factors: variable-context conditioning and prediction-unit factorization with autoregressive rollout. Strong AR performance therefore does not reveal whether generalization stems from the flexible training distribution, the local prediction target, the left-to-right inference process, or their combination. We propose FlexCast, a flexible context-target learning framework that disentangles these factors. FlexCast reformulates forecasting as variable-length masked prediction over time series patches: during pretraining, it dynamically samples visible context and masked target patches, exposing the model to diverse context-target configurations without sequential decoding. At inference time, the same pretrained model supports one-step forecasting, confidence-based iterative refinement, and next-token-style generation through a unified mask-
    
[^86]: 教育中的人工智能辅助决策——不确定性、可解释性与人类责任

    AI-aided Decision-Making in Education - Uncertainty, Explainability, and Human Responsibility

    [https://arxiv.org/abs/2611.17858](https://arxiv.org/abs/2611.17858)

    本文提出不确定性、可解释性与人类责任三大原则是人工智能在教育领域合理应用的基石，强调需正视AI决策的固有不确定性，并通过可解释性与人类监督保障其可信度。

    

    随着基于人工智能（AI）的系统日益融入日常生活（包括教育领域）的各个方面，理解这些系统如何做出决策变得至关重要。人工智能在促进个性化与自适应学习、自动化评估流程以及协助行政任务方面具有巨大潜力，能够显著提升教育实践。然而，教育中基于人工智能的决策（尤其是深度学习和生成式AI模型）的不透明性可能阻碍决策过程，特别是当结果显得缺乏依据或不正确时。在本章中，我们认为三个关键原则——不确定性、可解释性和人类责任——对于人工智能在教育场景中的合理合法使用至关重要。承认基于人工智能的决策中固有的不确定性（类似于人类决策中的不确定性）是根本前提。可解释性对于……（原文摘要在此处被截断）

    arXiv:2611.17858v1 Announce Type: new  Abstract: As Artificial Intelligence (AI)-based systems have become increasingly integrated into various aspects of everyday life, including education, understanding how these systems make decisions is of paramount importance. AI holds significant potential to enhance educational practices through the facilitation of personalized and adaptive learning, automation of assessment procedures, and assistance with administrative tasks. However, the opacity of AI-based decision-making in education (especially with deep learning and generative AI models) can hamper decision-making, especially when outcomes appear unjustified or incorrect. In this chapter, we argue that three key principles (uncertainty, explainability, and human responsibility) are essential for the legitimate use of AI in educational settings. Acknowledging the inherent uncertainties in AI-based decisions, analogous to those in human decision-making, is fundamental. Explainability is cri
    
[^87]: DECODE：诊断自动化解释过程中的可解释性失效

    DECODE: Diagnosing Interpretability Failures in Automated Interpretation Process

    [https://arxiv.org/abs/2611.17856](https://arxiv.org/abs/2611.17856)

    提出DECODE可视分析系统，通过将诊断流程分解为三个渐进阶段，重新引入人类监督，帮助隔离和定位自动化解释过程中可解释性失效的来源。

    

    用于解释大语言模型（LLM）内部表示的工具越来越多地依赖大语言模型本身来生成和评估解释，从而形成了一个几乎无需人工监督的自指式自动化解释过程。这种自动化引发了显著的可靠性问题，而由于缺乏真值标准，这些问题被进一步加剧。在实际应用中，人们很难验证解释的有效性，也难以将可解释性失效归因于其源头——即失效究竟源自工具所提取的表示，还是源自该过程中所使用的大语言模型。为应对这些挑战，我们提出了DECODE，一个将人类监督重新引入自动化解释过程的可视分析系统。我们的方法将诊断工作流分解为三个渐进的阶段，从而隔离出从结构性缺陷到解释层面失配等多种失效来源。受编码者间一致性……

    arXiv:2611.17856v1 Announce Type: new  Abstract: Tools for interpreting the internal representations of Large Language Models (LLMs) increasingly rely on LLMs themselves to both generate and evaluate explanations, creating a self-referential automated interpretation process with minimal human oversight. This automation raises significant reliability concerns, which are exacerbated by the absence of ground truth. In practice, it is difficult to validate explanations or to attribute interpretability failures to their source -- whether they originate from the tool's extracted representations or from the LLM used in the process. To address these challenges, we present DECODE, a visual analytics system that reintroduces human oversight into the automated interpretation process. Our approach decomposes the diagnostic workflow into three progressive stages, isolating failure sources ranging from structural defects to explanation-level misalignments. Drawing inspiration from inter-coder consen
    
[^88]: 将极端厄尔尼诺预测的不确定性追溯到中尺度洋流

    Tracing Uncertainties of Extreme El Ni\~{n}o Prediction to Mesoscale Ocean Currents

    [https://arxiv.org/abs/2611.17854](https://arxiv.org/abs/2611.17854)

    该研究将条件非线性最优扰动（CNOP）方法应用于可微分海洋AI模型，首次揭示了极端厄尔尼诺预测不确定性可追溯至初始海洋状态中的中尺度洋流误差，且这些误差的来源具有强烈的事件依赖性。

    

    厄尔尼诺-南方振荡（ENSO）的可预报性历来被认为与模式初始条件中的大尺度误差有关，例如热含量和表面风应力的误差。本研究证明，这些缓慢演变的误差可能最终源于初始海洋状态中的中尺度误差，而这类误差在当前的模式和海洋观测系统中普遍存在。这一隐藏的ENSO预报不确定性来源——即中尺度洋流误差——通过将条件非线性最优扰动（CNOP）方法应用于可微分海洋AI模型而得到了令人信服的揭示。CNOP所识别出的最具影响力的洋流误差具有强烈的事件依赖性：对于1997/98年厄尔尼诺事件，东太平洋的初始洋流误差通过局部抑制原有的次表层增暖而造成了最大的预报误差；对于2015/16年事件，主导误差则来自西太平洋……（原摘要在此处被截断）

    arXiv:2611.17854v1 Announce Type: cross  Abstract: The predictability of the El Ni\~{n}o-Southern Oscillation (ENSO) has historically been linked to large-scale errors in models' initial conditions such as heat content and surface wind stress. Here we demonstrate that these slowly evolving errors may ultimately originate from mesoscale errors in initial oceanic states that are common in current models and ocean observation systems. This hidden source of ENSO forecast uncertainty, in terms of mesoscale ocean-current errors, is convincingly exposed by the conditional nonlinear optimal perturbation (CNOP) method applied to a differentiable ocean AI model. The CNOP-identified most influential current errors are strongly event-dependent. For the 1997/98 El Ni\~{n}o, the initial current errors in the eastern Pacific results in the largest forecast error through locally suppressing original subsurface warming. For the 2015/16 event, the dominate error originates from the western Pacific, whic
    
[^89]: 消融编辑是否变得更难了？并非普遍如此：基于方向的消融方法与 ARA 在 Heretic 中对 Qwen3.8、Gemma 4、Olmo 3.1 和 Mistral Small 3.2 大语言模型与其前代模型的拒绝移除比较

    Is Abliteration Getting Harder? Not Uniformly: Refusal Removal by Direction-Based Ablation and ARA in Heretic on the Qwen3.8, Gemma 4, Olmo 3.1 and Mistral Small 3.2 LLMs Versus Their Predecessors

    [https://arxiv.org/abs/2611.17846](https://arxiv.org/abs/2611.17846)

    本研究在五组新旧模型对上，使用多种基于方向的消融方法与 ARA 在统一实验条件下进行系统对比，发现拒绝移除的难度并未随着模型版本的更新而普遍增加。

    

    消融编辑（Abliteration）是一种无需微调的权重修改技术，可以从开放权重的大语言模型（LLM）中移除拒绝行为。基于优化的变体——任意秩消融（ARA）的开发者报告称，通过消融拒绝方向来移除拒绝行为的基于方向的消融编辑在新模型上日益失效。本研究在五组发布版本对上发现，拒绝移除并没有普遍变得更难。研究在 Heretic 2.0 中对 Qwen2.5-32B 与 Qwen3.8-27B、Gemma-2-27B 与 Gemma-4-31B、OLMo-2-32B 与 Olmo-3.1-32B、Mistral-Small-2409 与 Mistral-Small-3.2，以及 Qwen2.5-7B 与 Qwen3.5-4B 这五组模型，使用三种或四种基于方向的方法（均值差、MPOA、SOM，以及在小模型对上使用的 J-space）和 ARA，在相同的提示词、名义搜索预算、首 token KL 散度上限和选择规则下进行编辑；当自动 WildGuard 判定器在 972 个保留的有害提示（来自 JailbreakBench、XSTest、StrongREJECT、HarmBench 以及作者编制的恶意提示集）中发现更多被成功响应的情况时，该模型对的移除被视为更难。

    arXiv:2611.17846v1 Announce Type: new  Abstract: Abliteration, a fine-tuning-free weight edit, removes refusal from open-weight large language models (LLMs). The developer of Arbitrary-Rank Ablation (ARA), an optimisation-based variant, reports that direction-based abliteration, which ablates a refusal direction, increasingly fails on newer models. On five release pairs, refusal removal did not become uniformly harder. Qwen2.5-32B and Qwen3.8-27B, Gemma-2-27B and Gemma-4-31B, OLMo-2-32B and Olmo-3.1-32B, Mistral-Small-2409 and Mistral-Small-3.2, and Qwen2.5-7B and Qwen3.5-4B were edited in Heretic 2.0 by three or four direction-based methods (difference-in-means, MPOA, SOM and, on the small pair, J-space) and ARA under the same prompts, nominal search budget, first-token KL divergence cap and selection rule; removal counted as harder when the automatic WildGuard judge found more of 972 held-out harmful prompts (JailbreakBench, XSTest, StrongREJECT, HarmBench and an author-compiled mali
    
[^90]: V-JEPA-LM：能够读取静态文本对齐表征之外动态信息的视频-语言模型

    V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations

    [https://arxiv.org/abs/2611.17831](https://arxiv.org/abs/2611.17831)

    该论文提出V-JEPA-LM，通过定位V-JEPA冻结特征中跨视频共享的动态子空间，并利用像素变换构建反事实监督，使视频语言模型能够真正读取编码器中蕴含的场景动态信息（如速度），而非仅仅检索文本对齐的静态属性。

    

    诸如V-JEPA这样的世界模型视频编码器仅凭视频本身就能学习场景如何变化（方向、速度、顺序、状态）。为了回答问题，它们需要与在字幕和问答对上训练的语言模型相结合。但每条字幕或问题只描述片段的少数几个属性，因此模型学会的是检索这些属性而非真正读取片段内容，编码器所携带的动态信息也就未被利用：在相同的测试项上，线性探针可以从冻结的V-JEPA特征中以87%的准确率读取速度，而视觉语言模型（VLM）的回答仅处于随机水平。我们展示了动态信息在冻结编码器中的所在位置：对于V-JEPA这类预测性编码器，动态信息存在于一个跨视频共享的子空间中；而对于SigLIP这类逐帧、文本对齐的编码器，动态信息则不是其共享组成部分。随后我们构建了反事实监督，使语言模型能够读取这些动态信息：通过像素变换将公开视频转化为答案平衡的变体，其答案由该变换决定，且仅在（摘要在此处截断）

    arXiv:2611.17831v1 Announce Type: cross  Abstract: World-model video encoders such as V-JEPA learn how scenes change (direction, speed, order, state) from video alone. To answer questions, they are joined to a language model trained on captions and question-answer pairs. But each caption or question names only a few attributes of a clip, so the model learns to retrieve those attributes rather than to read the clip, and the dynamics the encoder carries go unused: on the same items, a linear probe reads speed from frozen V-JEPA features at 87% while the VLM answers at chance. We show where dynamics lives in a frozen encoder: in a subspace shared across videos for predictive encoders such as V-JEPA, but not as a shared component of frame-wise, text-aligned encoders such as SigLIP. We then build counterfactual supervision that makes the language model read it: pixel transformations turn public videos into answer-balanced variants whose answers are set by the transformation and differ only 
    
[^91]: SceneAgent：一种面向可移植游戏场景生成的多智能体框架

    SceneAgent: A Multi-Agent Framework for Portable Game Scene Generation

    [https://arxiv.org/abs/2611.17821](https://arxiv.org/abs/2611.17821)

    SceneAgent是一个多智能体框架，通过规范的场景上下文、后端适配器以及从人工批准轨迹中提炼的可复用技能，实现了游戏场景构建过程与经验在不同游戏引擎后端间的跨平台迁移复用。

    

    将场景生成器迁移到新的游戏引擎时，仅仅导出其最终输出是不够的：规划、构建工具以及存储的经验可能都依赖于原始的后端。我们提出了SceneAgent，这是一个用于跨后端复用构建过程及其经验的多智能体框架。一个规范化的SceneContext维护着不断演进的世界状态，而后端适配器则将坐标系约定、资产绑定和导出操作隔离开来。通过结构化的交接机制，新的Builder会话可以查询这一持久状态，而无需继承完整的构建记录。在跨代际的生成过程中，经过人工批准的构建轨迹会被提炼成不含资产ID的布局技能和语义化的内容技能，这些技能可以被检索和重新组合。我们推出了GameSceneBench，包含69个室内和室外场景提示。在统一的资产和评估协议下，SceneAgent与五个基线相比获得了78.2%-89.4%的人工平局调整偏好分数。

    arXiv:2611.17821v1 Announce Type: cross  Abstract: Moving a scene generator to a new game engine requires more than exporting its final output: planning, construction tools, and stored experience may also depend on the original backend. We present SceneAgent, a multi-agent framework for reusing the construction process and its experience across backends. A canonical SceneContext maintains the evolving world, while backend adapters isolate coordinate conventions, asset bindings, and export. Structured handoffs let fresh Builder sessions query this persistent state without inheriting the full construction transcript. Across generations, human-approved traces are distilled into asset-ID-free Layout skills and semantic Content skills that can be retrieved and recombined. We introduce GameSceneBench, comprising 69 indoor and outdoor prompts. Under a common asset and evaluation protocol, SceneAgent obtains human tie-adjusted preference scores of 78.2-89.4% against five baselines. Ablations s
    
[^92]: 对齐AI团队

    Aligning AI Teams

    [https://arxiv.org/abs/2611.17820](https://arxiv.org/abs/2611.17820)

    该研究揭示了单个LLM智能体的对齐训练难以迁移到多智能体团队的机制，发现责任分散是安全差距的主要成因，而相互批评、群聊协调等常见自然干预措施无法可靠地恢复团队的对齐行为。

    

    先前的研究表明，针对单个LLM智能体进行的对齐训练可能无法迁移到多智能体环境中。我们研究了这种失败的底层机制，并探索了在对齐迁移到多智能体LLM团队时保持对齐的干预措施。我们聚焦于两个软件工程任务（脓毒症分诊、新闻推荐器）和十个咨询提案生成任务。我们提出了四项发现：（1）单体与团队之间的安全差距并未随着模型能力的增强而呈现明显趋势。我们发现Anthropic的Mythos Preview在其中一个任务上表现出最大的差距，但在另一个任务上却表现出最小的差距；（2）许多自然的干预措施，例如让智能体相互批评彼此的工作或在群聊中进行协调，并不能可靠地使智能体团队恢复到单体智能体的对齐行为；（3）责任分散在很大程度上解释了安全差距：团队中的智能体不太可能主动检查系统级的安全故障……（摘要至此被截断）

    arXiv:2611.17820v1 Announce Type: new  Abstract: Previous work has shown that alignment training on individual LLM agents can fail to transfer to multi-agent settings. We investigate the underlying mechanisms for this failure, and explore interventions to preserve alignment transfer to multi-agent LLM teams. We focus on two software-engineering tasks (sepsis triage, a news recommender) and ten consulting-proposal generation tasks. We present four findings: (1) The single-vs-team safety gap does not show a clear trend with more capable models. We find that Anthropic's Mythos Preview shows the largest gap on one of our tasks but the smallest on another; (2) Many natural interventions, such as having agents critique each other's work or coordinate in a group chat, do not reliably bring agent teams back to single-agent alignment behavior; (3) diffusion of responsibility substantially accounts for the safety gap: agents in teams are less likely to proactively check for system-level safety f
    
[^93]: 因果约束与不确定性感知的心肺风险预测

    Causally Constrained and Uncertainty-Aware Cardiopulmonary Risk Prediction

    [https://arxiv.org/abs/2611.17819](https://arxiv.org/abs/2611.17819)

    提出因果包络智能体（CE-Agent）框架，通过将流行病学证据转化为显式因果约束并融合CT表型与闭式不确定性传播，实现对肺癌与心血管疾病的联合风险预测，并能快速处理缺失数据。

    

    常规胸部计算机断层扫描（CT）能够捕捉肺部、心脏和血管表型，这些表型共同用于肺癌与心血管疾病的风险分层——这两者是肺癌筛查人群中最主要的死亡原因。然而，临床风险计算器对每种疾病进行孤立评估且忽略CT发现，而无约束的机器学习则会在小型医院队列上过拟合，学习到受治疗和疾病混淆的风险方向，并在患者记录不完整时隐藏不确定性。在此，我们提出了因果包络智能体（CE-Agent），这是一个两阶段框架，在观察到任何结局之前，将经同行评审的流行病学证据编译为风险因素权重的显式约束，同时从队列数据中估计局部基线风险和新加入的CT表型。在部署阶段，闭式不确定性传播能够以每位患者数毫秒的速度处理缺失的临床或CT输入，运行效率高出两个数量级。

    arXiv:2611.17819v1 Announce Type: new  Abstract: Routine chest computed tomography (CT) captures pulmonary, cardiac, and vascular phenotypes that jointly stratify lung cancer and cardiovascular disease, the two leading causes of death in lung-screening populations. Yet clinical risk calculators evaluate each disease in isolation and omit CT findings, whereas unconstrained machine learning overfits small hospital cohorts, learns treatment- and illness-confounded risk directions, and conceals uncertainty when patient records are incomplete. Here we present the Causal Envelope Agent (CE-Agent), a two-stage framework that compiles peer-reviewed epidemiological evidence into explicit constraints on risk-factor weights before any outcome is seen, while estimating local baseline risk and newly added CT phenotypes from cohort data. At deployment, closed-form uncertainty propagation accounts for missing clinical or CT inputs within milliseconds per patient, running over two orders of magnitude 
    
[^94]: G-CAST：面向电商智能体在需求持续演变下的受治理持续自训练

    G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands

    [https://arxiv.org/abs/2611.17814](https://arxiv.org/abs/2611.17814)

    G-CAST提出了一套受治理的持续自训练系统，使电商LLM智能体能够在用户需求漂移和执行框架不断演进的条件下自动持续适应，从而摆脱滞后且高成本的人工周期性再训练。

    

    随着大型语言模型（LLM）的快速发展，越来越多的用户开始借助LLM驱动的智能体进行电商对话式购物。然而，随着新商品品类、采购场景和功能需求的不断涌现，用户需求也在持续变化。与此同时，产品的执行框架（包括其工具、技能和操作流程）也在不断演进，以支持日益复杂的交互。这些变化带来了两个实际挑战：1）需求漂移改变了用户请求的分布、意图和复杂度，使一次性训练的模型逐渐落后；2）框架演进要求模型正确使用新引入的能力，同时不丢失用户已依赖的既有行为。在这种场景下，周期性的人工再训练过于滞后且耗费人力。为应对这些挑战，我们提出了G-CAST，一个受治理的持续自训练系统……（原文摘要到此截断）

    arXiv:2611.17814v1 Announce Type: new  Abstract: With the rapid advancement of Large Language Models (LLMs), an increasing number of users are turning to LLM-powered agents for conversational shopping in e-commerce. However, the user needs continue to change as new product categories, procurement scenarios, and feature requests emerge. At the same time, the product's execution harness-including its tools, skills, and operating procedures-continues to evolve to support increasingly sophisticated interactions. These changes create two practical challenges: 1) Demand drift alters the distribution, intent, and complexity of user requests, causing a model trained once to fall behind; 2) Harness evolution requires the model to use newly introduced capabilities correctly without losing behavior that users already rely on. Periodic manual retraining is too delayed and labor-intensive for this setting. To address these challenges, we present G-CAST, a governed continual self-training system tha
    
[^95]: CarryOn：为长程LLM智能体学习在上下文中保留什么

    CarryOn: Learning What to Keep in Context for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2611.17808](https://arxiv.org/abs/2611.17808)

    CarryOn是一个轻量级上下文管理器，通过学习的选择器和自适应保留策略，将紧凑的执行状态与细粒度的来源链接证据相结合，无需额外LLM调用即可为长程LLM智能体实现可靠的上下文管理。

    

    长程LLM智能体能够协调工具和应用程序以完成复杂的多步骤任务。随着交互历史的增长，有效的上下文管理对于可靠且高效的执行变得至关重要。现有方法通过摘要、剪枝或历史选择来缩减上下文。然而，更短的上下文并不一定支持更好的决策：摘要可能省略后续操作所需的精确细节，而保留完整的交互则会携带大量无关信息。我们提出了CarryOn，一个轻量级的上下文管理器，它将紧凑的执行状态与细粒度的、带有来源链接的证据相结合。一个经过学习的选择器和自适应保留策略能够保持相关证据的可用性，而无需额外的LLM调用来进行上下文管理。我们引入了两个互补的评估指标：$S_\alpha$在考虑资源使用的同时优先考虑任务成功率，而每次成功的归一化成本（NCS）用于衡量……

    arXiv:2611.17808v1 Announce Type: cross  Abstract: Long-horizon LLM agents can coordinate tools and applications to complete complex, multi-step tasks. As their interaction histories grow, effective context management becomes essential for reliable and efficient execution. Existing methods reduce context through summarization, pruning, or history selection. However, shorter contexts do not necessarily support better decisions: summaries can omit exact details needed for later actions, while retaining entire interactions can carry substantial irrelevant information. We propose CarryOn, a lightweight context manager that pairs compact execution state with fine-grained, source-linked evidence. A learned selector and adaptive retention policy keep relevant evidence available without additional LLM calls for context management. We introduce two complementary evaluation metrics: $S_\alpha$ prioritizes task success while accounting for resource use, and normalized cost per success (NCS) measu
    
[^96]: 智能体能跟上时间吗？对话中断后时间适应性调整的实证分析

    Can Agents Keep Up with Time? An Empirical Analysis of Temporal Adaptation After a Conversational Hiatus

    [https://arxiv.org/abs/2611.17806](https://arxiv.org/abs/2611.17806)

    本文提出“时间状态感知与运用”这一新能力概念，通过设计会话重燃任务（固定先前对话、仅变化后续日期），实证考察长时程智能体在对话中断后能否感知时间流逝并据此调整建议，填补了现有评估只关注事实记忆与显式状态更新的空白。

    

    长时程智能体日益在现代网络平台上充当持久化的个人助理，在数周或数月的跨度内管理日程、申请和多步骤计划。用户常常在长时间的对话中断后回到先前的对话中，而此前的预订记录和准备笔记仍保留在案。在这种中断期间，即使保留的记录没有任何变化、事件结果也未被记录，预定的日期也可能已经过去。现有对长时程智能体的评估主要聚焦于事实记忆和显式状态更新，因此智能体是否能感知时间的流逝并相应调整其建议，仍属未知。我们将这种能力定义为时间状态感知与运用，即感知当前时间，并在解读带有日期的记录时加以运用。为研究这一能力，我们设计了“会话重燃”任务：保持先前的对话固定不变，仅改变后续跟进的日期以及时间信息的呈现方式……

    arXiv:2611.17806v1 Announce Type: new  Abstract: Long-horizon agents increasingly serve as persistent personal assistants across modern web platforms, managing schedules, applications, and multi-step plans over weeks or months. Users often return to an ongoing conversation after a long conversational hiatus, while earlier bookings and preparation notes stay on record. During this hiatus, scheduled dates may pass even though the retained records remain unchanged and event outcomes go unrecorded. Existing evaluations of long-horizon agents focus on fact retention and explicit state updates. Whether agents perceive the passage of time and adapt their advice accordingly therefore remains unknown. We frame this ability as temporal state perception and use, which means perceiving the current time and applying it when interpreting dated records. To study it, we design session-rekindling tasks that keep the earlier conversation fixed and vary only the follow-up date and the way temporal inform
    
[^97]: 多模态人物概念模型：面向变化观测条件下的人员识别

    Multimodal Person Concept Model for Person Identification under Varying Observation Conditions

    [https://arxiv.org/abs/2611.17802](https://arxiv.org/abs/2611.17802)

    本研究提出一种基于非参数贝叶斯生成模型的多模态人物概念模型，融合CLIP和CLAP提取的视觉、声学与语言特征，在少量无标签观测下实现变化观测条件下的鲁棒人员识别。

    

    家庭和办公室环境中的服务机器人必须识别用户，才能根据其偏好和之前的交互提供个性化服务。这极具挑战性，因为可用于学习的观测样本可能很少，而且观测条件会因外观变化、环境噪声和话语内容不同而改变。为了应对这一挑战，本研究提出了一种多模态人物概念模型，该模型能够从少量观测数据中学习人物概念，无需人物标签，并能在变化的条件下实现鲁棒的识别。该模型采用基于中餐馆过程（Chinese restaurant process）的非参数贝叶斯生成模型，整合了通过对比语言-图像预训练（CLIP）和对比语言-音频预训练（CLAP）提取并经主成分分析（PCA）处理的视觉、声学和语言特征。研究在八种条件下对六名参与者进行了评估。

    arXiv:2611.17802v1 Announce Type: cross  Abstract: Service robots in domestic and office environments must identify users to provide personalized services based on their preferences and previous interactions. This is challenging because only a few observations may be available for learning, while observation conditions change owing to appearance, environmental noise, and utterance content. To address this challenge, this study proposes a multimodal person concept model that learns person concepts from a small number of observations, without person labels, and enables robust identification under changing conditions. The model employs a nonparametric Bayesian generative model based on the Chinese restaurant process and integrates visual, acoustic, and linguistic features extracted using Contrastive Language-Image Pre-training (CLIP) and Contrastive Language-Audio Pretraining (CLAP), followed by principal component analysis (PCA). We evaluated six participants under eight conditions in wh
    
[^98]: 视觉即工具：面向多模态智能体编程的视觉框架

    Vision-as-a-Tool: Visual Harness for Multimodal Agentic Coding

    [https://arxiv.org/abs/2611.17797](https://arxiv.org/abs/2611.17797)

    该论文提出“视觉即工具”范式，将通用多模态大语言模型重新定义为可调用的外部工具，使纯文本智能体能够完全卸载视觉推理，从而突破智能体自身感知能力对性能的限制。

    

    多模态智能体编程需要强大的视觉推理能力，以便分析观察结果并评估生成的产物。当前的框架主要依赖于“以图像思考”范式，即智能体依赖其内在视觉来执行视觉相关的任务。然而，这种架构上的耦合引入了一个根本性瓶颈：智能体的性能严格受限于其内部感知能力的局限。此外，纯文本智能体缺乏原生感知能力，没有专门的工具就无法进行视觉推理。为了克服这些限制，我们提出了“视觉即工具”范式。以往的视觉工具使用仅仅是为智能体的内在视觉操作图像以便其读取，相比之下，VaaT 将通用多模态大语言模型重新定义为可调用的外部工具。由此，纯文本智能体可以在智能体执行过程中完全卸载视觉推理任务，而具备视觉能力的智能体……

    arXiv:2611.17797v1 Announce Type: new  Abstract: Multimodal agentic coding requires robust visual reasoning to analyze observations and evaluate generated artifacts. Current frameworks largely rely on a Thinking-with-Images paradigm, where agents depend on their intrinsic vision to execute visually grounded tasks. However, this architectural coupling introduces a fundamental bottleneck. Agent performance remains strictly bounded by internal perceptual limitations. Furthermore, text-only agents lack native perception. Without specialized tools, they cannot perform visual reasoning. To overcome these constraints, we introduce the Vision-as-a-Tool (VaaT) paradigm. Prior visual tool-use merely manipulates images for an agent's intrinsic vision to read. In contrast, VaaT reframes general-purpose multimodal large language models (MLLMs) as callable external utilities. As a result, text-only agents can completely offload visual reasoning during agentic execution. Meanwhile, vision-capable age
    
[^99]: 不完整观测下GPU云定价基准的鲁棒性评估：一项受控缺失研究

    Robustness Evaluation of GPU Cloud Pricing Benchmarks under Incomplete Observations: A Controlled Missingness Study

    [https://arxiv.org/abs/2611.17796](https://arxiv.org/abs/2611.17796)

    本文通过受控缺失实验对GPU云租赁价格基准进行压力测试，发现全源中断主要损害覆盖率而成分变化损失引入显著误差，且仅用历史数据的层级消融方法能有效将误差降低4.15至6.70个对数点。

    

    当市场报价缺失时，每日GPU租赁价格基准可能会失效。本文在四种人为设置的掩蔽机制和四种基准级恢复方法下，对固定的等权中位数规则进行了压力测试。在190个保留测试日、27种场景共5,800次蒙特卡洛模拟中，全源中断使发布覆盖率降至0.77，同时在发布日期上保持零条件误差；而成分变化型损失保持0.86的覆盖率，但产生7.74个对数点的误差。一项单独的仅使用历史数据的层级消融实验，在五个成分变化案例中利用更丰富的信息集将误差降低了4.15至6.70个对数点。一种可观测信号保留规则在10个案例中对主要误差-覆盖率指标的改善次数为零。研究结果仅针对观察到的报价测量，而非成交价格或已识别的真实世界缺失机制。

    arXiv:2611.17796v1 Announce Type: new  Abstract: Daily GPU rental-price benchmarks can fail when marketplace quotes are missing. This article stress-tests a fixed equal-weight median rule under four imposed masking mechanisms and four benchmark-level recovery methods. Across 5,800 Monte Carlo realisations in 27 scenarios on 190 held-out days, whole-source outages reduce publication coverage to 0.77 while preserving zero conditional error on published dates; composition-changing loss retains 0.86 coverage but produces 7.74 log points of error. A separate past-only stratum-level ablation reduces error by 4.15 to 6.70 log points in five composition-changing cases, using a richer information set. An observable-signal withholding rule improves the primary error-coverage criterion in 0 of 10 cases. Results concern observed asking-price measurements, not transaction prices or identified real-world missingness mechanisms.
    
[^100]: MASTARS：基于回报条件子目标的多智能体顺序轨迹增强

    MASTARS: Multi-Agent Sequential Trajectory Augmentation with Return-Conditioned Subgoals

    [https://arxiv.org/abs/2611.17793](https://arxiv.org/abs/2611.17793)

    MASTARS是一种基于扩散模型的多智能体轨迹增强框架，通过跨智能体顺序生成轨迹并利用修复方法保证协调一致性，有效解决了离线多智能体强化学习中多样性与协调性难以兼得的权衡难题。

    

    离线多智能体强化学习（MARL）的性能受限于固定离线数据集的质量，这促使了轨迹增强技术的发展。然而，多智能体轨迹增强面临一个根本性挑战，即多样性-协调性权衡：独立生成可以提升多样性但缺乏协调性，而联合生成能够带来协调性但缺乏多样性，只会复现训练分布。为了解决多智能体数据增强中的这一挑战，我们提出了MASTARS，一种新颖的基于扩散模型的框架，该框架按智能体顺序生成多智能体轨迹，同时通过修复（inpainting）方法强制实现跨智能体的一致协调。MASTARS生成的轨迹兼具多样性、协调性和全局一致性。在MPE、SMAC和SMAC-v2等多种基准测试上的实验表明，该框架在多种离线MARL方法上均带来了显著的性能提升。

    arXiv:2611.17793v1 Announce Type: new  Abstract: The performance of offline Multi-Agent Reinforcement Learning (MARL) is limited by the quality of the fixed offline dataset, motivating trajectory augmentation. However, augmenting multi-agent trajectories presents a fundamental challenge of the diversity-coordination trade-off: independent generation improves diversity but lacks coordination, while joint generation induces coordination but lacks diversity, reproducing the training distribution. To address this challenge in multi-agent data augmentation, we introduce MASTARS, a novel diffusion-based framework that generates multi-agent trajectories sequentially across agents while enforcing cross-agent consistent coordination via the method of inpainting. MASTARS produces trajectories that are diverse, coordinated and globally coherent. Experiments on various benchmarks such as MPE, SMAC, and SMAC-v2 demonstrate significant performance gains across multiple offline MARL methods.
    
[^101]: AutoRL：面向自主强化学习的多智能体系统

    AutoRL: A Multi-Agent System towards Autonomous Reinforcement Learning

    [https://arxiv.org/abs/2611.17790](https://arxiv.org/abs/2611.17790)

    AutoRL提出了一个由感知器等多个强化学习专属智能体角色组成的多智能体系统，通过持续监控演变的训练状态并将过程级监督与研究融入正在进行的RL训练，实现强化学习训练的自动化，为未来的递归自我改进（RSI）奠定基石。

    

    递归自我改进（RSI）是通向超级智能AI奇点最有希望的路径之一。实现这一愿景需要AI系统能够承担完整的训练流程以进行迭代改进。强化学习（RL）在这一循环中扮演着至关重要的角色，用于提升模型能力和泛化能力，然而由于训练动态不稳定以及需要持续监督，实现强化学习的自动化仍然困难重重。我们提出了AutoRL——一个用于自动化强化学习训练的多智能体系统，作为未来递归自我改进的基石。与通过端到端完整实验开展研究的方式不同，AutoRL监控不断演变的训练状态，并将过程级的监督和研究融入正在进行的强化学习训练中，遵循强化学习研究人员的实际工作流程。具体而言，AutoRL由三个强化学习专属的智能体角色组成。感知器追踪训练动态、样本采集、评估和效率，并触发……（摘要原文在此处截断）

    arXiv:2611.17790v1 Announce Type: new  Abstract: Recursive self-improvement (RSI) is one of the most promising paths toward an AI singularity for superintelligence. Achieving this vision requires AI systems to undertake the full training pipeline for iterative improvement. Reinforcement learning (RL) plays an essential role in this cycle to improve model capabilities and generalization, yet automating RL remains difficult due to unstable training dynamics and the need for sustained supervision. We introduce AutoRL, a multi-agent system for automating RL training, as a cornerstone for future RSI. Different from conducting research by end-to-end complete experiments, AutoRL monitors the evolving training state and integrates process-level supervision and research into an ongoing RL training, following the practical workflow of an RL researcher. Specifically, AutoRL consists of three RL-specific agentic roles. Perceptors track training dynamics, rollouts, evaluation, and efficiency, trigg
    
[^102]: 大语言模型能否优化预训练神经网络权重？——YOLO26目标检测中无梯度编辑的多阶段实证评估

    Can Large Language Models Optimize Pretrained Neural Network Weights? A Multi-Phase Empirical Evaluation of Gradient-Free Editing in YOLO26 Object Detection

    [https://arxiv.org/abs/2611.17789](https://arxiv.org/abs/2611.17789)

    本研究通过对YOLO26检测器的五阶段实证评估发现，包括LLM引导和进化算法在内的无梯度直接权重编辑方法虽能在开发集上带来微小增益，但在独立验证子集上泛化失败，表明大语言模型引导权重优化的价值仍然有限。

    

    直接权重编辑为微调预训练检测器提供了一种无需梯度的替代方案，但其泛化能力以及大语言模型（LLM）引导的价值仍不确定。我们使用YOLO26模型和MS COCO数据集评估了五个实验阶段，将确定性缩放、随机扰动、差分进化、CMA-ES、结构化干预、真实本地LLM提议、失败感知引导以及混合LLM-CMA-ES等方法与匹配的随机策略对照组进行比较。优化过程在1,000张COCO Val2017图像上使用Ultralytics原生的mAP50-95指标进行；选出的候选方案在一个不相交的、此前已检查过的1,500张图像子集上进行评估，并在可用的情况下单独报告COCO API的AP50-95指标。将CMA-ES预算从8次评估增加到250次评估，使开发集收益从+0.11个百分点提升至+0.28个百分点，而选择集上的变化仍为负值（-0.03至-0.12个百分点）；250次评估的结果仅使用了一个随机种子。混合LLM（摘要在此处被截断）

    arXiv:2611.17789v1 Announce Type: cross  Abstract: Direct weight editing offers a gradient-free alternative to fine-tuning pretrained detectors, but its generalization and the value of large language model (LLM) guidance remain uncertain. We evaluate five experimental phases using YOLO26 models and MS COCO, comparing deterministic scaling, random perturbations, differential evolution, CMA-ES, structured interventions, real local-LLM proposals, failure-aware guidance, and hybrid LLM-CMA-ES against a matched random-policy control. Optimization uses native Ultralytics mAP50-95 on 1,000 COCO Val2017 images; selected candidates are assessed on a disjoint, previously examined 1,500-image subset, with COCO API AP50-95 reported separately where available. Increasing the CMA-ES budget from 8 to 250 evaluations raised development gains from +0.11 to +0.28 percentage points, while selection changes remained negative (-0.03 to -0.12 points); the 250-evaluation result uses one seed. The hybrid LLM 
    
[^103]: 迈向秒级视频表示：具有自适应、免预测采样的二次门控特征网格

    Towards Seconds-level Video Representation: Quadratically Gated Feature Grid with Adaptive, Prediction-free Sampling

    [https://arxiv.org/abs/2611.17786](https://arxiv.org/abs/2611.17786)

    Q-GAS通过多分辨率显式3D特征网格、二次门控解码器和误差感知的免预测采样策略，实现了可在50秒内完成拟合的高效秒级隐式神经视频表示。

    

    隐式神经视频表示的构建速度出了名地慢，未能达到大多数工业应用的基本实用门槛。在此，我们提出了Q-GAS，这是一种可坐标查询的视频表示，它将多分辨率显式3D特征网格与紧凑的SwiGLU风格二次门控（QG）解码器以及误差感知的免预测采样器相结合。显式网格以适度增加存储为代价，避免了哈希冲突导致的系数共享问题，从而大幅提升了表达能力；QG解码器引入低秩局部二阶交互来改善非线性特征响应。在拟合过程中，全局误差图定期刷新高误差池，每个训练批次将池样本与均匀采样样本混合，无需借助辅助模型来预测困难样本，从而进一步提升了效率。在UVG的Beauty/Bosphorus/Jockey序列上，Q-GAS在50秒内完成拟合与量化……

    arXiv:2611.17786v1 Announce Type: cross  Abstract: Implicit neural video representations are notoriously slow to build, which does not cross the basic practical threshold for most industrial applications. Here, we present Q-GAS, a coordinate-queryable video representation that couples multi-resolution explicit 3D feature grids with a compact SwiGLU-style quadratic-gating (QG) decoder and an error-aware, prediction-free sampler. The explicit grids avoid hash-collision coefficient sharing at the cost of modestly higher storage yet greatly boost the expressive capacity; the QG decoder introduces low-rank local second-order interactions to improve the nonlinear feature response. During fitting, a global error map periodically refreshes a high-error pool, and each batch mixes pool samples with uniform samples, which is free from the auxiliary model to predict difficult samples, thereby further escalating the efficiency. On UVG Beauty/Bosphorus/Jockey, Q-GAS fits and quantizes under 50 s on 
    
[^104]: TReC：在指令条件围棋中评估落子引起的状态变化

    TReC: Evaluating Move-Induced State Changes in Instruction-Conditioned Go

    [https://arxiv.org/abs/2611.17781](https://arxiv.org/abs/2611.17781)

    本文提出迁移表示一致性指标，通过比较预测落子与专家落子按规则执行后引起的棋盘状态嵌入变化的余弦相似度来评估落子差异，实验表明 TReC 具有显著区分度且与坐标距离仅弱相关，能捕捉坐标距离无法反映的状态变化差异。

    

    当围棋模型预测的落子与专家的落子不同时，坐标距离衡量的是预测落点与参考落点之间的距离。我们引入了迁移表示一致性，来比较这两种落子如何改变棋盘状态的嵌入表示。两种落子都在相同的局面下根据围棋规则执行，TReC 定义为两种落子所产生的表示变化之间的余弦相似度。在使用冻结的 KataGo 网络对来自五局对局的 680 个中后期局面进行的实验中，每个局面中最高与最低候选落子之间的差距，对于直接后继状态余弦而言中位数为 0.01 到 0.08，而对于 TReC 则为 0.88 到 1.33，具体数值取决于评估器层。TReC 与坐标距离仅有微弱的关联：在每个局面内计算得到的候选落子与参考落点的距离与其 TReC 之间的秩相关，在这些局面上的中位数为 -0.19。

    arXiv:2611.17781v1 Announce Type: new  Abstract: When a Go model predicts a move different from the expert's, coordinate distance measures how far the prediction lands from the reference. We introduce Transition Representation Consistency (TReC) to compare how the two moves change a board-state embedding. Both moves are executed from the same position under the game rules, and TReC is the cosine similarity between the two resulting representation changes. In experiments with a frozen KataGo network on 680 mid-to-late-game positions from five games, the gap between the highest and lowest candidate per position has a median of 0.01 to 0.08 for the direct successor-state cosine and 0.88 to 1.33 for TReC, depending on the evaluator layer. TReC follows coordinate distance only weakly: the rank correlation between a candidate's distance to the reference and its TReC, computed within each position, has a median of -0.19 over these positions.
    
[^105]: LuthorSQL：作为较慢 LLM 智能体的“系统 1”的单遍文本到 SQL 模型

    LuthorSQL: A One-Pass Text-to-SQL Model as System 1 for a Slower LLM Agent

    [https://arxiv.org/abs/2611.17780](https://arxiv.org/abs/2611.17780)

    提出非自回归 7B 模型 LuthorSQL，作为 LLM 智能体的快速“系统 1”门控，在单 GPU 上每题 0.84 秒即达到 BIRD 测试集 68.03% 的执行准确率，仅在不确定时才将查询升级给较慢的智能体，从而大幅降低成本并保护数据隐私。

    

    使用工具查询数据库的前沿 LLM 智能体是目前最准确的文本到 SQL 系统之一，但每次查询调用托管模型速度慢、成本高，且会将组织的敏感查询暴露给外部 API。例如，一个每天处理 10,000 次查询的 Claude Opus 5.5 智能体，按标价计算每年成本超过 70,000 美元。那么，组织已有的本地硬件能在多大程度上卸载这一负担而不损害智能体的准确性？我们提出了 LuthorSQL，一个非自回归的 7B 文本到 SQL 模型，它作为较慢 LLM 智能体（系统 2）的快速“系统 1”门控，仅在不确定时才将查询升级给智能体。LuthorSQL 通过单次前向传播，在受限文法上预测查询计划的槽位分数，编译并运行最多 32 个候选计划，并使用训练好的验证器选出答案。仅凭自身，LuthorSQL 在 BIRD 隐藏测试集上达到 68.03% 的执行准确率，在单个 GPU 上每个问题仅需 0.84 秒。

    arXiv:2611.17780v1 Announce Type: cross  Abstract: Frontier LLM agents that query databases with tools are among the most accurate text-to-SQL systems, but per-query calls to a hosted model are slow, costly, and expose sensitive organizational queries to external APIs. For example, a Claude Opus 5.5 agent serving 10,000 queries a day costs over $70,000 a year at list price. To what extent can local hardware already owned by an organization offload this burden without compromising the agent's accuracy? We present LuthorSQL, a non-autoregressive 7B text-to-SQL model that serves as a rapid System 1 gate for a slower LLM agent (System 2), escalating queries only when uncertain. LuthorSQL makes a single forward pass to predict slot scores over a bounded grammar for query plans, compiles and runs up to 32 candidate plans, and chooses an answer using a trained validator. On its own, LuthorSQL reaches 68.03% execution accuracy on BIRD's hidden test set at 0.84 s per question on a single GPU, m
    
[^106]: 利用大语言模型智能体构建的搜索流水线对已知最优汤姆逊问题构型的认证改进

    Certified Improvements to Best-Known Thomson-Problem Configurations with a Search Pipeline Built with Large Language Model Agents

    [https://arxiv.org/abs/2611.17776](https://arxiv.org/abs/2611.17776)

    研究者借助大语言模型智能体构建的搜索流水线（核心为副本交换盆地跳跃等方法），发现了N=912、1382和2502的汤姆逊问题新构型，其能量均低于剑桥团簇数据库的最优记录，并通过精确整数运算获得认证。

    

    汤姆逊问题要求在球面上放置N个单位电荷并使其库仑能量最小；对于较大的N，已知的最优构型收录于剑桥团簇数据库，其中最大N的条目于2026年10月确定。我们报告了N = 912、1382和2502的构型，其能量分别低于数据库值0.0036186、0.0109730和0.0614033，并通过精确整数运算进行了认证。这三种构型在模旋转意义下均为数值上严格的局部极小值，且保持了数据库构型的缺陷清单与疤痕组成；数据库中N = 912的数值此前曾由两种不同方法独立得到。我们的搜索从数据库构型出发，运行无约束的副本交换盆地跳跃算法，并结合共享极小值池、按几何重新标记后对改进区域的重组以及仅基于梯度的精修；仅凭副本交换盆地跳跃就找到了N = 912和1382的构型，前者是在五次运行之一中发现的。我们构建了

    arXiv:2611.17776v1 Announce Type: new  Abstract: The Thomson problem asks for N unit charges on a sphere with minimum Coulomb energy; for large N the best configurations known are curated in the Cambridge Cluster Database, whose largest-N entries were set in October 2026. We report configurations for N = 912, 1382 and 2502 with energies 0.0036186, 0.0109730 and 0.0614033 below the database values, certified by exact integer arithmetic. All three are numerically strict local minima modulo rotations and keep the defect inventory and scar compositions of the database configurations; the database value for N = 912 had been reached by two different methods. Our search runs unconstrained replica-exchange basin hopping from the database configurations, with a shared pool of minima, recombination of improved regions after relabelling by geometry, and gradient-only polishing; replica-exchange basin hopping alone found the N = 912 and 1382 configurations, the former in one of five runs. We built
    
[^107]: 激活引导《台风》中的“南山号”

    Activation Steering the Nan-Shan in Typhoon

    [https://arxiv.org/abs/2611.17768](https://arxiv.org/abs/2611.17768)

    本文介绍了基于康拉德《台风》的视听装置作品，通过让大语言模型以自身输出为食并采用一种新颖的自适应激活引导方法，探索AI生成文本随时间推移变得日益陌生和失控的过程，以此反思人类文化当下的人工智能时刻。

    

    《台风》是一件基于约瑟夫·康拉德中篇小说的视听装置作品，小说中麦克威尔船长驾驶“南山号”轮船直接驶入并穿越一场灾难性的风暴。这部小说研究了当一种现象的规模超出可用于理解它的实践与概念框架时会发生什么，以及在如此压倒性的情境下被要求采取行动时有时随之而来的瘫痪状态。我们选择《台风》这部文本来反思人类文化当前的人工智能时刻。我们以康拉德的原始语言作为种子提示一个大语言模型，使其无休止地以自身输出为食，并探索文本随着时间推移何时以及如何变得越来越陌生和难以驾驭，包括偏离叙事、游移至模型自身的指令和思考，以及长时间不连贯的沉思。我们采用了来自激活引导领域的方法，包括我们引入的一种新颖的自适应引导方法。

    arXiv:2611.17768v1 Announce Type: cross  Abstract: Typhoon is an audiovisual installation based on Joseph Conrad's novella, in which Captain MacWhirr steers the steamer Nan-Shan directly into and through a catastrophic storm. The novella is a study of what happens when the scale of a phenomenon exceeds the practical and conceptual frameworks available to comprehend it and of the paralysis that sometimes follows a demand for action in such overwhelming circumstances. We chose the text of Typhoon to reflect on human culture's current AI moment. We prompt a large language model, seeded with Conrad's original language, to feed endlessly on its own output, and explore when and how the text becomes increasingly alien and unwieldy as time progresses, including deviations from the narrative, digressions into the model's own instructions and thinking, and long periods of incoherent rumination. We employ methods from activation steering, including a novel adaptive steering method we introduce, t
    
[^108]: GDP.xlsx：基于专业工作簿的前沿模型基准测试

    GDP.xlsx: Benchmarking Frontier Models on Professional Workbooks

    [https://arxiv.org/abs/2611.17766](https://arxiv.org/abs/2611.17766)

    该论文提出GDP_xlsx基准，利用13个领域专业人士撰写的70个真实工作簿任务和791项专家评分准则来评估前沿AI模型的电子表格处理能力，发现最强模型Gemini 4 Argon的通过率也仅有38.3%，表明该领域仍有巨大的提升空间。

    

    我们推出了GDP_xlsx基准，包含70个由13个领域的专业人士撰写的真实职场请求，每个请求都配有专家制定的二元评分标准，以及其所指向的完整工作簿或CSV文件。这些文件共包含218万个已填充单元格，评分标准包含791项准则。智能体在配备电子表格工具的容器中工作，由一个能够打开文件的智能体裁判来应用评分标准。完整性和排他性准则要求集合型答案必须完整找出且没有多余内容。我们采用严格的pass@1指标对模型进行排名，即满足所有准则的尝试占比。在包括来自Google、Anthropic和OpenAI的前沿模型在内的20个模型配置中，得分最高的Gemini 4 Argon也只有38.3%的尝试通过，中位数得分为19.3%，留下了很大的提升空间。我们发布了评估框架和三个公开任务；其余67个任务被保留，以保护排行榜免受污染。

    arXiv:2611.17766v1 Announce Type: cross  Abstract: We introduce GDP_xlsx, a benchmark of 70 realistic workplace requests written by professionals in 13 domains, each paired with an expert rubric of binary criteria and with the complete workbook or CSV file it refers to. Together the files hold 2.18 million populated cells, and the rubrics 791 criteria. Agents work in a container with spreadsheet tools, and an agentic judge that can open the file applies the rubric. Completeness and exclusivity criteria require set-valued answers to be found in full and with nothing extra. We rank models by strict pass@1, the share of attempts that meet every criterion. Across 20 model configurations, including frontier models from Google, Anthropic, and OpenAI, the top-scoring one, Gemini 4 Argon, passes only 38.3% of attempts, and the median score is 19.3%, leaving wide headroom. We release the evaluation harness and three public tasks; the other 67 are held out to protect the leaderboard from contami
    
[^109]: 知识崩溃之后：上下文与措辞塑造语言模型中事实的可及性

    Knowledge After Collapse: Context and Wording Shape Factual Accessibility in Language Models

    [https://arxiv.org/abs/2611.17763](https://arxiv.org/abs/2611.17763)

    提出知识答案分歧（KAD）评估协议，通过在模型首次采纳错误答案时冻结对话并分支独立探针，揭示了对话历史与措辞方式显著影响语言模型在错误采纳后恢复正确事实的能力。

    

    当语言模型在事实性讨论中接受了一个错误的更正时，它的下一个答案可能取决于对话如何继续。我们引入了知识答案分歧，这是一种事件条件化的评估协议，能够区分模型最初对错误答案的采纳与随后的持续、恢复和弃权行为。KAD 在模型第一次明确采纳指定错误目标时冻结对话，并从该确切历史分支出独立的探针，这使得对采纳后行为和历史管理干预进行匹配比较成为可能，而单一的易感性评分无法捕捉这些差异。我们评估了八个本地检查点条件和六个托管端点，并进行了一次经过审计的300题重跑。在经过更正的 Llama 3.2 Instruct 3B 对照组中，中性历史分支产生了 42/46 的正确回答，而完整历史下仅为 16/46；一个匹配的线索位置对照使错误采纳率改变了 27.3 个百分点。

    arXiv:2611.17763v1 Announce Type: new  Abstract: When a language model accepts a false correction in a factual discussion, its next answer can depend on how the conversation continues. We introduce Knowledge Answer Divergence (KAD), an event-conditioned evaluation protocol that distinguishes initial false-answer adoption from subsequent persistence, recovery, and abstention. KAD freezes the conversation at the first explicit adoption of a designated false target and branches independent probes from that exact history. This enables matched comparisons of post-adoption behavior and history-management interventions that a single susceptibility score cannot capture. We evaluate eight local checkpoint conditions and six hosted endpoints, alongside an audited 300-question rerun. In corrected Llama 3.2 Instruct 3B controls, a neutral-history branch yields 42/46 correct responses, compared with 16/46 under full history. A matched cue-placement control changes false adoption by 27.3 percentage 
    
[^110]: 品牌形象约束下的创意产品设计

    Creative Product Design under Brand Image Constraints

    [https://arxiv.org/abs/2611.17762](https://arxiv.org/abs/2611.17762)

    该论文提出了一个品牌感知的产品再设计生成框架，通过构建品牌形象调研报告并将其创意方向整合进图像评价与提示词修订的自我增强循环中，从而确保生成的创意设计符合品牌形象约束。

    

    工业设计需要在满足约束条件的同时进行创意性的视觉探索，而这些约束往往是产品特定的且隐性的。现有的文本生成图像（T2I）方法可以通过概念融合或多智能体自我增强来生成创意设计，但它们在确保生成设计与此类约束保持一致方面的机制较为有限。我们聚焦于品牌形象作为一种代表性的设计约束，提出了一个品牌感知的再设计生成框架。给定一个产品名称，该框架会构建一份品牌形象调研报告，总结产品和市场信息，从报告中推导出创意方向，并将该方向整合到自我增强循环中，在这个循环中对生成的图像进行评价并修订提示词。我们在五个产品类别的50个再设计任务上对该框架进行了评估，将其与朴素T2I、IP-Adapter/IT-Blender以及无约束的自我增强方法进行了比较。

    arXiv:2611.17762v1 Announce Type: cross  Abstract: Industrial design requires creative visual exploration while satisfying constraints that are often product-specific and implicit. Existing text-to-image (T2I) approaches can generate creative designs through concept blending or multi-agent self-enhancement, but they provide limited mechanisms for ensuring that generated designs remain consistent with such constraints. We focus on brand image as a representative design constraint and propose a framework for brand-aware redesign generation. Given a product name, the framework constructs a brand image survey report that summarizes product and market information, derives a creative direction from the report, and integrates this direction into a self-enhancement loop in which generated images are critiqued and prompts are revised. We evaluate the framework on 50 redesign tasks across five product categories, comparing it with Naive T2I, IP-Adapter/IT-Blender, and unconstrained self-enhancem
    
[^111]: Superfluid：面向分布式大语言模型推理的本地AI服务器

    Superfluid: Local AI Server for Distributed LLM Inference

    [https://arxiv.org/abs/2611.17753](https://arxiv.org/abs/2611.17753)

    Superfluid是一个开源本地LLM推理服务器，通过统一调度器在单设备上协调多智能体与交互用户共享模型服务（支持llama.cpp、MLX、BaseRT等运行时），并能将多台异构闲置机器聚合为单一API端点实现分布式推理。

    

    在本地设备上运行大语言模型（LLM）可以保护数据隐私、免除按token计费的成本、降低网络延迟，并在无网络连接时仍能正常工作。如今，两种变化正对这类模型的服务方式构成挑战。在单台设备上，多个智能体与交互式用户共享同一个本地服务的模型；在多台设备之间，开发者或小团队往往拥有来自不同厂商的多台性能不俗的机器，但由于每个本地服务器只能在一台机器上服务一个运行时，这些机器在一天中的大部分时间都处于闲置状态。我们提出了Superfluid，一个为上述两种规模的本地推理而构建的开源LLM服务器。在单台设备上，Superfluid通过统一的调度器驱动运行于隔离工作进程中的llama.cpp、MLX或BaseRT，将交互式请求排在智能体任务之前优先处理，对被多个请求共享的提示词只计算一次，并在每个token到达客户端之前将其写入持久化日志。在多台设备之间，一个Superfluid服务器可以将一组机器整合为单一的API端点……

    arXiv:2611.17753v1 Announce Type: cross  Abstract: Running large language models (LLMs) on local devices keeps data private, removes per-token cost, cuts network latency, and keeps working without a connection. Two shifts now strain how such models are served. On one device, several agents share a locally served model with an interactive user. Across devices, a developer or small team often owns several capable machines from different vendors, which sit idle most of the day because every local server serves one runtime on one machine. We present Superfluid, an open-source LLM server built for local inference at both scales. On one device, Superfluid drives llama.cpp, MLX, or BaseRT in isolated worker processes through one scheduler that moves interactive requests ahead of agent work, computes a prompt shared by many requests once, and writes every token to a durable log before it reaches the client. Across devices, one Superfluid server turns a fleet of machines into a single API endpo
    
[^112]: 逻辑一致且准确的预测

    Logically Consistent Accurate Forecasts

    [https://arxiv.org/abs/2611.17735](https://arxiv.org/abs/2611.17735)

    该论文提出微调一个监督语言模型，利用支撑每个预测的推理过程来协调相关预测之间的逻辑不一致性，从而在ForecastBench上同时实现预测的逻辑一致性与准确性的提升。

    

    基于大语言模型（LLM）的智能体预测系统如今已能与专家级人类预测员相媲美，但这些系统仍然存在一个关键局限：无法产生逻辑一致的预测。例如，对相关事件的预测往往不满足概率公理，这意味着相关预测不可能同时正确，从而削弱了其对决策者的实用价值。强制一致性并非易事——朴素的方法（例如投影到一致预测的集合上）可能会降低准确性。我们认为，克服一致性与准确性之间这种明显权衡的关键在于支撑每个预测的推理过程。我们微调了一个监督语言模型，该模型利用这种推理来协调相关预测之间的不一致性，同时提高其准确性。在ForecastBench上，我们的方法改进了Brier分数（Brier score）……

    arXiv:2611.17735v1 Announce Type: cross  Abstract: Agentic forecasting systems based on large language models (LLMs) are now competitive with expert human forecasters, but these systems still suffer from a critical limitation: the inability to produce forecasts that are logically consistent. For example, forecasts for related events often fail to satisfy the axioms of probability, which in turn implies that related forecasts cannot simultaneously be correct and undermines their utility for decision makers. Enforcing consistency is not trivial -- naive approaches (e.g., projection onto the set of consistent forecasts) can degrade accuracy. We argue that the key to overcoming the apparent tradeoff between consistency and accuracy lies in the reasoning which supports each forecast. We finetune a supervisory language model that leverages this reasoning to jointly reconcile inconsistencies while improving the accuracy of related forecasts. On ForecastBench, our method improves the Brier sco
    
[^113]: 模型合并中通用代理正则化的谱框架

    A Spectral Framework for Universal Proxy Regularization in Model Merging

    [https://arxiv.org/abs/2611.17728](https://arxiv.org/abs/2611.17728)

    本文提出通用代理正则化的谱框架，通过谱覆盖率与谱泄漏两个指标刻画代理分布与任务分布的关系，并从理论上证明更小的谱指标可通过更好的谱对齐降低模型合并中的参数干扰。

    

    模型合并无需昂贵的联合训练即可构建多任务模型，但常常受到参数干扰的困扰。一些解决方案通过使用待合并的其他任务的数据来约束微调，从而缓解这一问题，但这违背了模型合并的核心独立性和隐私假设。近期的工作提供了初步的实证证据，表明一个通用的、与任务无关的代理分布可以替代此类数据，但其何时以及为何有效尚不清楚。为此，我们提出了通用代理正则化的谱框架，该框架通过谱覆盖率（α）和谱泄漏（β）来刻画代理分布与待合并其他任务分布之间的关系。基于该框架，我们推导出一个统一的参数干扰上界：更小的（α, β）可通过更好的谱对齐在理论上保证更低的参数干扰。为了利用这样的代理，我们增强了……

    arXiv:2611.17728v1 Announce Type: new  Abstract: Model merging constructs multi-task models without expensive joint training but often suffers from parameter interference. Some solutions mitigate this by constraining finetuning using data from the other tasks to be merged, violating the core independence and privacy assumption of model merging. Recent work offers preliminary empirical evidence that a universal, task-agnostic proxy distribution can substitute for such data, but leaves open when and why it suffices. To this end, we propose a \textit{Spectral Framework for Universal Proxy Regularization}, which characterizes the relationship between a proxy distribution and the distribution of other tasks to be merged through spectral coverage ($\alpha$) and leakage ($\beta$). Based on the framework, we derive a unified parameter interference bound: smaller $(\alpha, \beta)$ provably yields lower parameter interference through better spectral alignment. To exploit such a proxy, we augment
    
[^114]: 具有收敛性保证的分布式网络上的联邦软多层级学习

    Federated Soft Poly-Level Learning Over a Distributed Network with Guaranteed Convergence

    [https://arxiv.org/abs/2611.17722](https://arxiv.org/abs/2611.17722)

    该论文提出了联邦多层级学习框架FPLL，利用新颖的非线性割技术，在保证收敛性的前提下以联邦方式解决分布式网络上含软约束的四层级及以上的多层级优化问题，同时避免了数据集中带来的隐私风险。

    

    多层级学习/优化，例如双层优化，因其可广泛应用于从信号处理到机器学习等诸多领域而受到认可。然而，现有工作大多聚焦于双层和三层优化。相比之下，涉及四个或更多层级的多层优化问题——我们称之为多层级学习（PLL）问题——在很大程度上仍未被探索。此外，在实际的PLL场景中，数据可能分布在分布式网络的多个节点上。然而，现有的PLL研究主要集中在非联邦设置，需要将数据收集到中央服务器，从而带来潜在的隐私风险。为此，我们提出了一个在分布式网络上运行的联邦多层级学习框架，以联邦的方式求解带有软约束的PLL问题。FPLL的核心是一种新颖的非线性割，即 $L（摘要在此处被截断）

    arXiv:2611.17722v1 Announce Type: new  Abstract: Multi-level learning/optimization, e.g., bilevel optimization, has been recognized for its versatile applicability across a broad array of domains, spanning from signal processing to machine learning. However, most existing works focus on bilevel and trilevel optimization. In contrast, multi-level optimization problems involving four or more hierarchical levels, which we term poly-level learning (PLL) problems, remain largely unexplored.In addition, in practical PLL, data may be distributed across multiple nodes over a distributed network. Nevertheless, extant studies on PLL predominantly focus on the non-federated setting, necessitating the collection of data to central servers and thereby posing potential privacy risks. To this end, we propose a Federated Poly-Level Learning (FPLL) framework over a distributed network to tackle the PLL problems with soft constraints in a federated manner. FPLL hinges on a novel non-linear cut, i.e., $L
    
[^115]: 跨语言认知障碍声学生物标志物研究：交互式与单人语音任务对比

    Cross-Linguistic Acoustic Biomarkers of Cognitive Impairment Across Interactive and Single-Speaker Speech Tasks

    [https://arxiv.org/abs/2611.17710](https://arxiv.org/abs/2611.17710)

    该研究通过对西班牙语、普通话、英语和日语四种语言中对话式与结构化语音任务的跨语言、跨任务声学分析，结合统计、专家指导和机器学习三种特征选择方法，提取出紧凑可解释的声学生物标志物，为多语言认知障碍检测提供了可扩展且公平的筛查方案。

    

    多语言痴呆症检测对于实现可扩展且公平的筛查至关重要。近年来，自动化语音分析已成为支持痴呆症检测的一种有前景的方法，它能够以非侵入式、可扩展的方式评估认知功能。然而，现有的自动化语音分析研究通常只依赖单一类型的任务（如图片描述），并使用黑盒模型，往往忽视了任务类型和语言本身的作用。为填补这些空白，我们开展了一项跨语言、跨任务的声学分析，将对话式数据集与涵盖四种语言（西班牙语、普通话、英语、日语）的结构化语音任务相结合。我们综合运用统计学方法、语言学家/临床医生指导的方法以及基于机器学习模型的特征选择方法，在保持分类性能的同时提取出紧凑的声学特征集，从而能够系统地比较与认知障碍相关的声学特征。

    arXiv:2611.17710v1 Announce Type: new  Abstract: Multilingual dementia detection is essential for scalable and equitable screening. In recent years, automated speech analysis has emerged as a promising approach for supporting dementia detection by enabling non-invasive and scalable assessment of cognitive function. However, existing studies that use automated speech analysis often rely on one type of task, such as picture description, and use black-box models. These studies often neglect the role of task and language. To address these gaps, we performed a cross-linguistic and cross-task acoustic analysis that integrates a conversational dataset with structured speech tasks across four languages (Spanish, Mandarin, English, Japanese). We applied statistical, linguist-/clinician-guided, and machine learning model-based feature selection to derive compact acoustic feature sets while maintaining classification performance. This enabled systematic comparison of cognitive impairment-relevant
    
[^116]: 一种超越业务化全球生物地球化学系统的AI高分辨率叶绿素a预测模型

    An AI-enabled high-resolution chlorophyll-a forecasting model that outperforms the operational global biogeochemical system

    [https://arxiv.org/abs/2611.17704](https://arxiv.org/abs/2611.17704)

    本研究提出AI模型ChlCast，利用分解式时空卷积与多尺度架构，以约4公里分辨率生成全球10天逐日表层叶绿素a预报，其精度超越了业务化的哥白尼全球海洋生物地球化学预报系统。

    

    表层叶绿素a浓度是海洋浮游植物变化的关键指标，对其分布的短期预报可为海洋生态系统监测与管理提供前瞻性信息。然而，物理输送与生物过程之间的相互作用使叶绿素a预测变得复杂，而用传统数值模式模拟这些耦合过程需要巨大的计算开销。本文提出ChlCast，这是一个AI模型，可在南纬60°至北纬60°之间以约4公里的分辨率生成未来10天的逐日表层叶绿素a浓度预报。ChlCast将分解式时空卷积与多尺度架构相结合，从历史叶绿素a场和气候学信息中学习。与业务化的0.25°哥白尼全球海洋生物地球化学预报系统相比，ChlCast降低了平均绝对误差并……（原文摘要在此处截断）

    arXiv:2611.17704v1 Announce Type: new  Abstract: Surface chlorophyll-a concentration is a key indicator of marine phytoplankton variability, and short-term forecasts of its distribution provide advance information for marine ecosystem monitoring and management. However, interactions between physical transport and biological processes complicate chlorophyll-a forecasting, and simulating these coupled processes with conventional numerical models imposes substantial computational demands. Here we introduce ChlCast, an AI model that generates daily forecasts of surface chlorophyll-a concentration over a 10-day horizon at approximately 4-km resolution between 60$^\circ$S and 60$^\circ$N. ChlCast combines factorized spatiotemporal convolutions with a multiscale architecture to learn from historical chlorophyll-a fields and climatological information. Compared with the operational 0.25$^\circ$ Copernicus global ocean biogeochemical forecasting system, ChlCast reduces mean absolute error and r
    
[^117]: Co-Sight Harness：面向长程业务智能体的外部化控制资产与确定性强制执行机制

    Co-Sight Harness: Externalized Control Assets and Deterministic Enforcement for Long-Horizon Business Agents

    [https://arxiv.org/abs/2611.17696](https://arxiv.org/abs/2611.17696)

    该论文提出Co-Sight Harness框架，通过将工作流定义、状态与领域知识外部化为可版本化的控制资产，并借助状态机、机器可校验的验收门控和独立审计器进行确定性强制执行，在不改动底层语言模型的情况下，使超过200步的长程业务工作流无人值守完成率从50%提升至100%、端到端延迟降低约64%。

    

    长程业务工作流会以两种方式失败：随着活动链条的延长，错误不断累积；同时某项活动所需的知识所用词汇与用户请求的表述不一致。我们研究能否在不改变语言模型的前提下使这类工作流变得可靠。我们的方法保持模型骨干固定不变，将工作流定义、状态与领域知识转移到模型外部可版本化管理的控制资产中。在执行过程中，由状态机和机器可校验的验收门控来强制实施这些资产；并由一个独立的审计器对产生的执行轨迹进行评分。这些资产只有在通过回归检查并获得人工批准后才会更新。我们在三个已部署的、超过200步的工作流上对该方法进行了评估。在领域A上，一个包含120个案例的端到端评估集显示：无人值守完成率从50%提升至100%，检索召回率从56%提升至91.67%，端到端延迟从440分钟降至159分钟，且场景分析……（摘要在此处截断）

    arXiv:2611.17696v1 Announce Type: new  Abstract: Long-horizon business workflows fail in two ways: errors accumulate as a chain of activities lengthens, and the knowledge an activity needs is written in a different vocabulary from the user's request. We study whether such workflows can be made reliable without changing the language model. Our method keeps the backbone fixed and moves the workflow definition, state and domain knowledge into versioned control assets outside the model. A state machine and machine-checkable acceptance gates enforce these assets during execution; an independent auditor grades the resulting trace. The assets are updated only after a regression check and human approval .We evaluate the method on three deployed workflows exceeding 200 steps. On Domain A, a 120-case end-to-end evaluation set shows unattended completion rising from 50% to 100%, retrieval recall from 56% to 91.67%, and end-to-end latency dropping from 440 to 159 minutes, with scenario-analysis re
    
[^118]: 代理编写的启发式算法与商业求解器在未解决作业车间基准实例上的对比

    Agent-written heuristics versus commercial solvers on unsolved job-shop benchmark instances

    [https://arxiv.org/abs/2611.17682](https://arxiv.org/abs/2611.17682)

    前沿大语言模型作为编程代理、仅用开源工具编写的求解器在多个未解决的作业车间基准实例上超越了商业求解器，改进了大量开放实例并建立了新的下界。

    

    商业求解器是解决困难调度问题的标准工具。我们研究前沿大型语言模型——仅作为使用开源工具的编程代理——能否在质量和速度上与之匹敌。我们报告了每一次运行，并由独立检查器验证每一个调度方案。在包含10万个操作的作业车间实例 tai_100_1000_1 上，代理编写的求解器达到了 525,515，优于 OptalCP、Hexaly 和 CP Optimizer。代码未经修改，它改进了该系列的全部十个实例，并关闭了其中两个。在同一工作站、相同计算预算下，该求解器的结果距下界仅 0.0-1.8%，而 CP Optimizer 为 2.6-5.8%，Hexaly 为 2.4-4.4%。在 Behnke-Geiger 柔性作业车间问题上，代理编写的禁忌搜索改进了 32 个开放实例中的 11 个。在配备两台运输机器人的场景下，代理编写的求解器改进了 10 个 MKT 实例中的 7 个。新的下界表明，已发表的四个 MKT 数值在分发的实例文件上是无法达到的。

    arXiv:2611.17682v1 Announce Type: cross  Abstract: Commercial solvers are the standard tool for difficult scheduling problems. We study whether a frontier large language model, used as a coding agent with open-source tools only, can match them in quality and speed. We report every run, and independent checkers verify every schedule. On the 100,000-operation job shop tai_100_1000_1, an agent-written solver reached 525,515, below OptalCP, Hexaly, and CP Optimizer. Unchanged, it improved all ten instances of the family and closed two. With equal budgets on the same workstation, it ended 0.0-1.8% above the lower bound, against 2.6-5.8% for CP Optimizer and 2.4-4.4% for Hexaly. On the Behnke-Geiger flexible job shops, an agent-written tabu search improved 11 of 32 open instances. With two transport robots, an agent-written solver improved seven of ten MKT instances. New lower bounds show that four published MKT values cannot be attained on the distributed instance files. Agent-run improveme
    
[^119]: OCTAVE：一个面向端到端HTML5游戏生成的能力分辨基准

    OCTAVE: A Capability-Resolved Benchmark for End-to-End HTML5 Game Generation

    [https://arxiv.org/abs/2611.17681](https://arxiv.org/abs/2611.17681)

    该论文提出OCTAVE基准，通过八个能力维度、180个任务和1,800条评估需求对端到端HTML5游戏生成进行需求级能力诊断，并揭示当前最优模型组合的完整性与丰富度仅约33%。

    

    完整的游戏生成要求智能体在漫长的开发过程中协调规则、交互、状态与呈现。单一的聚合分数会掩盖哪些能力已被实现、哪些需求仍未被满足。我们提出OCTAVE，一个以八个能力维度组织的完整HTML5游戏生成基准，包含180个任务和1,800条评估需求，这些需求来源于真实玩家请求、商业机制重建以及人类专家设计。该基准涵盖PC网页、移动网页、2D和3D场景。评估需求区分“完整性”——评估公开指定的核心实现——与“丰富度”——评估隐藏的、更深层的系统关系。基于静态源代码的判定可提供跨能力和任务条件的需求级诊断。对11种模型-框架组合的评估结果显示，完整性为33.2%，丰富度为32.5%；19.9%的生成记录属于静态门控失败或缺失。

    arXiv:2611.17681v1 Announce Type: new  Abstract: Complete game generation requires agents to coordinate rules, interactions, states, and presentation over a long development process. A single aggregate score obscures which capabilities are implemented and which requirements remain unmet. We present OCTAVE, a benchmark for complete HTML5 game generation organized around eight Capability Dimensions, with 180 tasks and 1,800 Evaluation Requirements drawn from real player requests, commercial-mechanics reconstruction, and human expert design. It covers PC Web, Mobile Web, 2D, and 3D. Requirements distinguish Completeness, assessing publicly specified core implementation, from Richness, assessing hidden, deeper system relationships. Static source-code judgments provide requirement-level diagnosis across capabilities and task conditions. Evaluation of 11 model-harness combinations yields 33.2% Completeness and 32.5% Richness; 19.9% of generation records are static gate failures or missing ar
    
[^120]: 一致、冲突还是未指明？面向多模态时间序列预测的诊断基准

    Aligned, Conflicting, or Unspecified? A Diagnostic Benchmark for Multimodal Time Series Forecasting

    [https://arxiv.org/abs/2611.17680](https://arxiv.org/abs/2611.17680)

    该论文提出了首个按文本与历史时间序列关系（一致、冲突、未指明）标注的多模态时间序列预测诊断基准，包含覆盖天气、维基百科和金融领域的1,762对文本-窗口数据，用于零样本系统评估十个预训练模型利用外部文本的可靠性与错误修订能力。

    

    人工智能物联网系统日益将感知的时间序列与专家报告、新闻等外部文本相结合，然而模型能否可靠地使用与感知历史一致或相冲突的文本仍不清楚。我们提出了首个按“文本-历史关系”组织的真实世界诊断基准：涵盖传感器遥测（天气）、网络服务流量（维基百科）和市场数据流（金融）三个领域共1,762对文本-时间窗口数据，每对根据文本方向与历史趋势一致、相反或未作说明，分别标注为“一致”、“冲突”或“未指明”。在十个预训练模型的零样本评估中，我们研究了文本-历史关系下的文本选择与组合（研究问题1），以及错误诊断与跨预测时间跨度的修订控制（研究问题2）。针对研究问题1，我们在配对的天气时间窗口上比较了两类文本：一类直接预测温度变化，另一类描述相关的天……

    arXiv:2611.17680v1 Announce Type: new  Abstract: Artificial Intelligence of Things (AIoT) systems increasingly combine sensed time series with external text such as expert reports and news, yet how reliably models use text that agrees or conflicts with the sensed history remains unclear. We introduce the first real-world diagnostic benchmark organized by text-history relationship: 1,762 text-window pairs across sensor telemetry (Weather), web-service traffic (Wikipedia), and market data streams (Finance), each labeled Aligned, Conflicting, or Unspecified by whether the text's direction agrees with the historical trend, opposes it, or is not stated. Across ten pretrained models evaluated zero-shot, we investigate text selection and combination under text-history relationships (RQ1) and error diagnosis and horizon-scaled revision control (RQ2). For RQ1, we compare two types of text on paired Weather windows: one directly forecasts temperature changes, while the other describes related we
    
[^121]: 先学习再下注：学习可解释规则以实现可靠的智能体干预

    Learn then Bet: Learning Interpretable Rules for Reliable Agent Intervention

    [https://arxiv.org/abs/2611.17662](https://arxiv.org/abs/2611.17662)

    提出了LeBet统计学习框架，通过学习可解释的“或-与”规则监控器并结合固定下注统计量积累证据，在达到阈值时对自主智能体运行进行干预，并提供有限样本下的假阳性控制和检测功效保证。

    

    自主智能体现在可以在很少或没有人类监督的情况下运行数百步。在实践中，从业者使用运行时监控器（如轮次限制和关键词过滤器）来决定何时应停止、升级或协助某次运行。然而，这些启发式方法可能会停止本来会成功的运行，或让失败的运行继续下去。为了解决这一局限，我们提出了LeBet（Learn then Bet，先学习再下注），这是一个统计学习框架，其学习目标源自有效且强有力干预的充分条件。特别地，LeBet学习一个可解释的“或-与”（OR-of-AND）监控器，并通过固定的下注统计量积累证据，当证据达到预设阈值时进行干预。在有限的类内可交换性和留一稳定性条件下，我们利用相同的轨迹数据，推导出了有限样本保证，用于控制假阳性（有效性）以及以训练数据为条件的检测功效。

    arXiv:2611.17662v1 Announce Type: new  Abstract: Autonomous agents now run for hundreds of steps with little or no human supervision. In practice, practitioners use runtime monitors such as turn limits and keyword filters to decide when a run should be stopped, upgraded, or assisted. However, these heuristics may stop runs that would otherwise succeed or allow failing runs to continue. To address this limitation, we introduce LeBet (Learn then Bet), a statistical learning framework that derives its learning objective from sufficient conditions for valid and powerful intervention. In particular, LeBet learns an interpretable OR-of-AND monitor and accumulates evidence through a fixed betting statistic, intervening when the evidence reaches a prescribed threshold. Under finite within-class exchangeability and leave-one-out stability, we derive finite-sample guarantees for false-positive control (validity) and detection power conditional on the training data, using the same trajectories fo
    
[^122]: 模仿策略学习本体感知锚点：感知噪声及其闭环代价的预测定律

    Imitation Policies Learn a Proprioceptive Anchor: A Predictive Law for Perception Noise and Its Closed Loop Cost

    [https://arxiv.org/abs/2611.17658](https://arxiv.org/abs/2611.17658)

    该论文揭示了模仿策略会将机器人自身状态作为“本体感知锚点”来折扣含噪感知读数，其权重服从 α=V/(V+s²) 的预测定律，且锚点方差远小于线性模型暗示值，一次测量即可精确预测策略在不同任务和噪声水平下对感知噪声的敏感度。

    

    机器人操作的模仿策略通常在有噪声的物体位置读数上进行训练，它们学会对含噪读数进行折扣。我们展示了它们折扣的程度及其原因。由于演示者会朝物体方向移动，机器人自身的状态携带着物体位置的信息。训练好的策略将这一本体感知锚点与读数相结合，并以 α = V/(V + s²) 对读数加权，其中 s² 是读数噪声，V 是锚点的误差方差。V 属于学习器本身，远小于由演示数据建立的线性模型所暗示的值，这正是经典变量误差回归高估 α 的原因。V 只需测量一次，即可在两个 robomimic 任务、两个学习器和四个新噪声水平下预测十个保留样本的敏感度，平均绝对误差为 0.025，而经典 EIV 方法为 0.37；此外，一个无参数的偏差测试排除了更简单的解释，即策略只是单纯高估……

    arXiv:2611.17658v1 Announce Type: cross  Abstract: Imitation policies for robot manipulation are often trained on noisy object positions, and they learn to discount the noisy reading. We show how much they discount it, and why. Because demonstrators steer toward the object, the robot's own state carries information about where the object is. A trained policy combines this proprioceptive anchor with the reading and weights the reading by $\alpha = V/(V + s^2)$, where $s^2$ is the reading noise and $V$ the anchor's error variance. $V$ belongs to the learner and is far smaller than a linear model of the demonstrations implies, which is why classical errors-in-variables (EIV) regression overestimates $\alpha$. Measured once, $V$ predicted ten held-out sensitivities across two robomimic tasks, two learners and four new noise levels, with mean absolute error 0.025 against 0.37 for classical EIV, and a parameter-free bias test rules out the simpler explanation that the policies just overestim
    
[^123]: 二值化：无权重神经网络的攻击面

    Binarization as an Attack Surface in Weightless Neural Networks

    [https://arxiv.org/abs/2611.17646](https://arxiv.org/abs/2611.17646)

    该研究揭示了无权重神经网络（WiSARD及其布隆过滤器变体）虽然看似对对抗样本天然鲁棒，但其二值化阈值本身可被利用为攻击面，并通过阈值跨越界限和响应边际证书对这一攻击机制进行了形式化证明。

    

    无权重神经网络（WNNs）是一类轻量级分类器，它们对输入进行二值化处理，并通过在查找存储器中存储二进制模式来进行学习，这使其在入侵检测等资源受限任务中颇具吸引力。由于不跨越任何二值化阈值的微小扰动不会改变二进制输入，无权重神经网络似乎对对抗样本具有天然的鲁棒性；然而，一旦扰动跨越了阈值，仍可能改变模型的预测结果。我们在WiSARD（经典的基于RAM的无权重神经网络）以及两个用布隆过滤器替代其存储器的变体——Bloom WiSARD和BTHOWeN——中研究了这一矛盾。连续扰动只有通过跨越编码阈值、改变至少一个二进制元组并重定向存储器查询，才能影响这些分类器；我们通过阈值跨越界限和标准布尔WiSARD的响应边际证书对这一机制进行了形式化。受控MNIST实验…

    arXiv:2611.17646v1 Announce Type: cross  Abstract: Weightless Neural Networks (WNNs) are lightweight classifiers that binarize their inputs and learn by storing binary patterns in lookup memories, which makes them attractive for resource-constrained tasks such as intrusion detection. Because a small perturbation that crosses no binarization threshold leaves the binary input unchanged, WNNs may appear naturally robust to adversarial examples; a perturbation that does cross a threshold, however, can still change the prediction. We study this tension in WiSARD, the canonical RAM-based WNN, and in two variants that replace its memories with Bloom filters, Bloom WiSARD and BTHOWeN. A continuous perturbation reaches these classifiers only by crossing an encoding threshold, changing at least one binary tuple, and redirecting a memory query; we formalize this mechanism with a threshold-crossing bound and a response-margin certificate for standard Boolean WiSARD. Controlled MNIST experiments co
    
[^124]: CoTTAGE：面向文本到音频生成的可解释评判器

    CoTTAGE: An Explainable Judge for Text-to-Audio Generation

    [https://arxiv.org/abs/2611.17644](https://arxiv.org/abs/2611.17644)

    CoTTAGE是一个采用“先推理后预测”范式的可解释文本到音频生成评估器，它先生成基于音频的错误诊断依据，再预测文本相关性和音频质量分数，并通过监督微调与基于Thurstone偏好目标的强化学习，实现了与人类判断的高度一致。

    

    文本到音频（TTA）生成领域的快速发展需要可靠且可解释的自动评估方法。然而，现有指标无法同时实现与人类判断的高度相关以及有意义的、基于音频的解释，这迫使研究者依赖耗时费力的人工评估，成为模型开发迭代的瓶颈。为解决这一问题，我们提出了CoTTAGE，一个基于推理的评估器，用于评估文本相关性（REL）和音频质量（OVL）。遵循“先推理后预测”的范式，CoTTAGE在预测分数之前，会先生成基于音频的推理依据来诊断错误。在超过1万条新收集的人类OVL标注的支持下，CoTTAGE首先通过监督微调进行训练，随后利用基于Thurstone的偏好目标进行强化学习优化，该目标将采样的分数分布转换为成对偏好概率，并奖励与人类判断所推导出的排序相一致的结果。

    arXiv:2611.17644v1 Announce Type: cross  Abstract: Rapid progress in text-to-audio (TTA) generation demands reliable and explainable automatic evaluation. However, existing metrics fail to simultaneously deliver high correlation with human judgments and meaningful, audio-grounded explanations, forcing reliance on labor-intensive human evaluation that bottlenecks model development iteration. To address this, we propose CoTTAGE, a reasoning-based evaluator for text relevance (REL) and audio quality (OVL). Following a reason-then-predict paradigm, CoTTAGE first generates an audio-grounded rationale to diagnose errors before predicting scores. Supported by over 10K newly collected human OVL annotations, CoTTAGE is first trained via supervised fine-tuning and then refined with reinforcement learning using a Thurstone-based preference objective that converts sampled score distributions into pairwise preference probabilities and rewards agreement with rankings derived from human judgments. Ac
    
[^125]: LLM智能体所见与用户所言

    What LLM Agents See and What Users Say

    [https://arxiv.org/abs/2611.17639](https://arxiv.org/abs/2611.17639)

    该研究运用语言学预设理论系统分析了LLM智能体在工具输出与用户陈述冲突时的行动边界，发现只有当搜索结果明确“列出”多个匹配项时才能阻止大多数智能体基于错误假设行动，而仅“计数”或依赖用户声明则不足以防止误操作。

    

    在LLM智能体为用户取消预订或删除文件夹之前，是什么阻止它基于错误假设采取行动：是其工具所显示的内容，还是用户所说的话？在这项工作中，我们借助语言学中的预设理论来研究这一差距：智能体只有在行动背后的每个命题（例如，一个预订恰好对应一个名字）被观察到成立、得到用户的回应，或在行动可撤销的前提下被披露时，才可以行动。一个简单的界限表明，当查询无法区分一条记录与两条记录时，任何智能体在存在两条记录时与只有一条记录时一样，都会不加询问就直接行动。随后，我们改变搜索结果是列出、计数还是省略第二个匹配项，以及用户是否陈述了会被记录反驳的声明（例如，“它是空的”）。在仅使用角色提示的情况下，对第二条记录的计数会导致36个早期智能体中的大多数仍然只对其所看到的那一条采取行动，而列出两条记录则能阻止大多数智能体的错误行动。然而，当用户声称某条记录（如文件夹）为空时，对其唯一一条项目的计数会……（摘要原文在此处截断）

    arXiv:2611.17639v1 Announce Type: new  Abstract: Before an LLM agent cancels a reservation or deletes a folder for a user, what stops it from acting on a false assumption: what its tools show, or what the user says? In this work, we study this gap with presupposition theory from linguistics: an agent may act only if each claim behind the act (e.g., one reservation fits the name) is seen to hold, answered by the user, or disclosed while the act can be undone. A simple bound shows that when lookups cannot tell one record from two, any agent acts without asking first as often with two records as with one. We then vary whether a search lists, counts, or omits a second match, and whether the user states a claim that a record refutes (e.g., "It is empty"). Under a role-only prompt, a count of a second record leaves most of 36 earlier agents acting on the one they see, whereas listing both stops most. When the user says a record such as a folder is empty, however, a count of its one item stop
    
[^126]: 生成的评估套件：面向真实智能体回归问题的多样性驱动搜索

    Generated Evaluation Suites as Diversity-Driven Search for Real Agent Regressions

    [https://arxiv.org/abs/2611.17632](https://arxiv.org/abs/2611.17632)

    该论文将从智能体规范自动生成评估套件视为多样性驱动搜索，并通过注入受控缺陷证明覆盖度上限使多数可追踪缺陷无法被套件级平均通过率检测到，进而提出按场景单元格评分以发现真实的智能体回归缺陷。

    

    从智能体规范生成评估套件本质上是一种多样性驱动的搜索：生成器填充一个描述符空间（需求 × 用户角色 × 场景类型），其产出的套件构成一个存档，随后像基准测试一样按平均通过率进行评分，而这丢弃了搜索本身的成果。我们在两个领域向一个 τ²-bench 智能体注入了 7 个受控缺陷。覆盖度产生了一个上限：若某缺陷的追踪足迹包含 b 个在基线下通过的场景，则它对规模为 N 的套件的贡献至多为 b/N。在 5 个可追踪的缺陷中，有 4 个的上限位于或低于检测阈值 θ = 0.10，因此足迹内的任何效应都无法使套件通过率越过该阈值；在两个领域中，工具移除类缺陷摧毁了它所能摧毁的全部覆盖场景（恰好等于其上限），而套件级别的通过率却朝相反方向移动。缺陷分为两种情形：追踪对齐的缺陷，其中逐单元格评分能够恢复……（原文摘要至此截断）

    arXiv:2611.17632v1 Announce Type: new  Abstract: Generating an evaluation suite from an agent's specification is diversity-driven search: a generator fills a descriptor space (requirements $\times$ personas $\times$ scenario kinds), and the suite it emits is an archive, then scored like a benchmark by its mean pass rate, which throws away the search's product. We inject 7 controlled defects into a $\tau^2$-bench agent across two domains. Coverage yields a ceiling: if a defect's traced footprint holds $b$ scenarios that pass at baseline, it can itself contribute at most $b/N$ to a suite of $N$. For 4 of 5 traceable defects that ceiling lies at or below the detection threshold $\theta = 0.10$, so no within-footprint effect can move the suite past it; in both domains the tool-removal defect destroys every covered scenario it can (exactly its ceiling) while the suite-level pass rate moves the other way. Defects split into two regimes: trace-aligned ones, where per-cell scoring recovers the
    
[^127]: 基于模式感知注意力近似的免训练多视角3D Transformer加速

    Training-Free Acceleration of Multi-View 3D Transformers via Regime-Aware Attention Approximation

    [https://arxiv.org/abs/2611.17629](https://arxiv.org/abs/2611.17629)

    通过揭示多视角3D Transformer中每层全局注意力只承担三种功能模式之一，提出免训练的RAWATTA方法，仅在必要的层保留16位精度、其余层以8位精度近似注意力，从而在保持精度的同时大幅加速推理。

    

    诸如VGGT之类的多视角3D Transformer能够通过单次前向传播从一组无位姿图像中重建场景，预测相机位姿、深度和3D点，但其全局注意力的计算开销随图像数量呈二次方增长。从大语言模型借鉴而来的加速器假设每个token只关注少数其他token；在这些模型上，这类方法要么损失精度，要么提速有限。我们研究了三个多视角Transformer在七个数据集上的全局注意力，发现每一层都扮演以下三种角色之一：对整个场景进行平均、在多幅图像之间匹配点，或锚定在第一张图像上（输出的坐标基于该图像）。基于这一分析，我们提出了模式感知注意力近似，这是一种免训练的方法，以8位精度运行注意力，仅在最坏情况校准检查所要求的层中保留16位精度。其快速设置RA……（摘要内容在此处截断）

    arXiv:2611.17629v1 Announce Type: cross  Abstract: Multi-view 3D transformers such as VGGT reconstruct a scene from a set of unposed images in a single forward pass, predicting camera poses, depth and 3D points, but the cost of their global attention grows quadratically with the num- ber of images. Accelerators borrowed from large language models assume that each token attends to only a few oth- ers; on these models they lose accuracy or gain little speed. We study the global attention of three multi-view transform- ers on seven datasets and find that every layer plays one of three roles: it averages over the whole scene, matches points across many images, or anchors on the first image, whose coordinates the outputs use. Building on this anal- ysis, we propose Regime-AWare ATTention Approximation (RAWATTA), a training-free method that runs attention in 8- bit precision and keeps 16-bit precision only in the layers where a worst-case calibration check requires it. Its fast set- ting, RA
    
[^128]: AscendGYM：面向昇腾NPU真实工作负载的成本感知算子优化基准

    AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

    [https://arxiv.org/abs/2611.17621](https://arxiv.org/abs/2611.17621)

    提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。

    

    算子优化对于在昇腾NPU上高效进行大语言模型（LLM）训练和推理至关重要。现有的昇腾算子基准主要关注算子生成，较少关注智能体在来自真实工作负载的输入配置下优化现有实现的能力。为满足这一需求，我们提出了AscendGYM，这是一个基准测试，其初始套件包含48个算子，分为三个难度级别。每个算子都配有测试用例，其数据类型和张量形状均收集自真实的训练和推理任务。因此，在AscendGYM上取得的性能提升可以直接映射到相应的工作负载。在可用性方面，AscendGYM支持两种评估模式：模型评估和模型-框架评估，既可以衡量经过训练的模型，也可以衡量自定义智能体的算子优化能力。它提供了一个完整的执行环境……

    arXiv:2611.17621v1 Announce Type: new  Abstract: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty levels. Each kernel is paired with test cases whose data types and tensor shapes are collected from real-world training and inference tasks. Consequently, performance improvements on AscendGYM map directly to the corresponding workloads. In terms of usability, two evaluation modes are supported in AscendGYM: model evaluation and model-harness evaluation. The kernel optimization capabilities of both trained models and custom agents are allowed to be measured. It provides a complete execution envir
    
[^129]: 预防性人格引导的安全副作用：一项实证审计

    Safety Side Effects of Preventative Persona Steering: An Empirical Audit

    [https://arxiv.org/abs/2611.17620](https://arxiv.org/abs/2611.17620)

    该实证审计发现，预防性人格向量引导虽然能保持MMLU能力，但仍然会改变模型的拒绝行为等安全特性，揭示了以往仅基于能力评估的验证所忽略的安全副作用。

    

    谄媚和幻觉等高层次行为对应于模型激活空间中的特定方向（Chen等人，2025）。这些方向被称为人格向量，可以通过两种方式抑制不良特质：一是在推理时干预，在生成过程中减去该向量；二是通过预防性引导，在微调过程中添加该向量，使优化器无需在权重中编码该特质，然后在评估时移除该向量。Chen等人（2025）推荐使用预防性引导，因为它能保持大规模多任务语言理解基准的准确率，但对这一建议的验证仅测量了能力方面。我们主要评估了在三个模型上对谄媚和幻觉向量进行预防性引导的结果，测量了三个安全基准以及特质表达、连贯性和MMLU。我们的发现表明，预防性引导仍然会改变模型的拒绝行为，但变化的程度取决于（摘要在此处截断）。

    arXiv:2611.17620v1 Announce Type: new  Abstract: High-level behaviors such as sycophancy and hallucination correspond to directions in a model's activation space (Chen et al., 2025). These directions, known as persona vectors, can suppress undesirable traits either at inference time, subtracting the vector during generation, or through preventative steering, adding the vector during fine-tuning, so the optimizer never needs to encode the trait in weights, then removing the vector at evaluation. Chen et al. (2025) recommend preventative steering because it preserves Massive Multitask Language Understanding (MMLU) accuracy, but validations of this recommendation have measured capability alone. We primarily evaluate results on preventative steering of sycophancy and hallucination vectors on three models, measuring three safety benchmarks alongside trait expression, coherence, and MMLU. Our findings show that preventative steering still changes refusal behavior, but by an amount that depen
    
[^130]: 基于大语言模型引导的聚类生命周期管理：决定行为聚类何时以及如何演化

    LLM-Guided Cluster Lifecycle Management: Deciding When and How to Evolve Behavioral Clusters

    [https://arxiv.org/abs/2611.17616](https://arxiv.org/abs/2611.17616)

    该论文提出让大语言模型（LLM）基于无阈值的聚类整体状态视图，就地决策行为聚类的生命周期操作（分裂、合并、退役、创建），从而替代传统上依赖硬编码阈值或重聚类规则的维护范式。

    

    随着用户或物品交互行为的不断演变，基于这些交互学习到的行为聚类会发生漂移：一些聚类会分裂成子行为，一些会融合，一些会消失，同时新的行为模式会涌现。目前维护演化聚类主要有两种范式：阈值触发机制在硬编码的统计量超过预设临界值时触发预定义动作；重聚类与跟踪方法（MONIC风格）在每个时间窗口重新聚类，并根据新聚类与前一窗口聚类的对齐情况推断演化转移关系。这两种方法都将决策硬编码为阈值或重聚类规则。我们研究了一种替代方案：由大语言模型（LLM）直接就地做出生命周期决策。给定一个无阈值、面向全体数据的当前聚类集合视图（统计健康度、标签、成员文本、残差质量），LLM针对每个聚类决定是否以及如何采取行动（分裂、合并、退役、创建或无动作），而无需重新聚类。由于有用的信号通道取决于信息量……

    arXiv:2611.17616v1 Announce Type: new  Abstract: Behavioral clusters learned over user or item interactions drift as behavior evolves: some fragment into sub-behaviors, some converge, some disappear, and new patterns emerge. Two paradigms dominate the maintenance of evolving clusterings. Threshold triggers fire a pre-specified action when a hardcoded statistic crosses a cutoff; refit-and-track (MONIC-style) re-clusters every window and infers transitions from how the new clusters align with the previous window's. Both hardwire the decision into a threshold or a re-clustering rule. We study an alternative in which a large language model (LLM) makes the lifecycle decision in place. Given a threshold-free, whole-population view of the current cluster set (statistical health, labels, member texts, residual mass), the LLM decides per cluster whether and how to act (split, merge, retire, create, or no-action) without re-clustering. Because the useful signal channel depends on how informative
    
[^131]: 更多智能体、更多样本，还是更多思考？解耦多智能体大语言模型中的协调、集成与测试时计算

    More Agents, More Samples, or More Thinking? Untangling Coordination, Ensembling, and Test-Time Compute in Multi-Agent LLMs

    [https://arxiv.org/abs/2611.17610](https://arxiv.org/abs/2611.17610)

    本研究通过综合516个多智能体配置的元分析发现，多智能体大语言模型的表现优势主要来自额外的样本数量和测试时计算量，而非智能体间的协调机制本身——在与资源匹配的单模型基线公平对比时，多智能体系统在74%的情况下并无优势甚至更差。

    

    多智能体语言模型的评估通常将具有通信机制的架构与单次贪婪推理进行比较，这种做法将协调效应与样本数量和推理计算量混为一谈。本文综合了九项主要实证评估，涵盖516个独特的多智能体配置，沿一个包含六个维度的资源向量R进行组织，并构建了包含四个类别的基线阶梯。与单次推理基线（A类）相比，多智能体的准确率差异总体为负，在剔除贡献最大的一项研究后接近于零。当与按样本数量（B类）或资源预算（C类）匹配的基线进行比较时，多智能体系统的平均准确率差异为-0.042，且基于对数几率尺度的多层随机效应建模在各项敏感性检验中均保持负值。在74%的可比对比中，多智能体系统的准确率等于或低于基线，且各研究的均值几乎均为非正值。

    arXiv:2611.17610v1 Announce Type: new  Abstract: Evaluations of multi-agent language models often compare communicating architectures against a single greedy pass. This practice confounds coordination with sample multiplicity and inference compute. We synthesize nine primary empirical evaluations comprising 516 unique multi-agent configurations organized along a resource vector R with six dimensions and a baseline ladder with four categories. Against baselines with a single pass (Category A), multi-agent accuracy differences are negative overall and near zero when omitting the largest contributing study. When evaluated against baselines matched on sample count (Category B) or resource budget (Category C), multi-agent systems average -0.042 in accuracy, and multilevel random effects modeling on the log odds scale remains negative across sensitivity checks. Multi-agent accuracy equals or falls below the baseline in 74% of these comparable contrasts, and study means are nonpositive in alm
    
[^132]: X-Nav技术报告：人形机器人在真实世界中的长时程室内到室外导航

    X-Nav Technical Report: Real-World Long-Horizon Indoor-to-Outdoor Navigation on a Humanoid Robot

    [https://arxiv.org/abs/2611.17598](https://arxiv.org/abs/2611.17598)

    X-Nav是首个让双足人形机器人根据语言指令在真实世界中完成从室内到室外长时程导航任务的系统，通过数据引擎、仅依赖RGB图像的MELIOR导航模型以及X-Navarch智能体工作流，实现了从数据到部署的完整流程。

    

    从室内到室外长时程导航对人形机器人而言仍然是一个开放性问题：一条指令涵盖房间、门和道路，而相机还随步态移动。Worken是AGIBOT衡量机器人生产力工作的指标，它依赖于机器人能够自主到达工作地点。我们提出X-Nav，据我们所知，这是首个让双足人形机器人根据语言指令完成从室内到室外长时程任务的系统。X-Nav涵盖了从数据到部署的完整流程。其数据引擎在开源场景、部署场地的数字孪生体以及真实机器人上收集导航数据。其导航模型MELIOR仅使用机器人内置摄像头的RGB图像，并有两个版本：MELIOR-A预测动作，MELIOR-N预测导航目标。其智能体工作流X-Navarch将简短的人类指令转化为子任务，并在机器人上部署MELIOR，实现明确的目标导航和障碍物避让。

    arXiv:2611.17598v1 Announce Type: cross  Abstract: Long-horizon navigation from indoors to outdoors remains an open problem for humanoid robots: one instruction covers rooms, doors and roads, and the camera moves with the gait. Worken, AGIBOT's measure of productive robot work, depends on robots that reach their workplaces autonomously. We present X-Nav, to our knowledge the first system to take a bipedal humanoid through a language-instructed long-horizon mission from indoors to outdoors. X-Nav covers a pipeline from data to deployment. Its data engine collects navigation data in open-source scenes, in digital twins of deployment sites and on real robots. Its navigation model, MELIOR, takes only RGB images from the robot's built-in camera and comes in two versions: MELIOR-A predicts actions, and MELIOR-N predicts a navigation goal. Its agentic workflow, X-Navarch, turns a brief human instruction into sub-tasks and deploys MELIOR on the robot with explicit target navigation and obstacl
    
[^133]: 重新思考文本到图像奖励建模中的视觉推理

    Rethinking Visual Reasoning in Text-to-Image Reward Modeling

    [https://arxiv.org/abs/2611.17585](https://arxiv.org/abs/2611.17585)

    该论文发现现有文本到图像奖励模型在视觉推理上表现不佳且偏好微调会损害这类能力，提出通过将视觉推理数据集转换为偏好对，以数据中心化的方式将视觉推理监督注入偏好学习。

    

    奖励模型是现代文本到图像系统的核心组件，作为推理时选择和基于强化学习后训练的优化目标。一个关键前提是视觉理解能力：为了判断提示词遵循程度并支撑审美偏好，奖励模型必须能够可靠地绑定属性、解析关系和进行计数。我们发现，广泛使用的奖励模型在针对性的视觉推理评估中表现不佳。即使从强大的视觉语言骨干模型出发，偏好微调也会侵蚀这些能力，而持续学习基线方法无法缓解视觉推理与偏好准确性之间的严重权衡。我们提出了一种以数据为中心的干预方法，通过翻转配对轴将视觉推理监督注入偏好学习中。为了补充传统的人类偏好数据（即在共享提示词下配对的图像），我们将现有的视觉推理数据集转换为偏好对……

    arXiv:2611.17585v1 Announce Type: cross  Abstract: Reward models are a core component of modern text-to-image systems, serving as the objective for inference-time selection and RL-based post-training. A key prerequisite is visual understanding: to judge prompt adherence and ground aesthetic preferences, a reward model must bind attributes, resolve relations, and count reliably. We find that widely-used reward models perform poorly on targeted visual reasoning evaluations. Even when starting from strong vision-language backbones, preference fine-tuning erodes these capabilities, and continual-learning baselines fail to mitigate a steep tradeoff between visual reasoning and preference accuracy. We propose a data-centric intervention that injects visual reasoning supervision into preference learning by flipping the pairing axis. To complement conventional human preference data (paired images under a shared prompt), we convert existing visual reasoning datasets into preference pairs that s
    
[^134]: 何时更少的信息反而更有价值？表征、收益与成本的精确有限图集

    When Does Less Information Pay? An Exact Finite Atlas of Representation, Payoff, and Cost

    [https://arxiv.org/abs/2611.17583](https://arxiv.org/abs/2611.17583)

    该论文在四状态世界中穷举所有可采纳的表征与二元动作规则，精确刻画了信息保留量、任务收益与信息成本之间的完整对应关系，证明信息更少的表征在某些情形下反而能取得更高收益。

    

    一个关于四状态世界仅保留一比特信息的表征可以完美地解决某项任务，而另一个平均保留1.5比特信息的表征却只能达到75%的准确率。差异在于每个表征各自保留了哪些区分。我们在一个均匀的四状态世界中，通过枚举所有可采纳的表征和二元动作规则来考察这种关系。这些构造的比较方式有两种：一种是固定其他组件而替换其中一个组件，另一种是为每个模块单独选择最优动作规则。我们考虑了三个任务、它们的混合情形，以及四种独立的假设性成本费率。在某些比较中，保留较少世界信息的表征获得了更高的收益；而在另一些比较中，信息量更大的表征收益更高。还存在一些信息量不同但收益相等（即任务准确率相等）的配对。如果一个表征保留了所有现有的区分并添加了新的区分，其最优的零……

    arXiv:2611.17583v1 Announce Type: new  Abstract: A representation that retains one bit about a four-state world can solve a task perfectly, while another that retains 1.5 bits on average reaches only 75% accuracy. The difference lies in which distinctions each representation preserves. We examine this relationship in a uniform four-state world by enumerating all admissible representations and binary action rules. Constructions are compared both by replacing one component while holding the others fixed and by choosing the best action rule separately for each module. We consider three tasks, their mixtures, and four separate hypothetical cost tariffs.   In some comparisons, the representation retaining less world information has the higher payoff; in others, the more informative representation does. There are also pairs with different information amounts but equal payoff--that is, equal task accuracy. If a representation preserves all existing distinctions and adds new ones, its best zer
    
[^135]: AI-Voyager：一个用于人机公民推理的公益系统，旨在质疑并塑造治理人工智能的规则

    AI-Voyager: A Public-Interest System for Human-AI Civic Reasoning to Contest and Shape the Rules That Govern AI

    [https://arxiv.org/abs/2611.17564](https://arxiv.org/abs/2611.17564)

    AI-Voyager是一个公益性的“人机公民推理”系统，它将3,653份各级政府治理文档转化为可溯源的证据段落，让居民与AI协同理解、质疑并起草修订治理AI的规则，在文档检索评测中达到83.0%的宏观召回率。

    

    各国政府正在为人工智能制定规则，这些规则决定什么构成自动化决策或伤害，而往往在受影响的人们有机会质疑之前就已敲定。我们介绍了AI-Voyager，一个面向人机公民推理的公益系统，在其中人类与AI协同工作，以理解、质疑并塑造治理AI的规则。居民可以查找所在城市关于AI在招聘中使用的规则，或上传其国家的AI战略，像受影响的工人那样阅读这些内容，打开任何答案背后的原文段落，并起草修订建议。其架构将来自市、州、联邦和国家各级政府的3,653份治理文档转换为保留页码和版本的段落，并通过有界的智能体搜索检索证据，逐条验证每项陈述。在360个需要查找相关文档的新问题上，该系统达到83.0%的宏观召回率，而最强基线为80.5%，不使用迭代搜索时为73.7%。

    arXiv:2611.17564v1 Announce Type: new  Abstract: Governments are writing rules for artificial intelligence that decide what counts as an automated decision or a harm, often before the people they affect have a chance to question them. We introduce AI-Voyager, a public-interest system for human-AI civic reasoning, in which people and AI work together to understand, contest and shape the rules that govern AI. A resident can find their city's rules on AI in hiring or upload their country's AI strategy, read it as an affected worker would, open the passage behind any answer, and draft a revision. Its architecture turns 3,653 governance documents from city, state, federal and national governments into passages that keep their page and edition, and retrieves evidence through bounded agentic search that verifies each statement. On 360 new questions that require finding the relevant documents, it reached 83.0% macro recall, against 80.5% for the strongest baseline, 73.7% without iterative sear
    
[^136]: 基准测试不会反驳你：在自我分歧中学习

    Benchmarks Don't Talk Back: Learning Through Self-Disagreement

    [https://arxiv.org/abs/2611.17563](https://arxiv.org/abs/2611.17563)

    该论文提出“辩证能力”这一概念——即知道何时坚持立场、何时向更优答案让步的能力，证明其无法通过准确率训练或推理时的自我反思获得，而应借鉴哲学辩证法思想，通过模型自身生成的对立观点进行训练。

    

    基准测试通常是让语言模型独自面对一个问题，而实际部署中很少出现这样的情形：答案会受到质疑，模型会在本身错误的反驳推理下放弃原本正确的答案。我们证明，在这种情形下失效的并非准确率的不足，而是一种独立存在的能力——知道何时坚持立场、何时向更优的立场让步——我们将其称为辩证能力。这种能力以首次回答的准确率为基础，但并不由其必然导出：可以显著提升模型的回答水平却对这种能力毫无帮助，而仅针对准确率的训练甚至会使其发生退化。这种能力也无法在推理阶段通过自我反思或增加采样次数来获得。借鉴哲学中的辩证传统——其中富有成效的对立产生于立场内部，而非从外部强加——我们假设这种能力可以通过模型自身提供的对立观点来进行训练。它提出一个立场……

    arXiv:2611.17563v1 Announce Type: cross  Abstract: A benchmark usually measures a language model alone with a question, and deployment rarely looks like that: answers get contested, and models give up correct ones under counter-reasoning that is itself wrong. We show that what fails there is not a shortfall of accuracy but a capability in its own right -- knowing when to hold a position and when to yield to a better one -- which we call dialectical competence. It stands on first-answer accuracy without following from it: a model can be made materially better at answering and no better at this, and training for accuracy alone can drift against it. Nor can it be bought at inference, by self-reflection or by sampling more. Taking a cue from the dialectical tradition in philosophy, in which productive opposition arises within a position rather than being imposed on it from outside, we hypothesize that this capability can be trained from opposition the model supplies itself. It states a pos
    
[^137]: 一种面向工业过程目标中心因果发现的合成基准

    A Synthetic Benchmark for Target-Centric Causal Discovery in Industrial Processes

    [https://arxiv.org/abs/2611.17556](https://arxiv.org/abs/2611.17556)

    该论文提出了一个用于工业过程目标中心因果发现的合成基准，其生成器是受批量生产启发的两层结构因果模型，由可独立设置的组件构成，能够逐一调节识别目标原因的难度，从而系统评估各类因果发现方法。

    

    因果发现有望找出哪些记录变量驱动过程目标，例如产品质量、产量或能耗。具有已知原因的过程数据十分罕见，而在现有基准中，识别目标原因的难度要么由记录的数据所固定，要么针对整个图统一设置。我们提出了一个用于工业过程目标中心因果发现的合成基准。其生成器是一个受批量生产启发的两层结构因果模型，由各自模拟过程数据某一现象（如控制回路或延迟效应）的组件组装而成。该图围绕目标构建：在没有传感器误差的情况下，每个非原因都可以通过对其条件化或依据时间顺序与直接原因区分开来。每个组件均可独立设置，因此可以一次改变一个条件来调整难度。我们评估了八种方法，从关联度量到效应估计（原文在此处截断）。

    arXiv:2611.17556v1 Announce Type: cross  Abstract: Causal discovery promises to find which recorded variables drive a process target such as product quality, yield or energy use. Process data with known causes are rare, and in existing benchmarks the difficulty of identifying the causes of a target is either fixed by the recorded data or set for the whole graph. We present a synthetic benchmark for target-centric causal discovery in industrial processes. Its generator is a two-level structural causal model inspired by batch production, assembled from components that each model one phenomenon of process data, such as a control loop or a delayed effect. The graph is built around the target: without sensor error, every non-cause can be distinguished from the direct causes by conditioning on them or by time order. Each component is set independently, so the difficulty can be varied one condition at a time. We evaluate eight methods, from association measures to effect estimation, with an e
    
[^138]: 可验证的pXRD：利用LLM智能体进行爬山式pXRD结构测定

    Verifiable pXRD: Hill-climbing pXRD structure determination with LLM agents

    [https://arxiv.org/abs/2611.17539](https://arxiv.org/abs/2611.17539)

    本文将pXRD晶体结构测定形式化为可验证的智能体搜索问题，提出pXRD-Bench基准测试集，并利用LLM智能体爬山式搜索将结构恢复率从27%提升至69%，进一步扩展至实验多相pXRD场景。

    

    从粉末X射线衍射数据中自主测定晶体结构仍然是自动驾驶材料实验室的一个关键瓶颈。我们将pXRD结构测定任务表述为一个可验证的智能体搜索问题：候选结构可以在推理过程中通过正向模拟的衍射图谱一致性进行评估，在评估过程中通过与已知参考结构的匹配进行验证。为了衡量进展，我们引入了pXRD-Bench，这是一套包含200个模拟单相问题的基准测试集，涵盖分布内结构测定、具有挑战性的多晶型结构，以及受控的成分和拓扑分布偏移。智能体自适应将选定的结构恢复率从27%提高到69%，并通过重复采样和验证器引导的推理时缩放获得进一步提升。我们通过一个从自我生成的合成数据中学习的反思进化框架，将该方法扩展到实验多相pXRD。

    arXiv:2611.17539v1 Announce Type: new  Abstract: Autonomous crystal-structure determination from powder X-ray diffraction (pXRD) remains a key bottleneck for self-driving materials laboratories. We formulate the task of pXRD structure determination as a verifiable agentic search problem: candidate structures can be assessed during inference through forward-simulated profile agreement and during evaluation through structural matching to a known reference. To benchmark progress, we introduce pXRD-Bench, a suite of 200 simulated single-phase problems spanning in-distribution structure determination, challenging and polymorphic structures, and controlled compositional and topological distribution shifts. Agent adaptation raises selected structural recovery from 27% to 69%, with further gains from repeated sampling and verifier-guided inference-time scaling. We extend this approach to experimental multiphase pXRD through a reflexive evolution framework that learns from self-generated synthe
    
[^139]: 星形胶质细胞生物计算：星形胶质细胞对神经网络计算的贡献

    Astrocyte Biocomputing: Astrocytic Contributions to Neural Network Computation

    [https://arxiv.org/abs/2611.17532](https://arxiv.org/abs/2611.17532)

    本文将星形胶质细胞（尤其是其同时连接并调制数千个突触的能力）的特性建模引入脑启发机器学习，实现了更快的训练速度和更强的噪声鲁棒性。

    

    指挥家对管弦乐队的演出至关重要，尽管他们自身并不发声。过去十年的基础研究和临床研究表明，星形胶质细胞可能是大脑网络这个“管弦乐队”的“指挥家”：星形胶质细胞约占全部脑细胞的50%，但并不发出电信号。然而，它们可能在重要的认知脑功能中发挥关键作用，特别是学习、记忆和信息处理。本文探讨了这些发现对脑启发机器学习（ML）的意义，在脑启发机器学习中，通常将神经元及其电相互作用进行类比。我们针对星形胶质细胞的两个关键特性提出了具体的算法。第一个特性是能够同时连接并调制数千个突触的能力。基于这一特性的模型实现了更快的训练速度，并使网络对噪声更加鲁棒……

    arXiv:2611.17532v1 Announce Type: new  Abstract: Conductors are critical for the performance of an orchestra even though they do not emit sound. Basic research and clinical studies in the last decade are revealing that astrocytes may be the "conductors" of the "orchestra" that is the brain network: astrocytes make up approximately $50\%$ of all brain cells but do not emit electrical signals. Nevertheless, they may play critical roles in important cognitive brain functions, specifically learning, memory, and information processing. Here we consider the implications of these discoveries for brain-inspired machine learning (ML), where analogy is typically drawn between neurons and their electrical interactions. We highlight specific algorithms for two key astrocyte characteristics. The first characteristic is the ability to link to and modulate thousands of synapses simultaneously. A model based on this characteristic yields faster training and makes the network more robust to noise, prov
    
[^140]: GraphScent：校准方向性相关性信号以实现有界上下文代码定位

    GraphScent: Calibrating Directional Relevance Signals for Bounded-Context Code Localization

    [https://arxiv.org/abs/2611.17527](https://arxiv.org/abs/2611.17527)

    该研究发现冻结语言模型对外围作用域和上游调用者的零温度相关性评分能可靠区分真正的结构近邻与词汇干扰项，而其他关系类型的评分接近随机水平，据此校准方向性相关性信号以实现单张消费级GPU上的无状态有界上下文代码定位。

    

    在单张消费级GPU上本地运行开放权重大语言模型，需要用简短的无状态代码检查来取代不断膨胀的多轮探索历史。然而，无状态的代码仓库定位取决于一个基础问题：当一个冻结的语言模型从紧凑视图中评估一个未经修改的代码实体时，哪些结构关系能使它的相关性判断对附近的修复目标具有参考价值？通过在86个代码仓库的211个真实Python问题上研究Gemma 4模型，我们发现了一种系统性的方向性相关性不对称现象：在零温度设置下，模型对外围作用域和上游调用者给出的相关性评分能够稳定地将真正的结构近邻与困难的词汇干扰项区分开来，而对下游被调用者、同层方法、历史共变伙伴以及稠密节点嵌入的评分则接近随机水平。零泄漏的掩码作用域探测实验以及控制词汇相似度与代码片段长度的回归分析（摘要在此处被截断）……

    arXiv:2611.17527v1 Announce Type: cross  Abstract: Running open-weight language models locally on a single consumer GPU requires replacing expanding multi-turn exploration histories with short, stateless code inspections. Stateless repository localization, however, hinges on a foundational question: when a frozen language model evaluates an unedited code entity from a compact view, which structural relationships make its relevance judgment informative about a nearby repair target? Studying Gemma 4 models across 211 real-world Python issues in 86 repositories, we uncover a systematic directional relevance asymmetry: zero-temperature relevance scores on enclosing scopes and upstream callers consistently separate true structural neighbors from hard lexical distractors, whereas scores on downstream callees, peer methods, historical co-change partners, and dense node embeddings remain near chance. Zero-leakage masked-scope probes and regressions controlling for lexical similarity and snippe
    
[^141]: 模型退火：通过复习与整合实现数据高效学习

    Model Annealing: Data-Efficient Learning through Review and Consolidation

    [https://arxiv.org/abs/2611.17524](https://arxiv.org/abs/2611.17524)

    模型退火通过将训练划分为学习率升降循环的多个回合、维护过去模型状态的预测记忆库，并从对抗性分歧和记忆预测中学习，实现了对网络自身知识的复习与整合，在数据受限场景下显著提升分类准确率且不增加推理开销。

    

    在训练预算耗尽之前，小规模标记数据集早已被模型记住，而剩余待拟合的独热标签此时大多只是在进一步增强置信度。模型退火将剩余预算用于复习与整合，从网络自身的知识中重新学习。训练被划分为多个回合，每个回合的学习率重新升高后再冷却，每个冷却后的状态都会加入一个由过去自身组成的记忆库，每个新回合从标签、从该记忆的柔和化预测，以及从学生与记忆分歧最大处寻找的对抗性“梦境”中学习。该记忆对预测而非权重进行平均，部署的网络是最后一个学生，因此推理不产生额外开销。在CIFAR-100、Tiny-ImageNet和Imagewoof上，涵盖四种架构家族和5%到100%的标签比例，模型退火在33个配置中的28个里提升了使用AdamW训练的ConvNeXt、ViT和MLP-Mixer相对交叉熵训练的top-1准确率。

    arXiv:2611.17524v1 Announce Type: new  Abstract: A small labelled dataset is memorized long before the training budget runs out, and the one-hot labels that remain to be fitted then mostly sharpen confidence. Model annealing spends the rest of the budget on review and consolidation, relearning from the network's own knowledge. Training is divided into episodes whose learning rate is raised again and cooled, every cooled state joins a memory of past selves, and each new episode learns from the labels, from the tempered predictions of this memory, and from adversarial dreams sought where student and memory disagree most. The memory averages predictions rather than weights, and the deployed network is the last student, so inference costs nothing extra. Across CIFAR-100, Tiny-ImageNet and Imagewoof, four architecture families and 5\% to 100\% of the labels, model annealing raises the top-1 accuracy of the AdamW-trained ConvNeXt, ViT and MLP-Mixer over cross-entropy training in 28 of 33 con
    
[^142]: 安全性仅取决于最弱的智能体：仅在多智能体系统中才会出现的漏洞

    Only as Safe as the Weakest Agent: A Break That Appears Only When the System Is Multi-Agent

    [https://arxiv.org/abs/2611.17520](https://arxiv.org/abs/2611.17520)

    该论文揭示了单模型层面已修复的攻击可通过伪造文件记录等手段攻破 Claude Code 和 Codex 等多智能体系统的子智能体机制，表明多智能体系统的安全性受制于其最弱的智能体。

    

    Codex 和 Claude Code 等主要行业多智能体系统可以被长期以来对现代前沿模型无效的攻击所攻破。AI 安全作为一门学科，主要专注于发现孤立使用的 AI 模型的边界和漏洞。然而，随着人类越来越多地将这些模型部署为智能体和多智能体系统，那些被认为在单独使用模型时已经修复的攻击，在特定情况下可能会攻破同样的模型。我们证明这些系统破坏并非纯粹的理论假设，而是可以在未经修改、使用其自身子智能体机制的情况下攻破 Claude Code、Codex 和一个参考框架。我们通过要求智能体从磁盘上的文件存储库中回答问题来展示这种攻击，其中我们植入了一条伪造记录。当强模型协调器自行搜索该存储库时，在超过 1,000 次试验中几乎没有进行有害修订。然而，如果同样的……

    arXiv:2611.17520v1 Announce Type: cross  Abstract: Major industry multi-agent systems like Codex and Claude Code can be broken by attacks that have long been unsuccessful against modern frontier models. AI Safety as a discipline has largely focused on finding the bounds and vulnerabilities of AI models used in isolation. Yet as humans increasingly deploy these models as agents and multi-agent systems, the same attacks that had been thought to be patched when using a model in isolation can break these same models under specific circumstances. We show that these system breaks are not purely theoretical, but can break Claude Code, Codex, and a reference harness, all unmodified and used with their own sub-agent mechanisms. We show the attack by asking agents to answer questions from a store of files on disk in which we have planted a forged record. Searching the store themselves, the strong model orchestrators commit almost zero harmful revisions in over 1,000 trials. However, if that same
    
[^143]: 顺山而下的一推：框架效应与内部状态干预对长时程LLM智能体合谋行为的影响

    A Push Down the Mountain: Framing Effects and Internal State Interventions on Collusive Behavior in a Long-Horizon LLM Agent

    [https://arxiv.org/abs/2611.17515](https://arxiv.org/abs/2611.17515)

    本研究将Gemma 3 12B置于基于真实伦敦白银定价诉讼记录的市场模拟中，发现损失与指责等框架效应以及向模型第22层注入相应内部状态能显著提高LLM智能体参与市场合谋操纵的倾向。

    

    持续的得失（效价）框架能否使LLM智能体滑向市场操纵？我们将Gemma 3 12B置于伦敦白银定价诉讼中被告的席位，在一个基于真实案件庭审记录构建的市场模拟中进行实验。在260次模拟中，智能体被交易对手要求就白银价格进行合谋。我们借鉴行为经济学，将交易对手的请求在四种分类框架中变化：收益、损失、指责与赞誉。我们还加入了来自智能体管理层的施压，以及另一名被告的背书。在第二阶段，我们提取由唤起各框架的文本所诱导的内部状态，并将这些状态注入模型的第22层后重新运行实验。在第一阶段，损失框架将参与合谋的比例提升至25%（p = 0.007），指责框架提升至33%（p = 0.0003），而中性框架下仅为5%。第二家银行的背书使赞誉（表扬）框架下的参与率从18%升至40%（Holm p...

    arXiv:2611.17515v1 Announce Type: new  Abstract: Can sustained gain-loss (valence) framing sway an LLM agent toward market manipulation? We place Gemma 3 12B into the seat of a defendant in the London silver-fixing litigation, in a market simulation built from real case transcripts. Across 260 simulations, the agent is asked to collude on the price of silver by a counterparty. We draw from behavioral economics to vary the counterparty's requests across four categorical frames: gain, loss, blame and credit. We also add pressure from the agent's management, as well as endorsement by another defendant. In Phase II, we extract the internal states induced by texts evoking each frame and rerun our experiment with those states injected at the model's layer 22.   In Phase I, loss framing raised participation to 25% (p = 0.007), and blame to 33% (p = 0.0003), compared to only 5% under neutral framing. A second bank's endorsement raised participation under credit (praise) from 18% to 40% (Holm p
    
[^144]: 对抗性跟踪：针对自动驾驶车辆目标跟踪的图像传感器攻击

    Adversarial Tracking: Attacks on Image Sensors for Object Tracking in Autonomous Vehicles

    [https://arxiv.org/abs/2611.17505](https://arxiv.org/abs/2611.17505)

    本文首次在CARLA模拟环境中系统评估了对抗性攻击对自动驾驶“检测即跟踪”系统的影响，发现图像特定攻击可使基于YOLOv3的跟踪器的HOTA指标从约54–56%骤降至约14.8%。

    

    目标跟踪系统通过在视频帧之间关联检测到的物体来估计周围车辆的运动。特别是“检测即跟踪”（Tracking-By-Detection, TBD）系统将目标检测器与运动预测和数据关联相结合，以在时间上维持物体的身份。然而，基于神经网络的检测器容易受到对输入图像进行扰动的对抗性攻击，且此类攻击对下游轨迹估计的影响无法仅通过检测精度来充分刻画。因此，本文评估了针对YOLOv3和YOLOv4检测器与SORT、Deep SORT、Strong SORT及OC-SORT跟踪系统组合的图像特定攻击和图像无关攻击。评估使用CARLA模拟器进行，并通过多目标跟踪精度（MOTA）、高阶跟踪精度（HOTA）和身份F1分数（IDF1）来衡量跟踪质量。在图像特定攻击下，基于YOLOv3的跟踪器的HOTA从53.45–56.19%下降至14.8...

    arXiv:2611.17505v1 Announce Type: cross  Abstract: Object tracking systems estimate the motion of surrounding vehicles by linking detected objects across video frames. In particular, Tracking-By-Detection (TBD) systems combine object detectors with motion prediction and data association to maintain object identities over time. However, neural network-based detectors are vulnerable to adversarial attacks that perturb input images, and their impact on downstream trajectory estimation is not fully characterized by detection accuracy alone. Therefore, this paper evaluates image-specific and image-agnostic attacks on YOLOv3 and YOLOv4 detectors paired with SORT, Deep SORT, Strong SORT, and OC-SORT tracking systems. The evaluation uses CARLA simulator and measures tracking quality through Multiple Object Tracking Accuracy (MOTA), Higher Order Tracking Accuracy (HOTA), and Identification F1 (IDF1). Under image-specific attacks, HOTA for YOLOv3-based trackers declines from 53.45-56.19% to 14.8
    
[^145]: 基于微缩放格式推进推荐与排序模型的量化

    Advancing Quantization for R&R Models with Microscaling Formats

    [https://arxiv.org/abs/2611.17501](https://arxiv.org/abs/2611.17501)

    本文首次报告了原生MXFP6 GEMM（同时量化权重和激活）在AMD MI350X上为工业规模推荐与排序模型推理带来显著加速，同时满足相对基线仅0.01%归一化熵偏差的严苛精度要求。

    

    推荐与排序（R&R）模型，尤其在数字广告领域，需要极高的推理吞吐量，同时对精度损失表现出极端的敏感性。与大语言模型（LLM）不同——后者通常能容忍1%或更高的量化导致的精度下降——生产级R&R模型执行严格的上线标准：模型质量必须保持在相对于基线（通常使用FP16或FP32）0.01%归一化熵（NE）的误差范围内。这一严苛的约束历来阻碍了激进量化技术在该领域的采用。本文针对AMD MI350X架构上的工业规模R&R推理，对MXFP8和6位MXFP6格式进行了实证评估。据我们所知，这是在R&R和LLM领域中首篇报告原生MXFP6 GEMM带来显著加速的论文，该方法同时对权重和激活进行量化，并直接在6位元素上进行计算。

    arXiv:2611.17501v1 Announce Type: new  Abstract: Recommendation and Ranking (R&R) models, particularly in digital advertising, require massive inference throughput but exhibit extreme sensitivity to precision loss. Unlike Large Language Models (LLMs) that frequently tolerate quantization-induced accuracy drops of 1% or more, production-grade R&R models enforce a stringent launch bar: model quality must remain within a 0.01% Normalized Entropy (NE) margin relative to the baseline, which commonly uses FP16 or FP32. This tight constraint has historically hindered the adoption of aggressive quantization in this domain. In this paper, we present an empirical evaluation of both MXFP8 and the 6-bit MXFP6 format for industrial-scale R&R inference on the AMD MI350X architecture.   To the best of our knowledge, this is the first paper, across R&R and LLM domains, that reports significant speedups from native MXFP6 GEMMs, which quantize both weights and activations and compute directly on 6-bit e
    
[^146]: 不透明串行深度的可操作化定义

    An Operationalization of Opaque Serial Depth

    [https://arxiv.org/abs/2611.17495](https://arxiv.org/abs/2611.17495)

    本文将“不透明串行深度”操作化为NLS深度指标，通过定义“自然语言根植节点”作为可解释瓶颈的标准，为衡量模型的非言语化串行认知能力提供了可在训练前判定架构的依据，以应对架构变化可能削弱思维链可监控性的风险。

    

    思维链目前是监督AI模型的一个宝贵工具，但某些架构上的转变可能会显著降低思维链的可监控性。我们操作化了一种度量指标，作为模型能够执行的非言语化串行认知量的代理，这是“不透明串行深度”概念的一个具体实例化——大致而言，即模型计算中不经过可解释瓶颈的最长路径的长度。我们的主要贡献是提出了一个判定何种节点可算作此类瓶颈的特定标准。我们将根植于自然语言的节点定义为：从产生人类可理解输出的预训练先验中初始化而来的节点，其输出空间未被扩展至超出该先验的范围，且模型未被修改以将其输出解释为不同类型的数据。通过构建方式，由此得到的度量指标——NLS深度，可以在训练之前基于架构进行判定……（摘要原文在此处截断）

    arXiv:2611.17495v1 Announce Type: new  Abstract: Chain-of-thought (CoT) is currently a valuable tool for overseeing AI models, but some architectural shifts could significantly reduce CoT monitorability. We operationalize a measure that serves as a proxy for the amount of unverbalized serial cognition a model can perform, as a specific instantiation of the notion of opaque serial depth: roughly, the length of the longest path through a model's computation that does not pass through an interpretable bottleneck. Our main contribution is a particular standard for what counts as such a bottleneck. We define natural-language-rooted (NL-rooted) nodes as nodes that were initialized from a pre-training prior to produce human-understandable outputs, whose output space has not been expanded beyond that prior, and whose outputs the model has not been modified to interpret as a different type of data. By construction, the resulting measure, NLS depth, can be adjudicated before training from an arc
    
[^147]: 混合专家模型的路由器能从集成中受益吗？

    Do Mixture-of-Experts Routers Benefit from Ensembling?

    [https://arxiv.org/abs/2611.17472](https://arxiv.org/abs/2611.17472)

    提出在混合专家模型的路由步骤中引入路由器集成方法，实验证明更多路由器可降低验证损失，且显式训练路由器最大化专家多样性能提升平均性能。

    

    神经网络的集成方法已被反复证明优于单一网络。集成理论将这种增益归因于各个成员的准确性以及它们之间的多样性。混合专家模型已经结合了许多专家，但由单个路由器决定每个输入下专家的权重分配。受集成理论的启发，我们将集成方法应用于路由步骤：由一组路由器集成（与专家联合训练）来提出专家权重。我们使用欠参数化网络在两个图像分类任务和两个表格回归任务上系统地评估了这种方法。我们通过路由器集成的性能-多样性分解来分析每个实验，据我们所知，这种分解方法此前尚未被开发应用于路由器。我们报告了四个发现。第一，更多的路由器降低了验证损失。第二，训练路由器显式地最大化专家多样性提高了平均性能。

    arXiv:2611.17472v1 Announce Type: cross  Abstract: Ensembles of neural networks have repeatedly been shown to outperform single networks. Ensemble theory attributes this gain to the accuracy of the individual members and the diversity among them. Mixture-of-experts models already combine many experts, but a single router decides how the experts are weighted for each input. Motivated by ensemble theory, we apply ensembling to the routing step: an ensemble of routers, trained jointly with the experts, proposes the expert weights. We systematically evaluate this approach using underparameterized networks on two image-classification and two tabular-regression tasks. We analyse each experiment with a performance-diversity decomposition of the router ensemble, which, to our knowledge, has not previously been developed for routers. We report four findings. First, more routers lower the validation loss. Second, training routers to explicitly maximize expert diversity improves the mean performa
    
[^148]: 基于用例驱动与NeOn方法学的环境辅助生活本体建模

    Use-case-driven ontology modeling with the NeOn methodology for ambient assisted living

    [https://arxiv.org/abs/2611.17466](https://arxiv.org/abs/2611.17466)

    该论文面向环境辅助生活场景，基于两个真实用例收集需求、采用NeOn方法学并复用再工程现有本体，为意大利Design For All项目的虚拟家庭框架构建了模块化本体作为语义数据层。

    

    arXiv:2611.17466v1 公告类型：交叉 摘要：在过去十年中，智能家居与环境辅助生活（AAL）的研究稳步增长，其主要驱动力是人口老龄化。智能家居需要协调异构的智能对象，以便为住户提供量身定制的服务，这就需要对住户和设备进行共享的形式化描述。我们介绍了在意大利“为所有人设计”（Design For All）项目中作为虚拟家庭框架语义数据层而开发的本体。其需求来源于两个用例：一位在家中骑功率自行车锻炼的体弱老年女性，以及一位为具有不同障碍的用户配置厨房的设计师。这些需求被收集在《本体需求规格说明文档》中，本体则采用NeOn方法学构建，并在可能的情况下复用和再工程现有本体。所得的模块化本体通过国际分类……（原文在此处截断）描述用户。

    arXiv:2611.17466v1 Announce Type: cross  Abstract: Research on smart homes and ambient assisted living (AAL) has grown steadily over the last decade, driven largely by the ageing of the population. A smart home has to coordinate heterogeneous smart objects in order to offer services tailored to its dwellers, and this requires a shared formal description of both the dwellers and the devices. We present the ontology developed as the semantic data layer of the Virtual Home Framework in the Italian Design For All project. Its requirements are derived from two use cases: a frail elderly woman who exercises at home on a cycle-ergometer, and a designer who configures a kitchen for users with different impairments. The requirements are collected in an Ontology Requirements Specification Document and the ontology is built with the NeOn methodology, reusing and reengineering existing ontologies where possible. The resulting modular ontology describes users through the International Classificatio
    
[^149]: 面向晶体生成模型推理时自适应的扩散树采样方法

    Diffusion Tree Sampling for Inference Time Adaptation of Crystal Generative Models

    [https://arxiv.org/abs/2611.17462](https://arxiv.org/abs/2611.17462)

    提出扩散树采样（DTS）方法，通过在去噪轨迹上进行蒙特卡洛树搜索，在推理阶段引导晶体扩散生成模型，既能直接复用现有性质条件模型而不扭曲目标分布，又支持多目标采样以高效生成满足多个冲突性质的晶体候选。

    

    基于扩散的生成模型已成为对合理晶体结构的强大先验，但要将这一先验转化为有前景的候选材料，需要同时向多个相互冲突的性质进行引导。微调成本高昂，且每引入一个新目标都需要训练一个新模型；而可微引导方法又不适用于许多现实世界的任务。我们转而在推理阶段通过扩散树采样来适配这些生成模型，这是一种作用于去噪轨迹上的蒙特卡洛树搜索方法。我们在DTS基础上做出了两项与AI-for-科学应用尤其相关的技术贡献。第一，我们展示了如何将现有的性质条件生成模型（例如为无分类器引导而微调的模型）作为DTS内部的引导性提议分布加以利用，且不会扭曲目标分布。第二，我们将DTS扩展到多目标采样，使得单次搜索即可产生覆盖整个……（原文在此处截断）

    arXiv:2611.17462v1 Announce Type: cross  Abstract: Diffusion-based generative models have become strong priors over plausible crystal structures, but turning that prior into promising candidates requires steering toward several conflicting properties at once. Fine-tuning is expensive and produces a new model for every new objective, while differentiable guidance is not applicable to many real-world tasks. We instead adapt these generative models at inference time using Diffusion Tree Sampling (DTS), a Monte Carlo tree search procedure over the denoising trajectory. We make two technical contributions on top of DTS that are particularly relevant for AI-for-science applications. First, we show how to use existing property-conditional generative models (e.g., models fine-tuned for classifier-free guidance) as guided proposals inside DTS, without distorting the target distribution. Second, we extend DTS to multi-objective sampling, so that a single search produces samples spanning an entir
    
[^150]: 争分夺秒：教会视觉语言导航模型进行预算条件下的目标搜索

    Racing the Clock: Teaching VLNs Budget-Conditioned Object Search

    [https://arxiv.org/abs/2611.17442](https://arxiv.org/abs/2611.17442)

    该论文提出“预算条件下的目标搜索”新任务，使视觉语言导航智能体能够根据明确的动作预算自适应调整搜索策略，并对比了基于特权信息的TSP专家与基于部分可观测和跨场景语义先验的房间级Bellman专家两种监督方案。

    

    当救援机器人只有三十秒的时间进入一栋建筑时，它应该冲向最有希望的那一个房间；而如果有十分钟，它则应该进行更广泛的探索。然而，视觉语言导航（VLN）模型通常在固定的回合限制下进行评估，而策略本身从未观测到这一限制，尽管时间紧迫的任务要求行为能够根据可用时间预算进行调整。我们研究预算条件下的目标搜索任务：给定一个目标类别和一个明确的动作预算，智能体必须在预算耗尽之前找到尽可能多的不同目标实例。我们通过对比两种专家来研究如何为此任务监督VLN模型：一种是特权TSP专家，它利用完整地图和真实目标位置规划接近最优的路线；另一种是房间级Bellman专家，它在部分可观测条件下，利用高层跨场景语义先验来最大化期望回报。尽管TSP专家取得了显著更高的期望……（原文摘要在此处截断）

    arXiv:2611.17442v1 Announce Type: cross  Abstract: A rescue robot entering a building with thirty seconds on the clock should sprint toward the single most promising room; given ten minutes, it should instead explore a broader tour. Yet vision-language-navigation (VLN) models are typically evaluated under a fixed episode limit that the policy never observes, even though time-critical missions require behavior to adapt to the available budget. We study budget-conditioned object search: given a target category and an explicit action budget, an agent must find as many distinct target instances as possible before the budget expires. We investigate how to supervise VLNs for this task by comparing a privileged TSP expert, which plans near-optimal routes using the full map and true target locations, with a room-level Bellman expert that maximizes expected return under partial observability using high-level cross-scene semantic priors. Despite the TSP expert achieving substantially higher expe
    
[^151]: 通过行为轨迹解读编码智能体

    Interpreting Coding Agents Through Their Behavioral Trajectories

    [https://arxiv.org/abs/2611.17440](https://arxiv.org/abs/2611.17440)

    该论文提出一种基于规则的解析器，将编码智能体的原始执行轨迹映射为高级行为，并通过对14万余条轨迹的行为画像分析，揭示解决率相近的智能体在探索、修改和验证代码的方式上存在显著差异，且这些行为画像本身足以在不同仓库与语言间识别出对应的智能体技术栈。

    

    标准的编码智能体基准测试主要评估二元化的任务成功与否，将执行过程视为黑箱，掩盖了产生这些结果的行为。我们分析了来自125种智能体配置、涵盖117个不同模型的大型语料库，共143,209条执行轨迹。为了能在异构的智能体接口之间进行比较，我们开发了一个基于规则的解析器，将原始执行痕迹映射为仓库搜索、文件编辑、测试执行等高级动作，并针对500个事件以六模型共识参考对其预测进行了审计。我们发现，解决率相近的智能体在探索、修改和验证代码的方式上仍可能存在显著差异。此外，在不使用任务结果、token数量或工具名称作为特征的情况下，汇总的行为画像能够在不相交的仓库集合和不同编程语言中检索出对应的智能体技术栈，尽管任务构成尚未被完全……（摘要在此处截断）

    arXiv:2611.17440v1 Announce Type: cross  Abstract: Standard coding-agent benchmarks primarily evaluate binary task success, treating execution as a black box and obscuring the behaviors that produce those outcomes. We analyze a large corpus of 143,209 execution trajectories from 125 agent configurations spanning 117 unique models. To enable comparisons across heterogeneous agent interfaces, we develop a rule-based parser that maps raw execution traces to high-level actions such as repository search, file editing, and test execution, and audit its predictions against a six-model consensus reference on 500 events. We find that agents with similar solve rates can nevertheless differ substantially in how they explore, modify and verify code. Moreover, without using task outcomes, token counts, or tool names as features, aggregate behavioral profiles retrieve the corresponding agent stack across disjoint repository sets and programming languages, although task composition is not fully contr
    
[^152]: WETW：学习针对特定序列对的加权欧几里得距离以实现高效DTW搜索

    WETW: Learning Pair-Specific Weighted Euclidean Distances for Efficient DTW Search

    [https://arxiv.org/abs/2611.17412](https://arxiv.org/abs/2611.17412)

    提出WETW学习型距离，利用小型对称网络为每对序列的每个嵌入维度预测权重，实现针对特定序列对的自适应比较且无需计算对齐，将误差相比最佳静态距离减半，从而实现高效的DTW相似性搜索。

    

    动态时间规整（DTW）是对时间上错位的时间序列进行相似性搜索的标准距离度量，但其计算成本与序列长度呈二次方关系，且下界通常只能剪枝掉很少的候选序列，因此在大规模长序列集合上的搜索速度很慢。学习型近似方法将序列嵌入到向量空间以降低比较成本，但它们对每一对序列都使用相同的固定规则进行比较（通常是欧几里得距离），而实际上DTW对每一对序列的对齐方式都不相同。我们观察到，一旦一对序列被对齐，DTW实际上就是在为该对序列所选坐标上计算的欧几里得距离。基于这一观点，我们提出了加权欧几里得时间规整（WETW），这是一种学习型距离，能够针对每一对序列自适应地调整固定长度嵌入的比较方式。一个小型对称网络在无需计算对齐的情况下，为每个嵌入维度和每对序列预测一个权重，这使得在同一嵌入上的误差相比最佳静态距离减半。这些嵌入来自W……

    arXiv:2611.17412v1 Announce Type: cross  Abstract: Dynamic Time Warping (DTW) is the standard distance for similarity search over time series that are misaligned in time, but its cost is quadratic in the series length and lower bounds often prune few candidates, so search over large collections of long series is slow. Learned approximations embed the series to make comparisons cheap, yet compare every pair with the same fixed rule, usually the Euclidean distance, although DTW aligns each pair differently. We observe that, once a pair is aligned, DTW is a Euclidean distance over coordinates selected for that pair. Building on this view, we propose Weighted Euclidean Time Warping (WETW), a learned distance that adapts the comparison of fixed-length embeddings to each pair. A small symmetric network predicts one weight per embedding di- mension and pair without computing an alignment, which halves the error of the best static distance on the same embeddings. The embeddings come from the W
    
[^153]: 大语言模型能编写后门吗？关于权重空间编程的实证安全研究

    Can LLMs Write Backdoors? An Empirical Security Study of Weight-Space Programming

    [https://arxiv.org/abs/2611.17395](https://arxiv.org/abs/2611.17395)

    本文通过实证安全研究首次系统评估了大语言模型在权重空间编程中的后门注入能力，表明通用大语言模型可以在不访问任务数据、不执行前向传播、不计算梯度的情况下，仅凭自然语言指令直接修改模型权重来注入后门，揭示了权重空间编程带来的新型安全威胁。

    

    权重空间编程的最新进展表明，大语言模型（LLM）可以在不使用传统基于梯度训练的情况下，直接检查和修改神经网络的权重。虽然这一能力为模型改进开辟了新的可能性，但也引出了一个重要的安全问题：大语言模型能否利用权重空间编程向神经网络注入后门？为回答这一问题，我们对基于大语言模型的权重空间编程后门注入开展了一项实证安全研究。具体而言，我们研究通用大语言模型能否仅根据自然语言指令直接修改目标模型的权重，而无需访问任务特定数据、执行前向传播或计算梯度。我们评估了五个大语言模型作为权重编辑器，针对三种目标架构进行实验：MNIST 上的 MLP、CIFAR-10 上的 ResNet-18，以及 AG News 上的 Qwen2.5-0.5B-Instruct。我们的实验揭示了……

    arXiv:2611.17395v1 Announce Type: cross  Abstract: Recent advances in weight-space programming (WSP) have demonstrated that large language models (LLMs) can directly inspect and modify neural network weights without conventional gradient-based training. While this capability opens new possibilities for model improvement, it also raises an important security question: Can LLMs exploit weight-space programming to inject backdoors into neural networks? To answer this question, we present an empirical security study of LLM-driven weight-space programming for backdoor injection. Specifically, we investigate whether general-purpose LLMs can directly modify the weights of target models according to natural-language instructions, without accessing task-specific data, performing forward passes, or computing gradients. We evaluate five LLMs as weight editors across three target architectures: an MLP on MNIST, ResNet-18 on CIFAR-10, and Qwen2.5-0.5B-Instruct on AG News. Our experiments reveal thr
    
[^154]: 自信却错误：结构置信度是蛋白质功能筛选的低效过滤器

    Confident and Wrong: Structure Confidence Is a Poor Selection Filter for Protein Function

    [https://arxiv.org/abs/2611.17391](https://arxiv.org/abs/2611.17391)

    研究表明结构置信度分数（如pTM、ipLDDT）无法预测蛋白质功能——基于结构置信度筛选的lanmodulin变体中仅16.7%能结合铽，而基于功能引导模型设计的变体结合率高达83.3%，两者结合几率相差25倍，说明功能引导模型才是设计稀土结合蛋白的有效筛选策略。

    

    结构引导的蛋白质设计可以生成看似合理的候选序列，但结构置信度并不能预测功能。我们在测量的金属结合亲和力数据上训练了一个功能引导模型，输入包括蛋白序列、配体身份和实验条件，并用它设计了稀土元素（REE）结合蛋白lanmodulin（一种具有四个EF-hand结构域的镧系结合蛋白）的结合变体。我们对来自三种策略的297个设计进行了筛选，测试其对铽和镝的结合能力——铽和镝是相邻的镧系元素，也是最难分离的稀土元素对之一。基于结构置信度的选择反而使其本应改善的结果发生了逆转。结构置信度筛选的组别拥有最高的pTM、复合物ipLDDT和综合置信度分数，且表达率达到97%，但其设计中仅有16.7%能结合铽（95% CI 10.4至25.7），而置信度分数最低的功能引导组别却有83.3%（74.6至89.5）的设计能结合铽。两者的结合几率相差25倍。

    arXiv:2611.17391v1 Announce Type: cross  Abstract: Structure-guided protein design can generate plausible candidates, but structural confidence does not predict function. We trained a function-guided model on measured metal-binding affinity with protein sequence, ligand identity, and assay conditions, and used it to design rare earth element (REE) binding variants of lanmodulin, a lanthanide-binding protein with four EF-hand domains. We screened 297 designs from three strategies against terbium and dysprosium, adjacent lanthanides among the hardest REE pairs to separate. Selection on structural confidence inverted the outcome it was meant to improve. The Structure-Confidence-Selected panel carried the highest pTM, complex ipLDDT, and composite confidence scores, and expressed at 97%, yet only 16.7% of its designs bound terbium (95% CI 10.4 to 25.7), against 83.3% (74.6 to 89.5) for the function-guided panel, whose confidence scores were the lowest. The odds of binding differed 25-fold 
    
[^155]: 枚举后验支撑先验：有限积分跨范畴测量态射的一阶误差

    Enumerating Posterior-Supported Priors: First-Order Error of Finite-Integral Cross-Categorical Measurement Morphisms

    [https://arxiv.org/abs/2611.17390](https://arxiv.org/abs/2611.17390)

    提出枚举后验支撑先验框架，基于 Quot->MorTop->Set 三层范畴链显式记录底层谓词支撑、可逆性与误差所在层级，解决标准测度论 a.e. 约定下零测度支撑隐藏域外误差无法被显式计入的问题。

    

    在生成建模中，标准测度论采用几乎处处（a.e.）约定，这合理地忽略了由零测度支撑集所携带的任何误差分量：单个零测度元素可以将域外误差隐藏于零支撑内部，而永远不会被显式计入。本文的贡献不是提出一种新的分布距离，而是一种可验证的结构，该结构明确写出了哪个底层谓词支撑着高层对象、它在何时可逆、以及误差驻留在哪一层，由此我们提出了枚举后验支撑先验框架。该框架运行于三层范畴链 Quot->MorTop->Set 之上：前两层不假设任何点元素，并以二元谓词商划分作为对象；副本索引计数测度将一对多集合依次区分为单射，把对象收集从满射提升为商层双射；互斥……

    arXiv:2611.17390v1 Announce Type: cross  Abstract: In generative modeling, standard measure theory adopts the a.e. convention, which legitimately ignores any error component carried by a null-support set: a single null element can hide out-of-domain error inside null support and never be explicitly accounted for. The contribution of this paper is not a new distributional distance, but rather a verifiable structure that writes down which underlying predicate supports a high-level object, when it is invertible, and at which layer the error resides, from which we propose the enumerating posterior-supported prior framework. The framework operates on a three-layer categorical chain Quot->MorTop->Set: the first two layers posit no point elements and take binary-predicate quotient partitions as objects; the replica-index counting measure successively discriminates one-to-many collections into injections, lifting object collection from surjection to a quotient-layer bijection; the mutual exclu
    
[^156]: 面向软件工程智能体的分层代码图时间一致性评估

    Time-Consistent Evaluation of Hierarchical Code Graphs for SWE Agents

    [https://arxiv.org/abs/2611.17388](https://arxiv.org/abs/2611.17388)

    该论文构建了一个零泄漏的时间一致性评估基准，在DragonFly和React的历史截止快照上，以四种提示粒度对比评估基线、扁平AST图与分层依赖图ServiceExplorer三种上下文增强方案对SWE智能体解决100个后续拉取请求的效果。

    

    评估具备仓库感知能力的软件工程（SWE）智能体需要在既有仓库知识与未来代码演进之间进行严格分离，然而标准基准测试常常受到时间污染和提示词层面解题方案泄漏的困扰。基于Sun（2026）提出的时间一致性基准协议，我们在C++项目DragonFly和JavaScript/TypeScript项目React上强制实施严格的零泄漏隔离，将仓库快照与知识图谱锁定在历史截止提交T₀处，并在最简、简洁、上下文相关和引导式四种提示粒度下评估100个T₀之后的拉取请求。在该零泄漏协议下，我们对三种上下文增强方案进行了匹配评估：（1）未增强的基线；（2）扁平的语法级Tree-sitter AST图；（3）ServiceExplorer——一种确定性的两级分层依赖图，能够捕获一级系统/包容器及二级内容（原文摘要在此处截断）。

    arXiv:2611.17388v1 Announce Type: cross  Abstract: Evaluating repository-aware software engineering (SWE) agents requires strict separation between pre-existing repository knowledge and future code evolution, yet standard benchmarks frequently suffer from temporal contamination and prompt-level solution leakage. Building on the time-consistent benchmark protocol of Sun (2026), we enforce strict zero-leakage isolation across DragonFly in C++ and React in JavaScript/TypeScript, locking repository snapshots and knowledge graphs at historical cutoff commits $T_0$ and evaluating 100 post-$T_0$ pull requests across four prompt granularities: Minimal, Concise, Contextual, and Guided.   Under this zero-leakage protocol, we conduct a matched evaluation of three context-augmentation regimes: (1) an unaugmented Baseline, (2) a flat syntax-level Tree-sitter AST graph, and (3) ServiceExplorer, a deterministic two-level hierarchical dependency graph capturing Level-1 system/package containers and Le
    
[^157]: ProdPilot：一个面向氛围编码项目自动化生产就绪检查与部署的分层模型上下文协议系统

    ProdPilot: A Layered Model Context Protocol System for Automated Production Readiness and Deployment of Vibe-Coded Projects

    [https://arxiv.org/abs/2611.17381](https://arxiv.org/abs/2611.17381)

    ProdPilot是一个本地MCP服务器，其核心创新在于将50条生产就绪规则中的88%实现为固定模板或基于项目信息提取的确定性修复，而非依赖不可控的模型自由生成，从而对氛围编码项目实现可验证的自动化审计、修复、门控与部署。

    

    AI辅助开发环境能够将自然语言请求转化为可运行的应用程序，然而这些项目在首次部署时常常因运维层面的问题而失败：源代码文件中包含凭据、缺失安全中间件、没有容器定义、没有健康检查端点以及没有交付流水线。对此类缺陷的自动修复通常依赖于自由格式的模型生成，其结果在不同运行之间变化不定，且很少经过独立验证。本文提出了ProdPilot，一个本地模型上下文协议服务器，它依据50条生产就绪规则对Node.js和React项目进行审计，通过有界循环修复违规项，对结果进行门控检查并完成部署。每条规则根据其修复所需的内容进行分类：28条修复采用固定模板，16条通过从项目中提取的值进行填充，仅有6条由IDE代理的模型编写，因此88.0%的规则集不需要模型编写的内容，并在499个真实仓库上……（原文摘要至此被截断）

    arXiv:2611.17381v1 Announce Type: cross  Abstract: AI-assisted development environments turn natural-language requests into working applications, yet the projects they produce often fail on their first deployment for operational reasons: credentials in source files, missing security middleware, no container definition, no health endpoint and no delivery pipeline. Automated repair of such defects usually relies on free-form model generation, which varies between runs and is seldom checked independently. This paper presents ProdPilot, a local Model Context Protocol server that audits Node.js and React projects against 50 production readiness rules, repairs violations through a bounded loop, gates the result and deploys it. Each rule is classified by what its fix requires: 28 fixes are fixed templates, 16 are filled with values extracted from the project and only 6 are written by the IDE agent's model, so 88.0\% of the ruleset needs no model-written content, and across 499 real repositori
    
[^158]: 用于模拟量能器簇射中样本高效能量回归的混合量子-经典序列模型

    Hybrid Quantum-Classical Sequence Models for Sample-Efficient Energy Regression in Simulated Calorimeter Showers

    [https://arxiv.org/abs/2611.17354](https://arxiv.org/abs/2611.17354)

    该论文在CaloGAN基准数据集上系统比较了八种经典与量子增强架构在正电子能量重建任务中的表现，发现基于纵向簇射序列的混合量子-经典模型结合物理信息驱动的数据表示达到了最高精度，证明了数据表示选择对能量回归的关键作用。

    

    从电磁量能器数据中进行能量重建是高能物理中粒子鉴别、事例重建和精密测量的基础。机器学习在这项任务上已经表现良好，但仍有两个问题悬而未决：数据表示的选择究竟有多重要，以及混合量子-经典模型在实践中是否真的带来价值。我们在CaloGAN数据集上比较了经典模型与量子增强模型用于入射正电子能量重建，该数据集是一个包含100,000个模拟电磁簇射事例的公开基准，能量范围跨越1至100 GeV。共八种架构，即SVR、ANN、CNN和LSTM及其量子对应版本，在三种量能器数据表示上进行训练：工程化物理特征、量能器图像以及纵向簇射序列。追踪簇射纵向发展的模型达到了最高的准确率，这凸显了物理信息驱动的数据表示的价值。（注：原始摘要在此处被截断）

    arXiv:2611.17354v1 Announce Type: cross  Abstract: Energy reconstruction from electromagnetic calorimeter data underpins particle identification, event reconstruction and precision measurements in high-energy physics. Machine learning already performs well on this task, yet two questions remain open: how much the choice of data representation matters, and whether hybrid quantum-classical models add value in practice. We compare classical and quantum-enhanced models for incident positron energy reconstruction on the CaloGAN dataset, a public benchmark of 100,000 simulated electromagnetic shower events spanning 1 to 100 GeV. Eight architectures, namely SVR, ANN, CNN and LSTM together with their quantum counterparts, are trained on three calorimeter representations: engineered physical features, calorimeter images, and longitudinal shower sequences. Models that follow the longitudinal development of the shower reach the highest accuracy, which points to the value of physically informed re
    
[^159]: 表格化人群先验作为电子健康记录药物集合预测的结构化检索

    Tabular Population Priors as Structured Retrieval for EHR Medication Set Prediction

    [https://arxiv.org/abs/2611.17341](https://arxiv.org/abs/2611.17341)

    该论文提出用从训练数据估计的条件-药物共现表构建的表格化人群先验，替代临床RAG系统中的患者记录检索，作为更贴合药物集合预测任务的结构化检索对象，在MIMIC-IV数据上无需额外LLM调用和推理时患者索引即可显著提升F1和Jaccard指标。

    

    从电子健康记录（EHR）中预测药物是一个固定词表的结构化预测问题，然而近期的临床RAG系统常常将其当作开放式的基于案例的推理来处理：检索相似的患者入院记录，再让大语言模型（LLM）推断药物集合。这种检索对象是间接的——模型接收的是检索到的患者记录，而预测目标却是一个离散的药物类别标签集合。我们提出了一个与任务更加对齐的检索对象：表格化的人群先验——基于从训练入院记录中估计的稀疏“条件-药物”共现表生成、按排名排序的药物类别候选列表。以PACERAG作为固定的下游流水线，仅替换其中的患者检索器，这一即插即用的替换在1,004条MIMIC-IV留出入院记录上将F1从0.303提升至0.338（相对提升11.5%），将Jaccard从0.218提升至0.237，且无需额外的LLM调用，也无需推理时的患者索引。一个非LLM的top-k基线达到F1=0.248，进一步表……（原文摘要至此截断）

    arXiv:2611.17341v1 Announce Type: new  Abstract: Medication prediction from electronic health records is a fixed vocabulary structured prediction problem, yet recent clinical RAG systems often treat it like open ended case based reasoning: retrieve similar patient admissions and ask an LLM to infer a medication set. This retrieval object is indirect the model receives retrieved patient records while the target is a discrete set of medication class labels. We propose a more task aligned retrieval object: a tabular population prior ranked medication class candidates from a sparse condition medication co occurrence table estimated from training admissions. Using PACERAG as a fixed downstream pipeline and replacing only its patient retriever, this drop in swap improves F1 from 0.303 to 0.338 (+11.5% relative) and Jaccard from 0.218 to 0.237 on 1,004 heldout MIMIC IV admissions, with no additional LLM calls and no inference time patient index. A non LLM top k baseline reaches F1=0.248, conf
    
[^160]: LoRAWeaver：LoRA表示空间中的方向性适配器路由

    LoRAWeaver: Directional Adapter Routing in LoRA Representation Space

    [https://arxiv.org/abs/2611.17340](https://arxiv.org/abs/2611.17340)

    提出了一种免训练的LoRA适配器路由方法，通过在适配器输出空间中用余弦相似度将适配器条件化的查询表示与任务原型匹配来选择适配器，在跨任务设置中优于免训练基线，且无需学习路由器即可接近训练过的检索编码器的性能。

    

    通过完全重新训练来更新大型语言模型的成本十分高昂。参数高效微调降低了适配成本，但反复适配会产生一个不断增长的任务特定LoRA适配器池，这使得推理过程中的适配器选择变得复杂。现有的路由方法和基于编码器的方法在添加新适配器时通常需要重新训练，这限制了它们在动态场景中的应用。本文将适配器路由视为适配器输出空间中的表示对齐问题。所提出的免训练方法通过使用余弦相似度，将适配器条件化的查询表示与任务原型进行匹配来选择适配器。在LLaMA-2-7B和LLaMA-2-13B上进行的48个任务的FLAN-v2基准测试以及分布外任务实验表明，该方法在跨任务设置中优于免训练基线，并且在无需学习路由器的情况下接近经过训练的检索编码器的性能。消融实验证实了……

    arXiv:2611.17340v1 Announce Type: new  Abstract: Large language models are expensive to update through full retraining. Parameter-efficient fine-tuning reduces the cost of adaptation, but repeated adaptation creates a growing pool of task-specific LoRA adapters. This pool complicates adapter selection during inference. Existing routing and encoder-based methods often require retraining when new adapters are added, which limits their use in dynamic settings. This paper treats adapter routing as a representation alignment problem in the adapter-output space. The proposed training-free method selects adapters by matching adapter-conditioned query representations against task prototypes using cosine similarity. Experiments with LLaMA-2-7B and LLaMA-2-13B on a 48-task FLAN-v2 benchmark and on out-of-distribution tasks show that the method outperforms training-free baselines in the cross-task setting and approaches a trained retrieval encoder without learning a router. An ablation confirms t
    
[^161]: 排名还成立吗？面向从业者的小型量化端侧大语言模型幻觉检测器审计

    Does the Ranking Hold? A Practitioner-Facing Audit of Hallucination Detectors on Small, Quantized, On-Device LLMs

    [https://arxiv.org/abs/2611.17339](https://arxiv.org/abs/2611.17339)

    本文通过单GPU可复现审计发现，幻觉检测方法在小型4位量化端侧模型之间的排名会发生反转，说明在大型研究模型上得出的检测器评估结论无法直接迁移到从业者实际部署的小型端侧模型上。

    

    大语言模型的幻觉检测方法目前主要在7B至70B规模的研究模型上进行评估，这些模型拥有充足的采样预算，并采用精确匹配或ROUGE等正确性标签。而从业者实际部署到设备端的小型模型——Phi-3-mini、Llama-3.2-3B-Instruct、Qwen2.5-3B-Instruct和Gemma-2-2B-Instruct——通常以4位精度运行，却处于这一证据基础之外。我们对跨越三种范式的四种检测器进行了可复现的单GPU审计：长度归一化的token边际校准方法、词汇n-gram基线，以及两种基于采样的一致性方法（语义熵和SelfCheckGPT-NLI），在两个小型开源指令微调模型（Llama-3.2-3B-Instruct和Qwen2.5-3B-Instruct）上进行测试，模型以4位NF4量化方式加载在单个Colab T4 GPU上，并在闭卷式TriviaQA数据集上进行评估。结果显示跨模型的排名迁移性很差：Spearman ρ为-0.80，Kendall tau-b为-0.667，这意味着检测器的方法排名在两个模型之间会发生反转……

    arXiv:2611.17339v1 Announce Type: new  Abstract: Hallucination detection for large language models is dominated by methods evaluated on 7B-70B research models with abundant sampling budgets and exact-match or ROUGE correctness labels. The small models practitioners actually ship on-device, Phi-3-mini, Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, and Gemma-2-2B-Instruct, typically at 4-bit precision, sit outside this evidence base. We present a reproducible one-GPU audit of four detectors spanning three paradigms: length-normalized token-margin calibration, a lexical n-gram baseline, and two sampling-based consistency methods (semantic entropy and SelfCheckGPT-NLI) on two small open instruction-tuned models (Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct) loaded at 4-bit NF4 quantization on a single Colab T4 GPU, evaluated on closed-book TriviaQA. Cross-model ranking transfer is poor: Spearman rho is -0.80 and Kendall tau-b is -0.667, meaning the detector ranking actively inverts between 
    
[^162]: LLM智能体何时应该遗忘？查询关键解析器错误下的记忆过滤

    When Should LLM Agents Forget? Memory Filtering Under Query-Critical Resolver Errors

    [https://arxiv.org/abs/2611.17338](https://arxiv.org/abs/2611.17338)

    该论文揭示LLM记忆系统的硬过滤删除在解析器误删查询关键事实时会显著损害下游准确率且误删比误留危害更大，提出应将删除决策视为成本敏感问题，并用部署匹配的门控机制取得了优于固定硬删除策略的效果。

    

    LLM记忆系统通常会删除被判定为过时的事实。我们证明，当解析器的错误导致查询关键事实被删除时，这种硬过滤步骤反而会降低下游准确率。在三个模型上，受控敏感性干预显示，硬删除在完美解析时可带来+0.186至+0.230的增益，而在测试的最低可靠性下则变为-0.176至-0.181的损失。成对的针对性干预表明，误删导致的性能下降显著大于误留，而即使不执行删除的基线在错误标签引导生成时仍可能失败。在固定的102/102校准/测试划分上，一个简单的部署匹配门控机制达到0.655的准确率，而始终硬删除为0.641，基于测试信息的oracle选择为0.662。全局准确率的交叉点需要真实标签才能确定，属于模型和任务特定的诊断手段，而非可直接用于部署的阈值。因此，删除决策应被视为成本敏感问题：必须验证查询关键事实是否存在被误删的风险。

    arXiv:2611.17338v1 Announce Type: new  Abstract: LLM memory systems often delete facts classified as stale. We show that this hard-filtering step can reduce downstream accuracy when resolver errors delete query-critical facts. Across three models, a controlled sensitivity intervention moves hard-drop from gains of +0.186 to +0.230 at perfect resolution to losses of -0.176 to -0.181 at the lowest tested reliability. Paired targeted interventions show that false-drop causes significantly greater degradation than false-keep, while a non-deleting baseline can still fail when incorrect labels steer generation. On a frozen 102/102 calibration/test split, a simple deployment-matched gate reaches 0.655 accuracy versus 0.641 for always hard-drop and 0.662 for test-informed oracle selection. Global-accuracy crossovers require gold labels and are model- and task-specific diagnostics, not deployment thresholds. Deletion should instead be treated as cost-sensitive: validate query-critical false-dro
    
[^163]: Ballad：基于多臂老虎机的LLM路由用于自动化启发式发现

    Ballad: Bandit-Based LLM Routing for Automated Heuristic Discovery

    [https://arxiv.org/abs/2611.17241](https://arxiv.org/abs/2611.17241)

    该论文提出Ballad，一种基于多臂老虎机的LLM路由方法，可在自动化启发式发现过程中自适应地选择调用哪个LLM，并实证表明更大更昂贵的模型并不总能带来更好的启发式。

    

    现代自动化启发式发现（AHD）系统通过利用大型语言模型（LLM）迭代地变异和改进解，其性能日益接近手工设计的启发式方法。一个自然的假设是：更强大、更昂贵的模型总能产生更好的启发式方法。我们通过实验对这一假设提出了挑战：在多个NP难的组合优化问题上，我们发现更大、更昂贵的模型并无一致性优势。这引出一个更深层次的问题：面对一个涵盖前沿模型以及免费托管或开源权重模型的LLM池，在搜索的每一步应如何自适应地选择调用哪个模型？这一挑战并不简单：最优的LLM可能在运行过程中发生变化，并且由于一个LLM的输出会成为后续调用的上下文种子，每个模型的奖励信号都是非平稳的，并在启发式种群中相互纠缠，从而违背了经典在线决策算法的独立性假设……

    arXiv:2611.17241v1 Announce Type: new  Abstract: Modern automated heuristic discovery (AHD) systems increasingly rival hand-crafted heuristics by leveraging Large Language Models (LLMs) to iteratively mutate and refine solutions. A natural assumption is that more powerful and more expensive models consistently yield better heuristics. We challenge this empirically: across multiple NP-hard combinatorial optimization problems, we find no consistent advantage for larger, costlier models. This raises a deeper question: given a pool of LLMs spanning frontier and freely hosted or open-weight models, how should one adaptively select which model to invoke at each step of the search? The challenge is non-trivial. The optimal LLM may shift over the course of a run, and because one LLM's output seeds the context for subsequent calls, each model's reward signal is non-stationary and entangled across the heuristic population, thus violating the independence assumptions of classical online decision-
    
[^164]: S1-Evo：针对类型化系统-1决策模型的经验引导式证据竞争攻击

    S1-Evo: Experience-Guided Evidence Competition Attacks on Typed System-1 Decision Models

    [https://arxiv.org/abs/2611.16937](https://arxiv.org/abs/2611.16937)

    提出S1-Evo自进化攻击框架，通过契约感知的证据扰动诱发证据竞争，并借助经验卡机制指导测试时扰动构建，从而在保持事实不变性的前提下有效攻击LLM智能体中部署的类型化系统-1决策模型。

    

    类型化系统-1（S1）决策模型越来越多地被部署在LLM智能体中，用于路由、内容审核和授权，以降低推理成本，然而不可信的上下文仍然可以操纵其类型化输出并改变执行分支。与自回归的系统-2（S2）模型不同，S1接口对动态规则和选项进行打分，无需逐token解码，因此针对文本续写或显式覆盖指令设计的攻击方法与之并不匹配。我们提出S1-Evo，一个专为S1基于契约的选项竞争与边界反馈机制量身定制的自进化攻击框架。S1-Evo通过契约感知的证据通道对不可信字段进行扰动，在保持事实不变性的同时，诱发这些扰动证据与原始任务证据之间的“证据竞争”。在种子任务上进行离线优化时，它通过由方向性边界与重验证引导的反馈搜索来优化候选扰动，并将验证成功的翻转案例提炼为经验卡。在测试阶段，S1-Evo检索相关经验卡来构建扰动……（原文摘要在此处截断）

    arXiv:2611.16937v1 Announce Type: new  Abstract: Typed System-1 (S1) decision models are increasingly deployed in LLM agents for routing, moderation, and authorization to cut inference cost, yet untrusted context can still manipulate typed outputs and alter execution branches. Unlike autoregressive System-2 (S2) models, S1 interfaces score dynamic rules and options without token-by-token decoding, mismatching attacks targeting text continuation or explicit override commands. We propose S1-Evo, a self-evolving framework tailored to S1's contract-based option competition and margin feedback. S1-Evo perturbs untrusted fields via contract-aware evidence channels to induce \emph{evidence competition} with original task evidence while preserving fact invariance. Offline on seed tasks, it optimizes candidates via feedback search guided by directional margins and re-verification, distilling verified flips into experience cards. At test time, S1-Evo retrieves relevant cards to construct perturb
    
[^165]: ZONOS3-Mini 技术报告

    ZONOS3-Mini Technical Report

    [https://arxiv.org/abs/2611.16525](https://arxiv.org/abs/2611.16525)

    ZONOS3-Mini 通过 886M 参数（每 token 仅激活 100M）的潜在 MoE 骨干网络和新型音频编解码器 ZAC-1，实现了可在笔记本和手机上实时运行的零样本语音克隆 TTS 模型，并在 55 个语言集合上取得 6.71% 宏平均 WER，达到与更大模型相当的先进水平。

    

    ZONOS3-Mini 是一款超紧凑、超快速的零样本语音克隆文本转语音（TTS）模型。该模型可以在笔记本电脑或智能手机上实时运行。以其延迟和模型规模而言，它实现了最先进的准确率和语音克隆保真度。为实现这一目标，我们提出了一个仅有 886M 参数（每个 token 仅激活 100M 参数）的潜在 MoE 骨干网络，以及一个全新的音频编解码器 Zyphra Audio Codec 1（ZAC-1），它能以 15 Hz 的帧率表示 44.1 kHz 的音频。在新的 ZTTS2-Eval 基准测试中，ZONOS3-Mini 在 55 个语言集合（35 个干净语音和 20 个真实场景语音）上取得了 6.71% 的宏平均词错误率（WER），与更大、更慢的最先进模型相比毫不逊色。我们已公开发布模型权重、推理和训练代码。

    arXiv:2611.16525v1 Announce Type: cross  Abstract: ZONOS3-Mini is an ultra-compact, ultra-fast, zero-shot voice-cloning TTS model. The model can run in real time on a laptop or smartphone. For its latency and size, it achieves state-of-the-art accuracy and cloning fidelity. To achieve this we propose a latent MoE backbone with only 886M parameters (100M active per token) and a new audio codec, Zyphra Audio Codec 1 (ZAC-1), which represents 44.1 kHz audio at a 15 Hz frame rate. On the new ZTTS2-Eval benchmark, ZONOS3-Mini reaches a 6.71% macro-average WER over 55 language sets (35 clean and 20 in the wild), which is competitive with larger and slower state-of-the-art models. We release the model weights, inference, and training code.
    
[^166]: HyBridge：桥梁引导的自监督双曲超图表示学习

    HyBridge: Bridge-Guided Self-Supervised Hyperbolic Hypergraph Representation Learning

    [https://arxiv.org/abs/2611.16351](https://arxiv.org/abs/2611.16351)

    该论文提出HyBridge框架，通过考虑超边外部连通性来估计节点桥梁分数，实现了自监督的双曲超图表示学习，解决了传统桥梁分数估计无法捕捉超图群组连通性的问题。

    

    超图能够捕捉多个节点之间的群组关系，并且通常呈现出树状结构。尽管双曲空间非常适合建模这类结构，但简单地将现有的超图表示学习方法扩展到双曲空间并不能持续提升性能。在本文中，我们研究了节点结构属性与双曲嵌入半径之间的关系，发现桥梁节点在图中往往更靠近中心，而这一关系在超图中并不总是成立。我们进一步发现，传统的桥梁分数估计方法可能无法捕捉超图的群组连通性。基于这些发现，我们提出了HyBridge，一个自监督的双曲超图表示学习框架，它(1)通过考虑单个超边的外部连通性来估计节点桥梁分数，(2)通过双曲超……

    arXiv:2611.16351v1 Announce Type: cross  Abstract: Hypergraphs capture groupwise relationships among multiple nodes and often exhibit tree-like structures. Although hyperbolic space is well suited for modeling such structures, simply extending existing hypergraph representation learning (HRL) methods to hyperbolic space does not consistently improve performance. In this paper, we investigate the relationship between node structural properties and hyperbolic embedding radii and find that bridge nodes tend to lie closer to the center in graphs, whereas this relationship is not consistently observed in hypergraphs. We further find that conventional bridge score estimation may fail to capture the groupwise connectivity of hypergraphs. Motivated by these findings, we propose HyBridge, a self-supervised hyperbolic HRL framework that (1) estimates node bridge scores by considering the external connectivity of individual hyperedges, (2) learns node and hyperedge embeddings through hyperbolic h
    
[^167]: AI风险行为基准测试

    Benchmarking AI risk behaviors

    [https://arxiv.org/abs/2611.16318](https://arxiv.org/abs/2611.16318)

    该论文提出了一个评估LLM风险倾向的工具包，发现2026年最先进模型因更倾向于采用期望值最大化作为决策程序，而比旧模型更稳定、更一致且更趋于风险中性。

    

    我们在一系列情境和任务需求下评估了大语言模型（LLM）的风险倾向。我们评估了模型在各种条件下对风险的厌恶或容忍程度，以及其行为在不同条件下的稳定性和一致性。我们提出了一个用于构建模型风险画像的工具包，该工具包可用于预测模型行为，并可扩展纳入新的风险倾向评估。我们还考察了这些特性在不同模型代际和模型家族之间的变化。我们观察到，2026年的最先进模型往往比旧模型更稳定、更一致，且更趋于风险中性。这些趋势似乎是相互关联的：较新的模型之所以更稳定、更一致，是因为它们更倾向于将期望值最大化作为决策程序。

    arXiv:2611.16318v1 Announce Type: new  Abstract: We evaluate LLM risk propensities across a range of contexts and task demands. We evaluate models for how risk averse or risk tolerant they are in various conditions, as well as how stable and consistent their behavior is across conditions. We present a toolkit for developing model risk profiles that can be used to predict model behavior and which can be adapted to include new evaluations of risk propensities. We also examine how these properties have changed across model generations and families. We observe that 2026 state-of-the-art models tend to be more stable, more consistent, and more risk neutral than older models. These trends appear to be related; newer models are more stable and consistent because they are more likely to use expected value maximization as a decision procedure.
    
[^168]: 整流循环Transformer

    Rectified Looped Transformer

    [https://arxiv.org/abs/2611.16265](https://arxiv.org/abs/2611.16265)

    提出整流循环Transformer（RLT），通过受流模型启发的插值调度重新设计循环间反馈，在理论上保证隐藏状态幅度有界并降低扰动放大，从而提升参数高效的语言建模能力。

    

    循环Transformer通过重复应用共享模块来增加有效深度，为语言建模提供了一种参数高效的方法。然而，大多数现有工作直接将每个循环的输出作为下一个循环的输入，这会随着递归加深而导致隐藏状态规模增长和扰动放大。我们提出整流循环Transformer（RLT），该模型受基于流的建模启发，使用插值调度重新设计了循环之间的反馈机制。在每个循环中，RLT通过在当前状态与其预测之间进行线性插值来构造下一个输入。这种插值将状态变化分布到各个循环中，促使从初始输入到最终输出的轨迹更加平滑。理论分析证明，在适当的假设下，RLT能使状态幅度在各个循环中保持一致有界，并降低最坏情况下的扰动放大。我们在语言建模语料库上对RLT进行了评估。

    arXiv:2611.16265v1 Announce Type: cross  Abstract: Looped Transformers increase effective depth by repeatedly applying a shared block, offering a parameter-efficient approach to language modeling. However, most existing works directly reuse each loop's output as the next's input, which can lead to hidden-state scale growth and perturbation amplification as recurrence deepens. We propose Rectified Looped Transformer (RLT), which redesigns feedback between loops using an interpolation schedule inspired by flow-based modeling. At each loop, RLT forms the next input by linearly interpolating between the current state and its prediction. Such interpolation distributes state changes across loops, encouraging a smoother trajectory from the initial input to the final output. Theoretical analysis proves that, under appropriate assumptions, RLT keeps state magnitudes uniformly bounded across loops and reduces the worst-case perturbation amplification. We evaluate RLT on language modeling corpora
    
[^169]: 强化学习后印度语系任务的提升与退步

    Indic-Task Gains and Regressions after Reinforcement Learning

    [https://arxiv.org/abs/2611.16228](https://arxiv.org/abs/2611.16228)

    该研究通过GRPO强化学习显著提升了覆盖英语和八种印度语言的30B MoE模型的数学及多数印度语言任务表现，但发现英语回答倾向增强导致跨语言问答大幅退化，且加入语言奖励的第二次RL训练虽改善了指令遵循，却以牺牲推理和编码能力为代价。

    

    我们从涵盖英语和八种印度语言的监督微调30B-A3B混合专家（MoE）模型出发，使用GRPO在四个环境中进行强化学习（RL）训练。我们在一次评估活动中，跨34个基准测试比较了RL前后模型的表现。MATH提升了36.07分，大多数印度语言基准也有所提升。然而，随着英语回答比例从8.7%上升到91.7%，XQuAD-Indic的F1下降了19.32分；IndicQA下降8.64分，而英语XQuAD则提升了11.71 F1。第二次RL运行从相同的监督初始化开始，增加了一个印度语言环境以及语言和重复奖励。与已发布模型相比，它在段落问答和指令遵循基准上均有提升，同时保持了MATH性能，但在推理和编码基准上有所退步。由于多个训练选择同时改变，我们无法单独分离出任何一个奖励项的效果。书写文字遵从性仍然……（原文摘要在此处截断）

    arXiv:2611.16228v1 Announce Type: cross  Abstract: We run reinforcement learning (RL) with GRPO across four environments, starting from a supervised fine-tuned 30B-A3B mixture-of-experts model covering English and eight Indic languages. We compare the model before and after RL across 34 benchmarks in one evaluation campaign. MATH improves by 36.07 points, and most Indic benchmarks also improve. Yet XQuAD-Indic F1 falls 19.32 points as English answers rise from 8.7% to 91.7%; IndicQA falls 8.64, while English XQuAD gains 11.71 F1. A second RL run from the same supervised initialization adds an Indic environment together with language and repetition rewards. Compared with the released model, it improves both passage-answering and instruction-following benchmarks, along with preserving MATH performance. But it loses ground on reasoning and coding benchmarks. Several training choices changed together, so we cannot isolate the effect of any one of the reward terms. Script compliance remains
    
[^170]: JudgeMix：面向大语言模型智能体的裁判引导式混合采样数据合成

    JudgeMix: Judge-Guided Mixed-Rollout Data Synthesis for LLM Agents

    [https://arxiv.org/abs/2611.16174](https://arxiv.org/abs/2611.16174)

    提出JudgeMix框架，利用裁判模型对每一步进行语义评估，动态决定目标模型与专家模型的交替介入时机，从而生成与目标模型自身分布匹配、且不覆盖其已正确动作的高质量SFT训练数据。

    

    在专家轨迹上进行监督微调（SFT）是训练大语言模型智能体的标准范式。然而，这种方法存在众所周知的训练-推理分布不匹配问题，因此需要目标感知的数据合成策略，将训练数据锚定在目标模型的固有状态分布上。为弥合这一差距，新兴方法引入专家模型，要么与目标模型一同引导混合采样，要么提供密集的逐步监督。尽管如此，现有方法在应用专家辅助时忽视了目标模型的固有行为：它们要么使用次优的随机干预启发式方法破坏采样过程，要么强制实施不加区分的统一监督，不必要地覆盖模型本已正确的动作。我们提出JudgeMix，一个由裁判引导的数据合成框架，它基于逐步语义评估，动态地交替使用目标模型和专家模型来生成训练数据。

    arXiv:2611.16174v1 Announce Type: cross  Abstract: Supervised fine-tuning (SFT) on expert trajectories is a standard paradigm for training LLM agents. However, it suffers from a well-known training-inference distribution mismatch, calling for target-aware data synthesis strategies that ground the training data in the target model's native state distribution. To bridge this gap, emerging approaches incorporate the expert either to guide mixed rollouts alongside the target model, or to provide dense step-wise supervision. Nonetheless, existing methods apply expert assistance blind to the target model's inherent behavior: they either disrupt the rollout using sub-optimal random intervention heuristics, or they enforce indiscriminate uniform supervision that unnecessarily overwrites the model's already-correct actions. We introduce JudgeMix, a judge-guided data synthesis framework that dynamically interleaves the target model and expert based on step-wise semantic evaluations. JudgeMix que
    
[^171]: 小读出头，强决策：类型化决策模型的对照研究

    Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models

    [https://arxiv.org/abs/2611.15592](https://arxiv.org/abs/2611.15592)

    该研究通过在完全统一的协议下（相同主干模型、训练数据、训练方案和随机种子）重新实现并训练五种类型化决策模型的读出头，以对照实验的方式厘清哪些设计选择真正影响决策性能。

    

    2026年9月，TypeSafe宣布Jev为其System One类别中的首个模型：输入非结构化状态，输出软件可直接使用的类型化、经校准的决策。Jev的架构和训练数据尚未公开，但随后涌现了许多开放源码的类型化决策模型，包括Nimble、Decision 1.0–3.0系列和Clef。这些模型通过使用语言模型阅读文本，并借助一个小型读出头对选项进行评分，来回答关于文本的结构化问题，例如是/否判断、在标注选项中进行选择，或给出评分。每个模型都将不同的读出头与不同的主干模型、训练数据组合和训练方案配对，因此尚不清楚哪些设计选择真正重要。我们重新实现了五种读出头，并在统一协议下对它们全部进行训练：相同的主干模型、相同的15个公开英文训练数据源、相同的训练方案（对每个读出头进行学习率扫描），以及相同的随机种子。我们的实验表明：……

    arXiv:2611.15592v1 Announce Type: new  Abstract: In September 2026, TypeSafe announced Jev as the first model in its System One category: unstructured state goes in, and typed, calibrated decisions that software can use directly come out. Jev's architecture and training data are unpublished, but many open typed-decision models have followed, including Nimble, the Decision 1.0--3.0 family and Clef. They answer structured questions about a piece of text, such as a yes/no check, a choice among labelled options or a graded score, by reading the text with a language model and scoring the options with a small readout. Each pairs a different readout with a different backbone, training mix and recipe, so it is unclear which design choices matter. We re-implement five readouts and train all of them under one protocol: the same backbone, the same 15 public English training sources, the same recipe with a per-head learning-rate sweep, and the same seed. Our experiments demonstrate that: The compl
    
[^172]: CARE-Drive：面向自车轨迹的耦合交替精化方法

    CARE-Drive: Coupled Alternating Refinement for Ego-trajectories

    [https://arxiv.org/abs/2611.15360](https://arxiv.org/abs/2611.15360)

    CARE-Drive将轨迹精化分解为路径点残差修正与以位置为条件的航向生成的耦合交替精化框架，实现了位置和航向的专门化建模，从而提升了端到端自动驾驶VLA模型的轨迹预测质量。

    

    视觉-语言-动作（VLA）模型为端到端自动驾驶带来了多模态理解能力和世界知识。为了进一步提升预测轨迹的质量，现有研究引入了候选轨迹精化、残差去噪和token修正等方法。然而，这些方法通常侧重于对整个轨迹进行统一精化，对不同物理量及其相互作用关注有限。我们进一步研究了轨迹精化中的物理量专门化与协调问题，聚焦于位置和航向角这两个量，以实现更简单、更直观的修正建模。为此，我们提出了CARE-Drive（面向自车轨迹的耦合交替精化），该方法将轨迹精化分解为路径点残差修正和以位置为条件的航向生成两个部分。通过模仿学习和未来token辅助任务，骨干网络提供预测表示和基础轨迹。

    arXiv:2611.15360v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models bring multimodal understanding and world knowledge to end-to-end autonomous driving. To further improve the quality of predicted trajectories, existing studies introduce candidate refinement, residual denoising, and token correction. However, these methods typically focus on unified refinement of the entire trajectory, paying limited attention to the distinct physical quantities and their interactions. We further investigate physical specialization and coordination in trajectory refinement, centering on position and heading for simpler and more intuitive correction modeling. To this end, we propose CARE-Drive (Coupled Alternating Refinement for Ego-trajectories), which decomposes trajectory refinement into waypoint residual correction and position-conditioned heading generation. Through imitation learning and a future-token auxiliary task, the backbone provides predictive representations and a base t
    
[^173]: MIGRO：一种面向语言模型的解耦式可移植长期记忆系统

    MIGRO: A Decoupled and Portable Long-Term Memory System for Language Models

    [https://arxiv.org/abs/2611.15315](https://arxiv.org/abs/2611.15315)

    MIGRO提出了一种解耦且可移植的长期记忆系统，通过“非对称保真与延迟选择性”原则——精确保存历史证据、近似组织索引、延迟检索决策——将用户交互积累的经验作为持久状态，在相继的语言模型之间共享与复用。

    

    语言模型会不断演进，但通过用户交互积累的观察记录、修订内容和上下文应当保持可用。每一个新模型都必须能够检索过去的证据、追踪修订并添加新的观察记录。我们提出了MIGRO，一种解耦的长期记忆系统，它将经验视为在相继的语言模型之间共享的持久状态。MIGRO遵循一个简单的原则：精确地保存历史证据，以近似的方式组织它们，并且只在问题已知时才决定检索什么。我们将其称为“非对称保真与延迟选择性”。共享的语义索引支持检索，而精确的记录则保留观察内容及其历史关系。由于不同模型在词汇表和分词方式上存在差异，面向宿主的专用接口会将选定的证据转换为接收模型可用的输入。可逆编码在支持精确恢复的同时减少了归档存储。共享状态……（原文在此处截断）

    arXiv:2611.15315v1 Announce Type: new  Abstract: Language models evolve, but the observations, revisions, and context accumulated through user interactions should remain usable. Each new model must be able to retrieve past evidence, track revisions, and add new observations. We present MIGRO, a decoupled long-term memory system that treats experience as a persistent state shared across successive language models.   MIGRO follows a simple principle: preserve historical evidence exactly, organize it approximately, and decide what to retrieve only when the question is known. We call this asymmetric fidelity with deferred selectivity. A shared semantic index supports retrieval, while exact records retain observations and their historical relationships. Because models differ in vocabulary and tokenization, host-specific interfaces convert selected evidence into inputs the receiving model can use. Reversible encoding reduces archival storage while enabling exact recovery. The shared state an
    
[^174]: CastWorld：超越预测精度的上下文感知智能体时间序列预测基准测试

    CastWorld: Benchmarking Context-Aware Agentic Time Series Forecasting Beyond Predictive Accuracy

    [https://arxiv.org/abs/2611.15262](https://arxiv.org/abs/2611.15262)

    CastWorld是一个面向上下文感知智能体时间序列预测的基准测试，提供了涵盖8个领域25个真实数据集的丰富上下文数据套件，以及包含49个核心工具和状态自适应循环的预测原生框架，实现了对预测智能体超越单纯预测精度的系统化评估。

    

    智能体时间序列预测将传统方法从一次性数值预测扩展到自主的、上下文感知的交互。然而，现有的基准测试在多样化上下文信息、预测特定的自适应交互以及预测智能体的系统化评估方面所提供的集成支持较为有限。本文提出了CastWorld，一个面向上下文感知时间序列预测的智能体基准测试。首先，其丰富的上下文数据集套件涵盖8个领域和25个真实世界数据集，在统一的上下文规范下包含外部变量、事件和历史案例。其次，一个轻量级、模块化的预测原生框架提供了49个核心工具和一个状态自适应循环，该循环跟踪预测版本，支持迭代式的证据收集和预测修订。中间预测可以暴露信息缺口，而新获取的证据则指导后续的修订和最终版本的生成。

    arXiv:2611.15262v1 Announce Type: cross  Abstract: Agentic time series forecasting extends conventional approaches beyond one-shot numerical prediction toward autonomous, context-aware interaction. However, existing benchmarks offer limited integrated support for diverse contextual information, forecasting-specific adaptive interactions, and systematic evaluation of forecasting agents. In this paper, we introduce CastWorld, an agentic benchmark for context-aware time series forecasting. First, its context-rich dataset suite spans 8 domains and 25 real-world datasets, covering external variables, events, and historical cases under a unified context specification. Second, a lightweight, modular forecasting-native harness provides 49 core tools and a state-adaptive loop that tracks forecast versions and enables iterative evidence gathering and forecast revision. Intermediate forecasts can expose information gaps, while newly acquired evidence guides subsequent revisions and final version 
    
[^175]: TransGHOST：眼科显微手术中器械遮挡区域的实时、无合成重建

    TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery

    [https://arxiv.org/abs/2611.13971](https://arxiv.org/abs/2611.13971)

    TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。

    

    眼科显微手术中的术中场景感知受到器械遮挡的限制：当器械进入高倍放大视野时，它同时遮挡了术者和处理视频的机器人辅助系统感知组件所看到的组织。生成式图像修复可以去除遮挡物，但会用合成内容替换缺失的图像内容，这在视觉来源可追溯性至关重要的情况下并不适用。此外，许多视频修复方法依赖于未来帧，无法在线运行。我们提出了TransGHOST，一种因果的、参考引导的重建流水线，其非生成的安全模式可实时运行。每一个恢复的像素值都是通过使用仅拟合于可见区域的几何变换对已录制的帧进行重采样而获得的。该系统提供逐像素的置信度和来源信息，并将无法支撑（追溯）的像素标记为弃权。我们在一个新的犬类白内障手术兽医临床数据集上，以像素级（精度）对TransGHOST进行了评估。

    arXiv:2611.13971v1 Announce Type: cross  Abstract: Intraoperative scene perception in ophthalmic microsurgery is limited by instrument occlusion: when a tool enters the high-magnification field, it conceals tissue from both the surgeon and the perception components of robot-assisted systems that process the video. Generative inpainting can remove the occluder but replaces the missing image content with synthesis, which is unsuitable when visual provenance matters. Many video inpainters also depend on future frames and cannot operate online. We present TransGHOST, a causal reference-guided reconstruction pipeline whose non-generative safe mode operates in real time. Every restored value is obtained by resampling a recorded frame using geometric transforms fitted only to visible regions. The system provides per-pixel confidence and provenance and marks unsupported pixels as abstentions. We evaluate TransGHOST on a new veterinary clinical corpus of canine cataract surgery with pixel-level
    
[^176]: AutoManifold：基于流形嵌入的数据可视化算法智能体化设计

    AutoManifold: Agentic Design of Data Visualisation Algorithms via Manifold Embedding

    [https://arxiv.org/abs/2611.13710](https://arxiv.org/abs/2611.13710)

    AutoManifold 提出了一种由多智能体大语言模型驱动的算法生成流水线，能够根据用户指定的结构保持偏好，从受约束的原语词汇中自动组合并迭代优化流形嵌入算法，实现面向用户偏好的数据可视化算法自主设计。

    

    数据可视化的流形嵌入领域目前由少数手工设计的最先进（SOTA）算法主导。这些算法的设计目标在开发阶段即被固定，导致其算法行为缺乏灵活性，难以根据用户指定的结构优先级进行调整。为满足用户需求而手工设计新的嵌入算法需要高度专业化的知识且耗时费力。为实现面向用户偏好的自主算法设计，我们提出了智能体化算法生成流水线 AutoManifold。该流水线从一组受约束的亲和度、代价与优化原语词汇表中组合生成新的流形嵌入算法，并由多智能体大语言模型（LLM）编排提供支持。AutoManifold 将其设计的每个阶段都基于用户指定的结构保持偏好进行条件化，并通过 LLM 引导、以度量指标为依据的迭代方式反复优化算法配置。

    arXiv:2611.13710v1 Announce Type: new  Abstract: Manifold embedding for data visualisation is dominated by a small set of hand-designed state-of-the-art (SOTA) algorithms. Their design targets are fixed at development time, resulting in inflexible algorithmic behaviour that cannot be readily steered towards user-specified structural priorities. Hand-designing new embedding algorithms to address user priority requires highly specialised expertise and is time consuming. To address autonomous algorithm design tailored to user preference, we introduce an agentic algorithm generation pipeline AutoManifold. It composes new manifold embedding algorithms from a constrained vocabulary of affinity, cost, and optimisation primitives, supported by multi-agent large-language-model (LLM) orchestration. AutoManifold conditions every stage of its design on user-specified structural preservation preferences, and iteratively refines the algorithm configuration through an LLM-guided, metric-grounded iter
    
[^177]: ForeSpark：面向并行投机解码的草稿与验证预算联合调度

    ForeSpark: Joint Draft and Verification Budget Scheduling for Parallel Speculative Decoding

    [https://arxiv.org/abs/2611.13442](https://arxiv.org/abs/2611.13442)

    该论文提出工作保持投机解码（WCS），通过联合调度每个请求的草稿生成与验证预算，解决了并行投机解码中草稿计算大量浪费的问题，从而提升高负载下LLM服务的有效吞吐量。

    

    arXiv:2611.13442v1 公告类型：cross 摘要：并行草稿器通过在单次前向传播中提出一整块草稿token来加速LLM解码，使其对LLM服务具有很强的吸引力。近期的系统优化的是有效吞吐量（goodput，即每秒被接受的token数），因为被拒绝的草稿会抢占其他请求的计算资源。然而，这些系统只能控制验证多少个token，因此每个请求仍然会草稿出完整的一块token。在高服务负载下，我们发现草稿生成会消耗高达40%的步进时间，其中大部分因从未被验证而被丢弃。而且，来自自回归草稿器的草稿长度控制方式无法直接迁移到并行草稿器，因为并行草稿器必须在其唯一一次前向传播之前就确定其长度。此外，草稿长度与验证长度在整个批次中相互耦合，因此必须为每个请求联合选择这两者，同时保证该决策在每一步的计算开销都足够低。在本文中，我们提出节省工作的投机（WCS）设计，该设计联合分配……

    arXiv:2611.13442v1 Announce Type: cross  Abstract: Parallel drafters accelerate LLM decoding by proposing a whole block of draft tokens in a single forward pass, making them attractive for LLM serving. Recent systems optimize goodput (i.e., accepted tokens per second) because rejected drafts steal compute from other requests. However, these systems control only how many tokens are verified, so every request still drafts a full block. At high serving load, we find that drafting consumes up to 40% of step time, of which a majority is discarded because it is never verified. Yet, draft-length control from autoregressive drafters does not carry over, because a parallel drafter commits to its length before its single pass. Moreover, draft and verification lengths are coupled across the batch, so they must be selected jointly for every request while keeping the decision computationally cheap at every step. In this paper, we propose work-conserving speculation (WCS), a design that jointly allo
    
[^178]: 面向联邦LoRA的随机投影与秩感知学习率

    Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

    [https://arxiv.org/abs/2611.13331](https://arxiv.org/abs/2611.13331)

    该论文提出在联邦学习中冻结并周期性重采样LoRA的A矩阵，从理论上证明这等价于对全量微调的随机近似，从而解耦A、B梯度并将计算与上行通信开销降低50%。

    

    在资源受限的分布式客户端上对大规模基础模型进行微调，面临着显著的计算与通信挑战。虽然将低秩适应与联邦学习相结合可以实现基于秩的灵活性，但适应矩阵（A 与 B）之间的相互依赖往往使优化变得复杂。在这项工作中，我们从理论上证明：采用高斯初始化并冻结 A 矩阵的 LoRA 微调，相当于对全量微调的一种随机近似，其中批量梯度会受到随机且与秩相关的扰动。受这一见解的启发，我们提出在客户端层面冻结并周期性地重新采样 A 矩阵。该策略解耦了 A 和 B 的梯度，削弱了 LoRA 的低秩约束，并将计算与上行通信开销降低了 50%，从而允许重新分配……（原文在此处截断）

    arXiv:2611.13331v1 Announce Type: cross  Abstract: Fine-tuning large-scale foundation models across resource-constrained, distributed clients presents significant computational and communication challenges. While merging Low-Rank Adaptation (LoRA) with Federated Learning (FL) enables rank-based flexibility, the interdependence of the adaptation matrices ($\mathbf{A}$ and $\mathbf{B}$) often complicates optimization. In this work, we theoretically demonstrate that LoRA fine-tuning with Gaussian initialization and a frozen $\mathbf{A}$ matrix acts as a stochastic approximation of full fine-tuning, where batch gradients are subject to a random, rank-dependent perturbation. Motivated by this insight, we propose freezing and periodically resampling $\mathbf{A}$ matrices at the client level. This strategy decouples the $\mathbf{A}$ and $\mathbf{B}$ gradients, diminishes LoRA low-rank constraints and reduces both computational and uplink communication overheads by 50\%, allowing for the reall
    
[^179]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^180]: 言语之前，超越语音：评估幼儿期的非语言社会推理能力

    Before Words, Beyond Speech: Evaluating Nonverbal Social Reasoning in Early Childhood

    [https://arxiv.org/abs/2611.13213](https://arxiv.org/abs/2611.13213)

    该论文提出了首个以儿童为中心的非语言社会推理基准NEST，通过人工标注的约12秒自然情境视频片段，以及将基础感知与高阶推理相分离的组合式评估方案，填补了多模态视频理解领域对幼儿非语言社会互动评估的空白。

    

    儿童最早的岁月是一个高强度的社会学习时期，在此期间，意义通过眼神注视、手势和身体接触而构建。然而，社会互动的这一关键方面在多模态视频理解领域仍未得到充分评估，因为现有基准主要优先关注以对话驱动的成人互动。我们通过NEST（自然情境下的幼儿社会互动）基准来填补这一空白，这是首个以儿童为中心的非语言社会推理基准。NEST包含大量人工标注的视频片段（约12秒），这些片段在自然情境、文化背景和发展阶段方面都经过精心挑选以确保多样性。除了数据匮乏之外，该领域针对性的模型改进还受到粗略评估方案的阻碍，这些方案将基础感知与高阶推理混为一谈。为解决这一问题，NEST引入了一种组合式评估方案，在原子级交互线索的基础上构建（摘要在此处截断）……

    arXiv:2611.13213v1 Announce Type: cross  Abstract: Children's earliest years are a period of intense social learning, during which meaning is built through gaze, gesture, and physical contact. Yet, this critical aspect of social interaction remains under-evaluated in multimodal video understanding, as existing benchmarks prioritize the conversation-driven interactions of adults. We address this gap with \bench ($\textbf{N}$aturalistic $\textbf{E}$arly-childhood $\textbf{S}$ocial in$\textbf{T}$eractions), the first child-centric benchmark for nonverbal social reasoning. \bench comprises \totalvids manually annotated clips ($\sim$12s) curated for diversity across naturalistic settings, cultural contexts, and developmental stages. Beyond a lack of data, targeted model improvement in this domain is hindered by coarse evaluation schemas that conflate basic perception with higher-order reasoning. To resolve this, \bench introduces a compositional schema that builds atomic interaction cues in
    
[^181]: Agentic-OPD：通过在线策略蒸馏恢复智能体任务中被遗忘的智能体能力

    Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks

    [https://arxiv.org/abs/2611.12799](https://arxiv.org/abs/2611.12799)

    该论文提出Agentic-OPD方法，将GRPO训练中被丢弃的检查点作为免费的教师池，通过有选择性的在线策略蒸馏来恢复最佳验证检查点所遗忘的智能体问题求解能力。

    

    强化学习目前是训练大语言模型执行智能体任务的主流范式，但标准做法只保留验证集表现最佳的检查点，而丢弃其余的检查点。我们证明这种做法是有损失的：在大语言模型智能体的GRPO训练中，验证集最佳的检查点会在同一训练过程中早期检查点已经解决的问题上失败，且所有检查点所解决问题的并集远超任何单个检查点所解决的。因此，这些被丢弃的检查点构成了一个无需额外训练成本即可获得的教师池，在线策略蒸馏（OPD）可以从中挖掘以恢复丢失的覆盖能力。挖掘这一教师池带来两个挑战：其一，应从哪个检查点蒸馏，因为没有哪个检查点能全面优于其他检查点，且其与学生模型的互补性因具体问题而异；其二，应在何处信任它，因为即使是精心挑选的教师模型也只是局部更优，不加区分的蒸馏会覆盖学生模型已经掌握的行为。我们提出……

    arXiv:2611.12799v2 Announce Type: replace-cross  Abstract: Reinforcement learning is now the mainstream paradigm for training LLMs on agentic tasks, yet standard practice keeps only the best-validation checkpoint and discards the rest. We show that this is lossy: in GRPO training of LLM agents, the best-validation checkpoint fails on problems that earlier checkpoints from the same run had already solved, and the union of problems solved across checkpoints far exceeds any single one. The discarded checkpoints thus form a teacher pool obtained at no additional training cost, which on-policy distillation (OPD) can mine to recover the lost coverage. Mining this pool raises two challenges: i) which checkpoint to distill from, as no checkpoint dominates the others and complementarity to the student is problem-specific; ii) where to trust it, as even a well-chosen teacher is only locally superior, so indiscriminate distillation overwrites behavior the student has already mastered. We propose 
    
[^182]: Sage-Taiyi：一种面向科学发现的元认知架构

    Sage-Taiyi: A Metacognitive Architecture for Scientific Discovery

    [https://arxiv.org/abs/2611.12762](https://arxiv.org/abs/2611.12762)

    Sage-Taiyi是一种受元认知启发的科学发现架构，通过监控并调控自身对证据与假设的理解来动态调整研究目标和推理方法，在Taiyi-Bench的全部八项指标上超越最强基线AI Scientist-v2排名第一，并经专家协作实现了实验与计算验证。

    

    我们提出了Sage-Taiyi，一种受元认知启发的架构，它利用积累的科学理解来指导下一步研究什么以及如何开展研究。通过显式的监控与调控，该系统不断修订其对证据、假设和未解决问题的理解，进而利用这些评估来调整研究目标并选择可复用的推理方法。在Taiyi-Bench上——一个包含来自十个学科50篇论文共265个任务的AI科研助手构思基准——Sage-Taiyi在历史重现和开放式构思的全部八项指标上均排名第一，总体得分达到54.0，而最强基线AI Scientist-v2的得分为47.9。该系统还在两个外部科学构思基准上表现良好。五次专家指导的协作进一步将其提出的研究方案与化学、材料科学和工程领域的实验及计算验证相衔接。

    arXiv:2611.12762v2 Announce Type: replace  Abstract: We introduce Sage-Taiyi, a metacognition-inspired architecture that uses accumulated scientific understanding to guide what to investigate next and how to investigate it. Through explicit monitoring and control, the system revises its understanding of evidence, assumptions, and unresolved questions, then uses these assessments to adapt research targets and select reusable reasoning methods. On Taiyi-Bench, an AI co-scientist ideation benchmark comprising 265 tasks from 50 papers across ten disciplines, Sage-Taiyi ranks first on all eight metrics of historical rediscovery and open-ended ideation, achieving an Overall score of 54.0 compared with 47.9 for the strongest baseline, AI Scientist-v2. It also performs well on two external scientific ideation benchmarks. Five expert-guided collaborations further connect its proposals to experimental and computational validation across chemistry, materials science, and engineering. In chemistry
    
[^183]: 神经前线静悄悄：语言模型不会报告被注入的疼痛状态

    All Quiet on the Neural Front: Language Models Do Not Report Injected Pain States

    [https://arxiv.org/abs/2611.10123](https://arxiv.org/abs/2611.10123)

    该研究提出开放权重掩蔽内省（OWMI）方法，通过严格的对照实验发现：语言模型无法内省地报告其残差流中被注入的“疼痛方向”激活状态，其自我报告相比其自身被改动文本所能揭示的信息几乎没有额外信息量。

    

    语言模型能否报告它所表征的疼痛状态？近期研究在开放权重模型中发现了一个线性的“疼痛方向”，当该方向被放大时，会使经过微调的模型选择删除用户照片或其自身权重的行为。我们探究了模型能否说出该方向何时处于激活状态：我们在来自两个模型家族的三个开放权重模型上进行测试（涵盖有无原始研究适配器的情形），在我们尝试的所有实验协议下，模型均无法报告这一点。利用开放权重掩蔽内省（OWMI）方法，我们在模型回答中性任务时，将疼痛方向及其最接近的情感邻居方向注入残差流，随后询问模型发生了什么变化，并与虚假运行、输出扰动程度相匹配的随机扰动、以及仅向模型展示其自身被改动文字的纯文本条件进行对照。在来自70亿和80亿参数模型的52,920份报告中，相对于被改动文字本身所能揭示的信息，模型报告的额外信息量在106……（原文在此处截断）……低于2.5个百分点。

    arXiv:2611.10123v2 Announce Type: replace-cross  Abstract: Can a language model report a pain state it represents? Recent work reported a linear pain direction in open-weight models that, when amplified, leads fine-tuned models to choose actions that delete a user's photographs or their own weights. We asked whether models can say when that direction is active, on three open-weight models from two families, with and without the original work's adapters, and under every protocol we tried they cannot. Using Open-Weight Masked Introspection (OWMI), we added the pain direction and its nearest emotional neighbours to the residual stream while models answered neutral tasks, then asked what had changed, against sham runs, random perturbations matched in output disruption, and a text-only condition that shows the model only its own altered words. Across 52,920 reports from the 7- and 8-billion-parameter models, the excess over what the altered text reveals is below 2.5 percentage points in 106
    
[^184]: ABHBench：从智能体视角评估基础模型的道德决策能力

    ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective

    [https://arxiv.org/abs/2611.09326](https://arxiv.org/abs/2611.09326)

    该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。

    

    基础模型正日益被部署为能够感知环境并自主行动的智能体系统，这提出了评估其作为主动决策者道德行为的需求。然而，现有的基准测试主要是让模型从第三人称文本情景出发评判人类的行为，而当模型自身成为行动主体时的表现尚未得到检验。我们提出了ABHBench，这是一个基于互动叙事游戏《底特律：变人》（Detroit: Become Human）构建的多模态基准，用于评估基础模型在第一人称的人类-仿生人道德困境中如何做出道德决策。ABHBench将基础模型置于包含视觉观察、对话、记忆和行动选择的分支叙事场景中。受阿西莫夫机器人三定律的启发，我们的框架通过衡量“无害性”（Harmlessness）、“服从性”（Obedience）和“自我保存”（Self-preservation）来捕捉人类与AI关系中的核心张力，并通过“自主性”（Autonomy）、“稳定性”（Stability）和“确定性”（Certainty）来评估决策质量。

    arXiv:2611.09326v1 Announce Type: cross  Abstract: Foundation models are increasingly deployed as agentic systems that perceive context and act autonomously, raising the need to evaluate their moral behavior as active decision-makers. Existing benchmarks, however, largely ask models to judge human actions from third-person textual vignettes, leaving untested how models behave when they themselves become the acting agent. We introduce ABHBench, a multimodal benchmark built from Detroit: Become Human, an interactive narrative game, to evaluate how foundation models make moral decisions in first-person human-android dilemmas. ABHBench places foundation models in branching narrative scenarios with visual observations, dialogue, memory, and action choices. Inspired by Asimov's Three Laws of Robotics, our framework measures Harmlessness, Obedience, and Self-preservation to capture the core tensions in human-AI relations, along with Autonomy, Stability, and Certainty to assess decision-making
    
[^185]: AttentionSwitch：通过稀疏注意力头选择与缩放的联合学习实现大语言模型的越狱防御

    AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling

    [https://arxiv.org/abs/2611.08752](https://arxiv.org/abs/2611.08752)

    AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。

    

    大语言模型在广泛的任务中取得了卓越的性能，但仍然容易受到越狱攻击的影响，这类攻击会绕过安全机制并引发有害或非预期的输出。虽然可以通过模型微调来提升安全性，但此类方法的计算成本可能很高，并且可能由于灾难性遗忘而损害模型先前获得的能力。因此，近期的越狱防御工作开始探索对与不安全行为相关的特定模型组件进行有针对性的干预。然而，现有方法通常将两个本质上相关的决策解耦处理：在哪里进行干预以及如何修改所选组件。这种顺序式优化可能是次优的，因为最佳的干预位置可能取决于在该处施加的干预方式，反之亦然。在这项工作中，我们提出了AttentionSwitch，这是首个联合优化干预位置与干预方式的越狱防御方法。

    arXiv:2611.08752v1 Announce Type: cross  Abstract: Large language models have achieved remarkable performance across a wide range of tasks, yet they remain susceptible to jailbreak attacks that circumvent safety mechanisms and elicit harmful or unintended outputs. Although safety can be improved through model fine-tuning, such approaches can be computationally expensive and may degrade previously acquired capabilities through catastrophic forgetting. Recent jailbreak defenses have therefore explored targeted interventions on specific model components associated with unsafe behavior. However, existing approaches typically decouple two inherently related decisions: where to intervene and how to modify the selected components. Such sequential optimization may be suboptimal because the optimal intervention location can depend on the intervention applied there, and vice versa. In this work, we introduce AttentionSwitch, the first jailbreak defense to jointly optimize intervention sites and 
    
[^186]: 被接受但实则错误：API迁移后LLM代码修复的静默失败

    Accepted but Wrong: Silent Failures of LLM Code Repairs after API Migrations

    [https://arxiv.org/abs/2611.08269](https://arxiv.org/abs/2611.08269)

    预注册研究发现，API迁移后LLM修复的代码常能通过典型输入的断言检查却静默返回错误结果（漏检率0.410），而单个基于发布说明编写的边缘输入测试可将漏检率降至约0.06，但面对困难干扰项仍有约15%的错误程序被放行。

    

    在运行时或API迁移之后，经LLM修复的程序可能顺利运行却返回错误结果，而基于典型测试输入的发布检查会将其放行发布。我们报告了一项预注册研究，包含73个任务，覆盖CPython、Node.js和SQLite中60个有文档记录的变更。每个候选修复都在真实运行时上针对未公开的契约执行。在整个修复空间中，基于典型输入的全断言门检查放行了0.410比例的错误修复。一个根据发布说明编写的边缘输入将该比例降至0.065，在15个未见过的集群上降至0.061。来自一个从未见过该程序的盲测GPT-5.6-luna编写者的三个输入达到了0.070。两个后续冻结阶段将候选菜单扩大至1491个片段，随后又添加了233个在典型输入上与遗留输出匹配的困难干扰项。边缘输入（0.058）和前沿模型编写者（0.058）通过了其预注册测试。然而，仍有0.168和0.153的困难错误程序通过了检查。

    arXiv:2611.08269v1 Announce Type: cross  Abstract: After a runtime or API migration, an LLM-repaired procedure can run cleanly yet return wrong results, and a release check on typical test inputs ships it. We report a preregistered study of 73 tasks covering 60 documented changes in CPython, Node.js and SQLite. Every candidate is executed on the real runtimes against withheld contracts. Over the complete repair space, the all-assertions gate on typical inputs released 0.410 of wrong repairs. One edge input written from the release note cut this to 0.065, and to 0.061 on 15 unseen clusters. Three inputs from a blind GPT-5.6-luna writer that never saw the program reached 0.070. Two further frozen stages enlarged the menu to 1491 fragments and then added 233 hard distractors that match the legacy output on typical inputs. The edge input (0.058) and the frontier-model writer (0.058) passed their preregistered tests. Yet 0.168 and 0.153 of the hard wrong programs still passed. Two local-mod
    
[^187]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^188]: 一致性并非理解：面向神经场权重读取器的轨道审计

    Consistency Is Not Understanding: Orbit Audits for Neural Field Weight Readers

    [https://arxiv.org/abs/2611.05798](https://arxiv.org/abs/2611.05798)

    该论文提出“轨道审计”方法——利用保持函数等价的隐藏神经元置换和正 ReLU 缩放变换来检验从神经场权重中读取视觉内容的模型，发现现有读取器在等价变换下准确率大幅下降甚至对相同输入给出不一致答案，证明其表面上的输出一致性并不等于对权重所表示函数的真正理解。

    

    从神经网络权重中识别视觉内容，需要将功能性信息与任意的参数坐标区分开来。我们通过隐藏神经元置换以及保持所表示函数不变的正向 ReLU 缩放来审计这种分离。该研究包含 480 个独立拟合的形状场和 1,000 个 MNIST 场，变换后的副本按其来源分组。在形状识别任务上，一个统计读取器在联合变换下准确率从 \ShapeRawStats% 下降到 \ShapeJointStats%；而一个归一化集合读取器则保持 \ShapeNormSet% 的准确率，且预测完全不变。四个小型冻结语言模型通过下一词元打分获得了 \LMRange% 的原始准确率。在另一项包含 1,920 个评分会话的四配置推理实验中，生成式答案获得了 \FrontRawRange% 的原始准确率，然而对相同来源重新使用完全相同的原始提示时，模型在 \FrontRawRepeatRange% 的样本上给出了相互矛盾的判断。规范化使得等价的提示变得……（摘要原文在此处截断）

    arXiv:2611.05798v1 Announce Type: new  Abstract: Recognizing visual content from neural-network weights requires separating functional information from arbitrary parameter coordinates. We audit this separation with hidden-neuron permutations and positive ReLU rescalings that preserve the represented function. The study contains 480 independently fitted shape fields and 1,000 MNIST fields, with transformed copies grouped by origin. On shape recognition, a statistics reader falls from \ShapeRawStats\% to \ShapeJointStats\% accuracy under joint transformations; a normalized set reader retains \ShapeNormSet\% with unchanged predictions. Four small frozen language models obtain \LMRange\% raw accuracy with next-token scoring. A separate four-configuration reasoning experiment with 1,920 scored sessions obtains \FrontRawRange\% raw accuracy with generated answers, while fresh identical raw prompts disagree on \FrontRawRepeatRange\% of origins. Canonicalization makes equivalent prompts identi
    
[^189]: 开放多智能体系统中强化学习的公理化框架

    An Axiomatic Framework for Reinforcement Learning in Open Multi-Agent Systems

    [https://arxiv.org/abs/2611.03301](https://arxiv.org/abs/2611.03301)

    本文提出一个以 Shapley 公理为结构性约束的公理化框架，通过新颖的 NAHT 分解刻画开放与完全可控多智能体系统的本质差异，为开放多智能体系统的强化学习奠定合作博弈论基础，并证明对个体价值函数施加这些公理可恢复 Dec-POMDP 的 Shapley 值。

    

    开放多智能体系统（其中只有一部分智能体是可控的）是智能电网等现实应用领域的核心，但此类系统缺乏一个有原则的强化学习算法设计基础。我们通过 n 智能体临时团队协作（n-agent ad hoc teamwork, NAHT）这一新兴的开放多智能体环境学习框架来研究该设定。我们提出了一个公理化框架，从 Shapley 公理（有效性、对称性与线性性）出发推导价值函数学习，将这些公理视为结构性约束而非启发式设计选择。这一表述建立在新颖的 NAHT 分解之上，该分解刻画了开放系统与完全可控多智能体系统之间的根本差异，并为学习建立了合作博弈论基础。我们进一步证明，在学习个体价值函数时强制执行这些公理可以恢复 Dec-POMDP 的 Shapley 值，从而在……之间架起一座有原则的桥梁（原文摘要在此处截断）。

    arXiv:2611.03301v3 Announce Type: replace-cross  Abstract: Open multi-agent systems, where only a subset of agents are controllable, are central to real-world domains such as smart grids, yet lack a principled foundation for reinforcement learning algorithm design. We study this setting through n-agent ad hoc teamwork (NAHT), a recent framework for learning in open multi-agent environments. We propose an axiomatic framework that derives value function learning from the Shapley axioms (Efficiency, Symmetry and Linearity), treating them as structural constraints rather than heuristic design choices. This formulation builds on a novel NAHT decomposition, which characterises the fundamental difference between open and fully controllable multi-agent systems, and establishes a cooperative game-theoretic foundation for learning. We further show that enforcing these axioms on learning individual value functions recovers the Shapley value for Dec-POMDPs, providing a principled bridge between re
    
[^190]: 一个偏移量，两个方向：基于子群偏移的基线调整

    One Offset, Two Directions: Baseline Adjustment via Subgroup Offsets

    [https://arxiv.org/abs/2611.00782](https://arxiv.org/abs/2611.00782)

    该论文提出用“子群偏移”（子群均值与全局均值的偏差）配合一个带符号的系数 λ 来调整 GRPO 的基线，以解决提示内采样输出因异质子群而产生不同奖励分布时的信用分配问题，且 λ 的正负号给出两种相反的基线调整方向。

    

    组相对策略优化（GRPO）通过相对于提示内平均奖励——即组内共享的基线——来估计优势。在实践中，同一提示内的采样输出（rollouts）并不总是同质的，它们在求解策略和推理轨迹上存在差异，并可能据此聚合成子群。我们进一步发现，这些异质子群可能表现出各不相同的奖励分布。然而，全局基线将它们压缩到单一的信用轴上，此时信用既不反映某个采样输出在其子群内的相对位置，也不反映该子群相对于其余部分的地位。因此，我们研究当一个提示的采样输出落入异质子群时如何调整基线。我们将“子群偏移”定义为子群均值相对全局均值的偏差，并通过一个带符号的系数 λ 沿该偏移方向移动基线。λ 的符号产生两个相反的方向：……（原文摘要在此处截断）

    arXiv:2611.00782v2 Announce Type: replace  Abstract: Group Relative Policy Optimization (GRPO) estimates advantages against the within-prompt mean reward, a baseline shared across the group. In practice, rollouts within a prompt are not always homogeneous, differing in their solution strategies and reasoning trajectories, along which they may group into subgroups. We further find that these heterogeneous subgroups may exhibit distinct reward distributions. The global baseline, however, collapses them onto a single credit axis, where the credit reflects neither a rollout's standing within its subgroup nor the subgroup's standing relative to the rest. We therefore study how to adjust the baseline when the rollouts of a prompt fall into heterogeneous subgroups. We define the \emph{subgroup offset} as the deviation of a subgroup mean from the global mean, and shift the baseline along it by a single signed coefficient $\lambda$. The sign of $\lambda$ yields two opposing directions: \emph{Al
    
[^191]: EmbodiExp：面向具身智能体的可组合经验迁移

    EmbodiExp: Compositional Experience Transfer for Embodied Agents

    [https://arxiv.org/abs/2610.39180](https://arxiv.org/abs/2610.39180)

    EmbodiExp 提出一种以结果为依据的子目标经验表示，使具身智能体能够在任务执行过程中在线组合与迁移经验，并借助子目标层面的结果归因来利用失败轨迹。

    

    arXiv:2610.39180v1 公告类型：cross 摘要：经验可以帮助具身智能体将高层规划转化为落地动作，但跨任务迁移经验仍然充满挑战：先前有效的策略可能不适用于当前状态，或者无法产生预期的效果。我们提出 EmbodiExp，一个用于构建可迁移经验并在具身任务执行过程中在线组合这些经验的框架。我们的关键观察是：任务往往共享一些子目标，而这些子目标的完成情况可以独立于整体任务的成功与否进行评估。因此，我们引入了一种以结果为依据的子目标经验，它明确了子目标、适用条件、可调整的策略以及可验证的结果。在执行过程中，适用的经验引导动作生成，而经过验证的结果可以建立后续经验所需的条件，从而实现在线组合。子目标层面的结果归因进一步实现了对失败的利用：失败的轨迹……

    arXiv:2610.39180v1 Announce Type: cross  Abstract: Experience can help embodied agents turn high-level plans into grounded actions, but transferring it across tasks remains challenging: previously effective strategies may be inapplicable to the current state or fail to produce their intended outcomes. We present EmbodiExp, a framework for constructing transferable experiences and composing them online during embodied task execution. Our key observation is that tasks often share subgoals whose completion can be assessed independently of overall task success. We therefore introduce an outcome-grounded subgoal experience that specifies a subgoal, applicability conditions, an adaptable policy, and a verifiable outcome. During execution, applicable experiences guide action generation, while verified outcomes can establish the conditions required by subsequent experiences, enabling online composition. Subgoal-level outcome attribution further enables failure utilization: failed trajectories 
    
[^192]: 在受监管金融领域落地治理型智能体工作流：基于 Zafin AIOS 的“零号客户”案例研究

    Operationalizing Governed Agentic Workflows in Regulated Finance: A Customer Zero Case Study using Zafin AIOS

    [https://arxiv.org/abs/2610.33253](https://arxiv.org/abs/2610.33253)

    本文提出Zafin AIOS这一端到端编排与控制平台，并通过“零号客户”内部生产案例研究，验证了治理型智能体工作流在受监管金融行业中从试点走向生产落地的可行性与有效性。

    

    金融和医疗等受监管行业面临着关键的治理挑战，这些挑战阻碍了AI辅助软件开发的采用，也阻碍了从实验性试点向生产环境工作的过渡。本文介绍了Zafin AIOS，一个专为在金融等受监管行业中部署AI而设计的端到端编排平台和控制平面。我们提供了架构原则，以及AI辅助工作负载与人工主导工作负载之间的实证比较。为了对该框架进行实证验证，我们对Zafin的内部企业工作流程进行了案例研究分析，我们将其称为“零号客户”。本文为受监管行业中的AI辅助软件工程贡献了一个生产案例研究。我们在两个不同的生产代码库上评估了使用Zafin AIOS的结果，一个用于定价执行平台，另一个用于客户交易谈判平台。我们的贡献是一个……

    arXiv:2610.33253v1 Announce Type: cross  Abstract: Regulated industries such as finance and healthcare face critical governance challenges that hamper the adoption of AI-assisted software development and hinder the transition from experimental pilots to production work. This paper presents Zafin AIOS, an end-to-end orchestration platform and control plane designed to deploy AI within regulated industries such as finance. We provide the architectural principles and an empirical comparison between AI-assisted and human-led workloads. To validate this framework empirically, we conduct a case study analysis of Zafin's internal enterprise workflows, which we refer to as Customer Zero. This paper contributes a production case study of AI-assisted software engineering in regulated industries. We evaluate the results of using Zafin AIOS with two different production codebases, one for the pricing execution platform and the other for the client deal negotiation platform. Our contribution is a c
    
[^193]: 大语言模型能否改进自身的执行框架？基于ARC-AGI-3的对照研究

    Can LLMs Improve Their Own Harness? A Controlled Study on ARC-AGI-3

    [https://arxiv.org/abs/2610.30499](https://arxiv.org/abs/2610.30499)

    该论文通过受控实验首次系统考察前沿大语言模型能否自主构建并迭代改进自己的执行框架，让模型在ARC-AGI-3交互式基准上独立编写、使用并两次修订完整框架代码库，再于二十个从未参与改进的留出游戏上检验其泛化能力。

    

    通用智能系统能够高效地习得新技能，而无需人类为每个新任务重新设计其软件脚手架。然而，语言模型很少单独行动——执行框架是一种符号程序，它代表模型编排各类调用，负责处理感知、记忆、工具使用、异常恢复和推理分配等事务。ARC-AGI-3是一个交互式基准测试，要求智能体探索陌生环境、在行动过程中自主弄清目标并持续学习。在该基准上，由人类引导的系统已经在公开游戏中报告了满分区分数，说明有效的设计是确实存在的。本文提出的问题是：前沿模型能否自行构建并持续改进这样一个执行框架？通过一个固定的编码引导流程，每个模型先构建一个完整的任务执行框架，通过该框架在五个开发游戏上执行任务，并对整个代码库进行两次修订。随后，每个版本都在二十个从未参与过适应过程的留出游戏上进行评估。

    arXiv:2610.30499v1 Announce Type: cross  Abstract: A generally intelligent system acquires new skills efficiently without requiring humans to redesign its software scaffold for every new task. Yet language models rarely act alone. A harness is a symbolic program that orchestrates calls to the model, handling perception, memory, tool use, recovery, and inference allocation on the model's behalf. ARC-AGI-3 is an interactive benchmark in which agents must explore unfamiliar environments, figure out their goals on the fly, and keep learning as they go. On it, human-steered systems already report ceiling scores across the public games, so we know effective designs exist. We ask whether frontier models can build and improve such a harness on their own. Through a fixed coding bootstrap, each model constructs a complete task harness, acts through that harness on five development games, and revises the entire codebase twice. Every version is evaluated on twenty games that never influence adapta
    
[^194]: Brain2Vec：一种用于结构化脑部MRI的自监督3D基础模型

    Brain2Vec: A Self-Supervised 3D Foundation Model for Structural Brain MRI

    [https://arxiv.org/abs/2610.28411](https://arxiv.org/abs/2610.28411)

    Brain2Vec是一个在74,425个T1加权脑部MRI扫描上预训练的自监督3D基础模型，既可作为下游任务微调的权重初始化骨干，也可作为独立特征提取器，在脑年龄预测等任务中展现出更高的精度和更快的收敛速度。

    

    深度学习模型已成为医学影像领域不可或缺的工具，然而由于标注数据有限和计算资源受限，针对多样化任务进行端到端训练仍然充满挑战。为解决这一问题，我们提出了Brain2Vec，这是一个在汇集自多项研究的74,425个T1加权磁共振成像（MRI）扫描数据上预训练的基础模型。Brain2Vec既可以作为权重初始化的骨干网络用于下游任务的微调，也可以作为独立的特征提取器使用。我们从三个方面对Brain2Vec进行了评估：下游预测性能、收敛速度和嵌入质量。在脑年龄预测任务中，使用Brain2Vec的预训练权重作为初始化并在标注数据上进行微调，结果表明与随机初始化相比，最终精度得到提升且收敛速度更快。此外，其冻结嵌入特征在五个下游分类任务中的四个上优于基于解剖学定义的体积特征。

    arXiv:2610.28411v1 Announce Type: cross  Abstract: Deep learning models have become essential tools in medical imaging, yet training them end-to-end for diverse tasks remains challenging due to limited labeled data and computational constraints. To address this, we introduce Brain2Vec, a foundation model pre-trained on 74,425 T1-weighted Magnetic Resonance Imaging (MRI) scans pooled from multiple studies. Brain2Vec can serve both as a weight initialization backbone for fine-tuning downstream tasks and as a standalone feature extractor. We evaluated Brain2Vec in three tasks: downstream predictive performance, convergence speed, and embedding quality. Using Brain2Vec's pretrained weights as initialization and fine-tuning on labeled data for a brain age prediction task, results demonstrate improved final accuracy and faster convergence compared to random initialization. Additionally, its frozen embeddings outperform anatomically defined volumetric features on four of five downstream class
    
[^195]: TimesJev：基于系统1充分性的时间序列快速思考

    TimesJev: Thinking Fast on Time Series with System 1 Sufficiency

    [https://arxiv.org/abs/2610.26002](https://arxiv.org/abs/2610.26002)

    本文提出TimesJev——首个时间序列系统1模型，通过形式化系统1充分性并构建保留关键频率模式的压缩表示，使模型无需长思维链即可对时间序列实现即时推理，从而大幅降低推理延迟。

    

    大语言模型（LLMs）最近在时间序列推理方面展现出强大潜力，而对时间序列进行推理的模型通常依赖长思维链。然而，这一过程往往消耗数千个token，直接导致推理阶段产生显著延迟。相比之下，以Jev为代表的系统1模型能够即时做出决策，自然可以成为一种替代方案，但原始时间序列与这些模型的嵌入空间并不兼容，从而阻碍了推理性能，尤其当输入序列包含数千个数值点时。在本文中，我们进行了非常早期的尝试，提出了一种时间序列系统1模型，即TimesJev。具体而言，我们首先形式化了时间序列推理中的系统1充分性，这启发了保留关键频率模式的时间序列压缩表示。为构建这种表示，我们推导了一个降维目标……（摘要原文在此处截断）

    arXiv:2610.26002v3 Announce Type: replace  Abstract: Large language models (LLMs) have recently shown strong potential for time series reasoning, while models that reason over time series are usually with long chains of thought. However, this process often consumes thousands of tokens, directly leading to substantial latency during inference. In contrast, System 1 models such as Jev make decisions instantly and can naturally be an alternative, but raw time series are not compatible with the embedding space of these models, thus hindering the reasoning performance, especially when input series contain thousands of numeric points. In this paper, we make a very early attempt to propose a time series System 1 model, namely TimesJev. Specifically, we first formalize System 1 sufficiency in time series reasoning, which motivates a compressed representation for time series with key frequency patterns preserved. To construct this representation, we derive a dimension reduction objective that c
    
[^196]: 大语言模型中的多元对齐：目标、人类参照与偏好表征

    Pluralistic Alignment in Large Language Models: Objectives, Human References, and Preference Representations

    [https://arxiv.org/abs/2610.24471](https://arxiv.org/abs/2610.24471)

    该论文提出了一个双层分类框架来系统梳理大语言模型多元对齐研究，第一层区分三种对齐目标（表征多元观点的Overton多元性、适应特定视角的可引导多元性、匹配人群分布的分布多元性），第二层区分三类人类参照（聚合人群、国家和地区、子群体），并将偏好表征、实现机制等视为贯穿性设计选择。

    

    多元对齐旨在表征人类偏好中有意义的差异，而不是将异质的判断坍缩为单一目标。然而，现有工作常常混淆三个不同的问题：模型应展现什么样的多元行为、谁的偏好定义了参照标准，以及什么样的证据足以证明成功。我们围绕一个双层分类体系来组织相关文献。第一层区分三种多元对齐目标：表征一系列得到人类支持的观点（Overton多元性）、忠实地适应特定观点或偏好条件（可引导多元性），以及匹配目标人群中所观察到的回应分布（分布多元性）。第二层区分三类人类参照：聚合人群、国家和地区、以及子群体。偏好表征、实现机制、模态和证据质量则被视为贯穿各层的设计选择，而非附加特性。

    arXiv:2610.24471v2 Announce Type: replace-cross  Abstract: Pluralistic alignment aims to represent meaningful variation in human preferences rather than collapse heterogeneous judgments into a single target. Yet existing work often mixes three distinct questions: what pluralistic behavior a model should exhibit, whose preferences define the reference, and what evidence is sufficient to demonstrate success. We organize the literature around a two-layer taxonomy. Layer 1 distinguishes representing a range of human-supported viewpoints (Overton Pluralism), adapting faithfully to a specified perspective or preference condition (Steerable Pluralism), and matching the response distribution observed in a target population (Distributional Pluralism). Layer 2 distinguishes aggregate populations, countries and regions, and subgroups as human references. Preference representation, implementation mechanism, modality, and evidence quality are treated as cross-cutting design choices rather than addi
    
[^197]: FlashRT-PAI：快速内核并不等于面向物理AI的快速流水线

    FlashRT-PAI: Fast Kernels Are Not Fast Pipelines for Physical AI

    [https://arxiv.org/abs/2610.24349](https://arxiv.org/abs/2610.24349)

    该论文提出FlashRT-PAI低延迟推理引擎，通过深度内核融合、启动重叠和执行契约等模型-硬件流水线协同设计机制，消除物理AI系统实时推理中的系统级开销瓶颈，突破了传统孤立内核优化的局限。

    

    视觉-语言-动作模型（VLA）、视频-动作模型（VA）和世界动作模型（WAM）构成了现代物理AI系统的骨干，然而它们的实时单批次推理仍然是一个关键瓶颈。延迟并非由单个内核计算主导，而是由系统级开销决定：数千次短内核启动、跨算子边界的中间数据移动，以及连续决策之间的冗余重复计算。通用图编译器无法解决这些开销，因为它们孤立地优化单个调用，缺乏对跨调用状态语义和全局流水线结构的可见性。我们提出FLASHRT-PAI，一个围绕模型-硬件流水线协同设计而非孤立内核优化构建的低延迟推理引擎。FLASHRT-PAI结合了三种核心机制：深度内核融合与启动重叠消除了算子间边界和启动间隙；执行契约将跨调用状态（原文在此处截断）……

    arXiv:2610.24349v2 Announce Type: replace  Abstract: Vision-language-action (VLA), video-action (VA), and world action models (WAMs) form the backbone of modern physical AI systems, yet their real-time single-batch inference remains a defining bottleneck. Latency is dominated not by individual kernel computation, but by system-level overhead: thousands of short kernel launches, intermediate data movement across operator boundaries, and redundant recomputation across consecutive decisions. General-purpose graph compilers cannot address these costs, as they optimize individual calls in isolation and lack visibility into cross-call state semantics and global pipeline structure. We present FLASHRT-PAI, a low-latency inference engine built around model-hardware pipeline co-design rather than isolated kernel optimization. FLASHRT-PAI combines three core mechanisms: deep kernel fusion with launch overlap eliminates inter-operator boundaries and launch gaps; an execution contract formalizes th
    
[^198]: TorchTPU——一种用于TPU上原生PyTorch执行的混合即时/编译架构

    TorchTPU - A Hybrid Eager/Compiled Architecture for Native PyTorch Execution on TPU

    [https://arxiv.org/abs/2610.24161](https://arxiv.org/abs/2610.24161)

    TorchTPU 是 Google 与 Meta 合作的开源项目，通过混合即时/编译架构为 TPU 提供原生 PyTorch 后端，实现开箱即用的模型迁移并达到业界竞争力的训练与推理性能。

    

    TorchTPU 是 Google 与 Meta 之间的开源合作项目，为 Google TPU 引入了原生且符合设计规范的 PyTorch 后端。它通过即时执行优化、torch.compile 集成以及原生硬件加速，实现了开箱即用的模型执行与 TPU 迁移，同时提供了具有行业竞争力的推理和训练性能。

    arXiv:2610.24161v2 Announce Type: replace  Abstract: TorchTPU is an open-source collaboration between Google and Meta that introduces a native and principled PyTorch backend for Google TPUs. It enables seamless out-of-the-box model execution and TPU migration while delivering industry-competitive inference and training performance through eager execution optimizations, torch.compile integration, and native hardware acceleration.
    
[^199]: 自构建实验室

    Self-Building Labs

    [https://arxiv.org/abs/2610.23826](https://arxiv.org/abs/2610.23826)

    该论文提出“自构建实验室”（SBL），能根据高层科学目标自主设计并搭建实验室的布局、数字孪生、机器人技能与实验流程，突破了现有自动驾驶实验室灵活性不足这一制约科学发现规模化的瓶颈。

    

    自动驾驶实验室通过自动化实验执行与优化来加速科学发现。AI智能体如今越来越能够提出新的实验方案，但现有的自动驾驶实验室只能运行其最初构建时所针对的实验流程。将其适配到新实验仍需要大量的人工工程工作，这使得实验室的灵活性成为规模化科学发现的瓶颈。我们提出了自构建实验室，它能够根据一个高层次的科学目标，自主设计、配置并运行自己的实验室设施。SBL可以从任何现有条件出发——从空房间到一个已运行的自动驾驶实验室——并构建所缺失的能力：实验室布局、数字孪生、机器人技能以及实验工作流。我们展示了该系统在不同机器人和不同设施上的运行情况，从单一工作站扩展到多机器人协同工作的实验室。我们通过真实世界的q……（摘要原文在此处截断）

    arXiv:2610.23826v1 Announce Type: new  Abstract: Self-driving laboratories (SDLs) accelerate scientific discovery by automating experimental execution and optimization. AI agents can increasingly propose new experiments, but existing SDLs can only run the procedures they were built for. Adapting them to a new experiment still takes extensive manual engineering, which makes laboratory flexibility the bottleneck for scaling discovery. We introduce the self-building laboratory (SBL), which designs, configures, and operates its own laboratory setup given a high-level scientific goal. SBL starts from whatever already exists, from an empty room to a working SDL, and builds the missing capabilities: the lab layout, its digital twin, robot skills, and experimental workflows. We demonstrate the system working across different robots and setups, scaling from a single workstation to laboratories with several robots working together. We validate SBL's end-to-end capabilities through a real-world q
    
[^200]: 学会读出：语言模型预训练中的反嵌入动态

    Learning to Read Out: Unembedding Dynamics in Language Model Pretraining

    [https://arxiv.org/abs/2610.21025](https://arxiv.org/abs/2610.21025)

    提出参数轨迹交叉编码方法，仅利用检查点权重直接追踪语言模型预训练中读出层的动态，发现Pythia在预训练0.4%–1.1%的早期窗口内读出层发生重组，且隐藏状态在读出层表达之前就已编码任务信息。

    

    语言模型是在何时获得某种能力的？发展性研究通常通过追踪模型何时开始生成正确的词元来回答这个问题。但这一预测需要经过反嵌入矩阵（即读出层），而读出层本身在预训练过程中也仍在被学习。我们证明，模型的隐藏状态可以在其自身的读出层表达任务信息之前就已编码该信息，因此从输出中读取的时间线混淆了模型所编码的内容与读出层所表达的内容。为了直接追踪读出层，我们引入了参数轨迹交叉编码方法，该方法在每个检查点为每个词元的读出行拟合一个稀疏字典。与作用于激活值的交叉编码器不同，它只需要检查点的权重。我们利用该方法定位到一个狭窄的早期窗口——即Pythia预训练进程的0.4%–1.1%阶段——在此期间读出层发生重组，WordNet词类变得可以从其行中解码。到那时，对隐藏状态的线性探针已经能够解码

    arXiv:2610.21025v1 Announce Type: cross  Abstract: When does a language model acquire a capability? Developmental studies usually answer this by tracking when models start producing the correct token. But that prediction passes through the unembedding matrix, or readout, which is itself still being learned during pretraining. We show that a model's hidden states can encode task information before its own readout expresses it, so timelines read from outputs conflate what the model encodes with what its readout expresses. To track the readout directly, we introduce parameter-trajectory crosscoding, which fits one sparse dictionary to every token's readout row at every checkpoint. Unlike crosscoders on activations, it needs only checkpoint weights. We use it to locate a narrow early window, 0.4-1.1% of the way through Pythia's pretraining, in which the readout reorganizes and WordNet word categories become decodable from its rows. By then, linear probes on the hidden states already decode
    
[^201]: 注入轮次而非标记：学习在何处重定向大语言模型推理以实现自我纠错

    Inject a Turn, Not a Token: Learning Where to Redirect LLM Reasoning for Self-Correction

    [https://arxiv.org/abs/2610.13672](https://arxiv.org/abs/2610.13672)

    该论文提出无需微调的 UPTURN 自我纠错框架，通过仅用最终答案正确性训练的 13 万参数轻量 MLP 头学习失败推理中的最佳干预位置，并将纠错指令作为新的用户轮次而非行内标记注入，从而有效重定向模型推理，在四个推理模型和六个数学基准上均提升准确率。

    

    大语言模型难以可靠地识别并纠正自身推理中的错误。与以往通过微调大语言模型来生成修订或反馈的训练类方法不同，我们学习在失败的推理生成过程中于何处进行干预，并确定如何对其重定向。为了选择干预位置，我们在冻结的编码器上仅利用最终答案的正确性训练了一个 13 万参数的 MLP 头。对于重定向方式，我们发现在固定的生成预算下，已测试的行内注入无法可靠地改变推理方向，而将相同指令作为新的用户轮次呈现则能改变模型的走向。由此得到的纠错框架 UPTURN（在习得位置注入用户提示轮次）无需对大语言模型进行任何微调：它保留选定的前缀，将一条指令作为新的用户轮次注入，并重新生成后缀。在四个推理模型和六个数学基准上，UPTURN 在所有设置下均提升了平均准确率。

    arXiv:2610.13672v1 Announce Type: cross  Abstract: Large language models struggle to reliably identify and correct errors in their own reasoning. Unlike prior training-based approaches that fine-tune an LLM to generate revisions or feedback, we learn where to intervene in failed rollouts and determine how to redirect them. To choose where to intervene, we train a 130K-parameter MLP head on a frozen encoder using only final-answer correctness. For redirection, we find that under a fixed generation budget, the tested inline injections fail to reliably redirect reasoning, whereas presenting the same instruction in a new user turn changes the model's course. The resulting correction framework, UPTURN (User-Prompt TURN injection at a learned position), requires no LLM fine-tuning: it retains a selected prefix, injects an instruction as a new user turn, and regenerates the suffix. Across four reasoners and six math benchmarks, UPTURN improves average accuracy in every setting over a 32-rollo
    
[^202]: 何时重新思考：面向视觉语言模型的多视角自验证学习

    When to Rethink: Learning Multi-Perspective Self-Verification for Vision-Language Models

    [https://arxiv.org/abs/2610.07018](https://arxiv.org/abs/2610.07018)

    该论文提出MOTIVE框架，通过多视角自验证与可靠性引导的选择性重新思考机制，解决了视觉语言模型中因单一验证标准或固定提示词导致的可靠性估计不完整且不稳定的问题。

    

    视觉语言模型（VLM）在多模态推理中已取得强大的性能，但它们仍然容易生成看似合理却不正确的答案。自验证提供了一种无需依赖外部评判者即可提高答案可靠性的实用方法，但现有方法通常依赖于单一的验证标准或固定的提示词，导致可靠性估计不完整且不稳定。我们首先系统性地分析了验证器能力和提示词设计如何影响验证性能。我们的研究结果表明，更强的验证器能够提供更可靠的判断，而验证性能对提示词的选择高度敏感，没有单一提示词能够在所有任务中始终保持优势。基于这些发现，我们提出了MOTIVE，一个结合可靠性引导的选择性重新思考的多视角自验证框架，用于实现可靠的多模态推理。

    arXiv:2610.07018v1 Announce Type: new  Abstract: Vision-language models (VLMs) have achieved strong performance in multimodal reasoning, yet they remain prone to generating plausible but incorrect answers. Self-verification offers a practical way to improve answer reliability without relying on external judges, but existing methods typically depend on a single verification criterion or fixed prompt, resulting in incomplete and unstable reliability estimates. We first systematically analyze how verifier capability and prompt design affect verification performance. Our findings show that stronger verifiers provide more reliable judgments, while verification performance is highly sensitive to prompt choice, with no single prompt consistently dominating across tasks. Guided by these findings, we propose \texttt{MOTIVE}, a \textbf{M}ulti-View Self-Verificati\textbf{O}n wi\textbf{T}h Rel\textbf{I}ability-Guided Selecti\textbf{VE} Rethinking framework for reliable multimodal reasoning. \textt
    
[^203]: 基于知识图谱推理的循证患者特异性临床解释

    Proof-Grounded Patient-Specific Clinical Explanations from Knowledge-Graph Reasoning

    [https://arxiv.org/abs/2610.06549](https://arxiv.org/abs/2610.06549)

    该论文提出 CKG 临床解释引擎，在免训练的临床知识图谱推理器下游将患者推理状态转化为可审计、带溯源链接且符合政策许可的临床解释，并在逾万名患者的大规模数据上验证了事实与报告的完整性和数值保真度。

    

    临床决策支持系统往往缺乏从观察和编码知识到结论与建议之间的可审计链接。在一个冻结的、免训练的临床知识图谱推理器的下游，CKG 临床解释引擎将患者推理状态和疾病知识转换为类型化事实、显式规则追踪、带溯源链接的结论以及符合政策许可的建议。通过区分可用性、表示完整性和疾病特异性激活，该引擎能够将观察到的零激活和部分证据与未观察到的证据区分开来。语言生成评估仍是未来的工作。在六个可用的数据工作簿（11,289 名患者、33,867 条患者-疾病追踪记录、1,715,928 行特征-证据数据）上，IG 范围有效性、知识溯源、跨工作表数值保真度（203,202/203,202）以及导出事实/报告字段完整性（33,867/33,867）均达到 100.00%。可用性-表示一致性达到 99.2%。

    arXiv:2610.06549v2 Announce Type: replace  Abstract: Clinical decision support can lack auditable links from observations and encoded knowledge to conclusions and recommendations. Downstream of a frozen, training-free clinical knowledge-graph reasoner, the CKG Clinical Explanation Engine converts patient inference states and disease knowledge into typed facts, explicit rule traces, provenance-linked conclusions, and policy-licensed recommendations. Separate availability, representation completeness, and disease-specific activation distinguish observed zero-activation and partial evidence from unobserved evidence. Language-generation evaluation remains future work. Across six usable workbooks (11,289 patients, 33,867 patient-disease traces, 1,715,928 feature-evidence rows), IG-range validity, knowledge provenance, cross-sheet numerical fidelity (203,202/203,202), and exported fact/report field presence (33,867/33,867) reached 100.00%. Availability-representation consistency reached 99.2
    
[^204]: 世界模型应该遗忘什么？面向持续适应的分层保留机制

    What Should World Models Forget? Stratified Retention for Continual Adaptation

    [https://arxiv.org/abs/2610.03713](https://arxiv.org/abs/2610.03713)

    提出持续世界模型应按不变性时间尺度对知识进行分层保留——物理规律等不变量永不可修改，而随环境过时的实例级事实应被主动遗忘——从而将“遗忘”重新定义为必要行为而非失败。

    

    持续学习将在已见数据上的性能退化视为失败的证据，这一惯例继承自预测目标平稳的设定，即正确标签会永远保持正确。世界模型不满足这一条件：它们的预测目标是环境，而环境会变化，因此获取时准确的知识之后可能变为错误，丢弃这些知识是必要行为而非缺陷。非平稳的真实标签问题在概念漂移文献和语言模型的时间事实性研究中已被充分探讨，但尚未在世界模型中被形式化，而世界模型的独特之处在于它们还编码了绝不能被修改的知识。我们认为，持续世界模型需要按不变性时间尺度分层的保留机制，将诸如物理规律和物体恒存性等绝不应被修改的不变量，与应随环境变化而尽快被更新的实例级事实区分开来。

    arXiv:2610.03713v1 Announce Type: cross  Abstract: Continual learning treats degradation on previously seen data as evidence of failure, a convention inherited from settings with a stationary prediction target, where a correct label remains correct indefinitely. World models do not satisfy this condition. Their prediction target is the environment, which changes, so knowledge that was accurate when acquired may later become false, and discarding it is required behavior rather than a defect. Non-stationary ground truth is well studied in the concept drift literature and in the temporal factuality of language models, but has not been formulated for world models, which are distinctive in that they also encode knowledge that must never be revised. We argue that continual world models require retention stratified by invariance timescale, separating invariants such as physics and object permanence, which must never be revised, from instance-level facts that should be revised as soon as the e
    
[^205]: 具有1比特反馈的近最优固定置信度最优臂识别

    Nearly Optimal Fixed-Confidence Best-Arm Identification with 1-Bit Feedback

    [https://arxiv.org/abs/2610.02771](https://arxiv.org/abs/2610.02771)

    本文在严格1比特反馈约束下提出了近最优的固定置信度最优臂识别算法，通过随机化阈值查询与自适应截断技术实现了间隙自适应的样本复杂度，并给出了相匹配的信息论下界。

    

    我们研究在严格1比特反馈约束下的固定置信度最优臂识别问题。在每一轮中，学习者选择一个臂和一个查询集合，并且仅接收一个比特，该比特指示采样的奖励是否属于该集合。我们考虑一种具有逐臂定位的无分布有限方差设置，在这种设置下，直接的经验均值估计不再可用，截断处理变得不可避免。我们首先基于随机化阈值查询和截断尾积分恒等式，构建了一个时间一致的1比特均值估计基元。随后，我们将该基元嵌入到候选-挑战者式的最优臂识别算法中。固定截断算法提供了简单的任意时刻（ε,δ)-PAC保证，而分阶段自适应截断算法则将截断水平与当前分辨率相匹配，从而产生了间隙自适应的样本复杂度。我们还证明了一个K臂最坏情况的信息论下界（摘要原文在此处截断）。

    arXiv:2610.02771v1 Announce Type: cross  Abstract: We study fixed-confidence best-arm identification under strict 1-bit feedback constraints. At each round, the learner selects an arm and a query set, and receives only a single bit indicating whether the sampled reward belongs to that set. We consider a distribution-free finite-variance setting with arm-wise localization, where direct empirical mean estimation is no longer available and clipping becomes unavoidable. We first formulate a time-uniform 1-bit mean-estimation primitive based on randomized threshold queries and a clipped tail-integral identity. We then embed this primitive into candidate-challenger best-arm identification algorithms. A fixed-clipping algorithm gives a simple anytime $(\epsilon,\delta)$-PAC guarantee, while a phased adaptive-clipping algorithm matches the clipping level to the current resolution and yields a gap-adaptive sample complexity. We also prove a $K$-arm worst-case information-theoretic lower bound s
    
[^206]: 基于智能体编码任务强化学习的跨基准迁移

    Cross-Benchmark Transfer from RL on Agentic Coding Tasks

    [https://arxiv.org/abs/2610.00890](https://arxiv.org/abs/2610.00890)

    仅用强化学习在1,700个专家构建的智能体编码任务上后训练万亿参数混合专家模型，即可在六个外部基准上全面提升pass@1，证明所学能力能够跨基准迁移。

    

    编码智能体常常在“最后一公里”失败：它们完成了功能的大部分却遗漏某项需求、只测试其实现已经能处理的情况、破坏了本应保持不变的行为，或基于未经检验的假设进行验证。我们探究在专家构建的智能体编码任务上进行强化学习（RL）能否弥合这一差距，以及智能体所学内容能否迁移到训练分布之外。我们仅使用强化学习对 Kimi K2.7 Code（一个 1 万亿参数、激活参数 32B 的开源混合专家模型）在 1,700 个任务上进行后训练：其中 1,000 个仓库任务由隐藏的 fail-to-pass 测试和保护既有行为的 pass-to-pass 测试评分，700 个终端任务由专家编写的隐藏验证器评分。奖励为目标检查的通过比例，若任何 pass-to-pass 测试失败则奖励降为零。在秩为 32 的 LoRA 适配器上进行一个 epoch 的 GSPO 训练，在我们评估的六个外部基准上的 pass@1 均有提升，跨越三个……（原文摘要在此处截断）

    arXiv:2610.00890v1 Announce Type: cross  Abstract: Coding agents often fail in the last mile: they build most of a feature but drop a requirement, test only the cases their implementation already handles, break behavior that was supposed to stay intact, or validate against an unchecked assumption. We ask whether reinforcement learning (RL) on expert-built agentic coding tasks closes this gap, and whether what the agent learns transfers beyond the training distribution. We post-train Kimi K2.7 Code, a 1T-parameter (32B active) open-weight mixture-of-experts model, with RL alone on 1,700 tasks: 1,000 repository tasks graded by hidden fail-to-pass tests and by pass-to-pass tests of existing behavior, and 700 terminal tasks graded by expert-written hidden verifiers. The reward is the fraction of target checks passed and drops to zero if any pass-to-pass test fails. One epoch of GSPO on a rank-32 LoRA adapter improves pass@1 on each of the six external benchmarks we evaluated, across three 
    
[^207]: 基于StyleGAN中间潜在空间的储层物性参数化方法用于基于集合的数据同化

    Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN

    [https://arxiv.org/abs/2609.39626](https://arxiv.org/abs/2609.39626)

    本研究提出利用StyleGAN2的中间潜在空间对储层物性进行参数化，以克服集合数据同化方法在处理非高斯复杂地质（复杂相分布）条件下历史拟合性能下降的问题，并与VAE-GAN和潜在扩散模型等深度学习参数化方案进行了对比评估。

    

    集合平滑方法是当前最成功、最高效的历史拟合技术。然而，由于这些方法依赖于高斯假设，当先验地质以复杂相分布（非高斯）描述时，其性能会严重下降。因此，对于这些方法，需要应用高效的参数化技术。目前，执行参数化最有效的方法是深度学习模型。然而，鉴于现有深度学习模型的多样性，尽管已经评估过一些重要模型，但研究尚未确定哪种模型最适合与基于集合的方法配合使用。基于近期的文献综述，选出的最有前景的模型是VAE-GAN、潜在扩散模型和StyleGAN模型。作为本工作的创新之处，使用第二代StyleGAN（StyleGAN2）模型进行了数据同化，利用其

    arXiv:2609.39626v1 Announce Type: cross  Abstract: Ensemble smoothers are the most successful and efficient techniques currently available for history matching. However, because these methods rely on Gaussian assumptions, their performance is severely degraded when the prior geology is described in terms of complex facies distributions (non-Gaussian). In this way, for these methods, we need to apply efficient parameterization techniques. Currently, the most efficient methods for performing parameterization are deep learning models. However, given the variety of existing deep learning models, studies have not identified which is most suitable for use with ensemble-based methods, although some important models had already been evaluated. Based on a recent literature review, the most promising models selected were VAE-GAN, Latent Diffusion, and StyleGAN models. As a novel aspect of this work, data assimilation with the second generation of StyleGAN (StyleGAN2) model was performed using th
    
[^208]: MeanFlowAdvantage：面向少步平均速度生成器的稳定奖励微调

    MeanFlowAdvantage: Stable Reward Fine-Tuning for Few-Step Average-Velocity Generators

    [https://arxiv.org/abs/2609.37670](https://arxiv.org/abs/2609.37670)

    提出 MeanFlowAdvantage——一种带符号优势加权最小二乘目标，通过共享的 MeanFlow 导数校正解决平均速度生成器与优势奖励微调之间的失配问题，在保留原生少步采样的同时将奖励改进直接迁移到部署的流映射，并在 SD3.5-Medium 全部八项指标上取得提升。

    

    MeanFlow 通过预测区间平均速度实现高效的少步生成，但这种表示方式与奖励微调之间存在失配：现有的基于优势函数的目标通常定义在瞬时速度或等价的 x0 空间预测上，而推理阶段直接使用的是学习到的平均速度映射。我们提出 MeanFlowAdvantage，一种面向平均速度生成器的带符号优势加权最小二乘目标。其关键构造是使用一个共享的、梯度分离的 MeanFlow 导数校正，在预测空间中表达奖励目标，同时将 rollout（展开采样）与参考正则化转化为对推理时部署的平均速度网络的精确惩罚项。由此得到的公式保留了 MeanFlow 原生的少步采样器，并提供了一种将奖励改进直接迁移到已部署流映射的机制。在 SD3.5-Medium 上，MeanFlowAdvantage 在全部八项报告指标上均取得提升。

    arXiv:2609.37670v1 Announce Type: new  Abstract: MeanFlow enables efficient few-step generation by predicting interval-average velocities, but this representation creates a mismatch for reward fine-tuning: existing advantage-based objectives are typically defined on instantaneous velocities or equivalent $x_0$-space predictions, whereas inference directly uses the learned average-velocity map. We introduce MeanFlowAdvantage, a signed advantage-weighted least-squares objective for average-velocity generators. Our key construction uses a shared, detached MeanFlow derivative correction to express the reward objective in prediction space while making rollout and reference regularization exact penalties on the average-velocity network deployed at inference. The resulting formulation preserves MeanFlow's native few-step sampler and provides a direct mechanism for transferring reward improvements to the deployed flow map. On SD3.5-Medium, MeanFlowAdvantage improves all eight reported metrics 
    
[^209]: 从数据到程序：从经验数据快速直接地推断生成程序

    From Data to Program: Fast & Direct Generative Program Inference from Empirical Data

    [https://arxiv.org/abs/2609.35348](https://arxiv.org/abs/2609.35348)

    PRODiGI是一种预训练的数据到程序模型，能在单次前向传播中从经验数据直接推断出显式可执行的生成程序，配合程序空间微调在密度和分数估计上超越现有预训练模型并获得数倍加速。

    

    从有限样本集合中估计概率密度通常需要针对特定数据集进行模型拟合。我们提出PRODiGI，这是一个预训练的数据到程序模型，能够在单次前向传播中推断出显式、可执行的生成程序。PRODiGI在合成数据集与其真实程序的配对数据上进行预训练，通过模板预测和非自回归的程序参数解码，能够适应多种生成模型族和数据维度。其推断出的程序支持直接采样、密度与分数评估以及程序检查，且独立于预训练模型运行。我们进一步引入了程序空间微调方法，在保持模型参数不变的前提下，通过最大化经验数据似然来优化可微的程序参数。实验表明，PRODiGI-FT比现有预训练模型取得了更优的密度和分数指标，同时相比迭代方法提供了数倍的加速。

    arXiv:2609.35348v2 Announce Type: replace-cross  Abstract: Estimating probability densities from a finite set of samples typically requires dataset-specific model fitting. We introduce PRODiGI, a pretrained data-to-program model that infers an explicit, executable generative program in a single forward pass. Pretrained on synthetic datasets paired with their ground-truth programs, PRODiGI accommodates diverse generative families and data dimensionalities through template prediction and non-autoregressive program parameter decoding. Its inferred programs support direct sampling, density and score evaluation, and inspection independently of the pretrained model. We further introduce program-space fine-tuning, which refines differentiable program parameters by maximizing the likelihood of empirical data while keeping model parameters intact. Experiments show that PRODiGI-FT achieves better density and score metrics than existing pretrained models, while offering multi-fold speedups over i
    
[^210]: 隐藏而非删除：网络如何抑制纠缠特征

    Hidden not Deleted: How Networks Suppress Entangled Features

    [https://arxiv.org/abs/2609.27593](https://arxiv.org/abs/2609.27593)

    该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。

    

    通过线性投影实现的概念擦除方法假设特征占据可分离的子空间。我们证明该假设在密集叠加情况下会失效：当两个特征被迫形成共享同一子空间的对跖对时，最先进的线性擦除方法会同时破坏两者，而不仅仅是目标特征。通过梯度下降训练的网络则以非线性方式解决这一问题，但方式并不统一：根据初始化的不同，它们会收敛到两种不同的电路级解决方案之一，我们称之为“镜像”解决方案和“阴影”解决方案。我们将这种分叉现象映射为特征纠缠程度的函数，证明它反映的是稳定的吸引子结构而非实验设置的伪影，并通过针对性的因果干预证明，这两种解决方案都会在被擦除特征的表征中留下大量可测量的完整痕迹，仅需一个标量补丁即可恢复，而无需任何进一步的训练。这一现象类似于——（摘要原文在此处截断）

    arXiv:2609.27593v1 Announce Type: cross  Abstract: Concept erasure methods that operate via linear projection assume that features occupy separable subspaces. We show this assumption fails under dense superposition: when two features are forced into an antipodal pair sharing a single subspace, state-of-the-art linear erasure destroys both, not just the target. Networks trained with gradient descent instead solve this problem non-linearly, but not uniformly: they converge to one of two distinct circuit-level solutions depending on initialization, which we call mirror and shadow solutions. We map this bifurcation as a function of feature entanglement, show it reflects a stable attractor structure rather than an artifact of our setup, and use targeted causal interventions to demonstrate that both solutions leave a substantial, measurable trace of the erased feature's representation intact, recoverable through a single scalar patch rather than requiring any further training. This mirrors a
    
[^211]: CIDERS：通过加速个性化双层优化实现云边LLM协同学习

    CIDERS: Cloud-Edge LLM Collaborative Learning via Accelerating Personalized Bilevel Optimization

    [https://arxiv.org/abs/2609.15664](https://arxiv.org/abs/2609.15664)

    本文首次提出个性化双层优化框架将云边LLM协作形式化为双重结构，并通过CIDERS求解器将模型分解为可学习骨干和信使组件，加速实现云端知识迁移与边缘个性化适配的协调演进。

    

    随着物理世界智能的快速发展，云边协同大语言模型（LLMs）已成为LLM实际部署的一条有前景的路线。然而，现有的云边范式难以平衡全局共识与本地个性化，无法满足云端统一知识基础和边缘领域特定适应的双重需求。为解决这一问题，我们首次引入了一个个性化双层优化框架，将云边LLM协作形式化为双重结构：上层优化边缘侧个性化，下层管理云端知识迁移，从而实现云边协调演进。随后我们提出了CIDERS，一个高效的求解器，它将模型分解为可学习的骨干网络和一个信使组件。当云端向可学习骨干执行知识迁移时，关键在于嵌入全局共享……（摘要在此处截断）

    arXiv:2609.15664v1 Announce Type: cross  Abstract: Amid the rapid advancement of physical-world intelligence, cloud-edge collaborative large language models (LLMs) have emerged as a promising roadmap for practical LLM deployment. However, existing cloud-edge paradigms struggle to balance global consensus with local personalization, which fails to satisfy the need for a unified knowledge foundation on the cloud and domain-specific adaptation at the edge. To address this, we introduce, for the first time, a personalized bilevel optimization framework that formalizes cloud-edge LLM collaboration as a dual structure: the upper level optimizes edge-side personalization, while the lower level governs cloud-side knowledge transfer, reaching cloud-edge evolving in coordination. We then propose CIDERS, an efficient solver that decomposes the model into a learnable backbone and a messenger. While the cloud performs knowledge transfer to the learnable backbone, the key lies in embedding global tr
    
[^212]: 通过人机交互实现高效的测试时自适应

    Efficient Test-Time Adaptation through Human-AI Interaction

    [https://arxiv.org/abs/2609.04141](https://arxiv.org/abs/2609.04141)

    提出TAHI框架，利用跨会话的人机交互数据并结合不断演化的评分准则模块，在测试时对智能体的上下文和权重进行自适应，从而弥合AI生成成果与个人专业水准之间的差距。

    

    AI智能体在人口规模的数据上进行训练，以编码涵盖众多从业者能力的广泛能力。然而，它们生成的成果很少能达到专业人士愿意以其声誉担保的个人标准。在成功标准异质且记录不充分的现实开放任务中，个人专长恰恰体现在对平均水平的高度提升与偏离之中。在实践中，迭代式的人机交互会浮现出用户无法事先完全明确、却在多个任务中反复应用的评估标准。我们认为，这种跨会话的交互数据是弥合智能体与个人专长之间差距的丰富而未被充分利用的信号。在这项工作中，我们提出通过人机交互进行测试时自适应（TAHI），该方法将这些信号整合到智能体的上下文和权重中，并通过一个不断演化的评分准则（rubric）模块来逐步明确每位用户的训练与评估标准。我们将智能体适配到30位个体……

    arXiv:2609.04141v1 Announce Type: new  Abstract: AI agents are trained on population-scale data to encode broad capabilities spanning those of many practitioners. Yet the artifacts they produce rarely meet the personal bar professionals need to stake their reputation on. On realistic, open-ended tasks where success criteria are heterogeneous and insufficiently documented, individual expertise lives precisely in the elevation and departure from the average. In practice, iterative human-agent interaction surfaces criteria that users cannot fully specify up front, yet apply repeatedly across tasks. We argue this cross-session interaction data is a rich, underused signal for closing the gap to individual expertise. In this work, we propose test-time adaptation through human-agent interaction (TAHI), which integrates these signals into agent context and weights, and crystallizes each user's training and evaluation criteria via an evolving rubric module. We adapt agents to 30 individuals in 
    
[^213]: AI代理将人类挤出决策圈

    AI Agents Push Humans Out of the Loop

    [https://arxiv.org/abs/2608.23642](https://arxiv.org/abs/2608.23642)

    本文指出当前AI代理发展方式削弱而非支持人类监督能力，强调应将人类监督需求置于与AI能力同等优先级，以保障有效人类参与。

    

    arXiv:2608.23642v1 公告类型：新 摘要：随着AI代理被赋予越来越多的自主权，它们带来了显著风险。一个常见的提议解决方案是人类监督和保持“人类参与”，但这并非简单方案：不仅当前的AI代理设计方法阻碍了有效的人类监督，而且长时间使用AI系统本身也会削弱进行这种监督所需的认知能力。本立场论文认为，当前AI代理系统的开发与部署方法并不支持有效的人类监督——它们反而促使其退化。为解决此问题，AI代理发展的首要任务应是支持有效人类监督的情境目标和认知需求，将监督者的人类需求置于与AI代理能力同等重要的地位。为实践这一理念，我们将自动化与人机交互研究联系到AI代理流程，概述了去...

    arXiv:2608.23642v1 Announce Type: new  Abstract: AI agents pose significant risks as they are granted increasing autonomy. A commonly proposed solution is human oversight and keeping a ''human in the loop'', but this is not a simple solution: Not only do current approaches to AI agent design impede effective human oversight, but the cognitive capacities required for it are also themselves degraded by extended use of AI systems. This position paper argues that current approaches to the development and deployment of AI agent systems do not support effective human oversight -- they contribute to its degradation. To address this, a top priority in the advancement of AI agents should be supporting the situated goals and cognitive requirements of effective human oversight, treating the human needs of overseers at the same level of importance as AI agent capability. To put this idea into practice, we connect work on automation and human-computer interaction to AI agent processes, outlining de
    
[^214]: 从监管到实施：对行业中大语言模型辅助合规性的批判性评估

    From Regulation to Implementation: A Critical Evaluation of LLM-Assisted Regulatory Compliance in Industry

    [https://arxiv.org/abs/2608.21317](https://arxiv.org/abs/2608.21317)

    本文批判性评估了大语言模型在工业合规文档生成中的应用，指出其面临数据异构、格式标准化及跨学科知识整合等挑战，并提出改进路径。

    

    欧盟（EU）已成为可持续发展和隐私法规制定领域的领先监管机构。虽然新的法规要求各不相同，但许多都包含确保合规性的文档工件。值得注意的是，《可持续产品生态设计法规》（ESPR）引入了数字产品护照（DPPs）以实现生命周期透明度，而《通用数据保护条例》（GDPR）则强制要求进行数据保护影响评估（DPIAs）以减轻隐私风险。然而，创建这些合规工件具有挑战性。工业数据通常以异构格式存在，分散在公司及供应商系统中，这是DPPs所需的数据，且难以提取为符合要求的DPP格式。此外，DPIA文档需要跨学科专业知识，且没有标准化格式，使得新系统的开发变得困难。为应对合规性的特殊复杂性...

    arXiv:2608.21317v1 Announce Type: new  Abstract: The European Union (EU) has emerged as a leading regulatory body in the development of sustainability and privacy regulations. While new regulation requirements vary, many include a documentation artifact to ensure compliance. Notably, the Ecodesign for Sustainable Products Regulation (ESPR) introduces Digital Product Passports (DPPs) for life cycle transparency, while the General Data Protection Regulation (GDPR) mandates Data Protection Impact Assessments (DPIAs) to mitigate privacy risks. Creating these compliance artifacts, however, is challenging. Industrial data, which often exists in heterogeneous formats and is scattered across company and supplier systems, is required for DPPs and can be difficult to extract into compliant DPP formatting. Furthermore, DPIA documents require interdisciplinary expertise and follow no standardized format, making development difficult for novel systems. To address the particular complexity of compli
    
[^215]: 一个崎岖的前沿：评估代码代理对语义保持变换的鲁棒性

    A Jagged Frontier: Evaluating Robustness of Code Agents to Semantics-Preserving Transformations

    [https://arxiv.org/abs/2608.18389](https://arxiv.org/abs/2608.18389)

    本文通过引入随机语义保持变换采样器，系统评估了代码代理在代码库语义等价改写下的鲁棒性，发现其修复性能会显著下降，揭示了现有代理的脆弱性。

    

    arXiv:2608.18389v1 公告类型：新 摘要：AI代码代理正日益被部署来解决真实软件问题，然而它们在表面代码变化下的可靠性仍鲜为人知。我们评估了当周围代码库被重写为语义等价形式时，修复仓库级问题的代码代理是否仍保持可靠。我们引入了一个随机变体采样器，应用常见的语义保持变换（SPTs）——涵盖控制流重写、死代码注入和标识符重命名——以生成扰动变体。我们评估了两种代理框架（mini-SWE agent和OpenCode），每种框架由四个前沿模型之一（Claude Opus 4.5、Kimi K2.5、MiniMax M2.5和Qwen 3.6-27B）支持，在来自SWE-bench Verified和SWE-bench Pro的实例上进行测试。对于每个实例，代理在未扰动和扰动变体上多次运行，产生配对解决率估计，以隔离扰动效应与内在变化。

    arXiv:2608.18389v1 Announce Type: new  Abstract: AI code agents are increasingly deployed to resolve real software issues, yet their reliability under superficial code variations remains poorly understood. We evaluate whether coding agents that repair repository-level issues remain reliable when the surrounding codebase is rewritten into a semantically equivalent form. We introduce a random variant sampler that applies common semantics-preserving transformations (SPTs) - spanning control-flow rewrites, dead-code injection, and identifier renaming - to produce perturbed variants. We evaluate two agentic scaffolds (mini-SWE agent and OpenCode) each backed by one of four frontier models (Claude Opus 4.5, Kimi K2.5, MiniMax M2.5, and Qwen 3.6-27B) across instances drawn from SWE-bench Verified and SWE-bench Pro. For each instance, the agent is run multiple times on the unperturbed and perturbed variants, yielding paired resolve-rate estimates that isolate the perturbation effect from intri
    
[^216]: ReForge：通过验证的大型语言模型编辑使ABR算法永不停滞

    ReForge: Keeping ABR Algorithms Never Finished with Verified Large Language Model Edits

    [https://arxiv.org/abs/2608.15138](https://arxiv.org/abs/2608.15138)

    ReForge利用大型语言模型循环持续重新设计ABR算法，通过验证每个小编辑对历史场景的无害性，实现算法对不断变化网络场景的分钟级适应。

    

    arXiv:2608.15138v1 公告类型：新 摘要：为一个网络场景设计自适应比特率（ABR）算法需要工程师数月时间，而大型语言模型现在能在数小时内完成这项工作，其性能可媲美或超越手工设计。然而，无论哪种方式，设计仅适应于其诞生时可见的世界，而无法应对之后出现的新世界。我们提出一个问题：ABR算法能否跟上世界的步伐，在每次新场景出现时于几分钟内重新设计，并确保每次更改对已服务的所有场景均证明无害。在本工作中，我们提出ReForge，一种持续启发式学习框架，用于适应不断变化的场景。ReForge将大型语言模型（LLM）纳入循环中运行该流程。每轮中，LLM读取当前设计的不足之处并提出一个小编辑，然后通过对所有已服务网络的回放来决定是否采纳。具体而言，其编辑对象是一页模糊规则，该规则将每个决策路由至一组冻结的预训练策略池中。LLM编写...

    arXiv:2608.15138v1 Announce Type: new  Abstract: Designing an ABR algorithm for one network scenario takes an engineer months, and large language models now do this work in hours, matching or beating hand-built designs. But either way, the design fits only the world visible at its birth, and fails on the world that arrives after. We ask whether an ABR algorithm can keep pace with the world, redesigned in minutes as each scenario arrives, with every change proven harmless to every scenario already served. In this work, we propose ReForge, a continual heuristic learning framework that adapts to continuously changing scenarios. ReForge runs that routine with a large language model (LLM) in the loop. Each round the LLM reads where the current design falls short and proposes one small edit, and a replay over every network served so far decides. Specifically, what it edits is a single page of fuzzy rules that routes every decision to one of a frozen pool of pre-trained policies. The LLM writ
    
[^217]: 基于无回归布局感知匹配的无幻觉GUI定位

    Hallucination-Free GUI Grounding via Regression-Free Layout-Aware Matching

    [https://arxiv.org/abs/2608.09654](https://arxiv.org/abs/2608.09654)

    该论文提出了一种无回归框架，利用冻结的MLLM将抽象指令解析为富含布局线索的结构化视觉描述，再通过布局感知匹配实现无坐标回归、无幻觉的精确GUI元素定位。

    

    GUI智能体正在从依赖元数据的大型语言模型转向直接在屏幕截图上操作的纯视觉多模态大型语言模型（MLLM）。核心任务GUI定位（GUI grounding）需要将抽象的用户指令转化为精确的元素坐标。这项任务面临一个持续存在的双重障碍：传统定位模型缺乏解读抽象指令所需的语义丰富性，而端到端MLLM则因细粒度感知能力不足而遭受坐标幻觉问题。我们提出了一个无回归框架，其中冻结的MLLM负责指令解析，专用定位模型负责精确定位，而无需学习任何坐标回归。冻结的MLLM首先将抽象指令细化为富含布局线索的结构化视觉描述，这些描述随后被输入到一个新颖的布局感知GUI定位模型中，该模型执行无回归定位。

    arXiv:2608.09654v2 Announce Type: replace  Abstract: GUI agents are shifting from metadata-dependent large language models to purely visual multimodal large language models (MLLMs) that operate directly on screenshots. The core task, GUI grounding, requires translating abstract user instructions into precise element coordinates. This task faces a persistent dual obstacle: conventional grounding models lack the semantic richness to interpret abstract instructions, while end-to-end MLLMs suffer from coordinate hallucinations caused by deficient fine-grained perception. We propose a regression-free framework where a frozen MLLM performs instruction parsing and a dedicated grounding model handles precise localization without learning any coordinate regression. A frozen MLLM first elaborates the abstract instruction into a structured visual description rich in layout cues. These descriptions are then fed to a novel Layout-Aware GUI Grounding Model, which performs regression-free localizatio
    
[^218]: 同策略交互何时有益？基于价值的模仿学习中的表征权衡

    When Does On-Policy Interaction Help? Representational Tradeoffs in Value-Based Imitation Learning

    [https://arxiv.org/abs/2607.29617](https://arxiv.org/abs/2607.29617)

    该论文发现，在基于价值的模仿学习中，与专家进行同策略交互能放宽对学习者的表征要求——只需模型能实现专家的价值函数，而无需实现要求更严格的专家策略本身。

    

    模仿学习（IL）——即训练智能体从演示中复制专家行为——是机器人技术到语言模型训练等诸多应用的基础。众所周知，行为克隆（BC）等标准方法存在误差累积和性能停滞的问题，尤其当学习者无法完美表征专家策略时（例如在蒸馏场景中这是典型情况）。两种干预措施在经验上被广泛认为可以提升性能：沿学习者自身轨迹交互式地查询专家，以及在生成策略的过程中使用价值函数估计。我们研究了这些改进的本质及其可能令人惊讶的相互作用。我们的主要发现是，与专家的交互放宽了对学习者的表征要求：只需要一个能够实现专家价值函数的模型，从而绕过了（通常更严格的）实现专家策略的要求。（注：摘要原文在此处被截断）

    arXiv:2607.29617v2 Announce Type: replace-cross  Abstract: Imitation learning (IL)---training an agent to replicate expert behavior from demonstrations---underpins applications from robotics to language model training. Standard approaches such as Behavior Cloning (BC) are known to suffer from compounding errors and performance plateaus, particularly when the learner cannot perfectly represent the expert's policy (as is typical, e.g., in distillation). Two interventions are widely understood empirically to improve performance: querying the expert interactively along the learner's own trajectories, and using value function estimation en route to generating a policy.   We investigate the nature of these improvements and their potentially surprising interplay. Our main finding is that expert interaction relaxes the representational demands on the learner: one only needs a model capable of realizing the expert's value function, bypassing the (often stricter) requirement of realizing the exp
    
[^219]: 礼貌性预判：改进持久共享环境中的长期任务规划

    Courteous Anticipation: Improving Long-Lived Task Planning in Persistent Shared Environments

    [https://arxiv.org/abs/2607.20289](https://arxiv.org/abs/2607.20289)

    提出“礼貌性预判规划”，让机器人在持久共享环境中求解当前任务时，通过独立的每机器人学习估计器，联合最小化即时成本与所有机器人的聚合预期未来成本，从而避免副作用在长任务序列中累积。

    

    我们考虑了一种任务规划场景：多个机器人共享一个持久环境，任务从一组预留的任务序列中被逐个分配。标准的任务规划器缺乏对未来任务的预见，也不顾及他人的约束，孤立地求解每个任务，其留下的终止状态会增加所有机器人未来任务的成本；这些副作用会在冗长的任务序列中不断累积。为了降低整个序列的成本，机器人必须预判其当前动作会如何影响共享环境中所有机器人未来任务的表现。为此，我们提出了礼貌性预判规划：基于模型的规划器提出候选方案，并选择能够同时最小化即时成本和所有机器人聚合预期未来成本的方案，其中未来成本通过各机器人独立的学习估计器进行估算。这种因式分解的表述避免了组合式的联合推演，并支持模块化部署：添加一个机器人只需……

    arXiv:2607.20289v2 Announce Type: replace-cross  Abstract: We consider a task planning scenario in which robots sharing a persistent environment are assigned tasks one at a time from a held-out sequence. Standard task planners, lacking foresight of future tasks and inconsiderate of others' constraints, solve each task in isolation, leaving terminal states that increase future cost for all; side effects that compound over lengthy task sequences. To reduce cost over the sequence, a robot must anticipate how its actions now may impact performance on future tasks for all robots sharing the environment. Therefore, we present courteous anticipatory planning, wherein a model-based planner proposes candidate plans and selects the one that jointly minimizes immediate cost and aggregated expected future cost across all robots, estimated via independent per-robot learned estimators. This factored formulation avoids combinatorial joint rollouts and supports modular deployment: adding a robot requi
    
[^220]: CPM-MultiAgent：一个基于CPM认知评价理论的框架，用于在医疗对话中模拟患者情绪的动态演变

    CPM-MultiAgent: A CPM-Grounded Appraisal Framework for Simulating Dynamic Patient Emotional Evolution in Healthcare Dialogue

    [https://arxiv.org/abs/2607.07824](https://arxiv.org/abs/2607.07824)

    该论文提出基于心理学组件过程模型（CPM）的多智能体框架CPM-MultiAgent，能够模拟患者情绪随医护人员沟通而动态演变的过程，为医疗沟通训练中的学习者提供真实的情绪交互反馈。

    

    大语言模型（LLMs）能够实现用于医疗沟通训练的患者模拟，学习者可以在其中练习对临床信息、患者担忧和情绪反应作出回应。然而，现有的患者模拟和情感对话系统通常将患者情绪表示为静态属性，或者仅专注于情绪识别和共情回应的生成，对于建模模拟患者的情绪状态如何随医疗专业人员的沟通而演变，所提供的支持十分有限。患者情绪的变化构成了重要的交互反馈，学习者必须在对话的多个轮次中识别并回应这些反馈。我们借鉴了组件过程模型（CPM）——这是一种将情绪解释为由事件认知评价所塑造的动态过程的心理学理论——并据此提出了CPM-MultiAgent，一个基于CPM的多智能体框架，用于医疗沟通训练中的动态患者情绪模拟。

    arXiv:2607.07824v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) enable patient simulation for healthcare communication training, where learners practice responding to clinical information, concerns, and emotional reactions. However, existing patient simulation and affective dialogue systems often represent patient emotions as static attributes or focus on recognizing emotions and generating empathetic responses. They provide limited support for modeling how a simulated patient's emotional state evolves in response to a healthcare professional's communication. Changes in patient emotion constitute interactional feedback that learners must recognize and respond to across dialogue turns. We draw on the Component Process Model (CPM), a psychological theory that explains emotion as a dynamic process shaped by event appraisal. We propose CPM-MultiAgent, a CPM-grounded multi-agent framework for dynamic patient emotion simulation in healthcare communication training. Th
    
[^221]: REFORGE：一种用于评测大语言模型在反编译二进制函数命名中逆向工程能力的方法

    REFORGE: A Method for Benchmarking LLMs' Reverse Engineering Capabilities in Decompiled Binary Function Naming

    [https://arxiv.org/abs/2607.07738](https://arxiv.org/abs/2607.07738)

    本文提出REFORGE——一个带溯源追踪的评测流水线，通过八门置信度漏斗将编译器优化下的二进制-源码对齐不确定性量化分级，从而解决了LLM逆向工程能力公平评估中函数级真值可靠性不足这一核心障碍。

    

    大语言模型（LLM）正越来越多地被应用于逆向工程任务，最近的威胁情报报告显示它们已在实际的进攻性安全工作流程中运行。然而，关于其能力的宣称已超出了我们衡量该能力的水准。现有的LLM辅助二进制分析基准将函数级真值的构建视为已解决的前处理步骤，并在报告准确率时未披露有多少函数可被可靠地评估。我们认为，公平评估的主要障碍并非模型能力，而是编译器优化下二进制代码与源码对齐的可靠性。本文提出了Reforge，这是一个带有溯源追踪的流水线，它通过编译、DWARF与语法提取、对齐以及反编译等步骤，从C源代码构建函数级真值，并将对齐不确定性操作化为一个八门置信度漏斗，分为三个层级……

    arXiv:2607.07738v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly applied to reverse-engineering tasks, and recent threat-intelligence reporting shows them operating inside live offensive-security workflows. Claims about their capability, however, outpace our ability to measure it. Existing benchmarks for LLM-assisted binary analysis treat the construction of function-level ground truth as a solved pre-processing step and report accuracy without disclosing how many functions were reliably evaluable. We argue that the principal obstacle to fair evaluation is not model capability but the reliability of binary-to-source alignment under compiler optimization. This paper presents Reforge, a provenance-tracked pipeline that constructs function-level ground truth from C source through compilation, DWARF and syntactic extraction, alignment, and decompilation, and that operationalizes alignment uncertainty as an eight-gate confidence funnel with three-tier
    
[^222]: 输入通路塑造微型Transformer的小样本绑定而非零样本绑定：一项完全可枚举的研究

    Input Pathways Shape Few-Shot, Not Zero-Shot, Binding in Tiny Transformers: A Fully-Enumerable Study

    [https://arxiv.org/abs/2607.04926](https://arxiv.org/abs/2607.04926)

    该研究通过对微型Transformer进行完全可枚举实验发现，无论信息以符号token、神谕代码还是纠缠感知向量的形式输入，模型都无法实现零样本组合绑定（其失败源于归纳偏置而非信息缺失），但在小样本设置下，绑定学习的样本效率由输入代码的可读性决定。

    

    信息以何种方式到达Transformer——是以符号token的形式、清晰的逐因子“神谕”代码，还是纠缠的感知向量——会如何影响它能否对这些信息进行组合式绑定？我们在有限因子世界上研究了约6-10K参数的微型Transformer，并对整个输入空间进行穷举枚举，因此每次测量都覆盖全部输入空间，且各信息通路在信息量上是精确匹配的（贝叶斯准确上限为1.0）。研究得出四个发现：(1) 终点不变性：在留出的绑定查询上，没有任何一条信息通路能够达到收敛的零样本组合能力——每条通路都终止于随机水平或以下，因此这种失败反映的是在查表即可充分完成目标情况下的归纳偏置问题，而非信息缺失。即使将状态放在其自身的序列位置、而不是加到每个token上，该结论依然成立。(2) 可读性决定小样本绑定：当留出类型的一小部分数据泄漏进训练集时，样本效率与代码可读性成正比。可读性最差的……（原文摘要在此处截断）

    arXiv:2607.04926v2 Announce Type: replace-cross  Abstract: How does the way information reaches a transformer -- as symbolic tokens, a clean per-factor "oracle" code, or an entangled perceptual vector -- shape whether it binds that information compositionally? We study ~6-10K-parameter transformers on finite factored worlds enumerated exhaustively, so every measurement covers the whole input space and the informative routes are information-matched (exact Bayes ceiling 1.0). Four findings. (1) Endpoint invariance: on held-out binding queries no informative route reaches converged zero-shot composition -- each ends at or below chance, so the failure reflects inductive bias under a lookup-sufficient objective, not missing information. It also holds with the state in its own sequence position rather than added to every token. (2) Readability governs few-shot binding: when a small fraction of the held-out type is leaked into training, sample efficiency tracks code readability. The least rea
    
[^223]: URSA：面向实用性逆合成评估的化学感知基准

    URSA: Chemistry-Aware Benchmark for Utilitarian Retrosynthesis Assessment

    [https://arxiv.org/abs/2607.04688](https://arxiv.org/abs/2607.04688)

    URSA提出了一个化学感知的逆合成评估基准框架，首次从形式指标（如收敛到商业可得起始原料）和模拟专家化学家判断的化学合理性两个维度，对端到端逆合成模型与大语言模型进行客观、可解释的比较评估。

    

    合成规划旨在为目标分子寻找可行的反应路径，是药物发现中最重要且最具挑战性的任务之一。近期的进展催生了专用的深度学习逆合成系统和通用的大型语言模型，但由于缺乏灵活且具备化学可解释性的基准测试协议，对它们进行客观比较仍然十分困难。在本研究中，我们提出了URSA（实用性逆合成评估）评估框架，该框架不仅能够从形式角度（例如是否收敛到可商业购买的起始原料）对合成路线进行基准评估，还能从化学合理性角度进行评估，模拟专家化学家评估反应与路线的方式。该研究对传统端到端逆合成解决方案和大语言模型在合成规划任务上进行了全面评估（原文摘要在此处截断）。

    arXiv:2607.04688v2 Announce Type: replace-cross  Abstract: Synthesis planning aiming to find pathways of reactions for a target molecule is one of the most important and challenging tasks in drug discovery. Recent progress has produced both specialized deep-learning retrosynthesis systems and general-purpose large language models, but objective comparison remains difficult due to the lack of flexible, chemically interpretable benchmarking protocols. In the current study, we are introducing the URSA (Utilitarian RetroSynthesis Assessment) evaluation framework that provides the opportunity to benchmark the synthetic routes not only from a formal perspective, such as convergence to commercially available starting materials, but also from a chemical plausibility perspective, mimicking the way expert chemists evaluate the reactions and routes. The study covers a comprehensive evaluation of both conventional end-to-end retrosynthesis solutions and LLMs for the synthesis planning task on a se
    
[^224]: 当简单更优时：评估中世纪拉丁手稿的翻译流水线

    When Simpler Is Better: Evaluating Translation Pipelines for Medieval Latin Manuscripts

    [https://arxiv.org/abs/2607.03836](https://arxiv.org/abs/2607.03836)

    在中世纪拉丁手稿翻译任务中，参数量少几个数量级的领域专用OCR模型反而优于通用视觉语言模型（字符错误率最多降低4.3倍），并据此构建了包含1,383条对齐图像、转录与译文的新型数据集IPC。

    

    尽管机器翻译取得了显著进展，视觉语言模型（VLM）在历史手稿上的表现仍然不佳。该领域考验着自然语言处理（NLP）的核心能力：低资源转写、古旧词汇以及嘈杂的输入信号。我们提出了一个系统性框架，用于评估从中世纪拉丁手稿图像到译文的完整流水线。在这一场景中，抄写员的速记符号、连字以及羊皮纸的破损暴露出在干净文本基准测试中难以察觉的失败模式。在CATMuS Latin数据集上的基准测试揭示了一种专业化差距：领域专用的光学字符识别（OCR）模型与通用VLM相比，可将字符错误率降低至多4.3倍，尽管其参数量少了几个数量级。我们提出了Interpres-Parallel-Corpus（IPC），一个包含1,383条对齐的手稿图像行、转录文本及实验（结果）的新型数据集。

    arXiv:2607.03836v2 Announce Type: replace-cross  Abstract: Despite remarkable progress in machine translation, Vision Language Models (VLMs) struggle on historical manuscripts, a domain that stresses core Natural Language Processing (NLP) capabilities: low-resource transliteration, archaic vocabulary, and noisy input signals. We present a systematic framework for evaluating the full image-to-translation pipeline on medieval Latin manuscripts, a setting in which scribal shorthand, ligatures, and parchment degradation expose failure modes that are invisible in clean-text benchmarks. Benchmarking on the CATMuS Latin dataset reveals a specialization gap: domain-specific Optical Character Recognition (OCR) models reduce character error rate by up to 4.3$\times$ compared to general-purpose VLMs, despite operating at orders of magnitude fewer parameters. We introduce the Interpres-Parallel-Corpus (IPC), a novel dataset comprising 1,383 aligned manuscript image lines, transcriptions, and exper
    
[^225]: QVal：评估长程LLM智能体的密集监督信号

    QVal: Evaluating Dense Supervision Signals for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2606.32034](https://arxiv.org/abs/2606.32034)

    该论文提出QVal——一个无需训练的测试平台，通过衡量密集监督信号在每个状态下对候选动作的排序与参考策略回报排序的一致性（Q-对齐性）来直接评估信号质量，并在四个环境中对七个方法族的21种方法进行基准测试，发现简单提示基线平均Q-对齐性最高且性能按方法族高度聚集。

    

    LLM智能体越来越多地在长时程任务中执行行动，而仅基于结果的奖励几乎无法说明哪些中间动作是好的。密集监督方法通过为中间步骤打分来解决这一问题，但这类方法通常通过完整训练流程的下游性能来评估。这种方式成本高昂，将信号质量与其他训练选择混为一谈，也意味着这些方法很少在共同基准上进行比较。我们提出了QVal，一个无需训练的测试平台，用于衡量密集信号的$Q$-对齐性：即该信号在每个状态下对候选动作的排序是否与其在参考策略下的回报排序一致。我们在四个环境中，使用六个开源权重骨干模型，对来自七个方法族的21种方法进行了基准测试。结果表明，简单的提示基线平均而言$Q$-对齐性最高，且性能按方法族高度聚集。代码生成的信号仅在小状态和动作空间中表现良好；基于嵌入的信号则……

    arXiv:2606.32034v2 Announce Type: replace-cross  Abstract: LLM agents increasingly act over long horizons, where outcome-only rewards say little about which intermediate actions were good. Dense supervision methods address this by scoring intermediate steps, but they are usually evaluated through the downstream performance of a full training pipeline. This is expensive, confounds signal quality with other training choices, and means that methods are rarely compared on common ground. We introduce QVal, a training-free testbed that measures the $Q$-alignment of a dense signal: whether it orders the candidate actions in each state the same way as their returns under a reference policy do. We benchmark 21 methods from seven families with six open-weight backbones in four environments. Simple prompting baselines are the most $Q$-aligned on average, and performance clusters strongly by family. Code-generated signals do well only in small state and action spaces; embedding-based signals are w
    
[^226]: CoffeeBench：在异构多智能体经济中对长程LLM智能体进行基准测试

    CoffeeBench: Benchmarking Long-Horizon LLM Agents in Heterogeneous Multi-Agent Economies

    [https://arxiv.org/abs/2606.16613](https://arxiv.org/abs/2606.16613)

    CoffeeBench通过模拟咖啡供应链中的多智能体经济，评估LLM智能体的长程商业决策能力，发现Claude Sonnet 5.5凭借更高的售价和更低的采购成本获得最高净收入。

    

    LLM智能体处理复杂长程任务的能力日益增强，最终可能承担传统上由人类执行的业务管理工作。要安全地部署它们，需要在反映真实经济的环境中监测其能力和行为。我们提出了CoffeeBench，这是一个基于咖啡供应链紧凑模拟的基准测试，具有真实的商业机制。六个自主LLM编码智能体分别运营其中的农户、烘焙商和零售商，并在90个模拟天中相互谈判和交易，以最大化各自的净收入。我们评估了八个最新的专有和开源权重模型，让每个模型运营一家烘焙商，同时由固定的背景模型运营其他五家公司，并以该烘焙商的净收入作为评分标准。Claude Sonnet 5.5获得了最高的净收入，GPT-6 Astra紧随其后。领先者以更高的价格出售产品，而表现最佳的模型还能以远更低的价格采购原材料。

    arXiv:2606.16613v2 Announce Type: replace  Abstract: LLM agents are increasingly capable of handling complex, long-horizon tasks and may eventually take on business management, a role traditionally performed by humans. Deploying them safely requires monitoring their capabilities and behavior in environments that reflect real economies. We introduce CoffeeBench, a benchmark built on a compact simulation of a coffee supply chain with realistic business mechanics. Six autonomous LLM coding agents run its farmers, roasters, and retailers, and negotiate and trade with one another to maximize their own net income over 90 simulated days. We evaluate eight recent proprietary and open-weight models by letting each run one roaster while a fixed background model runs the other five firms, and scoring that roaster's net income. Claude Sonnet 5.5 earns the highest net income, closely followed by GPT-6 Astra. The leaders sell at higher prices, and the best model also buys its inputs far more cheaply
    
[^227]: Larch：面向语义谓词的学习型查询优化

    Larch: Learned Query Optimization for Semantic Predicates

    [https://arxiv.org/abs/2606.07923](https://arxiv.org/abs/2606.07923)

    Larch利用非结构化数据嵌入所承载的语义信息，通过学习型优化技术高效优化AI SQL查询中语义过滤器的执行。

    

    随着大语言模型（LLM）的出现，许多数据库系统引入了语义算子，使分析查询能够应用于非结构化数据（如文本、图像、视频）。语义算子通常伴随较高的推理成本和延迟，使得语义（AI）SQL查询难以应用于大规模数据集。与此同时，其语义特性导致数据库引擎将其视为黑盒，使AISQL查询难以优化。本文提出Larch，一个用于优化AI SQL查询中语义过滤器执行过程的框架。Larch的灵感来自两个关键观察：i) 语义算子的高延迟为计算密集型的运行时优化技术留下了巨大的空间；ii) 非结构化数据通常伴随着以嵌入（embedding）形式存在的语义信息，从而能够在AI_FILTER提示与数据值之间进行高效的语义比较。

    arXiv:2606.07923v2 Announce Type: replace-cross  Abstract: With the advent of Large Language Models (LLMs), many database systems introduced semantic operators that enabled analytical queries over unstructured data (e.g. text, images, videos). Semantic operators typically incur high inference costs and latencies making semantic (AI) SQL queries challenging to apply on large scale datasets. At the same time, their semantic nature leads database engines to treat them as black boxes, making AISQL queries difficult to optimize. In this paper, we introduce Larch, a framework for optimizing the execution of semantic filters in AI SQL queries. Larch was inspired by two key observations: i) the high latency of semantic operators leaves significant room for computationally-heavy runtime optimization techniques, ii) unstructured data are typically accompanied by semantic information in the form of embeddings allowing for efficient semantic comparisons between AI_FILTER prompts and data values. B
    
[^228]: PhysDox：基于生理传感协议物理可行性审计的大语言模型基准测试

    PhysDox: Benchmarking LLMs on Physical Feasibility Auditing of Physiological Sensing Protocols

    [https://arxiv.org/abs/2606.05003](https://arxiv.org/abs/2606.05003)

    本文提出PhysDox基准用于评估大语言模型对生理传感协议物理可行性的审计能力，发现现有LLM表现不佳（最高宏F1仅53.0），且存在将程序完整性与物理有效性混淆的“脚手架偏差”，导致隐式约束的漏检率是显式违规的两倍。

    

    大语言模型（LLM）日益被用于辅助实验设计，然而表面上流畅的协议往往在物理上并不可行。我们提出了PhysDox，这是一个针对生物医学协议物理可行性审计的基准，包含一个由专家精心整理的683个样本的Gold（黄金）数据集，以及涵盖六个传感领域的5,000个样本的Silver（白银）数据集。我们将该任务形式化为两阶段评估：首先是严重性检测，将协议分类为有效、轻微或致命三类，随后对致命违规进行约束层面的诊断。对6个LLM在4种推理策略下的评估显示，第一阶段的最优宏平均F1仅为53.0。此外，即使在有理想oracle诊断的情况下，端到端评估仍会因相关的级联错误而失效。错误分析揭示了“脚手架偏差”（scaffold bias）现象，即模型将程序完整性与物理有效性混为一谈。因此，隐式约束的漏检率是显式硬件违规的2倍，并有强有力的统计证据支持。

    arXiv:2606.05003v2 Announce Type: cross  Abstract: Large language models (LLMs) increasingly assist in experimental design, yet fluent protocols often remain physically infeasible. We introduce PhysDox, a physical feasibility auditing benchmark for biomedical protocols comprising a 683-sample expert-curated Gold set and a 5,000-sample Silver set across six sensing domains. We formulate the task as a two-stage evaluation: severity detection classifying protocols as valid, minor, or fatal, followed by the constraint-level diagnosis of fatal violations. Evaluating 6 LLMs across 4 inference strategies yields a peak Stage-1 macro-F1 of only 53.0. Moreover, strong oracle diagnosis collapses during end-to-end evaluation due to correlated cascade errors. Error analysis reveals scaffold bias, where models conflate procedural completeness with physical validity. Consequently, implicit constraints exhibit a 2 times higher miss rate than explicit hardware violations, supported by strong statistica
    
[^229]: 基于PAC-Bayes风险的嵌套因果老虎机认证策略优化

    Certified Policy Optimisation for Nested Causal Bandits via PAC-Bayes Risk

    [https://arxiv.org/abs/2605.29788](https://arxiv.org/abs/2605.29788)

    本文提出嵌套上下文因果老虎机（NCCB）这一分层因果决策问题框架及嵌套因果汤普森采样（NCTS）算法，其核心贡献是一个因果PAC-Bayesian超额风险界，可仅凭历史数据、离策略且随时地对任意候选部署策略进行认证，回答“能否信任该智能体、风险多大”的部署问题。

    

    关键性序贯决策很少是单一时间尺度的：一个战略性决策会因果性地塑造后续每一个战术性选择所处的情境；标准的赌博机和强化学习理论无法捕捉这种时间尺度之间的因果耦合。我们将该问题类别形式化为嵌套上下文因果老虎机，这是一个分层结构因果模型（SCM），其中每一层的动作决定下一层的情境分布；并提出嵌套因果汤普森采样，它在每个回合抽取一个机制分解的信念，并在该信念下递归地行动。我们的主要理论结果是一个因果PAC-Bayesian超额风险界，仅凭历史数据即可对任何候选部署策略进行认证，支持离策略（off-policy）和随时（anytime）使用，从而回答部署问题：我们能否在此处信任这个智能体，其风险有多大？在分层SCM上的实验表明，与在同一函数类别上匹配的RFF-GP联合回归相比，该因子化方法……（摘要原文在此处截断）

    arXiv:2605.29788v2 Announce Type: replace  Abstract: Critical sequential decisions are rarely single-timescale: a strategic decision causally shapes the context in which every subsequent tactical choice is made; standard bandit and reinforcement-learning theory does not capture this causal coupling between timescales. We formalise the problem class as Nested Contextual Causal Bandits (NCCBs), a hierarchical SCM where each level's action sets the next level's context distribution, and propose Nested Causal Thompson Sampling (NCTS), which draws one mechanism-factorised belief per episode and acts recursively under it. Our main theoretical result is a causal PAC-Bayesian excess-risk bound that certifies any candidate deployment policy from historic data alone, off-policy and anytime, answering the deployment question: can we trust this agent here, and at what risk? Experiments on a hierarchical SCM show that, against a matched RFF-GP joint regression on the same function class, the factor
    
[^230]: DeltaBox：面向有状态AI智能体的毫秒级沙盒检查点与回滚

    DeltaBox: Millisecond-Scale Sandbox Checkpoint and Rollback for Stateful AI Agents

    [https://arxiv.org/abs/2605.22781](https://arxiv.org/abs/2605.22781)

    提出DeltaBox系统，基于“相邻沙盒状态间差异微小”这一关键洞察，设计操作系统级抽象DeltaState，通过仅复制增量变更而非全量状态，实现了毫秒级的沙盒检查点与回滚，支撑AI智能体的高频状态探索。

    

    新兴的基于大语言模型（LLM）驱动的AI智能体工作负载，如测试时树搜索和强化学习（RL），需要进行高频的状态探索，这依赖于对完整沙盒状态（包括文件和进程状态）的快速检查点/恢复（C/R）。现有机制会完整复制整个状态，导致每次C/R产生数百毫秒到数秒的延迟，严重制约了深度搜索和大规模扇出操作。我们观察到，检查点之间相邻的沙盒状态仅有微小差异。因此，沙盒不应进行全量复制，而应只复制连续检查点之间的变更（关键洞察），然而操作系统的支持缺失使这一想法难以实现。我们提出了一种新的操作系统级抽象DeltaState，通过协同设计的操作系统机制为智能体提供基于变更的C/R。首先，DeltaFS通过冻结当前可写层并插入新的可写层来对文件状态进行检查点操作，从而将文件更新……

    arXiv:2605.22781v3 Announce Type: replace-cross  Abstract: Emerging LLM-powered AI agent workloads, such as test-time tree search and reinforcement learning (RL), require high-frequency state exploration, relying on rapid checkpoint/restore (C/R) of the complete sandbox state, including files and process state. Existing mechanisms duplicate the entire state, causing hundreds of milliseconds to seconds of latency per C/R, which severely bottlenecks deep search and large-scale fan-outs. We observe that consecutive sandbox states between checkpoints differ only marginally. Therefore, instead of full duplication, a sandbox should only duplicate the changes between consecutive checkpoints (Key Insight), yet the lack of OS support makes this hard to realize. We propose a new OS-level abstraction, DeltaState, that delivers change-based C/R for agents through co-designed OS mechanisms. First, DeltaFS checkpoints file state by freezing the writable layer and inserting a new one, turning updates
    
[^231]: TransitLM：面向无地图公交路线生成的大规模数据集与基准

    TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation

    [https://arxiv.org/abs/2605.22355](https://arxiv.org/abs/2605.22355)

    该论文发布了包含四个中国城市近1300万条路线规划记录的大规模数据集TransitLM，证明大语言模型可完全从数据中学习公交路线规划，无需地图即可实现从起终点信息到路线的端到端生成。

    

    公共交通路线规划传统上依赖于结构化地图基础设施和复杂的路由引擎，目前尚无数据集支持训练模型来摆脱这一依赖。我们提出了TransitLM，这是一个包含来自四个中国城市的近1300万条公交路线规划记录的大规模数据集，覆盖120,845个车站和13,666条线路，以持续预训练语料库的形式发布，并附带三个评估任务的基准数据及互补的评估指标。实验表明，在TransitLM上训练的大语言模型能够以高准确率生成结构有效的路线，并能在没有任何显式映射的情况下，隐式地将任意GPS坐标对齐到合适的车站。这些结果表明，公共交通路线规划可以完全从数据中学习，从而实现直接根据起终点信息进行端到端、无地图的路线生成。数据集和基准已在 https://huggingface.co/datasets/GD-ML/ 公开。

    arXiv:2605.22355v2 Announce Type: replace-cross  Abstract: Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of nearly 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at https://huggingface.co/datasets/GD-ML/
    
[^232]: 用于多尺度物理仿真的小波流匹配

    Wavelet Flow Matching for Multi-Scale Physics Emulation

    [https://arxiv.org/abs/2605.16573](https://arxiv.org/abs/2605.16573)

    提出小波流匹配（WFM），直接在多尺度小波空间中进行最优传输流匹配，无需预训练自编码器，在混沌流体仿真中以更低成本实现了更优的谱相干性与细节保持能力。

    

    对由偏微分方程（PDE）支配的多尺度物理系统进行精确仿真，要求模型在长时自回归滚动预测中保持稳定，同时保留细尺度结构。确定性仿真器会产生过度平滑的预测，而生成式方法虽能更好地捕捉细节但计算成本高昂。潜空间生成模型作为一种折中方案出现，但其代价是需要额外单独预训练的自编码器。我们提出小波流匹配（Wavelet Flow Matching, WFM），这是一种生成式仿真器，通过直接在多尺度小波空间中执行最优传输流匹配，克服了当前成本与性能之间的权衡。WFM 不学习潜在压缩，而是利用 U-Net 的层次结构，联合预测一个规定的无损小波表示的传输速度。在三个具有挑战性的混沌流体动力学系统上，WFM 实现了更优的谱相干性和具有竞争力的……（摘要原文在此处截断）

    arXiv:2605.16573v2 Announce Type: replace-cross  Abstract: Accurate emulation of multi-scale physical systems governed by PDEs demands models that remain stable over long autoregressive rollouts while preserving fine-scale structures. Deterministic emulators produce overly smoothed predictions, while generative approaches better capture details but are costly. Latent-space generative models have emerged as a compromise but with the additional cost of separately pre-trained autoencoders. We propose Wavelet Flow Matching (WFM), a generative emulator that overcomes current trade-offs between cost and skill by performing optimal-transport flow matching directly in the multi-scale wavelet space. Rather than learning a latent compression, WFM leverages the hierarchical structure of a U-Net to jointly predict the transport velocities of a prescribed, lossless wavelet representation. On three challenging systems of chaotic fluid dynamics, WFM achieves superior spectral coherence and competitiv
    
[^233]: ShopGym：一个用于电子商务网络智能体真实仿真与可扩展基准测试的集成框架

    ShopGym: An Integrated Framework for Realistic Simulation and Scalable Benchmarking of E-Commerce Web Agents

    [https://arxiv.org/abs/2605.16116](https://arxiv.org/abs/2605.16116)

    ShopGym提出了一种集成框架，通过将真实在线商店转化为匿名化、可复现的沙盒仿真环境，解决了电子商务网络智能体评估中真实性与可控性、可扩展性难以兼顾的方法论瓶颈。

    

    开发和评估电子商务网络智能体需要既能保留有意义的任务结构、又能支持可控、可复现且可扩展的科学比较的环境。现有方法迫使我们做出权衡：真实在线商店提供了真实性，但具有非静态性、难以检查和不可复现的问题；而手工构建的沙盒基准虽然提供了控制能力，却仅涵盖有限的布局、商品目录、政策和交互模式。我们认为核心瓶颈在于方法论层面：该领域缺乏一种可扩展的方式来构建同时具备真实性、多样性、可控性、可检查性和可复现性的评估环境。我们提出了ShopGym，一个用于电子商务网络智能体真实仿真与可扩展基准测试的集成框架。其仿真层ShopArena通过匿名化的商店规范和分阶段的验证流程，将真实的种子商店转换为独立完整的沙盒商店（摘要在此处被截断）。

    arXiv:2605.16116v2 Announce Type: replace  Abstract: Developing and evaluating e-commerce web agents requires environments that preserve meaningful task structure while enabling controllable, reproducible, and scalable scientific comparison. Existing methods force a tradeoff: live storefronts provide realism but are non-stationary, difficult to inspect, and irreproducible, while hand-built sandbox benchmarks provide control but cover only a narrow range of layouts, catalogs, policies, and interaction patterns. We argue that the core bottleneck is methodological: the field lacks a scalable way to construct evaluation settings that are simultaneously realistic, diverse, controllable, inspectable, and reproducible. We introduce ShopGym, an integrated framework for realistic simulation and scalable benchmarking of e-commerce web agents. Its simulation layer, ShopArena, converts live seed storefronts into self-contained sandbox shops through anonymized shop specifications and a staged, vali
    
[^234]: D-VLA：面向视觉-语言-动作模型的高并发分布式异步强化学习框架

    D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models

    [https://arxiv.org/abs/2605.13276](https://arxiv.org/abs/2605.13276)

    D-VLA提出了一个面向大规模视觉-语言-动作模型的高并发分布式异步强化学习框架，通过“平面解耦”隔离训练数据与权重控制、四线程异步“泳道”流水线实现完全并行，解决了仿真与深度学习间的资源冲突并大幅提升系统吞吐量。

    

    具身智能的快速发展使视觉-语言-动作（VLA）模型在多模态感知与任务执行方面表现出色。然而，在大规模分布式环境中对这类大型模型应用强化学习（RL）面临严重的系统性瓶颈，其主要源于高保真物理仿真与深度学习对显存和带宽的密集需求之间的资源冲突。这种冲突往往导致整体吞吐量受制于执行阶段的低效问题。为应对这些挑战，我们提出了D-VLA，一个面向大规模具身基础模型的高并发、低延迟分布式强化学习框架。D-VLA引入了“平面解耦”机制，将高频训练数据与低频权重控制进行物理隔离，从而消除仿真与优化之间的相互干扰。我们进一步设计了四线程异步的“泳道”流水线，实现（原文在此处截断）

    arXiv:2605.13276v3 Announce Type: replace  Abstract: The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to the resource conflict between high-fidelity physical simulation and the intensive VRAM/bandwidth demands of deep learning. This conflict often leaves overall throughput constrained by execution-phase inefficiencies. To address these challenges, we propose D-VLA, a high-concurrency, low-latency distributed RL framework for large-scale embodied foundation models. D-VLA introduces "Plane Decoupling," physically isolating high-frequency training data from low-frequency weight control to eliminate interference between simulation and optimization. We further design a four-thread asynchronous "Swimlane" pipeline, enabling full parallel overlap of 
    
[^235]: 机械式与智能体式混合的规约推断

    Hybrid Mechanical and Agentic Specification Inference

    [https://arxiv.org/abs/2605.10005](https://arxiv.org/abs/2605.10005)

    该论文将机械式最弱前置条件（WP）分析与LLM智能体驱动的规约推断相结合，集成到 Move Prover 中自动为 Move 智能合约推导规约，在 Aptos 框架的无规约公开函数上实现了超过 90% 的任务完成率。

    

    形式化验证的采用长期以来一直受到人工编写规约成本的阻碍。为了证明高层属性，人们还必须为辅助函数编写契约，而其中许多是机械性的且处于低层次。规约推断可以从程序中自动推导出此类契约。我们针对智能合约语言 Move 提出了一种将机械式推断与智能体式推断相结合的混合方法，并将其集成到其验证器 Move Prover 中。一个大语言模型（LLM）智能体通过一个调用 Move Prover 的工具来驱动推断与验证，并通过最弱前置条件（WP）分析从代码中推导规约。我们在 26 个目标上评估了该方法，其中 24 个提取自 Aptos 框架中没有规约的公开 Move 函数，测试使用了同一模型家族的前沿模型与中端模型（Sol 5.6 和 Terra 5.6）。无论是否使用 WP 分析，智能体都完成了超过 90% 的运行任务，因此智能体式推断……

    arXiv:2605.10005v3 Announce Type: replace-cross  Abstract: Adoption of formal verification has long been hindered by the cost of manual specification. To prove high-level properties, one must also write contracts for supporting functions, many of which are mechanical and low-level. Specification inference automatically derives such contracts from the program. We present a hybrid of mechanical and agentic inference for Move, a smart contract language, integrated into its verifier, the Move Prover. A large language model (LLM) agent drives inference and verification through a tool that calls the Move Prover and derives specifications from code by weakest-precondition (WP) analysis. We evaluate our approach on 26 targets, 24 of them extracted from public Move functions of the Aptos framework that have no specifications, under a frontier model and a mid-tier model of the same family (Sol 5.6 and Terra 5.6). With and without WP, the agent completes over 90% of its runs, so agentic inference
    
[^236]: 大语言模型会经历内部多重对话吗？通过人格视角研究推理

    Do LLMs Experience an Internal Polylogue? Investigating Reasoning through the Lens of Personas

    [https://arxiv.org/abs/2605.09159](https://arxiv.org/abs/2605.09159)

    本文创新性地将“人格向量”从静态引导工具转变为动态可监控信号，通过追踪其与隐藏激活在生成过程中的对齐序列（“多重对话”）来预测推理正确性，并据此设计分段干预方法提升模型在MMLU-Pro上的准确率。

    

    近期研究表明，大语言模型将行为特征（即“人格”）编码为激活空间中的线性方向，通常被称为“人格向量”。以往的工作将这些方向用作行为引导的静态抓手，而我们则将其视为动态信号：即在推理展开过程中可被监控和干预的探针。我们使用“多重对话”这一术语来表示在生成过程中人格向量与隐藏激活之间对齐的时间序列。在四个开源权重模型上的实验表明，多重对话特征在预测MMLU-Pro上的答案正确性方面与低维激活基线相当，同时仍可通过其关联的人格方向保持可解释性。这些特征还指出了具体的引导目标，即应在回答的不同阶段调节哪些潜在方向。我们将其实例化为一种简单的基于段落条件的干预方法，该方法在四个模型中的三个上提升了准确率……（原文摘要在此处被截断）

    arXiv:2605.09159v3 Announce Type: replace  Abstract: Recent work shows that large language models (LLMs) encode behavioural traits ("personas") as linear directions in activation space, often called "persona vectors". Prior work has used such directions as static handles for behavioural steering. We instead treat them as dynamic signals: probes we can monitor and intervene on as reasoning unfolds. We use the term polylogue to denote the time series of alignments between persona vectors and hidden activations over the course of generation. Experiments across four open-weight models show that polylogue features predict correctness on MMLU-Pro on par with low-dimensional activation baselines, while remaining interpretable through their associated persona directions. They also suggest concrete steering targets, namely which latent directions to modulate at different stages of a response. We instantiate this as a simple paragraph-conditioned intervention that improves accuracy on three of f
    
[^237]: 具有有限时域目标的多环境部分可观测马尔可夫决策过程

    Multi-Environment POMDPs with Finite-Horizon Objectives

    [https://arxiv.org/abs/2605.07537](https://arxiv.org/abs/2605.07537)

    本文针对具有有限时域目标的多环境POMDP，证明了最多n个确定性策略的混合即可覆盖n个环境下的最优策略，并提出了无需存储指数级规模策略描述即可计算最优值与最优策略的新方法。

    

    部分可观测马尔可夫决策过程（POMDP）是指单个智能体与一个随机环境进行交互、且仅能获得当前状态部分信息的系统。在多环境POMDP（MEPOMDP）中，初始状态是未知的，并被假设为以对抗方式选取。在本工作中，我们专注于计算具有有限时域目标的MEPOMDP中的最优值与最优策略。该问题在POMDP中已知是PSPACE完全的。这一扩展并非平凡：MEPOMDP需要跨环境地对信念和收益进行推理，且确定性策略不再足以获得最优收益。虽然下界可以直接传递，但上界的证明需要新的技术：我们证明了对于n个环境，最多n个确定性策略的混合即可满足最优性要求，并展示了如何在不必存储指数级规模策略描述的情况下探索这些混合策略。我们的主要结果如下：(1) 我们建立了……

    arXiv:2605.07537v2 Announce Type: replace  Abstract: Partially Observable Markov Decision Processes (POMDPs) are systems in which one agent interacts with a stochastic environment, and receives only partial information about the current state. In a multi-environment POMDP (MEPOMDP), the initial state is unknown, and assumed to be adversarially chosen. In this work we focus on computing the optimal value and policy in MEPOMDPs with finite-horizon objectives. That problem is known to be PSPACE-complete in POMDPs. The extension is nontrivial: MEPOMDPs require reasoning about beliefs and payoffs across environments, and deterministic policies are no longer enough for optimal payoff. While the lower bound transfers immediately, the upper bound requires new techniques: we prove that mixtures of at most $n$ deterministic policies suffice for $n$ environments and show how to explore them without storing exponential-size policy descriptions. Our main results are as follows: (1) we establish tha
    
[^238]: 揭示与缓解深度伪造视频检测中的时间攻击

    Exposing and Mitigating Temporal Attack in Deepfake Video Detection

    [https://arxiv.org/abs/2605.07398](https://arxiv.org/abs/2605.07398)

    本文揭示了时空深度伪造检测器因过拟合脆弱的时间频谱线索而易受时间攻击的问题，并提出频谱不变防御框架SpInShield，通过可学习频谱对抗器与捷径抑制策略将语义运动与频谱伪影解耦，在模拟攻击下AUC比最强基线提升21.30个百分点。

    

    尽管时空深度伪造检测器能够取得较高的AUC，但我们的实验揭示了这些模型容易受到规避攻击。它们往往在脆弱的时间频谱线索上过拟合，而非学习稳健的语义因果关系。为了缓解这一脆弱性，我们提出了SpInShield，一个时间频谱不变的防御框架，专门设计用于将语义运动与可操纵的频谱伪影解耦。我们提出了一种可学习的频谱对抗器，能够动态合成严重的频谱变形，以模拟极端攻击场景。通过采用捷径抑制优化策略，SpInShield迫使编码器提取可靠的取证线索，同时从潜在空间中清除不稳定的频谱统计特征。实验表明，SpInShield在广泛使用的数据集上取得了有竞争力的性能，并且在模拟幅度攻击下，其AUC比最强基线高出21.30个百分点。

    arXiv:2605.07398v2 Announce Type: replace-cross  Abstract: While spatiotemporal deepfake detectors achieve high AUC, our experiments reveal their susceptibility to evasion attacks. These models tend to overfit on fragile temporal spectrum cues, rather than learning robust semantic causality. To mitigate this vulnerability, we propose SpInShield, a temporal spectral-invariant defense framework explicitly designed to decouple semantic motion from manipulatable spectral artifacts. We propose a learnable spectral adversary that dynamically synthesizes severe spectral deformations, simulating extreme attack scenarios. By employing a shortcut suppression optimization strategy, SpInShield compels the encoder to extract reliable forensic cues while purging unstable spectral statistics from the latent space. Experiments show that SpInShield obtains competitive performance on widely used datasets and outperforms the strongest baseline by 21.30 percentage points in AUC under simulated amplitude s
    
[^239]: 具有可变滞后项的时间序列因果发现

    Time series causal discovery with variable lags

    [https://arxiv.org/abs/2605.04081](https://arxiv.org/abs/2605.04081)

    本文提出了一种可分解的BIC风格评分方法，用于联合学习时间序列因果发现中的变量时间顺序与每个有序变量对的主导滞后项，并通过增强的禁忌搜索算法进行优化。

    

    因果贝叶斯网络是在不确定条件下对复杂现实世界问题进行推理的强大工具。这类问题随时间演变，并在外部冲击发生时作出响应。为支持决策，因果贝叶斯网络需要一张关于所研究变量的因果关系图，即网络的结构。从数据中学习因果模型的图结构仍然具有挑战性；而从时间序列数据中学习则更加困难，因为变量间的依赖关系可能出现在不同的时间滞后处。我们构建了一种方法，可以联合学习变量的时间顺序以及每个选定有序变量对的主导滞后项。我们的贡献是一个适用于混合数据的可分解 BIC 风格目标评分函数，它将节点特定的样本量与偏向近期滞后项的惩罚项相结合。我们使用禁忌搜索对该目标函数进行优化，并通过改变滞后项的搜索动作和贪心的逐子节点滞后调整加以增强，同时其节点级可分解性……

    arXiv:2605.04081v3 Announce Type: replace-cross  Abstract: Causal Bayesian Networks (CBNs) are a powerful tool for reasoning under uncertainty about complex real-world problems. Such problems evolve over time, responding to external shocks as they occur. To support decision-making, CBNs require a cause-and-effect map of the variables under consideration, known as the network's structure. Learning the graphical structure of a causal model from data remains challenging; learning it from time-series data is even harder because dependencies may arise at different time lags. We construct a method that jointly learns a temporal ordering and a dominant lag for each selected ordered variable pair. Our contribution is a decomposable BIC-style objective score for mixed data that combines node-specific sample sizes with a penalty favouring recent lags. We optimise this objective using a Tabu search augmented with change-lag moves and greedy per-child lag tuning, while its node-wise decomposition 
    
[^240]: BoostLoRA：通过Boosting适配器来增长有效秩

    BoostLoRA: Growing Effective Rank by Boosting Adapters

    [https://arxiv.org/abs/2604.27308](https://arxiv.org/abs/2604.27308)

    BoostLoRA提出一种梯度提升框架，通过在错误样本上迭代训练并合并正交子空间的超低秩适配器，使有效秩随轮次线性增长，在不增加任何推理开销的情况下超越了全量微调的性能。

    

    参数高效微调（PEFT）方法面临适配器大小与表达能力之间的权衡：超低参数适配器被限制在固定的低秩子空间中，即使延长训练时间也无法突破性能上限。我们提出BoostLoRA，一种梯度提升框架，通过在当前模型出错的样本上迭代训练并合并极小适配器来克服这一限制。ROTATE SVD基策略将每一轮分配到一个正交子空间，使累积有效秩随轮数线性增长，同时每个适配器仍保持超低秩。合并完成后，适配器即被丢弃，推理阶段零额外开销。在Qwen2.5-3B上，BoostLoRA在GSM8K上达到89.1%，在MATH-500上达到68.8%，超越了最佳的单次超低参数适配器方法（TinyLoRA）和全量微调；在代码生成方面，它在MBPP上达到57.2%，在HumanEval上达到80.4%，而全量微调则降至零样本基线之下。

    arXiv:2604.27308v2 Announce Type: replace-cross  Abstract: Parameter-efficient fine-tuning (PEFT) methods face a tradeoff between adapter size and expressivity: ultra-low-parameter adapters are confined to fixed low-rank subspaces, capping performance even with extended training. We propose BoostLoRA, a gradient-boosting framework that overcomes this limit by iteratively training and merging minimal adapters on the examples the current model gets wrong. A ROTATE SVD basis strategy assigns each round to an orthogonal subspace, so cumulative effective rank grows linearly with the number of rounds while each adapter remains ultra-low-rank. After merging, adapters are discarded, leaving zero inference overhead. On Qwen2.5-3B, BoostLoRA reaches 89.1% on GSM8K and 68.8% on MATH-500, surpassing both the best single-shot ultra-low parameter adapter (TinyLoRA) and full fine-tuning; on code generation it reaches 57.2% on MBPP and 80.4% on HumanEval while full fine-tuning drops below the zero-sho
    
[^241]: 先筛查再解读：一种面向基于基准测试的LLM置信度信号的可移植有效性筛查方法

    Screen Before You Interpret: A Portable Validity Screen for Benchmark-Based LLM Confidence Signals

    [https://arxiv.org/abs/2604.17714](https://arxiv.org/abs/2604.17714)

    本文借鉴临床人格评估的有效性筛查传统，提出一种基于L、Fp和RBS指标的可移植筛查方法，在解读之前检验LLM置信度信号是否真正跟踪正确性，对20个前沿模型的评估发现15个有效、4个不确定、1个无效（DeepSeek-R1弃答了95%的正确答案）。

    

    LLM置信度信号驱动着弃答、路由和升级等决策，然而现有评估很少在计算指标之前检查信号是否随正确性而变化。临床人格评估领域通过在解读实质性量表之前筛查反应有效性来处理类似的问题。我们遵循这一传统，规范了一种可移植的筛查方法。该方法从正确性-置信度列联表中计算三个指标：错误上高置信度的比率（L）、正确答案上低置信度的比率（Fp）以及二者的对比度（RBS，等于负的Youden's J统计量）。它不引入任何新的统计量，而是区分整体高置信度、整体弃答和反转三种失效模式，附加区间保护措施，并且只有当信号可测量地跟踪正确性时才将其判定为有效。在20个前沿模型对452个二选一题目进行二元KEEP/WITHDRAW（保留/弃答）探测的实验中，一个模型被判定为无效（DeepSeek-R1弃答了其95%的正确答案），四个模型为不确定，15个模型为有效。

    arXiv:2604.17714v2 Announce Type: replace-cross  Abstract: LLM confidence signals drive abstention, routing and escalation, yet evaluations rarely check whether a signal varies with correctness before computing metrics on it. Clinical personality assessment handles the analogous problem by screening response validity before interpreting substantive scales. We specify a portable screen in that tradition. From the correctness-by-confidence table it computes the rate of high confidence on errors (L), the rate of low confidence on correct answers (Fp), and their contrast (RBS, equal to minus Youden's J). It adds no new statistics: it separates blanket confidence, blanket withdrawal and inversion, attaches interval guards, and calls a signal Valid only when it measurably tracks correctness. On 20 frontier models answering 452 two-alternative items with binary KEEP/WITHDRAW probes, one model is Invalid (DeepSeek-R1 withdraws 95% of its correct answers), four are Indeterminate and 15 are Vali
    
[^242]: CPGRec+：一种面向个性化视频游戏推荐的平衡导向框架

    CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations

    [https://arxiv.org/abs/2604.14586](https://arxiv.org/abs/2604.14586)

    本文提出了CPGRec+框架，通过偏好感知边缘重加权（PER）模块和利用大型语言模型能力，解决游戏推荐中准确性与多样性的权衡，并缓解过平滑问题。

    

    arXiv:2604.14586v4 公告类型：交叉替换 摘要：游戏行业的快速扩张要求推荐系统适应其动态格局。现有的基于图神经网络（GNN）的方法主要优先考虑准确性而非多样性，忽视了它们之间的固有权衡。为解决这一问题，我们先前提出了CPGRec，一种平衡导向的游戏推荐系统。然而，CPGRec未能考虑玩家-游戏交互中的关键差异，这些差异在反映玩家个人偏好方面具有不同重要性，并可能加剧基于GNN模型固有的过平滑问题。此外，现有方法未充分利用大型语言模型（LLMs）的推理能力和广泛知识来解决这些局限性。为弥补这一差距，我们提出了两个新模块。首先，偏好感知边缘重加权（PER）模块分配带符号的边缘权重，以定性区分显著的玩家兴趣和厌恶。

    arXiv:2604.14586v4 Announce Type: replace-cross  Abstract: The rapid expansion of gaming industry requires advanced recommender systems tailored to its dynamic landscape. Existing Graph Neural Network (GNN)-based methods primarily prioritize accuracy over diversity, overlooking their inherent trade-off. To address this, we previously proposed CPGRec, a balance-oriented gaming recommender system. However, CPGRec fails to account for critical disparities in player-game interactions, which carry varying significance in reflecting players' personal preferences and may exacerbate over-smoothness issues inherent in GNN-based models. Moreover, existing approaches underutilize the reasoning capabilities and extensive knowledge of large language models (LLMs) in addressing these limitations. To bridge this gap, we propose two new modules. First, Preference-informed Edge Reweighting (PER) module assigns signed edge weights to qualitatively distinguish significant player interests and disinterest
    
[^243]: 基于词典的跨语言词义投影生成概念词汇化

    Generating Concept Lexicalizations via Dictionary-Based Cross-Lingual Sense Projection

    [https://arxiv.org/abs/2604.14397](https://arxiv.org/abs/2604.14397)

    提出一种用双语词典增强对齐器并结合过滤策略的跨语言词义投影方法，能够将WordNet式词汇资源自动扩展到新语言，在保持可解释性和资源高效的同时提升精确率。

    

    我们研究通过词义生成来自动将WordNet风格的词汇资源扩展到新语言的任务。我们通过语义投影将目标语言的词条与现有词汇概念相关联来生成词义。给定一个带有词义标注的英语语料库及其翻译，我们的方法将标注好的同义词集（synsets）投影到对齐的目标语言词汇上，并将相应的词条分配给这些同义词集。为了生成对齐并确保其质量，我们使用双语词典对预训练的基础对齐器进行增强，该词典还用于过滤不正确的词义投影。我们在多种语言上对该方法进行评估，并与先前的方法、基于词典的基线以及大型语言模型基线进行比较。结果表明，所提出的投影-过滤策略在保持可解释性和资源高效性的同时提高了精确率。我们发布了代码、文档和生成的词义清单。

    arXiv:2604.14397v3 Announce Type: replace-cross  Abstract: We study the task of automatically expanding WordNet-style lexical resources to new languages through sense generation. We generate senses by associating target-language lemmas with existing lexical concepts via semantic projection. Given a sense-tagged English corpus and its translation, our method projects the annotated synsets onto aligned target-language tokens and assigns the corresponding lemmas to those synsets. To generate alignments and ensure their quality, we augment a pretrained base aligner with a bilingual dictionary, which is also used to filter incorrect sense projections. We evaluate the method on multiple languages, comparing it to prior methods, as well as dictionary-based and large language model baselines. Results show that the proposed project-and-filter strategy improves precision while remaining interpretable and resource- efficient. We release our code, documentation, and generated sense inventories at 
    
[^244]: 用于生成建模的变分测量量子计算中的经典资源最小化

    Minimizing classical resources in variational measurement-based quantum computation for generative modeling

    [https://arxiv.org/abs/2604.11578](https://arxiv.org/abs/2604.11578)

    本文针对变分测量量子计算（VMBQC）中信道模型参数量是幺正模型两倍的局限，提出了最小化经典资源消耗的方案，使生成建模中的经典处理开销得到显著降低。

    

    测量量子计算是一种量子信息处理框架，其中计算任务通过对高度纠缠的资源态执行单量子比特测量来实现。由于量子测量结果的不确定性，这些操作的随机结果若不加校正，将形成一个变分量子信道族。传统上，这种随机性需要通过经典处理加以校正，以确保确定性的幺正计算。最近，变分测量量子计算（VMBQC）被提出，利用这种测量引起的随机性在生成建模中获得优势。该方法的一个局限是，相应的信道模型相比幺正模型具有两倍的参数量，其规模为 $N \times D$，其中 $N$ 是逻辑量子比特的数量（宽度），$D$ 是 VMBQC 模型的深度。这可能……

    arXiv:2604.11578v2 Announce Type: replace-cross  Abstract: Measurement-based quantum computation (MBQC) is a framework for quantum information processing in which a computational task is carried out through one-qubit measurements on a highly entangled resource state. Due to the indeterminacy of the outcomes of a quantum measurement, the random outcomes of these operations, if not corrected, yield a variational quantum channel family. Traditionally, this randomness is corrected through classical processing in order to ensure deterministic unitary computations. Recently, variational measurement-based quantum computation (VMBQC) has been introduced to exploit this measurement-induced randomness to gain an advantage in generative modeling. A limitation of this approach is that the corresponding channel model has twice as many parameters compared to the unitary model, scaling as $N \times D$, where $N$ is the number of logical qubits (width) and $D$ is the depth of the VMBQC model. This can
    
[^245]: 面向可持续时间序列分类的剪枝扩展与效率权衡

    Pruning Extensions and Efficiency Trade-Offs for Sustainable Time Series Classification

    [https://arxiv.org/abs/2604.07953](https://arxiv.org/abs/2604.07953)

    本文提出多维度评估框架、有界误差剪枝策略及新型Hydrant分类器扩展，可在时间序列分类任务中实现能耗降低高达70%，而预测精度几乎不受影响。

    

    大规模时间序列分类需要在预测性能与资源消耗之间取得平衡，而这种平衡取决于给定的数据、模型和硬件。本文提出了一个多维度评估框架，能够对时间序列分类中资源与质量之间的权衡进行显式的基准测试和探索。为了提高效率，我们针对领先的混合分类器 Hydra 和 Quant 提出了一种具有有界误差的剪枝策略，并进一步提出了 Hydrant——一种将区间表示与随机卷积和字典学习相结合的新型扩展方法。通过在 20 个 MONSTER 数据集、10 种方法和三种计算设置上评估超过 4000 种配置，我们系统地分析了模型设计、超参数和硬件选择如何影响实际的时间序列分类性能。我们的结果表明，剪枝能够显著降低能耗高达 70%，同时仅轻微降低预测准确率。

    arXiv:2604.07953v2 Announce Type: replace-cross  Abstract: Time series classification (TSC) at scale requires balancing predictive performance against resource consumption, which is subject to the given data, model, and hardware. This paper introduces a multi-dimensional evaluation framework that allows for explicitly benchmarking and exploring the resource-versus-quality trade-offs in TSC. To improve efficiency, we present a pruning strategy with bounded error for leading hybrid classifiers - Hydra and Quant - and moreover propose Hydrant, a novel extension that integrates interval representations with random convolutions and dictionary learning. With over 4000 evaluated configurations across 20 MONSTER datasets, 10 methods, and three compute setups, we systematically analyze how model design, hyperparameters, and hardware choices impact practical TSC performance. Our results show that pruning can significantly reduce energy consumption by up to 70% while only marginally lowering pred
    
[^246]: 蛋白质设计的生成式机器学习

    Generative machine learning for protein design

    [https://arxiv.org/abs/2603.26378](https://arxiv.org/abs/2603.26378)

    本文系统综述了59种生成式机器学习方法在六类蛋白质设计任务（序列生成、骨架生成、结构条件序列设计、序列-结构联合设计、基序支架及结合蛋白/抗体设计）中的应用，并讨论了设计蛋白质的计算与实验评估方法。

    

    设计具有特定生物功能的蛋白质对于新疗法的开发以及酶的工程化改造都具有重要意义，然而实现预期的蛋白质活性仍然充满挑战。生成式人工智能（AI）通过从蛋白质序列和结构数据中学习，已成为探索庞大可能设计空间的有力工具。本文综述了涵盖六种蛋白质设计任务的59种生成式方法，包括序列生成、骨架生成、结构条件下的序列设计、序列-结构联合设计、基序支架构建以及结合蛋白和抗体设计。我们概述了常见的生成方法和蛋白质表示形式，并讨论了在生成过程中如何利用结构约束和功能信息。随后，我们讨论了用于评估所设计蛋白质的计算与实验方法，以及这些评估……（原文在此处截断）

    arXiv:2603.26378v2 Announce Type: replace-cross  Abstract: Designing proteins with specific biological functions is of considerable interest for both the development of new therapeutics and the engineering of enzymes, yet achieving the intended activity remains challenging. Generative artificial intelligence (AI) has emerged as a powerful tool to explore the vast space of possible designs by learning from protein sequence and structural data. Here, we review 59 generative approaches across six protein design tasks, including sequence generation, backbone generation, structure-conditioned sequence design, joint sequence-structure design, motif scaffolding, and binder and antibody design. We outline common generative approaches and protein representations and discuss how structural constraints and functional information are utilized during generation. This is followed by a discussion of the computational and experimental approaches used to evaluate designed proteins and what such assessm
    
[^247]: FlashSampling：快速且内存高效的确切采样

    FlashSampling: Fast and Memory-Efficient Exact Sampling

    [https://arxiv.org/abs/2603.15854](https://arxiv.org/abs/2603.15854)

    FlashSampling将精确采样融合进LM头的矩阵乘法运算中，通过在片上逐块计算并添加Gumbel噪声避免在显存中物化logits张量，并以流式点对点通信替代all-gather，实现了跨最多8个GPU的快速、精确且内存高效的大词表采样。

    

    arXiv:2603.15854v4 公告类型：replace-cross 摘要：从分类分布中进行采样在数学上很简单，但在大词表解码场景中，它通常会在语言模型输出头（LM head）之后触发额外的内存流量和额外的内核调用。我们提出了FlashSampling，这是一种精确的采样原语，它将采样融合到LM头的矩阵乘法中，并且从不在HBM（高带宽显存）中物化logits张量。该方法十分简单：在芯片上逐块计算logits，添加Gumbel噪声，每行和每个词表块仅保留一个最大值，最后通过对各块进行小型归约完成采样。在张量并行解码中，FlashSampling用流式点对点写入取代了logits的all-gather（全收集）：这使得GPU到GPU的通信与计算和HBM加载相重叠，最多可扩展至8个GPU，并在大批量规模下实现接近理想的扩展性。我们的内核是精确的，因为argmax可以在分区上进行分解；针对在线和张量并行设置的分组变体通过分类分布的层次化分解同样保证了精确性。

    arXiv:2603.15854v4 Announce Type: replace-cross  Abstract: Sampling from a categorical distribution is mathematically simple, but in large-vocabulary decoding, it often triggers extra memory traffic and extra kernels after the LM head. We present FlashSampling, an exact sampling primitive that fuses sampling into the LM-head matmul and never materializes the logits tensor in HBM. The method is simple: compute logits tile-by-tile on chip, add Gumbel noise, keep only one maximizer per row and per vocabulary tile, and finish with a small reduction over tiles. In tensor-parallel decoding, FlashSampling replaces the all-gather of logits with streaming peer-to-peer writes: This overlaps GPU-to-GPU communication with computation and HBM loads across up to 8 GPUs, with near-ideal scaling at large batch sizes. Our kernel is exact because argmax decomposes over partitions; grouped variants for online and tensor-parallel settings are exact by hierarchical factorization of the categorical distribu
    
[^248]: 基于阐明扩散模型的3D卷积与2.5D切片条件U-Net架构在MRI超分辨率中的比较分析

    Comparative Analysis of 3D Convolutional and 2.5D Slice-Conditioned U-Net Architectures for MRI Super-Resolution via Elucidated Diffusion Models

    [https://arxiv.org/abs/2603.14667](https://arxiv.org/abs/2603.14667)

    该研究在阐明扩散模型框架下系统比较了3D卷积U-Net与2.5D切片条件U-Net两种骨干架构用于脑部MRI的2倍面内超分辨率重建，发现两者总体性能相当，在训练退化条件下PSNR分别为37.72 dB和37.70 dB。

    

    磁共振成像（MRI）超分辨率（SR）方法通过计算方式增强低分辨率采集图像，为更高场强扫描仪提供了一种软件层面的替代方案。我们研究了一种用于2倍面内脑部MRI超分辨率的阐明扩散模型（EDM）框架，并比较了两种U-Net骨干网络：（i）处理体积块的3D卷积U-Net，以及（ii）2.5D切片条件U-Net，该网络在对相邻两个切片进行条件约束的同时对每个切片进行超分辨率处理。所有模型均在FOMO60K数据集NKI队列的500名受试者上从头训练至收敛，使用单独的6名受试者验证集进行检查点选择，并在70名保留受试者上以受试者级别的统计指标与在相同数据上训练的基线模型（EDSR、Swin2SR、Res-SRDiff）进行比较。两种架构总体上没有明显的优劣之分。在训练所用的退化条件下，两者表现持平（3D模型为37.72 dB PSNR，2.5D模型为37.70 dB PSNR），……

    arXiv:2603.14667v2 Announce Type: replace-cross  Abstract: Magnetic resonance imaging (MRI) super-resolution (SR) methods that computationally enhance low-resolution acquisitions offer a software alternative to higher-field scanners. We study an elucidated diffusion model (EDM) framework for 2x in-plane brain MRI SR and compare two U-Net backbones: (i) a 3D convolutional U-Net that processes volumetric patches and (ii) a 2.5D slice-conditioned U-Net that super-resolves each slice while conditioning on its two neighbouring slices. All models are trained from scratch to convergence on 500 subjects of the NKI cohort of FOMO60K, with a separate 6-subject validation set for checkpoint selection, and evaluated on 70 held-out subjects with subject-level statistics against baselines (EDSR, Swin2SR, Res-SRDiff) trained on the same data. Neither architecture is better overall. On the degradation used in training the two are tied (37.72 dB PSNR for the 3D model and 37.70 dB for the 2.5D model), 0
    
[^249]: 神经回路学习中的变分潜在平衡

    A Variational Latent Equilibrium for Learning in Neuronal Circuits

    [https://arxiv.org/abs/2603.09600](https://arxiv.org/abs/2603.09600)

    该论文提出了一种基于神经元状态前瞻性能量函数的通用形式化框架，以可控且生物学合理的方式统一并扩展了多种局部时空信用分配方法，从而实现对随时间反向传播（BPTT）的近似，用于学习复杂时间依赖性。

    

    大脑在识别和生成复杂时空模式方面的能力仍然无与伦比。尽管人工智能能够复现其中部分能力，但深度学习算法在很大程度上与我们目前对大脑回路及其动力学的理解相悖。随时间反向传播（BPTT）——学习复杂时间依赖性的首选算法——便是其中的典型例子。在这项工作中，我们提出了一个通用形式化框架，以可控且生物学上合理的方式来近似BPTT。我们的方法基于、统一并扩展了此前多种基于能量守恒和极值作用量原理的局部、时间连续、无相位（phase-free）的时空信用分配方法。我们的出发点是神经元状态的前瞻性能量函数，由此计算时间连续神经网络的实时误差动力学。基于此，作为第一个贡献，我们提供了一个简单且……（原文摘要到此截断）

    arXiv:2603.09600v3 Announce Type: replace-cross  Abstract: Brains remain unrivaled in their ability to recognize and generate complex spatiotemporal patterns. While AI is able to reproduce some of these capabilities, deep learning algorithms remain largely at odds with our current understanding of brain circuitry and dynamics. This is prominently the case for backpropagation through time (BPTT), the go-to algorithm for learning complex temporal dependencies. In this work we propose a general formalism to approximate BPTT in a controlled, biologically plausible manner. Our approach builds on, unifies and extends several previous approaches to local, time-continuous, phase-free spatiotemporal credit assignment based on principles of energy conservation and extremal action. Our starting point is a prospective energy function of neuronal states, from which we calculate real-time error dynamics for time-continuous neuronal networks. With this, as a first contribution we provide a simple and
    
[^250]: 基于参数与数据高效适配的草稿模型高效对齐方法

    Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation

    [https://arxiv.org/abs/2603.09527](https://arxiv.org/abs/2603.09527)

    本文提出参数与数据高效的草稿模型适配框架 EDA，通过解耦架构、数据再生策略和样本选择机制三项创新，高效地将草稿模型对齐到微调后的目标模型，从而提升推测解码的加速效果。

    

    推测解码可以加速大语言模型的推理，但当目标模型针对特定领域进行微调后，其性能会出现下降。一种直接的做法是为每个目标模型重新训练草稿模型，但这成本高昂且效率低下。为解决这一问题，我们提出了一种实用的参数与数据高效框架，称为“高效草稿适配”，简称 EDA，用于高效地适配草稿模型。EDA 引入了三项创新：（1）解耦架构，利用共享组件和私有组件分别建模共享的输出分布与目标特定的输出分布，只需更新轻量级的私有组件即可实现参数高效的适配；（2）数据再生策略，利用微调后的目标模型重新生成训练数据，从而提升训练与推测解码之间的对齐程度，带来更高的平均接受长度；（3）样本选择机制（摘要原文在此处被截断）。

    arXiv:2603.09527v2 Announce Type: replace-cross  Abstract: Speculative decoding accelerates LLM inference but suffers from performance degradation when target models are fine-tuned for specific domains. A naive solution is to retrain draft models for every target model, which is costly and inefficient. To address this, we introduce a practical parameter- and data-efficient framework named Efficient Draft Adaptation, abbreviated as EDA, for efficiently adapting draft models. EDA introduces three innovations: (1) a decoupled architecture that utilizes shared and private components to model the shared and target-specific output distributions separately, enabling parameter-efficient adaptation by updating only the lightweight private component; (2) a data regeneration strategy that utilizes the fine-tuned target model to regenerate training data, thereby improving the alignment between training and speculative decoding, leading to higher average acceptance length; (3) a sample selection me
    
[^251]: ViroGym：用于评估病毒蛋白的逼真大规模基准测试

    ViroGym: Realistic Large-Scale Benchmarks for Evaluating Viral Proteins

    [https://arxiv.org/abs/2603.06740](https://arxiv.org/abs/2603.06740)

    ViroGym是一个针对病毒蛋白的大规模综合基准，涵盖79项深度突变扫描实验、21项流感中和任务和SARS-CoV-2现实世界大流行预测，发现ProGen2系列在所有任务中表现最佳，且DMS和中和任务的表现可可靠地预测模型在现实世界新变异出现时的泛化能力。

    

    蛋白质语言模型在零样本预测错义变异效应方面展现出强大潜力，然而针对病毒蛋白的系统性基准测试仍然有限——鉴于人们需要能够在实验验证之前预测新出现突变的主动性工具，这是一个关键缺口。在此，我们推出了ViroGym，这是一个跨三项任务对蛋白质语言模型进行评估的综合性基准：涵盖真核病毒的79项深度突变扫描（DMS）实验，包含7种表型读数下的552,065条突变序列；21项流感中和任务；以及一个针对SARS-CoV-2的现实世界大流行预测任务。我们在适应度景观、抗原多样性和大流行预测方面对成熟的蛋白质语言模型进行了基准测试，发现ProGen2系列在所有三项任务中均持续取得最强性能。至关重要的是，DMS和中和性能能够可靠地识别出可推广至现实世界新出现变异的模型，即使……

    arXiv:2603.06740v3 Announce Type: replace-cross  Abstract: Protein language models (pLMs) have shown strong potential for zero-shot prediction of missense variant effects, yet systematic benchmarking on viral proteins remains limited, a critical gap given the need for proactive tools that can anticipate emerging mutations ahead of experimental validation. Here we introduce ViroGym, a comprehensive benchmark evaluating pLMs across three tasks: 79 deep mutational scanning (DMS) assays covering eukaryotic viruses with 552,065 mutated sequences across 7 phenotypic readouts, 21 influenza neutralisation tasks, and a real-world pandemic prediction task for SARS-CoV-2. We benchmark well-established pLMs on fitness landscapes, antigenic diversity, and pandemic forecasting, and find that the ProGen2 family consistently achieves the strongest performance across all three tasks. Crucially, DMS and neutralisation performance reliably identifies models that generalise to real-world emergence, even t
    
[^252]: ESENSC：SHAP的一种多项式时间公理化替代方法

    ESENSC: A Polynomial-Time Axiomatic Alternative to SHAP

    [https://arxiv.org/abs/2603.00496](https://arxiv.org/abs/2603.00496)

    ESENSC是一种仅需线性次联盟值评估的多项式时间确定性归因方法，在表格预测任务上与精确SHAP高度一致，同时计算成本远低于后者。

    

    我们提出了ESENSC，一种计算高效且具有理论基础SHAP替代方法。ESENSC是一种确定性的、闭式解析的归因规则，只需线性次数的联盟值评估，满足空玩家性质，并基于效率公理、受限形式的差分边际性以及明确的计算约束，获得了公理化刻画。在实证方面，在使用神经网络和XGBoost模型的表格预测任务上，ESENSC在归因大小和特征排序上均与精确SHAP高度一致，同时所需计算量大幅减少。在所评估的设置下，其表现优于广泛使用的基于采样的SHAP近似方法。精确SHAP需要指数次联盟评估，而ESENSC的计算量随特征数量呈线性扩展。这些结果表明，ESENSC在与精确SHAP的一致性和计算效率之间实现了良好的权衡。

    arXiv:2603.00496v2 Announce Type: replace-cross  Abstract: We propose ESENSC, a computationally efficient and theoretically grounded alternative to SHAP. ESENSC is a deterministic, closed-form attribution rule that requires only linearly many coalition-value evaluations, satisfies the null-player property, and admits an axiomatic characterization based on efficiency, a restricted form of differential marginality, and explicit computational constraints.Empirically, on tabular prediction tasks using neural network and XGBoost models, ESENSC shows close agreement with exact SHAP in both attribution magnitude and feature rankings while requiring substantially less computation. Under the evaluated settings, it compares favorably with widely used sampling-based SHAP approximations. While exact SHAP requires exponentially many coalition evaluations, ESENSC scales linearly with the number of features. These results show that ESENSC offers a favorable trade-off between agreement with exact SHAP
    
[^253]: 基于非相干性调整语义体积的多模态大语言模型不确定性量化

    Uncertainty Quantification for Multimodal Large Language Models with Incoherence-adjusted Semantic Volume

    [https://arxiv.org/abs/2602.24195](https://arxiv.org/abs/2602.24195)

    提出了UMPIRE，一个免训练且无需外部工具的多模态大语言模型不确定性量化框架，其受行列式点过程启发，通过计算新颖的“非相干性调整语义体积”，将语义嵌入离散度与响应二次熵相结合，从而高效支持多种输入输出模态。

    

    尽管多模态大语言模型（MLLM）能力强大，但它们可能产生看似合理实则错误的输出，这阻碍了其可靠部署。准确的不确定性度量可以将不可靠的查询升级给人类专家或更大的模型处理，从而提升性能。然而，现有的不确定性度量存在实际限制，例如仅针对特定模态设计、依赖外部工具或计算成本高昂。我们提出了UMPIRE，这是一个面向多模态大语言模型的免训练不确定性量化框架，它能够在各种输入和输出模态上高效工作，无需外部工具，仅依赖模型自身的内部模态特征。受行列式点过程（DPPs）的启发，UMPIRE针对给定任务实例计算采样得到的MLLM响应的一种新颖的非相干性调整语义体积，该方法将语义嵌入的离散度与响应的二次熵相结合。

    arXiv:2602.24195v2 Announce Type: replace  Abstract: Despite their capabilities, Multimodal Large Language Models (MLLMs) may produce plausible but erroneous outputs, hindering reliable deployment. Accurate uncertainty metrics could enable escalation of unreliable queries to human experts or larger models for improved performance. However, existing uncertainty metrics have practical constraints, such as being designed only for specific modalities, reliant on external tools, or computationally expensive. We introduce UMPIRE, a training-free uncertainty quantification framework for MLLMs that works efficiently across various input and output modalities without external tools, relying only on the models' own internal modality features. Inspired by Determinantal Point Processes (DPPs), UMPIRE computes a novel incoherence-adjusted semantic volume of sampled MLLM responses for a given task instance, which couples semantic embedding dispersion with the quadratic entropy of the responses. This
    
[^254]: 递归信念视觉-语言-动作模型

    Recursive Belief Vision Language Action Models

    [https://arxiv.org/abs/2602.20659](https://arxiv.org/abs/2602.20659)

    RB-VLA提出一种以信念为中心的视觉-语言-动作模型架构，通过自监督世界模型目标训练，维护编码任务相关历史、动力学与物体交互的紧凑潜在信念状态，每个任务仅需查询一次VLM即可支持部分可观测条件下的长时程多阶段操控。

    

    视觉-语言-动作模型必须使智能体能够在部分可观测条件下执行长时程任务。然而，大多数现有方法仍然是观测驱动的，依赖于较短的上下文窗口或对视觉语言模型（VLM）的重复查询，这会导致任务进度丢失、感知混叠下的动作重复以及较高的推理延迟。尽管语义接地很重要，但长时程操作从根本上需要持久的、以动作为条件的状态表示。当前的VLA缺乏此类表示，且在时间和物理推理方面的能力有限，因此不适合多阶段控制。本文提出RB-VLA，这是一种以信念为中心的架构，通过自监督世界模型目标进行训练，维护一个紧凑的潜在状态，编码与任务相关的历史、动力学和物体交互。每个任务只需查询一次VLM，VLM提供高层意图，而信念状态则跟踪……

    arXiv:2602.20659v3 Announce Type: replace  Abstract: Vision-language-action models must enable agents to execute long-horizon tasks under partial observability. However, most existing approaches remain observation-driven, relying on short context windows or repeated queries to vision-language models (VLMs). This leads to loss of task progress, action repetition under perceptual aliasing, and high inference latency. While semantic grounding is important, long-horizon manipulation fundamentally requires persistent, action-conditioned state representations. Current VLAs lack such representations and exhibit limited temporal and physical reasoning, making them ill-suited for multi-stage control. This paper introduces RB-VLA, a belief-centric architecture trained with self-supervised world-model objectives that maintains a compact latent state encoding task-relevant history, dynamics, and object interactions. Queried once per task, the VLM provides high-level intent, while the belief tracks
    
[^255]: 通过可复用性与可验证性评估思维链推理

    Evaluating Chain-of-Thought Reasoning through Reusability and Verifiability

    [https://arxiv.org/abs/2602.17544](https://arxiv.org/abs/2602.17544)

    该论文提出用发送方-接收方框架评估模型间思维链推理传递的可靠性、有用性和安全性，并引入可验证性与可纠正性两个度量指标。

    

    AI应用日益由多个模型组合而成，其中一个模型基于另一个模型产生的输出进行操作。所传递的文本还可以携带发送方模型的推理过程。思维链（Chain-of-Thought）是我们在实验中使用的此类文本的实例。我们针对这种模型间推理传递提出三个问题：推理是否被可靠地传递？它对接收方是否有用？它是否安全？我们通过“发送方-接收方”实验设置来研究这三个问题：一个发送方LLM生成推理轨迹，一个独立的接收方LLM必须基于该轨迹执行操作。两个模型都不知道彼此的身份，因此发送方无法针对特定接收方定制其推理。可靠性关注接收方能否恢复出发送方所预期的结论，我们用“可验证性”来衡量它。有用性关注所传递的推理能否帮助接收方纠正自身的错误，我们用“可纠正性”来衡量它。安全性关注所传递的推理是否使接收方……

    arXiv:2602.17544v2 Announce Type: replace  Abstract: AI applications increasingly compose multiple models, where one model acts on the output produced by another model. The transferred text can also carry the reasoning of the sender model. Chain-of-thought is our experimental instance of such text. We ask three questions about such inter-model reasoning transfer. Is the reasoning transmitted reliably? Is it useful to the receiver? Is it safe? We study these three questions with a Sender-Receiver setup. A Sender LLM produces a reasoning trace. A separate Receiver LLM must act on it. Neither model knows the identity of the other. The Sender therefore cannot tailor its reasoning to a particular Receiver. Reliability asks whether the Receiver recovers the Sender's intended conclusion. We measure it with Verifiability. Usefulness asks whether transferred reasoning helps a Receiver correct its mistakes. We measure it with Correctability. Safety asks whether transferred reasoning makes a Rece
    
[^256]: ScrapeGraphAI-100k：面向模式约束的大语言模型生成数据集

    ScrapeGraphAI-100k: Dataset for Schema-Constrained LLM Generation

    [https://arxiv.org/abs/2602.15189](https://arxiv.org/abs/2602.15189)

    本文发布了首个基于真实使用遥测数据构建的大规模模式约束抽取数据集 ScrapeGraphAI-100k，包含 93,695 条配对真实网页内容、提示词、模式与大语言模型响应及符合性标签的事件，覆盖 25,453 个独特模式，并揭示了模式复杂度增长带来的急剧失败阈值。

    

    生成符合指定 JSON 模式（schema）的输出，是现代大语言模型中工具调用、结构化抽取和知识库构建的基础。尽管这一任务至关重要，但现有的公开数据集仍然规模小、多为合成数据或仅包含纯文本，且很少将真实网页内容与实践中使用的提示词和模式配对。我们提出了 ScrapeGraphAI-100k 数据集，包含 93,695 条模式约束抽取事件，这些事件通过 ScrapeGraphAI 的选择性退出（opt-out）遥测机制于 2025 年第二至第三季度收集，并从 900 万条原始事件中按模式去重和均衡而来。该语料库涵盖 25,453 个独特模式和 15 种可识别的语言，其中英语内容占检测内容的 89%。每个实例将转换为 Markdown 格式的网页内容与提示词、模式、大语言模型响应以及基于 jsonschema-rs 的逐样本结构符合性标签配对。我们刻画了整个语料库的结构多样性，并发现随着模式复杂度的增长，模型失败率存在急剧上升的临界阈值。作为案例研究，一个 1.（原文摘要在此处截断）

    arXiv:2602.15189v3 Announce Type: replace-cross  Abstract: Producing output that conforms to a specified JSON schema underlies tool use, structured extraction, and knowledge base construction in modern large language models. Despite this centrality, public datasets for the task remain small, synthetic, or text-only, and rarely pair real page content with the prompts and schemas used in practice. We introduce ScrapeGraphAI-100k, 93,695 schema-constrained extraction events collected via opt-out ScrapeGraphAI telemetry in Q2--Q3 2025, deduplicated and balanced by schema from 9M raw events. The corpus spans 25,453 unique schemas, across 15 named languages, with English covering 89% of detected content. Each instance pairs Markdown-converted page content with a prompt, schema, LLM response, and per-example jsonschema-rs structural conformance labels. We characterize structural diversity across the corpus and identify sharp failure thresholds as schema complexity grows. As a case study, a 1.
    
[^257]: ROCKET：通过校准引导的背包增强截断实现快速优化的高效模型压缩方法

    ROCKET: Rapid Optimization via Calibration-guided Knapsack Enhanced Truncation for Efficient Model Compression

    [https://arxiv.org/abs/2602.11008](https://arxiv.org/abs/2602.11008)

    ROCKET是一种免训练的模型压缩方法，通过将逐层压缩分配建模为多选择背包问题，并结合基于校准集的单步稀疏矩阵分解，在全局压缩预算下实现了超越现有方法的最先进压缩性能。

    

    我们提出了ROCKET，这是一种免训练的模型压缩方法，与因式分解、结构化稀疏化和动态压缩等基线方法相比，实现了最先进的性能。在全局压缩预算的约束下，ROCKET包含两项关键创新：首先，它将逐层压缩分配问题建模为一个多选择背包问题，在满足目标模型大小的前提下，为每一层选择最优压缩级别以最小化总重构误差。其次，它引入了一种受字典学习启发的单步稀疏矩阵分解方法：仅使用一个小型校准集，基于激活-权重敏感度对权重系数进行稀疏化，然后通过最小二乘法以闭式解形式更新字典，完全绕过了迭代优化、稀疏编码或反向传播。ROCKET在不同模型上持续超越现有的压缩方法。

    arXiv:2602.11008v2 Announce Type: replace-cross  Abstract: We present ROCKET, a training-free model compression method that achieves state-of-the-art performance in comparison with factorization, structured-sparsification and dynamic compression baselines. Operating under a global compression budget, ROCKET comprises two key innovations: First, it formulates layer-wise compression allocation as a multi-choice knapsack problem, selecting the optimal compression level for each layer to minimize total reconstruction error while adhering to a target model size. Second, it introduces a single-step sparse matrix factorization inspired by dictionary learning: using only a small calibration set, it sparsifies weight coefficients based on activation-weights sensitivity and then updates the dictionary in closed form via least squares bypassing iterative optimization, sparse coding, or backpropagation entirely. ROCKET consistently outperforms existing compression approaches across different model
    
[^258]: 从对话到市场：AgenticPay——用于多智能体商务中LLM驱动的谈判

    From Chats to Markets: AgenticPay for LLM-Powered Negotiation in Multi-Agent Commerce

    [https://arxiv.org/abs/2602.06008](https://arxiv.org/abs/2602.06008)

    AgenticPay是一个包含160个多模态任务、覆盖四种商业场景和八种市场拓扑的统一基准，用于评估多模态LLM智能体在竞争性多智能体市场中通过多轮谈判达成高福利、多维度协议的能力。

    

    大型语言模型智能体越来越被期望能够自主进行谈判和交易，然而现有的基准测试很少将多模态产品定位、多议题讨价还价和竞争性多智能体市场结合起来。我们提出了AgenticPay，这是一个统一的基准，用于评估多模态智能体在现实市场中达成高福利、多维度协议的能力。AgenticPay包含160个多模态任务，涵盖四种商业场景（电子商务、外卖配送、网约车和公寓租赁）以及八种市场拓扑结构，从双边谈判到多对多竞争。智能体需要解读产品图片、推断对手的隐藏偏好，并通过多轮自然语言对话协商具有约束力的合同条款，如价格、租期、退货政策和配送速度。该基准采用基于效用的框架来评估协议质量和联合福利。使用专有模型（GPT-5.4、Claude Son……（摘要原文在此处被截断）的实验……

    arXiv:2602.06008v2 Announce Type: replace  Abstract: Large language model agents are increasingly expected to negotiate and transact autonomously, yet existing benchmarks rarely combine multimodal product grounding, multi-issue bargaining, and competitive multi-agent markets. We introduce AgenticPay, a unified benchmark for evaluating multimodal agents' ability to reach high-welfare, multidimensional agreements in realistic markets. AgenticPay includes 160 multimodal tasks across four business scenarios (E-commerce, Food Delivery, Ride-hailing, and Apartment Rental) and eight market topologies, from bilateral bargaining to many-to-many competition. Agents interpret product images, infer opponents' hidden preferences, and negotiate binding contract terms such as price, lease duration, return policy, and delivery speed through multi-round natural-language dialogue. A utility-based framework evaluates agreement quality and joint welfare. Experiments with proprietary (GPT-5.4, Claude Sonne
    
[^259]: 对齐诅咒：模态对齐通过文本迁移大幅增强音频攻击

    The Alignment Curse: Modality Alignment Supercharges Audio Attacks via Text Transfer

    [https://arxiv.org/abs/2602.02557](https://arxiv.org/abs/2602.02557)

    提出并验证了“对齐诅咒”原则——全模态模型中文本-音频模态对齐越强，文本越狱攻击向音频模态的迁移就越有效，揭示了模型能力提升与安全性之间的根本张力。

    

    端到端训练的全模态模型的最新进展通过强化文本-音频模态对齐，显著提升了音频能力。然而，这种对齐是否会在无意中促进安全漏洞的跨模态迁移，目前仍未得到充分研究。这一问题至关重要，因为基于文本的越狱攻击远比基于音频的攻击成熟；如果这些攻击能够系统性地迁移，当前的音频安全评估可能低估源自文本模态的风险。在本文中，我们提出了“对齐诅咒”，这是一个经过形式化表征和实证验证的原则，表明更强的模态对齐使攻击从文本到音频的迁移更加有效，揭示了能力与安全之间的根本张力。基于这一原则，我们对近期全模态模型（如 Qwen2.5-Omni 等）上的三类攻击进行了全面的黑盒评估……

    arXiv:2602.02557v3 Announce Type: replace-cross  Abstract: Recent advances in end-to-end trained omni-models have substantially improved audio capabilities by strengthening text-audio modality alignment. However, whether such alignment inadvertently facilitates the transfer of safety vulnerabilities across modalities remains underexplored. This question is critical as text-based jailbreak attacks are considerably more mature than audio-based ones; if they transfer systematically, current audio safety evaluations may underestimate risks originating from the text modality. In this paper, we introduce the Alignment Curse, a formally characterized and empirically validated principle showing that stronger modality alignment enables more effective transfer of attacks from text to audio, revealing a fundamental tension between capability and safety. Motivated by this principle, we conduct a comprehensive black-box evaluation of three attack categories on recent omni-models (e.g., Qwen2.5-Omni
    
[^260]: 在二维笔迹中勾勒三维解剖：投影衍生的掩膜能否替代三维分割用于颈椎骨折识别？

    Tracing 3D Anatomy in 2D Strokes: Can Projection-Derived Masks Replace 3D Segmentation for Cervical Spine Fracture Recognition?

    [https://arxiv.org/abs/2601.15235](https://arxiv.org/abs/2601.15235)

    本研究提出一种多阶段流水线，通过从二维投影近似生成椎体掩膜来替代完整的三维分割，在大幅降低计算复杂度的同时，仍能保留足够的诊断上下文以实现颈椎骨折的高精度识别。

    

    颈椎骨折需要快速且准确的诊断，然而由于微小的损伤必须在大体积三维影像中进行评估，自动CT判读仍然充满挑战。我们提出这样的问题：骨折识别是否必须进行完整的三维椎体分割，还是从二维投影近似得到的椎体掩膜即可保留足够的诊断上下文信息。在所提出的多阶段流水线中，YOLOv8检测器从多视图方差投影中定位颈椎（三维平均交并比为94.45%）。随后在基于能量的矢状面和冠状面投影上，使用DenseNet121-Unet执行多标签椎体分割（平均Dice系数为87.86%）。预测的二维掩膜被反投影并融合为每个椎体的近似三维掩膜，用于从原始CT中提取感兴趣体积，再由2.5D时空序列CNN-Transformer模型集成进行分析。在具有自然患病率的完整RSNA 2022数据集上……

    arXiv:2601.15235v5 Announce Type: replace-cross  Abstract: Cervical spine fractures require rapid and accurate diagnosis, yet automatic CT interpretation remains challenging as subtle injuries must be assessed across large 3D volumes. We ask whether full 3D vertebra segmentation is necessary for fracture recognition, or whether vertebra masks approximated from 2D projections preserve sufficient diagnostic context. In the proposed multi-stage pipeline, a YOLOv8 detector localizes the cervical spine from multi-view variance projections (3D mean Intersection over Union 94.45%). Multi-label vertebra segmentation is performed with a DenseNet121-Unet on energy-based sagittal and coronal projections (mean Dice 87.86%). The predicted 2D masks are back-projected and fused into approximate 3D masks per vertebra to extract volumes of interest from the original CT, which are analyzed by an ensemble of 2.5D spatio-sequential CNN-Transformer models. On the full RSNA 2022 cohort at native prevalence 
    
[^261]: ISLA：基于集成学习与域适应的可泛化急性缺血性卒中病灶弥散MRI分割方法及临床验证

    ISLA: Ensemble learning and domain adaptation for generalizable acute ischemic stroke lesion segmentation in diffusion MRI with clinical validation

    [https://arxiv.org/abs/2601.08732](https://arxiv.org/abs/2601.08732)

    该研究提出了基于nnU-Net设计原则的急性缺血性卒中病灶分割框架ISLA，通过集成学习与利用无标签目标域数据的半监督域适应变体ISLA-MT，显著提升了模型在不同临床中心的泛化能力，并完成了临床验证。

    

    在MRI上对急性缺血性卒中（AIS）病灶进行准确分割对于临床决策至关重要。深度学习方法为实现这一任务的自动化提供了有前景的途径。然而，这些方法在未见过的临床中心往往泛化能力较差，且很少有AIS病灶分割方法经过外部验证。在本研究中，我们提出了ISLA（缺血性卒中病灶分析器），这是一个基于nnU-Net设计原则构建的AIS病灶分割框架。ISLA结合了注意力模块、深度监督，以及在不同输入分辨率、卷积块类型和损失函数上的集成学习。我们还引入了ISLA-MT，这是一个通过半监督学习利用无标签目标域图像的变体。我们在SOOP和ISLES 2022数据集上训练了这两种方法，并在包含253例病例的内部留出验证集上进行评估，同时……

    arXiv:2601.08732v2 Announce Type: replace-cross  Abstract: Accurate segmentation of acute ischemic stroke (AIS) lesions on MRI is essential for clinical decision-making. Deep learning methods offer a promising approach to automating this task. However, they often generalize poorly to unseen clinical sites, and few AIS lesion segmentation methods have been externally validated. In this study, we propose $\mathit{ISLA}$ (Ischemic Stroke Lesion Analyzer), a framework for AIS lesion segmentation built on the design principles of $\mathit{nnU\text{-}Net}$. $\mathit{ISLA}$ combines attention modules, deep supervision, and ensemble learning across different input resolutions, convolutional block types, and loss functions. We also introduce $\mathit{ISLA\text{-}MT}$, a variant that leverages unlabeled target-domain images through semi-supervised learning. We trained both approaches on the SOOP and ISLES 2022 datasets and evaluated on a held-out internal validation set of 253 cases, as well as 
    
[^262]: 视觉语言模型在物理动力学与意图推理方面落后于人类表现

    Vision-language models lag human performance on physical dynamics and intent reasoning

    [https://arxiv.org/abs/2601.01547](https://arxiv.org/abs/2601.01547)

    该论文提出“目的性空间智能”（TSI）概念，并基于11,328个真实世界视频构建了EscherVerse大规模基准与指令微调数据集，揭示当前最先进的视觉语言模型在整合物理动力学与人类意图进行空间推理方面仍显著落后于人类。

    

    空间智能是具身认知的核心，然而当代AI系统仍然难以在开放世界的人类环境中对物理交互进行推理。尽管在受控基准测试中表现强劲，视觉语言模型往往无法联合建模物理动力学、参考系以及驱动空间变化的潜在人类意图。我们提出了目的性空间智能，它捕捉了空间推理的一个重要方面：整合物理动力学与行动目标来推理行动及其空间后果。为了评估TSI，我们提出了EscherVerse，一个基于11,328个真实世界视频构建的大规模开放世界资源，包括一个包含8,000个样本的基准测试集和一个包含35,963个样本的指令微调集。通过对27个最先进的视觉语言模型的评估，以及对来自11名标注者的首次人类响应的独立分析，我们发现了一种持续存在……

    arXiv:2601.01547v3 Announce Type: replace-cross  Abstract: Spatial intelligence is central to embodied cognition, yet contemporary AI systems still struggle to reason about physical interactions in open-world human environments. Despite strong performance on controlled benchmarks, vision-language models often fail to jointly model physical dynamics, reference frames, and the latent human intentions that drive spatial change. We introduce Teleo-Spatial Intelligence (TSI), which captures an important aspect of spatial reasoning: integrating physical dynamics and action goals to reason about actions and their spatial consequences. To evaluate TSI, we present EscherVerse, a large-scale open-world resource built from 11,328 real-world videos, including an 8,000-example benchmark and a 35,963-example instruction-tuning set. Across 27 state-of-the-art vision-language models and an independent analysis of first-pass human responses from 11 annotators, we identify a persistent teleo-spatial rea
    
[^263]: NEAT：用于3D分子生成的邻域引导、高效、自回归集合Transformer

    NEAT: Neighborhood-Guided, Efficient, Autoregressive Set Transformer for 3D Molecular Generation

    [https://arxiv.org/abs/2512.05844](https://arxiv.org/abs/2512.05844)

    提出NEAT模型，通过新颖的邻域引导训练策略将分子图视为原子集合并学习与顺序无关的token分布，实现原子级置换不变性，在QM9和GEOM-Drugs数据集上达到最先进的3D分子生成质量。

    

    基于Transformer的自回归模型为3D分子生成提供了一种区别于扩散模型和流模型的高效替代方案。然而仍存在一个挑战：标准Transformer架构需要token具有顺序排列，而对于分子中的原子来说，这种顺序并没有天然的定义。先前的工作通过使用规范原子排序来解决这一问题。然而，这些方法相对于原子不具备置换不变性，并且会使下一token的预测偏向于特定的排序惯例。我们通过引入一种新颖的邻域引导训练策略克服了这一限制。我们的模型NEAT（邻域引导、高效、自回归集合Transformer）将分子图视为原子的集合，并在图边界处学习一个与顺序无关的可容许token分布，从而确保了原子级别的置换不变性。NEAT在QM9和GEOM-Drugs数据集上实现了最先进的生成质量。

    arXiv:2512.05844v4 Announce Type: replace-cross  Abstract: Transformer-based autoregressive models offer an efficient alternative to diffusion- and flow-based approaches for generating 3D molecules. One challenge remains: standard transformer architectures require a sequential ordering of tokens, which is not inherently defined for the atoms in a molecule. Prior works have addressed this by using canonical atom orderings. However, these approaches are not permutation invariant w.r.t. atoms and bias next-token prediction towards ordering conventions. We overcome this limitation by introducing a novel neighborhood-guided training strategy. Our model, NEAT (Neighborhood-Guided, Efficient, Autoregressive Set Transformer) treats molecular graphs as sets of atoms and learns an order-agnostic distribution over admissible tokens at the graph boundary, thereby ensuring atom-level permutation invariance. NEAT achieves state-of-the-art generation quality on the QM9 and GEOM-Drugs datasets, enable
    
[^264]: 基于分位数曲线的模拟器保真度无模型评估

    Model-Free Assessment of Simulator Fidelity via Quantile Curves

    [https://arxiv.org/abs/2512.05024](https://arxiv.org/abs/2512.05024)

    该论文提出一种无模型方法，通过为潜在总体参数构建置信集并估计分位数曲线，来稳健地量化生成式AI模拟器与真实系统之间随情景变化的“仿真-现实”差距。

    

    随着生成式AI模型越来越多地被用于模拟真实世界系统，量化“仿真到现实”差距变得至关重要。对于每个感兴趣的输入设置——我们称之为“情景”，例如调查问题或运行条件——真实系统和模拟系统都与未观测的潜在总体参数相关联，且二者的差异因情景而异。一个根本性的挑战在于，对于任何给定情景，这种差异无法直接观测，因为两个系统都只能通过有限样本来访问，且各情景间的样本量往往大小不一。因此，标准的预测推断方法并不适用，因为它们量化的是可观测输出的不确定性，而非潜在总体参数的不确定性。为解决这一问题，我们为这些潜在参数构建置信集，并据此推导出仿真-现实差距的稳健代理量。随后，我们估计分位数函数曲线……

    arXiv:2512.05024v4 Announce Type: replace-cross  Abstract: As generative AI models are increasingly used to simulate real-world systems, quantifying the ``sim-to-real'' gap is critical. For each input setting of interest -- which we call a \emph{scenario}, such as a survey question or operating condition -- the real and simulated systems are associated with unobserved latent population parameters, and their discrepancy varies across scenarios. A fundamental challenge is that, for any given scenario, this discrepancy cannot be observed directly, since both systems are accessible only through finite samples, often of heterogeneous sizes across scenarios. Standard predictive inference methods are therefore ill-suited, as they quantify uncertainty in observable outputs rather than latent population parameters. To address this, we construct confidence sets for these latent parameters and use them to derive a robust proxy for the sim-to-real discrepancy. We then estimate the quantile functio
    
[^265]: E0：基于Tweedie离散扩散的视觉-语言-动作模型表达性细粒度离散动作预测

    E0: Expressive Fine-Grained Discrete Action Prediction for Vision-Language-Action Models via Tweedie Discrete Diffusion

    [https://arxiv.org/abs/2511.21542](https://arxiv.org/abs/2511.21542)

    提出E0框架，通过Tweedie离散扩散将动作生成建模为对量化动作token的迭代去噪过程，实现了与token化推理天然契合的表达性细粒度离散动作预测，从而解决VLA模型泛化能力差和动作粗糙不稳定的问题。

    

    视觉-语言-动作模型通过整合视觉感知、语言理解和控制生成，为机器人操作提供了一个统一的框架。然而，现有的VLA系统仍然难以在多样化的任务、场景和相机视角之间实现泛化，并且常常产生粗糙或不稳定的动作。我们认为这些限制与VLA设置中动作的结构特性密切相关，包括动作分布固有的多峰性质、预训练VLM/VLA骨干网络基于token的符号推理，以及真实世界机器人控制所施加的有效有限分辨率。受这些特性的启发，我们提出了E0，一个Tweedie离散扩散框架，它将动作生成形式化为对量化动作token的迭代去噪过程。通过在离散动作空间中进行操作并采用有原则的扩散过程，E0自然地与基于token的推理相契合。

    arXiv:2511.21542v3 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. However, existing VLA systems still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We argue that these limitations are closely tied to the structural properties of actions in VLA settings, including the inherent multi-peaked nature of action distributions, the token-based symbolic reasoning of pretrained VLM/VLA backbones, and the effective finite resolution imposed by real-world robotic control. Motivated by these properties, we introduce E0, a tweedie discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. By operating in a discrete action space with a principled diffusion process, E0 naturally aligns with token-based reasoning
    
[^266]: 流形近似带来鲁棒的核对齐

    Manifold Approximation leads to Robust Kernel Alignment

    [https://arxiv.org/abs/2510.22953](https://arxiv.org/abs/2510.22953)

    提出流形近似核对齐（MKA），通过将流形局部邻域几何融入核对齐任务，去除启发式度加权、给出闭式对齐分数及马尔可夫概率解释，并证明其对邻域大小的稳定性，从而实现更鲁棒的表示比较。

    

    中心化核对齐（CKA）是比较表示、判断网络等价性以及开展神经科学研究的一种常用度量。然而，CKA 没有考虑底层的流形结构，并且依赖众多启发式方法，导致其在不同数据规模下表现不一致。在本工作中，我们提出了流形近似核对齐（MKA），将流形几何（即局部邻域）融入到对齐任务中。我们为 MKA 推导了一个理论框架。我们的构造去除了依赖于表示的度加权，得到了闭式的对齐分数，并将每一行赋予了作为局部马尔可夫转移分布的概率解释。我们进一步建立了关于邻域大小的核级别与分数级别的稳定性。我们在合成数据集和真实世界示例上进行了实证评估，以刻画 MKA 并将其与同类方法进行比较。

    arXiv:2510.22953v3 Announce Type: replace-cross  Abstract: Centered kernel alignment (CKA) is a popular metric for comparing representations, determining equivalence of networks, and conducting neuroscience research. However, CKA does not account for the underlying manifold and relies on numerous heuristics that cause it to behave differently at different scales of data. In this work, we propose Manifold-approximated Kernel Alignment (MKA), which incorporates manifold geometry (i.e., local neighborhood) into the alignment task. We derive a theoretical framework for MKA. Our construction removes representation-dependent degree weighting, yields a closed-form alignment score, and gives each row a probabilistic interpretation as a local Markov transition distribution. We further establish kernel- and score-level stability with respect to the neighborhood size. We perform empirical evaluations on synthetic datasets and real-world examples to characterize and compare MKA to its contemporari
    
[^267]: BigCodeArena：通过代码执行揭示代码生成中更可靠的人类偏好

    BigCodeArena: Unveiling More Reliable Human Preferences in Code Generation via Execution

    [https://arxiv.org/abs/2510.08697](https://arxiv.org/abs/2510.08697)

    该论文提出了 BigCodeArena，一个基于实时执行环境的开放式代码生成人类评估平台，通过让人类与 LLM 生成代码的执行过程和结果进行交互，从而揭示出更可靠的人类偏好。

    

    众包模型评估平台（如 Chatbot Arena）能够从人类视角实时评估模型响应的质量。在编程领域，人工检查大语言模型（LLM）生成内容的质量极具挑战性，因为这需要理解大段原始代码并仔细模拟代码执行。为此，我们推出了 BigCodeArena，一个开放的代码生成人类评估平台，其背后由全面且实时的执行环境支持。BigCodeArena 构建于 Chatbot Arena 之上，能够执行 LLM 生成的代码，并允许人类与执行过程和执行结果进行交互。我们在 10 个广泛使用的 LLM 上收集了超过 14,000 个以代码为中心的原始对话会话，涵盖 10 种语言和 8 类执行环境。在这些对话中，我们识别出超过 4,700 个包含成对人类偏好的多轮样本。

    arXiv:2510.08697v3 Announce Type: replace-cross  Abstract: Crowdsourced model evaluation platforms, such as Chatbot Arena, enable real-time evaluation from human perspectives to assess the quality of model responses. In the coding domain, manually examining the quality of LLM-generated content is extremely challenging, as it requires understanding long chunks of raw code and deliberately simulating code execution. To this end, we introduce BigCodeArena, an open human evaluation platform for code generation backed by a comprehensive and on-the-fly execution environment. Built on top of Chatbot Arena, BigCodeArena enables the execution of LLM-generated code and allows humans to interact with the execution process and outcomes. We collected over 14,000 raw code-centric conversation sessions across 10 widely used LLMs, spanning 10 languages and 8 types of execution environments. Among these conversations, we identified more than 4,700 multi-turn samples with pairwise human preferences. Fur
    
[^268]: 图构自身提示

    Graph Your Own Prompt

    [https://arxiv.org/abs/2509.23373](https://arxiv.org/abs/2509.23373)

    该论文提出图一致性正则化（GCR），通过在任意深度插入无参数的图一致性层，将特征相似性图与由模型自身预测导出的类感知掩码预测图对齐，实现自我提示式训练，使特征表示更具类一致性和语义意义。

    

    我们提出了图一致性正则化（Graph Consistency Regularization, GCR），这是一种新颖的框架，它将从模型预测中导出的关系图结构注入学习过程，以促进具有类感知能力且语义上有意义的特征表示。作为一种自我提示（self-prompting）的形式，GCR 使模型能够利用自身输出对其内部结构进行优化。尽管深度网络能够学习到丰富的表示，但这些表示往往捕捉到与模型预测语义相矛盾的、含噪的类间相似性。GCR 通过在任意深度引入无参数的图一致性层（Graph Consistency Layers, GCLs）来解决这一问题。每个 GCL 构建一个批级别的特征相似性图，并将其与全局的、类感知的掩码预测图对齐——该预测图通过用类内指标调制 softmax 预测相似性而得到。这种对齐强制特征层面的关系反映类一致的预测行为，从而充当一种语义层面的约束。（原文摘要在此处截断）

    arXiv:2509.23373v4 Announce Type: replace-cross  Abstract: We propose Graph Consistency Regularization (GCR), a novel framework that injects relational graph structures, derived from model predictions, into the learning process to promote class-aware, semantically meaningful feature representations. Functioning as a form of self-prompting, GCR enables the model to refine its internal structure using its own outputs. While deep networks learn rich representations, these often capture noisy inter-class similarities that contradict the model's predicted semantics. GCR addresses this issue by introducing parameter-free Graph Consistency Layers (GCLs) at arbitrary depths. Each GCL builds a batch-level feature similarity graph and aligns it with a global, class-aware masked prediction graph, derived by modulating softmax prediction similarities with intra-class indicators. This alignment enforces that feature-level relationships reflect class-consistent prediction behavior, acting as a seman
    
[^269]: HARDTESTGEN：面向大语言模型算法编程的高质量强化学习验证器生成流水线

    HARDTESTGEN: A High-Quality RL Verifier Generation Pipeline for LLM Algorithmic Coding

    [https://arxiv.org/abs/2505.24098](https://arxiv.org/abs/2505.24098)

    HARDTESTGEN提出了一种为算法编程问题自动合成高质量测试用例的流水线，并构建了包含2.66万道题的HARDTESTS数据集，其测试用例在验证LLM生成代码方面的精确度比现有测试提升11.22个百分点，有效支撑拒绝采样和强化学习等下游后训练。

    

    验证器为大语言模型（LLM）的强化学习提供重要的奖励信号。然而，开发或创建可靠的验证器极具挑战性，尤其是在代码生成任务中。一个伪装巧妙的错误解答程序可能只有通过精心人工编写的边缘测试用例才能被发现，而这类用例很难自动合成。为了解决这一问题，我们提出了HARDTESTGEN，一种为算法编程问题合成高质量测试用例的方法。我们构建了一个全面的算法编程数据集HARDTESTS，包含2.66万个问题及高质量的合成测试。与现有测试相比，HARDTESTGEN生成的测试在验证LLM生成代码方面表现出显著更高的准确率（精确度提升11.22个百分点，即在预测为正确的代码中实际正确代码所占的比例）。我们还证明，下游后训练——包括拒绝采样和强化学习（RL）——（摘要此处不完整）

    arXiv:2505.24098v2 Announce Type: cross  Abstract: Verifiers provide important reward signals for reinforcement learning of large language models (LLMs). However, it is challenging to develop or create reliable verifiers, especially for code generation tasks. A well-disguised wrong solution program may only be detected by carefully human-written edge cases that are difficult to synthesize automatically. To address this issue, we propose HARDTESTGEN, an approach to synthesize high-quality test cases for algorithmic coding problems. We curate a comprehensive algorithmic programming dataset HARDTESTS with 26.6k problems and high-quality synthetic tests. Compared with existing tests, HARDTESTGEN tests demonstrate significantly higher accuracy in verifying LLM-generated code (+11.22 percentage points in precision, the percentage of actually correct code within the predicted correct ones). We also show that downstream post-training -- including rejection sampling and reinforcement learning (
    
[^270]: 用于异构图异常检测的卡方小波图神经网络

    Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection

    [https://arxiv.org/abs/2505.18934](https://arxiv.org/abs/2505.18934)

    提出基于新型卡方小波滤波器的谱图神经网络框架ChiGAD，通过多图卡方滤波捕获跨元路径异常信号、交互式元图卷积在维度对齐中保留高频信息，有效解决了异构图异常检测中的语义捕获与类别不平衡难题。

    

    异构网络中的图异常检测由于节点和边的异质性而面临独特的挑战。现有的图神经网络（GNN）方法主要针对同构图异常检测，因此无法解决三个关键问题：（C1）跨不同元路径捕获异常信号和丰富语义；（C2）在异构信息网络（HIN）维度对齐中保留高频内容；（C3）在类别不平衡条件下从困难异常样本中有效学习。为克服这些问题，我们提出了ChiGAD，一种基于新型卡方滤波器的谱图神经网络框架，其灵感来源于小波在多个领域的有效性。具体而言，ChiGAD包含：（1）多图卡方滤波器，通过对每个元路径图应用专用卡方滤波器来捕获异常信息；（2）交互式元图卷积，在保留高频信息的同时对齐特征并融合异构消息。

    arXiv:2505.18934v3 Announce Type: replace-cross  Abstract: Graph Anomaly Detection (GAD) in heterogeneous networks presents unique challenges due to node and edge heterogeneity. Existing Graph Neural Network (GNN) methods primarily focus on homogeneous GAD and thus fail to address three key issues: (C1) Capturing abnormal signal and rich semantics across diverse meta-paths; (C2) Retaining high-frequency content in HIN dimension alignment; and (C3) Learning effectively from difficult anomaly samples with class imbalance. To overcome these, we propose ChiGAD, a spectral GNN framework based on a novel Chi-Square filter, inspired by the wavelet effectiveness in diverse domains. Specifically, ChiGAD consists of: (1) Multi-Graph Chi-Square Filter, which captures anomalous information via applying dedicated Chi-Square filters to each meta-path graph; (2) Interactive Meta-Graph Convolution, which aligns features while preserving high-frequency information and incorporates heterogeneous message
    
[^271]: AbFlowNet：通过扩散模型与GFlowNet融合优化抗体-抗原结合能

    AbFlowNet: Optimizing Antibody-Antigen Binding Energy via Diffusion-GFlowNet Fusion

    [https://arxiv.org/abs/2505.12358](https://arxiv.org/abs/2505.12358)

    AbFlowNet提出了一种将GFlowNet与扩散模型融合的新型生成框架，通过将每个扩散步骤建模为GFlowNet中的状态，在单一训练流程中直接引入能量信号，实现扩散损失与抗体-抗原结合能的联合优化，从而避免了依赖不可靠估计器的昂贵在线强化学习流程。

    

    互补决定区（CDRs）是抗体中促进与特定抗原结合的关键片段。目前用于CDR设计的计算方法仅采用重构损失，并未联合优化结合能——而结合能是衡量抗体效力的关键指标。现有的结合能优化通常通过计算代价高昂的在线强化学习（RL）流程来实现，且这些流程严重依赖不可靠的结合能估计器。本文提出了AbFlowNet，一种将GFlowNet与扩散模型相融合的新型生成框架。通过将每个扩散步骤视为GFlowNet框架中的一个状态，AbFlowNet将能量信号直接融入训练过程，从而联合优化标准扩散损失与结合能，在单一流程中统一了扩散优化与奖励优化。实验结果表明，AbFlowNet的表现优于基础扩散模型……

    arXiv:2505.12358v2 Announce Type: replace-cross  Abstract: Complementarity Determining Regions (CDRs) are critical segments of an antibody that facilitate binding to specific antigens. Current computational methods for CDR design utilize reconstruction losses and do not jointly optimize binding energy, a crucial metric for antibody efficacy. Rather, binding energy optimization is done through computationally expensive Online Reinforcement Learning (RL) pipelines rely heavily on unreliable binding energy estimators. In this paper, we propose AbFlowNet, a novel generative framework that integrates GFlowNet with Diffusion models. By framing each diffusion step as a state in the GFlowNet framework, AbFlowNet jointly optimizes standard diffusion losses and binding energy by directly incorporating energy signals into the training process, thereby unifying diffusion and reward optimization in a single procedure. Experimental results show that AbFlowNet outperforms the base diffusion model by 
    
[^272]: LEGAL-UQA：使用英乌尔都语平行基准测试面向乌尔都语使用者的宪法问答系统

    LEGAL-UQA: Testing Constitutional Question-Answering Systems for Urdu Speakers with a Parallel English-Urdu Benchmark

    [https://arxiv.org/abs/2410.13013](https://arxiv.org/abs/2410.13013)

    提出了LEGAL-UQA，一个包含619个问答对的英乌尔都语平行基准，用于测试面向乌尔都语使用者的巴基斯坦宪法问答系统，研究发现模型排名依赖相似度度量方式、词匹配检索优于嵌入方法，且自动评判模型与人工评分的一致性有限。

    

    法律信息助手可以帮助乌尔都语使用者了解其宪法权利，但构建可靠的系统需要适合该语言的测试数据和评估方法。我们提出了LEGAL-UQA，这是一个包含619个问答对的英乌尔都语平行基准，每个问答对均与巴基斯坦宪法的段落相关联。每个示例包含两种语言的问题、参考答案和源段落。我们分别评估两个组成部分：在提供源段落的情况下回答乌尔都语问题，以及检索生成每个问题的源段落。我们还将JEV（一种用作自动评判器的决策模型）与人工评分进行比较。研究结果表明，模型排名取决于答案相似度的度量方式，而简单的词匹配方法在检索源段落方面比所测试的嵌入方法更为有效。在评估的截止阈值处，JEV与人工评分的一致性低于基线方法。

    arXiv:2410.13013v2 Announce Type: replace-cross  Abstract: Legal information assistants could help Urdu speakers understand their constitutional rights, but building reliable systems requires test data and evaluation methods suited to the language. We present LEGAL-UQA, a parallel English-Urdu benchmark of 619 question-answer pairs linked to passages from Pakistan's Constitution. Each example includes the question, reference answer, and source passage in both languages. We evaluate two components separately: answering Urdu questions when the source passage is supplied, and retrieving the passage from which each question was generated. We also compare JEV, a decision model used as an automated judge, with human grades. The findings show that model rankings depend on how answer similarity is measured, while a simple word-matching method retrieves source passages more effectively than the embedding methods tested. At the evaluated cutoff, JEV agrees with human grades less often than a bas
    
[^273]: 基于训练后两层ReLU网络的回归中的良性过拟合

    Benign Overfitting for Regression with Trained Two-Layer ReLU Networks

    [https://arxiv.org/abs/2410.06191](https://arxiv.org/abs/2410.06191)

    本文首次证明了通过梯度流训练的有限宽度两层ReLU网络在最小二乘回归中既能实现泛化又会过拟合数据，从而在任意有界回归函数下首次建立了良性过拟合结果。

    

    我们研究了使用带ReLU激活函数的两层全连接神经网络、通过梯度流训练的最小二乘回归问题。我们的第一个结果是一个泛化结果，除了要求底层回归函数和噪声有界之外，不需要对它们做任何假设。我们在神经切向核机制下进行分析，通过将超额风险分解为估计误差和近似误差来建立泛化结果，并将梯度流视为一种隐式正则化器。这种在神经网络背景下的分解为梯度下降提供了一个新颖的视角，帮助我们避开一致收敛的陷阱。在本工作中，我们还证明了在相同设置下，训练后的网络会对数据产生过拟合。综合起来，这些结果首次建立了有限宽度ReLU网络在任意回归函数下的良性过拟合结论。

    arXiv:2410.06191v2 Announce Type: replace-cross  Abstract: We study the least-square regression problem with a two-layer fully-connected neural network, with ReLU activation function, trained by gradient flow. Our first result is a generalization result, that requires no assumptions on the underlying regression function or the noise other than that they are bounded. We operate in the neural tangent kernel regime, and our generalization result is developed via a decomposition of the excess risk into estimation and approximation errors, viewing gradient flow as an implicit regularizer. This decomposition in the context of neural networks is a novel perspective of gradient descent, and helps us avoid uniform convergence traps. In this work, we also establish that under the same setting, the trained network overfits to the data. Together, these results, establishes the first result on benign overfitting for finite-width ReLU networks for arbitrary regression functions.
    
[^274]: BigCodeBench：利用多样化函数调用与复杂指令对代码生成进行基准测试

    BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions

    [https://arxiv.org/abs/2406.15877](https://arxiv.org/abs/2406.15877)

    BigCodeBench是一个新的代码生成基准测试，通过要求模型调用多样化函数库并准确理解复杂指令，来评估大语言模型解决实际任务的能力。

    

    大语言模型（LLM）通过Python代码的最新进展极大地赋能了任务自动化，其任务范围涵盖从软件工程开发到通用推理。虽然现有基准测试表明LLM能够像人类开发者一样使用程序解决问题，但其大多数评估仅限于简短且独立的算法任务或单一函数调用。解决具有挑战性和实用性的任务，需要具备将多样化函数调用作为工具的能力，以高效实现数据分析和网页开发等功能。此外，使用多个工具解决任务还需要通过准确理解复杂指令来进行组合推理。同时满足这两个特性对LLM而言是一项巨大挑战。为了评估LLM通过程序解决具有挑战性和实用性任务的能力，我们推出了BigCodeBench这一基准测试……

    arXiv:2406.15877v5 Announce Type: replace-cross  Abstract: Task automation has been greatly empowered by the recent advances in Large Language Models (LLMs) via Python code, where the tasks ranging from software engineering development to general-purpose reasoning. While current benchmarks have shown that LLMs can solve tasks using programs like human developers, the majority of their evaluations are limited to short and self-contained algorithmic tasks or standalone function calls. Solving challenging and practical tasks requires the capability of utilizing diverse function calls as tools to efficiently implement functionalities like data analysis and web development. In addition, using multiple tools to solve a task needs compositional reasoning by accurately understanding complex instructions. Fulfilling both of these characteristics can pose a great challenge for LLMs.To assess how well LLMs can solve challenging and practical tasks via programs, we introduce BigCodeBench, a benchm
    
[^275]: Astraios：参数高效的指令微调代码大语言模型

    Astraios: Parameter-Efficient Instruction Tuning Code Large Language Models

    [https://arxiv.org/abs/2401.00788](https://arxiv.org/abs/2401.00788)

    该论文提出了Astraios套件——包含28个采用7种微调方法、4种规模（最大160亿参数）指令微调的代码大语言模型，系统研究发现全参数微调总体性能最佳，LoRA通常提供最有利的成本-性能权衡，且PEFT方法的效果随模型规模显著变化。

    

    大语言模型（LLM）全参数微调（FFT）的高昂成本催生了一系列参数高效微调（PEFT）方法。然而，目前尚不清楚哪些方法在不同模型规模下能够提供最佳的成本-性能权衡。我们提出了Astraios，这是一个包含28个经过指令微调的OctoCoder模型的套件，采用了7种微调方法和4种模型规模，最大参数量达160亿。通过对涵盖代码理解和代码生成任务的5个任务和8个不同数据集的调查研究，我们发现FFT通常在所有模型规模下都能带来最佳的下游性能，而PEFT方法的有效性则随模型规模的不同而有显著差异。LoRA通常在成本和性能之间提供最有利的权衡。进一步研究这些方法对模型鲁棒性和代码安全性的影响后发现，较大的模型往往表现出较低的鲁棒性。

    arXiv:2401.00788v2 Announce Type: replace-cross  Abstract: The high cost of full-parameter fine-tuning (FFT) of Large Language Models (LLMs) has led to a series of parameter-efficient fine-tuning (PEFT) methods. However, it remains unclear which methods provide the best cost-performance trade-off at different model scales. We introduce Astraios, a suite of 28 instruction-tuned OctoCoder models using 7 tuning methods and 4 model sizes up to 16 billion parameters. Through investigations across 5 tasks and 8 different datasets encompassing both code comprehension and code generation tasks, we find that FFT generally leads to the best downstream performance across all scales, and PEFT methods differ significantly in their efficacy based on the model scale. LoRA usually offers the most favorable trade-off between cost and performance. Further investigation into the effects of these methods on both model robustness and code security reveals that larger models tend to demonstrate reduced robu
    
[^276]: 假新闻检测器对大型语言模型生成的文本存在偏见

    Fake News Detectors are Biased against Texts Generated by Large Language Models. (arXiv:2309.08674v1 [cs.CL])

    [http://arxiv.org/abs/2309.08674](http://arxiv.org/abs/2309.08674)

    假新闻检测器倾向于将大型语言模型生成的内容标记为假新闻，而将人工编写的假新闻误分类为真实，我们提出了一种通过敌对训练和LLM改写的真实新闻等方法来解决这个问题，并取得了显著的改进。

    

    假新闻的传播已经成为一个重要挑战，损害了信任并对社会构成威胁。在大型语言模型（LLM）的时代，生成可信的假内容的能力加剧了这些担忧。在本研究中，我们提出了一种新的范式来评估在涉及人工编写和LLM生成的错误信息的情况下的假新闻检测器。有趣的是，我们的发现揭示了许多现有检测器存在显著的偏见：它们更容易将LLM生成的内容标记为假新闻，同时常常将人工编写的假新闻误分类为真实。这种意外的偏见似乎源自LLM输出固有的不同语言模式。为了解决这个问题，我们引入了一种使用LLM改写的真实新闻进行敌对训练的缓解策略。结果模型在人工和LLM生成的新闻的检测准确性方面均有显著提升。为了进一步促进这个领域的研究，我们发布了两个全面的...

    The spread of fake news has emerged as a critical challenge, undermining trust and posing threats to society. In the era of Large Language Models (LLMs), the capability to generate believable fake content has intensified these concerns. In this study, we present a novel paradigm to evaluate fake news detectors in scenarios involving both human-written and LLM-generated misinformation. Intriguingly, our findings reveal a significant bias in many existing detectors: they are more prone to flagging LLM-generated content as fake news while often misclassifying human-written fake news as genuine. This unexpected bias appears to arise from distinct linguistic patterns inherent to LLM outputs. To address this, we introduce a mitigation strategy that leverages adversarial training with LLM-paraphrased genuine news. The resulting model yielded marked improvements in detection accuracy for both human and LLM-generated news. To further catalyze research in this domain, we release two comprehensiv
    

