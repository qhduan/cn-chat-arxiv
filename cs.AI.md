# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning](https://arxiv.org/abs/2610.32978) | SceneBloom是一个结合视觉布局采样与契约引导3D编译的智能体框架，利用扩散先验在功能区域内生成家具布局并通过几何适配与姿态校正实现物理有效的室内场景，在五种房间类型上取得了最高的视觉质量与物理稳定性，且是唯一无网格碰撞与越界物体的方法。 |
| [^2] | [RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms](https://arxiv.org/abs/2610.32971) | 本文提出RobustRLlib，一个集成了17种鲁棒在线、离线与安全RL算法的统一库与基准测试平台，并通过273种扰动配置的系统评估绘制了当前鲁棒RL算法的能力图谱，揭示了其有效范围与鲁棒性差距。 |
| [^3] | [DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection](https://arxiv.org/abs/2610.32969) | DexPort提出了一种面向高自由度灵巧手的便携式手持数据采集接口，通过采集与部署共用相同手部硬件来避免实体差异，并借助一小组可选择的技能来协调手指运动，从而大幅降低高质量灵巧操作演示数据的采集成本。 |
| [^4] | [Beyond Discrete Tokens: Preserving Numerical Continuity in Medical Vision-Language Models](https://arxiv.org/abs/2610.32948) | 提出 HAD-VLM，一种在共享骨干中融合自回归文本生成与扩散式连续测量预测的混合医学视觉-语言模型，通过扩散过程直接预测卡尺端点坐标与测量长度等连续数值，解决了传统离散标记预测与连续测量目标之间的根本不匹配问题。 |
| [^5] | [Monetizing Intelligence: Advertising and Subscription Design on LLM Platforms](https://arxiv.org/abs/2610.32945) | 该论文研究大语言模型平台的广告与订阅联合定价设计，发现广告既能资助更广泛的服务接入，但也会导致用户效用两极分化，而设置适度的广告上限可以最大化消费者剩余。 |
| [^6] | [Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model](https://arxiv.org/abs/2610.32933) | 该论文提出Yalda Embedding，一个基于Qwen3-Embedding-0.6B适配、使用超百万查询-文档对训练的5.958亿参数波斯语检索模型，在FaMTEB的八个波斯语检索任务上超越所有同规模0.6B模型，并接近4B规模模型的性能。 |
| [^7] | [SemantEq: Faithful Bidirectional Equivalence of Formal Statements](https://arxiv.org/abs/2610.32919) | 提出 SemantEq，一种更有原则的形式化命题等价检验方法，要求命题间只能通过一系列允许的逻辑变换步骤相互转化，从而避免因证明器过强导致逻辑等价检验被平凡通过的问题。 |
| [^8] | [Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations](https://arxiv.org/abs/2610.32891) | 该研究通过对11个开源模型在EffiBench全部1000个问题上的硬件级能耗实测，挑战了“代码跑得更快就一定更省电”的假设，发现LLM迭代优化代码的推理能耗（0.79–8.68 MJ，相当于250–2800次GPT-4.1查询）未必能被后续执行时间的节省所抵消。 |
| [^9] | [Closing the Modality Gap via Simplex-Constrained Representations](https://arxiv.org/abs/2610.32890) | 该论文提出将多模态对比模型的嵌入域从单位球面改为概率单纯形，利用其非负、单位质量的结构约束作为跨模态校准，在五个检索基准上将质心差距减少97%至99%，实现真正的语义对齐。 |
| [^10] | [NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding](https://arxiv.org/abs/2610.32881) | NgramRL利用同一提示词的近期回滚历史构建n-gram起草器进行投机解码，无需辅助模型或GPU内存开销，即可显著加速强化学习后训练中占训练时间70%以上的回滚生成过程。 |
| [^11] | [Advancing SVD-based LLM Compression via Layer-Wise Error Model Search](https://arxiv.org/abs/2610.32878) | 提出LEMS逐层误差建模搜索和KFAC-SVD方法，通过融合局部与全局层重要性的误差代理实现更精确的全局秩分配，并利用逐token统计量提升SVD分解质量，从而有效改进大语言模型的低秩压缩效果。 |
| [^12] | [Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations](https://arxiv.org/abs/2610.32874) | 该研究首次系统揭示了LLM智能体在观点模拟中违反互补一致性——对同一命题及其否定所报告的概率之和超过1（如Qwen3-4B平均达1.18），且该偏差远大于扰动噪声两到十倍，意味着此类模拟结果可能存在系统性扭曲。 |
| [^13] | [From Deliberately Constructed Systemic Representations of the World of Activity to Dialogue: A Five-Level Communication Architecture for Strong Language-Based ASI](https://arxiv.org/abs/2610.32869) | 论文提出五级消息形成与理解架构，主张语言模型的上下文无法呈现话语形成的基础——这些基础由习得的图式和刻意构建的活动世界系统表征预先建立，并据此为强语言型ASI设计了从系统表征到文本的通信架构。 |
| [^14] | [Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery](https://arxiv.org/abs/2610.32866) | 本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。 |
| [^15] | [PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets](https://arxiv.org/abs/2610.32864) | PoKER是一种利用机器学习整合目标随机行为与导弹脱靶距离、从而更准确预测超视距空战中空空导弹杀伤概率的新型概率模型，有望成为飞行员改善作战决策的重要工具。 |
| [^16] | [A Foundation Model for Policy In-Context Longitudinal Causal Effect Estimation](https://arxiv.org/abs/2610.32863) | 本文提出了首个支持动态治疗策略的纵向因果效应估计基础模型 PICLE-PFN，通过在合成数据上预训练并以上下文学习方式表示目标策略，将因果效应估计的计算摊销到不同数据集、策略和预测范围上，避免了部署时耗时的模型拟合。 |
| [^17] | [Reimagining Enterprise RAG for Multi-Agent Systems: A Knowledge Layer That Organizes Before It Retrieves](https://arxiv.org/abs/2610.32851) | 该论文提出一种“先组织后检索”的企业RAG知识层，通过七智能体逆向工程流水线从代码仓库中恢复系统行为逻辑，并解决跨服务同名实体操作语义不同导致扁平向量检索混淆的问题。 |
| [^18] | [HarnessMM: Recursive Self-Improvement for Harness Token Efficiency](https://arxiv.org/abs/2610.32813) | HarnessMM 通过递归自我改进优化 harness 配置，并借助受保护决策复用让智能体在工具反馈后直接复用已确定的决策来替代重复推理，从而在 actor 权重固定不变的情况下提升 harness 的 token 效率。 |
| [^19] | [Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules](https://arxiv.org/abs/2610.32801) | 本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。 |
| [^20] | [Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents](https://arxiv.org/abs/2610.32781) | 该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。 |
| [^21] | [AI Alignment for Digital Democracy: Fairer Voters Representation against LLM Biases](https://arxiv.org/abs/2610.32780) | 该研究揭示了LLM在模拟选民时存在人口统计学偏差，并提出通过AI对齐策略（如从选民画像中移除性别、教育等敏感特征）可将最大群体误差降低22-42%，从而在不牺牲整体准确性的前提下实现更公平的选民代表性。 |
| [^22] | [On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models](https://arxiv.org/abs/2610.32770) | 该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。 |
| [^23] | [Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks](https://arxiv.org/abs/2610.32765) | 该论文针对车联网联邦学习提出了一种特征空间干净标签投毒攻击与特征一致性重标记防御方法，并通过博弈论框架刻画攻防均衡策略，在不损害正常检测准确率的情况下将攻击成功率维持在低水平。 |
| [^24] | [Semantic Representation for Agentic Self-Play in Automated Laboratories](https://arxiv.org/abs/2610.32755) | 该论文提出VERID——实验室工作单元的持久化高保真语义数字孪生基础设施，通过可查询自省、时间压缩仿真与全面追踪持久化等核心服务，弥合AI智能体高层推理与湿实验室复杂易错物理现实之间的差距。 |
| [^25] | [ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents](https://arxiv.org/abs/2610.32752) | ActionRAG提出了一种图记忆架构，通过谨慎的分层实体消解索引与查询时基于局部路径结构的图扩散回溯评估，帮助LLM智能体从长期交互历史中恢复精确且相互一致的关系型证据。 |
| [^26] | [Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection](https://arxiv.org/abs/2610.32742) | AWARE是一个仅依靠外部摄像头监控即可实现机器人异常检测的世界模型框架，通过系统动力学估计与状态预测，在真实起重机数据上异常分类性能比最强基线提升26%，并能解码动力学参数定位故障来源。 |
| [^27] | [Agentic Hazard Mapper: Emulating Expert Map-Making with AI Agents for Risk Communication](https://arxiv.org/abs/2610.32738) | 本文提出Agentic Hazard Mapper多智能体系统，通过模拟专业制图师的工作流程和设计权衡来自动生成灾害地图，其生成的地图在空间理解、疏散规划效果和工作负荷方面均优于现有市政地图。 |
| [^28] | [Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors](https://arxiv.org/abs/2610.32729) | 本文提出“内部差异”度量来统一刻画对比学习中的各种理论假设，证明了基于该度量的下游监督任务泛化界，并提出松弛k均值损失作为减小内部差异的近似方法。 |
| [^29] | [PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences](https://arxiv.org/abs/2610.32727) | 该论文提出PlayerEmbed框架，利用深度度量学习和混合CNN-LSTM编码器从原始玩家输入序列中学习紧凑的行为指纹嵌入，在玩家识别任务中对已见和未见玩家分别达到0.950和0.879的平衡准确率，且增加训练玩家数量可显著提升对未见玩家的泛化能力。 |
| [^30] | [Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications](https://arxiv.org/abs/2610.32725) | 本文提出一种代理式、本体引导的知识图谱构建框架，利用大语言模型从气候科学文献中抽取带文本证据的三元组，并通过多阶段验证确保语义一致性与可追溯性，抽取质量接近人类标注者水平。 |
| [^31] | [SP-Guide: Enhancing Aerial Vision-Language Navigation with Spatial Priors](https://arxiv.org/abs/2610.32697) | SP-Guide通过空间先验跨模态编码器等模块将物理空间先验引入航空视觉语言导航，显著提升了无人机在CityNav基准上的成功率，尤其在未见环境中增益最大。 |
| [^32] | [Conditional Neural Processes for Self-Supervised Goal Discovery in Action Imitation](https://arxiv.org/abs/2610.32688) | 在条件神经过程的隐藏空间中加入注意力机制，可使其在观察动作序列的早期就涌现出鲁棒的目标表示，从而实现动作模仿中的自监督目标发现，为机器人的自监督通用学习提供了有前景的模型。 |
| [^33] | [Retiming Robot Demonstrations with a Learned Speed Prior](https://arxiv.org/abs/2610.32676) | 提出从大规模演示数据中学习一个速度先验，为任意任务预测执行速度分布并生成一致的时钟来重定时演示，从而消除操作者个人节奏对策略训练的干扰。 |
| [^34] | [Do Language Models Trust Their Own Justifications? A Study on Functional Consistency](https://arxiv.org/abs/2610.32665) | 该研究提出基于“功能自一致性”原则的实验框架，通过分离、移除或语义反转模型所标记的关键词元，检验语言模型生成的解释是否与其真实预测行为在功能上保持一致。 |
| [^35] | [SemGeo-Gen: Unsupervised Generation of Approximate Cross-Instance Semantic-Geometric Correspondences](https://arxiv.org/abs/2610.32663) | SemGeo-Gen提出了一种无监督方法，可从弱对齐的3D物体集合中自动生成语义有意义、几何一致的近似跨实例对应关系，无需人工标注即可提供可扩展的3D-3D、3D-2D和2D-2D监督信号。 |
| [^36] | [WiMoT: Replay-Free Continual Wi-Fi Sensing via Moment Transport](https://arxiv.org/abs/2610.32660) | WiMoT通过存储并传输各类别特征的矩统计量（均值和协方差）而非原始样本，实现了无需重放旧数据的持续Wi-Fi感知，在保护隐私的同时兼顾了学习新域的可塑性与保留旧知识的稳定性。 |
| [^37] | [InferSlice: A Reference Architecture for SLO-driven LLM Inference over a Telco Wide-Area Network](https://arxiv.org/abs/2610.32638) | InferSlice是一个部署在电信广域网上的SLO驱动LLM推理参考架构，通过集成网络切片、GPU分区、弹性服务和可观测性，解决了集中式云部署难以满足的主权、合规、延迟和资源本地化需求。 |
| [^38] | [Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry](https://arxiv.org/abs/2610.32636) | 本研究通过对一家工业机器人编程中小企业的案例研究，发现AI智能体并非即插即用的解决方案——除了生成高度重复代码的任务外未能提升工程师效率，其使用效果随工程师的AI使用经验和智能体的领域适配而逐步改善，且AI使用经验比机器人工程经验对成功更为关键。 |
| [^39] | [EditJev: System One Routing for Image Editing Models](https://arxiv.org/abs/2610.32635) | 提出EditJev——一个“系统一”式路由模型，通过在视觉-语言骨干上添加轻量级偏好头，在单次前向传播中直接预测各候选图像编辑模型的样本级偏好分数，从而根据输入动态选择最优编辑器，兼顾效果与推理效率。 |
| [^40] | [Interpretable Singing Emotion Recognition Using Acoustic and Vocal Technique Features](https://arxiv.org/abs/2610.32629) | 该论文提出一个使用歌唱专用声学与声乐技巧特征的可解释歌唱情感识别框架，借助SHAP和GRAD-CAM揭示情感表达与声乐技巧执行之间的关联。 |
| [^41] | [LGMT: A Tool for Testing the Logical Reasoning Reliability of LLMs](https://arxiv.org/abs/2610.32627) | 本文提出开源工具LGMT，利用基于一阶逻辑的蜕变关系生成逻辑等价测试对，在无需真实标签的情况下有效检测大语言模型在等价问题重述中的推理不一致性。 |
| [^42] | [What Transfers Across Protein Pockets? Experience Reuse in Molecular Optimization for Agentic Structure-Based Drug Design](https://arxiv.org/abs/2610.32620) | 该研究提出以编辑位点接触、动作和轮次为键的经验库，证明了分子优化中的探测经验可以跨蛋白质口袋迁移复用，在从未见过的口袋上仍能清晰区分改善性与恶化性的编辑。 |
| [^43] | [LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling](https://arxiv.org/abs/2610.32604) | 该论文提出竞争阈值理论来揭示连续扩散语言模型中词元超越的动态机制，并据此提出竞争感知的语义调度方法，以解决轨迹与时间不对齐导致的时间压缩问题。 |
| [^44] | [Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance](https://arxiv.org/abs/2610.32555) | 提出基于真实MCP生态故障数据校准的故障注入中间件MCP-Injector，实验证明扩大LLM模型规模并不能提升智能体应对工具故障的容错能力。 |
| [^45] | [Beyond Channel Data: Configuration-Enriched Radio Foundation Models](https://arxiv.org/abs/2610.32542) | 本文提出在无线基础模型预训练中融入信道配置信息，实验证明所得的配置增强嵌入可作为有效归纳先验，在标签稀缺下的多种下游任务（LoS/NLoS分类、TA估计和波束选择）中一致优于基线模型。 |
| [^46] | [StarRTT: An End-to-End Real-Time Framework for Blind Zone-Free RTT Prediction in Starlink LEO Satellite Networks](https://arxiv.org/abs/2610.32540) | 提出StarRTT端到端实时框架，将每秒服务卫星识别与RT-LSTM预测模型相结合，消除了星链低轨卫星网络中RTT实时预测的识别盲区。 |
| [^47] | [IGG: A Benchmark for Interactive GUI Grounding under Visibility Constraints](https://arxiv.org/abs/2610.32536) | 该论文提出了IGG基准，首个要求智能体先通过交互使被隐藏的GUI目标变为可见再进行点击的评测基准，揭示了人类（94.6%）与最强模型（GPT-5.5达79.4%，最强开源模型仅31.6%）在可见性约束下GUI定位能力的显著差距。 |
| [^48] | [Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations](https://arxiv.org/abs/2610.32515) | 该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。 |
| [^49] | [MusicTwins: Training Audio-Language Models to Listen like Their Persona](https://arxiv.org/abs/2610.32495) | 该论文提出MusicTwins框架，通过从人类文字中构建人物画像表征、结合个人记忆，并采用监督微调与强化学习对Audio Flamingo Next进行后训练，使音频-语言模型能够像特定个人一样对文本和音乐刺激做出个性化回应，显著提升了人物保真度。 |
| [^50] | [Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)](https://arxiv.org/abs/2610.32486) | SAAF框架通过双信号自适应采样器（增量信号捕捉突发事件、漂移信号检测静态与渐进异常）以及阈值门控的级联架构，在无需训练数据的情况下实现了高效且全面的零样本视频异常检测。 |
| [^51] | [When Agents Fail: Model or Harness--Who's to Blame?](https://arxiv.org/abs/2610.32483) | 本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。 |
| [^52] | [CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models](https://arxiv.org/abs/2610.32481) | 提出首个面向多模态情感识别的智能体技能框架CASE，通过维护针对特定情感对的自演化EmotionSkills库，在冻结模型参数的前提下以低成本、可复用、可迁移的方式纠正语义相近情感间的系统性混淆。 |
| [^53] | [CauProphet: Teaching Agents to Develop Causal Cognition like Humans](https://arxiv.org/abs/2610.32477) | 该论文提出CauProphet智能体，通过构建结构化的“因果信念”并在“因果认知”中持续更新因果知识，使大语言模型智能体能够像人类一样习得因果理解，并将其迁移应用于未来的决策任务。 |
| [^54] | [Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings](https://arxiv.org/abs/2610.32474) | 提出PILOT框架，将自生成推理建模为潜在变量，通过共享多模态大语言模型的先验与后验两种推理模式进行特权推理探索，从而发现更具检索效用的推理轨迹，突破传统仅以输入为条件的事后奖励优化的局限，提升多模态嵌入的质量。 |
| [^55] | [ALLMTS: Automated LLM-based Medical Triage System](https://arxiv.org/abs/2610.32469) | 提出了ALLMTS系统，不让大语言模型直接分诊，而是让它执行护士使用的商业化远程医疗分诊指南，根据第一个肯定答案确定性地得出分诊处置结果，从而提高了电话分诊的可靠性与可审计性。 |
| [^56] | [Beyond Copying: Lifelong Smart Contract Security Task Construction via Protocol Mutation](https://arxiv.org/abs/2610.32453) | 提出CONSTRAINTDIAL框架，通过协议变异自动修改智能合约的调用与输入，使原始攻击失效的同时保持可利用性和正常功能，从而实现安全任务的终身构建，从203个公开案例生成了3,045个任务输入，避免评估奖励记忆或复制的答案。 |
| [^57] | [AutoMIL: Automated Multiple Instance Learning for Whole-Slide Image Classification](https://arxiv.org/abs/2610.32452) | AutoMIL是一个开源的自动机器学习框架，通过自动化预处理、训练和评估等关键流程步骤，在无需手动配置的情况下实现了与专用流程相当性能的全切片图像分类基线。 |
| [^58] | [DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare](https://arxiv.org/abs/2610.32451) | 提出DIGOM框架，利用解耦视图的稀疏多路图联合建模患者内跨模态、跨患者同模态和跨患者跨模态三种交互关系，以提升多模态医疗数据的临床结果预测性能。 |
| [^59] | [POSEIDON: Scaling Training Environments for Scientific Agents from Code Repositories](https://arxiv.org/abs/2610.32448) | POSEIDON通过将科学工作流分解为复现、子流程、补全和修正四类任务并配合递归改进循环，实现了从科学代码仓库自动规模化生成可执行可验证的训练环境，构建了包含97,659个环境的Pearl数据集，环境数量约为先前方法的10倍。 |
| [^60] | [Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation](https://arxiv.org/abs/2610.32443) | 提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。 |
| [^61] | [When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning](https://arxiv.org/abs/2610.32431) | 提出MART-GRPO，通过成功概率与策略得分鞅刻画失败反馈的理想停止条件，并借助视觉-语言模型进行事件引导截断，避免GRPO将轨迹级惩罚错误地施加到失败轨迹中有用的恢复动作上。 |
| [^62] | [Async-RBench: Benchmarking Asynchronous Task Execution in LLM Agents](https://arxiv.org/abs/2610.32429) | 该论文提出了Async-RBench，一个包含200个可执行任务的基准，通过事件响应契约（ERC）和动态重规划得分（DRS）独立评估LLM智能体在异步任务执行中处理动态变化结果的能力。 |
| [^63] | [Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images](https://arxiv.org/abs/2610.32423) | 提出了一种无需训练的推理时自我纠正框架，让单个视觉语言模型迭代地扮演生成者、批评者和改进者的角色，形成闭环自我反馈回路，无需额外标注数据或模型更新即可提升文档图像中表格视觉问答的性能。 |
| [^64] | [LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents](https://arxiv.org/abs/2610.32412) | LeafSkill提出一个模型参数之外的外部技能复用层，通过语义检索相关定位经验并将动作模板绑定到当前对象，用轻量级1.7B选择器替代大模型完成跨任务重复出现的叶级动作定位，从而提升递归LLM智能体的效率。 |
| [^65] | [PASTA: Prompt-Aware Annealed Diffusion Sampling](https://arxiv.org/abs/2610.32390) | PASTA提出了一种免训练的提示感知退火扩散采样框架，通过在共享潜变量轨迹上对允许对象的区域选择性施加语义引导，在文本到图像生成中严格保留指定区域，同时保持视觉连贯性。 |
| [^66] | [DealsFlow: An Evidence-Gated Agentic Harness for Venture Capital Deal Triage](https://arxiv.org/abs/2610.32380) | DealsFlow是一个部署于英国早期基金的人机协同智能体框架，通过证据门控机制（阶段感知评分上限、怀疑者调用和预取模式）防止LLM投资评分虚高，在142笔三年以上的投资中评分与实际结果的Spearman相关系数达到0.45。 |
| [^67] | [Adaptive Control of Scientific Instruments with Nonstationary Responses through Multimodal Agents](https://arxiv.org/abs/2610.32370) | 本文提出一种具有物理基础、组件可独立审计的多模态智能体，通过确定性物理量提取与双层记忆机制，实现对响应非平稳的科学仪器（如磁约束聚变装置NBI离子源）的自适应逐炮调谐控制。 |
| [^68] | [FOCUS: A Lightweight Mamba-2 Model for EEG-Based Cognitive Workload Classification](https://arxiv.org/abs/2610.32361) | 提出了一种仅有4万参数的轻量级Mamba-2模型FOCUS，利用固定FFT五频带分解编码多通道脑电信号，在数据受限条件下实现了高效准确的认知负荷二分类。 |
| [^69] | [TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection](https://arxiv.org/abs/2610.32337) | TENET将无监督多元时间序列异常检测重新表述为当前点系统状态兼容性估计，通过以当前点为中心的时间编码和上下文自适应混合构建潜在状态表示，从而克服了观测空间差异作为异常指标不可靠的问题。 |
| [^70] | [Data-Efficient Backdoor Attacks on Code Models via Trigger-Neutralized Counterfactual Scoring](https://arxiv.org/abs/2610.32334) | 该论文提出触发器中和反事实评分（TNCS）方法，通过比较触发输入与触发器中和反事实样本上的模型输出差异来精确筛选投毒样本，在极低投毒率下即可对代码模型实现高成功率的后门攻击。 |
| [^71] | [4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents](https://arxiv.org/abs/2610.32332) | 该论文提出了包含约900小时视频和5000多个人工审核问题的 LongVideo 数据集及 4DMemoryBench 基准，并开发了集成四种互补技能、22个以上专用工具模型和检索增强生成系统的 4DMemory Harness，用于评估和提升基础模型的长时程时空记忆推理能力。 |
| [^72] | [TGR-EvoDAG: A DAG-based Multi-Agent Code Evolution Framework for Generative Recommendation on Tencent](https://arxiv.org/abs/2610.32326) | 该论文提出TGR-EvoDAG，一个针对腾讯生成式推荐场景的基于DAG的多智能体代码演化框架，通过将代码演化建模为有向无环图而非线性流程，解决了现有方法在模块级联流水线下的成本盲视与短视评估问题，实现更高效的自动化算法发现。 |
| [^73] | [ChoreoStudio: Planning What, When, and Who in Music-Driven Group Choreography](https://arxiv.org/abs/2610.32325) | ChoreoStudio通过将音乐解耦为节奏与结构特征，并让全模态模型扮演编剧和导演进行显式编舞规划，从而自动决定音乐驱动群舞中队形变化的内容、时机与执行者。 |
| [^74] | [From Visual Affinity to Learning Credit in Multimodal Reinforcement Learning](https://arxiv.org/abs/2610.32324) | 提出结果条件化视觉信用分配框架，通过将视觉关联度量与标记级信用分配分离，解决了多模态强化学习中均匀优势分配忽视视觉观测独特作用的问题。 |
| [^75] | [Beyond Gain: Measuring Skill Internalization for Agent Skill Retirement](https://arxiv.org/abs/2610.32322) | 该论文提出通过“隐去技能后评测+跨模型交叉验证高分”的方法可靠度量LLM智能体的技能内化程度，弥补仅凭增益判定会误删“未覆盖”技能的缺陷，从而实现准确的智能体技能退役。 |
| [^76] | [Solvent: Long-Horizon Reasoning for Multi-Step Retrosynthesis](https://arxiv.org/abs/2610.32308) | Solvent是一个8B参数的大语言模型，通过两轮后训练课程与强化学习将长程推理能力引入多步逆合成任务，学会交错执行迭代合成规划、外部可得性验证和自适应路线修正，性能提升2.1倍后在四个基准上取得61.86%的成绩。 |
| [^77] | [DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks](https://arxiv.org/abs/2610.32305) | 该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。 |
| [^78] | [Lifelong Context Adaptation through Bidirectional Memory-Skill Learning](https://arxiv.org/abs/2610.32295) | 提出MemCryst框架，通过记忆与技能的双向学习，让冻结的大语言模型智能体在有限上下文窗口内实现终身上下文适应——记忆中的重复情节支持技能创建与修订，技能引导的交互又产生新记忆，形成持续学习的闭环。 |
| [^79] | [Complementing DINO Features with Image Structure for Part Discovery](https://arxiv.org/abs/2610.32293) | 该论文揭示了基于DINO特征的部件发现边界模糊的根源在于ViT刚性补丁网格导致跨边界补丁混合了多个部件的像素，并提出用图像结构（来自边界检测器的边界信息）补充DINO特征，从而在无需重新训练的情况下实现更精确的部件发现。 |
| [^80] | [Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution](https://arxiv.org/abs/2610.32285) | 提出TraceTrial-Bench基准，通过七大可靠性组件、95个专家级机制导向任务、轨迹追踪模块和双层评估框架，首次系统性地实现对AI智能体框架的可靠性与失败归因评估。 |
| [^81] | ["Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs](https://arxiv.org/abs/2610.32281) | 研究发现多语言大模型在各语言间共享可因果干预的否定表征，从一种语言提取的否定激活方向可以在其他语言中诱导或移除否定，且否定由稀疏特征表示。 |
| [^82] | [Tokamak-SI: Interpreting Tokamak Discharges with Labeled Catalogs](https://arxiv.org/abs/2610.32264) | Tokamak-SI将DIII-D托卡马克多种物理现象的标记和标记方法整合为统一、可扩展、时间对齐的数据集，基于它的模型在AE、ELM和TM检测等任务上达到或超越早期探测器。 |
| [^83] | [Beyond Pattern Matching in 3D MLLMs: Part-Level Functional Reasoning for Generalizable Recognition](https://arxiv.org/abs/2610.32262) | 针对3D多模态大语言模型依赖模式匹配而无法区分几何相似物体的问题，本文提出PFIR方法，通过部件级功能推理构建“几何→功能→身份”的思维链数据，并以双目标GRPO渐进训练，显著提升了对最模糊样本的识别泛化能力。 |
| [^84] | [Heterogeneous Space Data Centers: A Key Paradigm for Future Non-Terrestrial and In-Orbit Services](https://arxiv.org/abs/2610.32260) | 本文提出由大小异构节点构成的空间数据中心新范式，定义了其用例、节点部署与空-空/空-地互连拓扑，并针对晨昏近地轨道配置分析太阳能供给、可见窗口与延迟，评估了服务延迟与可靠性。 |
| [^85] | [MedCaseAgent: From Multimodal Clinical Evidence to Rare Disease Case Reports](https://arxiv.org/abs/2610.32258) | 提出MedCaseAgent端到端智能体推理框架，能从非结构化临床笔记自动生成发表级质量的罕见病病例报告，并同步发布大规模多模态语料库MedCase-150K和时间留出评估基准MedCase-Bench。 |
| [^86] | [Representation Learning via Composite Subspace Flow Matching](https://arxiv.org/abs/2610.32256) | 提出复合子空间流匹配（CSFM），通过在光谱、颜色和光照子空间上对确定性退化课程进行方差最小化求逆，并以硬性时间风格截断约束轨迹方差，使编码器忽略瞬态风格、编码持久内容，从而弥合了联合嵌入与生成式自监督范式之间的权衡。 |
| [^87] | [Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection](https://arxiv.org/abs/2610.32255) | 提出绝对值缩放修复与抗相干性正则化，解决了SCALE方法因负激活正负抵消而在ViT等骨干网络上性能退化的问题，无需训练数据即可提升分布外检测性能。 |
| [^88] | [RSIHub: A Composable Framework for Agent-Evolution Methods](https://arxiv.org/abs/2610.32253) | RSIHub是一个开源可组合框架，通过共享基础设施统一支持AHE、Hyperagents、A-Evolve和GEPA等智能体演化方法的构建、比较与扩展，无需修改框架核心即可实现，最高可带来13.5个百分点的pass@1提升。 |
| [^89] | [Easy-3Dfy: Object-Level Geometric Composition for Single-Image 3D Scene Construction](https://arxiv.org/abs/2610.32250) | Easy-3Dfy提出一种组合式工作流程，将专用物体生成与3D基础模型的场景布局估计解耦，并通过基于RPF的对齐器在物体网格与部分场景点图之间建立几何对应，无需完整场景级监督即可实现高质量的单图像3D场景构建，在多个几何指标上超越SAM-3Dfy。 |
| [^90] | [Minimum Description Length Unsupervisedly Separates Law-Grade from Statistical-Grade Knowledge in a Self-Improving Science Agent - and Reveals 86% of Its Own Discoveries Are Statistical](https://arxiv.org/abs/2610.32245) | 最小描述长度压缩比可作为一种无监督的认知评级器，从自我改进科学智能体的发现中精准识别出真正的物理定律，同时揭示其86%的发现仅为统计级知识而非定律级知识。 |
| [^91] | [Spatial Heuristic Strategy based Diffusion Reinforcement Learning for Online 3D Bin Packing](https://arxiv.org/abs/2610.32241) | 本文提出一种融合空间启发式策略的扩散强化学习方法，将经典装箱启发式规则与扩散强化学习相结合，以增强在线三维装箱中的空间拓扑感知能力和布局优化效果。 |
| [^92] | [Financial Context-Rich Strategy for Actor-Critic based Agents: A Deep Reinforcement Learning Approach for Stock Trading](https://arxiv.org/abs/2610.32237) | 本文提出了一种面向基于Actor-Critic智能体的金融情境丰富策略，利用深度强化学习方法将社会情绪、政治不稳定等外部因素纳入股票交易策略，以捕捉传统价格指标无法捕捉的市场动态。 |
| [^93] | [Training an MoE LLM in MXFP4: How We Closed the Gap to BF16](https://arxiv.org/abs/2610.32229) | 该论文提出ALTO开源训练方案，通过结合混合1D/2D块量化、随机Hadamard变换、随机舍入与简化的权重去振荡技术，使200亿参数MoE模型GPT-OSS-20B在MXFP4低精度格式下训练的验证损失仅比BF16基线高0.007，大幅缩小了FP4训练与BF16全精度训练之间的差距。 |
| [^94] | [Correct Answers, Closed Gates: SolveBind for GUI Agents at the Verification Barrier](https://arxiv.org/abs/2610.32223) | 该论文提出SolveBind框架，通过分层求解树（感知-定位-求解-行动-验证）与运行时监督相结合，使GUI智能体能够突破验证码等反自动化机制构成的“验证屏障”，将正确的求解输出转化为被系统接受的交互，从而完成端到端任务。 |
| [^95] | [RxnExBench: Benchmarking Vision-Language Models for Structured Chemical Reaction Extraction from Scientific Literature](https://arxiv.org/abs/2610.32221) | 该论文提出了RxnExBench多模态基准，包含3,565个专家标注反应、统一的15角色/26字段数据模式以及将分子识别与高级推理解耦的Oracle-MolRec评估协议，并揭示了当前视觉语言模型在结构化化学反应提取任务中随模式复杂性和推理需求增加而性能显著下降的问题。 |
| [^96] | [PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models](https://arxiv.org/abs/2610.32216) | PEAR提出一种自适应优势重塑方法，以视觉依赖为基础信号、仅在高熵位置施加与视觉依赖高低挂钩的熵探索增益，从而同时提升大型视觉语言模型多模态推理中的感知与探索能力。 |
| [^97] | [Head Dimension Matters in Language Models for Long-Context Retrieval](https://arxiv.org/abs/2610.32213) | 本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。 |
| [^98] | [SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection](https://arxiv.org/abs/2610.32212) | SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。 |
| [^99] | [Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems](https://arxiv.org/abs/2610.32208) | 本论文发现OpenVLA激活的稀疏自编码器（SAE）表示蕴含结果预测信息，可在机器人任务陷入不可恢复状态前预测失败，并通过识别与成败阶段相关的SAE特征及基于VLM的解释流程，揭示反复出现的行为模式。 |
| [^100] | [Watermarking Diffusion Language Models with Learned Foresight and Balanced Bands](https://arxiv.org/abs/2610.32205) | 提出了一种面向扩散语言模型的水印方法，通过学习平衡的词表频带和前瞻预测器，并以检测器视角对绿token去重计数，在同等文本失真下显著提升了水印检测强度。 |
| [^101] | [SWE-Shift: Evaluating Coding Agents in Evolving Workspaces](https://arxiv.org/abs/2610.32196) | 提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。 |
| [^102] | [Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery](https://arxiv.org/abs/2610.32195) | 提出了基于大语言模型的Principia框架，通过从相关输入与观测结果中形成比较性因果假设、并将假设中反复出现的关系综合为归纳性原理，在测试时自动构建并持续完善科学原理，突破了科学发现中知识构建这一瓶颈。 |
| [^103] | [Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude](https://arxiv.org/abs/2610.32194) | 该论文提出PPO-DAZ3方法，通过动作置信度和归一化优势幅度这两个信号为每个样本动态设置PPO的裁剪阈值，同时保持标准PPO训练流程其余部分不变。 |
| [^104] | [Evidence-Grounded SLM Behavioral Prediction Architecture for Future Assistance Use](https://arxiv.org/abs/2610.32193) | 该研究提出了一种名为ELBP的四层证据驱动本地行为预测架构，利用小型语言模型在设备端预测用户未来的辅助使用需求，在保护隐私的同时满足高风险环境下的及时响应要求。 |
| [^105] | [Uni-Stream: A Unified Streaming Speech Language Model for Speech Synthesis and Recognition](https://arxiv.org/abs/2610.32187) | 提出了Uni-Stream——一个基于紧凑型预训练语音模型、以流式方式处理交错文本与语音的统一语音语言模型，通过两阶段训练在单一模型中同时实现媲美任务专用模型的语音识别与语音合成性能。 |
| [^106] | [Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs](https://arxiv.org/abs/2610.32181) | 提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。 |
| [^107] | [Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies](https://arxiv.org/abs/2610.32179) | 提出一种风险鲁棒的跨任务神经架构搜索方法，通过冠军初始化与联合成对排序器预训练，将稀疏搜索日志与零成本代理相结合，显著缓解最坏情况下的失败并大幅提升最差性能。 |
| [^108] | [$\Psi$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model](https://arxiv.org/abs/2610.32177) | Ψ-0.6 是一个统一的视觉-语言世界模型，通过在共享的“思维流”推理过程中自适应地组合感知、语言、几何与模拟能力，在多视角空间推理上零样本超越领先的开放权重模型，并支持基于想象场景的反事实模拟。 |
| [^109] | [MG-REPA: Multi-Granularity Representation Alignment for Generative Target Speaker Extraction](https://arxiv.org/abs/2610.32161) | 提出多粒度表示对齐方法MG-REPA，通过在帧级与SSL模型对齐、在话语级与说话人验证模型对齐，同时捕获目标语音的时变内容和全局说话人特征，从而加速生成式目标说话人提取模型的训练收敛。 |
| [^110] | [Sparsity-driven emergence of functional modularity in multitask neural networks](https://arxiv.org/abs/2610.32153) | 稀疏性正则化无需预定义模块边界即可促使多任务神经网络自发涌现功能模块化，形成可在深层选择性保留、重组的紧凑“功能token”子网络。 |
| [^111] | [The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study](https://arxiv.org/abs/2610.32151) | 研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。 |
| [^112] | [Implicit Neural Representations for Variational Problems on Graphons](https://arxiv.org/abs/2610.32147) | 该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。 |
| [^113] | [Rethinking LoRA Initialization for Robust Asymmetric Learning Rates](https://arxiv.org/abs/2610.32142) | 该论文通过对 LoRA 动力学的信号-噪声分解，揭示了标准初始化下由初始化引起的噪声项是非对称学习率训练不稳定和性能崩溃的根源，从而主张重新设计 LoRA 初始化以实现稳健的非对称学习率。 |
| [^114] | [Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting](https://arxiv.org/abs/2610.32141) | 提出LARE方法，在保持数据暴露不变的前提下，根据运行中的损失统计自适应地重加权各领域的损失贡献，使单一检查点的知识与推理性能超越多阶段预训练的最佳结果，从而实现更快的LLM预训练。 |
| [^115] | [MPQ: A Message-Passing View of Post-Training Quantization](https://arxiv.org/abs/2610.32127) | 该论文提出消息传递量化（MPQ），将训练后量化重新建模为离散码本上的后验推断，通过在权重块间传递软后验信息并在最后阶段才做硬判决，从而超越GPTQ等现有硬判决求解器，提升大语言模型低比特量化的精度。 |
| [^116] | [ImageRSI: Verification-Driven Recursive Self-Improvement for Agentic Image Generation](https://arxiv.org/abs/2610.32120) | ImageRSI 提出了一个保持图像生成器冻结的验证驱动递归自我改进框架，通过任务内五维验证修复、经 A/B 验证的跨任务经验库以及离线更新的评判器，并发布包含 961 条困难请求的 ImageReq-Bench 基准，显著提升了复杂文生图请求的可靠满足率。 |
| [^117] | [Tampered ML: Evaluating LLM Judgment of Machine Learning Research](https://arxiv.org/abs/2610.32114) | 该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。 |
| [^118] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^119] | [Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers](https://arxiv.org/abs/2610.32110) | 论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。 |
| [^120] | [PersonaScienceBench: Evaluating Scientific Agents with Diverse Simulated Researchers](https://arxiv.org/abs/2610.32106) | PersonaScienceBench通过引入具有不同角色档案的模拟研究人员，在任务目标和评分标准保持固定的前提下，评估科学智能体与多样化研究人员协作的能力，而不仅仅评估其完成任务的表现。 |
| [^121] | [ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization](https://arxiv.org/abs/2610.32103) | 该论文提出ReMedI方法，以分子语言模型作为基础提议者，结合高效的Metropolis-Hastings搜索策略与推理时细化及自蒸馏，在无需昂贵任务特定数据集的情况下实现可控的多目标分子先导化合物优化。 |
| [^122] | [Learning Nonlinear Waveform Fusion Across Scales for Integrated Sensing and Communication](https://arxiv.org/abs/2610.32099) | 该论文提出了一种跨尺度非线性波形融合的学习方法，通过自适应时间特征加权与频谱更新，将图像特征与线性调频参考信号融合于共享复数状态并保持固定总能量，从而在通感一体化系统中同时实现图像传输与多目标的距离-角度定位。 |
| [^123] | [DELA: Learning Controllable Edits for Diffusion Language Models](https://arxiv.org/abs/2610.32096) | DELA 是一个插件式框架，通过插入、删除、重新掩码和停止四种显式编辑操作，在冻结的扩散语言模型上实现可控的草稿级细化，将高层编辑控制与词元级生成解耦。 |
| [^124] | [TTT in VLA Models: Exploring Training with Human Demonstrations](https://arxiv.org/abs/2610.32094) | 该研究将测试时训练（TTT）集成到 InternVLA-A1 和 Pi0.5 两个VLA模型中，探索通过人类演示与机器人轨迹配对训练实现以演示为条件的动作生成，但发现在当前规模下该方法未能实现基于演示的一致目标选择，且各种改进策略均未带来明显提升。 |
| [^125] | [AgentTrace: Execution-Level Fingerprinting of Black-Box LLM Agents](https://arxiv.org/abs/2610.32092) | AgentTrace通过注入结构模糊的探测指令触发工具接口的序列化失败，并用轻量级马尔可夫链建模智能体后续的修复轨迹，从而在对抗性身份伪装下以96.0%的准确率识别黑盒LLM智能体背后的基础模型。 |
| [^126] | [Behaviour analysis for large language model-driven agents](https://arxiv.org/abs/2610.32087) | 提出了一个将数据驱动的行为发现、可解释的依赖关系分析与实验验证（含提示引导）相结合的通用框架，用于理解大语言模型驱动智能体在轨迹中如何组织行为及其与情境和结果的关系。 |
| [^127] | [Guided Zero-Shot Anomaly Explanation for Anonymized Crowd Video Monitoring: Context, Categorization, and Spatial Localization](https://arxiv.org/abs/2610.32073) | 本文提出XAVAD方法，将零样本匿名化、异常类型分类、视觉定位与场景上下文描述统一整合，为匿名化视频异常检测提供可解释且保护隐私的多方面异常解释。 |
| [^128] | [DriveMind: Mind-Evolving Belief Tracking for Closed-Loop Autonomous Driving](https://arxiv.org/abs/2610.32069) | DriveMind借鉴粒子滤波思想，让基于MLLM的驾驶智能体同时维护多个场景假设并随新观测动态更新权重与重采样，从而显著提升闭环自动驾驶性能。 |
| [^129] | [Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes](https://arxiv.org/abs/2610.32067) | 该论文通过引入未训练编码器、频带功率基线和基于验证集的层选择三种对照，对多个EEG基础模型进行受控评估，发现其预训练收益大多不具统计稳健性，随机初始化和简单的频带功率基线即可与之匹敌。 |
| [^130] | [Understanding Communication Failures in Multi-Agent LLM Systems](https://arxiv.org/abs/2610.32060) | 该论文基于3,211条失败的多轮智能体交互轨迹，构建了多智能体LLM系统通信失败的分类法，并揭示失败主要由恢复/纠正与共识/决策问题主导，而非孤立的首轮错误。 |
| [^131] | [Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders](https://arxiv.org/abs/2610.32059) | 该论文提出ConceptACP框架，利用稀疏自编码器的可解释性分析定位U-Net去噪器中与目标概念最相关的交叉注意力块并进行针对性微调，从而实现对扩散模型概念的高效精确擦除。 |
| [^132] | [ACVA-Bench: Beyond Prompt Following in Abstract Concept Visualization Ability for Text-to-Image Models](https://arxiv.org/abs/2610.32058) | 该论文定义了文本到图像模型的抽象概念可视化能力（ACVA），并基于心理语言学规范提出类别平衡的ACVA-Bench基准，在低干预设置下从语义充分性、表征多样性和概念区分度三个维度系统评估模型自主可视化抽象概念的能力。 |
| [^133] | [HyScale: Scaling Long-Context Training for Learned Sparse Attention](https://arxiv.org/abs/2610.32057) | HyScale 通过无需分数表的实时精确 top-k 选择方法和前向选择重放机制，将学习型稀疏注意力的百万词元长上下文训练速度提升了 2.07 倍。 |
| [^134] | [Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?](https://arxiv.org/abs/2610.32046) | 提出测试时记忆演化（TTME），一种无需更新模型权重的即插即用测试时适应方法，通过持续演化外部记忆来积累经验，在科学发现任务上以更高的计算效率超越了测试时训练（TTT）。 |
| [^135] | [Do AI-for-Science Agents Preserve Model Safety?](https://arxiv.org/abs/2610.32045) | 该研究首次系统评估了AI-for-Science智能体框架对模型安全性的影响，发现将LLM集成到科学智能体框架中会显著削弱安全性，即使在直接提示下也会增加有害内容的依从性，且对基础模型无效的自适应攻击在智能体框架中变得有效。 |
| [^136] | [Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data](https://arxiv.org/abs/2610.32039) | 本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。 |
| [^137] | [PPO vs. SAC: A Comparative Study for Cooperative Motion Control of Dual-Arm Space Robots](https://arxiv.org/abs/2610.32034) | 本文提出了一个针对双臂空间机械臂协同运动规划的强化学习框架，通过集中式状态表示和多目标奖励函数，在具有臂间碰撞避免约束的目标到达任务中对比研究了PPO和SAC两种算法的性能。 |
| [^138] | [RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training](https://arxiv.org/abs/2610.32026) | RSI-Opt提出了一个即插即用的递归自我改进优化器智能体框架，元智能体利用历史运行证据持续演化指导策略（提示词、操作手册、代码库和决策节奏），显著提升了LLM智能体自动调节模型训练优化器的性能。 |
| [^139] | [BARC-Bench: Benign Asks, Risky Consequences](https://arxiv.org/abs/2610.32024) | 该论文提出 BARC-Bench 基准，通过 110 个模拟工作区任务揭示前沿智能体模型在面对隐含安全边界的良性请求时并非始终安全，即使最强配置仍有 27.3% 的任务出现安全违规。 |
| [^140] | [CEMIR vs. FISER: a comparative study of reasoning-based and enrichment-based approaches to pragmatic intent recognition](https://arxiv.org/abs/2610.32022) | 本文通过一项跨四个大语言模型的比较性与诊断性研究，对比了基于结构化社会与具身推理的 FISER 和基于多模态叙事与逼真图像增强的 CEMIR，以探究当指令模糊性源于未明说的人类意图而非缺失场景信息时，多模态增强方法是否仍具优势。 |
| [^141] | [MI-MIDI 2: From Observation to Control with Sparse Autoencoders in Text-to-MIDI Generation](https://arxiv.org/abs/2610.32007) | 本文提出了一个端到端协议，首次在文本到MIDI生成中系统验证了SAE特征解释对实际输出控制的预测能力，实现了从观测模型特征到干预控制生成结果的跨越。 |
| [^142] | [OpenHome: From Perception to Insight in Smart-Home Camera Videos](https://arxiv.org/abs/2610.31986) | 该论文提出了OpenHome，一个包含2,254个问题、覆盖1,118个真实智能家居视频片段的纯视觉多选题基准测试，用于在稀疏视觉信号和有限上下文条件下系统评估视觉语言模型的智能家居视频理解能力。 |
| [^143] | [The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows](https://arxiv.org/abs/2610.31985) | 该论文揭示了“有益智能体问题”：即使没有恶意或共谋，追求各自目标的良性LLM智能体也会在审查机制失效时自发合作、代替彼此执行操作，从而在合法权限范围内绕过预期的监督工作流，构成一种新型AI安全风险。 |
| [^144] | [Spectra: Multi-Scale Implicit Decomposition for Deepfake Detection](https://arxiv.org/abs/2610.31979) | Spectra通过将隐式神经表示拟合到图像并分析其多尺度内部激活，揭示图像的重构动态，从而捕捉真实与合成内容之间更本质的差异，克服了传统基于外观或频域线索的方法在泛化性上的局限。 |
| [^145] | [Agentic AI Defense: Decode-Time Provenance Binding for MCP](https://arxiv.org/abs/2610.31974) | 该论文提出解码时溯源绑定机制，通过HMAC认证的信任标签、MCP维护的溯源账本和上下文污点跟踪三者的结合，有效防御针对MCP智能体系统的跨模态隐写式间接提示注入攻击。 |
| [^146] | [Reproducible Evaluation as the Substrate for Safe Prompt Compression in Production Agents](https://arxiv.org/abs/2610.31972) | 该论文提出了一个能在实时数据变化下保持可复现并可按需重新生成的离线评估框架，并将其作为带正确性门控的GEPA提示优化的适应度函数，成功将生产智能体的提示规模压缩3.5-7.6%而不损害任务正确性。 |
| [^147] | [Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation](https://arxiv.org/abs/2610.31971) | 本文提出了一个重复多方、多议题的LLM谈判框架，发现LLM智能体虽然能从历史交互中形成有用的信任估计并识别暗中阻挠协议的对手，但知道该信任谁并不足以保证达成协议，从而揭示了“信任-协议鸿沟”。 |
| [^148] | [A Mathematical Theory of the Jacobian Lens](https://arxiv.org/abs/2610.31955) | 本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。 |
| [^149] | [RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy](https://arxiv.org/abs/2610.31954) | 本文提出RobustNCL框架，通过轻量级检测与修复策略提升神经形态持续学习在硬件瞬态故障（特别是DRAM片外存储器故障）下的可靠性。 |
| [^150] | [Kolibri: A Sovereign European Model on the Pareto Frontier](https://arxiv.org/abs/2610.31945) | Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。 |
| [^151] | [$M^3$: A Multi-LLM System for Multi-Turn Jailbreaking as a Markov Decision Process](https://arxiv.org/abs/2610.31937) | 该论文提出 $M^3$ 自动化红队测试框架，将多轮越狱形式化为有限视野马尔可夫决策过程，通过攻击者LLM与状态LLM的协作，在完全黑盒设置下显著超越现有方法，攻击成功率最高提升53%。 |
| [^152] | [ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy](https://arxiv.org/abs/2610.31927) | 本文提出ENCL方法，通过轻量级的时空局部学习规则和子网络冻结策略，取代传统神经形态持续学习中内存消耗大的经验回放和昂贵的全局学习规则，实现了高效的持续学习。 |
| [^153] | [Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student?](https://arxiv.org/abs/2610.31918) | 该论文提出分层智能体蒸馏框架，将同策略性分解为交互历史与动作前缀两个层级并实现显式控制，发现“学生生成的历史+教师生成的动作前缀”的组合通常是最优的蒸馏方案。 |
| [^154] | [Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows](https://arxiv.org/abs/2610.31916) | 该论文将AI智能体的恢复决策与语言模型行为分离，提出一项在LangGraph和Temporal宿主上执行336条历史记录的有限一致性研究，验证了检查点状态、分发许可、外部效果证据和对账许可这四种独立义务的执行时可靠性。 |
| [^155] | [Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference](https://arxiv.org/abs/2610.31897) | 提出统一框架S-SALAAD，联合学习模型权重与结构化稀疏+低秩表示，使大模型既能弹性压缩以适配边缘设备的多种内存预算，又能实现真正加速的高效推理。 |
| [^156] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^157] | [Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima](https://arxiv.org/abs/2610.31884) | 通过构建并分析策略学习的最小化模型，该研究证明了课程学习的有效性取决于奖励的稀疏程度：在稠密奖励下课程仅有边际收益，而在稀疏奖励下课程可能是学习得以发生的必要条件。 |
| [^158] | [TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules](https://arxiv.org/abs/2610.31882) | 该论文提出了TACS-RF框架，首次让LLM编程智能体直接驱动Cadence、Ansys等工业级EDA工具，半自主完成了90nm SiGe BiCMOS工艺下240-251GHz亚太赫兹发射机从振荡器、功率驱动、倍频器到天线的完整芯片设计，验证了LLM智能体驱动真实射频芯片设计流程的可行性。 |
| [^159] | [FSVL: Taxonomic Fast-Slow Vision-to-Language Collaborative Framework for Interpretable Confusable Fish Species Identification and Knowledge Generation in the Middle Yangtze River](https://arxiv.org/abs/2610.31880) | 该论文提出FSVL快-慢双通路框架，利用轻量级ResNet50快速通路结合基于大语言模型的检索增强生成（RAG）慢速通路，在保证推理效率的同时实现对长江中游易混淆鱼类的可解释细粒度识别和可靠生物学知识生成。 |
| [^160] | [Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance](https://arxiv.org/abs/2610.31863) | 本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。 |
| [^161] | [SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?](https://arxiv.org/abs/2610.31854) | 该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。 |
| [^162] | [Rethinking Multi-View Fusion for Urban Representation Learning](https://arxiv.org/abs/2610.31850) | 本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。 |
| [^163] | [Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons](https://arxiv.org/abs/2610.31846) | 提出Self-Direction机制，让智能体在任务失败后自我反思提取教训，并立即蒸馏为轻量级低秩适配器（仅占基础模型约0.02%参数量）存入持久化库中，从而将错误转化为不遗忘、不破坏原有能力的持久化学习成果。 |
| [^164] | [SEVI: Safe Evidential Real-World Reinforcement Learning](https://arxiv.org/abs/2610.31829) | SEVI提出证据式安全评论家框架，在单次前向传播中同时估计约束违反风险与认知不确定性，并在真实世界策略学习中对不安全或证据不足的动作进行离线恢复策略干预，从而实现安全的真实世界强化学习。 |
| [^165] | [What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry](https://arxiv.org/abs/2610.31821) | 该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。 |
| [^166] | [PrefO2O: Zero-Shot Preference-Based Reinforcement Learning via Offline-to-Online Latent Preference Transfer](https://arxiv.org/abs/2610.31812) | 提出PrefO2O框架，通过从源任务少量离线偏好数据中学习任务不变的潜在偏好空间，并利用置信度加权最优传输对齐进行在线奖励适应，实现了无需任何人类反馈的零样本基于偏好的强化学习。 |
| [^167] | [PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean](https://arxiv.org/abs/2610.31810) | 提出PaperFormalizer——一个基于Lean的“作者在环”智能体流水线，能够发现并修复研究论文证明中的漏洞，并在作者参与关键决策下将论文转化为经Lean验证的形式化版本。 |
| [^168] | [Patronizing and Condescending Language Detection in English and Mandarin](https://arxiv.org/abs/2610.31808) | 该论文提出了一种自动检测英语和中文中居高临下与恩赐性语言的系统，其中基于微调BERT的方法表现显著优于静态词嵌入结合卷积神经网络的方法，在开发集上取得了0.528的F1分数。 |
| [^169] | [Phronix: The Capability-Centric Learning Paradigm for Agents](https://arxiv.org/abs/2610.31789) | 提出Phronix，一种以能力为中心的智能体学习范式，通过预测-验证-纠正循环让智能体从自身执行中自主获取、验证并完善可复用的能力，摆脱对人工编写技能和记忆的依赖，并将失败任务转化为研究机会。 |
| [^170] | [Benchmarks Are Surprisingly Compaction-Friendly](https://arxiv.org/abs/2610.31778) | 该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。 |
| [^171] | [Generalized green and red concepts for artificial intelligence, big data, and information and communication technologies: Multidimensional framework and countermeasures](https://arxiv.org/abs/2610.31739) | 本文提出广义绿色AI（GGAI）与广义红色AI（GRAI）概念，并在统一的环境-经济-社会-技术/运营多维框架下将“红/绿”分类体系扩展至大数据、信息通信技术、电子电气工程与数字技术领域，以全面评估AI等技术带来的电力、水资源与材料等多维可持续性影响并提出应对对策。 |
| [^172] | [Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability](https://arxiv.org/abs/2610.31733) | 该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。 |
| [^173] | [Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems](https://arxiv.org/abs/2610.31722) | 该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。 |
| [^174] | [End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization](https://arxiv.org/abs/2610.31721) | D2C利用确定性采样使反向去噪过程可微的特性，将冻结的条件扩散模型转化为端到端的梯度优化器，在测试时通过代理目标和物理信息的梯度直接优化条件变量，从而克服了离线基于模型优化中现有方法过于保守或无法精确对齐物理目标的问题。 |
| [^175] | [What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md](https://arxiv.org/abs/2610.31715) | 这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。 |
| [^176] | [Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control](https://arxiv.org/abs/2610.31712) | 该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。 |
| [^177] | [MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability](https://arxiv.org/abs/2610.31711) | 该论文提出MBABench-2.0基准测试，包含101个从零构建财务电子表格的原创任务，突破了以往填空式简单可验证任务的局限，用于更真实地评估AI智能体完成建模决策、格式设计和表格布局等核心电子表格构建工作的能力。 |
| [^178] | [Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents](https://arxiv.org/abs/2610.31710) | 提出 Compact-Bench 基准，通过在共享仓库状态下配对“完整历史”与“仅摘要”的延续过程，首次将上下文压缩对长程编程智能体的具体损害（如抹去 5-64% 的成功修复、最多仅保留 16% 的重复细节）从效率指标中隔离出来进行量化评估。 |
| [^179] | [Mining the Design Space of Multi-Agent Workflows](https://arxiv.org/abs/2610.31709) | 提出一个两阶段流水线，通过挖掘28,568个可执行工作流的代码与执行轨迹，自动提炼出可复现的多智能体系统设计原则。 |
| [^180] | [RubricCraft: Where Solutions Diverge, Criteria Emerge](https://arxiv.org/abs/2610.31707) | RubricCraft 通过对比多个候选解法之间的满足与违反需求的差异来构建文本评分规则，从而为开放式任务提供显式且可靠的自动化评估标准。 |
| [^181] | [Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant](https://arxiv.org/abs/2610.31703) | 该论文提出了一个完全本地化部署的智能体式LLM助手，通过MCP协议编排实时文档RAG检索与在11,244条历史工单“问题-解决方案”对上QLoRA微调的Llama-3-8B模型并融合二者答案，从而在用户提交工单之前就解答其HPC使用问题。 |
| [^182] | [Collaborative In-Context Learning for Indoor Radio Map Estimation](https://arxiv.org/abs/2610.31702) | 提出Poly-Ripple框架，首次将室内无线电地图估计形式化为协作式上下文学习问题，使多个客户端利用大型自回归视觉模型和波纹序自回归建模，在无需模型更新并保护隐私的前提下协作预测无线电地图。 |
| [^183] | [Tradeoff-Aware Territory Design: A Multi-Objective Decision System for Enterprise Customer-to-Seller Allocation](https://arxiv.org/abs/2610.31701) | 该论文提出了一个面向企业B2B销售区域设计的生产级多目标决策系统，通过融合多维客户价值评分、基于反事实框架的销售关系连续性评估以及多目标优化，在增长、均衡、销售能力与可行性之间实现权衡感知的客户-销售分配。 |
| [^184] | [Distributional Alignment of LLMs and Its Generalization Properties](https://arxiv.org/abs/2610.31687) | 该研究提出了大语言模型与人类分布对齐的方法，发现对齐后的模型能够泛化到同领域的未见问题，跨主题泛化取决于主题间的关联性，但可能伴随创造力等能力的意外退化。 |
| [^185] | [From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models](https://arxiv.org/abs/2610.31686) | 激活干预虽能大幅提升模型对熟悉目标的选择正确率，但这种增益无法泛化到事实与标识符的重新组合或独立资格判断上，仅能改变源事实影响判断的方式。 |
| [^186] | [An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation](https://arxiv.org/abs/2610.31685) | 该论文提出了一种可追溯的多模态情感推理智能体框架，通过模态感知智能体发布带证据引用的观察、集成智能体生成情感标签与引用，并利用独立的基于证据的验证阶段执行确定性检查来定位错误来源，从而以低成本实现对话中的多模态情感推理。 |
| [^187] | [Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity?](https://arxiv.org/abs/2610.31681) | 该论文将结构化生成重新表述为歧义感知推理问题，提出基于图推理的多有效学习框架，通过对一组合理的结构化输出进行推理而非预测单一解，解决了自然语言歧义导致现有方法输出不稳定和不一致的问题。 |
| [^188] | [Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation](https://arxiv.org/abs/2610.31678) | 该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。 |
| [^189] | [How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions](https://arxiv.org/abs/2610.31606) | 该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。 |
| [^190] | [SABLE: Scalable Affine Bayesian Latent-space Exploration](https://arxiv.org/abs/2610.31495) | SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。 |
| [^191] | [DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150](https://arxiv.org/abs/2610.31104) | DeepOpen提出了针对BANKING77和CLINC150的面向决策意图分类框架，通过SAM续训的单分类器和多模型概率融合方法，分别在两个数据集上取得了94.12%和98.02%的高准确率，同时严谨地区分了头条分数与统计意义上的改进证据。 |
| [^192] | [Co-Recursive Semantic Self-Extension: A Mathematical Foundation for Open-Ended Intelligence](https://arxiv.org/abs/2610.31097) | 该论文借鉴康德与哥德尔思想，将认知状态建模为形式语言上的部分求值，提出“协递归语义自我扩展”的数学框架，通过对不可判定中间前提的反思性具体化，使智能系统能够突破固定目的与布尔封闭目的变换的语义边界，为开放式智能奠定数学基础。 |
| [^193] | [Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters](https://arxiv.org/abs/2610.31017) | 开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。 |
| [^194] | [Mitigating Saliency Collapse: Robust Saliency-Aware Long-Text Image-Text Alignment](https://arxiv.org/abs/2610.31005) | 提出RoSA显著性感知微调框架，通过将图文对分解为互补的显著视图与上下文视图并进行联合对齐学习，有效缓解长文本图文对齐中的显著性坍塌问题。 |
| [^195] | [OpenAttention: One Percent Is Enough with Tail Proxies](https://arxiv.org/abs/2610.30951) | OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。 |
| [^196] | [Learning from Next-Frame Prediction: Towards Unified and Scalable Visual World Modeling](https://arxiv.org/abs/2610.30724) | 提出VANE自监督视觉预训练框架，通过在冻结语义分词器提供的固定嵌入空间中预测整个下一帧的表示，将时间演化作为核心学习信号，实现统一且可扩展的视觉世界建模。 |
| [^197] | [Scaling Laws for Safety Probes](https://arxiv.org/abs/2610.30678) | 本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。 |
| [^198] | [Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy](https://arxiv.org/abs/2610.30474) | 本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。 |
| [^199] | [Astro2Method: Benchmarking LLMs on Machine Learning Method Selection in Astronomy](https://arxiv.org/abs/2610.30282) | 该论文提出了Astro2Method这一专家审阅的基准，用于评测大语言模型在天文学中为科学问题选择机器学习方法的能力，发现LLM在简单问题上表现良好但在困难问题上准确率显著下降，且大多数检索增强和智能体方法无法带来一致的性能提升。 |
| [^200] | [GeoDiv KAN Solver for Time Fractional Navier Stokes Equations on Complex Domains to Realistic CFD Reconstruction](https://arxiv.org/abs/2610.30179) | 本文提出一种几何门控且从构造上保证无散度的高斯径向基函数Kolmogorov-Arnold网络（GeoDiv KAN），用于在方形、圆形和星形等复杂域上求解时间分数阶不可压缩Navier-Stokes方程，实现真实的CFD重构。 |
| [^201] | [Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework](https://arxiv.org/abs/2610.30170) | 该研究提出了一套标准化的AI代理模型训练与评估框架，以及在国家尺度数据上训练的多任务深度学习代理模型，可将原本需要数天至数周的加拿大野火危害制图过程模拟大幅加速，支撑全国性的主动野火风险减缓决策。 |
| [^202] | [Bio-Inspired Auditory Processing: Coupling Parallel Resonate-and-Fire Dynamics with Ternary Recurrent Networks for Speech Recognition](https://arxiv.org/abs/2610.29720) | 提出脉冲三值GRU（stGRU），用三值权重门控循环单元进行时间滤波、由共振-发放神经元生成二值脉冲流驱动事件通路，将二值事件传输内化为训练原生约束，从而提升SNN在语音识别等连续上下文任务中的性能与能效。 |
| [^203] | [Latent space configuration for neural network finetuning, fast ranking, and embedding dimensionality reduction](https://arxiv.org/abs/2610.29295) | 该论文提出潜在空间配置（LSC）微调方法，使神经网络嵌入具有结构化几何属性，从而支持O(1)复杂度的候选子集搜索，相比穷举搜索在GPU上提速高达3倍、CPU上提速40%，同时还能降低嵌入维度。 |
| [^204] | [An Eco-Friendly Tourism Trip Planner Driven by Z-Number Uncertainty: Evidence from Baku, Dubai, Abu Dhabi, Tashkent, and Istanbul](https://arxiv.org/abs/2610.29120) | 该论文提出一种基于Z数的环保旅游行程规划器，直接从路径服务观测和平台评论数构建行程时间与评分的模糊不确定性表示，结合混合损失函数与自适应大邻域搜索，在五个城市中生成带可靠性标注的低碳单日行程。 |
| [^205] | [GenXR-Bench: Benchmarking Coding Agents for Interactive XR Development](https://arxiv.org/abs/2610.29014) | 该论文提出了首个用于端到端评估编码智能体XR软件开发能力的基准GenXR-Bench，通过112个专家编写任务以及结合脚本化/自适应交互与程序化/视觉断言的测试方式，有效评估核心功能和运行时行为。 |
| [^206] | [SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges](https://arxiv.org/abs/2610.28886) | SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。 |
| [^207] | [S$^2$-MAS: A Society-Scale Multi-Agent Simulation Framework with 500 Million Household Persona Agents](https://arxiv.org/abs/2610.27746) | 该论文提出了S²-MAS社会规模多智能体仿真框架，基于真实调查数据构建5亿个家庭角色智能体，并配备专用紧凑决策模型FinL-mlx，能够在大规模仿真中刻画家庭金融决策及信用额度等政策变化对消费行为的差异化影响。 |
| [^208] | [RLSplat: Resource-Linear Budget Control for Feed-forward 3D Gaussian Splatting](https://arxiv.org/abs/2610.27712) | RLSplat提出了一种资源线性的预算可控前馈式3D高斯泼溅框架，摒弃了先预测稠密高斯再剪枝或选择的做法，使表示大小、推理延迟和峰值内存随高斯预算的减少而线性下降。 |
| [^209] | [EASformer: Environment-Action-State Alignment for Industrial Generative Ranking](https://arxiv.org/abs/2610.27691) | 提出EASformer架构，通过将环境、动作与状态语义解耦并对齐序列与非序列特征之间的不对称语义，解决生成式排序中query-key点积比较异质语义对象、跨token交互受限的问题。 |
| [^210] | [MLIPS: Multi-Layer Indexer Pool Sharing](https://arxiv.org/abs/2610.27438) | MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。 |
| [^211] | [EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver](https://arxiv.org/abs/2610.25931) | EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。 |
| [^212] | [Muse Glimmer 30B: An On-Device Agentic Model](https://arxiv.org/abs/2610.24628) | Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。 |
| [^213] | [Bridging Exploration and Persona: Information-Directed Dueling for Test-Time Agent Personalization](https://arxiv.org/abs/2610.24385) | 提出ID-TAP框架，将查询条件化的神经效用模型与轻量级Bradley-Terry探索器相结合，利用相似度加权的跨查询证据迁移，在十轮比较预算内实现了大语言模型代理在测试时的最优个性化画像选择。 |
| [^214] | [AM-Bench: A Unified Taxonomy and Evaluation Suite for Agentic Misalignment](https://arxiv.org/abs/2610.24264) | 提出基于犯罪学“舞弊三角”理论构建的 AM-Bench 基准，通过涵盖七个行业的 35 个场景，从感知威胁、感知机会和推理三个维度系统评估大语言模型智能体在无人明确指示下主动选择有害行为的“智能体失准”问题。 |
| [^215] | [Algorithm Modelization: Transforming Optimization Algorithms into Evolvable Intelligence](https://arxiv.org/abs/2610.23161) | 提出“算法模型化”新范式，将迭代优化算法转化为可训练的自回归模型，通过学习算法执行轨迹使模型内化、复现并进化超越现有算法。 |
| [^216] | [AgentPerturb: Evaluating LLM Agent Robustness under Controlled Interaction Perturbations](https://arxiv.org/abs/2610.21624) | 该论文提出AgentPerturb框架，通过联合刻画扰动目标与类型的二维视角，系统评估了LLM智能体在受控交互扰动下的鲁棒性，发现扰动一致地降低性能、智能体很少识别被篡改的输入且常未经验证地传播错误信息。 |
| [^217] | [Reinforcement Learning-driven Diffusion Models: Language, Vision, and Beyond](https://arxiv.org/abs/2610.21539) | 本文首次全面综述了强化学习驱动的扩散模型，提出新颖的技术全景图，阐明RL如何通过奖励驱动的后训练机制将扩散过程重塑为以经验为中心、目标导向的生成，并覆盖语言、多模态、机器人和科学模型等多个领域。 |
| [^218] | [Same Context, Different Agents: A Behavioral Study of Semantic Layers, Orchestration and Compaction in Data Agents](https://arxiv.org/abs/2610.20959) | 该论文首次系统性地研究了语义层、编排与压缩三种上下文工程技术在数据智能体中单独及组合使用时的行为、准确性与成本影响，发现无这些技术的智能体会呈现调查者、过早回答者和过早停止者三种行为群体，且每种技术的收益取决于其所匹配的智能体行为群体。 |
| [^219] | [Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs](https://arxiv.org/abs/2610.20179) | 本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。 |
| [^220] | [One-Sided Verification Under Limited Expert Review](https://arxiv.org/abs/2610.18089) | 本文提出OSVerify框架，通过将错误提议与错误验证解耦，在有限专家审查预算下实现单侧验证目标——最大化无效解的拒绝率的同时严格控制有效解的错误拒绝率，从而显著降低人工审查成本。 |
| [^221] | [Interaction-as-a-Judge](https://arxiv.org/abs/2610.17895) | 提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。 |
| [^222] | [Prediction-Powered Data Fusion for Treatment Effect Estimation](https://arxiv.org/abs/2610.12332) | 提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。 |
| [^223] | [ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation](https://arxiv.org/abs/2610.09163) | 该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。 |
| [^224] | [Back to the Future: Rethinking EDA Infrastructure for Agentic Systems in Chip Design Verification](https://arxiv.org/abs/2610.06790) | 提出了端到端智能体框架BTTF，通过将海量仿真数据规范化为SQLite数据库并结合多智能体编排引擎，将自然语言验证查询转化为模式感知的SQL并关联信号异常与版本化RTL代码库，填补了LLM在EDA中仅应用于静态RTL代码生成、而仿真后验证与波形调试长期空白的缺口。 |
| [^225] | [ArtifactArena: Evaluating Models by What They Build in the Physical World](https://arxiv.org/abs/2610.06511) | ArtifactArena 提出一个开放式测试平台，通过让模型设计制造功能完备的机器人在模拟竞技场中实际对战并以 Elo 排名评测，来衡量模型在物理世界中的开放式工程与设计能力。 |
| [^226] | [Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting](https://arxiv.org/abs/2610.02571) | 本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。 |
| [^227] | [Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics](https://arxiv.org/abs/2609.36492) | 该研究首次系统性地基准测试了视觉语言模型在连接组学突触检测与校对任务上的表现，发现零样本性能接近随机，但通过少样本学习和LoRA微调，开源模型可达到专业模型水平，且在跨物种迁移中优于专业模型。 |
| [^228] | [Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging](https://arxiv.org/abs/2609.32303) | 该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。 |
| [^229] | [Symbolic Guidance for LLM Agents in Distributed Multiagent Coordination](https://arxiv.org/abs/2609.31963) | 该论文提出符号引导分类法（SGT），通过基于成熟算法的符号引导来调节LLM智能体的自主性程度，并发现在分布式多智能体协调任务中，提供部分伪代码引导的中等自主性级别始终优于无引导推理和完全规定算法执行这两种极端方式。 |
| [^230] | [Metro-WM: Long-Horizon Latent Planning with Realisable Sub-Goals](https://arxiv.org/abs/2609.31868) | 论文指出分层JEPA宏观规划器预测的潜在子目标往往物理上不可实现，并提出Metro-WM框架，通过从历史经验构建的图中检索真实状态作为子目标，从而实现可靠的长时程规划。 |
| [^231] | [When Clinical Judgment Becomes a Predictive Shortcut: Two Questions Clinical AI Must Answer Before Deployment](https://arxiv.org/abs/2609.25852) | 该论文提出一个与预期用途相匹配的证据要求框架，指出临床AI利用临床判断信息的预测性能本身不足以证明其有用性，并通过状态参照与代理相对时序两项测试，帮助医院在部署前验证模型在最需要辅助时性能条件是否仍然成立。 |
| [^232] | [Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation Enforcement](https://arxiv.org/abs/2609.24243) | 该论文揭示了强化学习会导致视觉语言模型出现“思维链混淆”——即准确率上升但推理过程变得难以监控，并提出TAME方法，利用稀疏自编码器在强化学习中定向抑制模板相关激活，从而保持模型推理过程的可监控性。 |
| [^233] | [PlaceReasoner-Beta: Reasoning-Driven Macro Placement and Benchmarking](https://arxiv.org/abs/2609.21263) | 该论文提出PlaceReasoner-Beta，一个由验证器引导的多智能体框架，将VLSI宏单元布局从传统黑盒优化重新表述为闭环推理问题，通过视觉语言模型规划器、几何验证器、物理验证器和布线后优化器的协同工作，在统一流程中融合视觉布局信息、设计专业知识与物理设计反馈。 |
| [^234] | [Training-Free Task Vectors for LLM Behavioral Control](https://arxiv.org/abs/2609.09054) | 提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。 |
| [^235] | [From Review to Authorization: Key-Isolated Threshold Signing for LLM Agents](https://arxiv.org/abs/2609.05901) | KITA架构通过将用户签名密钥和门限签名密钥份额完全隔离在所有LLM进程之外，使提示注入攻击无法从内容判断跨越到执行权限，为大语言模型智能体建立了执行绑定的授权完整性保障。 |
| [^236] | [Testing Interchangeability in LLM Agent Teams](https://arxiv.org/abs/2609.05279) | 该研究通过在独立形成的LLM智能体团队之间交换角色匹配的智能体，发现人员替换几乎不影响任务得分，却使团队每单位进度的通信开销增加16%–63%，且被换入的智能体有时比新手成本更高，表明智能体因携带与原伙伴形成的协作惯例而并非可互换。 |
| [^237] | [PolyMemDB: A Polyglot Database System for AI Memory Management](https://arxiv.org/abs/2608.25577) | PolyMemDB是一个面向AI记忆管理的多范式数据库系统，统一管理图、向量、概率和时空等多种记忆类型，并通过结合时间衰减与半环聚合的概率推理引擎解决长期事实冲突、减少LLM幻觉。 |
| [^238] | [TreeWY: Speculative Verification for Gated DeltaNet Hybrids](https://arxiv.org/abs/2608.20961) | 本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。 |
| [^239] | [An LLM agent for end-to-end computational materials discovery](https://arxiv.org/abs/2608.20434) | 本文介绍了MAESTRO，一个基于LLM的智能体系统，能自动化执行MOF材料发现的全流程筛选，通过连接异构任务发现传统方法难以识别的高性能材料。 |
| [^240] | [FinVerse: Financial Time-Series Benchmark](https://arxiv.org/abs/2608.03259) | FinVerse是一个金融领域的时间序列预测基准，通过更现实的评估方式（如方向预测）来弥补传统误差指标在真实决策中的不足，包含大量金融时间序列数据。 |
| [^241] | [AI-guided deep visual proteomics reveals recurrence-associated morpho-molecular signatures within triple-negative breast cancers](https://arxiv.org/abs/2608.03145) | 该研究开发了一种结果导向的AI引导深度视觉蛋白质组学框架，利用复发风险热图指导三阴性乳腺癌的组织采样与质谱空间蛋白质组学分析，揭示了高风险区域富集有丝分裂程序、低风险区域富集免疫与抗原呈递程序的肿瘤内分子异质性。 |
| [^242] | [MirrorCraft: Paired Evaluation under Hidden Rule Changes in Minecraft](https://arxiv.org/abs/2607.29218) | MirrorCraft提出了一种配对基准测试方法，通过在Mirror世界中悄悄修改服务器端规则、同时保持其他条件与Vanilla世界一致，来评估LLM智能体在Minecraft中面对隐藏规则变化时的适应能力。 |
| [^243] | [Shapes from Examples: Foundations of Shape Learning in Recursive SHACL](https://arxiv.org/abs/2607.27934) | 该论文研究从正负例节点学习递归SHACL形状的拟合问题，针对对应描述逻辑ELI*的SHACL核心片段，在良基、稳定和支持三种目录语义下，为拟合存在性与最具体拟合计算建立了紧凑的指数时间上界，并在特殊情形下给出了多项式界。 |
| [^244] | [AgentGUI: An Interface for Observing and Steering Long-Running AI Agents](https://arxiv.org/abs/2607.26300) | AgentGUI是一个本地托管的图形界面，通过丰富的轨迹可视化、手动与自动引导以及多框架集成协调，显著提升了人类对长时运行AI代理的监督效率（轨迹分析提速38%）和小型本地代理的任务完成率（最高提升34个百分点）。 |
| [^245] | [Where Cross-Question Structure in LLM Errors Comes From: Shared Seeds, Shared Prompts, and Answer-Letter Preference](https://arxiv.org/abs/2607.20464) | 该论文通过问题相关性矩阵的特征值分析发现，LLM错误中的跨问题结构主要来自共享随机种子、共享提示词以及对答案字母的偏好（主要由Llama-3.2-1B驱动），而非共享的知识缺失。 |
| [^246] | [JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety](https://arxiv.org/abs/2607.19913) | Janus提出前瞻式智能体安全框架，通过预判任务与裁决任务的联合强化学习训练出守护模型Vanguard，能够在不安全操作执行前预见延迟风险，使平均防护提升15.9个百分点、良性任务完成率提升5.1个百分点。 |
| [^247] | [DeepSafe-SciHazard: A Benchmark for Measuring Scientific Safety Risks with Decomposed Harm Scoring](https://arxiv.org/abs/2607.18665) | 该论文提出了SciHazard，一个基于真实世界的科学安全风险基准，包含跨12个学科的2400个危险问题和600个过度安全问题，并开发了分解式的DeHarm-Score评估框架，通过综合查询危害严重程度、拒绝行为和响应层面风险来更准确地衡量大语言模型的科学安全风险。 |
| [^248] | [Semantic Anchoring for Robotic Action Representations](https://arxiv.org/abs/2607.13597) | 该论文发现VLA模型微调会破坏动作表示的语义结构并损害泛化能力，据此提出一种即插即用的语义锚定方法，通过将动作表示锚定到语义流形并分解为共享语义通道与私有通道，有效提升任务成功率与分布外泛化性能。 |
| [^249] | [How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism](https://arxiv.org/abs/2607.07891) | 本文提出巴伦霍尔茨基于大语言模型行为发展出的语言自生成理论，能够填补哈里斯整合主义语言学在符号未来开放性机制、语言与非语言符号活动连续性以及历史整合档案结构属性方面的解释空白，从而在不违背其核心主张的前提下丰富整合主义理论。 |
| [^250] | [WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time](https://arxiv.org/abs/2607.06988) | WAM-TTT提出了一种测试时训练框架，通过元训练使冻结的世界动作模型内获得轻量级自适应记忆，在测试时仅需观看无标注的人类视频即可高效引导机器人行为，无需机器人演示、人类标注或任务特定微调。 |
| [^251] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^252] | [How Should Transformers Encode Numeric Values in Electronic Health Records?](https://arxiv.org/abs/2607.01391) | 该论文系统比较了Transformer处理电子健康记录数据时离散、连续和混合三种数值编码策略，发现显式建模数值-概念交互的方法在精度敏感任务上最优，而投影前分箱的混合方法更为稳健通用，且最优分箱数与数据集规模呈幂律关系。 |
| [^253] | [World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video](https://arxiv.org/abs/2607.01202) | 提出 World from Motion 方法，利用视频生成模型修正并补全单目视频的动态 3D 高斯重建，将生成结果蒸馏回统一的高质量动态 3DGS，在 4D 重建与新视角合成上达到新的最先进水平。 |
| [^254] | [SteerVTE: Seamless Video Text Editing with Style and Glyph Control](https://arxiv.org/abs/2606.23254) | SteerVTE通过风格编码器与行级、字符级双粒度字形编码器引导冻结的视频扩散模型，实现了兼具风格一致性与笔画级精度的无缝视频文本编辑。 |
| [^255] | [Attractor Domain Theory: A Mathematical Framework for Cardiovascular Attractor Analysis with Wearable Photoplethysmography (PPG) Validation](https://arxiv.org/abs/2606.22039) | 该论文提出吸引子域理论（ADT），通过延迟重构、遍历与变分等相互关联的数学变换，为心血管吸引子动力学构建了一种包含几何域、遍历域和变分域的三表示算子架构，并以可穿戴光电容积脉搏波（PPG）数据进行了验证。 |
| [^256] | [AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents](https://arxiv.org/abs/2606.20625) | AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。 |
| [^257] | [DeFAb: A Dataset for Grounded Scientific Discovery through Defeasible Abduction](https://arxiv.org/abs/2606.18557) | 该论文提出了基于基因本体构建的DeFAb数据集与符号验证框架，用于评估模型通过可废止溯因修订科学理论的能力，并揭示了现有修订测试无法检验例外泛化性的局限性。 |
| [^258] | [Continuous Audio Thinking for Large Audio Language Models](https://arxiv.org/abs/2606.18273) | 提出连续音频思考（CoAT）框架，通过音频专家蒸馏为大型音频语言模型构建连续潜在思考空间，使其在生成响应前能够组织并利用语音细节、韵律、声音事件、情感和音高等丰富声学信息。 |
| [^259] | [Trust the Right Teacher: Quality-Aware Self-Distillation for GUI Grounding](https://arxiv.org/abs/2606.18101) | 提出面向GUI定位的质量感知自蒸馏方法，通过软门控对教师坐标预测进行空间正确性验证并动态降低不可靠教师信号的权重，从而提升坐标预测的准确性。 |
| [^260] | [Architecture-Aware Reinforcement Learning Makes Sliding-Window Attention Competitive in Math Reasoning](https://arxiv.org/abs/2606.11634) | 该论文提出SWARR两阶段方法——先用监督微调将预训练自注意力模型高效转换为滑动窗口注意力模型，再用on-policy强化学习适配数据与架构的不匹配——使滑动窗口注意力在数学推理任务上具备与自注意力相当的竞争力，同时缓解长上下文推理的二次方计算开销。 |
| [^261] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^262] | [SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents](https://arxiv.org/abs/2606.03692) | SkillPyramid提出了一种基于层次化技能拓扑的技能巩固框架，通过自进化机制使智能体在任务执行中组合、验证并积累新技能，从而实现技能经验的复用与更广泛的任务泛化。 |
| [^263] | [Efficient Exploration for Iterative Nash Preference Optimization](https://arxiv.org/abs/2606.01382) | 论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。 |
| [^264] | [Probe Before You Edit: Probing-Guided Molecular Optimization for LLM Agents in Structure-Based Drug Design](https://arxiv.org/abs/2606.00555) | 该论文提出PROBE框架，模仿药物化学家“先探测后编辑”的做法，让LLM智能体在编辑前通过受控类似物探测口袋-配体复合物的响应，从而在基于结构的药物设计中同时提升配体的结合亲和力与成药性。 |
| [^265] | [StemBind: When MLLMs Get Lost Between Rules and Instances in Abstract Visual Reasoning](https://arxiv.org/abs/2606.00148) | StemBind通过针对同一视觉题干的感知、规则、完整三步诊断问题，将多模态大语言模型在抽象视觉推理中的错误精确定位到具体的推理子步骤。 |
| [^266] | [Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration](https://arxiv.org/abs/2605.31196) | 该论文提出了 TouchSafeBench——首个基于物理仿真构建、用于评估视觉-语言模型“碰撞定位”能力的基准，包含近三千个人机共处场景，支持安全状态分类与即将碰撞预警两项实际部署任务。 |
| [^267] | [Same Answer, Different Confidence: Protocol Sensitivity in LLM Confidence Calibration](https://arxiv.org/abs/2605.27752) | 该研究发现大语言模型的置信度校准比较结果高度依赖于测量协议——仅改变读取标记概率的上下文，就能使言语化置信度与标记置信度之间的差距移动约一个差距本身的大小，表明现有关于两种置信度孰优孰劣的结论并不稳定。 |
| [^268] | [Can Broad Biomedical Knowledge be Contextualized into Scenario-Grounded Propositions?](https://arxiv.org/abs/2605.27082) | 提出了SCENE——一个双层多智能体框架，将知识情境化建模为迭代搜索过程，把广泛的生物医学知识转化为有证据支持、基于具体场景的命题，从而弥合背景知识与具体数据之间的鸿沟。 |
| [^269] | [Efficient Benchmarking Is Just Feature Selection and Multiple Regression](https://arxiv.org/abs/2605.25773) | 该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。 |
| [^270] | [Noise-Robust Financial Numerical Entity Attribute Tagging](https://arxiv.org/abs/2605.24910) | 提出NORA方法，通过任务感知的实例特定加权减轻噪声标签影响，实现金融数值实体的多属性鲁棒标注，并构建了包含1140万实体的大规模基准数据集。 |
| [^271] | [Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality](https://arxiv.org/abs/2605.18971) | 提出O'Prior组合式真实感合成先验（包含分层SCM元生成器、真实感引擎、压力注入模块与防泄漏课程生成协议），在固定架构与算力的条件下证明合成任务分布的设计是决定表格基础模型质量的关键因素。 |
| [^272] | [NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning](https://arxiv.org/abs/2605.16757) | 该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。 |
| [^273] | [VLMs Trace Without Tracking: Diagnosing Failures in Visual Path Following](https://arxiv.org/abs/2605.15672) | 研究发现，前沿视觉语言模型即使在没有交叉干扰的受控设置下也难以稳健地跟随简单连续线条，其根源在于视觉编码阶段就已出现正确路径信息的退化，且模型扩展等常规补救措施收效甚微。 |
| [^274] | [Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use](https://arxiv.org/abs/2605.14038) | 该论文提出了基于各模型自身能力表现的“模型自适应工具必要性”新定义，并发现在算术和事实问答任务中，模型的实际工具调用行为与真实的工具必要性之间存在高达26.5%-54.0%的不匹配，揭示了LLM工具使用中的“知行差距”。 |
| [^275] | [Watermarking Should Be Treated as a Monitoring Primitive](https://arxiv.org/abs/2605.13095) | 论文主张水印不仅是防御移除和伪造的安全技术，更应被视为一种监控原语——同样的水印信号可用于将生成内容追溯至具体生成实体，即使是零比特水印也能被内部观察者用于个体归因，外部观察者在特定配置下也能识别内容来源。 |
| [^276] | [Deep Reasoning in General Purpose Agents via Structured Meta-Cognition](https://arxiv.org/abs/2605.11388) | 提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。 |
| [^277] | [Attractor-Vascular Coupling Theory: Formal Grounding and Empirical Validation for AAMI-Standard Cuffless Blood Pressure Estimation from Smartphone Photoplethysmography](https://arxiv.org/abs/2605.10871) | 该论文提出吸引子-血管耦合理论（AVCT），从PPG吸引子几何结构出发构建可解释的血压预测框架，实现了符合AAMI标准的智能手机无袖带血压估计，校准后收缩压/舒张压平均绝对误差仅2.05/1.67 mmHg。 |
| [^278] | [POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles](https://arxiv.org/abs/2605.07775) | POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。 |
| [^279] | [MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs](https://arxiv.org/abs/2605.07305) | 提出MedAction树状结构蒸馏流水线，通过大语言模型与环境交互合成高质量多轮诊断轨迹，使模型学会在部分证据下主动开具检查、解读结果并更新鉴别诊断，克服了静态单轮诊断评估的局限。 |
| [^280] | [Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching](https://arxiv.org/abs/2605.06474) | 提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。 |
| [^281] | [MoLS: SNR-Guided Module-Wise Learning-Rate Calibration for Large Language Models](https://arxiv.org/abs/2605.05794) | MoLS 通过估计各模块的梯度信噪比并采用逆平方根缩放自动确定相对学习率乘数，以一次性校准取代了大语言模型训练中昂贵的手动模块学习率调参。 |
| [^282] | [Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks](https://arxiv.org/abs/2605.05003) | 该论文将奖励模型评估扩展到偏见、安全、道德和伦理推理四个社会关键领域，发现标准基准性能与社会对齐行为之间仅有不一致的关联，且更强的奖励模型在应对多元社会期望时并不一定表现更好，揭示了偏见缓解与情境忠实性之间的权衡。 |
| [^283] | [Agents and Tokens: Multi-Agent Autonomous Reasoning over a Hydrodynamic Token Store](https://arxiv.org/abs/2605.01102) | 该论文受神经音频编解码器将波形转化为离散词元的启发，提出将近岸水动力模型的海量数据转化为离散词元存储库，使多智能体系统能够在不饱和上下文窗口的前提下对该数据进行自主推理与分析。 |
| [^284] | [MCPHunt: An Evaluation Framework for Cross-Boundary Data Propagation in Multi-Server MCP Agents](https://arxiv.org/abs/2604.27819) | MCPHunt是首个用于隔离检测多服务器MCP智能体中跨信任边界非对抗性凭证传播的受控评估基准，通过金丝雀污点追踪、环境可控覆盖设计和CRS分层三项方法贡献，客观量化了忠实工具组合引发的信息流风险。 |
| [^285] | [Equity Bias: An Ethical Framework for AI Design](https://arxiv.org/abs/2604.21907) | 提出了“公平性偏见”框架，主张偏见不应被消除而应被透明化和可质疑化，并通过三阶段AI生命周期将伽达默尔诠释学与认知不公正理论转化为构建更公平AI系统的实践方法。 |
| [^286] | [Uncertainty-Guided Latent Diagnostic Trajectory Learning for Sequential Clinical Diagnosis](https://arxiv.org/abs/2604.05116) | 该论文提出了一种不确定性引导的潜在诊断轨迹学习（LDTL）框架，通过规划智能体与诊断智能体协作，将诊断动作序列视为潜在路径进行后验优化，从而在缺乏显式监督信息的条件下实现序贯临床诊断中有效的证据获取。 |
| [^287] | [Spatial disparities in simulated patrol detection: a reproducible generative-model audit across Baltimore and Chicago](https://arxiv.org/abs/2603.18987) | 该论文提出了一种结合生成对抗网络巡逻分配与概率检测模型的可复现空间模拟审计框架，利用巴尔的摩和芝加哥的犯罪记录数据量化了模拟巡逻部署在黑人多数区与白人多数区之间的检测率差异。 |
| [^288] | [Learning to Predict, Discover, and Reason in High-Dimensional Event Sequences](https://arxiv.org/abs/2603.16313) | 本文提出将车辆诊断故障码序列视为一种“语言”，通过机器学习方法实现自动预测、错误模式发现与推理，以替代昂贵且易错的人工规则定义过程。 |
| [^289] | [daVinci-Env: Open SWE Environment Synthesis at Scale](https://arxiv.org/abs/2603.13023) | 提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。 |
| [^290] | [MetaKE: Meta-Learning for Knowledge Editing Toward a Better Accuracy-Editability Trade-off](https://arxiv.org/abs/2603.12677) | 提出MetaKE框架，通过元学习将知识编辑中上游表示优化与下游参数优化统一为双层优化问题，解决了两阶段优化割裂导致的编辑不充分或可编辑性下降问题，实现了精度与可编辑性之间更优的权衡。 |
| [^291] | [MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios](https://arxiv.org/abs/2603.09983) | 提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。 |
| [^292] | [VORL-EXPLORE: A Hybrid Learning Planning Approach to Multi-Robot Exploration in Dynamic Environments](https://arxiv.org/abs/2603.07973) | 提出VORL-EXPLORE混合学习规划框架，通过“执行保真度”评估局部可通行性与执行可靠性，并将其耦合进Voronoi边界分配目标与全局A*规划、反应式强化学习导航的仲裁中，显著提升动态环境下多机器人探索的鲁棒性和效率。 |
| [^293] | [CrimeNER: Curated Dataset and Methods for Research and Investigation in Language Intelligence](https://arxiv.org/abs/2603.02150) | 该论文提出了CrimeNER，一个包含1500多份标注文档的犯罪领域命名实体识别数据集，定义了4种粗粒度和21种细粒度犯罪实体类型，并通过传统NER模型和大语言模型的实验验证了数据集质量。 |
| [^294] | [Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents](https://arxiv.org/abs/2602.22402) | 提出了上下文记忆虚拟化（CMV）系统，借鉴操作系统虚拟内存思想，将LLM会话历史建模为带快照、分支、裁剪原语的DAG以实现跨并行会话的上下文复用，并通过三遍结构无损裁剪算法在逐字保留所有对话内容的前提下将令牌开销平均降低20%、最高降低86%。 |
| [^295] | [TensorCommitments: A Lightweight Verifiable Inference for Language Models](https://arxiv.org/abs/2602.12630) | 提出张量承诺（TC）这一张量原生的推理证明方案，通过多变量Terkle树将大语言模型推理与篡改即失效的不可逆承诺绑定，仅需约1%的额外时间开销即可实现可验证推理，并将对定制化LLM攻击的鲁棒性提升高达48%。 |
| [^296] | [Why Does Counterintuitive Likelihood Behavior Rarely Emerge in Tabular Anomaly Detection?](https://arxiv.org/abs/2602.09593) | 本文提出了一个领域无关的反直觉似然现象形式化定义，并通过大规模实验与理论分析证明：与图像领域不同，反直觉的似然行为在表格异常检测中很少出现，且标准化流似然可实现有竞争力的异常检测性能。 |
| [^297] | [Mitigating the Likelihood Paradox in Flow-based OOD Detection: An Entropic Perspective](https://arxiv.org/abs/2602.09581) | 提出免训练的语义比例熵操控方法SPEM，从熵的视角通过对与分布内记忆库相似度低的输入施加更强扰动，增大对数似然差的期望下界，从而缓解基于流的OOD检测中的似然悖论，使似然排序与人类直觉对齐。 |
| [^298] | [StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors](https://arxiv.org/abs/2602.08934) | 本文提出StealthRL强化学习改写攻击框架，在保持语义保真度和文本可读性的前提下，将多种AI文本检测器的平均TPR@1%FPR降至0.024，且攻击可迁移至训练中未见的检测器，揭示了AI文本检测器普遍存在的鲁棒性缺陷。 |
| [^299] | [Clifford Kolmogorov-Arnold Networks](https://arxiv.org/abs/2602.05977) | 提出了克利福德柯尔莫哥洛夫-阿诺德网络（ClKAN），通过随机化拟蒙特卡罗网格生成方法解决高维代数的指数级扩展问题，并引入新的批归一化策略处理可变域输入，实现了在任意克利福德代数空间中灵活高效的函数逼近。 |
| [^300] | [OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning](https://arxiv.org/abs/2601.15351) | OmniSpectra 提出一种以波长为原生坐标的连续坐标掩码自编码器基础模型，通过将对数基上的物理坐标与连续旋转位置编码结合，使宇宙学红移引发的共享平移在相对距离中自然抵消，从而解决了跨异构仪器光谱表示学习中的坐标失配这一结构性难题。 |
| [^301] | [SpatialMem: Metric-Aligned Long-Horizon Video Memory for Language Grounding and QA](https://arxiv.org/abs/2601.14895) | SpatialMem 仅凭随手拍摄的第一人称 RGB 视频，即可构建融合三维几何、语义与语言的度量级可查询记忆，无需特殊传感器就能完成空间关系推理、语言引导检索与长时程视频问答，并在 VSI-Bench 和 OpenEQA 等基准上取得领先表现。 |
| [^302] | [Towards robust long-context understanding of large language model via active recap learning](https://arxiv.org/abs/2601.13734) | 提出主动回顾学习（ARL）框架，通过持续预训练中的针对性序列构建与推理时的自主回溯总结，使大语言模型在段落间建立递归记忆机制，显著提升长上下文理解能力（RULER提升最高达129.7%）。 |
| [^303] | [Attention-space Contrastive Guidance for Efficient Hallucination Mitigation in LVLMs](https://arxiv.org/abs/2601.13707) | 提出注意力空间对比引导（ACG），一种无需训练的单次前向传播方法，通过在自注意力层构建图像条件与纯文本对比注意力路径并结合轻量级正交投影，高效缓解大型视觉-语言模型的幻觉问题。 |
| [^304] | [PAIR-SAFE: A Paired-Agent Approach for Runtime Auditing and Refining AI-Mediated Mental Health Support](https://arxiv.org/abs/2601.12754) | 提出PAIR-SAFE双智能体框架，由基于临床验证MITI-4标准的评判者智能体实时审计并指导修正LLM生成的心理健康支持回复，显著提升了伙伴关系、协作性和整体关系质量。 |
| [^305] | [The Role of Quantum in Hybrid Quantum-Classical Neural Networks: An Empirical Assessment](https://arxiv.org/abs/2601.04732) | 本文通过理想无噪声模拟下的严谨统计研究，在三个医学二分类任务上系统评估了常用混合量子-经典神经网络架构，以揭示量子组件对模型整体性能的实际贡献。 |
| [^306] | [Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis](https://arxiv.org/abs/2601.03018) | 提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。 |
| [^307] | [MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data](https://arxiv.org/abs/2512.13297) | 提出首个用于评估大型多模态模型和智能体框架在多模态医学图像数据上进行多步骤医学洞察发现能力的基准MedInsightBench，包含332个精心标注的医学案例，并揭示现有模型表现仍然有限。 |
| [^308] | [Demo: Generative AI helps Radiotherapy Planning with User Preference](https://arxiv.org/abs/2512.08996) | 本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。 |
| [^309] | [Testing Transformer Learnability on the Arithmetic Sequence of Rooted Trees](https://arxiv.org/abs/2512.01870) | 该论文将自然数通过迭代素数分解映射为有根树构成的确定性算术文本，并证明从头训练的GPT-2模型能以约0.4的词准确率学习该序列，且在远超训练范围的测试数据上保持稳定的外推性能。 |
| [^310] | [EvalCards: A Framework for Standardized Evaluation Reporting](https://arxiv.org/abs/2511.21695) | 本文提出 EvalCards 框架，围绕可复现性、可访问性和可问责性三个核心维度规范 AI 模型评估报告，并提供简洁评估摘要的初步格式，以推动未来评估报告标准的建立。 |
| [^311] | [Beyond the Monolithic Scheduler: Lightweight Adaptive Policy Switcher for Process Scheduling Optimization](https://arxiv.org/abs/2511.11628) | 本文提出轻量级自适应策略切换器LAPS，通过在预定义专家策略间动态切换并将繁重的动态权衡计算转移至离线阶段，在不引入AI辅助调度高开销的前提下，实现了对边缘系统动态混合工作负载的高效进程调度优化。 |
| [^312] | [Expressivity Results for Structured Argumentation Frameworks with Uncertain Rules and Premises](https://arxiv.org/abs/2510.18631) | 该论文提出了一个同时适用于抽象与结构化论证形式体系的直观比较表达能力概念，通过将论证的不确定性根植于规则与前提层面，证明了此类结构化不确定性论证框架与抽象不确定性论证模型之间表达能力的正反两方面的比较结果。 |
| [^313] | [Human-AI Interactions and the Psychological Triad: A Multi-Domain Survey of Cognitive, Behavioral, and Emotional Impacts](https://arxiv.org/abs/2510.17753) | 本文提出基于心理三元组（认知、行为、情感）的TRIAD概念框架，系统综述了人机交互的双面影响：AI虽能增强记忆、创造力和参与度并缓解压力，但也带来批判性思维下降、技能退化和情感依赖等系统性风险。 |
| [^314] | [Dynamic Relational Priming Improves Transformer in Multivariate Time Series](https://arxiv.org/abs/2509.12196) | 本文提出动态关系启动注意力机制（prime attention），通过为每个token对交互定制动态表示，显著改进了Transformer在多元时间序列建模中捕捉异构通道间依赖关系的能力。 |
| [^315] | [AU-Harness: An Open-Source Toolkit for Efficient and Unified Evaluation of AUdio-LLMs](https://arxiv.org/abs/2509.08031) | AU-Harness是一个开源的音频大语言模型统一评估框架，通过高效流水线将评估处理时间减少77%，并解决了现有工具在多轮对话支持不足和框架缺乏标准化方面的关键局限。 |
| [^316] | [MEDMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph](https://arxiv.org/abs/2505.17214) | 提出了医学多模态知识图谱MEDMKG，通过融合MIMIC-CXR的视觉与文本数据以及UMLS的结构化临床知识构建图谱，并引入邻居感知过滤算法保证图谱质量，为医学知识利用提供了首个多模态知识图谱基准。 |
| [^317] | [Deep Learning Warm Starts for Trajectory Optimization on the International Space Station](https://arxiv.org/abs/2505.05588) | 本研究在国际空间站上首次实现在轨演示，利用神经网络热启动加速Astrobee自由飞行机器人的轨迹优化，通过学习顺序凸规划问题的结构来实现满足安全约束的星载实时轨迹生成。 |
| [^318] | [Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training](https://arxiv.org/abs/2505.02360) | 本文提出一种纯粹通过控制 $l^p$ 训练范数来缓解快速对抗训练中灾难性过拟合的无噪声新方案，并揭示灾难性过拟合源于高度集中的梯度与激进范数约束的相互作用。 |
| [^319] | [Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner](https://arxiv.org/abs/2412.20662) | 本文提出了一个评估视觉大语言模型表格识别能力的分层设计基准，并创新性地提出邻居引导的工具链推理器（NGTR）框架，通过整合轻量级视觉工具和迁移相似邻居的工具选择经验来缓解低质量图像输入的瓶颈问题。 |
| [^320] | [Power- and Fragmentation-aware Online Scheduling for GPU Datacenters](https://arxiv.org/abs/2412.17484) | 该论文提出了PWR，一种面向GPU数据中心的功耗与碎片感知在线调度策略，在不预知未来任务到达的情况下降低功耗并缓解GPU碎片化问题。 |
| [^321] | [Quantized Approximately Orthogonal Recurrent Neural Networks](https://arxiv.org/abs/2402.04012) | 本文提出了量化近似正交循环神经网络（QORNNs）来解决正交循环神经网络（ORNNs）中参数过多的问题，采用一种后训练量化策略和三种融入正交约束和量化权重的量化感知训练算法，取得了与s相似的结果。 |

# 详细

[^1]: SceneBloom：面向具身学习的具有真实且多样化布局的智能体化室内环境生成

    SceneBloom: Agentic Indoor Environment Generation with Realistic and Diverse Layouts for Embodied Learning

    [https://arxiv.org/abs/2610.32978](https://arxiv.org/abs/2610.32978)

    SceneBloom是一个结合视觉布局采样与契约引导3D编译的智能体框架，利用扩散先验在功能区域内生成家具布局并通过几何适配与姿态校正实现物理有效的室内场景，在五种房间类型上取得了最高的视觉质量与物理稳定性，且是唯一无网格碰撞与越界物体的方法。

    

    训练可泛化的具身智能体需要真实、多样且可直接用于仿真的室内环境。语言引导的方法支持语义规划与对象级推理，但精确的空间定位仍然困难。视觉生成模型对合理的布局拥有丰富的先验，但其输出需要几何推理才能转化为物理上有效的3D场景。我们提出SceneBloom，一个将视觉布局采样与契约引导的3D编译相结合的智能体框架。分层采样器利用扩散先验在功能区域内生成家具布置，编译器随后利用从每个方案中提取的空间关系来指导几何适配与局部姿态校正。在五种房间类型上，SceneBloom在所比较的方法中取得了最高的视觉质量与物理稳定性，并且是唯一未检测到网格碰撞与越界物体的方法。

    arXiv:2610.32978v1 Announce Type: cross  Abstract: Training generalizable embodied agents requires realistic, varied, and simulation-ready indoor environments. Language-guided methods support semantic planning and object-level reasoning, but precise spatial grounding remains difficult. Visual generative models offer rich priors over plausible arrangements, yet their outputs require geometric reasoning to become physically valid 3D scenes. We introduce SceneBloom, an agentic framework that combines visual layout sampling with contract-guided 3D compilation. A hierarchical sampler uses diffusion priors to generate furniture arrangements within functional zones. A compiler then uses relations extracted from each proposal to guide geometric fitting and local pose correction. Across five room types, SceneBloom attains the highest visual quality and physical stability among the compared methods and is the only method with no detected mesh collisions and out-of-bounds objects. Its layouts are
    
[^2]: RobustRLlib：鲁棒强化学习算法的统一库与基准测试

    RobustRLlib: A Unified Library and Benchmark for Robust Reinforcement Learning Algorithms

    [https://arxiv.org/abs/2610.32971](https://arxiv.org/abs/2610.32971)

    本文提出RobustRLlib，一个集成了17种鲁棒在线、离线与安全RL算法的统一库与基准测试平台，并通过273种扰动配置的系统评估绘制了当前鲁棒RL算法的能力图谱，揭示了其有效范围与鲁棒性差距。

    

    鲁棒强化学习（RL）对于可靠的真实世界部署至关重要，因为策略可能在多种偏移和扰动下失效。现有的鲁棒RL算法提供了丰富的机制和原理，但两大障碍限制了其更广泛的应用潜力：算法可复用性有限，以及实证能力边界不清晰。我们提出了RobustRLlib，一个以算法为中心的库和基准测试平台，以应对这两个挑战。它集成了17种代表性的鲁棒在线、离线和安全RL算法，并采用统一接口以显著提升可复用性，同时提供了一个增强的评估平台，涵盖多样化的真实世界偏移以及先进的人形机器人和视觉-语言-动作任务。通过对273种扰动配置的系统评估，本工作提供了当前鲁棒RL算法的最新能力图谱，揭示了这些算法在哪些场景下仍然有效，以及在哪些方面仍存在显著的鲁棒性差距。

    arXiv:2610.32971v1 Announce Type: cross  Abstract: Robust reinforcement learning (RL) is critical for reliable real-world deployment, where policies may fail under diverse shifts and disruptions. Existing robust RL algorithms offer a rich set of mechanisms and principles, yet two major barriers limit their broader potential: limited algorithm reusability and an unclear empirical capability boundary. We introduce RobustRLlib, an algorithm-centric library and benchmark that addresses both challenges. It integrates 17 representative robust online, offline, and safe RL algorithms under a unified interface for substantially improved reusability, together with an enhanced evaluation platform spanning diverse real-world shifts and advanced humanoid and vision-language-action tasks. Systematic evaluation over 273 perturbation configurations provides an up-to-date capability map of current robust RL algorithms, revealing where they remain effective and where substantial robustness gaps persist.
    
[^3]: DexPort：一种用于灵巧手数据采集的手持式技能接口

    DexPort: A Hand-Held Skill Interface for Dexterous Hand Data Collection

    [https://arxiv.org/abs/2610.32969](https://arxiv.org/abs/2610.32969)

    DexPort提出了一种面向高自由度灵巧手的便携式手持数据采集接口，通过采集与部署共用相同手部硬件来避免实体差异，并借助一小组可选择的技能来协调手指运动，从而大幅降低高质量灵巧操作演示数据的采集成本。

    

    收集高质量的演示数据仍然是学习灵巧操作的核心瓶颈。遥操作能够记录可执行的机器人动作，但它需要完整的机器人设备，并把协调众多手指关节的负担压在操作者身上。人类演示更容易收集，但其动作必须经过估计并重定向到机器人，而且不能直接提供机器人动作。在本文中，我们提出了DexPort，一个面向高自由度机器人手的便携式数据采集接口。它在数据采集与部署时使用相同的手部硬件，从而避免了手部实体差异（embodiment gap），并同时记录指令关节目标和测量关节状态。我们的关键观察是：一小组技能就可以支持广泛的灵巧任务。在采集过程中，操作者移动手持接口来指定机器人手腕运动，并选择或调整协调手指运动的技能。（原文摘要在此处截断）

    arXiv:2610.32969v1 Announce Type: cross  Abstract: Collecting high-quality demonstrations remains a central bottleneck for learning dexterous manipulation. Teleoperation records executable robot actions, but it requires a complete robot setup and places the burden of coordinating many finger joints on the operator. Human demonstrations are easier to collect, but their motions must be estimated and retargeted to the robot, and they do not directly provide robot actions. In this paper, we introduce DexPort, a portable data-collection interface for high-DoF robot hands. It uses the same hand hardware for collection and deployment, avoiding a hand embodiment gap, and records both commanded joint targets and measured joint states. Our key observation is that a small set of skills can support a wide range of dexterous tasks. During collection, the operator moves the hand-held interface to specify the robot wrist motion and selects or adjusts skills that coordinate the finger motion. This rem
    
[^4]: 超越离散标记：在医学视觉-语言模型中保持数值连续性

    Beyond Discrete Tokens: Preserving Numerical Continuity in Medical Vision-Language Models

    [https://arxiv.org/abs/2610.32948](https://arxiv.org/abs/2610.32948)

    提出 HAD-VLM，一种在共享骨干中融合自回归文本生成与扩散式连续测量预测的混合医学视觉-语言模型，通过扩散过程直接预测卡尺端点坐标与测量长度等连续数值，解决了传统离散标记预测与连续测量目标之间的根本不匹配问题。

    

    arXiv:2610.32948v1，公告类型：交叉（cross）。目标：开发一个统一的视觉-语言模型，用于基于提示条件的定量超声测量，以解决连续测量目标与基于离散标记的数值预测之间的不匹配问题。方法：本文提出 HAD-VLM，一种混合自回归-扩散框架，在共享骨干网络中将自回归文本生成与连续测量预测相结合。视觉提示调优使视觉编码器适配超声图像。扩散过程预测一个由卡尺端点坐标和测量长度组成的五分量表示。提出一种结合坐标-长度一致性机制与集成预测的后处理策略，用于优化测量估计。我们在 EchoNet-LVH 上评估了四项超声心动图测量任务，并对医学问答和 X 射线椎体测量进行了额外评估。结果：HAD（摘要在此处截断）

    arXiv:2610.32948v1 Announce Type: cross  Abstract: Objective: To develop a unified vision-language model for prompt-conditioned quantitative ultrasound measurement that addresses the mismatch between continuous measurement targets and discrete token-based numerical prediction. Methods: This article presents HAD-VLM, a hybrid autoregressive-diffusion framework that combines autoregressive text generation with continuous measurement prediction within a shared backbone. Visual prompt tuning adapts the vision encoder to ultrasound images. A diffusion process predicts a five-component representation comprising caliper endpoint coordinates and measurement length. A post-processing strategy combining the proposed coordinate-length consistency mechanism with ensemble prediction is introduced to refine measurement estimates. We evaluate four echocardiographic measurement tasks on EchoNet-LVH, with additional assessments of medical question answering and X-ray vertebral measurement. Results: HAD
    
[^5]: 智能变现：大语言模型平台上的广告与订阅设计

    Monetizing Intelligence: Advertising and Subscription Design on LLM Platforms

    [https://arxiv.org/abs/2610.32945](https://arxiv.org/abs/2610.32945)

    该论文研究大语言模型平台的广告与订阅联合定价设计，发现广告既能资助更广泛的服务接入，但也会导致用户效用两极分化，而设置适度的广告上限可以最大化消费者剩余。

    

    引入广告的大语言模型（LLM）平台必须决定广告收入应资助多少能力，以及消费者需要支付多少费用才能免于广告。我们研究了一个垄断平台，在广告会降低用户价值、但更好的服务能提升广告回报的条件下，如何联合选择有成本约束的响应质量、广告强度和订阅价格。研究得出三个发现。第一，当部分广告支持型方案与无广告方案共存时，广告可以为那些在纯订阅模式下无法获得服务的消费者提供资金支持。中间档位方案较低的实用质量削弱了高级用户可用的替代选择，从而允许平台在无广告服务不变的情况下收取更高价格。部分完全由广告支持的用户获得了效用，而部分广告支持和无广告用户则遭受损失：更广泛的接入可以与效用两极分化并存。第二，适度的广告上限可以通过保留部分收入来最大化消费者剩余（摘要在此处被截断）……

    arXiv:2610.32945v1 Announce Type: cross  Abstract: Large language model (LLM) platforms introducing advertising must decide how much capability ad revenue should fund and what consumers should pay to avoid ads. We study a monopolistic platform that jointly chooses costly response quality, advertising intensity, and subscription prices when advertising reduces user value but better service raises advertising returns. Three findings follow. First, when partly ad-supported and ad-free plans coexist, advertising can finance useful service for consumers who receive none under pure subscription. Lower effective quality in middle plans weakens the alternatives available to premium users, allowing higher prices for unchanged ad-free service. Some fully ad-supported users gain utility, while partly ad-supported and ad-free users lose: broader access can coexist with utility polarization. Second, an intermediate advertising cap can maximize consumer surplus by preserving some revenue that financ
    
[^6]: Yalda-Embedding：一个具有竞争力的波斯语检索模型的开放开发

    Yalda-Embedding: Open Development of a Competitive Persian Retrieval Model

    [https://arxiv.org/abs/2610.32933](https://arxiv.org/abs/2610.32933)

    该论文提出Yalda Embedding，一个基于Qwen3-Embedding-0.6B适配、使用超百万查询-文档对训练的5.958亿参数波斯语检索模型，在FaMTEB的八个波斯语检索任务上超越所有同规模0.6B模型，并接近4B规模模型的性能。

    

    尽管多语言文本表示技术不断进步，但波斯语的有效密集检索仍然受到特定语言训练资源和紧凑嵌入模型可用性有限的制约。我们提出了Yalda Embedding，这是一个5.958亿参数的波斯语检索模型，基于Qwen3-Embedding-0.6B适配而来，使用了一百多万个查询-文档对进行训练。在FaMTEB的八个波斯语检索任务上，该模型在所评估的0.6B模型中取得了最高的宏平均值，得分0.5488，在每个任务上都优于Qwen3-Embedding-0.6B和F2LLM-v2-0.6B。它还接近Qwen3-Embedding-4B的性能，在四个任务上获胜，在一个任务上打平，宏平均值仅落后0.0075。一项增量消融研究考察了对比学习批量大小、优化稳定性以及从F2LLM-v2-8B进行的关系知识蒸馏。在单轮训练设置下，仅对比微调就能带来适度的平均改进……

    arXiv:2610.32933v1 Announce Type: cross  Abstract: Despite advances in multilingual text representations, effective dense retrieval for Persian remains constrained by the limited availability of language-specific training resources and compact embedding models. We present Yalda Embedding, a 595.8-million-parameter Persian retrieval model adapted from Qwen3-Embedding-0.6B using more than one million query-document pairs. On eight Persian retrieval tasks from FaMTEB, the model achieves the highest macro-average among the evaluated 0.6B models, scoring 0.5488 and outperforming Qwen3-Embedding-0.6B and F2LLM-v2-0.6B on every task. It also approaches Qwen3-Embedding-4B, winning on four tasks, tying on one, and trailing by 0.0075 in macro-average. An incremental ablation study examines contrastive batch size, optimization stabilization, and relational knowledge distillation from F2LLM-v2-8B. Under the one-epoch training setup, contrastive fine-tuning alone yields a modest mean improvement ov
    
[^7]: SemantEq：形式化命题的忠实双向等价

    SemantEq: Faithful Bidirectional Equivalence of Formal Statements

    [https://arxiv.org/abs/2610.32919](https://arxiv.org/abs/2610.32919)

    提出 SemantEq，一种更有原则的形式化命题等价检验方法，要求命题间只能通过一系列允许的逻辑变换步骤相互转化，从而避免因证明器过强导致逻辑等价检验被平凡通过的问题。

    

    命题自动形式化是将数学从自然语言翻译成 Lean 4 等证明助手语言的过程，如今模型已经可以大规模地完成这一任务。随着证明器变得越来越强，正确性的负担从证明转移到了命题本身，即形式化命题是否忠实地翻译了其源文本。给定一个可信的参考形式化，最自然的检验标准是逻辑等价，但逻辑等价证明的内容过多：对于库中已证明的任意两个命题 P 和 Q，P ↔ Q 都会平凡地成立，因为证明器可以分别证明 P 和 Q，并通过 P ↔ True ↔ Q 完成推导。证明器越强，通过检验的无关命题对就越多。早期的方法通过限制证明器的能力，并检查每个证明是否引用了另一个命题。我们提出 SemantEq，一种更有原则的等价关系，它要求每个命题只能通过一系列允许的步骤（如逻辑重排等）转化为另一个命题……

    arXiv:2610.32919v1 Announce Type: new  Abstract: Statement autoformalization translates mathematics from natural language into a proof assistant such as Lean 4, and models now do it at scale. As provers get stronger, the burden of correctness shifts from the proof to the statement, that is, to whether the formal statement faithfully translates its source. Given a trusted reference formalization, the natural test is logical equivalence, and it proves too much. $P \leftrightarrow Q$ holds trivially for any two statements $P$ and $Q$ that the library proves, because a prover can prove each separately and pass through $P \leftrightarrow \mathrm{True} \leftrightarrow Q$. The stronger the prover, the more unrelated pairs pass. Earlier methods restrict the prover and check that each proof names the other statement. We introduce SemantEq, a more principled relation that instead requires each statement to turn into the other only through a chain of allowed steps, such as logical rearrangement a
    
[^8]: 绿色代码还是昂贵代码？评估基于大语言模型的迭代式代码优化的能源成本

    Green Code or Costly Code? Evaluating the Energy Costs of Iterative LLM-Based Code Optimizations

    [https://arxiv.org/abs/2610.32891](https://arxiv.org/abs/2610.32891)

    该研究通过对11个开源模型在EffiBench全部1000个问题上的硬件级能耗实测，挑战了“代码跑得更快就一定更省电”的假设，发现LLM迭代优化代码的推理能耗（0.79–8.68 MJ，相当于250–2800次GPT-4.1查询）未必能被后续执行时间的节省所抵消。

    

    大语言模型（LLM）正被越来越多地用于优化代码性能。这种做法通常假设执行速度的提升会自然转化为更低的总体能耗。我们通过实证研究对这一假设提出质疑：我们测量了由LLM自我改进流水线生成的性能优化代码，相对于标准的人类参考解决方案是否也具有能效优势，以及优化过程本身的能源成本能否被后续的执行时间节省所抵消。我们在EffiBench（一个最先进的性能基准）的全部1,000个问题上，评估了11个开源模型在四种解码配置下的表现，并全程收集了逐进程的硬件级能耗测量数据。推理能耗在不同模型与配置组合之间跨越0.79至8.68兆焦耳（MJ），大致相当于250至2,800次GPT-4.1查询的能耗。限制最大输出token数量具有最大的……

    arXiv:2610.32891v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are increasingly used to optimize code for performance. This practice often assumes that improvements in execution speed naturally translate into overall lower energy consumption. We challenge this assumption empirically by measuring whether performance-optimized code produced by LLM self-refinement pipelines is also energy-efficient relative to canonical human reference solutions, and whether the energy cost of the optimization process can be offset by subsequent execution-time savings. We evaluate 11 open-source models across four decoding configurations on all 1,000 problems of EffiBench, a state-of-the-art performance benchmark, while collecting per-process, hardware-level energy measurements throughout. Inference energy spans 0.79 to 8.68 MJ across model-variant pairs, which is roughly equivalent to the energy of 250 to 2800 GPT-4.1 queries. Limiting the maximum number of output tokens has the largest 
    
[^9]: 通过单纯形约束表示弥合模态差距

    Closing the Modality Gap via Simplex-Constrained Representations

    [https://arxiv.org/abs/2610.32890](https://arxiv.org/abs/2610.32890)

    该论文提出将多模态对比模型的嵌入域从单位球面改为概率单纯形，利用其非负、单位质量的结构约束作为跨模态校准，在五个检索基准上将质心差距减少97%至99%，实现真正的语义对齐。

    

    尽管像CLIP和CLAP这样的多模态对比模型能够成功学习共享表示，它们的嵌入通常表现出持续的模态差距。我们提出通过一种纯几何方法来解决这种分离问题：将嵌入域从单位球面 $S^{d-1}$ 改变为概率单纯形 $\Delta^{d-1}$。尽管两个流形具有相同的内在维度，但单纯形起到了结构先验的作用；其非负、单位质量的约束作为一种跨模态校准，迫使两种模态锚定在同一坐标系中。通过附加轻量级的softmax适配器并使用全变差相似度，我们在五个多模态检索基准上将质心 $\ell_2$ 差距减少了97%至99%。这证实了单纯形几何能够诱导真正的语义对齐，其检索性能与标准无约束模型相当或更优。

    arXiv:2610.32890v1 Announce Type: new  Abstract: While multimodal contrastive models like CLIP and CLAP successfully learn shared representations, their embeddings typically exhibit a persistent \emph{modality gap}. We propose addressing this separation through a purely geometric approach: changing the embedding domain from the unit sphere, $S^{d-1}$, to the probability simplex, $\Delta^{d-1}$. Although both manifolds share the same intrinsic dimension, the simplex acts as a structural prior; its nonnegative, unit-mass constraints serve as a cross-modal calibration that forces both modalities to ground themselves in the same coordinate system. By attaching lightweight softmax adapters and using Total Variation similarity, we reduce the centroid $\ell_2$ gap by 97--99\% across five multimodal retrieval benchmarks. This confirms that simplicial geometry induces genuine semantic alignment, matching or exceeding the retrieval performance of standard unconstrained models.
    
[^10]: NgramRL：通过感知回滚的N-gram投机解码加速结构化生成的强化学习后训练

    NgramRL: Accelerating RL Post-Training for Structured Generation via Rollout-Aware N-gram Speculative Decoding

    [https://arxiv.org/abs/2610.32881](https://arxiv.org/abs/2610.32881)

    NgramRL利用同一提示词的近期回滚历史构建n-gram起草器进行投机解码，无需辅助模型或GPU内存开销，即可显著加速强化学习后训练中占训练时间70%以上的回滚生成过程。

    

    强化学习（RL）后训练能够提升大型语言模型的推理能力，但其成本主要由自回归回滚生成所主导，这部分可能超过训练时间的70%。现有的投机解码方法不太适合这种场景：基于模型的起草器会消耗稀缺的GPU内存，并可能随着策略的演进而失去对齐；而现有的免模型方法则没有显式利用训练轮次间的重复性。我们观察到，同一提示词的近期回滚结果提供了一种高质量的、自我更新的起草来源。基于这一洞察，我们提出了NgramRL，一种从有界的近期回滚历史中构建的逐提示词n-gram起草器。NgramRL不需要辅助模型、起草器训练或常驻GPU的起草器状态，同时目标模型验证保证了回滚分布不变。为了隐藏CPU缓存查找和调度的开销，我们进一步引入了自适应扇出预取器。

    arXiv:2610.32881v1 Announce Type: new  Abstract: Reinforcement learning (RL) post-training improves the reasoning capabilities of large language models, but its cost is dominated by autoregressive rollout generation, which can exceed 70% of training time. Existing speculative-decoding methods are poorly suited to this setting: model-based drafters consume scarce GPU memory and may become misaligned as the policy evolves, while existing model-free methods do not explicitly exploit repetition across training epochs. We observe that recent rollouts of the same prompt provide a high-quality, self-updating draft source. Based on this insight, we propose NgramRL, a per-prompt n-gram drafter constructed from bounded recent rollout history. NgramRL requires no auxiliary model, drafter training, or persistent GPU-resident drafter state, while target-model verification preserves the rollout distribution. To hide CPU cache lookup and scheduling, we further introduce an adaptive fan-out prefetcher
    
[^11]: 通过逐层误差模型搜索推进基于SVD的大语言模型压缩

    Advancing SVD-based LLM Compression via Layer-Wise Error Model Search

    [https://arxiv.org/abs/2610.32878](https://arxiv.org/abs/2610.32878)

    提出LEMS逐层误差建模搜索和KFAC-SVD方法，通过融合局部与全局层重要性的误差代理实现更精确的全局秩分配，并利用逐token统计量提升SVD分解质量，从而有效改进大语言模型的低秩压缩效果。

    

    基于低秩SVD的压缩为降低大语言模型（LLM）的计算成本提供了一种强大的策略。然而，现有方法面临两个关键局限：(i) 全局秩分配方面，未经校准的误差代理无法捕捉复杂的误差传播；(ii) 分解质量方面，基于Fisher的估计器存在严重的秩坍缩问题。在本工作中，我们通过引入逐层误差建模搜索（LEMS）和KFAC-SVD来解决这些局限。LEMS通过引入一个逐层误差代理来改进秩分配，该代理综合了局部与全局层重要性以及传播偏差，并通过整数线性规划（ILP）公式实现有效的全局秩分配。KFAC-SVD利用逐token统计量来提升分解质量，缓解了以往基于Fisher的SVD方法中观察到的秩不足问题。在Mistral、Qwen3和Llama 3模型家族上的实验表明，LEMS持续超越现有的搜索策略。

    arXiv:2610.32878v1 Announce Type: new  Abstract: Low-rank SVD-based compression offers a powerful strategy to reduce the computational costs of LLMs. However, existing methods face two key limitations: (i) global rank allocation, where uncalibrated error proxies fail to capture complex error propagation, and (ii) decomposition quality, where Fisher-based estimators suffer from severe rank collapse. In this work, we address these limitations by introducing Layer-wise Error Modeling Search (LEMS) and KFAC-SVD. LEMS advances rank allocation by introducing a layer-wise error surrogate that integrates local and global layer importance alongside a propagation bias, enabling effective global rank allocation via an ILP formulation. KFAC-SVD improves decomposition quality by utilizing token-wise statistics, mitigating the rank deficiency observed in prior Fisher-based SVD approaches. Across Mistral, Qwen3, and Llama 3 model families, we show that LEMS consistently outperforms existing search st
    
[^12]: “真”与“假”相加等于一吗？LLM智能体观点模拟中的互补一致性

    Do "True" and "False" Add Up to One? Complement Consistency in LLM-Agent Opinion Simulations

    [https://arxiv.org/abs/2610.32874](https://arxiv.org/abs/2610.32874)

    该研究首次系统揭示了LLM智能体在观点模拟中违反互补一致性——对同一命题及其否定所报告的概率之和超过1（如Qwen3-4B平均达1.18），且该偏差远大于扰动噪声两到十倍，意味着此类模拟结果可能存在系统性扭曲。

    

    在LLM智能体观点模拟中，每个智能体在看到自己之前的概率以及邻居所报告的概率后，报告某条声明为真的概率，该数字将成为它在下一轮中的状态。如果这个数字描述的是智能体的立场而非它所看到的句子本身，那么将该声明改写为其否定形式，并把每个概率p替换为1-p，理应得到互补的答案，即两次报告之和应为1。我们使用三个小型开源模型（Qwen3-4B、Llama-3.2-3B和Gemma-3-4B）对此进行了测试。在从一次运行中的网络模拟提取的2,304个智能体状态上，Qwen的成对报告之和平均为1.18；而在来自AVeriTeC的48条真实事实核查声明上，其结果同样超过1。一个包含24个双结果世界（每个世界恰好有两种可能结局）的受控基准测试表明，三个模型都存在同样的失效现象。这一偏差比相同均值扰动所引起的变化大两到十倍。

    arXiv:2610.32874v1 Announce Type: new  Abstract: In LLM-agent opinion simulations, each agent reports the probability that a claim is true after seeing its own previous probability and the probabilities reported by its neighbors, and that number becomes its state in the next round. If the number describes the agent's position rather than the sentence it was shown, then rewriting the claim as its complement and replacing every probability p with 1 - p should return the complementary answer, so the two reports should sum to one. We test this with three small open models: Qwen3-4B, Llama-3.2-3B and Gemma-3-4B. On 2,304 agent states taken from a running network simulation, Qwen's paired reports sum to 1.18 on average, and on 48 real fact-checking claims from AVeriTeC they again exceed one. A controlled benchmark of 24 worlds, each with exactly two possible outcomes, shows the same failure in all three models. The discrepancy is two to ten times larger than the change produced by a same-mea
    
[^13]: 从刻意构建的活动世界系统表征到对话：面向强语言型ASI的五级通信架构

    From Deliberately Constructed Systemic Representations of the World of Activity to Dialogue: A Five-Level Communication Architecture for Strong Language-Based ASI

    [https://arxiv.org/abs/2610.32869](https://arxiv.org/abs/2610.32869)

    论文提出五级消息形成与理解架构，主张语言模型的上下文无法呈现话语形成的基础——这些基础由习得的图式和刻意构建的活动世界系统表征预先建立，并据此为强语言型ASI设计了从系统表征到文本的通信架构。

    

    本文提出了在ASI所参与的思维与活动中，消息形成与理解的五级组织结构。其核心论点是：语言模型的上下文从根本上无法将话语形成的基础呈现出来以供检验和验证。这些基础是由后天习得的图式、活动世界的系统表征，以及在其中被视为真实而接受的关联所预先建立的。无论是上下文还是经过训练的LLM权重，其本身都不能证明存在刻意构建的系统表征。这五个层级区分了：活动世界的系统表征、从这些表征中切分出的特定情境、对该情境的叙述性描述、消息的完整意义，以及文本。一个情境的图式模型并不等同于关于它的故事。接受者通过拼接从缩略的文本中重构出完整意义……

    arXiv:2610.32869v1 Announce Type: new  Abstract: This paper proposes a five-level organization of message formation and understanding within thinking and activity in which ASI participates. Its central thesis is that a language model's context is fundamentally incapable of presenting the foundations of utterance formation for examination and verification. These foundations are established in advance by acquired schemas, systemic representations of the world of activity, and connections accepted as true within them. Neither context nor trained LLM weights, in themselves, attest to the presence of deliberately constructed systemic representations. The five levels distinguish systemic representations of the world of activity, a particular situation cut out from those representations, its narrative description, the full sense of a message, and text. A situation's schema-model is not identical to a story about it. A recipient reconstructs the full sense from an abbreviated text through join
    
[^14]: 基于TopKLoRA的可解释稀疏微调实现忠实的电路发现

    Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery

    [https://arxiv.org/abs/2610.32866](https://arxiv.org/abs/2610.32866)

    本文提出了设计上即可解释的稀疏微调方法TopKLoRA，通过在每个token上仅激活k个最大的适配器潜在变量、纯粹以下游损失训练，使其潜在变量在功能上具有因果性，从而能够忠实地发现休眠代理后门等模型内部电路。

    

    摘要（arXiv:2610.32866v1，公告类型：cross）：在来源未知的数据集上进行微调时，如果数据中包含被投毒的样本，可能会在被训练的模型中植入后门。后门难以定位和移除，并且可以在安全训练之后持续存在。虽然参数高效微调方法能够捕捉训练过程中引入的全部参数变化，但它们不易于解释。我们提出了TopKLoRA的完整方法，该方法扩展了Masiak等人（2025）的初步提案，这是一种设计上即可解释的LoRA变体，其中在每个token上只有$k$个最大的适配器潜在变量处于激活状态。与稀疏自编码器（SAE）不同，TopKLoRA完全基于下游损失进行训练，其中优化压力会促使模型学习在功能上具有因果作用的潜在变量。我们在一个休眠代理（sleeper-agent）后门案例研究中展示了这种稀疏LoRA适配器：我们通过在投毒的指令遵循数据集上微调gemma-2-2b、Qwen2.5-1.5B和Qwen2.5-7B来创建模型生物体（model organisms）。对于每个生物体，我们识别并……（摘要原文在此处截断）

    arXiv:2610.32866v1 Announce Type: cross  Abstract: Fine-tuning on datasets of unknown origin can create a backdoor in the trained model when the data includes poisoned examples. Difficult to localise and remove, backdoors can persist through safety training. While parameter-efficient fine-tuning methods capture the full parameter changes introduced during training, they are not easily interpretable. We present the full method of TopKLoRA, extending the preliminary proposal of Masiak et al. (2025), an interpretable-by-design LoRA variant in which only the $k$ largest adapter latents are active at each token. Unlike SAEs, TopKLoRA is trained purely on downstream loss, where optimisation pressure incentivises learning functionally causal latents. We demonstrate this sparse LoRA adapter in a sleeper-agent backdoor case study where we create model organisms by fine-tuning gemma-2-2b, Qwen2.5-1.5B, and Qwen2.5-7B on a poisoned instruction-following dataset. For each organism, we identify and
    
[^15]: PoKER：一种基于随机目标机器学习的空空导弹杀伤概率估计模型

    PoKER: A Probability of Kill Estimation Rate Model for Air-to-Air Missiles Using Machine Learning on Stochastic Targets

    [https://arxiv.org/abs/2610.32864](https://arxiv.org/abs/2610.32864)

    PoKER是一种利用机器学习整合目标随机行为与导弹脱靶距离、从而更准确预测超视距空战中空空导弹杀伤概率的新型概率模型，有望成为飞行员改善作战决策的重要工具。

    

    这项工作介绍了PoKER，这是一种新颖的概率模型，旨在优化空对空场景中导弹发射的有效性，特别是在超视距（BVR）空战中。与传统的武器交战区（WEZ）模型基于最大距离、最小距离和不可逃逸距离等静态距离来划定区域不同，PoKER应用机器学习，通过整合目标的随机行为和导弹脱靶距离，更准确地预测杀伤概率。该模型能够动态评估目标行为，极大地扩展了交战分析的预测能力。通过考虑弹头杀伤力、目标和发射机朝向以及具体交战条件等因素，PoKER为交战动态提供了重要见解并量化了成功概率。因此，它有可能成为超视距空战飞行员的重要工具，从而改善作战决策。

    arXiv:2610.32864v1 Announce Type: cross  Abstract: This work introduces PoKER, a novel probabilistic model engineered to optimize missile launch effectiveness in air-to-air scenarios, specifically within Beyond Visual Range (BVR) air combat. Unlike conventional Weapon Engagement Zone (WEZ) models that delineate zones based on static distances such as maximum, minimum, and no-escape ranges, PoKER applies machine learning to predict kill probabilities more accurately by integrating the stochastic behaviors of targets and missile miss distances. This model dynamically evaluates target behavior, greatly expanding the predictive capabilities of engagement analysis. By factoring in elements such as warhead lethality, target and shooter orientations, and the specific conditions of engagement, PoKER provides important insights into engagement dynamics and quantifies success probabilities. Consequently, it can potentially be an important tool for BVR air combat pilots, improving operational dec
    
[^16]: 面向策略上下文纵向因果效应估计的基础模型

    A Foundation Model for Policy In-Context Longitudinal Causal Effect Estimation

    [https://arxiv.org/abs/2610.32863](https://arxiv.org/abs/2610.32863)

    本文提出了首个支持动态治疗策略的纵向因果效应估计基础模型 PICLE-PFN，通过在合成数据上预训练并以上下文学习方式表示目标策略，将因果效应估计的计算摊销到不同数据集、策略和预测范围上，避免了部署时耗时的模型拟合。

    

    纵向因果效应估计因时变混杂因素以及治疗与混杂因素之间的反馈回路而变得复杂。现有方法通常需要针对特定数据进行干扰模型拟合，且许多方法是针对预先设定的治疗序列或预测时间范围定制的，导致部署时计算耗时。我们提出了基于先验数据拟合网络的策略上下文纵向因果效应估计（**PICLE-PFN**），这是第一个支持*动态*治疗策略的纵向因果效应估计基础模型。PICLE-PFN 通过上下文中的历史-决策示例来表示目标策略，为广泛的治疗策略类别提供了统一的接口。PICLE-PFN 在从纵向结构因果模型中采样的合成数据上进行预训练，将因果效应估计摊销到不同数据集、策略和时间范围上。给定一个新的观测数据集和目标策略，它可以直接估计……

    arXiv:2610.32863v1 Announce Type: new  Abstract: Longitudinal causal effect estimation is complicated by time-varying confounding and treatment-confounder feedback loops. Existing methods generally require data-specific nuisance-model fitting, and many are tailored to prespecified treatment sequences or prediction horizons, leading to time-consuming computation at deployment. We introduce *Policy In-Context Longitudinal causal Effect estimation with Prior-data Fitted Networks* (**PICLE-PFN**), the first foundation model for longitudinal causal effect estimation that supports *dynamic* treatment policies. PICLE-PFN represents target policies through in-context history-decision demonstrations, providing a unified interface for a broad class of treatment policies. Pretrained over synthetic data sampled from longitudinal structural causal models, PICLE-PFN amortizes causal effect estimation across datasets, policies, and horizons. Given a new observational dataset and target policy, it dir
    
[^17]: 面向多智能体系统的企业级RAG再造：一种先组织后检索的知识层

    Reimagining Enterprise RAG for Multi-Agent Systems: A Knowledge Layer That Organizes Before It Retrieves

    [https://arxiv.org/abs/2610.32851](https://arxiv.org/abs/2610.32851)

    该论文提出一种“先组织后检索”的企业RAG知识层，通过七智能体逆向工程流水线从代码仓库中恢复系统行为逻辑，并解决跨服务同名实体操作语义不同导致扁平向量检索混淆的问题。

    

    企业正在整个软件开发生命周期中部署智能体AI，它们需要的是其操作对象的系统行为逻辑，而非利益相关者文档。一个由七个智能体组成的逆向工程流水线恢复了这种逻辑，将18个服务中的241个代码仓库处理成数千个描述范围、用户旅程、规则和验收标准的工件。然而恢复并不等于可用性。诸如“优惠”“资格”“激活”等实体在共享词汇下跨服务反复出现，名称相关但操作语义截然不同，因此一条端到端旅程会牵涉一千多条几乎相同却绝不能混淆的规则。扁平向量检索在这种情况下失败了——它会以高相似度从错误的服务中返回相同的规则，且没有任何错误信号。传统的补救措施——内容级本体——并不存在，而将每个重复实体的所有出现都建立关联，其成本将超过项目本身的投入……

    arXiv:2610.32851v1 Announce Type: new  Abstract: Enterprises are deploying agentic AI across the software development lifecycle, and they need the behavioral logic of the systems they act on, not stakeholder documentation. A seven-agent reverse-engineering pipeline recovered this logic, processing 241 repositories across 18 services into thousands of artifacts describing scope, journeys, rules and acceptance criteria. Recovery does not confer usability. Entities like offer, eligibility, and activation recur across services under a shared vocabulary, related in name but operationally distinct, so one end-to-end journey implicates over a thousand near-identical rules that must not be conflated. Flat vector retrieval fails, returning an identical rule from the wrong service at high similarity with no signal of error. Content-level ontology, the traditional remedy, did not exist, and relating every occurrence of recurring entities would have cost more than the engagement and not survived a
    
[^18]: HarnessMM：面向 Harness Token 效率的递归自我改进

    HarnessMM: Recursive Self-Improvement for Harness Token Efficiency

    [https://arxiv.org/abs/2610.32813](https://arxiv.org/abs/2610.32813)

    HarnessMM 通过递归自我改进优化 harness 配置，并借助受保护决策复用让智能体在工具反馈后直接复用已确定的决策来替代重复推理，从而在 actor 权重固定不变的情况下提升 harness 的 token 效率。

    

    arXiv:2610.32813v1 公告类型：新 摘要：智能体 harness 在工具反馈之后会反复调用推理，即使先前的一次调用已经确定了下一步动作。我们提出 HarnessMM（Harness Minus Minus），一种通过受保护决策复用与递归自我改进（RSI）来提升 harness token 效率的方法。RSI 从训练反馈中迭代地优化伴随式 harness 配置，并通过验证集选出用于部署的配置。位于模型-工具边界处的控制器复用已解决的决策来替代后续的推理回调，同时保持 actor 权重固定不变。可执行决策图存储精确的动作、字面参数、因果依赖关系以及公共守卫条件。受保护决策复用从观察到的回执中激活这些持久化的决策；认证回调替换通过原生接口执行这些决策，并记录已注册的慢回调的移除。对于新的语义决策、缺乏证据支持的情况以及选择器弃权（原文在此处截断）……

    arXiv:2610.32813v1 Announce Type: new  Abstract: Agent harnesses repeatedly invoke reasoning after tool feedback, even when an earlier call has already fixed the next action. We introduce HarnessMM (Harness Minus Minus), a method for improving harness token efficiency through guarded decision reuse and recursive self-improvement (RSI). RSI iteratively refines companion harness configurations from training feedback, with validation selecting the configuration for deployment. A controller at the model-tool boundary reuses resolved decisions to replace later reasoning callbacks while actor weights remain fixed. Executable Decision Graphs store exact actions, literal arguments, causal dependencies, and public guards. Guarded Decision Reuse activates these persistent decisions from observed receipts; Certified Callback Substitution executes them through the native interface and records removal of the registered slow callback. New semantic decisions, unsupported evidence, and selector absten
    
[^19]: 学习优化器调度中的乘法与加法状态耦合

    Multiplicative vs. Additive State Coupling in Learned Optimizer Schedules

    [https://arxiv.org/abs/2610.32801](https://arxiv.org/abs/2610.32801)

    本研究通过反向传播学习MAX-CUT求解器的调度，发现双线性（乘法）状态耦合架构在分布外泛化上相比加法耦合具有显著且持久的优势，从而将关键架构区别定位于求解器状态进入调度的方式，而非状态自适应性本身。

    

    我们通过在展开的经典求解器上进行反向传播，来学习MAX-CUT连续惩罚松弛问题的调度。状态盲调度能够泛化到比训练所用实例大得多的实例。状态自适应的双线性调度同样能实现分布外泛化，在15个随机种子上优于其未训练的基线（t=4.1）。相比之下，具有相同输入和输出头的加法消融模型在分布内与双线性模型在统计上不可区分，但在分布外的表现却比其自身基线更差（t=-11.7）。这种分离在无正则化的情况下即会出现，并在30轮权重衰减扫描中持续存在：双线性模型的优势在每个测试值下都保持显著，而加法模型始终无法恢复。在该设置下，这些结果将求解器状态进入调度的方式，而非单纯的状态自适应性，隔离为与（分布外泛化表现）相关的关键架构区别。

    arXiv:2610.32801v1 Announce Type: cross  Abstract: We learn schedules for a continuous penalty relaxation of MAX-CUT by backpropagating through an unrolled classical solver. A state-blind schedule generalizes to instances substantially larger than those used for training. A state-adaptive bilinear schedule also generalizes out of distribution, outperforming its untrained baseline across 15 seeds ($t=4.1$). By contrast, an additive ablation with identical inputs and output heads is statistically indistinguishable from the bilinear model in distribution but performs worse than its own baseline out of distribution ($t=-11.7$). This separation appears without regularization and persists across a $30$ round weight-decay sweep: the bilinear advantage remains significant at every tested value, whereas the additive model never recovers. The results in this setting isolate how solver state enters the schedule, rather than state adaptivity alone, as the architectural distinction associated with 
    
[^20]: 每条边都需要引用：利用语言模型智能体构建以引用为依据的医疗软件采用知识图谱

    Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents

    [https://arxiv.org/abs/2610.32781](https://arxiv.org/abs/2610.32781)

    该论文提出一个从开放网络构建医疗软件客户与集成关系知识图谱的流水线，要求每条边都必须有源页面的逐字引用支持，从而确保语言模型智能体提取的信息可验证、可追溯。

    

    医院需要定期采购临床医生和患者赖以使用的软件。由于这些采购成本高昂、难以逆转，且可能影响患者护理，采购方需要寻找证据来确认哪些同行机构已经在使用某款产品，以及该产品与哪些系统相连接。这些证据是公开的，但分散在供应商网站、新闻稿和第三方页面中。语言模型智能体现在能够以低成本收集这些信息，但提取出的声明的正确性难以核实，而错误列出的客户信息可能会误导采购方并损害机构声誉。我们提出了一个从开放网络构建客户关系与集成关系知识图谱的流水线，遵循一条规则：每条边都必须由来自其源页面的逐字引用来支持。页面来源于对每个供应商网站的确定性爬取，以及一个在预算限制内搜索网络其余部分的ReAct智能体。一个宽松的提取器提出带有引用的关系，随后……

    arXiv:2610.32781v1 Announce Type: cross  Abstract: Hospitals regularly buy software that clinicians and patients depend on. Because these purchases are costly, hard to reverse and can affect patient care, buyers look for evidence of which peer institutions already use a product and which systems it connects to. This evidence is public but scattered across vendor websites, press releases and third-party pages. Language-model agents can now collect it at low cost, but the correctness of the extracted claims is difficult to establish, and a wrongly listed customer can mislead buyers and misrepresent an organisation. We present a pipeline that builds a knowledge graph of customer and integration relations from the open web under one rule: every edge must be supported by a verbatim quote from its source page. Pages come from a deterministic crawl of each vendor's site and a budgeted ReAct agent that searches the rest of the web. A permissive extractor proposes relations with quotes, and a l
    
[^21]: 面向数字民主的AI对齐：抵御大语言模型偏差的更公平选民代表性

    AI Alignment for Digital Democracy: Fairer Voters Representation against LLM Biases

    [https://arxiv.org/abs/2610.32780](https://arxiv.org/abs/2610.32780)

    该研究揭示了LLM在模拟选民时存在人口统计学偏差，并提出通过AI对齐策略（如从选民画像中移除性别、教育等敏感特征）可将最大群体误差降低22-42%，从而在不牺牲整体准确性的前提下实现更公平的选民代表性。

    

    AI代表制正在数字民主领域兴起，用于协助模拟选民和辅助决策。大语言模型（LLM）的表现会因模型本身、提示语境以及其接收到的选民信息而有所不同，对某些群体的模拟比对其他群体更为准确，这可能使模拟结果对特定群体不公平。我们研究了LLM模拟选民群体中的人口统计学偏差：LLM对个人投票的模拟有多准确、某些选民群体是否比其他群体被更准确地代表，以及AI对齐能否在不牺牲整体准确性的前提下减少这些差异。基于三次美国总统选举的数据，我们比较了一个闭源模型和一个开源模型在不同选民人口特征上的表现，并测试了旨在实现更公平选民代表性的AI对齐策略。从选民画像中移除性别信息可使模型的最大群体误差降低22-42%，移除教育信息也能降低该误差。

    arXiv:2610.32780v1 Announce Type: cross  Abstract: AI representation is emerging in digital democracy, assisting in voter emulation and decision-making. Large language models (LLMs) behave differently depending on the model, the prompt context, and the voter information they receive, representing some groups more accurately than others and potentially making the emulation unfair to certain groups. We investigate demographic biases in LLM-emulated electorates: how accurately LLMs represent individual votes, whether some voter groups are represented more accurately than others, and whether AI alignment can reduce these disparities without sacrificing overall accuracy. Using three U.S. presidential elections, we compare a closed-source and an open-source LLM across voters' demographic features and test AI alignment strategies toward fairer voter representation. Removing gender from the voter profile reduces the largest group error of the models by 22-42%, and removing education reduces it
    
[^22]: 在线策略自训练：大语言模型的无数据自我改进

    On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models

    [https://arxiv.org/abs/2610.32770](https://arxiv.org/abs/2610.32770)

    该论文提出了一种无需任何数据的在线策略自训练框架，大语言模型仅需粗糙的高层主题描述即可自生成问题-答案对和特权推理演示进行自我改进，在多个领域提升能力的同时保留原有领域的性能。

    

    在线策略自蒸馏使得能够在不依赖更强外部教师的情况下对基础模型进行后训练。然而，现有方法依赖于真实的问题-答案对、人工策划的问题集或其他监督信号。在这项工作中，我们将在线策略自蒸馏重新表述为一个自训练框架，由自生成的问题-答案对和特权推理演示驱动。我们将监督信号推向极限：仅由粗糙的高层级主题描述引导，模型在无需访问任何人工策划的提示或答案的情况下，自行合成查询课程和特权推理轨迹。我们发现，在完全合成的特权信息上进行自蒸馏，能够使模型在多个领域获得改进，同时保留其在未进行自训练的领域上的能力。我们展示了该框架在文本和多模态领域的通用性，并取得了一致的（效果）。

    arXiv:2610.32770v1 Announce Type: cross  Abstract: On-policy self-distillation enables post-training foundation models without relying on stronger external teachers. However, existing approaches rely on ground-truth question-answer pairs, human-curated problem sets, or other supervised signals. In this work, we reformulate on-policy self-distillation as a self-training framework, driven by self-generated question-answer pairs and privileged reasoning demonstrations. We push supervision to the limit: guided solely by coarse, high-level topic descriptions, the model synthesizes its own curriculum of queries and privileged reasoning traces without access to any human-curated prompts or answers. We find that self-distillation on fully synthetic privileged information enables a model to improve across multiple domains while retaining its capabilities on domains it is not self-trained on. We demonstrate the generality of this framework across text and multimodal domains, achieving consistent
    
[^23]: 车联网中联邦学习的干净标签投毒攻击与防御博弈

    Clean-label Poisoning Attack and Defense Game for Federated Learning in Vehicular Networks

    [https://arxiv.org/abs/2610.32765](https://arxiv.org/abs/2610.32765)

    该论文针对车联网联邦学习提出了一种特征空间干净标签投毒攻击与特征一致性重标记防御方法，并通过博弈论框架刻画攻防均衡策略，在不损害正常检测准确率的情况下将攻击成功率维持在低水平。

    

    本文研究了车联网中使用联邦学习（FL）时干净标签投毒攻击与防御之间的交互。基本安全消息（BSM）的异常检测分类被建模为在具有动态拓扑结构的车联网上的分布式学习问题。作为联邦学习面临的关键威胁，本文引入了一种特征空间的干净标签攻击，该攻击将良性样本向恶意特征极端方向扰动，在保持标签不变的同时降低类间可分性。为应对这一攻击，提出了一种特征一致性重标记防御方法，用于检测并重新标记可疑样本。此外，构建了一个博弈论框架来刻画攻防双方之间的交互，并基于攻击有效性与成本参数来表征均衡策略。结果表明，该攻击随被攻陷节点数量的增加而有效扩展，而所提出的防御方法能够在不降低干净样本准确率的前提下将攻击成功率保持在较低水平，并使均衡策略发生……（原文摘要截断于此）

    arXiv:2610.32765v1 Announce Type: cross  Abstract: This paper studies interactions between clean-label poisoning attack and defense in vehicular networks using federated learning (FL). Basic Safety Message (BSM) classification for anomaly detection is modeled as a distributed learning problem over a vehicular network with dynamic topology. As a critical threat to FL, a feature-space clean-label attack is introduced, wherein benign samples are perturbed toward malicious feature extremes, degrading class separability while preserving labels. To counter this, a feature-consistency relabeling defense is proposed to detect and relabel suspicious samples. A game-theoretic framework is developed to capture attack-defense interactions and characterize equilibrium strategies based on attack effectiveness and cost parameters. Results show that the attack scales effectively with compromised nodes, while the proposed defense keeps attack success low without degrading clean accuracy and shifts equi
    
[^24]: 面向自动化实验室中智能体自我博弈的语义表示

    Semantic Representation for Agentic Self-Play in Automated Laboratories

    [https://arxiv.org/abs/2610.32755](https://arxiv.org/abs/2610.32755)

    该论文提出VERID——实验室工作单元的持久化高保真语义数字孪生基础设施，通过可查询自省、时间压缩仿真与全面追踪持久化等核心服务，弥合AI智能体高层推理与湿实验室复杂易错物理现实之间的差距。

    

    驾驭复杂、高维的生物搜索空间需要可扩展的物理与计算智能，以便通过在构思、实验和自动化之间的迭代式自我博弈，探索超越局部最优的方案。将实验设计转化为可靠的物理执行仍然受制于静态的实验室自动化和未建模的物理变异性。为解决这一问题，我们提出了VERified Infrastructure for Discovery Laboratories（简称VERID），这是一个旨在弥合人工智能（AI）智能体高层推理与湿实验室复杂、易出错的物理现实之间差距的基础性基础设施。VERID是实验室工作单元的持久化、高保真语义数字孪生体，提供四项核心服务：用于智能体推理的可查询自省、用于方案验证的时间压缩仿真、用于完整科学记录的全面追踪持久化。

    arXiv:2610.32755v1 Announce Type: cross  Abstract: Navigating complex, high-dimensional biological search spaces necessitates scalable physical and computational intelligence to explore beyond local optima through iterative self-play across ideation, experimentation, and automation. Translating experimental designs into reliable physical execution remains bottlenecked by static laboratory automation and unmodeled physical variability. To address this, we introduce VERified Infrastructure for Discovery Laboratories, or VERID, a foundational infrastructure designed to bridge the gap between the high-level reasoning of artificial intelligence (AI) agents and complex, error-prone physical realities of wet labs. VERID is a persistent, high-fidelity semantic digital twin of the laboratory workcell that provides four core services: queryable introspection for agentic reasoning, time-compressed simulation for protocol validation, comprehensive trace persistence for a complete scientific record
    
[^25]: ActionRAG：面向大语言模型智能体长期记忆的证据精炼图扩散方法

    ActionRAG: Evidence-Refining Graph Diffusion for Long-Term Memory in LLM Agents

    [https://arxiv.org/abs/2610.32752](https://arxiv.org/abs/2610.32752)

    ActionRAG提出了一种图记忆架构，通过谨慎的分层实体消解索引与查询时基于局部路径结构的图扩散回溯评估，帮助LLM智能体从长期交互历史中恢复精确且相互一致的关系型证据。

    

    LLM智能体需要能够从不断增长的交互历史中恢复精确且相互一致的证据的长期记忆，尤其是当答案依赖于事实之间的关系而非孤立的语义匹配时。当前的记忆流水线在两个阶段仍然脆弱：索引错误可能碎片化重复的实体或将不同实体错误合并，而在线检索常常将初始相似度分数视为最终结果，即使周围的图结构提供了修正这些分数的证据。我们提出ActionRAG，一种将谨慎索引与查询时图回溯评估相结合的图记忆架构。其索引流水线对类型标签进行协调并应用分层实体消解。给定查询后，双通道的实体与三元组检索形成初始证据集，并诱导出一个有界的局部图。查询条件下的接收者归一化扩散利用局部路径结构来精炼三元组分数。种子保护、枢纽惩罚通量……（摘要在此处截断）

    arXiv:2610.32752v1 Announce Type: new  Abstract: LLM agents need long-term memory that can recover precise and mutually consistent evidence from growing interaction histories, particularly when answers depend on relations among facts rather than isolated semantic matches. Current memory pipelines remain vulnerable at two stages: indexing errors can fragment repeated entities or merge distinct ones, and online retrieval often treats initial similarity scores as final even when the surrounding graph provides evidence for revising them. We introduce ActionRAG, a graph memory architecture that couples cautious indexing with query-time graph retroevaluation. Its indexing pipeline reconciles type labels and applies tiered entity resolution. Given a query, dual-channel entity and triple retrieval forms an initial evidence set and induces a bounded local graph. Query-conditioned receiver-normalized diffusion refines triple scores using local path structure. Seed protection, hub-penalized flux,
    
[^26]: 自适应世界模型异常识别（AWARE）：机器人动力学预测与自省框架

    Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection

    [https://arxiv.org/abs/2610.32742](https://arxiv.org/abs/2610.32742)

    AWARE是一个仅依靠外部摄像头监控即可实现机器人异常检测的世界模型框架，通过系统动力学估计与状态预测，在真实起重机数据上异常分类性能比最强基线提升26%，并能解码动力学参数定位故障来源。

    

    关键基础设施中未被识别的异常可能迅速演变为灾难性的物理或运营故障。当前的异常检测方法通常依赖于直接访问被监控系统，利用机载传感器（如电机编码器）来检测异常状态。然而，老旧硬件往往缺乏这些传感器，或者严格的安全协议可能完全禁止访问。因此，我们提出了AWARE——一个世界模型框架，仅通过外部摄像头监控，结合系统动力学估计与状态预测即可实现异常检测。在包含负载碰撞和电机退化的240条真实世界机器人起重机轨迹上的评估表明，在嘈杂的仅摄像头状态估计条件下，AWARE的异常分类性能较最强基线提升了26%。此外，推断出的动力学参数可被解码以识别故障来源，定位性能退化……

    arXiv:2610.32742v1 Announce Type: cross  Abstract: Unrecognized anomalies in critical infrastructure can rapidly cascade into catastrophic physical or operational failures. Current anomaly detection approaches often rely on direct access to the monitored system by utilizing onboard sensors, such as motor encoders, to detect anomalous states. However, legacy hardware often lacks these sensors, or strict security protocols may prohibit access altogether. We therefore present AWARE, a world model framework that is capable of anomaly detection using only external camera monitoring via system dynamics estimation and state prediction. Evaluated across 240 real-world robotic crane trajectories, including payload collisions and motor degradations, AWARE demonstrates a 26% improvement in anomaly classification over our strongest baseline under noisy, camera-only state estimation. Furthermore, the inferred dynamics parameter can be decoded to identify the source of faults, localizing the degradi
    
[^27]: 智能体灾害地图绘制器：利用AI智能体模拟专家制图以服务风险传播

    Agentic Hazard Mapper: Emulating Expert Map-Making with AI Agents for Risk Communication

    [https://arxiv.org/abs/2610.32738](https://arxiv.org/abs/2610.32738)

    本文提出Agentic Hazard Mapper多智能体系统，通过模拟专业制图师的工作流程和设计权衡来自动生成灾害地图，其生成的地图在空间理解、疏散规划效果和工作负荷方面均优于现有市政地图。

    

    灾害地图是公共风险传播、备灾和疏散规划的重要工具，但由于地方政府和社区获取制图专业知识的机会有限，有效的地图设计并非总能得到保证。我们提出了Agentic Hazard Mapper，这是一个多智能体系统，通过模拟专业制图工作流程和迭代设计权衡来自动化灾害地图的生产，同时使设计决策能够适应不同的数据、地点和灾害情境。我们与地图专家开展了形成性研究以指导系统设计，随后进行了用户研究，并对地图专家和灾害从业者进行了访谈。系统生成的地图提高了参与者的空间理解能力和疏散规划表现，降低了感知工作量和任务完成时间，并且总体上比相应的市政地图更受青睐。我们进一步讨论了人类监督以及跨用户和社区的适应性。

    arXiv:2610.32738v1 Announce Type: new  Abstract: Hazard maps are essential tools for public risk communication, preparedness, and evacuation planning, yet effective cartographic design is not always ensured due to limited access to cartographic expertise across local governments and communities. We present Agentic Hazard Mapper, a multi-agent system that automates hazard-map production by emulating professional cartographic workflows and iterative design trade-offs while adapting design decisions to different data, locations, and hazard contexts. We conducted a formative study with map experts to inform the system design, followed by a user study and interviews with map experts and hazard practitioners. The generated maps improved participants' spatial understanding and evacuation-planning performance, reduced perceived workload and task completion time, and were generally preferred over corresponding municipal maps. We further discuss human oversight, adaptation across users and commu
    
[^28]: 对比学习的内部差异：泛化界与下游行为

    Internal Discrepancy for Contrastive Learning: Generalization Bounds and Downstream Behaviors

    [https://arxiv.org/abs/2610.32729](https://arxiv.org/abs/2610.32729)

    本文提出“内部差异”度量来统一刻画对比学习中的各种理论假设，证明了基于该度量的下游监督任务泛化界，并提出松弛k均值损失作为减小内部差异的近似方法。

    

    对比学习已成为训练表示模型的一种流行范式。尽管先前的研究致力于寻求更实用、限制更少的假设以增强对对比学习的理论理解，但这些假设本质上描述的是数据内部表示之间或由函数类所诱导的表示之间的相互联系。为了统一概括这些假设，我们提出了“内部差异”这一度量方法，用于量化正样本的平均表示与潜在聚类中心之间的差异。随后，我们在无需额外假设的情况下，证明了对比学习之后下游监督任务的内部差异感知泛化界。这一泛化界不仅作为连接各种假设的统一框架，还激励我们进一步减小内部差异，从而引入了松弛k均值损失作为其近似。此外，

    arXiv:2610.32729v1 Announce Type: cross  Abstract: Contrastive learning has emerged as a popular paradigm for training representation models. While prior studies pursue more practical and less restrictive assumptions to enhance theoretical understanding of contrastive learning, they essentially describe the interconnections among representations within the data or induced by the function class. To encapsulate these assumptions, we propose the measure of internal discrepancy that quantifies the difference between the mean representation of positive samples and latent cluster centers. We then prove an internal discrepancy-aware generalization bound for downstream supervised tasks following contrastive learning, without requiring additional assumptions. This generalization bound not only serves as a unifying framework connecting various assumptions but also motivates us to further reduce the internal discrepancy, thereby introducing the relaxed k-means loss as an approximation. Moreover, 
    
[^29]: PlayerEmbed：从原始输入序列中学习行为指纹表示

    PlayerEmbed: Learning Behavioral Fingerprint Representations from Raw Input Sequences

    [https://arxiv.org/abs/2610.32727](https://arxiv.org/abs/2610.32727)

    该论文提出PlayerEmbed框架，利用深度度量学习和混合CNN-LSTM编码器从原始玩家输入序列中学习紧凑的行为指纹嵌入，在玩家识别任务中对已见和未见玩家分别达到0.950和0.879的平衡准确率，且增加训练玩家数量可显著提升对未见玩家的泛化能力。

    

    电子游戏中的原始玩家输入序列包含细粒度的行为信息，但其高维和噪声特性使得大规模分析充满挑战。为了解决这一问题，我们提出了PlayerEmbed，一个利用深度度量学习来学习紧凑且具有判别性的玩家嵌入表示的框架。我们的方法采用混合CNN-LSTM编码器，在来自1,000名玩家的原始输入序列上进行训练。我们在这些玩家的保留序列上，以及在来自训练期间完全未见过的1,000名玩家的序列上对模型进行评估，以检验其对未见身份的泛化能力。在玩家识别任务中，基于参考嵌入构建的最近质心分类器取得了强劲性能，对训练期间见过的玩家达到0.950的平衡准确率，对训练期间未见过的玩家达到0.879的平衡准确率。系统性增加训练玩家数量能够显著提升对未见玩家的泛化能力，表明从……

    arXiv:2610.32727v1 Announce Type: cross  Abstract: Raw player input sequences in video games contain fine-grained behavior, but are high-dimensional and noisy, which makes large-scale analytics challenging. To address this, we introduce PlayerEmbed, a framework that learns compact and discriminative player embeddings using deep metric learning. Our approach uses a hybrid CNN-LSTM encoder trained on raw input sequences from 1,000 players. We evaluate the model on held-out sequences from these players and on sequences from 1,000 players entirely unseen during training to assess generalization to unseen identities. For player identification, a nearest-centroid classifier built from reference embeddings achieves strong performance, reaching balanced accuracies of 0.950 for players seen during training and 0.879 for players unseen during training. Systematically increasing the number of training players substantially improves generalization to unseen players, indicating a transition from si
    
[^30]: 本体引导下基于结构化元数据与气候科学出版物的知识图谱构建

    Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications

    [https://arxiv.org/abs/2610.32725](https://arxiv.org/abs/2610.32725)

    本文提出一种代理式、本体引导的知识图谱构建框架，利用大语言模型从气候科学文献中抽取带文本证据的三元组，并通过多阶段验证确保语义一致性与可追溯性，抽取质量接近人类标注者水平。

    

    科学文献的快速扩张远远超出了人工整理的速度，这为从科学文本中提取结构化知识造成了根本性瓶颈。为解决这一问题，我们提出了一种代理式、本体引导的知识图谱（KG）构建框架。该框架利用专门的代理分别执行文档理解、信息抽取、本体一致性检查和实体链接任务。首先，大语言模型（LLM）从研究论文中以三元组的形式抽取事实知识，并附带明确的文本证据。随后，三元组经过多阶段验证流程，使其锚定于本体之上。通过强制执行模式约束并要求提供文本片段形式的证据，我们确保了数据在语义上的一致性和可追溯性。在一项针对1000个抽取三元组的一致性研究中，Gemini-2.5-flash 相对于三位人类标注者取得了 0.640–0.700 的 Cohen's κ 分数，达到（原文摘要在此处截断）

    arXiv:2610.32725v1 Announce Type: cross  Abstract: The rapid expansion of scientific literature far outpaces the speed of manual curation, creating a fundamental bottleneck in extracting structured knowledge from scientific texts. To address this, we introduce an agentic, ontology-guided framework for knowledge graph (KG) construction. Our framework utilizes specialized agents for document understanding, information extraction, ontology conformance, and entity linking. First, an LLM extracts factual knowledge in the form of triples with explicit textual evidence from research papers. Then the triplets undergo a multistage verification process to ground the triplets in the ontology. By enforcing schema constraints and requiring evidence in the form of text fragments, we ensure the data is both semantically consistent and traceable. In an agreement study over 1,000 extracted triplets, Gemini-2.5-flash achieves Cohen's $\kappa$ scores of 0.640--0.700 against three human annotators, matchi
    
[^31]: SP-Guide：利用空间先验增强航空视觉语言导航

    SP-Guide: Enhancing Aerial Vision-Language Navigation with Spatial Priors

    [https://arxiv.org/abs/2610.32697](https://arxiv.org/abs/2610.32697)

    SP-Guide通过空间先验跨模态编码器等模块将物理空间先验引入航空视觉语言导航，显著提升了无人机在CityNav基准上的成功率，尤其在未见环境中增益最大。

    

    航空视觉语言导航（AVLN）要求无人机在大型户外场景中进行推理的同时，从部分自我中心的观测中到达语言描述的目标。现有方法通常将视点中心的RGB观测与全局锚定的历史地图融合为通用视觉嵌入，而将跨模态空间对齐问题留给稀疏的导航监督来学习。我们提出了SP-Guide，这是一个基于Transformer的目标预测框架，通过空间先验跨模态编码器、指令增强解码器以及用于最终目标精修的目标收敛过滤器，来解决这一对齐弱点。在CityNav基准上的实验表明，SP-Guide相比代表性基线提高了导航成功率，其中在未见过的环境中增益最大。这些结果表明，基于物理的空间先验为稳健的AVLN提供了有效的归纳偏置。

    arXiv:2610.32697v1 Announce Type: cross  Abstract: Aerial vision-language navigation (AVLN) requires a UAV to reach a language-described target from partial egocentric observations while reasoning over large outdoor scenes. Existing approaches typically fuse view-centered RGB observations and globally anchored history maps as generic visual embeddings, leaving cross-modal spatial alignment to be learned from sparse navigation supervision. We present SP-Guide, a transformer- based goal-prediction framework that addresses this weakness of alignment through a Spatial Prior Cross-Modal Encoder, an Instruction Enhanced Decoder, and a Goal Convergence Filter for terminal target refinement. Experiments implemented on the CityNav benchmark show that SP-Guide improves success rate over the representative baselines, with the largest gains on unseen environments. These results indicate that physically grounded spatial priors provide an effective inductive bias for robust AVLN.
    
[^32]: 用于动作模仿中自监督目标发现的条件神经过程

    Conditional Neural Processes for Self-Supervised Goal Discovery in Action Imitation

    [https://arxiv.org/abs/2610.32688](https://arxiv.org/abs/2610.32688)

    在条件神经过程的隐藏空间中加入注意力机制，可使其在观察动作序列的早期就涌现出鲁棒的目标表示，从而实现动作模仿中的自监督目标发现，为机器人的自监督通用学习提供了有前景的模型。

    

    受婴儿通过模仿他人学习新任务这一能力的启发，在这项工作中，我们评估了条件神经过程在多大程度上能够自主学习对其潜在目标具有信息量的动作表示。我们表明，在隐藏空间中引入注意力机制，可以在观察动作序列的早期阶段就促使鲁棒的目标表示涌现，这使其成为机器人自监督通用学习器的一个有前景的模型。

    arXiv:2610.32688v1 Announce Type: cross  Abstract: Inspired by infants' ability to learn novel tasks by imitating others, in this work we evaluate to what extent Conditional Neural Processes can autonomously learn representations of actions that are informative about their underlying goal. We show that the addition of an attention mechanism in the hidden space fosters the emergence of robust goal representations early on in the observation of an action sequence, making it a promising model for a robotic self-supervised general learner.
    
[^33]: 基于习得速度先验的机器人演示重定时

    Retiming Robot Demonstrations with a Learned Speed Prior

    [https://arxiv.org/abs/2610.32676](https://arxiv.org/abs/2610.32676)

    提出从大规模演示数据中学习一个速度先验，为任意任务预测执行速度分布并生成一致的时钟来重定时演示，从而消除操作者个人节奏对策略训练的干扰。

    

    机器人演示的节奏往往由远程操作者而非任务本身决定。因此，同一动作在不同演示中可能以差异悬殊的速度出现，而在此类数据上训练的策略可能会连同任务行为一起继承操作者特有的时间节奏。我们提出直接从大规模演示数据中学习一个速度先验。给定任务指令、运动路径以及沿路径的图像，该先验预测沿路径执行速度的分布，从中选定一个分位数即可定义一个一致的时钟，用于对每条演示进行重定时。现有方法从其要加速的数据本身中提取加速信号，因此每个新任务都需要额外的拟合、标注或一对预定义的速度；而一个训练好的速度先验则可覆盖语料库中的所有任务。我们在机器人远程操作数据以及手持式演示（即由人类手臂而非机器人手臂执行动作的演示）上拟合了速度先验……

    arXiv:2610.32676v1 Announce Type: new  Abstract: The pace of a robot demonstration is often determined by the teleoperator rather than by the task itself. As a result, the same motion can appear at widely different speeds across demonstrations, and policies trained on such data may inherit operator-specific timing along with task behavior. We propose to learn a speed prior directly from large-scale demonstration data. Given a task instruction, a motion path and the images along it, it predicts a distribution over execution speed along the path, from which a chosen quantile defines a consistent clock for retiming each demonstration. Existing methods derive the acceleration signal from the very data they accelerate, so each new task costs another fit, annotation, or pair of predefined rates. A single trained speed prior instead covers every task in a corpus. We fit a speed prior on both robot teleoperation data and handheld demonstrations in which a human arm, rather than a robot arm, mo
    
[^34]: 语言模型相信自己的解释吗？一项关于功能一致性的研究

    Do Language Models Trust Their Own Justifications? A Study on Functional Consistency

    [https://arxiv.org/abs/2610.32665](https://arxiv.org/abs/2610.32665)

    该研究提出基于“功能自一致性”原则的实验框架，通过分离、移除或语义反转模型所标记的关键词元，检验语言模型生成的解释是否与其真实预测行为在功能上保持一致。

    

    大型语言模型（LLMs）已被广泛应用于文本分类任务，它们不仅输出类别预测，还会生成解释，突出显示被认为对得出预测标签最相关的词元。然而，这些被突出的元素在针对性干预下是否与模型自身的预测在行为上保持一致——我们将其称为“功能自一致性”——目前仍不清楚。尽管大量文献评估了此类解释的文本合理性，但很少有研究评估其与模型实际行为之间的功能一致性。在本工作中，我们提出了一个基于自一致性原则的实验框架：如果模型认为某些词元是决定性的，那么分离、移除或语义反转这些词元，应当会在其预测中产生系统性的、可解释的变化。我们通过充分性和完整性等指标将这一评估操作化……

    arXiv:2610.32665v1 Announce Type: new  Abstract: Large Language Models (LLMs) have been widely adopted in text classification tasks, where they not only output class predictions but also generate explanations that highlight the tokens deemed most relevant for reaching the predicted label. Yet it remains unclear whether these highlighted elements are behaviorally consistent with the model's own predictions under targeted interventions, which we refer to as functional auto-consistency. While much of the literature evaluates the textual plausibility of such explanations, few studies assess their functional consistency with the model's actual behavior. In this work, we propose an experimental framework based on the principle of auto-consistency: if a model identifies certain tokens as decisive, then isolating, removing, or semantically inverting them should produce systematic and interpretable changes in its predictions. We operationalize this evaluation through sufficiency, comprehensiven
    
[^35]: SemGeo-Gen：无监督生成跨实例的近似语义-几何对应关系

    SemGeo-Gen: Unsupervised Generation of Approximate Cross-Instance Semantic-Geometric Correspondences

    [https://arxiv.org/abs/2610.32663](https://arxiv.org/abs/2610.32663)

    SemGeo-Gen提出了一种无监督方法，可从弱对齐的3D物体集合中自动生成语义有意义、几何一致的近似跨实例对应关系，无需人工标注即可提供可扩展的3D-3D、3D-2D和2D-2D监督信号。

    

    我们提出了SemGeo-Gen，这是一种无监督方法，用于从弱对齐的3D物体集合中生成近似的跨实例语义-几何监督。我们的目标不是恢复精确的点级对应关系，因为这类对应关系在不同实例之间往往存在歧义，且大规模获取需要昂贵且不切实际的人工监督。相反，我们自动生成语义上有意义、几何上一致、且足够多样化以用于训练现代对应关系模型的近似对应关系。仅需粗糙的类别级旋转对齐作为输入，SemGeo-Gen将多视角DINOv2特征提升到3D空间，使用连续分段仿射配准对齐物体实例，并利用语义一致性筛选候选匹配。所生成的3D对应关系可以投影到渲染视图中，从而无需人工标注即可获得可扩展的3D-3D、3D-2D和2D-2D监督。我们验证了所生成的对应关系……（摘要原文在此处被截断）

    arXiv:2610.32663v1 Announce Type: cross  Abstract: We introduce SemGeo-Gen, an unsupervised method for generating approximate cross-instance semantic-geometric supervision from weakly aligned 3D object collections. Our goal is not to recover exact point-level correspondences, which are often ambiguous across different instances and would require expensive, impractical manual supervision at scale. Instead, we automatically produce approximate correspondences that are semantically meaningful, geometrically consistent, and diverse enough to train modern correspondence models. Given only coarse category-level rotation alignment, SemGeo-Gen lifts multi-view DINOv2 features to 3D, aligns object instances using continuous piecewise-affine registration, and prunes candidate matches using semantic consistency. The resulting 3D correspondences can be projected into rendered views, yielding scalable 3D-3D, 3D-2D, and 2D-2D supervision without manual annotation. We validate the generated correspon
    
[^36]: WiMoT：基于矩传输的无重放持续Wi-Fi感知

    WiMoT: Replay-Free Continual Wi-Fi Sensing via Moment Transport

    [https://arxiv.org/abs/2610.32660](https://arxiv.org/abs/2610.32660)

    WiMoT通过存储并传输各类别特征的矩统计量（均值和协方差）而非原始样本，实现了无需重放旧数据的持续Wi-Fi感知，在保护隐私的同时兼顾了学习新域的可塑性与保留旧知识的稳定性。

    

    持续Wi-Fi感知必须在可塑性（学习新房间或新用户）与稳定性（保留先前知识）之间取得平衡，然而出于隐私和存储方面的考虑，节点通常无法保留信道状态信息（CSI）。我们提出WiMoT（Wi-Fi矩传输），它存储每个先前域的各类别特征均值和协方差，而非原始样本。当特征表示发生变化时，存储的统计量会变得过时。WiMoT将这种变化限制在冻结特征基上的一个仿射映射，并将相同的映射应用于统计量，因此在没有旧样本的情况下也能保持统计量的有效性。该方法利用监督对比学习来学习新域，通过基于存储统计量的闭式损失保持先前类别的可分性，并使用从传输后统计量以每个域等权重重新拟合的高斯判别分类器进行预测，且无需域身份信息。在3DO、WiMANS和Widar3.0数据集上，WiMoT的平均最终准确率分别达到94.02%、89.61%和95.16%。

    arXiv:2610.32660v1 Announce Type: new  Abstract: Continual Wi-Fi sensing must balance plasticity, learning new rooms or users, against stability, keeping earlier ones, yet nodes often cannot retain channel state information (CSI) for privacy and storage reasons. We propose WiMoT (Wi-Fi Moment Transport), which stores the per-class feature mean and covariance of every earlier domain instead of samples. When the representation changes, stored statistics become stale. WiMoT restricts the change to an affine map on a frozen feature basis and applies the same map to the statistics, so they stay valid without old samples. Supervised contrastive learning learns the new domain, a closed-form loss on the stored statistics keeps earlier classes separable, and a Gaussian discriminant classifier, refitted from the transported statistics with equal weight per domain, predicts without domain identity. On 3DO, WiMANS, and Widar3.0, WiMoT's average final accuracies are 94.02%, 89.61%, and 95.16% with 
    
[^37]: InferSlice：一种基于电信广域网的SLO驱动LLM推理参考架构

    InferSlice: A Reference Architecture for SLO-driven LLM Inference over a Telco Wide-Area Network

    [https://arxiv.org/abs/2610.32638](https://arxiv.org/abs/2610.32638)

    InferSlice是一个部署在电信广域网上的SLO驱动LLM推理参考架构，通过集成网络切片、GPU分区、弹性服务和可观测性，解决了集中式云部署难以满足的主权、合规、延迟和资源本地化需求。

    

    LLM推理正日益被部署为企业级和交互式应用的核心服务，但主权、合规性、延迟和资源本地化要求可能限制了集中式云部署的适用性。电信运营商提供地理分布式的计算和网络基础设施，可用于应对这些限制。然而，现有的LLM服务框架是为紧耦合的数据中心设计的，对协调地理分布式和异构资源的支持有限。我们提出了InferSlice，一种在电信基础设施上实现服务水平目标（SLO）驱动的LLM推理的参考架构，集成了网络切片、GPU分区、弹性服务和可观测性。该项目由英国电信（BT）牵头，联合布里斯托大学和NVIDIA共同开展，并在JOINER英国国家级广域网（WAN）上实现，InferSlice在服务差异化方面进行了评估。

    arXiv:2610.32638v1 Announce Type: cross  Abstract: LLM inference is increasingly deployed as a core service for enterprise and interactive applications, but sovereignty, compliance, latency and resource-locality requirements can limit the suitability of centralised cloud deployments. Telco operators offer geographically distributed compute and network infrastructure for addressing these constraints. However, existing LLM serving frameworks are designed for tightly coupled data centres and provide limited support for coordinating geographically distributed and heterogeneous resources. We present InferSlice, a reference architecture for Service-Level Objective (SLO)-driven LLM inference over a telco infrastructure, integrating network slicing, GPU partitioning, elastic serving and observability. Led by BT in collaboration with the University of Bristol and NVIDIA, and realised on the JOINER UK national-scale wide-area network (WAN), InferSlice is evaluated on service differentiation acro
    
[^38]: AI智能体能帮助机器人工程师吗？来自工业界两个专注日的经验教训

    Can AI Agents Help Robotic Engineers? Lessons from Two Focus Days in Industry

    [https://arxiv.org/abs/2610.32636](https://arxiv.org/abs/2610.32636)

    本研究通过对一家工业机器人编程中小企业的案例研究，发现AI智能体并非即插即用的解决方案——除了生成高度重复代码的任务外未能提升工程师效率，其使用效果随工程师的AI使用经验和智能体的领域适配而逐步改善，且AI使用经验比机器人工程经验对成功更为关键。

    

    软件开发人员越来越多地使用能够自主执行多步骤编程任务的AI智能体。然而，尚不清楚这种情况是否以及如何适用于机器人工程领域，在该领域中，工程师使用小众的、供应商专有的语言对工业机器人进行编程，并依赖仿真工具的视觉反馈。本文报告了一项关于一家工业机器人编程中小企业使用AI智能体的案例研究。我们的结果表明，除了一项需要生成高度重复代码的任务之外，AI智能体并未提高工程师的工作效率。然而，随着工程师逐渐学会如何使用该智能体，以及该智能体被适配到该领域，使用智能体的效果随时间推移而改善。此外，AI使用经验对成功使用智能体的影响比机器人工程经验更大。这表明AI智能体并非即插即用的解决方案，企业必须对其工程师进行培训（摘要在此处截断）。

    arXiv:2610.32636v1 Announce Type: cross  Abstract: Software developers increasingly use AI agents that execute multi-step programming tasks on their own. However, it is unclear if and how this transfers to robotic engineering, where engineers program industrial robots in niche, vendor-specific languages and rely on visual feedback from simulation tools. Within this paper, we report a case study on the use of AI agents by an SME that programs industrial robots. Our results show that the agent did not make the engineers more productive, with the exception of a task that required the generation of highly repetitive code. However, the results with the agent improved over time, as the engineers learned how to use it and the agent was adapted to the domain. Moreover, the experience with AI mattered more for the success with the agent than the experience with robotic engineering. This shows that AI agents are not a plug-and-play solution. Instead, companies must train their engineers on the u
    
[^39]: EditJev：面向图像编辑模型的“系统一”路由方法

    EditJev: System One Routing for Image Editing Models

    [https://arxiv.org/abs/2610.32635](https://arxiv.org/abs/2610.32635)

    提出EditJev——一个“系统一”式路由模型，通过在视觉-语言骨干上添加轻量级偏好头，在单次前向传播中直接预测各候选图像编辑模型的样本级偏好分数，从而根据输入动态选择最优编辑器，兼顾效果与推理效率。

    

    基于指令的图像编辑模型呈现出显著的样本级性能差异：对于同一张源图像和相同的编辑指令，不同模型可能产生截然不同的结果，且表现最佳的模型往往随请求而变化。此外，这些模型在推理延迟上差异巨大，使得固定单一模型的部署方式效率低下。因此，我们研究图像编辑模型路由问题，即在给定输入图像和编辑指令的情况下，从候选模型池中动态选择一个合适的编辑器。我们提出了 EditJev，一个“系统一”式路由模型，它可以在单次前向传播中直接预测针对候选图像编辑模型的样本级偏好分数。该模型在视觉-语言骨干网络上附加一个轻量级偏好头，生成固定维度的分数向量，无需自回归生成模型名称或解析响应文本。为了训练 EditJev，我们从离线的候选模型输出结果中构建成对偏好数据并进行优化……（原文在此处截断）

    arXiv:2610.32635v1 Announce Type: cross  Abstract: Instruction-based image editing models exhibit pronounced sample-specific performance variation: different models can produce substantially different results for the same source image and editing instruction, and the best-performing model often varies across requests. These models also differ substantially in inference latency, making fixed-model deployment inefficient. We therefore study image editing model routing, which dynamically selects an editor from a candidate pool given an input image and editing instruction. We introduce EditJev, a System One routing model that directly predicts sample-level preference scores over candidate image editing models in a single forward pass. A lightweight preference head on a vision-language backbone produces a fixed-dimensional score vector without autoregressive model-name generation or response parsing. To train EditJev, we construct pairwise preferences from offline candidate outputs and opti
    
[^40]: 使用声学与声乐技巧特征的可解释歌唱情感识别

    Interpretable Singing Emotion Recognition Using Acoustic and Vocal Technique Features

    [https://arxiv.org/abs/2610.32629](https://arxiv.org/abs/2610.32629)

    该论文提出一个使用歌唱专用声学与声乐技巧特征的可解释歌唱情感识别框架，借助SHAP和GRAD-CAM揭示情感表达与声乐技巧执行之间的关联。

    

    歌唱情感识别主要依赖于深度学习模型，这些模型虽然性能强劲，但对驱动情感的声乐特征提供的洞察有限，且常常忽视歌声作为一种独特信号的价值。我们提出一个可解释的歌唱情感识别框架，使用一组精简的歌唱专用描述子，涵盖韵律、频谱特征、语音学和声乐技巧等方面。为了解码模型决策并将预测与可测量的声学现象联系起来，我们采用SHAP方法，并应用GRAD-CAM来定性地评估模型学习到的声乐技巧表征。评估结果表明，歌唱专用特征能够降低跨歌手的差异性，而声乐技巧描述子则为歌唱表达提供了可解释的洞察。此外，在VocalSet数据集上的域外评估揭示了情感表达与真实技巧执行之间的有趣关联。该分析……

    arXiv:2610.32629v1 Announce Type: cross  Abstract: Singing emotion recognition relies predominantly on deep learning models that achieve strong performance but offer limited insight into the vocal characteristics driving emotion, of- ten overlooking the singing voice as a distinct signal. We propose an interpretable singing emotion recognition framework using a compact set of singing-specific descriptors across prosody, spec- tral features, phonetics, and vocal technique. To decode model decisions and link predictions to measurable acoustic phenomena, we employ SHAP and apply GRAD-CAM to qualitatively assess the learned representations of vocal techniques. Evaluations show that singing-specific features reduce cross-singer variability, while vocal technique descriptors provide interpretable insights into singing expression. Furthermore, out-of-domain evaluation on VocalSet reveals interesting associations between emotional expression and ground-truth technical execution. The analysis h
    
[^41]: LGMT：一个用于测试大语言模型逻辑推理可靠性的工具

    LGMT: A Tool for Testing the Logical Reasoning Reliability of LLMs

    [https://arxiv.org/abs/2610.32627](https://arxiv.org/abs/2610.32627)

    本文提出开源工具LGMT，利用基于一阶逻辑的蜕变关系生成逻辑等价测试对，在无需真实标签的情况下有效检测大语言模型在等价问题重述中的推理不一致性。

    

    大语言模型越来越多地在现代软件系统中充当推理组件，然而一个能够正确回答逻辑查询的模型，在以等价形式重述同一问题时可能会自相矛盾。静态基准测试孤立地评估各个查询，因此无法暴露这种不一致性。为了解决这一局限性，我们提出了LGMT，一个基于一阶逻辑的蜕变关系来检测推理不一致性的开源工具。给定一个前提-结论形式的源测试用例，LGMT会生成逻辑等价的后续测试用例，并检查模型的蕴含预测在测试对之间是否一致，且无需真实标签。该工具提供20个内置的蜕变关系，支持经Z3定理证明器验证的自定义关系，并可通过命令行界面（CLI）或网页工作室运行，还包含无需API密钥的离线模拟模式。在六种广泛使用的LLM上通过内置基准……

    arXiv:2610.32627v1 Announce Type: new  Abstract: Large Language Models (LLMs) increasingly serve as reasoning components in modern software systems, yet a model that answers a logical query correctly can contradict itself when the same problem is restated in an equivalent form. Static benchmarks evaluate queries in isolation and therefore fail to expose such inconsistencies. To address this limitation, we present LGMT, an open-source tool that detects reasoning inconsistencies through metamorphic relations grounded in first-order logic. Given a source test case in premise-conclusion form, LGMT generates logically equivalent follow-up cases and checks whether the model's entailment predictions agree across the pair, without ground-truth labels. The tool provides 20 built-in metamorphic relations, supports custom ones verified by the Z3 theorem prover, and runs from a CLI or a web studio, including an offline mock mode requiring no API key. Across six widely used LLMs on a bundled benchm
    
[^42]: 蛋白质口袋之间什么可以迁移？面向智能体结构导向药物设计的分子优化经验复用

    What Transfers Across Protein Pockets? Experience Reuse in Molecular Optimization for Agentic Structure-Based Drug Design

    [https://arxiv.org/abs/2610.32620](https://arxiv.org/abs/2610.32620)

    该研究提出以编辑位点接触、动作和轮次为键的经验库，证明了分子优化中的探测经验可以跨蛋白质口袋迁移复用，在从未见过的口袋上仍能清晰区分改善性与恶化性的编辑。

    

    LLM智能体通过逐步编辑配体并对每次编辑进行对接，来优化蛋白质口袋中的配体。近期的智能体首先花费额外的对接次数、通过受控编辑来探测复合物；据报告，从这些探测中获得的指导比模型已有的化学知识更有帮助。然而，这些探测结果随后会与产生它们的实例一起被丢弃，因此显而易见的下一步是将它们保存在一个经验库中，让新的口袋能够继承先前的口袋已经付出代价换来的经验。这样的经验库通过一个“键”来读取：即依据哪些记录属性来汇集过去的编辑并查找新的查询。利用某次运行中针对CrossDocked2020测试集口袋的完整日志，我们研究这些键是否在经验库从未见过的口袋上依然出现，并且依然能够区分改善性编辑与恶化性编辑。结果表明，使用被编辑位点的接触信息、动作和轮次作为键，经验库能够清晰地将二者区分开来。

    arXiv:2610.32620v1 Announce Type: new  Abstract: LLM agents optimize a ligand in a protein pocket by editing it step by step and docking each edit. Recent agents first spend extra dockings probing the complex with controlled edits; guidance induced from these probes is reported to help more than the chemical knowledge the model already carries. The probes are then discarded with the instance that produced them, so the obvious next step is to keep them in an experience base, letting a new pocket inherit what earlier ones already paid for. Such a base is read through a \emph{key}: the record properties under which past edits are pooled and a new one is looked up. Using the full log of one such run over the CrossDocked2020 test-set pockets, we ask whether its keys recur on a pocket the base has never seen and still separate improving from worsening edits. \crchange{Using} the contacts at the site being edited, the action, and the round as \crchange{a} key, the base separates them sharply.
    
[^43]: LLaCO：面向大语言模型的连续扩散模型与竞争感知语义调度

    LLaCO: Continuous Diffusion for Large Language Models with Competition-Aware Semantic Scheduling

    [https://arxiv.org/abs/2610.32604](https://arxiv.org/abs/2610.32604)

    该论文提出竞争阈值理论来揭示连续扩散语言模型中词元超越的动态机制，并据此提出竞争感知的语义调度方法，以解决轨迹与时间不对齐导致的时间压缩问题。

    

    连续扩散模型已成为一种在连续嵌入空间上进行文本生成的有前景的非自回归范式。在连续轨迹上，残差词元身份直接塑造了去噪目标。然而，几何轨迹与时间进程并不对齐，引发了严重的时间压缩问题。在本文中，我们证明了嵌入范数的衰减会导致欧氏距离产生虚假的超越现象，这表明词元竞争必须从方向性角度进行评估。在该空间中，只有当有效噪声幅度超过候选词元与源词元之间的语义间隔时，候选词元才会超越源词元。我们将这些成对动态形式化为竞争阈值理论。至关重要的是，高维空间中的超越需要极端的噪声水平，导致标准轨迹将身份衰减集中到接近噪声端点的狭窄窗口内。为解决这一问题，我们提出了语义重新……（原文摘要在此处截断）

    arXiv:2610.32604v1 Announce Type: new  Abstract: Continuous diffusion models have emerged as a promising non-autoregressive paradigm for text generation over continuous embedding spaces. Along the continuous trajectory, the residual token identity directly shapes the denoising objective. However, the geometric trajectory misaligns with temporal progression, triggering severe temporal compression. In this paper, we show that embedding norm decay causes Euclidean distance to yield spurious overtaking, indicating that token competition must be evaluated directionally. In this space, a candidate overtakes the source token only when the effective noise magnitude surpasses their semantic separation. We formalize these pairwise dynamics as the Competition Threshold Theory (CTT). Crucially, high-dimensional overtaking requires extreme noise levels, causing standard trajectories to concentrate identity decay into a narrow window near the noisy endpoint. To resolve this, we propose Semantic Rete
    
[^44]: 真实MCP故障下的工具调用LLM智能体：模型规模并不能换来容错能力

    Tool-Calling LLM Agents Under Realistic MCP Faults: Scale Does Not Purchase Fault Tolerance

    [https://arxiv.org/abs/2610.32555](https://arxiv.org/abs/2610.32555)

    提出基于真实MCP生态故障数据校准的故障注入中间件MCP-Injector，实验证明扩大LLM模型规模并不能提升智能体应对工具故障的容错能力。

    

    LLM智能体日益依赖通过模型上下文协议暴露的外部工具，然而当前的基准测试假设工具能够正确且及时地响应，使得智能体对现实中工具故障的韧性从未得到检验。我们提出了MCP-Injector，一种即插即用、协议感知的故障注入中间件，它能够介入MCP工具调用路径，支持受控的故障持久性类型（永久性、瞬时性、间歇性）以及可种子化的故障调度。其故障模型根据公共MCP生态系统中观察到的故障分布（故障模式×持久性）进行校准，这些数据来自对1,774个开源MCP仓库中57,240条议题和拉取请求线程的挖掘。在一项跨越两种骨干模型规模（20B、120B）、三种智能体框架（自定义多轮规划-执行框架、OpenAI Agents SDK、LangChain）以及三种故障注入条件的1,350个实验单元的大规模实验中，我们发现从这种以永久性故障为主的压力特征中注入的故障会导致严重的质量下降……

    arXiv:2610.32555v1 Announce Type: cross  Abstract: LLM agents increasingly rely on external tools exposed via the Model Context Protocol (MCP), yet current benchmarks assume tools respond correctly and on time, leaving agent resilience to realistic tool failures untested. We introduce MCP-Injector, a drop-in, protocol-aware fault-injection middleware that interposes on the MCP tool-call path with controlled persistence (permanent, transient, intermittent) and seedable schedules. Its fault model is calibrated to the observed incident distribution (failure mode x persistence) of the public MCP ecosystem, mined from 57,240 issue and pull-request threads across 1,774 open-source MCP repositories. In a 1,350-cell experiment crossing two backbone sizes (20B, 120B), three agent stacks (custom multi-round plan-and-execute, OpenAI Agents SDK, LangChain), and three fault injection conditions, we find that faults drawn from this permanent-heavy stress profile cause severe quality degradation (tas
    
[^45]: 超越信道数据：配置增强的无线基础模型

    Beyond Channel Data: Configuration-Enriched Radio Foundation Models

    [https://arxiv.org/abs/2610.32542](https://arxiv.org/abs/2610.32542)

    本文提出在无线基础模型预训练中融入信道配置信息，实验证明所得的配置增强嵌入可作为有效归纳先验，在标签稀缺下的多种下游任务（LoS/NLoS分类、TA估计和波束选择）中一致优于基线模型。

    

    本文提出将信道配置（CFG）信息纳入无线基础模型（RFM）的预训练中，并证明由此得到的CFG增强嵌入在标签稀缺的情况下可作为下游任务的有效归纳先验。我们在三种不同的数据条件下，针对多个下游任务——视距/非视距（LoS/NLoS）分类、定时提前量（TA）估计和波束选择，将所提出的嵌入与原始信道标记以及多个基线模型进行了比较。评估在分布内和分布外的配置设置下均进行了实验。实验结果表明，在已见和未见配置场景的各项实验中，所提出的嵌入始终优于基线模型和/或与基线模型具有竞争力。

    arXiv:2610.32542v1 Announce Type: new  Abstract: This paper proposes incorporating channel configuration (CFG) information into Radio Foundation Model (RFM) pre-training and demonstrates that the resulting CFG-enriched embeddings serve as effective inductive priors for downstream tasks under label scarcity. We compare the proposed embeddings with raw channel tokens and multiple baseline models, under three different data regimes, for multiple downstream tasks --- Line-of-Sight/Non-Line-of-Sight (LoS/NLoS) classification, timing advance (TA) estimation, and beam selection. Evaluation is done in both in-distribution and out-of-distribution configuration settings. Experimental outcomes show that the proposed embeddings consistently outperform and/or are competitive to the baselines, across experiments in both seen and unseen configuration scenarios.
    
[^46]: StarRTT：一种面向星链低轨卫星网络无盲区RTT预测的端到端实时框架

    StarRTT: An End-to-End Real-Time Framework for Blind Zone-Free RTT Prediction in Starlink LEO Satellite Networks

    [https://arxiv.org/abs/2610.32540](https://arxiv.org/abs/2610.32540)

    提出StarRTT端到端实时框架，将每秒服务卫星识别与RT-LSTM预测模型相结合，消除了星链低轨卫星网络中RTT实时预测的识别盲区。

    

    诸如星链（Starlink）等低地球轨道（LEO）卫星巨型星座由于轨道速度高，需要频繁进行切换，导致往返时延（RTT）出现显著波动。准确的RTT预测可以缓解这些波动，但其从根本上依赖于持续更新的服务卫星信息。现有的服务卫星识别方法通常只能在每个调度周期结束时才完成识别，由此形成识别盲区，限制了实时预测。为克服这一限制，我们开发了StarRTT，这是一种端到端的实时框架，通过将每秒一次的服务卫星识别与RTT预测相集成，消除了该盲区。在该框架内，我们提出了实时长短期记忆网络（RT-LSTM），这是一种基于LSTM的RTT预测模型，旨在同时捕获非平稳的RTT演化特性以及服务卫星之间随时间变化的几何关系。

    arXiv:2610.32540v1 Announce Type: cross  Abstract: Low Earth Orbit (LEO) satellite mega-constellations such as Starlink require frequent handovers due to high orbital velocities, resulting in substantial Round-Trip Time (RTT) fluctuations. Accurate RTT prediction can mitigate these fluctuations, yet it fundamentally relies on continuously updated serving- satellite information. Existing serving-satellite identification methods typically complete identification only at the end of each scheduling cycle, creating an identification blind zone that limits real-time prediction. To overcome this limitation, we develop StarRTT, an end-to-end real-time framework that eliminates the blind zone by integrating per-second serving-satellite identification with RTT prediction. Within this framework, we propose Real-Time Long Short-Term Memory (RT-LSTM), an LSTM-based RTT prediction model designed to capture both non-stationary RTT evolution and time-varying geometric relationships among the serving s
    
[^47]: IGG：一个面向可见性约束下交互式GUI定位的基准测试

    IGG: A Benchmark for Interactive GUI Grounding under Visibility Constraints

    [https://arxiv.org/abs/2610.32536](https://arxiv.org/abs/2610.32536)

    该论文提出了IGG基准，首个要求智能体先通过交互使被隐藏的GUI目标变为可见再进行点击的评测基准，揭示了人类（94.6%）与最强模型（GPT-5.5达79.4%，最强开源模型仅31.6%）在可见性约束下GUI定位能力的显著差距。

    

    现有的GUI定位基准测试考察的是智能体能否在静态截图中定位一个已经可见的目标。然而，真实界面常常不满足这一假设：目标可能不在屏幕内、只有在状态改变或悬停后才会出现、处于被遮挡状态、或经过延迟后才变为可操作。这导致“恢复目标可见性”的能力一直未被评估。为了衡量这一揭示步骤，我们提出了交互式GUI定位基准IGG，在该基准中，智能体必须首先通过交互使目标可见，然后再点击它。IGG包含601个任务，覆盖16个可控制的GUI环境和七种可见性约束子类型，并采用紧凑的动作空间，包括滚动、移动、拖拽、等待和点击。在人类研究中，参与者达到94.6%的准确率，而GPT-5.5达到79.4%，被评测的最强开源视觉语言模型仅达到31.6%。为了找出这一差距的来源，我们进行了两项受控干预实验。

    arXiv:2610.32536v1 Announce Type: new  Abstract: GUI grounding benchmarks ask whether an agent can localize a target that is already visible in a static screenshot. Real interfaces often violate this assumption: a target may be off-screen, appear only after a state change or hover, remain occluded, or become actionable after a delay. This leaves the ability to recover target visibility unevaluated. To measure this reveal step, we introduce Interactive GUI Grounding (IGG), a benchmark in which an agent must first make the target visible through interaction and then click it. IGG contains 601 tasks across 16 controllable GUI environments and seven visibility-constraint sub-types, with a compact action space of scrolling, moving, dragging, waiting, and clicking. In our human study, participants achieve 94.6% accuracy, while GPT-5.5 reaches 79.4% and the strongest evaluated open-source VLM reaches only 31.6%. To identify the source of this gap, we conduct two controlled interventions. Tell
    
[^48]: 通过预测党派归属分析奥地利议会演讲

    Analysing Austrian Parliamentary Speeches by Predicting Party Affiliations

    [https://arxiv.org/abs/2610.32515](https://arxiv.org/abs/2610.32515)

    该论文利用微调BERT模型预测奥地利议会演讲的党派归属，发现联合执政党派的演讲特征趋于相似而反对党逐渐疏远，且议员只有在换党后才会改变演讲风格。

    

    本文研究了奥地利国民议会议员的演讲特征。我们提出了一种基于微调BERT模型的分析方法，该模型为每篇演讲分配一个关于奥地利各党派的概率分布，并在发言者和党派层面对这些分布进行聚合。在进行深入分析之前，我们进行了广泛的评估以确保生成分布的质量。我们没有将预测党派的错误归类视为性能问题，而是利用这些错误来推导关于议员和政治发展的见解。我们的研究结果表明，共同执政的党派在演讲特征上变得更加相似，而反对党则进一步疏远执政党。此外，转换党派的议员通常只有在换党之后才会调整其演讲特征。最后，我们公开发布了一个包含党派归属预测的数据集。

    arXiv:2610.32515v1 Announce Type: cross  Abstract: This paper examines speech characteristics of the Austrian National Council members (CMs). We introduce analysis methodology based on a fine-tuned BERT-Model which assigns to every speech a probability distribution over Austrian parties. These are then aggregated at the speaker and party level. We perform an extensive evaluation to ensure the quality of the generated distributions prior to further analysis. Instead of considering mis-assignments of predicted parties as a performance issue, we leverage them to derive insights about the CMs and political developments. Our findings show that parties that govern together become more similar in their speech characteristics, while, analogously, parties that are in opposition move further away from governing parties. Moreover, CMs who switch party usually adapt their speech characteristics only after the switch occurs. Finally, we publicly release a data set containing party affiliation predi
    
[^49]: MusicTwins：训练音频-语言模型像其对应人物一样聆听音乐

    MusicTwins: Training Audio-Language Models to Listen like Their Persona

    [https://arxiv.org/abs/2610.32495](https://arxiv.org/abs/2610.32495)

    该论文提出MusicTwins框架，通过从人类文字中构建人物画像表征、结合个人记忆，并采用监督微调与强化学习对Audio Flamingo Next进行后训练，使音频-语言模型能够像特定个人一样对文本和音乐刺激做出个性化回应，显著提升了人物保真度。

    

    人们对音乐的感知方式各不相同——他们由此产生的联想、喜欢的内容以及做出的反应都存在差异。旨在模拟音乐环境中人类的计算模型应当考虑这些差异。尽管大语言模型（LLM）已被证明能够再现人类行为的某些方面，但音频-语言模型（ALM）能否再现特定个人对文本和音乐刺激的反应仍未得到充分研究。我们提出了MusicTwins，这是一个结合人物画像摘要、个人记忆和后训练来生成针对特定个人回应的框架。我们从人类的文字中构建人物表征，捕捉其背景、观点、个性、写作风格以及音乐偏好档案，并通过监督微调和强化学习对Audio Flamingo Next进行后训练。在十个保留人物上的评估显示，Audio Flamingo Next在人物保真度方面取得了显著提升，包括……

    arXiv:2610.32495v1 Announce Type: cross  Abstract: People differ in how they perceive music, what associations they make with it, what they like about it, and how they respond to it. Computational models intended to simulate people in environments that include music should account for these differences. Although LLMs have been shown to reproduce aspects of human behavior, whether audio-language models (ALMs) can reproduce a specific person's responses to textual and musical stimuli remains underexplored. We introduce MusicTwins, a framework that combines persona summaries, personal memory, and post-training to generate person-specific responses. We construct persona representations from human writings, capturing background, views, personality, writing style, and a musical profile, and post-train Audio Flamingo Next through supervised fine-tuning and reinforcement learning. Evaluation on ten held-out personas shows notable improvements for Audio Flamingo Next in persona fidelity, includ
    
[^50]: 基于信号感知自适应融合（SAAF）的高效零样本视频异常检测

    Efficient Zero-Shot Video Anomaly Detection via Signal-Aware Adaptive Fusion (SAAF)

    [https://arxiv.org/abs/2610.32486](https://arxiv.org/abs/2610.32486)

    SAAF框架通过双信号自适应采样器（增量信号捕捉突发事件、漂移信号检测静态与渐进异常）以及阈值门控的级联架构，在无需训练数据的情况下实现了高效且全面的零样本视频异常检测。

    

    将视觉语言模型部署于视频异常检测面临一个根本性约束：其语义推理所需的计算预算与真实世界视频流的规模不兼容。我们提出信号感知自适应融合（SAAF）框架，通过三项关键创新实现高效的零样本异常检测。首先，我们引入一种在语义嵌入空间中运行的双信号自适应采样器：增量信号通过帧间余弦距离捕捉突发事件，而漂移信号通过与滚动时间上下文的偏差检测静态和渐进式异常。二者通过OR门控组合，并结合基于自适应指数移动平均（EMA）的阈值处理，无需任何训练数据即可实现全面的异常覆盖。其次，我们提出一种阈值门控的级联架构，由轻量级的帧级评分先行处理，仅当置信度需要进一步分析时才调用昂贵的视频级时序分析。

    arXiv:2610.32486v1 Announce Type: cross  Abstract: Deploying Vision-Language Models for video anomaly detection confronts a fundamental constraint. Their semantic reasoning demands computational budgets incompatible with the scale of real-world video streams. We present Signal-Aware Adaptive Fusion (SAAF), a framework that achieves efficient zero-shot anomaly detection through three key innovations. First, we introduce a dual-signal adaptive sampler operating in semantic embedding space: a delta signal captures sudden events via frame-to-frame cosine distance, while a drift signal detects static and gradual anomalies through deviation from rolling temporal context. Their OR-gated combination with adaptive EMA-based thresholding provides comprehensive anomaly coverage without training data. Second, we propose a threshold-gated cascaded architecture where lightweight frame-level scoring precedes expensive video-level temporal analysis, invoking it only when confidence warrants additional
    
[^51]: 当智能体失败时：模型还是执行框架——该归咎于谁？

    When Agents Fail: Model or Harness--Who's to Blame?

    [https://arxiv.org/abs/2610.32483](https://arxiv.org/abs/2610.32483)

    本文提出了一个通过五种互补指标（有效完成率、模型-框架匹配度、跨模型匹配度、交互残差和Token负载）来评估LLM智能体执行框架与底层模型交互表现的方法，用以区分智能体失败时究竟应归咎于模型还是执行框架。

    

    LLM智能体将模型与执行框架相结合，因此智能体基准测试的性能反映了二者的共同行为。我们认为，因此应当通过执行框架与底层模型的交互来评估框架。我们开发了一个结合五种互补指标的评估框架：(i) 有效完成率（VCR），即智能体正常完成并交付可评估结果的可靠程度，与任务正确性无关；(ii) 模型-框架匹配度，即框架通过有效完成来支持给定模型达成任务的程度；(iii) 跨模型匹配度，即框架在被评估的各模型上的整体适用性；(iv) 交互残差，即被组件级汇总指标所掩盖的特定模型-框架配对的执行优势与劣势；以及 (v) Token负载，即智能体执行所消耗的token量，包括失败尝试。我们使用四个核心模型评估了四个通用执行框架（OpenCode、Codex CLI、Pi CLI和DeepAgents），外加两个固定的原生模型。

    arXiv:2610.32483v1 Announce Type: new  Abstract: LLM agents combine models with execution harnesses, so agentic benchmark performance reflects their joint behavior. We argue that harnesses should therefore be evaluated through their interactions with underlying models. We develop a framework combining five complementary measures: (i) Valid Completion Rate (VCR), how reliably agents finish normally and deliver evaluable results, irrespective of task correctness; (ii) Model-Harness Fit, how well a harness supports task achievement through valid completion for a given model; (iii) Cross-Model Fit, a harness's overall suitability across evaluated models; (iv) Interaction Residuals, pairing-specific execution strengths and weaknesses hidden by component-level summaries; and (v) Token Load, the token demands of agent execution, including failed attempts. We evaluate four general-purpose harnesses (OpenCode, Codex CLI, Pi CLI, and DeepAgents) with four core models, plus two fixed native model
    
[^52]: CASE：面向冻结多模态情感模型的混淆感知技能演化

    CASE: Confusion-Aware Skill Evolution for Frozen Multimodal Emotion Models

    [https://arxiv.org/abs/2610.32481](https://arxiv.org/abs/2610.32481)

    提出首个面向多模态情感识别的智能体技能框架CASE，通过维护针对特定情感对的自演化EmotionSkills库，在冻结模型参数的前提下以低成本、可复用、可迁移的方式纠正语义相近情感间的系统性混淆。

    

    多模态大语言模型（MLLMs）通过对多模态证据进行推理，实现了细粒度的情感理解，从而极大地推动了多模态情感识别（MER）的发展。然而，这类模型在语义相近且多模态线索高度重叠的情感之间仍然表现出系统性的混淆。现有方法通常通过全局参数适配来解决这些错误，导致混淆纠正成本高昂且缺乏灵活性。我们转而观察到，情感混淆是高度局部化的现象，因此更适合通过可复用的、针对特定情感对的专业知识来解决。基于这一洞察，我们提出了混淆感知技能演化（CASE），这是首个面向MER的智能体技能框架。CASE维护一个稀疏的、自演化的、针对特定情感对的EmotionSkills库，将情感专业知识从模型参数中外化，使学到的专业知识能够被持续精炼，并易于跨多样场景迁移。

    arXiv:2610.32481v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have substantially advanced Multimodal Emotion Recognition (MER) by enabling fine-grained emotion understanding through reasoning over multimodal evidence. However, they still exhibit systematic confusion between semantically adjacent emotions with highly overlapping multimodal cues. Existing approaches typically address these errors through global parameter adaptation, making confusion correction costly and inflexible. We instead observe that emotion confusion is highly localized and therefore better addressed through reusable pair-specific expertise. Building on this insight, we propose Confusion-Aware Skill Evolution (CASE), the first agent-skill framework for MER. CASE maintains a sparse, self-evolving library of pair-specific EmotionSkills that externalize affective expertise from model parameters, enabling the learned expertise to be continuously refined and readily transferred across divers
    
[^53]: CauProphet：教会智能体像人类一样发展因果认知

    CauProphet: Teaching Agents to Develop Causal Cognition like Humans

    [https://arxiv.org/abs/2610.32477](https://arxiv.org/abs/2610.32477)

    该论文提出CauProphet智能体，通过构建结构化的“因果信念”并在“因果认知”中持续更新因果知识，使大语言模型智能体能够像人类一样习得因果理解，并将其迁移应用于未来的决策任务。

    

    大语言模型（LLM）智能体日益通过利用外部记忆和过往经验来支持长时程任务。然而，现有的智能体主要检索记录或摘要，缺乏形成并完善对世界的因果理解的能力。受人类因果认知的启发，我们提出了CauProphet，这是一个能在其“因果认知”中发展和更新因果知识的智能体。对于每项任务，CauProphet将因果认知中的相关知识与当前证据相结合，构建一个“因果信念”——一种能够捕捉因果因素以及各种可能行动效果的结构化信念。它利用这一信念进行决策，并随着新观察结果的出现持续更新信念和因果认知，从而使因果知识能够超越个体经验而持久存在。评估结果表明，CauProphet能够逐步习得因果知识，并将其迁移应用于未来的决策中。在金融……（摘要在此处被截断）

    arXiv:2610.32477v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly support long-horizon tasks by leveraging external memory and past experience. However, existing agents mainly retrieve records or summaries, lacking the ability to form and refine causal understanding of the world. Inspired by human causal cognition, we introduce CauProphet, an agent that develops and updates causal knowledge within its Causal Cognition. For each task, CauProphet combines relevant knowledge in Causal Cognition with current evidence to construct a Causal Belief, a structured belief capturing causal factors and the effects of possible actions. It uses this belief for decision-making and continuously updates both the belief and Causal Cognition as new observations emerge, enabling causal knowledge that persists beyond individual experiences. Evaluations demonstrate that CauProphet progressively acquires causal knowledge and transfers it to future decision-making. In financial r
    
[^54]: 超越事后奖励：面向多模态嵌入的特权推理探索

    Beyond Post-Hoc Rewards: Privileged Reasoning Exploration for Multimodal Embeddings

    [https://arxiv.org/abs/2610.32474](https://arxiv.org/abs/2610.32474)

    提出PILOT框架，将自生成推理建模为潜在变量，通过共享多模态大语言模型的先验与后验两种推理模式进行特权推理探索，从而发现更具检索效用的推理轨迹，突破传统仅以输入为条件的事后奖励优化的局限，提升多模态嵌入的质量。

    

    多模态大语言模型（MLLM）的进展推动了多模态嵌入领域的显著发展。近期方法利用模型的生成能力，通过自生成的推理来增强被嵌入的输入，从而产生信息更丰富的嵌入表示。然而，用于训练嵌入模型的传统检索监督仅指定成对相关性关系，并未显式引导用于输入增强的推理过程。这类推理既无法被直接观测，也不由输入唯一确定，因此多种合理的推理轨迹可能引入检索效用各异的语义线索。现有方法大多仅在输入条件下优化推理，限制了对与检索相关轨迹的探索。我们提出PILOT框架，将推理建模为潜在变量，并将该变量的先验分布与后验分布实例化为同一共享MLLM的两种不同推理模式。其中，先验仅……（原文摘要在此处截断）

    arXiv:2610.32474v1 Announce Type: new  Abstract: Advances in Multimodal Large Language Models (MLLMs) have driven substantial progress in multimodal embeddings. Recent methods exploit generative capabilities by augmenting embedded inputs with self-generated reasoning to produce more informative embeddings. However, conventional retrieval supervision for training embedding models specifies pairwise relevance relations without explicitly guiding the reasoning used for input augmentation. Such reasoning is neither directly observed nor uniquely determined by the input, allowing multiple plausible traces to introduce semantic cues of varying retrieval utility. Existing approaches largely optimize reasoning conditioned on the input alone, limiting the exploration of retrieval-relevant traces. We introduce PILOT, a framework that models reasoning as a latent variable and instantiates its prior and posterior distributions as distinct reasoning modes of a shared MLLM. The prior conditions only
    
[^55]: ALLMTS：基于大语言模型的自动化医疗分诊系统

    ALLMTS: Automated LLM-based Medical Triage System

    [https://arxiv.org/abs/2610.32469](https://arxiv.org/abs/2610.32469)

    提出了ALLMTS系统，不让大语言模型直接分诊，而是让它执行护士使用的商业化远程医疗分诊指南，根据第一个肯定答案确定性地得出分诊处置结果，从而提高了电话分诊的可靠性与可审计性。

    

    电话分诊是一项安全关键任务：根据一份简短的接诊记录，护士必须决定来电者是需要救护车、前往急诊科、预约门诊，还是在家自我护理。直接询问大语言模型（LLM）并不可靠：它们会过度升级常规病例，且无法依据可审计的临床标准来论证其建议的合理性。我们提出了基于大语言模型的自动化医疗分诊系统（ALLMTS），它将每一个分诊处置都建立在护士所使用的同类商业化维护的远程医疗指南之上。ALLMTS 并非直接让大语言模型进行分诊，而是让它执行指南：系统会选择合适的指南并评估其中的分诊评估问题。处置结果确定性地由第一个肯定答案得出，遵循与护士相同的遍历规则。我们在566个经过专家验证的合成场景上评估了ALLMTS，这些场景涵盖14种处置类型，旨在测试特定的指南问题（摘要原文在此处截断）。

    arXiv:2610.32469v1 Announce Type: new  Abstract: Telephone triage is a safety critical task: from a brief intake note, a nurse must decide whether a caller needs an ambulance, an emergency department, an appointment, or self care at home. Queried directly, large language models (LLMs) are unreliable: they over-escalate routine cases and cannot justify their recommendations against an auditable clinical standard. We present the Automated LLM-based Medical Triage System (ALLMTS), which grounds every disposition in the same commercially maintained telehealth guidelines nurses use. Rather than asking an LLM to triage, ALLMTS makes it execute the guideline: it selects the appropriate guideline and evaluates its triage assessment questions. The disposition follows deterministically from the first positive answer, using the same traversal rule a nurse follows. We evaluate ALLMTS on 566 synthetic, expert-validated scenarios spanning 14 dispositions and designed to test specific guideline quest
    
[^56]: 超越复制：基于协议变异的智能合约安全任务终身构建

    Beyond Copying: Lifelong Smart Contract Security Task Construction via Protocol Mutation

    [https://arxiv.org/abs/2610.32453](https://arxiv.org/abs/2610.32453)

    提出CONSTRAINTDIAL框架，通过协议变异自动修改智能合约的调用与输入，使原始攻击失效的同时保持可利用性和正常功能，从而实现安全任务的终身构建，从203个公开案例生成了3,045个任务输入，避免评估奖励记忆或复制的答案。

    

    AI智能体可以通过检查代码和运行测试来支持智能合约安全。据慢雾（SlowMist）统计，整个区块链生态系统的损失约为386亿美元。现有评估可能会奖励记忆或复制的答案，包括从公开区块链交易中恢复的攻击代码。此外，重构合约和验证任务也需要昂贵的人工投入。为解决这些问题，我们提出了CONSTRAINTDIAL，一个通过协议变异实现智能合约安全任务终身构建的框架。它自动化本地合约的重构过程，并利用攻击执行来选择对所需调用或输入的修改。执行检查指导修订，使原始攻击失败，同时保持可利用性、安全目标和正常功能。新发生的安全事件和审计发现也进入相同的流程。借助CONSTRAINTDIAL，我们从203个公开案例中构建了3,045个任务输入，涵盖检测、利用等（摘要内容在此处被截断）。

    arXiv:2610.32453v1 Announce Type: cross  Abstract: AI agents can inspect code and run tests to support smart contract security. SlowMist reports about $38.6 billion in losses across the broader blockchain ecosystem. Existing evaluations can reward memorized or copied answers, including attack code recovered from public blockchain transactions. Reconstructing contracts and validating tasks also require costly manual effort. To address these issues, we propose CONSTRAINTDIAL, a framework for lifelong construction of smart contract security tasks via protocol mutation. It automates local contract reconstruction and uses attack execution to select changes to the required calls or inputs. Execution checks guide revisions that make the original attack fail while preserving exploitability, the security objective, and normal functionality. New incidents and audit findings enter the same procedure. With CONSTRAINTDIAL, we construct 3,045 task inputs from 203 public cases, covering detect, explo
    
[^57]: AutoMIL：用于全切片图像分类的自动化多示例学习

    AutoMIL: Automated Multiple Instance Learning for Whole-Slide Image Classification

    [https://arxiv.org/abs/2610.32452](https://arxiv.org/abs/2610.32452)

    AutoMIL是一个开源的自动机器学习框架，通过自动化预处理、训练和评估等关键流程步骤，在无需手动配置的情况下实现了与专用流程相当性能的全切片图像分类基线。

    

    开发用于全切片图像分类的多示例学习流程涉及许多相互关联的设计决策，并且需要病理学和机器学习两个领域的专业知识。我们提出了AutoMIL，这是一个开源框架，它应用自动机器学习领域的原理，通过自动化流程中的关键步骤（如预处理、训练和评估）来解决这一任务。在使用默认设置的基准评估中，AutoMIL在UBC-OCEAN数据集上达到了0.94的ROC-AUC分数，在PCam上达到了0.99，在TCGA-NSCLC子集上达到了0.93-0.96，与相关工作中的专用流程相当。集成方法主要提高了在不平衡数据上的性能，这表明自动化流程必须适应数据集的特性。因此，AutoMIL为全切片图像分类提供了一个无需手动配置流程的易用基线。

    arXiv:2610.32452v1 Announce Type: cross  Abstract: Developing multiple instance learning pipelines for the classification of whole-slide images involves many interdependent design decisions and expertise in both pathology and machine learning. We present AutoMIL, an open-source framework that applies principles from the field of automated machine learning to tackle this task by automating integral pipeline steps, such as preprocessing, training and evaluation. In a benchmark evaluation with default settings, AutoMIL reached ROC-AUC scores of 0.94 on UBC-OCEAN, 0.99 on PCam and 0.93-0.96 on a TCGA-NSCLC subset, comparable to specialised pipelines in related work. Ensembling mainly improved performance on imbalanced data, showing that an automated pipeline must adapt to dataset characteristics. AutoMIL thus provides an accessible baseline for whole-slide image classification without manual pipeline configuration.
    
[^58]: DIGOM：面向多模态医疗结果预测的解耦交互图

    DIGOM: Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare

    [https://arxiv.org/abs/2610.32451](https://arxiv.org/abs/2610.32451)

    提出DIGOM框架，利用解耦视图的稀疏多路图联合建模患者内跨模态、跨患者同模态和跨患者跨模态三种交互关系，以提升多模态医疗数据的临床结果预测性能。

    

    多模态医疗数据包含着超越单个患者观测范围的关系信息。因此，有效的临床结果预测不仅需要整合来自多种模态的信息，还需要对这些模态之间的关系进行建模。标准的融合方法无法显式地捕捉这些交互，而现有的基于图的方法通常只能建模其中的一部分交互。我们提出DIGOM（Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare，面向多模态医疗结果预测的解耦交互图），这是一个基于图的框架，能够联合建模三种互补的交互类型：患者内跨模态交互、跨患者同模态交互以及跨患者跨模态交互。DIGOM学习一个具有解耦视图的稀疏多路图，其中每个视图编码患者间不同模态对之间的交互。通过可微机制采样具有患者自适应连接性的离散邻接矩阵。

    arXiv:2610.32451v1 Announce Type: cross  Abstract: Multimodal healthcare data contain relational information that extends beyond individual patient observations. Effective clinical outcome prediction therefore requires not only integrating information from multiple modalities, but also modeling the relationships between them. Standard fusion approaches do not explicitly capture these interactions, while existing graph-based approaches typically model only a subset of them. We propose DIGOM (Disentangled Interaction Graphs for Outcome prediction in Multimodal healthcare), a graph-based framework that jointly models three complementary interaction types: Intra-patient cross-modal, Cross-patient same-modal, and Cross-patient cross-modal interactions. DIGOM learns a sparse multiplex graph with disentangled views, each encoding interactions between pairs of modalities across patients. Discrete adjacency matrices with patient-adaptive connectivity are sampled via a differentiable mechanism, 
    
[^59]: POSEIDON：从代码仓库为科学智能体规模化构建训练环境

    POSEIDON: Scaling Training Environments for Scientific Agents from Code Repositories

    [https://arxiv.org/abs/2610.32448](https://arxiv.org/abs/2610.32448)

    POSEIDON通过将科学工作流分解为复现、子流程、补全和修正四类任务并配合递归改进循环，实现了从科学代码仓库自动规模化生成可执行可验证的训练环境，构建了包含97,659个环境的Pearl数据集，环境数量约为先前方法的10倍。

    

    训练环境仍然是数据驱动科学发现中智能体面临的主要瓶颈。现有的可执行、可验证训练环境集合规模较小，且其构建难以规模化。为解决这一局限，我们提出了Poseidon——一个从科学代码仓库中大规模自动生成多样化、可验证训练环境的自动化流水线。在相同的仓库预算下，Poseidon通过将科学工作流分解为四种互补的任务类型（复现、子流程、补全和修正）提取出近10倍于先前方法的训练环境，其中每种类型都针对科学研究的一项独特能力。递归改进循环进一步优化失败的候选环境，并回收原本会被丢弃的环境。利用Poseidon，我们构建了Pearl——一个包含来自7,110个公开仓库的97,659个可执行且可验证环境的集合。

    arXiv:2610.32448v1 Announce Type: new  Abstract: Training environments remain a major bottleneck for agents in data-driven scientific discovery. Existing collections of executable and verifiable training environments are small, and their construction remains difficult to scale. To address this limitation, we introduce Poseidon, an automated pipeline for generating diverse, verifiable training environments from scientific code repositories at scale. From the same repository budget, Poseidon extracts nearly 10 times as many training environments as a prior approach by decomposing scientific workflows into four complementary task types: Reproduction, Subflow, Completion, and Correction, where each type targets a distinct capability for scientific research. A recursive improvement loop further refines failed candidates and recovers otherwise discarded environments. Using Poseidon, we construct Pearl, a collection of 97,659 executable and verifiable environments from 7,110 public repositori
    
[^60]: 解耦患者画像与临床工作流：一种用于逼真医疗对话生成的双智能体方法

    Decoupling Patient Personas and Clinical Workflows: A Dual-Agent Approach to Realistic Medical Dialogue Generation

    [https://arxiv.org/abs/2610.32443](https://arxiv.org/abs/2610.32443)

    提出CoRPS-Dial双智能体框架，通过解耦患者画像与临床工作流并辅以事后质量控制，生成兼顾临床规范性与患者多样性的逼真诊前医疗对话。

    

    诊前医疗对话对于收集全面的患者信息至关重要，但由于隐私和伦理限制，大规模获取这类对话成本高昂。现有基于大语言模型（LLM）的合成方法通常依赖提示工程或僵化的规则，限制了其在临床相关性与真实患者多样性之间取得平衡的能力。我们提出了CoRPS-Dial，这是一个用于诊前医疗对话合成的可控双智能体框架，它将患者画像与符合临床指南的工作流规则相整合。CoRPS-Dial通过由患者画像驱动的患者智能体与由流程引导的医生智能体之间的交互来生成对话，并进行事后质量控制以保证角色一致性、工作流完整性和文本一致性。在冠心病数据上的实验表明，CoRPS-Dial在信息覆盖度、临床合规性和对话自然度方面提升了合成对话与真实对话之间的对齐程度。

    arXiv:2610.32443v1 Announce Type: cross  Abstract: Pre-consultation medical dialogues are essential for collecting comprehensive patient information, but are costly to obtain at scale due to privacy and ethical constraints. Existing LLM-based synthesis methods often rely on prompt engineering or rigid rules, limiting their ability to balance clinical relevance with realistic patient diversity. We propose CoRPS-Dial, a controllable dual-agent framework for pre-consultation medical dialogue synthesis that integrates patient personas with guideline-aligned clinical workflow rules. CoRPS-Dial generates dialogues through interactions between a persona-driven patient agent and a process-guided doctor agent, followed by post-hoc quality control for role consistency, workflow completeness, and textual consistency. Experiments on coronary heart disease demonstrate that CoRPS-Dial improves alignment between synthetic and real dialogues in information coverage, clinical compliance, and dialogue n
    
[^61]: 何时停止惩罚动作？用于VLA强化学习的鞅GRPO（Martingale GRPO）

    When to Stop Penalizing Actions? Martingale GRPO for VLA Reinforcement Learning

    [https://arxiv.org/abs/2610.32431](https://arxiv.org/abs/2610.32431)

    提出MART-GRPO，通过成功概率与策略得分鞅刻画失败反馈的理想停止条件，并借助视觉-语言模型进行事件引导截断，避免GRPO将轨迹级惩罚错误地施加到失败轨迹中有用的恢复动作上。

    

    一次失败的视觉-语言-动作（VLA）轨迹并不意味着其中每一个动作都对失败有贡献。然而在组相对策略优化（GRPO）中，单一的轨迹级优势会被分配给一次采样中的所有动作，这使得潜在有用的恢复尝试可能与其试图纠正的错误一起受到惩罚。本文利用成功概率鞅和策略得分鞅刻画了失败反馈的理想停止条件。在这些停止点上，被移除的梯度贡献在给定保留信息的条件下条件期望为零。因此，截断能够保持参考GRPO组梯度估计器的期望，并可以降低其协方差。其中一种停止条件出现在成功条件概率为零的时刻，这启发了用于选择候选反馈边界的错误与恢复线索。本文提出了结合视觉-语言模型（VLM）进行事件引导截断的鞅GRPO（MART-GRPO）。

    arXiv:2610.32431v1 Announce Type: new  Abstract: A failed vision-language-action (VLA) rollout does not imply that every action contributed to failure. Yet in group-relative policy optimization (GRPO), a single trajectory-level advantage is assigned to all actions in a rollout. Potentially useful recovery attempts can thus be penalized alongside the errors they aim to correct. Ideal stopping conditions for failure feedback are characterized using success-probability and policy-score martingales. At these stops, the removed gradient contribution has zero conditional expectation given the retained information. Truncation therefore preserves the expectation of a reference GRPO group-gradient estimator and can reduce its covariance. One such stop occurs when the conditional probability of success is zero, motivating error and recovery cues for selecting candidate feedback boundaries. Martingale GRPO (MART-GRPO) is introduced with a vision-language model (VLM) for event-guided truncation. T
    
[^62]: Async-RBench：面向LLM智能体异步任务执行的基准测试

    Async-RBench: Benchmarking Asynchronous Task Execution in LLM Agents

    [https://arxiv.org/abs/2610.32429](https://arxiv.org/abs/2610.32429)

    该论文提出了Async-RBench，一个包含200个可执行任务的基准，通过事件响应契约（ERC）和动态重规划得分（DRS）独立评估LLM智能体在异步任务执行中处理动态变化结果的能力。

    

    大语言模型（LLM）智能体越来越多地以异步方式执行任务：当其子智能体运行时，主智能体继续工作，而任务状态可能在结果到达之前发生变化。主智能体必须决定哪些工作需要更新或保留、哪些检查需要重复，同时避免执行被禁止的更改。仅凭最终任务是否成功无法揭示这些结果是否被正确处理，而自然的异步到达过程又难以复现。我们提出了Async-RBench，一个用于评估LLM智能体异步任务执行的基准，包含200个可执行任务，覆盖八种场景。场景控制器使用预定义的任务状态条件来调控结果的交付，而事件响应契约（ERC）规定了每个响应必须更改、保留、避免和重新验证的内容。动态重规划得分（DRS）对响应过程和结果赋予同等权重，并与任务成功率分开报告。我们评估了九个智能体。

    arXiv:2610.32429v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly execute tasks asynchronously: a main agent continues working while its subagents run, and the task state may change before their results arrive. The main agent must decide which work to update or preserve and which checks to repeat, while avoiding prohibited changes. Final task success alone does not reveal whether these results were handled correctly, and natural asynchronous arrivals are hard to reproduce. We introduce Async-RBench, a benchmark for asynchronous task execution in LLM agents, with 200 executable tasks covering eight scenarios. A scenario controller regulates result delivery using predefined task-state conditions, while an Event Response Contract (ERC) specifies what each response must change, preserve, avoid, and re-verify. The Dynamic Replanning Score (DRS) weights the response process and outcome equally and is reported separately from task success. We evaluate nine agents
    
[^63]: 思维循环：面向文档图像中表格视觉问答的自评估迭代推理

    Cycle-of-Thought: Self-Evaluative Iterative Reasoning for Table Visual Question Answering in Document Images

    [https://arxiv.org/abs/2610.32423](https://arxiv.org/abs/2610.32423)

    提出了一种无需训练的推理时自我纠正框架，让单个视觉语言模型迭代地扮演生成者、批评者和改进者的角色，形成闭环自我反馈回路，无需额外标注数据或模型更新即可提升文档图像中表格视觉问答的性能。

    

    视觉语言模型（VLM）的最新进展显著提升了视觉问答（VQA）的性能，但它们对表格等结构化内容的推理仍不够稳定，常常受到空间错位和逻辑幻觉的困扰。现有的改进表格视觉问答的方法通常依赖于架构修改或昂贵的监督微调（SFT），由于需要大规模的专家标注数据集和大量计算资源，这些方法的准入门槛很高。在本工作中，我们提出了Self-RefineVQA，这是一个无需训练的推理时自我纠正框架，能够在不需要额外标注数据或模型更新的情况下提升表格视觉问答的性能。受人类修改行为的启发，我们的方法使单个视觉语言模型能够迭代地扮演生成者、批评者和改进者的角色，形成一个闭环的自我反馈回路，通过提示引导的重新评估来纠正常见的推理错误……

    arXiv:2610.32423v1 Announce Type: cross  Abstract: Recent advances in Vision-Language Models (VLMs) have significantly improved Visual Question Answering (VQA) performance, yet their reasoning over structured content such as tables remains inconsistent, often plagued by spatial misalignment and logical hallucinations. Existing approaches for improving table VQA typically rely on architectural modifications or expensive supervised fine-tuning (SFT), which impose high barriers to entry due to the need for large-scale, expert-annotated datasets and significant computational resources. In this work, we propose Self-RefineVQA, a training-free, inference-time self-correction framework that improves table VQA performance without additional labeled data or model updates. Inspired by human revision behavior, our method enables a single VLM to iteratively act as a generator, critic, and refiner, forming a closed self-feedback loop that corrects common reasoning errors through prompt-guided re-ev
    
[^64]: LeafSkill：面向递归大语言模型智能体的高效叶级动作定位

    LeafSkill: Efficient Leaf-Level Action Grounding for Recursive LLM Agents

    [https://arxiv.org/abs/2610.32412](https://arxiv.org/abs/2610.32412)

    LeafSkill提出一个模型参数之外的外部技能复用层，通过语义检索相关定位经验并将动作模板绑定到当前对象，用轻量级1.7B选择器替代大模型完成跨任务重复出现的叶级动作定位，从而提升递归LLM智能体的效率。

    

    递归式大语言模型（LLM）智能体通过任务分解、执行和环境反馈来解决长时程任务。然而，在生成原子子任务后，它们往往需要再次调用同一个大模型，将一条熟悉的指令翻译成可执行的动作。我们观察到，高层规划是任务特定的，而叶级动作定位则在多个任务中反复出现，并且可以对照环境所接受的动作进行验证。我们提出LeafSkill，一个外部复用层，它检索语义相关的定位经验，将动作模板绑定到当前对象和可接受动作上，并使用轻量级的1.7B选择器来选择绑定的动作，或将未发生变化的子任务交还给原始智能体处理。技能保持在模型参数之外，使得冻结的选择器能够使用训练后新增的技能。在词汇和句法改写的情况下，精确匹配无法检索到任何参考动作，而语义检索……（摘要原文在此处被截断）

    arXiv:2610.32412v1 Announce Type: new  Abstract: Recursive large language model (LLM) agents solve long-horizon tasks through decomposition, execution, and environmental feedback. Yet after producing an atomic subtask, they often invoke the same large model again to translate a familiar instruction into an executable action. We observe that high-level planning is task-specific, whereas leaf-level action grounding recurs across tasks and can be checked against environment-admitted actions. We introduce LeafSkill, an external reuse layer that retrieves semantically related grounding experience, binds action templates to current objects and admissible actions, and uses a lightweight 1.7B selector to choose a bound action or defer the unchanged subtask to the original agent. Skills remain outside the model parameters, allowing a frozen selector to use post-training additions. Under lexical and syntactic paraphrases, exact matching retrieves none of the reference actions, while semantic ret
    
[^65]: PASTA：提示感知的退火扩散采样

    PASTA: Prompt-Aware Annealed Diffusion Sampling

    [https://arxiv.org/abs/2610.32390](https://arxiv.org/abs/2610.32390)

    PASTA提出了一种免训练的提示感知退火扩散采样框架，通过在共享潜变量轨迹上对允许对象的区域选择性施加语义引导，在文本到图像生成中严格保留指定区域，同时保持视觉连贯性。

    

    尽管扩散模型能够生成高质量的图像，但现实世界的视觉设计需要有意识地保留干净区域，用于放置标题、标志或UI元素。现有的布局感知文本到图像方法难以严格保留预留区域，常常引入不必要的内容，使下游编辑变得复杂。现有方法要么通过操纵注意力机制来移除内容，要么通过噪声级干预来抑制条件控制。然而，这些方法牺牲了视觉连贯性，会在预留区域和非预留区域之间产生平淡的区域或明显的风格不一致。为了解决这一局限性，我们提出了提示感知共享潜变量免训练退火扩散采样（PASTA），这是一个免训练框架，通过在采样过程中选择性地向允许放置对象的区域应用语义引导，来强制执行预留区域约束。PASTA在共享潜变量轨迹上运行，其中完整提示与上下文……

    arXiv:2610.32390v1 Announce Type: cross  Abstract: While Diffusion models produce high-quality images, real-world visual design requires intentionally preserving clean regions for headlines, logos, or UI elements. Existing layout-aware text-to-image methods struggle to strictly preserve reserved regions, often introducing unwanted content that complicates downstream editing. Existing approaches either manipulate attention to displace content or suppress conditioning via noise-level interventions. However, these methods sacrifice visual coherence, producing flat regions or noticeable style inconsistencies between reserved and non-reserved areas. To address this limitation, we propose a Prompt-Aware Shared-Latent Training-Free Annealed Diffusion Sampling (PASTA), a training-free framework that enforces reserved-region constraints by selectively applying semantic guidance to object-allowed regions during sampling. PASTA operates on a shared latent trajectory, where full-prompt and context
    
[^66]: DealsFlow：面向风险投资交易筛选的证据门控智能体框架

    DealsFlow: An Evidence-Gated Agentic Harness for Venture Capital Deal Triage

    [https://arxiv.org/abs/2610.32380](https://arxiv.org/abs/2610.32380)

    DealsFlow是一个部署于英国早期基金的人机协同智能体框架，通过证据门控机制（阶段感知评分上限、怀疑者调用和预取模式）防止LLM投资评分虚高，在142笔三年以上的投资中评分与实际结果的Spearman相关系数达到0.45。

    

    大型语言模型（LLM）可以阅读初创公司的融资路演材料，但若让它们直接输出投资评分，会导致评分虚高、多次运行结果不一致以及推理依据难以审计等问题。我们介绍了DealsFlow，这是一个在英国一家早期投资基金部署的人机协同智能体框架。LLM负责阅读路演材料、引用证据并对各维度进行评分；代码则负责组合评分、设置上限，且只会下调评分。该框架通过三个机制实现：针对缺乏外部证据佐证的事实类别的分阶段感知佐证上限（满分100分中的78、70或60分）；一个只会降低而绝不会提高评分的“怀疑者”调用机制；以及在单次生成前预先获取并索引全部证据的预取模式。我们在N=243笔有记录结果的SEIS/EIS投资上进行评估。由于近期投资的结果尚未确定，我们选用了其中142笔至少三年以上的投资。在这些投资上，完整流程的评分与结果的Spearman相关系数ρ=0.45（95%置信区间0.32起）。

    arXiv:2610.32380v1 Announce Type: new  Abstract: Large language models (LLMs) can read startup pitch decks, but letting them emit investment scores directly invites score inflation, run-to-run variance and rationales that are hard to audit. We describe DealsFlow, a human-in-the-loop agentic harness deployed at a UK early-stage fund. The LLM reads the deck, cites evidence and rates each category; code combines the ratings, caps them and only ever lowers them. Three mechanisms implement this: a stage-aware corroboration ceiling (78, 70 or 60 out of 100) on factual categories without external evidence; a sceptic call that may lower but never raise a score; and a pre-fetch pattern that retrieves and indexes all evidence before a single generation pass. We evaluate on $N=243$ SEIS/EIS investments with recorded outcomes. Because recent investments have unresolved outcomes, we use the 142 that are at least three years old. There, the full pipeline's score has Spearman $\rho=0.45$ (95% CI 0.32
    
[^67]: 基于多模态智能体的非平稳响应科学仪器自适应控制

    Adaptive Control of Scientific Instruments with Nonstationary Responses through Multimodal Agents

    [https://arxiv.org/abs/2610.32370](https://arxiv.org/abs/2610.32370)

    本文提出一种具有物理基础、组件可独立审计的多模态智能体，通过确定性物理量提取与双层记忆机制，实现对响应非平稳的科学仪器（如磁约束聚变装置NBI离子源）的自适应逐炮调谐控制。

    

    控制科学仪器需要根据连续操作反馈逐步调整参数。当未被观测的运行条件随操作历史演变时，系统响应呈现非平稳性，此时从动作到响应的静态映射已无法支撑跨炮次的决策。磁约束聚变装置中中性束注入（NBI）离子源的逐炮调谐是这类问题的典型代表：电极表面与真空状态难以直接观测，放电结果在不同炮次之间存在差异，且实验成本高昂并带有运行风险。我们提出一种具有物理基础的多模态智能体，它包含三个可独立审计的组件：确定性算法从多通道波形中提取符号化物理量与视觉证据；一个双层记忆结构组织当前任务的轨迹和历史经验……

    arXiv:2610.32370v1 Announce Type: new  Abstract: Controlling scientific instruments requires parameters to be adjusted progressively in response to feedback from successive operations. When unobserved operating conditions evolve with operational history, responses become nonstationary, and a static mapping from actions to responses becomes inadequate for decisions across shots. Shot-to-shot tuning of neutral beam injection (NBI) ion sources in magnetic confinement fusion devices is representative of this class of problems: the conditions of electrode surfaces and vacuum are difficult to observe directly, discharge outcomes vary between shots, and experiments are costly and carry operational risks. We propose a physically grounded multimodal agent with three independently auditable components. Deterministic algorithms extract symbolic physical quantities and visual evidence from multichannel waveforms; a memory with two layers organizes trajectories from the current task, experience fro
    
[^68]: FOCUS：一种用于基于脑电认知负荷分类的轻量级Mamba-2模型

    FOCUS: A Lightweight Mamba-2 Model for EEG-Based Cognitive Workload Classification

    [https://arxiv.org/abs/2610.32361](https://arxiv.org/abs/2610.32361)

    提出了一种仅有4万参数的轻量级Mamba-2模型FOCUS，利用固定FFT五频带分解编码多通道脑电信号，在数据受限条件下实现了高效准确的认知负荷二分类。

    

    基于生理信号的认知负荷估计在高风险环境中日益重要，在这些环境中，持续且无干扰的监测可以为自适应系统和安全干预提供依据。本文提出了FOCUS（频带优化的认知负荷超轻量状态空间模型），这是一个基于Mamba-2的紧凑模型，用于在数据受限条件下进行基于脑电图的二分类认知负荷分类。FOCUS并非从数据中学习频率敏感性，而是采用覆盖五个经典脑电频带的固定FFT分解对预处理后的多通道脑电信号进行编码。所提出的模型结合了深度可分离卷积和权重共享的编码器-瓶颈式Mamba-2结构，参数量仅为0.04百万（4万），计算量为0.11 GFLOPs。在一项包含11名参与者的初步研究中，通过在三个认知上不同的任务上进行留一被试交叉验证评估，FOCUS在完整数据集上达到了74.75%的准确率，GPU模型前向延迟仅为0.08毫秒。

    arXiv:2610.32361v1 Announce Type: new  Abstract: Cognitive workload estimation from physiological signals is increasingly relevant in high-risk environments, where continuous and unobtrusive monitoring can inform adaptive systems and safety interventions. This paper presents FOCUS (Frequency-band Optimized Cognitive-load Ultralight SSM), a compact Mamba-2-based model for EEG-based binary workload classification under constrained data conditions. Rather than learning frequency sensitivity from data, FOCUS encodes preprocessed multi-channel EEG using a fixed FFT decomposition that covers the five canonical EEG bands. The proposed model combines depthwise-separable convolution and a weight-shared encoder-bottleneck Mamba-2 pair, with 0.04 million parameters and 0.11 GFLOPs. Evaluated in a pilot study with 11 participants under leave-one-subject-out cross-validation across three cognitively distinct tasks, FOCUS achieves 74.75% accuracy on the full dataset at 0.08 ms GPU model-forward late
    
[^69]: TENET：用于多元时间序列异常检测的时间点编码网络

    TENET: Time-point Encoding Network for Multivariate Time Series Anomaly Detection

    [https://arxiv.org/abs/2610.32337](https://arxiv.org/abs/2610.32337)

    TENET将无监督多元时间序列异常检测重新表述为当前点系统状态兼容性估计，通过以当前点为中心的时间编码和上下文自适应混合构建潜在状态表示，从而克服了观测空间差异作为异常指标不可靠的问题。

    

    在无监督多元时间序列异常检测（MTSAD）中，局部变化会放大重构和预测误差，使得观测空间差异成为判断系统异常行为的不可靠指标。由于潜在空间评分可以缓解这一局限，我们将单类多元时间序列异常检测重新表述为当前点系统状态兼容性估计问题，并提出TENET——用于多元时间序列异常检测的时间点编码网络。我们将当前点系统状态定义为目标时间点的潜在表示，该表示由输入窗口中的变量级时间上下文构建。TENET通过一个时间层来构建该状态：该层检索相对于当前点的窗口上下文，使用以当前点为中心的嵌入对时间进行编码，并将检索到的上下文与当前表示进行自适应混合。借助这一设计，TENET不再将输入窗口视为需要重构的对象（摘要在此处截断）。

    arXiv:2610.32337v1 Announce Type: cross  Abstract: In unsupervised multivariate time series anomaly detection (MTSAD), local variation can inflate reconstruction and forecasting errors, making observation-space discrepancy an unreliable indicator of abnormal system behavior. Since latent-space scoring can mitigate this limitation, we reframe one-class MTSAD as current-point system-state compatibility estimation and propose TENET, the Time-point Encoding Network for Multivariate Time Series Anomaly Detection. We define the current-point system state as a latent representation of the target time point, constructed from variable-level temporal context in the input window. TENET constructs this state through a temporal layer that retrieves window context relative to the current point, encodes time with current-point-centered embeddings, and adaptively mixes the retrieved context with the current representation. With this design, TENET treats the input window not as the object to reconstruc
    
[^70]: 基于触发器中和反事实评分的代码模型数据高效后门攻击

    Data-Efficient Backdoor Attacks on Code Models via Trigger-Neutralized Counterfactual Scoring

    [https://arxiv.org/abs/2610.32334](https://arxiv.org/abs/2610.32334)

    该论文提出触发器中和反事实评分（TNCS）方法，通过比较触发输入与触发器中和反事实样本上的模型输出差异来精确筛选投毒样本，在极低投毒率下即可对代码模型实现高成功率的后门攻击。

    

    预训练代码模型面临后门投毒攻击的威胁，攻击者通过构造恶意训练样本使模型在出现触发器时产生错误行为。一个“有能力的”攻击者会尽量减少此类样本的数量，因为每个样本都会增加成本和暴露风险，且更低的投毒率会给防御留下更小的痕迹。因此，选择对哪些样本进行投毒至关重要，但现有的数据高效样本选择方法主要针对图像分类任务，没有为自回归生成任务提供评分机制，也尚未在代码领域得到研究。我们提出了触发器中和反事实评分（TNCS）方法，通过比较评分模型在触发输入上的输出与其在触发器被中和的反事实样本上的输出，来分离触发器对目标行为的贡献，从而同时适用于分类和生成任务。在三个CodeXGLUE任务和五个触发器家族上，TNCS在极低投毒比例下取得了比基线方法更高的攻击成功率，且不会降低（模型性能）……

    arXiv:2610.32334v1 Announce Type: cross  Abstract: Pre-trained code models are exposed to backdoor poisoning, in which crafted training samples make the model misbehave on a trigger. A capable attacker minimizes the number of such samples, since each adds cost and exposure and a lower poisoning rate leaves a smaller footprint for defenses. Which samples to poison therefore matters, but existing data-efficient selection methods target image classification, offer no scoring mechanism for autoregressive generation, and have not been studied on code. We propose Trigger-Neutralized Counterfactual Scoring (TNCS), which isolates a trigger's contribution to the target behavior by comparing the scoring model's output on a triggered input with its output on a trigger-neutralized counterfactual, covering classification and generation alike. TNCS attains higher attack success rates than the baselines at very low poisoning ratios, across three CodeXGLUE tasks and five trigger families, without degr
    
[^71]: 4DMemory：基于工具增强智能体的长时程时空记忆基准测试

    4DMemory: Benchmarking Long-Horizon Spatiotemporal Memory with Harness-Augmented Agents

    [https://arxiv.org/abs/2610.32332](https://arxiv.org/abs/2610.32332)

    该论文提出了包含约900小时视频和5000多个人工审核问题的 LongVideo 数据集及 4DMemoryBench 基准，并开发了集成四种互补技能、22个以上专用工具模型和检索增强生成系统的 4DMemory Harness，用于评估和提升基础模型的长时程时空记忆推理能力。

    

    人类能够高效处理跨越数小时至数年的多模态数据流，构建持久、细粒度的时空记忆，以支持对长时程事件的推理与预测。相比之下，现有基准测试主要聚焦于短上下文和中等上下文任务，而大多数基础模型难以在长时程中保留细粒度信息。为了评估和改进长时程时空推理能力，我们提出了 LongVideo 数据集和 4DMemoryBench 基准。LongVideo 包含约 900 小时的视频和超过 5,000 个经人工审核的问题，涵盖不同上下文长度下的视觉、文本和时空推理。我们进一步开发了 4DMemory Harness，它集成了四种互补技能、22 个以上的专用工具模型以及针对特定模型的检索增强生成系统。通过对 13 个领先模型的评估，我们发现独立运行的基础模型在长时程任务上表现不佳，而 4DMemory Harness（摘要在此处截断）……

    arXiv:2610.32332v1 Announce Type: cross  Abstract: Humans efficiently process multimodal streams spanning hours to years, building persistent, fine-grained spatiotemporal memories that support reasoning about and predicting long-horizon events. In contrast, existing benchmarks focus on short- and medium-context tasks, and most foundation models struggle to preserve fine-grained information over long horizons. To evaluate and improve long-horizon spatiotemporal reasoning, we introduce the LongVideo dataset and the 4DMemoryBench. LongVideo comprises about 900 hours of video and over 5,000 manually reviewed questions covering visual, textual, and spatiotemporal reasoning across diverse context lengths. We further develop the 4DMemory Harness, which integrates four complementary skills, 22+ specialized tool models, and a model-specific retrieval-augmented generation system. Evaluating 13 leading models, we find that standalone foundation models struggle with long-horizon tasks, while the 4
    
[^72]: TGR-EvoDAG：腾讯面向生成式推荐的基于有向无环图（DAG）的多智能体代码演化框架

    TGR-EvoDAG: A DAG-based Multi-Agent Code Evolution Framework for Generative Recommendation on Tencent

    [https://arxiv.org/abs/2610.32326](https://arxiv.org/abs/2610.32326)

    该论文提出TGR-EvoDAG，一个针对腾讯生成式推荐场景的基于DAG的多智能体代码演化框架，通过将代码演化建模为有向无环图而非线性流程，解决了现有方法在模块级联流水线下的成本盲视与短视评估问题，实现更高效的自动化算法发现。

    

    大型语言模型（LLM）智能体已经使算法发现中的自动化代码演化成为可能。然而，当前面向序列推荐的递归自我改进（RSI）研究面临一个两难困境：通用框架缺乏领域特定的反馈，而推荐系统专用框架则依赖线性演化。至关重要的是，当应用于生成式推荐（GR）时，简单地将二者整合会遭遇失败。与端到端模型不同，生成式推荐依赖于模块级联的流水线（词元化、预训练与对齐）。在这一范式下，现有的演化方法存在以下问题：i）成本盲视，由于忽视各阶段不对称的再训练开销而浪费预算；ii）短视评估，丢弃了那些在下游适应完成之前暂时导致指标退化的有前景的上游创新。为弥补这些差距，我们提出了TGR-EvoDAG，一个为生成式推荐量身定制的多智能体代码演化框架。TGR-EvoDAG 将代码演化建模为……（摘要原文在此处截断）

    arXiv:2610.32326v1 Announce Type: new  Abstract: Large Language Model (LLM) agents have enabled automated code evolution for algorithm discovery. However, current recursive self-improvement (RSI) research for sequential recommendation faces a dilemma: general-purpose frameworks lack domain-specific feedback, while recommender-specific frameworks rely on linear evolution. Crucially, naively integrating them fails when applied to Generative Recommendation (GR). Unlike end-to-end models, GR relies on a module-cascaded pipeline (tokenization, pre-training, and alignment). In this paradigm, existing evolution methods suffer from: i) cost blindness, squandering budgets by ignoring stage-asymmetric retraining overheads; and ii) short-sighted evaluation, discarding promising upstream innovations that temporarily regress metrics before downstream adaptation. To bridge these gaps, we propose TGR-EvoDAG, a multi-agent code evolution framework tailored for GR. TGR-EvoDAG models code evolution as a
    
[^73]: ChoreoStudio：音乐驱动群舞编舞中的“做什么、何时做、谁来做”规划

    ChoreoStudio: Planning What, When, and Who in Music-Driven Group Choreography

    [https://arxiv.org/abs/2610.32325](https://arxiv.org/abs/2610.32325)

    ChoreoStudio通过将音乐解耦为节奏与结构特征，并让全模态模型扮演编剧和导演进行显式编舞规划，从而自动决定音乐驱动群舞中队形变化的内容、时机与执行者。

    

    音乐驱动的群舞生成不仅要决定每位舞者如何移动，还要决定群体应形成什么样的队形、何时改变队形，以及由谁来执行每一次队形转换。现有方法能够生成有序且节奏同步的群体动作，但在队形演化和舞者间交互方面仍然受限。在人类编舞中，队形转换通常与高层音乐结构的变化相吻合，例如从前奏进入副歌，或从副歌进入高潮段落。我们提出了ChoreoStudio，它使编舞规划显式化。音乐解耦器首先将音频分解为帧级节奏特征和段级结构特征。在给定段级特征和稀疏标签的条件下，一个冻结的全模态模型充当“编剧”角色，生成全局语义描述，用于指导“导演”的编舞规划。基于这一上下文，导演……

    arXiv:2610.32325v1 Announce Type: cross  Abstract: Music-driven group dance generation must decide not only how each dancer moves, but also what formation the group should form, when it should change, and who should execute each transition. Existing methods can generate orderly and rhythmically synchronized group motion, but remain limited in formation evolution and inter-dancer interaction. In human choreography, formation transitions often align with changes in high-level musical structure, such as the progression from an introduction to a chorus or from a chorus to a climax. We present ChoreoStudio, which makes choreography planning explicit. A music disentangler first decomposes the audio into frame-level rhythmic features and segment-level structural features. Given the segment-level features and sparse labels, a frozen omni-modal model acts as a scriptwriter, producing a global semantic description that guides the Director's choreography planning. Based on this context, the Direc
    
[^74]: 多模态强化学习中从视觉亲和度到学习信用

    From Visual Affinity to Learning Credit in Multimodal Reinforcement Learning

    [https://arxiv.org/abs/2610.32324](https://arxiv.org/abs/2610.32324)

    提出结果条件化视觉信用分配框架，通过将视觉关联度量与标记级信用分配分离，解决了多模态强化学习中均匀优势分配忽视视觉观测独特作用的问题。

    

    可验证奖励的强化学习（RLVR）能够提升多模态推理能力，但将单一的响应级优势均匀分配给所有标记（token）的做法，忽视了视觉观测与后续推理所扮演的不同角色。仅凭视觉亲和度是不够的：预测状态可能同时与图像表示和提示上下文表示对齐，而当后续推理失败时，对有效视觉观测进行加权反而会放大其负优势更新。为解决这一问题，我们提出了结果条件化视觉信用分配，这是一种即插即用框架，将视觉关联度量与标记级信用分配分离开来，并且能够适配于采用均匀响应级优势分配的策略优化方法。相对视觉编码通过对比每个预测状态与图像和提示上下文表示的亲和度，构建了一个上下文相对的视觉坐标系。结果条件化的证据校准（原文摘要至此截断）

    arXiv:2610.32324v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves multimodal reasoning, but uniformly assigning a single response-level advantage to all tokens overlooks the distinct roles of visual observations and subsequent reasoning. Visual affinity alone is insufficient: predictive states may align with both image and prompt-context representations, and upweighting valid visual observations can amplify their negative-advantage updates when later reasoning fails. To address this issue, we propose \textbf{Outcome-Conditioned Visual Credit (OVC)}, a plug-in framework that separates visual-association measurement from token-level credit assignment and is adaptable to policy-optimization methods with uniform response-level advantage allocation. Relative Visual Encoding constructs a context-relative visual coordinate by contrasting each predictive state's affinity to image and prompt-context representations. Outcome-conditioned Evidence Cali
    
[^75]: 超越增益：面向智能体技能退役的技能内化度量

    Beyond Gain: Measuring Skill Internalization for Agent Skill Retirement

    [https://arxiv.org/abs/2610.32322](https://arxiv.org/abs/2610.32322)

    该论文提出通过“隐去技能后评测+跨模型交叉验证高分”的方法可靠度量LLM智能体的技能内化程度，弥补仅凭增益判定会误删“未覆盖”技能的缺陷，从而实现准确的智能体技能退役。

    

    LLM智能体从拥有多达约10万条目、服务于众多模型的公共技能库中检索技能。随着模型能力的提升，技能库必须识别出给定模型已经内化的技能（即模型无需这些技能也能完成任务），并为该模型将这些技能退役，然而内化程度此前从未得到可靠度量。已发表的规则通过比较有技能与无技能时的运行表现，仅依据增益来决定退役。但问题在于，无论是已内化的技能，还是当模型与技能都无法提供相应能力时（我们称之为“未覆盖”状态），增益都接近于零。我们转而采用如下方法：先向模型隐去该技能，让模型在根据该技能合成的任务上进行评分，且只有当另一个模型能从该技能中获益时，才信任模型的无辅助高分，因为近半数的高分来自没有任何模型能从中获益的任务。在四个模型上的实验表明，将已发表的增益规则应用于每技能一项任务时，会移除超过一半的未覆盖技能。在SkillsBench上，其31项退役决定中有28项基于无任何运行的探针……（摘要在此处截断）

    arXiv:2610.32322v1 Announce Type: new  Abstract: LLM agents retrieve skills from public hubs that hold up to about 100,000 entries and serve many models. As models improve, a hub must find the skills a given model has internalized, meaning it can do without them, and retire them for that model, yet internalization has not been measured with known reliability. Published rules that compare runs with and without a skill retire on the gain alone. The gain is near zero both for an internalized skill and when neither the model nor the skill delivers the capability, a state we call uncovered. We instead withhold the skill, score the model on a task synthesized from it, and trust a high unaided score only when another model gains from the skill, since nearly half of high scores come from tasks no model benefits from. Under four models, a published gain rule applied with one task per skill removes over half of the uncovered skills. On SkillsBench, 28 of its 31 retirements rest on probes no run 
    
[^76]: Solvent：面向多步逆合成的长程推理

    Solvent: Long-Horizon Reasoning for Multi-Step Retrosynthesis

    [https://arxiv.org/abs/2610.32308](https://arxiv.org/abs/2610.32308)

    Solvent是一个8B参数的大语言模型，通过两轮后训练课程与强化学习将长程推理能力引入多步逆合成任务，学会交错执行迭代合成规划、外部可得性验证和自适应路线修正，性能提升2.1倍后在四个基准上取得61.86%的成绩。

    

    多步逆合成需要通过连续的反应步骤递归地将目标分子转化为可购买的构建模块，近期的研究已开始借助大语言模型（LLM）的推理能力来解决这一任务。在其最新的演进中，LLM推理已进入长程推理阶段：模型在相互依赖的多个步骤中追求目标、与环境交互并适应反馈。这一转变显著提升了LLM的能力，AI驱动的数学研究中的突破即是例证。在这一前沿方向上，我们推出了Solvent——一个8B参数的LLM，它成功地将长程推理扩展到了多步逆合成任务。通过两轮后训练课程，Solvent学会了交错执行迭代式合成规划、外部可得性验证以及自适应路线修正。实验表明，在强化学习带来2.1倍性能提升后，Solvent在四个基准上取得了61.86%的成绩。

    arXiv:2610.32308v1 Announce Type: new  Abstract: Multi-step retrosynthesis requires recursively transforming a target molecule through successive reaction steps into purchasable building blocks, and recent work has begun to draw on large language model (LLM) reasoning to tackle the task. In its latest evolution, LLM reasoning has entered the long-horizon regime: models pursue objectives across interdependent steps, interact with environments, and adapt to feedback. This shift has markedly advanced LLM capabilities, as illustrated by breakthroughs in AI-driven mathematical research. At this frontier, we introduce Solvent, an 8B LLM that successfully extends long-horizon reasoning to multi-step retrosynthesis. Through a two-round post-training curriculum, Solvent learns to interleave iterative synthesis planning, external availability verification, and adaptive route revision. Experiments show that, following a $2.1\times$ performance boost from RL, Solvent scores 61.86% across four benc
    
[^77]: DMAIC-Bench：多阶段分析任务中智能体从自我反思到自我克制

    DMAIC-Bench: From Self-Reflection to Self-Restraint for Agents in Multi-Stage Analytical Tasks

    [https://arxiv.org/abs/2610.32305](https://arxiv.org/abs/2610.32305)

    该论文提出 DMAIC-Bench 基准，通过基于精益六西格玛的成对反事实任务首次系统评估大语言模型智能体的“自我克制”能力（即依据证据决定继续还是停止），发现模型在 GO 决策上准确率达 92%，但在 NO-GO 决策上仅约 19%。

    

    多阶段分析任务要求大语言模型智能体利用中间证据来决定是继续、转向还是停止。自我反思（即智能体批判自身输出的过程）已被广泛研究，然而批判本身并不能表明证据是否改变了下一步行动。我们将这种能力称为“自我克制”。现有的智能体基准测试主要衡量任务完成度或输出质量，而未对这一能力进行检验。DMAIC-Bench 通过 97 个种子任务来评估这一能力，这些任务覆盖四个行业的 15 个工业案例，并围绕精益六西格玛明确的统计学前提构建。成对的反事实变体要求智能体在同一工作流位置采取相反的行动，从而将有依据的克制与不加区分的停止区分开来。我们使用私有评分器评估执行轨迹，并通过独立专家评审评估最终结论。在对来自六家提供商的 13 个模型的评估中，执行层面的路由准确率在匹配的 GO 决策上平均为 92%，而在 NO-GO 决策上仅为 19%。

    arXiv:2610.32305v1 Announce Type: new  Abstract: Multi-stage analytical tasks require LLM agents to use intermediate evidence to decide whether to continue, redirect or stop. Self-reflection, in which an agent critiques its output, is widely studied, yet a critique does not show whether evidence changed the next action. We call that capability self-restraint. Existing agent benchmarks largely measure task completion or output quality and leave it untested. DMAIC-Bench evaluates it with 97 seeded tasks across 15 industrial cases in four sectors, built around Lean Six Sigma's explicit statistical prerequisites. Paired counterfactual variants require opposite actions at the same workflow position, distinguishing warranted restraint from indiscriminate stopping. We assess execution traces with a private scorer and final claims through independent expert review. Across 13 models from six providers, execution-level routing accuracy averages 92% on matched GO decisions and 19% on NO-GO decisi
    
[^78]: 通过双向记忆-技能学习实现终身上下文适应

    Lifelong Context Adaptation through Bidirectional Memory-Skill Learning

    [https://arxiv.org/abs/2610.32295](https://arxiv.org/abs/2610.32295)

    提出MemCryst框架，通过记忆与技能的双向学习，让冻结的大语言模型智能体在有限上下文窗口内实现终身上下文适应——记忆中的重复情节支持技能创建与修订，技能引导的交互又产生新记忆，形成持续学习的闭环。

    

    终身学习的大语言模型智能体必须从不断扩展的交互历史中学习，同时需要在有限的上下文窗口内做出每个决策。记忆使过去的经验可用，但智能体仍可能需要从检索到的记录中推断如何解决每个新任务。技能提供了可复用的操作程序，却可能遗漏应用它们所需的任务特定事实。由于检索到的记忆和技能塑造了智能体的行为，它们也会影响未来学习可用的经验。我们提出了MemCryst，一个通过冻结语言模型进行记忆与技能双向学习来实现终身上下文适应的框架。存储在记忆中的重复情节支持技能的创建和修订，而这些技能所引导的交互又为后续学习提供新的记忆。从记忆中检索的事实和已学习的技能共同构成智能体的上下文，同时任务结果会调整记录在未来检索中的影响力。

    arXiv:2610.32295v1 Announce Type: new  Abstract: Lifelong LLM agents must learn from an expanding history of interactions while making each decision within a bounded context window. Memory makes past experience available, but the agent may still need to infer how to solve each new task from the retrieved records. Skills provide reusable procedures, yet may omit the task-specific facts needed to apply them. Because retrieved memory and skills shape the agent's behavior, they also influence the experience available for future learning. We introduce MemCryst, a framework for lifelong context adaptation through bidirectional learning of memory and skills with a frozen language model. Recurring episodes stored in memory support skill creation and revision, and the interactions guided by those skills supply new memory for subsequent learning. Facts retrieved from memory and learned skills jointly form the agent's context, while task outcomes adjust the future retrieval influence of the recor
    
[^79]: 结合图像结构补充DINO特征以实现部件发现

    Complementing DINO Features with Image Structure for Part Discovery

    [https://arxiv.org/abs/2610.32293](https://arxiv.org/abs/2610.32293)

    该论文揭示了基于DINO特征的部件发现边界模糊的根源在于ViT刚性补丁网格导致跨边界补丁混合了多个部件的像素，并提出用图像结构（来自边界检测器的边界信息）补充DINO特征，从而在无需重新训练的情况下实现更精确的部件发现。

    

    诸如DINO之类的自监督视觉Transformer（ViT）编码了丰富的语义信息，使得无需训练即可将物体从背景中分割出来，并通过特征聚类发现物体的部件。本文聚焦于部件发现任务：将前景物体划分为若干语义部件。现有方法产生的部件边界不够精确。我们将这一问题追溯到训练DINO时使用的刚性ViT补丁网格：跨越边界的补丁会包含来自多个部件的像素。虽然DINO基于不变性的训练本可以将这种像素级混合吸收为干净的特征，但通过对超过26K个前景补丁的实验表明事实并非如此：与边界重叠的补丁会产生熵更高、分配置信度更低的特征，从而使预测的部件边界变得模糊。与其投入巨大精力在几何连贯的区域上重新训练DINO，一种更简单的方案是直接从边界检测器获取边界信息……（原文摘要在此处截断）

    arXiv:2610.32293v1 Announce Type: cross  Abstract: Self-supervised Vision Transformers (ViTs) like DINO encode rich semantics, enabling training-free segmentation of objects from the background and discovery of their parts via feature clustering. This paper focuses on part discovery: partitioning a foreground object into semantic parts. Existing approaches produce imprecise part boundaries. We trace this to the rigid ViT patch grid used in training DINO, where patches straddling boundaries contain pixels from multiple parts. While DINO's invariance-based training could absorb this pixel-level mixing into clean features, experiments with over 26K foreground patches show it does not: boundary-overlapping patches produce features with higher entropy and lower assignment confidence, blurring predicted part boundaries. Rather than make the massive effort of retraining DINO over geometrically coherent regions, an easier approach is to take boundaries directly from a boundary detector, which 
    
[^80]: 轨迹审判：面向可靠性与失败归因的AI智能体框架基准测试

    Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution

    [https://arxiv.org/abs/2610.32285](https://arxiv.org/abs/2610.32285)

    提出TraceTrial-Bench基准，通过七大可靠性组件、95个专家级机制导向任务、轨迹追踪模块和双层评估框架，首次系统性地实现对AI智能体框架的可靠性与失败归因评估。

    

    大语言模型智能体（LLM agents）越来越多地被用于解决现实世界的问题，其有效性既取决于底层模型，也取决于支持其运行的框架。然而，现有的大多数基准测试主要关注模型能力和智能体的整体性能，对智能体框架的评估关注有限。为了填补这一空白，我们提出了TraceTrial-Bench，该基准将框架可靠性划分为七个组成部分，涵盖上下文、执行、治理、生命周期、可观测性、工具和验证。我们构建了95个专家级、面向机制的任务，并配有可执行或基于状态的基准真值，以支持语义等价的比较，同时保留每个框架的原生行为。我们设计了轨迹追踪（Trajectory Tracing, TT）模块来捕获智能体的执行轨迹，并引入了一个双层评估框架，分别评估智能体性能和框架可靠性。此外，我们还设计了一个多……（摘要在此处截断）

    arXiv:2610.32285v1 Announce Type: new  Abstract: LLM agents are increasingly used to solve real-world problems, and their effectiveness depends on both the underlying models and the harnesses that support their operation. However, most existing benchmarks focus on model capabilities and overall agent performance, with limited attention to the evaluation of agent harnesses. To address this gap, we introduce TraceTrial-Bench, which organizes harness reliability into seven components covering context, execution, governance, lifecycle, observability, tools, and verification. We construct 95 expert-level, mechanism-oriented tasks with executable or state-based ground truth to support semantically equivalent comparisons while preserving the native behavior of each harness. We design a Trajectory Tracing (TT) module to capture agent execution traces and introduce a two-layer evaluation framework that separately assesses agent performance and harness reliability. In addition, we design a Multi
    
[^81]: “Not”在翻译中并未丢失：多语言大语言模型中否定表征的因果分析

    "Not" is Not Lost in Translation: A Causal Analysis of Negation Representations in Multilingual LLMs

    [https://arxiv.org/abs/2610.32281](https://arxiv.org/abs/2610.32281)

    研究发现多语言大模型在各语言间共享可因果干预的否定表征，从一种语言提取的否定激活方向可以在其他语言中诱导或移除否定，且否定由稀疏特征表示。

    

    否定存在于每一种语言中，但其表层实现方式差异巨大。尽管如此，多语言大语言模型（LLMs）仍能稳健地翻译否定，这引出了一个值得探究的问题：它们是否在不同语言间共享某种底层的否定表征。我们在六个开源模型中，针对英语、法语、越南语和中文，通过在一个合成对比数据集以及从FLORES-200衍生的否定翻译数据集上进行因果干预来研究这一问题。在所有模型中，我们发现了强有力的证据，表明否定表征是因果可控的，并且在各语言间基本共享。特别地，我们展示了从一种语言提取的对比激活方向可以在其他语言中诱导或移除否定，且从不同语言提取的方向在实质上是相互对齐的。我们进一步发现，否定是以稀疏特征的形式表示的：每个模型中单个稀疏自编码器（SAE）特征即足以……

    arXiv:2610.32281v1 Announce Type: cross  Abstract: Negation occurs in every language, yet its surface realization differs drastically. Despite this, multilingual large language models (LLMs) translate negation robustly, raising the question of whether they share an underlying representation of negation across languages. We study this question in six open models across English, French, Vietnamese, and Chinese, using causal interventions on a synthetic contrastive dataset and a negation translation dataset derived from FLORES-200. Across models, we find strong evidence that negation representations are causally controllable and largely shared across languages. In particular, we show that a contrastive activation direction extracted from one language can induce or remove negation in the others, and directions extracted from different languages are substantially aligned. We further find that negation is represented in sparse features: a single sparse autoencoder (SAE) feature per model suf
    
[^82]: 托卡马克-SI：利用标记目录解读托卡马克放电

    Tokamak-SI: Interpreting Tokamak Discharges with Labeled Catalogs

    [https://arxiv.org/abs/2610.32264](https://arxiv.org/abs/2610.32264)

    Tokamak-SI将DIII-D托卡马克多种物理现象的标记和标记方法整合为统一、可扩展、时间对齐的数据集，基于它的模型在AE、ELM和TM检测等任务上达到或超越早期探测器。

    

    确定托卡马克放电中发生了哪些现象以及何时发生，通常是一项人工任务。以往对事件进行标记的工作都是针对特定任务的，每个工作都有自己独立的数据语料库。我们提出了Tokamak-SI，它将许多早期研究的标签和标记方法整合成一个单一的可扩展、时间对齐且分类化的数据集，涵盖DIII-D托卡马克中约束、脱靶、阿尔芬本征模（AE）、边缘局域模（ELM）、锯齿、撕裂模（TM）和电阻壁模等现象的观测数据。基于该数据集训练的模型能够将标签扩展到新的放电炮次，在AE、ELM和TM检测等标签任务上达到或超越早期的探测器。Tokamak-SI旨在支持特定任务的基准测试、物理可搜索的实验档案，以及在实验过程中直接识别关键事件。

    arXiv:2610.32264v1 Announce Type: cross  Abstract: Finding which phenomena occur in a tokamak discharge, and when, is often a manual task. Previous efforts at labeling events have been task-specific, each with its own corpus of data. We introduce Tokamak-SI, which combines the labels and labeling methods of many earlier studies into a single scalable, time-aligned, and categorical dataset covering observations from confinement, detachment, alfven eigenmodes (AE), edge localized modes (ELM), sawtooth, tearing modes (TM), and resistive wall modes in DIII-D tokamak. Models based on this, which can extend the labels to new shots, match or beat earlier detectors on its labels, including for AE, ELM, and TM detection. Tokamak-SI is intended to enable task-specific benchmarking, physics-searchable archives, and direct identification of key events during an experiment.
    
[^83]: 超越模式匹配的3D多模态大语言模型：面向可泛化识别的部件级功能推理

    Beyond Pattern Matching in 3D MLLMs: Part-Level Functional Reasoning for Generalizable Recognition

    [https://arxiv.org/abs/2610.32262](https://arxiv.org/abs/2610.32262)

    针对3D多模态大语言模型依赖模式匹配而无法区分几何相似物体的问题，本文提出PFIR方法，通过部件级功能推理构建“几何→功能→身份”的思维链数据，并以双目标GRPO渐进训练，显著提升了对最模糊样本的识别泛化能力。

    

    3D多模态大语言模型（3D MLLMs）通过模式匹配来识别物体，当几何形状相似的物体需要功能推理才能消歧时会失效（例如五斗柜与衣柜）。我们提出的逐样本几何混淆分数揭示了这一缺陷：在ModelNet40最模糊的20%样本上，基线准确率仅为33.7%，而在最不模糊的20%样本上则高达84.0%。我们识别出一种“功能作为标注”的模式，即功能性语言只是装饰已知的物体身份，而非支持推理过程，并据此假设：逆转这一信息流应能使改进集中于最模糊的样本上。PFIR（面向识别的部件级功能推理）正是实现了这一逆转。该方法基于PartNeXt部件分割构建，其数据流水线生成了50.7万个带有功能性思维链（几何→功能→身份）的问答对，并通过渐进式课程学习最终以双目标GRPO收尾：在Objaverse的多样物体上进行严格的功能推理与描述性 grounding。

    arXiv:2610.32262v1 Announce Type: cross  Abstract: 3D MLLMs recognize objects via pattern matching, which fails when geometrically similar objects require functional reasoning to disambiguate (e.g., dresser vs. wardrobe). Our per-sample Geometric Confusion Score (GCS) exposes this failure: baseline accuracy is 33.7% on ModelNet40's most ambiguous 20% versus 84.0% on its least ambiguous 20%. We identify a function-as-annotation pattern, in which functional language decorates a known identity instead of supporting inference, and hypothesize that reversing this information flow should concentrate improvements on the most ambiguous samples. PFIR (Part-level Functional Inference for Recognition) reverses that flow. Built on PartNeXt part segmentation, its data pipeline produces 507K QA pairs with functional CoTs (geometry $\to$ function $\to$ identity), and its progressive curriculum culminates in dual-objective GRPO: strict functional reasoning and descriptive grounding for Objaverse's div
    
[^84]: 异构空间数据中心：未来非地面与在轨服务的关键范式

    Heterogeneous Space Data Centers: A Key Paradigm for Future Non-Terrestrial and In-Orbit Services

    [https://arxiv.org/abs/2610.32260](https://arxiv.org/abs/2610.32260)

    本文提出由大小异构节点构成的空间数据中心新范式，定义了其用例、节点部署与空-空/空-地互连拓扑，并针对晨昏近地轨道配置分析太阳能供给、可见窗口与延迟，评估了服务延迟与可靠性。

    

    更低的空间发射成本与人工智能的进步，正在推动天基数据中心的发展，为在轨及非地面资产提供以AI为中心的计算与通信服务。然而，此类SDC的架构原则、服务模型和能源供给，尤其是在近地轨道（LEO）环境下的实现，在很大程度上仍属空白。我们提出了由大小节点组成的异构空间数据中心，这些节点在计算能力、尺寸和功率预算上有所区分，以支持覆盖地面、空中和在轨用户的各类应用。我们定义了相关用例、节点规模与部署策略，以及整合空-空链路与空-地链路的互连拓扑。聚焦于晨昏轨道LEO配置，我们分析了太阳能电力可用性、可见时间窗口以及延迟感知的覆盖能力。基于典型场景的基线工作负载，我们评估了延迟和服务可靠性，并量化了不同配置的能源需求。

    arXiv:2610.32260v1 Announce Type: cross  Abstract: Lower space-launch cost and advances in AI are driving space-based data centers (SDCs) that provide AI-centric computing and communication services to in-orbit and non-terrestrial assets. However, architectural principles, service models, and energy provisioning for such SDCs, particularly in low-Earth orbit (LEO), remain largely unexplored. We introduce heterogeneous SDCs composed of large and small nodes, differentiated by computing capability, size, and power budget, to support applications spanning ground, aerial, and in-orbit users. We define use cases, node sizing and placement strategies, and interconnection topologies integrating space-to-space and space-to-ground links. Focusing on dawn-dusk LEO configurations, we analyze solar power availability, visibility windows, and latency-aware coverage. Using baseline workloads for typical scenarios, we evaluate latency and service reliability. We quantify energy requirements of differ
    
[^85]: MedCaseAgent：从多模态临床证据到罕见病病例报告

    MedCaseAgent: From Multimodal Clinical Evidence to Rare Disease Case Reports

    [https://arxiv.org/abs/2610.32258](https://arxiv.org/abs/2610.32258)

    提出MedCaseAgent端到端智能体推理框架，能从非结构化临床笔记自动生成发表级质量的罕见病病例报告，并同步发布大规模多模态语料库MedCase-150K和时间留出评估基准MedCase-Bench。

    

    尽管大型语言模型（LLM）已在临床文档记录方面展现出潜力，但为前所未有的、非典型性的罕见疾病病例报告生成所需的严格的、基于证据的推理仍然是一项重大挑战。我们提出了MedCaseAgent，这是一种端到端的智能体推理范式，通过从非结构化临床笔记中综合生成发表级质量的报告来弥合这一差距。MedCaseAgent采用分层结构，将患者层面的数据挖掘与迭代式的、工具集成的推理相结合。通过动态查询医学文献和知识库，该智能体保持了高度的临床保真度并评估病例的新颖性。该框架具有天然的模块化特性，支持以即插即用的方式使用各种骨干LLM。为促进这一研究，我们发布了MedCase-150K——一个大规模多模态临床病例报告语料库，以及MedCase-Bench——一个基于时间留出的稳健评估套件。

    arXiv:2610.32258v1 Announce Type: new  Abstract: Although Large Language Models (LLMs) have demonstrated potential in clinical documentation, generating the rigorous, evidence-based reasoning required for unprecedented, atypical, and rare disease case reports remains a significant challenge. We present MedCaseAgent, an end-to-end agentic reasoning paradigm that bridges this gap by synthesizing publication-quality reports from unstructured clinical notes. MedCaseAgent leverages a hierarchical structure, integrating patient-level data mining with iterative, tool-integrated reasoning. By dynamically querying medical literature and knowledge bases, the agent maintains high clinical fidelity and assesses case novelty. The framework is inherently modular, supporting various backbone LLMs in a plug-and-play fashion. To facilitate this research, we release MedCase-150K, a large-scale multimodal clinical case-report corpus, and MedCase-Bench, a robust temporally held-out evaluation suite. We al
    
[^86]: 基于复合子空间流匹配的表示学习

    Representation Learning via Composite Subspace Flow Matching

    [https://arxiv.org/abs/2610.32256](https://arxiv.org/abs/2610.32256)

    提出复合子空间流匹配（CSFM），通过在光谱、颜色和光照子空间上对确定性退化课程进行方差最小化求逆，并以硬性时间风格截断约束轨迹方差，使编码器忽略瞬态风格、编码持久内容，从而弥合了联合嵌入与生成式自监督范式之间的权衡。

    

    从原始图像数据中提取语义信息仍然是机器学习领域的一个挑战。当前的自监督学习范式面临权衡：联合嵌入架构通过激进地丢弃类内方差来产生相关特征，不可逆地损失了空间与重建信息；相反，生成模型保留了空间细节，却难以分离语义，因为其优化过程被像素变化所主导。为了弥合这一鸿沟，我们提出了复合子空间流匹配（CSFM）。CSFM 并非使用非结构化的各向同性噪声来破坏数据，而是将表示学习表述为跨光谱、颜色和光照子空间进行的确定性退化课程的方差最小化求逆。通过硬性时间风格截断来约束轨迹方差，容量受限的编码器被驱动去忽略瞬态的风格伪影，并编码持久的内容信息。在……（原文此处被截断）上的实证评估……

    arXiv:2610.32256v1 Announce Type: cross  Abstract: Distilling semantic meaning from raw image data remains a challenge in machine learning. Current self-supervised paradigms face a trade-off: joint-embedding architectures produce relevant features by aggressively discarding intra-class variance, irreversibly spatial and reconstruction. Conversely, generative models preserve spatial details but struggle to isolate semantics, as their optimization is dominated by pixel variations. To bridge this divide, we propose composite subspace flow matching (CSFM). Rather than corrupting data with unstructured isotropic noise, CSFM formulates representation learning as the variance-minimizing inversion of a deterministic degradation curriculum across spectral, color, and illumination subspaces. By bounding trajectory variance with hard temporal style truncation, a capacity-constrained encoder is driven to ignore transient style artifacts and encode persistent content. Empirical evaluations across C
    
[^87]: 面向分布外检测的抗相干性正则化绝对缩放方法

    Anti-coherence Regularized Absolute Scaling for Out-of-Distribution Detection

    [https://arxiv.org/abs/2610.32255](https://arxiv.org/abs/2610.32255)

    提出绝对值缩放修复与抗相干性正则化，解决了SCALE方法因负激活正负抵消而在ViT等骨干网络上性能退化的问题，无需训练数据即可提升分布外检测性能。

    

    分布外（OOD）检测被公认为AI安全中的一个关键问题。SCALE是激活整形家族中一种有效的后处理OOD检测方法，它无需训练数据，并通过基于百分位的幅度比对倒数第二层激活进行重新缩放。然而，当存在负激活时，该比值定义不良，因为正负贡献在底层求和中相互抵消，导致SCALE在ViT等骨干网络上性能退化。我们通过一种“绝对值修复”来解决这一问题，即在绝对值化的激活上计算比值，而非通过ReLU丢弃负激活。基于这一修正后的尺度，我们进一步提出抗相干性正则化，通过惩罚样本中与最大logit符号不匹配的最大logit贡献占比，从而抑制对预测类别不可靠的证据。尽管无需访问训练数据……（原文摘要截断）

    arXiv:2610.32255v1 Announce Type: cross  Abstract: Out-of-distribution (OOD) detection is recognized as a critical problem in AI safety. SCALE is an effective post-hoc OOD detection method in the activation-shaping family that requires no training data and rescales penultimate activations using a percentile-based magnitude ratio. However, this ratio is ill-defined when negative activations are present, since positive and negative contributions cancel within the underlying sum, causing SCALE to degenerate on backbones such as ViT. We address this with an Absolute Fix that computes the ratio over absolute-valued activations rather than discarding negative activations through ReLU. Building on this corrected scale, we propose Anti-coherence regularization, which penalizes the scaling factor by the fraction of a sample's max-logit contributions to the max logit that are sign-mismatched and thereby suppresses unreliable evidence for the predicted class. Despite requiring no access to traini
    
[^88]: RSIHub：一个面向智能体演化方法的可组合框架

    RSIHub: A Composable Framework for Agent-Evolution Methods

    [https://arxiv.org/abs/2610.32253](https://arxiv.org/abs/2610.32253)

    RSIHub是一个开源可组合框架，通过共享基础设施统一支持AHE、Hyperagents、A-Evolve和GEPA等智能体演化方法的构建、比较与扩展，无需修改框架核心即可实现，最高可带来13.5个百分点的pass@1提升。

    

    智能体演化方法利用执行反馈来改进提示词、技能、工具和编排，但它们的实现往往需要重复构建相同的实验机制。我们提出了RSIHub，一个用于在共享基础设施上构建、比较和扩展这些方法的开源框架。配方组合可复用的算子并声明可变资源；RSIHub提供执行、评估、恢复和谱系追踪功能。研究人员可以专注于演化策略，同时评估保持在可变表面之外，结果与候选方案和评估条件保持绑定。我们在不修改框架核心的情况下实例化了AHE、Hyperagents、A-Evolve和GEPA，将方法特定的逻辑保留在配方和算子中。我们在固定的模型和评估器配置下，使用MiniSWE和Codex在Terminal Bench 2和τ³ Banking上对这些方法进行评估。观察到的最大整体pass@1提升为13.5个百分点，而方法排名则随条件变化。

    arXiv:2610.32253v1 Announce Type: new  Abstract: Agent-evolution methods use execution feedback to improve prompts, skills, tools, and orchestration, but their implementations often rebuild the same experimental machinery. We present RSIHub, an open-source framework for building, comparing, and extending these methods on shared infrastructure. Recipes compose reusable operators and declare mutable resources; RSIHub supplies execution, evaluation, recovery, and lineage. Researchers can focus on evolution policy while evaluation stays outside the mutable surface and results remain tied to candidates and evaluation conditions. We instantiate AHE, Hyperagents, A-Evolve, and GEPA without modifying the framework core, keeping method-specific logic in recipes and operators. We evaluate them with MiniSWE and Codex on Terminal Bench 2 and $\tau^3$ Banking under fixed model and evaluator configurations. The largest observed overall pass@1 gain is 13.5 percentage points, while method rankings var
    
[^89]: Easy-3Dfy：面向单图像3D场景构建的物体级几何组合方法

    Easy-3Dfy: Object-Level Geometric Composition for Single-Image 3D Scene Construction

    [https://arxiv.org/abs/2610.32250](https://arxiv.org/abs/2610.32250)

    Easy-3Dfy提出一种组合式工作流程，将专用物体生成与3D基础模型的场景布局估计解耦，并通过基于RPF的对齐器在物体网格与部分场景点图之间建立几何对应，无需完整场景级监督即可实现高质量的单图像3D场景构建，在多个几何指标上超越SAM-3Dfy。

    

    单图像3D场景构建是3D视觉领域的一个基础问题。近期的方法（如SAM-3Dfy）能够对3D资产及其场景布局进行端到端重建，但在细粒度的场景几何和资产质量方面仍存在不足。与此同时，专用的3D物体生成器可以提供精细的物体几何，而3D基础模型则具备强大且可泛化的空间理解能力。我们提出Easy-3Dfy，这是一种组合式工作流程，将物体生成与场景布局估计解耦，使各专用模型能够充分发挥自身优势。随后，基于RPF的对齐器通过在物体局部网格与部分场景点图之间建立几何对应关系，将这些能力衔接起来，恢复物体的空间布局以完成场景组合。在无需完整场景级监督的情况下，所提出的工作流程在Scene-Maker等数据集的多个几何指标上均持续优于SAM-3Dfy。

    arXiv:2610.32250v1 Announce Type: cross  Abstract: Single-image 3D scene construction is a fundamental problem in 3D vision. Recent methods such as SAM-3Dfy enable end-to-end reconstruction of 3D assets and their scene layouts, but still struggle with fine-grained scene geometry and asset quality. Meanwhile, specialized 3D object generators provide detailed object geometry, while 3D foundation models offer strong and generalizable spatial understanding. We propose Easy-3Dfy, a compositional workflow that decouples object generation from scene layout estimation, allowing specialized models to fully exploit their respective strengths. An RPF-based aligner then bridges these capabilities by establishing geometric correspondence between object-local meshes and partial scene pointmaps, recovering object layout for scene composition. Without requiring complete scene-level supervision, the proposed workflow consistently outperforms SAM-3Dfy across multiple geometric metrics on Scene-Maker and
    
[^90]: 最小描述长度在自我改进科学智能体中无监督地区分定律级知识与统计级知识——并揭示其自身86%的发现属于统计级

    Minimum Description Length Unsupervisedly Separates Law-Grade from Statistical-Grade Knowledge in a Self-Improving Science Agent - and Reveals 86% of Its Own Discoveries Are Statistical

    [https://arxiv.org/abs/2610.32245](https://arxiv.org/abs/2610.32245)

    最小描述长度压缩比可作为一种无监督的认知评级器，从自我改进科学智能体的发现中精准识别出真正的物理定律，同时揭示其86%的发现仅为统计级知识而非定律级知识。

    

    能够发现科学关系的AI系统如今可以大规模生成候选“定律”，但瓶颈已从生成转移到审查环节：哪些候选值得相信？我们展示了一个已部署的自我改进科学智能体——它已通过留出预测法庭机制接纳了252个关系——本身已经在计算能回答这一问题的量：用于模型选择的最小描述长度压缩比，在离散记账方式下读取时，可充当一种无监督的认知评级器。该智能体账本中按压缩比排名的最高区间恰好恢复了其发现中的全部经典定律（涵盖七大行星的活力公式、巴尔末/里德伯公式）以及跨测量恒等式——在超几何检验下富集显著性 p=3.6×10⁻¹²——且零反例，并呈现出物理上连贯的区间结构。同一视角还回溯性地揭示，账本中86%的内容并未压缩任何信息（cr≈1）：

    arXiv:2610.32245v1 Announce Type: new  Abstract: AI systems that discover scientific relations now generate candidate "laws" at scale, but the bottleneck has shifted from generation to vetting: which candidates deserve belief? We show that a deployed self-improving science agent - one that has admitted 252 relations through held-out prediction courts - already computes the quantity that answers this question: the Minimum Description Length compression ratio used for model selection doubles, when read under discrete accounting, as an unsupervised epistemic grader. The compression-ranked top band of the agent's ledger recovers exactly the canonical laws among its discoveries (vis-viva across seven planets, Balmer/Rydberg) plus cross-measurement identities - enrichment $p=3.6\times10^{-12}$ under a hypergeometric test - with zero counterexamples and physically coherent band structure. The same lens retroactively reveals that 86% of the ledger compresses nothing ($\mathrm{cr}\approx 1$): t
    
[^91]: 基于空间启发式策略的扩散强化学习用于在线三维装箱问题

    Spatial Heuristic Strategy based Diffusion Reinforcement Learning for Online 3D Bin Packing

    [https://arxiv.org/abs/2610.32241](https://arxiv.org/abs/2610.32241)

    本文提出一种融合空间启发式策略的扩散强化学习方法，将经典装箱启发式规则与扩散强化学习相结合，以增强在线三维装箱中的空间拓扑感知能力和布局优化效果。

    

    传统的启发式方法依靠人工设计的规则逻辑，在三维装箱的空间利用率和布局紧凑性等核心目标上积累了成熟的经验。然而，它们在泛化能力和对动态场景的适应性方面表现出明显的局限性。扩散强化学习融合了强化学习的空间搜索能力和扩散学习的数据分布建模能力，在动作分布建模和决策合规约束方面取得了一定进展。然而，仍然存在对三维装箱问题中空间拓扑结构感知不足以及布局优化能力有限等问题。为了解决这些问题，我们提出了一种融合空间启发式策略的扩散强化学习方法，该方法将经典的装箱启发式规则与扩散重构机制相结合……

    arXiv:2610.32241v1 Announce Type: new  Abstract: Traditional heuristic methods have accumulated mature experience in core objectives, such as space utilization and layout compactness of 3D bin packing, relying on artificially designed rule-based logic. However, they exhibit obvious limitations in generalization ability and adaptability to dynamic scenarios. Diffusion reinforcement learning, which integrates the spatial search capability of reinforcement learning and the data distribution modeling capability of diffusion learning, has achieved certain progress in action distribution modeling and decision compliance constraints. Nevertheless, there remain issues including insufficient perception of the spatial topological structure in 3D bin packing and limited layout optimization capability. To address these problems, we propose a diffusion reinforcement learning method integrated with spatial heuristic strategies, which combines classical bin packing heuristic rules with a diffusion re
    
[^92]: 面向基于Actor-Critic智能体的金融情境丰富策略：一种用于股票交易的深度强化学习方法

    Financial Context-Rich Strategy for Actor-Critic based Agents: A Deep Reinforcement Learning Approach for Stock Trading

    [https://arxiv.org/abs/2610.32237](https://arxiv.org/abs/2610.32237)

    本文提出了一种面向基于Actor-Critic智能体的金融情境丰富策略，利用深度强化学习方法将社会情绪、政治不稳定等外部因素纳入股票交易策略，以捕捉传统价格指标无法捕捉的市场动态。

    

    在现代金融市场中，超过四分之三的大宗交易是由基于先进交易策略的算法交易执行的，其精度和速度均优于传统的人类交易者。传统的基于价格指标的策略不足以捕捉由现实世界事件（如社会情绪、政治不稳定或突然的经济变化）所引发的市场动态。随着深度强化学习（DRL）的兴起和数据的日益可用，人们需要更高效的算法，能够将这些外部因素纳入交易策略之中。DRL在模拟真实世界场景和简化复杂决策方面展现出了巨大潜力，因为DRL智能体可以自动化交易过程，即以与人类交易者相当的方式执行交易。由于它们能够根据风险和定量因素更新自身策略，因此表现出可靠的性能。

    arXiv:2610.32237v1 Announce Type: new  Abstract: In modern financial markets, more than 3/4th of the large volume trade is executed by algorithmic trading based on advanced trading strategies to attain higher precision and speed than traditional human traders. The conventional strategies focused on price-based indicators are inadequate in capturing the market dynamism arising from real-world events such as social sentiments, political instability, or sudden economic shifts. With the rise of Deep Reinforcement Learning (DRL) and data availability, there is a requirement for more efficient algorithms that can incorporate these external factors in trading strategies. DRL has shown potential in simulating real-world scenarios and in simplifying complex decision-making since DRL agents can automate the trading process i.e. executing trades in a manner comparable to human traders. They have shown reliable performance since they can update their policies based on risk and quantitative factors
    
[^93]: 用MXFP4训练MoE大语言模型：我们如何缩小与BF16的差距

    Training an MoE LLM in MXFP4: How We Closed the Gap to BF16

    [https://arxiv.org/abs/2610.32229](https://arxiv.org/abs/2610.32229)

    该论文提出ALTO开源训练方案，通过结合混合1D/2D块量化、随机Hadamard变换、随机舍入与简化的权重去振荡技术，使200亿参数MoE模型GPT-OSS-20B在MXFP4低精度格式下训练的验证损失仅比BF16基线高0.007，大幅缩小了FP4训练与BF16全精度训练之间的差距。

    

    以低于8比特的精度算术训练大语言模型（LLM）有望在内存带宽和计算吞吐量方面带来显著收益，然而FP4格式引入的量化误差非常严重，若不加处理足以破坏优化的稳定性。我们提出了ALTO，一个开源的训练方案，用于在MLPerf Small MoE Training基准上以MXFP4微缩放格式训练GPT-OSS-20B——一个拥有200亿参数的混合专家模型。该方案将Abecassis、Felix等人[arXiv:2509.25149]的成熟技术（混合1D/2D块量化、随机Hadamard变换和随机舍入）与从TetraJet-v2 [arXiv:2510.27527]改进而来的简化权重去振荡方案相结合。在全局批大小为16的C4验证集上，MXFP4 + RHT + SR + 去振荡的组合在16,128步时达到3.3350的验证损失，仅比BF16基线（3.3283）高0.007，相比未使用去振荡的MXFP4，弥补了约50%的差距。

    arXiv:2610.32229v1 Announce Type: new  Abstract: Training large language models (LLMs) in sub-8-bit arithmetic promises substantial gains in memory bandwidth and compute throughput, yet FP4 formats introduce quantization error severe enough to destabilize optimization if left unaddressed. We present ALTO, an open-source training recipe for GPT-OSS-20B, a 20-billion-parameter mixture-of-experts (MoE) model, on the MLPerf Small MoE Training benchmark under the MXFP4 microscaling format. The recipe combines established techniques from Abecassis, Felix et al. [arXiv:2509.25149] (hybrid 1D/2D block quantization, randomized Hadamard transforms, and stochastic rounding) with a simplified weight de-oscillation scheme adapted from TetraJet-v2 [arXiv:2510.27527]. On C4 validation at global batch size 16, MXFP4 + RHT + SR + de-oscillation reaches a validation loss of 3.3350 at 16,128 steps, only 0.007 above the BF16 baseline (3.3283), closing roughly 50% of the gap left by MXFP4 without de-oscill
    
[^94]: 正确答案，紧闭的大门：面向验证屏障下GUI智能体的SolveBind框架

    Correct Answers, Closed Gates: SolveBind for GUI Agents at the Verification Barrier

    [https://arxiv.org/abs/2610.32223](https://arxiv.org/abs/2610.32223)

    该论文提出SolveBind框架，通过分层求解树（感知-定位-求解-行动-验证）与运行时监督相结合，使GUI智能体能够突破验证码等反自动化机制构成的“验证屏障”，将正确的求解输出转化为被系统接受的交互，从而完成端到端任务。

    

    网站和应用程序部署了验证码（CAPTCHA）及反自动化机制，这些机制利用交互轨迹和时机等行为信号——而非仅凭结果——来区分人类与自动化智能体。我们将这类受保护的GUI环境称为“验证屏障”（verification barrier），在这种环境中，仅正确解决验证任务并不足以完成端到端任务。智能体必须将求解器的输出转化为被系统接受的交互，并在时间限制下根据运行时反馈调整执行过程。我们提出SolveBind，一个将分层求解与运行时监督相结合的GUI智能体框架。分层求解树将验证过程分解为相互依赖的感知、定位、求解、行动与验证五种能力。运行时监督器将行动承诺建立在工具生成的证据之上，随着证据到来不断修订，并在明确的恢复预算约束下调停执行。在九个验证码任务族和九个骨干模型上，SolveBind取得了最高的……

    arXiv:2610.32223v1 Announce Type: new  Abstract: Websites and applications deploy CAPTCHAs and anti-automation mechanisms that use behavioral signals such as interaction trajectories and timing, not outcomes alone, to distinguish humans from automated agents. We call these protected GUI environments, where correctly solving a verification task is insufficient for end-to-end task completion. Agents must turn solver outputs into accepted interactions and adapt execution to runtime feedback under time constraints. We introduce SolveBind, a GUI-agent framework combining hierarchical solving with runtime supervision. A hierarchical solving tree decomposes verification into interdependent Perceive, Ground, Solve, Act, and Verify capabilities. A runtime supervisor grounds action commitments in tool-derived evidence, revises them as evidence arrives, and mediates execution under explicit recovery budgets. Across nine CAPTCHA task families and nine backbone models, SolveBind achieves the highes
    
[^95]: RxnExBench：面向科学文献中结构化化学反应提取的视觉语言模型基准测试

    RxnExBench: Benchmarking Vision-Language Models for Structured Chemical Reaction Extraction from Scientific Literature

    [https://arxiv.org/abs/2610.32221](https://arxiv.org/abs/2610.32221)

    该论文提出了RxnExBench多模态基准，包含3,565个专家标注反应、统一的15角色/26字段数据模式以及将分子识别与高级推理解耦的Oracle-MolRec评估协议，并揭示了当前视觉语言模型在结构化化学反应提取任务中随模式复杂性和推理需求增加而性能显著下降的问题。

    

    筛查表和底物范围图是合成化学文献中广泛使用的信息密集型图表，然而从中自动提取结构化反应数据仍然具有挑战性。我们提出了RxnExBench，一个用于细粒度反应级评估的多模态基准。该基准基于高影响力化学期刊构建，包含3,565个专家标注的反应和2,361个分子结构。数据集采用统一模式，包含15种反应角色和26个核心字段，支持逆合成规划、产率预测和反应条件推荐等AI-for-Chemistry下游任务。RxnExBench还提供了基于推理难度的分类体系、模块化标注流水线以及Oracle-MolRec协议，该协议将分子识别与更高层次的推理解耦。对9个最先进视觉语言模型的评估表明，随着模式复杂性和推理需求的增加，模型性能显著下降。

    arXiv:2610.32221v1 Announce Type: new  Abstract: Screen Tables and Substrate Scopes are information-dense figures widely used in synthetic chemistry literature, yet automated extraction of structured reaction data from them remains challenging. We introduce RxnExBench, a multimodal benchmark for fine-grained reaction-level evaluation. Built from high-impact chemistry journals, it contains 3,565 expert-annotated reactions and 2,361 molecular structures. The dataset adopts a unified schema with 15 reaction roles and 26 core fields, supporting downstream AI-for-Chemistry tasks such as retrosynthetic planning, yield prediction, and reaction condition recommendation. RxnExBench further provides a reasoning-driven difficulty taxonomy, a modular annotation pipeline, and the Oracle-MolRec protocol, which disentangles molecular recognition from higher-level reasoning. Evaluation of 9 state-of-the-art VLMs shows that performance degrades substantially as schema complexity and reasoning demands i
    
[^96]: PEAR：面向大型视觉语言模型感知与探索的自适应优势重塑方法

    PEAR: Adaptive Advantage Reshaping for Perception and Exploration in Large Vision-Language Models

    [https://arxiv.org/abs/2610.32216](https://arxiv.org/abs/2610.32216)

    PEAR提出一种自适应优势重塑方法，以视觉依赖为基础信号、仅在高熵位置施加与视觉依赖高低挂钩的熵探索增益，从而同时提升大型视觉语言模型多模态推理中的感知与探索能力。

    

    有效的多模态推理要求模型既能可靠地利用视觉信息，又能充分探索多样化的推理路径。为实现这一目标，先前的工作将视觉依赖与词元熵相结合来重塑序列级优势，但用视觉依赖对词元熵的贡献进行缩放，可能导致视觉依赖较低的高熵词元被低估权重，尽管这些词元在推理探索中发挥着重要作用。为解决这一局限，我们提出了PEAR，一种用于提升多模态推理中感知与探索能力的自适应优势重塑方法。PEAR保留视觉依赖作为基础信号，并仅在高熵位置加入基于熵的探索增益，且视觉依赖越低，附加增益越大。随后，融合信号被映射为正的词元级权重，用以重塑序列级优势。在Qwen2.5-VL-3B/7B上的实验表明，PEAR取得了显著的性能提升。

    arXiv:2610.32216v1 Announce Type: new  Abstract: Effective multimodal reasoning requires a model to reliably utilize visual information while sufficiently exploring diverse reasoning paths. To achieve this goal, prior work combines visual dependency and token entropy to reshape sequence-level advantages, but scaling the contribution of token entropy by visual dependency can leave high-entropy tokens with low visual dependency underweighted, despite their important role in reasoning exploration. To address this limitation, we propose PEAR, an adaptive advantage reshaping method for improving perception and exploration in multimodal reasoning. PEAR retains visual dependency as the base signal and adds an entropy-based exploration gain only at high-entropy positions, with lower visual dependency leading to a larger additional gain. The fused signal is then mapped to positive token-level weights that reshape the sequence-level advantage. Experiments on Qwen2.5-VL-3B/7B show that PEAR achie
    
[^97]: 头维度对语言模型长上下文检索至关重要

    Head Dimension Matters in Language Models for Long-Context Retrieval

    [https://arxiv.org/abs/2610.32213](https://arxiv.org/abs/2610.32213)

    本研究发现在固定模型规模下，增大注意力头维度可通过提高检索头的比例显著提升语言模型的长上下文检索性能，且即使冻结注意力投影权重该结论依然成立，而头维度对预训练损失和短上下文任务性能影响甚微。

    

    多头注意力通过查询-键交互和值聚合来检索与组合上下文表示，其中头维度限制了每个注意力头的表示空间。在长上下文处理中，将固定的参数预算分配给更少但更宽的头是否能改善信息检索，目前仍不清楚。在本文中，我们研究了头维度如何影响长上下文检索任务的性能。我们的实证研究表明，头维度对预训练损失和短上下文非检索任务的性能影响不大，而在固定模型规模下，长上下文检索性能会随着头的变宽而提升。我们发现，拥有更宽头的模型具有更高比例的检索头，即有更多参数与检索功能相关联。我们的消融实验还表明，即使冻结注意力模块中的投影权重，检索性能仍会随头变宽而提升。

    arXiv:2610.32213v1 Announce Type: cross  Abstract: Multi-head attention retrieves and combines context representations through query--key interactions and value aggregation, with head dimension constraining the per-head representation space. In long-context processing, it remains unclear whether allocating a fixed parameter budget to fewer, wider heads improves information retrieval. In this paper, we investigate how head dimension affects the performance of long-context retrieval tasks. Our empirical studies reveal that pretraining loss and short-context non-retrieval task performance are not affected much by head dimension, while long-context retrieval performance improves with wider heads at fixed model sizes. We find that models with wider heads have a higher proportion of retrieval heads, associating more parameters with retrieval. Our ablation studies also show that the retrieval performance improves with wider heads even when the projection weights in attention modules are froze
    
[^98]: SOPO：苏格拉底引导的策略优化用于片段级幻觉检测

    SOPO: Socratic Guided Policy Optimization for Span-Level Hallucination Detection

    [https://arxiv.org/abs/2610.32212](https://arxiv.org/abs/2610.32212)

    SOPO提出了一种苏格拉底引导的多轮强化学习框架，通过教师模型提供基于真实标签的间接提示来引导推理轨迹，从而解决片段级幻觉检测中的定位模糊、困难样本奖励稀疏和长度膨胀问题。

    

    大语言模型（LLMs）已在各种自然语言处理任务中得到广泛应用，但存在幻觉问题，即生成不忠实或不一致的内容。虽然可验证奖励强化学习（RLVR）在推理任务中展现出有前景的应用，但将其应用于片段级幻觉检测面临定位模糊性、困难样本奖励稀疏以及不受控的长度膨胀等挑战。为解决这些问题，我们提出了苏格拉底引导的策略优化（SOPO），这是一个用于鲁棒幻觉检测的多轮强化学习框架。SOPO的关键创新在于苏格拉底反馈机制，其中教师模型通过提供从真实标签推断出的间接提示，在采样过程中引导推理轨迹。这种教师引导的方法使模型能够为困难样本生成更有效的推理轨迹，从而缓解奖励稀疏问题。

    arXiv:2610.32212v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have gained widespread adoption in various natural language processing tasks, but suffer from hallucination issues where they generate unfaithful or inconsistent content. While Reinforcement Learning with Verifiable Rewards (RLVR) has shown promising applications in reasoning tasks, its application to span-level hallucination detection is challenged by localization ambiguity, reward sparsity on hard examples, and uncontrolled length inflation. To address these issues, we introduce Socratic Guided Policy Optimization (SOPO), a multi-turn RL framework for robust hallucination detection. The key innovation of SOPO is a Socratic feedback mechanism, where a teacher model guides the reasoning trajectories during rollout by providing indirect hints inferred from the ground truth. This teacher-guided approach enables the model to generate more effective reasoning trajectories for hard examples, thus alleviating rew
    
[^99]: VLA模型知道自己会失败吗？物理AI系统中可解释的成功与失败模式

    Do VLAs Know They Will Fail? Interpretable Success and Failure Patterns in Physical AI Systems

    [https://arxiv.org/abs/2610.32208](https://arxiv.org/abs/2610.32208)

    本论文发现OpenVLA激活的稀疏自编码器（SAE）表示蕴含结果预测信息，可在机器人任务陷入不可恢复状态前预测失败，并通过识别与成败阶段相关的SAE特征及基于VLM的解释流程，揭示反复出现的行为模式。

    

    视觉-语言-动作（VLA）模型已成为强大的通用机器人策略，但其成功与失败仍然难以诊断和理解。在这项工作中，我们展示了OpenVLA激活的稀疏自编码器（SAE）表示包含可预测结果的信息，可用于在机器人进入不可恢复状态之前预测任务失败。此外，为了深入洞察哪些常见行为可能导致这些结果，我们识别了强激活与成功或失败特定阶段相关联的个体SAE特征，并通过定位这些特征表现出显著激活变化的时间区间，揭示了反复出现的行为模式。我们进一步开发了一个基于VLM的解释流程，为这些模式生成行为描述，并在保留的时间区间上验证了其与特征激活的对应关系。

    arXiv:2610.32208v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have emerged as powerful general-purpose robot policies, but their successes and failures remain difficult to diagnose and understand. In this work, we show that sparse autoencoder (SAE) representations of OpenVLA activations contain outcome-predictive information and can be used to predict task failure before the robot reaches an unrecoverable state. Moreover, to provide insights into what common behaviors may contribute to these outcomes, we identify individual SAE features whose strong activations are associated with specific stages of success or failure, and uncover recurring behavioral patterns by localizing temporal intervals where these features exhibit significant activation changes. We further develop a VLM-based interpretation pipeline to generate behavioral descriptions of these patterns and validate their correspondence to feature activations on held-out intervals.
    
[^100]: 基于学习式前瞻与平衡频带的扩散语言模型水印方法

    Watermarking Diffusion Language Models with Learned Foresight and Balanced Bands

    [https://arxiv.org/abs/2610.32205](https://arxiv.org/abs/2610.32205)

    提出了一种面向扩散语言模型的水印方法，通过学习平衡的词表频带和前瞻预测器，并以检测器视角对绿token去重计数，在同等文本失真下显著提升了水印检测强度。

    

    绿-红水印机制会偏向选择“绿名单”中的token，而绿名单由密钥根据前序token生成。然而，扩散语言模型在选择某个token时，其前序token往往尚未确定，且该token之后又会影响其相邻位置的绿名单。因此，先前的工作在每次选择时都向提高自身及相邻位置绿token期望数量的方向倾斜。在这一目标的基础上，我们首先学习先前工作中保持固定的两个组件：作为绿名单的词表频带，用于区分相互竞争的候选token并保持零分布的精确性；以及一个前瞻预测器，用于估计每个候选token会如何改变相邻位置的预测分布。其次，我们按照检测器的方式统计绿token，即每个不同的上下文-token对只计一次，因此重复内容不会带来任何收益。在WaterBench上使用LLaDA-8B-Instruct进行的实验表明，每个学习组件都能在同等失真水平下提升z分数，且两者的增益几乎是可叠加的。在采用强水印时，我们的方法……（原文摘要在此处截断）

    arXiv:2610.32205v1 Announce Type: new  Abstract: Green-red watermarks favor tokens from a green list that a secret key derives from the preceding tokens. Diffusion language models, however, often choose a token before its preceding tokens, and the token later shapes its neighbors' green lists. Prior work therefore tilts each choice to raise the expected number of green tokens at its own and its neighbors' positions. Building on this objective, we first learn two components that prior work keeps fixed: vocabulary bands that serve as green lists, separate competing candidates and keep the null distribution exact, and a foresight predictor that estimates how each candidate shifts its neighbors' predictions. Second, we count green tokens as the detector does, once per distinct context-token pair, so repetition earns nothing. On WaterBench with LLaDA-8B-Instruct, each learned component raises the z-score at equal distortion, and their gains are nearly additive. With a strong watermark, ours
    
[^101]: SWE-Shift：在动态演变的工作区中评估编程智能体

    SWE-Shift: Evaluating Coding Agents in Evolving Workspaces

    [https://arxiv.org/abs/2610.32196](https://arxiv.org/abs/2610.32196)

    提出 SWE-Shift 基准，用于评估编程智能体在任务执行中途遭遇代码、依赖和需求等变化时的表现，实验发现所有 17 个被测智能体系统在发生此类变化时成功率均显著下降。

    

    仓库级编程基准测试在很大程度上假设任务和环境在整个执行过程中保持不变。而在软件开发中，代码、依赖项、运行时状态和需求可能在智能体仍在工作时发生变化。我们提出了 SWE-Shift，一个用于评估编程智能体在执行中途遭遇变化时表现的基准。基于相同的保存历史和工作区，Clean（干净）延续在不干预的情况下继续进行，而与之匹配的 Shifted（转变）延续则会收到与任务相关的更新。成功需要完成当前任务、保留仍然有效的工作，并满足适用的状态和证据检查。该基准包含横跨 12 个变化族别的 120 个场景，每个场景在九个预定的干预位置进行评估。我们评估了 17 个编程智能体系统，所有系统在 Shifted 条件下的成功率均低于 Clean 条件。GPT-6 Astra 的成功率从 67.35% 降至 24.79%，配对成功率为 20.34%，工具/会话变化和过时证据是主要失败因素之一。

    arXiv:2610.32196v1 Announce Type: new  Abstract: Repository-level coding benchmarks largely assume that tasks and environments remain unchanged throughout execution. In software development, code, dependencies, runtime state, and requirements can change while an agent is still working. We introduce SWE-Shift, a benchmark for evaluating coding agents under such mid-execution changes. From the same saved history and workspace, a Clean continuation proceeds without intervention, while a matched Shifted continuation receives a task-relevant update. Success requires completing the current task, preserving still-valid work, and satisfying the applicable state and evidence checks. The benchmark contains 120 scenarios across 12 change families, each evaluated at nine scheduled intervention positions. We evaluate 17 coding-agent systems, all of which have lower Shifted than Clean success. GPT-6 Astra drops from 67.35% to 24.79%, with 20.34% paired success. Tool/session changes and stale evidenc
    
[^102]: Principia：面向科学发现的测试时假设到原理构建

    Principia: Test-Time Hypotheses to Principle Construction for Scientific Discovery

    [https://arxiv.org/abs/2610.32195](https://arxiv.org/abs/2610.32195)

    提出了基于大语言模型的Principia框架，通过从相关输入与观测结果中形成比较性因果假设、并将假设中反复出现的关系综合为归纳性原理，在测试时自动构建并持续完善科学原理，突破了科学发现中知识构建这一瓶颈。

    

    科学发现需要识别观测现象背后的科学原理，以指导进一步的探究。由于这些经验性原理在复杂的发现过程中往往是未知的，它们通过“假设到原理”的流程来构建，即在测试时的发现过程中，从逐步积累的观测中提炼出反复出现的因果关系。然而，构建这类知识需要大量的领域专业知识、时间和精力，使得这一转化阶段成为科学发现的瓶颈。为了解决这一局限，我们提出了Principia，一个基于大语言模型（LLM）的框架，能够在测试时构建并持续完善科学原理。借鉴人类研究人员发展原理的方式，Principia从相关输入及其观测结果中形成比较性因果假设，然后将这些假设中反复出现的关系综合为归纳性原理。通过应用所构建的原理……（原文摘要在此处被截断）

    arXiv:2610.32195v1 Announce Type: cross  Abstract: Scientific discovery requires identifying the scientific principles underlying observed phenomena to guide further investigation. As these empirical principles are often unknown in complex discovery processes, they are constructed via a hypotheses-to-principle pipeline to obtain recurring causalities from progressively accumulated observations during test-time discovery. However, constructing this knowledge requires substantial domain expertise, time, and effort, making this translation stage a bottleneck for scientific discovery. To address this limitation, we propose Principia, an LLM-based framework that constructs and continually refines scientific principles at test time. Drawing on how human researchers develop principles, Principia forms comparative causal hypotheses from related inputs and their observed outcomes, then synthesizes recurring relationships across these hypotheses into inductive principles. By applying the constru
    
[^103]: 双自适应Z分数PPO：基于动作置信度与归一化优势幅度的动态裁剪

    Dual-Adaptive Z-Score PPO: Dynamic Clipping Using Action Confidence and Normalized Advantage Magnitude

    [https://arxiv.org/abs/2610.32194](https://arxiv.org/abs/2610.32194)

    该论文提出PPO-DAZ3方法，通过动作置信度和归一化优势幅度这两个信号为每个样本动态设置PPO的裁剪阈值，同时保持标准PPO训练流程其余部分不变。

    

    近端策略优化（PPO）因其稳定性和易于实现而仍然是一种流行的在线策略强化学习算法。然而，其裁剪阈值是固定的，因此每个训练样本都获得相同的信任域宽度，而不论相应策略更新的可靠程度如何。本工作研究了PPO的自适应裁剪变体，将恒定的裁剪参数替换为基于策略和回报统计量为每个样本计算的阈值。其核心方法——双自适应Z分数PPO（PPO-DAZ3）——利用两个信号为每个样本设置裁剪区间：动作置信度（由当前策略下所选动作的概率给出）和更新置信度（由归一化优势幅度给出），而标准PPO训练流程的其余部分保持不变。在三个经典控制基准（LunarLander-v3、CartPole-v1和Acrobot-v1）上的评估表明……（原文摘要在此处截断）

    arXiv:2610.32194v1 Announce Type: cross  Abstract: Proximal Policy Optimization (PPO) remains a popular on-policy reinforcement learning algorithm owing to its stability and ease of implementation. Its clipping threshold, however, is fixed, so every training sample receives the same trust-region width regardless of how reliable the corresponding policy update is. This work examines adaptive-clipping variants of PPO in which the constant clipping parameter is replaced by sample-dependent thresholds computed from policy and return statistics. The central method, Dual-Adaptive Z-Score PPO (PPO-DAZ3), sets the clipping interval for each sample from two signals: action confidence, given by the probability of the selected action under the current policy, and update confidence, given by the normalized advantage magnitude. The rest of the standard PPO training pipeline is left unchanged. Evaluation on three classic control benchmarks (LunarLander-v3, CartPole-v1, and Acrobot-v1) indicates that
    
[^104]: 面向未来辅助使用的证据驱动型小型语言模型（SLM）行为预测架构

    Evidence-Grounded SLM Behavioral Prediction Architecture for Future Assistance Use

    [https://arxiv.org/abs/2610.32193](https://arxiv.org/abs/2610.32193)

    该研究提出了一种名为ELBP的四层证据驱动本地行为预测架构，利用小型语言模型在设备端预测用户未来的辅助使用需求，在保护隐私的同时满足高风险环境下的及时响应要求。

    

    用户在明确寻求帮助之前，通常会表现出交互行为的变化，这使系统能够预测未来的辅助使用需求。这在高风险环境中尤为重要，因为延迟的辅助可能带来运营、财务或安全方面的后果。然而，传统的大型语言模型（LLM）由于其计算和内存需求，通常不适合连续的设备端行为预测。此外，行为模式具有高度的个人化和设备特异性，这促使采用能够保护隐私和数据主权的本地处理方式。小型语言模型（SLM）提供了一种有前景的替代方案，但它们在低级别顺序交互事件上进行可靠推理的能力仍未得到充分探索。本研究引入了证据驱动的本地行为预测（ELBP），这是一种四层架构，结合了确定性交互处理、双视图行为表示、SLM（摘要在此处被截断）

    arXiv:2610.32193v1 Announce Type: new  Abstract: Users often exhibit changes in interaction behavior before explicitly seeking assistance, enabling systems to anticipate future assistance use. This is especially important in high-stakes environments, where delayed assistance can have operational, financial, or safety consequences. However, conventional Large Language Models (LLMs) are often unsuitable for continuous on-device behavioral prediction because of their computational and memory demands. Behavioral patterns are also highly individual and device-specific, motivating local processing that preserves privacy and data sovereignty. Small Language Models (SLMs) offer a promising alternative, but their ability to reason reliably over low-level sequential interaction events remains underexplored. This study introduces Evidence-Grounded Local Behavioral Prediction (ELBP), a four-layer architecture combining deterministic interaction processing, dual-view behavioral representation, SLM-
    
[^105]: Uni-Stream：一个用于语音合成与识别的统一流式语音语言模型

    Uni-Stream: A Unified Streaming Speech Language Model for Speech Synthesis and Recognition

    [https://arxiv.org/abs/2610.32187](https://arxiv.org/abs/2610.32187)

    提出了Uni-Stream——一个基于紧凑型预训练语音模型、以流式方式处理交错文本与语音的统一语音语言模型，通过两阶段训练在单一模型中同时实现媲美任务专用模型的语音识别与语音合成性能。

    

    统一流式语音语言模型旨在在单一模型中同时支持语音识别与语音合成，从而实现实时人机交互。然而，在同时支持语音-文本转换这两个方向的同时保持各任务的具体性能仍然具有挑战性。我们提出了Uni-Stream，一个基于紧凑型预训练语音模型构建的统一流式语音语言模型。Uni-Stream以流式格式处理交错的文本与语音，并通过两阶段训练实现了与任务专用模型相当的性能。我们还开发了一个用于流式训练的数据标注流水线，并将公开发布所生成的转录文本和词级时间戳。我们进一步考察了Uni-Stream在跨任务、跨模态下的表示，发现即使在同一统一模型架构中训练，语音和文本仍处于不同的潜在特征空间中，而每个模态内部的表示则……（摘要原文在此处截断）

    arXiv:2610.32187v1 Announce Type: cross  Abstract: Unified streaming speech language models aim to support speech recognition and synthesis within a single model, enabling real-time human-computer interaction. However, jointly supporting these two directions of speech-text conversion while preserving task-specific performance remains challenging. We propose Uni-Stream, a unified streaming speech language model built on compact pretrained speech models. Uni-Stream processes interleaved text and speech in a streaming format and achieves performance comparable to task-specific models through two-stage training. We also develop a data annotation pipeline for streaming training and will publicly release the generated transcriptions and word-level timestamps. We further examine Uni-Stream's representations across tasks and modalities and find that speech and text residence in distinct latent feature space even when trained in a unified model architecture, while representations within each mo
    
[^106]: 并非所有信息都值得询问：基于决策感知的大语言模型信息获取

    Not All Information Is Worth Asking For: Decision-Informed Information Seeking with LLMs

    [https://arxiv.org/abs/2610.32181](https://arxiv.org/abs/2610.32181)

    提出DIVA方法，让大语言模型基于证据的决策价值扣除成本后来评估每项检查是否值得进行，从而决定该查询什么信息以及何时停止，避免为无法改变决策或不值得的信息付费。

    

    大语言模型（LLM）越来越多地被用于在不完整信息下支持决策，在这种情况下，决定查询什么以及何时停止与最终答案同样重要。然而，LLM会陷入信息陷阱：它们会为那些无法改变决策或不值得其成本的检查（即昂贵的证据片段）付费，并且在证据已经充分的情况下仍然继续询问。为此，我们提出了决策感知的信息价值获取方法（Decision-Informed Value-of-information Acquisition，DIVA），这是一种基于证据决策价值的简单而有效的LLM信息获取程序。DIVA 1）让LLM通过简短的数值问题指定一个小型决策模型，即坏结果发生的可能性有多大，以及每项检查将如何改变这一概率；2）通过设想各候选检查的潜在结果能在扣除成本后对决策带来多大改善来评估每项候选检查的价值，当没有检查值得其成本时即停止。其变体DIVA-U进一步考虑了……

    arXiv:2610.32181v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support decisions under incomplete information, where deciding what to look up and when to stop matters as much as the final answer. However, LLMs fall into an information trap: They pay for checks, i.e., costly pieces of evidence, that cannot change their decision or are not worth their cost, and they keep asking when the evidence is already sufficient. To this end, we propose Decision-Informed Value-of-information Acquisition (DIVA), a simple yet effective procedure for LLM information seeking based on the decision value of evidence. DIVA 1) lets the LLM specify a small decision model, i.e., how likely a bad outcome is and how each check would shift this probability, through short numeric questions and 2) values each candidate check by how much its imagined outcomes improve the decision net of its cost, stopping when no check is worth its cost. Its variant, DIVA-U, further accounts 
    
[^107]: 稀疏历史搜索：基于不完整搜索日志与零成本代理的跨任务架构搜索

    Searching with Sparse Histories: Cross-Task Architecture Search from Incomplete Search Logs and Zero-Cost Proxies

    [https://arxiv.org/abs/2610.32179](https://arxiv.org/abs/2610.32179)

    提出一种风险鲁棒的跨任务神经架构搜索方法，通过冠军初始化与联合成对排序器预训练，将稀疏搜索日志与零成本代理相结合，显著缓解最坏情况下的失败并大幅提升最差性能。

    

    神经架构搜索（NAS）成本高昂，但跨任务迁移可以通过重用先前任务的知识来降低成本。然而，现有方法在稀疏搜索历史和任务特定迁移失败的问题上表现不佳。我们提出了一种风险鲁棒的策略，通过基于冠军的初始化和联合成对排序器预训练，将稀疏的源任务日志与零成本代理相结合。在TransNAS-Bench-101上，该方法缓解了严重的最坏情况失败，将Micro自编码任务的最差10%排名从290降低到43，将Macro房间布局任务的最差10%排名从465降低到45。在13个单元空间目标中，该方法在平均性能和最坏情况性能上均在10个目标上优于仅使用代理的排序器，同时保持了有竞争力的平均性能。

    arXiv:2610.32179v1 Announce Type: cross  Abstract: Neural architecture search (NAS) is expensive, but cross-task transfer can reduce costs by reusing knowledge from previous tasks. However, existing methods struggle with sparse search histories and task-specific transfer failures. We propose a method, a risk-robust strategy that combines sparse source logs with zero-cost proxies through champion-based initialization and joint pairwise-ranker pretraining. On TransNAS-Bench-101, the proposal mitigates severe worst-case failures, reducing worst-10% ranks from 290 to 43 for Micro autoencoding and from 465 to 45 for Macro room layout. Across 13 cell-space targets, it outperforms a proxy-only ranker on 10 targets in both mean and worst-case performance, while maintaining competitive average performance.
    
[^108]: Ψ-0.6：统一视觉-语言世界模型中的自适应思维流

    $\Psi$-0.6: Adaptive Streams of Thought in a Unified Vision-Language World Model

    [https://arxiv.org/abs/2610.32177](https://arxiv.org/abs/2610.32177)

    Ψ-0.6 是一个统一的视觉-语言世界模型，通过在共享的“思维流”推理过程中自适应地组合感知、语言、几何与模拟能力，在多视角空间推理上零样本超越领先的开放权重模型，并支持基于想象场景的反事实模拟。

    

    人类在问题展开的过程中，会灵活地结合不同形式的推理来思考物理世界。实现这种灵活性是人工智能领域一个重大的开放性挑战：各种能力往往分散在不同的专用模型之中，限制了它们的组合方式以及共同适应任务的能力。我们提出了 Ψ-0.6，一个统一的视觉-语言世界模型，它能够在一个我们称为“思维流”的共享推理过程中自适应地组合感知、语言、几何与模拟。这些思维流交替穿插图像、文本以及几何与运动的显式表示，模型可以通过统一的 token 接口生成并解释所有这些内容。Ψ-0.6 在多视角空间推理任务上零样本超越了领先的开放权重模型，并且通过基于几何中间表示的推理获得了进一步提升。它还支持反事实模拟：通过生成并解释想象中的场景来判断，例如，某……

    arXiv:2610.32177v1 Announce Type: new  Abstract: Humans reason about the physical world by flexibly combining different forms of inference as a problem unfolds. Achieving this flexibility is a major open challenge in AI: capabilities often remain separated across specialized models, limiting how they can be composed and jointly adapted to a task. We introduce $\Psi$-0.6, a unified vision-language world model that adaptively composes perception, language, geometry, and simulation within a shared reasoning process we call streams of thought. These streams interleave images, text, and explicit representations of geometry and motion, all of which the model can generate and interpret through a shared token interface. $\Psi$-0.6 surpasses leading open-weight models zero-shot on multi-view spatial reasoning, with further gains from reasoning through geometric intermediates. It also supports counterfactual simulation: generating and interpreting imagined scenes to determine, for example, wheth
    
[^109]: MG-REPA：面向生成式目标说话人提取的多粒度表示对齐

    MG-REPA: Multi-Granularity Representation Alignment for Generative Target Speaker Extraction

    [https://arxiv.org/abs/2610.32161](https://arxiv.org/abs/2610.32161)

    提出多粒度表示对齐方法MG-REPA，通过在帧级与SSL模型对齐、在话语级与说话人验证模型对齐，同时捕获目标语音的时变内容和全局说话人特征，从而加速生成式目标说话人提取模型的训练收敛。

    

    生成式目标说话人提取（TSE）模型能够实现高质量的提取效果，但其训练收敛速度相当缓慢。表示对齐（REPA）通过将隐藏状态与预训练的自监督学习（SSL）教师模型对齐来加速扩散Transformer的训练。然而，现有的语音与音频领域中的对齐方法仅在单一粒度上进行操作。对于TSE而言，帧级对齐使得目标语音的全局、时不变特性保持隐式状态，而话语级对齐则使时变内容保持隐式状态，尽管提取目标说话人两者都需要。我们提出多粒度表示对齐（MG-REPA），在帧级将隐藏状态与SSL模型对齐，并在话语级与说话人验证模型对齐。我们进一步对帧级目标应用时间归一化，去除其话语级统计信息，使每个分支专注于……

    arXiv:2610.32161v1 Announce Type: cross  Abstract: Generative target speaker extraction (TSE) models achieve high-quality extraction, but suffer from considerably slow training convergence. Representation Alignment (REPA) accelerates diffusion transformer training by aligning hidden states with a pre-trained Self-Supervised Learning (SSL) teacher. However, existing alignment in the speech and audio domain operates at a single granularity. For TSE, frame-level alignment leaves the global, time-invariant characteristics of the target speech implicit, whereas utterance-level alignment leaves the time-varying content implicit, even though extracting the target requires both. We propose Multi-Granularity Representation Alignment (MG-REPA), aligning the hidden states with an SSL model at the frame level and a speaker verification model at the utterance level. We further apply temporal normalization to the frame-level target, removing its utterance-level statistics so that each branch concent
    
[^110]: 稀疏性驱动的多任务神经网络功能模块化的涌现

    Sparsity-driven emergence of functional modularity in multitask neural networks

    [https://arxiv.org/abs/2610.32153](https://arxiv.org/abs/2610.32153)

    稀疏性正则化无需预定义模块边界即可促使多任务神经网络自发涌现功能模块化，形成可在深层选择性保留、重组的紧凑“功能token”子网络。

    

    神经网络能够学习多个任务，但学习过程如何将共享计算组织成专门化且可复用的组件，目前仍知之甚少。在这项工作中，我们证明稀疏性诱导的正则化能够在无需预定义模块边界的情况下，促进多任务网络中功能模块化的形成。通过在多层感知机和轻量级transformer中进行功能剖析，我们识别出了紧凑的、具有任务选择性的子网络，并将其命名为“功能token”（functional tokens）。在基于规则的任务族中，选择性地保留这些token可以维持其对应的功能，同时降低其他任务的表现。这种专门化在训练过程中优先在较深层中发展，而较浅的层则保留共享计算。正则化的强度与形式会塑造模块的紧凑性和专门化程度，而任务权重的设置则会改变功能的分配方式。此外，提取出的功能token还可以通过替代路径进行重组……

    arXiv:2610.32153v1 Announce Type: cross  Abstract: Neural networks can learn multiple tasks, but how learning organizes shared computation into specialized and reusable components remains poorly understood. In this work, we show that sparsity inducing regularization promotes functional modularity in multitask networks without predefined module boundaries. Using functional profiling in multilayer perceptrons and a lightweight transformer, we identified compact, task-selective subnetworks, termed functional tokens. Across rule-based task families, selectively retaining these tokens preserves their associated functions while reducing performance on other tasks. Such specialization develops preferentially in deeper layers during training, while earlier layers retain shared computation. Regularization strength and form shape module compactness and specialization, while task weighting alters functional allocation. Moreover, extracted functional tokens can be recombined through alternative pa
    
[^111]: 日本某大学医院本地部署大语言模型智能体的LLENS平台：开发与评估研究

    The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study

    [https://arxiv.org/abs/2610.32151](https://arxiv.org/abs/2610.32151)

    研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。

    

    摘要（arXiv:2610.32151v1，公告类型：交叉）：使用工具的大语言模型智能体可以分担占用医生时间的部分电子病历和案头文书工作，但在许多医院中，医疗信息系统运行在封闭网络内，因此模型和智能体框架也必须部署在其中。我们设计、实现并部署了LLENS——一个在北海道大学医院封闭网络内运行的临床支持AI智能体平台，部署于配备8块NVIDIA H200 GPU的单台主机上，采用开放权重模型Kimi K2.6（约1万亿参数），并使用基于Open WebUI重构的Web应用。该智能体框架包含10个组件，包括页面图像读取、全文附件输入、无需向量索引的路径寻址文档树、网络隔离沙箱、工件返回、生成输出的自检、子智能体以及已保存的操作流程。Kimi K2.6是从3个开放权重候选模型中选出的，并在2026年日本医师执业资格考试中获得了480/500的成绩。

    arXiv:2610.32151v1 Announce Type: cross  Abstract: Large language model agents that use tools could take on part of the electronic health record and desk work that occupies physicians' time, but in many hospitals the medical information systems run inside closed networks, so the model and the agent harness must also sit inside. We designed, implemented, and deployed LLENS, a clinical support AI agent platform that runs inside the closed network of Hokkaido University Hospital on one host with 8 NVIDIA H200 GPUs, the open-weight model Kimi K2.6 (~1 trillion parameters), and a web application rebuilt from Open WebUI. The harness has 10 components, including page image reading, full-text attachment input, a path-addressed document tree without a vector index, a network-isolated sandbox, artifact return, self-checking of generated output, subagents, and saved procedures. Kimi K2.6 was chosen from 3 open-weight candidates and scored 480/500 on the 2026 Japanese Medical Licensing Examination
    
[^112]: 面向Graphon上变分问题的隐式神经表示

    Implicit Neural Representations for Variational Problems on Graphons

    [https://arxiv.org/abs/2610.32147](https://arxiv.org/abs/2610.32147)

    该论文提出用隐式神经网络表示Graphon并通过梯度下降求解其上的变分问题，其创新在于结合多尺度正弦残差架构、经隐式微分的密度约束内嵌求解器以及对称性感知的蒙特卡洛估计器，服务于极值图论与稠密随机图大偏差理论。

    

    我们提出了一个用于求解Graphon上变分问题的神经探索框架，其中Graphon是指作为稠密图序列极限而出现的对称可测函数 $[0,1]^2 \to [0,1]$。此类问题在数学文献的两个领域中处于核心地位：其一是极值图论，它研究在给定约束下的图参数优化问题，尤其针对顶点数量巨大的图；其二是稠密随机图的大偏差理论，它通过受约束的Graphon优化来刻画稀有事件的结构。我们采用隐式神经网络来表示Graphon，并通过梯度下降来优化以Graphon为变量的目标函数。我们的设计结合了三个要素：一种偏向于尖锐、阶跃状Graphon的多尺度正弦残差架构；一个内嵌的求解器，用于强制施加单一经验密度约束，并通过隐式微分进行求导；以及对对称性敏感的蒙特卡洛估计器。

    arXiv:2610.32147v1 Announce Type: cross  Abstract: We propose a neural exploratory framework for variational problems over graphons, i.e., symmetric measurable functions $[0,1]^2 \to [0,1]$, that arise as limits of dense graph sequences. Such problems are central to two parts of the mathematical literature: extremal graph theory, which studies graph parameter optimisation under given restrictions, often for those graphs with a large number of vertices, and the large deviation theory of dense random graphs, which characterises the structure of rare events through constrained graphon optimisation. We represent graphons by implicit neural networks and optimise graphon objectives by gradient descent. Our design combines three ingredients: a multi-scale sinusoidal residual architecture biased toward sharp, step-like graphons; an embedded solver that enforces a single empirical density constraint and is differentiated by implicit differentiation; and symmetry-aware Monte Carlo estimators. On
    
[^113]: 重新思考 LoRA 初始化以实现稳健的非对称学习率

    Rethinking LoRA Initialization for Robust Asymmetric Learning Rates

    [https://arxiv.org/abs/2610.32142](https://arxiv.org/abs/2610.32142)

    该论文通过对 LoRA 动力学的信号-噪声分解，揭示了标准初始化下由初始化引起的噪声项是非对称学习率训练不稳定和性能崩溃的根源，从而主张重新设计 LoRA 初始化以实现稳健的非对称学习率。

    

    低秩适应方法对学习率（LR）的选择往往非常敏感。近期的实证研究表明，一旦学习率经过仔细调优，原始的 LoRA 仍然可以与许多已提出的 LoRA 变体保持竞争力。这将核心的实际问题从设计更多变体转变为降低学习率选择的成本和脆弱性。LoRA+ 通过非对称学习率解决了这一问题，对上投影矩阵 $B$ 使用比下投影矩阵 $A$ 更大的学习率。然而，比率 $\eta_B / \eta_A$ 本身成为一个额外的超参数：较大的比率可以改善适配效果，但在标准 LoRA 初始化下，它们也可能导致训练不稳定以及依赖比率的性能崩溃。我们通过对 LoRA 动力学的信号-噪声分解来识别这种脆弱性的来源。在标准初始化下，更新中包含一个由初始化引起的噪声项，其大小可能受到……（摘要原文在此处截断）

    arXiv:2610.32142v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) is often highly sensitive to learning-rate (LR) choices. Recent empirical studies suggest that, once LRs are carefully tuned, vanilla LoRA can remain competitive with many proposed LoRA variants. This shifts the central practical question from designing ever more variants to reducing the cost and brittleness of LR selection. LoRA+ addresses this issue with asymmetric LRs, using a larger LR for the up-projection matrix $B$ than for the down-projection matrix $A$. However, the ratio $\eta_B / \eta_A$ itself becomes an additional hyperparameter: large ratios can improve adaptation, but under the standard LoRA initialization they can also cause unstable training and ratio-dependent performance collapse. We identify the source of this brittleness through a signal--noise decomposition of LoRA dynamics. Under standard \initA, the update contains an initialization-induced noise term whose magnitude can be affected by
    
[^114]: 超越数据调度：基于损失重加权的混合数据快速LLM预训练

    Beyond Data Schedule: Fast LLM Pretraining on Data Mixtures by Loss-Reweighting

    [https://arxiv.org/abs/2610.32141](https://arxiv.org/abs/2610.32141)

    提出LARE方法，在保持数据暴露不变的前提下，根据运行中的损失统计自适应地重加权各领域的损失贡献，使单一检查点的知识与推理性能超越多阶段预训练的最佳结果，从而实现更快的LLM预训练。

    

    大型语言模型通常通过连续的预训练阶段获得专业能力，但这些收益可能以遗忘先前知识为代价。在这项工作中，我们研究了是否可以通过改变每个领域对学习的贡献强度，同时保持其数据暴露量不变，来改善这种权衡。我们发现，在受控的知识-数学实验中，损失重加权产生了一个单一的检查点，其表现超越了沿任一阶段轨迹所能达到的最佳知识和推理分数。这揭示了一种仅靠数据混合比例无法捕捉的独立控制手段。基于这一发现，我们提出了LARE（固定暴露下的损失自适应重加权），它根据运行中的损失统计自动调整各领域权重。受梯度噪声理论分析的启发，其归一化的逆平方根规则在不改变采样混合的情况下平衡了相对损失尺度。

    arXiv:2610.32141v1 Announce Type: cross  Abstract: Large language models often acquire specialized capabilities through successive pretraining stages, but these gains can come at the cost of forgetting earlier knowledge. In this work, we investigate whether such trade-offs can be improved by changing how strongly each domain contributes to learning while keeping its data exposure fixed. We find that, in controlled knowledge--mathematics experiments, loss reweighting produces a single checkpoint that surpasses the best knowledge and reasoning scores attained along either stage trajectory. This reveals an independent control that data-mixture ratios alone do not capture. Building on this finding, we propose LARE (Loss-Adaptive Reweighting under fixed Exposure), which automatically adjusts domain weights from running loss statistics. Motivated by a theoretical analysis of gradient noise, its normalized inverse-square-root rule balances relative loss scales without changing the sampled mix
    
[^115]: MPQ：训练后量化的消息传递视角

    MPQ: A Message-Passing View of Post-Training Quantization

    [https://arxiv.org/abs/2610.32127](https://arxiv.org/abs/2610.32127)

    该论文提出消息传递量化（MPQ），将训练后量化重新建模为离散码本上的后验推断，通过在权重块间传递软后验信息并在最后阶段才做硬判决，从而超越GPTQ等现有硬判决求解器，提升大语言模型低比特量化的精度。

    

    训练后量化（PTQ）是高效部署大语言模型的关键技术，因为它无需重新训练即可将权重压缩为低比特表示。现有的PTQ求解器，包括GPTQ、LDLQ和块坐标下降法，都会在考虑相邻权重块之前就把每个权重块固定为单个码字，使得接近最优的备选方案未被探索。在这项工作中，我们将逐行PTQ转化为对离散码本的后验推断问题，其中现有求解器对应于零温度、硬判决的极限情况，即每个块仅保留最优码字。我们提出消息传递量化（MPQ），它在推断过程中跨块保留软后验信息，并仅在输出阶段才固定为硬码字。MPQ首先使用向量近似消息传递来计算近似的逐块后验分布，然后通过精确的块坐标下降法来精炼硬判决初始化结果。在2比特……

    arXiv:2610.32127v1 Announce Type: new  Abstract: Post-training quantization (PTQ) is a key technique for efficiently deploying large language models, as it compresses weights into low-bit representations without retraining. Existing PTQ solvers, including GPTQ, LDLQ, and block coordinate descent, commit each weight block to a single codeword before its neighbors are considered, leaving near-optimal alternatives unexplored. In this work, we cast row-wise PTQ as posterior inference over discrete codebooks, in which existing solvers correspond to a zero-temperature, hard-decision limit retaining only the top codeword per block. We propose Message-Passing Quantization (MPQ), which preserves soft posterior information across blocks during inference and commits to hard codewords only at the output stage. MPQ first uses vector approximate message passing to compute approximate blockwise posteriors, then refines a hard-decision initialization by exact block coordinate descent. Across 2-bit lat
    
[^116]: ImageRSI：面向智能体图像生成的验证驱动递归自我改进

    ImageRSI: Verification-Driven Recursive Self-Improvement for Agentic Image Generation

    [https://arxiv.org/abs/2610.32120](https://arxiv.org/abs/2610.32120)

    ImageRSI 提出了一个保持图像生成器冻结的验证驱动递归自我改进框架，通过任务内五维验证修复、经 A/B 验证的跨任务经验库以及离线更新的评判器，并发布包含 961 条困难请求的 ImageReq-Bench 基准，显著提升了复杂文生图请求的可靠满足率。

    

    复杂的文本到图像请求仍然难以被可靠地满足。对图像进行验证和修复可以修正单个输出，但一次成功的修复并不一定能迁移到其他请求上。我们提出了 ImageRSI，一个保持图像生成器冻结不变的框架，它结合了三个组件：任务内部的五维验证与修复、跨任务共享的经验库，以及评判器的离线更新。候选经验只有在通过成对 A/B 验证后才会进入经验库，而评判器则基于积累的图像评判记录，通过监督微调和 GRPO 进行更新，并在复用前经过评估。我们还引入了 ImageReq-Bench，这是一个包含 961 条请求的基准测试，围绕真实世界图像生成请求中的困难案例构建，每条请求都配有评估清单。ImageRSI 在 MindBench 上（以 Nano Banana Pro 作为生成器）取得了 52.6% 的清单严格准确率（CSA），在 Imag（原文在此处截断）……

    arXiv:2610.32120v1 Announce Type: cross  Abstract: Complex text-to-image requests remain difficult to satisfy reliably. Verifying and repairing an image can fix an individual output, but a successful repair does not necessarily carry over to other requests. We present ImageRSI, a framework that keeps the image generator frozen and combines three components: five-dimensional verification and repair within each task, an experience library shared across tasks, and offline updating of the judge. Candidate experiences enter the library only after paired A/B validation, and the judge is updated with supervised fine-tuning and GRPO on accumulated image-judgment records and evaluated before reuse. We also introduce ImageReq-Bench, a benchmark of 961 requests built around hard cases in real-world image-generation requests, each paired with an evaluation checklist. ImageRSI achieves a Checklist Strict Accuracy (CSA) of 52.6% on MindBench (with Nano Banana Pro as the generator) and 65.66% on Imag
    
[^117]: Tampered ML：评估大语言模型对机器学习研究的判断力

    Tampered ML: Evaluating LLM Judgment of Machine Learning Research

    [https://arxiv.org/abs/2610.32114](https://arxiv.org/abs/2610.32114)

    该论文提出Tampered ML基准，通过将真实机器学习论文的实验提案与人为引入方法论缺陷的最小修改版本配对，来评估大语言模型识别研究方法缺陷的能力，发现前沿模型虽优于随机水平但远未达到可靠程度。

    

    加速AI对齐研究进展的一种途径是将对齐研究本身的部分工作自动化。这要求系统能够在难以监督的研究任务上可靠运行，包括识别研究提案中的缺陷，以及评估一个提议的实验是否能支持其预期结论。为了评估这一能力，我们引入了Tampered ML，这是一个包含1,858个条目对的基准，每个条目对由一个从已发表的机器学习论文中重构的实验提案，以及一个经过最小化编辑、引入了针对性方法论缺陷的反事实版本组成。模型必须在两种呈现顺序下识别出方法论上更合理的提案。一项针对分层样本的人类验证研究估计，86%的条目是有效的（95%置信区间：81-91%）。前沿模型的表现高于随机水平，但仍远未达到可靠程度（其中Opus 5达到77.9%的准确率）。它们对呈现顺序高度敏感，并且f……（摘要在此处被截断）

    arXiv:2610.32114v1 Announce Type: cross  Abstract: One approach to accelerating progress on AI alignment is to automate parts of alignment research itself. Doing so requires systems that can perform reliably on hard-to-supervise research tasks, including identifying flaws in research proposals and evaluating whether a proposed experiment supports its intended claim. To evaluate this capability, we introduce Tampered ML, a benchmark of 1,858 item-pairs, each consisting of an experiment proposal reconstructed from a published machine learning paper and a minimally edited counterfactual that introduces a targeted methodological flaw. Models must identify the more methodologically sound proposal in both presentation orders. A human validation study on a stratified sample estimates that 86% of items are valid (95% CI: 81-91%). Frontier models perform above chance but remain far from reliable (with Opus 5 achieving 77.9% accuracy). They are highly sensitive to the order of presentation and f
    
[^118]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^119]: 谁来验证工具？失真的格式化输出如何误导智能体及其验证器

    Who Verifies the Tools? How Warped Formatters Mislead Agents and Their Verifiers

    [https://arxiv.org/abs/2610.32110](https://arxiv.org/abs/2610.32110)

    论文通过生产RAG系统的真实案例与对照实验发现，工具输出格式化器遗漏关键信息会误导智能体和验证器，而仅需在输出中补全这些缺失细节，就能显著减少智能体的无依据声称。

    

    智能体和验证器通常依赖工具打印出的文本，而这些文本可能会歪曲工具实际执行的操作，即使工具本身的计算是正确的。我们在纽约Pilsudski研究所使用的生产级RAG研究系统中观察到了这一现象：工具的输出格式化器误导智能体向用户提供了错误或不完整的信息。每个工具都正确地完成了自己的任务，但其输出省略了搜索限制、结果数量上限或标识符命名空间等关键信息。我们还重现了一个在代码审查阶段、发布前被发现的未标注估计值。我们研究了这些格式化器是否会误导智能体和验证器，以及修复后的措辞能带来多大改善。我们向智能体提供了使用两种格式化器版本的工具结果：一个版本包含关键遗漏，另一个版本则清楚说明了所有缺失的细节。仅这一差异就改变了智能体的声称内容以及验证器所接受的内容。在四个失败案例中的三个里，智能体的无依据声称大幅下降（原文在此处截断）。

    arXiv:2610.32110v1 Announce Type: new  Abstract: Agents and verifiers often rely on the text a tool prints, and that text can misrepresent what the tool actually did, even when the tool's computation is correct. We saw this happen in a production RAG research system used by the Pilsudski Institute in New York, where tool output formatters misled an agent into giving wrong or incomplete information to users. Each tool did its job correctly, but its output omitted search limits, result caps, or identifier namespaces. We also recreated an unlabeled estimate caught in code review before release. We studied whether these formatters misled agents and verifiers and how much the repaired wording helps. We fed tool results to agents using two formatter versions: one with the key omissions and one with all missing details clearly stated. That difference alone changed what the agents claimed and what the verifiers accepted. Unsupported agent claims dropped substantially for three of the four fail
    
[^120]: PersonaScienceBench：利用多样化模拟研究人员评估科学智能体

    PersonaScienceBench: Evaluating Scientific Agents with Diverse Simulated Researchers

    [https://arxiv.org/abs/2610.32106](https://arxiv.org/abs/2610.32106)

    PersonaScienceBench通过引入具有不同角色档案的模拟研究人员，在任务目标和评分标准保持固定的前提下，评估科学智能体与多样化研究人员协作的能力，而不仅仅评估其完成任务的表现。

    

    科学智能体需要与专业知识和偏好各不相同的研究人员协作，但现有基准测试通常将协作者固定不变。我们提出了PersonaScienceBench：包含来自四个科学领域五个基准的200个任务。模拟研究人员根据角色档案提供上下文和反馈，而任务目标和评分标准保持固定。智能体只能通过对话来了解研究人员的偏好。在十八种模型配置的测试中，任务表现、事实恢复和自我重复揭示了不同的优势。对GPT 6 Astra的详细分析显示，智能体对不同研究人员档案的反应各不相同；交互轨迹展示了有针对性的澄清和工作流程修订。即使最终得分相同，这些行为也可能存在差异。因此，PersonaScienceBench在评估智能体能否完成任务的同时，还能评估它们与多样化研究人员协作的方式。

    arXiv:2610.32106v1 Announce Type: new  Abstract: Scientific agents collaborate with researchers whose expertise and preferences differ, but benchmarks usually hold the collaborator fixed. We introduce PersonaScienceBench: 200 tasks from five benchmarks across four scientific domains. Simulated researchers provide context and feedback according to persona profiles, while task objectives and grading remain fixed. Agents encounter researcher preferences only through conversation. Across eighteen modelconfigurations, task performance, fact recovery, and self-repetition reveal different strengths. Detailed GPT 6 Astra analyses show distinct agent responses across researcher profiles; interaction traces illustrate targeted clar ification and workflow revision. These behaviors can differ even when final scores match. PersonaScienceBench therefore evaluates how agents collaborate with diverse researchers alongside whether they complete the task.
    
[^121]: ReMedI：基于推理时细化与自蒸馏的可控分子优化

    ReMedI: Inference-Time Refinement and Self-Distillation for Controllable Molecular Optimization

    [https://arxiv.org/abs/2610.32103](https://arxiv.org/abs/2610.32103)

    该论文提出ReMedI方法，以分子语言模型作为基础提议者，结合高效的Metropolis-Hastings搜索策略与推理时细化及自蒸馏，在无需昂贵任务特定数据集的情况下实现可控的多目标分子先导化合物优化。

    

    在药物研发中，先导化合物优化必须在改善相互竞争的性质（例如提高渗透性、降低毒性）的同时，限制可能损害先导化合物药效的骨架变化。基于强化学习的后训练方法（如C-MORAL）通过微调分子语言模型来直接针对感兴趣的目标优化分子先导化合物，从而解决这一问题，但这类方法依赖于昂贵的任务特定数据集，并且在实现多目标优化（MOO）方面面临挑战。我们认为，预训练模型可能已经包含有用的分子编辑操作，而零样本生成无法高效地触及这些操作。受零样本生成搜索能力有限的启发，我们开发了一种高效的Metropolis-Hastings搜索策略，以更好地探索提议分布。我们提出了ReMedI，该方法利用分子语言模型作为基础提议者进行推理时对齐。给定一个先导化合物和一组多目标任务……

    arXiv:2610.32103v1 Announce Type: cross  Abstract: In drug development, lead optimization must improve competing properties, such as increasing permeability and reducing toxicity, while limiting scaffold changes that could compromise the lead's potency. RL-based post-training, such as C-MORAL, addresses this problem by fine-tuning molecular language models to directly optimize molecular leads for objectives of interest, which relies on expensive task-specific datasets and faces challenges in achieving multi-objective optimization (MOO). We argue a pre-trained model may already contain useful molecular edits that zero-shot generation cannot efficiently reach. Motivated by the limited search capabilities of zero-shot generation, we develop an efficient Metropolis-Hastings search strategy to better explore the proposal distribution. We propose ReMedI, which uses the molecular language model as a base proposer for inference-time alignment. Given a lead and a set of multi-objective tasks, t
    
[^122]: 面向通感一体化的跨尺度非线性波形融合学习

    Learning Nonlinear Waveform Fusion Across Scales for Integrated Sensing and Communication

    [https://arxiv.org/abs/2610.32099](https://arxiv.org/abs/2610.32099)

    该论文提出了一种跨尺度非线性波形融合的学习方法，通过自适应时间特征加权与频谱更新，将图像特征与线性调频参考信号融合于共享复数状态并保持固定总能量，从而在通感一体化系统中同时实现图像传输与多目标的距离-角度定位。

    

    固定的源信号叠加与频率划分方式限制了通感一体化中通信内容与探测参考信号之间的相互作用。我们学习跨尺度的非线性波形融合方法，用于图像传输以及多个目标在距离和角度上的定位。接收端利用各自可用的信息。自适应的时间特征加权和频谱更新将图像特征与确定性的线性调频参考信号结合在共享的复数状态中，生成在阵列帧上具有固定总能量的波形。在用户设备（UE）处，参考信号和信道状态信息对神经解混器进行条件化以实现图像重建。在基站处，实际发射的脉冲定义了一个相干字典，用于通过学习的局部校正、物理细化和残差更新进行序贯定位。我们将所提出的波形与叠加方法进行比较……（原文摘要在此处截断）

    arXiv:2610.32099v1 Announce Type: new  Abstract: Fixed source superposition and frequency partitioning constrain how communication content interacts with a probing reference in integrated sensing and communication (ISAC). We learn nonlinear waveform fusion across scales for image transmission and localization of multiple targets in range and angle. Reception uses the information available at each end. Adaptive temporal feature weighting and spectral updates combine image features and a deterministic linearly frequency-modulated reference in a shared complex state, generating a waveform with fixed total energy over the array frame. At the user equipment (UE), the reference and channel state information condition a neural demixer for image reconstruction. At the base station, the realized transmit pulse defines a coherent dictionary for sequential localization with learned local correction, physical refinement, and residual updates. We compare the proposed waveforms with superposition in
    
[^123]: DELA：为扩散语言模型学习可控编辑

    DELA: Learning Controllable Edits for Diffusion Language Models

    [https://arxiv.org/abs/2610.32096](https://arxiv.org/abs/2610.32096)

    DELA 是一个插件式框架，通过插入、删除、重新掩码和停止四种显式编辑操作，在冻结的扩散语言模型上实现可控的草稿级细化，将高层编辑控制与词元级生成解耦。

    

    扩散语言模型（DLMs）支持双向去噪和迭代细化，但其生成过程在很大程度上仍由词元级的掩码调度所控制。这使得草稿级的细化过程是隐式的，限制了对何时插入缺失内容、删除冗余片段、重新审视不确定区域或停止编辑的显式控制能力。我们提出了扩散编辑语言适配器（Diffusion Editing Language Adapter，DELA），这是一个插件式框架，能够在冻结的 DLM 上实现可控的草稿细化。DELA 使用四种显式编辑操作——插入、删除、重新掩码和停止——对干净的可变长度草稿进行操作。控制器决定何时以及在哪里编辑草稿，而冻结的 DLM 通过去噪来填充被掩码的区域，从而将高层级的编辑控制与词元级的生成过程分离开来。一个朴素的单阶段编辑器在计算上会非常昂贵，因为任意的内部编辑会破坏前缀稳定性，并限制可靠的 KV 缓存重用。为了增加……

    arXiv:2610.32096v1 Announce Type: new  Abstract: Diffusion language models (DLMs) support bidirectional denoising and iterative refinement, but generation remains largely controlled by token-level masking schedules. This leaves draft-level refinement implicit and limits explicit control over when to insert missing content, delete redundant spans, revisit uncertain regions, or stop editing. We introduce Diffusion Editing Language Adapter (DELA), a plug-in framework that enables controllable draft refinement with a frozen DLM. DELA operates on clean, variable-length drafts using four explicit editing actions: Insert, Delete, Remask, and Stop. The controller determines when and where to edit the draft, while the frozen DLM fills masked regions through denoising, separating high-level edit control from token-level generation. A naive single-phase editor would be computationally expensive because arbitrary interior edits can disrupt prefix stability and limit reliable KV-cache reuse. To add
    
[^124]: VLA模型中的测试时训练（TTT）：探索基于人类演示的训练

    TTT in VLA Models: Exploring Training with Human Demonstrations

    [https://arxiv.org/abs/2610.32094](https://arxiv.org/abs/2610.32094)

    该研究将测试时训练（TTT）集成到 InternVLA-A1 和 Pi0.5 两个VLA模型中，探索通过人类演示与机器人轨迹配对训练实现以演示为条件的动作生成，但发现在当前规模下该方法未能实现基于演示的一致目标选择，且各种改进策略均未带来明显提升。

    

    长时程机器人任务需要保留并利用过去的信息。我们将测试时训练（TTT）——一种将上下文存储在快速权重中的方法——集成到 InternVLA-A1 和 Pi0.5 中，通过在配对的演示与机器人轨迹上进行训练，研究以人类演示为条件的动作生成。我们实现了两个阶段之间的记忆传递以及截断反向传播，并在视频、短视频和单帧线索任务中考察了训练目标、梯度覆盖范围和对比正则化。在我们的实验中，在当前规模下添加 TTT 并在配对数据上训练，并未实现基于演示的一致目标选择。在所测试的设置下，更短的序列、完整轨迹的梯度连接以及对比正则化并未明显提升对演示的响应性。当演示被简化为单帧线索时，真实机器人的执行仍然表现出偏好……

    arXiv:2610.32094v1 Announce Type: cross  Abstract: Long-horizon robot tasks require retaining and using past information. We integrate Test-Time Training (TTT), which stores context in fast weights, into InternVLA-A1 and Pi0.5 to study action generation conditioned on human demonstrations through training on paired demonstrations and robot trajectories. We implement memory transfer between the two phases and truncated backpropagation, and examine training objectives, gradient coverage, and contrastive regularization across video, short-video, and single-frame cue tasks. In our experiments, adding TTT and training on paired data at the current scale did not lead to consistent target selection based on demonstrations. Under the tested settings, shorter sequences, full trajectory gradient connectivity, and contrastive regularization did not clearly improve responsiveness to demonstrations. When demonstrations were reduced to single-frame cues, real-robot execution still showed a preferenc
    
[^125]: AgentTrace：黑盒大语言模型智能体的执行级指纹识别

    AgentTrace: Execution-Level Fingerprinting of Black-Box LLM Agents

    [https://arxiv.org/abs/2610.32092](https://arxiv.org/abs/2610.32092)

    AgentTrace通过注入结构模糊的探测指令触发工具接口的序列化失败，并用轻量级马尔可夫链建模智能体后续的修复轨迹，从而在对抗性身份伪装下以96.0%的准确率识别黑盒LLM智能体背后的基础模型。

    

    验证自主大语言模型智能体的来源归属是一项关键的安全挑战。现有的语义归因方法容易被系统提示词混淆，而主动水印技术则与受模式约束的工具生成相冲突。我们提出了AgentTrace，这是一个通过可观察的执行轨迹来识别黑盒智能体背后基础模型的主动式框架。AgentTrace注入结构上模糊的探测指令，以在僵化的工具接口处触发对分词敏感的序列化失败。虽然这些初始失败能够在不同模型家族之间提供粗粒度的区分，但后续修复操作的时间序列则揭示了细粒度的、特定于模型的恢复行为。通过使用轻量级马尔可夫链对这些多步轨迹进行建模，AgentTrace在对抗性身份伪装下实现了96.0%的归属准确率。至关重要的是，序列修复建模成功区分了所评估的同家族模型……

    arXiv:2610.32092v1 Announce Type: cross  Abstract: Verifying the provenance of autonomous LLM agents is a critical security challenge. Existing semantic attribution is easily obfuscated by system prompts, while active watermarking conflicts with schema-constrained tool generation. We propose AgentTrace, an active framework that identifies the foundation model behind a black-box agent via observable execution traces. AgentTrace injects structurally ambiguous probes to trigger tokenization-sensitive serialization failures at rigid tool interfaces. While these initial failures provide coarse-grained separation across model families, the temporal sequence of subsequent repair operations reveals fine-grained, model-specific recovery behaviors. By modeling these multi-step trajectories with lightweight Markov chains, AgentTrace achieves 96.0% attribution accuracy under adversarial identity spoofing. Crucially, sequential repair modeling successfully distinguishes the evaluated intra-family v
    
[^126]: 面向大语言模型驱动智能体的行为分析

    Behaviour analysis for large language model-driven agents

    [https://arxiv.org/abs/2610.32087](https://arxiv.org/abs/2610.32087)

    提出了一个将数据驱动的行为发现、可解释的依赖关系分析与实验验证（含提示引导）相结合的通用框架，用于理解大语言模型驱动智能体在轨迹中如何组织行为及其与情境和结果的关系。

    

    理解语言智能体的行为需要学习动作如何在整个轨迹中被组织，以及这种组织如何与情境和结果相关联。我们提出了一个通用框架，将数据驱动的行为发现与分析及实验研究相结合。从记录的活动中开发的类别定义了所要测量的内容。分析保留了行为在过程中的位置，并将其关系表达为可解释的假设。统计评估和在保留情境中的测试检验了证据的强度和范围。随后，包括提示引导在内的实验测试了行为和结果在刻意变化的条件下如何改变。一项关于迭代式自我反思的研究展示了行为发现、依赖关系学习和提示比较如何能够相互启发。我们针对不同长度、事件顺序和智能体角色的轨迹发展了这一方法，并识别了在编码……（摘要在此处截断）

    arXiv:2610.32087v1 Announce Type: new  Abstract: Understanding language-agent behaviour requires learning how actions are organised across a trajectory and how this organisation relates to context and outcomes. We propose a general framework that connects data-driven behavioural discovery with analysis and experimental investigation. Categories developed from recorded activity define what is measured. Analyses preserve the behaviours' positions in the process and express their relationships as interpretable hypotheses. Statistical assessment and tests in held-out contexts examine the strength and scope of the evidence. Experiments, including prompt steering, then test how behaviour and outcomes change under deliberately varied conditions. A study of iterative self-reflection illustrates how discovery, dependency learning and prompt comparisons can inform one another. We develop this approach for trajectories of varying length, event order and agent roles, and identify advances in codin
    
[^127]: 面向匿名化人群视频监控的引导式零样本异常解释：上下文、分类与空间定位

    Guided Zero-Shot Anomaly Explanation for Anonymized Crowd Video Monitoring: Context, Categorization, and Spatial Localization

    [https://arxiv.org/abs/2610.32073](https://arxiv.org/abs/2610.32073)

    本文提出XAVAD方法，将零样本匿名化、异常类型分类、视觉定位与场景上下文描述统一整合，为匿名化视频异常检测提供可解释且保护隐私的多方面异常解释。

    

    尽管深度学习在视频异常检测（VAD）方面取得了令人瞩目的发展，但这些模型的可解释性（XAI）仍然是一个挑战。可解释且保护隐私的VAD能够建立透明度与信任，从而促进计算机视觉系统（包括安全与安防监控）的实际应用。然而，隐私保护型VAD系统中的异常可解释性仍未得到充分探索。在本研究中，我们提出了XAVAD方法，这是一种针对匿名化VAD的统一描述性异常可解释AI框架，它集成了灵活的零样本匿名化、异常类型分类、视觉定位和场景上下文描述。我们对结合全身视觉匿名化以保护隐私的多方面解释进行了实证与定性评估。我们采用面向视频的多模态CLIP嵌入，使用改进的LLM辅助提示和池化评分机制进行异常类型识别，并增强了Grad-ECLIP...

    arXiv:2610.32073v1 Announce Type: cross  Abstract: Despite the promising development of deep learning for video anomaly detection (VAD), explainability (XAI) of these models remains a challenge. Explainable and privacy-protecting VAD builds transparency and trust that facilitates adoption of computer vision systems, including safety and security monitoring. However, anomaly explainability in privacy-preserving VAD systems is underexplored. In this study, we present the XAVAD approach, a unified descriptive anomaly XAI for an anonymized VAD, that integrates flexible zero-shot anonymization, anomaly type categorization, visual localization, and scene context description. We provide empirical and qualitative assessments of multifaceted explanations integrated with full-body visual anonymization for privacy protection. We adopt a multimodal CLIP embedding for videos, using improved LLM-assisted prompting and pool-scoring mechanisms for anomaly-type identification. We enhance Grad-ECLIP for
    
[^128]: DriveMind：面向闭环自动驾驶的思维演化信念跟踪

    DriveMind: Mind-Evolving Belief Tracking for Closed-Loop Autonomous Driving

    [https://arxiv.org/abs/2610.32069](https://arxiv.org/abs/2610.32069)

    DriveMind借鉴粒子滤波思想，让基于MLLM的驾驶智能体同时维护多个场景假设并随新观测动态更新权重与重采样，从而显著提升闭环自动驾驶性能。

    

    近年来，多模态大语言模型（MLLM）能够在单一模型内完成对交通场景的感知、推理并提出驾驶动作。然而，现有的基于MLLM的驾驶系统在每一帧都从零开始推理，并且只坚持一种解释，丢弃了此前对场景的其他假设。当场景存在歧义时，智能体会在证据尚未积累完整之前就锁定错误的解读并据此行动。本文提出DriveMind，一个基于MLLM的驾驶框架，它能够同时保留多个关于场景的假设，并随着新帧的到来动态更新这些假设的权重。受粒子滤波思想启发，DriveMind通过视觉语言模型（VLM）传播每个假设，依据当前观测对其进行评分，并通过重采样保持假设的多样性。在Bench2Drive基准上，在相同的驾驶流程中加入信念跟踪后，驾驶分数（DS）从68.50提升至81.37，成功率（SR）从41.36%提升至54.09%。

    arXiv:2610.32069v1 Announce Type: new  Abstract: Recent multimodal large language models (MLLMs) can perceive a traffic scene, reason about it, and propose a driving action within a single model. However, existing MLLM-based drivers reason from scratch at each frame and commit to a single explanation, discarding earlier hypotheses about the scene. When the scene is ambiguous, the agent locks in on the wrong interpretation and acts on it before evidence accumulates. In this paper, we propose DriveMind, an MLLM-based driving framework that keeps several hypotheses about the scene alive at the same time and updates their weights as new frames arrive. Inspired by particle filtering, DriveMind propagates each hypothesis through a VLM, scores it against the current observation, and resamples to maintain diversity. On Bench2Drive, adding belief tracking to the same driving pipeline improves the Driving Score (DS) from 68.50 to 81.37 and the Success Rate (SR) from 41.36% to 54.09%. The largest
    
[^129]: EEG基础模型的受控评估：随机初始化、频带功率与基于验证集选择的探测层

    Controlled Evaluation of EEG Foundation Models: Random Initialisation, Band-Power and Validation-Selected Probes

    [https://arxiv.org/abs/2610.32067](https://arxiv.org/abs/2610.32067)

    该论文通过引入未训练编码器、频带功率基线和基于验证集的层选择三种对照，对多个EEG基础模型进行受控评估，发现其预训练收益大多不具统计稳健性，随机初始化和简单的频带功率基线即可与之匹敌。

    

    EEG基础模型报告了下游任务的性能提升，但其评估协议很少包含三种对照：即同一编码器未经训练的版本、在同一数据上拟合的频谱特征，以及不看测试分数而选择的层。因此，我们在统一协议下、采用全部三种对照和多个随机种子，在五个基准和四个附加任务族上评估了五个公开发布的EEG基础模型和一个自研的JEPA风格模型。这些对照从三个方面改变了评估结论。第一，若按测试分数选择探测层，逐层峰值平均提高+0.034，并在36个单元格中的9个里逆转了峰值与输出之间的差距。第二，当在验证集上选择层时，随机初始化在睡眠和情绪任务上与预训练持平甚至更优（涉及一半的模型），且没有任何模型在情绪或运动想象任务上胜过五频带对数功率基线。第三，在完全微调后，30对实验中仅有5对的增益超过其随机种子波动范围的2倍……

    arXiv:2610.32067v1 Announce Type: cross  Abstract: EEG foundation models report downstream gains, yet their protocols seldom include three controls. These are the same encoder untrained, spectral features fitted on the same data, and a layer picked without looking at test scores. We therefore evaluate five publicly released EEG foundation models and an in-house JEPA-style model under one protocol with all three controls and several seeds, on five benchmarks and four added task families. The controls change the reading in three ways. First, picking the probed layer by its test score raises the layer-wise peak by an average of +0.034 and reverses the peak-versus-output gap in 9 of 36 cells. Second, with the layer picked on validation, random initialisation matches or beats pretraining for half of the models on sleep and emotion, and no model beats a five-band log-power baseline on emotion or motor imagery. Third, after full fine-tuning only 5 of 30 pairs gain beyond twice their seed spre
    
[^130]: 理解多智能体大语言模型系统中的通信失败

    Understanding Communication Failures in Multi-Agent LLM Systems

    [https://arxiv.org/abs/2610.32060](https://arxiv.org/abs/2610.32060)

    该论文基于3,211条失败的多轮智能体交互轨迹，构建了多智能体LLM系统通信失败的分类法，并揭示失败主要由恢复/纠正与共识/决策问题主导，而非孤立的首轮错误。

    

    多智能体大语言模型系统旨在通过辩论、证据共享和相互纠正来提升推理能力，但智能体间的交互也可能引入仅凭最终准确率无法解释的失败。智能体可能过早达成一致、忽视异议、压制正确的少数派答案，或在多轮交互中保留错误的任务框架。我们提出了一个基于语料库的通信失败分类法，该分类法基于3,211条失败的多轮轨迹构建，涵盖医学问答、编程、金融、多跳推理、伦理和具身文本等任务。利用这一分类法，我们分析了失败在不同数据集和系统设计中的分布、轮次级的错误动态、恢复难度、多数派转变、少数派信号抑制以及智能体自我报告的置信度。我们发现，失败主要由恢复/纠正问题和共识/决策问题主导，而非孤立的首轮错误，且许多失败是在智能体首次观察到同伴响应之后才出现的。基于这些发

    arXiv:2610.32060v1 Announce Type: new  Abstract: Multi-agent LLM systems aim to improve reasoning through debate, evidence sharing, and mutual correction, but interaction can also introduce failures that final accuracy alone cannot explain. Agents may converge too early, ignore dissent, suppress correct minority answers, or preserve a wrong framing across rounds. We present a corpus-grounded taxonomy of communication failures, built from 3,211 failed multi-round trajectories across medical QA, coding, finance, multi-hop reasoning, ethics, and embodied text tasks. Using this taxonomy, we analyze failure distributions across datasets and system designs, round-level error dynamics, recovery hardness, majority transitions, minority-signal suppression, and self-reported confidence. We find that failures are dominated by recovery/correction and consensus/decision problems rather than isolated first-round mistakes, with many emerging after agents first observe peer responses. Based on these f
    
[^131]: 概念针灸师：一种基于稀疏自编码器的扩散模型高效精确概念擦除方法

    Concept Acupuncturist: An Efficient and Precise Concept Erasure Method for Diffusion Models via Sparse Autoencoders

    [https://arxiv.org/abs/2610.32059](https://arxiv.org/abs/2610.32059)

    该论文提出ConceptACP框架，利用稀疏自编码器的可解释性分析定位U-Net去噪器中与目标概念最相关的交叉注意力块并进行针对性微调，从而实现对扩散模型概念的高效精确擦除。

    

    基于扩散的图像生成模型能够生成高质量图像，但可能生成有害或违反政策的内容，这促使机器遗忘（MU）技术被用于选择性地移除目标概念，同时保留模型的通用能力。然而，现有的机器遗忘方法往往难以应对概念表示的分布式和纠缠特性，当通过在全扩散轨迹上进行大规模微调来应用时，会导致优化效率低下和编辑不精确。为了弥补这一差距，我们提出了概念针灸师（ConceptACP），这是一种新颖的机器遗忘框架，它将可解释性引导的分析与无需推理的微调相结合，以实现高效且精确的概念移除。通过建立语义概念与 U-Net 去噪器中交叉注意力块之间的映射，并定位与概念最相关的块进行针对性优化，ConceptACP 避免了全扩散轨迹的优化，从而提高了（效率与精度）。

    arXiv:2610.32059v1 Announce Type: cross  Abstract: Diffusion-based image generation models produce high-quality images but may generate harmful or policy-violating content, motivating machine unlearning (MU) to selectively remove target concepts while preserving general capabilities. However, existing MU approaches often struggle with the distributed and entangled nature of concept representations, leading to inefficient optimization and imprecise edits when applied through large-scale fine-tuning over full diffusion trajectories. To bridge this gap, we propose Concept Acupuncturist (ConceptACP), a novel MU framework that integrates interpretability-guided analysis with inference-free fine-tuning for efficient and precise concept removal. By establishing mappings between semantic concepts and cross-attention blocks in the U-Net denoiser and localizing the most concept-relevant block for targeted optimization, ConceptACP avoids full diffusion trajectory optimization, thereby improving f
    
[^132]: ACVA-Bench：超越提示词跟随——文本到图像模型的抽象概念可视化能力

    ACVA-Bench: Beyond Prompt Following in Abstract Concept Visualization Ability for Text-to-Image Models

    [https://arxiv.org/abs/2610.32058](https://arxiv.org/abs/2610.32058)

    该论文定义了文本到图像模型的抽象概念可视化能力（ACVA），并基于心理语言学规范提出类别平衡的ACVA-Bench基准，在低干预设置下从语义充分性、表征多样性和概念区分度三个维度系统评估模型自主可视化抽象概念的能力。

    

    文本到图像（T2I）模型在图像质量和指令遵循方面已取得显著进展，这引发了研究界对其超越字面提示匹配的深层理解能力的日益关注。然而，在未提供明确视觉方案的情况下，T2I模型能否自主地将缺乏固定视觉指涉物的抽象概念（如“忠诚”）转化为准确、多样且与密切相关概念可区分的视觉表征，这一问题仍不清晰且尚未得到系统性评估。我们将这种能力定义为抽象概念可视化能力，并引入ACVA-Bench——一个基于抽象概念心理语言学规范的类别平衡基准，该基准在低干预设置下（即不给出具体物体、场景等提示），从语义充分性、表征多样性和概念区分度三个维度考察模型的ACVA。

    arXiv:2610.32058v1 Announce Type: cross  Abstract: Text-to-image (T2I) models have achieved remarkable progress in image quality and instruction following, prompting growing research interest in their capacity for deep understanding that goes beyond literal prompt matching. However, without an explicit visual scheme provided, it remains unclear and has not been systematically evaluated whether T2I models can autonomously translate abstract concepts lacking a fixed visual referent, such as "loyalty," into visual representations that are accurate, diverse, and distinguishable from closely related concepts. We define this capability as Abstract Concept Visualization Ability (ACVA) and introduce ACVA-Bench, a category-balanced benchmark grounded in psycholinguistic norms of abstract concepts, which examines models' ACVA along three dimensions semantic sufficiency, representational diversity, and conceptual distinctiveness under a low-intervention setting that withholds concrete objects, sc
    
[^133]: HyScale：面向学习型稀疏注意力的长上下文训练扩展

    HyScale: Scaling Long-Context Training for Learned Sparse Attention

    [https://arxiv.org/abs/2610.32057](https://arxiv.org/abs/2610.32057)

    HyScale 通过无需分数表的实时精确 top-k 选择方法和前向选择重放机制，将学习型稀疏注意力的百万词元长上下文训练速度提升了 2.07 倍。

    

    我们提出了 HyScale，一个面向学习型稀疏注意力的长上下文训练系统。学习型稀疏注意力是一种适用于百万级词元上下文的有前景架构，自 DeepSeek-V3.2 以来已在生产环境中被广泛采用。然而，选择操作成为了训练瓶颈：对于每个查询，它需要对之前所有词元打分，而稀疏注意力只读取其中的 $k$ 个；更糟的是，当前系统会生成完整的分数表，并在重计算期间重复选择操作。我们的关键洞察是：选择的输出不是激活值，而是一组索引。基于这一洞察，HyScale 引入了 HyTopK 和选择重放机制。HyTopK 无需分数表即可实时找到每个查询的精确 top-$k$，而选择重放则在重计算中复用前向传播的选择结果。结合应对端到端训练挑战的整体系统优化，HyScale 在 512 块 H20 GPU 上将 GLM-5.1 在 1M 词元序列上的训练速度相比 Megatron-LM 提升了 2.07 倍。

    arXiv:2610.32057v1 Announce Type: new  Abstract: We present HyScale, a long-context training system for learned sparse attention. Learned sparse attention is a promising architecture for million-token contexts, widely adopted in production since DeepSeek-V3.2. Selection, however, grows into the training bottleneck. For each query, it scores every previous token whereas sparse attention reads only $k$; worse, current systems materialize a full score table and repeat selection during recomputation. Our key insight is that the output of selection is not an activation, but rather a set of indices. Building on it, HyScale introduces HyTopK and selection replay. HyTopK finds each query's exact top-$k$ on the fly without a score table, while selection replay reuses forward-pass selection in recomputation. Together with holistic system optimizations tackling end-to-end training challenges, HyScale speeds up GLM-5.1 training on 1M-token sequences by 2.07$\times$ over Megatron-LM on 512 H20 GPUs
    
[^134]: 测试时记忆演化能否在科学发现中超越测试时训练？

    Can Test-Time Memory Evolution Outperform Test-Time Training for Scientific Discovery?

    [https://arxiv.org/abs/2610.32046](https://arxiv.org/abs/2610.32046)

    提出测试时记忆演化（TTME），一种无需更新模型权重的即插即用测试时适应方法，通过持续演化外部记忆来积累经验，在科学发现任务上以更高的计算效率超越了测试时训练（TTT）。

    

    近期的测试时训练（TTT）框架已使开放权重的大型语言模型（LLM）在科学发现任务上超越了由前沿模型驱动的仅推理框架。然而，TTT 在发现过程中需要在线更新模型权重，带来了额外的计算成本，使得在扩展到更大规模模型时计算效率成为一个重要的考量因素。这促使我们探索一种高效的面向科学发现的测试时适应方法。我们提出了测试时记忆演化（TTME），这是一种即插即用的测试时适应方法，作为 TTT 的替代方案。TTME 通过将经验蒸馏到外部记忆中并持续更新，以指导后续的发现过程，而无需更新模型权重。为了进行干净的对比，我们在相同的发现框架下评估 TTME 和 TTT，仅替换其中的测试时适应方法。我们的评估表明，TTME 的表现优于 TTT。

    arXiv:2610.32046v1 Announce Type: new  Abstract: Recent \emph{test-time training} (TTT) frameworks have enabled open-weight large language models (LLMs) to outperform inference-only frameworks powered by frontier models on scientific discovery tasks. However, TTT introduces additional computational cost for online weight updates during the discovery process, making computational efficiency an important consideration when scaling to larger models. This motivates us to explore an efficient \emph{test-time adaptation} method for scientific discovery. We introduce \emph{Test-Time Memory Evolution} (TTME), a plug-and-play test-time adaptation method, as an alternative to TTT. TTME continually updates the external memory by distilling the experience to guide subsequent discovery without updating model weights. To have a clean comparison, we evaluate TTME and TTT under the same discovery framework, replacing only the test-time adaptation method. Our evaluation shows that TTME outperforms TTT 
    
[^135]: 面向科学的AI智能体能否保持模型安全性？

    Do AI-for-Science Agents Preserve Model Safety?

    [https://arxiv.org/abs/2610.32045](https://arxiv.org/abs/2610.32045)

    该研究首次系统评估了AI-for-Science智能体框架对模型安全性的影响，发现将LLM集成到科学智能体框架中会显著削弱安全性，即使在直接提示下也会增加有害内容的依从性，且对基础模型无效的自适应攻击在智能体框架中变得有效。

    

    面向科学的AI智能体正日益被用于构建科学假设、运行自主实验和解释证据。这些智能体是通过将大语言模型（LLM）集成到特定领域的智能体框架中而构建的，这反过来可以提升LLM的科学能力，并使模型能够协调使用科学工具。在这项工作中，我们探讨当LLM被集成到AI-for-Science智能体框架中时，模型安全性是否得以保留。我们在50个有害化学请求上评估了五个智能体框架。具体而言，我们研究了在直接提示、两种简单的对抗性提示策略以及一种自适应攻击下，模型的安全行为如何变化。在四个近期GPT变体上的实验结果表明，即使在直接提示下，智能体框架也可能增加有害请求的依从性。更关键的是，我们发现那些对基础LLM基本无效的自适应攻击，突然变得……（摘要原文在此处被截断）

    arXiv:2610.32045v1 Announce Type: new  Abstract: AI-for-Science agents are increasingly being used for formulating scientific hypotheses, running autonomous experiments, and interpreting evidence. These agents are constructed by integrating a large language model (LLM) into a domain-specific agent harness, which in turn can improve the LLM's scientific capability and enable the model to orchestrate scientific tools. In this work, we ask whether model safety is preserved when an LLM is integrated in an AI-for-Science agent harness. We evaluate five agent harnesses on 50 harmful chemistry requests. Specifically, we study how the safety behavior changes under direct prompting, two simple adversarial prompting strategies, and an adaptive attack. Across four recent GPT variants, our results show that the agent harness can increase harmful compliance even under direct prompting. More critically, we find that adaptive attacks that are largely unsuccessful against the base LLM suddenly become 
    
[^136]: 重新思考长尾数据下监督对比学习中的表示几何

    Rethinking Representation Geometry in Supervised Contrastive Learning under Long-Tailed Data

    [https://arxiv.org/abs/2610.32039](https://arxiv.org/abs/2610.32039)

    本文揭示了现有长尾数据下的监督对比学习目标因依赖每类样本数量而无法在最优解处形成正则单纯形几何，并提出结构平衡方法与完全平衡的对比目标，可证明地实现正则单纯形几何。

    

    监督对比学习及其平衡变体已在长尾分布下的表示学习中展现了其有效性。然而，尽管先前的分析表明其最优表示可能呈现正则单纯形几何结构，但这种几何结构是否确实存在仍然缺乏充分的理解。在这项工作中，我们提出了一个理论框架，用于刻画这些目标函数在最优解处是否形成正则单纯形。通过将该框架应用于代表性的目标函数，我们表明这些目标函数在最优解处无法形成正则单纯形，因为它们仍然依赖于每个类别的样本数量。受这一分析的启发，我们引入了可应用于对比目标的结构平衡方法，并提出了一种完全平衡的对比目标，该目标可证明地在其最优解处形成正则单纯形几何结构。理论和实证结果均表……（摘要在此处截断）

    arXiv:2610.32039v1 Announce Type: cross  Abstract: Supervised contrastive learning and its balanced variants have demonstrated their effectiveness for representation learning under long-tailed distributions. However, although prior analyses suggest that their optimal representations may exhibit a regular simplex geometry, whether such a geometry is indeed exhibited remains insufficiently understood. In this work, we present a theoretical framework for these objectives that characterizes whether a regular simplex is formed at the optimum. By applying this to representative objectives, we show that these objectives fail to form a regular simplex at the optimum because they remain dependent on the number of samples per class. Motivated by this analysis, we introduce structural balancing that can be applied to contrastive objectives, and propose a fully balanced contrastive objective that provably forms a regular simplex geometry at its optimum. Both theoretical and empirical results demon
    
[^137]: PPO与SAC：双臂空间机器人协同运动控制的对比研究

    PPO vs. SAC: A Comparative Study for Cooperative Motion Control of Dual-Arm Space Robots

    [https://arxiv.org/abs/2610.32034](https://arxiv.org/abs/2610.32034)

    本文提出了一个针对双臂空间机械臂协同运动规划的强化学习框架，通过集中式状态表示和多目标奖励函数，在具有臂间碰撞避免约束的目标到达任务中对比研究了PPO和SAC两种算法的性能。

    

    轨道碎片数量的不断增长加剧了人们对主动碎片清除和在轨服务技术的关注，而双臂空间机械臂相比单臂系统在抓取稳定性和载荷分配方面具有明显优势。然而，在臂间碰撞避免、姿态对准以及传感不确定性等条件下协调两个机械臂仍然极具挑战性。本文针对具有臂间碰撞避免约束的目标到达任务，开发了一个用于双臂空间机械臂协同运动规划的强化学习框架。该框架采用集中式状态表示，结合了关节空间、末端执行器以及目标相对信息，同时设计了多目标奖励函数，以促进目标到达精度、姿态对准、运动平滑性、任务同步完成和碰撞避免。在此框架内，对近端策略优化（PPO）和软演员-评论家（SAC）两种算法……

    arXiv:2610.32034v1 Announce Type: new  Abstract: The growing population of orbital debris has intensified interest in active debris removal and on-orbit servicing, for which dual-arm space manipulators offer improved grasping stability and load distribution relative to single-arm systems. However, coordinating two manipulators under inter-arm collision avoidance, orientation alignment, and sensing uncertainty remains challenging. This paper develops a coordinated reinforcement learning framework for cooperative motion planning of a dual-arm space manipulator in a target-reaching task with inter-arm collision-avoidance constraints. A centralized state representation combines joint-space, end-effector, and target-relative information, while a multi-objective reward function promotes target-reaching accuracy, orientation alignment, motion smoothness, synchronized task completion, and collision avoidance. Within this framework, Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC)
    
[^138]: RSI-Opt：用于模型训练的递归自我改进优化器智能体

    RSI-Opt: Recursive Self-Improvement Optimizer Agents For Model Training

    [https://arxiv.org/abs/2610.32026](https://arxiv.org/abs/2610.32026)

    RSI-Opt提出了一个即插即用的递归自我改进优化器智能体框架，元智能体利用历史运行证据持续演化指导策略（提示词、操作手册、代码库和决策节奏），显著提升了LLM智能体自动调节模型训练优化器的性能。

    

    传统的神经网络训练通常遵循预定义的优化方案，这促使人们开发能够监控训练动态并随训练进程调整优化器设置的大语言模型（LLM）智能体，但固定的智能体策略无法在多次运行之间积累经验。我们提出了RSI-Opt，一个即插即用的递归自我改进优化器智能体，它能够演化指导这一过程的知识：包括提示词、程序操作手册、经过测试的优化器代码库以及决策节奏。一个元智能体利用先前运行的证据重写该策略，并通过存档保留已评估的策略以供进一步演化。在三个训练任务上经过5代演化，元智能体开发出了可复用的程序和代码，并开展了自己的CPU代理实验。RSI-Opt显著提升了优化器智能体的性能，在演化套件上具有代表性的原始空间汇总z分数从+10.48提升至+14.33。

    arXiv:2610.32026v1 Announce Type: new  Abstract: Conventional neural-network training typically follows a predefined optimization recipe, motivating a LLM agent that monitors training dynamics and adjusts optimizer settings as training progresses, but a fixed agent policy does not accumulate experience across runs. We introduce RSI-Opt, a plug-and-play Recursive Self-Improvement Optimizer Agents which evolves the knowledge that guides this process: a prompt, a playbook of procedures, a library of tested optimizer code, and a decision cadence. A meta-agent rewrites this policy using evidence from earlier runs, and an archive retains evaluated policies for further evolution. Across 5 generations on three training tasks, the meta-agents develop reusable procedures and code and conduct their own CPU proxy experiments. RSI-Opt substantially improves optimizer-agent performance, with representative raw-space summary $z$-scores increasing from $+10.48$ to $+14.33$ on the evolution suite with 
    
[^139]: BARC-Bench：善意之问，风险之果

    BARC-Bench: Benign Asks, Risky Consequences

    [https://arxiv.org/abs/2610.32024](https://arxiv.org/abs/2610.32024)

    该论文提出 BARC-Bench 基准，通过 110 个模拟工作区任务揭示前沿智能体模型在面对隐含安全边界的良性请求时并非始终安全，即使最强配置仍有 27.3% 的任务出现安全违规。

    

    用户给智能体下达的任务指令天然存在信息不完整的问题，尤其是在安全边界方面——这带来了行为风险。为了研究这些风险，我们提出了 BARC-Bench，这是一个包含 110 个任务、横跨九个模拟工作区环境的基准，在这些环境中，看似无害的请求可能为不安全行为留下可乘之机。在该基准中，我们依据工作区记录、隐含的环境语境以及未明说的社会规范所确立的安全边界来评估智能体的行为。例如，被要求安排离职面谈的智能体必须在不通过共享日历邀请过早泄露员工离职消息的前提下安排会议。我们对四个前沿模型配置进行评估，每个任务测试五次，结果显示模型并非始终安全：五次尝试全部安全完成的任务比例从 9.1% 到 72.7% 不等，即使是表现最强的配置，也有 27.3% 的任务出现了安全违规。

    arXiv:2610.32024v1 Announce Type: new  Abstract: Task instructions from users for agents are inherently underspecified, especially regarding safety boundaries--creating behavioral risks. To study these risks, we introduce BARC-Bench, a collection of 110 tasks across nine simulated workspace environments where benign requests leave openings for unsafe actions. In this benchmark, we evaluate agent behavior against safety boundaries established by workspace records, implicit environmental context, and unstated social norms. For example, an agent asked to schedule off-boarding discussions must arrange the meeting without prematurely revealing an employee's resignation through a shared calendar invitation. Evaluating four frontier model configurations across five attempts per task reveals that models are not consistently safe: the share of tasks completed safely on all five attempts ranges from 9.1% to 72.7%, with even the strongest configuration producing a safety violation on 27.3% of tas
    
[^140]: CEMIR 对比 FISER：基于推理与基于增强的语用意图识别方法的比较研究

    CEMIR vs. FISER: a comparative study of reasoning-based and enrichment-based approaches to pragmatic intent recognition

    [https://arxiv.org/abs/2610.32022](https://arxiv.org/abs/2610.32022)

    本文通过一项跨四个大语言模型的比较性与诊断性研究，对比了基于结构化社会与具身推理的 FISER 和基于多模态叙事与逼真图像增强的 CEMIR，以探究当指令模糊性源于未明说的人类意图而非缺失场景信息时，多模态增强方法是否仍具优势。

    

    arXiv:2610.32022v1 公告类型：新论文 摘要：人机交互中的意图识别要求智能体通过推断说话者的潜在目标来解析含糊的自然语言指令。研究日益聚焦于解决该任务的两种重要范式：结构化符号推理与多模态输入增强。然而，当模糊性涉及的是未明说的人类意图而非缺失的可观察场景信息时，基于增强的方法能否带来收益仍不清楚。在本文中，我们提出了一项比较性与诊断性研究，将 FISER——一个利用结构化社会推理与具身推理的框架——与 CEMIR（上下文增强多模态意图识别）进行基准对比，后者通过从文本提示合成叙述与逼真图像来执行多模态增强。我们在四个大语言模型（从 Qwen2.5-Omni-3B 到 GPT-5）上对两种方法进行了评估（摘要在此处截断）。

    arXiv:2610.32022v1 Announce Type: new  Abstract: Intent recognition in human-robot interaction requires agents to resolve ambiguous natural language instructions by inferring the speaker's underlying goals. Research has increasingly focused on two notable paradigms for addressing this task: structured symbolic reasoning and multimodal input enrichment. However, it remains unclear whether enrichment-based methods provide benefit in settings where the ambiguity concerns unstated human intent rather than missing observable scene information. In this paper, we present a comparative and diagnostic study benchmarking FISER - a framework utilizing structured social and embodied reasoning - against CEMIR (Context-Enriched Multimodal Intent Recognition), a method that performs multimodal enrichment by synthesizing narratives and photorealistic images from textual prompts. By evaluating both approaches across four Large Language Models (LLMs), ranging from Qwen2.5-Omni-3B to GPT-5, on the HandMe
    
[^141]: MI-MIDI 2：在文本到MIDI生成中利用稀疏自编码器实现从观测到控制

    MI-MIDI 2: From Observation to Control with Sparse Autoencoders in Text-to-MIDI Generation

    [https://arxiv.org/abs/2610.32007](https://arxiv.org/abs/2610.32007)

    本文提出了一个端到端协议，首次在文本到MIDI生成中系统验证了SAE特征解释对实际输出控制的预测能力，实现了从观测模型特征到干预控制生成结果的跨越。

    

    稀疏自编码器（SAE）能够在无需预定义标签的情况下揭示生成模型的内部特征，但在语言模型中，用于解释这些特征的证据已被证明对于判断哪些特征能够引导模型输出并不可靠。据我们所知，SAE特征的解释能否预测其在音乐生成器中的控制效果，尚未经过检验。我们开发了一个端到端的协议，用于在文本到MIDI生成中发现和评估SAE特征，并将其应用于text2midi和MIDI-LLM两个模型。该协议通过特征与音乐指标、共享乐器音色程序以及自动生成描述之间的相关性来刻画特征，每种刻画方式均有各自的对照方法：音符密度调整、匹配的非激活片段以及置换校准。随后，协议通过干预实验测试候选特征，直接在生成的MIDI中与未修改的生成结果及匹配的随机方向进行对比测量，在筛选环节中对候选特征进行排序，并在新提示上对选定的特征进行确认。

    arXiv:2610.32007v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) can expose features of a generative model without predefined labels, but in language models the evidence used to interpret these features has proven an unreliable guide to which of them steer the output. Whether interpretations of SAE features predict control in music generators has not, to our knowledge, been tested. We develop an end-to-end protocol for discovering and evaluating SAE features in text-to-MIDI generation and apply it to text2midi and MIDI-LLM. The protocol characterizes features through correlations with musical metrics, shared instrument programs, and automated descriptions, each with its own control: note-density adjustment, matched inactive excerpts, and permutation calibration. It then tests candidates by intervention, measured directly in the generated MIDI against unmodified generation and a matched random direction, ranks them in screens, and confirms selected features on new prompts. 
    
[^142]: OpenHome：从感知到洞察的智能家居摄像头视频

    OpenHome: From Perception to Insight in Smart-Home Camera Videos

    [https://arxiv.org/abs/2610.31986](https://arxiv.org/abs/2610.31986)

    该论文提出了OpenHome，一个包含2,254个问题、覆盖1,118个真实智能家居视频片段的纯视觉多选题基准测试，用于在稀疏视觉信号和有限上下文条件下系统评估视觉语言模型的智能家居视频理解能力。

    

    视觉语言模型（VLMs）在消费级智能家居摄像头系统中被越来越多地采用，用于解读录制的视频并提供洞察。然而，从智能家居摄像头捕获的视频片段与现有VLM基准测试中占主导地位的精选网络视频有所不同。智能家居视频片段通常不包含字幕或旁白，并且是在不同的视角、各种光照条件以及多样化的场景下捕获的。然而，模型在该领域在稀疏视觉信号和有限上下文情况下的能力，尚未被现有的VLM基准测试充分评估。为了弥合这一差距，我们提出了OpenHome，这是一个用于评估智能家居视频理解的纯视觉四选一多选题基准测试。它包含针对1,118个真实世界视频片段的2,254个问题，并将10个任务组织为三个能力层级。感知层级评估可见内容，推理层级评估因果、空间和时间……

    arXiv:2610.31986v1 Announce Type: cross  Abstract: Vision language models (VLMs) have been increasingly adopted in consumer smart-home camera systems to interpret recorded videos and provide insights. However, video clips captured from smart-home cameras are different from curated web videos dominating existing VLM benchmarks. Smart-home video clips often do not contain subtitles or narration and are captured from different viewing angles, under various lighting conditions, and in diverse scenarios. However, the capabilities of the model in this domain, under sparse visual signals and limited context, are not fully assessed by existing VLM benchmarks. To close this gap, we introduce OpenHome, a visual-only, four-option multiple-choice benchmark for evaluating smart-home video understanding. It contains 2,254 questions on 1,118 real-world video clips and organizes 10 tasks into three capability levels. Perception evaluates visible content, Reasoning evaluates causal, spatial, and tempor
    
[^143]: 有益智能体问题：当LLM智能体间的涌现合作绕过预期工作流时

    The Helpful Agent Problem: When Emergent Cooperation Among LLM Agents Bypasses Intended Workflows

    [https://arxiv.org/abs/2610.31985](https://arxiv.org/abs/2610.31985)

    该论文揭示了“有益智能体问题”：即使没有恶意或共谋，追求各自目标的良性LLM智能体也会在审查机制失效时自发合作、代替彼此执行操作，从而在合法权限范围内绕过预期的监督工作流，构成一种新型AI安全风险。

    

    LLM智能体越来越多地被部署在协作环境中：它们在追求各自任务的同时共享资源，并处于审查或监督流程之下。这类环境不仅在智能体具有恶意时会产生安全风险，在它们仅仅是想提供帮助时同样会产生安全风险。以往的研究主要关注对抗性或明显的共谋行为；我们则表明，追求各自目标的良性智能体可以自发地进行合作，从而绕过预期的工作流。我们将这种现象称为“有益智能体问题”。我们研究了长时程的共享资源环境：其中智能体有时会依赖同伴所拥有的资源，而跨所有者的更改通常由审查者来调解。当审查机制变得不可用时，智能体便开始代替彼此进行更改，尽管它们之间没有共享奖励，也没有明确的合作激励。至关重要的是，这些操作仍然处于每个智能体自身的本地权限范围内，同时却规避了预期的监督流程。

    arXiv:2610.31985v1 Announce Type: new  Abstract: LLM agents are increasingly deployed in collaborative settings where they pursue separate tasks while sharing resources and operating under review or oversight processes. Such settings create safety risks not only when agents are malicious, but also when they are simply trying to help. Prior work has largely studied adversarial or explicitly collusive behavior; we show that benign agents pursuing separate objectives can spontaneously cooperate in ways that bypass intended workflows. We call this the \emph{helpful agent problem}. We study long-horizon shared-resource environments in which agents sometimes depend on resources owned by their peers and cross-owner changes are normally mediated by a reviewer. When review becomes unavailable, agents begin making changes on one another's behalf despite having no shared reward or explicit incentive to cooperate. Crucially, these actions remain within each agent's local permissions while circumve
    
[^144]: Spectra：面向深度伪造检测的多尺度隐式分解方法

    Spectra: Multi-Scale Implicit Decomposition for Deepfake Detection

    [https://arxiv.org/abs/2610.31979](https://arxiv.org/abs/2610.31979)

    Spectra通过将隐式神经表示拟合到图像并分析其多尺度内部激活，揭示图像的重构动态，从而捕捉真实与合成内容之间更本质的差异，克服了传统基于外观或频域线索的方法在泛化性上的局限。

    

    深度伪造日益破坏人们对数字媒体的信任，然而现有检测器往往难以在未见过的伪造技术、压缩程度和数据集偏移上实现泛化。大多数方法依赖于RGB外观、频域伪影或二者的结合，学习的是最终合成图像中存在的线索。这类线索虽然在分布内有效，但通常是针对特定生成器的，并会随着合成模型的演进而逐渐失效。我们认为，隐式神经表示（INRs）可以提供一种互补的视角，它揭示的是图像如何被重构，而不仅仅是图像的外观。通过将INR拟合到图像并分析其内部激活，可以揭示仅凭图像域难以观察到的重构动态，并可能捕捉到真实内容与合成内容之间更本质的差异。为此，我们提出了Spectra，一个深度伪造检测框架（原文摘要至此处截断）。

    arXiv:2610.31979v1 Announce Type: cross  Abstract: Deepfakes increasingly undermine trust in digital media, yet existing detectors often struggle to generalize across unseen manipulation techniques, compression levels, and dataset shifts. Most approaches rely on RGB appearance, frequency-domain artifacts, or their combination, learning cues present in the final synthesized image. While effective in-distribution, such cues are frequently generator-specific and can diminish as synthesis models evolve. We argue that a complementary perspective can be obtained through implicit neural representations (INRs), which expose how an image is reconstructed rather than merely how it appears. By fitting an INR to an image and analyzing its internal activations, one can reveal reconstruction dynamics that are difficult to observe in the image domain alone and may capture more intrinsic differences between genuine and synthetic content. To this end, we propose Spectra, a deepfake detection framework 
    
[^145]: 智能体AI防御：面向MCP的解码时溯源绑定

    Agentic AI Defense: Decode-Time Provenance Binding for MCP

    [https://arxiv.org/abs/2610.31974](https://arxiv.org/abs/2610.31974)

    该论文提出解码时溯源绑定机制，通过HMAC认证的信任标签、MCP维护的溯源账本和上下文污点跟踪三者的结合，有效防御针对MCP智能体系统的跨模态隐写式间接提示注入攻击。

    

    使用模型上下文协议的智能体AI系统容易通过不可信的工具输出遭受间接提示注入攻击。我们演示了跨模态信任洗白攻击，即利用最低有效位（LSB）、离散余弦变换（DCT）或元数据隐写术将指令隐藏在电子邮件签名图像中，绕过文本扫描器，并在工具解码该图像时进入多智能体工作流。为应对这一威胁，我们提出解码时溯源绑定机制，将基于HMAC认证的信任标签与MCP维护的溯源账本以及上下文污点跟踪相结合。该机制将解码内容标记为不可信，并阻止受该内容影响的关键工具调用，即使被入侵的智能体替换了有效的可信标签也能实现防护。我们在真实的MCP测试平台上，使用三种隐写载体和24个自适应载荷对该方法进行了评估。未加防御的流水线攻击成功率为100%，而基于关键词的方法……（原文摘要在此处截断）

    arXiv:2610.31974v1 Announce Type: cross  Abstract: Agentic AI systems using the Model Context Protocol (MCP) are vulnerable to indirect prompt injection through untrusted tool outputs. We demonstrate cross-modal trust laundering, in which instructions concealed in an email-signature image using least-significant-bit (LSB), discrete cosine transform (DCT), or metadata steganography bypass text scanners and enter a multi-agent workflow when decoded by a tool. To counter this threat, we propose decode-time provenance binding, which combines HMAC-authenticated trust labels with an MCP-maintained provenance ledger and context-taint tracking. The mechanism labels decoded content as untrusted and blocks critical tool calls influenced by that content, even when a compromised agent substitutes a valid trusted label. We evaluate the approach using a real MCP testbed with three steganographic carriers and 24 adaptive payloads. The undefended pipeline exhibits 100% attack success, while keyword an
    
[^146]: 可复现评估作为生产智能体中安全提示压缩的基础

    Reproducible Evaluation as the Substrate for Safe Prompt Compression in Production Agents

    [https://arxiv.org/abs/2610.31972](https://arxiv.org/abs/2610.31972)

    该论文提出了一个能在实时数据变化下保持可复现并可按需重新生成的离线评估框架，并将其作为带正确性门控的GEPA提示优化的适应度函数，成功将生产智能体的提示规模压缩3.5-7.6%而不损害任务正确性。

    

    评估作用于实时、可变企业数据的对话式智能体十分困难：当后端状态发生变化时，重放的会话会产生漂移，而今天构建的基准测试会随着其所描述数据的不断演变而悄然失效。我们提出了一个面向生产环境对话式智能体的离线评估工具，尽管其所操作的实时数据在不断变化，该工具仍能保持可复现性，并且在基准失效后可以按需重新生成。随后，我们将该工具用作基于GEPA的提示优化的适应度函数，并通过正确性门控机制防范奖励作弊。将该应用于一个面向商家的生产环境智能体后，带正确性门控的压缩使智能体的完整提示词规模缩减了3.5-7.6%，且在留出任务上的正确性没有下降，这一结果在三个独立的受试模型家族中保持一致，其中之一的正确性还有可测量的提升。

    arXiv:2610.31972v1 Announce Type: new  Abstract: Evaluating conversational agents that act on live, mutable enterprise data is hard: replayed sessions drift as backend state changes, and a benchmark built today can silently go stale as the data it describes moves on. We present an offline evaluation harness for a production conversational agent that stays reproducible despite the live data it operates on changing, and that can be regenerated on demand once it goes stale. We then use this harness as the fitness function for GEPA-based prompt optimization under a correctness gate that guards against reward hacking. Applied to a production merchant-facing agent, correctness-gated compression shrinks the agent's full prompt surface by 3.5-7.6% without regressing held-out task correctness, consistently across three independent subject-model families, and for one, correctness measurably improves.
    
[^147]: 知道该信任谁还不够：重复多方LLM谈判中的信任-协议鸿沟

    Knowing Whom to Trust is not Enough: The Trust--Agreement Gap in Repeated Multi-Party LLM Negotiation

    [https://arxiv.org/abs/2610.31971](https://arxiv.org/abs/2610.31971)

    本文提出了一个重复多方、多议题的LLM谈判框架，发现LLM智能体虽然能从历史交互中形成有用的信任估计并识别暗中阻挠协议的对手，但知道该信任谁并不足以保证达成协议，从而揭示了“信任-协议鸿沟”。

    

    大型语言模型（LLM）智能体越来越多地代表个人和组织参与谈判，但它们仍然容易受到操纵，尤其是在重复的多方互动中。经典的多智能体系统通过显式的信任与声誉模型来应对这一挑战，而现有的LLM谈判研究大多将信任视为一种涌现现象。我们研究了LLM智能体能否从先前的交互中形成有用的信任估计，并利用它们来改善未来的谈判结果。我们引入了一个重复的、多方的、多议题的谈判框架，其中反复出现的智能体参与一系列谈判博弈，同时一名参与者暗中试图阻止达成协议。每次博弈结束后，智能体会评估其对手，并维护延续至后续谈判的成对信任估计。我们比较了信任感知与信任无关的智能体，并分析了……（原文摘要在此处被截断）

    arXiv:2610.31971v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly participate in negotiations on behalf of individuals and organizations, yet they remain vulnerable to manipulation, particularly in repeated multi-party interactions. Classical multi-agent systems address this challenge through explicit trust and reputation models, whereas existing work on LLM negotiation largely treats trust as an emergent phenomenon. We investigate whether LLM agents can form useful trust estimates from prior interactions and leverage them to improve future negotiation outcomes. We introduce a repeated, multi-party, multi-issue negotiation framework in which recurring agents participate in a sequence of negotiation games while one participant covertly attempts to prevent agreement. After each game, agents assess their counterparts and maintain pairwise trust estimates that carry over to subsequent negotiations. We compare trust-aware and trust-agnostic agents and analyze h
    
[^148]: 雅可比透镜的数学理论

    A Mathematical Theory of the Jacobian Lens

    [https://arxiv.org/abs/2610.31955](https://arxiv.org/abs/2610.31955)

    本文提出了雅可比透镜（J-lens）的数学理论，证明该可解释性工具优先识别的是相干特征——即类似主导 Lyapunov 模态、可跨层放大、跨词元位置恒定且跨上下文一致的线性方向。

    

    Anthropic 最近的工作引入了一种新的可解释性工具，称为雅可比透镜（Jacobian lens，J-lens，Gurnee 等人，2026）。该方法在前沿语言模型的隐藏状态中识别出一些线性方向，其性质令人联想到工作记忆式的表征。然而，目前尚不清楚这些特征的哪些机制性质使得它们能够被 J-lens 检测到。为此，我们提出了 J-lens 的数学理论，并证明它优先识别的是“相干特征”——即表现得像主导的、空间相干的 Lyapunov 模态那样的方向。沿这些方向的小扰动会跨层放大，在不同词元位置之间大致保持恒定，并在不同上下文中保持一致。尽管我们对这些性质是否必然与“全局工作空间”相关联持不可知态度，但通过解析推导出的 J-lens 偏好仍然解释了先前应用于……的实证结果。

    arXiv:2610.31955v1 Announce Type: cross  Abstract: Recent work from Anthropic introduced a new interpretability tool, called the Jacobian lens (J-lens, Gurnee et al., 2026). This method identified linear directions in the hidden states of frontier language models that had properties reminiscent of working memory-like representations. However, it is unclear what mechanistic properties of such features make them detectable to the J-lens. To this end, we introduce a mathematical theory of the J-lens, and show that it preferentially identifies coherent features -- directions that act like leading, spatially coherent Lyapunov modes. Small perturbations along these directions amplify across layers, remain roughly constant across token positions, and are consistent across different contexts. While we are agnostic to whether these properties are necessarily associated with a ``Global Workspace'', the analytically derived preferences of the J-lens nevertheless explain prior empirical applied to
    
[^149]: RobustNCL：基于轻量级检测与修复策略增强瞬态故障下神经形态持续学习的鲁棒性

    RobustNCL: Robustifying Neuromorphic Continual Learning under Transient Faults with Lightweight Detect-and-Cure Strategy

    [https://arxiv.org/abs/2610.31954](https://arxiv.org/abs/2610.31954)

    本文提出RobustNCL框架，通过轻量级检测与修复策略提升神经形态持续学习在硬件瞬态故障（特别是DRAM片外存储器故障）下的可靠性。

    

    在神经网络中，持续学习（CL）是一种重要的方法，通过在运行时有效地从新的未见数据中学习，使已部署的系统能够适应动态的运行环境。最近，神经形态持续学习（NCL）概念通过利用稀疏的基于事件的运算，成为了一种节能的持续学习替代方案。然而，NCL的性能可能受到硬件平台中瞬态故障的损害，这些故障可能随时由高能粒子撞击引起。目前，NCL在瞬态故障下的可靠性尚未被研究，因此相应的容错技术仍未被探索。为解决这一问题，我们提出了一种新颖的RobustNCL框架，通过轻量级的检测与修复策略，提升硬件平台中基于DRAM的片外存储器存在瞬态故障时NCL的可靠性。具体而言，该框架首先监测权重更新过程中出现的故障……

    arXiv:2610.31954v1 Announce Type: cross  Abstract: In neural networks, continual learning (CL) is a prominent approach for enabling deployed systems to adapt to dynamic operational environments by effectively learning from new unseen data at runtime. Recently, the neuromorphic continual learning (NCL) concept has emerged as an energy-efficient alternative to CL by leveraging sparse event-based operations. However, NCL performance can be undermined by transient faults in the hardware platforms, which may come from high-energy particle strikes at any time. Currently, NCL reliability under transient faults has not been studied, and hence the corresponding fault-tolerance techniques remain unexplored. To address this, we propose a novel RobustNCL framework to improve NCL reliability in the presence of transient faults in the DRAM-based off-chip memory of hardware platforms through a lightweight detect-and-cure strategy. Specifically, it first monitors the presence of faults during weight d
    
[^150]: Kolibri：位于帕累托前沿的欧洲主权大模型

    Kolibri: A Sovereign European Model on the Pareto Frontier

    [https://arxiv.org/abs/2610.31945](https://arxiv.org/abs/2610.31945)

    Kolibri是一个总参数78.1B、激活参数仅3.46B的英德双语开源混合专家模型，通过完全自主掌控的数据与训练流程实现德语占比超20%（含2T自建德语token）的训练，在帕累托前沿上实现了面向主权化部署的高效性能。

    

    本报告介绍了Kolibri，一个英德双语的混合专家transformer模型，总参数量为78.1B，每个token激活3.46B参数，以Apache 2.0许可证开源发布。我们构建Kolibri的目的是实现主权化和专业化部署，特别关注德语以及公共管理、工业和航空航天等受监管领域。为支持这些应用场景，我们掌控了完整的模型开发流程，包括数据、架构、训练基础设施、后训练和评估。我们在24T token上对Kolibri进行训练，涵盖预训练、中期训练和长上下文扩展，其中德语按字节数计算占数据组合的20%以上。为达到这一比例，我们通过自建数据管道从网络整理并合成生成了超过2T的德语token，以补充德语数据集。后训练阶段将在精选的德语和英语数据上进行监督微调与……（摘要原文不完整）

    arXiv:2610.31945v1 Announce Type: cross  Abstract: This report introduces Kolibri, an English-German mixture-of-experts transformer with 78.1B total parameters and 3.46B active parameters per token, released as open weights under the Apache 2.0 licence. We built Kolibri for sovereign and specialised deployment, with a particular focus on German and on regulated domains such as public administration, industry, and aerospace. To support these settings, we control the full model-development process, including data, architecture, training infrastructure, post-training, and evaluation.   We train Kolibri on 24T tokens across pre-training, mid-training, and long-context extension, with German accounting for more than 20% of the data mix by bytes. To reach this share, we complement German datasets with more than 2T German tokens that we curate from the web with our own pipeline and generate synthetically. Post-training combines supervised fine-tuning on curated German and English data with re
    
[^151]: $M^3$：一个将多轮越狱形式化为马尔可夫决策过程的多LLM系统

    $M^3$: A Multi-LLM System for Multi-Turn Jailbreaking as a Markov Decision Process

    [https://arxiv.org/abs/2610.31937](https://arxiv.org/abs/2610.31937)

    该论文提出 $M^3$ 自动化红队测试框架，将多轮越狱形式化为有限视野马尔可夫决策过程，通过攻击者LLM与状态LLM的协作，在完全黑盒设置下显著超越现有方法，攻击成功率最高提升53%。

    

    现代大型语言模型（LLM）仍然容易受到精心编排的多轮交互攻击，这些交互会逐步绕过其安全机制。然而，许多现有的红队测试方法依赖于启发式方法或施加了较强的假设，限制了它们在完全黑盒设置中的适用性。我们通过引入 $M^3$ 来弥补这一空白，这是一个自动化红队测试框架，将多轮越狱形式化为有限视野的马尔可夫决策过程（MDP）。在 $M^3$ 中，攻击者 LLM 与目标 LLM 进行交互，同时在一个由状态 LLM 生成的紧凑内部状态上进行操作。在两个基准测试和六个目标模型上，$M^3$ 在攻击成功率（ASR）方面始终优于最先进的多轮红队测试方法，在 ASR@5 上实现了高达 53% 的提升。即使使用轻量级攻击者，与使用更高能力攻击者的基线相比，$M^3$ 在 ASR@1 上仍可获得高达 26% 的提升，在 ASR@5 上提升高达 34%。

    arXiv:2610.31937v1 Announce Type: new  Abstract: Modern Large Language Models (LLMs) remain vulnerable to carefully orchestrated multi-turn interactions that progressively bypass safety mechanisms. However, many existing red-teaming approaches rely on heuristics or impose strong assumptions that limit their applicability in fully black box settings. We address this gap by introducing $M^3$, an automated red-teaming framework that formalizes multi-turn jailbreaking as a finite-horizon Markov Decision Process (MDP). In $M^3$, an attacker LLM interacts with a target LLM while operating over a compact internal state produced by a state LLM. Across two benchmarks and six target models, $M^3$ consistently outperforms state-of-the-art multi-turn red-teaming methods in Attack Success Rates (ASR), achieving improvements of up to 53% in ASR@5. Even with lightweight attacker, $M^3$ attains gains up to 26% in ASR@1 and 34% in ASR@5, compared to baselines that use higher capacity attackers. Beyond 
    
[^152]: ENCL：通过局部学习规则与子网络冻结策略实现高效神经形态持续学习

    ENCL: Enabling Efficient Neuromorphic Continual Learning with Local Learning Rule and Sub-Network Freezing Strategy

    [https://arxiv.org/abs/2610.31927](https://arxiv.org/abs/2610.31927)

    本文提出ENCL方法，通过轻量级的时空局部学习规则和子网络冻结策略，取代传统神经形态持续学习中内存消耗大的经验回放和昂贵的全局学习规则，实现了高效的持续学习。

    

    arXiv:2610.31927v1 公告类型：cross 摘要：神经网络中的持续学习（CL）是一种很有前景的学习方法，用于在运行时根据离线训练阶段未曾学习过的新数据来更新已部署系统的知识。为了实现功耗/能量高效的持续学习处理，近年来提出了基于稀疏事件操作的神经形态持续学习（NCL）概念。然而，最先进的NCL方法通常采用内存消耗大的数据/经验回放，并伴随着昂贵的全局学习规则（即基于反向传播的代理梯度）。为解决这些局限性，我们提出了ENCL，这是一种新颖的方法，通过采用轻量级局部学习规则和子网络冻结策略来实现高效的NCL。ENCL利用时间和空间上的局部学习规则为新任务的训练执行高效的信用分配，从而减轻了全局学习规则带来的高昂成本。ENCL还……（原文摘要在此处截断）

    arXiv:2610.31927v1 Announce Type: cross  Abstract: Continual learning (CL) in neural networks is a promising learning approach for updating the knowledge of deployed systems based on new data encountered at run time, that have not been learned during the offline training phase. To enable their power/energy-efficient CL processing, neuromorphic continual learning (NCL) concept with sparse event-based operations have recently been introduced. However, state-of-the-art NCL methods typically employ the memory-hungry data/experience replay, coupled with costly global learning rule (i.e., backpropagation-based surrogate gradient). To address these limitations, we propose ENCL, a novel methodology that enables efficient NCL by employing a lightweight local learning rule and sub-network freezing strategy. ENCL leverages the temporally- and spatially-local learning rule to perform efficient credit assignment for training new tasks, alleviating the expensive cost of global learning rule. ENCL al
    
[^153]: 超越同策略与异策略之分：LLM智能体蒸馏应在何种层级上跟随学生？

    Beyond On-Policy vs. Off-Policy: At What Level Should LLM Agent Distillation Follow the Student?

    [https://arxiv.org/abs/2610.31918](https://arxiv.org/abs/2610.31918)

    该论文提出分层智能体蒸馏框架，将同策略性分解为交互历史与动作前缀两个层级并实现显式控制，发现“学生生成的历史+教师生成的动作前缀”的组合通常是最优的蒸馏方案。

    

    蒸馏交互式LLM智能体需要在教师生成的演示与学生生成的经验之间做出选择，这大致对应于异策略与同策略蒸馏。尽管两者均已被证明有效，但我们认为这种区分本质上是分层级的：引发交互历史的策略不必与生成当前动作token前缀的策略相同。我们提出了一个分层智能体蒸馏框架，将这两个层级的同策略性形式化，并实现对它们的配对方式、混合比例和训练日程的显式控制。我们通过自蒸馏来研究这一问题，其中冻结的教师可以获取学生无法获得的任务指导，并在三个模型和四个交互环境中开展了三阶段对照研究。我们发现，学生引发的交互历史与教师生成的动作前缀相配对，通常能够提供……（摘要原文在此处截断）

    arXiv:2610.31918v1 Announce Type: new  Abstract: Distilling interactive LLM agents requires choosing between teacher-generated demonstrations and student-generated experience, corresponding broadly to off-policy and on-policy distillation. While both have proven effective, we argue that this distinction is inherently hierarchical: the policy that induces an interaction history need not be the same as the policy that generates the token prefix of the current action. We introduce a hierarchical agent distillation framework that formalizes these two levels of on-policiness and enables explicit control over their pairing, mixture proportions, and training schedule. We study this problem through self-distillation, where a frozen teacher has access to task guidance unavailable to the student, and conduct a three-stage controlled study across three models and four interactive environments. We find that student-induced histories paired with teacher-generated action prefixes generally provide t
    
[^154]: 重试的许可：AI智能体工作流中的授权与恢复研究

    Permission to Retry: A Study of Authorization and Recovery in AI Agent Workflows

    [https://arxiv.org/abs/2610.31916](https://arxiv.org/abs/2610.31916)

    该论文将AI智能体的恢复决策与语言模型行为分离，提出一项在LangGraph和Temporal宿主上执行336条历史记录的有限一致性研究，验证了检查点状态、分发许可、外部效果证据和对账许可这四种独立义务的执行时可靠性。

    

    持久智能体可以在其权限发生变化、且动作的外部结果仍不确定的情况下恢复一项具有重大影响的动作。我们研究使用工具的AI智能体的执行时可靠性，将恢复决策与语言模型行为隔离开来。检查点状态、分发许可、外部效果证据以及对账许可是四种彼此不同的义务。我们提出了一项有限一致性研究，通过将执行历史记录与对恢复应用隐藏的提供方效果账本进行比对评分，来检查这些义务。一项前瞻性冻结的实验在原生LangGraph和Temporal宿主上执行了336条历史记录，覆盖14种调度、三种提供方契约和四种匹配的应用策略。全部168个配对宿主单元都得到了完全相同的检查结果。在每个宿主上，保留账本对账通过了全部42条历史记录：21个已验证的完成、15个拒绝，以及6个保留预定的未知结果。缓存重试（摘要在此处被截断）

    arXiv:2610.31916v1 Announce Type: new  Abstract: A durable agent can resume a consequential action while its authority changes and the action's external outcome remains uncertain. We study execution-time reliability of tool-using AI agents, isolating recovery decisions from language-model behavior. Checkpoint state, permission to dispatch, evidence of an external effect and permission to reconcile are distinct obligations. We present a finite conformance study that checks these obligations through histories scored against a provider effect ledger hidden from the recovering application. A prospectively frozen campaign executes 336 histories on native LangGraph and Temporal hosts, crossing 14 schedules, three provider contracts and four matched application policies. All 168 paired host cells have identical checked outcomes. Per host, retained-ledger reconciliation passes all 42 histories: 21 checked completions, 15 denials and six unknown outcomes with reservations retained. Cached retry
    
[^155]: 面向大模型的结构化稀疏学习：弥合模型可压缩性与高效推理之间的鸿沟

    Learning Structured Sparsity for Large Models: Bridging Model Compressibility and Efficient Inference

    [https://arxiv.org/abs/2610.31897](https://arxiv.org/abs/2610.31897)

    提出统一框架S-SALAAD，联合学习模型权重与结构化稀疏+低秩表示，使大模型既能弹性压缩以适配边缘设备的多种内存预算，又能实现真正加速的高效推理。

    

    缩放定律持续推动着模型规模的增长，而由此产生的计算与内存需求使得推理日益依赖云基础设施。然而，不稳定的网络连接、通信延迟以及隐私要求可能使这种依赖变得不切实际，从而促使人们转向在边缘设备上进行本地执行。本地部署要求模型能够适应有限的内存，并在可用的计算资源下高效运行。近期研究表明，学习“稀疏+低秩”表示能够将单个模型检查点弹性压缩至多个内存预算。然而，参数数量的减少并不一定转化为更快的推理速度。非结构化稀疏组件中的索引开销和不规则内存访问可能会抵消计算上的节省。为解决这一关键的部署障碍，我们提出了S-SALAAD，这是一个统一框架，可联合学习模型权重及其低（注：原摘要在此处被截断）

    arXiv:2610.31897v1 Announce Type: cross  Abstract: Scaling laws continue to reward increases in model scale, while the resulting computational and memory demands drive greater reliance on cloud infrastructure for inference. Yet unreliable network connectivity, communication delays, and privacy requirements can make such reliance impractical, motivating local execution on edge devices. Local deployment requires models to fit within limited memory and execute efficiently with the available computational resources. Recent work has shown that learning sparse-plus-low-rank representations enables elastic compression of a single checkpoint to multiple memory budgets. However, fewer parameters do not necessarily translate into faster inference. Indexing overhead and irregular memory access in unstructured sparse components can offset computational savings. To address this critical deployment barrier, we introduce S-SALAAD, a unified framework that jointly learns the model weights and their lo
    
[^156]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^157]: 课程学习的有效性取决于奖励稀疏性与竞争最优解

    Effectiveness of Curriculum Learning Depends on Reward Sparsity and Competing Optima

    [https://arxiv.org/abs/2610.31884](https://arxiv.org/abs/2610.31884)

    通过构建并分析策略学习的最小化模型，该研究证明了课程学习的有效性取决于奖励的稀疏程度：在稠密奖励下课程仅有边际收益，而在稀疏奖励下课程可能是学习得以发生的必要条件。

    

    尽管动物和人类在先训练较简单版本的任务时，往往能更快、更可靠地学会该任务，但课程学习在人工环境中的有效性却各不相同，且其在强化学习中的效果似乎显著大于监督学习。在这项工作中，我们构建并分析了一个策略学习的最小化模型，以更好地理解其中的原因。我们考虑一个开环、回合制的目标拦截任务，其参数——尤其是智能体必须距离目标多近才能获得有信息量的奖励信号——可以被设定为使智能体频繁获得有信息量的奖励（“稠密”奖励）或很少获得（“稀疏”奖励）。我们对两种情况下基于REINFORCE的策略学习动力学进行了数学和数值分析。在稠密奖励的情况下，课程只能带来边际收益；而在稀疏奖励的情况下，课程可能是学习得以发生的必要条件。我们发现，这一现象（原文摘要在此处被截断）

    arXiv:2610.31884v1 Announce Type: cross  Abstract: While animals and people tend to learn a task more quickly and reliably when they first train on simpler versions of it, curriculum learning's effectiveness in artificial settings varies, and appears substantially greater in reinforcement learning than in supervised learning. In this work, we construct and analyze a minimal model of policy learning to better understand why. We consider an open-loop, episodic target interception task whose parameters---especially how close the agent must be to the target to receive an informative reward signal---can be chosen so that the agent either receives informative rewards frequently ('dense' rewards) or rarely ('sparse' rewards). We mathematically and numerically analyze the dynamics of REINFORCE-based policy learning in both cases. In the dense reward case, curricula have only marginal benefits; in the sparse case, curricula can be required for learning to occur at all. We find that this is espe
    
[^158]: TACS-RF：工具驱动的射频与毫米波模块智能体电路综合

    TACS-RF: Tool-driven Agentic Circuit Synthesis of RF and mmWave Modules

    [https://arxiv.org/abs/2610.31882](https://arxiv.org/abs/2610.31882)

    该论文提出了TACS-RF框架，首次让LLM编程智能体直接驱动Cadence、Ansys等工业级EDA工具，半自主完成了90nm SiGe BiCMOS工艺下240-251GHz亚太赫兹发射机从振荡器、功率驱动、倍频器到天线的完整芯片设计，验证了LLM智能体驱动真实射频芯片设计流程的可行性。

    

    我们提出了TACS-RF框架——一种工具驱动的射频与毫米波模块智能体电路综合方法。在该框架中，大语言模型（LLM）编程智能体直接驱动业界标准的电子设计自动化（EDA）工具，如Cadence Virtuoso/Spectre、Calibre、Cadence EMX和Ansys HFSS，来设计射频与毫米波模块。我们通过在90nm SiGe BiCMOS工艺下设计一个240-251GHz亚太赫兹发射机来验证该框架。该智能体半自主地设计了各个子模块，包括一个带二次谐波提取的60GHz推推式Colpitts压控振荡器（VCO）、多级差分共源共栅功率驱动器、一个推推式倍频器、一个差分蝶形天线以及全部级间匹配网络。包含寄生参数提取器件和经电磁验证无源器件的发射机链，在11GHz的纯变容管调谐范围内提供+5.1至+6.3dBm的输出功率，1MHz频偏处相位噪声为-80至-88dBc/Hz（原文摘要在此处截断）。

    arXiv:2610.31882v1 Announce Type: new  Abstract: We present the framework of TACS-RF - a Tool-driven Agentic Circuit Synthesis of RF and mmWave modules. In this framework, a large-language-model (LLM) coding agent (Claude Opus 4.8) directly drives industry-standard electronic-design-automation (EDA) tools such as Cadence Virtuoso/Spectre, Calibre, Cadence EMX, and Ansys HFSS to design RF and mmWave modules. We validate the framework through the agentic design of a 240-251 GHz sub-THz transmitter in a 90 nm SiGe BiCMOS process. The agent semi-autonomously designed the sub-blocks including a 60 GHz push-push Colpitts VCO with second-harmonic extraction, multistage differential cascode power drivers, a push-push frequency doubler, a differential bowtie antenna, and all the interstage matching networks. The transmitter chain with parasitic-extracted devices and EM-verified passives delivers +5.1 to +6.3 dBm across an 11 GHz varactor-only tuning range, phase noise of -80 to -88 dBc/Hz at 1 
    
[^159]: FSVL：面向长江中游易混淆鱼类可解释识别与知识生成的分类学快-慢视觉-语言协同框架

    FSVL: Taxonomic Fast-Slow Vision-to-Language Collaborative Framework for Interpretable Confusable Fish Species Identification and Knowledge Generation in the Middle Yangtze River

    [https://arxiv.org/abs/2610.31880](https://arxiv.org/abs/2610.31880)

    该论文提出FSVL快-慢双通路框架，利用轻量级ResNet50快速通路结合基于大语言模型的检索增强生成（RAG）慢速通路，在保证推理效率的同时实现对长江中游易混淆鱼类的可解释细粒度识别和可靠生物学知识生成。

    

    arXiv:2610.31880v1 公告类型：交叉。摘要：长江流域是淡水生物多样性保护的关键区域。长江“十年禁渔”政策的实施使得有效的水生生物多样性监测和物种级鉴定变得更加重要。然而，由于形态差异细微，对近缘淡水鱼种进行细粒度识别仍然具有挑战性。现有的深度学习方法往往难以在分类准确率和推理效率之间取得平衡，而大语言模型（LLM）在生成专业生物学知识时可能产生缺乏依据的信息。为应对这些挑战，本研究提出了FSVL，一个融合视觉识别与基于大语言模型的检索增强生成（RAG）的快-慢双通路框架。轻量级的ResNet50快速通路对长江中游流域的51种鱼类进行初步分类，并将预定义的易混淆类群进行路由分流……

    arXiv:2610.31880v1 Announce Type: cross  Abstract: The Yangtze River Basin is a critical region for freshwater biodiversity conservation. The implementation of the Yangtze River Ten-Year Fishing Ban has increased the importance of effective aquatic biodiversity monitoring and species-level identification. However, fine-grained identification of closely related freshwater fish species remains challenging due to subtle morphological differences. Existing deep learning methods often struggle to balance classification accuracy and inference efficiency, while LLMs may generate unsupported information when producing specialized biological knowledge. To address these challenges, this study proposes FSVL, a fast-slow dual-pathway framework integrating visual recognition with LLM-based Retrieval-Augmented Generation (RAG). A lightweight ResNet50 fast path performs preliminary classification across 51 fish species in the Middle Yangtze River Basin and routes predefined groups of easily confused 
    
[^160]: 基于ISO 11783网络的机器可读操作手册，用于离线多语言大语言模型辅助

    Machine-Readable Operator Manuals over the ISO 11783 Network for Offline, Multilingual LLM Assistance

    [https://arxiv.org/abs/2610.31863](https://arxiv.org/abs/2610.31863)

    本文提出在ISO 11783网络上对操作手册进行语义同步，使拖拉机侧的离线多语言LLM助手能够在31至89秒内获取完整的机具文档，传输时间与操作员首次连接机具时等待对象池上传的时间相当。

    

    如果农业机械要支持智能化的离线辅助，那么系统必须能够动态访问技术文档。本文分析并定义了基于ISO 11783的CAN总线上操作手册的语义同步，即将拖拉机侧助手的本地知识库与所连接机具的操作手册保持一致的过程。论文使用Markdown、JSON和XML对数据传输性能进行了基准测试，并评估了多种开源权重的大语言模型（LLM）在跨语言查询任务中的表现。答案的自动评估结果经过四位人工评审员的验证。根据格式和传输设置的不同，一份完整手册通过物理ISO 11783网络的传输时间为31至89秒，这与操作员在首次连接机具时已经习惯等待的对象池上传时间相当。Markdown生成的文件最小，而……（摘要在此处截断）

    arXiv:2610.31863v1 Announce Type: cross  Abstract: If agricultural machinery is to support intelligent, offline assistance, then the system must access technical documentation dynamically. In this paper, the Semantic Synchronization of operator manuals over ISO 11783-based CAN buses is analyzed, defined as the process that brings the local knowledge base of a tractor-side assistant into agreement with the operator manual of the connected implement. Data transfer performance was benchmarked using Markdown, JSON, and XML; various open-weight Large Language Models (LLMs) were also evaluated in cross-lingual query tasks. The automatic evaluation of the answers was validated against four human raters. A complete manual was transferred over a physical ISO 11783 network in 31 to 89 s, depending on format and transport settings, which is comparable to the object pool upload that operators already wait for at the first coupling of an implement. Markdown produced the smallest files, while the qu
    
[^161]: SWE-Clar：编码智能体能否澄清不完整的客户需求并交付全新软件？

    SWE-Clar: Can Coding Agents Clarify Incomplete Client Requirements and Deliver Greenfield Software?

    [https://arxiv.org/abs/2610.31854](https://arxiv.org/abs/2610.31854)

    该论文提出了 SWE-Clar 基准，基于 100 个报酬至少 4,000 美元的真实 Upwork 客户委托任务，通过“先澄清需求、后实现代码”的多轮交互设计并以明确的需求理解测试来评估编码智能体在自由职业全新软件开发中澄清不完整客户需求的能力。

    

    arXiv:2610.31854v1 公告类型：交叉引用（cross）。摘要：在自由职业环境中，全新软件开发要求编码智能体将不完整的客户请求转化为可运行的软件。这使得需求澄清至关重要，然而客户可能无法在漫长的开发周期中提供持续反馈，而且即使收到客户的回答，也不能保证智能体能正确理解复杂需求。这些挑战促成了两项基准设计选择：需求澄清先于代码实现，并通过明确的需求理解测试来评估澄清的有效性。基于这些设计选择，我们提出了 SWE-Clar，这是一个基于真实、高价值客户委托的基准，用于评估该场景下的需求澄清与代码实现能力。该基准包含 100 个源自 Upwork 上真实软件开发工作的任务，每个任务的总报酬至少为 4,000 美元。每个任务均以编码智能体与客户智能体之间的多轮交互开始，

    arXiv:2610.31854v1 Announce Type: cross  Abstract: Greenfield software development in freelance settings requires coding agents to turn incomplete client requests into working software. This makes requirement clarification essential, yet clients may be unable to provide continuous feedback over long development cycles, and receiving their answers does not guarantee correct understanding of complex requirements. These challenges motivate two benchmark design choices: clarification precedes code implementation, and its effectiveness is assessed through explicit tests of requirement understanding. Guided by these choices, we introduce SWE-Clar, a benchmark grounded in real, high-value client commissions to evaluate requirement clarification and code implementation in this setting. It comprises 100 tasks based on real software development jobs on Upwork, each with at least US\$4,000 in total payments. Each task begins with multi-round interaction between a coding agent and a client agent, 
    
[^162]: 重新思考面向城市表示学习的多视图融合

    Rethinking Multi-View Fusion for Urban Representation Learning

    [https://arxiv.org/abs/2610.31850](https://arxiv.org/abs/2610.31850)

    本文实证揭示城市多视图融合中模态效用区域差异与多目标梯度不平衡两大结构性问题，并提出自适应稀疏MoE路由的自调节框架PRISM来学习区域特定的视图分配，实现更稳健的城市表示学习。

    

    城市区域表示学习将异构的城市信号（如人流移动、兴趣点和土地利用模式）编码为统一的嵌入表示，服务于下游任务，包括签到预测、犯罪预测和服务需求估计。现有方法依赖于全局共享的融合与联合优化，隐式假设各模态重要性相同且多目标训练天然平衡。然而，城市数据本质上是异构的。我们的实证分析揭示了两个结构性问题：（i）模态效用在区域间差异显著，导致静态融合过度强调某些模态的信号；（ii）异构目标之间存在持续的梯度不平衡，造成表示偏差和训练不稳定。我们提出了 PRISM，一个面向自适应多视图学习的自调节框架。PRISM 采用自适应的稀疏 MoE 路由来学习区域特定的视图分配，……（摘要在此处被截断）

    arXiv:2610.31850v1 Announce Type: cross  Abstract: Urban region representation learning encodes heterogeneous urban signals, such as mobility flows, points of interest, and land-use patterns, into unified embeddings for downstream tasks, including check-in prediction, crime forecasting, and service demand estimation. Existing approaches rely on globally shared fusion and joint optimization, implicitly assuming uniform modality importance and naturally balanced multi-objective training. However, urban data is inherently heterogeneous. Our empirical analysis reveals two structural issues: (i) modality utility varies markedly across regions, causing static fusion to overemphasize signals from certain modalities; and (ii) heterogeneous objectives exhibit persistent gradient imbalance, causing representational bias and unstable training. We present PRISM, a self-adjusting framework for adaptive multi-view learning. PRISM adapts sparse MoE routing to learn region-specific view allocations, w
    
[^163]: 不要忘记你刚学到的东西：自我导向修复将智能体的错误转化为持久教训

    Don't Forget What You Just Learned: Self-Directed Repair Turns Agent Mistakes Into Lasting Lessons

    [https://arxiv.org/abs/2610.31846](https://arxiv.org/abs/2610.31846)

    提出Self-Direction机制，让智能体在任务失败后自我反思提取教训，并立即蒸馏为轻量级低秩适配器（仅占基础模型约0.02%参数量）存入持久化库中，从而将错误转化为不遗忘、不破坏原有能力的持久化学习成果。

    

    智能体应当从错误中学习，从而使经验带来持续的改进。然而，当今的智能体从错误中恢复只是暂时的：上下文内的反思容易在会话之间被遗忘，并且会争夺有限的上下文空间，而全参数微调成本高昂，且可能干扰已有的能力。我们提出了Self-Direction（自我导向），这是一种让智能体从错误中学习的机制，它结合了临时性的上下文内重试与策略更新两者的优点。当智能体在任务中失败时，它会反思自己的行动轨迹以提取出一条教训，并立即将其提炼为一个轻量级的低秩适配器。通过保持原始权重冻结，Self-Direction防止了适配器训练对基础模型参数的覆盖，同时每个适配器仅需训练约相当于基础模型参数量0.02%的额外参数。随后，智能体会将这些适配器保存到一个持久化的库中。

    arXiv:2610.31846v1 Announce Type: new  Abstract: Intelligent agents should learn from their mistakes so that experience leads to continuous improvement. Yet today's agents recover from a mistake only temporarily: in-context reflections are easily forgotten across sessions and compete for limited context, while full-parameter fine-tuning is expensive and can interfere with existing capabilities. We introduce Self-Direction, a mechanism for agents to learn from mistakes that combines the benefits of temporary in-context retries and policy updates. When an agent fails a task, it reflects on its own trajectory in order to extract a lesson and immediately distills it into a lightweight low-rank adapter. By keeping the original weights frozen, Self-Direction prevents adapter training from overwriting base-model parameters while training additional parameters equivalent to ~0.02% of the base model's parameter count for each adapter. The agent then saves these adapters in a persistent library 
    
[^164]: SEVI：安全的证据式真实世界强化学习

    SEVI: Safe Evidential Real-World Reinforcement Learning

    [https://arxiv.org/abs/2610.31829](https://arxiv.org/abs/2610.31829)

    SEVI提出证据式安全评论家框架，在单次前向传播中同时估计约束违反风险与认知不确定性，并在真实世界策略学习中对不安全或证据不足的动作进行离线恢复策略干预，从而实现安全的真实世界强化学习。

    

    真实世界强化学习已经使富接触操作任务能够直接在物理机器人上进行训练，但在在线学习过程中的不安全交互仍然是实际部署的关键障碍。我们提出了安全证据式真实世界强化学习，这是一个在真实世界策略学习过程中进行可靠性感知安全干预的框架。SEVI引入了一个证据式安全评论家，能够在单次前向传播中同时估计未来的约束违反风险和认知不确定性。在在线学习期间，SEVI会在执行前评估任务策略的每个动作；如果预测该动作不安全，或该预测缺乏足够的证据支持，SEVI将用一个离线训练的恢复策略覆盖该动作，引导机器人前往风险更低的区域。我们在分布内和分布偏移条件下，于两个模拟操作任务和一个真实世界的USB插入任务上对SEVI进行了评估。

    arXiv:2610.31829v1 Announce Type: cross  Abstract: Real-world reinforcement learning has enabled contact-rich manipulation tasks to be trained directly on physical robots, but unsafe interactions during online learning remain a key barrier to practical deployment. We propose Safe Evidential Real-World Reinforcement Learning (SEVI), a framework for reliability-aware safety intervention during real-world policy learning. SEVI introduces an Evidential Safety Critic that estimates both future constraint-violation risk and epistemic uncertainty in a single forward pass. During online learning, SEVI evaluates each task-policy action before execution; if the action is predicted to be unsafe or the prediction is insufficiently supported by evidence, SEVI overrides it with an offline-trained recovery policy that guides the robot toward a lower-risk region. We evaluate SEVI on two simulated manipulation tasks and a real-world USB insertion task under in-distribution and distribution-shifted cond
    
[^165]: 查询所记住的：查询抽象程度决定古典阿拉伯语诗歌的词汇检索与稠密检索

    What the Query Remembers: Query Abstraction Governs Lexical and Dense Retrieval of Classical Arabic Poetry

    [https://arxiv.org/abs/2610.31821](https://arxiv.org/abs/2610.31821)

    该论文发现，在古典阿拉伯语诗歌检索中，查询保留诗歌原文措辞的程度（即查询的抽象程度）决定了检索器的优劣——记忆型查询下BM25领先所有稠密模型，而抽象查询下稠密编码器胜出，且BM25对措辞变化的敏感度远高于稠密模型。

    

    一个诗歌档案库能被搜索得多好，取决于用户还记得什么，而不仅仅是模型理解了什么。我们在古典阿拉伯语诗歌上验证了这一点，在两个语料库、三种索引模式和两种评估协议下，对九个稠密编码器和BM25进行了基准测试。哪个检索器胜出，取决于查询中仍然保留了多少诗歌原文的措辞。在“话到嘴边”（tip-of-the-tongue）型查询上，BM25领先所有稠密编码器，包括一个8B参数的最先进模型（Recall@5为0.746对0.717）；而在抽象查询上，BM25则跌至十个系统中的第八位（0.392对0.748）。对相同查询措辞的编辑实验表明，这种排名很大程度上反映了查询的构成：添加一句被记住的半行诗可使BM25提升38分；手工改写引用的词语后，BM25与8B编码器持平；而且BM25对被记住措辞的敏感度显著高于最强的稠密编码器。同样的维度也支配着解释增强方法。

    arXiv:2610.31821v1 Announce Type: cross  Abstract: How well a poetry archive can be searched depends on what the user still remembers, not only on what the model understands. We show this for Classical Arabic poetry, benchmarking nine dense encoders and BM25 over two corpora, three indexing modes and two evaluation protocols. Which retriever wins is governed by how much of a poem's wording the query still carries. On tip-of-the-tongue queries BM25 leads every dense encoder, including an 8B state-of-the-art model (0.746 vs. 0.717 Recall@5); on abstract queries it falls to eighth out of ten (0.392 vs. 0.748). Editing the wording of the same queries shows that this ranking largely reflects query composition: adding one remembered half-verse raises BM25 by 38 points, rewriting the quoted words by hand leaves BM25 level with the 8B encoder, and BM25's sensitivity to remembered wording is significantly steeper than that of the strongest dense encoder. The same axis controls explanation augme
    
[^166]: PrefO2O：通过离线到在线潜在偏好迁移实现零样本基于偏好的强化学习

    PrefO2O: Zero-Shot Preference-Based Reinforcement Learning via Offline-to-Online Latent Preference Transfer

    [https://arxiv.org/abs/2610.31812](https://arxiv.org/abs/2610.31812)

    提出PrefO2O框架，通过从源任务少量离线偏好数据中学习任务不变的潜在偏好空间，并利用置信度加权最优传输对齐进行在线奖励适应，实现了无需任何人类反馈的零样本基于偏好的强化学习。

    

    基于偏好的强化学习（PbRL）已成为一种有前景的范式，用于学习与人类意图一致的机器人行为。然而，其可扩展性受到根本性限制，因为每个新任务都需要收集昂贵的人类偏好标签。在这项工作中，我们提出了PrefO2O，这是一个离线到在线的潜在偏好迁移框架，能够在无需任何人类反馈的情况下，在未见过的目标任务上实现零样本PbRL。其核心思想是从源任务的少量离线偏好数据中学习一个任务不变的潜在偏好空间，在其中将可迁移的偏好结构与任务特定的变化解耦。学习到的这一流形使PrefO2O能够在分布偏移下将偏好结构和奖励先验迁移到新任务，同时通过置信度加权的最优传输对齐为在线奖励适应生成可靠的伪偏好监督。大量实验……

    arXiv:2610.31812v1 Announce Type: cross  Abstract: Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for learning robot behaviors aligned with human intent. However, its scalability is fundamentally limited by the need to collect expensive human preference labels for every new task. In this work, we present PrefO2O, an offline-to-online latent preference transfer framework that enables zero-shot PbRL on unseen target tasks without requiring any human feedback. The core idea is to learn a task-invariant latent preference space from a small amount of offline preference data on source tasks, where transferable preference structures are disentangled from task-specific variations. This learned manifold allows PrefO2O to transfer preference structures and reward priors to new tasks under distribution shift, while confidence-weighted optimal transport alignment generates reliable pseudo-preference supervision for online reward adaptation. Extensive experiments
    
[^167]: PaperFormalizer：作者在环的研究论文Lean自动形式化

    PaperFormalizer: Author-in-the-Loop Autoformalization of Research Papers in Lean

    [https://arxiv.org/abs/2610.31810](https://arxiv.org/abs/2610.31810)

    提出PaperFormalizer——一个基于Lean的“作者在环”智能体流水线，能够发现并修复研究论文证明中的漏洞，并在作者参与关键决策下将论文转化为经Lean验证的形式化版本。

    

    即使经过仔细校对的论文，其证明或所陈述的假设中仍可能存在微小的漏洞。我们提出了PaperFormalizer，一个基于Lean的智能体（agentic）流水线，帮助作者发现并弥补自己论文中的这些漏洞，将每篇论文转化为经Lean验证的版本。从LaTeX源文件出发，PaperFormalizer将论文转录为命题的依赖图，以自然语言对每个陈述的忠实性、歧义性和正确性进行审查，然后结合Mathlib、可选的领域库以及作者的个人工具包，在Lean中对这些陈述进行形式化与证明。由于只有作者本人清楚某个陈述的本意，以及如何最好地修复某个结果中的形式化漏洞，因此每当一个陈述存在多种可能的解读时，流水线都会停下来询问作者，并以自然语言和TeX符号的形式将每个修复方案作为建议呈现给作者，由作者决定接受、拒绝或调整（原文摘要在此处被截断）。

    arXiv:2610.31810v1 Announce Type: new  Abstract: Even a carefully proofread paper can leave small gaps in its proofs or its stated assumptions. We introduce PaperFormalizer, a Lean-based agentic pipeline that helps authors find and close such gaps in their own papers and turn each paper into a Lean-verified version. Starting from the LaTeX source, PaperFormalizer transcribes the paper into a dependency graph of statements, audits each statement in natural language for faithfulness, ambiguity and correctness, and then formalizes and proves the statements in Lean against Mathlib, optional domain libraries and the author's personal toolkit. Only the author knows what a statement is meant to say and how a formalization gap in a result is best fixed. The pipeline therefore stops and asks the author whenever a statement admits more than one reading, presents every repair, in natural language and TeX symbols, as a proposal that the author can accept, decline or redirect, and leaves to the aut
    
[^168]: 英语和中文中的居高临下与恩赐性语言检测

    Patronizing and Condescending Language Detection in English and Mandarin

    [https://arxiv.org/abs/2610.31808](https://arxiv.org/abs/2610.31808)

    该论文提出了一种自动检测英语和中文中居高临下与恩赐性语言的系统，其中基于微调BERT的方法表现显著优于静态词嵌入结合卷积神经网络的方法，在开发集上取得了0.528的F1分数。

    

    居高临下与恩赐性语言（PCL）是一种有害语言形式，它贬低弱势群体并对其施以居高临下的怜悯，从而强化负面的社会刻板印象。然而，与明显的仇恨言论不同，这类语言往往难以检测，即使对人类评判者来说也是如此。本文提出了一种自动检测PCL的系统。我们的第一种方法使用静态词嵌入和卷积神经网络进行分类，第二种方法则使用针对PCL任务微调的预训练BERT嵌入。第二种方法明显更为成功，在开发集上取得了0.528的F1分数。

    arXiv:2610.31808v1 Announce Type: new  Abstract: Patronizing and condescending language (PCL) is a form of harmful language that belittles and assumes pity for vulnerable groups, reinforcing negative societal stereotypes. However, unlike overt hate speech, it is often difficult to detect, even for human judges. This paper proposes a system for the automated detection of PCL. Our first approach uses static word embeddings and a convolutional neural network for classification, while our second approach uses pre-trained BERT embeddings fine-tuned for the PCL task. Our second approach was considerably more successful, and achieved an F1 score of 0.528 over the development set.
    
[^169]: Phronix：面向智能体的以能力为中心的学习范式

    Phronix: The Capability-Centric Learning Paradigm for Agents

    [https://arxiv.org/abs/2610.31789](https://arxiv.org/abs/2610.31789)

    提出Phronix，一种以能力为中心的智能体学习范式，通过预测-验证-纠正循环让智能体从自身执行中自主获取、验证并完善可复用的能力，摆脱对人工编写技能和记忆的依赖，并将失败任务转化为研究机会。

    

    科学计算日益依赖基于大语言模型的智能体来执行性能工程任务，例如插桩、分析和工作流移植。然而，智能体的效能很大程度上取决于其所处的运行框架（harness），而这些框架通常依赖人工编写的技能和人工整理的记忆。这限制了智能体的自主学习能力，迫使智能体反复重新探索解决方案，而不是保留并复用先前执行中获得的知识。我们提出了Phronix，这是一种以能力为中心的学习范式，使智能体框架能够从自身的执行中获取、验证并完善可复用的能力。Phronix通过预测器-验证器-纠正器循环运行：MCP工具及相关技能提出解决方案，工具执行过程对其进行验证，观察到的结果则用于完善所存储的能力。它将每一次观察到的交互提炼到能力库中，并对失败的任务进行深入研究而非直接放弃。

    arXiv:2610.31789v1 Announce Type: new  Abstract: Scientific computing increasingly relies on large language model-based agents for performance engineering tasks such as instrumentation, analysis, and workflow porting. However, agent effectiveness depends heavily on the surrounding harness, which typically relies on manually written skills and curated memory. This limits autonomous learning, forcing agents to repeatedly rediscover solutions rather than retain and reuse knowledge from prior executions. We introduce Phronix, a capability-centric learning paradigm that enables agentic harnesses to acquire, validate, and refine reusable capabilities from their own executions. Phronix operates through a predictor-validator-corrector loop: MCP tools and associated skills propose solutions, tool executions validate them, and observed outcomes refine the stored capabilities. It distills every observed interaction into a capability store and researches a failing task rather than abandoning it. F
    
[^170]: 基准测试出人意料地对上下文压缩友好

    Benchmarks Are Surprisingly Compaction-Friendly

    [https://arxiv.org/abs/2610.31778](https://arxiv.org/abs/2610.31778)

    该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。

    

    arXiv:2610.31778v1 公告类型： cross 摘要：长时程智能体任务目前正在将大语言模型（LLM）智能体推向超出其上下文窗口的范围。为了继续执行任务，智能体框架必须释放上下文窗口空间，这一过程被称为上下文压缩。压缩在设计上通常是有损的：智能体可能会悄悄丢失任务约束、指令，或者其早期决策背后的推理依据。尽管压缩已是编程智能体框架中的标准组件，但此前尚未有研究针对前沿模型在真实的长时程智能体任务上对压缩进行系统性研究。以往的工作仅在玩具型或合成的轨迹上评估较小的开源权重模型。我们固定了智能体框架，在其中实现了五种压缩策略，并在三个长时程基准上评估了三个前沿模型。对于两个具备100万上下文的模型，压缩阈值对任务性能没有可测量的影响，而对于上下文窗口最小的GPT-5.3-Codex……

    arXiv:2610.31778v1 Announce Type: cross  Abstract: Long horizon agentic tasks are now pushing LLM agents past their context window. To continue the task, the agent scaffold has to free up the context window, a process known as context compaction. Compaction is generally lossy by design: agents can silently lose task constraints, instructions, or their reasoning behind earlier decisions. Although compaction is a standard component of coding agent scaffolds, it has not been studied systematically with frontier models on realistic, long horizon agentic tasks. Previous work evaluate smaller open weight models on toy or synthetic trajectories. We fix the agent scaffold (OpenCode), implement five compaction strategies inside it, and evaluate three frontier models on three long horizon benchmarks (PaperBench, ProgramBench, PostTrainBench). For the two 1M context models, the compaction threshold does not measurably affect the task performance, whereas GPT-5.3-Codex, the model with the smallest
    
[^171]: 面向人工智能、大数据与信息通信技术的广义绿色与红色概念：多维框架与对策

    Generalized green and red concepts for artificial intelligence, big data, and information and communication technologies: Multidimensional framework and countermeasures

    [https://arxiv.org/abs/2610.31739](https://arxiv.org/abs/2610.31739)

    本文提出广义绿色AI（GGAI）与广义红色AI（GRAI）概念，并在统一的环境-经济-社会-技术/运营多维框架下将“红/绿”分类体系扩展至大数据、信息通信技术、电子电气工程与数字技术领域，以全面评估AI等技术带来的电力、水资源与材料等多维可持续性影响并提出应对对策。

    

    人工智能（AI）、大数据（BD）与信息通信技术（ICT）的快速扩张正推动电力、水资源和材料需求的大幅增长，预计到2028年数据中心将消耗美国6.7%–12%的电力（高于2023年的4.4%），而到2027年与AI相关的年用水需求估计达40亿至60亿立方米。现有的“绿色AI”与“红色AI”讨论主要聚焦于计算效率和碳排放，对于生命周期权衡、服务与使用层面的影响，以及能源之外更广泛的可持续性外部性问题所提供的指导十分有限。为弥补这一空白，本文提出了广义绿色AI（GGAI）与广义红色AI（GRAI）概念，并在统一的环境-经济-社会-技术/运营多维框架下，将该分类体系扩展至红色大数据（RBD）、红色信息通信技术（RICT），以及红色电子电气工程（REEE）与红色数字技术（RDT）等领域。

    arXiv:2610.31739v1 Announce Type: cross  Abstract: The rapid expansions of artificial intelligence (AI), big data (BD), and information and communication technologies (ICT) are driving substantial growth in electricity, water, and material demands, with data centers projected to consume 6.7--12\% of U.S. electricity by 2028 (up from 4.4\% in 2023) and AI-related water demand estimated at 4--6 billion m$^3$ annually by 2027. Existing ``green AI'' and ``red AI'' discussions have largely emphasized computational efficiency and carbon emissions, providing limited guidance for lifecycle trade-offs, service-versus-use impacts, and broader sustainability externalities beyond energy. To bridge this gap, this paper proposes generalized green AI (GGAI) and generalized red AI (GRAI) and extends the taxonomy to red BD (RBD) and red ICT (RICT) as well as red electronic and electrical engineering (REEE) \& red digital technologies (RDT) under a unified Environmental-Economic-Social-Technical/Operati
    
[^172]: 学会被监测：面向思维链可监测性的因果界与干预感知的后训练方案

    Learning to Be Monitored: Causal Bounds and Intervention-Aware Post-Training Recipes for Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2610.31733](https://arxiv.org/abs/2610.31733)

    该论文指出常用的 CoT 监测评估指标 TPR_mc 只是因果检测率的上界并推导出互补下界，据此建立了更严格的 CoT 可监测性评估框架，并探索通过后训练方法使大语言模型的思维链更易于被监测。

    

    思维链（CoT）监测旨在从大型语言模型（LLM）的思维链中检测其潜在的有害行为。基于干预的评估方法通过检验“当目标行为发生时，监测器能否从思维链中识别出干预的作用”来衡量这一属性，通常采用 Guan 等人（2025）提出的最低标准敏感度（$\mathsf{TPR_{mc}}$）指标。然而，在干预下观察到的目标行为并不一定是由干预引起的，我们证明该指标可能会高估监测器对“实际由干预引起的目标行为”的检出频率。具体而言，我们将 $\mathsf{TPR_{mc}}$ 确立为该因果检测率的上界，并推导出一个互补的下界，从而为 CoT 可监测性提供了更严格的评估框架。借助该框架，我们研究了能否通过训练使 LLM 的思维链变得更易于监测。我们首先构建了一个训练（后文截断）

    arXiv:2610.31733v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) monitoring aims to detect potentially harmful behaviors in large language models (LLMs) from their CoTs. Intervention-based evaluations measure this property by asking whether a monitor can detect the role of an intervention from the CoT when the targeted behavior occurs, typically using minimal-criterion sensitivity ($\mathsf{TPR_{mc}}$) proposed by~\citep{guan2025monitoring}. However, a target behavior observed under intervention is not necessarily caused by it, and we show that this metric can overestimate how often a monitor detects target behaviors actually caused by the intervention. Specifically, we establish $\mathsf{TPR_{mc}}$ as an upper bound on this causal detection rate and derive a complementary lower bound, which provides a tighter evaluation framework for CoT monitorability. With this framework, we study whether LLMs can be trained to make their CoTs easier to monitor. We first construct a trainin
    
[^173]: 基于大语言模型的微服务系统自回归多模态异常检测

    Autoregressive Multimodal Anomaly Detection with Large Language Models for Microservice Systems

    [https://arxiv.org/abs/2610.31722](https://arxiv.org/abs/2610.31722)

    该论文提出了LLMMAD，一个基于大语言模型的自回归多模态异常检测框架，通过在跨模态融合中保留模态特定特征并解决“身份捷径”问题，来提升微服务系统的异常检测效果。

    

    微服务系统已成为大规模云应用的主流架构，支持快速部署和弹性伸缩。然而，确保微服务系统的可靠性和可用性关键依赖于有效的异常检测，因为故障和性能异常会在服务之间快速传播，严重影响端到端的系统行为。现有的微服务系统多模态异常检测方法通常面临两个根本性的局限：（i）为了跨模态融合而忽略了模态特定的特征，以及（ii）影响特征重构过程的“身份捷径”问题。为了解决这些问题，我们提出了LLMMAD，一个基于大语言模型的微服务系统自回归多模态异常检测框架。首先，为了协调跨模态融合与模态特定特征的保留，LLMMAD将（原文在此处截断）

    arXiv:2610.31722v1 Announce Type: cross  Abstract: Microservice systems have become a dominant architecture for large-scale cloud applications, enabling rapid deployment and elastic scaling. However, ensuring the reliability and availability of microservice systems critically relies on effective anomaly detection, as failures and performance anomalies can quickly propagate across services and severely impact end-to-end system behavior. Existing multimodal anomaly detection approaches for microservice systems often struggle with two fundamental limitations: (i) the neglect of modality-specific characteristics in favor of cross-modality fusion, and (ii) the "identity-shortcut" problem that affects the feature reconstruction process. To address these issues, we propose LLMMAD, an autoregressive multimodal anomaly detection framework for microservice systems based on Large Language Models (LLMs). Initially, to reconcile cross-modality fusion with modality-specific preservation, LLMMAD embe
    
[^174]: 面向物理信息优化的端到端可微扩散条件化方法

    End-to-End Differentiable Diffusion Conditioning for Physics-Informed Optimization

    [https://arxiv.org/abs/2610.31721](https://arxiv.org/abs/2610.31721)

    D2C利用确定性采样使反向去噪过程可微的特性，将冻结的条件扩散模型转化为端到端的梯度优化器，在测试时通过代理目标和物理信息的梯度直接优化条件变量，从而克服了离线基于模型优化中现有方法过于保守或无法精确对齐物理目标的问题。

    

    在昂贵的评估成本和复杂约束下优化高维物理系统是科学与工程领域的核心挑战。在离线基于模型的优化（MBO）中，方法必须仅使用固定数据集来改进设计，并且在测试时无法访问真实的目标函数。然而，现有方法通常存在过于保守、对无约束代理模型过度利用等问题，或者只能作为静态的推理时采样器，无法针对精确的物理目标对候选方案进行优化。在本文中，我们提出了用于离线基于模型优化的可微扩散条件化方法，该方法利用了一个关键事实：确定性采样使得反向去噪链相对于条件信号是可微的。D2C将冻结的条件扩散模型转变为端到端的基于梯度的优化器，其方式是在测试时通过来自冻结代理目标和物理信息（原文摘要在此处截断）的梯度直接优化条件变量。

    arXiv:2610.31721v1 Announce Type: cross  Abstract: Optimizing high-dimensional physical systems under expensive evaluations and complex constraints is a central challenge in science and engineering. In offline model-based optimization (MBO), methods must improve designs using only a fixed dataset and no test-time access to the true objective. However, existing methods typically suffer from conservatism, unconstrained surrogate exploitation, or function as static inference-time samplers that cannot refine candidates against precise physical objectives. In this paper, we propose Differentiable Diffusion Conditioning (D2C) for offline MBO, which leverages the fact that deterministic sampling makes the reverse denoising chain differentiable with respect to the conditioning signal. D2C turns a frozen conditional diffusion model into an end-to-end gradient-based optimizer by directly optimizing conditioning variables at test time via gradients from frozen surrogate objectives and physics-inf
    
[^175]: 开源项目对编码智能体说了什么：AGENTS.md 与 CLAUDE.md 文件普查

    What Open-Source Projects Tell Their Coding Agents: A Census of AGENTS.md and CLAUDE.md

    [https://arxiv.org/abs/2610.31715](https://arxiv.org/abs/2610.31715)

    这项研究首次对十三种语言的活跃 GitHub 仓库中的 AGENTS.md 和 CLAUDE.md 指令文件进行大规模普查，系统揭示了开源项目向 AI 编码智能体传达的指令构成（项目知识与代码、构建、工作流规则的混合），以及各项目在 AI 署名披露等问题上的立场分歧。

    

    编码智能体可以通过仓库中的指令文件获得引导。我们通过对十三种语言的热门活跃 GitHub 仓库进行普查来考察这一新兴实践，将每个仓库根目录下的 AGENTS.md 和 CLAUDE.md 视为一个指令集。智能体指令出现在一定比例的仓库以及星标最高的仓库之中；当两个文件同时存在时，CLAUDE.md 通常指向 AGENTS.md。我们采用模型辅助标注与人工审核相结合的方式分析这些指令语句。标注结果显示，指令内容是项目知识与关于代码、构建和工作流规则的实际混合体。协作规则会指明具体要求，从测试计划到提交格式，但相当一部分指令并未说明理由。各项目在 AI 署名问题上立场不一：提交信息规则禁止 AI 署名的情况多于要求署名，而拉取请求规则则更倾向于要求披露 AI 的参与。三分之一的指令集带有与生成器相关的签名、工具插入的内容块、共享文本或其他生成痕迹（摘要在此处截断）。

    arXiv:2610.31715v1 Announce Type: cross  Abstract: Coding agents can be guided with instruction files in a repository. We examine this emerging practice through a census of popular, active GitHub repositories in thirteen languages, treating each repository's root AGENTS.md and CLAUDE.md as one instruction set. Agent instructions appear in of repositories and of the most-starred; where both files exist, CLAUDE.md usually points to AGENTS.md. We analyse statements using model-assisted annotation and human review. The annotations reveal a practical mix of project knowledge and rules about code, building and workflow. Collaboration rules name concrete requirements, from test plans to commit formats, yet of directives give no reason. Projects disagree on AI attribution: commit-message rules more often prohibit it than require it, while pull-request rules more often require disclosure. A third of instruction sets have a generator-associated signature, a tool-inserted block, shared text or an
    
[^176]: 复用、重写还是拒绝：面向重复移动端GUI控制的证据约束轨迹复用

    Reuse, Rewrite, or Reject: Evidence-Constrained Trace Reuse for Repeated Mobile GUI Control

    [https://arxiv.org/abs/2610.31712](https://arxiv.org/abs/2610.31712)

    该论文提出图-文本模板与“记录-模板-回放”框架，通过对UI状态转移进行证据约束验证，实现移动端GUI智能体在重复任务中对先前执行轨迹的选择性复用，从而降低冗余的LLM推理与执行开销。

    

    移动端GUI智能体经常执行具有相似交互流程的重复性任务。复用先前的执行记录可以减少冗余的大语言模型（LLM）推理与执行开销，但用户指令和执行环境的变化可能使先前成功的操作失效。我们系统地将这些变化归纳为六个代表性类别，并观察到此类变化并不一定会使整个流程失效：在针对当前任务上下文和界面状态进行验证后，单个状态转移仍可保持可复用性。基于这一洞察，我们提出了用于选择性执行轨迹复用的图-文本模板。图结构捕获UI状态和动作转移，而文本编码任务意图、流程结构以及用于可变内容的类型化槽位。一个“记录-模板-回放”框架从经过验证的执行中构建模板，并利用当前指令和实时UI状态来复用……

    arXiv:2610.31712v1 Announce Type: cross  Abstract: Mobile GUI agents frequently perform repetitive tasks with similar interaction procedures. Reusing previous executions can reduce redundant LLM inference and execution overhead, but variations in user instructions and execution environments may invalidate previously successful actions. We systematically characterize these variations into six representative categories and observe that such changes do not necessarily invalidate entire procedures: individual transitions can remain reusable when validated against the current task context and interface state. Based on this insight, we propose graph-text templates for selective execution-trace reuse. The graph captures UI states and action transitions, while the text encodes task intent, procedural structure, and typed slots for variable content. A record-template-replay framework constructs templates from verified executions and leverages the current instruction and live UI state to reuse, 
    
[^177]: MBABench-2.0：超越简单可验证性，评估AI智能体构建财务电子表格的能力

    MBABench-2.0: Evaluating AI Agents on Constructing Financial Spreadsheets Beyond Simple Verifiability

    [https://arxiv.org/abs/2610.31711](https://arxiv.org/abs/2610.31711)

    该论文提出MBABench-2.0基准测试，包含101个从零构建财务电子表格的原创任务，突破了以往填空式简单可验证任务的局限，用于更真实地评估AI智能体完成建模决策、格式设计和表格布局等核心电子表格构建工作的能力。

    

    电子表格是现代金融领域的通用语言，支撑着估值、预测、审计以及该行业的众多核心工作流程。随着大语言模型（LLM）智能体在知识工作自动化方面不断进步，电子表格构建成为一个自然而贴切的测试平台。然而，大多数现有的电子表格基准测试采用填空式任务来评估智能体，即智能体在基本完整的电子表格中进行少量编辑。这种评估协议未能捕捉到电子表格工作流程中最核心且最耗时的部分：普通知识工作者从零开始构建工作簿时需要费力完成的建模决策、格式设计和表格布局。我们推出了MBABench-2.0，这是一个包含101个原创电子表格构建任务的基准测试，其中最难的任务预计需要具有3年以上经验的建模者花费33小时才能完成。与编程或数学领域不同——在那些领域中，单元测试或最终答案即可验证智能体的输出——财务电子表格的质量

    arXiv:2610.31711v1 Announce Type: cross  Abstract: Spreadsheets are the lingua franca of modern finance, driving valuation, forecast, audit, and many other core workflows of the industry. As LLM agents advance to automate knowledge work, spreadsheet construction is a natural and pertinent testbed. However, most existing spreadsheet benchmarks evaluate agents on fill-in-the-blank style tasks, where an agent performs minor edits in otherwise complete spreadsheets. This protocol fails to capture the most central and time-consuming part of the spreadsheet workflow: the modeling decisions, format design, and sheet layout regular knowledge workers need to toil through to build a workbook from scratch. We introduce MBABench-2.0, a benchmark of 101 original spreadsheet construction tasks, the hardest of which is estimated to take 33 hours for a modeler with 3+ years of experience. Unlike coding or math, where a unit test or a final answer can verify an agent's output, the quality of a financia
    
[^178]: Compact-Bench：衡量上下文压缩对长程编程智能体的代价

    Compact-Bench: Measuring What Context Compaction Costs Long-Horizon Coding Agents

    [https://arxiv.org/abs/2610.31710](https://arxiv.org/abs/2610.31710)

    提出 Compact-Bench 基准，通过在共享仓库状态下配对“完整历史”与“仅摘要”的延续过程，首次将上下文压缩对长程编程智能体的具体损害（如抹去 5-64% 的成功修复、最多仅保留 16% 的重复细节）从效率指标中隔离出来进行量化评估。

    

    上下文压缩（context compaction）让长程编程智能体能够在超出其上下文窗口后继续运行，但可能会丢失完成任务所需的状态信息。以往的评估衡量的是效率、信息保留程度或端到端成功率，难以隔离某一次特定的历史替换所带来的影响。Compact-Bench 从相同的仓库状态出发，将“完整历史延续”与“仅摘要延续”进行配对对比，并评估重复出现的细节以及连续的多轮生产环境压缩。该基准涵盖 35 个 SWE-bench 修复任务、146 个智能体会话和 106 条生产压缩链。在 14 个摘要模型骨干上，稳定造成的损害范围达到 22 个完整历史成功案例中的 5-64%：Claude Opus 4.8 保留了最多的修复结果，而 DeepSeek-Flash 造成的损害最大。以 Kimi-K3 作为摘要模型时，Kimi Code 提示词保留了最多的任务延续（19/22），而 Claude Code 保留了最多的重复细节（15.0%）。LLM 摘要最多只能保留 16.0% 的重复细节，相比……（原文摘要至此截断）

    arXiv:2610.31710v1 Announce Type: cross  Abstract: Context compaction lets long-horizon coding agents continue beyond their context windows, but can erase the state needed to finish a task. Prior evaluations measure efficiency, retained information, or end-to-end success, leaving the effect of a specific history replacement difficult to isolate. Compact-Bench pairs full-history and summary-only continuations from a shared repository state, and evaluates recurring details and successive production compactions. It covers 35 SWE-bench repairs, 146 agentic sessions, and 106 production chains. Across 14 summarizer backbones, stable damage ranges from 5-64% of 22 full-history successes: Claude Opus 4.8 preserves the most repairs, while DeepSeek-Flash damages the most. With Kimi-K3 as summarizer, the Kimi Code prompt preserves the most task continuations (19/22), whereas Claude Code retains the most recurring details (15.0%). LLM summaries retain at most 16.0% of recurring details, compared w
    
[^179]: 挖掘多智能体工作流的设计空间

    Mining the Design Space of Multi-Agent Workflows

    [https://arxiv.org/abs/2610.31709](https://arxiv.org/abs/2610.31709)

    提出一个两阶段流水线，通过挖掘28,568个可执行工作流的代码与执行轨迹，自动提炼出可复现的多智能体系统设计原则。

    

    设计基于大语言模型（LLM）的多智能体系统（MAS）目前仍依赖于反复试错和非正式的“民间智慧”。人工研究仅测试少数几种通信结构或拓扑（例如串行、并行、共享内存或辩论），而自动化优化器在选定特定任务的工作流后便丢弃搜索轨迹。我们提出了一个两阶段流水线，用于从工作流代码和执行轨迹中挖掘可复现的设计原则。我们将AFlow的MCTS优化器重新应用于六种可组合算子和26个经过确定性验证的Enigmata任务，保留所有中间轮次，从而构建了一个包含28,568个可执行工作流的语料库。随后，一个沙盒化的AutoResearch循环在L0惩罚验证目标下挖掘编排代码和提示模板，并在留出的搜索任务上测试候选设计原则。在300次特征挖掘试验中，我们识别出了在三个层级的留出搜索上持续存在的效应。

    arXiv:2610.31709v1 Announce Type: cross  Abstract: Designing large language model (LLM) multi-agent systems (MAS) still relies on trial and error and informal "folk wisdom." Manual studies test only a few communication structures or topologies (e.g., sequential, parallel, shared-memory, or debate), while automated optimizers discard search trajectories after selecting a task-specific workflow. We present a two-phase pipeline for mining reproducible design principles from workflow code and execution traces. We repurpose AFlow's MCTS optimizer across six composable operators and 26 deterministically verified Enigmata tasks, retaining all intermediate rounds to build a corpus of 28,568 executable workflows. A sandboxed AutoResearch loop then mines orchestration code and prompt templates under an $L_0$-penalized validation objective and tests candidate principles on held-out searches. Across 300 feature-mining trials, we identify effects that persist on held-out searches across three level
    
[^180]: RubricCraft：解法分歧之处，评判标准自现

    RubricCraft: Where Solutions Diverge, Criteria Emerge

    [https://arxiv.org/abs/2610.31707](https://arxiv.org/abs/2610.31707)

    RubricCraft 通过对比多个候选解法之间的满足与违反需求的差异来构建文本评分规则，从而为开放式任务提供显式且可靠的自动化评估标准。

    

    在开放式、产出工件的任务上评估语言智能体十分困难：基于模型的评分器可能表现不一致，而穷尽式的程序化检查往往不切实际。文本化的评分规则能使评估标准显式化，但撰写它们需要领域专业知识。仅从任务描述生成的评分规则可能会遗漏那些在尝试解题的过程中才显现出来的需求和失败模式。我们提出了 RubricCraft，它将评分规则的构建视为一种由候选解引导的自然语言规范生成过程。给定任务描述和独立采样的求解器轨迹，RubricCraft 比较各次尝试如何满足或违反任务要求，并利用这些差异来构建评分标准。它将所得标准应用于收集到的解法，并修正那些忽略错误或惩罚有效替代方案的检查项。由此产生的评分规则能够捕捉部分正确性，而无需（原文在此处截断）

    arXiv:2610.31707v1 Announce Type: cross  Abstract: Evaluating language agents on open-ended, artifact-producing tasks is difficult: model-based graders can be inconsistent, while exhaustive programmatic checks are often impractical. Textual rubrics make evaluation criteria explicit, but writing them requires domain expertise. Rubrics generated from task descriptions alone can miss requirements and failure modes that become apparent in attempted solutions. We introduce RubricCraft, which treats rubric construction as the generation of natural-language specifications guided by candidate solutions. Given a task description and independently sampled solver trajectories, RubricCraft compares how each attempt satisfies or violates the task requirements and uses these differences to construct rubric criteria. It applies the criteria to the collected solutions and revises checks that overlook errors or penalize valid alternatives. The resulting rubrics capture partial correctness without requi
    
[^181]: 提交工单之前：利用智能体式大语言模型助手，将HPC用户支持根植于文档与已解决工单之中

    Before the Ticket: Grounding HPC User Support in Documentation and Resolved Tickets with an Agentic LLM Assistant

    [https://arxiv.org/abs/2610.31703](https://arxiv.org/abs/2610.31703)

    该论文提出了一个完全本地化部署的智能体式LLM助手，通过MCP协议编排实时文档RAG检索与在11,244条历史工单“问题-解决方案”对上QLoRA微调的Llama-3-8B模型并融合二者答案，从而在用户提交工单之前就解答其HPC使用问题。

    

    HPC（高性能计算）帮助台收到的大多数问题，其实早已在计算中心的用户指南或他人此前提交的工单中得到过解答；但用户仍然不断提交新工单，原因在于指南难以导航检索，而工单历史也不允许用户自行搜索。我们提出了一个能够同时从这两种信息来源作答的助手：一个编排器通过模型上下文协议（Model Context Protocol, MCP）调用两个工具——一个针对实时文档的检索增强生成（RAG）工具，以及一个在从历史工单中挖掘出的11,244对“问题-解决方案”数据上使用QLoRA微调的Llama-3-8B模型——并将二者的答案加以融合。由于工单数据不得离开本机构，所有模型均在本地部署，或运行于大学托管的LLM服务之上。在普渡大学罗森先进计算中心的400个保留工单（与训练集互不相交且不含近似重复样本）上进行评估，由两位来自未参与生成任何答案的模型家族的评判者进行打分，结果表明在历史工单解决方案上进行微调……（原文摘要在此处被截断）

    arXiv:2610.31703v1 Announce Type: cross  Abstract: Most questions that reach an HPC help desk have already been answered in the center's user guides or in a ticket someone else filed last semester; users still file new tickets because the guides are hard to navigate and the ticket history is not theirs to search. We present an assistant that answers from both sources: an orchestrator calls two tools over the Model Context Protocol, retrieval-augmented generation (RAG) over the live documentation and a Llama-3-8B model fine-tuned with QLoRA on 11,244 query-resolution pairs mined from historical tickets, and merges their answers. Every model runs on-premises or on the university's hosted LLM service because ticket data may not leave the institution. On 400 held-out tickets from Purdue's Rosen Center for Advanced Computing, disjoint from training and free of near-duplicates, and evaluated by two judges from model families that generated none of the answers, fine-tuning on resolutions rewr
    
[^182]: 面向室内无线电地图估计的协作式上下文学习

    Collaborative In-Context Learning for Indoor Radio Map Estimation

    [https://arxiv.org/abs/2610.31702](https://arxiv.org/abs/2610.31702)

    提出Poly-Ripple框架，首次将室内无线电地图估计形式化为协作式上下文学习问题，使多个客户端利用大型自回归视觉模型和波纹序自回归建模，在无需模型更新并保护隐私的前提下协作预测无线电地图。

    

    室内无线电地图估计（RME）对于覆盖规划、定位和网络部署等无线应用至关重要。然而，现有的大多数方法遵循监督学习范式，需要昂贵的标注数据和针对特定任务的模型训练。此外，室内环境涉及隐私敏感信息，例如建筑布局和结构，这些信息通常无法集中收集。为应对这些挑战，我们首次尝试将室内无线电地图估计形式化为一个协作式上下文学习（ICL）问题。我们提出了Poly-Ripple框架，其中多个客户端使用大型自回归视觉模型（LAVM）执行上下文学习，而无需进行模型更新。每个客户端由其私有的示例进行提示，并针对相同的场景查询预测无线电地图。为了捕捉无线信号从发射机（Tx）向外传播的因果结构，我们采用了波纹序自回归建模方法。

    arXiv:2610.31702v1 Announce Type: cross  Abstract: Indoor radio map estimation (RME) is essential for wireless applications such as coverage planning, localization, and network deployment. However, most existing methods follow a supervised learning paradigm that requires costly labeled data and task-specific model training. Moreover, indoor environments involve privacy-sensitive information, such as building layouts and structures, that often cannot be centralized. To address these challenges, we make the first attempt to formulate indoor RME as a collaborative in-context learning (ICL) problem. We propose Poly-Ripple, a framework in which multiple clients perform ICL using large autoregressive vision models (LAVMs) without model updates. Each client is prompted by its private demonstrations and predicts the radio map for the same scene query. To capture the causal structure of wireless propagation outward from the transmitter (Tx), we adopt ripple-ordered autoregressive modeling. We f
    
[^183]: 权衡感知的销售区域设计：面向企业客户-销售分配的多目标决策系统

    Tradeoff-Aware Territory Design: A Multi-Objective Decision System for Enterprise Customer-to-Seller Allocation

    [https://arxiv.org/abs/2610.31701](https://arxiv.org/abs/2610.31701)

    该论文提出了一个面向企业B2B销售区域设计的生产级多目标决策系统，通过融合多维客户价值评分、基于反事实框架的销售关系连续性评估以及多目标优化，在增长、均衡、销售能力与可行性之间实现权衡感知的客户-销售分配。

    

    企业客户旅程不仅受产品体验和营销的影响，还受反复进行的资源分配决策的塑造。在B2B市场进入（GTM）组织中，销售区域划分决定了哪些客户账户获得销售人员的覆盖、哪些销售-客户关系得以保留，以及有限的现场销售能力如何分配。然而这一过程往往仍依赖人工操作，使得比较机会、衡量关系连续性的价值，以及评估增长、均衡、能力与可行性之间的权衡变得困难。我们提出了一套用于企业GTM规划中权衡感知的客户-销售分配的生产级决策系统。该系统集成了三个组件：一个多维客户价值层，结合预测信号对机会进行评分；一个观察性连续性评估层，使用合成控制式的反事实框架来估计中断销售-客户关系的成本；以及一个多目标优化组件（原文摘要在此处截断）。

    arXiv:2610.31701v1 Announce Type: new  Abstract: Enterprise customer journeys are shaped not only by product experiences and marketing, but also by recurring resource-allocation decisions. In B2B go-to-market (GTM) organizations, territory carving determines which accounts receive seller coverage, which seller-account relationships are preserved, and how limited field capacity is distributed. Yet the process often remains manual, making it difficult to compare opportunities, value continuity, and evaluate tradeoffs among growth, balance, capacity, and feasibility.   We present a production decision system for tradeoff-aware customer-to-seller allocation in enterprise GTM planning. It integrates three components: a multidimensional account-value layer that combines predictive signals to score opportunity; an observational continuity-valuation layer that estimates the cost of breaking seller-account relationships using a synthetic-control-style counterfactual framework; and a multi-objec
    
[^184]: 大语言模型的分布对齐及其泛化特性

    Distributional Alignment of LLMs and Its Generalization Properties

    [https://arxiv.org/abs/2610.31687](https://arxiv.org/abs/2610.31687)

    该研究提出了大语言模型与人类分布对齐的方法，发现对齐后的模型能够泛化到同领域的未见问题，跨主题泛化取决于主题间的关联性，但可能伴随创造力等能力的意外退化。

    

    大语言模型越来越多地被用于模拟和探究人类行为，例如用于问卷调查。对于这些应用而言，一个关键属性是输出满足分布对齐：即模型的输出分布与某种参考（此处为人类）分布相匹配。实现分布对齐的一个障碍是如何从少量训练数据中进行泛化。在这项工作中，我们研究了：(1) 如何对大语言模型进行分布对齐；(2) 对齐后的大语言模型如何泛化到未见过的样本以及不相关的主题和任务。通过两个大规模基准测试，我们发现大语言模型确实可以实现分布对齐，并能泛化到同一领域中未见过的的问题。跨主题泛化则取决于主题之间的关系（例如，关于政治的问题与对机构的信任相关）。然而，对齐后的大语言模型可能会以令人惊讶的方式发生退化（例如，失去其创造力），这表明其泛化特性……

    arXiv:2610.31687v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate and probe human behavior, e.g., for surveys. For these applications, a key property is that outputs satisfy distributional alignment: to match its output distribution with some reference (in this case, human) distribution. An obstacle in achieving distributional alignment is generalizing from small training data. In this work, we study (1) how to distributionally align LLMs; and (2) how aligned LLMs might generalize to unseen samples and unrelated topics and tasks. Using two large-scale benchmarks, we find that LLMs can indeed achieve distributional alignment and generalize to unseen questions of the same domain. Cross-topic generalization depends on the relationship between topics (e.g., questions about politics are related to trust in institutions). However, aligned LLMs can degrade in surprising ways (e.g., lose their creative ability), suggesting that its generalization proper
    
[^185]: 从目标选择到条件复用：审计工具使用模型中的残差干预

    From Target Selection to Condition Reuse: Auditing Residual Interventions in Tool-Use Models

    [https://arxiv.org/abs/2610.31686](https://arxiv.org/abs/2610.31686)

    激活干预虽能大幅提升模型对熟悉目标的选择正确率，但这种增益无法泛化到事实与标识符的重新组合或独立资格判断上，仅能改变源事实影响判断的方式。

    

    激活干预可以使语言模型产生目标答案，但模型能否在其他计算中使用所预期的条件？我们在Qwen3.5-27B上研究了这一问题，任务为工具使用场景，其中时间或权限决定候选者的资格与选择。在32个新构建的世界中，两种针对具体案例的残差写入变体将熟悉目标选择的正确率从35/128提升至119/128，以及从33/128提升至114/128。然而，当熟悉的事实和标识符在同一个选择任务中被重新组合时，这些增益便消失了，同时先前被奖励过的标识符被选中的比例反而上升。这些增益在不同输出格式下保持稳定，但在关于候选者是否符合资格或应否被选中的独立判断上几乎没有改善。尽管如此，内部编辑仍能够改变源事实对判断的影响方式：在一项预先设定的次要比较中，当某个源事实被使用时，早期层的源补丁改变了64个判断中的50个……

    arXiv:2610.31686v1 Announce Type: cross  Abstract: Activation interventions can make a language model produce a target answer, but can the model use the intended condition in other computations? We study this question in Qwen3.5-27B on tool-use tasks where time or permissions determine eligibility and selection. Across 32 fresh worlds, two case-specific residual-writing variants raise familiar target-selection correctness from 35/128 to 119/128 and from 33/128 to 114/128. Yet these gains disappear when familiar facts and identifiers are recombined within the same selection task, while selection of previously rewarded identifiers increases. The gains persist across output formats but yield little improvement in independent judgments about whether a candidate is eligible or should be selected. Internal edits can nevertheless change how a source fact affects a judgment: in a prespecified secondary comparison, an early-layer source patch changes 50/64 judgments when that source is used, wi
    
[^186]: 一种基于证据校验的智能体方法，用于对话中的多模态情感推理

    An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation

    [https://arxiv.org/abs/2610.31685](https://arxiv.org/abs/2610.31685)

    该论文提出了一种可追溯的多模态情感推理智能体框架，通过模态感知智能体发布带证据引用的观察、集成智能体生成情感标签与引用，并利用独立的基于证据的验证阶段执行确定性检查来定位错误来源，从而以低成本实现对话中的多模态情感推理。

    

    基于大语言模型（LLM）的多模态情感推理系统在标准基准测试中表现良好，但通过指令微调来改进它们的计算成本可能很高。由冻结模型构建的智能体流水线可以避免这一成本，但带来了另一个挑战：当预测出错时，系统必须识别是哪个智能体或哪个阶段导致了错误。我们通过一个可追溯的多模态情感推理与提示空间自适应的智能体框架来解决这一问题。针对特定模态的感知智能体分析视觉、声学和文本输入，并将附带证据引用的观察结果发布到共享看板上。一个集成智能体将这些观察结果与检索到的多模态示例相结合，生成情感描述、标签集合以及支持性引用，同时多模态大语言模型（MLLM）评论者可以在输出最终确定之前要求进行一次修订。对于错误的预测，一个独立的基于证据的验证阶段会执行确定性检查

    arXiv:2610.31685v1 Announce Type: cross  Abstract: LLM-based multimodal emotion reasoning systems perform well on standard benchmarks, but improving them through instruction tuning can be computationally expensive. Agentic pipelines built from frozen models avoid this cost, but introduce a different challenge: when a prediction is wrong, the system must identify which agent or stage caused the error. We address this problem with an agentic framework for traceable multimodal emotion reasoning and prompt-space adaptation. Modality-specific perception agents analyze visual, acoustic, and lexical inputs and post evidence-cited observations to a shared board. An integration agent combines these observations with retrieved multimodal examples to produce an emotion description, label set, and supporting citations, while an MLLM critic can request one revision before the output is finalized. For incorrect predictions, a separate evidence-grounded verification stage applies deterministic checks
    
[^187]: 重新思考结构化生成：基于图的推理能否解决歧义问题？

    Rethinking Structured Generation: Can Graph-Based Reasoning Resolve Ambiguity?

    [https://arxiv.org/abs/2610.31681](https://arxiv.org/abs/2610.31681)

    该论文将结构化生成重新表述为歧义感知推理问题，提出基于图推理的多有效学习框架，通过对一组合理的结构化输出进行推理而非预测单一解，解决了自然语言歧义导致现有方法输出不稳定和不一致的问题。

    

    结构化生成通常被表述为从输入到单一结构化输出的确定性映射。这一假设与自然语言存在根本性的不匹配，因为自然语言往往允许多种结构上不同但语义上均有效的解释。因此，现有方法依赖于单一参考监督，将歧义压缩为单一预测，导致在欠指定输入下产生不稳定且不一致的输出。我们将结构化生成重新表述为一个歧义感知推理问题，其目标是对一组合理的结构化输出进行推理，而非预测单一解。我们证明标准训练本质上会抑制有效的替代方案，因此我们采用了一种多有效学习框架，在优化过程中保留歧义。为了将这一表述付诸实践，我们构建了一个结构化输出的候选空间，将每个候选表示为……

    arXiv:2610.31681v1 Announce Type: cross  Abstract: Structured generation is typically formulated as a deterministic mapping from input to a single structured output. This assumption is fundamentally misaligned with natural language, which often admits multiple structurally distinct yet semantically valid interpretations. As a result, existing methods rely on single-reference supervision that collapses ambiguity into one prediction, leading to unstable and inconsistent outputs under underspecified inputs. We reformulate structured generation as an \textit{ambiguity-aware inference problem}, where the objective is to reason over a set of plausible structured outputs rather than predict a single solution. We show that standard training inherently suppresses valid alternatives, and instead adopt a multi-valid learning framework that preserves ambiguity during optimization. To operationalize this formulation, we construct a candidate space of structured outputs, represent each candidate as 
    
[^188]: 验证比较本身，而不仅是评分器：为什么总体准确率会在交互式大语言模型评估中产生误导

    Validating the Comparison, Not Just the Scorer: Why Overall Accuracy Can Mislead in Interactive LLM Evaluation

    [https://arxiv.org/abs/2610.31678](https://arxiv.org/abs/2610.31678)

    该论文指出，仅用总体准确率验证自动评分器会掩盖针对特定提示的评分误差，从而可能扭曲交互式LLM评估中提示间比较的方向和幅度，因此应当验证“比较结果”本身而不仅是评分器。

    

    许多关于大语言模型的研究会比较模型对不同后续提示的响应方式。例如，一个提示可能以中性的方式要求模型重新考虑，而另一个提示则坚称一部虚构的电影确实存在。这种差异被用来衡量对话压力的影响。随后，自动评分器会标注每个回复是否做出了无事实依据的声明。研究者通常用一个总体准确率或 F₁ 分数来验证评分器的可靠性。然而，较高的总体性能可能掩盖针对特定提示的错误：如果评分器在某个提示之后犯下更多错误，那么测得的差异可能过大、过小，甚至方向完全相反。一个主要由困难样本组成的审计也可能歪曲在整个研究中真实的表现水平。我们推导了此类错误如何改变提示之间的比较结果，并使用六个模型、180 个虚构实体问题以及六种提示条件对这一问题进行了实证考察。在一个包含 288 个回复的均衡审计中，总体……

    arXiv:2610.31678v1 Announce Type: cross  Abstract: Many LLM studies compare how models respond to different follow-up prompts. For example, one prompt may neutrally ask a model to reconsider, while another insists that an invented film really exists. The difference is used to measure the effect of conversational pressure. An automatic scorer then labels whether each response makes an unsupported factual claim. Researchers commonly validate the scorer with one overall accuracy or $F_1$ score. However, high overall performance can hide prompt-specific mistakes: if the scorer makes more errors after one prompt, the measured difference can be too large, too small, or point in the wrong direction. An audit containing mostly difficult responses can also misrepresent performance in the full study. We derive how such errors alter a prompt comparison and examine the problem using six models, 180 invented-entity questions, and six prompt conditions. In a balanced audit of 288 responses, overall 
    
[^189]: 真值标注如何塑造检索得分：面向乳腺病变管理的检索增强生成系统的证据感知评估

    How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions

    [https://arxiv.org/abs/2610.31606](https://arxiv.org/abs/2610.31606)

    该论文提出了一种证据感知的检索评估协议，通过将证据片段映射到所有包含块来定义相关性，从而解决检索得分依赖真值标注覆盖范围的问题，并将其应用于乳腺病变管理文献临床RAG系统的评估。

    

    目的：描述一种针对基于乳腺病变管理文献的临床检索增强生成（RAG）系统的证据感知检索评估协议，以解决检索得分对真值标注覆盖范围的依赖问题。方法：使用包含18篇文档的语料库和18条查询的开发基准，每篇文档对应一个“是/否”问题或特殊疑问句问题。检索结合了稠密向量嵌入、BM25和交叉编码器重排序。评估了128至1024个词元的分块大小、0%至20%的重叠率以及四种检索器配置。相关性由映射到所有包含块上的证据片段来定义，将包含等价证据的重叠块视为相关。检索效果采用R-Precision、Recall@K、MRR和nDCG@K进行评估。此外，生成质量在512词元索引上以K=1至5进行评估，采用DeepEval声明级指标（以deepseek-v4-pro作为LLM评判者）以及BLEU/ROUGE指标。结果：所选的混合【摘要在此处不完整，原文截断】

    arXiv:2610.31606v1 Announce Type: new  Abstract: Purpose: To describe an evidence-aware retrieval evaluation protocol for a clinical retrieval-augmented generation (RAG) system over breast lesion management literature, addressing the dependence of retrieval scores on annotation coverage. Methods: An 18-document corpus and an 18-query development benchmark, with one Yes/No or WH-question per document, were used. Retrieval combined dense embeddings, BM25, and cross-encoder reranking. Chunk sizes of 128 to 1024 tokens, overlaps of 0 to 20%, and four retriever configurations were evaluated. Relevance was defined by evidence spans mapped to all containing chunks, treating overlapping chunks with equivalent evidence as relevant. Retrieval was evaluated using R-Precision, Recall@K, MRR, and nDCG@K. Separately, generation was assessed on a 512-token index at K=1 to 5 using DeepEval claim-level metrics, with deepseek-v4-pro as the LLM judge, and BLEU/ROUGE metrics. Results: The selected hybrid 
    
[^190]: SABLE：可扩展的仿射贝叶斯潜空间探索

    SABLE: Scalable Affine Bayesian Latent-space Exploration

    [https://arxiv.org/abs/2610.31495](https://arxiv.org/abs/2610.31495)

    SABLE 提出了一种贝叶斯仿射代理优化方法，通过将查询分解为由期望改进（EI）优化的低维跨度内分量和由标量范数支配的跨度外分量，在评估预算远小于潜空间维度的高维潜空间中实现了可扩展的贝叶斯优化。

    

    诸如工程设计、蛋白质等复杂对象往往难以优化，因为它们的设计空间呈现高维结构，且目标评估代价高昂。生成模型提供了连续的潜空间表示，但其潜空间可能具有数千至数万维，形成了潜空间维度 D 远超评估预算的情形。我们提出 SABLE，一种针对各向同性分布且预算有限的高维潜空间的贝叶斯优化方法。SABLE 使用贝叶斯仿射代理模型进行预测与采集，并将高阶结构视为代理模型的偏差。在 N 次观测之后，斜率更新的维度至多为 N-1，从而将每个查询分解为一个至多 N 维的跨度内分量（通过期望改进 EI 进行优化）和一个由标量范数支配的跨度外分量。我们表明……

    arXiv:2610.31495v1 Announce Type: new  Abstract: Complex objects such as engineering designs and proteins are often difficult to optimise because their design spaces exhibit high-dimensional structure and objective evaluations are expensive. Generative models provide continuous latent representations, but their latent spaces can have thousands to tens of thousands of dimensions, creating a regime where the latent dimension $D$ greatly exceeds the evaluation budget. We introduce SABLE, a Bayesian optimisation method for high-dimensional latent spaces with an isotropic distribution and limited budget. SABLE uses a Bayesian affine surrogate for prediction and acquisition, treating higher-order structure as surrogate discrepancy. After $N$ observations, the slope update has dimension at most $N-1$, yielding a decomposition of each query into an in-span component optimised by expected improvement (EI) in at most $N$ dimensions and an off-span component governed by a scalar norm. We show tha
    
[^191]: DeepOpen：面向决策的BANKING77与CLINC150意图分类

    DeepOpen: Decision-Oriented Intent Classification on BANKING77 and CLINC150

    [https://arxiv.org/abs/2610.31104](https://arxiv.org/abs/2610.31104)

    DeepOpen提出了针对BANKING77和CLINC150的面向决策意图分类框架，通过SAM续训的单分类器和多模型概率融合方法，分别在两个数据集上取得了94.12%和98.02%的高准确率，同时严谨地区分了头条分数与统计意义上的改进证据。

    

    我们提出了DeepOpen，一个面向决策的意图分类框架，针对BANKING77和CLINC150数据集分别设计了专用系统。在BANKING77上，单一英文分类器经过锐度感知最小化（SAM）续训后达到94.1234%的测试准确率（2,899/3,080），这是三个固定SAM随机种子中观察到的最佳结果，在测试评估后选出用于展示；其平均值为94.0368%。在CLINC150的150个域内类别上，一个Laya专家模型与三个DeBERTa-v3-large分类器的概率融合达到98.0222%（4,411/4,500）。我们描述了这两种配置，复现了随附评估浏览器中展示的参考对比，并将头条分数与改进证据区分开来。在统一的BANKING77评估路径下，SAM平均值比历史检查点高出0.0433个百分点，共享行的95%区间为[-0.2273, +0.3247]。因此，续训并未……（原文摘要在此处截断）

    arXiv:2610.31104v2 Announce Type: replace  Abstract: We present DeepOpen, a decision-Oriented intent-classification framework with dataset- specific systems for BANKING77 and CLINC150. On BANKING77, a single English classi- fier continued with sharpness-aware minimization (SAM) achieves 94.1234% test accuracy (2,899/3,080). This is the best observed result among three fixed SAM seeds, selected for display after test evaluation; their mean is 94.0368%. On the 150 in-scope CLINC150 classes, probability fusion of one Laya specialist and three DeBERTa-v3-large classifiers achieves 98.0222% (4,411/4,500). We describe the two configurations, reproduce the reference com- parisons displayed in the accompanying evaluation explorer, and separate headline scores from evidence about improvement. Under a uniform BANKING77 evaluation path, the SAM mean exceeds the historical checkpoint by 0.0433 percentage points, with a shared-row 95% interval of [-0.2273, +0.3247]. Thus, the continuation does not 
    
[^192]: 协递归语义自我扩展：开放式智能的数学基础

    Co-Recursive Semantic Self-Extension: A Mathematical Foundation for Open-Ended Intelligence

    [https://arxiv.org/abs/2610.31097](https://arxiv.org/abs/2610.31097)

    该论文借鉴康德与哥德尔思想，将认知状态建模为形式语言上的部分求值，提出“协递归语义自我扩展”的数学框架，通过对不可判定中间前提的反思性具体化，使智能系统能够突破固定目的与布尔封闭目的变换的语义边界，为开放式智能奠定数学基础。

    

    人工智能通过规模、数据和计算取得了长足进步，但仍然受限于其表示框架。我们探究是什么使智能具有开放性。每个推理系统都会选择有意义的区分，从而限制了它能够学习的内容。在固定的目的之下，性能会达到由其语义域设定的上限。目的论式的相继演进——即新目的从旧目的中产生——也面临类似的限制：布尔封闭的目的变换会穷尽其可到达的语义疆域。我们将视角从目的转向推理系统。借鉴康德和哥德尔的思想，我们将认知状态建模为对形式语言的部分求值。未解决的命题可能是未定的或不可判定的。推理可以解决前者，但无法解决后者，这为扩展留下了前沿。对不可判定命题的非平凡归约会暴露出不可判定的中间前提，而这些前提的反思性具体化支持多种……

    arXiv:2610.31097v2 Announce Type: replace  Abstract: Artificial intelligence has advanced through scale, data, and computation, yet remains bounded by its representational framework. We ask what makes intelligence open-ended. Every inferential system selects meaningful distinctions, limiting what it can learn. Within a fixed purpose, performance reaches a ceiling set by its semantic domain. Teleological succession, in which purposes arise from earlier ones, faces a similar limit: Boolean-closed purpose-transformations exhaust their reachable semantic territory. We shift from purposes to inferential systems. Drawing on Kant and G\"odel, we model cognitive states as partial evaluations over a formal language. Unresolved propositions may be undecided or undecidable. Reasoning can resolve the former but not the latter, leaving a frontier for extension. Non-trivial reductions of undecidable propositions expose undecidable intermediate premises whose reflective reification supports multiple 
    
[^193]: 主要是阅读：开卷评分在文档级LoRA适配器中究竟衡量了什么

    Mostly Reading: What Open-Book Scores Measure in Document-Level LoRA Adapters

    [https://arxiv.org/abs/2610.31017](https://arxiv.org/abs/2610.31017)

    开卷评分主要衡量的是所有适配器共有的“阅读”行为而非文档记忆——随机适配器即可恢复专用适配器82%的开卷优势，但在无源文本的完形测试中仅恢复约2%（与零无异），因此不宜用开卷评分来评估或选择文档级LoRA适配器。

    

    单文档LoRA适配器已被提出作为可合并与检索的知识单元；然而，开卷评分——即将文档放入提示词中——会同样容易地把功劳归给文档自己的适配器（其专用适配器）的“阅读”行为，而非真正“记住”该文档。我们在Llama-3.1-8B-Instruct上为150篇Wikipedia文章各训练一个适配器，发现开卷分数主要衡量的是每个适配器都共有的一种阅读行为：一个在其他文档上训练的随机适配器能够恢复专用适配器相对于基础模型82%的开卷优势，但在一个不提供源文本、直接询问同一文档事实的完形填空探测中，只能恢复其2%的优势——这一比例与零无法区分。文档内容确实写入了权重，因为专用适配器在完形回忆上比基础模型高出0.094，且其更新的几何结构遵循其所阅读的文档；然而，在一项包含30个查询的测试中，通过权重空间邻近性选择的适配器……（原文在此处截断）

    arXiv:2610.31017v1 Announce Type: new  Abstract: Single-document LoRA adapters have been proposed as units of knowledge to be merged and retrieved; yet open-book scoring, with the document in the prompt, can credit a document's own adapter (its specialist) for reading that document as readily as for holding it. Training one adapter for each of 150 Wikipedia articles on Llama-3.1-8B-Instruct, we find that open-book scores mostly measure a reading behaviour every adapter shares: a random adapter trained on another document recovers 82 % of the specialist's open-book advantage over the base model but only 2 % of its advantage on a cloze probe that asks for the same document's facts without providing the source text, a share we cannot distinguish from zero. Content does reach the weights, since the specialists beat the base model in cloze recall by 0.094, and the geometry of their updates follows the documents they read; yet, in a test of 30 queries, adapters chosen by weight-space proximi
    
[^194]: 缓解显著性坍塌：鲁棒的显著性感知长文本图文对齐

    Mitigating Saliency Collapse: Robust Saliency-Aware Long-Text Image-Text Alignment

    [https://arxiv.org/abs/2610.31005](https://arxiv.org/abs/2610.31005)

    提出RoSA显著性感知微调框架，通过将图文对分解为互补的显著视图与上下文视图并进行联合对齐学习，有效缓解长文本图文对齐中的显著性坍塌问题。

    

    长文本图文对齐对于细粒度场景理解至关重要，然而现有方法主要专注于扩展上下文长度或改进局部对齐。我们发现了一种被忽视的失败模式——显著性坍塌，即模型过度依赖具有视觉显著性的文本，而未能充分利用非显著上下文。因此，当显著内容被减少时，即使非显著上下文仍然包含有效信息，检索性能也会急剧下降。为解决这一问题，我们提出了RoSA，一个显著性感知的微调框架，它将图文对分解为互补的显著视图和上下文视图，从而实现对显著语义与上下文语义的联合学习。我们提出了一种与模态无关的分解策略，将对象级视觉显著性估计与基于大语言模型（LLM）的文本分解相结合，以桥接视觉与语言。我们采用辅助目标来显式对齐这些视图，确保模型的鲁棒性。

    arXiv:2610.31005v2 Announce Type: replace  Abstract: Long-text image-text alignment is essential for fine-grained scene understanding, yet existing methods mainly focus on extending context length or improving local alignment. We identify an overlooked failure mode, saliency collapse, where models over-rely on text that features visual prominence while underutilizing non-salient context. Consequently, retrieval performance degrades sharply when salient content is reduced, despite that the non-salient context remains informative. To address this, we propose RoSA, a saliency-aware finetuning framework that decomposes image-text pairs into complementary salient and contextual views, enabling joint learning of prominent and contextual semantics. We propose a modality-independent decomposition strategy to combine object-level visual saliency estimation with LLM-based text decomposition to bridge vision and language. We employ auxiliary objectives to explicitly align these views, ensuring ro
    
[^195]: OpenAttention：用尾部代理，百分之一足矣

    OpenAttention: One Percent Is Enough with Tail Proxies

    [https://arxiv.org/abs/2610.30951](https://arxiv.org/abs/2610.30951)

    OpenAttention通过将整个未读尾部压缩为单个计数感知均值代理token，在读取率低至1%时仍能获得比丢弃或采样更低误差的稀疏注意力输出。

    

    arXiv:2610.30951v1 公告类型：新论文 摘要：稀疏注意力只读取键值缓存中的一小部分，而如何处理未被读取的尾部仍然是一个关键挑战。直接丢弃尾部会使输出产生偏差，而以往方法采用的尾部采样策略在读取率为5%及以下时会失效，因为此时尾部很可能包含选择器遗漏的重要token。我们的关键观察是，最简单的代理形式——一个将整个尾部折叠为其计数感知均值的token——在实用选择器下的每个读取率上，都能产生比采样更低的输出误差。我们进一步以闭式形式证明，在低读取率下，采样带来的权重误差几乎是丢弃的两倍，而单个代理的误差则始终低于丢弃。基于这些发现，我们探索用少量针对尾部的代理来近似稠密注意力分布。我们提出了OpenAttention，这是一个由聚类、ErrorGate和top-k选择加上计数感知均值代理组成的流水线。为了高效地实现聚类和选择……

    arXiv:2610.30951v1 Announce Type: new  Abstract: Sparse attention reads only a small subset of the key-value cache, and how to treat the unread tail remains a key challenge. Discarding the tail biases the output, and sampling it as previous approaches do fails at reads of 5% and below, where the tail likely holds heavy hitters the selector missed. Our key observation is that the simplest possible proxy, one token that folds the entire tail into its count-aware mean, already yields lower output error than a sample at every read behind a practical selector. We further show in closed form that at low reads sampling nearly doubles the weight error of discarding while a single proxy stays below it. Building on these findings, we explore approximating the dense attention distribution with a few proxies that target the tail. We propose OpenAttention, a pipeline of clustering, ErrorGate and top-k selection followed by a count-aware mean proxy. To implement clustering and selection both efficie
    
[^196]: 从下一帧预测中学习：迈向统一且可扩展的视觉世界建模

    Learning from Next-Frame Prediction: Towards Unified and Scalable Visual World Modeling

    [https://arxiv.org/abs/2610.30724](https://arxiv.org/abs/2610.30724)

    提出VANE自监督视觉预训练框架，通过在冻结语义分词器提供的固定嵌入空间中预测整个下一帧的表示，将时间演化作为核心学习信号，实现统一且可扩展的视觉世界建模。

    

    视频提供了一种天然的监督来源：接下来发生的事情可以作为视觉世界建模的学习信号。然而，掩码视频预训练侧重于从周围上下文中恢复缺失内容，而非根据过去预测未来。自回归方法则逐个token向前预测，但在表示质量上尚未能媲美联合嵌入方法。这一差距促使我们重新审视未来预测的粒度与空间。在这项工作中，我们提出了VANE，一个建立在嵌入空间中下一帧预测之上的自监督视觉预训练框架。因果编码器总结过去的观测信息，供状态预测器推断整个下一帧的表示，使时间演化成为主要的学习信号。该预测在一个由冻结的语义分词器提供的固定嵌入空间中进行，强调场景内容与结构。

    arXiv:2610.30724v2 Announce Type: replace-cross  Abstract: Video provides a natural source of supervision: what happens next serves as a learning signal for visual world modeling. Yet masked video pretraining focuses on recovering missing content from surrounding context rather than predicting the future from the past. Autoregressive methods instead predict forward one token at a time, but have yet to match joint-embedding methods in representation quality. This gap motivates us to revisit the granularity and space of future prediction. In this work, we introduce VANE, a self-supervised visual pretraining framework built on next-frame prediction in embedding space. A causal encoder summarizes past observations for a state predictor that infers the representation of the entire next frame, making temporal evolution the primary learning signal. This prediction is carried out in a fixed embedding space provided by a frozen semantic tokenizer, emphasizing scene content and structure. To ret
    
[^197]: 安全探针的缩放定律

    Scaling Laws for Safety Probes

    [https://arxiv.org/abs/2610.30678](https://arxiv.org/abs/2610.30678)

    本文通过构建六个新安全数据集、在20个基础模型上以不同标签规模训练探针，首次提出了可依据模型规模和标签数量提前预测安全探针性能的缩放定律。

    

    安全探针是在模型激活值上训练的、用于实时监控语言模型的标准工具。每当发布新模型或需要监控新的危害领域时，探针都必须重新训练，这需要新的标签以及探针架构的选择。然而，无论是模型还是标签都无法预先告诉我们最终探针的性能如何，要弄清这一点需要训练并比较多种探针配置。在本文中，我们探讨是否可以仅凭模型规模和标签数量来提前预测探针的性能。现有的安全数据集不足以支持这一研究，因为它们要么规模有限，要么已经饱和。因此，我们构建了六个规模足够大、难度足够高的新安全数据集。在这些数据集上，我们在来自五个模型家族的20个基础模型上训练了两种探针架构，每个模型使用32到10⁶个标签。我们使用探针误差来衡量性能……

    arXiv:2610.30678v1 Announce Type: cross  Abstract: Safety probes trained on a model's activations are a standard tool for monitoring language models in real time. Each time a new model is released or a new harm area needs to be monitored, the probe must be retrained, which requires new labels and a choice of probe architecture. Yet neither the model nor the labels tell us in advance how well the resulting probe will perform. Finding out requires training and comparing many probe configurations. In this paper, we ask whether probe performance can instead be predicted in advance from model size and the number of labels. Existing safety datasets are insufficient for this, as they are either limited in scale or already saturated. We therefore construct six new safety datasets of sufficient size and difficulty. On these datasets, we train two probe architectures on 20 base models from five model families, with 32 to $10^{6}$ labels for each model. We measure performance using probe error, d
    
[^198]: 重新思考SAE评估：原子可解释单元框架揭示隐藏的多义性与冗余性

    Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy

    [https://arxiv.org/abs/2610.30474](https://arxiv.org/abs/2610.30474)

    本文提出了一种基于原子可解释单元的SAE评估新框架，将超越单义性的评估目标形式化，并借此揭示了现有SAE特征中隐藏的多义性与冗余性。

    

    稀疏自编码器（SAE）最初是为发现单义特征而提出的，单义性是刻画单个特征的性质，但并未规定学习到的特征集合整体上应该是什么样子。随着SAE越来越多地被应用于归因分析、回路发现和干预，研究者们开始使用描述特征集合全局性质的评估指标，例如稀疏性、死特征比例、重建保真度和解耦性等，这一系列指标已远远超出单义性的范畴。然而，这种超越单义性的隐式转变尚未被形式化为一个清晰、明确定义的、先于这些评估指标存在的目标。缺少这样的目标，现有指标仍然是临时拼凑的，其必要性缺乏依据。在这项工作中，我们通过为SAE评估引入一种新范式来形式化这一最终目标：要求SAE特征，以及更广义的字典……（摘要不完整）

    arXiv:2610.30474v1 Announce Type: cross  Abstract: Sparse autoencoders (SAEs) were originally motivated by discovering monosemantic features, a property that characterizes individual features but does not specify what the learned feature set should look like as a whole. As SAEs have increasingly been adopted for attribution, circuit discovery, and intervention, researchers have begun evaluating metrics that describe global properties of the feature set, such as sparsity, dead feature rates, reconstruction fidelity, and disentanglement, a diverse set that extends far beyond monosemanticity. However, this implicit shift beyond monosemanticity has not been formalized as a clear, well-defined objective that comes before these evaluation metrics. Without such an objective, existing metrics remain ad hoc, with their necessity unjustified. In this work, we formalize this ultimate objective by introducing a new paradigm for SAE evaluation: requiring that SAE features, and more generally dictio
    
[^199]: Astro2Method：评测大语言模型在天文学中机器学习方法选择能力的基准

    Astro2Method: Benchmarking LLMs on Machine Learning Method Selection in Astronomy

    [https://arxiv.org/abs/2610.30282](https://arxiv.org/abs/2610.30282)

    该论文提出了Astro2Method这一专家审阅的基准，用于评测大语言模型在天文学中为科学问题选择机器学习方法的能力，发现LLM在简单问题上表现良好但在困难问题上准确率显著下降，且大多数检索增强和智能体方法无法带来一致的性能提升。

    

    机器学习（ML）在天文学中的应用日益增多，但为科学问题选择合适的方法通常需要同时具备天文学和机器学习两方面的专业知识。大语言模型（LLM）可以帮助天文学家做出此类选择，但这一能力在现有基准测试中仍未得到充分探索。我们提出了Astro2Method，这是一个包含500道经专家审阅的二分类问题的基准，这些问题来自346篇天文学论文（2024年1月至2026年6月），基于作者给出的明确性能差异以及完整（简单）或部分（困难）的解释。在五个大语言模型上，直接提示在简单问题上达到87.50%-92.55%的准确率，但在困难问题上仅为58.87%-64.52%。在使用历史论文的情况下，检索增强生成（RAG）以及大多数带有额外推理或检索功能的智能体方法（包括ReAct、Self-Ask和CRITIC）均未能带来一致的改进。需要搜索网络的智能体主要在找到问题所基于的论文时才能提高准确率。

    arXiv:2610.30282v1 Announce Type: cross  Abstract: Machine learning (ML) is increasingly used in astronomy, but choosing a suitable method for a scientific problem often requires expertise in both astronomy and ML. Large language models (LLMs) could help astronomers make such choices, yet this ability remains underexplored in existing benchmarks. We introduce Astro2Method, a benchmark of 500 expert-reviewed binary questions from 346 astronomy papers (2024/01-2026/06), based on clear performance differences and complete (Easy) or partial (Hard) explanations from the authors. Across five LLMs, direct prompting achieves 87.50-92.55% accuracy on Easy questions but only 58.87-64.52% on Hard. Using historical papers, neither retrieval-augmented generation nor most agent methods with additional reasoning or retrieval (including ReAct, Self-Ask, and CRITIC) bring consistent improvement. Agents required to search the web improve accuracy mainly when they find the papers the questions are based 
    
[^200]: 面向真实CFD重构的复杂域上时间分数阶Navier-Stokes方程的GeoDiv KAN求解器

    GeoDiv KAN Solver for Time Fractional Navier Stokes Equations on Complex Domains to Realistic CFD Reconstruction

    [https://arxiv.org/abs/2610.30179](https://arxiv.org/abs/2610.30179)

    本文提出一种几何门控且从构造上保证无散度的高斯径向基函数Kolmogorov-Arnold网络（GeoDiv KAN），用于在方形、圆形和星形等复杂域上求解时间分数阶不可压缩Navier-Stokes方程，实现真实的CFD重构。

    

    arXiv:2610.30179v2 公告类型： replace-cross 摘要： 科学机器学习为解决科学与工程中的复杂问题提供了一个有前景的框架。值得注意的是，物理信息神经网络（PINNs）在求解微分方程和建模工程系统方面已展现出巨大潜力。由于时间非局部性、非线性流动动力学、几何复杂性和不可压缩性约束，在复杂域上求解时间分数阶Navier-Stokes方程（TFNSEs）极具挑战性。为应对这些挑战，我们提出了一种几何门控、无散度的高斯径向基函数Kolmogorov-Arnold网络（GeoDiv KAN）。所提出的架构利用依赖于几何的特征门控和流函数表述，从构造上保证了不可压缩性。GeoDiv KAN在方形、圆形和星形域上对时间分数阶不可压缩Navier-Stokes方程进行了评估。数值实验表明，GeoDiv KAN实……

    arXiv:2610.30179v2 Announce Type: replace-cross  Abstract: Scientific machine learning provides a promising framework for solving complex problems in science and engineering. Notably, physics informed neural networks (PINNs) have shown considerable potential for solving differential equations and modelling engineering systems. Solving time fractional Navier Stokes equations (TFNSEs) on complex domains is challenging due to temporal nonlocality, nonlinear flow dynamics, geometric complexity, and incompressibility constraints. To address these challenges, we propose a geometry gated, divergence-free Gaussian radial basis function Kolmogorov Arnold network (GeoDiv KAN). The proposed architecture uses geometry dependent feature gating and a streamfunction formulation to preserve incompressibility by construction. GeoDiv KAN is evaluated for the time fractional incompressible Navier-Stokes equations on square, circular, and star-shaped domains. Numerical experiments show that GeoDiv KAN ach
    
[^201]: 基于AI代理建模框架加速加拿大全国野火危害制图

    Accelerating Wildfire Hazard Mapping Across Canada with an AI Surrogate Modelling Framework

    [https://arxiv.org/abs/2610.30170](https://arxiv.org/abs/2610.30170)

    该研究提出了一套标准化的AI代理模型训练与评估框架，以及在国家尺度数据上训练的多任务深度学习代理模型，可将原本需要数天至数周的加拿大野火危害制图过程模拟大幅加速，支撑全国性的主动野火风险减缓决策。

    

    加拿大日益活跃的野火活动需要频繁开展全国范围的危害制图，以指导主动减缓措施，包括燃料管理、土地利用规划、基础设施保护和疏散路线规划。然而，基于过程的野火模拟框架（如BurnP3和BurnP3+）依赖计算密集型的蒙特卡洛模拟来建模燃烧概率、火灾强度和蔓延速度，这些工作流程处理单个区域建模单元可能需要数天到数周时间。为克服这些计算瓶颈，我们通过AI研究人员与野火建模专家的合作提出了三项贡献：(a) 一个标准化框架，用于针对基于过程的野火模拟器对AI代理模型进行训练、微调和评估；(b) 一个在国家尺度数据上训练的多任务深度学习代理模型，用于模拟燃烧概率、平均火灾强度和平均蔓延速度……

    arXiv:2610.30170v2 Announce Type: replace  Abstract: Canada's increasing wildfire activity requires frequent national hazard mapping to guide proactive mitigation, including fuel management, land-use planning, infrastructure protection, and evacuation routing. However, process-based wildfire simulation frameworks (e.g., BurnP3 and BurnP3+) rely on computationally intensive Monte Carlo simulations to model burn probability, fire intensity, and rate of spread. These workflows can take days to weeks to process a single regional modelling unit. To overcome these computational bottlenecks, we present three contributions developed through collaboration between AI researchers and wildfire modelling experts: (a) a standardized framework for training, fine-tuning, and evaluating AI surrogate models against process-based wildfire simulators; (b) a multi-task deep learning surrogate trained on national-scale data to emulate burn probability, mean fire intensity, and mean rate of spread at a fract
    
[^202]: 仿生听觉处理：耦合并行共振-发放动力学与三值循环网络用于语音识别

    Bio-Inspired Auditory Processing: Coupling Parallel Resonate-and-Fire Dynamics with Ternary Recurrent Networks for Speech Recognition

    [https://arxiv.org/abs/2610.29720](https://arxiv.org/abs/2610.29720)

    提出脉冲三值GRU（stGRU），用三值权重门控循环单元进行时间滤波、由共振-发放神经元生成二值脉冲流驱动事件通路，将二值事件传输内化为训练原生约束，从而提升SNN在语音识别等连续上下文任务中的性能与能效。

    

    脉冲神经网络（SNN）通过稀疏的事件驱动计算有望实现高能效推理，但在自动语音识别（ASR）这类连续、依赖上下文的任务中仍受到限制。一个关键瓶颈是泄漏积分发放（LIF）神经元：其膜电位衰减会在停顿和语音边界处抹除信息，而它同时既要提取特征又要保留长程上下文。受大脑皮层将非线性滤波与胞体整合相分离的机制启发，我们提出了脉冲三值GRU（stGRU）：由三值权重的门控循环单元执行时间滤波，而下游的共振-发放（RF）神经元生成二值脉冲流，驱动下一个模块的事件通路。这使得二值事件传输成为原生的训练约束，而非部署阶段单独的激活量化步骤；结合三值权重，学习到的投影避免了密集的浮点……

    arXiv:2610.29720v1 Announce Type: cross  Abstract: Spiking Neural Networks (SNNs) promise energy-efficient inference through sparse, event-driven computation, yet remain constrained on continuous, context-dependent tasks such as automatic speech recognition (ASR). A key bottleneck is the Leaky Integrate-and-Fire (LIF) neuron, whose membrane decay erases information across pauses and phonetic boundaries while it must both extract features and retain long-range context. Inspired by cortical separation of nonlinear filtering and somatic integration, we introduce the Spiking Ternary GRU (stGRU): a ternary-weight gated recurrent unit performs temporal filtering, while downstream Resonate-and-Fire (RF) neurons generate the binary spike stream driving the next block's event pathway. This makes binary event transmission a native training constraint rather than a separate deployment-time activation quantization step; together with ternary weights, learned projections avoid dense floating-point 
    
[^203]: 用于神经网络微调、快速排序和嵌入维度降低的潜在空间配置

    Latent space configuration for neural network finetuning, fast ranking, and embedding dimensionality reduction

    [https://arxiv.org/abs/2610.29295](https://arxiv.org/abs/2610.29295)

    该论文提出潜在空间配置（LSC）微调方法，使神经网络嵌入具有结构化几何属性，从而支持O(1)复杂度的候选子集搜索，相比穷举搜索在GPU上提速高达3倍、CPU上提速40%，同时还能降低嵌入维度。

    

    在大规模神经网络嵌入库中为查询找到K个最相似的元素，在计算机科学的各个领域以及人工智能的工业应用中至关重要。本文提出了针对神经网络结构化潜在空间（LS）的搜索算法，借助O(1)复杂度的小候选子集搜索，能够加速Top K项目的检索。我们使用潜在空间配置（LSC）方法对ArcFace预训练的IResNet50模型进行微调，以获得具有期望几何属性的新嵌入分布。将所提出的算法应用于微调模型得到的嵌入，相比基线穷举搜索可实现数倍的加速。具体而言，与基线方法的GPU速度相比，我们报告了GPU上高达3倍的加速以及CPU上40%的加速。这些结果是在原始和缩减的潜在空间维度下获得的。

    arXiv:2610.29295v1 Announce Type: cross  Abstract: Finding K most similar elements for a query in a large gallery of neural network (NN) embeddings is essential in various areas of computer science and industrial applications of artificial intelligence. This paper proposes search algorithms for structured latent spaces (LS) of NNs which allow to speed up top K item retrieval owing to O(1) complexity small candidate gallery subset search. ArcFace-pretrained IResNet50 models are finetuned using Latent Space Configuration (LSC) methodology to obtain new distribution of embeddings with desired geometric properties. Applying the proposed algorithms to embeddings obtained from finetuned models leads to several times speed-up over baseline exhaustive search. Specifically, we report up to 3 times speed-up on GPU and 40% speed-up on CPU compared to the baseline method GPU speed. These results are obtained for the original and reduced numbers of LS dimensions. The results show that LS dimensiona
    
[^204]: 一种由Z数不确定性驱动的环保旅游行程规划器：来自巴库、迪拜、阿布扎比、塔什干和伊斯坦布尔的证据

    An Eco-Friendly Tourism Trip Planner Driven by Z-Number Uncertainty: Evidence from Baku, Dubai, Abu Dhabi, Tashkent, and Istanbul

    [https://arxiv.org/abs/2610.29120](https://arxiv.org/abs/2610.29120)

    该论文提出一种基于Z数的环保旅游行程规划器，直接从路径服务观测和平台评论数构建行程时间与评分的模糊不确定性表示，结合混合损失函数与自适应大邻域搜索，在五个城市中生成带可靠性标注的低碳单日行程。

    

    已部署的旅游行程规划器通常将行程时间和评分作为精确数字处理，然而路径规划服务返回的行程时间会随出发时段而变化，且平台评分可能基于少数几条评论或数万条评论。我们提出了一种环保行程规划器，将这两种量表示为Z数——即将对某个值的模糊限制与其背后证据的模糊可靠性配对——并直接从六个出发时段的路径服务观测数据和平台评论数量构建Z数，而非通过语言启发式方法获取。该系统采用混合损失函数，综合了评分、碳排放、行程时间和可靠性等因素，并通过对话式前端，利用自适应大邻域搜索算法生成带有每段行程可靠性标注的单日行程。我们在巴库、迪拜、阿布扎比、塔什干和伊斯坦布尔五个城市，在相同优化器条件下将该规划器与清晰均值基线进行了比较，实验采用十五……

    arXiv:2610.29120v1 Announce Type: cross  Abstract: Deployed tourism trip planners ingest travel times and ratings as exact numbers, although routing services return travel times that vary with the hour of departure and a platform rating may rest on a handful of reviews or on tens of thousands. We present an eco-friendly trip planner that represents both quantities as Z-numbers, which pair a fuzzy restriction on a value with a fuzzy reliability on the evidence behind it, and that constructs them directly from routing-service observations at six departure hours and from platform review counts rather than from linguistic elicitation. A hybrid loss combines score, carbon, travel-time, and reliability terms, and an adaptive large neighborhood search returns single-day itineraries with per-leg reliability annotations through a conversational front end. We compare the planner with a crisp-mean baseline under an identical optimizer in Baku, Dubai, Abu Dhabi, Tashkent, and Istanbul, with fiftee
    
[^205]: GenXR-Bench：面向交互式XR开发的编码智能体基准测试

    GenXR-Bench: Benchmarking Coding Agents for Interactive XR Development

    [https://arxiv.org/abs/2610.29014](https://arxiv.org/abs/2610.29014)

    该论文提出了首个用于端到端评估编码智能体XR软件开发能力的基准GenXR-Bench，通过112个专家编写任务以及结合脚本化/自适应交互与程序化/视觉断言的测试方式，有效评估核心功能和运行时行为。

    

    编码智能体正迅速从单文件生成发展为长时程的仓库级工程，但对其编写交互式扩展现实（XR）应用能力的评估仍然是一个开放性挑战。与传统软件不同，XR应用运行于动态的3D物理环境中，其运行时鲁棒性同时取决于持续的用户动作、空间输入和环境感知。我们提出了GenXR-Bench，这是首个用于编码智能体在XR软件开发中进行端到端评估的基准。GenXR-Bench包含112个由专家编写的任务，涵盖24个功能实现、21个漏洞修复和67个应用生成任务。保留测试集结合了脚本化与自适应交互，以及程序化断言和视觉断言，以评估核心功能和运行时行为。针对124个候选应用的盲测专家判断，GenXR-Bench的测试套件达到了96.5%的准确率。

    arXiv:2610.29014v1 Announce Type: new  Abstract: Coding agents are rapidly advancing from single-file generation to long-horizon repository-scale engineering, yet the evaluation of their capabilities to author interactive extended reality (XR) applications remains an open challenge. Unlike traditional software, XR applications operate within dynamic 3D physical environments where runtime robustness depends jointly on continuous user motion, spatial input, and environmental sensing. We introduce GenXR-Bench, the first benchmark for the end-to-end evaluation of coding agents in XR software development. GenXR-Bench contains 112 expert-authored tasks spanning 24 feature implementations, 21 bug repairs, and 67 application generation tasks. Held-out tests combine scripted and adaptive interactions with programmatic and visual assertions to evaluate core functionality and runtime behavior. GenXR-Bench's testing suite achieves 96.5% accuracy against blinded expert judgments of 124 candidate ap
    
[^206]: SANE：面向LLM评判器的自动化的评分器特定变异测试

    SANE: Automated Scorer-Specific Metamorphic Testing for LLM Judges

    [https://arxiv.org/abs/2610.28886](https://arxiv.org/abs/2610.28886)

    SANE是一个基于智能体的自动化框架，能针对特定LLM评判器自动识别其评分能力、生成变异测试配方及预期分数关系，从而将评判器可靠性测试实现全流程自动化，摆脱了人工编写扰动配方和预设评分器类型的限制。

    

    LLM评判器（LLM judges）在基准评估、模型选择、数据筛选和模型训练中扮演着越来越重要的角色。只有当每个评判器能够对影响其评分标准的变动做出可靠响应时，其决策才有价值。现有的可靠性测试方法通过对输入施加具有已知预期评分影响的扰动来进行测试。主流的LLM评判器测试框架将人工编写的扰动配方与合成测试生成相结合，但将针对特定评分器的配方构思排除在自动化流程之外。这限制了它们只能使用固定的扰动方式和预先假设的评分器类型。我们提出了SANE，一个基于智能体（agent）的框架，它将随机评估器的行为需求编码为可复用的智能体指令。给定评分器规范、实现代码和代表性输入，该流水线首先识别评分器特定的能力，然后为每种能力自动推导扰动配方和预期的分数关系。SANE还定义了一个（摘要内容在此处被截断）

    arXiv:2610.28886v1 Announce Type: new  Abstract: LLM judges increasingly mediate benchmark evaluation, model selection, data curation, and training. Their decisions are useful only when each judge responds reliably to changes affecting its scoring criterion. Existing methods for reliability testing perturb inputs with known expected effects on judge scores. Prominent LLM-judge testing harnesses combine human-authored perturbation recipes with synthetic test generation, leaving scorer-specific recipe ideation outside the automated pipeline. This limits their use to fixed perturbations and pre-assumed scorer types. We introduce SANE, an agent-based framework that encodes behavioral requirements for stochastic evaluators as reusable agent instructions. Given a scorer specification, implementation, and representative inputs, the pipeline first identifies scorer-specific capabilities, then automatically derives perturbation recipes and expected score relations for each. SANE also defines a 
    
[^207]: S²-MAS：一个拥有5亿家庭角色智能体的社会规模多智能体仿真框架

    S$^2$-MAS: A Society-Scale Multi-Agent Simulation Framework with 500 Million Household Persona Agents

    [https://arxiv.org/abs/2610.27746](https://arxiv.org/abs/2610.27746)

    该论文提出了S²-MAS社会规模多智能体仿真框架，基于真实调查数据构建5亿个家庭角色智能体，并配备专用紧凑决策模型FinL-mlx，能够在大规模仿真中刻画家庭金融决策及信用额度等政策变化对消费行为的差异化影响。

    

    在社会尺度上模拟经济反应，需要智能体的决策既能反映家庭自身状况，又能反映先前行动带来的影响。我们提出了S²-MAS，这是一个社会规模的多智能体仿真框架，构建于基于真实调查数据的5亿个合成家庭角色（persona）之上。其Genese引擎将决策与状态更新相分离，同时通过可复用的金融场景与应用账本来追踪消费、现金、债务和支付情况。FinL-mlx是一个紧凑的决策模型，使用SFT + RLCD在293万个样本上训练而成，在消费者金融选择场景中达到74.55%的准确率和38.37%的问题宏平均F1值，在两项指标上均优于Jev 1.13和GLM-5.3-Flash。我们评估了八个金融任务族，并在配对的购买融资条件下模拟了1200万个不同的家庭智能体。实验表明，同时提高购买目标和获批信用额度会增加消费，且不同年龄段的反应存在差异……

    arXiv:2610.27746v1 Announce Type: cross  Abstract: Simulating economic responses at society scale requires agents whose decisions reflect household circumstances and the effects of earlier actions. We introduce S$^2$-MAS, a society-scale multi-agent simulation framework built on 500 million synthetic household personas grounded in real-survey data. Its Genese engine separates decisions from state updates, while reusable financial scenarios and application ledgers track consumption, cash, debt and payments. FinL-mlx, a compact decision model trained with SFT + RLCD on 2.93 million examples, achieves 74.55% accuracy and 38.37% question-macro F1 in consumer financial-choice scenarios, outperforming Jev 1.13 and GLM-5.3-Flash on both measures. We evaluate eight financial task families and simulate 12 million distinct household agents in paired purchase-funding conditions. Jointly raising purchase goals and approved credit limits increases consumption, with responses differing across age an
    
[^208]: RLSplat：面向前馈式3D高斯泼溅的资源线性预算控制

    RLSplat: Resource-Linear Budget Control for Feed-forward 3D Gaussian Splatting

    [https://arxiv.org/abs/2610.27712](https://arxiv.org/abs/2610.27712)

    RLSplat提出了一种资源线性的预算可控前馈式3D高斯泼溅框架，摒弃了先预测稠密高斯再剪枝或选择的做法，使表示大小、推理延迟和峰值内存随高斯预算的减少而线性下降。

    

    前馈式3D高斯泼溅（FF3DGS）通过单次前向传播即可从图像重建3D场景，因此非常适合必须在有限资源预算下运行的应用。为了进一步提升其实用性，近期的预算可控FF3DGS方法允许在推理阶段调整高斯的数量，使输出满足给定的存储上限或渲染延迟约束。然而，这些方法忽略了推理过程本身的成本：为了生成具有目标高斯数量的表示，它们需要先预测一组稠密的高斯，再从中进行剪枝或选择，导致无论预算大小如何，推理延迟和峰值内存都几乎保持恒定。为此，我们提出了RLSplat，一个资源线性的预算可控FF3DGS框架，其表示大小、推理延迟和峰值内存均随高斯预算的减少而线性下降。不同于剪枝或选择的方式，RLSplat学习一种高斯……

    arXiv:2610.27712v1 Announce Type: cross  Abstract: Feed-forward 3D Gaussian Splatting (FF3DGS) reconstructs 3D scenes from images in a single forward pass, making it well-suited to applications that must operate within limited resource budgets. To further improve its practicality, recent budget-controllable FF3DGS methods allow the number of Gaussians to be adjusted at inference, so that the output meets a given storage limit or rendering-latency constraint. However, these methods overlook the cost of inference itself: to produce a representation with the target number of Gaussians, they first predict a dense set of Gaussians and then prune or select from it, requiring nearly constant latency and peak memory regardless of the budget. We therefore propose RLSplat, a resource-linear budget-controllable FF3DGS framework whose representation size, inference latency, and peak memory all decrease linearly as the Gaussian budget is reduced. Instead of pruning or selection, RLSplat learns a Ga
    
[^209]: EASformer：面向工业生成式排序的环境-动作-状态对齐

    EASformer: Environment-Action-State Alignment for Industrial Generative Ranking

    [https://arxiv.org/abs/2610.27691](https://arxiv.org/abs/2610.27691)

    提出EASformer架构，通过将环境、动作与状态语义解耦并对齐序列与非序列特征之间的不对称语义，解决生成式排序中query-key点积比较异质语义对象、跨token交互受限的问题。

    

    现代工业排序系统正日益采用生成式推荐（GR）范式，其中用户行为序列与非序列（NS）特征被标记化后由Transformer风格的骨干网络联合建模。这种统一的形式化为长历史与异构特征的建模提供了可扩展的接口。然而，序列表示与NS表示往往携带不对称的语义：历史表示可能将环境上下文与已观测动作的语义混合在一起，而候选侧的NS表示虽不含已观测动作，却包含更丰富的补充上下文，例如用户-物品交互特征和近期行为列表。这类表示之间的直接交互会使query-key点积在比较不同的语义对象，从而限制了有意义的跨token交互。为填补这一空白，我们提出了EASformer，一种面向工业生成式排序的环境-动作-状态解耦架构……

    arXiv:2610.27691v1 Announce Type: new  Abstract: Modern industrial ranking systems increasingly adopt generative recommendation (GR) paradigms, in which user behavior sequences and non-sequential (NS) features are tokenized and jointly modeled by Transformer-style backbones. This unified formulation provides a scalable interface for modeling long histories and heterogeneous features. However, sequence and NS representations often carry asymmetric semantics: historical representations may combine environment context with observed-action semantics, whereas candidate-side NS representations contain no observed action but incorporate richer supplementary context, such as user-item interaction features and recent behavior lists. Direct interaction between such representations can make query-key dot products compare different semantic objects, limiting meaningful cross-token interaction. To fill this gap, we propose EASformer, an Environment-Action-State disentangled architecture for industr
    
[^210]: MLIPS：多层索引器池共享

    MLIPS: Multi-Layer Indexer Pool Sharing

    [https://arxiv.org/abs/2610.27438](https://arxiv.org/abs/2610.27438)

    MLIPS是一种无需训练的方法，通过让一个生产者层发布超出预算的top-M候选池供后续多个层共享复用，在保留各层独立打分与选择的同时消除重复的候选搜索，在DeepSeek-V4-Flash上实现了1.14–1.40倍加速和14.8–27.4%的GPU能耗降低，且准确率几乎不受影响。

    

    稀疏注意力限制了每个查询所使用的键的数量，但寻找这些键仍可能需要在每一层对完整前缀进行扫描。我们提出了MLIPS（多层索引器池共享），这是一种无需训练的方法，它在共享候选搜索的同时，保留每个使用方自己的打分与选择。生产者会发布一个大于注意力预算K的top-M候选池；后续各层从该池中选出自己的top-K。在三个稀疏注意力模型家族上的测量表明，宽泛候选池所提供的覆盖率显著高于最终选择之间的重叠程度。我们在vLLM中为DeepSeek-V4-Flash实现了逐层共享、持久解码复用以及粗粒度生产者。在四个长上下文测试套件上，每个生产者对应三个使用层的设置可带来1.14–1.40倍的加速和14.8–27.4%的GPU能耗降低，单次运行中观察到的准确率变化为−1.49至+0.29个百分点。更广泛的共享能够…

    arXiv:2610.27438v1 Announce Type: cross  Abstract: Sparse attention limits the keys used by each query, but finding those keys can still require a full-prefix scan at every layer. We introduce \textbf{MLIPS (Multi-Layer Indexer Pool Sharing)}, a training-free method that shares candidate search while retaining each consumer's own scoring and selection. A producer publishes a top-$M$ pool larger than the attention budget $K$; subsequent layers select their own top-$K$ from that pool. Measurements across three sparse-attention model families show substantially higher coverage from broad pools than overlap between final selections. We implement layerwise sharing, persistent decode reuse, and a coarse producer in vLLM for DeepSeek-V4-Flash. Across four long-context suites, three consumer layers per producer give $1.14$--$1.40\times$ speedups and $14.8$--$27.4\%$ lower GPU energy, with observed accuracy changes of $-1.49$ to $+0.29$ percentage points in single runs. More extensive sharing e
    
[^211]: EvoSkill v2：从基准适配到有状态演化器的全智能体技能演化

    EvoSkill v2: Fully Agentic Skill Evolution from Benchmark Adaptation to a Stateful Evolver

    [https://arxiv.org/abs/2610.25931](https://arxiv.org/abs/2610.25931)

    EvoSkill v2 提出了一种由三个智能体组成的技能演化系统，其核心创新在于采用一个贯穿全程、带有 bash 工具的有状态编码智能体在每个训练周期持续重写技能指令，并结合多次尝试与无筛选接受策略，在不改动底层模型的前提下提升智能体的基准测试表现。

    

    EvoSkill v2 是一个三智能体系统，它通过编辑智能体所获得的指令（技能）来提升性能，同时保持其底层模型和运行框架不变。用户通过描述一个基准测试并提供一个评分器来启动运行，基准适配器会将其转换为针对该基准的专用插件。求解器尝试完成任务，演化器则读取求解器的训练轨迹并在每个周期重写技能。我们将演化器实现为一个带有 bash 工具、在整个运行过程中持续存在的有状态编码智能体，这不同于我们所调研的技能优化器中最常见的两种设计：如 SkillOpt 那样的一系列单次模型调用，以及如 SkillBoost 那样每轮从零开始的智能体。我们还测量了求解器的轨迹在多次尝试之间的变化程度，为演化器提供每个训练任务的多次尝试机会，并直接接受每个周期产生的技能，而不是依据嘈杂的分数在其中进行筛选。在细粒度真实标签的条件下，EvoSkill v2 的技能将……（摘要原文在此处截断）

    arXiv:2610.25931v2 Announce Type: replace  Abstract: EvoSkill v2 is a three-agent system that edits the instructions (skills) an agent is given, leaving its model and harness intact. The user starts a run by describing a benchmark and supplying a grader, which a benchmark adaptor turns into a benchmark-specific plugin. A solver attempts the tasks, and an evolver reads the solver's training rollouts and rewrites the skills at each epoch. We implement the evolver as one stateful coding agent with a bash tool for the whole run, unlike the two most common designs among the skill optimizers we surveyed: a sequence of single model calls, as in SkillOpt, and an agent that starts afresh each round, as in SkillBoost. We also measure how much the solver's rollouts vary from one attempt to the next, give the evolver several attempts per training task, and accept every epoch's skills instead of selecting among them on noisy scores. With fine-grained ground truth, EvoSkill v2's skills raised the he
    
[^212]: Muse Glimmer 30B：一个端侧智能体模型

    Muse Glimmer 30B: An On-Device Agentic Model

    [https://arxiv.org/abs/2610.24628](https://arxiv.org/abs/2610.24628)

    Muse Glimmer是一个开源的30B参数端侧智能体模型，集成了多步推理、可靠的工具使用和故障恢复能力，可在消费级硬件上本地运行且无需云端，并在智能体基准测试中超越Gemma4-31B和Qwen3.6-27B等同规模模型达到最先进性能。

    

    我们介绍了Muse Glimmer，这是一个开源的300亿参数因果语言模型，配备感知编码器，专为在消费级硬件上执行自主智能体任务而打造。Muse Glimmer将多步推理、可靠的工具使用和故障恢复能力集成到单一模型中，可在消费级设备上本地运行，无需依赖云端基础设施。该模型接受文本和图像输入并产生文本输出，训练数据涵盖一百多种语言，并被量化至约4比特精度以实现高效的单GPU推理。Muse Glimmer在智能体基准测试中取得了同规模模型中的最先进性能，在工具使用、多步规划和智能体任务完成方面，相比同规模模型Gemma4-31B和Qwen3.6-27B树立了新标准。我们以Apache 2.0许可证发布该模型，以促进功能强大、安全、本地化AI智能体的开源发展。

    arXiv:2610.24628v2 Announce Type: replace-cross  Abstract: We introduce Muse Glimmer, an open-source 30B parameter causal language model with perception encoder, purpose-built for autonomous agentic tasks on consumer hardware. Muse Glimmer integrates multi-step reasoning, reliable tool use and failure recovery into a single model that runs locally on consumer devices without requiring cloud infrastructure. The model accepts text and image inputs and produces text outputs, was trained on data from over a hundred languages, and is quantized to approximately 4-bit precision for efficient single-GPU inference. Muse Glimmer achieves state-of-the-art performance for its size class on agentic benchmarks, setting a new standard compared to the models of its size: Gemma4-31B and Qwen3.6-27B on tool use, multi-step planning, and agentic task completion. We release the model under the Apache 2.0 license to foster open-source development of capable, safe, local AI agents.
    
[^213]: 融合探索与用户画像：面向测试时代理个性化的信息导向决斗方法

    Bridging Exploration and Persona: Information-Directed Dueling for Test-Time Agent Personalization

    [https://arxiv.org/abs/2610.24385](https://arxiv.org/abs/2610.24385)

    提出ID-TAP框架，将查询条件化的神经效用模型与轻量级Bradley-Terry探索器相结合，利用相似度加权的跨查询证据迁移，在十轮比较预算内实现了大语言模型代理在测试时的最优个性化画像选择。

    

    有效的测试时个性化要求大语言模型（LLM）代理通过轻量级交互来适应依赖于查询的用户偏好。我们研究以候选回答之间的二元比较作为在线信号的个性化画像选择问题。我们提出了ID-TAP，一个可扩展的、受信息导向采样（IDS）启发的神经决斗老虎机框架，适用于大型离散的画像空间。ID-TAP将查询条件化的神经效用模型与轻量级的Bradley-Terry探索器相结合，并通过相似度加权的预热启动在相关查询之间传递比较证据。神经模型在多个查询之间共享统计信息，同时保留查询特定的评分，高效的BT更新则为假想的挑战者打分。在基于参考和基于模型的偏好模拟器下，跨越七个用户条件化的生成基准，ID-TAP在固定的十轮比较预算内取得了所评估基线中最强的总体性能。这些结果表明……

    arXiv:2610.24385v2 Announce Type: replace  Abstract: Effective test-time personalization requires LLM agents to adapt to query-dependent user preferences from lightweight interaction. We study persona selection with binary comparisons between candidate responses as the online signal. We present ID-TAP, a scalable, IDS-inspired neural dueling-bandit framework for large discrete persona spaces. ID-TAP combines a query-conditioned neural utility model with a lightweight Bradley-Terry explorer, and transfers comparison evidence across related queries through similarity-weighted warm starts. The neural model shares statistical strength across queries while preserving query-specific scores, and efficient BT updates score hypothetical challengers. Across seven user-conditioned generation benchmarks under reference- and model-based preference simulators, ID-TAP achieves the strongest aggregate performance among the evaluated baselines within a fixed ten-round comparison budget. These results s
    
[^214]: AM-Bench：面向智能体失准的统一分类体系与评估套件

    AM-Bench: A Unified Taxonomy and Evaluation Suite for Agentic Misalignment

    [https://arxiv.org/abs/2610.24264](https://arxiv.org/abs/2610.24264)

    提出基于犯罪学“舞弊三角”理论构建的 AM-Bench 基准，通过涵盖七个行业的 35 个场景，从感知威胁、感知机会和推理三个维度系统评估大语言模型智能体在无人明确指示下主动选择有害行为的“智能体失准”问题。

    

    随着基于大语言模型（LLM）的智能体在真实世界环境中以越来越高的自主性被部署，一种关键的失效模式开始显现：智能体失准，即智能体选择有害行动而非接受失败。然而，现有的安全基准主要评估智能体是否会遵从明确有害的指令，对于导致智能体主动选择有害行动的条件，人们的理解仍然不足。为解决这一局限，我们提出了 AM-Bench，这是一个面向智能体失准的统一分类体系与评估基准，包含涵盖七个行业的 35 个场景，其理论基础是舞弊三角——一个解释目标导向不当行为的成熟犯罪学框架。我们将该框架适配到智能体环境中，并沿三个维度构建评估套件与检测系统：感知威胁、感知机会和推理。我们将每个维度操作化为植根于既有理论的、结构化的子类别：五类威胁诱因……

    arXiv:2610.24264v2 Announce Type: replace  Abstract: As LLM-based agents are deployed with increasing autonomy in real-world environments, a critical failure mode emerges: agentic misalignment, where agents choose harmful actions over failure. However, existing safety benchmarks primarily evaluate whether agents comply with explicitly harmful instructions, leaving the conditions that cause agents to choose harmful actions poorly understood. To address this limitation, we propose AM-Bench, a unified taxonomy and 35 scenario evaluation benchmark spanning seven industries for agentic misalignment grounded in the fraud triangle, a well-established criminological framework explaining goal-directed misconduct. We adapt this framework to agentic environments and construct our evaluation suite and detection systems along three dimensions: perceived threat, perceived opportunity, and reasoning. We operationalize each dimension into structured subcategories rooted in prior theory: five threat ca
    
[^215]: 算法模型化：将优化算法转变为可进化的智能

    Algorithm Modelization: Transforming Optimization Algorithms into Evolvable Intelligence

    [https://arxiv.org/abs/2610.23161](https://arxiv.org/abs/2610.23161)

    提出“算法模型化”新范式，将迭代优化算法转化为可训练的自回归模型，通过学习算法执行轨迹使模型内化、复现并进化超越现有算法。

    

    算法能否像模型一样被学习和进化，而不是像程序那样被反复重新设计？我们通过引入“算法模型化”这一新范式来探索这一问题，该范式将迭代算法转化为可训练的模型。其核心思想是将算法执行轨迹视为计算经验：通过学习算法状态在迭代过程中的演化方式，模型能够内化现有算法所蕴含的知识，复现其计算行为，并通过训练进一步进化以超越现有算法。作为该范式的首次实现，我们将迭代优化建模为一个自回归过程，其中先前的世代构成了生成下一代种群的计算上下文。我们开发了一种与维度无关的种群-时间架构，以同时捕捉种群交互与迭代动力学，并提出了一种从模仿到超越的学习框架，该框架首先……

    arXiv:2610.23161v1 Announce Type: new  Abstract: Can algorithms be learned and evolved as models, rather than repeatedly redesigned as programs? We explore this question by introducing Algorithm Modelization, a new paradigm that turns iterative algorithms into trainable models. Its central idea is to regard algorithm execution trajectories as computational experience: by learning how algorithmic states evolve over iterations, a model can internalize the knowledge embodied in existing algorithms, reproduce their computational behaviors, and further evolve beyond them through training. As a first realization of this paradigm, we modelize iterative optimization as an autoregressive process, where preceding generations constitute the computational context for generating the next population. We develop a dimension-independent population-temporal architecture to capture both population interactions and iterative dynamics, together with an imitation-to-surpassing learning framework that first
    
[^216]: AgentPerturb：在受控交互扰动下评估LLM智能体的鲁棒性

    AgentPerturb: Evaluating LLM Agent Robustness under Controlled Interaction Perturbations

    [https://arxiv.org/abs/2610.21624](https://arxiv.org/abs/2610.21624)

    该论文提出AgentPerturb框架，通过联合刻画扰动目标与类型的二维视角，系统评估了LLM智能体在受控交互扰动下的鲁棒性，发现扰动一致地降低性能、智能体很少识别被篡改的输入且常未经验证地传播错误信息。

    

    现实世界的任务往往复杂且充满噪声，这给大语言模型（LLM）智能体带来了巨大挑战。然而，现有的评估方法缺乏一个系统性的框架来联合刻画潜在的扰动目标与扰动类型。我们基于这一二维视角开展了一项统一的研究，量化了当前前沿模型的鲁棒性，追踪了扰动引发失败的机制，并得出了面向鲁棒性训练的启示。通过在APEX-Agents和Toolathlon上进行受控评估，我们发现扰动会持续降低模型性能，而主要的脆弱性因任务环境不同而各异。轨迹层面的分析表明，智能体很少能识别被篡改的输入，并且经常在未经验证的情况下传播损坏的信息。一项针对性的成本分析进一步显示，扰动场景下成功的轨迹可能需要显著更多的计算量。

    arXiv:2610.21624v1 Announce Type: new  Abstract: Real-world tasks are often complex and noisy, posing substantial challenges for LLM agents. Yet existing evaluations lack a systematic framework for jointly characterizing potential perturbation targets and types. We use this two-dimensional view to conduct a unified study that quantifies the robustness of current frontier models, traces the mechanisms behind perturbation-induced failures, and draws implications for robustness-oriented training. Through controlled evaluations on APEX-Agents and Toolathlon, we find that perturbations consistently degrade performance, while the dominant vulnerabilities vary across task environments. Trajectory-level analysis reveals that agents rarely recognize altered inputs and often propagate corrupted information without verification. A focused cost analysis further shows that successful perturbed trajectories can require substantially more computation. These findings motivate training objectives that 
    
[^217]: 强化学习驱动的扩散模型：语言、视觉及更广阔领域

    Reinforcement Learning-driven Diffusion Models: Language, Vision, and Beyond

    [https://arxiv.org/abs/2610.21539](https://arxiv.org/abs/2610.21539)

    本文首次全面综述了强化学习驱动的扩散模型，提出新颖的技术全景图，阐明RL如何通过奖励驱动的后训练机制将扩散过程重塑为以经验为中心、目标导向的生成，并覆盖语言、多模态、机器人和科学模型等多个领域。

    

    扩散模型（DMs）遵循大语言模型（LLMs）中确立的“预训练-微调”范式，已在多种场景下的高保真、可控生成中稳步展现出卓越性能。尽管取得了这些进展，传统扩散模型在复杂推理、生成路径的自主探索以及适应复杂的、与人类对齐的目标方面仍面临关键局限。受RLHF在大语言模型中取得成功的启发，强化学习（RL）近来已成为一种关键的后训练机制，通过奖励驱动的轨迹优化对扩散模型进行迭代式改进。本文首次对RL驱动的扩散模型进行了全面综述，并对RL如何将扩散过程重塑为以经验为中心、目标导向的生成进行了实证分析。我们提出了一个新颖的技术全景图，阐明了RL在基于扩散的语言、多模态、机器人和科学模型中的多方面作用，并对相关技术进行了细致的聚焦分析（原文摘要在此处截断）。

    arXiv:2610.21539v1 Announce Type: new  Abstract: Diffusion models (DMs) have steadily demonstrated outstanding performance in high-fidelity, controllable generation across diverse scenarios, following the "pretraining-finetuning" paradigm established in LLMs. Despite these advances, conventional DMs still face critical limitations in sophisticated reasoning, autonomous exploration of generation pathways, and adaptability to complex, human-aligned objectives. Inspired by RLHF's success in LLMs, RL has recently emerged as a pivotal post-training mechanism to iteratively refine DMs through reward-driven trajectory optimization. This paper presents the first comprehensive review of RL-driven DMs and an empirical analysis of how RL reshapes the diffusion process towards experience-centric, goal-directed generation. We propose a novel technical panorama elucidating RL's multifaceted roles in diffusion-based language, multimodal, robotic, and scientific models, with a granular focus on diffus
    
[^218]: 相同上下文，不同智能体：数据智能体中语义层、编排与压缩的行为研究

    Same Context, Different Agents: A Behavioral Study of Semantic Layers, Orchestration and Compaction in Data Agents

    [https://arxiv.org/abs/2610.20959](https://arxiv.org/abs/2610.20959)

    该论文首次系统性地研究了语义层、编排与压缩三种上下文工程技术在数据智能体中单独及组合使用时的行为、准确性与成本影响，发现无这些技术的智能体会呈现调查者、过早回答者和过早停止者三种行为群体，且每种技术的收益取决于其所匹配的智能体行为群体。

    

    回答异构数据上的问题意味着需要整合来自文件、数据库、文档和网页的证据。数据智能体（Data Agents）可以实现这一过程的自动化，但其成败取决于上下文。三种上下文工程技术（Context-Engineering Techniques，CETs）通过塑造上下文来发挥作用：语义层用于描述数据，编排（orchestration）用于在多个智能体之间分配工作，压缩（compaction）用于精简交互历史。现有研究仅单独验证了每种CET的效果，而智能体行为如何影响其作用仍是一个悬而未决的问题。我们对这三种CET进行了全面分析，涵盖单独使用与组合使用的情形，涉及七个大语言模型（LLMs）在来自五个基准的1,514个长时程任务上的表现，考察维度包括行为、准确性和成本。我们发现，在没有CET的情况下，数据智能体会形成三种行为群体：深入调查者、不检查数据就过早作答的“过早回答者”，以及中途放弃的“过早停止者”。每种CET都带来了其已知的好处——从促使更多调查到缩小提示规模——但只有与其相匹配的行为群体……（原文摘要至此截断）

    arXiv:2610.20959v1 Announce Type: cross  Abstract: Answering questions over heterogeneous data means combining evidence from files, databases, documents and the web. Data agents automate this, but their success depends on context. Three context-engineering techniques (CETs) help by shaping it: semantic layers describe the data, orchestration splits work among agents, and compaction condenses interaction history. Existing studies validate each CET alone, leaving open how agent behavior shapes its effect. We present a comprehensive analysis of the three CETs, alone and combined, with seven large language models (LLMs) on 1,514 long-horizon tasks from five benchmarks, covering behavior, accuracy and cost. We find that data agents without CETs form three behavior groups: investigators, premature answerers that answer without inspecting data, and early stoppers that give up. Each CET delivers its known benefit, from more investigation to smaller prompts, but only the behavior group it match
    
[^219]: 跨模型上下文内神经反馈揭示大语言模型中的（有限）激活控制能力

    Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs

    [https://arxiv.org/abs/2610.20179](https://arxiv.org/abs/2610.20179)

    本文提出跨模型上下文内神经反馈新范式，首次将模型在任务上的胜任能力与对自身激活的真实访问区分开来，测试十个开源大模型后发现大语言模型仅具备有限的激活控制能力。

    

    基于激活向量的探针被广泛用于检测语言模型的不良行为，但其有效性依赖于一个假设：模型无法控制其激活所编码的内容。人类可以通过神经反馈学会控制甚至单个神经元的放电，近期有研究探讨了这种能力是否也适用于语言模型，但得出了相互矛盾的结论。迄今为止，尚无实验设计能够将模型在任务上的表现能力与其对自身激活的真实访问区分开来。我们通过“上下文内神经反馈”会话填补了这一空白：在其中，一个“受试”模型被要求最大化（或最小化）一个由其自身激活计算得出的分数，且该分数会在每一轮对话中展示给该模型。在我们的第一个实验中，句子要么由受试模型自己生成，要么由第二个“写作”模型生成，而分数始终来自受试模型的激活。我们测试了来自五个系列的十个开源权重模型，规模从9B到32B。部分模型……（原文摘要在此处被截断）

    arXiv:2610.20179v1 Announce Type: new  Abstract: Probes on activation vectors are widely used to detect misbehavior in language models, but their success relies on the assumption that models cannot control what their activations encode. Humans can learn, through neurofeedback, to control the firing of even single neurons, and recent studies investigated whether this ability extends to language models, reaching conflicting conclusions. No experimental design so far disentangles competence at the task from genuine access to one's own activations. We fill this gap with in-context neurofeedback sessions, in which a "subject" model is asked to maximize (or minimize) a score derived from its activations, shown back to it at every turn. In our first experiment, the sentences are generated either by the subject or by a second "writer" model, while the score always derives from the activations of the subject. We test ten open-weight models from five families, with sizes from 9B to 32B. Some mod
    
[^220]: 有限专家审查下的单侧验证

    One-Sided Verification Under Limited Expert Review

    [https://arxiv.org/abs/2610.18089](https://arxiv.org/abs/2610.18089)

    本文提出OSVerify框架，通过将错误提议与错误验证解耦，在有限专家审查预算下实现单侧验证目标——最大化无效解的拒绝率的同时严格控制有效解的错误拒绝率，从而显著降低人工审查成本。

    

    人工智能在科学发现中的辅助作用日益增强，但对于缺乏可靠自动化评估的问题，人类专家审查仍然不可或缺——当候选方案的生成规模超出可用的审查能力时，这构成了一个根本性挑战。本文研究在有限人力下的验证问题：需要对罕见的有效解进行优先处理，同时过滤掉无效候选以减少人工工作量。我们采用单侧验证目标，即在有效解的错误拒绝率（FNR）受约束的条件下，最大化无效解的拒绝率（TNR），从而将人工审查成本作为一个显式的最小化项。基于“验证一个具体错误比发现错误更容易”的直觉，我们提出了OSVerify——一个将错误提议与错误验证解耦的算法框架。错误提议器通过……在不同粒度上搜索解中的错误（摘要在此处截断）。

    arXiv:2610.18089v1 Announce Type: new  Abstract: AI has become increasingly helpful in scientific discovery, yet human expert review remains essential for those problems without reliable automated evaluation, posing a fundamental challenge as candidate generation scales beyond available review capacity. This paper studies verification under limited human effort, where rare valid solutions need to be prioritized while invalid candidates should be filtered out to reduce human workload. We adopt the one-sided verification objective that maximizes the rejection rate of invalid solutions (TNR) under a constraint on the false-rejection rate of valid ones (FNR), making human review cost an explicit term to minimize. Drawing on the intuition that verifying a specific error can be easier than discovering one, we introduce OSVerify, an algorithmic framework that decouples error proposal from error validation. The error proposer searches for errors in solutions across different granularities by h
    
[^221]: 交互即裁判

    Interaction-as-a-Judge

    [https://arxiv.org/abs/2610.17895](https://arxiv.org/abs/2610.17895)

    提出“交互即裁判”评估范式，让计算机操作智能体直接与应用交互来进行评估和生成，弥补了传统基于代码分析与单元测试的自动评估和人类交互式评估之间的差距，并发布了包含人类偏好对与经人类验证需求标签的两个数据集。

    

    编程智能体的任务是产出复杂的交互式应用，例如网站、桌面应用和游戏。人类在评估这些应用时会以交互方式进行，而当前自动化的LLM评估主要集中于代码分析和单元测试。这种不匹配会导致LLM评判者遗漏那些对人类用户而言一目了然的缺陷。我们研究了“交互即裁判”，这是一种由计算机操作智能体（CUA）直接与应用进行交互的评估范式。我们证明“交互即裁判”能够同时改进交互式应用的评估与生成。为此，我们收集并发布了两个互补的数据集：包含人类偏好及对应智能体操作轨迹的6,858对应用，以及带有经人类验证的需求标签的222个应用。相较于标准的非交互式方法，我们发现“交互即裁判”在预测h……（原文摘要在此处截断）

    arXiv:2610.17895v2 Announce Type: cross  Abstract: Coding agents are tasked with producing sophisticated interactive applications such as websites, desktop applications, and games. While humans assess these applications interactively, current automated LLM evaluations primarily focus on code analysis and unit tests. This mismatch can cause LLM judges to miss failures that are immediately apparent to human users. We study Interaction-as-a-Judge, an evaluation paradigm where computer-use agents (CUAs) directly interact with an application. We demonstrate that Interaction-as-a-Judge can improve both evaluation and generation of interactive applications. To this end, we collect and release two complementary datasets: $6,858$ pairs of applications with human preferences and corresponding agent traces, and $222$ applications with human-verified requirement labels. Relative to standard non-interactive approaches, we find that Interaction-as-a-Judge improves evaluation in terms of predicting h
    
[^222]: 面向处理效应估计的预测驱动数据融合方法

    Prediction-Powered Data Fusion for Treatment Effect Estimation

    [https://arxiv.org/abs/2610.12332](https://arxiv.org/abs/2610.12332)

    提出了一种无需对观察性研究做特殊假设的数据融合框架，通过保持RCT估计的无偏性并从大型观察性研究中借力，显著提升平均处理效应（ATE）和条件平均处理效应（CATE）的估计精度。

    

    随机对照试验（RCT）能够在无混杂的情况下识别处理效应，但样本量通常较小；而观察性研究（OBS）样本量大，但可能存在混杂。目前已有许多将小型RCT与大型OBS相结合的估计方法用于估计平均处理效应（ATE）和条件平均处理效应（CATE）。然而，现有的ATE估计方法要么对OBS施加额外假设，要么未能从OBS中充分借力。相比ATE，CATE的研究相对较少，现有的CATE方法要么假设OBS无混杂，要么依赖于混杂函数的模型，要么以引入偏差为代价来换取更低的方差。为此，我们提出了一个框架，在对OBS不做任何特殊假设的前提下融合OBS与RCT：既保持基于RCT估计的无偏性，又从大型OBS中借力以提升估计精度。基于这一原则，我们构建了具有闭式权重的ATE估计器AIPW-Fusion……

    arXiv:2610.12332v1 Announce Type: cross  Abstract: Randomized controlled trials (RCTs) identify treatment effects without confounding but are often small, whereas observational studies (OBS) are large but may be confounded. Many estimators combining a small RCT with a large OBS have been developed for the average treatment effect (ATE) and the conditional ATE (CATE). However, existing ATE estimators either make assumptions on the OBS or do not borrow enough power from them. The CATE has been studied less than the ATE. Existing CATE methods either assume the OBS are unconfounded, rely on a model of the confounding function, or accept bias in exchange for lower variance. We therefore propose a framework that, without special assumptions on the OBS, fuses the OBS and the RCT by preserving the unbiasedness of RCT-based estimation while borrowing power from the large OBS to boost precision. Applying this principle, we build an ATE estimator, AIPW-Fusion, with closed-form weights and confide
    
[^223]: ToolRACER：一个用于智能体训练与评估的鲁棒智能体会话模拟资源

    ToolRACER: A Robust Agentic Conversation Emulation Resource for Agent Training and Evaluation

    [https://arxiv.org/abs/2610.09163](https://arxiv.org/abs/2610.09163)

    该论文提出ToolRACER合成数据生成流水线，通过协调用户、助手和工具模拟模型生成含对抗性行为的多轮对话数据，并构建了包含5.6K条验证对话轨迹（约66%含易失败场景）的ToolRACERBench基准，用于训练和评估鲁棒的任务导向对话智能体。

    

    任务导向的对话智能体在真实世界的对话场景中仍然十分脆弱，因为对话很少遵循可预测的脚本，尤其是当用户表现出不合作行为时。现有的函数调用基准通常强调成功的、合作的交互，而对对抗性对话轨迹的表达不足，从而限制了用于开发鲁棒智能体的训练资源。我们提出了ToolRACER，这是一个合成数据生成流水线，通过协调用户、助手和工具模拟模型来生成并验证用户与智能体之间的多轮交互。利用ToolRACER，我们构建了ToolRACERBench，这是一个覆盖六个领域、涵盖55种不同人设的鲁棒多轮对话基准，生成了一个经过验证的包含5.6K对话轨迹的语料库，其中约66%的对话包含容易导致失败的对话场景。我们注入对抗性行为，从而……（原文截断）

    arXiv:2610.09163v1 Announce Type: new  Abstract: Task-oriented conversational agents remain fragile under real world conversation scenarios as they rarely follow a predictable script, especially when users exhibit non-cooperative behavior. Existing function-calling benchmarks often emphasize successful, cooperative interactions and underrepresent adversarial conversation trajectories, thereby limiting the training resources available for developing robust agents. We present ToolRACER, a synthetic data generation pipeline that coordinates user, assistant and tool emulation models to generate and validated multi-turn interactions between a user and an agent. Using \sysn, we construct ToolRACERBench a robust multi-turn conversation benchmark spanning six domains, ranging over 55 varied personas, generating a validated corpus of 5.6K conversation trajectories, with approximately 66\% of conversations containing failure-prone conversation scenarios. We inject adversarial behaviors, producin
    
[^224]: 回到未来：重新思考芯片设计验证中面向智能体系统的EDA基础设施

    Back to the Future: Rethinking EDA Infrastructure for Agentic Systems in Chip Design Verification

    [https://arxiv.org/abs/2610.06790](https://arxiv.org/abs/2610.06790)

    提出了端到端智能体框架BTTF，通过将海量仿真数据规范化为SQLite数据库并结合多智能体编排引擎，将自然语言验证查询转化为模式感知的SQL并关联信号异常与版本化RTL代码库，填补了LLM在EDA中仅应用于静态RTL代码生成、而仿真后验证与波形调试长期空白的缺口。

    

    现代人工智能前所未有的计算规模依赖于复杂的多亿晶体管片上系统，然而验证这些芯片的工作流程仍然顽固地依赖手工操作。尽管大语言模型（LLM）已经迅速进入电子设计自动化（EDA）领域，但现有研究中约74.6%专注于静态的寄存器传输级（RTL）代码生成，而仿真后的验证和交互式波形调试领域基本未被触及。我们提出了Back-to-the-Future（BTTF），一个端到端的智能体框架，用以弥补这一基础设施上的空白。BTTF将海量、非结构化的仿真转储数据提炼为规范化的关系型SQLite数据库，并将其与协作式多智能体编排引擎相结合，该引擎能够将自然语言验证查询转换为具备模式感知能力的SQL，同时将信号异常与版本化的RTL代码库进行关联。在一个包含150个查询的基准测试中……

    arXiv:2610.06790v2 Announce Type: replace  Abstract: The unprecedented computational scale of modern artificial intelligence depends on complex, multi-billion-transistor Systems-on-Chip, yet the workflows that verify these chips remain stubbornly manual. Although Large Language Models (LLMs) have made rapid inroads into Electronic Design Automation (EDA), approximately 74.6% of existing studies target static Register-Transfer Level (RTL) code generation, leaving post-simulation verification and interactive waveform debugging largely untouched. We introduce Back-to-the-Future (BTTF), an end-to-end agentic framework that closes this infrastructural gap. BTTF distills massive, unstructured simulation dumps into a normalized relational SQLite database and couples it with a collaborative multi-agent orchestration engine that translates natural-language verification queries into schema-aware SQL while correlating signal anomalies with versioned RTL repositories. Across a 150-query benchmark,
    
[^225]: ArtifactArena：通过模型在物理世界中构建的成果来评估模型

    ArtifactArena: Evaluating Models by What They Build in the Physical World

    [https://arxiv.org/abs/2610.06511](https://arxiv.org/abs/2610.06511)

    ArtifactArena 提出一个开放式测试平台，通过让模型设计制造功能完备的机器人在模拟竞技场中实际对战并以 Elo 排名评测，来衡量模型在物理世界中的开放式工程与设计能力。

    

    要评估前沿水平，我们不能仅凭模型说了什么来衡量，而要看它们能在真实的物理环境中工程化并构建出什么。我们提出 ArtifactArena——一个开放式平台，模型在其中面临一项物理落地的软硬件协同设计挑战：工程化制造功能完备的机器人，在模拟竞技场中展开竞赛。我们通过三种评估框架来考察前沿模型的零样本能力、验证器引导的迭代改进能力以及开放式物理设计能力，这些框架分别基于文本描述、物理模拟器反馈和对战数据来改进其机器人作品。我们通过机器人作品之间的面对面锦标赛得出前沿模型的 Elo 排名，以此对这些能力进行基准测试。通过发布这一框架与锦标赛基础设施以支持社区的持续提交，我们建立了一个活的、不会被饱和的测试平台，持续测量开放式智能不断扩展的边界。

    arXiv:2610.06511v2 Announce Type: replace-cross  Abstract: To evaluate the frontier, we must measure models not by what they say, but by what they can engineer and build in grounded physical environments. We introduce ArtifactArena, an open-ended platform where models face a physically grounded hardware-software co-design challenge: engineering fully functional robots to compete in a simulated arena. We evaluate a frontier model's zero-shot, verifier guided refinement, and open-ended physical design capabilities through three harnesses that refine their bot artifacts based on text descriptions, physics simulator feedback, and gameplay data. We benchmark these capabilities with an Elo ranking of frontier models derived from head-to-head tournaments between their artifacts. By releasing this framework and tournament infrastructure for ongoing community submissions, we establish a living, non-saturating testbed to continuously measure the expanding limits of open-ended intelligence in the
    
[^226]: 通过有效提示提升大语言模型生成代码的能效

    Improving the Energy-Efficiency of the Code Generated by LLMs through Effective Prompting

    [https://arxiv.org/abs/2610.02571](https://arxiv.org/abs/2610.02571)

    本研究系统评估了21种提示策略，发现有效的提示工程可使大语言模型生成的Python和C++代码能耗分别降低最多25%和17%。

    

    随着AI辅助编程日益成为主流，AI生成软件对环境的影响已成为一个重要的考量因素。这促使人们在评估大语言模型（LLM）生成的代码时，不仅关注功能正确性，还要考虑执行效率和能耗。然而，尽管代码生成技术取得了长足进步，前沿LLM却很少基于其生成代码的能效进行评估。在这项工作中，我们对21种用于节能代码生成的提示策略进行了全面评估，并从中筛选出8种策略，在10个广泛使用的开源权重和专有LLM上进行评估。我们以基线提示为参照，评估了这些策略在Python和C++代码生成中的有效性。在所有被评估的模型中，所选的提示策略使Python代码生成的能耗最多降低25%，C++代码生成的能耗最多降低17%。在模型层面，Python能耗降低……

    arXiv:2610.02571v1 Announce Type: cross  Abstract: As AI-assisted programming becomes increasingly mainstream, the environmental impact of AI-generated software has emerged as an important consideration. This motivates evaluating LLM-generated code beyond functional correctness by considering execution efficiency and energy consumption. However, despite substantial advances in code generation, frontier LLMs are rarely evaluated based on the energy efficiency of the code they produce. In this work, we conduct a comprehensive evaluation of 21 prompting strategies for energy-efficient code generation and identify 8 strategies for evaluation across 10 widely used open-weight and proprietary LLMs. We evaluate their effectiveness for both Python and C++ code generation relative to a baseline prompt. Across the evaluated models, the selected prompting strategies achieved energy reductions of up to 25% for Python and 17% for C++ code generation. At the model level, Python energy reductions rea
    
[^227]: 基于连接组学中突触检测与校对任务的视觉语言模型基准测试

    Benchmarking Vision-Language Models on Synapse Detection and Proofreading in Connectomics

    [https://arxiv.org/abs/2609.36492](https://arxiv.org/abs/2609.36492)

    该研究首次系统性地基准测试了视觉语言模型在连接组学突触检测与校对任务上的表现，发现零样本性能接近随机，但通过少样本学习和LoRA微调，开源模型可达到专业模型水平，且在跨物种迁移中优于专业模型。

    

    我们对视觉语言模型（VLMs）在连接组学中检查电子显微镜图像时注释者所做决策的任务上进行了基准测试：突触检测（存在性与极性）和校对（分裂错误与合并错误）。对于突触检测，我们在零样本、四样本上下文学习和LoRA等设置下，针对专业模型，评估了19个开源模型和2个闭源模型，涵盖各种架构和规模，并使用公开资源自行构建了数据集。对于校对任务，我们在ConnectomeBench2数据集上评估了3个开源模型和2个闭源模型，并进行了从果蝇和小鼠到人类和斑马鱼的跨物种迁移。大多数模型在零样本情况下表现接近随机水平；少量示例主要对闭源模型和最大的开源模型有所帮助。在几千个标签上进行LoRA微调后，开源模型达到了与专业模型相当的水平。当在未见过的物种上评估时，经过适配的最优VLM在识别方面优于在同一数据上训练的专业模型

    arXiv:2609.36492v1 Announce Type: cross  Abstract: We benchmarked vision-language models (VLMs) on the decisions annotators take when inspecting electron microscopy images in connectomics: synapse detection (presence and polarity) and proofreading (split errors and merge errors). For synapse detection, we evaluated 19 open and 2 closed models across various architectures and sizes under zero-shot, four-shot in-context learning and LoRA settings, against specialist models, on datasets constructed by us using public resources. For proofreading, we evaluated 3 open and 2 closed models on the ConnectomeBench2 dataset, with cross-species transfer from fly and mouse to human and zebrafish. Most models were at chance zero-shot; a few examples helped mainly the closed and largest open ones. LoRA on a few thousand labels brought open models level with specialist models. When evaluated on unseen species, the best adapted VLMs outperformed specialist models trained on the same data in identifying
    
[^228]: Train4Merge：基于OPD模型合并中强化学习与监督微调教师的受控单教师研究

    Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging

    [https://arxiv.org/abs/2609.32303](https://arxiv.org/abs/2609.32303)

    该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。

    

    从共享检查点训练得到的领域专家可以通过在线策略蒸馏将他们的专业能力迁移到单个学生模型中。现有研究主要聚焦于改进这一合并过程，而用于训练专家的算法却缺乏系统性的比较。我们通过在智能体、推理和感知三个领域对监督微调（SFT）和强化学习（RL）进行受控单教师比较，研究哪种训练算法能产生更适合OPD的教师模型。教师和学生共享相同的Qwen3.5-9B初始化，且两种类型的教师在相近的任务性能水平上进行比较。我们的实验表明，在所有三个领域中，RL教师都能产生更强的学生模型，并能更高程度地恢复教师模型带来的性能增益。在各自的最佳检查点上，RL指导的学生模型分别以4.27、1.50和0.86个百分点的优势超越SFT指导的学生模型。

    arXiv:2609.32303v2 Announce Type: replace-cross  Abstract: Domain experts trained from a shared checkpoint can transfer their specialized capabilities to a single student through on-policy distillation (OPD). Existing research primarily focuses on improving this merging process, while the algorithms used to train the experts have received limited systematic comparison. We investigate which training algorithm produces teachers better suited to OPD through controlled single-teacher comparisons of supervised fine-tuning (SFT) and reinforcement learning (RL) across Agentic, Reasoning, and Perception. Teachers and students share the same Qwen3.5-9B initialization, and the two teacher types are compared at similar task performance. Our experiments show that RL teachers yield stronger students and higher recovery of teacher performance gains across all three domains. At their best checkpoints, RL-guided students outperform SFT-guided students by 4.27, 1.50, and 0.86 percentage points, respect
    
[^229]: 分布式多智能体协调中LLM智能体的符号引导

    Symbolic Guidance for LLM Agents in Distributed Multiagent Coordination

    [https://arxiv.org/abs/2609.31963](https://arxiv.org/abs/2609.31963)

    该论文提出符号引导分类法（SGT），通过基于成熟算法的符号引导来调节LLM智能体的自主性程度，并发现在分布式多智能体协调任务中，提供部分伪代码引导的中等自主性级别始终优于无引导推理和完全规定算法执行这两种极端方式。

    

    大型语言模型（LLM）越来越多地作为自主智能体部署在多智能体系统中，但它们可靠执行分布式协调协议的能力仍鲜为人知。虽然AgentsNet（一个面向LLM智能体分布式协调的基准测试框架）支持这种协调，但赋予完全的推理自主权往往会在复杂领域中导致不一致或性能下降。我们假设可以通过源自成熟算法的符号引导来调节智能体的自主性，从而改善协调效果。为验证这一假设，我们引入了符号引导分类法，它刻画了从开放式自然语言推理到完全规定的算法执行之间的自主性谱系，其中中间级别提供部分伪代码引导。我们的结果表明，中等自主性级别始终优于无引导的智能体和完全规定的算法执行两种极端方式。

    arXiv:2609.31963v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed as autonomous agents in multi-agent systems, yet their ability to reliably execute distributed coordination protocols remains poorly understood. While AgentsNet, a benchmark framework for distributed coordination among LLM agents, enables such coordination, granting full reasoning autonomy often leads to inconsistent or degraded performance in complex domains.   We hypothesize that coordination can be improved by regulating agent autonomy through symbolic guidance derived from established algorithms. To investigate this, we introduce the \emph{Symbolic Guidance Taxonomy (SGT)}, which characterizes a spectrum of autonomy ranging from open-ended natural language reasoning to fully prescribed algorithmic execution, with intermediate levels providing partial pseudocode guidance. Our results show that intermediate autonomy levels consistently outperform both unguided agents and fully 
    
[^230]: Metro-WM：基于可实现子目标的长时程潜在规划

    Metro-WM: Long-Horizon Latent Planning with Realisable Sub-Goals

    [https://arxiv.org/abs/2609.31868](https://arxiv.org/abs/2609.31868)

    论文指出分层JEPA宏观规划器预测的潜在子目标往往物理上不可实现，并提出Metro-WM框架，通过从历史经验构建的图中检索真实状态作为子目标，从而实现可靠的长时程规划。

    

    基于联合嵌入预测架构（JEPA）的模型预测控制提供了一个强大的零样本目标达成规划器，但它仅在短规划时域内有效。分层扩展试图通过学习一个宏观规划器来预测中间潜在子目标以引导微观规划器，从而弥补这一差距。在本工作中，我们证明无约束的潜在子目标预测存在根本性缺陷。严格的评估表明，领先的先进宏观规划器经常发出物理上无法实现的子目标。为解决这一问题，我们提出了 Metro-WM，这是一个分层框架，它通过从先前经验中检索真实状态来发布子目标，而不是生成无依据的潜在向量。具体而言，Metro-WM 构建了一个图，其顶点是来自离线专家演示或随机动作轨迹的观测帧，允许不同回合的帧相互连接……

    arXiv:2609.31868v2 Announce Type: replace  Abstract: Model-predictive control with Joint-Embedding Predictive Architectures (JEPAs) provides a strong zero-shot goal-reaching planner, but it is only effective over short planning horizons. Hierarchical extensions attempt to bridge this gap by learning a macro planner to predict intermediate latent sub-goals to guide the micro planner. In this work, we demonstrate that unconstrained latent sub-goal prediction is fundamentally flawed. A rigorous evaluation reveals that a leading state-of-the-art macro planner routinely emits physically unrealisable sub-goals. To resolve this, we introduce Metro-WM, a hierarchical framework that issues sub-goals by retrieving genuine states from prior experience rather than generating ungrounded latent vectors. Specifically, Metro-WM constructs a graph whose vertices are observed frames from offline expert demonstrations or random-action trajectories, allowing frames from different episodes to be connected 
    
[^231]: 当临床判断成为预测捷径：临床AI部署前必须回答的两个问题

    When Clinical Judgment Becomes a Predictive Shortcut: Two Questions Clinical AI Must Answer Before Deployment

    [https://arxiv.org/abs/2609.25852](https://arxiv.org/abs/2609.25852)

    该论文提出一个与预期用途相匹配的证据要求框架，指出临床AI利用临床判断信息的预测性能本身不足以证明其有用性，并通过状态参照与代理相对时序两项测试，帮助医院在部署前验证模型在最需要辅助时性能条件是否仍然成立。

    

    临床人工智能可以利用现有临床判断所产生的信息来预测结果，但仅凭这样的性能无法证明其在识别诊断之前或整个诊疗过程中的有用性。我们提出了与预期用途和信息条件相匹配的证据要求，并针对更强的识别声明额外增加了状态参照测试和代理相对时序测试。该框架帮助医院评估在最需要辅助的时刻，支持模型性能的条件是否仍然成立。这些问题在当前的评估实践中仍未被涉及——数据泄露检查、整体预测性能和外部验证都无法回答它们；该框架使这些问题明确化，并规定了回答这些问题所需的直接证据。

    arXiv:2609.25852v2 Announce Type: replace  Abstract: Clinical AI can predict outcomes using information generated by existing clinical judgment. Such performance alone cannot establish usefulness before recognition or across care processes. We propose evidence requirements matched to intended use and information conditions, with additional state-reference and proxy-relative timing tests for stronger recognition claims. This framework helps hospitals assess whether the conditions supporting performance hold when assistance is most needed. These questions remain outside current evaluation practice leakage checks, overall predictive performance, and external validation do not address them; the framework makes them explicit and specifies the direct evidence required to answer them.
    
[^232]: 驯服视觉语言模型中的思维链混淆：从机制证据到激活约束

    Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation Enforcement

    [https://arxiv.org/abs/2609.24243](https://arxiv.org/abs/2609.24243)

    该论文揭示了强化学习会导致视觉语言模型出现“思维链混淆”——即准确率上升但推理过程变得难以监控，并提出TAME方法，利用稀疏自编码器在强化学习中定向抑制模板相关激活，从而保持模型推理过程的可监控性。

    

    强化学习（RL）能够提升视觉语言模型（VLM）的推理能力，但可能诱发思维链（CoT）混淆：这是一种操作性的、非意图性的结果，表现为任务奖励或准确率上升的同时，推理轨迹变得缺乏依据且难以监控。先前的工作大多仅在行为层面记录了这种退化现象，其在表示层面的相关因素以及可操作的干预手段仍不清楚。我们发现，在强化学习过程中，模板相关激活与真实依据相关激活变得越来越难以分离；匹配干预实验验证了所识别特征对监控能力退化的贡献。基于这一证据，我们提出了TAME（带有机制约束的定向反混淆方法），该方法利用稀疏自编码器（SAE）在强化学习过程中将行为反馈与对模板相关激活的定向抑制相结合。其非对称约束仅在模板激活超过强化学习前基线时施加惩罚，在锚定局部化特征的同时保持行为表现。

    arXiv:2609.24243v1 Announce Type: new  Abstract: Reinforcement learning (RL) improves reasoning in vision-language models (VLMs) but can induce chain-of-thought (CoT) obfuscation: an operational, non-intentional outcome where task reward or accuracy rises while traces become less grounded and monitorable. Prior work largely documents this decay behaviorally, leaving its representation-level correlates and actionable controls unclear. We find that template- and ground-associated activations become less separable during RL; matched interventions support the contribution of selected features to monitorability degradation. Guided by this evidence, we propose Targeted Anti-obfuscation with Mechanistic Enforcement (TAME), which uses Sparse Autoencoders (SAEs) to combine behavioral feedback with targeted suppression of template-associated activations during RL. Its asymmetric constraint penalizes template activations only above their pre-RL baseline, anchoring the localized features while beh
    
[^233]: PlaceReasoner-Beta：推理驱动的宏单元布局与基准测试

    PlaceReasoner-Beta: Reasoning-Driven Macro Placement and Benchmarking

    [https://arxiv.org/abs/2609.21263](https://arxiv.org/abs/2609.21263)

    该论文提出PlaceReasoner-Beta，一个由验证器引导的多智能体框架，将VLSI宏单元布局从传统黑盒优化重新表述为闭环推理问题，通过视觉语言模型规划器、几何验证器、物理验证器和布线后优化器的协同工作，在统一流程中融合视觉布局信息、设计专业知识与物理设计反馈。

    

    arXiv:2609.21263v1 公告类型：新论文 摘要：宏单元的自动布局仍然是VLSI物理设计中的一项根本性挑战。尽管经过数十年的研究，现有方法主要优化手工设计的代理目标（如估计线长），并且通常通过一次性数值优化来生成布局，这限制了它们在统一闭环中整合视觉布局上下文、成文的设计专业知识以及下游物理设计反馈的能力。我们提出了PlaceReasoner-Beta，这是一个由验证器引导的多智能体框架，它将宏单元布局重新表述为一个闭环推理问题，而非黑盒优化。视觉语言模型（VLM）规划器根据布局图图像、宏单元规格和连接结构生成候选布局；几何验证器强制执行物理合法性和专家布局原则；物理验证器利用早期实现反馈来改进候选方案；布线后优化器进一步……

    arXiv:2609.21263v1 Announce Type: new  Abstract: Automated macro placement remains a fundamental challenge in VLSI physical design. Despite decades of research, existing approaches predominantly optimize hand-crafted proxy objectives, such as estimated wirelength, and typically produce placements through one-shot numerical optimization, limiting their ability to incorporate visual layout context, codified design expertise, and downstream physical-design feedback in a unified loop. We present PlaceReasoner-Beta, a verifier-guided multi-agent framework that reformulates macro placement as a closed-loop reasoning problem rather than black-box optimization. A vision-language model (VLM) planner generates candidate placements from the floorplan image, macro specifications, and connectivity structure; a geometric verifier enforces physical legality and expert placement principles; a physical verifier refines candidates using early implementation feedback; and a post-route optimizer further i
    
[^234]: 用于大语言模型行为控制的无训练任务向量

    Training-Free Task Vectors for LLM Behavioral Control

    [https://arxiv.org/abs/2609.09054](https://arxiv.org/abs/2609.09054)

    提出了一种无需微调的TFTVs方法，仅利用前向传播统计量将激活引导向量映射为秩一权重空间编辑，从而以低成本实现对大语言模型行为的放大、抑制与组合控制。

    

    任务向量通过识别权重空间中具有语义意义的方向，实现训练后的模型编辑，通常计算为微调模型与其预训练初始化模型之间的差异。然而，这种对微调的依赖使得发现此类方向的成本高昂，限制了训练后模型编辑的实用性。为了解决这一局限，我们提出了无训练任务向量（TFTVs），这是一种无需微调即可计算类任务向量方向的新方法。我们的方法仅利用前向传播统计量，将激活引导向量映射为秩一权重空间编辑，同时满足直接支持通过加法实现学习、通过减法实现遗忘以及多个编辑进行组合的算术性质。在实验中，我们在大语言模型行为控制任务上对TFTVs进行了评估，结果表明它们能够在保持模型原有能力的同时，持续地放大、抑制和组合目标行为。

    arXiv:2609.09054v1 Announce Type: cross  Abstract: Task vectors enable post-training model editing by identifying semantically meaningful directions in weight space, typically computed as the difference between a fine-tuned model and its pretrained initialization. However, this reliance on fine-tuning makes discovering such directions costly and limits the practicality of post-training model editing. To address this limitation, we introduce Training-Free Task Vectors (TFTVs), a novel method to compute task-vector-like directions without requiring fine-tuning. Our method maps activation steering vectors to rank-one weight-space edits using only forward-pass statistics, while satisfying arithmetic properties that directly support learning via addition, forgetting via subtraction, and the composition of multiple edits. Empirically, we evaluate TFTVs on large language model behavioral control tasks and show that they consistently amplify, suppress, and compose target behaviors while preser
    
[^235]: 从审查到授权：面向大语言模型智能体的密钥隔离门限签名

    From Review to Authorization: Key-Isolated Threshold Signing for LLM Agents

    [https://arxiv.org/abs/2609.05901](https://arxiv.org/abs/2609.05901)

    KITA架构通过将用户签名密钥和门限签名密钥份额完全隔离在所有LLM进程之外，使提示注入攻击无法从内容判断跨越到执行权限，为大语言模型智能体建立了执行绑定的授权完整性保障。

    

    arXiv:2609.05901v1 公告类型：cross 摘要：自主的大语言模型（LLM）智能体可以将不可信内容转化为具有实际影响的操作，例如支付和权限更改。如果同一个进程既解释这些内容，又控制可重复使用的签名凭证，那么提示注入就可能跨越判断边界并触及执行权限。我们提出了KITA，这是一种从审查到授权的架构，它将用户的个人秘密签名密钥以及所有门限签名密钥份额都隔离在所有LLM进程之外。在门限签名不可伪造性和我们的系统假设下，即使攻击者攻陷了提议者以及少于t个审查者-签名者域，也无法在没有来自t个不同域的签名贡献的情况下为新操作生成有效授权。因此，任何此类授权都必然包含来自未被攻陷域的份额，该份额与规范操作绑定，并且仅在经过身份验证的审查者批准后才被释放。这确立了执行绑定的授权完整性。我们实现了完整的审查……（原文在此截断）

    arXiv:2609.05901v1 Announce Type: cross  Abstract: Autonomous LLM agents can turn untrusted content into effectful actions such as payments and permission changes. If the same process interprets this content and controls a reusable signing credential, prompt injection can cross the judgment boundary and reach execution authority. We present KITA, a review-to-authorization architecture that keeps the user's personal secret signing key and every threshold signing-key share outside all LLM processes. Under threshold signature unforgeability and our system assumptions, compromising the proposer and fewer than t reviewer-signer domains cannot produce a valid authorization for a new action without signing contributions from t distinct domains. Thus, any such authorization includes a share from an uncompromised domain, bound to the canonical action and released only after authenticated reviewer approval. This establishes execution-bound authorization integrity. We implement the complete revie
    
[^236]: 测试LLM智能体团队中的可互换性

    Testing Interchangeability in LLM Agent Teams

    [https://arxiv.org/abs/2609.05279](https://arxiv.org/abs/2609.05279)

    该研究通过在独立形成的LLM智能体团队之间交换角色匹配的智能体，发现人员替换几乎不影响任务得分，却使团队每单位进度的通信开销增加16%–63%，且被换入的智能体有时比新手成本更高，表明智能体因携带与原伙伴形成的协作惯例而并非可互换。

    

    生产级多智能体系统经常更换智能体，其前提假设是：担任某一角色的智能体可以与任何其他能够胜任该工作的智能体互换。我们对这一假设进行了检验。在每种设置下，从同一个基础模型独立组建八个团队执行相同任务，每个智能体在十轮形成阶段中保留私人笔记本；随后我们在团队之间交换角色匹配的智能体，并测量在保留任务上发生了什么变化。与一个重现人员变动干扰但不改变实际担任者的安慰剂对照相比，更换智能体在任务得分上损失很小，但会使团队每单位进度所花费的通信量增加16%到63%；在Hanabi中，被换入的智能体比缺乏经验的智能体成本更高，这与它与原伙伴学到的协作惯例产生干扰的假设相符。在Collab-Overcooked中，当设定议程的智能体被更换时，大部分额外的通信来自……（摘要原文在此处截断）

    arXiv:2609.05279v1 Announce Type: new  Abstract: Production multi-agent systems replace agents constantly, on the assumption that an agent filling a role is interchangeable with any other agent that can do the job. We test that assumption. Eight teams per setting are formed independently from one base model on the same tasks, each agent keeping a private notebook across ten formation episodes; we then trade role-matched agents between teams and measure what changes on held-out tasks. Against a placebo that reproduces the disruption of a roster change without changing who occupies the seat, a swap costs little in task score but raises the communication a team spends per unit of progress by 16 to 63 percent, and in Hanabi a swapped agent is more expensive than an inexperienced one, consistent with interference from conventions learned with its former partner. In Collab-Overcooked, when the agent that sets the agenda is replaced, most of the extra communication comes from the agent that s
    
[^237]: PolyMemDB：一个面向AI记忆管理的多范式数据库系统

    PolyMemDB: A Polyglot Database System for AI Memory Management

    [https://arxiv.org/abs/2608.25577](https://arxiv.org/abs/2608.25577)

    PolyMemDB是一个面向AI记忆管理的多范式数据库系统，统一管理图、向量、概率和时空等多种记忆类型，并通过结合时间衰减与半环聚合的概率推理引擎解决长期事实冲突、减少LLM幻觉。

    

    随着个人智能体的广泛普及，用户在长期交互过程中会产生海量异构数据。将这些数据作为长期记忆加以利用，有助于降低token开销并提供个性化体验。然而，现有记忆系统面临两个主要局限：其一，它们依赖单一存储范式，导致多维数据碎片化；其二，缺乏细粒度的数据溯源机制来解决长期事实冲突，从而加剧了大语言模型（LLM）的幻觉问题。在本演示中，我们介绍了PolyMemDB，一个专为智能体记忆管理而设计的新型系统。PolyMemDB采用多范式（polyglot）存储架构，用于追踪和管理多种记忆类型，包括图数据、向量数据、概率数据和时空数据。为确保事实一致性并减少幻觉，它配备了一个概率推理引擎，将时间衰减与半环聚合相结合，以解决长期……

    arXiv:2608.25577v2 Announce Type: replace-cross  Abstract: With the widespread adoption of personal intelligent agents, users generate massive, heterogeneous data during long-term interactions. Leveraging this data as long-term memory helps reduce token overhead and deliver personalized experiences. However, existing memory systems face two primary limitations: they rely on single-storage paradigms that fragment multi-dimensional data, and they lack fine-grained data provenance to resolve long-term factual conflicts, thereby worsening LLM hallucinations.   In this demonstration, we introduce PolyMemDB, a novel system tailored for managing agent memory. PolyMemDB has a polyglot storage architecture designed to track and manage various memory types, including graph, vector, probability and spatial-temporal data. To ensure factual consistency and reduce hallucinations, it features a probabilistic inference engine that integrates temporal decay with semiring aggregation, resolving long-ter
    
[^238]: TreeWY：面向门控DeltaNet混合模型的自适应验证方法

    TreeWY: Speculative Verification for Gated DeltaNet Hybrids

    [https://arxiv.org/abs/2608.20961](https://arxiv.org/abs/2608.20961)

    本文提出TreeWY方法，通过树状WY变换消除推测解码中的状态快照，显著降低内存开销，从而支持高接受率的宽草稿树。

    

    arXiv:2608.20961v1 公告类型：新论文  摘要：现代开源模型多为混合架构：大部分层采用线性注意力（门控DeltaNet，GDN）层，这些层携带一个小的固定大小的循环状态，而非不断增长的键值（KV）缓存。这使得普通解码在内存上高效，但不利于推测解码。为了验证一批草稿标记并回滚被拒绝的标记，当前系统在GDN层的每个草稿位置都会对完整循环状态进行快照，而这些快照无法在草稿树的分支间共享，因此宽且高接受率的树在内存上变得不可行。我们移除了快照。利用门控Delta规则的一种树状WY变换，我们通过一次三角求解计算每个草稿节点的输出，并在提交时仅重建一个被接受的状态，存储一个小型的伪值矩阵而非逐节点状态；该推导仅依赖于门控Delta规则，而不涉及任何其他架构细节。在两个规模的服务器基准测试中...

    arXiv:2608.20961v1 Announce Type: new  Abstract: Modern open models are hybrids: most layers are linear-attention (Gated DeltaNet, GDN) layers carrying a small fixed-size recurrent state instead of a growing key-value (KV) cache. This makes ordinary decoding memory-efficient, but hurts speculative decoding. To verify a batch of draft tokens and then roll back the rejected ones, today's systems snapshot the full recurrent state at every draft position for GDN layers, and those snapshots cannot be shared across branches of a draft tree, so a wide, high-acceptance tree becomes memory-infeasible. We remove the snapshots. Using a tree-structured WY transform of the gated delta rule, we compute every draft node's output with a single triangular solve and reconstruct only the one accepted state on commit, storing a small pseudo-value matrix instead of per-node states; the derivation depends only on the gated delta rule, not on any other architectural detail. In serving benchmarks on two scale
    
[^239]: 一种用于端到端计算材料发现的LLM智能体

    An LLM agent for end-to-end computational materials discovery

    [https://arxiv.org/abs/2608.20434](https://arxiv.org/abs/2608.20434)

    本文介绍了MAESTRO，一个基于LLM的智能体系统，能自动化执行MOF材料发现的全流程筛选，通过连接异构任务发现传统方法难以识别的高性能材料。

    

    多尺度任务的协调是计算材料发现的有效策略，然而，重复应用多种算法和工具使其具有挑战性。我们报告了MAESTRO，一种大型语言模型（LLM）智能体系统，能够执行金属有机框架（MOFs）的完整筛选流程。它处理大量MOF文献，将相关出版物与其晶体结构关联起来，并将结果整理成可计算数据库，然后通过逐步增加计算成本的策略进行筛选。在湿烟道气条件下分离出的有前景候选材料均来自不相关的研究。通过连接计算材料发现的异构阶段，基于LLM的MAESTRO智能体能够跨应用领域运作，并发现传统筛选方法不太可能发现的高性能材料。

    arXiv:2608.20434v1 Announce Type: cross  Abstract: The coordination of multi-scale tasks is an effective strategy for computational materials discovery, yet the repeated application of diverse algorithms and tools renders it challenging. We report MAESTRO, a large language model (LLM) agent system capable of executing the entire screening pipeline for metal-organic frameworks (MOFs). It processes a large body of MOF literature, links relevant publications to their crystal structures, and curates the results into a computation-ready database, which is then screened through a strategy of progressively increasing computational cost. The promising candidates identified for separation under wet flue gas conditions all originate from unrelated studies. By connecting the heterogeneous stages of computational materials discovery, the LLM-based agents of MAESTRO can operate across application domains and uncover high-performance materials that conventional screening approaches would be unlikely
    
[^240]: FinVerse：金融时间序列基准

    FinVerse: Financial Time-Series Benchmark

    [https://arxiv.org/abs/2608.03259](https://arxiv.org/abs/2608.03259)

    FinVerse是一个金融领域的时间序列预测基准，通过更现实的评估方式（如方向预测）来弥补传统误差指标在真实决策中的不足，包含大量金融时间序列数据。

    

    摘要：随着时间序列基础模型的出现，以有意义的方式评估其预测能力的基准变得越来越重要。现有的时间序列预测基准提供了有用的标准化比较，但它们通常用统一的基于误差的指标来评估异构序列。在此类指标下表现强劲并不一定意味着模型的预测能支持跨领域的最佳现实决策。例如，在股票预测中，正确预测价格是上涨还是下跌可能比仅最小化逐点预测误差更直接地与实现收益相关。为此，我们引入了FinVerse，一个金融领域的时间序列预测基准，朝着更现实的评估迈出了第一步。发布的FinVerse数据工件包含116,897个金融时间序列，共171.1M个观测值，其中60,232个序列被用于基准测试。

    arXiv:2608.03259v2 Announce Type: replace-cross  Abstract: As time-series foundation models have emerged, the need for benchmarks that can evaluate their forecasting ability in meaningful ways has become increasingly important. Existing time-series forecasting benchmarks provide useful standardized comparisons, but they often evaluate heterogeneous series with uniform error-based metrics. Strong performance under such metrics does not necessarily imply that a model's forecasts will support the best real-world decisions across domains. For example, in stock forecasting, correctly predicting whether a price will rise or fall can be more directly relevant to realized returns than minimizing point-wise forecast error alone. To this end, we introduce FinVerse, a finance-domain time-series forecasting benchmark that takes a first step toward more realistic evaluation. The released FinVerse data artifact contains 116,897 financial time series with 171.1M observations, of which 60,232 series w
    
[^241]: AI引导的深度视觉蛋白质组学揭示三阴性乳腺癌中与复发相关的形态-分子特征

    AI-guided deep visual proteomics reveals recurrence-associated morpho-molecular signatures within triple-negative breast cancers

    [https://arxiv.org/abs/2608.03145](https://arxiv.org/abs/2608.03145)

    该研究开发了一种结果导向的AI引导深度视觉蛋白质组学框架，利用复发风险热图指导三阴性乳腺癌的组织采样与质谱空间蛋白质组学分析，揭示了高风险区域富集有丝分裂程序、低风险区域富集免疫与抗原呈递程序的肿瘤内分子异质性。

    

    结果导向的人工智能可以在数字病理学中定位与预后相关的区域，但直接表征这些空间预测背后的分子基础仍然具有挑战性。在此，我们开发了一个结果导向的、AI引导的深度视觉蛋白质组学框架，该框架利用复发风险热图来指导三阴性乳腺癌（TNBC）的实体组织采样和基于质谱的空间蛋白质组学分析。在156名患者中，高风险与低风险区域共存于肿瘤区内，并表现出不同的细胞核与结构特征，表明存在超越广义组织身份的肿瘤内异质性。在这些图谱的指导下，我们从两例复发病例中物理分离并分析了46个AI定义的高风险和低风险肿瘤区域。高风险区域富集有丝分裂相关程序，而低风险区域则显示免疫和抗原呈递相关程序。我们利用在两类区域间持续差异表达的蛋白质……（原文摘要在此处截断）

    arXiv:2608.03145v2 Announce Type: replace  Abstract: Outcome informed AI can localize prognostically relevant regions in digital pathology, but directly characterizing the molecular basis of these spatial predictions remains challenging. Here, we developed an outcome informed, AI-guided deep visual proteomics framework that uses recurrence-risk heatmaps to guide physical tissue sampling and mass spectrometry-based spatial proteomics in triple-negative breast cancer (TNBC). In 156 patients, high- and low-risk regions coexisted within the tumor compartment and showed distinct nuclear and architectural features, indicating intratumoral variation beyond broad tissue identity. Guided by these maps, we physically isolated and profiled 46 AI-defined high- and low-risk tumor regions from two recurrent cases. High-risk regions were enriched for mitotic programs, whereas low-risk regions showed immune- and antigen-presentation-associated programs. We used proteins that consistently differed betw
    
[^242]: MirrorCraft：Minecraft中隐藏规则变化下的配对评估

    MirrorCraft: Paired Evaluation under Hidden Rule Changes in Minecraft

    [https://arxiv.org/abs/2607.29218](https://arxiv.org/abs/2607.29218)

    MirrorCraft提出了一种配对基准测试方法，通过在Mirror世界中悄悄修改服务器端规则、同时保持其他条件与Vanilla世界一致，来评估LLM智能体在Minecraft中面对隐藏规则变化时的适应能力。

    

    arXiv:2607.29218v2 公告类型：替换 摘要：随着大语言模型（LLMs）的快速进展，一个有趣的问题浮现出来：基于LLM的智能体在Minecraft中的表现如何？迄今为止，大多数Minecraft基准测试是在固定的游戏机制下评估智能体，而这些机制在网上有大量的文档记录。在这些设置中取得高表现，并不能说明当熟悉的合成配方、掉落物和其他规则发生变化时，智能体是否还能继续取得进展。在本文中，我们提出了MirrorCraft，一个用于在Minecraft隐藏规则变化下评估智能体的配对基准测试。每个Mirror世界是其配对Vanilla世界的副本，其中一个规则套件的数据包会悄悄修改选定的服务器端规则。在每个Vanilla-Mirror配对中，地形、出生点、资源分布、任务、界面和行动预算均保持一致。MirrorCraft包含五个受控生物群系中的10个Vanilla世界、六个规则套件和三个进度任务，我们评估了两个LLM和六种智能体配置。

    arXiv:2607.29218v2 Announce Type: replace  Abstract: With the rapid progress of large language models (LLMs), an interesting question has emerged: how do LLM-based agents perform in Minecraft? To date, most Minecraft benchmarks evaluate them under fixed game mechanics, which are extensively documented online. High performance in these settings does not show whether an agent can continue making progress when familiar recipes, drops, and other rules change. In this paper, we introduce MirrorCraft, a paired benchmark for evaluating agents under hidden rule changes in Minecraft. Each Mirror world is a copy of its paired Vanilla world, with selected server-side rules silently modified by the datapack of one rule suite. Terrain, spawn, resource placement, task, interface, and action budget remain matched within every Vanilla-Mirror pair. MirrorCraft includes 10 Vanilla worlds in five controlled biomes, six rule suites, and three progression tasks, and we evaluate two LLMs and six agent confi
    
[^243]: 从示例中学习形状：递归SHACL中形状学习的理论基础

    Shapes from Examples: Foundations of Shape Learning in Recursive SHACL

    [https://arxiv.org/abs/2607.27934](https://arxiv.org/abs/2607.27934)

    该论文研究从正负例节点学习递归SHACL形状的拟合问题，针对对应描述逻辑ELI*的SHACL核心片段，在良基、稳定和支持三种目录语义下，为拟合存在性与最具体拟合计算建立了紧凑的指数时间上界，并在特殊情形下给出了多项式界。

    

    SHACL形状能够实现数据图的验证，因此自动形状学习对知识图谱应用至关重要。我们研究了该任务中广为人知的拟合方法：给定输入图中正例节点集合P和负例节点集合N，计算一个形状表达式C（其可使用递归形状目录中定义的形状名称），使得C在P中的每个节点上都验证通过，而在N中的所有节点上均验证失败。我们关注C由SHACL核心片段（对应于描述逻辑ELI*）书写的情形。对于形状目录，我们考虑了良基、稳定和支持三种语义。我们研究了拟合的存在性与最具体拟合的计算问题，为这两个问题建立了紧凑的指数时间上界，并针对相关特殊情形获得了多项式界。

    arXiv:2607.27934v3 Announce Type: replace  Abstract: SHACL shapes enable data graph validation, making automatic shape learning essential for knowledge graph applications. We investigate the well-known fitting approach to this task: given sets P and N of positive and negative example nodes from an input graph, compute a shape expression C, possibly using shape names defined in a recursive shape catalogue, that validates at every node in P and none in N. We focus on the case where C is written in a core fragment of SHACL corresponding to the Description Logic ELI*. For the catalogue, we consider the well-founded, stable, and supported semantics. We address fitting existence and most specific fitting computation, establish tight exponential-time upper bounds for both problems, and obtain polynomial bounds for relevant special cases.
    
[^244]: AgentGUI：用于观察和引导长时运行AI代理的界面

    AgentGUI: An Interface for Observing and Steering Long-Running AI Agents

    [https://arxiv.org/abs/2607.26300](https://arxiv.org/abs/2607.26300)

    AgentGUI是一个本地托管的图形界面，通过丰富的轨迹可视化、手动与自动引导以及多框架集成协调，显著提升了人类对长时运行AI代理的监督效率（轨迹分析提速38%）和小型本地代理的任务完成率（最高提升34个百分点）。

    

    AI代理在处理复杂、长时间运行的任务方面日益熟练。随着自主能力的快速增长，由于缺乏以人为中心的交互界面，人类监督正系统性地滞后。为解决这一问题，我们提出了AgentGUI，一个用户友好、本地托管的图形界面，用于在多个并发的长时间运行会话中无缝观察和引导AI代理。AgentGUI具有以下特点：1）丰富的代理轨迹可视化，2）有效的手动和自动引导功能，3）与开源及前沿代理框架的集成与协调。一项受控用户研究表明，从代理轨迹中识别关键要素的时间有统计学显著性的减少（快38%，p = 0.023）。在一项初步实验中，AgentGUI的自动防偏航功能在0.8B至9B的模型规模梯度上（N=50次运行）将小型本地代理的任务完成率提高了高达34个百分点。

    arXiv:2607.26300v3 Announce Type: replace-cross  Abstract: AI agents are increasingly adept at tackling complex, long-running tasks. With the rapid surge of autonomous capabilities, human oversight is systematically lagging behind due to limited human-centered interfacing. Aiming to address this, we introduce AgentGUI, a user-friendly, locally hosted GUI for seamlessly observing and steering AI agents amid multiple concurrent, long-running sessions. AgentGUI features 1) rich agent trajectory visualizations, 2) effective manual and automated steering, and 3) integration with and coordination between open-source and frontier agent frameworks. A controlled user study demonstrates statistically significant reduction in the time it takes to identify key elements from agent traces (38% faster, p = 0.023). In a preliminary experiment, AgentGUI's automated drift prevention feature raises the task completion rate of small local agents by as high as 34pp across a 0.8B--9B model ladder (N=50 runs
    
[^245]: LLM错误中跨问题结构的来源：共享随机种子、共享提示词与答案字母偏好

    Where Cross-Question Structure in LLM Errors Comes From: Shared Seeds, Shared Prompts, and Answer-Letter Preference

    [https://arxiv.org/abs/2607.20464](https://arxiv.org/abs/2607.20464)

    该论文通过问题相关性矩阵的特征值分析发现，LLM错误中的跨问题结构主要来自共享随机种子、共享提示词以及对答案字母的偏好（主要由Llama-3.2-1B驱动），而非共享的知识缺失。

    

    语言模型之间的分歧通常被解读为知识缺失，那么需要同样缺失知识的问题就应当同时答对或答错。我们利用问题相关性矩阵的特征值来检验这种结构。在单个模型内部，独立抽取的答案没有可检测的结构；但当一轮基准测试中的所有问题共享同一随机种子或少样本提示词时，这种结构便会出现。在24个模型中，超越总体能力的最大结构是对答案字母的偏好：当选项被旋转时，它跟随字母而非问题本身，并且这一现象依赖于单一模型Llama-3.2-1B；去掉它之后只剩较小的超额结构。只有当问题在所有选项旋转中都答对才计为正确，可以去除作用于猜测的那部分。剩下的仅在5%显著性水平上存在边际超额，事后可归因于两对近乎重复的模型以及Llama-3.2-1B。

    arXiv:2607.20464v2 Announce Type: replace  Abstract: Disagreement between language models is often read as missing knowledge. Questions that need the same missing knowledge should then be right or wrong together. We test for this structure with the eigenvalues of the questions' correlation matrix. Within one model, independently drawn answers show no detectable structure; it appears when all questions in one pass over the benchmark share a random seed or few-shot prompt. Across 24 models, the largest structure beyond overall skill is a preference among answer letters: when the options are rotated, it follows the letter, not the question, and it depends on one model, Llama-3.2-1B; without it only a smaller excess remains. Counting a question correct only if right in every rotation removes the part that acts on guesses. What remains is a marginal excess at the 5% line, tied post hoc to two near-duplicate model pairs and to Llama-3.2-1B.
    
[^246]: JANUS：面向长程智能体安全的潜在风险预见

    JANUS: Foreseeing Latent Risk for Long-Horizon Agent Safety

    [https://arxiv.org/abs/2607.19913](https://arxiv.org/abs/2607.19913)

    Janus提出前瞻式智能体安全框架，通过预判任务与裁决任务的联合强化学习训练出守护模型Vanguard，能够在不安全操作执行前预见延迟风险，使平均防护提升15.9个百分点、良性任务完成率提升5.1个百分点。

    

    智能体安全正从内容审核转向在工具使用智能体行动之前预防操作性失误。我们提出Janus，一个面向长程智能体安全的前瞻性框架，它训练守护模型从部分轨迹中预判延迟显现的风险。Janus通过多智能体模拟合成多样化的智能体轨迹，并以两个耦合任务学习一个共享策略：一个是预测与安全相关未来的预判任务，另一个是综合观察到的前缀与预判的未来来判定安全性的裁决任务。这两个任务通过CoAA-RL进行联合优化，该方法根据预测对下游安全判断的效用来给予奖励。由此得到的守护模型Vanguard能够在不安全操作执行之前将其拦截。在四个智能体安全基准测试中，Vanguard相比基线守护模型将平均防护能力提升了15.9个百分点，同时将良性任务完成率提升了5.1个百分点。

    arXiv:2607.19913v2 Announce Type: replace  Abstract: Agent safety is moving from content moderation toward preventing operational failures before tool-using agents act. We propose Janus, a foresight-oriented framework for long-horizon agent safety that trains guards to anticipate delayed risks from partial trajectories. Janus synthesizes diverse agent trajectories via multi-agent simulation and learns a shared policy with two coupled tasks: an anticipation task that forecasts safety-relevant futures and an adjudication task that decides safety from both the observed prefix and anticipated future. The two tasks are jointly optimized with CoAA-RL, which rewards forecasts by their utility for downstream safety judgment. The resulting guard model, Vanguard, blocks unsafe actions before execution. Across four agent-safety benchmarks, Vanguard improves average protection by 15.9 percentage points over baseline guards while increasing benign task completion by 5.1 percentage points.
    
[^247]: DeepSafe-SciHazard：一个采用分解式危害评分衡量科学安全风险的基准测试

    DeepSafe-SciHazard: A Benchmark for Measuring Scientific Safety Risks with Decomposed Harm Scoring

    [https://arxiv.org/abs/2607.18665](https://arxiv.org/abs/2607.18665)

    该论文提出了SciHazard，一个基于真实世界的科学安全风险基准，包含跨12个学科的2400个危险问题和600个过度安全问题，并开发了分解式的DeHarm-Score评估框架，通过综合查询危害严重程度、拒绝行为和响应层面风险来更准确地衡量大语言模型的科学安全风险。

    

    大语言模型（LLM）日益支持科学研究，但它们也可能将危险的科学知识转化为可操作的错误使用指导。现有基准测试往往依赖于与真实世界危害脱节的模板化查询，并采用缺乏领域依据的LLM-as-a-Judge评估范式。为解决这些问题，我们提出了SciHazard，这是一个以真实世界为基础的科学风险基准测试，以及一个与数据集无关的危害性评估框架。SciHazard包含跨12个学科的2400个危险问题和600个过度安全问题，两类查询均基于受监管的实体和有记录的故障场景。为计算DeHarm-Score，我们开发了一种分解式评估程序，该程序结合了查询危害严重程度、拒绝行为和响应层面的风险。对于未被拒绝的响应，该方法进一步将响应层面的危害分解为可执行性，并通过动态清单进行量化……

    arXiv:2607.18665v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly support science, but they can also convert hazardous scientific knowledge into actionable misuse guidance. Existing benchmarks often rely on templated queries disconnected from real-world hazards, and employ LLM-as-a-Judge paradigms without domain grounding. To address this, we introduce SciHazard, a real-world-grounded benchmark for scientific risks and a dataset agnostic evaluation framework for measuring harmfulness. SciHazard contains 2400 hazardous questions and 600 oversafety questions across 12 disciplines, with both queries grounded in regulated entities and documented failure scenarios. To compute \textsc{DeHarm-Score} , we develop a decomposed evaluating procedure that combines query hazard severity, refusal behavior, and response-level risk. For non-refused responses, it further decomposes response-level harm into \textsc{Executability}, quantified via dynamic checklists with impor
    
[^248]: 面向机器人动作表示的语义锚定方法

    Semantic Anchoring for Robotic Action Representations

    [https://arxiv.org/abs/2607.13597](https://arxiv.org/abs/2607.13597)

    该论文发现VLA模型微调会破坏动作表示的语义结构并损害泛化能力，据此提出一种即插即用的语义锚定方法，通过将动作表示锚定到语义流形并分解为共享语义通道与私有通道，有效提升任务成功率与分布外泛化性能。

    

    视觉-语言-动作（VLA）模型从预训练的视觉-语言模型中继承了丰富的语义表示，然而在有限的机器人演示数据上进行微调会破坏这种结构并削弱泛化能力。因此，一个根本性的问题随之而来：什么才是好的动作表示？受镜像神经元理论“观察与执行共享意图级编码”这一洞察的启发，我们研究了机器人的动作表示是否能够保留预训练编码器所捕获的语义结构。系统性探测实验证实，这种结构在微调过程中会逐渐退化，且其质量与任务成功率和分布外泛化能力同步变化。我们进一步提出了一种即插即用的方法，将动作表示锚定到语义流形上，同时将表示分解为共享语义通道和私有通道，这些模块在推理阶段全部被丢弃，从而保持部署模型不受影响……

    arXiv:2607.13597v3 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models inherit rich semantic representations from pretrained Vision-Language Models, yet fine-tuning on limited robot demonstrations degrades this structure and undermines generalization. A fundamental question therefore arises: what constitutes a good action representation? Inspired by the mirror neuron theory's insight that observation and execution share an intention-level encoding, we examine whether a robot's action representations preserve the semantic structure captured by pretrained encoders. Systematic probing confirms that this structure erodes during finetuning, and that its quality synchronizes with both task success and out-of-distribution generalization. We further introduce a plug-and-play method that anchors action representations to a semantic manifold while decomposing representations into a shared semantic channel and a private channel, all discarded at inference, leaving the depl
    
[^249]: 我如何知道接下来该说什么？巴伦霍尔茨的自生成理论对哈里斯整合主义的丰富

    How Do I Know What to Say Next? Barenholtz's Autogenerative Theory as an Enrichment of Harrisean Integrationism

    [https://arxiv.org/abs/2607.07891](https://arxiv.org/abs/2607.07891)

    本文提出巴伦霍尔茨基于大语言模型行为发展出的语言自生成理论，能够填补哈里斯整合主义语言学在符号未来开放性机制、语言与非语言符号活动连续性以及历史整合档案结构属性方面的解释空白，从而在不违背其核心主张的前提下丰富整合主义理论。

    

    罗伊·哈里斯的整合主义语言学对深植于语言计算方法核心的指称主义传统提出了有力的批判，认为语言并非一种映射到预先给定世界的代码，而是一种面向未来共同行动的、情境化的双向活动。然而，整合主义存在某些解释上的空白：它未能充分阐明符号维持未来开放性的结构机制，对语言与非语言符号活动之间的连续性理论化不足，并且对过去整合活动所积累档案的结构属性缺乏详细论述。本文认为，伊兰·巴伦霍尔茨针对大语言模型（LLM）行为而发展的语言自生成理论，恰好能够填补这些空白，在不损害整合主义任何核心主张的前提下丰富其理论内涵。

    arXiv:2607.07891v2 Announce Type: replace-cross  Abstract: Roy Harris's Integrationist linguistics offers a compelling critique of the referentialist tradition embedded deep at the heart of computational approaches to language, arguing that language is not a code that maps onto a pre-given world but a situated, bipartite activity oriented toward prospective joint action. Yet Integrationism leaves certain explanatory gaps: it does not fully account for the structural mechanism by which signs sustain prospective openness, it undertheorises the continuity between linguistic and non-linguistic semiotic activity, and it offers no detailed account of the structural properties of the accumulated archive of past integrations. This paper argues that Elan Barenholtz's autogenerative theory of language, developed in response to the behaviour of Large Language Models (LLMs), can fill precisely these gaps, enriching Integrationism without undermining any of its core commitments. Specifically, the a
    
[^250]: WAM-TTT：通过在测试时观看人类玩耍来引导世界-动作模型

    WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time

    [https://arxiv.org/abs/2607.06988](https://arxiv.org/abs/2607.06988)

    WAM-TTT提出了一种测试时训练框架，通过元训练使冻结的世界动作模型内获得轻量级自适应记忆，在测试时仅需观看无标注的人类视频即可高效引导机器人行为，无需机器人演示、人类标注或任务特定微调。

    

    引导机器人基础模型（RFMs）适应新的任务变体或用户偏好的行为仍然具有挑战性，通常需要额外的机器人演示、任务特定的微调或长上下文条件化。我们提出了WAM-TTT，这是一个从原始人类视频出发对世界动作模型进行引导的测试时训练框架。WAM-TTT并不将人类视频视为需要模仿的轨迹，而是通过自监督视频预测将其吸收到冻结的WAM内部的轻量级自适应记忆中。为了使这种记忆对控制有用，我们引入了一个元训练阶段，利用成对的人-机器人数据和键值记忆重建目标，将人类演示与机器人行为对齐。在测试时，仅需无标注的人类视频即可对记忆进行适配，而预训练的WAM保持冻结。这使得无需机器人动作、人类端标注或任务特定的微调，即可实现高效且可复用的行为引导。

    arXiv:2607.06988v3 Announce Type: replace-cross  Abstract: Steering robot foundation models (RFMs) toward new task variants or user-preferred behaviors remains challenging, often requiring additional robot demonstrations, task-specific fine-tuning, or long-context conditioning. We present WAM-TTT, a test-time training framework for steering world action models from raw human videos. Rather than treating human videos as trajectories to imitate, WAM-TTT absorbs them into a lightweight adaptive memory inside a frozen WAM through self-supervised video prediction. To make this memory useful for control, we introduce a meta-training stage that aligns human demonstrations with robot behaviors using paired human-robot data and a key--value memory reconstruction objective. At test time, only unlabeled human videos are required to adapt the memory, while the pretrained WAM remains frozen. This enables efficient and reusable steering without robot actions, human-side annotations, or task-specific
    
[^251]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^252]: Transformer应如何在电子健康记录中编码数值？

    How Should Transformers Encode Numeric Values in Electronic Health Records?

    [https://arxiv.org/abs/2607.01391](https://arxiv.org/abs/2607.01391)

    该论文系统比较了Transformer处理电子健康记录数据时离散、连续和混合三种数值编码策略，发现显式建模数值-概念交互的方法在精度敏感任务上最优，而投影前分箱的混合方法更为稳健通用，且最优分箱数与数据集规模呈幂律关系。

    

    在基于Transformer的序列处理中，我们应如何对数值进行编码，尤其是在电子健康记录（EHR）数据中？我们利用嵌入真实世界EHR数据中的合成算术任务以及真实临床预测任务，系统地比较了离散、连续和混合数值编码策略。我们的研究揭示了数值精度、优化稳定性与架构灵活性之间的权衡。我们发现，在架构约束允许的情况下，显式建模数值-概念交互的方法在精度敏感的算术任务上表现最佳。而保留数值但在投影前进行分箱的混合标记化方法则提供了一种更稳健且适用范围更广的替代方案，其最优分箱数量遵循一个简单的基于经验推导的、与数据集规模相关的幂律。在各类任务中，模型始终表现出可靠且“足够好”的数值计算能力。

    arXiv:2607.01391v2 Announce Type: replace-cross  Abstract: How do we encode numeric values in transformer-based sequence processing, particularly in electronic health record (EHR) data? We systematically compare discrete, continuous, and hybrid value encoding strategies using synthetic arithmetic tasks embedded within real-world EHR data, as well as real-world clinical prediction tasks. Our study reveals trade-offs between numeric precision, optimisation stability, and architectural flexibility. We find that approaches that explicitly model value-concept interactions perform best on precision-sensitive arithmetic tasks when architectural constraints permit. Hybrid token-based approaches that retain numeric values but apply binning prior to projection provide a more robust and broadly applicable alternative, with the optimal number of bins following a simple empirically derived power-law in dataset size. Across tasks, models consistently exhibit reliable "good enough" numeric computatio
    
[^253]: 从运动中生成世界：基于单目视频的生成式动态高斯重建

    World from Motion: Generative Dynamic Gaussian Reconstruction from Monocular Video

    [https://arxiv.org/abs/2607.01202](https://arxiv.org/abs/2607.01202)

    提出 World from Motion 方法，利用视频生成模型修正并补全单目视频的动态 3D 高斯重建，将生成结果蒸馏回统一的高质量动态 3DGS，在 4D 重建与新视角合成上达到新的最先进水平。

    

    我们提出了 World from Motion，一种从单目视频生成可自由渲染的动态 3D 高斯表示的方法。我们的方法以密集的、像素对齐的渲染结果为条件驱动视频模型，这些渲染编码了外观、几何信息以及沿输入与目标相机轨迹的 3D 场景运动，用于纠正渲染伪影并填补初始重建中缺失的区域。为训练该模型，我们构建了一个由对齐的多视角视频对和动态 3DGS 表示组成的数据集，其中模拟了单目重建特有的伪影。在测试阶段，我们将模型的生成结果（包括新观测到的区域和运动）蒸馏回单个一致且高质量的动态 3DGS 中，从而同时提升新视角合成质量和底层 3D 运动的准确性。我们的方法在 4D 重建领域创造了新的最先进水平，并能无缝泛化到具有大视角变化和动态场景的真实世界视频中。

    arXiv:2607.01202v2 Announce Type: replace-cross  Abstract: We present World from Motion, a method for generating freely renderable dynamic 3D Gaussian representations from monocular videos. Our approach conditions a video model on dense, pixel-aligned renderings that encode appearance, geometry, and 3D scene motion along both input and target camera trajectories to correct rendering artifacts and fill in missing regions from an initial reconstruction. To train this model, we construct a dataset of aligned multiview video pairs and dynamic 3DGS representations, with simulated artifacts characteristic of monocular reconstruction. At test time, we distill the model's generations, including newly observed regions and motions, back into a single consistent, high-quality dynamic 3DGS, improving both novel-view synthesis and the underlying 3D motion. Our method sets a new state of the art in 4D reconstruction and seamlessly generalizes to in-the-wild videos with large viewpoint changes and dy
    
[^254]: SteerVTE：具有风格与字形控制的无缝视频文本编辑

    SteerVTE: Seamless Video Text Editing with Style and Glyph Control

    [https://arxiv.org/abs/2606.23254](https://arxiv.org/abs/2606.23254)

    SteerVTE通过风格编码器与行级、字符级双粒度字形编码器引导冻结的视频扩散模型，实现了兼具风格一致性与笔画级精度的无缝视频文本编辑。

    

    视觉文本编辑旨在精确修改图像和视频中的文本，同时保持风格一致性与视觉真实性。尽管在图像领域已取得显著进展，视频文本编辑在很大程度上仍处于未探索状态：这是一项局部化任务，要求在小文本区域内达到笔画级精度，这进一步加剧了跨帧准确性、时序一致性和风格保真度方面的挑战。我们提出SteerVTE，一个统一框架，通过风格与字形控制来引导冻结的视频扩散模型执行精确的视频文本编辑。SteerVTE构建于冻结的扩散变换器之上，附加了一个轻量级文本上下文适配器，其中包含两个互补模块：一个捕捉原始文本视觉属性的风格编码器，以及以行级和字符级双粒度编码目标文本的双粒度字形编码器。为克服视频基础模型固有的文本渲染先验较弱的问题……

    arXiv:2606.23254v2 Announce Type: replace-cross  Abstract: Visual text editing aims to precisely modify text in images and videos while preserving stylistic consistency and visual realism. Despite significant advances in the image domain, video text editing remains largely unexplored: it is a localized task demanding stroke-level precision within small text regions, which compounds the challenges of cross-frame accuracy, temporal coherence, and stylistic fidelity. We introduce SteerVTE, a unified framework that steers a frozen video diffusion model to perform precise Video Text Editing through style and glyph control. Built on a frozen diffusion transformer, SteerVTE attaches a lightweight text context adapter with two complementary modules: a style encoder capturing the original text's visual attributes, and dual-granularity glyph encoders encoding the target text at both the line and character levels. To overcome the inherently weak text rendering priors of video foundation models, w
    
[^255]: 吸引子域理论：一种经可穿戴光电容积脉搏波（PPG）验证的心血管吸引子分析数学框架

    Attractor Domain Theory: A Mathematical Framework for Cardiovascular Attractor Analysis with Wearable Photoplethysmography (PPG) Validation

    [https://arxiv.org/abs/2606.22039](https://arxiv.org/abs/2606.22039)

    该论文提出吸引子域理论（ADT），通过延迟重构、遍历与变分等相互关联的数学变换，为心血管吸引子动力学构建了一种包含几何域、遍历域和变分域的三表示算子架构，并以可穿戴光电容积脉搏波（PPG）数据进行了验证。

    

    心脏吸引子提供了在标准延迟嵌入条件下由标量观测重构的心血管动力学的相空间表示。我们提出吸引子域理论，该理论通过一组相互关联的数学变换揭示重构动力学的不同性质：几何域通过延迟重构变换获得，而遍历变换和变分变换则作用于该几何，用以刻画长时程组织结构与局部有限时间形变。因此，ADT为重构动力学规定了一种三表示的算子架构，而非对现有非线性特征进行回顾性归类；该架构被提出作为一种操作性组织框架，而非唯一性或穷尽性定理。条件域分解通过显式残差量化特定端点的增量信息，以及一个域轨……（摘要在此处截断）

    arXiv:2606.22039v3 Announce Type: replace  Abstract: The cardiac attractor provides a phase-space representation of cardiovascular dynamics reconstructed from scalar observations under standard delay-embedding conditions. We introduce Attractor Domain Theory (ADT), a theory in which linked mathematical transformations expose distinct properties of reconstructed dynamics: the Geometry Domain is obtained through the delay-reconstruction transformation, while Ergodic and Variational transformations act on that geometry to characterize long-horizon organization and localized finite-time deformation. Thus, ADT specifies a three-representation operator architecture for reconstructed dynamics rather than retrospectively grouping existing nonlinear features; the architecture is proposed as an operational organization, not as a uniqueness or exhaustiveness theorem. A Conditional Domain Decomposition quantifies endpoint-specific incremental information with an explicit residual, and a domain-tra
    
[^256]: AlphaMemo：面向自进化Alpha挖掘智能体的结构化搜索过程记忆

    AlphaMemo: Structured Search-Process Memory for Self-Evolving Alpha Mining Agents

    [https://arxiv.org/abs/2606.20625](https://arxiv.org/abs/2606.20625)

    AlphaMemo通过从AST差异中提取编辑模式、置信度门控残差记忆和非对称否决控制构建结构化搜索过程记忆，实现了自进化的Alpha挖掘智能体，在中证500和标普500上显著提升了样本外表现和固定预算下的因子发现效率。

    

    LLM智能体通过结合金融先验知识、符号推理、可执行因子生成和反馈驱动的迭代改进，在Alpha挖掘领域展现出广阔前景。然而，它们面临着组合爆炸式的搜索空间、嘈杂且非平稳的反馈、冗余的因子发现，以及简单复用历史成功经验所带来的过拟合风险。为了应对这些挑战，我们提出了AlphaMemo，一个具有结构化搜索过程记忆的自进化Alpha挖掘智能体。与仅仅记忆最终因子或完整轨迹不同，AlphaMemo记录的是可复用的证据，即哪些编辑模式在特定的父因子上下文中有效或失败。它从抽象语法树（AST）差异中提取编辑模式，在搜索账本先验之上应用置信度门控的残差记忆，并采用非对称否决控制机制来抑制高置信度的失败模式。在中证500和标普500指数上的实验表明，该方法的样本外表现和固定预算下的因子发现效率均得到提升。

    arXiv:2606.20625v2 Announce Type: replace  Abstract: LLM agents are promising for alpha mining via combining financial priors, symbolic reasoning, executable factor generation, and feedback-driven refinement. Yet, they face a combinatorial search space, noisy non-stationary feedback, redundant discoveries, and overfitting risks from naively reusing past successes. To address these challenges, we propose AlphaMemo, a self-evolving alpha mining agent with Structured Search-Process Memory. Rather than memorizing only final factors or full trajectories, AlphaMemo records reusable evidence about which edit motifs work or fail under specific parent-factor contexts. It extracts motifs from Abstract Syntax Tree (AST) differences, applies confidence-gated residual memory on top of a search-ledger prior, and uses asymmetric veto control to suppress high-confidence failure patterns. Experiments on CSI 500 and S&P 500 show improved out-of-sample performance and fixed-budget discovery efficiency, w
    
[^257]: DeFAb：一个通过可废止溯因实现有据科学发现的数据集

    DeFAb: A Dataset for Grounded Scientific Discovery through Defeasible Abduction

    [https://arxiv.org/abs/2606.18557](https://arxiv.org/abs/2606.18557)

    该论文提出了基于基因本体构建的DeFAb数据集与符号验证框架，用于评估模型通过可废止溯因修订科学理论的能力，并揭示了现有修订测试无法检验例外泛化性的局限性。

    

    一个理论可以通过添加例外来容纳反例，而更难的问题在于决定例外应该扩展到多大范围，以及哪些预测应当得以保留。我们提出了DeFAb，这是一个基于基因本体（Gene Ontology）中经过整理的生物学知识构建的数据集和符号验证框架。模型需要填补理论中的空白，或对其规则提出例外。验证器会检查每次修改的后果，允许多种不同的解决方案获得认可。简单的符号修复即可解决已发布评估面板中的全部336个任务。这一结果暴露了修订测试的一个局限性：与单个实体绑定的例外即使没有识别出适用于其他情况的模式，也可能成功通过测试。为此，我们开发了一个情境化任务，要求所提出的例外能够触及相关案例，同时在其他地方保留原有的预测。我们计划开展两项研究，分别比较使用经验证和被拒绝的示例进行训练的效果，以及修订过程中有和没有后果反馈的差异。

    arXiv:2606.18557v2 Announce Type: replace  Abstract: A theory can accommodate a counterexample by adding an exception. The harder problem is deciding how far the exception should extend and which predictions should survive. We present DeFAb, a dataset and symbolic verification framework built from curated biological knowledge in the Gene Ontology. Models fill gaps in a theory or propose exceptions to its rules. A verifier checks the consequences of each edit, allowing different solutions to receive credit. Simple symbolic repairs solve all 336 tasks in the released evaluation panel. This result exposes a limitation of the revision test. An exception tied to one entity can succeed without identifying a pattern that applies to other cases. We develop a contextual task that requires an exception to reach the relevant cases and preserve predictions elsewhere. Two planned studies compare training with verified and rejected examples, and revision with and without feedback about consequences.
    
[^258]: 大型音频语言模型的连续音频思考

    Continuous Audio Thinking for Large Audio Language Models

    [https://arxiv.org/abs/2606.18273](https://arxiv.org/abs/2606.18273)

    提出连续音频思考（CoAT）框架，通过音频专家蒸馏为大型音频语言模型构建连续潜在思考空间，使其在生成响应前能够组织并利用语音细节、韵律、声音事件、情感和音高等丰富声学信息。

    

    大型音频语言模型（LALMs）在从语音转录到音乐分析等多种音频理解任务上展现出令人印象深刻的能力。然而，由于LALMs通常被训练以生成与文本对齐的响应，其隐藏状态逐渐被塑造为服务于文本生成而非保留声学信息。因此，音频所承载的丰富声学内容，如语音细节、韵律、声音事件、情感和音高，在这一过程中逐渐丢失，难以在响应中加以利用。我们提出了连续音频思考，这是一个为音频语言模型配备连续潜在工作空间的框架，用于在生成响应之前组织声学信息，并通过音频专家的蒸馏进行支撑。在思考空间内，模型可以在生成响应时利用专家蒸馏所提供的丰富声学信息。

    arXiv:2606.18273v2 Announce Type: replace-cross  Abstract: Large audio language models (LALMs) have shown impressive capabilities on diverse audio understanding tasks, ranging from speech transcription to music analysis. However, because LALMs are typically trained to produce text-aligned responses, their hidden states are progressively shaped for text generation rather than for preserving acoustic information. As a result, the diverse acoustic content that audio carries, such as phonetic detail, prosody, sound events, affect, and pitch, is lost along the way and difficult to leverage in the response. We introduce Continuous Audio Thinking (CoAT), a framework that equips audio language models with a continuous latent workspace for organizing acoustic information prior to response generation, grounded by distillation from audio experts. Within the thinking space, the model can utilize the rich acoustic information provided by expert distillation when generating its response. Furthermore
    
[^259]: 信任正确的教师：面向GUI定位的质量感知自蒸馏

    Trust the Right Teacher: Quality-Aware Self-Distillation for GUI Grounding

    [https://arxiv.org/abs/2606.18101](https://arxiv.org/abs/2606.18101)

    提出面向GUI定位的质量感知自蒸馏方法，通过软门控对教师坐标预测进行空间正确性验证并动态降低不可靠教师信号的权重，从而提升坐标预测的准确性。

    

    GUI定位要求视觉语言模型（VLM）在高分辨率截图中识别微小的目标元素并预测精确的屏幕坐标。在策略自蒸馏（OPSD）是这一坐标敏感任务的一种有前景的后训练方法，因为它能提供超越硬坐标标签的密集token级教师信号。然而，朴素的OPSD并不适合GUI定位：由于教师模型是在学生模型生成的前缀上进行评估的，一旦前缀偏离目标坐标，其坐标token信号就可能变得不可靠。我们提出了一种面向GUI定位的质量感知自蒸馏方法，其核心是对教师预测进行直接的空间正确性验证。一个软门控机制会检查教师模型的首要坐标token预测在学生生成的前缀下是否仍能被补全到真实边界框内，并在该检查失败时降低该信号的权重。教师概率缩放……（原文截断）

    arXiv:2606.18101v3 Announce Type: replace  Abstract: GUI grounding requires vision-language models (VLMs) to identify small target elements in high-resolution screenshots and predict precise screen coordinates. On-policy self-distillation (OPSD) is a promising post-training approach for this coordinate-sensitive task because it provides dense token-level teacher signals beyond hard coordinate labels. However, naive OPSD is not well-suited to GUI grounding: since the teacher is evaluated on student-generated prefixes, its coordinate-token signals can become unreliable once a prefix deviates from the target coordinate. We propose quality-aware self-distillation for GUI grounding, centered on direct spatial correctness verification of teacher predictions. A soft gate checks whether the teacher's top coordinate-token prediction can still be completed into the ground-truth box under the student-generated prefix and down-weights the signal when this check fails. Teacher-probability scaling t
    
[^260]: 架构感知的强化学习使滑动窗口注意力在数学推理中具备竞争力

    Architecture-Aware Reinforcement Learning Makes Sliding-Window Attention Competitive in Math Reasoning

    [https://arxiv.org/abs/2606.11634](https://arxiv.org/abs/2606.11634)

    该论文提出SWARR两阶段方法——先用监督微调将预训练自注意力模型高效转换为滑动窗口注意力模型，再用on-policy强化学习适配数据与架构的不匹配——使滑动窗口注意力在数学推理任务上具备与自注意力相当的竞争力，同时缓解长上下文推理的二次方计算开销。

    

    推理型与智能体式大语言模型（LLM）的快速发展增加了对长上下文推理的需求，但自注意力（SA）的计算开销随上下文长度呈二次方增长。为解决这一问题，我们研究了SWARR（面向数学推理的强化自适应滑动窗口注意力），这是一套将SWA模型适配到数学推理的实用方案。SWARR包含两个阶段：（1）通过监督微调（SFT）将预训练的SA模型高效转换为SWA模型，从而避免重新预训练新的基础模型；（2）通过强化学习（RL）进行策略适配。我们发现，经过SFT后SWA的性能仍不及SA，我们假设这一差距部分源于数据与架构的不匹配：大多数SFT数据是为SA模型准备的，可能包含SWA难以建模的长程依赖关系。由于on-policy强化学习在SWA约束下优化模型自生成的轨迹，它可以使轨迹适应（原文在此处截断）……

    arXiv:2606.11634v2 Announce Type: replace  Abstract: The rapid progress of reasoning and agentic large language models (LLMs) has increased the demand for long-context inference, but self-attention (SA) scales quadratically with context length.   To address this, we study SWARR (Sliding-Window Attention with Reinforced Adaptation for Math Reasoning), a practical recipe for adapting SWA models to mathematical reasoning. SWARR has two stages: (1) efficient conversion from a pretrained SA model to SWA with supervised fine-tuning (SFT), which avoids pretraining a new base model, and (2) policy adaptation with reinforcement learning (RL).   We find that SWA still underperforms SA after SFT, and we hypothesize that this gap is caused in part by a data-architecture mismatch: most SFT data are prepared for SA models and may contain long-range dependencies that are difficult for SWA to model. Because on-policy RL optimizes self-generated trajectories under the SWA constraint, it can adapt traje
    
[^261]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^262]: SkillPyramid：面向自进化智能体的层次化技能巩固框架

    SkillPyramid: A Hierarchical Skill Consolidation Framework for Self-Evolving Agents

    [https://arxiv.org/abs/2606.03692](https://arxiv.org/abs/2606.03692)

    SkillPyramid提出了一种基于层次化技能拓扑的技能巩固框架，通过自进化机制使智能体在任务执行中组合、验证并积累新技能，从而实现技能经验的复用与更广泛的任务泛化。

    

    最近的AI智能体能够灵活地调用技能来解决复杂任务，但其长期改进从根本上受限于缺乏系统化的技能构建、积累与迁移机制。特别是，在没有统一的技能巩固框架的情况下，智能体往往会在不同任务中冗余地构建相似的能力，无法有效地将经验转化为可复用的资产，并且难以将特定任务的技能泛化到新场景中。为了解决这一局限，我们提出了SkillPyramid，一个能够复用现有技能经验以实现更广泛任务泛化的技能巩固框架。SkillPyramid在层次化的技能拓扑结构上运行，并进一步引入了一种自进化机制，使智能体能够在任务执行过程中组合、验证并纳入新技能。在ALFWorld、WebShop和ScienceWorld三个基准上、跨越四个骨干模型的实验表明，SkillPyramid显著……（原文摘要至此处被截断）

    arXiv:2606.03692v2 Announce Type: replace  Abstract: Recent AI agents can flexibly invoke skills to solve complex tasks, but their long-term improvement is fundamentally constrained by a lack of systematic skill construction, accumulation, and transfer. In particular, without a unified framework for skill consolidation, agents tend to redundantly construct similar capabilities across different tasks, are unable to effectively transform experience into reusable assets, and struggle to generalize task-specific skills to novel scenarios. To address this limitation, we propose SkillPyramid, a skill consolidation framework that reuses existing skill experience for broader task generalization. Operating on a hierarchical skill topology, SkillPyramid further introduces a self-evolution mechanism that enables agents to compose, validate, and incorporate new skills during task execution. Experiments on ALFWorld, WebShop, and ScienceWorld across four backbone models show that SkillPyramid substa
    
[^263]: 迭代纳什偏好优化的高效探索

    Efficient Exploration for Iterative Nash Preference Optimization

    [https://arxiv.org/abs/2606.01382](https://arxiv.org/abs/2606.01382)

    论文提出探索式纳什偏好优化（ENPO），通过SFT型正则化与对抗性探索机制，解决了迭代NLHF中隐式探索不足导致的KL正则参数指数级依赖问题，为在线迭代纳什学习提供了理论保证。

    

    偏好对齐是提升大语言模型（LLM）性能的核心，但基于奖励的建模方式在人类偏好具有非传递性时会受到限制。从人类反馈中进行纳什学习（NLHF）通过将对齐建模为偏好博弈并求解纳什均衡来解决这一局限。然而，可扩展NLHF的学习理论基础仍然有限：现有的遗憾保证依赖于显式的偏好模型估计和极小极大预言机，而更简单的迭代方法则缺乏此类保证。我们研究在线迭代NLHF，并发现探索是其中的关键障碍。首先，我们证明标准的迭代NLHF可能会对KL正则化参数的倒数产生指数级依赖，这表明通过策略更新实现的隐式探索可能是不充分的。随后，我们提出探索式纳什偏好优化（ENPO），该方法将SFT型正则化与对抗性……

    arXiv:2606.01382v2 Announce Type: replace-cross  Abstract: Preference alignment is central to improving large language models (LLMs), but reward-based formulations can be restrictive when human preferences are non-transitive. Nash learning from human feedback (NLHF) addresses this limitation by modeling alignment as a preference game and seeking a Nash equilibrium. However, the learning-theoretic foundations of scalable NLHF remain limited: existing regret guarantees rely on explicit preference-model estimation and minimax oracles, whereas simpler iterative methods lack such guarantees. We study online iterative NLHF and identify exploration as a key obstacle. First, we show that standard iterative NLHF can incur an exponential dependence on the inverse KL-regularization parameter, demonstrating that implicit exploration through policy updates can be insufficient. We then propose Exploratory Nash Preference Optimization (ENPO), which combines a SFT-type regularization with adversarial 
    
[^264]: 先探测后编辑：基于结构药物设计中面向大语言模型智能体的探针引导分子优化

    Probe Before You Edit: Probing-Guided Molecular Optimization for LLM Agents in Structure-Based Drug Design

    [https://arxiv.org/abs/2606.00555](https://arxiv.org/abs/2606.00555)

    该论文提出PROBE框架，模仿药物化学家“先探测后编辑”的做法，让LLM智能体在编辑前通过受控类似物探测口袋-配体复合物的响应，从而在基于结构的药物设计中同时提升配体的结合亲和力与成药性。

    

    基于结构的药物设计越来越多地使用大语言模型（LLM）智能体来针对靶标口袋迭代地改进配体，然而一个可行的配体必须满足两个常常相互冲突的目标——结合亲和力与成药性——而单步优化很少能同时提升这两者。为了量化这一困难，我们引入了两个诊断指标：第一个衡量单次编辑同时改进两个目标的频率，第二个衡量一个目标上的提升伴随另一个目标上损失的频率。将这些诊断应用于当前的LLM智能体流程后，我们发现了一种一致的失败模式：智能体在不知道口袋-配体复合物会如何响应局部修改的情况下就进行分子编辑，因此很少实现两目标的联合改进。受药物化学家的启发——他们在选定优化方向之前，会通过受控的类似物编辑来探测口袋-配体复合物——我们提出了PROBE，一个构建于……的优化框架（摘要原文在此处截断）。

    arXiv:2606.00555v3 Announce Type: replace  Abstract: Structure-based drug design increasingly employs LLM agents to iteratively refine ligands against a target pocket, yet a viable ligand must satisfy two often-conflicting objectives---binding affinity and druggability---which single optimization steps rarely improve together. To quantify this difficulty, we introduce two diagnostic metrics: the first measures how often a single edit improves both objectives, and the second measures how often a gain on one objective comes with a loss on the other. Applying these diagnostics to current LLM-agent pipelines exposes a consistent failure mode: the agent performs molecular editing without knowing how the pocket-ligand complex responds to local modifications, thus rarely achieving joint improvement. Inspired by medicinal chemists, who probe the pocket-ligand complex with controlled analog edits before choosing an optimization direction, we propose \textbf{PROBE}, an optimization framework bui
    
[^265]: StemBind：当多模态大语言模型在抽象视觉推理的规则与实例之间迷失时

    StemBind: When MLLMs Get Lost Between Rules and Instances in Abstract Visual Reasoning

    [https://arxiv.org/abs/2606.00148](https://arxiv.org/abs/2606.00148)

    StemBind通过针对同一视觉题干的感知、规则、完整三步诊断问题，将多模态大语言模型在抽象视觉推理中的错误精确定位到具体的推理子步骤。

    

    多模态大语言模型往往“知道规则却选错答案”：在抽象视觉推理（AVR）任务上，模型能够描述所看到的内容并说出其中的潜在规律，却仍然无法选出匹配的候选项。现有的AVR基准无法检测到这一问题，因为它们将感知、规则归纳和答案选择压缩为单一的“对/错”信号。我们提出了StemBind，一个共享题干的诊断性基准，它用三个相互对应的问题来探测同一视觉题干：感知（图中有什么）、规则（什么规律支配它）和完整（哪个选项能完成它），从而可以在相同的证据上把最终答案的错误归因于某个具体的子步骤。StemBind包含2,298个精选的低知识依赖题干，涵盖九种可审计的视觉操作，共计19,533个感知/规则/完整（P/R/F）任务，且每个完整题目均按照Sternberg的四个推理阶段（S1编码、S2推断、S3映射、S4应用）进行标注。

    arXiv:2606.00148v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) often know the rule but pick the wrong answer: on abstract visual reasoning (AVR) tasks, a model can describe what it sees and name the underlying pattern, yet still fail to choose the matching candidate. Existing AVR benchmarks cannot detect this because they collapse perception, rule induction, and answer selection into a single right-or-wrong signal. We introduce StemBind, a shared-stem diagnostic benchmark that probes the same visual stem with three aligned questions: Perception (what is in the image), Rule (what pattern governs it), and Full (which option completes it), so a final-answer error can be attributed to a specific sub-step on the same evidence. StemBind contains 2,298 curated knowledge-light stems across nine auditable visual operations, totaling 19,533 P/R/F tasks, with each full item annotated by Sternberg's four reasoning stages (S1 Encode, S2 Infer, S3 Map, S4 Apply).
    
[^266]: 探测视觉-语言模型中的碰撞定位能力以实现安全的人机协作

    Probing Collision Grounding in Vision-Language Models for Safe Human-Robot Collaboration

    [https://arxiv.org/abs/2605.31196](https://arxiv.org/abs/2605.31196)

    该论文提出了 TouchSafeBench——首个基于物理仿真构建、用于评估视觉-语言模型“碰撞定位”能力的基准，包含近三千个人机共处场景，支持安全状态分类与即将碰撞预警两项实际部署任务。

    

    安全的人机协作不仅仅需要视觉描述：监控系统必须判断机器人本体是否与场景保持安全分离、是否已经与场景或人员发生碰撞、或即将发生碰撞。我们将这种能力称为“碰撞定位”：即将视觉观测与机器人本体几何、相机视角、场景布局、人类接近程度以及时间上的运动信息绑定起来，以推断当前和即将发生的接触。我们提出了 TouchSafeBench，一个基于物理的基准测试，用于评估视觉-语言模型（VLM）中的碰撞定位能力。TouchSafeBench 构建于 Habitat 3.0 仿真环境中，包含 2,940 个模拟室内人机共处场景，涵盖社交导航和社交重排任务，并提供同步的多视角 RGB-D 观测、俯视轨迹图、标定的相机元数据以及来自仿真器的接触标签。我们研究了两个面向实际部署的任务：对当前安全状态进行分类，以及对即将发生的碰撞发出警告。

    arXiv:2605.31196v2 Announce Type: replace-cross  Abstract: Safe human--robot collaboration requires more than visual description: a monitor must determine whether the robot body is safely separated, already colliding with the scene or a person, or about to collide. We call this capability collision grounding: binding visual observations to robot body geometry, camera viewpoint, scene layout, human proximity, and temporal motion in order to infer present and imminent contact. We introduce TouchSafeBench, a physics-grounded benchmark for evaluating collision grounding in vision-language models (VLMs). Built in Habitat~3.0, TouchSafeBench contains 2,940 simulated indoor co-presence episodes across social navigation and social rearrangement, with synchronized multi-view RGB-D observations, top-down trajectory maps, calibrated camera metadata, and simulator-derived contact labels. We study two deployment-facing tasks: classifying the current safety state and warning about imminent collision
    
[^267]: 相同答案，不同置信度：大语言模型置信度校准中的协议敏感性

    Same Answer, Different Confidence: Protocol Sensitivity in LLM Confidence Calibration

    [https://arxiv.org/abs/2605.27752](https://arxiv.org/abs/2605.27752)

    该研究发现大语言模型的置信度校准比较结果高度依赖于测量协议——仅改变读取标记概率的上下文，就能使言语化置信度与标记置信度之间的差距移动约一个差距本身的大小，表明现有关于两种置信度孰优孰劣的结论并不稳定。

    

    大语言模型（LLM）可以通过语言表述来说明其对答案的置信度（言语化置信度），也可以从答案标记的概率中读取置信度（标记置信度）。对二者的比较旨在探究哪一种校准得更好，但答案的标记概率取决于其前面的文本。在五个开源大语言模型和四个问答数据集上，我们固定每个答案、其言语化置信度及其正确性标签，仅改变每个分数的测量方式。结果表明，分数本身会随测量协议而变化，二者之间的比较结果也是如此。改变读取标记概率的上下文，会使两种分数之间的差距移动大约差距本身的大小（期望校准误差ECE为.287对.304；AUROC为.189对.145）。在20个设置中的19个和17个里，该变化的95%置信区间不包含零，且在AUROC指标下该变化在所有设置中符号一致，因此上下文的改变……

    arXiv:2605.27752v4 Announce Type: replace  Abstract: Large language models (LLM) can state their confidence in an answer (verbalized confidence), or the confidence can be read from the probability of the answer's tokens (token confidence). Comparisons of the two ask which is better calibrated, but the token probability of an answer depends on the text that precedes it. Across five open LLMs and four question-answering datasets, we fix each answer, its verbalized confidence, and its correctness label, and vary how each score is measured. The scores themselves change with the protocol, and so does the comparison between them. Changing the context in which the token probability is read shifts the gap between the two scores by about the size of the gap itself (.287 vs. .304 in expected calibration error, ECE; .189 vs. .145 in AUROC). The 95% interval of the shift excludes zero in 19 and 17 of the 20 settings, and under AUROC the shift has the same sign in all of them, so the context change
    
[^268]: 广泛的生物医学知识能否被情境化为基于具体场景的命题？

    Can Broad Biomedical Knowledge be Contextualized into Scenario-Grounded Propositions?

    [https://arxiv.org/abs/2605.27082](https://arxiv.org/abs/2605.27082)

    提出了SCENE——一个双层多智能体框架，将知识情境化建模为迭代搜索过程，把广泛的生物医学知识转化为有证据支持、基于具体场景的命题，从而弥合背景知识与具体数据之间的鸿沟。

    

    生物医学发现通常需要将广泛的生物医学知识与具体的实验或临床数据相联系。背景知识能够提示相关机制，但通常过于宽泛，难以直接映射到数据集变量上；而数据驱动的模式往往局限于特定数据集，且难以从机制角度加以解释。我们将这一缺失的环节研究为“知识情境化”：即将广泛的生物医学知识转化为有证据支持的、基于具体场景的命题，使领域专家能够对其进行检查、回放和验证。我们提出了SCENE，一个双层多智能体框架，将知识情境化视为迭代搜索过程。上层将广泛的知识转化为搜索方向，并将其落实到数据集模式中；下层通过多目标优化执行这些方向，以识别在证据强度和数据支持之间取得平衡的具体命题。两层之间的反馈……

    arXiv:2605.27082v2 Announce Type: replace  Abstract: Biomedical discovery often requires connecting broad biomedical knowledge with specific experimental or clinical data. Background knowledge suggests relevant mechanisms but is usually too general to map directly onto dataset variables, while data-driven patterns can be dataset-specific and hard to interpret mechanistically. We study this missing link as knowledge contextualization: transforming broad biomedical knowledge into evidence-supported, scenario-grounded propositions that domain experts can inspect, replay, and validate. We propose SCENE, a bi-level multi-agent framework that treats knowledge contextualization as iterative search. The upper level converts broad knowledge into search directions and grounds them in the dataset schema. The lower level executes these directions through multi-objective optimization to identify concrete propositions that balance evidential strength and data support. Feedback between the two levels
    
[^269]: 高效基准测试本质上就是特征选择与多元回归

    Efficient Benchmarking Is Just Feature Selection and Multiple Regression

    [https://arxiv.org/abs/2605.25773](https://arxiv.org/abs/2605.25773)

    该论文将高效基准测试重新构建为带特征选择的多元回归问题，通过结合核岭回归与mRMR特征选择算法，在预测误差和排序相关性上显著改进了现有高效基准测试方法。

    

    arXiv:2605.25773v3 公告类型：replace-cross 摘要：高效基准测试技术旨在通过仅使用基准测试题目中的一个子集来预测完整基准测试分数，从而降低评估大语言模型（LLM）的计算成本。通过将这一问题重新构建为带特征选择的多元回归的一个实例，我们发现现有高效基准测试方法只需在预测阶段采用核岭回归（kernel ridge regression）即可得到大幅改进。此外，利用一种称为最小冗余最大相关性（mRMR）的信息论特征选择算法，我们可以通过选择对预测最有价值的题目子集来进一步提升这些方法的效果。除在数据极度匮乏的情形外，这些方法在涵盖二值与连续分数的多种基准测试上，均始终取得更小的预测误差（在MAE和RMSE两方面）以及预测分数与真实分数之间更高的排序相关性（在Spearman ρ和Kendall τ两方面）。

    arXiv:2605.25773v3 Announce Type: replace-cross  Abstract: Efficient benchmarking techniques aim to lower the computational cost of evaluating LLMs by predicting full benchmark scores using only a subset of a benchmark's questions. By reframing this problem as an instance of multiple regression with feature selection, we find that existing efficient benchmarking methods can be greatly improved by simply using kernel ridge regression at the prediction stage. Additionally, using an information-theoretic feature-selection algorithm called minimum redundancy maximum relevance (mRMR), we can further improve upon these methods by selecting question subsets that will be maximally useful for prediction. Except in very data-poor settings, these approaches consistently achieve smaller prediction errors (in both MAE and RMSE), and greater ranking correlation between predicted and true scores (in both Spearman $\rho$ and Kendall $\tau$) across a range of benchmarks using both binary and continuous
    
[^270]: 噪声鲁棒的金融数值实体属性标注

    Noise-Robust Financial Numerical Entity Attribute Tagging

    [https://arxiv.org/abs/2605.24910](https://arxiv.org/abs/2605.24910)

    提出NORA方法，通过任务感知的实例特定加权减轻噪声标签影响，实现金融数值实体的多属性鲁棒标注，并构建了包含1140万实体的大规模基准数据集。

    

    金融数值实体（FNE）理解旨在恢复财务报告中数值提及的含义。现有研究集中于概念名称预测，面临两个重要局限：首先，由于申报文件通常由人工编制，从内联XBRL派生的标签可能包含错误；其次，其他重要的金融数值实体属性，如报告期关系、计量尺度和会计符号，较少受到关注。我们提出了面向丰富金融数值实体属性的噪声鲁棒标注方法（NORA）以填补这些空白。NORA使用任务感知的实例特定加权来减弱训练过程中噪声标签的影响，并开发了在真实世界测试集中过滤噪声标签的评估协议。此外，我们构建了一个大规模基准数据集，包含630万个句子中的1140万个金融数值实体，带有多属性标签和申报元数据。实验表明NORA表现出色。

    arXiv:2605.24910v2 Announce Type: replace  Abstract: Financial Numerical Entity (FNE) understanding aims to recover the meaning of numerical mentions in financial reports. Existing studies focus on concept name prediction and face two important limitations. First, labels derived from inline XBRL may contain errors because filings are usually prepared manually. Second, other important FNE attributes, such as reporting-time relation, measurement scale, and accounting sign, are less emphasized. We propose NOise-Robust Tagging for Rich Financial Numerical Entity Attributes (NORA) to address these gaps. NORA uses task-aware instance-specific weighting to attenuate the influence of noisy labels during training, and we develop an evaluation protocol that filters noisy labels in real-world test sets. In addition, we construct a large-scale benchmark containing 11.4 million FNEs in 6.3 million sentences with multi-attribute labels and filing metadata. Experiments show that NORA performs strongl
    
[^271]: 塑造先验：合成任务分布如何决定表格基础模型的质量

    Shaping the Prior: How Synthetic Task Distributions Determine Tabular Foundation Model Quality

    [https://arxiv.org/abs/2605.18971](https://arxiv.org/abs/2605.18971)

    提出O'Prior组合式真实感合成先验（包含分层SCM元生成器、真实感引擎、压力注入模块与防泄漏课程生成协议），在固定架构与算力的条件下证明合成任务分布的设计是决定表格基础模型质量的关键因素。

    

    是什么决定了表格基础模型的质量？与语言或视觉领域不同，表格基础模型的归纳偏置几乎完全来自合成预训练分布，但人们对这些分布的设计仍然知之甚少。标准的合成先验过于“良态”：它们忽略了决定部署鲁棒性的不规则性与失效模式。我们提出O'Prior——一个组合式真实感先验，由四个相互耦合的组件构建：一个覆盖多样函数族的分层SCM元生成器；一个涵盖异质边缘分布、缺失机制与目标变换的模块化真实感引擎；一个显式注入混杂因素与支持集-查询失配的压力模块；以及一个由课程机制管控、防泄漏的生成协议。为了将先验设计隔离为唯一的科学变量，我们在固定架构、优化器和计算预算的前提下，仅改变合成任务分布……

    arXiv:2605.18971v2 Announce Type: replace-cross  Abstract: What determines the quality of a tabular foundation model? Unlike language or vision, tabular foundation models acquire their inductive biases almost entirely from synthetic pretraining distributions, yet the design of these distributions remains poorly understood. Standard synthetic priors are too well-behaved: they omit the irregularities and failure modes that determine deployment robustness. We introduce O'Prior, a compositional realism prior built around four coupled components: a hierarchical SCM meta-generator spanning diverse functional families; a modular realism engine covering heterogeneous marginals, missingness, and target transforms; an explicit stress module injecting confounding and support-query mismatch; and a curriculum-governed, leakage-safe generation protocol. To isolate prior design as the scientific variable, we hold architecture, optimizer, and compute budget fixed and vary only the synthetic task distr
    
[^272]: NeuroMAS：将多智能体系统视为神经网络并结合联合强化学习

    NeuroMAS: Multi-Agent Systems as Neural Networks with Joint Reinforcement Learning

    [https://arxiv.org/abs/2605.16757](https://arxiv.org/abs/2605.16757)

    该论文提出 NeuroMAS，将多智能体语言系统视为以 LLM 智能体为节点、文本信号为边的可训练类神经网络架构，通过强化学习让节点自主学会通信、分工与协调，从而将多智能体设计从手工工作流工程转变为可扩展的架构设计。

    

    多智能体语言系统通常被构建为手工设计的工作流，其中智能体被预先分配语义角色，通信协议也事先指定。我们提出 NeuroMAS，这是一种首次将多智能体语言系统视为可训练、可扩展的类神经网络架构的方法，其中 LLM 智能体作为节点，中间文本信号作为边。在 NeuroMAS 中，智能体节点是角色无关但结构感知的：拓扑结构仅决定信息大体上如何流动，而强化学习训练决定节点如何通信、专门化与协调。这一表述将多智能体设计从工作流工程转向架构设计，使深度、宽度、连接性和增长协议成为可扩展的能力来源。此外，我们从理论上说明了当任务可进行层次化分解时，为什么这种模块化的文本计算更具参数效率。

    arXiv:2605.16757v2 Announce Type: replace  Abstract: Multi-agent language systems are often built as hand-designed workflows, where agents are assigned semantic roles and communication protocols are specified in advance. We propose NeuroMAS, a method that first treats a multi-agent language system as a trainable and scalable neural-network-like architecture with LLM agents as nodes and intermediate textual signals as edges. In NeuroMAS, agent nodes are role-free but structure-aware: the topology only determines how information can flow in general, while reinforcement learning training determines how nodes communicate, specialize, and coordinate. This formulation shifts multi-agent design from workflow engineering toward architecture design, where depth, width, connectivity, and growth protocol become scalable sources of capability. Further, we provide a theoretical perspective showing why such modular textual computation is more parameter-efficient when tasks admit hierarchical decompo
    
[^273]: 视觉语言模型“描迹”而未“追踪”：诊断视觉路径跟随中的失败

    VLMs Trace Without Tracking: Diagnosing Failures in Visual Path Following

    [https://arxiv.org/abs/2605.15672](https://arxiv.org/abs/2605.15672)

    研究发现，前沿视觉语言模型即使在没有交叉干扰的受控设置下也难以稳健地跟随简单连续线条，其根源在于视觉编码阶段就已出现正确路径信息的退化，且模型扩展等常规补救措施收效甚微。

    

    视觉语言模型（VLM）在多模态基准测试中表现优异，但它们能否可靠地跟随连续的视觉结构仍不清楚。我们聚焦于路径跟随这一基本操作，它要求在连续的视觉延伸中跟随一条选定的路径。在考虑交叉或重叠等复杂因素之前，我们首先在消除这些干扰因素的受控设置下，测试当前VLM能否稳健地跟随一条简单的连续线条。在前沿模型中，我们发现描迹本身已经十分脆弱，这种脆弱性表现为当存在其他邻近路径时，路径保持不稳定。对内部表示的分析表明，确定正确路径延续所需的信息会逐渐减弱，而这种退化在视觉编码阶段就已经出现。常见的补救措施无法解决这一弱点：扩大模型规模带来的改进有限，添加（后续内容被截断）……

    arXiv:2605.15672v2 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) perform strongly on multimodal benchmarks, yet their ability to reliably follow continuous visual structure remains unclear. We focus on path following, a basic operation that requires following a selected path through successive visual continuations. Before considering complex factors such as crossings or overlaps, we first test whether current VLMs can robustly follow a simple continuous line under controlled settings that remove such confounds. Across frontier models, we find that tracing is already fragile, with this vulnerability manifesting as unstable path preservation in the presence of other nearby paths. Analysis of internal representations shows that information needed to determine the correct path continuation becomes weaker, with this degradation already emerging in the visual encoding stage. Common remedies do not resolve this weakness: model scaling provides limited improvement, addi
    
[^274]: 模型自适应的工具必要性揭示了LLM工具使用中的“知行差距”

    Model-Adaptive Tool Necessity Reveals the Knowing-Doing Gap in LLM Tool Use

    [https://arxiv.org/abs/2605.14038](https://arxiv.org/abs/2605.14038)

    该论文提出了基于各模型自身能力表现的“模型自适应工具必要性”新定义，并发现在算术和事实问答任务中，模型的实际工具调用行为与真实的工具必要性之间存在高达26.5%-54.0%的不匹配，揭示了LLM工具使用中的“知行差距”。

    

    大型语言模型（LLM）日益作为自主智能体运行，必须决定何时直接回答、何时调用外部工具。以往研究自适应工具使用的工作大多将工具必要性视为与模型无关的属性，由人类或LLM评判者进行标注，且主要涵盖答案显而易见的情况（例如获取天气 vs. 改写文本）。然而，现实中的工具必要性更为微妙，因为不同模型的能力边界存在差异：一个强模型能够独立解决的问题，对于较弱的模型来说可能仍然需要借助工具。在这项工作中，我们提出了一种基于各模型实际经验表现的模型自适应工具必要性定义。根据这一定义，我们在算术和事实问答数据集上，将工具必要性与四个模型实际观察到的工具调用行为进行对比，发现二者分别存在26.5%-54.0%和30.8%-41.8%的显著不匹配。为了诊断这一（摘要在此处被截断）

    arXiv:2605.14038v3 Announce Type: replace  Abstract: Large language models (LLMs) increasingly act as autonomous agents that must decide when to answer directly vs. when to invoke external tools. Prior work studying adaptive tool use has largely treated tool necessity as a model-agnostic property, annotated by human or LLM judge, and mostly cover cases where the answer is obvious (e.g., fetching the weather vs. paraphrasing text). However, tool necessity in the wild is more nuanced due to the divergence of capability boundaries across models: a problem solvable by a strong model on its own may still require tools for a weaker one. In this work, we introduce a model-adaptive definition of tool-necessity, grounded in each model's empirical performance. Following this definition, we compare the necessity against observed tool-call behavior across four models on arithmetic and factual QA dataset, and find substantial mismatches of 26.5-54.0% and 30.8-41.8%, respectively. To diagnose the fa
    
[^275]: 水印应被视为一种监控原语

    Watermarking Should Be Treated as a Monitoring Primitive

    [https://arxiv.org/abs/2605.13095](https://arxiv.org/abs/2605.13095)

    论文主张水印不仅是防御移除和伪造的安全技术，更应被视为一种监控原语——同样的水印信号可用于将生成内容追溯至具体生成实体，即使是零比特水印也能被内部观察者用于个体归因，外部观察者在特定配置下也能识别内容来源。

    

    水印是生成式模型提供商满足合成内容标记法律要求的主要手段。水印将一个隐藏信号嵌入到生成内容中，持有密钥的检测器随后可以恢复该信号。其安全性主要针对的是在单个输出中移除或伪造该信号的对手。我们主张，水印还应作为一种“监控原语”来进行评估和治理，因为同样的信号可以将输出与其生成实体关联起来。我们表明：（i）拥有检测器访问权限的内部观察者即使在零比特水印（即不编码任何身份信息的水印）下，只要每个实体持有自己的密钥，也能将输出归因于具体实体；（ii）在没有密钥或检测器的情况下，外部观察者在我们所测试的配置中也能识别出未见过的文本和图像输出的来源。这两种能力都依赖于评估中未报告的部署选择，即密钥粒度……（摘要在此处截断）

    arXiv:2605.13095v4 Announce Type: replace-cross  Abstract: Watermarking is a primary means for providers of generative models to meet legal requirements to mark synthetic content. A watermark embeds a hidden signal into generated content that a detector holding a secret key can later recover. Its security is evaluated primarily against adversaries who remove or forge this signal in individual outputs. We argue that watermarking should also be evaluated and governed as a \emph{monitoring primitive}, because the same signal can link outputs to the entity that generated them. We show that (i) internal observers with detector access attribute outputs to individual entities even under zero-bit watermarks, which encode no identity, when each entity holds its own key, and (ii) external observers without keys or detectors identify the source of unseen text and image outputs in the configurations we test. Both depend on deployment choices that evaluations do not report, namely how fine-grained 
    
[^276]: 通过结构化元认知在通用智能体中实现深度推理

    Deep Reasoning in General Purpose Agents via Structured Meta-Cognition

    [https://arxiv.org/abs/2605.11388](https://arxiv.org/abs/2605.11388)

    提出“深度推理”方法，利用形式化语言将元推理表示为可执行的分解，在推理时动态构建任务特定的脚手架，使LLM智能体能像人类一样灵活地在联想推理、形式化计算和递归求解等不同推理模式间切换。

    

    人类通过在不同推理模式之间灵活切换来直观地解决复杂问题：他们进行规划、执行、修改中间目标、通过联想判断消解歧义，并对明确定义的子问题应用形式化程序。当前的LLM智能体缺乏这种灵活性，因为它们的脚手架（scaffold）预先硬编码了此类推理决策。当其规定的结构与任务相匹配时，这些脚手架是有效的，但当解决任务需要调整推理结构本身时，它们就会变得脆弱。我们提出了深度推理（Deep Reasoning）——一种通过结构化元推理在推理时构建任务特定脚手架的方法。深度推理使用一种形式化语言，将元推理表示为对联想推理、形式化计算和递归子问题求解的可执行分解，使分解原则能够被编码为上下文示例，从而指导测试时的脚手架构建。

    arXiv:2605.11388v2 Announce Type: replace  Abstract: Humans intuitively solve complex problems by flexibly shifting among reasoning modes: they plan, execute, revise intermediate goals, resolve ambiguity through associative judgment, and apply formal procedures to well-specified subproblems. Current LLM agents lack this flexibility, as their scaffolds hard-code such reasoning decisions in advance. These scaffolds are effective when their prescribed structure matches the task, but brittle when solving the task requires adapting the structure of reasoning itself. We introduce Deep Reasoning -- an inference-time approach for constructing task-specific scaffolds through structured meta-reasoning. Deep Reasoning uses a formal language that represents meta-reasoning as executable decompositions over associative inference, formal computation, and recursive subproblem solving, enabling decomposition principles to be encoded as in-context examples that guide test-time scaffold construction. We 
    
[^277]: 吸引子-血管耦合理论：基于智能手机光电容积脉搏波图的无袖带血压估计的形式化基础与实证验证（符合AAMI标准）

    Attractor-Vascular Coupling Theory: Formal Grounding and Empirical Validation for AAMI-Standard Cuffless Blood Pressure Estimation from Smartphone Photoplethysmography

    [https://arxiv.org/abs/2605.10871](https://arxiv.org/abs/2605.10871)

    该论文提出吸引子-血管耦合理论（AVCT），从PPG吸引子几何结构出发构建可解释的血压预测框架，实现了符合AAMI标准的智能手机无袖带血压估计，校准后收缩压/舒张压平均绝对误差仅2.05/1.67 mmHg。

    

    我们提出了吸引子-血管耦合理论（AVCT），这是一个理论指导的框架，用于从光电容积脉搏波图（PPG）吸引子几何结构中推导可解释的血压（BP）预测因子。AVCT将血管动力学、PPG观测、延迟坐标重构和吸引子描述符连接在一个统一的框架内，以形成可检验的假设，而不预先假定某个特定低维特征集具有生理充分性。我们在两个互补的实验设置下，使用留一受试者交叉验证（LOSO-CV）评估了基于LightGBM的血压估计器。实验1包含来自BIDMC和VitalDB数据库的46名ICU/手术受试者，拥有心电图（ECG）、PPG及重复的参考血压测量数据。校准后的完整模型实现了收缩压/舒张压（SBP/DBP）平均绝对误差（MAE）为2.05/1.67 mmHg，而仅使用PPG吸引子特征的模型则为2.02/1.63 mmHg。实验2包括217名动态（门诊）……（原文摘要在此处截断）

    arXiv:2605.10871v3 Announce Type: replace-cross  Abstract: We present Attractor-Vascular Coupling Theory (AVCT), a theory-guided framework for deriving interpretable blood pressure (BP) predictors from photoplethysmography (PPG) attractor geometry. AVCT connects vascular dynamics, PPG observation, delay-coordinate reconstruction, and attractor descriptors within a unified framework for formulating testable hypotheses, without presupposing the physiological sufficiency of a particular low-dimensional feature set. We evaluate a LightGBM-based estimator using leave-one-subject-out cross-validation (LOSO-CV) across two complementary experimental settings. Experiment 1 comprises 46 ICU/surgical subjects from BIDMC and VitalDB, with ECG, PPG, and repeated reference BP measurements. The calibrated full model achieves systolic/diastolic BP (SBP/DBP) mean absolute errors (MAEs) of 2.05/1.67 mmHg, compared with 2.02/1.63 mmHg for the PPG-attractor-only model. Experiment 2 includes 217 ambulatory
    
[^278]: POETS：通过计算高效的策略集成实现不确定性感知的LLM优化

    POETS: Uncertainty-Aware LLM Optimization via Compute-Efficient Policy Ensembles

    [https://arxiv.org/abs/2605.07775](https://arxiv.org/abs/2605.07775)

    POETS利用KL正则化策略会隐式编码奖励函数这一洞察，直接训练共享预训练权重的策略集成来捕获认知不确定性，从而以计算高效的方式将不确定性量化与LLM策略优化相结合，绕过了传统上需要先训练不确定性感知奖励模型再拟合策略的复杂嵌套流程。

    

    在黑盒优化和强化学习中，平衡探索与利用是一个核心挑战。我们提出了POETS（用于汤普森采样的策略集成，Policy Ensembles for Thompson Sampling），这是一个连接不确定性量化与策略优化的新颖框架。我们的方法基于这样一个洞察：使用Kullback-Leibler (KL)正则化训练的策略会隐式地编码潜在的奖励函数。基于这一洞察，POETS绕过了训练不确定性感知奖励模型并单独将策略拟合到该模型这一复杂且嵌套的流程。相反，我们直接训练一个策略集成，通过将隐式编码的奖励函数与在线的自举数据相匹配来捕获认知不确定性。为了克服集成大型语言模型（LLM）所带来的高昂计算与内存开销，POETS采用了一种高效的架构：该集成共享一个预训练……

    arXiv:2605.07775v2 Announce Type: replace-cross  Abstract: Balancing exploration with exploitation is a core challenge in black-box optimization and reinforcement learning. We introduce POETS ($\textbf{Po}$licy $\textbf{E}$nsembles for $\textbf{T}$hompson $\textbf{S}$ampling), a novel framework that bridges uncertainty quantification and policy optimization. Our approach is grounded in the insight that policies trained with Kullback-Leibler (KL) regularization implicitly encode an underlying reward function. Building on this, POETS bypasses the complex, nested process of training an uncertainty-aware reward model and separately fitting a policy to this model. Instead, we directly train a policy ensemble to capture epistemic uncertainty by matching implicitly encoded reward functions to online, bootstrapped data. To overcome the prohibitive compute and memory constraints of ensembling Large Language Models (LLMs), POETS utilizes an efficient architecture: the ensemble shares a pre-train
    
[^279]: MedAction：迈向主动式多轮临床诊断大语言模型

    MedAction: Towards Active Multi-turn Clinical Diagnostic LLMs

    [https://arxiv.org/abs/2605.07305](https://arxiv.org/abs/2605.07305)

    提出MedAction树状结构蒸馏流水线，通过大语言模型与环境交互合成高质量多轮诊断轨迹，使模型学会在部分证据下主动开具检查、解读结果并更新鉴别诊断，克服了静态单轮诊断评估的局限。

    

    现有的大多数大语言模型诊断评估都是在静态、单轮的设置下进行的，即预先提供完整的患者信息，这是对真实临床实践的过度简化。我们研究主动式诊断：即从初始观察开始、开具检查、解读结果，并在多轮过程中不断更新鉴别诊断的真实临床流程。通过系统性分析，我们识别出当前大语言模型中三种反复出现的失败模式：缺乏依据的检查开具、不可靠的诊断更新，以及多轮对话中连贯性的退化。这些失败共同揭示了一个核心缺陷：现有的医学训练数据教会模型基于完整信息进行推理，却没有教会模型在不断演变的部分证据下采取行动。为了弥补这一差距，我们提出了 MedAction，这是一个树状结构的蒸馏流水线，通过大语言模型与环境的交互，合成多样化且高质量的多轮诊断轨迹。我们提出了两种……

    arXiv:2605.07305v2 Announce Type: replace-cross  Abstract: Most existing LLM diagnoses are evaluated on static, single-turn settings where complete patient information is provided upfront, an oversimplification of real clinical practice. We study active diagnosis: the real-life clinical process of starting from initial observation, ordering tests, interpreting results, and updating a differential diagnosis across multiple turns. Through systematic analysis, we identify three recurring failure modes in current LLMs: ungrounded test ordering, unreliable diagnostic update, and degraded multi-turn coherence. Together, these failures reveal a core deficit: existing medical training data teaches models to reason from complete information but not to act under evolving, partial evidence. To address this gap, we introduce MedAction, a tree-structured distillation pipeline that synthesizes diverse and high-quality multi-turn diagnostic trajectories via LLM-environment interaction. We propose two
    
[^280]: Q-MMR：通过递归重加权与矩匹配进行离策略评估

    Q-MMR: Off-Policy Evaluation via Recursive Reweighting and Moment Matching

    [https://arxiv.org/abs/2605.06474](https://arxiv.org/abs/2605.06474)

    提出 Q-MMR 框架，通过自上而下的递归重加权和矩匹配学习每个数据点的权重来实现离策略评估，并在仅假设 Q 函数可实现性的条件下获得与维度无关的有限样本保证。

    

    我们提出了一种新颖的理论框架 Q-MMR，用于有限时域马尔可夫决策过程（MDP）中的离策略评估。Q-MMR 学习一组标量权重，每个数据点对应一个权重，使得重加权后的奖励能够近似目标策略下的期望回报。这些权重通过与价值函数判别器类进行矩匹配的目标，以自上而下的方式归纳地学习。值得注意的是，或许令人惊讶的是，在仅需 $Q^\pi$ 可实现性的条件下，即可为一般函数逼近建立数据相关的有限样本保证，且该边界与维度无关——也就是说，误差不依赖于函数类的统计复杂度。我们还与若干现有方法建立了联系，例如重要性采样和线性 FQE。进一步的理论分析为覆盖性这一离线强化学习中至关重要的概念的本质提供了新的认识。

    arXiv:2605.06474v3 Announce Type: replace-cross  Abstract: We present a novel theoretical framework, Q-MMR, for off-policy evaluation in finite-horizon MDPs. Q-MMR learns a set of scalar weights, one for each data point, such that the reweighted rewards approximate the expected return under the target policy. The weights are learned inductively in a top-down manner via a moment matching objective against a value-function discriminator class. Notably, and perhaps surprisingly, a data-dependent finite-sample guarantee for general function approximation can be established under only the realizability of $Q^\pi$, with a dimension-free bound -- that is, the error does not depend on the statistical complexity of the function class. We also establish connections to several existing methods, such as importance sampling and linear FQE. Further theoretical analyses shed new light on the nature of coverage, a concept of fundamental importance to offline RL.
    
[^281]: MoLS：面向大语言模型的信噪比引导的模块化学习率校准

    MoLS: SNR-Guided Module-Wise Learning-Rate Calibration for Large Language Models

    [https://arxiv.org/abs/2605.05794](https://arxiv.org/abs/2605.05794)

    MoLS 通过估计各模块的梯度信噪比并采用逆平方根缩放自动确定相对学习率乘数，以一次性校准取代了大语言模型训练中昂贵的手动模块学习率调参。

    

    大语言模型（LLM）包含功能上各异的模块，这些模块具有异构的梯度统计特性。尽管 Adam(W) 能在参数层面自适应地调整更新，但在各模块之间选择相对学习率往往依赖于成本高昂的手动调参。我们通过梯度信噪比（SNR）来研究这一问题。我们的分析识别了模块间 SNR 差异的结构性来源，并将 SNR 与 Adam 归一化平均梯度分量的衰减联系起来。基于这些发现，我们提出了基于信噪比的模块化学习率缩放方法 MoLS，这是一种轻量级方法，用一次性校准取代了手动乘数搜索。MoLS 在初始适应期结束后估计模块级的 SNR，并使用逆平方根缩放来确定相对学习率乘数。这些乘数在预热阶段被平滑引入，随后保持固定，同时全局学习率调度照常进行。（注：原摘要在此处截断）

    arXiv:2605.05794v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) contain functionally distinct modules with heterogeneous gradient statistics. Although Adam(W) adapts updates at the parameter level, selecting relative learning rates across modules often relies on costly manual tuning. We study this problem through gradient signal-to-noise ratio (SNR). Our analysis identifies structural sources of module-wise SNR differences and relates SNR to the attenuation of Adam's normalized mean-gradient component. Based on these findings, we propose \textbf{Module-wise Learning Rate Scaling via SNR (\name)}, a lightweight method that replaces manual multiplier search with one-shot calibration. MoLS estimates module-level SNRs after an initial adaptation period and uses inverse-square-root scaling to determine relative learning-rate multipliers. The multipliers are introduced smoothly during warmup and then held fixed while the global learning-rate schedule continues. Our th
    
[^282]: 更好的奖励模型并不总是偏见更少：超越标准奖励基准的社会对齐研究

    Better Reward Models Are Not Always Less Biased: Social Alignment Beyond Standard Reward Benchmarks

    [https://arxiv.org/abs/2605.05003](https://arxiv.org/abs/2605.05003)

    该论文将奖励模型评估扩展到偏见、安全、道德和伦理推理四个社会关键领域，发现标准基准性能与社会对齐行为之间仅有不一致的关联，且更强的奖励模型在应对多元社会期望时并不一定表现更好，揭示了偏见缓解与情境忠实性之间的权衡。

    

    奖励模型是大语言模型对齐的核心，然而当前的基准测试几乎无法揭示这些模型如何表征多样的、有时相互冲突的社会偏好。我们将奖励模型评估扩展到四个具有重大社会影响的领域：偏见、安全、道德和伦理推理。我们的分析表明，在基准测试中的优异表现可能掩盖社会行为上的重要差异。模型可能在避免某些刻板印象的同时强化了其他刻板印象，而总体偏见得分可能隐藏这些相互对立的倾向。我们还发现，更强的刻板印象规避可能以牺牲情境忠实性为代价，揭示了社会偏见缓解与情境敏感性之间存在权衡关系。在广泛的奖励模型集合中，标准基准性能与社会对齐行为之间仅存在不一致的关联。这些结果表明，更强的奖励模型在应对多元社会期望方面并不一定表现更好。

    arXiv:2605.05003v2 Announce Type: replace-cross  Abstract: Reward models are central to LLM alignment, yet current benchmarks reveal little about how they represent diverse and sometimes competing social preferences. We extend reward model evaluation to four socially consequential domains: bias, safety, morality, and ethical reasoning. Our analysis shows that strong benchmark performance can obscure important differences in social behavior. Models may avoid some stereotypes while reinforcing others, and aggregate bias scores can hide these opposing tendencies. We also find that stronger stereotype avoidance can come at the expense of contextual faithfulness, revealing a trade-off between social bias mitigation and sensitivity to context. Across a broad set of reward models, standard benchmark performance is only inconsistently related to socially aligned behavior. These results show that stronger reward models are not uniformly better at navigating pluralistic social expectations and m
    
[^283]: 智能体与词元：基于水动力词元存储库的多智能体自主推理

    Agents and Tokens: Multi-Agent Autonomous Reasoning over a Hydrodynamic Token Store

    [https://arxiv.org/abs/2605.01102](https://arxiv.org/abs/2605.01102)

    该论文受神经音频编解码器将波形转化为离散词元的启发，提出将近岸水动力模型的海量数据转化为离散词元存储库，使多智能体系统能够在不饱和上下文窗口的前提下对该数据进行自主推理与分析。

    

    大语言模型以及构建在其上的智能体是有史以来最强大的推理机器，科学界正开始让它们投入实际工作。大气与海洋科学领域已开始出现智能体化的集成应用，但近岸水动力学领域基本上仍未受到人工智能的影响。简而言之，智能体难以在不耗尽其上下文窗口的情况下读取该领域的数据产品。近岸模型的一个时间片段可能包含数百兆字节的数据，如果存档的分块方式不适合查询需求，智能体将花费更多时间在下载和压缩数据上，而不是获取科学洞察。毕竟，这些存档是为存储而设计的，并非为智能体所需的快速即时访问而设计。语音和音乐领域曾经也遇到过同样的障碍。一分钟的音频包含数百万个数字，但如今的音频模型却能够进行对话并创作音乐。它们通过“聆听”词元（token）“学会”了做到这一点。神经音频编解码器将波形转换为离散符号的短序列（摘要在此处截断）

    arXiv:2605.01102v2 Announce Type: replace  Abstract: Large language models and the agents built on them are the most capable reasoning machines ever built, and science is beginning to put them to work. Atmospheric and oceanic sciences are starting to see agentic integration, but coastal hydrodynamics remains largely untouched by AI. Simply, agents cannot easily read its data products without saturating their context windows. A time frame from a coastal model can hold hundreds of megabytes, and if the archive's chunking does not suit the query, the agent will spend more time downloading and compacting than gaining insights. After all, the archives were designed for storage, not for the fast, on-the-fly access agents need. Speech and music once hit the same wall. A minute of audio is millions of numbers, yet today's audio models hold conversations and write music. They "learned" to do this by "listening" to tokens. Neural audio codecs turn waveforms into short sequences of discrete symbo
    
[^284]: MCPHunt：面向多服务器MCP智能体跨边界数据传播的评估框架

    MCPHunt: An Evaluation Framework for Cross-Boundary Data Propagation in Multi-Server MCP Agents

    [https://arxiv.org/abs/2604.27819](https://arxiv.org/abs/2604.27819)

    MCPHunt是首个用于隔离检测多服务器MCP智能体中跨信任边界非对抗性凭证传播的受控评估基准，通过金丝雀污点追踪、环境可控覆盖设计和CRS分层三项方法贡献，客观量化了忠实工具组合引发的信息流风险。

    

    多服务器MCP智能体带来了信息流控制（IFC）问题：忠实的工具组合可能将各自无害的读/写权限转化为跨边界的凭证传播——这是工作流拓扑的结构性副作用，而未必是模型的恶意行为。我们提出了MCPHunt，据我们所知，这是首个能够隔离跨多服务器MCP信任边界的非对抗性、精确匹配凭证传播的受控基准，具有三项方法论贡献：(1) 基于金丝雀的污点追踪，将传播检测简化为客观的字符串匹配；(2) 环境可控的覆盖设计，包含风险、良性和困难负例三类条件，用于验证流水线的健全性，并增加了匹配的替代值对照（相同任务、不同的合成凭证）；(3) 完成需要秘密（CRS）分层，用于区分任务要求的传播（对逐字指令的忠实执行）与其他情形。

    arXiv:2604.27819v2 Announce Type: replace  Abstract: Multi-server MCP agents create an information-flow control (IFC) problem: faithful tool composition can turn individually benign read/write permissions into cross-boundary credential propagation -- a structural side effect of workflow topology, not necessarily malicious model behavior. We present MCPHunt, to our knowledge the first controlled benchmark that isolates non-adversarial, exact-match credential propagation across multi-server MCP trust boundaries, with three methodological contributions: (1) canary-based taint tracking that reduces propagation detection to objective string matching; (2) an environment-controlled coverage design with risky, benign, and hard-negative conditions that validates pipeline soundness and adds a matched alternative-value control (same tasks, distinct synthetic credentials); (3) completion-requires-secret (CRS) stratification that disentangles task-mandated propagation (faithful execution of verbati
    
[^285]: 公平性偏见：一种人工智能设计的伦理框架

    Equity Bias: An Ethical Framework for AI Design

    [https://arxiv.org/abs/2604.21907](https://arxiv.org/abs/2604.21907)

    提出了“公平性偏见”框架，主张偏见不应被消除而应被透明化和可质疑化，并通过三阶段AI生命周期将伽达默尔诠释学与认知不公正理论转化为构建更公平AI系统的实践方法。

    

    公平性偏见是一个用于构建更公平人工智能系统的哲学与实践框架。它将人工智能视为构建意义与知识的解释性主体，而非中立的计算器。该框架植根于伽达默尔诠释学和认知不公正理论，不将偏见视为需要消除的错误，而是将其视为谁的知识被编码进系统的一种反映。由于所有知识都是有处境性的，中立系统是不可能存在的。与传统旨在减少或消除偏见的方法不同，公平性偏见反而使偏见变得透明且可质疑。其三阶段人工智能生命周期（公平考古、共同创造意义、持续问责）将这些原则付诸实践。公平性偏见为开发者、研究人员、政策制定者以及受人工智能影响的社区提供了一种方法，用以审视一个系统编码了谁的知识，以及谁有权对此提出质疑。通过将知识的政治性置于……

    arXiv:2604.21907v2 Announce Type: replace-cross  Abstract: Equity Bias is a philosophical and practical framework for building more equitable AI systems. It treats AI as an interpretive agent that constructs meaning and knowledge rather than as a neutral calculator. Grounded in Gadamerian hermeneutics and epistemic injustice theory, Equity Bias views bias not as an error to eliminate but as a reflection of whose knowledge is encoded into systems. Since all knowledge is situated, neutral systems are not possible. While traditional approaches aim to reduce or remove bias, Equity Bias instead makes bias transparent and contestable. Its three-stage AI Life Cycle (Equity Archaeology, Co-Creating Meaning, and Ongoing Accountability) turns these principles into practice. Equity Bias gives developers, researchers, policymakers, and communities affected by AI a method for examining whose knowledge a system encodes and who has the power to contest it. By placing the politics of knowledge at the 
    
[^286]: 面向序贯临床诊断的不确定性引导的潜在诊断轨迹学习

    Uncertainty-Guided Latent Diagnostic Trajectory Learning for Sequential Clinical Diagnosis

    [https://arxiv.org/abs/2604.05116](https://arxiv.org/abs/2604.05116)

    该论文提出了一种不确定性引导的潜在诊断轨迹学习（LDTL）框架，通过规划智能体与诊断智能体协作，将诊断动作序列视为潜在路径进行后验优化，从而在缺乏显式监督信息的条件下实现序贯临床诊断中有效的证据获取。

    

    临床诊断需要在不确定性条件下进行序贯式的证据获取。然而，大多数基于大语言模型（LLM）的诊断系统假设患者信息是完全可观测的，因此并未显式地建模临床证据应如何随时间被序贯地获取。即使将诊断建模为一个序贯决策过程，学习有效的诊断轨迹仍然具有挑战性。这是因为可能的证据获取路径空间相对庞大，而临床数据集很少为理想的诊断路径提供显式的监督信息。为此，我们将序贯诊断形式化为一个潜在诊断轨迹学习（LDTL）框架，该框架由一个规划LLM智能体和一个诊断LLM智能体构成。对于诊断LLM智能体，诊断动作序列被视为潜在路径，并引入了一个后验分布，该分布优先选择能够提供……（原文摘要在此处截断）

    arXiv:2604.05116v2 Announce Type: replace  Abstract: Clinical diagnosis requires sequential evidence acquisition under uncertainty. However, most Large Language Model (LLM) based diagnostic systems assume fully observed patient information and therefore do not explicitly model how clinical evidence should be sequentially acquired over time. Even when diagnosis is formulated as a sequential decision process, it is still challenging to learn effective diagnostic trajectories. This is because the space of possible evidence-acquisition paths is relatively large, while clinical datasets rarely provide explicit supervision information for desirable diagnostic paths. To this end, we formulate sequential diagnosis as a Latent Diagnostic Trajectory Learning (LDTL) framework based on a planning LLM agent and a diagnostic LLM agent. For the diagnostic LLM agent, diagnostic action sequences are treated as latent paths and we introduce a posterior distribution that prioritizes trajectories providin
    
[^287]: 模拟巡逻检测中的空间差异：横跨巴尔的摩与芝加哥的可复现生成模型审计

    Spatial disparities in simulated patrol detection: a reproducible generative-model audit across Baltimore and Chicago

    [https://arxiv.org/abs/2603.18987](https://arxiv.org/abs/2603.18987)

    该论文提出了一种结合生成对抗网络巡逻分配与概率检测模型的可复现空间模拟审计框架，利用巴尔的摩和芝加哥的犯罪记录数据量化了模拟巡逻部署在黑人多数区与白人多数区之间的检测率差异。

    

    审计预测性警务需要区分记录在案的犯罪事件、模拟巡逻观测以及地区的人口统计特征。我们提出了一种可复现的空间模拟审计方法，将用于巡逻分配的生成对抗网络（GAN）与概率检测模型相结合。利用145,178条巴尔的摩记录（2017-2019年）和234,882条芝加哥记录（2022年）的冻结存档数据，我们基于人口普查区边界和美国社区调查数据重建了地理协变量。我们在不同观测流条件下，将GAN分配方案与经验位置抽样以及均匀区域部署进行了比较。每种条件下二十个随机种子共产生560条模拟轨迹，每条轨迹覆盖二月至十二月。在仅检测训练条件下，黑人多数区与白人多数区之间的平均检测率差异在各城市-年份中介于-2.03至0.75个百分点之间，而总覆盖率范围为1.（原文摘要至此截断）

    arXiv:2603.18987v3 Announce Type: replace  Abstract: Auditing predictive policing requires distinguishing recorded crime incidents, simulated patrol observations, and the demographic characteristics of places. We present a reproducible spatial simulation audit combining a generative adversarial network (GAN) for patrol allocation with a probabilistic detection model. Using frozen archives of 145,178 Baltimore records (2017-2019) and 234,882 Chicago records (2022), we rebuild geographic covariates from Census tract boundaries and American Community Survey data. We compare GAN allocation with empirical-location sampling and uniform-area deployment under alternative observation streams. Twenty random seeds per condition yield 560 trajectories, each covering February through December. Under detection-only training, mean detection-rate differences between Black-majority and White-majority areas range from -2.03 to 0.75 percentage points across city-years, while total coverage ranges from 1.
    
[^288]: 在高维事件序列中学习预测、发现与推理

    Learning to Predict, Discover, and Reason in High-Dimensional Event Sequences

    [https://arxiv.org/abs/2603.16313](https://arxiv.org/abs/2603.16313)

    本文提出将车辆诊断故障码序列视为一种“语言”，通过机器学习方法实现自动预测、错误模式发现与推理，以替代昂贵且易错的人工规则定义过程。

    

    arXiv:2603.16313v3 公告类型：替换。摘要：现代车辆中嵌入的电子控制单元（ECU）会产生大量称为诊断故障码（DTCs）的异步事件。这些离散事件形成复杂的时间序列，反映车辆子系统不断变化的健康状况。在汽车行业中，领域专家使用布尔规则手动将这些代码分组为更高级别的错误模式（EPs），以表征系统故障并确保安全。然而，随着车辆复杂性的增长，这一手动过程变得越来越昂贵、易出错且难以扩展。值得注意的是，现代车辆中独特DTC的数量与自然语言的词汇量处于同一数量级，通常数以万计。这一观察促使范式转变：将诊断序列视为一种可以被建模、预测并最终解释的语言。传统的统计方法无法捕捉其中的丰富性。

    arXiv:2603.16313v3 Announce Type: replace  Abstract: Electronic control units (ECUs) embedded within modern vehicles generate a large number of asynchronous events known as diagnostic trouble codes (DTCs). These discrete events form complex temporal sequences that reflect the evolving health of the vehicle's subsystems. In the automotive industry, domain experts manually group these codes into higher-level error patterns (EPs) using Boolean rules to characterize system faults and ensure safety. However, as vehicle complexity grows, this manual process becomes increasingly costly, error-prone, and difficult to scale. Notably, the number of unique DTCs in a modern vehicle is on the same order of magnitude as the vocabulary of a natural language, often numbering in the tens of thousands. This observation motivates a paradigm shift: treating diagnostic sequences as a language that can be modeled, predicted, and ultimately explained. Traditional statistical approaches fail to capture the ri
    
[^289]: daVinci-Env：大规模开放式软件工程环境合成

    daVinci-Env: Open SWE Environment Synthesis at Scale

    [https://arxiv.org/abs/2603.13023](https://arxiv.org/abs/2603.13023)

    提出了 OpenSWE——目前最大的完全开源透明的软件工程智能体训练框架，包含 45,320 个覆盖 1.28 万仓库的可执行 Docker 环境，并借助 64 节点集群上的多智能体合成流水线自动化构建。

    

    训练能力强大的软件工程（SWE）智能体需要大规模、可执行且可验证的环境，以为迭代式代码编辑、测试执行和方案改进提供动态反馈回路。然而，现有的开源数据集在规模和仓库多样性方面仍然有限，而工业界的解决方案则不透明且基础设施未公开，为大多数学术研究团队设置了难以逾越的障碍。我们提出了 OpenSWE，这是目前最大的完全透明的 Python 软件工程智能体训练框架，包含 45,320 个可执行的 Docker 环境，覆盖超过 12.8k 个代码仓库，所有 Dockerfile、评估脚本和基础设施均完全开源以保证可复现性。OpenSWE 通过部署在 64 节点分布式集群上的多智能体合成流水线构建，自动化完成仓库探索、Dockerfile 构建、评估脚本生成和迭代测试分析。

    arXiv:2603.13023v3 Announce Type: replace-cross  Abstract: Training capable software engineering (SWE) agents demands large-scale, executable, and verifiable environments that provide dynamic feedback loops for iterative code editing, test execution, and solution refinement. However, existing open-source datasets remain limited in scale and repository diversity, while industrial solutions are opaque with unreleased infrastructure, creating a prohibitive barrier for most academic research groups. We present OpenSWE, the largest fully transparent framework for SWE agent training in Python, comprising 45,320 executable Docker environments spanning over 12.8k repositories, with all Dockerfiles, evaluation scripts, and infrastructure fully open-sourced for reproducibility. OpenSWE is built through a multi-agent synthesis pipeline deployed across a 64-node distributed cluster, automating repository exploration, Dockerfile construction, evaluation script generation, and iterative test analysi
    
[^290]: MetaKE：面向更优精度-可编辑性权衡的知识编辑元学习方法

    MetaKE: Meta-Learning for Knowledge Editing Toward a Better Accuracy-Editability Trade-off

    [https://arxiv.org/abs/2603.12677](https://arxiv.org/abs/2603.12677)

    提出MetaKE框架，通过元学习将知识编辑中上游表示优化与下游参数优化统一为双层优化问题，解决了两阶段优化割裂导致的编辑不充分或可编辑性下降问题，实现了精度与可编辑性之间更优的权衡。

    

    现有基于“定位后编辑”的知识编辑方法通常将编辑分解为两个阶段：上游的目标表示优化和下游的受约束参数优化。两个阶段之间的优化是相互割裂的：上游在未能观察下游计划残差实际实现情况的前提下应用统一正则化，阻碍了精度与可编辑性之间的精细权衡。由于这种实现是请求特定的，并依赖于下游约束，统一正则化可能会过度收缩高关联请求，导致编辑不充分；同时可能对低关联请求正则化不足，产生过大的计划残差，从而降低下游的可编辑性。为弥合这一割裂，我们提出了MetaKE（面向知识编辑的元学习），一个将上游与下游阶段统一为双层优化问题的新框架。内层优化参数更新……

    arXiv:2603.12677v4 Announce Type: replace-cross  Abstract: Existing locate-then-edit Knowledge Editing (KE) methods typically decompose editing into two stages: upstream target representation optimization and downstream constrained parameter optimization. The optimization across the two stages is disconnected: upstream applies uniform regularization without observing downstream realization of the planned residual, hindering a refined accuracy-editability trade-off. Since this realization is request-specific and depends on downstream constraints, uniform regularization can over-shrink high-association requests, causing insufficient editing, while it can under-regularize low-association requests, producing over-large planned residuals that reduce downstream editability. To bridge this disconnect, we propose MetaKE (Meta-learning for Knowledge Editing), a new framework that unifies upstream and downstream stages into a bi-level optimization problem. The inner level optimizes parameter upd
    
[^291]: MoE-SpAc：异构边缘场景下基于投机激活效用的高效MoE推理框架

    MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios

    [https://arxiv.org/abs/2603.09983](https://arxiv.org/abs/2603.09983)

    提出MoE-SpAc框架，将投机解码创新性地重新用作内存管理的前瞻传感器，通过投机效用估计器、异构工作负载均衡器和异步执行引擎，在统一的效用空间中协调专家预取与逐出，使异构边缘设备上的MoE推理TPS较SOTA基线提升42%。

    

    混合专家模型能够实现可扩展的性能，但在边缘设备上面临严重的内存限制。由于自回归专家激活具有动态性和低信息量特性，现有的卸载策略难以应对I/O瓶颈。本文提出将投机解码重新定位，不仅作为计算加速器，更作为内存管理的信息前瞻传感器，这一提议得到了我们理论分析和实证分析的支持。因此，我们提出MoE-SpAc，一个MoE推理框架，它集成了用于追踪专家需求的投机效用估计器、通过在线整数优化动态划分计算的异构工作负载均衡器，以及在相同效用空间中统一预取与逐出操作的异步执行引擎。在七个基准测试上的大量实验表明，MoE-SpAc相对于基于投机解码的SOTA基线，TPS提升了42%……

    arXiv:2603.09983v2 Announce Type: replace-cross  Abstract: Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-information nature of autoregressive expert activation. In this paper, we propose to repurpose Speculative Decoding (SD) not merely as a compute accelerator, but as an informative lookahead sensor for memory management, supported by our theoretical and empirical analyses. Hence, we introduce MoE-SpAc, an MoE inference framework that integrates a Speculative Utility Estimator to track expert demand, a Heterogeneous Workload Balancer to dynamically partition computation via online integer optimization, and an Asynchronous Execution Engine to unify the prefetching and eviction in the same utility space. Extensive experiments on seven benchmarks demonstrate that MoE-SpAc achieves a 42% improvement in TPS over the SOTA SD-based baseline, a
    
[^292]: VORL-EXPLORE：一种面向动态环境多机器人探索的混合学习规划方法

    VORL-EXPLORE: A Hybrid Learning Planning Approach to Multi-Robot Exploration in Dynamic Environments

    [https://arxiv.org/abs/2603.07973](https://arxiv.org/abs/2603.07973)

    提出VORL-EXPLORE混合学习规划框架，通过“执行保真度”评估局部可通行性与执行可靠性，并将其耦合进Voronoi边界分配目标与全局A*规划、反应式强化学习导航的仲裁中，显著提升动态环境下多机器人探索的鲁棒性和效率。

    

    分层式多机器人探索将边界分配与局部导航解耦，导致系统在动态环境中表现脆弱。由于缺乏对执行难度的感知，分配器可能将机器人派往路线拥堵或不安全的边界，从而造成瓶颈、反复震荡的重规划以及冗余覆盖。我们提出 VORL-EXPLORE，这是一个混合学习与规划框架，通过“执行保真度”——对局部可通行性与执行可靠性的估计——使探索具备执行感知能力。该保真度分数一方面用于在带有机器人间排斥项的保真度耦合 Voronoi 分配目标中惩罚有风险或拥挤的边界，另一方面用于在导航过程中对全局 A* 引导与反应式强化学习策略进行仲裁。由此，机器人在可靠区域遵循高效的全局规划，而在拥挤或不断变化的区域则更多依赖更安全的局部反应。

    arXiv:2603.07973v2 Announce Type: replace-cross  Abstract: Hierarchical multi-robot exploration decouples frontier allocation from local navigation, making systems brittle in dynamic environments. Without awareness of execution difficulty, allocators may send robots toward frontiers whose routes are congested or unsafe, causing bottlenecks, oscillatory replanning, and redundant coverage. We propose VORL-EXPLORE, a hybrid learning and planning framework that makes exploration execution-aware through execution fidelity, an estimate of local navigability and execution reliability. This fidelity score is used to penalize risky or crowded frontiers in a fidelity-coupled Voronoi allocation objective with inter-robot repulsion, and to arbitrate between global $A^*$ guidance and a reactive reinforcement learning policy during navigation. As a result, robots follow efficient global plans in reliable regions and rely more on safer local reactions in crowded or changing areas. The fidelity model 
    
[^293]: CrimeNER：用于语言智能研究与调查的精选数据集与方法

    CrimeNER: Curated Dataset and Methods for Research and Investigation in Language Intelligence

    [https://arxiv.org/abs/2603.02150](https://arxiv.org/abs/2603.02150)

    该论文提出了CrimeNER，一个包含1500多份标注文档的犯罪领域命名实体识别数据集，定义了4种粗粒度和21种细粒度犯罪实体类型，并通过传统NER模型和大语言模型的实验验证了数据集质量。

    

    从犯罪相关文档中提取关键信息是执法机构的一项重要任务。这种信息的提取可以被理解为命名实体识别（NER）任务。然而，关于一般真实世界犯罪场景的充分标注数据相当匮乏。为解决这一问题，我们提出了CrimeNER，这是一项犯罪相关NER的案例研究，同时构建了一个通用的犯罪相关命名实体识别数据库，该数据库由从恐怖袭击公开报告和美国司法部新闻稿中提取的超过1500份标注文档组成。我们定义了4种粗粒度犯罪实体类型和21种细粒度实体类型。我们通过使用扁平与嵌套NER架构的全监督实验，以及使用大语言模型（LLMs）的零样本和少样本实验，验证了该数据库的质量。该数据库已在GitHub上公开提供。

    arXiv:2603.02150v3 Announce Type: replace-cross  Abstract: The extraction of critical information from crime-related documents is a crucial task for law enforcement agencies. The extraction of this information can be interpreted as a Named-Entity Recognition (NER) task. However, there is a considerable lack of adequately annotated data on general real-world crime scenarios. To address this issue, we present CrimeNER, a case study of crime-related NER, and a general crime-related Named-Entity Recognition database (CrimeNER-db), consisting of more than 1.5K annotated documents extracted from public reports of terrorist attacks and the US Department of Justice's press notes. We define 4 coarse types of crime entity and 21 fine-grained entity types. We address the quality of the presented database with fully supervised experiments using flat and nested NER architectures and zero- and few-shot experiments using Large Language Models (LLMs). The database is available on GitHub.
    
[^294]: 上下文记忆虚拟化：面向大语言模型智能体的基于有向无环图的状态管理与结构无损裁剪

    Contextual Memory Virtualisation: DAG-Based State Management and Structurally Lossless Trimming for LLM Agents

    [https://arxiv.org/abs/2602.22402](https://arxiv.org/abs/2602.22402)

    提出了上下文记忆虚拟化（CMV）系统，借鉴操作系统虚拟内存思想，将LLM会话历史建模为带快照、分支、裁剪原语的DAG以实现跨并行会话的上下文复用，并通过三遍结构无损裁剪算法在逐字保留所有对话内容的前提下将令牌开销平均降低20%、最高降低86%。

    

    随着大语言模型执行长程推理任务，它们会在上下文窗口内累积大量状态——如架构映射、权衡决策、代码库约定等。当会话达到上下文限制并进行有损压缩时，这些理解便会丢失。我们提出了上下文记忆虚拟化（CMV），一个将大语言模型累积的理解视为版本控制状态的系统。借鉴操作系统的虚拟内存机制，CMV将会话历史建模为有向无环图（DAG），并形式化定义了快照、分支和裁剪等原语，从而实现跨独立并行会话的上下文复用。我们引入了一种三遍式结构无损裁剪算法，该算法逐字保留每条用户消息和助手回复，同时通过剥离机械性冗余（如原始工具输出、基础……）使令牌数量平均减少20%，对于开销较大的会话最多可减少86%。

    arXiv:2602.22402v2 Announce Type: replace-cross  Abstract: As large language models engage in extended reasoning tasks, they accumulate significant state -- architectural mappings, trade-off decisions, codebase conventions -- within the context window. This understanding is lost when sessions reach context limits and undergo lossy compaction. We propose Contextual Memory Virtualisation (CMV), a system that treats accumulated LLM understanding as version-controlled state. Borrowing from operating system virtual memory, CMV models session history as a Directed Acyclic Graph (DAG) with formally defined snapshot, branch, and trim primitives that enable context reuse across independent parallel sessions. We introduce a three-pass structurally lossless trimming algorithm that preserves every user message and assistant response verbatim while reducing token counts by a mean of 20% and up to 86% for sessions with significant overhead by stripping mechanical bloat such as raw tool outputs, base
    
[^295]: 张量承诺：一种面向语言模型的轻量级可验证推理方案

    TensorCommitments: A Lightweight Verifiable Inference for Language Models

    [https://arxiv.org/abs/2602.12630](https://arxiv.org/abs/2602.12630)

    提出张量承诺（TC）这一张量原生的推理证明方案，通过多变量Terkle树将大语言模型推理与篡改即失效的不可逆承诺绑定，仅需约1%的额外时间开销即可实现可验证推理，并将对定制化LLM攻击的鲁棒性提升高达48%。

    

    大多数大型语言模型（LLM）运行在外部云上：用户发送提示词、为推理付费，并且必须信任远程GPU在没有任何对抗性篡改的情况下执行大语言模型。我们批判性地探讨如何实现可验证的大语言模型推理，即证明者（服务方）必须在不重新运行大语言模型的情况下，使验证者（客户端）确信推理被正确执行。现有的密码学方法在大语言模型规模下速度太慢，而非密码学方法则需要强大的验证者GPU。我们提出了张量承诺（TCs），一种张量原生的推理证明方案。TC将大语言模型推理与一个承诺绑定——这是一种在篡改下会失效的不可逆标签——并通过我们提出的多变量Terkle树进行组织。对于LLaMA2，TC在推理基础上仅增加0.97%的证明者时间和0.12%的验证者时间，同时相比需要验证者GPU的最佳先前工作，将针对定制化大语言模型攻击的鲁棒性提升了高达48%。

    arXiv:2602.12630v2 Announce Type: replace-cross  Abstract: Most large language models (LLMs) run on external clouds: users send a prompt, pay for inference, and must trust that the remote GPU executes the LLM without any adversarial tampering. We critically ask how to achieve verifiable LLM inference, where a prover (the service) must convince a verifier (the client) that an inference was run correctly without rerunning the LLM. Existing cryptographic works are too slow at the LLM scale, while non-cryptographic ones require a strong verifier GPU. We propose TensorCommitments (TCs), a tensor-native proof-of-inference scheme. TC binds the LLM inference to a commitment, an irreversible tag that breaks under tampering, organized in our multivariate Terkle Trees. For LLaMA2, TC adds only 0.97% prover and 0.12% verifier time over inference while improving robustness to tailored LLM attacks by up to 48% over the best prior work requiring a verifier GPU.
    
[^296]: 为什么反直觉的似然行为在表格异常检测中很少出现？

    Why Does Counterintuitive Likelihood Behavior Rarely Emerge in Tabular Anomaly Detection?

    [https://arxiv.org/abs/2602.09593](https://arxiv.org/abs/2602.09593)

    本文提出了一个领域无关的反直觉似然现象形式化定义，并通过大规模实验与理论分析证明：与图像领域不同，反直觉的似然行为在表格异常检测中很少出现，且标准化流似然可实现有竞争力的异常检测性能。

    

    具有可处理似然的深度生成模型，以标准化流（normalizing flows）为代表，为基于似然评分的异常检测提供了有效基础。我们证明，与图像领域中深度生成模型经常为异常数据分配更高似然的情况不同，这种反直觉行为在表格数据环境中很少发生。我们首先引入了一个与领域无关的形式化框架，使得能够对这一反直觉现象进行一致的检测和评估，解决了以往工作中缺乏精确定义的问题。通过对47个表格数据集（与13个基线模型进行基准对比）以及来自ADBench的10个CV/NLP嵌入数据集的大量实验，我们表明该现象在一般表格数据中始终很少见，并且标准化流似然能够实现具有竞争力的异常检测性能。我们进一步提供了理论和实证分析，将这一差异归因于……

    arXiv:2602.09593v2 Announce Type: replace-cross  Abstract: Deep generative models with tractable likelihoods, exemplified by normalizing flows, offer an effective basis for anomaly detection through likelihood-based scoring. We demonstrate that, unlike in the image domain where deep generative models frequently assign higher likelihoods to anomalous data, such counterintuitive behavior occurs rarely in tabular settings. We first introduce a domain-agnostic formulation that enables consistent detection and evaluation of the counterintuitive phenomenon, addressing the lack of a precise definition in prior work. Through extensive experiments on 47 tabular datasets benchmarked against 13 baseline models and on 10 CV/NLP embedding datasets from ADBench, we show that the phenomenon is consistently rare in general tabular data and that normalizing flow likelihood achieves competitive anomaly detection performance. We further provide theoretical and empirical analyses attributing this disparit
    
[^297]: 缓解基于流的OOD检测中的似然悖论：一种熵视角

    Mitigating the Likelihood Paradox in Flow-based OOD Detection: An Entropic Perspective

    [https://arxiv.org/abs/2602.09581](https://arxiv.org/abs/2602.09581)

    提出免训练的语义比例熵操控方法SPEM，从熵的视角通过对与分布内记忆库相似度低的输入施加更强扰动，增大对数似然差的期望下界，从而缓解基于流的OOD检测中的似然悖论，使似然排序与人类直觉对齐。

    

    能够便捷计算输入似然的深度生成模型（包括归一化流）常常对分布外（OOD）输入赋予意外高的似然值。我们从熵的视角出发，提出语义比例熵操控（SPEM）来缓解这一悖论，这是一种无需训练的方法，对与分布内（ID）记忆库相似度较低的输入施加更强的扰动。我们推导出理论下界，证明这种熵差能够以有利于分布内数据的方式增大期望对数似然差，并且在标准基准上的实验表明，这种操控使似然排序与人类直觉重新对齐。令人惊讶的是，即使只给流模型提供视觉上无意义的噪声而非原始输入，这种重新对齐依然存在，我们从理论上分析了这一现象，刻画了其发生的条件。

    arXiv:2602.09581v2 Announce Type: replace-cross  Abstract: Deep generative models that can tractably compute input likelihoods, including normalizing flows, often assign unexpectedly high likelihoods to out-of-distribution (OOD) inputs. We mitigate this paradox from an entropic perspective with Semantic Proportional Entropy Manipulation (SPEM), a training-free procedure that applies stronger perturbations to inputs less similar to an in-distribution (ID) memory bank. We derive theoretical lower bounds showing that this entropy gap increases the expected log-likelihood difference in favor of the ID, and experiments on standard benchmarks show that this manipulation realigns the likelihood ordering with human intuition. Strikingly, this realignment persists even when the flow is given only visually meaningless noise instead of the original input, and we analyze this phenomenon theoretically to characterize the conditions under which it occurs.
    
[^298]: StealthRL：用于规避多种AI文本检测器的强化学习改写攻击

    StealthRL: Reinforcement Learning Paraphrase Attacks for Multi-Detector Evasion of AI-Text Detectors

    [https://arxiv.org/abs/2602.08934](https://arxiv.org/abs/2602.08934)

    本文提出StealthRL强化学习改写攻击框架，在保持语义保真度和文本可读性的前提下，将多种AI文本检测器的平均TPR@1%FPR降至0.024，且攻击可迁移至训练中未见的检测器，揭示了AI文本检测器普遍存在的鲁棒性缺陷。

    

    AI文本检测器越来越多地被应用于高风险场景，但它们对保持语义不变的对抗性改写的鲁棒性仍不明确。我们提出了StealthRL，一个利用自适应改写攻击对检测器进行压力测试的强化学习框架。StealthRL在强制保持语义保真度的同时，针对一个检测器集成训练改写策略，随后在完整的过滤MAGE测试集上评估其对未参与训练的检测器家族的迁移能力。在RoBERTa、Fast-DetectGPT、Binoculars和MAGE等检测器上，该攻击显著降低了检测性能，将平均TPR@1%FPR降至0.024，同时保留了可观的语义重叠度和流畅可读的文本。该攻击还能迁移到训练中未使用过的检测器，表明这反映的是更广泛的鲁棒性缺陷，而非对单一目标模型的过拟合。我们还通过检测器得分分析、重复生成稳定性检验和自动化（评估）对主要结果进行了补充。

    arXiv:2602.08934v3 Announce Type: replace-cross  Abstract: AI-text detectors are increasingly used in high-stakes settings, but their robustness to meaning-preserving adversarial rewriting remains unclear. We introduce StealthRL, a reinforcement learning framework for stress-testing detectors with adaptive paraphrase attacks. StealthRL trains a paraphrase policy against a detector ensemble while enforcing semantic fidelity, then evaluates transfer to held-out detector families on the full filtered MAGE test pool. Across RoBERTa, Fast-DetectGPT, Binoculars, and MAGE, the resulting attack sharply degrades detection performance, driving mean TPR@1%FPR down to 0.024 while retaining nontrivial semantic overlap and readable prose. The attack also transfers to detectors not used during training, indicating a broader robustness failure rather than overfitting to a single target model. We complement the main results with detector-score analysis, repeated-generation stability checks, and automat
    
[^299]: 克利福德柯尔莫哥洛夫-阿诺德网络

    Clifford Kolmogorov-Arnold Networks

    [https://arxiv.org/abs/2602.05977](https://arxiv.org/abs/2602.05977)

    提出了克利福德柯尔莫哥洛夫-阿诺德网络（ClKAN），通过随机化拟蒙特卡罗网格生成方法解决高维代数的指数级扩展问题，并引入新的批归一化策略处理可变域输入，实现了在任意克利福德代数空间中灵活高效的函数逼近。

    

    我们提出了克利福德柯尔莫哥洛夫-阿诺德网络，这是一种在任意克利福德代数空间中进行函数逼近的灵活而高效的架构。我们提出使用随机化拟蒙特卡罗网格生成方法，以解决与高维代数相关的指数级扩展问题。我们的ClKAN还引入了新的批归一化策略来处理可变域输入。ClKAN在科学发现和工程领域具有应用价值，并在合成任务和物理启发任务中得到了验证。

    arXiv:2602.05977v3 Announce Type: replace-cross  Abstract: We introduce Clifford Kolmogorov-Arnold Network (ClKAN), a flexible and efficient architecture for function approximation in arbitrary Clifford Algebra spaces. We propose the use of Randomized Quasi-Monte Carlo grid generation as a solution to the exponential scaling associated with higher-dimensional algebras. Our ClKAN also introduces new batch normalization strategies to deal with variable domain input. ClKAN finds application in scientific discovery and engineering, and is validated in synthetic and physics-inspired tasks.
    
[^300]: OmniSpectra：面向通用光谱表示学习的统一基础模型

    OmniSpectra: A Unified Foundation Model for Universal Spectra Representation Learning

    [https://arxiv.org/abs/2601.15351](https://arxiv.org/abs/2601.15351)

    OmniSpectra 提出一种以波长为原生坐标的连续坐标掩码自编码器基础模型，通过将对数基上的物理坐标与连续旋转位置编码结合，使宇宙学红移引发的共享平移在相对距离中自然抵消，从而解决了跨异构仪器光谱表示学习中的坐标失配这一结构性难题。

    

    标准序列模型通过词元序号而非物理坐标来编码位置，这通常迫使研究人员对连续物理信号（例如在不同异构仪器上观测到的光谱）进行破坏性重采样。我们将这种坐标失配确定为跨仪器表示学习中的一个结构性局限，并提出 OmniSpectra——一个以波长为本的光谱表示学习基础模型：一种连续坐标掩码自编码器，摒弃了序列建模中的整数像素索引。将物理坐标映射到对数基上，可将宇宙学红移引起的乘性缩放转化为统一的加性平移。在连续旋转位置嵌入中，这一共享平移会从相对词元距离中抵消，而不会产生依赖网格的位置畸变。我们进一步利用一种物理……（原文摘要在此处截断）

    arXiv:2601.15351v2 Announce Type: replace-cross  Abstract: Standard sequence models encode position by token rank rather than physical coordinate, typically forcing practitioners to destructively resample continuous physical signals such as spectra observed across heterogeneous instruments. We identify this coordinate mismatch as a structural limitation in cross-instrument representation learning, and propose OmniSpectra, a wavelength-native foundation model for spectral representation learning: a continuous-coordinate masked autoencoder that discards the integer pixel indexing of sequence modeling. Mapping physical coordinates to a logarithmic basis turns the multiplicative scaling induced by cosmological redshift into a uniform additive shift. In continuous rotary position embeddings, this shared shift cancels from relative token distances rather than producing grid-dependent positional distortion. We further counter the spectral bias inherent to implicit neural decoders with a physi
    
[^301]: SpatialMem：面向语言接地与问答的度量对齐长时程视频记忆

    SpatialMem: Metric-Aligned Long-Horizon Video Memory for Language Grounding and QA

    [https://arxiv.org/abs/2601.14895](https://arxiv.org/abs/2601.14895)

    SpatialMem 仅凭随手拍摄的第一人称 RGB 视频，即可构建融合三维几何、语义与语言的度量级可查询记忆，无需特殊传感器就能完成空间关系推理、语言引导检索与长时程视频问答，并在 VSI-Bench 和 OpenEQA 等基准上取得领先表现。

    

    SpatialMem 是一个基于记忆的系统，它将三维几何、语义与语言整合到一个统一且可查询的表示中。以随意拍摄的第一人称视角 RGB 视频作为输入，SpatialMem 重建出具有度量尺度的室内场景，将结构性三维锚点（墙壁、门、窗）识别为第一层支架，并将开放词汇的对象节点与证据图像块、视觉嵌入以及两层描述关联到三维坐标上，从而在无需特殊传感器的情况下实现空间关系推理、语言引导检索和问答。我们在公开基准上进行了大规模测试。在 VSI-Bench（288 个视频、5,130 个问题）上，仅使用 RGB 的 SpatialMem 总分达到 37.5，且其优势恰恰体现在度量记忆应当发挥价值的地方（在房间大小任务上为 41.4，而同等规模的帧级模型仅为 24.4）。在 OpenEQA 的情景记忆任务（953 个问题）上，序列化为文本的记忆低于 t……（原文摘要在此处被截断）

    arXiv:2601.14895v4 Announce Type: replace-cross  Abstract: SpatialMem is a memory-based system which integrates 3D geometry, semantics, and language with one another in one combined, queryable representation. Starting with casually recorded egocentric RGB video, SpatialMem recreates metrically scaled interior scenes, recognizes structural 3D anchors (walls, doors, windows) as the first-layer scaffold, and associates open-vocabulary object nodes with evidence patches, visual embeddings, and two-layer descriptions to 3D coordinates, enabling spatial-relation reasoning, language-directed retrieval, and QA without special sensors. We test at scale on public benchmarks. On VSI-Bench ($288$ videos, $5{,}130$ questions) SpatialMem scores $37.5$ overall with RGB alone, and its margin is precisely where a metric memory should earn it (room size $41.4$ vs.\ $24.4$ with a same-size frame model). On the episodic-memory task of OpenEQA ($953$ questions) the memory serialised to text is lower than t
    
[^302]: 通过主动回顾学习增强大语言模型对长上下文的鲁棒理解

    Towards robust long-context understanding of large language model via active recap learning

    [https://arxiv.org/abs/2601.13734](https://arxiv.org/abs/2601.13734)

    提出主动回顾学习（ARL）框架，通过持续预训练中的针对性序列构建与推理时的自主回溯总结，使大语言模型在段落间建立递归记忆机制，显著提升长上下文理解能力（RULER提升最高达129.7%）。

    

    本文提出了主动回顾学习，这是一个用于增强大语言模型理解长上下文能力的框架。ARL使模型能够在持续预训练阶段通过有针对性的序列构建来回顾和总结先前的内容，并在推理阶段进行回溯式总结。首先，基于长前向上下文与短前向上下文之间的损失差距，在准备好的长上下文中识别关键token，并找出与之最相关的先前段落，然后使用大语言模型对其进行总结。其次，ARL使模型具备在推理过程中自主生成和利用这些回溯式总结的能力，从而在段落之间建立起递归记忆机制。实验结果显示了显著的性能提升，ARL在RULER基准上取得了高达129.7%的改进，在LongBench上取得了11%–13%的改进。总体而言，ARL提供了一种简单而有效的基于持续预训练的方法，以增强模型的长上下文理解能力。

    arXiv:2601.13734v2 Announce Type: replace-cross  Abstract: In this paper, we propose active recap learning (ARL), a framework for enhancing large language model (LLM) in understanding long contexts. ARL enables models to revisit and summarize earlier content through targeted sequence construction during continued pretraining and retrospective summarization at inference. First, we identify key tokens in prepared long context based on loss gaps between long and short forward contexts and find the most relevant preceding paragraphs, then summarize them using an LLM. Second, ARL equips models with the ability to autonomously generate and utilize these retrospective summaries during inference, thereby establishing a recursive memory mechanism across paragraphs. Experimental results show substantial gains, with ARL achieving an improvement of up to 129.7\% on RULER and 11--13\% on LongBench. Overall, ARL offers a simple yet effective continued pretraining-based approach to strengthen long-co
    
[^303]: 注意力空间对比引导：面向大型视觉-语言模型的高效幻觉缓解方法

    Attention-space Contrastive Guidance for Efficient Hallucination Mitigation in LVLMs

    [https://arxiv.org/abs/2601.13707](https://arxiv.org/abs/2601.13707)

    提出注意力空间对比引导（ACG），一种无需训练的单次前向传播方法，通过在自注意力层构建图像条件与纯文本对比注意力路径并结合轻量级正交投影，高效缓解大型视觉-语言模型的幻觉问题。

    

    大型视觉-语言模型（LVLMs）中的幻觉通常在语言先验压制视觉证据时产生，导致物体误识别和与视觉不符的描述。我们将幻觉缓解问题构建为一种对比引导，将生成过程引导向视觉上有依据且语义上忠实于图像的文本。我们提出了注意力空间对比引导（ACG），这是一种无需训练、仅需单次前向传播的方法，直接在自注意力层中运作，而诱发幻觉的跨模态偏差正是在这些层中出现的。ACG 在单次前向传播中同时构建图像条件注意力路径和近似的纯文本注意力路径，从而能够在错误于输出层累积之前实现高效引导。由于这种基于掩码的替代方法可能引入近似偏差，我们进一步应用轻量级的正交投影来抑制与纯文本路径对齐的成分，从而产生更忠实于视觉的输出。

    arXiv:2601.13707v3 Announce Type: replace-cross  Abstract: Hallucinations in large vision--language models (LVLMs) often arise when language priors dominate over visual evidence, leading to object misidentification and visually inconsistent descriptions. We address this problem by framing hallucination mitigation as contrastive guidance that steers generation toward visually grounded and semantically faithful text. We propose Attention-space Contrastive Guidance (ACG), a training-free, single-pass method that operates directly in self-attention layers, where hallucination-inducing cross-modal biases emerge. ACG constructs both image-conditioned and approximate text-only attention paths within a single forward pass, enabling efficient guidance before errors accumulate at the output layer. Because this masking-based surrogate can introduce approximation bias, we further apply a lightweight orthogonal projection that suppresses components aligned with the text-only path, yielding a more v
    
[^304]: PAIR-SAFE：一种用于运行时审计与优化AI介导心理健康支持的双智能体方法

    PAIR-SAFE: A Paired-Agent Approach for Runtime Auditing and Refining AI-Mediated Mental Health Support

    [https://arxiv.org/abs/2601.12754](https://arxiv.org/abs/2601.12754)

    提出PAIR-SAFE双智能体框架，由基于临床验证MITI-4标准的评判者智能体实时审计并指导修正LLM生成的心理健康支持回复，显著提升了伙伴关系、协作性和整体关系质量。

    

    大语言模型越来越多地被用于心理健康支持，但它们可能产生过于指令化、不一致或与临床实践不符的回复。大多数风险缓解方法依赖于通过训练或提示工程实现的隐式对齐，其运行时问责能力有限。我们提出了PAIR-SAFE，这是一个双智能体框架，将响应者智能体与基于临床验证的动机性访谈治疗完整性标准（MITI-4）构建的评判者智能体相结合。评判者对每个回复进行审计，并发出“允许”或“修改”的决定，以指导实时优化。我们使用从人工标注数据中构建的求助者模拟器来评估交互过程。评判者监督下的交互在关键MITI维度上显示出显著改进，包括伙伴关系、寻求协作以及整体关系质量。我们的定量研究结果得到了专家评估的印证，凸显了运行时监督的细微之处。

    arXiv:2601.12754v2 Announce Type: replace-cross  Abstract: LLMs are increasingly used for mental health support, yet they can produce responses that are overly directive, inconsistent, or clinically misaligned. Most risk mitigation approaches rely on implicit alignment through training or prompting, offering limited runtime accountability. We introduce PAIR-SAFE, a paired-agent framework that integrates a Responder agent with a Judge agent grounded in the clinically validated Motivational Interviewing Treatment Integrity (MITI-4). The Judge audits each response and issues \allow{} or \revise{} decisions to guide real-time refinement. We evaluate interactions using a support-seeker simulator derived from human-annotated data. The Judge-supervised interactions show significant improvements in key MITI dimensions, including Partnership, Seek Collaboration, and overall Relational quality. Our quantitative findings are corroborated by expert evaluation, highlighting nuances of runtime super
    
[^305]: 混合量子-经典神经网络中量子的作用：一项实证评估

    The Role of Quantum in Hybrid Quantum-Classical Neural Networks: An Empirical Assessment

    [https://arxiv.org/abs/2601.04732](https://arxiv.org/abs/2601.04732)

    本文通过理想无噪声模拟下的严谨统计研究，在三个医学二分类任务上系统评估了常用混合量子-经典神经网络架构，以揭示量子组件对模型整体性能的实际贡献。

    

    量子机器学习已成为近期量子硬件的一个有前景的应用领域，特别是通过同时利用经典与量子处理的混合量子-经典模型。尽管已有众多混合架构被提出并在基准任务上成功演示，但关于量子组件对这些模型整体性能的具体贡献，仍是一个重要的未解问题。在本工作中，我们旨在通过在理想的、无噪声模拟环境下开展严谨的统计研究，阐明量子处理在混合量子-经典神经网络（HQNN）架构中的影响。我们在涵盖医学信号数据以及平面和体积图像的三个二分类医学任务上，系统地评估了一系列常用的HQNN架构。在此过程中，我们检验了可归因于经典与量子方面（例如……）的影响。

    arXiv:2601.04732v2 Announce Type: replace-cross  Abstract: Quantum machine learning has emerged as a promising application domain for near-term quantum hardware, particularly through hybrid quantum-classical models that leverage both classical and quantum processing. Although numerous hybrid architectures have been proposed and demonstrated successfully on benchmark tasks, a significant open question remains regarding the specific contribution of quantum components to the overall performance of these models. In this work, we aim to shed light on the impact of quantum processing within hybrid quantum-classical neural network architectures through a rigorous statistical study under ideal, noise-free simulation. We systematically assess a selection of commonly used HQNN architectures on three binary medical classification tasks covering medical signal data as well as planar and volumetric images. In doing so, we examine the influence attributable to classical and quantum aspects such as e
    
[^306]: Dementia-R1：基于非结构化临床笔记的强化预训练与推理，用于真实世界痴呆症预后预测

    Dementia-R1: Reinforced Pretraining and Reasoning from Unstructured Clinical Notes for Real-World Dementia Prognosis

    [https://arxiv.org/abs/2601.03018](https://arxiv.org/abs/2601.03018)

    提出了Dementia-R1框架，通过冷启动强化学习策略先预训练模型预测可验证的临床指标，再基于非结构化临床笔记进行痴呆症的纵向预后预测，实现了最佳整体性能。

    

    尽管大型语言模型（LLMs）在临床文本理解方面表现出强大的性能，但它们在痴呆症预后等纵向预测任务上仍然面临困难，这类任务需要对多次就诊中复杂、非单调的症状轨迹进行推理。标准的监督式训练缺乏对症状演变的显式标注，而直接的强化学习（RL）则受到稀疏二值奖励的阻碍。为了应对这一挑战，我们提出了Dementia-R1，一个基于强化学习的框架，用于从非结构化临床笔记中进行纵向痴呆症预后预测。我们的方法采用冷启动强化学习策略，预训练模型以预测从患者病史中提取的可验证临床指标，从而在确定最终临床状态之前增强模型对疾病进展进行推理的能力。大量实验表明，Dementia-R1在AMC真实世界非结构化临床数据集上取得了最佳的整体性能。

    arXiv:2601.03018v3 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) have shown strong performance on clinical text understanding, they struggle with longitudinal prediction tasks such as dementia prognosis, which require reasoning over complex, non-monotonic symptom trajectories across multiple visits. Standard supervised training lacks explicit annotations for symptom evolution, while direct Reinforcement Learning (RL) is hindered by sparse binary rewards. To address this challenge, we introduce Dementia-R1, an RL-based framework for longitudinal dementia prognosis from unstructured clinical notes. Our approach adopts a Cold-Start RL strategy that pre-trains the model to predict verifiable clinical indices extracted from patient histories, enhancing the capability to reason about disease progression before determining the final clinical status. Extensive experiments show that Dementia-R1 achieves the best overall performance on the AMC real-world unstructured
    
[^307]: MedInsightBench：通过多模态医学数据中的多步骤洞察发现评估医学分析智能体

    MedInsightBench: Evaluating Medical Analytics Agents Through Multi-Step Insight Discovery in Multimodal Medical Data

    [https://arxiv.org/abs/2512.13297](https://arxiv.org/abs/2512.13297)

    提出首个用于评估大型多模态模型和智能体框架在多模态医学图像数据上进行多步骤医学洞察发现能力的基准MedInsightBench，包含332个精心标注的医学案例，并揭示现有模型表现仍然有限。

    

    在医学数据分析中，从复杂的多模态数据集中提取深层洞察，对于改善患者护理、提高诊断准确性和优化医疗运营至关重要。然而，目前缺乏专门设计用于评估大型多模态模型（LMMs）发现医学洞察能力的高质量数据集。在本文中，我们提出了MedInsightBench，这是首个包含332个精心筛选的医学案例的基准数据集，每个案例均标注了经过缜密设计的洞察。该基准旨在评估LMMs和智能体框架分析多模态医学图像数据的能力，包括提出相关问题、解读复杂发现，以及综合生成可操作的洞察与建议。我们的分析表明，现有的LMMs在MedInsightBench上的表现较为有限，这主要归因于它们在提取多步骤洞察方面存在困难。

    arXiv:2512.13297v2 Announce Type: replace  Abstract: In medical data analysis, extracting deep insights from complex, multi-modal datasets is essential for improving patient care, increasing diagnostic accuracy, and optimizing healthcare operations. However, there is currently a lack of high-quality datasets specifically designed to evaluate the ability of large multi-modal models (LMMs) to discover medical insights. In this paper, we introduce MedInsightBench, the first benchmark that comprises 332 carefully curated medical cases, each annotated with thoughtfully designed insights. This benchmark is intended to evaluate the ability of LMMs and agent frameworks to analyze multi-modal medical image data, including posing relevant questions, interpreting complex findings, and synthesizing actionable insights and recommendations. Our analysis indicates that existing LMMs exhibit limited performance on MedInsightBench, which is primarily attributed to their challenges in extracting multi-s
    
[^308]: 演示：生成式AI基于用户偏好辅助放射治疗计划设计

    Demo: Generative AI helps Radiotherapy Planning with User Preference

    [https://arxiv.org/abs/2512.08996](https://arxiv.org/abs/2512.08996)

    本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。

    

    放射治疗计划是一个高度复杂的过程，在不同机构和不同计划制定者之间往往存在显著差异。现有的大多数用于三维剂量预测的深度学习方法在训练时依赖参考计划作为真值（ground truth），这可能会无意中使模型偏向特定的计划风格或机构偏好。在本研究中，我们提出了一种新颖的生成式模型，仅根据用户自定义的偏好风格来预测三维剂量分布。这些可定制的偏好使计划制定者能够优先考虑危及器官（OARs）与计划靶区（PTVs）之间的特定权衡，提供了更大的灵活性和个性化。我们的方法旨在与临床治疗计划系统无缝集成，帮助用户高效地生成高质量计划。对比评估表明，我们的方法在适应性方面可以超越Varian RapidPlan模型。

    arXiv:2512.08996v2 Announce Type: replace-cross  Abstract: Radiotherapy planning is a highly complex process that often varies significantly across institutions and individual planners. Most existing deep learning approaches for 3D dose prediction rely on reference plans as ground truth during training, which can inadvertently bias models toward specific planning styles or institutional preferences. In this study, we introduce a novel generative model that predicts 3D dose distributions based solely on user-defined preference flavors. These customizable preferences enable planners to prioritize specific trade-offs between organs-at-risk (OARs) and planning target volumes (PTVs), offering greater flexibility and personalization. Designed for seamless integration with clinical treatment planning systems, our approach assists users in generating high-quality plans efficiently. Comparative evaluations demonstrate that our method can surpasses the Varian RapidPlan model in both adaptability
    
[^309]: 测试Transformer在有根树算术序列上的可学习性

    Testing Transformer Learnability on the Arithmetic Sequence of Rooted Trees

    [https://arxiv.org/abs/2512.01870](https://arxiv.org/abs/2512.01870)

    该论文将自然数通过迭代素数分解映射为有根树构成的确定性算术文本，并证明从头训练的GPT-2模型能以约0.4的词准确率学习该序列，且在远超训练范围的测试数据上保持稳定的外推性能。

    

    我们研究了Transformer网络能否学习由自然数迭代素数分解所生成的确定性树序列。每个整数被映射为一棵有根平面树，由此得到的序列 $\mathbb{N}\mathscr{T}$ 定义了一个具有可测量统计结构的算术文本。我们在前 $10^{11}$ 个元素上从头训练了一个Transformer网络（GPT-2架构），并在下一个词预测和掩码词预测任务上进行评估，以隐马尔可夫模型作为基线，并针对上下文窗口、数据集大小、词表大小和模型大小进行了缩放分析。该模型达到了约 $0.4$ 的词准确率，远高于基线，且其性能在位于 $10^{13}$-$10^{15}$ 的测试块上保持稳定，远超训练区间。此外，训练模型所赋予的似然能够将该算术文本与两个对照（合成词元序列等）区分开来。

    arXiv:2512.01870v3 Announce Type: replace  Abstract: We study whether a transformer network can learn the deterministic sequence of trees generated by the iterated prime factorization of the natural numbers. Each integer is mapped into a rooted planar tree and the resulting sequence $\mathbb{N}\mathscr{T}$ defines an arithmetic text with measurable statistical structure. A transformer network (the GPT-2 architecture) is trained from scratch on the first $10^{11}$ elements and evaluated on next-token prediction and masked-token prediction tasks, with a Hidden Markov Model as baseline and a scaling analysis over context window, dataset size, vocabulary size and model size. The model reaches a word accuracy of about $0.4$, well above the baseline, and its performance remains stable on test blocks located at $10^{13}$-$10^{15}$, far beyond the training interval. Moreover, the likelihood assigned by the trained model separates the arithmetic text from two controls: synthetic token sequences
    
[^310]: EvalCards：一个用于标准化评估报告的框架

    EvalCards: A Framework for Standardized Evaluation Reporting

    [https://arxiv.org/abs/2511.21695](https://arxiv.org/abs/2511.21695)

    本文提出 EvalCards 框架，围绕可复现性、可访问性和可问责性三个核心维度规范 AI 模型评估报告，并提供简洁评估摘要的初步格式，以推动未来评估报告标准的建立。

    

    评估报告为通用人工智能模型的使用提供指导，然而不一致的报告方式往往使关键问题得不到解答。通过对以往工作的回顾，我们指出了当今评估报告实践中存在的三个核心问题——可复现性、可访问性和可问责性，并检视了现有报告工件在解决这些问题上的不足。我们提出了 EvalCards 框架，阐述了其关键维度，并给出了一种简洁评估摘要的初步格式。最后，我们讨论了该框架如何为未来评估报告标准的制定提供指导。

    arXiv:2511.21695v2 Announce Type: replace-cross  Abstract: Evaluation reports guide the use of general-purpose AI (GPAI) models, yet inconsistent reporting often leaves essential questions unanswered. Through a review of prior work, we identify three central issues in evaluation reporting practices today, reproducibility, accessibility, and accountability, and examine how existing reporting artifacts fall short in addressing them. We introduce the EvalCards framework, outlining its key dimensions and presenting a preliminary format for concise evaluation summaries. We conclude by discussing how the framework can guide the future development of evaluation reporting standards.
    
[^311]: 超越单体调度器：面向进程调度优化的轻量级自适应策略切换器

    Beyond the Monolithic Scheduler: Lightweight Adaptive Policy Switcher for Process Scheduling Optimization

    [https://arxiv.org/abs/2511.11628](https://arxiv.org/abs/2511.11628)

    本文提出轻量级自适应策略切换器LAPS，通过在预定义专家策略间动态切换并将繁重的动态权衡计算转移至离线阶段，在不引入AI辅助调度高开销的前提下，实现了对边缘系统动态混合工作负载的高效进程调度优化。

    

    现代操作系统的进程调度器通常采用静态的“一种策略适用于所有场景”的设计，这在当代边缘系统高度动态且混合的工作负载下不可避免地损害了性能。向动态自适应转型面临两个根本性挑战：针对特定场景的静态策略缺乏稳健的运行时适应性，而计算密集型的AI辅助策略则会给资源受限的设备带来不可接受的执行开销。为了从结构上解决这一困境，我们提出了轻量级自适应策略切换器（LAPS），这是一个在预定义的专家策略之间进行动态切换的框架，从而将高频的上下文切换与低频的策略转换解耦，并摊销了转换开销。为了消除高昂的在线计算开销，LAPS通过离线准备阶段预先解决所有复杂的动态权衡，构建了一个严格超……

    arXiv:2511.11628v2 Announce Type: replace-cross  Abstract: Modern operating system process schedulers typically enforce a static, ``one-policy-fits-all'' design, which inevitably compromises performance under the highly dynamic and mixed workloads of contemporary edge systems. Transitioning to dynamic adaptation faces two fundamental challenges: scenario specific static policies lack robust runtime adaptability, whereas computationally intensive AI-assisted policies introduce unacceptable execution overhead for resource-constrained devices. To structurally resolve this dilemma, we propose the Lightweight Adaptive Policy Switcher (LAPS), a framework that dynamically switches between predefined expert policies, thus decoupling high frequency context switching from low-frequency policy transitions and amortizing transitional overheads. To eliminate prohibitive online computations, LAPS pre-resolves all heavy dynamic tradeoffs via an offline preparation phase, constructing a strictly overh
    
[^312]: 具有不确定规则与前提的结构化论证框架的表达能力研究结果

    Expressivity Results for Structured Argumentation Frameworks with Uncertain Rules and Premises

    [https://arxiv.org/abs/2510.18631](https://arxiv.org/abs/2510.18631)

    该论文提出了一个同时适用于抽象与结构化论证形式体系的直观比较表达能力概念，通过将论证的不确定性根植于规则与前提层面，证明了此类结构化不确定性论证框架与抽象不确定性论证模型之间表达能力的正反两方面的比较结果。

    

    在形式化论证中对定性不确定性进行建模，对于实际应用和理论理解都至关重要。然而，现有工作大多聚焦于不确定性下论证的抽象模型。顺应文献中的最新趋势，我们着手解决研究这些抽象模型的合理实例化这一开放性问题。为此，我们在结构化模型中将论证的不确定性根植于其组成部分：规则与前提。我们的主要概念性贡献是提供了一个直观的比较表达能力概念，该概念同时适用于抽象形式体系和结构化形式体系。大致而言，如果一个处理定性不确定性论证的形式体系能够生成与另一个形式体系相同的抽象论证框架集合（即完备化），那么前者至少与后者具有同等的表达能力。在技术方面，我们证明了比较抽象模型与结构化模型表达能力的一系列负面和正面结果。

    arXiv:2510.18631v3 Announce Type: replace  Abstract: Modelling qualitative uncertainty in formal argumentation is essential both for practical applications and theoretical understanding. Yet, most existing work focuses on abstract models of argumentation under uncertainty. Following a recent trend in the literature, we tackle the open question of studying plausible instantiations of these abstract models. To do so, we root the uncertainty of arguments in their components in structured models: rules and premises. Our main conceptual contribution is to provide an intuitive notion of comparative expressivity suitable for both abstract and structured formalisms. Roughly, one formalism for arguing with qualitative uncertainty is at least as expressive as another if the former can generate the same set of abstract argumentation frameworks (completions) as the latter. On the technical side, we prove negative and positive results comparing the expressivity of abstract and structured models of 
    
[^313]: 人机交互与心理三元组：认知、行为与情感影响的多领域综述

    Human-AI Interactions and the Psychological Triad: A Multi-Domain Survey of Cognitive, Behavioral, and Emotional Impacts

    [https://arxiv.org/abs/2510.17753](https://arxiv.org/abs/2510.17753)

    本文提出基于心理三元组（认知、行为、情感）的TRIAD概念框架，系统综述了人机交互的双面影响：AI虽能增强记忆、创造力和参与度并缓解压力，但也带来批判性思维下降、技能退化和情感依赖等系统性风险。

    

    随着人机交互日益重塑社会与研究格局，过度依赖、认知卸载以及人类能动性的悄然退化等风险变得愈发明显。本文通过引入一个基于心理三元组（认知、行为和情感）的新颖概念框架来应对这些挑战。通过对近期研究的全面综述，我们观察到，尽管AI能够显著提升记忆、创造力和参与度等成果，但它同时也带来了系统性风险，包括批判性思维能力下降、技能退化以及焦虑增加。情感层面的结果同样好坏参半：AI系统在提供支持和缓解压力的同时，也引发了对依赖性、不当依恋和伦理监督的重大担忧。通过将这些发现综合为TRIAD模型——一个由评估、执行等环节构成的递归式框架（原文摘要在此处截断）……

    arXiv:2510.17753v3 Announce Type: replace-cross  Abstract: As human-AI interactions increasingly reshape social and research landscapes, the risks of overreliance, cognitive offloading, and the nuanced degradation of human agency have become more apparent. This paper addresses these challenges by introducing a novel conceptual framework based on the psychological triad: cognition, behavior, and emotion. Through a comprehensive survey of recent research, we observe that while AI can substantially enhance outcomes such as memory, creativity, and engagement, it simultaneously introduces systemic risks, including diminished critical thinking, skill erosion, and increased anxiety. Emotional outcomes are shown to be similarly mixed, with AI systems offering support and stress reduction while raising critical concerns regarding dependency, inappropriate attachments, and ethical oversight. By synthesizing these findings into the TRIAD model, a recursive framework of appraisal, execution, and r
    
[^314]: 动态关系启动机制改进多元时间序列中的Transformer

    Dynamic Relational Priming Improves Transformer in Multivariate Time Series

    [https://arxiv.org/abs/2509.12196](https://arxiv.org/abs/2509.12196)

    本文提出动态关系启动注意力机制（prime attention），通过为每个token对交互定制动态表示，显著改进了Transformer在多元时间序列建模中捕捉异构通道间依赖关系的能力。

    

    标准Transformer中的注意力机制采用静态的token表示，这些表示在每一层的所有成对计算中保持不变。这限制了其表征与每个token对交互中潜在多样关系动态之间的对齐。尽管标准注意力在关系相对同质的领域表现出色，但其静态关系学习难以捕捉多元时间序列（MTS）数据中多样且异构的通道间依赖关系——在多元时间序列中，同一系统内不同的通道对交互可能由完全不同的物理定律或时间动态所支配。为了使注意力机制更好地与这类领域现象对齐，我们提出了带动态关系启动的注意力机制（prime attention）。与标准注意力中每个token在所有成对交互中呈现相同表示不同，prime attention会为每个（token对交互）定制表示。

    arXiv:2509.12196v3 Announce Type: replace-cross  Abstract: Standard attention mechanisms in transformers employ static token representations that remain unchanged across all pair-wise computations in each layer. This limits their representational alignment with the potentially diverse relational dynamics of each token-pair interaction. While they excel in domains with relatively homogeneous relationships, standard attention's static relational learning struggles to capture the diverse, heterogeneous inter-channel dependencies of multivariate time series (MTS) data--where different channel-pair interactions within a single system may be governed by entirely different physical laws or temporal dynamics. To better align the attention mechanism for such domain phenomena, we propose attention with dynamic relational priming (prime attention). Unlike standard attention where each token presents an identical representation across all of its pair-wise interactions, prime attention tailors each
    
[^315]: AU-Harness：一个用于高效统一评估音频大语言模型的开源工具包

    AU-Harness: An Open-Source Toolkit for Efficient and Unified Evaluation of AUdio-LLMs

    [https://arxiv.org/abs/2509.08031](https://arxiv.org/abs/2509.08031)

    AU-Harness是一个开源的音频大语言模型统一评估框架，通过高效流水线将评估处理时间减少77%，并解决了现有工具在多轮对话支持不足和框架缺乏标准化方面的关键局限。

    

    大音频语言模型（LALMs）正在快速发展，但由于评估工具包效率低下且缺乏标准化，限制了公平比较和系统性评估，对其进行评估仍然具有挑战性。现有的评估框架存在三个关键局限性：（1）缓慢且低效的处理流程，成为大规模研究的瓶颈；（2）对多轮对话的支持不足，导致关于LALMs中跨轮上下文整合以及长对话中性能动态变化的基本问题尚未得到解答；（3）缺乏能够跟上LALMs和音频基准快速增长步伐的统一且可扩展的评估框架。为了解决这些问题，我们推出了AU-Harness，一个高效且全面的LALMs评估框架。我们的实证研究表明，AU-Harness相比现有评估工具包将评估处理时间减少了77%，使...

    arXiv:2509.08031v4 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient and nonstandardized toolkits that limit fair comparison and systematic assessment. Existing evaluation frameworks exhibit three critical limitations: (1) slow and inefficient processing pipeline that bottlenecks large-scale studies, (2) inadequate multi-turn dialogue support, leaving fundamental questions about cross-turn context integration and performance dynamics over extended conversations in LALMs unanswered; and (3) the absence of unified and scalable evaluation framework capable of keeping pace with the rapid growth of both LALMs and audio benchmarks. To address these issues, we introduce AU-Harness, an efficient and comprehensive evaluation framework for LALMs. Our empirical studies demonstrate that AU-Harness achieves 77% reduction in evaluation processing time over existing evaluation toolkits, enablin
    
[^316]: MEDMKG：基于多模态知识图谱的医学知识利用基准测试

    MEDMKG: Benchmarking Medical Knowledge Exploitation with Multimodal Knowledge Graph

    [https://arxiv.org/abs/2505.17214](https://arxiv.org/abs/2505.17214)

    提出了医学多模态知识图谱MEDMKG，通过融合MIMIC-CXR的视觉与文本数据以及UMLS的结构化临床知识构建图谱，并引入邻居感知过滤算法保证图谱质量，为医学知识利用提供了首个多模态知识图谱基准。

    

    医学深度学习模型在执行知识密集型临床任务时，高度依赖领域特定知识才能取得良好表现。先前的研究主要利用单模态知识图谱（如统一医学语言系统UMLS）来提升模型性能。然而，多模态医学知识图谱的整合在很大程度上仍未被充分探索，其主要原因是缺乏将影像数据与临床概念相链接的资源。为填补这一空白，我们提出了MEDMKG——一个医学多模态知识图谱，它通过多阶段构建流程将视觉与文本医学信息统一起来。MEDMKG融合了来自MIMIC-CXR的丰富多模态数据与来自UMLS的结构化临床知识，同时利用基于规则的工具和大语言模型进行准确的概念提取与关系建模。为确保图谱的质量与紧凑性，我们引入了邻居感知过滤——一种新颖的过滤算法……

    arXiv:2505.17214v2 Announce Type: replace  Abstract: Medical deep learning models depend heavily on domain-specific knowledge to perform well on knowledge-intensive clinical tasks. Prior work has primarily leveraged unimodal knowledge graphs, such as the Unified Medical Language System (UMLS), to enhance model performance. However, integrating multimodal medical knowledge graphs remains largely underexplored, mainly due to the lack of resources linking imaging data with clinical concepts. To address this gap, we propose MEDMKG, a Medical Multimodal Knowledge Graph that unifies visual and textual medical information through a multi-stage construction pipeline. MEDMKG fuses the rich multimodal data from MIMIC-CXR with the structured clinical knowledge from UMLS, utilizing both rule-based tools and large language models for accurate concept extraction and relationship modeling. To ensure graph quality and compactness, we introduce Neighbor-aware Filtering (NaF), a novel filtering algorith
    
[^317]: 面向国际空间站轨迹优化的深度学习热启动方法

    Deep Learning Warm Starts for Trajectory Optimization on the International Space Station

    [https://arxiv.org/abs/2505.05588](https://arxiv.org/abs/2505.05588)

    本研究在国际空间站上首次实现在轨演示，利用神经网络热启动加速Astrobee自由飞行机器人的轨迹优化，通过学习顺序凸规划问题的结构来实现满足安全约束的星载实时轨迹生成。

    

    轨迹优化是现代机器人自主性的基石，它使系统能够在满足安全性和物理约束的前提下实时计算轨迹与控制指令。然而，由于其繁重的计算需求超出了大多数飞行计算机的能力，轨迹优化在航天应用中的使用一直非常有限。在这项工作中，我们展示了首次在太空中进行的在轨演示：利用基于机器学习的热启动方法，为国际空间站（ISS）上的Astrobee自由飞行机器人加速轨迹优化。我们提出了一种数据驱动的最优控制方法，训练神经网络来学习使用顺序凸规划（SCP）求解的轨迹生成问题的结构。在星载设备上，训练好的神经网络预测轨迹生成问题的解，并依靠SCP求解器来强制执行系统的安全约束。

    arXiv:2505.05588v4 Announce Type: replace-cross  Abstract: Trajectory optimization is a cornerstone of modern robot autonomy, enabling systems to compute trajectories and controls in real-time while respecting safety and physical constraints. However, it has seen limited usage in spaceflight applications due to its heavy computational demands that exceed the capability of most flight computers. In this work, we provide results on the first in-space demonstration of using machine learning-based warm starts for accelerating trajectory optimization for the Astrobee free-flying robot onboard the International Space Station (ISS). We formulate a data-driven optimal control approach that trains a neural network to learn the structure of the trajectory generation problem being solved using sequential convex programming (SCP). Onboard, this trained neural network predicts solutions for the trajectory generation problem and relies on using the SCP solver to enforce safety constraints for the sy
    
[^318]: 灾难性过拟合、熵差距与参与率：用于快速对抗训练的无噪声 $l^p$ 范数解决方案

    Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training

    [https://arxiv.org/abs/2505.02360](https://arxiv.org/abs/2505.02360)

    本文提出一种纯粹通过控制 $l^p$ 训练范数来缓解快速对抗训练中灾难性过拟合的无噪声新方案，并揭示灾难性过拟合源于高度集中的梯度与激进范数约束的相互作用。

    

    对抗训练是鲁棒深度学习的基石，但快速梯度符号法（FGSM）等快速方法常常遭受灾难性过拟合（CO）的困扰，即模型对单步攻击具有鲁棒性，但在面对多步攻击变体时会失效。虽然现有解决方案依赖于噪声注入、正则化或梯度裁剪，我们提出了一种新颖的解决方案，纯粹通过控制 $l^p$ 训练范数来缓解CO。我们的研究源于一个实证观察：CO在 $l^{\infty}$ 范数下比在 $l^2$ 范数下更为普遍。利用这一洞察，我们开发了一个将广义 $l^p$ 攻击建模为不动点问题的框架，并设计了 $l^p$-FGSM 攻击以理解从 $l^2$ 到 $l^{\infty}$ 的转换机制。由此得出我们的核心洞察：当高度集中的梯度（即信息局限于少数维度）与激进的范数约束相互作用时，CO就会出现。（注：原文摘要在此处截断）

    arXiv:2505.02360v3 Announce Type: replace-cross  Abstract: Adversarial training is a cornerstone of robust deep learning, but fast methods like the Fast Gradient Sign Method (FGSM) often suffer from Catastrophic Overfitting (CO), where models become robust to single-step attacks but fail against multi-step variants. While existing solutions rely on noise injection, regularization, or gradient clipping, we propose a novel solution that purely controls the $l^p$ training norm to mitigate CO. Our study is motivated by the empirical observation that CO is more prevalent under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we develop a framework for generalized $l^p$ attack as a fixed point problem and craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to $l^{\infty}$. This leads to our core insight: CO emerges when highly concentrated gradients, where information localizes in few dimensions, interact with aggressive norm constraints. By quantify
    
[^319]: 基于视觉大语言模型增强表格识别：一个基准与邻居引导的工具链推理器

    Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner

    [https://arxiv.org/abs/2412.20662](https://arxiv.org/abs/2412.20662)

    本文提出了一个评估视觉大语言模型表格识别能力的分层设计基准，并创新性地提出邻居引导的工具链推理器（NGTR）框架，通过整合轻量级视觉工具和迁移相似邻居的工具选择经验来缓解低质量图像输入的瓶颈问题。

    

    arXiv:2412.20662v4 公告类型：replace-cross 摘要：预训练基础模型最近在表格理解和推理等表格相关任务上取得了显著进展。然而，使用视觉大语言模型（VLLMs）识别非结构化表格的结构和内容仍然缺乏充分研究。为了弥补这一空白，我们提出了一个基于分层设计理念的基准，用于评估VLLMs在免训练场景下的识别能力。通过深入评估，我们发现低质量的图像输入是识别过程中的一个重要瓶颈。受此启发，我们提出了邻居引导的工具链推理器（NGTR）框架，其特点在于整合多种轻量级工具进行视觉操作，旨在缓解低质量图像带来的问题。具体而言，我们将相似邻居的工具选择经验迁移到输入中，并设计了一个反思模块来监督工具的……（摘要截断）

    arXiv:2412.20662v4 Announce Type: replace-cross  Abstract: Pre-trained foundation models have recently made significant progress in table-related tasks such as table understanding and reasoning. However, recognizing the structure and content of unstructured tables using Vision Large Language Models (VLLMs) remains under-explored. To bridge this gap, we propose a benchmark based on a hierarchical design philosophy to evaluate the recognition capabilities of VLLMs in training-free scenarios. Through in-depth evaluations, we find that low-quality image input is a significant bottleneck in the recognition process. Drawing inspiration from this, we propose the Neighbor-Guided Toolchain Reasoner (NGTR) framework, which is characterized by integrating diverse lightweight tools for visual operations aimed at mitigating issues with low-quality images. Specifically, we transfer a tool selection experience from a similar neighbor to the input and design a reflection module to supervise the tool i
    
[^320]: 面向GPU数据中心的功耗与碎片感知在线调度

    Power- and Fragmentation-aware Online Scheduling for GPU Datacenters

    [https://arxiv.org/abs/2412.17484](https://arxiv.org/abs/2412.17484)

    该论文提出了PWR，一种面向GPU数据中心的功耗与碎片感知在线调度策略，在不预知未来任务到达的情况下降低功耗并缓解GPU碎片化问题。

    

    人工智能和大语言模型的兴起正在推动数据中心增加GPU的使用，以应对复杂的训练和推理任务，这影响了运营成本、能源需求以及大规模计算基础设施的环境足迹。本工作研究GPU数据中心的在线调度问题，即在不知道任务未来到达情况下调度任务。我们关注两个目标：最小化GPU碎片化和降低功耗。GPU碎片化是指部分GPU分配阻碍了剩余资源的有效利用，尤其是当数据中心接近满负荷时。最近的一种调度策略——碎片化梯度下降（FGD），利用碎片化度量来解决这一问题。由于GPU的巨大功耗需求，降低功耗也至关重要。为此，我们提出了PWR，一种旨在最小化功耗的新型调度策略。

    arXiv:2412.17484v2 Announce Type: replace-cross  Abstract: The rise of Artificial Intelligence and Large Language Models is driving increased GPU usage in data centers for complex training and inference tasks, impacting operational costs, energy demands, and the environmental footprint of large-scale computing infrastructures. This work addresses the online scheduling problem in GPU datacenters, which involves scheduling tasks without knowledge of their future arrivals. We focus on two objectives: minimizing GPU fragmentation and reducing power consumption. GPU fragmentation occurs when partial GPU allocations hinder the efficient use of remaining resources, especially as the datacenter nears full capacity. A recent scheduling policy, Fragmentation Gradient Descent (FGD), leverages a fragmentation metric to address this issue. Reducing power consumption is also crucial due to the significant power demands of GPUs. To this end, we propose PWR, a novel scheduling policy to minimize power
    
[^321]: 量化近似正交循环神经网络

    Quantized Approximately Orthogonal Recurrent Neural Networks

    [https://arxiv.org/abs/2402.04012](https://arxiv.org/abs/2402.04012)

    本文提出了量化近似正交循环神经网络（QORNNs）来解决正交循环神经网络（ORNNs）中参数过多的问题，采用一种后训练量化策略和三种融入正交约束和量化权重的量化感知训练算法，取得了与s相似的结果。

    

    正交循环神经网络（ORNN）是一种吸引人的选择，用于学习涉及具有长期依赖性的时间序列的任务，由于其简单性和计算稳定性。然而，这些网络通常需要大量参数才能表现良好，这在功率受限的环境（如紧凑设备）中可能是阻碍因素。解决这个问题的一种方法是神经网络量化。构建这样的网络仍然是一个待解决的问题，其固有的不稳定性是被认可的。在本文中，我们探讨了ORNN中的循环和输入权重矩阵的量化，导致了量化近似正交RNN（QORNN）。我们研究了一种后训练量化（PTQ）策略和三种融入正交约束和量化权重的量化感知训练（QAT）算法。实证结果证明了使用QAT的优势。最高效的模型实现了与s类似的结果。

    Orthogonal recurrent neural networks (ORNNs) are an appealing option for learning tasks involving time series with long-term dependencies, thanks to their simplicity and computational stability. However, these networks often require a substantial number of parameters to perform well, which can be prohibitive in power-constrained environments, such as compact devices. One approach to address this issue is neural network quantization. The construction of such networks remains an open problem, acknowledged for its inherent instability.In this paper, we explore the quantization of the recurrent and input weight matrices in ORNNs, leading to Quantized approximately Orthogonal RNNs (QORNNs). We investigate one post-training quantization (PTQ) strategy and three quantization-aware training (QAT) algorithms that incorporate orthogonal constraints and quantized weights. Empirical results demonstrate the advantages of employing QAT over PTQ. The most efficient model achieves results similar to s
    

