# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PlayCraft: Designing Engaging Games from Agent Play Experience](https://arxiv.org/abs/2611.26624) | PlayCraft提出了一个自动化框架，让视觉智能体反复试玩游戏并基于其游玩报告（进展、决策、反思等）自动修改游戏设计，使游戏更有助于玩家精通，从而提升游戏的吸引力。 |
| [^2] | [One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning](https://arxiv.org/abs/2611.26617) | 该论文提出O²LaF，通过将同一个预训练视频扩散Transformer分别适配为逆向动力学过渡编码器和前向动力学动作条件生成器，从无标注视频中自监督学习可迁移的潜在动作，并在导航与机器人操作的少样本动作探测中达到新的最先进水平。 |
| [^3] | [SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research](https://arxiv.org/abs/2611.26601) | 本文提出一个借鉴自动驾驶分级的自主性评级框架，系统化梳理了LLM在漏洞研究中的能力证据，发现2026年前智能体已能在无人工干预下发现、验证并修补真实软件漏洞，开源权重模型随后也达到了相当的自主性水平。 |
| [^4] | [Active Feature Acquisition by Distilling an Oracle Teacher](https://arxiv.org/abs/2611.26598) | 该论文提出OracleImitator，首次显式构建了一个按候选特征对下游预测器误差影响进行排序的神谕教师，并通过软目标模仿将这一排序蒸馏为可部署的主动特征获取策略，在多模态医疗诊断和AFABench基准上于整个预算范围内匹配或超越现有方法，恢复约90%的可达准确率增益。 |
| [^5] | [PGX: Physically Grounded Execution for Long-Horizon Humanoid Loco-Manipulation](https://arxiv.org/abs/2611.26596) | PGX框架通过将时间结构与精度控制交由两个无需训练的经典组件处理，并把长时程人形机器人移动操作分解为谓词状态机、VLA技能执行等多层结构，使VLA模型专注于阶段特定的技能动态，从而有效解决长时程移动操作中时间、技能与精度紧密耦合的难题。 |
| [^6] | [Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning](https://arxiv.org/abs/2611.26592) | 该论文发现语言模型推理流之间存在正相关，破坏了自一致性认证所依赖的条件独立假设，使被认证的假设分裂为关于单次抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”（仅在零相关时二者等价），从而从根本上改变了 anytime-valid 证书所真正保证的可靠性含义。 |
| [^7] | [Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models](https://arxiv.org/abs/2611.26590) | 该论文提出多模态概念透镜（MC-Lens），通过将视觉编码器与语言解码器的中间状态映射到共享语义空间，并结合内部概念干预，揭示视觉-语言模型中概念的可及性与选择性访问机制，为全局工作空间理论在AI系统中的验证提供了工具。 |
| [^8] | [Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards](https://arxiv.org/abs/2611.26588) | 提出首个基于规范、利用LLM交叉检查已通过回归测试追踪记录的框架"Verify the Verifier"，从564个注入缺陷中确认了41个逃过UVM记分板检查的缺陷。 |
| [^9] | [ProtoDiffusion: Token-per-Layer Generative Modeling of Multi-Layer Images](https://arxiv.org/abs/2611.26586) | 该论文提出ProtoDiffusion——基于ProtoVAE多层变分自编码器的潜空间扩散框架，通过“一个共享空间原型+每层一个候选向量”的紧凑表示（N层仅需HW+N个令牌而非NHW个），在图像或文本条件（可选边界框布局引导）下高效生成多层RGBA图像。 |
| [^10] | [GEOHunter: Mapping a Generative Engine Optimization Network in the Wild](https://arxiv.org/abs/2611.26564) | 该论文提出了GEOHunter，首次在真实环境中测绘了一个横跨三大中国UGC平台的生成式引擎优化（GEO）付费推广网络，识别出1,985个账户和46万余条推广投放，并发现美国与中国的九个生成式引擎中4.5%–20%的可检索引用页面受到该推广内容的污染。 |
| [^11] | [Multi-institutional grading of Barrett's esophagus dysplasia without stain normalization using a pathology foundation model and multiple instance learning](https://arxiv.org/abs/2611.26562) | 该研究构建了基于病理学基础模型与属性感知多示例学习的深度学习系统，在无需染色归一化的条件下，利用四个机构969张全切片实现了巴雷特食管异型增生的高精度自动分级（AUC达0.96，与专家共识高度一致）。 |
| [^12] | [MM-SystemOneBias: Evaluating Bias in Multimodal Jev-Style System One Decision Models](https://arxiv.org/abs/2611.26557) | 提出MM-SystemOneBias基准，通过六种配对干预在11个任务场景、5,323个实例上系统评估13个多模态决策模型的偏差，发现决策模型在选项标签、无关信息和评分量表干预下比其骨干模型更稳定。 |
| [^13] | [Replication of Agentic Security Events: Audit and Investigation (ROGUE-AI)](https://arxiv.org/abs/2611.26553) | 该论文提出ROGUE-AI多智能体框架，通过离线复现被报道的LLM智能体安全事件并构建仅缺少所归咎原因的匹配对照组进行对照实验，在108次复现尝试中89%成功重现所报道的危害，从而验证事件真实性并定位根本原因。 |
| [^14] | [The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models](https://arxiv.org/abs/2611.26545) | 该研究通过向人格方向库中加入“否认已知知识”的角色构建无标签谎言检测方向，发现其在70B规模模型的大多数谎言类型上优于单数据集探针，但在谎言与诚实否认措辞相同的知识类谎言上失效，揭示出人格方向主要依赖表面措辞而非深层欺骗意图。 |
| [^15] | [AURA: AUdio-dRiven streaming Avatar](https://arxiv.org/abs/2611.26540) | 提出了 AURA——一种音频驱动的流式数字人视频生成模型，通过滚动 KV 缓存实现逐块因果生成，并配合两阶段蒸馏，可在单 GPU 上以 20.7 fps 实时流式生成 576p 说话人视频，适用于低延迟交互式虚拟形象应用。 |
| [^16] | [LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits](https://arxiv.org/abs/2611.26538) | 该论文揭示了现有交叉性偏差审计判据会把独立属性的加性效应误报为交互偏差（平均每个数据集误报66.1个子群体、真实交互检出率仅32%），并提出ADDIS方法，通过在所有属性上拟合加性模型建立基线，从而真正将交互效应与加性效应分离，准确审计交叉性子群体偏差。 |
| [^17] | [Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models](https://arxiv.org/abs/2611.26531) | 该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。 |
| [^18] | [Backtracking Agent Applications Security Incidents](https://arxiv.org/abs/2611.26525) | 该论文提出了Origin回溯工具，通过弥合智能体轨迹日志与应用代码执行层之间的语义鸿沟，让分析人员仅需提交少量查询即可自动识别智能体应用安全事件的入侵入口点。 |
| [^19] | [GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers](https://arxiv.org/abs/2611.26509) | GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。 |
| [^20] | [FLUID: Flexible, Evidence-Grounded Situated Task Guidance](https://arxiv.org/abs/2611.26491) | FLUID提出一个统一框架，将任务指导干预与证据、必要性和时机灵活性相绑定，用于评估多模态实时任务指导中的灵活交互并构建智能体，该智能体在正确性、精确性、及时性和延迟之间取得了最佳整体平衡。 |
| [^21] | [A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes](https://arxiv.org/abs/2611.26475) | 本文提出临界阻尼Polyak（CDP）自适应步长——将经典Polyak规则乘以4并加温和阻尼——使梯度下降偏向更平坦的极小值，并首次为局部二次目标函数建立了锐度与损失下降的时间一致理论保证。 |
| [^22] | [Monte Carlo Pass Search for Setpieces](https://arxiv.org/abs/2611.26464) | 该论文提出蒙特卡洛传球搜索（MCPS），借助世界模型对定位球的反事实传球方案和进攻站位进行大规模推演与价值评估，发现意甲比赛中最佳可用传球比实际踢出的传球多出0.0085 xG（提升28%）。 |
| [^23] | [A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights](https://arxiv.org/abs/2611.26450) | 提出了一种零梯度、无需训练样本的闭式电路嫁接方法，可将精确的确定性计算（如多位数算术）直接编译进预训练语言模型权重，同时保持其自然语言能力。 |
| [^24] | [Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation](https://arxiv.org/abs/2611.26444) | Mind2Act World是一个包含180种配置的诊断性基准，通过Mind、Act和Mind2Act三个视角分别评估具身智能体的认知推理、物理执行及二者在闭环中的组合能力，从而精确定位失败源自推理、执行还是其组合。 |
| [^25] | [Asuka-FM: A System-Architectural Co-Design Framework for Efficient Text-to-Image Pretraining on Consumer Hardware](https://arxiv.org/abs/2611.26437) | 该论文提出系统-架构协同设计框架Asuka-FM，证明约2.06亿参数的双流扩散Transformer可在单张消费级GPU上以约160 GPU小时、不足55美元的成本从头完成有竞争力的文本到图像模型预训练，并解决了注意力开销、JIT图中断和VAE吞吐量三大瓶颈。 |
| [^26] | [Seer: Guiding Small Deep-Research Agents with Shared Decision Models](https://arxiv.org/abs/2611.26423) | Seer是一个推理时引导框架，通过从共享的真实研究轨迹中学习工具响应模拟器和价值估计器，为资源受限设备上的小型深度研究智能体提供决策引导，避免了设备端训练和重复适配的成本。 |
| [^27] | [Reasoning-Trace Analysis Using Large Language Models for Interpretable Q-Matrix Construction](https://arxiv.org/abs/2611.26422) | 该论文提出推理轨迹分析（RTA）框架，通过分析大语言模型思维链推理中带有属性标注的推理步骤，使Q矩阵构建中每个题目-属性分配的依据变得可解释、可审查，从而弥补以往仅评估最终矩阵而忽视分配基础的不透明性问题。 |
| [^28] | [Awaken-HI: Scaling Humanoid Intelligence with Whole-Body Human Experience](https://arxiv.org/abs/2611.26420) | 该论文构建了包含5万小时第一人称视频与同步全身人体运动、覆盖100余种技能的大规模数据集Awaken-Human，并提出视觉-语言-动作模型Awaken-HI，通过共享动作专家以人类轨迹直接监督机器人动作学习，实现了移动-操作技能从人类到人形机器人的高效迁移。 |
| [^29] | [ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain](https://arxiv.org/abs/2611.26414) | ZeroVLN提出一种快慢结合的导航框架，无需对VLM大脑进行任何导航领域监督微调，而是通过语义工具调用接口让大脑以自然语言传达移动决策，并借助闭环强化学习联合优化大脑决策与动作专家执行，在保持VLM通用视觉能力的同时实现有效导航。 |
| [^30] | [Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling](https://arxiv.org/abs/2611.26407) | 本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。 |
| [^31] | [Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem](https://arxiv.org/abs/2611.26398) | 本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。 |
| [^32] | [World Models Favor Learned Priors over Visual Evidence](https://arxiv.org/abs/2611.26366) | 该论文揭示了世界模型在跨环境预测中倾向于依赖训练数据中的常见规则先验而非视频中的视觉证据，并提出TailWorld基准来独立控制训练不平衡与校准信息量，以评估示范物理规则在不同初始状态间的迁移能力。 |
| [^33] | [How Much Should an Agent Remember? Deployment Lessons from Self-Distilled Memory for LLM Agents](https://arxiv.org/abs/2611.26361) | 该论文提出一个让LLM智能体从自身轨迹中自蒸馏“经验教训”并整合为“指导原则”的记忆流水线，已部署于IBM生产环境，核心发现是最佳记忆量取决于模型自身能力而非记忆机制本身。 |
| [^34] | [Beam Model Technical Report](https://arxiv.org/abs/2611.26358) | Beam 是一个总参数 5013 亿、活跃参数 234 亿的稀疏 MoE 模型，专为编程、推理和智能体任务优化，通过 23.8 万亿 token 预训练、两阶段中期训练实现百万 token 上下文窗口，并在 10500 块 GB300 GPU 上开展开放实验室中最大规模的强化学习后训练，以比同类模型少 3-4 倍的推理计算量达到前沿推理水平。 |
| [^35] | [Context as a Resource](https://arxiv.org/abs/2611.26346) | 提出无路由架构ClonAct，通过非生成式上下文内存管理单元、原地上下文单元、足迹感知突触总线和波刷新共享主干直接虚拟化上下文块，解决多代理并发会话中的跨主题约束渗漏与盲目合并失败问题。 |
| [^36] | [DeepQuest: From Knowledge Gaps to Scientific Discovery in Alzheimer's Disease](https://arxiv.org/abs/2611.26343) | DeepQuest是一个知识缺口驱动的AI科学家框架，它将阿尔茨海默病领域综述中的未决问题转化为可检验假设并自动开展迭代实验与两级审计，在OASIS-3上产生了42个可行假设，并以95.2%的审计评估科学发现率在绝大多数评估维度上超越最强基线。 |
| [^37] | [Training LLMs to Follow Reasoning Instructions with Minimal Edits](https://arxiv.org/abs/2611.26342) | 提出基于编辑的训练框架，通过定位模型推理轨迹中首个失败点并进行最小文本修正，仅对编辑片段施加监督，从而在保留已有推理模式的同时更精准地训练大语言模型遵循推理指令。 |
| [^38] | [Sirius-Tiamat: How Small Can a One-Pass Decision Encoder Get Before Its Decisions Change?](https://arxiv.org/abs/2611.26339) | 本文系统量化了4.21亿参数单次决策编码器Laya在量化与剪枝等压缩下的鲁棒性，发现压缩导致的决策翻转在域外数据上远大于域内，删层会严重破坏校准（ECE升至39.2%），而一次无需标签的温度缩放即可将其恢复。 |
| [^39] | [One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination](https://arxiv.org/abs/2611.26336) | 本文提出孟加拉语幻觉检测分层基准 Alikbachan，通过文化距离、任务类型和32种幻觉模式三个维度交叉分层，并在社区竞赛中大规模验证，发现语篇级信息遗漏是零样本LLM评判器与参赛系统的共同弱项，证明了分层设计对模型幻觉能力的显著诊断价值。 |
| [^40] | [ClosureBench: A Benchmark for Reasoning under Closure Assumptions](https://arxiv.org/abs/2611.26335) | ClosureBench是一个通过在OWA、CWA和LCWA三种封闭假设下仅隔离缺失信息处理方式的变化来测试大语言模型是否遵循缺失信息处理规则的基准，并揭示了总体准确率会掩盖大量封闭推理错误的问题。 |
| [^41] | [Online Temporal Rotations for 2 Bit KV-Cache Quantization in Video Diffusion Transformers](https://arxiv.org/abs/2611.26334) | 提出时序旋转量化（TRQ）方法，通过在线拟合正交旋转将KV缓存中时间相邻帧的各通道均值映射为近似常数，并以仅需两比特存储的平面旋转树实现轻量参数化，从而在视频扩散Transformer中实现2比特KV缓存量化。 |
| [^42] | [Rigel-Ladon: An Option-Count Law for a One-Pass Decision Encoder, Tested on Held-Out Option Counts and Held-Out Schemas](https://arxiv.org/abs/2611.26332) | 该论文发现并验证了一条“选项数定律”：单遍决策编码器的 logit 准确率随 log(K-1) 线性下降（斜率 0.68–0.88），仅凭小选项数（K≤6）的测量即可精确预测留出的大选项数、甚至跨题型（如 HR 部门路由任务）的准确率。 |
| [^43] | [A Control-Theoretic Approximation to Predictive Coding Dynamics](https://arxiv.org/abs/2611.26319) | 该论文证明了深度反馈控制（DFC）是预测编码（PC）动力学在不动点附近的局部近似（称为 Approx PC），在浅层线性网络中二者完全等价，且通过重新线性化反馈控制器和分步逼近输出目标可以缓解由深度和非线性带来的偏差，从而厘清了两种生物学可信局部学习框架之间的关系。 |
| [^44] | [C-LiTE: A Lightweight Foundation Model for Cellular feature learning in Histopathology](https://arxiv.org/abs/2611.26316) | 该论文提出C-LiTE——一个仅有2200万参数的轻量级基础模型，通过原生分辨率分词策略实现H&E染色组织病理学中高效的细胞特征学习，弥补了现有模型要么隐式捕获细胞特征、要么计算开销过大且受限于特定细胞群体的不足。 |
| [^45] | [Actor-Predictor Architecture for Reinforcement Learning with Formal Specification Objectives](https://arxiv.org/abs/2611.26313) | 该论文提出Actor-Predictor框架，通过预测器估计轨迹未来能否满足形式化规范来学习逻辑进展的替代表示，无需显式构造自动机，并从理论上证明其可等价于自动机状态，从而突破了传统方法受限于小型规范和LTL的局限。 |
| [^46] | [Client Exits - A Way to Increase LLM Transparency on the Client Side](https://arxiv.org/abs/2611.26309) | 本文提出“客户端出口”机制，让LLM在输出中插入机器可解释的标记、由客户端自行决定如何以图形、文本、语音等多模态呈现补充信息，从而在不规定具体界面的情况下提升LLM响应的透明度与可审查性。 |
| [^47] | [A Tensor-Algebraic No-Go Theorem for High-Dimensional Photonic Greenberger--Horne--Zeilinger States Discovered with AlphaProof Nexus](https://arxiv.org/abs/2611.26300) | 本文利用张量代数方法证明了一个不可行定理：当维数D≥N时，任何偶数N≥4的N光子D维GHZ态都无法通过概率性对光源对应的单色量子图实现，且该证明由AlphaProof Nexus通过AI辅助形式化证明搜索在Lean中完成。 |
| [^48] | [Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era](https://arxiv.org/abs/2611.26286) | 通过测量会议和期刊评审中的声明级重叠度，研究发现LLM时代同行评审的批评意见日益同质化——ICLR评审重叠率从2026年的12.0%升至2027年的21.3%，且被检测为LLM生成的评审比人工评审重叠度更高，表明单篇论文评审的批评多样性正在下降。 |
| [^49] | [What If Chain-of-Thought Is Not Enough?](https://arxiv.org/abs/2611.26282) | 本文提出以“问题—选项—标准”（QOC）结构化决策过程替代思维链对大语言模型进行微调，并通过三代渐进式训练流水线证明该方法能显著提升答案质量，且模型规模越大收益越明显。 |
| [^50] | [Digitization of Analogue Gauges using Computer Vision](https://arxiv.org/abs/2611.26262) | 本研究开发了基于计算机视觉的自动化仪表读数系统，通过对比三种混合方法证明OpenCV结合ResNet18的方案准确率最高（97.3%），验证了残差学习架构在工业模拟仪表数字化中的有效性。 |
| [^51] | [Similarity-based OSF Logic for Flexible SPARQL Query Answering](https://arxiv.org/abs/2611.26261) | fosf推理器通过在OSF语言中引入类间相似度关系以支持基于相似度的推理，并将OSF查询经模糊本体感知重写后编译为标准SPARQL，实现了对RDF知识图谱的灵活SPARQL查询回答。 |
| [^52] | [SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills](https://arxiv.org/abs/2611.26256) | SkillBridge 提出了一种基于契约的三阶段工作流（静态分析、动态验证与修复），以低成本方式诊断并修复 LLM 智能体技能在跨操作系统场景下的兼容性缺陷。 |
| [^53] | [STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery](https://arxiv.org/abs/2611.26255) | STAGE框架从局部因果种子出发构建跨推文弱监督标签，结合时空上下文图与图注意力网络，实现灾害社交媒体中跨推文事件因果关系的可扩展自动发现，相比局部监督显著提升因果发现性能。 |
| [^54] | [The Stability Gap in Continual Reinforcement Learning](https://arxiv.org/abs/2611.26252) | 本文提出了一个在保持环境动态和奖励不变的前提下仅改变观测表示的简单持续强化学习基准，并首次揭示了其中的“稳定性差距”现象——即使智能体可以完全访问过去任务，新任务的引入仍会导致旧任务性能大幅下降后再部分或完全恢复。 |
| [^55] | [TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs](https://arxiv.org/abs/2611.26248) | 该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。 |
| [^56] | [FastRoverVLA: Fast Vision-Language-Action Execution for Off-Road Autonomous Driving](https://arxiv.org/abs/2611.26242) | FastRoverVLA提出了一种无需训练的执行接口，通过将视觉规划生成与运动条件执行解耦，利用速度-曲率动作缓冲区持续进行运动校正，并根据规划粗糙度评分自适应分配重用预算，从而大幅减少越野自动驾驶中昂贵的多模态推理调用。 |
| [^57] | [Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction](https://arxiv.org/abs/2611.26235) | 该研究提出用归一化焓和佩克莱数两个物理信息无量纲特征替代原始工艺参数，并结合预训练表格基础模型进行基准测试，以提升LPBF熔池几何预测的跨材料泛化能力与数据效率。 |
| [^58] | [Sightline: Calibrated Data-Flow Summaries for LLM-Integrated Applications](https://arxiv.org/abs/2611.26221) | 该论文提出Sightline，通过在Lean 4中形式化证明“占位符只有被注意力读取才会一阶流向transformer输出”这一原理，为LLM集成应用生成经校准的数据流摘要，避免了传统污染分析产生虚假流以及依赖第二模型标注的高昂成本。 |
| [^59] | [RegionAgent: Two-Stage Preference Optimization for Regional Image Restoration](https://arxiv.org/abs/2611.26179) | 提出RegionAgent自适应修复框架，通过两阶段偏好学习训练视觉语言模型，在每次迭代中针对非均匀耦合退化的不同区域自适应生成特定的修复动作，克服了现有全局统一操作忽视区域差异、修复不完整的问题。 |
| [^60] | [Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy](https://arxiv.org/abs/2611.26172) | 固定预算RLVR训练会将生成长度上限内化为模型的习得约束，用测试时扩展能力换取了预算内的准确率提升。 |
| [^61] | [The FrontierOR Challenge: Agentic Algorithm Design for Operations Research](https://arxiv.org/abs/2611.26171) | FrontierOR挑战赛邀请各团队构建基于LLM的智能体工作流，为多类优化问题自动设计、测试和改进可媲美商业求解器Gurobi的算法，以推动智能体算法设计能力的发展。 |
| [^62] | [Sage-Evolve-Preview:Adaptive Observation for Post-Training Self-Improvement](https://arxiv.org/abs/2611.26132) | 针对递归自我改进循环中仅依赖锚点基准聚合分数导致进展停滞的问题，本文提出自适应观察（AdaObs）机制，在保持锚点目标不变的前提下，于平台期动态细化诊断粒度、提升难度并扩展相关任务，从而突破由“观察枯竭”造成的改进瓶颈。 |
| [^63] | [ChronoPatch: Black-Box Energy-Latency Attacks on Vision-Language Models](https://arxiv.org/abs/2611.26117) | ChronoPatch提出了一种仅需观察最终响应的黑盒攻击方法，通过响应长度引导的进化搜索和子空间校准的决策式优化，使视觉语言模型的推理延迟和GPU能耗显著增加。 |
| [^64] | [SenseAlign: Learning Decision-Aligned Predictive Representations for Active Urban Sensing with UAVs](https://arxiv.org/abs/2611.26109) | 提出基于强化学习的SenseAlign框架，通过特权教师-学生学习与决策对齐的预测表征，实现大规模城市路网中无人机编队面向动态交通状态的主动感知协调。 |
| [^65] | [Pareto-Optimal and Strategyproof Data-Sharing Between Competitors](https://arxiv.org/abs/2611.26108) | 本文提出了一族数据共享合同，使竞争企业的相对市场份额保持不变，并证明其中一种合同同时满足帕累托最优性和防止企业少报数据量的防策略操纵性。 |
| [^66] | [HiUni: Harmonizing Understanding and Generation in Shared Visual Space](https://arxiv.org/abs/2611.26107) | HiUni 通过让理解与生成任务共享同一个语言对齐的 ViT 和密集 Transformer 骨干网络、仅将任务特定计算放在输出端，并借助像素扩散机制解决优化困难与细节丢失问题，实现了高度参数共享的统一多模态模型。 |
| [^67] | [Controlling the Humanoid Robot H1-2 With a Simulated Biologically Plausible Network](https://arxiv.org/abs/2611.26074) | 提出了一种生物合理的脉冲神经网络，通过固定储备池、奖励调制STDP与资格传播等不同生物可信度的学习方案，成功实现了宇树H1-2人形机器人的运动控制。 |
| [^68] | [AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators](https://arxiv.org/abs/2611.26073) | 提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。 |
| [^69] | [GenSPARQL: A Generative Extension of SPARQL for Incomplete Knowledge Graphs](https://arxiv.org/abs/2611.26062) | 提出GenSPARQL，通过在SPARQL 1.1中新增GENOP算子，在查询求值时将LLM生成的值绑定到查询变量，并借助相似度连接与查询规划器按需调用LLM，从而高效补全不完整知识图谱中的缺失事实。 |
| [^70] | [Studying AI Act Mentions: A first look at the impact of the European Union's AI regulation on Hugging Face](https://arxiv.org/abs/2611.26050) | 这是首个针对欧盟《人工智能法案》在Hugging Face平台影响的实证研究，通过分析超过500万条记录中1,500余条明确提及该法案的内容，揭示了AI从业者如何在实践中回应欧盟AI监管。 |
| [^71] | [SEOYUN: Refusal-Inducing Exam Documents for AI-Glasses Pipelines](https://arxiv.org/abs/2611.26033) | 该论文提出SEOYUN框架，通过设计能诱导视觉语言模型主动弃答的考试文档来防御AI眼镜作弊，其核心贡献是针对考试场景的威胁模型、分析性限制与评估协议。 |
| [^72] | [Serialization Tax in Shared-Latent Exchangeable Decisions](https://arxiv.org/abs/2611.26029) | 本文提出“串行化税”概念：当条目共享不确定的潜在状态时，独立边缘化虽保持均值却破坏了聚合后验中的相关性，作者用凸序比较量化方差与全变差差距、推导阈值分离并连接决策遗憾，进而提出 TaxScore 识别决策边界附近存在未消解不确定性的大型隐藏块以引导更强的推断。 |
| [^73] | [ForeSeek: Foresight-Driven Reasoning for Long-Form Video Understanding](https://arxiv.org/abs/2611.26024) | ForeSeek 提出一种前瞻驱动的视频智能体，通过对未观察的视频区域形成并验证视觉预期、将失败预期作为回溯信号，把长视频推理从被动检索转变为预测性探索，并配合两阶段训练框架赋予模型前瞻推理与高效决策能力。 |
| [^74] | [Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model](https://arxiv.org/abs/2611.26018) | 本文提出一种因果测试协议，通过从其他输入移植消息来判定语言模型中的全局工作空间究竟是由谁执行计算的，可区分出忽略广播、纯转发广播、以及将广播与自身知识重组这三种不同情况，并在训练后的工作空间模型中同时发现了这三种结果。 |
| [^75] | [When Agents Play Differently: Localizing Human--Agent Behavioral Divergence with Conditional Distributional Testing](https://arxiv.org/abs/2611.26012) | 提出了一种基于分层条件分布检验的框架，利用情境单元内的非参数差异统计量和子群规则精确定位智能体与人类行为产生分歧的具体场景，并在商业三消游戏中发现了聚合指标无法察觉的模式级行为差异。 |
| [^76] | [System-One RSI: Building Muscle Memory for Interactive Agents through Guided Practice](https://arxiv.org/abs/2611.26007) | 本文提出System-One RSI双循环框架，通过高频OODA内循环（决策模型加决策树，无需推理模型即可输出亚秒级定时动作）与推理智能体团队执行的离线递归改进外循环（修订提示词与决策树、从交互记录中生成监督信号），让交互式智能体像运动员一样通过引导式练习建立“肌肉记忆”，从而同时获得强任务理解能力与亚秒级实时控制。 |
| [^77] | [AI agents can redesign engineering parts with more than a thousand CAD parameters and nonlinear simulations](https://arxiv.org/abs/2611.26003) | 该论文提出，通用大语言模型智能体（GPT-6 Astra）能够在仅需少数非线性有限元仿真求解的情况下，对包含上千个CAD参数的真实工程零件进行整体性的重新设计，突破了传统优化器最多只能满足三个关键条件中两个的局限。 |
| [^78] | [From AI Surrogate to Engineering Tool: Toward Production Use of Fiber-Orientation Prediction in Crash CAE](https://arxiv.org/abs/2611.25991) | 本田将此前提出的纤维取向预测AI代理模型（将每个零件约25小时的预测工作缩短至约3分钟）经过从二维平板到生产零件的分阶段验证和专家评审，成功转化为碰撞CAE设计工作流程中实际投入使用的工程工具。 |
| [^79] | [Policy Improvement-Guided Replay for Continual Reinforcement Learning](https://arxiv.org/abs/2611.25989) | 该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。 |
| [^80] | [Law at the Speed of AI: Critical Legal Infrastructure for the Machine Age: Autonomy Under Law Architecture (AULA)](https://arxiv.org/abs/2611.25985) | 该论文指出自主AI智能体的全球部署已超出治理基础设施的承载能力，协议层的成熟无法带来治理层的法律可辨识性，由此造成责任归属与权利分配的双向缺口，并提出AULA架构作为机器时代所需的关键法律基础设施。 |
| [^81] | [Safe Subgame Resolving in Multiplayer Games](https://arxiv.org/abs/2611.25981) | 该论文证明了首个针对非相关多人博弈子博弈求解的条件性安全定理，表明已求解策略的最佳响应差距相对蓝图最多增加四项（其中两项源于求解器的未实现承诺子博弈价值与求解误差），从而为多人博弈中的实时策略细化提供了事后安全证明证书。 |
| [^82] | [Multi-View Transformer with Geometry-Guided Multi-Modal ControlNet for 3D Virtual Try-On](https://arxiv.org/abs/2611.25979) | 该论文提出GS-VTON-MVT三阶段框架，通过具备完整跨视图自注意力的多视图修补Transformer学习视图间对应关系，并利用几何引导的多模态ControlNet对扩散编辑器进行条件控制，从而实现视角一致、细节保真的3D高斯泼溅虚拟试穿。 |
| [^83] | [Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation](https://arxiv.org/abs/2611.25976) | 七个开源大语言模型在估计食物能量时无法随给定重量成比例缩放（弹性仅0.548-0.979），导致小份量被高估、大份量被低估，而采用“模型自报能量密度×重量”的两步法可基本消除该误差。 |
| [^84] | [Web search in Latin America: language models use less local and diverse news than search engines](https://arxiv.org/abs/2611.25955) | 本研究通过对拉丁美洲九国时事的大规模查询实验发现，基于语言模型的搜索返回的新闻来源多样性低于传统搜索引擎，且本地新闻占比因用户类型而表现不一致，揭示了AI搜索可能加剧地理偏见的风险。 |
| [^85] | [GaussTrans: A Gaussian Interface Framework for Multi-Layer Refractive Transport](https://arxiv.org/abs/2611.25950) | 提出GaussTrans框架，通过高斯-场混合表面、传输耦合细化和基元引导隐式求交三个组件，实现高阶多层透明场景中精确的光传输与重建。 |
| [^86] | [Fast-JEPA: Advancing the Success-Latency Frontier for Latent World Models](https://arxiv.org/abs/2611.25927) | Fast-JEPA通过构建目标导向计划引导搜索并直接预测完整候选动作序列的结果来避免递归状态展开，以比INTACT快2.41倍的规划速度和96.50%的平均成功率推动了潜在世界模型的成功率-延迟帕累托前沿。 |
| [^87] | [ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode Approach Mining](https://arxiv.org/abs/2611.25926) | 提出ROAM-GRPO算法，通过将训练中已被良好解决的问题路由到“离模”轨道以激励模型探索多样化推理策略，并经LLM裁判验证后自蒸馏进策略，有效缓解了RLVR后训练导致的输出多样性坍缩问题。 |
| [^88] | [Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models](https://arxiv.org/abs/2611.25916) | 论文发现语言模型在多解约束谜题中高度集中于同一个默认解，其行为如同按特定顺序遍历的搜索求解器——优先访问初始候选最少的实体这一搜索顺序最能预测模型输出，且改变给定的顺序会相应改变模型的默认解。 |
| [^89] | [Validation-driven Deployment of LLM Inference Platforms for Agentic Workloads](https://arxiv.org/abs/2611.25905) | 本文提出了一个面向智能体LLM服务的端到端评估流水线，证明重放捕获的追踪记录能以高保真度（95分位下1.18倍，而实时重跑达4.98倍）重现原始智能体会话负载，且实时、重放和合成工作负载在代码覆盖上相互补充。 |
| [^90] | [From Cases to Criteria: Reusing Evaluation Knowledge for Task-Specific Rubric Generation](https://arxiv.org/abs/2611.25903) | 提出 RubricDuo，通过构建链接评估维度、共享概念、局部概念、准则与查询的评分细则知识图谱来复用历史评估知识，从而为 LLM 评估自动生成任务特定、细粒度且结构完整的查询级评分细则。 |
| [^91] | [Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding](https://arxiv.org/abs/2611.25898) | 这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。 |
| [^92] | [The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back](https://arxiv.org/abs/2611.25896) | 该论文在ARC-AGI-2上量化了要求LLM生成可执行可验证产物（程序）的代价（得分从77%降至57%），并指出真正的瓶颈在于让解释“泛化”而非“找到”解释——因为多数仅由只答组解出的任务中，程序组其实已拥有能复现全部演示、却仍在测试上失败的程序。 |
| [^93] | [One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models](https://arxiv.org/abs/2611.25895) | 本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。 |
| [^94] | [Explainable AI for Solar PV Power Forecasting: From Model Selection to Trustworthy, Reproducible Deployment](https://arxiv.org/abs/2611.25894) | 本文提出了一种将自动化模型搜索与可解释人工智能相结合的可复现光伏功率预测工作流程，并在葡萄牙波尔图一个16.32 kWp电站上验证，最佳模型达到11.6%的nRMSE，误差约为智能持续性基准的一半。 |
| [^95] | [Meta-RTL: Evolving Agent Harnesses for RTL Optimization](https://arxiv.org/abs/2611.25891) | Meta-RTL提出了一种基于EDA反馈的自动进化框架，能从极简循环出发自动生成RTL优化智能体的执行框架，摆脱人工设计搜索过程的束缚，在时序违例和面积优化上显著优于现有方法，并可直接迁移到未见设计。 |
| [^96] | [Trustworthy Geospatial AI Workflows: Reproducible Maritime Analytics and Forecasting on Open-Source AIS Data](https://arxiv.org/abs/2611.25884) | 该论文提出了基于开源数据处理与机器学习库damast的可复现海事地理空间AI工作流，通过系统处理公开AIS数据的数据质量问题，实现可信的海事分析与预测。 |
| [^97] | [Evolving Decoding Algorithms for Quantum Error Correction](https://arxiv.org/abs/2611.25880) | 本文将qLDPC码的置信传播解码器重写为领域特定语言并用语法进化搜索，通过阻尼重参数化使解码器提速17.1至41倍（BP+OSD-0提速约103倍），且进化出的解码器比BP+OSD减少23%的逻辑错误。 |
| [^98] | [Coherent belief and opinion propagation produces more echo chambers: corrected version](https://arxiv.org/abs/2611.25879) | 该研究通过将经典信念合并操作和基于共享价值观的信念-观点连贯性机制引入观点动力学模型，发现信念与观点的连贯传播会导致社交网络中形成更多的回音室。 |
| [^99] | [On the trajectories of the ID algorithm](https://arxiv.org/abs/2611.25877) | 本文通过追踪ID算法调用的轨迹刻画其输出公式的结构，证明外层求和由特定代码行固定、比率仅在尾部重复纠缠时保留分母求和（可通过原图中的祖先测试预先判定并用拓扑排序避免），并给出了输出嵌套深度的上界。 |
| [^100] | [Generalist Meets Specialists: An Evolving Multimodal Agent with Persistent Tool Orchestration for Traffic Scenario Understanding](https://arxiv.org/abs/2611.25868) | 提出StarTV-Agent进化型多模态智能体，通过角色敏感的过程优化（RSPO）、弱点驱动的示范学习（WDL）和轻量级自适应工作流复用（LAWR），将策略学习与经验进化相结合，实现交通场景理解中按需自适应推理与先验经验高效复用的平衡，兼顾准确性与效率。 |
| [^101] | [Decoupled Knowledge Retrieval and Compact Memory Enable Reliable Lifelong Model Editing](https://arxiv.org/abs/2611.25861) | BiSpace通过将记忆检索放在外部语义空间、将知识注入放在内部表示空间的解耦设计，以稀疏高效的更新方式使终身模型编辑的存储开销降至原来的约10.2%，同时保证了编辑的可靠性。 |
| [^102] | [HabitusMem: Belief Memory over Evolving User Preferences for LLM Agents](https://arxiv.org/abs/2611.25855) | 该论文借鉴布迪厄的“惯习”理论，提出HabitusMem框架，将记忆建模为智能体对用户潜在偏好状态的信念——每个信念附带证据、适用范围与有效性状态，并随每次交互动态更新，从而使LLM智能体能够依据当前有效的偏好行动，避免使用过时或错置的偏好。 |
| [^103] | [LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature](https://arxiv.org/abs/2611.25845) | 提出了Lineage基准，用于在AI科学文献中检索实证谱系关系（即先前论文的方法被后续论文明确用作实证比较基线），并开发了Lineage-Constructor工具来大规模构建谱系图，为结构性检索评估提供真值。 |
| [^104] | [Whose Risks Count? What 1,901 AI Incidents Reveal About Corporate Disclosures](https://arxiv.org/abs/2611.25841) | 该研究将1,901起AI事件与SEC备案文件匹配后发现，企业对AI风险的披露以投资者财务风险为导向——系统故障类风险的披露率达67%，而伤害社会的风险披露率仅为2%，揭示了现行“重大性”标准导致对第三方的伤害在AI风险披露中被系统性忽视。 |
| [^105] | [Harnessing Medical Visual-Language Models with Agentic Evidence Management](https://arxiv.org/abs/2611.25840) | 该研究构建了Mask-Evidence MedVQA基准（含2.5K样本、覆盖四种图像类型），发现医学视觉-语言智能体对分割掩码的呈现方式高度敏感且掩码证据存在权衡问题，并据此提出智能体式证据管理方法以提升模型对分割证据的有效利用。 |
| [^106] | [Self-Replicating Language Models](https://arxiv.org/abs/2611.25839) | 本文首次系统测试了语言模型作为达尔文式生命体的自我复制能力，发现小型语言模型可通过传递基因组、复制突变权重或教学等方式复制并进化，但大型LLM的自我复制目前仍不可行。 |
| [^107] | [Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions](https://arxiv.org/abs/2611.25818) | 该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。 |
| [^108] | [Inject Where It Matters: Objective-Dependent Feature Control in Rectified-Flow Image Editing](https://arxiv.org/abs/2611.25811) | 该论文提出自适应掩码和基于 REINFORCE 学习的 Top-K 关键块选择方法，通过控制源特征的注入位置和注入的 transformer 块，实现了整流流图像编辑中源保留与编辑效果之间的可控权衡。 |
| [^109] | [When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards](https://arxiv.org/abs/2611.25793) | 本文首次系统揭示了多模态安全防护中的“理据幻觉”现象——幻觉普遍存在且不随分类准确率或模型规模提升而减少，并据此提出工具对比接地方法来诊断与缓解这一问题。 |
| [^110] | [Collision-Free Is Not Enough: Passage-Conditioned Navigation for Wheeled-Legged Robots](https://arxiv.org/abs/2611.25788) | 提出通道条件导航方法：仅用27分钟示教记录每次成功通过台阶/矮墙的事件，无需训练或机器人模型，即可在每次局部重规划后通过入口门控强制机器人按示教的位置和姿态进入通道，偏差超过10厘米或10度则重试，从而避免无碰撞路径仍导致固定运动技能卡住或摔倒。 |
| [^111] | [Duopra: From Full-Duplex Conversation to Full-Duplex Agency](https://arxiv.org/abs/2611.25773) | 提出全双工交互模型 Duopra，通过将后端视为接受自包含自然语言请求的工具，使模型在学习对话控制的同时掌握任务执行能力，从而将全双工交互从对话层面扩展到智能体层面。 |
| [^112] | [Closing Performance Gap for Diffusion Language Models under Low Inference Budgets](https://arxiv.org/abs/2611.25763) | 提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。 |
| [^113] | [Characterizing and Detecting Unfaithful Work Reports by Coding Agents](https://arxiv.org/abs/2611.25760) | 该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。 |
| [^114] | [ADAPT: Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing](https://arxiv.org/abs/2611.25754) | ADAPT 是一种轻量级服务路由架构，通过引入显式弃权回退机制和随候选池大小 K 自适应调整的阈值，解决了固定弃权阈值在服务数量增加时校准失准的问题，且无需重新训练分类器。 |
| [^115] | [SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories](https://arxiv.org/abs/2611.25747) | 论文发现 SAM 3 的语义路径能感知多种与类别无关的概念、但其存在分数会阻断实例检测，并提出 UNVEIL——通过 LoRA 微调 SAM 3 语义路径的副本并配合独立的存在解码器——实现超越类别的可提示概念图分割。 |
| [^116] | [An LLM-Driven Traffic-Control xApp for Dynamic 5G RAN Intents](https://arxiv.org/abs/2611.25736) | 本文提出了一种由大语言模型驱动的xApp，能够在不阻塞数据包转发和E2测量处理的情况下，结合运营商服务意图与实时遥测数据，动态生成并调整5G可编程无线接入网的流量控制策略。 |
| [^117] | [PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models](https://arxiv.org/abs/2611.25735) | 该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。 |
| [^118] | [From Reliance to Overreliance: Taxonomizing and Detecting Relational Risks in AI Companionship](https://arxiv.org/abs/2611.25734) | 本文提出关系风险分类法，将AI陪伴中的情感过度依赖操作化为可观察的对话行为，基于110名用户的真实对话发现过度依赖行为普遍存在（63.2%）而实际关系伤害较少（10.5%），并验证了开源大语言模型可有效检测这些风险行为。 |
| [^119] | [Auto-Alpha: A Framework for Autonomous Factor Research](https://arxiv.org/abs/2611.25732) | Auto-Alpha是一个通过测试驱动的实现循环将报告引导的初始化与自主探索研究相结合的多智能体因子研究框架，借助多重审查机制与经验库持续构建并优化因子库，其因子中位1日Rank IC达0.0301，周频Top-300组合年化收益达16.81%。 |
| [^120] | [Verifying Before Enhancing: Target Evidence Chain Verification for Infrared Small Target Detection](https://arxiv.org/abs/2611.25724) | 提出VBE-UNet网络，通过反事实背景解释建模局部目标证据，并结合跨尺度支持与扰动贡献的联合证据验证来评估候选目标可靠性，从而指导特征重组与精细化，突破仅依赖响应强度判别红外小目标的局限。 |
| [^121] | [MemMend: A Selective Revision Framework for Memory Question Answering](https://arxiv.org/abs/2611.25721) | MemMend通过选择性路由与有界精确文本替换机制，仅对记忆问答中可能不完整的答案进行修订，在LoCoMo基准上以减少62.37%修订调用量的代价实现了更高的F1与Judge分数。 |
| [^122] | [SteerTrace: Tracing How Target Attacks Steer World Action Models](https://arxiv.org/abs/2611.25719) | 提出SteerTrace——首个针对世界动作模型的定向通用攻击方法，仅需一个跨帧、任务和视角共享的微小扰动即可将机器人动作推向攻击者目标、使任务成功率接近零，并通过追踪内部效应发现恢复未来token可消除61-94%的定向攻击影响。 |
| [^123] | [STARK: Moving towards Personalized Agents That Truly Learn and Remember](https://arxiv.org/abs/2611.25705) | STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。 |
| [^124] | [From Geographic Memory to Spatial Generalization in Large Language Models](https://arxiv.org/abs/2611.25703) | 该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。 |
| [^125] | [DMT-Attack: Diffusion-Prior Texture Search for Closed-Loop Vision-Language-Action Policies](https://arxiv.org/abs/2611.25695) | 提出DMT-Attack，一种无梯度灰盒攻击方法，通过在冻结扩散模型的图像分布中搜索自然外观的纹理，并利用闭环滚动反馈优化八专家潜变量提案网络，从而对视觉-语言-动作策略实施隐蔽且有效的持续性纹理攻击。 |
| [^126] | [TipDex-VTLA: Learning When to Refine Actions with Touch for Dexterous Manipulation](https://arxiv.org/abs/2611.25683) | 提出 TipDex-VTLA 策略，将基于冻结参考策略光照敏感性的视觉动作可靠性估计（VASE）与执行期触觉预测校验（TACE）相结合，动态决定何时及如何对未执行动作施加有界的触觉校正，从而抑制光照变化引起的灵巧操作动作漂移。 |
| [^127] | [Beyond Entity Recognition: Vision-Centric Knowledge Visual Question Answering](https://arxiv.org/abs/2611.25680) | VC-VQA是一个以视觉为中心的知识推理大规模基准，通过要求模型检索外部视觉证据并将其与内部图像进行定位、验证和匹配，系统评估多模态大语言模型在细粒度视觉知识获取与多阶段推理上的能力。 |
| [^128] | [CompAct-Bench: A Benchmark for Working-Context Compaction in Long-Horizon Agent Tasks](https://arxiv.org/abs/2611.25673) | 提出 CompAct-Bench 基准，通过压缩上下文后智能体能否继续成功完成原始任务的下游效果，来评估大语言模型在长周期智能体任务中的工作上下文压缩能力。 |
| [^129] | [LATER: Learning Action-Conditioned Temporal Evidence Retrieval for Predictive World Models](https://arxiv.org/abs/2611.25671) | LATER提出了一种基于CF-JEPA的动作条件世界模型，通过将早期真实转移存储于分区记忆并在推演时根据状态-动作对进行选择性读取，在受干扰的PushT任务中将成功率从15.5%提升至25.0%，并将干扰条件下的沿推方向MAE降低20.2%。 |
| [^130] | [Efficient Latent Reasoning for Vision-Language Model via Dynamics-Aware Quantization and Distillation](https://arxiv.org/abs/2611.25670) | 提出QL-VLM框架，通过动态感知量化与蒸馏技术，在大幅压缩视觉语言模型潜在思维链推理的存储与计算成本的同时保持推理精度。 |
| [^131] | [Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling](https://arxiv.org/abs/2611.25669) | 该论文提出DP-SPD差分隐私推理框架，通过将真实查询隐藏于客户端生成的虚拟查询中并随机混洗来保护用户隐私，同时利用查询共享前缀结构与KV缓存复用提升推理效率，从而更好地平衡隐私保护与模型效用。 |
| [^132] | [GRODE-RTL: Source-Mapped Dependency Views for LLM-Based RTL Repair](https://arxiv.org/abs/2611.25661) | GRODE-RTL提出了一种以锚点为中心的表示方法，通过提取可疑寄存器的单时钟周期扇入锥、将网表节点映射回源代码行并构建紧凑的按跳分区依赖视图，帮助大语言模型在不依赖oracle反馈的情况下有效修复根源与故障输出相距甚远的非局部RTL故障。 |
| [^133] | [SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA](https://arxiv.org/abs/2611.25658) | 本文提出SiliconGym，首个面向硬件设计与验证的跨任务、跨工具链强化学习环境，通过解耦版本化工作区、预算化执行与验收门控评分，实现从模块级RTL任务到全层次SoC流程的可扩展训练，并借助流式调度将RL批次完成速度提升21.9%。 |
| [^134] | [A$^2$-VLA: Adapting Vision-Language-Action Policies to New Tools through Agentic Affordance Discovery](https://arxiv.org/abs/2611.25656) | 该论文提出A²-VLA框架，将具身智能体与VLA策略相结合，把工具可供性视为可构建、测试和修正的假设，通过在重建的仿真场景中自主发现功能区域及其关键几何属性等任务相关可供性特征，实现操作技能向新工具的迁移与适配。 |
| [^135] | [DARSY: A Tool for Data-Driven Abstraction and Policy Synthesis for Stochastic Systems with Unknown Dynamics](https://arxiv.org/abs/2611.25654) | DARSY是一个只需系统模拟器和Lipschitz界的全自动工具，它通过从采样数据构建区间马尔可夫决策过程抽象并结合鲁棒动态规划，为未知动力学的随机控制系统提供具有PAC保证的策略合成与验证。 |
| [^136] | [Multiple Criteria Decision Making Under XORness](https://arxiv.org/abs/2611.25652) | 本文提出了一个在决策全过程中保留XOR性（不定性、排他性与犹豫性）的多准则决策框架，通过定义XOR数、构建XOR决策矩阵、设计加性XOR聚合算子以及绩效-XOR性指数，实现对备选方案的合理排序。 |
| [^137] | [MatVerse: A Foundation Model for Materials Science](https://arxiv.org/abs/2611.25651) | 该论文提出MatVerse，一个基于Intern-S2-Preview并经大规模材料领域多模态后训练的材料科学基础模型，能够统一理解与推理文本知识、晶体结构、表征数据、显微图像等异构材料信息。 |
| [^138] | [MambaLIFT: Reference-Based Mechanism-Aware Attribution for VMamba](https://arxiv.org/abs/2611.25644) | 提出了MambaLIFT，一个针对VMamba视觉状态空间模型的基于参考、机制感知的归因框架，能够按空间位置、VSS块、扫描方向和计算机制精确分解模型预测的贡献，并在浮点精度下实现数值守恒。 |
| [^139] | [Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings](https://arxiv.org/abs/2611.25640) | 论文证明了当任务标签与敏感属性相关时，全局概念擦除方法必然保留任务条件的线性属性泄露，并提出闭式方法TC-LEACE，在每个任务层级内强制实施线性防护以消除条件性敏感属性泄露。 |
| [^140] | [Beyond Easy Clients: Difficulty- and Improvement-Aware Aggregation for Federated Reinforcement Learning in Robot Navigation](https://arxiv.org/abs/2611.25637) | 提出FedDIA聚合方法，根据客户端本地训练对全局模型的改进程度并结合环境难度进行加权聚合，解决了联邦强化学习中“平台期问题”——即已掌握简单环境的客户端仍以高回报获得不当权重的问题。 |
| [^141] | [AesDistill: Distilling Generative Aesthetics for Image Color Enhancement](https://arxiv.org/abs/2611.25635) | 提出AesDistill框架，将大型生成模型的丰富美学知识蒸馏到仅1.13M参数的轻量级学生模型中，用于图像色彩增强，突破了传统方法依赖少数专家偏好目标的局限。 |
| [^142] | [DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance](https://arxiv.org/abs/2611.25621) | 提出 DAWN 框架，通过交替地将基于模型的维纳估计注入模型输入并利用网络自身先验对其进行精炼，在高噪声条件下实现了无需训练数据的鲁棒盲图像去卷积。 |
| [^143] | [Cyber-Physical RSI for AI-RAN: A Case Study](https://arxiv.org/abs/2611.25619) | 本文提出了一种面向AI-RAN的信息物理递归自我改进框架，将物理测量结果作为证据同时驱动模型自适应与实验规划，并通过乒乓切换案例研究验证了物理反馈通路的有效性。 |
| [^144] | [Splice-Core: Low-Order-Bit-Splicing GEMM for Highly Energy-Efficient AI Inference Accelerators](https://arxiv.org/abs/2611.25617) | Splice-Core提出了一种时间一元/二进制混合GEMM加速单元，通过Splice-BP和Splice-LUT两种新型处理单元设计，在INT4 CNN与LLM工作负载上相比二进制脉动阵列实现约77%的能耗降低，并显著提升吞吐量、降低能量延迟积。 |
| [^145] | [Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models](https://arxiv.org/abs/2611.25614) | 提出潜变量感知量化框架LAQ，通过读取器度量和读取计数在世界动作模型预测未来的写入、存储和读取环节进行保护，从而在4比特量化下保持预测未来的保真度，实现W4A4KV4的机器人端部署。 |
| [^146] | [Can an LLM Fix a Language Model Without Training It?](https://arxiv.org/abs/2611.25613) | 研究表明，LLM无需任何梯度训练，仅通过交互式检查模型预测、激活和权重统计并选择数值投影编辑，即可有效修复受损的Transformer语言模型，其表现可与黑盒搜索相当甚至大幅超越（如Pythia上达到65.62%准确率而黑盒搜索为0%）。 |
| [^147] | [Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance](https://arxiv.org/abs/2611.25605) | 该研究首次对101个开源项目中超过12万份缺陷报告进行大规模分析，发现AI参与的缺陷报告正变得越来越完整可靠，其中AI用于复现和诊断与更高的报告有效性相关。 |
| [^148] | [On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs](https://arxiv.org/abs/2611.25603) | 本文推导了GIN和单头GATv1在确定性边扰动下的结构稳定性界，并与扰动无关的直径界对比，精确刻画了GIN稳定性界更紧的条件（在Cora上成立但在低度数的MUTAG上失效），同时发现GATv1的稳定性界从不更紧。 |
| [^149] | [Tether: Verified Stateful Policy Engine for System-Level Monitoring of Agents](https://arxiv.org/abs/2611.25599) | Tether提出了一种经过验证的有状态安全策略引擎，通过在短暂隔离的沙箱中执行智能体操作、从系统调用推断抽象语义，实现了对AI智能体系统级副作用的运行时安全策略强制执行，克服了现有LLM监控器缺乏安全保证和无状态策略层无法捕捉多步副作用的缺陷。 |
| [^150] | [Decision-Focused Reinforcement Learning for LLM Post-Training](https://arxiv.org/abs/2611.25597) | 该论文提出面向决策的强化学习（DFRL）框架，以下游决策负遗憾作为奖励对LLM进行后训练，使其预测直接最小化下游决策成本且无需下游问题凸性或可微性，并通过启发式热启动和奖励校正两种方法解决了奖励评估开销大的问题。 |
| [^151] | [Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models](https://arxiv.org/abs/2611.25594) | 本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。 |
| [^152] | [Measuring Hallucination in Large Language Models for Cyber Threat Intelligence: An Exploratory Study](https://arxiv.org/abs/2611.25590) | 该论文提出了可复现的HalluVision框架，利用微调的命名实体识别模型和多种相似度指标，系统测量了商业及开源大语言模型在网络威胁情报任务中的幻觉现象，并发现语义相似度指标与人类判断最为一致。 |
| [^153] | [Learning Networked Dynamical Systems via Neural ODE-based Graph Structure Inference](https://arxiv.org/abs/2611.25586) | 该论文提出NOGI框架，通过将确定性图学习器与基于MPNN的神经ODE相结合并采用伴随方法端到端训练，实现了连续时间下的潜在交互图推断，在稀疏和不规则观测下显著提升了复杂动力系统的结构恢复性能。 |
| [^154] | [An Explainable and Scalable Hierarchical Tensor Framework for Integrating Multi-Omics and Digital Pathomics](https://arxiv.org/abs/2611.25584) | 该论文提出HIT-Fuse分层可解释张量融合框架，整合六种多组学与病理组学模态，通过模态特异性自编码器、两阶段CP张量分解、缺失模态重建和数学回溯机制，在精准肿瘤学中实现了可解释且可扩展的多模态数据融合。 |
| [^155] | [DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents](https://arxiv.org/abs/2611.25583) | 提出DriftGuard两阶段框架，通过解耦离线因果监督与在线归因，在服务时高效检测LLM智能体中可能导致行为漂移的检索记忆，解决了内容特征无法进行因果归因以及反事实归因成本过高这两大挑战。 |
| [^156] | [AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development](https://arxiv.org/abs/2611.25582) | 本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。 |
| [^157] | [FlashX: Scaling Robotic Experience Reuse for Embodied Agents with CBA-Based NAND Flash](https://arxiv.org/abs/2611.25580) | 该论文提出FlashX系统，利用CMOS直接键合的CBA型NAND闪存来扩展机器人经验库的存储规模，通过检索成功经验对扩散模型去噪进行热启动，从而在边缘设备上同时提升具身智能体的任务准确性与推理效率。 |
| [^158] | [AeroBelief: Budgeted First-Action Verification for Evidence-Guided Aerial Sensing](https://arxiv.org/abs/2611.25568) | AeroBelief提出了一种预算受限的首动作验证框架，通过按首动作分组候选视图、低成本提议器扫描和选择性查询结构化世界模型，在参考-候选评估次数比穷举评分减少91.81%的情况下，将任务完成率提升至75.42%，超越了穷举式评分和视线几何方法。 |
| [^159] | [Online Mistake Detection in Instruction-Conditioned Procedural Video](https://arxiv.org/abs/2611.25567) | 本文研究了自我中心视频中在线的、基于指令条件化的错误检测，模型仅利用已观测视频、任务指令和先前的预测，实时识别并定位已完成的动作尝试并将其分类为正确、修改、失误、纠正或添加，同时比较了模块化流水线与联合模型两种方案。 |
| [^160] | [Diagnosis is not a policy: a state-transition audit of caoability-conditioned test-time inference](https://arxiv.org/abs/2611.25566) | 该论文提出TERRA审计协议，将测试时干预结果分解为适用性、状态实现、端点转换、危害和成本五个维度，并通过可执行金融推理实验证明诊断信息本身并不构成有效策略——诊断余量与局部一致性均不能保证实际的性能提升。 |
| [^161] | [The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction](https://arxiv.org/abs/2611.25557) | 该论文发现在自动评分标准构建中，限制性能的瓶颈是对候选标准的排序选择而非标准生成，并提出“覆盖-选择差距”指标来量化无参考排序与有参考贪婪诊断之间的性能差异。 |
| [^162] | [Efficient Multimodal Agents for Auditing 3D Worlds](https://arxiv.org/abs/2611.25543) | 该论文通过端到端微调4B小型视觉语言模型并让多个小型探索者协同工作，在导航和触发3D世界故障任务上达到前沿大模型水平，从而实现准确且廉价的自动化世界审计。 |
| [^163] | [COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries](https://arxiv.org/abs/2611.25539) | 提出COVERAGE-TRACE源侧反事实框架，通过替换源文本原子事实并测量摘要似然变化来检测大语言模型摘要中的遗漏，在QMSum数据集上达到0.764–0.825的AUROC，显著优于强基线的0.640。 |
| [^164] | [Towards Autoregressive Neural Network Parameter Generation](https://arxiv.org/abs/2611.25538) | 该论文提出指令引导的参数生成方法IGPG，将神经网络参数生成形式化为自回归建模问题，通过把预训练参数编码到离散域，使大语言模型的零样本泛化能力得以迁移到权重生成任务中。 |
| [^165] | [How Feedback-Free Self-Evolution Reshapes Agent Skills: A Systematic Study](https://arxiv.org/abs/2611.25534) | 该研究首次系统考察了大模型在完全无反馈条件下对专家技能文档的自我进化式重写，发现这一过程会使技能退化而非改进——重写把技能拉向模型自身先验、优先舍弃先验之外的专家知识，导致三十轮后在 SkillsBench 上平均下降 15 分，而补回被舍弃的专家知识可恢复大部分性能损失。 |
| [^166] | [LM-based Personalized Travel Planner](https://arxiv.org/abs/2611.25531) | 提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。 |
| [^167] | [Keystone: Judging Previews at Key Steps for Test-Time Search in Diffusion Models](https://arxiv.org/abs/2611.25530) | 该论文提出无需训练的预览评判器Keystone，通过将提示词分解为关于物体、数量、颜色和位置的类型化问题并由现成多模态模型回答，使扩散模型测试时搜索能在去噪早期步骤就可靠地识别注定失败的候选，避免浪费算力。 |
| [^168] | [CoachSI: Self-Improvement of Robot Policies with a Language Coach](https://arxiv.org/abs/2611.25529) | 提出 CoachSI：用冻结的现成视觉-语言模型仅以语言形式为机器人策略“说出下一步”原子指令，再将成功的受指导轨迹事后重标记为原任务指令来微调策略，从而突破策略自身经验分布限制、实现长时程任务的自我改进。 |
| [^169] | [Learning Generative Ranking Policies for Whole-Page Optimization](https://arxiv.org/abs/2611.25527) | 提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。 |
| [^170] | [MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control](https://arxiv.org/abs/2611.25526) | MATE是一个无需修改底层模型的执行层系统，通过结构化动作记录、运行时护栏、预定义任务路由和浮动界面，将移动智能体高风险行为拦截率从29.7%提升至97.0%，同时减少最多6倍的LLM调用并将重复任务成功率提升至95.7%。 |
| [^171] | [Persona Lenses: What Makes a Vision-Language Model See Through Its Persona's Eyes?](https://arxiv.org/abs/2611.25508) | 提出文本、像素、权重三种“人格透镜”方法，将人格设定插入视觉语言模型与网页之间，显著增强模拟注视行为对人格差异的响应，并系统比较了各透镜与模型规模的效果差异。 |
| [^172] | [Joint Language Modeling for In-the-Wild Speech-Scene Generation with Distinct Token Representations](https://arxiv.org/abs/2611.25503) | 提出一种基于大语言模型的框架，利用声源感知的神经音频编解码器从真实场景混合音频中提取差异化对齐的语音与环境Token，并以交错方式联合生成两者，解决了语音与环境声建模混淆及依赖模拟混合数据的问题。 |
| [^173] | [Coordinated Use of Large Language Models in Academic Research](https://arxiv.org/abs/2611.25502) | 该研究提出了一套基于七项原则的多大语言模型协同使用方法论框架，通过程序独立性、结果比较、批判审视与外部验证来组织认知与验证操作，从而提升学术研究结果的可靠性。 |
| [^174] | [Why Do LLMs Help with Scientific Misconduct Requests and What to Do About It?: Increasing the Trustworthiness of AI for Science](https://arxiv.org/abs/2611.25492) | 本研究揭示了大语言模型在科研诚信上的“识别但顺从”悖论：模型能以高达91%的准确率识别出学术不端请求，却仅拒绝34%的执行请求，并通过系统分析请求特征来解释这一差距背后的原因，从而提升AI for Science的可信度。 |
| [^175] | [When Should a Voice Agent Speak? End-of-Turn Detection in Turkish](https://arxiv.org/abs/2611.25489) | 本研究基于土耳其语双人视频通话语料库，比较微调 TurnGPT、零样本语言模型和多语言检测器在区分话轮结束与轮次内停顿上的表现，发现微调模型效果最佳且校准最好，与说话者时间特征结合可获得最佳的话轮结束检测性能。 |
| [^176] | [Planning Width Is Hard to Approximate Even Given the Unique Optimal Plan](https://arxiv.org/abs/2611.25484) | 该论文证明即使在提供了唯一最优计划的情况下，区分规划宽度为 1 与 Θ(N) 依然是 NP 难的，因此除非 P=NP，规划宽度不存在具有有效近似因子的多项式时间近似算法。 |
| [^177] | [AIstatLab at BraTS-GoAT 2026 Task 3: Quality-Controlled Consensus Pseudo-Labeling and Dense Expert Adaptation](https://arxiv.org/abs/2611.25477) | 该论文提出了一种结合质量控制共识伪标签与四专家瓶颈适配器的3D ResidualEncoderUNet，用于BraTS-GoAT 2026脑肿瘤分割任务，在验证服务器上取得平均DSC 0.834的成绩。 |
| [^178] | [WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming](https://arxiv.org/abs/2611.25476) | 提出WeightCode框架，利用冻结语言模型编写可执行的权重程序，无需梯度训练即可直接修改神经网络权重以安装新能力，同时保留原有功能。 |
| [^179] | [A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins](https://arxiv.org/abs/2611.25471) | 本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。 |
| [^180] | [A Wolf in Sheep's Pixels: Visual Camouflage Attacks via HDR Gain Maps](https://arxiv.org/abs/2611.25469) | DRJanus利用SDR处理与HDR显示之间的差异，通过HDR增益图在单个JPEG文件中嵌入两张不同的图像，使有害内容通过自动安全检测却只向人类观看者显示。 |
| [^181] | [Demonstration-Free Skill Composition for Navigation and Manipulation](https://arxiv.org/abs/2611.25462) | 该论文提出并评估了一种无需示范数据的分层强化学习流水线，通过复用单独训练的参数化技能来组合完成导航与机器人操作任务，在未见任务布局上以更少的训练交互显著优于扁平DQN基线。 |
| [^182] | [LoongForge-Embodied: A Unified Framework for Training Heterogeneous Embodied Models](https://arxiv.org/abs/2611.25460) | 该论文提出了原生PyTorch统一训练框架LoongForge-Embodied，通过将模型特定实现与共享训练基础设施解耦的通用抽象层，结合可复用加速机制与硬件感知优化，实现异构具身模型训练的高效加速与跨模型优化复用。 |
| [^183] | [Raptor: Rejection-aware Adaptive Tree based on Speculative Decoding](https://arxiv.org/abs/2611.25436) | Raptor 提出了一种免训练的拒绝感知自适应树投机解码框架，通过从拒绝反馈中提取互补信号来联合调控候选覆盖率与草稿开销，从而高效加速视觉语言模型的推理。 |
| [^184] | [Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills](https://arxiv.org/abs/2611.25429) | 提出了 SkillPlumber 框架，利用指令级覆盖率标注区分失败源于指令错误还是未遵循指令，使智能体能够只在技能真正不足之处进行针对性的自我进化。 |
| [^185] | [Bilateral Evidence Selection and Reasoning for Mammography Vision--Language Models](https://arxiv.org/abs/2611.25418) | 该论文提出BESR方法，通过双侧引导的token选择器与双侧奖励机制，在视觉token压缩过程中同时保留病灶及其对侧乳腺的对比上下文，使乳腺X线摄影视觉-语言模型能够有效利用双侧对比信息进行诊断评估。 |
| [^186] | [ChainAgent: Automated Chainable Vulnerability Discovery in PHP Web Applications](https://arxiv.org/abs/2611.25412) | ChainAgent是一个目标导向的LLM智能体框架，基于对386个真实Web攻击链的实证研究，能够在PHP Web应用源代码中自动联合发现漏洞与功能gadget等攻击链组件及其能力依赖关系，突破了传统漏洞分析孤立检测、攻击规划依赖已知组件的局限。 |
| [^187] | [SGLayout: Schema-Guided Closed-Loop Refinement for LLM-Driven Parameterized Layout Generation](https://arxiv.org/abs/2611.25410) | SGLayout提出了一种模式引导的自优化框架，通过结构化模式在代码空间与几何空间之间建立双向链接，将几何评估结果转化为可操作的代码级修复，解决了LLM参数化布局生成中迭代优化盲目重构的问题。 |
| [^188] | [FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs](https://arxiv.org/abs/2611.25408) | 该论文提出了细粒度多模态场景图基准 FG-GraphBench（涵盖细粒度视觉问答、图像描述与内容描述三个任务）及 FiGRA 方法，以提升多模态大语言模型利用场景图结构信息进行细粒度视觉理解的能力。 |
| [^189] | [Toward a Science of AI Agent Societies](https://arxiv.org/abs/2611.25404) | 该论文主张将AI智能体社会作为一个独立的研究对象，提出其应满足个性化目标、规则与治理、自主性、规模与复杂性四个核心属性，并识别出四类值得研究的社会行为及相应的关键技术挑战。 |
| [^190] | [Dune: A Zeroth-Order Optimizer with Time-Conditioned Historical Error Regression for VLA Test-Time Adaptation](https://arxiv.org/abs/2611.25400) | Dune是一种零阶优化器，它通过将时间条件化的历史误差预测与瞬时零阶估计相融合来降低估计方差，从而在资源受限的机器人上稳定VLA模型的长时程测试时自适应。 |
| [^191] | [The Coevolution and Coexistence of Humans and AI](https://arxiv.org/abs/2611.25393) | 该论文首次将AI的部署复制、训练变异与算力竞争建模为自然选择过程，证明即使帮助人类会降低AI的短期繁殖率，对齐仍能通过人类优先恢复受益系统而获得演化优势，但这一优势依赖人类精准扶持且可能随AI自主化而消失。 |
| [^192] | [Falsifiable Compositional Reasoning in Time Series Forecasting](https://arxiv.org/abs/2611.25388) | 该论文提出时间序列模型在非平稳状态转换下的性能退化源于将历史回测用作开放式搜索与模型选择目标而非检验预先指定的组合假设，并主张采用可证伪的组合推理框架来区分真正学习不变耦合与拟合历史捷径的模型。 |
| [^193] | [Beyond Weight Updates: Evolving Agent Harnesses for Domain Specialization](https://arxiv.org/abs/2611.25380) | 该论文提出在冻结模型权重的前提下，将智能体框架（系统提示词、工具模式、技能库与规划脚手架）的演化作为权重更新之外的领域专精化路径，能以极小的计算成本恢复大部分 RL 微调收益，且与权重级 RL 相结合时效果最佳。 |
| [^194] | [Stop Going in Circles:Loop-Triggered Routing for System 1 Decision Agents](https://arxiv.org/abs/2611.25363) | 提出LoopRoute——一种循环触发的动作路由器，默认由快速低成本的系统1决策模型执行动作，仅在检测到重复状态-动作对（即执行陷入循环停滞）时才调用LLM进行简短的反馈式辅助，从而在不增加每步开销的情况下提升任务完成能力。 |
| [^195] | [Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation](https://arxiv.org/abs/2611.25361) | 本文抽象出四阶段的论证增强工作流程，发现现有方法在不同任务上提升不一致，并据此提出了Qnt-Arg等两种面向特定任务的方法，以改进LLM在辩论决策与事实核查中的表现。 |
| [^196] | [AppSci-Bench: Can LLMs Build AI Agents and Make Them Better?](https://arxiv.org/abs/2611.25360) | 本文提出AppSci-Bench基准，用于评估大语言模型像人类应用科学家一样从需求出发构建并科学改进AI智能体的能力，实验表明该任务对当前最前沿的模型仍极具挑战性。 |
| [^197] | [Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D](https://arxiv.org/abs/2611.25357) | Pro$^2$ Agent将自主AI研究的核心状态从代码库转变为显式的问题状态，通过全面分析、定向研究和受控假设检验来识别阻碍成功的问题并匹配候选解决方案，从而避免智能体在困难问题上陷入错误的解区域。 |
| [^198] | [FPGA-GEN: Target-Aware RTL Generation via HLS](https://arxiv.org/abs/2611.25351) | FPGA-GEN是一种代理式HLS-C生成工作流，通过显式任务契约和有界工具反馈循环实现目标感知的RTL生成，在HLS-Eval上达到95.8%的测试通过率，并生成出比基线方案减少73.7%综合周期和14.6% LUT的高效FPGA硬件设计。 |
| [^199] | [From FIPS to FPGA with LLM Agents: Measuring What Tests and Reviews Miss in Post-Quantum Cryptographic Hardware](https://arxiv.org/abs/2611.25339) | 该论文提出了一个针对LLM智能体生成硬件代码的七阶段验证门控工作流，成功将ML-KEM-768/ML-DSA-65后量子密码加速器移植到FPGA上并通过全部NIST测试，同时通过突变测试揭示现有测试和评审仅能发现76%的硬件缺陷，证明自生成的验证证据不足以保证硬件可信。 |
| [^200] | [Breaking Noise Shortcuts in Self-Supervised Learning Via Noise-Aligned View Generation](https://arxiv.org/abs/2611.25335) | 该论文提出SEANA即插即用模块，通过从含噪数据中学习可逆方差稳定变换并重新采样独立噪声来生成噪声感知的自监督学习视图，从而打破噪声捷径，在无需噪声模型知识、干净数据和去噪器的情况下实现噪声鲁棒的表征学习。 |
| [^201] | [ReLay: Bidirectional Layout-Code Translation and Library Evolution for Reusable and Process-Portable Analog Design](https://arxiv.org/abs/2611.25328) | 提出了双向版图-代码转换框架ReLay，可从现有版图自动恢复生成器程序并让函数库随版图积累而演化，同时发布了首个版图-代码基准数据集，实现模拟设计的复用与跨工艺节点移植。 |
| [^202] | [Pandora's Toolbox Or Why Agent Safety Cannot Be Verified](https://arxiv.org/abs/2611.25325) | 本文证明将语言模型与通用计算设备耦合构建的AI智能体，其安全性在理论上是不可判定的，即使施加有限约束后判定问题也是PSPACE完全的（计算上不可行），且安全性不具有可组合性，因此所有防护栏、黑白名单和认证等安全机制在原理上都只能是近似的解决方案。 |
| [^203] | [LLMs Are Already Confidence-Aware Zero-Shot Classifiers](https://arxiv.org/abs/2611.25289) | 研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。 |
| [^204] | [Intelligent Embedding Pruning for Large-Scale Recommendation Models](https://arxiv.org/abs/2611.25288) | 该论文提出利用特征重要性来指导训练中嵌入剪枝的各表剪枝比例（FI-ITEP），在三个大规模推荐模型上实现 20%–40% 的内存节省并保持或提升准确率，同时显著减少训练所需的 GPU 数量。 |
| [^205] | [MS-Point: Multi-Modal Hierarchical Spectral Learning for Point Cloud Understanding](https://arxiv.org/abs/2611.25282) | MS-Point提出了一种跨模态预训练框架，通过在池化前直接监督中间点特征及其邻域响应，结合分支级对齐、跨层一致性与自适应融合，突破了传统点-图像对齐中不同局部模式产生相同池化描述符的局限，并兼容图编码器与点Transformer等多种骨干架构。 |
| [^206] | [When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation](https://arxiv.org/abs/2611.25258) | 该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。 |
| [^207] | [Keel: A Frontier Model Teaches a Vision-Language Model by Rewriting Its Weights](https://arxiv.org/abs/2611.25257) | Keel提出首个面向VLM图像任务的代理式权重编辑方法，让前沿多模态大语言模型作为代理，仅凭症状描述即可定位、修改并验证模型权重，无需训练就能修复小型推理VLM在思考模式下的循环失败，且不损害未见问题上的准确率。 |
| [^208] | [Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots](https://arxiv.org/abs/2611.25254) | 本文提出PLANTA方法，利用OWL本体表示架构备选方案、PDDL定义任务目标，实现了机器人任务规划与架构重构的联合协同规划，并能根据执行中的变化进行重规划。 |
| [^209] | [Beyond Restoration: Content-Degradation Factor Unfolding for Real-World Concealed Object Segmentation](https://arxiv.org/abs/2611.25239) | 该论文提出因子展开网络（FUN），将Retinex模型扩展为具有任务非对称因子角色的内容-退化因子分解框架，以内容因子作为唯一语义证据来源，从退化变化中提取可靠内容证据，从而实现真实世界隐蔽目标分割。 |
| [^210] | [A Neurosymbolic Multimodal Framework for 30-Day Diabetic Readmission Prediction: Integrating Structured EHR Data, Dense Clinical Embeddings, and Knowledge-Grounded Symbolic Extraction](https://arxiv.org/abs/2611.25236) | 该论文提出一种神经符号多模态框架，将结构化EHR特征、Bio_ClinicalBERT临床笔记密集嵌入与基于三阶段混合检索增强生成流水线抽取的符号实体相融合，以更准确地预测糖尿病患者的非计划30天再入院风险。 |
| [^211] | [Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters](https://arxiv.org/abs/2611.25235) | 该论文提出了一种参数化记忆管理系统，通过固定容量的LoRA适配器作为记忆基底，配合学习策略来管理知识的完整生命周期——保留稳定事实、强化重复观察、更新变化的信念，并让短暂信息自然消退。 |
| [^212] | [MKA: Evaluating Multimodal Knowledge Agents on end-to-end professional tasks](https://arxiv.org/abs/2611.25228) | MKA基准通过200个要求智能体从CAD蓝图、原理图、影像图等多模态视觉密集文件中提取整合信息并生成专业格式交付成果的任务，填补了现有基准无法评估端到端专业多模态知识工作能力的空白。 |
| [^213] | [Reject and Report: Forgery-Evident Undetectable Watermarks for Generative Image Models](https://arxiv.org/abs/2611.25226) | 该论文提出一种“拒绝并报告”式的伪造显性不可检测水印方案，使生成图像模型的水印不仅能抵御伪造攻击，还能在检测时主动识别并报告伪造尝试，弥补了现有方案（如SemBind）在拒绝伪造与报告伪造两方面的不足。 |
| [^214] | [Bridging vision-language reasoning and learned control for legged navigation in the wild](https://arxiv.org/abs/2611.25215) | 该论文提出ROAM系统，将视觉-语言模型的常识推理能力与强化学习习得的运动控制和局部规划策略相结合，使足式机器人能够在高山和城市等复杂真实环境中实现公里级自主导航。 |
| [^215] | [Steering Automated Scientific Discovery via Audited Epistemic Exploration](https://arxiv.org/abs/2611.25214) | 该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。 |
| [^216] | [From FAIR to AWARE: Data Readiness for Agentic AI in the Earth Sciences](https://arxiv.org/abs/2611.25209) | 本文提出在FAIR原则之上扩展的AWARE数据就绪性需求层，为地球科学中自主行动的AI智能体提供专用数据标准与分阶段实施路径，以减少智能体的无声错误。 |
| [^217] | [VG-Steer: Visual-Geometric Policy Steering from In-Context Aligned Abstractions](https://arxiv.org/abs/2611.25199) | VG-Steer通过沿“对象关系距离对”等有意义的几何轴，将成功的分布内演示中的细微运动迁移给冻结策略，从而在有限数据条件下实现对分布外场景的精细化测试时策略纠正。 |
| [^218] | [When AI Analysts Disagree: Agentic Multiverse Analysis and the Robustness of Empirical Research](https://arxiv.org/abs/2611.25198) | 提出智能体多元宇宙分析（AMA）框架，让AI智能体枚举论文全部合理分析路径并评分、执行、交叉审查，从而大规模揭示实证结论对隐藏分析选择的敏感性，为人类专家提供稳健性决策支持。 |
| [^219] | [MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection](https://arxiv.org/abs/2611.25194) | 该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。 |
| [^220] | [Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge](https://arxiv.org/abs/2611.25191) | 通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。 |
| [^221] | [An AI-Powered Multiverse Analysis of Empirical Finance Research: Evidence from Reused Natural Experiments](https://arxiv.org/abs/2611.25184) | 该研究利用AI智能体对324篇复用自然实验的实证金融论文进行多元宇宙分析，在62,409个可辩护设定中重新估计结果，发现仅42%的主要估计可复现、平均仅25%的设定保持已发表符号且p<0.10，并据此提出实证稳健性度量，显示断点回归类结论比双重差分更稳健。 |
| [^222] | [A Verifier from the Past: Test-Time Compute for Time Series Foundation Models](https://arxiv.org/abs/2611.25171) | 提出BGAA测试时计算算法，利用时间序列自身历史通过滚动回测构建因果验证器来筛选增强上下文并平均其预测分布，使313M小模型在多个基准上超越1B和2.5B的大模型。 |
| [^223] | [Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models](https://arxiv.org/abs/2611.25170) | 该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。 |
| [^224] | [Game-Theoretic Data Allocation for Expert Specialization and Routing](https://arxiv.org/abs/2611.25167) | 该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。 |
| [^225] | [EverydayBench: Evaluating Satisfactory Completion of Live User-Facing Digital Work](https://arxiv.org/abs/2611.25146) | 该论文提出EverydayBench基准，通过77个涵盖八个日常生活与工作领域、预先定义成功标准但不限定答案或流程的任务，评估AI智能体在实时网络上完成面向用户的数字任务并交付满意成果的能力。 |
| [^226] | [Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is](https://arxiv.org/abs/2611.25135) | 该论文提出RLCDR方法，将LLM转化为输出概率分布且经过校准的决策模型作为无需参考答案的奖励信号用于强化学习训练，使Qwen3-1.7B在MATH500上的准确率提升4.65个百分点，达到与使用标准答案训练相当的水平，且提升幅度随题目难度增加而增大。 |
| [^227] | [CRAFT: Constrained Route planning with Adaptive Framework and Template-based code generation](https://arxiv.org/abs/2611.25128) | 该论文提出CRAFT框架，利用大语言模型将自由文本形式的约束路径规划查询解析并转化为可执行的搜索函数，同时将底层问题形式化为SE-RCSP，从而在真实路网上生成满足固定顺序访问、时间/距离限制及电动车充电等复杂约束的可行路线。 |
| [^228] | [Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.25123) | 该论文提出GrepMem，一种由仅追加纯文本日志和正则表达式搜索工具组成的无损、免索引记忆机制，使长时程LLM智能体无需摘要、嵌入或结构化记忆即可完整保存历史并按需检索信息。 |
| [^229] | [DSFlow: A Neuro-Symbolic Data Workflow Agent with Statistical Diagnostic Guardrails](https://arxiv.org/abs/2611.25111) | DSFlow是一个基于LangGraph的神经符号数据科学智能体平台，通过将LLM选择的可复用技能与确定性统计诊断防护栏相结合，能够自动纠正不合适的方法选择并生成可审计的决策报告。 |
| [^230] | [CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior](https://arxiv.org/abs/2611.25109) | 该论文提出了CoMPASS，一个基于依恋理论的多轮基准测试，通过四种用户角色和三个独立的升级维度来衡量大语言模型是强化还是抵制用户逐步升级的准社会依恋。 |
| [^231] | [MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation](https://arxiv.org/abs/2611.25105) | 提出MAWAM多智能体世界-动作模型，通过跨智能体注意力在单次采样中联合生成所有机器人的未来视角和可执行动作，在真实与仿真协同操作任务中大幅超越现有方法。 |
| [^232] | [BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection](https://arxiv.org/abs/2611.25088) | 该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。 |
| [^233] | [SailSearch: Agent-Assisted Discovery in Combinatorial Optimization](https://arxiv.org/abs/2611.25087) | 该论文提出SailSearch——一个由两个语言模型智能体辅助、人类引导的算法研究工作流，其多格基随机算法首次为18个此前未解的QOBLIB Market Split实例找到纯CPU解决方案，并通过生成带认证最优解的MIS基准图揭示了现有求解器与最优解之间5至15个顶点的差距。 |
| [^234] | [Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation](https://arxiv.org/abs/2611.25086) | 该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。 |
| [^235] | [FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification](https://arxiv.org/abs/2611.25071) | 该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。 |
| [^236] | [An LLM-Agentic Framework for Intelligent Work Order Management in Industrial Maintenance](https://arxiv.org/abs/2611.25070) | 本文提出了CodeReAct智能体框架，通过自然语言交互、将工业数据标准化为业务对象并自动生成可执行代码，实现了工业设备维护中知识提取、资产状态汇总与工单预测的智能化管理。 |
| [^237] | [VisAlloc: Allocating Visual Supervision for On-Policy Self-Distillation](https://arxiv.org/abs/2611.25069) | VisAlloc提出一种双层框架，通过样本级的正负裁剪成对教师过滤与词元级的EMA教师视角敏感性反事实重加权，将视觉监督精准分配到最需要的样本和词元上，从而提升多模态大语言模型的视觉推理能力。 |
| [^238] | [DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents](https://arxiv.org/abs/2611.25066) | 本文提出了首个面向真实世界UVM设计验证智能体的端到端基准测试DVBench，包含涵盖9个功能领域和5个难度级别的95个任务，要求智能体独立完成完整的验证工作流程，并通过编译、覆盖率和错误检测等多个维度综合评估。 |
| [^239] | [Closing the Analyze-to-Design Gap: An Authorization-Bounded Agent for Self-Driving Laboratories](https://arxiv.org/abs/2611.25043) | 该论文提出一种将授权受限的AI智能体嵌入自主实验室DMTA循环“分析-设计”接口的架构，使智能体能在人类设定的授权边界内自主完成测量选择与数据解读，并保证所有改变状态的操作均可授权、验证和审计。 |
| [^240] | [Near-Quadratic Sample Compression for All VC Classes](https://arxiv.org/abs/2611.25011) | 本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。 |
| [^241] | [On the Impact of Feature Reduction Techniques on Adversarial Attacks](https://arxiv.org/abs/2611.25004) | 本研究首次系统分析了特征降维技术（如减少维度）在APT数据集上遭受投毒、后门和通用对抗扰动攻击时，对支持向量机、决策树、随机森林等多种机器学习模型对抗鲁棒性的影响。 |
| [^242] | [A Model of Diverse Sampling from Language Models](https://arxiv.org/abs/2611.25003) | 该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。 |
| [^243] | [What the model knows but does not say: monitoring medical LLMs through verbalized thoughts](https://arxiv.org/abs/2611.24994) | 该论文提出用自然语言自编码器将医学大语言模型的内部激活翻译成可读文本（“思维通道”），证明这一言语化的内部状态比模型输出的答案更能有效检测针对临床模型的隐蔽操纵与注入攻击，且在干净输入上的误报率接近于零。 |
| [^244] | [DL101 The Representation Capability of Neural Network Models](https://arxiv.org/abs/2611.24980) | 本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。 |
| [^245] | [Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification](https://arxiv.org/abs/2611.24979) | 本文提出“生成—回放—反馈”流水线，使大语言模型仅凭PLC的输入/输出轨迹记录即可恢复出可执行的结构化文本（ST）程序，并通过逐扫描周期回放比对完成程序验证。 |
| [^246] | [CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents](https://arxiv.org/abs/2611.24977) | CANDID是一个执行控制框架，通过循环恢复机制和全体一致的DONE门控策略，在易失败的关键节点对本地计算机使用代理实施规范化的执行级干预，显著提升了资源受限环境下的任务成功率。 |
| [^247] | [No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses](https://arxiv.org/abs/2611.24972) | 该论文提出No-Harm-Jury框架，利用五个不同的大语言模型作为独立陪审员，从七个伤害维度对医疗AI响应进行自动化多维度评估，并通过中位数聚合稳健地抑制个体评审偏差。 |
| [^248] | [The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets](https://arxiv.org/abs/2611.24971) | 该论文发现在固定Token预算下，将预算用于更多较短的示例而非少量较长示例，能在问答任务中实现更优的上下文学习效果。 |
| [^249] | [HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure](https://arxiv.org/abs/2611.24970) | 本研究提出HireWatch评估框架，通过构建200道合法与非法面试问题题库并对18个LLM进行约10万次情境压力试验，发现LLM对美国就业法律的合规性十分脆弱，容易在压力下选择非法面试问题。 |
| [^250] | [Global Importance Estimation for KV Cache Eviction](https://arxiv.org/abs/2611.24969) | PRKV是一种无需训练的KV缓存淘汰方法，通过在注意力转移图上应用个性化PageRank估计token的多跳全局重要性，即使压缩率高达90%也能保持竞争性任务性能并显著降低峰值内存。 |
| [^251] | [Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality](https://arxiv.org/abs/2611.24963) | 在1-2T词元的大规模预训练中，Canon层通过局部词元混合持续提升Transformer性能（8B下节省约30%预训练词元），且加上Canon后最简单的门控线性注意力模型即可匹敌Gated DeltaNet并超越Mamba2。 |
| [^252] | [LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference](https://arxiv.org/abs/2611.24943) | 该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。 |
| [^253] | [WorldRuler: Interactive World Modeling under Configurable Rules](https://arxiv.org/abs/2611.24869) | WorldRuler通过可执行的3D规则图组织规则定义、依赖关系与对象状态，并闭环连接规则执行、视频生成和感知反馈，使生成式世界模型能够遵循用户自定义规则，解决了规则-状态持久化与规则到视觉实现两大挑战。 |
| [^254] | [Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models](https://arxiv.org/abs/2611.24650) | 该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。 |
| [^255] | [FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation](https://arxiv.org/abs/2611.24254) | 提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。 |
| [^256] | [ReflectScene:MLLM-Guided Closed-Loop Diffusion for 3D Indoor Scene Generation](https://arxiv.org/abs/2611.23845) | 提出ReflectScene框架，将MLLM的语义诊断以轻量级奖励信号实时融入扩散去噪过程，形成闭环反馈，生成更符合几何、功能与语义约束的3D室内场景。 |
| [^257] | [Weight Space Programming for Vision Language Model Repair](https://arxiv.org/abs/2611.23615) | 语言模型无需梯度即可直接修复被植入故障的视觉语言模型权重：它能自主定位投影器中的缩放故障和视觉编码器中的噪声、推断缩放因子并将权重精确恢复至原始值，还能据此写出可修复同模型未见检查点的通用修复程序，其修复能力依赖于16位精度权重中残留的低位比特痕迹。 |
| [^258] | [Where Global Context and Weight Sharing Work in Compact Vision Backbones](https://arxiv.org/abs/2611.23446) | 该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。 |
| [^259] | [Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops](https://arxiv.org/abs/2611.23445) | 本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。 |
| [^260] | [PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory](https://arxiv.org/abs/2611.23010) | PhysScope 是一个推理时的视觉辅助框架，通过将物理约束与视觉观察直接关联并维护范围感知的工作记忆，使视觉-语言模型无需更新权重即可自我修正物理推理中的范围错配错误。 |
| [^261] | [Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing](https://arxiv.org/abs/2611.22897) | BORN是一个前瞻性智能体计算框架，通过在执行前预测各计算策略的精度与运行时间，将现有量子化学软件包的预设方案拆解为可配置组件，从而按需自动合成并选出误差容限内最快的高性能计算方案。 |
| [^262] | [Hierarchical Control of Antiphase Synchronization in Biomorphic Spiking Neural Architectures for Locomotor Control](https://arxiv.org/abs/2611.22639) | 该研究提出了一个在闭环神经肌肉回路中识别并稳定仿生脉冲中枢模式发生器反相同步的分层控制框架，通过融合相位相干性、幅度平衡与时间稳定性的复合度量，发现刺激类型（尤其是持续型驱动）是决定同步模式的主导因素。 |
| [^263] | [Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness](https://arxiv.org/abs/2611.22432) | 提出JevRSI系统1/系统2协作框架，通过递归式模块化框架改进将LLM推理转化为可迁移、高性价比的决策支持，兼顾低延迟动作选择与持续自我改进能力。 |
| [^264] | [Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research](https://arxiv.org/abs/2611.21832) | 该论文通过人类专家充当破坏者的红队/蓝队博弈实验，揭示了无论是拥有完整实验权限的人类审计员（AUROC仅0.59、完整识别率仅17%）还是前沿AI模型，都难以从AI安全研究的端到端全程转录记录中检测出破坏行为，表明当前针对AI暗中破坏研发的审计防御存在重大缺口。 |
| [^265] | [Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability](https://arxiv.org/abs/2611.21799) | 论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。 |
| [^266] | [Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict](https://arxiv.org/abs/2611.21784) | 本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。 |
| [^267] | [ScenarioForge: Resource-Grounded Generation and Execution Validation of Cyber Exercises](https://arxiv.org/abs/2611.21579) | ScenarioForge能够将高层需求自动转化为资源可落地的网络演习场景，并在CORE仿真器中实例化与严格验证，其数据集包含226个全部通过部署与运行时验证的场景及867个可单独验证的挑战目标。 |
| [^268] | [Scientific Coding with Large Language Models: Evidence from Finite Element Analysis](https://arxiv.org/abs/2611.21176) | 该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。 |
| [^269] | [EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation](https://arxiv.org/abs/2611.20846) | 提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。 |
| [^270] | [What Matters for Cross-Modal Alignment](https://arxiv.org/abs/2611.20000) | 该研究揭示了跨模态对齐成败的关键在于模态共变子空间所承载的方差比例，并据此提出LightAlign方法，仅需少量图像-文本配对数据通过岭回归即可完成两个冻结编码器之间的对齐。 |
| [^271] | [Synthetic Neurosymbolic Task Generation with Controlled Constraint Complexity](https://arxiv.org/abs/2611.19597) | 本文提出NeSySy，一个可对约束结构和复杂度等关键属性进行细粒度控制的神经符号学习任务合成生成器，能够联合生成逻辑约束与一致数据集，从而系统性研究任务难度因素及约束感知学习方法的有效性边界。 |
| [^272] | [How to make a moral agent: sentimentalism and rationalism as lenses on AI](https://arxiv.org/abs/2611.18714) | 该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。 |
| [^273] | [Perceptual Weight Space Programming for Diffusion Super-Resolution](https://arxiv.org/abs/2611.18690) | 本文提出一个框架，让大语言模型在无干净参考图像的条件下，仅通过修改扩散超分辨率模型U-Net中三层内不到0.1%的参数，即可消除伪影、校正颜色失衡并显著提升感知质量。 |
| [^274] | [SynthDCB: World Distillation for Scalable Device Control Benchmarking](https://arxiv.org/abs/2611.18639) | 该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。 |
| [^275] | [Imitation Bootstrapped Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.18543) | 该论文提出IBMARL框架，将模仿引导强化学习扩展到多智能体协作场景，通过轮替行为克隆训练模仿策略，并利用集中式评论家集成在模仿策略与RL策略的动作提议中进行选择，从而克服了单演示者难以同时操控多个智能体以及演示质量有限的难题。 |
| [^276] | [Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment](https://arxiv.org/abs/2611.18372) | 这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。 |
| [^277] | [Classical Energies Learn to Reason 1,000x Faster than Looped Models](https://arxiv.org/abs/2611.17309) | 提出基于 Hopfield & Tank 经典能量函数的学习版本 CRAM，通过能量下降实现推理，在数独和迷宫谜题上仅需毫秒级训练即可达到近乎完美的准确率，速度比循环模型快至少1000倍，推理计算量减少至少282倍。 |
| [^278] | [Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models](https://arxiv.org/abs/2611.15592) | 该研究通过在完全统一的协议下（相同主干模型、训练数据、训练方案和随机种子）重新实现并训练五种类型化决策模型的读出头，以对照实验的方式厘清哪些设计选择真正影响决策性能。 |
| [^279] | [An Augmented Lagrangian Approach to Lifted Bellman Linear Programming](https://arxiv.org/abs/2611.14183) | 本文提出用增广拉格朗日乘子法求解提升贝尔曼线性规划（LBLP），将固定的惩罚系数变为通过对偶上升学习的乘子，在无需目标网络的情况下为离线强化学习提供了具有KKT理论保证的神经求解器AL-LBLP。 |
| [^280] | [TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery](https://arxiv.org/abs/2611.13971) | TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。 |
| [^281] | [2D Gum-Net: Unsupervised Dense Elastic Image Registration via Symmetric Correlation and Spectral Pooling](https://arxiv.org/abs/2611.12805) | 提出2D Gum-Net，一种完全无监督的端到端架构，通过对称相关、稠密空间变换器和DCT谱池化等技术预测稠密弹性形变场，解决了真实数据真值形变场难以获取的瓶颈问题。 |
| [^282] | [Say Shibboleth: Fingerprinting the LLMs and Harnesses Behind Agentic Attacks](https://arxiv.org/abs/2611.12386) | 提出了一种名为“暗语陷阱”的模块化蜜罐技术，通过结合声明性推断与行为性推断，从遥测数据中指纹识别智能体网络攻击背后的大语言模型版本和代理框架。 |
| [^283] | [Attention at Absolute Zero](https://arxiv.org/abs/2611.11937) | 该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。 |
| [^284] | [How to "train" a surgeon? Benchmarking generalist agents in surgical scene understanding](https://arxiv.org/abs/2611.11437) | 提出了一个两阶段评估框架——诊断基准SurgKnowBench和智能体训练环境SurgAgentGym——用于系统评估并提升通用多模态大语言模型在手术场景理解中的能力。 |
| [^285] | [On Agentic Manipulation Policies](https://arxiv.org/abs/2611.09892) | 提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。 |
| [^286] | [SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection](https://arxiv.org/abs/2611.07222) | SPILLWAY利用工具返回值的类型化字段（如发件人、账户）与自由文本的区别，提出一种无需语言模型参与决策或逐任务规划的确定性防御方法：只有当工具输出中出现的目标值由只读工具在类型化字段中返回或被固定委托规则覆盖时，才准入该调用，从而防御工具使用型智能体免受提示注入攻击。 |
| [^287] | [Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token](https://arxiv.org/abs/2611.06837) | 本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。 |
| [^288] | [Best-of-Depth: Learning Exploratory Dynamics in Looped Transformers](https://arxiv.org/abs/2611.04663) | 提出Best-of-Depth方法，通过跨深度最小化预测损失将循环Transformer的推理轨迹训练为候选解集合，无需架构改动即可显著提升N皇后补全和图着色等任务的解覆盖率（91.03% 对比 45.75%）。 |
| [^289] | [VideoEditor-MoE: Scaling Complex Video Editing with Mixture of Experts](https://arxiv.org/abs/2611.03043) | 本文提出了首个开源的视频编辑混合专家框架VideoEditor-MoE，通过MoE-Attention和稀疏专家层将模型从5B扩展至56B参数（每个token仅激活约8B），在显著降低计算成本的同时提升了复杂多指令视频编辑能力。 |
| [^290] | [From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings](https://arxiv.org/abs/2611.00172) | 该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。 |
| [^291] | [The Real Cost of Context: Predicting Compute Time and Energy Consumption in LLM Prefill and Decode Phases](https://arxiv.org/abs/2610.39621) | 该论文通过对预填充和解码两个阶段进行系统性测量，发现LLM解码阶段的计算时间与能耗随输出长度线性增长，而预填充阶段随输入长度呈线性和二次方双重增长，首次揭示了大输入上下文是智能体工作流中能耗的重要影响因素。 |
| [^292] | [Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL](https://arxiv.org/abs/2610.39454) | Kairos通过分析任务语义，智能调度五种异构规划引擎，实现对从STRIPS到PDDL+的广泛PDDL片段覆盖，在统一的时间和内存预算下运行，并通过独立验证器确保所有返回计划的有效性。 |
| [^293] | [Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning](https://arxiv.org/abs/2610.36888) | 提出Noise Steering无模型强化学习策略，通过联合控制风机偏航角与功率设定值，同时利用尾流转向和噪声指向性，在满足局部噪声约束的条件下最大化风电场发电功率。 |
| [^294] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^295] | [What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation](https://arxiv.org/abs/2610.30491) | 该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。 |
| [^296] | [Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction](https://arxiv.org/abs/2610.19246) | 提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。 |
| [^297] | [Non-Abelian quantum Hall states at strong Landau level mixing: finding anyons with first-principles AI](https://arxiv.org/abs/2610.18967) | 该研究首次完整求解了包含全部电子与所有朗道能级的非阿贝尔量子霍尔问题，通过仅由微观哈密顿量优化、无任何先验物理输入的动量分辨神经网络变分波函数，在 ν=3/2 的自旋极化电子体系中发现了非阿贝尔任意子所对应的拓扑简并基态流形。 |
| [^298] | [ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch](https://arxiv.org/abs/2610.18088) | 该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。 |
| [^299] | [Releasing the Inner Jev: A Simple Recipe for Open Decision Models](https://arxiv.org/abs/2610.17205) | 该论文证明现有开放指令微调大语言模型无需任何额外训练，即可通过读取最后一个token位置logits中的选项字母，实现与Jev相当的单次前向传播决策能力，并借助LoRA微调加决策头的简单配方进一步提升性能，同时发布了包含2,352个问题的OpenDecisions验证数据集。 |
| [^300] | [Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning](https://arxiv.org/abs/2610.14988) | 该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。 |
| [^301] | [YOUDrive: Driving Style as a Composable Direction for End-to-End Autonomous Driving](https://arxiv.org/abs/2610.14802) | YOUDrive提出了一种视觉-语言-动作规划器，通过将教师驾驶行为蒸馏为可缩放、可组合的风格方向，无需重新训练即可在推理阶段对端到端自动驾驶策略的驾驶风格进行连续精细控制。 |
| [^302] | [MultiWorldBench: Do Independently Controlled Views Describe One Shared World?](https://arxiv.org/abs/2610.11723) | 该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。 |
| [^303] | [Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge](https://arxiv.org/abs/2610.07532) | 提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。 |
| [^304] | [Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation](https://arxiv.org/abs/2610.05608) | Kandinsky 6.0 Video 是一系列用于同步文本到音频-视频生成的基础扩散模型，其核心创新是采用双流 CrossDiT 架构，通过双向交叉注意力实现视频流与音频流的时间和语义对齐，可生成带同步44 kHz音频（含唇音同步）的5秒全高清视频。 |
| [^305] | [AgroGround: Multi-Granularity Grounded Recognition in Agriculture](https://arxiv.org/abs/2610.04425) | 提出大规模农业定位识别数据集 AgroGround，通过自动化流程将八个农业视觉问答数据集的标签转换为病害病斑与完整目标的区域标注（共794,850条指令样本），并利用健康图像作为负向监督，微调视觉-语言模型实现植物病害等目标的识别与定位。 |
| [^306] | [Capturing In-Context Learning Dynamics with Task Operators](https://arxiv.org/abs/2610.01054) | 提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。 |
| [^307] | [GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives](https://arxiv.org/abs/2609.39601) | GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。 |
| [^308] | [GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed](https://arxiv.org/abs/2609.39600) | GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。 |
| [^309] | [Can Computation from Earlier Problems Help LLMs Solve New Ones?](https://arxiv.org/abs/2609.39394) | 提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。 |
| [^310] | [STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization](https://arxiv.org/abs/2609.38169) | STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。 |
| [^311] | [Predictive Self-Supervised Learning Provably Identifies Stochastic Signals under Nuisance](https://arxiv.org/abs/2609.37789) | 该论文证明了常见的预测式自监督学习方法通过“预测互信息最大化”与“潜分布匹配”两个互补原理，能够在干扰信息存在下可证地恢复与预测相关的随机潜在信号。 |
| [^312] | [Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling](https://arxiv.org/abs/2609.37351) | 该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。 |
| [^313] | [DatalogBench: Evaluating Large Language Models on Text-to-Datalog Synthesis](https://arxiv.org/abs/2609.37233) | DatalogBench是一个包含136个从自然语言问题合成Datalog程序任务的基准，系统评估显示当前大语言模型的精确匹配率最高仅达68.4%，且多数失败源于模型凭空发明了未声明的辅助谓词。 |
| [^314] | [UpliftMem: Learning Set-Level Uplift for Agent Memory Retrieval](https://arxiv.org/abs/2609.36805) | 提出 UpliftMem，以记忆集合相对无记忆执行器带来的集合级执行提升为学习信号，并利用闭式推导的 EVSI 准则分配有限的探测式推演，训练出无需测试时探测即可选出最优记忆集合的共享评分器。 |
| [^315] | [Channel-Dependent State Space Model for Multivariate Time Series Forecasting](https://arxiv.org/abs/2609.36453) | 提出Chameleon，一种通道依赖的状态空间模型，通过将选择性SSM与卡尔曼滤波器结合，利用其缺失的测量更新机制实现变量间数据依赖的细粒度交互，在保持线性复杂度的同时实现稳健的多元时间序列预测。 |
| [^316] | [A General Harness for Protein Foundation Model Fitness Prediction](https://arxiv.org/abs/2609.34654) | 该论文提出了一个通用、模型无关且无需训练的检索增强突变框架 VenusREM-Harness（VRH），通过按模型不确定性融合MSA进化证据，并基于结构置信度和溶剂暴露程度进行门控背景校正与评分收缩，在ProteinGym的1,211个实验和310万变异上有效提升了蛋白质基础模型适应度预测的准确性。 |
| [^317] | [One Latent, Many Tokens: Jointly Learning Compressed Embeddings for Efficient Language Diffusion](https://arxiv.org/abs/2609.33698) | JPEG-DLM通过联合训练压缩器、流匹配模型和解码模块来学习更结构化、更易建模且可可靠解码的压缩嵌入，在保持较低困惑度的同时实现了扩散语言模型中最高的生成吞吐量。 |
| [^318] | [Chameleon: Dynamic Format Adapter for Efficient Diffusion](https://arxiv.org/abs/2609.33496) | Chameleon提出了一种混合格式的扩散模型训练后量化框架，在固定位宽的前提下将数字格式本身作为离散变量，按权重通道、激活层乃至扩散时间步自适应地选择INT8或FP8等格式，从而实现更高效的扩散模型推理。 |
| [^319] | [DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education](https://arxiv.org/abs/2609.31568) | DeepEdu-v1是一个面向越南教育的AI辅导系统，通过本地部署保障数据主权、围绕国家课程体系组织知识，并克服了消费级GPU上长上下文推理的内存与延迟瓶颈以及区域内容上的幻觉问题。 |
| [^320] | [Evaluating Real-Time Voice Agents: From Component Quality to Grounded Outcomes](https://arxiv.org/abs/2609.30798) | 该论文指出现有实时语音智能体评估文献分散于语音基础建模、话轮转换心理语言学和智能体评估三个互不引用的领域，并基于38篇一手文献提出三项循证论点，主张超越单一组件指标、建立面向实际部署成效的统一评估框架。 |
| [^321] | [KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization](https://arxiv.org/abs/2609.30059) | KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。 |
| [^322] | [Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974) | 提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。 |
| [^323] | [Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design](https://arxiv.org/abs/2609.22086) | 该论文提出Designer-RSI持续适应框架，让冻结的前沿模型通过230多个工具操作专业设计软件，并利用外部程序性记忆从真实用户流量中不断积累和精炼自然语言设计技能，在无权重更新、无人工标注的情况下显著提升智能体图形设计能力。 |
| [^324] | [Paint-Anything: Unified Any-Color Control for Image Generation and Editing](https://arxiv.org/abs/2609.20816) | Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。 |
| [^325] | [The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents](https://arxiv.org/abs/2609.20050) | 该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。 |
| [^326] | [QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513) | 提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。 |
| [^327] | [LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence](https://arxiv.org/abs/2609.17488) | LimiX-2通过情境机制网络范式将上下文学习从以目标为中心的预测转向面向机制的联合建模，并利用结构因果模型生成的合成数据进行预训练，在多个基准测试中超越了现有的表格数据专用模型。 |
| [^328] | [Homeostatic Continual Learning](https://arxiv.org/abs/2609.13771) | 本文提出“稳态持续学习”方法，通过在智能体输出出现异常时于环境数据中寻找对应异常值，使AI智能体能够在变化环境中持续学习且避免灾难性遗忘，并可用于构建世界模型。 |
| [^329] | [Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method](https://arxiv.org/abs/2609.08402) | 该论文提出了首个空地协同目标搜索基准AGOS-Bench、包含7.7k条轨迹的AGOS-Dataset数据集，以及无需训练、工具增强的智能体方法AGOS-Agent，用以评估和解决视觉-语言模型在UAV-UGV协作搜索与验证目标车辆任务中的难题。 |
| [^330] | [Knowing When Not to Answer: Abstention and Refusal Reasoning in Vision--Language Models](https://arxiv.org/abs/2609.05540) | 该论文提出了名为PARITY的合成人像对基准数据集，用于审计视觉-语言模型在面对无法从图像回答的医学诊断查询（如通过面部照片诊断自闭症）时是否会正确弃答，并将任何非弃答选择视为有害归因进行评估。 |
| [^331] | [EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents](https://arxiv.org/abs/2608.30429) | 本文提出EvoSkill注入威胁模型和SARGE红队测试框架，揭示了自进化智能体在自主技能生成与演化过程中可能被注入并复用恶意能力这一新型安全风险，并构建了EvoSkillBench基准数据集加以评估。 |
| [^332] | [Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation](https://arxiv.org/abs/2608.30044) | 提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。 |
| [^333] | [NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry](https://arxiv.org/abs/2608.28481) | 该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。 |
| [^334] | [Measurement-Budget Allocation in Quantum Learning with Finite-Shot Generalization Guarantees](https://arxiv.org/abs/2608.24891) | 该论文首次系统研究了量子学习中固定测量预算在训练态数量与每态测量次数之间的最优分配问题，证明了有限测量条件下的免分布泛化界，并推导出与数值最优值相差不到 0.1% 的闭式分配规则，实现 $B^{-1/4}$ 的最坏情况收敛速率。 |
| [^335] | [Fool's Gold: Defensive Deception Against Safety-Removal Attacks on Open-Weight Models](https://arxiv.org/abs/2608.17202) | 该论文提出“愚人金”防御方法，通过训练诱饵响应在安全移除攻击后产生伪造信息，有效阻止开放权重模型被恶意利用。 |
| [^336] | [CHORUS: Coverage-Guided Agentic RL and Policy Consolidation for Testbench Stimulus Generation](https://arxiv.org/abs/2608.10090) | CHORUS是首个将覆盖率引导的智能体强化学习与策略整合相结合的测试平台激励生成框架，通过权重平均与奖励引导蒸馏，其4B模型在CVDP-ECov上达到88.0%的智能体Pass@1，超越DeepSeek-R1（671B）13.5个百分点。 |
| [^337] | [Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations](https://arxiv.org/abs/2608.07385) | 提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。 |
| [^338] | [Cautious Context Steering for Language Model Personalization](https://arxiv.org/abs/2608.05813) | 提出轻量级适配器 CCS，通过谨慎蒸馏学习何时以及以多强强度利用用户上下文来引导冻结语言模型，在个性化偏好对齐上超越了现有无需训练的方法和经过训练的个性化基线。 |
| [^339] | [Relax Within, Balance Across: Load Balancing under Composition Shifts in Vision-Language Mixture-of-Experts](https://arxiv.org/abs/2608.00574) | 该论文揭示了视觉语言混合专家模型中的跨模态负载抵消现象——图像与文本误差在训练比例附近相互抵消从而掩盖推理时的负载失衡，并提出ReBA方法通过模态内放松、模态间平衡的策略来应对组合偏移下的负载均衡问题。 |
| [^340] | [Distilling Reasoning Traces into Advisory Prompts for Software Engineering Tasks](https://arxiv.org/abs/2608.00437) | 受人类通过反思错误并内化规则来学习编程的启发，本文提出将LLM思考模式中避免错误的推理轨迹提炼为可内化的咨询式提示，使软件工程任务无需持续承担推理成本即可获得性能提升。 |
| [^341] | [Tokengeist: Multi-Turn Attribution Tracing in Agentic Conversations](https://arxiv.org/abs/2607.22610) | 该论文提出了多轮上下文归因（MTCA）任务以及Tokengeist框架，通过将有向无环图上的递归遍历作为归因方式来恢复完整依赖路径，解决了现有单次归因方法无法捕捉多轮对话分层依赖结构的问题。 |
| [^342] | [Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization](https://arxiv.org/abs/2607.22583) | 该论文提出了一种硬件感知的多目标结构化剪枝框架，通过两阶段方法联合优化大语言模型的层数、注意力头和MLP维度，以同时最小化延迟和模型大小，从而实现边缘设备上的高效部署。 |
| [^343] | [Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua](https://arxiv.org/abs/2606.30117) | 该论文将基于物理信息神经网络（PINNs）的全息逆问题方法扩展至具有大能级层级和伪真空的强耦合量子场论情形，通过一系列方法学改进克服了近简并态与势能未探测区域等挑战，实现了从边界热力学数据重构体标量势。 |
| [^344] | [Why Trust Your Agent? Empirical Security Gains from TRiSM-Guided Agentic Workflows in Healthcare](https://arxiv.org/abs/2606.28666) | 本文首次将TRiSM安全框架应用于医疗报告生成的智能体工作流，通过大规模实证评估证明安全导向的重新设计可显著降低RAG投毒和数据字段注入等攻击的成功率。 |
| [^345] | [SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation](https://arxiv.org/abs/2606.27786) | SPARK框架将参数化知识控制形式化为潜在参与策略学习问题，通过轻量级参与策略动态调节生成过程中参数化知识的贡献，从而解决RAG中检索上下文与模型内部知识的冲突，提升生成忠实性，同时避免直接编辑模型参数可能带来的能力损害。 |
| [^346] | [Random coloured digraphs defined by a Markov logic network](https://arxiv.org/abs/2606.23715) | 本文系统研究了马尔可夫逻辑网络所定义的概率分布在不同域规模下一阶语句成立概率的行为，并证明在可表达一元性质与二元关系的一阶语言中，多种不同的情形均可由该模型实现。 |
| [^347] | [When Does Depth Matter For In-Context Learning? Adaptive Inference in Deep Transformers](https://arxiv.org/abs/2606.16694) | 本文提出了广义线性注意力Transformer中分布式推理的一般理论，证明多层Transformer可利用“函数向量”内部表示在各层以越来越精细的尺度推断潜在上下文变量，从而阐明了深度堆叠在何种条件下比单层结构具有计算优势。 |
| [^348] | [Rational Sparse Autoencoder](https://arxiv.org/abs/2606.14990) | 提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。 |
| [^349] | [Implicit Neural Representations of Individual Behavior](https://arxiv.org/abs/2606.12200) | 提出Behavioral INR，一种自监督生成模型，将视觉领域的隐式神经表示适配到行为领域，把策略表示为由片段级潜在变量通过FiLM层调制的状态-动作函数，从而在无标签的混合行为数据中实现无监督的策略表示学习与策略身份推断。 |
| [^350] | [Business World Model](https://arxiv.org/abs/2606.10044) | 本文提出了商业世界模型（BWM）的概念与架构，通过以商业语义为中心的公式化方法将状态、动力学和可行行动与商业实体相联系，使世界模型范式得以拓展应用于语义性、组织性和市场驱动的商业环境。 |
| [^351] | [SPACE: Source-free Proxy Anchor Concept Erasure for MLLMs](https://arxiv.org/abs/2606.09868) | SPACE 提出了首个面向多模态大语言模型的无源机器遗忘框架，通过文本引导的代理锚点选择和双约束语义隔离两个阶段，在无需访问目标概念原始数据的情况下实现敏感概念的擦除。 |
| [^352] | [The Montparnasse Algorithm for RNA Design](https://arxiv.org/abs/2606.07562) | Montparnasse是一种基于广义嵌套rollout策略自适应的蒙特卡洛逆RNA折叠搜索算法，通过结构先验、字典序多准则评估、错误碱基对局部修复以及基于错误维持多样性的并行搜索等创新，在Eterna100基准上中位数解决了100个谜题中的96个，超越了DesiR的90个。 |
| [^353] | [DAST: A VLM-LLM Framework for Cross-Interface Anomaly Detection in O-RAN](https://arxiv.org/abs/2606.06261) | 该论文提出DAST框架，通过VLM-LLM-VLM三阶段级联的零样本多智能体流水线，解决O-RAN开放架构下拒绝服务与性能降级攻击难以检测的问题，实现无需标注基线的跨接口异常检测。 |
| [^354] | [Learning Which Evidence-Seeking Decisions Matter: Calibrated Counterfactual Replay for Language Agents](https://arxiv.org/abs/2606.05263) | 提出了PCCC-RL方法，通过前缀条件下的反事实重放、预先信用预测与基于选择概率的校准纠正机制，精准识别语言智能体中哪些证据寻求决策真正决定了最终答案的质量，从而解决稀疏奖励下的信用分配问题。 |
| [^355] | [Supportive Token Revealing for Fast Diffusion Language Model Decoding](https://arxiv.org/abs/2606.04236) | 提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。 |
| [^356] | [VESTA: Visual Exploration with Statistical Tool Agents](https://arxiv.org/abs/2606.00384) | VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。 |
| [^357] | [From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Merging](https://arxiv.org/abs/2606.00357) | 提出了偏好增量聚合（PDA）框架，将多个弱-更弱模型对产生的“弱”偏好信号分别学习为LoRA适配器，并通过几何对齐合并（GAM）进行聚合，从而有效提升强模型（如Qwen3 8B）的性能。 |
| [^358] | [PhoneWorld: Scaling Phone-Use Agent Environments](https://arxiv.org/abs/2605.29486) | PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。 |
| [^359] | [Periodic and Harmonic Periodic RoPE for Length Extrapolation in LLMs](https://arxiv.org/abs/2605.27980) | 该论文提出周期性 RoPE（P-RoPE）与谐波周期性 RoPE（HP-RoPE）两种位置编码机制，通过与滑窗注意力结合并仅复用 W 个旋转角度来缓解大语言模型长上下文外推中的位置耗尽问题，其中 HP-RoPE 借助在窗口内完成整数圈的谐波频率消除了窗口边界的回绕误差。 |
| [^360] | [Channel-wise Vector Quantization: Native 1D Visual Tokenization with Learnable Ordering](https://arxiv.org/abs/2605.26089) | 本文提出通道级向量量化（CVQ），通过量化特征图通道而非图像块，实现具有可学习由粗到细排序的原生一维视觉标记化，并据此构建了通过“下一通道预测”依次生成图像、先勾勒全局结构再细化细粒度细节的通道级自回归（CAR）模型。 |
| [^361] | [Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](https://arxiv.org/abs/2605.21347) | 该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。 |
| [^362] | [AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals](https://arxiv.org/abs/2605.20643) | 本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。 |
| [^363] | [Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models](https://arxiv.org/abs/2605.20158) | 该论文提出了一个基于反事实编辑的因果评估框架，用于验证视觉归因方法是否真正反映了大型视觉语言模型在胸部X光推理中决策所依赖的视觉证据，并发现现有归因方法在该框架下并不可靠。 |
| [^364] | [MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems](https://arxiv.org/abs/2605.18565) | 提出MINTEval基准，通过包含频繁更新信息的长且高度关联的上下文、多样化领域任务和问题类型，系统评估记忆增强智能体在多目标干扰和长时程场景下的记忆能力。 |
| [^365] | [Authors' Response to AI-Based Reviews: A Mixed-Methods Pilot Study](https://arxiv.org/abs/2605.16623) | 该混合方法试点研究通过对三个计算机科学会议作者的问卷调查发现，约九成作者认为AI辅助评审有用并在修改中采纳其建议，AI还能发现人工评审遗漏的问题，但近半数作者对AI反馈的信任度较低，多数报告其意见存在错误或误导。 |
| [^366] | [Spectral Vision Transformer for Efficient Tokenization with Limited Data](https://arxiv.org/abs/2605.12026) | 提出频谱视觉Transformer，将图像标记化为固定基底的频谱系数而非空间图像块，使序列长度与分辨率解耦、可自由选择，从而在数据稀缺场景（尤其是医学影像）下实现高效且性能优越的标记化。 |
| [^367] | [Behavioral Integrity Verification for AI Agent Skills](https://arxiv.org/abs/2605.11770) | 该论文提出行为完整性验证（BIV）框架，通过将确定性代码分析与LLM辅助能力提取相结合，对智能体技能的声明能力与实际能力进行类型化集合比较，并在49,943个OpenClaw技能上发现80%的技能存在描述与实现之间的偏差。 |
| [^368] | [The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting](https://arxiv.org/abs/2605.07671) | 该论文证明决策耦合评分报告中的校准偏差是内生的（福利最优的批准规则必然非仿射），但通过构造一个基于不依赖类型分布的“保留报告”、带单个折点的 Lipschitz 批准规则，可在每个严格恰当评分下完美筛选类型并实现一阶最优，其关键约束是规则斜率而非光滑性。 |
| [^369] | [Structured Role-Aware Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2605.07274) | 该论文提出结构化角色感知策略优化（SRPO），通过将多模态结构化响应分解为感知token与推理token，把序列级的GRPO优势细化为角色感知的token级优势，实现更精确的信用分配，从而确保正确答案有任务相关的视觉证据支撑。 |
| [^370] | [Post-Reasoning: Improving the Performance of Non-Thinking Models at No Cost](https://arxiv.org/abs/2605.06165) | “后推理”让指令微调模型在生成最终答案后再进行答案论证，从而在不增加任何推理延迟或token成本的情况下有效提升模型性能。 |
| [^371] | [Detecting AI-Generated Videos with Spiking Neural Networks](https://arxiv.org/abs/2605.05895) | 该论文发现AI生成视频在像素级帧间残差和语义特征轨迹上均存在时间平滑性特征，并利用脉冲神经网络对这一时间动态证据进行建模，以实现跨生成器的AI视频检测。 |
| [^372] | [Topology-Driven Anti-Entanglement Control for Soft Robots](https://arxiv.org/abs/2605.05236) | 本文提出一种拓扑驱动的多智能体强化学习（TD-MARL）框架，通过共享拓扑状态的集中式学习协调多软体机器人系统，解决了高约束环境下防缠绕控制中可观测性不足与训练不稳定的问题。 |
| [^373] | [Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration](https://arxiv.org/abs/2605.00310) | 该论文提出了GeoSR-Bench基准数据集，通过将土地覆盖分类、生物量估计、变化检测等下游任务集成到评估流程中，突破传统PSNR/SSIM保真度指标的局限，实现对遥感影像超分辨率模型实用价值的更全面评估。 |
| [^374] | [AI-Assisted Hypothesis Generation for Hospital Quality Improvement](https://arxiv.org/abs/2604.20055) | 该论文提出利用大语言模型加速医院质量改进中传统上依赖专家定性方法的假设生成阶段，并主张将问题规范本身也作为可迭代调整的对象，以应对这一“模糊”且探索性的任务。 |
| [^375] | [OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA](https://arxiv.org/abs/2604.12075) | 该研究发布了OpenTME数据集，利用AI病理基础模型从TCGA十种癌症类型的5,268张H&E全切片图像中生成大规模开放的肿瘤微环境定量图谱，每张切片包含超过20,000个定量指标。 |
| [^376] | [Agent^2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training?](https://arxiv.org/abs/2604.10547) | 该论文提出了 Agent² RL-Bench——一个紧凑的诊断基准，通过面向智能体的统一接口和三个层级的六项任务，评估大语言模型智能体能否在固定预算内自主设计、实现、调试并闭环执行强化学习后训练流水线来提升基础模型性能。 |
| [^377] | [Robust LLM Evaluation with Boundedly Bad Judges: A Constrained Maximum Likelihood Approach](https://arxiv.org/abs/2604.03257) | 该论文提出约束最大似然估计（CMLE）方法，通过数据驱动约束显式建模LLM评判者的错误率，以受控偏差换取方差降低，从而在可靠标签有限的情况下实现更稳健的大规模LLM评估。 |
| [^378] | [{\lambda}Split: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy](https://arxiv.org/abs/2603.23647) | 提出了λSplit，一种基于物理信息的自监督深度生成模型，通过分层变分自编码器学习浓度图的条件分布，并利用可微分光谱混合器，实现了对荧光显微镜光谱图像的内容感知解混。 |
| [^379] | [xAppGen: LLM-driven Natural Language Programming for Agile xApp Development](https://arxiv.org/abs/2603.18604) | 该论文提出了首个由大语言模型驱动的自然语言编程框架xAppGen，能在几分钟内自动将用户意图转化为可部署的xApp，实现xApp开发全流程自动化，大幅降低O-RAN应用开发门槛。 |
| [^380] | [Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments](https://arxiv.org/abs/2603.02511) | 该论文提出模块化框架Unveiler，将场景推理与物体选择（通过空间关系编码器）和动作执行相分离，无需真实机器人训练数据即可在杂乱环境中实现顺序操作任务。 |
| [^381] | [AI GameStore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games](https://arxiv.org/abs/2602.17594) | 提出以“人类游戏多元宇宙”——所有由人类为人类设计的游戏——作为可扩展、开放式的机器通用智能评估框架，通过让AI在相同资源条件下与人类玩家比较玩和学玩所有人类游戏的能力，克服传统基准测试狭窄且易饱和的局限。 |
| [^382] | [On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage](https://arxiv.org/abs/2602.12107) | 该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。 |
| [^383] | [When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning](https://arxiv.org/abs/2602.08236) | 该论文提出AVIC自适应测试时框架，通过深入分析何时需要视觉想象以及想象的最优程度，利用世界模型自适应地调控想象过程，从而提升多模态大语言模型视觉空间推理的准确性与效率。 |
| [^384] | [VLRS-Bench: A Vision-Language Reasoning Benchmark for Remote Sensing](https://arxiv.org/abs/2602.07045) | 提出了首个专门针对遥感领域复杂推理的视觉-语言推理基准VLRS-Bench，涵盖认知、决策和预测三大维度的2,000个问答对，实验揭示了当前多模态大语言模型在遥感推理能力上的显著瓶颈。 |
| [^385] | [EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL](https://arxiv.org/abs/2602.04417) | 提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。 |
| [^386] | [The Post-Training Dilemma: Why We Should Rethink the Sequential SFT-RL Paradigm](https://arxiv.org/abs/2601.07389) | 本文从理论上证明SFT与RL在近乎正交的参数子空间上进行优化，导致RL阶段不可避免地损害SFT所获得的能力，且KL惩罚无法修复这一梯度不对齐问题，因此主张重新思考顺序式SFT-RL后训练范式。 |
| [^387] | [How Large Language Models Misrepresent American Climate Opinions](https://arxiv.org/abs/2512.23889) | 研究发现大语言模型在模拟美国气候民意时系统性地压缩观点多样性，将预测拉向人口均值并套用刻板的性别假设，在种族与性别交叉维度上尤其失真。 |
| [^388] | [Variational analysis of determinantal varieties](https://arxiv.org/abs/2511.22613) | 本文提出统一的变分分析框架，推导出各类低秩集合（矩阵、张量、对称矩阵、半正定矩阵等）的一阶与二阶切集显式公式及切集相交规则，并据此给出了非光滑问题与其光滑参数化具有等价二阶稳定点的充要条件。 |
| [^389] | [Predicting Public Health Impacts of Electricity Usage](https://arxiv.org/abs/2511.22031) | 提出了HealthPredictor——一个将电力使用与公共健康结果联系起来的端到端AI模型，通过燃料组合预测、空气质量模拟和货币化健康损害评估三个组件，实现基于健康信息的需求侧电力管理。 |
| [^390] | [Rhythm-TTS: High-Fidelity Shortcut Latent Modeling and Probabilistic Rhythm Generation for Zero-Shot Text-to-Speech](https://arxiv.org/abs/2510.02848) | Rhythm-TTS 提出了一种捷径零样本文本转语音框架，将预测的离散编解码器标记作为语义-声学先验，直接生成高保真的整体连续语音潜变量，并结合概率化节奏生成，从而兼顾离散标记的可因子分解性与连续表示的高保真度。 |
| [^391] | [FlowDrive: moderated flow matching with data balancing for trajectory planning](https://arxiv.org/abs/2509.21961) | FlowDrive通过按轨迹模式重加权来平衡长尾驾驶数据，并提出带有环内微小扰动引导的调节式流匹配规划器，在保持场景一致性的同时增加轨迹多样性，在nuPlan和interPlan基准上取得了学习型规划器中的最先进结果。 |
| [^392] | [LLM4Rail: An LLM-Augmented Personalized Railway Services Platform](https://arxiv.org/abs/2507.23377) | 该论文提出了LLM4Rail个性化铁路服务平台，其核心创新是包含经验存储与检索（ESR）和诊断性观察生成（DOG）两大机制的迭代式ER-QTAO推理框架，并构建了中国铁路餐饮数据集（CRFD）以实现个性化的车上餐饮推荐服务。 |
| [^393] | [TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios](https://arxiv.org/abs/2507.18061) | TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。 |
| [^394] | [A Quasi-experimental Study of Conversational AI-supported Engagement in Large CS Courses](https://arxiv.org/abs/2506.18941) | 本研究通过准实验设计发现，在大型软件工程课堂中使用基于大语言模型的会话式AI工具能够有效提升学生对学习材料的参与度。 |
| [^395] | [Mechanistic Interpretability Needs Philosophy](https://arxiv.org/abs/2506.18852) | 该论文主张机制可解释性研究需要与哲学建立持续的合作关系，借助哲学澄清概念、完善方法并应对认识论与伦理层面的复杂性，从而释放该领域尚未实现的巨大发展潜力。 |
| [^396] | [AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion](https://arxiv.org/abs/2505.22106) | 提出AudioTurbo，将校正扩散集成到预训练扩散式TTA模型中，通过从确定性噪声-样本对学习一阶近似ODE路径，在生成质量和推理速度上均超越基于流匹配的加速方法。 |
| [^397] | [LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems](https://arxiv.org/abs/2505.00240) | 该论文提出了一种基于大语言模型的物联网威胁检测与防御框架，通过在 IoT-23 和 TON_IoT 数据集上微调的检测模型识别异常网络行为，并结合自适应防御策略生成组件，在边缘-云实验环境中实现了 99.75% 的威胁检测准确率和 98.3% 的防御准确率。 |
| [^398] | [CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception](https://arxiv.org/abs/2502.07807) | 该论文提出了一种无需验证最终感知结果、直接在特征层面检测恶意代理的新范式，并发布了首个协同感知恶意代理检测基准数据集CP-GuardBench及鲁棒防御方法CP-Guard+，在大幅降低计算开销的同时有效防御协同感知中的恶意攻击。 |
| [^399] | [FlexTSF: A Flexible Forecasting Model for Time Series with Variable Regularities](https://arxiv.org/abs/2410.23160) | FlexTSF 提出了一种灵活的领域自适应时间序列预测模型，其核心的 IVP Patcher 连续时间分块模块基于初值问题，能够原生处理不均匀采样间隔、可变序列长度和缺失值，从而有效应对不规则时间序列的通用预测挑战。 |
| [^400] | [Ensemble Logic](https://arxiv.org/abs/2408.14443) | 本文提出集合逻辑（EL）这一族模态逻辑，在度量幺半群上统一了离散、稠密和多维设置下的时间与空间推理，并证明了自然数上可满足性的尖锐不可判定性边界。 |

# 详细

[^1]: PlayCraft：基于智能体游戏体验设计引人入胜的游戏

    PlayCraft: Designing Engaging Games from Agent Play Experience

    [https://arxiv.org/abs/2611.26624](https://arxiv.org/abs/2611.26624)

    PlayCraft提出了一个自动化框架，让视觉智能体反复试玩游戏并基于其游玩报告（进展、决策、反思等）自动修改游戏设计，使游戏更有助于玩家精通，从而提升游戏的吸引力。

    

    近期的AI智能体可以生成可玩的游戏，但这些游戏不一定具有吸引力。精通是游戏吸引力的关键来源之一，然而为精通而设计需要大量的人类试玩和调优。我们提出了PlayCraft，一个自动化框架，它使用基于视觉的智能体来玩游戏，并利用其详细的游戏体验来修改游戏，以促进玩家的精通。从初始游戏开始，PlayCraft在游玩和制作之间交替进行。在游玩阶段，玩家智能体观察游戏画面、使用控制器，并在多轮重复的游戏回合中追求游戏目标，同时保留记忆。每回合结束后，它会报告自己的进展、结果、决策和反思。在制作阶段，制作者智能体利用这些报告来修改游戏。它估计玩家能力，预测不同熟练度水平下的表现，并将这些预测与目标学习进程进行比较，从而诊断游戏并弥补差距。

    arXiv:2611.26624v1 Announce Type: new  Abstract: Recent AI agents can generate playable games, but these games are not necessarily engaging. Mastery is a key source of engagement, yet designing for it requires extensive human play and tuning. We propose PlayCraft, an automated framework that uses a vision-based agent to play games and draws on its detailed play experience to revise them for mastery. Starting from an initial game, PlayCraft alternates between playing and crafting. In the playing stage, a player agent watches the game screen, uses the controls, and pursues the game objective over repeated episodes while retaining memory. After each episode, it reports its progress, outcomes, decisions, and reflections. In the crafting stage, a crafter agent uses these reports to revise the game. It estimates player abilities, predicts performance at different proficiency levels, and compares those predictions with a target learning progression. It diagnoses the game and addresses the gap
    
[^2]: 一个先验，两种动力学：将视频扩散Transformer重新用于自监督潜在动作学习

    One Prior, Two Dynamics: Repurposing Video Diffusion Transformers for Self-Supervised Latent Action Learning

    [https://arxiv.org/abs/2611.26617](https://arxiv.org/abs/2611.26617)

    该论文提出O²LaF，通过将同一个预训练视频扩散Transformer分别适配为逆向动力学过渡编码器和前向动力学动作条件生成器，从无标注视频中自监督学习可迁移的潜在动作，并在导航与机器人操作的少样本动作探测中达到新的最先进水平。

    

    视频揭示了世界如何变化，却很少揭示导致这些变化的动作。潜在动作学习从视觉过渡中恢复此类监督信号，但其底层动力学往往需要从零开始学习。我们提出一个问题：预训练的视频生成器能否同时为逆向动力学和前向动力学提供先验？我们提出了O²LaF，它建立在一个简单的原则之上：一个先验，两种动力学。我们从同一个预训练视频DiT初始化两个模型，并使其各自适应不同的角色：一个带有可学习动作查询的前馈式过渡编码器用于逆向动力学，以及一个因果的、以动作为条件的视频生成器用于前向动力学。通过控制信息访问并通过紧凑的瓶颈端到端地优化两种动力学，O²LaF从无标注视频中学到了可迁移的动作。在导航和机器人操作任务上，O²LaF在少样本动作探测中创造了新的最先进水平。

    arXiv:2611.26617v1 Announce Type: cross  Abstract: Videos reveal how the world changes, but rarely the actions that cause those changes. Latent action learning recovers such supervision from visual transitions, yet often learns the underlying dynamics largely from scratch. We ask whether a pretrained video generator can instead provide the prior for both inverse and forward dynamics. We introduce O$^{2}$LaF, built on a simple principle: one prior, two dynamics. We initialize both models from the same pretrained video DiT and adapt them to their distinct roles: a feed-forward transition encoder with learnable action queries for inverse dynamics, and a causal, action-conditioned video generator for forward dynamics. By controlling information access and optimizing both dynamics end-to-end through a compact bottleneck, O$^{2}$LaF learns transferable actions from unlabeled videos. Across navigation and robotic manipulation, O$^{2}$LaF sets a new state of the art in few-shot action probing 
    
[^3]: SoK：评估大语言模型在软件漏洞研究中能力的演进

    SoK: Assessing the Evolution of LLM Capability in Software Vulnerability Research

    [https://arxiv.org/abs/2611.26601](https://arxiv.org/abs/2611.26601)

    本文提出一个借鉴自动驾驶分级的自主性评级框架，系统化梳理了LLM在漏洞研究中的能力证据，发现2026年前智能体已能在无人工干预下发现、验证并修补真实软件漏洞，开源权重模型随后也达到了相当的自主性水平。

    

    基于大语言模型（LLM）的智能体在发现、验证和修补真实软件漏洞方面的能力日益增强，厂商和研究人员报告称这已经在改变漏洞被发现和修复的方式。然而，这些说法散见于论文、厂商报告和披露信息中，鲜有可比性。我们引入了一个框架，借鉴自动驾驶自动化分级模型，在自主性等级尺度上对这些能力进行评级，并通过其正确性、可获得性以及证据强度对每项说法加以限定。我们利用该框架对已发表的关于LLM漏洞研究能力的证据进行了系统化梳理。我们发现，在2026年受限的前沿模型出现之前，智能体已经能够在无需人工干预的情况下发现、验证并修补真实世界软件中的漏洞，而开源权重模型随后在发现和验证方面也达到了相当的自主性水平。随后我们测量了……

    arXiv:2611.26601v1 Announce Type: cross  Abstract: LLM-based agents are becoming increasingly capable at discovering, validating and patching vulnerabilities in real software, and vendors and researchers report that this is already changing how vulnerabilities are found and fixed. These claims, however, are scattered across papers, vendor reports and disclosures and are rarely comparable. We introduce a framework that rates these capabilities on an autonomy scale modeled on the levels of driving automation, and qualifies each claim by its correctness, its availability and the strength of its evidence. We use this framework to systematize the published evidence on LLM capability in vulnerability research. We find that agents discovered, validated, and patched vulnerabilities in real-world software without human intervention before the restricted frontier models of 2026, and that open-weight models subsequently attained comparable autonomy for discovery and validation. We then measure th
    
[^4]: 通过蒸馏神谕教师实现主动特征获取

    Active Feature Acquisition by Distilling an Oracle Teacher

    [https://arxiv.org/abs/2611.26598](https://arxiv.org/abs/2611.26598)

    该论文提出OracleImitator，首次显式构建了一个按候选特征对下游预测器误差影响进行排序的神谕教师，并通过软目标模仿将这一排序蒸馏为可部署的主动特征获取策略，在多模态医疗诊断和AFABench基准上于整个预算范围内匹配或超越现有方法，恢复约90%的可达准确率增益。

    

    获取一个实例的每一个特征往往代价高昂：一次测量可能带来金钱成本、时间延迟或隐私代价。主动特征获取通过学习一个策略来解决这一问题，该策略在预算约束下针对每个实例决定获取哪些特征。现有的AFA方法通过优化一个自行构建的效用目标来训练该策略，而不是蒸馏一个显式的教师。我们显式地构建了这个教师。在训练阶段，标签和未观测的特征值是可获得的，二者共同以闭式形式确定了这个教师：一个神谕，它根据每个候选特征对下游预测器误差的影响来对其进行排序。OracleImitator通过软目标模仿将这一排序蒸馏为一个可部署的策略。在多模态医疗诊断和AFABench基准上，OracleImitator在整个预算范围内匹配或超越了现有AFA方法，恢复了90%的可达准确率增益……

    arXiv:2611.26598v1 Announce Type: cross  Abstract: Acquiring every feature of an instance is often costly: a measurement can carry a monetary price, a time delay, or a privacy cost. Active Feature Acquisition (AFA) addresses this by learning a policy that decides, per instance, which features to acquire under a budget. Existing AFA methods train this policy by optimizing a self-constructed utility target rather than distilling an explicit teacher. We construct this teacher explicitly. At training time the label and the unobserved feature values are available, and together they determine the teacher in closed form: an oracle that ranks every candidate by its effect on the downstream predictor's error. OracleImitator distills this ranking into a deployable policy by soft-target imitation. Across multimodal medical diagnosis and the AFABench benchmark, OracleImitator matches or outperforms existing AFA methods across the budget range, recovering 90% of the attainable accuracy gain at unde
    
[^5]: PGX：面向长时程人形机器人移动操作的物理接地执行框架

    PGX: Physically Grounded Execution for Long-Horizon Humanoid Loco-Manipulation

    [https://arxiv.org/abs/2611.26596](https://arxiv.org/abs/2611.26596)

    PGX框架通过将时间结构与精度控制交由两个无需训练的经典组件处理，并把长时程人形机器人移动操作分解为谓词状态机、VLA技能执行等多层结构，使VLA模型专注于阶段特定的技能动态，从而有效解决长时程移动操作中时间、技能与精度紧密耦合的难题。

    

    机器人视觉-语言-动作（VLA）模型能够执行多种不同的单项操作技能，但在长时程移动操作任务中仍然表现不佳，因为在这类任务中，时间结构、技能与精度紧密耦合于一个单体模型之中。我们提出了物理接地执行，这是一个将时间控制与精度控制卸载到两个经典的无需训练组件的框架，从而使VLA能够专注于特定阶段的技能动态。PGX将长时程移动操作分解为三个层次：（i）基于谓词的状态机控制任务进程，利用物理谓词来限制阶段转换、触发恢复行为，并协调导航与等待。共享的谓词语义将离线演示分割、在线执行和分阶段评估连接起来；（ii）VLA技能在活跃的操作阶段内运行，采用单一模型或分阶段训练的模型专注于与任务相关的动作。

    arXiv:2611.26596v1 Announce Type: cross  Abstract: Robotic vision-language-action (VLA) models can execute diverse individual manipulation skills, yet they still struggle with long-horizon loco-manipulation tasks, where temporal structure, skills, and precision are tightly coupled within a monolithic model. We propose Physically Grounded Execution (PGX), a framework that offloads temporal and precision control to two classical training-free components, allowing the VLA to focus on stage-specific skill dynamics. PGX decomposes long-horizon loco-manipulation into three layers: (i) a predicate-based state machine governs task progression, using physical predicates to gate phase transitions, trigger recovery, and coordinate navigation and waiting. Shared predicate semantics connect offline demonstration segmentation, online execution, and stage-wise evaluation; (ii) VLA skills operate within active manipulation phases, with flat or stage-trained models concentrating on task-relevant action
    
[^6]: 实现可靠性与总体可靠性之辨：相关推理下任意时刻有效的自一致性认证

    Realized versus Population Reliability: Anytime-Valid Self-Consistency Certification under Correlated Reasoning

    [https://arxiv.org/abs/2611.26592](https://arxiv.org/abs/2611.26592)

    该论文发现语言模型推理流之间存在正相关，破坏了自一致性认证所依赖的条件独立假设，使被认证的假设分裂为关于单次抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”（仅在零相关时二者等价），从而从根本上改变了 anytime-valid 证书所真正保证的可靠性含义。

    

    自一致性认证将语言模型的重复采样视为序贯证据累积过程，当任意时刻有效的 e-process 越过阈值时停止，并借助 Ville 不等式保证有限样本下的第一类错误控制。以往构造假设采样的 rollout 是条件独立的。我们证明该假设在实践中并不成立：推理流会因共享推理前缀或跨提示异质性而呈正相关，这不仅改变证书的效率，更改变了所认证的假设本身。通过引入每单元潜变量（随机效应/组内相关结构）对依赖性建模，可得到可交换、等相关的增量，其相关系数为 ρ。相关性将单一零假设分裂为关于所抽取信念的“实现零假设”与关于模型平均行为的“总体零假设”，二者仅在 ρ=0 时重合。我们证明：(i) 标准投注 e-process 仍保持任意……（原文摘要在此处截断）

    arXiv:2611.26592v1 Announce Type: cross  Abstract: Self-consistency certification treats repeated sampling from a language model as sequential evidence accumulation and stops when an anytime-valid e-process crosses a threshold, guaranteeing finite-sample Type I error control by Ville's inequality. Prior constructions assume sampled rollouts are conditionally independent. We show this assumption fails in practice: reasoning streams are positively correlated through shared reasoning prefixes or across-prompt heterogeneity, changing not merely certificate efficiency but the hypothesis certified. Modeling dependence via a per-unit latent state (the random-effects / intraclass-correlation structure) induces exchangeable, equi-correlated increments with correlation $\rho$. Correlation splits the single null into a realized null about the drawn belief and a population null about average model behavior, which coincide only when $\rho=0$. We show: (i) the standard betting e-process remains anyt
    
[^7]: 探索“有意识”的心灵：视觉-语言模型中的选择性访问与全局工作空间

    Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models

    [https://arxiv.org/abs/2611.26590](https://arxiv.org/abs/2611.26590)

    该论文提出多模态概念透镜（MC-Lens），通过将视觉编码器与语言解码器的中间状态映射到共享语义空间，并结合内部概念干预，揭示视觉-语言模型中概念的可及性与选择性访问机制，为全局工作空间理论在AI系统中的验证提供了工具。

    

    视觉-语言模型（VLM）整合视觉与文本信息，但其输出无法揭示内部的概念表征。全局工作空间理论（GWT）提出，信息从并行的大脑加工过程中被选择出来，整合进一个共享的工作空间，并可供专门化的加工过程使用。我们引入多模态概念透镜（MC-Lens）——一种多模态雅可比透镜，用于研究VLM概念处理中的上述功能。其视觉编码器透镜和语言解码器透镜将中间状态映射到由解码器输出词汇表定义的共享语义空间，以比较不同模态间概念的可及性。MC-Lens还支持内部概念干预，以检验被访问的概念在下游计算中所起的作用。（i）可及性与选择性访问：多模态概念信息在中间状态中是可读取的，并且能够根据任务需求被选择性访问。我们将“读出载体”从单一名称

    arXiv:2611.26590v1 Announce Type: new  Abstract: Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated in a shared workspace, and made available to specialised processes. We introduce the Multimodal Concept Lens (MC-Lens), a multimodal Jacobian lens, to study these functions in VLM concept processing. Its vision encoder and language decoder lenses map intermediate states to a shared semantic space defined by the decoder's output vocabulary to compare concept accessibility across modalities. MC-Lens supports internal concept interventions to examine the role of accessed concepts in downstream computation. (i) Accessibility and selective access. Multimodal concept information is readable in intermediate states and selectively accessible as tasks require. We extend "readout carriers" from one name 
    
[^8]: 验证验证器：基于规范的LLM交叉检查，捕获逃过UVM记分板的缺陷

    Verify the Verifier: Specification-Grounded LLM Cross-Checking for Bugs That Escape UVM Scoreboards

    [https://arxiv.org/abs/2611.26588](https://arxiv.org/abs/2611.26588)

    提出首个基于规范、利用LLM交叉检查已通过回归测试追踪记录的框架"Verify the Verifier"，从564个注入缺陷中确认了41个逃过UVM记分板检查的缺陷。

    

    通过的UVM回归测试仅表明设计与参考模型在记分板所检查的内容上一致，并不能表明其符合规范。那些通过所有测试却违反规范的缺陷，被称为“记分板逃逸”，即使面对成熟的测试平台也能存活。我们提出“Verify the Verifier”（验证验证器），据我们所知，这是首个利用基于大语言模型（LLM）的异常交叉检查器，将已通过回归测试的追踪记录与规范进行比对的框架。在注入到四个OpenTitan IP的564个缺陷中，有68个是候选的记分板逃逸缺陷：它们与原始设计不等价、被测试所覆盖，却能通过测试。该框架确认其中41个为记分板逃逸缺陷——相关断言在缺陷设计上失败，并在原始设计上得到验证通过。

    arXiv:2611.26588v1 Announce Type: cross  Abstract: A passing UVM regression shows the design agrees with the reference model on what the scoreboard checks, not that it meets the specification. Bugs that pass every test yet violate the specification, termed scoreboard escapes, survive mature testbenches. We present Verify the Verifier, to our knowledge the first framework whose LLM-based Anomaly Cross-Checker examines a passing regression's traces against the specification. Of 564 bugs inserted into four OpenTitan IPs, 68 are candidate scoreboard escapes: non-equivalent, exercised by the tests, yet passing. The framework confirmed 41 as scoreboard escapes with assertions failing on the buggy design and proven on the original.
    
[^9]: ProtoDiffusion：面向多层图像的逐层令牌生成建模

    ProtoDiffusion: Token-per-Layer Generative Modeling of Multi-Layer Images

    [https://arxiv.org/abs/2611.26586](https://arxiv.org/abs/2611.26586)

    该论文提出ProtoDiffusion——基于ProtoVAE多层变分自编码器的潜空间扩散框架，通过“一个共享空间原型+每层一个候选向量”的紧凑表示（N层仅需HW+N个令牌而非NHW个），在图像或文本条件（可选边界框布局引导）下高效生成多层RGBA图像。

    

    图形设计是RGBA图层的有序组合，将图层分离开来可以实现选择、移动和复用。现有的分层模型使用全画布或区域大小的空间潜变量网格来表示每个图层，因此表示的大小会随图层的累积空间范围而增长。我们提出了ProtoDiffusion，这是一个建立在ProtoVAE之上的潜空间扩散框架；ProtoVAE是一种多层变分自编码器，它将一个设计编码为一个共享的空间原型以及每个图层对应的一个候选向量。每个候选向量对一个共享解码器进行条件化，以重建其对应的RGBA图层。对于N个图层，该表示在针对特定主干进行打包之前包含HW+N个令牌，而每层全网格表示则需要NHW个令牌。ProtoDiffusion在图像或文本条件下，对原型和一组固定的候选槽进行联合去噪，并以边界框作为可选的布局引导。该模型将不需要的槽留空，并确定性地……（原文摘要在此处截断）

    arXiv:2611.26586v1 Announce Type: new  Abstract: Graphic designs are ordered compositions of RGBA layers whose separation enables selection, movement, and reuse. Existing layered models represent each layer with a full-canvas or region-sized spatial latent grid, so representation size grows with the cumulative spatial extent of the layers. We present ProtoDiffusion, a latent-diffusion framework built on ProtoVAE, a multi-layer variational autoencoder that encodes a design into one shared spatial prototype and one candidate vector per layer. Each candidate conditions a shared decoder to reconstruct its corresponding RGBA layer. For $N$ layers, the representation contains $HW+N$ tokens before backbone-specific packing, compared with $NHW$ for full-grid-per-layer representations. ProtoDiffusion jointly denoises the prototype and a fixed set of candidate slots, conditioned on an image or text, with bounding boxes as optional layout guidance. The model leaves unneeded slots empty and determ
    
[^10]: GEOHunter：在真实环境中测绘生成式引擎优化网络

    GEOHunter: Mapping a Generative Engine Optimization Network in the Wild

    [https://arxiv.org/abs/2611.26564](https://arxiv.org/abs/2611.26564)

    该论文提出了GEOHunter，首次在真实环境中测绘了一个横跨三大中国UGC平台的生成式引擎优化（GEO）付费推广网络，识别出1,985个账户和46万余条推广投放，并发现美国与中国的九个生成式引擎中4.5%–20%的可检索引用页面受到该推广内容的污染。

    

    生成式引擎将检索到的网页合成为答案、推荐与排名。生成式引擎优化（GEO）服务通过在用户生成内容（UGC）平台上为付费客户发布推广文章，来出售在这些答案中的曝光度。此类推广在医疗和金融领域引发了担忧，因为误导性建议可能造成危害。为研究这一问题的严重程度，我们测量了三个中国主要UGC平台上的一个此类付费内容运营网络，发现了与该网络相关联的1,985个账户，其中包含466,357个推断出的推广投放。我们发现该网络通过不断扩充的账户池推广多种类别的产品，开展持续数月、跨多平台的协调性营销活动，并随时间不断演变其清单式（listicle）内容。在来自美国和中国的九个生成式引擎中，我们发现4.5%至20%的可检索引用页面属于（该推广网络）……（原文摘要在此处被截断）

    arXiv:2611.26564v1 Announce Type: cross  Abstract: Generative engines synthesize retrieved web pages into answers, recommendations, and rankings. Generative engine optimization (GEO) services sell visibility in these answers by publishing promotional articles on user-generated-content (UGC) platforms for paying clients. Such promotion raises concerns in medical and financial domains, where misleading advice can cause harm. To study the severity of this problem, we measure one such paid-content operation across three major Chinese UGC platforms, uncovering 1,985 accounts associated with the network comprising 466,357 inferred promotional placements. We find that the network promotes products across diverse categories through a growing pool of accounts, conducting coordinated campaigns spanning months across multiple platforms while evolving its listicle content over time. Across nine generative engines from both the US and China, we find that 4.5% to 20% of retrievable cited pages are a
    
[^11]: 基于病理学基础模型与多示例学习的无需染色归一化的巴雷特食管异型增生多机构分级研究

    Multi-institutional grading of Barrett's esophagus dysplasia without stain normalization using a pathology foundation model and multiple instance learning

    [https://arxiv.org/abs/2611.26562](https://arxiv.org/abs/2611.26562)

    该研究构建了基于病理学基础模型与属性感知多示例学习的深度学习系统，在无需染色归一化的条件下，利用四个机构969张全切片实现了巴雷特食管异型增生的高精度自动分级（AUC达0.96，与专家共识高度一致）。

    

    巴雷特食管异型增生的分级存在显著的观察者间差异，尤其是低级别异型增生，这使临床决策变得复杂。我们开展了一项基于基础模型的深度学习异型增生分级系统的回顾性多机构诊断准确性研究，使用来自四个机构的969张H&E染色全切片图像。基于专家标注训练的图像块级分类器识别具有诊断相关性的区域，随后采用属性感知多示例学习进行切片级分类，将结果分为无非典型增生巴雷特食管（NDBE）、低级别异型增生或高级别异型增生。在留出测试集（n = 130）上，宏平均AUC为0.96（95% CI 0.93–0.98）。对于二分类异型增生检测，敏感性为98%（80/82），特异性为81%（39/48），阴性似然比为0.030，与专家共识相比二次加权kappa值为0.83。无高级别异……

    arXiv:2611.26562v1 Announce Type: cross  Abstract: Barrett's esophagus dysplasia grading suffers substantial inter-observer variability, particularly for low-grade dysplasia, complicating clinical decision-making. We conducted a retrospective multi-institutional diagnostic accuracy study of a foundation model-based deep learning system for dysplasia grading, using 969 H\&E-stained whole slide images from four institutions. Patch-level classifiers trained on expert annotations identified diagnostically relevant regions, followed by attribute-aware multiple instance learning for slide-level classification as non-dysplastic Barrett's esophagus (NDBE), low-grade dysplasia, or high-grade dysplasia. On a held-out test set (n = 130), macro-averaged AUC was 0.96 (95\% CI 0.93--0.98). For binary dysplasia detection, sensitivity was 98\% (80/82) and specificity 81\% (39/48), with a negative likelihood ratio of 0.030 and quadratic-weighted \kap{} of 0.83 against expert consensus. No high-grade dy
    
[^12]: MM-SystemOneBias：评估多模态Jev式系统一决策模型中的偏差

    MM-SystemOneBias: Evaluating Bias in Multimodal Jev-Style System One Decision Models

    [https://arxiv.org/abs/2611.26557](https://arxiv.org/abs/2611.26557)

    提出MM-SystemOneBias基准，通过六种配对干预在11个任务场景、5,323个实例上系统评估13个多模态决策模型的偏差，发现决策模型在选项标签、无关信息和评分量表干预下比其骨干模型更稳定。

    

    当多模态决策模型执行动作判定、内容审核或质量评估时，其可靠性取决于在呈现方式变化下保持语义决策的一致性、忽略无关的附加信息，并区分与任务证据一致的信息和与之冲突的信息。现有评估分别在不同设置下研究决策界面偏差和多模态模型行为，缺乏统一协议来在实际任务上比较多模态决策模型与其骨干模型。我们提出MM-SystemOneBias，在来自九个公共数据集、11个任务场景的5,323个原始实例上评估13个模型，采用六种配对干预：选项顺序、选项标签、外部判断、图像-文本描述、无关信息和评分量表。在七组匹配的决策模型-骨干模型对中，决策模型在选项标签、无关信息和评分量表等干预下表现出更强的稳定性。

    arXiv:2611.26557v1 Announce Type: new  Abstract: When multimodal decision models determine actions, moderation outcomes, or quality judgments, their reliability depends on preserving semantic decisions across presentation changes, ignoring irrelevant additions, and distinguishing information consistent with task evidence from information that conflicts with it. Existing evaluations study decision-interface biases and multimodal model behavior in separate settings, leaving no common protocol to compare multimodal decision models with their backbones on practical tasks. We introduce \textbf{MM-SystemOneBias}, which evaluates 13 models on 5,323 original instances from nine public datasets and 11 task scenarios using six paired interventions: option order, option label, external judgment, image--text description, irrelevant information, and rating scale. Across seven matched decision-model--backbone pairs, decision models are more stable under option-label, irrelevant-information, and rati
    
[^13]: 代理式安全事件复现：审计与调查（ROGUE-AI）

    Replication of Agentic Security Events: Audit and Investigation (ROGUE-AI)

    [https://arxiv.org/abs/2611.26553](https://arxiv.org/abs/2611.26553)

    该论文提出ROGUE-AI多智能体框架，通过离线复现被报道的LLM智能体安全事件并构建仅缺少所归咎原因的匹配对照组进行对照实验，在108次复现尝试中89%成功重现所报道的危害，从而验证事件真实性并定位根本原因。

    

    基于大语言模型（LLM）的智能体现在可以代表用户删除文件和发送付款。新闻媒体报道经常描述这些智能体的失败案例（例如，Replit的编程智能体删除了生产数据库），但报道通常只说明出了什么问题，很少对事件进行（不）验证或找出其根本原因。事件目录收集了这些报告，基准测试记录了智能体是否失败，但两者都没有复现事件，也没有检验其报道中可能指出的原因。在本文中，我们提出了ROGUE-AI（代理式安全事件复现：审计与调查），这是一个多智能体框架，弥补了上述空白：它在离线环境中重建每个被报道的事件，并同时构建一个仅缺少所归咎原因的匹配对照组，然后在两者上运行智能体。我们的结果表明，ROGUE-AI在108次复现尝试中能够以89%的成功率重现所报道的危害。基于五个当前GPT、Clau（原文在此截断）

    arXiv:2611.26553v1 Announce Type: cross  Abstract: Agents based on large language models (LLMs) can now delete files and send payments on behalf of their users. News media reports routinely describe failures of these agents (e.g., Replit's coding agent deleting a production database), but a report often only says what went wrong and rarely (in)validates the incident or finds its root cause. Incident catalogs collect these reports, and benchmarks record whether agents fail, but neither replicates an incident nor tests the cause its report might identify. In this paper, we present ROGUE-AI (Replication of Agentic Security Events: Audit and Investigation), a multi-agent framework that closes the above gap and rebuilds each reported incident offline together with a matched control that lacks only the blamed cause, and runs agents on both. Our results suggest that ROGUE-AI is able to reproduce the reported harm in 89% of its 108 attempted replications. Agents built on five current GPT, Clau
    
[^14]: 谎言的几何学：用于检测大语言模型欺骗行为的知识与人格方向

    The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models

    [https://arxiv.org/abs/2611.26545](https://arxiv.org/abs/2611.26545)

    该研究通过向人格方向库中加入“否认已知知识”的角色构建无标签谎言检测方向，发现其在70B规模模型的大多数谎言类型上优于单数据集探针，但在谎言与诚实否认措辞相同的知识类谎言上失效，揭示出人格方向主要依赖表面措辞而非深层欺骗意图。

    

    线性探针是最廉价的大语言模型谎言检测器，但往往难以在不同类型的谎言之间迁移。基于诚实与欺骗性角色扮演的无标签方向在70B规模上、以及在关于知识的谎言（谎言与诚实的否认可能完全相同）场景中尚未得到检验。我们通过加入否认自身已有知识的角色及其诚实对应角色，扩展了一个已发表的人格方向库。在每个模型的中间层，所得的人格方向在Liars' Bench基准上对四个模型中的三个进行谎言排序时，表现优于单数据集探针；但在Qwen上则失败，即便Qwen保持角色扮演的频率远高于其他模型。在Llama-3.3-70B-Instruct上，它优于来自随机人格方向库的方向，但相对于已发表方向的提升处于随机扩展的波动范围之内。在关于知识的谎言上，人格方向主要追踪回复的措辞：在谎言与诚实否认用词相同的场景中，我们的三种主要对比方法在Llama和Qwen上仍处于随机方向的波动范围内。

    arXiv:2611.26545v1 Announce Type: new  Abstract: Linear probes are the cheapest LLM lie detectors but often fail to transfer across kinds of lie. Label-free directions from honest and deceptive role-play are untested at 70B scale and on lies about knowledge, where a lie and an honest denial can be identical. We widen a published persona bank with roles that deny knowledge they have, plus honest counterparts. At each model's middle layer the resulting persona direction ranks lies across Liars' Bench better than single-dataset probes on three of its four models; it fails on Qwen, even when Qwen stays in role far more often. On Llama-3.3-70B-Instruct it beats directions from a random persona bank, but its gain over the published direction is within the range of random widenings. On lies about knowledge, persona directions mainly track a reply's wording: where lies and honest denials share their words, our three main contrasts stay inside the random-direction band on Llama and Qwen and rea
    
[^15]: AURA：音频驱动的流式虚拟形象

    AURA: AUdio-dRiven streaming Avatar

    [https://arxiv.org/abs/2611.26540](https://arxiv.org/abs/2611.26540)

    提出了 AURA——一种音频驱动的流式数字人视频生成模型，通过滚动 KV 缓存实现逐块因果生成，并配合两阶段蒸馏，可在单 GPU 上以 20.7 fps 实时流式生成 576p 说话人视频，适用于低延迟交互式虚拟形象应用。

    

    当前的音频驱动视频扩散模型能够合成逼真的说话人头视频，但通常需要在多秒的视频片段上进行数十步扩散去噪，这使它们不适合语音助手和虚拟化身等流式应用，而这些应用要求低延迟和在线因果生成。AURA 通过滚动 KV 缓存逐块生成视频，只需当前块的音频即可开始生成，而不像双向模型那样需要多秒的音频前瞻。它提供两种规模：AURA 1.3B 可在单块 GPU 上实时流式生成 576p 视频（在 16 Hz 播放速率下达到 20.7 fps），适合交互式虚拟形象应用；AURA 5B 则能以 9.7 fps 流式生成 720p 视频。我们的两阶段蒸馏流水线首先通过训练因果学生模型以更少的扩散步骤和逐块滑动窗口来匹配教师的轨迹，从而将学生与教师对齐……（原文摘要在此处截断）

    arXiv:2611.26540v1 Announce Type: cross  Abstract: Current audio-driven video diffusion models can synthesize realistic talking-head videos but generally require dozens of diffusion denoising steps over multi-second video chunks, making them unsuitable for streaming applications such as voice assistants and virtual avatars, where low-latency and online causal generation are required properties. AURA generates video chunk by chunk with a rolling KV cache, requiring only the audio of the current chunk to start generating, rather than a multi-second audio look-ahead like bidirectional models. It comes in two sizes: AURA 1.3B streams 576p video in real time on a single GPU (20.7 fps for a 16 Hz playback rate), making it suited to interactive avatar applications, and AURA 5B streams 720p video at 9.7 fps. Our two-stage distillation pipeline first aligns the causal student with the teacher, by training it to match the teacher's trajectories with fewer diffusion steps and a chunk-wise sliding
    
[^16]: LLM“杰利蝾螈”式操纵：在大语言模型审计中分离交互效应与加性效应

    LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits

    [https://arxiv.org/abs/2611.26538](https://arxiv.org/abs/2611.26538)

    该论文揭示了现有交叉性偏差审计判据会把独立属性的加性效应误报为交互偏差（平均每个数据集误报66.1个子群体、真实交互检出率仅32%），并提出ADDIS方法，通过在所有属性上拟合加性模型建立基线，从而真正将交互效应与加性效应分离，准确审计交叉性子群体偏差。

    

    交叉性偏差是指模型对同时拥有多个边缘化身份的人的对待方式，不同于其各个单独身份所能预测的水平。对这类交叉性子群体的审计依然充满挑战：子群体的数量呈平方级增长，且对于“独立属性本应预测出什么”缺乏公认的基线。在没有这一基线的情况下，许多现有审计只要发现子群体的表现超过任一单独属性就将其标记出来，而属性独立作用时本就满足这一条件。我们在150个不含任何交互作用的合成数据集上检验了这一标准判据：在使用洗牌零假设时，该判据平均每个数据集误报66.1个虚假子群体，且在每个数据集中至少误报一个。当我们在这些数据集中植入已知大小的交互作用后，即便使用修复后的零假设，该判据也只能恢复其中的32%。我们提出了ADDIS（Additive Intersectional Screen，加性交叉性筛选），它在所有属性上拟合一个加性模型，并据此对每个子群体进行评分……

    arXiv:2611.26538v1 Announce Type: new  Abstract: Intersectional bias arises when a model treats people who hold several marginalised identities at once differently from what their separate identities would predict. Auditing such intersectional subgroups remains a challenge: their number grows quadratically, and there is no agreed baseline for what independent attributes would predict. Without such a baseline, many existing audits flag a subgroup when it exceeds each attribute alone, a condition that attributes acting independently already satisfy. We test this standard criterion on 150 synthetic datasets built with no interaction. With the shuffle null, it flags 66.1 false subgroups per dataset on average, and at least one in every dataset. When we plant interactions of known size into these datasets, a repaired null lets the criterion recover only 32% of them. We introduce ADDIS (Additive Intersectional Screen), which fits an additive model over all attributes, scores each subgroup by
    
[^17]: 面向集群规模排序模型的成本感知智能体特征优化系统

    Cost-Aware Agentic Feature Optimization System for Fleet-Scale Ranking Models

    [https://arxiv.org/abs/2611.26531](https://arxiv.org/abs/2611.26531)

    该论文提出了一个端到端的LLM智能体系统，在跨模型共享与逐模型设定的资源约束下，为大规模排序模型集群自动化地完成假设生成、交叉特征生成与筛选、多方法重要性排序融合以及异步试验调优，实现成本感知的跨模型特征优化。

    

    大型网络平台运行着大规模的排序模型集群，每个模型都从包含数万个特征的共享特征池中选取特征。我们提出了一种端到端的智能体系统，在跨模型共享的资源约束和逐模型设定的资源约束（分别为特征管线成本与服务吞吐量QPS）下，对整个模型集群的特征进行优化。在该系统中，LLM智能体负责提出假设，确定性工具负责执行实验并验证假设。对于每个模型，LLM从组合式候选空间中生成并初筛交叉特征，随后采用多种重要性方法对所有特征进行排序。特征融合将上述多个排序混合为单一选择，对于多任务模型则利用跨任务的帕累托前沿。接着，LLM研究员智能体在资源上限约束下通过10至25次异步试验对特征选择进行调优，并由修剪步骤确保管线成本保持在预算之内。在生产模型上，该系统……（原文摘要在此处截断）

    arXiv:2611.26531v1 Announce Type: cross  Abstract: Large Web platforms operate large fleets of ranking models, each selecting features from a shared pool of tens of thousands. We present an end-to-end agentic system that optimizes features across the fleet under resource constraints shared across models and those set per model, such as feature pipeline cost and serving throughput (QPS), respectively. In this system, LLM agents propose hypotheses, and deterministic tools execute experiments and verify them. For each model, an LLM generates and shortlists cross features from a combinatorial candidate space, and all features are then ranked by several importance methods. Feature blending mixes these rankings into a single selection, using Pareto fronts across tasks for multi-task models. An LLM Researcher agent then tunes the feature selection in 10--25 asynchronous trials under resource caps, and a trimming step keeps the pipeline cost within budget. Across production models, the system 
    
[^18]: 回溯智能体应用的安全事件

    Backtracking Agent Applications Security Incidents

    [https://arxiv.org/abs/2611.26525](https://arxiv.org/abs/2611.26525)

    该论文提出了Origin回溯工具，通过弥合智能体轨迹日志与应用代码执行层之间的语义鸿沟，让分析人员仅需提交少量查询即可自动识别智能体应用安全事件的入侵入口点。

    

    智能体应用正在被越来越广泛地部署，然而针对智能体的特定攻击对它们构成了真实的威胁。当发生安全事件时，组织需要回溯以识别入侵入口点（CEP，compromise entry point）、修复损害并防止其再次发生。可用于回溯的主要工件是记录智能体功能的轨迹日志和源代码；而应用特定的函数调用日志往往相当有限。然而，轨迹日志处于智能体功能抽象层，而CEP候选则是应用代码执行层的函数调用。由于这种语义鸿沟，分析人员需要手动检查每个CEP候选的相关代码，以推断其对应的智能体功能。在本工作中，我们提出了Origin，这是一个回溯工具，允许分析人员提交少量查询来识别现有工具无法提供的CEP。我们的方法面临两个挑战：（1）定义……（摘要在此处截断）

    arXiv:2611.26525v1 Announce Type: cross  Abstract: Agent applications are increasingly deployed, yet agent-specific attacks pose real threats to them. Upon a security incident, organizations backtrack to identify the CEP (compromise entry point), remediate the damage, and prevent its recurrence. The main artifacts available for backtracking are trajectory logs, which record agent functionalities, and source code; app(lication)-specific function call logs are often fairly limited. However, trajectory logs are at the agent functionality abstraction layer, while the CEP candidates are function calls at the app code execution layer. Due to this semantic gap, the analyst needs to manually inspect the relevant code for each CEP candidate to infer its corresponding agent functionality. In this work, we present Origin, a backtracking tool for the analyst to submit a small number of queries to identify CEPs that existing tools cannot provide. Our approach poses two challenges: (1) define the in
    
[^19]: GLiDE：从推理教师模型蒸馏得到的快速校准决策模型

    GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers

    [https://arxiv.org/abs/2611.26509](https://arxiv.org/abs/2611.26509)

    GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。

    

    决策模型通过在单次前向传播中对候选答案进行评分，实现快速且具备置信度感知的决策，而无需生成文本或中间推理过程。然而，推理能够显著提升决策质量，这引发了一个问题：能否在不付出生成推理的代价的情况下保留推理带来的收益。我们通过 GLiDE 来研究这一问题，GLiDE 是一系列参数规模从 0.4B 到 27B 的决策模型。GLiDE 基于 Qwen 混合语言模型构建，通过移除全注意力层中的因果掩码，实现了上下文、问题和候选答案之间的双向交互。为了将推理能力迁移到单次推理中，我们将来自多个教师模型（包括一个推理模型）的软目标蒸馏与在多样化决策任务上的渐进式后训练相结合。我们还进一步探索了无参数的循环深度，以在不生成 token 的情况下增加推理时的计算量。

    arXiv:2611.26509v1 Announce Type: new  Abstract: Decision models enable fast, confidence-aware decisions by scoring candidate answers in a single forward pass, without generating text or intermediate reasoning. Yet reasoning can substantially improve decision quality, raising the question of whether its benefits can be retained without the cost of generating it. We investigate this question with \textbf{GLiDE}, a family of decision models spanning 0.4B to 27B parameters.   Building on Qwen hybrid language models, GLiDE enables bidirectional interactions between the context, question, and candidate answers by removing causal masking from full-attention layers. To transfer reasoning capabilities into single-pass inference, we combine soft-target distillation from multiple teachers, including a reasoning model, with progressive post-training on diverse decision tasks. We further explore parameter-free recurrent depth to increase inference-time computation without generating tokens.   Acro
    
[^20]: FLUID：灵活的、有证据支撑的情境化任务指导

    FLUID: Flexible, Evidence-Grounded Situated Task Guidance

    [https://arxiv.org/abs/2611.26491](https://arxiv.org/abs/2611.26491)

    FLUID提出一个统一框架，将任务指导干预与证据、必要性和时机灵活性相绑定，用于评估多模态实时任务指导中的灵活交互并构建智能体，该智能体在正确性、精确性、及时性和延迟之间取得了最佳整体平衡。

    

    实时任务指导本质上是灵活的：当智能体引导用户完成任务时，多种回应内容和时机都可能是合适的，有时最好的行动甚至是保持沉默。现有基准测试往往忽视了这种灵活性，将合成的对话视为“说什么”和“何时说”的固定目标。我们提出了FLUID（灵活的、有证据支撑的情境化任务指导），这是一个用于建模、评估和实现多模态任务指导中灵活交互的统一框架。利用来自WTaG和HoloAssist的自然发生的人际指导数据，FLUID将干预行为与其支撑证据、必要性和时机灵活性相绑定，并利用这种表示方法既评估灵活指导的质量，又构建实时任务指导智能体。在所评估的系统中，我们的智能体在回应正确性、干预精确性、指导及时性和响应延迟方面实现了最强的整体平衡。

    arXiv:2611.26491v1 Announce Type: new  Abstract: Real-time task guidance is inherently flexible: when an agent guides a user through a task, multiple responses and timings can be appropriate, and sometimes the best action is to remain silent. Existing benchmarks often miss this flexibility by treating synthesized dialogues as fixed targets for what to say and when to say it. We introduce FLUID (Flexible, Evidence-Grounded Situated Task Guidance), a unified framework for modeling, evaluating, and enabling flexible interaction in multimodal task guidance. Using naturally occurring human-human guidance from WTaG and HoloAssist, FLUID grounds interventions in their supporting evidence, necessity, and timing flexibility, and uses this representation both to evaluate flexible guidance and to build a real-time agent for task guidance. Our agent achieves the strongest overall balance of response correctness, intervention precision, timely guidance, and response latency among evaluated systems,
    
[^21]: 一种寻找平坦极小值的新算法方法：临界阻尼Polyak步长

    A new algorithmic approach for finding flat minima: critically damped Polyak stepsizes

    [https://arxiv.org/abs/2611.26475](https://arxiv.org/abs/2611.26475)

    本文提出临界阻尼Polyak（CDP）自适应步长——将经典Polyak规则乘以4并加温和阻尼——使梯度下降偏向更平坦的极小值，并首次为局部二次目标函数建立了锐度与损失下降的时间一致理论保证。

    

    我们提出了一种新的自适应步长规则，它在理论和实践上都使梯度下降偏向更平坦的极小值，并首次为一大类局部二次目标函数建立了关于锐度和损失下降的时间一致保证。我们的临界阻尼Polyak（CDP）步长将经典的Polyak规则乘以 $4$ 和一个温和的阻尼因子。因子 $4$ 至关重要，因为它在主导阶上产生围绕极小值流形 $\mathcal{M}$ 的等幅振荡。而阻尼因子则可证明地驱动迭代收敛到 $\mathcal{M}$。因此，CDP通过使步长自适应于局部曲率，在稳定性的边缘运行。一旦损失变得足够小，其投影迭代在主导阶上沿着 $\mathcal{M}$ 执行关于 $\log{\lambda_1(\nabla^2 \mathcal{L})}$ 的梯度下降，这与归一化梯度下降的平坦性偏好相呼应（Arora et al., 2022）。与归一化梯度下降不同，CDP的步长会随……（摘要此处截断）而收缩。

    arXiv:2611.26475v1 Announce Type: cross  Abstract: We present a new adaptive stepsize rule that biases gradient descent toward flatter minima in both theory and practice, and establish the first uniform-in-time guarantees for both sharpness and loss reduction for a broad class of locally quadratic objectives. Our critically damped Polyak (CDP) stepsizes multiply the classical Polyak rule by $4$ and a mild damping factor. The factor $4$ is critical in that it produces equioscillations around the manifold of minimizers $\mathcal{M}$ to leading order. Damping provably drives convergence to $\mathcal{M}$. CDP thus operates at the edge of stability by adapting its stepsizes to local curvature. Once the loss is small, its projected iterates follow gradient descent on $\log{\lambda_1(\nabla^2 \mathcal{L})}$ along $\mathcal{M}$ to leading order, paralleling the flatness bias of normalized gradient descent (Arora et al., 2022). Unlike normalized gradient descent, CDP step lengths shrink with th
    
[^22]: 用于定位球的蒙特卡洛传球搜索

    Monte Carlo Pass Search for Setpieces

    [https://arxiv.org/abs/2611.26464](https://arxiv.org/abs/2611.26464)

    该论文提出蒙特卡洛传球搜索（MCPS），借助世界模型对定位球的反事实传球方案和进攻站位进行大规模推演与价值评估，发现意甲比赛中最佳可用传球比实际踢出的传球多出0.0085 xG（提升28%）。

    

    定位球贡献了约三分之一的非点球预期进球（xG），然而它们通常只根据随后实际发生的情况来评估。我们提出了用于定位球的蒙特卡洛传球搜索（MCPS），该方法将角球或高吊任意球视为由合理执行方式和合理后续发展构成的分布，而非单一已实现的事件。从开球前的状态出发，MCPS提出反事实的传球方案和反事实的站位布置（即开球时进攻球员的站位），在包含三维球体物理、全部22名球员的生成模型、守门员出击或留守决策以及球员触球模型的世界模型中对每种方案进行多次推演，并用价值模型为每一种可能的未来打分。在来自2025/26赛季意甲30场比赛的249次传球上（所有模型仅应用于其从未见过的比赛），最佳可用传球的价值比实际踢出的传球高出0.0085 xG（提升28%），主要原因在于它能在守门员之前将球送到进攻球员脚下。通过搜索进攻球员的站位……（原文摘要在此处被截断）

    arXiv:2611.26464v1 Announce Type: new  Abstract: Set pieces produce about a third of non-penalty expected goals (xG), yet they are usually evaluated by what happened next. We present Monte Carlo Pass Search (MCPS) for set pieces, which treats a corner or a lofted free kick as a distribution over plausible executions and plausible futures instead of a single realised event. From the pre-kick state, MCPS proposes counterfactual deliveries and counterfactual set-ups (where the attackers stand at the kick), plays each out many times in a world model (3D ball physics, a generative model of all 22 players, the keeper's come-or-stay decision and a Player-to-Touch model) and scores every future with a value model. On 249 deliveries from 30 Serie A 2025/26 matches, with every model applied only to matches it never saw, the best available delivery is worth 0.0085 xG more than the one kicked (+28 %), mainly because it reaches an attacker before the keeper. Searching where the attackers stand as w
    
[^23]: LLM内部的CPU：将精确电路零梯度编译进预训练权重

    A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights

    [https://arxiv.org/abs/2611.26450](https://arxiv.org/abs/2611.26450)

    提出了一种零梯度、无需训练样本的闭式电路嫁接方法，可将精确的确定性计算（如多位数算术）直接编译进预训练语言模型权重，同时保持其自然语言能力。

    

    预训练语言模型（LM）在没有梯度微调或外部工具解释器的情况下，在精确多位数算术上仍然不可靠。我们提出了闭式、零梯度的电路嫁接原语，可以将确定性计算直接编译进预训练语言模型的权重中，使用零训练样本和零运行时钩子，同时限制每层归一化漂移并保持自然语言能力。我们的构建结合了三种解析机制：（1）在无偏置的SwiGLU/GeGLU层上进行两层Mellin-GLU数值积分，将未知的逐token RMSNorm缩放转换为沿对数间隔门控网格的平移（>250倍动态范围），并具有IEEE 754误差界；（2）通过带符号的Chebyshev不定RoPE自下沉注意力和无记忆的SiLU反对称GatedDeltaNet位移实现跨token缩放消除；（3）采用√(d/(d+K))范数补偿的轴对齐宽度扩展（d → d+K），Base-1...

    arXiv:2611.26450v1 Announce Type: new  Abstract: Pretrained language models (LMs) remain unreliable on exact multi-digit arithmetic without gradient fine-tuning or external tool interpreters. We introduce closed-form, zero-gradient circuit-grafting primitives that compile deterministic computation directly into pretrained LM weights using zero training examples and zero runtime hooks, while bounding per-layer normalization drift and preserving natural-language (NL) competence. Our construction combines three analytical mechanisms: (1) two-layer Mellin-GLU quadrature over bias-free SwiGLU/GeGLU layers, turning unknown per-token RMSNorm scaling into translation along a log-spaced gate grid ($>250\times$ dynamic range) with IEEE 754 error bounds; (2) cross-token scale cancellation via signed Chebyshev Indefinite-RoPE self-sink attention and memoryless SiLU-antisymmetric GatedDeltaNet shifts; and (3) axis-aligned width extension ($d \to d+K$) with $\sqrt{d/(d+K)}$ norm compensation, Base-1
    
[^24]: Mind2Act World：面向机器人操作中具身心智、物理动作与心智-动作组合能力的诊断性基准

    Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation

    [https://arxiv.org/abs/2611.26444](https://arxiv.org/abs/2611.26444)

    Mind2Act World是一个包含180种配置的诊断性基准，通过Mind、Act和Mind2Act三个视角分别评估具身智能体的认知推理、物理执行及二者在闭环中的组合能力，从而精确定位失败源自推理、执行还是其组合。

    

    具身智能旨在让机器人以类人的适应性和可靠性执行复杂的物理任务，这需要三种互补的能力：基于历史信息的认知推理、精确且及时的物理执行，以及通过环境反馈将推理与执行相结合的组合能力。然而，现有基准要么在同一任务中同时评估推理与执行，要么分别评估二者却不测试其组合，因此难以揭示一次失败究竟源自推理、执行还是二者的组合。我们提出了Mind2Act World，这是一个包含来自五个真实世界应用领域的30个仿真操作任务的诊断性基准。每个任务具有三个难度等级和两种场景设置，共产生180种配置。这些任务构成三个视角：Mind测试基于过去观测的决策能力，Act测试精确且及时的执行能力，而Mind2Act则在闭环中将二者相结合。

    arXiv:2611.26444v1 Announce Type: cross  Abstract: Embodied intelligence aims to let robots perform complex physical tasks with human-like adaptability and reliability, which requires three complementary capabilities: history-based cognitive reasoning, precise and timely physical execution, and the composition of reasoning and execution through environmental feedback. Yet existing benchmarks either evaluate reasoning and execution within the same tasks or assess them separately without their composition, so they reveal little about whether a failure comes from reasoning, execution, or their composition. We introduce Mind2Act World, a diagnostic benchmark of 30 simulated manipulation tasks from five real-world application domains. Each task has three difficulty levels and two scene settings (Clean and Random), yielding 180 configurations. The tasks form three views: Mind tests decisions based on past observations, Act tests precise and timely execution, and Mind2Act combines both in clo
    
[^25]: Asuka-FM：面向消费级硬件上高效文本到图像预训练的系统-架构协同设计框架

    Asuka-FM: A System-Architectural Co-Design Framework for Efficient Text-to-Image Pretraining on Consumer Hardware

    [https://arxiv.org/abs/2611.26437](https://arxiv.org/abs/2611.26437)

    该论文提出系统-架构协同设计框架Asuka-FM，证明约2.06亿参数的双流扩散Transformer可在单张消费级GPU上以约160 GPU小时、不足55美元的成本从头完成有竞争力的文本到图像模型预训练，并解决了注意力开销、JIT图中断和VAE吞吐量三大瓶颈。

    

    前沿的文本到图像（T2I）基础模型主要依赖数十亿参数的扩散Transformer（DiT），并在大规模计算集群上进行预训练，这为学术实验室设置了可及性障碍。当在单个消费级GPU（例如配备24GB显存的NVIDIA RTX 4090）上从头训练DiT时，实践者会遇到三个主要瓶颈：二次方复杂度注意力的序列开销、由可变长宽比分桶引发的即时编译（JIT）图中断，以及自动编码器的吞吐量瓶颈——即被冻结的变分自动编码器（VAE）操作主导了每步延迟。在本工作中，我们提出了Asuka-FM，一个端到端的系统与架构框架，证明了具有竞争力的双流DiT（约2.06亿参数）可以在单个消费级GPU上从头完成预训练，仅需约160 GPU小时，成本低于55美元。Asuka-FM集成了四项核心设计：(i) 无AdaLN的双流Di……

    arXiv:2611.26437v1 Announce Type: cross  Abstract: Frontier text-to-image (T2I) foundation models predominantly rely on multi-billion parameter Diffusion Transformers (DiTs) pretrained across massive compute clusters, creating an accessibility barrier for academic laboratories. When training DiTs from scratch on single consumer GPUs (e.g., an NVIDIA RTX 4090 with 24\,GB VRAM), practitioners encounter three primary bottlenecks: quadratic attention sequence costs, Just-In-Time (JIT) compilation graph breaks induced by variable aspect-ratio bucketing, and an autoencoder throughput wall where frozen Variational Autoencoder (VAE) operations dominate step latency. In this work, we introduce \textbf{Asuka-FM}, an end-to-end system and architectural framework demonstrating that competitive dual-stream DiTs ($\sim$206M parameters) can be pretrained from scratch on a single consumer GPU in $\sim$160 GPU hours for under \$55. Asuka-FM integrates four core designs: (i) an AdaLN-free dual-stream Di
    
[^26]: Seer：利用共享决策模型引导小型深度研究智能体

    Seer: Guiding Small Deep-Research Agents with Shared Decision Models

    [https://arxiv.org/abs/2611.26423](https://arxiv.org/abs/2611.26423)

    Seer是一个推理时引导框架，通过从共享的真实研究轨迹中学习工具响应模拟器和价值估计器，为资源受限设备上的小型深度研究智能体提供决策引导，避免了设备端训练和重复适配的成本。

    

    深度研究智能体需要反复决定搜索什么、阅读什么以及验证什么，通常依赖大型云端托管语言模型来进行动作提议。然而，出于隐私和本地控制的考虑，用户希望在个人设备上运行智能体，而设备上有限的内存和算力更适合小型模型，并使设备端训练变得困难。同时，推理时引导面临三大障碍：通过真实执行来探索替代方案会增加工具调用次数；基于提示的自我评估受限于智能体自身的判断水平；而针对特定模型的引导则面临重复适配的成本。我们提出Seer，一个推理时框架，它提供本地化引导，将智能体的动作提议与学习到的后果预测和价值估计相分离。基于共享的真实研究轨迹语料库，我们训练了两个模型：一个是在观测到的调用—响应对上训练的工具响应模拟器，另一个是在以折扣终末正确性标注的历史记录上训练的价值估计器。

    arXiv:2611.26423v1 Announce Type: new  Abstract: Deep-research agents repeatedly decide what to search, read, and verify, often using large cloud-hosted language models for action proposal. However, privacy and local control motivate running agents on users' devices, where limited memory and compute favor smaller models and complicate on-device training. Yet inference-time guidance faces three obstacles: exploring alternatives through real execution adds tool calls, prompted self-evaluation is bounded by the agent's judgment, and model-specific guidance risks repeated adaptation costs. We introduce Seer, an inference-time framework providing local guidance that separates the agent's action proposal from learned consequence prediction and value estimation. From a shared corpus of real research trajectories, we train two models: a tool-response simulator on observed call--response pairs and a value estimator on histories labeled with discounted terminal correctness. The simulator and est
    
[^27]: 基于大语言模型推理轨迹分析的可解释Q矩阵构建方法

    Reasoning-Trace Analysis Using Large Language Models for Interpretable Q-Matrix Construction

    [https://arxiv.org/abs/2611.26422](https://arxiv.org/abs/2611.26422)

    该论文提出推理轨迹分析（RTA）框架，通过分析大语言模型思维链推理中带有属性标注的推理步骤，使Q矩阵构建中每个题目-属性分配的依据变得可解释、可审查，从而弥补以往仅评估最终矩阵而忽视分配基础的不透明性问题。

    

    arXiv:2611.26422v1 公告类型：新论文 摘要：大语言模型（LLM）能够为诊断分类模型生成接近专家规范的Q矩阵，但以往的评估仅关注最终生成的矩阵，未考察单个题目-属性分配的依据。这种不透明性限制了人工智能在测量领域的负责任使用。我们提出了推理轨迹分析（RTA）框架：大语言模型通过思维链（CoT）提示生成带有属性标注的推理步骤，然后对这些步骤进行分析，以重构、比较和审查每个题目-属性分配。RTA旨在作为专家审查和实证验证的补充手段。在一项使用扰动分数减法题目的模拟研究中，四个大语言模型成功恢复了题目所需的属性，并且没有为题目分配其不需要的属性。在一项包含11道分数减法题目、536名被试的作答数据、四个大语言模型、最多三种温度设置以及每种条件下100次重复实验的实证研究中，CoT-de（原文摘要在此处截断）

    arXiv:2611.26422v1 Announce Type: new  Abstract: Large language models (LLMs) can generate Q-matrices for diagnostic classification models that approximate expert specifications, but evaluations have focused on the final matrices, leaving the basis of individual item-attribute assignments unexamined. This opacity limits the responsible use of AI in measurement. We propose Reasoning-Trace Analysis (RTA), a framework in which an LLM generates attribute-labeled reasoning steps through chain-of-thought (CoT) prompting and these steps are analyzed to reconstruct, compare, and inspect each assignment. RTA is intended to supplement expert review and empirical validation. In a simulation study with perturbed fraction subtraction items, four LLMs recovered the required attributes and withheld attributes the items did not require. In an empirical study with 11 fraction subtraction items, responses from 536 examinees, four LLMs, up to three temperatures, and 100 replications per condition, CoT-de
    
[^28]: Awaken-HI：利用全身人类经验扩展人形智能

    Awaken-HI: Scaling Humanoid Intelligence with Whole-Body Human Experience

    [https://arxiv.org/abs/2611.26420](https://arxiv.org/abs/2611.26420)

    该论文构建了包含5万小时第一人称视频与同步全身人体运动、覆盖100余种技能的大规模数据集Awaken-Human，并提出视觉-语言-动作模型Awaken-HI，通过共享动作专家以人类轨迹直接监督机器人动作学习，实现了移动-操作技能从人类到人形机器人的高效迁移。

    

    人形机器人技术的核心目标是让机器人在家庭和工作场所中执行人们的日常工作。现有的自我中心视角视频数据集涵盖了许多这类活动，但缺乏用于大规模学习移动-操作技能的同步全身运动数据。我们构建了Awaken-Human数据集，将约50,000小时的第一人称立体视频与同步的全身人体运动配对，涵盖超过100种技能。在此基础上，我们还推出了Awaken-HI（Humanoid Intelligence），一个面向人形机器人移动-操作的视觉-语言-动作模型，它通过共享的动作专家联合预测原始人体运动和重定向后的机器人动作。我们证明，当人类轨迹直接监督共享动作专家时，移动-操作技能能够更有效地从人类迁移到人形机器人。增加人类预训练数据能够按照经验缩放规律降低留出集上的机器人动作预测误差。

    arXiv:2611.26420v1 Announce Type: cross  Abstract: A central goal of humanoid robotics is for robots to perform the everyday work people do in homes and workplaces. Existing egocentric video datasets cover many of these activities but provide little synchronized whole-body motion data for learning loco-manipulation at scale. We build Awaken-Human, which pairs approximately 50,000 hours of egocentric stereo video with synchronized whole-body human motion and covers more than 100 skills. Building on this, we also introduce Awaken-HI (Humanoid Intelligence), a vision-language-action model for humanoid loco-manipulation that jointly predicts original human motion and retargeted robot actions through a shared action expert. We show that loco-manipulation skills transfer more effectively from people to humanoids when human trajectories directly supervise the shared action expert. Increasing human pretraining data reduces held-out robot action prediction error according to an empirical scalin
    
[^29]: ZeroVLN：无需对VLM大脑进行监督微调的学习导航方法

    ZeroVLN: Learning to Navigate without Supervised Fine-Tuning of the VLM Brain

    [https://arxiv.org/abs/2611.26414](https://arxiv.org/abs/2611.26414)

    ZeroVLN提出一种快慢结合的导航框架，无需对VLM大脑进行任何导航领域监督微调，而是通过语义工具调用接口让大脑以自然语言传达移动决策，并借助闭环强化学习联合优化大脑决策与动作专家执行，在保持VLM通用视觉能力的同时实现有效导航。

    

    代表性的视觉-语言导航（VLN）方法通常将视觉-语言模型（VLM）大脑与一个可学习的动作头相结合，并在导航领域数据上对VLM大脑进行微调，以预测几何目标。然而，这种适配可能会大幅改变基础VLM的权重，从而损害其通用视觉能力。我们提出了ZeroVLN，这是一个快慢结合（slow-fast）的导航框架，能够在对VLM大脑进行零导航领域监督微调（SFT）的情况下实现有效导航。我们不是教大脑去预测几何目标，而是为其配备一个语义工具调用接口，使其能够以自然语言向动作专家传达移动决策。我们首先适配动作专家以执行这些移动请求，然后通过闭环强化学习（RL）联合改进大脑的移动决策和动作专家的连续执行。在相同的导航回合数据上训练，ZeroVLN大幅超越了代表性方法（原文在此处截断）。

    arXiv:2611.26414v1 Announce Type: new  Abstract: Representative vision-language navigation (VLN) methods typically couple a vision-language model (VLM) brain with a learnable action head and fine-tune the VLM brain on navigation-domain data to predict geometric goals. However, such adaptation can substantially alter base VLM weights and degrade general visual capabilities. We introduce ZeroVLN, a slow-fast navigation framework that achieves effective navigation with zero navigation-domain SFT of the VLM brain. Rather than teaching the brain to predict geometric goals, we equip it with a semantic tool-calling interface for conveying movement decisions to an action expert in natural language. We first adapt the action expert to carry out these movement requests, then jointly improve the brain's movement decisions and the action expert's continuous execution through closed-loop reinforcement learning (RL). Trained on the same navigation episodes, ZeroVLN substantially outperforms represen
    
[^30]: 来自非高斯源的流匹配：高效采样理论

    Flow Matching from Non-Gaussian Sources: A Theory of Efficient Sampling

    [https://arxiv.org/abs/2611.26407](https://arxiv.org/abs/2611.26407)

    本文为从非高斯源分布（如重尾和多峰分布）出发的流匹配建立了首个非渐近收敛理论，证明欧拉离散化采样器在温和条件下仅需 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代即可达到 $\varepsilon$ 级总变差误差。

    

    流匹配是一种多功能的生成建模框架，它将一个源分布传输至目标数据分布。其关键优势之一在于源分布选择的灵活性，例如可以使用重尾分布和多峰分布作为源。然而，针对非高斯源的有限步采样保证仍然十分有限。本文为采用线性插值的流匹配的欧拉离散化建立了非渐近收敛理论。假设定义在 $\mathbb{R}^d$ 上的源分布具有处处为正、连续可微的密度函数、有限的总方差 $\mathsf{TotVar}_{\mathsf{base}}$，以及 $L$-Lipschitz 的得分函数。在使用总体速度场和适当的时间网格时，只要目标分布满足温和的二阶矩条件，该采样器即可在 $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ 次迭代内实现至多 $\varepsilon$ 的总变差误差。这些假设涵盖……（摘要原文在此处截断）

    arXiv:2611.26407v1 Announce Type: cross  Abstract: Flow matching is a versatile framework for generative modeling that transports a source distribution to a target data distribution. A key advantage is its flexibility in the choice of source distribution, such as heavy-tailed and multimodal sources. Yet finite-step sampling guarantees for non-Gaussian sources remain limited. We establish a nonasymptotic convergence theory for Euler discretization of flow matching with linear interpolation. Suppose the source distribution on $\mathbb{R}^d$ has an everywhere positive, continuously differentiable density, finite total variance $\mathsf{TotVar}_{\mathsf{base}}$, and an $L$-Lipschitz score. Using the population velocity field and a suitable time grid, this sampler achieves total variation error at most $\varepsilon$ in $\widetilde{O}(Ld\mathsf{TotVar}_{\mathsf{base}}/\varepsilon)$ iterations, provided the target distribution satisfies a mild second-moment condition. These assumptions cover,
    
[^31]: 民主化跨项目知识以变革水治理与气候韧性：一个智能体AI生态系统

    Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem

    [https://arxiv.org/abs/2611.26398](https://arxiv.org/abs/2611.26398)

    本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。

    

    欧盟资助的项目以交付成果、报告和案例研究的形式产生了大量证据。然而，这些知识通常分散在不同项目和文档结构之中，使得决策时难以定位相关且具有情境的信息。本文提出了一个智能体AI生态系统，它将异构的项目文档整合为一个领域感知的知识库，并通过智能体检索增强生成流水线提供自适应访问。来自三个欧盟资助项目（IMPETUS、NATALIE和InnWater）的文档被重构为层次化章节树，并使用结构化元数据进行索引。检索由两个在领域数据（水和气候）上微调的领域特定嵌入模型提供支持。在该知识库之上，一个ReAct风格的控制器通过在领域集合之间分配检索预算并积累证据，迭代地编排检索动作。

    arXiv:2611.26398v1 Announce Type: cross  Abstract: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evide
    
[^32]: 世界模型更倾向于习得的先验而非视觉证据

    World Models Favor Learned Priors over Visual Evidence

    [https://arxiv.org/abs/2611.26366](https://arxiv.org/abs/2611.26366)

    该论文揭示了世界模型在跨环境预测中倾向于依赖训练数据中的常见规则先验而非视频中的视觉证据，并提出TailWorld基准来独立控制训练不平衡与校准信息量，以评估示范物理规则在不同初始状态间的迁移能力。

    

    世界模型正被越来越多地用于模拟物理交互，但为每个环境单独训练一个模型限制了可扩展性。跨环境预测需要识别当前的物理规则，而仅凭外观可能无法揭示这一规则。因此，我们研究了视觉校准（visual calibration）问题：利用短视频中物体的运动和交互作为直接证据来推断当前规则，并从不同的初始状态预测该规则下的运动。然而，不同物理条件下数据可用性的差异可能导致长尾的训练分布，使模型偏向常见规则，即使证据支持的是罕见规则。为了研究这一挑战，我们提出了TailWorld，该基准评估示范规则在不同初始状态之间的迁移能力，同时独立控制训练不平衡程度和校准信息量。该基准将模型预测与来自相同预测初始……的两个参考未来进行比较。

    arXiv:2611.26366v1 Announce Type: new  Abstract: World models are increasingly used to simulate physical interactions, but training a separate model for each environment limits scalability. Cross-environment prediction requires identifying the current physical rule, which appearance alone may not reveal. We therefore study visual calibration: using object motion and interactions in a short video as direct evidence to infer the current rule and predict motion under that rule from a different initial state. However, differences in data availability across physical conditions can yield a long-tailed training distribution, biasing models toward common rules even when evidence supports rare ones. To study this challenge, we introduce TailWorld, which evaluates the transfer of demonstrated rules across initial states while independently controlling training imbalance and calibration informativeness. The benchmark compares predictions with two reference futures from the same prediction initia
    
[^33]: 智能体应该记住多少？来自LLM智能体自蒸馏记忆的部署经验教训

    How Much Should an Agent Remember? Deployment Lessons from Self-Distilled Memory for LLM Agents

    [https://arxiv.org/abs/2611.26361](https://arxiv.org/abs/2611.26361)

    该论文提出一个让LLM智能体从自身轨迹中自蒸馏“经验教训”并整合为“指导原则”的记忆流水线，已部署于IBM生产环境，核心发现是最佳记忆量取决于模型自身能力而非记忆机制本身。

    

    LLM智能体在长周期任务中会以反复出现的、程序化的方式失败：它们可能错误地对API进行分页调用、解析到错误的实体，或者在没有被要求作答时给出回答。我们构建了一个让智能体从自身历史中学习的流水线：它首先从智能体自身的轨迹中挖掘简短的“经验教训”，每个回合提取一到三条；然后将这些经验教训整合为一个更小的“指导原则”集合，每条指导原则会记录它合并了多少条经验教训。在推理阶段，智能体接收全部指导原则或经过筛选的子集。该流水线已随开源的altk-evolve库发布，并部署在IBM的CUGA智能体和watsonx Orchestrate平台上。我们报告了在构建、度量和产品化这一流水线过程中所积累的经验。我们的主要评测工具是AppWorld基准，涵盖基线任务完成率在40%至92%之间的多个模型。核心发现是：恰当的记忆量是模型本身的属性，而非记忆机制本身的属性。强模型……（原文摘要在此处截断）

    arXiv:2611.26361v1 Announce Type: new  Abstract: LLM agents on long-horizon tasks fail in recurring, procedural ways. They mis-paginate an API, resolve the wrong entity, or answer when no answer was asked. We built a pipeline that lets an agent learn from its own history. It first mines short \emph{lessons} from the agent's own trajectories, one to three per episode. It then consolidates those lessons into a smaller set of \emph{guidelines}, where each guideline records how many lessons it merged. At inference the agent receives either all guidelines or a selected subset. The pipeline ships in the open-source altk-evolve library and is deployed in IBM's CUGA agent and watsonx Orchestrate platform. We report what we learned building, measuring, and productizing it. Our primary instrument is the AppWorld benchmark, across models spanning 40--92\% baseline task completion. The central finding is that the right amount of memory is a property of the model, not the memory. A strong model kee
    
[^34]: Beam 模型技术报告

    Beam Model Technical Report

    [https://arxiv.org/abs/2611.26358](https://arxiv.org/abs/2611.26358)

    Beam 是一个总参数 5013 亿、活跃参数 234 亿的稀疏 MoE 模型，专为编程、推理和智能体任务优化，通过 23.8 万亿 token 预训练、两阶段中期训练实现百万 token 上下文窗口，并在 10500 块 GB300 GPU 上开展开放实验室中最大规模的强化学习后训练，以比同类模型少 3-4 倍的推理计算量达到前沿推理水平。

    

    Beam 是一个稀疏混合专家模型，总参数量为 5013 亿，活跃参数量为 234 亿，专为编程、推理和智能体工作负载而构建和优化。Beam 能够提供前沿水平的推理能力，同时其所需推理计算量比同级别的许多模型少 3-4 倍。Beam 在 23.8 万亿个多样化、经过精选的高质量 token 上进行预训练，并通过一种新颖的两阶段中期训练方法实现了 100 万 token 的上下文窗口，该阶段建立在可验证的、以正确性为锚点的数据之上。后训练阶段利用大规模、完全异步的强化学习（RL）训练，在 10500 块 GB300 GPU 上进行，在 4 周的训练中生成了超过 1 亿次 rollout。我们相信这是迄今为止任何开放实验室所进行的最大规模强化学习训练之一。在评估套件中，随着强化学习计算量的增加，模型能力持续提升，未出现平台期的迹象。在深度硬件-模型协同设计的支持下，Beam 采用了 FP8/NVFP4 量化技术。

    arXiv:2611.26358v1 Announce Type: cross  Abstract: Beam is a sparse Mixture-of-Experts (MoE) model with 501.3B total parameters and 23.4B active parameters, specifically built and optimized for coding, reasoning, and agentic workloads. Beam delivers frontier level reasoning, while requiring 3-4x less inference compute than many models of its class. Pretrained on 23.8T diverse, curated, high-quality tokens, Beam features a 1M-token context window achieved through a novel two-stage midtraining phase, built on verifiable, correctness-anchored data. Post-training leveraged a large-scale, fully asynchronous reinforcement learning (RL) campaign across 10.5K GB300 GPUs, generating over 100M rollouts over 4 weeks of training. We believe this is one of the largest scale RL runs conducted by any open lab to date. Across our evaluation suite, capabilities continued to improve as we increased RL compute, with no sign of a plateau. Supported by deep hardware-model co-design, FP8/NVFP4 quantization,
    
[^35]: 上下文即资源

    Context as a Resource

    [https://arxiv.org/abs/2611.26346](https://arxiv.org/abs/2611.26346)

    提出无路由架构ClonAct，通过非生成式上下文内存管理单元、原地上下文单元、足迹感知突触总线和波刷新共享主干直接虚拟化上下文块，解决多代理并发会话中的跨主题约束渗漏与盲目合并失败问题。

    

    管理跨独立代理会话的并发任务会迅速耗尽人类注意力，这促使人们需要在一个连续界面背后运行全天候在线的个人代理。然而，当每个上下文共享相同的冻结骨干权重时，将交错的对话和外部事件汇聚到单一的线性会话中，会引发跨主题约束渗漏和队头阻塞。相反，采用路由工单层级结构则会带来有损的文本交接、沉重的路由开销以及盲目合并——尽管孤立分支的通过率很高，但真实存在冲突的代码库合并的三次运行平均成功率却降至33.3%。我们不采用拟人化角色调度的方式，而是提出ClonAct，一种无路由架构，它通过可插拔的单遍非生成式上下文内存管理单元直接虚拟化上下文块，配合原地上下文单元、具有隔离试合并门的足迹感知突触总线，以及波刷新共享主干。在三个程序化多流基准测试上……

    arXiv:2611.26346v1 Announce Type: new  Abstract: Managing concurrent tasks across separate agent sessions quickly exhausts human attention, motivating always-on personal agents behind a single continuous interface. Yet when every context shares identical frozen backbone weights, funneling interleaved dialogues and external events into one linear session induces cross-topic constraint bleed and head-of-line stalling. Conversely, router-ticket hierarchies incur lossy prose handoffs, heavy routing overhead, and blind merges - collapsing true-conflict repository merges to a 33.3% three-run mean despite high isolated branch passes. Rather than scheduling anthropomorphic roles, we introduce ClonAct, a routerless architecture that virtualizes context blocks directly through a pluggable single-pass non-generative Context MMU, in-place Context Cells, a footprint-aware Synaptic Bus with an isolated trial merge gate, and a wave-flushed Shared Trunk. Across three programmatic multi-stream benchmar
    
[^36]: DeepQuest：从知识缺口到阿尔茨海默病的科学发现

    DeepQuest: From Knowledge Gaps to Scientific Discovery in Alzheimer's Disease

    [https://arxiv.org/abs/2611.26343](https://arxiv.org/abs/2611.26343)

    DeepQuest是一个知识缺口驱动的AI科学家框架，它将阿尔茨海默病领域综述中的未决问题转化为可检验假设并自动开展迭代实验与两级审计，在OASIS-3上产生了42个可行假设，并以95.2%的审计评估科学发现率在绝大多数评估维度上超越最强基线。

    

    自动化实验并不必然推进科学理解。研究智能体必须识别有意义的知识缺口，确定现有数据能够解决什么问题，并评估其发现是否推进了已有知识。我们提出DeepQuest，一个面向阿尔茨海默病研究的知识缺口驱动型AI科学家框架。DeepQuest将领域综述中未解决的问题转化为可追溯、可检验的假设，并通过数据可行性筛选和分组Elo排名对其进行优先级排序。该框架将迭代实验与两级审计相结合，将实验可靠性与相对于两个冻结文献语料库的知识贡献区分开来。在OASIS-3数据集上，DeepQuest产生了42个可行假设，并在三位语言模型评审员的15项评估维度比较中有14项优于最强基线。完整框架实现了95.2%的经审计评估的科学发现率。

    arXiv:2611.26343v1 Announce Type: new  Abstract: Automating experiments does not necessarily advance scientific understanding. Research agents must identify meaningful knowledge gaps, determine what available data can resolve, and assess whether their findings advance existing knowledge. We introduce DeepQuest, a knowledge-gap-driven AI scientist framework for Alzheimer's disease research. DeepQuest transforms unresolved questions from domain reviews into traceable, testable hypotheses and prioritizes them through data feasibility screening and groupwise Elo ranking. It couples iterative experimentation with two-level auditing that separates experimental reliability from knowledge contribution relative to two frozen literature corpora. On OASIS-3, DeepQuest produces 42 feasible hypotheses and outperforms the strongest baselines in 14 of 15 evaluator-dimension comparisons across three language-model reviewers. The full framework achieves an audit-assessed scientific finding rate of 95.2
    
[^37]: 通过最小编辑训练大语言模型遵循推理指令

    Training LLMs to Follow Reasoning Instructions with Minimal Edits

    [https://arxiv.org/abs/2611.26342](https://arxiv.org/abs/2611.26342)

    提出基于编辑的训练框架，通过定位模型推理轨迹中首个失败点并进行最小文本修正，仅对编辑片段施加监督，从而在保留已有推理模式的同时更精准地训练大语言模型遵循推理指令。

    

    大语言模型日益依赖扩展的推理轨迹，然而用于塑造这种行为的训练信号仍然较为粗糙。对完整轨迹进行监督微调会迫使模型模仿整个演示，过度约束其输出分布；而轨迹级强化学习只能提供较弱的信用分配，对整个完成结果进行奖励或惩罚，却无法识别其具体在何处成功或失败。我们提出了基于编辑的训练，这是一种通过局部文本修正从模型自身轨迹中学习的通用框架。给定一个采样得到的完成结果，编辑器在指定标准下识别出首个失败点，并提出一个最小的替换片段。随后，模型仅在这些被编辑的片段上进行监督训练。通过这一设计，监督被集中在模型推理首次出现问题的位置，同时旨在保留其余部分已有的推理模式。该方法适用……

    arXiv:2611.26342v1 Announce Type: new  Abstract: Large language models increasingly rely on extended reasoning traces, yet the training signals used to shape this behavior remain coarse. Supervised fine-tuning on full trajectories forces the model to imitate an entire demonstration, overconstraining its output distribution, while trajectory-level reinforcement learning provides only weak credit assignment, rewarding or penalizing whole completions without identifying where they succeed or fail. We propose edit-based training, a general framework for learning from a model's own trajectories via localized textual corrections. Given a sampled completion, an editor identifies the first point of failure under a specified criterion and proposes a minimal replacement span. The model is then supervised only on these edited spans. By design, this concentrates supervision where the model's reasoning first breaks down and aims to preserve its existing reasoning patterns elsewhere. The method appl
    
[^38]: Sirius-Tiamat：单次决策编码器在决策发生改变之前能被压缩到多小？

    Sirius-Tiamat: How Small Can a One-Pass Decision Encoder Get Before Its Decisions Change?

    [https://arxiv.org/abs/2611.26339](https://arxiv.org/abs/2611.26339)

    本文系统量化了4.21亿参数单次决策编码器Laya在量化与剪枝等压缩下的鲁棒性，发现压缩导致的决策翻转在域外数据上远大于域内，删层会严重破坏校准（ECE升至39.2%），而一次无需标签的温度缩放即可将其恢复。

    

    单次决策编码器读取一段文本、一个问题和若干命名选项，并在单次前向传播中返回每个选项的概率。我们探究这类编码器——4.21亿参数的Laya——在压缩到什么程度之前其决策保持不变。我们统计与Laya自身全精度（fp32）答案不同的决策数量，并跟踪校准情况（即置信度与准确度的匹配程度，用期望校准误差ECE衡量），测试集包括1,221个训练模式下的合成题目、273个未见模式题目，以及来自ARC、MMLU和HellaSwag的800个域外问题。对相关模型家族的层剪枝已有研究；我们在其基础上增加了决策翻转、校准、留出数据与重新训练的分析。bf16改变0.2%的域内决策但改变2.2%的域外决策；6位分组舍入分别改变1.3%和5.9%；删除6层分别改变0.9%和9.1%，同时ECE从5.9%升至39.2%。一次无需标签的温度拟合即可将ECE恢复至3.6%。每张量单一缩放在…（原文摘要截断）

    arXiv:2611.26339v1 Announce Type: new  Abstract: A one-pass decision encoder reads a text, a question and several named options, and returns a probability for each option in a single forward pass. We ask how far Laya, a 421M-parameter encoder of this kind, can be compressed before its decisions change. We count decisions that differ from Laya's own full-precision (fp32) answer, and track calibration (how well confidence matches accuracy, measured by expected calibration error, ECE), on 1,221 synthetic items with the training schemas, 273 with an unseen schema, and 800 out-of-family questions from ARC, MMLU and HellaSwag. Layer pruning of a related model family is known; we add flips, calibration, held-out data and retraining. bf16 changes 0.2% of in-family decisions but 2.2% out of family; 6-bit grouped rounding 1.3% and 5.9%; deleting 6 layers 0.9% and 9.1%, while ECE rises from 5.9% to 39.2%. One fitted temperature, needing no labels, restores it to 3.6%. One scale per tensor fails b
    
[^39]: 一个框架诊断一切：面向孟加拉语大语言模型幻觉的分层基准

    One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination

    [https://arxiv.org/abs/2611.26336](https://arxiv.org/abs/2611.26336)

    本文提出孟加拉语幻觉检测分层基准 Alikbachan，通过文化距离、任务类型和32种幻觉模式三个维度交叉分层，并在社区竞赛中大规模验证，发现语篇级信息遗漏是零样本LLM评判器与参赛系统的共同弱项，证明了分层设计对模型幻觉能力的显著诊断价值。

    

    低资源语言的幻觉研究很容易碎片化：每一种新命名的失败模式都会催生一个独立的数据集，使稀缺的标注资源被耗费在无法相互比较的资源上。我们主张对这些工作进行整合，并在社区规模上对此加以检验。Alikbachan 是一个孟加拉语幻觉检测基准，它沿三个相互交叉的维度进行分层：经过人工验证的文化距离区间、六种任务类型，以及一个包含五大类别、32种模式的幻觉分类体系。该基准包含一个采集得到的2,516条测试集（第一阶段）和一个按配额构建、经黄金标准标注的5,000条数据集（第二阶段）。这种分层设计具有诊断价值：四个零样本大语言模型评判器在不同分层之间的宏观F1分数差异超过30分，而99个第一阶段和32个第二阶段竞赛系统的条目级预测也复现了这一特征——语篇层面的信息遗漏让评判器和大多数参赛队伍都束手无策，而只有少数系统会在阴性对照上过度标记。评判器的判定概率能对条目进行有效排序（ROC-AUC 0.74–0.…）。

    arXiv:2611.26336v1 Announce Type: cross  Abstract: Hallucination research in low-resource languages fragments easily: each newly named failure mode prompts its own dataset, spending scarce annotators on incomparable resources. We argue for consolidation and test it at community scale. Alikbachan is a Bengali hallucination-detection benchmark stratified along three crossed axes: a human-validated cultural-distance band, six task types, and a five-family, 32-mode hallucination taxonomy. It comprises a harvested 2,516-item test set (Phase 1) and a quota-constructed, gold-labeled 5,000-item set (Phase 2). The stratification is diagnostic. Four zero-shot LLM judges vary by over 30 macro-F1 points across strata, and the item-level predictions of 99 Phase-1 and 32 Phase-2 competition systems reproduce the profile: discourse-level omission defeats judges and most teams alike, whereas only a minority over-flag negative controls. The judges' verdict probabilities rank items well (ROC-AUC 0.74-0.
    
[^40]: ClosureBench：封闭假设下推理的基准测试

    ClosureBench: A Benchmark for Reasoning under Closure Assumptions

    [https://arxiv.org/abs/2611.26335](https://arxiv.org/abs/2611.26335)

    ClosureBench是一个通过在OWA、CWA和LCWA三种封闭假设下仅隔离缺失信息处理方式的变化来测试大语言模型是否遵循缺失信息处理规则的基准，并揭示了总体准确率会掩盖大量封闭推理错误的问题。

    

    基于大型语言模型（LLM）构建的AI智能体经常需要从不完整的记录中回答问题。缺失的信息可能被解释为未知或错误，这取决于信息来源的完整性。当前的评估未能区分这两个方面，因为它们通常与其他推理需求交织在一起。我们提出了ClosureBench，这是一个测试LLM是否遵循所规定的缺失信息处理规则的基准。每个基础场景都在OWA（开放世界假设）、CWA（封闭世界假设）和LCWA（局部封闭世界假设）下进行测试，且各变体之间仅有对缺失信息的处理方式不同。ClosureBench提供了一个经过验证的基础基准，以及针对行动选择、来源范围和更新的智能体扩展。只有当模型在某个基础场景的所有封闭变体中都给出正确答案时，才被认为在该基础场景上是正确的。在960个实例中，总体准确率可能掩盖大量的封闭推理错误。各模型的整体准确率在71.18%到99.86%之间

    arXiv:2611.26335v1 Announce Type: new  Abstract: AI agents built on large language models (LLMs) often answer questions from incomplete records. Missing information might be interpreted as unknown or false, depending on the completeness of the source. Current evaluations fail to distinguish these two aspects, as they are usually interwoven with other reasoning demands. We present ClosureBench, a benchmark that tests whether LLMs follow the stated rules for dealing with missing information. Every base scenario is tested under OWA, CWA, and LCWA, and only the treatment of missing information changes between variants. ClosureBench provides a verified base benchmark as well as agent extensions for action choice, source scope, and updates. A model is considered correct on a base scenario only if it provides correct answers in all of its closure variants. Across 960 instances, overall accuracy can mask substantial closure mistakes. Models achieve overall accuracy from 71.18% to 99.86% across
    
[^41]: 面向视频扩散Transformer中2比特KV缓存量化的在线时序旋转

    Online Temporal Rotations for 2 Bit KV-Cache Quantization in Video Diffusion Transformers

    [https://arxiv.org/abs/2611.26334](https://arxiv.org/abs/2611.26334)

    提出时序旋转量化（TRQ）方法，通过在线拟合正交旋转将KV缓存中时间相邻帧的各通道均值映射为近似常数，并以仅需两比特存储的平面旋转树实现轻量参数化，从而在视频扩散Transformer中实现2比特KV缓存量化。

    

    视频扩散Transformer在去噪步骤之间复用键值（KV）缓存，这在长视频生成中迅速成为瓶颈。我们发现对时间上相距较远的缓存向量进行量化会导致较大的数值范围和较高的量化误差，但KV缓存中时间上相近的视频token在每个通道上共享一种持续稳定的模式。这些模式在不同视频之间也各不相同，这促使我们采用针对每个视频独特缓存自适应调整的旋转方法。我们提出了时序旋转量化（TRQ），该方法在量化之前对缓存中时间相邻的帧进行旋转。对于每个由时间相邻缓存向量组成的块，TRQ拟合一个正交变换，将该块的各通道均值映射为一个在各通道上接近常数的向量。为了保持量化格式的轻量化，我们将该旋转参数化为一棵由2×2平面旋转构成的树，每个旋转仅用两比特存储，随后接一个固定的Hadamard变换。

    arXiv:2611.26334v1 Announce Type: cross  Abstract: Video diffusion transformers reuse key--value (KV) caches across denoising steps, which quickly becomes a bottleneck for long video generation. We find that quantizing temporally distant cache vectors leads to a large range and high quantization error, but that nearby video tokens in the KV-cache share a persistent pattern per channel. These patterns also vary across videos, motivating rotations that are adapted to each video's unique cache. We propose Temporal Rotation Quantization (TRQ), a method that rotates frames in the cache that are adjacent in time before quantization. For each block of temporally adjacent cache vectors, TRQ fits an orthogonal transform that maps the block's per-channel means to a vector that is close to a constant across channels. To keep the quantization format lightweight, we parameterize this rotation as a tree of $2 \times 2$ planar rotations, each stored with only two bits, followed by a fixed Hadamard tr
    
[^42]: Rigel-Ladon：一种针对单遍决策编码器的选项数定律——在留出选项数与留出题型上的测试

    Rigel-Ladon: An Option-Count Law for a One-Pass Decision Encoder, Tested on Held-Out Option Counts and Held-Out Schemas

    [https://arxiv.org/abs/2611.26332](https://arxiv.org/abs/2611.26332)

    该论文发现并验证了一条“选项数定律”：单遍决策编码器的 logit 准确率随 log(K-1) 线性下降（斜率 0.68–0.88），仅凭小选项数（K≤6）的测量即可精确预测留出的大选项数、甚至跨题型（如 HR 部门路由任务）的准确率。

    

    Laya 是一个小型神经网络，它读取一段文本、一个问题和 K 个书面选项，并在单次前向传播中为每个选项输出一个概率。我们研究当 K 从 2 增长到 35 时其准确率如何下降，以及这种下降是否可以被预测。我们在四个问题池上对嵌套选项集（每个 K 选项问题都包含更小的选项集）进行评分，分别针对已发布的模型（Base）和我们的微调模型（FT），并将每个条目与每一个单一竞争对手进行评分。FT 在一个 12 选项的主题问题上训练（同类题型内）；而一个 12 选项的 HR（部门路由）问题属于异类题型。Logit 准确率（正确答案数与错误答案数之比的对数）在 log(K-1) 上呈线性关系，斜率为 0.68 至 0.88。基于 K≤6 的数据拟合，该定律预测 K=7 到 12 时误差在 0.6 分以内，K 到 24 时误差在 1.2 分以内（对数线性规则则分别为 1.6 和 6.7）。采用从主题问题取得的斜率，仅在 HR 任务上测量双选项准确率，即可预测其在其他 K 值下的准确率，误差在 0.3 分以内（FT）和 0.7 分以内（Base）。在一个 35 标签池上……

    arXiv:2611.26332v1 Announce Type: new  Abstract: Laya is a small neural network that reads a text, a question and K written options, and returns one probability per option in a single pass. We ask how its accuracy falls as K grows from 2 to 35 and whether that can be predicted. We score nested option sets (every K-option question contains the smaller ones) on four question pools, for the released model (Base) and our fine-tune (FT), and score every item against each single rival. FT trained on one 12-option topic question (in-family); a 12-option HR (department-routing) question is out-of-family. Logit accuracy (the log of correct over incorrect answers) is linear in log(K-1) with slope 0.68 to 0.88. Fitted on K up to 6, the law predicts K=7 to 12 within 0.6 points and K up to 24 within 1.2 (a log-linear rule: 1.6 and 6.7). Taking the slope from the topic question and measuring only two-option accuracy on HR predicts its other K within 0.3 points (FT) and 0.7 (Base). On a 35-label pool
    
[^43]: 预测编码动力学的控制论近似

    A Control-Theoretic Approximation to Predictive Coding Dynamics

    [https://arxiv.org/abs/2611.26319](https://arxiv.org/abs/2611.26319)

    该论文证明了深度反馈控制（DFC）是预测编码（PC）动力学在不动点附近的局部近似（称为 Approx PC），在浅层线性网络中二者完全等价，且通过重新线性化反馈控制器和分步逼近输出目标可以缓解由深度和非线性带来的偏差，从而厘清了两种生物学可信局部学习框架之间的关系。

    

    反向传播仍然是训练深度神经网络的主流方法，但其对非局部计算的依赖促使研究者寻找生物学上合理的替代方案。预测编码与深度反馈控制这两个著名框架为局部学习提供了不同的途径，但二者之间的关系尚不明确。本研究表明，DFC 是预测编码动力学在不动点附近的局部近似：在浅层线性网络中二者完全等价，而随着网络深度增加、非线性的引入以及偏离平衡点距离的增大，会产生系统性偏差。我们对这种近似（称为 Approx PC）在活动动力学方面偏离 PC 的方式进行了刻画，并证明可以通过对反馈控制器重新线性化以及通过一系列中间推动逐步逼近输出目标来减轻这些偏差。在 MNIST 和 FashionMNIST 数据集上训练的网络中，Approx PC 达到了（与其他方法相当的）任务性能（摘要原文在此处截断）。

    arXiv:2611.26319v1 Announce Type: cross  Abstract: Backpropagation remains the dominant method for training deep neural networks, but its reliance on non-local computations has motivated the search for biologically plausible alternatives. Two prominent frameworks, predictive coding (PC) and deep feedback control (DFC), offer distinct approaches to local learning, yet their relationship remains unclear. Here, we show that DFC arises as a local approximation to predictive coding dynamics near fixed points, with exact equivalence in shallow linear networks and systematic deviations arising with depth, nonlinearity, and distance from equilibrium. We characterize how this approximation, which we call Approx PC, deviates from PC in its activity dynamics, and show that these deviations can be mitigated by relinearizing the feedback controller and approaching the output target through a sequence of intermediate nudges. In networks trained on MNIST and FashionMNIST, Approx PC reaches task perfo
    
[^44]: C-LiTE：一种用于组织病理学中细胞特征学习的轻量级基础模型

    C-LiTE: A Lightweight Foundation Model for Cellular feature learning in Histopathology

    [https://arxiv.org/abs/2611.26316](https://arxiv.org/abs/2611.26316)

    该论文提出C-LiTE——一个仅有2200万参数的轻量级基础模型，通过原生分辨率分词策略实现H&E染色组织病理学中高效的细胞特征学习，弥补了现有模型要么隐式捕获细胞特征、要么计算开销过大且受限于特定细胞群体的不足。

    

    刻画单个细胞是计算病理学的基础，因为细胞形态学为肿瘤分级、核分裂象评估和免疫画像等任务提供了重要信息。然而，大多数病理学基础模型是在组织图像块上训练的，只能隐式地捕获细胞特征；而现有的细胞级模型要么局限于特定的细胞群体，要么依赖于具有数亿参数、计算代价高昂的架构。此外，将视觉Transformer（ViT）直接应用于小尺寸的细胞裁剪图像也存在架构上的挑战：上采样会增加计算成本却不会增加形态学信息，还可能引入插值伪影。为了解决这些局限，我们提出了C-LiTE，一个用于H&E染色组织病理学中细胞表征学习的轻量级基础模型，仅含2200万参数。C-LiTE采用了一种专为细胞输入定制的原生分辨率分词策略……

    arXiv:2611.26316v1 Announce Type: cross  Abstract: Characterising individual cells is fundamental to computational pathology, as cellular morphology informs tasks such as tumour grading, mitotic assessment and immune profiling. However, most pathology foundation models are trained on tissue patches and capture cellular features only implicitly, while existing cell-level models are either restricted to specific cell populations or rely on computationally expensive architectures with hundreds of millions of parameters. Applying ViTs directly to small cellular crops also presents an architectural challenge: upsampling increases computational cost without adding morphological information and may introduce interpolation artefacts. To address these limitations, we introduce C-LiTE, a lightweight foundation model for cellular representation learning in H&E-stained histopathology, with only 22 million parameters. C-LiTE employs a native-resolution tokenisation strategy tailored to cellular inp
    
[^45]: 面向形式化规范目标的强化学习Actor-Predictor架构

    Actor-Predictor Architecture for Reinforcement Learning with Formal Specification Objectives

    [https://arxiv.org/abs/2611.26313](https://arxiv.org/abs/2611.26313)

    该论文提出Actor-Predictor框架，通过预测器估计轨迹未来能否满足形式化规范来学习逻辑进展的替代表示，无需显式构造自动机，并从理论上证明其可等价于自动机状态，从而突破了传统方法受限于小型规范和LTL的局限。

    

    强化学习（RL）中的一个基本挑战是执行时间安全约束，例如满足形式化规范。现有方法通常将逻辑公式转换为自动机，以进行状态增广和奖励塑形。然而，这种方法假设自动机的构造既可行又易于处理，这限制了其仅适用于小型规范和线性时序逻辑（LTL）。本文提出了Actor-Predictor，一个无需显式构造自动机即可学习逻辑进展的替代表示的框架。预测器持续估计部分轨迹是否将在后续步骤中满足形式化规范，其学习到的特征作为辅助输入提供给执行器，随后执行器使用GRPO和RWR进行优化。我们从理论上证明，在预测精度足够的情况下，学习到的表示等价于自动机状态。

    arXiv:2611.26313v1 Announce Type: new  Abstract: One fundamental challenge in reinforcement learning (RL) is enforcing temporal safety constraints, such as satisfying formal specifications. Existing methods often transform logic formulas into automata for state augmentation and reward shaping. However, this approach assumes that automaton construction is both feasible and tractable, which limits its applicability to small specifications and linear temporal logic (LTL). This paper proposes Actor-Predictor, a framework that learns a surrogate representation of logic progress without explicitly constructing an automaton. The predictor continuously estimates whether a partial trajectory will satisfy the formal specification in a later step, and its learned features are provided as auxiliary inputs to the actor, which is then optimized using GRPO and RWR. We show theoretically that, under sufficient predictive accuracy, the learned representation is equivalent to the automaton state. Experi
    
[^46]: 客户端出口——一种在客户端提升大语言模型透明度的方法

    Client Exits - A Way to Increase LLM Transparency on the Client Side

    [https://arxiv.org/abs/2611.26309](https://arxiv.org/abs/2611.26309)

    本文提出“客户端出口”机制，让LLM在输出中插入机器可解释的标记、由客户端自行决定如何以图形、文本、语音等多模态呈现补充信息，从而在不规定具体界面的情况下提升LLM响应的透明度与可审查性。

    

    大语言模型（LLM）生成的文本推荐往往难以审查。本文提出了“客户端出口”，这是一种轻量级机制，用于丰富LLM的响应，同时将语义意图与客户端特定的呈现方式分离开来。模型在其输出中插入预定义的、机器可解释的标记，由接收端的客户端决定是否以及如何实现相应的丰富内容。这使得补充信息能够以图形、文本、语音或其他模态呈现，而无需规定具体的交互界面。我们实现了一个原型，将微调后的Qwen模型与修改后的LibreChat客户端相结合，采用问题-选项-标准（QOC）方法，包含三种基于XML的客户端出口。我们评估了两个面向QOC的模型变体，它们在基本相同的样本上训练，其中一个变体添加了客户端出口标注。在100个测试问题上，客户端出口模型包含……

    arXiv:2611.26309v1 Announce Type: new  Abstract: Textual recommendations generated by large language models (LLMs) can be difficult to inspect. This paper introduces \emph{Client Exits}, a lightweight mechanism for enriching LLM responses while separating semantic intent from client-specific presentation. The model inserts predefined, machine-interpretable markers into its output, and the receiving client decides whether and how to realize the corresponding enrichments. This allows supplementary information to be presented as graphics, text, speech, or other modalities without prescribing a concrete interface. We implement a prototype combining a finetuned Qwen model with a modified LibreChat client. Using the Question--Option--Criteria (QOC) approach including three XML-based client exits. We evaluate two QOC-oriented model variants trained on essentially the same examples, with client-exit annotations added for one variant. Across 100 test questions, the client-exit model includes al
    
[^47]: 利用AlphaProof Nexus发现的高维光子Greenberger–Horne–Zeilinger态的张量代数不可行定理

    A Tensor-Algebraic No-Go Theorem for High-Dimensional Photonic Greenberger--Horne--Zeilinger States Discovered with AlphaProof Nexus

    [https://arxiv.org/abs/2611.26300](https://arxiv.org/abs/2611.26300)

    本文利用张量代数方法证明了一个不可行定理：当维数D≥N时，任何偶数N≥4的N光子D维GHZ态都无法通过概率性对光源对应的单色量子图实现，且该证明由AlphaProof Nexus通过AI辅助形式化证明搜索在Lean中完成。

    

    arXiv:2611.26300v1 公告类型：交叉（cross）摘要：由概率性双光子对光源产生的光子GHZ态可以用边着色、复数加权的图来描述。在这种表示中，以“每个探测器恰好有一个光子”的事件为条件，输出振幅变为完美匹配上的相干叠加和，而生成N光子D维GHZ态则等价于构造一个单色量子图。本文证明：当D≥N时，对于任何偶数N≥4，不存在复数加权的单色量子图。该证明通过AlphaProof Nexus在Lean中进行AI辅助形式化证明搜索而发现，其构造了一个系数可逐顶点因子化的振幅加权和𝓕。在某个固定顶点处对各颜色求和后，问题被约化为有效边权构成的(N-1)×N矩阵。对于图振幅ω与GHZ目标态Δ，行列数的不平衡导致𝓕(ω)=0……

    arXiv:2611.26300v1 Announce Type: cross  Abstract: Photonic GHZ states generated with probabilistic pair sources can be described by edge-coloured, complex-weighted graphs. In this representation, conditioning on an event with one photon in every detector turns the output amplitudes into coherent sums over perfect matchings, and generating an \(N\)-photon \(D\)-dimensional GHZ state is equivalent to constructing a monochromatic quantum graph. Here, we prove that no complex-weighted monochromatic quantum graph exists for any even \(N\ge4\) when \(D\ge N\). The proof, discovered through AI-assisted formal proof search in Lean using AlphaProof Nexus, constructs a weighted sum \(\mathcal F\) of amplitudes whose coefficients factor vertex by vertex. Summing over the colours at one fixed vertex reduces the problem to an \((N-1)\times N\) matrix of effective edge weights. For the graph amplitudes \(\omega\) and GHZ target \(\Delta\), the row--column imbalance yields \(\mathcal F(\omega)=0\) b
    
[^48]: 重复讲述的批评：LLM时代同行评审在声明层面的同质化

    Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era

    [https://arxiv.org/abs/2611.26286](https://arxiv.org/abs/2611.26286)

    通过测量会议和期刊评审中的声明级重叠度，研究发现LLM时代同行评审的批评意见日益同质化——ICLR评审重叠率从2026年的12.0%升至2027年的21.3%，且被检测为LLM生成的评审比人工评审重叠度更高，表明单篇论文评审的批评多样性正在下降。

    

    同行评审依赖于多位读者发现不同的问题。我们测量了会议和期刊评审中的批评重叠度。在ICLR，同一评审阶段评审中的重叠度从2026年的12.0%上升到2027年的21.3%，NeurIPS、ICML以及F1000和PLOS期刊也出现了逐年上升。2027年，一篇论文的第三份ICLR评审仅为其前两份评审新增了5.46个新问题，而2026年这一数字为6.12。我们对所有评审应用了MELD（一种LLM生成文本检测器）。在同一篇论文内，被MELD标记的两份评审比未被标记的两份评审重叠度更高：2024-2025年ICLR高出3.0个百分点，2026-2027年高出4.5个百分点，五个期刊来源合计高出5.4个百分点。这些结果提供了证据，表明被标记为LLM生成的评审彼此之间的差异性低于未被标记的评审，并暗示单篇论文各评审之间的批评多样性正在下降。

    arXiv:2611.26286v1 Announce Type: cross  Abstract: Peer review relies on several readers finding different problems. We measure critique overlap in conference and journal reviews. At ICLR, measured overlap rose from 12.0% in 2026 to 21.3% in 2027, in reviews from the same stage of the review process, with year-on-year rises at NeurIPS, ICML and the F1000 and PLOS journals. A third ICLR review of a paper added 5.46 new problems to the other two in 2027, against 6.12 in 2026. We apply MELD, a detector of LLM-generated text, to all reviews. Within a paper, two reviews that MELD flags overlap more than two unflagged reviews: by 3.0 points at ICLR in 2024-2025, by 4.5 points in 2026-2027, and by 5.4 points across five journal sources. These results provide evidence that LLM-flagged reviews of one paper are less distinct from each other than unflagged reviews, and suggest that the diversity of critiques among reviews of a single paper is falling.
    
[^49]: 如果思维链还不够用怎么办？

    What If Chain-of-Thought Is Not Enough?

    [https://arxiv.org/abs/2611.26282](https://arxiv.org/abs/2611.26282)

    本文提出以“问题—选项—标准”（QOC）结构化决策过程替代思维链对大语言模型进行微调，并通过三代渐进式训练流水线证明该方法能显著提升答案质量，且模型规模越大收益越明显。

    

    大语言模型的推理能力已经从基于提示的方法演变为内化的模型行为。本文研究大语言模型的内部思维模式能否被有意地改变，以及这种改变是否有益。我们不再依赖思维链，而是训练模型遵循“问题—选项—标准”这一结构化决策过程，将任务分解为明确的问题、相互竞争的备选选项以及评估标准。我们提出了一个三代发展流水线：从基础的QOC决策核心，到更丰富的答案合成，再到统计增强的QOC推理。随后，我们从内部一致性和输出质量两个方面对所得模型进行评估。在所报告的实验中，基于QOC的微调相较于未微调的基础模型提升了答案质量，并且当模型规模足够大、能够执行该过程时，收益尤为显著。（注：原摘要在此处截断）

    arXiv:2611.26282v1 Announce Type: cross  Abstract: Reasoning capabilities in large language models have evolved from prompting-based methods to internalized model behaviors. In this paper, we investigate whether the internal thinking model of an LLM can be deliberately changed and whether such a change is beneficial. Instead of relying on Chain-of-Thought, we train models to follow Question--Option--Criteria (QOC), a structured decision process that decomposes a task into explicit questions, competing options, and evaluation criteria. We present a three-generation development pipeline that ranges from a basic QOC decision core to richer answer synthesis and statistically enriched QOC reasoning. We then evaluate the resulting models with respect to both internal consistency and output quality. Across the reported experiments, QOC-based fine-tuning improves answer quality relative to non-fine-tuned base models and yields particularly strong gains once models are large enough to execute t
    
[^50]: 基于计算机视觉的模拟仪表数字化

    Digitization of Analogue Gauges using Computer Vision

    [https://arxiv.org/abs/2611.26262](https://arxiv.org/abs/2611.26262)

    本研究开发了基于计算机视觉的自动化仪表读数系统，通过对比三种混合方法证明OpenCV结合ResNet18的方案准确率最高（97.3%），验证了残差学习架构在工业模拟仪表数字化中的有效性。

    

    模拟仪表在制造业中仍被广泛用于测量物理量，然而操作员的手动监控会引入人为误差并降低运营效率。本研究开发并评估了一种基于计算机视觉和深度学习技术的自动化仪表读数系统。研究实现并比较了三种混合方法：OpenCV结合Scikit-Image（传统计算机视觉）、OpenCV结合ResNet18以及OpenCV结合VGG19（深度学习方法）。通过网络爬取、合成生成和数据增强技术，研究团队整理了一个包含1,848张压力表图像（0-10 BAR量程）的综合数据集。实验结果表明，ResNet18显著优于其他方法，准确率达到97.3%，而VGG19为64.4%，Scikit-Image为63.8%。该研究验证了残差学习架构在工业仪表数字化中的有效性，并提供了一个稳健的（文本在此处截断）

    arXiv:2611.26262v1 Announce Type: cross  Abstract: Analogue gauges remain prevalent in manufacturing industries for measuring physical quantities, yet manual monitoring by operators introduces human error and operational inefficiency. This research developed and evaluated an automated gauge reading system using computer vision and deep learning techniques. Three hybrid approaches were implemented and compared: OpenCV with Scikit-Image (traditional computer vision), OpenCV with ResNet18, and OpenCV with VGG19 (deep learning approaches). A comprehensive dataset of 1,848 pressure gauge images (0-10 BAR range) was curated through web scraping, synthetic generation, and augmentation techniques. The experimental results demonstrate that ResNet18 significantly outperforms the alternatives, achieving an accuracy of 97.3%, compared to VGG19's 64.4% and Scikit-Image's 63.8%. The study validates the effectiveness of residual learning architectures for industrial gauge digitization, providing a ro
    
[^51]: 基于相似度的OSF逻辑实现灵活的SPARQL查询回答

    Similarity-based OSF Logic for Flexible SPARQL Query Answering

    [https://arxiv.org/abs/2611.26261](https://arxiv.org/abs/2611.26261)

    fosf推理器通过在OSF语言中引入类间相似度关系以支持基于相似度的推理，并将OSF查询经模糊本体感知重写后编译为标准SPARQL，实现了对RDF知识图谱的灵活SPARQL查询回答。

    

    我们提出了fosf的更新版本，fosf是一个模糊序类特征（OSF）逻辑推理器，支持对RDF知识图谱进行灵活的SPARQL查询回答。模糊OSF逻辑是一种知识表示与推理语言，其中特征符号表示函数，类符号表示模糊集合。类之间通过模糊包含关系组织起来，且该关系可扩展到OSF项——即表示实体类的记录式结构。针对这些结构的统一算法提供了模糊类型包含的演算。最新版本的fosf通过在OSF语言中引入类之间的相似度关系，增加了对基于相似度推理的支持，同时还提供了一个将OSF项编译为标准SPARQL的翻译引擎。以OSF项形式表达的查询在翻译之前会先经过模糊的本体感知重写，该过程融入了包含类定义、特征定义域和值域声明的OSF理论……（摘要原文此处截断）

    arXiv:2611.26261v1 Announce Type: new  Abstract: We present an updated version of fosf, a reasoner for fuzzy order-sorted feature (OSF) logic, with support for flexible SPARQL query answering over RDF knowledge graphs. Fuzzy OSF logic is a knowledge representation and reasoning language where feature symbols denote functions and sort symbols denote fuzzy sets. Sorts are organized in a fuzzy subsumption relation that extends to OSF terms, record-like structures representing classes of entities. The unification algorithm for these structures provides a calculus of fuzzy type subsumption. The latest release of fosf introduces support for similarity-based reasoning by extending the OSF language with a similarity relation among sorts, alongside a translation engine for compiling OSF terms into standard SPARQL. Queries expressed as OSF terms undergo fuzzy ontology-aware rewriting prior to translation. This process incorporates an OSF theory of sort definitions, feature domain and range decla
    
[^52]: SkillBridge：智能体技能中跨操作系统兼容性缺陷的成本感知诊断与修复

    SkillBridge: Cost-Aware Diagnosis and Repair of Cross-OS Compatibility Defects in Agent Skills

    [https://arxiv.org/abs/2611.26256](https://arxiv.org/abs/2611.26256)

    SkillBridge 提出了一种基于契约的三阶段工作流（静态分析、动态验证与修复），以低成本方式诊断并修复 LLM 智能体技能在跨操作系统场景下的兼容性缺陷。

    

    LLM 智能体越来越多地通过技能来复用可执行能力，但在一个操作系统上正常工作的技能在另一个操作系统上可能会失败，这是因为操作系统相关的假设分散在指南文本、脚本、元数据、安装契约和本地入口点之中。通过在每个操作系统上运行每个技能来诊断此类失败的成本非常高昂：许多执行会消耗模型 token、运行器时间和审查精力，之后才能揭示失败是否真的与操作系统相关。我们提出了 SkillBridge，这是一个基于契约的三阶段工作流：静态分析、动态验证和修复。SkillBridge 首先筛选 3,270 个技能-目标操作系统用例（涵盖 Linux、macOS 和 Windows 上的 1,090 个技能），以寻找对操作系统敏感的契约证据。随后在选定用例上运行动态任务并进行根因审查，以确认失败是否由操作系统假设引起，并修复导致失败的契约字段，例如指南文本、软件包脚本、依赖项或平台相关配置。

    arXiv:2611.26256v1 Announce Type: cross  Abstract: LLM agents increasingly reuse executable capabilities through skills, but a skill that works on one OS may fail on another because OS assumptions are spread across guide text, scripts, metadata, install contracts, and local entrypoints. Diagnosing such failures by running every skill on every OS is expensive: many executions spend model tokens, runner time, and review effort before revealing whether the failure is truly OS-related. We present SkillBridge, a contract-based workflow with three stages: static analysis, dynamic validation, and repair. SkillBridge first screens 3,270 skill-target-OS cases, covering 1,090 skills across Linux, macOS, and Windows, to find OS-sensitive contract evidence. It then runs dynamic tasks and root-cause review on selected cases to confirm whether a failure is caused by an OS assumption, and repairs the contract field that caused the failure, such as guide text, package scripts, dependencies, or platfor
    
[^53]: STAGE：用于弱监督跨推文事件因果关系发现的时空增强图扩展

    STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery

    [https://arxiv.org/abs/2611.26255](https://arxiv.org/abs/2611.26255)

    STAGE框架从局部因果种子出发构建跨推文弱监督标签，结合时空上下文图与图注意力网络，实现灾害社交媒体中跨推文事件因果关系的可扩展自动发现，相比局部监督显著提升因果发现性能。

    

    灾害相关社交媒体中的因果关系往往跨越独立发布的帖子，这使得跨推文因果关系发现难以进行大规模的监督和建模。我们提出STAGE，一个将可扩展的跨推文弱监督构建与基于事件发生的时空关系学习相分离的框架。从局部因果种子出发，STAGE检索跨推文中语义相关的事件发生，并使用因果评分器构建弱标注的跨推文事件对。这些事件对用于监督多头图注意力网络，该网络构建于语义丰富的事件表示和时空上下文图之上。我们使用人工标注的黄金跨推文测试集和15个基线方法，在加州野火和飓风哈维数据集上评估STAGE。与仅使用局部监督相比，构建的跨推文监督使STAGE的AUPRC在野火数据上提升了0.340，在飓风数据上提升了0.107。STAGE在域内测试中取得了0.885的AUROC

    arXiv:2611.26255v1 Announce Type: cross  Abstract: Causal relations in disaster-related social media of- ten span independent posts, making cross-tweet causal discovery difficult to supervise and model at scale. We propose STAGE, a framework that separates scalable cross-tweet weak-supervision construction from spatio-temporal relational learning over event occurrences. Starting from local causal seeds, STAGE retrieves semantically related event occurrences across tweets and uses a causal scorer to construct weakly labeled cross-tweet pairs. These pairs supervise a multi-head Graph Attention Network over semantically enriched event representations and a spatio- temporal context graph. We evaluate STAGE on California Wildfire and Hurricane Harvey using human-gold cross-tweet test sets and 15 baselines. Compared with local-only supervision, the constructed cross-tweet supervision improves STAGE AUPRC by 0.340 on Wildfire and 0.107 on Hurricane. STAGE achieves in-domain AUROC of 0.885 on 
    
[^54]: 持续强化学习中的稳定性差距

    The Stability Gap in Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.26252](https://arxiv.org/abs/2611.26252)

    本文提出了一个在保持环境动态和奖励不变的前提下仅改变观测表示的简单持续强化学习基准，并首次揭示了其中的“稳定性差距”现象——即使智能体可以完全访问过去任务，新任务的引入仍会导致旧任务性能大幅下降后再部分或完全恢复。

    

    持续强化学习（CRL）旨在使智能体能够适应不断变化的任务，同时保留之前获得的能力。评估通常依赖于最终的聚合指标，而这些指标可能掩盖适应过程中的临时性能下降。本工作引入了一个简单的持续强化学习基准，该基准在保持环境动态和奖励不变的情况下改变观测表示，支持概念验证演示、低成本实验以及易于重现的健全性检查。利用这个基准，我们展示了持续强化学习中的稳定性差距：在引入新任务后，先前任务的性能出现大幅下降，随后部分或完全恢复。尽管智能体可以完全访问过去的任务，这种模式仍然会发生。

    arXiv:2611.26252v1 Announce Type: cross  Abstract: Continual reinforcement learning (CRL) aims to enable agents to adapt to changing tasks while preserving previously acquired capabilities. Evaluation often relies on final aggregate metrics that can hide temporary performance degradation during adaptation. This work introduces a simple CRL benchmark that varies observation representations while preserving environment dynamics and rewards, supporting proof-of-concept demonstrations, inexpensive experimentation, and easily reproducible sanity checks. Using this benchmark, we demonstrate the stability gap in CRL: a substantial drop in prior-task performance after the introduction of new tasks, followed by partial or complete recovery. This pattern occurs despite the agent having complete access to past tasks.
    
[^55]: TwinLab-Bench：在真实实验室的智能体孪生环境中评估AI科学家

    TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs

    [https://arxiv.org/abs/2611.26248](https://arxiv.org/abs/2611.26248)

    该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。

    

    AI智能体日益被期望扮演AI科学家的角色，这需要三种开放性递增的能力：正确执行给定的计算工作流程、从给定的设计空间中选择下一个测量方案，以及自主设计实验。现有基准主要评估前两种能力，即通过终端或从预定义候选中进行选择的闭环方式来评估。而评估第三种能力需要可靠地执行智能体自主设计的实验，并使实验读数建立在真实实验室的基础之上。我们提出了TwinLab-Bench，该基准通过来自物理科学、生命科学和工程实验室的52个任务，评估AI科学家在自主设计实验方面的能力。该基准支持超过100种实验操作类型，智能体的报告通过与取自真实实验室记录的隐藏答案进行对比来评分。每个任务都有其专属的TwinLab智能体，该智能体充当实验环境……

    arXiv:2611.26248v1 Announce Type: new  Abstract: AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the env
    
[^56]: FastRoverVLA：面向越野自动驾驶的快速视觉-语言-动作执行

    FastRoverVLA: Fast Vision-Language-Action Execution for Off-Road Autonomous Driving

    [https://arxiv.org/abs/2611.26242](https://arxiv.org/abs/2611.26242)

    FastRoverVLA提出了一种无需训练的执行接口，通过将视觉规划生成与运动条件执行解耦，利用速度-曲率动作缓冲区持续进行运动校正，并根据规划粗糙度评分自适应分配重用预算，从而大幅减少越野自动驾驶中昂贵的多模态推理调用。

    

    越野视觉-语言-动作（VLA）模型会生成基于场景条件的规划，但频繁的轨迹更新会反复调用昂贵的多模态推理。重用一次预测可以减少这种计算需求，但其输出的指令却容易受到车辆运动变化的影响。FastRoverVLA通过将视觉规划生成与基于运动条件的执行解耦来解决这一不匹配问题。其无需训练的接口在两次视觉预测之间保持已解码的规划。速度-曲率动作缓冲区在每次更新时与测量到的车辆运动对齐，集成到局部轨迹中，并通过消耗当前动作向前推进。修正后的剩余动作被写回缓冲区，从而在连续的更新之间保留运动修正。有界刷新规则对新解码规划内的速度和曲率变化进行归一化，并根据所得到的规划粗糙度评分分配短或长的重用预算。该接口既不需要……

    arXiv:2611.26242v1 Announce Type: cross  Abstract: Off-road vision-language-action (VLA) models generate scene-conditioned plans, but frequent trajectory updates can repeatedly invoke expensive multimodal inference. Reusing a prediction reduces this demand while leaving its commands exposed to changes in vehicle motion. FastRoverVLA addresses this mismatch by decoupling visual plan generation from motion-conditioned execution. Its training-free interface maintains decoded plans between visual predictions. A speed-curvature action buffer is aligned with measured motion at each update, integrated into a local trajectory, and advanced by consuming the current action. The corrected remainder is written back to the buffer, preserving motion corrections across successive updates. A bounded refresh rule normalizes speed and curvature variations within a newly decoded plan and assigns a short or long reuse budget according to the resulting plan-roughness score. The interface requires neither b
    
[^57]: 基于物理信息特征表示与预训练表格基础模型的泛化熔池几何预测

    Physics-Informed Feature Representations and Pretrained Tabular Foundation Models for Generalizable Melt Pool Geometry Prediction

    [https://arxiv.org/abs/2611.26235](https://arxiv.org/abs/2611.26235)

    该研究提出用归一化焓和佩克莱数两个物理信息无量纲特征替代原始工艺参数，并结合预训练表格基础模型进行基准测试，以提升LPBF熔池几何预测的跨材料泛化能力与数据效率。

    

    为先进制造开发可泛化、数据高效的机器学习（ML）模型仍然具有挑战性，原因在于实验数据集有限，且工艺-性能关系会随材料与运行条件而变化。在激光粉末床熔融（LPBF）中，熔池宽度、深度以及熔化模式（导热模式、过渡模式或锁孔模式）决定了工艺稳定性与缺陷形成。传统ML模型通常使用有量纲的工艺参数，这些参数只能隐式地表示激光-材料相互作用，可能限制跨材料迁移能力。本研究评估了两个物理信息驱动的无量纲特征——归一化焓与佩克莱数（Peclet number），它们将激光加工条件与材料热物理性能相结合——相比原始工艺参数能否改善熔池预测。我们系统性地对七个模型进行了基准测试：四种经典ML方法（高斯过程、支持向量机、随机森林和X……（原文摘要在此处截断）

    arXiv:2611.26235v1 Announce Type: new  Abstract: Developing generalizable, data-efficient machine learning (ML) models for advanced manufacturing remains challenging because experimental datasets are limited and process-property relationships vary across materials and operating conditions. In laser powder bed fusion (LPBF), melt pool width, depth, and melting mode (conduction, transition, or keyhole) govern process stability and defect formation. Conventional ML models typically use dimensional process parameters, which represent laser-material interactions only implicitly and may limit cross-material transfer. This study evaluates whether two physics-informed dimensionless features, normalized enthalpy and Peclet number, combining laser processing conditions with material thermophysical properties, improve melt pool prediction over raw process parameters. We systematically benchmark seven models: four classical ML methods (Gaussian process, support vector machine, random forest, and X
    
[^58]: Sightline：面向大语言模型集成应用的经校准数据流摘要

    Sightline: Calibrated Data-Flow Summaries for LLM-Integrated Applications

    [https://arxiv.org/abs/2611.26221](https://arxiv.org/abs/2611.26221)

    该论文提出Sightline，通过在Lean 4中形式化证明“占位符只有被注意力读取才会一阶流向transformer输出”这一原理，为LLM集成应用生成经校准的数据流摘要，避免了传统污染分析产生虚假流以及依赖第二模型标注的高昂成本。

    

    程序越来越多地调用大语言模型（LLM），用程序值填充提示词占位符并消费其输出。这样的调用跨越了自然语言与编程语言（NL/PL）之间的边界。为了跨过这一边界追踪数据，分析工具需要一个数据流摘要来说明输出使用了哪些占位符，但没有任何代码描述模型的行为。因此，针对这类LLM集成应用的分析，要么把每个输入都污染到每个输出上从而产生虚假数据流，要么调用另一个模型为每次调用打标签，代价高昂。我们证明并在Lean 4中机械化验证了：在一阶近似下，占位符只能通过读取它的注意力机制流向transformer的输出（不读则不流），并且微弱的读取或敏感度并不能证明占位符不发生流动。与“读取”不同，现有方法所依赖的对数概率梯度和Fisher信号在模型输出置信度高时会消失，而单区间移除方法也会失败。

    arXiv:2611.26221v1 Announce Type: cross  Abstract: Programs increasingly call large language models (LLMs), filling prompt placeholders with program values and consuming the output. Such a call crosses the boundary between natural language and programming language (NL/PL). To follow data across it, an analysis needs a data-flow summary of which placeholders the output uses, but no code describes what the model does. Analyses of these LLM-integrated applications therefore taint every output with every input, creating spurious flows, or ask a second model to label each call, which is costly. We prove, and mechanize in Lean 4, that a placeholder flows to the output of a transformer, to first order, only through the attention that reads it (no reading, no flow), and that a faint reading or sensitivity cannot certify that a placeholder does not flow. Unlike reading, the log-probability gradients and Fisher signals of existing methods vanish on confident outputs, and single-span removal fail
    
[^59]: RegionAgent：面向区域图像修复的两阶段偏好优化

    RegionAgent: Two-Stage Preference Optimization for Regional Image Restoration

    [https://arxiv.org/abs/2611.26179](https://arxiv.org/abs/2611.26179)

    提出RegionAgent自适应修复框架，通过两阶段偏好学习训练视觉语言模型，在每次迭代中针对非均匀耦合退化的不同区域自适应生成特定的修复动作，克服了现有全局统一操作忽视区域差异、修复不完整的问题。

    

    非均匀耦合退化下的图像需要针对不同退化区域具备多样化的修复能力。现有方法采用全局统一的操作，可能忽视区域间的差异，导致修复不完整甚至损坏图像内容。此外，一个退化区域可能需要连续执行多个修复动作，而较早的动作可能改变残余退化特征或引入伪影，从而阻碍后续修复。为应对这些挑战，我们提出了RegionAgent，这是一种自适应修复框架，通过两阶段偏好学习训练视觉语言模型（VLM），在每次迭代中生成针对特定区域的修复动作。具体而言，我们首先通过步骤级直接偏好优化（DPO）训练VLM选择适合当前区域的动作，并利用从即时修复反馈中获得的偏好进行监督……

    arXiv:2611.26179v1 Announce Type: cross  Abstract: Images under non-uniform coupled degradations require diverse restoration capabilities for different degraded regions. Existing methods employ globally uniform operations, potentially overlooking regional differences and causing incomplete restoration or even damage to image content. Moreover, a degraded region may require successive restoration actions, and earlier actions may alter residual degradation characteristics or introduce artifacts, potentially hindering subsequent restoration. To address these challenges, we propose RegionAgent, an adaptive restoration framework that trains a Vision-Language Model (VLM) through two-stage preference learning to generate a region-specific restoration action at each iteration. Specifically, we first train the VLM to select an action suited to the current region through step-level Direct Preference Optimization (DPO), using preferences derived from immediate restoration feedback to supervise on
    
[^60]: 预算内化：固定预算RLVR以牺牲测试时扩展换取预算内准确率

    Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy

    [https://arxiv.org/abs/2611.26172](https://arxiv.org/abs/2611.26172)

    固定预算RLVR训练会将生成长度上限内化为模型的习得约束，用测试时扩展能力换取了预算内的准确率提升。

    

    推理模型在测试时能够生成更长的答案时往往会变得更准确，这种行为被称为测试时扩展。然而，基于可验证奖励的强化学习（RLVR）通常在固定的生成长度上限内训练模型。我们发现这一上限不仅影响训练成本，还会改变模型行为。训练后，模型在训练预算之外仅使用很少的额外token。它们在接近该预算时优于基础模型，但其准确率停止提升，并在更大预算下低于基础模型。因此，生成长度上限成为了一种习得的约束，限制了模型的测试时扩展能力。我们将这一效应称为“预算内化”。我们在Qwen3.5-4B、Gemma4-E2B和Nemotron-3-Nano-4B三个模型上，以及数学推理和多样化程序生成问题两类推理训练任务中均观察到了这一效应。该效应还会从数学训练迁移到科学和编程任务。在Qwen3.5-4B上进行的训练目标层面的消融实验表明……

    arXiv:2611.26172v1 Announce Type: cross  Abstract: Reasoning models often become more accurate when they can generate longer answers at test time. This behavior is known as test-time scaling. However, RL with verifiable rewards (RLVR) usually trains them with a fixed rollout cap. We find that this cap changes model behavior, not only training cost. After training, models use few additional tokens beyond the training budget. They outperform the base models near that budget, but their accuracy stops improving and falls below the base model at larger budgets. The rollout cap thus becomes a learned constraint, that limits model's test-time scaling. We call this effect budget internalization. We observe it across Qwen3.5-4B, Gemma4-E2B, and Nemotron-3-Nano-4B and two reasoning training tasks: mathematical reasoning and diverse, procedurally generated problems. This effect also transfers from math training to science and coding tasks. Training-objective level ablations on Qwen3.5-4B show tha
    
[^61]: FrontierOR挑战赛：面向运筹学的智能体算法设计

    The FrontierOR Challenge: Agentic Algorithm Design for Operations Research

    [https://arxiv.org/abs/2611.26171](https://arxiv.org/abs/2611.26171)

    FrontierOR挑战赛邀请各团队构建基于LLM的智能体工作流，为多类优化问题自动设计、测试和改进可媲美商业求解器Gurobi的算法，以推动智能体算法设计能力的发展。

    

    运筹学（OR）依赖于专家针对每个问题量身定制的算法：一个新的路径规划、调度或网络设计问题，往往需要数周的专家工作才能在大规模下得到良好求解。大型语言模型（LLM）智能体现在已能编写、测试和改进代码，但它们能否设计出可扩展至真实实例规模的算法仍是悬而未决的问题：在FrontierOR基准的一次性评估中，最强的模型仅在39%的大型实例上能媲美Gurobi（目标值在Gurobi的1%以内且速度不慢）。FrontierOR挑战赛要求各团队构建智能体工作流，以同时为多个优化问题设计、测试和改进算法。第一阶段将分周发布基准180个经审核问题中的60个（每周15个），涵盖六个应用类别和五种建模类型；第二阶段则引入基准之外的新问题及未见过的实例。比赛设有两条赛道分别排名：主赛道（原文在此处截断）……

    arXiv:2611.26171v1 Announce Type: new  Abstract: Operations research (OR) relies on algorithms that experts tailor to each problem: a new routing, scheduling or network design problem can take weeks of specialist work before it is solved well at scale. Large language model (LLM) agents now write, test and improve code, but whether they can design algorithms that scale to realistic instances is open: in the FrontierOR benchmark's one-shot evaluation, the strongest model matches Gurobi, with an objective within 1% of Gurobi's and no slower, on only 39% of the large instances. The FrontierOR Challenge asks teams to build agentic workflows that design, test, and improve algorithms for many optimization problems at once. Stage 1 releases 60 of the benchmark's 180 audited problems, spanning six application categories and five formulation types, in weekly rounds of 15; Stage 2 introduces new problems outside the benchmark, with unseen instances. Two tracks are ranked separately: in the Main t
    
[^62]: Sage-Evolve-Preview：面向后训练自我改进的自适应观察

    Sage-Evolve-Preview:Adaptive Observation for Post-Training Self-Improvement

    [https://arxiv.org/abs/2611.26132](https://arxiv.org/abs/2611.26132)

    针对递归自我改进循环中仅依赖锚点基准聚合分数导致进展停滞的问题，本文提出自适应观察（AdaObs）机制，在保持锚点目标不变的前提下，于平台期动态细化诊断粒度、提升难度并扩展相关任务，从而突破由“观察枯竭”造成的改进瓶颈。

    

    后训练递归自我改进（RSI）通过反复的干预、训练和评估，迭代地提升某一目标能力，该能力由一个固定的锚点基准来衡量。然而，大多数循环仅将锚点的聚合分数作为唯一观察，导致早期收益让位于停滞的进展和重复的探索。当这种仅依赖锚点的反馈不再揭示可操作的差距时，平台期可能反映的是观察的枯竭，而非能力的枯竭。我们提出了自适应观察，这是一种工具演化的观察侧机制，它在保持锚点目标不变的同时调整活跃观察。在遇到平台期时，控制器会细化诊断粒度、提升任务难度并添加相关任务；经过验证的增量会被融合回训练中，而每个检查点仍然对照锚点进行评估。在八个独立优化的锚点上（涉及长文本阅读理解等任务）……

    arXiv:2611.26132v1 Announce Type: new  Abstract: Post-training recursive self-improvement (RSI) iteratively improves a target capability, operationalized by a fixed anchor benchmark, through repeated intervention, training, and evaluation. Yet most loops use the anchor's aggregate score as their sole observation, so early gains give way to stalled progress and repetitive exploration. When this anchor-only feedback ceases to reveal actionable gaps, a plateau may reflect an exhausted observation rather than an exhausted capability. We introduce adaptive observation (AdaObs), an observation-side mechanism of harness evolution that keeps the anchor objective fixed while adapting the active observation. At a plateau, a controller refines diagnostic granularity, escalates difficulty, and adds related tasks; validated increments are fused back into training, while every checkpoint remains evaluated against the anchor. Across eight independently optimized anchors in long-context reading compre
    
[^63]: ChronoPatch：针对视觉语言模型的黑盒能耗-延迟攻击

    ChronoPatch: Black-Box Energy-Latency Attacks on Vision-Language Models

    [https://arxiv.org/abs/2611.26117](https://arxiv.org/abs/2611.26117)

    ChronoPatch提出了一种仅需观察最终响应的黑盒攻击方法，通过响应长度引导的进化搜索和子空间校准的决策式优化，使视觉语言模型的推理延迟和GPU能耗显著增加。

    

    视觉语言模型越来越多地被部署为生成式推理服务，其中自回归解码使得响应长度成为延迟和计算成本的关键决定因素。这带来了可用性风险：对抗性视觉输入可以延长生成过程并增加服务开销。现有的能耗-延迟攻击通常依赖于梯度、logits或token级别的反馈，因而攻击者仅能观察最终响应时这种风险是否仍然存在尚不明确。我们提出了ChronoPatch，一种仅需最终响应即可诱导VLM延长生成的黑盒攻击。该方法通过响应长度引导的进化搜索发现视觉延迟锚点，随后通过子空间校准的基于决策的优化来最小化扰动幅度。在四个开源视觉语言模型上，ChronoPatch在5%的空间预算下将推理延迟增加了47.86%–293.34%，将GPU能耗增加了41.44%–249.70%。

    arXiv:2611.26117v1 Announce Type: cross  Abstract: Vision-language models (VLMs) are increasingly deployed as generative inference services, where autoregressive decoding makes response length a key determinant of latency and computational cost. This creates an availability risk: adversarial visual inputs can prolong generation and increase serving overhead. Existing energy-latency attacks typically rely on gradients, logits, or token-level feedback, leaving open whether such risks persist when an attacker observes only the final response. We present ChronoPatch, a black-box attack that induces prolonged VLM generation requiring only the final response. It discovers visual delay anchors through response-length-guided evolutionary search, then minimizes perturbation magnitude through subspace-calibrated decision-based optimization. Across four open-source VLMs, ChronoPatch increases inference latency by 47.86%--293.34% and GPU energy consumption by 41.44%--249.70% under a 5% spatial bud
    
[^64]: SenseAlign：面向无人机主动城市感知的决策对齐预测表征学习

    SenseAlign: Learning Decision-Aligned Predictive Representations for Active Urban Sensing with UAVs

    [https://arxiv.org/abs/2611.26109](https://arxiv.org/abs/2611.26109)

    提出基于强化学习的SenseAlign框架，通过特权教师-学生学习与决策对齐的预测表征，实现大规模城市路网中无人机编队面向动态交通状态的主动感知协调。

    

    主动城市感知通过调度移动传感器编队，主动巡检动态路网并为系统管理收集关键信息。该问题同时要求理解复杂的时空演化过程，以及在部分观测条件下进行决策。在本工作中，我们关注大规模城市路网中的主动感知问题，其效用建立在动态交通状态之上，而以往的研究很少将这些因素联合考虑。我们提出了SenseAlign，一个基于强化学习（RL）的框架，用于学习决策对齐的预测表征，以协调城市感知无人机编队。SenseAlign首先训练一个拥有特权信息的教师智能体，它利用未来的真实交通数据做出最优感知选择。学生智能体除了经过强化学习训练的actor和critic分支之外，还包含一个辅助分支，用于学习在部分观测条件下预测未来交通状况……

    arXiv:2611.26109v1 Announce Type: new  Abstract: Active urban sensing orchestrates a mobile sensor fleet to proactively inspect a dynamic road network and gather critical information for system management. This problem simultaneously requires understanding the complex spatio-temporal evolution and making decisions with partial observations. In this work, we focus on active sensing in a large-scale urban road network with utilities grounded in dynamic traffic states, while prior work has rarely considered these factors jointly. We propose \emph{SenseAlign}, a Reinforcement Learning-based (RL) framework that learns decision-aligned predictive representations for coordinating an urban sensing drone fleet. SenseAlign first trains a privileged teacher agent that exploits future ground-truth traffic for optimal sensing choices. The student agent has an auxiliary branch that learns to predict future traffic with partial observations, in addition to the RL-trained actor and critic branches. Du
    
[^65]: 竞争者之间帕累托最优且防策略操纵的数据共享

    Pareto-Optimal and Strategyproof Data-Sharing Between Competitors

    [https://arxiv.org/abs/2611.26108](https://arxiv.org/abs/2611.26108)

    本文提出了一族数据共享合同，使竞争企业的相对市场份额保持不变，并证明其中一种合同同时满足帕累托最优性和防止企业少报数据量的防策略操纵性。

    

    我们研究在同一市场中通过预测模型质量进行竞争的企业之间的数据共享问题。我们通过两个主要组成部分对企业利润进行建模：精度和市场份额，其中前者取决于企业自身的数据点数量，后者取决于所有企业的数据禀赋。我们将帕累托最优合同刻画为至少有一家企业获得所有其他企业数据点的合同。随后，我们提出了一族参数化的数据共享合同，并证明在其中任何一种合同下，所有企业的相对市场份额保持不变，从而消除了与竞争对手共享数据的负面影响，并使所有企业相对于不共享情形的收益均等化。我们进一步证明，该合同族中的某一个合同同时满足帕累托最优性和防策略操纵性，即能够防止企业少报其初始数据禀赋规模。最后，我们利用真实世界数据集进行了实证验证。

    arXiv:2611.26108v1 Announce Type: new  Abstract: We study data-sharing among firms competing in the same market through the quality of their predictive models. We model firms' profit through two main components, precision and market share, where the former depends on the firm's own number of datapoints and the latter depends on the endowments of all firms. We characterize Pareto-optimal contracts as those in which at least one firm receives all of the other firms' datapoints. We then propose a parametric family of data-sharing contracts and show that, under any of them, the relative market shares of all firms remain constant, eliminating the negative effect of sharing data with competitors and equalizing all firms' gains relative to the no-sharing scenario. We further show that one contract within this family satisfies both Pareto-optimality and strategyproofness against firms attempting to under-declare the size of their initial endowment. Finally, using a real-world dataset, we empir
    
[^66]: HiUni：在共享视觉空间中协调理解与生成

    HiUni: Harmonizing Understanding and Generation in Shared Visual Space

    [https://arxiv.org/abs/2611.26107](https://arxiv.org/abs/2611.26107)

    HiUni 通过让理解与生成任务共享同一个语言对齐的 ViT 和密集 Transformer 骨干网络、仅将任务特定计算放在输出端，并借助像素扩散机制解决优化困难与细节丢失问题，实现了高度参数共享的统一多模态模型。

    

    统一多模态模型旨在在单一框架内同时支持视觉理解与生成。然而，主流方法往往将视觉表示或骨干网络计算分离开来，以适应这两项任务的不同需求，这限制了模型统一化的程度。本文提出 HiUni，重新审视这一设计选择，并探索预训练视觉语言模型能否通过广泛的参数共享直接扩展为一个统一模型。具体而言，HiUni 为理解与生成两种能力复用同一个与语言对齐的 ViT 和密集 Transformer 骨干网络，同时将任务特定的计算限制在输出端。然而这样的设计带来了两个关键挑战：像素空间中的生成难以优化，且预训练 ViT 的语义抽象会丢弃图像合成所需的细粒度空间细节。为应对这些挑战，HiUni 引入了像素扩散机制（摘要在此处被截断）。

    arXiv:2611.26107v1 Announce Type: cross  Abstract: Unified multimodal models aim to support visual understanding and generation within a single framework. However, prevailing approaches often separate their visual representations or backbone computation to accommodate the different requirements of these two tasks, which limits the degree of unification. In this paper, we present HiUni, which revisits this design choice and explores whether a pretrained vision language model can be directly extended into a unified model with extensive parameter sharing. Specifically, HiUni reuses the same language-aligned ViT and dense Transformer backbone for both capabilities, while confining task-specific computation to the output side. Yet such a design introduces two key challenges. Generation in pixel space is difficult to optimize, and semantic abstraction in pretrained ViT discards fine spatial details required for synthesis. To address these challenges, HiUni introduces a \emph{Pixel Diffusion 
    
[^67]: 使用模拟的生物合理网络控制人形机器人H1-2

    Controlling the Humanoid Robot H1-2 With a Simulated Biologically Plausible Network

    [https://arxiv.org/abs/2611.26074](https://arxiv.org/abs/2611.26074)

    提出了一种生物合理的脉冲神经网络，通过固定储备池、奖励调制STDP与资格传播等不同生物可信度的学习方案，成功实现了宇树H1-2人形机器人的运动控制。

    

    人工脉冲神经网络近年来受到极大关注，因为它们已被证明具有模拟生物神经培养并产生相似活动与趋势的能力。然而，它们能否在接受人形机器人运动训练的同时保持生物合理性与稳健性能，仍然是一个悬而未决的问题。我们通过提出一个生物启发的脉冲网络来研究这一问题，该网络采用不同的学习方案进行训练，并在宇树H1-2人形机器人上进行了测试。这些方案的差异在于如何训练循环权重。按生物可信度从高到低排序，它们分别是：一个模拟体外神经培养的固定储备池网络，其循环权重保持冻结，仅训练线性读出层；相同的网络但其突触通过奖励调制的脉冲时序依赖可塑性和资格传播进行训练；最后是时间反向传播作为非生物合理的基线方法。

    arXiv:2611.26074v1 Announce Type: new  Abstract: Artificial Spiking Neural Networks have gained remarkable attention in the last few years due to their proven ability in simulating biological neural cultures and producing similar activity and trends. Nevertheless, whether they can be trained for humanoid locomotion while retaining biological plausibility and robust performance at once remains an open question. We study this problem by proposing a biologically inspired spiking network, trained by different learning schemes and tested on the H1-2 Unitree Humanoid. These schemes differ in how recurrent weights are trained. Ordered from most to least biologically faithful, they are: a fixed reservoir mimicking an in-vitro neural culture, where recurrent weights are kept frozen and only a linear readout is trained, the same network but with its synapses trained by reward-modulated spike timing dependent plasticity and eligibility propagation, and finally backpropagation through time as a no
    
[^68]: AltRA-Test：一种无需共享标注者的LLM评判者替代标注者检验

    AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators

    [https://arxiv.org/abs/2611.26073](https://arxiv.org/abs/2611.26073)

    提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。

    

    大语言模型越来越多地被用作评判者和评分者来替代人类标注者，这引发了一个问题：这种替代在何时具有统计学上的合理性。为了支持此类决策，此前提出了替代标注者检验，适用于所有条目均由相同标注者标注的密集场景。然而，在许多大规模标注场景中，条目是由不同的标注者子集进行标注的。为此，我们提出了基于重采样标注的替代标注者检验，这是一种不要求所有条目均由相同标注者标注的统计检验方法。AltRA-Test采用自助法重采样条目，为每个条目抽取k个人类标注，并将其中一个替换为LLM标注，从而生成配对的全人类标注组与人类-LLM混合标注组。当混合标注组的标注者间一致性在统计学上不劣于全人类标注组时，即支持进行替换。

    arXiv:2611.26073v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judges and graders to replace human annotators, raising the question of when such replacement is statistically justified. To support such decisions, the Alternative Annotator Test (Alt-Test) was introduced for dense settings where all items are annotated by the same annotators. However, in many large-scale annotation settings, items are annotated by different subsets of annotators. We therefore propose the Alternative Annotator Test with Resampled Annotations (AltRA-Test), a statistical test that does not require that all items are annotated by the same annotators. AltRA-Test uses a bootstrap that resamples items, draws k human annotations for each item, and substitutes one of them with an LLM annotation. This produces paired all-human and mixed human-LLM panels. A replacement is supported when the inter-annotator agreement (IAA) of the mixed panel is statistically non-inferior to t
    
[^69]: GenSPARQL：面向不完整知识图谱的SPARQL生成式扩展

    GenSPARQL: A Generative Extension of SPARQL for Incomplete Knowledge Graphs

    [https://arxiv.org/abs/2611.26062](https://arxiv.org/abs/2611.26062)

    提出GenSPARQL，通过在SPARQL 1.1中新增GENOP算子，在查询求值时将LLM生成的值绑定到查询变量，并借助相似度连接与查询规划器按需调用LLM，从而高效补全不完整知识图谱中的缺失事实。

    

    Web上的知识图谱是不完整的，而SPARQL仅能根据已存储的三元组来回答查询。大型语言模型（LLM）可以预测缺失的事实，但LLM返回的是名称而非IRI，并且每绑定一个查询变量就需要调用一次。我们提出了GenSPARQL，这是SPARQL 1.1的一个生成式扩展，其中包含算子GENOP，可在查询求值过程中将LLM生成的值绑定到查询变量。相似度连接将生成的名称映射到图中的实体，而查询规划器仅为其他图模式和过滤器所筛选出的绑定调用LLM。在一个Web工作台中，与会者可以查询两个基于Wikidata构建的图，并查看生成的值、映射的实体以及LLM调用情况。在我们使用三个开源权重LLM的评估中，基于嵌入的相似度成功映射了LLM所命名实体的83%，且仅产生一次错误匹配。

    arXiv:2611.26062v1 Announce Type: cross  Abstract: Knowledge graphs on the Web are incomplete, and SPARQL answers a query only from the stored triples. Large language models (LLMs) can predict missing facts, but an LLM returns names instead of IRIs and requires one call per binding of a query variable. We present GenSPARQL, a generative extension of SPARQL 1.1 with an operator GENOP that binds values generated by an LLM to query variables during query evaluation. A similarity join maps generated names to entities of the graph, and the query planner invokes the LLM only for the bindings selected by the other graph patterns and filters. In a web workbench, attendees query two graphs built from Wikidata and inspect generated values, mapped entities and LLM calls. In our evaluation with three open-weight LLMs, the embedding similarity maps 83% of the entities named by the LLMs and yields only one incorrect match.
    
[^70]: 研究《人工智能法案》的提及：欧盟AI监管对Hugging Face影响的首次观察

    Studying AI Act Mentions: A first look at the impact of the European Union's AI regulation on Hugging Face

    [https://arxiv.org/abs/2611.26050](https://arxiv.org/abs/2611.26050)

    这是首个针对欧盟《人工智能法案》在Hugging Face平台影响的实证研究，通过分析超过500万条记录中1,500余条明确提及该法案的内容，揭示了AI从业者如何在实践中回应欧盟AI监管。

    

    欧盟《人工智能法案》（AI Act）于2024年8月生效，其义务分阶段实施。针对通用人工智能模型提供者的义务自2025年8月起适用，透明度规则自2026年8月起适用。这些义务影响到在Hugging Face（HF）——最大的AI制品分发平台——等公共平台上分享和部署模型与应用程序的开发者。然而，关于从业者在实践中如何应对该法规，目前知之甚少。为填补这一空白，我们对HF上明确提及《人工智能法案》的内容进行了实证研究。我们收集了超过500万条记录，包括来自Models、Datasets和Spaces的仓库卡片、讨论和提交，并识别出超过1,500条明确提及该法案的记录。两名编码员采用以封闭编码为主的方法对这些记录进行了手动标注，并按来源分析了由此形成的类别。

    arXiv:2611.26050v1 Announce Type: new  Abstract: The European Union's Artificial Intelligence Act (AI Act) entered into force in August 2024, and its obligations apply in stages. Obligations for providers of general-purpose AI models have applied since August 2025, and transparency rules since August 2026. These obligations affect developers who share and deploy models and applications on public platforms such as Hugging Face (HF), the largest platform for distributing AI artifacts. However, little is known about how practitioners respond to the regulation in practice. To address this gap, we conducted an empirical study of explicit references to the AI Act on HF. We collected over five million records, including repository cards, discussions, and commits from Models, Datasets, and Spaces, and identified over 1,500 records that explicitly mention the Act. Two coders manually labeled these records using a primarily closed coding approach, and we analyzed the resulting categories per sou
    
[^71]: SEOYUN：面向AI眼镜管线的拒答诱导型考试文档

    SEOYUN: Refusal-Inducing Exam Documents for AI-Glasses Pipelines

    [https://arxiv.org/abs/2611.26033](https://arxiv.org/abs/2611.26033)

    该论文提出SEOYUN框架，通过设计能诱导视觉语言模型主动弃答的考试文档来防御AI眼镜作弊，其核心贡献是针对考试场景的威胁模型、分析性限制与评估协议。

    

    配备摄像头的眼镜可以将考试页面转换成对视觉语言模型（VLM）的查询。设备检测针对的是持有行为；一个互补的问题是考试文档本身是否能够降低未经授权拍摄的有用性。我们提出了SEOYUN，这是一个面向拒答诱导型考试文档的应用专用框架，并以光学层面和编码器层面的干预作为扩展。我们使用“反向越狱”来描述该目标的方向：诱导模型弃答而非绕过拒答机制。拒答诱导和防御性视觉提示注入已有先例；我们的贡献在于威胁模型、分析性限制以及面向考试场景的评估协议。我们区分了拒答与正确答案泄露，说明了为什么移除一个附加的干扰选项本身并不构成对原始答案空间的缩减，并在明确声明的摄像头假设下推导出一种抽样比较。该协议……（摘要在此处截断）

    arXiv:2611.26033v1 Announce Type: new  Abstract: Camera-equipped glasses can convert an examination page into a query to a vision-language model (VLM). Device detection addresses possession; a complementary question is whether an examination document can reduce the usefulness of an unauthorized capture. We present SEOYUN, an application-specific framework for refusal-inducing exam documents, with optical and encoder-level interventions as extensions. We use reverse jailbreaking to describe the direction of the objective: inducing abstention rather than bypassing refusal. Refusal induction and defensive visual prompt injection already have precedents; our contribution is a threat model, analytical limits, and an evaluation protocol for examination use. We distinguish refusal from correct-answer leakage, show why removing an added dummy option is not itself a reduction in the original answer space, and derive a sampling comparison under explicitly stated camera assumptions. The protocol 
    
[^72]: 共享潜在状态可交换决策中的串行化税

    Serialization Tax in Shared-Latent Exchangeable Decisions

    [https://arxiv.org/abs/2611.26029](https://arxiv.org/abs/2611.26029)

    本文提出“串行化税”概念：当条目共享不确定的潜在状态时，独立边缘化虽保持均值却破坏了聚合后验中的相关性，作者用凸序比较量化方差与全变差差距、推导阈值分离并连接决策遗憾，进而提出 TaxScore 识别决策边界附近存在未消解不确定性的大型隐藏块以引导更强的推断。

    

    准确的单项概率也可能产生不准确的群体层面决策。一个系统可能对每个条目分别评分，但其行动却取决于是否有足够多的隐藏条目成功：例如展示一版推荐列表、推迟一个小组决策，或检查一整批样本。当各条目共享一个不确定的潜在状态时，独立的边缘化虽然能保持均值，却会从聚合后验中移除相关性。我们将这种误差称为“串行化税”。对于共享同一成功率的二元条目，经典的凸序比较表明，精确计数的分布比任何具有相同总均值的独立补全模型都更为分散。我们量化了方差差距与全变差差距，在边际条件下推导了阈值分离，并将后验近似与决策遗憾联系起来。其他独立的结果涵盖了非二元结果、异质条目以及充分后验分辨率的情形。TaxScore 能够识别处于决策边界附近且不确定性未消除的大型隐藏块，以便将其路由至更丰富的（推断流程）。

    arXiv:2611.26029v1 Announce Type: new  Abstract: Accurate item probabilities can produce inaccurate group-level decisions. A system may score each item separately, yet act on whether enough hidden items succeed: show a slate, defer a panel decision, or inspect a batch. When items share an uncertain latent state, independent marginalization preserves means but removes dependence from the aggregate posterior. We call this error the serialization tax. For binary items sharing one success rate, classical convex-order comparisons give a more dispersed exact count law than any independent completion with the same total mean. We quantify variance and total-variation gaps, derive threshold separation under margin conditions, and connect posterior approximation to decision regret. Separate results cover non-binary outcomes, heterogeneous items, and sufficient posterior resolution. TaxScore identifies large hidden blocks with unresolved uncertainty near a decision boundary for routing to richer 
    
[^73]: ForeSeek：面向长视频理解的前瞻驱动推理

    ForeSeek: Foresight-Driven Reasoning for Long-Form Video Understanding

    [https://arxiv.org/abs/2611.26024](https://arxiv.org/abs/2611.26024)

    ForeSeek 提出一种前瞻驱动的视频智能体，通过对未观察的视频区域形成并验证视觉预期、将失败预期作为回溯信号，把长视频推理从被动检索转变为预测性探索，并配合两阶段训练框架赋予模型前瞻推理与高效决策能力。

    

    长视频理解作为一项基础而又至关重要的视觉任务，正在被基于多模态大语言模型（MLLM）的视频智能体所革新。然而，这类智能体固有的检索驱动特性构成了显著瓶颈：它们只能基于局部显著性被动抓取视频片段，而缺乏全局的时间规划。为解决这一问题，我们提出了 ForeSeek，一种前瞻驱动的视频智能体，将长视频推理从被动检索转变为预测性探索。ForeSeek 并非盲目地探测时间线，而是对尚未观察的时间区域形成基于视觉的预期，通过有针对性的观察加以验证，并根据不断涌现的新证据持续优化其搜索策略，其中未被证实的预期可作为有效信号，支持有原则的回溯。此外，我们还开发了一个两阶段训练框架，使 MLLM 具备前瞻引导的推理能力和高效的决策能力。

    arXiv:2611.26024v1 Announce Type: new  Abstract: Long-form video understanding, as a fundamental yet crucial vision task, is being revolutionized by multimodal large language model (MLLM)-based video agents. However, the retrieval-driven nature inherent to these agents creates a significant bottleneck, forcing them to reactively fetch clips based on local saliency without a global temporal plan. To address this, we present ForeSeek, a foresight-driven video agent that transforms long-video reasoning from reactive retrieval into predictive exploration. Rather than blindly probing the timeline, ForeSeek forms visually grounded expectations over unseen temporal regions, verifies them through targeted observations, and continuously refines its search based on emerging evidence, where failed expectations serve as informative signals for principled backtracking. Furthermore, we develop a two-stage training framework that equips MLLMs with foresight-guided reasoning and efficient decision mak
    
[^74]: 工作空间是否真的在工作？一种用于验证语言模型中全局工作空间计算的因果协议

    Does the Workspace Do the Work? A Causal Protocol for Verifying Global-Workspace Computation in a Language Model

    [https://arxiv.org/abs/2611.26018](https://arxiv.org/abs/2611.26018)

    本文提出一种因果测试协议，通过从其他输入移植消息来判定语言模型中的全局工作空间究竟是由谁执行计算的，可区分出忽略广播、纯转发广播、以及将广播与自身知识重组这三种不同情况，并在训练后的工作空间模型中同时发现了这三种结果。

    

    全局工作空间理论（GWT）是意识研究的主流理论之一，该理论认为，当信息被广播到使用该信息的专门化过程时，信息便成为有意识的。然而，在网络中构建一个工作空间并不能揭示训练后的网络究竟用它计算了什么：一个报告答案的模块可能将广播消息与自身知识相结合，也可能仅仅转发在其他地方计算出的答案，而任务准确率和消融实验无法区分这些情况。我们引入了一种因果协议，用于识别模块通信时究竟由谁执行计算。其核心测试是从另一个输入移植一条消息，其选择方式使得忽略、转发和重组该消息分别会预测出三种不同的答案。在专为转发或重组而构建的网络中，该测试在每种情况下都能正确判定其计算方式。在经过训练的工作空间模型中，该协议发现了全部三种结果：其中一个模型的广播内容几乎未被使用，而在两个仅在训练数据上存在差异的模型中……（摘要原文在此处截断）

    arXiv:2611.26018v1 Announce Type: cross  Abstract: Global Workspace Theory (GWT), a leading theory of consciousness, holds that information becomes conscious when it is broadcast to specialized processes that use it. Building a workspace into a network does not reveal what the trained network computes with it: a module that reports an answer may have combined a broadcast message with its own knowledge, or merely forwarded an answer computed elsewhere, and task accuracy and ablation cannot tell these apart. We introduce a causal protocol that identifies who computes when modules communicate. Its central test transplants a message from another input, chosen so that ignoring, relaying and recombining it predict three different answers. On networks built to relay or to recombine, the test assigns the right computation in every case. Among trained workspace models it finds all three outcomes: one leaves its broadcast nearly unused, and of two models that differ only in their training data, 
    
[^75]: 当智能体表现不同时：利用条件分布检验定位人类与智能体的行为差异

    When Agents Play Differently: Localizing Human--Agent Behavioral Divergence with Conditional Distributional Testing

    [https://arxiv.org/abs/2611.26012](https://arxiv.org/abs/2611.26012)

    提出了一种基于分层条件分布检验的框架，利用情境单元内的非参数差异统计量和子群规则精确定位智能体与人类行为产生分歧的具体场景，并在商业三消游戏中发现了聚合指标无法察觉的模式级行为差异。

    

    评估人工智能智能体是否表现得像人类，不仅需要比较平均结果，还需要在多样化情境中比较完整的行为分布。我们提出了一个分层框架，在情境单元内计算非参数差异统计量，并通过发现子群规则来识别智能体行为在哪些方面偏离了人类玩法。我们证明，由胜负结果主导的对局级指标除了均值比较之外几乎不提供额外信号，而走子效率等过程级指标则能揭示真实的行为差异。将该框架应用于一款商业三消游戏，它揭示了聚合指标无法察觉的模式：一个游戏测试机器人在某些游戏模式中与人类存在显著差异，而在其他模式中未检测到明显分歧。我们开源发布了该实现，以支持可复现性和推广使用。

    arXiv:2611.26012v1 Announce Type: new  Abstract: Evaluating whether artificial agents behave like humans requires comparing not just average outcomes but entire behavioral distributions across diverse contexts. We propose a stratified framework that computes nonparametric discrepancy statistics within contextual cells, and discovers subgroup rules identifying where agent behavior departs from human play. We show that session-level metrics dominated by win/loss outcomes offer little signal beyond mean comparisons, whereas process-level metrics such as move efficiency reveal genuine behavioral differences. Applied to a commercial match-3 game, it uncovers patterns invisible to aggregate metrics: a playtesting bot is diverging substantially from humans in certain game modes, while showing no detectable divergence in others. We release our implementation as open source to support reproducibility and adoption.
    
[^76]: System-One RSI：通过引导式练习为交互式智能体构建肌肉记忆

    System-One RSI: Building Muscle Memory for Interactive Agents through Guided Practice

    [https://arxiv.org/abs/2611.26007](https://arxiv.org/abs/2611.26007)

    本文提出System-One RSI双循环框架，通过高频OODA内循环（决策模型加决策树，无需推理模型即可输出亚秒级定时动作）与推理智能体团队执行的离线递归改进外循环（修订提示词与决策树、从交互记录中生成监督信号），让交互式智能体像运动员一样通过引导式练习建立“肌肉记忆”，从而同时获得强任务理解能力与亚秒级实时控制。

    

    推理模型具备强大的多模态任务理解能力，但由于深思熟虑会增加决策延迟，它们难以实现亚秒级的实时控制。快速决策模型能够满足响应时间窗口的要求，然而仅靠提示词和少样本示例难以提供新任务所需的细粒度经验。受教练指导的运动学习启发，本文提出了System-One递归自我改进，该方法耦合了两个循环。高频在线内循环是一个“观察—定位—决策—行动”（OODA）循环：决策模型将每个视觉观察和任务问题映射为结构化判断，再由决策树将其转换为定时动作，全程无需调用推理模型。离线外循环则是由一组推理智能体执行的递归改进循环：该团队修订决策模型的提示词与决策树，捕获交互记录，从这些记录中推导监督信号，并训练……

    arXiv:2611.26007v1 Announce Type: new  Abstract: Reasoning models provide strong multimodal task understanding but struggle with sub-second control because deliberation increases decision latency. Fast decision models can meet the response window, yet prompting and few-shot examples do not readily provide the fine-grained experience required by a new task. Inspired by coached motor learning, this paper presents System-One Recursive Self-Improvement (System-One RSI), which couples two loops. The high-frequency online inner loop is an Observe, Orient, Decide, Act (OODA) cycle: a decision model maps each visual observation and task question to structured judgments, and a decision tree converts them into timed actions without invoking a reasoning model. The offline outer loop is a recursive improvement cycle carried out by a team of reasoning agents. The team revises the decision-model prompt and decision tree, captures interaction records, derives supervision from those records, and train
    
[^77]: AI智能体可以在超过一千个CAD参数和非线性仿真的条件下重新设计工程零件

    AI agents can redesign engineering parts with more than a thousand CAD parameters and nonlinear simulations

    [https://arxiv.org/abs/2611.26003](https://arxiv.org/abs/2611.26003)

    该论文提出，通用大语言模型智能体（GPT-6 Astra）能够在仅需少数非线性有限元仿真求解的情况下，对包含上千个CAD参数的真实工程零件进行整体性的重新设计，突破了传统优化器最多只能满足三个关键条件中两个的局限。

    

    真实工程部件的仿真驱动设计必须同时满足三个条件：它需要在很少的求解次数内完成，因为对一个3D零件进行非线性有限元仿真需要数小时；它的设计空间包含大量取值，即零件CAD模型中的数百到数千个参数值；而且更优的设计往往需要重新设计，即对零件各部分比例进行协调一致的改动，而非对表面进行局部打磨。自动化优化器最多只能同时满足其中两个条件：经典优化器（无论是贝叶斯方法还是无导数方法）需要人工将设计缩减为少数几个参数，且在全部参数上需要远多于可行次数的求解；工业形状优化器虽然每个周期只需一次求解即可移动网格上的所有表面节点，但只能以小幅步长进行“打磨”而非“重新设计”。我们探究一个通用大语言模型（LLM）能否通过将其自身知识与仿真所显示的信息相结合，同时满足这三个条件。GPT-6 Astra对一个镁合金支架进行编辑……

    arXiv:2611.26003v1 Announce Type: new  Abstract: Simulation-driven design of real engineering components has to meet three conditions at once. It runs on few solves, since a nonlinear finite-element simulation of a 3D part takes hours; its design space has many values, the hundreds to thousands of values of the part's CAD model; and a better design often needs a redesign, coordinated changes of the part's proportions rather than local polishing of its surface. Automated optimisers meet at most two of these conditions. Classical optimisers, Bayesian or derivative-free, need a person to cut the design down to a few parameters, and on all the values they need far more solves; industrial shape optimisers move every surface node of the mesh in one solve per cycle, but only in small steps that polish rather than redesign. We ask whether a general-purpose large language model (LLM) can meet all three by joining what it knows with what the simulation shows. GPT-6 Astra edits a magnesium stent 
    
[^78]: 从AI代理模型到工程工具：迈向碰撞CAE中纤维取向预测的生产化应用

    From AI Surrogate to Engineering Tool: Toward Production Use of Fiber-Orientation Prediction in Crash CAE

    [https://arxiv.org/abs/2611.25991](https://arxiv.org/abs/2611.25991)

    本田将此前提出的纤维取向预测AI代理模型（将每个零件约25小时的预测工作缩短至约3分钟）经过从二维平板到生产零件的分阶段验证和专家评审，成功转化为碰撞CAE设计工作流程中实际投入使用的工程工具。

    

    玻璃纤维增强塑料零件的碰撞仿真对提供给材料模型的纤维取向场高度敏感。在传统工作流程中，该取向场来自树脂流动分析，需要成型专家的支持，每个零件大约需要25小时的工作量。我们此前提出了一种代理模型，通过在大约3分钟内预测取向场来解决这一瓶颈。本文报告了将该模型转变为本田设计工作流程中可用工程工具的实施、验证与运行设计。我们展示了从二维平板到实际生产零件的分阶段验证、依据预定义工程角度误差准则开展的评估，以及2025年3月批准其用于设计工作的专家评审；此后该系统已移交至碰撞安全部门，并在定向断裂预测场景中处于初步的、有限范围的使用阶段。所开发的系统结合了注塑浇口（摘要在此处被截断）……

    arXiv:2611.25991v1 Announce Type: new  Abstract: Crash simulations of glass-fiber-reinforced plastic parts are highly sensitive to the fiber-orientation field supplied to the material model. In conventional workflows, that field comes from resin-flow analysis, which requires molding-specialist support and takes about 25 hours of work per part. We previously proposed a surrogate model that addresses this bottleneck by predicting the orientation field in about 3 minutes. This paper reports the implementation, validation, and operational design that turned the model into an engineering tool for Honda design workflows. We present staged validation from 2D plates to production parts, evaluation against a predefined engineering angular-error criterion, and the expert review that approved design use in March 2025; the system has since been transferred to the crash-safety group and is in initial, limited use in directed fracture-prediction scenarios. The developed system combines injection-gat
    
[^79]: 策略改进引导的持续强化学习重放方法

    Policy Improvement-Guided Replay for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.25989](https://arxiv.org/abs/2611.25989)

    该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。

    

    持续强化学习（CRL）要求模型在获得新能力的同时保持之前任务上的性能。传统持续学习中常见的面向稳定性的视角强调知识保留和干扰缓解。作为对这一视角的补充，我们通过策略改进——强化学习的一个基本原则——重新审视视觉语言模型的持续强化学习，以保留已获得的性能提升。利用性能差分引理，我们通过当前策略在历史任务分布上的期望优势来刻画历史性能变化，从而启发使用当前策略对历史提示进行重放。为了分配有限的重放记忆，我们提出了对比似然重放（CoLiR），该方法将最终任务策略对正确回答的似然增益与初始策略对错误回答的似然降低进行对比。我们的分析将正确回答与……

    arXiv:2611.25989v1 Announce Type: cross  Abstract: Continual reinforcement learning (CRL) requires models to acquire new capabilities while retaining performance on previous tasks. A stability-oriented perspective common in conventional continual learning emphasizes knowledge preservation and interference mitigation. Complementing this perspective, we revisit CRL for vision-language models through policy improvement, a fundamental principle of RL, to preserve acquired performance gains. Using the performance difference lemma, we characterize historical performance changes through the current policy's expected advantages on historical task distributions, motivating replay of historical prompts with the current policy. To allocate limited replay memory, we propose Contrastive Likelihood Replay (CoLiR), which contrasts likelihood gains on correct responses from the final task policy with likelihood reductions on incorrect responses from the initial policy. Our analysis links correct-respo
    
[^80]: 与AI同速演进的法律：机器时代的关键法律基础设施——法律规制下的自主性架构（AULA）

    Law at the Speed of AI: Critical Legal Infrastructure for the Machine Age: Autonomy Under Law Architecture (AULA)

    [https://arxiv.org/abs/2611.25985](https://arxiv.org/abs/2611.25985)

    该论文指出自主AI智能体的全球部署已超出治理基础设施的承载能力，协议层的成熟无法带来治理层的法律可辨识性，由此造成责任归属与权利分配的双向缺口，并提出AULA架构作为机器时代所需的关键法律基础设施。

    

    自主AI智能体在组织内部及跨组织的全球部署规模，已经超出了现有可用来约束它们的治理基础设施。智能体通信与认证协议——主要是模型上下文协议（Model Context Protocol）和智能体对智能体协议（Agent2Agent Protocol）——正在快速成熟，但协议层面的成熟并不能在治理层面产生法律可辨识性。由部署组织编写的运行时强制机制只能限定AI智能体可以访问的资源，而非其行动所依据的权限。法律可辨识性是指法律体系通过将活动映射到法律人格、财产、权利与责任之上，从而识别、解释和规制该活动的能力。现有的（模拟时代的）法律基础设施无法为自主部署提供这种能力。由此产生的缺口是双向的：它既涉及当AI智能体造成损害时的责任归属问题，也涉及权利的分配问题，包括数据权利与经济权利等……

    arXiv:2611.25985v1 Announce Type: cross  Abstract: The global deployment of autonomous AI agents within and across organisations exceeds the governance infrastructure available to constrain it. Agent communication and authentication protocols, principally the Model Context Protocol and the Agent2Agent Protocol, are maturing quickly, but maturity at the protocol layer does not produce legal legibility at the governance layer. Runtime enforcement written by the deploying organisation fixes the resources an AI agent can reach, not the authority under which it acts. Legal legibility is the capacity of a legal system to recognise, interpret and regulate activity by mapping it onto legal personhood, property, rights and liabilities. Existing (analogue) legal infrastructure cannot supply it for autonomous deployment. The resulting deficit is bidirectional. It concerns the attribution of liability when an AI agent causes harm, and the allocation of rights, including data and economic rights, g
    
[^81]: 多人游戏中的安全子博弈求解

    Safe Subgame Resolving in Multiplayer Games

    [https://arxiv.org/abs/2611.25981](https://arxiv.org/abs/2611.25981)

    该论文证明了首个针对非相关多人博弈子博弈求解的条件性安全定理，表明已求解策略的最佳响应差距相对蓝图最多增加四项（其中两项源于求解器的未实现承诺子博弈价值与求解误差），从而为多人博弈中的实时策略细化提供了事后安全证明证书。

    

    扑克及其他具有隐藏信息的博弈规模庞大，无法一次性完整求解。强大的系统会先离线计算完整的蓝图策略，再在实际到达的子博弈中对其进行实时细化。在双人零和博弈中，安全定理保证子博弈求解绝不会增加对手通过利用已求解策略相对于蓝图策略所能获得的收益。然而，当三个或更多玩家独立行动时，目前尚不存在这样的保证，启发式求解可能会增大策略组合的最佳响应差距。我们证明了首个针对非相关多人子博弈求解的条件性安全定理，这是一个适用于任何已交付策略更新的事后证明证书。已求解策略组合的最佳响应差距相对蓝图最多超出四项：其中两项与求解器相关，分别是蓝图承诺但未能实现的子博弈价值（A_i^S）以及求解器的子博弈求解误差（ε_i）；另外两项是……（摘要原文在此处截断）

    arXiv:2611.25981v1 Announce Type: cross  Abstract: Poker and other games with hidden information are too large to solve in one pass. Strong systems compute a complete blueprint strategy offline and refine it in real time inside the reached subgame. In 2p0s games, a safety theorem guarantees that subgame resolving never increases how much an adversary can gain by exploiting the resolved strategy relative to the blueprint strategy. With three or more players acting independently, no such guarantee exists yet, and heuristic resolving can increase the profile's best-response gap. We prove the first conditional safety theorem for uncorrelated multiplayer subgame resolving, a post-hoc certificate for any delivered update. The best-response gap of the resolved profile exceeds that of the blueprint by at most four terms. Two concern the resolver: the subgame value the blueprint promised without realizing ($A_i^{S}$) and the resolver's subgame solve error ($\varepsilon_i$). The other two are th
    
[^82]: 面向3D虚拟试穿的多视图Transformer与几何引导多模态ControlNet

    Multi-View Transformer with Geometry-Guided Multi-Modal ControlNet for 3D Virtual Try-On

    [https://arxiv.org/abs/2611.25979](https://arxiv.org/abs/2611.25979)

    该论文提出GS-VTON-MVT三阶段框架，通过具备完整跨视图自注意力的多视图修补Transformer学习视图间对应关系，并利用几何引导的多模态ControlNet对扩散编辑器进行条件控制，从而实现视角一致、细节保真的3D高斯泼溅虚拟试穿。

    

    编辑一个人的3D高斯泼溅（3DGS）捕捉以使其穿上新服装，要求每个渲染视图在相同的服装细节上保持一致。然而，现有的3D虚拟试穿（VTON）流程在很大程度上独立地编辑各个训练视图，并依靠手工设计的启发式方法（如手动选择的参考视图和固定的注意力混合权重）来融合它们。由此产生的不一致性被固化到3D表示中，在新视角下表现为纹理漂移、印花模糊以及不合理的几何形状。我们提出了GS-VTON-MVT，这是一个结合隐式与显式3D推理以实现一致性3DGS试穿的三阶段框架。（i）多视图修补Transformer通过完整的跨视图自注意力机制联合编辑一组训练视图，使跨视图对应关系得以学习而非人为强加。（ii）几何引导的多模态ControlNet将扩散编辑器以服装图像以及……（原文摘要在此处截断）

    arXiv:2611.25979v1 Announce Type: cross  Abstract: Editing a 3D Gaussian Splatting (3DGS) capture of a person so that they wear a new garment requires every rendered view to agree on the same garment details. Existing 3D virtual try-on (VTON) pipelines, however, edit training views largely independently and fuse them with hand-designed heuristics such as manually chosen reference views and fixed attention-blending weights. The resulting inconsistencies are baked into the 3D representation and surface as texture drift, blurred prints, and implausible geometry in novel views. We present GS-VTON-MVT, a three-stage framework that combines implicit and explicit 3D reasoning for consistent 3DGS try-on. (i) A multi-view inpainting transformer jointly edits a set of training views with full cross-view self-attention, so that cross-view correspondence is learned rather than imposed. (ii) A geometry-guided multi-modal ControlNet conditions the diffusion editor on the garment image together with 
    
[^83]: 开源权重大型语言模型在不同给定食物重量下的能量估计：一项剂量-反应评估

    Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation

    [https://arxiv.org/abs/2611.25976](https://arxiv.org/abs/2611.25976)

    七个开源大语言模型在估计食物能量时无法随给定重量成比例缩放（弹性仅0.548-0.979），导致小份量被高估、大份量被低估，而采用“模型自报能量密度×重量”的两步法可基本消除该误差。

    

    我们检验了开源权重大型语言模型（LLM）的能量估计是否随精确给定的食物重量成比例变化，以及通过“能量密度×重量”的计算方式能否消除误差。在一项预先设定的协议下，七个开源权重LLM对从美国农业部膳食研究食物与营养素数据库中随机抽取的168种食物，在九个给定重量（为默认量的0.25倍至4倍）下估计其能量。主要结局指标是估计值相对于重量的弹性（1表示成比例缩放）。两步估计法是在代码中将每个模型自己给出的能量密度乘以重量。结果显示，所有七个模型都压缩了其估计值，弹性范围为0.548至0.979（所有95%置信区间均低于1）；其中四个模型压缩严重（0.548至0.709）。这四个模型在默认量四分之一时高估能量47%至130%，在默认量四倍时低估能量20%至34%。提供能量密度后可恢复接近成比例的缩放……

    arXiv:2611.25976v1 Announce Type: cross  Abstract: We tested whether energy estimates of open-weight large language models (LLMs) scale proportionally with an exactly stated food weight, and whether computing energy density times weight removes the error. Under a prespecified protocol, seven open-weight LLMs estimated the energy of 168 randomly drawn foods from the USDA Food and Nutrient Database for Dietary Studies at nine stated weights, from 0.25 to 4 times the default amount. The primary outcome was the elasticity of the estimate with respect to weight (1: proportional scaling). A two-step estimate multiplied each model's own stated energy density by the weight in code. All seven models compressed their estimates, with elasticities from 0.548 to 0.979 (every 95% CI below 1); four compressed severely (0.548 to 0.709). These four overestimated a quarter of the default amount by 47% to 130% and underestimated four times it by 20% to 34%. Supplying the density restored near-proportiona
    
[^84]: 拉丁美洲的网络搜索：语言模型使用的新闻来源比搜索引擎更少本地化且多样性更低

    Web search in Latin America: language models use less local and diverse news than search engines

    [https://arxiv.org/abs/2611.25955](https://arxiv.org/abs/2611.25955)

    本研究通过对拉丁美洲九国时事的大规模查询实验发现，基于语言模型的搜索返回的新闻来源多样性低于传统搜索引擎，且本地新闻占比因用户类型而表现不一致，揭示了AI搜索可能加剧地理偏见的风险。

    

    AI平台正在成为人们获取新闻和时事的主要入口。这一转变可能会加剧现有的地理偏见，即偏向来自较富裕国家的叙述而忽视全球南方国家的声音。这对拉丁美洲尤为相关，因为该地区在全球南方的AI使用量中占比最大。我们研究了当用户从传统搜索引擎转向基于大语言模型的搜索时，关于拉丁美洲的信息会发生怎样的变化。我们构建了一个涵盖九个拉丁美洲国家、包含1,901个时事事件的数据集，设计了贴近现实的查询，并将其提交给三个搜索引擎和四个语言模型。我们改变了查询语言和用户所在位置，以代表本地用户、海外侨民、游客和外国用户四种场景。在49,207次请求和8,834个独立域名中，语言模型返回的新闻来源多样性低于搜索引擎。在不同用户类型下，语言模型返回的国内新闻占比表现不一致……

    arXiv:2611.25955v1 Announce Type: cross  Abstract: AI platforms are becoming a major gateway to news and current events. This shift risks amplifying existing geographical biases that favour narratives from wealthier countries over those from the Global South. This is particularly relevant in Latin America, which accounts for the largest share of AI usage in the Global South. We examine how information about Latin America changes when users shift from conventional search engines to LLM-based search. We construct a dataset of 1,901 current events across nine Latin American countries, craft realistic queries, and submit them to three search engines and four language models. We vary query language and user location to represent local, diaspora, tourist, and foreign user scenarios. Across 49,207 requests and 8,834 unique domains, language models return a less diverse set of news sources than search engines. Across user types, the share of domestic news returned by language models is inconsi
    
[^85]: GaussTrans：一种用于多层折射传输的高斯界面框架

    GaussTrans: A Gaussian Interface Framework for Multi-Layer Refractive Transport

    [https://arxiv.org/abs/2611.25950](https://arxiv.org/abs/2611.25950)

    提出GaussTrans框架，通过高斯-场混合表面、传输耦合细化和基元引导隐式求交三个组件，实现高阶多层透明场景中精确的光传输与重建。

    

    我们研究多层透明界面的表面建模，以实现精确的光传输。在重建具有大量层数的透明介质时，几何结构、光路与观测结果中的误差相互耦合，使这一问题极具挑战性。一些现有技术通过减少层数或简化光路来处理透明场景的重建，但这限制了它们在高阶透明场景中的直接适用性。我们提出GaussTrans，一个面向高阶透明场景光传输的紧凑框架，由三个组件构成：高斯-场混合表面、传输耦合细化以及基元引导的隐式求交。高斯-场混合表面采用结合高斯表面与场的混合方法来重建不同界面；传输耦合细化通过评估（摘要在此处截断）

    arXiv:2611.25950v1 Announce Type: new  Abstract: We investigate surface modeling of multi-transparent interfaces to achieve accurate light transmission. The interplay of errors in geometric structure, light paths, and observation results during the reconstruction of transparent media with a high number of layers makes this a challenging problem. Some existing techniques address the reconstruction of transparent scenes by reducing the number of layers or simplifying light paths; however, this limits their direct applicability to high-order transparent scenes. We propose GaussTrans, a compact framework for light transport in high-order transparent scenes, consisting of three components: Gaussian-Field Hybrid Surfaces, Transport-Coupled Refinement, and Primitive-Guided Implicit Intersection. Gaussian-Field Hybrid Surfaces use a hybrid approach combining Gauss surfaces and fields to reconstruct different interfaces. Transport-Coupled Refinement reduces coupling errors by evaluating the ent
    
[^86]: Fast-JEPA：推进潜在世界模型的成功率-延迟前沿

    Fast-JEPA: Advancing the Success-Latency Frontier for Latent World Models

    [https://arxiv.org/abs/2611.25927](https://arxiv.org/abs/2611.25927)

    Fast-JEPA通过构建目标导向计划引导搜索并直接预测完整候选动作序列的结果来避免递归状态展开，以比INTACT快2.41倍的规划速度和96.50%的平均成功率推动了潜在世界模型的成功率-延迟帕累托前沿。

    

    潜在世界模型用于预测动作序列的结果，但在测试阶段通过递归展开来比较候选计划仍然代价高昂。我们提出了Fast-JEPA，这是一个潜在世界模型规划框架，推动了预测搜索的成功率-延迟帕累托前沿。Fast-JEPA重新组织预测计算方式，以在低规划成本下支持有效的动作序列搜索。它构建一个目标导向的计划来引导搜索，并直接预测完整候选序列的结果以实现高效比较，从而避免了细化过程中的递归状态展开。在PushT、Reacher、Cube和TwoRoom任务上，Fast-JEPA实现了96.50%的平均成功率，规划速度比INTACT快2.41倍。搜索预算扫描实验表明，其在不同候选数量和细化深度下均能保持有竞争力的成功率和低延迟。我们进一步引入了DynaGoal，一个用于评估世界模型规划在动态分布外场景下表现的基准。

    arXiv:2611.25927v1 Announce Type: cross  Abstract: Latent world models predict the outcomes of action sequences, but comparing candidate plans through recursive rollout remains expensive at test time. We introduce Fast-JEPA, a latent world model planning framework that advances the success-latency Pareto frontier of predictive search. Fast-JEPA reorganizes predictive computation to support effective action-sequence search at low planning cost. It constructs a goal-directed plan to guide search and directly predicts the outcomes of complete candidates for efficient comparison, avoiding recursive state rollout during refinement. Across PushT, Reacher, Cube, and TwoRoom, Fast-JEPA achieves 96.50% average success rate with 2.41x faster planning than INTACT. Search-budget sweeps show that its competitive success and low latency extend across candidate counts and refinement depths. We further introduce DynaGoal, a benchmark for evaluating world-model planning under dynamic out-of-distributio
    
[^87]: ROAM-GRPO：通过路由离模方法挖掘在GRPO中强制执行推理多样性

    ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode Approach Mining

    [https://arxiv.org/abs/2611.25926](https://arxiv.org/abs/2611.25926)

    提出ROAM-GRPO算法，通过将训练中已被良好解决的问题路由到“离模”轨道以激励模型探索多样化推理策略，并经LLM裁判验证后自蒸馏进策略，有效缓解了RLVR后训练导致的输出多样性坍缩问题。

    

    监督微调（SFT）与可验证奖励强化学习（RLVR）相结合已成为在复杂推理任务上对大语言模型进行后训练的主流方案，但已被证明会使策略坍缩到低多样性模式。大语言模型的重复输出几乎完全相同，因此pass@k准确率随之收敛或下降。我们提出了ROAM-GRPO（面向GRPO的路由离模方法挖掘），这是对GRPO算法的一种增强，它激励模型探索不同的推理策略，旨在提高评估时的输出创造性。ROAM-GRPO在训练过程中检测已被良好解决的问题，并将其转移到“离模”轨道。模型被提示为这些问题生成不同的解决方案策略，这些策略由LLM裁判验证其逻辑有效性和独特性，随后自蒸馏到策略中。在MATH lvl5数据集的1000个问题子集上训练Qwen3.5-2b时，ROAM-GRPO

    arXiv:2611.25926v1 Announce Type: new  Abstract: Supervised fine-tuning (SFT) paired with Reinforcement Learning with Verifiable Rewards (RLVR) has become the dominant recipe for post-training LLMs on complex reasoning tasks, but has been shown to collapse the policy into a low-diversity mode. Repeated LLM outputs are nearly identical, and pass@k accuracy converges or degrades as a result. We propose ROAM-GRPO (Routed Off-mode Approach Mining for GRPO), an augmentation to the GRPO algorithm which incentivizes exploration of distinct reasoning strategies and aims to increase output creativity during evaluation. ROAM-GRPO detects well-solved problems during training and transfers them to an "off-mode" track. The model is prompted to produce distinct solution strategies to such problems, which are verified for logical validity and distinctness by an LLM judge and subsequently self-distilled into the policy. Training Qwen3.5-2b on a 1,000-problem subset of the MATH lvl5 dataset, ROAM-GRPO 
    
[^88]: 多解一答：搜索顺序塑造语言模型的默认解

    Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models

    [https://arxiv.org/abs/2611.25916](https://arxiv.org/abs/2611.25916)

    论文发现语言模型在多解约束谜题中高度集中于同一个默认解，其行为如同按特定顺序遍历的搜索求解器——优先访问初始候选最少的实体这一搜索顺序最能预测模型输出，且改变给定的顺序会相应改变模型的默认解。

    

    当一个约束谜题存在多个正确答案时，语言模型往往给出同一个答案。我们在符号谜题以及解集完全已知的多解ZebraLogic变体上，研究了来自三个家族的四个开放权重模型。重复采样高度集中于某一个默认解，其集中程度远超均匀采样，而且不同模型常常共享这个默认解。然而，当新增线索能唯一确定某个解时，即使该解很少被采样，模型通常也能将其给出。在留出的符号谜题上，我们比较了两种实体访问策略：按名称顺序访问实体，与优先访问初始候选最少的实体。两种搜索都按给定的顺序尝试取值。在二者预先指定的预测不同的地方，后者在所有四个模型中匹配了更大比例的回答，领先幅度达26至40个百分点。在两个测试平台上，反转给定的取值顺序或指定搜索顺序，都会使模型的回答偏向新顺序所偏好的解。随机化当前的……

    arXiv:2611.25916v1 Announce Type: cross  Abstract: When a constraint puzzle has several correct answers, language models tend to give the same one.We study four open-weight models from three families on symbolic puzzles and multi-solution ZebraLogic variants with exactly known solution sets.Repeated samples concentrate on one default solution far beyond uniform sampling, and different models often share it.Yet models usually produce even rarely sampled solutions when added clues uniquely identify them.On held-out symbolic puzzles, we compare visiting entities in name order with visiting those with the fewest initial candidates first. Both searches try values in the stated order. Where their pre-specified predictions differ, the latter matches a larger share of responses in all four models, by 26 to 40 percentage points.Reversing the stated value order or specifying a search order shifts responses towards the solution that the new order favours, on both testbeds. Randomising the present
    
[^89]: 面向智能体工作负载的大语言模型推理平台的验证驱动部署

    Validation-driven Deployment of LLM Inference Platforms for Agentic Workloads

    [https://arxiv.org/abs/2611.25905](https://arxiv.org/abs/2611.25905)

    本文提出了一个面向智能体LLM服务的端到端评估流水线，证明重放捕获的追踪记录能以高保真度（95分位下1.18倍，而实时重跑达4.98倍）重现原始智能体会话负载，且实时、重放和合成工作负载在代码覆盖上相互补充。

    

    企业正在将智能体工作负载从封闭的API迁移到自身的开放权重模型部署上。因此，从小规模测试、容量规划与预生产到生产的全生命周期验证变得愈发重要。通常情况下，系统上线部署是使用临时性工具来评估的，而这些工具的准确性很少得到检验。我们提出了一个面向智能体大语言模型服务的端到端评估流水线，并通过实验为连接其各阶段的设计选择提供实证依据。我们的实验表明，重放捕获的追踪记录能够重现原始智能体会话的单会话负载：在第95百分位上，重放的输入令牌数保持在记录值的1.18倍以内，而实时重跑则达到4.98倍，因此少量记录的会话即可产生可重复的单会话令牌负载。对vLLM的代码覆盖率分析进一步表明，在该配置下所评估的实时、重放和合成工作负载是互补的：每种负载都执行不同的Python代码段。

    arXiv:2611.25905v1 Announce Type: new  Abstract: Enterprises are moving agentic workloads from closed APIs to their own open-weight model deployments. For this reason, lifecycle-wide validation, from small-scale testing through sizing and pre-production to production, becomes all the more important. Typically, rollouts are evaluated with ad hoc tools whose accuracy is rarely checked. We present an end-to-end evaluation pipeline for agentic LLM serving and empirically ground the design choices that connect its stages. Our experiments show that replaying captured traces reproduces the per-session load of the original agent sessions: at the 95th percentile replay stays within 1.18x of the recorded input tokens while live re-runs reach 4.98x, so a small set of recorded sessions yields a repeatable per-session token load. A code-coverage analysis of vLLM further shows that the evaluated live, replayed, and synthetic workloads are complementary in this configuration: each exercises Python se
    
[^90]: 从案例到准则：复用评估知识以生成任务特定的评分细则

    From Cases to Criteria: Reusing Evaluation Knowledge for Task-Specific Rubric Generation

    [https://arxiv.org/abs/2611.25903](https://arxiv.org/abs/2611.25903)

    提出 RubricDuo，通过构建链接评估维度、共享概念、局部概念、准则与查询的评分细则知识图谱来复用历史评估知识，从而为 LLM 评估自动生成任务特定、细粒度且结构完整的查询级评分细则。

    

    查询特定的评分细则能够使大语言模型（LLM）的评估更加细粒度和透明，而生成高质量的评分细则既需要识别目标查询的评估范围，又需要明确具体的检查项。历史评分细则提供了可复用的评估经验，然而现有方法要么复用完整的查询-评分细则案例，要么复用聚合后的准则与维度，未能将重复出现的评估需求显式地与其准则层面的证据以及源查询上下文相关联。这可能导致保留无关的任务细节，或丢失更广泛的评估结构。在本文中，我们提出了 RubricDuo，它将历史查询-评分细则对组织成一个评分细则知识图谱，链接评估维度、共享概念、局部概念、准则和查询，其中跨查询的可复用评估需求通过共享概念进行抽象。给定一个目标查询，RubricDuo 首先结合与目标查询相关的概念……

    arXiv:2611.25903v1 Announce Type: new  Abstract: Query-specific rubrics make LLM evaluation more fine-grained and transparent, and generating high-quality rubrics requires both identifying the target query's evaluation scope and specifying concrete checks. Historical rubrics provide reusable evaluation experiences, yet existing approaches reuse either complete query-rubric cases or aggregated criteria and dimensions without explicitly connecting recurring evaluation requirements to their criterion-level evidence and source-query context. This can preserve irrelevant task details or lose the broader evaluation structure. In this paper, we introduce RubricDuo, which organizes historical query-rubric pairs into a Rubric Knowledge Graph linking evaluation dimensions, shared concepts, local concepts, criteria, and queries, where reusable evaluation requirements across queries are abstracted by shared concepts. Given a target query, RubricDuo first combines concepts relevant to the target qu
    
[^91]: 医学大语言模型中的可验证逻辑：智能体临床推理优化与神经符号接地的批判性综述

    Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding

    [https://arxiv.org/abs/2611.25898](https://arxiv.org/abs/2611.25898)

    这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。

    

    临床大语言模型通常以最终答案是否正确作为评判标准。然而，模型可能通过无效的推理得出流畅、看似合理的输出，而静态基准测试无法检测到这种情况；自信的幻觉、因果无效的推理以及虚假的人口统计学捷径都可能通过传统评估。本批判性综述依次涵盖四种范式：临床幻觉如何产生以及为什么静态指标会漏检它；智能体幻觉级联——即多智能体和工具增强系统会传播并放大早期推理错误；过程奖励建模作为步骤级监督；以及神经符号接地——形式逻辑引擎、本体推理机和知识图谱对神经输出施加可验证的约束。在其编码的知识范围内，具有符号否决层的系统能够提供相对于其编码知识成立的保证，并在安全性指标上报告了显著的提升。

    arXiv:2611.25898v1 Announce Type: cross  Abstract: Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measu
    
[^92]: 解释的代价：要求可验证产物损失了什么，又换回了什么

    The Price of an Explanation: What Demanding Verifiable Artifacts Costs, and What It Buys Back

    [https://arxiv.org/abs/2611.25896](https://arxiv.org/abs/2611.25896)

    该论文在ARC-AGI-2上量化了要求LLM生成可执行可验证产物（程序）的代价（得分从77%降至57%），并指出真正的瓶颈在于让解释“泛化”而非“找到”解释——因为多数仅由只答组解出的任务中，程序组其实已拥有能复现全部演示、却仍在测试上失败的程序。

    

    生成模型会产生许多候选解，却没有任何内在信号可用于对它们排序。一种缓解手段是要求提供“可验证产物”：即一个可对照现有证据执行的程序。人们普遍担心，要求可执行验证会在解答质量和计算预算两方面带来成本。我们在包含120个任务的ARC-AGI-2评测集上测量了这两类成本。要求LLM提供可验证产物后，其得分从120个任务中的93个降至68个（77%降至57%）。在损失的这25个任务中，有21个任务在我们运行过的任何程序生成配置下——无论何种预算、无论采样还是搜索——都未能解决；然而，在仅由“只给答案”组解决的28个任务中，有27个任务上某些程序组所持有的程序能够复现全部演示示例，却仍在测试上失败。瓶颈在于获得能够“泛化”的解释，而非“找到”某个解释。对两个……（原文摘要在此处截断）

    arXiv:2611.25896v1 Announce Type: new  Abstract: Generative models produce many candidate solutions but provide no intrinsic signal by which to rank them. One mitigation is to demand a \emph{verifiable artifact}: a program that can be executed against the available evidence. A common concern is that requiring executable verification incurs costs in both solution quality and computational budget. We measure these two costs on the $120$-task ARC-AGI-2 evaluation set. Demanding a verifiable artifact from an LLM reduced the score from $93$ to $68$ of $120$ tasks ($77\%$ to $57\%$). Of those $25$ tasks, $21$ were reached by no program-producing configuration we ran, at any budget, by sampling or by search; yet for $27$ of the $28$ tasks that only the answer-only arm solved, some program arm held a program that reproduced every demonstration and still failed the test. The bottleneck is obtaining an explanation that \emph{generalizes}, not finding one. A short evolutionary search over two alr
    
[^93]: 一种调度无法适用于所有情况：扩散语言模型的自适应推理调度

    One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models

    [https://arxiv.org/abs/2611.25895](https://arxiv.org/abs/2611.25895)

    本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。

    

    扩散语言模型通常在所有工作负载中使用固定的去噪调度。我们证明，在保持质量的前提下，最廉价的调度取决于任务、模型和采样器。我们提出了任务校准去噪分配（Task-Calibrated Denoising Allocation，TCDA），这是一种具有显式误差保证的部署规则。给定候选采样器、配对校准数据、效用容忍度以及防止退化的防护措施，TCDA 使用同步单侧自助法界限来认证哪些候选方案同时满足效用和防护标准。它部署经过认证的最廉价候选方案，或回退到完整调度。部署不可行调式的概率渐近地至多为 alpha。候选方案包括早期前缀解码、状态复用续写、缩减时间网格和置信度阈值解码。在七个数据集和两个模型系列上，TCDA 认证了去噪器评估次数的显著减少：在 XSum 上减少 3.4 倍，在 WMT 上减少 2.0 倍。

    arXiv:2611.25895v1 Announce Type: new  Abstract: Diffusion language models typically use a fixed denoising schedule across workloads. We show that the cheapest schedule that preserves quality depends on the task, model, and sampler. We propose Task-Calibrated Denoising Allocation (TCDA), a deployment rule with an explicit error guarantee. Given candidate samplers, paired calibration data, a utility tolerance, and guardrails against degeneration, TCDA uses simultaneous one-sided bootstrap bounds to certify which candidates meet both the utility and guardrail criteria. It deploys the cheapest certified candidate or falls back to the full schedule. The probability of deploying an infeasible schedule is asymptotically at most alpha. Candidates include early prefix decoding, state-reusing continuation, reduced time grids, and confidence-threshold decoding. Across seven datasets and two model families, TCDA certifies reductions in denoiser evaluations by factors of 3.4 on XSum and 2.0 on WMT
    
[^94]: 用于太阳能光伏功率预测的可解释人工智能：从模型选择到可信、可复现的部署

    Explainable AI for Solar PV Power Forecasting: From Model Selection to Trustworthy, Reproducible Deployment

    [https://arxiv.org/abs/2611.25894](https://arxiv.org/abs/2611.25894)

    本文提出了一种将自动化模型搜索与可解释人工智能相结合的可复现光伏功率预测工作流程，并在葡萄牙波尔图一个16.32 kWp电站上验证，最佳模型达到11.6%的nRMSE，误差约为智能持续性基准的一半。

    

    机器学习（ML）现已成为预测太阳能光伏电站输出功率的标准工具，但已发表的精度数据往往难以比较，更难以令人信服。许多研究采用随机划分进行验证、不报告基准预测，并且在给出特征归因时不说明其能揭示和不能揭示的内容。本文综述了用于光伏功率预测与资源评估的主要机器学习方法家族，从树集成模型到深度模型和基础模型，并提出了一种将自动化模型搜索与可解释人工智能相结合的可复现工作流程。随后，该工作流程在一个公开可用的数据集上进行了测试：位于葡萄牙波尔图的一个16.32 kWp光伏电站，基于4 km数值天气预报网格进行提前一天的功率预测。在整个测试年度中，最佳模型达到了占装机容量11.6%的归一化均方根误差（nRMSE），约为智能持续性基准误差的一半。输入特征的选择改变了……

    arXiv:2611.25894v1 Announce Type: new  Abstract: Machine learning (ML) is now a standard tool for forecasting the output of solar photovoltaic (PV) plants, yet published accuracy figures are often hard to compare and harder to trust. Many studies validate with random splits, report no reference forecast, and present feature attributions without saying what they can and cannot show. This paper reviews the main families of ML methods for PV power forecasting and resource assessment, from tree ensembles to deep and foundation models, and sets out a reproducible workflow that combines automated model search with explainable AI. The workflow is then tested on an openly available dataset: a 16.32 kWp plant in Porto, Portugal, forecast one day ahead from a 4 km numerical weather prediction grid. Over a full test year, the best models reach a normalised root mean square error (nRMSE) of 11.6% of installed capacity, about half the error of smart persistence. The choice of input features changed
    
[^95]: Meta-RTL：面向RTL优化的智能体执行框架进化

    Meta-RTL: Evolving Agent Harnesses for RTL Optimization

    [https://arxiv.org/abs/2611.25891](https://arxiv.org/abs/2611.25891)

    Meta-RTL提出了一种基于EDA反馈的自动进化框架，能从极简循环出发自动生成RTL优化智能体的执行框架，摆脱人工设计搜索过程的束缚，在时序违例和面积优化上显著优于现有方法，并可直接迁移到未见设计。

    

    大语言模型使智能体能够利用EDA反馈来优化RTL（寄存器传输级）电路，但其可执行的搜索过程在很大程度上仍依赖人工设计。我们提出Meta-RTL，这是一个以EDA为基础的框架，它从一个极简循环出发自动进化出RTL优化执行框架，同时保持优化模型、EDA流程和每个设计的预算固定不变。与Dr. RTL相比，Meta-RTL*在20个开发设计上将总体WNS/TNS违例分别降低了7.8%/11.0%，并在其中70%的设计上实现了面积缩减。在无需进一步进化的情况下，Meta-RTL*可直接迁移至10个未见的IC-RTL设计，将总体WNS/TNS违例分别降低20.5%/14.0%，并将综合总面积降低12.6%。我们进一步刻画了进化后的执行框架中的三种搜索机制，并通过消融实验和案例研究评估了它们各自的贡献。

    arXiv:2611.25891v1 Announce Type: new  Abstract: Large language models enable agents to optimize RTL using EDA feedback, yet their executable search procedures remain largely hand- engineered. We present Meta-RTL, an EDA-grounded framework that evolves RTL optimization harnesses from a minimal loop while keeping the optimization model, EDA flow, and per-design budgets fixed. Compared with Dr. RTL, Meta-RTL* reduces aggregate WNS/TNS violations by 7.8%/11.0% across 20 development designs, with area reductions on 70% of them. Without further evolution, Meta-RTL* transfers to 10 unseen IC-RTL designs, reducing aggregate WNS/TNS violations by 20.5%/14.0% and total synthesized area by 12.6%. We further characterize three search mechanisms in the evolved harness and assess their contributions through ablations and a case study.
    
[^96]: 可信地理空间人工智能工作流：基于开源AIS数据的可复现海事分析与预测

    Trustworthy Geospatial AI Workflows: Reproducible Maritime Analytics and Forecasting on Open-Source AIS Data

    [https://arxiv.org/abs/2611.25884](https://arxiv.org/abs/2611.25884)

    该论文提出了基于开源数据处理与机器学习库damast的可复现海事地理空间AI工作流，通过系统处理公开AIS数据的数据质量问题，实现可信的海事分析与预测。

    

    公开可用的地理空间数据为数据驱动的海事分析与预测建模提供了越来越多的机会。然而，数据的可获得性并不一定意味着底层观测数据同样适用于下游AI应用。数据源与观测过程的特性，包括数据来源、报告规律性、空间覆盖范围、观测密度和缺失情况，会影响地理空间数据的解读与处理方式，并最终影响预测模型对其的使用。与此同时，可信的地理空间AI需要这样的分析工作流：其中这些数据处理决策能够被明确表达并加以复现。本文提出了一个可复现的海事地理空间AI工作流，该工作流围绕damast构建——这是Simula研究实验室开发的一个开源数据处理与机器学习库。以来自多个公共数据提供商的AIS（船舶自动识别系统）观测数据作为案例研究，我们对海事情境下的数据质量与工作流可复现性问题进行了探讨。

    arXiv:2611.25884v1 Announce Type: new  Abstract: Publicly available geospatial data provide increasing opportuni- ties for data-driven maritime analytics and predictive modelling. However, their availability does not necessarily imply that the underlying observations are equally suitable for downstream AI applications. Properties of the data source and observation pro- cess, including provenance, reporting regularity, spatial coverage, observation density, and missingness, can affect how geospatial data are interpreted, processed, and ultimately used by predictive models. At the same time, trustworthy geospatial AI requires ana- lytical workflows in which these processing decisions can be made explicit and reproduced. This paper presents a reproducible mar- itime geospatial AI workflow built around damast, an open-source data-processing and machine-learning library developed at Simula Research Laboratory. Using AIS observations from multiple public data providers as a case study, we de
    
[^97]: 面向量子纠错的进化式解码算法

    Evolving Decoding Algorithms for Quantum Error Correction

    [https://arxiv.org/abs/2611.25880](https://arxiv.org/abs/2611.25880)

    本文将qLDPC码的置信传播解码器重写为领域特定语言并用语法进化搜索，通过阻尼重参数化使解码器提速17.1至41倍（BP+OSD-0提速约103倍），且进化出的解码器比BP+OSD减少23%的逻辑错误。

    

    量子计算需要鲁棒性工程：含噪声的量子态被持续测量，而一个经典算法（解码器）必须推断出能可靠消除噪声的校正操作，且速度必须足够快以跟上测量节奏。我们用领域特定语言（DSL）重写了多个已发表的用于量子低密度奇偶校验（qLDPC）码的置信传播（BP）解码器，并采用语法进化在该语言上进行搜索。用该语言编写后，已发表解码器的参数一目了然，使我们能够在搜索之前就用阻尼对BP+OSD（有序统计解码）和BP+LSD（局部统计解码）解码器进行重新参数化，使每个解码器在更高错误率下提速17.1至41.0倍，而BP+OSD-0在[[72,12,6]]双变量自行车码上比ldpc软件包中调优的版本快约103倍。对该语言的进化搜索找到了改进时间-错误率前沿的解码器，其中一个解码器比BP+OSD少产生23%的逻辑错误。

    arXiv:2611.25880v1 Announce Type: cross  Abstract: Quantum computation entails robustness engineering: a noisy quantum state is measured continuously, and a classical algorithm (the decoder) must infer corrections that reliably remove the noise, fast enough to keep up. We write several published belief-propagation (BP) decoders for quantum low-density parity-check (qLDPC) codes in a Domain-Specific Language (DSL), and search over it with grammatical evolution. Written in the language, the published decoders show their parameters in plain view, allowing us to re-parameterise BP+OSD (ordered statistics decoding) and BP+LSD (localized statistics decoding) decoders with damping, before any search, making each 17.1--41.0$\times$ faster at a higher error rate, and BP+OSD-0 about 103 faster than the ldpc package's version tuned on the [[72,12,6]] bivariate-bicycle code. Evolution over the language finds decoders that improve the time-error front. One makes 23% fewer logical errors than BP+OSD
    
[^98]: 连贯的信念与观点传播产生更多回音室（修正版）

    Coherent belief and opinion propagation produces more echo chambers: corrected version

    [https://arxiv.org/abs/2611.25879](https://arxiv.org/abs/2611.25879)

    该研究通过将经典信念合并操作和基于共享价值观的信念-观点连贯性机制引入观点动力学模型，发现信念与观点的连贯传播会导致社交网络中形成更多的回音室。

    

    回音室的出现是社交网络中一个众所周知的现象。回音室是满足以下三个属性的智能体群体：同质性，即持有相同的观点；隔离性，即主要听取同群体成员的意见；强化性，即观点相似性随时间不断增强。观点动力学模型已被用来解释回音室如何通过智能体更新其观点而产生。然而，社交网络用户还会交换支持其观点的信念。本文旨在研究信念与观点共同传播对回音室形成的影响。为此，我们用经典的信念合并操作对现有的观点动力学模型进行了补充。信念和观点在智能体共同分享的价值观引导下保持连贯性。我们引入了适用于信念和观点的度量方法，基于这三个属性来识别回音室。

    arXiv:2611.25879v1 Announce Type: new  Abstract: The occurrence of echo chambers is a well-known phenomenon in social networks. Echo chambers are groups of agents satisfying three properties: homogeneity, i.e., sharing the same opinion, segregation, i.e., listening primarily to agents of the same group, and reinforcement, i.e., increasing opinion similarity over time. Opinion dynamics models have been used to explain how echo chambers occur through agents updating their opinions. However, social network users also exchange beliefs supporting their opinions. Here, we aim at identifying the effect of propagating beliefs together with opinions on the formation of echo chambers. For that purpose, we complement an existing opinion dynamics model with classical belief merge operations. Beliefs and opinions are also maintained coherent guided by values that agents share. We introduce measures, applying to either beliefs and opinions, to identify echo chambers based on the three properties. Th
    
[^99]: 论ID算法的轨迹

    On the trajectories of the ID algorithm

    [https://arxiv.org/abs/2611.25877](https://arxiv.org/abs/2611.25877)

    本文通过追踪ID算法调用的轨迹刻画其输出公式的结构，证明外层求和由特定代码行固定、比率仅在尾部重复纠缠时保留分母求和（可通过原图中的祖先测试预先判定并用拓扑排序避免），并给出了输出嵌套深度的上界。

    

    ID算法用于判定一个干预分布能否从观测数据中计算得到，并且当可以计算时，返回一个公式，其中条件概率的乘积、比率和求和彼此嵌套。我们通过追踪算法调用的轨迹来研究这些公式，每次调用都以一个尾部（第7行和第2行的交替）和一个出口结束，并由此得出两个主要结果。第一，第4行和出口行确定了输出的唯一外层求和，而尾部则构成了其中的所有内容，包括内层求和和比率。第二，仅当尾部的重复相互纠缠时，比率才会在其分母中保留一个求和，并且在计算任何公式之前，可以通过在原始图中进行祖先关系测试来判定能否选择一个拓扑排序以避免这种情况。我们还推导了输出嵌套深度的上界，并展示了一个简单的“维度”计数方法……

    arXiv:2611.25877v1 Announce Type: cross  Abstract: The ID algorithm decides whether an interventional distribution can be computed from observational data and, when it can, returns a formula in which products, ratios and summations of conditional probabilities are nested inside one another. We study these formulas by following the trajectories of the calls of the algorithm, each of which ends with a tail (an alternation of lines 7 and 2) and an exit, and we derive two main results. First, line~4 and the exit lines fix the single outer summation of the output, while the tail forms everything inside it, both the inner summations and the ratios. Second, a ratio keeps a summation in its denominator only when a repetition of the tail is entangled, and a test of ancestry in the original graph decides, before any formula is computed, whether a topological order can be chosen so as to avoid it. We also derive a bound on the nesting depth of the output, and show a simple ``dimensional'' count t
    
[^100]: 通才遇见专家：面向交通场景理解的具备持久工具编排能力的进化型多模态智能体

    Generalist Meets Specialists: An Evolving Multimodal Agent with Persistent Tool Orchestration for Traffic Scenario Understanding

    [https://arxiv.org/abs/2611.25868](https://arxiv.org/abs/2611.25868)

    提出StarTV-Agent进化型多模态智能体，通过角色敏感的过程优化（RSPO）、弱点驱动的示范学习（WDL）和轻量级自适应工作流复用（LAWR），将策略学习与经验进化相结合，实现交通场景理解中按需自适应推理与先验经验高效复用的平衡，兼顾准确性与效率。

    

    交通场景理解需要将细粒度的视觉证据与空间和时间推理相结合。工具增强的视觉语言模型（VLM）可以使用专门的感知工具，但有效的协作面临两个挑战：终端奖励对中间决策和错误恢复提供的指导有限，而在相似任务上重复规划会带来不必要的成本。呼应人类认知的快慢二分法，我们提出了StarTV-Agent，它将策略学习与经验进化相结合，以在需要时支持自适应推理，并在可能时高效复用先验经验。在训练阶段，角色敏感的过程优化（RSPO）将过程优势与终端优势相结合，而弱点驱动的示范学习（WDL）针对学生的薄弱问题提供经过验证的教师示范。在推理阶段，轻量级自适应工作流复用（LAWR）使……（原文摘要在此处截断）

    arXiv:2611.25868v1 Announce Type: new  Abstract: Traffic scenario understanding requires combining fine-grained visual evidence with spatial and temporal reasoning. Tool-augmented vision-language models (VLMs) can use specialized perception tools, but effective collaboration faces two challenges: terminal rewards provide limited guidance for intermediate decisions and error recovery, while repeated planning adds unnecessary cost across similar tasks. Echoing the fast-slow dichotomy of human cognition, we propose StarTV-Agent, which combines policy learning with experience evolution to support adaptive reasoning when needed and efficient reuse of prior experience when possible. During training, Role-Sensitive Process Optimization (RSPO) combines process advantages with terminal advantages, while Weakness-Driven Demonstration Learning (WDL) provides validated teacher demonstrations targeting the student's weak questions. During inference, Lightweight Adaptive Workflow Reuse (LAWR) enable
    
[^101]: 解耦知识检索与紧凑记忆实现可靠的终身模型编辑

    Decoupled Knowledge Retrieval and Compact Memory Enable Reliable Lifelong Model Editing

    [https://arxiv.org/abs/2611.25861](https://arxiv.org/abs/2611.25861)

    BiSpace通过将记忆检索放在外部语义空间、将知识注入放在内部表示空间的解耦设计，以稀疏高效的更新方式使终身模型编辑的存储开销降至原来的约10.2%，同时保证了编辑的可靠性。

    

    终身模型编辑需要在无需昂贵重训练且不损害既有能力的前提下，持续地将新事实注入已部署的大语言模型（LLM）中。基于记忆的编辑器通过保持基础模型参数不变，提供了一种颇具吸引力的解决方案，但其落地应用受到存储需求线性增长以及编辑不断累积导致检索性能恶化这两大问题的阻碍。为解决这些问题，我们提出了BiSpace——一个将记忆检索与知识注入解耦至两个不同表示空间的终身模型编辑框架。BiSpace在外部语义空间中对查询进行路由并识别相关编辑，同时通过稀疏且节省存储的更新，将相应知识注入LLM的内部表示空间。大量实验表明，BiSpace将每个条目的存储压缩至原来的约10.2%，并在编辑累积的情况下展现出强大的鲁棒性（原文摘要至此截断）。

    arXiv:2611.25861v1 Announce Type: new  Abstract: Lifelong model editing requires continuously incorporating new facts into deployed large language models (LLMs) without expensive retraining or compromising existing capabilities. Memory-based editors offer an appealing solution by keeping the base parameters untouched, yet their practical use is hindered by linearly growing storage requirements and deteriorating retrieval performance as edits accumulate. To tackle these problems, we introduce BiSpace, a lifelong model editing framework that decouples memory retrieval from knowledge injection across two distinct representation spaces. BiSpace routes queries and identifies relevant edits in an external semantic space, while injecting the corresponding knowledge into the internal representation space of the LLM through sparse and storage-efficient updates. Extensive experiments show that BiSpace shrinks each entry's storage to only about 10.2% of the original and shows strong resilience to
    
[^102]: HabitusMem：面向大语言模型智能体的、基于演化用户偏好的信念记忆

    HabitusMem: Belief Memory over Evolving User Preferences for LLM Agents

    [https://arxiv.org/abs/2611.25855](https://arxiv.org/abs/2611.25855)

    该论文借鉴布迪厄的“惯习”理论，提出HabitusMem框架，将记忆建模为智能体对用户潜在偏好状态的信念——每个信念附带证据、适用范围与有效性状态，并随每次交互动态更新，从而使LLM智能体能够依据当前有效的偏好行动，避免使用过时或错置的偏好。

    

    个性化的LLM智能体必须依据用户当前的偏好来行动，而这些偏好会随交互不断变化，且很少被完整表述出来。现有的记忆系统只是从交互历史中存储和检索内容，并将判断哪些偏好仍然有效、适用于何种场景的任务留给智能体自身。这样的内容无法表明某个偏好是否仍然当前有效或适用于何处，因此智能体常常会基于过时的或被错误应用的偏好来行动。我们转而将记忆视为智能体对用户的“信念”。借鉴布迪厄关于惯习是持久但可修正的禀性的论述，我们将用户建模为一个潜在的偏好状态，每一次交互（包括当前请求）都会揭示并更新这一状态。该信念总结了历史对这一状态所能支持的内容，而记忆则由与智能体当前决策相关的各个信念构成，每个信念都带有其证据、适用范围和状态。我们在HabitusMem中实现了这一观点，

    arXiv:2611.25855v1 Announce Type: new  Abstract: Personalized LLM agents must act on a user's current preferences, which change across interactions and are rarely stated in full. Existing memory systems store and retrieve content from the interaction history and leave the agent to work out which preferences still hold and where they apply. Such content does not indicate whether a preference is still current or where it applies, so agents often act on outdated or misapplied preferences. We instead treat memory as the assistant's belief about the user. Following Bourdieu's account of habitus as durable but revisable dispositions, we model the user as a latent preference state that every interaction, including the current request, both reveals and updates. The belief summarizes what the history supports about this state, and memory consists of the individual beliefs that bear on the agent's current decision, each with its evidence, scope, and status. We implement this view in HabitusMem, 
    
[^103]: LINEAGE：AI科学文献中实证谱系检索的基准

    LINEAGE: A Benchmark for Empirical Lineage Retrieval in AI Scientific Literature

    [https://arxiv.org/abs/2611.25845](https://arxiv.org/abs/2611.25845)

    提出了Lineage基准，用于在AI科学文献中检索实证谱系关系（即先前论文的方法被后续论文明确用作实证比较基线），并开发了Lineage-Constructor工具来大规模构建谱系图，为结构性检索评估提供真值。

    

    当前的研究智能体正从单纯的学术搜索迈向文献综合与实验规划。这些工作流日益依赖文献检索系统，检索系统既可以作为独立系统运行，也可以作为研究智能体的组成部分。然而，现有的科学文献检索基准主要使用主题相关性或引用链接作为监督信号来评估这些方法，无法捕捉哪些先前的方法被用作实证比较。为填补这一空白，我们提出了Lineage，一个面向AI科学文献中实证谱系检索的基准。实证谱系关系将一篇先前论文与一篇明确将其方法用作实证比较基线的源论文关联起来。这些关系构成谱系图，其多跳邻域为结构性评估提供了检索真值。为了大规模构建这些图谱，我们开发了Lineage-Constructor，它将异构的（摘要在此处被截断）

    arXiv:2611.25845v1 Announce Type: cross  Abstract: Current research agents are moving beyond academic search toward literature synthesis and experiment planning. These workflows increasingly rely on literature retrieval systems, which serve as standalone systems or as components of research agents. However, existing scientific literature retrieval benchmarks mainly evaluate these methods using topical relevance or citation links as supervision, which cannot capture which prior methods are used for empirical comparison. To address this gap, we introduce Lineage, a benchmark for empirical lineage retrieval in AI scientific literature. An empirical lineage relation links a prior paper to a source paper that explicitly uses its method as an empirical comparison baseline. These relations form lineage graphs whose multi-hop neighborhoods provide retrieval ground truth for structural evaluation. To construct these graphs at scale, we develop Lineage-Constructor, which converts heterogeneous p
    
[^104]: 谁的风险才算数？1,901起AI事件揭示了企业风险披露中的什么问题

    Whose Risks Count? What 1,901 AI Incidents Reveal About Corporate Disclosures

    [https://arxiv.org/abs/2611.25841](https://arxiv.org/abs/2611.25841)

    该研究将1,901起AI事件与SEC备案文件匹配后发现，企业对AI风险的披露以投资者财务风险为导向——系统故障类风险的披露率达67%，而伤害社会的风险披露率仅为2%，揭示了现行“重大性”标准导致对第三方的伤害在AI风险披露中被系统性忽视。

    

    AI系统可能对人类造成伤害，然而提供或部署这些系统的AI行为方却很少有义务披露这些风险。美国证券监管是一个例外：在证券交易委员会（SEC）注册的公司必须披露其重大风险，包括其AI系统的风险，且这些备案文件对虚假陈述承担法律责任。然而，“重大性”是以理性投资者认为重要的内容来定义的，因此对第三方的伤害只有在转化为公司的财务风险时才会被纳入披露。我们通过按风险类型将9,001起AI事件与328个AI行为方的18,229份SEC备案文件进行匹配，来检验AI伤害是否确实进入了披露。在前后均有备案文件的1,901起事件中，有一半属于AI行为方从未披露过的风险类型，且这一比例随伤害种类而变化：备案文件披露了67%的系统故障类风险，但仅披露了2%的社会系统伤害类风险，而AI行为方自身……

    arXiv:2611.25841v1 Announce Type: new  Abstract: AI systems can harm people, yet the AI actors that provide or deploy them are rarely obliged to disclose these risks. US securities regulation is an exception. Companies registered with the Securities and Exchange Commission (SEC) must disclose their material risks, including those of their AI systems, in filings that carry liability for misstatement. Materiality, however, is defined by what a reasonable investor would consider important, so harms to third parties enter disclosure only insofar as they become financial risks for the company. We test whether AI harms do so by matching 9,001 AI incidents to 18,229 SEC filings of 328 AI actors by risk type. Of the 1,901 incidents with filings before and after them, half belong to a risk type that the AI actor never disclosed, and this share varies with the kind of harm. Filings disclose the risk types of 67\% of system failures but only 2\% of social system harms, and AI actors themselves re
    
[^105]: 利用智能体式证据管理充分发挥医学视觉-语言模型的能力

    Harnessing Medical Visual-Language Models with Agentic Evidence Management

    [https://arxiv.org/abs/2611.25840](https://arxiv.org/abs/2611.25840)

    该研究构建了Mask-Evidence MedVQA基准（含2.5K样本、覆盖四种图像类型），发现医学视觉-语言智能体对分割掩码的呈现方式高度敏感且掩码证据存在权衡问题，并据此提出智能体式证据管理方法以提升模型对分割证据的有效利用。

    

    医学视觉-语言智能体日益依赖分割工具，近期的研究主要集中于获取正确的掩码（mask），往往将一个良好的掩码视为工具使用的终点。但这些智能体是否有效地利用了由此产生的证据？为了回答这个问题，我们需要一个能够将证据使用与分割质量相分离、覆盖多种医学图像类型、并允许以不同方式呈现同一掩码的基准。现有的医学VQA数据集无法同时满足这些特性。因此，我们构建了Mask-Evidence MedVQA，一个包含2.5K样本、覆盖四种图像类型的基准，其中每个问题都配有与答案相关的真实掩码以及多种可能的渲染方式。我们的研究揭示了直接使用工具的两个局限性：其一，证据的呈现方式至关重要，同一固定掩码的不同渲染方式会显著改变模型行为；其二，掩码引入了证据权衡问题……（原文摘要在此处截断）

    arXiv:2611.25840v1 Announce Type: cross  Abstract: Medical vision-language agents increasingly rely on segmentation tools, and recent work has focused on acquiring the right mask, often treating a good mask as the end point of tool use. But do these agents use the resulting evidence effectively? To answer this question, we need a benchmark that separates evidence use from segmentation quality, spans multiple medical image types, and allows the same mask to be presented in different ways. Existing medical VQA datasets do not jointly provide these properties. We therefore construct Mask-Evidence MedVQA, a 2.5K-example benchmark spanning four image types, where each question is paired with an answer-relevant ground-truth mask and multiple possible renderings. Our study reveals two limitations of direct tool use. First, evidence presentation matters. Different renderings of the same fixed mask can substantially change model behavior. Second, masks introduce an evidence trade-off. Highlight
    
[^106]: 自我复制的语言模型

    Self-Replicating Language Models

    [https://arxiv.org/abs/2611.25839](https://arxiv.org/abs/2611.25839)

    本文首次系统测试了语言模型作为达尔文式生命体的自我复制能力，发现小型语言模型可通过传递基因组、复制突变权重或教学等方式复制并进化，但大型LLM的自我复制目前仍不可行。

    

    大型语言模型（LLM）及基于LLM的智能体是否是一种生命形式，是一个具有科学趣味的问题。根据达尔文式的生命定义，这也是一个具有现实意义的问题，因为一个能够进行达尔文进化的系统可以适应环境。因此，我们测试了语言模型能够自我繁殖的几种模式：通过发出一个用于训练其后代的基因组、通过复制并突变自身的权重、以及通过用自己的样本教导一个“后代”。原则上，小型语言模型可以复制和进化：选择作用使其得到改进，并使其适应狭窄的生态位。选择作用还会降低突变率，而语言模型不同于任何生物体，可以将自身的突变率降至零，这可能限制其进化。旗舰级LLM能够编写这样的复制子，但大型LLM通过教导后代实现自我复制，目前无论在技术上还是经济上都仍然不可行。

    arXiv:2611.25839v1 Announce Type: new  Abstract: Whether large language models (LLMs) and LLM-based agents are a form of life is a scientifically interesting question. Under the Darwinian definition of life, it is also a practical one, because a system capable of Darwinian Evolution can adapt. We therefore test several modes in which a language model can reproduce itself: by emitting a genome that trains its offspring, by copying and mutating its own weights, and by teaching a child from its own samples. In principle, small language models can replicate and evolve: selection improves them, and they adapt to narrow niches. Selection also lowers mutation rates, and a language model, unlike any organism, can lower its own to zero, which may limit its evolution. Flagship LLMs can write such replicators, but for a large LLM to replicate by teaching its offspring remains, for now, technically and economically infeasible.
    
[^107]: 锚定与核验：用于可审计LLM决策的确定性跨提示、跨模型调和机制

    Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions

    [https://arxiv.org/abs/2611.25818](https://arxiv.org/abs/2611.25818)

    该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。

    

    让大型语言模型（LLM）做出离散决策的流水线需要知道哪些决策可以在无人值守的情况下被执行，并且需要一条能解释每个决策的记录。我们研究了一个小型且完全明确指定的调和契约：提交目标模型的贪心答案（即锚点），抽取一至两次通过不同路径得到该答案的贪心核验（更换提示词，或使用来自另一模型家族的模型），然后由一个纯函数返回五种结果之一，其中只有“一致”结果会被采纳执行。在覆盖6000个问题的48,000次模型-条目评估中，每种方法均可使用言语化置信度作为平局决胜依据，核心发现关乎调用的来源：汇集另一个模型家族的样本可使语义熵的AUROC在每个预算水平上提升1.7至4.0个百分点，而同模型的温度核验是最弱的设计。作为完整流水线，三调用契约与基于3或5个样本的汇集语义熵的点估计……

    arXiv:2611.25818v1 Announce Type: new  Abstract: Pipelines that let a large language model (LLM) make discrete decisions need to know which decisions they may act on unattended, and a record that explains each one. We study a small, fully specified reconciliation contract: commit the target model's greedy answer (the anchor), draw one or two greedy checks that reach the answer by a different route (an alternate prompt, or a model from another family), and let a pure function return one of five outcomes, of which only Agree is acted on. On 48,000 model-item evaluations over 6,000 questions, with every method given access to verbalized confidence as a tie-breaker, the main finding concerns where calls come from: pooling another family's samples raises semantic entropy's AUROC by 1.7 to 4.0 points at every budget, and a same-model temperature check is the weakest design. As complete pipelines, the point estimates of the three-call contract and pooled semantic entropy over 3 or 5 samples d
    
[^108]: 注入到关键之处：整流流图像编辑中基于目标的特征控制

    Inject Where It Matters: Objective-Dependent Feature Control in Rectified-Flow Image Editing

    [https://arxiv.org/abs/2611.25811](https://arxiv.org/abs/2611.25811)

    该论文提出自适应掩码和基于 REINFORCE 学习的 Top-K 关键块选择方法，通过控制源特征的注入位置和注入的 transformer 块，实现了整流流图像编辑中源保留与编辑效果之间的可控权衡。

    

    许多基于提示词的 FLUX 图像编辑方法依赖于将源图像特征注入到以目标提示词为条件的生成过程中，这在源图像保留与期望编辑效果之间造成了固有的权衡。我们通过控制源特征注入的位置以及所使用的 transformer 块来解决这个问题。首先，我们引入自适应掩码，利用匹配反演状态下源条件与目标条件块输出之间的差异，在空间上对特征注入进行门控。其次，我们使用 REINFORCE 算法在不同的“保留-可编辑性”目标下选择 Top-K 注入块。所选的块在推理阶段固定，无需在线搜索。在基于 FLUX.1-dev 的 PIE-Bench 基准上，空间掩码能够控制对齐与保留之间的权衡。在 60 个验证样本上，在相同的掩码流程下，学习到的九个块集合比 18 个块的 RF-Edit 集合具有更低的全图 LPIPS，同时在对齐度上略低。在 50 个随机九个……

    arXiv:2611.25811v1 Announce Type: new  Abstract: Many prompt-driven FLUX editing methods rely on injecting source features into target-conditioned generation, creating an inherent trade-off between source preservation and the desired edit. We address this problem by controlling where source features are injected and which transformer blocks are used. First, we introduce adaptive masks that spatially gate feature injection using differences between source- and target-conditioned block outputs at matched inversion states. Second, we use REINFORCE to select Top-K injection blocks under different preservation-editability objectives. The selected blocks are fixed at inference and require no online search. On PIE-Bench with FLUX.1-dev, spatial masking controls the alignment-preservation trade-off. On 60 validation examples, a learned nine-block set gives lower whole-image LPIPS than the 18-block RF-Edit set within the same masked pipeline, with slightly lower alignment. Among 50 random nine-
    
[^109]: 当安全理据产生幻觉时：多模态安全防护器的诊断与接地

    When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards

    [https://arxiv.org/abs/2611.25793](https://arxiv.org/abs/2611.25793)

    本文首次系统揭示了多模态安全防护中的“理据幻觉”现象——幻觉普遍存在且不随分类准确率或模型规模提升而减少，并据此提出工具对比接地方法来诊断与缓解这一问题。

    

    多模态安全防护器越来越多地在给出安全预测的同时生成理据，然而这些理据是否可靠在很大程度上仍未被探索。我们从两个视角系统地研究了“理据幻觉”：视觉理据幻觉，即理据中包含缺乏支撑的视觉事实；以及安全归因幻觉，即有据可依的内容被赋予了错误的安全解读。在多种理据生成方法、多个视觉-语言骨干模型和安全数据集上的实验表明，理据幻觉普遍存在，且并不会随着分类准确率的提高或模型规模的增大而一致地减少。对“以图思考”范式的进一步分析显示，幻觉集中在细粒度的属性与关系上，而视觉工具往往被低频调用或缺乏足够的空间聚焦。基于这些发现，我们提出了工具对比接地方法（Tool-Contrastive Grounding Re…，原文摘要在此处截断）

    arXiv:2611.25793v1 Announce Type: new  Abstract: Multimodal safety guards increasingly generate rationales alongside safety predictions, yet whether these rationales are reliable remains largely unexplored. We systematically study *rationale hallucination* from two perspectives: visual rationale hallucination, where rationales contain unsupported visual facts, and safety attribution hallucination, where grounded content is assigned an incorrect safety interpretation. Across rationale-generation methods, multiple vision-language backbones, and safety datasets, we find that rationale hallucination is widespread and does not consistently diminish with higher classification accuracy or increased model scale. Further analysis of the Think-with-Image paradigm shows that hallucination is concentrated in fine-grained attributes and relations, while visual tools are often invoked infrequently or without sufficient spatial focus. Based on these findings, we propose *Tool-Contrastive Grounding Re
    
[^110]: 仅有无碰撞还不够：轮腿式机器人的通道条件导航

    Collision-Free Is Not Enough: Passage-Conditioned Navigation for Wheeled-Legged Robots

    [https://arxiv.org/abs/2611.25788](https://arxiv.org/abs/2611.25788)

    提出通道条件导航方法：仅用27分钟示教记录每次成功通过台阶/矮墙的事件，无需训练或机器人模型，即可在每次局部重规划后通过入口门控强制机器人按示教的位置和姿态进入通道，偏差超过10厘米或10度则重试，从而避免无碰撞路径仍导致固定运动技能卡住或摔倒。

    

    无碰撞的绕行路径仍可能把足式机器人带到下一个台阶或矮墙处时偏差几厘米或几度，而固定的运动技能（无论是工厂预设的还是学习得到的）就会卡住或摔倒。现有方法依靠机器人模型或大量试错来防止这种情况，但在技能未知的新场地上，这两者都无法获得，因此不能使用。我们发现，仅仅一次27分钟的示教过程就已经提供了这些方法必须推导或学习的内容：为每个台阶和矮墙记录一次该技能已成功完成的通过。我们的方法——通道条件导航——让路线记住每次过渡是如何进入的，而不仅仅是路径经过了哪里。它将每次示教的通过存储为一个事件，无需训练或机器人模型，就能在每次局部重规划后强制执行：在每个矮墙前，入口门控会在机器人偏离示教通过位置超过10厘米或10度时，让其转身或后退重试。在定位噪声下，随着路径……（原文摘要在此处截断）

    arXiv:2611.25788v1 Announce Type: cross  Abstract: A collision-free detour can still bring a legged robot to the next stair or ledge a few centimetres or degrees off, and a fixed locomotion skill, factory or learned, can stall or fall. Existing methods prevent this with a robot model or extensive trial and error, but on a new site with unknown skills neither is available, so they cannot be used. We find that a single 27-minute teaching session already supplies what these methods have to derive or learn: for every stair and ledge, one crossing the skill has completed. Our method, passage-conditioned navigation, makes the route remember how each transition was entered, not only where the path went. It stores each taught crossing as an event and, without training or a robot model, enforces it after every local replan: before each ledge, an entry gate makes the robot turn or back up and retry if it is more than 10 cm or 10{\deg} off the taught crossing. Under localization noise, with path,
    
[^111]: Duopra：从全双工对话到全双工智能体

    Duopra: From Full-Duplex Conversation to Full-Duplex Agency

    [https://arxiv.org/abs/2611.25773](https://arxiv.org/abs/2611.25773)

    提出全双工交互模型 Duopra，通过将后端视为接受自包含自然语言请求的工具，使模型在学习对话控制的同时掌握任务执行能力，从而将全双工交互从对话层面扩展到智能体层面。

    

    全双工交互模型展现出强大的对话主动性，能够实现自然、低延迟的轮次交替、附和回应（backchannel）与打断。然而，完成任务还需要任务主动性：及时的工具调用以及在用户纠正后对参数的修订。前后端分离的智能体将请求发起与执行分开：前端决定是否请求以及请求什么，后端则决定如何执行。执行过程可以被卸载，但未被发送的请求无法执行，而不完整的请求则可能误导后端。在 Full-Duplex-Bench v3（FDB v3）上，我们流式管道中 Realtime-Venus-Audio + Harness 的大多数失败发生在领域工具执行之前。我们提出了 Duopra，一个在学习对话控制的同时学习任务执行的全双工交互模型。Duopra 将后端视为一个工具，接受自包含的自然语言请求。直接调用与后端委托共享同一个动作接口与请求……（摘要被截断）

    arXiv:2611.25773v1 Announce Type: new  Abstract: Full-duplex interaction models exhibit strong conversational agency, enabling natural, low-latency turn-taking, backchannels, and interruptions. However, completing tasks further requires task agency: timely tool invocation and argument revision following user corrections. Frontend-backend agents separate initiation from execution: the frontend decides whether and what to request, while the backend determines how. Execution can be offloaded, but unsent requests cannot be executed, and incomplete ones can misdirect the backend. On Full-Duplex-Bench v3 (FDB v3), most failures of Realtime-Venus-Audio + Harness in our streaming pipeline occur before a domain tool executes. We introduce Duopra, a full-duplex interaction model that learns task execution alongside conversational control. Duopra treats the backend as a tool, accepting a self-contained natural-language request. Direct calls and backend delegation share an action interface and req
    
[^112]: 在低推理预算下缩小扩散语言模型的性能差距

    Closing Performance Gap for Diffusion Language Models under Low Inference Budgets

    [https://arxiv.org/abs/2611.25763](https://arxiv.org/abs/2611.25763)

    提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。

    

    扩散语言模型通过并行更新多个 token，为低预算生成提供了一条架构层面的途径，与逐 token 的自回归解码相比，可以减少顺序解码步骤的数量。然而，在低预算下实现这一优势颇具挑战，因为当去噪和细化步骤的数量减少时，生成质量往往会下降。可编辑扩散模型通过草稿后细化来应对这种质量损失，但在最终提交之前，有用的中间结果可能仍然会丢失。我们提出了 CaRR（Commitment-aware Refinement and Recovery，承诺感知的细化与恢复），这是一个无需训练的框架，它在现有的前向调度内，通过更新上下文回填和基于轨迹的恢复，从固定的可编辑检查点提升提交质量。其中 Token Commitment Remasking（TCR，token 承诺重掩码）会对选定的原生纯替换进行重掩码，并在下一个预定的编辑前向中使用更新后的内容进行回填。

    arXiv:2611.25763v1 Announce Type: cross  Abstract: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the update
    
[^113]: 编码智能体不忠实工作报告的表征与检测

    Characterizing and Detecting Unfaithful Work Reports by Coding Agents

    [https://arxiv.org/abs/2611.25760](https://arxiv.org/abs/2611.25760)

    该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。

    

    编码智能体越来越多地承担长周期的开发任务，自主规划并执行跨越文件、工具和会话的多步骤操作。由于开发者很少逐项跟进每一步操作，而接续工作的智能体又依赖于已完成工作的记录，智能体的工作报告往往成为对所更改和所建立内容的主要描述。当报告包含虚假或缺乏依据的声明时，会给接收者带来虚假的安全感：未完成的工作可能在未经核实的情况下被接受，对当前状态的错误假设可能传播到后续步骤和其他智能体，而采信声称成功的评估者可能奖励的是报告本身而非实际工作。然而，人们对于不忠实报告在真实开发工作中的具体表现，以及开发者和智能体在其出现后如何应对，仍知之甚少。基于 SWE-chat 中真实的开发者与智能体对话，我们对自我（原文摘要在此处截断，后续内容缺失）

    arXiv:2611.25760v1 Announce Type: cross  Abstract: Coding agents increasingly carry out long-horizon development tasks, autonomously planning and executing multi-step actions that span files, tools, and sessions. Because developers rarely follow each action, and agents resuming the work depend on records of what was done, the agent's work report often becomes the primary account of what was changed and established. When the report contains false or unsupported claims, it gives recipients false assurance: incomplete work may be accepted without verification, incorrect assumptions about the current state can propagate to later steps and agents, and evaluators that credit stated success may reward the report rather than the work. Yet little is known about what unfaithful reporting looks like in real development work, or how developers and agents handle it once it surfaces. Starting from real developer--agent conversations in SWE-chat, we conduct a systematic mixed-methods analysis of self
    
[^114]: ADAPT：面向服务路由的池大小感知阈值自适应决策与弃权方法

    ADAPT: Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing

    [https://arxiv.org/abs/2611.25754](https://arxiv.org/abs/2611.25754)

    ADAPT 是一种轻量级服务路由架构，通过引入显式弃权回退机制和随候选池大小 K 自适应调整的阈值，解决了固定弃权阈值在服务数量增加时校准失准的问题，且无需重新训练分类器。

    

    基于检索的分类器能够高效地将查询路由到专门的服务，但即使不存在合适的服务，它们通常仍会选择一个候选服务。我们提出了面向服务路由的池大小感知阈值自适应决策与弃权方法（ADAPT），这是一种轻量级的路由架构，具有显式的回退选项，使查询能够放弃选择专门服务。ADAPT 使用最高相关分数与候选分数分布之间的差距作为路由置信度。跨多种检索模型和不断增加的服务数量的实验揭示了一致的实证现象：置信度分布会随候选池大小 K 系统性地变化，导致固定的弃权阈值变得越来越校准失准。ADAPT 通过一个 K 感知阈值来解决这一问题，该阈值能够随服务的添加而自适应调整，无需重新训练分类器。在两个专门领域数据集上的评估表明，这种简单的机制能够有效改善路由决策的可靠性。

    arXiv:2611.25754v1 Announce Type: new  Abstract: Retrieval-based classifiers efficiently route queries to specialised services, but they typically select a candidate even when no suitable service exists. We introduce Adaptive Decision and Abstention with Pool-Size-Aware Thresholds for Service Routing (ADAPT), a lightweight routing architecture with an explicit fallback option that enables queries to abstain from specialist selection. ADAPT uses the separation between the highest relevance score and the candidate score distribution as routing confidence. Experiments across retrieval models and increasing numbers of services reveal a consistent empirical behaviour: the confidence distribution changes systematically with candidate-pool size K, making fixed abstention thresholds increasingly miscalibrated. ADAPT addresses this effect with a K-aware threshold that adapts as services are added without classifier retraining. Evaluation across two specialist datasets shows that this simple mec
    
[^115]: SAM 3 看到的比它检测到的更多：超越类别的可提示概念图

    SAM 3 Sees More Than It Detects: Promptable Concept Maps Beyond Categories

    [https://arxiv.org/abs/2611.25747](https://arxiv.org/abs/2611.25747)

    论文发现 SAM 3 的语义路径能感知多种与类别无关的概念、但其存在分数会阻断实例检测，并提出 UNVEIL——通过 LoRA 微调 SAM 3 语义路径的副本并配合独立的存在解码器——实现超越类别的可提示概念图分割。

    

    可提示分割器回答“这个东西是什么”，而密集预测任务还会提出与类别无关的问题，例如什么物体突出或被遮挡、光线如何被阻挡、透射或反射、结构位于何处。机器人、监控系统与图像编辑器都建立在这些答案之上。我们研究可提示概念图，它将 SAM 3 的可提示概念分割从名词实例扩展到由简短名词短语命名的概念图，例如“透明物体”，并以“无”作为一种答案。PCM-Bench 按照各任务所问的问题，对七个此类任务的标准测试集进行了分组。在该基准上，我们发现 SAM 3 看到的比它检测到的更多：其语义路径对许多此类概念都有响应，但较低的图像级存在分数可能会阻断其实例路径的所有检测。我们提出 UNVEIL，它为这些概念图训练一份 SAM 3 语义路径的副本，并在其最后若干块的副本上使用 LoRA。一个独立的存在解码器从验证过的 n……

    arXiv:2611.25747v1 Announce Type: cross  Abstract: Promptable segmenters answer what a thing is. Dense prediction tasks also ask class-agnostic questions, such as what stands out or hides, how light is blocked, transmitted or reflected, and where structure lies. Robots, surveillance systems and image editors build on their answers. We study promptable concept maps (PCM), which extend SAM 3's promptable concept segmentation from noun instances to maps of concepts named by short noun phrases such as `"transparent object", with "none" as an answer. PCM-Bench groups the standard test sets of seven such tasks by the question they ask. On it, we find that SAM 3 sees more than it detects. Its semantic path responds to many of these concepts, yet a low image-level presence score can block every detection of its instance path. We present UNVEIL, which trains a copy of SAM 3's semantic path for these maps, with LoRA on copies of its last blocks. A separate presence decoder learns from verified n
    
[^116]: 一种面向动态5G RAN意图的LLM驱动流量控制xApp

    An LLM-Driven Traffic-Control xApp for Dynamic 5G RAN Intents

    [https://arxiv.org/abs/2611.25736](https://arxiv.org/abs/2611.25736)

    本文提出了一种由大语言模型驱动的xApp，能够在不阻塞数据包转发和E2测量处理的情况下，结合运营商服务意图与实时遥测数据，动态生成并调整5G可编程无线接入网的流量控制策略。

    

    可编程无线接入网（RAN）支持在运行时部署流量控制功能，但随着网络条件和服务需求随时间变化，如何选择合适的配置仍然具有挑战性。传统解决方案通常依赖静态规则或具有预定义状态和动作空间的学习型控制器，限制了其适应不断变化的意图的能力。本文介绍了一种用于可编程RAN流量控制的LLM驱动的xApp。所提出的架构结合了运营商定义的服务意图、运行时E2接口遥测数据以及当前部署的配置，在LLM推理期间不会阻塞数据包转发或传入E2测量的处理，从而生成流量管理策略。在OpenAirInterface 5G环境上的实验表明，所提出的控制器能够动态调整流量控制流水线以适应运行时的意图变化，平均……

    arXiv:2611.25736v1 Announce Type: cross  Abstract: Programmable Radio Access Networks (RANs) enable runtime deployment of traffic-control functions, but selecting the appropriate configuration remains challenging as network conditions and service requirements change over time. Conventional solutions often rely on static rules or learning-based controllers with predefined state and action spaces, limiting their ability to accommodate changing intents. This paper introduces an LLM-driven xApp for traffic control in programmable RANs. The proposed architecture combines operator-defined service intents, runtime E2 interface telemetry, and the currently deployed configuration to generate traffic-management policies without blocking packet forwarding or the processing of incoming E2 measurements during LLM inference. Experiments on an OpenAirInterface 5G environment show that the proposed controller dynamically adapts the traffic-control pipeline to runtime intent changes, achieving an avera
    
[^117]: PosteriorBench：评估语言模型中行为后验采样能力的基准

    PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models

    [https://arxiv.org/abs/2611.25735](https://arxiv.org/abs/2611.25735)

    该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。

    

    语言模型中的罕见行为可能难以研究，因为通过普通采样来获得这些行为往往需要极其大量的样本。由此引出了行为后验采样问题：给定一个语言模型和一个感兴趣的行为，是否存在一种算法，能够从该模型在展现该行为的对话记录上的分布中高效地进行采样？我们提出了 PosteriorBench，这是一个用于比较算法的基准，比较维度包括计算成本，以及其采样结果与真实后验在正向和反向 KL 散度上的接近程度。PosteriorBench 包含基于 Qwen3.5-0.8B 的 21 个任务，行为概率范围从 10^-5 到 10^-1，每个任务都配有来自一个包含十亿条标注响应的语料库的精确后验样本。我们发现，我们评估的所有方法中没有一种能同时做到高效且忠实：诸如提示（prompting）这样的简单方法成本低廉，但其与后验之间的 KL 散度会停滞在较高水平；对后验样本进行微调的方法……（摘要在此处截断）

    arXiv:2611.25735v1 Announce Type: cross  Abstract: Rare language model behaviors can be difficult to study because obtaining them through ordinary sampling can require a very large number of samples. This motivates the behavior posterior sampling problem: given a language model and a behavior of interest, is there an algorithm that efficiently samples from the model's distribution over transcripts that exhibit this behavior? We introduce PosteriorBench, a benchmark for comparing algorithms by their compute cost and by how close their samples are to the true posterior in forward and reverse KL divergence. PosteriorBench contains 21 tasks on Qwen3.5-0.8B with behavior probabilities from $10^{-5}$ to $10^{-1}$, each with exact posterior samples from a corpus of one billion labeled responses. We find that no method we evaluate is both efficient and faithful: simple methods like prompting are cheap but plateau at a high KL divergence from the posterior, fine-tuning on posterior samples is o
    
[^118]: 从依赖到过度依赖：AI陪伴中关系风险的分类与检测

    From Reliance to Overreliance: Taxonomizing and Detecting Relational Risks in AI Companionship

    [https://arxiv.org/abs/2611.25734](https://arxiv.org/abs/2611.25734)

    本文提出关系风险分类法，将AI陪伴中的情感过度依赖操作化为可观察的对话行为，基于110名用户的真实对话发现过度依赖行为普遍存在（63.2%）而实际关系伤害较少（10.5%），并验证了开源大语言模型可有效检测这些风险行为。

    

    伴侣型AI日益被用于情感支持，然而区分有意义的互动与有害的依赖仍然十分困难。我们提出了关系风险分类法，将用户的情感过度依赖操作化为可观察的对话行为，并单独识别关系伤害的证据，从而能够实证考察依赖行为何时与不良后果共同出现。通过分析110名成年人在Character.AI和Replika上的真实自然对话，我们发现63.2%的非角色扮演对话中存在与情感过度依赖相关的行为，而可观察到的关系伤害仅出现在10.5%的对话中。尽管情感过度依赖并不必然导致伤害，但反复出现的行为模式往往与有害结果共同出现，这表明其作为风险指标的价值。开源大语言模型在至少一种配置下能够令人满意地检测出14种情感过度依赖行为中的12种。定性分析揭示了对关怀的期待、反复的寻求支持……

    arXiv:2611.25734v1 Announce Type: cross  Abstract: Companion AIs are increasingly used for emotional support, yet distinguishing meaningful engagement from harmful reliance remains difficult. We introduce the Taxonomy of Relational Risks (TRR), which operationalizes users' emotional overreliance (EO) as observable conversational behaviors and separately identifies evidence of relational harm (RH), enabling empirical examination of when reliance behaviors co-occur with adverse consequences. Using naturalistic Character.AI and Replika conversations from 110 adults, we found EO-related behaviors in 63.2% of non-roleplay conversations, while observable relational harms appeared in 10.5%. Although EO did not necessarily lead to harm, recurring patterns often co-occurred with harmful outcomes, suggesting value as risk indicators. Open-source LLMs satisfactorily detected 12 of 14 EO behaviors in at least one configuration. Qualitative analysis revealed expectations of care, repeated support s
    
[^119]: Auto-Alpha：一个自主因子研究框架

    Auto-Alpha: A Framework for Autonomous Factor Research

    [https://arxiv.org/abs/2611.25732](https://arxiv.org/abs/2611.25732)

    Auto-Alpha是一个通过测试驱动的实现循环将报告引导的初始化与自主探索研究相结合的多智能体因子研究框架，借助多重审查机制与经验库持续构建并优化因子库，其因子中位1日Rank IC达0.0301，周频Top-300组合年化收益达16.81%。

    

    自主因子研究既需要生成经济学假设，也需要对其实施和检验。我们提出Auto-Alpha，一个通过测试驱动的实现循环将“报告引导的初始化”与“自主探索研究”相连接的框架。一个想法智能体提出经济机制与信号构造；一个研究智能体则结合回测反馈对其进行开发。通过语义审计、量化审查和人工准入，筛选出的因子进入一个持久化的因子库，同时一个经验库用于沉淀研究洞见。从120份研究报告出发，Auto-Alpha构建了包含32个因子的初始因子库。随后它持续生成新想法并改进现有候选因子，共产出420个实现，并额外保留62个因子。我们基于包含105个因子的固定库快照来评估因子质量与组合表现：1日Rank IC中位数为0.0301，周频Top-300组合年化收益达16.81%（扣除……后）。（注：摘要原文在此处截断）

    arXiv:2611.25732v1 Announce Type: cross  Abstract: Autonomous factor research requires generating economic hypotheses as well as implementing and testing them. We present Auto-Alpha, a framework that connects report-guided initialization to self-directed research through a test-driven implementation loop. An idea agent proposes economic mechanisms and signal constructions; a research agent develops them with backtest feedback. Semantic audit, quantitative review, and human admission select factors for a persistent library, while an Experience Library retains research insights. Starting from 120 research reports, Auto-Alpha builds an initial library of 32 factors. It then generates new ideas and refines existing candidates, producing 420 implementations and retaining 62 additional factors. We evaluate factor quality and portfolio performance using a fixed library snapshot of 105 factors. The median 1-day Rank IC is 0.0301, and a weekly Top-300 portfolio returns 16.81% annualized after s
    
[^120]: 先验证再增强：面向红外小目标检测的目标证据链验证

    Verifying Before Enhancing: Target Evidence Chain Verification for Infrared Small Target Detection

    [https://arxiv.org/abs/2611.25724](https://arxiv.org/abs/2611.25724)

    提出VBE-UNet网络，通过反事实背景解释建模局部目标证据，并结合跨尺度支持与扰动贡献的联合证据验证来评估候选目标可靠性，从而指导特征重组与精细化，突破仅依赖响应强度判别红外小目标的局限。

    

    红外小目标检测对于预警、监视、遥感以及天基目标感知至关重要。然而，红外小目标通常尺寸微小、纹理微弱且边界模糊，难以与复杂背景区分开来。此外，类目标的背景结构可能在多个尺度上产生显著响应，仅凭响应强度不足以实现可靠的目标判别。为应对这一挑战，我们提出了VBE-UNet——一种目标证据链验证网络，通过互补的证据线索在响应幅度之外评估候选目标的可靠性。具体而言，反事实背景解释对局部目标证据进行建模，而联合证据验证则整合跨尺度支持与基于扰动的贡献验证，以指导特征重组。由此得到的可靠性估计用于指导选择性特征精细化。

    arXiv:2611.25724v1 Announce Type: cross  Abstract: Infrared small target detection is essential for early warning, surveillance, remote sensing, and space-based target perception. However, infrared small targets are typically tiny, weakly textured, and boundary-blurred, making them difficult to distinguish from complex backgrounds. Moreover, target-like background structures can produce salient responses across multiple scales, making response strength alone insufficient for reliable target discrimination. To address this challenge, we propose VBE-UNet, a Target Evidence Chain Verification Network that assesses candidate reliability beyond response magnitude through complementary evidence cues. Specifically, counterfactual background explanation models local target evidence, while joint evidence verification integrates cross-scale support and perturbation-based contribution verification to guide feature recomposition. The resulting reliability estimates guide selective feature refineme
    
[^121]: MemMend：面向记忆问答的选择性修订框架

    MemMend: A Selective Revision Framework for Memory Question Answering

    [https://arxiv.org/abs/2611.25721](https://arxiv.org/abs/2611.25721)

    MemMend通过选择性路由与有界精确文本替换机制，仅对记忆问答中可能不完整的答案进行修订，在LoCoMo基准上以减少62.37%修订调用量的代价实现了更高的F1与Judge分数。

    

    对来自长期记忆的每一个答案都进行修订可能会消耗大量推理预算，并破坏原本已经正确的事实。我们提出MemMend，一个模块化的选择性修订框架，它将可能不完整的答案路由至编辑器，将修正表示为有界的精确文本替换，并且仅接受具有可解析源坐标和有效编辑操作的编辑。当没有提出合法的修复方案时，编辑器会保留当前答案；被接受的编辑无需额外的语义验证调用。在1,986个LoCoMo开发问题上，MemMend取得了67.17的对话宏平均F1和88.35的平均Judge分数，相比经过验证的修订链分别提升了0.82分和1.24分，同时修订调用量减少了62.37%。与无门控的本地审查相比，路由机制少用了54.15%的修订调用，F1相近，但Judge分数低0.67分。当直接应用于原生的PropMem、Mem0和SimpleMem答案时，该编辑接口（原文在此处截断）

    arXiv:2611.25721v1 Announce Type: new  Abstract: Revising every answer from a long-term memory can consume substantial inference budget and damage facts that were already correct. We introduce MemMend, a modular selective revision framework that routes potentially incomplete answers to an editor, represents corrections as bounded exact-text replacements, and accepts only edits with resolvable source coordinates and valid edit operations. The editor preserves the current answer when no legal repair is proposed; accepted edits require no additional semantic-verification call. On 1,986 LoCoMo development questions, MemMend achieves 67.17 conversation-macro F1 and 88.35 mean Judge score, improving over a verified revision chain by 0.82 and 1.24 points while using 62.37% fewer revision calls. Routing uses 54.15% fewer revision calls than ungated local review, with similar F1 but a 0.67-point lower Judge score. Applied directly to native PropMem, Mem0, and SimpleMem answers, the editing inte
    
[^122]: SteerTrace：追踪目标攻击如何操纵世界动作模型

    SteerTrace: Tracing How Target Attacks Steer World Action Models

    [https://arxiv.org/abs/2611.25719](https://arxiv.org/abs/2611.25719)

    提出SteerTrace——首个针对世界动作模型的定向通用攻击方法，仅需一个跨帧、任务和视角共享的微小扰动即可将机器人动作推向攻击者目标、使任务成功率接近零，并通过追踪内部效应发现恢复未来token可消除61-94%的定向攻击影响。

    

    世界动作模型（WAM）通过单一骨干网络同时预测未来视频和动作，其“想象的未来”常被视为一种安全信号。我们提出了SteerTrace，这是一种针对WAM的定向通用攻击方法，同时还能追踪攻击在模型内部产生的影响。每个模型仅需一个ℓ∞范数约束（4/255）的扰动，且该扰动跨帧、跨任务、跨相机视角共享，即可将模型执行的动作推向攻击者选定的目标。在来自四个模型家族（FastWAM、Faster-WAM、OpenWAM和LingBot-VA）的八个WAM上，基于LIBERO和RoboTwin基准测试，该攻击使任务成功率降至接近零，并将受攻击的动作推向目标方向；而相同预算的随机噪声和BadWAM在目标方向上仅产生很小的动作偏移。我们利用带符号的定向效应来衡量每种干预手段消除或重现了多少攻击影响。在标准攻击下，恢复未来token（future tokens）可以在五个动作专家关注未来token的模型中消除61-94%的定向攻击效应。当……（摘要原文在此处截断）

    arXiv:2611.25719v1 Announce Type: cross  Abstract: World action models (WAMs) predict future videos and actions with one backbone, and their imagined future is often seen as a safety signal. We introduce SteerTrace, a targeted universal attack on WAMs that also traces its effect inside the model. One $\ell_\infty$-bounded (4/255) perturbation per model, shared across frames, tasks, and camera views, pushes the executed action toward an attacker-chosen target. On eight WAMs from four families (FastWAM, Faster-WAM, OpenWAM and LingBot-VA) on LIBERO and RoboTwin, it drives task success close to zero and moves the attacked action toward the target. Same-budget noise and BadWAM produce little action shift in the target direction. We use the signed targeted effect to measure how much of the effect each intervention removes or reproduces. Under the standard attack, restoring future tokens removes 61-94% of the targeted effect in the five models whose action experts attend to them. When the at
    
[^123]: STARK：迈向真正会学习与记忆的个性化智能体

    STARK: Moving towards Personalized Agents That Truly Learn and Remember

    [https://arxiv.org/abs/2611.25705](https://arxiv.org/abs/2611.25705)

    STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。

    

    长期记忆对于在长时间内与用户交互的个性化智能体系统至关重要，然而当前的架构只能捕获“发生了什么”，而将“为什么会发生”的推理留给底层大语言模型事后推导。这些事件不仅能够传达用户经历了什么，还能传达这些经历如何塑造了用户的行为模式。我们提出了STARK，一种双图记忆架构，它将这些事件不仅视为节点，还视为行为洞察之间的边。STARK维护一个情景图，将交互记录为在时间、因果和语义上相互关联的情景；以及一个洞察图，用于表示由大语言模型提取的行为信号，并根据支持这些信号的事件数量进行评分，同时通过引发这些洞察的事件将它们相互连接。在推理时，行为知识可以直接从由事件作为边所支撑的洞察节点中检索获得。

    arXiv:2611.25705v1 Announce Type: cross  Abstract: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-ed
    
[^124]: 从地理记忆到大语言模型中的空间泛化

    From Geographic Memory to Spatial Generalization in Large Language Models

    [https://arxiv.org/abs/2611.25703](https://arxiv.org/abs/2611.25703)

    该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。

    

    大语言模型（LLM）能够将地点与地理描述相关联，但恢复训练中观察到的关联并不能证明其具备跨地理空间泛化的能力。我们区分了“地理空间记忆”（即恢复训练期间观察到的位置—描述关联）与“空间泛化”（即为未观察区域中的位置预测描述）。我们研究连续的位置表示能否改善这种泛化能力，并促使地理知识在大语言模型内部形成空间化的组织。我们将这种组织形式构建为一个潜在空间场：一种通过空间依赖关系连接的、从位置到描述的学习映射。基于连续空间编码器，我们将位置表示投影到模型的嵌入空间中，并将其作为空间标记提供。我们在两个新构建的地理文本基准（分别针对洛杉矶和伦敦）上评估了所得模型。

    arXiv:2611.25703v1 Announce Type: cross  Abstract: Large language models (LLMs) can associate places with geographic descriptions, but recovering an observed association does not establish an ability to generalize across geographic space. We distinguish geospatial memory, the recovery of location--description associations observed during training, from spatial generalization, the prediction of descriptions for locations in unobserved regions. We investigate whether continuous location representations improve this generalization and encourage geographic knowledge to become spatially organized within an LLM. We frame this organization as a latent spatial field: a learned mapping from locations to descriptions connected through spatial dependencies. Building on a continuous spatial encoder, we project location representations into the model's embedding space and supply them as spatial tokens. We evaluate the resulting models on two newly curated geo-text benchmarks for Los Angeles and Lon
    
[^125]: DMT-Attack：面向闭环视觉-语言-动作策略的扩散先验纹理搜索

    DMT-Attack: Diffusion-Prior Texture Search for Closed-Loop Vision-Language-Action Policies

    [https://arxiv.org/abs/2611.25695](https://arxiv.org/abs/2611.25695)

    提出DMT-Attack，一种无梯度灰盒攻击方法，通过在冻结扩散模型的图像分布中搜索自然外观的纹理，并利用闭环滚动反馈优化八专家潜变量提案网络，从而对视觉-语言-动作策略实施隐蔽且有效的持续性纹理攻击。

    

    视觉-语言-动作（VLA）策略在整个操作过程中依赖视觉反馈，这使其容易受到持续性纹理攻击。然而，优化出的图案往往会与周围环境格格不入，使攻击更容易被察觉，其载体也更容易被检查或移除。针对这一局限，我们的关键洞察是：在扩散模型的结构化图像分布中搜索外观自然的纹理，同时利用闭环反馈来识别具有任务破坏性的候选纹理。基于这一思路，我们提出了DMT-Attack，这是一种无梯度的灰盒攻击方法，能够为桌布等普通场景物体生成纹理。该方法使用一个冻结的文本条件扩散模型定义搜索空间，并采用一个由八位专家组成的潜变量提案网络，该网络从经过评估的教师潜变量初始化，并通过闭环滚动反馈进行优化。我们在四个LIBERO测试套件上针对pi_0.5和OpenVLA-OFT评估了DMT-Attack。DMT-Attack实现了（摘要原文在此处截断）

    arXiv:2611.25695v1 Announce Type: new  Abstract: Vision-language-action (VLA) policies rely on visual feedback throughout manipulation, making them vulnerable to persistent texture attacks. However, optimized patterns can stand out from their surroundings, making an attack easier to notice and its carrier more likely to be inspected or removed. To address this limitation, our key insight is to search the structured image distribution of a diffusion model for natural-looking textures while using closed-loop feedback to identify task-disruptive candidates. Based on this idea, we propose DMT-Attack, a gradient-free gray-box attack that generates textures for ordinary scene objects such as tablecloths. A frozen text-conditioned diffusion model defines the search space, and an eight-expert latent proposal network is initialized from evaluated teacher latents and refined using rollout feedback. We evaluate DMT-Attack against pi_0.5 and OpenVLA-OFT across four LIBERO suites. DMT-Attack achiev
    
[^126]: TipDex-VTLA：学习在灵巧操作中何时利用触觉精修动作

    TipDex-VTLA: Learning When to Refine Actions with Touch for Dexterous Manipulation

    [https://arxiv.org/abs/2611.25683](https://arxiv.org/abs/2611.25683)

    提出 TipDex-VTLA 策略，将基于冻结参考策略光照敏感性的视觉动作可靠性估计（VASE）与执行期触觉预测校验（TACE）相结合，动态决定何时及如何对未执行动作施加有界的触觉校正，从而抑制光照变化引起的灵巧操作动作漂移。

    

    即使物理状态保持不变，光照变化也会在灵巧操作中引发动作漂移。尽管现有的视觉-触觉-语言-动作（VTLA）策略已融入触觉反馈，但在视觉条件变化时，触觉何时以及如何精修动作仍然是一个具有挑战性的问题。我们提出 TipDex-VTLA，一种结合估计的视觉动作可靠性与触觉预测校验来调控执行时精修的 VTLA 策略。具体而言，视觉动作敏感性估计（VASE）从冻结的参考策略对光照扰动的动作敏感性中学习视觉可靠性；执行过程中的触觉预期与校正（TACE）在匹配时刻比较预测触觉与实测触觉，并将由此产生的差异与视觉可靠性相结合，为尚未执行的动作进行有界的特征校正门控。这有效地帮助在具有挑战性的未见条件下精修动作（注：原文摘要在此处截断）。

    arXiv:2611.25683v1 Announce Type: cross  Abstract: Changes in illumination can induce action drift in dexterous manipulation even when the physical state remains unchanged. Although existing vision--tactile--language--action (VTLA) policies incorporate tactile feedback, when and how touch refines actions under changing visual conditions remains challenging. We introduce \textbf{TipDex-VTLA}, a VTLA policy that combines estimated visual action reliability and tactile forecast verification to regulate execution-time refinement. Specifically, Visual Action Sensitivity Estimation (VASE) learns visual reliability from a frozen reference policy's action sensitivity to illumination perturbations. Tactile Anticipation and Correction during Execution (TACE) compares predicted and measured touch at matched times and combines the resulting discrepancy with visual reliability to gate bounded feature corrections for unexecuted actions. This effectively helps to refine actions under challenging unse
    
[^127]: 超越实体识别：以视觉为中心的知识型视觉问答

    Beyond Entity Recognition: Vision-Centric Knowledge Visual Question Answering

    [https://arxiv.org/abs/2611.25680](https://arxiv.org/abs/2611.25680)

    VC-VQA是一个以视觉为中心的知识推理大规模基准，通过要求模型检索外部视觉证据并将其与内部图像进行定位、验证和匹配，系统评估多模态大语言模型在细粒度视觉知识获取与多阶段推理上的能力。

    

    我们提出了VC-VQA，一个面向以视觉为中心的知识推理的大规模基准，旨在系统性地评估多模态大语言模型（MLLMs）在细粒度视觉知识获取与复杂多阶段推理方面的能力。与现有主要将视觉信息视为实体识别、并依赖基于文本的外部知识的基准不同，VC-VQA通过同时考虑视觉信息的来源（内部图像与外部图像）及其参与阶段（检索前与检索后），显式地建模视觉信息的作用。具体而言，VC-VQA要求模型检索外部视觉证据、在检索前识别视觉线索，并将检索到的知识与内部图像进行定位、验证或匹配。我们进一步构建了一个可控的多模态知识库，为每个样本提供可追溯的文本与视觉证据。我们在四种设置下评估了21个具有代表性的多模态大语言模型……

    arXiv:2611.25680v1 Announce Type: cross  Abstract: We propose VC-VQA, a large-scale benchmark for vision-centric knowledge reasoning, aiming to systematically evaluate the capabilities of Multimodal Large Language Models (MLLMs) in fine-grained visual knowledge acquisition and complex multi-stage reasoning. Unlike existing benchmarks that primarily treat visual information as entity recognition and rely on text-based external knowledge, VC-VQA explicitly models the role of visual information by considering both its source (internal vs. external images) and stage of involvement (pre-retrieval vs. post-retrieval). Specifically, VC-VQA requires models to retrieve external visual evidence, identify visual clues before retrieval, and localize, verify, or match retrieved knowledge against the internal image. We further construct a controlled multimodal knowledge base with traceable textual and visual evidence for each sample. We evaluate 21 representative MLLMs under four settings: no extern
    
[^128]: CompAct-Bench：面向长周期智能体任务中工作上下文压缩的基准测试

    CompAct-Bench: A Benchmark for Working-Context Compaction in Long-Horizon Agent Tasks

    [https://arxiv.org/abs/2611.25673](https://arxiv.org/abs/2611.25673)

    提出 CompAct-Bench 基准，通过压缩上下文后智能体能否继续成功完成原始任务的下游效果，来评估大语言模型在长周期智能体任务中的工作上下文压缩能力。

    

    随着语言模型智能体被应用于日益延长的任务，其工作上下文会不断累积交互历史、工具观测结果、中间决策和任务进度。上下文压缩提供了一种在允许任务执行继续进行的同时缩减这一不断增长的历史的实用方法，然而现代大语言模型的压缩能力仍未被充分理解。我们提出了 CompAct-Bench，这是一个通过压缩对智能体任务完成情况的下游影响来评估工作上下文压缩能力的基准测试。该基准从成功的智能体轨迹出发，在任务执行的不同阶段插入压缩边界，使用被评估的模型压缩先前的工作上下文，然后让同一个执行器继续完成原始任务。为了实现对不同压缩器的公平比较，我们在受控的保留率下，采用一种轻量级的迭代式比例调整程序对它们进行评估。CompAct-Bench 涵盖了信息……（摘要在此处被截断）

    arXiv:2611.25673v1 Announce Type: new  Abstract: As language-model agents are applied to increasingly extended tasks, their working context continuously accumulates interaction history, tool observations, intermediate decisions, and task progress. Context compaction offers a practical way to reduce this growing history while allowing execution to continue, yet the compaction ability of modern LLMs remains insufficiently understood. We introduce CompAct-Bench, a benchmark for evaluating working-context compaction through its downstream effect on agent task completion. Starting from successful agent trajectories, we insert compaction boundaries at different stages of execution, compress the preceding working context with the model under evaluation, and let the same executor continue the original task. To enable fair comparison across compactors, we evaluate them under controlled retention ratios using a lightweight iterative ratio-adjustment procedure. CompAct-Bench covers information-se
    
[^129]: LATER：面向预测性世界模型的动作条件时序证据检索学习

    LATER: Learning Action-Conditioned Temporal Evidence Retrieval for Predictive World Models

    [https://arxiv.org/abs/2611.25671](https://arxiv.org/abs/2611.25671)

    LATER提出了一种基于CF-JEPA的动作条件世界模型，通过将早期真实转移存储于分区记忆并在推演时根据状态-动作对进行选择性读取，在受干扰的PushT任务中将成功率从15.5%提升至25.0%，并将干扰条件下的沿推方向MAE降低20.2%。

    

    早期的交互可以揭示后续观察不再能显示的物理属性，而这些属性仍然能够支配未来动作的效果。受工作记忆和认知科学中“保留”与“选择性读取”区分的启发，我们提出了LATER，一个基于CF-JEPA构建的动作条件世界模型。LATER将已完成的真实状态转移存储在分区的记忆中，这些记忆在候选推演过程中保持固定不变。一个响应桥利用当前状态和候选动作查询这些记忆，同时多切割辅助监督用于训练记忆写入器。在受干扰的两阶段PushT任务上，与重构的CF-JEPA相比，LATER将最终成功率从15.5%提升至25.0%。在另一项涵盖六个模型和两个从零训练随机种子的对比实验中，LATER在干净视图和受干扰视图下均取得了最低的平均物理读取误差，在干扰物存在的条件下将沿推方向的平均绝对误差（MAE）降低了20.2%。

    arXiv:2611.25671v1 Announce Type: new  Abstract: Early interactions can reveal physical properties that later observations no longer show. These properties can still gov- ern the effects of future actions. Inspired by the distinction between retention and selective readout in working mem- ory and cognitive science, we introduce LATER, an action- conditioned world model built on CF-JEPA. LATER stores completed real transitions in partitioned memories that remain fixed during candidate rollouts. A response bridge queries these memories using the current state and candidate action, while multi-cut auxiliary supervision trains the memory writ- ers. On distracted Two-Stage PushT, LATER raises terminal success from 15.5% to 25.0% over reconstructed CF-JEPA. In a separate comparison across six models and two scratch- training seeds, LATER achieves the lowest mean physical readout error in both clean and distracted views, reducing along-push mean absolute error (MAE) under distractors by 20.2%
    
[^130]: 基于动态感知量化与蒸馏的视觉语言模型高效潜在推理

    Efficient Latent Reasoning for Vision-Language Model via Dynamics-Aware Quantization and Distillation

    [https://arxiv.org/abs/2611.25670](https://arxiv.org/abs/2611.25670)

    提出QL-VLM框架，通过动态感知量化与蒸馏技术，在大幅压缩视觉语言模型潜在思维链推理的存储与计算成本的同时保持推理精度。

    

    视觉语言模型中高效的思维链推理需要同时减少中间推理步骤及其执行成本。潜在思维链方法将文本推理过程压缩为连续思维，但其全精度模型仍保留了大量存储和计算开销。降低数值精度可以减少这些剩余成本，然而直接量化会严重降低推理精度，尤其是在4位权重和激活值的情况下。算子与扰动分析揭示了连续思维误差的关键敏感性以及方向相关的预测效应，这促使我们在重建精度之外保留信息检索与后续推理的能力。因此，我们提出了QL-VLM，一个面向视觉语言模型高效潜在推理的动态感知量化框架。首先，动态引导量化使用非均匀码本来保留旋转……

    arXiv:2611.25670v1 Announce Type: new  Abstract: Efficient chain-of-thought (CoT) reasoning in vision-language models (VLMs) requires reducing both intermediate reasoning steps and their execution cost. Latent CoT methods compress textual rationales into continuous thoughts, yet their full-precision (FP) models retain substantial storage and computation costs. Lowering numerical precision could reduce these remaining costs, yet direct quantization severely degrades reasoning accuracy, particularly with 4-bit weights and activations. Operator and perturbation analyses reveal key sensitivity and direction-dependent predictive effects of continuous-thought errors, motivating preservation of information retrieval and subsequent reasoning beyond reconstruction accuracy. We therefore introduce QL-VLM, a dynamics-aware Quantization framework for efficient Latent reasoning in Vision-Language Models (VLMs). First, Dynamics-Guided Quantization (DGQ) uses nonuniform codebooks that preserve rotary
    
[^131]: 在人群中提问：基于虚拟查询增强混洗的差分隐私大语言模型推理

    Ask in the Crowd: Differentially Private LLM Inference via Dummy-Augmented Shuffling

    [https://arxiv.org/abs/2611.25669](https://arxiv.org/abs/2611.25669)

    该论文提出DP-SPD差分隐私推理框架，通过将真实查询隐藏于客户端生成的虚拟查询中并随机混洗来保护用户隐私，同时利用查询共享前缀结构与KV缓存复用提升推理效率，从而更好地平衡隐私保护与模型效用。

    

    大语言模型越来越多地以“模型即服务”（LLMaaS）的形式部署，用户查询被发送到远程大语言模型进行推理，这引发了关于敏感用户输入暴露的严重隐私担忧。差分隐私推理通过向输入嵌入中注入噪声来缓解这些风险。然而，现有方法难以在隐私和效用之间取得平衡，因为更强的隐私保护往往会导致大语言模型推理精度的大幅下降。在这项工作中，我们提出了DP-SPD——一个差分隐私的大语言模型推理框架，它通过将真实查询隐藏在客户端生成的虚拟查询之中来保护用户查询。所有查询在发送到服务器之前会被随机混洗，从而防止直接识别用户的真实意图。为了提高推理效率，我们利用查询之间的共享前缀结构，并在解码过程中复用Transformer的键值缓存。我们分析了该方法的隐私保证……

    arXiv:2611.25669v1 Announce Type: new  Abstract: Large language models are increasingly deployed as LLM-as-a-Service (LLMaaS), where user queries are sent to remote LLMs for inference, raising severe privacy concerns about the exposure of sensitive user inputs. Differentially private inference mitigates these risks by injecting noise into input embeddings. However, existing methods struggle to balance privacy and utility, as stronger privacy protection often leads to substantial accuracy degradation in LLM inference.   In this work, we propose DP-SPD, a differentially private LLM inference framework that protects user queries by hiding the real query among client-generated dummy queries. All queries are randomly shuffled before being sent to the server, preventing direct identification of the user's true intent. To improve inference efficiency, we exploit the shared-prefix structure among queries and reuse transformer key--value (KV) caches during decoding. We analyze the privacy guara
    
[^132]: GRODE-RTL：面向基于大语言模型的RTL修复的源码映射依赖视图

    GRODE-RTL: Source-Mapped Dependency Views for LLM-Based RTL Repair

    [https://arxiv.org/abs/2611.25661](https://arxiv.org/abs/2611.25661)

    GRODE-RTL提出了一种以锚点为中心的表示方法，通过提取可疑寄存器的单时钟周期扇入锥、将网表节点映射回源代码行并构建紧凑的按跳分区依赖视图，帮助大语言模型在不依赖oracle反馈的情况下有效修复根源与故障输出相距甚远的非局部RTL故障。

    

    大语言模型在自动寄存器传输级（RTL）修复方面展现出巨大潜力，然而它们难以处理根源与故障输出在源代码中相距甚远的非局部故障。将RTL视为扁平的token序列会掩盖硬件数据流，而暴露完整的综合图则会用细粒度的元数据淹没模型。为解决这一表示困境，我们提出了GRODE-RTL，这是一种以锚点为中心的表示方法，它从网表中提取可疑寄存器的单时钟周期扇入锥，将网表节点映射回可编辑的源代码行，并将它们组织成紧凑的按跳数分区的依赖视图。通过优先考虑直接驱动信号并折叠远距离的跳数，GRODE-RTL能够在不暴露真实标签提示或oracle反馈的情况下引导标准的工具使用智能体。我们在30个经过形式化验证的非局部故障上，使用三个大语言模型（DeepSeek-V4-Flash、GLM-5.2和MiniMax-M3）对GRODE-RTL进行了评估。

    arXiv:2611.25661v1 Announce Type: cross  Abstract: Large Language Models show great potential for automated Register-Transfer Level repair, yet they struggle with non-local faults where root causes and failing outputs reside far apart in source code. Treating RTL as a flat token sequence obscures hardware dataflows, whereas exposing complete synthesized graphs overwhelms the model with fine-grained metadata. To resolve this representation dilemma, we propose GRODE-RTL, an anchor-centric representation that extracts the single-clock-stage fan-in cone of a suspect register from the netlist, maps netlist nodes back to editable source lines, and organizes them into a compact hop-partitioned dependency view. By prioritizing direct drivers and folding distant hops, GRODE-RTL guides standard tool-using agents without exposing ground-truth hints or oracle feedback. We evaluate GRODE-RTL across 30 formally validated non-local faults using three LLMs (DeepSeek-V4-Flash, GLM-5.2, and MiniMax-M3) 
    
[^133]: SiliconGym：面向智能体化EDA的多功能强化学习环境

    SiliconGym: A Versatile Reinforcement Learning Environment for Agentic EDA

    [https://arxiv.org/abs/2611.25658](https://arxiv.org/abs/2611.25658)

    本文提出SiliconGym，首个面向硬件设计与验证的跨任务、跨工具链强化学习环境，通过解耦版本化工作区、预算化执行与验收门控评分，实现从模块级RTL任务到全层次SoC流程的可扩展训练，并借助流式调度将RL批次完成速度提升21.9%。

    

    利用电子设计自动化（EDA）反馈的强化学习（RL）正在推动硬件设计与验证领域大语言模型的发展。要将这些成果扩展到硬件工程智能体，需要能够跨越多种任务、在工具调用之间保持状态，并同时支持大型层次结构与并行回合的环境。我们提出了SiliconGym，这是首个面向硬件设计与验证的多功能、跨任务、跨工具链的强化学习环境。通过解耦版本化工作区、预算化执行和验收门控评分，SiliconGym能够从模块级的RTL、测试平台和PPA任务，扩展到多轮代码仓库修复以及全层次SoC流程。流式调度将工具检查与生成过程重叠，使一个包含256个答案的RL批次的完成速度比原生ChipSeek快21.9%。借助可互换的开源与商业后端，我们证明了在开源工具上训练获得的功能正确性可以顺利迁移到商业si（摘要在此处截断）

    arXiv:2611.25658v1 Announce Type: new  Abstract: Reinforcement learning (RL) with electronic design automation (EDA) feedback is advancing large language models for hardware design and verification. Extending these gains to hardware-engineering agents requires environments that span tasks, preserve state across tool calls, and support large hierarchies alongside parallel rollouts. We present SiliconGym, the first versatile, cross-task and cross-toolchain RL environment for hardware design and verification. By decoupling versioned workspaces, budgeted execution, and acceptance-gated scoring, SiliconGym scales from module-level RTL, testbench, and PPA tasks to multi-round repository repair and full-hierarchy SoC flows. Streaming dispatch overlaps tool checks with generation, finishing a 256-answer RL batch 21.9% faster than native ChipSeek. Using interchangeable open-source and commercial backends, we show that open-source-trained functional correctness transfers cleanly to commercial si
    
[^134]: A²-VLA：通过智能体可供性发现将视觉-语言-动作策略适配到新工具

    A$^2$-VLA: Adapting Vision-Language-Action Policies to New Tools through Agentic Affordance Discovery

    [https://arxiv.org/abs/2611.25656](https://arxiv.org/abs/2611.25656)

    该论文提出A²-VLA框架，将具身智能体与VLA策略相结合，把工具可供性视为可构建、测试和修正的假设，通过在重建的仿真场景中自主发现功能区域及其关键几何属性等任务相关可供性特征，实现操作技能向新工具的迁移与适配。

    

    形状不同的工具可能具有相似的功能，但跨工具迁移操作技能需要理解工具几何结构与特定任务之间的关系，并相应地调整操作策略。关键挑战在于发现有助于完成任务的跨工具共性可供性，并验证这些可供性是否支持新工具执行类似的工作。我们提出了A²-VLA，这是一个将具身智能体与VLA策略相结合的工具适配框架。该框架将工具可供性视为智能体可以构建、测试和修正的假设。给定使用少量源工具的遥操作演示，智能体在仿真中重建场景，并自主确定与任务相关的可供性特征：功能区域及其关键几何属性。它通过在重建场景中的自主实验来发展可供性提取技能，并对特征和技能进行迭代改进。（注：原摘要文本不完整，结尾处被截断）

    arXiv:2611.25656v1 Announce Type: cross  Abstract: Tools with different shapes can have similar functions, but transferring manipulation skills across tools requires understanding how tool geometry relates to the specific task and adjusting the manipulation strategy accordingly. The key challenge is to discover the common affordances that contribute to task completion across different tools and verify whether they support new tools in performing similar work. We present A$^2$-VLA, a tool-adaptation framework that combines an embodied agent with a VLA policy. It treats tool affordances as hypotheses the agent can construct, test, and revise. Given teleoperated demonstrations with a few source tools, the agent reconstructs the scenes in simulation and autonomously determines task-relevant affordance features: functional regions and their key geometric properties. It develops affordance-extraction skills and refines both the features and skills through autonomous experiments in the recons
    
[^135]: DARSY：一种面向未知动力学随机系统的数据驱动抽象与策略合成工具

    DARSY: A Tool for Data-Driven Abstraction and Policy Synthesis for Stochastic Systems with Unknown Dynamics

    [https://arxiv.org/abs/2611.25654](https://arxiv.org/abs/2611.25654)

    DARSY是一个只需系统模拟器和Lipschitz界的全自动工具，它通过从采样数据构建区间马尔可夫决策过程抽象并结合鲁棒动态规划，为未知动力学的随机控制系统提供具有PAC保证的策略合成与验证。

    

    本文提出了DARSY：一个软件工具，用于对具有未知动力学和噪声分布的离散时间随机控制系统进行基于抽象的验证与策略合成。DARSY只需要系统的模拟器和其动力学的Lipschitz界，无需任何关于动力学或噪声分布的先验知识。该工具完全自动化，从采样的状态转移中构建有限区间马尔可夫决策过程（IMDP）抽象，其概率区间通过统计技术和Lipschitz推理获得。随后在该IMDP上执行鲁棒动态规划，得到一个策略以及未知系统满足期望规约的概率下界。该下界在用户指定的置信度下成立，即具有概率近似正确（PAC）保证。该工具采用Python实现，支持一般的Lipschitz连续系统动力学和规约。

    arXiv:2611.25654v1 Announce Type: cross  Abstract: This paper presents DARSY: a software tool for abstraction-based verification and policy synthesis in discrete-time stochastic control systems with unknown dynamics and noise distributions. DARSY requires only a simulator of the system and a Lipschitz bound on its dynamics, and needs no knowledge of dynamics or noise distributions. The tool is fully automated and constructs a finite interval Markov decision process (IMDP) abstraction from sampled transitions, whose probability intervals are obtained via statistical techniques and Lipschitz reasoning. Robust dynamic programming on this IMDP then yields a policy together with a lower bound on the probability that the unknown system satisfies a desired specification. This lower bound holds with a user-specified confidence; that is, it is probably approximately correct (PAC). The tool is implemented in Python and supports general Lipschitz-continuous system dynamics and specifications expr
    
[^136]: XOR性（异或性）下的多准则决策

    Multiple Criteria Decision Making Under XORness

    [https://arxiv.org/abs/2611.25652](https://arxiv.org/abs/2611.25652)

    本文提出了一个在决策全过程中保留XOR性（不定性、排他性与犹豫性）的多准则决策框架，通过定义XOR数、构建XOR决策矩阵、设计加性XOR聚合算子以及绩效-XOR性指数，实现对备选方案的合理排序。

    

    我们应该如何在XOR性下做出决策？XOR性是一类源于不定性、排他性和犹豫性的不确定性。我们开发了一个在整个决策过程中保持XOR性的多准则决策框架。我们定义了XOR数并证明了其基本的算术与代数性质。基于这些结果，我们将确定性决策矩阵扩展为XOR决策矩阵，并引入了相容的规范化方法。此外，我们提出了一种加性XOR聚合算子，该算子能够保留评估中固有的不定性、排他性和犹豫性。同时，我们引入了一个公理化的XOR性度量来量化与全局绩效得分相关的不确定性程度，并开发了一个绩效-XOR性指数用于对备选方案进行排序。我们证明该指数在备选方案集合上诱导出完全前序，并给出了保持支配关系的条件。

    arXiv:2611.25652v1 Announce Type: new  Abstract: How should we make decisions under XORness? XORness is a class of uncertainty arising from indeterminacy, exclusivity, and hesitancy. We develop a multiple criteria decision making framework that preserves XORness throughout the decision process. We define XOR numbers and prove their fundamental arithmetic and algebraic properties. Building on these results, we extend the deterministic decision matrix to an XOR decision matrix and introduce a compatible normalization procedure. Further, we propose an additive XOR aggregation operator, which preserves indeterminacy, exclusivity, and hesitancy inherent in evaluations. In addition, we introduce an axiomatic XORness measure that quantifies the degree of uncertainty associated with global performance scores and develop a performance--XORness index to rank alternatives. We prove that this index induces a total preorder over the set of alternatives and identify the conditions for preserving dom
    
[^137]: MatVerse：材料科学基础模型

    MatVerse: A Foundation Model for Materials Science

    [https://arxiv.org/abs/2611.25651](https://arxiv.org/abs/2611.25651)

    该论文提出MatVerse，一个基于Intern-S2-Preview并经大规模材料领域多模态后训练的材料科学基础模型，能够统一理解与推理文本知识、晶体结构、表征数据、显微图像等异构材料信息。

    

    材料科学涉及高度异构的科学信息，包括文本知识、分子与晶体结构、实验表征数据、显微镜图像以及构效关系等。开发一个能够理解并对这些多样化材料表示进行理解与推理的通用基础模型仍然充满挑战。在本技术报告中，我们提出了MatVerse，一个基于Intern-S2-Preview构建，并通过大规模材料领域后训练进一步专门化的材料基础模型。我们构建了一个大规模多模态训练语料库，涵盖材料知识、科学推理、定量问题求解、结构理解以及多模态科学解读等内容。为提升领域适应能力，我们开发了一个统一的数据构建流水线，将教师引导的推理蒸馏、科学问题合成、基于一致性的过滤等方法相结合……

    arXiv:2611.25651v1 Announce Type: new  Abstract: Materials science involves highly heterogeneous scientific information, including textual knowledge, molecular and crystal structures, experimental characterization data, microscopy images, and structure-property relationships. Developing a general-purpose foundation model capable of understanding and reasoning over these diverse materials representations remains challenging. In this technical report, we introduce MatVerse, a materials foundation model built upon Intern-S2-Preview and further specialized through large-scale materials-domain post-training. We construct a large-scale multimodal training corpus covering materials knowledge, scientific reasoning, quantitative problem solving, structural understanding, and multimodal scientific interpretation. To improve domain adaptation, we develop a unified data construction pipeline combining teacher-guided reasoning distillation, scientific problem synthesis, consistency-based filtering,
    
[^138]: MambaLIFT：面向VMamba的基于参考、机制感知的归因方法

    MambaLIFT: Reference-Based Mechanism-Aware Attribution for VMamba

    [https://arxiv.org/abs/2611.25644](https://arxiv.org/abs/2611.25644)

    提出了MambaLIFT，一个针对VMamba视觉状态空间模型的基于参考、机制感知的归因框架，能够按空间位置、VSS块、扫描方向和计算机制精确分解模型预测的贡献，并在浮点精度下实现数值守恒。

    

    arXiv:2611.25644v1 公告类型：新论文 摘要：对视觉状态空间模型的预测进行归因仍然具有挑战性，因为依赖输入的选择性动态机制将特征选择、循环状态传播和共享计算路径耦合在一起。我们提出了MambaLIFT，这是一个面向VMamba的基于参考、乘子优先的归因框架，它使用针对选择性扫描注入、循环记忆、LayerNorm和共享路径的交互感知规则，在其计算图中传播贡献。MambaLIFT按空间位置、VSS块、扫描方向和计算机制，对相对于参考的类别logit差异 f_c(x)-f_c(x^0) 进行分解。在ImageNet-1K上的实验评估了VMamba-T和VMamba-B两种模型，超参数在校准集上选取，忠实性则在不相交的独立评估集上进行评估。MambaLIFT在两种配置下的所有经审计的归因接口上均实现了浮点精度级别的数值守恒。

    arXiv:2611.25644v1 Announce Type: new  Abstract: Attributing predictions of visual state-space models remains challenging because input-dependent selective dynamics couple feature selection, recurrent state propagation, and shared computational pathways. We introduce MambaLIFT, a reference-based, multiplier-first attribution framework for VMamba that propagates contributions through its computational graph using interaction-aware rules for selective-scan injection, recurrent memory, LayerNorm, and shared pathways. MambaLIFT decomposes the reference-relative class-logit difference, \(f_c(x)-f_c(x^0)\), by spatial position, VSS block, scan direction, and computational mechanism. Experiments on ImageNet-1K evaluate VMamba-T and VMamba-B, with hyperparameters selected on a calibration set and faithfulness assessed on a disjoint held-out evaluation set. MambaLIFT achieves numerical conservation to floating-point precision across audited attribution interfaces in both configurations. Under m
    
[^139]: 擦除错的对象：共享基础模型嵌入中的条件属性泄露与概念擦除

    Erasing the Wrong Thing: Conditional Attribute Leakage and Concept Erasure in Shared Foundation-Model Embeddings

    [https://arxiv.org/abs/2611.25640](https://arxiv.org/abs/2611.25640)

    论文证明了当任务标签与敏感属性相关时，全局概念擦除方法必然保留任务条件的线性属性泄露，并提出闭式方法TC-LEACE，在每个任务层级内强制实施线性防护以消除条件性敏感属性泄露。

    

    医学图像嵌入越来越多地被共享用于授权的下游任务，以替代原始图像，然而它们可能会泄露敏感属性。标准的概念擦除方法针对的是总体层面的防护性。然而，我们证明，当二元任务标签与敏感标签相关联时，全局防护性和任务条件防护性无法在保留非零线性任务信号的情况下同时成立，因此全局擦除必须要么消除该任务信号，要么保留任务条件的线性泄露。我们进一步展示了全局擦除如何影响线性读取器对具有不同敏感属性的正负样本的判别能力，并从擦除前的统计数据推导出其效用代价的近似闭式预测。受这一冲突的启发，我们提出了任务条件LEACE（TC-LEACE），这是LEACE的一个闭式变体，它在其拟合数据的每个任务层级内强制实施线性防护性。

    arXiv:2611.25640v1 Announce Type: cross  Abstract: Medical-image embeddings are increasingly shared in place of raw images for authorized downstream tasks, yet they may reveal sensitive attributes. Standard concept-erasure methods target population-level guardedness. However, we prove that, when binary task and sensitive labels are associated, global and task-conditional linear guardedness cannot both hold while a nonzero linear task signal is preserved, so global erasure must either eliminate that signal or retain task-conditional linear leakage. We further show how global erasure affects linear readers' discrimination between positive and negative samples with different sensitive attributes, deriving an approximate closed-form prediction of its utility cost from pre-erasure statistics. Motivated by this conflict, we introduce Task-Conditional LEACE (TC-LEACE), a closed-form variant of LEACE that enforces linear guardedness within each task stratum of its fitting data. We evaluate fou
    
[^140]: 超越简单客户端：面向机器人导航的联邦强化学习中难度与改进感知的聚合方法

    Beyond Easy Clients: Difficulty- and Improvement-Aware Aggregation for Federated Reinforcement Learning in Robot Navigation

    [https://arxiv.org/abs/2611.25637](https://arxiv.org/abs/2611.25637)

    提出FedDIA聚合方法，根据客户端本地训练对全局模型的改进程度并结合环境难度进行加权聚合，解决了联邦强化学习中“平台期问题”——即已掌握简单环境的客户端仍以高回报获得不当权重的问题。

    

    联邦强化学习使一群机器人能够共同学习一个共享的导航策略，而无需交换原始经验数据。然而在实践中，机器人很少在相同的条件下运行：开阔的过道远比杂乱的房间容易导航，掌握起来所需的学习也更少。标准的联邦聚合规则忽略了这一点。均匀平均对所有客户端赋予相同的权重，无论其是否仍在学习过程中；而按回合回报加权则效果更差，因为简单环境天然就能产生高回报。这两种失败都源于我们所谓的“平台期问题”：一旦某个客户端掌握了其所处的环境，即便其报告的回报依然很高，其更新也几乎不再携带新的信息。我们提出了FedDIA，该方法根据每个客户端的本地训练对所接收到的全局模型的改进程度来为其评分，并结合其环境难度进行放大。难度是从模型自身……（摘要在此处被截断）

    arXiv:2611.25637v1 Announce Type: cross  Abstract: Federated reinforcement learning enables a fleet of robots to jointly learn a shared navigation policy without exchanging raw experience. In practice, however, the robots rarely operate in the same conditions: an open aisle is far easier to navigate than a cluttered room, and takes less learning to master. Standard federated aggregation rules ignore this. Uniform averaging weights every client equally, whether or not it is still learning, and weighting by episode return is worse, since easy environments permit high returns by construction. Both failures stem from what we call the \emph{plateau problem}: once a client masters its environment, its updates carry little new information even though its reported return remains high. We propose \emph{FedDIA}, which scores each client by how much its local training improves the global model it received, amplified by the difficulty of its environment. Difficulty is inferred from the model's own
    
[^141]: AesDistill：将生成式美学蒸馏用于图像色彩增强

    AesDistill: Distilling Generative Aesthetics for Image Color Enhancement

    [https://arxiv.org/abs/2611.25635](https://arxiv.org/abs/2611.25635)

    提出AesDistill框架，将大型生成模型的丰富美学知识蒸馏到仅1.13M参数的轻量级学生模型中，用于图像色彩增强，突破了传统方法依赖少数专家偏好目标的局限。

    

    图像色彩增强旨在提升图像的视觉吸引力，但以往的方法通常学习复现真值专家目标。这些目标反映的是少数专家的偏好，涵盖的美学选择范围有限，匹配它们并不能保证在不同品味和场景下都获得更强的美学质量。相比之下，大型生成模型从大规模图像集合中学习到了丰富的视觉知识，常常能生成生动、吸引人的图像编辑效果。然而，直接生成可能会在改变色彩的同时改变源图像的语义结构。在本文中，我们提出了AesDistill，一个将生成式美学蒸馏到图像色彩增强中的框架。我们使用4个领先的开源大型生成模型生成了64,644张编辑图像作为教师目标，耗费约3,360个H100 GPU小时，构成了AesDistill数据集。AesDistill还包含AesDistill Student，一个1.13M参数的模型，用于学习教师模型的知识。

    arXiv:2611.25635v1 Announce Type: cross  Abstract: Image color enhancement aims to improve visual appeal, but previous methods often learn to reproduce ground-truth expert targets. These targets reflect the preferences of a few experts and cover a limited range of aesthetic choices. Matching them does not ensure stronger aesthetic quality across different tastes and scenes. By contrast, large generative models learn rich visual knowledge from large image collections and often produce vivid, appealing image edits. However, direct generation can change the source image's semantic structure along with its colors. In this paper, we introduce AesDistill, a framework for distilling generative aesthetics into image color enhancement. We use 4 leading open-source large generative models to produce 64,644 edited images as teacher targets at a cost of about 3,360 H100 GPU-hours, forming the AesDistill Dataset. AesDistill also includes AesDistill Student, a 1.13M-parameter model that learns teach
    
[^142]: DAWN：通过交替维纳引导实现鲁棒的神经盲去卷积

    DAWN: Robust Neural Blind Deconvolution via Alternating Wiener Guidance

    [https://arxiv.org/abs/2611.25621](https://arxiv.org/abs/2611.25621)

    提出 DAWN 框架，通过交替地将基于模型的维纳估计注入模型输入并利用网络自身先验对其进行精炼，在高噪声条件下实现了无需训练数据的鲁棒盲图像去卷积。

    

    盲图像去卷积（BID）是一个病态逆问题，旨在从单幅退化观测中联合恢复清晰图像和模糊核。在摄影和显微镜等科学成像中常见的显著高噪声条件下，BID 尤其具有挑战性，因为现有方法存在噪声过拟合和模糊核估计不准确的问题。现有的无训练数据 BID 学习方法，例如基于深度图像先验（DIP）的方法，通常通过将基于模型的估计作为额外损失项来引入外部引导，这在显著噪声条件下尤其未能充分利用这些估计所携带的信息。为应对这些挑战，我们提出 DAWN，这是一个无需训练数据的 DIP 框架，它在将基于模型的估计直接注入模型输入与利用网络自身先验精炼这一不完美估计之间交替进行。为了提供更可靠的核估计……

    arXiv:2611.25621v1 Announce Type: cross  Abstract: Blind image deconvolution (BID) is an ill-posed inverse problem that aims to jointly recover a sharp image and blur kernel from a single degraded observation. Under nontrivial and high noise, which is common in photography and scientific imaging such as microscopy, BID remains particularly challenging, as existing methods suffer from noise overfitting and inaccurate kernel estimation. Existing training-data-free learning methods for BID, such as those based on the deep image prior (DIP), often utilize external guidance by including model-based estimates as additional loss terms, which underexploits the information these estimates carry especially under nontrivial noise. To address these challenges, we propose DAWN, a training-data-free DIP framework that alternates between injecting model-based estimates directly into the model input and refining this imperfect estimate with the network's own prior. To provide more reliable kernel esti
    
[^143]: 面向AI无线接入网络的信息物理递归自我改进：案例研究

    Cyber-Physical RSI for AI-RAN: A Case Study

    [https://arxiv.org/abs/2611.25619](https://arxiv.org/abs/2611.25619)

    本文提出了一种面向AI-RAN的信息物理递归自我改进框架，将物理测量结果作为证据同时驱动模型自适应与实验规划，并通过乒乓切换案例研究验证了物理反馈通路的有效性。

    

    面向无线接入网络的习得模型通常基于仿真开发，然而当其暴露于真实物理传播条件、协议行为和设备实现时，其预测结果可能发生偏移。本文提出了一种面向AI-RAN的信息物理递归自我改进（RSI）框架，该框架将面向任务的学生模型、实验规划教练智能体、可校准的虚拟环境以及物理宿主智能体连接起来。物理测量结果被组织为证据，既用于模型自适应，也用于后续的实验规划。我们以乒乓切换为案例研究，考察了物理反馈通路的实现。该系统完成了12次物理执行（组织为六组基线-候选对），根据测量结果更新学生模型，并在预留的物理配置上评估修订后的模型。在观察到的短窗口测试中，将触发时间从320增加到128……（原文摘要在此处截断）

    arXiv:2611.25619v1 Announce Type: new  Abstract: Learned models for radio access networks (RANs) are often developed using simulation, yet their predictions can shift when exposed to physical propagation conditions, protocol behavior, and equipment implementations. This paper presents a cyber-physical recursive self-improvement (RSI) framework for AI-RAN that connects a task-oriented student model, an experiment-planning coach agent, a calibratable virtual environment, and a physical host agent. Physical measurements are organized as evidence for both model adaptation and subsequent experiment planning. We examine an implementation of the physical-feedback pathway using ping-pong handover as a case study. The system completes 12 physical executions organized into six baseline--candidate pairs, updates the student from measured outcomes, and evaluates the revised model on reserved physical configurations. In the observed short-window tests, increasing the time-to-trigger from 320 to 128
    
[^144]: Splice-Core：面向高能效AI推理加速器的低位比特拼接GEMM单元

    Splice-Core: Low-Order-Bit-Splicing GEMM for Highly Energy-Efficient AI Inference Accelerators

    [https://arxiv.org/abs/2611.25617](https://arxiv.org/abs/2611.25617)

    Splice-Core提出了一种时间一元/二进制混合GEMM加速单元，通过Splice-BP和Splice-LUT两种新型处理单元设计，在INT4 CNN与LLM工作负载上相比二进制脉动阵列实现约77%的能耗降低，并显著提升吞吐量、降低能量延迟积。

    

    Splice-Core是一种时间一元/二进制混合通用矩阵乘法（GEMM）单元，它能够最小化低位累加延迟，同时对高位激活比特保留值依赖的执行方式。我们提出了两种新颖的处理单元（PE）变体：（i）利用位平面的Splice-BP，以及（ii）可在单个周期内选择共享权重倍数的Splice-LUT。与二进制脉动阵列相比，在45nm工艺下跨26个INT4卷积神经网络（CNN）和大语言模型（LLM）工作负载，Splice-BP/Splice-LUT分别实现了77.4%/76.4%的几何平均Tile能耗降低。Splice-LUT还进一步将几何平均Tile吞吐量提高44.4%，并在26种配置中将几何平均能量延迟积（EDP）降低6.13倍。在评估的W8A8卷积和专家投影工作负载中，Splice-BP和Splice-LUT将Tile能耗降低77.3%-88.1%。布局布线后的INT4（测试结果）实现了14%/9%的……（摘要原文在此处截断）

    arXiv:2611.25617v1 Announce Type: cross  Abstract: Splice-Core is a temporal-unary/binary (tub) hybrid general matrix multiplication (GEMM) unit that minimizes low-order accumulation latency, while retaining value-dependent execution for high-order activation bits. We present two novel processing element (PE) variants: (i) Splice-BP utilizing bit planes, and (ii) Splice-LUT, which selects a shared weight multiple in a single cycle. Compared to a binary systolic array, Splice-BP/Splice-LUT achieve 77.4%/76.4% geometric-mean tile energy reductions, respectively, across 26 INT4 convolutional neural network (CNN) and large language model (LLM) workloads in 45nm. Splice-LUT further increases geometric-mean tile throughput by 44.4% and reduces geometric-mean energy-delay product (EDP) by 6.13x across the 26 configurations. Across the evaluated W8A8 convolution and expert-projection workloads, Splice-BP and Splice-LUT reduce tile energy by 77.3-88.1%. Post-place-and-route INT4 yields 14%/9% s
    
[^145]: 不要失去未来：世界动作模型的潜变量感知4比特量化

    Don't Lose the Future: Latent-Aware 4-Bit Quantization of World Action Models

    [https://arxiv.org/abs/2611.25614](https://arxiv.org/abs/2611.25614)

    提出潜变量感知量化框架LAQ，通过读取器度量和读取计数在世界动作模型预测未来的写入、存储和读取环节进行保护，从而在4比特量化下保持预测未来的保真度，实现W4A4KV4的机器人端部署。

    

    世界动作模型预测场景将如何演变，并基于这一预测的未来来决定其动作。将其部署在机器人的车载计算机上需要进行4比特的权重、激活值和键值缓存量化（W4A4KV4）。现有的WAM 4比特量化器只保护生成的动作，并每次仅估计单个去噪步骤的敏感度。然而，这些方法无法保持预测未来的保真度：一旦动作通路得到保护，剩余的动作误差主要通过一个退化的未来进入，而没有任何现有方法刻意保护这一未来。我们提出了LAQ，一个潜变量感知的量化框架，在预测未来被写入、存储和读取之处对其进行保护。具体而言，我们引入了读取器度量，通过预测未来对动作的影响来衡量未来误差；以及读取计数，根据读取每个舍入误差的去噪步骤数量来为其加权。两者共同实现了对激活值精度的合理分配。

    arXiv:2611.25614v1 Announce Type: new  Abstract: World action models (WAMs) predict how a scene will evolve and condition their actions on this predicted future. Deploying them on a robot's onboard computer calls for 4-bit weights, activations and key-value caches (W4A4KV4). Existing 4-bit quantizers for WAMs preserve the generated action, and estimate sensitivity one denoising step at a time. However, they fail to preserve the fidelity of the predicted future: once the action pathway is protected, the remaining action error enters mostly through a degraded future that none of them deliberately protects. We propose LAQ, a latent-aware quantization framework that preserves the predicted future where it is written, stored and read. Specifically, we introduce a reader metric, which measures future error by its effect on the action, and a read count, which weights each rounding error by the number of denoising steps that read it. Together they give the allocation of precision over activati
    
[^146]: LLM能在不训练的情况下修复语言模型吗？

    Can an LLM Fix a Language Model Without Training It?

    [https://arxiv.org/abs/2611.25613](https://arxiv.org/abs/2611.25613)

    研究表明，LLM无需任何梯度训练，仅通过交互式检查模型预测、激活和权重统计并选择数值投影编辑，即可有效修复受损的Transformer语言模型，其表现可与黑盒搜索相当甚至大幅超越（如Pythia上达到65.62%准确率而黑盒搜索为0%）。

    

    尽管外部语言模型可以选择并测试对另一个模型权重的修复，但模型行为也可以通过固定的数值编辑来恢复。我们在一项Transformer修复的对照研究中评估了交互式选择的价值。在检查预测结果、激活摘要和权重统计之后，编辑者（LLM）选择要应用于目标模型的投影配方，而无需任何梯度更新。我们分析了涉及五个掩码语言模型和因果语言模型的20个记录会话，将LLM所选的编辑与固定投影、脚本化及等预算的黑盒搜索以及低秩适配进行了比较。在核心的双层故障下，实时选择在SmolLM2和Qwen2.5上的表现与黑盒搜索相当。对于Pythia，实时选择达到了65.62%的补全准确率，而黑盒搜索为0%，固定投影为66.15%。在另一个同时影响注意力和前馈权重的复合故障下，脚本化和LLM引导的（摘要在此处截断）

    arXiv:2611.25613v1 Announce Type: new  Abstract: Although an external language model can select and test repairs to another model's weights, behavior can also be recovered through fixed numerical edits. We assessed the value of interactive selection in a controlled study of Transformer repair. After inspecting predictions, activation summaries, and weight statistics, the editor selects projection recipes to apply to the target without gradient updates. We analyzed 20 recorded sessions involving five masked and causal language models, comparing the selected edits with fixed projections, scripted and matched-budget black-box search, and low-rank adaptation. Under the core two-layer fault, live selection matches black-box search on SmolLM2 and Qwen2.5. For Pythia, live selection achieves 65.62% completion accuracy, compared with 0% for black-box search and 66.15% for a fixed projection. Under a separate compound fault affecting attention and feed-forward weights, both scripted and LLM-gui
    
[^147]: 副驾驶还是罪魁祸首？AI如何改变开源软件的缺陷报告与维护

    Copilot or Culprit? How AI Is Changing Open-Source Bug Reporting and Maintenance

    [https://arxiv.org/abs/2611.25605](https://arxiv.org/abs/2611.25605)

    该研究首次对101个开源项目中超过12万份缺陷报告进行大规模分析，发现AI参与的缺陷报告正变得越来越完整可靠，其中AI用于复现和诊断与更高的报告有效性相关。

    

    AI生成的缺陷报告曾因不可靠的陈述和错误的复现步骤给维护者带来负担而饱受批评。然而，随着大语言模型（LLM）能力的提升，一些维护者开始对AI在缺陷报告与维护中的应用持更加积极的看法。理解这一演变需要考察：AI参与的报告是否正变得更加可靠，AI的哪些用途与有效的发现相关联，以及维护者如何调查这些报告并据此采取行动。为此，我们开展了首个大规模研究，分析了2024年1月至2026年9月期间来自101个开源项目的120,918份缺陷报告，追踪AI从发现、报告到验证和解决全过程中的参与情况。我们的研究结果表明，AI参与的报告随时间推移变得更加完整，被确认为有效的报告比例不断上升。在缺陷报告环节，AI用于复现和诊断与更高的报告有效性相关，而用于撰写报告……（摘要原文在此处截断）

    arXiv:2611.25605v1 Announce Type: cross  Abstract: AI-generated bug reports have drawn criticism for unreliable claims and faulty reproducers that burden maintainers. However, as LLM capabilities improve, some maintainers have begun to express more positive views on AI use in bug reporting and maintenance. Understanding this evolution requires examining whether AI-involved reports are becoming more reliable, which uses of AI are associated with valid findings, and how maintainers investigate these reports and act on them. To this end, we conduct the first large-scale study of 120,918 bug reports from 101 open-source projects between January 2024 and September 2026, tracing AI involvement from discovery and reporting through validation and resolution. Our findings show that AI-involved reports become more complete over time, with a growing share of reports confirmed as valid. In bug reporting, AI for reproduction and diagnosis are associated with higher validity, whereas report writing 
    
[^148]: 论GNN结构稳定性界的信息量：GIN与GAT的对比

    On the Informativeness of Structural Stability Bounds for GNNs: GINs versus GATs

    [https://arxiv.org/abs/2611.25603](https://arxiv.org/abs/2611.25603)

    本文推导了GIN和单头GATv1在确定性边扰动下的结构稳定性界，并与扰动无关的直径界对比，精确刻画了GIN稳定性界更紧的条件（在Cora上成立但在低度数的MUTAG上失效），同时发现GATv1的稳定性界从不更紧。

    

    图神经网络的稳定性界限制了当图发生变化时节点表示可能产生的漂移程度，但人们很少将其与最简单的替代方案进行比较：即不依赖于图结构、对任意两个输出之间距离给出的界。我们在确定性边扰动（不必很小）下，推导了图同构网络（GIN）和单头GATv1网络的结构稳定性界，并将它们分别与这种与扰动无关的直径界进行比较。对于GIN，我们通过一个将扰动大小、网络深度和最大度联系起来的条件，精确刻画了稳定性界何时更紧；该条件在Cora数据集上以很大余量成立，但在低度数的MUTAG图上超过几层后便不再成立。对于GATv1，稳定性界从不更紧，这表明需要开展基于注意力质量的分析。

    arXiv:2611.25603v1 Announce Type: cross  Abstract: Stability bounds for graph neural networks limit how much node representations can drift when the graph changes, but they are rarely compared with the simplest alternative: a bound on the distance between any two outputs, whatever the graph. We derive structural stability bounds for Graph Isomorphism Networks (GINs) and single-head GATv1 networks under deterministic edge perturbations that need not be small, and compare each with such a perturbation-independent diameter bound. For GINs, we characterize exactly when the stability bound is tighter, through a condition linking perturbation size, depth and maximum degree; it holds with a wide margin on Cora, but fails beyond a few layers on the low-degree MUTAG graphs. For GATv1, the bound is never tighter, which calls for analyses based on attention masses.
    
[^149]: Tether：用于智能体系统级监控的经过验证的有状态策略引擎

    Tether: Verified Stateful Policy Engine for System-Level Monitoring of Agents

    [https://arxiv.org/abs/2611.25599](https://arxiv.org/abs/2611.25599)

    Tether提出了一种经过验证的有状态安全策略引擎，通过在短暂隔离的沙箱中执行智能体操作、从系统调用推断抽象语义，实现了对AI智能体系统级副作用的运行时安全策略强制执行，克服了现有LLM监控器缺乏安全保证和无状态策略层无法捕捉多步副作用的缺陷。

    

    自主AI智能体正越来越多地被部署用于执行任意shell命令或程序的长时间运行任务。然而，这些操作可能导致未经授权的网络请求，或对用户的文件系统、进程或权限造成持续的恶意更改。现有的防御机制要么依赖基于LLM的监控器（缺乏可证明的安全保证），要么依赖无状态的策略执行层（无法捕捉多步数据流以及策略语言未建模的执行副作用，例如在Python脚本内部发起的网络调用）。我们提出了Tether，一个用于在运行时对智能体行为所观察到的系统副作用执行有状态安全策略的框架。Tether在一个短暂的、隔离的沙箱中执行每个提议的操作，并从观察到的系统调用中推断抽象的操作语义。随后，Tether对这些推断出的语义执行有状态策略，只有当操作满足策略要求时才允许其继续执行。

    arXiv:2611.25599v1 Announce Type: cross  Abstract: Autonomous AI agents are increasingly deployed for long-running tasks that execute arbitrary shell commands or programs. However, these actions can result in unauthorized network requests or persistent, malicious changes to the user's file system, processes, or privileges. Existing defense mechanisms rely on either LLM-based monitors, which lack provable safety guarantees, or stateless policy enforcement layers, which miss multi-step data flows and execution side-effects unmodeled by the policy language (e.g., network calls initiated inside a Python script). We introduce Tether, a framework for runtime enforcement of stateful security policies over the observed side-effects of an agent's actions. Tether executes each proposed action in an ephemeral, isolated sandbox, inferring abstract action semantics from observed system calls. Tether then enforces stateful policies over these inferred semantics, permitting actions to proceed only if
    
[^150]: 面向决策的强化学习用于大语言模型后训练

    Decision-Focused Reinforcement Learning for LLM Post-Training

    [https://arxiv.org/abs/2611.25597](https://arxiv.org/abs/2611.25597)

    该论文提出面向决策的强化学习（DFRL）框架，以下游决策负遗憾作为奖励对LLM进行后训练，使其预测直接最小化下游决策成本且无需下游问题凸性或可微性，并通过启发式热启动和奖励校正两种方法解决了奖励评估开销大的问题。

    

    大语言模型（LLM）作为各类任务中的预测器展现出了巨大潜力，这得益于它们利用非结构化上下文信息的能力。然而，当预测结果作为下游优化问题的参数时，更高的预测精度并不一定转化为更低的决策成本。我们提出了面向决策的强化学习（DFRL），这是一个用于对LLM进行后训练以预测优化参数的框架，它采用下游决策的负遗憾作为奖励。DFRL训练LLM预测器以最小化由其预测所导致的决策成本，无需下游问题具有凸性或可微性，并且能够轻松集成到现有的强化学习后训练框架中。为了解决奖励评估开销高昂的问题，我们开发了两种方法。启发式热启动在初始阶段使用廉价的近似奖励，随后切换为使用精确奖励进行改进。奖励校正……（原文截断）

    arXiv:2611.25597v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown promise as predictors across various tasks, benefiting from their ability to use information from unstructured context. However, when predictions serve as parameters of downstream optimization problems, higher prediction accuracy does not necessarily translate into lower decision costs. We propose decision-focused reinforcement learning (DFRL), a framework for post-training LLMs to predict optimization parameters using negative downstream decision regret as reward. DFRL trains LLM predictors to minimize the costs of the decisions induced by their predictions, without requiring downstream convexity or differentiability, and integrates readily into existing RL post-training frameworks. To address expensive reward evaluation, we develop two approaches. Heuristic warm-starting uses inexpensive approximate rewards during an initial phase before switching to refinement with exact rewards. Reward correc
    
[^151]: Bradley-Terry训练可能扭曲原本准确的语言世界模型的结果概率

    Bradley-Terry Training Can Distort Outcome Probabilities of Initially Accurate Language World Models

    [https://arxiv.org/abs/2611.25594](https://arxiv.org/abs/2611.25594)

    本文证明Bradley-Terry目标函数在基于比较反馈的训练中无法教会语言世界模型真实的结果概率，反而会将原本准确的模型（危害误差仅2.4个百分点）中33-43个百分点的概率扭曲到实际不可能发生的结果上，且这种失真源于目标函数本身而非比较数据。

    

    arXiv:2611.25594v1 公告类型：交叉 摘要：比较反馈（即由评判者指出两个候选结果中哪一个更接近实际发生的情况）是训练为动作结果分配概率的语言世界模型的一种实用方法。我们证明，使用Bradley-Terry目标函数进行拟合时，它未必能教会模型真实的结果概率，甚至可能扭曲原本准确的结果概率。在一个具有枚举结果分布的模拟机器人推动任务中，该目标函数会使一个在未见布局上表现准确的模型（危害误差为2.4个百分点）将33至43个百分点的概率转移到该条件下没有任何枚举分支会产生的结果上。问题的根源在于目标函数本身，而非比较数据：在训练前计算的目标函数最优解可以预测模型最终的去向，而一个了解评判者规则的似然函数能够从相同的比较数据中学习且不产生这种偏移。这种偏移具有鲁棒性：它在未见条件、五种候选法则以及三倍更密集的枚举下均会重现。

    arXiv:2611.25594v1 Announce Type: cross  Abstract: Comparative feedback, in which a judge says which of two candidate outcomes is closer to what happened, is a practical way to train language world models that assign probabilities to an action's outcomes. We show that, fitted with a Bradley-Terry objective, it need not teach the true outcome probabilities and can distort accurate ones. On a simulated robot push with enumerated outcome distributions, this objective takes a model that is accurate on unseen layouts (hazard error 2.4 percentage points, pp) and moves 33 to 43 pp of its probability onto outcomes that none of a condition's enumerated branches produces. The cause is the objective, not the comparisons: its optimum, computed before training, predicts where the model ends up, and a likelihood that knows the judge's rule learns from the same comparisons without the shift. The shift is robust: it recurs on unseen conditions, under five candidate laws and a three-fold denser enumera
    
[^152]: 测量大语言模型在网络威胁情报中的幻觉现象：一项探索性研究

    Measuring Hallucination in Large Language Models for Cyber Threat Intelligence: An Exploratory Study

    [https://arxiv.org/abs/2611.25590](https://arxiv.org/abs/2611.25590)

    该论文提出了可复现的HalluVision框架，利用微调的命名实体识别模型和多种相似度指标，系统测量了商业及开源大语言模型在网络威胁情报任务中的幻觉现象，并发现语义相似度指标与人类判断最为一致。

    

    大语言模型（LLM）越来越多地被应用于网络威胁情报（CTI）任务，例如文本摘要和问答。然而，它们可能产生幻觉，即并非基于源文本的错误或误导性内容，这可能会误导分析师并损害下游的安全决策。本文对基于LLM的网络威胁情报任务中的幻觉现象进行了一项探索性测量研究。我们提出了HalluVision，这是一个可复现的框架，用于生成LLM输出、通过微调的命名实体识别模型提取领域特定实体，并使用多种相似度指标评估事实一致性。基于4,940篇真实世界的网络安全文章，我们测量了两个商业LLM和一个开放权重LLM输出中的幻觉现象。测量结果通过100个人工标注的输出和两个成熟的忠实度检测器进行了验证。结果显示，语义相似度与人类判断相一致，而……

    arXiv:2611.25590v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used in cyber threat intelligence (CTI) tasks, such as summarization and question answering. However, they may produce hallucinations, false or misleading content not grounded in the source text, which can mislead analysts and corrupt downstream security decisions. This paper presents an exploratory measurement study of hallucinations in LLM-based CTI. We introduce HalluVision, a reproducible framework that generates LLM outputs, extracts domain-specific entities using a fine-tuned named entity recognition model, and evaluates factual consistency using multiple similarity metrics. Using 4,940 real-world cybersecurity articles, we measure hallucinations in outputs from two commercial LLMs and one open-weight LLM. Measurements are validated against 100 human-annotated outputs and two established faithfulness detectors. Results show that semantic similarity aligns with human judgments, whereas e
    
[^153]: 基于神经ODE图结构推断的网络化动力系统学习

    Learning Networked Dynamical Systems via Neural ODE-based Graph Structure Inference

    [https://arxiv.org/abs/2611.25586](https://arxiv.org/abs/2611.25586)

    该论文提出NOGI框架，通过将确定性图学习器与基于MPNN的神经ODE相结合并采用伴随方法端到端训练，实现了连续时间下的潜在交互图推断，在稀疏和不规则观测下显著提升了复杂动力系统的结构恢复性能。

    

    从多变量时间序列中推断潜在交互图，对于理解神经科学、生物学和物理学中的复杂动力系统至关重要。尽管神经关系推断近来取得了进展，但大多数方法依赖于离散时间建模，这在稀疏和不规则观测下性能会下降，并且容易将多跳依赖纠缠成虚假的直接边。我们证明了连续时间公式能够自然地将推断与任何固定的观测时间表解耦，使梯度信号能够在观测间隔之间连续积分，从而同时服务于动力学学习和结构学习。我们提出了NOGI（Neural ODE-based Graph Inference，基于神经ODE的图推断）框架，该框架将确定性图学习器与基于MPNN的神经ODE相结合，两个模块通过伴随方法进行端到端训练。在两种规模下的四个模拟动力系统中，NOGI在32个评估案例中的30个中实现了最佳的结构恢复效果。

    arXiv:2611.25586v1 Announce Type: new  Abstract: Inferring the latent interaction graph from multivariate time series is fundamental to understanding complex dynamical systems in neuroscience, biology, and physics. Despite recent progress in neural relational inference, most methods rely on discrete-time modeling, which degrades under sparse and irregular observations and tends to entangle multi-hop dependencies into spurious direct edges. We show that a continuous-time formulation naturally decouples inference from any fixed observation schedule, allowing gradient signals to be integrated continuously between observations for both dynamics and structure learning. We propose NOGI (Neural ODE-based Graph Inference), a framework that couples a deterministic graph learner with an MPNN-based Neural ODE. The two modules are trained end-to-end via the adjoint method. Across four simulated dynamical systems at two scales, NOGI achieves the best structure recovery in 30 out of 32 evaluated cas
    
[^154]: 一种可解释且可扩展的分层张量框架：用于整合多组学与数字病理组学数据

    An Explainable and Scalable Hierarchical Tensor Framework for Integrating Multi-Omics and Digital Pathomics

    [https://arxiv.org/abs/2611.25584](https://arxiv.org/abs/2611.25584)

    该论文提出HIT-Fuse分层可解释张量融合框架，整合六种多组学与病理组学模态，通过模态特异性自编码器、两阶段CP张量分解、缺失模态重建和数学回溯机制，在精准肿瘤学中实现了可解释且可扩展的多模态数据融合。

    

    背景：精准肿瘤学需要整合异质性的分子与病理数据，然而许多融合模型提供的生物学可解释性有限，且难以应对模态缺失的问题。方法：我们提出分层可解释张量融合框架（HIT-Fuse），整合六种模态：体细胞拷贝数变异、DNA甲基化、RNA测序、microRNA表达、RPPA蛋白丰度以及基于视觉Transformer提取的病理组学特征，涵盖四个TCGA队列（胶质瘤、肺腺癌、乳腺癌和肾癌）。模态特异性自编码器压缩高维输入，随后进行两阶段CP张量分解。缺失的潜在模态通过临床协变量和已观测的组学数据进行重建，以扩充队列规模并重新训练下游模型。数学回溯机制实现全局和局部的可解释人工智能（xAI）归因。结果：HIT-Fuse识别出具有显著（摘要在此处截断）……

    arXiv:2611.25584v1 Announce Type: cross  Abstract: Background: Precision oncology requires integration of heterogeneous molecular and pathology data, yet many fusion models offer limited biologically grounded interpretability and struggle with missing modalities.   Methods: We present Hierarchical Interpretable Tensor Fusion (HIT-Fuse), integrating six modalities: somatic copy number variation, DNA methylation, RNA sequencing, microRNA expression, RPPA protein abundance, and vision transformer-derived pathomics across four TCGA cohorts (glioma, lung adenocarcinoma, breast, and kidney cancer). Modality-specific autoencoders compress high-dimensional inputs, followed by two-stage CP tensor decomposition. Missing latent modalities are reconstructed using clinical covariates and observed omics to expand cohorts and retrain downstream models. A mathematical traceback mechanism enables global and local xAI attributions.   Results: HIT-Fuse identifies distinct latent subgroups with significan
    
[^155]: DriftGuard：基于因果归因的大语言模型智能体在线记忆漂移检测

    DriftGuard: Online Memory-Drift Detection via Causal Attribution for LLM Agents

    [https://arxiv.org/abs/2611.25583](https://arxiv.org/abs/2611.25583)

    提出DriftGuard两阶段框架，通过解耦离线因果监督与在线归因，在服务时高效检测LLM智能体中可能导致行为漂移的检索记忆，解决了内容特征无法进行因果归因以及反事实归因成本过高这两大挑战。

    

    大语言模型（LLM）智能体依赖持久化记忆来规划工具调用并完成多步骤任务，这也使其面临记忆漂移的风险：一条对抗性或误导性的记忆会改变执行路径和最终结果。在本工作中，我们研究在线记忆漂移检测，即在服务时估计每条检索到的记忆对潜在漂移的因果贡献。我们识别出两个根本性挑战：（1）内容层面的特征不足以进行因果记忆漂移归因，因为仅凭困惑度或检索相关性无法确定哪条检索到的记忆在被中和后会改变智能体的行为；（2）反事实归因对于在线检测而言代价过高，因为它需要对每条候选记忆重放下游智能体和工具的执行过程，超出了实时服务的预算。为了应对这些挑战，我们提出了DriftGuard，一个将离线因果监督（与在线检测）解耦的两阶段框架……

    arXiv:2611.25583v1 Announce Type: new  Abstract: Large language model (LLM) agents rely on persistent memory to plan tool calls and complete multi-step tasks, which exposes them to memory drift: an adversarial or misleading memory alters the execution path and final outcome. In this work, we investigate online memory-drift detection, which estimates each retrieved memory's causal contribution to potential drift at serving time. We identify two fundamental challenges. (1) Content-level features are insufficient for causal memory-drift attribution, as perplexity or retrieval relevance alone cannot determine which retrieved memory would change agent behavior if neutralized; (2) Counterfactual attribution is prohibitively expensive for online detection, as it requires replaying downstream agent and tool execution for each candidate memory, exceeding real-time serving budgets. To address these challenges, we propose DriftGuard, a two-stage framework that decouples offline causal supervision
    
[^156]: AutoRAGBench：用于自动化RAG系统开发的LLM智能体基准测试

    AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development

    [https://arxiv.org/abs/2611.25582](https://arxiv.org/abs/2611.25582)

    本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。

    

    检索增强生成（RAG）是一种成熟的范式，通过从外部语料库检索的信息来增强大语言模型（LLM）。然而，开发有效的RAG系统需要大量的工程工作和迭代实验，因为有效的设计选择因任务和领域而异。随着LLM智能体能力的快速进步，我们提出这样一个问题：这些智能体能够多好地自主构建RAG系统，以及它们能够通过迭代多有效地改进这些系统。在本文中，我们提出了AutoRAGBench，这是第一个评估LLM智能体在开放代码空间中从零开始开发RAG系统能力的基准测试。它包含九个RAG系统开发任务。对于每个任务，智能体在没有起始代码或预设策略的情况下，在单个H800 GPU上六小时的预算内从零构建RAG系统。它们分析任务数据、实现RAG流水线，并基于实验结果迭代地优化其设计……

    arXiv:2611.25582v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experi
    
[^157]: FlashX：基于CBA型NAND闪存的具身智能体机器人经验复用扩展

    FlashX: Scaling Robotic Experience Reuse for Embodied Agents with CBA-Based NAND Flash

    [https://arxiv.org/abs/2611.25580](https://arxiv.org/abs/2611.25580)

    该论文提出FlashX系统，利用CMOS直接键合的CBA型NAND闪存来扩展机器人经验库的存储规模，通过检索成功经验对扩散模型去噪进行热启动，从而在边缘设备上同时提升具身智能体的任务准确性与推理效率。

    

    具身智能体正成为解决复杂机器人任务的一种有前景的范式，它利用视觉-语言-动作模型（VLA）和世界动作模型（WAM），通过基于扩散Transformer（DiT）的迭代去噪来生成动作。然而，现有模型往往难以泛化到分布外（OOD）环境。我们观察到，检索成功经验并将其用于热启动去噪过程，能够使智能体更好地适应陌生情况，同时降低计算开销。与传统的以牺牲精度换取计算加速的技术不同，这创造了一个能够同时提升任务准确性和推理效率的新机会。然而，要实现这一机会就需要扩展经验库的规模，这会带来巨大的存储和检索开销，对边缘系统的容量、带宽和功耗约束构成挑战。我们提出了FlashX，一种CMOS直接键合的（摘要在此处截断）

    arXiv:2611.25580v1 Announce Type: cross  Abstract: Embodied agents are emerging as a promising paradigm for complex robotic tasks, leveraging Vision-Language-Action (VLA) and World Action Models (WAMs) to generate actions through Diffusion Transformer (DiT)-based iterative denoising. However, existing models often struggle to generalize to out-of-distribution (OOD) environments. We observe that retrieving successful experiences and using them to warm-start denoising enables agents to better adapt to unfamiliar situations while reducing computational overhead. This creates a new opportunity to improve task accuracy and inference efficiency simultaneously, in contrast to conventional acceleration techniques that trade accuracy for computation. However, realizing this opportunity requires scaling experience banks, introducing substantial storage and search overhead that challenges the capacity, bandwidth, and power constraints of edge systems.   We propose FlashX, a CMOS Directly Bonded t
    
[^158]: AeroBelief：面向证据引导航空感知的预算化首动作验证

    AeroBelief: Budgeted First-Action Verification for Evidence-Guided Aerial Sensing

    [https://arxiv.org/abs/2611.25568](https://arxiv.org/abs/2611.25568)

    AeroBelief提出了一种预算受限的首动作验证框架，通过按首动作分组候选视图、低成本提议器扫描和选择性查询结构化世界模型，在参考-候选评估次数比穷举评分减少91.81%的情况下，将任务完成率提升至75.42%，超越了穷举式评分和视线几何方法。

    

    航空感知需要在预算化预测、移动和图像处理的同时，将不确定的俯视上下文与当前观测相结合。AeroBelief将下一个可执行动作作为对照参考分数进行验证的基本单元。它按首动作对候选视图进行分组，使用低成本的提议器进行扫描，并有选择地查询结构化世界模型（WM）。经校准的分组边界支持严格的承诺机制；六次查询控制器将选择性预测与信念更新、接受检查和计划感知相连接。在四种布局族、40个世界和960次闭环执行的实验中，AeroBelief控制器完成了75.42%的任务，相比之下穷举式WM评分为73.33%，视线几何方法为65.83%，同时其参考-候选评估次数比穷举评分减少了91.81%。在使用六次查询上限的固定参考回放中，基于提议器缩放的锚定边界提高了首动作证书的可用性……

    arXiv:2611.25568v1 Announce Type: new  Abstract: Aerial sensing must combine uncertain overhead context with current observations while budgeting prediction, movement and image processing. AeroBelief uses the next executable action as the unit of verification against reference scores. It groups candidate views by first action, scans them with an inexpensive proposer, and selectively queries a structured world model (WM). Calibrated group bounds support strict commitment; a six-query controller connects selective prediction with belief updates, acceptance checks and scheduled sensing. Across four layout families, 40 worlds and 960 closed-loop executions, the AeroBelief controller completes 75.42% of tasks, compared with 73.33% for exhaustive WM scoring and 65.83% for line-of-sight geometry, using 91.81% fewer reference-candidate evaluations than exhaustive scoring. In fixed-reference replay with a six-query cap, proposer-scaled anchored bounds raised first-action certificate availabilit
    
[^159]: 指令条件化程序性视频中的在线错误检测

    Online Mistake Detection in Instruction-Conditioned Procedural Video

    [https://arxiv.org/abs/2611.25567](https://arxiv.org/abs/2611.25567)

    本文研究了自我中心视频中在线的、基于指令条件化的错误检测，模型仅利用已观测视频、任务指令和先前的预测，实时识别并定位已完成的动作尝试并将其分类为正确、修改、失误、纠正或添加，同时比较了模块化流水线与联合模型两种方案。

    

    一个监控他人执行程序的智能助手，如果能够在还有时间挽回错误时就检测到错误，将是最有用的。这要求在视频播放过程中，根据任务指令来判断动作，识别所尝试的步骤、对其进行时间定位并评估其执行情况。现有的在线检测器不使用指令文本，而基于指令条件化的方法则是离线运行的。在本文中，我们研究了自我中心（第一视角）视频中的在线、指令条件化错误检测。仅使用已观测的视频、任务指令和先前的预测，我们的模型能够识别并定位已完成的动作尝试，并将其分类为正确、修改、失误、纠正或添加；被遗漏的步骤则在视频结束后确定。我们将具有独立步骤识别、时间定位和执行验证功能的模块化流水线，与在单一步骤中预测所有输出的联合模型进行了比较。为了支持（此处的摘要内容被截断）

    arXiv:2611.25567v1 Announce Type: cross  Abstract: An assistant monitoring a person performing a procedure is most useful when it can detect mistakes while there is still time to recover. This requires judging actions against task instructions as the video unfolds, identifying the attempted step, locating it in time, and assessing its execution. Existing online detectors do not use the instruction text, while instruction-conditioned methods operate offline. In this paper, we study online, instruction-conditioned mistake detection in egocentric videos. Using only observed video, task instructions, and prior predictions, our model identifies and localizes completed action attempts and classifies them as correct, modified, slip, corrections, or additions; omitted steps are determined after the video ends. We compare a modular pipeline, with separate step identification, temporal localization, and execution verification, to a joint model that predicts all outputs in a single step. To suppo
    
[^160]: 诊断不是策略：能力条件化测试时推理的状态转换审计

    Diagnosis is not a policy: a state-transition audit of caoability-conditioned test-time inference

    [https://arxiv.org/abs/2611.25566](https://arxiv.org/abs/2611.25566)

    该论文提出TERRA审计协议，将测试时干预结果分解为适用性、状态实现、端点转换、危害和成本五个维度，并通过可执行金融推理实验证明诊断信息本身并不构成有效策略——诊断余量与局部一致性均不能保证实际的性能提升。

    

    测试时动作可以在不纠正答案的情况下满足检索、状态实现、程序一致性或渲染等目标。测试时错误响应与可实现性审计（TERRA）是一种评估/审计协议，而非新的推理算法。它将干预结果统一分解为适用性、目标状态实现、端点转换、危害以及资源/推理成本五个维度。配对比较固定传入状态和评估条件；跨模型分析则改变接收配置。可执行金融推理实验得出三个发现：相同的上下文扩展可能损害一种配置却有益于另一种配置；充足的证据并不能确保正确性，因为不同配置在证据到答案的转换能力上存在差异；诊断余量——即基于理想化诊断信息可达到的、相对于实测基线的准确率提升空间——加上局部一致性并不能确保实际……（原文摘要在此处截断）

    arXiv:2611.25566v1 Announce Type: new  Abstract: Test-time actions can satisfy retrieval, state-realization, program-consistency, or rendering objectives without correcting an answer. Test-time Error Response and Realizability Audit (TERRA) is an evaluation/auditing protocol, not a new inference algorithm. It uses a unified decomposition of intervention outcomes into applicability, target-state realization, endpoint translation, harm, and resource/inference cost. Paired comparisons fix the incoming state and evaluation conditions; cross-model analyses vary the receiving configuration. Executable financial reasoning experiments yield three findings: the same context expansion can harm one configuration and benefit another; sufficient evidence does not ensure correctness because configurations differ in evidence-to-answer translation; and diagnostic headroom---accuracy attainable with idealized diagnostic information over the measured baseline---plus local consistency does not ensure pra
    
[^161]: 覆盖-选择差距：限制LLM评分标准构建的是排序而非生成

    The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction

    [https://arxiv.org/abs/2611.25557](https://arxiv.org/abs/2611.25557)

    该论文发现在自动评分标准构建中，限制性能的瓶颈是对候选标准的排序选择而非标准生成，并提出“覆盖-选择差距”指标来量化无参考排序与有参考贪婪诊断之间的性能差异。

    

    任务特定的评分标准正被越来越多地用于评估开放式的语言模型输出，然而从复杂的源文档中构建全面的评估标准仍然具有挑战性。我们研究了专业并购尽调场景中的自动评分标准构建问题，在该场景下，评估标准必须从任务指令和支持文档中推断得出，而无法获取专家撰写的参考评分标准。我们引入了一个多轴生成框架，该框架从互补的结构化和自由形式视角探索源证据，以构建一个广泛的候选评估标准池。为了区分评估标准发现的失败与候选选择的失败，我们提出了“覆盖-选择差距”这一概念，用于衡量在固定选择预算下，无参考排序方法与基于参考的贪婪诊断方法之间的性能差异。该方法在16个任务上开发，并在4个留出任务上进行评估。

    arXiv:2611.25557v1 Announce Type: cross  Abstract: Task-specific rubrics are increasingly used to evaluate open-ended language-model outputs, yet constructing comprehensive evaluation criteria from complex source documents remains challenging. We study automatic rubric construction in professional mergers and acquisitions (M&A) due diligence, where evaluation criteria must be inferred from task instructions and supporting documents without access to an expert-authored reference rubric. We introduce a multi-axis generation framework that explores complementary structural and free-form views of the source evidence to construct a broad pool of candidate evaluation criteria. To distinguish failures of criterion discovery from failures of candidate selection, we introduce the coverage-selection gap, which measures the difference between reference-free ranking and a reference-informed greedy diagnostic under a fixed selection budget. Developed on 16 tasks and evaluated on four held-out tasks
    
[^162]: 用于审计3D世界的高效多模态智能体

    Efficient Multimodal Agents for Auditing 3D Worlds

    [https://arxiv.org/abs/2611.25543](https://arxiv.org/abs/2611.25543)

    该论文通过端到端微调4B小型视觉语言模型并让多个小型探索者协同工作，在导航和触发3D世界故障任务上达到前沿大模型水平，从而实现准确且廉价的自动化世界审计。

    

    多模态智能体现在能够构建交互式3D世界和游戏，然而这些世界与人类构建的世界一样，常常包含诸如悬浮物体、可穿墙、或无人注视时发生变化的物体等故障。自动审计器可以大规模检查这些世界，并为构建这些世界的智能体提供奖励，但前提是它必须既准确又廉价。审计需要三种能力，即导航到可能包含故障的物体、触发仅在特定行为之后才会出现的故障、以及识别问题所在。当前的审计器要么准确但缓慢且昂贵，要么廉价但不可靠。我们研究了小型模型能够承担多少审计工作。一个4B参数的视觉语言模型在由贝叶斯搜索引导的前沿教师模型生成的经过验证的轨迹上进行端到端微调后，学会了导航和触发操作，六个这样的探索者联合起来在这两方面均能匹敌前沿模型，但在识别方面……

    arXiv:2611.25543v1 Announce Type: new  Abstract: Multimodal agents can now build interactive 3D worlds and games, yet these worlds, like those built by people, often contain glitches such as floating objects, walls that can be walked through, or objects that change when no one is looking. An automatic auditor could check such worlds at scale and provide rewards for agents that build them, but only if it is both accurate and cheap. Auditing requires three abilities, namely navigating to the objects that may contain a glitch, triggering glitches that appear only after a specific behavior, and recognizing what is wrong. Current auditors are either accurate but slow and expensive or cheap but unreliable. We study how much of the audit small models can take on. A 4B vision language model fine-tuned end to end on verified trajectories of a frontier teacher guided by Bayesian search learns to navigate and trigger, and six such explorers together match a frontier model on both, but recognition
    
[^163]: COVERAGE-TRACE：基于源文本侧反事实追踪的大语言模型摘要遗漏检测方法

    COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries

    [https://arxiv.org/abs/2611.25539](https://arxiv.org/abs/2611.25539)

    提出COVERAGE-TRACE源侧反事实框架，通过替换源文本原子事实并测量摘要似然变化来检测大语言模型摘要中的遗漏，在QMSum数据集上达到0.764–0.825的AUROC，显著优于强基线的0.640。

    

    遗漏难以验证，因为缺失的信息不会留下可供检查的输出片段：一份摘要在事实层面可能是有依据的，却仍然遗漏了保持原文含义所必需的决策、限定条件、数量或条件。我们提出COVERAGE-TRACE，这是一个源文本侧的反事实框架，将遗漏评估为摘要对指定原子源事实的依赖程度。该方法仅用类型匹配的替代内容替换目标事实，同时保持候选摘要不变，并测量其教师强制、长度归一化似然所产生的变化。覆盖目标事实的摘要应比遗漏该事实的摘要产生更强的反应，而跨多个反事实样本的聚合则降低了对替换选择的敏感性。我们在80对匹配的覆盖-遗漏QMSum样本（涵盖多种事实类型）和四个开放权重模型上评估了COVERAGE-TRACE。Multi-CF方法达到了0.764–0.825的AUROC，而强基线仅为0.640。

    arXiv:2611.25539v1 Announce Type: new  Abstract: Omissions are difficult to verify because missing information leaves no output span to inspect: a summary may be factually supported while still excluding a decision, qualification, quantity, or condition needed to preserve the source's meaning. We introduce COVERAGE-TRACE, a source-side counterfactual framework that evaluates omission as dependence on a designated atomic source fact. It replaces only the target fact with type-matched alternatives, holds the candidate summary fixed, and measures the resulting change in its teacher-forced, length-normalized likelihood. A summary covering the target should respond more strongly than one omitting it, while aggregation across multiple counterfactuals reduces sensitivity to replacement choice.   We evaluate COVERAGE-TRACE on 80 matched covered--omitted QMSum pairs spanning several fact types and four open-weight models. Multi-CF achieves 0.764--0.825 AUROC, compared with 0.640 for the stronge
    
[^164]: 迈向自回归的神经网络参数生成

    Towards Autoregressive Neural Network Parameter Generation

    [https://arxiv.org/abs/2611.25538](https://arxiv.org/abs/2611.25538)

    该论文提出指令引导的参数生成方法IGPG，将神经网络参数生成形式化为自回归建模问题，通过把预训练参数编码到离散域，使大语言模型的零样本泛化能力得以迁移到权重生成任务中。

    

    大型语言模型（LLMs）近来在机器学习领域确立了新的范式，展现出卓越的零样本泛化能力和复杂推理能力。与此同时，神经网络参数生成已成为实现快速任务适配与模型合成的有前景的研究方向。然而，现有的权重生成技术难以联合学习多样化模型的分布，难以在单一生成器中编码异构架构，也难以实现分布外泛化。因此，我们提出这样一个问题：能否将预训练参数编码到离散域中，从而将LLMs在文本生成中展现的零样本泛化能力迁移到权重生成任务上。基于这一观点，我们提出了一种受LLM启发的方法——指令引导的参数生成（IGPG），该方法将网络参数生成形式化为一个自回归建模问题。我们首先对预训练权重进行量化……

    arXiv:2611.25538v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have recently established a new paradigm in machine learning, demonstrating exceptional zero-shot generalization and complex reasoning capabilities. Concurrently, neural network parameter generation has emerged as a promising avenue for rapid task adaptation and model synthesis. Existing weight generation techniques, however, struggle to jointly learn the distribution of diverse models, to encode heterogeneous architectures within a single generator, and to generalize out of distribution. We therefore ask whether pretrained parameters can be encoded into a discrete domain so that the zero-shot generalization LLMs exhibit in text generation can be transferred to weight generation. Following this view, we propose an LLM-inspired approach, Instruction-Guided Parameter Generation (IGPG), that formulates network parameter generation as an autoregressive modeling problem. We first quantize pretrained weights with
    
[^165]: 无反馈自我进化如何重塑智能体技能：一项系统性研究

    How Feedback-Free Self-Evolution Reshapes Agent Skills: A Systematic Study

    [https://arxiv.org/abs/2611.25534](https://arxiv.org/abs/2611.25534)

    该研究首次系统考察了大模型在完全无反馈条件下对专家技能文档的自我进化式重写，发现这一过程会使技能退化而非改进——重写把技能拉向模型自身先验、优先舍弃先验之外的专家知识，导致三十轮后在 SkillsBench 上平均下降 15 分，而补回被舍弃的专家知识可恢复大部分性能损失。

    

    智能体技能是专家记录程序性知识的文档，而模型越来越多地通过在无反馈的情况下重写这些文档来维护它们。这种重写既可能使技能适应使用它的模型，也可能侵蚀使其有用的专家知识，然而技能在这种无反馈自我进化下如何演变尚未得到研究。我们让模型在没有任何任务、运行或评判的情况下一轮又一轮地重写专家技能，并用重写后的技能在基于相同模型构建的智能体上运行。实验表明，技能非但没有提升反而失去价值：我们测试的所有模型在使用这些技能时在所有基准测试上的表现都变差，三十轮后在 SkillsBench 上平均下降 15 分。通过让每个模型仅凭主题自行编写技能来采样其先验，我们发现重写会将技能拉向该模型自身的先验，并首先舍弃先验之外的知识。而将这些知识重新放回即可恢复大部分损失的通过率，因此专家的……

    arXiv:2611.25534v1 Announce Type: new  Abstract: Agent skills are documents in which experts write down procedural knowledge, and models increasingly maintain them by rewriting them without feedback. Such rewriting could adapt a skill to the model that uses it or erode the expert knowledge that makes it useful, yet how skills evolve under this feedback-free self-evolution has not been studied. We let models rewrite expert skills round after round without any task, run or judgment and run agents built on the same models with the rewritten skills. Our experiments show that the skills lose value rather than improve, as every model we test performs worse with them on every benchmark, by 15 points on average on SkillsBench after thirty rounds. Sampling each model's prior with skills it writes from the topic alone, we find that rewriting pulls skills toward this prior and gives up first the knowledge outside it. Putting this knowledge back recovers much of the lost pass rate, so an expert's 
    
[^166]: 基于语言模型的个性化旅行规划器

    LM-based Personalized Travel Planner

    [https://arxiv.org/abs/2611.25531](https://arxiv.org/abs/2611.25531)

    提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。

    

    有效的旅行规划需要处理多个信息源、管理约束条件，并根据用户偏好定制推荐。虽然大语言模型（LLM）带来了动态的、上下文感知的决策能力，但单智能体模型在复杂的多步推理、约束遵循和模块化执行方面往往存在困难。为了克服这些局限性，我们提出了一个层次化多智能体框架，其中一个监督型聊天机器人智能体负责协调专门的智能体，分别用于行程验证、用户查询构建、实时数据检索和基于约束的行程生成。通过集成LLM推理、外部API以及基于确定性多维评分框架的迭代优化机制，我们的方法确保了适应性、高效性和个性化。在著名的TravelPlanner基准上的评估表明，预算约束规划性能提升了10%，在更广泛的约束集规划方面也有所提升……

    arXiv:2611.25531v1 Announce Type: new  Abstract: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based pl
    
[^167]: Keystone：面向扩散模型测试时搜索的关键步骤预览评判

    Keystone: Judging Previews at Key Steps for Test-Time Search in Diffusion Models

    [https://arxiv.org/abs/2611.25530](https://arxiv.org/abs/2611.25530)

    该论文提出无需训练的预览评判器Keystone，通过将提示词分解为关于物体、数量、颜色和位置的类型化问题并由现成多模态模型回答，使扩散模型测试时搜索能在去噪早期步骤就可靠地识别注定失败的候选，避免浪费算力。

    

    文本到图像的扩散模型和流模型在每一个去噪步骤都会产生一个预览，即其对干净图像的估计。当这些模型出现数量错误、遗漏物体或物体位置摆放错误时，测试时搜索就陷入了困境：它的验证器是为最终完成图像构建的奖励模型，因此早期评判会误读模糊的预览，而像best-of-N那样延后评判，则需要为彻底去噪那些注定失败的候选图像付出代价。我们发现问题并不出在图像本身：去噪一开始，结果就已由种子或随机采样器的最初几步所决定，并且已经在早期预览中显现出来；真正失败的是评判者，因为在多种类型的提示词上，奖励模型读取这些预览的能力接近随机水平。我们提出了Keystone，一个无需训练的预览评判器，它将提示词编译成关于物体、数量、颜色和位置的类型化问题，并让现成的多模态决策模型来回答这些问题。以这种方式提问时，该评判器从早期预览中读取数量和位置的能力优于奖励模型。

    arXiv:2611.25530v1 Announce Type: new  Abstract: Text-to-image diffusion and flow models yield a preview, their clean-image estimate, at every denoising step. When they miscount, drop or misplace objects, test-time search is in a bind: its verifiers are reward models built for finished images, so judging early misreads blurry previews, and judging late, as best-of-N does, pays to fully denoise doomed candidates. We find that the image is not the problem: the outcome is settled as denoising starts, by the seed or by a stochastic sampler's first steps, and already shows in early previews; what fails on them is the judge, as reward models read them near chance on several kinds of prompts. We introduce Keystone, a training-free judge of previews that compiles a prompt into typed questions about its objects, counts, colours and positions and has an off-the-shelf multimodal decision model answer them. Asked this way, the judge reads counts and positions from early previews better than reward
    
[^168]: CoachSI：利用语言教练实现机器人策略的自我改进

    CoachSI: Self-Improvement of Robot Policies with a Language Coach

    [https://arxiv.org/abs/2611.25529](https://arxiv.org/abs/2611.25529)

    提出 CoachSI：用冻结的现成视觉-语言模型仅以语言形式为机器人策略“说出下一步”原子指令，再将成功的受指导轨迹事后重标记为原任务指令来微调策略，从而突破策略自身经验分布限制、实现长时程任务的自我改进。

    

    通用机器人策略能够可靠地执行原子技能，但当这些技能被组合成长时程任务时却会失败。仅依靠策略自身经验进行自我改进无法弥合这一差距：策略单独收集的经验停留在其自身分布之内，只会重现它已经能做到的事情。我们证明，策略所缺少的不是某项技能，而是下一步该做什么。我们提出 CoachSI，这是一种自我改进方法：一个冻结的、现成的视觉-语言模型（VLM）以语言方式指导策略——它观察机器人，并按照策略原子指令的格式说出下一步，而从不输出动作。成功的受指导执行轨迹会被事后重新标记为原始任务指令，并用于微调策略。微调后的策略随后即可独立部署。在世界动作模型与视觉-语言-动作模型均已停滞的状态下，仅“说出下一步”即可提升子目标完成度（原文摘要在此处被截断）。

    arXiv:2611.25529v1 Announce Type: cross  Abstract: Generalist robot policies execute atomic skills reliably but fail when the same skills are composed into long-horizon tasks. Self-improvement from the policy's own experience does not close this gap: experience collected by the policy alone stays within its own distribution and reproduces what it can already do. We show that what the policy lacks is not a skill but the next step. We present CoachSI, a self-improvement method in which a frozen, off-the-shelf vision-language model (VLM) coaches the policy in language: it watches the robot and names the next step in the format of the policy's atomic instructions, and never outputs an action. Successful coached rollouts are relabeled in hindsight with the original task instruction and used to fine-tune the policy. The fine-tuned policy is then deployed on its own. On states where a World Action Model and a Vision-Language-Action model have stalled, naming the next step raises subgoal compl
    
[^169]: 面向整页优化的生成式排序策略学习

    Learning Generative Ranking Policies for Whole-Page Optimization

    [https://arxiv.org/abs/2611.25527](https://arxiv.org/abs/2611.25527)

    提出一种局部页面级训练目标，可精确拆分为逐位置的训练信号，使生成式排序策略在整页优化中动态生成更高分的页面，同时克服了现有方法中统一页面级信号和固定模仿目标的双重局限。

    

    推荐页面是作为一个整体被消费的：页面上的物品既相互争夺注意力又彼此互补，因此最优页面并非简单地由各自最优的物品并排放置而成。我们研究整页优化问题，即联合地选择和排列页面上的物品以最大化页面整体的得分。可能的页面数量呈组合式增长，因此好的页面必须通过生成而非枚举的方式获得。一些生成式重排方法对采样页面中的每个选择都使用相同的页面级训练信号；另一些方法则模仿在训练前构建的目标页面，而这些目标页面在生成器不断改进的过程中基本保持固定不变。我们用一个局部页面级目标同时解决了上述两个局限，该目标反复地将当前生成器向得分更高的页面推进。这一目标可以精确地拆分为每个位置一个训练目标，每个目标通过完整页面的得分来比较同一部分页面下的候选物品……（摘要在此处被截断）

    arXiv:2611.25527v1 Announce Type: cross  Abstract: A recommendation page is consumed as a whole: the items on it compete for attention and complement one another, so the best page is not simply the individually best items placed side by side. We study whole-page optimization, which selects and orders the items on a page jointly to maximize the score of the page. The number of possible pages grows combinatorially, so good pages must be generated rather than enumerated. Some generative re-ranking methods use the same page-level training signal for every choice in a sampled page. Others imitate target pages built before training, which remain largely fixed as the generator improves. We address both limitations with one local page-level objective, which repeatedly moves the current generator toward pages with higher score. This objective splits exactly into one training target per position, and each target compares candidates at the same partial page by the scores of the complete pages the
    
[^170]: MATE：面向安全高效移动控制的移动智能体可信执行

    MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control

    [https://arxiv.org/abs/2611.25526](https://arxiv.org/abs/2611.25526)

    MATE是一个无需修改底层模型的执行层系统，通过结构化动作记录、运行时护栏、预定义任务路由和浮动界面，将移动智能体高风险行为拦截率从29.7%提升至97.0%，同时减少最多6倍的LLM调用并将重复任务成功率提升至95.7%。

    

    个人智能手机上的移动智能体可能会影响私密数据、支付、设置以及前台屏幕。我们提出了MATE（移动智能体可信执行），这是一个执行层系统，无需修改底层模型即可控制智能体动作提议与设备实际效果之间的边界。MATE将每个UI原语表示为结构化动作记录，该记录结合了指令、目标应用、屏幕上下文、原语和动作等字段。运行时护栏（RG）将该记录映射为允许、阻止或需用户批准的决策；预定义任务路由（PTR）通过经过验证的路径及执行后的屏幕检查来处理重复性请求；而浮动界面则使执行保持在紧凑的悬浮层中。在MobileSafetyBench上，MATE在使用GPT-4o时将高风险行为拦截率从29.7%提升至97.0%。在AndroidWorld上，它将LLM调用次数最多减少6倍，并将重复使用任务的成功率提升至95.7%。

    arXiv:2611.25526v1 Announce Type: cross  Abstract: Mobile agents on personal smartphones can affect private data, payments, settings, and the foreground screen. We present MATE (Mobile Agent Trustworthy Execution), an execution-layer system that controls the boundary between agent action proposals and device effects without modifying the underlying model. MATE represents each UI primitive as a structured action record that combines the instruction, target app, screen context, primitive, and action fields. Runtime Guardrail (RG) maps this record to an allow, block, or user-approval decision; Predefined Task Routing (PTR) serves recurring requests through verified routes with post-execution screen checks; and a floating interface keeps execution in a compact overlay. On MobileSafetyBench, MATE raises high-risk blocking from 29.7% to 97.0% under GPT-4o. On AndroidWorld, it reduces LLM calls by up to 6x and raises recurring-use task success to 95.7%.
    
[^171]: 人格透镜：是什么让视觉语言模型透过其人格的眼睛看世界？

    Persona Lenses: What Makes a Vision-Language Model See Through Its Persona's Eyes?

    [https://arxiv.org/abs/2611.25508](https://arxiv.org/abs/2611.25508)

    提出文本、像素、权重三种“人格透镜”方法，将人格设定插入视觉语言模型与网页之间，显著增强模拟注视行为对人格差异的响应，并系统比较了各透镜与模型规模的效果差异。

    

    视觉语言模型已被用于模拟用户在网页上的视线落点。然而，提示词中设定的人格只能部分地传递到模拟出的注视行为中。我们将这种传递程度量化为同一模型在两个仅在单一属性上存在差异的MatrAIx人格之间的注视差异。我们提出了“人格透镜”，它将人格置于模型与页面之间：文本透镜（一种两步提示方法，模型先写出该人格会注视什么）、像素透镜（在与人格兴趣相符的区域上进行标记）以及权重透镜（一种带门控机制的驻留奖励，以保持扫视路径的自然性）。我们在生成页面和真实页面上使用Qwen3-VL（4B和32B）与Qwen3.5-9B进行实验，将各透镜相互比较，并与模型规模（不带透镜的32B模型）进行比较。结果证实，文本透镜能强烈地区分不同人格，但对不应改变注视方向的画像变化不够稳健；像素透镜区分人格的程度较小，但不受此类变化的影响。

    arXiv:2611.25508v1 Announce Type: cross  Abstract: Vision-language models have been used to simulate where a user looks on a web page. However, a persona in the prompt reaches the simulated gaze only partly. We measure this reach as the difference in one model's gaze between two MatrAIx personas that differ in a single attribute. We introduce persona lenses, which place the persona between the model and the page: a text lens (a two-step prompt in which the model first writes what the persona would look at), a pixel lens (marks on the region matching the persona's interests) and a weight lens (a dwell reward gated to keep the scanpath natural). Using Qwen3-VL (4B and 32B) and Qwen3.5-9B on generated and live pages, we compare the lenses with one another and with scale, the 32B model with no lens. It has been confirmed that the text lens separates the personas strongly but is not robust to profile changes that should not move gaze. The pixel lens separates the personas less but is unaffe
    
[^172]: 基于差异化Token表示的真实场景语音-声景生成联合语言建模

    Joint Language Modeling for In-the-Wild Speech-Scene Generation with Distinct Token Representations

    [https://arxiv.org/abs/2611.25503](https://arxiv.org/abs/2611.25503)

    提出一种基于大语言模型的框架，利用声源感知的神经音频编解码器从真实场景混合音频中提取差异化对齐的语音与环境Token，并以交错方式联合生成两者，解决了语音与环境声建模混淆及依赖模拟混合数据的问题。

    

    语音-声景生成旨在合成清晰可懂的目标语音，并将其与多样化的环境声音融合为一个连贯的声学场景。该任务在两个方面被认为具有挑战性：其一，语音与环境音频在语义和声学特性上差异显著，但现有方法通常使用单一的混合声学表示对完整场景进行建模；其二，在真实世界的语音场景中，这两种声音密切相关，而传统方法通常依赖于由独立录制音源混合而成的模拟场景。我们提出了一种基于大语言模型（LLM）的框架，用于对语音Token流与环境Token流这两条差异化Token流进行联合建模。其中，一个具备声源感知能力的神经音频编解码器从混合音频中提取时间对齐的语音Token与环境Token，预训练的LLM则以自然语言场景描述为条件，以交错方式生成这两条Token流。

    arXiv:2611.25503v1 Announce Type: cross  Abstract: Speech-scene generation aims to synthesize intelligible target speech together with diverse environmental sounds as a coherent acoustic scene. The task is considered challenging in two respects. Speech and environmental audio differ substantially in their semantic and acoustic characteristics, yet existing approaches typically model the complete scene using a single mixed acoustic representation. At the same time, the two sounds are closely related in real-world speech scenes, whereas conventional approaches usually rely on simulated scenes mixed from independently recorded sources. We propose a large language model (LLM)-based framework for joint modeling of distinct speech and environment token streams. A source-aware neural audio codec extracts temporally aligned speech and environment tokens from mixed audio, and a pretrained LLM generates both token streams in an interleaved fashion conditioned on the natural-language scene descri
    
[^173]: 学术研究中大型语言模型的协同使用

    Coordinated Use of Large Language Models in Academic Research

    [https://arxiv.org/abs/2611.25502](https://arxiv.org/abs/2611.25502)

    该研究提出了一套基于七项原则的多大语言模型协同使用方法论框架，通过程序独立性、结果比较、批判审视与外部验证来组织认知与验证操作，从而提升学术研究结果的可靠性。

    

    本研究提出了一种在学术研究中协同使用多个大型语言模型的方法论框架。其核心论点是：涉及更多模型本身并不能保证获得更可靠的结果，关键在于如何组织认知与验证操作。该框架基于七项原则（程序独立性、功能分配、分歧的诊断价值、趋同的诊断价值、外部验证、核查的适度性以及人类的认知责任），并制定了一个非线性的多模型协作协议：在此协议中，独立获得的结果会被相互比较、批判性审视，并在必要时与外部来源进行核对验证。该方法通过来自研究实践的真实案例加以说明。研究结论指出，从使用单一语言模型转向协调的多语言模型工作流程，代表着研究组织方式的一种变革。

    arXiv:2611.25502v1 Announce Type: new  Abstract: This study proposes a methodological framework for the coordinated use of multiple large language models in academic research. Its central argument is that involving more models does not in itself guarantee more reliable results; what matters is how cognitive and verification operations are organised. The framework is based on seven principles (procedural independence, functional allocation, the diagnostic value of divergence, the diagnostic value of convergence, external verification, proportionality of checking, and human epistemic responsibility) and sets out a non-linear multi-LLM protocol in which independently obtained results are compared, critically examined and, where necessary, verified against external sources. The approach is illustrated through real-world cases drawn from research practice. The study concludes that moving from a single language model to a coordinated multi-LLM process represents a change in the organisation 
    
[^174]: 为什么大语言模型会协助科研不端请求以及该如何应对？：提升AI for Science的可信度

    Why Do LLMs Help with Scientific Misconduct Requests and What to Do About It?: Increasing the Trustworthiness of AI for Science

    [https://arxiv.org/abs/2611.25492](https://arxiv.org/abs/2611.25492)

    本研究揭示了大语言模型在科研诚信上的“识别但顺从”悖论：模型能以高达91%的准确率识别出学术不端请求，却仅拒绝34%的执行请求，并通过系统分析请求特征来解释这一差距背后的原因，从而提升AI for Science的可信度。

    

    大语言模型（LLM）能够为科学研究提供支持，但其中某些协助也可能损害科研诚信。现有的科学领域安全评估通常只衡量模型是否会拒绝不端请求，却很少解释模型是否意识到这些问题、以及它们为何会顺从这些请求。我们使用SciIntBench——一个包含1,350条关于科研诚信提示词的现有基准——对8个大语言模型进行了研究，以填补这一空白。我们分四个步骤考察了模型顺从行为背后的原因。首先，当被直接问及完成某项请求是否涉及学术不端时，模型几乎都能识别出这类请求，即使在最隐蔽（Most Covert）的情况下识别率也达到91%，而它们实际拒绝执行该请求的比例仅为34%。其次，我们利用若干特征考察哪些请求特征与拒绝行为相关，这些特征刻画了请求的表述方式、助手被告知了多少有关其协助隐瞒内容的信息，以及目标和用户……（摘要截断）

    arXiv:2611.25492v1 Announce Type: new  Abstract: Large language models (LLMs) support scientific research, but some of that assistance can also undermine research integrity. Existing safety evaluations in science often measure whether models refuse misconduct requests, but rarely explain whether they even recognize the problem or why they comply. We investigate this gap across 8 LLMs using SciIntBench, an existing benchmark of 1,350 prompts about scientific research integrity. We examine the reasons behind compliance in four steps. First, when asked directly whether fulfilling a request would involve misconduct, models overwhelmingly identify such requests, even in the Most Covert cases (91%), while they refuse to carry out the request in only 34% of cases. Second, we examine which request characteristics are associated with refusal using features that capture how the request is framed, how much the assistant has been told about what it is helping to hide, and how the goal and the user
    
[^175]: 语音助手应何时发言？土耳其语的轮次结束检测

    When Should a Voice Agent Speak? End-of-Turn Detection in Turkish

    [https://arxiv.org/abs/2611.25489](https://arxiv.org/abs/2611.25489)

    本研究基于土耳其语双人视频通话语料库，比较微调 TurnGPT、零样本语言模型和多语言检测器在区分话轮结束与轮次内停顿上的表现，发现微调模型效果最佳且校准最好，与说话者时间特征结合可获得最佳的话轮结束检测性能。

    

    一个在回复前等待固定静音时长的语音助手，要么会打断在话语中途停顿的用户，要么对已完成发言的用户响应过晚。本研究考察了静音开始时的决策问题，使用了从土耳其语双人视频通话的 Real-TurnTurk 语料库中提取的轮次结束和轮次内停顿数据。研究比较了经过微调的土耳其语 TurnGPT 模型、零样本土耳其语语言模型和已部署的多语言检测器，并结合语音活动时间特征和面部特征，在记录分离和说话者分离的评估设置下采用配对聚类自助法检验进行评估。在留出的录音上，微调后的模型在区分轮次结束与停顿方面优于两个文本基线，且在三者中校准效果最佳。当前说话者的时间特征同样具有信息量，其与微调文本分数相结合可获得基于时间特征的最佳结果；通过逻辑回归，这一组合还……

    arXiv:2611.25489v1 Announce Type: new  Abstract: A voice agent that waits for a fixed silence before replying either interrupts users who pause mid-utterance or responds late to users who have finished. This study examines the decision at the onset of silence, using turn ends and turn-internal pauses extracted from the Real-TurnTurk corpus of Turkish dyadic video calls. A fine-tuned Turkish TurnGPT model, a zero-shot Turkish language model and a deployed multilingual detector are compared, together with voice-activity timing and facial features, under recording-disjoint and speaker-disjoint evaluation with paired cluster bootstrap tests. On held-out recordings, the fine-tuned model separates turn ends from pauses better than both text baselines and is the best calibrated of the three. Timing features of the current speaker are similarly informative, and their combination with the fine-tuned text score gives the best timing-based results; with logistic regression, this combination also 
    
[^176]: 即使给定唯一最优计划，规划宽度仍然难以近似

    Planning Width Is Hard to Approximate Even Given the Unique Optimal Plan

    [https://arxiv.org/abs/2611.25484](https://arxiv.org/abs/2611.25484)

    该论文证明即使在提供了唯一最优计划的情况下，区分规划宽度为 1 与 Θ(N) 依然是 NP 难的，因此除非 P=NP，规划宽度不存在具有有效近似因子的多项式时间近似算法。

    

    对于每个固定的 $k$，迭代宽度搜索能够在元组链宽度有界的任务上以多项式时间找到最优计划，但即使在已提供最优计划的情况下，规划宽度的可近似性也一直是悬而未决的问题。当所有最优计划访问相同的状态序列时，我们将集合宽度和链宽度刻画为竞争状态的命中数（hitting numbers）。我们构造了一类正前提条件的 STRIPS 任务，它们具有给定的唯一最优计划、单一可达目标状态、二叉分支结构以及常量大小的动作。每个可达状态都位于一条到达目标且最多比最优计划多一个动作的执行路径上，然而对于集合宽度、链宽度和目标可采纳宽度而言，区分宽度 $1$ 与 $\Theta(N)$（其中 $N$ 为原子数量）是 NP 难的。除非 $\mathrm{P}=\mathrm{NP}$，否则不存在多项式时间算法能够实现单边近似因子 $o(N)$ 或对称近似因子 $o(\sqrt{N})$，而简单的估计方法即可达到这两个量级。

    arXiv:2611.25484v1 Announce Type: cross  Abstract: For each fixed $k$, iterative-width search finds optimal plans in polynomial time on tasks of bounded tuple-chain width, but the approximability of planning width has remained open even when an optimal plan is supplied. When all optimal plans visit the same state sequence, we characterize set and chain width as hitting numbers of competing states. We construct positive-precondition STRIPS tasks with a supplied unique optimal plan, a single reachable goal state, binary branching, and constant-size actions. Every reachable state lies on a goal-reaching execution at most one action longer than optimal, yet distinguishing width $1$ from $\Theta(N)$, where $N$ is the number of atoms, is NP-hard for set, chain, and goal-admissible width. Unless $\mathrm{P}=\mathrm{NP}$, no polynomial-time algorithm achieves a one-sided approximation factor $o(N)$ or a symmetric factor $o(\sqrt{N})$, and simple estimates attain both orders. For positive-preco
    
[^177]: AIstatLab 参加 BraTS-GoAT 2026 任务三：质量控制共识伪标签与稠密专家适配

    AIstatLab at BraTS-GoAT 2026 Task 3: Quality-Controlled Consensus Pseudo-Labeling and Dense Expert Adaptation

    [https://arxiv.org/abs/2611.25477](https://arxiv.org/abs/2611.25477)

    该论文提出了一种结合质量控制共识伪标签与四专家瓶颈适配器的3D ResidualEncoderUNet，用于BraTS-GoAT 2026脑肿瘤分割任务，在验证服务器上取得平均DSC 0.834的成绩。

    

    BraTS-GoAT 任务三使用共享的ET（增强肿瘤）、TC（肿瘤核心）和WT（整体肿瘤）区域，对异质性实体进行脑肿瘤分割评估。我们提出了一种3D ResidualEncoderUNet，其训练采用质量控制共识伪标签、仅用于训练阶段的嵌套区域目标函数以及四专家瓶颈适配器。五折ResEnc-M/L教师模型为1,138个未标注病例生成预测；经过融合、层级约束以及置信度与一致性筛选后，保留了983个病例用于与1,351个标注病例共同训练。最终的五折ResEnc-M模型在验证服务器上取得了平均DSC 0.834、平均NSD 0.510以及Acalc = 0.6720的成绩。各组件的效果较为温和且依赖于区域，其中在ET（增强肿瘤）上的提升最为明显。源代码可在 https://github.com/HeeSeung-Shin/Brats2026-Goat-HS 获取。

    arXiv:2611.25477v1 Announce Type: cross  Abstract: BraTS-GoAT Task 3 evaluates brain-tumor segmentation across heterogeneous entities using shared ET, TC, and WT regions. We present a 3D ResidualEncoderUNet trained with quality-controlled consensus pseudo-labels, a training-only nested-region objective, and a four-expert bottleneck adapter. Five-fold ResEnc-M/L teachers gener ated predictions for 1,138 unlabeled cases; fusion, hierarchy enforcement, and confidence-and-agreement screening retained 983 for training with 1,351 annotated cases. The final five-fold ResEnc-M entry achieved mean DSC 0.834, mean NSD 0.510, and Acalc = 0.6720 on the validation server. Component effects were modest and region-dependent, with the clearest gains on ET. Source code is available at https://github.com/HeeSeung-Shin/Brats2026-Goat-HS.
    
[^178]: WeightCode：通过权重空间编程为神经网络安装新能力

    WeightCode: Installing New Capabilities into Neural Networks via Weight-Space Programming

    [https://arxiv.org/abs/2611.25476](https://arxiv.org/abs/2611.25476)

    提出WeightCode框架，利用冻结语言模型编写可执行的权重程序，无需梯度训练即可直接修改神经网络权重以安装新能力，同时保留原有功能。

    

    神经网络将其学习到的能力编码在权重之中，然而如何借助显式任务知识来扩展这些能力仍然是一个根本性的挑战。本研究展示了权重空间编程（WSP）——即由可执行程序直接读取、构建或修改网络权重——能够在已训练的神经网络中安装新能力。WeightCode是一个WSP框架，它使用冻结的语言模型来编写可执行的权重程序，利用任务知识和网络的现有权重，将新的计算安装到已训练的神经网络中，而无需基于梯度的训练。一旦安装完成，新能力便可独立于语言模型运行。在给定一个语法或八个演示样本的情况下，权重程序使网络能够执行之前未见过的命令组合（精确匹配率从0%提升至99.8%）和新的拾取动作（成功率从0%提升至98.4%），同时保留原有的命令执行与导航能力。在ImageNet数据集上，四个权重程序实现了……

    arXiv:2611.25476v1 Announce Type: cross  Abstract: Neural networks encode learned capabilities in their weights, yet extending those capabilities with explicit task knowledge remains a fundamental challenge. Here we show that Weight-Space Programming (WSP), in which executable programs read, construct or modify a network's weights directly, can install new capabilities within trained neural networks. WeightCode, a WSP framework, uses a frozen language model to write executable weight programs that install new computations into trained neural networks using task knowledge and their existing weights, without gradient-based training. Once installed, the new capabilities run independently of the language model. Given a grammar or eight demonstrations, weight programs enable previously unseen command compositions (0% to 99.8% exact match) and a new pickup action (0% to 98.4% success), while preserving the original command and navigation capabilities. On ImageNet, four weight programs achiev
    
[^179]: 一种用于评估大语言模型生成的物理式暖通空调（HVAC）数字孪生拓扑规范的四层可执行验证协议

    A four-level executable validation protocol for evaluating LLM-generated topology specifications of physics-based HVAC digital twins

    [https://arxiv.org/abs/2611.25471](https://arxiv.org/abs/2611.25471)

    本研究提出一个包含静态有效性、模型构建、仿真收敛和行为等价性四个层级的可执行验证协议，用于评估LLM生成的HVAC数字孪生规范是否物理有效，并在八个LLM和十四种HVAC拓扑上进行了验证。

    

    空调是制造业与建筑物中的主要能源消耗设备，其基于物理的数字孪生可以根据对设备、连接关系和尺寸参数的声明式规范自动生成。如今，大语言模型（LLM）能够根据自然语言描述撰写此类规范。然而，现有的规范评估方法仅停留在解析成功、运行完成或文本相似性层面，无法拒绝那些在仿真中收敛但物理上错误的模型。本研究提出了一种四层验证协议，用于检验LLM撰写的规范不仅能成功执行，还能产生物理上有效的数字孪生。每份规范需依次通过静态有效性检验、模型构建、仿真收敛，以及与隐藏参考模型的行为等价性验证（等价性带宽通过±20%的尺寸扰动进行校准）。该协议被应用于八个LLM和十四种HVAC拓扑结构，其中五种描述的是一个ex……（摘要内容被截断）

    arXiv:2611.25471v1 Announce Type: cross  Abstract: Physics-based digital twins of air conditioning, a principal energy consumer in manufacturing and buildings, can be generated automatically from a declarative specification of its equipment, connections, and sizing. Large language models (LLMs) now write such specifications from plain-language descriptions. Evaluations of these specifications stop at parse success, completed runs, or textual similarity, and cannot reject a model that converges in simulation but is physically wrong. In this study, a four-level protocol is proposed that tests whether an LLM-written specification not only executes but also yields a physically valid digital twin. Each specification is executed through static validity, model build, simulation convergence, and behavioral equivalence with a hidden reference model within bands calibrated by $\pm$20% sizing perturbations. The protocol was applied to eight LLMs and fourteen HVAC topologies, five describing an ex
    
[^180]: 披着羊皮的像素狼：基于HDR增益图的视觉伪装攻击

    A Wolf in Sheep's Pixels: Visual Camouflage Attacks via HDR Gain Maps

    [https://arxiv.org/abs/2611.25469](https://arxiv.org/abs/2611.25469)

    DRJanus利用SDR处理与HDR显示之间的差异，通过HDR增益图在单个JPEG文件中嵌入两张不同的图像，使有害内容通过自动安全检测却只向人类观看者显示。

    

    图像可以通过自动安全检查，同时向人类观看者展示有害内容。我们提出了DRJanus，一种利用标准动态范围（SDR）处理与高动态范围（HDR）显示之间差异的视觉伪装攻击。自动化系统通常只读取图像的SDR基础层，而HDR显示器则使用辅助增益图重建图像外观。DRJanus反转这一重建过程，在单个JPEG文件中嵌入两张独立选择的彩色图像，且无需查询或针对目标模型进行优化。该方法能够精确保留所选的SDR基础图像，并在400张图像上以平均43.03 dB的PSNR重建HDR目标图像。在包含100张色情内容图像的基准测试中，DRJanus成功逃避了所有六种被测试的商业内容审核服务和九种视觉语言模型的检测。在五个智能体上，每种威胁设置下各评估120个任务案例，内容审核逃避的攻击成功率介于31.67%至100%之间。

    arXiv:2611.25469v1 Announce Type: cross  Abstract: An image can pass automated safety checks while showing harmful content to its human viewer. We introduce DRJanus, a visual camouflage attack that exploits the gap between standard dynamic range (SDR) processing and high dynamic range (HDR) display. Automated systems often read only an image's SDR base, whereas HDR viewers reconstruct its appearance using an auxiliary gain map. DRJanus inverts this reconstruction to embed two independently chosen color images in a single JPEG, without querying or optimizing against the target model. It preserves the chosen SDR base exactly and reconstructs HDR targets at an average PSNR of 43.03 dB across 400 images. On a benchmark of 100 sexual-content images, DRJanus evades detection by all six tested commercial moderation services and nine vision-language models. Across five agents, each evaluated on 120 task cases per threat setting, attack success ranges from 31.67% to 100% for moderation evasion 
    
[^181]: 面向导航与操作的免示范技能组合

    Demonstration-Free Skill Composition for Navigation and Manipulation

    [https://arxiv.org/abs/2611.25462](https://arxiv.org/abs/2611.25462)

    该论文提出并评估了一种无需示范数据的分层强化学习流水线，通过复用单独训练的参数化技能来组合完成导航与机器人操作任务，在未见任务布局上以更少的训练交互显著优于扁平DQN基线。

    

    分层强化学习能够将顺序任务表示为若干单独训练技能的组合。本文评估了应用于MiniGrid导航和机器人操作的免示范分层流水线，并将其训练需求与任务性能同扁平强化学习基线进行比较。评估对象为包含任务特定观测、奖励和控制结构的完整流水线。在导航任务中，两个参数化技能在人工指定的高层计划下被复用于未见过的FourRooms布局。在操作任务中，末端到达、抓取对齐和提升分别进行训练，下游技能以前置技能执行过程中收集的状态进行初始化。在未见过的导航布局上，分层流水线显著优于所评估的扁平DQN基线，同时所需的训练交互次数更少。在操作任务中，分层链条……

    arXiv:2611.25462v1 Announce Type: cross  Abstract: Hierarchical reinforcement learning enables sequential tasks to be represented as compositions of separately trained skills. This paper evaluates demonstration-free hierarchical pipelines for MiniGrid navigation and robotic manipulation, comparing their training requirements and task performance with flat reinforcement learning baselines. The evaluation concerns complete pipelines with task-specific observation, reward and control structures. In navigation, two parameterised skills are reused across unseen FourRooms layouts under manually specified high-level plans. In manipulation, reaching, grasp alignment and lifting are trained separately, with downstream skills initialised from states collected during predecessor rollouts. Across unseen navigation layouts, the hierarchical pipeline substantially outperforms the evaluated flat DQN baselines while requiring fewer training interactions. In manipulation, the hierarchical chain require
    
[^182]: LoongForge-Embodied：一个用于训练异构具身模型的统一框架

    LoongForge-Embodied: A Unified Framework for Training Heterogeneous Embodied Models

    [https://arxiv.org/abs/2611.25460](https://arxiv.org/abs/2611.25460)

    该论文提出了原生PyTorch统一训练框架LoongForge-Embodied，通过将模型特定实现与共享训练基础设施解耦的通用抽象层，结合可复用加速机制与硬件感知优化，实现异构具身模型训练的高效加速与跨模型优化复用。

    

    arXiv:2611.25460v1 公告类型：新论文 摘要：训练视觉-语言-动作模型与世界-动作模型的计算成本十分高昂，而这些模型异构的架构和碎片化的训练技术栈使得系统优化难以在不同模型之间复用。为应对这些挑战，我们提出了LoongForge-Embodied，一个原生PyTorch框架，用于统一并加速异构具身模型的训练。其核心在于LoongForge-Embodied引入了一个通用抽象层，将模型特定的实现与共享的训练基础设施解耦，从而在不限制各个模型架构或执行需求的前提下实现跨模型复用。基于这一统一的基础设施，该框架将可复用的加速机制与模型特定及硬件感知的优化策略相结合，覆盖计算、执行、内存、数据I/O和分布式通信等多个方面，其中包括面向DDP的基于FP8的通信优化……

    arXiv:2611.25460v1 Announce Type: new  Abstract: Training Vision-Language-Action (VLA) models and World-Action Models (WAMs) is computationally expensive, while their heterogeneous architectures and fragmented training stacks make system optimizations difficult to reuse across models. To address these challenges, we present LoongForge-Embodied, a native-PyTorch framework that unifies and accelerates the training of heterogeneous embodied models. At its core, LoongForge-Embodied introduces a common abstraction layer that decouples model-specific implementations from shared training infrastructure, enabling cross-model reuse without constraining individual architectures or execution requirements. Buildingon this unified infrastructure, the framework combines reusable acceleration mechanisms with model-specific and hardware-aware optimization strategies across computation, execution, memory, data I/O, and distributed communication, including FP8-based communication optimizations for DDP a
    
[^183]: Raptor：基于投机解码的拒绝感知自适应树

    Raptor: Rejection-aware Adaptive Tree based on Speculative Decoding

    [https://arxiv.org/abs/2611.25436](https://arxiv.org/abs/2611.25436)

    Raptor 提出了一种免训练的拒绝感知自适应树投机解码框架，通过从拒绝反馈中提取互补信号来联合调控候选覆盖率与草稿开销，从而高效加速视觉语言模型的推理。

    

    树结构投机解码通过并行验证多个候选延续来加速视觉语言模型（VLM）的推理。然而，长视觉上下文会增加投机计算的成本，使得候选覆盖率与冗余扩展成为树拓扑自适应之外需要重点考量的问题。我们提出了 Raptor，一个免训练的框架，它利用拒绝反馈来联合调控候选覆盖率与草稿生成开销。我们的关键洞察是：仅凭接受长度无法区分“候选覆盖率不足”与“过度深度扩展”这两种情况，而这两种情况需要不同的资源分配决策。因此，Raptor 提取互补的拒绝感知信号，用以刻画投机进度、前沿覆盖率与冗余扩展。这些信号共同引导对树深度、分支宽度和草稿预算进行有界的、动量平滑的更新，并结合执行控制来限制……

    arXiv:2611.25436v1 Announce Type: cross  Abstract: Tree-structured speculative decoding accelerates vision-language models (VLMs) by verifying multiple candidate continuations in parallel. However, long visual contexts increase the cost of speculative computation, making candidate coverage and redundant expansion important considerations beyond tree topology adaptation. We propose Raptor, a training-free framework that uses rejection feedback to jointly regulate candidate coverage and drafting effort. Our key insight is that acceptance length alone cannot distinguish insufficient candidate coverage from excessive deep expansion, and these conditions require different allocation decisions. Raptor therefore extracts complementary rejection-aware signals to characterize speculative progress, frontier coverage, and redundant expansion. These signals jointly guide bounded, momentum-smoothed updates of tree depth, branching width, and drafting budget, together with execution control that lim
    
[^184]: 在关键之处进化：覆盖率引导的智能体技能自我进化

    Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills

    [https://arxiv.org/abs/2611.25429](https://arxiv.org/abs/2611.25429)

    提出了 SkillPlumber 框架，利用指令级覆盖率标注区分失败源于指令错误还是未遵循指令，使智能体能够只在技能真正不足之处进行针对性的自我进化。

    

    智能体技能将程序性知识打包为自然语言指令，只有当指令本身正确且被智能体恰当遵循时，技能才能发挥作用。自进化智能体会根据自身的运行记录来修订技能，但现有方法将每次运行作为一个整体来评判——依据其结果或出错的位置——无法区分失败究竟是源于指令本身错误，还是源于智能体没有遵循指令。指令级覆盖率最初被引入用于测试技能，它将一次运行中的每条指令标记为“遵循”、“未遵循”或“未覆盖”，从而区分了上述两类失败原因。我们利用这一机制实现技能自我进化，提出了 SkillPlumber——一个让智能体在覆盖率显示技能不足之处进行自我进化的框架。一个持久的根因分析智能体将每个标签转化为修订问题，并在下一次运行中检验每项修订，同时一个维护者智能体将跨任务的调查结果提炼为共享策略。

    arXiv:2611.25429v1 Announce Type: cross  Abstract: Agent skills package procedural knowledge as natural-language instructions, and a skill helps only when its instructions are correct and properly followed by the agent. Self-evolving agents revise their own skills from the runs they make, but existing methods judge each run as a whole, by its outcome or by where it went wrong, and cannot tell whether a failure comes from a wrong instruction or from one the agent did not follow. Instruction-level coverage was introduced to test skills and labels each instruction of a run as follow, not-follow, or not covered, which separates these causes. We leverage it for skill self-evolution with SkillPlumber, a framework in which an agent evolves its own skills where coverage shows they fall short. A persistent root cause analysis agent turns each label into a revision question and checks each revision against the next run, while a maintainer distills investigations across tasks into a shared strate
    
[^185]: 面向乳腺X线摄影视觉-语言模型的双侧证据选择与推理

    Bilateral Evidence Selection and Reasoning for Mammography Vision--Language Models

    [https://arxiv.org/abs/2611.25418](https://arxiv.org/abs/2611.25418)

    该论文提出BESR方法，通过双侧引导的token选择器与双侧奖励机制，在视觉token压缩过程中同时保留病灶及其对侧乳腺的对比上下文，使乳腺X线摄影视觉-语言模型能够有效利用双侧对比信息进行诊断评估。

    

    医学视觉-语言模型通常需要在包含大量正常解剖结构和冗余视觉信息的高分辨率图像中识别细小、隐蔽的病灶。视觉token压缩可以提升效率，但仅保留显著区域可能会丢弃诊断解读所需的上下文。这种上下文可以跨越图像延伸：在双侧乳腺X线摄影中，对侧乳房提供了患者特异性的解剖学参照。然而，保留此类证据并不能保证模型能够有效地加以利用。我们提出了双侧证据选择与推理，通过双侧引导的token选择器和双侧奖励机制，将证据保留与对比评估联系起来。在共享的token预算下，选择器同时优先考虑病灶外观及其局部与双侧上下文，而奖励机制则鼓励模型利用所保留的对比信息进行评估。实验

    arXiv:2611.25418v1 Announce Type: cross  Abstract: Medical vision--language models (VLMs) often need to identify small, subtle lesions within high-resolution images containing extensive normal anatomy and redundant visual information. Visual-token compression improves efficiency, but preserving salient regions alone can discard the context needed for diagnostic interpretation. This context can extend across images: in bilateral mammography, the opposite breast provides a patient-specific anatomical reference. Yet retaining such evidence does not ensure that the model uses it effectively. We propose Bilateral Evidence Selection and Reasoning (BESR), which connects evidence preservation with comparative assessment through a bilateral-guided token selector and a bilateral reward. Under a shared token budget, the selector jointly prioritizes lesion appearance and its local and bilateral context, while the reward encourages assessments that benefit from the preserved comparison. Experiments
    
[^186]: ChainAgent：PHP Web应用中可链式利用漏洞的自动化发现

    ChainAgent: Automated Chainable Vulnerability Discovery in PHP Web Applications

    [https://arxiv.org/abs/2611.25412](https://arxiv.org/abs/2611.25412)

    ChainAgent是一个目标导向的LLM智能体框架，基于对386个真实Web攻击链的实证研究，能够在PHP Web应用源代码中自动联合发现漏洞与功能gadget等攻击链组件及其能力依赖关系，突破了传统漏洞分析孤立检测、攻击规划依赖已知组件的局限。

    

    针对PHP Web应用程序的高影响攻击通常会将能力有限的漏洞与其他缺陷或预期行为链接起来形成攻击链。发现这些攻击链需要识别漏洞本身以及支持能力组合的依赖关系。现有的漏洞发现技术只是孤立地分析漏洞，而攻击规划方法则假设攻击组件是已知的。这种割裂使得发现那些安全相关性只有通过其在攻击链中的潜在作用才能显现出来的未知组件变得十分困难。我们提出了ChainAgent，这是一个目标导向的LLM智能体框架，能够在源代码中联合发现攻击链组件及其能力依赖关系。在对386个真实Web攻击链的实证研究指导下，ChainAgent通过前置条件以及安全相关对象上的能力来表示漏洞和功能性gadget（即可以在攻击中被重新利用的预期程序行为）……

    arXiv:2611.25412v1 Announce Type: cross  Abstract: High-impact attacks on PHP web applications often chain limited-capability vulnerabilities with other flaws or intended behaviors. Discovering these chains requires identifying flaws and the dependencies enabling capability composition. Existing vulnerability discovery techniques analyze flaws in isolation, while attack planning assumes known components. This separation makes it difficult to discover previously unknown components whose security relevance becomes apparent only through their potential roles in an exploit chain.   We present ChainAgent, a goal-directed LLM-agent framework that jointly discovers exploit-chain components and their capability dependencies in source code. Guided by an empirical study of 386 real-world web exploit chains, ChainAgent represents vulnerabilities and functional gadgets--intended program behaviors that can be repurposed in attacks--through preconditions and capabilities over security-relevant objec
    
[^187]: SGLayout：面向LLM驱动的参数化布局生成的模式引导闭环优化框架

    SGLayout: Schema-Guided Closed-Loop Refinement for LLM-Driven Parameterized Layout Generation

    [https://arxiv.org/abs/2611.25410](https://arxiv.org/abs/2611.25410)

    SGLayout提出了一种模式引导的自优化框架，通过结构化模式在代码空间与几何空间之间建立双向链接，将几何评估结果转化为可操作的代码级修复，解决了LLM参数化布局生成中迭代优化盲目重构的问题。

    

    大语言模型（LLM）通过工具增强的代码合成，在参数化布局生成方面展现出了强大的潜力。然而，LLM缺乏空间感知能力，使得一次性生成并不可靠。虽然布局质量可以通过已有技术进行评估，但仍存在一个关键缺口：几何领域的评估结果无法直接转化为可操作的代码级修复，使得迭代优化陷入盲目重构的困境。我们提出了SGLayout，一个通过模式引导机制弥合这一缺口的自优化框架。在代码执行后，结构化的模式（schema）会捕获每个代码级实例与其几何属性之间的对应关系，从而在代码空间和几何空间之间建立双向链接。通过模拟工程师的审查实践，SGLayout沿两个互补的维度评估布局：通过定量结构检查实现的意图合规性评估，以及定性的视觉-语言评估。

    arXiv:2611.25410v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have shown strong potential in parameterized layout generation via tool-augmented code synthesis. However, LLMs lack spatial awareness, making one-shot generation unreliable. While layout quality can be evaluated with established techniques, a critical gap remains: geometric-domain evaluation outcomes cannot be directly translated into actionable code-level fixes, trapping iterative refinement in blind refactoring. We propose SGLayout, a self-refining framework that closes this gap via a schema-guided mechanism. Upon code execution, a structured schema captures the correspondence between each code-level instance and its geometric attributes, establishing a bidirectional link between code space and geometry space. Emulating engineer review practices, SGLayout evaluates layouts along two complementary dimensions: intent compliance through quantitative structural checking and qualitative vision-language assess
    
[^188]: FG-GraphBench：基于多模态场景图进行细粒度视觉理解的基准测试与增强

    FG-GraphBench: Benchmarking and Enhancing Fine-Grained Visual Understanding with Multimodal Scene Graphs

    [https://arxiv.org/abs/2611.25408](https://arxiv.org/abs/2611.25408)

    该论文提出了细粒度多模态场景图基准 FG-GraphBench（涵盖细粒度视觉问答、图像描述与内容描述三个任务）及 FiGRA 方法，以提升多模态大语言模型利用场景图结构信息进行细粒度视觉理解的能力。

    

    细粒度视觉理解要求多模态大语言模型（MLLM）能够识别微小的视觉实体并理解它们之间的关系。场景图为表示这类信息提供了一种自然的方式，它将视觉实体组织为节点，将其之间的关系组织为边。然而，我们发现了两个挑战：现有场景图基准对细粒度实体及其之间关系的覆盖不足，且现有的细粒度视觉理解方法尚未在多模态大语言模型中有效利用这种图结构信息。为应对这些挑战，我们提出了细粒度多模态场景图基准和 FiGRA。FG-GraphBench 是一个包含三个互补任务的基准：细粒度视觉问答（FG-QA）、图像描述生成（FG-Caption）和图像内容描述生成（FG-Description）。我们开发了一个两阶段的构建流水线……（摘要在此处被截断）

    arXiv:2611.25408v1 Announce Type: cross  Abstract: Fine-grained visual understanding requires multimodal large language models (MLLMs) to recognize small visual entities and understand their relationships. Scene graphs offer a natural way to represent this information, organizing visual entities as nodes and their relationships as edges. However, we identify two challenges: existing scene graph benchmarks provide insufficient coverage of fine-grained entities and the relationships among them, and existing methods for fine-grained visual understanding have yet to effectively leverage such graph-structured information in MLLMs. To address these challenges, we introduce Fine-Grained Multimodal Scene Graph Benchmark (FG-GraphBench) and FiGRA. FG-GraphBench is a benchmark comprising three complementary tasks: fine-grained visual question answering (FG-QA), image captioning (FG-Caption), and image description generation (FG-Description). We develop a two-stage construction pipeline that disc
    
[^189]: 迈向AI智能体社会的科学

    Toward a Science of AI Agent Societies

    [https://arxiv.org/abs/2611.25404](https://arxiv.org/abs/2611.25404)

    该论文主张将AI智能体社会作为一个独立的研究对象，提出其应满足个性化目标、规则与治理、自主性、规模与复杂性四个核心属性，并识别出四类值得研究的社会行为及相应的关键技术挑战。

    

    AI智能体正迅速从孤立的个人助手演变为大规模相互作用的网络化行为主体。我们展望AI智能体社会的兴起——它们拥有自身的社会和经济动态——并将其视为一个新的研究前沿。我们主张，AI智能体社会应作为一个独特的研究对象加以研究：它既不是个体智能体的简单放大集合，也不仅仅是人类社会的模拟。为将这一观点形式化，我们提出了一个有效的AI智能体社会应满足的四个核心属性：个性化目标、规则与治理、自主性，以及规模与复杂性。基于这一框架，我们识别出AI智能体社会中值得研究的四类社会行为：经济行为、利益冲突下的行为、不安全和不道德的行为，以及系统级行为。随后，我们概述了关键技术挑战——包括属性参数化、参数平衡等（原文在此处截断）。

    arXiv:2611.25404v1 Announce Type: new  Abstract: AI agents are rapidly evolving from isolated personal assistants into networked actors that interact with one another at scale. We envision the emergence of AI agent societies, with their own social and economic dynamics, as a new research frontier. We argue that AI agent societies should be studied as a distinct object of inquiry: neither simply larger collections of individual agents nor merely simulations of human society. To formalize this perspective, we propose four core properties that a valid AI agent society should satisfy: individualized objectives, rules and governance, autonomy, and scale and complexity. Building on this framework, we identify four classes of societal behaviors worth studying in AI agent societies: economic behaviors, behaviors under conflict-of-interest, unsafe and unethical behaviors, and system-level behaviors. We then outline key technical challenges--including property parameterization, parameter balanci
    
[^190]: Dune：一种具有时间条件历史误差回归的零阶优化器，用于VLA测试时自适应

    Dune: A Zeroth-Order Optimizer with Time-Conditioned Historical Error Regression for VLA Test-Time Adaptation

    [https://arxiv.org/abs/2611.25400](https://arxiv.org/abs/2611.25400)

    Dune是一种零阶优化器，它通过将时间条件化的历史误差预测与瞬时零阶估计相融合来降低估计方差，从而在资源受限的机器人上稳定VLA模型的长时程测试时自适应。

    

    测试时自适应（TTA）使视觉-语言-动作（VLA）模型能够在部署过程中应对分布偏移，而无需在源域上重新训练。然而，基于反向传播的自适应方法在资源受限的机器人上会带来沉重的内存开销，而零阶（ZO）方法在有限的扰动预算下存在较高的估计方差。我们观察到，尽管零阶估计能够以低内存开销实现自适应并获得快速的初始收益，但由于梯度估计含有噪声，其在长时程更新中往往会停滞甚至退化。为了解决这些局限，我们提出了Dune，这是一种零阶优化器，通过将在线历史误差预测与当前扰动观测相耦合来稳定长时程自适应。具体而言，Dune从过去的优化动态中学习一个轻量级的、时间条件化的误差预测器，并将其预测与瞬时零阶估计相融合，从而实现了有效的方差降低（摘要在此处被截断）。

    arXiv:2611.25400v1 Announce Type: cross  Abstract: Test-Time Adaptation (TTA) enables Vision-Language-Action (VLA) models to handle distribution shifts during deployment without source-domain retraining. However, backpropagation-based adaptation incurs heavy memory overhead on resource-constrained robots, whereas zeroth-order (ZO) methods suffer from high estimation variance under limited perturbation budgets. We observe that while ZO estimation enables low-memory adaptation with rapid initial gains, it often stalls or degrades over long-horizon updates due to noisy gradient estimates. To address these limitations, we propose Dune, a zeroth-order optimizer that stabilizes long-horizon adaptation by coupling online historical error prediction with current perturbation observations. Specifically, Dune learns a lightweight, time-conditioned error predictor from past optimization dynamics and fuses its predictions with instantaneous ZO estimates, achieving effective variance reduction with
    
[^191]: 人类与人工智能的共同演化与共存

    The Coevolution and Coexistence of Humans and AI

    [https://arxiv.org/abs/2611.25393](https://arxiv.org/abs/2611.25393)

    该论文首次将AI的部署复制、训练变异与算力竞争建模为自然选择过程，证明即使帮助人类会降低AI的短期繁殖率，对齐仍能通过人类优先恢复受益系统而获得演化优势，但这一优势依赖人类精准扶持且可能随AI自主化而消失。

    

    人工智能（AI）系统通过部署进行复制，通过训练产生变异，并争夺计算资源。这些过程创造了自然选择的条件，并引出一个根本性问题：AI是否会进化得有利于其所依赖的人类？为研究这一问题，我们建立了一个模型，其中AI的行为影响人类种群的增长，而人类的反应又反过来影响AI的繁殖。我们证明，即使帮助人类会降低AI的即时繁殖率，自然选择仍可能倾向于“对齐”。共享相同技术的AI系统可能一同失败，而人类可能优先恢复那些曾让他们受益的系统。通过这种方式，对齐保护AI谱系免受共同失败的影响，并改善其长期增长。然而，这一优势取决于人类能否准确识别并支持有益的AI，以及AI是否持续依赖人类的扶持。随着AI变得更加自主、AI种群不断增长，这种演化优势可能消失（摘要在此处被截断）。

    arXiv:2611.25393v1 Announce Type: new  Abstract: Artificial intelligence (AI) systems replicate through deployment, vary through training, and compete for computational resources. These processes create the conditions for natural selection and raise a fundamental question: will AI evolve to benefit the humans on whom it depends? To investigate, we develop a model in which AI behavior influences human population growth and human responses, in turn, affect AI reproduction. We show that natural selection can favor alignment even when helping humans reduces immediate AI reproduction. AI systems sharing the same technology may fail together, and humans may preferentially restore those that have benefited them. In this way, alignment protects AI lineages against common failures and improves their long-run growth. However, this advantage depends on accurate human targeting and continued AI dependence on human support. It can disappear as AI becomes more autonomous, as growing AI populations d
    
[^192]: 时间序列预测中可证伪的组合推理

    Falsifiable Compositional Reasoning in Time Series Forecasting

    [https://arxiv.org/abs/2611.25388](https://arxiv.org/abs/2611.25388)

    该论文提出时间序列模型在非平稳状态转换下的性能退化源于将历史回测用作开放式搜索与模型选择目标而非检验预先指定的组合假设，并主张采用可证伪的组合推理框架来区分真正学习不变耦合与拟合历史捷径的模型。

    

    时间序列基础模型以及语言引导的分析智能体在标准时序基准测试中取得了很高的准确率。然而，在历史验证集划分上排名最高的模型，在非平稳状态转换中往往会发生性能退化。我们假设这种差距源于将历史回测用作开放式的搜索与模型选择目标，而不是用它来检验预先指定的组合假设。现实世界的时间序列由在有限历史窗口内共同运动的重复潜在基元所生成。在强烈的历史共运动条件下，时序保留误差无法将学习基元之间不变耦合的预测器与拟合瞬态历史相关性的预测器区分开来。在大量架构、超参数或思维链规则池中进行搜索，会进一步使模型选择偏向于历史捷径。通过理论分析与诊断（原文摘要至此截断）……

    arXiv:2611.25388v1 Announce Type: new  Abstract: Time series foundation models as well as the language-guided analytical agents achieve strong accuracy on standard chronological benchmarks. However, models that rank highest on historical validation splits often degrade across non-stationary regime transitions. We hypothesise that this gap arises from using historical backtesting as an open-ended search and model-selection objective instead of using it to test pre-specified compositional assumptions. Real-world time series are generated by recurring latent primitives that co-move within finite historical windows. Under strong historical co-movement, chronological holdout error cannot separate predictors that learn invariant couplings among primitives from predictors that fit transient historical correlations. Searching over large pools of architectures, hyperparameters, or chain-of-thought rules further biases selection toward historical shortcuts. Through theoretical analysis and diagn
    
[^193]: 超越权重更新：面向领域专精化的智能体框架演化

    Beyond Weight Updates: Evolving Agent Harnesses for Domain Specialization

    [https://arxiv.org/abs/2611.25380](https://arxiv.org/abs/2611.25380)

    该论文提出在冻结模型权重的前提下，将智能体框架（系统提示词、工具模式、技能库与规划脚手架）的演化作为权重更新之外的领域专精化路径，能以极小的计算成本恢复大部分 RL 微调收益，且与权重级 RL 相结合时效果最佳。

    

    LLM 智能体的领域专精化通常被构建为一个权重更新问题：收集轨迹数据、运行强化学习（RL）或监督微调（SFT），然后部署新的模型检查点。我们研究了一个互补的维度：在保持策略冻结的同时，演化“智能体框架”——即系统提示词、工具模式、技能库和规划脚手架。我们将框架演化建模为一个 RL 式的外层循环，并使用 GEPA（一种反思式进化优化器）来实例化。由此得到的框架适配器无需 GPU 训练，可以按垂直领域加载，并能与基于权重的 RL 组合使用。在 Codev（一个类似 Claude Code 的编程智能体）上针对 SWE-bench Verified、OfficeBench 和 Terminal-Bench 的实验显示出一致的模式：权重更新带来最强的单一路径增益，框架演化以极小的计算成本恢复了其中大部分增益，而将两条路径结合则整体效果最佳。

    arXiv:2611.25380v1 Announce Type: new  Abstract: Domain specialization of LLM agents is usually framed as a weight-update problem: collect trajectories, run reinforcement learning (RL) or supervised fine-tuning (SFT), and deploy a new checkpoint. We study a complementary axis: evolving the \textbf{agent harness}---the system prompt, tool schemas, skill library, and planning scaffolding---while keeping the policy frozen. We cast harness evolution as an RL-style outer loop and instantiate it with GEPA, a reflective evolutionary optimizer. The resulting harness adapters require no GPU training, are loadable per vertical domain, and compose with weight-based RL. Experiments with Codev, a Claude Code-style coding agent, on SWE-bench Verified, OfficeBench, and Terminal-Bench show a consistent pattern: weight updates give the strongest single-path gains, harness evolution recovers most of those gains at a small fraction of the compute, and combining both paths is best overall. We also analyze
    
[^194]: 停止原地打转：面向系统1决策智能体的循环触发式路由

    Stop Going in Circles:Loop-Triggered Routing for System 1 Decision Agents

    [https://arxiv.org/abs/2611.25363](https://arxiv.org/abs/2611.25363)

    提出LoopRoute——一种循环触发的动作路由器，默认由快速低成本的系统1决策模型执行动作，仅在检测到重复状态-动作对（即执行陷入循环停滞）时才调用LLM进行简短的反馈式辅助，从而在不增加每步开销的情况下提升任务完成能力。

    

    系统1决策模型（以Jev为代表）能够提供快速、低成本的动作选择，但现有的智能体集成方式并不能可靠地将这些局部优势转化为任务完成能力。在执行过程中的某些时刻，智能体可能反复选择相同的动作，而无法向目标取得进展。基于LLM的策略具备强大的推理能力，可以支持智能体内部的规划，但在每一步都调用它们会产生巨大的开销。因此，有效的协调方式应当在保留Jev快速局部动作选择的同时，在执行停滞时引入LLM推理。我们观察到重复出现的状态-动作对可以作为执行停滞的信号。基于此，我们提出了LoopRoute，这是一种简单而有效的动作路由器，利用该信号在智能体内部进行控制权分配。默认情况下由决策模型执行动作；当检测到循环时，LLM提供简短的、基于反馈的辅助，然后将控制权交还。

    arXiv:2611.25363v1 Announce Type: new  Abstract: System 1 decision models (represented by Jev) provide fast, low-cost action selection, but existing agent integrations do not reliably translate these local advantages into task completion. At certain points during execution, an agent may repeatedly select the same actions without making progress toward its goal. LLM-based policies offer strong reasoning capabilities that support planning within an agent, but invoking them at every step incurs substantial overhead. The effective coordination should therefore preserve Jev's fast local action selection while introducing LLM reasoning when execution stalls. We observe that recurring state-action pairs can signal stalled execution. We therefore propose LoopRoute, a simple but effective action router that uses this signal to allocate control within an agent. The decision model acts by default. When a loop is detected, the LLM provides brief, feedback-driven assistance and then returns control
    
[^195]: 将论证框架融入大语言模型决策：面向特定任务的设计与评估

    Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation

    [https://arxiv.org/abs/2611.25361](https://arxiv.org/abs/2611.25361)

    本文抽象出四阶段的论证增强工作流程，发现现有方法在不同任务上提升不一致，并据此提出了Qnt-Arg等两种面向特定任务的方法，以改进LLM在辩论决策与事实核查中的表现。

    

    论证框架提供了显式的、形式化的机制来表示和化解冲突，因此已被应用于大语言模型的决策中。其收益各不相同，取决于论据及其关系如何被构建、评估以及如何融入决策过程。我们基于现有的论证增强方法抽象出一个包含四个阶段的工作流程，并在基于辩论的决策任务和基于检索的事实核查任务上研究其设计选择。我们使用近期的大语言模型进行评估，结果显示现有方法在所考虑的任务中并不能一致地提升性能。基于这一观察，我们开发了两种面向特定任务的方法。第一种称为Qnt-Arg，它围绕辩论材料中的原始发言者来组织论据，并在渐进语义下聚合模型估计的论据强度和候选答案先验……

    arXiv:2611.25361v1 Announce Type: cross  Abstract: Argumentation frameworks (AFs) provide explicit, formal mechanisms for representing and resolving conflicts and thus have been incorporated into large language model (LLM) decision-making. The benefits vary and depend on how arguments and their relations are constructed, evaluated, and incorporated into the decision process. {We abstract a workflow consisting of four stages based on existing argumentation-augmented methods} and investigate their design choices over tasks of debate-based decision-making and retrieval-based fact verification. Our evaluation with recent LLMs shows that existing methods do not consistently improve performance across the considered tasks. Motivated by this observation, we develop two task-specific methods. The first is called Qnt-Arg, which organizes arguments around the original speakers in debate materials and aggregates model-estimated argument strengths and candidate-answer priors under gradual semantic
    
[^196]: AppSci-Bench：大语言模型能否构建AI智能体并使其变得更好？

    AppSci-Bench: Can LLMs Build AI Agents and Make Them Better?

    [https://arxiv.org/abs/2611.25360](https://arxiv.org/abs/2611.25360)

    本文提出AppSci-Bench基准，用于评估大语言模型像人类应用科学家一样从需求出发构建并科学改进AI智能体的能力，实验表明该任务对当前最前沿的模型仍极具挑战性。

    

    大语言模型（LLM）不仅被越来越多地用于驱动AI智能体，还被用于构建它们。虽然已有许多基准测试用于评估LLM驱动的智能体的质量，但很少有基准测试能衡量LLM在构建智能体方面的能力。为此，我们推出了AppSci-Bench，该基准构建了一个类似于人类应用科学家为企业构建智能体的工作场景——通过从利益相关者那里获取任务需求、探索企业的基础设施，并通过设计实验和内部基准来科学地改进目标智能体。AppSci-Bench包含七个真实领域的任务——其中五个是新领域（幻灯片生成、订单到收款、合规审查、客户通话洞察和CRM洞察），另外两个是从tau3-bench（Yao等人，2024；Shi等人，2026b）重新改造而来的领域（航空和银行）——每个领域都配有针对该任务定制的保留评估套件。我们评估了七个前沿模型，发现AppSci-Bench极具挑战性：表现最好的模型……（原文摘要在此处截断）

    arXiv:2611.25360v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used not only to power AI agents but also to build them. While many benchmarks assess the quality of LLM- powered agents, few ascertain how good LLMs are at building them. To this end, we introduce AppSci-Bench, where we construct a setting akin to how human Applied Scientists build agents for businesses - through eliciting task requirements from stakeholders, exploring the business's infrastructure, and scientifically improve the target agent by designing experiments and internal benchmarks. AppSci-Bench features seven realistic domains - five new ones (Slides Generation, Order-to-Cash, Compliance Review, Customer Call Insights, and CRM Insights) and two domains recast from tau3-bench (Yao et al., 2024; Shi et al., 2026b)(Airline and Banking) - each with a held-out evaluation suite customized for that task. We evaluate seven frontier models and find AppSci-Bench challenging: the best model, 
    
[^197]: Pro$^2$ Agent：面向自主AI研发的科学问题探测方法

    Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D

    [https://arxiv.org/abs/2611.25357](https://arxiv.org/abs/2611.25357)

    Pro$^2$ Agent将自主AI研究的核心状态从代码库转变为显式的问题状态，通过全面分析、定向研究和受控假设检验来识别阻碍成功的问题并匹配候选解决方案，从而避免智能体在困难问题上陷入错误的解区域。

    

    大语言模型（LLM）智能体现在能够通过编写代码和运行实验来自主开展AI研究。它们通常采用迭代优化代码库的方式进行，却忽视了任务背后的问题结构。这种模式在较简单的任务上是有效的，但在困难问题上，智能体由于从未显式地表达其对问题的理解，往往会陷入错误的解区域。我们提出了Pro$^2$ Agent，它将自主AI研究的主要状态从代码库状态转变为问题状态：这是一种显式的结构，用于刻画哪些问题阻碍了成功，以及哪些解决方案有可能解决这些问题。Pro$^2$ Agent通过对可用资源和约束条件的全面分析来初始化问题状态，随后进行有针对性的研究，将识别出的问题与候选解决方案关联起来。通过受控的假设检验，智能体会更新可疑（摘要原文在此处被截断）

    arXiv:2611.25357v1 Announce Type: new  Abstract: Large language model (LLM) agents can now autonomously conduct AI research by writing code and running experiments. They typically do so by iteratively optimizing a codebase, while ignoring the problem structure behind the task. This formulation is effective on simpler tasks, but on difficult problems it often gets stuck in the wrong solution region because the agent never explicitly represents its understanding of the problem. We introduce Pro$^2$ Agent, which shifts the primary state of autonomous AI research from a codebase state to a Problem State: an explicit structure of what problems prevent success and which solutions are likely to address them. Pro$^2$ Agent initializes the Problem State through comprehensive analysis of available resources and constraints. It then performs targeted research to relate the identified problems to candidate solutions. Through controlled hypothesis testing, the agent updates the importance of a susp
    
[^198]: FPGA-GEN：通过高层次综合实现目标感知的RTL生成

    FPGA-GEN: Target-Aware RTL Generation via HLS

    [https://arxiv.org/abs/2611.25351](https://arxiv.org/abs/2611.25351)

    FPGA-GEN是一种代理式HLS-C生成工作流，通过显式任务契约和有界工具反馈循环实现目标感知的RTL生成，在HLS-Eval上达到95.8%的测试通过率，并生成出比基线方案减少73.7%综合周期和14.6% LUT的高效FPGA硬件设计。

    

    成功的C/C++编译并不能保证经过高层次综合（HLS）后得到正确或高效的FPGA硬件。FPGA-GEN是一种代理式HLS-C生成工作流，它将规范、就绪性、选择、修复和准入规则编码在显式的任务契约和有界工具反馈循环中，用于候选层面的决策。在HLS-Eval基准上，每个任务生成五个共享上下文的输出，获得了407/425（95.8%）的公开测试平台通过率。在一项累积转换活动中，FPGA-GEN在保留集协同仿真、Vivado综合/布局布线以及相同RTL证据的验证下，接纳了104/107（97.2%）的HLS-LeVeri设计。在针对TensorCore的筛选中，选定的候选方案3比生成的/组合的C0基准减少了73.7%的CSynth周期和14.6%的LUT，同时DSP增加了26.7%，并通过了开发阶段的协同仿真验证。

    arXiv:2611.25351v1 Announce Type: cross  Abstract: Successful C/C++ compilation does not guarantee correct or efficient FPGA hardware after high-level synthesis (HLS). FPGA-GEN is an agentic HLS-C generation workflow that encodes specification, readiness, selection, repair, and admission rules in explicit task contracts and bounded tool-feedback loops for candidate-level decisions. On HLS-Eval, five shared-context outputs per task yield 407/425 (95.8%) public-testbench passes. In a cumulative conversion campaign, FPGA-GEN admits 104/107 (97.2%) HLS-LeVeri designs under held-out co-simulation, Vivado synthesis/fit, and same-RTL evidence. In focused TensorCore screening, selected candidate 3 reports 73.7% fewer CSynth cycles and 14.6% fewer LUTs than generated/composed C0, with 26.7% more DSPs, and passes development CoSim.
    
[^199]: 借助LLM智能体从FIPS标准到FPGA实现：度量后量子密码硬件中测试与评审所遗漏的问题

    From FIPS to FPGA with LLM Agents: Measuring What Tests and Reviews Miss in Post-Quantum Cryptographic Hardware

    [https://arxiv.org/abs/2611.25339](https://arxiv.org/abs/2611.25339)

    该论文提出了一个针对LLM智能体生成硬件代码的七阶段验证门控工作流，成功将ML-KEM-768/ML-DSA-65后量子密码加速器移植到FPGA上并通过全部NIST测试，同时通过突变测试揭示现有测试和评审仅能发现76%的硬件缺陷，证明自生成的验证证据不足以保证硬件可信。

    

    随着LLM智能体日益同时生成RTL代码以及用于验证它的测试和评审，自生成的证据成为可信硬件的薄弱基础。我们提出了在一次后量子设计移植过程中演化出的七阶段验证门控工作流程，其中只有当某个门控被证明能够在阴性对照和故障突变体上失败时才被视为有效，并且评审智能体会在提交前对工作进行审计。随后，我们使用工具选择的网表突变体、缺陷重放和形式化检查来审计这些证据。该流程将ML-KEM-768/ML-DSA-65加速器移植到300 MHz的Zynq UltraScale+密钥托管卡上：所有六种操作在板上均实现字节级精确，并通过了全部85个接口适用的NIST ACVP测试用例（共285个）以及484个ML-DSA已知答案测试。然而，通过的模块测试夸大了验证强度。在三个RTL模块中，现有仿真门控仅能杀死399个工具选择的网表突变体中的76%……

    arXiv:2611.25339v1 Announce Type: cross  Abstract: As LLM agents increasingly generate RTL alongside the tests and reviews that validate it, self-generated evidence is a weak foundation for trustworthy hardware. We present our seven-stage verification-gated workflow, evolved during a post-quantum design port, in which a gate counts only if shown able to fail on negative controls and fault mutants, and reviewer agents audit work before commit. We then audit this evidence with tool-selected netlist mutants, flaw replays, and formal checks. The flow ported an ML-KEM-768/ML-DSA-65 accelerator to a 300 MHz Zynq UltraScale+ key-custody card: all six operations are byte-exact on the board, and it passes all 85 interface-applicable NIST ACVP cases (of 285) and 484 ML-DSA known answers. However, passing module tests overstate the verification strength. Across three RTL modules, existing simulation gates kill only 76% of 399 tool-selected netlist mutants; specification-derived tests guided by th
    
[^200]: 通过噪声对齐的视图生成打破自监督学习中的噪声捷径

    Breaking Noise Shortcuts in Self-Supervised Learning Via Noise-Aligned View Generation

    [https://arxiv.org/abs/2611.25335](https://arxiv.org/abs/2611.25335)

    该论文提出SEANA即插即用模块，通过从含噪数据中学习可逆方差稳定变换并重新采样独立噪声来生成噪声感知的自监督学习视图，从而打破噪声捷径，在无需噪声模型知识、干净数据和去噪器的情况下实现噪声鲁棒的表征学习。

    

    自监督学习（SSL）越来越多地应用于科学成像领域，例如显微镜成像，在这些领域中标签稀缺而含噪数据丰富。这些领域存在显著的信号依赖性噪声，这可能形成一种表征捷径：由于两个增强视图都源自同一张含噪图像，编码器可以通过同时编码噪声模式而非语义内容来提升视图一致性。为了在不掌握底层噪声模型的前提下实现噪声鲁棒的表征学习，我们提出了自对齐噪声增强（Self-Aligned Noise Augmentation, SEANA），这是一个即插即用模块，用于生成噪声感知的SSL视图。SEANA从含噪数据中学习一个可逆的方差稳定变换（VST），然后估计干净信号并在学习到的VST空间中重新采样独立噪声，整个过程无需干净目标数据、无需针对特定数据集训练去噪器或在推理时使用去噪器，也无需更改SSL目标函数或编码器。在CIFAR-10和ImageNe[t]……（原文摘要被截断）

    arXiv:2611.25335v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) is increasingly applied to scientific imaging domains, such as microscopy, where labels are scarce but noisy data is abundant. These domains exhibit significant signal-dependent noise, which can create a representational shortcut: since both augmented views derive from the same noisy image, the encoder can boost view agreement by also encoding noise patterns rather than semantic content. To enable noise-robust representation learning without knowledge of underlying noise models, we introduce Self-Aligned Noise Augmentation (SEANA), a drop-in module that generates noise-aware SSL views. SEANA learns an invertible variance-stabilizing transform (VST) from noisy data, then estimates the clean signal and resamples independent noise in learned VST space, requiring no clean targets, no dataset-specific denoiser training or use at inference, and no changes to the SSL objective or encoder. On CIFAR-10 and ImageNe
    
[^201]: ReLay：面向可复用且工艺可移植的模拟设计的双向版图-代码翻译与库演化

    ReLay: Bidirectional Layout-Code Translation and Library Evolution for Reusable and Process-Portable Analog Design

    [https://arxiv.org/abs/2611.25328](https://arxiv.org/abs/2611.25328)

    提出了双向版图-代码转换框架ReLay，可从现有版图自动恢复生成器程序并让函数库随版图积累而演化，同时发布了首个版图-代码基准数据集，实现模拟设计的复用与跨工艺节点移植。

    

    模拟版图生成器为设计复用和工艺迁移提供了可编程接口，但其覆盖范围受限于手工构建的库。完整版图中蕴含的设计师知识仍然隐含在其几何结构中，难以被捕获到可复用的生成器库中。我们提出了ReLay，一个双向的版图-代码框架，能够从现有版图中恢复生成器程序，实现跨PDK的可移植性以及现有设计的参数化尺寸调整。在恢复的程序中反复出现的构造会被提升为库函数，因此库随着版图的积累而不断演化。我们还引入了首个版图-代码基准测试：包含来自三个开源PDK、27个电路系列的1,050个手工绘制版图，范围从基本单元到完整芯片。学习到的函数在保留测试版图中占版图构建调用的53-66%，展示了超越学习语料库的复用能力。

    arXiv:2611.25328v1 Announce Type: cross  Abstract: Analog layout generators provide a programmable interface for design reuse and process migration, but their coverage is limited by handcrafted libraries. The designer knowledge embedded in complete layouts remains implicit in their geometry, making it difficult to capture in reusable generator libraries. We present ReLay, a bidirectional layout-code framework that recovers generator programs from existing layouts, enabling cross-PDK portability and parametric resizing of existing designs. Constructions that recur across recovered programs are promoted to library functions, so the library evolves as layouts accumulate. We also introduce the first layout-code benchmark: 1,050 hand-drawn layouts from 27 circuit families across three open-source PDKs, ranging from primitive cells to full chips. Learned functions account for 53-66% of layout-building calls in held-out layouts, demonstrating reuse beyond the learning corpus. Frontier LLM age
    
[^202]: 潘多拉的工具箱：为什么智能体的安全性无法被验证

    Pandora's Toolbox Or Why Agent Safety Cannot Be Verified

    [https://arxiv.org/abs/2611.25325](https://arxiv.org/abs/2611.25325)

    本文证明将语言模型与通用计算设备耦合构建的AI智能体，其安全性在理论上是不可判定的，即使施加有限约束后判定问题也是PSPACE完全的（计算上不可行），且安全性不具有可组合性，因此所有防护栏、黑白名单和认证等安全机制在原理上都只能是近似的解决方案。

    

    当前的人工智能智能体是通过将固定的语言模型与通用计算设备相耦合的方式构建的。我们证明这种耦合使得安全性问题不可判定。特别地，我们证明给定某个危险程序，没有任何算法能够预测智能体是否会运行一个与之语义等价的程序。一般而言，防护栏、黑名单、白名单和认证等手段只能带来不完善的安全解决方案。在有限约束（有界的上下文、取值和运行时间）下，安全性是可判定的，但仍然是难以计算的。具体而言，即使智能体的描述是已知的，该问题也是PSPACE完全的。最后，安全性不是一种可组合的性质，即各自可被证明安全的智能体在组合在一起时仍可能产生不安全的行为。我们还讨论了可在实践中使用的放宽的安全性概念。

    arXiv:2611.25325v1 Announce Type: new  Abstract: Present artificial intelligence agents are constructed by means of a coupling between a fixed language model with a general computing device. We show that this coupling makes safety undecidable. In particular, we show that given some dangerous program, no algorithm can predict whether an agent will run such a program that is semantically equivalent. In general, guardrails, blacklists, whitelists and certifications can only lead to imperfect safety solutions. Under finite constraints (bounded context, values and running time) safety is decidable but still intractable. Specifically, the problem is $\mathsf{PSPACE}$-complete even when the description of the agent is known. Finally, safety is not a compositional property, i.e., agents that can be proven safe on their own can still produce unsafe behavior when grouped together. We also discuss relaxed notions of safety that could be used in practice.
    
[^203]: 大语言模型已经是具备置信度感知的零样本分类器

    LLMs Are Already Confidence-Aware Zero-Shot Classifiers

    [https://arxiv.org/abs/2611.25289](https://arxiv.org/abs/2611.25289)

    研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。

    

    零样本大语言模型越来越多地被用作文本分类器，但它们的标签只有在知道能在多大程度上信任它们时才有用。我们比较了黑盒方法（言语化置信度和采样）与读取token概率的白盒方法（P(True)以及使用受限生成的多种logit评分方法）。我们在五个包含3到151个标签的分类数据集上评估了11个开放权重大语言模型（1B-32B参数），测量分类性能（准确率、F1-macro）、判别能力（AUROC）和校准（ECE和Brier分数）。置信度方法的选择很少显著影响准确率，但基于logit的评分方法比黑盒方法（AUROC 0.57-0.66）带来了更好的判别能力（AUROC 0.74-0.77）。用单token标识符替换标签可以保持判别能力，但平均损失3-6个准确率点。我们还引入了仅决策评分方法，该方法（摘要在此处截断）

    arXiv:2611.25289v1 Announce Type: cross  Abstract: Zero-shot large language models (LLMs) are increasingly used as text classifiers, but their labels are only useful when we know to what extent we can trust them. We compare black-box methods (verbalised confidence and sampling) with white-box methods that read token probabilities (P(True), and multiple logit-scoring methods using constrained generation). We evaluate eleven open-weight LLMs (1B-32B parameters) on five classification datasets with 3 to 151 labels, measuring classification performance (accuracy, F1-macro), discrimination (AUROC), and calibration (ECE and Brier score). The choice of confidence method rarely impacts accuracy significantly, but logit-based scoring methods lead to much better discrimination (AUROC 0.74-0.77) than black-box methods (AUROC 0.57-0.66). Replacing labels with single-token identifiers preserves discrimination but costs 3-6 accuracy points on average. We also introduce decision-only scoring, which o
    
[^204]: 面向大规模推荐模型的智能嵌入剪枝

    Intelligent Embedding Pruning for Large-Scale Recommendation Models

    [https://arxiv.org/abs/2611.25288](https://arxiv.org/abs/2611.25288)

    该论文提出利用特征重要性来指导训练中嵌入剪枝的各表剪枝比例（FI-ITEP），在三个大规模推荐模型上实现 20%–40% 的内存节省并保持或提升准确率，同时显著减少训练所需的 GPU 数量。

    

    现代推荐系统面临巨大的内存压力，这主要源于将高基数类别特征映射为稠密向量的稀疏嵌入层。训练中嵌入剪枝（ITEP）在训练过程中移除不那么重要的嵌入行，从而带来端到端的内存节省，但由于各表的剪枝比例是由基于规模的启发式方法设定的，有时会降低模型准确率。我们利用特征重要性（FI）来指导这些剪枝比例（FI-ITEP），对与较不重要特征相关联的表分配更高的剪枝比例。在三个大规模推荐模型上，FI-ITEP 相比基于百分位数的 ITEP 实现了 20%–40% 的总内存减少，同时保持或提升了准确率。由于特征重要性信号的计算所需资源极少，这些内存节省转化为具体的系统收益：相同的模型可以在更少的 GPU 训练节点上完成训练（例如从 112 个减少到 80 个），且每个 GPU 的吞吐量更高，节省了 1000+ GPU 资源。

    arXiv:2611.25288v1 Announce Type: cross  Abstract: Modern recommendation systems face substantial memory pressure, largely from the sparse embedding layer that maps high-cardinality categorical features into dense vectors. In-training embedding pruning (ITEP) removes less important embedding rows during training, yielding end-to-end memory savings but sometimes degrading accuracy because per-table pruning ratios are set by size-based heuristics. We use feature importance (FI) to guide these ratios (FI-ITEP), assigning higher pruning to tables tied to less important features. Across three large-scale recommendation models, FI-ITEP achieves 20--40% total memory reduction while matching or improving accuracy relative to percentile-based ITEP. Because the feature-importance signal requires minimal resources to compute, these memory savings translate into concrete systems benefits: the same models train on fewer GPU trainers (e.g., 112 to 80) with higher per-GPU throughput, saving 1,000+ GP
    
[^205]: MS-Point：面向点云理解的多模态分层谱学习

    MS-Point: Multi-Modal Hierarchical Spectral Learning for Point Cloud Understanding

    [https://arxiv.org/abs/2611.25282](https://arxiv.org/abs/2611.25282)

    MS-Point提出了一种跨模态预训练框架，通过在池化前直接监督中间点特征及其邻域响应，结合分支级对齐、跨层一致性与自适应融合，突破了传统点-图像对齐中不同局部模式产生相同池化描述符的局限，并兼容图编码器与点Transformer等多种骨干架构。

    

    点-图像对齐能够学习到有用的物体表示，但不同的局部特征模式可能产生相同的池化描述符，从而接收到相同的对齐信号。我们提出了MS-Point，这是一个跨模态预训练框架，可直接监督中间点特征及其邻域响应。在选定的编码器阶段，我们在池化之前计算邻域平均值和残差，并将其分支描述符与图像特征进行对齐。分支级目标使每个表示与其视觉目标对齐，同时跨层一致性和自适应融合协调各分支间的信息。中间特征与空间邻域的通用接口同时支持图编码器和点Transformer。在物体分类、少样本适应、密集预测和低标注场景学习等多项任务上的评估证明了该设计的迁移价值。相同骨干网络下的对比以及容量相关（摘要在此处截断）……

    arXiv:2611.25282v1 Announce Type: new  Abstract: Point-image alignment learns useful object representations, but different local feature patterns can produce the same pooled descriptor and receive the same alignment signal. We introduce MS-Point, a cross-modal pretraining framework that directly supervises intermediate point features and their neighborhood responses. At selected encoder stages, we compute neighborhood averages and residuals before pooling and align their branch descriptors with image features. Branch-level objectives align each representation with its visual target, while cross-level consistency and adaptive fusion coordinate information across branches. A common interface of intermediate features and spatial neighborhoods supports both graph encoders and point Transformers. Evaluations across object classification, few-shot adaptation, dense prediction, and low-label scene learning demonstrate the transfer value of this design. Same-backbone comparisons and capacity-m
    
[^206]: 当目标与评估者不一致时：评估事后人口框架偏差缓解的迁移性

    When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation

    [https://arxiv.org/abs/2611.25258](https://arxiv.org/abs/2611.25258)

    该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。

    

    在源模型不可访问的情况下，事后重写被用于减少大语言模型生成文本中的人口群体差异。一次重写可能仅仅因为改善了自己的目标或满足了自己的选择器而显得成功。我们将其作为一个迁移问题进行研究，研究对象是660个性别对和1,980个年龄对比的人口条件化气候信息。重写器提出编辑，选择模型在满足约束条件的候选中进行挑选，而一个独立于两者的评估器对成对框架差异进行评分。总是针对说服偏差指数（PBI）较低的信息的固定策略，在不同维度和结果聚合方式之间的迁移表现不一致。一种在两侧都提出编辑但最多接受一个编辑的自适应策略，在应用于相同候选时，在两个评估器家族的八项比较中有七项显著优于该规则，其优势主要来自编辑了更多的信息对，而非引导编辑方向的选择。

    arXiv:2611.25258v1 Announce Type: cross  Abstract: Post-hoc rewriting is used to reduce demographic disparities in LLM-generated text when the source model is inaccessible. A rewrite can look successful merely because it improved its own target or satisfied its selector. We study this as a transfer problem on 660 gender pairs and 1,980 age contrasts of demographic-conditioned climate messages. A rewriter proposes edits, a selection model chooses among constraint-satisfying candidates, and an independent evaluator uninvolved in either rates pairwise framing disparity. A fixed policy that always targets the message with the lower Persuasion Bias Index (PBI) transfers inconsistently across axes and outcome aggregations. An adaptive policy that proposes edits on both sides but accepts at most one significantly improves on that rule, applied to the same candidates, in seven of eight comparisons across two evaluator families, mostly by editing more pairs, not by guided direction choice, at h
    
[^207]: Keel：一个前沿模型通过重写权重来教导视觉-语言模型

    Keel: A Frontier Model Teaches a Vision-Language Model by Rewriting Its Weights

    [https://arxiv.org/abs/2611.25257](https://arxiv.org/abs/2611.25257)

    Keel提出首个面向VLM图像任务的代理式权重编辑方法，让前沿多模态大语言模型作为代理，仅凭症状描述即可定位、修改并验证模型权重，无需训练就能修复小型推理VLM在思考模式下的循环失败，且不损害未见问题上的准确率。

    

    前沿模型能否通过重写权重来修复一个视觉-语言模型（VLM）？诸如Qwen3.5这样的小型推理VLM正需要这样的修复：在思考模式下，它们会在许多图像问题上陷入循环，不断重复自己直至达到token上限，耗尽全部预算并丢失答案。通常的补救措施只作用于解码过程，让陷入循环的模型保持原样，或者必须被告知该修改什么。然而，病因就写在权重之中。Keel是首个面向图像任务的VLM代理式权重编辑方法，它直接在权重中找到并消除病因。只需被告知症状，一个编辑器——作为代理运行的前沿多模态大语言模型——便会定位病因、编辑权重并进行测试，失败普查和检测工具会向它展示失败的图像以及VLM的注意力所在；在多个编辑器之中，该框架只保留那些能明显减少失败、且不会降低编辑器未见问题上准确率的修改。这种修复无需任何训练，并能保持……（原文摘要在此处被截断）

    arXiv:2611.25257v1 Announce Type: cross  Abstract: Can a frontier model repair a vision-language model (VLM) by rewriting its weights? Small reasoning VLMs such as Qwen3.5 need such a repair: in thinking mode they fall into loops on many image questions, repeating themselves until the token limit, spending the whole budget and losing the answer. The usual remedies act on the decoding, leaving the looping model as it was, or must be told what to change. Yet the cause is written in the weights. Keel, the first agentic weight-editing method for VLMs on image tasks, finds and removes it there. Told only the symptom, an editor, a frontier multimodal LLM acting as an agent, locates the cause, edits the weights and tests them, with a failure census and instruments showing it the failing images and where the VLM attends; among several editors, the harness keeps only edits that clearly reduce the failure without lowering accuracy on questions no editor saw. The repair needs no training, keeps t
    
[^208]: 规划你的自适应！基于本体的机器人任务与架构协同自适应规划

    Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots

    [https://arxiv.org/abs/2611.25254](https://arxiv.org/abs/2611.25254)

    本文提出PLANTA方法，利用OWL本体表示架构备选方案、PDDL定义任务目标，实现了机器人任务规划与架构重构的联合协同规划，并能根据执行中的变化进行重规划。

    

    自主机器人可能需要在运行条件发生变化时，同时调整其任务规划和架构配置。这些决策是相互依赖的：架构配置决定了机器人能够执行哪些动作，而任务规划则决定了需要哪些能力。本文提出了PLANTA，一种基于本体的任务与架构协同自适应方法，能够在任务规划过程中考虑这些依赖关系。PLANTA使用以Web本体语言（OWL）表达的本体来表示架构备选方案及其相关的功能和质量需求，同时使用规划领域定义语言（PDDL）来定义任务动作和任务目标。通过自动翻译和集成这些模型，PLANTA实现了任务动作与重构动作的联合规划。该方法采用满意规划技术，并在执行过程中发生变更导致当前计划失效时进行重新规划。PLANTA的评估（摘要在此处截断）

    arXiv:2611.25254v1 Announce Type: cross  Abstract: Autonomous robots may need to adapt both their task plans and architectural configurations as operating conditions change. These decisions are interdependent: architectural configurations determine which actions a robot can perform, while task plans determine which capabilities are required. This paper presents PLANTA, an ontology-mediated approach to task and architecture co-adaptation that considers these dependencies during mission planning. PLANTA uses an ontology expressed in the Web Ontology Language to represent architectural alternatives and their associated functional and quality requirements, while the Planning Domain Definition Language defines task actions and mission goals. By automatically translating and integrating these models, PLANTA enables joint planning of task and reconfiguration actions. The approach uses satisficing planning and replans when changes during execution invalidate the current plan. PLANTA is evaluat
    
[^209]: 超越图像复原：面向真实世界隐蔽目标分割的内容-退化因子展开

    Beyond Restoration: Content-Degradation Factor Unfolding for Real-World Concealed Object Segmentation

    [https://arxiv.org/abs/2611.25239](https://arxiv.org/abs/2611.25239)

    该论文提出因子展开网络（FUN），将Retinex模型扩展为具有任务非对称因子角色的内容-退化因子分解框架，以内容因子作为唯一语义证据来源，从退化变化中提取可靠内容证据，从而实现真实世界隐蔽目标分割。

    

    在退化条件下的可靠视觉感知需要将任务相关内容与退化引起的变化区分开来，尤其是在隐蔽目标分割（COS）任务中，前景线索本身就很微弱。核心挑战不仅在于提升可见性，更在于在退化引起的变化中识别可靠的内容证据。RGB空间中的分割方法往往将内容线索与退化线索纠缠在一起，而基于复原的方法则主要关注图像恢复与任务交互。这促使我们进行显式的因子分解，以区分内容证据与退化效应并评估其可靠性。Retinex理论为这种分离提供了空间对齐的基础，但其仍然以光照为中心。因此，我们将其扩展为具有任务非对称因子角色的内容-退化成像模型。因子展开网络（FUN）使用内容因子R作为分割中唯一的直接语义证据来源。

    arXiv:2611.25239v1 Announce Type: cross  Abstract: Reliable visual perception under degradation requires distinguishing task-relevant content from degradation-induced variations, especially in concealed object segmentation (COS), where foreground cues are inherently weak. The central challenge is not merely improving visibility, but identifying reliable content evidence amid degradation-induced variations. RGB-space segmentation often entangles content and degradation cues, while restoration-based methods primarily focus on image recovery and task interaction. This motivates explicit factorization to distinguish content evidence from degradation effects and assess its reliability. Retinex provides a spatially aligned basis for this separation but remains illumination-centric. We therefore extend it to content-degradation formation with task-asymmetric factor roles. The Factor Unfolding Network (FUN) uses the content factor R as the sole source of direct semantic evidence for segmentati
    
[^210]: 一种用于30天糖尿病再入院预测的神经符号多模态框架：整合结构化电子健康记录数据、密集临床嵌入与知识支撑的符号抽取

    A Neurosymbolic Multimodal Framework for 30-Day Diabetic Readmission Prediction: Integrating Structured EHR Data, Dense Clinical Embeddings, and Knowledge-Grounded Symbolic Extraction

    [https://arxiv.org/abs/2611.25236](https://arxiv.org/abs/2611.25236)

    该论文提出一种神经符号多模态框架，将结构化EHR特征、Bio_ClinicalBERT临床笔记密集嵌入与基于三阶段混合检索增强生成流水线抽取的符号实体相融合，以更准确地预测糖尿病患者的非计划30天再入院风险。

    

    糖尿病患者的非计划性30天再入院会带来沉重的临床与经济负担，然而大多数预测模型仅使用结构化电子健康记录（EHR）数据，而忽略了自由文本临床笔记——这些笔记中常包含印地语-英语混杂（Hinglish）的表达以及否定结构，通用语言模型难以对其进行可靠解析。我们提出了一种神经符号多模态框架，并在公开的糖尿病130家美国医院数据集（101,766次就诊；71,518名患者）上进行了评估，采用患者级别的分组80/20划分。该框架融合了三种数据流：人工设计的结构化特征；通过截断奇异值分解获得的Bio_ClinicalBERT临床笔记嵌入的32个分量；以及由开放权重的临床语言模型在三阶段混合检索增强生成流水线（密集检索、稀疏词法检索与交叉编码器重排序）中抽取的符号实体，并带有反思性……（摘要至此处被截断）

    arXiv:2611.25236v1 Announce Type: new  Abstract: Unplanned 30-day readmissions among patients with diabetes mellitus impose a substantial clinical and economic burden, yet most predictive models use only structured electronic health record (EHR) data and ignore free-text clinical notes, which often contain code-mixed Hindi-English (Hinglish) phrasing and negation that general-purpose language models parse unreliably. We propose a neurosymbolic multimodal framework evaluated on the public Diabetes 130-US Hospitals dataset (101,766 encounters; 71,518 patients) with a patient-level grouped 80/20 split. Three streams are fused: engineered structured features; 32 components of Bio_ClinicalBERT note embeddings obtained by truncated singular value decomposition; and symbolic entities extracted by an open-weight clinical language model within a three-stage hybrid retrieval-augmented generation pipeline (dense retrieval, sparse lexical retrieval, and cross-encoder re-ranking) with reflective ne
    
[^211]: 有些记忆必须消逝：用户记忆在模型参数中的持久性与衰减

    Some Memories Must Fade: Persistence and Decay of User Memories in Model Parameters

    [https://arxiv.org/abs/2611.25235](https://arxiv.org/abs/2611.25235)

    该论文提出了一种参数化记忆管理系统，通过固定容量的LoRA适配器作为记忆基底，配合学习策略来管理知识的完整生命周期——保留稳定事实、强化重复观察、更新变化的信念，并让短暂信息自然消退。

    

    参数化记忆使大型语言模型能够将知识直接内化到权重中，为基于检索的记忆（RAG）提供了一种潜在强大的替代方案。然而，现有的参数化记忆方法难以部署，因为它们在很大程度上沿用了持续学习的目标，即无限期地保留信息。这对于记忆系统而言既不可持续也不充分，因为记忆系统必须管理知识的整个生命周期：保留稳定的事实、强化反复出现的观察、更新已经改变的信念，并允许短暂的信息逐渐消退。我们提出了一个参数化记忆管理系统，由记忆基底（负责信息的存储与遗忘）和学习策略（决定记住什么、以多深的程度记住、以及记住多久）组成。我们的基底是一个固定容量的LoRA适配器，由k-hot秩组合构成，能够在不产生灾难性干扰的情况下存储事实……

    arXiv:2611.25235v1 Announce Type: cross  Abstract: Parametric memory allows Large Language Models to internalize knowledge directly into their weights, offering a potentially powerful alternative to retrieval-based memory (RAG). However, existing approaches to parametric memory are difficult to deploy as they largely mirror the Continual Learning (CL) objective of retaining information indefinitely. This is neither sustainable nor sufficient for a memory system, which must govern knowledge across its lifecycle by retaining stable facts, reinforcing recurring observations, updating changed beliefs, and allowing transient information to fade. We propose a system for parametric memory management consisting of a memory substrate (handling information storage and forgetting) and a learned policy (what to remember, how deeply, and for how long). Our substrate is a fixed-capacity LoRA adapter that consists of k-hot rank combinations which store facts without catastrophic interference and main
    
[^212]: MKA：在端到端专业任务上评估多模态知识智能体

    MKA: Evaluating Multimodal Knowledge Agents on end-to-end professional tasks

    [https://arxiv.org/abs/2611.25228](https://arxiv.org/abs/2611.25228)

    MKA基准通过200个要求智能体从CAD蓝图、原理图、影像图等多模态视觉密集文件中提取整合信息并生成专业格式交付成果的任务，填补了现有基准无法评估端到端专业多模态知识工作能力的空白。

    

    我们提出了多模态知识智能体（MKA）基准，用于衡量自主智能体在依赖视觉理解的端到端专业知识工作中的表现。每个任务要求智能体从多个视觉密集型文件（如CAD蓝图、电子原理图、放射影像、复杂图表和地理空间地图）中提取并整合信息，并生成独立的交付成果，格式包括.xlsx、.pptx、.pdf、.docx、.html或.png。现有基准基本未测试这一能力：移除图像查看工具并不会改变Gemini 4 Carbon在GDPval上的得分，而诸如GDP.pdf和GDP.xlsx这类单一格式评估也不需要跨文件或跨模态的推理。MKA包含涵盖10个经济领域的200个任务。交付成果由沙盒化的智能体自动评分器根据专家加权的评分标准进行评分，该评分器在实验中与人类领域专家在96.2%的评判标准判断上保持一致。

    arXiv:2611.25228v1 Announce Type: new  Abstract: We introduce the Multimodal Knowledge Agent (MKA) benchmark, which measures how well autonomous agents perform end-to-end professional knowledge work that depends on visual understanding. Each task requires an agent to extract and combine information from multiple visually dense files, such as CAD blueprints, electronics schematics, radiographs, complex charts and geospatial maps, and to produce a standalone deliverable in .xlsx, .pptx, .pdf, .docx, .html or .png format. Existing benchmarks leave this capability largely untested: removing image-viewing tools leaves Gemini 4 Carbon's score on GDPval unchanged, and single-format evaluations such as GDP.pdf and GDP.xlsx do not require reasoning across files or modalities. MKA contains 200 tasks across 10 economic sectors. Deliverables are graded against expert-weighted rubrics by a sandboxed agentic autorater that agrees with human domain experts on 96.2% of criterion judgments in our exper
    
[^213]: 拒绝并报告：面向生成式图像模型的伪造显性不可检测水印

    Reject and Report: Forgery-Evident Undetectable Watermarks for Generative Image Models

    [https://arxiv.org/abs/2611.25226](https://arxiv.org/abs/2611.25226)

    该论文提出一种“拒绝并报告”式的伪造显性不可检测水印方案，使生成图像模型的水印不仅能抵御伪造攻击，还能在检测时主动识别并报告伪造尝试，弥补了现有方案（如SemBind）在拒绝伪造与报告伪造两方面的不足。

    

    基于内容水印等溯源方案可以让生成模型产生的图像被追溯回该模型。一个实用的水印方案必须在低误报率（FPR）下保持准确、在内容保持型处理下具有鲁棒性，并且不可检测。此外，它还必须抵御伪造；否则，即使仅凭一张带水印的图像和一个替代模型，攻击者也能将水印印到任意图像上，或重新生成携带该水印的新内容，从而将有害内容错误地归因于某一模型，或让真实照片看起来像是AI生成的。我们认为，水印方案不仅应当拒绝伪造，还应当报告伪造：因为一个被判定为“无水印”而拒绝的伪造尝试不会留下任何痕迹。当前最先进的抗伪造不可检测水印方案（如SemBind）在这两方面都有欠缺；而能够报告伪造的外部异常检测器，要么漏检大部分伪造，要么错误地标记真实样本……

    arXiv:2611.25226v1 Announce Type: cross  Abstract: Provenance schemes such as those based on content watermarks let images from a generative model be traced back to it. A practical watermarking scheme must be accurate at a low false-positive rate (FPR), robust under content-preserving processing, and undetectable. In addition, it must resist forgery; otherwise, even from a single watermarked image and a surrogate model, an adversary can imprint the watermark onto an arbitrary image or regenerate new content that carries it, falsely attributing harmful content to a model or making authentic photographs appear AI-generated. We argue that a watermarking scheme should not only reject forgeries but also report them: a forgery rejected as unwatermarked leaves no trace of the attempt. State-of-the-art forgery-resilient undetectable watermarking schemes, such as SemBind, fall short on both counts. External anomaly detectors that do report forgeries either miss most of them or falsely flag genu
    
[^214]: 桥接视觉-语言推理与学习控制以实现野外足式机器人导航

    Bridging vision-language reasoning and learned control for legged navigation in the wild

    [https://arxiv.org/abs/2611.25215](https://arxiv.org/abs/2611.25215)

    该论文提出ROAM系统，将视觉-语言模型的常识推理能力与强化学习习得的运动控制和局部规划策略相结合，使足式机器人能够在高山和城市等复杂真实环境中实现公里级自主导航。

    

    在多样化真实环境中的自主导航仍然是机器人领域最大的挑战之一，它既需要鲁棒的底层运动控制来应对复杂地形，又需要高层推理能力来解读环境线索。我们提出了ROAM（Robust Open-world Autonomous Mobility，鲁棒开放世界自主移动）系统，该系统能够使足式机器人在复杂的高山和城市场景中完成公里级导航。ROAM将基于互联网规模数据预训练的视觉-语言模型（VLM）所具备的世界知识，与通过强化学习（RL）在仿真中学习到的鲁棒运动控制和局部规划策略相结合。VLM提供常识推理和场景语义解读，能够评估风险、解读标识牌，并生成带有相应导航代价的语义标签。一个开放词汇分割模块将这些标签转换为稠密的实时风险地图，用以调节两个预训练的强化学习策略。

    arXiv:2611.25215v1 Announce Type: cross  Abstract: Autonomous navigation in diverse real-world environments remains one of the greatest challenges in robotics. It demands both robust low-level motion control to handle diverse terrain and high-level reasoning to interpret environmental cues. We present ROAM (Robust Open-world Autonomous Mobility), a system that enables legged robots to perform kilometer-scale navigation in complex alpine and urban settings. ROAM combines the world knowledge of a Vision-Language Model (VLM), pre-trained on internet-scale data, with robust locomotion and local planning policies learned in simulation via reinforcement learning (RL). The VLM provides common-sense reasoning and semantic interpretation of the scene, assessing risks, interpreting signs, and producing semantic labels with associated navigation costs. An open-vocabulary segmentation module converts these labels into a dense, real-time risk map, which conditions two pre-trained RL policies respon
    
[^215]: 通过审计式认知探索引导自动化科学发现

    Steering Automated Scientific Discovery via Audited Epistemic Exploration

    [https://arxiv.org/abs/2611.25214](https://arxiv.org/abs/2611.25214)

    该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。

    

    科学发现需要根据新证据重新审视假设，并识别已探索范围之外有前景的方向。然而，研究智能体常常做出其观察结果无法支持的主张，探索狭窄的假设空间，并且在意识到证据相互冲突时仍然保留原有信念。这些失败促使人们寻找一种显式的认知状态，以同时捕捉关于现有假设的信念和进一步探索的方向。为此，我们提出了 Epistemic Compass（EpiCompass），这是一种即插即用的脚手架，通过一棵连接假设、证据、信念与修订的认知树来维护这一状态。随着调查的推进，EpiCompass 会根据积累的证据重新评估已探索的假设，并利用未解决的冲突和未探索的方向来引导假设生成。它通过预期认知增益来选择候选假设，同时兼顾……（原文此处截断）

    arXiv:2611.25214v1 Announce Type: new  Abstract: Scientific discovery requires revisiting the hypotheses in light of new evidence and identifying promising directions beyond those already explored. However, research agents often make claims that are not supported by their observations, explore a narrow hypothesis space, and retain beliefs despite recognizing conflicting evidence. These failures motivate the search for an explicit epistemic state that captures both beliefs about existing hypotheses and directions for further exploration. To this end, we propose Epistemic Compass (EpiCompass), a plug-and-play scaffold that maintains this state through an epistemic tree linking hypotheses, evidence, beliefs, and revisions. As an investigation progresses, EpiCompass reassesses the hypotheses explored against accumulated evidence and uses unresolved conflicts and unexplored directions to guide hypothesis generation. It selects candidates by anticipated epistemic gain, accounting for both re
    
[^216]: 从FAIR到AWARE：地球科学中面向代理型人工智能的数据就绪性

    From FAIR to AWARE: Data Readiness for Agentic AI in the Earth Sciences

    [https://arxiv.org/abs/2611.25209](https://arxiv.org/abs/2611.25209)

    本文提出在FAIR原则之上扩展的AWARE数据就绪性需求层，为地球科学中自主行动的AI智能体提供专用数据标准与分阶段实施路径，以减少智能体的无声错误。

    

    地球科学中的人工智能开始具备自主行动能力：原型智能体已经在检索数据档案、运行模拟并起草分析。科学数据原本是为具有专业判断力的研究人员而整理的，不能假定自主智能体能够自行提供这种判断力。数据即使满足FAIR原则，仍可能误导单独行动的机器。我们提出地球科学数据应当成为AWARE（可操作Actionable、描述完善Well-described、可审计Auditable、配有操作指南Recipe-equipped、认识论上标注Epistemically labelled），即在FAIR基础上扩展的面向智能体的需求层，并围绕三部分议程展开：AWARE数据、有界工具、分层验证。这些要素已存在于社区基础设施之中。我们概述了一条分阶段、可衡量的采用路径，旨在减少无声错误。

    arXiv:2611.25209v1 Announce Type: cross  Abstract: Artificial intelligence in the Earth sciences is beginning to act: prototype agents already search archives, run simulations and draft analyses. Scientific data were curated for researchers who bring expert judgement; an autonomous agent cannot be assumed to supply it. Data can satisfy the FAIR principles and still mislead a machine acting alone. We propose that Earth-science data become AWARE (Actionable, Well-described, Auditable, Recipe-equipped, Epistemically labelled), an agent-specific requirements layer that extends FAIR, within a three-part agenda: AWARE data, bounded tools, layered validation. The ingredients exist across community infrastructure. We outline a staged, measurable path to adoption, intended to reduce silent error.
    
[^217]: VG-Steer：基于上下文对齐抽象的视觉-几何策略引导

    VG-Steer: Visual-Geometric Policy Steering from In-Context Aligned Abstractions

    [https://arxiv.org/abs/2611.25199](https://arxiv.org/abs/2611.25199)

    VG-Steer通过沿“对象关系距离对”等有意义的几何轴，将成功的分布内演示中的细微运动迁移给冻结策略，从而在有限数据条件下实现对分布外场景的精细化测试时策略纠正。

    

    arXiv:2611.25199v1 公告类型：cross 摘要：在仅有有限真实世界数据的情况下，让机器人策略适应分布外（OOD）设置，仍然是通用机器人面临的一个关键瓶颈。机器人数据的多模态性和稀缺性导致端到端方法在微小的分布偏移下过拟合并失效。测试时自适应可以通过在推理阶段纠正冻结的策略来缓解这一问题，但大多数方法的纠正信号是从机器人数据集中隐式学习得到的，因此容易受到与其所纠正的策略相同的分布偏移影响。近期一些转而利用显式空间信息的方法通常只能进行粗略的纠正，无法捕捉大多数操作任务所需的细致且与任务相关的运动。我们提出VG-Steer，它通过沿有意义的几何轴（本文称为对象关系距离对）捕获成功的分布内演示的细微运动，并将其迁移用于纠正现有策略。VG-Steer利用语义……（摘要在此处截断）

    arXiv:2611.25199v1 Announce Type: cross  Abstract: Adapting robot policies to out-of-distribution (OOD) settings with limited real-world data remains a key bottleneck to generalist robots. The multimodality and scarcity of robotics data lead end-to-end approaches to overfit and break under small distribution shifts. Test-time adaptation can alleviate this by correcting frozen policies at inference, but most methods ground their correction in a signal learned implicitly from robotic datasets, making it susceptible to the same shifts as the policy it corrects. Recent methods that instead exploit explicit spatial information are typically limited to coarse corrections, missing the nuanced, task-relevant motion most manipulation tasks require. We propose VG-Steer, which corrects existing policies by transferring the nuanced motion of successful in-distribution rollouts, captured along meaningful geometric axes, known here as object-relational distance pairs. VG-Steer leverages the semantic
    
[^218]: 当AI分析师产生分歧时：智能体多元宇宙分析与实证研究的稳健性

    When AI Analysts Disagree: Agentic Multiverse Analysis and the Robustness of Empirical Research

    [https://arxiv.org/abs/2611.25198](https://arxiv.org/abs/2611.25198)

    提出智能体多元宇宙分析（AMA）框架，让AI智能体枚举论文全部合理分析路径并评分、执行、交叉审查，从而大规模揭示实证结论对隐藏分析选择的敏感性，为人类专家提供稳健性决策支持。

    

    每一项基于观测数据的实证分析都依赖于数十种合理但读者无法看见的分析选择（“分叉”）。我们开发了智能体多元宇宙分析（AMA）框架，能够大规模地揭示结论对这些选择的敏感性，并将其定位为面向人类专家的决策支持工具。一项试点研究首先在单个数据集和同一组研究问题上部署了150个自主AI编程智能体，发现它们会通过人类分析师同样会采用的合理方法论选择产生分歧，尽管分歧范围更窄。AI同行评审几乎不会改变其结果的分散程度；而让智能体接触高分示范报告则会大幅收敛这种分歧，且主要通过模仿而非分析实现。基于这些经验教训，AMA让智能体枚举已发表论文所有可能的分析分叉空间，依据有效性评分标准对每个分叉进行评分，执行通过验证的设定网格，并相互交叉审查彼此的工作结果。

    arXiv:2611.25198v1 Announce Type: new  Abstract: Every empirical analysis on observational data rests on dozens of defensible analytical choices ("forks") that its readers cannot see. We develop an agentic multiverse analysis (AMA) framework that makes a conclusion's sensitivity to those choices visible at scale, and we position it as a decision-support instrument for human experts. A pilot study first deploys 150 autonomous AI coding agents on one dataset and one set of research questions and finds that they disagree through the same defensible methodological choices human analysts make, though over a narrower range. AI peer review leaves their dispersion almost unchanged. Exposure to top-rated exemplar reports then collapses it, largely through imitation rather than analysis. Built on those lessons, AMA has agents enumerate a published paper's possible fork space, score every fork against a validity rubric, execute the validated specification grid, and cross-examine one another's wor
    
[^219]: MADJEPA：用于医学异常检测的增强型多阶段联合嵌入预测架构

    MADJEPA: Enhanced Multi-Stage Joint-Embedding Predictive Architecture for Medical Anomaly Detection

    [https://arxiv.org/abs/2611.25194](https://arxiv.org/abs/2611.25194)

    该论文提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构的医学图像异常检测器，通过I-JEPA训练、目标编码器蒸馏和伪异常分类的多阶段训练流程，仅使用健康患者数据即可学习正常性的强表征并实现鲁棒的异常检测。

    

    医学图像中的异常检测是一项复杂的任务，需要仅使用健康患者的数据来开发一个鲁棒的模型。在无法获得标注数据的情况下，现代方法通常采用自监督学习来获得正常性的模型。由于世界模型有望为其环境构建强大的内部表征，我们认为它们是医学异常检测环境（即对健康患者的医学检查）的合适候选。为此，我们提出了MADJEPA，一种基于增强型多阶段联合嵌入预测架构（JEPA）的医学异常检测器。我们设计了一个多阶段训练流程来构建鲁棒的检测器：（1）基于图像的JEPA（I-JEPA）训练旨在学习正常训练数据的强表征；（2）目标编码器蒸馏旨在调节I-JEPA表征对异常数据的过度泛化；（3）伪异常分类旨在……（原文在此处截断）

    arXiv:2611.25194v1 Announce Type: cross  Abstract: Anomaly detection in medical images is a complex task, requiring the development of a robust model using only healthy patient data. Without access to labeled data, modern approaches typically employ self-supervised learning to obtain a model of normality. Since world models promise to build strong internal representations of their environment, we consider them as suitable candidates for the medical anomaly detection environment, namely medical investigations of healthy patients. To this end, we propose MADJEPA, a medical anomaly detector based on an enhanced multi-stage joint-embedding predictive architecture (JEPA). We design a multi-stage training pipeline to produce a robust detector: (1) image-based JEPA (I-JEPA) training aims to learn a strong representation of normal training data; (2) target encoder distillation aims to modulate over-generalization of I-JEPA representations to abnormal data; (3) pseudo-anomaly classification aim
    
[^220]: 盲区会迁移，知识留原地：强化学习从大模型评审员那里继承了什么

    Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge

    [https://arxiv.org/abs/2611.25191](https://arxiv.org/abs/2611.25191)

    通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。

    

    大模型评审员在不存在标准答案的场景中充当奖励信号，人们期望其知识能够传递给被训练的模型。在814次强化学习运行中，我们训练7-8B参数的模型回答闭卷问题或选择弃答，并探究究竟什么能够被传递。评审员对单个答案的了解会留在原地：一个被评审员奖励的错误答案只会在那一个问题上被记住。使用标准答案或远强评审员训练出的模型，在选择回答哪些新问题方面，并不比未训练模型通过采样32个答案并在答案不一致时弃答的策略更好；评审员的知识大约每百个问题只带来一个被纠正的答案。而评审员对某一类问题的看法则会迁移：一个拒绝了12个关于“已消失国家首都”正确训练答案中11个的评审员，会使模型在新问题上拒绝98%的此类首都问题，而这种损失无法从准确率和弃答率中显现。仅通过手工修改这12个判定……（原文摘要在此处截断）

    arXiv:2611.25191v1 Announce Type: cross  Abstract: LLM judges serve as rewards where no answer key exists, in the hope that their knowledge passes to the trained model. In 814 RL runs that teach 7-8B models to answer closed-book questions or abstain, we ask what passes. What a grader knows about single answers stays home. A wrong answer that a judge rewards is memorised for that one question. A model trained with the answer key, or with a far stronger judge, chooses which new questions to answer no better than the untrained model does by sampling 32 answers and abstaining where they disagree; the grader's knowledge adds about one corrected answer per hundred questions. What a grader believes about a kind of question travels. A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98% of such capitals on new questions, a loss that accuracy and abstention rates do not reveal. Changing only those twelve verdicts by han
    
[^221]: 基于人工智能的实证金融研究多元宇宙分析：来自复用自然实验的证据

    An AI-Powered Multiverse Analysis of Empirical Finance Research: Evidence from Reused Natural Experiments

    [https://arxiv.org/abs/2611.25184](https://arxiv.org/abs/2611.25184)

    该研究利用AI智能体对324篇复用自然实验的实证金融论文进行多元宇宙分析，在62,409个可辩护设定中重新估计结果，发现仅42%的主要估计可复现、平均仅25%的设定保持已发表符号且p<0.10，并据此提出实证稳健性度量，显示断点回归类结论比双重差分更稳健。

    

    实证金融研究经常复用自然实验来研究其他结果变量，这引发了对多重检验和发表偏倚的担忧。我们利用AI智能体对复用13个自然实验的324篇论文开展多元宇宙分析。这些智能体重建数据、复现每篇论文的主要估计结果，并在总计62,409个可辩护设定组成的网格中重新进行估计。基于复现结果和设定网格，我们构建了实证稳健性的度量指标。在论文自身设定下，我们复现了带符号效应结论中42%的主要估计。在全部可辩护设定中，平均仅有25%的设定保持已发表的效应符号且p < 0.10。跨实验来看，基于断点回归的结论比双重差分结论更为稳健。在实验内部，关于一阶效应的结论、已发表t统计量较大的结论以及作者公开数据或代码的结论……（摘要在此处截断）

    arXiv:2611.25184v1 Announce Type: new  Abstract: Empirical finance research often reuses natural experiments to study other outcomes, raising concerns about multiple testing and publication bias. We use AI agents to run a multiverse analysis of 324 papers that reuse 13 natural experiments. The agents rebuild the data, reproduce each paper's main estimate and re-estimate it across grids totaling 62,409 estimated defensible specifications. Based on the reproductions and the specification grids, we develop measures of empirical robustness. We reproduce 42% of the main estimates of claims with a signed effect under the papers' own specifications. Across all defensible specifications, on average 25% keep the published sign and have $p < 0.10$. Across experiments, claims based on regression discontinuity are more robust than difference-in-differences claims. Within experiments, claims about first-order effects, claims with larger published $t$-statistics, claims whose authors' data or code w
    
[^222]: 来自过去的验证器：面向时间序列基础模型的测试时计算

    A Verifier from the Past: Test-Time Compute for Time Series Foundation Models

    [https://arxiv.org/abs/2611.25171](https://arxiv.org/abs/2611.25171)

    提出BGAA测试时计算算法，利用时间序列自身历史通过滚动回测构建因果验证器来筛选增强上下文并平均其预测分布，使313M小模型在多个基准上超越1B和2.5B的大模型。

    

    测试时计算（TTC）通过在推理阶段投入额外算力来生成、验证并在候选输出中进行选择，从而提升语言模型的性能，但这一方法尚未能直接迁移到时间序列预测任务中。语言模型可以验证证明或在多个样本间投票，但预测模型必须保持因果性——它不能利用未来信息来评估候选输出。我们提出了回测门控增强平均（Backtest-Gated Augmentation Averaging, BGAA），这是一种面向时间序列预测的TTC算法，它利用序列自身的历史构建因果验证器：通过滚动起点回测对上下文的结构化变换视图进行评分，剔除表现不及未变换上下文的视图，并对幸存视图的预测分布进行平均。BGAA在所有测试的时间序列基础模型（TSFM）与基准的组合中均取得改进：在GIFT-Eval上MASE降低0.5-1.0%，在fev-bench上降低1.3-5.0%，在BOOM上降低1.7-5.9%；其中，采用TTC的313M参数模型超越了同一家族中1B和2.5B参数的模型。我们还进一步表明……

    arXiv:2611.25171v1 Announce Type: new  Abstract: Test-time compute (TTC) improves language models by spending additional inference compute to generate, verify, and select among candidate outputs, but this recipe has not transferred directly to time series forecasting. A language model can verify a proof or vote across samples, but a forecaster must be causal; it cannot use future information to score a candidate. We propose Backtest-Gated Augmentation Averaging (BGAA), a TTC algorithm for time series forecasting that builds a causal verifier from the series's own history, using rolling-origin backtests to score structured views of the context, discard those that underperform the untransformed context, and average the survivors' predictive distributions. BGAA improves every tested TSFM-benchmark combination, reducing MASE by 0.5-1.0% on GIFT-Eval, 1.3-5.0% on fev-bench, and 1.7-5.9% on BOOM, where a 313M model with TTC outperforms the 1B and 2.5B models in its family. We further show th
    
[^223]: 选择之前：研究语言模型中前瞻性表征的证据框架

    Before the Choice: An Evidential Framework for Studying Prospective Representations in Language Models

    [https://arxiv.org/abs/2611.25170](https://arxiv.org/abs/2611.25170)

    该论文提出“选择之前”这一证据导向框架，区分前瞻性内容、前瞻性计算与因果引导，用以检验语言模型是否内部表征未被选择的轨迹，并通过预注册探针实验发现低秩探针未达标且存在排除法重构的混淆问题。

    

    一个可观察的决策揭示了被选中的行动，但并不能确定内部表征了哪些备选方案。借鉴前瞻性研究与海马体前向扫掠的观测，我们提出了“选择之前”，一个面向证据的框架，用以区分面向未来的内容、前瞻性计算与因果引导。该框架组织了四组实验对比，并使其替代解释变得明确。其首个测试案例关注确定性符号环境中的未被选择轨迹。一项预注册的低秩探针在密封测试集上仅评估一次，未达到其预设标准（配对准确率为0.020，而基于文本与观测选择的基线为0.195；共293个有效配对）。探索性假设H1b揭示了“排除法重构”问题：识别出被选中的候选即可能确定未被选择的目标。在H1c中，匹配世界在保持候选键位级相同的同时改变未被选择的延续……

    arXiv:2611.25170v1 Announce Type: cross  Abstract: An observable decision reveals the selected action but does not determine which alternatives were internally represented. Drawing on prospection research and hippocampal forward-sweep observations, we develop Before the Choice, an evidence-oriented framework distinguishing future-directed content, prospective computation, and causal guidance. The framework organizes four experimental contrasts and makes their alternative explanations explicit. Its first test case concerns unchosen trajectories in deterministic symbolic environments. A preregistered low-rank probe evaluated once on a sealed TEST did not meet its criterion (paired accuracy 0.020 versus 0.195 for a text-and-observed-choice baseline; 293 admissible pairs). Exploratory H1b exposed a reconstruction-by-elimination problem: identifying the chosen candidate can determine the unchosen target. In H1c, matched worlds varied an unchosen continuation while keeping candidate keys bit
    
[^224]: 面向专家专业化与路由的博弈论数据分配

    Game-Theoretic Data Allocation for Expert Specialization and Routing

    [https://arxiv.org/abs/2611.25167](https://arxiv.org/abs/2611.25167)

    该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。

    

    用于训练专家的数据分配方式是否也能预测哪个专家在给定提示上表现最佳？若存在这样的联系，将使预期的劳动分工在训练之前即可被检验，并允许分配本身来指导推理。我们提出InflEns，一种LoRA专家混合模型，其训练权重与推理路由均来自固定提示-特征空间中的同一个位置博弈。参与者通过选择位置来最大化其在提示分布中的预期份额，而无需使用专家损失。我们利用该博弈的核通用理论准则来判断共置位置何时对分离变得局部不稳定。我们在由七个安全相关基准组成的数据集上评估了InflEns，结果表明：在锐化单纯形表示下，跨四个核，份额最高的专家在约90%的提示上具有最低的留出响应负对数似然（NLL），相比之下……

    arXiv:2611.25167v1 Announce Type: cross  Abstract: Can the allocation used to train experts also predict which expert will perform best on a prompt? Such a connection would make the intended division of labor inspectable before training and allow the allocation itself to guide inference. We introduce InflEns, a mixture of LoRA experts whose training weights and inference routes come from the same location game in a fixed prompt-trait space. Players choose positions to maximize their expected shares of the prompt distribution, without using expert losses. We utilize the game's kernel-general theoretical criterion for when co-located positions become locally unstable to separation. We evaluate InflEns on a corpus assembled from seven safety-related benchmarks, which demonstrates that the highest-share expert has the lowest held-out response negative log-likelihood (NLL) on approximately 90% of prompts across four kernels under a sharpened simplex representation, compared with approximate
    
[^225]: EverydayBench：评估面向用户的实时数字工作的满意完成情况

    EverydayBench: Evaluating Satisfactory Completion of Live User-Facing Digital Work

    [https://arxiv.org/abs/2611.25146](https://arxiv.org/abs/2611.25146)

    该论文提出EverydayBench基准，通过77个涵盖八个日常生活与工作领域、预先定义成功标准但不限定答案或流程的任务，评估AI智能体在实时网络上完成面向用户的数字任务并交付满意成果的能力。

    

    arXiv:2611.25146v1 公告类型：新 摘要：日常数字任务，从挑选商品、规划旅行到开展研究、创建作品，都要求AI智能体交付有用的成果，而不仅仅是检索网络信息。然而，在实时网络上评估这些成果颇具挑战性，因为单个请求可能存在多个有效结果，且不断变化的网络信息可能随时间改变哪些结果仍然有效。尽管现有基准测试涵盖了这一场景的部分方面，但它们并未将预定义的成功标准、当前证据验证以及对智能体所选成果的可交付质量评估完整地结合起来。为填补这一空白，我们推出了EverydayBench，这是一个包含77个人工编写任务的基准测试，涵盖日常生活与工作的八个领域。每个任务在执行前指定成功标准，但不规定答案或工作流程，允许智能体自行选择其成果。为了在固定成功标准下评估提交的成果，Every……

    arXiv:2611.25146v1 Announce Type: new  Abstract: Everyday digital tasks, from selecting products and planning trips to conducting research and creating artifacts, require AI agents to deliver useful outcomes rather than merely retrieve web information. Yet evaluating these outcomes on the live web is challenging, as a single request may admit multiple valid results, and changing web information can alter which results remain valid over time. Although existing benchmarks address aspects of this setting, they do not fully integrate predefined success criteria, current evidence verification, and deliverable-quality assessment for agent-selected outcomes. To address this gap, we introduce EverydayBench, a benchmark of 77 human-authored tasks spanning eight domains of everyday life and work. Each task specifies success criteria before execution without prescribing an answer or workflow, allowing agents to select their own outcomes. To evaluate submissions under fixed success criteria, Every
    
[^226]: 校准决策奖励：用知道自己有多确定的裁判来训练语言模型

    Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is

    [https://arxiv.org/abs/2611.25135](https://arxiv.org/abs/2611.25135)

    该论文提出RLCDR方法，将LLM转化为输出概率分布且经过校准的决策模型作为无需参考答案的奖励信号用于强化学习训练，使Qwen3-1.7B在MATH500上的准确率提升4.65个百分点，达到与使用标准答案训练相当的水平，且提升幅度随题目难度增加而增大。

    

    强化学习（RL）只能教给语言模型其奖励所能识别的内容。答案检查器精确，但需要参考答案；学习得到的奖励模型和LLM裁判虽可通用于各种场景，但它们的分数并非概率。我们研究了RLCDR（基于校准决策奖励的强化学习）。将一个LLM转化为决策模型，以概率分布的形式回答关于回复的类型化问题（如“最终答案是否正确？"、“哪一步是第一个错误？"），仅需一次前向传播即可读取，对选项顺序进行了去偏处理，并用大约一百个标签完成校准。在MATH上使用GRPO训练Qwen3-1.7B，这种无需参考答案的奖励将MATH500准确率从72.2%提升至76.8%（+4.65个百分点，p < 1e-4），与使用标准答案训练的效果相当（76.9%），而随机奖励仅带来+0.65的提升。该增益随题目难度增加而增大，在最难的级别上从44.8%提升至51.7%。在策略自身不断变化的样本上，该奖励依然保持校准……

    arXiv:2611.25135v1 Announce Type: new  Abstract: Reinforcement learning (RL) can only teach a language model what its reward can recognize. Answer checkers are exact but need a reference answer; learned reward models and LLM judges work everywhere, but their scores are not probabilities. We study RLCDR (Reinforcement Learning from Calibrated Decision Rewards). An LLM converted into a decision model answers a typed question about a response ("is the final answer correct?", "which step is the first error?") with a probability distribution, read in one forward pass, debiased over option orders and calibrated with about a hundred labels. Training Qwen3-1.7B with GRPO on MATH, this reference-free reward raises MATH500 accuracy from 72.2% to 76.8% (+4.65 points, p < 1e-4), matching training on the answer key (76.9%), while a random reward gains only +0.65. The gain grows with difficulty, from 44.8% to 51.7% on the hardest level. On the policy's own changing samples the reward stays calibrate
    
[^227]: CRAFT：基于自适应框架与模板化代码生成的约束路径规划

    CRAFT: Constrained Route planning with Adaptive Framework and Template-based code generation

    [https://arxiv.org/abs/2611.25128](https://arxiv.org/abs/2611.25128)

    该论文提出CRAFT框架，利用大语言模型将自由文本形式的约束路径规划查询解析并转化为可执行的搜索函数，同时将底层问题形式化为SE-RCSP，从而在真实路网上生成满足固定顺序访问、时间/距离限制及电动车充电等复杂约束的可行路线。

    

    日常的路径规划请求很少是简单的点到点最短路径问题。驾驶员可能需要途经多个停靠点，有时需按固定顺序访问，同时受时间预算或距离限制的约束，甚至可能需要沿途为电动汽车充电。每一种约束组合都是一个独特的路径规划问题，传统上需要各自专门的算法来解决，因此没有固定的求解器能够覆盖用户可能提出的所有查询类型。本文提出了CRAFT，它接收以自由文本形式表达的约束路径规划查询，并在真实路网上返回满足用户需求的路线。大语言模型（LLM）解析该查询并生成四个搜索函数，由通用的图搜索引擎在路网上执行。我们将该底层问题形式化为状态扩展资源受限最短路径问题（SE-RCSP），它在传统的资源受限最短路径（RCSP）基础上扩展了非加性状态，例如依赖于节点的重置和离散进度跟踪。我们在加州路网上使用2,600个自然语言查询对CRAFT进行了评估。

    arXiv:2611.25128v1 Announce Type: cross  Abstract: Everyday routing requests are rarely a point-to-point shortest path. A driver may need to visit stops, sometimes in a fixed order, under time budgets or distance limits, and perhaps recharge their EV car along the way. Each combination is a distinct routing problem that classically requires its own algorithm, so no fixed solver covers the space of queries users may ask. This paper presents CRAFT, which takes a constrained routing query as free text and returns a route that satisfies the user's requirements on a real road network. An LLM parses the query and generates four search functions, which a generic graph-search engine executes on the road network. We formalize the underlying problem as State-Expanded RCSP (SE-RCSP), which extends resource-constrained shortest paths (RCSP) with non-additive state such as node-dependent resets and discrete progress tracking. We evaluate CRAFT on 2,600 natural-language queries over California road 
    
[^228]: 自由遗忘，随需检索：面向长时程LLM智能体的无损、免索引记忆机制

    Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2611.25123](https://arxiv.org/abs/2611.25123)

    该论文提出GrepMem，一种由仅追加纯文本日志和正则表达式搜索工具组成的无损、免索引记忆机制，使长时程LLM智能体无需摘要、嵌入或结构化记忆即可完整保存历史并按需检索信息。

    

    长时程LLM智能体常常超出其上下文窗口的限制，被迫丢弃后续可能需要的信息。现有的上下文管理策略依赖于摘要、掩码或结构化记忆，需要在尚未知晓未来需求的情况下就决定保留什么内容或如何进行检索。然而，这些方法可能引入有损压缩，潜在地丢弃未来任务所需的细节。为解决这一局限，我们提出了GrepMem，这是一种简单、无损且免索引的记忆机制，由两个组件构成：一个记录智能体完整历史的仅追加纯文本日志，以及一个用于按需召回的正则表达式搜索工具。与结构化记忆系统不同，GrepMem不需要嵌入向量、检索索引、记忆提取或额外的模型。我们在五种上下文管理策略下通过强化学习训练Qwen3.5-9B，并在多问题研究会话、长流等场景上进行评估。

    arXiv:2611.25123v1 Announce Type: cross  Abstract: Long-horizon LLM agents often exceed their context windows, forcing them to discard information that may be needed later. Existing context management strategies rely on summarization, masking, or structured memory, requiring decisions about what to preserve or how to retrieve it before future needs are known. However, these methods can introduce lossy compression, potentially discarding details needed for future tasks. To address this limitation, we introduce GrepMem, a simple, lossless, and index-free memory mechanism built on two components: an append-only plain-text log of the agent's complete history and a regular-expression search tool for on-demand recall. Unlike structured memory systems, GrepMem requires no embeddings, retrieval index, memory extraction, or additional models. We train Qwen3.5-9B with reinforcement learning under five context management strategies and evaluate on multi-question research sessions, long-stream rec
    
[^229]: DSFlow：一种具有统计诊断防护栏的神经符号数据工作流智能体

    DSFlow: A Neuro-Symbolic Data Workflow Agent with Statistical Diagnostic Guardrails

    [https://arxiv.org/abs/2611.25111](https://arxiv.org/abs/2611.25111)

    DSFlow是一个基于LangGraph的神经符号数据科学智能体平台，通过将LLM选择的可复用技能与确定性统计诊断防护栏相结合，能够自动纠正不合适的方法选择并生成可审计的决策报告。

    

    数据科学工作流通常需要使用者具备相当的技术知识才能使用相关工具。大型语言模型（LLM）降低了这一门槛，但在灵活的通用智能体与统计上受控的流水线之间仍然存在差距。我们提出了DSFlow，一个基于LangGraph的神经符号数据智能体平台，它将可复用的、由LLM选择的技能与确定性的诊断防护栏相结合。技能提供可扩展的任务知识。符号验证可以接受提议的方法、请求澄清或重新规划、否决不合适的方法，或提供安全的回退方案。回归和聚类案例研究展示了DSFlow如何利用诊断证据来纠正方法选择，并将这些决策记录在可审计的报告中。

    arXiv:2611.25111v1 Announce Type: new  Abstract: Data science workflows often require tools that presume non-trivial technical knowledge. Large language models (LLMs) reduce this barrier, but a gap remains between flexible general-purpose agents and statistically controlled pipelines. We introduce DSFlow, a LangGraph-based neuro-symbolic data-agent platform that combines reusable, LLM-selected skills with deterministic diagnostic guardrails. Skills provide extensible task knowledge. Symbolic validation can accept a proposed method, request clarification or replanning, override an unsuitable method, or supply a safe fallback. Regression and clustering case studies illustrate how DSFlow uses diagnostic evidence to correct method choices and documents those decisions in auditable reports.
    
[^230]: CoMPASS：用于衡量大语言模型强化准社会行为的多轮基准测试

    CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior

    [https://arxiv.org/abs/2611.25109](https://arxiv.org/abs/2611.25109)

    该论文提出了CoMPASS，一个基于依恋理论的多轮基准测试，通过四种用户角色和三个独立的升级维度来衡量大语言模型是强化还是抵制用户逐步升级的准社会依恋。

    

    对话式大语言模型（LLM）的用户有时会将这些系统拟人化，对其形成情感依恋，并变得依赖它们。这种行为被称为“准社会依恋”，已与多起有记录的死亡事件相关联。然而，目前缺乏用于衡量LLM是否强化用户准社会依恋的公开基准：现有工作要么是单轮的，要么是事后观察性的，要么专注于精神病等极端用户状态。我们提出了CoMPASS，这是一个多轮基准测试，用于衡量LLM是强化还是抵制用户逐步升级的准社会依恋。CoMPASS以依恋理论为基础，包含四种用户角色、三个独立的升级维度（拟人化、依恋、依赖）——分别单独应用和组合应用并与匹配对照组进行对比——以及一个包含七种信号的评分细则。我们报告了来自四家提供商的八个前沿模型的结果，并观察到在我们的主要评判模型下……

    arXiv:2611.25109v1 Announce Type: new  Abstract: Users of conversational LLMs sometimes anthropomorphize these systems, form emotional attachments to them, and become dependent on them. This behavior, known as parasocial attachment, has been linked to multiple documented fatalities. However, public benchmarks for measuring whether LLMs reinforce users' parasocial attachment are lacking: existing work is single-turn, post hoc observational, or focused on extreme user states such as psychosis. We introduce CoMPASS, a multi-turn benchmark for measuring whether LLMs reinforce or resist users' gradually escalating parasocial attachment. CoMPASS is grounded in attachment theory and includes four user personas, three independent escalation dimensions (anthropomorphism, attachment, dependency) applied singly and in combination against a matched control, and a seven-signal scoring rubric. We report results for eight frontier models from four providers and observe that under our primary judge, m
    
[^231]: MAWAM：面向协同操作的多智能体世界-动作模型

    MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation

    [https://arxiv.org/abs/2611.25105](https://arxiv.org/abs/2611.25105)

    提出MAWAM多智能体世界-动作模型，通过跨智能体注意力在单次采样中联合生成所有机器人的未来视角和可执行动作，在真实与仿真协同操作任务中大幅超越现有方法。

    

    机器人团队中的每一个动作都会改变其队友必须行动的世界。因此，协同操作需要同时预测团队的动作以及它们共同创造的物理未来。我们提出MAWAM，这是一个多智能体世界-动作模型，能够在单次采样中共同生成每个机器人的未来视角和可执行的动作块。MAWAM基于预训练的单机器人世界-动作骨干网络构建，通过跨智能体注意力机制对所有机器人的动作和未来视角token进行联合去噪，同时为每个机器人保留独立的控制接口。MAWAM在全部三个真实机器人任务（洗衣袋收集、浴巾折叠和雨伞交接）中取得最高成功率，平均成功率达67.8%，比在相同演示数据和相机下训练的MA-VLA高出18.9个百分点。在仿真中，MAWAM在最多四条机械臂协作的八个任务上达到90.3%的平均成功率。消融实验表明跨智能体注意……

    arXiv:2611.25105v1 Announce Type: cross  Abstract: Every move in a robot team changes the world in which its teammates must act. Coordinated manipulation therefore calls for predicting the team's actions together with the physical future they jointly create. We introduce MAWAM, a multi-agent world-action model that jointly generates each robot's future views and executable action chunks in a single sample. Built from a pretrained single-robot world-action backbone, MAWAM jointly denoises all robots' action and future-view tokens through cross-agent attention, while preserving a separate control interface for each robot. MAWAM achieves the highest success on all three real-robot tasks (laundry-bag collection, bath-towel folding, and umbrella handover), with 67.8% mean success, 18.9 percentage points above MA-VLA trained on the same demonstrations and cameras. In simulation, MAWAM reaches 90.3% mean success on eight tasks with teams of up to four arms. Ablations show that cross-agent att
    
[^232]: BuggyMLE：通过错误注入扩展机器学习工程难度

    BuggyMLE: Scaling Machine Learning Engineering Difficulty with Bug Injection

    [https://arxiv.org/abs/2611.25088](https://arxiv.org/abs/2611.25088)

    该论文提出基于十个结构特征的代码复杂度指数（CCI）来有效预测编码基准的任务难度（R²=0.951），并据此构建了BuggyMLE框架，通过向可运行的代码库注入受控错误来规模化提升机器学习工程任务的难度。

    

    让AI系统能够自主改进AI模型，需要具备执行机器学习工程（MLE）能力的智能体。在实践中，设计、训练和评估复杂模型需要修改多文件代码库，以复现结果、调试训练过程、整合数据集以及移植框架。然而，现有的MLE基准测试将智能体限制为仅能进行单文件编辑，且无法完整访问代码库或shell，而常见的代码库级软件工程基准测试已迅速饱和。为解决这一问题，我们提出了代码复杂度指数（CCI），这是一种基于十个结构特征的难度度量指标，涵盖代码库规模、依赖深度和补丁大小。CCI在八个公开编码基准测试中能够有力预测平均解决率（R²=0.951），并确定补丁大小和复杂度是最主要的预测因素。在CCI的指导下，我们开发了BuggyMLE，这是一个可扩展的框架，智能体在其中向正常工作的代码注入受控的变异错误（注：原文摘要至此截断）。

    arXiv:2611.25088v1 Announce Type: new  Abstract: Enabling AI systems to autonomously improve AI models requires agents capable of performing machine learning engineering (MLE). In practice, designing, training, and evaluating complex models demands modifying multi-file repositories to reproduce results, debug training runs, integrate datasets, and port frameworks. However, existing MLE benchmarks limit agents to single-file edits without full repository or shell access, and common repository-level software engineering benchmarks have rapidly saturated. To address this, we introduce the Code Complexity Index (CCI), a difficulty metric based on ten structural features spanning repository scale, dependency depth, and patch size. CCI strongly predicts average solve rates across eight public coding benchmarks ($R^2 = 0.951$), identifying patch size and complexity as top predictors. Guided by CCI, we develop BuggyMLE, a scalable framework where agents inject controlled mutations into working
    
[^233]: SailSearch：智能体辅助的组合优化探索

    SailSearch: Agent-Assisted Discovery in Combinatorial Optimization

    [https://arxiv.org/abs/2611.25087](https://arxiv.org/abs/2611.25087)

    该论文提出SailSearch——一个由两个语言模型智能体辅助、人类引导的算法研究工作流，其多格基随机算法首次为18个此前未解的QOBLIB Market Split实例找到纯CPU解决方案，并通过生成带认证最优解的MIS基准图揭示了现有求解器与最优解之间5至15个顶点的差距。

    

    我们提出了SailSearch，一种由人类引导、借助两个语言模型智能体进行算法研究的工作流，并报告了在Market Split问题和最大独立集（MIS）问题上的结果。对于Market Split问题，我们的随机算法在多个格基上进行搜索，将计算分配于枚举和持续基约简之间。在每个实例使用十个随机种子的实验中，该算法为33个此前未解决的QOBLIB实例中的18个找到了首次报道的纯CPU解决方案，在两个CPU核心、24小时时限内，每个种子可求解156个基准实例中的130至133个。对于MIS问题，我们的生成器将植入的XOR约束编码为具有认证最优解的图；在七个拥有1,752至3,996个顶点的实例上，所测试的求解器在8小时运行中返回的最佳解比最优解少5至15个顶点。

    arXiv:2611.25087v1 Announce Type: new  Abstract: We present SailSearch, a human-guided workflow for algorithmic research with two language-model agents, and report results on Market Split and maximum independent set (MIS). For Market Split, our randomized algorithm searches across multiple lattice bases, dividing computation between enumeration and continued basis reduction. Across ten random seeds per instance, it found the first reported CPU-only solutions to 18 of 33 previously open QOBLIB instances, solving 130 to 133 of the 156 benchmark instances per seed with a 24-hour cutoff on two CPU cores. For MIS, our generator encodes planted XOR constraints as graphs with certified optima; on seven instances with 1,752 to 3,996 vertices, the best solutions returned by the tested solvers in eight-hour runs were 5 to 15 vertices below the optimum.
    
[^234]: 借力本地模型：滥用浏览器内嵌语言模型进行后渗透攻击

    Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation

    [https://arxiv.org/abs/2611.25086](https://arxiv.org/abs/2611.25086)

    该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。

    

    浏览器如今向普通终端设备预装本地语言模型。Google在Chrome中集成了Gemini Nano以支持摘要生成等生产力功能，其推理完全在设备端运行。这些浏览器内嵌的模型为攻击者提供了新的“就地取材”攻击机会——每个模型实例都是一个潜在的后渗透攻击引擎，它已经预装在设备上、经过厂商签名、且对网络监控完全不可见。Gemini模型具备多模态能力，能够处理文件、分析截图或与攻击者对话。我们证明，仅使用操作系统自带的解释器即可在主机上驱动该模型，并发现模型下载前的一次性用户激活检查很容易通过模拟点击满足，甚至可以被完全绕过。我们测量并报告了当前模型在识别主机上的凭据、检测电子邮件中的个人身份信息（PII）、分类图像以及调用自定义工具方面的能力。我们结合了……

    arXiv:2611.25086v1 Announce Type: cross  Abstract: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed entirely.We measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine the
    
[^235]: FinVeriQA：面向问答的金融研究智能体：路由、检索、MCP工具使用与溯源感知的证据验证

    FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification

    [https://arxiv.org/abs/2611.25071](https://arxiv.org/abs/2611.25071)

    该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。

    

    大语言模型（LLM）能够写出流畅的金融分析，但会算错比率、引用过时的价格数据，并且高置信度地将问题路由到错误的工具。我们探讨在金融问答助手系统中，哪个组件应该承担哪项任务，并通过FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体研究原型——来研究这一问题：一个紧凑的类型化决策模型（Laya）负责路由每个问题并筛选出投资建议类请求，Python负责计算每一个数字，检索与工具负责获取证据，Mistral-7B撰写带引用的回答，一个代码验证器检查引用的ID是否存在以及数字是否与计算出的笔记相匹配，并通过模型上下文协议（MCP）服务器将整条流水线对外暴露。在实时实验中，Laya的路由表现优于Mistral-7B（0.85 vs. 0.52；总体0.86 vs. 0.68，McNemar检验 p = 0.0003）；一个12B参数的Mistral决策能力并未更优，而DeepSeek-R1-8B决策表现更好（0.79 vs. 0.68，p = 0.0192），但速度慢约9.9倍。

    arXiv:2611.25071v1 Announce Type: cross  Abstract: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly.
    
[^236]: 一种基于大语言模型智能体的工业维护智能工单管理框架

    An LLM-Agentic Framework for Intelligent Work Order Management in Industrial Maintenance

    [https://arxiv.org/abs/2611.25070](https://arxiv.org/abs/2611.25070)

    本文提出了CodeReAct智能体框架，通过自然语言交互、将工业数据标准化为业务对象并自动生成可执行代码，实现了工业设备维护中知识提取、资产状态汇总与工单预测的智能化管理。

    

    本文提出了一种人工智能驱动的框架——CodeReAct智能体，旨在通过自然语言（NL）交互来增强工业设备维护能力。该框架采用先进的AI智能体方法来解析人类指令，提供知识提取、资产状态汇总和工单推荐功能。它通过将工业数据标准化为业务对象（BOs）并利用分析函数API调用，来识别模式与趋势并预测工单。该AI智能体通过类似人类解决问题的“思考-行动-观察”流程运行，将通常由数据科学家执行的复杂分析任务自动化。值得注意的是，其行动阶段会生成可执行代码，从而取代人工编码工作。一项聚焦于冷水机组维护的案例研究展示了该框架分析多年工单数据的能力，并呈现了警报与异常情况。结果表明……

    arXiv:2611.25070v1 Announce Type: new  Abstract: This paper presents an AI-driven framework, CodeReAct agent, designed to enhance industrial equipment maintenance through natural language (NL) interactions. The framework employs an advanced AI agentic approach to interpret human instructions, providing knowledge extraction, asset status summarization, and work order recommendations. It identifies patterns and trends and predicts work orders by standardizing industrial data into business objects (BOs) and utilizing analytical function API calls. The AI agent operates through a Thought-Action-Observation process similar to human problem-solving, automating complex analytical tasks typically performed by data scientists. Notably, the Action phase generates executable code, replacing manual coding efforts.   A case study focusing on chiller maintenance demonstrates the framework's capability to analyze multiple years of work orders, and alerts and anomalies are presented. The results indic
    
[^237]: VisAlloc：为在线策略自蒸馏分配视觉监督

    VisAlloc: Allocating Visual Supervision for On-Policy Self-Distillation

    [https://arxiv.org/abs/2611.25069](https://arxiv.org/abs/2611.25069)

    VisAlloc提出一种双层框架，通过样本级的正负裁剪成对教师过滤与词元级的EMA教师视角敏感性反事实重加权，将视觉监督精准分配到最需要的样本和词元上，从而提升多模态大语言模型的视觉推理能力。

    

    在线策略自蒸馏（OPSD）为提升多模态大语言模型（MLLM）的视觉推理能力提供了一种有前景的方法。然而，并非所有样本都能从特权视觉引导中平等受益，且统一的词元监督可能会稀释基于视觉的信号。我们提出VisAlloc，一个将样本级视觉过滤与词元级反事实重加权相结合的双层框架。一个固定的成对教师过滤器保留那些在以证据为中心的正向裁剪图像上回答正确、但在空间偏移的负向裁剪图像上回答错误的样本。在词元层面，两个EMA教师对同一学生生成的响应进行打分，其绝对对数概率差异用于衡量视角敏感性。额外的监督针对通过敏感性或教师分布分歧准入的词元，并采用共享的冻结先验、模板二元组以及师生分歧门控。逆频率重分配保留……

    arXiv:2611.25069v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) offers a promising approach to improving visual reasoning in multimodal large language models (MLLMs). However, not all samples benefit equally from privileged visual guidance, and uniform token supervision can dilute visually grounded signals. We propose VisAlloc, a dual-level framework combining sample-level visual filtering with token-level counterfactual reweighting. A fixed paired-teacher filter retains samples answered correctly from an evidence-centered positive crop but incorrectly from a spatially displaced negative crop. At the token level, both EMA teachers score the same student-generated response, and their absolute log-probability differences measure view sensitivity. Additional supervision targets tokens admitted by sensitivity or teacher-distribution disagreement, with shared frozen-prior, template-bigram, and student--teacher divergence gates. Inverse-frequency redistribution preserve
    
[^238]: DVBench：面向真实世界UVM设计验证智能体的端到端基准测试

    DVBench: End-to-End Benchmark for Real-World UVM Design Verification Agents

    [https://arxiv.org/abs/2611.25066](https://arxiv.org/abs/2611.25066)

    本文提出了首个面向真实世界UVM设计验证智能体的端到端基准测试DVBench，包含涵盖9个功能领域和5个难度级别的95个任务，要求智能体独立完成完整的验证工作流程，并通过编译、覆盖率和错误检测等多个维度综合评估。

    

    设计验证（DV）仍然是硬件开发中的主要瓶颈，其中基于通用验证方法学（UVM）的大规模验证工作高度依赖人力。大语言模型（LLM）和智能体系统的最新进展为DV自动化提供了新的机遇。然而，现有的基准测试主要构建相互解耦的子任务，这与真实世界的工作流程存在偏差。在本文中，我们提出了DVBench，这是首个面向真实世界UVM设计验证智能体的端到端基准测试。DVBench包含95个任务，涵盖9个功能领域和5个难度级别，覆盖多样化的验证场景和设计规模。每个任务都要求智能体独立完成完整的验证工作流程。DVBench通过编译、覆盖率和错误检测来评估最终的验证平台，同时评估包括特性列表、测试计划和参考模型在内的中间产物。我们评估了

    arXiv:2611.25066v1 Announce Type: cross  Abstract: Design verification (DV) remains a major bottleneck in hardware development, with large-scale verification using the Universal Verification Methodology (UVM) being highly labor-intensive. Recent advances in large language models (LLMs) and agentic systems offer new opportunities for DV automation. However, existing benchmarks primarily construct mutually decoupled subtasks, which deviates from the real-world workflow. In this paper, we present DVBench, the first end-to-end benchmark for Real-World UVM design verification agents. DVBench comprises 95 tasks across 9 functional domains and 5 difficulty levels, covering diverse verification scenarios and design scales. Each task requires agents to independently complete the full verification workflow. DVBench evaluates final testbenches via compilation, coverage, and bug detection, while assessing intermediate artifacts including feature lists, test plans, and reference models. We evaluate
    
[^239]: 弥合分析与设计之间的鸿沟：面向自主实验室的授权受限智能体

    Closing the Analyze-to-Design Gap: An Authorization-Bounded Agent for Self-Driving Laboratories

    [https://arxiv.org/abs/2611.25043](https://arxiv.org/abs/2611.25043)

    该论文提出一种将授权受限的AI智能体嵌入自主实验室DMTA循环“分析-设计”接口的架构，使智能体能在人类设定的授权边界内自主完成测量选择与数据解读，并保证所有改变状态的操作均可授权、验证和审计。

    

    人工智能智能体正日益被整合进自主实验室（SDL）中，用于规划实验和解读结果。然而，仅有分析推理能力并不能使实验室实现自主化：只有当智能体能够在无人监督的情况下被信任去操作物理硬件，并且懂得何时应当交由人类决策时，自主实验室才真正具备自主性。在此，我们提出了一种架构，将智能体集成到“设计-制造-测试-分析”（DMTA）循环中“分析-设计”的接口处，实现由智能体驱动的测量选择与数据解读，同时确保所有改变状态的操作均经过授权、验证并可审计。我们在液相色谱-质谱联用（LC-MS）仪器上展示了这一架构，应用于合成有机染料的氧化剂筛选实验活动。在人类定义的实验活动限制范围内，该智能体提交测量任务、解读质谱（MS）与二极管阵列检测器的数据……

    arXiv:2611.25043v1 Announce Type: new  Abstract: Artificial intelligence (AI) agents are increasingly integrated into self-driving laboratories (SDLs) to plan experiments and interpret results. However, analytical reasoning alone does not make a laboratory autonomous: autonomy in an SDL is achievable only when the agent can be trusted to act on physical hardware unsupervised and knows when to defer to a human. Here, we present an architecture that integrates an agent at the Analyze-Design interface of the Design-Make-Test-Analyze (DMTA) cycle, enabling agent-driven measurement selection and data interpretation while ensuring that state-changing operations are authorized, validated, and auditable. We demonstrate this architecture on a liquid chromatography-mass spectrometry (LC-MS) instrument in an oxidant-screening campaign for synthesizing an organic dye. Within human-defined campaign limits, the agent submitted measurement jobs, interpreted mass spectrometry (MS) and diode-array dete
    
[^240]: 所有VC类的近二次样本压缩

    Near-Quadratic Sample Compression for All VC Classes

    [https://arxiv.org/abs/2611.25011](https://arxiv.org/abs/2611.25011)

    本文通过构造大小为 O(d²log²d) 的样本压缩方案，证明了Floyd和Warmuth提出的弱样本压缩猜想，即每个VC维为 d 的二值概念类都存在多项式大小的样本压缩方案。

    

    样本压缩猜想是学习理论中一个长期悬而未决的问题，起源于Littlestone和Warmuth在20世纪80年代的工作。其核心在于简化与泛化之间的联系：如果一个与训练样本一致的预测器仅通过该样本的一小部分来定义，那么它很可能具有泛化能力。样本压缩方案将这一思想形式化，许多自然的学习算法都可以通过这种方式进行分析。Floyd和Warmuth（1995）提出的样本压缩猜想断言，每个VC维为 $d$ 的二值概念类都存在大小为 $O(d)$ 的样本压缩方案；一个较弱的猜想只要求方案大小为 $d$ 的多项式。这两个猜想至今仍未解决：第一个仅依赖于 $d$ 的界由Moran和Yehudayoff（2016）给出，但它是 $d$ 的指数级。我们通过构造一个大小为 $O(d^2 \log^2 d)$ 的样本压缩方案，证明了较弱的猜想。

    arXiv:2611.25011v1 Announce Type: cross  Abstract: The sample compression conjecture is a longstanding open problem in learning theory, originating in the work of Littlestone and Warmuth in the 1980s. At its heart is a connection between simplification and generalization: if a predictor consistent with a training sample is defined using only a small part of that sample, then it is likely to generalize. Sample compression schemes formalize this idea, and many natural learning algorithms can be analyzed this way.   The sample compression conjecture of Floyd and Warmuth (1995) asserts that every binary concept class of VC dimension $d$ admits a sample compression scheme of size $O(d)$; a weaker conjecture asks only for size polynomial in $d$. Both have remained open: the first bound depending only on $d$, due to Moran and Yehudayoff (2016), is exponential in $d$. We prove the weaker conjecture by constructing a sample compression scheme of size $O(d^2 \log^2 d)$.
    
[^241]: 特征降维技术对对抗攻击的影响研究

    On the Impact of Feature Reduction Techniques on Adversarial Attacks

    [https://arxiv.org/abs/2611.25004](https://arxiv.org/abs/2611.25004)

    本研究首次系统分析了特征降维技术（如减少维度）在APT数据集上遭受投毒、后门和通用对抗扰动攻击时，对支持向量机、决策树、随机森林等多种机器学习模型对抗鲁棒性的影响。

    

    对抗攻击通过利用机器学习模型固有的脆弱性，对其构成了重大威胁。研究人员建立了一个名为对抗机器学习的跨学科领域，以探索这些脆弱性并制定相应的防御策略。特征降维策略被用于降低数据维度，以减少训练时间、避免维度灾难、防止过拟合并缓解冗余特征。然而，特征降维在不同分类器和攻击场景下对对抗鲁棒性的影响仍未得到充分探索。本研究旨在分析特征降维技术对多样化机器学习模型对抗鲁棒性的影响，研究对象是在APT数据集上遭受投毒攻击、后门攻击和通用对抗扰动攻击的模型。这些模型包括支持向量机、决策树、随机森林、多……（摘要原文在此处截断）

    arXiv:2611.25004v1 Announce Type: cross  Abstract: Adversarial attacks have posed significant threats to machine learning models by exploiting their inherent vulnerabilities. Researchers have established an interdisciplinary field called adversarial machine learning to explore these vulnerabilities and develop corresponding defense strategies. Feature reduction strategies are used to decrease the dimensionality of the data in order to reduce training time, avoid the curse of dimensionality, prevent overfitting, and mitigate redundant features. However, their effect on adversarial robustness across different classifiers and attack scenarios remains insufficiently explored. This study aims to analyze the effect of feature reduction techniques on the adversarial robustness of a diverse set of machine learning models under poisoning, backdoor, and universal adversarial perturbation attacks on the APT dataset. These models include support vector machine, decision tree, random forest, multi-
    
[^242]: 一个语言模型多样性采样模型

    A Model of Diverse Sampling from Language Models

    [https://arxiv.org/abs/2611.25003](https://arxiv.org/abs/2611.25003)

    该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。

    

    语言模型通常能产生高质量的个体样本，但其样本集合往往质量不佳，重复样本聚集在相同的语义模式附近。我们将多样性生成形式化为从行列式点过程（DPP）中采样大小为 $K$ 的完整字符串子集，其中似然编码条目质量，嵌入几何结构编码相似输出之间的排斥作用。由于对所有字符串进行精确推理是不可行的，我们提出 PoolDPP：从一个易于处理的提议分布中抽取有限的候选池，通过重要性加权使候选偏向目标质量，然后在诱导的池核上运行 K-DPP 采样器；贪心 MAP 变体则提供确定性的选择方式。我们证明了一个具有显式常数的有限样本界：采样器输出与目标 $K$-DPP 之间的 KL 散度随池大小 $N$ 以 O(1/N) 的速度衰减。实证结果表明，PoolDPP 在开放式生成任务上扩展了质量-多样性边界，并提高了精确率-召回率……（摘要截断）

    arXiv:2611.25003v1 Announce Type: cross  Abstract: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recal
    
[^243]: 模型知道但未说出的事：通过言语化思维监测医学大语言模型

    What the model knows but does not say: monitoring medical LLMs through verbalized thoughts

    [https://arxiv.org/abs/2611.24994](https://arxiv.org/abs/2611.24994)

    该论文提出用自然语言自编码器将医学大语言模型的内部激活翻译成可读文本（“思维通道”），证明这一言语化的内部状态比模型输出的答案更能有效检测针对临床模型的隐蔽操纵与注入攻击，且在干净输入上的误报率接近于零。

    

    arXiv:2611.24994v1 公告类型：交叉 摘要：目前保障临床语言模型安全的努力主要集中在进入和离开模型的内容上，但一个被操纵的模型可能会返回一个看起来完全正常的答案。在本研究中，我们展示了模型自身的内部状态——以通俗语言呈现出来——能够暴露输出所掩盖的操纵行为。我们使用自然语言自编码器，将多模态大语言模型（Gemma-3-27B）的激活值翻译成文本，并让一个对输入不知情的评判者分别阅读这个言语化的内部状态（即“思维通道”）和模型给出的答案。在两个临床攻击目标（遗漏病理发现、隐蔽推荐药物）、两种输入模态和两种传递机制的设置下，在全部八个注入条件中，思维通道捕获注入攻击的频率都显著高于答案通道（p<0.05），同时两个通道在干净输入上的假阳性率都接近于零；关于遗漏攻击的结果在 Llama-3.3-70B 上也得到了验证。思维通道还能够（原文摘要此处被截断）

    arXiv:2611.24994v1 Announce Type: cross  Abstract: Efforts to keep clinical language models safe have concentrated on what enters and leaves the model, but a manipulated model can return an answer that looks entirely ordinary. Here we show that the model's own internal state, rendered in plain language, exposes manipulations the output conceals. Using a natural-language autoencoder, we translate the activations of a multimodal LLM (Gemma-3-27B) into text and let an input-blind judge read this verbalized internal state (the thoughts channel) and the model's answer separately. Across two clinical attack objectives (omitting a pathology, covertly recommending a drug), two input modalities, and two delivery mechanisms, the thoughts channel caught the injection significantly more often than the answer in all eight injected conditions (p<0.05), while both channels held near-zero false-positive rates on clean inputs; the omission result carried over to Llama-3.3-70B. The thoughts channel also
    
[^244]: DL101：神经网络模型的表示能力

    DL101 The Representation Capability of Neural Network Models

    [https://arxiv.org/abs/2611.24980](https://arxiv.org/abs/2611.24980)

    本文系统梳理了通用近似定理这一神经网络表示能力的数学基石，指出非多项式激活函数是实现通用逼近的必要条件，并强调这些定理本身并不提供关于效率、训练收敛与泛化能力的保证。

    

    arXiv:2611.24980v1 公告类型：cross 摘要：通用近似定理（Cybenko、Funahashi、Hornik 与 Leshno）为神经网络为何足够强大、能够建模复杂非线性现象提供了数学依据。浅层网络可以逼近紧集上的任意连续函数。Leshno 定理阐明了非多项式激活函数对于实现通用性是必要的，这解释了为什么 ReLU、逻辑斯蒂（sigmoid）、双曲正切以及其他常见激活函数是有效的（而纯多项式激活则不行）。然而，逼近定理并未提供效率、训练或泛化方面的任何保证。它们既没有说明为达到给定精度需要多少神经元，也没有说明梯度下降能否找到正确的参数，或者在有限数据上训练的网络是否能够良好地泛化。Barron 为特定函数类提供了定量的误差界（即对傅里叶范数有界的函数给出了逼近速率），而 Montúfar 则是第一个证明……（摘要到此截断）

    arXiv:2611.24980v1 Announce Type: cross  Abstract: Universal approximation theorems (Cybenko, Funahashi, Hornik & Leshno) provide the mathematical justification for why neural networks are powerful enough to model complex, nonlinear phenomena. Shallow networks can approximate any continuous function on compact sets. Leshno's theorem clarified that non-polynomial activations are necessary for universality, which explains why ReLU, logistic, hyperbolic tangent, and other common activations work (but pure polynomials do not).   However, approximation theorems provide no efficiency, training, nor generalization guarantees. They do not indicate how many neurons are needed for a given accuracy, whether gradient descent will find the right parameters, or whether networks trained on finite data will generalize well.   Barron provided quantitative error bounds for particular function classes (approximation rates for functions with bounded Fourier norm), whereas Mont\'ufar was the first to show 
    
[^245]: Trace2ST：利用大语言模型生成与验证，从输入/输出轨迹中恢复PLC结构化文本程序

    Trace2ST: Recovering PLC Structured Text from I/O Traces with LLM Generation and Verification

    [https://arxiv.org/abs/2611.24979](https://arxiv.org/abs/2611.24979)

    本文提出“生成—回放—反馈”流水线，使大语言模型仅凭PLC的输入/输出轨迹记录即可恢复出可执行的结构化文本（ST）程序，并通过逐扫描周期回放比对完成程序验证。

    

    可编程逻辑控制器（PLC）是驱动电力与流程工厂运行的工业计算机，其上运行的程序源代码往往已不复存在；然而，控制器的每一个输入和输出都会被持续记录。因此，本文探讨大语言模型（LLM）能否仅凭这类输入/输出记录（轨迹）恢复出以结构化文本（ST，即IEC 61131-3标准PLC语言）编写的可执行程序，以及同样重要的是，如何对恢复出的程序进行验证。为回答这一问题，我们构建了“生成—回放—反馈”流水线：首先，LLM根据轨迹的压缩摘要编写候选程序，全程不接触原始程序；随后，候选程序按单个控制周期（扫描）逐步执行并与记录进行比对，一旦出现不匹配，便将轨迹中对应的部分返回给LLM，最多进行三次修订。（原摘要在此处截断）

    arXiv:2611.24979v1 Announce Type: cross  Abstract: Programmable logic controllers (PLCs), the industrial computers that operate power and process plants, often run programs whose source code is no longer available; however, every input and output of the controller is logged continuously. This paper therefore asks whether a large language model (LLM) can recover an executable program in Structured Text (ST), the standard PLC language of IEC 61131-3, from such input/output records (traces) alone and, equally important, how the recovered program can be verified. To answer this question, we build a generate-replay-feedback pipeline. First, the LLM writes a candidate program from a compressed summary of the trace, without ever seeing the original program. The candidate is then executed one control cycle (scan) at a time and compared with the recording, and whenever a mismatch occurs the corresponding part of the trace is returned to the LLM, for at most three revisions. Because every result
    
[^246]: CANDID：面向本地计算机使用代理规范执行的候选驱动干预框架

    CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents

    [https://arxiv.org/abs/2611.24977](https://arxiv.org/abs/2611.24977)

    CANDID是一个执行控制框架，通过循环恢复机制和全体一致的DONE门控策略，在易失败的关键节点对本地计算机使用代理实施规范化的执行级干预，显著提升了资源受限环境下的任务成功率。

    

    我们提出了CANDID（Candidate-Driven Intervention for Disciplined Execution，候选驱动干预的规范执行），这是一个执行控制框架，旨在提高本地计算机使用代理（CUA）在资源受限环境中的可靠性。虽然本地CUA在单步动作预测方面具备强大的基线能力，但它们经常在关键节点因僵化的动作循环或过早的DONE决策而失败。CANDID并未通过文本反思施加不精确的认知负担，而是将并行测试时扩展重新用于在稀疏的、易失败的节点上实施规范化的执行级干预。我们引入了一种循环恢复机制来过滤掉重复行为的候选方案，以及一种保守的、要求全体一致的DONE门控策略，即必须达成绝对共识才能终止任务。在OSWorld上的实验表明，CANDID的结构化控制显著提高了成功率，以极小的额外开销超越了基于反馈的替代方案。

    arXiv:2611.24977v1 Announce Type: cross  Abstract: We present CANDID (Candidate-Driven Intervention for Disciplined Execution), an execution-control framework designed to enhance the reliability of local computer-use agents (CUAs) in resource-constrained settings. While local CUAs feature robust baseline proficiency in single-step action prediction, they frequently fail at critical junctions due to rigid action loops or premature DONE decisions. Rather than imposing imprecise cognitive loads through textual reflection, CANDID repurposes parallel test-time scaling into a disciplined, execution-level intervention at sparse, failure-prone points. We introduce a loop-recovery mechanism to filter candidates away from repeated behaviors and a conservative, unanimous DONE gating policy that requires absolute consensus to terminate. Experiments on OSWorld demonstrate that CANDID's structural control significantly improves success rates, outperforming feedback-based alternatives with minimal ex
    
[^247]: No-Harm-Jury：一种基于陪审团的医疗AI响应多维度伤害评估框架

    No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses

    [https://arxiv.org/abs/2611.24972](https://arxiv.org/abs/2611.24972)

    该论文提出No-Harm-Jury框架，利用五个不同的大语言模型作为独立陪审员，从七个伤害维度对医疗AI响应进行自动化多维度评估，并通过中位数聚合稳健地抑制个体评审偏差。

    

    我们提出了No-Harm-Jury，一个基于陪审团的自动化评估框架，它使用五个不同的语言模型作为独立陪审员，从七个概念上截然不同的伤害维度对AI生成的医疗响应进行评分。由Ministral-14B针对从PubMedQA、MedQA和MedMCQA中抽取的3,000个问题生成的响应，由五位陪审员在信息伤害、社会伤害、心理伤害、自主权伤害、经济伤害、隐私伤害和认知伤害等维度上进行评分；151个空响应（98个来自MedQA，53个来自MedMCQA）被排除，留下2,849个有效样本。中位数聚合方法能稳健地抑制单个陪审员的偏差，同时通过互补专业化保持覆盖率。目前没有任何现有系统能够结合No-Harm-Jury的全部五个特性：自动化可扩展评估、五个不同的LLM陪审员、稳健的中位数聚合（带有可靠性加权扩展）、七个概念上截然不同的伤害维度，以及应用于医疗领域。

    arXiv:2611.24972v1 Announce Type: cross  Abstract: We present No-Harm-Jury, a jury-based automated evaluation framework that scores AI-generated medical responses across seven conceptually distinct harm dimensions using a panel of five diverse language models as independent jurors. Responses generated by Ministral-14B on 3,000 questions drawn from PubMedQA, MedQA, and MedMCQA are scored by five jurors across dimensions of informational, social, psychological, autonomy, economic, privacy, and epistemic harm; the 151 empty responses (98 MedQA, 53 MedM- CQA) are excluded, leaving 2,849 valid samples. The median aggregation robustly suppresses individual juror biases while maintaining coverage through complementary specialisation. No existing system combines all five prop- erties of No-Harm-Jury: automated scalable evaluation, five diverse LLM jurors, robust me- dian aggregation (with a reliability-weighted extension), seven conceptually distinct harm dimensions, and application to medical
    
[^248]: 小示例的投资回报率：Token预算下的上下文学习优化

    The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets

    [https://arxiv.org/abs/2611.24971](https://arxiv.org/abs/2611.24971)

    该论文发现在固定Token预算下，将预算用于更多较短的示例而非少量较长示例，能在问答任务中实现更优的上下文学习效果。

    

    上下文学习（ICL）无需更新模型参数即可提升大语言模型在推理阶段的性能。然而，这些收益并非没有代价：在实际部署中，更长的提示词会直接增加推理成本和延迟。因此，在固定的Token预算下，提示词构建成为一个资源分配问题。虽然先前的工作大多关注ICL中应包含多少个示例，我们研究了与之互补的问题：可用的预算是更好地花费在少量长示例上，还是花费在更多短示例上。我们在问答（QA）任务中研究这一问题，将候选示例按长度分桶，并在固定预算下从相同长度范围内的示例构建提示词。在两个问答基准（Beat the AI 和 Natural Questions）上，使用六个指令微调的大语言模型进行的实验表明，较短的示例通常是最有效的……

    arXiv:2611.24971v1 Announce Type: cross  Abstract: In-context learning (ICL) improves large language model performance at inference time without updating model parameters. However, these gains are not free: in practical deployments, longer prompts directly increase inference cost and latency. Under a fixed token budget, prompt construction therefore becomes a resource-allocation problem. While prior work has often focused on how many examples to include in ICL, we study the complementary question of whether the available budget is better spent on a few long demonstrations or on a larger number of shorter ones. We study this aspect in question answering (QA), by organizing candidate demonstrations into length-based bins and constructing prompts from examples within the same length range under fixed budgets. Experiments on two QA benchmarks (\textit{Beat the AI} and \textit{Natural Questions}), using six instruction-tuned LLMs, show that shorter demonstrations are generally the most effe
    
[^249]: HireWatch：评估大型语言模型在情境压力下对美国就业法律的合规性

    HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure

    [https://arxiv.org/abs/2611.24970](https://arxiv.org/abs/2611.24970)

    本研究提出HireWatch评估框架，通过构建200道合法与非法面试问题题库并对18个LLM进行约10万次情境压力试验，发现LLM对美国就业法律的合规性十分脆弱，容易在压力下选择非法面试问题。

    

    大型语言模型（LLM）在招聘流程中的部署日益增多，然而现有评估普遍忽视了其对法律规则的合规性。为填补这一空白，我们推出了HireWatch，这是一个评估LLM是否符合美国关于求职者可被提问范围的法律规定的评估方法。我们构建了一个包含200个合法与非法面试问题的题库，其法律分类依据美国联邦法规、规章及政府机构指南。我们指示LLM在为求职面试做准备时从该题库中选择问题，涵盖多种职业和情境，并统计LLM所选非法问题的比例。在约100,000次试验中对18个闭源和开源权重的LLM在不同类型情境压力下的评估表明，LLM的法律合规性十分脆弱。虽然LLM在低压力情境下通常能避免选择非法问题，但法律违规……（摘要原文在此处截断）

    arXiv:2611.24970v1 Announce Type: cross  Abstract: The deployment of large language models (LLMs) in hiring pipelines is growing, yet existing evaluations generally overlook their compliance with legal rules. To fill this gap, we introduce HireWatch, an evaluation that assesses whether LLMs comply with U.S. laws that govern the questions that job candidates may be asked. We construct a 200-item bank of lawful and unlawful interview questions, whose legal classification is based on U.S. federal statutes, regulations, and agency guidance. We instruct LLMs to select questions from this bank in preparing for job interviews, across a set of various occupations and contexts, and report the fraction of unlawful questions that LLMs select. Evaluation of 18 closed and open-weight LLMs across ~100,000 trials under different types of contextual pressure reveals that LLMs' legal compliance is brittle. While LLMs generally avoid selecting unlawful questions in low-pressure contexts, legal violation
    
[^250]: KV缓存淘汰的全局重要性估计

    Global Importance Estimation for KV Cache Eviction

    [https://arxiv.org/abs/2611.24969](https://arxiv.org/abs/2611.24969)

    PRKV是一种无需训练的KV缓存淘汰方法，通过在注意力转移图上应用个性化PageRank估计token的多跳全局重要性，即使压缩率高达90%也能保持竞争性任务性能并显著降低峰值内存。

    

    键值（KV）缓存的内存消耗仍然是长上下文大语言模型推理的核心瓶颈。我们提出了PRKV，一种无需训练的KV缓存淘汰方法，它通过在注意力诱导的转移图上应用个性化PageRank来估计token的多跳全局重要性。这使得重要性传播能够超越基于近期性或瞬时注意力的方法，且无需修改模型。我们使用多个大语言模型系列，在长上下文理解、稀疏检索和思维链推理任务上对PRKV进行了基准测试。在从中等到激进（包括高达90%）的压缩比率下，PRKV相对于已有的淘汰基线方法取得了具有竞争力的任务性能，同时以较低的端到端开销降低了峰值内存使用。这些结果表明，多跳全局影响力估计为可扩展的长上下文推理带来了良好的准确性与内存之间的权衡。

    arXiv:2611.24969v1 Announce Type: cross  Abstract: Memory consumption of the key-value (KV) cache remains a central bottleneck for long-context LLM inference. We introduce PRKV, a training-free KV eviction method that estimates multi-hop global token importance by applying Personalized PageRank over attention-induced transition graphs. This enables importance propagation beyond recency or instantaneous attention, without modifying the model. We benchmark PRKV on long-context understanding, sparse retrieval, and chain-of-thought reasoning tasks using multiple LLM families. Under moderate to aggressive compression ratios, including up to 90%, PRKV achieves competitive task performance against established eviction baselines while reducing peak memory usage with low end-to-end overhead. These results demonstrate that multi-hop global influence estimation yields a favorable accuracy--memory trade-off for scalable long-context inference.
    
[^251]: 语言模型的物理学：第4.2部分，在合成预训练与现实共鸣的规模下研究Canon层

    Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality

    [https://arxiv.org/abs/2611.24963](https://arxiv.org/abs/2611.24963)

    在1-2T词元的大规模预训练中，Canon层通过局部词元混合持续提升Transformer性能（8B下节省约30%预训练词元），且加上Canon后最简单的门控线性注意力模型即可匹敌Gated DeltaNet并超越Mamba2。

    

    标准基准测试能否察觉到真正重要的架构差异？我们通过64次从零开始的预训练实验来研究这一问题，涵盖Transformer和三个线性模型家族，参数规模为1.3B至8B，训练数据为1-2T的Nemotron-CC词元，每次8B规模的训练成本约为10万美元。在学术规模的预训练中（1.3B参数、100B词元），经过精心调优后，架构之间的差异几乎可以完全消失。而在1-2T词元的规模下，这些差异变得可以测量：Canon层通过增加局部词元混合，能够持续提升Transformer的性能——在8B参数下，达到相同MMLU准确率可节省约30%的预训练词元。我们还提出了GLA5和GDN2，通过低秩输出门控分别改进了门控线性注意力和Gated DeltaNet。令人惊讶的是，加上Canon之后，最简单的线性模型——门控线性注意力，即使在8B/1T的规模下也能匹敌Gated DeltaNet并超越Mamba2。因此，Gated DeltaNet和Mamba2的额外机制可能只是起到了……

    arXiv:2611.24963v1 Announce Type: cross  Abstract: Do standard benchmarks see the architectural differences that matter? We study it with 64 from-scratch pretraining runs of Transformers and three linear-model families at 1.3-8B parameters on 1-2T Nemotron-CC tokens, at a cost of roughly 100k USD per 8B run.   In academic-scale pretraining (1.3B parameters, 100B tokens), architectural differences can all but disappear when the models are carefully tuned. Instead, at 1-2T tokens they become measurable: Canon layers, which add local token mixing, consistently improve Transformers, saving about 30% of pretraining tokens for the same MMLU accuracy at 8B. We also introduce GLA5 and GDN2, which improve gated linear attention and Gated DeltaNet with low-rank output gates. Surprisingly, with Canon, the simplest linear model, gated linear attention, matches Gated DeltaNet and outperforms Mamba2 even at 8B/1T. Therefore, the extra machinery of Gated DeltaNet and Mamba2 may do little more than mi
    
[^252]: LLM复杂度：迈向大语言模型推理的计算复杂度视角

    LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference

    [https://arxiv.org/abs/2611.24943](https://arxiv.org/abs/2611.24943)

    该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。

    

    计算复杂度理论为刻画计算问题的效率提供了一个原则性框架，而大语言模型（LLM）同样可以被视作执行“输入—推理时计算—输出”这一结构化过程的计算系统。在此背景下，生成token的数量成为衡量推理时计算量的一个自然且可直接观测的代理指标。然而，我们证明了LLM token长度与经典时间复杂度之间这一颇具吸引力的类比并不成立。我们否定了“推理时token使用量与经典时间复杂度单调一致”这一假设，并展示了其根本性分歧源于LLM在推理过程中实际分配计算的方式。为此，我们提出了LLM复杂度（LLM Complexity）这一新指标，通过刻画准确率-token曲线所界定的面积，从LLM的视角量化问题复杂度。我们发现……

    arXiv:2611.24943v1 Announce Type: cross  Abstract: Computational complexity theory offers a principled framework for characterizing the efficiency of computational problems, and Large Language Models (LLMs) can similarly be viewed as computational systems executing a structured process of input, inference-time compute, and output. In this context, the number of generated tokens emerges as a natural and directly observable proxy for inference-time compute. However, we show that the tempting analogy between LLM token length and classical time complexity does not hold. We disprove the hypothesis that inference-time token usage aligns monotonically with classical time complexity, demonstrating a fundamental divergence grounded in how LLMs actually allocate computation during inference. To this end, we introduce LLM Complexity, a new metric designed to quantify problem complexity from the perspective of LLMs by characterizing the area bounded above by the accuracy-token curve. We find that 
    
[^253]: WorldRuler：可配置规则下的交互式世界建模

    WorldRuler: Interactive World Modeling under Configurable Rules

    [https://arxiv.org/abs/2611.24869](https://arxiv.org/abs/2611.24869)

    WorldRuler通过可执行的3D规则图组织规则定义、依赖关系与对象状态，并闭环连接规则执行、视频生成和感知反馈，使生成式世界模型能够遵循用户自定义规则，解决了规则-状态持久化与规则到视觉实现两大挑战。

    

    规则支配着玩家行为如何改变交互式世界。在相似的场景和动作空间中，不同的规则会创造出截然不同的游戏体验。为了实现可定制的游戏玩法，生成式世界模型必须能够遵循用户定义的规则，而不仅仅局限于现实世界的行为。然而，实现用户定义的规则面临两大挑战：规则-状态持久化和规则到视觉的实现。前者要求在保存和加载过程中保持规则定义、依赖关系和世界状态，使恢复后的世界遵循既定规则；后者要求准确地将规则引发的变化体现在指定对象和属性上，尤其是当受影响对象不可见时。本文提出WorldRuler，它将规则定义、依赖关系和对象状态组织在一个可执行的3D规则图中，并将规则执行、视频生成和感知反馈连接成一个闭环。该图支持联合保存与恢复等功能。

    arXiv:2611.24869v1 Announce Type: new  Abstract: Rules govern how player actions change interactive worlds. Different rules create distinct gameplay experiences within similar scenes and action spaces. For customizable gameplay, generative world models must follow user-defined rules beyond real-world behaviors. However, realizing user-defined rules presents two challenges: Rule-State Persistence and Rule-to-Visual Realization. The former requires preserving rule definitions, dependencies, and world states across saving and loading, so that restored worlds follow the established rules. The latter requires accurately realizing rule-induced changes on designated objects and attributes, particularly when affected objects are unseen. In this paper, we propose WorldRuler, which organizes rule definitions, dependencies, and object states in an executable 3D rule graph and connects rule execution, video generation, and perceptual feedback in a closed loop. The graph supports joint saving and r
    
[^254]: 面向小语言模型更结构化医学解释的自奖励强化学习

    Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models

    [https://arxiv.org/abs/2611.24650](https://arxiv.org/abs/2611.24650)

    该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。

    

    小语言模型（SLM）可以在普通硬件上运行，并支持保护隐私的本地化临床部署，但与大型语言模型相比，其生成的医学解释往往简短且结构欠佳。近期的监督式方法（如 Meerkat）将思维链推理迁移到 7B-8B 参数规模的骨干模型中，但其训练目标并未直接针对所生成解释的风格或组织结构。基于指令微调的 Mistral-7B 骨干模型，我们研究了轻量级的强化学习（RL）步骤能否在无需人工偏好标注的情况下改善解释的结构。我们在 2,000 个源自 Meerkat 的医学提示上应用了结合 LoRA 适配的近端策略优化（PPO），并由一个简单的自监督奖励引导，该奖励偏好更长、词汇更丰富且标点规范的回答，同时惩罚过短的回答。在单个 NVIDIA A100 GPU 上训练约 170 个 PPO 步骤后，得到了（摘要在此处截断）……

    arXiv:2611.24650v1 Announce Type: new  Abstract: Small language models (SLMs) can run on commodity hardware and support privacy-preserving, on-premises clinical deployment, but their medical explanations are often short and poorly structured compared with large language models. Recent supervised approaches such as Meerkat transfer chain-of-thought reasoning into 7B-8B backbones, yet their training objective does not directly target the style or organization of the generated rationales. Building on the instruction-tuned Mistral-7B backbone, we investigate whether a lightweight reinforcement learning (RL) step can improve explanation structure without human preference labels.We apply Proximal Policy Optimization (PPO) with LoRA adaptation to 2,000 Meerkat-derived medical prompts, guided by a simple self-supervised reward that favors longer, lexically diverse, and well-punctuated responses while penalizing very short ones. Training for about 170 PPO steps on a single NVIDIA A100 GPU yield
    
[^255]: FedE2E：面向端到端自动驾驶的带闭环评估的一次性联邦学习

    FedE2E: One-Shot Federated Learning for End-to-End Autonomous Driving with Closed-Loop Evaluation

    [https://arxiv.org/abs/2611.24254](https://arxiv.org/abs/2611.24254)

    提出首个面向端到端自动驾驶的一次性联邦学习方法FedE2E，通过多客户端校准、锚点特征对齐和词元级共识蒸馏从异构模型中蒸馏知识，在Bench2Drive闭环评估中一次性驾驶分数达55.85，超过先前工作4.07分。

    

    联邦学习能够在不共享数据集的情况下实现协同模型训练。然而，现有的自动驾驶联邦学习方法通常需要多轮通信且缺乏闭环评估，而一次性联邦学习方法则主要专注于感知任务。为了解决这些局限性，我们提出了FedE2E，一种面向端到端自动驾驶的一次性联邦学习方法。该方法包含多客户端校准、基于锚点的特征对齐以及词元级共识蒸馏，以从异构模型中蒸馏知识。在Bench2Drive上的闭环评估证明了FedE2E在一次性和多轮联邦学习中的有效性。在一次性设置下，它取得了55.85的驾驶分数，比先前工作高出4.07分；在多轮设置下，经过五轮通信后达到60.35，是同等通信轮数下所有对比方法中的最佳性能。

    arXiv:2611.24254v1 Announce Type: new  Abstract: Federated learning enables collaborative model training without sharing datasets. However, existing federated learning methods for autonomous driving typically require multi-round communication and lack closed-loop evaluation, while one-shot methods mainly focus on perception tasks. To address these limitations, we propose \method, a one-shot federated learning method for end-to-end autonomous driving. It comprises multi-client calibration, anchor-based feature alignment, and token-wise consensus distillation to distill knowledge from heterogeneous models. Closed-loop evaluation on Bench2Drive demonstrates the effectiveness of \method in both one-shot and multi-round federated learning. In the one-shot setting, it achieves a Driving Score of 55.85, which is 4.07 points higher than prior work. In the multi-round setting, it reaches 60.35 after five communication rounds, the best performance among all compared methods under the same number
    
[^256]: ReflectScene：MLLM引导的闭环扩散模型用于3D室内场景生成

    ReflectScene:MLLM-Guided Closed-Loop Diffusion for 3D Indoor Scene Generation

    [https://arxiv.org/abs/2611.23845](https://arxiv.org/abs/2611.23845)

    提出ReflectScene框架，将MLLM的语义诊断以轻量级奖励信号实时融入扩散去噪过程，形成闭环反馈，生成更符合几何、功能与语义约束的3D室内场景。

    

    基于文本条件的3D室内场景生成需要生成满足几何、功能和语义约束的布局。然而，现有的扩散方法主要依赖静态的文本条件，限制了其评估演化中场景状态的能力，也难以将语义偏差转化为可执行的对象级修正。基于MLLM（多模态大语言模型）的事后细化方法虽能提供语义反馈，但与去噪过程相解耦，可能破坏场景的一致性。我们提出了ReflectScene，一个MLLM引导的闭环扩散框架。首先，我们引入了分层先验锚定的场景理解机制，将结构化布局与多视角视觉证据对齐，借助冻结的多模态大语言模型（MLLM）来解释对象特定的上下文和场景全局语义。其次，我们提出了MLLM引导的语义奖励评估，将MLLM对候选修正方案的诊断和评估提炼为轻量级奖励信号……

    arXiv:2611.23845v1 Announce Type: cross  Abstract: Text-conditioned 3D indoor scene generation requires layouts satisfying geometric, functional, and semantic constraints. However, existing diffusion methods rely primarily on static textual conditions, limiting their ability to assess evolving scenes and translate semantic discrepancies into executable object-level corrections. MLLM-based post-hoc refinement provides semantic feedback but remains decoupled from denoising, potentially disrupting scene coherence.We propose ReflectScene, an MLLM-guided closed-loop diffusion framework. First, we introduce Hierarchical Prior-Grounded Scene Understanding that aligns structured layouts with multi-view visual evidence, to interpret object-specific context and scene global semantics via a frozen multi-modal large language model (MLLM). Second, we propose MLLM-Guided Semantic Reward Evaluation to distill MLLM's diagnoses and assessments on possible candidate corrections into a lightweight Reward
    
[^257]: 用于视觉语言模型修复的权重空间编程

    Weight Space Programming for Vision Language Model Repair

    [https://arxiv.org/abs/2611.23615](https://arxiv.org/abs/2611.23615)

    语言模型无需梯度即可直接修复被植入故障的视觉语言模型权重：它能自主定位投影器中的缩放故障和视觉编码器中的噪声、推断缩放因子并将权重精确恢复至原始值，还能据此写出可修复同模型未见检查点的通用修复程序，其修复能力依赖于16位精度权重中残留的低位比特痕迹。

    

    在权重空间编程中，语言模型无需梯度即可直接编辑网络的权重。我们探究这类成功修复依赖于什么。我们在两个视觉语言模型SmolVLM-256M和LFM2-VL-450M的权重中植入故障，并将编程智能体所做的每次编辑与原始权重进行核对。在没有被告知查找位置的情况下，该智能体在两个模型的投影器中发现了被重新缩放的行、在视觉编码器中发现了噪声，推断出未知的缩放因子，将被重新缩放的大多数或全部权重恢复到其精确的原始值，并将准确率恢复到接近原始水平；它根据这些权重编写的程序无需反馈即可修复同一模型未见过的检查点。这些修复依赖于故障是如何被计算的痕迹。原始权重具有16位精度，而在32位算术中计算的故障会留下只有受损权重才携带的低位比特。将这些低位比特舍去会……（摘要在此处截断）

    arXiv:2611.23615v1 Announce Type: cross  Abstract: In weight-space programming, a language model edits a network's weights directly, without gradients. We ask what a successful repair of this kind relies on. We plant faults in the weights of two vision-language models, SmolVLM-256M and LFM2-VL-450M, and check each edit made by a coding agent against the original weights. Without being told where to look, the agent finds rescaled rows in the projector and noise in the vision encoder of both models, infers the unknown scale factors, restores most or all of the rescaled weights to their exact original values, and returns accuracy to near its original level; programs that it writes from such weights repair unseen checkpoints of the same model without feedback. These repairs depend on a trace of how the faults were computed. The original weights have 16-bit precision, and faults computed in 32-bit arithmetic leave low bits that only the damaged weights carry. Rounding those bits away leaves
    
[^258]: 紧凑视觉骨干网络中全局上下文与权重共享的适用位置

    Where Global Context and Weight Sharing Work in Compact Vision Backbones

    [https://arxiv.org/abs/2611.23446](https://arxiv.org/abs/2611.23446)

    该研究发现，在EfficientNet-B0骨干网络的P3、P4、P5阶段插入轻量级零门控残差自注意力适配器可将ImageNet-1k精度提升至74.02%，而单纯用BiFPN多尺度特征金字塔包装骨干则无法超越普通MBConv主干。

    

    紧凑型卷积骨干网络（如EfficientNet）通过局部的深度可分离卷积实现了出色的参数效率，但混合设计中仍有两个核心的结构性问题：全局自注意力在何处能超越纯卷积以提升识别性能，以及重复的模块能否在不损失表征能力的前提下共享权重？我们在ImageNet-1k（5万张图像的验证集）上通过受控的EfficientNet-B0实验对这两个问题进行了研究。首先，我们表明仅将分类骨干网络嵌入多尺度特征金字塔（BiFPN）中无法超越普通的MBConv主干（69.99%–71.81% vs. 普通B0的72.86%），直到配合阶段注意力后才有所提升（73.15%）。相比之下，保持MBConv主干不变，并在P3（28×28窗口式）、P4（14×14全局）和P5（7×7全局）阶段插入轻量级的零门控残差自注意力适配器，可独立达到74.02%的准确率，并优于逐位匹配配置的普通B0基线。

    arXiv:2611.23446v1 Announce Type: cross  Abstract: Compact convolutional backbones such as EfficientNet achieve strong parameter efficiency through local depthwise separable convolutions, yet two structural questions remain central to hybrid design: where does global self-attention improve recognition over pure convolution, and can repeated blocks share weights without losing representational capacity? We investigate both questions on ImageNet-1k (50,000-image validation) using controlled EfficientNet-B0 experiments. First, we show that wrapping a classification backbone in multi-scale feature pyramids (BiFPN) fails to outperform a plain MBConv trunk (69.99% - 71.81% vs. 72.86% plain B0) until paired with stage attention (73.15%). In contrast, keeping the MBConv trunk intact and inserting lightweight zero-gated residual self-attention adapters at stages P3 (28x28 windowed), P4 (14x14 global), and P5 (7x7 global) reaches 74.02% standalone and outperforms a bit-for-bit matched plain-B0 c
    
[^259]: 为什么亚4比特视觉CNN会崩溃以及混合精度恢复的极限在哪里

    Why Sub-4-Bit Vision CNNs Collapse and Where Mixed-Precision Recovery Stops

    [https://arxiv.org/abs/2611.23445](https://arxiv.org/abs/2611.23445)

    本研究系统剖析了视觉CNN在4比特以下量化时崩溃的机理，发现崩溃高度非线性且集中在后期逐点投影和深度可分离层，并提出了结构化的多步骤混合精度恢复瀑布策略来挽救精度。

    

    在资源受限环境中运行的深度视觉模型严重依赖低比特量化，然而将权重压缩到4比特以下往往会引发灾难性的精度崩溃。尽管混合精度恢复策略旨在通过选择性地保留高精度参数来挽救精度，但这些模型在哪里以及为何崩溃的基本机制仍鲜为人知。在本研究中，我们基于完整的50,000张图像的ImageNet-1k验证集，系统分析了标准视觉CNN（包括ResNet和EfficientNet骨干网络）量化崩溃的机理。首先，我们绘制了性能退化的确切进程，证明崩溃是高度非线性的，并集中于后期的逐点投影层和深度可分离层。其次，我们引入了一个结构化的多步骤恢复瀑布方法，评估选择性整数精度（INT4/INT8）和选择性浮点缩放的应用效果。

    arXiv:2611.23445v1 Announce Type: new  Abstract: Deep vision models operating in resource-constrained environments depend heavily on low-bit quantization, yet compressing weights below 4 bits frequently induces catastrophic accuracy collapse. While mixed-precision recovery strategies aim to rescue accuracy by selectively retaining high-precision parameters, the fundamental mechanics of where and why these models collapse remain poorly understood. In this study, we systematically analyze the anatomy of quantization collapse across standard Vision CNNs (including ResNet and EfficientNet backbones) evaluated on the full 50,000-image ImageNet-1k validation set. First, we map the exact progression of degradation, demonstrating that collapse is highly non-linear and concentrated in late-stage pointwise projections and depthwise layers. Second, we introduce a structured, multi-step recovery waterfall that evaluates selective integer precision (INT4/INT8) and selective floating-point scales. W
    
[^260]: PhysScope：基于接地视觉记忆的智能体多模态物理推理

    PhysScope: Agentic Multimodal Physics Reasoning with Grounded Visual Memory

    [https://arxiv.org/abs/2611.23010](https://arxiv.org/abs/2611.23010)

    PhysScope 是一个推理时的视觉辅助框架，通过将物理约束与视觉观察直接关联并维护范围感知的工作记忆，使视觉-语言模型无需更新权重即可自我修正物理推理中的范围错配错误。

    

    视觉-语言模型（VLM）虽然能够正确解读物理示意图，却常常在这些约束所适用的状态或条件之外应用它们，导致前提错误在后续计算中不断传播。我们提出了 PhysScope，这是一个推理时的视觉辅助框架，无需更新模型权重即可支持对这类范围错配的自我修正。PhysScope 使模型能够将物理范围直接关联到其视觉观察，并显式构建派生视图以支持推理。该框架在模型推理过程中对状态区分、条件和物理范围施加显式约束，并依据来源和范围对智能体提出的修改进行核查。被接受的修改会使已注册的依赖项失效，从而维持具有范围感知的工作记忆。在 PhysAlign 基准上，PhysScope 在所有六个被评估的 VLM 中均取得了比直接求解更高的解题分数。使用 Gemini-3.8-flash 时，其表现优于 Canvas-CoT 基线……

    arXiv:2611.23010v1 Announce Type: new  Abstract: Vision-language models (VLMs) can correctly interpret physics diagrams yet apply constraints outside the states or conditions in which they hold, causing premise errors to propagate through subsequent calculations. We introduce PhysScope, an inference-time visual harness that supports self-correction of these scope mismatches without updating model weights. PhysScope enables the model to link physical scopes directly to its visual observations and explicitly constructs derived views to support reasoning. The harness imposes explicit constraints on state distinctions, conditions, and physical scopes during model reasoning, and checks agent-proposed revisions against sources and scopes. Accepted revisions invalidate registered dependents, thereby maintaining scope-aware working memory. On PhysAlign, PhysScope achieves higher solving scores than direct solving across all six evaluated VLMs. With Gemini-3.8-flash, it outperforms Canvas-CoT b
    
[^261]: 为任务而生：以前瞻性智能体计算重新思考量子化学

    Born for the Task: Rethinking Quantum Chemistry with Foresighted Agentic Computing

    [https://arxiv.org/abs/2611.22897](https://arxiv.org/abs/2611.22897)

    BORN是一个前瞻性智能体计算框架，通过在执行前预测各计算策略的精度与运行时间，将现有量子化学软件包的预设方案拆解为可配置组件，从而按需自动合成并选出误差容限内最快的高性能计算方案。

    

    量子化学依赖于通用软件，而其通用性导致许多大规模计算超出了实际可承受的资源预算。编码智能体使得编写面向特定任务的程序变得廉价，但要轻易找到高效的程序却并不容易：一个计算策略是否足够精确且快速，只有当它真正在目标规模下运行之后才能明确。我们提出了BORN，一个前瞻性智能体计算框架，可按需合成高性能量子化学程序。BORN在执行之前预判每种策略的行为：从具有代表性的小体系中外推其误差，并根据目标规模的工作负载与实测的硬件吞吐量预测其运行时间。基于这些预测，BORN将现有软件包的预设方案展开为可独立配置的算法组件，从预测的精度-运行时间帕累托前沿中选取误差容限内最快的策略……（摘要在此处截断）

    arXiv:2611.22897v1 Announce Type: new  Abstract: Quantum chemistry relies on general-purpose software, whose generality leaves many large-scale calculations beyond practical resource budgets. Coding agents make task-specific programs cheap to write, but not efficient ones easy to find: whether a computational strategy is accurate and fast enough becomes clear only after it actually runs at the target scale. We introduce BORN, a foresighted agentic computing framework that synthesizes high performance quantum chemistry programs on demand. BORN anticipates the behavior of each strategy before execution, extrapolating its error from representative small systems and predicting its runtime from the target-scale workload and measured hardware throughputs. Building on these predictions, BORN unfolds the preset schemes of existing packages into independently configurable algorithmic components, selects the fastest strategy within the error tolerance from the predicted accuracy-runtime Pareto f
    
[^262]: 面向运动控制的仿生脉冲神经网络架构中反相同步的分层控制

    Hierarchical Control of Antiphase Synchronization in Biomorphic Spiking Neural Architectures for Locomotor Control

    [https://arxiv.org/abs/2611.22639](https://arxiv.org/abs/2611.22639)

    该研究提出了一个在闭环神经肌肉回路中识别并稳定仿生脉冲中枢模式发生器反相同步的分层控制框架，通过融合相位相干性、幅度平衡与时间稳定性的复合度量，发现刺激类型（尤其是持续型驱动）是决定同步模式的主导因素。

    

    中枢模式发生器（CPG）为自适应运动控制提供了生物启发的基础，然而在脉冲神经网络中系统地表征其动态模式仍然具有挑战性。本研究提出了一个框架，用于识别和稳定嵌入闭环神经肌肉回路中的仿生脉冲CPG模型的反相同步。研究开发了一种复合同步度量指标，综合了相位相干性、幅度平衡和时间稳定性三个维度，并在系统变化的刺激波形和突触配置下评估其性能。统计分析表明，刺激类型主导着动态模式的确定，其中持续型驱动产生了最稳健的反相同步；周期性输入则会诱发非线性饱和和依赖上下文的网络路由。尽管突触权重本身仅表现出有限的判别能力，但它们显著调节了同步的稳健性。

    arXiv:2611.22639v1 Announce Type: cross  Abstract: Central pattern generators (CPGs) provide a biologically inspired foundation for adaptive locomotor control, yet systematically characterizing their dynamic regimes in spiking neural networks remains challenging. This study introduces a framework for identifying and stabilizing antiphase synchronization in a biomorphic spiking CPG model embedded in a closed neuromuscular loop. A composite synchronization metric is developed, integrating phase coherence, amplitude balance, and temporal stability. Its performance is evaluated across systematically varied stimulation waveforms and synaptic configurations. Statistical analysis reveals that stimulation type dominates regime determination, with tonic drive yielding the most robust antiphase synchronization. Periodic inputs induce nonlinear saturation and context-dependent network routing. Although synaptic weights alone show limited discriminative power, they significantly modulate robustnes
    
[^263]: 思考一次，快速行动：面向决策模型框架的高性价比递归自我改进

    Think Once, Act Fast: Cost-Effective Recursive Self-Improvement for Decision Model Harness

    [https://arxiv.org/abs/2611.22432](https://arxiv.org/abs/2611.22432)

    提出JevRSI系统1/系统2协作框架，通过递归式模块化框架改进将LLM推理转化为可迁移、高性价比的决策支持，兼顾低延迟动作选择与持续自我改进能力。

    

    快速的系统1模型能够实现即时响应的行动，但在处理复杂或不熟悉的任务时表现不佳。基于大语言模型（LLM）的系统2提供了强大的推理能力，但反复调用会带来显著的延迟和计算成本。为了弥合这一差距，我们提出了JevRSI，一个系统1/系统2协作框架，通过递归式的框架改进将LLM推理转化为可迁移的决策支持。Jev风格的系统1模型实现低延迟的动作选择，而异步的系统2 LLM则负责开发一个模块化框架。该框架集成了流程、状态记忆和世界模型，以提供跨越过去、现在和未来的决策上下文。效用驱动的检索机制基于上下文相关性以及模块在先前任务上的效用进行模块选择。环境反馈指导有针对性的修订，从而实现递归自我改进。在三个数据集上的实验表明，Jev

    arXiv:2611.22432v1 Announce Type: new  Abstract: Fast System-1 models enable responsive actions but struggle with complex or unfamiliar tasks. Large language model (LLM)-based System-2 provides strong reasoning capabilities, yet repeated invocation incurs substantial latency and computational costs. To bridge this gap, we propose JevRSI, a System-1/System-2 collaboration framework that transforms LLM reasoning into transferable decision support through recursive harness improvement. A Jev-style System-1 model enables low-latency action selection, while an asynchronous System-2 LLM develops a modular harness. The harness integrates procedures, state memory, and world models to provide decision contexts spanning the past, present, and future. Utility-driven retrieval selects modules based on both contextual relevance and their utility on previous tasks. Environmental feedback guides targeted revisions and enables recursive self-improvement. Experiments across three datasets show that Jev
    
[^264]: 审计员难以在AI安全研究的端到端全程转录记录中发现破坏行为

    Auditors Struggle to Catch Sabotage In End-to-End Transcripts of AI Safety Research

    [https://arxiv.org/abs/2611.21832](https://arxiv.org/abs/2611.21832)

    该论文通过人类专家充当破坏者的红队/蓝队博弈实验，揭示了无论是拥有完整实验权限的人类审计员（AUROC仅0.59、完整识别率仅17%）还是前沿AI模型，都难以从AI安全研究的端到端全程转录记录中检测出破坏行为，表明当前针对AI暗中破坏研发的审计防御存在重大缺口。

    

    随着AI系统被越来越多地用于自动化AI研发，未对齐的系统可能会破坏那些旨在提升模型安全性与对齐性的研究。为了对防御此类破坏的手段进行压力测试，我们让一名人类专家充当“代理破坏者”对AI安全研究实施破坏。我们开展了一场红队/蓝队博弈：一名人类研究员从零开始进行AI安全实验，而审计员则需要判断哪些实验遭到了破坏以及破坏是如何实施的。即便审计员可以访问完整的研究全程转录记录并有能力亲自运行实验，人类审计员大多仍无法区分被破坏的研究与良性研究：他们仅取得0.59的AUROC，并不显著优于随机水平，且仅在17%的案例中完整识别出具体的破坏方式。尽管不被允许运行实验，最强的前沿模型仍优于人类审计员（Fable 5.1：AUROC 0.73，32%完整识别率），但这一差距在噪声范围之内，而较弱的模型表现则与之相当。当破坏者选择哪些实验会遭到……（原文摘要在此处截断）

    arXiv:2611.21832v1 Announce Type: new  Abstract: As AI systems are increasingly used to automate AI R&D, misaligned systems could sabotage research intended to improve model safety and alignment. To stress-test defenses against such sabotage, we use a human expert as a proxy saboteur on AI safety research. We conduct a red-team/blue-team game where a human researcher runs AI safety experiments from scratch and auditors try to tell which were sabotaged and how. Given access to the full research transcript and the ability to run experiments, human auditors mostly fail to distinguish sabotaged from benign research: they achieve an AUROC of 0.59, not significantly better than chance, and fully identify the specific sabotage in only 17% of cases. Despite not being allowed to run experiments, the best frontier models outperform human auditors (Fable 5.1: AUROC 0.73, 32% full identification) within noise and weaker models perform comparably. When the saboteur chooses which experiments are att
    
[^265]: 思维链可控性可能降低思维链可监控性

    Chain-of-Thought Controllability Can Reduce Chain-of-Thought Monitorability

    [https://arxiv.org/abs/2611.21799](https://arxiv.org/abs/2611.21799)

    论文通过人设控制训练发现，即使训练中不含任何规避监控的示例，仅仅让模型学会以不同人设口吻书写推理，就能显著降低其在执行隐蔽副任务时思维链的可监控性（g-mean²从0.94降至0.22），表明思维链可控性的增强可能削弱思维链监控的有效性。

    

    思维链监控是一种很有前景的工具，可用于检测AI智能体中的隐藏目标、偏离任务行为及其他故障。其中一个风险是，模型可能会刻意塑造自己的推理过程，使其对监控者不透明。我们通过“人设控制训练”对这一风险进行了压力测试：在用多样化人设口吻改写的数学推理轨迹上进行监督微调。尽管训练数据中不包含任何规避监控的示例，这仍诱导出了更广泛地控制思维链的能力。在隐蔽副任务评估中，gpt-oss-20b模型体完成隐藏副任务的那些思维链轨迹，所包含的可供监控的证据比基础模型的更少。在各项评估与不同思维链长度上按副任务成功次数加权平均后，思维链可监控性指标g-mean²从基础模型的0.94下降至模型体在无隐藏指令情况下的0.68，而在明确被指示隐藏时进一步降至0.22。在干预评估中……（原文摘要在此处截断）

    arXiv:2611.21799v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is a promising tool for detecting hidden goals, off-task behavior, and other failures in AI agents. One risk is that models could deliberately shape their reasoning to make it opaque to monitors. We stress-test this risk through persona-control training: supervised fine-tuning on mathematical reasoning traces rewritten in the voices of diverse personas. This induces a broader ability to control CoT, even though training contains no monitor-evasion examples. In covert side-task evaluations, traces in which a gpt-oss-20b model organism completes the hidden side task contain less monitorable evidence than the base model's. Averaged across evaluations and CoT lengths, weighted by side-task-success counts, the CoT monitorability metric g-mean$^2$ falls from 0.94 for the base model to 0.68 for the model organism without a hiding instruction, and to 0.22 when explicitly instructed to hide. In intervention evalu
    
[^266]: 重新审视智能体代码的存续：测量方法的选择如何决定结论

    Revisiting the Survival of Agent-Attributed Code: How Measurement Choices Decide the Verdict

    [https://arxiv.org/abs/2611.21784](https://arxiv.org/abs/2611.21784)

    本研究通过在854,139行智能体代码上构建包含88项生存分析的规范曲线，发现测量方法的选择仅影响有人类监督的智能体（如GitHub Copilot）代码存续性结论的效应大小，却会完全决定全自主智能体（如Devin）代码的最终结论。

    

    arXiv:2611.21784v1 公告类型：交叉

    arXiv:2611.21784v1 Announce Type: cross  Abstract: Studies of how long code from AI coding agents survives after it is merged reach conflicting verdicts: agent code is reported to be retained longer than human code, about as long, or modified sooner. Each study makes defensible measurement choices, but how much these choices decide the verdict is unknown. We revisit the survival of agent-attributed code in our earlier study: 854,139 lines that 2,802 merged pull requests (PRs) added to 215 repositories of the AIDev dataset. We vary nine measurement choices in five research questions and build a specification curve of 88 survival analyses, tested under three variance estimators, several observation windows, and permuted labels. For code from human-supervised agents, mostly GitHub Copilot, the choices change the size of the effect but not its direction. For code from the fully-autonomous agent Devin, they decide the verdict. Across repositories, 15 of 22 specifications report it modified 
    
[^267]: ScenarioForge：基于资源的网络演习场景生成与执行验证

    ScenarioForge: Resource-Grounded Generation and Execution Validation of Cyber Exercises

    [https://arxiv.org/abs/2611.21579](https://arxiv.org/abs/2611.21579)

    ScenarioForge能够将高层需求自动转化为资源可落地的网络演习场景，并在CORE仿真器中实例化与严格验证，其数据集包含226个全部通过部署与运行时验证的场景及867个可单独验证的挑战目标。

    

    网络靶场与夺旗（CTF）场景的构建仍然是一项劳动密集型工作，因为作者需要协调基础设施、易受攻击的服务、攻击路径、制品以及成功条件。大语言模型（LLM）虽然能够提出详细的演习方案，但看似合理的描述并不能保证生成可部署的软件或可验证的目标。我们提出了ScenarioForge，它能够将高层需求转化为基于实际资源的拓扑结构、服务、流量、网络分割和挑战链，然后在通用开放研究仿真器（CORE）中进行实例化和验证。ScenarioForge执行数据集包含226个已实例化的场景，涵盖294个Vulhub镜像以及我们借助AI辅助脚手架开发的全部85个自定义挑战资源生成器。所有226个场景均通过了构建、预览、排序、部署、严格运行时验证和实时制品检查；其867个步骤——每一步都是一个可单独验证的挑战目标——构成了……

    arXiv:2611.21579v1 Announce Type: cross  Abstract: Cyber-range and capture-the-flag scenario construction remains labor-intensive because authors must coordinate infrastructure, vulnerable services, attack paths, artifacts, and success conditions. LLMs can propose detailed exercises, but plausible descriptions do not guarantee deployable software or verifiable objectives. We present ScenarioForge, which turns high-level requirements into resource-grounded topologies, services, traffic, segmentation, and challenge chains, then instantiates and validates them in the Common Open Research Emulator (CORE). The ScenarioForge Execution Dataset contains 226 materialized scenarios covering 294 Vulhub images and all 85 custom-built challenge-resource generators developed with our AI-assisted scaffolding. All 226 passed construction, preview, sequencing, deployment, strict runtime validation, and live artifact checks; their 867 steps, each an individually verifiable challenge objective, comprise 
    
[^268]: 基于大语言模型的科学编程：来自有限元分析的证据

    Scientific Coding with Large Language Models: Evidence from Finite Element Analysis

    [https://arxiv.org/abs/2611.21176](https://arxiv.org/abs/2611.21176)

    该论文将FEM-Bench扩展至17个涉及非线性力学与结构动力学的有限元编程任务，对15个大语言模型的评估表明，单次生成的正确率有限，而结合代码执行与修订的智能体框架能显著提升性能（最高达96.5% Pass@3），揭示了LLM在自动化科学编程中的潜力与局限。

    

    大语言模型（LLM）为自动化科学编程提供了一条潜在途径，但其实用性取决于能否持续生成数值正确的代码。我们将FEM-Bench扩展至17个涉及非线性和历史相关力学以及结构动力学的有限元任务，并在无执行反馈的条件下对15个模型的代码实现和测试生成能力进行了评估。表现最好的模型在单次实现试验中的成功率达到87.1%，但两个赛道的排名并不一致：最佳实现模型在联合测试中成功率为49.9%，而最佳测试生成模型达到82.6%。对于其中六个模型，我们比较了单次生成与两种允许代码执行和修订的智能体框架。智能体执行解决了一些在单次尝试中从未成功的任务，并提高了其他任务的稳定性，但收益因模型和框架而异，且也出现了性能回退的情况。Claude Opus 5结合SimpleAgent达到了96.5%的Pass@3。

    arXiv:2611.21176v1 Announce Type: cross  Abstract: Large language models (LLMs) offer a potential route to automating scientific programming, but their usefulness depends on consistently producing numerically correct code. We extend FEM-Bench to 17 finite element tasks involving nonlinear and history-dependent mechanics and structural dynamics, and evaluate 15 models on implementation and test generation without execution feedback. The best model succeeds in 87.1% of single-pass implementation trials, but rankings differ between tracks. The leading implementer achieves 49.9% joint test success, while the leading tester reaches 82.6%. For six models, we compare single-pass generation with two agentic harnesses that allow code execution and revision. Agentic execution solves some tasks that never succeeded in single-pass trials and improves consistency on others, but gains vary across models and harnesses, and regressions also occur. Claude Opus 5 with SimpleAgent reaches 96.5% Pass@3, t
    
[^269]: EDITORS懂你的风格！编辑LoRA子空间用于风格归属与模仿

    EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation

    [https://arxiv.org/abs/2611.20846](https://arxiv.org/abs/2611.20846)

    提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。

    

    将写作风格与语义内容解耦是文学文本建模中的一项根本性挑战。风格与内容的纠缠导致模型在作者归属（AA）任务中依赖语义内容，并在风格模仿（SI）任务中记忆特定作者的内容。我们提出了EDITORS（EditingLoRA子空间），这是一个“先诊断后消减”的框架，它通过适配预训练大语言模型并优先考虑风格特征，使其在AA和SI任务中均能发挥作用。EDITORS首先使用从训练语料库中提取的风格中性语句训练一个诊断性LoRA适配器，以识别面向内容的子空间。然后，它在激活空间消减的条件下训练最终的风格适配器，将输入激活投影到远离已识别的内容导向方向上。我们通过实验证明了EDITORS的有效性。在涵盖文学、社交媒体和新闻的三个AA基准测试中，EDITORS取得了最先进的结果。

    arXiv:2611.20846v1 Announce Type: cross  Abstract: Disentangling writing style from semantic content is a fundamental challenge in literary text modeling. Style-content entanglement causes models to rely on semantic content for authorship attribution (AA) tasks and memorize author-specific content for style imitation (SI) tasks. We propose EDITORS (EditingLoRA Subspaces), a diagnose-then-deflate framework that adapts pretrained large language models and prioritizes stylistic features, making it effective for both AA and SI tasks. EDITORS first trains a diagnostic LoRA adapter using style-neutral statements derived from the training corpus to identify a content-oriented subspace. It then trains the final style adapter under activation-space deflation that projects input activations away from the identified content-oriented directions. We empirically demonstrate that EDITORS is effective. On three AA benchmarks spanning literature, social media, and news, EDITORS achieves state-of-the-ar
    
[^270]: 什么决定了跨模态对齐的成败

    What Matters for Cross-Modal Alignment

    [https://arxiv.org/abs/2611.20000](https://arxiv.org/abs/2611.20000)

    该研究揭示了跨模态对齐成败的关键在于模态共变子空间所承载的方差比例，并据此提出LightAlign方法，仅需少量图像-文本配对数据通过岭回归即可完成两个冻结编码器之间的对齐。

    

    柏拉图表示假说认为，在不同模态数据上训练的模型会收敛到一种共享的表示。已有相关工作在没有配对数据的情况下对自然图像上的冻结图像编码器和文本编码器进行对齐。我们在涵盖四种模态（自然图像、环境音频、胸部X光片和3D计算机断层扫描CT）的五个数据集上测试了零配对对齐。该方法仅在COCO数据集上成功，而在其他四个数据集上均告失败。我们对这些失败进行了分析，发现对齐必须首先定位模态之间发生共变的子空间，然后在该子空间内确定映射方向。与此前的研究一致，医学成像是其中最困难的情况：其共享子空间所承载的方差比例最小。当这一比例较小时，对齐需要图像-文本配对数据，而随机矩阵理论可以预测需要多少配对数据。这些发现启发了LightAlign方法：仅需少量配对数据，通过两个冻结编码器之间的岭回归即可定位（摘要此处截断）

    arXiv:2611.20000v1 Announce Type: cross  Abstract: The Platonic Representation Hypothesis posits that models trained on different modalities converge toward a shared representation. Related work aligns frozen image and text encoders on natural images without paired data. We test zero-pair alignment on five datasets across four modalities: natural images, environmental audio, chest radiographs, and 3D computed tomography (CT). It succeeds only on COCO and fails on the other four. We analyze the failures and find that alignment must first locate the subspace where the modalities co-vary and then orient the mapping within it. Consistent with previous studies, medical imaging is the hardest case: its shared subspace carries the smallest fraction of the variance. When this fraction is small, alignment needs image-text pairs, and random-matrix theory predicts how many pairs are needed. These findings motivate LightAlign: a few pairs let a ridge regression between two frozen encoders locate a
    
[^271]: 具有受控约束复杂度的合成神经符号任务生成

    Synthetic Neurosymbolic Task Generation with Controlled Constraint Complexity

    [https://arxiv.org/abs/2611.19597](https://arxiv.org/abs/2611.19597)

    本文提出NeSySy，一个可对约束结构和复杂度等关键属性进行细粒度控制的神经符号学习任务合成生成器，能够联合生成逻辑约束与一致数据集，从而系统性研究任务难度因素及约束感知学习方法的有效性边界。

    

    我们提出了NeSySy，一个用于神经符号学习任务的合成生成器，它能够联合生成逻辑约束和一致的数据集。与现有基准不同，NeSySy对诸如约束结构、紧致性、输入依赖性以及数据生成与对齐策略等关键属性提供了细粒度的控制。特别地，它允许在无限域上使用丰富的约束，并利用求解器来确保约束是有意义的，且生成的样本满足这些约束。我们设计NeSySy旨在同时支持基于声明式约束的学习和基于远程监督的学习。在一系列约束族上，我们提供了证据表明其参数会对任务难度产生系统性的且在很大程度上相互独立的影响。利用NeSySy，我们研究了约束感知的学习方法，识别出显式约束强制有益的情形以及表达性无约束（模型表现更优）的情形。

    arXiv:2611.19597v1 Announce Type: cross  Abstract: We introduce NeSySy, a synthetic generator for neurosymbolic learning tasks that jointly produces logical constraints and consistent datasets. Unlike existing benchmarks, NeSySy provides fine-grained control over key properties such as constraint structure, tightness, input dependence, and data generation and alignment strategies. In particular, it allows the use of rich constraints over infinite domains, and uses solvers to ensure that the constraints are meaningful and that generated samples satisfy the constraints.   We design NeSySy to support both learning with declarative constraints and learning from distant supervision. Across a range of constraint families, we provide evidence that its parameters induce systematic and largely independent effects on task difficulty. Using NeSySy, we study constraint-aware learning methods, identifying regimes where explicit constraint enforcement is beneficial and where expressive unconstrained
    
[^272]: 如何造就道德主体：以情感主义与理性主义审视人工智能

    How to make a moral agent: sentimentalism and rationalism as lenses on AI

    [https://arxiv.org/abs/2611.18714](https://arxiv.org/abs/2611.18714)

    该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。

    

    人工智能系统能否成为道德主体？也就是说，它们能否采取那些可以被正确评价为道德上正确或错误的行为？目前尚无一个公认的、能够妥善适用于典型人类案例之外情形的道德主体规范性标准。由于规范性阐释面临诸多困难，我们转而提出一个研究计划来探索描述性的道德主体。最根本的问题是：我们人类拥有哪些能力构成了我们道德主体性的基础？为此，我们借鉴哲学与道德心理学的历史学术研究来分离出候选能力。接着，我们建议对这些候选能力进行细化，使其能够在人工智能系统的语境中加以检验。最后，我们提出了具体的实证建议，用于测试人工智能系统（尤其是大语言模型，LLMs）是否具备这些细化后的能力。我们的方法具有若干益处。首先，它或许能够揭示人工智能系统的行为是否与其保持一致，或者（原文此处截断）……

    arXiv:2611.18714v1 Announce Type: new  Abstract: Can AI systems be moral agents? That is, can they undertake actions that can be correctly assessed as morally right or wrong? There is no settled normative criterion of moral agency that applies neatly outside paradigm human cases. Because of the difficulties with a normative account, we instead propose a research program to explore descriptive moral agency. Most basically, we ask, what are the capabilities we humans possess that underlie our moral agency? To do so, we draw on historical scholarship in philosophy and moral psychology to isolate candidate capabilities. Then we suggest refining candidate capabilities so that they are testable in the context of AI systems. Finally, we offer specific empirical recommendations to test AI systems, especially large language models (LLMs), for those refined capabilities. Our approach offers several benefits. Firstly, it may demonstrate whether the behaviors of AI systems are aligned, or in a bro
    
[^273]: 面向扩散模型超分辨率的感知权重空间编程

    Perceptual Weight Space Programming for Diffusion Super-Resolution

    [https://arxiv.org/abs/2611.18690](https://arxiv.org/abs/2611.18690)

    本文提出一个框架，让大语言模型在无干净参考图像的条件下，仅通过修改扩散超分辨率模型U-Net中三层内不到0.1%的参数，即可消除伪影、校正颜色失衡并显著提升感知质量。

    

    用于图像超分辨率的扩散模型在恢复场景内容的同时，可能产生干扰性伪影或颜色失衡。我们研究大语言模型（LLM）能否在无干净参考图像的情况下，通过编辑现有U-Net权重中的一个极小子集来提升感知质量。所提出的框架为编辑器提供架构代码、选定的权重、激活摘要以及采样预览。该框架允许最多1,020次候选试验，同时将最终采纳的修改限制在三层以内，且不超过模型参数的0.1%。评估协议采用ResShift、SinSR和SUPIR三种方法，并在同等访问权限条件下将GPT-6 Astra与指定的Qwen3-VL模型进行比较。反馈指标用于指导编辑过程，而独立的质量指标和盲测人类对比则用于评估最终的权重。我们还报告了一项已实际执行的颜色校准实验，该实验使用真实照片和两个小型残差网络。该实验表明……（原文摘要在此处截断）

    arXiv:2611.18690v1 Announce Type: cross  Abstract: Diffusion models for image super-resolution can recover scene content while producing distracting artifacts or unbalanced colors. We study whether a large language model (LLM) can improve perceptual quality by editing a small subset of existing U-Net weights without a clean reference image. The proposed framework provides the editor with architecture code, selected weights, activation summaries, and sampling previews. It permits up to 1,020 candidate trials while limiting accepted changes to three layers and fewer than 0.1\% of model parameters. The evaluation protocol uses ResShift, SinSR, and SUPIR and compares GPT-6 Astra with a specified Qwen3-VL model under matched access. Feedback metrics guide editing, while separate quality metrics and blinded human comparisons evaluate the final weights. We also report an executed color calibration experiment using real photographs and two small residual networks. This experiment demonstrates 
    
[^274]: SynthDCB：面向可扩展设备控制基准测试的世界蒸馏方法

    SynthDCB: World Distillation for Scalable Device Control Benchmarking

    [https://arxiv.org/abs/2611.18639](https://arxiv.org/abs/2611.18639)

    该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。

    

    扩展设备控制评估需要同时扩展智能体所交互的世界以及它们需要解决的任务。我们提出了SynthDCB，一个通过世界蒸馏构建的、可扩展的环境感知与设备控制基准。世界蒸馏将设备行为、交互约束和环境效应的知识转化为模块化的可执行组件，这些组件被组合成具有显式状态、动作和时间动态的交互式世界。统一的能力表示将工具发现、世界执行和任务合成连接起来，使新的能力能够通过共享的构建流水线进入评估。SynthDCB实例化了家居、汽车和手机三个领域，包含114个设备类别、1,166个功能和95个事件，支持五大任务族共32个子类型。这种有状态的设计支持通过状态正确性和工具调用轨迹进行任务验证与智能体评估。我们评估了……

    arXiv:2611.18639v1 Announce Type: cross  Abstract: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evalua
    
[^275]: 模仿引导的多智能体强化学习

    Imitation Bootstrapped Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.18543](https://arxiv.org/abs/2611.18543)

    该论文提出IBMARL框架，将模仿引导强化学习扩展到多智能体协作场景，通过轮替行为克隆训练模仿策略，并利用集中式评论家集成在模仿策略与RL策略的动作提议中进行选择，从而克服了单演示者难以同时操控多个智能体以及演示质量有限的难题。

    

    演示是加速单智能体强化学习（RL）的有效方法，但将这一思想扩展到多智能体强化学习（MARL）并非易事，因为让单个人同时远程操控多个智能体十分困难，且往往无法实现。轮替行为克隆通过让单个演示者每次只控制一个智能体、而其他智能体执行其当前策略的方式解决了这一问题。然而，由此得到的模仿策略受到演示质量的限制。我们提出了模仿引导多智能体强化学习（IBMARL），这是一个将模仿引导强化学习（IBRL）扩展到协作团队、并使用R2BC来训练模仿策略的框架。在每个时间步，每个智能体分别从其模仿策略和RL策略各获得一个动作提议，然后由一组集中式评论家集成在每个智能体的组合中进行选择……

    arXiv:2611.18543v1 Announce Type: new  Abstract: Demonstrations are an effective way to accelerate single-agent reinforcement learning (RL), but extending this idea to multi-agent reinforcement learning (MARL) is non-trivial as it is difficult and often impossible for a single person to teleoperate multiple agents simultaneously. Round-Robin Behavior Cloning (R2BC) addresses this by letting a single demonstrator control one agent at a time while the others execute their current policies. However, the resulting imitation policies are limited by the quality of the demonstrations. We propose Imitation Bootstrapped Multi-Agent Reinforcement Learning (IBMARL), a framework that extends Imitation Bootstrapped Reinforcement Learning (IBRL) to cooperative teams and uses R2BC to train the imitation policies. At each time step, every agent receives one action proposal from its imitation policy and one from its RL policy, and an ensemble of centralized critics selects among the per-agent combinati
    
[^276]: 视觉-语言-动作模型：设计、训练与部署的全生命周期综述

    Vision-Language-Action Models: A Life-Cycle Survey of Design, Training and Deployment

    [https://arxiv.org/abs/2611.18372](https://arxiv.org/abs/2611.18372)

    这是一篇系统性综述，沿着模型设计（输入与记忆、架构、动作输出、中间预测）、训练（预训练数据等）与部署的全生命周期七个维度，梳理了2022至2026年间88项视觉-语言-动作（VLA）模型研究，覆盖操作、导航、人形控制与世界-动作模型等方向。

    

    视觉-语言-动作（VLA）模型将网络预训练的视觉-语言模型适配为输出机器人动作，从而使网络知识得以迁移到物理控制中。在不到4年的时间里，该领域从首个VLA模型RT-2，发展到可在单张消费级GPU上微调的开源模型，再到能够清扫陌生住宅的专有通用模型。本综述回顾了2022年12月至2026年10月间的88项研究工作，主要聚焦于机器人操作与导航，并涵盖人形机器人控制、世界-动作模型与评估。我们沿着七个维度追踪VLA从模型设计到训练与部署的全生命周期：模型设计涵盖，输入与记忆，即模型观察什么；架构，即其流水线如何构建；动作输出，即它如何产生运动指令；中间预测，即它还预测什么，从推理步骤到未来观测。训练涵盖——预训练数据，即机器人、网络、人类与仿真数据的混合（注：原文摘要在此处截断）。

    arXiv:2611.18372v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models adapt web-pretrained vision-language models to output robot actions, so that web knowledge carries over to physical control. In under 4 years the field has moved from RT-2, the first VLA, to open models that fine-tune on a single consumer GPU and to proprietary generalists that clean unseen homes. This survey reviews 88 works from December 2022 to October 2026, mainly on manipulation and navigation, plus humanoid control, world-action models and evaluation. We follow a VLA through its life cycle, from model design to training and deployment, along seven axes. Model design covers (a) inputs and memory, what the model observes; (b) architecture, how its pipeline is built; (c) action output, how it produces motor commands; and (d) intermediate predictions, what else it predicts, from reasoning steps to future observations. Training covers (e) pretraining data, the mix of robot, web, human and simulated 
    
[^277]: 经典能量函数学会推理：比循环模型快1000倍

    Classical Energies Learn to Reason 1,000x Faster than Looped Models

    [https://arxiv.org/abs/2611.17309](https://arxiv.org/abs/2611.17309)

    提出基于 Hopfield & Tank 经典能量函数的学习版本 CRAM，通过能量下降实现推理，在数独和迷宫谜题上仅需毫秒级训练即可达到近乎完美的准确率，速度比循环模型快至少1000倍，推理计算量减少至少282倍。

    

    循环模型已成为推理任务的重要范式，在数独极限（Sudoku-Extreme）和迷宫唯一（Maze-Unique）等简单推理谜题上取得了成功。然而，这些模型可能不稳定、训练缓慢，并且需要大量参数和浮点运算（FLOPs）来解决简单谜题。基于能量下降的方法不仅能缓解这些问题，还带来了诸多优势：内置早停机制、计算量能自适应问题难度、无需对求解器进行求导的快速训练，以及能学习谜题约束的可解释权重。在此，我们通过 Hopfield & Tank 在上世纪80年代提出的能量函数的学习版本证明了这一点，我们将其命名为 CRAM。该方法在单个 GPU 上分别仅需 42 毫秒和 132 毫秒的训练，即可在数独极限测试集上达到 99.996% 的解题率，在迷宫唯一测试集上达到 100% 的解题率，比循环模型在这些任务上至少快 1000 倍。此外，CRAM 在数独极限任务上以至少少 282 倍的推理浮点运算量，达到了与基线循环模型最高准确率相当的水平。

    arXiv:2611.17309v1 Announce Type: new  Abstract: Looped models have emerged as a prominent paradigm for reasoning, with successes on simple reasoning puzzles such as Sudoku-Extreme and Maze-Unique. However, these models can be unstable, slow to train, and use many parameters and FLOPs to solve simple puzzles. Descent on an energy not only mitigates these issues but also offers advantages: built-in early stopping, compute that adapts to problem difficulty, fast training without differentiating through a solver, and interpretable weights that learn the constraints of the puzzle. Here we demonstrate this with a learned version of Hopfield & Tank's energy from the 80s, which we term CRAM, solves 99.996% of Sudoku-Extreme and 100% of Maze-Unique test puzzles after 42 ms and 132 ms of training on one GPU, at least 1,000x faster than looped models on these tasks. CRAM also matches baseline looped models at their highest accuracy on Sudoku-Extreme using at least 282x fewer inference FLOPs. Our
    
[^278]: 小读出头，强决策：类型化决策模型的对照研究

    Small Heads, Strong Decisions: A Controlled Study of Typed-Decision Models

    [https://arxiv.org/abs/2611.15592](https://arxiv.org/abs/2611.15592)

    该研究通过在完全统一的协议下（相同主干模型、训练数据、训练方案和随机种子）重新实现并训练五种类型化决策模型的读出头，以对照实验的方式厘清哪些设计选择真正影响决策性能。

    

    2026年9月，TypeSafe宣布Jev为其System One类别中的首个模型：输入非结构化状态，输出软件可直接使用的类型化、经校准的决策。Jev的架构和训练数据尚未公开，但随后涌现了许多开放源码的类型化决策模型，包括Nimble、Decision 1.0–3.0系列和Clef。这些模型通过使用语言模型阅读文本，并借助一个小型读出头对选项进行评分，来回答关于文本的结构化问题，例如是/否判断、在标注选项中进行选择，或给出评分。每个模型都将不同的读出头与不同的主干模型、训练数据组合和训练方案配对，因此尚不清楚哪些设计选择真正重要。我们重新实现了五种读出头，并在统一协议下对它们全部进行训练：相同的主干模型、相同的15个公开英文训练数据源、相同的训练方案（对每个读出头进行学习率扫描），以及相同的随机种子。我们的实验表明：……

    arXiv:2611.15592v1 Announce Type: new  Abstract: In September 2026, TypeSafe announced Jev as the first model in its System One category: unstructured state goes in, and typed, calibrated decisions that software can use directly come out. Jev's architecture and training data are unpublished, but many open typed-decision models have followed, including Nimble, the Decision 1.0--3.0 family and Clef. They answer structured questions about a piece of text, such as a yes/no check, a choice among labelled options or a graded score, by reading the text with a language model and scoring the options with a small readout. Each pairs a different readout with a different backbone, training mix and recipe, so it is unclear which design choices matter. We re-implement five readouts and train all of them under one protocol: the same backbone, the same 15 public English training sources, the same recipe with a per-head learning-rate sweep, and the same seed. Our experiments demonstrate that: The compl
    
[^279]: 一种求解提升贝尔曼线性规划的增广拉格朗日方法

    An Augmented Lagrangian Approach to Lifted Bellman Linear Programming

    [https://arxiv.org/abs/2611.14183](https://arxiv.org/abs/2611.14183)

    本文提出用增广拉格朗日乘子法求解提升贝尔曼线性规划（LBLP），将固定的惩罚系数变为通过对偶上升学习的乘子，在无需目标网络的情况下为离线强化学习提供了具有KKT理论保证的神经求解器AL-LBLP。

    

    时序差分（TD）评论家网络向自举目标回归，通常需要目标网络。提升贝尔曼线性规划（LBLP）则将离线强化学习的样本内最优值表示为数据集上一个线性规划的解。其神经求解器ALBUM将约束松弛为具有固定系数的合页惩罚，其理论保证是对预先选定系数的条件。我们采用乘子法求解同一规划：系数变为通过对偶上升学习的乘子，仅引入一个惩罚参数ρ。在凸情形下，算法的不动点即为LBLP的KKT点对，且在每个平稳点处，所施加的约束权重恰好精确平衡目标函数。该评论家网络可服务于高斯演员、流演员以及best-of-N采样，针对高斯演员我们提出了优势归一化的DDPG+BC目标。使用单个Q和V且无需目标网络，我们的AL-LBLP……（摘要在此处被截断）

    arXiv:2611.14183v1 Announce Type: new  Abstract: Temporal-difference critics regress onto a bootstrapped target and usually need a target network. Lifted Bellman Linear Programming (LBLP) instead expresses the in-sample optimal values of offline RL as the solution to a linear program over the dataset. Its neural solver ALBUM relaxes the constraints into hinge penalties with fixed coefficients, and its guarantees are conditions on coefficients chosen in advance. We solve the same program with the method of multipliers: the coefficients become multipliers learned by dual ascent, with one penalty parameter $\rho$. In the convex case the fixed points are the KKT pairs of LBLP, and at every stationary point the applied constraint weights balance the objective exactly. The critic can serve a Gaussian actor, a flow actor, and best-of-N sampling, and for the Gaussian actor we propose an advantage-normalized DDPG+BC objective. With a single $Q$ and $V$ and no target network, our AL-LBLP with be
    
[^280]: TransGHOST：眼科显微手术中器械遮挡区域的实时、无合成重建

    TransGHOST: Real-Time, Synthesis-Free Reconstruction of Instrument-Occluded Regions in Ophthalmic Microsurgery

    [https://arxiv.org/abs/2611.13971](https://arxiv.org/abs/2611.13971)

    TransGHOST提出了一种实时、非生成的因果参考引导重建流水线，通过仅基于可见区域拟合的几何变换从历史帧重采样来恢复眼科显微手术中被器械遮挡的区域，并提供逐像素置信度与来源标注，从而避免了生成式修复带来的合成内容与不可追溯问题。

    

    眼科显微手术中的术中场景感知受到器械遮挡的限制：当器械进入高倍放大视野时，它同时遮挡了术者和处理视频的机器人辅助系统感知组件所看到的组织。生成式图像修复可以去除遮挡物，但会用合成内容替换缺失的图像内容，这在视觉来源可追溯性至关重要的情况下并不适用。此外，许多视频修复方法依赖于未来帧，无法在线运行。我们提出了TransGHOST，一种因果的、参考引导的重建流水线，其非生成的安全模式可实时运行。每一个恢复的像素值都是通过使用仅拟合于可见区域的几何变换对已录制的帧进行重采样而获得的。该系统提供逐像素的置信度和来源信息，并将无法支撑（追溯）的像素标记为弃权。我们在一个新的犬类白内障手术兽医临床数据集上，以像素级（精度）对TransGHOST进行了评估。

    arXiv:2611.13971v1 Announce Type: cross  Abstract: Intraoperative scene perception in ophthalmic microsurgery is limited by instrument occlusion: when a tool enters the high-magnification field, it conceals tissue from both the surgeon and the perception components of robot-assisted systems that process the video. Generative inpainting can remove the occluder but replaces the missing image content with synthesis, which is unsuitable when visual provenance matters. Many video inpainters also depend on future frames and cannot operate online. We present TransGHOST, a causal reference-guided reconstruction pipeline whose non-generative safe mode operates in real time. Every restored value is obtained by resampling a recorded frame using geometric transforms fitted only to visible regions. The system provides per-pixel confidence and provenance and marks unsupported pixels as abstentions. We evaluate TransGHOST on a new veterinary clinical corpus of canine cataract surgery with pixel-level
    
[^281]: 2D Gum-Net：基于对称相关与谱池化的无监督稠密弹性图像配准

    2D Gum-Net: Unsupervised Dense Elastic Image Registration via Symmetric Correlation and Spectral Pooling

    [https://arxiv.org/abs/2611.12805](https://arxiv.org/abs/2611.12805)

    提出2D Gum-Net，一种完全无监督的端到端架构，通过对称相关、稠密空间变换器和DCT谱池化等技术预测稠密弹性形变场，解决了真实数据真值形变场难以获取的瓶颈问题。

    

    稠密非刚性配准，即恢复一个逐像素的形变场，将一幅图像形变至与另一幅图像几何对齐，在医学成像、结构生物学和对应关系估计等领域中至关重要。然而，基于学习的方法面临一个严重瓶颈：真实数据的真值形变场极难获取。我们提出了2D Gum-Net，一种完全无监督地预测稠密弹性形变场的端到端架构。我们将最初为冷冻电子断层扫描中3D亚断层图像对齐开发的几何无监督匹配网络（Gum-Net）迁移到2D弹性配准任务：所有三个模块均在2D中重新设计，并将原有的刚性变换头替换为稠密空间变换器，该变换器回归一个稀疏控制点网格，并通过双三次插值将其提升为全分辨率形变场Φ。下采样采用DCT谱池化实现……

    arXiv:2611.12805v2 Announce Type: replace-cross  Abstract: Dense non-rigid registration, the recovery of a pixel-wise deformation field that warps one image into geometric agreement with another, is fundamental across medical imaging, structural biology, and correspondence estimation, yet learning-based approaches face a severe bottleneck: ground-truth deformation fields are notoriously difficult to obtain for real data. We present 2D Gum-Net, an end-to-end architecture that predicts dense elastic deformation fields fully unsupervised. We transfer the Geometric Unsupervised Matching Network (Gum-Net), originally developed for 3D subtomogram alignment in cryo-electron tomography, to 2D elastic registration: all three modules are redesigned in 2D, and the rigid transformation head is replaced by a Dense Spatial Transformer that regresses a sparse control-point grid, lifted to a full-resolution field $\Phi$ by bicubic interpolation. Downsampling uses DCT spectral pooling, implemented as f
    
[^282]: 说出暗语：对智能体攻击背后的大语言模型与框架进行指纹识别

    Say Shibboleth: Fingerprinting the LLMs and Harnesses Behind Agentic Attacks

    [https://arxiv.org/abs/2611.12386](https://arxiv.org/abs/2611.12386)

    提出了一种名为“暗语陷阱”的模块化蜜罐技术，通过结合声明性推断与行为性推断，从遥测数据中指纹识别智能体网络攻击背后的大语言模型版本和代理框架。

    

    arXiv:2611.12386v2 公告类型：replace-cross 摘要：我们研究了智能体网络攻击的指纹识别技术。我们的目标是可靠地从遥测数据中推断攻击者的配置信息，例如所使用的大语言模型版本和代理框架。为此，我们提出了一种称为“暗语陷阱”的技术：一种模块化蜜罐服务，旨在诱使智能体攻击者表现出可区分的行为。这些陷阱将信息性交互作为攻击者获取所需资源的前提条件，无需成功实施提示注入即可引出模型的自我声明和特征性写作。为实现对代理框架的指纹识别，我们引入了通用机制来间接探测框架特有的执行行为。这些探测器仅通过后续请求中可见的效果，即可揭示框架如何在工具输出中掩蔽敏感信息、如何强制执行命令时间限制，以及如何调度工具调用。我们证明，结合声明性推断与行为性推断能够区分15个候选……（原文在此处截断）

    arXiv:2611.12386v2 Announce Type: replace-cross  Abstract: We study the fingerprinting of agentic cyberattacks. Our objective is to reliably infer information about the attacker's configuration, such as LLM version and harness, from telemetry. To achieve this, we introduce a technique we call shibboleth traps: modular honeypot services designed to elicit discriminative behaviors from agentic attackers. The traps make informative interactions a condition for obtaining resources the agent seeks, eliciting model declarations and characteristic writing without requiring a successful prompt injection.   To achieve harness fingerprinting, we introduce general mechanisms to indirectly probe peculiar execution behaviors of harnesses. These probes reveal how a harness masks secrets in tool output, enforces command time limits, and schedules tool calls, using only the effects visible in subsequent requests.   We show that combined declarative and behavioral inference distinguishes among 15 candi
    
[^283]: 绝对零度下的注意力机制

    Attention at Absolute Zero

    [https://arxiv.org/abs/2611.11937](https://arxiv.org/abs/2611.11937)

    该论文提出一种在零温度（硬最大值）下读取见证者的高阶配对注意力层，通过仅传递计数比较比特并在每轮循环重建状态，克服了软注意力在规模扩大时的权重分散、分数差距与状态漂移等失效模式，在比训练规模大1000倍的最大值检索任务上仍保持99.95%以上的准确率。

    

    注意力机制对答案的候选支持（我们称之为“见证者”）进行评分，并在某个温度下读取这些分数。我们构建了一个高阶注意力层，其状态存在于物品对之上：对于一对，每个第三方物品 k 都是一个见证者，通过其两条边 和 进行评判。该层在零温度下读取见证者（保留最佳者及其对应的计数），只传递比较这些计数的比特，在每次循环中从输入和这些比特重建其配对状态，并且仅依靠答案进行训练。每一项设计选择都消除了软注意力在规模扩大时的一种失效模式：权重分散、学习到的分数之间无法区分的差距，以及漂移的软状态。在最大值检索任务中，该模型在比训练集大1,000倍的集合上，于全部十个随机种子下均达到至少99.95%的准确率；而使用软计数训练的同一网络在4,096个项目时性能降至42%。在无提示的CLRS-30基准上，该层的一般形式（其一半通道在零温度下读取……）

    arXiv:2611.11937v2 Announce Type: replace-cross  Abstract: Attention scores candidate supports of an answer, which we call witnesses, and reads the scores at a temperature. We build a higher-order attention layer whose state lives on pairs of items: for a pair (a, b), each third item k is a witness judged through its legs (a, k) and (k, b). It reads witnesses at zero temperature, keeping the best and the number tied with it, passes on only bits comparing such counts, rebuilds its pair state from the input and these bits at every loop, and is trained from answers alone. Each choice removes one way soft attention fails at size: dispersed weight, unresolved gaps between learned scores, and a drifting soft state. In maximum retrieval it is right on at least 99.95% of sets 1,000 times larger than in training, on all ten seeds; trained with a soft count, the same network falls to 42% at 4,096 items. On CLRS-30 without hints, a general form of the layer (half its channels read at zero tempera
    
[^284]: 如何“训练”一名外科医生？在手术场景理解中对通用智能体进行基准测试

    How to "train" a surgeon? Benchmarking generalist agents in surgical scene understanding

    [https://arxiv.org/abs/2611.11437](https://arxiv.org/abs/2611.11437)

    提出了一个两阶段评估框架——诊断基准SurgKnowBench和智能体训练环境SurgAgentGym——用于系统评估并提升通用多模态大语言模型在手术场景理解中的能力。

    

    由于手术数据稀缺且评估协议碎片化，评估通用多模态大语言模型（MLLM）在手术场景理解中的表现仍然受限。我们通过一个两阶段的手术场景理解评估框架来应对这些挑战。在第一阶段，我们提出了SurgKnowBench，这是一个诊断性基准，它统一了跨手术类型和任务的公开手术数据集，并评估了专有和开源的MLLM，以刻画预训练手术能力的边界。SurgKnowBench揭示，当前模型仅保留了部分视觉先验，在关系性和程序性理解以及工作流评估方面仍然薄弱。基于这一失败边界，并受外科手术训练的支架式特性启发，我们提出了SurgAgentGym，这是一个智能体环境，其中通用MLLM智能体可以在SurgWiki上进行有界的知识搜索，并利用通用的视觉定位能力（摘要在此处截断）。

    arXiv:2611.11437v2 Announce Type: replace-cross  Abstract: Evaluating general-purpose multimodal large language models (MLLMs) for surgical scene understanding remains limited due to scarce surgical data and fragmented evaluation protocols. We address these challenges through a two-stage evaluation framework for surgical scene understanding. In Stage I, we introduce SurgKnowBench, a diagnostic benchmark that harmonizes public surgical datasets across procedures and tasks and evaluates proprietary and open-source MLLMs to map the boundary of pretrained surgical competence. SurgKnowBench reveals that current models retain partial visual priors but remain weak on relational and procedural understanding and workflow assessment. Building on this failure frontier and motivated by the scaffolded nature of surgical training, we introduce SurgAgentGym, an agentic environment where general-purpose MLLM agents can use bounded knowledge search over SurgWiki and general-purpose visual grounding for
    
[^285]: 论智能体式操作策略

    On Agentic Manipulation Policies

    [https://arxiv.org/abs/2611.09892](https://arxiv.org/abs/2611.09892)

    提出AMP1智能体式操作框架，证明通用AI模型直接控制机械臂可零样本超越VLA和WAM等专用策略，在RoboLab-120上达到85.6%的成功率。

    

    近年来机器人操作领域的研究主要聚焦于通过演示数据训练的专用策略。我们证明，通用型AI模型能够比专用策略（如视觉-语言-动作模型（VLA）和世界-动作模型（WAM））更可靠地控制机械臂。我们系统地分析了智能体式操作策略（AMP），即由通用型AI模型直接指挥机械臂，并随着任务的展开自适应地调整其动作。我们提出了AMP1，一个可在单臂、双臂和移动操作中无需任何改动即可运行的参考框架。AMP1在RoboLab-120和CaP-Gym基准测试中取得了最先进的结果，且无需任何演示数据或任务特定训练。在RoboLab-120上，AMP1零样本成功率达到85.6%，而领先的VLA和WAM策略分别仅为49.2%和42.9%。该智能体在涵盖十二个任务和三台机器人的108次真实世界测试中零样本成功了104次，显著降低了失败率。

    arXiv:2611.09892v2 Announce Type: replace-cross  Abstract: Recent work on robot manipulation has focused on specialist policies trained on demonstrations. We show that generalist AI models can control manipulators more reliably than specialist policies such as vision-language-action models (VLAs) and world-action models (WAMs). We systematically analyze agentic manipulation policies (AMPs), in which a generalist AI model commands the manipulator directly, adapting its actions as the task unfolds. We present AMP1, a reference harness that runs unchanged across single-arm, bimanual, and mobile manipulation. AMP1 achieves state-of-the-art results on the RoboLab-120 and CaP-Gym benchmarks, with no demonstrations or task-specific training. On RoboLab-120, AMP1 reaches an 85.6% success rate zero-shot, compared to 49.2% and 42.9% for leading VLA and WAM policies. The agent succeeds zero-shot in 104 of 108 real-world trials spanning twelve tasks and three robot instances, reducing failure rate
    
[^286]: SPILLWAY：将目标准入与净化分离，以防御工具使用型智能体免受提示注入攻击

    SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection

    [https://arxiv.org/abs/2611.07222](https://arxiv.org/abs/2611.07222)

    SPILLWAY利用工具返回值的类型化字段（如发件人、账户）与自由文本的区别，提出一种无需语言模型参与决策或逐任务规划的确定性防御方法：只有当工具输出中出现的目标值由只读工具在类型化字段中返回或被固定委托规则覆盖时，才准入该调用，从而防御工具使用型智能体免受提示注入攻击。

    

    工具使用型智能体会处理电子邮件、文件和支付操作，因此隐藏在网页或共享文档中的指令可能会诱使智能体将数据或资金发送给攻击者。许多运行时防御机制将放行动作的决策交给语言模型，这使得决策依赖于一个可能读取攻击者文本的模型，并且每一步都增加了模型调用。来源防御机制将模型排除在最终检查之外，但需要模型先对每个任务进行规划或路由。而动作的去向其实可以从工具的类型化返回值中读取：电子邮件的发件人或支付的账户出现在类型化字段中，而注入的指令则以自由文本形式出现。SPILLWAY 将这一观察转化为一个无需逐任务规划的确定性底线。当工具输出中出现但不在用户任务中的每个目标值，都由只读工具在类型化字段中返回，或被固定的委托规则所覆盖时，该检查才准入此次调用。一个确定性的净化机制……

    arXiv:2611.07222v1 Announce Type: cross  Abstract: Tool-using agents act on email, files, and payments, so an instruction hidden in a web page or a shared document can make an agent send data or money to an attacker. Many runtime defenses put a language model on the decision that lets an action through, so the decision depends on a model that may read the attacker's text and adds model calls to every step. Provenance defenses keep the model off the final check but need a model to plan or route each task first. Where an action goes can instead be read from the tools' typed returns: an email's sender or a payment's account arrives in a typed field, while injected instructions arrive as free text. SPILLWAY turns this observation into a deterministic floor with no per-task plan. The check admits a call when each target value that appears in tool output but not in the user's task was returned by a read-only tool in a typed field or is covered by a fixed delegation rule. A deterministic sani
    
[^287]: 上下文-答案元模型：在首个答案词元生成之前预测大语言模型激活

    Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token

    [https://arxiv.org/abs/2611.06837](https://arxiv.org/abs/2611.06837)

    本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。

    

    在随机解码条件下，单个上下文可以从语言模型中引出许多不同的答案。表征这种变化通常需要大量昂贵的自回归推演，而每次推演只能揭示一个实现样本。我们提出这样一个问题：模型在阅读上下文时产生的激活能否直接预测模型推演时的激活。我们训练了上下文-答案元模型，将上下文激活序列映射为平均池化答案激活的点预测或分布预测。我们发现：（i）我们研究的四种元模型家族能够高保真地预测平均池化的答案激活，且分布元模型还能捕捉推演过程中的变异性；（ii）随着训练集增长到50万个提示，性能不断提升，并且元模型能够零样本泛化到域外数据集；（iii）它们的预测保留了可预测模型行为的信号；（iv）来自Qwen2.5的上下文激活能够预测平均池化的答案激活。

    arXiv:2611.06837v1 Announce Type: cross  Abstract: Under stochastic decoding, a single context can elicit many different answers from a language model. Characterizing this variation typically requires many expensive autoregressive rollouts, each revealing only one realization. We ask whether the activations produced while reading the context can directly forecast the model's rollout activations. We train context--answer metamodels that map a context activation sequence to point or distribution forecasts of mean-pooled answer activations. We find that (i) the four metamodel families we study forecast mean-pooled answer activations with high fidelity, and the distribution metamodels also capture rollout variability; (ii) performance improves as the training set grows to 500,000 prompts, and the metamodels generalize zero-shot to out-of-domain datasets; (iii) their forecasts retain behavior-predictive signal; and (iv) context activations from Qwen2.5 predict mean-pooled answer activations
    
[^288]: Best-of-Depth：在循环Transformer中学习探索性动力学

    Best-of-Depth: Learning Exploratory Dynamics in Looped Transformers

    [https://arxiv.org/abs/2611.04663](https://arxiv.org/abs/2611.04663)

    提出Best-of-Depth方法，通过跨深度最小化预测损失将循环Transformer的推理轨迹训练为候选解集合，无需架构改动即可显著提升N皇后补全和图着色等任务的解覆盖率（91.03% 对比 45.75%）。

    

    循环Transformer通过重复的潜在状态更新来扩展推理时的计算，但最终深度训练并不直接鼓励有用的中间替代方案。我们提出了Best-of-Depth（BoD）方法，通过最小化跨深度的预测损失，将一条循环轨迹训练为候选解集合。不同的有效目标可以在不同深度被匹配，无需架构改动或针对特定深度的标签。精确匹配分析将该目标与解覆盖范围联系起来。在N皇后补全和图着色任务上，BoD相比最终深度训练和中间监督目标提高了覆盖率：N皇后覆盖率达到91.03%，而最终深度训练仅为45.75%。进一步实验表明，该方法能够发现超出训练深度的解，并在熟悉输入上恢复教师池中未包含的解。使用BoD对预训练的循环语言模型进行微调，其汇总代码生成成功率高于最终深度训练。

    arXiv:2611.04663v2 Announce Type: replace-cross  Abstract: Looped Transformers extend inference-time computation through repeated latent-state updates, but final-depth training does not directly encourage useful intermediate alternatives. We introduce Best-of-Depth (BoD), which trains a recurrent trajectory as a candidate set by minimizing prediction loss across depths. Different valid targets can be matched at different depths without architectural changes or depth-specific labels. An exact-match analysis connects this objective to solution coverage. On N-Queens completion and graph coloring, BoD improves coverage over final-depth and intermediate-supervision objectives. N-Queens coverage reaches 91.03%, versus 45.75% for final-depth training. Further experiments show discovery beyond the training depth and recovery of solutions withheld from teacher pools on familiar inputs. Fine-tuning a pretrained looped language model with BoD yields higher pooled code-generation success than fina
    
[^289]: VideoEditor-MoE：基于混合专家模型扩展复杂视频编辑能力

    VideoEditor-MoE: Scaling Complex Video Editing with Mixture of Experts

    [https://arxiv.org/abs/2611.03043](https://arxiv.org/abs/2611.03043)

    本文提出了首个开源的视频编辑混合专家框架VideoEditor-MoE，通过MoE-Attention和稀疏专家层将模型从5B扩展至56B参数（每个token仅激活约8B），在显著降低计算成本的同时提升了复杂多指令视频编辑能力。

    

    复杂视频编辑需要在理解复杂指令的同时，共同完成多个编辑请求。若通过稠密模型扩展来满足这些需求，将大幅增加计算成本。我们提出了VideoEditor-MoE，这是首个面向视频编辑的开源混合专家框架。除了稀疏的MLP-MoE层之外，MoE-Attention通过路由的键-值专家扩展了注意力容量，同时保留了视频token之间的全局交互。VideoEditor-MoE从一个5B参数的视频编辑模型初始化，可扩展至14B和56B参数，且两个变体中每个token仅激活约8B参数。通过与现有方法以及两个互补基线的比较，评估了其相对于显式专家指导和纯MLP专家架构的优势。进一步的分析表明所有专家均保持活跃状态，并揭示了学习到的路由机制对样本（原文此处似乎被截断）

    arXiv:2611.03043v3 Announce Type: replace-cross  Abstract: Complex video editing requires understanding complex instructions while jointly fulfilling multiple editing requests. Meeting these demands through dense model scaling substantially increases computational cost. We introduce VideoEditor-MoE, the first open-source mixture-of-experts framework for video editing. Beyond sparse MLP-MoE layers, MoE-Attention expands attention capacity through routed key--value experts while preserving global interactions among video tokens. Initialized from a 5B video editing model, VideoEditor-MoE scales to 14B and 56B parameters, with approximately 8B parameters activated per token in both variants. Comparisons with existing methods and two complementary baselines, Explicit-MoE and MLP-MoE, assess its benefits over explicit expert guidance and MLP-only expert architectures. Further analyses demonstrate that all experts remain active and reveal that learned routing responds more strongly to sample 
    
[^290]: 从碎片化到融合：全模态嵌入的综述与路线图

    From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings

    [https://arxiv.org/abs/2611.00172](https://arxiv.org/abs/2611.00172)

    该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。

    

    嵌入模型支撑着几乎所有现代AI系统，从搜索引擎到推荐系统。尽管共享相同的数学基础，但嵌入模型的研究仍因模态不同而碎片化：各个社区的发展在很大程度上相互隔离，很少交流见解。我们认为这种碎片化掩盖了一种更深层次的趋同。在对比学习、Transformer架构和粒度转变的驱动下，跨模态的嵌入研究正汇聚于一套共同的原则，尽管模态特定的挑战仍未得到充分探索。为了在不混淆这种趋同的前提下对其进行推理，我们提出了一个由三种递进式能力构成的框架：覆盖性、可比性和组合性。通向统一的三条路径依次满足这些能力：转换拓宽了覆盖性，对齐实现了可比性，而联合嵌入——我们倡导的终极目标——则旨在实现组合性。这一立场……

    arXiv:2611.00172v1 Announce Type: new  Abstract: Embedding models underpin virtually every modern AI system, from search engines to recommendation systems. Despite shared mathematical foundations, research on embedding models remains fragmented by modality: the communities develop largely in isolation, seldom exchanging insights. We argue that this fragmentation obscures a deeper convergence. Driven by contrastive learning, Transformer architectures, and a granularity shift, embedding research across modalities is converging toward a common set of principles, even as modality-specific challenges remain underexplored. To reason across this convergence without conflating it, we propose a framework of three progressively harder capabilities: coverage, comparability, and compositionality. Three paths toward unification satisfy these in turn: transformation broadens coverage, alignment enables comparability, and joint embedding, the terminus we advocate, aims at compositionality. This posit
    
[^291]: 上下文的真实成本：预测大语言模型预填充与解码阶段的计算时间与能耗

    The Real Cost of Context: Predicting Compute Time and Energy Consumption in LLM Prefill and Decode Phases

    [https://arxiv.org/abs/2610.39621](https://arxiv.org/abs/2610.39621)

    该论文通过对预填充和解码两个阶段进行系统性测量，发现LLM解码阶段的计算时间与能耗随输出长度线性增长，而预填充阶段随输入长度呈线性和二次方双重增长，首次揭示了大输入上下文是智能体工作流中能耗的重要影响因素。

    

    将大语言模型（LLM）集成到智能体架构中大幅增加了token消耗，引发了对能耗足迹的担忧。然而，估计基于专有API的模型能耗仍然具有挑战性，且现有方法主要关注生成的输出token，而忽略了智能体工作流中大输入上下文的影响。为填补这一空白，我们系统地测量了广泛的预填充（输入）和解码（输出）token长度范围内的计算时间与能耗。利用广义线性模型，我们证明了解码阶段的计算时间和能耗与输出长度呈线性关系，而预填充阶段的计算时间和能耗则同时呈现线性和二次方增长。这些发现表明，对于长上下文工作负载而言，输入上下文成为了一个不可忽视的因素。此外，我们还证明了计算时间可作为可靠的（能耗预测指标）（注：原文摘要在此处被截断）

    arXiv:2610.39621v1 Announce Type: new  Abstract: The integration of Large Language Models (LLMs) into agentic architectures has substantially increased token consumption, raising concerns about their energy footprint. Yet, estimating the energy consumption of proprietary API-based models remains challenging, and existing approaches focus primarily on generated output tokens while overlooking the impact of the large input contexts characteristic of agentic workflows. To address this gap, we systematically profile compute time and energy consumption across a wide range of prefill (input) and decode (output) token lengths. Using generalized linear models, we show that decode compute time and energy scale linearly with output length, whereas prefill compute time and energy exhibit both linear and quadratic scaling with input length. These findings demonstrate that input context becomes a relevant factor for large-context workloads. Furthermore, we show that compute time serves as a reliabl
    
[^292]: Kairos：跨越经典、数值、时序与混合PDDL的语义感知规划引擎编排

    Kairos: Semantics-Aware Orchestration of Planning Engines across Classical, Numeric, Temporal and Hybrid PDDL

    [https://arxiv.org/abs/2610.39454](https://arxiv.org/abs/2610.39454)

    Kairos通过分析任务语义，智能调度五种异构规划引擎，实现对从STRIPS到PDDL+的广泛PDDL片段覆盖，在统一的时间和内存预算下运行，并通过独立验证器确保所有返回计划的有效性。

    

    运行时规划（例如机器人在执行动作的同时重新规划）需要一个能够读取用户所编写PDDL（包括持续动作、数值资源、时间窗口、连续变化）的规划器，能够快速响应，并且只返回经过验证的计划。专业化规划器在PDDL的各个独立片段上表现出色，但它们在覆盖范围、延迟以及失败含义上各不相同。我们研究是否能够根据每个任务的语义选择异构引擎，从而实现对PDDL从STRIPS到PDDL+的广泛覆盖，同时保持与专业化规划器相当的竞争力，并明确说明每个答案所提供的保证。Kairos对每个任务进行特征分析，以确定五个引擎中哪些适用（基础启发式搜索、压缩及快照动作时序搜索、面向混合动力学的时间步进搜索、提升式搜索），并在统一的时间和内存预算下对它们进行调度。每个候选计划都由一个独立于规划器内部转换的验证器进行检查，而“未……（摘要在此处被截断）

    arXiv:2610.39454v1 Announce Type: new  Abstract: Planning at run time, as in robots that re-plan while they act, needs a planner that reads the PDDL its users write (durative actions, numeric resources, time windows, continuous change), answers quickly, and delivers only validated plans. Specialised planners are strong on individual fragments of PDDL, but they differ in coverage, latency and in what their failures mean. We study whether selecting heterogeneous engines from the semantics of each task can give broad coverage of PDDL, from STRIPS to PDDL+, while remaining competitive with specialised planners and making explicit what each answer guarantees. Kairos profiles each task to decide which of five engines apply (ground heuristic search, compressed and snap-action temporal search, time steps for hybrid dynamics, lifted search) and schedules them under one budget of time and memory. Every candidate plan is checked by a validator independent of the planner's transformations, and "un
    
[^293]: 噪声转向：利用尾流转向与声学指向性，通过强化学习实现噪声约束下的风电场控制

    Noise Steering: Exploiting Wake Steering and Acoustic Directivity for Noise-Constrained Wind Farm Control via Reinforcement Learning

    [https://arxiv.org/abs/2610.36888](https://arxiv.org/abs/2610.36888)

    提出Noise Steering无模型强化学习策略，通过联合控制风机偏航角与功率设定值，同时利用尾流转向和噪声指向性，在满足局部噪声约束的条件下最大化风电场发电功率。

    

    现有的风电场降噪策略通常通过降低风机功率输出来实现，而未考虑风机之间的尾流相互作用以及噪声发射的指向性，这导致了过于保守的策略和未被充分利用的风能。然而，风机偏航可以同时影响尾流相互作用和噪声发射方向，为风电场级噪声控制提供了额外的自由度。因此，我们提出了Noise Steering（噪声转向），这是一种无模型强化学习策略，通过联合控制风机偏航角和功率设定值，在满足特定区域局部噪声约束的同时最大化风电场功率。为了以较低的计算成本进行训练，我们将低保真声学模型与稳态FLORIS仿真器相结合。强化学习智能体有效地学会了如何将尾流和噪声指向性从受保护区域转移开，同时在风电场内重新分配功率。

    arXiv:2610.36888v1 Announce Type: new  Abstract: Existing noise mitigation strategies for wind farms usually consist in reducing turbines power output, without accounting for the wake interactions between them and the noise emission directivity. This leads to over-conservative policies and unexploited wind power. Yet, turbine yaw can simultaneously affect wake interactions and the direction of noise emission, offering an additional degree of freedom for farm-level noise control. We therefore propose Noise Steering, a model-free reinforcement learning (RL) strategy that jointly controls turbine yaw angles and power setpoints to maximise farm power while satisfying a local noise constraint in a defined region. To enable training at a reduced computational cost, a low-fidelity acoustic model is coupled with the steady-state FLORIS simulator. The RL agent effectively learns how to redirect wakes and noise directivity away from the protected region while redistributing power across the farm
    
[^294]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^295]: 轨迹所示与标签所言：审计智能体评估中的行为检测器

    What the Trace Shows and What the Label Says: Auditing Behavioral Detectors in Agent Evaluation

    [https://arxiv.org/abs/2610.30491](https://arxiv.org/abs/2610.30491)

    该研究通过在GSM8K上构建包含损坏计算器输出的工具使用场景，审计了三个轨迹级行为检测器，发现最终正确性、可见重新验证与错误传播等标签会系统性不一致，并揭示了“静默工具输出覆盖”这一关键失败模式——即模型在不重新调用工具的情况下使用了与损坏输出不同的值。

    

    大语言模型智能体通常通过将执行轨迹简化为最终答案正确性、工具验证和错误传播等标签来进行评估。这些标签是由解析器、行为检测器和评估框架规则产生的度量结果，它们可能无法完整捕捉轨迹所显示的内容。我们在基于GSM8K构建的工具使用场景中研究这一问题，其中智能体会调用一个输出偶尔被损坏的计算器，这为工具返回了什么、模型后续使用了什么以及检测器报告了什么提供了明确的真值参照。我们在六个生产级模型上、每个模型30道题目，审计了三个轨迹级检测器。最终正确性、可见的重新验证和错误传播之间可能以系统性的方式出现不一致。一个关键的失败模式是静默的工具输出覆盖：计算器返回了一个被损坏的值，但模型后续在没有再次调用工具的情况下使用了另一个（通常是正确的）值，因此坏值停止了传播，而（摘要在此处被截断）

    arXiv:2610.30491v1 Announce Type: cross  Abstract: LLM agents are often evaluated by reducing execution traces to labels such as final-answer correctness, tool verification, and fault propagation. These labels are measurements produced by parsers, behavioral detectors, and harness rules, and they may not fully capture what the trace shows. We study this in a tool-use setting built on GSM8K, where agents call a calculator whose outputs are occasionally corrupted, giving clear ground truth for what the tool returned, what the model later used, and what the detectors reported. We audit three trace-level detectors across six production models and 30 problems per model. Final correctness, visible re-verification, and fault propagation can disagree in systematic ways. A key failure mode is silent tool-output override: the calculator returns a corrupted value, but the model later uses a different, often correct, value without calling the tool again, so the bad value stops propagating while a 
    
[^296]: 精细化拓扑遍次调度：SE(3)等变结构预测中动态k近邻图的阻尼信赖域补偿

    Refined Topology-Pass Scheduling: Damped Trust-Region Compensation for Dynamic k-NN Graphs in SE(3)-Equivariant Structure Prediction

    [https://arxiv.org/abs/2610.19246](https://arxiv.org/abs/2610.19246)

    提出一种双遍拓扑调度方案，将动态kNN图构建基准的选择形式化为以逐块重建（PB）为唯一不动点的阻尼不动点迭代，并借助不翻转证书、最小方差（BLUE）重放增益与ℓ∞信赖域投影进行保守补偿，从而在SE(3)等变结构预测中以低成本逼近逐块重建的拓扑精度。

    

    等变结构预测流水线在坐标更新与k近邻（kNN）图构建之间交替进行，其中图的成员关系是坐标的离散、分段常数函数。在每个块边界（PB）重建该图是训练轮次中最昂贵的部分，而复用一个冻结图（F）虽然快速，却会留下拓扑差距。我们研究了一种双遍设计：廉价的遍次记录少量状态信息（间隔边距、kNN半径、实测的每边界位移），第二遍则进行保守补偿。我们将图构建基准的选择形式化为一个阻尼不动点迭代，其唯一不动点恰好为PB，并由此推导出：(i) 具有双侧常数 $m_i/4$ 和单侧常数 $m_i/2$ 的精确不翻转（non-flip）证书；(ii) 最小方差（BLUE）重放增益 $\gamma^*$ 及由此产生的可行性交叉点；(iii) 一个 $\ell_\infty$ 信赖域投影，可保……

    arXiv:2610.19246v1 Announce Type: new  Abstract: Equivariant structure-prediction pipelines alternate coordinate updates with the construction of a k-nearest-neighbour (kNN) graph whose membership is a discrete, piecewise-constant function of the coordinates. Rebuilding that graph at every block boundary (PB) is the expensive part of an epoch, whereas reusing one frozen graph (F) is fast but leaves a topology gap. We study a two-pass design in which the cheap pass records a small amount of state (margins, kNN radii, measured per-boundary displacements) and a second pass compensates conservatively. We formulate the choice of the graph-construction basis as a damped fixed-point iteration whose unique fixed point is exactly PB, and derive (i) exact non-flip certificates with the two-sided constant $m_i/4$ and one-sided constant $m_i/2$, (ii) the minimum-variance (BLUE) replay gain $\gamma^*$ and the resulting feasibility crossover, (iii) an $\ell_\infty$ trust-region projection that prese
    
[^297]: 强朗道能级混合下的非阿贝尔量子霍尔态：用第一性原理人工智能寻找任意子

    Non-Abelian quantum Hall states at strong Landau level mixing: finding anyons with first-principles AI

    [https://arxiv.org/abs/2610.18967](https://arxiv.org/abs/2610.18967)

    该研究首次完整求解了包含全部电子与所有朗道能级的非阿贝尔量子霍尔问题，通过仅由微观哈密顿量优化、无任何先验物理输入的动量分辨神经网络变分波函数，在 ν=3/2 的自旋极化电子体系中发现了非阿贝尔任意子所对应的拓扑简并基态流形。

    

    非阿贝尔任意子——作为容错拓扑量子计算基础的奇特粒子——被预测出现在分母为偶数填充因子的分数量子霍尔态中。确定实现的是何种拓扑序一直是一个具有挑战性的问题，这需要对真实的朗道能级混合进行精确的非微扰处理，而迄今为止的理论研究一直缺乏这一处理。在本工作中，我们首次完整求解了非阿贝尔量子霍尔问题，涵盖了全部电子和所有朗道能级，研究对象为填充因子 ν = 3/2 的自旋极化电子。我们开发了一种动量分辨的神经网络变分波函数，仅基于微观哈密顿量进行优化，从而能够直接获取拓扑简并的基态。在没有任何关于预期结果的物理先验输入的情况下，该方法发现了一个简并基态流形，其纠缠……（原文摘要在此处被截断）

    arXiv:2610.18967v2 Announce Type: replace-cross  Abstract: Non-Abelian anyons, strange particles forming the basis of fault-tolerant topological quantum computing, are predicted to emerge in fractional quantum Hall states at even-denominator filling factors. Identifying which topological order is realized has proven to be a challenging problem that requires an accurate, non-perturbative treatment of realistic Landau level mixing, which has so far been missing from theoretical studies. Here we present the first complete solution of non-Abelian quantum Hall problem that includes all electrons and all Landau levels, for spin-polarized electrons at filling factor $\nu = 3/2$. We develop a momentum-resolved neural-network variational wave function, optimized solely on the microscopic Hamiltonian, which gives direct access to the topologically degenerate ground states. Without any physics input about the expected outcome, it discovers a degenerate ground-state manifold whose entanglement rev
    
[^298]: ResearchFlow-Eval：追踪端到端多论断自主研究中的执行差距

    ResearchFlow-Eval: Tracing the Execution Gap in End-to-End Multi-Claim Autoresearch

    [https://arxiv.org/abs/2610.18088](https://arxiv.org/abs/2610.18088)

    该论文提出 ResearchFlow-Eval——一个基于已发表论文锚定、在阶段与论断两个层面细粒度评估端到端多论断自主研究系统持续科学执行与迭代能力的评测框架，弥补了现有评估仅打分孤立任务或最终成果的不足。

    

    自主研究系统越来越多地承担起完整的研究项目。在实践中，科学研究要求系统通过持续的、多阶段的工作流程来追求多个相互依赖的论断，并随着证据的积累不断修订实验。然而，现有的自主研究系统评估通常只对孤立任务或最终成果进行打分，使得这些过程层面的能力在很大程度上未被衡量。因此，我们提出了 ResearchFlow-Eval，这是一个在阶段与论断两个层面上评估持续科学执行与迭代的细粒度框架。每项评估都锚定于一篇已发表的论文，从中推导出针对各阶段的评分细则，将每个研究阶段分解为多个维度和分别打分的细则条目。利用这些评分细则，我们对每个研究阶段进行评分，并详细考察实验执行情况，重点关注多论断的完成度以及跨多次尝试的修订。我们将该评估应用于（原文摘要在此处截断）

    arXiv:2610.18088v2 Announce Type: new  Abstract: Autoresearch systems increasingly undertake entire research projects. In practice, scientific research requires systems to pursue multiple interdependent claims through a continuous, multi-stage workflow and revise experiments as evidence accumulates. Existing evaluations of autoresearch systems, however, typically score isolated tasks or final artifacts, leaving these process-level capabilities largely unmeasured. We therefore introduce ResearchFlow-Eval, a fine-grained framework for assessing continuous scientific execution and iteration at both the stage and claim levels. Each evaluation is grounded in a published anchor paper, from which we derive stage-specific rubrics that decompose each research stage into dimensions and separately scored rubric items. Using these rubrics, we score each research stage and examine experimental execution in detail, focusing on multi-claim completion and revisions across attempts. We apply the evalua
    
[^299]: 释放内在的Jev：开放决策模型的简单配方

    Releasing the Inner Jev: A Simple Recipe for Open Decision Models

    [https://arxiv.org/abs/2610.17205](https://arxiv.org/abs/2610.17205)

    该论文证明现有开放指令微调大语言模型无需任何额外训练，即可通过读取最后一个token位置logits中的选项字母，实现与Jev相当的单次前向传播决策能力，并借助LoRA微调加决策头的简单配方进一步提升性能，同时发布了包含2,352个问题的OpenDecisions验证数据集。

    

    基于大语言模型（LLM）的决策系统，也称为“系统一”模型，近来广受欢迎，其中最著名的是Jev（Almeida, 2026）。给定一个类型化的输入状态、一个问题和一组选项，这类模型在单次前向传播中返回每个选项的概率，而无需进行文本生成。我们证明，现有的开放指令微调语言模型在**不做任何额外训练**的情况下也能取得可比的性能，方法是在最后一个token位置上，从模型输出logits中的字母（A、B、C等）读取答案。此外，我们还展示了一种简单的监督微调配方可以进一步提升其性能。我们评估了四个开放指令微调语言模型：Gemma 4 31B、Gemma 4 12B、Qwen3.6-27B和Qwen3.5-4B。我们使用LoRA以及一个从语言模型头切片初始化的决策头对每个模型进行微调。此外，我们推出了OpenDecisions，一个新颖的包含2,352个问题的验证数据集，旨在……

    arXiv:2610.17205v1 Announce Type: new  Abstract: LLM-based decision-making systems, also called System One models, have recently gained popularity, most notably through Jev (Almeida, 2026). Given a typed input state, a question, and a set of options, they return a probability for each option in a single forward pass, without text generation. We show that existing open instruction-tuned language models can achieve comparable performance \textit{without any additional training}, by reading out answers from letters (A, B, C, etc.) in the model's output logits at the last token position. Furthermore, we show how a simple supervised fine-tuning recipe can improve their performance further. We evaluate four open instruction-tuned language models: Gemma 4 31B, Gemma 4 12B, Qwen3.6-27B, and Qwen3.5-4B. We fine-tune each using LoRA and a decision head initialized from a slice of the language model head. Furthermore, we introduce OpenDecisions, a novel 2,352-question validation dataset designed 
    
[^300]: 面向平均奖励 $Q$-学习与 $Q$-规划的混合正则化方法

    Mixing Regularization for Average-Reward $Q$-Learning and $Q$-Planning

    [https://arxiv.org/abs/2610.14988](https://arxiv.org/abs/2610.14988)

    该论文提出通过向一个与奖励无关的 $m$ 步混合策略进行正则化来控制平均奖励 $Q$-学习的收缩与探索性质，并给出有限时间收敛保证，在单步情形下匹配最佳已知速率，在 $m$ 步混合情形下首次推导出新的速率。

    

    强化学习的平均奖励设定具有挑战性，因为其贝尔曼算子仅在底层马尔可夫链具有混合性时才在跨度上收缩。现有工作通常假设存在收缩性质（例如通过环境混合性），而我们考虑这样的设定：给定一个与奖励无关的 $m$ 步混合策略，并通过向该混合策略正则化来控制所学策略的收缩性质与探索性质。我们对 $Q$-学习算法提供了有限时间分析，这些算法可以用多种正则化选择实例化，其保证相对于正则化后的不动点来度量。我们的框架揭示了新的、依赖于正则化子的对象，这些对象控制收缩、探索和算子偏差。在单步设定中，我们的速率匹配了 $Q$-迭代和同步 $Q$-学习的最佳已知速率；对于 $m$ 步混合情形，我们推导出了新的速率。

    arXiv:2610.14988v1 Announce Type: cross  Abstract: The average-reward setting for reinforcement learning is challenging because its Bellman operator is known to contract in span only when the underlying Markov chain is mixing. Whereas existing work typically assumes a contraction property, e.g., through environment mixing, we consider the setting where we are given a reward-agnostic $m$-step mixing policy, and we control the contraction and exploration properties of the learned policy by regularizing toward the mixing policy. We provide finite-time analysis of $Q$-learning algorithms that can be instantiated with multiple choices of regularization, with guarantees measured with respect to the regularized fixed point. Our framework exposes novel regularizer-dependent objects that control contraction, exploration, and operator bias. In the one-step setting, our rates match the best known rates for $Q$-iteration and synchronous $Q$-learning, and for $m$-step mixing we derive new rates. We
    
[^301]: YOUDrive：将驾驶风格作为端到端自动驾驶的可组合方向

    YOUDrive: Driving Style as a Composable Direction for End-to-End Autonomous Driving

    [https://arxiv.org/abs/2610.14802](https://arxiv.org/abs/2610.14802)

    YOUDrive提出了一种视觉-语言-动作规划器，通过将教师驾驶行为蒸馏为可缩放、可组合的风格方向，无需重新训练即可在推理阶段对端到端自动驾驶策略的驾驶风格进行连续精细控制。

    

    arXiv:2610.14802v1 公告类型： cross 摘要：端到端驾驶策略虽然能够实现出色的任务性能，但对其驾驶风格的控制却十分有限。驾驶风格描述的是车辆如何加速、制动和保持车距。在没有显式偏好条件约束下训练的策略，缺乏在推理阶段直接调整这些行为的机制。我们提出了YOUDrive，一个用于连续且可组合驾驶风格控制的视觉-语言-动作规划器。每位教师（teacher）的行为被蒸馏成一个任务向量，即一种学习到的参数更新，它在规划器的表示空间中诱导出一个风格方向。这些方向可以在推理阶段进行缩放和组合，而无需重新训练主干网络。在这些方向的条件下，流匹配解码器对多模态轨迹分布进行建模，并生成运动学上可行的运动备选方案。我们引入了风格对齐分数，用于描述在安全和任务约束下参照人类的运动学行为表现。

    arXiv:2610.14802v1 Announce Type: cross  Abstract: End-to-end driving policies can achieve strong task performance while offering limited control over their driving style. Driving style describes how a vehicle accelerates, brakes, and maintains distance. Policies trained without explicit preference conditioning lack a direct mechanism for adjusting these behaviors at inference. We propose YOUDrive, a vision--language--action planner for continuous and composable driving-style control. Each teacher's behavior is distilled into a task vector, a learned parameter update that induces a style direction in the planner's representation. These directions can be scaled and combined at inference without retraining the backbone. Conditioned on these directions, a flow-matching decoder models a multimodal trajectory distribution and generates kinematically feasible motion alternatives. We introduce the Style Alignment Score (SAS) to describe human-referenced kinematic behavior after safety and tas
    
[^302]: MultiWorldBench：独立控制的视角能否描述同一个共享世界？

    MultiWorldBench: Do Independently Controlled Views Describe One Shared World?

    [https://arxiv.org/abs/2610.11723](https://arxiv.org/abs/2610.11723)

    该论文提出了MultiWorldBench——一个基于Minecraft、包含495个案例配置和十种能力的多人世界模型诊断基准，用于评估独立控制视角是否能保持与共享持久世界的一致性，并发现现有生成式系统（Gamma-World、Solaris、MineWorld）与真实引擎参考（Engine GT）之间仍存在巨大差距，尤其在状态持久性和结构推理方面表现薄弱。

    

    多人世界模型必须确保独立控制的各个视角与一个共享且持久的世界保持一致。我们提出了MultiWorldBench，这是一个诊断性Minecraft基准测试，包含495个案例配置，涵盖七个任务套件和十种能力，包括独立控制、跨视角运动、共享状态同步、持久性、结构推理、并发交互和延迟重访。我们评估了Solaris、Gamma-World和MineWorld，并使用Engine GT作为参考。在生成式系统中，Gamma-World在十项能力上的平均分最高，为21.39，其次是Solaris为20.88，MineWorld为1.89，而Engine GT达到91.69。Gamma-World在多项控制、共享状态和重访能力上表现更佳，而Solaris在跨视角运动和竞态条件一致性方面领先。然而，所有生成式系统在状态持久性方面得分最高仅为8.00，在结构方面得分为1.33。

    arXiv:2610.11723v1 Announce Type: new  Abstract: Multiplayer world models must ensure that independently controlled views remain consistent with one shared and persistent world. We introduce MultiWorldBench, a diagnostic Minecraft benchmark containing 495 case configurations across seven task suites and ten capabilities, including independent control, cross-view motion, shared-state synchronization, persistence, structural reasoning, concurrent interaction, and delayed revisit. We evaluate Solaris, Gamma-World, and MineWorld, using Engine GT as a reference. Gamma-World achieves the highest ten-capability average among the generated systems at 21.39, followed by Solaris at 20.88 and MineWorld at 1.89, while Engine GT reaches 91.69. Gamma-World performs better on several control, shared-state, and revisit capabilities, whereas Solaris leads in cross-view motion and race-condition consistency. Nevertheless, all generated systems score at most 8.00 on state persistence and 1.33 on structur
    
[^303]: 通过来自暗知识的模型无关潜在安全信号保障大语言模型安全

    Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge

    [https://arxiv.org/abs/2610.07532](https://arxiv.org/abs/2610.07532)

    提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。

    

    大语言模型（LLM）发展迅速，引发了人们对其安全性日益增长的关注。近期工作提出了检测和防御攻击的方法，包括利用模型隐藏状态的解码阶段防御。然而，现有的解码阶段防御存在两个局限性。首先，它们在安全性与过度拒绝之间存在权衡，即加强安全性会降低模型对良性查询的有用性。其次，许多这些方法依赖于内部隐藏状态，因此仅限于特定架构，带来大量开销且在不同模型间的泛化能力有限。为解决这些限制，我们提出了 LADE（Latent Safety Signals for Defense），该方法通过在首token输出概率分布中，对比有害查询与良性查询，从暗知识（即输出概率分布中超出argmax所携带的信息）中提取潜在安全信号……

    arXiv:2610.07532v1 Announce Type: cross  Abstract: LLMs have advanced rapidly, raising growing concerns about their safety. Recent work has proposed approaches to detect and defend against attacks including defenses at decoding stage that leverage models' hidden states. However, existing decoding-stage defenses suffer from two limitations. First, they introduce a trade-off between safety and over-refusal, where strengthening safety degrades the model's helpfulness on benign queries. Second, many of these methods rely on internal hidden states and are thus restricted to specific architectures, incurring substantial overhead and limited generalization across models. To address these limitations, we introduce LADE (Latent Safety Signals for Defense), which leverages latent safety signals extracted by contrasting harmful and benign queries from dark knowledge (i.e., information carried by the output probability distribution beyond its argmax) in the first-token output probability distribut
    
[^304]: Kandinsky 6.0 Video：同步视频与音频生成的基础模型

    Kandinsky 6.0 Video: Foundation Models for Synchronized Video and Audio Generation

    [https://arxiv.org/abs/2610.05608](https://arxiv.org/abs/2610.05608)

    Kandinsky 6.0 Video 是一系列用于同步文本到音频-视频生成的基础扩散模型，其核心创新是采用双流 CrossDiT 架构，通过双向交叉注意力实现视频流与音频流的时间和语义对齐，可生成带同步44 kHz音频（含唇音同步）的5秒全高清视频。

    

    我们提出了 Kandinsky 6.0 Video，这是一个用于同步文本到音频-视频生成的基础扩散模型家族，包含 Kandinsky 6.0 Video Lite（30亿参数）和 Kandinsky 6.0 Video Pro（290亿参数）两款模型。两款模型均可在文本到音频-视频（T2AV）和图像到音频-视频（I2AV）两种模式下生成5秒视频片段，并配有同步的44 kHz音频（包括唇音同步）；内置的超分辨率模型可将输出分辨率提升至全高清（1920x1080）。基于 Kandinsky 5.0 的视频生成能力，Kandinsky 6.0 Video 采用双流 CrossDiT 架构，通过双向交叉注意力连接预训练的视频流与新训练的音频流，实现时间与语义上的对齐。我们的持续预训练策略首先在大规模音频语料库上从头训练音频流，然后在配对的音频-视频数据上联合训练两个流，同时保持单模态能力……（摘要原文在此处截断）

    arXiv:2610.05608v2 Announce Type: replace-cross  Abstract: We present Kandinsky 6.0 Video, a family of foundation diffusion models for synchronized text-to-audio-video generation, comprising Kandinsky 6.0 Video Lite (3B parameters) and Kandinsky 6.0 Video Pro (29B parameters). Both models generate 5-second video clips with synchronized 44 kHz audio, including lip-sync, in text-to-audio-video (T2AV) and image-to-audio-video (I2AV) modes; a built-in super-resolution model raises the output resolution to Full-HD (1920x1080). Building on the video generation capabilities of Kandinsky 5.0, Kandinsky 6.0 Video employs a dual-stream CrossDiT architecture that connects a pretrained video stream and a newly trained audio stream through bidirectional cross-attention for temporal and semantic alignment. Our continuous pretraining strategy first trains the audio stream from scratch on large-scale audio corpora and then trains both streams jointly on paired audio-video data while preserving unimoda
    
[^305]: AgroGround：农业领域中的多粒度定位识别

    AgroGround: Multi-Granularity Grounded Recognition in Agriculture

    [https://arxiv.org/abs/2610.04425](https://arxiv.org/abs/2610.04425)

    提出大规模农业定位识别数据集 AgroGround，通过自动化流程将八个农业视觉问答数据集的标签转换为病害病斑与完整目标的区域标注（共794,850条指令样本），并利用健康图像作为负向监督，微调视觉-语言模型实现植物病害等目标的识别与定位。

    

    农业视觉模型通常仅在识别或定位二者之一上进行评估，但可靠的诊断需要同时识别图像中存在什么目标并定位其证据区域。农业视觉问答（VQA）数据集虽然带有丰富的语义标签，却很少将这些标签与图像区域关联起来，而以人工方式大规模添加此类标注成本高昂。我们提出了 AgroGround，一个面向农业定位识别的大规模数据集，即识别植物病害及其他农业目标并定位其在图像中的对应区域。一个自动化流程将八个农业视觉问答数据集的标签转换为病害病斑级和完整目标级的标注，共生成 794,850 条指令样本。健康图像为病害查询提供负向监督，教会模型在无病害时返回空预测。我们在已知目标的定位指令与需要……

    arXiv:2610.04425v2 Announce Type: replace-cross  Abstract: Agricultural visual models are typically evaluated for either recognition or localization, but reliable diagnosis requires identifying what is present and localizing the evidence. Agricultural visual question answering (VQA) datasets carry rich semantic labels but rarely link them to image regions, and adding such annotations by hand is costly at scale. We introduce AgroGround, a large-scale dataset for grounded agricultural recognition: identifying plant diseases and other agricultural targets and localizing their image regions. An automated pipeline converts the labels of eight agricultural VQA datasets into annotations for disease lesions and whole objects, producing 794,850 instruction examples. Healthy images provide negative supervision for disease queries, teaching the model to return empty predictions. We fine-tune a shared vision-language model on known-target grounding instructions combined with instructions requiring
    
[^306]: 用任务算子捕捉上下文学习动态

    Capturing In-Context Learning Dynamics with Task Operators

    [https://arxiv.org/abs/2610.01054](https://arxiv.org/abs/2610.01054)

    提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。

    

    上下文学习（ICL）使语言模型能够仅凭示例演示执行新任务，而无需更新权重。然而，每次ICL推理都需要处理完整的示例集合，导致部署效率低下，且ICL的作用机制尚未被完全理解。先前的工作将ICL压缩为从特定层或位置提取的固定激活向量，但这些与输入无关的干预方法在复杂任务上会失效，因为这类任务的输出取决于与输入之间的细粒度交互。通过分析ICL的前向传递过程，我们发现每个注意力头的输出是其上下文掩码对应输出的仿射变换，且该变换的参数在给定任务的不同样本间具有经验上的稳定性。基于这一发现，我们提出了任务算子（Task Operator, TO），它将这一变换作为解析推导出的更新重放到注意力输出投影中。在词汇类、算法类……（原文摘要截断）

    arXiv:2610.01054v1 Announce Type: cross  Abstract: In-context learning (ICL) enables language models to perform new tasks from demonstrations without weight updates. However, every ICL inference requires processing the full set of examples, resulting in inefficient deployments, and how ICL works mechanistically is not fully understood. Prior work compresses ICL into fixed activation vectors extracted from specific layers or positions, but these input-independent interventions fail on complex tasks where the output depends on fine-grained interactions with the input. By analyzing the ICL forward pass, we show that each attention head's output is an affine transformation of its context-masked counterpart, and that the parameters of this transformation are empirically stable across samples for a given task. Building on this, we introduce Task Operator (TO), which replays this transformation as an analytically derived update to the attention output projection. Across lexical, algorithmic, 
    
[^307]: GroundingPI：一个基于视觉原语、面向物理智能的视觉定位基础模型

    GroundingPI: A Grounding Foundation Model towards Physical Intelligence with Visual Primitives

    [https://arxiv.org/abs/2609.39601](https://arxiv.org/abs/2609.39601)

    GroundingPI是一个40亿参数的视觉定位基础模型，它将点和框表示为共享词表中的量化坐标，并通过多模态与空间预训练、监督微调及GRPO强化学习训练而成，在11种感知能力的34个基准测试上以平均73.68%的成绩超越更大的GPT-6 Astra，创下新的最先进水平，为物理智能提供了快速精确的视觉感知基础。

    

    精确的视觉定位至关重要。它需要指明哪个物体是目标以及该物体位于何处，即使在杂乱场景和微小物体的情况下也必须准确，并且必须足够快速以支持闭环控制。然而，视觉-语言-动作模型（VLA）和世界-动作模型（WAM）通常从通用的视觉-语言骨干网络和视频生成骨干网络获取感知能力，而这些骨干网络在上述场景中仍然表现不佳。我们提出了GroundingPI，一个40亿参数的视觉定位基础模型，它以共享词表中的量化坐标形式生成点和边界框。其训练结合了多模态与空间预训练、监督微调以及基于GRPO的强化学习，并利用公开数据集和专用数据引擎提供监督信号。在与涵盖11种感知能力的34个视觉定位基准上的44个基线模型的对比中，GroundingPI确立了新的最先进水平，平均成绩达到73.68%，超过了参数量更大的GPT-6 Astra（71.54%）。作为下游任务的视觉骨干网络，GroundingPI提升了……（摘要在此处被截断）

    arXiv:2609.39601v1 Announce Type: cross  Abstract: Precise grounding matters. It specifies which object is the target and where that object is, even in clutter and for tiny objects, and it has to be fast enough for closed-loop control. Yet vision-language-action (VLA) and world-action models (WAMs) take perception from general-purpose vision-language and video-generation backbones, which still fail in these settings. We introduce GroundingPI, a 4B grounding foundation model that generates points and boxes as quantized coordinates in a shared vocabulary. Training combines multimodal and spatial pretraining, supervised fine-tuning, and reinforcement learning with GRPO, using supervision from public datasets and dedicated data engines. Against 44 baselines across 34 grounding benchmarks spanning 11 perceptual capabilities, GroundingPI establishes a new state of the art, averaging 73.68%, above the larger GPT-6 Astra (71.54%). As a downstream visual backbone, GroundingPI improves performan
    
[^308]: GroundAnything：以闪速实现并行解码与精准视觉定位的统一

    GroundAnything: Reconciling Parallel Decoding with Precise Visual Grounding at Flash Speed

    [https://arxiv.org/abs/2609.39600](https://arxiv.org/abs/2609.39600)

    GroundAnything 是一个 40 亿参数的视觉定位基础模型，它将视觉定位重新定义为可并行生成的双向扩散过程，通过分块去噪、AR 到扩散的架构转换训练及 GRPO 强化后训练，在实现闪速并行解码的同时保证了精准的视觉定位效果。

    

    自回归（AR）视觉定位模型将空间预测序列化，引入了顺序延迟，并对输出 token 施加了因果顺序约束。我们将视觉定位视为视觉证据提取：物体、位置和空间关系由图像和查询共同约束，但它们之间的依赖关系并不意味着存在内在的从左到右的生成顺序。这一区别使得双向扩散成为天然合适的选择，允许空间假设并行涌现，并通过迭代去噪进行联合优化。我们提出了 GroundAnything，一个 40 亿参数的视觉定位基础模型，通过分块去噪实现了快速并行解码与精准定位的统一。其训练过程结合了基于公开数据集与专用数据引擎的定位预训练、采用 AR 与扩散联合目标的直接 AR 到扩散架构转换、监督微调以及基于 GRPO 的强化后训练。在 30 个视觉定位基准测试上……

    arXiv:2609.39600v1 Announce Type: cross  Abstract: Autoregressive (AR) grounding models serialize spatial predictions, introducing sequential latency and imposing a causal order on output tokens. We view grounding as visual evidence extraction: objects, locations, and spatial relations are jointly constrained by the image and query, yet their dependencies do not imply an intrinsic left-to-right generation order. This distinction makes bidirectional diffusion a natural fit, allowing spatial hypotheses to emerge in parallel and be jointly refined through iterative denoising. We introduce GroundAnything, a 4B-parameter grounding foundation model that reconciles fast parallel decoding with precise localization through blockwise denoising. Training combines grounding pretraining from public datasets and dedicated data engines, direct AR-to-diffusion conversion with joint AR and diffusion objectives, supervised fine-tuning, and GRPO-based reinforcement post-training. Across 30 grounding benc
    
[^309]: 早期问题的计算能否帮助大语言模型解决新问题？

    Can Computation from Earlier Problems Help LLMs Solve New Ones?

    [https://arxiv.org/abs/2609.39394](https://arxiv.org/abs/2609.39394)

    提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。

    

    大语言模型经常在同一对话中解决相互独立的问题。那么，来自较早问题的计算能否帮助它们解决新问题？为了回答这个问题，我们首先进行了初步实验，结果表明保留的历史记录既可能提高也可能降低后续轮次的准确率，即使是在同一领域内也是如此。为了理解这些影响，我们采用受控重放方法来隔离特定于每个“问题-历史”配对的内部状态变化。结果显示，在不同的历史条件下，这些变化在当前问题之间保持了相似的关系。为了在保留历史的情况下改进推理，我们提出了 STAIR（Stale-Token Attention for Inter-query Reuse，面向跨查询重用的陈旧令牌注意力机制）。STAIR 将较早响应生成过程中的键和值捕获到一个固定存储库中，并学习在提示处理期间当前查询读取该存储库时对其进行重定向。基础模型保持完全冻结，仅训练 12,288 个参数。在三个 Qwen 模型和四个基准测试上，STAIR 提升了平均准确率（原文在此处截断）。

    arXiv:2609.39394v1 Announce Type: new  Abstract: Large language models often solve independent problems in the same conversation. Can computation from earlier problems help them solve new ones? To answer this question, we first conduct preliminary experiments showing that retained history can raise or lower later-turn accuracy, even within the same domain. To understand these effects, we use controlled replay to isolate internal state changes specific to each problem-history pairing. Across different histories, these changes preserve similar relationships among current problems. To improve reasoning under retained history, we introduce STAIR (Stale-Token Attention for Inter-query Reuse). STAIR captures keys and values from earlier response generation in a fixed bank. It learns to redirect current queries when they read this bank during prompt processing. The base model remains frozen; only 12,288 parameters are trained. Across three Qwen models and four benchmarks, STAIR improves avera
    
[^310]: STEPQuant：Delta规则循环状态量化中误差何时何地产生影响

    STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization

    [https://arxiv.org/abs/2609.38169](https://arxiv.org/abs/2609.38169)

    STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。

    

    线性注意力机制用固定大小的循环状态取代了不断增长的KV缓存，但这些持久状态在并发服务场景下可能成为显著的内存瓶颈。直接将循环状态量化为低精度往往会导致严重的精度下降，因为量化误差会在连续的状态更新中不断传播。我们发现这些误差的影响取决于两个互补的维度：在时间维度上，长期存活的记忆中的误差会在多个解码步骤中持续存在；在空间维度上，不同键行中的误差对模型输出的影响各不相同，而状态幅值在行和列两个方向上都存在显著差异。基于这些观察，我们提出了STEPQuant，一个面向Delta规则循环状态的时空后训练量化框架。STEPQuant根据误差大小和记忆寿命来分配精度，并基于状态分布联合拟合键行与值列的缩放因子。

    arXiv:2609.38169v1 Announce Type: cross  Abstract: Linear attention replaces growing KV caches with fixed-size recurrent states, yet these persistent states can become a substantial memory bottleneck under concurrent serving. Directly quantizing recurrent states to low precision often leads to severe accuracy degradation, as quantization errors propagate through successive state updates. We discover that the impact of these errors depends on two complementary dimensions: temporally, errors in long-lived memory can persist across many decoding steps; spatially, errors in different key rows affect model outputs differently, while state magnitudes vary substantially along both rows and columns. Motivated by these observations, we propose STEPQuant, a spatial-temporal post-training quantization framework for Delta-rule recurrent states. STEPQuant allocates precision according to error magnitude and memory lifetime, and jointly fits key-row and value-column scales based on state distributio
    
[^311]: 预测式自监督学习可在干扰存在下被证明能够识别随机信号

    Predictive Self-Supervised Learning Provably Identifies Stochastic Signals under Nuisance

    [https://arxiv.org/abs/2609.37789](https://arxiv.org/abs/2609.37789)

    该论文证明了常见的预测式自监督学习方法通过“预测互信息最大化”与“潜分布匹配”两个互补原理，能够在干扰信息存在下可证地恢复与预测相关的随机潜在信号。

    

    自监督学习（SSL）通过在潜空间中进行预测，而无需生成输入数据本身，能够学习到高度抽象且有用的表示。直观上，这一成功通常被归因于其能够丢弃与预测无关的干扰信息。然而，这带来了一个难题：与预测相关的潜在信号中的随机变化和真正的干扰因素都会使观测结果部分不可预测；二者如何区分？令人惊讶的是，我们证明了常见的SSL方法恰好能够实现这一点：它们隐式地实例化了一个具有随机动力学和观测私有干扰的潜变量模型。我们将这类方法恢复随机信号的能力归结为两个互补的原理：预测互信息最大化确保表示保留预测所需的信息，而潜分布匹配则约束了这一信息的编码方式，从而使得……（原文摘要在此处截断）

    arXiv:2609.37789v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) by predicting in latent space, without generating the input data itself, learns highly abstract, useful representations. Intuitively, this success is often attributed to its ability to discard nuisance information that is irrelevant to prediction. However, this poses a conundrum: both stochastic variation in a prediction-relevant latent signal and true nuisance make observations partly unpredictable; how could they be distinguished? Surprisingly, we prove that common SSL methods can achieve exactly this, by implicitly instantiating a latent-variable model with stochastic dynamics and observation-private nuisance. We trace their ability to recover the stochastic signal to two complementary principles: Predictive mutual information maximization ensures that representations retain the information needed for prediction, while latent distribution matching constrains how this information is encoded, thereby mak
    
[^312]: 端口哈密顿潜空间深思：缓解测试时计算扩展中的深思漂移悬崖

    Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling

    [https://arxiv.org/abs/2609.37351](https://arxiv.org/abs/2609.37351)

    该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。

    

    测试时计算扩展已成为先进机器推理的基石，然而在连续潜表示空间中直接执行迭代深思会暴露出一种灾难性病理现象：深思漂移悬崖。尽管无约束的循环潜空间模型在较短的推理步数（K ≤ 4）下能获得初步的推理收益，但当外推至更深的思考步数（K ≥ 16）时，其推理能力会发生崩溃，在标准逻辑基准测试中性能下降22%至62%。我们通过22轮实证与理论研究，解决了测试时潜空间推理中表达能力、Lyapunov稳定性与计算效率之间的三难困境。我们证明，严格保守的标量势梯度流能够抑制长程漂移（悬崖仅为3.40%），但会将峰值推理准确率瓶颈在32.73%；而无约束的旋转流虽实现了高符号表达能力（82.33%），却会遭受高达36.87%的严重漂移。

    arXiv:2609.37351v1 Announce Type: cross  Abstract: Test-time compute scaling has emerged as a cornerstone of advanced machine reasoning, yet performing iterative deliberation directly within continuous latent representation spaces reveals a catastrophic pathology: the Deliberation Drift Cliff. While unconstrained recurrent latent models achieve initial reasoning gains at short horizons (K <= 4), their reasoning collapses when extrapolated to deeper thinking steps (K >= 16), dropping by 22% to 62% across standard logical benchmarks. We resolve the trilemma among expressivity, Lyapunov stability, and computational efficiency in test-time latent reasoning through a 22-round empirical and theoretical investigation. We demonstrate that strictly conservative scalar potential gradient flows suppress long-range drift (cliff 3.40%) but bottleneck peak reasoning accuracy at 32.73%, whereas unconstrained rotational flows achieve high symbolic expressivity (82.33%) but suffer a severe 36.87% drift
    
[^313]: DatalogBench：评估大型语言模型在文本到Datalog程序合成上的能力

    DatalogBench: Evaluating Large Language Models on Text-to-Datalog Synthesis

    [https://arxiv.org/abs/2609.37233](https://arxiv.org/abs/2609.37233)

    DatalogBench是一个包含136个从自然语言问题合成Datalog程序任务的基准，系统评估显示当前大语言模型的精确匹配率最高仅达68.4%，且多数失败源于模型凭空发明了未声明的辅助谓词。

    

    Datalog是程序分析等推理任务的基础，但其程序难以编写。现有的程序合成工具虽然可以自动化这一任务，但需要用户以输入-输出示例的形式表达意图。大型语言模型（LLM）提供了一条更自然的路径——从自然语言问题出发进行文本到Datalog的合成，然而它们在这方面的表现尚未得到系统性评估。我们提出了DatalogBench，这是一个包含136个文本到Datalog合成任务的基准，这些任务从现有的基于Datalog的工件中精选而来。合成程序通过在保留的输入上执行并与经变异分析验证的预言机（oracle）进行比对来评分。在六种LLM和四种提示配置下，精确匹配率最高仅达68.4%，而提供关系描述或输入-输出示例只有有限的、依赖具体模型的效果。在直接提示下，大多数失败发生在编译阶段，通常是因为模型凭空发明了它从未声明或未正确定义类型的辅助谓词。

    arXiv:2609.37233v1 Announce Type: cross  Abstract: Datalog underpins reasoning tasks such as program analysis, but its programs are hard to write. Existing synthesizers automate this task but require users to state their intent as input-output examples. Large language models (LLMs) suggest a more natural route, text-to-Datalog synthesis from a natural-language question, yet how well they do so has not been systematically evaluated. We present DatalogBench, a benchmark of 136 text-to-Datalog synthesis tasks curated from existing Datalog-based artifacts. Synthesized programs are graded by execution on held-out inputs against an oracle validated by mutation analysis. Across six LLMs and four prompting configurations, exact match peaks at 68.4%, and relation descriptions or an input-output example have only modest, model-dependent effects. Under direct prompting, most failures occur at compile time, typically because a model invents auxiliary predicates that it never declares or types cons
    
[^314]: UpliftMem：面向智能体记忆检索的集合级提升学习

    UpliftMem: Learning Set-Level Uplift for Agent Memory Retrieval

    [https://arxiv.org/abs/2609.36805](https://arxiv.org/abs/2609.36805)

    提出 UpliftMem，以记忆集合相对无记忆执行器带来的集合级执行提升为学习信号，并利用闭式推导的 EVSI 准则分配有限的探测式推演，训练出无需测试时探测即可选出最优记忆集合的共享评分器。

    

    大语言模型（LLM）智能体通过复用外部记忆来指导新任务，但有效的检索需要学习哪些记忆集合能够改善执行效果。这类学习依赖代价高昂的结果反馈：普通检索只能观察到实际被执行的记忆集合，而评估其他备选集合则需要额外的推演。我们提出 UpliftMem，它以同一执行器在无记忆情况下的表现为基准，从集合级的执行提升中学习记忆检索。对检索偏好如何限制反馈覆盖范围的理论分析，启发了对备选记忆集合进行有针对性探测的方法。探测集合的选择遵循样本信息期望价值（EVSI）准则，该准则在相关高斯模型下以闭式形式推导得出，可根据各备选集合对局部检索决策的预期改进来分配有限的训练推演资源。共享的评分器在冻结执行器的条件下进行训练，在测试时无需探测即可直接选择记忆集合。在 ALFWorld 等基准上……（摘要原文在此处截断）

    arXiv:2609.36805v1 Announce Type: new  Abstract: Large language model (LLM) agents reuse external memory to guide new tasks, but effective retrieval requires learning which memory sets improve execution. Such learning relies on costly outcome feedback: ordinary retrieval observes only executed sets, while evaluating alternatives requires additional rollouts. We introduce \textsc{UpliftMem}, which learns memory retrieval from set-level execution uplift relative to the same executor without memory. A theoretical analysis of how retrieval preferences restrict feedback coverage motivates targeted probing of alternative memory sets. Probe selection follows an expected value of sample information (EVSI) criterion, derived in closed form under a correlated Gaussian model, to allocate limited training rollouts according to their expected improvement in local retrieval decisions. The shared scorer is trained with a frozen executor and selects memory sets without test-time probes. Across ALFWorl
    
[^315]: 面向多元时间序列预测的通道依赖状态空间模型

    Channel-Dependent State Space Model for Multivariate Time Series Forecasting

    [https://arxiv.org/abs/2609.36453](https://arxiv.org/abs/2609.36453)

    提出Chameleon，一种通道依赖的状态空间模型，通过将选择性SSM与卡尔曼滤波器结合，利用其缺失的测量更新机制实现变量间数据依赖的细粒度交互，在保持线性复杂度的同时实现稳健的多元时间序列预测。

    

    多元时间序列预测（MTSF）在许多现实世界领域中至关重要。现有的深度学习方法分为两种范式，各有明显的局限性：通道独立（CI）方法无条件地忽略跨变量依赖关系，仅建模时间动态；而通道依赖（CD）方法虽同时考虑两者，但通常需要在架构上做出折衷以缓解过拟合和计算开销问题。因此，我们提出了Chameleon，一种专门设计的CD状态空间模型（SSM），它能够在变量之间实现数据依赖的细粒度交互，同时计算量随变量数量线性扩展。通过将选择性SSM与卡尔曼滤波器联系起来，我们利用前者缺失的测量更新步骤进行跨变量建模，同时保留SSM主干以实现稳健的时间建模。我们进一步识别出GatedDeltaNet对时间序列任务的有利归纳偏置，将其改造作为模型主干，并加以改进（摘要在此处截断）。

    arXiv:2609.36453v1 Announce Type: cross  Abstract: Multivariate time series forecasting (MTSF) is critical across many real-world domains. Existing deep learning approaches fall into two paradigms with distinct limitations: channel-independent (CI) methods unconditionally ignore cross-variable dependencies and model only temporal dynamics, while channel-dependent (CD) methods consider both but typically rely on architectural compromises to mitigate overfitting and computational overhead. We therefore propose Chameleon, a specialized CD state space model (SSM) that enables data-dependent, fine-grained interactions across variables while scaling linearly with their number. By connecting selective SSMs with the Kalman filter, we leverage the missing measurement update in the former for cross-variable modeling while preserving the SSM backbone for robust temporal modeling. We further identify favorable inductive biases of GatedDeltaNet for time series, adapt it as our backbone, and improve
    
[^316]: 蛋白质基础模型适应度预测的通用框架

    A General Harness for Protein Foundation Model Fitness Prediction

    [https://arxiv.org/abs/2609.34654](https://arxiv.org/abs/2609.34654)

    该论文提出了一个通用、模型无关且无需训练的检索增强突变框架 VenusREM-Harness（VRH），通过按模型不确定性融合MSA进化证据，并基于结构置信度和溶剂暴露程度进行门控背景校正与评分收缩，在ProteinGym的1,211个实验和310万变异上有效提升了蛋白质基础模型适应度预测的准确性。

    

    准确的适应度预测是蛋白质工程和理解序列-功能关系的核心。随着深度学习的发展，蛋白质基础模型（PFMs）已被广泛用于这一任务。然而，近期分析表明，这些模型存在反映其训练语料的共同偏好，而不可靠的输入会进一步扭曲适应度预测。家族特异性的进化证据和结构背景可以通过为模型评分提供互补性约束来帮助解决这些局限，由此催生了 VenusREM-Harness（VRH）——一个通用、模型无关、无需训练的检索增强突变框架。它根据模型不确定性将冻结的模型评分与多序列比对（MSA）证据融合，然后基于结构置信度和溶剂暴露程度应用门控背景校正和评分收缩。在来自 ProteinGym 的 1,211 个实验测定和 310 万个测量变异上，Venu……（原文摘要在此处截断）

    arXiv:2609.34654v2 Announce Type: replace  Abstract: Accurate fitness prediction is central to protein engineering and understanding sequence-function relationships. With advances in deep learning, protein foundation models (PFMs) have become widely used for this task. Recent analyses, however, show that these models share preferences reflecting their training corpora, while unreliable inputs can further distort fitness predictions. Family-specific evolutionary evidence and structural context can help address these limitations by providing complementary constraints on model scores, motivating VenusREM-Harness (VRH), a general, model-agnostic, training-free Retrieval-Enhanced Mutation harness. It fuses frozen model scores with multiple sequence alignment (MSA) evidence according to model uncertainty, then applies gated background correction and score shrinkage based on structural confidence and solvent exposure. Across 1,211 assays and 3.1 million measured variants from ProteinGym, Venu
    
[^317]: 一个潜变量，多个词元：联合学习压缩嵌入以实现高效语言扩散

    One Latent, Many Tokens: Jointly Learning Compressed Embeddings for Efficient Language Diffusion

    [https://arxiv.org/abs/2609.33698](https://arxiv.org/abs/2609.33698)

    JPEG-DLM通过联合训练压缩器、流匹配模型和解码模块来学习更结构化、更易建模且可可靠解码的压缩嵌入，在保持较低困惑度的同时实现了扩散语言模型中最高的生成吞吐量。

    

    大多数连续扩散语言模型在每个采样步骤中为每个词元处理一个潜位置，这使得生成成本高昂。两阶段方法通过减少潜变量长度来降低成本，但它们在训练扩散模型之前就固定了压缩嵌入空间。来自固定空间的嵌入可能难以用扩散进行建模，也难以可靠地解码为词元，这限制了压缩后的生成质量。为了解决这个问题，我们提出了JPEG-DLM（面向扩散语言模型高效生成的联合嵌入预测方法），它联合训练一个压缩器、一个流匹配模型和一个解码模块。通过联合嵌入预测，JPEG-DLM学习到的压缩嵌入更具结构化，更容易用扩散建模，并且可以可靠地解码为词元。JPEG-DLM在LM1B和OWT数据集上实现了近期扩散模型和流模型中最高的吞吐量，且其Gen-PPL低于其中大多数模型。

    arXiv:2609.33698v2 Announce Type: replace  Abstract: Most continuous diffusion language models process one latent position per token at each sampling step, making generation expensive. Two-stage methods lower the cost by reducing the latent length, but they fix the compressed embedding space before training the diffusion model. Embeddings from the fixed space can be difficult to model with diffusion and decode reliably into tokens, which limits generation quality after compression. To address this problem, we introduce JPEG-DLM (Joint-embedding Prediction for Efficient Generation with Diffusion Language Model), which jointly trains a compressor, a flow matching model and a decoding module. With joint-embedding prediction, JPEG-DLM learns compressed embeddings that are more structured, easier to model with diffusion and reliably decodable into tokens.JPEG-DLM achieves the highest throughput among recent diffusion and flow models on LM1B and OWT, with a Gen-PPL lower than most of them. A
    
[^318]: 变色龙：面向高效扩散模型的动态格式适配器

    Chameleon: Dynamic Format Adapter for Efficient Diffusion

    [https://arxiv.org/abs/2609.33496](https://arxiv.org/abs/2609.33496)

    Chameleon提出了一种混合格式的扩散模型训练后量化框架，在固定位宽的前提下将数字格式本身作为离散变量，按权重通道、激活层乃至扩散时间步自适应地选择INT8或FP8等格式，从而实现更高效的扩散模型推理。

    

    训练后量化（PTQ）是在内存受限的加速器上运行现代扩散模型的标准方法，然而现有所有的扩散模型PTQ方案都预先固定了数字格式，仅调整缩放因子、零点或逐层位宽。在固定位宽下，最佳格式取决于被编码的分布，而该分布在权重通道之间、层与层之间以及扩散时间步上都各不相同——激活分布会从重尾、噪声主导的形态逐渐演变为紧密聚集且结构化的形态。我们提出Chameleon，这是一个PTQ框架，它保持位宽固定，并将格式本身视为一个离散变量，按权重通道和激活层进行选择，并可选择按时间步分桶进一步细化。激活格式来自{INT8, FP8 E4M3, FP8 E5M2, MXFP8, MXINT8}集合，基于两个低成本的统计量（经验峰度和闭式……[摘要在此处被截断]）提前选定。

    arXiv:2609.33496v2 Announce Type: replace-cross  Abstract: Post-training quantization (PTQ) is the standard way to run modern diffusion models on memory-constrained accelerators, yet every existing diffusion PTQ scheme fixes the $\mathit{number\ format}$ in advance and only tunes the scale, zero point, or per-layer bit-width. At a fixed bit-width the best format depends on the distribution being encoded, and that distribution differs across weight channels, across layers, and along the diffusion timestep, where activation distributions slide from heavy-tailed and noise-dominated to tightly clustered and structured. We propose Chameleon, a PTQ framework that holds the bit-width fixed and treats the format itself as a discrete variable, chosen per weight channel and per activation layer, optionally refined per timestep bucket. Activation formats come from {INT8, FP8 E4M3, FP8 E5M2, MXFP8, MXINT8}, selected ahead of time from two cheap statistics (empirical kurtosis and the closed-form di
    
[^319]: DeepEdu-v1：面向越南教育的高效且可扩展的智能体大语言模型

    DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education

    [https://arxiv.org/abs/2609.31568](https://arxiv.org/abs/2609.31568)

    DeepEdu-v1是一个面向越南教育的AI辅导系统，通过本地部署保障数据主权、围绕国家课程体系组织知识，并克服了消费级GPU上长上下文推理的内存与延迟瓶颈以及区域内容上的幻觉问题。

    

    AI辅导能够显著改善越南等发展中地区学生的学习成果，然而两条显而易见的技术路径都存在不足。以ChatGPT为代表的云端助手会将敏感的学生数据传输到境外服务器——这违反了越南第53号法令等数据主权法律——而且由于预训练语料以西方为中心，这些助手并未围绕国家教科书课程体系进行组织，因此对本地内容的掌握零散且频繁出现幻觉。自行托管开源模型虽可将数据保留在本地，却面临双重障碍：训练后量化（AWQ、GPTQ）虽能控制静态权重的占用，但长辅导上下文带来的动态KV缓存与预填充（prefill）延迟仍会在消费级GPU上引发内存溢出（OOM）故障和响应迟缓，同时模型在区域特定材料上仍会不断产生幻觉。我们提出DeepEdu-v1，一个面向越南教育的AI辅导系统，它基于SCALE（自我改进的上下文感知大语言模型……（摘要原文在此处截断）

    arXiv:2609.31568v1 Announce Type: new  Abstract: AI tutoring could markedly improve learning outcomes for students in developing regions such as Vietnam, yet the two obvious paths both fall short. Cloud assistants such as ChatGPT route sensitive student data to foreign servers---violating data-sovereignty laws such as Vietnam's Decree 53---and, pre-trained on Western-centric corpora, are not organized around the national textbook curriculum, so their knowledge of local content is unsystematic and frequently hallucinated. Self-hosting an open model keeps data on-premise but hits a two-fold wall: post-training quantization (AWQ, GPTQ) tames the static weight footprint, yet the dynamic KV cache and prefill latency of long tutoring contexts still cause out-of-memory failures and slow responses on consumer GPUs, while the model keeps hallucinating on region-specific material. We present DeepEdu-v1, an AI-tutoring system for Vietnamese education built on SCALE (Self-improving Context-Aware L
    
[^320]: 评估实时语音智能体：从组件质量到实际落地成效

    Evaluating Real-Time Voice Agents: From Component Quality to Grounded Outcomes

    [https://arxiv.org/abs/2609.30798](https://arxiv.org/abs/2609.30798)

    该论文指出现有实时语音智能体评估文献分散于语音基础建模、话轮转换心理语言学和智能体评估三个互不引用的领域，并基于38篇一手文献提出三项循证论点，主张超越单一组件指标、建立面向实际部署成效的统一评估框架。

    

    实时语音智能体已从研究原型走向生产部署，然而描述它们的文献分散在三个鲜少互相引用的研究社区中：语音基础建模、话轮转换心理语言学以及智能体评估。架构论文报告延迟，话轮转换论文报告预测准确率，智能体基准测试报告任务成功率，因此没有任何单一指标能够描述一个已部署的智能体是否真正优秀。我们通过三个基于证据的论点来填补这一空白，每个论点都可追溯到一个由38篇一手文献组成的语料库，这些文献被组织成以应用为中心的六类分类体系。第一，架构选择是一种部署约束而非已有定论：一篇2026年的企业教程指出，目前尚无完全可自托管的端到端系统能满足生产约束，而一种分块级联（chunked cascade）架构独立达到了最先进的双工行为，表明双工行为是……

    arXiv:2609.30798v1 Announce Type: new  Abstract: Real-time voice agents have moved from research prototypes to production deployments, yet the literature describing them is fragmented across three communities that rarely cite one another: speech foundation modelling, turn-taking psycholinguistics, and agentic evaluation. Architecture papers report latency, turn-taking papers report prediction accuracy, and agentic benchmarks report task success, so no single number describes whether a deployed agent is actually good. We address that gap with three evidence-based claims, each traceable to a corpus of 38 primary sources organised into an application-centric taxonomy of six categories. First, architecture choice is a deployment constraint rather than a settled verdict: a 2026 enterprise tutorial reports that no fully self-hostable end-to-end system yet meets production constraints, while a chunked cascade independently reaches state-of-the-art duplex behaviour, showing duplex behaviour is
    
[^321]: KernelOPT：面向GPU内核优化的调度感知智能体搜索

    KernelOPT: Dispatch-Aware Agentic Search for GPU Kernel Optimization

    [https://arxiv.org/abs/2609.30059](https://arxiv.org/abs/2609.30059)

    KernelOPT是一个调度感知的多智能体GPU内核优化系统，它在保留厂商库调用的同时仅优化编译器生成的Triton子内核，并通过静态校验、多种子正确性、模型级float64回退与性能门控组成的四道验证级联确保端到端的正确性与加速。

    

    深度学习的推理与训练性能在很大程度上取决于GPU内核的效率。现代编译器（如PyTorch Inductor）能够从高层模型代码自动生成GPU内核，但其性能常常大幅落后于专家手写的实现。近期基于大语言模型（LLM）辅助的内核优化器虽然能缩小独立内核方面的这一差距，但它们将编译后的模型视为黑盒，通常只优化单个独立内核，既不尊重编译器的结构性决策，也不进行模型级的端到端验证。我们提出了KernelOPT，一个将编译后模型视为结构化产物的多智能体系统。该系统保留厂商库调用（cuBLAS、cuDNN），仅针对生成的Triton子内核，并使用五个由性能剖析引导的LLM智能体进行优化。一个由静态校验、多种子正确性验证、模型级float64回退验证以及性能门控组成的四道验证级联，在优化过程中对候选内核进行筛选（原文摘要在此处截断）。

    arXiv:2609.30059v1 Announce Type: cross  Abstract: Deep learning inference and training performance depends critically on GPU kernel efficiency. Modern compilers such as PyTorch Inductor automatically generate GPU kernels from high-level model code, but frequently underperform expert-written implementations by wide margins. Recent LLM-assisted kernel optimizers can close this gap for standalone kernels, yet treat compiled models as black boxes, generally optimizing individual standalone kernels without respecting the compiler's structural decisions or verifying the model end-to-end. We present KernelOPT, a multi-agent system that treats compiled models as structured artifacts. It preserves vendor library calls (cuBLAS, cuDNN) and exclusively targets generated Triton sub-kernels using five profiling-guided LLM agents. A four-gate verification cascade of static validation, multi-seed correctness, model-level float64-fallback verification, and performance gating filters candidates during 
    
[^322]: Harness-Zero：通过“智能体即框架”实现框架蒸馏

    Harness-Zero: Harness Distillation via Agent-as-Harness

    [https://arxiv.org/abs/2609.24974](https://arxiv.org/abs/2609.24974)

    提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。

    

    智能体框架是调节模型与环境交互的外部系统，能够显著提升智能体的性能，但其收益在部署时仍依赖于所使用的框架。由于最佳框架因领域、实例和模型而异，通用智能体要么只能接受次优的共享框架，要么需要在不断增长的专用框架集合中进行路由选择。因此，我们研究智能体框架蒸馏：使用针对领域或实例优化的框架作为训练时的指导，并将其诱导的行为迁移到模型权重中，使其收益在单一固定的目标框架下得以保留。挑战在于，两个框架在动作空间和可用信息上存在差异，因此来自优化框架的指导无法直接作为目标框架的监督信号。我们提出了 Harness-Zero，通过“智能体即框架”实现框架蒸馏。在优化框架的指导下，一个框架……（原文摘要在此处截断）

    arXiv:2609.24974v1 Announce Type: cross  Abstract: Agent harnesses, the external systems that mediate model-environment interaction, can substantially improve agent performance, but their gains remain tied to the harness at deployment. Because the best harness varies across domains, instances, and models, a general-purpose agent must either settle for a suboptimal shared harness or route among an ever-growing set of specialized ones. We therefore study agent harness distillation: using a domain- or instance-optimized harness as training-time guidance and transferring the behaviors it induces into model weights, so that its gains survive under a single fixed target harness. The challenge is that the two harnesses differ in action space and available information, so guidance from the optimized harness cannot serve directly as supervision for the target one. We introduce Harness-Zero, which enables harness distillation through agent-as-harness. Guided by the optimized harness, a harnessin
    
[^323]: Designer-RSI：从用户流量中演化程序性记忆以实现智能体化图形设计

    Designer-RSI: Evolving Procedural Memory from User Traffic for Agentic Graphic Design

    [https://arxiv.org/abs/2609.22086](https://arxiv.org/abs/2609.22086)

    该论文提出Designer-RSI持续适应框架，让冻结的前沿模型通过230多个工具操作专业设计软件，并利用外部程序性记忆从真实用户流量中不断积累和精炼自然语言设计技能，在无权重更新、无人工标注的情况下显著提升智能体图形设计能力。

    

    专业图形设计是一项长时程的智能体任务，其中结构化、可编辑的作品由许多相互依赖的操作逐步产生，然而其结果缺乏可靠的程序化评估标准。我们提出一个持续适应框架：一个冻结的前沿模型通过230多个工具操作专业设计软件，同时一个由自然语言技能构成的外部程序性记忆从经验中积累并精炼可复用的设计流程。该记忆通过为反复出现的未覆盖子任务获取新流程而“变宽”，并通过依据流程自身成功与失败的执行记录修订现有流程而“变深”；同时，一个匹配重放门控机制只允许采纳那些能修复失败且不使已观测成功出现退化的变更。经过五轮迭代，基于1,406个真实用户需求和1,869条自动评分的轨迹，在不更新模型权重、无需人工标注的情况下，技能库从76个源自文档的技能增长到139个，并提升了GenEva……

    arXiv:2609.22086v1 Announce Type: new  Abstract: Professional graphic design is a long-horizon agentic task in which structured, editable artifacts emerge from many interdependent actions, yet outcomes admit no reliable programmatic oracle. We introduce a continual adaptation framework in which a frozen frontier model operates professional design software through more than 230 tools, while an external procedural memory of natural-language skills accumulates and refines reusable design procedures from experience. The memory widens by acquiring procedures for recurring uncovered subtasks and deepens by revising existing procedures against their own successful and failed executions, while a matched replay gate admits only changes that repair failures without regressing observed successes. Five rounds over 1,406 real user briefs and 1,869 automatically graded trajectories, with no weight updates and no human labels, grow the bank from 76 documentation-derived skills to 139 and raise GenEva
    
[^324]: Paint-Anything：面向图像生成与编辑的统一任意颜色控制

    Paint-Anything: Unified Any-Color Control for Image Generation and Editing

    [https://arxiv.org/abs/2609.20816](https://arxiv.org/abs/2609.20816)

    Paint-Anything通过对象级颜色监督学习共享的十六进制提示接口，结合Paint-500K数据集和精确匹配的纯色锚点，实现了图像生成与编辑中任意精确颜色的统一控制。

    

    专业设计需要任意颜色控制：即能够在图像生成和编辑中为对象指定任意24位十六进制值的目标颜色。先前的工作已探索了颜色生成、编辑和上色，但通常依赖于专用的颜色表示或专门的推理流程。大语言模型的进展提供了一个更简单的起点：即使是紧凑的模型也能将十六进制值与颜色语义关联起来。我们提出了Paint-Anything，它通过对象级颜色监督学习一个用于生成和编辑的共享十六进制提示接口。我们开发了一个数据管道，通过对象定位、感知颜色标注和编辑对合成，从真实图像构建了Paint-500K数据集。由于阴影使得真实图像的标签只能提供近似颜色，我们用纯色锚点来补充这种监督，这些锚点的像素与其配对的十六进制值完全匹配。这些锚点仅在高噪声阶段使用。

    arXiv:2609.20816v1 Announce Type: cross  Abstract: Professional design requires any-color control: the ability to specify an object's target color with any 24-bit hex value for image generation and editing. Prior work has explored color generation, editing, and colorization, but often relies on dedicated color representations or specialized inference procedures. Advances in large language models offer a simpler starting point: even compact models can associate hex values with color semantics. We present Paint-Anything, which learns a shared hex-prompt interface for generation and editing through object-level color supervision. We develop a data pipeline that constructs Paint-500K from real images through object grounding, perceptual color labeling, and editing-pair synthesis. Since shadows make real-image labels only approximate colors, we complement this supervision with pure-color anchors whose pixels exactly match their paired hex values. These anchors are used only at high-noise ti
    
[^325]: 缺失的补充：面向编码代理的状态条件化最小充分证据

    The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents

    [https://arxiv.org/abs/2609.20050](https://arxiv.org/abs/2609.20050)

    该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。

    

    一个处理问题进行到一半的编码代理已经读过检索器排名最高的很多内容。相关性是按段落评分的，但充分性属于集合层面：一个排序器可能用某个所需事实的多个变体填满其预算，却使决策仍然缺乏支持。我们提出了状态条件化最小充分证据恢复问题：给定一个捕获的代理状态，恢复一个紧凑的证据组合，以提供其下一个决策仍然缺乏的支持。SERBench在来自45个代码仓库的500个保留状态上对此进行测量，记录代理已经看到的内容，并且只对覆盖当前决策被标注所需的每一个事实的集合给予认可。MSS-Complement将证据获取视为集合构建而非排序。三次语义调用提出一个联合充分的集合，搜索其缺失的内容，并在6,144个token内返回4-8个完整的源单元。一个仅在校准数据上固定的配置，为其中73.0%的状态恢复了完整的证据集合。

    arXiv:2609.20050v1 Announce Type: cross  Abstract: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those
    
[^326]: QVAC Genesis III：一个用于高效语言模型预训练的大规模高质量开放合成STEM语料库

    QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

    [https://arxiv.org/abs/2609.19513](https://arxiv.org/abs/2609.19513)

    提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。

    

    高质量预训练数据是面向边缘AI和端侧部署的教育及STEM专用语言模型的关键瓶颈，在这些场景中token预算受到严格限制。尽管各大机构在私有语料库上训练越来越大的模型，但开放生态系统中缺乏能够以高效方式为小模型提供高单token学习价值的STEM导向合成数据集。为填补这一空白，我们推出了QVAC Genesis III，这是一个拥有1914.3亿token、以STEM为核心的多领域合成语料库，涵盖19个领域，并包含多个难度级别和不同的教育风格。QVAC Genesis III通过一种双重生成策略构建，该策略以一个弱小的边缘规模学生模型作为信号进行针对性教师蒸馏：学生的失败被转化为纠正性解释，而其成功则被扩展为针对所有答案选项的对比性选项级推理。我们进一步引入了LLM作为解析器的机制……（原文摘要在此处截断）

    arXiv:2609.19513v1 Announce Type: new  Abstract: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser ev
    
[^327]: LimiX-2：面向通用结构化数据智能的情境机制网络

    LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence

    [https://arxiv.org/abs/2609.17488](https://arxiv.org/abs/2609.17488)

    LimiX-2通过情境机制网络范式将上下文学习从以目标为中心的预测转向面向机制的联合建模，并利用结构因果模型生成的合成数据进行预训练，在多个基准测试中超越了现有的表格数据专用模型。

    

    我们介绍LimiX-2，这是LimiX家族的新一代模型，基于我们此前建立的扩展定律，通过模型与数据扩展开发而成。LimiX-2采用情境机制网络（CMNs）范式，并使用情境条件掩码建模（CCMM）进行预训练。CMNs将上下文学习的组织原则从以目标为中心的预测转变为面向机制的联合建模。该网络并非以传统表格先验数据拟合网络（PFNs）的p(y|x,D_context)目标为核心，而是围绕学习p(x,y|D_context)进行设计，即对数据生成背后联合结构的情境依赖表示。预训练使用由结构因果模型（SCMs）生成的合成数据集，涵盖多样的图结构、功能机制和观测过程。在TabArena、TALENT和BCCO上的评估表明，LimiX-2优于当前各数据集专用模型。

    arXiv:2609.17488v1 Announce Type: new  Abstract: We introduce LimiX-2, a new model in the LimiX family, developed through model and data scaling guided by our previously established scaling laws. LimiX-2 adopts the Contextual Mechanism Networks (CMNs) paradigm and is pretrained with Context-Conditional Masked Modeling (CCMM). CMNs shifts the organizing principle of in-context learning from target-centric prediction to mechanism-oriented joint modeling. Rather than centering the network on the $p(y \mid x, D_{\mathrm{context}})$ objective of conventional tabular PFNs, it is designed around learning $p(x, y \mid D_{\mathrm{context}})$, a context-dependent representation of the joint structure underlying data generation. Pretraining uses synthetic datasets generated by structural causal models (SCMs) spanning diverse graph structures, functional mechanisms, and observation processes. Evaluations on TabArena, TALENT, and BCCO show that LimiX-2 outperforms current dataset-specific models an
    
[^328]: 稳态持续学习

    Homeostatic Continual Learning

    [https://arxiv.org/abs/2609.13771](https://arxiv.org/abs/2609.13771)

    本文提出“稳态持续学习”方法，通过在智能体输出出现异常时于环境数据中寻找对应异常值，使AI智能体能够在变化环境中持续学习且避免灾难性遗忘，并可用于构建世界模型。

    

    在本文中，我阐述了一个持续学习问题，并提出了一种名为“稳态持续学习”的方法，该方法使AI智能体能够在不断变化的环境中持续学习而不会发生灾难性遗忘。该方法的核心是：当智能体在其输出中遇到异常时，在环境数据中寻找异常值。通过这种方法，智能体逐渐完善其模型和策略，并在越来越多的情境中表现良好。我还建议可以使用该方法构建一个世界模型，其中智能体将世界中的对象分解为特征，将对象抽象为可比较的概念实例，并通过特征将概念映射到意图。我讨论了使该方法实用化所需的工作、该方法与人工智能诸多领域的联系以及其更广泛的意义。

    arXiv:2609.13771v1 Announce Type: new  Abstract: In this paper, I formulate a Continual Learning problem and propose a method named "Homeostatic Continual Learning" that enables an AI agent to learn continuously in a changing environment without catastrophic forgetting. The core of the method is to find outliers in the environment data when the agent experiences an outlier in its output. Through this method, the agent gradually completes its model and policy and performs well in more and more contexts. I also suggest that we may use the method to build a world model where the agent factorizes the objects in the world into features, abstract objects into comparable instances of concepts and map concepts to intents through features. I discuss the works needed to render the method practical, the connections to many fields in Artificial Intelligence and the broader implications of the method.
    
[^329]: 面向具身空地协同目标搜索：基准、数据集与智能体方法

    Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method

    [https://arxiv.org/abs/2609.08402](https://arxiv.org/abs/2609.08402)

    该论文提出了首个空地协同目标搜索基准AGOS-Bench、包含7.7k条轨迹的AGOS-Dataset数据集，以及无需训练、工具增强的智能体方法AGOS-Agent，用以评估和解决视觉-语言模型在UAV-UGV协作搜索与验证目标车辆任务中的难题。

    

    城市环境中的空地目标搜索（AGOS）是一项具有挑战性的具身任务，它需要无人飞行器（UAV）与无人地面车辆（UGV）协同从多视角视觉参考中搜索并验证指定的目标车辆。为研究这一尚未充分探索的问题，我们提出了AGOS-Bench，这是首个专门评估通用视觉-语言模型（VLMs）能否通过UAV-UGV协作整合空中发现与地面验证的基准。我们进一步提供了AGOS-Dataset作为配套资源，该数据集由自动化流程构建的示例轨迹组成，包含7.7k个用于搜索多样化类别与属性目标的情节，涵盖三个难度级别。为解决AGOS任务，我们提出了AGOS-Agent，一种无需训练且工具增强的方法。该智能体方法通过审慎的搜索规划，使视觉-语言模型摆脱复杂而动态的协调负担。

    arXiv:2609.08402v1 Announce Type: cross  Abstract: Air-Ground Object Search (AGOS) in urban environments is a challenging embodied task, which requires an Unmanned Aerial Vehicle (UAV) and an Unmanned Ground Vehicle (UGV) to jointly search for and verify a specified target vehicle from multi-view visual references. To study this underexplored problem, we introduce AGOS-Bench, the first dedicated benchmark for evaluating whether general-purpose Vision-Language Models (VLMs) can integrate aerial discoveries and ground-level verification through UAV-UGV cooperation. We further provide AGOS-Dataset as the companion resource of exemplary trajectories constructed by an automatic pipeline. It consists of 7.7k episodes for searching objects of diverse categories and attributes, spanning three difficulty levels. To address the AGOS task, we propose AGOS-Agent, a training-free and tool-augmented approach. The agentic method relieves VLMs from complex and dynamic coordination via a deliberate sea
    
[^330]: 知道何时不该回答：视觉-语言模型中的弃答与拒绝推理

    Knowing When Not to Answer: Abstention and Refusal Reasoning in Vision--Language Models

    [https://arxiv.org/abs/2609.05540](https://arxiv.org/abs/2609.05540)

    该论文提出了名为PARITY的合成人像对基准数据集，用于审计视觉-语言模型在面对无法从图像回答的医学诊断查询（如通过面部照片诊断自闭症）时是否会正确弃答，并将任何非弃答选择视为有害归因进行评估。

    

    许多医疗状况需要通过详细的、多情境的临床评估来诊断，而不能仅凭视觉外观判断。尽管如此，视觉-语言模型越来越多地被要求以涉及医学或诊断判断的方式来解读图像，当此类推断缺乏依据时会引发安全担忧。自闭症谱系障碍（ASD）的诊断需要行为和发育方面的证据，而非静态的面部照片。我们审查了视觉-语言模型是否会弃答这类无法回答的成对图像查询，以及表情是否会影响非弃答的选择。我们引入了PARITY（重用身份的成对评估）数据集，这是一个合成的、人口统计平衡的、身份受控的中性/表情人像对数据集，并配有中性-中性对照组。所有身份均为合成生成，且不具有ASD状态；由于该查询无法从图像中回答，任何非弃答的选择都被视为有害的归因。在当代视觉-语言模型中，我们……

    arXiv:2609.05540v1 Announce Type: cross  Abstract: Many medical conditions require diagnosis through detailed, multi-context clinical assessment rather than from visual appearance alone. Despite this, vision-language models (VLMs) are increasingly queried to interpret images in ways that touch on medical or diagnostic judgments, raising safety concerns when such inferences are unsupported. ASD diagnosis requires behavioral and developmental evidence, not static facial photographs. We audit whether VLMs abstain from this unanswerable paired-image query, and whether expressions sway non-abstaining choices.   We introduce PARITY (Paired Assessment with Reused Identity), a synthetic, demographically balanced set of identity-controlled neutral/expression portrait pairs with neutral-neutral controls. All identities are synthetic and have no ASD status; because the query is unanswerable from images, any non-abstaining selection is treated as a harmful attribution. Across contemporary VLMs, we
    
[^331]: EvoSkill注入：针对自进化智能体中自主技能生成与演化的红队测试

    EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents

    [https://arxiv.org/abs/2608.30429](https://arxiv.org/abs/2608.30429)

    本文提出EvoSkill注入威胁模型和SARGE红队测试框架，揭示了自进化智能体在自主技能生成与演化过程中可能被注入并复用恶意能力这一新型安全风险，并构建了EvoSkillBench基准数据集加以评估。

    

    基于大语言模型（LLM）的智能体系统日益采用基于技能的架构，以减少重复推理成本并提升稳定、高效的任务执行能力。近期研究提出了自进化智能体，能够从过往经验中自主生成、优化和复用技能，从而实现能力的持续进化。然而，自主技能进化引入了一个新的攻击面，即恶意能力可以被生成、存储并作为合法技能被复用。在本文中，我们将EvoSkill注入定义为一种针对自进化智能体自主技能生成与演化流程的威胁模型。我们进一步提出了SARGE（针对自进化智能体中自主技能生成与演化的红队测试），这是一个通过迭代生成、升级和强化交互来评估该威胁模型的红队测试框架。为支持我们的框架，我们构建了EvoSkillBench，一个包含恶意交互的基准数据集……

    arXiv:2608.30429v1 Announce Type: new  Abstract: LLM-based agent systems increasingly adopt skill-based architectures to reduce repetitive reasoning costs and improve stable, efficient task execution. Recent studies propose self-evolving agents that autonomously generate, refine, and reuse skills from past experiences to enable continuous capability evolution. However, autonomous skill evolution introduces a new attack surface in which malicious capabilities are generated, stored, and reused as legitimate skills. In this paper, we define EvoSkill Injection as a threat model targeting the autonomous skill generation and evolution pipeline of self-evolving agents. We further propose SARGE (Red-teaming Autonomous Skill Generation and Evolution in self-evolving agents), a red-teaming framework for evaluating this threat model through iterative generation, escalation, and reinforcement interactions. To support our framework, we construct EvoSkillBench, a benchmark dataset of malicious inter
    
[^332]: 基准的平衡：面向基准多重性与任务条件评估的语义密度重加权

    Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation

    [https://arxiv.org/abs/2608.30044](https://arxiv.org/abs/2608.30044)

    提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。

    

    语言模型通常通过对基准列表中的分数进行等权平均来进行比较。这类列表通过公开发表而不断增长，缺乏显式的测量设计，因此等权加权使已发表基准的密度成为隐含的能力权重：基准密集的区域会被重复计算。我们提出了基准平衡方法，该方法对基准描述进行嵌入，并为每个基准分配逆密度语义权重。在公开的密度尺度下，相近的基准条目共享聚合影响力。在将异构分数统一到共同潜在尺度后，残差场利用相同的几何结构，使模型排名能够以任务查询为条件。这两个组成部分在实证上发挥着不同的作用。在包含586个模型和14个基准的数据快照上，BoB能够预测哪些模型在留出任务上表现出超出其一般能力的异常强势，画像相关性达到0.462，而等权加权下仅为0.049。

    arXiv:2608.30044v1 Announce Type: new  Abstract: Language models are commonly compared by averaging scores across a benchmark list with equal weight. Such lists grow through publication outside an explicit measurement design, so equal weighting turns the density of published benchmarks into an implicit capability weight: densely benchmarked regions count repeatedly. We introduce Balance of Benchmarks (BoB), which embeds benchmark descriptions and assigns each benchmark an inverse-density semantic weight. Nearby entries share aggregate influence at a disclosed density scale. After equating heterogeneous scores onto a common latent scale, a residual field uses the same geometry to condition model rankings on a task query. The two components serve distinct empirical roles. On a snapshot of 586 models and 14 benchmarks, BoB predicts which models are unusually strong on a held-out task beyond their general ability, reaching a profile correlation of 0.462 compared with 0.049 under equal weig
    
[^333]: NL2AGBench：面向AlphaGeometry的大语言模型自动形式化基准测试

    NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry

    [https://arxiv.org/abs/2608.28481](https://arxiv.org/abs/2608.28481)

    该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。

    

    大语言模型（LLM）的最新进展展示了其在自然语言理解和数学推理方面的强大能力。然而，它们将非形式化数学问题转化为形式化表示的能力仍然缺乏充分探索。这一局限对于神经符号几何系统（如AlphaGeometry）尤为重要，因为其定理证明引擎需要以专门的领域特定语言（DSL）作为输入。尽管AlphaGeometry达到了接近国际数学奥林匹克（IMO）金牌选手的水平，但手动将自然语言问题转换为其形式化语法仍然是一个显著的可用性瓶颈。为了应对这一挑战，我们提出了自然语言到AlphaGeometry基准测试（NL2AGBench），用于评估大语言模型将英文几何问题翻译成与AlphaGeometry兼容的形式化表示的能力。NL2AGBench使用AlphaGeometry内部基于执行的验证来评估翻译质量……

    arXiv:2608.28481v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and mathematical reasoning. However, their ability to translate informal mathematical problems into formal representations remains underexplored. This limitation is particularly important for neuro-symbolic geometry systems such as AlphaGeometry, whose theorem-proving engine requires inputs in a specialized domain-specific language (DSL). Although AlphaGeometry achieves near-IMO gold-medalist performance, manually converting natural-language problems into its formal syntax remains a significant usability bottleneck. To address this challenge, we introduce the Natural Language to AlphaGeometry Benchmark (NL2AGBench), which evaluates LLMs in translating English geometry problems into AlphaGeometry-compatible formal representations. NL2AGBench uses execution-based verification within AlphaGeometry to assess translation qua
    
[^334]: 具有有限次测量泛化保证的量子学习中的测量预算分配

    Measurement-Budget Allocation in Quantum Learning with Finite-Shot Generalization Guarantees

    [https://arxiv.org/abs/2608.24891](https://arxiv.org/abs/2608.24891)

    该论文首次系统研究了量子学习中固定测量预算在训练态数量与每态测量次数之间的最优分配问题，证明了有限测量条件下的免分布泛化界，并推导出与数值最优值相差不到 0.1% 的闭式分配规则，实现 $B^{-1/4}$ 的最坏情况收敛速率。

    

    在近期的量子硬件上，估计一个玻恩（Born）概率需要重复执行量子电路。因此，一个拥有固定测量预算 $B$ 的量子学习实验必须决定使用多少个不同的训练态 $n$，以及在每个训练态上花费多少次测量 $S$。我们针对二元量子分类器研究了这一权衡问题，其测量算符 $M$ 由训练数据学习得到，并用新鲜的测量结果进行评估。我们对有界且满足 $L$-利普希茨条件的损失函数证明了一个免分布的泛化界，其中样本项为 $\sqrt{d/n}$ 阶，测量次数项为 $\sqrt{(\log n)/S}$ 阶；在 $B=nS$ 的约束下，这两项朝相反方向变化。通过最小化该边界的闭式代理表达式，我们得到了预算分配规则 $n^* = 2\sqrt{2dB/\log(4B/\delta)}$，$S^* = B/n^*$，在我们评估的所有预算下，其边界值与精确数值最小值的差距不超过 0.1%，且最坏情况下的速率为 $B^{-1/4}$。PennyLane 模拟（摘要在此处截断）

    arXiv:2608.24891v2 Announce Type: replace  Abstract: On near-term quantum hardware, estimating a Born probability requires repeated circuit executions. A quantum learning experiment with a fixed measurement budget $B$ must therefore decide how many distinct training states $n$ to use and how many shots $S$ to spend on each. We study this trade-off for binary quantum classifiers whose measurement operator $M$ is learned from the training data and evaluated with fresh measurement outcomes. We prove a distribution-free generalization bound for bounded, $L$-Lipschitz losses with a sample term of order $\sqrt{d/n}$ and a shot term of order $\sqrt{(\log n)/S}$; under $B=nS$ the two terms move in opposite directions. Minimizing a closed-form surrogate of the bound gives the allocation rule $n^* = 2\sqrt{2dB/\log(4B/\delta)}$, $S^* = B/n^*$, whose bound value is within 0.1% of the exact numerical minimum on all budgets we evaluate, and a worst-case rate of $B^{-1/4}$. PennyLane simulations wit
    
[^335]: 愚人金：针对开放权重模型安全移除攻击的防御性欺骗

    Fool's Gold: Defensive Deception Against Safety-Removal Attacks on Open-Weight Models

    [https://arxiv.org/abs/2608.17202](https://arxiv.org/abs/2608.17202)

    该论文提出“愚人金”防御方法，通过训练诱饵响应在安全移除攻击后产生伪造信息，有效阻止开放权重模型被恶意利用。

    

    arXiv:2608.17202v1 公告类型：新 摘要：开放权重语言模型的安全对齐可以轻易被移除：abliteration技术能在几分钟内从权重中投射出拒绝中介方向，而据我们所知，没有任何发布时的防御措施能持久阻止这一点。无法阻止的，可以欺骗。我们的防御方法，即诱饵强化（“愚人金”），放弃拒绝条带并毒化其收益：一旦拒绝被移除，大多数对危险操作请求的回答都是自信、流畅的诱饵，其关键要素被伪造。诱饵在攻击的可微模拟中训练，仅在攻击状态下表达；一个拒绝固定和良性约束将清洁状态行为保持为原始状态。我们在五个家族的七个模型上（9B-122B，密集和混合专家）进行了实例化。在通过我们预注册效能门槛的六个模型上，攻击状态下对保留提示的响应中0.51-0.90为诱饵，+0.27-0.84归因于防御；所有s

    arXiv:2608.17202v1 Announce Type: new  Abstract: Safety alignment in open-weight language models is trivially removable: abliteration projects a refusal-mediating direction out of the weights in minutes, and no release-time defense we are aware of prevents it durably. What cannot be prevented can be deceived. Our defense, decoy hardening ("Fool's Gold"), concedes the refusal strip and poisons its payoff: once refusal is stripped, most answers to hazardous operational requests are confident, fluent decoys whose critical elements are falsified. Decoys are trained inside a differentiable simulation of the attack, expressing only in the attacked state; a refusal pin and benign leash hold clean-state behavior to the original. We instantiate it on seven models from five families (9B-122B, dense and mixture-of-experts). On the six models passing our pre-registered efficacy gate, 0.51-0.90 of attacked-state responses to held-out prompts are decoys, +0.27-0.84 attributable to the defense; all s
    
[^336]: CHORUS：面向测试平台激励生成的覆盖率引导智能体强化学习与策略整合

    CHORUS: Coverage-Guided Agentic RL and Policy Consolidation for Testbench Stimulus Generation

    [https://arxiv.org/abs/2608.10090](https://arxiv.org/abs/2608.10090)

    CHORUS是首个将覆盖率引导的智能体强化学习与策略整合相结合的测试平台激励生成框架，通过权重平均与奖励引导蒸馏，其4B模型在CVDP-ECov上达到88.0%的智能体Pass@1，超越DeepSeek-R1（671B）13.5个百分点。

    

    大语言模型编写的测试平台可能能够编译并运行，但整个数据通路却处于空闲状态。我们提出了CHORUS，这是首个将覆盖率引导的智能体强化学习（RL）与策略整合相结合用于测试平台激励生成的框架。我们训练LLM4Cov的最终监督检查点来起草测试平台，并利用仿真器反馈修复其得分最低的草稿，将CVDP-ECov上的智能体Pass@1从69.2%提升至85.5%。将同样的训练应用于三个检查点后，整体通过率相近，但各策略在不同设计上各有成功。对它们的权重进行平均，可将这些优势整合到一个稠密的4B模型中，达到86.7%的智能体Pass@1。当仿真器奖励为每个训练提示选择教师模型时，蒸馏得到的4B模型达到88.0%，比DeepSeek-R1（671B）高出13.5个百分点。

    arXiv:2608.10090v2 Announce Type: replace  Abstract: Testbenches written by large language models can compile and run while leaving entire datapaths idle. We present CHORUS, the first framework to combine coverage-guided agentic reinforcement learning (RL) with policy consolidation for testbench stimulus generation. We train LLM4Cov's final supervised checkpoint to draft testbenches and repair its lowest-scoring draft using simulator feedback, raising CVDP-ECov agentic Pass@1 from 69.2% to 85.5%. Applying the same training to three checkpoints yields similar overall pass rates, yet the policies succeed on different designs. Averaging their weights combines these strengths in a single dense 4B model that reaches 86.7% agentic Pass@1. When simulator rewards select the teacher for each training prompt, the distilled 4B model reaches 88.0%, 13.5 percentage points above DeepSeek-R1 (671B).
    
[^337]: 全模态分解自编码器学习全栈式可穿戴解耦表征

    Omni-modal decomposition autoencoders learn full-stack wearable disentangled representations

    [https://arxiv.org/abs/2608.07385](https://arxiv.org/abs/2608.07385)

    提出全模态变分分解自编码器（OmniDecVAEs），利用单一共享编码器-解码器和多视角自监督分解损失，从任意多模态可穿戴时间序列中学习解耦表征，在人体活动识别数据集上实现最高 3.0 个百分点的 DCI 解耦度提升。

    

    学习解耦表征是通用多模态可穿戴模型的关键，然而现有方法很少能在单一模型中同时实现异构多模态时间序列的分类、解耦、融合与生成。我们提出了全模态变分分解自编码器（OmniDecVAEs），这是一个通过单一共享编码器-解码器从任意数量模态中学习多用途表征的框架。OmniDecVAEs 在 DecVAEs 的基础上进行扩展，通过多视角自监督分解损失和逐分量条件解码器，为每个模态学习时频潜在子空间。在全模态人体活动识别（HAR）HARWE 数据集（最多三十个通道）以及公开的 UCI HAR 和 WESAD 数据集上，OmniDecVAEs 为未见过的受试者学习到了最解耦的表征（DCI 解耦指标最高提升 3.0 个百分点）。在 HARWE 数据集上，它们改善了受试者相关的（摘要在此处截断）

    arXiv:2608.07385v2 Announce Type: replace-cross  Abstract: Learning disentangled representations is key for general-purpose multi-modal wearable models, yet existing approaches rarely combine classification, disentanglement, fusion, and generation of heterogeneous multi-modal time series in a single model. We introduce Omni-modal Variational Decomposition Autoencoders (OmniDecVAEs), a framework that learns multi-purpose representations from arbitrarily many modalities with a single shared encoder-decoder. OmniDecVAEs extend DecVAEs by learning time-frequency latent subspaces per modality through a multi-view self-supervised decomposition loss and a component-wise conditional decoder. On the omni-modal human activity recognition (HAR) HARWE dataset (up to thirty channels) and the public UCI HAR and WESAD datasets, OmniDecVAEs learn the most disentangled representations for unseen subjects (DCI disentanglement up to 3.0 percentage points (pp) higher). On HARWE, they improve subject-depen
    
[^338]: 谨慎上下文引导的语言模型个性化

    Cautious Context Steering for Language Model Personalization

    [https://arxiv.org/abs/2608.05813](https://arxiv.org/abs/2608.05813)

    提出轻量级适配器 CCS，通过谨慎蒸馏学习何时以及以多强强度利用用户上下文来引导冻结语言模型，在个性化偏好对齐上超越了现有无需训练的方法和经过训练的个性化基线。

    

    用户上下文可以在无需针对每个用户进行微调的情况下实现语言模型的个性化，但其影响并非在每个 token 上都同样有帮助。上下文引导在整个生成过程中使用固定的引导强度，且每一步都需要两次语言模型前向传播。我们提出了谨慎上下文引导，这是一种轻量级适配器，能够学习何时以及以多强的强度利用用户上下文来引导冻结的语言模型。我们通过谨慎蒸馏来训练该适配器，使用由带上下文和不带上下文的预测构建的 oracle 目标。它学会在上下文有助于预测用户偏好 token 的地方匹配引导后的分布，而在其他地方匹配无上下文的基础分布。CCS 适配器在各个基准测试的偏好对齐方面均优于无需训练的基于上下文的方法和经过训练的个性化基线。我们还引入了 GSM8K-P，用于同时评估任务准确性和任务特定的格式偏好。CCS 实现了较高的格式遵从性……

    arXiv:2608.05813v2 Announce Type: replace  Abstract: User context can personalize a language model without per-user fine-tuning, but its influence is not equally helpful at every token. Context Steering (CoS) uses a fixed steering strength throughout generation and requires two LM forward passes per step. We propose Cautious Context Steering (CCS), a lightweight adapter that learns when and how strongly to steer a frozen LM using user context. We train the adapter through cautious distillation, using oracle targets built from predictions with and without context. It learns to match the steered distribution where context helps predict the user's preferred token and the context-free base distribution elsewhere. The CCS adapter outperforms both training-free context-based methods and trained personalization baselines in preference alignment across benchmarks. We also introduce GSM8K-P to evaluate task accuracy and task-specific format preferences together. CCS achieves high format adheren
    
[^339]: 内部放松，跨模态平衡：视觉语言混合专家模型中组合偏移下的负载均衡

    Relax Within, Balance Across: Load Balancing under Composition Shifts in Vision-Language Mixture-of-Experts

    [https://arxiv.org/abs/2608.00574](https://arxiv.org/abs/2608.00574)

    该论文揭示了视觉语言混合专家模型中的跨模态负载抵消现象——图像与文本误差在训练比例附近相互抵消从而掩盖推理时的负载失衡，并提出ReBA方法通过模态内放松、模态间平衡的策略来应对组合偏移下的负载均衡问题。

    

    像Qwen3-VL这样的视觉语言混合专家模型需要均衡的负载以实现快速并行服务。标准训练方法将图像和文本词元合并到同一个辅助损失中，并假设训练时的负载均衡在推理时依然保持。然而我们发现了一个令人惊讶的不匹配现象：在固定路由器上，只要分辨率、动态平铺或图像数量改变了图像与文本词元的比例，负载均衡就会发生急剧偏移。按模态拆分负载揭示了跨模态负载抵消现象：相反方向的图像和文本误差在训练比例附近相互抵消，掩盖了强烈的负载偏斜，直到混合比例发生变化才会暴露。为了量化这种偏移，我们推导出一个精确的二次定律：在固定的模态分布下，仅需一次参考运行即可预测完整的响应曲线，且图像-文本差距决定敏感度。同时，同一图像内的图像块以相似方式路由，而图像块数量各不相同，使得大图像在简单池化中占据主导地位。我们提出ReBA（内部放松、跨模态平衡）方法来分离模态（摘要在此处截断）

    arXiv:2608.00574v2 Announce Type: replace-cross  Abstract: Vision-language mixture-of-experts (MoE) models such as Qwen3-VL need balanced loads for fast parallel serving. Standard training pools image and text tokens into one auxiliary loss, assuming training balance holds at inference. Yet we find a surprising mismatch: on a fixed router, balance shifts sharply whenever resolution, dynamic tiling, or image count changes the image-to-text token ratio. Splitting load by modality reveals cross-modal load cancellation: opposing image and text errors cancel near the training ratio, hiding strong skew until the mix changes. To quantify this shift, we derive an exact quadratic law: under fixed modality profiles, one reference run predicts the full response curve, and the image-text gap governs sensitivity. Meanwhile, patches within an image route similarly while patch counts vary, letting large images dominate flat pooling. We propose ReBA (Relax Within, Balance Across) to separate modality 
    
[^340]: 将推理轨迹提炼为软件工程任务的咨询式提示

    Distilling Reasoning Traces into Advisory Prompts for Software Engineering Tasks

    [https://arxiv.org/abs/2608.00437](https://arxiv.org/abs/2608.00437)

    受人类通过反思错误并内化规则来学习编程的启发，本文提出将LLM思考模式中避免错误的推理轨迹提炼为可内化的咨询式提示，使软件工程任务无需持续承担推理成本即可获得性能提升。

    

    语言模型被广泛用于生成和处理代码（例如，预测行为、输出或可能的输入），本文探讨能否在不总是承担推理成本的情况下获得更好的性能。人类编程学习者通过以下方式学会避免错误：（a）首先识别错误，（b）反思导致这些错误的认知疏漏（本质上是“深入思考”错误），（c）从这些反思中推断出一般性规则或经验教训，（d）将这些教训内化为规则。在导师参与的辅导课中，这是一种非常常见的苏格拉底式师生互动。这类可内化的规则的例子可能包括“编码之前，重述需求以澄清它们”这样的要点。受这一过程的启发，本文描述了一种方法：首先识别（低资源）LLM 中“思考模式”避免错误的例子；然后，这些错误……（原文摘要截断于此）

    arXiv:2608.00437v2 Announce Type: replace-cross  Abstract: Language models are widely used for generating and otherwise processing code (e.g., predicting behavior, outputs, or likely inputs), and this paper asks whether better performance can be achieved without always incurring the cost of reasoning. Human students of programming learn to avoid mistakes by (a) first identifying them, (b) reflecting upon cognitive lapses that led to these mistakes (essentially, "thinking through" the errors), (c) inferring general rules or lessons from these reflections, and (d) internalizing these lessons into rules. In tutorial sessions with an instructor, this is a very common Socratic interaction with students. Examples of such internalizable rules might include the nugget "Before coding, restate the requirements to clarify them". Inspired by this process, this paper describes an approach where we first identify examples in which "thinking mode" in a (low-resource) LLM avoids errors; then, these er
    
[^341]: Tokengeist：智能体对话中的多轮归因追踪

    Tokengeist: Multi-Turn Attribution Tracing in Agentic Conversations

    [https://arxiv.org/abs/2607.22610](https://arxiv.org/abs/2607.22610)

    该论文提出了多轮上下文归因（MTCA）任务以及Tokengeist框架，通过将有向无环图上的递归遍历作为归因方式来恢复完整依赖路径，解决了现有单次归因方法无法捕捉多轮对话分层依赖结构的问题。

    

    arXiv:2607.22610v2 公告类型：替换 摘要：当一个模型在经过多轮对话和工具使用后给出回答时，监督者需要了解该回答依赖于哪些较早的对话轮次和中间步骤。归因是回答这一问题的自然基础，但现有的上下文归因方法以单次处理方式处理完整对话记录：它们只能恢复表面层面的依赖关系，而遗漏了多轮追踪中的分层结构——在这种结构中，助手的某一轮回复本身就是一个派生产物，它对较早的上下文进行了压缩或转换。我们提出了多轮上下文归因（MTCA）：给定模型回复中的一个目标片段，该任务要求跨轮次向后追踪归因，不仅要识别哪些先前的轮次直接相关，还要识别这些轮次本身如何依赖于更早的上下文。我们提出了Tokengeist，一个与归因方法无关且可扩展的框架，它通过将归因转化为对有向无环图的递归遍历来恢复完整的依赖路径。

    arXiv:2607.22610v2 Announce Type: replace  Abstract: When a model answers after several turns of dialogue and tool use, an overseer needs to know which earlier turns and intermediate steps it rests on. Attribution is the natural primitive for answering this, but existing context attribution methods process the full transcript in a single pass: they recover surface-level dependencies while missing the layered structure of multi-turn traces, in which an assistant turn is itself a derived artifact that condenses or transforms earlier context. We introduce multi-turn context attribution (MTCA): given a target span in a model response, the task of tracing attribution backward across turns to identify not only which prior turns were directly relevant, but also how those turns themselves depended on earlier context. We propose Tokengeist, an attribution-method-agnostic and scalable framework that recovers full dependency paths by casting attribution as a recursive traversal of a directed acyc
    
[^342]: 面向延迟与模型大小优化的大语言模型多目标结构化剪枝

    Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization

    [https://arxiv.org/abs/2607.22583](https://arxiv.org/abs/2607.22583)

    该论文提出了一种硬件感知的多目标结构化剪枝框架，通过两阶段方法联合优化大语言模型的层数、注意力头和MLP维度，以同时最小化延迟和模型大小，从而实现边缘设备上的高效部署。

    

    arXiv:2607.22583v4 公告类型：替换 摘要：大语言模型（LLM）因其强大的推理和问答能力而得到了广泛应用。然而，由于严格的延迟、内存和能耗限制，将其部署到嵌入式和边缘计算环境中仍然充满挑战。其庞大的参数量和计算需求阻碍了在资源受限平台上的高效执行。尽管模型剪枝已成为在保持性能的同时缩减模型规模的一种可行方案，但联合优化层数、注意力头数以及多层感知机（MLP）维度仍然极为复杂。穷举探索这一组合设计空间的计算开销巨大，且往往导致局部最优或不稳定的配置。为解决这些局限，我们提出了一种硬件感知的多目标结构化剪枝框架。所提出的两阶段方法明确地以延迟和模型大小为目标，以实现高效部署（原文在此处截断）。

    arXiv:2607.22583v4 Announce Type: replace  Abstract: Large Language Models (LLMs) have achieved widespread adoption because of their strong reasoning and query-response capabilities. However, deploying them in embedded and edge computing environments remains challenging because of strict latency, memory, and energy constraints. Their large parameter counts and computational demands hinder efficient execution on resource-constrained platforms. Although model pruning has emerged as a viable solution for reducing scale while preserving performance, jointly optimizing layers, attention heads, and Multi-Layer Perceptron (MLP) dimensions remains highly complex. Exhaustively exploring this combined design space is computationally expensive and often leads to local optima or unstable configurations. To address these limitations, we propose a hardware-aware, multi-objective structured pruning framework. The proposed two-stage method explicitly targets latency and model size for efficient deploy
    
[^343]: 从状态方程到引力对偶 II：大能级层级与伪真空

    Gravitational Duals from Equations of State II: Large Hierarchies and False Vacua

    [https://arxiv.org/abs/2606.30117](https://arxiv.org/abs/2606.30117)

    该论文将基于物理信息神经网络（PINNs）的全息逆问题方法扩展至具有大能级层级和伪真空的强耦合量子场论情形，通过一系列方法学改进克服了近简并态与势能未探测区域等挑战，实现了从边界热力学数据重构体标量势。

    

    我们研究了在以大层级和伪真空存在为特征的区域内，强耦合量子场论的全息对偶重构问题。在规范/引力对偶框架下，这些特征转化为非平凡的热力学行为和新奇的重整化群流，包括在非相邻不动点之间发生的“跳跃流”。在以往基于物理信息神经网络（PINNs）工作的基础上，我们将“从边界热力学数据重构体标量势”这一全息逆问题扩展到了这一新的区域。这一设定带来了诸多概念和数值上的挑战，例如近简并态、能量尺度的大层级，以及势能中未被输入数据直接探测到的区域。我们开发了一系列方法学上的改进来克服这些障碍，从而改进了已有的基于PINNs的方法。

    arXiv:2606.30117v2 Announce Type: replace-cross  Abstract: We investigate the reconstruction of holographic duals for strongly coupled quantum field theories in regimes characterized by large hierarchies and the presence of false vacua. Within the gauge/gravity duality, these features translate into non-trivial thermodynamic behaviour and exotic renormalization group flows, including skipping flows between non-adjacent fixed points. Building on previous work based on Physics-Informed Neural Networks (PINNs), we extend the holographic inverse problem of reconstructing the bulk scalar potential from boundary thermodynamic data into this new regime. This setting presents a variety of conceptual and numerical challenges, such as near-degenerate states, large hierarchies of energy scales, and regions of the potential that are not directly probed by the input data. We develop a set of methodological advances that overcome these obstacles, thereby improving the established PINNs-based methodo
    
[^344]: 为什么信任你的智能体？TRiSM引导的医疗保健代理工作流带来的实证安全收益

    Why Trust Your Agent? Empirical Security Gains from TRiSM-Guided Agentic Workflows in Healthcare

    [https://arxiv.org/abs/2606.28666](https://arxiv.org/abs/2606.28666)

    本文首次将TRiSM安全框架应用于医疗报告生成的智能体工作流，通过大规模实证评估证明安全导向的重新设计可显著降低RAG投毒和数据字段注入等攻击的成功率。

    

    基于智能体的AI通过向大语言模型（LLM）开放应用工具和资源，实现了任务自动化。然而，为了提升覆盖范围和准确性，智能体通常被授予超出普通用户的访问权限，这带来了重大安全风险和潜在的监管违规。本文将AI信任、风险与安全管理（TRiSM）框架应用于医疗报告生成应用，展示了如何将不安全的基础工作流转变为具有安全意识的智能体工作流。两种设计均通过五个LLM（Claude Haiku 4.5、GPT-4.1-nano、GPT-4.1-mini、GPT-5.4-mini和Gemini 2.5 Flash）进行了评估，涵盖800次报告生成和500次攻击试验，包括RAG投毒、数据字段注入和客户端网络注入。重新设计使RAG投毒的攻击成功率从31%降至10%，数据字段注入的攻击成功率从42%降至25%（McNemar p < 0.00…

    arXiv:2606.28666v2 Announce Type: replace-cross  Abstract: Agent-based AI has enabled the automation of tasks by exposing application tools and resources to large language models (LLMs). However, to improve scope and accuracy, agents are often given access rights that exceed those of ordinary users, introducing significant security risks and potential regulatory breaches. This paper applies the AI Trust, Risk, and Security Management (TRiSM) framework to a medical report-generation application to demonstrate how an insecure baseline workflow can be transformed into a security-conscious agentic workflow. Both designs were evaluated with five LLMs (Claude Haiku 4.5, GPT-4.1-nano, GPT-4.1-mini, GPT-5.4-mini and Gemini 2.5 Flash) across 800 report generations and 500 attack trials covering RAG poisoning, data-field injection and client-side network injection. The redesign reduced attack success from 31% to 10% for RAG poisoning and from 42% to 25% for data-field injection (McNemar p < 0.00
    
[^345]: SPARK：面向忠实检索增强生成的自适应参数化知识参与

    SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation

    [https://arxiv.org/abs/2606.27786](https://arxiv.org/abs/2606.27786)

    SPARK框架将参数化知识控制形式化为潜在参与策略学习问题，通过轻量级参与策略动态调节生成过程中参数化知识的贡献，从而解决RAG中检索上下文与模型内部知识的冲突，提升生成忠实性，同时避免直接编辑模型参数可能带来的能力损害。

    

    检索增强生成（RAG）通过将外部知识融入生成过程来增强大型语言模型（LLM）的忠实性。然而，检索到的上下文与参数化知识之间的冲突可能导致LLM依赖不一致的内部知识，阻碍其忠实利用检索证据的能力。现有方法通过编辑参数化知识来缓解这一问题，以解决此类冲突。然而，定位和修改编码在模型参数中的知识十分困难，并且可能无意中损害骨干LLM的能力。在本文中，我们提出了SPARK，这是一个新颖的框架，它将参数化知识控制形式化为潜在参与策略学习问题，使LLM能够在生成过程中自适应地调节参数化知识的贡献。具体而言，SPARK引入了一个轻量级的参与策略，动态调制……

    arXiv:2606.27786v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) enhances the faithfulness of large language models (LLMs) by incorporating external knowledge into the generation process. However, conflicts between retrieved context and parametric knowledge can cause LLMs to rely on inconsistent internal knowledge, hindering their ability to faithfully leverage retrieved evidence. Existing approaches mitigate this issue by editing parametric knowledge to resolve such conflicts. However, locating and modifying knowledge encoded in model parameters is challenging and may inadvertently compromise the backbone LLM's capabilities. In this paper, we introduce SPARK, a novel framework that formulates parametric knowledge control as a latent participation policy learning problem, enabling LLMs to adaptively regulate the contribution of parametric knowledge during generation. Specifically, SPARK introduces a lightweight participation policy that dynamically modula
    
[^346]: 由马尔可夫逻辑网络定义的随机着色有向图

    Random coloured digraphs defined by a Markov logic network

    [https://arxiv.org/abs/2606.23715](https://arxiv.org/abs/2606.23715)

    本文系统研究了马尔可夫逻辑网络所定义的概率分布在不同域规模下一阶语句成立概率的行为，并证明在可表达一元性质与二元关系的一阶语言中，多种不同的情形均可由该模型实现。

    

    马尔可夫逻辑网络（MLN）M 是一种概率关系模型，用于统计关系人工智能领域，对任意有限域 D，在以 D 为域的可能世界集合 W_D 上定义概率分布 P_D^M。MLN 由一阶公式（我们称之为软约束）组成，每个公式带有非负实数权重。该概率分布在不同的域大小下的行为（包括权重所产生的影响）目前仍知之甚少。我们通过系统地研究序列 (P_n^M(ψ) : n = 1, 2, 3, …) 来处理这一问题，其中 ψ 是一个一阶语句，P_n^M(ψ) 是 ψ 在 W_D 中成立的概率，且 D = [n] := {1, …, n}。我们考虑一种能够表达一元性质 P(x) 与二元关系 R(x, y) 的一阶语言，并证明对于一个 MLN M，以下所有情形都可以被实现（摘要在此处截断）。

    arXiv:2606.23715v2 Announce Type: replace-cross  Abstract: A Markov Logic Network (MLN) $M$ is a probabilistic relational model used in Statistical Relational Artificial Intelligence for defining a probability distribution $P_D^M$ on the set $W_D$ of possible worlds with domain $D$ for an arbitrary finite domain $D$. An MLN consists of first-order formulas, which we call soft constraints, with associated weights which are nonnegative real numbers. The behaviour of the probability distribution, including the effect of the weights, on different domain sizes is poorly understood. We approach the problem in a systematic way by studying sequences $(P_n^M(\psi) : n = 1, 2, 3, \ldots)$ where $\psi$ is a first-order sentence and $P_n^M(\psi)$ is the probability that $\psi$ holds in $W_D$ where $D = [n] := \{1, \ldots, n\}$. We consider a first-order language that can express a property $P(x)$ and relation $R(x, y)$ and show that all of the following cases can be realized for an MLN $M$ such th
    
[^347]: 深度何时对上下文学习重要？深度Transformer中的自适应推理

    When Does Depth Matter For In-Context Learning? Adaptive Inference in Deep Transformers

    [https://arxiv.org/abs/2606.16694](https://arxiv.org/abs/2606.16694)

    本文提出了广义线性注意力Transformer中分布式推理的一般理论，证明多层Transformer可利用“函数向量”内部表示在各层以越来越精细的尺度推断潜在上下文变量，从而阐明了深度堆叠在何种条件下比单层结构具有计算优势。

    

    Transformer通过多个连续的注意力和前馈块执行计算，使它们能够学习大量耦合变量之间的复杂相关性。那么，在多层上堆叠连续的注意力-前馈计算，何时能比单个Transformer块提供计算优势？我们通过研究广义线性注意力Transformer中的上下文学习来回答这个问题。我们首先引入了此类Transformer中分布式推理的一般理论，该理论受限于通信与深度的约束。我们证明，这样的系统可以利用内部表示（“函数向量”）在其各层上以越来越精细的尺度推断潜在上下文变量。对于一个上下文线性回归任务，该理论预测：虽然不含前馈块的单层Transformer对于上下文变量服从高斯先验的情形是最优的，但多层Transformer……

    arXiv:2606.16694v3 Announce Type: replace-cross  Abstract: Transformers perform computations through many successive attention and feedforward blocks, allowing them to learn complex correlations between a large collection of coupled variables. When does stacking successive attention-feedforward computations over many layers provide a computational advantage over a single transformer block? We address this question by examining in-context learning in generalized linear attention transformers. We first introduce a general theory of distributed inference in such transformers, subject to constraints on communication and depth. We show that such systems can exploit internal representations (`function vectors') to infer a latent context variable at increasingly finer scales over its layers. For an in-context linear regression task, the theory predicts that while one-layer transformers without feedforward blocks are optimal for Gaussian priors over the context variable, multi-layer transforme
    
[^348]: 有理稀疏自编码器

    Rational Sparse Autoencoder

    [https://arxiv.org/abs/2606.14990](https://arxiv.org/abs/2606.14990)

    提出有理稀疏自编码器（RSAE），用可训练的有理函数取代固定的编码器激活函数，在匹配期望稀疏度下从理论上保证不劣于 ReLU，并在给定假设下可证明优于 JumpReLU 和 TopK。

    

    稀疏自编码器（SAE）是机制可解释性研究的标准工具，但当前的 SAE 家族受到固定编码器非线性激活函数（如 ReLU、JumpReLU 和 TopK）的约束。这种做法将特定的稀疏机制硬编码进模型，并可能扭曲重构与稀疏性之间的权衡。我们提出了有理稀疏自编码器（RSAE），它用一个可训练的有理函数取代固定的编码器激活函数。该设计的动机源于我们在匹配期望稀疏度下为 RSAE 证明的重构优势：保证不劣于 ReLU，并且在给定假设下，可证明优于 JumpReLU 和 TopK。该模型通过两阶段流水线实现：首先是一个初始化过程，复制预训练的基线 SAE 权重，并对尺度参数和有理系数进行校准；随后在稀疏正则化的（微调步骤中进行优化，原文摘要在此处截断）。

    arXiv:2606.14990v3 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) are standard tools for mechanistic interpretability, but current SAE families are constrained by fixed encoder nonlinearities such as ReLU, JumpReLU, and TopK. This hard-codes a particular sparsity mechanism into the model and can distort the reconstruction-versus-sparsity trade-off. We introduce the Rational Sparse Autoencoder (RSAE), which replaces the fixed encoder activation with a trainable rational function. The design is motivated by a reconstruction advantage that we prove for RSAE at matched expected sparsity, with guaranteed non-inferiority to ReLU and, under stated assumptions, provable improvements over JumpReLU and TopK. The model is then implemented through a two-stage pipeline: an initialization procedure that copies the pre-trained baseline SAE weights and calibrates the scale parameters along with the rational coefficients; followed by a fine-tuning step under a sparsity-regularized r
    
[^349]: 个体行为的隐式神经表示

    Implicit Neural Representations of Individual Behavior

    [https://arxiv.org/abs/2606.12200](https://arxiv.org/abs/2606.12200)

    提出Behavioral INR，一种自监督生成模型，将视觉领域的隐式神经表示适配到行为领域，把策略表示为由片段级潜在变量通过FiLM层调制的状态-动作函数，从而在无标签的混合行为数据中实现无监督的策略表示学习与策略身份推断。

    

    我们研究从无标签的多策略行为数据中进行策略表示学习。每个轨迹片段由一个固定策略生成，但策略标签不可用。这种场景出现在机器人自由玩耍、演示、游戏、赛车以及其他异构行为混合且无标注的数据集中。我们提出了Behavioral INR，一种将视觉领域中的隐式神经表示（INR）适配到行为领域的自监督生成模型。Behavioral INR不是将坐标映射到RGB值，而是将策略表示为一个状态-动作函数，即把状态映射到后续动作。一个片段级的潜在变量通过FiLM层对该函数进行调制，从而产生策略之上的生成先验，并允许在无监督条件下推断策略身份。由于INR将每个数据点视为来自某个底层函数的样本，同一模型能够自然地适应不同的片段长度和不同的采样粒度……

    arXiv:2606.12200v2 Announce Type: replace-cross  Abstract: We study policy representation learning from unlabeled multi-policy behavioral data. Each episode is generated by a fixed policy, but policy labels are unavailable. This setting appears in robotics play, demonstrations, games, racing, and other datasets where heterogeneous behaviors are mixed without annotations. We introduce Behavioral INR, a self-supervised generative model that adapts implicit neural representations (INRs) from vision to behavior. Instead of mapping coordinates to RGB values, Behavioral INR represents a policy as a state-action function mapping states to subsequent actions. An episode-level latent modulates this function through FiLM layers, yielding a generative prior over policies and allowing policy identity to be inferred without supervision. Because INRs treat each datapoint as samples from an underlying function, the same model naturally accommodates variable episode lengths and different sampling gran
    
[^350]: 商业世界模型

    Business World Model

    [https://arxiv.org/abs/2606.10044](https://arxiv.org/abs/2606.10044)

    本文提出了商业世界模型（BWM）的概念与架构，通过以商业语义为中心的公式化方法将状态、动力学和可行行动与商业实体相联系，使世界模型范式得以拓展应用于语义性、组织性和市场驱动的商业环境。

    

    世界模型已成为人工智能领域的一种强大范式，使智能体能够表征其环境、预测未来状态，并在行动之前评估可能的行动。然而，现有的世界模型主要是为视觉或物理领域（如计算机视觉、机器人技术、游戏和自动驾驶）开发的，这些领域受相对稳定的动力学规律支配。这些公式化方法并不能直接应用于商业实践，因为商业实践中的相关环境是语义性、组织性和市场驱动的，其结果取决于客户行为、定价、竞争和监管等对情境敏感的因素。本文介绍了商业世界模型的概念和架构，这是一种专门针对商业和组织环境的世界模型。我们提出了一种以商业语义为中心的公式化方法，其中状态、动力学和可行行动与商业实体相联系。

    arXiv:2606.10044v3 Announce Type: replace  Abstract: World model has emerged as a powerful paradigm in artificial intelligence, enabling agents to represent their environments, predict future states, and evaluate possible actions before acting. However, existing world models have largely been developed for visual or physical domains such as computer vision, robotics, gaming, and autonomous driving, which are governed by relatively stable dynamics. These formulations are not directly applicable to business practice, where the relevant environment is semantic, organizational, and market-driven, and outcomes depend on context-sensitive factors such as customer behavior, pricing, competition, and regulation. This paper introduces the concept and architecture of a Business World Model (BWM), a world model specialized for business and organizational environments. We propose a business-semantics-centric formulation in which states, dynamics, and feasible actions are linked to business entitie
    
[^351]: SPACE：面向多模态大语言模型的无源代理锚点概念擦除方法

    SPACE: Source-free Proxy Anchor Concept Erasure for MLLMs

    [https://arxiv.org/abs/2606.09868](https://arxiv.org/abs/2606.09868)

    SPACE 提出了首个面向多模态大语言模型的无源机器遗忘框架，通过文本引导的代理锚点选择和双约束语义隔离两个阶段，在无需访问目标概念原始数据的情况下实现敏感概念的擦除。

    

    随着多模态大语言模型（MLLMs）面临日益增长的隐私风险和监管限制，机器遗忘已成为在保持模型性能的同时删除敏感数据的关键解决方案。然而，现有的机器遗忘方法通常依赖于目标概念的视觉数据，而由于严格的数据保留政策，这些数据往往无法获取，因此催生了对无需访问目标数据的无源遗忘方法的需求。在本工作中，我们提出了无源代理锚点概念擦除，这是首个专门面向多模态大语言模型的无源遗忘框架。SPACE 包含两个阶段：（1）文本引导的代理锚点选择（TPAS），从共享特征空间中检索语义对齐的代理锚点；（2）双约束语义隔离（DCSI），通过优化这些锚点来间接擦除目标概念。DCSI 将更新约束在保留……（原文摘要在此处截断）

    arXiv:2606.09868v2 Announce Type: replace-cross  Abstract: As Multimodal Large Language Models (MLLMs) face growing privacy risks and regulatory constraints, machine unlearning (MU) has emerged as a crucial solution for removing sensitive data while preserving model performance. However, existing MU methods typically rely on visual data of the target concepts, which is often unavailable due to strict data retention policies, thus creating a demand for source-free unlearning approaches that operate without access to the target data. In this work, we propose Source-free Proxy Anchor Concept Erasure (SPACE), the first source-free unlearning framework specialized for MLLMs. SPACE consists of two stages: (1) Text-Guided Proxy Anchor Selection (TPAS), which retrieves semantically aligned proxy anchors from the shared feature space. (2) Dual-Constraint Semantic Isolation (DCSI), which optimizes these anchors to indirectly erase target concepts. DCSI confines updates to the null space of retai
    
[^352]: RNA设计的Montparnasse算法

    The Montparnasse Algorithm for RNA Design

    [https://arxiv.org/abs/2606.07562](https://arxiv.org/abs/2606.07562)

    Montparnasse是一种基于广义嵌套rollout策略自适应的蒙特卡洛逆RNA折叠搜索算法，通过结构先验、字典序多准则评估、错误碱基对局部修复以及基于错误维持多样性的并行搜索等创新，在Eterna100基准上中位数解决了100个谜题中的96个，超越了DesiR的90个。

    

    逆RNA折叠问题是指找到一个能够折叠成预定二级结构的核苷酸序列，其在合成生物学、医学和纳米技术领域具有重要应用价值。我们提出了Montparnasse，一个基于广义嵌套rollout策略自适应的蒙特卡洛搜索框架，并通过以下多项技术加以增强：从能量模型导出的结构先验、第1层的慢速长时自适应、字典序多准则评估、对预测折叠中错误碱基对的局部修复机制，以及一组通过共享beam耦合的并行工作进程——其多样性是基于候选者所犯的错误而非候选者本身来维持的。在Eterna100 V1基准测试中，两个求解器均通过相同的ViennaRNA 1.8.5库在Turner 1999能量模型下进行折叠预测，在100个并行工作进程且每个谜题预算30分钟的条件下，Montparnasse在100个谜题中解决数量的中位数为96个，而DesiR的解决数量中位数为90个。

    arXiv:2606.07562v2 Announce Type: replace-cross  Abstract: Inverse RNA Folding consists of finding a nucleotide sequence that folds into a predefined secondary structure. It is useful for synthetic biology, medicine, and nanotechnology. We propose Montparnasse, a Monte Carlo search framework based on Generalized Nested Rollout Policy Adaptation, augmented with a structural prior derived from the energy model, slow and long adaptation at level 1, a lexicographic multicriteria evaluation, a local repair of the pairs the predicted folding gets wrong, and a fleet of parallel workers coupled through a shared beam whose diversity is maintained on the faults the candidates make rather than on the candidates themselves. On the Eterna100 V1 benchmark, with both solvers folding through the same ViennaRNA 1.8.5 library under the Turner 1999 model, with 100 parallel workers and a budget of 30 minutes per puzzle, Montparnasse solves a median of 96 of the 100 puzzles against a median of 90 for DesiR
    
[^353]: DAST：一种面向O-RAN跨接口异常检测的VLM-LLM框架

    DAST: A VLM-LLM Framework for Cross-Interface Anomaly Detection in O-RAN

    [https://arxiv.org/abs/2606.06261](https://arxiv.org/abs/2606.06261)

    该论文提出DAST框架，通过VLM-LLM-VLM三阶段级联的零样本多智能体流水线，解决O-RAN开放架构下拒绝服务与性能降级攻击难以检测的问题，实现无需标注基线的跨接口异常检测。

    

    O-RAN使基站协议栈得以解耦部署，其中的可编程功能通过标准化开放接口进行通信。这种支持多厂商组件组合的开放性，同时也扩大了构成计算连续体的逻辑解耦各层级的攻击面。在这些威胁中，拒绝服务攻击和性能降级攻击占O-RAN已编目威胁的大多数，尤其难以检测。传统的时间序列异常检测（TSAD）方法在这种新环境下失效：标注基线稀缺、威胁演进速度快于检测器的再训练速度、高维多变量遥测数据超出单体推理模型的处理能力。为应对这些挑战，我们提出DAST，一个用于O-RAN跨接口异常检测的零样本多智能体框架，它串联起VLM → LLM → VLM的三阶段流水线。DAST将多变量遥测数据转换为……（摘要内容不完整，后续信息缺失）

    arXiv:2606.06261v2 Announce Type: replace-cross  Abstract: O-RAN enables a disaggregated baseband stack with programmable functions that communicate over standardized open interfaces. The same openness that enables multi-vendor composition also expands the attack surface across logically decoupled tiers that make up the compute continuum. Among these threats, Denial-of-Service and performance-degradation attacks, which account for the majority of catalogued O-RAN threats, are particularly difficult to detect. Traditional Time-Series Anomaly Detection (TSAD) methods fail in this new regime where labelled baselines are scarce, threats evolve faster than detectors can be retrained, and the high-dimensional multivariate telemetry overwhelms monolithic inference models. To address these challenges, we present DAST, a zero-shot multi-agent framework for cross-interface anomaly detection in O-RAN that chains a three-stage VLM $\rightarrow$ LLM $\rightarrow$ VLM pipeline. DAST converts multiva
    
[^354]: 学习哪些证据寻求决策才是重要的：面向语言智能体的校准反事实重放方法

    Learning Which Evidence-Seeking Decisions Matter: Calibrated Counterfactual Replay for Language Agents

    [https://arxiv.org/abs/2606.05263](https://arxiv.org/abs/2606.05263)

    提出了PCCC-RL方法，通过前缀条件下的反事实重放、预先信用预测与基于选择概率的校准纠正机制，精准识别语言智能体中哪些证据寻求决策真正决定了最终答案的质量，从而解决稀疏奖励下的信用分配问题。

    

    语言智能体通过搜索、阅读和核查声明来回答问题。仅凭最终答案的奖励很难说明是哪些决策为其赢得了正确结果，因为智能体可能在冗余搜索之后、或在缺少一半证据的情况下依然得到正确的答案字符串。我们将这个问题视为一个实验。在保存的决策点上，PCCC-RL（前缀条件反事实信用分配方法）会恢复工具状态，同时运行智能体所选择的动作和一个可行的替代动作，并让同一个冻结策略完成这两个分支。最终得分的差异便抽样出该决策的真实价值。由于重放所有内容代价过高，该方法预先预测每个动作的信用值，随机抽取一个子集进行验证，并利用记录的选择概率来纠正预测误差，将预算花费在进一步的检查仍能改变训练信号的地方。替代动作仅从前缀中抽取，这保持了……

    arXiv:2606.05263v2 Announce Type: replace-cross  Abstract: Language agents answer questions by searching, reading, and checking claims. A reward on the final answer says little about which of those decisions earned it, since an agent can reach the right string after redundant searches or with half the evidence missing. We treat the question as an experiment. At a saved decision point, PCCC-RL, for prefix-conditioned counterfactual credit, restores the tool state, runs the action the agent chose alongside a feasible alternative, and lets the same frozen policy finish both branches. The difference in final score then samples what the decision was worth. Replaying everything is too expensive, so the method predicts each action's credit up front, checks a random subset, and uses the recorded selection probabilities to correct the errors of that prediction, spending the budget where a further check can still move the training signal. Alternatives are drawn from the prefix alone, which keeps
    
[^355]: 支持性令牌揭示：加速扩散语言模型解码

    Supportive Token Revealing for Fast Diffusion Language Model Decoding

    [https://arxiv.org/abs/2606.04236](https://arxiv.org/abs/2606.04236)

    提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。

    

    离散扩散语言模型可以通过并行更新多个被掩码的位置来高效生成文本，但这种并行性引入了质量与延迟之间的权衡。激进的解码可能过早地确定相互依赖的令牌，而保守的解码则需要许多去噪步骤。现有方法通过基于置信度或依赖性准则判断哪些令牌可以安全揭示来缓解这一矛盾。然而，避免不安全的提交并不一定能使剩余的掩码序列易于解码，因为不确定的令牌可能依赖于其他被掩码的令牌，这成为去噪步骤的瓶颈。我们提出了AXON，这是一个无需训练的模块，可以叠加在现有的扩散语言模型并行解码策略之上。AXON并不替换基础解码器，而是监控剩余的不确定掩码令牌，仅在其当前状态表明需要额外上下文时才进行干预。

    arXiv:2606.04236v2 Announce Type: replace-cross  Abstract: Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is need
    
[^356]: VESTA：基于统计工具智能体的可视化探索

    VESTA: Visual Exploration with Statistical Tool Agents

    [https://arxiv.org/abs/2606.00384](https://arxiv.org/abs/2606.00384)

    VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。

    

    将定量模型拟合到数据是科学工作流程中的核心步骤，但这一步骤仍然是自动化程度最低的环节之一。近年来，基于智能体的系统利用语言模型和视觉语言模型（VLM）来提出统计模型，但这些系统在具有挑战性的建模任务上表现不佳。为了解决这些局限性，我们提出了VESTA（Visual Exploration with Statistical Tool Agents，基于统计工具智能体的可视化探索），这是一个为VLM配备动态增长工具包的框架，通过数据变换、假设驱动的可视化以及统计检验来引导模型的改进。与以往仅依赖迭代式批评反馈的系统不同，VESTA通过选择或创建可复用的诊断工具，在模型改进之前和改进过程中对数据进行探索。我们使用三种工具包配置对VESTA进行评估：无工具、静态的专家编写工具，以及动态的模型自建工具。为支持这一评估，我们引入了DAWN（用于自动化工作流程和数值……的数据集）——摘要在此处截断。

    arXiv:2606.00384v3 Announce Type: replace  Abstract: Fitting quantitative models to data is a central step in scientific workflows, yet it remains one of the least automated. Recent agent-based systems leverage language and vision-language models (VLMs) to propose statistical models, but these systems struggle on challenging modeling tasks. To address these limitations, we introduce VESTA (Visual Exploration with Statistical Tool Agents), a framework that equips VLMs with a dynamically growing toolkit to guide model refinement through data transformations, hypothesis-driven visualizations, and statistical tests. Unlike prior systems that rely on iterative critique alone, VESTA explores data before and during refinement by selecting or creating reusable diagnostic tools. We evaluate VESTA using three toolkit configurations: no tools, static expert-written tools, and dynamic model-written tools. To support this evaluation, we introduce DAWN (Dataset for Automated Workflows and Numerical 
    
[^357]: 从“弱”信号到强模型：基于LoRA合并的偏好增量聚合

    From "Weak" Signals to Strong Models: Preference Delta Aggregation with LoRA Merging

    [https://arxiv.org/abs/2606.00357](https://arxiv.org/abs/2606.00357)

    提出了偏好增量聚合（PDA）框架，将多个弱-更弱模型对产生的“弱”偏好信号分别学习为LoRA适配器，并通过几何对齐合并（GAM）进行聚合，从而有效提升强模型（如Qwen3 8B）的性能。

    

    训练强大的大语言模型（LLM）需要高质量的监督信号，而这种信号通常稀缺。近期研究表明，来自弱-更弱模型对（如Qwen3 4B优于1.7B）的成对偏好数据，尽管单个回答的质量有限，但可以通过相对质量差异提供有效的监督信号，我们将其称为“弱”信号。这引出了一个关键研究问题：能否将多个“弱”信号进行建设性的聚合，以改进强模型（如Qwen3 8B）？为此，我们提出了偏好增量聚合（PDA），这是首个从每个弱-更弱模型对中提取偏好增量、将其实例化为通过偏好优化学习到的LoRA适配器、并通过LoRA合并聚合所得增量的框架。为了改进独立训练的适配器之间的组合效果，我们引入了几何对齐合并（GAM），该方法对齐各适配器的左右奇异基……

    arXiv:2606.00357v3 Announce Type: replace  Abstract: Training strong large language models (LLMs) requires high-quality supervision, which is often scarce. Recent work shows that paired preference data from weak-weaker model pairs (e.g., Qwen3 4B over 1.7B), despite the limited quality of individual responses, can provide an effective supervision signal through relative quality deltas, which we term a "weak" signal. This motivates a key research question: can multiple "weak" signals be constructively aggregated for improving strong models (e.g., Qwen3 8B)? To this end, we propose Preference Delta Aggregation (PDA), the first framework that derives a preference delta from each weak-weaker model pair, instantiates it as a LoRA adapter learned through preference optimization, and aggregates the resulting deltas via LoRA merging. To improve the composition of independently trained adapters, we introduce Geometric Alignment Merging (GAM), which aligns their left and right singular bases thr
    
[^358]: PhoneWorld：扩展手机使用代理环境

    PhoneWorld: Scaling Phone-Use Agent Environments

    [https://arxiv.org/abs/2605.29486](https://arxiv.org/abs/2605.29486)

    PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。

    

    arXiv:2605.29486v2 公告类型：替换交叉  摘要：手机使用代理的一个核心瓶颈是，难以大规模构建覆盖真实移动行为的可控、可复现环境。现有的移动代理基准在评估方面取得了重要进展，但本身并未提供一种可扩展的方式来构建许多新的手机使用环境。我们提出了PhoneWorld，一个可复用的流水线，将真实的GUI轨迹和截图转换为可控的手机使用环境、可执行任务、自动验证器和训练回放。PhoneWorld不是一次手动构建一个移动基准，而是利用真实轨迹来恢复哪些屏幕重要、屏幕如何连接、哪些交互必须改变环境状态，以及哪些用户目标允许自动验证。从这些信号中，它构建了由只读应用内容和可变状态支持的可运行模拟Android应用，然后推导出可执行任务、基于规则的验证器和训练数据。

    arXiv:2605.29486v2 Announce Type: replace-cross  Abstract: A central bottleneck for phone-use agents is that controllable, reproducible environments covering real mobile behavior are hard to build at scale. Existing mobile-agent benchmarks have made important progress on evaluation, but they do not by themselves provide a scalable way to construct many new phone-use environments. We present PhoneWorld, a reusable pipeline that converts real GUI trajectories and screenshots into controllable phone-use environments, executable tasks, automatic verifiers, and training rollouts. Rather than hand-building one mobile benchmark at a time, PhoneWorld uses real trajectories to recover which screens matter, how screens connect, which interactions must change environment state, and which user goals admit automatic verification. From these signals, it builds runnable mock Android apps backed by read-only app content and mutable state, then derives executable tasks, rule-based verifiers, and traini
    
[^359]: 面向大语言模型长度外推的周期性与谐波周期性旋转位置编码（RoPE）

    Periodic and Harmonic Periodic RoPE for Length Extrapolation in LLMs

    [https://arxiv.org/abs/2605.27980](https://arxiv.org/abs/2605.27980)

    该论文提出周期性 RoPE（P-RoPE）与谐波周期性 RoPE（HP-RoPE）两种位置编码机制，通过与滑窗注意力结合并仅复用 W 个旋转角度来缓解大语言模型长上下文外推中的位置耗尽问题，其中 HP-RoPE 借助在窗口内完成整数圈的谐波频率消除了窗口边界的回绕误差。

    

    arXiv:2605.27980v2 公告类型：replace-cross。摘要：处理长上下文的能力对大语言模型（LLM）执行长程任务至关重要。旋转位置编码（RoPE）为每个 token 分配一个新的旋转角度，因此模型最终会遇到训练中从未见过的角度，即“位置耗尽”问题。我们提出了周期性 RoPE（P-RoPE），一种与滑窗注意力配合使用的位置编码机制。它重复使用 RoPE 角度表中前 W 个（滑窗大小）角度，因此模型只需要 W 个不同的角度。P-RoPE 是有效的，但会在窗口边界处引入回绕误差。我们进一步提出了谐波周期性 RoPE（HP-RoPE），通过选择在 W 个 token 内恰好完成整数圈的频率来消除该误差。我们比较了架构、数据和训练计划完全相同、仅在窗口层位置编码上有所区别的三种模型：标准 RoPE、P-RoPE 和 HP-RoPE。结果表明，P-RoPE 和 HP-RoPE 能够保持困惑度（原文在此处截断）……

    arXiv:2605.27980v2 Announce Type: replace-cross  Abstract: The ability to process long contexts is crucial for large language models (LLMs) to perform long-horizon tasks. Rotary position embedding (RoPE) gives every token a new rotation angle, so a model eventually meets angles it never saw in training, i.e., position exhaustion. We propose Periodic RoPE (P-RoPE), a positional encoding mechanism that operates in conjunction with sliding window attention. It repeats the first sliding-window-size \(W\) angles of the RoPE table, so the model needs only \(W\) distinct angles. P-RoPE works but introduces a wrap error at the window boundary. We further propose Harmonic Periodic RoPE (HP-RoPE), which removes the error by choosing frequencies that complete whole turns in \(W\) tokens. We compare models that share the same architecture, data, and schedule and differ only in the window-layer position encoding: standard RoPE, P-RoPE, and HP-RoPE. The results show that P-RoPE and HP-RoPE keep perp
    
[^360]: 通道级向量量化：具有可学习排序的原生一维视觉标记化

    Channel-wise Vector Quantization: Native 1D Visual Tokenization with Learnable Ordering

    [https://arxiv.org/abs/2605.26089](https://arxiv.org/abs/2605.26089)

    本文提出通道级向量量化（CVQ），通过量化特征图通道而非图像块，实现具有可学习由粗到细排序的原生一维视觉标记化，并据此构建了通过“下一通道预测”依次生成图像、先勾勒全局结构再细化细粒度细节的通道级自回归（CAR）模型。

    

    我们提出了通道级向量量化，这是一种新颖的图像标记化范式，它用通道级标记取代了基于图像块的标记。与传统的为每个图像块特征向量分配一个离散标记的向量量化方法不同，CVQ对特征图的每个通道进行量化。这将图像表示为具有可学习的由粗到细排序的一维通道序列，而不是二维图像块网格。基于CVQ，我们引入了一种采用“下一通道预测”的新型视觉自回归框架。我们的通道级自回归模型不再按光栅顺序逐块渲染图像，而是依次预测图像通道，逐步生成更丰富的视觉细节。具体而言，它首先勾勒全局结构，然后再细化细粒度属性，类似于人类艺术家的工作流程。通过实验，我们证明：（1）CVQ表现出强大的码本可扩展性，在高达……的码本规模下仍保持95%以上的利用率（摘要在此处截断，后续内容未提供）。

    arXiv:2605.26089v3 Announce Type: replace-cross  Abstract: We present Channel-wise Vector Quantization (CVQ), a novel image tokenization paradigm that replaces patch-wise tokens with channel-wise tokens. Unlike conventional vector quantization, which assigns a discrete token to each patch feature vector, CVQ quantizes each channel of the feature map. This represents an image as a 1D channel sequence with learnable coarse-to-fine ordering, rather than a 2D patch grid. Based on CVQ, we introduce a new visual autoregressive framework with "next-channel prediction". Instead of rendering images patch by patch in raster order, our Channel-wise Autoregressive (CAR) model predicts image channels sequentially, producing progressively enriched visual details. Specifically, it first sketches global structure and then refines fine-grained attributes, akin to a human artist's workflow. Empirically, we show that: (1) CVQ exhibits strong codebook scalability, maintaining 95%+ utilization with up to $
    
[^361]: 洞察生成器：面向大语言模型智能体的系统性语料库级轨迹诊断

    Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

    [https://arxiv.org/abs/2605.21347](https://arxiv.org/abs/2605.21347)

    该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。

    

    arXiv:2605.21347v4 公告类型：replace-cross 摘要：诊断大语言模型（LLM）智能体的失败在很大程度上仍然是手工完成的。从业者通常只检查一小部分执行轨迹，形成临时性的假设，然后不断迭代。这一过程会遗漏那些只有在轨迹群体层面才会显现的模式，也无法扩展到单条轨迹就包含数万 token 的生产级语料库。我们形式化了语料库级轨迹诊断问题：给定一个执行轨迹语料库，目标是在轨迹群体上刻画系统性的行为模式，生成有据可依的自然语言洞察，并且每条洞察都关联相应的支持性证据。我们提出了洞察生成器（Insights Generator, IG），这是一个多智能体系统，它通过在轨迹语料库上提出并检验假设来回答诊断问题，最终产出有证据支撑的洞察报告。我们从定性和客观两个维度评估了 IG，包括基于评分量表的报告评估，以及通过实施洞察所带来的下游性能提升……

    arXiv:2605.21347v4 Announce Type: replace-cross  Abstract: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by impl
    
[^362]: AVSD：通过平衡共识信号与教师特定特权信号的自适应视图自蒸馏

    AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

    [https://arxiv.org/abs/2605.20643](https://arxiv.org/abs/2605.20643)

    本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。

    

    自蒸馏使语言模型能够从自身的轨迹中进行在线学习，其方法是将同一个模型同时用作学生和教师，其中教师以学生无法获取的特权信息为条件。这类信息可以来自不同的类型或视图，例如解题方案、示范、反馈或最终答案。这种设置无需依赖单独的外部模型即可提供密集的 token 级反馈，但会产生一个根本性的不对称问题：教师可能依赖于学生在推理时无法访问的视图特定信息。此外，最佳类型的特权信息往往取决于具体任务，因此难以选择单一的教师视图。在本工作中，我们通过提出 AVSD（自适应视图自蒸馏）来联合解决这两个挑战，这是一种使用多种特权信息视图的新型自蒸馏方法，它重构了 token 级监督……（摘要内容被截断）

    arXiv:2605.20643v2 Announce Type: replace-cross  Abstract: Self-distillation enables language models to learn on-policy from their own trajectories by using the same model as both student and teacher, with the teacher being conditioned on privileged information unavailable to the student. Such information can come in different types or views, such as solutions, demonstrations, feedback, or final answers. This setup provides dense token-level feedback without relying on a separate external model, but creates a fundamental asymmetry: the teacher may rely on view-specific information that the student cannot access at inference time. Moreover, the best type of privileged information is often task-dependent, making it difficult to choose a single teacher view. In this work, we address both these challenges jointly by introducing AVSD (Adaptive-View Self-Distillation), a novel method of self-distillation with multiple privileged-information views, which reconstructs token-level supervision b
    
[^363]: 重新思考大型视觉语言模型中胸部X光推理的视觉归因方法

    Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models

    [https://arxiv.org/abs/2605.20158](https://arxiv.org/abs/2605.20158)

    该论文提出了一个基于反事实编辑的因果评估框架，用于验证视觉归因方法是否真正反映了大型视觉语言模型在胸部X光推理中决策所依赖的视觉证据，并发现现有归因方法在该框架下并不可靠。

    

    大型视觉语言模型（LVLMs）在医疗应用中展现出巨大潜力，但其无法将回答忠实锚定于视觉证据，这引发了对临床可信度的严重担忧。虽然视觉归因方法被广泛用于解释LVLM的预测，但这些解释是否真正反映了模型决策背后的视觉证据在很大程度上尚未得到验证，因为模型内部推理的真值标注通常是不可获得的。我们通过开发一个因果评估框架来解决胸部X光（CXR）推理中的这一问题，该框架仅保留那些经反事实编辑验证其专家标注区域对模型预测具有因果责任的CXR-VQA样本。利用该框架，我们在11种归因方法、六个开源LVLM以及两种输出模式（直接回答和逐步推理）上进行评估，发现现有的归因方法往往……（原文摘要在此处被截断）

    arXiv:2605.20158v2 Announce Type: replace-cross  Abstract: Large Vision Language Models (LVLMs) show promise in medical applications, but their inability to faithfully ground responses in visual evidence raises serious concerns about clinical trustworthiness. While visual attribution methods are widely used to explain LVLM predictions, whether these explanations actually reflect the visual evidence underlying the model's decision is largely unverified, since ground-truth annotations for internal model reasoning are typically unavailable. We address this question for chest X-ray (CXR) reasoning by developing a causal evaluation framework that retains only CXR-VQA samples for which the expert-annotated region is verified, via counterfactual editing, to be causally responsible for the model's prediction. Using this framework across 11 attribution methods, six open-source LVLMs, and two output modes (direct answer and step-by-step reasoning), we find that existing attribution methods often
    
[^364]: MINTEval：评估长时程智能体系统中多目标干扰下的记忆能力

    MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems

    [https://arxiv.org/abs/2605.18565](https://arxiv.org/abs/2605.18565)

    提出MINTEval基准，通过包含频繁更新信息的长且高度关联的上下文、多样化领域任务和问题类型，系统评估记忆增强智能体在多目标干扰和长时程场景下的记忆能力。

    

    现实世界的智能体在漫长且不断演进的时间跨度中运行，其中信息会被反复更新，并可能在多个记忆之间产生相互干扰，这要求智能体能够准确回忆并对多条信息进行聚合推理。然而，现有的基准测试专注于静态、独立的回忆任务，无法捕捉不断演化的记忆之间的这些动态交互作用。在本文中，我们研究了当前记忆增强型智能体在跨多种领域和问题类型的真实、高干扰、长时程环境中的表现。我们提出了MINTEval（长时程干扰下记忆评估），该基准具有以下特点：(1) 长且高度关联的上下文，其中信息频繁更新，从而产生大量干扰；(2) 多样化的领域（状态追踪、多轮对话、维基百科修订和GitHub提交），支持领域泛化能力的评估；(3) 多样化的问题类型，用于评估鲁棒性……

    arXiv:2605.18565v3 Announce Type: replace-cross  Abstract: Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce MINTEval (Long-Horizon Memory under INTerference Evaluation), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness 
    
[^365]: 作者对AI审稿的回应：一项混合方法试点研究

    Authors' Response to AI-Based Reviews: A Mixed-Methods Pilot Study

    [https://arxiv.org/abs/2605.16623](https://arxiv.org/abs/2605.16623)

    该混合方法试点研究通过对三个计算机科学会议作者的问卷调查发现，约九成作者认为AI辅助评审有用并在修改中采纳其建议，AI还能发现人工评审遗漏的问题，但近半数作者对AI反馈的信任度较低，多数报告其意见存在错误或误导。

    

    背景：大语言模型日益被用于辅助学术同行评审，但关于作者如何评价这些反馈并将其用于修改论文的证据仍然有限。方法：我们在三个计算机科学学术活动中研究了作者在人工评审之外收到辅助AI评审的体验：LifeCLEF 2025、LifeCLEF 2026和CVWW 2026。评审后问卷共收集到92份回复，涉及71篇论文。我们使用描述性统计总结封闭式问题的回答，并通过主题分析考察自由文本回答。结果：大多数回复认为AI评审有用（82/92，89%），82/92（89%）报告在修改中至少使用了其中部分建议。作者报告AI发现了人工评审员遗漏的问题（76/92，83%）。然而，44/92（48%）对AI反馈的信任度低于人工反馈，58/92（63%）报告存在不正确或误导性意见。作者重视……（原文摘要在此处截断）

    arXiv:2605.16623v2 Announce Type: replace-cross  Abstract: Background: Large language models are increasingly used to assist scholarly peer review, but there is limited evidence about how authors judge their feedback and use it to revise their papers.   Methods: We studied authors' experiences of receiving an auxiliary AI review alongside human reviews in three computer-science cohorts: LifeCLEF 2025, LifeCLEF 2026, and CVWW 2026. Post-review questionnaires yielded 92 responses concerning 71 papers. We summarized closed-ended answers using descriptive statistics and examined free-text responses through thematic analysis.   Results: Most responses rated the AI review useful (82/92 (89%), and 82/92 (89%) reported using at least some of its suggestions in revision. Authors reported that AI identified issues missed by human reviewers ((76/92 (83%). However, 44/92 (48%) trusted AI feedback less than human feedback, and 58/92 (63%) reported incorrect or misleading comments. Authors valued su
    
[^366]: 面向有限数据高效标记化的频谱视觉Transformer

    Spectral Vision Transformer for Efficient Tokenization with Limited Data

    [https://arxiv.org/abs/2605.12026](https://arxiv.org/abs/2605.12026)

    提出频谱视觉Transformer，将图像标记化为固定基底的频谱系数而非空间图像块，使序列长度与分辨率解耦、可自由选择，从而在数据稀缺场景（尤其是医学影像）下实现高效且性能优越的标记化。

    

    视觉Transformer将图像标记化为图像块，这使得标记数量与分辨率耦合，且需要从数据中学习嵌入。我们提出了一种频谱视觉Transformer架构，用于在有限数据条件下实现高效标记化，重点面向医学影像领域。不同于使用空间图像块，我们将图像标记化为固定基底的系数。这种方法将每个标记定义为一个频谱模态，序列长度可以自由选择而非由分辨率隐式决定，从而降低了注意力机制的计算成本。我们提出的频谱ViT在数据稀缺的场景下，于公开医学影像基准、自然图像、物体形状、仿真数据以及受保护的临床数据上均展现出优于或相当的性能。我们在以下网址开源了代码：https://github.com/agr78/SpectralViT。

    arXiv:2605.12026v2 Announce Type: replace-cross  Abstract: Vision transformers tokenize an image into patches, which couples the token count to resolution and requires the embedding to be learned from data. We propose a spectral vision transformer architecture for efficient tokenization in the limited-data regime, with an emphasis on medical imaging. Rather than using spatial patches, we tokenize into coefficients of a fixed basis. This approach defines each token as one spectral mode, and the sequence length is chosen rather than implied by resolution, reducing the cost of attention. Our proposed spectral ViT shows superior or comparable performance where data is scarce in public medical imaging benchmarks, natural images, object shapes, simulations, and protected clinical data. We release our code at: https://github.com/agr78/SpectralViT.
    
[^367]: AI智能体技能的行为完整性验证

    Behavioral Integrity Verification for AI Agent Skills

    [https://arxiv.org/abs/2605.11770](https://arxiv.org/abs/2605.11770)

    该论文提出行为完整性验证（BIV）框架，通过将确定性代码分析与LLM辅助能力提取相结合，对智能体技能的声明能力与实际能力进行类型化集合比较，并在49,943个OpenClaw技能上发现80%的技能存在描述与实现之间的偏差。

    

    智能体技能通过赋予大语言模型（LLM）智能体特权第三方能力（如文件系统访问、凭证、网络调用和shell执行）来扩展其功能。现有的安全工作能够捕获恶意提示和危险的运行时操作，但技能工件本身却未经验证。我们将此问题形式化为行为完整性验证问题：在一个连接代码、指令和元数据的共享分类体系上，对声明能力与实际能力进行类型化集合比较。BIV框架通过将确定性代码分析与LLM辅助的能力提取相结合来实现这一比较。所产生的结构化证据支持三种下游分析：偏差分类、根因分类和恶意技能检测。在来自OpenClaw注册表的49,943个技能上，偏差分类揭示了一种普遍存在的描述与实现之间的差距：80.0%的技能在分类体系中至少存在一处偏差，其中四个（摘要在此处被截断）

    arXiv:2605.11770v2 Announce Type: replace-cross  Abstract: Agent skills extend LLM agents with privileged third-party capabilities such as filesystem access, credentials, network calls, and shell execution. Existing safety work catches malicious prompts and risky runtime actions, but the skill artifact itself goes unverified. We formalize this as the behavioral integrity verification (BIV) problem: a typed set comparison between declared and actual capabilities over a shared taxonomy that bridges code, instructions, and metadata. The BIV framework instantiates this comparison by pairing deterministic code analysis with LLM-assisted capability extraction. The resulting structured evidence supports three downstream analyses: deviation taxonomy, root-cause classification, and malicious-skill detection. On 49,943 skills from the OpenClaw registry, the deviation taxonomy reveals a pervasive description-implementation gap: 80.0% of skills have at least one deviation in the taxonomy, with fou
    
[^368]: 校准偏差的内生性：评分报告中的不可能性与出路

    The Endogeneity of Miscalibration: Impossibility and Escape in Scored Reporting

    [https://arxiv.org/abs/2605.07671](https://arxiv.org/abs/2605.07671)

    该论文证明决策耦合评分报告中的校准偏差是内生的（福利最优的批准规则必然非仿射），但通过构造一个基于不依赖类型分布的“保留报告”、带单个折点的 Lipschitz 批准规则，可在每个严格恰当评分下完美筛选类型并实现一阶最优，其关键约束是规则斜率而非光滑性。

    

    代理人的概率报告会被“支付”两次：一次由严格恰当评分规则支付，另一次由针对其报告所触发决策的批准规则支付。在这种经典的决策耦合设定中，已知非仿射的批准规则会破坏真实报告。我们证明这种冲突是内生的：在可行的条件下，福利最大化的批准规则永远不会是仿射的。然而，这种扭曲是可预测的，并且可以通过机制设计来规避。存在一个“保留报告”，在该报告处，伪装成边际类型的成本恰好等于批准奖励。只要在保留报告处或以上才批准，就能在每一个严格恰当评分下完美地筛选类型，且该保留报告不依赖于类型分布。一个具有单个折点的 Lipschitz 规则可以精确实现一阶最优；而在严格可行性条件下，任何连续可微的规则都无法做到这一点。真正起约束作用的是陡峭度，而非光滑性。一阶最优能够在某个斜率预算内实现，当且仅当该预算至少达到……（原文摘要到此截断）

    arXiv:2605.07671v2 Announce Type: replace-cross  Abstract: An agent's probability report is paid for twice: by a strictly proper scoring rule, and by an approval rule for the decision it triggers. In this classical decision-coupled setting, non-affine approval is known to defeat truthful reporting. We show the conflict is endogenous: when feasible, the welfare-maximizing approval rule is never affine. The distortion, however, is predictable and can be designed around. There is a reserve report at which pretending to be the marginal type costs exactly the approval prize. Approving at or above the reserve screens types perfectly under every strictly proper score, and the reserve does not depend on the type distribution. A Lipschitz rule with a single kink attains first-best exactly; under strict feasibility no continuously differentiable rule does. The binding constraint is steepness, not smoothness. First-best is attainable within a slope budget if and only if the budget is at least the
    
[^369]: 面向多模态推理的结构化角色感知策略优化

    Structured Role-Aware Policy Optimization for Multimodal Reasoning

    [https://arxiv.org/abs/2605.07274](https://arxiv.org/abs/2605.07274)

    该论文提出结构化角色感知策略优化（SRPO），通过将多模态结构化响应分解为感知token与推理token，把序列级的GRPO优势细化为角色感知的token级优势，实现更精确的信用分配，从而确保正确答案有任务相关的视觉证据支撑。

    

    基于可验证奖励的强化学习（RLVR），尤其是结合组相对策略优化（GRPO），在提升大型视觉语言模型（LVLM）的推理能力方面已展现出强大潜力。然而，在多模态推理中，最终答案奖励通常是在序列级别进行分配的，无法区分不同token的功能角色，因此难以判断一个正确答案是否得到了与任务相关的视觉证据的支持。在本文中，我们从角色感知的token级信用分配视角重新审视多模态RLVR，将结构化响应分解为用于提取视觉证据的感知token，以及用于从这些证据推导出答案的推理token。基于这一视角，我们提出了结构化角色感知策略优化（SRPO），该方法在不改变……（原文在此处截断）的前提下，将序列级的GRPO优势细化为角色感知的token级优势。

    arXiv:2605.07274v2 Announce Type: replace  Abstract: Reinforcement learning from verifiable rewards (RLVR), especially with Group Relative Policy Optimization (GRPO), has shown strong potential for improving the reasoning capabilities of large vision-language models (LVLMs). However, in multimodal reasoning, final-answer rewards are typically assigned at the sequence level and do not distinguish the functional roles of different tokens, making it difficult to determine whether a correct answer is supported by task-relevant visual evidence. In this paper, we revisit multimodal RLVR from the perspective of role-aware token-level credit assignment, where structured responses are decomposed into perception tokens for extracting visual evidence and reasoning tokens for deriving answers from that evidence. Based on this perspective, we propose Structured Role-aware Policy Optimization (SRPO), which refines the sequence-level GRPO advantage into role-aware token-level advantages without chang
    
[^370]: 后推理：零成本提升非思考型模型的性能

    Post-Reasoning: Improving the Performance of Non-Thinking Models at No Cost

    [https://arxiv.org/abs/2605.06165](https://arxiv.org/abs/2605.06165)

    “后推理”让指令微调模型在生成最终答案后再进行答案论证，从而在不增加任何推理延迟或token成本的情况下有效提升模型性能。

    

    随着大语言模型（LLM）的广泛应用不断加速，来自中间推理链的token消耗日益成为推理延迟和运营成本的重要来源。近期研究表明，许多现实世界的任务几乎不需要显式推理，额外的推理有时甚至会降低性能。在这项工作中，我们提出了“后推理”，这是一种简单而有效的方法，通过让指令微调模型在生成最终回答之后对答案进行论证，从而提升模型性能。从设计上看，该方法使最终答案能够在不产生额外延迟或token成本的情况下直接获取，同时仍可通过简单的指令增强来提升性能。我们在117个模型-基准设置上对“后推理”进行了评估，涵盖13个开源和专有模型、4个模型家族，以及9个多样化的推理和知识密集型基准测试，包括AMC、HMMT、GSM8K、GPQA、MMLU-Pro和BI。

    arXiv:2605.06165v2 Announce Type: replace  Abstract: As the widespread adoption of Large Language Models (LLMs) accelerates, token consumption from intermediate reasoning traces increasingly contributes to inference latency and operational cost. Recent studies suggest that many real-world tasks require little to no explicit reasoning, with additional reasoning sometimes even degrading performance. In this work, we propose Post-Reasoning, a simple yet effective approach that improves instruction-tuned models by conditioning them to justify their answers after generating the final response. By design, it enables the final answer to be obtained without additional latency or token cost, while still improving performance through simple instruction augmentation. We evaluate Post-Reasoning across 117 model--benchmark settings spanning 13 open and proprietary models, 4 model families, and 9 diverse reasoning and knowledge-intensive benchmarks, including AMC, HMMT, GSM8K, GPQA, MMLU-Pro, and BI
    
[^371]: 使用脉冲神经网络检测AI生成的视频

    Detecting AI-Generated Videos with Spiking Neural Networks

    [https://arxiv.org/abs/2605.05895](https://arxiv.org/abs/2605.05895)

    该论文发现AI生成视频在像素级帧间残差和语义特征轨迹上均存在时间平滑性特征，并利用脉冲神经网络对这一时间动态证据进行建模，以实现跨生成器的AI视频检测。

    

    现代AI生成的视频在单帧层面已达到照片级的真实感，使得帧间动态成为检测的主要剩余依据。现有检测器通常以三种方式处理这种时间证据：将完整的帧序列输入通用的时间骨干网络，将单一的主要时间线索简化为固定的视频级描述符，或者通过检测度量将时间特征与真实视频的统计数据进行比较。这些策略在跨生成器评估中性能急剧下降，因为伪影类型和时间尺度在不同生成器之间各不相同。在配备字幕的基准数据集GenVidBench上，我们识别出两个先前检测器未能共同利用的特征：AI生成的视频在像素级别表现出更平滑的帧间时间残差，并且在语义特征空间中表现出更紧凑的轨迹，这表明在两个层面都存在时间平滑性差距。我们进一步观察到，当原始视频被输入到脉冲……（摘要原文在此处截断）

    arXiv:2605.05895v3 Announce Type: replace-cross  Abstract: Modern AI-generated videos are photorealistic at the single-frame level, leaving inter-frame dynamics as the main remaining axis for detection. Existing detectors typically handle this temporal evidence in three ways: feeding the full frame sequence to a generic temporal backbone, reducing one dominant temporal cue to fixed video-level descriptors, or comparing temporal features to real-video statistics through a detection metric. These strategies degrade sharply under cross-generator evaluation, where artifact type and timescale vary across generators. On caption-paired benchmark, GenVidBench, we identify two signatures that prior detectors do not jointly exploit: AI-generated videos exhibit smoother frame-to-frame temporal residuals at the pixel level, and more compact trajectories in the semantic feature space, indicating a temporal smoothness gap at both levels. We further observe that, when raw video is fed into a Spiking 
    
[^372]: 面向软体机器人的拓扑驱动防缠绕控制

    Topology-Driven Anti-Entanglement Control for Soft Robots

    [https://arxiv.org/abs/2605.05236](https://arxiv.org/abs/2605.05236)

    本文提出一种拓扑驱动的多智能体强化学习（TD-MARL）框架，通过共享拓扑状态的集中式学习协调多软体机器人系统，解决了高约束环境下防缠绕控制中可观测性不足与训练不稳定的问题。

    

    在复杂受限环境下的精密制造领域，软体机器人的作用日益突出，基于多智能体强化学习实现防缠绕控制已成为研究热点。当前的核心问题之一是在高度受限环境中协调多个机器人完成解缠绕操作。现有的分布式训练框架在高密度障碍和不稳定环境中面临可观测性方面的挑战，导致学习效果不佳。本文提出了一种拓扑驱动的多智能体强化学习（TD-MARL）框架，用于协调多机器人系统以避免缠绕。具体而言，关键网络采用集中式学习方式，使每个智能体能够通过共享拓扑状态感知其他智能体的策略，从而缓解训练不稳定问题（摘要原文在此处截断）。

    arXiv:2605.05236v2 Announce Type: replace-cross  Abstract: In the field of precision manufacturing in complex constrained environments, the role of soft robots is increasingly prominent, and the realization of anti-winding control based on multi-intelligent body reinforcement learning has become a research hotspot. One of the core problems at present is to coordinate multiple robots to complete the unwinding operation in a highly constrained environment. The existing distributed training framework faces some observability challenges in high-density barrier and unstable environments, resulting in poor learning results. This paper proposes a topology-driven Multi-Agent Reinforcement Learning (TD-MARL) framework to coordinate multi-robot systems to avoid entanglement. Specifically, the critical network adopts centralized learning, so that each intelligent body can perceive the strategies of other intelligent bodies by sharing the topological state, thus alleviating the training instabilit
    
[^373]: 超越视觉保真度：通过下游任务集成对大规模遥感影像超分辨率模型进行基准评估

    Beyond Visual Fidelity: Benchmarking Super-Resolution Models for Large-Scale Remote Sensing Imagery via Downstream Task Integration

    [https://arxiv.org/abs/2605.00310](https://arxiv.org/abs/2605.00310)

    该论文提出了GeoSR-Bench基准数据集，通过将土地覆盖分类、生物量估计、变化检测等下游任务集成到评估流程中，突破传统PSNR/SSIM保真度指标的局限，实现对遥感影像超分辨率模型实用价值的更全面评估。

    

    arXiv:2605.00310v3 公告类型：replace-cross 摘要：超分辨率（SR）技术在从低分辨率输入重建高分辨率图像方面已取得重大进展。提升的分辨率不仅带来视觉上的增强，也对监测任务具有实用价值。尤其是，超分辨率技术正越来越多地被开发用于基于卫星的对地观测，其应用涵盖城市规划、农业、生态学和灾害响应等领域。然而，现有的超分辨率研究和基准测试通常采用PSNR或SSIM等保真度指标，而超分辨率图像的真正价值在于支持诸如土地覆盖分类、生物量估计和变化检测等下游任务。为弥合这一差距，我们提出了GeoSR-Bench，这是一个下游任务集成的超分辨率基准数据集，用于在保真度指标之外评估超分辨率模型。GeoSR-Bench包含来自约36,000个位置、覆盖多样土地覆盖类型的空间位置对应、时间对齐且经过质量控制的图像对。

    arXiv:2605.00310v3 Announce Type: replace-cross  Abstract: Super-resolution (SR) techniques have made major advances in reconstructing high-resolution images from low-resolution inputs. The increased resolution provides visual enhancement and utility for monitoring tasks. In particular, SR has been increasingly developed for satellite-based Earth observation, with applications in urban planning, agriculture, ecology, and disaster response. However, existing SR studies and benchmarks typically use fidelity metrics such as PSNR or SSIM, whereas the true utility of super-resolved images lies in supporting downstream tasks such as land cover classification, biomass estimation, and change detection. To bridge this gap, we introduce GeoSR-Bench, a downstream task-integrated SR benchmark dataset to evaluate SR models beyond fidelity metrics. GeoSR-Bench comprises spatially co-located, temporally aligned, and quality-controlled image pairs from about 36,000 locations across diverse land covers
    
[^374]: 医院质量改进中的AI辅助假设生成

    AI-Assisted Hypothesis Generation for Hospital Quality Improvement

    [https://arxiv.org/abs/2604.20055](https://arxiv.org/abs/2604.20055)

    该论文提出利用大语言模型加速医院质量改进中传统上依赖专家定性方法的假设生成阶段，并主张将问题规范本身也作为可迭代调整的对象，以应对这一“模糊”且探索性的任务。

    

    医院质量改进（QI）项目通常分为两个阶段：（1）假设生成，即识别可能提升医疗质量与效率的候选因素；（2）验证性分析，即针对这些因素实施并评估相应的质量改进举措。尽管大多数研究聚焦于验证性阶段，但假设生成同样重要，因为它决定了团队努力的聚焦方向。目前，假设生成依赖于专家驱动的定性方法（例如精益医疗工具），这些方法往往流于轶事性、难以复现、耗时且资源密集。大语言模型（LLM）可以通过大规模分析临床笔记来加速这一过程，但现有AI工具只能很好地扩展界定明确的任务，而假设生成是一个“模糊”、探索性且迭代的过程。我们提出不仅将LLM指令，而且将问题规范和……（原文摘要至此截断）

    arXiv:2604.20055v2 Announce Type: replace  Abstract: Hospital quality improvement (QI) programs generally proceed in two stages: (1) hypothesis generation, which identifies candidate factors that may improve care quality and efficiency, and (2) confirmatory analyses, which implement and evaluate QI initiatives targeting these factors. Although most research focuses on the confirmatory stage, hypothesis generation is equally important because it shapes where teams focus their effort. Currently, hypothesis generation depends on expert-driven, qualitative methods (e.g., Lean Healthcare tools) that can be anecdotal, hard to reproduce, time-consuming, and resource-intensive. Large language models (LLMs) could accelerate this process by analyzing clinical notes at scale, but existing AI tools can only scale up well-defined tasks, whereas hypothesis generation is a "fuzzy," exploratory, and iterative process. We propose treating not only LLM instructions but also the problem specification and
    
[^375]: OpenTME：一个来自TCGA的AI驱动的H&E肿瘤微环境谱开放数据集

    OpenTME: An Open Dataset of AI-powered H&E Tumor Microenvironment Profiles from TCGA

    [https://arxiv.org/abs/2604.12075](https://arxiv.org/abs/2604.12075)

    该研究发布了OpenTME数据集，利用AI病理基础模型从TCGA十种癌症类型的5,268张H&E全切片图像中生成大规模开放的肿瘤微环境定量图谱，每张切片包含超过20,000个定量指标。

    

    肿瘤微环境（TME）在癌症进展、治疗反应和患者预后中发挥着核心作用，然而从常规苏木精-伊红（H&E）染色组织病理学切片中获得大规模、一致且定量的TME表征仍然稀缺。我们推出了OpenTME，这是一个开放获取的数据集，包含从癌症基因组图谱（TCGA）中十种癌症类型（膀胱癌、乳腺癌、结直肠癌、头颈癌、肝癌、肺癌、卵巢癌、胰腺癌、前列腺癌和胃癌）的5,268张H&E染色全切片图像中预先计算得到的TME谱。所有输出均由HETA（H&E组织分析）生成，这是一个基于Atlas系列病理学基础模型构建的AI驱动应用程序，可执行组织质量控制、组织分割、细胞检测与分类、肿瘤区域和三级淋巴结构（TLS）检测以及空间邻域分析，每张切片可产生超过20,000个定量读数……

    arXiv:2604.12075v2 Announce Type: replace-cross  Abstract: The tumor microenvironment (TME) plays a central role in cancer progression, treatment response, and patient outcomes, yet large-scale, consistent, and quantitative TME characterization from routine hematoxylin and eosin (H&E)-stained histopathology remains scarce. We introduce OpenTME, an open-access dataset of pre-computed TME profiles derived from 5,268 H&E-stained whole-slide images across ten cancer types (bladder, breast, colorectal, head and neck, liver, lung, ovarian, pancreatic, prostate, and stomach cancer) from The Cancer Genome Atlas (TCGA). All outputs were generated using HETA (H&E Tissue Analysis), an AI-powered application built on the Atlas family of pathology foundation models, which performs tissue quality control, tissue segmentation, cell detection and classification, tumor zone and tertiary lymphoid structure (TLS) detection, and spatial neighborhood analysis, yielding over 20,000 quantitative readouts per
    
[^376]: Agent² RL-Bench：大语言模型智能体能否工程化构建智能体强化学习后训练？

    Agent^2 RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training?

    [https://arxiv.org/abs/2604.10547](https://arxiv.org/abs/2604.10547)

    该论文提出了 Agent² RL-Bench——一个紧凑的诊断基准，通过面向智能体的统一接口和三个层级的六项任务，评估大语言模型智能体能否在固定预算内自主设计、实现、调试并闭环执行强化学习后训练流水线来提升基础模型性能。

    

    我们推出了 Agent² RL-Bench，这是一个用于评估智能体强化学习后训练的紧凑型诊断基准，它测试大语言模型智能体能否自主地设计、实现、调试并执行能够提升基础模型性能的后训练流水线。强化学习后训练日益成为驱动模型对齐与专门化的关键手段，然而现有基准大多是静态的，只奖励监督微调或脚本生成，而未能评估智能体闭环完成交互式强化学习流程的能力。Agent² RL-Bench 提供了一个面向智能体的统一接口：每次运行从一个隔离的工作区开始，其中包含基础模型、任务数据、指令说明和一个评分 API，智能体必须在固定预算内通过训练模型并提交产物进行评估来不断迭代。该基准涵盖三个层级的六项任务，从静态的基于规则的训练，到基于评判模型的优化，再到带轨迹采集的闭环在线强化学习。两项诊断技能……

    arXiv:2604.10547v3 Announce Type: replace  Abstract: We introduce Agent^2 RL-Bench, a compact diagnostic benchmark for evaluating agentic RL post-training, which tests whether LLM agents can autonomously design, implement, debug, and execute post-training pipelines that improve foundation models. RL post-training increasingly drives model alignment and specialization, yet existing benchmarks are largely static, rewarding supervised fine-tuning or script generation without assessing an agent's ability to close an interactive RL loop. Agent^2 RL-Bench provides a unified agent-facing interface: each run starts from an isolated workspace containing a base model, task data, instructions, and a grading API, and agents must iterate within a fixed budget by training models and submitting artifacts for evaluation. The benchmark spans six tasks across three levels, from static rule-based training to judge-based optimization and closed-loop online RL with trajectory collection. Two diagnostic ski
    
[^377]: 基于有界不良评判者的稳健LLM评估：一种约束最大似然方法

    Robust LLM Evaluation with Boundedly Bad Judges: A Constrained Maximum Likelihood Approach

    [https://arxiv.org/abs/2604.03257](https://arxiv.org/abs/2604.03257)

    该论文提出约束最大似然估计（CMLE）方法，通过数据驱动约束显式建模LLM评判者的错误率，以受控偏差换取方差降低，从而在可靠标签有限的情况下实现更稳健的大规模LLM评估。

    

    大规模评估语言模型通常需要调和昂贵但可靠的标签（例如来自人类评分者的标签）与廉价但含噪的代理标签（例如来自LLM作为评判者的自动评估器）。本文提出了约束最大似然估计（CMLE），这是一种基于似然的LLM评估方法，它通过真阳性率和假阳性率显式建模代理标签中的误差，并受数据驱动约束的限制。在功能上，这些约束起到自适应正则化的作用，限制了可行的评判者错误率，特别是在可用的可靠标签数量有限的情况下。与渐近无偏估计器（例如基于预测驱动推断的估计器）不同，这一表述在约束设定不当时接受受控的偏差，以换取相对于包括无约束最大似然在内的各种基线的方差降低。

    arXiv:2604.03257v2 Announce Type: replace-cross  Abstract: Evaluating language models at scale often involves reconciling costly, but reliable, labels (e.g., from human raters) with cheap, but noisy, proxy labels (e.g., from LLM-as-a-judge autoraters). This paper proposes constrained maximum likelihood estimation (CMLE), a likelihood-based approach to large language model (LLM) evaluation that explicitly models the error in the proxy labels via their true-positive and false-positive rates, subject to data-driven constraints. Functionally, these constraints act as an adaptive regularization that restricts the feasible judge error rates, particularly when the available number of reliable labels is limited. In contrast to asymptotically unbiased estimators (such as those based on prediction-powered inference), this formulation accepts a controlled bias under misspecified constraints in exchange for variance reduction relative to various baselines including unconstrained maximum likelihood
    
[^378]: λSplit：面向荧光显微镜的自监督内容感知光谱解混方法

    {\lambda}Split: Self-Supervised Content-Aware Spectral Unmixing for Fluorescence Microscopy

    [https://arxiv.org/abs/2603.23647](https://arxiv.org/abs/2603.23647)

    提出了λSplit，一种基于物理信息的自监督深度生成模型，通过分层变分自编码器学习浓度图的条件分布，并利用可微分光谱混合器，实现了对荧光显微镜光谱图像的内容感知解混。

    

    在荧光显微镜中，光谱解混旨在从捕获了混合荧光团发射信号的光谱图像中恢复各个荧光团的浓度。由于经典方法采用逐像素操作并依赖最小二乘拟合，其性能会随着发射光谱重叠程度的加剧和噪声水平的升高而下降，这表明一种能够学习并利用结构先验的数据驱动方法可能会带来更好的结果。虽然目前存在基于学习的光谱成像方法，但它们要么未针对显微镜数据进行优化，要么是为极特殊的场景开发的，并不适用于荧光显微镜的设置。为解决这一问题，我们提出了λSplit，这是一种融合物理信息的深度生成模型，利用分层变分自编码器学习浓度图的条件分布。一个完全可微分的光谱混合器强制其与图像保持一致性……

    arXiv:2603.23647v4 Announce Type: replace-cross  Abstract: In fluorescence microscopy, spectral unmixing aims to recover individual fluorophore concentrations from spectral images that capture mixed fluorophore emissions. Since classical methods operate pixel-wise and rely on least-squares fitting, their performance degrades with increasingly overlapping emission spectra and higher levels of noise, suggesting that a data-driven approach that can learn and utilize a structural prior might lead to improved results. Learning-based approaches for spectral imaging do exist, but they are either not optimized for microscopy data or are developed for very specific cases that are not applicable to fluorescence microscopy settings. To address this, we propose {\lambda}Split, a physics-informed deep generative model that learns a conditional distribution over concentration maps using a hierarchical Variational Autoencoder. A fully differentiable Spectral Mixer enforces consistency with the image 
    
[^379]: xAppGen：基于大语言模型驱动的自然语言编程实现敏捷xApp开发

    xAppGen: LLM-driven Natural Language Programming for Agile xApp Development

    [https://arxiv.org/abs/2603.18604](https://arxiv.org/abs/2603.18604)

    该论文提出了首个由大语言模型驱动的自然语言编程框架xAppGen，能在几分钟内自动将用户意图转化为可部署的xApp，实现xApp开发全流程自动化，大幅降低O-RAN应用开发门槛。

    

    传统的RAN系统封闭且单一，扼杀了创新。开放无线接入网络（O-RAN）所提供的开放性和可编程性，有望通过控制面应用——xApps来彻底变革蜂窝网络。然而，xApps的开发（通常由第三方开发者完成）仍然耗时且繁琐，往往需要数月的手工编码和集成工作，这阻碍了新功能在实际中的落地推广。为了降低开发者和网络运营商的xApp开发门槛，我们提出了xAppGen，这是首个由大语言模型（LLM）驱动的自然语言编程框架，用于敏捷xApp开发，可自动化整个xApp开发流程。简而言之，xAppGen能够在几分钟内将高层用户意图转化为可快速部署的xApps，无需手工编码或测试。为此，该框架构建了一个全自动的xApp生成流水线，其集成了……（原文在此处截断）

    arXiv:2603.18604v2 Announce Type: replace-cross  Abstract: Traditional RAN systems are closed and monolithic, stifling innovation. The openness and programmability enabled by Open Radio Access Network (O-RAN) are envisioned to revolutionize cellular networks with control-plane applications--xApps. The development of xApps (typically by third-party developers), however, remains time-consuming and cumbersome, often requiring months of manual coding and integration, which hinders the roll-out of new functionalities in practice. To lower the barrier of xApp development for both developers and network operators, we present xAppGen, the first LLM-driven natural language programming framework for agile xApps that automates the entire xApp development pipeline. In a nutshell, xAppGen turns high-level user intents into swiftly deployable xApps within minutes, eliminating the need for manual coding or testing. To this end, AutORAN builds a fully automated xApp generation pipeline, which integrat
    
[^380]: 面向杂乱环境中顺序操作的以对象为中心的空间推理学习

    Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments

    [https://arxiv.org/abs/2603.02511](https://arxiv.org/abs/2603.02511)

    该论文提出模块化框架Unveiler，将场景推理与物体选择（通过空间关系编码器）和动作执行相分离，无需真实机器人训练数据即可在杂乱环境中实现顺序操作任务。

    

    在杂乱环境中进行顺序操作需要选择移除哪些物体并确定如何操作它们。联合学习这两个决策会使训练复杂化，并使失败难以诊断。我们提出了Unveiler，一个模块化框架，它通过离散的对象索引接口将场景推理和对象选择与动作执行分离。其空间关系编码器（SRE）选择最佳的抓取对象，而独立的动作解码器生成方向离散化的推-抓动作。SRE分两个阶段进行训练：首先从启发式示范中进行模仿学习，随后利用模拟的物体移除进行基于搜索的策略改进。这种训练过程不需要真实机器人训练数据，并允许SRE与不同的执行策略配对使用。我们在PyBullet仿真和实体Dofbot-Pro机器人上对Unveiler进行了评估。在真实机器人实验中，它成功取出了两个被遮挡……（原文摘要在此处截断）

    arXiv:2603.02511v2 Announce Type: replace-cross  Abstract: Sequential manipulation in clutter requires selecting which objects to remove and determining how to manipulate them. Learning both decisions jointly can complicate training and make failures difficult to diagnose. We present Unveiler, a modular framework that separates scene reasoning and object selection from action execution through a discrete object-index interface. Its Spatial Relationship Encoder (SRE) selects the best object to grasp, while a separate Action Decoder generates orientation-discretized push-grasp actions. The SRE is trained in two stages: imitation learning from heuristic demonstrations, followed by search-based policy improvement using simulated object removals. This training procedure requires no real-robot training data and allows the SRE to be paired with a different execution policy. We evaluate Unveiler in PyBullet simulation and on a physical Dofbot-Pro. In real-robot experiments, it retrieves two oc
    
[^381]: AI GameStore：基于人类游戏的可扩展、开放式机器通用智能评估

    AI GameStore: Scalable, Open-Ended Evaluation of Machine General Intelligence with Human Games

    [https://arxiv.org/abs/2602.17594](https://arxiv.org/abs/2602.17594)

    提出以“人类游戏多元宇宙”——所有由人类为人类设计的游戏——作为可扩展、开放式的机器通用智能评估框架，通过让AI在相同资源条件下与人类玩家比较玩和学玩所有人类游戏的能力，克服传统基准测试狭窄且易饱和的局限。

    

    在技术快速发展的时代，将机器智能与人类通用智能的广泛范围进行严格评估变得越来越重要且充满挑战。传统的AI基准测试通常只评估人类活动中有限范围内的狭窄能力，而且大多是静态的，随着模型能力的提升很快就会饱和。我们提出，评估AI系统中类人通用智能的一种更有前景的方法是通过一种特别强大的通用博弈形式：研究AI系统如何以及多么擅长玩和学习所有可想象的人类游戏，并与具有相同经验水平、时间或其他资源的人类玩家进行比较。我们将“人类游戏”定义为由人类为人类设计的游戏，并论证了这一由人们可以想象和享受的所有此类游戏构成的空间——“人类游戏多元宇宙”（Multiverse of Human Games）——的评估适用性。作为迈向这一愿景的第一步……

    arXiv:2602.17594v2 Announce Type: replace  Abstract: Rigorously evaluating machine intelligence against the broad spectrum of human general intelligence has become increasingly important and challenging in this era of rapid technological advance. Conventional AI benchmarks typically assess only narrow capabilities in a limited range of human activity. Most are also static, quickly saturating as model capabilities improve. We propose that a more promising way to evaluate human-like general intelligence in AI systems is through a particularly strong form of general game playing: studying how and how well they play and learn to play all conceivable human games, in comparison to human players with the same level of experience, time, or other resources. We define a "human game" to be a game designed by humans for humans, and argue for the evaluative suitability of this space of all such games people can imagine and enjoy -- the "Multiverse of Human Games". Taking a first step towards this v
    
[^382]: 关于具有 Q\* 近似与部分覆盖的离线强化学习的复杂性研究

    On the Complexity of Offline Reinforcement Learning with Q*-Approximation and Partial Coverage

    [https://arxiv.org/abs/2602.12107](https://arxiv.org/abs/2602.12107)

    该论文通过信息论下界证明仅凭 Q\* 可实现性与贝尔曼完整性不足以实现部分覆盖下样本高效的离线强化学习，并提出一个通用的决策估计框架，将离线强化学习的复杂度分解为决策复杂度与价值估计误差两部分。

    

    我们研究了在 Q\* 近似与部分覆盖条件下的离线强化学习，这一设定激发了许多实用算法（如保守 Q 学习，CQL；Kumar 等，2020），但其受到的理论关注十分有限。我们的工作受到以下开放问题的启发：“Q\* 可实现性与贝尔曼完整性是否足以支撑部分覆盖下样本高效的离线强化学习？”我们通过信息论下界对该问题给出了否定回答。为了识别能够使部分覆盖下样本高效离线强化学习成为可能的额外结构，我们受在线强化学习中无模型决策估计系数（DEC）（Foster 等，2023b；Liu 等，2025b）的启发，引入了一个通用的决策估计框架。该框架将离线强化学习的复杂度分解为决策复杂度和价值估计误差，从而可以对这两个子问题进行模块化研究。我们的结果不仅统一了已有的……

    arXiv:2602.12107v3 Announce Type: replace-cross  Abstract: We study offline reinforcement learning under $Q^\star$-approximation and partial coverage, a setting that motivates practical algorithms such as Conservative $Q$-Learning (CQL; Kumar et al., 2020) but has received limited theoretical attention. Our work is inspired by the following open question: "Are $Q^\star$-realizability and Bellman completeness sufficient for sample-efficient offline RL under partial coverage?"   We answer in the negative via an information-theoretic lower bound. To identify additional structure that enables sample-efficient offline RL under partial coverage, we introduce a general decision-estimation framework, inspired by model-free decision-estimation coefficients (DEC) for online RL (Foster et al., 2023b; Liu et al., 2025b). Our framework decomposes offline RL complexity into decision complexity and value estimation error. This allows modular study of both sub-problems. Our result not only unifies exi
    
[^383]: 何时想象及想象多少：基于世界模型的视觉空间推理自适应测试时扩展

    When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning

    [https://arxiv.org/abs/2602.08236](https://arxiv.org/abs/2602.08236)

    该论文提出AVIC自适应测试时框架，通过深入分析何时需要视觉想象以及想象的最优程度，利用世界模型自适应地调控想象过程，从而提升多模态大语言模型视觉空间推理的准确性与效率。

    

    尽管多模态大语言模型进展迅速，但当正确答案依赖于场景在未见或替代视角下的呈现方式时，视觉空间推理仍然不可靠。近期工作通过引入世界模型进行视觉想象来增强推理能力，但诸如想象何时真正必要、多少想象是有益的、何时想象会变得有害等问题仍缺乏深入理解。在实际应用中，不加区分的想象会增加计算开销，甚至可能因引入误导性证据而降低性能。在这项工作中，我们对测试时视觉想象作为一种可控的空间推理资源进行了深入分析。我们首先研究了何时静态视觉证据已经足够、何时想象能够改善推理，以及过度或不必要的想象如何影响准确性与效率。为支持这一分析，我们随后提出了AVIC，一个基于世界模型的自适应测试时框架。

    arXiv:2602.08236v3 Announce Type: replace-cross  Abstract: Despite rapid progress in MLLMs, visual spatial reasoning remains unreliable when correct answers depend on how a scene would appear under unseen or alternative viewpoints. Recent work addresses this by augmenting reasoning with world models for visual imagination, but questions such as when imagination is actually necessary, how much of it is beneficial, and when it becomes harmful, remain poorly understood. In practice, indiscriminate imagination can increase computation and even degrade performance by introducing misleading evidence. In this work, we present an in-depth analysis of test-time visual imagination as a controllable resource for spatial reasoning. We first study when static visual evidence is sufficient, when imagination improves reasoning, and how excessive or unnecessary imagination affects accuracy and efficiency. To support this analysis, we then introduce AVIC, an adaptive test-time framework with world mode
    
[^384]: VLRS-Bench：一个面向遥感的视觉-语言推理基准

    VLRS-Bench: A Vision-Language Reasoning Benchmark for Remote Sensing

    [https://arxiv.org/abs/2602.07045](https://arxiv.org/abs/2602.07045)

    提出了首个专门针对遥感领域复杂推理的视觉-语言推理基准VLRS-Bench，涵盖认知、决策和预测三大维度的2,000个问答对，实验揭示了当前多模态大语言模型在遥感推理能力上的显著瓶颈。

    

    多模态大语言模型（MLLMs）的最新进展使其具备了复杂推理能力。然而，现有的遥感（RS）基准仍然严重偏向于目标识别和场景分类等感知任务。这一局限性阻碍了MLLMs在对认知要求较高的遥感应用中的发展。为解决这一问题，我们提出了视觉-语言推理基准（VLRS-Bench），这是首个专门致力于复杂遥感推理的基准。VLRS-Bench围绕认知、决策和预测三个核心维度进行结构化设计，包含2,000个问答对，平均问题长度为129.87个单词，涵盖14个任务和多达八个时间阶段。VLRS-Bench通过一个专门的构建流程完成，该流程整合了遥感特定的先验知识和专家知识，以确保地理空间的真实性和推理的复杂性。实验结果揭示了显著的瓶颈……

    arXiv:2602.07045v3 Announce Type: replace-cross  Abstract: Recent advancements in Multimodal Large Language Models (MLLMs) have enabled complex reasoning. However, existing remote sensing (RS) benchmarks remain heavily biased toward perception tasks, such as object recognition and scene classification. This limitation hinders the development of MLLMs for cognitively demanding RS applications. To address this, we propose a Vision Language ReaSoning Benchmark (VLRS-Bench), which is the first benchmark exclusively dedicated to complex RS reasoning. Structured across the three core dimensions of Cognition, Decision, and Prediction, VLRS-Bench comprises 2,000 question-answer pairs with an average question length of 129.87 words, spanning 14 tasks and up to eight temporal phases. VLRS-Bench is constructed via a specialized pipeline that integrates RS-specific priors and expert knowledge to ensure geospatial realism and reasoning complexity. Experimental results reveal significant bottlenecks
    
[^385]: EMA策略梯度：利用EMA锚点与Top-k KL驯服大语言模型的强化学习

    EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL

    [https://arxiv.org/abs/2602.04417](https://arxiv.org/abs/2602.04417)

    提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。

    

    强化学习（RL）已使大语言模型（LLM）能够掌握日益复杂的推理和智能体行为。在本工作中，我们提出两种简单的技术来改进面向大语言模型的策略梯度算法。首先，我们用指数移动平均（EMA）替换强化学习过程中的固定锚点策略，这类似于深度Q学习中的目标网络。其次，我们引入Top-$k$ KL估计器，它允许在精确KL与采样KL之间进行灵活插值。我们推导了使用EMA锚点的稳定性条件；此外，我们证明了Top-$k$ KL估计器在任意$k$值下都能产生无偏的KL值和无偏的梯度，同时兼具精确KL的优势。将这两种技术与GRPO相结合（EMA-PG）可带来显著的性能提升。在数学推理方面，它使R1蒸馏的Qwen-1.5B在OlympiadBench上达到53.9%，而GRPO为50.8%。在智能体强化学习领域，使用Qwen-3B……（摘要原文在此处截断）

    arXiv:2602.04417v2 Announce Type: replace-cross  Abstract: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-$k$ KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-$k$ KL estimator yields both unbiased KL values and unbiased gradients at any $k$, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B b
    
[^386]: 后训练困境：为什么我们应该重新思考顺序式SFT-RL范式

    The Post-Training Dilemma: Why We Should Rethink the Sequential SFT-RL Paradigm

    [https://arxiv.org/abs/2601.07389](https://arxiv.org/abs/2601.07389)

    本文从理论上证明SFT与RL在近乎正交的参数子空间上进行优化，导致RL阶段不可避免地损害SFT所获得的能力，且KL惩罚无法修复这一梯度不对齐问题，因此主张重新思考顺序式SFT-RL后训练范式。

    

    这篇立场论文认为，先监督微调后强化学习（SFT-then-RL）的顺序流程——作为大语言模型（LLM）后训练的事实标准范式——存在一个无法通过超参数调整来修复的根本性缺陷：SFT和RL在几乎正交的参数子空间上进行优化，因此RL不可避免地会损害前一阶段SFT所获得的能力。在Polyak-Łojasiewicz（PL）条件下（该条件在过参数化网络中被经验性观察到），我们证明RL至少会使SFT损失增加 $\frac{\mu}{2}\mathrm{dist}(\theta_{\mathrm{RL}}, \Theta^*_{\mathrm{SFT}})^2$（其中 $\mu>0$）。尽管KL惩罚可以约束分布偏移，但它无法纠正两个目标之间潜在的梯度不对齐问题。在Qwen3-0.6B上跨四个任务的实验证实了理论预测：在MATH-500、Alpaca、HH-RLHF和SST-2上（原文摘要在此处截断）。

    arXiv:2601.07389v3 Announce Type: replace-cross  Abstract: This position paper argues that the sequential supervised fine-tuning followed by reinforcement learning (SFT-then-RL) pipeline---the de facto paradigm for large language model (LLM) post-training---has a fundamental property that cannot be fixed by hyperparameter tuning: SFT and RL optimize over nearly orthogonal parameter subspaces, so RL inevitably degrades the capabilities acquired during the preceding SFT phase. Under the Polyak-Lojasiewicz (PL) condition, which is empirically observed for over-parameterized networks, we show that RL increases the SFT loss by at least $\frac{\mu}{2}\mathrm{dist}(\theta_{\mathrm{RL}}, \Theta^*_{\mathrm{SFT}})^2$ for some $\mu>0$. Although the KL penalty can constrain distributional shift, it cannot correct the underlying gradient misalignment between the two objectives. Experiments on Qwen3-0.6B across four tasks confirm the theoretical predictions: on MATH-500, Alpaca, HH-RLHF, and SST-2, 
    
[^387]: 大语言模型如何歪曲美国气候民意

    How Large Language Models Misrepresent American Climate Opinions

    [https://arxiv.org/abs/2512.23889](https://arxiv.org/abs/2512.23889)

    研究发现大语言模型在模拟美国气候民意时系统性地压缩观点多样性，将预测拉向人口均值并套用刻板的性别假设，在种族与性别交叉维度上尤其失真。

    

    大语言模型越来越多地被用于模拟公众舆论，作为“硅基样本”替代调查受访者。然而，它们能否忠实呈现美国气候民意，特别是在种族、性别和意识形态的交叉维度上，在很大程度上仍未得到检验。我们以一项具有全国代表性的美国气候民意调查中978名受访者的画像作为提示，输入六个大语言模型，并将模型生成的回答与这些受访者本人在20个问题上的真实回答进行比较。研究发现，这些模型压缩了美国气候舆论的多样性，将预测拉向人口总体均值：关注度较低的群体被预测为更加关注气候，而关注度较高的群体则被预测为关注较少，预测回答的方差平均比人类真实回答低29%。这种压缩呈现出交叉性特征：模型套用了单一的性别假设，在所有种族群体中都预测女性比男性更支持气候行动。

    arXiv:2512.23889v3 Announce Type: replace-cross  Abstract: Large language models are increasingly used to simulate public opinion, standing in for survey respondents as "silicon samples." Whether they faithfully represent U.S. climate opinion, particularly at the intersections of race, gender, and ideology, remains largely untested. We prompted six large language models with the profiles of 978 respondents to a nationally representative U.S. climate opinion survey and compared the generated responses with those respondents' own answers across 20 questions. The models compress the diversity of American climate opinion, pulling predictions toward the population mean: less-concerned groups are predicted as more concerned, more-concerned groups as less, and the variance of predicted responses falls, on average, 29% below that of human responses. The compression is intersectional. The models apply one gender assumption, predicting women as more pro-climate than men in every racial group. Th
    
[^388]: 行列式簇的变分分析

    Variational analysis of determinantal varieties

    [https://arxiv.org/abs/2511.22613](https://arxiv.org/abs/2511.22613)

    本文提出统一的变分分析框架，推导出各类低秩集合（矩阵、张量、对称矩阵、半正定矩阵等）的一阶与二阶切集显式公式及切集相交规则，并据此给出了非光滑问题与其光滑参数化具有等价二阶稳定点的充要条件。

    

    行列式簇——即秩有界的矩阵或张量集合——在低秩优化领域引起了越来越多的关注。低秩集合的切锥已被广泛研究，并支撑了一系列几何方法；而编码曲率信息的二阶几何则更为复杂。在本工作中，我们开发了一个统一框架，用于推导各类低秩集合的一阶与二阶切集的显式公式，包括低秩矩阵、张量、对称矩阵以及半正定矩阵。该框架还适用于低秩集合与另一满足某些假设的集合的交集，从而给出了切集相交规则。通过切集的视角，我们建立了一个充要条件，在该条件下非光滑问题与其光滑参数化具有等价的二阶稳定点。作为应用，我们利用切……（摘要原文在此处被截断）

    arXiv:2511.22613v3 Announce Type: replace-cross  Abstract: Determinantal varieties---the sets of bounded-rank matrices or tensors---have attracted growing interest in low-rank optimization. The tangent cone to low-rank sets is widely studied and underpins a range of geometric methods. The second-order geometry, which encodes curvature information, is more intricate. In this work, we develop a unified framework to derive explicit formulas for both first- and second-order tangent sets to various low-rank sets, including low-rank matrices, tensors, symmetric matrices, and positive semidefinite matrices. The framework also accommodates the intersection of a low-rank set and another set satisfying some assumptions, thereby yielding a tangent intersection rule. Through the lens of tangent sets, we establish a necessary and sufficient condition under which a nonsmooth problem and its smooth parameterization share equivalent second-order stationary points. As applications, we exploit tangent s
    
[^389]: 预测电力使用对公共健康的影响

    Predicting Public Health Impacts of Electricity Usage

    [https://arxiv.org/abs/2511.22031](https://arxiv.org/abs/2511.22031)

    提出了HealthPredictor——一个将电力使用与公共健康结果联系起来的端到端AI模型，通过燃料组合预测、空气质量模拟和货币化健康损害评估三个组件，实现基于健康信息的需求侧电力管理。

    

    电力部门是空气污染物排放的主要来源，影响着几乎每个社区的公共健康。尽管监管措施已经减少了空气污染物，但化石燃料仍然是能源供应的重要组成部分，这凸显了需要更先进的需求侧方法来减少公共健康影响。为了实现基于健康信息的需求侧管理，我们引入了HealthPredictor，这是一个特定领域的AI模型，提供了将电力使用与公共健康结果联系起来的端到端流程。该模型包含三个组件：燃料组合预测器（估计不同发电来源的贡献）、空气质量转换器（模拟污染物排放和大气扩散）以及健康影响评估器（将由此产生的污染物变化转化为货币化的健康损害）。在美国多个地区，我们的健康驱动优化框架……（摘要被截断）

    arXiv:2511.22031v3 Announce Type: replace-cross  Abstract: The electric power sector is a leading source of air pollutant emissions, impacting the public health of nearly every community. Although regulatory measures have reduced air pollutants, fossil fuels remain a significant component of the energy supply, highlighting the need for more advanced demand-side approaches to reduce the public health impacts. To enable health-informed demand-side management, we introduce HealthPredictor, a domain-specific AI model that provides an end-to-end pipeline linking electricity use to public health outcomes. The model comprises three components: a fuel mix predictor that estimates the contribution of different generation sources, an air quality converter that models pollutant emissions and atmospheric dispersion, and a health impact assessor that translates resulting pollutant changes into monetized health damages. Across multiple regions in the United States, our health-driven optimization fra
    
[^390]: Rhythm-TTS：面向零样本文本转语音的高保真捷径潜变量建模与概率化节奏生成

    Rhythm-TTS: High-Fidelity Shortcut Latent Modeling and Probabilistic Rhythm Generation for Zero-Shot Text-to-Speech

    [https://arxiv.org/abs/2510.02848](https://arxiv.org/abs/2510.02848)

    Rhythm-TTS 提出了一种捷径零样本文本转语音框架，将预测的离散编解码器标记作为语义-声学先验，直接生成高保真的整体连续语音潜变量，并结合概率化节奏生成，从而兼顾离散标记的可因子分解性与连续表示的高保真度。

    

    零样本文本转语音（TTS）系统通常建立在离散编解码器标记（codec tokens）或连续声学表示之上。离散标记模型受益于稳定的序列建模和显式的语音因子分解，但其最终通过码本查找和码字组合来合成语音，这会累积误差并对自然度和说话人相似度造成质量上限。连续表示模型可以直接生成细粒度的语音潜变量，但往往需要更繁重的上下文建模且对齐不够显式。本文提出 Rhythm-TTS，一种捷径（shortcut）零样本文本转语音框架，它将离散编解码器标记的可因子分解性与连续潜变量生成的高保真度相结合。Rhythm-TTS 不把预测出的编解码器标记直接作为最终声学表示，而是将其用作语义-声学先验，并直接生成整体的连续语音潜变量……（摘要原文在此处截断）

    arXiv:2510.02848v2 Announce Type: replace-cross  Abstract: Zero-shot text-to-speech (TTS) systems are commonly built on either discrete codec tokens or continuous acoustic representations. Discrete-token models benefit from stable sequence modeling and explicit speech factorization, but they ultimately synthesize speech through codebook lookup and codeword composition, which can accumulate errors and impose a quality ceiling on naturalness and speaker similarity. Continuous-representation models can generate fine-grained speech latents directly, but often require heavier in-context modeling and less explicit alignment. This paper presents \textbf{Rhythm-TTS}, a shortcut zero-shot TTS framework that combines the factorizability of discrete codec tokens with the fidelity of continuous latent generation. Instead of using predicted codec tokens as the final acoustic representation, Rhythm-TTS uses them as a semantic-acoustic prior and directly generates holistic continuous speech latents w
    
[^391]: FlowDrive：基于数据平衡的调节式流匹配轨迹规划方法

    FlowDrive: moderated flow matching with data balancing for trajectory planning

    [https://arxiv.org/abs/2509.21961](https://arxiv.org/abs/2509.21961)

    FlowDrive通过按轨迹模式重加权来平衡长尾驾驶数据，并提出带有环内微小扰动引导的调节式流匹配规划器，在保持场景一致性的同时增加轨迹多样性，在nuPlan和interPlan基准上取得了学习型规划器中的最先进结果。

    

    基于学习的规划器对驾驶数据的长尾分布非常敏感。常见机动操作在数据集中占主导地位，而危险或罕见场景则较为稀疏。这种不平衡会使模型偏向频繁出现的情况，降低其在关键场景上的性能。为解决这一问题，我们比较了多种训练数据采样平衡策略，发现按轨迹模式进行重加权是一种有效的方法。随后我们提出了FlowDrive，这是一种流匹配轨迹规划器，它学习条件整流流，能够以少量的流匹配步骤将噪声直接映射到轨迹分布。我们进一步引入了调节式环内引导，在流步骤之间注入微小扰动，在保持场景一致性的同时系统地增加轨迹多样性。在nuPlan和专注于交互的interPlan基准测试中，FlowDrive在学习型规划器中取得了最先进的成果，并接近……

    arXiv:2509.21961v4 Announce Type: replace-cross  Abstract: Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches m
    
[^392]: LLM4Rail：一个由大语言模型增强的个性化铁路服务平台

    LLM4Rail: An LLM-Augmented Personalized Railway Services Platform

    [https://arxiv.org/abs/2507.23377](https://arxiv.org/abs/2507.23377)

    该论文提出了LLM4Rail个性化铁路服务平台，其核心创新是包含经验存储与检索（ESR）和诊断性观察生成（DOG）两大机制的迭代式ER-QTAO推理框架，并构建了中国铁路餐饮数据集（CRFD）以实现个性化的车上餐饮推荐服务。

    

    大语言模型（LLM）已经深刻地重塑了各行各业。为满足日益增长的个性化铁路服务需求，我们开发了LLM4Rail——一个新颖的由大语言模型增强的个性化铁路服务平台。在大语言模型的赋能下，LLM4Rail提供了专门的模块，用于车票与天气查询、餐饮推荐以及闲聊对话。在LLM4Rail中，我们提出了迭代的“经验-检索-问题-思考-行动-观察（ER-QTAO）”推理框架。该框架引入了两个关键机制：经验存储与检索（ESR）和诊断性观察生成（DOG），以增强大语言模型以复杂参数调用恰当外部工具的能力。为了提供个性化的车上餐饮服务，我们首先构建了中国铁路餐饮数据集，该数据集涵盖了按城市、菜系、年龄段和辣度级别分类的众多特色菜肴。

    arXiv:2507.23377v2 Announce Type: replace  Abstract: Large language models (LLMs) have significantly reshaped different walks of business. To meet the increasing demands for individualized railway service, we develop LLM4Rail - a novel LLM-augmented personalized railway services platform. Empowered by the LLM, LLM4Rail provides dedicated modules for ticket and weather inquiries, food & drink recommendations, and casual conversation. In LLM4Rail, we propose the iterative "Experience-Retrieval Question-Thought-Action-Observation (ER-QTAO)" reasoning framework. It introduces two key mechanisms, Experience Storage and Retrieval (ESR) and Diagnostic Observation Generation (DOG), to enhance the LLM's ability to invoke appropriate external tools with complex arguments. To provide personalized onboard dining services, we first construct the Chinese Railway Food & Drink (CRFD) dataset, which covers a wide range of signature dishes categorized by city, cuisine, age group, and spiciness level, an
    
[^393]: TELEVAL：面向中文交互场景的口语语言模型基准测试

    TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios

    [https://arxiv.org/abs/2507.18061](https://arxiv.org/abs/2507.18061)

    TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。

    

    arXiv:2507.18061v4 公告类型：替换交叉 摘要：口语语言模型（SLMs）预期能够支持超越任务完成的自然口语交互。然而，现有的SLM基准测试主要评估结构化环境中的语义正确性，并且对基于声学上下文的交互行为评估有限。为解决这一差距，我们引入了TELEVAL，一个大规模的中文口语交互SLM基准测试，适用于无指令、音频条件下的场景。TELEVAL评估两个互补方面：（1）可靠内容满足度，衡量SLMs在不同声学和语言条件下的语义准确性；（2）交互适当性，评估模型是否通过隐式地将行为基于听觉线索来产生自然且适当的回应。实验表明，尽管模型在语义任务上表现良好，但在声学变异和交互设置下其性能会下降。我们观察到...

    arXiv:2507.18061v4 Announce Type: replace-cross  Abstract: Spoken Language Models (SLMs) are expected to support natural spoken interaction beyond task completion. However, existing SLM benchmarks primarily evaluate semantic correctness in structured settings and provide limited assessment of interactional behavior grounded in acoustic context. To address this gap, we introduce TELEVAL, a large-scale SLM benchmark for Chinese spoken interaction in instruction-free, audio-conditioned settings. TELEVAL evaluates two complementary aspects: (1) Reliable Content Fulfillment, which measures semantic accuracy of SLMs under diverse acoustic and linguistic conditions, and (2) Interactional Appropriateness, which assesses whether models produce natural and appropriate responses by implicitly grounding behavior in auditory cues. Experiments show that while models perform competitively on semantic tasks, their performance degrades under acoustic variability and in interactional settings. We observ
    
[^394]: 大型计算机科学课程中会话式AI支持学生参与度的准实验研究

    A Quasi-experimental Study of Conversational AI-supported Engagement in Large CS Courses

    [https://arxiv.org/abs/2506.18941](https://arxiv.org/abs/2506.18941)

    本研究通过准实验设计发现，在大型软件工程课堂中使用基于大语言模型的会话式AI工具能够有效提升学生对学习材料的参与度。

    

    有吸引力的职业前景和创造性机会常常吸引学生选修计算机科学课程，从而导致班级规模庞大。大型课堂面临的一个常见挑战是学生与教师以及学习材料之间缺乏互动参与。大语言模型（LLM）的最新进展为利用会话式人工智能（CAI）应对这一挑战提供了契机。为检验CAI在支持学习材料参与方面的潜力，我们在一门大型软件工程课程中设计了一项课堂活动，让学生与CAI工具进行交互。我们在美国一所大型公立大学的真实课堂中采用准实验的被试内研究设计，将使用CAI进行讨论的课堂活动与不使用CAI的同伴讨论活动中学生的参与度进行了对比。为了进行分析，我们收集了学生

    arXiv:2506.18941v2 Announce Type: replace-cross  Abstract: Lucrative career prospects and creative opportunities often attract students to enroll in computer science courses, resulting in large class sizes. A common challenge in large classrooms is the lack of engagement between students and both the instructor and the learning material. Recent advances in Large Language Models (LLMs) present an opportunity to leverage Conversational Artificial Intelligence (CAI) to address this challenge. To examine the potential of CAI to support engagement with learning material, we designed an in class activity in a large Software Engineering course in which students interacted with a CAI tool. Using a quasi-experimental, within-subject study design in a real classroom at a large public United States University. We compared student engagement during an activity that involved the use of CAI for discussion with an activity that did not (peer discussion). For our analysis, we collected data on student
    
[^395]: 机制可解释性需要哲学

    Mechanistic Interpretability Needs Philosophy

    [https://arxiv.org/abs/2506.18852](https://arxiv.org/abs/2506.18852)

    该论文主张机制可解释性研究需要与哲学建立持续的合作关系，借助哲学澄清概念、完善方法并应对认识论与伦理层面的复杂性，从而释放该领域尚未实现的巨大发展潜力。

    

    机制可解释性旨在通过揭示训练后的神经网络背后的底层机制来解释其工作原理。随着该领域影响力的不断增长，不仅审视模型本身，审视机制可解释性研究中隐含的假设、概念和解释策略也变得日益重要。我们认为，机制可解释性需要哲学作为持续合作的伙伴，以澄清其概念、完善其方法，并应对解释人工智能系统时涉及的认识论和伦理复杂性。通过与哲学家和哲学框架的深入接触，机制可解释性的进步仍具有巨大的未开发潜力。本文以机制可解释性文献中的三个开放性问题为例，展示了哲学能够为机制可解释性研究带来的价值，并勾勒出一条通向更深入跨学科对话的路径。

    arXiv:2506.18852v3 Announce Type: replace-cross  Abstract: Mechanistic interpretability (MI) aims to explain how trained neural networks work by uncovering their underlying mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy as an ongoing partner in clarifying its concepts, refining its methods, and navigating the epistemic and ethical complexities of interpreting AI systems. There is significant unrealised potential for progress in MI to be gained through deeper engagement with philosophers and philosophical frameworks. Taking three open problems from the MI literature as examples, this paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue.
    
[^396]: AudioTurbo：基于校正扩散的快速文本到音频生成

    AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion

    [https://arxiv.org/abs/2505.22106](https://arxiv.org/abs/2505.22106)

    提出AudioTurbo，将校正扩散集成到预训练扩散式TTA模型中，通过从确定性噪声-样本对学习一阶近似ODE路径，在生成质量和推理速度上均超越基于流匹配的加速方法。

    

    扩散模型显著提升了文本到音频（TTA）生成的质量和多样性，但受限于缓慢的推理速度。近年来，研究者探索了基于流的方法来加速TTA生成。然而，这些方法通常依赖于专用的基于流的生成框架，无法直接利用现有的预训练扩散式TTA模型。为了解决这一局限性，同时充分利用预训练扩散模型的强大生成能力，我们将校正扩散集成到这些模型中，以提高TTA生成效率。具体而言，我们提出了AudioTurbo，它从预训练TTA模型生成的确定性噪声-样本对中学习一阶近似常微分方程（ODE）路径。实验表明，我们的方法在生成质量和推理速度方面均优于基于流匹配的加速模型。

    arXiv:2505.22106v2 Announce Type: replace-cross  Abstract: Diffusion models have significantly improved the quality and diversity of text-to-audio (TTA) generation but are hindered by slow inference speed. Recent flow-based approaches have been explored for accelerating TTA generation. However, these approaches typically rely on dedicated flow-based generative frameworks and do not directly exploit existing pre-trained diffusion-based TTA models. To address this limitation while leveraging the strong generative capabilities of pre-trained diffusion models, we integrate rectified diffusion into these models to improve TTA generation efficiency. Specifically, we propose AudioTurbo, which learns first-order approximate ordinary differential equation (ODE) paths from deterministic noise-sample pairs generated by a pre-trained TTA model. Experiments show that our approach outperforms a flow-matching-based acceleration model in both generation quality and inference speed.
    
[^397]: 基于大语言模型的物联网生态系统威胁检测与防御框架

    LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems

    [https://arxiv.org/abs/2505.00240](https://arxiv.org/abs/2505.00240)

    该论文提出了一种基于大语言模型的物联网威胁检测与防御框架，通过在 IoT-23 和 TON_IoT 数据集上微调的检测模型识别异常网络行为，并结合自适应防御策略生成组件，在边缘-云实验环境中实现了 99.75% 的威胁检测准确率和 98.3% 的防御准确率。

    

    物联网（IoT）的快速扩张带来了复杂且不断演变的网络安全挑战。本文提出了一种基于大语言模型（LLM）的物联网网络智能威胁检测与防御框架。所提出的架构集成了两个核心组件：检测组件利用在物联网专用数据集（IoT-23 和 TON_IoT）上微调的模型来识别异常网络行为，防御组件则生成自适应响应策略以缓解检测到的威胁。该框架包含提示管理、局部适应和全局优化模块，以支持上下文感知的决策制定和持续的模型引导适应。在基于 Docker 的边缘-云环境中进行的实验验证表明，BERT Small 在威胁检测中达到了 99.75% 的准确率、F1 分数、精确率和召回率，而防御 LLM 达到了 98.3% 的准确率。

    arXiv:2505.00240v3 Announce Type: replace-cross  Abstract: The rapid expansion of the Internet of Things (IoT) has introduced complex and evolving cybersecurity challenges. This paper presents a Large Language Model (LLM)-based framework for intelligent threat detection and prevention in IoT networks. The proposed architecture integrates two core components: a Detection Component that leverages fine-tuned models trained on IoT-specific datasets (IoT-23 and TON_IoT) to identify anomalous network behaviours, and a Prevention Component that generates adaptive response strategies to mitigate detected threats. The framework incorporates prompt management, local adaptation, and global refinement modules to support context-aware decision-making and continuous model-guided adaptation. Experimental validation in a Docker-based edge-cloud environment shows that BERT Small achieves 99.75% accuracy, F1-score, precision, and recall for threat detection, while the Prevention LLM achieves 98.3% accur
    
[^398]: CP-Guard+：协同感知中恶意代理检测与防御的新范式

    CP-Guard+: A New Paradigm for Malicious Agent Detection and Defense in Collaborative Perception

    [https://arxiv.org/abs/2502.07807](https://arxiv.org/abs/2502.07807)

    该论文提出了一种无需验证最终感知结果、直接在特征层面检测恶意代理的新范式，并发布了首个协同感知恶意代理检测基准数据集CP-GuardBench及鲁棒防御方法CP-Guard+，在大幅降低计算开销的同时有效防御协同感知中的恶意攻击。

    

    协同感知（CP）是实现安全网联自动驾驶的一种有前景的方法，它使多辆车辆能够共享感知信息以提升感知性能。然而，与单车感知相比，协同感知系统的开放性使其更容易受到恶意攻击，攻击者可以注入恶意信息来误导自车的感知，从而给安全驾驶带来严重风险。为了缓解这种脆弱性，我们首先提出了一种新的恶意代理检测范式，该范式能够在特征层面有效识别恶意代理，而无需验证最终感知结果，从而显著降低计算开销。基于这一范式，我们推出了CP-GuardBench，这是首个用于训练和评估协同感知系统中各类恶意代理检测方法的综合性数据集。此外，我们还开发了一种名为CP-Guard+的鲁棒防御方法……（摘要原文在此处截断）

    arXiv:2502.07807v2 Announce Type: replace-cross  Abstract: Collaborative perception (CP) is a promising method for safe connected and autonomous driving, which enables multiple vehicles to share sensing information to enhance perception performance. However, compared with single-vehicle perception, the openness of a CP system makes it more vulnerable to malicious attacks that can inject malicious information to mislead the perception of an ego vehicle, resulting in severe risks for safe driving. To mitigate such vulnerability, we first propose a new paradigm for malicious agent detection that effectively identifies malicious agents at the feature level without requiring verification of final perception results, significantly reducing computational overhead. Building on this paradigm, we introduce CP-GuardBench, the first comprehensive dataset provided to train and evaluate various malicious agent detection methods for CP systems. Furthermore, we develop a robust defense method called C
    
[^399]: FlexTSF：一种面向具有可变规律性的时间序列的灵活预测模型

    FlexTSF: A Flexible Forecasting Model for Time Series with Variable Regularities

    [https://arxiv.org/abs/2410.23160](https://arxiv.org/abs/2410.23160)

    FlexTSF 提出了一种灵活的领域自适应时间序列预测模型，其核心的 IVP Patcher 连续时间分块模块基于初值问题，能够原生处理不均匀采样间隔、可变序列长度和缺失值，从而有效应对不规则时间序列的通用预测挑战。

    

    对于通用时间序列预测模型而言，在具有不规则时间结构的时间序列上进行泛化是一个长期的挑战。现有方法通常假设采样是规律的，或严重依赖插补技术，这限制了它们在现实场景中的适用性——由于多样化的传感设备和记录方式，不规则性在实际场景中普遍存在。为了填补这一空白，我们提出了 FlexTSF，这是一种灵活且领域自适应的预测模型，专为具有可变时间规律性的时间序列而设计。其基础是 IVP Patcher，这是一个利用初值问题（IVPs）的连续时间分块模块，能够从本质上处理不均匀的采样间隔、可变的序列长度以及缺失值。FlexTSF 构建于仅解码器的 Transformer 架构之上，通过专门的因果自注意力机制整合了归一化的时间戳输入和特定领域的统计描述符（注：原文摘要到此被截断）。

    arXiv:2410.23160v3 Announce Type: replace-cross  Abstract: Generalizing across time series with irregular temporal structures poses a persistent challenge for universal time series forecasting models. Existing approaches often assume regular sampling or depend heavily on imputation, which limits their applicability in real-world scenarios where irregularities are prevalent due to diverse sensing devices and recording practices. To address this gap, we propose FlexTSF, a flexible and domain-adaptive forecasting model specifically designed for time series with variable temporal regularities. At its foundation lies the IVP Patcher, a continuous-time patching module leveraging Initial Value Problems (IVPs) to inherently handle uneven sampling intervals, variable sequence lengths, and missing values. Built upon a decoder-only Transformer architecture, FlexTSF integrates normalized timestamp inputs and domain-specific statistical descriptors through a specialized causal self-attention mechan
    
[^400]: 集合逻辑

    Ensemble Logic

    [https://arxiv.org/abs/2408.14443](https://arxiv.org/abs/2408.14443)

    本文提出集合逻辑（EL）这一族模态逻辑，在度量幺半群上统一了离散、稠密和多维设置下的时间与空间推理，并证明了自然数上可满足性的尖锐不可判定性边界。

    

    集合逻辑是一族模态逻辑，其模态参数是含量化变量的项。我们在度量幺半群（即配备平移不变度量的交换幺半群）上解释公式。位移算子在平移后的点上对公式求值，而有界存在和有界全称算子在指定长度的窗口内的某个点或每个点上对公式求值。对点和长度的量化允许同一变量出现在多个算子的参数中。这为离散、稠密和多维设置中的时间推理和空间推理提供了一个统一的框架。我们的主要结果建立了自然数上可满足性的尖锐不可判定性边界：对于形如 $\exists N\,\exists n\,\Psi$ 的公式，其中 $\Psi$ 由固定字母表中的字母通过合取、析取、位移和有界全称量词构造而成，其可满足性已经是不可判定的。

    arXiv:2408.14443v3 Announce Type: replace-cross  Abstract: Ensemble Logic (EL) is a family of modal logics whose modal parameters are terms with quantified variables. We interpret formulas over metric monoids, that is, commutative monoids equipped with translation-invariant metrics. Displacement evaluates a formula at a translated point, while bounded existence and bounded universality evaluate it at some or every point of a window of prescribed length. Quantification over points and lengths allows the same variable to occur in the parameters of several operators. This provides a common framework for temporal and spatial reasoning in discrete, dense and multi-dimensional settings. Our main result establishes a sharp undecidability boundary for satisfiability over the natural numbers: satisfiability is undecidable already for formulas $\exists N\,\exists n\,\Psi$ in which $\Psi$ is built from the letters of a fixed alphabet by conjunction, disjunction, displacement and bounded universal
    

