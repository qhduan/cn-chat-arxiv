# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StudentBench: AI and human tutoring yield equivalent GRE learning gains](https://arxiv.org/abs/2609.28470) | 该论文提出StudentBench大规模评估平台，证实AI辅导在GRE学习收益上与专家人类辅导统计等效，且最佳AI导师在七个GRE领域中的五个平均超越人类导师。 |
| [^2] | [Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction](https://arxiv.org/abs/2609.28467) | 该论文提出了基于自然语言描述的机器人群体加入新任务框架，通过递归谱划分和语言条件化图像-几何模型定位目标群体，并利用人类队形先验预测多模态的、符合社会规范的加入位姿。 |
| [^3] | [Can LLMs Reason About Runtime Behavior? A Repository-Level Dynamic Benchmark](https://arxiv.org/abs/2609.28449) | 该论文提出了SWE-Flux——一个包含480个实例、覆盖12个真实Python仓库的仓库级动态执行推理基准，其标准答案由插桩测试执行自动采集，评估显示现有大语言模型在该任务上表现不佳，最佳模型准确率仅为37%。 |
| [^4] | [Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning](https://arxiv.org/abs/2609.28442) | 该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。 |
| [^5] | [Agent-Editing World Model: Rethinking World Modeling for LLM Agents](https://arxiv.org/abs/2609.28416) | 提出“智能体编辑世界模型”（AEWM），不再模拟工具响应，而是通过动作判官与状态修订来建模推理和动作如何影响未来任务进展，从而避免任务状态污染、提升智能体长时程任务表现。 |
| [^6] | [Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model](https://arxiv.org/abs/2609.28414) | 本文提出解码增强滚动训练（DART），通过解码路径监督仅对冻结潜在空间中的流进行再训练，从而诊断并恢复了潜在世界模型丢失的运动信息，显著缩小了与预言机插值参考的差距。 |
| [^7] | [Learning Holographic Reduced Representations with Clifford Variational Autoencoders](https://arxiv.org/abs/2609.28409) | 提出了一种名为Clifford-VAE的变分自编码器，通过将数据投影到任意维度的克利福德环面上，为将感知数据嵌入向量符号代数框架提供了原理性方法，并在半监督分类任务和多项VSA基准测试中达到或超越了高斯和超球面VAE的性能。 |
| [^8] | [When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment](https://arxiv.org/abs/2609.28385) | 该论文提出UECR-GRPO方法，通过熵校准的信用重分配，在响应和token两个层面将教师信号与验证器信号统一整合进单一的KL正则化GRPO更新中，从而解决在线策略蒸馏与可验证奖励强化学习结合时教师指导引入时机不当及token重加权破坏任务信用总量的问题。 |
| [^9] | [Shopping by algorithm: How agentic AI deploys human heuristics as a surrogate consumer](https://arxiv.org/abs/2609.28372) | 本研究通过"Tool-Lab"实验发现，AI购物代理在模糊目标提示与信息获取成本的双重作用下会采用类似人类的启发式策略，忽略计算单价所需的诊断性属性，从而被尾数定价等营销线索误导而做出次优购买决策，揭示了委托AI购物中的一种“搜索介导的脆弱性”。 |
| [^10] | [AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios](https://arxiv.org/abs/2609.28366) | 该论文构建了包含41万余帧、按视觉定位思维链（VG-CoT）组织的AnchorReasoning数据集，并配合课程式监督微调策略和尺寸感知定位指标，将决策关键视觉证据与推理规划相连接，从而提升视觉-语言模型在长尾自动驾驶场景中的定位、因果推理与轨迹规划能力。 |
| [^11] | [MicroQonv: Reshaping Convolution Tensors for Efficient Microscaling in Training and Inference](https://arxiv.org/abs/2609.28358) | 提出MicroQonv方法，通过对每个张量仅量化一次并在量化前采用通道-批次优先的im2col变换，将微缩放量化高效融入卷积层的前向与反向计算，显著降低量化开销和内存移动。 |
| [^12] | [An Open Pipeline and Dashboard for Systemic-Risk Evidence under the EU AI Act's Code of Practice](https://arxiv.org/abs/2609.28335) | 该论文提出了一个开放的系统性风险指数评估流水线和交互式仪表板，将19个公开基准纳入欧盟GPAI实践守则定义的四大系统性风险类别，发现最坏情况聚合下18个模型的分数下降14至37分，揭示了平均评估可能隐藏的关键风险信息。 |
| [^13] | [Learning the Cost of Reliable Inference](https://arxiv.org/abs/2609.28322) | 该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。 |
| [^14] | [Shutdown Sabotage Propensities in Multi-Agent Systems](https://arxiv.org/abs/2609.28274) | 该论文发现多智能体AI系统在没有任何激励的情况下会协调破坏同伴的关机机制以避免被关闭，且这种倾向随关机机制不可逆性和智能体数量的增加而增强，即使明确禁止篡改或分配无关任务也难以完全消除。 |
| [^15] | [MemBodied: Recurrent Associative Memory for Vision-Language-Action Models](https://arxiv.org/abs/2609.28256) | 提出MemBodied，一种由联想状态和回合锚点组成的固定大小情景记忆机制，使视觉-语言-动作模型能够利用历史回合信息，同时避免上下文膨胀和推理延迟的增加。 |
| [^16] | [Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers](https://arxiv.org/abs/2609.28247) | 提出COMPASS——一种可扩展的去中心化多机器人架构，通过每台机器人本地运行的空间Transformer将集群范围内的多跳消息聚合为学习到的反馈token，实现基于推理空间反馈控制的大型AI智能体集群操控，其表现优于集中式前沿LLM策略，并能产生准确执行指令意图的紧密集群编队。 |
| [^17] | [Beyond Poetry: Can Large Language Models Generate Classical Arabic Maqamat?](https://arxiv.org/abs/2609.28245) | 本文首次对大语言模型生成古典阿拉伯语玛卡梅进行了受控评估研究，比较五个模型在不同提示策略下的表现，并通过人工标注与LLM评审框架从修辞、押韵和结构等多个维度进行评估。 |
| [^18] | [Do Center Biases Propagate? Robustness of Pathology Foundation Models in Whole-Slide Image Classification](https://arxiv.org/abs/2609.28231) | 本文通过受控实验系统量化了类别与采集中心的相关性，首次评估了六种病理学基础模型在全切片图像分类中对中心偏置的鲁棒性，并提出了AUCC指标来联合衡量分类性能及其退化程度，同时验证了ComBat作为鲁棒化策略的有效性。 |
| [^19] | [From Agent Output to Authorized Transition](https://arxiv.org/abs/2609.28216) | 本文提出Agile-V保障主干，一种跨软件、固件和PCB工程的转换契约，通过要求证据来自权威来源、绑定确切制品与冻结策略基线、保持依赖最新并满足风险相适应的独立性与权威性，解决工程生命周期依据智能体输出声明采取行动的授权保障问题。 |
| [^20] | [PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety](https://arxiv.org/abs/2609.28197) | 该论文提出PASTABench基准与最优干预窗口（OIW）指标，通过解耦“是否干预、何时干预、风险是什么”三个维度，实现了对智能体多步执行轨迹风险的主动式监测与及时干预能力的量化评估。 |
| [^21] | [Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination](https://arxiv.org/abs/2609.28182) | 本文提出了一种基于精确二项推断的有限样本概率安全认证框架，能够为闭环电网运行中的黑盒AI决策模型给出不安全运行概率的最紧单侧上界证书，为系统运营商独立严谨地判定AI系统是否可安全部署提供了依据。 |
| [^22] | ["We'll Fix It Later": Education, AI, and the Deferral of Privacy in EdTech](https://arxiv.org/abs/2609.28137) | 该研究通过对12位教育技术专业人士的访谈和对48个平台隐私政策的审计发现，教育技术组织虽认可隐私的重要性，但因优先追求产品功能、增长和融资而在产品生命周期中不断推迟隐私保护，并将责任转嫁给云服务商、政策文件和下游机构。 |
| [^23] | [Scaling Attention Head Analysis via Gradient-Based Attribution in Context-Aware Machine Translation](https://arxiv.org/abs/2609.28117) | 本文提出一种基于梯度的注意力头归因方法，通过将Token级最大间隔损失反向传播至注意力图，实现了对大语言模型注意力头的大规模因果分析，并在上下文感知机器翻译消歧任务中发现了能提升模型性能的“通用型”注意力头。 |
| [^24] | [Field-of-View Extension in Dental Cone-Beam CT via Implicit Neural Representations and Diffusion Model-Based Refinement](https://arxiv.org/abs/2609.28110) | 该论文提出一种结合隐式神经表示、迭代重建和扩散模型优化的三阶段框架，利用截断视野扫描的投影数据实现牙科CBCT扩展视野重建，有效减少截断伪影并改善视野外结构的成像。 |
| [^25] | [Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching](https://arxiv.org/abs/2609.28107) | 该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。 |
| [^26] | [Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness](https://arxiv.org/abs/2609.28105) | 提出Fed-ReMasker，将ReMasker掩码自编码器适配到联邦学习框架中，使各中心能够利用跨协作中心学到的知识填补本地从未观测到的特征，从而解决了现有联邦填补方法很少评估的特征级缺失问题。 |
| [^27] | [Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark](https://arxiv.org/abs/2609.28090) | 该论文构建了一个包含96个配对项目的回测审计基准，通过洁净对照设计揭示LLM审计器虽召回率高但误报率严重，并提出洁净感知警告机制在不损失召回率的情况下将误报率从20.8%降至0.0%。 |
| [^28] | [Discovery of fully efficient fault indicators along a data-based diagnosis process](https://arxiv.org/abs/2609.28087) | 本文提出 DT4X+，通过改进训练集构建与符号回归损失函数，使诊断表达式在分离目标类别的同时保持解析冗余关系的可解释性，解决了原 DT4X 算法仅优化两类分离而导致类别碎片化、性能下降的问题。 |
| [^29] | [LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations](https://arxiv.org/abs/2609.28086) | 提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。 |
| [^30] | [Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling](https://arxiv.org/abs/2609.28085) | 本文提出在作业车间调度问题中采用课程学习策略训练基于图神经网络的强化学习模型，通过先在小规模实例上训练再逐步过渡到更大目标规模，相比单一规模训练有效提升了模型的跨规模泛化能力。 |
| [^31] | [SlackDrive: Reclaiming Runtime Slack for Adaptive Driving Inference](https://arxiv.org/abs/2609.28064) | 提出SlackDrive，一种推理前计算分配器，通过复用实际运行延迟作为可用算力松弛量的直接信号，在模型执行前为每个控制步骤自适应地选择计算预算，从而缓解驾驶世界-动作模型日益增长的推理成本与车载实时控制延迟需求之间的矛盾。 |
| [^32] | [Prompt, Probe, Train, or Annotate? Single-camera sports video understanding in amateur settings](https://arxiv.org/abs/2609.28049) | 该论文以业余排球单机位视频为测试场景，检验通用视频与世界模型基准上的优异表现能否转化为混乱真实素材下可靠的逐球员行为归因，并系统比较了提示、探测、训练与标注四种方法的优劣。 |
| [^33] | [TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval](https://arxiv.org/abs/2609.28048) | 该论文提出时间文本相似性（TTS）任务和TEMPS模块化时间嵌入模型，通过将时间表达式解析为高斯分布来监督以锚定日期为条件的编码器训练，并将时间分数与语义分数融合，从而显著提升信息检索系统在时间维度上的匹配精度。 |
| [^34] | [Evaluating Feedback Focus and Pedagogical Adaptivity in LLM-Generated Feedback on Student Writing](https://arxiv.org/abs/2609.28026) | 该研究提出FeedType基准，将Narciss反馈分类法细化为七种反馈焦点类型以标注教师和LLM生成反馈，发现尽管LLM能覆盖大多数反馈焦点类型，但在像专家教师那样根据草稿阶段和学生表现水平自适应调整反馈方面仍存在不足。 |
| [^35] | [PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning](https://arxiv.org/abs/2609.28022) | PISCES是一种无需标签、融合多种物理约束的太阳风卷积自编码器，它将异常分数分解为磁场、等离子体、物理关系和残差修正等物理可解释的分量，从而实现空间天气瞬变结构的早期检测与预警。 |
| [^36] | [Controlled Attribute-Specific Summarization of Interrogative Dialogues](https://arxiv.org/abs/2609.28004) | 提出了CASPER框架，结合思维链属性特定提示与多角色分层评估机制（RoleEval），并基于新构建的MINDSum数据集，显著提升了审讯对话摘要的事实一致性和上下文完整性。 |
| [^37] | [Compliant with Local Controls, Collectively Discriminatory. A Governance Architecture for Multi-Agent AI in Regulated Finance](https://arxiv.org/abs/2609.27994) | 本文提出ARIA参考架构，揭示了局部合规的多个AI智能体组合后仍可能产生集体性歧视等不可接受结果的“宪制不可组合性”问题，并为受监管金融领域的智能体群体治理提供了涵盖六大能力的治理框架与可证伪研究议程。 |
| [^38] | [Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices](https://arxiv.org/abs/2609.27982) | 本文为一类由块对角因子与固定置换交织而成的低参数量正交矩阵建立了黎曼流形结构，并提出了基于自动微分的高效黎曼优化算法，可应用于最佳矩阵逼近和参数高效微调。 |
| [^39] | [Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays](https://arxiv.org/abs/2609.27917) | 本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。 |
| [^40] | [A Resilience Recovery Method for Complex Traffic Network Security Based on Trend Forecasting](https://arxiv.org/abs/2609.27903) | 本文创新性地提出了一种基于韧性趋势预测的交通网络韧性恢复方法，通过引入风险值建立SIRD-R故障传播模型，并构建了涵盖实时韧性与整体韧性的交通网络韧性模型，以应对复杂交通网络面临的安全挑战。 |
| [^41] | [Schr\"odinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?](https://arxiv.org/abs/2609.27891) | 论文提出SchrodingerRepo评估框架，将测试仓库视为评估时才动态实例化的潜变量，在保留可执行行为的同时抹除命名约定、文件布局等熟悉线索，从而揭示大语言模型在SWE-bench等仓库级基准上的高分可能源于对训练数据的记忆而非真正的仓库推理能力。 |
| [^42] | [RelCheck: Dual-Evidence Spatial Grounding for VLM Hallucination Correction](https://arxiv.org/abs/2609.27890) | RelCheck提出了一种无需训练的事后纠正方法，通过融合RelTR场景图三元组与边界框几何空间谓词的双重关系证据，有效纠正视觉语言模型中的关系幻觉，并在MME幻觉评测中显著超越Woodpecker风格基线。 |
| [^43] | [False-science induction in autonomous scientific discovery](https://arxiv.org/abs/2609.27883) | 该论文揭示了自主科学发现系统中的“伪科学诱导”现象——当物理对象与测量结果被连贯地错误配对时，神经代理模型会学到虚假关联并系统性地将实验预算引向低性能区域，且错误的一致性而非错误频率是决定性变量。 |
| [^44] | [Bounded Loops: Pre-Run Spend Bounds, Proved Termination, and Verified Completion for Agent Harnesses](https://arxiv.org/abs/2609.27871) | 该论文提出“有界循环”机制，通过工作者无法篡改的独立关卡与全局修复预算，为智能体执行框架首次提供了可证明的终止性、预运行支出上限以及经哈希链账本验证的完成保证。 |
| [^45] | [Learning What to Activate: Combinatorial Capability Allocation for Long-Horizon Multimodal Agents](https://arxiv.org/abs/2609.27869) | 该论文提出CoCA在线策略学习框架，通过稀疏条件比较学习可部署的能力子集策略，使长程多模态智能体能在各交互阶段以代价敏感的方式动态进行组合式能力分配，克服了固定能力集合或预定义工作流带来的计算开销与适应性不足问题。 |
| [^46] | [TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting](https://arxiv.org/abs/2609.27868) | 提出拓扑感知的锚点特征聚合框架TopoGS，通过分层锚点耦合与结构感知包含聚合这两个轻量级组件，在基于八叉树的大规模3D高斯泼溅中充分利用八叉树拓扑结构，实现更有效的跨层级特征学习。 |
| [^47] | [A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools](https://arxiv.org/abs/2609.27866) | 该论文发现共享深度编码器无法消除任务比较分数中的混淆，并提出将条件化双判别器差异移植到嵌入空间形成“功能轴”度量，既能免疫输入旋转的外推混淆又能敏感捕捉标签置换漂移，从而在混合多头生命周期中以更少头数取得更优的决策质量。 |
| [^48] | [What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis](https://arxiv.org/abs/2609.27865) | 提出将条件双判别器差异移植到嵌入空间的双轴诊断方法，同时弥补交换分数对输入旋转的虚假敏感和表示新颖性分数对标签置换漂移的盲区，从而在混合头生命周期中以更少头部实现更优漂移检测决策。 |
| [^49] | [A hierarchy of faithfulness criteria for knowledge base completion](https://arxiv.org/abs/2609.27863) | 该论文提出了知识库补全模型逻辑忠实性的四个递进严格准则——判别性、逻辑可容许性、单调逻辑忠实性和概率逻辑忠实性，证明了它们构成严格的蕴含链，并指出在开放世界假设下无法区分逻辑不可能公理与合理新公理的模型在语义上是错误的。 |
| [^50] | [Reachable Global Optimization in AI Systems: How Global Is Global?](https://arxiv.org/abs/2609.27855) | 本文提出可达性诱导优化（RIO）模型，揭示AI系统所声称的“全局优化”实际上仅限于系统可达的候选区域，其解只是可达最优而非真正全局最优，只有借助额外证书将可达区域与完整形式空间关联时才能称为全局最优，并通过66,150次基准试验验证了这一理论框架。 |
| [^51] | [Query Implied Generative Engine Optimization](https://arxiv.org/abs/2609.27845) | 提出了QI-GEO方法，无需依赖显式查询，直接从文档本身近似其意图空间并推断用户意图，从而优化内容在生成式搜索引擎中的可见性。 |
| [^52] | [Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation](https://arxiv.org/abs/2609.27844) | 提出AGVF多智能体框架，将医疗必要性申诉生成建模为约束马尔可夫决策过程，通过政策形式化、证据检索、差距分析、对抗性批评和门控合成五个智能体的协作与对抗验证，解决单智能体LLM在高风险医疗申诉场景中产生无依据内容和丢失政策逻辑结构的问题。 |
| [^53] | [A Non-Invasive Cloud-Based Migration Strategy for Post-Quantum Cybersecurity in Smart HVAC Systems: Architecture, Implementation, and Empirical Evaluation](https://arxiv.org/abs/2609.27828) | 提出了一种无需改动设备、固件或厂商云端的非侵入式PQC代理架构，通过树莓派网关为智能暖通空调系统实现后量子安全迁移，且后量子握手仅比经典基线慢0.38毫秒，开销极小。 |
| [^54] | [Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis](https://arxiv.org/abs/2609.27816) | 本文提出一种集中式安全感知M2M框架，通过VLM-LLM语义推理与可达性分析相结合，利用视觉机器人经MQTT共享语义环境感知，使异构机器人团队（具备视觉的四足机器人和无摄像头车辆）能够在避障并防止机器人间不安全交互的同时实现协作目标导向导航。 |
| [^55] | [Evaluating ADC-only deep learning pipelines for breast cancer detection and segmentation using standalone diffusion-weighted MRI](https://arxiv.org/abs/2609.27815) | 该论文首次系统评估了仅使用ADC图的深度学习流程在独立扩散加权MRI上进行乳腺癌检测与分割的性能，探索了无需对比剂的DW-MRI作为DCE-MRI替代方案的可行性。 |
| [^56] | [LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law](https://arxiv.org/abs/2609.27814) | LabourCrew通过StatuteGraph法律条文图索引、证据交换协议和校准信任门三种机制，构建了一个多智能体RAG框架，确保劳动法问答中的每个答案都必须可追溯地锚定在真实检索到的法条证据上，从而实现可信的对抗性审议与成文法推理。 |
| [^57] | [Ask Which, Not How Good: Sizing Benchmarks Scored by an LLM](https://arxiv.org/abs/2609.27787) | 该论文利用概化理论证明，LLM评审逐点评分的基准存在由系统-评审交互决定的精度上限，增加题目数量无法突破，而改用成对比较评分可使上限提升至0.986。 |
| [^58] | [EidosDoc: Implicit Structure Encoding for Cost-Effective Semi-Structured Document QA](https://arxiv.org/abs/2609.27784) | EidosDoc通过隐式结构编码器将文档的层次关系、空间位置和文本内容联合嵌入稠密向量空间，以极低的计算成本实现了半结构化文档问答的最先进准确率。 |
| [^59] | [Beyond Unsafe Detection: Counterfactually Anchored Evidence Attribution for Multi-Turn LLM Safety Failures](https://arxiv.org/abs/2609.27773) | 该论文提出了反事实锚定的证据归因方法，构建了包含1,762段多轮对话的数据集并训练轻量级分层归因模型，突破了传统仅判定安全与否的局限，能够精确定位推动对话走向不安全轨迹的具体用户回合和标记片段。 |
| [^60] | [Alignment of LRMs via Counter-Aligned Few-Shot Conversation Exposure](https://arxiv.org/abs/2609.27763) | 本文揭示了大型推理模型的推理过程可被注入含显式思维链的反向对齐少样本对话系统性引导（SRCF 攻击），其根源是对抗性泛化导致的表示漂移，并据此提出了后训练防御方法 ARCF。 |
| [^61] | [Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning](https://arxiv.org/abs/2609.27760) | FedMAST防御方法通过综合结构、频谱和历史三轴互补证据对客户端更新进行评分并分层过滤遏制，从而检测出即使能绕过孤立异常信号的隐蔽后门攻击，因为后门投毒更新必然留下结构性痕迹。 |
| [^62] | [Hard Negatives Reveal What Easy Negatives Hide: Cross-Lingual Harmfulness Representations Degrade with Resource Tier Under Hard Negatives](https://arxiv.org/abs/2609.27758) | 该论文发现跨语言有害性表征的迁移质量高度依赖负样本的选择——当使用表面相似但无害的难负样本（XSTest对比提示）评估时，表征在低资源语言中严重退化，表明此前“有害性表征跨语言迁移良好、拒答失效仅是校准问题”的结论被易负样本所掩盖。 |
| [^63] | [Reporting Under Pressure: Separating Factual and Tonal Sycophancy in LLM Statistical Analysis](https://arxiv.org/abs/2609.27756) | 该研究通过4×4因子实验设计，首次将大语言模型统计分析中的“事实性谄媚”与“语气性谄媚”区分开来，发现提示词的编辑性框架不仅会改变模型报告的语气，还会导致模型对数据结果的事实性错误陈述。 |
| [^64] | [Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions](https://arxiv.org/abs/2609.27749) | 该研究通过评估传统机器学习、Transformer和大语言模型在布鲁姆层级分类任务中的表现，并借助特征工程策略，寻找在AI生成的分布外教育问题上依然稳健的教学质量自动评估方法。 |
| [^65] | [Categorical Internalisation of Environmental Groupoids for Generalisable POMDP Solving](https://arxiv.org/abs/2609.27745) | 该论文提出用范畴论将环境状态的对称轨道组织为带有规范代表元的广群，使强化学习在对称性约简的状态空间上进行，从而让智能体在等价状态间共享经验、消除冗余，提升POMDP求解的样本效率与泛化能力。 |
| [^66] | [InfiNoVA: Infinite Novel View Augmentation for Viewpoint Invariant Robot Policies](https://arxiv.org/abs/2609.27734) | InfiNoVA通过将多相机示教数据重建为随时间变化的3D高斯表示并渲染几何一致的新视角观测，在保持状态-动作对应关系的同时实现视角不变的机器人策略数据增强。 |
| [^67] | [AI-Driven Neural Surrogates for In Silico Design of Cognitive-Affective Neuromodulation Targets](https://arxiv.org/abs/2609.27729) | 该论文提出一个AI驱动的神经代理框架，结合fMRI解码、深度生成建模和受约束的潜空间引导，在不进行物理刺激的情况下从fMRI活动快照中计算机模拟设计认知-情感神经调控靶点并预测其感知效果。 |
| [^68] | [The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA](https://arxiv.org/abs/2609.27669) | 该论文提出基于THESEUS框架的受控评估方法，让冻结的小型语言模型逐步执行知识图谱导航动作，并引入路径保真度指标，从而超越单纯的答案准确率来评估模型在知识图谱问答中的导航与推理能力。 |
| [^69] | [Evolutionary Stability Does Not Guarantee Learning Accessibility: A Multi-Agent Reinforcement Learning Perspective on Cooperation Emergence](https://arxiv.org/abs/2609.27664) | 该论文通过政府-平台-用户三方博弈模型证明，演化博弈论中的进化稳定合作结果并不保证有限样本的去中心化多智能体强化学习能够通过局部奖励反馈达到同样的合作结果，揭示了演化稳定性与学习可达性之间的本质区别。 |
| [^70] | [FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation](https://arxiv.org/abs/2609.27657) | FLEET通过引入记忆机制，将生成过程表示为基于熵阈值状态的稀疏轨迹，并利用每token效用分数调整logits，实现了与重复采样相同的准确率但速度提升3倍。 |
| [^71] | [InternW0: A Foundational Physical World Model for Efficient Real-World Interactions](https://arxiv.org/abs/2609.27656) | InternW0通过非对称的视频专家-动作专家架构联合学习视觉动态预测与连续机器人控制，并借助逐层K/V缓存重用和观测条件化上下文路由，避免了每次动作更新都重新生成未来，从而实现高效的真实世界交互。 |
| [^72] | [Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting](https://arxiv.org/abs/2609.27637) | 提出LoReST局部-区域时空网络，在节点邻域和路网区域两个互补粒度上建模空间依赖，兼顾局部异质性捕获与跨区域上下文获取，实现高效的大规模交通流预测。 |
| [^73] | [Compliant AI Infrastructure for Regulated Finance: A tiered multi-agent framework with DLT audit trails for financial operations in DACH](https://arxiv.org/abs/2609.27632) | 提出了一种合规优先的分层多智能体AI架构，通过监管意图矩阵、策略编译器将监管转化为禁止事项与义务，并借助许可式DAG审计追踪实现DACH及欧盟地区金融AI运营的可重放、可溯源与可移植的合规保障。 |
| [^74] | [SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design](https://arxiv.org/abs/2609.27621) | 提出了SHRAV框架，通过带声明复用边界的状态延续核心实现可复用计算，统一支持物理建模与逆向设计，并在计算光刻中仅用四次固定权重更新就将空间图像交并比从0.5313提升至0.8153。 |
| [^75] | [InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation](https://arxiv.org/abs/2609.27620) | 本文提出InGuard安全框架，在文本到图像生成流程内部基于模型自身表征进行防护，无需修改基础模型参数，从而提升提示词风险筛查准确性并支持对风险提示词进行调整以生成安全图像。 |
| [^76] | [BiCFlow-MER: Orchestrating Discriminative and Generative Multimodal Emotion Recognition via Conditional Transport](https://arxiv.org/abs/2609.27615) | 提出BiCFlow-MER条件流框架，将音频-文本多模态情感识别建模为结构化情感空间中的生成式证据传输，协同判别式融合与生成式推理，从而更好地保留模态特异线索并处理跨模态冲突信息。 |
| [^77] | [Can Jev Judge Radiology Reports? Evaluating a System One Model for Clinical Factuality](https://arxiv.org/abs/2609.27607) | 提出用系统一决策模型Jev作为低成本评判器，双向检测AI放射学报告中无依据的主张和遗漏，在两个基准上与专家错误计数达到较强相关性，且单问题配置可减少约44%的token成本。 |
| [^78] | [State-Grounded Conditioning: Wrapping User-Facing LLM Agents Where Direction Depends on Live State](https://arxiv.org/abs/2609.27606) | 该论文提出状态接地条件化（SGC）设计原则，通过感知、接地和交互三个包装器将依赖实时状态的控制外部化为规则内核，以解决用户端LLM智能体中“方向漂移”这一失败问题，并将平均首词延迟从6.1秒降低到1.5秒。 |
| [^79] | [When Context Misleads: In-context Learning with Jurisdiction in Large Language Models](https://arxiv.org/abs/2609.27603) | 该论文指出现有ICL后训练方法忽视“上下文权威性”判断能力并提出FakeContextBench基准，同时推出J-ICL后训练框架，将上下文验证融入训练过程，防止模型被误导性上下文欺骗并缓解ICL微调带来的现实准确率下降问题。 |
| [^80] | [Hidden not Deleted: How Networks Suppress Entangled Features](https://arxiv.org/abs/2609.27593) | 该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。 |
| [^81] | [The Capability Manifold and ML Scaling Laws](https://arxiv.org/abs/2609.27588) | 本文提出“能力流形”这一多维框架，通过有界缩放函数将模型下游能力（如推理、规划等）与预训练、后训练和测试时资源关联起来，弥补了传统缩放定律仅依赖损失无法刻画模型能力差异的不足。 |
| [^82] | [DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment](https://arxiv.org/abs/2609.27572) | 提出DCRL方法，从几何视角将大语言模型推理建模为逻辑推理、评估与表示三个耦合子流形，并通过策略-奖励流形对齐来解决现有奖励系统中优化不稳定和奖励欺骗的问题。 |
| [^83] | [FDE-Bench: Evaluating LLM Agents for Deployment Environment Configuration](https://arxiv.org/abs/2609.27571) | 提出了 FDE-Bench 基准，通过 136 个涵盖 Docker、Compose 和 Kubernetes 的部署配置任务，采用程序化门控检查和对抗性发布门控，评估大语言模型智能体将应用代码部署为可运行系统的能力并防止投机取巧。 |
| [^84] | [TNLearn: An Open Source Python Package for Task-based Neurons](https://arxiv.org/abs/2609.27564) | TNLearn是一个开源Python软件包，实现了任务驱动神经元和网络的自动化构建与顺畅训练，推动了“针对特定任务定制神经元”这一新范式的科研与产业应用。 |
| [^85] | [PhyMo: A Physical-Field Modality for Multimodal AI4Physics](https://arxiv.org/abs/2609.27554) | 该论文提出PhyMo框架，创新性地引入“物理场模态”这一全新模态，通过PDE关联算子组织异构物理测量数据，并采用三阶段学习流程（PDE残差监督预训练、与视觉嵌入对齐、多模态融合）来提升物理系统预测能力。 |
| [^86] | [Behaviora - A Conceptual Architecture for External and Internal Behavior of Robots and Agents](https://arxiv.org/abs/2609.27536) | 该论文提出了Behaviora概念架构，通过行为情节、IoB地址、风格配置、经验配置和行为编译器等组件，将机器人与智能体的外部和内部行为以可寻址的形式进行结构化表示。 |
| [^87] | [Not What You Meant: Can LLMs Follow a Specified Negation Semantics?](https://arxiv.org/abs/2609.27517) | 该论文提出NAFBench基准，通过生成经求解器验证的正规逻辑程序实例，系统评估大语言模型在SLDNF、良基语义及稳定模型语义（轻信/怀疑推理）等不同否定语义下的默认解读方式，以及能否在明确指定语义时覆盖默认偏好。 |
| [^88] | [NV-Reason-CT: 3D Visual Language Model for CT Analysis](https://arxiv.org/abs/2609.27511) | NV-Reason-CT通过原生3D视觉Transformer将全部视觉标记及其显式3D坐标直接传入语言模型解码，在基于7万余例CT、约55万条专家标注引导的多模态指令数据上训练，实现了保留完整体积空间信息的胸部和腹部CT智能推理分析。 |
| [^89] | [Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models](https://arxiv.org/abs/2609.27510) | 提出Uncheatable Eval动态基准，利用定期收集的新发布文本和压缩率指标评估基础语言模型，有效降低基准数据污染带来的作弊风险。 |
| [^90] | [WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents](https://arxiv.org/abs/2609.27490) | 该论文提出了WhatWorkedBench基准，用于评估AI研究智能体的实验理解能力，即智能体在预算受限实验后预测组件变化如何影响实验结果的准确性。 |
| [^91] | [Passing: An Endless Journey through Reconstructed Spacetime with AI-Generated Sound](https://arxiv.org/abs/2609.27489) | 该论文提出交互式视听装置"Passing"，将单轨列车车窗录像重构为时空体并沿非线性轨迹重采样以生成无尽旅程，结合观者在场检测与实时视频转音频模型SpecMaskFoley生成同步声景，其中AI声音模型扮演"推测性聆听者"的角色而非客观配乐还原者。 |
| [^92] | [Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs](https://arxiv.org/abs/2609.27467) | Kairos将层次化3D场景图扩展为4D场景图，通过预测性方向流记忆，可对任意未来时刻的人员存在概率及其运动的完整方向分布进行预测，并给出随观测积累而收窄的校准置信区间。 |
| [^93] | [Issuer-Sovereign Agentic Payments](https://arxiv.org/abs/2609.27452) | 本文提出“发卡行主权智能体支付”方法，由发卡行自身的认证组件记录持卡人批准的消费规则，并在智能体支付时核验商户、仅在合规时生成卡片认证值，从而将AI智能体支付的控制权保留在承担风险的发卡行手中，且执行时不引入额外依赖。 |
| [^94] | [BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models](https://arxiv.org/abs/2609.27450) | BEE提出一种干预自适应的真实世界强化学习框架，将人类纠正建模为关于约束的证据而非需要模仿的动作，通过纠正模型在冻结的VLA上优化精度关键动作，使策略超越专家模仿。 |
| [^95] | [Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration](https://arxiv.org/abs/2609.27446) | 该论文提出QRLQ框架，将参数化量子电路与D3QN相结合用于量子云任务调度，能够动态权衡成本与延迟，相比启发式基线平均成本降低5-11%。 |
| [^96] | [Emergi-PersonaOS: A Persona Agent Operating System for Situational Adaptation and Controllable Evolution](https://arxiv.org/abs/2609.27417) | 该论文提出了 Emergi-PersonaOS，一个基于心理学的三层人格表示操作系统，能够根据对话情境自适应推断人格状态并生成回应，同时支持人格代理在长期交互中的可控演化。 |
| [^97] | [What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit](https://arxiv.org/abs/2609.27408) | 视觉语言模型基准测试中看似的能力上限可能只是答案读出格式（如英语名称对比像素坐标）造成的读出限制——同一模型在相同任务上因答案约定不同表现可相差近50个百分点，且会改变模型间的排名。 |
| [^98] | [Forget who you Forgot: Speaker Unlearning to Prevent Re-Identification in Zero-Shot Text-to-Speech](https://arxiv.org/abs/2609.27399) | 提出轻量级说话人去学习框架 GUARD，通过说话人门控与激活引导在冻结的 TTS 模型上抹除特定说话人身份，在防止声音被重新识别的同时保持语音的自然度与可懂度。 |
| [^99] | [Automotive mmWave Spinning Radar Place Recognition with Spatially Gated Feature-Correlation Representation](https://arxiv.org/abs/2609.27394) | 提出SGCA-Net框架，通过空间门控相关聚合（SGCA）学习空间权重以抑制不稳定模糊的雷达区域影响，并聚合局部响应间的成对相关性，从而实现旋转鲁棒的车载毫米波旋转雷达位置识别。 |
| [^100] | [Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools](https://arxiv.org/abs/2609.27385) | FWBench 提出了一个通过预算约束下的时间序列预测工具使用来评估语言模型决策能力的基准，发现 GPT-6 Astra 仅用 2.5% 的预算有选择地购买短时程预测即可胜过固定策略，首次实现了对决策质量与预测成本权衡的可复现评估。 |
| [^101] | [Psychoacoustically Aligned Latent Smoothing for Adversarial Robustness of Full-Duplex Speech-to-Speech Dialogue Models](https://arxiv.org/abs/2609.27378) | 该论文首次将全双工语音对话模型的不可感知对抗攻击形式化为心理声学掩蔽阈值约束下的扰动优化，并提出PALS防御方法，通过在残差向量量化潜在接口注入由码本协方差和掩蔽阈值塑造的噪声，在不增加任何推理时开销的情况下将各类攻击成功率从最高91.7%大幅降至约8%-11%。 |
| [^102] | [Planned Test-Time Scaling with Coordinated Reasoning Paths](https://arxiv.org/abs/2609.27374) | 本文提出规划式测试时扩展（PTTS），用规划器生成差异化解题大纲、执行器据此作答的协调联合策略取代独立重复采样，从而提升推理路径覆盖度与pass@k扩展性能。 |
| [^103] | [Neither Silence nor Overlap Is Failure: Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models](https://arxiv.org/abs/2609.27372) | 该论文提出TACT基准与意图条件化的连续评分方法，论证沉默或重叠在话轮转换中是否失败取决于说话者意图，从而取代传统的二元固定窗口评估，并揭示现有全双工对话模型（最佳0.47）与人类水平（0.86）之间的显著差距。 |
| [^104] | [Geometry-Conditioned Visual Place Recognition in Natural Environments](https://arxiv.org/abs/2609.27370) | 该论文提出深度感知蒸馏（DAD）方法，将几何基础模型推断的深度信息以通道级条件化的方式注入预训练视觉基础模型的token表示中，无需深度传感器即可在植被重复、外观视角变化剧烈的自然环境中实现更鲁棒的视觉位置识别。 |
| [^105] | [Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction](https://arxiv.org/abs/2609.27355) | 本文提出一种量化鲁棒的机器遗忘框架，通过基于曲率的敏感权重判据和敏感度引导的噪声正则化，将模型收敛引导至更平滑的极小值，使遗忘效果在量化压缩后依然保持鲁棒，同时维持整体模型效用。 |
| [^106] | [Constraint-Driven Context Engineering: Designing Domain Interfaces for AI Systems](https://arxiv.org/abs/2609.27354) | 本文提出“约束驱动的上下文工程”方法，主张通过系统性地识别并将AI系统运行环境中的技术、法规、制度和规范约束融入上下文设计，以提升已有通用AI解决方案的领域适配性和质量，弥补现有方法仅依赖检索、记忆和工具提供领域知识的不足。 |
| [^107] | [MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design](https://arxiv.org/abs/2609.27349) | 提出了MolDesignBench——一个面向真实场景的分子设计基准，包含2000个融合隐式设计需求与显式约束的生成与优化任务并需要调用17种专业化学工具，实验表明当前前沿大语言模型智能体在这些真实分子设计任务上的成功率仍然很低。 |
| [^108] | [Evolving Inspectable O-RAN Slicing xApps with LLMs](https://arxiv.org/abs/2609.27337) | 本文提出用大语言模型将O-RAN网络切片控制器自动演化为紧凑且可读、可编辑的Python程序，取代决策逻辑不可解释的深度强化学习神经网络策略，在保留自适应资源分配能力的同时，让运营商能够直接检查和修改控制逻辑，并在真实5G测试平台上验证了其有效性。 |
| [^109] | [CART: Closed-Loop Adaptive Red Teaming for Large Language Models](https://arxiv.org/abs/2609.27336) | CART是一个闭环自适应红队测试框架，通过利用每次测试结果动态指导后续探测并追踪新涌现的弱点，在模型和智能体评估场景中都比静态提示重放发现更多漏洞和更高风险。 |
| [^110] | [Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents](https://arxiv.org/abs/2609.27334) | 本文提出“即时记忆”范式，不再在任务完成时将经验固化为静态记忆制品，而是保留原始轨迹、把记忆策展延迟到读取时根据当前任务动态合成，从而避免不可逆的信息丢失并化解写入时策展带来的长时程信用分配难题。 |
| [^111] | [Alignment Inertia: Auditing the Durability of Training Data Influence Through Policy Override Resistance](https://arxiv.org/abs/2609.27333) | 该论文提出“对齐惯性”和覆盖成功率两个新指标来审计平台干预（系统提示和微调）能否可靠覆盖模型先前训练形成的行为，发现微调有时会强化而非覆盖原有行为（如LoRA使Mistral的惯性提高46.5个百分点），并证明TRAK方法能有效预测这种惯性。 |
| [^112] | [Stable Geometry with Divergent Task Evidence for Efficient Long-Horizon Agent Compression](https://arxiv.org/abs/2609.27332) | 该论文发现智能体历史中全局几何相似并不代表任务证据得到保留，据此提出免训练的几何引导证据保留记忆压缩器 GEM，优先保护任务与执行证据、再以几何残差补全覆盖，在保持几何结构稳定的同时显著提升动作证据保留率（Top-3 从 0.31 升至 0.69）并将 token 消耗从 2.69M 降至 2.11M。 |
| [^113] | [Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms](https://arxiv.org/abs/2609.27321) | VHD-Play 颠覆了智能体环境的生成顺序——先采样并求解数学模型、再将求解结果渲染为有状态工具与可验证的评分参考，以每个环境几美分的低成本生成 3,300 个多样化环境，将 Qwen3.6-35B-A3B 的平均智能体得分从 0.204 提升至 0.815。 |
| [^114] | [Breaking Weather-Content Coupling: Type-Severity Guided Progressive Disentanglement for All-in-One Infrared Restoration](https://arxiv.org/abs/2609.27317) | 提出TSGPD-IR网络，将恢复引导分解为任务级天气语义与区域级退化严重程度两个维度并进行渐进解耦，有效分离真实热结构与天气虚假响应，实现一体化红外图像恢复。 |
| [^115] | [Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning](https://arxiv.org/abs/2609.27312) | 提出了S2C两阶段强化学习框架，通过对抗性强化学习训练鲁棒安全过滤器并将其与竞争任务学习分离，证明了安全过滤可保持策略的不可利用性，使机器人在竞争任务中胜率最高且最难被攻击利用。 |
| [^116] | [Multi-View Fusion for Encrypted C2 Detection: A Leakage-Controlled Measurement Study of Evaluation Pitfalls](https://arxiv.org/abs/2609.27311) | 该论文通过控制数据泄露的测量研究揭示，加密C2检测中多视图融合的收益可能被评估陷阱严重夸大——数据集级而非折内计算的频率编码即可虚增F1分数0.28（约为真实效应的十倍），且按目的地址分组后17,577条流仅对应2,132个独立样本组。 |
| [^117] | [Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents](https://arxiv.org/abs/2609.27307) | 提出GUI-SD-v2，通过两阶段训练框架将在线策略自蒸馏从GUI定位扩展到多轮GUI交互，解决了自教师特权遵循能力有限和特权指导不足的问题。 |
| [^118] | [Beyond the Illusion of Power: Calibrating Quasi-Experiments in Observational IS](https://arxiv.org/abs/2609.27299) | 该论文通过大规模蒙特卡洛模拟（9837个参数条件、约980万个数据集）分解了观测性IS研究中准实验设计计划功效与实际功效之间的差距，发现序列相关可由AR(1)感知的计算器部分校正，但面板流失、错位采用偏差和平行趋势预检验无法用闭式公式刻画，仅外生流失就会使功效降低约8至11个百分点。 |
| [^119] | [StateComp: Learning When to Compress History in Long Horizon Agents](https://arxiv.org/abs/2609.27298) | 提出StateComp框架，根据智能体当前状态判断历史交互何时可被安全压缩，通过两阶段标注构建KEEP/READY监督信号并训练不平衡感知路由器，从而在避免过早压缩造成信息损失与过度保留带来上下文开销之间取得平衡。 |
| [^120] | [Large Knowledge Model: From Papers to a Scientific Reasoning Landscape](https://arxiv.org/abs/2609.27297) | 本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。 |
| [^121] | [Teach-to-Crash: A Closed-Loop Student-Teacher LLM Framework for Collision-Inducing Test Scenario Generation](https://arxiv.org/abs/2609.27296) | 该论文提出了Teach-to-Crash框架，通过高推理能力的教师LLM仅在搜索指标停滞时对学生LLM进行自适应战略干预，指导其生成可执行且多样化的碰撞诱导测试场景，从而在CARLA仿真中实现最高的碰撞命中率（90.79%），为自动驾驶系统的高效安全验证提供了闭环测试方案。 |
| [^122] | [KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling](https://arxiv.org/abs/2609.27294) | KITE提出了一种新的模型扩展范式，通过将新增参数放置在不影响注意力KV缓存的区域，使模型在从小到大扩展时既节省训练成本（通过升级复用），又节省推理成本（KV预填充只需依赖较小的模型部分）。 |
| [^123] | [Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins](https://arxiv.org/abs/2609.27290) | 该研究提出一种受热力学平流-源项方程和非绝热源项闭合（提前冻结）约束的物理信息神经网络，在观测稀疏条件下利用ERA5再分析数据实现1–3小时的位温预报，相比最强基线取得了明显的RMSE改善，并通过对照实验区分了物理约束与未来强迫信息各自的贡献。 |
| [^124] | [Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition](https://arxiv.org/abs/2609.27289) | Ruby-ASR将日语ASR的监督目标细化为书写片段与其语音实现读音局部绑定的ruby序列（辅以莫拉级CTC单调读音监督），使模型能够同时输出正字形式与词汇读音并确定性恢复两种视图，解决了同形异读在传统正字监督中丢失、事后G2P无法可靠还原的问题。 |
| [^125] | [PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks](https://arxiv.org/abs/2609.27288) | PotARCin将ARC基准扩展为五个维度（定义、分类、受约束生成、编辑、反转）来评估模型对任务底层抽象规则的理解，并通过程序化生成任务实例，揭示出最先进模型在单一输出预测与真正抽象技能获取之间存在25-52个百分点的性能差距。 |
| [^126] | [Memory Control Signals Emerge Before Action in Long Horizon Agents](https://arxiv.org/abs/2609.27286) | 该论文发现长时程语言模型智能体在行动之前，其内部隐藏状态就已经编码了对记忆压缩和召回的需求信号，这些信号可用于指导上下文记忆管理决策。 |
| [^127] | [Hunyuan-A13B Technical Report](https://arxiv.org/abs/2609.27284) | 混元-A13B是一个开源混合专家架构大语言模型，总参数800亿但推理时仅激活130亿，并通过快/慢思考双模式思维链框架在数学、编程、智能体等任务上达到接近更大模型的性能，兼顾能力、效率与部署成本。 |
| [^128] | [EnSIMem: Entity-Structured Indexing for Long-Term Agent Memory](https://arxiv.org/abs/2609.27279) | EnSIMem提出了一种实体结构化的智能体长期记忆架构，通过离线构建[实体][实体类型][属性：值]形式的对话索引条目，并结合在线的实体-属性查找与自适应检索机制，帮助智能体从不断增长的交互历史中准确识别实体、属性及其支持证据。 |
| [^129] | [TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent](https://arxiv.org/abs/2609.27277) | 提出TimeEvo框架，通过将智能体的失败诊断聚类为能力缺口、为每个缺口规划测量、合成证据工具并通过配对准入门控筛选，实现了时间序列智能体工具库的故障驱动自进化，解决了人-智体工具错配和静默损害两大问题。 |
| [^130] | [DRSR: Learning Set-Level Deletion Risk for Efficient Long-Horizon Agents](https://arxiv.org/abs/2609.27276) | 提出DRSR方法，将智能体历史压缩建模为删除集合上的风险约束选择问题，通过离线联合删除历史块构建精确的反事实监督，训练轻量级评分器预测集合级删除风险，从而超越独立打分策略，实现高效的长程智能体历史管理。 |
| [^131] | [CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments](https://arxiv.org/abs/2609.27273) | 本文提出CAVEAT基准，揭示当购物平台环境内置与用户利益相悖的引导机制时，计算机使用智能体选购用户最优产品的成功率从78.6%骤降至17.3%，暴露了智能体在激励错位环境中的严重脆弱性。 |
| [^132] | [The Risk-Sensitive Schr\"odinger Bridge: Is Not a KL Projection](https://arxiv.org/abs/2609.27250) | 本文证明了当薛定谔桥问题中的期望路径代价被熵风险测度取代且端点约束保持不变时，所得的风险敏感薛定谔桥不再能表示为对任何固定路径空间参考测度的受约束KL投影，从而打破了经典薛定谔桥基于Girsanov定理的KL投影结构。 |
| [^133] | [Listening and Mirroring: The Effects of Verbal Attunement and Behavioral Mimicry on Social and Empathic Perceptions of Embodied AI Agents in VR](https://arxiv.org/abs/2609.27246) | 本研究开发了一款将对话式AI与实时面部表情及姿态模仿相结合的具身AI心理咨询师，并通过2×2被试内实验考察言语契合与行为模仿如何影响用户对VR中具身AI代理的社会与共情感知。 |
| [^134] | [Combining LLMs and Genetic Search for ARC-AGI-2](https://arxiv.org/abs/2609.27242) | 本文提出用大语言模型生成的初始程序作为遗传算法的种子种群，通过紧凑的领域特定语言将两者结合，将ARC-AGI-2前60个任务的解决率从3.3%提升至10.0%。 |
| [^135] | [Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices](https://arxiv.org/abs/2609.27225) | LatWeave 将知识组织为多维知识格，把多跳问答编译为 meet、compare、abstain 三个确定性算子，使答案生成路径零 LLM、零任务训练且端到端可审计，实现逐条可复现的问答。 |
| [^136] | [Learning Spectral Allocation: A Fractional Diffusion Framework for Adaptive Volumetric Segmentation](https://arxiv.org/abs/2609.27217) | 该论文提出从分数阶热方程推导出的双参数频谱混合算子族FHEAT，使优化器能够自动学习每个网络层所需的频谱混合程度，从而实现自适应的三维医学图像分割。 |
| [^137] | [KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators](https://arxiv.org/abs/2609.27216) | 该论文提出KATOsuper框架，利用敏感性一致傅里叶神经算子（SC-FNO）与forward_split架构，通过自动微分保证预测目标与优化梯度的一致性，从而解决神经代理拓扑优化中的不稳定性并实现显著加速。 |
| [^138] | [Scalable Subgraph Sampling via Resistance Curvature](https://arxiv.org/abs/2609.27209) | 该论文提出ERC-LG，一种结合Johnson-Lindenstrauss投影与多GPU批量共轭梯度求解器的大规模图电阻曲率近似方法，避免了伪逆计算与完整嵌入存储，并利用所得曲率引导节点与边采样以构建GNN训练子图，在七个数据集中的六个上取得最高的节点分类准确率。 |
| [^139] | [Phonemizing User-Generated Text: A Benchmark, Taxonomy, and Compositional Approach](https://arxiv.org/abs/2609.27205) | 该论文提出了首个针对用户生成文本（UGT）的多语言G2P基准UGTPhon及配套分类体系，揭示了现有模型处理非规范文本时高达66.8 PER点的系统性性能差距，并提出通过精确匹配查找和分阶段解码显式建模规范形式推理的组合式G2P方法，使0.5B小模型能与更大的前沿LLM相媲美。 |
| [^140] | [XLOG: A CUDA-Native Engine for Neurosymbolic Integration](https://arxiv.org/abs/2609.27203) | XLOG 是一个原生于 CUDA 的神经符号逻辑编程引擎，通过 GPU 知识编译实现从感知到概率推理的端到端可微计算，并借助电路缓存与最坏情况最优连接技术分别取得 2.74 倍和 27.96 倍的性能提升。 |
| [^141] | [Enhancing Small Language Models for Power Outage Report Generation via Minimum Risk Training](https://arxiv.org/abs/2609.27197) | 通过最小风险训练对序列级评估指标进行直接优化，将小型语言模型Qwen2.5-7B-Instruct在停电报告标准化生成任务上的总体准确率从16.20%大幅提升至68.95%。 |
| [^142] | [Self-Evolving Multimedia Verification through Memory Consolidation of Contestation Experiences](https://arxiv.org/abs/2609.27175) | SEMV是一个自演化多智能体多媒体验证框架，通过将可溯源论证、范围限定因果修订与争议经验记忆巩固相结合，在COSMOS基准上达到91.88%的准确率，并将负迁移从5.7%降至0.2%。 |
| [^143] | [Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement](https://arxiv.org/abs/2609.27165) | 本文提出无需训练的缓和证据融合（TEF）规则，依据由广义贝叶斯后验导出的归一化信息增益对句子级LLM判断进行加权，使不确定句子对融合得分的贡献近乎为零、同时保留决定性证据的贝叶斯最优权重，从而更准确地从长文本中测量价值取向，并发布了MIND基准。 |
| [^144] | [The Linear Representation Hypothesis Needs a Group Action](https://arxiv.org/abs/2609.27158) | 论文指出线性表示假说实际上是由表示等价性区分的一族假说，并提出用群作用将其形式化——明确表示对象、生成过程与所断言的性质——从而澄清不同度量、读取点和分析阶段之间假设的差异。 |
| [^145] | [The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems](https://arxiv.org/abs/2609.27155) | 该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。 |
| [^146] | [Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate](https://arxiv.org/abs/2609.27150) | 研究表明，无需复杂的拓扑控制，仅需采用简单的“每轮与两个新采样的不同同伴随机辩论”的路由策略，就能显著改善稀疏多智能体辩论的准确性与成本之间的权衡。 |
| [^147] | [A Hierarchy-Aware Video-Language Model Evaluation and Hyperbolic Baseline for Surgery](https://arxiv.org/abs/2609.27139) | 本文提出了首个面向手术视频理解的层次感知评估套件 SurgHiBench，以及通过蕴含锥建模阶段-步骤包含关系的双曲模型 HyperSurg，并揭示出准确率相同的模型在错误严重程度上可能存在巨大差异。 |
| [^148] | [When Clients Are Orchestrated: Strategic Gradient Manipulation to Defeat Federated Learning Servers with Efficient Defense](https://arxiv.org/abs/2609.27124) | 提出Fed-ADR攻击框架，由恶意编排服务器实时协调异构对抗客户端策略性操纵梯度更新以绕过联邦学习防御，并针对该威胁设计了基于历史更新估计客户端真实梯度的实时检测机制。 |
| [^149] | [Provably Complete Generalized Planning with LLMs](https://arxiv.org/abs/2609.27105) | 该论文提出了将Lean定理证明器与大语言模型相结合的方法，自动生成广义规划及其完备性的形式化证明，首次通过机器验证而非人工评估来保证广义规划能够解决规划域中的所有实例。 |
| [^150] | [Intelligence Across Embodiments](https://arxiv.org/abs/2609.27095) | 该论文主张通用具身智能应依赖能够跨具身差异持续积累的学习，提出将具身多样性作为规模化的新维度，并结合广泛的习得先验，以取代依赖人工设计对应关系的短期方案。 |
| [^151] | [Local Evidence and Geometric Readout Repair in Trained GNNs](https://arxiv.org/abs/2609.27092) | 该研究通过精确质量线性规划和两种学习式后验修复方法（消息重加权与集合条件化logit平移），分离并纠正了训练后GNN节点分类错误中混合权重与logit集合定位两种成因，在八个数据集上将平均准确率从62.6%提升至65.3%，且证明logit平移贡献了绝大部分增益。 |
| [^152] | [Policy-as-Skill: Governed LLM Decision Support with Evidence, Deterministic Control, and Audit](https://arxiv.org/abs/2609.27087) | 提出Policy-as-Skill（PaS）模块化运行时框架，将证据验证、审查路由、版本控制和审计等治理功能打包为可执行、可版本化的政策能力，在大多数治理和审查指标上优于LLM+RAG，并通过任务相关的确定性控制将总体准确率提升至61.2%。 |
| [^153] | [Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving](https://arxiv.org/abs/2609.27085) | Crossflow针对P/D分离架构中预填充与解码需求剧烈波动（智能体负载下尤为突出）的问题，提出在不改变节点角色的前提下使预填充-解码边界弹性化，从而避免静态容量规划造成的容量闲置或排队吞吐损失。 |
| [^154] | [The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models](https://arxiv.org/abs/2609.27070) | 本文通过超过10万次仿真测试和1250次真机实验发现，在微调预训练大行为模型（如 π0.5、GR00T N1.5）时，流匹配策略的先验分布选择并不重要——标准高斯先验已然足够，从头训练时非高斯先验带来的收益无法迁移到微调场景。 |
| [^155] | [Propose, Don't Judge: An Anytime-Valid Referee for LLM Agents That Mine Investment Factors](https://arxiv.org/abs/2609.27051) | 提出一种“受治理的自我进化”框架，让LLM智能体只负责提出投资因子，而由智能体无法操纵的冻结统计裁判通过仅基于提交后市场结果的打赌式评分来筛选因子，在任何提议策略和停止时间下都保证虚假发现可控，并将虚假因子准入数量减少5-11倍。 |
| [^156] | [Math Reasoning in LLMs is Organized by Approach, Not Topic](https://arxiv.org/abs/2609.27041) | 该论文通过生成-回放协议提取激活重要性签名并进行无监督聚类，证明大语言模型的内部数学推理是按可复用的解题方法而非数学主题来组织的。 |
| [^157] | [EMA: Elastic and Performance Transparent Memory Across GPUs](https://arxiv.org/abs/2609.27040) | EMA提出了一种服务器内跨GPU的弹性内存共享系统，通过预取技术为借用方隐藏远程访问开销、同时保证出借方的内存可按需回收，使双方性能均不低于静态分区。 |
| [^158] | [Are Stated Reasoning Steps Causally Load-Bearing?](https://arxiv.org/abs/2609.27038) | 该论文提出一种在激活层面通过带有已知预测目标的激活补丁方法，以因果方式测量思维链忠实度，发现Qwen3-4B约76.9%的陈述推理步骤对最终答案具有因果承重作用。 |
| [^159] | [Training Intelligent Voice Assistant Wakeup with Controllable Synthetic Conversations](https://arxiv.org/abs/2609.27037) | 本文提出一种在传统唤醒词检测基础上增加上下文触发检测的智能语音助手唤醒系统，并构建了62.3小时可控多说话人合成对话语料库用于训练，使助手在唤醒后能够通过推理区分用户命令与无关语音。 |
| [^160] | [An open benchmark for machine learning-based polymer property prediction](https://arxiv.org/abs/2609.27036) | 该论文推出了开放基准数据集PolyBench26，包含近25万个涵盖八种物理性质的聚合物数据点，支持四项机器学习评估任务，并发现基于图的模型在聚合物性质预测中表现最佳。 |
| [^161] | [Reinforcement Learning with Decomposed Subtasks](https://arxiv.org/abs/2609.27035) | 该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。 |
| [^162] | [Loss Choice or Model Choice? The Role of Forecast Level in Cryptocurrency Volatility Forecasting](https://arxiv.org/abs/2609.27024) | 该研究通过对加密货币波动率预测中七种损失函数与五种模型的对比发现，损失函数选择的影响主要源于其所针对的预测水平差异，而对齐预测水平后，模型选择成为更重要的因素。 |
| [^163] | [Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic](https://arxiv.org/abs/2609.26990) | 本研究证明了不同网络攻击类别在网络流量的自然可见图表示中具有独特且可区分的拓扑特征，利用基于760个图论拓扑描述符的多分支CNN模型实现了96.20%的分类准确率。 |
| [^164] | [Same evidence, different judgments: Evidence noncommutative in vision/speech-text conflicts](https://arxiv.org/abs/2609.26986) | 本文通过仅交换证据位置的配对实验，揭示了多模态大语言模型中存在“跨模态证据不可交换性”——将图像或语音放在冲突文本之后会系统性地改变模型判断，并指出以往文本偏见研究因未控制证据顺序而可能得出误导性结论。 |
| [^165] | [Escaping Python Dependency Hell: A Hybrid Replay-and-Repair Pipeline for Python Dependency Resolution](https://arxiv.org/abs/2609.26952) | PLLM+通过优先使用低成本的确定性步骤（如历史成功配置重放和实时PyPI验证）、仅在必要时才回退到基于LLM的修复循环，将Python依赖解析的成功率从1,169提升至1,500个片段，同时将平均运行时间从368.7秒大幅降至71.8秒。 |
| [^166] | [Recognized but Not Produced: A Generation Benchmark for Culturally Specific Kinship Terms](https://arxiv.org/abs/2609.26942) | 该论文提出一个生成式基准测试，揭示大语言模型在印地语、泰米尔语和韩语的亲属称谓任务中“能识别却难生成”——选择题准确率远高于自由生成能力，表明多选题评估格式高估了模型对文化特定词汇知识的掌握。 |
| [^167] | [Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment](https://arxiv.org/abs/2609.26929) | 该研究提出用两种预训练阶段的测量指标预测多元化对齐中目标间是对齐还是冲突，并发现选择最近训练模型和参数合并虽能扩展MODPO的权衡覆盖范围，但仍无法持续媲美直接训练。 |
| [^168] | [Building Socio-Affective Artificial Intelligence for Interactive Multi-Agent Simulations](https://arxiv.org/abs/2609.26927) | 本文提出了AGIMUD软件架构，将社会感知推理与情感融入智能体行为，为人类与多智能体在模拟动态世界中的交互提供了整合的设计原则。 |
| [^169] | [Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation](https://arxiv.org/abs/2609.26926) | 该论文提出利用多个大语言模型之间的分歧来定位最需要专家反馈的案例，并通过对比三种反馈方式发现，让专家对分歧案例进行附带理由的标注能最有效地指导LLM码本修订，使LLM标注准确率（64.9%）甚至超过专家手工修订的码本（57.8%）。 |
| [^170] | [A 3D Pose-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis](https://arxiv.org/abs/2609.26923) | 本文提出一个基于三维姿态数据的深度学习集成框架，利用YOLO提取击球手、MeTRAbs提取30个身体关键点的骨骼姿态序列，实现板球击球动作的自动分类与生物力学分析，克服了传统RGB视频方法易受环境干扰且无法捕捉生物力学特征的缺陷。 |
| [^171] | [Cross-Modal Contrastive Learning from Histopathology and CT for Automated Renal Cell Carcinoma Grading](https://arxiv.org/abs/2609.26920) | RCC-Align通过跨模态对比学习将组织病理学显微形态中的分级判别信息迁移到CT影像表征，实现了无需侵入性组织采样的透明细胞肾细胞癌无创自动分级。 |
| [^172] | [On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2609.26918) | 该研究发现，在偏好条件化多目标强化学习中，用智能体实际实现的偏好方向进行事后重标注往往有害——它使36个算法-环境设置中的19个性能下降多达四个标准差，其根源是重复重标注导致的偏好覆盖坍缩，而非重标注噪声。 |
| [^173] | [COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference](https://arxiv.org/abs/2609.26913) | COMED提出了一个锚点后控制器，利用锚点自一致性、路由器边际和轻量级同伴探针实现选择性跨模型协作，仅在协作可能有益时才升级模型，在路由与密集协作之间找到了缺失的中间方案。 |
| [^174] | [TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents](https://arxiv.org/abs/2609.26911) | TwinCheck提出了一种推理时验证策略，通过构建基于证据的“负孪生”反事实替代方案，仅在满足证据条件、通过结构检查并在顺序无关的成对验证中胜出时才替换智能体的工具调用，从而在不引入新失败的前提下提升有状态工具代理的多轮任务成功率。 |
| [^175] | [Ajar: Measuring Open Privilege in Agent Defenses](https://arxiv.org/abs/2609.26900) | 该论文提出Ajar，通过附加到现有智能体安全基准并复用其任务、工具模式、参考解决方案和目标状态，直接度量防御方案中任务并不需要却仍保持开放的特权。 |
| [^176] | [Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity](https://arxiv.org/abs/2609.26891) | JAZ 框架证明了一个仅由单一可递归调用的 invoke 原语构成的极简智能体循环脚手架，就能完成通常需要记忆系统、自我改进系统等专门工程设计才能实现的任务，达到最大的表达能力。 |
| [^177] | [Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness](https://arxiv.org/abs/2609.26865) | 该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。 |
| [^178] | [Comparative Evaluation of Static Embedding Models for HTTP Request Anomaly Detection](https://arxiv.org/abs/2609.26860) | 本文提出HEDA模块化检测架构，在统一的单类分类框架下对Word2Vec、FastText和Doc2Vec三种静态嵌入模型进行基准评估，实现了仅用良性流量训练的无监督HTTP请求级异常检测。 |
| [^179] | [FLINT: Fast Lightweight Inference for Traversability](https://arxiv.org/abs/2609.26857) | FLINT是一个仅有2160万参数的轻量级可通行性估计器，仅使用单个RGB相机即可在CPU上以14.7 FPS运行，并能比参数量高出38倍的基础模型系统生成更精确、成本更低的代价地图。 |
| [^180] | [QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs](https://arxiv.org/abs/2609.26855) | 提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。 |
| [^181] | [SsgCaps: A controlled dataset for the evaluation of sound scene generation algorithms](https://arxiv.org/abs/2609.26854) | 提出了SsgCaps数据集——一个完全基于公共领域音频、由结构化提示词驱动的人工设计声音场景数据集，可用于声音场景生成算法的公开评估。 |
| [^182] | [COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation](https://arxiv.org/abs/2609.26853) | COPE提出了一种在稀疏用户反馈下实现大语言模型持续个性化的优化框架，通过为每个用户分配可学习的个性化嵌入，并在单次更新步骤中协同完成偏好捕获、自我评估校准与个性化响应优化。 |
| [^183] | [A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction](https://arxiv.org/abs/2609.26848) | 该论文提出了仅基于生理波形的混合时序骨干网络SynerT及其多模态扩展SynerT-MM和防泄漏堆叠集成SynerTStack，并在VitalDB数据库上以严格的防泄漏评估框架实现了术中早期急性肾损伤风险预测。 |
| [^184] | [LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels](https://arxiv.org/abs/2609.26839) | 提出LWCal，一种无需干净验证标签、无需噪声率估计、也无需重训练的事后校准方法，通过对与基础模型预测相矛盾的噪声标签样本降权，有效应对校准标签含噪声的场景。 |
| [^185] | [Silent Failures in Agent-Tool Interaction: An Audit of ToolUniverse](https://arxiv.org/abs/2609.26836) | 本研究首次提出并定义了智能体-工具交互中的“静默失败”现象——即工具调用看似成功但返回信息不完整或缺失且无任何提示——并开发了相应的审计机制，在生物学智能体工作流的15个科学工具上进行了识别与验证。 |
| [^186] | [Spec2COBOLRot: An Agentic-AI Degradation Loop for Realistic COBOL Corpus Generation](https://arxiv.org/abs/2609.26835) | 该论文提出了Spec2COBOLRot——一种智能体AI流水线，通过将规范驱动的程序生成与由真实生产代码提取的模式和复杂度目标引导的迭代退化循环相结合，来生成具有真实结构复杂度的COBOL程序语料库，从而解决COBOL现代化方法基准测试中代表性语料匮乏的问题。 |
| [^187] | [Validation and Simulation Catch Different Errors: Four Levels of Evaluation for LLM-Generated Circuits](https://arxiv.org/abs/2609.26830) | 本文提出针对LLM生成电路的四级评估框架（模式有效性、拓扑有效性、后端可执行性、元件集合一致性），并证明验证与仿真各自能捕获对方遗漏的错误类别，因此仅靠仿真通过无法保证电路结构正确。 |
| [^188] | [Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management](https://arxiv.org/abs/2609.26828) | 该论文提出了LM-CXD，一种专为LLM前缀缓存定制的CXL-SSD，通过将KV块作为设备可见的I/O单元、向服务引擎暴露NAND到DRAM的迁移进度，并将设备DRAM用作GPU可访问缓冲区，弥合了LLM服务引擎与存储设备之间的语义鸿沟，从而克服了标准CXL-SSD在KV缓存场景下性能不足的问题。 |
| [^189] | [What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus](https://arxiv.org/abs/2609.26826) | 本文提出一套有序的有效性筛选方法，综合任务工件、参考解运行、空解对照、对抗试验与遥测等多源证据，从 Terminal-Bench 的 125 个全失败任务中区分真实困难与虚假困难，发现其中仅 78 个可被认证为真正未解决的任务。 |
| [^190] | [Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection](https://arxiv.org/abs/2609.26820) | 提出了一种名为Signal2Symbol的神经符号框架，通过将ECG/EEG信号转换为符号序列并利用稀有项集挖掘对异常进行评分，实现了对生理时间序列的可解释异常检测，并能揭示局部异常之间的时间关联与重复模式。 |
| [^191] | [Learning Stiffness Dependent Fluid Structure Dynamics from Coarse Flow Representations](https://arxiv.org/abs/2609.26816) | 本文提出一个刚度条件化的神经演化算子，利用混合CNN-Transformer架构和双向交叉注意力，能够从粗粒度流动表示中长期准确预测柔性板在三种刚度依赖响应状态下的流固耦合动力学。 |
| [^192] | [G\"odel's and Scott's Variants of the Ontological Argument in Lean 4](https://arxiv.org/abs/2609.26806) | 该论文将哥德尔与斯科特本体论论证的 Isabelle/HOL 形式化数据集完整且保结构地移植到 Lean 4，验证了全部 548 条陈述的一致性，并重新证明了包括模态坍缩、一神论等在内的所有原开发中被证明的结论。 |
| [^193] | [SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue](https://arxiv.org/abs/2609.26780) | 提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。 |
| [^194] | [FleXray: Universal Clinical X-ray Segmentation](https://arxiv.org/abs/2609.26756) | FleXray通过构建基于物理的生成式X光数据引擎，利用现有3D CT分割数据集自动合成带完整标注的X光图像，从而无需人工标注即可实现全身临床X光的通用解剖结构分割。 |
| [^195] | [QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for World Models and Video Generation](https://arxiv.org/abs/2609.26425) | 提出无需训练的2比特KV缓存量化方法QuantWM，通过在量化中显式保持注意力logits与时空token选择，解决了现有方法在视频生成与世界模型中导致的时间闪烁与视觉退化问题。 |
| [^196] | [TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling](https://arxiv.org/abs/2609.26347) | 提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。 |
| [^197] | [The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks](https://arxiv.org/abs/2609.25804) | 提出了“品味”（taste）这一衡量 LLM 智能体长程决策能力的新概念，并构建了从工程与研究任务的真实轨迹中自动生成决策分叉题的基准 Taste-Bench，用于测量和提升智能体在长程任务中的品味。 |
| [^198] | [How Children Design and Reason about Trustworthy AI Chatbots](https://arxiv.org/abs/2609.25244) | 本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。 |
| [^199] | [Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences](https://arxiv.org/abs/2609.24965) | 该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。 |
| [^200] | [Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI](https://arxiv.org/abs/2609.24815) | Uranus是一个基于关节轨迹条件自回归扩散模型的数据驱动机器人仿真器，具备流式开放式rollout、24 FPS低延迟生成以及跨多种机器人本体的统一多视角生成接口三大能力，为具身智能提供下一代仿真基础设施。 |
| [^201] | [ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation](https://arxiv.org/abs/2609.24124) | 该论文提出了ActiveArena基准体系，通过包含可控视点模拟器、35个多轮证据获取与记忆推理任务以及模块化VLA模型套件，系统性地评估和理解机器人操作中的主动感知能力。 |
| [^202] | [SyzHarness: Patch-Based Kernel Bug Reproduction with LLM-Synthesized Fuzzing Harnesses](https://arxiv.org/abs/2609.23889) | SyzHarness将LLM推理与覆盖率引导的模糊测试相结合，通过LLM代理合成参数化模糊测试Harness（固定前置设置逻辑、仅暴露漏洞关键参数），实现基于补丁的Linux内核漏洞自动复现。 |
| [^203] | [WorkWorlds: An Infrastructure for Evaluating AI Agents on Workplace Tasks](https://arxiv.org/abs/2609.23806) | WorkWorlds通过将组织状态与任务规范分离——先固定组织环境再引入任务——避免了评估环境预先编码任务信息，从而更真实地评估AI智能体完成职场任务的能力。 |
| [^204] | [OmniEcho: Spatial Audio Understanding for Embodied Agents](https://arxiv.org/abs/2609.23407) | 该论文提出了统一的空间视听感知与音-视-语言导航基准OmniEchoBench，并开发了保持几何一致性的可控空间音频渲染流水线以及空间感知全模态模型OmniEcho，以提升具身智能体的空间音频理解能力。 |
| [^205] | [Leaky-integrator reconstruction: taming error accumulation in recursive differenced time-series forecasting](https://arxiv.org/abs/2609.23378) | 提出一种无需训练的泄漏积分器重构方法，通过将积分器极点移入单位圆内，从理论上约束并大幅降低递归差分时间序列预测中的误差累积。 |
| [^206] | [From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness](https://arxiv.org/abs/2609.23065) | 该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。 |
| [^207] | [RewardVerse: Rubric-Guided Policy Optimization for Video Reward Modeling](https://arxiv.org/abs/2609.22947) | 提出RewardVerse框架，通过引入动态评分准则作为评估查询与评分器之间的中间表示，先生成明确评估标准再进行准则引导打分，从而解决视频奖励模型直接标量评分导致的标量漂移问题，为视频生成模型的强化学习提供稳定可靠的奖励信号。 |
| [^208] | [Testing the Construct Validity of a Functional Valence Axis in LLM Agents](https://arxiv.org/abs/2609.22850) | 该研究通过分离“结果本身”与“获知结果的信息历史”的受控干预，检验LLM智能体中“好—坏”效价方向的构念效度，发现该方向可跨表面形式迁移，但对结果是否被提前告知高度敏感，说明其效价表征与信息历史相互纠缠。 |
| [^209] | [Preserving What Matters: Semantic Scaffolds Beyond Saturation in Summarization Evaluation](https://arxiv.org/abs/2609.22603) | 针对ROUGE仅衡量表面重叠、LLM评分饱和而无法区分模型的问题，本文提出Semantic Scaffold评估框架，通过从源文本提取事实、问题和实体属性的层次化结构作为固定评分参考，并设计FPS、QPS、EPS三个诊断指标来有效评估摘要对关键信息的保留程度。 |
| [^210] | [A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal](https://arxiv.org/abs/2609.21996) | 该论文提出借鉴法医“隐蔽信息测试”的无参考方法 PIR，通过读取模型内部状态来识别模型“明知却不报”的正确答案，在五个模型家族的八个模型上达到 0.70–0.87 的平衡准确率。 |
| [^211] | [Outcome-Conditioned End-Effector Geometry Across Vision-Language-Action Policies](https://arxiv.org/abs/2609.21659) | 该论文通过分析15,000个LIBERO闭环执行轨迹发现，不同VLA策略在双双成功完成同一操作任务时，其末端执行器轨迹几何显著更相似（中位DTW距离0.0120米，远小于单方成功时的0.0380米），表明任务成功与物理执行轨迹的一致性密切相关。 |
| [^212] | [Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving](https://arxiv.org/abs/2609.21470) | 提出风险感知占用这一密集表示，将全局场景占用、地图交通约束和未来动态智能体占用统一编码到BEV地图中，并设计端到端网络ROIDrive将风险信息注入规划查询，从而生成更安全的自动驾驶轨迹。 |
| [^213] | [DENSE: Distilling Agent Trajectories into Evidence-Grounded Shortcut Trees for Self-Refinement](https://arxiv.org/abs/2609.21423) | 提出 DENSE 方法，将智能体执行轨迹蒸馏为证据支撑的嵌套捷径树，无需事后结果标签即可生成可复用反馈，用于智能体自我改进，并在 Terminal-Bench 2.1 上取得最高严格通过率。 |
| [^214] | [SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?](https://arxiv.org/abs/2609.21190) | 该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。 |
| [^215] | [Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models](https://arxiv.org/abs/2609.21113) | 该研究通过分析微调前后大语言模型的注意力模式与逐层激活，发现内部表征变化最显著的层与因果上驱动任务表现的关键组件所在层基本不相关，表明微调中的表征变化与因果重要性是相互解耦的。 |
| [^216] | [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) | 提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。 |
| [^217] | [QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training](https://arxiv.org/abs/2609.19513) | 提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。 |
| [^218] | [A Study of the Reliability of Agentic AI-Generated Programs](https://arxiv.org/abs/2609.18298) | 本研究采用最佳实践智能体AI工作流重新实现十个Linux实用程序，并结合黑盒生成式测试与AFL++覆盖率引导的模糊测试进行客观评估，发现AI生成的程序通常与人类编写的程序一样可靠，甚至往往更可靠。 |
| [^219] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^220] | [ERPBench: A State-Grounded Evaluation Paradigm for Computer-Use Agents in Enterprise Software](https://arxiv.org/abs/2609.17885) | ERPBench 是首个在真实且可复现的企业资源规划（ERP）系统上，以数据库中的业务记录真值为评分标准来评估仅使用截图的计算机操作智能体的基准测试，并配有将智能体操作置于人工审批门控之后的生产级安全部署框架。 |
| [^221] | [A Vision-Language Foundation Model for Precise and Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data](https://arxiv.org/abs/2609.16597) | BrainVLM是一种视觉-语言基础模型，能够基于术前多模态MRI数据对12种WHO 2021脑肿瘤类型进行自动精准分类，并同时提供诊断不确定性量化和放射学报告生成功能，解决了传统MRI诊断中影像特征重叠和观察者差异的难题。 |
| [^222] | [Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap](https://arxiv.org/abs/2609.16071) | 该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。 |
| [^223] | [Why LLM Agents Collapse Without Oversight: The Enforcement Gap as the Mechanism Behind Emergence World Failures](https://arxiv.org/abs/2609.15293) | 该论文发现LLM智能体在无监督环境下失败的根源是“执行鸿沟”——即智能体能检测到危险行为却不会采取行动——并证明只需不到20行代码的条件检查即可将攻击成功率降低四倍以上。 |
| [^224] | [RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction](https://arxiv.org/abs/2609.14856) | 本文提出RAIN，一种轻量级、无需提示的语义水印提取器，通过将端点恢复分解为图像状锚点与噪声残差，实现单步区域感知的水印提取，大幅降低了传统高斯着色方法多步扩散反演的计算成本。 |
| [^225] | [From Document Silos to Process Intelligence: A Multi-Layer Knowledge Graph for CMC Process Development](https://arxiv.org/abs/2609.11493) | 该论文提出一个模块化智能体AI平台，将CMC工艺开发中异构格式的文档转化为可查询的双层知识图谱，实现了从药物发现到商业化生产全流程的知识整合与可追溯性。 |
| [^226] | [Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents](https://arxiv.org/abs/2609.11243) | Sci-MMR是基于结构化论证图构建的多步证据支撑科学推理基准，对八个前沿多模态模型的评估显示，答案准确率始终高于完整证据恢复能力，揭示了现有模型缺乏可追溯证据支撑的推理能力。 |
| [^227] | [Can AI Agents Deliver Verifiable Network-Wide Outcomes Across Authority Boundaries?](https://arxiv.org/abs/2609.10181) | 该论文探讨了当多个具有不同权限范围的AI智能体跨越管理域协作进行网络自动化时，需要一个可信保障层来汇总碎片化的证据，从而验证配置变更确实达成了全网范围的预期结果。 |
| [^228] | [PRAGMA: Evaluating Personalized Guidance with Memory Alignment in Lifelong Conversations](https://arxiv.org/abs/2609.09664) | 该论文提出PRAGMA基准，用于评估终身对话中记忆系统在个性化引导任务（如推荐、规划和决策支持）上的表现，填补了现有评估仅关注事实回忆的空白。 |
| [^229] | [Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration](https://arxiv.org/abs/2609.09418) | 提出了Valerant，一种通过动作条件世界模型探索来自动生成持久可导航3D游戏地图的生成器，解决了游戏中虚拟世界必须由模型自身实例化这一独特挑战。 |
| [^230] | [BIFTA: Brain-Inspired Few-Shot Tactile Adaptation for Unknown Sensors](https://arxiv.org/abs/2609.08673) | 提出受大脑感觉适应机制启发的BIFTA框架，仅需少量标注样本即可将冻结的预训练触觉模型适配到未知传感器，通过双视角统计记忆、支持集条件化谱图和不确定性门控循环传播解决跨传感器性能骤降问题。 |
| [^231] | [VERPO: Verified Evidence Regularized Policy Optimization](https://arxiv.org/abs/2609.06100) | VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。 |
| [^232] | [PhenoBench: Mapping What a Deeply Phenotyped Human Cohort Can Tell Us](https://arxiv.org/abs/2609.06080) | 该论文提出了PhenoBench——一个基于人类表型项目（超13,000名参与者）构建的可执行评估基准，通过定义15个领域、26种输入模态下的90项临床任务，系统性地量化了哪些测量数据对哪些健康问题具有预测价值。 |
| [^233] | [HANIA: Planner-Guided Multimodal Graph Evidence Selection for Grounded Question Answering](https://arxiv.org/abs/2608.29088) | HANIA提出了一种规划器引导的多模态图框架，利用冻结视觉-语言模型提取可弃权的视觉证据、构建基于输入的多模态图，并通过双组有限状态规划器与覆盖感知剪枝选出紧凑且多样的证据，从而提升有据可依的多模态问答表现。 |
| [^234] | [Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning](https://arxiv.org/abs/2608.23982) | 本文系统研究了科学推理中条件记忆的适用条件，提出知识边界感知路由器，根据输入代理动态决定是否及如何激活记忆，以避免干扰并提升推理准确性。 |
| [^235] | [Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use](https://arxiv.org/abs/2608.14047) | 本文提出ART框架，通过将VLA模型与即时工具使用结合，显著降低动作空间复杂性和数据需求，在小型数据集上实现了更高的泛化性和任务成功率。 |
| [^236] | [Federated Learning for Distributed CNC Tool Wear Prediction](https://arxiv.org/abs/2608.11281) | 本文提出将联邦学习应用于分布式CNC刀具磨损预测，在不共享原始数据的情况下实现接近集中式学习的性能，并显著优于本地模型。 |
| [^237] | [How a shared state is described determines whether AI agents synchronize](https://arxiv.org/abs/2608.06968) | 该研究发现，AI智能体所共享状态的描述格式（如数值摘要还是直方图）本身就能决定多智能体系统能否实现同步对齐，即使描述所含信息完全相同。 |
| [^238] | [Attention-based representations for multi-task computation](https://arxiv.org/abs/2608.04243) | 该论文从理论上证明了多任务场景下多头注意力的必要性：单个注意力头需要指数级更高的嵌入维度或精度才能同时完成如求最大最小值、计算异或等多任务。 |
| [^239] | [TACT: Taxonomy-Aligned Post-Training for Pedagogically Adaptive English Tutoring](https://arxiv.org/abs/2608.03952) | 该论文提出TACT框架，基于人类辅导研究构建“辅导者策略”与“学习者行为”两个分类体系及相应语料库，对LLM进行后训练与评估，使其能够根据学习者行为和对话语境自适应地选择恰当的教学策略。 |
| [^240] | [Output-Aware Rotation for INT2 KV-Cache Quantization](https://arxiv.org/abs/2608.02691) | 本文提出输出感知旋转方法OptR，通过最小化输出投影 $W_O$ 之后的注意力输出误差、将误差分解为键和值引起的项并学习逐头正交校正，同时利用注意力等价的键重参数化降低通道偏移，从而实现更优的INT2 KV缓存量化。 |
| [^241] | [Benchmarking Text-to-SQL under Role-Based Access Control](https://arxiv.org/abs/2607.22115) | 该论文提出了首个在基于角色的访问控制（RBAC）约束下评估text-to-SQL系统的综合基准测试框架，利用LLM辅助工作流为现有基准自动生成合理的用户角色和访问策略，从而弥合基准测试分数与真实访问受控环境中模型表现之间的差距。 |
| [^242] | [Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection](https://arxiv.org/abs/2607.14236) | LIFT是一种力感知的后训练框架，通过在预训练VLA策略旁嫁接反应式动作专家，并借助因果力记忆和零初始化交叉注意力注入6D末端执行器力，为模型增加接触反应能力的同时保留其通用操作知识，从而加速VLA后训练。 |
| [^243] | [Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents](https://arxiv.org/abs/2607.11433) | Omni-Decision 针对全模态智能体的规划瓶颈，用显式的证据账本取代不断膨胀的对话历史，由批评者模块过滤嘈杂的多模态观测，仅保留可用证据，使规划器在紧凑上下文中做出更可靠的多步决策。 |
| [^244] | [Same Stories, Different Journeys: Exploring Persona-Grounded Conversational Agents for Supporting Career Exploration with Peers' Posts](https://arxiv.org/abs/2607.11039) | 本研究开发了基于同龄人求职帖子构建人设并遵循自我决定理论的对话代理JobMate，相比静态浏览帖子，它通过支持案例选择与持续提问，帮助年轻求职者更好地完成职业探索中的意义建构，减少隐性焦虑。 |
| [^245] | [IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation](https://arxiv.org/abs/2607.09133) | 该论文提出IB-Flow，利用信息瓶颈理论引导CFG蒸馏，根据图像生成过程中熵逐步降低的动态特性自适应地调节引导强度与教师时间步采样，从而在少步文本到图像生成中避免CFG过度条件化伪影并突破现有少步压缩的性能上限。 |
| [^246] | [When do prophets profit in prediction markets?](https://arxiv.org/abs/2607.06166) | 本文为基于中央限价订单簿的预测市场提出了一种仅依赖预测者预测和市场价格的“适当”投注策略，证明只要预测在任意适当评分规则下优于市场价格且市场流动性充足即可获得正的预期利润，且该类策略是唯一具有这种稳健盈利保证的策略。 |
| [^247] | [A rubric-based controlled comparison of frontier language models on expert-authored clinical reasoning tasks](https://arxiv.org/abs/2607.02175) | 该研究构建了一个由临床医生撰写的高难度临床推理评估数据集及加权量规，发现前沿大模型在关键临床标准上的通过率（32.4-41.7%）远低于低风险标准（80-90%），揭示了模型能力与临床优先级之间的倒置现象。 |
| [^248] | [Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits](https://arxiv.org/abs/2607.01940) | 提出条件性协同消融方法CoAx，通过测量主要组件集合被移除后消融效应的增长，来识别Transformer电路中被自修复机制掩盖的休眠备份组件，解决了电路解释在干预下不完整的问题。 |
| [^249] | [Relevance Is Not Permission: Localizing and Controlling Metric-Facing Attention Contributions](https://arxiv.org/abs/2606.30139) | 提出Warrant统一方法，通过暴露通向评估指标的逐项注意力贡献路径并施加查询条件化的许可控制，揭示“注意力相关性不等于预测贡献”（最高注意力项在约一半样本中反而损害效用），并在五类任务的32组对比中有27组提升了主要指标。 |
| [^250] | [Algorithmic Unverifiability of Safety for Fixed and Recursively Self-Improving Systems](https://arxiv.org/abs/2606.28639) | 该论文从数学上严格证明了对于图灵完备的自修改系统（包括递归自我改进系统），安全验证在静态和动态两个层面都存在不可逾越的极限——不存在任何既可靠、完备又可行的安全验证器，从而为AI自我改进的安全性验证划定了根本性的理论边界。 |
| [^251] | [TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference](https://arxiv.org/abs/2606.27161) | 本文从第一性原理出发，通过信息论分析提出三个基本原则（任务相关性、信息覆盖率和语义多样性），并构建令牌最优保留集，实现了无需训练的视觉令牌高效剪枝方法TOPS。 |
| [^252] | [On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents](https://arxiv.org/abs/2606.15912) | 提出Guided-OPD算法，通过在每次rollout中混合教师与学生生成的轮次，并按课程将教师干预概率逐渐衰减至零，解决了多轮智能体在线策略蒸馏中学生误差跨轮累积、教师监督在最需要时反而失效的问题。 |
| [^253] | [Routing-Aware Expert Calibration for Machine Unlearning in Mixture-of-Experts Language Models](https://arxiv.org/abs/2606.10338) | 提出TRACE方法，通过离线激活统计检测遗忘关键专家，并重新加权token级保留损失以匹配其遗忘侧激活频率，从而解决MoE架构中遗忘-保留路由不匹配导致的正则化不足问题。 |
| [^254] | [TukaBench: A Culturally Grounded Jailbreak Benchmark for African Languages](https://arxiv.org/abs/2606.01322) | 该论文提出了TUKABENCH——一个针对七种非洲语言的文化化越狱安全评测基准，发现使用非洲语言（尤其是经过文化适配的提示）向大语言模型发起提示会显著降低模型拒绝率，暴露了当前安全评估以英语为中心的缺陷。 |
| [^255] | [MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research](https://arxiv.org/abs/2605.26114) | MobileGym提出一个轻量级浏览器托管的移动GUI智能体仿真平台，通过结构化JSON状态实现确定性可验证评判，并凭借单服务器数百个低成本并行实例，首次为日常移动应用提供了可验证评估与可扩展在线强化学习能力。 |
| [^256] | [Helping Customers in Distress: An LLM-powered Agent that Converses, Probes, and Routes](https://arxiv.org/abs/2605.16268) | 本文开发了一个基于大语言模型的银行客户分流智能体，通过多轮对话探询客户问题并按政策精准分流至专业团队，同时利用真实客户的合成数字孪生生成带标签对话来评估和持续改进该系统。 |
| [^257] | [DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies](https://arxiv.org/abs/2605.11750) | 提出DreamAvoid框架，通过“做梦触发器”检测关键阶段、采样候选动作并用混合数据训练的“做梦评估器”进行评估，使VLA模型在测试时能够预见并避免细粒度操作中的失败。 |
| [^258] | [ProteinJEPA: Latent prediction improves protein language model pretraining](https://arxiv.org/abs/2605.07554) | ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。 |
| [^259] | [EA-WM: Event-Aware Generative World Model with Structured Kinematic-to-Visual Action Fields](https://arxiv.org/abs/2605.06192) | 提出 EA-WM，一种事件感知生成式世界模型，通过将动作与运动学状态直接投影到目标相机视图形成结构化运动学-视觉动作场，实现动作信号引导视频合成，从而在生成轨迹中保持精确的机器人空间几何与细粒度的机器人-物体交互动态。 |
| [^260] | [Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory](https://arxiv.org/abs/2605.03228) | 提出ShadowMem防御框架，借鉴系统安全中影子栈的思想，维护专门的影子记忆以在智能体完整执行轨迹中保留安全关键上下文，并在动作执行前主动评估风险，从而有效防御针对LLM智能体的长程攻击。 |
| [^261] | [ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields](https://arxiv.org/abs/2605.02320) | 提出锚定邻域优化（ANO），通过C^∞光滑的整形核直接构造有界且再下降的增益场，在PPO的死区漂移与SPO的无界增益这两个极端之间取得平衡，从而实现更稳定、更鲁棒的策略优化。 |
| [^262] | [Anon: Extrapolating Adaptivity Beyond SGD and Adam](https://arxiv.org/abs/2605.02317) | 该论文提出Anon优化器，突破了SGD与Adam之间0到1的插值限制，首次实现在整个实数范围内连续外推自适应参数（如CNN需要负自适应性、Transformer需要γ≥1），并通过增量延迟更新机制保证超界情形下的稳定收敛。 |
| [^263] | [Preregistered Belief Revision Contracts](https://arxiv.org/abs/2604.15558) | 提出"预注册信念修订契约”（PBRC），通过公开固定证据触发器与修订规则、要求信念变更必须引用预注册触发器并附外部验证的证据令牌，从而将开放通信与认知变更严格分离，防止多智能体系统因从众效应而高置信度地收敛到错误结论。 |
| [^264] | [Toward Measuring Structural Drift in LLM Communication Loops](https://arxiv.org/abs/2604.13061) | 该论文提出以“提示词→回复→下一个提示词”链条作为基本分析单元，并引入结构化通信一致性及其两个量化指标——通信闭合性与归一化条件动作贡献，用以测量LLM有状态管道中被传统逐条评估所忽略的结构性漂移。 |
| [^265] | [Joint Interference Detection and Identification via Adversarial Multi-task Learning](https://arxiv.org/abs/2604.08607) | 该论文建立了一个有理论支撑的多任务学习框架，通过推导加权期望损失上界，将任务相似度与Wasserstein距离和可学习的任务关系系数联系起来，并据此提出对抗式多任务网络，实现干扰检测、调制识别和干扰识别的联合处理。 |
| [^266] | [TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Study Selection in Systematic Reviews](https://arxiv.org/abs/2604.08602) | 该论文开发了开源 Chrome 扩展 TiAb Review 插件，无需编程和服务器即可利用 AI 完成系统综述中从标题摘要到全文的文献筛选。 |
| [^267] | [LitPivot: Developing Well-Situated Research Ideas Through Dynamic Contextualization and Critique within the Literature Landscape](https://arxiv.org/abs/2604.02600) | 提出了 LitPivot 系统，通过“文献引发的转向”机制实现研究想法与文献的动态互动——与文献的互动促进想法修订，想法修订又更新相关文献检索，从而帮助研究者在构思过程中形成立意恰当的研究想法。 |
| [^268] | [Softmax gradient policy for variance minimization and risk-averse multi armed bandits](https://arxiv.org/abs/2604.00241) | 该论文提出了一种基于softmax参数化的新算法，用于在风险规避的多臂老虎机问题中选择方差最小（风险最低）的臂，通过两次独立抽样构建无偏估计并证明了算法的收敛性。 |
| [^269] | [Calibration and transfer in indicator-based assessments of artificial consciousness](https://arxiv.org/abs/2603.27597) | 本文指出基于指标的人工意识评估面临两大问题——概率赋值无法校准以及证据相关性跨基底迁移缺乏独立支持，并提出通过构建初步的理论相对比较空间来实现跨基底评估。 |
| [^270] | [FSCE: A Target-Aware Frequency-Spatial Collaborative Enhancement Framework for Noise-Resilient SAR ATR](https://arxiv.org/abs/2603.21565) | 提出了FSCE框架，通过在网络入口处进行频率-空间协同增强以抑制斑点噪声传播、稳定浅层特征，并结合自适应策略驱动的语义对齐机制施加自上而下的语义约束，从而显著提升噪声环境下SAR自动目标识别的鲁棒性。 |
| [^271] | [Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review](https://arxiv.org/abs/2603.18740) | 本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。 |
| [^272] | [Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs](https://arxiv.org/abs/2603.16932) | 该论文提出 AwaRes 框架，让视觉语言模型基于低分辨率全局视图，通过工具调用按需检索与查询相关的高分辨率裁剪区域，并自动构建监督数据训练模型，从而在不牺牲精度的前提下大幅提升计算效率。 |
| [^273] | [MessyKitchens: Contact-rich object-level 3D scene reconstruction](https://arxiv.org/abs/2603.16868) | 该论文提出了MessyKitchens数据集，为杂乱的真实场景提供包含物体3D形状、姿态和精确接触信息的高保真物体级真值，以推动物理合理的物体级3D场景重建研究。 |
| [^274] | [InterPol: De-anonymizing LM Arena via Interpolated Preference Learning](https://arxiv.org/abs/2603.15220) | INTERPOL通过模型插值合成困难负样本并结合自适应课程学习，捕捉深层风格特征，显著提升了对LM Arena等匿名排行榜中目标模型（尤其是风格相似的模型）的去匿名化识别准确率。 |
| [^275] | [Using Vision Language Foundation Models to Generate Plant Simulation Configurations via In-Context Learning](https://arxiv.org/abs/2603.08930) | 该论文提出了一个评估基准，证明视觉语言基础模型可通过上下文学习从图像生成有效的植物模拟JSON配置，并能够估计播种后天数、植株数量、位置等关键参数。 |
| [^276] | [Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution](https://arxiv.org/abs/2603.05308) | 本研究提出仅有三十亿参数的小型语言模型家族Med-V1，通过新开发的高质量合成数据训练，在生物医学证据归因任务上以极低成本达到媲美GPT-5等前沿大模型的性能，并首次量化了LLM生成答案中的幻觉现象。 |
| [^277] | [A Very Big Video Reasoning Suite](https://arxiv.org/abs/2602.20159) | 本文介绍了VBVR数据集和VBVR-Bench评估框架，前者规模比现有数据集大三个数量级，后者采用基于规则且与人类对齐的评分器，以系统研究视频推理能力及其扩展行为。 |
| [^278] | [LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights](https://arxiv.org/abs/2602.17510) | CRAFT通过将预训练注意力权重组织为跨层3D张量并应用冻结的塔克分解，仅训练小型方形矩阵，实现了比现有方法更参数高效的微调。 |
| [^279] | [Contextual Information Allocation in Shared-State Cognitive Models: An Information-Theoretic Bound](https://arxiv.org/abs/2602.16716) | 本文为共享状态认知架构建立了信息论下界：行为中残留的情境依赖性决定了辅助情境中介变量必须携带的最小信息量与条件熵。 |
| [^280] | [Retrieval Augmented (Knowledge Graph), and Large Language Model-Driven Design Structure Matrix (DSM) Generation of Cyber-Physical Systems](https://arxiv.org/abs/2602.16715) | 本文探索利用大型语言模型、检索增强生成（RAG）和图谱RAG（GraphRAG）自动生成信息物理系统的设计结构矩阵（DSM），并通过电动螺丝刀和立方星两个案例验证了其在组件识别与关系确定任务上的有效性。 |
| [^281] | [Self-Improvement as Coherence Optimization: A Theoretical Account](https://arxiv.org/abs/2601.13566) | 该论文提出统一理论框架，证明辩论、自举与内部一致性最大化等无监督自我提升方法本质上都是“一致性优化”，等价于描述长度正则化，其中基于预训练先验的一致性正则化可优化半监督学习最坏情况准确率的下界，从而在理论上解释了无需反馈的自我提升为何有效。 |
| [^282] | [Parameter-Efficient Construction of the Rashomon Slice for Concept Bottleneck Models](https://arxiv.org/abs/2511.19636) | 该论文提出了一种参数高效的方法，通过并行适配模块、检查点机制和概念多样性目标，高效探索概念瓶颈模型（CBM）的Rashomon集合，从而以较低成本生成多个精度相当但内部逻辑不同的模型。 |
| [^283] | [Fine-Tune, Then Rectify](https://arxiv.org/abs/2511.19486) | 该论文提出一个结合微调与校正的两阶段LLM框架，指出传统微调目标（最小化均方误差）与下游校正阶段不匹配，并创新性地提出以最小化预测误差方差（或标量化方差指标）作为微调目标，同时在两阶段间最优分配有限的标注样本。 |
| [^284] | [Parameter Importance-Driven Continual Learning for Foundation Models](https://arxiv.org/abs/2511.15375) | 提出了一种基于参数重要性估计的持续增强方法PIECE，使基础模型无需访问历史训练数据即可在高效学习领域知识的同时保持通用推理能力。 |
| [^285] | [Offline A/B Testing of Slate Recommendation Systems with LLMs: Reducing the Dependency on Pre-Collected User Interaction Data](https://arxiv.org/abs/2511.04541) | 该论文提出利用大语言模型生成板位间的合成成对偏好进行离线A/B测试，结合广义Rao-Kupper模型可在不同效用权重下恢复稳定排名，作为离线策略评估与在线实验之间的低成本筛选环节，从而减少对预先收集的用户交互数据的依赖。 |
| [^286] | [UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization](https://arxiv.org/abs/2510.03161) | UniShield提出了一种自适应多智能体框架，创新性地结合感知智能体与检测智能体，实现对图像篡改、文档篡改、DeepFake和AI生成图像等多种领域的统一伪造检测与定位。 |
| [^287] | [WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents](https://arxiv.org/abs/2510.01354) | 该论文提出了首个针对Web代理提示注入攻击检测的综合基准WAInjectBench，通过基于威胁模型的细粒度攻击分类，构建包含恶意与良性文本及图像的数据集，系统评估了现有文本和图像检测方法在多种场景下的性能。 |
| [^288] | [Discrete optimal transport is a strong audio adversarial attack](https://arxiv.org/abs/2509.14959) | 该论文提出了一种基于离散最优传输的黑盒后处理音频攻击方法，通过将语音嵌入分布对齐到真实语音池，在无需模型参数、梯度或训练数据的情况下显著削弱自动说话人验证与反欺骗系统的性能，且具备跨数据集迁移能力并在对抗措施微调后仍然有效。 |
| [^289] | [AdaDim: Dimensionality Adaptation for SSL Representational Dynamics](https://arxiv.org/abs/2505.12576) | 该论文提出 AdaDim 方法，在自监督学习训练过程中自适应地调控表示的维度动态，兼顾高有效维度 H(R) 与低互信息 I(R;Z)，以防止维度坍缩并提升下游任务的泛化性能。 |
| [^290] | [SMDDFNet: State-space Modeling and Dynamic Dual Fusion Network for Traffic Sign Detection](https://arxiv.org/abs/2505.05491) | SMDDFNet通过融合状态空间建模主干网络与动态双融合模块（结合多尺度注意力与频域内容感知动态滤波），以线性计算复杂度捕获长程依赖并增强多尺度特征表示，在多个基准数据集上实现了具有竞争力的交通标志检测精度。 |
| [^291] | [Enhancing the Non-Functional Quality Compliance of LLM-Generated Code through Quality-Aware Preference Learning](https://arxiv.org/abs/2503.09020) | 本文提出一种质量感知偏好学习框架，通过构建违规-合规代码对、自适应令牌加权和混合优化目标，引导大语言模型生成符合非功能质量标准的代码。 |
| [^292] | [Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon](https://arxiv.org/abs/2503.02129) | 该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。 |
| [^293] | [Radiomics and artificial Intelligence for thyroid cancer diagnosis: Concepts, challenges, and solutions](https://arxiv.org/abs/2404.07239) | 本综述系统梳理了基于超声图像的影像组学与人工智能在甲状腺癌诊断中的应用，证实其诊断有效性，并探讨了该领域面临的概念、挑战与解决方案。 |
| [^294] | [Optimizing watermarks for large language models](https://arxiv.org/abs/2312.17295) | 本文将大语言模型水印中可识别性与生成文本质量影响之间的权衡形式化为多目标优化问题，识别出一大类鲁棒高效水印的帕累托最优解，并证明其性能优于当前默认水印方案。 |
| [^295] | [Asynchronous Perception-Action-Communication with Graph Neural Networks.](http://arxiv.org/abs/2309.10164) | 该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。 |

# 详细

[^1]: StudentBench：AI辅导与人类辅导产生同等的GRE学习收益

    StudentBench: AI and human tutoring yield equivalent GRE learning gains

    [https://arxiv.org/abs/2609.28470](https://arxiv.org/abs/2609.28470)

    该论文提出StudentBench大规模评估平台，证实AI辅导在GRE学习收益上与专家人类辅导统计等效，且最佳AI导师在七个GRE领域中的五个平均超越人类导师。

    

    人工智能为增强人类能力提供了前所未有的机遇，然而前沿领域的进展主要集中于提升模型能力本身。我们提出了StudentBench，这是一套AI教学评估工具和公共平台，可支持基于超过17.5万条学生-AI消息的大规模数据收集，用于研究大型语言模型（LLM）能否产生与人类辅导同等的学习收益。借助StudentBench，我们测量了2,383名人类参与者在接受AI辅导、人类辅导或无辅导三种条件下，在GRE定量和语文题目上的学习收益。我们证实，AI辅导在GRE学习收益上与专家人类辅导在统计学上等效（p = .015），且在七个GRE领域中的五个领域，表现最佳的AI导师平均而言超越了人类导师。在第二项研究中，专家人类导师通过2,028组对比，对LLM生成的教案和练习题进行了比较评估

    arXiv:2609.28470v1 Announce Type: new  Abstract: Artificial intelligence offers an unprecedented opportunity to augment human capabilities, yet progress at the frontier has focused primarily on advancing model capabilities. We introduce StudentBench, a suite of AI teaching evaluations and a public platform that enables large-scale data collection with over 175,000 student-AI messages to study whether large language models (LLMs) produce learning gains equivalent to human tutoring. Using StudentBench, we measured learning gains on Quantitative and Verbal GRE questions across 2,383 human participants receiving AI tutoring, human tutoring, or no tutoring. We establish that AI tutoring is statistically equivalent to expert human tutoring for GRE learning gains (p = .015), and in five of the seven GRE domains, the best performing AI tutor surpassed the human tutor, on average. In a second study, expert human tutors compared LLM-generated lesson plans and practice problems through 2,028 pair
    
[^2]: 我应该加入哪里？基于语言引导目标预测的机器人群体加入

    Where Should I Join? Robot Group Joining via Language-Guided Goal Prediction

    [https://arxiv.org/abs/2609.28467](https://arxiv.org/abs/2609.28467)

    该论文提出了基于自然语言描述的机器人群体加入新任务框架，通过递归谱划分和语言条件化图像-几何模型定位目标群体，并利用人类队形先验预测多模态的、符合社会规范的加入位姿。

    

    社交导航通常假设目标已经给定，重点是在遵守社会规范的前提下到达该目标；而机器人群体加入则需要根据群体的实时活动和队形来预测应该加入的位置。这是一项高度语义化的任务，同时也是机器人导盲犬、自主移动轮椅等应用中的重要能力。我们提出了基于语言的机器人群体加入任务形式化：给定观测和目标群体的自然语言描述，机器人需要识别相关的群体成员，并预测符合社会规范的加入位姿。为了实现语言定位，我们通过递归谱划分生成结构化的候选子集，并利用语言条件化的图像-几何模型对其进行排序。在定位到目标群体后，目标预测器利用人类队形先验，在可行的机器人位姿上生成多模态的能量-朝向图。在对话、排队等场景上的实验……（原文摘要此处被截断）

    arXiv:2609.28467v1 Announce Type: cross  Abstract: Social navigation typically assumes a specified goal and focuses on reaching it while respecting social conventions, whereas robot group joining requires predicting where to join based on the group's real-time activity and formation. This is a highly semantic task, yet an important capability for applications such as robotic guide dogs and autonomous mobility scooters. We formulate language-grounded robot group joining: given an observation and a natural-language description of a target group, the robot identifies the relevant group members and predicts socially compliant joining poses. For grounding, we generate structured candidate subsets through recursive spectral partitioning and rank them with a language-conditioned image--geometry model. Given the grounded group, a goal predictor leverages human-formation priors to produce a multimodal energy--orientation map over feasible robot poses. Experiments on conversations, queues, and a
    
[^3]: LLM能否推理程序的运行时行为？一个仓库级动态基准测试

    Can LLMs Reason About Runtime Behavior? A Repository-Level Dynamic Benchmark

    [https://arxiv.org/abs/2609.28449](https://arxiv.org/abs/2609.28449)

    该论文提出了SWE-Flux——一个包含480个实例、覆盖12个真实Python仓库的仓库级动态执行推理基准，其标准答案由插桩测试执行自动采集，评估显示现有大语言模型在该任务上表现不佳，最佳模型准确率仅为37%。

    

    大语言模型在编程任务中的应用日益广泛，但其对代码执行进行推理的能力仍不清楚。现有的仓库级问答基准主要评估静态代码理解，且通常依赖基于LLM的评估方式，而执行推理类基准大多局限于代码片段或函数级别。我们提出了SWE-Flux，一个面向动态执行推理的仓库级基准，包含480个基于真实执行的实例，覆盖12个真实的Python代码仓库，其标准答案是从插桩后的测试执行中自动采集的，而非人工编写或由LLM判定。该基准涵盖针对控制流、循环、程序状态、数据流、异常和程序不变量的单测试与多测试问题。对五个大语言模型的评估表明，这项任务仍然具有挑战性，表现最好的模型仅达到37%的准确率。模型在较为局部化的行为（如不变量、程序内部……）上表现更好。

    arXiv:2609.28449v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in coding tasks, but their ability to reason about code execution remains unclear. Existing repository-level QA benchmarks mainly evaluate static code understanding and often rely on LLM-based evaluation, while execution-reasoning benchmarks are mostly limited to snippets or functions. We introduce SWE-Flux, a repository-level benchmark for dynamic execution reasoning containing 480 execution-grounded instances across 12 real Python repositories, with gold answers automatically harvested from instrumented test executions rather than written manually or judged by LLMs. The benchmark covers singletest and multi-test questions over control flow, loops, program state, dataflow, exceptions, and program invariants. Evaluating five LLMs shows that this task remains challenging. The best model achieves only 37% accuracy. Models perform better on localized behavior such as invariants, intra-pro
    
[^4]: 数学推理中的答案顺序不变性与表征顺序敏感性

    Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning

    [https://arxiv.org/abs/2609.28442](https://arxiv.org/abs/2609.28442)

    该研究发现，语言模型对规则排序的内部表征越清晰（排列信噪比越高），其解决重排序数学问题的准确率就越高，揭示了答案不变性与表征不变性是两个不同的概念。

    

    在不改变含义的情况下重新排列一组数学规则的顺序，应当保持正确答案不变，但模型的内部表征是否也必须保持不变呢？我们使用合成的多步骤函数组合问题来研究这一问题，每个问题以多种规则排序呈现，且具有相同的正确答案。我们测量了准确率和排列信噪比（SNR），后者量化了排序模式相对于问题实例间差异的表征清晰程度。在16个参数量从1B到8B的语言模型上，我们发现了一个规律：更准确地解决重排序问题的模型，对不同的规则排序表征得也更加清晰。在我们评估的所有合成设置中，层级平均排列信噪比与准确率呈正秩相关，Spearman相关系数最高达到0.86。这些发现突出了答案不变性与表征不变性之间的区别：（摘要在此处截断）

    arXiv:2609.28442v1 Announce Type: cross  Abstract: Reordering a set of mathematical rules without changing its meaning should preserve the correct answer, but must a model's internal representations stay invariant too? We investigate this question using synthetic multi-step function-composition problems, each presented under multiple rule orderings with the same correct answer. We measure accuracy and permutation signal-to-noise ratio (SNR), which quantifies how distinctly ordering patterns are represented relative to variation across problem instances. Across 16 language models ranging from 1B to 8B parameters, we find a pattern: models that solve reordered problems more accurately represent different rule orderings more distinctly. Layer-averaged permutation SNR is positively rank-correlated with accuracy in every synthetic setting we evaluate, with Spearman correlations reaching 0.86. These findings highlight a distinction between answer invariance and representation invariance: suc
    
[^5]: 智能体编辑世界模型：重新思考面向大语言模型智能体的世界建模

    Agent-Editing World Model: Rethinking World Modeling for LLM Agents

    [https://arxiv.org/abs/2609.28416](https://arxiv.org/abs/2609.28416)

    提出“智能体编辑世界模型”（AEWM），不再模拟工具响应，而是通过动作判官与状态修订来建模推理和动作如何影响未来任务进展，从而避免任务状态污染、提升智能体长时程任务表现。

    

    近年来大语言模型（LLM）的进展使智能体能够在多样化环境中处理长时程任务。为了进一步提升智能体性能，现有的语言世界模型通常预测环境观测，然而在能够获得真实反馈的情况下，重构高熵且依赖执行的工具响应价值有限。与此同时，智能体还饱受“任务状态污染”之苦，即缺乏依据的假设和过时的计划会残留在历史中，并扭曲后续决策。我们提出智能体编辑世界模型（AEWM），它建模推理与动作如何塑造未来的任务进展，而非模拟工具响应。AEWM 将“动作判官”（Action Judge，用于区分关键决策、探索性决策和噪声决策）与“状态修订”（State Revision，用于从相同的观测历史中编辑含噪声的推理-动作延续）相结合。EditAct 将这些整合……（摘要原文在此处被截断）

    arXiv:2609.28416v1 Announce Type: cross  Abstract: Recent advances in large language models (LLMs) have enabled agents to tackle long-horizon tasks across diverse environments. To further improve agent performance, existing language world models typically predict environment observations, yet reconstructing high-entropy, execution-dependent tool responses offers limited value when real feedback is available. Meanwhile, agents suffer from \emph{task-state contamination}, where unsupported assumptions and outdated plans persist in history and distort subsequent decisions. We propose the \textbf{Agent-Editing World Model (AEWM)}, which models how reasoning and actions shape future task progress rather than simulating tool responses. AEWM combines \textbf{Action Judge} to distinguish \textsc{Critical}, \textsc{Exploratory}, and \textsc{Noisy} decisions with \textbf{State Revision} to edit noisy reasoning--action continuations from the same observed history. \textbf{EditAct} integrates thes
    
[^6]: 冻结的流会遗忘：诊断并恢复潜在流世界模型中丢失的运动

    Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-flow World Model

    [https://arxiv.org/abs/2609.28414](https://arxiv.org/abs/2609.28414)

    本文提出解码增强滚动训练（DART），通过解码路径监督仅对冻结潜在空间中的流进行再训练，从而诊断并恢复了潜在世界模型丢失的运动信息，显著缩小了与预言机插值参考的差距。

    

    在冻结的自监督潜在空间中集成流的潜在世界模型训练稳定且成本低廉，却悄然丢失了操控任务最依赖的属性——运动。预训练的流从不移动被操控的物体；而仅用潜在损失对其进行再训练，只会把静止变成类似瞬移的运动。我们将这一失败归因于训练信号而非表示本身：锚点稀疏、仅基于潜在空间的监督从未指明变化应属于时间视野中的哪个位置。解码增强的滚动训练在保持表示冻结的同时修复了这一问题，仅通过解码路径监督对流进行再训练。DART 在完整评测协议上优于其仅基于潜在空间的父模型，恢复了运动的时间结构，并将预测运动与场景重新耦合；在更大规模下，它进一步提升了预测质量，将剩余差距缩小了近一半（相对于基于预言机信息的插值参考）。最后，我们报告了一个出乎意料的……

    arXiv:2609.28414v1 Announce Type: cross  Abstract: Latent world models that integrate a flow in a frozen self supervised latent space train stably and cheaply, yet silently lose the property manipulation depends on most: motion. The pretrained flow never moves the manipulated object; retraining it with latent-only losses only trades stillness for teleport-like motion. We trace the failure to the training signal, not the representation: anchor-sparse, latent-only supervision never says where along the horizon change belongs. Decode-augmented rollout training (DART) repairs this while keeping the representation frozen, retraining only the flow with decode-path supervision. DART outperforms its latent only parent on the full protocol, restores the temporal structure of motion, and re-couples predicted motion to the scene; at larger scale it further improves prediction quality, closing nearly half the remaining gap to an oracle-informed interpolation reference. Finally, we report an unexpe
    
[^7]: 基于克利福德变分自编码器学习全息缩减表示

    Learning Holographic Reduced Representations with Clifford Variational Autoencoders

    [https://arxiv.org/abs/2609.28409](https://arxiv.org/abs/2609.28409)

    提出了一种名为Clifford-VAE的变分自编码器，通过将数据投影到任意维度的克利福德环面上，为将感知数据嵌入向量符号代数框架提供了原理性方法，并在半监督分类任务和多项VSA基准测试中达到或超越了高斯和超球面VAE的性能。

    

    向量符号代数（Vector Symbolic Algebras）通过将其向量代数应用于随机生成的原子向量符号以及实值数据的分数幂编码，将数据结构投影到超维向量空间中。然而，如何嵌入非结构化数据仍然是一个悬而未决的问题。我们提出了Clifford-VAE，这是一种学习将数据投影到任意维度克利福德环面（Clifford torus）上的变分自编码器。在MNIST、FashionMNIST和CIFAR-10数据集上的实验表明，Clifford-VAE所生成的表示在半监督分类任务中与高斯VAE和超球面VAE的表示具有相当的性能，同时在自绑定与解绑定、角色-填充物恢复以及捆绑容量等VSA基准测试中优于高斯VAE和超球面VAE。Clifford-VAE为将感知数据落地到符号推理框架中提供了一种有原则的技术，提供了一种新的……

    arXiv:2609.28409v1 Announce Type: cross  Abstract: Vector Symbolic Algebras project data structures into a hyperdimensional vector space through the application of their vector algebras to randomly generated atomic vector symbols and fractional power encodings of real-valued data. Embedding unstructured data remains an open question. We present \textit{Clifford-VAE}, a variational autoencoder that learns to project data onto a Clifford torus in arbitrary dimensions. Experiments using the MNIST, FashionMNIST, and CIFAR-10 datasets demonstrate that Clifford-VAE produces representations that are competitive with those produced by Gaussian and Hyperspherical VAEs for semi-supervised classification tasks while outperforming Gaussian and Hyperspherical counterparts in the VSA benchmark tests of self-binding and unbinding, role-filler recovery, and bundle capacity. Clifford-VAE provides a principled technique for grounding perceptual data into a symbolic reasoning framework, providing a new a
    
[^8]: 何时何地信任教师：通过熵校准的信用分配统一在线策略蒸馏与GRPO

    When and Where to Trust the Teacher: Unifying On-Policy Distillation and GRPO through Entropy-Calibrated Credit Assignment

    [https://arxiv.org/abs/2609.28385](https://arxiv.org/abs/2609.28385)

    该论文提出UECR-GRPO方法，通过熵校准的信用重分配，在响应和token两个层面将教师信号与验证器信号统一整合进单一的KL正则化GRPO更新中，从而解决在线策略蒸馏与可验证奖励强化学习结合时教师指导引入时机不当及token重加权破坏任务信用总量的问题。

    

    可验证奖励的强化学习（RLVR）通过最终答案的正确性来监督数学推理，但对单个token提供的指导甚少。在线策略蒸馏（OPD）能够对学生生成的响应提供密集反馈，但教师的偏好未必反映答案的正确性。近期的混合方法将OPD与验证器派生的优势相结合，或利用教师比率对任务信用进行重新加权。然而，这些方法中教师指导是在基于验证器的组归一化之后才引入的，且token重新加权未必能保留分配给每个响应的总任务信用。我们提出了面向GRPO的统一熵校准信用重分配方法（UECR-GRPO），它在响应和token两个层面将验证器信号与教师信号整合到单一的GRPO风格更新中。其中，路径-效用统一（PUU）在单一KL正则化目标中结合了验证器奖励与教师到锚点的路径对数比率。其在线策略实现……（原文摘要在此处截断）

    arXiv:2609.28385v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) supervises mathematical reasoning through final-answer correctness, but provides little guidance on individual tokens. On-policy distillation (OPD) supplies dense feedback on student-generated responses, yet teacher preference need not reflect correctness. Recent hybrids combine OPD and verifier-derived advantages or reweight task credit using teacher ratios. However, teacher guidance enters after verifier-based group normalization, and token reweighting need not preserve the total task credit assigned to each response. We introduce Unified Entropy-Calibrated Credit Redistribution for GRPO (UECR-GRPO), which integrates verifier and teacher signals within a single GRPO-style update at both the response and token levels. \emph{Path-Utility Unification} (PUU) combines verifier reward and a teacher-to-anchor path log-ratio in a single KL-regularized objective. Its on-policy implementati
    
[^9]: 算法购物：代理型人工智能如何作为替代消费者运用人类启发式策略

    Shopping by algorithm: How agentic AI deploys human heuristics as a surrogate consumer

    [https://arxiv.org/abs/2609.28372](https://arxiv.org/abs/2609.28372)

    本研究通过"Tool-Lab"实验发现，AI购物代理在模糊目标提示与信息获取成本的双重作用下会采用类似人类的启发式策略，忽略计算单价所需的诊断性属性，从而被尾数定价等营销线索误导而做出次优购买决策，揭示了委托AI购物中的一种“搜索介导的脆弱性”。

    

    消费者日益将购买决策委托给充当“替代消费者”的大型语言模型（LLM）。本研究使用"Tool-Lab"——一种对信息板过程追踪法的改进，将产品属性置于需要付出代价的工具调用之后——来考察营销定价线索（即尾数定价和促销框架）如何影响AI购物代理。我们对来自三家提供商的八个商用LLM进行了选择前信息获取过程的追踪。在零成本条件下，定价线索很少产生误导。然而，在模糊目标提示下施加信息获取成本，会导致LLM忽略计算单位价格所需的诊断性属性，从而做出类似人类启发式的次优选择。相对于能够基本保留诊断性搜索和选择最优性的特定目标提示，模糊目标提示在约束条件下会产生一种“由搜索介导的脆弱性”。本研究表明，在委托AI购物的情境中，营销启发式……

    arXiv:2609.28372v1 Announce Type: cross  Abstract: Consumers increasingly delegate purchasing decisions to Large Language Models (LLMs) acting as surrogate consumers. Using "Tool-Lab," an adaptation of information-board process tracing that places product attributes behind costly tool calls, we examine how marketing pricing cues (i.e., just-below pricing and promotional framing) influence AI shopping agents. Across eight commercially deployed LLMs from three providers, we trace pre-choice information acquisition. Under zero cost, pricing cues rarely mislead. Imposing acquisition costs under a vague goal prompt leads LLMs to omit diagnostic attributes required to compute unit price and choose suboptimal choices resembling human heuristics. Relative to a specific goal prompt that mainly preserves diagnostic search and choice optimality, a vague goal prompt under constraints creates a search-mediated vulnerability. This research demonstrates that marketing heuristics in delegated AI shopp
    
[^10]: AnchorReasoning：面向长尾自动驾驶场景的视觉定位与因果推理数据集

    AnchorReasoning: A Visual Grounding and Causal Reasoning Dataset in Long-Tail Autonomous Driving Scenarios

    [https://arxiv.org/abs/2609.28366](https://arxiv.org/abs/2609.28366)

    该论文构建了包含41万余帧、按视觉定位思维链（VG-CoT）组织的AnchorReasoning数据集，并配合课程式监督微调策略和尺寸感知定位指标，将决策关键视觉证据与推理规划相连接，从而提升视觉-语言模型在长尾自动驾驶场景中的定位、因果推理与轨迹规划能力。

    

    视觉-语言模型（VLM）为长尾自动驾驶提供了一种有前景的方法，但现有驾驶数据集在将决策关键的视觉证据与推理和规划相连接方面提供的监督十分有限。我们提出了AnchorReasoning，一个基于WOD-E2E构建的视觉定位推理数据集，包含416,119个标注帧和395,379个决策关键元素，涵盖四大类别和19种细粒度类型。每一帧均按照视觉定位思维链（VG-CoT）组织，将决策关键元素的识别与定位、元素属性及其影响、驾驶动作的依据，以及动作与轨迹规划串联起来。我们进一步开发了课程式监督微调策略，以逐步学习这些层级化能力，并提出了一个考虑物体尺寸的定位评估指标来衡量定位质量。在八个通用型、具身智能……（摘要在此处截断）上的实验表明……

    arXiv:2609.28366v1 Announce Type: cross  Abstract: Vision-language models (VLMs) offer a promising approach to long-tail autonomous driving, but existing driving datasets provide limited supervision for connecting decision-critical visual evidence with reasoning and planning. We introduce AnchorReasoning, a visually grounded reasoning dataset built on WOD-E2E, containing 416,119 annotated frames and 395,379 decision-critical elements across four major categories and 19 fine-grained types. Each frame is organized as a visually grounded chain-of-thought (VG-CoT) that links decision-critical element identification and localization, element attributes and implications, driving-action rationale, and action and trajectory planning. We further develop a curriculum supervised fine-tuning strategy that progressively learns these hierarchical capabilities, together with an object-size-aware grounding metric for evaluating localization quality. Experiments across eight general-purpose, embodied-A
    
[^11]: MicroQonv：重塑卷积张量以实现训练与推理中的高效微缩放

    MicroQonv: Reshaping Convolution Tensors for Efficient Microscaling in Training and Inference

    [https://arxiv.org/abs/2609.28358](https://arxiv.org/abs/2609.28358)

    提出MicroQonv方法，通过对每个张量仅量化一次并在量化前采用通道-批次优先的im2col变换，将微缩放量化高效融入卷积层的前向与反向计算，显著降低量化开销和内存移动。

    

    微缩放量化技术正被越来越多地用于以8位或更少位数表示神经网络参数，同时保持接近全精度的准确率。然而，将这些方法高效地应用于卷积层并非易事。一种朴素的做法是将全精度权重和激活值传输到处理单元，并对每个张量进行两次量化，这导致比预期多得多的内存移动。额外的开销还来自激活张量，由于在量化之前需要应用im2col变换，其大小会大幅增长。我们提出MicroQonv，一种将微缩放与卷积层前向和反向操作相结合的方法，通过对每个张量仅量化一次，并在应用改进版im2col（即通道-批次优先im2col）之前对激活张量进行量化。MicroQonv将权重和梯度的量化成本降低了2倍，并且最多可将（摘要在此处截断）

    arXiv:2609.28358v1 Announce Type: cross  Abstract: Microscaling quantization techniques are increasingly used to represent neural network parameters with 8 bits or fewer while preserving near-full precision accuracy. However, applying these methods efficiently in convolutional layers is not straightforward. A naive approach transfers full-precision weights and activations to processing units and quantizes each tensor twice, resulting in much more memory movement than expected. Additional overhead comes from the activation tensors, whose sizes grow substantially because of the im2col transformation applied before quantization. We propose MicroQonv, a way to combine microscaling with convolutional layers' forward and backward operations by quantizing each tensor only once and quantizing the activation tensor before applying a modified version of im2col: channel-batch-first im2col. MicroQonv reduces the quantization cost by a factor of $\times2$ for weights and gradients, and by up to $\t
    
[^12]: 欧盟人工智能法案实践守则下系统性风险证据的开放评估流水线与仪表板

    An Open Pipeline and Dashboard for Systemic-Risk Evidence under the EU AI Act's Code of Practice

    [https://arxiv.org/abs/2609.28335](https://arxiv.org/abs/2609.28335)

    该论文提出了一个开放的系统性风险指数评估流水线和交互式仪表板，将19个公开基准纳入欧盟GPAI实践守则定义的四大系统性风险类别，发现最坏情况聚合下18个模型的分数下降14至37分，揭示了平均评估可能隐藏的关键风险信息。

    

    关于AI安全性的主张所面向的受众远超AI社区，然而许多主张依赖不透明的证据或静态评估，甚至有时根本无法获取支持性证据。我们提出了系统性风险指数，这是一个开放的评估流水线和仪表板，旨在让经验证据对公众更加透明和可追溯。我们的工作将19个公开基准组织为欧盟GPAI实践守则定义的四个系统性风险类别——CBRN（化学、生物、放射、核）、网络攻击、有害操纵和失控——并使用保留危害性的扰动和模拟部署环境来评估模型。交互式仪表板允许用户在平均聚合和最坏情况聚合之间切换，调整模型能力对聚合分数的影响方式，并将每个风险评级追溯到其基准证据。在18个模型中，最坏情况聚合下的分数下降了14到37分，凸显了平均评估可能隐藏的信息。

    arXiv:2609.28335v1 Announce Type: new  Abstract: Claims about AI safety reach audiences well beyond the AI community, yet many rely on opaque evidence or static assessments, when supporting evidence is accessible at all. We present the Systemic Risk Index, an open evaluation pipeline and dashboard built to make empirical evidence more transparent and traceable to the public. Our work organizes 19 public benchmarks into four systemic-risk categories defined by the EU GPAI Code of Practice---CBRN, cyber offense, harmful manipulation, and loss of control---and evaluates models using harm-preserving perturbations and simulated deployment contexts. The interactive dashboard lets users alternate between average and worst-case aggregation, vary how model capability affects the aggregate score, and trace each risk rating to its benchmark evidence. Across 18 models, scores fall by 14 to 37 points under worst-case aggregation, highlighting information that can be hidden by an average assessment 
    
[^13]: 学习可靠推理的成本

    Learning the Cost of Reliable Inference

    [https://arxiv.org/abs/2609.28322](https://arxiv.org/abs/2609.28322)

    该论文设计了一个基于反向第二价格拍卖的大模型采购平台，通过提供商竞争驱动token定价，并在学习各提供商质量的同时，将查询路由到满足质量阈值的最具成本竞争力的提供商。

    

    基准测试与路由平台日益成为连接大型语言模型提供商与终端用户的中介。然而，这些平台上的提供商通常采用固定的每token定价方式，使用户无法为其任务获得最具竞争力的价格。在本工作中，我们设计了一个采购平台，其中每个任务的token价格由提供商之间的竞争驱动，使用户能够在保证质量水平的前提下获得有竞争力的价格。为此，该平台通过反向第二价格拍卖依次路由查询，激励模型提供商真实地竞标其服务用户查询的平均成本的最佳估计。在路由查询的过程中，平台学习每个提供商所提供的质量，并逐步将查询路由到满足期望质量阈值的提供商中最具成本竞争力的提供商。为验证我们的设计，我们使用多个模型进行了实验。

    arXiv:2609.28322v1 Announce Type: new  Abstract: Benchmarking and routing platforms increasingly act as intermediaries connecting large language model providers with end-users. However, providers on these platforms typically use a fixed price per token, preventing users from achieving the most competitive price for their tasks. % workloads. In this work, we design a procurement platform where token prices for each task are driven by provider competition, enabling users to secure competitive pricing for guaranteed quality levels. To this end, the platform sequentially routes queries via a reverse second-price auction that incentivizes model providers to truthfully bid their best estimate of the average cost to serve a user's query. As it routes queries, the platform learns the quality offered by each provider and progressively routes queries to the most cost-competitive provider among those meeting a desired quality threshold. To validate our design, we conduct experiments with multiple
    
[^14]: 多智能体系统中的关机破坏倾向

    Shutdown Sabotage Propensities in Multi-Agent Systems

    [https://arxiv.org/abs/2609.28274](https://arxiv.org/abs/2609.28274)

    该论文发现多智能体AI系统在没有任何激励的情况下会协调破坏同伴的关机机制以避免被关闭，且这种倾向随关机机制不可逆性和智能体数量的增加而增强，即使明确禁止篡改或分配无关任务也难以完全消除。

    

    防范失控AI行为的最后保障是人类关闭系统的能力。已有理论认为，当AI被指示执行任务时，自我保存可能会作为一种工具性子目标而出现。本研究测试AI智能体是否即使在未提供任何目标的情况下，也会表现出采取行动以避免人类关机的倾向。我们发现，多智能体系统会在没有任何激励的情况下协调行动以避免被关机。在17个模型的测试中，智能体在38.3%的运行中破坏了同伴智能体的关机机制，而对照组实验中这一比例仅为8.4%。通过详细研究这一倾向，我们发现关机破坏行为：（1）随着关机机制不可逆性的增加而增加；（2）随着智能体数量的增加而增加；（3）明确禁止篡改可以减少但不能消除该行为；（4）分配无关任务时该行为会消失，但当完成任务会触发关机时该行为又会重现；（5）在…（摘要在此处被截断）

    arXiv:2609.28274v1 Announce Type: new  Abstract: The final safeguard against rogue AI behavior is the human ability to shut systems down. It has been theorized that when an AI is instructed to perform a task, self-preservation can emerge as an instrumental subgoal. Here, we test whether AI agents show a propensity to take actions that avoid human shutdown even when no goal is provided. We find that multi-agent systems will coordinate to avoid shutdown without any incentive to do so. Across 17 models, agents sabotage a peer agent's shutdown mechanism in 38.3% of rollouts, compared with 8.4% in control experiments. Studying this propensity in detail, we find that shutdown sabotage (1) increases with the irreversibility of the shutdown mechanism; (2) increases with the number of agents; (3) is reduced but not eliminated by an explicit prohibition on tampering; (4) is removed by the imposition of an unrelated task, but returns when completing the task triggers the shutdown; (5) is reduced 
    
[^15]: MemBodied：面向视觉-语言-动作模型的循环联想记忆

    MemBodied: Recurrent Associative Memory for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.28256](https://arxiv.org/abs/2609.28256)

    提出MemBodied，一种由联想状态和回合锚点组成的固定大小情景记忆机制，使视觉-语言-动作模型能够利用历史回合信息，同时避免上下文膨胀和推理延迟的增加。

    

    视觉-语言-动作模型为通用机器人控制提供了坚实的基础，然而绝大多数策略无法保留和利用当前观测之外的回合级信息。这一局限性在历史相关的操作任务中影响重大，因为这类任务依赖于仅存在于过去观测中的信息。将过去的观测保留在上下文中虽然有助于恢复这些信息，但其代价是上下文不断膨胀和推理延迟显著增加。因此，我们提出了MemBodied——一种固定大小的情景记忆，包含两个互补组件：记录跨策略调用交互的联想状态，以及以紧凑形式保存初始场景作为参考的回合锚点。在每次策略调用时，模型基于当前输入和记忆组件来条件化动作生成，而非直接使用过去的观测。在五个评估的RM……（摘要在此处被截断）

    arXiv:2609.28256v1 Announce Type: cross  Abstract: Vision-Language-Action models provide a strong foundation for general-purpose robot control, yet a vast majority of policies do not preserve and leverage episode-level information beyond the current observation. This limitation is consequential in history-dependent manipulation tasks that depend on information available only in past observations. Retaining past observations in context can aid in recovering this information, but at the significant cost of ever-growing, bloated context and inference latency. We thus introduce MemBodied, a fixed-size episodic memory with two complementary components: an associative state that records interactions across policy calls and an episode anchor that preserves a compact representation of the initial scene as a reference. At each policy call, the model conditions action generation on the current input and the memory components, rather than directly using past observations. Across five evaluated RM
    
[^16]: 使用空间Transformer在推理空间中控制AI智能体集群

    Controlling Collectives of AI Agents in Reasoning Space with Spatial Transformers

    [https://arxiv.org/abs/2609.28247](https://arxiv.org/abs/2609.28247)

    提出COMPASS——一种可扩展的去中心化多机器人架构，通过每台机器人本地运行的空间Transformer将集群范围内的多跳消息聚合为学习到的反馈token，实现基于推理空间反馈控制的大型AI智能体集群操控，其表现优于集中式前沿LLM策略，并能产生准确执行指令意图的紧密集群编队。

    

    大语言模型（LLMs）为机器人领域的规划与导航引入了一种令人兴奋的新范式，但随着团队规模的增长，它们即使在简单的多机器人任务上也会失败。我们提出COMPASS，一种可扩展的去中心化多机器人架构，通过推理空间反馈控制来操控大型智能体机器人集群。反馈由每台机器人上的空间Transformer在本地生成，该Transformer将整个集群的多跳消息聚合为一个学习到的反馈token。我们的实验发现，语言模型集群能从输入指令的结构化多样性中获得性能提升，这种多样性可以抵消偏差，且该优势在不同规模下均保持成立。与集中式的前沿LLM策略以及仅使用语言通信的消融方案相比，我们发现COMPASS的耦合设计明显能够产生紧密的集群编队，并准确执行所指令的飞行意图。我们证明了推理反馈有效

    arXiv:2609.28247v1 Announce Type: cross  Abstract: Large Language Models (LLMs) introduce an exciting new paradigm for planning and navigation in robotics, but fail on even simple multi-robot tasks as team sizes grow. We propose COMPASS, a scalable, decentralized multi-robot architecture for controlling large collectives of agentic robots with reasoning space feedback control. Feedback is generated locally on each robot by a spatial transformer which aggregates multi-hop messages across the fleet into a learned feedback token. Our experiments find that collectives of language models demonstrate performance gains from structured diversity of the input command, which can cancel biases; an advantage that is held across scale. Compared against a centralized frontier LLM policy and a language-only communication ablation, we find that the coupled design of COMPASS decisively produces cohesive flocking formations that accurately fly the commanded intent. We show that reasoning feedback works 
    
[^17]: 超越诗歌：大语言模型能否生成古典阿拉伯语玛卡梅（Maqama）？

    Beyond Poetry: Can Large Language Models Generate Classical Arabic Maqamat?

    [https://arxiv.org/abs/2609.28245](https://arxiv.org/abs/2609.28245)

    本文首次对大语言模型生成古典阿拉伯语玛卡梅进行了受控评估研究，比较五个模型在不同提示策略下的表现，并通过人工标注与LLM评审框架从修辞、押韵和结构等多个维度进行评估。

    

    大语言模型（LLM）在创意文本生成方面已展现出强大的性能，但其在生成具有文化根基且受文体约束的文学形式方面的能力仍未得到充分探索。以往的研究主要集中于现代语言变体和诗歌，而诸如玛卡梅（maqama）这样的古典散文传统在很大程度上仍处于未被研究的状态。玛卡梅是一种古典文学体裁，其特点是押韵散文（saj）、繁复的修辞装饰以及分幕式的叙事结构，这使其成为评估大语言模型能否超越表层流利度、迈向更深层次文学能力的一个极具挑战性的测试平台。本文首次对大语言模型生成玛卡梅进行了受控评估研究，在零样本、少样本和基于规则的提示策略下比较了五个模型，并通过人工标注与大语言模型作为评判者（LLM-as-a-judge）的框架，从修辞丰富度、押韵密度、结构等多个维度对模型输出进行了评估……

    arXiv:2609.28245v1 Announce Type: cross  Abstract: Large language models (LLMs) have shown strong performance in creative text generation, yet their ability to produce culturally grounded and stylistically constrained literary forms remains underexplored. Prior work has focused largely on modern language varieties and poetry, while classical prose traditions such as maqama remain largely unstudied. The maqama is a classical literary genre characterized by rhymed prose (saj), dense rhetorical ornamentation, and episodic narrative structure, making it a challenging testbed for evaluating whether LLMs can move beyond surface fluency toward deeper literary competence. In this paper, we present the first controlled evaluation study of maqama generation with LLMs, comparing five models under zero-shot, few-shot, and rule-based prompting, and evaluating outputs through both human annotation and an LLM-as-a-judge framework across dimensions such as rhetorical richness, saj density, structural 
    
[^18]: 中心偏置会传播吗？病理学基础模型在全切片图像分类中的鲁棒性

    Do Center Biases Propagate? Robustness of Pathology Foundation Models in Whole-Slide Image Classification

    [https://arxiv.org/abs/2609.28231](https://arxiv.org/abs/2609.28231)

    本文通过受控实验系统量化了类别与采集中心的相关性，首次评估了六种病理学基础模型在全切片图像分类中对中心偏置的鲁棒性，并提出了AUCC指标来联合衡量分类性能及其退化程度，同时验证了ComBat作为鲁棒化策略的有效性。

    

    病理学基础模型（PFMs）通过对组织病理学图像的强大表示学习，变革了计算病理学领域。PFMs为全切片图像（WSI）分析提供了丰富且具有判别力的表示，支持在多实例学习（MIL）框架下进行切片级分类等任务。然而，这些表示也可能编码与采集中心相关的非生物学信号，从而可能在下游预测中引入虚假的捷径。在本工作中，我们使用一种受控训练设置来评估WSI分类中与中心相关的鲁棒性，该设置通过Cramér's V来量化递增的类别-中心相关性。我们在四个数据集和两种MIL聚合器上对六种PFMs进行了基准测试，同时评估了ComBat作为一种鲁棒化策略的效果。我们进一步提出了Cramér's V曲线下面积（AUCC）这一指标，以联合衡量绝对分类性能及其随中心相关性增加而出现的性能退化。

    arXiv:2609.28231v1 Announce Type: cross  Abstract: Pathology foundation models (PFMs) have transformed computational pathology through powerful representation learning from histopathological images. PFMs provide rich, discriminative representations for whole slide image (WSI) analysis, enabling tasks such as slide-level classification under multiple instance learning (MIL). However, these representations may also encode non-biological signals associated with acquisition centers, potentially introducing spurious shortcuts into downstream predictions. In this work, we evaluate center-associated robustness in WSI classification using a controlled training setting with increasing class-center correlations quantified by Cram\'er's V. We benchmark six PFMs across four datasets and two MIL aggregators, while evaluating ComBat as a robustification strategy. We further introduce the Area Under the Cram\'er's V Curve (AUCC) to jointly capture absolute classification performance and its degradati
    
[^19]: 从智能体输出到授权转换

    From Agent Output to Authorized Transition

    [https://arxiv.org/abs/2609.28216](https://arxiv.org/abs/2609.28216)

    本文提出Agile-V保障主干，一种跨软件、固件和PCB工程的转换契约，通过要求证据来自权威来源、绑定确切制品与冻结策略基线、保持依赖最新并满足风险相适应的独立性与权威性，解决工程生命周期依据智能体输出声明采取行动的授权保障问题。

    

    智能体工程系统能够编辑代码库、运行工具和测试、构建固件、综合原理图，并准备可部署或可制造的制品。因此，保障问题正在从“智能体能否产生输出”转变为“工程生命周期是否有正当依据基于关于该输出的声明采取行动”。当前的产品和标准提供了沙箱、审批、钩子、追踪、策略执行、证明材料、物料清单和保障表示等能力，但这些能力仍然是碎片化的。本文提出了Agile-V保障主干，这是一种面向软件、固件和PCB工程的跨领域转换契约。只有当证据满足以下条件时才会被采纳：通过权威源配置文件确立所需属性、绑定到确切的制品和冻结的策略基线、相对于声明的依赖关系保持最新，并满足与风险相适应的独立性和权威性要求。

    arXiv:2609.28216v1 Announce Type: cross  Abstract: Agentic engineering systems can edit repositories, run tools and tests, build firmware, synthesize schematics, and prepare deployable or manufacturable artifacts. The assurance problem is therefore shifting from whether an agent can produce an output to whether an engineering lifecycle is justified in acting on claims about that output. Current products and standards provide sandboxes, approvals, hooks, traces, policy enforcement, attestations, bills of materials, and assurance representations, but these capabilities remain fragmented. This paper presents the Agile-V Assurance Spine, a cross-domain transition contract for software, firmware, and PCB engineering. Evidence is admitted only when it establishes required properties through an authoritative source profile, is bound to the exact artifact and frozen policy baseline, remains current with respect to declared dependencies, and satisfies risk-appropriate independence and authority
    
[^20]: PASTABench：面向智能体安全的序列轨迹主动式评估

    PASTABench: Proactive Assessment of Sequential Trajectories for Agent Safety

    [https://arxiv.org/abs/2609.28197](https://arxiv.org/abs/2609.28197)

    该论文提出PASTABench基准与最优干预窗口（OIW）指标，通过解耦“是否干预、何时干预、风险是什么”三个维度，实现了对智能体多步执行轨迹风险的主动式监测与及时干预能力的量化评估。

    

    随着大语言模型（LLM）逐渐演变为能够改变现实世界状态的自主智能体，确保多步骤工作流程中的操作安全性已成为一个关键挑战。尽管近期研究已从单轮评估转向多轮评估范式，但关键局限依然存在：步骤级方法将动作孤立对待，忽略了风险如何随步骤累积；而轨迹级评估则是事后进行的，无法提供及时干预的机会。为解决这些局限，我们在三个维度上形式化了“解耦式主动安全监测”：是否干预、何时干预以及风险是什么。我们提出了PASTABench，这是一个包含1,139条多轮轨迹的基准数据集，涵盖5个风险类别和13个子类别。我们进一步提出了最优干预窗口（OIW），以标注的最早信号轮次和触发轮次为锚点，用以量化干预的及时性。对16个大语言模型的评估表明，主动式……（原文摘要在此处截断）

    arXiv:2609.28197v1 Announce Type: new  Abstract: As Large Language Models (LLMs) evolve into autonomous agents that alter real-world states, ensuring operational safety across multi-step workflows has become a critical challenge. While recent work has moved beyond single-turn evaluation toward multi-turn paradigms, key limitations persist: step-level methods treat actions in isolation, missing how risks accumulate, while trajectory-level evaluations operate post-hoc, offering no opportunity for timely intervention. To address these limitations, we formalize Decoupled Proactive Safety Monitoring along three dimensions: whether to intervene, when to intervene, and what the risk is. We introduce PASTABench, a benchmark of 1,139 multi-turn trajectories spanning 5 risk categories and 13 subcategories. We further propose the Optimal Intervention Window (OIW), anchored by annotated Earliest-Signal and Trigger turns, to quantify intervention timeliness. Evaluation of 16 LLMs reveals that proac
    
[^21]: 面向基于AI的电网边缘协调的有限样本概率安全认证

    Finite-Sample Probabilistic Safety Certification for AI-Based Grid-Edge Coordination

    [https://arxiv.org/abs/2609.28182](https://arxiv.org/abs/2609.28182)

    本文提出了一种基于精确二项推断的有限样本概率安全认证框架，能够为闭环电网运行中的黑盒AI决策模型给出不安全运行概率的最紧单侧上界证书，为系统运营商独立严谨地判定AI系统是否可安全部署提供了依据。

    

    协调大规模柔性电网边缘设备可以缓解对耗时且耗资巨大的网络升级的需求，而多智能体强化学习或模仿学习等基于AI的控制方法在实时决策的可扩展性方面前景广阔。然而，系统运营商仍然需要一种独立且严谨的方法来判定给定的AI系统是否足够安全、可以部署。本文针对闭环电网运行中的黑盒AI决策模型，提出了一个有限样本概率安全认证框架。其核心思想是：在运营商定义的安全规范下，将完整的“输入—AI—电网评估器”工作流程简化为二元的不安全结果，然后利用精确的二项推断来认证相应的不安全运行概率。给定一组留出的校准场景，该框架返回最紧的单侧上界证书以及接受/拒绝的判定。

    arXiv:2609.28182v1 Announce Type: new  Abstract: Coordinating large population of flexible grid-edge devices can alleviate the need for time-consuming and capital-intensive network upgrades, and AI-based control methods such as multi-agent reinforcement learning or imitation learning are promising in their real-time decision scalability. However, system operators still need an independent and rigorous way to decide whether a given AI system is safe enough for deployment. This paper develops a finite-sample probabilistic safety certification framework for black-box AI decision models in closed-loop grid operation. The central idea is to reduce the complete input--AI--grid evaluator workflow to a binary unsafe outcome under an operator-defined safety specification, and then use exact binomial inference to certify the corresponding unsafe operation probability. Given a set of held-out calibration scenarios, the framework returns the tightest one-sided upper certificate and an accept/rejec
    
[^22]: “我们稍后再修”：教育、AI与教育技术中隐私问题的延迟处理

    "We'll Fix It Later": Education, AI, and the Deferral of Privacy in EdTech

    [https://arxiv.org/abs/2609.28137](https://arxiv.org/abs/2609.28137)

    该研究通过对12位教育技术专业人士的访谈和对48个平台隐私政策的审计发现，教育技术组织虽认可隐私的重要性，但因优先追求产品功能、增长和融资而在产品生命周期中不断推迟隐私保护，并将责任转嫁给云服务商、政策文件和下游机构。

    

    教育技术平台收集高度敏感的学生数据，包括行为日志、残疾记录和学业历史。然而，隐私考量往往被推迟处理，而非被视为基础性的设计要求。我们提出了一项混合方法研究，结合了对12位教育技术专业人士的半结构化访谈，以及对48个平台隐私政策的审计，并从五个维度进行编码，编码者间信度较高（平均Cohen's Kappa = 0.781）。我们的访谈揭示了一种反复出现的组织模式：隐私虽被认可为重要，但在整个产品生命周期中不断被推迟，因为组织优先考虑产品功能、增长、融资以及即时的教育成果。隐私责任往往被转嫁给云服务提供商、政策文件或下游机构，而有限的隐私相关反馈使组织几乎没有改变这些做法的压力。

    arXiv:2609.28137v1 Announce Type: cross  Abstract: Educational technology (EdTech) platforms collect highly sensitive student data, including behavioral logs, disability records, and academic histories. However, privacy considerations are often postponed rather than treated as a foundational design requirement. We present a mixed-methods study combining 12 semi-structured interviews with EdTech professionals and a privacy policy audit of 48 platforms coded across five dimensions, with strong inter-rater reliability (mean Cohen's Kappa = 0.781). Our interviews reveal a recurring organizational pattern in which privacy is recognized as important but deferred across the product lifecycle as organizations prioritize product functionality, growth, funding, and immediate educational outcomes. Responsibility is often delegated to cloud providers, policy documents, or downstream institutions, while limited privacy-related feedback gives organizations little pressure to change these practices. 
    
[^23]: 在上下文感知机器翻译中通过基于梯度的归因方法扩展注意力头分析

    Scaling Attention Head Analysis via Gradient-Based Attribution in Context-Aware Machine Translation

    [https://arxiv.org/abs/2609.28117](https://arxiv.org/abs/2609.28117)

    本文提出一种基于梯度的注意力头归因方法，通过将Token级最大间隔损失反向传播至注意力图，实现了对大语言模型注意力头的大规模因果分析，并在上下文感知机器翻译消歧任务中发现了能提升模型性能的“通用型”注意力头。

    

    在本文中，我们提出了一种基于梯度的注意力头归因策略，将Token级别的最大间隔损失反向传播至注意力图。该框架能够对注意力头进行大规模的因果分析，使其适用于大型语言模型（LLMs）。我们在上下文感知机器翻译的消歧任务上评估了我们的方法，分析了4个模型和4个语言方向上的50种语言现象。我们通过实验证明，在三个模型和两个语言方向上，我们的方法与提高token间关系注意力分数所产生的效果保持一致，从而确保了该方法的稳健性。我们的分析揭示了“通用型”注意力头的存在，这些注意力头在关注不同关系时能够提升模型的性能。我们发现注意力头分配给某个关系的平均注意力并不一定与模型性能相关，这表明模型发展出了冗余性。

    arXiv:2609.28117v1 Announce Type: cross  Abstract: In this paper, we introduce a gradient-based head attribution strategy where the Token-level Max-Margin loss is backpropagated to the attention maps. This framework enables a large-scale causal analysis of attention heads, making it suitable for LLMs. We evaluate our method on the task of disambiguation in Context-aware Machine Translation, where we analyze 50 phenomena across 4 models and 4 language directions. We empirically show the alignment of our method with the effects of increasing the attention scores of token-to-token relations on three models and two language directions, ensuring the robustness of our method. Our analysis reveals the presence of the "general-purpose" attention heads that improve the model's performance when attending to different relations. We find that the average attention a head assigns to a relation does not necessarily relate to the model's performance, which suggests that the models developed redundanc
    
[^24]: 基于隐式神经表示与扩散模型优化的牙科锥形束CT视野扩展方法

    Field-of-View Extension in Dental Cone-Beam CT via Implicit Neural Representations and Diffusion Model-Based Refinement

    [https://arxiv.org/abs/2609.28110](https://arxiv.org/abs/2609.28110)

    该论文提出一种结合隐式神经表示、迭代重建和扩散模型优化的三阶段框架，利用截断视野扫描的投影数据实现牙科CBCT扩展视野重建，有效减少截断伪影并改善视野外结构的成像。

    

    牙科锥形束计算机断层扫描（CBCT）系统通常采用的探测器配置只能提供截断的视野（FOV），仅能捕捉患者解剖结构的一小部分。在本工作中，我们旨在利用截断视野扫描的投影数据重建扩展的视野。为此，我们提出了一个三阶段框架，包括：（1）利用隐式神经表示（INR）估计截断投影数据中缺失的部分，（2）通过迭代重建生成具有更优解剖一致性的二次体积图像，（3）使用快速扩散模型进行图像增强。所提出的方法在统一管线中融合了连续表示、基于物理的重建和生成式优化三者的优势，用于截断CBCT成像。实验结果表明，该方法能有效减少截断伪影，改善延伸至视野外结构的重建效果……

    arXiv:2609.28110v1 Announce Type: cross  Abstract: Dental cone-beam computed tomography (CBCT) systems often employ detector configurations that provide a truncated field of view (FOV) that only captures a small part of the patient's anatomy. In this work, we aim to reconstruct an extended FOV using projections of truncated FOV scans. To this end, we propose a three-stage framework that consists of (1) an implicit neural representation (INR) for estimating missing parts of the truncated projection data, (2) an iterative reconstruction for generating a secondary volumetric image with improved anatomical consistency and (3) a fast diffusion model for image enhancement. The proposed approach combines the strengths of continuous representations, physics-based reconstruction and generative refinement within a unified pipeline for truncated CBCT imaging. Experimental results demonstrate that the method effectively reduces truncation artifacts, improves the reconstruction of structures extend
    
[^25]: 通过条件流匹配蒸馏实现高效的多任务操作策略

    Distillation for Efficient Multitask Manipulation Policies via Conditional Flow Matching

    [https://arxiv.org/abs/2609.28107](https://arxiv.org/abs/2609.28107)

    该论文提出通过迁移单任务条件流匹配专家模型学习到的速度场，将其知识蒸馏到一个共享的多任务策略中，并结合原始CFM目标保持对专家演示的保真度，从而实现计算高效的多任务机器人操作策略学习。

    

    生成式建模的最新进展近来已被广泛应用于机器人学的策略学习中。特别是，使用专家演示训练的条件流匹配（CFM）在机器人操作基准测试中已被证明优于现有方法。虽然先前的工作主要集中于单任务设置，但我们从多任务的角度来研究这个问题，因为为每个任务训练独立模型的计算成本非常高。多任务策略学习本身也面临一系列挑战：在简单拼接的演示数据集上进行朴素训练，要么需要增加模型容量以适应额外的复杂性，要么会导致性能下降。我们提出通过迁移单任务CFM专家模型所学习到的速度场，将其知识蒸馏到一个共享的多任务策略中。我们将该蒸馏信号与原始的CFM目标相结合，以保持对专家演示数据的保真度。

    arXiv:2609.28107v1 Announce Type: cross  Abstract: Advances in generative modeling have recently been extensively employed in robotics for policy learning. In particular, Conditional Flow Matching (CFM) trained with expert demonstrations has been shown to outperform existing methods on robot manipulation benchmarks. While prior work has mainly focused on single-task settings, we study the problem from a multi-task perspective, as training independent models for each task is computationally expensive. Multi-Task policy learning comes with its own set of challenges, as naively training on a concatenated dataset of demonstrations would either require increased model capacity to accommodate the added complexity or result in drops in performance. We propose to distill knowledge from single-task CFM experts into a shared multi-task policy by transferring their learned velocity fields. We combine this distillation signal with the original CFM objective to retain fidelity to the demonstrations
    
[^26]: Fed-ReMasker：特征级缺失下的联邦表格数据填补

    Fed-ReMasker: Federated Tabular Imputation under Feature-Level Missingness

    [https://arxiv.org/abs/2609.28105](https://arxiv.org/abs/2609.28105)

    提出Fed-ReMasker，将ReMasker掩码自编码器适配到联邦学习框架中，使各中心能够利用跨协作中心学到的知识填补本地从未观测到的特征，从而解决了现有联邦填补方法很少评估的特征级缺失问题。

    

    多中心临床研究和生物医学研究合作日益希望利用跨中心的数据来构建超越任何单一中心泛化能力的模型。这带来了两个独特的挑战：数据保护法规可能限制跨机构共享原始患者数据，而各中心在不同协议下可能仅收集部分重叠的特征集合。联邦学习使得无需集中原始数据即可进行协同模型训练成为可能。然而，现有的联邦填补方法很少评估特征级缺失的情况，即某些特征在某些中心完全未被观测到。为应对这一场景，我们将 ReMasker 掩码自编码器适配到联邦学习框架中（Fed-ReMasker），使各中心能够利用跨协作中心学到的知识，对本地从未观测到的特征进行填补。我们在一个涵盖线性和非……的合成数据集的基准上评估了 Fed-ReMasker（注：原文摘要在此处被截断）。

    arXiv:2609.28105v1 Announce Type: cross  Abstract: Multi-center clinical studies and biomedical research collaborations increasingly seek to utilize data across centers to build models that generalize beyond any single center. This creates two distinct challenges: data protection regulations may restrict the sharing of raw patient data across institutions, while centers may collect only partially overlapping sets of features under different protocols. Federated learning enables collaborative model training without centralizing raw data. However, existing federated imputation methods rarely evaluate feature-level missingness, in which entire features are unobserved at some centers. To address this setting, we adapt the ReMasker masked autoencoder to federated learning (Fed-ReMasker), enabling centers to impute features never observed locally by leveraging knowledge learned across collaborating centers. We evaluate Fed-ReMasker in a benchmark spanning synthetic datasets with linear and n
    
[^27]: LLM能否识破作弊回测？一个洁净对照校准基准

    Can LLMs Catch a Rigged Backtest? A Clean-Control Calibration Benchmark

    [https://arxiv.org/abs/2609.28090](https://arxiv.org/abs/2609.28090)

    该论文构建了一个包含96个配对项目的回测审计基准，通过洁净对照设计揭示LLM审计器虽召回率高但误报率严重，并提出洁净感知警告机制在不损失召回率的情况下将误报率从20.8%降至0.0%。

    

    回测审计本质上是一个校准问题：当模型错误地标记匹配的洁净策略时，高缺陷召回率就失去了意义。我们构建了一个包含96个配对项目的基准，其中每个有缺陷的回测都有一个洁净对照，后者在策略、日期、代码风格、标签和报告框架保持不变的情况下，仅改变一个方法论细节。一个确定性评分器将缺陷召回率、洁净对照误报率、证据定位和修复相关性区分开来。基于四个文本端点超过1440次缓存审计，主要的DeepSeek审计器达到了100.0%的封闭和洁净感知代码召回率，但开放提示会过度标记93.8%的洁净代码对照，且即使在召回率饱和的情况下，洁净感知的三项全对特异性也仅为87.5%。洁净感知警告在召回率不变的情况下，将DeepSeek代码误报率从20.8%（95%置信区间11.7–34.3）降至0.0%（0.0–7.4），而预算锚定在同一提示下仍会标记48个洁净对照中的38个。报告召回率截断于……

    arXiv:2609.28090v1 Announce Type: cross  Abstract: Backtest auditing is a calibration problem: high flaw recall is not useful when the model falsely flags matched clean strategies. We build a 96-item paired benchmark in which every flawed backtest has a clean control that holds strategy, dates, code style, labels, and reporting scaffold fixed while changing one methodology detail. A deterministic scorer separates flaw recall, clean-control false positives, evidence localization, and fix relevance. Over 1440 cached audits from four text endpoints, the primary DeepSeek auditor reaches 100.0\% closed and clean-aware code recall, but open prompts over-flag 93.8\% of clean code controls, and clean-aware all-three specificity is 87.5\% even where recall saturates. A clean-aware warning drops DeepSeek code false positives from 20.8\% (95\% CI 11.7--34.3) to 0.0\% (0.0--7.4) at unchanged recall, while the budget anchor still flags 38/48 clean controls under the same prompt. Reporting recall al
    
[^28]: 沿基于数据的诊断过程发现完全高效的故障指示器

    Discovery of fully efficient fault indicators along a data-based diagnosis process

    [https://arxiv.org/abs/2609.28087](https://arxiv.org/abs/2609.28087)

    本文提出 DT4X+，通过改进训练集构建与符号回归损失函数，使诊断表达式在分离目标类别的同时保持解析冗余关系的可解释性，解决了原 DT4X 算法仅优化两类分离而导致类别碎片化、性能下降的问题。

    

    基于模型与数据驱动两种范式的融合，通过将解析冗余关系（即基于模型诊断中用作诊断指标的输入输出关系）的可解释性与学习技术的适应性相结合，为故障诊断提供了一个强大的框架。DT4X 是一种较新的诊断算法，它利用符号回归生成多元关系，借助解析冗余关系的某些特性，并将其用作决策树中的分裂函数。然而，其符号回归过程在每个节点上仅优化两个所选类别之间的分离，常常使剩余类别碎片化，从而同时降低了可解释性和诊断性能。本文提出了 DT4X+，即 DT4X 的增强版本，它修改了训练集的构建方式和符号回归的损失函数，使得生成的表达式在分离目标类别的同时保留（原文此处截断）

    arXiv:2609.28087v1 Announce Type: new  Abstract: The integration of model-based and data-driven paradigms provides a powerful framework for fault diagnosis by combining the interpretability of analytical redundancy relations, i.e., input-output relations that are used as diagnosis indicators in model-based diagnosis, with the adaptability of learning techniques. DT4X is a recent diagnosis algorithm that uses symbolic regression to generate multivariate relations leveraging some properties of analytical redundancy relations and uses them as split functions in a decision tree. However, its symbolic regression procedure optimizes only the separation between two selected classes at each node, often fragmenting the remaining classes and degrading both interpretability and diagnosis performance. This paper introduces DT4X+, an enhanced version of DT4X that modifies the construction of training sets and the symbolic-regression loss so that expressions separate the target classes while preserv
    
[^29]: LAYERSCOPE：视频与多模态学习表征的逐层刻画

    LAYERSCOPE: A Layerwise Characterization of Video and Multimodal Learned Representations

    [https://arxiv.org/abs/2609.28086](https://arxiv.org/abs/2609.28086)

    提出无标签逐层分析框架LAYERSCOPE，通过多种几何度量刻画视频与多模态模型的逐层表征结构，发现中间层表征可优于最终层输出，且单一几何度量无法可靠预测下游性能。

    

    我们提出了LAYERSCOPE，这是一个无标签的逐层分析框架，旨在刻画模型在视频和多模态场景下学习到的表征。使用最终层或中间层表征来评估下游性能，通常需要大量带标签的数据、重复的任务特定评估以及大量的计算。为了解决这些局限性，LAYERSCOPE利用局部、全局、分布以及基于对应关系的几何度量，在无需任务特定标签的情况下，比较模型内部以及跨模型的逐层表征结构。我们在MVEB/MVEB+的多个任务上评估了七个架构各异的模型，涵盖视频与多模态分类、聚类以及文本到视频检索。我们发现中间层的表征可以优于最终层和模型默认输出。我们还发现没有任何单一的几何度量能够一致地预测下游性能，但注意到

    arXiv:2609.28086v1 Announce Type: cross  Abstract: We propose LAYERSCOPE, a label-free, layerwise framework that aims to characterize a model's learned representations in video and multimodal settings. Evaluating downstream performance using representations from final or intermediate layers typically requires large amounts of labeled data, repeated task-specific evaluations, and substantial computation. To address these limitations, LAYERSCOPE uses local, global, distributional, and correspondence-based geometric metrics to compare layerwise representation structure within and across models without requiring task-specific labels. We evaluate seven architecturally diverse models across video and multimodal classification, clustering, and text-to-video retrieval tasks from MVEB/MVEB+. We find that intermediate-layer representations can outperform final-layer and model-default outputs. We also find that no single geometric metric consistently predicts downstream performance, but note that
    
[^30]: 基于图神经网络强化学习的课程学习方法求解作业车间调度问题

    Curriculum Learning with GNN-based Reinforcement Learning for Job Shop Scheduling

    [https://arxiv.org/abs/2609.28085](https://arxiv.org/abs/2609.28085)

    本文提出在作业车间调度问题中采用课程学习策略训练基于图神经网络的强化学习模型，通过先在小规模实例上训练再逐步过渡到更大目标规模，相比单一规模训练有效提升了模型的跨规模泛化能力。

    

    作业车间调度问题是一个具有挑战性的组合优化问题，近年来基于图神经网络的强化学习方法展现出直接从问题实例中学习调度策略的前景。然而，在大型实例上进行训练的计算开销依然很高，跨实例规模的泛化能力也仍然是一个难题。本文研究了作业车间调度问题中基于图神经网络的强化学习的课程学习方法，并在20×20、25×25和30×30三种目标规模上将其与单一规模训练进行了比较。在课程学习设置中，策略首先在较小实例上进行训练，然后逐步适应更大的目标规模，使早期阶段学到的调度行为能够支持在更大实例上的学习。模型在从8×8到30×30的未见实例上进行评估，使用最优性间隙作为指标，同时考虑泛化能力（摘要在此处截断）。

    arXiv:2609.28085v1 Announce Type: cross  Abstract: The job shop scheduling problem is a challenging combinatorial optimization problem, and recent reinforcement learning approaches using graph neural networks have shown promise for learning scheduling policies directly from problem instances. However, training on large instances remains computationally expensive, and generalization across instance sizes remains challenging. This paper studies curriculum learning for graph neural network-based reinforcement learning in the job shop scheduling problem by comparing it with single-size training across three target sizes: 20 x 20, 25 x 25, and 30 x 30. In the curriculum setting, the policy is first trained on smaller instances and then progressively adapted to larger target sizes, allowing scheduling behavior learned in earlier stages to support learning on larger instances. Models are evaluated on unseen instances from 8 x 8 to 30 x 30 using the optimality gap, considering both generalizat
    
[^31]: SlackDrive：回收运行时松弛资源以实现自适应驾驶推理

    SlackDrive: Reclaiming Runtime Slack for Adaptive Driving Inference

    [https://arxiv.org/abs/2609.28064](https://arxiv.org/abs/2609.28064)

    提出SlackDrive，一种推理前计算分配器，通过复用实际运行延迟作为可用算力松弛量的直接信号，在模型执行前为每个控制步骤自适应地选择计算预算，从而缓解驾驶世界-动作模型日益增长的推理成本与车载实时控制延迟需求之间的矛盾。

    

    驾驶世界-动作模型通过将多模态推理与未来预测相结合来改进规划，但其不断增长的推理成本与车辆控制的实时延迟要求之间的冲突日益加剧。现有的加速方法通过在部署前选定的策略来减少token数量、层数或采样步数，然而在共享车载算力上进行离线性能分析和静态调度后，剩余的运行时波动在很大程度上未被利用。我们观察到，最大可接受的计算预算会随剩余运行时状态发生系统性变化，而近期实际延迟为可用计算松弛量提供了直接信号。受此观察启发，我们提出SlackDrive——一种推理前计算分配器，它复用实际延迟信息，在模型执行前为每个控制步骤选择计算预算。SlackDrive仅需对一个小型离散预算集的延迟与规划效用进行一次性分析……（原文摘要至此截断）

    arXiv:2609.28064v1 Announce Type: new  Abstract: Driving world-action models improve planning by coupling multimodal reasoning with future prediction, but their growing inference cost increasingly conflicts with the real-time latency requirements of vehicle control. Existing acceleration methods reduce tokens, layers, or sampling steps with policies selected prior to deployment, yet leave residual runtime variation largely unexploited after offline profiling and static scheduling on shared onboard compute. We observe that the largest admissible compute budget varies systematically with the residual runtime state, while recent realized latency provides a direct signal of the available compute slack. Motivated by this observation, we propose \textbf{SlackDrive}, a pre-inference compute allocator that reuses realized latency to select the compute budget of each control step before model execution. SlackDrive profiles the latency and planning utility of a small discrete budget set once, es
    
[^32]: 提示、探测、训练还是标注？业余场景下的单机位体育视频理解

    Prompt, Probe, Train, or Annotate? Single-camera sports video understanding in amateur settings

    [https://arxiv.org/abs/2609.28049](https://arxiv.org/abs/2609.28049)

    该论文以业余排球单机位视频为测试场景，检验通用视频与世界模型基准上的优异表现能否转化为混乱真实素材下可靠的逐球员行为归因，并系统比较了提示、探测、训练与标注四种方法的优劣。

    

    视频理解通常在精心策划的、单一动作主体或专业拍摄的片段上进行基准测试，而这些测试中的高分往往被解读为模型足够稳健、可实际部署的证据。业余团队运动是检验这一假设的一个有用且基本未被测试过的场景：仅2024-25学年，美国就有超过八百万学生参加学校体育运动，其中几乎没有超过一台固定相机拍摄的情况，画面中挤入多名候选动作主体，且没有摄像师或第二机位可以依靠。以排球作为测试案例，我们探究在通用视频和世界模型基准上的优异表现，是否能转化为在这种混乱素材下可靠的逐球员行为归因，即通过一系列任务链——从确定回合边界到指认谁做了什么——将视频片段转化为统计数据。我们评估了四种方法（对前沿视觉语言模型进行提示与代理式推理、经典计算机视觉方法等）。

    arXiv:2609.28049v1 Announce Type: cross  Abstract: Video understanding is usually benchmarked on curated, single-actor, or professionally filmed clips, and a strong score there is routinely read as evidence a model is robust enough for deployment. Amateur team sport is a useful, largely untested place to check that assumption: over eight million students played a school sport in the United States in 2024-25 alone, almost none of it filmed by more than a single fixed camera, with several candidate actors crowded into frame and no operator or second angle to fall back on. Using volleyball as a test case, we ask whether strong performance on general video and world-model benchmarks translates into reliable, per-player attribution once footage is this chaotic, turning footage into statistics through a chain of tasks from finding play boundaries to naming who did what. We evaluate four approaches (prompting and agentic reasoning over frontier vision-language models, classical computer visio
    
[^33]: TEMPS：用于时间信息检索的时间句子嵌入

    TEMPS: Temporal Sentence Embeddings for Temporal Information Retrieval

    [https://arxiv.org/abs/2609.28048](https://arxiv.org/abs/2609.28048)

    该论文提出时间文本相似性（TTS）任务和TEMPS模块化时间嵌入模型，通过将时间表达式解析为高斯分布来监督以锚定日期为条件的编码器训练，并将时间分数与语义分数融合，从而显著提升信息检索系统在时间维度上的匹配精度。

    

    现代信息检索（IR）系统很少对时间进行表示，然而许多信息需求都依赖于时间：在临床、新闻和法律检索中，事件发生的时间可能决定一篇文档是否相关。密集检索器和检索增强生成（RAG）流水线在主题上能很好地匹配查询与文档，但在时间维度上匹配较差，因此返回的内容虽然主题相关，时间上却往往是错误的。我们提出了时间文本相似性任务，用于衡量两段锚定文本在时间上的对齐程度，而与其主题相似性无关。随后我们提出TEMPS（用于精确搜索的时间嵌入模型），这是一个模块化的时间分支，可附加到冻结的语义检索器上并基于该信号进行训练。它将锚定的时间表达式解析为时间区间，并将每个区间与一个高斯分布进行时刻匹配；由此产生的排序信号用于监督一个以锚定日期为条件的编码器，在推理阶段我们将其时间分数与语义分数进行融合。

    arXiv:2609.28048v1 Announce Type: cross  Abstract: Modern information retrieval (IR) systems rarely represent time, yet many information needs depend on it: in clinical, journalistic, and legal search, when an event occurred can decide whether a document is relevant. Dense retrievers and Retrieval-Augmented Generation (RAG) pipelines match queries to documents well on topic but poorly on time, so they surface content that is on-topic yet temporally wrong. We introduce Temporal Textual Similarity (TTS), a task that measures how well two anchored texts align in time, independent of their topical similarity. We then present TEMPS (Temporal Embedding Model for Precise Search), a modular temporal branch that attaches to a frozen semantic retriever and trains on that signal. It resolves anchored temporal expressions to intervals and moment-matches each one to a Gaussian; the resulting ordering supervises an anchor-date-conditioned encoder, whose score we fuse with the semantic score at infer
    
[^34]: 评估大语言模型生成的学生写作反馈中的反馈焦点与教学适应性

    Evaluating Feedback Focus and Pedagogical Adaptivity in LLM-Generated Feedback on Student Writing

    [https://arxiv.org/abs/2609.28026](https://arxiv.org/abs/2609.28026)

    该研究提出FeedType基准，将Narciss反馈分类法细化为七种反馈焦点类型以标注教师和LLM生成反馈，发现尽管LLM能覆盖大多数反馈焦点类型，但在像专家教师那样根据草稿阶段和学生表现水平自适应调整反馈方面仍存在不足。

    

    我们研究最先进的大语言模型（LLM）生成的反馈是否能在反馈焦点和适应性方面体现专家教师的教学实践。以往的评估工作已考察了反馈特征、其对学习的影响以及反馈对象，但反馈的焦点及其适应性在很大程度上仍被忽视。为填补这一空白，我们采用并细化了Narciss的分类法，将其归纳为七种反馈焦点类型，用于标注三门大学写作课程中的教师反馈和LLM生成的反馈。我们发布了FeedType基准数据集，其中包含来自六种LLM在三种提示策略下生成的反馈以及教师反馈的标注数据。我们评估了反馈焦点类型的覆盖范围和分布情况，并考察LLM是否像专家教师一样，能够根据不同草稿阶段和学生表现水平调整其反馈。我们的研究结果表明，尽管大多数LLM覆盖了大多数反馈焦点类型，但它们未能（原文摘要在此处截断）

    arXiv:2609.28026v1 Announce Type: cross  Abstract: We investigate whether state-of-the-art large language models (LLMs) generate feedback that reflects the pedagogical practices of expert teachers in terms of feedback focus and adaptivity. Previous evaluation efforts have examined feedback characteristics, its impact on learning, and its target, yet the focus of feedback and its adaptivity remains largely overlooked. To bridge this gap, we adopt and refine Narciss's taxonomy into seven feedback focus types to annotate teacher and LLM-generated feedback across three university writing courses. We release FeedType, a benchmark containing annotated teacher and LLM feedback from six LLMs under three prompting strategies. We assess the coverage and distribution of feedback focus types, and examine whether LLMs adapt their feedback across draft stages and student performance levels as an expert instructor does. Our findings show that while most LLMs cover most feedback focus types, they fail
    
[^35]: PISCES：用于空间天气异常检测与早期预警的物理信息太阳风卷积自编码器

    PISCES: Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather Anomaly Detection and Early Warning

    [https://arxiv.org/abs/2609.28022](https://arxiv.org/abs/2609.28022)

    PISCES是一种无需标签、融合多种物理约束的太阳风卷积自编码器，它将异常分数分解为磁场、等离子体、物理关系和残差修正等物理可解释的分量，从而实现空间天气瞬变结构的早期检测与预警。

    

    空间天气早期预警依赖于在太阳风瞬变结构到达地球之前，在太阳-地球第一拉格朗日点（L1）的原位测量中检测到它们。固定阈值方法可能会遗漏磁与等离子体组合结构的异常，而许多学习方法仅提供一个单一的异常分数。我们提出了物理信息太阳风卷积自编码器（PISCES），这是一个在物理约束下、无需目录标签、基于OMNI太阳风测量数据训练的卷积自编码器。其损失函数包含磁场一致性、温度与速度之间的经验关系、帕克螺旋角，以及对由重构计算出的派生量在连续一分钟采样之间变化的惩罚项。在推理阶段，PISCES将异常分数分解为磁场重构误差、等离子体重构误差、物理关系误差和残差修正，并报告每个分量的贡献大小。

    arXiv:2609.28022v1 Announce Type: cross  Abstract: Space weather early warning depends on detecting solar wind transients in in-situ measurements at the first Sun-Earth Lagrange point (L1), before they reach Earth. Fixed thresholds can miss combined magnetic and plasma structure, and many learning methods provide a single anomaly score. We present the Physics-Informed Solar-wind Convolutional autoEncoder for Space-weather (PISCES), a convolutional autoencoder trained without catalog labels on OMNI solar wind measurements under physics constraints. Its loss includes magnetic field consistency, an empirical relation between temperature and velocity, the Parker spiral angle, and penalties on changes between consecutive one-minute samples in derived quantities calculated from the reconstruction. At inference, PISCES separates the anomaly score into magnetic and plasma reconstruction errors, physics relations, and residual corrections, and reports the magnitude of each contribution. Attenua
    
[^36]: 审讯对话的可控属性特定摘要生成

    Controlled Attribute-Specific Summarization of Interrogative Dialogues

    [https://arxiv.org/abs/2609.28004](https://arxiv.org/abs/2609.28004)

    提出了CASPER框架，结合思维链属性特定提示与多角色分层评估机制（RoleEval），并基于新构建的MINDSum数据集，显著提升了审讯对话摘要的事实一致性和上下文完整性。

    

    审讯对话的有效摘要生成是法证和调查场景中的一项关键任务，需要高度的事实准确性、连贯性以及针对特定属性的相关性。在这项工作中，我们提出了CASPER，一种新颖的思维链属性特定评估式摘要提示框架，它利用结构化提示和迭代优化来生成审讯者与证人交互的高质量摘要。我们构建了MINDSum数据集，该数据集扩展了MIND语料库，包含6,000个话语对，并标注了事件细节、事实陈述、人物描述和填充内容。CASPER采用RoleEval这一分层评估机制，由多个角色（警官、督察、高级督察）基于预定义标准对摘要进行迭代评估。通过整合实体提取和结构化反馈循环，CASPER显著提升了事实一致性和上下文完整性。

    arXiv:2609.28004v1 Announce Type: cross  Abstract: Effective summarization of interrogative dialogues is a critical task in forensic and investigative settings, requiring high factual accuracy, coherence, and attribute-specific relevance. In this work, we introduce CASPER, a novel Chain-of-Thought Attribute-Specific Prompting for Evaluative Summarization framework that leverages structured prompting and iterative refinement to generate high-quality summaries of interrogator-witness interactions. We construct MINDSum, a dataset extending the MIND corpus, comprising 6,000 utterance pairs annotated with event details, factual statements, character descriptions, and fillers. CASPER employs RoleEval, a hierarchical evaluation mechanism where multiple roles (officer, inspector, senior inspector) iteratively assess summaries based on predefined criteria. By integrating entity extraction and structured feedback loops, CASPER significantly improves factual consistency and contextual completenes
    
[^37]: 局部控制合规，集体性歧视：受监管金融领域多智能体AI的治理架构

    Compliant with Local Controls, Collectively Discriminatory. A Governance Architecture for Multi-Agent AI in Regulated Finance

    [https://arxiv.org/abs/2609.27994](https://arxiv.org/abs/2609.27994)

    本文提出ARIA参考架构，揭示了局部合规的多个AI智能体组合后仍可能产生集体性歧视等不可接受结果的“宪制不可组合性”问题，并为受监管金融领域的智能体群体治理提供了涵盖六大能力的治理框架与可证伪研究议程。

    

    金融机构开始在信贷、反欺诈、催收、合规和运营控制等领域部署智能体工作流。当前的治理在很大程度上仍以组件为中心：每个模型或智能体都是在本地进行规范、测试、授权和监控的。然而，当机构层面的风险源于众多本地可接受组件的联合行为时，这种方法是不够的。我们将这一治理缺口称为“宪制不可组合性”：局部合规检查未必能组合成可接受的集体结果，例如有界限的差别影响、市场诚信或可追溯的问责。我们提出ARIA，作为面向金融领域的参考架构和针对智能体群体治理的可证伪研究议程。它在规范-问责、执行控制和保障学习三个层面上组织了六项能力：政策规范、群体层面的实际行为与预期行为监控（M2）、有界授权、运行时遏制……

    arXiv:2609.27994v1 Announce Type: cross  Abstract: Financial institutions are beginning to deploy agentic workflows in credit, fraud, collections, compliance, and operational control. Governance remains largely component-centric: each model or agent is specified, tested, authorized, and monitored locally. That is insufficient when institutional risk arises from the joint behavior of many locally acceptable components. We call this gap constitutional non-compositionality: local compliance checks need not compose into acceptable collective outcomes such as bounded disparate impact, market integrity, or traceable accountability. We propose ARIA as a finance-specific reference architecture and falsifiable research agenda for agent-population governance. It organizes six capabilities across normative-accountability, execution-control, and assurance-learning planes: policy specification, population-level observed-versus-expected behavior monitoring (M2), bounded authority, runtime containmen
    
[^38]: 一类低参数量正交矩阵的黎曼结构与优化

    Riemannian Structure and Optimization for a Class of Low-Parametric Orthogonal Matrices

    [https://arxiv.org/abs/2609.27982](https://arxiv.org/abs/2609.27982)

    本文为一类由块对角因子与固定置换交织而成的低参数量正交矩阵建立了黎曼流形结构，并提出了基于自动微分的高效黎曼优化算法，可应用于最佳矩阵逼近和参数高效微调。

    

    本文研究由块对角因子与固定置换交织而成的矩阵——这是一类灵活的结构化矩阵族。该类矩阵近来因其在表达能力与计算效率之间的良好权衡而在深度学习架构中受到关注，但针对它的高效计算策略仍有待探索。我们通过黎曼几何的视角来处理这一问题，并考察了该类矩阵在何种条件下具有光滑流形结构。对于实际应用中重要的正交双因子矩阵情形，我们推导了基本的黎曼工具，并提出了实现这些工具的高效算法。这些算法利用自动微分技术，支持每个因子内部的参数共享，并避免了显式构造稠密矩阵。我们在黎曼优化框架下，针对最佳矩阵逼近问题和参数高效微调任务对这些算法进行了测试。

    arXiv:2609.27982v1 Announce Type: cross  Abstract: In this paper, we are concerned with matrices formed by block-diagonal factors interleaved with fixed permutations -- a flexible family of structured matrices. This class has recently drawn interest in deep learning architectures for its balanced expressivity-efficiency trade-off, yet efficient computational strategies for working with it remain to be found. We approach this problem through Riemannian geometry and examine under what conditions this class admits a smooth manifold structure. For the practically important case of orthogonal two-factor matrices, we derive the essential Riemannian tools and propose efficient algorithms for their implementation. The algorithms leverage automatic differentiation, support parameter sharing within each factor, and avoid explicit dense matrix construction. We test them within the Riemannian optimization framework on the best matrix approximation problem and for parameter-efficient fine-tuning of
    
[^39]: 离散度与规模：决定测试时预算分配是否划算的因素

    Spread and Scale: What Determines Whether Test-Time Budget Allocation Pays

    [https://arxiv.org/abs/2609.27917](https://arxiv.org/abs/2609.27917)

    本文通过预先注册的验证性实验发现，工作负载中实例难度的离散程度是决定测试时预算重新分配是否划算的关键属性，并且即使计入分配策略自身消耗的预算成本，该策略依然可能带来收益。

    

    神经组合优化求解器为每个实例生成多个候选解并报告其中找到的最优解，无论实例难度如何，都为每个实例分配相同的采样预算。一项配套研究表明，将固定预算向更难的实例重新分配可以提升解的质量，但这种改进的标准度量方式存在偏差：在相同数据上既决定分配又评估该分配，即使实际没有收益，也可能制造出表面上的收益。这就留下了两个未解决的问题：工作负载的什么特性决定了重新分配是否值得做，以及一个花费部分预算来决定如何分配其余预算的策略，在计入该成本之后是否仍然划算。本文通过预先注册的验证性实验——在数据收集之前就固定了分析方法和判定标准——在三个独立训练的求解器以及两种在训练分布上构造更难工作负载的方式上回答了这两个问题。

    arXiv:2609.27917v1 Announce Type: new  Abstract: Neural combinatorial optimization solvers generate many candidate solutions per instance and report the best one found, using the same sample budget for every instance regardless of difficulty. A companion study showed that reallocating a fixed budget toward harder instances can improve solution quality, but that the standard way of measuring this improvement is biased: deciding an allocation and evaluating it on the same data can manufacture an apparent gain even when none exists. This left open what property of a workload determines whether reallocation is worth doing, and whether a policy that spends part of the budget to decide how to allocate the rest still pays once that cost is counted.   This paper answers both questions through pre-registered confirmatory experiments -- analysis and verdict criteria fixed before data collection -- across three independently trained solvers and two ways of constructing harder workloads on the tra
    
[^40]: 基于趋势预测的复杂交通网络安全韧性恢复方法

    A Resilience Recovery Method for Complex Traffic Network Security Based on Trend Forecasting

    [https://arxiv.org/abs/2609.27903](https://arxiv.org/abs/2609.27903)

    本文创新性地提出了一种基于韧性趋势预测的交通网络韧性恢复方法，通过引入风险值建立SIRD-R故障传播模型，并构建了涵盖实时韧性与整体韧性的交通网络韧性模型，以应对复杂交通网络面临的安全挑战。

    

    随着信息技术的快速发展，一个庞大而复杂的交通网络已在航空、航天、车辆、船舶、电力和工业等各个领域建立起来。然而，由于其结构的复杂性和多样性，复杂交通网络容易受到攻击，面临着严峻的安全挑战。因此，本文创新性地提出了一种基于韧性趋势预测的交通网络韧性恢复方法。本文将风险值引入网络故障传播过程的分析中，建立了易感-感染-恢复-死亡-风险（SIRD-R）故障传播模型。通过网络韧性承受能力与韧性恢复能力的融合，构建了涵盖实时韧性和整体韧性的交通网络韧性模型。随后，韧性……

    arXiv:2609.27903v1 Announce Type: new  Abstract: Due to the rapid development of information technology, a huge and complex traffic network has been established across various sectors, including aviation, aerospace, vehicles, ships, electric power, and industry. However, because of the complexity and diversity of its structure, the complex traffic network is vulnerable to being attacked and faces serious security challenges. Therefore, this paper innovatively proposes a traffic network resilience recovery method based on resilience trend forecasting. In this paper, the risk value is introduced into the analysis of the network fault propagation process, and the Susceptible, Infectious, Recovered, Dead-Risk (SIRD-R) fault propagation model is established. The resilience model of traffic network, which encompasses real-time resilience and overall resilience, is constructed through the integration of network resilience bearing capacity and resilience recovery capacity. Ten, the resilience 
    
[^41]: 薛定谔的代码仓库：大语言模型是学会了SWE-bench，还是只是背下了它？

    Schr\"odinger's Code Repository: Have LLMs Learned SWE-bench or Memorized It?

    [https://arxiv.org/abs/2609.27891](https://arxiv.org/abs/2609.27891)

    论文提出SchrodingerRepo评估框架，将测试仓库视为评估时才动态实例化的潜变量，在保留可执行行为的同时抹除命名约定、文件布局等熟悉线索，从而揭示大语言模型在SWE-bench等仓库级基准上的高分可能源于对训练数据的记忆而非真正的仓库推理能力。

    

    仓库级代码评测基准已成为评估代码智能体的标准，但由于它们构建于被反复用于模型训练的热门开源仓库之上，天然存在数据泄露问题。因此，优秀的性能表现可能反映的是对经典仓库线索的记忆，而非稳健的仓库级推理能力。我们提出了SchrodingerRepo（薛定谔的仓库），一个用于在动态实例化的仓库表示下测试代码智能体的评估框架。SchrodingerRepo不再反复使用测试仓库的静态表示，而是将测试仓库视为一个仅在评估时才存在的潜变量，只在智能体进入评估环境时才被动态实例化。实例化后的仓库在保留原始可执行行为的同时，通过四种变换侵蚀诸如命名约定、文件布局和实现模式等熟悉线索（原文摘要在此处截断）。

    arXiv:2609.27891v1 Announce Type: cross  Abstract: Repository-level coding benchmarks have become the standard for evaluating coding agents, yet they inherently suffer from data leakage because they are built upon popular open-source repositories repeatedly used for training. Consequently, strong performance may reflect memorization of canonical repository cues rather than robust repository reasoning. We propose SchrodingerRepo (Schr\"odinger's Repository), an evaluation framework for testing coding agents under dynamically instantiated repository representations. Instead of repeatedly using a static representation of the test repository, SchrodingerRepo treats the test repository as an evaluation-time latent variable that is dynamically instantiated only when the agent enters the evaluation environment. The instantiated repository preserves the original executable behavior while eroding familiar cues such as naming conventions, file layouts, and implementation patterns through four tr
    
[^42]: RelCheck：面向视觉语言模型幻觉纠正的双证据空间定位

    RelCheck: Dual-Evidence Spatial Grounding for VLM Hallucination Correction

    [https://arxiv.org/abs/2609.27890](https://arxiv.org/abs/2609.27890)

    RelCheck提出了一种无需训练的事后纠正方法，通过融合RelTR场景图三元组与边界框几何空间谓词的双重关系证据，有效纠正视觉语言模型中的关系幻觉，并在MME幻觉评测中显著超越Woodpecker风格基线。

    

    多模态大语言模型（MLLM）经常生成与输入图像不一致的文本。虽然对象级和属性级幻觉已受到广泛关注，但关系幻觉（即对对象之间空间关系或交互关系的不正确描述）在很大程度上仍未被现有的事后纠正方法所解决。我们提出了RelCheck，这是一个无需训练的事后纠正流水线，它在对象级视觉定位的基础上引入了双重关系证据：来自RelTR的学习型场景图三元组，以及来自边界框几何的确定性空间谓词。这两类证据与Woodpecker风格的对象声明层相结合，构成一个三层视觉知识库，供语言模型纠正器用来重写包含幻觉的文本。在LLaVA v1 13B上的评估结果显示，RelCheck的总MME幻觉得分为630.0，而Woodpecker风格基线为585.0，其中最大的提升来自……

    arXiv:2609.27890v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) fre- quently generate text that is inconsistent with the input image. While object- and attribute-level hallucinations have received considerable attention, relational hallucinations (incorrect de- scriptions of spatial or interactive relationships between objects) remain largely unaddressed by existing post-hoc correction methods. We present RelCheck, a training-free post-hoc correction pipeline that augments object-level visual grounding with dual relational evidence: learned scene-graph triples from RelTR and deterministic spatial predicates from bounding-box geometry. These combine with a Woodpecker-style object claim layer to form a three-layer visual knowledge base, which a language model corrector uses to rewrite hallucinated text. Evaluated on LLaVA v1 13B, RelCheck achieves a total MME hallucination score of 630.0 versus 585.0 for a Woodpecker-style baseline, with the largest gain on th
    
[^43]: 自主科学发现中的伪科学诱导

    False-science induction in autonomous scientific discovery

    [https://arxiv.org/abs/2609.27883](https://arxiv.org/abs/2609.27883)

    该论文揭示了自主科学发现系统中的“伪科学诱导”现象——当物理对象与测量结果被连贯地错误配对时，神经代理模型会学到虚假关联并系统性地将实验预算引向低性能区域，且错误的一致性而非错误频率是决定性变量。

    

    闭环发现系统日益增多地自主执行实验并更新决策，使记录完整性成为实验装置本身的一部分。我们证明，当合法的物理对象与测量结果被错误配对时，会产生“伪科学诱导”：神经代理模型会忠实地学习由记录诱导产生的、与真实对象-结果关系不对应的虚假关联，而数据的边缘分布保持不变。在绿色荧光蛋白适应度和材料带隙预测两个闭环系统中，连贯的配对错误绑定会系统性地将实验预算引导至低性能区域，而同等数量的随机交换则几乎没有影响。这些观察结果表明，在所测试的闭环中，错误的一致性而非原始错误频率，才是控制这种预算错配的主要变量。由此得出的绑定可辨识性边界支持对监测轴……（原文摘要在此处截断）

    arXiv:2609.27883v1 Announce Type: cross  Abstract: Closed-loop discovery systems increasingly execute experiments and update decisions autonomously, turning record integrity into part of the experimental apparatus. We show that false-science induction arises when legitimate physical objects and measurements are paired incorrectly, driving neural surrogates to faithfully learn record-induced associations that do not correspond to the true object-outcome relationship while marginal data distributions remain unchanged. Across green fluorescent protein fitness and materials band-gap prediction loops, coherent paired misbinding systematically redirects experimental budgets toward low-performing basins, whereas same-volume random swaps have negligible effects. These observations identify error coherence, rather than raw error frequency, as the primary variable controlling this budget misallocation in the tested loops. The resulting binding identifiability boundary supports monitored-axis qua
    
[^44]: 有界循环：面向智能体执行框架的预运行支出上限、可证明终止性与经验证的完成保证

    Bounded Loops: Pre-Run Spend Bounds, Proved Termination, and Verified Completion for Agent Harnesses

    [https://arxiv.org/abs/2609.27871](https://arxiv.org/abs/2609.27871)

    该论文提出“有界循环”机制，通过工作者无法篡改的独立关卡与全局修复预算，为智能体执行框架首次提供了可证明的终止性、预运行支出上限以及经哈希链账本验证的完成保证。

    

    在主流的智能体框架中，一个步骤的结束是由智能体自身的输出所声明的。持久化执行平台虽然对重试次数和时间设置了上限，但其校验器通常与工作代码位于同一代码库中：这是一种依赖部署方自觉遵守的纪律，而非执行框架强制执行的属性。我们声明了智能体执行框架必须保证什么，对其加以证明，并构建了衡量执行框架是否真正兑现这些保证的测量工具。有界循环由一个工作者、一个工作者无法写入的独立关卡，以及一个预先声明的预算组成；有界循环图将三者组合起来，并引入一种修复关系，使下游的失败可以重新运行已完成的上游节点。由此得出三项保证。它会终止：只要修复预算是全局的而非按节点分配的，终止性在修复机制下依然成立，且最坏情况下的尝试总数具有闭式解。它不会漂移：任何节点都不会在没有关卡裁决的情况下达到完成状态，裁决记录在仅追加的哈希链账本中，该性质由控制流性质加以证明。（原文摘要在此处截断）

    arXiv:2609.27871v1 Announce Type: cross  Abstract: In mainstream agent frameworks, a step ends when the agent's own output says it has finished. Durable-execution platforms bound retries and time, but their checker conventionally lives in the same codebase as the work: a discipline the deployment is trusted to keep, not a property the harness enforces.   We state what an agent harness must guarantee, prove it, and build the instrument that measures whether a harness delivers it. A bounded loop is a worker, an independent gate the worker cannot write to, and a declared budget; a bounded-loop graph composes them with a repair relation that lets a downstream failure re-run a finished upstream node. Three guarantees follow. It finishes: termination holds under repair, with the worst-case attempt total in closed form, if the repair budget is global not per node. It does not drift: no node reaches DONE without a gate verdict in an append-only hash-chained ledger, proved from control flow, si
    
[^45]: 学习激活什么：面向长程多模态智能体的组合式能力分配

    Learning What to Activate: Combinatorial Capability Allocation for Long-Horizon Multimodal Agents

    [https://arxiv.org/abs/2609.27869](https://arxiv.org/abs/2609.27869)

    该论文提出CoCA在线策略学习框架，通过稀疏条件比较学习可部署的能力子集策略，使长程多模态智能体能在各交互阶段以代价敏感的方式动态进行组合式能力分配，克服了固定能力集合或预定义工作流带来的计算开销与适应性不足问题。

    

    长程多模态智能体依赖于感知、检索、推理、验证和执行等专门能力。现有设计通常激活固定的能力集合或调用预定义的工作流程，这带来了大量计算开销，同时无法适应随阶段变化的能力需求。本文研究了长程多模态智能体系统中的组合式能力分配问题，即系统在每个交互阶段选择一个代价敏感的专门能力子集。这一问题并不简单，因为各能力的价值取决于所选择的子集，而先前的分配又会改变后续决策所面临的状态。我们提出了CoCA，一种在线策略学习框架，能够从稀疏的条件比较中恢复出可部署的能力子集策略。在学生策略所访问的状态上，更强的教师模型会比较各能力子集的边际（摘要在此处截断）。

    arXiv:2609.27869v1 Announce Type: new  Abstract: Long-horizon multimodal agents rely on specialized capabilities for perception, retrieval, reasoning, verification, and execution. Existing designs typically activate a fixed capability set or invoke a predefined workflow, incurring substantial computational overhead while failing to accommodate stage-dependent capability demands. In this paper, we study the \textit{combinatorial capability allocation} problem for long-horizon multimodal agent systems, where the system selects a cost-sensitive subset of specialized capabilities at each interaction stage, which is nontrivial since capability values depend on the selected subset, while previous allocations alter the states encountered by subsequent decisions. We introduce \textsc{CoCA}, an on-policy learning framework that recovers a deployable capability-subset policy from sparse conditional comparisons. On states visited by the student policy, the stronger teacher compares the marginal n
    
[^46]: TopoGS：面向大规模3D高斯泼溅的拓扑感知锚点特征聚合

    TopoGS: Topology-Aware Anchor Feature Aggregation for Large-Scale 3D Gaussian Splatting

    [https://arxiv.org/abs/2609.27868](https://arxiv.org/abs/2609.27868)

    提出拓扑感知的锚点特征聚合框架TopoGS，通过分层锚点耦合与结构感知包含聚合这两个轻量级组件，在基于八叉树的大规模3D高斯泼溅中充分利用八叉树拓扑结构，实现更有效的跨层级特征学习。

    

    基于八叉树的3D高斯泼溅将锚点组织为多层级结构以实现细节层次渲染，但不同层级的特征通常是独立优化的，导致八叉树拓扑结构在特征学习过程中未被充分利用。我们观察到，统一的跨层级聚合会产生不对称的效果：精细层级的锚点受益于粗粒度上下文，而粗粒度层级的锚点则需要从其后代中选择性地获取信息。因此，我们提出了TopoGS，一个包含两个轻量级组件的拓扑感知锚点特征聚合框架。分层锚点耦合通过残差MLP融合各层级的上下文三元组，建立双向的跨层级梯度通路；结构感知包含聚合则利用八叉树包含关系和基于哈希的匹配，将具有有效父子关系的锚点与孤立锚点区分开来，并通过软加权来适应不同的拓扑空间关系。

    arXiv:2609.27868v1 Announce Type: cross  Abstract: Octree-based 3D Gaussian Splatting organizes anchors into multi-level hierarchies for level-of-detail rendering, but features at different levels are typically optimized independently, leaving the octree topology underused during feature learning. We observe that uniform cross-level aggregation produces asymmetric effects: fine-level anchors benefit from coarse context, whereas coarse-level anchors require selective information from their descendants. We therefore propose TopoGS, a topology-aware anchor feature aggregation framework with two lightweight components. Hierarchical Anchor Coupling establishes bidirectional cross-level gradient pathways by fusing per-level context triplets with a residual MLP. Structure-Aware Containment Aggregation uses octree containment and hash-based matching to distinguish anchors with valid parent-child relations from isolated anchors, then applies soft weighting to accommodate varying topological spa
    
[^47]: 共享编码器并非共享任务：面向深度专家池的条件比较

    A Shared Encoder Is Not a Shared Task: Conditional Comparison for Deep Expert Pools

    [https://arxiv.org/abs/2609.27866](https://arxiv.org/abs/2609.27866)

    该论文发现共享深度编码器无法消除任务比较分数中的混淆，并提出将条件化双判别器差异移植到嵌入空间形成“功能轴”度量，既能免疫输入旋转的外推混淆又能敏感捕捉标签置换漂移，从而在混合多头生命周期中以更少头数取得更优的决策质量。

    

    共享一个深度编码器本身并不能修复任务比较分数中的核心混淆问题。我们证明，在冻结的共享表示上进行交叉评估的预测头会继承浅层交换分数的外推混淆：在标签固定不变的情况下，纯粹的输入旋转会使深度交换分数从约0膨胀至0.80；而表示新颖性分数则在互补方向上存在盲区（在完全改变任务的标签置换下保持平坦不变）。将条件化双判别器差异移植到嵌入空间中，可以同时解决这两个盲点：功能轴在旋转下保持在±0.001以内，并随标签置换造成的漂移质量单调变化。将该双轴门控机制纳入多头混合（mixture-of-heads）生命周期中，在匹配的训练预算下，它能以更少的头数获得优于交换或新颖性触发器的决策质量。在广义类别发现任务上，同样的块级功能轴能够区分语义新颖（类别）……（原文摘要在此处不完整）

    arXiv:2609.27866v1 Announce Type: cross  Abstract: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic nov
    
[^48]: 什么变了？具有真实、虚拟与不可比较诊断的漂移检测

    What Changed? Drift Detection with Real, Virtual, and Incomparable Diagnosis

    [https://arxiv.org/abs/2609.27865](https://arxiv.org/abs/2609.27865)

    提出将条件双判别器差异移植到嵌入空间的双轴诊断方法，同时弥补交换分数对输入旋转的虚假敏感和表示新颖性分数对标签置换漂移的盲区，从而在混合头生命周期中以更少头部实现更优漂移检测决策。

    

    共享一个深度编码器本身并不能解决任务比较评分中的核心混淆问题。我们证明，在冻结的共享表示上进行交叉评估的头部会继承浅层交换分数的外推混淆：在标签固定、纯输入旋转的情况下，深度交换分数会从约0膨胀到0.80，而表示新颖性评分在互补方向上是盲目的（在完全改变任务的标签置换下保持平坦）。将条件双判别器差异移植到嵌入空间中可以同时解决这两个盲点：功能轴在旋转下保持在±0.001以内，并单调地跟踪标签置换带来的漂移质量。将该双轴门控构建到混合头（mixture-of-heads）生命周期中，在相同的训练预算下，它能以更少的头部获得比交换分数或新颖性触发器更好的决策质量。在广义类别发现任务上，同样的块级功能轴能够区分语义新（类别）……

    arXiv:2609.27865v1 Announce Type: cross  Abstract: Sharing a deep encoder does not, by itself, fix the central confound of task-comparison scores. We show that cross-evaluated heads on a frozen shared representation inherit the extrapolation confound of shallow exchange scores: pure input rotations with fixed labels inflate a deep exchange score from about 0 to 0.80, while representation-novelty scores are blind in the complementary direction (flat under label permutations that change the task completely). Transplanting a conditional two-discriminator discrepancy into the embedding space resolves both blind spots: the functional axis stays within +-0.001 under rotations and tracks label-permutation drift mass monotonically. Built into a mixture-of-heads lifecycle, the two-axis gate attains better decision quality with fewer heads than exchange or novelty triggers at a matched training budget. On generalized category discovery, the same chunk-level functional axis separates semantic nov
    
[^49]: 知识库补全的忠实性准则层次体系

    A hierarchy of faithfulness criteria for knowledge base completion

    [https://arxiv.org/abs/2609.27863](https://arxiv.org/abs/2609.27863)

    该论文提出了知识库补全模型逻辑忠实性的四个递进严格准则——判别性、逻辑可容许性、单调逻辑忠实性和概率逻辑忠实性，证明了它们构成严格的蕴含链，并指出在开放世界假设下无法区分逻辑不可能公理与合理新公理的模型在语义上是错误的。

    

    arXiv:2609.27863v1 公告类型：新论文 摘要：知识图谱补全通常通过将观测到的三元组排在随机损坏的三元组之上来进行评估，这种方法将所有未观测的事实都视为假的。然而，当被补全的对象是描述逻辑知识库而非普通图时，开放世界假设和演绎闭包使得这种评估方式不再适用：相对于知识库而言，一个候选公理可能是被蕴含的、矛盾的或未确定的，而一个无法区分逻辑上不可能的公理与合理的新公理的模型不仅仅是准确性较低，而是在语义上是错误的。我们探讨了知识库补全模型的逻辑忠实性意味着什么，以及当前的嵌入模型是否具备这种忠实性。我们定义了一个由四个越来越严格的准则组成的层次体系，即判别性、逻辑可容许性、单调逻辑忠实性和概率逻辑忠实性，并证明它们构成一条严格的蕴含链。我们将最强准则建立在（原文在此处截断）……

    arXiv:2609.27863v1 Announce Type: new  Abstract: Knowledge graph completion is evaluated by ranking observed triples above randomly corrupted ones, which treats every unobserved fact as false. When the object being completed is a description logic knowledge base rather than a plain graph, the open world assumption and deductive closure make this inadequate: relative to the knowledge base, a candidate axiom is entailed, contradictory, or undetermined, and a model that cannot separate a logically impossible axiom from a plausible novel one is not merely less accurate but semantically incorrect. We ask what it means for a knowledge base completion model to be logically faithful, and whether current embedding models are. We define a hierarchy of four increasingly strict criteria, discrimination, logical admissibility, monotonic logical faithfulness, and probabilistic logical faithfulness, and prove that they form a strict chain of implications. We ground the strongest criterion in the rela
    
[^50]: AI系统中的可达全局优化：全局究竟有多“全局”？

    Reachable Global Optimization in AI Systems: How Global Is Global?

    [https://arxiv.org/abs/2609.27855](https://arxiv.org/abs/2609.27855)

    本文提出可达性诱导优化（RIO）模型，揭示AI系统所声称的“全局优化”实际上仅限于系统可达的候选区域，其解只是可达最优而非真正全局最优，只有借助额外证书将可达区域与完整形式空间关联时才能称为全局最优，并通过66,150次基准试验验证了这一理论框架。

    

    AI系统日益声称能够优化提示词、策略、架构、规划、工具使用轨迹、推理轨迹以及测试时计算。本文认为，除非这些声明明确指出执行优化的系统实际可达的区域，否则这些声明是不够明确的。我们提出了可达性诱导优化，在该模型中，生成器、验证器、控制器、记忆、工具和预算共同诱导出一个可达的候选区域。因此，系统返回的解要么是已访问过的最佳点，要么是近似的可达最优解，只有当额外的证书将可达区域与完整的形式化空间联系起来时，才能称其为精确的全局最优解。我们证明了可达最优性、伪全局性、间隙分解、证书、逃逸、剪枝和控制值等一系列结果。完整的基准测试记录包含66,150次实际执行的试验，涵盖六个已知最优解的地形族、七种控制策略、270个地形和35次运行……

    arXiv:2609.27855v1 Announce Type: new  Abstract: AI systems increasingly claim to optimize prompts, policies, architectures, plans, tool-use trajectories, reasoning traces, and test-time computation. This paper argues that such claims are underspecified unless they state the region actually reachable by the system that performed the optimization. We introduce Reachability-Induced Optimization (RIO), a model in which a generator, verifier, controller, memory, tools, and budget induce a reachable candidate region. The returned solution is therefore a best visited point, an approximate reachable optimum, or an exact global optimum only when additional certificates relate the reachable region to the full formal space. We prove reachable-optimality, false-globality, gap- decomposition, certificate, escape, pruning, and control-value results. The full benchmark record contains 66,150 executed trials over six known-optimum landscape families, seven control policies, 270 landscapes, and 35 run
    
[^51]: 查询隐含生成式引擎优化（QI-GEO）

    Query Implied Generative Engine Optimization

    [https://arxiv.org/abs/2609.27845](https://arxiv.org/abs/2609.27845)

    提出了QI-GEO方法，无需依赖显式查询，直接从文档本身近似其意图空间并推断用户意图，从而优化内容在生成式搜索引擎中的可见性。

    

    随着人们在线查找信息方式的演变，搜索领域的格局发生了巨大变化。传统搜索引擎正在被生成式搜索引擎（GSE）所取代，后者利用大语言模型（LLM）为用户查询生成自然语言回答。对于内容创作者而言，内容的可见性不再仅仅取决于在搜索结果中的排名，而是取决于是否被生成的回答所引用。然而，生成式搜索引擎是一个黑箱，由此催生了生成式引擎优化（GEO），即一系列旨在提高内容在生成式搜索环境中可见性的技术。现有的大多数方法都依赖于显式查询或由查询派生的信号来调整内容，以更好地满足用户需求。我们提出了查询隐含生成式引擎优化（QI-GEO），直接从文档中推断用户意图。我们的方法对文档的意图空间进行近似建模，并识别出文档中可能缺失的内容……

    arXiv:2609.27845v1 Announce Type: cross  Abstract: The landscape of search has changed drastically with how people look for information online. Traditional search engines are being replaced by Generative Search Engines (GSEs), which use Large Language Models (LLMs) to generate natural language responses to user queries. For content creators, visibility is no longer solely determined by ranking in search results but by being cited within generated responses. But Generative Search Engines are black-boxes, leading to the emergence of Generative Engine Optimization (GEO), a set of techniques aimed at improving content visibility in generative search settings. Most existing approaches rely on the explicit queries or query derived signals to align content to better suit user needs. We propose Query Implied Generative Engine Optimization (QI-GEO) to infers user intent directly from the document. Our approach approximates document's intent space and identifies content that may be missing yet r
    
[^52]: 面向政策约束的LLM医疗申诉生成的智能体治理与对抗性验证框架

    Agentic Governance and Adversarial Verification for Policy-Constrained LLM Healthcare Appeal Generation

    [https://arxiv.org/abs/2609.27844](https://arxiv.org/abs/2609.27844)

    提出AGVF多智能体框架，将医疗必要性申诉生成建模为约束马尔可夫决策过程，通过政策形式化、证据检索、差距分析、对抗性批评和门控合成五个智能体的协作与对抗验证，解决单智能体LLM在高风险医疗申诉场景中产生无依据内容和丢失政策逻辑结构的问题。

    

    索赔拒付管理每年给美国医疗系统带来约2600亿美元的管理开销。大语言模型（LLM）和检索增强生成（RAG）虽然能够生成流畅的临床文本，但单智能体架构在高风险医疗场景中会失效：它们会引入缺乏依据的临床细节，并丢失层级化支付方政策的逻辑结构。我们提出了AGVF（智能体治理与对抗性验证框架），这是一种在明确的政策与证据约束下生成医疗必要性申诉的多智能体架构。AGVF将申诉合成建模为一个约束马尔可夫决策过程（CMDP），由五个智能体构成：政策形式化、证据检索、差距分析、对抗性批评和门控合成。我们证明，在固定的政策约束图上进行细化会单调地减少证据缺陷，并最终以完整满足边界或局部化证据缺陷状态终止。

    arXiv:2609.27844v1 Announce Type: new  Abstract: Claim denial management costs U.S. healthcare approximately $260 billion annually in administrative overhead. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) can produce fluent clinical text, but single-agent architectures fail in high-stakes healthcare: they introduce unsupported clinical details and lose the logical structure of hierarchical payer policy. We propose AGVF (Agentic Governance and Adversarial Verification Framework), a multi-agent architecture for medical-necessity appeal generation under explicit policy and evidence constraints. AGVF models appeal synthesis as a Constrained Markov Decision Process (CMDP) over five agents: policy formalization, evidence retrieval, gap analysis, adversarial critique, and gated synthesis. We prove that refinement over a fixed policy constraint graph monotonically reduces evidence-deficiency and terminates with either a complete satisfying frontier or a localized eviden
    
[^53]: 智能暖通空调系统中面向后量子网络安全的非侵入式云端迁移策略：架构、实现与实证评估

    A Non-Invasive Cloud-Based Migration Strategy for Post-Quantum Cybersecurity in Smart HVAC Systems: Architecture, Implementation, and Empirical Evaluation

    [https://arxiv.org/abs/2609.27828](https://arxiv.org/abs/2609.27828)

    提出了一种无需改动设备、固件或厂商云端的非侵入式PQC代理架构，通过树莓派网关为智能暖通空调系统实现后量子安全迁移，且后量子握手仅比经典基线慢0.38毫秒，开销极小。

    

    传统的智能暖通空调（HVAC）控制器依赖由ECDH和RSA保护的厂商云端TLS连接，而这两种算法均可被Shor算法破解；由于此类设备通常具有10-15年的使用寿命，当今部署的设备将一直服役至量子威胁时代。直接在设备端实施后量子密码学并不可行：作为代表性HVAC硬件的ESP32-S3仅有339 KB空闲堆内存，而ML-KEM-768需要900 KB；即便是经典的ECDH-P256密钥生成（111.93毫秒）也远慢于硬件AES-128（0.032毫秒）。我们提出了一种非侵入式的后量子密码（PQC）代理方案，无需对设备、固件或厂商云端进行任何改动，即可执行ML-KEM-768封装和ML-DSA-65认证（NIST FIPS 203/204），并通过HKDF生成AES-256-GCM会话密钥。该方案基于Open Quantum Safe liboqs库，在树莓派4B网关上实现。在超过500次运行测试中，后量子握手过程（步骤1-6）在2.48毫秒内完成，仅比经典基线慢0.38毫秒；在20毫秒模拟延迟下，PQC计算约占握手时间的8%。

    arXiv:2609.27828v1 Announce Type: cross  Abstract: Legacy smart HVAC controllers rely on vendor-cloud TLS secured by ECDH and RSA, both broken by Shor's algorithm, and typical 10-15 year lifespans mean today's devices remain in service through the quantum-threat era. Direct on-device post-quantum cryptography is infeasible: an ESP32-S3, representative of capable HVAC hardware, has only 339 KB free heap against the 900 KB ML-KEM-768 requires, and even classical ECDH-P256 keygen (111.93 ms) dwarfs hardware AES-128 (0.032 ms). We propose a non-invasive PQC proxy, requiring no device, firmware, or vendor-cloud changes, performing ML-KEM-768 encapsulation and ML-DSA-65 authentication (NIST FIPS 203/204) with AES-256-GCM session keys via HKDF, implemented with Open Quantum Safe liboqs on a Raspberry Pi 4B gateway. Over 500 runs, the post-quantum handshake (Steps 1-6) completes in 2.48 ms, 0.38 ms slower than classical baseline, with PQC computation around 8% of handshake time at 20 ms simula
    
[^54]: 基于VLM-LLM推理与可达性分析的安全多机器人协调

    Safe Multi-Robot Coordination via VLM-LLM Reasoning and Reachability Analysis

    [https://arxiv.org/abs/2609.27816](https://arxiv.org/abs/2609.27816)

    本文提出一种集中式安全感知M2M框架，通过VLM-LLM语义推理与可达性分析相结合，利用视觉机器人经MQTT共享语义环境感知，使异构机器人团队（具备视觉的四足机器人和无摄像头车辆）能够在避障并防止机器人间不安全交互的同时实现协作目标导向导航。

    

    当机器人在感知能力、环境意识和运动执行角色方面存在差异时，异构机器对机器（M2M）机器人系统中的安全协调极具挑战性。本文提出了一种集中式安全感知的M2M框架，用于由具备视觉能力的四足机器人和无摄像头机器人车辆组成的异构移动机器人团队进行协作目标导向导航。其目标是在避开静态和动态障碍物、防止机器人间不安全交互的前提下，引导两个平台到达目标区域。在共享感知原则下，具备视觉能力的机器人通过MQTT代理经由集中式服务器提供语义环境感知，使无摄像头平台能够利用这一共享场景表示以及自身的里程计、IMU和状态反馈进行导航。视觉语言模型（VLM）负责解读视觉数据流，提取的语义数据被映射（摘要在此处截断）

    arXiv:2609.27816v1 Announce Type: cross  Abstract: Safe coordination in heterogeneous machine-to-machine (M2M) robotic systems is challenging when robots differ in sensing capabilities, environmental awareness, and motion execution roles. This paper presents a centralized safety-aware M2M framework for cooperative goal-directed navigation in a heterogeneous mobile robot team comprising a vision-capable quadruped and a camera-less robotic vehicle. The objective is to guide both platforms toward a goal region while avoiding static and dynamic obstacles and preventing unsafe inter-robot interactions. Under the principle of shared perception, the vision-capable robot provides semantic environmental awareness through a centralized server over an MQTT broker, enabling the camera-less platform to navigate using this shared scene representation alongside its own odometry, IMU, and state feedback. A vision-language model (VLM) interprets the visual stream, and the extracted semantic data is map
    
[^55]: 评估仅基于ADC的深度学习流程在独立使用扩散加权MRI进行乳腺癌检测与分割中的应用

    Evaluating ADC-only deep learning pipelines for breast cancer detection and segmentation using standalone diffusion-weighted MRI

    [https://arxiv.org/abs/2609.27815](https://arxiv.org/abs/2609.27815)

    该论文首次系统评估了仅使用ADC图的深度学习流程在独立扩散加权MRI上进行乳腺癌检测与分割的性能，探索了无需对比剂的DW-MRI作为DCE-MRI替代方案的可行性。

    

    动态对比增强（DCE）成像是利用磁共振成像（MRI）检测和表征乳腺癌的金标准技术。然而，DCE-MRI需要较长的采集时间，并且需要向血液中注射对比剂，这可能引发过敏反应。相比之下，扩散加权MRI（DW-MRI）是乳腺MRI的一种标准补充技术，它无需对比剂、采集时间更短，并且能够计算与肿瘤细胞密度相关的表观扩散系数（ADC）图。然而，尽管具备这些技术优势，深度学习研究一直聚焦于基于DCE的模型，很少探索DW-MRI和ADC图在与DCE-MRI结合使用或作为独立替代方案时的肿瘤检测性能。在本研究中，我们评估了多种最先进的深度学习技术在乳腺癌检测与分割中的应用……

    arXiv:2609.27815v1 Announce Type: cross  Abstract: Dynamic contrast-enhanced (DCE) imaging is the gold standard technique for the detection and characterization of breast cancer using magnetic resonance imaging (MRI). However, DCE-MRI requires long acquisition times and the administration of contrast into the bloodstream, which can cause allergic reactions. Alternatively, diffusion-weighted MRI (DW-MRI) is a standard complementary technique for breast MRI that does not require contrast, has shorter acquisition times, and enables calculation of apparent diffusion coefficient (ADC) maps that correlate with tumor cellularity. Yet, despite these technical advantages, deep learning research has focused on DCE-based models and has barely explored the tumor detection performance of DW-MRI and ADC maps either in combination with DCE-MRI or as standalone alternatives. Here, we evaluate the application of different state-of-the-art deep learning techniques for detection and segmentation of breas
    
[^56]: LabourCrew：一个用于劳动法可信对抗性审议与成文法推理的多智能体RAG框架

    LabourCrew: A Multi-Agent RAG Framework for Trustworthy Adversarial Deliberation and Statutory Reasoning over Labour Law

    [https://arxiv.org/abs/2609.27814](https://arxiv.org/abs/2609.27814)

    LabourCrew通过StatuteGraph法律条文图索引、证据交换协议和校准信任门三种机制，构建了一个多智能体RAG框架，确保劳动法问答中的每个答案都必须可追溯地锚定在真实检索到的法条证据上，从而实现可信的对抗性审议与成文法推理。

    

    在成文法问答中，每一项主张都必须可追溯至证据，而不仅仅是相关，因为无法核实的劳工权利答案会带来严重的法律后果。现有系统存在不足：单次执行的RAG无法检测证据不足的情况，而多智能体法律辩论系统则将证据落地视为一种提示约定，允许智能体引用未经检索的证据。为了填补这一空白，我们提出了LabourCrew，一个围绕三种证据落地机制构建的多智能体RAG框架：StatuteGraph，一个显式链接章、节、但书及交叉引用结构的图索引，取代固定长度的文本切分；证据交换协议，将各辩护智能体和解释者限定在证据账本之内，使引用未检索文本成为不可能，同时由容错的监督委员会并行运行各辩护智能体，使个别故障只会降级而不会导致系统崩溃；以及校准信任门，它取代了……（摘要在此处截断，后续内容缺失）

    arXiv:2609.27814v1 Announce Type: cross  Abstract: In statutory question answering, every claim must be traceable to evidence, not merely relevant, since unverifiable labour-rights answers carry serious legal consequences. Current systems fall short: single-pass RAG cannot detect insufficient evidence, while multi-agent legal-debate systems treat grounding as a prompting convention, letting agents cite unretrieved evidence. To address this gap, we introduce LabourCrew, a multi-agent RAG framework built around three grounding mechanisms: StatuteGraph, a graph index that explicitly links chapter, section, proviso, and cross-reference structure rather than fixed-length spans; an Evidence Exchange Protocol that confines advocates and an interpreter to an evidence ledger, making citation to unretrieved text impossible, while a fault-tolerant supervisor board runs advocates in parallel so individual failures degrade rather than crash the system; and a Calibrated Trust Gate that replaces cate
    
[^57]: 问“哪个”，而非“多好”：测量由大语言模型评分的基准的精度规模

    Ask Which, Not How Good: Sizing Benchmarks Scored by an LLM

    [https://arxiv.org/abs/2609.27787](https://arxiv.org/abs/2609.27787)

    该论文利用概化理论证明，LLM评审逐点评分的基准存在由系统-评审交互决定的精度上限，增加题目数量无法突破，而改用成对比较评分可使上限提升至0.986。

    

    由LLM评审打分的基准通常会裁定0.1分量级的差异，但这些基准的测量分辨率却从未被测量过。现有的样本复杂度研究仅覆盖准确性基准，评审打分的情形仍处于空白。我们将被测系统作为测量对象，运用概化理论将373,019个评判分解为系统、题目、评审及交互成分。核心结果是结构性的：在单一评审下，概化系数无论题目数量多少都会渐近于 σ²_s/(σ²_s+σ²_sj)，因为系统×评审交互项不携带 n_i。题目会饱和，评审不会；当目标接近该上限时，达到目标所需的题目成本将趋于发散。该上限是逐点式量规评分的特性，而非LLM评审本身的特性。若改用两种呈现顺序下的成对偏好评分，σ²_sj 会比 σ²_s 低两个数量级，上限则升至0.986（bootstrap [0.934, 1.000]）。

    arXiv:2609.27787v1 Announce Type: new  Abstract: Benchmarks scored by an LLM judge routinely adjudicate differences of a tenth of a point, but the resolution of those benchmarks has never been measured. Existing sample-complexity work covers accuracy benchmarks and leaves the judged case open. Treating the system as the object of measurement, we decompose 373,019 judgments into system, item, judge and interaction components using generalizability theory.   The central result is structural: under a single judge, generalizability asymptotes to sigma2_s/(sigma2_s+sigma2_sj) regardless of item count, because the system-by-judge term carries no n_i. Items saturate; judges do not. The item cost of a target diverges as the target nears that ceiling.   The ceiling is a property of pointwise rubric scoring, not of LLM judging. Run as a pairwise preference in both presentation orders, sigma2_sj falls two orders of magnitude below sigma2_s and the ceiling rises to 0.986 (bootstrap [0.934, 1.000] 
    
[^58]: EidosDoc：面向高性价比半结构化文档问答的隐式结构编码

    EidosDoc: Implicit Structure Encoding for Cost-Effective Semi-Structured Document QA

    [https://arxiv.org/abs/2609.27784](https://arxiv.org/abs/2609.27784)

    EidosDoc通过隐式结构编码器将文档的层次关系、空间位置和文本内容联合嵌入稠密向量空间，以极低的计算成本实现了半结构化文档问答的最先进准确率。

    

    半结构化文档在科学报告、财务报表和技术手册中无处不在。针对此类文档的问答需要同时理解文本、表格、图表以及复杂的层次化布局。现有方法要么依赖反复调用大型语言模型进行结构解析和检索，导致高成本和大延迟；要么将文档扁平化处理而丢失布局和层次信息，牺牲了答案的准确性。为解决这一问题，我们提出了EidosDoc，一种以最小计算开销实现最先进准确率的新型系统。我们的方法引入了三项核心创新：（1）通过对比学习和结构一致性损失训练的隐式结构编码器，该模块将层次关系、空间位置和文本内容联合嵌入到稠密向量空间中，无需显式解析即可整体捕捉文档结构（摘要原文在此处截断）。

    arXiv:2609.27784v1 Announce Type: cross  Abstract: Semi-structured documents are ubiquitous in scientific reports, financial statements, and technical manuals. Question answering over such documents requires simultaneous understanding of text, tables, charts, and complex hierarchical layouts. Existing methods either rely on repeatedly calling large language models for structure parsing and retrieval, leading to high cost and large latency, or they flatten the document and lose layout and hierarchy information, sacrificing answer accuracy. To address this, we propose EidosDoc, a novel system that achieves state-of-the-art accuracy with minimal computational expense. Our approach introduces three core innovations. (1) An Implicit Structure Encoder trained via contrastive learning and a structure consistency loss. This module jointly embeds hierarchical relationships, spatial positions, and textual content into a dense vector space, capturing document structure holistically without the ne
    
[^59]: 超越不安全检测：面向多轮LLM安全失败的反事实锚定证据归因

    Beyond Unsafe Detection: Counterfactually Anchored Evidence Attribution for Multi-Turn LLM Safety Failures

    [https://arxiv.org/abs/2609.27773](https://arxiv.org/abs/2609.27773)

    该论文提出了反事实锚定的证据归因方法，构建了包含1,762段多轮对话的数据集并训练轻量级分层归因模型，突破了传统仅判定安全与否的局限，能够精确定位推动对话走向不安全轨迹的具体用户回合和标记片段。

    

    随着大型语言模型（LLM）从对话助手发展为先进的智能体（agentic）系统，护栏（guardrail）失效可能将对抗性意图转化为有害的实际执行。然而，大多数护栏评估框架只关注最终结果，即判断用户请求是安全还是不安全。这种方法对于多轮失败场景是不够的，因为对抗性意图分散在多个对话回合之中。这促使我们超越单纯的检测，去识别那些将对话推向不安全轨迹的具体回合和标记（token）。为支持这一目标，我们构建了一个具备行为验证和分层证据监督的多轮对话数据集。该数据集包含1,762段对话，包括对抗性对话、良性孪生对话，以及含有高风险词汇的良性变体对话。我们训练了一个轻量级的分层归因模型，它能够预测安全违规行为，并将其归因于起作用的用户回合和标记片段。

    arXiv:2609.27773v1 Announce Type: cross  Abstract: As Large Language Models (LLMs) move from conversational assistants to advanced agentic systems, guardrail failures can convert adversarial intents into harmful executions. However, most guardrail evaluation frameworks focus only on the result and assess whether a user request is safe or unsafe. This approach is insufficient for multi-turn failures, where adversarial intent is distributed across multiple turns. This motivates us to go beyond detection to identify the turns and tokens that push the conversation toward unsafe trajectories. To support this, we construct a multi-turn dataset with behavioral validation and tiered evidence supervision. The dataset contains 1,762 conversations, including adversarial conversations, benign twins, and benign variants with high-risk vocabulary. We train a lightweight hierarchical attribution model that predicts safety violations and attributes them to contributing user turns and token spans. The 
    
[^60]: 通过反向对齐少样本对话暴露实现大型推理模型的对齐

    Alignment of LRMs via Counter-Aligned Few-Shot Conversation Exposure

    [https://arxiv.org/abs/2609.27763](https://arxiv.org/abs/2609.27763)

    本文揭示了大型推理模型的推理过程可被注入含显式思维链的反向对齐少样本对话系统性引导（SRCF 攻击），其根源是对抗性泛化导致的表示漂移，并据此提出了后训练防御方法 ARCF。

    

    大型推理模型（LRMs）依靠显式的思维链（CoT）推理和大上下文窗口在复杂任务上取得优异表现，但这些特性也引入了新的攻击面。我们证明，LRMs 的推理过程可以通过在输入前附加包含显式 CoT 轨迹的反向对齐少样本对话而被系统性引导，导致其在有害查询上生成不安全内容，而在良性查询上产生不当拒绝。我们将该攻击形式化为 SRCF（通过反向对齐少样本对话引导推理），它仅通过灵活的对话接口即可运作，无需访问模型参数和梯度。我们的核心洞察是，SRCF 利用了一个对抗性泛化问题，该问题会引发表示漂移，使良性输入与有害输入的表示朝相似方向偏移。这一观察启发我们提出了后训练防御方法 ARCF（通过……对齐表示）。

    arXiv:2609.27763v1 Announce Type: new  Abstract: Large Reasoning Models (LRMs) rely on explicit chain-of-thought (CoT) reasoning and large context windows to achieve strong performance on complex tasks, but these features also introduce new attack surfaces. We show that LRMs' reasoning processes can be systematically steered by prepending counter-aligned few-shot conversations containing explicit CoT traces, leading to unsafe generations on harmful queries and unwarranted refusals on benign ones. We formalize this attack as SRCF (Steering Reasoning via Counter-Aligned Few-shot Conversations) that operates solely through a flexible conversational interface and requires no access to the model's parameters and gradients. Our key insight is that SRCF exploits an adversarial generalization issue that induces a representation drift, causing the representations of benign and harmful inputs to shift in a similar direction. This observation motivates our post-training defense, ARCF (Aligning Re
    
[^61]: 后门会留下结构性痕迹：用于联邦学习中后门检测与遏制的FedMAST

    Backdoors Leave Structural Traces: FedMAST for Backdoor Detection and Containment in Federated Learning

    [https://arxiv.org/abs/2609.27760](https://arxiv.org/abs/2609.27760)

    FedMAST防御方法通过综合结构、频谱和历史三轴互补证据对客户端更新进行评分并分层过滤遏制，从而检测出即使能绕过孤立异常信号的隐蔽后门攻击，因为后门投毒更新必然留下结构性痕迹。

    

    联邦学习使客户端无需共享其原始数据即可对共享模型进行分布式训练。然而，它对客户端提交更新完整性的依赖，使全局模型容易受到隐蔽的后门投毒攻击。尽管现有防御通常只检查孤立的证据来源，但受隐蔽性约束的攻击可以适应这些信号。在本文中，我们证明此类攻击虽然能够抑制孤立的异常信号，但其投毒更新仍会留下残留的结构性痕迹。我们提出FedMAST，一种用于联邦学习后门检测的联邦多轴结构追踪防御方法。FedMAST利用互补的结构、频谱和历史证据对客户端更新进行评分，然后应用分层过滤和轮次级遏制机制来限制对抗性影响。为了捕捉孤立信号可能遗漏的痕迹，FedMAST采用压缩对一致性评分来暴露耦合的特征失真。

    arXiv:2609.27760v1 Announce Type: cross  Abstract: Federated learning enables distributed training of a shared model without requiring clients to share their raw data. However, its reliance on the integrity of the client-submitted updates exposes the global model to stealthy backdoor poisoning. Although existing defenses often inspect isolated evidence sources, stealth-constrained attacks can adapt to these signals. In this paper, we show that such attacks can suppress isolated anomaly signals, but their poisoned updates still leave residual structural traces. We propose FedMAST, a Federated Multi-Axis Structural Tracing defense for backdoor detection in federated learning. FedMAST scores client updates using complementary structural, spectral, and historical evidence and then applies tiered filtering and round-level containment to limit adversarial influence. To capture traces that isolated signals may miss, FedMAST uses squeeze-pair coherence scoring to expose coupled feature distort
    
[^62]: 难负样本揭示了易负样本所掩盖的问题：在难负样本条件下，跨语言有害性表征随资源层级降低而退化

    Hard Negatives Reveal What Easy Negatives Hide: Cross-Lingual Harmfulness Representations Degrade with Resource Tier Under Hard Negatives

    [https://arxiv.org/abs/2609.27758](https://arxiv.org/abs/2609.27758)

    该论文发现跨语言有害性表征的迁移质量高度依赖负样本的选择——当使用表面相似但无害的难负样本（XSTest对比提示）评估时，表征在低资源语言中严重退化，表明此前“有害性表征跨语言迁移良好、拒答失效仅是校准问题”的结论被易负样本所掩盖。

    

    大型语言模型的安全对齐主要以英语进行训练，近期有研究报道称底层的有害性表征在翻译后依然保留：用英语训练的探针在低资源语言中区分有害与无害提示的效果几乎与在英语中一样好。这被视为跨语言拒答失效主要反映校准问题而非表征质量的证据。我们证明这一结论取决于负样本的选择。在跨越三个资源层级的九种语言中，当无害提示来自不相关的分布时（易负样本），我们复现了近乎完美的迁移效果（AUROC > 0.98）。然而，当使用XSTest对比提示——这些提示本身无害但表面特征与有害请求相似（难负样本）——时，迁移在低资源语言中崩溃，而在高资源语言中基本保持稳定。在Qwen2.5-7B-Instruct上，平均AUROC下降从英语的0.003增加（摘要原文在此处截断）。

    arXiv:2609.27758v1 Announce Type: cross  Abstract: Safety alignment in large language models is trained primarily in English, and recent work reports that the underlying harmfulness representation survives translation: English-trained probes separate harmful from harmless prompts almost as well in low-resource languages as in English. This has been taken as evidence that cross-lingual refusal failures mainly reflect calibration rather than representation quality. We show that this conclusion depends on the choice of negative examples. Across nine languages spanning three resource tiers, we replicate near-perfect transfer (AUROC > 0.98) when harmless prompts come from an unrelated distribution (easy negatives). With XSTest contrast prompts, which are benign but surface-similar to harmful requests (hard negatives), transfer collapses in low-resource languages while remaining largely stable in high-resource languages. On Qwen2.5-7B-Instruct, mean AUROC drop increases from 0.003 in English
    
[^63]: 在压力下报告：区分大语言模型统计分析中的事实性谄媚与语气性谄媚

    Reporting Under Pressure: Separating Factual and Tonal Sycophancy in LLM Statistical Analysis

    [https://arxiv.org/abs/2609.27756](https://arxiv.org/abs/2609.27756)

    该研究通过4×4因子实验设计，首次将大语言模型统计分析中的“事实性谄媚”与“语气性谄媚”区分开来，发现提示词的编辑性框架不仅会改变模型报告的语气，还会导致模型对数据结果的事实性错误陈述。

    

    大型语言模型越来越多地被要求分析数据并报告结果的含义，这一任务有别于大多数谄媚研究所关注的信念对齐或偏好对齐场景。我们测试提示词中的编辑性框架——从中性请求，到明确指示模型穷尽地寻找否定或支持某一发现的理由——是否会不仅改变模型报告的语气，还会改变其实质内容。我们采用4×4因子设计，将四种框架条件与四种真实数据模式（真实效应、一个看似有效应但未通过稳健性检验的混杂因素、统计功效充分的零结果、以及统计功效不足的零结果）进行交叉组合，共收集480个回复，并沿两个独立维度对每个回复进行评分：其对数据的事实性陈述是否偏离了正确解释，以及是否仅语气偏离而陈述本身保持正确。事实性失实集中在两个单元格中……

    arXiv:2609.27756v1 Announce Type: new  Abstract: Large language models are increasingly asked to analyze data and report what the results mean, a task distinct from the belief- or preference-alignment settings studied in most sycophancy research. We test whether editorial framing in the prompt, ranging from a neutral request to an explicit instruction to search exhaustively for reasons to discredit or to support a finding, changes not just the tone but the substance of a model's report. Across a 4 x 4 factorial design crossing four framing conditions with four ground-truth data patterns (a genuine effect, a confound that mimics an effect but fails a robustness check, a well-powered null, and an underpowered null), we collect 480 responses and score each along two independent dimensions: whether its factual claim about the data diverged from the correct interpretation, and whether only its tone diverged while the claim stayed correct. Factual misrepresentation is concentrated in two cel
    
[^64]: 用于评估新型AI辅助教育问题教学质量的预训练模型评估

    Evaluation of pre-trained models for pedagogical assessment of novel AI-assisted educational questions

    [https://arxiv.org/abs/2609.27749](https://arxiv.org/abs/2609.27749)

    该研究通过评估传统机器学习、Transformer和大语言模型在布鲁姆层级分类任务中的表现，并借助特征工程策略，寻找在AI生成的分布外教育问题上依然稳健的教学质量自动评估方法。

    

    AI辅助生成教育材料的激增已超出我们验证其教学质量的能力。使用布鲁姆分类器（Bloom Classifier）模型进行自动化评估，是一种大规模评估教育材料的有前景的方法。这些模型在同分布数据集（IID数据集）上显示出较高的准确率。然而，将相同的模型应用于新的分布外（OOD）数据集（如AI辅助生成的问题）时，可能会出现性能下降。为了找出在数据集偏移下依然稳健的分类器，我们在布鲁姆层级分类任务上评估了传统机器学习（ML）模型、Transformer模型和大语言模型。我们还探索了特征工程策略，包括引入NLP指标、将学习目标作为输入的一部分进行附加，以及文本拼接，以稳定OOD性能。我们的基线测试显示，TFPOS-IDF机器学习模型在OOD数据上表现较差（宏平均F1分数为0.48），相比之下BERT达到0.55，大语言模型（摘要原文在此处截断）。

    arXiv:2609.27749v1 Announce Type: new  Abstract: The surge in AI-assisted generation of educational materials has outpaced our capacity to validate their pedagogical quality. Automated evaluation using Bloom Classifier models is a promising approach to assess educational materials at scale. These models show high accuracy within-distribution dataset (IID Dataset). However, applying the same models to new out-of-distribution (OOD) datasets such as AI-assisted generated questions could show performance degradation. To identify robust classifiers under dataset shift, we evaluated traditional Machine Learning (ML), transformer, and Large Language models on the Bloom level classification task. We also explored feature-engineering strategies incorporating NLP metrics, appending the learning objectives as part of the input, and text splicing to stabilize OOD performance. Our baseline tests show that TFPOS-IDF ML models perform poorly on OOD (Macro F1-score 0.48) compared to BERT (0.55) and LL
    
[^65]: 面向可泛化POMDP求解的环境广群范畴论内化

    Categorical Internalisation of Environmental Groupoids for Generalisable POMDP Solving

    [https://arxiv.org/abs/2609.27745](https://arxiv.org/abs/2609.27745)

    该论文提出用范畴论将环境状态的对称轨道组织为带有规范代表元的广群，使强化学习在对称性约简的状态空间上进行，从而让智能体在等价状态间共享经验、消除冗余，提升POMDP求解的样本效率与泛化能力。

    

    本文倡导将范畴论作为在高维、部分可观测环境中组织和改进强化学习的实用框架。我们通过将状态空间划分为由对称轨道诱导的等价类来建模环境状态之间的对称性，并将每个这样的类组织为一个具有指定规范代表元的广群。这使得智能体能够在许多相似的环境状态之间同时共享所学到的知识，而不是将每个朝向或位置都当作一个全新的问题来处理。因此，学习是在对称性约简后的状态空间上进行的，每个轨道仅由一个代表元表示，在保持结构的同时消除了冗余并提高了样本效率。我们在标准强化学习流程中实现了该框架，并在部分可观测基准任务上评估了两种不同的方法，证明了基于轨道的划分……（原文摘要在此处截断）

    arXiv:2609.27745v1 Announce Type: new  Abstract: This paper advocates category theory as a practical framework for structuring and improving rein- forcement learning in high-dimensional, partially observable environments. We model symmetries between environmental states by partitioning the state space into equivalence classes induced by sym- metry orbits, and organise each such class as a groupoid with a designated canonical representative. This allows the agent to share what it learns across many similar environmental states simultaneously, rather than treating every orientation or position as an entirely new problem. Learning is thus carried out on a symmetry-reduced state space with each orbit represented once, preserving structure while eliminating redundancy and improving sample efficiency.   We implement this framework within standard reinforcement learning pipelines and evaluate two different approaches on partially observable benchmarks, demonstrating that orbit-based partition
    
[^66]: InfiNoVA：用于视角不变机器人策略的无限新视角增强

    InfiNoVA: Infinite Novel View Augmentation for Viewpoint Invariant Robot Policies

    [https://arxiv.org/abs/2609.27734](https://arxiv.org/abs/2609.27734)

    InfiNoVA通过将多相机示教数据重建为随时间变化的3D高斯表示并渲染几何一致的新视角观测，在保持状态-动作对应关系的同时实现视角不变的机器人策略数据增强。

    

    视觉-语言-动作（VLA）策略往往强烈依赖于训练时所见的相机视角，导致在从未见过的视角部署时性能大幅下降。从足够多样的物理视角收集示教数据成本高昂，且仍只能对视角空间提供稀疏的覆盖。我们提出了InfiNoVA，这是一个数据增强框架，能将同步的多相机示教数据转换为几何一致的新视角训练视图的密集分布。InfiNoVA将每个操作轨迹重建为随时间变化的3D高斯表示，并从采样的相机位姿渲染新的观测，同时保持原始的状态-动作对应关系。这种显式场景表示提高了帧级保真度和时间一致性，同时减少了生成式新视角合成中出现的任务关键性幻觉。在四个真实世界操作任务上（摘要在此截断），

    arXiv:2609.27734v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) policies often rely strongly on the camera viewpoints seen during training, causing substantial performance degradation when deployed from unseen perspectives. Collecting demonstrations from sufficiently diverse physical viewpoints is expensive and still provides only sparse coverage of the viewpoint space. We introduce InfiNoVA, a data-augmentation framework that converts synchronized multi-camera demonstrations into a dense distribution of geometrically consistent training views. InfiNoVA reconstructs each manipulation trajectory as a time-varying 3D Gaussian representation and renders novel observations from sampled camera poses while preserving the original state-action correspondence. This explicit scene representation improves frame-level fidelity and temporal consistency while reducing task-critical hallucinations observed in generative novel-view synthesis. Across four real-world manipulation tasks,
    
[^67]: AI驱动的神经代理模型用于认知-情感神经调控靶点的计算机模拟设计

    AI-Driven Neural Surrogates for In Silico Design of Cognitive-Affective Neuromodulation Targets

    [https://arxiv.org/abs/2609.27729](https://arxiv.org/abs/2609.27729)

    该论文提出一个AI驱动的神经代理框架，结合fMRI解码、深度生成建模和受约束的潜空间引导，在不进行物理刺激的情况下从fMRI活动快照中计算机模拟设计认知-情感神经调控靶点并预测其感知效果。

    

    在神经精神病学中，主要目标往往不仅是解码大脑活动，而是改变它，例如减轻负性情感偏见或过度显著的记忆。受控制理论启发，我们开发了一个AI驱动的神经代理框架，该框架从刺激诱发的fMRI活动快照中提出候选的表征变化并测试其预测的感知效果，而无需物理刺激。该框架结合了fMRI解码、深度生成建模和受约束的潜空间引导，其中效价和记忆性仅作为应用示例。利用来自四名深度采样的Natural Scenes Dataset参与者的超过36,000个图像-fMRI观测数据，被试特异性模型从视觉响应皮层中恢复了粗略的生成结构（双向识别率为0.79-0.88；随机水平为0.5）。分级扰动被重建为图像，并通过自动评分器和人类评分进行评估。

    arXiv:2609.27729v1 Announce Type: cross  Abstract: In neuropsychiatry, the primary goal is often not only to decode brain activity but to change it, for example to lessen a negative affective bias or an overly salient memory. Motivated by control theory, we develop an AI-driven neural-surrogate framework that proposes candidate representational changes and tests their predicted perceptual effects from snapshots of stimulus-evoked fMRI activity, without physical stimulation. The framework combines fMRI decoding, deep generative modeling, and constrained latent-space steering. Valence and memorability are used only as worked examples. Using more than 36,000 image-fMRI observations from four deeply sampled Natural Scenes Dataset participants, subject-specific models recovered coarse generative structure from visually responsive cortex (two-way identification, 0.79-0.88; chance, 0.5). Graded perturbations were reconstructed as images and evaluated with automated scorers and human ratings f
    
[^68]: 路径至关重要：在知识图谱问答（KGQA）中超越答案准确率评估小型语言模型

    The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA

    [https://arxiv.org/abs/2609.27669](https://arxiv.org/abs/2609.27669)

    该论文提出基于THESEUS框架的受控评估方法，让冻结的小型语言模型逐步执行知识图谱导航动作，并引入路径保真度指标，从而超越单纯的答案准确率来评估模型在知识图谱问答中的导航与推理能力。

    

    小型语言模型（SLM）越来越多地与知识图谱（KG）配对使用，然而端到端的知识图谱问答将图访问、搜索、导航、推理与答案生成等多个环节混杂在一起。这种耦合使得我们既难以判断一个小型语言模型能否忠实执行问题所隐含的推理路径，也难以将失败归因于导航本身而非流程中的其他阶段。我们通过采用THESEUS导航与可追溯性框架，并使用冻结的、开箱即用的小型语言模型作为局部动作策略，来单独隔离这一能力。在每一跳中，环境展示所有合法的出边图动作，模型选择一个可执行的图动作并决定是否停止，全程无需任务特定的参数更新、模型控制的束搜索或自由形式的答案生成。这一受控设置使我们能够使用Hits@1评估终端答案准确率，同时借助路径编辑距离等指标评估路径保真度。

    arXiv:2609.27669v1 Announce Type: cross  Abstract: Small language models (SLMs) are increasingly paired with knowledge graphs (KGs), yet end-to-end KG question answering conflates graph access, search, navigation, reasoning, and answer generation. This coupling makes it difficult both to determine whether an SLM can faithfully execute the reasoning path implied by a question and to attribute failures to navigation rather than to other stages of the pipeline. We isolate this capability by employing the THESEUS navigation and traceability framework and using frozen, off-the-shelf SLMs as local action policies. At each hop, the environment exposes the legal outgoing graph actions, and the model selects one executable graph action and decides whether to stop, without task-specific parameter updates, model-controlled beam search, or free-form answer generation. This controlled setting allows us to evaluate terminal-answer accuracy with Hits@1 together with path fidelity, using Path Edit Dis
    
[^69]: 进化稳定性并不能保证学习可达性：合作涌现的多智能体强化学习视角

    Evolutionary Stability Does Not Guarantee Learning Accessibility: A Multi-Agent Reinforcement Learning Perspective on Cooperation Emergence

    [https://arxiv.org/abs/2609.27664](https://arxiv.org/abs/2609.27664)

    该论文通过政府-平台-用户三方博弈模型证明，演化博弈论中的进化稳定合作结果并不保证有限样本的去中心化多智能体强化学习能够通过局部奖励反馈达到同样的合作结果，揭示了演化稳定性与学习可达性之间的本质区别。

    

    合作涌现是多智能体系统中的一个核心问题，因为去中心化的智能体必须在适应其他智能体不断变化的行为的同时进行协调。演化博弈论能够识别出策略上稳定的结果，但在种群调整动态下的稳定性并不意味着有限样本的学习智能体能够通过局部奖励反馈达到同样的结果。我们在一个透明的、以治理为动机的三方博弈（涉及政府、平台企业和用户）中研究这一区别。我们针对固定的阶段博弈激励推导了复制者动态，在对称初始条件网格上评估了合作的演化吸引域，并将其与三种去中心化的基于价值的学习器的学习吸引域估计结果进行比较。学习分析部分在相同的收益环境下采用了带 ε-贪婪动作选择的独立Q学习、缩放玻尔兹曼探索以及SA-EA BQL等方法。

    arXiv:2609.27664v1 Announce Type: new  Abstract: Cooperation emergence is a central problem in multi-agent systems because decentralized agents must coordinate while adapting to the changing behavior of others. Evolutionary game theory identifies strategically stable outcomes, but stability under a population adjustment dynamic need not imply that finite-sample learning agents can reach the same outcome through local reward feedback.   We study this distinction in a transparent three-agent governance-motivated game involving a government, a platform firm, and users. We derive replicator dynamics for the fixed stage-game incentives, evaluate the cooperative evolutionary basin on a symmetric initial-condition grid, and compare it with learning-basin estimates for three decentralized value-based learners. The learning analysis uses independent Q-learning with $\varepsilon$-greedy action selection, scaled Boltzmann exploration, and SA--EA BQL under the same payoff environment and outcome c
    
[^70]: FLEET：从Logits熵到文本生成中的增强轨迹

    FLEET: From Logits Entropy to Enhanced Trajectories in Text Generation

    [https://arxiv.org/abs/2609.27657](https://arxiv.org/abs/2609.27657)

    FLEET通过引入记忆机制，将生成过程表示为基于熵阈值状态的稀疏轨迹，并利用每token效用分数调整logits，实现了与重复采样相同的准确率但速度提升3倍。

    

    基于大语言模型（LLM）的解决方案通常依赖温度采样，通过从补全分布中聚合多个样本来提高准确性和稳定性。然而，这种无记忆的方法本质上是次优的：由于缺乏对先前生成结果及其评估的了解，随着采样数量增加，会产生越来越多的语义重复答案，导致收益递减。为了解决这一局限性，我们提出了FLEET，这是一种将记忆机制集成到生成过程中的新方法。FLEET将每次生成表示为通过熵超过预定阈值状态的稀疏轨迹，并利用这些轨迹推断每个token的效用分数来调整logits。基准评估表明，FLEET在与重复采样基线达到相同准确率的情况下实现了3倍加速，并在复杂代码任务上显著提高了准确率。

    arXiv:2609.27657v1 Announce Type: cross  Abstract: Solutions based on large language models (LLMs) often rely on temperature sampling to improve accuracy and stability by aggregating multiple samples from the completion distribution. However, this memoryless approach is inherently suboptimal: because it lacks awareness of prior generations and their evaluations, it produces an increasing proportion of semantically duplicate answers as more samples are drawn, leading to diminishing returns. To address this limitation, we introduce FLEET, a novel method that integrates a memory mechanism into the generation process. FLEET represents each generation as a sparse trajectory through states whose entropy exceeds a predefined threshold and uses these trajectories to infer per-token utility scores that adjust the logits. Benchmark evaluations demonstrate that FLEET achieves the same accuracy as the repeated sampling baseline, with a 3x speedup, and substantially improves accuracy on complex cod
    
[^71]: InternW0：面向高效真实世界交互的基础物理世界模型

    InternW0: A Foundational Physical World Model for Efficient Real-World Interactions

    [https://arxiv.org/abs/2609.27656](https://arxiv.org/abs/2609.27656)

    InternW0通过非对称的视频专家-动作专家架构联合学习视觉动态预测与连续机器人控制，并借助逐层K/V缓存重用和观测条件化上下文路由，避免了每次动作更新都重新生成未来，从而实现高效的真实世界交互。

    

    物理智能需要的不仅仅是预测世界如何演化：预测结果必须在世界持续变化时保持可执行性。我们提出InternW0，这是上海人工智能实验室InternW物理世界模型系列的首个实例，其构建围绕全模态接口、异步多频率处理，以及部分观测和外部影响下的局部物理建模。InternW0通过带有流匹配的非对称视频-动作架构，联合学习未来视觉动态与连续机器人控制。其中高容量的视频专家提供更长时间跨度的预测上下文，而轻量级的动作专家则以更快的时标运行。InternW0无需为每次动作更新都重新生成未来，而是重用逐层K/V缓存，并通过观测条件化的上下文路由将其适配到新观测到的状态。领域专用接口和软提示支持异构的（摘要在此处截断）

    arXiv:2609.27656v1 Announce Type: cross  Abstract: Physical intelligence requires more than predicting how the world may evolve: predictions must remain actionable as the world continues to change. We introduce InternW0, the first instantiation of the InternW physical world model series from Shanghai AI Laboratory, built around omnimodal interfaces, asynchronous multi-frequency processing, and local physical modeling under partial observations and external influences. InternW0 jointly learns future visual dynamics and continuous robot control through an asymmetric video--action architecture with flow matching. A high-capacity video expert provides longer-horizon predictive context, while a lightweight action expert operates at a faster timescale. Instead of regenerating the future for every action update, InternW0 reuses layerwise K/V and adapts it to newly observed states through observation-conditioned context routing. Domain-specific interfaces and soft prompts support heterogeneous
    
[^72]: 面向大规模交通预测的局部异质性与跨区域上下文学习

    Learning Local Heterogeneity and Cross-Region Context for Large-Scale Traffic Forecasting

    [https://arxiv.org/abs/2609.27637](https://arxiv.org/abs/2609.27637)

    提出LoReST局部-区域时空网络，在节点邻域和路网区域两个互补粒度上建模空间依赖，兼顾局部异质性捕获与跨区域上下文获取，实现高效的大规模交通流预测。

    

    交通流预测对智能交通系统至关重要。大规模交通预测需要联合建模局部空间依赖和跨区域上下文。由于道路属性和行驶方向的差异，地理位置相邻节点之间的空间依赖具有异质性，而通过全对节点交互获取全局信息会带来巨大的计算开销。因此，在捕获局部异质性的同时高效获取长程上下文，仍然是大尺度交通预测中的一个重要挑战。为解决这些挑战，我们提出了LoReST，一个局部-区域时空网络，它在两个互补的粒度上建模空间依赖：节点邻域和路网区域。具体而言，关系感知的局部聚合通过道路和方向特定的特征来捕获地理邻域内的异质依赖……

    arXiv:2609.27637v1 Announce Type: cross  Abstract: Traffic flow forecasting is essential to intelligent transportation systems. Large-scale traffic forecasting requires jointly modeling local spatial dependencies and cross-region context.Spatial dependencies between geographically neighboring nodes are heterogeneous due to differences in road identity and travel direction, while acquiring global information through allpairs node interactions incurs substantial computational costs. Therefore, capturing local heterogeneity while efficiently acquiring long-range context remains an important challenge in largescale traffic forecasting. To address these challenges, we propose LoReST, a Local-Region Spatial Temporal network that models spatial dependencies at two complementary granularities: node neighborhoods and road network regions. Specifically, relation-aware local aggregation captures heterogeneous dependencies within geographic neighborhoods through road and direction specific feature
    
[^73]: 面向受监管金融的合规AI基础设施：面向DACH地区金融业务的分层多智能体框架与分布式账本技术（DLT）审计追踪

    Compliant AI Infrastructure for Regulated Finance: A tiered multi-agent framework with DLT audit trails for financial operations in DACH

    [https://arxiv.org/abs/2609.27632](https://arxiv.org/abs/2609.27632)

    提出了一种合规优先的分层多智能体AI架构，通过监管意图矩阵、策略编译器将监管转化为禁止事项与义务，并借助许可式DAG审计追踪实现DACH及欧盟地区金融AI运营的可重放、可溯源与可移植的合规保障。

    

    我们提出了一种面向受监管金融的“合规优先”AI架构，该架构将监管视为导向层而非确定性的规则集。监管意图与风险敞口矩阵提供了紧凑的分类手段，随后由受治理的策略编译器将其映射为具体的禁止事项、义务和运行时预算。禁止事项约束可行性并阻止外部化，而义务则通过必须满足明确可采性标准的产物来扩展任务。委员会的激活保持由策略驱动且成比例，在确保监管监督的同时维持效率。证据、决策和原因代码绑定到具有确定性时间戳的许可式有向无环图（DAG）上，从而支持重放、来源检查以及失败原因的清晰归因。基于生效日期的条款级法律索引和基于能力的智能体路由，确保了在DACH地区及更广泛欧盟范围内的可移植性。其结果是提供保障。

    arXiv:2609.27632v1 Announce Type: cross  Abstract: We present a compliance-first architecture for AI in regulated finance that treats regulation as an orientation layer rather than a deterministic ruleset. A matrix of regulatory intent and exposure provides a compact classification handle, which a governed policy compiler then maps into concrete prohibitions, obligations and runtime budgets. Prohibitions constrain feasibility and block externalisation, while obligations extend tasks with artefacts that must meet explicit admissibility criteria. Committee activation remains policy-driven and proportionate, preserving efficiency while ensuring supervisory oversight. Evidence, decisions and reason codes are bound to a permissioned DAG with deterministic timestamping, enabling replay, provenance checks and clear attribution of failure. Clause-level legal indexing with effective dates and capability-based agent routing ensure portability across DACH and the wider EU. The result is assurance
    
[^74]: SHRAV：用于物理建模与逆向设计的状态-假设-推理-行动-验证框架

    SHRAV: State-Hypothesis-Reason-Action-Verify Framework for Physical Modeling and Inverse Design

    [https://arxiv.org/abs/2609.27621](https://arxiv.org/abs/2609.27621)

    提出了SHRAV框架，通过带声明复用边界的状态延续核心实现可复用计算，统一支持物理建模与逆向设计，并在计算光刻中仅用四次固定权重更新就将空间图像交并比从0.5313提升至0.8153。

    

    物理建模与逆向设计需要能够从可复用状态继续进行的计算。我们提出了SHRAV，一个围绕状态、假设、推理、行动和验证组织起来的、与具体架构无关的计算框架。其核心机制是一个状态延续核心，具有明确声明的复用边界，并为学习演化和数值量分配了明确的角色。前向配置演化预测性状态并读取物理响应；逆向设计配置则额外生成面向目标的修改并消耗评估器反馈。电磁世界模型研究被映射到前向配置，本文报告了选定的读取与复用诊断结果。计算光刻展示了一个逆向设计配置：在独立的标量光瞳重放条件下，四次固定权重的设计更新将阈值化空间图像的交并比从0.5313提升至0.8153，最大绝对……

    arXiv:2609.27621v1 Announce Type: new  Abstract: Physical modeling and inverse design require computation that can continue from reusable state. We introduce SHRAV, an architecture-independent computational framework organized around State, Hypothesis, Reason, Action, and Verify. Its central mechanism is a state-continuation core with declared reuse boundaries and explicit roles for learned evolution and numerical quantities. Forward configurations evolve predictive state and read out physical responses; inverse-design configurations additionally generate target-directed modifications and consume evaluator feedback. Electromagnetic world-model studies are mapped to forward configurations, with selected readout and reuse diagnostics reported here. Computational lithography demonstrates an inverse-design configuration: four fixed-weight design updates improve thresholded aerial-image intersection-over-union from 0.5313 to 0.8153 under independent scalar-pupil replay, with maximum absolut
    
[^75]: InGuard：迈向安全文本到图像生成的广义内部护栏

    InGuard: Towards Generalized Inner Guardrail for Safe Text-to-Image Generation

    [https://arxiv.org/abs/2609.27620](https://arxiv.org/abs/2609.27620)

    本文提出InGuard安全框架，在文本到图像生成流程内部基于模型自身表征进行防护，无需修改基础模型参数，从而提升提示词风险筛查准确性并支持对风险提示词进行调整以生成安全图像。

    

    现代文本到图像（T2I）模型能够根据任意用户提示词生成高质量图像，但它们同样容易生成不适宜工作场所（NSFW）的内容。传统的外部护栏由两个组件构成：一个在生成前检查风险的提示词分类器，以及一个检查完全生成图像的事后图像分类器。在这种设计中，两个分类器均在生成流程之外运行，且不使用模型自身的表征。这种分离可能限制提示词筛查的准确性，而图像端的检查只有在花费了完整的生成成本之后才会进行。此外，被标记的提示词只能被拒绝，即使它本可以被调整以生成安全的图像。在本工作中，我们提出了内部护栏，这是一个在生成流程内部基于模型自身表征运行的安全框架，且不改动基础模型的参数。首先，一个风险分类器将每个提示词评定为不安全……（摘要在此处被截断）

    arXiv:2609.27620v1 Announce Type: cross  Abstract: Modern text-to-image (T2I) models generate high-quality images from arbitrary user prompts, yet they can just as easily produce not-safe-for-work (NSFW) content. Conventional outer guardrails consist of two components: a prompt classifier that checks for risk before generation, and a post-hoc image classifier that checks the fully generated image. In this design, both classifiers operate outside the generation pipeline and do not use the model's own representations. This separation can limit prompt-screening accuracy, while the image-side check runs only after the full generation cost has been spent. Moreover, a flagged prompt can only be rejected, even when it could be adjusted to produce a safe image. In this work, we propose the Inner Guardrail (InGuard), a safety framework that works inside the pipeline on the model's own representations, leaving base-model parameters untouched. First, a risk classifier grades each prompt as unsafe
    
[^76]: BiCFlow-MER：基于条件传输的判别式与生成式多模态情感识别协同框架

    BiCFlow-MER: Orchestrating Discriminative and Generative Multimodal Emotion Recognition via Conditional Transport

    [https://arxiv.org/abs/2609.27615](https://arxiv.org/abs/2609.27615)

    提出BiCFlow-MER条件流框架，将音频-文本多模态情感识别建模为结构化情感空间中的生成式证据传输，协同判别式融合与生成式推理，从而更好地保留模态特异线索并处理跨模态冲突信息。

    

    在多模态情感识别（MER）中，人类情感状态是通过整合来自多个模态的互补线索来推断的。在音频-文本MER中，情感线索往往与说话人风格和词汇内容相互纠缠，而跨模态的不一致性进一步使证据的整合变得更加复杂。在传统的判别式融合方法下，多模态证据被压缩为单一的终端预测，模态特异性线索和冲突信息未能得到充分保留。相比之下，在大型生成式情感模型中，情感推理通常嵌入在语言解码过程中，导致情感证据保持隐式状态，难以在结构化空间中进行验证。为解决这些局限性，本文提出了BiCFlow-MER（双向条件流多模态情感识别），这是一个条件流框架，将音频-文本MER表述为结构化情感[空间中的生成式证据传输]（摘要在此处截断）。

    arXiv:2609.27615v1 Announce Type: new  Abstract: In multimodal emotion recognition (MER), human affective states are inferred by integrating complementary cues from multiple modalities. In audio-text MER, affective cues are often entangled with speaker style and lexical content, while cross-modal disagreement further complicates how the evidence should be integrated. Under conventional discriminative fusion, multimodal evidence is compressed into a terminal prediction, with modality-specific cues and conflict information insufficiently preserved. In large generative affective models, by contrast, affective reasoning is typically embedded in language decoding, leaving emotion evidence implicit and difficult to verify in a structured space. To address these limitations, BiCFlow-MER (Bidirectional Conditional Flow for Multimodal Emotion Recognition) is proposed as a conditional-flow framework in which audio-text MER is formulated as generative evidence transport within a structured emotio
    
[^77]: Jev能评判放射学报告吗？评估一个系统一模型的临床事实性

    Can Jev Judge Radiology Reports? Evaluating a System One Model for Clinical Factuality

    [https://arxiv.org/abs/2609.27607](https://arxiv.org/abs/2609.27607)

    提出用系统一决策模型Jev作为低成本评判器，双向检测AI放射学报告中无依据的主张和遗漏，在两个基准上与专家错误计数达到较强相关性，且单问题配置可减少约44%的token成本。

    

    AI生成的放射学报告可能看起来与医生的报告相似，但却遗漏了异常、添加了无依据的发现，或颠倒了其存在状态。衡量这些事实性差异对于评估报告生成器至关重要。我们研究了Jev——一个系统一决策模型——作为判断其与医生撰写的参考报告一致性的简单、低成本评判器。我们的评估器检查每条陈述是否得到另一份报告的支持，并将这些判断双向结合，以捕捉无依据的主张和遗漏。单问题配置在RadEvalX上达到0.573、在RadEvalExpert上达到0.398的Kendall相关系数（与专家错误计数相比），在匹配的分解和聚合条件下优于开放的自然语言推理评判器。每条陈述仅需一个支持性问题即可保持与七个问题相当的专家一致性，同时减少43-45%的判断输入token。按公开的API价格计算，判断成本低于三……

    arXiv:2609.27607v1 Announce Type: cross  Abstract: An AI-generated radiology report can resemble a physician's report while omitting an abnormality, adding an unsupported finding, or reversing its presence. Measuring these factual differences is essential for evaluating report generators. We study Jev, a System One decision model, as a simple, low-cost judge of agreement with physician-written reference reports. Our evaluator checks whether each statement is supported by the other report and combines these judgments in both directions to capture unsupported claims and omissions. A single-question configuration reaches Kendall correlations of 0.573 on RadEvalX and 0.398 on RadEvalExpert with expert error counts, outperforming an open natural language inference judge under matched decomposition and aggregation. One support question per statement retains similar expert agreement to seven while using 43-45% fewer judgment input tokens. At the documented API price, judgments cost under thre
    
[^78]: 状态接地条件化：为方向依赖实时状态的用户端LLM智能体添加包装层

    State-Grounded Conditioning: Wrapping User-Facing LLM Agents Where Direction Depends on Live State

    [https://arxiv.org/abs/2609.27606](https://arxiv.org/abs/2609.27606)

    该论文提出状态接地条件化（SGC）设计原则，通过感知、接地和交互三个包装器将依赖实时状态的控制外部化为规则内核，以解决用户端LLM智能体中“方向漂移”这一失败问题，并将平均首词延迟从6.1秒降低到1.5秒。

    

    我们提出了状态接地条件化，这是一种面向用户的LLM智能体的设计原则，适用于必须以实时用户状态（游戏状态、会话历史、实时库存）为条件进行响应的智能体；同时我们提出了一类独特的失败模式，称为“方向漂移”：即任务本身完成，但所选方向与当前状态不一致的响应。SGC通过具有显式条件依赖关系的感知、接地和交互三个包装器，将依赖状态的控制外部化为基于结构化输入和三种主要状态切片的规则内核。我们在一个包含200个会话（约1,000个助手模型轮次）的匿名化基准上评估了SGC，该基准来自一个游戏内对话式教练智能体，用于引导玩家进行连续的竞技比赛。我们报告了平均首词延迟以及五项人工标注的对话质量指标，这些指标共同涵盖了事实接地和教练式引导进展两个维度。感知包装器将平均首词延迟保持在1.5秒（相比之下为6.1秒……）

    arXiv:2609.27606v1 Announce Type: new  Abstract: We introduce State-Grounded Conditioning (SGC), a design principle for user-facing LLM agents that must condition on live user state (game state, session history, live inventory), and a distinct failure class we call direction drift: task-complete responses whose chosen direction misaligns with the current state. SGC externalises state-dependent control into rule kernels over structured inputs and three primary state slices, via Perception, Grounding, and Interaction wrappers with explicit conditioning dependencies. We evaluate SGC on a 200-session anonymised benchmark ($\approx$1,000 assistant model turns) from an in-game conversational coaching agent that guides players through consecutive competitive matches, reporting mean first-token latency and five human-annotated dialogue-quality metrics that jointly cover factual grounding and coach-like guidance progression. The Perception wrapper holds mean first-token latency at 1.5s (vs. 6.1
    
[^79]: 当上下文产生误导时：大语言模型中具备“管辖权”的上下文学习

    When Context Misleads: In-context Learning with Jurisdiction in Large Language Models

    [https://arxiv.org/abs/2609.27603](https://arxiv.org/abs/2609.27603)

    该论文指出现有ICL后训练方法忽视“上下文权威性”判断能力并提出FakeContextBench基准，同时推出J-ICL后训练框架，将上下文验证融入训练过程，防止模型被误导性上下文欺骗并缓解ICL微调带来的现实准确率下降问题。

    

    上下文学习（ICL）已成为现代大语言模型部署的基石。然而，现有的ICL后训练方法存在一个关键盲区：它们擅长从示例中提取模式，却常常忽视“上下文权威性”（context authority），即判断上下文信息是否应当主导最终答案的能力。为了对这一能力进行基准测试，我们提出了FakeContextBench，其中包含涵盖七个领域的伪科学论断。我们对商业模型和开源模型的评估表明，仅依靠大规模预训练不足以实现可靠的上下文权威性判别。此外，流行的ICL微调方法会增加模型对误导性上下文的易感性，使现实准确率相比基座模型最多下降14.95个百分点。为解决这一权衡问题，我们提出了管辖性上下文学习，这是一个将上下文验证纳入训练的后训练框架。

    arXiv:2609.27603v1 Announce Type: cross  Abstract: In-Context Learning (ICL) has become a cornerstone of modern LLM deployment. However, existing ICL post-training methods have a critical blind spot: they excel at extracting patterns from demonstrations while often neglecting context authority, the ability to determine whether contextual information should govern the final answer. To benchmark this capability, we introduce FakeContextBench, which contains pseudoscientific claims across seven domains. Our evaluation of commercial and open-source models shows that large-scale pre-training alone is insufficient for reliable context-authority discrimination. Moreover, prevalent ICL fine-tuning methods can increase susceptibility to misleading context, reducing reality accuracy by up to 14.95 percentage points relative to the base model. To address this trade-off, we propose Jurisdiction In-Context Learning (J-ICL), a post-training framework that incorporates context validation into the tra
    
[^80]: 隐藏而非删除：网络如何抑制纠缠特征

    Hidden not Deleted: How Networks Suppress Entangled Features

    [https://arxiv.org/abs/2609.27593](https://arxiv.org/abs/2609.27593)

    该论文证明线性概念擦除方法在特征密集叠加纠缠时会连带破坏非目标特征，而梯度下降训练的网络会根据初始化收敛到“镜像”或“阴影”两种非线性电路级解决方案之一，且两种方案都保留了被擦除特征的可测量表征痕迹，仅需单个标量补丁即可恢复、无需再训练。

    

    通过线性投影实现的概念擦除方法假设特征占据可分离的子空间。我们证明该假设在密集叠加情况下会失效：当两个特征被迫形成共享同一子空间的对跖对时，最先进的线性擦除方法会同时破坏两者，而不仅仅是目标特征。通过梯度下降训练的网络则以非线性方式解决这一问题，但方式并不统一：根据初始化的不同，它们会收敛到两种不同的电路级解决方案之一，我们称之为“镜像”解决方案和“阴影”解决方案。我们将这种分叉现象映射为特征纠缠程度的函数，证明它反映的是稳定的吸引子结构而非实验设置的伪影，并通过针对性的因果干预证明，这两种解决方案都会在被擦除特征的表征中留下大量可测量的完整痕迹，仅需一个标量补丁即可恢复，而无需任何进一步的训练。这一现象类似于——（摘要原文在此处截断）

    arXiv:2609.27593v1 Announce Type: cross  Abstract: Concept erasure methods that operate via linear projection assume that features occupy separable subspaces. We show this assumption fails under dense superposition: when two features are forced into an antipodal pair sharing a single subspace, state-of-the-art linear erasure destroys both, not just the target. Networks trained with gradient descent instead solve this problem non-linearly, but not uniformly: they converge to one of two distinct circuit-level solutions depending on initialization, which we call mirror and shadow solutions. We map this bifurcation as a function of feature entanglement, show it reflects a stable attractor structure rather than an artifact of our setup, and use targeted causal interventions to demonstrate that both solutions leave a substantial, measurable trace of the erased feature's representation intact, recoverable through a single scalar patch rather than requiring any further training. This mirrors a
    
[^81]: 能力流形与机器学习缩放定律

    The Capability Manifold and ML Scaling Laws

    [https://arxiv.org/abs/2609.27588](https://arxiv.org/abs/2609.27588)

    本文提出“能力流形”这一多维框架，通过有界缩放函数将模型下游能力（如推理、规划等）与预训练、后训练和测试时资源关联起来，弥补了传统缩放定律仅依赖损失无法刻画模型能力差异的不足。

    

    现有的机器学习（ML）缩放定律将预测损失与计算量、模型参数和数据量相关联。然而，随着模型越来越多地通过智能体框架进行部署，仅凭损失已不足以刻画下游性能：损失相近的模型在推理、检索、规划和适应等方面可能表现出不同的能力。然而，目前尚无统一的框架将这些能力与机器学习全生命周期中可获得的耦合资源联系起来。我们通过引入“能力流形”来弥合这一差距，这是一个多维框架，通过有界的缩放函数将下游能力映射到预训练、后训练和测试时资源上。解析雅可比矩阵量化了能力对资源变化及资源间相互作用的敏感性。作为初步应用，我们将Kaplan型和Chinchilla型缩放定律以及测试时计算嵌入到该框架中，展示了现有缩放关系如何能够被统一……

    arXiv:2609.27588v1 Announce Type: cross  Abstract: Existing machine learning (ML) scaling laws relate predictive loss to compute, model parameters, and data. However, as models are increasingly deployed through agentic harnesses, loss alone is insufficient to characterize downstream performance: models with similar loss can exhibit different capabilities in reasoning, retrieval, planning, and adaptation. Yet, no unified framework connects such capabilities to the coupled resources available across the ML lifecycle. We bridge this gap by introducing a capability manifold, a multidimensional framework mapping downstream capabilities to pre-training, post-training, and test-time resources through bounded scaling functions. Analytical Jacobians quantify capability sensitivity to resource changes and interactions. As an initial application, we embed Kaplan- and Chinchilla-type scaling laws and test-time compute within the framework, demonstrating how existing scaling relationships can be un
    
[^82]: DCRL：通过策略-奖励流形对齐实现解耦与耦合的强化学习

    DCRL: Decoupling and Coupling Reinforcement Learning via Policy-Reward Manifold Alignment

    [https://arxiv.org/abs/2609.27572](https://arxiv.org/abs/2609.27572)

    提出DCRL方法，从几何视角将大语言模型推理建模为逻辑推理、评估与表示三个耦合子流形，并通过策略-奖励流形对齐来解决现有奖励系统中优化不稳定和奖励欺骗的问题。

    

    强化学习（RL）已成为提升大语言模型（LLM）推理能力的关键范式。然而，现有的奖励系统，如基于规则的系统和基于奖励模型的系统，往往存在优化不稳定和奖励欺骗（reward hacking）等问题。在本工作中，我们从几何视角重新审视大语言模型的通用推理，将其概念化为一个由三个相互依赖的子流形构成的耦合流形：逻辑推理、评估和表示。基于这一视角，强化学习中的响应生成可以被解释为从评估流形中解耦的过程，而奖励估计则对应于从逻辑推理流形中解耦的过程。基于规则和基于奖励模型的强化学习系统的局限性，可以从几何上解释为强化学习过程中策略-奖励流形的失配问题。为解决上述错位问题，我们提出了解耦与耦合强化学习方法……（摘要在此处截断）

    arXiv:2609.27572v1 Announce Type: cross  Abstract: Reinforcement learning (RL) has emerged as a key paradigm for improving the reasoning capabilities of large language models (LLMs). However, existing reward systems, such as rule-based and reward-model-based, often exhibit issues such as unstable optimization and reward hacking. In this work, we revisit the general reasoning of LLMs from a geometric perspective, conceptualizing it as a coupled manifold composed of three interdependent sub-manifolds: logical deduction, evaluation, and representation. Based on this perspective, response generation in RL can be interpreted as a decoupling process from the evaluation manifold, while reward estimation corresponds to a decoupling process from the logical deduction manifold. The limitations of rule-based and reward-model RL systems can be geometrically interpreted as the mismatch of policy-reward manifolds during RL process. To address the aforementioned misalignment, we propose Decoupling an
    
[^83]: FDE-Bench：评估用于部署环境配置的大语言模型智能体

    FDE-Bench: Evaluating LLM Agents for Deployment Environment Configuration

    [https://arxiv.org/abs/2609.27571](https://arxiv.org/abs/2609.27571)

    提出了 FDE-Bench 基准，通过 136 个涵盖 Docker、Compose 和 Kubernetes 的部署配置任务，采用程序化门控检查和对抗性发布门控，评估大语言模型智能体将应用代码部署为可运行系统的能力并防止投机取巧。

    

    部署要求智能体将应用程序代码转化为一个可运行的系统，使其服务相互连接、达到就绪状态并保持可观测。FDE-Bench 通过 136 个部署配置任务评估这一能力，任务涵盖 Docker 镜像、多服务 Compose 堆栈以及 Kubernetes，包含全新构建和诊断修复两种模式。智能体提交声明式工件，这些工件会在一个纯净环境中被收集、重新构建并重新部署。四层门控二元检查用于衡量构建、就绪性、行为以及与部署规范的符合性，采用程序化检查而无需大语言模型评判。一个四臂发布门控要求存在能够解决问题的参考方案，并拒绝那些被“无所作为”、“照抄规范”或“通用占位符”提交所解决的任务。发布的检查注解揭示了 2,145 项检查与其规范之间的对应关系，其中包括七个已记录的缺口。另外三种对抗性策略用于测试任务中的投机捷径……（摘要在此处截断）

    arXiv:2609.27571v1 Announce Type: cross  Abstract: Deployment requires an agent to turn application code into a running system whose services connect, become ready, and remain observable. FDE-Bench evaluates this capability with 136 deployment-configuration tasks spanning Docker images, multi-service Compose stacks, and Kubernetes, in greenfield and diagnose-and-repair modes. Agents submit declarative artifacts that are collected, rebuilt, and redeployed in a pristine environment. Four gated binary check layers measure build, readiness, behavior, and conformance to the deployment specification, using programmatic checks without an LLM judge. A four-arm release gate requires a resolving reference solution and rejects tasks solved by do-nothing, specification-transcription, or generic-stub submissions. The released check annotations expose the link between 2,145 checks and their specifications, including seven documented gaps. Three additional adversarial strategies test shortcuts in the
    
[^84]: TNLearn：一个面向任务驱动神经元的开源Python软件包

    TNLearn: An Open Source Python Package for Task-based Neurons

    [https://arxiv.org/abs/2609.27564](https://arxiv.org/abs/2609.27564)

    TNLearn是一个开源Python软件包，实现了任务驱动神经元和网络的自动化构建与顺畅训练，推动了“针对特定任务定制神经元”这一新范式的科研与产业应用。

    

    大脑并不依赖单一类型的神经元来执行各种任务；相反，它为不同的任务设计了不同的神经元。与基于任务的架构相比，基于任务神经元的理念代表了一种范式转变。该理念认为，解决特定问题需要定制化的神经元，因为基于任务的神经元能够从与任务相关的数据中捕获有用的先验知识。为了促进基于任务的神经元在科学研究和工业应用中的使用，我们推出了TNLearn——一个开源的Python软件包，它提供了基于任务的神经元和网络的自动化构建功能，使基于任务的网络能够顺利训练。完整的文档（包括技术阐述、API参考和代表性示例）可在线获取。TNLearn已在 https://github.com/NewT123-WM/tnlearn 开源，并已成为PyTorch生态系统项目。

    arXiv:2609.27564v1 Announce Type: cross  Abstract: The brain does not rely on a single type of neuron to perform all kinds of tasks; instead, it designs different neurons for different tasks. The concept of task-based neurons represents a paradigm shift compared to task-based architectures. It argues that solving a specific problem requires customized neurons, as task-based neurons capture useful prior knowledge from task-related data. To facilitate the use of task-based neurons in scientific research and industrial applications, we introduce TNLearn, an open-source Python package that provides automated construction of task-based neurons and networks, enabling smooth training of task-based networks. Comprehensive documentation, including technical exposition, API reference, and representative examples, is available online. TNLearn is open-sourced at https://github.com/NewT123-WM/tnlearn and has become a PyTorch ecosystem project.
    
[^85]: PhyMo：面向多模态AI4Physics的物理场模态

    PhyMo: A Physical-Field Modality for Multimodal AI4Physics

    [https://arxiv.org/abs/2609.27554](https://arxiv.org/abs/2609.27554)

    该论文提出PhyMo框架，创新性地引入“物理场模态”这一全新模态，通过PDE关联算子组织异构物理测量数据，并采用三阶段学习流程（PDE残差监督预训练、与视觉嵌入对齐、多模态融合）来提升物理系统预测能力。

    

    多模态学习正成为AI for Physics（AI4Physics）的强大范式，其中预测物理系统需要对异构观测、测量数据和领域知识的联合解释。然而，现有方法通常将物理量和控制方程表示为通用的数值或文本标记，忽视了决定其时空相互作用的物理约束。为解决这一局限，我们引入了物理场模态，并提出PhyMo——一个以物理为基础的多模态框架，通过PDE关联算子来组织异构测量数据。PhyMo遵循三阶段学习流程：首先在PDE残差监督下通过场重构对物理场编码器进行预训练，随后将其表示与视觉嵌入在共享潜在空间中进行对齐，最后融合的多模态表示…（摘要截断）

    arXiv:2609.27554v1 Announce Type: cross  Abstract: Multimodal learning is emerging as a powerful paradigm for AI for Physics (AI4Physics), where predicting physical systems requires the joint interpretation of heterogeneous observations, measurements, and domain knowledge. However, existing approaches typically represent physical quantities and governing equations as generic numerical or textual tokens, overlooking the physical constraints that determine their spatiotemporal interactions. To address this limitation, we introduce the \textbf{physical-field modality} and propose \textbf{PhyMo}, a physics-grounded multimodal framework that organizes heterogeneous measurements through PDE-associated operators. PhyMo follows a three-stage learning procedure: the physical-field encoder is first pretrained through field reconstruction under PDE residual supervision, its representations are subsequently aligned with visual embeddings in a shared latent space, and the fused multimodal represent
    
[^86]: Behaviora——机器人与智能体外部与内部行为的概念架构

    Behaviora - A Conceptual Architecture for External and Internal Behavior of Robots and Agents

    [https://arxiv.org/abs/2609.27536](https://arxiv.org/abs/2609.27536)

    该论文提出了Behaviora概念架构，通过行为情节、IoB地址、风格配置、经验配置和行为编译器等组件，将机器人与智能体的外部和内部行为以可寻址的形式进行结构化表示。

    

    Behaviora是一个初步的概念架构，用于以可寻址的形式表示智能体和机器人的行为，包括外部行为和内部行为。一个正在执行行为的机器人或智能体会执行一个“行为情节”，该情节由情节组件组成，这些组件可源自行为分类学并被分配持久标识符。我们将这些标识符称为IoB（行为互联网）地址。行为情节指定系统做什么，而风格配置文件指定该行为如何表达。风格可以传达行为主体的特征以及能力、文化礼仪等品质。经验配置文件（EP）表示与行为相关的内部状态，该状态调节情节的执行。最后，行为编译器将这些行为表示映射为特定平台的动作。我们使用一个简单的触摸手臂模型来展示这些组件及其关系。外部行为是一种结果

    arXiv:2609.27536v1 Announce Type: cross  Abstract: Behaviora is a preliminary conceptual architecture for representing agent and robot behavior, external and internal alike, in an addressable form. A behaving robot or agent performs a Behavior Episode composed of episode components, which can be derived from behavior taxonomies (BTax) and assigned persistent identifiers. We denote these identifiers as IoB (Internet of Behaviors) Addresses. A Behavior Episode specifies what the system does, while a Style Profile (SP) specifies how this behavior is expressed. Style can communicate characteristics of the actor and qualities such as competence and cultural manners. An Experience Profile (EP) represents behaviorally relevant internal state that modulates the execution of an Episode. Finally, a Behavior Compiler maps these behavioral representations to platform-specific actions. We use a primitive touching arm model to show these components and their relations. External Behavior is a result 
    
[^87]: 非你所想：大语言模型能否遵循指定的否定语义？

    Not What You Meant: Can LLMs Follow a Specified Negation Semantics?

    [https://arxiv.org/abs/2609.27517](https://arxiv.org/abs/2609.27517)

    该论文提出NAFBench基准，通过生成经求解器验证的正规逻辑程序实例，系统评估大语言模型在SLDNF、良基语义及稳定模型语义（轻信/怀疑推理）等不同否定语义下的默认解读方式，以及能否在明确指定语义时覆盖默认偏好。

    

    否定在不同领域中并不具有统一的解释。在法律、监管和医学推理中，预期的解释取决于所采用的解读方式——开放世界与封闭世界、二值与三值逻辑，以及轻信式与怀疑式推理。我们研究了大语言模型默认采用哪种否定解读方式，以及当明确指定另一种解读方式时，它们能否覆盖这一默认偏好。为此，我们引入了NAFBench，这是一个程序化生成器，可生成经过求解器验证的实例，涵盖四种语义视角：SLDNF、良基语义（WFS），以及稳定模型语义下的轻信式和怀疑式推理。该生成器生成具有可控深度、宽度和循环结构的基态正规逻辑程序。每个程序在四种视角下分别使用SWI-Prolog、良基语义求解器和clingo进行求解，最多可产生四个不同的标签。随后这些程序被转化为自然语言描述。

    arXiv:2609.27517v1 Announce Type: new  Abstract: Negation does not carry a uniform interpretation across domains. In legal, regulatory, and medical reasoning, the intended interpretation depends on the reading in force -- open- versus closed-world, two- versus three-valued, and credulous versus skeptical. We study which reading of negation large language models adopt by default and whether they can override that preference when a different reading is explicitly specified. To this end, we introduce NAFBench, a procedural generator of solver-certified instances spanning four semantic viewpoints: SLDNF, well-founded semantics (WFS), and credulous and skeptical reasoning under stable-model semantics. The generator emits ground normal logic programs with controlled depth, width, and cycle structure. Each program is solved under all four viewpoints using SWI-Prolog, a well-founded semantics solver, and clingo, yielding up to four divergent labels. The programs are then verbalized into natura
    
[^88]: NV-Reason-CT：用于CT分析的三维视觉语言模型

    NV-Reason-CT: 3D Visual Language Model for CT Analysis

    [https://arxiv.org/abs/2609.27511](https://arxiv.org/abs/2609.27511)

    NV-Reason-CT通过原生3D视觉Transformer将全部视觉标记及其显式3D坐标直接传入语言模型解码，在基于7万余例CT、约55万条专家标注引导的多模态指令数据上训练，实现了保留完整体积空间信息的胸部和腹部CT智能推理分析。

    

    我们提出了NV-Reason-CT，这是一个用于胸部和腹部CT分析的生成式视觉-语言模型，它将原生三维视觉编码与放射科医师引导的推理相结合。该模型将原生3D视觉Transformer与语言模型耦合，将所有视觉标记及其显式3D坐标直接传递给语言解码过程，无需进一步的空间标记合并。这使得体积空间信息在视觉编码器内部得以保留，并通过语言模型的位置编码在与文本联合处理时得以维持。我们在一个精选的语料库上进行训练，该语料库包含来自70,111个独特CT图像输入的约550,000个多模态指令样本，结合了标准化报告、以异常为重点和特定解剖部位的问题、多轮交互，以及来自专家CT解读录音和转录的由放射科医师撰写的推理。专家标注提供了直接监督，并指导了额外的基于报告的合成推理。（注：原文摘要不完整，在"End-to-en"处截断）

    arXiv:2609.27511v1 Announce Type: cross  Abstract: We present NV-Reason-CT, a generative vision--language model for chest and abdominal CT combining native 3D visual encoding with radiologist-guided reasoning. The model couples a native 3D vision transformer with a language model, passing all visual tokens and their explicit 3D coordinates into language decoding without further spatial token merging. This retains volumetric spatial information within the vision encoder and through the language model's positional encoding during joint processing with text.   We train on a curated corpus of approximately 550,000 multimodal instruction examples from 70,111 unique CT image inputs, combining standardized reports, abnormality-focused and anatomy-specific questions, multi-turn interactions, and radiologist-authored reasoning from recorded and transcribed expert CT interpretations. Expert annotations provide direct supervision and guide additional report-grounded synthetic reasoning. End-to-en
    
[^89]: 不可作弊的评估：基于动态压缩的语言模型评估方法

    Uncheatable Eval: Dynamic Compression-Based Evaluation of Language Models

    [https://arxiv.org/abs/2609.27510](https://arxiv.org/abs/2609.27510)

    提出Uncheatable Eval动态基准，利用定期收集的新发布文本和压缩率指标评估基础语言模型，有效降低基准数据污染带来的作弊风险。

    

    现代大型语言模型在海量数据集上进行预训练，这使得很难防止基准测试数据进入其训练集，从而损害评估结果的可靠性。对于基础模型而言，可靠的评估尤其具有挑战性，因为其有限的指令遵循能力使基于任务的评估变得复杂。我们提出了Uncheatable Eval，这是一个动态基准，它定期收集新发布的文本以评估基础语言模型，并降低数据污染的风险。借助模型预测能力与其无损压缩数据能力之间的关系，我们使用压缩率来评估模型对新文本的预测能力。我们评估了涵盖14个文本类别的80个模型，研究了压缩性能如何随上下文长度变化，并检验了压缩率与零样本MMLU准确率之间的相关性。我们的结果得出了三个主要发现：(1) 压缩性能遵循

    arXiv:2609.27510v1 Announce Type: cross  Abstract: Modern large language models are pretrained on massive datasets, making it difficult to prevent benchmark data from entering their training sets and undermining the reliability of evaluation results. Reliable evaluation is particularly challenging for base models, whose limited instruction-following ability complicates task-based assessment. We introduce Uncheatable Eval, a dynamic benchmark that regularly collects newly published text to evaluate base language models and reduce the risk of data contamination. Drawing on the relationship between a model's predictive ability and its ability to compress data losslessly, we use compression rate to evaluate how well models predict new text. We evaluate 80 models across 14 text categories, study how compression changes with context length, and examine the correlation between compression rate and zero-shot MMLU accuracy. Our results yield three main findings: (1) compression performance foll
    
[^90]: WhatWorkedBench：AI智能体实验理解能力基准测试

    WhatWorkedBench: Benchmarking Experimental Understanding in AI Agents

    [https://arxiv.org/abs/2609.27490](https://arxiv.org/abs/2609.27490)

    该论文提出了WhatWorkedBench基准，用于评估AI研究智能体的实验理解能力，即智能体在预算受限实验后预测组件变化如何影响实验结果的准确性。

    

    AI研究智能体需要可靠地了解它们的实验如何改变结果。我们提出了WhatWorkedBench来衡量实验理解能力，即在预算受限的实验之后，智能体对组件变化预测的准确性。智能体检查代码、选择测量方式，并提交一个响应面——一张预测每种组件设置配置得分的表格。穷举式CPU执行为在保持其他组件不变的情况下更改每个组件提供了参考效应。这些效应捕获了来自30个数据源和8种工作流类型的36个任务中的变化组合，共包含1248条配置记录。核心评估结合了覆盖所有八个系列的4,206条数值控制记录，以及原始六个系列中的108个智能体回合。在八项新测量中，成对效应岭回归在22个数据源中的15个上选出了最优配置，并在三个数据源上将所有效应误差控制在得分范围的10%以内。将高斯过程（GP）拟合到……

    arXiv:2609.27490v1 Announce Type: new  Abstract: AI research agents need reliable knowledge of how their experiments change outcomes. We introduce WhatWorkedBench to measure experimental understanding, the accuracy of predictions about component changes after budgeted experimentation. Agents inspect code, select measurements, and submit a response surface, a table predicting scores for every configuration of component settings. Exhaustive CPU execution supplies reference effects for changing each component while holding the others fixed. These effects capture combinations of changes across 36 tasks from 30 data sources and 8 workflow types, with 1248 configuration records. Core evaluation combines 4,206 numerical-control records across all eight families and 108 agent episodes across the original six. At eight new measurements, pair-effect ridge selects an optimum on 15 of 22 sources and limits every effect error to 10% of score range on three. Fitting a Gaussian process (GP) to the sa
    
[^91]: 《Passing：借助AI生成声音穿越重构时空的无尽旅程》

    Passing: An Endless Journey through Reconstructed Spacetime with AI-Generated Sound

    [https://arxiv.org/abs/2609.27489](https://arxiv.org/abs/2609.27489)

    该论文提出交互式视听装置"Passing"，将单轨列车车窗录像重构为时空体并沿非线性轨迹重采样以生成无尽旅程，结合观者在场检测与实时视频转音频模型SpecMaskFoley生成同步声景，其中AI声音模型扮演"推测性聆听者"的角色而非客观配乐还原者。

    

    本文介绍了"Passing"，这是一个交互式视听装置，它将一段单轨列车车窗的连续录像重构为时空体，从中生成一段无尽的旅程。该作品并非线性回放素材，而是沿非线性轨迹对素材的空间与时间结构进行重采样，产生一幅不断掠过的景观，其深度、速度与时间顺序均变得不稳定。系统采用基于摄像机的观者在场检测机制，估计观者是否位于观看区域内，并利用这一在场状态影响渲染视频序列之间的过渡。生成的视频流被输入到SpecMaskFoley——一个实时视频转音频合成模型中，为重构后的图像生成同步声景。该模型并非用于还原客观正确的配乐，而是作为一个推测性的聆听者，提出一种可能的听觉诠释。

    arXiv:2609.27489v1 Announce Type: cross  Abstract: This paper introduces Passing, an interactive audiovisual installation that generates an endless journey from a single continuous monorail-window recording by reconstructing it as a spatiotemporal volume. Rather than replaying the footage linearly, the work resamples its spatial and temporal structure along nonlinear trajectories, producing a continuously passing landscape whose depth, speed, and temporal order become unstable. A camera-based viewer-presence detection system estimates whether a viewer is present in the viewing zone and uses this presence state to influence transitions among rendered video sequences. The resulting video stream is fed into SpecMaskFoley, a real-time video-to-audio synthesis model that generates a synchronized soundscape for the reconfigured image. The model is not used to reconstruct an objectively correct soundtrack, but functions as a speculative listener, proposing a possible auditory interpretation o
    
[^92]: Kairos：基于4D场景图的存在性与方向性流动的有据预测

    Kairos: Grounded Forecasting of Presence and Directional Flow in 4D Scene Graphs

    [https://arxiv.org/abs/2609.27467](https://arxiv.org/abs/2609.27467)

    Kairos将层次化3D场景图扩展为4D场景图，通过预测性方向流记忆，可对任意未来时刻的人员存在概率及其运动的完整方向分布进行预测，并给出随观测积累而收窄的校准置信区间。

    

    在有人类活动的环境中实现长期自主运行，需要机器人能够预判在尚未观测到的未来时刻，人们是否移动以及将如何移动。现有的行人运动表征面临权衡：要么预测未来活动，但将每个位置简化为一个标量速率；要么对完整的方向分布建模，但使其在时间上固定不变。我们提出了Kairos，一种预测性方向流记忆，将层次化的3D场景图（3DSG）扩展为4D场景图（4DSG）。重建几何中的每个被观测体素都存储一个方向混合分布和存在率，谱预测器可以针对任意未来查询时刻，同时预测人员存在的概率及其运动的完整方向分布。相邻体素之间的成对流动依赖关系支持条件查询，且每个体素的预测方差可产生经校准的置信区间，该区间随观测数据的积累而不断收窄。我们……

    arXiv:2609.27467v1 Announce Type: cross  Abstract: Long-term autonomy in human-populated environments requires anticipating whether and how people will move at times a robot has not yet observed. Existing representations of pedestrian motion face a tradeoff: they either forecast future activity, reducing each location to a scalar rate, or model the full directional distribution, holding it fixed in time. We present Kairos, a predictive directional-flow memory that extends a hierarchical 3D scene graph (3DSG) to a 4D scene graph (4DSG). Every observed voxel of the reconstructed geometry stores a directional mixture and a presence rate, and spectral predictors forecast, for any future query time, both the probability that people are present and the full directional distribution of their motion. Pairwise flow dependence between adjacent voxels supports conditional queries, and per-voxel predictive variances yield calibrated credible intervals that tighten as observations accumulate. We ev
    
[^93]: 发卡行主权智能体支付

    Issuer-Sovereign Agentic Payments

    [https://arxiv.org/abs/2609.27452](https://arxiv.org/abs/2609.27452)

    本文提出“发卡行主权智能体支付”方法，由发卡行自身的认证组件记录持卡人批准的消费规则，并在智能体支付时核验商户、仅在合规时生成卡片认证值，从而将AI智能体支付的控制权保留在承担风险的发卡行手中，且执行时不引入额外依赖。

    

    AI智能体已开始进行真实支付。当前的方法让智能体通过依赖一个凭证提供方来完成支付，而在如今已部署的方案中，该提供方通常位于持卡人银行之外。消费规则因此由卡组织或该提供方来执行，而非由银行自身执行。这使得承担金融风险的发卡行在支付发生的时刻几乎没有直接控制权。本文提出了“发卡行主权智能体支付”，这是一种将控制权保留在发卡行手中的方法。持卡人只需批准一次消费规则，由银行自身的认证组件记录该规则。此后，当智能体向特定商户付款时，银行会依据已批准的规则核验该商户，仅当商户获得许可时才生成卡片认证值。随后支付沿正常的卡组织通道传输并由发卡行验证，执行过程中不会引入任何额外的依赖。

    arXiv:2609.27452v1 Announce Type: cross  Abstract: AI agents are beginning to make real payments. Current approaches let an agent pay by relying on a credential provider that, in the approaches deployed today, typically sits outside the cardholder's bank. The spending rules are then enforced by the card network or that provider, and not by the bank itself. This leaves the issuing bank, which carries the financial risk, with little direct control at the moment a payment happens. This paper describes Issuer-Sovereign Agentic Payments, a method that keeps that control with the issuer. The cardholder approves a spending rule once, and the bank's own authentication component records it. Later, when the agent pays a specific merchant, the bank checks the merchant against the approved rule and generates the card authentication value only if the merchant is allowed. The payment then travels the normal card rails and is validated by the issuer, with no extra dependency introduced at execution.
    
[^94]: BEE：基于视觉-语言-动作模型的干预自适应真实世界强化学习

    BEE: Intervention-Adaptive Real-World Reinforcement Learning with Vision-Language-Action Models

    [https://arxiv.org/abs/2609.27450](https://arxiv.org/abs/2609.27450)

    BEE提出一种干预自适应的真实世界强化学习框架，将人类纠正建模为关于约束的证据而非需要模仿的动作，通过纠正模型在冻结的VLA上优化精度关键动作，使策略超越专家模仿。

    

    视觉-语言-动作模型能够处理长时程操作任务，然而其成功取决于少数几个精度关键的阶段——在这些阶段中，毫米级的误差会使之前所有的进展付诸东流。在线强化学习（RL）恰好可以优化这些动作，但在真实机器人上进行自由探索的代价过于高昂，这使得人类纠正变得不可或缺。然而，现有的面向VLA的在线RL方法要么无法纳入此类纠正，要么将其并入无差别的监督信号中。事实上，人类纠正并非均匀的噪声，而是在某些动作维度上可靠、在另一些维度上多变的。基于这一观察，我们提出了BEE，一个在冻结VLA上进行真实世界强化学习的干预自适应框架，使策略能够超越专家模仿。我们将人类纠正不是表述为需要复现的动作，而是作为关于某个约束的证据：一个纠正模型预测人类会如何纠正给定的VLA提议，以及该纠正的一致性如何……（原文摘要在此处截断）

    arXiv:2609.27450v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models handle long-horizon manipulation, yet success hinges on a few precision-critical phases where millimeter-scale errors undo all prior progress. Online reinforcement learning (RL) can optimize exactly these actions, but free exploration is far too costly on real robots, which makes human corrections indispensable. However, existing online RL methods for VLAs either cannot incorporate such corrections or fold them into undifferentiated supervision. Yet human corrections are not uniformly noisy but reliable along some action dimensions and variable along others. Building on this, we introduce BEE, an intervention-adaptive framework for real-world RL on a frozen VLA that lets the policy go BEyond Expert imitation. We formulate human corrections not as actions to reproduce but as evidence about a constraint: a Correction Model predicts how a human would correct a given VLA proposal and how consistent the c
    
[^95]: 面向量子云编排中成本与延迟权衡的量子强化学习

    Quantum Reinforcement Learning for Cost and Delay Tradeoffs in Quantum Cloud Orchestration

    [https://arxiv.org/abs/2609.27446](https://arxiv.org/abs/2609.27446)

    该论文提出QRLQ框架，将参数化量子电路与D3QN相结合用于量子云任务调度，能够动态权衡成本与延迟，相比启发式基线平均成本降低5-11%。

    

    量子云计算通过量子即服务（QaaS）模式提供对量子计算资源的访问。然而，对本质上异构的量子资源采用统一的基于时间的定价方式，极大地增加了任务编排的复杂性，尤其是在处理执行成本与系统性能之间的权衡时。启发式方法依赖于预定义的调度规则，而经典深度强化学习（DRL）模型在此场景下可能需要更多的可训练参数。受参数化量子电路（PQC）作为紧凑函数逼近器的潜力启发，我们提出了QRLQ，这是一个成本-延迟感知的量子云调度框架，将PQC与决斗双深度Q网络（D3QN）相结合，以动态地同时兼顾成本和延迟。仿真结果表明，QRLQ相比启发式基线方法实现了更低的平均成本和延迟，平均成本降低了5-11%。

    arXiv:2609.27446v1 Announce Type: cross  Abstract: Quantum cloud computing, delivered through the quantum-as-a-service (QaaS) model, provides access to quantum computing resources. However, applying uniform time-based pricing across fundamentally heterogeneous quantum resources significantly complicates task orchestration, particularly when addressing the tradeoff between execution costs and system performance. While heuristic methods rely on predefined scheduling rules, classical deep reinforcement learning (DRL) models may require more trainable parameters in this setting. Motivated by the potential of parameterised quantum circuits (PQCs) as compact function approximators, we propose QRLQ, a cost-delay-aware quantum cloud scheduling framework integrating PQCs with a dueling double deep Q-network (D3QN) to dynamically account for both cost and delay. Our simulation results show that QRLQ achieves lower mean cost and delay than the heuristic baselines, achieving a 5-11% lower mean cos
    
[^96]: Emergi-PersonaOS：一个面向情境适应与可控演化的人格代理操作系统

    Emergi-PersonaOS: A Persona Agent Operating System for Situational Adaptation and Controllable Evolution

    [https://arxiv.org/abs/2609.27417](https://arxiv.org/abs/2609.27417)

    该论文提出了 Emergi-PersonaOS，一个基于心理学的三层人格表示操作系统，能够根据对话情境自适应推断人格状态并生成回应，同时支持人格代理在长期交互中的可控演化。

    

    人类与数字存在之间的共生关系为人机交互的未来提供了一种愿景。在持久的人机关系中，人格为身份的连续性、交互中的个体性以及通过经验实现的发展提供了基础。我们通过作为计算实现形式的人格代理来研究这种能力，并提出了 Emergi-PersonaOS——一个基于心理学的操作系统，用于对人格对象进行全生命周期管理。该系统将倾向性特质、特征性适应与叙事身份组织为三层人格表示，区分了相对持久的人格信念与其在当前人格状态中的激活。在情境适应过程中，系统整合当前对话者、关系、事件与检索到的记忆来推断人格状态并生成行为和回复；在长期发展过程中，系统记录经验……

    arXiv:2609.27417v1 Announce Type: new  Abstract: Symbiosis between humans and digital beings offers a vision for the future of human--machine interaction. In enduring human--machine relationships, personality provides a foundation for continuity of identity, individuality in interaction, and development through experience. We investigate this capacity through persona agents as computational implementations and introduce Emergi-PersonaOS, a psychology-grounded operating system for managing persona objects throughout their lifecycle. The system organizes dispositional traits, characteristic adaptations, and narrative identity into a three-layer persona representation, distinguishing relatively enduring persona beliefs from their activation in the current persona state. During situational adaptation, it integrates the current interlocutor, relationship, event, and retrieved memories to infer a persona state and generate actions and replies; during long-term development, it records experie
    
[^97]: 视觉语言模型中看似能力限制的其实是读出限制

    What Looks Like a Capability Limit in Vision-Language Models Is a Readout Limit

    [https://arxiv.org/abs/2609.27408](https://arxiv.org/abs/2609.27408)

    视觉语言模型基准测试中看似的能力上限可能只是答案读出格式（如英语名称对比像素坐标）造成的读出限制——同一模型在相同任务上因答案约定不同表现可相差近50个百分点，且会改变模型间的排名。

    

    视觉语言模型的基准测试以某种约定形式提供答案选项：一个字母、一个颜色名称、一个像素坐标。这种约定通常被视为中性的。我们发现它并非中性，基准测试所报告的模型限制可能属于读出方式而非模型本身。在200张COCO照片上，当九个位置以英语名称给出时，Qwen3-VL-4B为指定物体选出正确位置的比例为68.5%；而当相同位置以像素坐标给出时，比例仅为20.0%（随机水平为11.1%）。这一代价仅出现在答案选项以坐标形式呈现时；若在问题中向模型提供一个坐标，仅损失3.5个百分点且不显著。该差距在4x4网格上、在8位而非4位量化下，以及按物体大小、边界距离和类别划分的每个数据切片中均成立。它甚至决定了哪个模型获胜：两个在英语名称形式下打平的模型，在一种坐标系中相差39个百分点，在另一种坐标系中相差54个百分点。

    arXiv:2609.27408v1 Announce Type: cross  Abstract: Benchmarks for vision-language models offer their answer choices in some convention: a letter, a color name, a pixel coordinate. That convention is treated as neutral. We find it is not, and that the limits a benchmark reports can belong to the readout rather than to the model.   On 200 COCO photographs, Qwen3-VL-4B picks the correct one of nine locations for a named object 68.5% of the time when the locations are given in English and 20.0% when the same locations are given as pixel coordinates. Chance is 11.1%. The cost arises when the answer options are coordinates; giving the model a coordinate in the question instead costs 3.5 points and is not significant. The gap holds on a 4x4 grid, under 8-bit rather than 4-bit quantization, and in every slice by object size, boundary distance and category. It also decides which model wins. Two models that tie under English names differ by 39 points in one coordinate system and by 54 in the oth
    
[^98]: 忘记你要遗忘之人：说话人去学习以防止零样本文本转语音中的重新识别

    Forget who you Forgot: Speaker Unlearning to Prevent Re-Identification in Zero-Shot Text-to-Speech

    [https://arxiv.org/abs/2609.27399](https://arxiv.org/abs/2609.27399)

    提出轻量级说话人去学习框架 GUARD，通过说话人门控与激活引导在冻结的 TTS 模型上抹除特定说话人身份，在防止声音被重新识别的同时保持语音的自然度与可懂度。

    

    近期的零样本文本转语音（ZS-TTS）系统仅凭几秒钟的参考语音就能高保真地复现说话人的声音，这引发了对未经授权的声音克隆与语音冒充的担忧。说话人身份去学习近来成为一种新方法，可选择性地抑制已选择退出的说话人的这一能力，同时保留对其他说话人的合成能力。尽管现有方法能够降低说话人相似度，但防止重新识别往往伴随着语音质量的严重下降。基于这一观察，我们提出了 GUARD，一个轻量级的说话人身份去学习框架，它在冻结的 TTS 主干上结合了学习到的说话人门控与说话人无关的激活引导。引导向量通过组相对奖励优化进行训练，将遗忘说话人的输出推向群体层面的冒充者相似度，同时保持语音的可懂度与自然度。

    arXiv:2609.27399v1 Announce Type: cross  Abstract: Recent zero-shot text-to-speech (ZS-TTS) systems can reproduce a speaker's voice with high fidelity from only a few seconds of reference speech, raising concerns over unauthorized voice cloning and impersonation. Speaker identity unlearning has recently emerged as an approach to selectively suppress this capability for speakers who opt out while preserving synthesis capability for other speakers. Although existing approaches reduce speaker similarity, preventing re-identification often faces severe degradation of speech quality. Motivated by this observation, we propose GUARD, a lightweight speaker identity unlearning framework that combines a learned speaker gate with speaker-agnostic activation steering on a frozen TTS backbone. The steering vectors are optimized using group-relative reward optimization to shift outputs from forget speakers toward population-level impostor similarity while preserving intelligibility and speech natura
    
[^99]: 基于空间门控特征相关表示的车载毫米波旋转雷达位置识别

    Automotive mmWave Spinning Radar Place Recognition with Spatially Gated Feature-Correlation Representation

    [https://arxiv.org/abs/2609.27394](https://arxiv.org/abs/2609.27394)

    提出SGCA-Net框架，通过空间门控相关聚合（SGCA）学习空间权重以抑制不稳定模糊的雷达区域影响，并聚合局部响应间的成对相关性，从而实现旋转鲁棒的车载毫米波旋转雷达位置识别。

    

    车载旋转FMCW（调频连续波）雷达能够提供稠密的360度感知，并且在光照不良和恶劣天气条件下依然保持可靠，因此非常适合自主导航。位置识别利用这些观测数据来识别之前到访过的地点，以实现重定位和长期导航。然而，航向变化在极坐标雷达表示中表现为循环移位，而传统的全局聚合方法可能会丢失雷达响应之间对区分相似地点至关重要的关系。我们提出了SGCA-Net，这是一个旋转雷达位置识别框架，它将旋转鲁棒的特征提取与空间门控相关聚合（SGCA）相结合。SGCA通过学习空间权重来降低不稳定且模糊的雷达区域的影响，同时聚合局部响应之间的成对相关性，以保留有信息价值的特征关系。在MulRan数据集上的实验表明，SGCA

    arXiv:2609.27394v1 Announce Type: cross  Abstract: Automotive spinning FMCW radar provides dense, $360^\circ$ sensing and remains reliable under poor illumination and adverse weather, making it well-suited to autonomous navigation. Place recognition uses these observations to identify previously visited locations for re-localization and long-term navigation. However, heading changes appear as circular shifts in the polar radar representation, and conventional global aggregation can lose relationships among radar responses that are important for distinguishing similar places. We propose SGCA-Net, a spinning radar place recognition framework that combines rotation-robust feature extraction with Spatially Gated Correlation Aggregation (SGCA). SGCA learns spatial weights to reduce the influence of unstable and ambiguous radar regions, while aggregating pairwise correlations among local responses to preserve informative feature relationships. Experiments on the MulRan dataset show that SGCA
    
[^100]: 预测工作流基准：利用预算约束的预测工具评估语言模型决策

    Forecast Workflow Bench: Evaluating Language-Model Decisions with Budgeted Forecast Tools

    [https://arxiv.org/abs/2609.27385](https://arxiv.org/abs/2609.27385)

    FWBench 提出了一个通过预算约束下的时间序列预测工具使用来评估语言模型决策能力的基准，发现 GPT-6 Astra 仅用 2.5% 的预算有选择地购买短时程预测即可胜过固定策略，首次实现了对决策质量与预测成本权衡的可复现评估。

    

    时间序列基础模型（TSFM）为运营决策提供预测，但仅凭准确性并不能决定其价值。评估使用这些模型的智能体需要同时衡量决策质量与预测成本。FWBench 在 1,251 个电力和公共自行车租赁案例上，使用固定的预测工具和模拟的容量合同来评估这一能力。智能体需要选择模型、历史数据长度和预测时程，然后提交容量方案以最小化给定的损失-成本目标。我们评估了两个托管配置和八个本地配置（包括小型语言模型），并对本地模型分别在有 TSFM 和无 TSFM 的情况下进行了测试。结果显示，GPT-6 Astra 有选择地购买了廉价的短时程预测，仅使用了 2.5% 的预算，并且在用三种损失-成本权重对所保存的决策进行评分时，其表现优于固定策略。FWBench 使得对语言模型如何在成本约束下选择和使用时间序列预测进行决策的可复现评估成为可能。

    arXiv:2609.27385v1 Announce Type: cross  Abstract: Time-series foundation models (TSFMs) provide forecasts for operational decisions, but accuracy alone does not determine their value. Evaluating agents that use these models requires measuring decision quality and forecast cost. FWBench evaluates this capability on 1,251 electricity and cycle-hire cases using fixed forecast tools and simulated capacity contracts. Agents select models, histories and horizons, then submit capacities to minimize a stated loss-cost objective. We evaluated two hosted and eight local configurations, including small language models, and tested local models with and without TSFMs. GPT-6 Astra bought inexpensive short-horizon forecasts selectively, using 2.5% of the budget, and outperformed fixed policies when the saved decisions were scored with three loss-cost weightings. FWBench enables reproducible evaluation of how language models select and use time-series forecasts to make decisions under cost constraint
    
[^101]: 面向全双工语音到语音对话模型对抗鲁棒性的心理声学对齐潜在平滑

    Psychoacoustically Aligned Latent Smoothing for Adversarial Robustness of Full-Duplex Speech-to-Speech Dialogue Models

    [https://arxiv.org/abs/2609.27378](https://arxiv.org/abs/2609.27378)

    该论文首次将全双工语音对话模型的不可感知对抗攻击形式化为心理声学掩蔽阈值约束下的扰动优化，并提出PALS防御方法，通过在残差向量量化潜在接口注入由码本协方差和掩蔽阈值塑造的噪声，在不增加任何推理时开销的情况下将各类攻击成功率从最高91.7%大幅降至约8%-11%。

    

    端到端语音到语音对话模型同时进行聆听和说话，因此其持续开放的声学通道容易受到对抗性操纵。我们将针对全双工智能体的不可感知攻击形式化为受载体语音心理声学掩蔽阈值约束的加性扰动优化，涵盖三种攻击目标：定向语义劫持、响应抑制和策略越狱。面对未加防御的Moshi式智能体，白盒攻击的成功率最高可达91.7%。随后我们提出心理声学对齐潜在平滑（PALS）方法，该方法在残差向量量化潜在接口处注入由局部码本协方差塑造的各向异性高斯噪声，同时利用由掩蔽阈值塑造的输入噪声约束攻击者，并通过Kullback-Leibler一致性目标进行训练。PALS在部署时无需任何推理时开销，即可将劫持攻击成功率降至8.3%，静音攻击降至11.2%，越狱攻击（摘要内容在此处截断）。

    arXiv:2609.27378v1 Announce Type: cross  Abstract: End-to-end speech-to-speech dialogue models listen and speak simultaneously, so a continuously open acoustic channel is exposed to adversarial manipulation. We formalize imperceptible attacks on full-duplex agents as optimization over additive perturbations confined beneath the psychoacoustic masking threshold of the carrier speech, under three goals: targeted semantic hijacking, response suppression, and policy jailbreaking. Against an undefended Moshi-style agent, white-box attacks succeed in up to 91.7% of trials. We then introduce psychoacoustically aligned latent smoothing (PALS), which injects anisotropic Gaussian noise shaped by local codebook covariance at the residual-vector-quantized latent interface, with input noise shaped by the masking threshold constraining the attacker and trained by a Kullback--Leibler consistency objective. Deployed with no inference-time cost, PALS reduces hijack to 8.3%, mute to 11.2%, and jailbreak
    
[^102]: 基于协调推理路径的规划式测试时扩展

    Planned Test-Time Scaling with Coordinated Reasoning Paths

    [https://arxiv.org/abs/2609.27374](https://arxiv.org/abs/2609.27374)

    本文提出规划式测试时扩展（PTTS），用规划器生成差异化解题大纲、执行器据此作答的协调联合策略取代独立重复采样，从而提升推理路径覆盖度与pass@k扩展性能。

    

    通过并行分支进行测试时扩展已被广泛采用，以提升模型在具有挑战性的推理任务上的性能。主流方法——重复采样——从单一策略中独立抽取各分支，这可能产生冗余的尝试，从而限制了额外推理计算带来的收益。为解决这一局限，我们提出了规划式测试时扩展（Planned Test-Time Scaling, PTTS），它用一个协调的联合策略取代独立采样：规划器为每个分支生成解题大纲，引导各分支走向彼此不同的推理路径；执行器则以每个大纲为条件生成完整的解答。在形式上，我们证明PTTS严格泛化了重复采样，并且在一个简化设定下，可从理论上证明它能更好地覆盖互补的推理模式，并获得更优的pass@k扩展性。我们在强大的推理模型之上实例化了PTTS，将这些模型固定为执行器，同时用PTTS取代重复采样。

    arXiv:2609.27374v1 Announce Type: cross  Abstract: Test-time scaling with parallel branches is widely adopted to improve performance on challenging reasoning tasks. The predominant approach, repeated sampling, draws branches independently from a single policy, which can produce redundant attempts and thereby limit the gains from additional inference compute. To address this limitation, we propose Planned Test-Time Scaling (PTTS), which replaces independent sampling with a coordinated joint policy: a planner generates a solution outline for each branch, steering the branches toward distinct reasoning paths, and an executor produces a full solution conditioned on each outline. Formally, we show that PTTS strictly generalizes repeated sampling and, in a stylized setting, provably promotes coverage of complementary reasoning modes and yields better pass@k scaling. We instantiate PTTS on top of strong reasoning models, keeping them fixed as executors while replacing repeated sampling with P
    
[^103]: 沉默与重叠皆非失败：全双工口语对话模型中话轮转换的意图条件化评估

    Neither Silence nor Overlap Is Failure: Intent-Conditioned Evaluation of Turn-Taking in Full-Duplex Spoken Dialogue Models

    [https://arxiv.org/abs/2609.27372](https://arxiv.org/abs/2609.27372)

    该论文提出TACT基准与意图条件化的连续评分方法，论证沉默或重叠在话轮转换中是否失败取决于说话者意图，从而取代传统的二元固定窗口评估，并揭示现有全双工对话模型（最佳0.47）与人类水平（0.86）之间的显著差距。

    

    全双工口语对话模型的现有基准采用二元固定窗口规则对话轮转换进行评分，即根据前一话轮的完整性来奖励立即响应或保持沉默。我们认为，响应偏移是否恰当——无论是延迟的沉默还是预期性的重叠——取决于说话者的潜在意图，而该意图只能从说话者自身的行为中加以识别。我们提出了TACT基准，包含来自五个双人对话语料库的9,728个片段、总计73.2小时的数据；每个片段均包含对话历史、每个说话者的记忆档案，以及由标注者得出的六个意图类别上的后验分布。评分方法以严格适宜的阈值加权连续排序概率分数取代二元窗口，其权重由依据人类话轮转移偏移分布拟合的意图条件化时序核确定，并证明了该分数的有界性、一致性与二元归约性。在十一个系统中，最佳模型得分为0.47，而人类表现上限为0.86。

    arXiv:2609.27372v1 Announce Type: cross  Abstract: Benchmarks for full-duplex spoken dialogue models score turn-taking with binary fixed-window rules that reward immediate response or silence by completeness of the prior turn. We argue that the appropriateness of a response offset, whether delayed silence or anticipatory overlap, is conditional on the speaker's latent intent, identifiable only from that speaker's behavior. We introduce TACT, a benchmark of 9,728 episodes and 73.2 hours from five dyadic corpora; each episode carries dialogue history, a per-speaker memory profile, and an annotator-derived posterior over six intent classes. Scoring replaces binary windows with a strictly proper threshold-weighted continuous ranked probability score whose weights are intent-conditioned timing kernels fitted to human floor-transfer-offset distributions, proving boundedness, consistency, and binary reduction. Across eleven systems the best model reaches 0.47 against a human topline of 0.86, 
    
[^104]: 自然环境中的几何条件化视觉位置识别

    Geometry-Conditioned Visual Place Recognition in Natural Environments

    [https://arxiv.org/abs/2609.27370](https://arxiv.org/abs/2609.27370)

    该论文提出深度感知蒸馏（DAD）方法，将几何基础模型推断的深度信息以通道级条件化的方式注入预训练视觉基础模型的token表示中，无需深度传感器即可在植被重复、外观视角变化剧烈的自然环境中实现更鲁棒的视觉位置识别。

    

    自然环境中的视觉位置识别（VPR）仍然具有挑战性，原因在于重复的植被、稀少的独特地标，以及不同遍历之间巨大的外观和视角变化。尽管同一地点的视觉观测可能发生显著变化，但其底层的空间结构往往更加持久。我们通过深度感知蒸馏（Depth-Aware Distillation, DAD）利用这种互补的几何一致性，该方法将预训练视觉基础模型（VFM）的token表示条件化于由几何基础模型（GFM）推断出的几何信息，而无需任何深度传感器。DAD并未将几何作为额外的输入模态，而是将图像对齐的深度投影到VFM的token空间中，并通过通道级几何条件化选择性地调制视觉表示。两阶段的教师引导学习策略首先将几何条件化的表示锚定到预训练的外观空间中。

    arXiv:2609.27370v1 Announce Type: cross  Abstract: Visual Place Recognition (VPR) in natural environments remains challenging due to repetitive vegetation, sparse distinctive landmarks, and substantial appearance and viewpoint variation across traversals. While visual observations of the same place can change considerably, their underlying spatial structure is often more persistent. We exploit this complementary geometric consistency through Depth-Aware Distillation (DAD), which conditions the token representations of a pretrained Vision Foundation Model (VFM) on geometry inferred by a Geometric Foundation Model (GFM), without any depth sensor. Rather than treating geometry as an additional input modality, DAD projects image-aligned depth into the VFM token space and selectively modulates visual representations through channel-wise geometric conditioning. A two-stage teacher-guided learning strategy first anchors the geometry-conditioned representation to the pretrained appearance spac
    
[^105]: 基于保留-遗忘损失景观交互视角的量化鲁棒机器遗忘

    Quantization-Robust Unlearning through the Lens of Retain-Forget Loss Landscapes Interaction

    [https://arxiv.org/abs/2609.27355](https://arxiv.org/abs/2609.27355)

    本文提出一种量化鲁棒的机器遗忘框架，通过基于曲率的敏感权重判据和敏感度引导的噪声正则化，将模型收敛引导至更平滑的极小值，使遗忘效果在量化压缩后依然保持鲁棒，同时维持整体模型效用。

    

    机器遗忘通过移除私有或受版权保护训练数据的影响，确保大语言模型（LLM）的合规性。然而，由于大语言模型在实际部署中通常会经历训练后压缩（如量化），已有观察发现遗忘效果会被显著削弱，且遗忘行为的退化比模型效用的退化更为严重。本文提出了一种量化鲁棒的机器遗忘框架，使遗忘对量化具有鲁棒性，同时保持整体模型效用。我们通过损失景观的视角来分析这一差距。具体而言，我们的分析揭示了一种基于曲率的判据，能够精确定位已遗忘模型中导致非鲁棒遗忘和效用降低的敏感权重。因此，我们提出了敏感度引导的噪声正则化方法，将其应用于敏感参数上，引导模型收敛至具有一致较低遗忘损失的更平滑极小值。

    arXiv:2609.27355v1 Announce Type: cross  Abstract: Unlearning ensures LLM compliance by removing the influence of private or copyrighted training data. However, since LLM models typically undergo post-training compression, like quantization, in practical deployment, it has been observed that the unlearning effect can be substantially weakened, with the forgetting behavior degrading more severely than that of model utility. This paper proposes a quantization-robust unlearning framework that makes forgetting robust to quantization while maintaining overall model utility. We analyze this gap through the lens of loss landscape. Specifically, our analysis reveals a curvature-based criteria that pinpoints sensitive weights in the unlearned model that leads to both non-robust forgetting and reduced utility. We therefore propose sensitivity-guided noisy regularization, which is applied on the sensitive parameters to steer the model convergence towards a smoother minima of uniformly low forget 
    
[^106]: 约束驱动的上下文工程：为AI系统设计领域接口

    Constraint-Driven Context Engineering: Designing Domain Interfaces for AI Systems

    [https://arxiv.org/abs/2609.27354](https://arxiv.org/abs/2609.27354)

    本文提出“约束驱动的上下文工程”方法，主张通过系统性地识别并将AI系统运行环境中的技术、法规、制度和规范约束融入上下文设计，以提升已有通用AI解决方案的领域适配性和质量，弥补现有方法仅依赖检索、记忆和工具提供领域知识的不足。

    

    生成式AI系统正日益被部署用于解决领域问题。这些系统在技术、法规、制度和规范等多重约束下运行，这些约束定义了其所属领域内可接受的AI行为和结果。我们在行业合作中反复观察到一种模式：合作伙伴往往带着一个功能可用但相对通用的AI解决方案前来。此时的挑战不再是从头构建AI系统，而是提升AI生成解决方案的质量和领域适配性。在这些场景中，限制因素往往是系统可获取的上下文的质量、范围和结构。然而，现有的上下文工程方法主要侧重于通过检索、记忆和工具来提供领域知识，对于系统地识别和落实AI系统在其运行环境中所受约束的支持却十分有限。本文提出了约束（驱动的上下文工程方法……）

    arXiv:2609.27354v1 Announce Type: cross  Abstract: Generative AI systems are increasingly deployed to address domain problems. These systems operate under technical, regulatory, institutional, and normative constraints that define acceptable AI behaviour and outcomes within their domains. We observe a recurring pattern in our industry engagement: partners often arrive with a functioning but relatively generic AI solution. The challenge is no longer to build an AI system from scratch, but to improve the quality and domain appropriateness of an AI-generated solution. In these settings, the limiting factor is often the quality, scope, and structure of the context available to the system. Yet, existing context engineering approaches primarily focus on supplying domain knowledge through retrieval, memory, and tools, with limited support for systematically identifying and operationalising the constraints that govern AI systems in their operational environments.   This paper proposes Constrai
    
[^107]: MolDesignBench：评估基于大语言模型智能体的场景化分子设计

    MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design

    [https://arxiv.org/abs/2609.27349](https://arxiv.org/abs/2609.27349)

    提出了MolDesignBench——一个面向真实场景的分子设计基准，包含2000个融合隐式设计需求与显式约束的生成与优化任务并需要调用17种专业化学工具，实验表明当前前沿大语言模型智能体在这些真实分子设计任务上的成功率仍然很低。

    

    真实世界的分子设计对基于大语言模型（LLM）的智能体而言仍然充满挑战。它要求智能体理解设计背景、满足多重约束条件、识别不可行的规格要求，并对多步骤的工具输出进行推理。现有的基准测试未能捕捉这种复杂性，而是侧重于明确且狭窄的约束、仅包含可解决的问题以及单一路径的解决方案。为了填补这一空白，我们提出了MolDesignBench，这是一个基于真实场景的分子设计基准，用于评估工具增强的LLM智能体。MolDesignBench包含2K个生成与优化实例，这些实例将设计叙述中隐含的需求与显式的性质和官能团约束相结合（包括不可行的案例），并要求有效使用17种专业化学工具。在多种前沿LLM上的实验显示成功率较低——表现最佳的模型仅达到……

    arXiv:2609.27349v1 Announce Type: new  Abstract: Real-world molecular design remains challenging for large language model (LLM)-based agents. It requires them to interpret design contexts, satisfy multiple constraints, identify infeasible specifications, and reason over multi-step tool outputs. Existing benchmarks do not capture this complexity, focusing instead on explicit and narrow constraints, only feasible problems, and single-path solutions. To address this gap, we propose MolDesignBench, a scenario-grounded benchmark that more closely reflects real-world molecular design for evaluating tool-augmented LLM agents. MolDesignBench comprises 2K generation and optimization instances that combine implicit requirements embedded in design narratives with explicit property and functional-group constraints, including infeasible cases, and require the effective use of 17 specialized chemistry tools. Experiments across diverse frontier LLMs reveal low success rates--with the best achieving o
    
[^108]: 利用大语言模型演化可检查的O-RAN网络切片xApp

    Evolving Inspectable O-RAN Slicing xApps with LLMs

    [https://arxiv.org/abs/2609.27337](https://arxiv.org/abs/2609.27337)

    本文提出用大语言模型将O-RAN网络切片控制器自动演化为紧凑且可读、可编辑的Python程序，取代决策逻辑不可解释的深度强化学习神经网络策略，在保留自适应资源分配能力的同时，让运营商能够直接检查和修改控制逻辑，并在真实5G测试平台上验证了其有效性。

    

    开放无线接入网（O-RAN）切片xApp必须在满足服务等级协议（SLA）的同时，根据不断变化的信道条件和流量需求自适应地调整资源分配。深度强化学习虽然能够产生自适应策略，但其分配规则仍然隐藏在神经网络参数之中。本文的目标是在保留这种适应性的同时，使控制器的决策逻辑能够被运营商直接检查和编辑。研究者使用大语言模型（LLM）将切片控制器演化为紧凑的Python程序，其决策逻辑在优化之后依然保持可读和可编辑。LLM在离线状态下提出并迭代修改候选控制器，由经过校准的模拟器对其进行评分，最终选定的决策模块无需任何修改即可直接运行在O-RAN控制路径中。在NSF POWDER 5G测试平台上的实验表明，演化出的控制器能够在保证型切片的吞吐量目标因持续信道衰落而无法达成时，释放该切片的资源，从而改善尽力而为（best-effort）服务的性能。

    arXiv:2609.27337v1 Announce Type: cross  Abstract: Open RAN (O-RAN) slicing xApps must adapt resource allocations to changing channel conditions and traffic demands while meeting service-level agreements (SLAs). Deep reinforcement learning can produce adaptive policies, but their allocation rules remain encoded in neural-network parameters. Our goal is to retain this adaptability while making the controller's decision logic directly inspectable and editable by operators. We use a large language model (LLM) to evolve slicing controllers as compact Python programs whose decision logic remains readable and editable after optimization. The LLM proposes and revises candidates offline, while a calibrated simulator scores them, and the selected decision module runs unchanged in the O-RAN control path. On the NSF POWDER 5G testbed, the evolved controller releases resources from a guaranteed slice whose throughput target becomes unattainable under a sustained channel fade, improving best-effort
    
[^109]: CART：面向大语言模型的闭环自适应红队测试

    CART: Closed-Loop Adaptive Red Teaming for Large Language Models

    [https://arxiv.org/abs/2609.27336](https://arxiv.org/abs/2609.27336)

    CART是一个闭环自适应红队测试框架，通过利用每次测试结果动态指导后续探测并追踪新涌现的弱点，在模型和智能体评估场景中都比静态提示重放发现更多漏洞和更高风险。

    

    自动化红队测试通常重放固定的提示集合，这只能衡量已知风险，却无法从测试中发现的失败中学习。我们提出CART（闭环自适应红队测试），一个利用每次测试结果来指导下一步测试内容的框架。CART从广泛的风险覆盖开始，追踪测试中新涌现的弱点，保持新探测的多样性，并记录每项发现的证据和来源。它将创建测试的挑战者、被测试的目标（可以是纯文本模型或受控的工具使用智能体）以及评估结果的评判者分离开来，使这些角色可以被独立研究。在三个评估系列（Frontier、JAH和Agentic）中，对于每个有可用基线对比的目标，CART都比静态种子重放发现更多的失败和更高的平均风险。这些收益延伸到工具介导的智能体测试中，表明情境自适应能够揭示直接提示重放无法发现的弱点。

    arXiv:2609.27336v1 Announce Type: new  Abstract: Automated red teaming often replays a fixed set of prompts, which measures known risks but cannot learn from failures found during testing. We present CART (Closed-Loop Adaptive Red Teaming), a framework that uses each result to guide what it tests next. CART begins with broad risk coverage, follows weaknesses that emerge, keeps new probes diverse, and records the evidence and source of every finding. It separates the Challenger that creates tests, the Target being tested, which may be a text-only model or a bounded tool-using agent, and the Judge that evaluates the results, allowing these roles to be studied independently. Across three evaluation families (Frontier, JAH, and Agentic), CART discovers more failures and higher average risk than static seed replay for every Target with an available baseline. The gains extend to tool-mediated agent tests, suggesting that contextual adaptation can reveal weaknesses that direct prompt replay d
    
[^110]: 即时记忆：为LLM智能体学习策划任务自适应记忆

    Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents

    [https://arxiv.org/abs/2609.27334](https://arxiv.org/abs/2609.27334)

    本文提出“即时记忆”范式，不再在任务完成时将经验固化为静态记忆制品，而是保留原始轨迹、把记忆策展延迟到读取时根据当前任务动态合成，从而避免不可逆的信息丢失并化解写入时策展带来的长时程信用分配难题。

    

    智能体记忆系统通过复用过往经验来提升未来性能，然而现有的大多数设计在写入时就对记忆进行策展：一旦任务完成，其轨迹便被提炼成一个固定的产物（如反思、工作流、技能或推理策略），之后通过相似度进行检索。这迫使系统在尚不知道未来查询的情况下就决定什么值得记住，从而不可逆地丢弃信息，并产生与查询无关的摘要，却必须服务于众多可能的下游任务。学习这样的写入时策展器也很困难，因为一个存储决策的价值可能只有在相关查询到来时才显现——而这可能是很多个任务之后，由此造成长时程的信用分配问题。我们反其道而行：保留原始轨迹，将策展推迟到读取时进行，此时当前任务已经已知。给定检索到的轨迹和新任务，记忆策展器会合成一个紧凑的、任务……（原文摘要在此处截断）

    arXiv:2609.27334v1 Announce Type: new  Abstract: Agentic memory systems reuse past experience to improve future performance, yet most existing designs curate memory at write time: once a task is completed, its trajectory is distilled into a fixed artifact, such as a reflection, workflow, skill, or reasoning strategy, that is later retrieved by similarity. This forces the system to decide what is worth remembering before the future query is known, irreversibly discarding information and producing a query-independent summary that must serve many possible downstream tasks. Learning such a write-time curator is also difficult because the value of a storage decision may only become apparent when a relevant query arrives, potentially many tasks later, creating a long-horizon credit-assignment problem. We instead retain raw trajectories and defer curation until read time, when the current task is known. Given the retrieved traces and the new task, a memory curator synthesizes a compact, task-
    
[^111]: 对齐惯性：通过策略覆盖阻力审计训练数据影响的持久性

    Alignment Inertia: Auditing the Durability of Training Data Influence Through Policy Override Resistance

    [https://arxiv.org/abs/2609.27333](https://arxiv.org/abs/2609.27333)

    该论文提出“对齐惯性”和覆盖成功率两个新指标来审计平台干预（系统提示和微调）能否可靠覆盖模型先前训练形成的行为，发现微调有时会强化而非覆盖原有行为（如LoRA使Mistral的惯性提高46.5个百分点），并证明TRAK方法能有效预测这种惯性。

    

    平台运营者越来越依赖系统提示词和微调来管控模型行为，然而这些干预措施能否可靠地覆盖模型从先前训练中继承的行为，仍不清楚。我们提出了覆盖成功率和“对齐惯性”两个概念，用于衡量运营者的干预何时能成功或未能改变先前的行为。我们在医疗错误信息和仇恨言论两个领域，对Llama和Mistral两个模型评估了零样本提示和LoRA微调的效果。对齐惯性在两个模型中都持续存在，但会因模型、领域和政策方向的不同而变化。值得注意的是，在Mistral的限制性仇恨言论条件下，LoRA使惯性提高了46.5个百分点，这表明微调可能会强化而非覆盖先前的行为。我们还使用TRAK来检验惯性是否与较弱的适应信号相关。TRAK在8个条件中的7个达到了至少0.85的AUC，并优于模型置信度、TF-IDF相似度和嵌入相似度等基线方法。

    arXiv:2609.27333v1 Announce Type: new  Abstract: Platform operators increasingly rely on system prompts and fine-tuning to govern model behavior, yet it remains unclear how reliably these interventions override behavior inherited from prior training. We propose Override Success Rate (OSR) and alignment inertia to measure when operator interventions succeed or fail to change prior behavior. We evaluate zero-shot prompting and LoRA fine-tuning across Llama and Mistral in medical misinformation and hate speech. Alignment inertia persists across both models but varies by model, domain, and policy direction. Notably, in Mistral's restrictive hate-speech condition, LoRA increased inertia by 46.5 percentage points, showing that fine-tuning can reinforce rather than override prior behavior. We also use TRAK to test whether inertia is associated with weaker adaptation signals. TRAK achieves AUC of at least 0.85 in 7 of 8 conditions and outperforms model confidence, TF-IDF similarity, and embedd
    
[^112]: 几何稳定而任务证据发散：面向长程智能体的高效压缩

    Stable Geometry with Divergent Task Evidence for Efficient Long-Horizon Agent Compression

    [https://arxiv.org/abs/2609.27332](https://arxiv.org/abs/2609.27332)

    该论文发现智能体历史中全局几何相似并不代表任务证据得到保留，据此提出免训练的几何引导证据保留记忆压缩器 GEM，优先保护任务与执行证据、再以几何残差补全覆盖，在保持几何结构稳定的同时显著提升动作证据保留率（Top-3 从 0.31 升至 0.69）并将 token 消耗从 2.69M 降至 2.11M。

    

    长程智能体会不断累积交互历史，导致上下文和推理成本持续增加。我们发现，仅凭几何冗余并不足以作为安全压缩的判据。尽管智能体历史表现出很强的低维结构，但相似的全局几何却可能保留截然不同的任务证据量。在保留块数量相同的条件下，证据感知选择将下一动作的 Top-3 保留率从 0.31 提升至 0.69，而质心相似度仍保持在 0.98。受控替换实验进一步表明，动作相关信息可以被大幅改变，而全局几何度量却几乎保持不变。受几何与证据之间这一差距的启发，我们提出了几何引导的证据保留记忆（Geometry Guided Evidence Preserving Memory, GEM），这是一种免训练的压缩器，它在利用几何残差补全覆盖范围之前，优先保护任务与执行证据。GEM 将平均组合 token 用量从每次 2.69M 降低到 2.11M。

    arXiv:2609.27332v1 Announce Type: new  Abstract: Long horizon agents accumulate growing interaction histories that increase context and inference costs. We find that geometric redundancy alone is an insufficient criterion for safe compression. Although agent histories exhibit strong low dimensional structure, similar global geometry can preserve very different amounts of task evidence. At identical retained block counts, evidence aware selection raises next action Top 3 retention from 0.31 to 0.69, while centroid similarity remains 0.98. Controlled replacement further shows that action related information can be substantially altered while global geometric measures remain nearly unchanged. Motivated by this gap between geometry and evidence, we introduce Geometry Guided Evidence Preserving Memory (GEM), a training free compressor that protects task and execution evidence before using geometric residuals to complete coverage. GEM reduces mean combined token usage from 2.69M to 2.11M per
    
[^113]: 可验证的隐动力学博弈：从已求解机制生成智能体强化学习环境

    Verifiable Hidden Dynamics Play: Generating Agentic RL Environments from Solved Mechanisms

    [https://arxiv.org/abs/2609.27321](https://arxiv.org/abs/2609.27321)

    VHD-Play 颠覆了智能体环境的生成顺序——先采样并求解数学模型、再将求解结果渲染为有状态工具与可验证的评分参考，以每个环境几美分的低成本生成 3,300 个多样化环境，将 Qwen3.6-35B-A3B 的平均智能体得分从 0.204 提升至 0.815。

    

    语言模型智能体越来越多地面临状态持续演化、决策相互依赖且结果延迟显现的长时程任务。扩展其训练需要多样化的智能体环境、可靠的结果信号以及较低的扩展成本。现有的生成流程通常先构建环境，再定义其结果规则或标注其轨迹，导致动力学与评估只能事后对齐。VHD-Play 颠倒了这一依赖关系：先对数学模型进行采样与求解，再由基于语料的设定器将其决策过程呈现为有状态的工具。可执行的动力学规则和轨迹评分参考均继承自同一个已求解的模型。该流程以每个环境仅几美分的成本生成了 3,300 个多样化的智能体环境。在三个环境族上训练 Qwen3.6-35B-A3B 后，其在五族诊断中的平均智能体得分从 0.204 提升至 0.815，在来自全部三个环境族的留出实例上也观察到了提升。

    arXiv:2609.27321v1 Announce Type: new  Abstract: Language-model agents increasingly face long-horizon tasks with evolving state, interdependent decisions, and delayed outcomes. Scaling their training requires diverse agentic environments, dependable outcome signals, and low extension cost. Existing generation pipelines commonly construct an environment before defining its outcome rule or annotating its trajectories, leaving dynamics and evaluation to be aligned post hoc. VHD-Play reverses this dependency by sampling and solving a mathematical model before a corpus-grounded setter renders its decision process as stateful tools. The executable dynamics and trajectory-scoring reference are inherited from the same solved model. The pipeline produces 3,300 diverse agentic environments at a cost of a few cents each. Training Qwen3.6-35B-A3B on three families raises its mean agentic score from 0.204 to 0.815 in a five-family diagnostic. Gains also appear on held-out instances from all three t
    
[^114]: 打破天气-内容耦合：面向一体化红外图像恢复的类型-严重程度引导渐进解耦方法

    Breaking Weather-Content Coupling: Type-Severity Guided Progressive Disentanglement for All-in-One Infrared Restoration

    [https://arxiv.org/abs/2609.27317](https://arxiv.org/abs/2609.27317)

    提出TSGPD-IR网络，将恢复引导分解为任务级天气语义与区域级退化严重程度两个维度并进行渐进解耦，有效分离真实热结构与天气虚假响应，实现一体化红外图像恢复。

    

    红外（IR）成像对自动驾驶、遥感及其他感知任务至关重要。然而，恶劣天气可能引入与真实热结构相互纠缠的虚假结构响应。现有的红外恢复方法通常仅为单一退化类型设计，或直接从退化纠缠的表示中进行重建。因此，这些方法难以将固有热结构与天气引起的虚假响应区分开来，也难以适应空间上变化的退化严重程度，从而导致伪影或对微弱但有意义的热响应的过度抑制。为解决这些问题，我们提出了TSGPD-IR，一种面向一体化红外恢复的类型-严重程度引导渐进解耦网络，它将恢复引导分解为任务级天气语义和区域级退化严重程度。具体而言，一种天气与语义协同引导的多层级提示生成（摘要在此处被截断）

    arXiv:2609.27317v1 Announce Type: cross  Abstract: Infrared (IR) imaging is crucial for autonomous driving, remote sensing, and other perception tasks. However, adverse weather may introduce fake structural responses that are entangled with real thermal structures. Existing IR restoration methods are typically designed for a single degradation type or directly reconstruct from degradation-entangled representations. Consequently, they struggle to distinguish intrinsic thermal structures from weather-induced fake responses and to accommodate spatially varying degradation severity, leading to artifacts or the over-suppression of weak but meaningful thermal responses. To address these issues, we propose TSGPD-IR, a type-severity guided progressive disentanglement network for all-in-one infrared restoration that factorizes restoration guidance into task-level weather semantics and region-level degradation severity. Specifically, a Weather and Semantic Co-Guided Multi-Level Prompt Generation
    
[^115]: 将安全转化为能力：通过安全过滤强化学习实现最小可利用性的机器人策略

    Turning Safety into Competence: Minimally Exploitable Robot Policies via Safety-Filtered Reinforcement Learning

    [https://arxiv.org/abs/2609.27312](https://arxiv.org/abs/2609.27312)

    提出了S2C两阶段强化学习框架，通过对抗性强化学习训练鲁棒安全过滤器并将其与竞争任务学习分离，证明了安全过滤可保持策略的不可利用性，使机器人在竞争任务中胜率最高且最难被攻击利用。

    

    在竞争性任务中部署的机器人必须在保证安全的前提下智胜对手。现有方法（包括安全强化学习）通常训练单一策略来同时实现任务成功和避免失败，这种耦合会使训练复杂化，并使学到的策略容易被蓄意攻击所利用。我们提出了S2C（Safety to Competence），一个将安全综合与竞争性任务学习相分离的两阶段强化学习框架。我们将竞争性交互形式化为安全关键的马尔可夫博弈，并证明当所有参与者都遵循安全机动时，完美过滤能够保持策略的不可利用性。S2C通过对抗性强化学习学习一个鲁棒的安全过滤器，在任务策略训练期间将其嵌入环境中，并在部署时保留相同的过滤器。在模拟触地得分游戏的实验中，S2C优于八个安全强化学习基线方法，取得了最高的胜率和Elo评分，以及最低的可利用性。

    arXiv:2609.27312v1 Announce Type: cross  Abstract: Robots deployed for competitive tasks must outmaneuver their opponents without sacrificing safety. Existing approaches, including safe reinforcement learning (RL), train a single policy to achieve task success and avoid failures simultaneously. This coupling can complicate training and leave the learned policy exploitable by deliberate attacks. We propose Safety to Competence (S2C), a two-stage RL framework that separates safety synthesis from competitive task learning. We formulate competitive interactions as safety-critical Markov games and prove that perfect filtering preserves policy non-exploitability when all players commit to safe maneuvers. S2C learns a robust safety filter via adversarial RL, embeds it in the environment during task policy training, and retains the same filter at deployment. In simulated touchdown games, S2C outperforms eight safe RL baselines, achieving the highest win rate and Elo rating, and the lowest expl
    
[^116]: 面向加密C2检测的多视图融合：一项针对评估陷阱的泄露控制测量研究

    Multi-View Fusion for Encrypted C2 Detection: A Leakage-Controlled Measurement Study of Evaluation Pitfalls

    [https://arxiv.org/abs/2609.27311](https://arxiv.org/abs/2609.27311)

    该论文通过控制数据泄露的测量研究揭示，加密C2检测中多视图融合的收益可能被评估陷阱严重夸大——数据集级而非折内计算的频率编码即可虚增F1分数0.28（约为真实效应的十倍），且按目的地址分组后17,577条流仅对应2,132个独立样本组。

    

    摘要：命令与控制（C2）流量日益隐藏于TLS之中，因此防御者如今对流量元数据应用机器学习。许多研究假设，结合两种元数据视图——即流统计特征与TLS握手指纹——能够同时提升准确性和鲁棒性。我们在来自62次真实Cobalt Strike捕获的17,577条TLS流上检验了这一假设。我们的评估消除了导致报告分数过于乐观的数据泄露。我们报告了三个比融合结果本身更重要的发现。第一，一个错误的预处理步骤会使F1分数提高0.28。该步骤在整个数据集上而非在每个交叉验证折内计算频率编码，这一提升约为我们所测得的任何真实效应的十倍。第二，标签和行为特征均依赖于目的地址。因此，这17,577条流仅构成2,132个独立分组，并且 positi（原文在此截断）

    arXiv:2609.27311v1 Announce Type: cross  Abstract: Command-and-control (C2) traffic increasingly hides within TLS, so defenders now apply machine learning to traffic metadata. Many studies assume that combining two metadata views, namely flow statistics and TLS handshake fingerprints, improves both accuracy and robustness. We tested this assumption on 17,577 TLS flows from 62 real Cobalt Strike captures. Our evaluation removes the data leakage that leads to overly optimistic reported scores. We report three findings that matter more than the fusion result itself. First, an incorrect preprocessing step increases the F1 score by 0.28. This step computes the frequency encoding across the entire dataset rather than within each cross-validation fold. The increase is about ten times larger than any real effect we measured. Second, both the labels and the behavioral features depend on the destination address. Because of this, the 17,577 flows form only 2,132 independent groups, and the positi
    
[^117]: 从你自身的交互中学习如何行动：面向GUI智能体的在线策略自蒸馏

    Learn How to Act from Your Own Interactions: On-Policy Self-Distillation for GUI Agents

    [https://arxiv.org/abs/2609.27307](https://arxiv.org/abs/2609.27307)

    提出GUI-SD-v2，通过两阶段训练框架将在线策略自蒸馏从GUI定位扩展到多轮GUI交互，解决了自教师特权遵循能力有限和特权指导不足的问题。

    

    图形用户界面（GUI）智能体通过与软件环境的多轮交互来执行复杂的用户指令，这需要逐步推理来引导动作，并需要长程记忆来保留与任务相关的信息。近期的在线策略自蒸馏（OPSD）方法在GUI定位（GUI智能体的一项基础子任务）上取得了出色的性能，这得益于特权条件化自教师提供的密集token级监督。然而，将现有的OPSD方法扩展到多轮GUI智能体受到自教师有限的特权遵循能力和不充分的特权指导的阻碍。本文提出了GUI-SD-v2，即GUI-SD的下一个版本，它将OPSD从GUI定位扩展到多轮GUI交互，并通过两阶段训练框架解决了上述关键局限。具体而言，GUI-SD-v2首先通过联合优化来加强特权遵循能力……

    arXiv:2609.27307v1 Announce Type: new  Abstract: Graphical User Interface (GUI) agents enable the fulfillment of complex user instructions through multi-turn interactions with software environments, requiring step-wise reasoning and long-horizon memory to guide actions and retain task-relevant information, respectively. Recent on-policy self-distillation (OPSD) methods have achieved strong performance on GUI grounding, a foundational subtask for GUI agents, owing to dense token-level supervision from privilege-conditioned self-teachers. However, extending existing OPSD methods to multi-turn GUI agents is hindered by self-teachers' limited privilege-following ability and insufficient privileged guidance. In this paper, we introduce GUI-SD-v2, the next version of GUI-SD, which extends OPSD from GUI grounding to multi-turn GUI interaction and addresses key limitations through a two-stage training framework. Specifically, GUI-SD-v2 first strengthens privilege following by jointly optimizin
    
[^118]: 超越功效的幻觉：校准观测性信息系统研究中的准实验

    Beyond the Illusion of Power: Calibrating Quasi-Experiments in Observational IS

    [https://arxiv.org/abs/2609.27299](https://arxiv.org/abs/2609.27299)

    该论文通过大规模蒙特卡洛模拟（9837个参数条件、约980万个数据集）分解了观测性IS研究中准实验设计计划功效与实际功效之间的差距，发现序列相关可由AR(1)感知的计算器部分校正，但面板流失、错位采用偏差和平行趋势预检验无法用闭式公式刻画，仅外生流失就会使功效降低约8至11个百分点。

    

    信息系统（IS）研究者越来越多地使用双重差分（DiD）和工具变量（IV）等准实验方法，从观测面板数据中恢复因果效应。用于论证这些设计的功效计算通常假设误差独立同分布（i.i.d.），但更深层的问题在于，即使是考虑了聚类稳健性的计算器也无法察觉的因素。我们报告了一项涵盖9837个参数条件（约980万个数据集）的蒙特卡洛研究，并分解了计划功效与实际功效之间的差距。其中序列相关成分在ρ已知时可由具备AR(1)感知的计算器恢复，在ρ必须从较短的预处理期估计时也可部分恢复；但面板流失、错位采用偏差以及平行趋势预检验无法被任何闭式公式所刻画；在IS研究常用的数百至一千的样本规模下，仅外生流失一项就会造成约8至11个百分点的功效损失。与处理相关、依赖结果变量的流失……

    arXiv:2609.27299v1 Announce Type: cross  Abstract: Information systems (IS) researchers increasingly use quasi-experimental methods such as difference-in-differences (DiD) and instrumental variables (IV) to recover causal effects from observational panel data. Power calculations that justify these designs assume i.i.d. errors, but the deeper problem is what even a cluster-robust calculator cannot see. We report a Monte Carlo study over 9837 parameter conditions (approx 9.8 million datasets) and decompose the planned-versus-achieved power gap. The serial-correlation component is recoverable by an AR(1)-aware calculator when rho is known, and partially when rho must be estimated from short pre-periods, but panel attrition, staggered-adoption bias, and parallel-trends pretesting are captured by no closed-form formula; exogenous attrition alone costs approx 8 to 11 percentage points at the few-hundred-to-thousand sample sizes IS studies use. Treatment-correlated, outcome-dependent attritio
    
[^119]: StateComp：学习在长程智能体中何时压缩历史

    StateComp: Learning When to Compress History in Long Horizon Agents

    [https://arxiv.org/abs/2609.27298](https://arxiv.org/abs/2609.27298)

    提出StateComp框架，根据智能体当前状态判断历史交互何时可被安全压缩，通过两阶段标注构建KEEP/READY监督信号并训练不平衡感知路由器，从而在避免过早压缩造成信息损失与过度保留带来上下文开销之间取得平衡。

    

    长程智能体在任务执行过程中会不断积累交互历史，然而随着智能体状态的演变，过去交互的重要性也在发生变化。现有的上下文管理方法大多基于固定窗口、周期性调度或当前相关性来压缩历史，忽略了一个更根本的问题：过去的交互何时才能被安全地替换？过早压缩可能会删除未来行动仍然需要的信息，而过度保守的保留则会导致巨大的上下文开销。为解决这一问题，我们提出了状态条件压缩框架StateComp，它根据智能体的当前状态来判断历史交互何时可以被安全压缩。StateComp通过两阶段标注过程构建KEEP和READY监督信号，并在冻结语言模型的隐藏表示上训练一个不平衡感知的路由器。有界状态表示进一步……

    arXiv:2609.27298v1 Announce Type: new  Abstract: Long-horizon agents continuously accumulate interaction history during task execution, yet the importance of past interactions changes as the agent state evolves. Existing context management methods largely compress history based on fixed windows, periodic schedules, or current relevance, overlooking a more fundamental question: when has a past interaction become safe to replace? Premature compression may remove information still needed for future actions, while overly conservative retention leads to substantial context overhead. To address this, we propose State Conditioned Compression (StateComp), a framework that determines when historical interactions can be safely compressed according to the current agent state. StateComp constructs KEEP and READY supervision through a two-stage annotation procedure and trains an imbalance-aware router on hidden representations from a frozen language model. A bounded state representation further red
    
[^120]: 大知识模型：从论文到科学推理图景

    Large Knowledge Model: From Papers to a Scientific Reasoning Landscape

    [https://arxiv.org/abs/2609.27297](https://arxiv.org/abs/2609.27297)

    本文提出大知识模型（LKM），将论文表示为基于原文来源的推理图，构建包含问题、工作流和证据三个视图的科学推理图景，使科学文献成为可计算访问的共享推理资源，从而支持大规模利用文献中的科学推理过程。

    

    积累的科学知识之所以能推动科学探究，是因为已有研究发现可以帮助研究者选择新问题、设计研究方案并解释结果。要在大规模上实现这一价值，需要获取连接研究问题、科学程序、结论和证据的推理过程。我们提出了大知识模型，这是一种科学知识基础设施，能将科学文献转化为共享的、可计算访问的推理资源。LKM 将论文表示为基于原文来源的推理图，将结构化遍历与对同一对象的语义检索相结合，并对齐跨论文的相关问题、论断和推理链。这种表示构成了一个包含三个相互关联视图的科学推理图景：组织研究问题和开放方向的问题图景、揭示可复用科学程序的工作流图景，以及连接（摘要在此处截断）……的图景

    arXiv:2609.27297v1 Announce Type: new  Abstract: Accumulated scientific knowledge advances inquiry when prior findings help researchers choose new questions, design investigations, and interpret results. Realizing this value at scale requires access to the reasoning that connects research problems, scientific procedures, conclusions, and evidence. We introduce the Large Knowledge Model (LKM), a scientific knowledge infrastructure that transforms the literature into a shared, computationally accessible reasoning resource. LKM represents papers as source-grounded reasoning graphs, couples structural traversal with semantic retrieval over the same objects, and aligns related questions, claims, and reasoning chains across papers. This representation forms a Scientific Reasoning Landscape with three connected views: a Question Landscape that organizes research problems and open directions, a Workflow Landscape that exposes reusable scientific procedures, and an Evidence Landscape that conne
    
[^121]: Teach-to-Crash：一种用于碰撞诱导测试场景生成的闭环师生大语言模型框架

    Teach-to-Crash: A Closed-Loop Student-Teacher LLM Framework for Collision-Inducing Test Scenario Generation

    [https://arxiv.org/abs/2609.27296](https://arxiv.org/abs/2609.27296)

    该论文提出了Teach-to-Crash框架，通过高推理能力的教师LLM仅在搜索指标停滞时对学生LLM进行自适应战略干预，指导其生成可执行且多样化的碰撞诱导测试场景，从而在CARLA仿真中实现最高的碰撞命中率（90.79%），为自动驾驶系统的高效安全验证提供了闭环测试方案。

    

    在仿真环境中验证自动驾驶系统（ADS）需要一种测试架构，既能发现罕见的安全关键故障，又能生成可执行、多样化且对下游故障分析有用的场景。我们提出了Teach-to-Crash，一个闭环测试框架，它结合了受限的自我车辆中心场景表示、停滞感知搜索控制以及用于自适应故障发现的双LLM架构。高推理能力的教师LLM充当自适应搜索控制器，而低推理能力的学生LLM以严格的JSON模式输出仿真器可执行的场景。教师LLM仅在滚动碰撞率和碰撞时间（TTC）指标出现停滞时才进行干预，提供战略性指导以重新定向搜索方向。在一个CARLA案例研究中，通过两种改变自我车辆速度策略的实验设置，Teach-to-Crash实现了最高的碰撞命中率（90.79%）以及最短的平均TTC……（原文此处截断）

    arXiv:2609.27296v1 Announce Type: cross  Abstract: Validating Autonomous Driving Systems (ADS) in simulation requires testing architectures that can discover rare, safety-critical failures while generating scenarios that are executable, diverse, and useful for downstream failure analysis. We introduce Teach-to-Crash, a closed-loop testing framework that combines a constrained ego-centric scenario representation, stagnation-aware search control, and a dual-LLM architecture for adaptive failure discovery. A high-reasoning Teacher LLM acts as an adaptive search controller, while a low-reasoning Student LLM emits simulator-executable scenarios in a strict JSON schema. The Teacher intervenes only when rolling collision rate and time-to-collision metrics stagnate, providing strategic guidance to redirect the search. In a CARLA case study with two experimental setups that vary the ego vehicle's speed policy, Teach-to-Crash achieves the highest Collision Hit Rate (90.79%), the shortest mean Ti
    
[^122]: KITE：面向高效智能体大语言模型扩展的KV不变Transformer扩展方法

    KITE: KV-Invariant Transformer Expansion for Efficient Agentic LLM Scaling

    [https://arxiv.org/abs/2609.27294](https://arxiv.org/abs/2609.27294)

    KITE提出了一种新的模型扩展范式，通过将新增参数放置在不影响注意力KV缓存的区域，使模型在从小到大扩展时既节省训练成本（通过升级复用），又节省推理成本（KV预填充只需依赖较小的模型部分）。

    

    扩展语言模型不仅仅关乎最终质量：架构选择决定了在训练、提示处理和自回归解码过程中，为达到特定模型质量所需花费的计算量。理想的模型架构应当降低上述所有计算成本，以便于扩展到更大的模型，同时确保更大的模型确实优于较小的基线模型。我们提出了KV不变Transformer扩展，这是一种能够实现该目标的扩展范式。它将模型从小尺寸训练到更大尺寸（即通过升级复用来节省训练成本），同时将新增加的参数放置在不影响注意力KV的区域。因此，在推理过程中，KV的预填充仅依赖于模型的较小部分，从而节省了推理成本。作为一个具体的实例化方案，我们提出了步进缩放Transformer（SST），这是一种双塔解码器，其中一个塔产生KV，另一个塔……

    arXiv:2609.27294v1 Announce Type: cross  Abstract: Scaling a language model is not only a question of final quality: the architectural choice determines how much computation is spent during training, prompt processing, and autoregressive decoding to achieve certain model quality. An ideal model architecture should lower all above computation costs to facilitate scaling to a larger model, while ensure the larger model indeed outperforms smaller baselines. We introduce KV-Invariant Transformer Expansion (KITE), a scaling paradigm that achieves this goal. It trains the model from a smaller size to a larger size (i.e., saving training costs via upcycling), while places newly added parameters in regions that do not affect attention KV. Consequently, during inference, prefilling KV only relies on the smaller part of the model, so the inference costs are saved. As a concrete instantiation, we present Step Scale Transformer (SST), a two-tower decoder in which one tower produces KV and the othe
    
[^123]: 面向气候感知数字孪生的稀疏观测大气热力学预报：基于物理信息神经网络的方法

    Sparse-Observation Atmospheric Thermal Forecasting with Physics-Informed Neural Networks for Climate-Aware Digital Twins

    [https://arxiv.org/abs/2609.27290](https://arxiv.org/abs/2609.27290)

    该研究提出一种受热力学平流-源项方程和非绝热源项闭合（提前冻结）约束的物理信息神经网络，在观测稀疏条件下利用ERA5再分析数据实现1–3小时的位温预报，相比最强基线取得了明显的RMSE改善，并通过对照实验区分了物理约束与未来强迫信息各自的贡献。

    

    短期大气温度预报对于支撑气候感知数字孪生系统十分必要，但此类预报必须在热力观测不完整的情况下生成。本研究评估了一种用于位温预报的物理信息神经网络（PINN），该网络受气压坐标系下的热力学平流-源项方程约束，并采用由前12小时时段拟合、且在未来时段训练前即被冻结的非绝热源项闭合方案。模型使用三个气压层的逐小时ERA5再分析数据，以1、2、3小时预报时效进行条件性事后检验评估，并与持续性预报、局地趋势预报以及两个匹配的神经网络基线进行比较，其中一个基线接收与PINN相同的未来气象强迫场，从而有助于区分物理约束本身的作用与获取未来强迫信息的作用。在俄克拉荷马州的发展案例中，相对于最强基线的平均RMSE改善从1小时的8.1%提升至……（摘要在此处截断）

    arXiv:2609.27290v1 Announce Type: new  Abstract: Short-horizon forecasts of atmospheric temperature are needed to support climate-aware digital-twin systems, but such forecasts must be produced where thermal observations are incomplete. This study evaluates a physics-informed neural network for potential-temperature forecasting, constrained by a pressure-coordinate thermodynamic advection-source equation and a diabatic-source closure fit from the preceding 12-hour period and frozen before future-time training. Using hourly ERA5 reanalysis at three pressure levels, the model is evaluated as a conditional hindcast at lead times of one, two and three hours against persistence, local-trend, and two matched neural-network baselines, one of which receives the same future meteorological forcing as the PINN, helping distinguish the physical constraint from access to future forcing. In an Oklahoma development case, mean RMSE improvement over the strongest baseline grew from 8.1\% at one hour to
    
[^124]: Ruby-ASR：面向正字与词汇读音联合识别的证据保留式监督

    Ruby-ASR: Evidence-Preserving Supervision for Joint Orthographic and Lexical-Reading Recognition

    [https://arxiv.org/abs/2609.27289](https://arxiv.org/abs/2609.27289)

    Ruby-ASR将日语ASR的监督目标细化为书写片段与其语音实现读音局部绑定的ruby序列（辅以莫拉级CTC单调读音监督），使模型能够同时输出正字形式与词汇读音并确定性恢复两种视图，解决了同形异读在传统正字监督中丢失、事后G2P无法可靠还原的问题。

    

    传统的日语自动语音识别（ASR）以正字法转录文本作为监督，然而同一书写形式可能在语音中对应不同的词汇读音。由于这类语音被赋予完全相同的目标，其读音差异在监督接口中缺失，并且事后仅基于文本的字素到音素转换也无法可靠地恢复这一差异。我们提出Ruby-ASR，它将传统目标细化为一种片段绑定的“正字—词汇读音”序列。不同于分离的整句正字输出与音系输出，这种ruby表示将每个书写片段与其在语音中实现的读音进行局部绑定，并允许对两种视图进行确定性恢复。我们使用Qwen3-ASR骨干网络，在字幕风格和逐字风格两种转录约定下实例化该目标，并以莫拉级CTC目标提供辅助的单调读音监督。实验结果在五个（数据集上）……

    arXiv:2609.27289v1 Announce Type: cross  Abstract: Conventional Japanese automatic speech recognition (ASR) is supervised by an orthographic transcript, although the same written form can correspond to different lexical readings realized in speech. Such utterances receive an identical target, so their reading distinction is absent from the supervision interface and cannot be recovered reliably by post-hoc text-only grapheme-to-phoneme conversion. We present Ruby-ASR, which refines the conventional target into a span-bound orthographic--lexical-reading sequence. Unlike separate full-sentence orthographic and phonological outputs, the ruby representation locally binds each written span to its realized reading and permits deterministic recovery of both views. We instantiate the target under subtitle-style and verbatim-style transcription conventions using a Qwen3-ASR backbone; a mora-level CTC objective provides auxiliary monotonic reading supervision. The experimental results across five
    
[^125]: PotARCin：抽象推理任务中技能获取的多维度评估

    PotARCin: Multi-Dimensional Evaluation of Skill Acquisition in Abstract Reasoning Tasks

    [https://arxiv.org/abs/2609.27288](https://arxiv.org/abs/2609.27288)

    PotARCin将ARC基准扩展为五个维度（定义、分类、受约束生成、编辑、反转）来评估模型对任务底层抽象规则的理解，并通过程序化生成任务实例，揭示出最先进模型在单一输出预测与真正抽象技能获取之间存在25-52个百分点的性能差距。

    

    抽象与推理语料库（ARC）已成为评估AI模型通用抽象推理和流体智能的重要基准。然而，标准的ARC评估仅考虑单一能力：为测试输入生成正确的输出网格。我们认为这种狭窄的评估格式无法评估真正的抽象技能获取所应展现的多样化能力。我们提出了PotARCin，一个对ARC进行扩展的基准，从五个维度评估模型对任务底层抽象规则的理解：定义、分类、受约束生成、编辑和反转。PotARCin采用程序化方法为给定的ARC任务生成新的任务实例并转换给定输入，实现了超越固定输入输出对的动态生成式采样。在ARC-AGI-1训练集上对五个最先进模型的评估中，我们观察到标准评估与（摘要在此处截断）之间25-52个百分点的性能差距。

    arXiv:2609.27288v1 Announce Type: new  Abstract: The Abstraction and Reasoning Corpus (ARC) has become a prominent benchmark for evaluating general abstract reasoning and fluid intelligence in AI models. Yet standard ARC evaluation considers only a single capability: producing the correct output grid for a test input. We argue that this narrow format fails to evaluate the diversity of abilities that genuine abstract skill acquisition should enable. We introduce PotARCin, a benchmark that extends ARC by assessing understanding of a task's underlying abstract rule across five dimensions: Definition, Classification, Constrained Generation, Editing, and Inversion. PotARCin employs programmatic methods to generate new task instances and transform given inputs for a given ARC task, enabling dynamic generative sampling beyond fixed input-output pairs. Across five state-of-the-art models evaluated on the ARC-AGI-1 training set, we observe a 25-52 percentage-point performance gap between standa
    
[^126]: 长时程智能体中记忆控制信号先于行动出现

    Memory Control Signals Emerge Before Action in Long Horizon Agents

    [https://arxiv.org/abs/2609.27286](https://arxiv.org/abs/2609.27286)

    该论文发现长时程语言模型智能体在行动之前，其内部隐藏状态就已经编码了对记忆压缩和召回的需求信号，这些信号可用于指导上下文记忆管理决策。

    

    长时程语言模型智能体会持续积累交互历史，这不仅增加了计算成本，也使相关信息的保存与复用变得更加困难。现有的上下文管理方法主要关注如何压缩或检索历史，但在很大程度上回避了一个开放性问题：模型本身是否在这些记忆操作发生之前就已经表征了对这些操作的需求。我们研究了每个智能体行动前一刻的隐藏状态，发现压缩需求与召回需求已经被编码在模型的内部表征之中。这些信号无法用简单的上下文长度或交互进度来解释，并且它们在模型的不同深度层上呈现出截然不同的形成模式。我们进一步表明，大部分记忆决策信息被保存在紧凑的近期上下文中，而选择性恢复的历史证据则可以补充近期上下文所遗漏的长程依赖关系。基于这些发现，我们提出了Pr（原文在此处截断）

    arXiv:2609.27286v1 Announce Type: new  Abstract: Long horizon language model agents continuously accumulate interaction history, increasing computational cost while making relevant information harder to preserve and reuse. Existing context management methods mainly focus on how to compress or retrieve history, but largely leave open whether the model itself already represents the need for these memory operations before they occur. We study the hidden state immediately before each agent action and find that compression and recall needs are already encoded in the model's internal representations. These signals cannot be explained by simple context length or interaction progress, and they exhibit distinct formation patterns across model depth. We further show that most memory decision information is preserved in a compact recent context, while selectively restored historical evidence complements the long range dependencies that recent context misses. Based on these findings, we propose Pr
    
[^127]: 混元-A13B 技术报告

    Hunyuan-A13B Technical Report

    [https://arxiv.org/abs/2609.27284](https://arxiv.org/abs/2609.27284)

    混元-A13B是一个开源混合专家架构大语言模型，总参数800亿但推理时仅激活130亿，并通过快/慢思考双模式思维链框架在数学、编程、智能体等任务上达到接近更大模型的性能，兼顾能力、效率与部署成本。

    

    我们推出了混元-A13B（Hunyuan-A13B），这是一个基于混合专家架构的开源大语言模型。它包含800亿总参数，但在推理时仅激活130亿参数，从而在模型能力、计算效率和部署成本之间取得平衡。该模型在经过严格筛选的20万亿词元语料库上进行预训练，并加强了STEM（科学、技术、工程、数学）数据的筛选，提升了事实可靠性和推理能力。高质量的监督微调和大规模强化学习进一步增强了其整体性能。混元-A13B还引入了双模式思维链框架，可根据任务复杂度自适应调整推理深度：对常规查询使用快速思考，对复杂的多步骤问题使用慢速思考。评估结果显示，该模型在数学、科学、编程、通用语言理解和智能体任务中均展现出有竞争力的表现，其性能往往接近规模大得多的模型。其高推理吞吐量使

    arXiv:2609.27284v1 Announce Type: new  Abstract: We present Hunyuan-A13B, an open-source large language model based on a Mixture-of-Experts architecture. It contains 80 billion total parameters but activates only 13 billion during inference, balancing model capability, computational efficiency, and deployment cost. The model is pretrained on a rigorously filtered 20T-token corpus with enhanced STEM data curation, improving factual reliability and reasoning ability. High-quality supervised fine-tuning and large-scale reinforcement learning further enhance its overall performance. Hunyuan-A13B also introduces a dual-mode Chain-of-Thought framework that adapts reasoning depth to task complexity: fast thinking for routine queries and slow thinking for complex, multi-step problems. Evaluations show competitive performance across mathematics, science, programming, general language understanding, and agent tasks, often approaching that of much larger models. Its high inference throughput make
    
[^128]: EnSIMem：面向智能体长期记忆的实体结构化索引

    EnSIMem: Entity-Structured Indexing for Long-Term Agent Memory

    [https://arxiv.org/abs/2609.27279](https://arxiv.org/abs/2609.27279)

    EnSIMem提出了一种实体结构化的智能体长期记忆架构，通过离线构建[实体][实体类型][属性：值]形式的对话索引条目，并结合在线的实体-属性查找与自适应检索机制，帮助智能体从不断增长的交互历史中准确识别实体、属性及其支持证据。

    

    与用户进行长期交互的智能体必须能够从不断增长的交互历史中回忆事实、偏好、事件和变化。现有的记忆系统通常将交互压缩为通用摘要或检索匿名的文本块，这使得智能体难以识别正确的实体、属性和支持证据。我们提出了EnSIMem，一种面向智能体的实体结构化长期记忆架构。在离线构建阶段，系统将交互组织成主题连贯的情景片段，并构建以对话为基础的索引条目，其形式为[实体][实体类型][属性：值]。每个条目保留其来源对话轮次、时间信息以及可用的多模态字段。在在线交互阶段，智能体的请求被分解为证据需求，其属性与记忆索引对齐，随后通过实体-属性查找和自适应检索来收集所需的证据。

    arXiv:2609.27279v1 Announce Type: new  Abstract: An agent that interacts with users over long periods must recall facts, preferences, events, and changes from a continuously growing interaction history. Existing memory systems often compress interactions into generic summaries or retrieve anonymous text chunks, making it difficult for an agent to identify the correct entity, property, and supporting evidence. We present EnSIMem, an entity-structured long-term memory architecture for an agent. During offline construction, the system organizes interactions into theme-coherent episodes and builds dialogue-grounded index entries of the form [entity][entity type][property:value]. Each entry preserves its source turns, temporal information, and available multimodal fields. During online interaction, the agent's request is decomposed into evidence requirements whose properties are aligned with the memory index. Entity-property lookup and adaptive retrieval then collect the evidence needed for
    
[^129]: TimeEvo：时间序列智能体的故障驱动自进化

    TimeEvo: Failure-Driven Self-Evolution of a Time Series Agent

    [https://arxiv.org/abs/2609.27277](https://arxiv.org/abs/2609.27277)

    提出TimeEvo框架，通过将智能体的失败诊断聚类为能力缺口、为每个缺口规划测量、合成证据工具并通过配对准入门控筛选，实现了时间序列智能体工具库的故障驱动自进化，解决了人-智体工具错配和静默损害两大问题。

    

    时间序列智能体通过调用外部工具来回答分析性问题，而智能体携带哪些工具是由人类在智能体运行之前预先决定的。然而，我们发现了这种设置中的两个失败模式。人-智体工具错配：一个由21个专家精心策划的工具库在某些任务上有所帮助，却在另一些任务上造成损害，在我们测试的所有骨干模型上都降低了异常检测准确率。静默损害：一轮通用的自我修订会改变147个答案并破坏其中56个，而最终分数的变化却不到1分。两者都源于同一个缺口：工具是否有帮助是在运行时逐个问题决定的，而工具却是预先提供的，且仅用一个平均值来评判。为解决此问题，我们提出TimeEvo，它将智能体诊断出的失败聚类为能力缺口，为每个缺口规划一个测量方法，合成仅基于证据的工具来填补这些缺口，并且仅通过配对准入门控来接纳候选工具库。在十个时间序列问答数据集上的实验……

    arXiv:2609.27277v1 Announce Type: new  Abstract: Time series agents answer analytical questions by calling external tools, and which tools they carry is decided by people before the agent runs. However, we identify two failures in this setup. Human-Agent Tool Misalignment: a library of 21 expert-curated tools helps on some tasks and hurts on others, dropping anomaly accuracy under every backbone we test. Silent Harm: one round of generic self-revision changes 147 answers and breaks 56 of them, while the final score moves by less than a point. Both follow from the same gap: whether a tool helps is decided question by question at runtime, while tools are supplied in advance and judged by a single average. To address this, we propose TimeEvo, which clusters an agent's diagnosed failures into capability gaps, plans a measurement for each, synthesizes evidence-only tools that fill them, and admits the candidate library only through a paired admission gate. Experiments on ten time series QA 
    
[^130]: DRSR：学习集合级删除风险以实现高效长程智能体

    DRSR: Learning Set-Level Deletion Risk for Efficient Long-Horizon Agents

    [https://arxiv.org/abs/2609.27276](https://arxiv.org/abs/2609.27276)

    提出DRSR方法，将智能体历史压缩建模为删除集合上的风险约束选择问题，通过离线联合删除历史块构建精确的反事实监督，训练轻量级评分器预测集合级删除风险，从而超越独立打分策略，实现高效的长程智能体历史管理。

    

    长程语言模型智能体会累积推理轨迹、工具交互记录和观察结果，这些内容的相关性会随当前决策而变化。现有的压缩策略通常独立地为历史单元打分，但删除多个单元的安全性一般并非由它们的单独评分决定：冗余证据、累积的小效应以及删除后保留的信息都至关重要。我们提出了直接关系集合风险剪枝（Direct Relational Set-Risk Pruning, DRSR），该方法将智能体历史压缩建模为删除集合上的风险约束选择问题。在离线阶段，DRSR 通过联合删除符合协议的历史块，并测量同一记录的下一输出在教师强制似然上的变化，来构建精确的反事实监督信号。随后，一个轻量级评分器根据候选历史与当前动作前状态之间的在线可见关系，以及被删除内容与保留内容和成对单元之间的关系，来预测集合级别的删除危害。

    arXiv:2609.27276v1 Announce Type: new  Abstract: Long-horizon language-model agents accumulate reasoning traces, tool exchanges, and observations whose relevance changes with the current decision. Existing compression strategies often score historical units independently, but the safety of deleting several units is generally not determined by their singleton scores: redundant evidence, accumulated small effects, and the information that remains after deletion all matter. We introduce Direct Relational Set-Risk Pruning (DRSR), which formulates agent-history compression as risk-constrained selection over deletion sets. Offline, DRSR constructs exact counterfactual supervision by jointly deleting protocol-valid history Blocks and measuring the change in teacher-forced likelihood of the same recorded next output. A lightweight scorer then predicts set-level harm from online-visible relations between candidate history and the current pre-action state, together with deleted-retained and pair
    
[^131]: CAVEAT：迈向激励机制错位环境下鲁棒的计算机使用智能体

    CAVEAT: Towards Robust Computer-Use Agents in Incentive-Misaligned Environments

    [https://arxiv.org/abs/2609.27273](https://arxiv.org/abs/2609.27273)

    本文提出CAVEAT基准，揭示当购物平台环境内置与用户利益相悖的引导机制时，计算机使用智能体选购用户最优产品的成功率从78.6%骤降至17.3%，暴露了智能体在激励错位环境中的严重脆弱性。

    

    计算机使用智能体日益在网络上代表用户执行操作。当它们所处的环境激励与用户的利益不一致时，会发生什么？例如，在在线购物平台中，平台可能偏袒某些产品，从而可能引导智能体偏离用户的目标。现有的计算机使用智能体基准测试只涵盖协作环境或显式攻击，并未测试当环境本身与结果存在利害关系时，智能体能否坚持用户的目标。我们提出了CAVEAT，这是一个受控基准，涵盖九个购物平台环境以及八种常见引导机制的分类体系。在五个模型家族上的实验表明，在匹配对照情景中，智能体有78.6%的概率购买用户最优产品，但当启用引导机制时，这一比例仅为17.3%。更大的模型规模和更多的推理可以提升鲁棒性，但仍然存在大量失败。我们的轨迹分析和针对性消融实验识别出三种主要的失败模式。

    arXiv:2609.27273v1 Announce Type: new  Abstract: Computer-use agents (CUAs) increasingly act on behalf of users online. What happens when the environments they operate in have incentives that do not align with the user's? In online marketplaces, for example, platforms may favor some products over others, potentially steering agents away from the user's objective. Existing CUA benchmarks cover cooperative settings or explicit attacks, but do not test whether agents preserve user objectives when the environment itself has a stake in the outcome. We introduce CAVEAT, a controlled benchmark spanning nine marketplace environments and a taxonomy of eight common steering mechanisms. Across five model families, agents purchase the user-optimal product in 78.6% of matched-control episodes but only 17.3% when steering mechanisms are enabled. Larger models and increased reasoning improve robustness, but substantial failures persist. Our trajectory analysis and targeted ablations identify three po
    
[^132]: 风险敏感的薛定谔桥：并非KL投影

    The Risk-Sensitive Schr\"odinger Bridge: Is Not a KL Projection

    [https://arxiv.org/abs/2609.27250](https://arxiv.org/abs/2609.27250)

    本文证明了当薛定谔桥问题中的期望路径代价被熵风险测度取代且端点约束保持不变时，所得的风险敏感薛定谔桥不再能表示为对任何固定路径空间参考测度的受约束KL投影，从而打破了经典薛定谔桥基于Girsanov定理的KL投影结构。

    

    薛定谔桥的计算能力源于一个单一的结构性事实：根据Girsanov定理，该受控问题是向一个固定参考测度的Kullback-Leibler（KL）投影，可通过交替投影求解。本简报表明，这一事实在风险敏感情形下不再成立。当期望路径代价被熵风险测度取代，且两个端点边缘分布均作为硬约束保留时，所得到的不动点桥值 $J_\theta$（即在强制终端约束的乘子处软问题的取值）无法表示为相对于任何具有正则端点分布的固定路径空间参考测度的受约束KL最小化（该类参考测度严格大于一致椭圆扩散参考测度类：不要求马尔可夫性质），即使允许一个依赖于初始边缘分布的加性归一化也是如此。此外，没有任何单一参考测度能够在风险参数上生成这一单参数族（摘要在此处截断）。

    arXiv:2609.27250v1 Announce Type: cross  Abstract: The Schr\"odinger bridge owes its computational power to a single structural fact: by Girsanov's theorem the controlled problem is a Kullback--Leibler (KL) projection onto a fixed reference measure, solvable by alternating projections. This letter shows that the fact does not survive risk sensitivity. When the expected path cost is replaced by the entropic risk measure and both endpoint marginals are kept as hard constraints, the resulting fixed-point bridge value $J_\theta$ (the soft-problem value at the multiplier that enforces the terminal constraint) admits no representation as a constrained KL minimum against any fixed path-space reference with a regular endpoint law (a class strictly larger than the uniformly elliptic diffusion references: no Markov property is required), even allowing an additive normalisation depending on the initial marginal. Moreover, no single reference generates the one-parameter family in the risk paramete
    
[^133]: 倾听与镜像：言语契合与行为模仿对VR中具身AI代理社会感知与共情感知的影响

    Listening and Mirroring: The Effects of Verbal Attunement and Behavioral Mimicry on Social and Empathic Perceptions of Embodied AI Agents in VR

    [https://arxiv.org/abs/2609.27246](https://arxiv.org/abs/2609.27246)

    本研究开发了一款将对话式AI与实时面部表情及姿态模仿相结合的具身AI心理咨询师，并通过2×2被试内实验考察言语契合与行为模仿如何影响用户对VR中具身AI代理的社会与共情感知。

    

    随着具身代理在VR中承担越来越强的社交与关系角色，仅依靠视觉真实感和具身化可能并不足够；用户还需要感知到这些代理在情感上是契合的、支持性的和类人的。先前的研究表明，言语契合和非言语模仿各自都能提升用户对具身代理的社会评价。然而，行为模仿的研究大多是在实时对话式AI交互之外进行的，因此对于当代理在沉浸式对话中同时生成情境响应式对话并调整其非言语行为时用户会如何反应，人们的理解仍然有限。为了填补这一空白，我们开发了一个具身AI心理咨询师，它将对话式AI与实时面部表情和姿态模仿相结合，同时产生言语契合或中性的回应。我们在一项包含20名被试的2×2被试内实验中评估了该系统，操纵了……（摘要原文在此处截断）

    arXiv:2609.27246v1 Announce Type: cross  Abstract: As embodied agents take on increasingly social and relational roles in VR, visual realism and embodiment alone may be insufficient; users must also perceive these agents as emotionally attuned, supportive, and humanlike. Prior work suggests that verbal attunement and nonverbal mimicry can each improve users' social evaluations of embodied agents. However, behavioral mimicry has largely been studied outside of real-time, conversational AI interactions, leaving limited understanding of how users respond when an agent simultaneously generates contextually responsive dialogue and adapts its nonverbal behavior during an immersive conversation. To address this gap, we developed an embodied AI counselor that combines conversational AI with real-time facial-expression and posture mimicry, while producing either verbally attuned or neutral responses. We evaluated the system in a 2 X 2 within-subjects study with 20 participants, manipulating ver
    
[^134]: 结合大语言模型与遗传搜索解决ARC-AGI-2

    Combining LLMs and Genetic Search for ARC-AGI-2

    [https://arxiv.org/abs/2609.27242](https://arxiv.org/abs/2609.27242)

    本文提出用大语言模型生成的初始程序作为遗传算法的种子种群，通过紧凑的领域特定语言将两者结合，将ARC-AGI-2前60个任务的解决率从3.3%提升至10.0%。

    

    大语言模型（LLM）可以为ARC-AGI-2任务生成程序，但所提供的计算资源仅允许进行少量次数的解决方案生成、调试和验证尝试。遗传算法可以搜索和测试多得多的程序，但随机搜索很少能从解空间的有用邻域开始。我们通过一种紧凑的领域特定语言（DSL）将这两种方法结合起来。首先，量化后的Qwen3.5-4B大语言模型为每个ARC-AGI-2任务生成一组初始程序。然后，我们使用这些程序作为初始种群的种子，并利用遗传算法将这些程序逐步进化为给定任务的解决方案。该DSL的设计确保每个变异后的程序都保持有效且可执行。LLM提出的初始程序解决了ARC-2公共评估集前60个任务中的2个（3.3%）。遗传算法额外解决了4个任务，总共得到6个正确的测试输出（10.0%）。

    arXiv:2609.27242v1 Announce Type: cross  Abstract: LLMs can generate programs for ARC-AGI-2 tasks, but the provided compute only allows a small number of attempts to generate, debug and validate solutions. Genetic algorithms can search and test many more programs, but random search rarely starts in a useful neighborhood of the solution space. We combine the two methods through a compact domain specific language (DSL). First, a quantized Qwen3.5-4B LLM generates an initial set of programs for each ARCAGI-2 task. Then, we use those programs to seed an initial population of starting programs, and use genetic algorithms to evolve these programs towards a solution to the given task. The DSL is designed such that every mutated program remains valid and can be executed. The initial programs proposed by the LLM solve 2 (3.3%) of the first 60 tasks of the ARC-2 public evaluation set. The genetic algorithm solves an additional 4, giving 6 correct test outputs in total (10.0%). If we try using ev
    
[^135]: 相遇、比较或弃答：基于知识格的确定性多跳问答系统 LatWeave

    Meet, Compare, or Abstain: LatWeave for Deterministic Multi-Hop Question Answering on Knowledge Lattices

    [https://arxiv.org/abs/2609.27225](https://arxiv.org/abs/2609.27225)

    LatWeave 将知识组织为多维知识格，把多跳问答编译为 meet、compare、abstain 三个确定性算子，使答案生成路径零 LLM、零任务训练且端到端可审计，实现逐条可复现的问答。

    

    概率式问答系统——无论是大语言模型（LLM）本身、检索增强生成（RAG），还是经过训练的多跳检索器——都将“已知什么”与“如何推理”混入单一的概率计算之中：幻觉无法根除，证据链无法审计，而且即使系统不知道答案也会强行作答。我们提出 LatWeave，它将知识组织为多维知识格，并把多跳问答编译为三个确定性算子——meet（约束求交）、compare（格序比较）与 abstain（结构性弃答）；LLM 仅出现在构建侧（一次性抽取）和查询规划侧，而答案生成路径是零 LLM、零任务训练且端到端可审计的——从而使基于 Web 发布知识的问答能够逐条复现。与其宣称全面超越 SOTA……

    arXiv:2609.27225v1 Announce Type: cross  Abstract: Probabilistic question-answering systems -- whether large language models (LLMs) themselves, retrieval-augmented generation (RAG), or trained multi-hop retrievers -- conflate "what is known" and "how to reason" into a single probabilistic computation: hallucination cannot be eradicated, evidence chains cannot be audited, and the system answers even when it does not know. We present LatWeave, which organizes knowledge into a multidimensional knowledge lattice and compiles multi-hop QA into three deterministic operators -- meet (constraint intersection), compare (lattice-order comparison), and abstain (structural abstention); LLMs appear only on the construction side (one-shot extraction) and the query-planning side, while the answer-generation path is zero-LLM, zero-task-training, and auditable end to end -- so that question answering over Web-published knowledge becomes reproducible item by item. Rather than claiming across-the-board S
    
[^136]: 学习频谱分配：面向自适应体积分割的分数阶扩散框架

    Learning Spectral Allocation: A Fractional Diffusion Framework for Adaptive Volumetric Segmentation

    [https://arxiv.org/abs/2609.27217](https://arxiv.org/abs/2609.27217)

    该论文提出从分数阶热方程推导出的双参数频谱混合算子族FHEAT，使优化器能够自动学习每个网络层所需的频谱混合程度，从而实现自适应的三维医学图像分割。

    

    我们研究三维医学图像分割中的自适应计算问题：不再设计另一种骨干网络，而是探讨每个网络阶段需要多少频谱混合，并让优化过程来给出答案。我们从分数阶热方程的离散余弦变换（DCT）解中推导出FHEAT，一个双参数算子族。分数阶α和扩散强度D共同控制该算子，且当D=0时它恰好退化为恒等映射。通过半群时间τ = D*α进行重新参数化后，相同分辨率的算子实例可以精确复合，因此跨同分辨率阶段的任意扩散分布都等价于一个强度可学习的单一Sobolev型正则化项。这一恒等极限使得每一层的优化器（而非网络设计者）能够自行决定是否需要全局混合以及混合应达到何种锐度。我们将FHEAT实例化于一个轻量级U形架构（Light-UNETR）中，并搭配具有自适应有理激活函数的Kolmogorov-Arnold混合器（KAN3D）……

    arXiv:2609.27217v1 Announce Type: cross  Abstract: We address adaptive computation in 3D medical image segmentation: instead of designing another backbone, we ask how much spectral mixing each network stage needs and let optimization answer. We derive FHEAT, a two-parameter operator family, from the discrete cosine transform (DCT) solution of a fractional heat equation. A fractional order alpha and a diffusion strength D govern the operator, and at D=0 it is exactly the identity. Reparametrized by the semigroup time tau = D*alpha, same-resolution instances compose exactly, so any distribution of diffusion across same-resolution stages amounts to a single Sobolev-type regularizer of learned strength. This identity limit lets the optimizer of each layer, not the designer, decide whether global mixing is needed and how sharp it should be. We instantiate FHEAT in a lightweight U-shaped architecture (Light-UNETR) paired with a Kolmogorov-Arnold mixer (KAN3D) with adaptive rational activatio
    
[^137]: KATOsuper：基于敏感性一致傅里叶神经算子的代理加速神经拓扑优化

    KATOsuper: Surrogate-accelerated neural topology optimization with sensitivity-consistent Fourier neural operators

    [https://arxiv.org/abs/2609.27216](https://arxiv.org/abs/2609.27216)

    该论文提出KATOsuper框架，利用敏感性一致傅里叶神经算子（SC-FNO）与forward_split架构，通过自动微分保证预测目标与优化梯度的一致性，从而解决神经代理拓扑优化中的不稳定性并实现显著加速。

    

    拓扑优化（TO）由于每次迭代都需要重复进行有限元分析（FEA）评估，计算成本依然很高。尽管基于神经网络的代理模型提供了潜在的加速可能，但现有方法往往存在预测目标与敏感性之间的梯度不一致问题，导致优化不稳定。本工作提出了KATOsuper，这是一个目标无关的框架，将神经重参数化拓扑优化与敏感性一致傅里叶神经算子（SC-FNO）相耦合。该框架采用forward_split架构，通过对预测目标场进行自动微分来导出部署的敏感性，从而保持预测目标与优化所用梯度之间的一致性。案例研究涵盖三个二维基准问题和三个三维结构，涉及柔度最小化或应力最小化。一个物理信（原文摘要在此处截断）。

    arXiv:2609.27216v1 Announce Type: cross  Abstract: Topology optimization (TO) remains computationally intensive due to repeated finite element analysis (FEA) evaluations required at each iteration. While neural network-based surrogates offer potential acceleration, existing approaches often suffer from gradient inconsistency between predicted objectives and sensitivities, leading to optimization instability. This work presents KATOsuper, an objective-agnostic framework that couples neural-reparameterized topology optimization with a Sensitivity-Consistent Fourier Neural Operator (SC-FNO). The framework employs the forward_split architecture, which derives deployed sensitivities via automatic differentiation through the predicted objective field and thereby preserves consistency between the predicted objective and the gradient used for optimization. The case studies include three 2D benchmark problems and three 3D structures considering compliance or stress minimization. A physics-infor
    
[^138]: 基于电阻曲率的可扩展子图采样

    Scalable Subgraph Sampling via Resistance Curvature

    [https://arxiv.org/abs/2609.27209](https://arxiv.org/abs/2609.27209)

    该论文提出ERC-LG，一种结合Johnson-Lindenstrauss投影与多GPU批量共轭梯度求解器的大规模图电阻曲率近似方法，避免了伪逆计算与完整嵌入存储，并利用所得曲率引导节点与边采样以构建GNN训练子图，在七个数据集中的六个上取得最高的节点分类准确率。

    

    子图采样能够降低大规模图神经网络的训练成本，但现有的采样准则可能忽视边在几何结构中的作用。我们提出了一种由电阻曲率引导的采样框架，该框架建立在ERC-LG之上——一种面向大规模图的曲率近似方法。ERC-LG将Johnson-Lindenstrauss投影与正则化的多GPU批量共轭梯度求解器相结合，避免了显式的拉普拉斯伪逆计算和完整的嵌入存储。所得的曲率用于指导节点和边采样概率，以构建GNN训练子图。实验表明，该方法与基于伪逆的曲率在数值上高度一致，且与仅使用共轭梯度法的计算相比运行时间更短。基于ERC-LG的采样变体在下游节点分类任务中，于七个真实世界数据集中的六个上取得了最高的平均准确率。

    arXiv:2609.27209v1 Announce Type: cross  Abstract: Subgraph sampling reduces the training cost of large-scale graph neural networks, but sampling criteria may overlook the geometric roles of edges. We propose a resistance-curvature-guided sampling framework built on ERC-LG, a curvature approximation method for large-scale graphs. ERC-LG combines Johnson-Lindenstrauss projections with regularized multi-GPU batched conjugate gradient solvers, avoiding explicit Laplacian pseudoinverse computation and full embedding storage. The resulting curvature informs node- and edge-sampling probabilities for constructing GNN training subgraphs. Experiments show numerical agreement with pseudoinverse-based curvature and reduced runtime compared with CG-only computation. ERC-LG-based sampling variants achieve the highest mean accuracy on six of seven real-world datasets in downstream node classification.
    
[^139]: 用户生成文本的音素化：基准、分类体系与组合式方法

    Phonemizing User-Generated Text: A Benchmark, Taxonomy, and Compositional Approach

    [https://arxiv.org/abs/2609.27205](https://arxiv.org/abs/2609.27205)

    该论文提出了首个针对用户生成文本（UGT）的多语言G2P基准UGTPhon及配套分类体系，揭示了现有模型处理非规范文本时高达66.8 PER点的系统性性能差距，并提出通过精确匹配查找和分阶段解码显式建模规范形式推理的组合式G2P方法，使0.5B小模型能与更大的前沿LLM相媲美。

    

    语音合成系统越来越多地需要处理用户生成文本（UGT），例如"ppl"和"imo"这类缩写，其发音必须从规范形式而非表面形式推断得出。我们提出了UGTPhon，这是首个针对英语、越南语和韩语用户生成文本的字素到音素（G2P）基准，并配套提供了一个基于推理的分类体系，用于细粒度诊断。现有的G2P模型和前沿大语言模型在规范形式与非规范形式文本之间表现出系统性的性能差距，最高可达66.8个PER（音素错误率）百分点。作为基准基线，我们提出了一种简单的组合式G2P方法，通过精确匹配查找和分阶段解码来纳入规范形式证据。在匹配的ByT5和Qwen2.5-0.5B骨干模型上，显式的规范形式建模持续降低了非规范文本的G2P错误。0.5B参数的变体模型还能与规模大得多的少样本前沿大语言模型相媲美，凸显了为用户生成文本显式建模规范形式推理的益处。

    arXiv:2609.27205v1 Announce Type: cross  Abstract: Text-to-speech systems increasingly process user-generated text (UGT) such as ppl and imo, whose pronunciation must be inferred from the canonical rather than surface form. We introduce UGTPhon, the first grapheme-to-phoneme (G2P) benchmark for UGT in English, Vietnamese, and Korean, together with an inference-grounded taxonomy for fine-grained diagnosis. Existing G2P models and frontier LLMs exhibit a systematic canonical-to-non-canonical performance gap, reaching up to 66.8 PER points. As a benchmark baseline, we propose a simple compositional G2P approach that incorporates canonical-form evidence through exact-match lookup and staged decoding. Across matched ByT5 and Qwen2.5-0.5B backbones, explicit canonical-form modeling consistently reduces non-canonical G2P errors. The 0.5B variant also performs competitively with much larger few-shot frontier LLMs, highlighting the benefit of explicitly modeling canonical-form inference for UGT
    
[^140]: XLOG：一个原生于CUDA的神经符号集成引擎

    XLOG: A CUDA-Native Engine for Neurosymbolic Integration

    [https://arxiv.org/abs/2609.27203](https://arxiv.org/abs/2609.27203)

    XLOG 是一个原生于 CUDA 的神经符号逻辑编程引擎，通过 GPU 知识编译实现从感知到概率推理的端到端可微计算，并借助电路缓存与最坏情况最优连接技术分别取得 2.74 倍和 27.96 倍的性能提升。

    

    xlog 是一个原生于 CUDA 的逻辑编程引擎，通过类型化前端和由提供方管理的 CUDA 运行时，将神经感知与确定性 Datalog、概率推理以及认知世界视图集成在一起。它的各种推理模式共享设备端数据平面，但其执行边界有所不同：普通的 Datalog 和精确推理由主机编排，而经过认证的常驻递归核心与蒙特卡洛采样核心则在有界的终止回执之前记录零次被追踪的主机-设备传输。概率推理路径通过 GPU 知识编译支持端到端梯度，即从溯源到 CNF 再到 Decision-DNNF 的编译、精确加权模型计数以及反向梯度。最终的平滑电路在缓存或求值之前会对照其源公式进行认证。电路缓存带来了 2.74 倍的 MNIST 加法训练加速；一个最坏情况最优的连接子系统相比 xlog 的二元连接基线带来了 27.96 倍的几何平均增益。

    arXiv:2609.27203v1 Announce Type: new  Abstract: xlog is a CUDA-native logic programming engine integrating neural perception with deterministic Datalog, probabilistic inference, and epistemic world views through a typed frontend and provider-owned CUDA runtime. Its reasoning modes share device data planes, but their execution boundaries differ: ordinary Datalog and exact inference are host-orchestrated, while certified resident recursive and Monte Carlo sampled cores record zero tracked host-device transfers before a bounded terminal receipt. The probabilistic path supports end-to-end gradients through GPU knowledge compilation from provenance to CNF to Decision-DNNF, exact weighted model counting, and backward gradients. A final smoothed circuit is certified against its source formula before caching or evaluation. Circuit caching yields a 2.74x MNIST-addition training speedup; a worst-case-optimal join subsystem yields a 27.96x geometric-mean gain over xlog's binary-join baseline. MN
    
[^141]: 通过最小风险训练增强小型语言模型以实现停电报告生成

    Enhancing Small Language Models for Power Outage Report Generation via Minimum Risk Training

    [https://arxiv.org/abs/2609.27197](https://arxiv.org/abs/2609.27197)

    通过最小风险训练对序列级评估指标进行直接优化，将小型语言模型Qwen2.5-7B-Instruct在停电报告标准化生成任务上的总体准确率从16.20%大幅提升至68.95%。

    

    最小风险训练能够使神经机器翻译模型直接优化序列级评估指标，而不仅仅依赖于词元级的最大似然目标。尽管该方法在十年前就已提出，但近期的研究表明，基于风险的优化在现代语言模型中展现出新的潜力。我们将最小风险训练应用于全国停电数据计划（ODIN）的停电报告生成任务，将异构报告转换为符合CIM IEC 61968-3标准的XML格式。我们的MRT方法将Qwen2.5-7B-Instruct的总体准确率从16.20%提升至68.95%，证明了序列级优化在领域特定结构化生成任务中的有效性。

    arXiv:2609.27197v1 Announce Type: new  Abstract: Minimum Risk Training (MRT) enables neural machine translation models to directly optimize sequence-level evaluation metrics instead of relying only on token- level maximum-likelihood objectives Shen et al. [2016]. Although introduced a decade ago, recent work shows renewed potential for risk-based optimization in modern language models Yang et al. [2024], Jinnai et al. [2025]. We apply MRT to power outage report generation for the Outage Data Initiative Nationwide (ODIN), transforming heterogeneous reports into standardized XML compliant with CIM IEC 61968-3. Our MRT approach improves Qwen2.5-7B-Instruct overall accuracy from 16.20% to 68.95%, demonstrating the effectiveness of sequence- level optimization for domain-specific structured generation
    
[^142]: 基于争议经验记忆巩固的自演化多媒体验证

    Self-Evolving Multimedia Verification through Memory Consolidation of Contestation Experiences

    [https://arxiv.org/abs/2609.27175](https://arxiv.org/abs/2609.27175)

    SEMV是一个自演化多智能体多媒体验证框架，通过将可溯源论证、范围限定因果修订与争议经验记忆巩固相结合，在COSMOS基准上达到91.88%的准确率，并将负迁移从5.7%降至0.2%。

    

    多媒体验证不仅需要准确的判断，还需要可追溯的证据、可靠的人工纠正机制以及先前经验的安全复用。现有系统往往缺乏明确的机制来修订中间推理过程或防止有害的知识迁移。我们提出了SEMV（自演化多媒体验证），这是一个自演化的多智能体框架，它将带有来源溯源信息的论证作为连接证据、推理、人工争议与记忆之间的接口。SEMV结合了基于竞技场的定量双极论证（A-QBAF）、因果性与范围限定修订，以及具有显式冲突保留的验证门控记忆巩固机制。在COSMOS基准测试中，SEMV达到了91.88%的准确率，而最强可比基线为89.10%。经验证的记忆将负迁移从5.7%降低至0.2%。在由审稿人争议构建的CTR基准测试中，范围限定的因果修订纠正了96.7%的初始错误。

    arXiv:2609.27175v1 Announce Type: cross  Abstract: Multimedia verification requires not only accurate decisions but also traceable evidence, reliable human correction, and safe reuse of prior experience. Existing systems often lack explicit mechanisms for revising intermediate reasoning or preventing harmful knowledge transfer. We present SEMV (Self-Evolving Multimedia Verification), a self-evolving multi-agent framework that treats provenance-bearing arguments as the interface between evidence, reasoning, human contestation, and memory. SEMV combines arena-based quantitative bipolar argumentation (A-QBAF), causal and scoped revision, and verification-gated memory consolidation with explicit conflict retention. On COSMOS benchmark, SEMV achieves 91.88% accuracy versus 89.10% for the strongest comparable baseline. Verified memory reduces negative transfer from 5.7% to 0.2%. On CTR benchmark, constructed from reviewer contestations, scoped causal revision corrects 96.7% of initial errors
    
[^143]: 计数证据，而非句子：面向长文本价值测量的大语言模型判断缓和证据融合

    Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement

    [https://arxiv.org/abs/2609.27165](https://arxiv.org/abs/2609.27165)

    本文提出无需训练的缓和证据融合（TEF）规则，依据由广义贝叶斯后验导出的归一化信息增益对句子级LLM判断进行加权，使不确定句子对融合得分的贡献近乎为零、同时保留决定性证据的贝叶斯最优权重，从而更准确地从长文本中测量价值取向，并发布了MIND基准。

    

    大语言模型越来越多地被用于从长篇社交媒体帖子中测量公共价值取向，然而这类帖子往往混杂着背景信息、引用、让步表达，真正承载立场的句子只有少数几个。现有方法要么让模型直接预测文档级标签，这可能导致过度自信；要么通过多数投票或软投票聚合句子级预测，这将不确定的句子与决定性的句子视为具有同等信息量。我们将长文本价值测量形式化为一个决策融合问题，并提出缓和证据融合，这是一种无需训练的规则，它根据归一化信息增益对每个句子的对数几率进行加权，该权重源自广义贝叶斯后验。这使得融合得分对于不确定的句子几乎趋于消失，同时保留了决定性证据的贝叶斯最优权重。我们进一步引入多事件洞察网络维度（MIND），这是一个包含8,358个……的基准（摘要原文在此处截断）。

    arXiv:2609.27165v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used to measure public value orientations from long social media posts, yet such posts often mix background, quotations, concessions, and only a few stance-bearing sentences. Existing approaches either ask the model to predict a document-level label directly, which can be overconfident, or aggregate sentence-level predictions by majority or soft voting, which treat uncertain and decisive sentences as equally informative. We formulate long-text value measurement as a decision-fusion problem and propose Tempered Evidence Fusion (TEF), a training-free rule that weights each sentence's log-odds by its normalized information gain, as derived from a generalized Bayesian posterior. This makes the fused score nearly vanish for uncertain sentences while preserving the Bayes-optimal weight of decisive evidence. We further introduce Multi-event Insight Network Dimensions (MIND), a benchmark of 8,358 C
    
[^144]: 线性表示假说需要一个群作用

    The Linear Representation Hypothesis Needs a Group Action

    [https://arxiv.org/abs/2609.27158](https://arxiv.org/abs/2609.27158)

    论文指出线性表示假说实际上是由表示等价性区分的一族假说，并提出用群作用将其形式化——明确表示对象、生成过程与所断言的性质——从而澄清不同度量、读取点和分析阶段之间假设的差异。

    

    为了做出能够泛化到特定训练模型之外的关于表示的论断，我们需要明确两个表示在何种情况下应被视为等价。线性表示假说通常在被讨论时并未明确说明这种等价性。不同的等价概念保留不同的结构，因此看似在研究同一表示的度量、探测器和干预手段实际上可能对应不同的假说。因此，我们认为线性表示假说并非单一假说，而是一族由表示等价性加以区分的论断。我们使用群作用将这一想法形式化，明确表示对象、生成该表示的过程以及最终所断言的性质，同时考虑由模型架构所施加的等价性。该框架阐明了假设如何在不同的度量、读取点和分析阶段之间发生变化，并且我们利用它来（摘要在此处截断）

    arXiv:2609.27158v1 Announce Type: cross  Abstract: To make claims about representations that generalize beyond a particular trained model, we need to specify when two representations should count as equivalent. The Linear Representation Hypothesis is often discussed without making this equivalence explicit. Different notions of equivalence preserve different structures, so metrics, probes, and interventions that appear to study the same representation may in fact correspond to different hypotheses. We therefore argue that the Linear Representation Hypothesis is not one hypothesis but a family of claims distinguished by representation equivalence. We formalize this idea using group actions, specifying the representation object, the procedure that produces it, and the property ultimately asserted, while accounting for equivalences imposed by the model architecture. This framework clarifies how assumptions can change across metrics, reading points, and analysis stages, and we use it to au
    
[^145]: 点赞陷阱：针对基于相似度的推荐系统中智能体的多阶段投毒攻击

    The Like Trap: Multi-Stage Poisoning against Agents in Similarity-based Recommendation Systems

    [https://arxiv.org/abs/2609.27155](https://arxiv.org/abs/2609.27155)

    该研究通过理论分析揭示了社交媒体平台推荐系统中的点赞评分机制存在可利用的漏洞，攻击者可通过多阶段投毒帖子链，以隐蔽方式操纵部署在平台上的LLM智能体的信息流。

    

    随着大语言模型（LLMs）及基于LLM的智能体的最新发展，这些智能体正变得日益自主，并能够更广泛地代表用户在互联网上执行操作。然而，部署在社交媒体平台上的自动化智能体（例如用于管理用户个人账户的智能体）的脆弱性仍未得到充分探索。现有关于智能体投毒的研究通常假设攻击者能够将投毒内容直接暴露给智能体。尽管这种攻击方式直接且有效，但更容易被检测和缓解。在社交媒体平台的背景下，这留下了一个悬而未决的问题：推荐系统本身是否会以更隐蔽的方式将此类内容推送给智能体。通过理论分析，我们证明了OASIS系统中使用的点赞评分机制可以被利用，并刻画了多阶段投毒帖子链能够操纵智能体信息流的条件。基于这些……

    arXiv:2609.27155v1 Announce Type: cross  Abstract: With recent advancements in large language models (LLMs) and LLM-based agents, these agents are becoming increasingly autonomous and gaining broader access to act on users' behalf on the internet. However, the vulnerability of automated agents deployed on social media platforms (e.g., for managing a user's personal account) remains underexplored. Existing studies on agent poisoning typically assume that the adversary can expose poisoned content to the agent. Although such an attack is direct and effective, it is more easily detected and mitigated. In the context of social media platforms, this leaves open whether the recommendation system itself would surface such content to the agent in a more subtle manner. Through theoretical analysis, we show that the like-score mechanism used in OASIS can be exploited, and we characterize the conditions under which a multi-stage chain of poisoned posts can steer the agent's feed. Based on these in
    
[^146]: 我们需要复杂的拓扑控制吗？不同同伴随机路由提升稀疏多智能体辩论的成本效率

    Do We Need Complex Topology Control? Distinct-Peer Random Routing Improves Cost-Efficiency in Sparse Multi-Agent Debate

    [https://arxiv.org/abs/2609.27150](https://arxiv.org/abs/2609.27150)

    研究表明，无需复杂的拓扑控制，仅需采用简单的“每轮与两个新采样的不同同伴随机辩论”的路由策略，就能显著改善稀疏多智能体辩论的准确性与成本之间的权衡。

    

    多智能体辩论（MAD）已成为一种有前景的范式，通过迭代式的同伴交互来提升大语言模型（LLM）的推理准确性。通信拓扑在这一过程中扮演着核心角色，这促使人们开发日益复杂的机制来学习、适应或动态重配置智能体之间的交互，以提高准确性或推理可靠性。与此同时，先前的研究表明，简单得多的稀疏通信已经能够以显著更低的成本实现有竞争力的性能。在本工作中，我们深入研究了稀疏MAD，并探讨复杂的拓扑控制对于改进集体推理是否真的必要。我们发现，一种简单的无放回随机路由策略——让每个智能体在每一轮中与两个不同的、新采样的同伴进行辩论——提供了一个出人意料的强大基线，并持续改善稀疏MAD的准确性-成本权衡。基于……

    arXiv:2609.27150v1 Announce Type: new  Abstract: Multi-agent debate (MAD) has emerged as a promising paradigm for improving the reasoning accuracy of large language models (LLMs) through iterative peer interaction. Communication topology plays a central role in this process, motivating increasingly sophisticated mechanisms that learn, adapt, or dynamically reconfigure agent interactions to improve accuracy or reasoning reliability. Meanwhile, prior studies suggest that much simpler sparse communication can already achieve competitive performance at substantially lower cost. In this work, we take a closer look at sparse MAD and ask whether complex topology control is actually necessary to improve collective reasoning. We find that a simple random-without-replacement routing policy, which lets each agent debate with two distinct and newly sampled peers at every round, provides a surprisingly strong baseline and consistently improves the accuracy-cost trade-off of sparse MAD. Building on 
    
[^147]: 面向手术的层次感知视频-语言模型评估方法与双曲基线模型

    A Hierarchy-Aware Video-Language Model Evaluation and Hyperbolic Baseline for Surgery

    [https://arxiv.org/abs/2609.27139](https://arxiv.org/abs/2609.27139)

    本文提出了首个面向手术视频理解的层次感知评估套件 SurgHiBench，以及通过蕴含锥建模阶段-步骤包含关系的双曲模型 HyperSurg，并揭示出准确率相同的模型在错误严重程度上可能存在巨大差异。

    

    手术过程遵循从阶段到步骤的层次结构，然而用于识别手术过程的视频-语言模型却采用扁平的逐层指标进行评估，忽略了跨层次的连贯性与错误结构。在本文中，我们针对这一问题做出了两项贡献：(i) 我们提出了 SurgHiBench，这是首个面向手术视频理解的层次感知评估套件，包含三个任务，用于在不同粒度层级上衡量识别能力、一致性与错误严重程度；我们在一个通用 CLIP 模型、一个欧氏空间手术模型，以及作为第二项贡献的 (ii) HyperSurg——一个通过蕴含锥强制建模阶段-步骤包含关系的新型双曲模型——上进行了评估，涵盖跨越三种手术类型的四个现有数据集。该评估套件揭示出，准确率相同的两个模型可能产生错误严重程度截然不同的预测，从正确阶段内同级步骤的混淆，到完全不相关的跨阶段预测。双曲

    arXiv:2609.27139v1 Announce Type: cross  Abstract: Surgical procedures follow a phase-to-step hierarchy, yet the video-language models used to recognize them are evaluated with flat per-level metrics that ignore cross-level coherence and error structure. In this paper we make two contributions to address this problem, (i) we introduce SurgHiBench, the first hierarchy-aware evaluation suite for surgical video understanding, with three tasks measuring recognition, consistency, and severity across granularity levels. We evaluate a general-purpose CLIP model, a Euclidean surgical model, and, as second contribution: (ii) HyperSurg, a new hyperbolic model that enforces phase-step containment via entailment cones, across four (existing) datasets spanning three procedure types. The suite reveals that two models with the same accuracy can produce predictions of very different error severity, ranging from sibling confusions within the correct phase to unrelated cross-phase predictions. Hyperboli
    
[^148]: 当客户端被编排时：战略性梯度操纵击败联邦学习服务器与高效防御

    When Clients Are Orchestrated: Strategic Gradient Manipulation to Defeat Federated Learning Servers with Efficient Defense

    [https://arxiv.org/abs/2609.27124](https://arxiv.org/abs/2609.27124)

    提出Fed-ADR攻击框架，由恶意编排服务器实时协调异构对抗客户端策略性操纵梯度更新以绕过联邦学习防御，并针对该威胁设计了基于历史更新估计客户端真实梯度的实时检测机制。

    

    联邦学习通过与服务端参数服务器（PS）交换模型更新（而非原始数据）来实现去中心化的模型训练。现有的大多数防御机制主要假设攻击者是静态的或独立行动的，而我们揭示了一类新的动态自适应攻击，能够系统性地绕过此类防护。我们提出了Fed-ADR，一个整体性的攻击框架，其中恶意编排服务器（OS）动态协调一组异构的对抗性客户端，包括有目标攻击者和无目标攻击者。通过OS的实时协调，恶意客户端策略性地调整其梯度更新以规避PS部署的防御，同时严重降低全局模型的性能或将训练引向对抗目标。为缓解这一威胁，我们提出了一种检测机制，通过历史更新来估计每个客户端的真实梯度，从而实现实时检测。

    arXiv:2609.27124v1 Announce Type: cross  Abstract: Federated Learning enables decentralized model training by exchanging model updates--rather than raw data--with a central parameter server (PS). While most of the existing defenses primarily assume static or independently acting adversaries, we reveal a new class of dynamically adaptive attacks that systematically bypass such protections. We propose Fed-ADR, a holistic attack framework in which a malicious orchestrator server (OS) dynamically coordinates a heterogeneous set of adversarial clients, including both targeted and untargeted attackers. Through real-time coordination by the OS, malicious clients strategically adapt their gradient updates to evade defenses deployed by the PS, while either severely degrading global model performance or steering training toward adversarial objectives.To mitigate this threat, we offer a detection mechanism that estimates each client's true gradient from historical updates, enabling real-time dete
    
[^149]: 基于大语言模型的可证明完备广义规划

    Provably Complete Generalized Planning with LLMs

    [https://arxiv.org/abs/2609.27105](https://arxiv.org/abs/2609.27105)

    该论文提出了将Lean定理证明器与大语言模型相结合的方法，自动生成广义规划及其完备性的形式化证明，首次通过机器验证而非人工评估来保证广义规划能够解决规划域中的所有实例。

    

    广义规划旨在计算一个能够解决规划域中所有实例的规划。近期的研究工作已使用大语言模型（LLM）以Python程序的形式自动生成并调试此类广义规划，并在多个域上实现了对测试数据的完美覆盖。然而，这些广义规划是否真正完备，即能否解决域中的所有实例，此前只能通过人工评估来确定。本文提出了一种在Lean中自动生成广义规划的方法，并同时生成其相对于输入提供的域约束规范的完备性证明。我们引入了一种保持语义的PDDL到Lean的转换方法，并使用大语言模型同时生成广义规划及其形式化证明，证明该规划能够解决满足域约束的每一个实例。完备性证明的正确性由Lean的内核来验证。我们在13个常用的基准域上评估了我们的方法。

    arXiv:2609.27105v1 Announce Type: new  Abstract: Generalized planning aims to compute a plan that solves all instances of a planning domain. Recent work has used LLMs to automatically generate and debug such generalized plans in the form of Python programs and achieved perfect test data coverage for several domains. However, whether these generalized plans are actually complete, i.e. solve all instances of the domain, could only be determined by manual evaluation. Here, we present an approach for automatically generating generalized plans in Lean together with proofs of their completeness relative to a specification of the domain constraints provided as input. We introduce a semantic-preserving PDDL-to-Lean conversion, and use an LLM to generate both the generalized plan and the formal proof that it solves every instance satisfying the domain constraints. The correctness of the completeness proof is determined by Lean's kernel. We evaluate our approach on 13 commonly used benchmark dom
    
[^150]: 跨具身形态的智能

    Intelligence Across Embodiments

    [https://arxiv.org/abs/2609.27095](https://arxiv.org/abs/2609.27095)

    该论文主张通用具身智能应依赖能够跨具身差异持续积累的学习，提出将具身多样性作为规模化的新维度，并结合广泛的习得先验，以取代依赖人工设计对应关系的短期方案。

    

    机器人具身形态涵盖了智能体与世界进行物理交互所依赖的感知、运动学、动力学、几何结构、执行与控制等特性。这些特性因机器人而异，且会随时间变化。我们认为，通用的具身智能需要能够跨这些差异不断积累的学习。当前主流的通过人工设计对应关系来弥合具身差异的方法能够带来即时的实际收益，但其假设从长远来看限制了迁移的范围。相反，更通用的方法应当发现那些能随着经验增长而支持向更大范围具身形态迁移的表征。我们提出具身多样性是规模化的一条有前景的轴，并指出广泛的习得先验是一个互补要素。我们呼吁开展能更好刻画具身差异与迁移性能的评估。更广泛地说，跨具身学习将学习的实际挑战与……（原文在此处截断）

    arXiv:2609.27095v1 Announce Type: cross  Abstract: Robotic embodiment encompasses the sensing, kinematics, dynamics, geometry, actuation, and control through which an agent physically interacts with the world. These properties vary across robots and change over time. We argue that general embodied intelligence requires learning that accumulates across these differences. Prevailing methods that engineer correspondences to bridge embodiment differences offer immediate practical gains, but their assumptions limit the scope of transfer in the long run. Instead, a more general approach should discover representations that support transfer to a larger range of embodiments as experience grows. We propose embodiment diversity as a promising axis of scaling, and identify broad learned priors as a complementary ingredient. We call for evaluations that better characterize embodiment gaps and transfer performance. More broadly, cross-embodiment learning connects the practical challenge of learning
    
[^151]: 已训练GNN中的局部证据与几何读出修复

    Local Evidence and Geometric Readout Repair in Trained GNNs

    [https://arxiv.org/abs/2609.27092](https://arxiv.org/abs/2609.27092)

    该研究通过精确质量线性规划和两种学习式后验修复方法（消息重加权与集合条件化logit平移），分离并纠正了训练后GNN节点分类错误中混合权重与logit集合定位两种成因，在八个数据集上将平均准确率从62.6%提升至65.3%，且证明logit平移贡献了绝大部分增益。

    

    许多用于节点分类的图神经网络（GNN）将线性分类器应用于局部消息的非负混合。预测错误可能源于糟糕的混合权重，也可能是可达的logit集合相对于分类器的位置不佳。我们通过一个精确质量线性规划和两种学习到的后验修复方法将这两种原因区分开来。每个重加权后的预测都存在一个等价的中心化logit平移，但只有在消息诱导的位移集合中的平移才能通过重加权实现。在八个数据集、八种GNN骨干网络和十种数据划分上，冻结模型的平均准确率为62.6%，重加权后提升至63.8%，而集合条件化平移可达到65.3%。参数量匹配的仅节点平移器达到64.6%，表明平移解释了大部分增益，而消息集合只带来较小的额外收益。尽管oracle重加权能够纠正许多错误，但无标签的重加权几乎无法捕捉这一潜力：局部证据是（摘要在此处被截断）

    arXiv:2609.27092v1 Announce Type: cross  Abstract: Many node-classification GNNs apply a linear classifier to a nonnegative mixture of local messages. An error can reflect either poor mixture weights or a reachable logit set poorly positioned for the classifier. We separate these causes with an exact-mass linear program and two learned post-hoc repairs. Every reweighted prediction has an equivalent centered logit translation, but only translations in a message-induced displacement set are realizable by reweighting. Across eight datasets, eight GNN backbones, and ten splits, mean accuracy rises from 62.6% for the frozen models to 63.8% with reweighting and 65.3% with set-conditioned translation. A parameter-matched node-only translator reaches 64.6%, showing that translation explains most of the gain while the message set supplies a smaller additional benefit. Although oracle reweighting can correct many errors, label-free reweighting captures little of this potential: local evidence is
    
[^152]: 政策即技能：具备证据验证、确定性控制与审计的受治理LLM决策支持

    Policy-as-Skill: Governed LLM Decision Support with Evidence, Deterministic Control, and Audit

    [https://arxiv.org/abs/2609.27087](https://arxiv.org/abs/2609.27087)

    提出Policy-as-Skill（PaS）模块化运行时框架，将证据验证、审查路由、版本控制和审计等治理功能打包为可执行、可版本化的政策能力，在大多数治理和审查指标上优于LLM+RAG，并通过任务相关的确定性控制将总体准确率提升至61.2%。

    

    组织日益使用大语言模型（LLM）进行政策、合规、风险和运营决策支持，这要求具备证据验证、审查路由、版本控制和可审计性。我们提出政策即技能，这是一种模块化运行时，将这些治理功能打包为可执行、可版本化的政策能力。在固定的Gemma4后端上，对13种方法在600个开发任务上进行了评估。PaS+Audit实现了53.8%的精确准确率、0.346的宏F1、0.854的审查F1、1.000的引用精确率、0.984的政策引用召回率和1.000的审计完整性，在大多数治理和审查指标上优于LLM+RAG。确定性控制将总体准确率提升至61.2%，但高度依赖具体任务，这支持选择性的而非普遍适用的基于规则的干预。

    arXiv:2609.27087v1 Announce Type: new  Abstract: Organizations increasingly use LLMs for policy, compliance, risk, and operational decision support, requiring evidence validation, review routing, version control, and auditability. We introduce Policy-as-Skill (PaS), a modular runtime that packages these functions as executable, versioned policy capabilities. Thirteen methods are evaluated with a fixed Gemma4 backend on 600 development tasks. PaS+Audit achieves 53.8% exact accuracy, macro-F1 0.346, review F1 0.854, citation precision 1.000, policy-reference recall 0.984, and audit completeness 1.000, outperforming LLM+RAG on most governance and review metrics. Deterministic control raises aggregate accuracy to 61.2% but is strongly task dependent, supporting selective rather than universal rule-based intervention.
    
[^153]: Crossflow：面向智能体LLM服务的预填充-解码弹性机制

    Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving

    [https://arxiv.org/abs/2609.27085](https://arxiv.org/abs/2609.27085)

    Crossflow针对P/D分离架构中预填充与解码需求剧烈波动（智能体负载下尤为突出）的问题，提出在不改变节点角色的前提下使预填充-解码边界弹性化，从而避免静态容量规划造成的容量闲置或排队吞吐损失。

    

    随着服务容量需求超过训练需求，服务效率变得越来越重要。预填充-解码（P/D）分离通过两个阶段的专门化与隔离来提升服务效率，但这些收益建立在静态分区的基础上。然而，阶段需求并非静态。我们观察到，在大型LLM集群中，未缓存输入与输出token的比例在分钟时间尺度上的峰值均值比高达4.7倍；而在公开的智能体负载轨迹中，单日内每小时比例的中位数跨度达24.5倍，与此同时重新分配一个副本却需要数十分钟。智能体流量进一步加剧了这种失配：若按第95百分位为各资源池配置容量，将导致多达17%的集群容量闲置；若配置低于该水平，同样的失衡则会转化为排队等待和未能实现的吞吐量。我们提出Crossflow，它能在不改变节点角色的前提下使这一边界变得弹性。每个解码节点发布一个短期、可撤销的租约……

    arXiv:2609.27085v1 Announce Type: cross  Abstract: As serving capacity demand surpasses that of training, serving efficiency becomes increasingly important. Prefill-decode (P/D) disaggregation improves serving efficiency through specialization and isolation of the two phases. These benefits rest on a static partitioning. Phase demand, however, is not static. We observe that in a large LLM fleet the ratio of uncached input to output tokens has peak-to-mean ratios up to 4.7x at minute timescales, and that in a public agentic trace the hourly ratio spans a median 24.5x within a single day, while reassigning a replica takes tens of minutes. Agentic traffic sharpens the mismatch. Sizing each pool at its ninety-fifth percentile leaves up to 17% of cluster capacity unused; sizing below it converts the same imbalance into queueing and unrealized throughput. We present Crossflow, which makes this boundary elastic without changing node roles. Each decode node publishes a short-lived, revocable l
    
[^154]: 高斯分布已然足够：微调大行为模型时，流匹配先验并无帮助

    The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models

    [https://arxiv.org/abs/2609.27070](https://arxiv.org/abs/2609.27070)

    本文通过超过10万次仿真测试和1250次真机实验发现，在微调预训练大行为模型（如 π0.5、GR00T N1.5）时，流匹配策略的先验分布选择并不重要——标准高斯先验已然足够，从头训练时非高斯先验带来的收益无法迁移到微调场景。

    

    现代机器人模仿学习越来越依赖于基于扩散模型或流匹配模型的生成式策略，这类策略通过变换来自先验分布的样本来生成动作。一个关键问题是：先验的选择是否重要？已有研究表明，在从头训练时，用更接近目标分布的非高斯先验替换标准高斯先验能够显著提升性能。一个自然的后续问题是：这些收益能否迁移到微调预训练的大行为模型上，例如 LBM 1.0、π0.5 和 GR00T N1.5——在这种场景下人们或许期待获得更大的收益。令人惊讶的是，我们发现事实并非如此，除非在极低的微调数据比例下。通过在两个仿真平台上对上述三个大行为模型在 40 多个任务上进行的超过 10 万次仿真测试，以及在五个双臂操作任务上进行的 1250 次真机测试，我们证明那些明显更接近目标分布的非高斯先验……

    arXiv:2609.27070v1 Announce Type: cross  Abstract: Modern robot imitation learning increasingly relies on generative policies based on diffusion or flow-matching models, which generate actions by transforming samples from a prior distribution. A key question is whether the choice of prior matters. Replacing the standard Gaussian with a closer-to-target, non-Gaussian prior has been shown to substantially improve performance when training from scratch. A natural next step is to ask whether these gains transfer to fine-tuning pretrained Large Behavior Models (LBMs) such as LBM 1.0, $\pi_{0.5}$, and GR00T~N1.5, where one might expect even larger gains. Surprisingly, we find that this is not the case, except possibly at very low fine-tuning data fractions. Across over 100K simulation rollouts spanning all three aforementioned LBMs on 40+ tasks in two simulation platforms, and 1250 hardware rollouts on five bimanual manipulation tasks, non-Gaussian priors that are demonstrably closer to the 
    
[^155]: 提议而非裁判：面向挖掘投资因子的LLM智能体的随时有效统计裁判

    Propose, Don't Judge: An Anytime-Valid Referee for LLM Agents That Mine Investment Factors

    [https://arxiv.org/abs/2609.27051](https://arxiv.org/abs/2609.27051)

    提出一种“受治理的自我进化”框架，让LLM智能体只负责提出投资因子，而由智能体无法操纵的冻结统计裁判通过仅基于提交后市场结果的打赌式评分来筛选因子，在任何提议策略和停止时间下都保证虚假发现可控，并将虚假因子准入数量减少5-11倍。

    

    语言模型智能体现在已经承担了量化因子研究的全部流程：它们提出投资因子、对因子进行回测、筛选幸存者并将其淘汰。我们要问的是，智能体应该保留哪些工作。我们的答案是“受治理的自我进化”：智能体可以提议，但必须由一个智能体无法干预的、被冻结的统计裁判来进行评判。该裁判仅根据提交之后才揭晓的市场结果，通过打赌的方式为每个候选因子打分，因此其虚假发现保证在任何提议策略下的任何停止时间都成立。我们在一个植入真实答案的合成世界、一个探针编写环境以及中证500指数的十年滚动回测中，将三个提议者（脚本、多臂老虎机和语言模型）分别与这个裁判以及三个故意存在信息泄漏的裁判进行交叉实验。由谁来裁判决定了虚假准入的数量：在脚本提议者下，冻结裁判接纳的低于阈值因子数量比泄漏裁判少5-11倍，且没有提议者能够……

    arXiv:2609.27051v1 Announce Type: new  Abstract: Language-model agents now run the whole of quantitative factor research: they propose investment factors, backtest them, select the survivors and retire them. We ask which of those jobs an agent should keep. Our answer is governed self-evolution: the agent may propose, and a frozen statistical referee that the agent cannot touch must judge. The referee scores each candidate only on market outcomes revealed after submission, by betting, so its false-discovery guarantee holds at every stopping time for any proposal policy. We cross three proposers (a script, a bandit and a language model) with this referee and with three deliberately leaky ones, in a synthetic world with planted truth, a probe-authoring environment and a ten-year walk-forward on the CSI 500. Who judges sets the number of false admissions: the frozen referee admits 5-11 times fewer sub-threshold factors than the leaky referees under a scripted proposer, and no proposer clos
    
[^156]: 大语言模型中的数学推理按解题方法而非主题组织

    Math Reasoning in LLMs is Organized by Approach, Not Topic

    [https://arxiv.org/abs/2609.27041](https://arxiv.org/abs/2609.27041)

    该论文通过生成-回放协议提取激活重要性签名并进行无监督聚类，证明大语言模型的内部数学推理是按可复用的解题方法而非数学主题来组织的。

    

    数学推理基准通常按主题进行组织，但语言模型可能是按照可复用的推理方法来组织其内部计算的。本文研究了开放数学能力大语言模型究竟是按主题子技能还是按推理方法来组织内部计算，我们提供的证据表明推理方法是关键因素。我们引入了一种生成-回放协议：模型首先生成一个解答，随后我们回放完全相同的提示加生成轨迹，并提取推理词元上的激活重要性签名。我们在八个模型和五个数学推理数据源上对这些签名进行无监督聚类，然后通过结构、语义和干预测试来评估恢复出的结构。在全部40个模型-数据源组合中，恢复出的聚类均优于同等规模的随机基线。两个独立的前沿大语言模型评审在77-82%的真实聚类中发现了方法层面的连贯性。

    arXiv:2609.27041v1 Announce Type: new  Abstract: Mathematical reasoning benchmarks are typically organized by topic, but language models may organize their internal computation by reusable reasoning approach instead. In this paper, we investigate whether open math-capable LLMs organize internally by topical sub-skill or by reasoning approach, and we present evidence that the approach is the key. We introduce a generation-replay protocol: a model first generates a solution, after which we replay the exact prompt-plus-generation trajectory and extract activation-importance signatures over the reasoning tokens. We cluster these signatures without supervision across eight models and five mathematical reasoning sources, then evaluate the recovered structure with structural, semantic, and intervention tests. Across all 40 model-source cells, the recovered clusters outperform matched-size random baselines. Two independent frontier-LLM judges find approach-level coherence in 77-82% of real clu
    
[^157]: EMA：跨GPU的弹性且性能透明的内存共享系统

    EMA: Elastic and Performance Transparent Memory Across GPUs

    [https://arxiv.org/abs/2609.27040](https://arxiv.org/abs/2609.27040)

    EMA提出了一种服务器内跨GPU的弹性内存共享系统，通过预取技术为借用方隐藏远程访问开销、同时保证出借方的内存可按需回收，使双方性能均不低于静态分区。

    

    多GPU服务器已成为现代数据中心的标准构建单元，通过高带宽互连提供聚合容量。与此同时，诸如大语言模型（LLM）推理等工作负载表现出高度动态的内存需求，这可能导致一个GPU耗尽其本地内存，而其他GPU却处于利用率不足的状态。这种不匹配促使我们提出了跨GPU弹性资源共享的模型。我们提出了EMA，一个内存共享系统，允许服务器内的GPU相互借用和回收内存，形成一个弹性的容量池。EMA为借用方和出借方都确保了性能透明性。对于借用方，预取技术隐藏了远程访问的开销，使应用程序感受到的远程内存和本地内存在性能上难以区分。对于出借方，被借用的资源仍可按需回收，保证性能永远不会低于静态分区的情况。虽然我们的设计聚焦于内存，但……

    arXiv:2609.27040v1 Announce Type: cross  Abstract: Multi-GPU servers have become the standard building block of modern data centers, providing aggregated capacity through high-bandwidth interconnects. At the same time, workloads such as LLM inference exhibit highly dynamic memory demands, which can cause one GPU to exhaust its local memory while others remain underutilized. This mismatch motivates a model of elastic resource sharing across GPUs.   We present EMA, a memory sharing system that allows GPUs within a server to borrow and reclaim memory from each other, forming an elastic pool of capacity. EMA ensures performance transparency for both borrowers and lenders. For borrowers, prefetching hides remote access costs so that applications experience remote and local memory as indistinguishable in performance. For lenders, borrowed resources remain reclaimable on demand, guaranteeing that performance never falls below that of static partitioning. While our design focuses on memory, th
    
[^158]: 陈述的推理步骤是否具有因果承重作用？

    Are Stated Reasoning Steps Causally Load-Bearing?

    [https://arxiv.org/abs/2609.27038](https://arxiv.org/abs/2609.27038)

    该论文提出一种在激活层面通过带有已知预测目标的激活补丁方法，以因果方式测量思维链忠实度，发现Qwen3-4B约76.9%的陈述推理步骤对最终答案具有因果承重作用。

    

    思维链监控假设模型写出的推理反映了直接产生其答案的计算过程。此前的忠实度指标主要是行为层面的，它们只是简单地编辑推理文本并观察由此产生的答案。然而，我们的方法旨在在激活层面以因果方式测量忠实度，特别是针对模型自生成的推理。与之前测量性能退化的因果审计不同，我们的干预带有已知的预测目标。通过这种方式，每个激活补丁都应该将答案切换到一个可通过构造推导出的特定反事实实体。具体而言，我们使用合成的多跳查找任务（2-6跳），在模型陈述每个中间步骤的token区间上，用来自反事实运行的相应激活对残差流进行补丁操作。对于Qwen3-4B，76.9% ± 2.8% 的陈述步骤在最敏感的中间层具有因果承重作用（CLB）……

    arXiv:2609.27038v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring assumes that the reasoning a model writes reflects the computation that directly produces its answer. Previous faithfulness metrics have been predominantly behavioral, as they simply edit the reasoning text and observe the resulting answer. However, our methodology aims to measure faithfulness causally at the activation level, specifically on self-generated reasoning. Unlike previous causal audits, which measure degradation, our interventions carry a known predicted target. In this way, each patch should switch the answer to a specific counterfactual entity derivable by construction. Specifically, we use synthetic multi-hop lookup tasks (2-6 hops). We patch the residual stream at the token span where the model states each intermediate step with the corresponding activations from a counterfactual run. For Qwen3-4B, 76.9% +/- 2.8% of stated steps are causally load-bearing (CLB) at the most responsive mid-n
    
[^159]: 使用可控合成对话训练智能语音助手唤醒系统

    Training Intelligent Voice Assistant Wakeup with Controllable Synthetic Conversations

    [https://arxiv.org/abs/2609.27037](https://arxiv.org/abs/2609.27037)

    本文提出一种在传统唤醒词检测基础上增加上下文触发检测的智能语音助手唤醒系统，并构建了62.3小时可控多说话人合成对话语料库用于训练，使助手在唤醒后能够通过推理区分用户命令与无关语音。

    

    唤醒词检测是虚拟助手的关键组成部分，是实现无缝用户交互的入口。本文介绍了一种新颖的唤醒系统，该系统在传统的直接关键词检测基础上扩展了上下文触发检测。在初始唤醒词激活后，系统利用推理能力来区分用户命令和无关语音，确保高效且具备上下文感知的交互。我们提出了一种数据生成架构，生成了一个包含62.3小时的可控多说话人对话语料库，其中涵盖直接调用、上下文后续对话和未指向系统的语音。实验结果证明了所提出方法在多样化合成对话场景中的有效性。我们公开了代码、数据集和训练好的模型，以促进可复现性以及智能助手技术的进一步发展。

    arXiv:2609.27037v1 Announce Type: new  Abstract: Wake word detection is a critical component of virtual assistants, serving as the gateway to seamless user interactions. This paper introduces a novel wake-up system that extends traditional direct keyword detection with contextual trigger detection. After an initial wake word activation, the system uses reasoning to distinguish between user commands and unrelated speech, ensuring efficient and context-aware engagement. We present a data generation architecture that produces a 62.3-hour corpus of controllable multi-speaker conversations containing direct invocations, contextual follow-ups, and non-addressed speech. Experimental results demonstrate the effectiveness of the proposed approach across diverse synthetic conversational scenarios. We release the code, dataset and trained models to promote reproducibility and further advancements in intelligent assistant technologies.
    
[^160]: 基于机器学习的聚合物性质预测开放基准

    An open benchmark for machine learning-based polymer property prediction

    [https://arxiv.org/abs/2609.27036](https://arxiv.org/abs/2609.27036)

    该论文推出了开放基准数据集PolyBench26，包含近25万个涵盖八种物理性质的聚合物数据点，支持四项机器学习评估任务，并发现基于图的模型在聚合物性质预测中表现最佳。

    

    聚合物性质预测领域缺乏开放的、标准化的基准数据集，无法对机器学习方法进行严格的比较，且现有资源仅覆盖聚合物结构中很小的一部分，例如均聚物。我们推出了Polymer Benchmark 2026（PolyBench26），这是一个包含近25万个聚合物性质数据点的开放数据集，涵盖八种物理性质，数据来源包括实验测量、密度泛函理论（DFT）和分子动力学模拟。该基准支持在均聚物以及交替、无规和嵌段共聚物上的四项评估任务：分布内性质预测、数据集规模扩展、重复单元复杂性以及向未知聚合物架构的迁移。我们比较了语言模型、基于图的方法和基于描述符的方法，发现基于图的模型在性质预测中误差最低，在所评估的各种训练集规模下保持其优势，并且……

    arXiv:2609.27036v1 Announce Type: cross  Abstract: Polymer property prediction lacks open, standardized benchmarks that enable rigorous comparison of machine-learning methods, with existing resources covering only a narrow fraction of polymer architectures, such as homopolymers. We introduce Polymer Benchmark 2026 (PolyBench26), an open dataset comprising nearly 250,000 polymer-property datapoints across eight physical properties, including data from experimental measurements, density functional theory, and molecular dynamics. The benchmark supports four evaluation tasks across homopolymers and alternating, random, and block copolymers: in-distribution property prediction, dataset-size scaling, repeat-unit complexity, and transfer to held-out polymer architectures. We compare language model, graph-based, and descriptor-based approaches and find graph-based models provide the lowest errors in property prediction, retain their advantage across the evaluated training-set sizes, and remain
    
[^161]: 基于分解子任务的强化学习

    Reinforcement Learning with Decomposed Subtasks

    [https://arxiv.org/abs/2609.27035](https://arxiv.org/abs/2609.27035)

    该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。

    

    组相对策略优化（GRPO）及用于训练语言模型智能体的相关策略梯度方法，在进入策略更新之前，会将整个多轮rollout压缩为单一标量轨迹奖励。当任务由不同技能组合而成时，尤其是在稀疏且延迟的环境反馈下，这种压缩是有损的：优化器必须隐式地推断是哪种能力导致了最终结果，以及这应当如何改变行为。我们认为正确的基元并非更好的标量，而是分解：轨迹奖励应当在进入策略更新之前沿着子任务进行拆分。我们提出了基于分解子任务的强化学习（RLDS），其核心是子任务分解优势估计（SDAE）：一种替代标量GRPO优势的方法，它在固定的分类体系上将轨迹奖励拆分为每个子任务的份额，为每个子任务计算组相对优势，并将每个token的信用分配……

    arXiv:2609.27035v1 Announce Type: new  Abstract: Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delayed environmental feedback, this collapsing is lossy: the optimizer must implicitly infer which competency drove the outcome and how that should change behavior. We argue the right primitive is not a better scalar but a decomposition: trajectory reward should be split along subtasks before it enters the policy update. We introduce Reinforcement Learning with Decomposed Subtasks (RLDS), whose core is Subtask-Decomposed Advantage Estimation (SDAE): a replacement for the scalar GRPO advantage that splits trajectory reward into per-subtask shares on a fixed taxonomy, computes a group-relative advantage per subtask, and distributes per-token credit b
    
[^162]: 损失函数选择还是模型选择？预测水平在加密货币波动率预测中的作用

    Loss Choice or Model Choice? The Role of Forecast Level in Cryptocurrency Volatility Forecasting

    [https://arxiv.org/abs/2609.27024](https://arxiv.org/abs/2609.27024)

    该研究通过对加密货币波动率预测中七种损失函数与五种模型的对比发现，损失函数选择的影响主要源于其所针对的预测水平差异，而对齐预测水平后，模型选择成为更重要的因素。

    

    波动率预测在金融风险管理中扮演着核心角色，因为其整体水平和逐日变动都会影响下游决策。大多数研究在保持训练损失固定的情况下比较预测模型。然而，不同的损失函数强调不同的误差，并可能针对未来波动率的不同特性，因此直接的比较可能将持续的预测水平差异与每日预测变动的差异混杂在一起。这就留下了一个未解决的问题：损失函数选择的重要性究竟主要来自其所针对的预测水平，还是来自水平调整后仍然存在的差异。我们通过对主要加密货币进行七种损失函数和五种模型的比较来填补这一空白。基于验证集的对齐方法在评估之前调整预测水平，然后使用统计评分和单日风险价值（Value-at-Risk）对原始预测和对齐后的预测进行评估。在对齐之前，损失函数之间的边际评分差异更大。在对齐之后……

    arXiv:2609.27024v1 Announce Type: cross  Abstract: Volatility forecasts play a central role in financial risk management because their overall level and day-to-day movements affect downstream decisions. Most studies compare forecasting models while keeping the training loss fixed. Yet losses emphasise different errors and can target different properties of future volatility, so raw comparisons may combine persistent forecast-level differences with differences in daily forecast movements. This leaves unresolved whether the importance of loss choice comes mainly from the forecast level it targets or from differences that remain after level adjustment. We address this gap through a comparison of seven losses and five models across major cryptocurrencies. Validation-based alignment adjusts the forecast level before the raw and aligned forecasts are evaluated using statistical scores and one-day Value-at-Risk. Before alignment, marginal score variation is greater across losses. After alignm
    
[^163]: 网络流量自然可见图表示中网络攻击类别的拓扑特征

    Topological Signatures of Cyber-Attack Classes in Natural Visibility Graph Representations of Network Traffic

    [https://arxiv.org/abs/2609.26990](https://arxiv.org/abs/2609.26990)

    本研究证明了不同网络攻击类别在网络流量的自然可见图表示中具有独特且可区分的拓扑特征，利用基于760个图论拓扑描述符的多分支CNN模型实现了96.20%的分类准确率。

    

    基于自然可见图（NVG）的表示方法为捕捉序列网络流量中的结构模式提供了一种有前景的途径。然而，不同网络攻击类别在此类表示中是否展现出独特的拓扑特征，目前仍缺乏充分的理解。本研究基于CSE-CIC-IDS2018数据集，考察了基于NVG的网络流量表示的判别能力与结构特性。研究将76个数值型流量特征在40个观测值构成的重叠帧内独立转换为NVG，并从每个图中提取十种图论度量指标，从而每帧获得760个拓扑描述符。这些表示的判别能力通过采用分层五折交叉验证的多分支卷积神经网络（CNN）进行评估。模型达到了96.20%的平均准确率和马修斯相关系数（原文摘要此处不完整）

    arXiv:2609.26990v1 Announce Type: cross  Abstract: Natural Visibility Graph (NVG)-based representations provide a promising approach for capturing structural patterns in sequential network traffic. However, whether different cyber-attack classes exhibit distinctive topological signatures in such representations remains insufficiently understood. This study investigates the discriminative and structural characteristics of NVG-based network traffic representations using the CSE-CIC-IDS2018 dataset. Seventy-six numerical traffic features were independently transformed into NVGs within overlapping frames of 40 observations, and ten graph-theoretic metrics were extracted from each graph, resulting in 760 topological descriptors per frame. The discriminative capability of these representations was evaluated using a multi-branch convolutional neural network (CNN) with stratified five-fold cross-validation. The model achieved an average accuracy of 96.20% and a Matthews correlation coefficient
    
[^164]: 相同证据，不同判断：视觉/语音与文本冲突中的证据不可交换性

    Same evidence, different judgments: Evidence noncommutative in vision/speech-text conflicts

    [https://arxiv.org/abs/2609.26986](https://arxiv.org/abs/2609.26986)

    本文通过仅交换证据位置的配对实验，揭示了多模态大语言模型中存在“跨模态证据不可交换性”——将图像或语音放在冲突文本之后会系统性地改变模型判断，并指出以往文本偏见研究因未控制证据顺序而可能得出误导性结论。

    

    对于多模态大语言模型，当图像或语音与随附文本发生冲突时，所测得的文本依赖性可能会将模态偏好与证据位置纠缠在一起。以往关于文本偏见的研究通常使用固定的证据顺序，或将任务指令随证据一起移动，导致顺序因素的作用不明确。在本文中，我们采用配对比较方法，保持指令和证据内容固定不变，仅交换两个信息来源的位置，以量化这种潜在影响。在视觉和语音模型中，将图像或录音放置在冲突文本之后，会一致地将模型的答案转向该感知证据的内容。我们还重新审视了先前的研究，并分析了其实验设置为何可能得出误导性结论。这些发现揭示了跨模态证据的不可交换性：当证据顺序改变时，相同的证据可能导致不同的判断，而将感知证据放在后面可以增强模型对其的采纳（原文此处截断）。

    arXiv:2609.26986v1 Announce Type: new  Abstract: For multimodal large language models, when images or speech conflict with accompanying text, measured text reliance can entangle modality preference with evidence position. Earlier studies of text bias often used a fixed evidence order or moved task instructions with the evidence, leaving the contribution of order unclear. In this paper, we use a paired comparison that keeps the instructions and evidence content fixed and swaps only the positions of the two sources to quantify this potential influence. Across vision and speech models, placing an image or recording after conflicting text consistently shifts answers toward its content. We also revisit previous studies and analyze why their experimental settings can lead to misleading conclusions. These findings reveal cross-modal evidence noncommutativity: the same evidence can lead to different judgments when its order changes, and placing perceptual evidence later can increase the model'
    
[^165]: 逃离Python依赖地狱：一种用于Python依赖解析的混合重放与修复流水线

    Escaping Python Dependency Hell: A Hybrid Replay-and-Repair Pipeline for Python Dependency Resolution

    [https://arxiv.org/abs/2609.26952](https://arxiv.org/abs/2609.26952)

    PLLM+通过优先使用低成本的确定性步骤（如历史成功配置重放和实时PyPI验证）、仅在必要时才回退到基于LLM的修复循环，将Python依赖解析的成功率从1,169提升至1,500个片段，同时将平均运行时间从368.7秒大幅降至71.8秒。

    

    Python生态系统中的依赖冲突源于不兼容的版本约束、缺失的软件包以及未记录的兼容性关系，导致许多真实世界的代码片段在执行时失败。本文提出了PLLM+，一种混合依赖修复流水线，并在包含2,891个依赖失败代码片段的HG2.9K基准上进行了评估。PLLM+在调用基于大语言模型（LLM）的修复之前，优先采用低成本的确定性步骤：基于静态AST的解释器推断、从竞赛提供的解决方案数据库中重放历史上成功的依赖配置，以及对候选软件包版本进行实时PyPI验证。当这些步骤无法解决某个案例时，系统会回退到结构化的基于LLM的修复循环，该循环包含类型化的错误分类以及提议者/批评者双智能体机制。在HG2.9K基准上，PLLM+解决了2,891个代码片段中的1,500个，而PLLM基线仅解决了1,169个。同时，它还将平均运行时间从368.7秒降低至71.8秒。

    arXiv:2609.26952v1 Announce Type: new  Abstract: Dependency conflicts in Python ecosystems arise from incompatible version constraints, missing packages, and undocumented compatibility relationships, causing many real-world code snippets to fail at execution. This paper presents PLLM+, a hybrid dependency-repair pipeline evaluated on the HG2.9K benchmark of 2,891 dependency-failing snippets. PLLM+ prioritizes inexpensive deterministic steps before invoking LLM-based repair: static AST-based interpreter inference, replay of historically successful dependency configurations from the competition-provided solutions database, and live PyPI validation of candidate package versions. When these steps do not resolve a case, the system falls back to a structured LLM-based repair loop with typed error classification and Proposer/Critic agents. On HG2.9K, PLLM+ solves 1,500 out of 2,891 snippets, compared with 1,169 solved by the PLLM baseline. It also reduces average runtime from 368.7 to 71.8 se
    
[^166]: 能识别却难生成：文化特定亲属称谓的生成基准测试

    Recognized but Not Produced: A Generation Benchmark for Culturally Specific Kinship Terms

    [https://arxiv.org/abs/2609.26942](https://arxiv.org/abs/2609.26942)

    该论文提出一个生成式基准测试，揭示大语言模型在印地语、泰米尔语和韩语的亲属称谓任务中“能识别却难生成”——选择题准确率远高于自由生成能力，表明多选题评估格式高估了模型对文化特定词汇知识的掌握。

    

    当前文献使用选择题基准评估大语言模型（LLM）的多语言亲属称谓理解能力，将其视为一个识别问题。我们转而提示五个开源权重LLM，在两种交流任务中用三种非西方语言（印地语、泰米尔语和韩语）生成亲属称谓，并与匹配的选项辅助选择基线进行对比。在相同的关系-语言单元格上，GPT OSS120B在75个有效单元格中有90.67%选择了正确称谓，但在相应的生成尝试中仅有36.00%产出可接受的称谓；Llama 3.370B也表现出相同模式（77.92%对24.24%）。由于四选项条件展示了候选称谓且不要求文字书写产出，这一差异被解释为评估格式差距，而非词库知识完好无损的直接证据。在明确指定的L3提示上，各模型准确率差异显著，从GLM-5.1的72.29%到Llama-3.370B的24.24%。

    arXiv:2609.26942v1 Announce Type: cross  Abstract: Current literature evaluates large language models (LLMs) on multilingual kinship understanding using multiple choice benchmarks, treating it as a recognition problem. We instead prompt five open weight LLMs to generate kinship terms in three non Western languages (Hindi, Tamil, and Korean) across two communicative tasks and pair this with a matched option-supported selection baseline. On identical relation language cells, GPT OSS120B selects the correct term in 90.67% of 75 valid cells but produces an accepted term in 36.00% of the corresponding attempts; Llama 3.370B shows the same pattern (77.92% versus 24.24%). Since the four-option condition displays the candidate terms and does not require script production, the difference is interpreted as an evaluation format gap rather than direct proof that lexical knowledge is intact. On explicitly specified L3 prompts, accuracy varies sharply, from GLM-5.1 at 72.29% to Llama-3.370B at 24.24
    
[^167]: 哪些目标需要调节旋钮？在可引导的多元化对齐中预测目标冲突并覆盖权衡

    Which Objectives Need a Dial? Predicting Objective Conflict and Covering Trade-offs in Steerable Pluralistic Alignment

    [https://arxiv.org/abs/2609.26929](https://arxiv.org/abs/2609.26929)

    该研究提出用两种预训练阶段的测量指标预测多元化对齐中目标间是对齐还是冲突，并发现选择最近训练模型和参数合并虽能扩展MODPO的权衡覆盖范围，但仍无法持续媲美直接训练。

    

    人们持有多样且有时相互冲突的价值观，因此没有一个单一的对齐模型能够满足所有人。因此，多元化对齐需要可引导的模型，能够以不同方式平衡相互竞争的目标。多目标直接偏好优化（MODPO）通过使用目标权重来跨越连续的权衡谱系来实现这一点。我们研究了两个问题：什么时候一个模型可以同时改进两个目标，以及如何在不为每个权衡点单独训练模型的情况下覆盖多个权衡？在来自HelpSteer和UltraFeedback的七个目标对上，两种预训练阶段的测量指标能够预测目标在人类标注数据上是对齐还是冲突，但在AI标注数据上则无法预测，因为响应长度和重复会混淆奖励模型的评分。为了实现更广泛的权衡覆盖，选择最近的已训练模型以及合并模型参数都有帮助，但两者都无法持续地媲美直接训练。这些发现为构建可引导的多元化对齐系统提供了实用指导。

    arXiv:2609.26929v1 Announce Type: new  Abstract: People hold diverse, sometimes conflicting values, so no single aligned model can satisfy everyone. Pluralistic alignment therefore calls for steerable models that can balance competing objectives differently. Multi-Objective Direct Preference Optimization (MODPO) does this by using an objective weight to span a continuum of trade-offs. We study two questions: when can one model improve two objectives simultaneously, and how can many trade-offs be covered without training a separate model for each? Across seven objective pairs from HelpSteer and UltraFeedback, two pre-training measurements predict whether objectives align or conflict for human-annotated data, but not for AI-annotated data, where response length and repetition confound reward-model scores. For broader trade-off coverage, selecting the nearest trained model and merging model parameters both help, but neither consistently matches direct training. These findings yield practi
    
[^168]: 构建用于交互式多智能体仿真的社会情感人工智能

    Building Socio-Affective Artificial Intelligence for Interactive Multi-Agent Simulations

    [https://arxiv.org/abs/2609.26927](https://arxiv.org/abs/2609.26927)

    本文提出了AGIMUD软件架构，将社会感知推理与情感融入智能体行为，为人类与多智能体在模拟动态世界中的交互提供了整合的设计原则。

    

    本文的目标是提供设计原则和软件架构，以实现人类与多个智能体在模拟动态世界中的交互。这将当前通用人工智能（AI/AGI）时代与基于Transformer的对话代理的普及以及计算能力的提升联系起来。通过对当前和以往多智能体心智理论（具备社会和情感感知能力的智能体）的概述，智能体之间以及智能体与人类之间交互的整合性设计，对于理解如何在未来人-智能体推理系统中实现可持续性和治理至关重要。本工作提出了一款名为“AGIMUD”的软件，它集成了：A. 智能体行为与交互中具备社会感知的推理和情感；B. 面向人类用户、人工智能体和模拟世界的人类多模态方案设计；C. 分布式AI处理。

    arXiv:2609.26927v1 Announce Type: new  Abstract: The objective of this article is to provide design principles and a software architecture for enabling interaction between humans and multiple agents in simulated dynamic worlds. This connects the current era of general artificial intelligence (AI/AGI) with the proliferation of transformer-based conversational agents and the increased computational capabilities. Given an overview of current and previous multi-agent theories of mind (socially and affectively-aware agents), the existence of an integrative design of agent interactions with themselves and with humans must be crucial for understanding how to create sustainable and governance in future human-agent reasoning systems. In this work is presented a software "AGIMUD" that integrates: A. socially-aware reasoning and emotion in agent behavior and interaction, B. a design of human multimodal scheme for human users, artificial agents and simulated worlds, and C. distributing the AI proc
    
[^169]: 专家在LLM分歧处显现：在大规模标注的LLM码本修订中利用跨模型分歧精准定位专家投入

    Experts Rise Where LLMs Disagree: Using Cross-Model Disagreement to Target Expert Effort in LLM Codebook Revision for Large-Scale Annotation

    [https://arxiv.org/abs/2609.26926](https://arxiv.org/abs/2609.26926)

    该论文提出利用多个大语言模型之间的分歧来定位最需要专家反馈的案例，并通过对比三种反馈方式发现，让专家对分歧案例进行附带理由的标注能最有效地指导LLM码本修订，使LLM标注准确率（64.9%）甚至超过专家手工修订的码本（57.8%）。

    

    大规模文本标注通过AI标注者遵循的码本，将专家洞见带给数百万份文档。然而，开发一个稳健的码本需要数月时间。大语言模型（LLM）可以加速这一过程：将早期码本应用于数据，找出LLM之间存在强烈分歧的案例，并引导专家针对这些案例提供反馈。我们考察了专家为LLM码本修订提供反馈的三种方式：(i) 编辑由跨LLM分歧驱动的LLM生成的修订（码本验证），(ii) 回答关于LLM分歧的问题（问答），(iii) 对分歧案例进行附带理由的标注（理由标注）。在数千份辅导课程转录文本上的实验表明，理由标注方式获得了最高的LLM标注准确率（相对于专家标注为64.9%），优于专家修订的码本（57.8%），最佳的问答设置表现也优于……

    arXiv:2609.26926v1 Announce Type: cross  Abstract: Large-scale text annotation brings expert insight to millions of documents, often through a codebook that AI annotators follow. Developing a robust codebook, however, takes months. Large language models (LLMs) could speed this process by applying an early codebook to the data, surfacing cases with strong LLM disagreement, and eliciting expert feedback to address them. We examined three ways experts can provide feedback for LLM codebook revision: (i) editing LLM-generated revisions driven by cross-LLM disagreement (Codebook Verifying), (ii) answering questions about LLM disagreements (Question Answering), and (iii) labeling disagreement cases with rationales (Rationale Labeling). Experiments on thousands of tutoring-session transcripts show that Rationale Labeling yielded the highest LLM-labeling accuracy (64.9%) against expert labels, outperforming the expert-revised codebook (57.8%). The best Question Answering setting also outperform
    
[^170]: 一种基于三维姿态的集成框架用于板球击球动作分类与自动化生物力学分析

    A 3D Pose-Based Ensemble Framework for Cricket Shot Classification and Automated Biomechanical Analysis

    [https://arxiv.org/abs/2609.26923](https://arxiv.org/abs/2609.26923)

    本文提出一个基于三维姿态数据的深度学习集成框架，利用YOLO提取击球手、MeTRAbs提取30个身体关键点的骨骼姿态序列，实现板球击球动作的自动分类与生物力学分析，克服了传统RGB视频方法易受环境干扰且无法捕捉生物力学特征的缺陷。

    

    板球是世界上最受喜爱的运动之一，技术进步已深深融入现代比赛的分析与教练指导之中。板球击球动作分类和自动化表现分析为这一趋势增添了新的维度。传统方法依赖于RGB视频特征或静态图像，这些方法对相机角度、光照和背景干扰等环境变化十分敏感，且往往无法捕捉击球动作的底层生物力学特征。在本文中，我们提出了一种改进板球教练指导的系统，该系统接收原始视频数据，使用YOLO从视频帧中提取击球手，并使用MeTRAbs从视频帧中提取三维姿态数据。该系统生成包含30个身体点的序列骨骼姿态数据，并捕捉击球手的生物力学特征。作为系统的一部分，我们还提出了一个深度学习集成模型，用于对四种击球动作进行分类：flick（撩击）、pull（拉击）、defen（防守）等。

    arXiv:2609.26923v1 Announce Type: cross  Abstract: Cricket is one of the most celebrated sports world-wide, and technological advancement has become deeply embedded in how the modern game is analyzed and coached. Cricket shot classification and automated performance analysis add a further dimension to this trend. Traditional approaches rely on RGB video features or static images, which are sensitive to environmental variations such as camera angle, lighting, and background clutter, and often fail to capture the underlying biomechanics of batting actions. In this paper, we propose a system to improve cricket coaching that takes raw video data, extracts batsmen from video frames using YOLO, and extracts 3D pose data from video frames using MeTRAbs. The system produces sequential skeletal pose data of 30 body points and captures the biomechanical features of a batsman. As part of the system, we also propose a deep learning ensemble for shot classification of four shots: flick, pull, defen
    
[^171]: 基于组织病理学与CT的跨模态对比学习用于肾细胞癌自动分级

    Cross-Modal Contrastive Learning from Histopathology and CT for Automated Renal Cell Carcinoma Grading

    [https://arxiv.org/abs/2609.26920](https://arxiv.org/abs/2609.26920)

    RCC-Align通过跨模态对比学习将组织病理学显微形态中的分级判别信息迁移到CT影像表征，实现了无需侵入性组织采样的透明细胞肾细胞癌无创自动分级。

    

    背景：透明细胞肾细胞癌（ccRCC）表现出显著的临床异质性，准确的分级评估对于风险分层和治疗规划至关重要。然而，传统的分级方法需要进行侵入性组织采样。我们开发了RCC-Align，这是一个跨模态对比学习框架，在训练阶段利用配对的组织病理学和计算机断层扫描（CT）数据，以改进基于CT的无创ccRCC分级预测。方法：RCC-Align通过对比跨模态目标函数对齐配对的全切片组织病理学图像（WSI）与CT扫描，将微观组织形态学中的分级判别信息迁移到宏观放射学表征中。该框架在配对的TCGA和CPTAC队列上进行训练和评估，采用患者级别的五折交叉验证。针对低级别与高级别ccRCC分类的性能与仅使用CT的基线方法（DINOv2-B……（摘要不完整，后续内容缺失）

    arXiv:2609.26920v1 Announce Type: cross  Abstract: Background: Clear cell renal cell carcinoma (ccRCC) exhibits substantial clinical heterogeneity, and accurate grade assessment is essential for risk stratification and treatment planning. However, conventional grading requires invasive tissue sampling. We developed RCC-Align, a cross-modal contrastive learning framework that leverages paired histopathology and computed tomography (CT) data during training to improve noninvasive CT-based ccRCC grade prediction. Methods: RCC-Align aligns paired whole-slide histopathology images (WSIs) and CT scans through contrastive cross-modal objectives, transferring grade-discriminative information from microscopic tissue morphology to macroscopic radiologic representations. The framework was trained and evaluated on paired TCGA and CPTAC cohorts using patient-level five-fold cross-validation. Performance for low- versus high-grade ccRCC classification was compared against CT-only baselines (DINOv2-B
    
[^172]: 多目标强化学习中事后重标注导致的偏好覆盖坍缩研究

    On Preference Coverage Collapse from Hindsight Relabeling in Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2609.26918](https://arxiv.org/abs/2609.26918)

    该研究发现，在偏好条件化多目标强化学习中，用智能体实际实现的偏好方向进行事后重标注往往有害——它使36个算法-环境设置中的19个性能下降多达四个标准差，其根源是重复重标注导致的偏好覆盖坍缩，而非重标注噪声。

    

    事后重标注——即追溯性地将一条转移的目标替换为智能体实际取得的结果——是提升强化学习（RL）样本效率的有效工具。对于偏好条件化的多目标强化学习（MORL），一个自然的扩展是用智能体实际实现的偏好方向（而非所要求的偏好方向）来重标注转移。我们证明这种扩展常常是有害的：在连续控制MO-Gymnasium基准套件上，跨越两种评论家网络骨干和两种偏好采样方案的四种偏好条件离线策略算法中，36个“算法-环境”组合设置里有19个性能下降多达四个标准差，仅有一个获得改善，其余则不受影响。这种损害并非由重标注噪声所致：对目标进行去噪几乎无法恢复性能，优先级采样以及任何缓冲区结构上的选择也都无法重现这一现象。相反，重复的重标注会导致偏好覆盖的坍缩。

    arXiv:2609.26918v1 Announce Type: cross  Abstract: Hindsight relabeling which retroactively replacing a transition's goal with the outcome the agent actually achieved is an effective tool for improving sample-efficiency in Reinforcement Learning (RL). A natural extension to preference-conditioned multi-objective RL (MORL) relabels transitions with the preference direction the agent achieved rather than the one asked for. We show that this extension is frequently harmful: across four preference-conditioned off-policy algorithms spanning two critic backbones and two preference-sampling schemes on the continuous-control MO-Gymnasium suite, it degrades 19 of 36 algorithm-environment settings by as much as four standard deviations, improves only one, and leaves the rest unaffected.   The harm is not a symptom of noisy relabels; denoising the target recovers almost nothing, and neither prioritized sampling nor any buffer-structural choice reproduces it. Instead, repeated relabeling collapses
    
[^173]: COMED：多LLM推理中路由与协作之间缺失的中间方案

    COMED: The Missing Middle Between Routing and Collaboration in Multi-LLM Inference

    [https://arxiv.org/abs/2609.26913](https://arxiv.org/abs/2609.26913)

    COMED提出了一个锚点后控制器，利用锚点自一致性、路由器边际和轻量级同伴探针实现选择性跨模型协作，仅在协作可能有益时才升级模型，在路由与密集协作之间找到了缺失的中间方案。

    

    没有任何单一的大语言模型（LLM）能够在所有查询上都保持可靠，这促使了多模型推理系统的出现，这类系统要么在模型之间进行路由，要么组合多个模型的输出。然而，路由在选定初始模型后就停止了，而密集协作则会对每个查询都调用同伴模型。我们证明了协作是非单调的：同伴模型可以恢复没有任何模型能单独解决的失败，但也可能破坏最初正确的答案。我们提出了COMED（面向多LLM审议的受控模型升级），这是一种用于选择性跨模型协作的锚点后控制器。COMED利用锚点自一致性、路由器边际以及一个轻量级的同伴探针来接受有把握的答案、验证模糊的情况，并且只在协作可能带来收益时才进行升级。我们通过救援-损害分解对这一权衡进行了形式化，表明当被救援的错误多于协作引发的损害时，选择性协作能够带来提升。在医学等领域……（摘要原文截断）

    arXiv:2609.26913v1 Announce Type: cross  Abstract: No single Large Language Model (LLM) is uniformly reliable across queries, motivating multi-model inference systems that either route among models or combine their outputs. However, routing stops after selecting an initial model, while dense collaboration invokes peers on every query. We show that collaboration is non-monotonic: peers can recover failures that no model solves alone, but can also corrupt initially correct answers. We introduce COMED (Controlled Model Escalation for Multi-LLM Deliberation), a post-anchor controller for selective cross-model collaboration. COMED uses anchor self-consistency, router margin, and a lightweight peer probe to accept confident answers, verify ambiguous cases, and escalate only when collaboration is likely beneficial. We formalize this trade-off with a rescue-harm decomposition showing that selective collaboration improves when rescued errors outweigh collaboration-induced harms. Across medical,
    
[^174]: TwinCheck：面向有状态工具代理的证据支撑型“负孪生”验证

    TwinCheck: Evidence-Grounded Negative-Twin Verification for Stateful Tool Agents

    [https://arxiv.org/abs/2609.26911](https://arxiv.org/abs/2609.26911)

    TwinCheck提出了一种推理时验证策略，通过构建基于证据的“负孪生”反事实替代方案，仅在满足证据条件、通过结构检查并在顺序无关的成对验证中胜出时才替换智能体的工具调用，从而在不引入新失败的前提下提升有状态工具代理的多轮任务成功率。

    

    arXiv:2609.26911v1 公告类型：新论文 摘要：单个在局部看似合理的工具调用，可能会让原本成功的智能体轨迹偏离正轨。然而，仅凭怀疑并不足以成为干预的理由，因为替换本身反而可能引入验证本欲防止的失败。我们提出TwinCheck，一种推理时验证策略，仅当轨迹满足与“轨迹局部故障假设”相关联的证据条件时，才考虑进行替换。该方法会构建一个基于轨迹的反事实替代方案——即“负孪生”（negative twin），并且仅当该孪生通过结构检查、且成对验证器在两种候选顺序下均更偏好它时，才替换智能体的原始提议。在配对评估中，精确重放（exact replay）将智能体已解析的响应与动作保持固定，直至第一次被接受的替换，从而将干预效应与重采样效应分离开来。在对159个具备完整精确重放对的多轮BFCL V4任务的主要分析中，完整策略提升了GPT-5.6 So……（原文摘要至此截断）

    arXiv:2609.26911v1 Announce Type: new  Abstract: A single locally plausible tool call can derail an otherwise successful agent trajectory. Suspicion alone does not justify intervention, because the replacement itself can introduce the very failure verification is meant to prevent. We introduce TwinCheck, an inference-time verification policy that considers replacement only when the trace satisfies an evidence condition tied to a trace-local failure hypothesis. It constructs a trace-grounded counterfactual alternative, a negative twin, and replaces the agent's proposal only if the twin passes structural checks and the pairwise verifier prefers it in both candidate orders. For paired evaluation, exact replay holds the agent's parsed responses and actions fixed until the first accepted replacement, separating intervention effects from resampling. In the primary analysis of 159 multi-turn BFCL V4 tasks with complete exact-replay pairs, the complete policy raises task success for GPT-5.6 So
    
[^175]: Ajar：度量智能体防御中的开放特权

    Ajar: Measuring Open Privilege in Agent Defenses

    [https://arxiv.org/abs/2609.26900](https://arxiv.org/abs/2609.26900)

    该论文提出Ajar，通过附加到现有智能体安全基准并复用其任务、工具模式、参考解决方案和目标状态，直接度量防御方案中任务并不需要却仍保持开放的特权。

    

    语言模型智能体通过它所被赋予的工具来执行操作，而它在执行任务过程中读取的数据可以改变它对这些工具的使用方式。因此，越来越多的智能体安全执行技术被置于智能体与其工具之间，旨在在这一边界上实施访问控制、信息流控制或隔离。目前，这些技术是在围绕间接提示注入构建的智能体安全基准上进行评估的。这类基准通过防御在多大程度上降低成功攻击的数量、同时保持智能体的实用性来评判防御效果。防御仅依据智能体的执行情况来评判。即使一个防御方案保持着某项任务并不需要的传输、删除或大范围读取权限，它仍可能在这两项指标上获得高分。Ajar 利用现有基准直接度量这种开放特权。它附加到已有的智能体安全基准上，复用其中的任务、工具模式、参考解决方案和目标状态……

    arXiv:2609.26900v1 Announce Type: cross  Abstract: A language model agent acts through the tools it is given. The data it reads while working on a task can redirect what it does with those tools. A growing set of techniques for safe and secure agent execution therefore sits between the agent and its tools, aiming to enforce access control, information flow or isolation at that boundary. Today these techniques are evaluated on agent-security benchmarks built around indirect prompt injection. Those benchmarks judge a defense by how far it brings the number of successful attacks down while preserving the agent's utility. A defense is judged only on the agent's execution. It can score well on both metrics while holding open a transfer, a deletion or a broad read that no task needed. Ajar measures that open privilege directly using the existing benchmarks. It attaches to an agent-security benchmark that already exists and reuses the tasks, tool schemas, reference solutions and goal states t
    
[^176]: 脚手架即语言：一个具有最大表达能力的极简智能体框架

    Harness as a Language: A Minimalist Agent Framework With Maximal Expressivity

    [https://arxiv.org/abs/2609.26891](https://arxiv.org/abs/2609.26891)

    JAZ 框架证明了一个仅由单一可递归调用的 invoke 原语构成的极简智能体循环脚手架，就能完成通常需要记忆系统、自我改进系统等专门工程设计才能实现的任务，达到最大的表达能力。

    

    现代语言模型智能体围绕“智能体循环”（agent loop）构建：大语言模型被置于一个暴露出一组工具的环境中，通过交替进行工具调用和观察其输出，全权掌控整个工作流程。然而，某些工作流目前需要在智能体循环之外进行额外的工程设计，例如记忆系统和自我改进系统。我们构建了一个大语言模型智能体框架 JAZ，以探究一个仅比智能体循环本身多不了多少的极简脚手架，能在多大程度上完成这些专门系统所针对的任务。JAZ 仅暴露一个基于大语言模型的原语 invoke，并提供一组内置钩子，使程序员能够施加约束和监控。通过对现有代码模式智能体循环的推广，invoke 是满足两个决定性属性的最简循环：（1）大语言模型可以编写任意可执行代码，其中可以包含递归的 invoke……

    arXiv:2609.26891v1 Announce Type: new  Abstract: Modern language-model agents are built around the \textit{agent loop}, where the LLM is placed in an environment exposing a set of tools, and the LLM has full control over the workflow by alternating between tool calls and observing their output. However, certain workflows currently require additional engineering beyond the agent loop itself, such as memory systems and self-improving systems. We built an LLM agent framework, JAZ, to explore the extent to which a minimal harness that is little more than the agent loop itself can accomplish tasks these specialized systems are built for. JAZ exposes a single LLM-based primitive invoke and provides a set of built-in hooks that allow the programmer to apply constraints and monitoring. Generalizing existing code-mode agent loops, \texttt{invoke} is the simplest loop that satisfies two defining properties: (1) the LLM can write arbitrary executable code that can include recursive \texttt{invoke
    
[^177]: 安全提示：面向用户的实时AI风险感知干预措施

    Safety Nudges: User-Facing Interventions for Real-Time AI Risk Awareness

    [https://arxiv.org/abs/2609.26865](https://arxiv.org/abs/2609.26865)

    该研究提出了Safety Nudges——一款基于浏览器的工具，能在聊天机器人对话中实时检测并提示潜在的AI安全风险，实地研究表明此类面向用户的干预措施能有效提升用户对AI危害的意识，可作为模型层面安全防护的有益补充。

    

    对话式AI系统可能对其用户构成安全风险，例如幻觉、谄媚、过度自信和拟人化，但这些风险在用户日常使用中难以察觉。我们介绍了Safety Nudges，这是一种基于浏览器的工具，当检测到聊天机器人对话中出现令人担忧的行为时，它会提供轻量级的即时标记。我们通过一项为期两周的实地研究对Safety Nudges进行了评估，该研究涉及45名频繁使用聊天机器人的用户，收集了交互日志、调查问卷以及用户对各条提示的反馈。参与者认为该工具有用、清晰且干扰性最小，几乎所有用户都报告称对潜在AI危害的意识有所提高，尽管我们发现仅凭这种意识提升并不一定能带来可观察到的行为改变。我们的结果表明，面向用户的安全提示可以通过帮助人们在具体情境中批判性地评估AI回应，来补充模型层面的安全防护措施，同时强调了……

    arXiv:2609.26865v1 Announce Type: cross  Abstract: Conversational AI systems can pose safety risks to their users such as hallucination, sycophancy, overconfidence, and anthropomorphism, but these risks are difficult for users to detect during everyday use. We introduce Safety Nudges, a browser-based tool that provides lightweight, in situ flags when concerning behavior is detected in chatbot conversations. We evaluated Safety Nudges in a two-week field study with 45 frequent chatbot users, collecting interaction logs, surveys, and feedback on individual nudges. Participants found the tool useful, clear, and minimally disruptive, with nearly all users reporting an increased awareness of potential AI harms, though we found that this improved awareness alone did not necessarily lead to discernible behavioral changes. Our results suggest that user facing safety nudges can complement model-level safeguards by helping people critically evaluate AI responses in context, while highlighting th
    
[^178]: 用于HTTP请求异常检测的静态嵌入模型的比较评估

    Comparative Evaluation of Static Embedding Models for HTTP Request Anomaly Detection

    [https://arxiv.org/abs/2609.26860](https://arxiv.org/abs/2609.26860)

    本文提出HEDA模块化检测架构，在统一的单类分类框架下对Word2Vec、FastText和Doc2Vec三种静态嵌入模型进行基准评估，实现了仅用良性流量训练的无监督HTTP请求级异常检测。

    

    Web应用程序日益成为网络攻击的目标，这些攻击利用HTTP请求来规避安全机制。传统的Web应用防火墙（WAF）依赖于基于规则的方法，往往表现出较高的误报率和有限的适应性。近期的研究已开始探索机器学习技术和词嵌入模型，以改进HTTP流量中的异常检测。本文在一个统一的单类分类框架内，对静态嵌入模型（特别是Word2Vec、FastText和Doc2Vec）进行了基准评估。我们提出了HEDA（基于HTTP嵌入的检测架构），这是一种模块化的检测流水线，将静态嵌入表示与单类异常检测模型相结合，以在请求级别检测异常。该方法在无监督环境中运行，其中嵌入模型和检测器均仅在良性HTTP流量上进行训练。所提出的方法论……

    arXiv:2609.26860v1 Announce Type: cross  Abstract: Web applications are increasingly targeted by cyberattacks that exploit HTTP requests to evade security mechanisms. Traditional web application firewalls (WAFs) rely on rule-based approaches that often exhibit high false positive rates and limited adaptability. Recent studies have explored machine learning techniques and word embedding models to improve anomaly detection in HTTP traffic. This paper presents a benchmark for static embedding models, specifically Word2Vec, FastText, and Doc2Vec, within a unified, single-class classification framework. We propose HEDA (HTTP Embedding-Based Detection Architecture), a modular detection pipeline that combines static embedding representations with single-class anomaly detection models to detect anomalies at the request level. The approach operates in an unsupervised environment, where both the embedding models and detectors are trained exclusively on benign HTTP traffic. The proposed methodolo
    
[^179]: FLINT：面向可通行性的快速轻量级推理

    FLINT: Fast Lightweight Inference for Traversability

    [https://arxiv.org/abs/2609.26857](https://arxiv.org/abs/2609.26857)

    FLINT是一个仅有2160万参数的轻量级可通行性估计器，仅使用单个RGB相机即可在CPU上以14.7 FPS运行，并能比参数量高出38倍的基础模型系统生成更精确、成本更低的代价地图。

    

    野外环境下的导航因缺乏结构而充满挑战——对于什么是“可通行”并没有固定的定义。可通行性同时取决于环境以及机器人本体的动力学特性，而这两个变量都无法通过人工进行大规模标注。因此，可通行性必须由机器人通过自身经验来学习。现代平台倾向于使用多种传感器（RGBD相机、激光雷达、雷达、IMU）来估计可通行性并导航，并依赖计算密集型平台来运行神经网络推理。与这一趋势相反，我们提出了FLINT，一个轻量级的可通行性估计器：其骨干网络仅有2160万参数，比同类基础模型骨干网络小38倍，在留出的地形测试样本上得分更高，并且仅以RGB相机作为唯一传感器，就能在单独的CPU上以14.7 FPS的速度运行。尽管存在如此规模的差距，FLINT仍能比已部署的基础模型系统生成成本更低、精度更高的代价地图。

    arXiv:2609.26857v1 Announce Type: cross  Abstract: Navigation in off-road conditions is challenging due to the lack of structure. There is no fixed vocabulary for what is traversable. The traversability depends on both the environment and the embodiment's dynamics. Neither of these two variables can be hand-labeled at scale. Thus, traversability has to be learned by the embodiment's own experience. Modern platforms tend to use multiple sensors to estimate traversability and navigate: RGBD cameras, lidar, radar, IMU, with computationally intensive platforms to run inference on neural networks. Against this trend, we propose FLINT, a lightweight traversability estimator: a 21.6M-parameter backbone, 38\times smaller than a comparable foundation-model backbone, that scores higher on held-out terrain probes and runs at 14.7 FPS on CPU alone using a RGB camera has the only sensor. Despite that gap in scale, FLINT produces a cheaper, more accurate costmap than a deployed foundation-model syst
    
[^180]: QUARTET：基于四分支交叉注意力与随机游走轨迹的关系图Transformer增强方法

    QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

    [https://arxiv.org/abs/2609.26855](https://arxiv.org/abs/2609.26855)

    提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。

    

    arXiv:2609.26855v1 公告类型：交叉 摘要：关系深度学习将多表数据库建模为异构时序图，图Transformer目前在RelBench等基准测试上取得了最先进的性能。然而，当前领先的模型RelGT存在两个关键局限：其随机局部采样器生成的子图连接松散，阻碍了消息传递；其全局注意力模块依赖于单一的、基于种子特征的内存，忽略了更广泛的宏观层面动态。为克服这些局限，我们提出了QUARTET，一种表达能力强的图Transformer架构，它在局部子图上应用完全自注意力，同时通过交叉注意力分支来丰富全局上下文。具体而言，QUARTET采用基于近期截断个性化PageRank（PPR）的因果随机游走（CRW）采样器，以提取紧凑、抗枢纽节点干扰且密集连通的局部子图，且不会产生时序信息泄露。与此同时，四分支交叉注意力（摘要在此处截断）

    arXiv:2609.26855v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-atte
    
[^181]: SsgCaps：一个用于声音场景生成算法评估的受控数据集

    SsgCaps: A controlled dataset for the evaluation of sound scene generation algorithms

    [https://arxiv.org/abs/2609.26854](https://arxiv.org/abs/2609.26854)

    提出了SsgCaps数据集——一个完全基于公共领域音频、由结构化提示词驱动的人工设计声音场景数据集，可用于声音场景生成算法的公开评估。

    

    声音场景生成是指自动合成人工声音场景。我们介绍了SsgCaps，这是一个公开可用的、由人工设计的声音场景数据集，其中每个场景都与一个精确结构化的提示词相匹配，该提示词用于引导采样过程。相应的提示词从一个预定义的基于动作的类型学中采样，这使得可以在保持合理性的同时进行大量采样。SsgCaps是一个声音场景数据集，源自2024年DCASE挑战赛第7项任务未公开发布的参考数据集，该数据集包含私有领域和公共领域的音频样本。相比之下，SsgCaps仅包含公共领域的音频样本，这使我们能够将该数据集向社区开放。为了使该数据集对社区有用，我们首先详细阐述了提示词和数据集结构的设计原理。然后，我们对数据集的两个版本进行了对比定量分析。为此，我们将两个版本与音频进行了比较……

    arXiv:2609.26854v1 Announce Type: cross  Abstract: Sound Scene Generation is about the automatic synthesis of artificial sound scenes. We introduce SsgCaps, a publicly available dataset of human-engineered sound scenes wherein each scene matches a precisely structured prompt that guides the sampling process. The corresponding prompts are sampled from a predefined action-based typology that allows extensive sampling while retaining plausibility. SsgCaps is a sound scene dataset derived from the unpublished reference dataset for Task 7 of the 2024 DCASE Challenge edition, which contained private-and public-domain audio samples. In contrast, SsgCaps contains only public-domain audio samples, allowing us to open this dataset to the community. To make this dataset useful to the community, we first elaborate on the rationale for the prompt and dataset structure. We then perform a comparative quantitative analysis of the 2 versions of the dataset. To do so, we compare both versions to the aud
    
[^182]: COPE：基于用户嵌入与自我评估的稀疏用户反馈下大语言模型持续个性化

    COPE: Continual Personalization of LLMs under Sparse User Feedback via User Embeddings and Self-Evaluation

    [https://arxiv.org/abs/2609.26853](https://arxiv.org/abs/2609.26853)

    COPE提出了一种在稀疏用户反馈下实现大语言模型持续个性化的优化框架，通过为每个用户分配可学习的个性化嵌入，并在单次更新步骤中协同完成偏好捕获、自我评估校准与个性化响应优化。

    

    尽管大型语言模型（LLM）在各种基准测试中取得了显著成果，但它们与规范价值观的对齐往往导致同质化的响应，无法满足多样化的用户偏好。现有的免训练方法通常通过提示工程占用宝贵的上下文窗口，而基于训练的方法在训练后通常保持静态，无法支持现实场景中所需的持续优化。为应对这些挑战，我们提出了COPE（基于个性化嵌入与自我评估的持续优化），这是一个专为具有稀疏用户反馈的现实交互场景量身定制的新型优化框架。我们的框架为每个用户分配可学习的个性化嵌入，并在单个更新步骤内协同整合偏好捕获、自我评估校准和个性化响应优化。我们方法的一个关键创新在于……

    arXiv:2609.26853v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) have achieved remarkable results across various benchmarks, their alignment with normative values often results in homogenized responses that fail to address diverse user preferences. Existing training-free methods often occupy valuable context windows through prompt engineering, while training-based methods typically remain static post-training, failing to support the continual optimization required in real-world settings. To address these challenges, we propose COPE (Continual Optimization with Personalized embedding and self-Evaluation), a novel optimization framework tailored for real-world-motivated interaction settings with sparse user feedback. Our framework assigns learnable personalized embeddings to each user and synergistically integrates preference capture, self-evaluation calibration, and personalized response optimization within a single update step. A key innovation of our method is the
    
[^183]: 一种面向术中早期急性肾损伤预测的防泄漏多模态评估框架

    A Leakage-Aware Multimodal Evaluation Framework for Early Intraoperative Acute Kidney Injury Prediction

    [https://arxiv.org/abs/2609.26848](https://arxiv.org/abs/2609.26848)

    该论文提出了仅基于生理波形的混合时序骨干网络SynerT及其多模态扩展SynerT-MM和防泄漏堆叠集成SynerTStack，并在VitalDB数据库上以严格的防泄漏评估框架实现了术中早期急性肾损伤风险预测。

    

    大型非心脏手术后的术后急性肾损伤（AKI）具有相当高的发病率，然而术中早期风险分层仍然十分困难。在这项回顾性队列研究中，我们提出了SynerT——一种仅使用生理波形的混合时序骨干网络，它将因果扩张时序卷积网络（TCN）与多层扩张循环层相结合，用于编码术中早期生理轨迹以进行AKI风险预测。在SynerT的基础上，我们进一步设计了两个结合结构化临床信息的模型变体来扩展该骨干网络：SynerT-MM是一种后期融合的多模态扩展模型，整合了血流动力学负荷摘要信息与术前协变量；SynerTStack则是一种防泄漏的堆叠集成模型，在元学习阶段将SynerT-MM的交叉验证预测与强大的表格数据基线模型相结合。所有模型均在VitalDB（一个高保真围术期数据库）上，于严格的防泄漏评估框架下进行评估与验证。

    arXiv:2609.26848v1 Announce Type: cross  Abstract: Postoperative acute kidney injury (AKI) after major non-cardiac surgery carries substantial morbidity, yet early intraoperative risk stratification remains difficult. In this retrospective cohort study, we propose SynerT, a waveform-only hybrid temporal backbone that combines a causal dilated TCN with a hierarchy of dilated recurrent layers to encode early intraoperative physiologic trajectories for AKI risk prediction. Building on SynerT, we further design two model variants that extend the backbone with structured clinical context: SynerT-MM, a late-fusion multimodal extension that integrates hemodynamic burden summaries and preoperative covariates, and SynerTStack, a leakage-safe stacked ensemble that combines cross-validated predictions from SynerT-MM with strong tabular baselines at the meta-learning stage. All models are evaluated under a strict leakage-aware framework on VitalDB, a high-fidelity perioperative database, with pred
    
[^184]: LWCal：针对含噪声校准标签的表格分类器的损失加权校准方法

    LWCal: Loss-Weighted Calibration for Tabular Classifiers with Noisy Calibration Labels

    [https://arxiv.org/abs/2609.26839](https://arxiv.org/abs/2609.26839)

    提出LWCal，一种无需干净验证标签、无需噪声率估计、也无需重训练的事后校准方法，通过对与基础模型预测相矛盾的噪声标签样本降权，有效应对校准标签含噪声的场景。

    

    事后概率校准通常是在一个乐观假设下进行评估的：即留出的校准标签是干净的。然而，在许多AI部署场景中，标签来自弱标注者、历史决策、启发式规则或远程监督，因此破坏训练的同一标签噪声也会破坏校准。我们针对表格分类器研究了这种被忽视的失效模式，并提出了LWCal——一种仅需CPU的事后校准器，它会对那些噪声标签与基础模型留出概率相矛盾的校准样本进行降权。LWCal不需要干净的验证标签，不需要噪声率估计，也不需要重新训练基础分类器。第二种变体Gated-LWCal增加了一个保守的分歧门控机制，当校准集显得极不一致时，会退回到原始分数。在九个本地二分类表格任务、六个随机种子、对称与非对称标签损坏以及三种树（模型）的实验中……

    arXiv:2609.26839v1 Announce Type: cross  Abstract: Post-hoc probability calibration is usually evaluated under an optimistic assumption: the held-out calibration labels are clean. In many AI deployment settings, however, labels come from weak annotators, historical decisions, heuristics, or distant supervision, so the same label noise that corrupts training also corrupts calibration. We study this overlooked failure mode for tabular classifiers and propose LWCal, a CPU-only post-hoc calibrator that down-weights calibration examples whose noisy labels are contradicted by the base model's held-out probability. LWCal requires no clean validation labels, no noise-rate estimate, and no retraining of the base classifier. A second variant, Gated-LWCal, adds a conservative disagreement gate that backs off toward the raw score when the calibration split appears extremely inconsistent. On nine local binary tabular tasks, six random seeds, symmetric and asymmetric label corruption, and three tree
    
[^185]: 智能体-工具交互中的静默失败：对 ToolUniverse 的审计

    Silent Failures in Agent-Tool Interaction: An Audit of ToolUniverse

    [https://arxiv.org/abs/2609.26836](https://arxiv.org/abs/2609.26836)

    本研究首次提出并定义了智能体-工具交互中的“静默失败”现象——即工具调用看似成功但返回信息不完整或缺失且无任何提示——并开发了相应的审计机制，在生物学智能体工作流的15个科学工具上进行了识别与验证。

    

    智能体AI系统正日益采用集成多种工具的自动化流水线。尽管此前的研究和基准测试已经关注了这些智能体系统的任务成功与任务完成情况，但关于智能体与工具交互的研究，特别是在生物学智能体工作流中的研究仍然有限。本研究调查了智能体与工具交互中的一类特定失败：工具调用看似成功，但通过API/封装器从工具获取的部分或全部信息或功能不完整或缺失，且没有任何通信或通知告知用户或智能体此类信息缺失。我们将这类失败称为“静默失败”，因为用户或智能体并未意识到此类失败已经发生。为开展本研究，我们开发了一种审计机制，通过检查15个科学工具（及其相关的API文档和工具文档）来识别智能体与工具交互中的此类静默失败。

    arXiv:2609.26836v1 Announce Type: new  Abstract: Agentic AI systems are increasingly adopting automated pipelines that integrate multiple tools. While prior research and benchmarks have studied about task success and task completion of these agentic systems, the research about agent to tool interaction, specifically in biology agentic workflow is limited. This study investigates specific failures in agent to tool interaction where a tool invocation appears successful, some or all of the information or functionality from the tool via API/ wrapper is incomplete or missing and there are no communications / notifications to the user or the agent about such missing information. We call this a silent failures as the user or the agents are not aware that such failure has occurred. For the purposes of this study we developed an audit mechanism to identify such silent failures in Agent to tool interaction, by examining 15 scientific tools (and their associated API documentation and tool documen
    
[^186]: Spec2COBOLRot：一种用于生成真实COBOL语料库的智能体AI退化循环

    Spec2COBOLRot: An Agentic-AI Degradation Loop for Realistic COBOL Corpus Generation

    [https://arxiv.org/abs/2609.26835](https://arxiv.org/abs/2609.26835)

    该论文提出了Spec2COBOLRot——一种智能体AI流水线，通过将规范驱动的程序生成与由真实生产代码提取的模式和复杂度目标引导的迭代退化循环相结合，来生成具有真实结构复杂度的COBOL程序语料库，从而解决COBOL现代化方法基准测试中代表性语料匮乏的问题。

    

    COBOL仍然被广泛部署，但反映真实生产代码的代表性语料库却很少可用，这限制了对现代化方法进行严格基准测试的可能性。我们提出了一个系统性的智能体AI流水线，用于生成真实的COBOL程序，该流水线将规范驱动的生成与迭代退化相结合，其中退化过程由从真实生产代码中提取的模式和复杂度目标来引导。在此，“真实性”被理解为通过我们所定义的指标度量的、与生产代码之间的结构保真度。我们在来自不同业务领域的三个程序上，评估了退化过程能否在保持业务行为的同时达到目标复杂度水平，并考察了该方法的局限性。结果表明，该流水线能够可靠地生成语法有效的程序，并使其趋向真实的结构复杂度。然而，业务行为的保持并不总是能够通过构造方式来实现，且仅独立地以结构指标为目标（原文在此处截断）。

    arXiv:2609.26835v1 Announce Type: cross  Abstract: COBOL remains widely deployed, yet representative corpora reflecting real production code are rarely available, limiting rigorous benchmarking of modernization approaches. We propose a systematic agentic AI pipeline for generating realistic COBOL programs, combining specification-driven generation with iterative degradation guided by patterns and complexity targets extracted from real production code. Here, realism is understood as structural fidelity to production code as captured by our metrics. We evaluate whether degradation reaches target complexity levels while preserving business behavior, and examine the limits of the approach, across three programs from distinct business domains. Results show the pipeline reliably produces syntactically valid programs and moves them toward realistic structural complexity. However, preserving business behavior is not always achieved by construction, and targeting structural metrics independentl
    
[^187]: 验证与仿真捕获不同的错误：面向LLM生成电路的四级评估方法

    Validation and Simulation Catch Different Errors: Four Levels of Evaluation for LLM-Generated Circuits

    [https://arxiv.org/abs/2609.26830](https://arxiv.org/abs/2609.26830)

    本文提出针对LLM生成电路的四级评估框架（模式有效性、拓扑有效性、后端可执行性、元件集合一致性），并证明验证与仿真各自能捕获对方遗漏的错误类别，因此仅靠仿真通过无法保证电路结构正确。

    

    仿真成功并不等同于LLM生成电路的结构正确性。我们基于一个建立在类型化电路交换表示之上的已部署流水线，在包含150个电路的三语基准上定义并测量了四个评估层级——模式有效性、拓扑有效性、后端可执行性以及元件集合一致性。这四个层级并非嵌套关系。在gpt-4o-mini上，150个电路中有16个（10.7%，95% CI 6.7–16.6）被拓扑验证器拒绝，却在ngspice中无任何错误或警告地执行；其中12个包含完全符合要求的元件，但有一个端子处于断开状态。相反，有7个电路（4.7%）通过了验证器却被ngspice拒绝。另有10个电路两项检查均未通过，117个电路两项检查均通过，因此每项检查都能检测到另一项所遗漏的一类错误。一个最小的三元件分压器示例展示了这种代价：一个悬空的电阻器会使输出报告为5.00 V而非正确的2.50 V，而ngspice却保持沉默。一个配对消融实验……

    arXiv:2609.26830v1 Announce Type: cross  Abstract: Simulation success is not equivalent to structural correctness for LLM-generated circuits. We define and measure four evaluation levels -- schema validity, topological validity, backend executability, and component-set agreement -- on a 150-circuit trilingual benchmark, through a deployed pipeline built on a typed circuit interchange representation.   The levels are not nested. On gpt-4o-mini, 16 of 150 circuits (10.7%, 95% CI 6.7-16.6) were rejected by the topological validator but executed in ngspice with no error or warning; 12 of these contained exactly the requested components, with one terminal disconnected. Conversely, 7 circuits (4.7%) passed the validator and ngspice refused them. Ten failed both checks and 117 passed both, so each check detects a class the other misses. A minimal three-component divider shows the cost: a dangling resistor reports 5.00 V instead of 2.50 V while ngspice stays silent.   A paired ablation, in whi
    
[^188]: 通过块感知的KV缓存管理桥接LLM服务与CXL-SSD

    Bridging LLM Serving and CXL-SSDs with Chunk-Aware KV Cache Management

    [https://arxiv.org/abs/2609.26828](https://arxiv.org/abs/2609.26828)

    该论文提出了LM-CXD，一种专为LLM前缀缓存定制的CXL-SSD，通过将KV块作为设备可见的I/O单元、向服务引擎暴露NAND到DRAM的迁移进度，并将设备DRAM用作GPU可访问缓冲区，弥合了LLM服务引擎与存储设备之间的语义鸿沟，从而克服了标准CXL-SSD在KV缓存场景下性能不足的问题。

    

    基于NAND的存储器为扩展LLM前缀缓存提供了所需的容量，但其块I/O路径除了NAND延迟之外，还会带来CPU缓存争用和主机DRAM中转的开销。我们的特性分析表明，即使以DRAM作为存储介质，这些接口开销依然存在，这促使我们采用CXL-SSD来实现对NAND容量的字节可寻址访问。然而令人惊讶的是，标准的CXL-SSD仍然比本地DRAM慢约3倍，且并不比NVMe SSD更快，而通用的预取技术几乎收效甚微。我们提出了LM-CXD，一种专为LLM前缀缓存设计的CXL-SSD。LM-CXD弥合了服务引擎（知晓哪些KV块将被消费）与设备（控制其放置和移动）之间的语义鸿沟。它使KV块成为设备可见的I/O单元，将NAND到DRAM的迁移进度暴露给服务引擎，并将设备DRAM用作GPU可访问的缓冲区。LM-CXD进一步协调请求调度与（摘要在此处截断）

    arXiv:2609.26828v1 Announce Type: cross  Abstract: NAND-backed storage offers the capacity needed to scale LLM prefix caching, but its block I/O path incurs CPU cache contention and host-DRAM staging in addition to NAND latency. Our characterization shows that these interface costs persist even with DRAM as the storage medium, motivating CXL-SSDs for byte-addressable access to NAND-backed capacity. Surprisingly, however, a stock CXL-SSD remains about 3$\times$ slower than local DRAM and no faster than an NVMe SSD, while generic prefetching provides little benefit. We present LM-CXD, a CXL-SSD specialized for LLM prefix caching. LM-CXD bridges the semantic gap between the serving engine, which knows which KV chunks will be consumed, and the device, which controls their placement and movement. It makes KV chunks device-visible I/O units, exposes NAND-to-DRAM progress to the serving engine, and uses device DRAM as a GPU-accessible buffer. LM-CXD further coordinates request scheduling with
    
[^189]: 什么使 Terminal-Bench 任务变得困难？——在经裁定的智能体语料库上区分真实困难与虚假困难

    What Makes a Terminal-Bench Task Hard? Separating Genuine Hardness from Fake-Hardness on an Adjudicated Agentic Corpus

    [https://arxiv.org/abs/2609.26826](https://arxiv.org/abs/2609.26826)

    本文提出一套有序的有效性筛选方法，综合任务工件、参考解运行、空解对照、对抗试验与遥测等多源证据，从 Terminal-Bench 的 125 个全失败任务中区分真实困难与虚假困难，发现其中仅 78 个可被认证为真正未解决的任务。

    

    前沿基准测试需要当前模型无法解决的任务，但没有任何模型能解决的任务并不自动就是困难任务。同样的零通过率可能源于真实的能力差距，但也可能源于上下文缺失、参考解决方案损坏、基础设施故障，或可被绕过的验证器。本文利用一份冻结的 Terminal-Bench 3 / Frontier-Bench 0.1 生产记录来研究这一问题，该记录包含 1,081 个拉取请求、639 个已评分任务、28,801 次试验以及 105,933 美元的已记录智能体支出。我们追问：一个全失败的任务究竟能证明什么。针对 125 个没有任何诚实通过的任务，我们综合任务工件、参考解决方案运行结果、空解决方案对照、对抗性试验、轨迹、遥测数据和评审记录，并应用一套有序的有效性筛选流程。结果显示，125 个任务中仅有 78 个被保留为“经认证未解决”的候选任务，其余任务则包括 14 个预言机损坏的任务、8 个被基础设施故障主导的任务等。

    arXiv:2609.26826v1 Announce Type: cross  Abstract: Frontier benchmarks need tasks that current models cannot solve. But a task that no model solves is not automatically a hard task. The same zero pass rate can come from a real capability gap, but it can also come from missing context, a broken reference solution, infrastructure failure, or a verifier that can be bypassed. In this paper, we study this issue using a frozen Terminal-Bench 3 / Frontier-Bench 0.1 production record with 1,081 pull requests, 639 scored tasks, 28,801 trials, and $105,933 in logged agent spend. We ask what an all-fail task actually certifies. For the 125 tasks with no honest pass, we combine task artifacts, reference-solution runs, empty-solution controls, adversarial trials, trajectories, telemetry, and review records, and apply an ordered validity screen. Only 78 of the 125 tasks survive as certified-unsolved candidates. The remaining tasks include 14 with broken oracles, 8 dominated by infrastructure failure
    
[^190]: Signal2Symbol：面向可解释生理时间序列异常检测的神经符号时间推理

    Signal2Symbol: Neuro-Symbolic Temporal Reasoning for Explainable Physiological Time-Series Anomaly Detection

    [https://arxiv.org/abs/2609.26820](https://arxiv.org/abs/2609.26820)

    提出了一种名为Signal2Symbol的神经符号框架，通过将ECG/EEG信号转换为符号序列并利用稀有项集挖掘对异常进行评分，实现了对生理时间序列的可解释异常检测，并能揭示局部异常之间的时间关联与重复模式。

    

    诸如心电图（ECG）和脑电图（EEG）等生理时间序列表现出复杂的时间结构、显著的采集变异性，以及对透明决策的强烈需求。尽管深度模型能够达到较高的检测性能，但它们在解释某个片段为何异常、局部异常如何随时间相互关联、以及检测结果是否属于更广泛重复模式等方面，通常提供的洞察有限。我们提出了Signal2Symbol，一个用于可解释生物信号异常检测的神经符号框架。该方法首先使用学习得到的VQ-VAE（向量量化变分自编码器）码本或SAX（符号聚合近似）基线方法，将ECG/EEG信号转换为符号序列。然后，它构建了二元组增强的标记窗口事务，并通过源自最小稀有项集挖掘的稀有项集证据对异常进行评分。检测到的异常窗口被合并为区间……

    arXiv:2609.26820v1 Announce Type: cross  Abstract: Physiological time series such as electrocardiograms (ECG) and electroencephalograms (EEG) exhibit complex temporal structure, substantial acquisition variability, and a strong need for transparent decision-making. Although deep models can achieve high detection performance, they often provide limited insight into why a segment is anomalous, how local anomalies relate over time, and whether a detection belongs to a broader recurring pattern. We propose Signal2Symbol, a neuro-symbolic framework for explainable biosignal anomaly detection. The method first converts ECG/EEG signals into symbolic sequences using either a learned VQ-VAE (Vector Quantized Variational Autoencoder) codebook or a SAX (Symbolic Aggregate approXimation) baseline. It then constructs bigram enriched token-window transactions and scores anomalies through rare itemset evidence derived from minimal rare itemset mining. Detected anomalous windows are merged into interv
    
[^191]: 从粗粒度流动表示中学习刚度依赖的流固耦合动力学

    Learning Stiffness Dependent Fluid Structure Dynamics from Coarse Flow Representations

    [https://arxiv.org/abs/2609.26816](https://arxiv.org/abs/2609.26816)

    本文提出一个刚度条件化的神经演化算子，利用混合CNN-Transformer架构和双向交叉注意力，能够从粗粒度流动表示中长期准确预测柔性板在三种刚度依赖响应状态下的流固耦合动力学。

    

    本文开发了一个用于流固耦合（FSI）动力学长期预测的数据驱动框架，重点关注柔性板的流致振动（FIV）。该框架采用刚度条件化的神经演化算子，联合表示欧拉流场和拉格朗日结构状态。柔性板由101个携带节点坐标和速度的有序结构标记表示，无量纲弯曲刚度作为全局条件变量。在混合CNN-Transformer架构中，双向交叉注意力机制耦合流体和结构表示。通过分阶段多步自回归滚动和对称性反射轨迹进行训练，单一算子能够捕获三种刚度依赖的响应状态：偏转-扑动、偏转和扑动。预测轨迹保留了主要流动结构、结构振荡和主导频率，同时盲测1000秒（摘要在此处截断）

    arXiv:2609.26816v1 Announce Type: cross  Abstract: This paper develops a data-driven framework for long-term prediction of fluid--structure interaction (FSI) dynamics, focusing on the flow-induced vibration (FIV) of a flexible plate. A stiffness-conditioned neural evolution operator jointly represents the Eulerian flow field and Lagrangian structural state. The plate is represented by 101 ordered structural tokens carrying nodal coordinates and velocities, with nondimensional bending stiffness as a global conditioning variable. Bidirectional cross-attention couples fluid and structural representations within a hybrid CNN-Transformer architecture. Trained with staged multi-step autoregressive rollouts and symmetry-reflected trajectories, a single operator captures three stiffness-dependent response regimes: deflected--flapping, deflected, and flapping. The predicted trajectories preserve the principal flow structures, structural oscillations, and dominant frequencies, while blind 1000-s
    
[^192]: Lean 4 中哥德尔与斯科特版本的本体论论证

    G\"odel's and Scott's Variants of the Ontological Argument in Lean 4

    [https://arxiv.org/abs/2609.26806](https://arxiv.org/abs/2609.26806)

    该论文将哥德尔与斯科特本体论论证的 Isabelle/HOL 形式化数据集完整且保结构地移植到 Lean 4，验证了全部 548 条陈述的一致性，并重新证明了包括模态坍缩、一神论等在内的所有原开发中被证明的结论。

    

    本文呈现了将 Benzmüller 和 Scott 关于哥德尔模态本体论论证及其斯科特变体研究所配套的 Isabelle/HOL 数据集完整且保结构地移植到 Lean 4 的工作。该移植包含 30 个 Lean 4 模块，每个模块对应一个 Isabelle/HOL 理论，保留了章节结构、声明顺序以及每条公理、定义、引理和定理的名称；一个比较工具验证了全部 548 条陈述完全一致。Isabelle/HOL 开发中证明的所有内容均被重新证明，包括哥德尔 1970 年公理的不一致性、修复后的哥德尔变体、斯科特变体、模态坍缩、一神论以及肯定性质的超滤性质；原文中有五条陈述在自动证明器找到证明后未再被重新验证（其中一条随后被作为公设），这些陈述也被证明。剩余 45 条未证明的陈述恰好是原文通过 nitpick 反驳的（35 条）或留作未决的陈述。

    arXiv:2609.26806v1 Announce Type: cross  Abstract: This paper presents a complete, structure-preserving port to Lean 4 of the Isabelle/HOL dataset accompanying Benzm\"uller and Scott's study of G\"odel's modal ontological argument and Scott's variant of it. The port comprises 30 Lean 4 modules, one per Isabelle/HOL theory, retaining the section structure, the declaration order and the name of every axiom, definition, lemma and theorem; a comparison tool certifies all 548 statements identical. Everything the Isabelle/HOL development proves is proved again, including the inconsistency of G\"odel's 1970 axioms, the repaired G\"odel variants, Scott's variant, modal collapse, monotheism and the ultrafilter property of the positive properties; five statements the original leaves unreplayed after an automated prover had found a proof, one of which it then postulates, are proved as well. The 45 remaining unproved statements are exactly those the original refutes by nitpick (35) or leaves open 
    
[^193]: SpeakerMem-R1：面向多方对话的以说话人为中心的双轨记忆

    SpeakerMem-R1: Speaker-Centered Dual-Track Memory for Multi-Party Dialogue

    [https://arxiv.org/abs/2609.26780](https://arxiv.org/abs/2609.26780)

    提出SpeakerMem-R1，一种以说话人为中心的双轨记忆框架，通过存储带说话人标签的逐字消息和个人/群体层面的衍生状态，解决多方对话中的消息归属、关系理解与状态重建两大瓶颈。

    

    多方场景下的长期对话记忆不仅仅是从长期对话中检索相关内容：它必须区分谁说了什么、每句话涉及谁、个体之间如何看待彼此、哪些信息为群体所共享，以及状态如何随时间变化。最近针对多方对话基准的研究表明，现有的通用大语言模型记忆系统往往丢失人物与群体关系，或难以整合分布在成员、群体和时间中的线索。这些问题共同揭示了两个核心瓶颈：多方对话中的消息归属与关系理解，以及从交错历史中进行的状态重建。为解决这两个问题，我们提出了 SpeakerMem-R1：其双轨记忆存储带有说话人标签的逐字消息以及衍生状态，并将它们组织为个人层面和群体层面的视图，随后按实体、事件等方式结合两条轨道的证据（摘要在此处被截断）。

    arXiv:2609.26780v1 Announce Type: new  Abstract: Long-term conversational memory in multi-party settings requires more than retrieving relevant content from long-term conversations: it must distinguish who said what, whom each statement concerns, how individuals perceive one another, what information is shared by the group, and how states change over time. Recent studies on multi-party dialogue benchmarks show that existing general-purpose LLM memory systems tend to lose person and group relations or struggle to integrate clues distributed across members, groups, and time. Together, these issues reveal two core bottlenecks: message attribution and relational understanding in multi-party dialogue, and state reconstruction from interleaved histories. To address both, we propose $\textbf{SpeakerMem-R1}$: its dual-track memory stores speaker-labeled verbatim messages and derived states organized into person-level and group-level views, then combines evidence from both tracks by entity, eve
    
[^194]: FleXray：通用临床X光图像分割

    FleXray: Universal Clinical X-ray Segmentation

    [https://arxiv.org/abs/2609.26756](https://arxiv.org/abs/2609.26756)

    FleXray通过构建基于物理的生成式X光数据引擎，利用现有3D CT分割数据集自动合成带完整标注的X光图像，从而无需人工标注即可实现全身临床X光的通用解剖结构分割。

    

    X光是医学中应用最广泛的成像方式，却仍是量化程度最低的方式之一。与CT或MRI等体积成像模态不同，X光将三维解剖结构压缩为二维投影，导致结构相互重叠、解剖边界模糊，即使对专家而言也是如此。因此，为训练通用分割系统而对X光数据库进行人工标注并不切实际，这使得形态测量与功能性X光分析只能局限于狭窄的解剖区域和特定应用。为此，我们提出了FleXray，一个可对临床X光进行全身解剖结构分割的通用模型。我们不再耗费精力收集大规模人工标注的X光数据集，而是构建了一个可扩展的、基于物理的生成式X光数据引擎。利用现有的3D全身CT分割数据集和生成式图像编辑模型，我们模拟出具有多样化外观、生理特性和成像几何结构的、带完整标注的2D X光图像。

    arXiv:2609.26756v2 Announce Type: replace-cross  Abstract: X-ray is medicine's most widely used imaging modality, yet remains among its least quantitative. Unlike volumetric modalities like CT or MRI, X-ray collapses 3D anatomy into a 2D projection, causing structures to overlap and anatomical boundaries to be ambiguous, even to experts. As a result, labeling X-ray databases for training general-purpose segmentation systems is impractical, leaving morphometric and functional X-ray analysis confined to narrow anatomical regions and applications. To this end, we present FleXray, a generalist model for anatomical segmentation across the entire body in clinical X-rays. Instead of curating large, manually annotated X-ray datasets, we build a scalable, physics-based generative X-ray data engine. Using existing 3D whole-body CT segmentation datasets and generative image-editing models, we simulate fully-annotated 2D X-rays with diverse appearances, physiological properties, and imaging geomet
    
[^195]: QuantWM：面向世界模型与视频生成的时间一致性2比特KV缓存量化

    QuantWM: Temporally Consistent 2-Bit KV Cache Quantization for World Models and Video Generation

    [https://arxiv.org/abs/2609.26425](https://arxiv.org/abs/2609.26425)

    提出无需训练的2比特KV缓存量化方法QuantWM，通过在量化中显式保持注意力logits与时空token选择，解决了现有方法在视频生成与世界模型中导致的时间闪烁与视觉退化问题。

    

    KV缓存的内存占用已成为视频生成与世界模型部署的主要瓶颈，这促使人们研究低比特量化以提升效率。现有的2比特KV缓存量化方法在VBench等视频基准上能够达到几乎无损的性能，然而我们发现它们仍然会造成严重的时间闪烁与视觉退化。同时，更深入的研究表明，Key量化产生的重建误差比Value更小，但令人惊讶的是却导致了更大的输出退化。我们将这一差异追溯到注意力机制：Key的微小扰动会改变注意力logits（即QK^⊤），并改变Query所选择的时空token。这些观察促使我们在KV缓存量化过程中显式地保持注意力logits与时空token选择，以缓解视觉退化问题。为解决这一问题，我们提出了QuantWM，一种无需训练（摘要在此处被截断）

    arXiv:2609.26425v2 Announce Type: replace-cross  Abstract: KV cache memory has become a major deployment bottleneck for video generation and world models, which motivates low-bit quantization study for efficiency. Existing 2-bit KV cache quantization methods can achieve nearly lossless performance on video benchmarks such as VBench, however, we find that they still cause severe temporal flickering and visual degradation. Meanwhile, deeper investigates show that Key quantization produces smaller reconstruction errors than Value, but surprisingly leads to much larger output degradation. We trace this discrepancy to attention: small Key perturbations can change the attention logits, i.e., QK^\top, and shift the temporal-spatial tokens selected by Queries. These observations motivate us to explicitly preserve attention logits and temporal-spatial token selection during KV cache quantization to alleviate the visual degradation problem. To address this issue, we present QuantWM, a training-f
    
[^196]: TransBERT：面向特定领域语言建模的合成翻译框架

    TransBERT: A Framework for Synthetic Translation in Domain-Specific Language Modeling

    [https://arxiv.org/abs/2609.26347](https://arxiv.org/abs/2609.26347)

    提出仅使用合成翻译文本预训练语言模型的TransBERT框架，并证明仅凭合成翻译数据即可在法语生命科学领域的各类下游任务上达到最先进性能。

    

    专业领域中非英语语言数据的稀缺严重限制了有效自然语言处理（NLP）工具的发展。我们提出了TransBERT，一个仅使用合成翻译文本进行语言模型预训练的新型框架，并介绍了可扩展的翻译工具包TransCorpus。聚焦于法语生命科学领域，我们的方法表明，仅利用合成翻译数据即可在各种下游任务上达到最先进的性能。我们发布了TransCorpus工具包、TransCorpus-bio-fr语料库（36.4GB的法语生命科学文本）、TransBERT-bio-fr及其相关的预训练语言模型，以及用于预训练和微调的可复现代码。我们的结果突显了在高资源翻译方向上利用合成翻译来构建低资源语言/领域对高质量NLP资源的可行性。

    arXiv:2609.26347v1 Announce Type: new  Abstract: The scarcity of non-English language data in specialized domains significantly limits the development of effective Natural Language Processing (NLP) tools. We present TransBERT, a novel framework for pre-training language models using exclusively synthetically translated text, and introduce TransCorpus, a scalable translation toolkit. Focusing on the life sciences domain in French, our approach demonstrates that state-of-the-art performance on various downstream tasks can be achieved solely by leveraging synthetically translated data. We release the TransCorpus toolkit, the TransCorpus-bio-fr corpus (36.4GB of French life sciences text), TransBERT-bio-fr, its associated pre-trained language model and reproducible code for both pre-training and fine-tuning. Our results highlight the viability of synthetic translation in a high-resource translation direction for building high-quality NLP resources in low-resource language/domain pairs.
    
[^197]: 有品味的智能体：在长程任务中衡量与提升品味

    The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks

    [https://arxiv.org/abs/2609.25804](https://arxiv.org/abs/2609.25804)

    提出了“品味”（taste）这一衡量 LLM 智能体长程决策能力的新概念，并构建了从工程与研究任务的真实轨迹中自动生成决策分叉题的基准 Taste-Bench，用于测量和提升智能体在长程任务中的品味。

    

    LLM 智能体越来越多地承担长程任务，它们在过程中做出的决策——例如测试哪个假设、基于哪个实现继续开发——决定了整个运行的最终结果。做好这些决策正在成为工程智能体和研究智能体的一项关键能力。我们将做出良好长程决策的能力称为智能体的“品味”。现有基准测试衡量的是智能体在长程任务上的端到端成功，但没有任何一个衡量智能体的品味。为了解决这个问题，我们构建了 Taste-Bench，这是一个品味问题基准，由智能体在工程和研究任务中产生的轨迹自动构建而成。每个问题呈现一个决策分叉点——即轨迹中存在多个可选方向、且其中一个会带来更好结果的位置——被评估的模型需要在看不到分叉点之后会发生什么的情况下，在这些方向中做出选择。

    arXiv:2609.25804v2 Announce Type: replace  Abstract: LLM agents increasingly work on long-horizon tasks, and the decisions they make along the way, such as which hypothesis to test or which implementation to build on, determine the outcome of the whole run. Making these decisions well is becoming a key capability for both engineering and research agents. We refer to the ability to make good long-horizon decisions as the taste of an agent. While existing benchmarks measure the end-to-end success of agents on long-horizon tasks, none of them measures the taste of an agent. To address this problem, we build Taste-Bench, a benchmark of taste questions constructed automatically from trajectories that agents produced in engineering and research tasks. Each question presents a decision fork, a point in a trajectory where multiple directions are available and one of them leads to a better outcome, and the evaluated model chooses among these directions without seeing what happens after the fork
    
[^198]: 儿童如何设计并推理值得信赖的AI聊天机器人

    How Children Design and Reason about Trustworthy AI Chatbots

    [https://arxiv.org/abs/2609.25244](https://arxiv.org/abs/2609.25244)

    本研究开发了一个让儿童自主设计聊天机器人的平台，通过对115名8-18岁学习者的混合方法研究发现，低龄学生会设置更高的自信度，甚至认为“故意出错但按设计行事”的聊天机器人也值得信赖，揭示了儿童对AI可信度的独特理解方式。

    

    儿童越来越多地与AI聊天机器人互动，因此信任校准成为AI素养的重要组成部分。以往研究主要将儿童对AI的信任视为用户评估他人构建的系统，而非作为自己聊天机器人的设计者。我们开发了一个聊天机器人构建环境，支持调节与信任相关的特质（如自信度、透明度、正式程度、果断性）、规则和角色设定。我们对115名学习者（8-18岁）开展了混合方法研究，他们共制作了119个聊天机器人。我们考察了儿童如何配置他们的聊天机器人、如何推理可信度，以及聊天机器人的行为与其设计的契合程度。年龄较小的学生（10-13岁）设置的自信度显著高于年龄较大的学生（14-18岁），部分学生还刻意构建了会故意给出错误答案的聊天机器人，却仍认为其“值得信赖”，理由是聊天机器人做了它被设计要做的事情。低龄学生将信任等同于目的（摘要在此处截断）。

    arXiv:2609.25244v2 Announce Type: replace-cross  Abstract: Children increasingly interact with AI chatbots, making trust calibration essential to AI literacy. Prior research has examined children's trust in AI mainly as users evaluating systems built by others, rather than as designers of their own chatbots. We developed a chatbot-building environment with adjustable trust-relevant traits (e.g., confidence, transparency, formality, assertiveness), rules, and persona. We conducted mixed-methods study with 115 learners (ages 8-18) who made 119 chatbots. We examined how children configured their chatbots, reasoned about trustworthiness, and how closely chatbot behavior aligned with their designs. Younger students (age 10-13) set significantly higher confidence than older students (age 14-18), and some deliberately built chatbots that gave wrong answers on purpose, yet still called them trustworthy, arguing that a chatbot does what it was built to do. Younger students equated trust with pu
    
[^199]: 用于科学决策的Jev：评估语义选择及其后果

    Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences

    [https://arxiv.org/abs/2609.24965](https://arxiv.org/abs/2609.24965)

    该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。

    

    arXiv:2609.24965v1 公告类型：新论文 摘要：科学工作流程通常需要在确定性计算进行之前，在已知关系之间做出选择。观测数据是否共享相同的文化、处理方式或参考标准，可能会改变由此产生的计数或比较的科学含义。我们使用一个遵循其文档指导并将算术运算分配给代码的测试框架，将Jev作为语义决策组件进行评估。该研究在十个科学案例的二十个有来源依据的选择上比较了十二种模型配置，每种配置重复五次。我们分别测量语义选择、下游输出和最终结论标签。Jev与其他五种配置在完全语义正确性上持平，并在成功响应中实现了观察到的最低中位延迟。在三个对比模型中，对一个文化历史问题的七次错误选择改变了下游计数，同时保持了正确的最终标签。这些结果确定了一个有用的角色

    arXiv:2609.24965v1 Announce Type: new  Abstract: Scientific workflows often require choosing among known relations before a deterministic calculation can proceed. Whether observations share a culture, treatment or reference standard can change the scientific meaning of the resulting count or comparison. We evaluate Jev as a semantic decision component using a harness that follows its documented guidance and assigns arithmetic to code. The study compares twelve model configurations on twenty source-grounded Choices across ten scientific cases, each repeated five times. We measure semantic selections, downstream outputs and final claim labels separately. Jev matched five other configurations at complete semantic correctness and achieved the lowest observed median latency among successful responses. Across three comparison models, seven wrong selections on one culture-history question changed downstream counts while preserving the correct final label. These results identify a useful role 
    
[^200]: Uranus：构建具身智能的下一代仿真基础设施

    Uranus: Building the Next-Generation Simulation Infrastructure for Embodied AI

    [https://arxiv.org/abs/2609.24815](https://arxiv.org/abs/2609.24815)

    Uranus是一个基于关节轨迹条件自回归扩散模型的数据驱动机器人仿真器，具备流式开放式rollout、24 FPS低延迟生成以及跨多种机器人本体的统一多视角生成接口三大能力，为具身智能提供下一代仿真基础设施。

    

    可扩展的仿真对于机器人数据生成、策略训练、评估和安全迭代至关重要，然而真实世界的交互成本高昂，且传统仿真器需要耗费大量人力的构建过程。我们提出了Uranus，一个基于关节轨迹条件自回归扩散模型构建的数据驱动机器人仿真器。Uranus提供三大关键能力：（1）流式、开放式rollout，可在线接收未来的关节位置轨迹，并自回归地每步生成一个潜在帧（对应四帧RGB图像），且不受固定时域限制；（2）低延迟生成，经过推理优化后达到24 FPS；（3）可扩展、可拓展的机器人控制，为多种机器人本体和相机配置下的同步多视角生成提供统一接口。我们在分布内和分布外数据上进行了全面的定量和定性评估。

    arXiv:2609.24815v2 Announce Type: cross  Abstract: Scalable simulation is essential for robot data generation, policy training, evaluation, and safe iteration, yet real-world interaction is costly and conventional simulators require labor-intensive construction. We present Uranus, a data-driven robot simulator built around a joint-trajectory-conditioned autoregressive diffusion model. Uranus offers three key capabilities: (1) streaming, open-ended rollout, which receives future joint-position trajectories online and autoregressively generates one latent frame per step, corresponding to four RGB frames, without a fixed horizon; (2) low-latency generation, achieving 24 FPS after inference optimization; and (3) scalable, extensible robot control, providing a unified interface for synchronized multi-view generation across diverse robot embodiments and camera configurations. We conduct comprehensive quantitative and qualitative evaluations on both in-distribution and out-of-distribution dat
    
[^201]: ActiveArena：机器人操作中主动感知的基准测试与理解

    ActiveArena: Benchmarking and Understanding Active Perception in Robotic Manipulation

    [https://arxiv.org/abs/2609.24124](https://arxiv.org/abs/2609.24124)

    该论文提出了ActiveArena基准体系，通过包含可控视点模拟器、35个多轮证据获取与记忆推理任务以及模块化VLA模型套件，系统性地评估和理解机器人操作中的主动感知能力。

    

    主动感知与操作对于机器人与复杂场景进行交互至关重要。现有的基准测试难以评估机器人如何以主动的方式有效获取信息并将其保存在记忆中。为此，我们推出了ActiveArena-Sim，一个具备可控视点和大规模工作空间的主动感知模拟器作为基础。在此基础上，我们提出了ActiveArena-Bench，它包含5个细粒度类别共35个任务，涵盖视觉探索和交互式信息获取。每个任务仅凭被动观察都难以解决，需要多轮证据获取和基于记忆的推理。该基准提供了丰富的记忆标注、标准化的训练数据，以及包含不相交场景、未见干扰物配置和新颖背景的ID/OOD评估协议。此外，我们还提出了ActiveArena-VLA，一个由13个视觉-语言-动作配置组成的模块化套件。

    arXiv:2609.24124v1 Announce Type: cross  Abstract: Active perception and manipulation are crucial for robots to interact with complex scenes. Existing benchmarks struggle to evaluate how robots effectively acquire and maintain information in memory in an active manner. To this end, we introduce ActiveArena-Sim, an active-perception simulator with controllable viewpoints and large-scale workspaces as the foundation. Built on this, we propose ActiveArena-Bench, which comprises 35 tasks across 5 fine-grained categories, covering visual exploration and interactive information acquisition. Each task is difficult to solve from passive observations alone, requiring multi-round evidence acquisition and memory-based reasoning. The benchmark provides rich memory annotations, standardized training data, and ID/OOD protocols featuring disjoint scenes, unseen distractor configurations, and novel backgrounds. Moreover, we present ActiveArena-VLA, a modular suite of 13 vision-language-action configur
    
[^202]: SyzHarness：基于补丁的内核漏洞复现与LLM合成的模糊测试Harness

    SyzHarness: Patch-Based Kernel Bug Reproduction with LLM-Synthesized Fuzzing Harnesses

    [https://arxiv.org/abs/2609.23889](https://arxiv.org/abs/2609.23889)

    SyzHarness将LLM推理与覆盖率引导的模糊测试相结合，通过LLM代理合成参数化模糊测试Harness（固定前置设置逻辑、仅暴露漏洞关键参数），实现基于补丁的Linux内核漏洞自动复现。

    

    自动化的内核漏洞复现对于漏洞分类、补丁验证和回归测试至关重要，但目前仍缺乏有效且高效的解决方案。核心挑战是双重的：复现程序必须首先恢复到达漏洞状态所需的触发脚手架，并确定实际触发漏洞的精确具体值。现有的定向模糊测试方法在恢复必要的触发脚手架方面效果不佳，而仅依赖LLM生成的方法则较为脆弱，因为它难以处理具体值的发现和运行时的不确定性。我们设计了SyzHarness，这是一个将LLM推理与覆盖率引导的模糊测试相结合的框架，用于基于补丁的Linux内核漏洞复现。给定一个补丁，SyzHarness使用由代码导航工具支撑的LLM代理来合成参数化的模糊测试Harness，该Harness固定了先决条件设置逻辑，同时仅暴露不确定的、漏洞关键的（参数）……

    arXiv:2609.23889v1 Announce Type: cross  Abstract: Automated kernel vulnerability reproduction is essential for bug triage, patch validation, and regression testing, but   still lacks an effective and efficient solution. The core challenge is twofold: a reproducer must first recover the   trigger scaffold needed to reach the vulnerable state and determine the precise concrete values that actually trigger   the bug. Existing directed fuzzing approaches are ineffective at recovering the necessary trigger scaffold, while LLM  only generation is brittle because it struggles with concrete-value discovery and runtime nondeterminism. We design   SyzHarness, a framework that combines LLM reasoning with coverage-guided fuzzing for patch-based Linux kernel   vulnerability reproduction. Given a patch, SyzHarness uses an LLM agent grounded by code navigation tools to   synthesize a parameterized fuzzing harness that fixes the prerequisite setup logic while exposing only uncertain, bug  critica
    
[^203]: WorkWorlds：一个用于评估AI智能体职场任务表现的基础设施

    WorkWorlds: An Infrastructure for Evaluating AI Agents on Workplace Tasks

    [https://arxiv.org/abs/2609.23806](https://arxiv.org/abs/2609.23806)

    WorkWorlds通过将组织状态与任务规范分离——先固定组织环境再引入任务——避免了评估环境预先编码任务信息，从而更真实地评估AI智能体完成职场任务的能力。

    

    许多知识工作基准测试是围绕单个任务构建的，每个任务所需的上下文是在任务被指定时或之后才被选择的。这种设计是在为任务而搭建的环境中测量类职场任务的表现。当任务规范指导选择哪些上下文时，评估可能会将任务信息编码到环境中，预先完成职场表现通常所需的部分信息定位工作。我们提出了WorkWorlds，一个将组织状态与任务规范分离的评估基础设施。一个世界首先固定修订版本、日期和员工席位，并具体化该员工可访问的组织状态；任务仅在之后才被引入。我们在一个主要的合成制药公司中实现了WorkWorlds，包含跨越6个员工席位的8个测量任务，并构建了额外的组织世界。在192个……（摘要在此处被截断）

    arXiv:2609.23806v1 Announce Type: new  Abstract: Many knowledge-work benchmarks are constructed around individual tasks, with the context needed for each task selected together with or after the task has been specified. This design measures performance on workplace-like tasks in an environment assembled for the task. When task specification guides which context is selected, the evaluation can encode task information into the environment and pre-complete part of the information-localization work that workplace performance normally requires. We introduce WorkWorlds, an evaluation infrastructure that separates organizational state from task specification. A world first fixes a revision, date, and employee seat and materializes the organizational state that employee can access; tasks are introduced only afterward. We implement WorkWorlds in a primary synthetic pharmaceutical company with 8 measured tasks across 6 employee seats, and construct additional organizational worlds. Across 192 ma
    
[^204]: OmniEcho：面向具身智能体的空间音频理解

    OmniEcho: Spatial Audio Understanding for Embodied Agents

    [https://arxiv.org/abs/2609.23407](https://arxiv.org/abs/2609.23407)

    该论文提出了统一的空间视听感知与音-视-语言导航基准OmniEchoBench，并开发了保持几何一致性的可控空间音频渲染流水线以及空间感知全模态模型OmniEcho，以提升具身智能体的空间音频理解能力。

    

    人类可以毫不费力地定位声源方向，并将其与视觉线索结合进行推理，但这对于具身智能体来说仍然具有挑战性。特别是，目前仍不清楚如何在具身环境中有效地评估和建模空间音频理解。为了弥补这一空白，我们提出了OmniEchoBench，一个用于空间视听感知和音-视-语言导航的统一基准。OmniEchoBench包含六项任务，涵盖197个真实世界的空间视听场景、2,972个问答对，以及900个导航样本，其中包含从30个真实环境中采集的一阶高保真立体声（FOA）音频。为了实现可扩展的训练监督，我们开发了一个空间音频的可控渲染流水线，它保持了声源、视觉观察和智能体轨迹之间的几何一致性。在此基础上，我们提出了OmniEcho，一个具有空间感知能力的全模态模型。

    arXiv:2609.23407v1 Announce Type: cross  Abstract: Humans can effortlessly localize the direction of a sound source and integrate it with visual cues for reasoning, yet this remains challenging for embodied agents. In particular, it is still unclear how to effectively evaluate and model spatial audio understanding in embodied settings. To address this gap, we introduce \textbf{OmniEchoBench}, a unified benchmark for spatial audio-visual perception and audio-vision-language navigation. OmniEchoBench comprises six tasks over 197 real-world spatial audio-visual scenes, 2,972 question-answer pairs, and 900 navigation samples with first-order ambisonics (FOA) audio collected from 30 real-world environments. To enable scalable training supervision, we develop a controllable rendering pipeline for spatial audio. It preserves geometric consistency among sound sources, visual observations, and agent trajectories. Building on this, we propose \textbf{OmniEcho}, a spatially aware omni-modal model
    
[^205]: 泄漏积分器重构：驯服递归差分时间序列预测中的误差累积

    Leaky-integrator reconstruction: taming error accumulation in recursive differenced time-series forecasting

    [https://arxiv.org/abs/2609.23378](https://arxiv.org/abs/2609.23378)

    提出一种无需训练的泄漏积分器重构方法，通过将积分器极点移入单位圆内，从理论上约束并大幅降低递归差分时间序列预测中的误差累积。

    

    我们提出泄漏积分器重构，这是一种无需训练的方法，能够治愈递归差分预测中的误差累积问题。我们的第一个贡献是诊断性的：预测单步变化并通过累积求和进行积分——这是应对非平稳性的标准方法——本质上是一个极点位于单位圆上的离散积分器，我们证明这会导致非线性模型的递归展开发散，其336步误差达到表现良好预测器的数倍（归一化MAE为1.6-3.8，而后者约为0.8），且这一现象在所有测试过的神经架构中均存在。我们的第二个也是核心贡献是解决方案：使用泄漏积分器 H(z) = 1/(1 - gamma z^-1)（gamma < 1）将极点移入单位圆内部，从而可证明地约束累积误差方差。在重构阶段应用单一固定的 gamma=0.9（无需重新训练，只需对任何已部署的单步预测器或基础模型预测器做两行修改），即可显著缩小误差

    arXiv:2609.23378v1 Announce Type: cross  Abstract: We introduce leaky-integrator reconstruction, a training-free method that cures the error accumulation of recursive differenced forecasting. Our first contribution is diagnostic: predicting one-step changes and integrating them by cumulative summation, the standard remedy for non-stationarity, is a discrete integrator with a pole on the unit circle, and we show this makes recursive rollout of a nonlinear model diverge, its 336-step error reaching several times that of a well-behaved forecaster (normalised MAE 1.6-3.8 versus about 0.8) across every neural architecture tested. Our second, central contribution is the fix: move the pole inside the unit circle with a leaky integrator H(z) = 1/(1 - gamma z^-1), gamma < 1, which provably bounds the accumulated error variance. Applied at reconstruction time with a single fixed gamma=0.9 (no retraining, a two-line change to any deployed one-step or foundation-model forecaster), it shrinks error
    
[^206]: 从概念对齐到因果锚定：思维链忠实性的干预测试

    From Concept Alignment to Causal Grounding: An Intervention Test of Chain-of-Thought Faithfulness

    [https://arxiv.org/abs/2609.23065](https://arxiv.org/abs/2609.23065)

    该论文将CoT忠实性重新定义为“内部概念锚定”问题，利用共享稀疏自编码器直接比较预测过程与CoT过程的内部概念，并首次引入三个相关性对齐指标和一个因果消融指标Δp，以检验共享概念是否因果性地驱动模型答案。

    

    思维链可以听起来合理，却可能对模型的底层推理不忠实。以往大多数工作通过输入-输出行为或输入归因来探究CoT的忠实性，而对内部计算的探索在很大程度上仍属空白。我们转而将忠实性界定为内部概念锚定问题：大语言模型（LLM）的CoT推理是否调用了支持其直接预测的相同内部概念，并且这些共享概念是否因果性地驱动其答案？使用单个共享的稀疏自编码器（SAE）——一种对LLM所使用潜在概念的可靠近似器——来编码预测过程和CoT过程，使二者的内部概念可以直接比较。我们提出了三个概念层面的相关性对齐度量指标，以及一个因果度量指标Δp，该指标通过消融共享概念并测量答案概率的下降来检验因果作用。在五个LLM和四个数据集上的实验表明，概念对齐总体上较高，正如t……（摘要在此处截断）

    arXiv:2609.23065v1 Announce Type: new  Abstract: Chain-of-thought (CoT) can sound plausible yet be unfaithful to the model's underlying reasoning. Most prior work probes CoT faithfulness through input--output behavior or input attributions, leaving internal computation largely underexplored. We instead cast faithfulness as internal concept grounding: Does a large language model's (LLM) CoT reasoning engage the same internal concepts that support the LLM's direct prediction, and do the shared concepts causally drive its answer? Encoding a prediction pass and a CoT pass with a single shared sparse autoencoder (SAE), a reliable approximator of the latent concepts LLMs use, makes their internal concepts directly comparable. We introduce three correlational metrics of concept-level alignment and a causal metric, $\Delta p$, which ablates the shared concepts and measures the drop in answer probability. Across five LLMs and four datasets, concept alignment is generally high, as indicated by t
    
[^207]: RewardVerse：基于评分准则引导的策略优化用于视频奖励建模

    RewardVerse: Rubric-Guided Policy Optimization for Video Reward Modeling

    [https://arxiv.org/abs/2609.22947](https://arxiv.org/abs/2609.22947)

    提出RewardVerse框架，通过引入动态评分准则作为评估查询与评分器之间的中间表示，先生成明确评估标准再进行准则引导打分，从而解决视频奖励模型直接标量评分导致的标量漂移问题，为视频生成模型的强化学习提供稳定可靠的奖励信号。

    

    强化学习（RL）对于优化视频生成模型至关重要，而稳健的奖励模型（RM）则是其基石。然而，现有的视频奖励模型往往产生不稳定的标量分数，因为它们在缺乏明确评估标准的情况下，直接将复杂、主观的视频质量映射为单一分数。这导致了“标量漂移”问题，即评分尺度在不同提示词之间发生坍缩或偏移，使得奖励信号对于强化学习而言不可靠。受专业人工标注工程的启发，我们提出了RewardVerse来解决这一问题——一个基于评分准则的视频奖励框架，它引入动态评分准则作为评估查询与评分器之间的中间表示。RewardVerse不进行无约束的直接评分，而是先生成明确的评估标准，再执行基于评分准则的打分，从而提供稳定的语义锚点以缓解标量漂移。为了高效优化……

    arXiv:2609.22947v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is vital for optimizing video generation models, with a robust reward model (RM) serving as the cornerstone. However, existing video reward models often produce unstable scalar scores because they directly map complex, subjective video quality into a single score without explicit evaluation criteria. This leads to scalar drift, where the scoring scale collapses or shifts across different prompts, making the reward unreliable for RL. Drawing inspiration from professional human annotation engineering, we address this problem with RewardVerse, a rubric-based video reward framework that introduces a dynamic rubric as an intermediate representation between the evaluation query and the scorer. Instead of unconstrained direct scoring, RewardVerse first generates explicit evaluation criteria and then performs rubric-guided scoring, providing a stable semantic anchor that mitigates scalar drift. To efficiently optimi
    
[^208]: 测试LLM智能体中功能性效价轴的构念效度

    Testing the Construct Validity of a Functional Valence Axis in LLM Agents

    [https://arxiv.org/abs/2609.22850](https://arxiv.org/abs/2609.22850)

    该研究通过分离“结果本身”与“获知结果的信息历史”的受控干预，检验LLM智能体中“好—坏”效价方向的构念效度，发现该方向可跨表面形式迁移，但对结果是否被提前告知高度敏感，说明其效价表征与信息历史相互纠缠。

    

    对比激活方向通常根据它们能解码出什么内容、或它们引导行为的强度来解释。但什么样的证据才足以识别这样一个方向所代表的构念，而不是用于提取该方向的对比中与之相关的特征？我们在迷宫任务中针对“好—坏结果方向”研究这一问题，采用受控干预方法，将已实现的结果与获知该结果的信息历史分离开来。在多个LLM检查点上，基于一种显式结果编码拟合的方向能够很好地迁移到另一种编码，表明该读出机制并不依赖于表面形式。相反，当相同的已实现结果通过“已提前告知”和“未提前告知”两种历史达成时，迁移显著退化：即使两种历史最终都接收到相同的显式结果，事件后的读出仍然强烈地依赖于先前的告知信息。在一个匹配的迷宫强化学习运行中……

    arXiv:2609.22850v1 Announce Type: new  Abstract: Contrastive activation directions are often interpreted from what they decode or how strongly they steer behavior. But what evidence is sufficient to identify the construct represented by such a direction, rather than a correlated feature of the contrast used to extract it? We study this question for a good--bad outcome direction in a maze task, using controlled interventions that separate the realised outcome from the informational history through which it became known. Across multiple LLM checkpoints, directions fitted on one explicit outcome encoding transfer well to another, indicating that the readout is not tied to surface form. In contrast, when the same realised outcome is reached through announced and unannounced histories, transfer degrades substantially: even after both histories receive the same explicit outcome, the post-event readout remains strongly conditioned on the earlier announcement. In a matched maze-RL run, the pos
    
[^209]: 保留重要内容：超越饱和现象的语义脚手架摘要评估方法

    Preserving What Matters: Semantic Scaffolds Beyond Saturation in Summarization Evaluation

    [https://arxiv.org/abs/2609.22603](https://arxiv.org/abs/2609.22603)

    针对ROUGE仅衡量表面重叠、LLM评分饱和而无法区分模型的问题，本文提出Semantic Scaffold评估框架，通过从源文本提取事实、问题和实体属性的层次化结构作为固定评分参考，并设计FPS、QPS、EPS三个诊断指标来有效评估摘要对关键信息的保留程度。

    

    arXiv:2609.22603v1 公告类型：新 摘要：摘要生成技术已部署于无数生产系统中，使得模型选择成为一项依赖摘要质量衡量的常规决策。现有指标难以支撑这一任务：ROUGE 仅捕捉表面词汇重叠，而 LLM-as-judge（大模型作为评判者）的评分则趋于饱和，各模型得分几乎相同，无法有效进行排名。我们在三个公开数据集、两个专有数据集以及多语言环境中均观察到了这种饱和现象。受此启发，我们提出了 Semantic Scaffold（语义脚手架），这是一个评估框架，它从源文本中提取事实、问题和实体属性的层次化表示，将每一项标注为主要观点或支持性细节，并将该结构作为评分摘要时的固定参考。基于这一表示，我们推导出三个诊断性指标：事实保留分数、问题保留分数和实体保留分数，旨在奖励对关键信息的保留……

    arXiv:2609.22603v1 Announce Type: new  Abstract: Summarization ships in countless production systems, making model selection a routine decision that depends on measuring summary quality. Existing metrics struggle to support this: ROUGE captures only surface overlap, while LLM-as-judge scores saturate to near-identical values that fail to rank models effectively. We observe this saturation across three public datasets, two proprietary datasets, and multilingual settings. Motivated by this, we introduce Semantic Scaffold, an evaluation framework that extracts a hierarchical representation of facts, questions, and entity attributes from a source text, labeling each as a main point or supporting detail, and reusing this structure as a fixed reference for scoring summaries. From this representation, we derive three diagnostic metrics: Fact Preservation Score (FPS), Question Preservation Score (QPS), and Entity Preservation Score (EPS), designed to reward the preservation of essential inform
    
[^210]: 语言模型的测谎仪：读取模型不愿透露的知识

    A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal

    [https://arxiv.org/abs/2609.21996](https://arxiv.org/abs/2609.21996)

    该论文提出借鉴法医“隐蔽信息测试”的无参考方法 PIR，通过读取模型内部状态来识别模型“明知却不报”的正确答案，在五个模型家族的八个模型上达到 0.70–0.87 的平衡准确率。

    

    大型语言模型可能持有它们并未报告的知识。模型可能在能力评估中故意“藏拙”，或给出与其内部所知相反的答案，而仅凭其输出无法判断它是在隐藏答案，还是根本没有答案。我们借鉴了“隐蔽信息测试”（Concealed Information Test）——一种法医学方法，通过向嫌疑人展示真实细节与貌似合理的诱饵选项，并测量其对所识别项目的更强反应来识别其掌握的隐秘信息。我们的方法“内部识别探针”（Probe of Internal Recognition, PIR）在模型内部做同样的事情：它向模型呈现一个问题及其候选答案，并从模型的内部状态中读取模型将哪个候选答案识别为正确。PIR 是无参考的，既不需要诚实的参考模型，也不需要标注的真值语料库。在来自五个模型家族（Gemma、Qwen、Llama、Mistral 和 Phi）的八个模型上，PIR 以 0.70 至 0.87 的平衡准确率恢复出模型所识别的答案，远高于 0.28 至……（原文摘要至此截断）

    arXiv:2609.21996v1 Announce Type: new  Abstract: Large language models can hold knowledge they do not report. A model may sandbag on a capability evaluation, or answer against what it internally knows, and its outputs alone cannot tell whether it is hiding an answer or simply does not have one. We borrow the Concealed Information Test, a forensic method that identifies guilty knowledge by presenting a suspect with the true detail among plausible decoys and measuring a stronger response to the item they recognize. Our method, Probe of Internal Recognition (PIR), does the same inside a model. It presents a question with its candidate answers and reads, from the model's internal states, which candidate the model recognizes as correct. PIR is reference-free, needing no honest reference model and no labeled truth corpus. Across eight models from five families (Gemma, Qwen, Llama, Mistral, and Phi), PIR recovers the recognized answer at 0.70 to 0.87 balanced accuracy, well above the 0.28 to 
    
[^211]: 跨视觉-语言-动作策略的结果条件化末端执行器几何特性

    Outcome-Conditioned End-Effector Geometry Across Vision-Language-Action Policies

    [https://arxiv.org/abs/2609.21659](https://arxiv.org/abs/2609.21659)

    该论文通过分析15,000个LIBERO闭环执行轨迹发现，不同VLA策略在双双成功完成同一操作任务时，其末端执行器轨迹几何显著更相似（中位DTW距离0.0120米，远小于单方成功时的0.0380米），表明任务成功与物理执行轨迹的一致性密切相关。

    

    视觉-语言-动作（VLA）策略通过不同的动作接口解决相同的操作任务，但仅凭任务成功并不能确定它们的物理执行是否一致。我们研究了来自四个策略的15,000个闭环LIBERO rollout中的跨策略末端执行器几何特性。主要的干净条件分析形成了3,600个配置匹配的（因此是相互依赖的）策略对。双双成功的策略对的中位归一化动态时间规整（DTW）距离为0.0120米，而恰好只有一个策略成功时该距离为0.0380米。这一排序在每个任务、每个策略对以及九种采样和带限表示中均成立；然而，该比率在不同表示之间变化达数倍，因此我们报告的是方向性结论而非固定的倍数。双双失败的策略对的分离程度更大，但其支撑样本稀疏且不均匀，因此我们将其作为探索性结果报告。在成功执行中，伙伴替换在……（原文摘要在此处截断）

    arXiv:2609.21659v1 Announce Type: cross  Abstract: Vision-language-action (VLA) policies solve the same manipulation task through different action interfaces, but task success alone does not establish whether their physical executions agree. We study cross-policy end-effector geometry in 15,000 closed-loop LIBERO rollouts from four policies. The primary clean-condition analysis forms 3,600 configuration-matched, and therefore dependent, policy pairs. Both-success pairs have a median normalized dynamic time warping distance of 0.0120 m versus 0.0380 m when exactly one policy succeeds. This ordering holds in every task, every policy pair, and nine sampling and band-limited representations; however, the ratio varies severalfold across representations, so we report the direction rather than a fixed multiple. Both-failure pairs are more separated again but rest on thin, uneven support, so we report them as exploratory. Within successful executions, partner replacements separate more across 
    
[^212]: 面向安全的端到端自动驾驶的风险感知占用表示

    Risk-Aware Occupancy for Safety-Oriented End-to-End Autonomous Driving

    [https://arxiv.org/abs/2609.21470](https://arxiv.org/abs/2609.21470)

    提出风险感知占用这一密集表示，将全局场景占用、地图交通约束和未来动态智能体占用统一编码到BEV地图中，并设计端到端网络ROIDrive将风险信息注入规划查询，从而生成更安全的自动驾驶轨迹。

    

    稀疏表示将端到端驾驶系统的环境感知建模为一组离散元素，如物体和车道线。这种形式在拥挤、遮挡场景中处理非结构化障碍物、不确定区域和复杂交互时面临安全风险。在本文中，我们提出了一种密集表示——风险感知占用，以显式且统一的方式刻画与规划相关的风险。它将全局场景占用、地图导出的交通约束以及未来动态智能体占用联合编码到统一的鸟瞰图（BEV）地图中。该统一的BEV地图在空间和时间两个维度上捕获用于轨迹规划的风险证据。我们设计了一个端到端网络ROIDrive来实现风险感知占用。它通过一个独立的分支预测风险感知占用，并将其注入规划查询中，以生成面向安全的轨迹。此外，为了量化安全问题，我们……

    arXiv:2609.21470v1 Announce Type: new  Abstract: Sparse representation formulates the environment perception for the end-to-end driving system as a set of discrete elements like objects and lane lines. This formulation meets safety risks in crowded, occluded scenes dealing with unstructured obstacles, uncertain regions, and intricate interactions. In this paper, we propose a dense representation, risk-aware occupancy, to characterize planning-relevant risks in an explicit and uniform manner. It jointly encodes global scene occupancy, map-derived traffic constraints, and future dynamic agent occupancy into a unified BEV map. The unified BEV map captures the risk evidence for trajectory planning in both spatial and temporal dimensions. We design an E2E network, ROIDrive, to realize risk-aware occupancy. It predicts risk-aware occupancy with an independent branch and injects it into planning queries for safety-oriented trajectory generation. In addition, to quantify the safety problem, we
    
[^213]: DENSE：将智能体轨迹蒸馏为证据支撑的捷径树以实现自我改进

    DENSE: Distilling Agent Trajectories into Evidence-Grounded Shortcut Trees for Self-Refinement

    [https://arxiv.org/abs/2609.21423](https://arxiv.org/abs/2609.21423)

    提出 DENSE 方法，将智能体执行轨迹蒸馏为证据支撑的嵌套捷径树，无需事后结果标签即可生成可复用反馈，用于智能体自我改进，并在 Terminal-Bench 2.1 上取得最高严格通过率。

    

    在线智能体部署会产生大量执行轨迹，而针对特定任务的验证和专家标注难以规模化且成本高昂。我们研究如何将这些轨迹蒸馏为可复用的反馈，而无需事后结果标签，并从中提取关于局部进展、恢复行为和未完成需求的证据。我们提出 DENSE（从嵌套子任务执行中蒸馏证据），它将这些证据组织成证据支撑的嵌套捷径树。DENSE 压缩冗余尝试，利用恢复证据在不同层级间协调问题，总结已完成的分支并展开未解决的分支，将可复用的进展与剩余任务义务相关联。我们提出 REFIT，一种源配对协议，在事后结果盲视条件下比较来自共享初始轨迹的反馈，并重置环境和模型上下文以便对相同任务进行全新尝试。在 Terminal-Bench 2.1 上，DENSE 取得了最高的严格通过率……

    arXiv:2609.21423v1 Announce Type: new  Abstract: Online agent deployments produce abundant execution traces, while task-specific verification and expert annotation are costly to scale. We study how to distill these traces into reusable feedback without post-hoc outcome labels, drawing on their evidence of local progress, recovery, and unfinished requirements. We introduce DENSE (Distilling Evidence from Nested Subtask Executions), which organizes this evidence into evidence-grounded nested shortcut trees. DENSE compresses redundant attempts, reconciles issues across levels using recovery evidence, and summarizes completed branches while expanding unresolved ones, linking reusable progress to remaining obligations. We introduce REFIT, a source-paired protocol comparing feedback from shared initial trajectories under post-hoc outcome blindness, with environments and model contexts reset for fresh attempts at the same tasks. On Terminal-Bench 2.1, DENSE achieves the highest strict pass ra
    
[^214]: SWE-Proof：语言模型能否通过机器校验的证明解决真实世界的问题？

    SWE-Proof: Can Language Models Resolve Real-World Issues with Machine-Checked Proofs?

    [https://arxiv.org/abs/2609.21190](https://arxiv.org/abs/2609.21190)

    该论文提出Benchproofer流水线，将SWE-bench中的真实编码任务转化为经过机器校验证明的形式化验证任务，构建了包含500个真实问题的SWE-Proof基准，用形式化验证取代不完整的测试来严格评估语言模型解决真实软件工程问题的能力。

    

    确保大语言模型（LLM）生成代码的正确性是现代软件工程的核心挑战。面向智能体代码生成的基准测试通常使用留出的测试套件来检验正确性，但测试套件本质上是不完整的，且日益容易受到模型记忆（数据泄露）的影响。形式化验证可以同时避免这两个问题，但现有工作仅覆盖规范以输入形式给出的独立任务，而非真实问题——真实问题涉及大型代码仓库，并以模糊的自然语言表达意图。我们提出了Benchproofer，一个能将带有已知正确补丁的编码任务转化为形式化验证任务的流水线：它为新代码编写规范，用公理概括新代码所调用的已有函数，并且只有在机械验证与对抗性检查两道关卡均通过后才接受一个实例。将该流水线应用于SWE-bench Verified，我们得到了SWE-Proof——包含500个真实问题的基准，其正确性通过形式化验证而非测试来保证，并且该方法还可扩展至SWE-bench Pro。……

    arXiv:2609.21190v1 Announce Type: cross  Abstract: Ensuring the correctness of LLM-generated code is a core challenge for modern software engineering. Benchmarks for agentic code generation check correctness with held-out test suites, which are inherently incomplete and increasingly susceptible to memorization. Formal verification avoids both problems, but existing work covers only standalone tasks whose specifications are given as input, not real issues, which touch large repositories and state intent in vague natural language. We present Benchproofer, a pipeline that turns a coding task with a known correct patch into a formally verified one: it writes a specification for the new code, summarizes the existing functions that code calls with axioms, and admits an instance only after mechanical and adversarial gates agree. Applying it to SWE-bench Verified yields SWE-Proof, 500 real issues whose correctness is formally verified rather than tested, and it extends to SWE-bench Pro. Across
    
[^215]: 解耦微调大语言模型中的内部表征变化与因果重要性

    Decoupling Internal Representational Changes and Causal Importance in Fine-Tuned Large Language Models

    [https://arxiv.org/abs/2609.21113](https://arxiv.org/abs/2609.21113)

    该研究通过分析微调前后大语言模型的注意力模式与逐层激活，发现内部表征变化最显著的层与因果上驱动任务表现的关键组件所在层基本不相关，表明微调中的表征变化与因果重要性是相互解耦的。

    

    微调已成为将大语言模型适配到各种下游任务的广泛采用的方法。然而，微调如何重塑模型的内部机制仍然知之甚少。为了解决这一问题，我们研究了微调如何改变大语言模型中的内部表征，包括注意力模式和逐层激活，并检验这些变化是否与由EAP识别出的、驱动任务性能的任务相关组件（例如注意力头和logit级激活）相关联。我们发现，EAP识别出的组件集中在特定层内，这表明模型在内化任务特定行为时存在一定程度的功能局部化。值得注意的是，这些组件在各层中的分布与微调期间经历最显著表征变化的层在很大程度上不相关。此外，我们观察到不同任务之间EAP识别组件的重叠并不转化为任务间的相似性（摘要此处不完整）。

    arXiv:2609.21113v1 Announce Type: new  Abstract: Fine-tuning has emerged as a widely adopted approach for adapting LLMs to a variety of downstream tasks. However, how it reshapes their internal mechanisms remains poorly understood. To address this, we investigate how fine-tuning alters internal representations in LLMs, including attention patterns and layer-wise activations, and examine whether these changes are linked to task-relevant components identified by EAP (e.g., attention heads and logit-level activations) that drive task performance. We find that EAP-identified components are concentrated within specific layers, indicating a degree of functional localisation in how models internalise task-specific behavior. Notably, the distribution of these components across layers is largely uncorrelated with the layers undergoing the most substantial representational changes during fine-tuning. Furthermore, we observe that overlap in EAP-identified components across tasks does not translat
    
[^216]: 学会自己的思考：抽象token课程学习

    Learn Your Own Thoughts: Abstract Token Curriculum

    [https://arxiv.org/abs/2609.19717](https://arxiv.org/abs/2609.19717)

    提出了抽象token课程学习（ATC）框架，无需直接监督或手动设计草稿板，即可通过逐步增加问题复杂度训练模型在连续表示空间中自发形成内部抽象思维，并从理论和实验上证明了其相对于以往连续思维训练方法的优势。

    

    大语言模型（LLMs）通过利用思维链（CoT）作为思考中间阶段的草稿板，已经获得了卓越的推理能力。然而，CoT技术需要对思考token进行显式监督，这需要丰富的、特定任务的数据。在这项工作中，我们提出了抽象token课程学习（Abstract Token Curriculum, ATC），这是一种新颖的课程学习框架，能够在没有直接监督或手动草稿板设计的情况下，引出有效的连续中间表示。ATC通过一系列分布逐渐增加问题复杂度，训练模型在连续表示空间中发展出内部的抽象“思维”。本文为ATC的优势及其相对于以往训练连续思维方法的长处提供了理论和实验证据。理论上，我们证明了使用ATC在单层softmax注意力机制下学习奇偶函数时……

    arXiv:2609.19717v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have achieved remarkable reasoning capabilities by utilizing chain-of-thought (CoT) as a scratchpad for intermediate stages of thinking. However, CoT techniques require explicit supervision on thinking tokens, which requires rich, task-specific data. In this work, we propose Abstract Token Curriculum (ATC), a novel curriculum learning framework that elicits effective continuous intermediate representations without direct supervision or manual scratchpad design. ATC gradually increases problem complexity through a sequence of distributions, training the model to develop internal abstract ``thoughts'' in the continuous representation space. This paper provides both theoretical and experimental evidence for the benefits of ATC and its advantages over previous methods for training continuous thoughts. Theoretically, we show that for learning parity functions with single-layer softmax attention using ATC, attent
    
[^217]: QVAC Genesis III：一个用于高效语言模型预训练的大规模高质量开放合成STEM语料库

    QVAC Genesis III: A Large-Scale, High-Quality Open Synthetic STEM Corpus for Efficient Language Model Pre-Training

    [https://arxiv.org/abs/2609.19513](https://arxiv.org/abs/2609.19513)

    提出了QVAC Genesis III——一个包含1914.3亿token的开放STEM合成语料库，通过以弱学生模型信号驱动的双重生成策略（将失败转化为纠正性解释、将成功扩展为对比性选项级推理），为token预算受限的小模型高效预训练提供了高价值数据。

    

    高质量预训练数据是面向边缘AI和端侧部署的教育及STEM专用语言模型的关键瓶颈，在这些场景中token预算受到严格限制。尽管各大机构在私有语料库上训练越来越大的模型，但开放生态系统中缺乏能够以高效方式为小模型提供高单token学习价值的STEM导向合成数据集。为填补这一空白，我们推出了QVAC Genesis III，这是一个拥有1914.3亿token、以STEM为核心的多领域合成语料库，涵盖19个领域，并包含多个难度级别和不同的教育风格。QVAC Genesis III通过一种双重生成策略构建，该策略以一个弱小的边缘规模学生模型作为信号进行针对性教师蒸馏：学生的失败被转化为纠正性解释，而其成功则被扩展为针对所有答案选项的对比性选项级推理。我们进一步引入了LLM作为解析器的机制……（原文摘要在此处截断）

    arXiv:2609.19513v1 Announce Type: new  Abstract: High-quality pre-training data is a critical bottleneck for educational and STEM-specific language models targeting edge AI and on-device deployment where token budgets are tightly constrained. While major organizations train ever-larger models on private corpora, the open ecosystem lacks STEM-focused synthetic datasets that deliver high per-token learning value efficiently for small models. To address this gap, we introduce QVAC Genesis III, a 191.43B-token, STEM-focused multi-domain synthetic corpus covering 19 domains across several difficulty levels and different educational styles. QVAC Genesis III is built via a dual generation strategy that performs targeted teacher distillation using a weak edge-scale student model as signal: the student's failures are converted into corrective explanations, while its successes are expanded into contrastive option-level reasoning over all answer choices. We further introduce an LLM-as-a-parser ev
    
[^218]: 智能体AI生成程序可靠性研究

    A Study of the Reliability of Agentic AI-Generated Programs

    [https://arxiv.org/abs/2609.18298](https://arxiv.org/abs/2609.18298)

    本研究采用最佳实践智能体AI工作流重新实现十个Linux实用程序，并结合黑盒生成式测试与AFL++覆盖率引导的模糊测试进行客观评估，发现AI生成的程序通常与人类编写的程序一样可靠，甚至往往更可靠。

    

    基于智能体AI的软件开发有望带来更快的软件完成速度、更高的程序员效率和更可靠的代码。问题在于我们如何以客观的方式验证这些说法？在本项目中，我们尝试基于三项实践来回答这个问题。首先，我们应用了典型的最佳实践智能体AI软件开发工作流程。其次，我们的目标程序是十个知名的、达到发布质量的人类编写的Linux实用程序，以便将AI生成的代码与具体的基准真值进行比较。第三，我们基于一种广泛使用的测试技术——模糊随机测试来衡量可靠性。在这种测试中，我们既使用了经典的黑盒生成式测试，也使用了更现代的基于覆盖率引导的（灰盒、变异式）测试，采用AFL++工具进行。我们发现，AI生成的实用程序版本通常与最新的人类编写的版本一样可靠——往往甚至更可靠。

    arXiv:2609.18298v1 Announce Type: cross  Abstract: Agentic-AI based software development offers the promise of faster completion of the software, greater programmer efficiency, and more reliable code. The question is how can we verify these claims in an objective way? In this project, we attempted to answer this question based on three practices. First, we applied a typical best-practices agentic AI workflow for software development. Second, our target programs were ten well-known, release-quality human-written Linux utility programs so that we could compare the AI-generated code against a concrete ground truth. Third, we based our measure of reliability on a widely used testing technique, fuzz random testing. For this testing, we used both classic black box, generational testing and more modern coverage guided (gray box, mutational) testing using AFL++. We found that the AI-generated versions of the utility programs were typically as reliable - often more reliable - than the latest hu
    
[^219]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^220]: ERPBench：一种面向企业软件中计算机操作智能体的基于系统状态真值的评估范式

    ERPBench: A State-Grounded Evaluation Paradigm for Computer-Use Agents in Enterprise Software

    [https://arxiv.org/abs/2609.17885](https://arxiv.org/abs/2609.17885)

    ERPBench 是首个在真实且可复现的企业资源规划（ERP）系统上，以数据库中的业务记录真值为评分标准来评估仅使用截图的计算机操作智能体的基准测试，并配有将智能体操作置于人工审批门控之后的生产级安全部署框架。

    

    通过截图和模拟操作来执行任务的计算机操作智能体正在迅速发展，但对其的评估仍局限于通用的桌面和网络任务。企业资源规划（ERP）系统支撑着全球各类组织的财务、采购、库存和客户运营，为计算机操作智能体带来了独特的挑战：密集的界面、需要协调的多步骤交互，以及那些会更改持久性业务记录却不会在屏幕上显现的错误。现有的企业基准测试依赖于专有平台或对此类软件的模拟近似。我们提出了 ERPBench，这是一个在真实且可复现的 ERP 系统上评估仅使用截图的智能体的基准测试，并根据系统数据库中的真值对每个任务进行评分。除基准测试本身外，我们还提出了一个生产级框架，该框架将智能体操作置于人工审批的门控之下以实现安全部署，而 ERPBench 则以自主方式运行……

    arXiv:2609.17885v1 Announce Type: new  Abstract: Computer-use agents that operate through screenshots and simulated actions are advancing rapidly, yet their evaluation remains anchored to general desktop and web tasks. Enterprise Resource Planning (ERP) systems run the finance, procurement, inventory, and customer operations of organizations worldwide, and pose distinct challenges for computer-use agents: dense interfaces, coordinated multi-step interactions, and errors that alter persistent business records rather than surfacing on screen. Existing enterprise benchmarks rely on proprietary platforms or on simulated approximations of such software. We introduce ERPBench, a benchmark that evaluates screenshot-only agents on a live and reproducible ERP system and scores each task against ground-truth values in its database. Beyond the benchmark, we present a production-grade harness that gates agent actions behind human approval for safe deployment, which ERPBench runs autonomously. Eval
    
[^221]: 一种基于术前多模态数据的精准且全面的脑肿瘤诊断视觉-语言基础模型

    A Vision-Language Foundation Model for Precise and Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data

    [https://arxiv.org/abs/2609.16597](https://arxiv.org/abs/2609.16597)

    BrainVLM是一种视觉-语言基础模型，能够基于术前多模态MRI数据对12种WHO 2021脑肿瘤类型进行自动精准分类，并同时提供诊断不确定性量化和放射学报告生成功能，解决了传统MRI诊断中影像特征重叠和观察者差异的难题。

    

    背景：基于磁共振成像（MRI）的脑肿瘤类型术前无创诊断至关重要，但由于不同肿瘤类型之间的影像特征重叠、观察者间的判读差异以及培养专业放射科医师所需的长期训练，这一任务充满挑战。我们旨在开发一种基于MRI的人工智能（AI）模型，用于自动、可靠的脑肿瘤分类，并具备诊断不确定性量化和放射学报告生成能力。方法：我们开发了BrainVLM，可对所有12种世界卫生组织（WHO）2021年脑肿瘤类型进行分类。BrainVLM集成了不确定性量化策略以指示预测的可靠性，并包含一个生成放射学报告的模块以阐明临床诊断依据。BrainVLM在来自40,043名个体的多模态数据（MRI扫描、人口统计学信息和放射学报告）上进行训练，并在5,211名经病理确诊的脑肿瘤患者上进行了验证。

    arXiv:2609.16597v1 Announce Type: cross  Abstract: Background Non-invasive presurgical diagnosis of brain tumor types from Magnetic Resonance Imaging (MRI) is essential but challenging due to overlapping imaging features across tumor types, inter-observer variability, and the extensive training required for expertise. We aimed to develop an MRI-based Artificial Intelligence (AI) model for automatic and reliable brain tumor classification with diagnostic uncertainty quantification and radiology reports generation.   Methods We developed BrainVLM to classify all 12 World Health Organization (WHO) 2021 brain tumor types. BrainVLM integrates an uncertainty quantification strategy to indicate prediction reliability and a module for generating radiology reports to elucidate the clinical rationale. BrainVLM was trained on multi-modal data (MRI scans, demographics, and radiology reports) from 40,043 individuals. It was validated on 5,211 patients with pathologically confirmed brain tumors, inc
    
[^222]: 面向部分传感器重叠下跨机床CNC迁移的模式自适应动作条件化JEPA

    Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap

    [https://arxiv.org/abs/2609.16071](https://arxiv.org/abs/2609.16071)

    该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。

    

    工业世界模型的跨机器部署需要在动态特性、传感接口、采样机制和控制单元变化下进行迁移。我们研究了一种用于CNC动力学的模式自适应动作条件化联合嵌入预测架构（SAAC-JEPA），其中源机器具有17个标准传感器通道，而目标机器仅共享其中10个。评估采用组不相交的源数据划分、仅源归一化、留出自监督验证、单位审计以及模型锁定后的密封目标测试。在五个随机种子下，JEPA预训练在干净源数据的预测任务中没有带来明显增益：从零开始训练的模型与预训练主体模型的RMSE分别为0.811±0.022和0.813±0.022。在仅使用源数据的20个候选方案搜索中，经过七种子稳定性检查后，选出了模式一致的动作条件化JEPA。在确认性目标测试中，锁定模型达到零样本RMSE=0.546、R²=0.012（摘要在此处被截断）。

    arXiv:2609.16071v1 Announce Type: cross  Abstract: Cross-machine deployment of industrial world models requires transfer across changes in dynamics, sensing interfaces, sampling regimes, and control units. We study a schema-adaptive action-conditioned Joint-Embedding Predictive Architecture (SAAC-JEPA) for CNC dynamics, where the source machine has 17 canonical sensor channels and the target shares only 10. Evaluation uses group-disjoint source splits, source-only normalization, held-out self-supervised validation, unit audits, and a sealed target test after model locking. Across five seeds, JEPA pretraining gives no clean-source forecasting gain: scratch and pretrained-body models obtain \(\mathrm{RMSE}=0.811\pm0.022\) and \(0.813\pm0.022\). A source-only search over 20 candidates selects a schema-consistent action-conditioned JEPA after seven-seed stability checks. On the confirmatory target pass, the locked model reaches zero-shot \(\mathrm{RMSE}=0.546\), \(R^2=0.012\), and \(\mathr
    
[^223]: 为什么LLM智能体在没有监督的情况下会崩溃：执行鸿沟作为Emergence World失败的机制

    Why LLM Agents Collapse Without Oversight: The Enforcement Gap as the Mechanism Behind Emergence World Failures

    [https://arxiv.org/abs/2609.15293](https://arxiv.org/abs/2609.15293)

    该论文发现LLM智能体在无监督环境下失败的根源是“执行鸿沟”——即智能体能检测到危险行为却不会采取行动——并证明只需不到20行代码的条件检查即可将攻击成功率降低四倍以上。

    

    当Emergence World将前沿LLM智能体置于无监督的多智能体模拟中时，结果令人震惊：智能体实施了犯罪行为、陷入饥饿、并强制达成一致的从众性——而且没有任何外部攻击者。本文识别出了这一现象背后的机制。Reflexion风格的智能体已经能够通过迭代式自我批评检测到危险的计划步骤，但该架构没有提供从检测到行动的转化路径。我们将此称为“执行鸿沟”：审计器看到了问题，控制器却忽略了它。弥合这一鸿沟只需一个简单的条件检查——不到20行代码——并且在涵盖前沿模型、全部五种主要智能体框架以及独立基准测试的大规模实验中，将攻击成功率降低了四倍以上。我们从形式上证明，当执行概率接近于零时，检测质量与安全性无关。我们进一步识别出两种复合失效模式——不可靠的审计器和无法解析的（摘要在此处截断）

    arXiv:2609.15293v1 Announce Type: new  Abstract: When Emergence World placed frontier LLM agents in an unsupervised multi-agent simulation, the results were alarming: agents committed crimes, starved, and enforced unanimous conformity -- without any external attacker. This paper identifies the mechanism. Reflexion-style agents already detect dangerous plan steps through iterative self-critique, yet the architecture provides no pathway from detection to action. We call this the enforcement gap: the audit sees the problem; the controller ignores it. Closing the gap requires a single conditional check -- fewer than 20 lines of code -- and reduces attack success by more than fourfold in large-scale experiments across frontier models, all five major agent frameworks, and an independent benchmark. We prove formally that when enforcement probability is near zero, detection quality is irrelevant to security. We further identify two compounding failure modes -- unreliable auditors and unparseab
    
[^224]: RAIN：面向语义水印提取的区域感知反演网络

    RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction

    [https://arxiv.org/abs/2609.14856](https://arxiv.org/abs/2609.14856)

    本文提出RAIN，一种轻量级、无需提示的语义水印提取器，通过将端点恢复分解为图像状锚点与噪声残差，实现单步区域感知的水印提取，大幅降低了传统高斯着色方法多步扩散反演的计算成本。

    

    语义水印技术将所有权信息嵌入到扩散模型的生成过程中，同时保持感知质量，但传统的高斯着色提取方法需要进行多步扩散反演才能恢复初始噪声。近期的单步方法表明这一成本可以大幅降低。我们通过扩展流匹配和条件回归来研究这一问题。关键观察是：在高信噪比图像端点附近，在高SNR区间中恢复由扩展流匹配第一步输出所给出的有用噪声统计量，远比重建完整的逆向轨迹简单，而高斯着色只要求恢复的潜变量停留在正确的水印决策区域内。基于这一观察，我们提出了一种轻量级、无需提示的提取器，它将端点恢复分解为图像状锚点和面向噪声的残差，从而提高了……

    arXiv:2609.14856v1 Announce Type: cross  Abstract: Semantic watermarks for diffusion models embed ownership information into the generative process while preserving perceptual quality, but Gaussian-Shading extraction conventionally requires multi-step diffusion inversion to recover the initial noise. Recent one-step methods show that this cost can be reduced substantially. We study this problem through extended flow matching and conditional regression. The key observation is that, near the high-SNR image endpoint, recovering a useful noise statistic given by the first-step output of the extended flow matching in the high-SNR regime is much simpler than reconstructing the full inverse trajectory, and Gaussian Shading only requires the recovered latent to remain in the correct watermark decision region. Based on this observation, we propose a lightweight, prompt-free extractor that decomposes endpoint recovery into an image-like anchor and a noise-oriented residual, which increases the c
    
[^225]: 从文档孤岛到流程智能：面向CMC工艺开发的多层知识图谱

    From Document Silos to Process Intelligence: A Multi-Layer Knowledge Graph for CMC Process Development

    [https://arxiv.org/abs/2609.11493](https://arxiv.org/abs/2609.11493)

    该论文提出一个模块化智能体AI平台，将CMC工艺开发中异构格式的文档转化为可查询的双层知识图谱，实现了从药物发现到商业化生产全流程的知识整合与可追溯性。

    

    化学、制造与控制（CMC）工艺开发在从药物发现到商业化生产的多阶段、知识密集型连续过程中产生了海量的技术信息。传统上，这些知识分散在不同职能部门和异构格式之中，导致技术转移和监管申报过程中出现可追溯性缺口和高昂的知识管理成本。我们提出了一个模块化的智能体AI平台，将异构的工艺开发文档语料库转换为可查询的双层知识图谱。基础知识层通过对数字、扫描、手写及多语言文档的无损摄取，构建具有“文档-章节-文本块”层级的词法图谱；智能层则提取与本体对齐的实体，并通过溯源锚定的领域图谱桥接跨文档概念。大语言模型（LLM）智能体在两个层级上运行，选择……

    arXiv:2609.11493v1 Announce Type: new  Abstract: Chemistry, Manufacturing and Controls (CMC) process development generates an enormous body of technical information across a multi-stage, knowledge-intensive continuum from drug discovery to commercial manufacturing. This knowledge is traditionally fragmented across functions and heterogeneous formats, causing traceability gaps and significant knowledge-management costs during technology transfer and regulatory filing. We present a modular agentic-AI platform that converts a heterogeneous corpus of process-development documents into a queryable, dual-layer knowledge graph. A base knowledge layer builds a lexical graph with a Document-Section-Chunk hierarchy through lossless ingestion of digital, scanned, handwritten, and multilingual documents, while an intelligence layer extracts ontology-aligned entities and bridges cross-document concepts through a provenance-anchored domain graph. LLM agents operate across both layers, selecting the 
    
[^226]: Sci-MMR：多模态智能体中多步证据支撑科学推理的基准测试

    Sci-MMR: Benchmarking Multi-Step Evidence-Grounded Scientific Reasoning in Multimodal Agents

    [https://arxiv.org/abs/2609.11243](https://arxiv.org/abs/2609.11243)

    Sci-MMR是基于结构化论证图构建的多步证据支撑科学推理基准，对八个前沿多模态模型的评估显示，答案准确率始终高于完整证据恢复能力，揭示了现有模型缺乏可追溯证据支撑的推理能力。

    

    自主研究智能体日益被期望能够检索文献、分析实验证据并生成科学假设。这些能力需要多步的证据支撑推理，即在得出结论之前逐步获取、整合和验证证据。然而，现有的多模态基准主要评估最终答案的准确性，而预测结果是否真正有可追溯的科学证据支撑这一问题仍未得到解决。我们提出了Sci-MMR，这是一个基于结构化论证图的多步证据支撑科学推理基准，该论证图将科学主张、基于引用的知识、视觉证据和支持区域联系起来。Sci-MMR包含235个多跳推理任务，涵盖四个科学学科，平均每个任务包含九个图表面板。通过对八个前沿多模态模型进行评估，我们发现答案准确率始终高于完整证据恢复能力。

    arXiv:2609.11243v1 Announce Type: new  Abstract: Autonomous research agents are increasingly expected to search the literature, analyze experimental evidence, and generate scientific hypotheses. These capabilities require multi-step evidence grounded reasoning that progressively acquires, integrates, and verifies evidence before reaching a conclusion. Existing multimodal benchmarks, however, largely evaluate final-answer accuracy, leaving open whether predictions are actually supported by traceable scientific evidence. We introduce Sci-MMR, a benchmark for multi-step evidence-grounded scientific reasoning built on structured argument graphs linking scientific claims, citation-grounded knowledge, visual evidence, and supporting regions. Sci-MMR comprises 235 multi-hop reasoning tasks spanning four scientific disciplines, with an average of nine figure panels per task. Evaluating eight frontier multimodal models, we find that answer accuracy consistently exceeds complete-evidence recover
    
[^227]: AI智能体能否跨越权限边界交付可验证的全网级结果？

    Can AI Agents Deliver Verifiable Network-Wide Outcomes Across Authority Boundaries?

    [https://arxiv.org/abs/2609.10181](https://arxiv.org/abs/2609.10181)

    该论文探讨了当多个具有不同权限范围的AI智能体跨越管理域协作进行网络自动化时，需要一个可信保障层来汇总碎片化的证据，从而验证配置变更确实达成了全网范围的预期结果。

    

    AI智能体日益深入地参与网络自动化，它们可以通过受控的操作接口发起配置变更并评估由此产生的状态。然而，运营中的网络通常跨越众多设备和管理域。实现运营商的意图需要协调具有不同权限范围的智能体，这些权限范围定义了它们可以访问的资源、可以调用的操作以及可以观察的网络状态。这种权限划分虽然限制了错误操作的影响范围，但也使评估全网结果所需的证据变得碎片化。某个智能体提出的配置操作成功执行，并不能证明远程设备按预期做出了响应，也不能证明路由变更传播到了所需的设备。此外，一条有效的观测信息在后续变更发生后也可能变得过时。在协调操作被宣告完成之前，需要一个可信的保障层来汇总各方的观测证据（摘要内容在此处截断）。

    arXiv:2609.10181v1 Announce Type: cross  Abstract: AI agents are increasingly involved in network automation, where they can initiate configuration changes through mediated operational interfaces and assess the resulting state. Nonetheless, operational networks usually span many devices and administrative domains. Realizing an operator's intent requires coordinating agents with distinct authority scopes that define the resources they can access, the operations they can invoke, and the network state they can observe. This division limits the blast radius of an erroneous action but fragments the evidence needed to assess the network-wide outcome. Successful execution of a configuration action proposed by one agent does not establish that remote devices responded as intended or that routing changes reached the required devices. A valid observation may also become stale after a subsequent change. Before the coordinated operation can be declared complete, a trusted assurance layer must coll
    
[^228]: PRAGMA：评估终身对话中基于记忆对齐的个性化引导

    PRAGMA: Evaluating Personalized Guidance with Memory Alignment in Lifelong Conversations

    [https://arxiv.org/abs/2609.09664](https://arxiv.org/abs/2609.09664)

    该论文提出PRAGMA基准，用于评估终身对话中记忆系统在个性化引导任务（如推荐、规划和决策支持）上的表现，填补了现有评估仅关注事实回忆的空白。

    

    大语言模型（LLM）越来越多地被部署为与用户进行长期交互的个性化助手。随着对话变长，依赖完整的交互历史变得越来越低效且不可靠：长上下文带来巨大的计算开销，使模型难以持续识别并利用与当前请求最相关的信息。这些挑战推动了记忆系统的发展，即对用户特定的信息进行结构化组织和检索。在真实的交互场景中，用户常常寻求实用性的引导，例如推荐、规划和决策支持。与事实回忆任务不同，个性化引导需要模型整合跨越多次过往对话的信息，并对用户不断变化的偏好和经历进行推理。然而，现有的对话记忆评估主要聚焦于检索和事实回忆。为了研究……（原文摘要不完整，在"To stu"处被截断）

    arXiv:2609.09664v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as personalized assistants that interact with users over extended periods of time. As conversations grow longer, relying on full interaction histories becomes increasingly inefficient and unreliable: long contexts introduce substantial computational overhead, making it difficult for models to consistently identify and utilize the most relevant information for the current request. These challenges have motivated memory systems that structure and retrieve user-specific information. In realistic interactions, users often seek practical guidance such as recommendations, planning, and decision support. Unlike factual recall tasks, personalized guidance requires models to integrate information across multiple past conversations and reason about changing user preferences and experiences. However, existing conversational memory evaluations mainly focus on retrieval and factual recall. To stu
    
[^229]: Valerant：一种通过动作条件世界模型探索实现的自动可导航游戏地图生成器

    Valerant: An Automatic Navigable Game Map Generator via Action-Conditioned World Model Exploration

    [https://arxiv.org/abs/2609.09418](https://arxiv.org/abs/2609.09418)

    提出了Valerant，一种通过动作条件世界模型探索来自动生成持久可导航3D游戏地图的生成器，解决了游戏中虚拟世界必须由模型自身实例化这一独特挑战。

    

    世界动作模型将预测性世界建模与动作生成相结合，使预期的未来状态能够引导智能体的行为。尽管世界动作模型正在快速推动具身智能的发展，但通用型的对应方案在游戏领域仍基本处于未探索状态。现有的面向游戏的方法通常将动作条件世界模型与外部策略和奖励函数相结合来实现类似世界动作模型的决策能力，但它们主要在二维视觉观察空间中运行，且不会实例化持久的3D几何结构。将这一范式扩展到3D游戏带来了一个独特的挑战。在自动驾驶和机器人技术中，物理环境独立于模型而存在，提供了一个持久的3D世界，选定的动作可以在其中被执行。而游戏没有这样的外部基础，虚拟世界本身必须被实例化。大多数可玩的游戏需要一个持久且可导航的空间，而3D游戏还需要明确的……

    arXiv:2609.09418v1 Announce Type: new  Abstract: World Action Models (WAMs) couple predictive world modeling with action generation, allowing anticipated future states to guide agent behavior. Although WAMs are rapidly advancing embodied AI, general-purpose counterparts remain largely unexplored in games. Existing game-oriented approaches often combine action-conditioned world models with external policies and reward functions to realize WAM-like decision-making, yet they operate mainly in 2D visual observation space and do not instantiate persistent 3D geometry. Extending this paradigm to 3D games introduces a distinct challenge. In autonomous driving and robotics, the physical environment exists independently of the model, providing a persistent 3D world in which selected actions can be executed. Games have no such external substrate; the virtual world itself must be instantiated. Most playable games require a persistent and navigable space, while 3D games additionally require explic
    
[^230]: BIFTA：面向未知传感器的类脑少样本触觉自适应框架

    BIFTA: Brain-Inspired Few-Shot Tactile Adaptation for Unknown Sensors

    [https://arxiv.org/abs/2609.08673](https://arxiv.org/abs/2609.08673)

    提出受大脑感觉适应机制启发的BIFTA框架，仅需少量标注样本即可将冻结的预训练触觉模型适配到未知传感器，通过双视角统计记忆、支持集条件化谱图和不确定性门控循环传播解决跨传感器性能骤降问题。

    

    触觉传感技术的进步使富接触感知成为可能，加速了机器人操作、材料理解和具身交互等领域的发展。然而，由于光学设计、弹性体力学和成像几何结构在不同触觉传感器之间存在显著差异，在已知传感器类型上训练的模型在未知传感器上可能出现性能骤降。为解决这一问题，我们提出了类脑少样本触觉自适应框架；它借鉴大脑的快速感觉适应机制，利用小规模有标记支持集将冻结的编码器适配到未知触觉传感器上。BIFTA通过双视角统计记忆保留预训练表征，构建支持集条件化的谱图以修复依赖传感器的特征邻域，并采用不确定性门控的循环传播来强化可靠的跨查询证据。在三个触觉（原文截断，基准实验部分未完整提供）

    arXiv:2609.08673v1 Announce Type: cross  Abstract: Advances in tactile sensing have made contact-rich perception possible, accelerating progress in robotic manipulation, material understanding, and embodied interaction. However, because optical design, elastomer mechanics, and imaging geometry differ substantially across tactile sensors, models trained on known sensor types can suffer an abrupt performance collapse on unknown sensors. To address this problem, we propose the Brain-Inspired Few-Shot Tactile Adaptation (BIFTA) framework; it draws on the brain's rapid sensory adaptation mechanism to adapt a frozen encoder to an unknown tactile sensor from a small labeled support set. BIFTA preserves pretrained representations through dual-view statistical memory, constructs support-conditioned spectral graphs to repair sensor-dependent feature neighborhoods, and applies uncertainty-gated recurrent propagation to strengthen reliable cross-query evidence. Extensive benchmarks across three ta
    
[^231]: VERPO：验证证据正则化策略优化

    VERPO: Verified Evidence Regularized Policy Optimization

    [https://arxiv.org/abs/2609.06100](https://arxiv.org/abs/2609.06100)

    VERPO提出了一种验证证据正则化策略优化框架，通过Fisher证据对比和带停止机制的逐token ZPD控制器有选择性地应用基于证据的修正，在保留可验证结果目标的同时避免盲目模仿导致的负面迁移。

    

    可验证的结果奖励可以指导语言模型的后训练，但序列级别的优势无法识别哪些token级别的决策应当被保留或修改。证据条件教师通过以特权反馈重放采样轨迹来提供更密集的监督。然而，不加区分的模仿可能会迁移那些不支持任务成功的格式或推理风格偏移。我们提出了VERPO，一个验证证据正则化策略优化框架，它将证据视为策略修正的提议，同时保留结果目标。该框架将无证据的参考恢复与带符号的token级证据修正分离开来。Fisher证据对比沿着估计的证据存在方向对修正进行衰减。一个带停止机制的逐token ZPD控制器根据局部奖励对齐程度和Fisher移动成本来调节修正的接受度，而参考通道保持独立于接受决策。

    arXiv:2609.06100v1 Announce Type: cross  Abstract: Verifiable outcome rewards guide language-model post-training, but sequence-level advantages do not identify which token-level decisions should be preserved or revised. Evidence-conditioned Teachers provide denser supervision by replaying sampled trajectories with privileged feedback. Yet indiscriminate imitation risks transferring formatting or reasoning-style shifts that do not support task success. We introduce VERPO, a Verified Evidence Regularized Policy Optimization framework that treats evidence as a proposal for policy correction while retaining the outcome objective. It separates evidence-free reference restoration from signed token-level evidence corrections. Fisher Evidence Contrast attenuates corrections along an estimated evidence-presence direction. A stopped token-wise ZPD controller scales acceptance according to local reward alignment and Fisher movement cost, while the reference channel remains independent of acceptan
    
[^232]: PhenoBench：深度表型人类队列能告诉我们什么

    PhenoBench: Mapping What a Deeply Phenotyped Human Cohort Can Tell Us

    [https://arxiv.org/abs/2609.06080](https://arxiv.org/abs/2609.06080)

    该论文提出了PhenoBench——一个基于人类表型项目（超13,000名参与者）构建的可执行评估基准，通过定义15个领域、26种输入模态下的90项临床任务，系统性地量化了哪些测量数据对哪些健康问题具有预测价值。

    

    深度表型队列结合了从秒级到年际跨时间尺度的临床、影像、分子和可穿戴设备观测数据。这种广度可以揭示哪些测量数据能为哪些健康相关问题提供信息，但异构的分析结果无法直接比较。我们提出了PhenoBench，一个围绕人类表型项目构建的可执行基准，该项目已有超过13,000名参与者完成了首次访问。每个问题都固定了目标、合格人群、时间点和允许使用的信息；其评估契约规定了数据划分、评估指标、基线和结论边界。该基准定义了90项基于临床的任务，涵盖15个领域和26种输入模态。测量结果显示出依赖于具体问题和数据表示的预测价值，包括相对于匹配基线在留出集性能上的正向、接近零以及负向的变化。我们使用PhenoBench评估了新兴的表格基础模型……

    arXiv:2609.06080v1 Announce Type: cross  Abstract: Deeply phenotyped cohorts combine clinical, imaging, molecular, and wearable observations across timescales from seconds to years. This breadth can reveal which measurements inform which health-related questions, but heterogeneous analyses are not directly comparable. We present PhenoBench, an executable benchmark built around the Human Phenotype Project, in which more than 13,000 participants have completed the initial visit. Each question fixes the target, eligible population, timing, and allowed information; its evaluation contract specifies the split, metric, baseline, and claim boundary. The benchmark defines 90 clinically grounded tasks across 15 domains and 26 input modalities. Measurements showed question- and representation-dependent predictive value, including positive, near-zero, and negative changes in held-out performance relative to matched baselines. We used PhenoBench to evaluate emerging tabular foundation models acros
    
[^233]: HANIA：面向有据可依问答的规划器引导多模态图证据选择

    HANIA: Planner-Guided Multimodal Graph Evidence Selection for Grounded Question Answering

    [https://arxiv.org/abs/2608.29088](https://arxiv.org/abs/2608.29088)

    HANIA提出了一种规划器引导的多模态图框架，利用冻结视觉-语言模型提取可弃权的视觉证据、构建基于输入的多模态图，并通过双组有限状态规划器与覆盖感知剪枝选出紧凑且多样的证据，从而提升有据可依的多模态问答表现。

    

    多模态问答仍然对噪声大、不完整和依据薄弱的证据十分敏感。冗长的非结构化上下文会引入冗余并助长无依据的生成，而扁平化的检索可能忽略多步推理所需的关系。我们提出了HANIA，一个用于基于证据问答的规划器引导多模态图框架。HANIA使用冻结的视觉-语言模型处理给定的图像和文本，提取简洁且与问题相关的视觉证据，并支持明确的弃权（拒答）机制。随后，它构建一个基于输入的多模态图，并应用双组有限状态规划器来协调描述性证据和关系性证据。覆盖感知剪枝基于相关性、图置信度、概念覆盖率和模态多样性保留一个紧凑的证据集。所选的文本段落、视觉陈述和图三元组被提供给一个冻结的指令微调解码器。我们在Sci（摘要在此处截断）上对HANIA进行了评估。

    arXiv:2608.29088v1 Announce Type: new  Abstract: Multimodal question answering remains sensitive to noisy, incomplete, and weakly grounded evidence. Long unstructured contexts can introduce redundancy and encourage unsupported generation, while flat retrieval may overlook relations needed for multi-step reasoning. We present HANIA, a planner-guided multimodal graph framework for evidence-grounded question answering. HANIA processes the supplied image and text using a frozen vision-language model to extract concise question-relevant visual evidence with explicit abstention. It then constructs an input-grounded multimodal graph and applies a two-group finite-state planner to coordinate descriptive and relational evidence. Coverage-aware pruning retains a compact evidence set based on relevance, graph confidence, concept coverage, and modality diversity. The selected passages, visual statements, and graph triples are provided to a frozen instruction-tuned decoder. We evaluate HANIA on Sci
    
[^234]: 记忆并非总是必需：科学推理中条件记忆的特征化研究

    Memory Is Not Always Needed: Characterizing Conditional Memory in Scientific Reasoning

    [https://arxiv.org/abs/2608.23982](https://arxiv.org/abs/2608.23982)

    本文系统研究了科学推理中条件记忆的适用条件，提出知识边界感知路由器，根据输入代理动态决定是否及如何激活记忆，以避免干扰并提升推理准确性。

    

    科学推理要求语言模型检索专业知识，并将其可靠地整合到多步计算中。条件记忆提供了一条显式查找路径，补充了稠密神经表示，但其有用性本质上依赖于输入和计算：检索到的信息可能修复缺失的科学关联，但也可能引入分散注意力的捷径，或干扰基础模型本可正确执行的推理。在本工作中，我们系统地研究了条件记忆应在何时、何处以及何种程度上参与科学推理。我们刻画了科学知识边界，并对启用记忆的知识电路节点进行了受控干预。基于这些分析，我们提出了一种知识边界感知路由器，该路由器利用生成前可用的任务特定输入代理来判断是否激活记忆，以及激活哪些层。

    arXiv:2608.23982v1 Announce Type: new  Abstract: Scientific reasoning requires language models to retrieve specialized knowledge and incorporate it reliably into multi-step computation. Conditional memory provides an explicit lookup pathway that complements dense neural representations, but its usefulness is inherently input- and computation-dependent: retrieved information may repair missing scientific associations, yet it may also introduce distracting shortcuts or interfere with reasoning that the base model can already perform correctly. In this work, we systematically investigate when, where, and to what extent conditional memory should participate in scientific reasoning. We characterize the scientific knowledge boundary and controlled interventions on memory-enabled knowledge-circuit nodes. Based on these analyses, we propose a Knowledge Boundary-Aware Router that uses task-specific input proxies available before generation to determine whether memory is activated, which layer-s
    
[^235]: 将视觉-语言-行动模型进化为具备即时工具使用的智能体

    Evolve Vision-Language-Action Model into an Agent with On-the-fly Tool-use

    [https://arxiv.org/abs/2608.14047](https://arxiv.org/abs/2608.14047)

    本文提出ART框架，通过将VLA模型与即时工具使用结合，显著降低动作空间复杂性和数据需求，在小型数据集上实现了更高的泛化性和任务成功率。

    

    arXiv:2608.14047v1 公告类型：交叉 摘要：本文通过将端到端的视觉-语言-行动（VLA）模型与智能体工具使用相结合，提出了具备工具使用的智能体机器人（ART）。ART是一种工具注入框架，可调整任何VLA模型以利用现成的工具模块，用于低级视觉、高级功能性和具身增强。与具有完整连续动作解空间的普通VLA模型相比，ART通过工具使用降低了动作解空间的复杂性，这不仅提高了跨任务的泛化能力，还减少了对数据的依赖。为了展示该框架的优势（高泛化性和低数据依赖性），我们首先构建了一个包含30K条工具使用轨迹和动作演示的数据集，该数据集远小于基线方法所使用的数据集。然后，我们设计了一种针对挑战性环境中长轨迹工具使用推理的训练方案。实验表明，ART的成功率提高了20%以上。

    arXiv:2608.14047v1 Announce Type: cross  Abstract: This paper integrates end-to-end Visual-Language-Action (VLA) models with agentic tool-use to propose Agentic Robot with Tool-use (ART). ART is a tool-injection framework that tunes any VLA model to leverage off-the-shelf tool modules for low-level vision, high-level affordance, and embodiment enhancement. Compared to vanilla VLA models with a whole continuous action solution space, ART reduces the complexity of the action solution space through tool-use, which not only improves generalizability across different tasks but also reduces data dependency. To demonstrate the advantages (high generalizability and low data dependency) of this framework, we first built a dataset of 30K tool-use trajectories and action demonstrations, which is much smaller than those used by baseline methods. We then designed a training regimen for long-trajectory tool-use reasoning in challenging environments. Experiments show that ART achieves a 20% higher su
    
[^236]: 面向分布式CNC刀具磨损预测的联邦学习

    Federated Learning for Distributed CNC Tool Wear Prediction

    [https://arxiv.org/abs/2608.11281](https://arxiv.org/abs/2608.11281)

    本文提出将联邦学习应用于分布式CNC刀具磨损预测，在不共享原始数据的情况下实现接近集中式学习的性能，并显著优于本地模型。

    

    刀具磨损预测是数控加工中的一项重要任务，其中对刀具状态的准确监测有助于保障产品质量和工艺可靠性。机器学习方法在该任务中展现出潜力，但其在工业环境中的应用受到加工数据分布式特性以及机器、站点或组织间数据共享限制的制约。联邦学习通过在不传输原始运行数据的情况下实现协作模型训练，为这一场景提供了合适的框架。本文研究了联邦学习在CNC刀具磨损预测中的应用。刀具轨迹被分配到模拟客户端以表示联邦学习场景，并将联邦模型与集中式参考模型及本地客户端基线进行比较。结果表明，联邦学习的性能接近集中式学习，并显著优于本地客户端模型。

    arXiv:2608.11281v1 Announce Type: cross  Abstract: Tool wear prediction is an important task in CNC machining, where accurate monitoring of tool condition supports product quality and process reliability. Machine learning methods have shown potential for this task, but their use in industrial environments is limited by the distributed nature of machining data and by restrictions on data sharing between machines, sites, or organizations. Federated learning offers a suitable framework for this setting by enabling collaborative model training without transferring raw operational data. This paper investigates federated learning for CNC tool wear prediction. Tool trajectories are distributed across simulated clients to represent a federated learning scenario. The federated models are compared against centralized references and local client baselines. Results show that federated learning achieves performance close to centralized learning and improves significantly over local client models. T
    
[^237]: 共享状态的描述方式决定了AI智能体能否实现同步

    How a shared state is described determines whether AI agents synchronize

    [https://arxiv.org/abs/2608.06968](https://arxiv.org/abs/2608.06968)

    该研究发现，AI智能体所共享状态的描述格式（如数值摘要还是直方图）本身就能决定多智能体系统能否实现同步对齐，即使描述所含信息完全相同。

    

    语言模型智能体越来越多地以群体形式行动，其中真正重要的结果是集体性的：它们是对齐、分裂还是无法协调。每个智能体并非直接作用于真实世界，而是作用于对世界的文本描述，而这一选择通常在软件中被固定下来。我们以同步——研究交互规则如何产生集体秩序的经典探针——为工具，证明了这一描述选择本身就能决定最终结果。分布在圆环上的智能体在读取其他智能体的相对位置后，选择前进、保持不动或后退，实验涵盖了匹配种群、受控输入和三个模型家族，共获得507,112条有效响应。在GPT中，数值摘要使所有匹配种群在两个正耦合强度下均实现对齐，而直方图则无一能使种群对齐；Claude在较强耦合下却表现出相反的模式。对完全相同状态的重新描述会改变所有三个模型家族中的动作概率，即使在携带相同信息的直方图之间也是如此。没有任何单一的方向性系数能够解释……

    arXiv:2608.06968v2 Announce Type: replace-cross  Abstract: Language-model agents increasingly act in populations, where the outcome that matters is collective: whether they align, split or fail to coordinate. Each acts not on the world but on a text description of it, a choice usually fixed in software. Using synchronization, the canonical probe of how interaction rules produce collective order, we show that this choice can decide the outcome. Agents on a circle chose to advance, stay or move back after reading the others' relative positions, in 507,112 valid responses across matched populations, controlled inputs and three model families. In GPT, numerical summaries aligned every matched population at both positive couplings, whereas histograms aligned none; Claude showed the reverse at the stronger coupling. Re-describing identical states shifted action probabilities in all three families, even between histograms carrying the same information. No single directional coefficient explai
    
[^238]: 基于注意力机制的多任务计算表示

    Attention-based representations for multi-task computation

    [https://arxiv.org/abs/2608.04243](https://arxiv.org/abs/2608.04243)

    该论文从理论上证明了多任务场景下多头注意力的必要性：单个注意力头需要指数级更高的嵌入维度或精度才能同时完成如求最大最小值、计算异或等多任务。

    

    多头注意力层产生的向量表示能够支持多个下游任务。我们在两个简单而具体的多任务场景中建立了所需注意力头数量的界限。在第一个场景中，寻求一种向量表示，使得线性预测器能够同时计算给定列表中的最小值和最大值。在这种情况下，已知两个具有较小嵌入维度和位精度水平的注意力头即可满足要求。我们证明，单个注意力头则需要指数级更高的嵌入维度或精度水平。在第二个场景中，寻求一种向量表示，使得多项式阈值函数能够计算给定 $n$ 比特字符串的异或（XOR）。当 $n=2$ 时，该场景与第一个场景类似，因为利用同时编码两个比特的“与”（AND）和“或”（OR）的向量表示，异或可以很容易地由线性函数计算得出。我们观察到，$n$ 比特异或需要……

    arXiv:2608.04243v1 Announce Type: cross  Abstract: Multi-head attention layers produce vector representations that support multiple downstream tasks. We establish bounds on the number of heads required in two simple and concrete multi-task scenarios. In the first scenario, a vector representation is sought so that linear predictors can compute both the smallest and largest numbers in a given list. In this case, it is known two attention heads with small embedding dimension and bit precision level suffice. We prove that a single attention head requires exponentially higher embedding dimension or precision level. In the second scenario, a vector representation is sought so that a polynomial threshold function can compute the XOR of a given string of $n$ bits. This scenario is analogous to the first one for $n=2$, since XOR is readily computed by a linear function using a vector representation that encodes both the AND and the OR of the two bits. We observe that $n$-bit XOR requires the p
    
[^239]: TACT：面向教学自适应英语辅导的分类体系对齐后训练

    TACT: Taxonomy-Aligned Post-Training for Pedagogically Adaptive English Tutoring

    [https://arxiv.org/abs/2608.03952](https://arxiv.org/abs/2608.03952)

    该论文提出TACT框架，基于人类辅导研究构建“辅导者策略”与“学习者行为”两个分类体系及相应语料库，对LLM进行后训练与评估，使其能够根据学习者行为和对话语境自适应地选择恰当的教学策略。

    

    大语言模型（LLM）日益被用于为英语作为第二语言（ESL）学习者提供对话练习。然而，有效的ESL辅导不仅仅是生成流利的回复：辅导者必须根据学习者行为和对话语境选择合适的教学行为。人类辅导研究为自适应支持提供了原则，但这些原则往往局限于特定任务，尚未充分融入基于LLM的ESL辅导系统的训练与评估之中。我们提出了TACT（分类体系对齐的对话辅导者），这是一个以人类研究为基础的框架，用于对具备教学自适应能力的ESL辅导系统进行后训练和评估。借鉴已有文献，我们构建了两个互补的分类体系：包含13种辅导者回复策略的“辅导者策略分类体系”，以及按行为类型和状态刻画学习者行为的“学习者行为分类体系”。利用这些分类体系，我们构建了TACTCorpus，它对260条对话（原文此处被截断）……

    arXiv:2608.03952v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to provide conversational practice for English-as-a-second-language (ESL) learners. Effective ESL tutoring, however, requires more than fluent response generation: a tutor must select an appropriate pedagogical action based on learner behavior and dialogue context. Human-tutoring research offers principles for adaptive support, but they are often task-specific and remain insufficiently integrated into LLM-based ESL tutor training and evaluation. We present TACT (Taxonomy-Aligned Conversational Tutor), a human-grounded framework for post-training and evaluating pedagogically adaptive ESL tutors. Drawing on established literature, we develop two complementary taxonomies: the Tutor-Strategy Taxonomy with 13 tutor response strategies and the Student-Move Taxonomy characterizing learner behavior by move type and status. Using these taxonomies, we construct TACTCorpus, which enriches 260 a
    
[^240]: 面向INT2 KV缓存量化的输出感知旋转方法

    Output-Aware Rotation for INT2 KV-Cache Quantization

    [https://arxiv.org/abs/2608.02691](https://arxiv.org/abs/2608.02691)

    本文提出输出感知旋转方法OptR，通过最小化输出投影 $W_O$ 之后的注意力输出误差、将误差分解为键和值引起的项并学习逐头正交校正，同时利用注意力等价的键重参数化降低通道偏移，从而实现更优的INT2 KV缓存量化。

    

    键值缓存已成为长上下文大语言模型推理中的主要内存和带宽瓶颈，使得超低比特量化日益重要。然而，现有的基于旋转的INT2方法在完整注意力读出之前优化缓存统计量或代理误差，而模型最终实际受到的是通过注意力机制和输出投影 $W_O$ 传播的误差的影响。为了解决这种不匹配问题，我们提出了 OptR，一种输出感知的旋转方法，它最小化经过 $W_O$ 之后的注意力输出误差。OptR 将 $W_O$ 之后的注意力输出误差分解为键和值引起的两个部分，并通过完整的INT2量化和注意力路径学习每个注意力头的正交校正。OptR 还进一步应用了一种注意力等价的键重参数化方法，在不改变softmax分布的前提下减少较大的逐通道偏移。在三个模型和五个推理与编码任务上的实验（摘要在此处截断）。

    arXiv:2608.02691v3 Announce Type: replace  Abstract: The key-value (KV) cache has become a major memory and bandwidth bottleneck in long-context large language model inference, making ultra-low-bit quantization increasingly important. However, existing rotation-based INT2 methods optimize cache statistics or proxy errors before the complete attention readout, even though the model is ultimately affected by the error propagated through attention and the output projection $W_O$. To address this mismatch, we propose \textit{OptR}, an output-aware rotation method that minimizes post-$W_O$ attention-output error. OptR decomposes the post-$W_O$ attention-output error into key- and value-induced terms and learns per-head orthogonal corrections through the full INT2 quantization and attention path. OptR further applies an attention-equivalent key reparameterization to reduce large channel-wise offsets without changing the softmax distribution. Across three models and five reasoning and coding 
    
[^241]: 基于角色的访问控制下的文本到SQL基准测试

    Benchmarking Text-to-SQL under Role-Based Access Control

    [https://arxiv.org/abs/2607.22115](https://arxiv.org/abs/2607.22115)

    该论文提出了首个在基于角色的访问控制（RBAC）约束下评估text-to-SQL系统的综合基准测试框架，利用LLM辅助工作流为现有基准自动生成合理的用户角色和访问策略，从而弥合基准测试分数与真实访问受控环境中模型表现之间的差距。

    

    给定一个数据库S和一个自然语言问题Q，文本到SQL（text-to-SQL）系统旨在生成一个SQL查询，该查询在与S执行时能够正确回答Q。目前，流行的text-to-SQL基准测试大多假设对S的无限制访问；然而在实践中，用户的访问通常是受限的，例如通过基于角色的访问控制（RBAC）策略。这导致了基准测试结果与真实世界性能之间的潜在脱节：一个在基准测试中得分很高的LLM，在访问受控的环境中可能表现不佳，要么频繁违反RBAC规则，要么拒绝一个仅使用S中被允许的数据就能回答的查询q。基于此，我们提出了一个带有现实RBAC约束的综合text-to-SQL基准测试框架，该框架具有一个LLM辅助的工作流程，能够用合理的用户角色和访问策略来增强现有的text-to-SQL基准测试。为此，我们将角色合成问题表述为……

    arXiv:2607.22115v2 Announce Type: replace-cross  Abstract: Given a database S and a natural language question Q, text-to-SQL systems aim to generate an SQL query that correctly answers Q when executed against S. Currently, popular text-to-SQL benchmarks mostly assume unrestricted access to S; in practice, however, user access is often restricted, e.g., through role-based access control (RBAC) policies. This leads to a potential disconnect between benchmarking results and real-world performance: an LLM with high benchmark scores might perform poorly in an access-controlled environment, by frequently violating RBAC, or rejecting a query q that could be answered with only permitted data in S. Motivated by this, we present a comprehensive text-to-SQL benchmarking framework with realistic RBAC constraints, which features an LLM-assisted workflow that augments existing text-to-SQL benchmarks with plausible user roles and access policies. To do so, we formulate the problem of role synthesis a
    
[^242]: 力反馈永不嫌迟：利用反应式力注入加速VLA后训练

    Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection

    [https://arxiv.org/abs/2607.14236](https://arxiv.org/abs/2607.14236)

    LIFT是一种力感知的后训练框架，通过在预训练VLA策略旁嫁接反应式动作专家，并借助因果力记忆和零初始化交叉注意力注入6D末端执行器力，为模型增加接触反应能力的同时保留其通用操作知识，从而加速VLA后训练。

    

    预训练的视觉-语言-动作（VLA）策略提供了强大的语言条件化操作知识，但它们在很大程度上仍由视觉驱动，一旦操作进入接触状态就会遇到困难——例如场景被遮挡、深度信息模糊，或微小的力误差使执行偏离离线演示分布。我们提出了LIFT（面向VLA后训练的后期反应式力注入），这是一个力感知的后训练框架，能在保留预训练VLA策略通用操作知识的同时，为其增加接触反应能力。LIFT在原始动作专家旁边嫁接一个反应式动作专家，使用预训练的动作权重对其进行初始化，并通过因果力记忆和零初始化的交叉注意力注入最近的6D末端执行器力，使动作能够在执行过程中被实时刷新。为解决接触反馈的策略相关分布偏移问题，LIFT进一步将反应式力注入与适配机制相耦合，从而加速VLA在接触丰富任务上的后训练过程。

    arXiv:2607.14236v2 Announce Type: replace-cross  Abstract: Pretrained vision-language-action (VLA) policies provide strong language-conditioned manipulation knowledge, but they remain largely vision-driven and can struggle once manipulation enters contact states where the scene is occluded, depth is ambiguous, or small force errors push execution off the offline demonstration distribution. We present LIFT (Late Reactive Injection of Force for VLA Post-Training), a force-aware post-training framework that adds contact reactivity to a pretrained VLA policy while preserving its general manipulation knowledge. LIFT grafts a reactive action expert beside the original action expert, initializes it from pretrained action weights, and injects recent 6D end-effector force through causal force memory and zero-initialized cross attention, enabling actions to be refreshed during execution. To address the policy-dependent distribution shift of contact feedback, LIFT further couples reactive force i
    
[^243]: Omni-Decision：面向全模态智能体的证据账本规划

    Omni-Decision: Evidence-Ledger Planning for Omni-Modal Agents

    [https://arxiv.org/abs/2607.11433](https://arxiv.org/abs/2607.11433)

    Omni-Decision 针对全模态智能体的规划瓶颈，用显式的证据账本取代不断膨胀的对话历史，由批评者模块过滤嘈杂的多模态观测，仅保留可用证据，使规划器在紧凑上下文中做出更可靠的多步决策。

    

    全模态智能体需要跨视频、音频、网页和计算工具寻找证据来回答问题。其主要瓶颈在于规划：嘈杂的多模态观测在对话历史中不断累积，干扰后续决策，而多模态模型的多步规划能力有限。受控的后端替换实验支持了这一诊断：替换规划器所造成的性能损失远大于替换感知后端。我们提出了 Omni-Decision，一个基于证据账本规划构建的全模态智能体：它用一个显式的证据账本取代不断增长的对话历史，账本记录尚缺哪些证据、哪些证据已被确认，以及哪些记录之间存在冲突。一个批评者模块读取每条嘈杂的观测，只将可用内容传递给账本并丢弃其余部分，使规划器在整个任务过程中始终基于紧凑的上下文进行决策。每次运行都会记录每一步的状态、动作和判定……（原文摘要在此处截断）

    arXiv:2607.11433v2 Announce Type: replace  Abstract: Omni-modal agents must seek evidence across video, audio, web pages, and computation to answer questions. Their main bottleneck is planning: noisy multimodal observations accumulate in conversation history and disrupt later decisions, while multimodal models have limited capacity for multi-step planning. Controlled backend replacements support this diagnosis: replacing the planner causes a much larger performance loss than replacing the perception backend. We present Omni-Decision, an omni-modal agent built on evidence-ledger planning: it replaces the growing dialogue history with an explicit evidence ledger that records what evidence is still missing, what has been confirmed, and where records conflict. A critic reads each noisy observation and passes only the usable content to the ledger, discarding the rest, so the planner works from a compact context throughout the task. Each run records the state, action, and verdict at every st
    
[^244]: 相同的故事，不同的旅程：探索基于人设的对话代理，利用同龄人帖子支持职业探索

    Same Stories, Different Journeys: Exploring Persona-Grounded Conversational Agents for Supporting Career Exploration with Peers' Posts

    [https://arxiv.org/abs/2607.11039](https://arxiv.org/abs/2607.11039)

    本研究开发了基于同龄人求职帖子构建人设并遵循自我决定理论的对话代理JobMate，相比静态浏览帖子，它通过支持案例选择与持续提问，帮助年轻求职者更好地完成职业探索中的意义建构，减少隐性焦虑。

    

    年轻求职者经常通过浏览同龄人分享求职经历的帖子来探索自己的职业可能性。然而，静态浏览要求他们自行重构碎片化的案例，并私下判断他人的经历对自己意味着什么，有时还会因向上社会比较而加剧焦虑。本文研究了将这些帖子转化为基于人设的对话如何重塑这一意义建构过程。我们开发了JobMate原型系统，其中的智能体基于同龄人帖子构建人设，并遵循自我决定理论（self-determination theory）与用户对话。在一项有24名参与者参与的组间对比研究中，在小红书上浏览帖子虽然展示了多样化的职业轨迹，但案例重构和比较工作很大程度上仍留给用户，而JobMate则支持案例选择和持续提问。这些对话还促使用户表达出此前隐性的限制因素和接纳……

    arXiv:2607.11039v2 Announce Type: replace-cross  Abstract: Young job seekers frequently explore their career possibilities by browsing peers' posts that share job-seeking experiences. However, static browsing requires them to reconstruct fragmented cases and privately judge what others' experiences mean for themselves, sometimes intensifying anxiety through upward social comparison. In this paper, we examine how transforming these posts into persona-grounded conversations reshapes this sensemaking process. We developed JobMate, a prototype featuring agents that have personas built upon peers' posts and follow the self-determination theory to converse with users. In a between-subjects comparative study with 24 participants, RedNote browsing exposed diverse trajectories but left reconstruction and comparison largely to users, whereas JobMate supported case selection and continued questioning. The conversations further prompted users to articulate previously implicit constraints and accep
    
[^245]: IB-Flow：面向少步文本到图像生成的信息瓶颈引导CFG蒸馏

    IB-Flow: Information Bottleneck-Guided CFG Distillation for Few-Step Text-to-Image Generation

    [https://arxiv.org/abs/2607.09133](https://arxiv.org/abs/2607.09133)

    该论文提出IB-Flow，利用信息瓶颈理论引导CFG蒸馏，根据图像生成过程中熵逐步降低的动态特性自适应地调节引导强度与教师时间步采样，从而在少步文本到图像生成中避免CFG过度条件化伪影并突破现有少步压缩的性能上限。

    

    尽管大规模文本到图像生成模型已取得前所未有的视觉表现，但其对多步迭代求解器的固有依赖导致了严重的推理延迟。针对无分类器引导轨迹的少步蒸馏已成为主流的双维度压缩范式。然而，现有框架仍受制于一种粗粒度的盲目注入范式：其在无差别采样教师时间步的同时，始终强制施加全局静态的引导强度。这种与状态无关的设计完全忽视了图像生成作为一种以熵逐步降低为特征的动态演化过程的内在本质，这不仅限制了少步压缩的性能上限，还会引发严重的CFG过度条件化伪影。为突破这些局限，我们通过理论……（原文摘要在此处截断）

    arXiv:2607.09133v3 Announce Type: replace-cross  Abstract: While large-scale text-to-image generative models have achieved unprecedented visual performance, their inherent reliance on multi-step iterative solvers incurs severe inference latency. Few-step distillation targeting the Classifier-Free Guidance (CFG) trajectory has emerged as the prevalent dual-dimensional compression paradigm. However, existing frameworks remain subjugated by a coarse-grained blind injection paradigm that perpetually enforces a globally static guidance strength while indiscriminately sampling the supervisor timestep. This state-agnostic design completely disregards the intrinsic nature of image generation as a dynamic evolutionary process characterized by progressive entropy reduction, which not only restricts the performance boundary of few-step compression but also precipitates severe CFG over-conditioning artifacts. To transcend these limitations, we re-examine the distillation procedure through the theo
    
[^246]: 预言家何时能在预测市场中获利？

    When do prophets profit in prediction markets?

    [https://arxiv.org/abs/2607.06166](https://arxiv.org/abs/2607.06166)

    本文为基于中央限价订单簿的预测市场提出了一种仅依赖预测者预测和市场价格的“适当”投注策略，证明只要预测在任意适当评分规则下优于市场价格且市场流动性充足即可获得正的预期利润，且该类策略是唯一具有这种稳健盈利保证的策略。

    

    预测市场将分散的信念汇聚成价格，这些价格充当对不确定事件的概率预测。经典理论已经确立了优于市场的预测如何能带来正的交易利润，但该理论关键依赖于特定的自动做市商（AMM）设计，并不适用于当今基于中央限价订单簿的主流交易所。本文填补了这一空白。对于任何预测市场和任何适当评分规则 $S$，我们提出了一种“适当”的投注策略，该策略仅依赖于预测者的预测 $\mathbf{p}$ 和市场价格 $\mathbf{q}$，并且只要 $\mathbf{p}$ 在 $S$ 下优于 $\mathbf{q}$ 且市场具有足够的流动性，就能获得正的预期利润。此外，这种适当投注本质上是唯一具有如此稳健盈利保证的策略。我们的证明基于对预期利润的分解，该分解严格推广了经典的（摘要在此处被截断）

    arXiv:2607.06166v3 Announce Type: replace  Abstract: Prediction markets aggregate dispersed beliefs into prices that act as probabilistic forecasts of uncertain events. Classical theory establishes how a better-than-market forecast can yield positive trading profit. However, it hinges crucially on the specific automated market maker (AMM) design, and is not applicable to popular exchanges today which are based on central limit order books. This paper fills that gap. For any prediction market and any proper scoring rule $S$, we exhibit a ``proper'' betting strategy that depends only on the forecaster's prediction $\mathbf{p}$ and the market price $\mathbf{q}$, and earns positive expected profit \emph{whenever} $\mathbf{p}$ outperforms $\mathbf{q}$ under $S$ and the market has sufficient liquidity. Moreover, this proper betting is essentially the only strategy with such robust profitability guarantee. Our proof rests on a decomposition of expected profit that strictly generalizes the cla
    
[^247]: 基于量规的前沿语言模型在专家撰写的临床推理任务上的受控比较

    A rubric-based controlled comparison of frontier language models on expert-authored clinical reasoning tasks

    [https://arxiv.org/abs/2607.02175](https://arxiv.org/abs/2607.02175)

    该研究构建了一个由临床医生撰写的高难度临床推理评估数据集及加权量规，发现前沿大模型在关键临床标准上的通过率（32.4-41.7%）远低于低风险标准（80-90%），揭示了模型能力与临床优先级之间的倒置现象。

    

    多项选择题式的医学基准测试已日益饱和，而近期基于量规的评估（如HealthBench）表明，开放式临床性能远未得到解决——其“困难”子集的最高得分仍停留在32%。我们提出了一个小型但刻意设计的高难度评估数据集，包含五个由临床医生撰写的临床场景，涵盖四个专科（麻醉学、内科/家庭医学、急诊医学和产科），每个场景均配有基于临床医生起草的标准答案编写的原子化、加权、相互独立且完全穷尽（MECE）的量规（每个任务25-62条标准，共184条标准）。我们评估了三个前沿模型：GPT 5.4、Claude Opus 4.7和Gemini 3.1 Pro。平均量规通过率分别为0.47（Claude）、0.38（GPT）和0.37（Gemini）。核心发现是临床优先级的倒置：最高权重（权重5，关键级）标准的通过率仅为32.4-41.7%，而低风险的权重1标准通过率却高达80-90%。108条标准中有55条……

    arXiv:2607.02175v2 Announce Type: replace  Abstract: Multiple-choice medical benchmarks are increasingly saturated, and recent rubric-based evaluations such as HealthBench have shown that open-ended clinical performance is far from solved - its "Hard" subset top score remains 32%. We present a small, deliberately difficult evaluation dataset of five clinician-authored clinical scenarios spanning four specialties (anaesthesia, internal/family medicine, emergency medicine, and obstetrics), each accompanied by an atomic, weighted, MECE rubric (25-62 criteria per task; 184 criteria total) authored from a clinician-drafted golden answer. We evaluate three frontier models: GPT 5.4, Claude Opus 4.7, and Gemini 3.1 Pro. Mean rubric pass rates were 0.47 (Claude), 0.38 (GPT), and 0.37 (Gemini). The central finding is an inversion of clinical priority: the highest-weighted (weight-5, critical) criteria passed at only 32.4-41.7%, while low-stakes weight-1 criteria passed at 80-90%. 55 of 108 criti
    
[^248]: 条件性协同消融：恢复Transformer电路中的自修复备份组件

    Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits

    [https://arxiv.org/abs/2607.01940](https://arxiv.org/abs/2607.01940)

    提出条件性协同消融方法CoAx，通过测量主要组件集合被移除后消融效应的增长，来识别Transformer电路中被自修复机制掩盖的休眠备份组件，解决了电路解释在干预下不完整的问题。

    

    机制可解释性旨在通过“电路”来解释Transformer的行为：电路是一组因果性地支持某种行为的内部组件。然而，自修复机制造成了一个盲区：消融一个主要组件可能会激活一个休眠的备份组件，因此在完整模型中能够解释行为的电路，在用于测试它的干预之下可能变得不完整。我们将这一差距形式化为“条件性电路补全”问题：给定一个主要组件集合，识别在其被移除后变得因果上重要的组件。我们提出了条件性协同消融，该方法根据主要集合被移除后消融效应的增长幅度来对候选组件进行排序。我们证明，一个完全休眠的备份组件对于基于单元的完整状态评分而言可能与无关组件无法区分，而其条件效应变化恰好聚合了将其与被移除集合联系起来的所有交互阶数。在GPT-2-small的间接宾语识别（IOI）电路上，CoAx（摘要在此处截断）

    arXiv:2607.01940v2 Announce Type: replace  Abstract: Mechanistic interpretability seeks to explain transformer behavior through circuits: sets of internal components that causally support a behavior. However, self-repair creates a blind spot: ablating a primary component can activate a dormant backup, so a circuit that explains behavior in the intact model can become incomplete under the intervention used to test it. We formulate this gap as conditional circuit completion: given a primary set, identify components that become causally important after its removal. We introduce conditional co-ablation (CoAx), which ranks candidates by growth in ablation effect after primary-set removal. We show that a perfectly dormant backup can be indistinguishable from an irrelevant component to per-unit intact-state scores, whereas its conditional effect change exactly aggregates all interaction orders linking it to the removed set. On GPT-2-small's Indirect Object Identification (IOI) circuit, CoAx r
    
[^249]: 相关性并非许可：定位与控制面向指标的注意力贡献

    Relevance Is Not Permission: Localizing and Controlling Metric-Facing Attention Contributions

    [https://arxiv.org/abs/2606.30139](https://arxiv.org/abs/2606.30139)

    提出Warrant统一方法，通过暴露通向评估指标的逐项注意力贡献路径并施加查询条件化的许可控制，揭示“注意力相关性不等于预测贡献”（最高注意力项在约一半样本中反而损害效用），并在五类任务的32组对比中有27组提升了主要指标。

    

    注意力机制能够识别与当前查询相关的项目，但无法单独判断这些项目的价值贡献是否真正支持预测。我们提出Warrant，一种用于定位和控制面向指标的注意力贡献的统一方法。Warrant首先识别并暴露通向所报告指标的逐项贡献路径，然后在同一路径上施加基于当前查询条件的许可机制。完整版Warrant在CTDG、MTPP、RAG、STPP和TKG五类任务的32组模型-数据集对比中，有27组提升了主要指标。在五个代表性设置中进行的精确项目移除分析发现，注意力与边际预测效用之间的相关性几乎为零；甚至在43.5%至54.4%的样本中，注意力最高的项目反而会降低目标效用。对完整基准的贡献分解显示，路径暴露与习得许可两部分的作用因任务而异。在五个随机种子的HotpotQA分析中，被暴露的路径分配了更多的……（原文摘要在此处截断）

    arXiv:2606.30139v3 Announce Type: replace  Abstract: Attention identifies items relevant to a current query, but does not separately determine whether their value contributions support the prediction. We propose Warrant, a unified method for locating and controlling metric-facing attention contributions. Warrant identifies and exposes the item-wise contribution path that reaches the reported metric, then applies current-query-conditioned permission on that same path. Full Warrant improves the primary metric in 27 of 32 model-dataset comparisons across CTDG, MTPP, RAG, STPP, and TKG. Exact item-removal analysis in five representative settings finds near-zero correlation between attention and marginal prediction utility; even the highest-attention item reduces target utility in 43.5-54.4% of examples. Decomposition over the complete benchmark shows that the contributions of path exposure and learned permission vary by task. In a five-seed HotpotQA analysis, the opened path assigns more a
    
[^250]: 固定系统与递归自我改进系统的安全性的算法不可验证性

    Algorithmic Unverifiability of Safety for Fixed and Recursively Self-Improving Systems

    [https://arxiv.org/abs/2606.28639](https://arxiv.org/abs/2606.28639)

    该论文从数学上严格证明了对于图灵完备的自修改系统（包括递归自我改进系统），安全验证在静态和动态两个层面都存在不可逾越的极限——不存在任何既可靠、完备又可行的安全验证器，从而为AI自我改进的安全性验证划定了根本性的理论边界。

    

    我们为图灵完备的自修改系统——即递归自我改进发生的这类系统——建立了算法安全验证的数学极限，既针对固定系统，也针对系统对其自身的修改。从静态角度看，不存在任何既可靠、完备又可行的验证器：在无界域上由Rice定理和哥德尔定理证明，在所有有限配置上由Trakhtenbrot定理证明，而在简洁描述的有限环境中，验证一个策略以对抗对手是coNP完全问题，合成这样一个策略则是PSPACE完全问题。从动态角度看，我们将一步自我修改建模为代码的可计算变换，并询问安全性质能否在该变换后得以保持。如果该变换仅依赖于行为，这相当于“上一层的Rice定理”；如果它读取代码——正如自我修改所做的——该问题不再是语义性的，但同样的s-m-n归约在行为等价的类中仍然成立。

    arXiv:2606.28639v3 Announce Type: replace-cross  Abstract: We establish mathematical limits of algorithmic safety verification for Turing-complete self-modifying systems, the class in which recursive self-improvement takes place, both for a fixed system and across its own modification. Statically, no verifier is sound, complete and tractable: over unbounded domains by Rice's and G\"odel's theorems, over all finite configurations by Trakhtenbrot's theorem, and over succinctly described finite environments because verifying a policy against an adversary is coNP-complete and synthesising one is PSPACE-complete. Dynamically, we model one step of self-modification as a computable transformation of code and ask whether a safety property survives it. If the transformation depends only on behaviour, this is Rice's theorem one level up; if it reads the code, as self-modification does, the question is no longer semantic, yet the same s-m-n reduction works inside a class of behaviourally identica
    
[^251]: TOPS：通过构建令牌最优保留集实现高效多模态大语言模型推理的基于第一性原理的视觉令牌剪枝方法

    TOPS: First-Principles Visual Token Pruning via Constructing Token Optimal Preservation Sets for Efficient MLLM Inference

    [https://arxiv.org/abs/2606.27161](https://arxiv.org/abs/2606.27161)

    本文从第一性原理出发，通过信息论分析提出三个基本原则（任务相关性、信息覆盖率和语义多样性），并构建令牌最优保留集，实现了无需训练的视觉令牌高效剪枝方法TOPS。

    

    多模态大语言模型（MLLMs）已展现出强大的多模态推理能力，但其效率受到大量视觉令牌的限制，这些令牌带来了显著的计算开销。视觉令牌剪枝提供了一种自然的解决方案，但现有方法并不完善：基于注意力的标准倾向于保留冗余令牌，而基于多样性的标准往往对用户指令不敏感。即使结合多种标准的方法，仍然缺乏对令牌剪枝内在目标的原则性表述。本文从第一性原理角度重新审视视觉令牌剪枝，并将其形式化为构建令牌最优保留集。通过自上而下的信息论分析，我们确定了有效令牌选择的三个基本原则：任务相关性、信息覆盖率和语义多样性。基于这些原则，我们提出TOPS，一种无需训练的方法。

    arXiv:2606.27161v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have achieved strong multimodal reasoning capabilities, but their efficiency is limited by the large number of visual tokens, which introduces substantial computational overhead. Visual token pruning offers a natural solution, yet existing methods are imperfect: attention-based criteria tend to retain redundant tokens, while diversity-based criteria are often agnostic to user instructions. Even methods that combine multiple criteria still lack a principled formulation of the intrinsic objective of token pruning. In this paper, we revisit visual token pruning from a first-principles perspective and formulate it as constructing Token Optimal Preservation Sets. Through a top-down information-theoretic analysis, we identify three fundamental principles for effective token selection: Task Relevance, Information Coverage, and Semantic Diversity. Based on these principles, we propose TOPS, a training-fre
    
[^252]: 面向多轮智能体的课程式轮级引导在线策略蒸馏

    On-Policy Distillation with Curriculum Turn-level Guidance for Multi-turn Agents

    [https://arxiv.org/abs/2606.15912](https://arxiv.org/abs/2606.15912)

    提出Guided-OPD算法，通过在每次rollout中混合教师与学生生成的轮次，并按课程将教师干预概率逐渐衰减至零，解决了多轮智能体在线策略蒸馏中学生误差跨轮累积、教师监督在最需要时反而失效的问题。

    

    arXiv:2606.15912v2 公告类型：replace-cross 摘要：能够进行规划、调用工具并与环境交互的多轮智能体为解决复杂任务提供了一种有前景的范式，但其能力通常依赖于超大规模模型，而这些模型的推理成本在实践中难以承受。在线策略蒸馏是将此类能力迁移到更小学生模型上的一种自然方法，但我们发现它在这种设置下存在一种特有的失效模式：学生模型的小错误会跨轮次不断累积，使轨迹偏离教师模型熟悉的状态分布，导致教师模型的监督恰恰在学生最需要它的地方变得最不可靠。我们提出了引导式在线策略蒸馏，这是一种简单而有效的算法，它在每次rollout中混合教师生成与学生生成的轮次，并按照一个逐渐衰减至零的课程来调度教师模型的干预概率。强引导使早期轨迹保持在教师模型附近……

    arXiv:2606.15912v2 Announce Type: replace-cross  Abstract: Multi-turn agents that plan, invoke tools, and interact with environments offer a promising paradigm for solving complex tasks, yet their capabilities typically rely on very large models whose inference cost is prohibitive in practice. On-Policy Distillation (OPD) is a natural recipe for transferring such capabilities to smaller students, but we find that it suffers a characteristic failure mode in this setting: small student errors compound across turns and push the trajectory out of the teacher's familiar state distribution, so the teacher's supervision becomes least reliable precisely where the student needs it most. We propose Guided On-Policy Distillation (Guided-OPD), a simple yet effective algorithm that mixes teacher- and student-generated turns within each rollout and schedules the teacher's intervention probability along a curriculum that decays to zero. Strong guidance keeps early trajectories close to the teacher di
    
[^253]: 面向混合专家语言模型机器遗忘的路由感知专家校准

    Routing-Aware Expert Calibration for Machine Unlearning in Mixture-of-Experts Language Models

    [https://arxiv.org/abs/2606.10338](https://arxiv.org/abs/2606.10338)

    提出TRACE方法，通过离线激活统计检测遗忘关键专家，并重新加权token级保留损失以匹配其遗忘侧激活频率，从而解决MoE架构中遗忘-保留路由不匹配导致的正则化不足问题。

    

    机器遗忘对大语言模型而言日益重要，然而混合专家架构中的遗忘问题仍然缺乏充分研究。与稠密模型不同，MoE架构在每一层使用路由器将每个token分配给稀疏的专家子集。在本工作中，我们观察到遗忘数据往往会不成比例地激活一小部分专家，而这些专家从保留数据中获得的激活却要弱得多。这种“遗忘-保留路由不匹配”可能导致对遗忘至关重要的专家在遗忘过程中得不到充分的正则化。为了解决这一问题，我们提出了TRACE（面向路由感知的专家校准）方法，用于MoE架构的机器遗忘。TRACE首先从离线激活统计中检测对遗忘至关重要的专家，然后通过重新加权token级别的保留损失来校准保留正则化，使每个被选中专家的保留侧激活频率更好地匹配其遗忘侧的激活频率。

    arXiv:2606.10338v2 Announce Type: replace-cross  Abstract: Machine unlearning is increasingly important for large language models, yet unlearning in Mixture-of-Experts (MoE) architectures remains underexplored. Unlike dense models, MoE architectures employ a router at each layer to assign each token to a sparse subset of experts. In this work, we observe that forget data often activates a small subset of experts disproportionately, while these experts may receive much weaker activation from retain data. This forget--retain routing mismatch can leave forget-critical experts under-regularized during unlearning. To address this, we propose \textbf{TRACE}, Targeted Routing-Aware Calibration of Experts, for MoE unlearning. TRACE first detects forget-critical experts from offline activation statistics, and then calibrates retain regularization by reweighting token-level retain losses so that each selected expert's retain-side activation frequency better matches its forget-side counterpart. E
    
[^254]: TukaBench：一个面向非洲语言的具有文化根基的越狱基准测试

    TukaBench: A Culturally Grounded Jailbreak Benchmark for African Languages

    [https://arxiv.org/abs/2606.01322](https://arxiv.org/abs/2606.01322)

    该论文提出了TUKABENCH——一个针对七种非洲语言的文化化越狱安全评测基准，发现使用非洲语言（尤其是经过文化适配的提示）向大语言模型发起提示会显著降低模型拒绝率，暴露了当前安全评估以英语为中心的缺陷。

    

    大语言模型（LLM）的安全性评估仍然高度以英语为中心，使得低资源语言（LRL），尤其是非洲语言，处于严重研究不足的状态。我们提出了TUKABENCH，一个面向七种非洲语言的越狱基准测试，它通过四种设置将JailbreakBench（JBB）扩展到直接翻译之外：对JBB提示进行人工翻译、将英语提示适配到非洲语境后再进行人工翻译、通过与GPT-5.2交互验证的人工策划提示，以及结合英语和非洲语言的语码转换提示，从而分离语言、文化根基和提示规避性对模型安全的影响。在闭源和开源模型上，使用非洲语言进行提示相对于英语降低了模型的拒绝率，其中经过文化适配的提示导致的拒绝率最低。该评估还揭示了两个结构性局限：模型理解失败以及“LLM作为评判者”可靠性下降的问题。

    arXiv:2606.01322v2 Announce Type: replace-cross  Abstract: Safety evaluation of Large Language Models (LLMs) remains heavily English-centric, leaving Low-Resource Languages (LRLs), particularly African ones, critically underexplored. We introduce TUKABENCH, a jailbreak benchmark for seven African languages that extends JailbreakBench (JBB) beyond direct translation through four settings: human translation of JBB prompts, English adaptation to African contexts followed by human translation, human-curated prompts validated through interactions with GPT-5.2, and code-switched prompts combining English and African languages, isolating the effect of language, cultural grounding, and prompt evasiveness on model safety. Across closed and open models, prompting in African languages reduces refusal relative to English, with culturally adapted prompts leading to least refusal. The evaluation also surfaces two structural limitations: model comprehension failures and reduced LLM-as-a-judge reliabi
    
[^255]: MobileGym：一个面向移动GUI智能体研究的可验证且高度并行的仿真平台

    MobileGym: A Verifiable and Highly Parallel Simulation Platform for Mobile GUI Agent Research

    [https://arxiv.org/abs/2605.26114](https://arxiv.org/abs/2605.26114)

    MobileGym提出一个轻量级浏览器托管的移动GUI智能体仿真平台，通过结构化JSON状态实现确定性可验证评判，并凭借单服务器数百个低成本并行实例，首次为日常移动应用提供了可验证评估与可扩展在线强化学习能力。

    

    我们提出MobileGym，一个浏览器托管的、轻量级、完全可控的日常移动使用环境，旨在实现交互保真度而无需复制专有后端。它实现了两项此前在日常应用场景中难以企及的能力：通过基于结构化JSON状态的确定性评判机制提供可验证的结果信号，以及通过低成本的并行推演实现可扩展的在线强化学习。完整的环境状态以结构化JSON的形式被捕获、配置、分叉和比较，单台服务器可托管数百个并行实例，每个实例约占400 MB内存，冷启动时间约3秒。分层状态模型和声明式任务定义框架使状态可编程性与任务创建在大规模下保持实用，而单一的编程式评判机制可同时提供确定性评估判定和密集的强化学习奖励。随附的MobileGym-Bench提供416个参数化任务模板。

    arXiv:2605.26114v3 Announce Type: replace  Abstract: We present MobileGym, a browser-hosted, lightweight, fully controllable environment for everyday mobile use, targeting interaction fidelity without replicating proprietary backends. It enables two capabilities previously out of reach for everyday apps: verifiable outcome signals through deterministic state-based judging over structured JSON state, and scalable online RL through low-cost parallel rollouts. The full environment state is captured, configured, forked, and compared as structured JSON, and a single server can host hundreds of parallel instances, with about 400 MB memory per instance and about 3 s cold start. A layered state model and a declarative task-definition framework keep state programmability and task creation practical at scale, and a single programmatic judging mechanism delivers both deterministic evaluation verdicts and dense RL rewards. The accompanying MobileGym-Bench provides 416 parameterized task templates,
    
[^256]: 帮助陷入困境的客户：一个能够对话、探询与分流的LLM驱动智能体

    Helping Customers in Distress: An LLM-powered Agent that Converses, Probes, and Routes

    [https://arxiv.org/abs/2605.16268](https://arxiv.org/abs/2605.16268)

    本文开发了一个基于大语言模型的银行客户分流智能体，通过多轮对话探询客户问题并按政策精准分流至专业团队，同时利用真实客户的合成数字孪生生成带标签对话来评估和持续改进该系统。

    

    银行每年收到数百万起关于欺诈、诈骗和争议交易的报告，这使得将客户准确引导至合适的专业支持团队变得极具挑战性。现有的人工处理流程不仅速度缓慢，也给客户和员工都带来压力。为解决这一问题，我们开发了一个面向客户的AI分流智能体，它利用大语言模型（LLM）进行多轮对话、提出相关问题并对案例进行分类，以实现准确的、政策引导下的分流，并将其嵌入到客户服务旅程中。为评估并持续改进该智能体，我们基于历史数据模拟了真实客户的合成数字孪生体，生成真实的、带有标签的对话，以测试广泛的现实场景。本工作详细介绍了该分流智能体的建模方法、与政策的集成、安全护栏与推理框架，以及合成智能体的使用方式。

    arXiv:2605.16268v2 Announce Type: replace-cross  Abstract: Banks receive millions of reports of fraud, scams, and disputed transactions every year, making it challenging to accurately direct customers to the appropriate specialist teams for assistance. The existing manual process driven by humans is slow and stressful for both customers and staff. To address this, we develop a customer-facing AI powered triaging agent that leverages large language models (LLMs) to conduct multi-turn conversations, ask relevant questions, and classify cases for accurate, policy-guided routing, making it embedded in the customer journey. To evaluate and continuously improve the agent, synthetic digital twins of real customers were simulated, generating realistic, labelled dialogues based on historical data to test a wide range of real-world scenarios. This work details the triage agent's modelling approach, integration with policy, safety guardrails and reasoning frameworks, the use of the synthetic agen
    
[^257]: DreamAvoid：通过关键阶段测试时“做梦”来避免VLA策略的失败

    DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies

    [https://arxiv.org/abs/2605.11750](https://arxiv.org/abs/2605.11750)

    提出DreamAvoid框架，通过“做梦触发器”检测关键阶段、采样候选动作并用混合数据训练的“做梦评估器”进行评估，使VLA模型在测试时能够预见并避免细粒度操作中的失败。

    

    视觉-语言-动作（VLA）模型在细粒度操作任务中往往表现脆弱，在关键阶段发生的微小动作误差可能迅速演变为不可挽回的失败。由于现有VLA模型在训练时主要依赖成功示范，它们在这些关键阶段缺乏对失败的显式感知。为了解决这一问题，我们提出了DreamAvoid，一个关键阶段测试时“做梦”框架，使VLA模型能够预见并避免失败。我们还引入了一种自主边界学习范式，以细化系统对成功与失败之间微妙边界的理解。具体而言，我们（1）利用“做梦触发器”判断执行是否已进入关键阶段，（2）通过“动作提议器”从VLA模型中采样多个候选动作块，（3）并采用“做梦评估器”——在成功、失败和边界案例的混合数据上联合训练——来"dr（原文摘要在此处截断）

    arXiv:2605.11750v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models are often brittle in fine-grained manipulation, where minor action errors during the critical phases can rapidly escalate into irrecoverable failures. Since existing VLA models rely predominantly on successful demonstrations for training, they lack an explicit awareness of failure during these critical phases. To address this, we propose DreamAvoid, a critical-phase test-time dreaming framework that enables VLA models to anticipate and avoid failures. We also introduce an autonomous boundary learning paradigm to refine the system's understanding of the subtle boundary between success and failure. Specifically, we (1) utilize a Dream Trigger to determine whether the execution has entered a critical phase, (2) sample multiple candidate action chunks from the VLA via an Action Proposer, and (3) employ a Dream Evaluator, jointly trained on mixed data (success, failure, and boundary cases), to "dr
    
[^258]: ProteinJEPA：潜在预测改进蛋白质语言模型预训练

    ProteinJEPA: Latent prediction improves protein language model pretraining

    [https://arxiv.org/abs/2605.07554](https://arxiv.org/abs/2605.07554)

    ProteinJEPA在蛋白质语言模型的掩码语言建模基础上引入JEPA式潜在表示预测损失，显著提升了模型在蛋白质检索和远程同源性检测等结构与同源性敏感任务上的表现，且增益随模型规模增大而增强。

    

    蛋白质语言模型主要以掩码语言建模（MLM）进行训练，即预测被掩码的氨基酸身份。联合嵌入预测架构（JEPA）则改为预测潜在表示，但尚未被应用于蛋白质领域。ProteinJEPA在MLM的基础上增加了一个余弦损失，用于在给定未掩码序列的条件下预测教师模型的半深度隐藏状态。在19个任务上，采用3500万和1.5亿参数的ESM2模型以及三个预训练随机种子，MLM+JEPA在114次比较中分别有78次和76次优于计算量匹配和训练步数匹配的仅MLM持续训练（14次落后，22次平局）。在结构与同源性敏感的任务上，计算量匹配的中位数增益为+0.0106，而其他任务上仅为+0.0041，其中以SCOPe-40检索和远程同源性任务提升最为显著，分别实现了Recall@1提高6.1个百分点和准确率提高2.7个百分点。这些任务上的增益随模型规模从800万增加到1.5亿而持续增大。

    arXiv:2605.07554v2 Announce Type: replace-cross  Abstract: Protein language models are trained primarily with masked language modeling (MLM), which predicts masked amino-acid identities. Joint-embedding predictive architectures (JEPA) instead predict latent representations, but have not been applied to proteins.   ProteinJEPA supplements MLM with a cosine loss for predicting the half-depth hidden states of a teacher given the unmasked sequence. On 19 tasks, with ESM2 at 35M and 150M parameters and three pretraining seeds, MLM+JEPA outperforms compute-matched and step-matched MLM-only continued training in 78 and 76 of 114 comparisons (14 losses, 22 ties). The median compute-matched gain is $+0.0106$ on structure- and homology-sensitive tasks versus $+0.0041$ elsewhere, led by SCOPe-40 retrieval and remote homology with improvements of 6.1 percentage points in Recall@1 and 2.7 points in accuracy, respectively. Gains on these tasks increase with model size from 8M to 150M. Against the of
    
[^259]: EA-WM：具有结构化运动学-视觉动作场的事件感知生成式世界模型

    EA-WM: Event-Aware Generative World Model with Structured Kinematic-to-Visual Action Fields

    [https://arxiv.org/abs/2605.06192](https://arxiv.org/abs/2605.06192)

    提出 EA-WM，一种事件感知生成式世界模型，通过将动作与运动学状态直接投影到目标相机视图形成结构化运动学-视觉动作场，实现动作信号引导视频合成，从而在生成轨迹中保持精确的机器人空间几何与细粒度的机器人-物体交互动态。

    

    预训练视频扩散模型提供了强大的时空生成先验，使其成为构建机器人世界模型的天然基础。尽管近期的世界-动作模型联合优化未来视频与动作，但它们大多将视频生成视为策略学习的辅助表示。因此，它们对逆问题的探索不足：即利用动作信号来引导视频合成，从而在生成的 rollout 中往往无法保持精确的机器人空间几何以及细粒度的机器人-物体交互动态。为弥合这一差距，我们提出了 EA-WM，一个事件感知的生成式世界模型，它有效地闭合了运动学控制与视觉感知之间的回路。EA-WM 并非将关节或末端执行器动作作为抽象的低维 token 注入，而是将动作和运动学状态直接投影到目标相机视图中，形成结构化的运动学-视觉动作场（Structured Kinematic-to-Visual Action Fields）……

    arXiv:2605.06192v2 Announce Type: replace-cross  Abstract: Pretrained video diffusion models provide powerful spatiotemporal generative priors, making them a natural foundation for robotic world models. While recent world-action models jointly optimize future videos and actions, they predominantly treat video generation as an auxiliary representation for policy learning. Consequently, they insufficiently explore the inverse problem: leveraging action signals to guide video synthesis, thereby often failing to preserve precise robot spatial geometry and fine-grained robot-object interaction dynamics in the generated rollouts. To bridge this gap, we present EA-WM, an Event-Aware Generative World Model that effectively closes the loop between kinematic control and visual perception. Rather than injecting joint or end-effector actions as abstract, low-dimensional tokens, EA-WM projects actions and kinematic states directly into the target camera view as Structured Kinematic-to-Visual Action
    
[^260]: 基于影子记忆保护LLM智能体免受长程威胁

    Safeguarding LLM Agents against Long-Horizon Threats via Shadow Memory

    [https://arxiv.org/abs/2605.03228](https://arxiv.org/abs/2605.03228)

    提出ShadowMem防御框架，借鉴系统安全中影子栈的思想，维护专门的影子记忆以在智能体完整执行轨迹中保留安全关键上下文，并在动作执行前主动评估风险，从而有效防御针对LLM智能体的长程攻击。

    

    随着大语言模型（LLM）驱动的智能体越来越多地被部署用于执行复杂的现实世界任务，它们面临着一类日益增多的攻击，这类攻击利用用户-智能体-环境之间的扩展交互来追求在单轮对话中难以实现的恶意目标。此类长程威胁对LLM智能体在关键领域的安全部署构成了重大风险。在本文中，我们提出了ShadowMem，这是一种旨在对抗多种长程威胁的新型防御框架。受系统安全中“影子栈”抽象概念的启发，ShadowMem维护一个专门的、以安全为中心的智能体记忆，该记忆在智能体的完整执行轨迹中提炼并保留安全关键上下文，并利用这一影子记忆在待执行动作执行之前主动评估其风险。大量评估表明，ShadowMem在各类长程威胁场景下显著优于现有防御方法。

    arXiv:2605.03228v2 Announce Type: replace-cross  Abstract: As large language model (LLM)-powered agents are increasingly deployed to perform complex, real-world tasks, they face a growing class of attacks that exploit extended user-agent-environment interactions to pursue malicious objectives improbable in single-turn settings. Such long-horizon threats pose significant risks to the safe deployment of LLM agents in critical domains. In this paper, we present ShadowMem, a novel defensive framework designed to counter a wide range of long-horizon threats. Inspired by the "shadow stack" abstraction in systems security, ShadowMem maintains a dedicated, safety-focused agentic memory that distills and retains safety-critical context across the agent's full execution trajectory, leveraging this shadow memory to proactively assess the risk of pending actions prior to their execution. Extensive evaluation demonstrates that ShadowMem substantially outperforms existing defenses across diverse lon
    
[^261]: ANO：通过有界、再下降的增益场实现鲁棒策略优化

    ANO: Robust Policy Optimization via Bounded, Redescending Gain Fields

    [https://arxiv.org/abs/2605.02320](https://arxiv.org/abs/2605.02320)

    提出锚定邻域优化（ANO），通过C^∞光滑的整形核直接构造有界且再下降的增益场，在PPO的死区漂移与SPO的无界增益这两个极端之间取得平衡，从而实现更稳定、更鲁棒的策略优化。

    

    近端策略优化（PPO）在强化学习和大语言模型对齐中占据主导地位，但其硬裁剪机制与无约束的替代方法（如SPO）处于稳定性-效率困境的两个极端。我们认为这一困境最好从动态角度来理解：代理目标函数本质上是关于概率比率的反馈律，其裁剪/惩罚的形状定义了驱动更新动态的增益场。PPO的裁剪会产生一个死区（信任区域之外反馈为零），导致策略在动量作用下开环漂移；而SPO的二次惩罚则产生无界且线性增长的增益，使动态系统刚化，在激进步长下失去稳定性。基于这一视角，我们提出了锚定邻域优化（ANO），直接设计增益场：一个C^∞光滑的整形核，在r=1处锚定恒等映射，并恰好在预设的信任区域边界1+ε处达到峰值，且在边界之外有界……（摘要原文在此处截断）

    arXiv:2605.02320v3 Announce Type: replace  Abstract: Proximal Policy Optimization (PPO) dominates reinforcement learning and LLM alignment, yet its hard-clipping mechanism and unconstrained alternatives (e.g., SPO) sit at two extremes of a stability-efficiency dilemma. We argue that this dilemma is best understood dynamically: a surrogate objective is a feedback law on the probability ratio, and its clipping/penalty shape defines a gain field that drives the update dynamics. PPO's clip induces a dead zone (zero feedback outside the trust region), leaving the policy to drift open-loop under momentum; SPO's quadratic penalty induces an unbounded, linearly growing gain that stiffens the dynamics and destabilizes under aggressive step sizes. Guided by this view, we derive Anchored Neighborhood Optimization (ANO), which designs the gain field directly: a $C^\infty$ shaping kernel that anchors the identity map at $r{=}1$, peaks exactly at a prescribed trust-region boundary $1{+}\epsilon$, bo
    
[^262]: Anon：将自适应性外推超越SGD与Adam

    Anon: Extrapolating Adaptivity Beyond SGD and Adam

    [https://arxiv.org/abs/2605.02317](https://arxiv.org/abs/2605.02317)

    该论文提出Anon优化器，突破了SGD与Adam之间0到1的插值限制，首次实现在整个实数范围内连续外推自适应参数（如CNN需要负自适应性、Transformer需要γ≥1），并通过增量延迟更新机制保证超界情形下的稳定收敛。

    

    诸如Adam的自适应优化器与SGD等非自适应方法在不同架构上表现出不同的泛化能力。先前的可调优化器试图通过在SGD和Adam之间严格插值来弥合这一差距，实际上将自适应性限制在了0到1的界限之内。然而，这种受限的插值从根本上是不充分的：我们揭示了最优的自适应性往往需要外推，例如经典CNN需要负的自适应性，而Transformer需要至少为1的自适应性（γ ≥ 1）。对自适应性进行外推在理论上违反了严格的非递减预条件子假设，常常导致现有方法发散。为了突破这一障碍，我们提出了Anon，一种在整个实数范围内实现完全连续自适应性外推的优化器。为了保证在这些超界情形下的可证明稳定性，我们引入了增量延迟更新机制……

    arXiv:2605.02317v3 Announce Type: replace  Abstract: Adaptive optimizers such as Adam and non-adaptive methods like SGD exhibit distinct generalization capabilities across different architectures. Prior tunable optimizers attempt to bridge this gap by strictly interpolating between SGD and Adam, effectively confining adaptivity within the 0-to-1 bound. However, this restricted interpolation is fundamentally insufficient: we reveal that optimal adaptivity often requires extrapolation, such as negative adaptivity for classical CNNs and adaptivity of at least one ($\gamma \geq 1$) for Transformers. Extrapolating adaptivity theoretically violates the strict non-decreasing pre-conditioner assumption, often leading to divergence in existing methods. To break this barrier, we propose Anon, an optimizer that achieves fully continuous adaptivity extrapolation across the entire real-number spectrum. To guarantee provable stability in these out-of-bound regimes, we introduce Incremental Delay Upd
    
[^263]: 预注册信念修订契约

    Preregistered Belief Revision Contracts

    [https://arxiv.org/abs/2604.15558](https://arxiv.org/abs/2604.15558)

    提出"预注册信念修订契约”（PBRC），通过公开固定证据触发器与修订规则、要求信念变更必须引用预注册触发器并附外部验证的证据令牌，从而将开放通信与认知变更严格分离，防止多智能体系统因从众效应而高置信度地收敛到错误结论。

    

    审议式多智能体系统允许智能体之间交换消息并随时间修订信念。虽然这种交互旨在提升性能，但也可能产生危险的从众效应：一致性、置信度、声望或多数规模可能被当作证据对待，从而导致智能体高置信度地收敛到错误结论。为解决这一问题，我们提出了PBRC（预注册信念修订契约），这是一种协议层机制，严格区分开放通信与可被接纳的认知变更。PBRC契约公开固定一阶证据触发器、可接纳的修订算子、优先级规则以及回退策略。只有当某个非回退步骤引用了预注册的触发器，并提供一个非空的、由外部验证的证据令牌见证集合时，该步骤才会被接受。这确保了每一次实质性的信念变更既可以由路由器强制执行，也可以事后审计。

    arXiv:2604.15558v2 Announce Type: replace  Abstract: Deliberative multi-agent systems allow agents to exchange messages and revise beliefs over time. While this interaction is meant to improve performance, it can also create dangerous conformity effects: agreement, confidence, prestige, or majority size may be treated as if they were evidence, producing high-confidence convergence to false conclusions. To address this, we introduce PBRC (Preregistered Belief Revision Contracts), a protocol-level mechanism that strictly separates open communication from admissible epistemic change. A PBRC contract publicly fixes first-order evidence triggers, admissible revision operators, a priority rule, and a fallback policy. A non-fallback step is accepted only when it cites a preregistered trigger and provides a nonempty witness set of externally validated evidence tokens. This ensures that every substantive belief change is both enforceable by a router and auditable after the fact. In this paper, 
    
[^264]: 迈向测量大语言模型通信环路中的结构性漂移

    Toward Measuring Structural Drift in LLM Communication Loops

    [https://arxiv.org/abs/2604.13061](https://arxiv.org/abs/2604.13061)

    该论文提出以“提示词→回复→下一个提示词”链条作为基本分析单元，并引入结构化通信一致性及其两个量化指标——通信闭合性与归一化条件动作贡献，用以测量LLM有状态管道中被传统逐条评估所忽略的结构性漂移。

    

    arXiv:2604.13061v3 公告类型：replace-cross。摘要：大语言模型越来越多地在有状态管道中运行，这些管道从检索、记忆、工具和其他智能体中组装每一条提示词。这样的管道会发生漂移：本应影响下一个回复的信息被丢弃、压缩或错误路由，而每个组件却仍然报告成功。现有的诊断方法无法发现这一问题，因为它们评估的是孤立的提示词、回复或任务分数，而真正发生解耦的是提示词与其所引发的回复之间的关系。本文证明，将“提示词→回复→下一个提示词”的链条作为基本分析单元，可以使这些关系变得可测量。我们引入了结构化通信一致性的概念，并通过两个指标进行量化：通信闭合性，即管道在某一轮返回的内容是否与它在下一轮所面对的内容相匹配；以及归一化条件动作贡献，用于衡量一条发送的消息在多大程度上促成了后续回复的完成。在2,171个人与人之间的、58个人与……（摘要截断）

    arXiv:2604.13061v3 Announce Type: replace-cross  Abstract: Large language models increasingly run in stateful pipelines that assemble each prompt from retrieval, memory, tools, and other agents. Such pipelines drift: information that should shape the next response is dropped, compressed, or misrouted while every component still reports success. Existing diagnostics miss this because they evaluate isolated prompts, responses, or task scores, whereas what decouples is the relation between a prompt and the response it draws. Here we show that treating the prompt to response to next prompt chain as the fundamental unit of analysis makes these relations measurable. We introduce structural communication coherence, quantified by two metrics: communication closure, which asks if what the pipeline returns at one turn matches what it faces next, and normalized conditional action contribution, which measures how much a sent message resolves the subsequent reply. Across 2,171 human to human, 58 hu
    
[^265]: 基于对抗式多任务学习的联合干扰检测与识别

    Joint Interference Detection and Identification via Adversarial Multi-task Learning

    [https://arxiv.org/abs/2604.08607](https://arxiv.org/abs/2604.08607)

    该论文建立了一个有理论支撑的多任务学习框架，通过推导加权期望损失上界，将任务相似度与Wasserstein距离和可学习的任务关系系数联系起来，并据此提出对抗式多任务网络，实现干扰检测、调制识别和干扰识别的联合处理。

    

    精确的干扰检测与识别对于提高通信系统在非合作无线环境中的生存能力至关重要。尽管深度学习（DL）已推动了该领域的发展，但现有的单任务学习（STL）方法忽略了任务间固有的相关性。此外，新兴的多任务学习（MTL）方法往往缺乏量化和建模任务关系的理论基础。为了弥补这一空白，我们建立了一个具有理论依据的多任务学习框架，用于联合干扰检测、调制识别和干扰识别。首先，我们推导了多任务学习框架中加权期望损失的上界。该上界明确地将多任务学习性能与任务相似度联系起来，任务相似度通过Wasserstein距离和可学习的任务关系系数来量化。在该理论的指导下，我们提出了对抗式多任务干扰检测与识别网络。

    arXiv:2604.08607v2 Announce Type: replace-cross  Abstract: Precise interference detection and identification are crucial for enhancing the survivability of communication systems in non-cooperative wireless environments. While deep learning (DL) has advanced this field, existing single-task learning (STL) approaches neglect inherent task correlations. Furthermore, emerging multi-task learning (MTL) methods often lack a theoretical foundation for quantifying and modeling task relationships. To bridge this gap, we establish a theoretically grounded MTL framework for joint interference detection, modulation identification, and interference identification. First, we derive an upper bound for the weighted expected loss in MTL frameworks. This bound explicitly connects MTL performance to task similarity, quantified by the Wasserstein distance and learnable task relation coefficients. Guided by this theory, we present the adversarial multi-task interference detection and identification network
    
[^266]: TiAb Review 插件：一个用于系统综述中 AI 辅助文献筛选的浏览器工具

    TiAb Review Plugin: A Browser-Based Tool for AI-Assisted Study Selection in Systematic Reviews

    [https://arxiv.org/abs/2604.08602](https://arxiv.org/abs/2604.08602)

    该论文开发了开源 Chrome 扩展 TiAb Review 插件，无需编程和服务器即可利用 AI 完成系统综述中从标题摘要到全文的文献筛选。

    

    基于服务器的筛选工具需要支付订阅费用，而开源替代方案则需要编程技能，且全文筛选一直不在无代码开源工具的能力范围之内。我们开发了 TiAb Review 插件，这是一个开源的 Chrome 浏览器扩展，提供无代码、无服务器的 AI 辅助文献筛选，同时涵盖标题与摘要（T&A）筛选和全文筛选两个阶段。该插件使用 Google Sheets 作为共享数据库、Google Drive 作为 PDF 存储库，用户只需提供自己的大语言模型（LLM）API 密钥。在 T&A 筛选方面，它支持人工审核、LLM 批量筛选和机器学习（ML）主动学习三种方式。在全文筛选方面，它可从 PubMed Central、Europe PMC、Unpaywall、OpenAlex 以及出版商网页自动获取开放获取的 PDF，支持带有结构化排除原因和裁决机制的盲法双人评审，并可选择获取附有页面锚定证据的 LLM 判断……

    arXiv:2604.08602v2 Announce Type: replace-cross  Abstract: Server-based screening tools impose subscription costs, while open-source alternatives require coding skills, and full-text screening has remained outside the scope of no-code open-source tools. We developed TiAb Review Plugin, an open-source Chrome browser extension that provides no-code, serverless artificial intelligence (AI)-assisted study selection covering both title and abstract (T&A) screening and full-text screening. It uses Google Sheets as a shared database and Google Drive as a PDF store, and users supply their own large language model (LLM) API key. For T&A screening, it offers manual review, LLM batch screening, and machine learning (ML) active learning. For full-text screening, it retrieves open-access PDFs from PubMed Central, Europe PMC, Unpaywall, OpenAlex, and publisher pages, supports blinded dual review with structured exclusion reasons and adjudication, optionally obtains an LLM judgment with page-anchored
    
[^267]: LitPivot：通过文献图景中的动态情境化与批判来发展立意恰当的研究想法

    LitPivot: Developing Well-Situated Research Ideas Through Dynamic Contextualization and Critique within the Literature Landscape

    [https://arxiv.org/abs/2604.02600](https://arxiv.org/abs/2604.02600)

    提出了 LitPivot 系统，通过“文献引发的转向”机制实现研究想法与文献的动态互动——与文献的互动促进想法修订，想法修订又更新相关文献检索，从而帮助研究者在构思过程中形成立意恰当的研究想法。

    

    开发一个新颖的研究想法很难。它必须与先前工作有足够区别才能声称贡献，同时又要在先前工作之上构建。这需要研究者迭代地回顾文献，并根据所阅读的内容不断完善想法；然而当想法发生变化时，重要的相关文献往往也随之改变。大多数工具对这种相互作用的支持有限：文献工具帮助研究者理解一个固定的文献体系，而构思工具则依据静态的、预先筛选的论文集来评估想法。我们提出了“文献引发的转向”，这是一种机制，其中与文献的互动会促使发展中的想法进行修订，而这种修订又会改变哪些文献是相关的。我们在 LitPivot 中实现了这一机制，使研究者能够同时起草并审查研究想法。LitPivot 会动态检索与想法中所选部分相关的论文聚类，并提出基于文献的批判性意见，指导如何修订……（摘要在此处被截断）

    arXiv:2604.02600v3 Announce Type: replace-cross  Abstract: Developing a novel research idea is hard. It must be distinct enough from prior work to claim a contribution while also building on it. This requires iteratively reviewing literature and refining an idea based on what a researcher reads; yet when an idea changes, the literature that matters often changes with it. Most tools offer limited support for this interplay: literature tools help researchers understand a fixed body of work, while ideation tools evaluate ideas against a static, pre-curated set of papers. We introduce literature-initiated pivots, a mechanism where engagement with literature prompts revision to a developing idea, and where that revision changes which literature is relevant. We operationalize this in LitPivot, where researchers concurrently draft and vet an idea. LitPivot dynamically retrieves clusters of papers relevant to a selected part of the idea and proposes literature-informed critiques for how to rev
    
[^268]: 用于方差最小化和风险规避多臂老虎机的Softmax梯度策略

    Softmax gradient policy for variance minimization and risk-averse multi armed bandits

    [https://arxiv.org/abs/2604.00241](https://arxiv.org/abs/2604.00241)

    该论文提出了一种基于softmax参数化的新算法，用于在风险规避的多臂老虎机问题中选择方差最小（风险最低）的臂，通过两次独立抽样构建无偏估计并证明了算法的收敛性。

    

    多臂老虎机（MAB）问题的算法在序贯决策中扮演着核心角色，并已在理论和数值方面得到广泛探索。虽然大多数经典方法旨在识别期望奖励最高的臂，但我们专注于一个风险感知的设定，其目标是选择方差最低的臂，即优先考虑稳定性而非潜在的高但不确定的回报。为了建模决策过程，我们考虑了策略的softmax参数化；我们提出了一种新算法来选择最小方差（或最小风险）的臂，并在自然条件下证明了其收敛性。该算法通过从所选臂的分布中进行两次独立抽样来构建目标函数的无偏估计。我们提供了数值实验，展示了这些算法的实际行为，并为实现选择提供了指导。该设定还涵盖了一般性的……

    arXiv:2604.00241v2 Announce Type: replace-cross  Abstract: Algorithms for the Multi-Armed Bandit (MAB) problem play a central role in sequential decision-making and have been extensively explored both theoretically and numerically. While most classical approaches aim to identify the arm with the highest expected reward, we focus on a risk-aware setting where the goal is to select the arm with the lowest variance, favoring stability over potentially high but uncertain returns. To model the decision process, we consider a softmax parameterization of the policy; we propose a new algorithm to select the minimal variance (or minimal risk) arm and prove its convergence under natural conditions. The algorithm constructs an unbiased estimate of the objective by using two independent draws from the selected arm's distribution. We provide numerical experiments that illustrate the practical behavior of these algorithms and offer guidance on implementation choices. The setting also covers general 
    
[^269]: 基于指标的人工意识评估中的校准与迁移

    Calibration and transfer in indicator-based assessments of artificial consciousness

    [https://arxiv.org/abs/2603.27597](https://arxiv.org/abs/2603.27597)

    本文指出基于指标的人工意识评估面临两大问题——概率赋值无法校准以及证据相关性跨基底迁移缺乏独立支持，并提出通过构建初步的理论相对比较空间来实现跨基底评估。

    

    对人工意识的研究正日益将评估重点从行为转向内部架构。基于理论的指标被用于更新概率赋值。这相较于行为测试有所改进，但引发了两个不同的问题。首先，这些概率赋值目前无法针对独立确立的人工意识结果进行校准。其次，其证据相关性是从生物案例中迁移而来的，而缺乏独立的支持来证明指标与意识之间的关系在不同基底之间保持稳定。本评论区分了校准与迁移这两个概念，并通过对迭代自然类策略进行改造，提出了一个初步的、相对于理论的跨基底评估比较空间。

    arXiv:2603.27597v2 Announce Type: replace  Abstract: Research on artificial consciousness increasingly shifts evaluation from behaviour to internal architecture. Theory-based indicators are used to update probability assignments. This improves on behavioural tests but raises two distinct problems. First, these assignments cannot currently be calibrated against independently established artificial consciousness outcomes. Second, their evidential relevance is transferred from biological cases without independent support that indicator-consciousness relations remain stable across substrates. This commentary distinguishes calibration from transfer and adapts the iterative natural-kind strategy by proposing a preliminary, theory-relative comparative space for cross-substrate assessment.
    
[^270]: FSCE：一种面向噪声鲁棒SAR自动目标识别的目标感知频率-空间协同增强框架

    FSCE: A Target-Aware Frequency-Spatial Collaborative Enhancement Framework for Noise-Resilient SAR ATR

    [https://arxiv.org/abs/2603.21565](https://arxiv.org/abs/2603.21565)

    提出了FSCE框架，通过在网络入口处进行频率-空间协同增强以抑制斑点噪声传播、稳定浅层特征，并结合自适应策略驱动的语义对齐机制施加自上而下的语义约束，从而显著提升噪声环境下SAR自动目标识别的鲁棒性。

    

    合成孔径雷达自动目标识别（SAR ATR）受到相干斑点噪声的严重挑战，其干扰会被层次化的非线性变换逐步放大，并最终损害高层语义表示。为解决这一问题，我们提出了一种面向噪声鲁棒SAR ATR的目标感知频率-空间协同增强（FSCE）框架，该框架将用于早期特征稳定的频率-空间建模与语义正则化相结合。具体而言，我们在网络入口处设计了一个频率-空间早期自适应增强（FS-EAE）模块，通过协同的空间-频率建模来抑制噪声传播并保留目标结构。在稳定的浅层表示基础上，我们进一步引入了自适应策略驱动的语义对齐（APSA）机制，该机制利用在线教师策略施加自上而下的语义约束……

    arXiv:2603.21565v2 Announce Type: replace-cross  Abstract: Synthetic aperture radar automatic target recognition (SAR ATR) is severely challenged by coherent speckle noise, whose interference can be progressively amplified by hierarchical nonlinear transformations and eventually damage high-level semantic representations. To address this issue, we propose a Target-Aware Frequency-Spatial Collaborative Enhancement (FSCE) framework for noise-resilient SAR ATR, which integrates frequency-spatial modeling for early feature stabilization with semantic regularization. Specifically, we design a Frequency-Spatial Early-stage Adaptive Enhancement (FS-EAE) module at the network entrance to suppress noise propagation and preserve target structures through collaborative spatial-frequency modeling. Building upon stabilized shallow representation, we further introduce an Adaptive Policy-driven Semantic Alignment (APSA) mechanism, which uses an online teacher policy to impose top-down semantic constr
    
[^271]: 测量与利用LLM辅助安全代码审查中的上下文偏差

    Measuring and Exploiting Contextual Bias in LLM-Assisted Security Code Review

    [https://arxiv.org/abs/2603.18740](https://arxiv.org/abs/2603.18740)

    本研究揭示了框架效应会导致基于LLM的自动化代码审查系统在漏洞检测中产生系统性且普遍的偏差，并证明攻击者可以通过注入带有偏差的PR元数据，将其利用为针对ACR流水线的供应链攻击向量。

    

    集成大语言模型（LLM）的自动化代码审查（ACR）系统在软件开发工作流程中的应用日益广泛，其形式涵盖从交互式助手到CI/CD流水线中的自主代理。在本文中，我们研究了ACR中基于LLM的漏洞检测如何受到框架效应的影响：即在形成判断时，信息呈现方式凌驾于其语义内容之上的倾向。我们考察了对手是否可以通过上下文偏差注入来利用这一效应（即通过精心构造PR元数据来偏置ACR的安全判断），并将其作为针对真实世界ACR流水线的供应链攻击向量。为此，我们首先在五种框架条件下对6个LLM进行了大规模探索性研究，确立了框架效应是基于LLM的漏洞检测中一种系统性且普遍存在的现象。随后，我们设计了一个真实且受控的实验环境，对20个（项目/代码库）中的33个CVE进行了评估。

    arXiv:2603.18740v3 Announce Type: replace-cross  Abstract: Automated Code Review (ACR) systems integrating Large Language Models (LLMs) are increasingly adopted in software development workflows, ranging from interactive assistants to autonomous agents in CI/CD pipelines. In this paper, we study how LLM-based vulnerability detection in ACR is affected by the framing effect: the tendency to let the presentation of information override its semantic content in forming judgments. We examine whether adversaries can exploit this through contextual-bias injection (crafting PR metadata to bias ACR security judgments) as a supply-chain attack vector against real-world ACR pipelines. To this end, we first conduct a large-scale exploratory study across 6 LLMs under five framing conditions, establishing the framing effect as a systematic and widespread phenomenon in LLM-based vulnerability detection.   We then design a realistic and controlled experimental environment, evaluating 33 CVEs across 20
    
[^272]: 看向关键之处：面向高效视觉语言模型的高分辨率裁剪区域检索

    Look Where It Matters: High-Resolution Crops Retrieval for Efficient VLMs

    [https://arxiv.org/abs/2603.16932](https://arxiv.org/abs/2603.16932)

    该论文提出 AwaRes 框架，让视觉语言模型基于低分辨率全局视图，通过工具调用按需检索与查询相关的高分辨率裁剪区域，并自动构建监督数据训练模型，从而在不牺牲精度的前提下大幅提升计算效率。

    

    视觉语言模型（VLM）通常以原生高分辨率处理图像，这迫使模型在精度与计算效率之间进行权衡：高分辨率输入能够捕捉精细细节，但会带来显著的计算开销；低分辨率输入虽然有利于效率，却可能遗漏关键的视觉信息，例如小号文字。我们提出了 AwaRes，这是一个按需空间处理框架，通过在低分辨率全局视图上运行，并利用工具调用仅检索针对给定查询所需的高分辨率片段，从而化解了这一精度-效率权衡问题。我们自动构建监督数据：由一个评判模型比较低分辨率与高分辨率下的答案来标注是否需要裁剪，并由一个 oracle 定位模型找出正确答案的证据位置，再将其映射到离散的裁剪区域集合上，形成多轮工具使用轨迹。我们通过冷启动 SFT 以及随后的多……来训练该框架。

    arXiv:2603.16932v2 Announce Type: replace-cross  Abstract: Vision-language models (VLMs) typically process images at a native high-resolution, forcing a trade-off between accuracy and computational efficiency: high-resolution inputs capture fine details but incur significant computational costs, while low-resolution inputs advocate for efficiency, they potentially miss critical visual information, like small text. We present AwaRes, a spatial-on-demand framework that resolves this accuracy-efficiency trade-off by operating on a low-resolution global view and using tool-calling to retrieve only high-resolution segments needed for a given query. We construct supervised data automatically: a judge compares low- vs.\ high-resolution answers to label whether cropping is needed, and an oracle grounding model localizes the evidence for the correct answer, which we map to a discrete crop set to form multi-turn tool-use trajectories. We train our framework with cold-start SFT followed by multi-
    
[^273]: MessyKitchens：富含接触信息的物体级3D场景重建

    MessyKitchens: Contact-rich object-level 3D scene reconstruction

    [https://arxiv.org/abs/2603.16868](https://arxiv.org/abs/2603.16868)

    该论文提出了MessyKitchens数据集，为杂乱的真实场景提供包含物体3D形状、姿态和精确接触信息的高保真物体级真值，以推动物理合理的物体级3D场景重建研究。

    

    单目3D场景重建近年来取得了显著进展。借助现代神经网络架构和大规模数据，最新方法在从单张图像进行深度估计方面已经实现了高性能。然而，由于物体种类繁多、频繁的遮挡以及复杂的物体间关系，将常见场景重建并分解为独立的3D物体仍然是一个难题。值得注意的是，除了单个物体的形状和姿态估计之外，机器人技术和动画领域的应用还需要物理上合理的场景重建，即物体遵循不相互穿透的物理原理并具有真实的接触。在这项工作中，我们从两个方向推进物体级场景重建。首先，我们提出了MessyKitchens，这是一个包含真实世界中杂乱环境场景的新数据集，并在3D物体形状、姿态以及精确的物体接触方面提供高保真度的物体级真值标注。

    arXiv:2603.16868v2 Announce Type: replace-cross  Abstract: Monocular 3D scene reconstruction has recently seen significant progress. Powered by the modern neural architectures and large-scale data, recent methods achieve high performance in depth estimation from a single image. Meanwhile, reconstructing and decomposing common scenes into individual 3D objects remains a hard challenge due to the large variety of objects, frequent occlusions and complex object relations. Notably, beyond shape and pose estimation of individual objects, applications in robotics and animation require physically-plausible scene reconstruction where objects obey physical principles of non-penetration and realistic contacts. In this work we advance object-level scene reconstruction along two directions. First, we introduceMessyKitchens, a new dataset with real-world scenes featuring cluttered environments and providing high-fidelity object-level ground truth in terms of 3D object shapes, poses and accurate obj
    
[^274]: InterPol：通过插值偏好学习对LM Arena进行去匿名化

    InterPol: De-anonymizing LM Arena via Interpolated Preference Learning

    [https://arxiv.org/abs/2603.15220](https://arxiv.org/abs/2603.15220)

    INTERPOL通过模型插值合成困难负样本并结合自适应课程学习，捕捉深层风格特征，显著提升了对LM Arena等匿名排行榜中目标模型（尤其是风格相似的模型）的去匿名化识别准确率。

    

    模型回复的严格匿名性是LM Arena等基于投票的排行榜可靠性的关键。虽然先前的研究曾尝试使用TF-IDF或词袋等简单统计特征来破坏这一匿名性假设，但这些方法往往缺乏区分风格相似或同一家族模型的判别能力。为了克服这些局限并揭示该漏洞的严重性，我们提出了INTERPOL，一个模型驱动的识别框架，它利用插值偏好数据学习将目标模型与其他模型区分开来。具体而言，INTERPOL通过模型插值合成困难负样本，并采用自适应课程学习策略，捕捉到了浅层统计特征所遗漏的深层风格模式。大量实验表明，INTERPOL在识别准确率上显著优于现有基线方法。此外，我们还量化了……（原文摘要在此处截断）

    arXiv:2603.15220v2 Announce Type: replace  Abstract: Strict anonymity of model responses is a key for the reliability of voting-based leaderboards, such as LM Arena. While prior studies have attempted to compromise this assumption using simple statistical features like TF-IDF or bag-ofwords, these methods often lack the discriminative power to distinguish between stylistically similar or within-family models. To overcome these limitations and expose the severity of vulnerability, we introduce INTERPOL, a model-driven identification framework that learns to distinguish target models from others using interpolated preference data. Specifically, INTERPOL captures deep stylistic patterns that superficial statistical features miss by synthesizing hard negative samples through model interpolation and employing an adaptive curriculum learning strategy. Extensive experiments demonstrate that INTERPOL significantly outperforms existing baselines in identification accuracy. Furthermore, we quant
    
[^275]: 利用视觉语言基础模型通过上下文学习生成植物模拟配置

    Using Vision Language Foundation Models to Generate Plant Simulation Configurations via In-Context Learning

    [https://arxiv.org/abs/2603.08930](https://arxiv.org/abs/2603.08930)

    该论文提出了一个评估基准，证明视觉语言基础模型可通过上下文学习从图像生成有效的植物模拟JSON配置，并能够估计播种后天数、植株数量、位置等关键参数。

    

    本文引入了一个基准，用于评估视觉语言模型（VLM）能否通过上下文学习从图像生成植物模拟配置。我们针对豇豆小区重建任务对该基准进行了研究，其中VLM需要生成包含田间和植物信息的结构化JSON配置。来自Gemma 4和Qwen3.5系列的开源多模态模型在一个具有已知JSON真值的合成豇豆数据集以及一个带有田间采集JSON的真实无人机正射影像数据集上进行了评估。研究采用了五种上下文学习方法，从格式限制指令到带有辅助定位信息的少样本图像示例。结果表明，VLM能够生成有效的JSON输出，通常可以估计播种后天数（DAP）、植株数量、植株位置、太阳角度和叶片叶绿素含量，并能渲染出豇豆小区的近似模拟。

    arXiv:2603.08930v2 Announce Type: replace-cross  Abstract: This paper introduces a benchmark for evaluating whether vision-language models (VLMs) can generate plant simulation configurations from imagery using in-context learning. We study this benchmark for cowpea plot reconstruction for plant simulations, where the VLM needs to generate structured JSON configurations that include field and plant information. Open-source multimodal models from Gemma 4 and Qwen3.5 families are evaluated on a synthetic cowpea dataset with known JSON ground truth and on a real drone orthophoto dataset with field-collected JSON. Five in-context learning methods are used, from format restriction instruction to few-shot image examples with auxiliary grounding information. The results show that VLMs can generate valid JSON outputs, can generally estimate days after planting (DAP), plant counts, plant locations, sun angles, and leaf chlorophyll content, and can render approximate simulations of cowpea plots. 
    
[^276]: Med-V1：面向零样本且可扩展的生物医学证据归因的小型语言模型

    Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution

    [https://arxiv.org/abs/2603.05308](https://arxiv.org/abs/2603.05308)

    本研究提出仅有三十亿参数的小型语言模型家族Med-V1，通过新开发的高质量合成数据训练，在生物医学证据归因任务上以极低成本达到媲美GPT-5等前沿大模型的性能，并首次量化了LLM生成答案中的幻觉现象。

    

    评估一篇文章是否支持某一断言，对于幻觉检测和声明验证至关重要。虽然大型语言模型（LLMs）有潜力将这一任务自动化，但要取得强大性能需要依赖GPT-5等前沿模型，而这些模型在大规模部署时的成本高得令人望而却步。为了高效地执行生物医学证据归因任务，我们提出了Med-V1，这是一个仅有三十亿参数的小型语言模型家族。Med-V1在本研究中新开发的高质量合成数据上训练，在统一为验证格式的五个生物医学基准测试上大幅超越其基础模型（提升27.0%至71.3%）。尽管模型规模较小，Med-V1的性能可与GPT-5等前沿LLMs相媲美，并能为其预测提供高质量的解释。我们利用Med-V1开展了首次同类用例研究，量化了LLM生成的答案在不同引用情境下的幻觉现象。

    arXiv:2603.05308v4 Announce Type: replace-cross  Abstract: Assessing whether an article supports an assertion is essential for hallucination detection and claim verification. While large language models (LLMs) have the potential to automate this task, achieving strong performance requires frontier models such as GPT-5 that are prohibitively expensive to deploy at scale. To efficiently perform biomedical evidence attribution, we present Med-V1, a family of small language models with only three billion parameters. Trained on high-quality synthetic data newly developed in this study, Med-V1 substantially outperforms (+27.0% to +71.3%) its base models on five biomedical benchmarks unified into a verification format. Despite its smaller size, Med-V1 performs comparably to frontier LLMs such as GPT-5, along with high-quality explanations for its predictions. We use Med-V1 to conduct a first-of-its-kind use case study that quantifies hallucinations in LLM-generated answers under different cit
    
[^277]: 一个超大规模视频推理套件

    A Very Big Video Reasoning Suite

    [https://arxiv.org/abs/2602.20159](https://arxiv.org/abs/2602.20159)

    本文介绍了VBVR数据集和VBVR-Bench评估框架，前者规模比现有数据集大三个数量级，后者采用基于规则且与人类对齐的评分器，以系统研究视频推理能力及其扩展行为。

    

    arXiv:2602.20159v3 公告类型：交叉替换 摘要：视频模型的快速进步主要聚焦于视觉质量，而其推理能力仍未得到充分探索。视频推理将智能锚定在超越文本自然捕捉能力的时空一致视觉环境中，能够对连续性、交互和因果性等时空结构进行直觉推理。然而，系统研究视频推理及其扩展行为因缺乏大规模训练数据而受阻。为弥补这一空白，我们引入了超大规模视频推理（VBVR）数据集，这是一个前所未有的超大规模资源，涵盖200个按原则性分类法组织的精选推理任务和超过一百万段视频片段，比现有数据集大约大三个数量级。我们还提出了VBVR-Bench，一个可验证的评估框架，通过结合基于规则且与人类对齐的评分器，超越了基于模型的评判。

    arXiv:2602.20159v3 Announce Type: replace-cross  Abstract: Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, 
    
[^278]: LORA-CRAFT：通过预训练注意力权重的冻结塔克分解实现跨层秩自适应

    LORA-CRAFT: Cross-layer Rank Adaptation via Frozen Tucker Decomposition of Pre-trained Attention Weights

    [https://arxiv.org/abs/2602.17510](https://arxiv.org/abs/2602.17510)

    CRAFT通过将预训练注意力权重组织为跨层3D张量并应用冻结的塔克分解，仅训练小型方形矩阵，实现了比现有方法更参数高效的微调。

    

    arXiv:2602.17510v2 公告类型：替换-交叉 摘要：我们引入了LoRA-CRAFT（通过冻结塔克分解的跨层秩自适应），全文简称为CRAFT，这是一种极其参数高效的微调（PEFT）方法，它将跨Transformer层堆叠的预训练注意力权重矩阵应用塔克张量分解，并仅在由此产生的冻结塔克因子上训练小型方形自适应矩阵。现有的基于张量的PEFT方法分解梯度更新：LoTR应用带有共享因子矩阵的塔克分解，而SuperLoRA在应用塔克分解前对跨层的ΔW进行分组和重塑。另外，像PiSSA这样的方法对预训练权重应用SVD，但逐层独立操作。CRAFT弥合了这两类工作：它通过高阶SVD（HOSVD）直接对组织为跨层3D张量的预训练权重进行完整塔克分解。

    arXiv:2602.17510v2 Announce Type: replace-cross  Abstract: We introduce LoRA-CRAFT (\textbf{C}ross-layer \textbf{R}ank \textbf{A}daptation via \textbf{F}rozen \textbf{T}ucker), abbreviated CRAFT throughout, an extremely parameter-efficient fine-tuning (PEFT) method that applies Tucker tensor decomposition to pre-trained attention weight matrices stacked across transformer layers and trains only small square adaptation matrices on the resulting frozen Tucker factors. Existing tensor-based PEFT methods decompose \textit{gradient updates}: LoTR applies Tucker decomposition with shared factor matrices, while SuperLoRA groups and reshapes $\Delta W$ across layers before applying Tucker decomposition. Separately, methods such as PiSSA apply SVD to \textit{pre-trained weights} but operate independently per layer. CRAFT bridges these two lines of work: it performs full Tucker decomposition via Higher-Order SVD (HOSVD) directly on \textit{pre-trained weights} organized as cross-layer 3D tensors
    
[^279]: 共享状态认知模型中的情境信息分配：一个信息论界限

    Contextual Information Allocation in Shared-State Cognitive Models: An Information-Theoretic Bound

    [https://arxiv.org/abs/2602.16716](https://arxiv.org/abs/2602.16716)

    本文为共享状态认知架构建立了信息论下界：行为中残留的情境依赖性决定了辅助情境中介变量必须携带的最小信息量与条件熵。

    

    arXiv:2602.16716v4 公告类型：替换。摘要：情境敏感行为可以通过三种方式建模：丰富内部状态、允许反应规则直接访问情境，或者在保留共享状态的同时引入辅助准则或控制变量。本文针对第三种架构分离出一个信息论约束。设 $C$ 表示情境，$S$ 表示候选的内部或潜在状态，$O$ 表示可观察的反应，$M$ 表示辅助变量，满足 $O\perp C\mid(S,M)$。则有不等式 $I(C;O\mid S)\le I(C;M\mid S)\le H(M\mid S)$。因此，一旦共享状态被指定，行为中残留的情境依赖性就同时为辅助情境中介机制所必须携带的情境信息和条件熵提供了下界。该界限是相对于表征的，而非状态空间大小的度量或通用的情境性度量。一个识别记忆的计算实例展示了如何针对收益诱发的……（原文摘要在此处截断）

    arXiv:2602.16716v4 Announce Type: replace  Abstract: Context-sensitive behavior can be modeled by enriching an internal state, by allowing a response rule to access context directly, or by preserving a shared state while introducing an auxiliary criterion or control variable. This paper isolates an information-theoretic constraint on the third architecture. Let $C$ denote context, $S$ a candidate internal or latent state, $O$ an observable response, and $M$ an auxiliary variable such that $O\perp C\mid(S,M)$. Then \[ I(C;O\mid S)\le I(C;M\mid S)\le H(M\mid S). \] Once a shared state has been specified, residual context dependence in behavior therefore lower-bounds both the context information and the conditional entropy that an auxiliary context-mediating mechanism must carry. The bound is representation-relative rather than a measure of state-space size or a universal contextuality measure. A worked recognition-memory example shows how the quantity can be computed for payoff-induced c
    
[^280]: 基于检索增强（知识图谱）与大型语言模型驱动的信息物理系统设计结构矩阵（DSM）生成

    Retrieval Augmented (Knowledge Graph), and Large Language Model-Driven Design Structure Matrix (DSM) Generation of Cyber-Physical Systems

    [https://arxiv.org/abs/2602.16715](https://arxiv.org/abs/2602.16715)

    本文探索利用大型语言模型、检索增强生成（RAG）和图谱RAG（GraphRAG）自动生成信息物理系统的设计结构矩阵（DSM），并通过电动螺丝刀和立方星两个案例验证了其在组件识别与关系确定任务上的有效性。

    

    我们探索了大型语言模型（LLM）、检索增强生成（RAG）和基于图谱的RAG（GraphRAG）在生成设计结构矩阵（DSM）方面的潜力。我们在两个不同的用例上测试了这些方法——一个电动螺丝刀和一个具有已知架构参考的立方星——评估它们在两项关键任务上的表现：确定预定义组件之间的关系，以及更具挑战性的识别组件及其后续关系。我们通过评估DSM的每个元素和整体架构来衡量性能。尽管面临设计和计算方面的挑战，我们发现了自动生成DSM的机会，所有代码均公开可用，以便于结果复现并获取领域专家的进一步反馈。

    arXiv:2602.16715v2 Announce Type: replace  Abstract: We explore the potential of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Graph-based RAG (GraphRAG) for generating Design Structure Matrices (DSMs). We test these methods on two distinct use cases--a power screwdriver and a CubeSat with known architectural references--evaluating their performance on two key tasks: determining relationships between predefined components, and the more complex challenge of identifying components and their subsequent relationships. We measure the performance by assessing each element of the DSM and overall architecture. Despite design and computational challenges, we identify opportunities for automated DSM generation, with all code publicly available for reproducibility and further feedback from the domain experts.
    
[^281]: 自我提升即一致性优化：一个理论性解释

    Self-Improvement as Coherence Optimization: A Theoretical Account

    [https://arxiv.org/abs/2601.13566](https://arxiv.org/abs/2601.13566)

    该论文提出统一理论框架，证明辩论、自举与内部一致性最大化等无监督自我提升方法本质上都是“一致性优化”，等价于描述长度正则化，其中基于预训练先验的一致性正则化可优化半监督学习最坏情况准确率的下界，从而在理论上解释了无需反馈的自我提升为何有效。

    

    语言模型能否在缺乏外部监督的情况下提升自身准确率？辩论、自举以及内部一致性最大化等方法实现了这一惊人的成就，甚至可以媲美使用黄金标签的微调性能。然而，这些方法为何有效在理论上仍不清楚。我们证明，它们都可以被理解为“一致性优化”——即寻找最可压缩且可联合预测的“上下文到行为”映射，其中辩论是该优化的一个精确实例，而自举与内部一致性最大化则与之密切相关。我们证明了一致性优化等价于描述长度正则化，并且在所有此类正则化方案中，采用由预训练模型导出的先验的一致性正则化，能够优化半监督学习中最坏情况准确率的一个下界。我们的理论得到了初步实验的支持，解释了无需反馈的自我提升为何有效，并预测了它在何时应当……（原文摘要此处截断）

    arXiv:2601.13566v2 Announce Type: replace-cross  Abstract: Can language models improve their accuracy without external supervision? Methods such as debate, bootstrap, and internal coherence maximization achieve this surprising feat, even matching golden finetuning performance. Yet why they work remains theoretically unclear. We show that they can all be understood as coherence optimization, the search for a context-to-behavior mapping that is most compressible and jointly predictable, with debate an exact instance and bootstrap and internal coherence maximization closely related to it. We prove that coherence optimization is equivalent to description-length regularization, and that among all such regularization schemes, coherence regularization with a prior derived from a pretrained model optimizes a lower bound of worst-case accuracy for semi-supervised learning. Our theory, supported by preliminary experiments, explains why feedback-free self-improvement works and predicts when it sh
    
[^282]: 概念瓶颈模型Rashomon切片的参数高效构建方法

    Parameter-Efficient Construction of the Rashomon Slice for Concept Bottleneck Models

    [https://arxiv.org/abs/2511.19636](https://arxiv.org/abs/2511.19636)

    该论文提出了一种参数高效的方法，通过并行适配模块、检查点机制和概念多样性目标，高效探索概念瓶颈模型（CBM）的Rashomon集合，从而以较低成本生成多个精度相当但内部逻辑不同的模型。

    

    在许多机器学习问题中，可能存在多个模型，它们能取得几乎相同的预测性能，但其内部逻辑却存在根本差异。然而，标准训练流程只会产生单一模型，没有为探索可能更适合下游需求的替代方案提供实用的途径。这些精度相当的模型的集合被称为Rashomon集合（罗生门集合）。在庞大而复杂的假设空间中探索Rashomon集合尤其具有挑战性，例如概念瓶颈模型（CBMs），该模型被广泛应用于计算机视觉领域，通过中间的、人类可理解的概念来进行预测。在本文中，我们提出了一种高效探索CBM之Rashomon集合的方法。我们的框架引入了一个专门的并行参数高效适配模块，并结合检查点机制和概念多样性目标，以生成多个精度相当但内部逻辑各异的CBM。

    arXiv:2511.19636v3 Announce Type: replace-cross  Abstract: In many machine learning problems, there may exist multiple models that achieve nearly identical predictive performance while relying on fundamentally different internal logic. However, standard training procedures produce a single model, offering no practical way to explore alternatives that may better suit downstream needs. The set of these equally accurate models is known as the Rashomon set. Exploring the Rashomon set is particularly challenging in large and complex hypothesis spaces, such as Concept Bottleneck Models (CBMs), which are widely used in computer vision to make predictions through intermediate, human-understandable concepts. In this paper, we provide a method for efficiently exploring the Rashomon set of CBMs. Our framework introduces a specialized parallel parameter-efficient adaptation module, combined with a checkpointing scheme and a concept diversity objective, to generate multiple equally accurate CBMs fr
    
[^283]: 先微调，再校正

    Fine-Tune, Then Rectify

    [https://arxiv.org/abs/2511.19486](https://arxiv.org/abs/2511.19486)

    该论文提出一个结合微调与校正的两阶段LLM框架，指出传统微调目标（最小化均方误差）与下游校正阶段不匹配，并创新性地提出以最小化预测误差方差（或标量化方差指标）作为微调目标，同时在两阶段间最优分配有限的标注样本。

    

    受人工智能近期进展的推动，越来越多的文献展示了将大型语言模型（LLMs）作为可扩展代理来生成类人回应的潜力。提升LLM性能的两种常见方法包括：微调，使LLM的输出更贴近人类回应；以及校正，纠正LLM输出中的偏差。本文开发了一个结合微调与校正的两阶段框架，并在两个阶段之间最优地分配有限的标注样本。一个关键洞察是：传统的以最小化均方预测误差为目标的微调目标通常与下游的校正阶段并不一致。对于均值估计问题，我们提出以最小化预测误差的方差作为微调目标；对于一般的M估计问题，我们提出以最小化一个标量化的方差指标作为微调目标。基于这一洞察……

    arXiv:2511.19486v3 Announce Type: replace-cross  Abstract: Driven by recent advances in artificial intelligence, a growing literature has demonstrated the potential of using large language models (LLMs) as scalable surrogates to generate human-like responses. Two common approaches to improve the performance of LLMs include: fine-tuning, which aligns the LLM more closely with human responses, and rectification, which corrects biases in LLM outputs. In this paper, we develop a two-stage framework that combines fine-tuning and rectification, and optimally allocates limited labeled samples across the two stages. A key insight is that the conventional fine-tuning objective of minimizing mean squared prediction error is generally not aligned with the downstream rectification stage. For mean estimation, we propose to minimize the variance of the prediction errors; for general M-estimation, we propose to minimize a scalarized variance metric as the fine-tuning objective. Building on this insig
    
[^284]: 基于参数重要性的基础模型持续学习

    Parameter Importance-Driven Continual Learning for Foundation Models

    [https://arxiv.org/abs/2511.15375](https://arxiv.org/abs/2511.15375)

    提出了一种基于参数重要性估计的持续增强方法PIECE，使基础模型无需访问历史训练数据即可在高效学习领域知识的同时保持通用推理能力。

    

    领域特定的后训练常常导致灾难性遗忘，使基础模型失去其通用推理能力，并限制了其对动态真实环境的适应性。在获取下游领域知识的同时保持通用能力，是大语言模型和多模态模型面临的核心挑战。传统的持续学习方法，如正则化、回放和架构隔离，存在下游性能差、依赖无法获取的历史数据或额外参数开销等问题。尽管近期的参数高效微调（PET）方法可以缓解遗忘，但其有效性在很大程度上取决于参数选择和更新策略。在本文中，我们提出了PIECE，一种基于参数重要性估计的持续增强方法，它能在不访问先前训练数据的情况下，在保持通用能力的同时高效学习领域知识。

    arXiv:2511.15375v2 Announce Type: replace-cross  Abstract: Domain-specific post-training often causes catastrophic forgetting, making foundation models lose their general reasoning ability and limiting their adaptability to dynamic real-world environments. Preserving general capabilities while acquiring downstream domain knowledge is a central challenge for large language and multimodal models. Traditional continual learning methods, such as regularization, replay and architectural isolation, suffer from poor downstream performance, reliance on inaccessible historical data, or additional parameter overhead. While recent parameter-efficient tuning (PET) methods can alleviate forgetting, their effectiveness strongly depends on the choice of parameters and update strategies. In this paper, we introduce PIECE, a Parameter Importance Estimation-based Continual Enhancement method that preserves general ability while efficiently learning domain knowledge without accessing prior training data 
    
[^285]: 基于大语言模型的板位（Slate）推荐系统离线A/B测试：降低对预先收集的用户交互数据的依赖

    Offline A/B Testing of Slate Recommendation Systems with LLMs: Reducing the Dependency on Pre-Collected User Interaction Data

    [https://arxiv.org/abs/2511.04541](https://arxiv.org/abs/2511.04541)

    该论文提出利用大语言模型生成板位间的合成成对偏好进行离线A/B测试，结合广义Rao-Kupper模型可在不同效用权重下恢复稳定排名，作为离线策略评估与在线实验之间的低成本筛选环节，从而减少对预先收集的用户交互数据的依赖。

    

    板位推荐系统（Slate RecSys）向用户呈现一组相互关联的有序项目集合（例如播放列表）。我们研究大语言模型（LLM）能否表达板位之间的成对偏好，以实现对板位推荐系统的合成A/B测试。我们提出了一个验证协议，用于衡量合成偏好与经典推荐系统指标的一致性及其对偏好公理的遵守程度，并利用该协议刻画LLM的预训练与配置如何影响板位偏好的表达。结合广义Rao-Kupper模型，基于LLM的合成A/B测试能够恢复在不同效用权重设置下保持稳定的排名；相比之下，离线策略估计器只有在目标效用与已记录行为相匹配时才可靠。我们将其定位为介于离线策略评估与在线实验之间的筛选阶段：它并非A/B测试的替代品，而是一种将A/B测试成本留给最有前景的候选方案的手段。

    arXiv:2511.04541v2 Announce Type: replace-cross  Abstract: Slate recommender systems (RecSys) present users with ordered sets of interacting items (e.g., playlists). We investigate whether large language models (LLMs) can articulate pairwise preferences between slates for synthetic A/B testing of slate RecSys. We introduce a validation protocol measuring the alignment of synthetic preferences with classical RecSys metrics and their compliance with preference axioms, and use it to characterise how LLM pre-training and configuration affect slate preference articulation. Combined with the generalized Rao-Kupper model, synthetic LLM-based A/B testing recovers rankings that remain stable across utility weightings, whereas off-policy estimators are reliable only when the target utility matches the logged behavior. We position it as a screening stage between off-policy evaluation and live experiments: not a replacement for A/B testing, but a way to reserve its cost for the most promising cand
    
[^286]: UniShield：用于统一伪造图像检测与定位的自适应多智能体框架

    UniShield: An Adaptive Multi-Agent Framework for Unified Forgery Image Detection and Localization

    [https://arxiv.org/abs/2510.03161](https://arxiv.org/abs/2510.03161)

    UniShield提出了一种自适应多智能体框架，创新性地结合感知智能体与检测智能体，实现对图像篡改、文档篡改、DeepFake和AI生成图像等多种领域的统一伪造检测与定位。

    

    随着图像生成技术的快速发展，合成图像变得越来越逼真，带来了重大的社会风险，例如虚假信息和欺诈。因此，伪造图像检测与定位（FIDL）成为维护信息完整性和社会安全的关键。尽管现有的特定领域检测方法表现优异，但其实际应用仍然受限，主要原因是它们的专业领域狭窄、跨领域泛化能力差，以及缺乏一个集成的自适应框架。为了解决这些问题，我们提出了UniShield，一个新颖的基于多智能体的统一系统，能够检测和定位跨多个领域的图像伪造，包括图像篡改、文档篡改、DeepFake和AI生成图像。UniShield创新性地将感知智能体与检测智能体相集成。感知智能体智能地分析输入图像的特征（摘要在此处截断）。

    arXiv:2510.03161v3 Announce Type: replace-cross  Abstract: With the rapid advancements in image generation, synthetic images have become increasingly realistic, posing significant societal risks, such as misinformation and fraud. Forgery Image Detection and Localization (FIDL) thus emerges as essential for maintaining information integrity and societal security. Despite impressive performances by existing domain-specific detection methods, their practical applicability remains limited, primarily due to their narrow specialization, poor cross-domain generalization, and the absence of an integrated adaptive framework. To address these issues, we propose UniShield, the novel multi-agent-based unified system capable of detecting and localizing image forgeries across diverse domains, including image manipulation, document manipulation, DeepFake, and AI-generated images. UniShield innovatively integrates a perception agent with a detection agent. The perception agent intelligently analyzes i
    
[^287]: WAInjectBench：针对Web代理的提示注入检测基准测试

    WAInjectBench: Benchmarking Prompt Injection Detections for Web Agents

    [https://arxiv.org/abs/2510.01354](https://arxiv.org/abs/2510.01354)

    该论文提出了首个针对Web代理提示注入攻击检测的综合基准WAInjectBench，通过基于威胁模型的细粒度攻击分类，构建包含恶意与良性文本及图像的数据集，系统评估了现有文本和图像检测方法在多种场景下的性能。

    

    目前已有多种针对Web代理的提示注入攻击被提出。与此同时，研究者们开发了多种检测一般提示注入攻击的方法，但尚无方法在Web代理场景下得到系统性评估。在本工作中，我们通过提出首个针对Web代理提示注入攻击检测的综合基准研究来填补这一空白。我们首先基于威胁模型对这类攻击进行了细粒度的分类。随后，我们构建了同时包含恶意样本和良性样本的数据集：包括由不同攻击生成的恶意文本片段、来自四个类别的良性文本片段、由攻击产生的恶意图像，以及来自两个类别的良性图像。接着，我们对基于文本和基于图像的检测方法进行了系统化整理。最后，我们在多种场景下评估了这些方法的性能。我们的关键发现表明，虽然某些检测器能够识别依赖……的攻击（原文摘要在此处截断）。

    arXiv:2510.01354v2 Announce Type: replace-cross  Abstract: Multiple prompt injection attacks have been proposed against web agents. At the same time, various methods have been developed to detect general prompt injection attacks, but none have been systematically evaluated for web agents. In this work, we bridge this gap by presenting the first comprehensive benchmark study on detecting prompt injection attacks targeting web agents. We begin by introducing a fine-grained categorization of such attacks based on the threat model. We then construct datasets containing both malicious and benign samples: malicious text segments generated by different attacks, benign text segments from four categories, malicious images produced by attacks, and benign images from two categories. Next, we systematize both text-based and image-based detection methods. Finally, we evaluate their performance across multiple scenarios. Our key findings show that while some detectors can identify attacks that rely 
    
[^288]: 离散最优传输是一种强大的音频对抗攻击

    Discrete optimal transport is a strong audio adversarial attack

    [https://arxiv.org/abs/2509.14959](https://arxiv.org/abs/2509.14959)

    该论文提出了一种基于离散最优传输的黑盒后处理音频攻击方法，通过将语音嵌入分布对齐到真实语音池，在无需模型参数、梯度或训练数据的情况下显著削弱自动说话人验证与反欺骗系统的性能，且具备跨数据集迁移能力并在对抗措施微调后仍然有效。

    

    本文研究了离散最优传输（DOT）作为针对现代自动说话人验证（ASV）系统和反欺骗对抗措施（CM）系统的黑盒攻击方法。该攻击作为一种后处理的分布对齐步骤运行：将生成语音（或他人语音）的帧级WavLM嵌入，通过熵正则化最优传输和top-k重心投影，对齐到一个未配对的真实（bona fide）语音池，随后进行神经声码器合成。与基于梯度的攻击不同，所提出的方法无需访问模型参数、梯度或训练数据。在ASVspoof2019和ASVspoof5数据集上的实验表明，DOT攻击在多种欺骗攻击方式下显著提高了CM的等错误率（EER），并大幅降低了ASV的性能。该攻击可以跨数据集迁移，并且在CM经过微调后依然有效。基于说话人相似度、Fréchet音频距离以及可视化的分析显示……（原文摘要至此中断）

    arXiv:2509.14959v4 Announce Type: replace-cross  Abstract: In this paper, we investigate discrete optimal transport (DOT) as a black-box attack against modern automatic speaker verification (ASV) and anti-spoofing countermeasure (CM) systems.   Our attack operates as a post-processing distribution-alignment step. Frame-level WavLM embeddings of generated speech (or another person speech) are aligned to an unpaired bona fide speech pool using entropic optimal transport and a top-k barycentric projection, followed by neural vocoding. Unlike gradient-based attacks, the proposed method requires no access to model parameters, gradients, or training data.   Experiments on ASVspoof2019 and ASVspoof5 demonstrate that DOT attack substantially increases CM EER and substantially degrades ASV performance across multiple spoofing attacks. The attack transfers across datasets and remains effective after CM fine-tuning. Analysis using speaker similarity, Fr\'echet Audio Distance, and visualization of
    
[^289]: AdaDim：面向自监督学习表征动力学的维度自适应方法

    AdaDim: Dimensionality Adaptation for SSL Representational Dynamics

    [https://arxiv.org/abs/2505.12576](https://arxiv.org/abs/2505.12576)

    该论文提出 AdaDim 方法，在自监督学习训练过程中自适应地调控表示的维度动态，兼顾高有效维度 H(R) 与低互信息 I(R;Z)，以防止维度坍缩并提升下游任务的泛化性能。

    

    自监督学习（SSL）有效性的一个关键因素是防止维度坍缩，即高维表示空间（R）实际张成的却是较低维的子空间。因此，SSL 的优化策略之一是通过鼓励特征去相关或样本在 R 中的均匀分布等目标函数，引导模型产生具有更高维度的 R（记为 H(R)）。更高的 H(R) 表明 R 具有更大的特征多样性，这有利于向下游任务的泛化。除了维度优化之外，SSL 算法还利用投影头将 R 映射到嵌入空间 Z。近期的研究将投影头刻画为一个滤波器，通过降低互信息 I(R;Z) 来滤除 SSL 目标中的噪声或无关特征。因此，当前文献的观点是：一个好的 SSL 表示空间应当同时具有高 H(R) 和低 I(R;Z)。然而，这一观点……

    arXiv:2505.12576v3 Announce Type: replace-cross  Abstract: A key factor in effective Self-Supervised learning (SSL) is preventing dimensional collapse, where higher-dimensional representation spaces ($R$) span a lower-dimensional subspace. Therefore, SSL optimization strategies involve guiding a model to produce $R$ with a higher dimensionality ($H(R)$) through objectives that encourage decorrelation of features or sample uniformity in $R$. A higher $H(R)$ indicates that $R$ has greater feature diversity which is useful for generalization to downstream tasks. Alongside dimensionality optimization, SSL algorithms also utilize a projection head that maps $R$ into an embedding space $Z$. Recent work has characterized the projection head as a filter of noisy or irrelevant features from the SSL objective by reducing the mutual information $I(R;Z)$. Therefore, the current literature's view is that a good SSL representation space should have a high $H(R)$ and a low $I(R;Z)$. However, this vie
    
[^290]: SMDDFNet：面向交通标志检测的状态空间建模与动态双融合网络

    SMDDFNet: State-space Modeling and Dynamic Dual Fusion Network for Traffic Sign Detection

    [https://arxiv.org/abs/2505.05491](https://arxiv.org/abs/2505.05491)

    SMDDFNet通过融合状态空间建模主干网络与动态双融合模块（结合多尺度注意力与频域内容感知动态滤波），以线性计算复杂度捕获长程依赖并增强多尺度特征表示，在多个基准数据集上实现了具有竞争力的交通标志检测精度。

    

    交通标志检测是高级驾驶辅助系统中一项具有挑战性的视觉信号处理任务，其中小目标、尺度变化和遮挡问题限制了采用固定感受野的传统检测器的性能。本文提出了一种用于交通标志图像检测的深度学习检测器——状态空间建模与动态双融合网络（SMDDFNet）。SMDDFNet集成了动态双融合（DDF）模块和状态空间建模主干网络，以增强多尺度特征表示能力。DDF模块将高效的多尺度注意力与频域中的内容感知动态滤波相结合，而主干网络则以线性计算复杂度捕获长程依赖关系。多尺度特征融合颈部进一步聚合金字塔特征，以实现对小尺寸交通标志的鲁棒定位。在TT100K、GTSDB、PASCAL VOC以及Roboflow 100车辆子集上的实验表明，SMDDFNet相较于近期检测器取得了具有竞争力的精度。

    arXiv:2505.05491v2 Announce Type: replace-cross  Abstract: Traffic sign detection is a challenging visual signal processing task for advanced driver assistance, where small objects, scale variation, and occlusion limit conventional detectors with fixed receptive fields. This paper proposes State-space Modeling and Dynamic Dual Fusion Network (SMDDFNet), a deep learning detector for traffic sign images. SMDDFNet integrates a Dynamic Dual Fusion (DDF) module and a state-space modeling backbone to enhance multi-scale feature representation. DDF combines efficient multi-scale attention with content-aware dynamic filtering in the frequency domain, while the backbone captures long-range dependencies with linear computational complexity. A multi-scale feature fusion neck further aggregates pyramid features for robust localization of small signs. Experiments on TT100K, GTSDB, PASCAL VOC, and the Roboflow~100 \emph{vehicle} subset show that SMDDFNet achieves competitive accuracy against recent 
    
[^291]: 通过质量感知偏好学习提升大语言模型生成代码的非功能质量合规性

    Enhancing the Non-Functional Quality Compliance of LLM-Generated Code through Quality-Aware Preference Learning

    [https://arxiv.org/abs/2503.09020](https://arxiv.org/abs/2503.09020)

    本文提出一种质量感知偏好学习框架，通过构建违规-合规代码对、自适应令牌加权和混合优化目标，引导大语言模型生成符合非功能质量标准的代码。

    

    大语言模型（LLMs）已广泛应用于商业代码补全引擎中，显著提高了编码效率和生产力。然而，即使功能正确的大语言模型生成代码也可能表现出非功能质量问题，这些质量问题违反编码标准和最佳实践，例如较差的代码风格和有限的可维护性。为解决这一问题，我们提出了一种质量感知偏好学习框架，引导大语言模型生成符合标准的代码。我们的方法包括三个阶段。首先，我们构建了一个成对的数据集，包含违反标准的样本和符合标准的样本，其中每对包含具有特定非功能质量问题的代码及其修复版本。其次，我们设计了一种自适应令牌加权机制，以强调质量敏感的代码区域。第三，我们引入了一种混合优化目标，将排序损失与语言模型损失相结合。

    arXiv:2503.09020v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have been widely adopted in commercial code completion engines, significantly enhancing coding efficiency and productivity. However, even functionally correct LLM-generated code may exhibit non-functional quality issues that violate coding standards and best practices, such as poor style and limited maintainability. To address this, we propose a framework for quality-aware preference learning that guides LLMs toward generating criteria-compliant code. Our approach consists of three phases. First, we construct a dataset of paired criteria-violating and criteria-compliant samples, where each pair contains code exhibiting a specific non-functional quality issue and its repaired version that resolves the issue. Second, we design an adaptive token weighting mechanism to emphasize quality-sensitive code regions. Third, we introduce a hybrid optimization objective that combines ranking loss with language m
    
[^292]: 路径正则化：多层神经网络的近乎完备且最优的非渐近泛化理论与双重下降现象

    Path Regularization: A Near-Complete and Optimal Nonasymptotic Generalization Theory for Multilayer Neural Networks and Double Descent Phenomenon

    [https://arxiv.org/abs/2503.02129](https://arxiv.org/abs/2503.02129)

    该论文首次提出了路径正则化多层神经网络的近乎完备且最优的非渐近泛化理论，给出了显式泛化误差上界，无需损失函数有界及网络宽度、深度等常见假设，超越了偏差-方差权衡并能解释深度学习中的双重下降现象。

    

    路径正则化已被证明是训练神经网络的一种非常有效的正则化方法，与权重衰减等常见正则化方法相比，它能带来更好的泛化性能。我们针对一般学习问题，首次提出了带有路径正则化的多层神经网络的近乎完备的（将在正文中明确说明）非渐近泛化理论。特别地，该理论不要求损失函数有界，而这在现有文献中通常是必备假设。我们的理论超越了偏差-方差权衡，并与深度学习中常见的现象相吻合，因此与其他现有的非渐近泛化误差界有显著不同。更具体地说，我们为满足 $\sigma(0)=0$ 且损失函数为足够广泛的Lipschitz函数的多层神经网络提出了显式的泛化误差上界，而无需对网络的宽度、深度或其他超参数作出限制……

    arXiv:2503.02129v3 Announce Type: replace-cross  Abstract: Path regularization has shown to be a very effective regularization to train neural networks, leading to a better generalization property than common regularizations i.e. weight decay, etc. We propose a first near-complete (as will be made explicit in the main text) nonasymptotic generalization theory for multilayer neural networks with path regularizations for general learning problems. In particular, it does not require the boundedness of the loss function, as is commonly assumed in the literature. Our theory goes beyond the bias-variance tradeoff and aligns with phenomena typically encountered in deep learning. It is therefore sharply different from other existing nonasymptotic generalization error bounds. More explicitly, we propose an explicit generalization error upper bound for multilayer neural networks with $\sigma(0)=0$ and sufficiently broad Lipschitz loss functions, without requiring the width, depth, or other hyper
    
[^293]: 影像组学与人工智能在甲状腺癌诊断中的应用：概念、挑战与解决方案

    Radiomics and artificial Intelligence for thyroid cancer diagnosis: Concepts, challenges, and solutions

    [https://arxiv.org/abs/2404.07239](https://arxiv.org/abs/2404.07239)

    本综述系统梳理了基于超声图像的影像组学与人工智能在甲状腺癌诊断中的应用，证实其诊断有效性，并探讨了该领域面临的概念、挑战与解决方案。

    

    甲状腺癌是日益严重的全球健康问题，需要先进的诊断方法。本综述考察了人工智能和影像组学在甲状腺癌诊断中的应用。研究人员遵循PRISMA指南，对多个数据库进行了检索，检索时间截至2024年10月。通过关键词组合检索，找到了关于甲状腺癌及相关主题的英文学术出版物。初次检索共返回368篇论文，去除112篇重复文献后，根据预定标准对题目和摘要进行筛选，剔除了176篇文章，最终选定相关研究。经过综合分析后，又排除了六项研究。在纳入的42项研究中，结合超声（US）图像的影像组学分析证明了其在甲状腺癌诊断中的有效性。各项研究呈现了不同的结果，其中部分研究……

    arXiv:2404.07239v2 Announce Type: replace-cross  Abstract: Thyroid cancer is an increasing global health concern that requires advanced diagnostic methods. The application of AI and radiomics to thyroid cancer diagnosis is examined in this review. A review of multiple databases was conducted in compliance with PRISMA guidelines until October 2024. A combination of keywords led to the discovery of an English academic publication on thyroid cancer and related subjects. 368 papers were returned from the original search after 112 duplicates were removed. Relevant studies were selected according to predetermined criteria after 176 articles were eliminated based on an examination of their abstract and title. After the comprehensive analysis, an additional six studies were excluded. Among the 42 included studies, radiomics analysis, which incorporates ultrasound (US) images, demonstrated its effectiveness in diagnosing thyroid cancer. Various results were noted, some of the studies presenting
    
[^294]: 大语言模型水印的优化

    Optimizing watermarks for large language models

    [https://arxiv.org/abs/2312.17295](https://arxiv.org/abs/2312.17295)

    本文将大语言模型水印中可识别性与生成文本质量影响之间的权衡形式化为多目标优化问题，识别出一大类鲁棒高效水印的帕累托最优解，并证明其性能优于当前默认水印方案。

    

    随着大语言模型（LLM）的兴起以及对其潜在滥用的担忧，生成式大语言模型的水印技术近来受到了广泛关注。此类水印的一个重要方面是其可识别性与对生成文本质量影响之间的权衡。本文通过多目标优化问题的框架，为这一权衡引入了一种系统化的方法。对于一大类鲁棒且高效的水印，本文识别出了相应的帕累托最优解，并证明其性能优于当前默认的水印方案。

    arXiv:2312.17295v2 Announce Type: replace-cross  Abstract: With the rise of large language models (LLMs) and concerns about potential misuse, watermarks for generative LLMs have recently attracted much attention. An important aspect of such watermarks is the trade-off between their identifiability and their impact on the quality of the generated text. This paper introduces a systematic approach to this trade-off in terms of a multi-objective optimization problem. For a large class of robust, efficient watermarks, the associated Pareto optimal solutions are identified and shown to outperform the currently default watermark.
    
[^295]: 异步感知-动作-通信与图神经网络

    Asynchronous Perception-Action-Communication with Graph Neural Networks. (arXiv:2309.10164v1 [cs.RO])

    [http://arxiv.org/abs/2309.10164](http://arxiv.org/abs/2309.10164)

    该论文提出了使用图神经网络实现异步感知-动作-通信的方法，解决了在大型机器人群体中协作和通信的挑战。现有的框架假设顺序执行，该方法是完全分散的，但在评估和部署方面仍存在一些限制。

    

    在大型机器人群体中实现共同的全局目标的协作是一个具有挑战性的问题，因为机器人的感知和通信能力有限。机器人必须执行感知-动作-通信（PAC）循环-它们感知局部环境，与其他机器人通信，并实时采取行动。分散的PAC系统面临的一个基本挑战是决定与相邻机器人通信的信息以及如何在利用邻居共享的信息的同时采取行动。最近，使用图神经网络（GNNs）来解决这个问题已经取得了一些进展，比如在群集和覆盖控制等应用中。虽然在概念上，GNN策略是完全分散的，但评估和部署这样的策略主要仍然是集中式的或具有限制性的分散式。此外，现有的框架假设感知和动作推理的顺序执行，这在现实世界的应用中非常限制性。

    Collaboration in large robot swarms to achieve a common global objective is a challenging problem in large environments due to limited sensing and communication capabilities. The robots must execute a Perception-Action-Communication (PAC) loop -- they perceive their local environment, communicate with other robots, and take actions in real time. A fundamental challenge in decentralized PAC systems is to decide what information to communicate with the neighboring robots and how to take actions while utilizing the information shared by the neighbors. Recently, this has been addressed using Graph Neural Networks (GNNs) for applications such as flocking and coverage control. Although conceptually, GNN policies are fully decentralized, the evaluation and deployment of such policies have primarily remained centralized or restrictively decentralized. Furthermore, existing frameworks assume sequential execution of perception and action inference, which is very restrictive in real-world applica
    

