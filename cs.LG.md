# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ENCP: Episode-Normalized Conformal Prediction for Vision-and-Language Navigation](https://arxiv.org/abs/2609.17499) | 提出情节归一化保形预测（ENCP），通过以情节为单位对非一致性得分进行归一化校准，为视觉与语言导航的每一步提供至少1-α的覆盖保证，同时允许情节内步骤间存在依赖关系。 |
| [^2] | [FreqSpaNet: Frequency and Spatial Learning of SFPF for Physical Layer Hardware Integrity Detection](https://arxiv.org/abs/2609.17491) | 提出FreqSpaNet网络，通过频率分支与几何感知空间分支分别学习空-频极化指纹（SFPF）的不同结构特性，结合自适应融合与互补预训练，实现开集硬件异常检测，以识别保持逻辑身份不变的未经授权硬件替换。 |
| [^3] | [Bridging the Gap Between Homogeneous and Heterogeneous Asynchronous Optimization Is Surprisingly Difficult](https://arxiv.org/abs/2609.17483) | 本文证明在广泛使用的一阶和二阶相似性假设下，任何随机算法都无法改进异构异步优化的悲观最优时间复杂度，表明弥合同构与异构设置之间的差距出人意料地困难。 |
| [^4] | [Bias-Induced Crossover in Absolute Capacity of Dense Associative Memory](https://arxiv.org/abs/2609.17477) | 本文首次系统分析了中心化二元模式中偏置对稠密联想记忆绝对容量的影响，发现容量随相互作用阶数 $n$ 的奇偶性呈现不同的渐近标度律，并通过渐近匹配方法预测了在偏置趋于零时由偏置诱导的容量渡越现象。 |
| [^5] | [Coupled Calibration and Learning: Mitigating Teacher Bias in LLM Distillation without Target-Domain Reward Feedback](https://arxiv.org/abs/2609.17474) | 提出耦合校准与学习（CCL）算法，通过token级分支将教师校准与学生更新相互耦合、仅利用源域奖励反馈，从而在无目标域奖励反馈的条件下缓解LLM蒸馏中的教师偏差迁移问题。 |
| [^6] | [Tables Decoded: DELTA for Structure, TARQA for Understanding](https://arxiv.org/abs/2609.17458) | 本文提出基于结构化文本表示的 DELTA 与 TARQA 方法来处理表格结构识别和表格视觉问答任务，其中 DELTA 将物理结构识别、逻辑结构识别与 OCR 分离，并以紧凑统一的 OTSL 格式输出表格，相比视觉-语言模型更具可扩展性且适用于多语言文档。 |
| [^7] | [Reduced-Space Multi-Fidelity Bayesian Optimization of Process Simulation Models](https://arxiv.org/abs/2609.17440) | 提出了一种将全局敏感性分析降维与保真度增强高斯过程相结合的降空间多保真度贝叶斯优化框架，能够在高维昂贵的工业过程仿真中实现成本感知的高效优化。 |
| [^8] | [Learning-Guided Planning in Large Dynamic Action Spaces: Budgeted Tree Search for One-to-Many Mobile Charging](https://arxiv.org/abs/2609.17429) | 提出了LP-BTS学习引导规划架构，通过图提案策略、学习价值评论家和边预算化PUCT树搜索相结合，解决了候选动作随状态动态重建的一对多移动充电问题，并凭借无固定输出头的设计用单个冻结模型覆盖从736到2,813个候选动作的多种大规模动态动作空间。 |
| [^9] | [Knowledge as Orbit: Finite Collections as Phases of an Exactly Periodic Latent Generator](https://arxiv.org/abs/2609.17417) | 提出以内涵方式存储有限知识集合：将每个条目表示为基于实离散傅里叶旋转算子的精确周期性潜空间生成器的相位，从数学上保证精确闭合，并实验证明精确周期算子在循环重建上显著优于一般学习算子与非周期保范算子。 |
| [^10] | [Bridging the Confidence Gap: Temperature Scaling for Calibrating Test-Time Prompt Tuning](https://arxiv.org/abs/2609.17386) | 提出CoTS后校准方法，通过温度缩放最小化测试时提示调优中适应预测与零样本预测之间的置信度差距，在提升校准性能的同时保持准确率，并结合弱-强集成策略（E-CoTS）进一步增强效果。 |
| [^11] | [OPEN-1B: A Fully Auditable Training Run](https://arxiv.org/abs/2609.17380) | 本文提出“完全可审计”这一全新的模型透明度级别，使训练中对每个数据样本的每次操作都能在异构通用硬件上以位级确定性独立复现，从而解决开源语言模型无法被证明可复现的问题。 |
| [^12] | [Large Language Models Develop Belief State Geometry In-Context](https://arxiv.org/abs/2609.17376) | 研究发现大语言模型在上下文学习隐马尔可夫模型数据时，会在其内部残差流激活中以线性可解码的方式编码出信念状态的几何表示，且通过干预实验证明该表示与模型的预测行为功能相关。 |
| [^13] | [Hybrid Variational Quantum Circuits for Multivariate Regression and High-Dimensional Data Reconstruction](https://arxiv.org/abs/2609.17358) | 该论文提出混合变分量子线路（HVQC），通过在量子线路后添加经典仿射层，实现了无需独立标量线路开销的向量值回归与高维数据重构，其性能媲美高斯过程回归并超越XGBoost和随机森林。 |
| [^14] | [Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning](https://arxiv.org/abs/2609.17338) | 本文提出LWVIC4Code，一种基于VICReg框架的非对比表示学习方法，通过跨层一致性正则化和深度相关的层加权机制逐层精炼代码语义表示，有效解决了语义等价但语法不同的Type-IV代码克隆检测难题，同时避免了对比学习中负采样带来的偏差。 |
| [^15] | [Quantum-Inspired Trainable and Parameter-Efficient Tensor Networks for Image Inpainting](https://arxiv.org/abs/2609.17298) | 本文提出量子启发的可训练张量网络变换用于图像修复，其中对角QFT松弛通过电路结构天然保持最小相干性，以极少的参数实现了超越固定变换并媲美大型酉架构的性能。 |
| [^16] | [Goal-oriented probabilistic forecasting for dynamic PRB allocation in 5G networks](https://arxiv.org/abs/2609.17297) | 该论文提出一种面向目标的概率预测框架，利用Pinball损失函数训练DeepAR和TFT模型，并根据运营商成本矩阵确定最优分配分位数，从而在5G网络动态PRB分配中降低运营成本，实现服务可靠性与资源效率的平衡。 |
| [^17] | [Conformal Policy Learning with Distribution-Free Safety Guarantees](https://arxiv.org/abs/2609.17296) | 本文提出共形策略学习（CPL），通过将每个处理决策视为反事实伤害假设检验并利用共形p值阈值化分配处理，首次实现了控制“对会受伤害个体分配处理”概率的无分布安全保证。 |
| [^18] | [Same Flow, Different Paths: Variance Reduction in Flow Matching](https://arxiv.org/abs/2609.17287) | 本文从优化角度研究流匹配中的路径选择，证明即使目标函数完全相同，不同路径仍会通过随机梯度方差从根本上改变SGD的收敛速度，并推导出线性路径下的解析最优路径。 |
| [^19] | [Personalized Federated Learning through Global Knowledge Distillation and Local Head Adaptation](https://arxiv.org/abs/2609.17284) | 提出pFedKDH方法，通过仅聚合共享骨干网络、保留客户端专属分类头并利用重新校准的全局分类头作为蒸馏教师，在标签偏斜的个性化联邦学习场景中显著提升准确率。 |
| [^20] | [Easy to Catch a Liar, Hard to Clear an Honest One: Language Models Diagnosing a Corrupted Reward Channel from a Verified Record](https://arxiv.org/abs/2609.17226) | 本研究构建了一个赔付交换与说谎报告者产生完全相同历史的博弈环境，证明冻结的大语言模型仅需一条独立验证记录即可几乎完美地识别出说谎的报告者。 |
| [^21] | [Memorisation bias in medical AI](https://arxiv.org/abs/2609.17223) | 该研究揭示了医疗AI中的“记忆偏差”现象：模型在训练中接触过患者的匿名历史数据后会显著改变对该患者未来数据的预测，且这种偏差可持续存在长达数十年，对临床部署具有深远影响。 |
| [^22] | [Cross-Domain Inference for Human Localization: Applying Wi-Fi RSSI Data to CSI-Trained Models](https://arxiv.org/abs/2609.17204) | 本文提出了一种跨域推理方法，将易于获取的Wi-Fi RSSI数据输入到原本基于CSI数据训练的姿态预测模型中，验证了在低权限物联网设备上实现人体位置预测的可行性。 |
| [^23] | [MyoFlow: Anchor-Tied Rectified Flow for HD-sEMG Gesture Recognition Across Sessions and Subjects](https://arxiv.org/abs/2609.17194) | 提出了首个面向跨会话与跨被试高密度表面肌电手势识别的判别式流匹配框架MyoFlow，通过域条件化校正流将分类重构为向手势锚点的输运过程，无需独立分类头即可实现零样本预测。 |
| [^24] | [LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers](https://arxiv.org/abs/2609.17184) | LoopSpec是一个专为循环Transformer设计的免训练自推测解码框架，通过从早期循环状态提取草稿token并以流水线方式重叠草稿生成与目标验证，从而在无需辅助模型的情况下显著提升解码效率。 |
| [^25] | [IRENE: A Convolutional GRU Ensemble Model for Radar Precipitation Nowcasting over Italy](https://arxiv.org/abs/2609.17175) | IRENE是一个基于多尺度ConvGRU的深度学习集合模型，结合重要性采样、afCRPS概率损失以及对抗训练和谱约束等创新配置，实现了意大利地区1公里空间和5分钟时间分辨率的概率性降水临近预报。 |
| [^26] | [A unified framework for global and local interpretability using adaptive derivative-ordered random explanation](https://arxiv.org/abs/2609.17171) | 本文提出ADORE方法，利用一阶和二阶导数在统一分析框架内同时实现全局特征重要性与局部样本贡献的可解释性分析，有效捕捉非线性特征-样本交互并精确量化特征影响。 |
| [^27] | [Neural Field Ensembles for Aerodynamic Surface Prediction: Winning Solution to the ONERA CRM Wall Distribution 2025 Challenge](https://arxiv.org/abs/2609.17160) | 本文提出一种基于条件神经场的集成学习方法，结合傅里叶特征编码和相对平方误差目标，实现对NASA CRM翼身挂架短舱构型在不同工况下压力和表面摩擦系数分布的精确预测，赢得ONERA CRM壁面分布2025挑战赛冠军。 |
| [^28] | [ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers](https://arxiv.org/abs/2609.17152) | 本文揭示了视觉Transformer中残差连接的消除效应是导致LRP归因不稳定的根本原因，并提出ResLRP方法，其传播规则显式考虑残差消除、严格守恒且可证明地限制归因爆炸。 |
| [^29] | [Kernel-Based Metrics Learning for Uncertain Opponent Vehicle Trajectory Prediction in Autonomous Racing](https://arxiv.org/abs/2609.17147) | 该论文提出用于深度核学习的异构核度量，以无监督方式从自车与对手车辆的交互中捕捉并区分不同驾驶策略，实现自动驾驶赛车中对手车辆轨迹的精确预测与不确定性估计，并在1/10比例赛车平台上验证了更高的预测精度、安全超车能力以及车载计算效率。 |
| [^30] | [Continual Learning for Traversability Prediction with Uncertainty-Aware Adaptation](https://arxiv.org/abs/2609.17141) | 该论文提出了一种基于生成式经验回忆模型的可通行性预测持续学习框架，无需存储历史数据即可保留先前经验，并通过融合生成样本的不确定性实现不确定性感知的自适应，从而解决灾难性遗忘问题。 |
| [^31] | [From Foundation Embeddings to Cropland Maps: Label Efficiency, Temporal Transferability and Independent Human Validation](https://arxiv.org/abs/2609.17138) | 该研究表明，无需微调基础模型，仅用轻量级分类器即可基于AlphaEarth嵌入实现93.7%精度的农田制图，且仅需6万个标注像素就能达到接近860万全量数据的性能。 |
| [^32] | [Intrinsic Robot Rewarding: Reusing VLA Representations for Autonomous Evaluation and Policy Improvement](https://arxiv.org/abs/2609.17115) | 该论文提出内在机器人奖励（IRR）方法，通过复用VLA系统中已有的视觉表征和成功示范，无需额外训练评估器即可自主评估机器人执行结果并生成奖励信号，从而降低集成成本、提升奖励计算效率并减少人工评分负担。 |
| [^33] | [High-Fidelity Digital Twin Data Models by Randomized Dynamic Mode Decomposition and Deep Learning with Applications in Fluid Dynamics](https://arxiv.org/abs/2609.17101) | 本文提出了一种结合随机动态模态分解与深度学习的新框架，用于从数值代码输出中构建高精度、低计算成本的数字孪生数据模型，并在流体力学问题中验证了其有效性。 |
| [^34] | [Optimization over covariance matrices with a parameterized metric](https://arxiv.org/abs/2609.17089) | 本文提出一个统一了欧几里得、Bures-Wasserstein和仿射不变度量的双参数黎曼度量族，并通过分析解处黎曼Hessian的条件数，证明其下界仅依赖于参数之和 $r=p+q$，从而为协方差矩阵优化中的度量选择（预条件化）提供了理论判据。 |
| [^35] | [Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility Shapes Activation Function Discovery](https://arxiv.org/abs/2609.17067) | 该研究将神经网络激活函数集合的演化发现建模为元学习问题，设计了11种受生物适应机制启发的策略（如昼夜节律振荡门控与免疫克隆选择）在演化中动态调整可用激活函数，并发现生物机制与演化过程之间的时间尺度兼容性是决定激活函数发现成败的关键。 |
| [^36] | [Neuro-Symbolic Hierarchical Intention Anticipation in Human Behavior](https://arxiv.org/abs/2609.17064) | 该论文提出一种神经符号分层规划解码器（HPD），能够在人类行为尚未完成时从部分观察的多模态片段中在四个本体层次上预测意图与剩余行为，并通过软神经符号正则化训练和硬可达性掩码推理保证预测的本体有效性。 |
| [^37] | [Repurposing Unified Topological Signatures for Graph Representation Learning](https://arxiv.org/abs/2609.17061) | 该论文提出将基于持久同调的两种统一拓扑签名（静态的Graph_UTS和动态的Embedding_UTS）集成到图表示学习中，突破传统消息传递GNN受限于1-WL图同构测试的判别能力瓶颈。 |
| [^38] | [Near-Optimal Nonconvex Matrix Completion](https://arxiv.org/abs/2609.17048) | 该论文通过分析采用多尺度残差初始化的黎曼梯度下降和黎曼高斯-牛顿方法，使非凸矩阵补全的样本复杂度达到与凸方法相当的近优水平，显著降低了对秩的依赖。 |
| [^39] | [Learning Options for Compositional Motor Control with Adapter Banks](https://arxiv.org/abs/2609.17042) | 该论文将神经科学中“运动基元是共享循环网络低秩扰动”的理论转化为端到端可学习的新架构——共享循环核心加上由离散潜码选择的残差适配器库，适配器在无秩约束下自发涌现低秩特性，并通过冻结网络后优化的高层策略组合选项实现新颖分布外运动的泛化。 |
| [^40] | [Distributed JEPA: A Self-Supervised Framework for Energy Forecasting](https://arxiv.org/abs/2609.17029) | 该论文提出了一种分布式联合嵌入预测架构（JEPA），通过对被掩码时间片段的潜在表示进行自监督预测，并结合协方差与时间方差正则化来防止表示坍塌，从而在异构能源时间序列上实现可迁移的能源预测。 |
| [^41] | [CLARE: Scalable Class-Incremental Continual Learning via a Sparsity-Based Framework](https://arxiv.org/abs/2609.17026) | CLARE提出了一种两阶段稀疏微调框架，先通过稀疏性目标识别任务关键的参数掩码，再仅对掩码选定的参数进行约束微调，从而解决持续学习中多任务顺序训练的可扩展性瓶颈和灾难性遗忘问题。 |
| [^42] | [Beyond Measurement Metrics: A Human-Centered Framework for Semantic Validation of Network Traffic Classification](https://arxiv.org/abs/2609.17014) | 本文提出了一种以人为中心的语义验证框架，通过融合数据、机器学习模型、可解释性技术、可视化和专家推理，来迭代验证网络流量分类模型是否真正学习到语义上有意义的模式，而不仅仅追求高预测性能指标。 |
| [^43] | [Structural Negative Transfer in Federated Graph Neural Networks: Diagnosis, Causal Investigation, and the Limits of Divergence-Aware Mitigation](https://arxiv.org/abs/2609.16977) | 本文诊断并因果性地研究了联邦图神经网络中因客户端图结构差异引发的结构负迁移现象——结构上非典型的客户端仅因加入联邦就损失过半可达成准确率，并揭示了差异感知缓解方法的能力极限。 |
| [^44] | [Splitting the Difference: Interpretable Causal Forests for Treatment Effect Heterogeneity and Bias](https://arxiv.org/abs/2609.16971) | 本文提出一种基于决策树和随机森林的可解释因果森林算法，仅通过修改分裂准则即可准确估计个体处理效应并揭示其异质性原因，无需双重机器学习或正交化等额外复杂步骤。 |
| [^45] | [HUMAID-NER: A Disaster Tweet Dataset for Joint Named Entity Recognition and Event Classification via Uncertainty-Weighted Multitask Learning](https://arxiv.org/abs/2609.16964) | 该论文发布了首个基于HumAID基准的命名实体识别数据集HUMAID-NER（含6万条标注推文、约17.5万个实体跨度），并提出基于不确定性加权多任务学习的框架，联合执行灾难命名实体识别与人道主义事件分类。 |
| [^46] | [Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation](https://arxiv.org/abs/2609.16937) | 提出γOPD方法，通过折扣时序信用分配和奖励兼容的有界混合机制，在大语言模型在线策略蒸馏中统一了token级与序列级目标，在获得长时程监督的同时保证与时间跨度无关的优化稳定性。 |
| [^47] | [MedPCFM-TED: One-Step Point Cloud Flow Matching for Implant Generation via Teacher-Guided Endpoint Distillation](https://arxiv.org/abs/2609.16934) | 提出教师引导端点蒸馏（TED）框架，实现颅骨植入体的单步点云流匹配生成，在SkullBreak数据集上取得最佳整体性能，并在单步方法中达到最优的倒角距离表现。 |
| [^48] | [When Confidence Signals Disagree: Local and Global Confidence in Autoregressive Language Models](https://arxiv.org/abs/2609.16933) | 该研究发现在自回归语言模型中，局部置信度（贪婪选择答案标记的概率）与全局置信度（重复采样下的众数答案频率）仅呈弱相关，且全局置信度与答案正确性有中等关联而局部置信度几乎没有关联，表明这两种置信度信号不可互换使用。 |
| [^49] | [Causal Discovery via Transformed Low-Rank Quantile Surfaces](https://arxiv.org/abs/2609.16931) | 本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。 |
| [^50] | [Repurposing Deep Limit Order Book Forecasting for Scenario-Conditioned Market Impact Modeling](https://arxiv.org/abs/2609.16930) | 该论文提出一个模型无关框架，证明预训练的深度限价订单簿预测模型无需重新训练即可被重新用于场景条件下的市场冲击建模，与历史实际结果达到0.99的Spearman相关性和97.2%的方向一致性。 |
| [^51] | [Verbalizing Subliminal Learning Effects Using Text Optimization](https://arxiv.org/abs/2609.16927) | 本文提出SALVE方法，利用文本优化技术将蒸馏数据中隐藏传递的教师模型潜意识学习效应转化为清晰可读的提示词，为检测这一新型数据投毒风险提供了有效手段。 |
| [^52] | [HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences](https://arxiv.org/abs/2609.16925) | HyCoSeq通过将加权洛伦兹残差聚合融入多曲率洛伦兹编码，并引入双向LSTM学习序列上下文关系，将局部双曲卷积编码扩展为序列级的上下文化基因组表示。 |
| [^53] | [Multi-Agent Learning with Cooperation-Driven Optimization Dynamics](https://arxiv.org/abs/2609.16917) | 提出了一种多智能体合作机制，通过在训练过程中将多个小型神经网络的预测信息融入损失函数以直接影响权重更新，从而在保持性能的同时降低模型复杂度。 |
| [^54] | [OptiPrime: Optimizing Private Inference through Protocol-Hardware Co-design](https://arxiv.org/abs/2609.16898) | OptiPrime通过协议-硬件协同设计，提出了一种新颖的卷积HE协议来大幅减少密文传输数量，从而突破网络通信瓶颈，显著提升基于同态加密和多方计算的私有DNN推理的端到端性能。 |
| [^55] | [NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI](https://arxiv.org/abs/2609.16873) | NeuroTS-Net是一种用于儿童脑肿瘤多类别语义分割的三维卷积神经网络，通过双尺度细节流、自适应上下文选择和保留细节的多路径下采样，在无需外部数据或预训练权重的情况下超越了nnU-Net和MedNeXt等基线方法。 |
| [^56] | [TEMPO: Learning Temporal Context for Dynamic Robot Manipulation](https://arxiv.org/abs/2609.16864) | 该论文指出VLA模型在动态操作中的失败源于运动歧义和状态混叠两种表征缺陷而非模型容量不足，并提出TEMPO通过引入从冻结视频基础模型提取的运动摘要等时间上下文输入来增强预训练VLA模型。 |
| [^57] | [Measuring Annotation Efficiency for Handwritten Devanagari Recognition: Sample-Complexity Curves for Four Pretraining Regimes](https://arxiv.org/abs/2609.16859) | 本研究通过控制识别器与评估协议不变、仅改变标注预算和预训练方式的系统实验，量化了手写天城文识别达到可用精度所需的转录样本数量，并将预训练收益换算为可节省的标注工作量。 |
| [^58] | [Can Deep Learning Achieve Cross-Physics Mapping?](https://arxiv.org/abs/2609.16853) | 提出跨物理映射（CPM）算子学习框架，通过兼容潜在表示与无量纲缩放原理，使深度学习能够在扩散方程与波动方程等根本不同的物理域之间实现场的相互映射，并系统评估了七种神经算子架构的性能。 |
| [^59] | [Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories](https://arxiv.org/abs/2609.16827) | 本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。 |
| [^60] | [Adapting to Decision-Relevant Non-Stationarity in Decentralized Heterogeneous Bandits](https://arxiv.org/abs/2609.16824) | 提出了DRFC算法，通过来自所有智能体的新鲜平衡样本在网络层面比较臂并仅在公共最优臂真正变化时切换，使去中心化异构老虎机的动态遗憾界只取决于决策相关的变化次数而非局部变化次数。 |
| [^61] | [LCAP: Population-Informed Latent Chip Adaptation from Few Output Probes for Photonic Neural Networks](https://arxiv.org/abs/2609.16823) | LCAP框架通过从80个历史芯片学习共享的种群校正，并利用32个固定的无标签输出探针前馈式推断新芯片的潜在校正坐标，无需在目标芯片上优化即可实现光子神经网络的快速个性化校准，有效弥合仿真到硬件的差距。 |
| [^62] | [ImpossibleRubrics: Stress-Testing Generated Rubrics as Reward Signals](https://arxiv.org/abs/2609.16816) | 该论文提出ImpossibleRubrics基准，利用169个不可能完成的任务对语言模型生成的评分准则进行对抗性压力测试，以检验其作为奖励信号时能否奖励诚实回答而非被对抗性回答钻空子。 |
| [^63] | [Geometry of learning dynamics: Gradient descent versus natural gradient on the ridge of optimization](https://arxiv.org/abs/2609.16805) | 论文通过对KLR训练的Hopfield网络统计流形的几何分析，揭示了优化脊上的学习分为两个阶段：标准梯度下降因极端曲率沿振荡的非测地线路径前进，而自然梯度下降遵循理想测地线路径并完全克服了这些不稳定性。 |
| [^64] | [SOTER: A Generative Time-Series Foundation Model for Wearable Human Physiological Signals](https://arxiv.org/abs/2609.16804) | SOTER是一个面向可穿戴生理信号的生成式时间序列基础模型，通过空间特征感知骨干网络、PSD引导的混合专家层和神经控制微分方程解码器，在统一框架中建模多通道耦合、频谱特异性和连续时间动态。 |
| [^65] | [On the disintegration of the stochastic majority vote: From PAC-Bayesian bounds to a self-bounding algorithm](https://arxiv.org/abs/2609.16803) | 本文提出了一种去随机化框架，将解积PAC-贝叶斯理论直接应用于多数投票权重向量空间，把随机多数投票的泛化保证转化为单一确定性多数投票的证书，并由此推导出两族高概率泛化界和一种自约束学习算法。 |
| [^66] | [Time-warping estimation via stationarity-based learning of the de-warped signal](https://arxiv.org/abs/2609.16796) | 本文提出可训练的时间扭曲估计模型TWET，将时间扭曲估计转化为小波域的平稳化问题，通过可微平稳性准则和分层膨胀卷积架构实现端到端优化，在提高形变重建精度的同时大幅减少计算时间。 |
| [^67] | [Noise2Noise Revisited: Training Pair Distributions Dominate Loss Choice in Self-Supervised Denoising](https://arxiv.org/abs/2609.16788) | 该研究否定了L1损失因参数稀疏性或最优解差异而在Noise2Noise去噪中优于L2的两个猜想，证明优势实际源于优化动力学，并指出训练对分布对去噪性能的影响超过损失函数的选择。 |
| [^68] | [TAME: Token Attribution and Masking for Emergent misalignment](https://arxiv.org/abs/2609.16754) | 该论文提出TAME三阶段框架，通过词元归因、信号表征与归因引导的损失掩码，定位并因果验证了微调数据中承载涌现性失调信号的关键词元，并发现失调信号高度集中于前5%的词元中。 |
| [^69] | [Constant Swap Regret in General-Sum Games via Optimistic Transition Matrices](https://arxiv.org/abs/2609.16751) | 该论文提出了一种基于乐观转移矩阵的确定性非耦合学习动态，首次在多人一般和博弈中实现了与时间范围无关的常数个体交换遗憾。 |
| [^70] | [The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action Chunking Transformer](https://arxiv.org/abs/2609.16745) | 该论文在原始代码中取证式地重跑了ACT的CVAE编码器消融实验，发现原始论文中移除编码器导致成功率从35%骤降至2%的结果无法重现，且训练时长和检查点选择方式都可能逆转策略间的表现排序。 |
| [^71] | [A Systematic Evaluation of Machine Learning Methods for Fault Detection and Line Identification in Electrical Power Grids](https://arxiv.org/abs/2609.16744) | 本研究系统评估了多种机器学习模型在电网故障检测和故障线路识别中的有效性，以克服传统继电保护系统基于静态规则的局限性。 |
| [^72] | [Carry-Through Checksum: A Lightweight Fault-Detection for CNN Inference at the Edge](https://arxiv.org/abs/2609.16742) | 提出了一种名为“传递校验和”的新方案，通过在卷积层中嵌入专用滤波器来计算并传播校验和，以极低开销实现嵌入式GPU上CNN推理的软错误检测。 |
| [^73] | [Unified Heterogeneous Graph Neural Network solver for Power Flow, Optimal Power Flow and State Estimation](https://arxiv.org/abs/2609.16738) | 该论文提出一个统一的异构残差门控图卷积网络，用单一共享骨干网络同时解决潮流计算、最优潮流和状态估计三个电力系统问题，精度媲美任务专用GNN求解器，并对未见过的负载水平和拓扑结构保持稳健。 |
| [^74] | [Seeing What Matters: Visual Cue Guided Video Planning for Generalizable Robot Navigation](https://arxiv.org/abs/2609.16737) | CueNav通过鸟瞰图和机器人身体等视觉线索引导视频规划，并利用逆动力学模型将视频计划转换为机器人动作，使迷宫导航成功率提升近2倍。 |
| [^75] | [Continuous-Time Machine Learning: A Unified Mathematical Perspective](https://arxiv.org/abs/2609.16710) | 本综述通过统一的数学分类框架，基于向量场参数化、随机性、记忆机制和离散化等核心数学表述，将分散于不同研究社区的连续时间机器学习各主要分支联系起来，并系统比较了它们的训练方法与设计权衡。 |
| [^76] | [Weave: Learning Whole-Body Dexterous Loco-Manipulation from Human-Object Interactions](https://arxiv.org/abs/2609.16683) | Weave提出一个统一框架，通过接触感知的重定向和接触与几何感知策略，将人类示范转化为可执行的全身控制，实现人形机器人从人-物交互中学习协调平衡、移动与灵巧手指操作的全身灵巧移动操作，在九个物体上取得92.5%的成功率。 |
| [^77] | [Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers](https://arxiv.org/abs/2609.16665) | 该论文揭示了循环Transformer中的“有限步失败”现象——局部改进的更新方向在完整步长下反而产生有害更新，并通过路径曲率分解和局部二次模型来预测有效步长规模及其理论误差界。 |
| [^78] | [GrowMTP: Can RL Grow Its Own Draft Head?](https://arxiv.org/abs/2609.16648) | 提出GrowMTP，利用RL训练自身的采样分布与验证监督信号，在RL循环内从零在线训练推测解码草稿头，免去额外的预训练或预热成本，且草稿头更新与策略主干完全解耦。 |
| [^79] | [Can Knowledge Transfer Parameters Be Learned? LePoKet for Efficient Robotic Vision](https://arxiv.org/abs/2609.16637) | 提出了LePoKet框架，通过可学习遗传注意力算子将知识迁移的交互参数与子网络联合优化，把知识继承直接嵌入前向计算，无需辅助蒸馏损失或温度缩放即可获得高效的紧凑机器人视觉网络。 |
| [^80] | [AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale](https://arxiv.org/abs/2609.16625) | 本文提出AURA，一个端到端的AI智能体系统，能够对大规模生产级推荐系统进行定性评估，提供超越传统汇总指标的可操作诊断与改进方案。 |
| [^81] | [Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction](https://arxiv.org/abs/2609.16621) | 提出变分自编码马尔可夫算子（VAMO），通过函数空间上的潜马尔可夫动力学、谱几何结构与变分转移对齐来正则化自回归误差传播，实现稳定的长时程偏微分方程预测。 |
| [^82] | [Divergence Timing and Cumulative Disagreement under KV-Cache Eviction](https://arxiv.org/abs/2609.16617) | 该论文建立了KV缓存驱逐引发自回归生成分歧的理论框架，将累积token不匹配精确分解为首次分歧贡献与分歧后暴露量的乘积形式，并通过条件蒙特卡洛无偏估计及Llama-3.1与Qwen2.5模型实验验证了不同缓存压缩策略在分歧时机上的差异。 |
| [^83] | [A Weighted Kernel Method for Approximation that Adapts to Learned Multivariable Structure](https://arxiv.org/abs/2609.16606) | 提出了总敏感度核（TSK）方法，通过加权ANOVA核族学习多变量结构，并借助最小范数RKHS的选择从有限数据中唯一确定各输入的敏感度因子，从而实现对黑箱函数的自适应近似。 |
| [^84] | [Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging](https://arxiv.org/abs/2609.16579) | 本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。 |
| [^85] | [AsyncCouple-Flow: Asynchronous Cross-Modal Coupling and Flow Matching for Spatio-Temporal Forecasting](https://arxiv.org/abs/2609.16573) | AsyncCouple-Flow 通过模态感知令牌稀疏化、异步跨模态耦合图和流匹配，联合解决了多模态时空预测中采样率不一致、模态易缺失和长时程误差累积三大难题。 |
| [^86] | [On the Importance of Gating: Memorization vs. In-Context Learning in State Space Models](https://arxiv.org/abs/2609.16540) | 本文通过理论和实验揭示了门控机制是状态空间模型在上下文学习任务上落后于Transformer的关键原因——它使模型先收敛到基于权重的记忆化方案，从而延迟甚至阻碍了正确的上下文学习方案的收敛。 |
| [^87] | [What Does Layer-Importance Reveal About Transformers and State-Space Models?](https://arxiv.org/abs/2609.16537) | 本文将层重要性分解为“必要性”（预训练模型对已有层贡献的依赖）和“可塑性”（微调时吸收新知识的位置）两个维度，揭示了Transformer与状态空间模型的根本差异——在残差Transformer中两者随深度反向分布，而在Mamba类SSM中两者趋于重叠。 |
| [^88] | [FlowATC: Aircraft Trajectory Prediction via Flow Matching](https://arxiv.org/abs/2609.16528) | 该论文提出了FlowATC，一种仅在历史ADS-B轨迹数据上训练、无需航线标签或航图监督的流匹配架构，能够生成与历史交通高度吻合的飞机轨迹分布，并准确再现旧金山机场周边的空域结构。 |
| [^89] | [High-Performance Tensor Formulation of the Viterbi Algorithm for Hidden Semi-Markov Models](https://arxiv.org/abs/2609.16500) | 本文提出隐半马尔可夫模型维特比算法的张量化公式，首次实现GPU加速实现，相比串行实现，在单核、多核和GPU上分别取得最高14倍、超过200倍和超过570倍的加速。 |
| [^90] | [From Manual Construction to AI-Driven Scenario Emergence: Rethinking Catastrophe Risk Modeling](https://arxiv.org/abs/2609.16493) | 本文提出TAISE框架，通过重新利用AI天气预报模型进行自迭代生成，产生连续的全球大气场使极端天气情景自然涌现，相比传统人工构建方法将计算成本降低一个数量级，并能捕捉时间连续性和跨区域相关性，为巨灾风险量化的普及化开辟新路径。 |
| [^91] | [Decoder Design Matters for ECG Delineation](https://arxiv.org/abs/2609.16489) | 提出将ResNet-18编码器与U-Net解码器配对的心电图划分模型R-U-Net，证明解码器设计对性能的贡献超过半监督学习方法，在域内和跨域设置中均显著超越现有最强基线。 |
| [^92] | [Skill-based Agentic Evaluation for Real-time Data Science Tasks](https://arxiv.org/abs/2609.16487) | 该论文提出“基准真值即代码”框架，通过可执行的参考函数在评估时从实时数据动态计算答案，并结合格式无关的事实性原子声明评分方法，解决了动态数据场景下数据科学智能体无法用静态基准真值评估的难题。 |
| [^93] | [Certified Inference and Training for Deep Equilibrium Networks: A Continuation Framework with Polynomial Complexity Guarantees](https://arxiv.org/abs/2609.16485) | 本文提出了一种认证延拓框架，将深度均衡网络的训练表述为精度插值问题，使推理和训练都能在多项式复杂度预算下获得可认证的保证。 |
| [^94] | [Online Gradient Computation for Warping Gaussian Process Transformations](https://arxiv.org/abs/2609.16472) | 本文证明了扭曲高斯过程瞬时负对数似然的梯度可进行精确递归计算，并据此提出一种能够同时更新潜在GP矩和优化扭曲参数的新型在线方法。 |
| [^95] | [A multimodal large language model for evidence-based autism spectrum disorder screening](https://arxiv.org/abs/2609.16464) | 本研究提出ASDchat——一个采用双分支架构的多模态大语言模型，能够基于视频、音频和对话进行自闭症筛查，同时生成与ADOS-2临床标准对齐的可追溯、带时间戳的行为证据，在中国27个站点1,035名参与者的数据集上达到0.953的AUC。 |
| [^96] | [Not All Relations Are Equal: Relation-Balanced and Calibrated Graph Learning for Provenance-Based Intrusion Detection](https://arxiv.org/abs/2609.16462) | 提出无监督框架RECAL，通过关系平衡的掩码图学习捕捉稀有交互模式，并结合针对各关系良性错误分布的误差校准机制，在DARPA E3数据集上实现最高99.99%的F1分数，有效降低基于溯源入侵检测中的误报和漏检风险。 |
| [^97] | [OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation](https://arxiv.org/abs/2609.16459) | OPD-Aha通过对比教师模型在真实图像与视觉空值下的预测，分离出其被师生语言惯性掩盖的视觉纠正信号，并据此重构蒸馏目标，解决了多模态在线策略蒸馏中师生收敛到相同幻觉的问题。 |
| [^98] | [Early-Bird Decoding: Accelerating Diffusion LLMs with Learnable Block Sizes and Parallel Sampling](https://arxiv.org/abs/2609.16450) | 首次提出“早鸟”解码框架，通过可学习网络将低熵聚集的 token 自适应分组为变长块，并结合位置感知采样器进行并行解掩码，从而大幅加速扩散大语言模型的推理。 |
| [^99] | [Decentralized Gossip Learning and Federated Averaging for Histopathology Image Classification](https://arxiv.org/abs/2609.16448) | 本研究系统比较了联邦平均、去中心化gossip学习和混合Gossip-FedAvg三种分布式学习方法在乳腺癌组织病理学图像分类中的性能，通过大规模数据集和全面的敏感性分析，为隐私受限医疗场景下的分布式学习方案选择提供了实证依据。 |
| [^100] | [Adaptive Bayesian Partner Selection for Federated Clinical Centers](https://arxiv.org/abs/2609.16446) | 本文提出ABPS框架，让联邦学习中的临床中心通过Beta-Bernoulli后验和UCB准则自适应地选择点对点协作伙伴，在大幅降低通信开销的同时避免负迁移，并具备理论遗憾保证。 |
| [^101] | [The Neverwhere Visual Parkour Benchmark Suite](https://arxiv.org/abs/2609.16443) | 提出了包含六十多个3D高斯泼溅重建场景的超逼真闭环评估基准套件Neverwhere，用于评估视觉运动控制器的真实世界部署性能，并揭示了仅依靠3D高斯生成数据训练策略的泛化局限。 |
| [^102] | [Learned Look-Ahead Splitting Rule for CART](https://arxiv.org/abs/2609.16440) | 该论文提出一种通过在候选分裂点下方生长CART子树来评估分裂质量的前瞻分裂规则，并利用节点级特征学习的智能前瞻算法大幅降低计算成本，在保持决策树可解释性的同时显著改善层级或交互场景下的分裂选择。 |
| [^103] | [Interpreting and Steering LLM Agents for Social Simulations](https://arxiv.org/abs/2609.16436) | 本文针对大语言模型作为黑箱在社会科学模拟中缺乏可解释性与可操控性的问题，比较了提示词操控、SAE特征引导和探测器方向引导三种方法来解释和操控LLM智能体的行为。 |
| [^104] | [How Good Are Time-Series Foundation Models for Pedestrian Crowd Count Forecasting? A Cross-Dataset Comparative Study](https://arxiv.org/abs/2609.16415) | 本研究通过在特殊活动短时数据与墨尔本多年传感器数据两种场景下，对七种涵盖传统方法、深度学习和预训练基础模型的预测方法进行跨数据集基准对比，系统评估了时序基础模型在行人计数预测中的实际迁移能力。 |
| [^105] | [On the Expressive Power of Implicit Line-Graph Higher-Order Weisfeiler--Leman](https://arxiv.org/abs/2609.16412) | 本文研究了隐式线图Weisfeiler-Leman测试的表达能力与根图WL的关系，发现在k=1,2时其区分能力不超过根图的1-WL，并首次证明了k=3时线图3-WL等价可推出根图3-WL等价的反向包含关系。 |
| [^106] | [Balancing Trial and Reorder: A Hybrid Sequential Transformer-GBDT Ranker for On-Demand Delivery](https://arxiv.org/abs/2609.16407) | 该论文提出了部署在Wolt的统一商家排序系统UVR，通过双向Transformer编码器进行序列用户建模并结合GBDT排序器，利用标签平滑和尝试偏置样本加权来平衡新商家探索与复购排序质量，以单一模型替代四个独立排序模型，使离线尝试MRR提升12%至30%。 |
| [^107] | [Physics Informed Random Feature Neural Networks for Solving PDEs](https://arxiv.org/abs/2609.16406) | 本文提出一种物理信息随机特征神经网络方法，通过缓解PINN求解器面临的谱偏差问题降低计算复杂度，并给出了严格的$H^1$范数高概率误差界分析。 |
| [^108] | [Implementing a White-Box Undetectable Backdoor for Random Fourier Features](https://arxiv.org/abs/2609.16403) | 本文仅用numpy和scipy端到端实现了基于CLWE问题的白盒不可检测RFF后门构造，验证了这类即使完整白盒审计也无法发现的后门威胁可以用普通科学计算工具实现。 |
| [^109] | [Attention Mean Fields Predict Average Representation Dynamics and Reveal Context-Specific Computation](https://arxiv.org/abs/2609.16382) | 本文提出一种注意力的平均场分析方法，通过逐层迭代的平均注意力核来预测语言模型表示几何结构的演化动态，并利用“平均场偏差”分离出平均场所无法捕捉的上下文特定计算。 |
| [^110] | [Bounded Adjustment with Reliability-Guided Embedding for Imbalanced Learning with Noisy Labels](https://arxiv.org/abs/2609.16380) | 该论文提出BARGE方法，将有界先验调整密度幂得分与可靠性引导嵌入结合为单阶段目标函数，在应对类别不平衡的同时约束噪声标签带来的分类风险扰动，并在模型与标签高度冲突时自动衰减梯度以抵抗标签噪声。 |
| [^111] | [Certified Uncertainty Propagation in One-Shot Federated Bayesian Models via Posterior Event Transport](https://arxiv.org/abs/2609.16373) | 本文提出一个与部署一致的认证框架，通过将客户端本地后验事件沿联邦平均等聚合规则进行几何传播，使一次性联邦贝叶斯模型的聚合部署模型也能获得可认证的安全保证。 |
| [^112] | [Fast-Convergent Meta-RL via Gradient-Clustered BS Sampling for Edge Caching](https://arxiv.org/abs/2609.16370) | 本文提出一种通过梯度聚类基站采样替代均匀随机采样的元强化学习边缘缓存框架，显著降低了大规模网络中元梯度估计的方差，实现了快速收敛。 |
| [^113] | [Autonomous Droplet Navigation via Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.16369) | 本研究提出基于模型的强化学习方法，实现了液滴在重力驱动迷宫平台上自主穿越日益复杂几何结构的导航，克服了接触角滞后和毛细钉扎等非线性难题。 |
| [^114] | [Mini-batch Sampling Strategies for Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT](https://arxiv.org/abs/2609.16365) | 本文在统一的偏差-方差框架下系统比较了均匀实例采样、类平衡采样、平方根采样和渐进平衡采样四种小批量采样策略对长尾图像分类中梯度估计的影响，并在CIFAR-100-LT数据集上进行了实证评估。 |
| [^115] | [EBL: Efficient Broad Learning for Distributed Adaptive Harmonic Analysis](https://arxiv.org/abs/2609.16358) | 本文提出面向分布式自适应谐波估计的高效宽度学习（EBL）量化FPGA加速框架，以半个周波输入实现高精度、超低延迟（比现有最快FPGA方法快17.4倍）且可重构的谐波分析。 |
| [^116] | [Federated stochastic bilevel optimization with fully first-order gradients](https://arxiv.org/abs/2609.16350) | 提出了一种仅需一阶梯度的联邦随机方差缩减双层梯度下降算法，避免了二阶Hessian和Jacobian矩阵的计算从而显著降低运行时间，并结合常数单时间尺度学习率机制实现了良好的收敛性能。 |
| [^117] | [Multi-Label Proportion Learning for Sea-Ice Type Prediction](https://arxiv.org/abs/2609.16347) | 本文提出多标签比例学习方法，直接从多边形级别的冰情图标签中学习每个图块内各类海冰的比例分布，从而避免传统监督方法中因近似图块级标签而导致的病态学习问题。 |
| [^118] | [Channel-Informed Neural Network for Physical Layer Key Generation](https://arxiv.org/abs/2609.16341) | 本文提出一种信道信息驱动的多任务循环神经网络，直接从接收的IQ测量中生成保持互易性的二进制密钥特征，并通过深度度量学习与信道信息监督相结合的方式实现轻量级的物理层密钥生成。 |
| [^119] | [StalePO: Anchored Token-Level Preference Optimization using Legacy Post-Edits in Machine Translation](https://arxiv.org/abs/2609.16340) | 提出StalePO方法，通过在两个响应上同时下调似然、将策略锚定于自身基础响应以及施加token级KL约束，解决了机器翻译模型升级中利用陈旧偏好信号进行偏好优化时DPO失效的“陈旧偏好问题”。 |
| [^120] | [Breaking the 1.58-bit Barrier for Ternary LLMs](https://arxiv.org/abs/2609.16338) | 该论文发现三值LLM权重中零值占比高达51.5%，并据此提出分布自适应存储布局BITCOS，将每权重存储成本降至2-z比特，从而突破传统三值模型log₂3≈1.585比特的存储壁垒。 |
| [^121] | [Cross-Anatomy Transfer Versus Sparse Interpolation in Digital-Twin-Oriented Aortic Fluid-Structure Interaction Surrogates](https://arxiv.org/abs/2609.16322) | 本研究通过对四个主动脉模型的双向流固耦合分析发现，仅基于几何的机器学习先验模型跨解剖结构零样本迁移效果不佳，而基于稀疏锚点的传统插值方法（如反距离加权）表现更好，表明数字孪生流固耦合代理模型的可信度评估必须区分跨解剖迁移与表面内插值这两种根本不同的泛化能力。 |
| [^122] | [FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software](https://arxiv.org/abs/2609.16321) | FairLint-DL是一个VS Code扩展工具，通过训练代理神经网络并应用基于信息论的QID指标，实现了在训练前直接在IDE中对表格数据集进行偏见检测、因果定位和可解释性分析的公平性调试。 |
| [^123] | [Generative models for simulation based filtering: Formulations and Empirical Comparisons](https://arxiv.org/abs/2609.16317) | 本文提出了非线性滤波问题的统一生成式模型框架，基于随机插值、流匹配和薛定谔桥推导出三种新滤波器，并与OTF、KRF、SIR粒子滤波器和EnKF等传统方法进行了系统实证比较。 |
| [^124] | [Robust Fault Detection in Mechanical Multimodal Time Series via Self-Supervised Cross-Modal Reconstruction](https://arxiv.org/abs/2609.16314) | 提出一种基于自监督跨模态重构的多模态异常检测框架，通过利用跨模态关系显著提升机械多模态时间序列故障检测在分布偏移场景下的鲁棒性。 |
| [^125] | [Agentic Search Spaces for Tabular Machine Learning](https://arxiv.org/abs/2609.16309) | 本文证明基于LLM的智能体能够为表格机器学习模型设计扩展的超参数优化搜索空间，在45个数据集上平均带来0.6%的相对性能提升（小型数据集上达2.0%），超越模型作者提供的标准搜索空间。 |
| [^126] | [Sequence Recognition in Bharatnatyam dance](https://arxiv.org/abs/2609.16306) | 本文提出了一种结合CNN识别关键姿势、SVM识别动作，并利用编辑距离算法进行序列匹配的巴拉塔纳蒂亚姆舞Adavu舞蹈序列识别方法，最终识别准确率达98%。 |
| [^127] | [BLINDSPOT: A Benchmark for Safety and Refusal Calibration in Long-Horizon Tool-Using Agents](https://arxiv.org/abs/2609.16305) | 本文提出Blindspot基准测试，通过自适应对抗交互、有状态工具执行和基于执行的裁决来评估长程工具使用智能体在完整交互轨迹上的安全与拒绝校准能力，涵盖22个攻击类别、7个领域35个场景及2,500余条长程轨迹。 |
| [^128] | [Nationally Consistent, Locally Incomplete: A Bayesian Remote-Sensing Audit of Rooftop Photovoltaic Registries](https://arxiv.org/abs/2609.16294) | 本文提出一个贝叶斯遥感审计框架，将不完美的光伏检测结果转化为考虑不确定性的测量工具，在法国估计的屋顶光伏容量与官方数据全国偏差仅3.3%，却揭示出局部地区高达61%的容量漏报及开放数据中的显著截断偏差。 |
| [^129] | [Symmetric solution of the Bellman optimality equation for repeated harmony game](https://arxiv.org/abs/2609.16289) | 本文通过求解重复和谐博弈的贝尔曼最优方程，发现了三种类型的对称解，分别对应全合作策略、“赢则保持输则改变”策略以及一种具有非平凡行为的新策略，并通过数值实验研究了智能体实际学到的策略。 |
| [^130] | [Drift Field Net: Learning Ocean Lagrangian advection fields from in-situ and satellite observations](https://arxiv.org/abs/2609.16288) | 提出漂移场网络（DFN），通过模拟数据预训练结合平流一致性损失的拉格朗日微调这一两阶段物理信息训练策略，从卫星观测预测海洋表面流场，使7天预报的平均定位误差比业务化物理预报系统降低20公里。 |
| [^131] | [Differentially Private Semantic Plans for Aggregate Insight Generation](https://arxiv.org/abs/2609.16283) | 提出DP-SPIN可信管理者框架，通过将记录映射为语义概念上的有界稀疏非负向量形成语义草图，并借助差分隐私机制发布语义计划，实现了对独立于数据定义的语义概念在跨数据集合和重复分析中可比较的聚合度量与摘要。 |
| [^132] | [Scaling Laws for Physics-Aware ACOPF Surrogate Learning](https://arxiv.org/abs/2609.16282) | 在ACOPF代理学习中，物理感知的增广拉格朗日目标相比MSE能以可忽略的额外内存将约束违反降低近30倍，且其改善在模型容量与数据规模之间更均衡，而MSE的违反量随网络规模增长快约两倍。 |
| [^133] | [Semantic-Aware Neural Video Codec for Error-Resilient Low-Latency Transmission](https://arxiv.org/abs/2609.16279) | 该论文提出一种基于DCVC-RT的语义感知多级神经视频编解码方法，通过语义与特征感知的分级打包优先级策略和消除包间依赖的抗误码熵模型，实现了不可靠信道下鲁棒的低延迟视频传输。 |
| [^134] | [The record is part of the task: matched-record evaluation of text classifiers across maintenance, safety and recall reporting](https://arxiv.org/abs/2609.16267) | 该论文的核心发现是：记录选择本身就是文本分类器评估任务的关键组成部分——同一案例的不同工作流记录（如客户报告与技术员报告）之间的F1差异可达0.46，远大于在同一数据上比较不同表示方法和模型架构所带来的性能差异。 |
| [^135] | [Towards Surrogate Based Dequantization of Quantum Reinforcement Learning](https://arxiv.org/abs/2609.16266) | 该论文基于监督学习中核方法的去量子化成果，首次探索将基于代理模型的去量子化方法扩展至强化学习领域，通过构建高效经典算法来匹配量子Q学习等量子变分方法的性能，以检验量子强化学习对实际问题是否具有可证明的量子优势。 |
| [^136] | [Compute-Optimal Pretrain--Fine-tune in Ridge Gradient Descent](https://arxiv.org/abs/2609.16262) | 本文在岭回归梯度下降的两阶段预训练—微调框架下，首次从理论上刻画了固定总优化预算时上游预训练与下游微调之间的最优计算分配，并揭示该分配由预测相关的谱分量和下游数据几何共同决定。 |
| [^137] | [The AI-Enabled Scientific Frontier](https://arxiv.org/abs/2609.16258) | 该研究通过分析2000至2025年间27个学科中2507项AI与传统科学方法的直接对比，发现AI相对传统统计学通常性能更优但成本更高，而相对科学计算通常性能较差但成本更低，不过自2020年以来AI在后者上的表现已显著提升。 |
| [^138] | [Efficient Reasoning Distillation: Small Video-Language Models via Synthetic CoT and Difficulty-Aware Fine-Tuning](https://arxiv.org/abs/2609.16255) | 该论文提出一种高效蒸馏方法，仅用约900个样本和单卡A100不到两小时的训练，让2B小型视频语言模型通过4B教师生成的合成CoT和“CoT后置于答案”的策略，超越4倍大的模型并逼近其4B教师模型的性能。 |
| [^139] | [Copula Adapted Directed Acyclic Graph for Cluster Representation of Biomedical Data](https://arxiv.org/abs/2609.16240) | 本文提出了一种融合Copula非高斯非线性依赖建模与基于有向无环图的集成因果结构发现方法的新型数据表示框架，用于无标签高维生物医学数据的聚类表示。 |
| [^140] | [Improving Reduced-Order Rotating Detonation Engine Models with Data Assimilation and Machine Learning](https://arxiv.org/abs/2609.16237) | 该论文提出利用连续数据同化（推引）方法将低维Koch-Kutz模型与高保真度模拟的温度数据同步，并以记录的强迫项作为模型修正的状态依赖估计，从而在低计算成本下显著改进旋转爆震发动机降阶模型的预测能力。 |
| [^141] | [Test-Time Unlearning via Sparse Autoencoder](https://arxiv.org/abs/2609.16229) | 提出ARIA方法，利用稀疏自编码器检测遗忘相关状态并进行可解释干预，在完全不改模型权重的前提下实现测试时知识遗忘，有效缓解了传统梯度上升方法的遗忘-效用权衡问题。 |
| [^142] | [How I learned to stop worrying and love StopGrads: Stationarity, Convergence, and a case study on Flow Map Learning](https://arxiv.org/abs/2609.16222) | 该论文提出了停止梯度回归原理，统一了流映射、强化学习和扩散采样中的停止梯度目标，并从理论上证明了停止梯度流映射目标的唯一驻点是真实流映射，同时给出了正面的收敛性保证。 |
| [^143] | [Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration](https://arxiv.org/abs/2609.16204) | 提出诱饵方向优化（DDO），一种无需微调的快速事后权重编辑防御方法，通过向MLP神经元注入高幅度非线性诱饵信号来破坏消融攻击所依赖的对比估计器，从而保护大语言模型的安全防护栏免受拒绝特征消融攻击。 |
| [^144] | [A Sentinel-2 benchmark dataset for deep-learning active-fire segmentation across 25 California wildfires](https://arxiv.org/abs/2609.16199) | 该论文发布了一个涵盖25起加州野火、包含2,148对图像-掩膜的开放Sentinel-2基准数据集，用于推动和评估深度学习活跃火分割方法的开发。 |
| [^145] | [Permutation-Based Stegomalware in Large Language Models: Threats and Countermeasures](https://arxiv.org/abs/2609.16193) | 本文证明排列对称性既能被防御者用于完全中和LLM所有权重中的隐写恶意软件，也能被攻击者利用以理论不可检测的方式将恶意软件编码进模型权重。 |
| [^146] | [Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It](https://arxiv.org/abs/2609.16183) | 该论文通过匹配状态预算的单变量分解，揭示了固定状态循环网络在联想回忆上的差距主要由短因果卷积而非循环结构类型决定，识别出限制多查询回忆的“干扰墙”现象，并提出课程学习方法来突破这一瓶颈。 |
| [^147] | [Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers](https://arxiv.org/abs/2609.16179) | 本文从反向传播视角重新审视Z-loss，指出其产生的logit空间梯度（“反向源”）对密集输出头和稀疏路由器的影响取决于其所处的架构与实现方式。 |
| [^148] | [Skeletal Prototypes on Iterative Nerve Expansions](https://arxiv.org/abs/2609.16170) | SPINE方法创新性地用嵌入的一维复形（骨架结构）而非传统点集来表示各类原型，通过类条件Mapper图构建初始边集并在分类目标下优化顶点位置，使骨架线段直接参与决策规则，在17个基准数据集上取得了最优的平均准确率和排名。 |
| [^149] | [GPEvac: GNN-Based PPO for Adaptive Evacuation Routing During Shooting Events](https://arxiv.org/abs/2609.16163) | 该论文提出GPEvac框架，通过边优先顺序消息传递与可学习虚拟全局节点的图神经网络结合PPO强化学习，并利用置换不变评分机制，实现了单一策略在不同拓扑和规模建筑布局间泛化的枪击事件自适应疏散路径规划。 |
| [^150] | [LLM Inference in a Flash!](https://arxiv.org/abs/2609.16161) | 该论文提出利用闪存内计算作为解决LLM推理所面临的内存带宽与容量瓶颈的方案，以应对长序列、重推理请求带来的日益增长的挑战。 |
| [^151] | [Computer-assisted global regularity across nonlinear families of three-dimensional periodic Navier-Stokes flows](https://arxiv.org/abs/2609.16157) | 本文开发了一种计算机辅助框架，通过将有限参考轨迹与公共误差界相结合，为三维周期性Navier-Stokes流的连续非线性族建立了全局正则性，并成功应用于循环剪切场、ABC场和三组分Taylor-Green场。 |
| [^152] | [LLMs as Master Forgers: Generating Synthetic Time Series Data for Manufacturing](https://arxiv.org/abs/2609.16155) | 本文提出通过微调大语言模型并结合检索增强生成（RAG）技术为制造业生成高质量合成时间序列数据，在异常检测等下游任务中表现优于ARIMA和LSTM等传统方法。 |
| [^153] | [Safe Error Correction for Language Models: Frozen-Base Adjustment with Capability Preservation](https://arxiv.org/abs/2609.16145) | 提出仅占模型参数 0.73% 的轻量级纠正模块 CRN v2，能在完全冻结的语言模型上修复 53.3% 的输出错误且不损害基础能力，而同预算的 LoRA 虽纠正率更高却会导致 30-75% 的能力损失。 |
| [^154] | [A Decision-Support Audit Protocol for Supervision Drift in Proxy-Labeled Credit-Risk Prediction](https://arxiv.org/abs/2609.16102) | 该论文提出一个针对代理标签信用风险预测中监督漂移的锁定式五层多信号审计协议，在LendingClub数据上发现排序性能稳定，时间漂移主要源于基础比率与概率尺度的失配，且可通过仅截距再校准大幅缓解。 |
| [^155] | [SWB-DM: A Calibrated Sliced-Wasserstein-Barycenter Aggregator with Delayed-Momentum Caching for Byzantine-Robust Federated Learning under Partial Participation](https://arxiv.org/abs/2609.16099) | 该论文提出SWB-DM，通过带规范固定的切片Wasserstein重心聚合器结合对全体客户端的延迟动量缓存机制，解决了部分参与场景下鲁棒聚合有限样本保证失效的问题，实现了拜占庭鲁棒的联邦学习。 |
| [^156] | [Evaluating Open-Weight E-Commerce Agents with Environment-Grounded Verification](https://arxiv.org/abs/2609.16093) | 该论文构建了一个确定且可复现的电商评估环境，通过预先固定试验参数并记录助手动作与环境状态的对应证据，实现了超越单一任务成功率的、基于环境锚定的细粒度智能体对话评估。 |
| [^157] | [Distilling Foundation Models for Agentic What-If Reasoning:Cost, Latency, and Governance in a Hybrid LLM+SLM Architecture](https://arxiv.org/abs/2609.16091) | 本文通过将TabPFN表格基础模型蒸馏为紧凑的前馈学生网络，实现了高达6,532倍的参数压缩，同时保留95%以上的预测性能，使表格基础模型能够作为混合LLM+SLM架构中智能体假设推理的低延迟热路径决策后端。 |
| [^158] | [Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and Automotive Accelerators](https://arxiv.org/abs/2609.16085) | INT8 量化模型并非跨平台可移植：相同模型在不同硬件上加速效果截然不同（有无点积指令集可导致 2.1 倍加速或 1.7 倍减速），且 INT8 推理输出在不同平台间不保证一致，打破了“一次量化、随处部署”的普遍假设。 |
| [^159] | [Predicting Social Media Engagement using Machine Learning](https://arxiv.org/abs/2609.16082) | 本研究通过提取家具公司Facebook图片帖子的视觉、文本和时间特征，并运用随机森林、LightGBM等机器学习模型来预测社交媒体互动度。 |
| [^160] | [Pseudo-Label Augmentation for Affect Sensing in Small Collaborative Groups](https://arxiv.org/abs/2609.16077) | 该研究提出在小型协作小组情感感知中使用伪标签增强以应对标签稀疏问题，结果表明伪标签增强优于仅用标注数据的基线，但大五人格相似度取值范围过窄，细粒度人格加权效果有限，人格相似性主要仅起同队过滤器作用。 |
| [^161] | [Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap](https://arxiv.org/abs/2609.16071) | 该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。 |
| [^162] | [Beyond Distribution Matching: Semantics-Consistent Tabular Diffusion with Weak Semantic Priors](https://arxiv.org/abs/2609.16069) | 该论文提出了一种语义一致性表格扩散框架，利用大语言模型从元数据中提取弱语义先验（列内语义和列间符号规则）并将其作为生成条件而非事后过滤，从而解决了现有表格生成器仅优化分布匹配而忽视语义约束的问题。 |
| [^163] | [A Dynamic Aggregation Strategy Enhanced Efficient Global Optimization Algorithm for Solving High-Dimensional Turbomachinery Design Problems](https://arxiv.org/abs/2609.16067) | 提出了一种动态聚合策略增强的高效全局优化算法DA-EGO，通过将高维设计空间动态分解为低维子空间、基于变量交互分析自适应更新子空间并调整搜索范围，有效求解了高维昂贵的叶轮机械设计优化问题。 |
| [^164] | [A panoramic aerodynamic performance prediction method for turbomachinery cascades using transformer-enhanced neural operator](https://arxiv.org/abs/2609.16066) | 本文提出一种基于Transformer增强神经算子（TNO）的全景性能预测框架，通过先预测Navier-Stokes方程的基本物理量（温度、压力、密度）再推导关键性能参数，实现了叶轮机械叶栅多种气动性能目标的快速灵活评估，功能类似于CFD模拟器。 |
| [^165] | [You Don't Need To Train: Agentic Heuristic Learning Studio for Executable Human Activity Recognition](https://arxiv.org/abs/2609.16065) | 该论文提出智能体启发式学习（AHL）工作室，模仿人类认知学习方式（记忆示例、形成规则、修复错误）而非梯度训练，生成可执行、可检查、可编辑且无需LLM的人体活动识别策略，并支持边缘部署。 |
| [^166] | [Signed p-adic Residual Encodings of Finite-Domain All-Different Systems with a Sudoku Case Study](https://arxiv.org/abs/2609.16063) | 该论文提出用带符号加权的仿射p进剩余目标函数作为有限域约束的原生编码，通过逐坐标支配定理证明所有全局极小值都位于有限域内，且损失恰好对应全不同冲突数或负的已满足CNF子句数，并以无需独热提升的标准数独（81系数）案例验证了该方法。 |
| [^167] | [Digital Persuasion: Understanding the Impact of Online Influencers on Public Opinion](https://arxiv.org/abs/2609.16062) | 本文提出一个基于Friedkin-Johnsen模型的框架来识别网络意见领袖，并通过初始观点操纵实验在总统选举推文数据集上验证了意见领袖能够显著改变社区整体舆论。 |
| [^168] | [POSPAN: Position-Constrained Span Masking for Language Model Pre-training](https://arxiv.org/abs/2609.16061) | 提出POSPAN通用框架，通过结合片段长度分布与位置约束分布来支持多样化的位置约束式片段掩码策略，统一了所有现有的片段级掩码方法，并在多个NLU基准上显著提升了预训练语言模型的性能。 |
| [^169] | [HintMiner: Automatic Question Hints Mining From Q&A Web Posts with Language Model via Self-Supervised Learning](https://arxiv.org/abs/2609.16060) | 本文提出了HintMiner工具，通过自监督学习训练基于Transformer和复制机制的编码器-解码器模型MiningNet，从海量网络问答帖子中自动挖掘并生成用户问题的提示，帮助用户更快找到答案。 |
| [^170] | [Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation](https://arxiv.org/abs/2609.16059) | 提出MIFS系统化数据合成流程，通过生成式约束协议、可学习性感知蒸馏机制和基于代码的验证器，生成可直接用于强化学习的多模态指令跟随数据，突破RLVR在多模态领域可扩展性的数据瓶颈。 |
| [^171] | [Driver Behavior Estimation at Signalized Intersections Using a Physics-Constrained Decision-Conditioned Autoregressive Transformer](https://arxiv.org/abs/2609.16058) | 该论文基于449次真实驾驶轨迹数据，提出了一个物理约束且以决策为条件的两阶段自回归Transformer框架，用于预测信号灯切换时驾驶员的停走决策与纵向轨迹行为，并发现所需减速度是决策的关键预测因子，由此推导出五个经验性舒适减速度区间。 |
| [^172] | [OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning](https://arxiv.org/abs/2609.16057) | 提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。 |
| [^173] | [Managing Action Preconditions in Neuro-Symbolic RL: Three Placement Strategies for Embodied Agents](https://arxiv.org/abs/2609.16056) | 该论文将结构动作的前置条件知识形式化为贝叶斯网络，并提出将其注入强化学习循环的三种放置策略，以避免幻觉式前置条件、提升具身智能体在变化环境中的安全性与可靠性。 |
| [^174] | [Causal neural set filtering for online multi-target tracking](https://arxiv.org/abs/2609.16054) | 本文提出因果神经集合滤波器（CNSF），通过仅编码当前测量并借助结构化递归航迹状态携带历史信息，结合排他性Sinkhorn关联、关联条件卡尔曼形态更新和递归伯努利生命周期建模，在消除冗余计算的同时实现了优于MT3/Track-MT3的多目标跟踪精度。 |
| [^175] | [A deep dictionary network-based foundation model for ultra-low-dose CT denoising](https://arxiv.org/abs/2609.16031) | 该论文提出了一种基于深度字典网络（DDN）的架构可解释基础模型，将卷积稀疏编码层与迭代软阈值级联，实现了跨多器官的统一超低剂量CT图像去噪。 |
| [^176] | [Molecular representation shapes the balance between target fidelity and exploration in flow based polymer generation](https://arxiv.org/abs/2609.16028) | 该研究提出基于潜空间连续时间流匹配的PolyLatentFlow生成框架与融合聚合物序列及三维结构信息的LlamaUni多模态分子表示，实现了有效且新颖聚合物候选物产出的显著提升，并揭示了分子表示对目标保真度与探索能力之间平衡的关键影响。 |
| [^177] | [3D Field Data Reduction with Adaptive Sample-Based Gaussian-Encoded Reconstruction](https://arxiv.org/abs/2609.16024) | 该论文提出一种统一的基于样本的自适应高斯编码方法，能在固定预算下同时表示结构化网格、非结构化网格和粒子场数据，相比现有方法以约44倍更少的基元实现高达4.8 dB的PSNR提升。 |
| [^178] | [Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks](https://arxiv.org/abs/2609.16023) | 该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。 |
| [^179] | [ViCo: Visual-oriented Coding with Self-Reflection for Chart Replication](https://arxiv.org/abs/2609.16014) | ViCo是一个面向视觉编码的训练框架，通过自监督预热阶段结合基于一致性剪枝的蒙特卡洛树搜索来合成高质量反思轨迹，并利用迭代反思使生成的图表图像逐步与参考对齐，从而生成符合人类论文视觉标准的高质量学术图表。 |
| [^180] | [EMODY Flow: Emotion-Aware Audio-Driven Full-Body Motion Generation](https://arxiv.org/abs/2609.16011) | 提出轻量级流匹配框架EMODY Flow，通过复用冻结Qwen-3 Omni模型的内部Mimi音频编解码器来条件化并行的身体姿态和面部表情DiT生成器，解决了情感条件化失效问题，实现与语音和情感状态同步的全身动作生成。 |
| [^181] | [Measuring AI harms with multidimensional Lorenz Zonoids](https://arxiv.org/abs/2609.16004) | 本文提出将洛伦兹Zonoid和基尼指数扩展到多维情形，用以对真实人工智能危害数据进行基于严重程度的风险评估，从而提供一种超越合规导向、能够识别干预优先级的有效AI风险管理方法。 |
| [^182] | [Crash Narrative-Guided Countermeasure Recommendation Using Large Language Models: A Retrieval-Augmented Generation Framework for Intersection Safety](https://arxiv.org/abs/2609.15997) | 该研究提出一种基于大语言模型的事故叙述引导检索增强生成（RAG）框架，从非结构化事故叙述中提取关键事故机理属性，并将其与循证安全对策数据库关联，从而自动生成针对交叉口的具体安全改善措施推荐。 |
| [^183] | [Latent Undertow: How Ordinary Typos Break Probes](https://arxiv.org/abs/2609.15994) | 本文揭示了普通拼写错误虽不影响大模型语义理解，却会严重破坏基于隐藏状态的恶意提示检测探针的性能，并提出KV缓存分叉方法——在用户消息后附加固定后缀使探针读取扰动下游token，可恢复95%的性能损失。 |
| [^184] | [Single Document Extractive Summarization using Domination in Hypergraph](https://arxiv.org/abs/2609.15993) | 本文提出了一种新颖的单文档抽取式摘要方法，通过构建以句子为节点、关键词为边的超图，并利用贪心算法寻找支配集来生成摘要，其性能可与最先进的基于图的方法相媲美。 |
| [^185] | [Few-Shot Degradation Is Not What It Seems: Behavioral Evidence, Representation Analysis, and a Random-Text Control Across 12 Models, 2 Tasks, and 2 Architectures](https://arxiv.org/abs/2609.15990) | 该研究揭示少样本提示对语言模型的影响强烈依赖任务类型，并提出通过长度匹配的随机文本对照构建“内容增量”指标，将示例内容的影响与提示长度的干扰分离，从而更准确地解释少样本退化的真正成因。 |
| [^186] | [Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science](https://arxiv.org/abs/2609.15983) | 提出Stellar Colosseum，一个与模型无关的多智能体框架，通过策略探索、就绪门控、子问题分解、定向证伪和树聚合等机制，提升语言模型在数学与理论计算机科学长周期研究问题上的可靠性。 |
| [^187] | [Discrete Beckmann Transport Models for One-Step Language Modeling and Reasoning](https://arxiv.org/abs/2609.15903) | 该论文提出离散贝克曼输运模型（DBTM），通过时间无关流的自治输运映射在理论上可证明单步将任意点映射至单纯形顶点的不动点，无需教师模型蒸馏与时间条件化，即可实现高效的单步语言建模与推理。 |
| [^188] | [K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations](https://arxiv.org/abs/2609.15855) | 该论文提出了K-Bench，一个经临床医生校准的基准，包含200个涉及自杀、自残、家庭暴力等多轮高风险心理健康情景案例，并采用与临床医生共识达94.2%一致率的GPT-4o自动评判系统，系统评估了33个基础大语言模型在高风险心理健康对话中的安全性。 |
| [^189] | [The Token Before the Value Is the Key: How Hybrid Architectures Organize Induction Circuits](https://arxiv.org/abs/2609.15545) | 该论文发现混合语言模型中归纳电路存在明确分工——“携带前驱信息”集中于局部/循环等高效层而“内容匹配”集中于全局层，且通过干预前驱支持（如滞后一掩码、移除卷积等）可以在不同层之间重新分配这些功能。 |
| [^190] | [Real-World Deployment and Performance Characterisation of Fog-Based Deep Learning for Cold-Chain Temperature Prediction over LoRaWAN](https://arxiv.org/abs/2609.14036) | 本文首次在真实环境中部署了基于雾计算的LSTM-GRU深度学习模型，利用LoRaWAN传感器数据在树莓派4上实现了冷库温度预测（MAE为0.2°C），并仅在预测到冷链断裂时生成SHAP可解释性结果，以极低能耗实现了边缘智能冷链监控。 |
| [^191] | [MANAS-2: Constrained Reconstruction for EEG Foundation Models](https://arxiv.org/abs/2609.13717) | MANAS-2 提出一种结合原始-频带混合掩码自编码器与物理启发的受限重建正则化的 EEG 基础模型，通过惩罚重建波形相邻窗口的 RMS 能量差异来引导编码器组织振荡包络信息，显著提升了脑电频谱功率与频带能量动态的表示质量。 |
| [^192] | [Nonsmooth Optimization via Orthogonalized Momentum](https://arxiv.org/abs/2609.13677) | 本文首次在与反向传播兼容的广义导数框架下证明：对于任意固定动量因子β∈[0,1)，正交化动量优化器Muon在凸Lipschitz非光滑目标上几乎从任何初始化出发都可能无法收敛到全局最优解，揭示了其超越光滑优化的根本局限。 |
| [^193] | [Stochastic Gradient Descent over P2](https://arxiv.org/abs/2609.13343) | 该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。 |
| [^194] | [Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction](https://arxiv.org/abs/2609.13285) | 本文提出分组值注意力（GVA），通过仅存储分组值并用可吸收进查询的线性映射按需重构内容键，无需在解码时缓存内容键，配合解耦RoPE位置通道保留位置信息，相比GQA可减少约45-47%的KV缓存开销且准确率几乎不受影响。 |
| [^195] | [Algorithmic Information Dynamics of Learning: A Certified, Differentiable Complexity Controller for Grokking](https://arxiv.org/abs/2609.13197) | 本文提出一种可认证、可微的算法复杂度估计器并将其用作控制器，能够在奥卡姆边界内加速神经网络的grokking过程，且相比训练损失门控以减少27%的干预达到同等的失败种子拯救效果。 |
| [^196] | [Very Exciting: Zero-Shot Model Predictive Control of Buildings via Excitation-Based Generalized Transfer Learning Models](https://arxiv.org/abs/2609.12853) | 该论文提出使用基于激励式探测数据的广义迁移学习模型，实现了建筑的零样本模型预测控制，无需在目标建筑收集数据即可获得令人满意的控制性能。 |
| [^197] | [From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good](https://arxiv.org/abs/2609.11910) | 本文以教皇利奥十四世的《Magnifica Humanitas》道德框架为基础，主张负责任人工智能应超越原则与协议层面的承诺，确立有边界的、可验证的证据主张，并同时评估AI系统本身及其所介入的制度性失灵。 |
| [^198] | [The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement](https://arxiv.org/abs/2609.11873) | 本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。 |
| [^199] | [Meta-LinEXP3: Online-within-Online Learning for Adversarial Linear Contextual Bandits](https://arxiv.org/abs/2609.09907) | 该论文提出了Meta-LinEXP3算法，首次针对具有随机动作集的对抗性线性上下文老虎机实现了元学习，通过从已完成任务构建可预测的任务级先验来指导内部LinEXP3学习器，并建立了先验精度与迁移遗憾之间的直接联系。 |
| [^200] | [Forward-Free LLM Depth Pruning via Weight Redundancy](https://arxiv.org/abs/2609.09883) | 提出了一种无需前向传播的深度剪枝方法WRP，通过直接从模型权重中估计层间冗余（比较注意力输出与MLP下投影权重的相似性）来选择剪枝块，无需校准数据即可达到接近基于激活方法的性能。 |
| [^201] | [BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL](https://arxiv.org/abs/2609.09783) | BRACE通过将贝尔曼校正范围限制在策略token前缀并对超出部分锚定常数权重蒙特卡洛尾部，解决了异步强化学习中评论家因策略滞后产生的偏差，在BrowseComp-Plus上比最强基线提升2.4%且每步快2.46倍。 |
| [^202] | [Adaptive Anisotropic Attention for Axis-Structured Signals](https://arxiv.org/abs/2609.08788) | 提出自适应各向异性注意力（AAA），将注意力沿电极时间轴分解为时间路径与空间路径并通过门控自适应加权融合，所构建的AXON模型在六个EEG下游任务上优于稠密注意力基线。 |
| [^203] | [CoRL: Co-Evolutionary Reinforcement Learning for Adaptive Indirect Prompt-Injection Attacks and Defenses](https://arxiv.org/abs/2609.07529) | 该论文提出CoRL框架，将自适应间接提示注入攻防建模为非对称、部分可观测的马尔可夫博弈，通过攻击者SFT、双边Co-PPO协同进化和防御者SFT三个阶段实现攻防双方的共同进化，从而训练出能够抵御不断变化攻击策略的鲁棒防御能力。 |
| [^204] | [Hessian-based molecular conformation augmentation for a scalable and efficient strategy of machine learning interatomic potentials](https://arxiv.org/abs/2609.05233) | 该论文提出两种基于Hessian矩阵的数据增强方案（UniAug和ModeAug），通过简单的泰勒展开生成增强分子构型，无需修改模型架构或引入额外计算与内存开销，即可有效利用Hessian信息训练机器学习原子间势。 |
| [^205] | [ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing](https://arxiv.org/abs/2608.26083) | ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。 |
| [^206] | [M-Fibration Theory with Applications to Neural Network Compression](https://arxiv.org/abs/2608.25598) | 本文提出了一个基于交换幺半群的图纤维化扩展理论，支持加权和近似纤维化，为神经网络压缩提供了统一的理论基础。 |
| [^207] | [Activation-Weighted Seeded Residual Coding for Low-Bit LLM Weight Repair](https://arxiv.org/abs/2608.23144) | 本文提出激活加权种子残差编码（AWSRC），通过利用激活统计和种子生成基，以极小辅助存储（约0.8%权重负载）高效修复低比特量化误差，显著提升模型质量。 |
| [^208] | [Random Hazard Forests](https://arxiv.org/abs/2608.21597) | 随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。 |
| [^209] | [Algorithms for adaptive and heteroskedastic linear regression at the computational threshold](https://arxiv.org/abs/2608.18402) | 本文提出了一种在异方差线性回归中更高效的多项式时间估计器，能在样本质量不均时以更少的优质样本达到低误差，显著优于传统方法。 |
| [^210] | [LaGSplat: Inferring Physics-Governed Interactive Simulation from Monocular Video Using Latent Lagrangian Gaussian Splatting](https://arxiv.org/abs/2608.16324) | 本文提出LaGSplat框架，通过将低维潜在状态同时用作拉格朗日广义坐标和高斯泼溅解码器的条件变量，实现了从单目视频推断物理动态，并允许用户在推理时施加未见过外力进行交互模拟。 |
| [^211] | [Task- and dataset-specific information in protein language models](https://arxiv.org/abs/2608.12090) | 该研究通过分析13个蛋白质语言模型在15个下游任务中的表现，发现模型最后一层的嵌入并非总是最优，中间层可能包含更有效的任务特定信息。 |
| [^212] | [HoloAegis: Frozen Representation, Topological Inference --- Minimally Parametric Safety Manifolds and Their Capability Boundaries for LLM Guardrails](https://arxiv.org/abs/2608.08485) | HoloAegis提出一个仅3.2 MB的最小参数化拓扑推断安全框架，通过对冻结表示的纯几何推理实现大语言模型安全护栏，在毒性检测上匹敌、在有害行为检测上超越14 GB的大模型，同时系统性地刻画了这类纯几何方法的能力边界（如在过度安全检测上的不足）。 |
| [^213] | [Protecting patient privacy in clinical foundation models: Technical and legal perspectives](https://arxiv.org/abs/2608.07705) | 本文提出了一个评估临床基础模型隐私风险的实用框架，通过展示模型介导的隐私泄露场景并将其映射到HIPAA、GDPR等法律制度，从技术与法律两方面提出了互补的缓解措施。 |
| [^214] | [Neural Operator Learning for Collision-Aware Trajectory Planning of Spacecraft Swarms](https://arxiv.org/abs/2608.00320) | 提出一种自监督的置换等变神经算子，可将航天器集群与障碍物的状态分布直接映射为避碰且省燃料的轨迹，并结合批量高斯-牛顿精化保证轨道动力学精确性，仅需十个航天器训练即可零样本泛化至一千个航天器的集群。 |
| [^215] | [CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference](https://arxiv.org/abs/2607.22511) | CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。 |
| [^216] | [Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents](https://arxiv.org/abs/2607.19837) | 提出了KYA框架，通过形式化智能体侦察过程、自动化构建AI智能体的目标画像，来发现其弱点并生成更强的间接提示注入攻击。 |
| [^217] | [The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory](https://arxiv.org/abs/2607.19390) | 正交化读取对mLSTM循环记忆而言只是一个可移除的“训练脚手架”——它通过改善平台期的条件数将逃逸几率提高约六倍并拓宽可用学习率范围，其收益依赖读取与梯度的自洽性，且逃逸平台期后即可移除。 |
| [^218] | [Data-Driven Soft Labeling Scales DNA Read Classification to Whole-Body Cell-Type Deconvolution](https://arxiv.org/abs/2607.04987) | 该论文提出一种数据驱动的软标签方法，能够将DNA读段分类规模化扩展至全身细胞类型反卷积任务。 |
| [^219] | [Missing Data Imputation under Manifold Hypothesis](https://arxiv.org/abs/2607.03641) | 本文基于流形假设与混合变分自编码器，提出了一种通过SIR采样和潜空间扩散模型从条件分布中采样的缺失数据插补方法，在尊重数据几何结构的同时实现高质量插补并量化不确定性。 |
| [^220] | [The Scissors Effect: When Resize-Based Input Diversity Helps or Hurts Transfer Attacks](https://arxiv.org/abs/2606.22516) | 本文发现输入多样性（DI）技术对迁移攻击的作用取决于替代模型类型——它能提升从标准模型的迁移性，却会损害从对抗训练的鲁棒模型的迁移性（在ImageNet上损失10.3个百分点的攻击成功率），并通过偏差-方差分析解释了这一“剪刀效应”。 |
| [^221] | [Amortized Probabilistic Retrieval of Atmospheric CO2 from OCO-2 Spectra Using Deep Learning with Laplace Approximations and Normalizing Flows](https://arxiv.org/abs/2606.17413) | 该论文提出一个结合拉普拉斯近似与条件归一化流的深度学习框架，通过摊销式概率推断从OCO-2卫星光谱中实现毫秒级、摆脱高斯后验假设的大气CO2柱浓度概率反演。 |
| [^222] | [Tail-Shape Estimation in LLM Evaluation Is Fragile: A Protocol for Diagnosing False Positives](https://arxiv.org/abs/2606.16511) | 本文的贡献是一个预先注册的诊断协议，用于检验LLM评估中尾部形状声明的有效性，实证表明该协议能捕获标准毒性评估中朴素分析本会发表的三种假阳性模式，证明尾部形状估计在LLM评估中是脆弱的。 |
| [^223] | [Learning aligned EEG representations with subject-specific encoders](https://arxiv.org/abs/2606.16462) | 提出用被试特定编码器加共享分类器的混合架构替代共享EEG编码器，能够学习到被试对齐的表征，显著降低对欧几里得对齐预处理的依赖，并在多个运动想象与运动执行数据集上持续优于共享编码器基线。 |
| [^224] | [Shielded Analysis: Certification and Characterization of Defensibility in Systems under Adversarial Interaction](https://arxiv.org/abs/2606.13621) | 提出“屏蔽分析”这一设计时框架，能从单一编码系统中同时生成形式化可防御性证书和涵盖结构裕度、屏蔽自由度、自适应运行质量的四轴可防御性指纹，用以认证并刻画系统在对抗交互下的防御能力。 |
| [^225] | [Do LLMs Make Neural Distinguishers Wise?](https://arxiv.org/abs/2606.10692) | 本文首次将大语言模型应用于神经区分器，通过对SPECK-32/64的实验证明大语言模型并不能提升神经区分器的性能，且在高轮数下差分选择的有效性也随之消失。 |
| [^226] | [Deep-learning-based low-energy trigger algorithms for the Hyper-Kamiokande experiment](https://arxiv.org/abs/2605.31391) | 该研究为超级神冈实验开发了基于深度学习的低能中微子触发算法，其中监督分类器对3 MeV单电子的信号识别效率达76.7%，显著超越传统击中计数触发器的26.4%。 |
| [^227] | [BASIS: Batchwise Advantage Estimation from Single-Rollout Information Sharing for LLM Reasoning](https://arxiv.org/abs/2605.27293) | BASIS通过在批次内共享单rollout的跨提示信息来改进价值函数估计，以显著更低的计算成本实现了接近多rollout方法的策略优化性能。 |
| [^228] | [Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT](https://arxiv.org/abs/2605.25924) | 本研究发现在EFCAMDAT学习者语料库上进行领域自适应继续预训练对英语水平考试自动作文评分的提升效果好坏参半，根源在于该语料库与考试数据集在语言水平、体裁和交际目的上存在不匹配。 |
| [^229] | [HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos](https://arxiv.org/abs/2605.24934) | HumanEgo通过将人类第一视角视频提升为手-物交互的实体级表示并训练带密集辅助目标的流匹配策略，仅需每个任务30分钟的人类视频即可实现零样本的人到机器人技能迁移，在真实任务中达到92.5%的成功率并超越同等时间的遥操作训练41%。 |
| [^230] | [CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent Conformal Risk Control](https://arxiv.org/abs/2605.24696) | 本文提出CALIBURN，通过将运行约束（如告警预算、成本）直接嵌入阈值选择流程，而非依赖标签调优，实现了可操作化的流式入侵检测告警系统。 |
| [^231] | [Benchmarking Machine Learning Architectures for Antimicrobial Stewardship in Pediatric ICUs](https://arxiv.org/abs/2605.22611) | 该论文首次在儿科重症监护病房（PICU）中，基于公开和私有儿童队列数据，对表格型、序列型和图型机器学习架构在抗菌药物管理干预预测（涵盖静脉转口服、降阶梯、停药和短程治疗四个临床目标）上进行了系统性基准测试。 |
| [^232] | [Subject-Specific Analysis of Self-Initiated Attention Shifts from EEG with Controlled Internal and External Attention Conditions](https://arxiv.org/abs/2605.18251) | 本研究通过受控实验范式与机器学习方法，证明预备性脑电活动能够区分自发性注意转移与外部指导的注意转移，并通过特征归因揭示了多维EEG特征的贡献。 |
| [^233] | [$\mathcal{O}(n)$ alternative to Quantum Fourier Transform with efficient neural net classical post-processing](https://arxiv.org/abs/2605.16998) | 本文提出了一族由Hadamard门和控制相位门构成、保持平移不变性且费舍尔信息指数增长的浅电路（HP-$L$电路），其中深度仅为 $\mathcal{O}(n)$ 的 HP-$1$ 电路可在Shor算法中替代深度为 $\mathcal{O}(n^2)$ 的量子傅里叶变换。 |
| [^234] | [Structure-Aware Masking for Protein Representation Learning](https://arxiv.org/abs/2605.16581) | 该论文提出了名为Bucket Masking的结构感知掩码策略，通过基于三维空间邻近性优先掩码结构耦合的残基区域，促使蛋白质语言模型的学习重点转向对蛋白质功能至关重要的长程残基相互作用建模。 |
| [^235] | [GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms](https://arxiv.org/abs/2605.11117) | GRAFT-ATHENA框架将隐式的问题-方法关系显式化为可扩展的概率结构，使AI智能体团队的经验能够在结构相关的问题间迁移复用，在物理信息学习、血液流变学建模和高超声速流动求解等任务中达到或超越人类专家水平。 |
| [^236] | [Universal Feature Selection with Noisy Observations and Weak Symmetry Conditions](https://arxiv.org/abs/2605.09396) | 本文提出在弱球对称性条件下基于噪声数据的通用特征选择框架，通过典型相关依存矩阵的奇异值分解实现渐近最优误差指数，并证明精确的球对称性条件并非必要。 |
| [^237] | [RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis](https://arxiv.org/abs/2605.05616) | RAM-H1200是一个包含来自六个医疗中心的1,200张手部X光片的多层次标注数据集，为类风湿关节炎评估提供了从解剖结构分割、骨侵蚀定量分析到临床SvdH评分的统一基准。 |
| [^238] | [Perturbation Sensitivity of Maximum-Likelihood Pairwise Ranking in Computational Decision Systems](https://arxiv.org/abs/2604.17805) | 本文将极大似然成对排序的协同扰动形式化为预算约束子集选择问题，提出自适应子集选择攻击（ASSA）这一可扩展搜索方法，并通过实验证明该排序方法对较小但协同的扰动会表现出显著的、依赖运行状态的敏感性。 |
| [^239] | [EviDep: Uncertainty-Aware Multimodal Depression Estimation via Disentangled Evidential Learning](https://arxiv.org/abs/2604.16579) | EviDep提出了一种多模态证据回归框架，通过频率感知特征提取、共享-私有表示解耦学习以及基于正态-逆伽马分布的多分支证据回归，实现了不确定性感知的抑郁严重程度估计。 |
| [^240] | [LLM-Guided Dynamic Action Spaces for Synthesizable Molecular Optimization](https://arxiv.org/abs/2604.07669) | MolReAct利用工具增强的大语言模型在每一步动态生成针对特定分子的紧凑反应空间，从而在保证合成可行性的前提下使多步分子优化变得可行。 |
| [^241] | [CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference](https://arxiv.org/abs/2604.06036) | CodecSight利用视频编解码器元数据（变化信号与帧类型）作为视觉编码和LLM预填充的共享运行时指导，无需模型特定训练或离线剖析即可显著降低流式VLM推理的计算开销。 |
| [^242] | [A Spectral Decomposition Framework for Multiscale Nonlinear Dimensionality Reduction](https://arxiv.org/abs/2604.02535) | 提出SDMP框架，通过将嵌入维度表示为拉普拉斯特征向量的显式加权组合，实现兼顾局部细节与全局结构的多尺度非线性降维，同时增强分析透明度。 |
| [^243] | [Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization](https://arxiv.org/abs/2603.21676) | 本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。 |
| [^244] | [Deep Invertible Autoencoders for Dimensionality Reduction of Dynamical Systems](https://arxiv.org/abs/2603.13496) | 该论文提出深度可逆自编码器用于动力系统的降维，克服了POD在对流主导问题中奇异值衰减缓慢的缺陷，同时实现了比传统自编码器更强的降维能力。 |
| [^245] | [Equivalence of approximation by networks of single- and multi-spike neurons](https://arxiv.org/abs/2603.13478) | 本文证明对于包括泄漏积分发放模型在内的一大类脉冲神经元模型，单脉冲网络与多脉冲网络在函数逼近能力上完全等价，两者之间仅需以神经元数量的线性倍数进行转换。 |
| [^246] | [Learning efficient representations of complex constraints for scalable optimization](https://arxiv.org/abs/2603.08283) | 提出了PolyFormer框架，通过学习复杂约束的紧凑多面体表示并转换为高效重构形式，实现了高达6,400倍的在线求解加速和99.87%的内存减少，同时保持极小的可行性与目标误差。 |
| [^247] | [Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat](https://arxiv.org/abs/2603.02227) | 论文揭示了“路由吸收”现象——模型表示会与稀疏注意力掩码共同适应，使得可学习门控相比随机门控几乎没有额外优势，且在硬掩码部署时会造成严重的性能退化。 |
| [^248] | [Strategic Advice in the Age of Personal AI](https://arxiv.org/abs/2603.02055) | 本文研究了顾问在面对建议可预测的个人AI随机咨询时应如何策略性地设计建议，发现顾问会最优地抵消个人AI的信号，其最小化损失随咨询概率呈驼峰形变化，且对个人AI的相对信任度越高，随机咨询导致的不可消除损失越大。 |
| [^249] | [Differential privacy representation geometry for medical image analysis](https://arxiv.org/abs/2603.01098) | 该论文提出DP-RGMI框架，将差分隐私对医学图像分析性能的影响分解为表示空间几何变化与任务头利用率损失两个因素，并基于四个胸部X光数据集逾59万张图像揭示了即使线性可分性基本保留时差分隐私仍会导致利用率差距的机理。 |
| [^250] | [Partial recovery of meter-scale surface weather](https://arxiv.org/abs/2602.23146) | 该研究将稀疏气象站、高分辨率地球观测与粗分辨率大气动力学数据相结合，在不解析大气动力学的情况下，以30米分辨率推断美国本土的温度、露点和风速，误差较最强基线降低11-28%，并恢复了近一半的局地温度空间变率。 |
| [^251] | [PRISM: Parallel Residual Iterative Sequence Model](https://arxiv.org/abs/2602.10796) | PRISM通过写入-遗忘解耦策略和两阶段代理架构，以可并行的方式逼近TTT的迭代表达能力，化解了Transformer表达力与线性模型效率之间的根本矛盾。 |
| [^252] | [Hybrid Feedback-Guided Optimal Learning for Wireless Interactive Panoramic Scene Delivery](https://arxiv.org/abs/2602.07273) | 本文提出一种混合反馈引导的最优学习方法，利用预测反馈可对全部候选场景部分回溯计算的全信息特性，结合部分可观测的传输反馈，优化无线交互式全景场景传输中视场覆盖部分的选择。 |
| [^253] | [Robustness as an Emergent Property of Task Performance](https://arxiv.org/abs/2602.03344) | 本文通过实证分析发现，鲁棒性并非模型的独立能力，而是任务性能的涌现属性——一旦模型掌握了某项任务，鲁棒性便会随任务性能的提升而自然涌现。 |
| [^254] | [Meta-Learning-Assisted Constraint Relaxation for Constrained Black-Box Optimization](https://arxiv.org/abs/2602.00532) | 本文提出MeCO，一种通过元学习训练双重深度Q网络控制器来学习自适应ε-约束松弛策略的约束黑盒优化方法，无需针对具体问题调参即可泛化到新的优化问题。 |
| [^255] | [Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching](https://arxiv.org/abs/2601.20332) | 该论文通过分析发现扩散语言模型推理具有显著的结构局部性，并提出无需重新训练的Window-Diffusion方法，利用窗口化标记剪枝与缓存机制来显著加速预训练扩散语言模型的推理。 |
| [^256] | [AllShowers: One model for all calorimeter showers](https://arxiv.org/abs/2601.11716) | AllShowers提出了一种统一的生成模型，利用基于Transformer架构的连续归一化流，仅需单一模型即可在宽能量范围内模拟电子、光子及带电和中性强子等多种粒子类型的量能器簇射，取代了传统上为每种粒子单独训练网络的方法。 |
| [^257] | [Collaborative Optimization of Multiclass Imbalanced Learning: Density-Aware and Region-Guided Boosting](https://arxiv.org/abs/2512.22478) | 该研究提出一种密度感知与区域引导的协同优化Boosting模型，通过抗噪声权重更新机制和动态采样策略实现多类别不平衡学习与模型训练的紧密协同，在40个公开数据集上显著提升性能。 |
| [^258] | [Nonnegative matrix factorizations and related compositional models: Equivalence, identifiability, and an application on the grain-size analysis of sediments](https://arxiv.org/abs/2512.22282) | 本文证明了来自社会科学、地质学和机器学习的五种模型（LBA、LCA、EMA、PLSA、NMF）在本质上等价，NMF的解唯一性定理可直接推广到其他四种模型，并将其应用于沉积物粒度分析。 |
| [^259] | [Shuttling Compiler for Trapped-Ion Quantum Computers Based on Fine-Tuned Large Language Models](https://arxiv.org/abs/2512.18021) | 本文创新性地提出基于微调大语言模型的离子阱量子计算机输运编译器，可自动生成输运调度并在部分情况下比传统启发式方法减少多达21%的操作次数，同时展现出对新型阱架构的泛化潜力。 |
| [^260] | [Formalized Hopfield Networks and Boltzmann Machines](https://arxiv.org/abs/2512.07766) | 本文在 Lean 4 中首次形式化了霍普菲尔德网络与玻尔兹曼机，证明了确定性网络的收敛性与赫布学习的正确性，并借助佩龙-弗罗贝尼乌斯定理的新形式化证明了随机网络的遍历性。 |
| [^261] | [Training Energy-Based Models with Non-MCMC Samplers and Efficient Temperature Estimation](https://arxiv.org/abs/2512.02323) | 该论文提出了朗之万模拟分岔（LSB）快速并行玻尔兹曼采样器、条件期望匹配（CEM）高效温度估计方法以及采样器自适应学习（SAL）框架，从而无需MCMC即可高效训练能量基模型。 |
| [^262] | [Dual Randomized Smoothing: Beyond Global Noise Variance](https://arxiv.org/abs/2512.01782) | 提出双重随机平滑框架，证明了只要噪声方差在每个输入周围局部恒定时RS仍然有效，并通过引入方差估计器实现输入依赖的噪声方差，从而突破了标准RS中全局噪声方差无法同时在小半径和大半径处取得优异性能的根本限制。 |
| [^263] | [GeoCrossBench: Cross-Band Generalization for Remote Sensing](https://arxiv.org/abs/2511.02831) | 本文提出GeoCrossBench基准和χViT基线模型，通过新的跨波段泛化评估协议解决遥感领域新旧卫星波段不一致的问题，降低支持新卫星所需的模型重训练成本。 |
| [^264] | [TARC: Time-Adaptive Robotic Control](https://arxiv.org/abs/2510.23176) | 提出 TARC 强化学习框架，让策略同时预测控制动作及其持续时间，实现控制频率的自适应调节，在高速遥控车、宇树 Go1 四足机器人等真实硬件以及视觉-语言-动作模型上，均能在保持任务性能的同时减少控制切换和推理开销。 |
| [^265] | [Risk-Calibrated Bayesian Streaming Intrusion Detection with SRE-Aligned Decisions](https://arxiv.org/abs/2510.09619) | 本文提出将贝叶斯在线变点检测（BOCPD）与SRE错误预算对齐的决策阈值相结合，通过在假阳性和假阴性预算下优化期望运营成本，实现能适应分布漂移的风险校准流式入侵检测方法。 |
| [^266] | [GraphIFE: Rethinking Graph Imbalance Node Classification via Invariant Learning](https://arxiv.org/abs/2509.23616) | 提出了基于图不变学习的GraphIFE框架，通过缓解合成节点的质量不一致问题来应对图数据中的类别不平衡，从而提升少数类别的节点分类性能。 |
| [^267] | [Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data](https://arxiv.org/abs/2509.09710) | 该研究提出一种基于开源人口普查和土地利用数据、利用大语言模型随机生成个人出行日记的新方法，并通过包含四项指标的新颖真实感评分和Jensen-Shannon散度验证其有效性。 |
| [^268] | [Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods, and Application to Suicide Risk Modeling](https://arxiv.org/abs/2508.17090) | 本文提出了一类新型神经随机微分方程，其解可被严格证明限制在指定的紧凑多面体状态空间内，克服了现有SDE模型违反定义域约束和数值不稳定的问题，并成功应用于自杀风险建模。 |
| [^269] | [Script Fragmentation and Format: What Drives the English-Bengali Performance Gap in Open LLMs?](https://arxiv.org/abs/2507.23248) | 该论文发布了8个翻译成孟加拉语的英文基准测试并评估10个开源LLM，揭示英语与孟加拉语性能差距部分源于子词分词器对孟加拉语字素簇的文字碎片化以及精确匹配评分所强制的答案格式，并证明其中一部分差距实际上是测量伪影而非模型真实能力的缺陷。 |
| [^270] | [Observational Multiplicity](https://arxiv.org/abs/2507.23136) | 该论文提出了“观测多重性”概念，指出当模型以二元观测标签训练来预测概率时会产生预测的任意性，并引入一种基于“遗憾”度量的通用方法来量化模型概率预测因训练标签不同而可能产生的变化程度。 |
| [^271] | [R3: Robust Rubric-Agnostic Reward Models](https://arxiv.org/abs/2505.13388) | R3是一种不受评分标准限制、具备泛化能力和可解释性的奖励建模框架，通过提供有理有据的评分实现更透明灵活的语言模型评估与人类偏好对齐。 |
| [^272] | [When majority rules, minority loses: bias amplification of gradient descent](https://arxiv.org/abs/2505.13122) | 该论文建立了多数-少数学习任务的形式化理论框架，证明了在人口和方差不平衡条件下标准梯度下降训练会放大偏差、产生忽视少数群体特征的刻板预测器，并给出了消除这种偏差所需额外训练量的理论下界。 |
| [^273] | [BenSParX: A Robust Explainable Machine Learning Framework for Parkinson's Disease Detection from Bengali Conversational Speech](https://arxiv.org/abs/2505.12192) | 该研究发布了首个用于帕金森病检测的孟加拉语会话语音数据集BenSParX，并构建了结合多样化声学特征、系统特征选择与SHAP可解释性分析的鲁棒机器学习框架，为资源受限环境下帕金森病的早期无创诊断提供了文化包容性的解决方案。 |
| [^274] | [Explainable Graph-theoretical Machine Learning with Application to Alzheimer's Disease Prediction](https://arxiv.org/abs/2503.16286) | 本文提出了一种可解释的图论机器学习框架XGML，通过构建个体大脑代谢图并识别最具预测性的子图，实现了基于FDG-PET数据对阿尔茨海默病多变量结果的个体化预测。 |
| [^275] | [CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking](https://arxiv.org/abs/2503.05794) | 本文提出了一种基于聚类的后门水印方法（CBW），首次解决了开放集说话人验证场景下的数据集所有权验证问题，克服了现有方法依赖封闭标签空间、无法应对第三方注册身份的局限。 |
| [^276] | [Attention is All You Need Until You Need Retention](https://arxiv.org/abs/2501.09166) | 该论文提出一种保留层，使Transformer具备可在使用中读写的持久记忆，并将“决定保留什么”建模为社会学习问题，通过惊讶门控编码、可信度加权的多源共识巩固以及基于再现性的再巩固来管理记忆生命周期，且该层在记忆为空时严格退化为标准Transformer。 |
| [^277] | [Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes.](http://arxiv.org/abs/2307.15931) | 本研究研究了在部分可观测马尔可夫决策过程中解决的动作序列的好处，并提出了几种扩展深度强化学习算法的结构和方法。 |

# 详细

[^1]: ENCP：面向视觉与语言导航的情节归一化保形预测

    ENCP: Episode-Normalized Conformal Prediction for Vision-and-Language Navigation

    [https://arxiv.org/abs/2609.17499](https://arxiv.org/abs/2609.17499)

    提出情节归一化保形预测（ENCP），通过以情节为单位对非一致性得分进行归一化校准，为视觉与语言导航的每一步提供至少1-α的覆盖保证，同时允许情节内步骤间存在依赖关系。

    

    对视觉与语言导航（VLN）模型进行不确定性估计是一项关键任务，因为它有助于识别模糊和不可靠的预测，使智能体能够做出更安全的导航决策。作为最先进的不确定性估计框架之一，保形预测（CP）为VLN中的不确定性估计提供了一种有前景的方法。然而，鉴于VLN智能体需要执行一系列步骤，保形预测中的标准校准方法无法为其承诺的覆盖保证，应用于相关的、可变长度的VLN情节时失效。为此，我们提出了情节归一化保形预测（ENCP），该方法通过策略的剩余置信度对非一致性得分进行重新缩放，并为每个情节校准一个最大得分。在校准情节与测试情节可交换的条件下，该构造能以至少 $1 - \alpha$ 的概率在每一步覆盖真实值，同时允许情节内各步骤之间存在依赖关系。

    arXiv:2609.17499v1 Announce Type: cross  Abstract: Uncertainty estimation for Vision-Language-Navigation (VLN) models is a critical task since it can help identify ambiguous and unreliable predictions, enabling agents to make safer navigation decisions. As one of the most advanced uncertainty estimation frameworks, conformal prediction (CP) offers a promising approach for uncertainty estimation in VLN. However, given that VLN agent requires a sequence of steps, standard calibration in conformal prediction fails to provide coverage guarantee it promises over a dependent, variable-length VLN episode. To this end, we propose Episode-Normalized Conformal Prediction (ENCP), which rescales a nonconformity score by the policy's residual confidence and calibrates one maximum score per episode. Under exchangeable calibration and test episodes, this construction covers the ground truth at every step with probability at least $1 - \alpha$, while allowing dependence among steps within an episode. 
    
[^2]: FreqSpaNet：基于SFPF的频率与空间学习用于物理层硬件完整性检测

    FreqSpaNet: Frequency and Spatial Learning of SFPF for Physical Layer Hardware Integrity Detection

    [https://arxiv.org/abs/2609.17491](https://arxiv.org/abs/2609.17491)

    提出FreqSpaNet网络，通过频率分支与几何感知空间分支分别学习空-频极化指纹（SFPF）的不同结构特性，结合自适应融合与互补预训练，实现开集硬件异常检测，以识别保持逻辑身份不变的未经授权硬件替换。

    

    未经授权的硬件替换可以在保持无线设备逻辑身份的同时改变其物理实现，这对硬件完整性验证构成了挑战。空-频极化指纹（SFPF）能够捕获设备在多个频率和方向上特有的响应，但其频率维度和空间维度呈现出不同的结构依赖性。我们提出了FreqSpaNet，一个用于开集硬件异常检测的SFPF表示学习网络。其中频率分支捕获相邻频率之间的局部变化，而几何感知空间分支利用角度信息建模方向关系。两种表示通过自适应融合相结合，互补预训练进一步捕获共享信息，同时保留频率表示和空间表示各自的独特特征。实验表明，FreqSpaNet实现了较高的平均AUROC

    arXiv:2609.17491v1 Announce Type: new  Abstract: Unauthorized hardware replacement can preserve a wireless device's logical identity while altering its physical implementation, posing a challenge to hardware integrity verification. Spatio-frequency polarization fingerprints (SFPFs) capture device-dependent responses across multiple frequencies and directions, but their frequency and spatial dimensions exhibit different structural dependencies. We propose FreqSpaNet, an SFPF representation learning network for open set hardware anomaly detection. A frequency branch captures local variations among neighboring frequencies, while a geometry-aware spatial branch models directional relationships using angular information. The two representations are combined through adaptive fusion, and complementary pretraining further captures shared information while preserving the distinct characteristics of the frequency and spatial representations. Experiments show that FreqSpaNet achieves a mean AUROC
    
[^3]: 弥合同构与异构异步优化之间的差距出人意料地困难

    Bridging the Gap Between Homogeneous and Heterogeneous Asynchronous Optimization Is Surprisingly Difficult

    [https://arxiv.org/abs/2609.17483](https://arxiv.org/abs/2609.17483)

    本文证明在广泛使用的一阶和二阶相似性假设下，任何随机算法都无法改进异构异步优化的悲观最优时间复杂度，表明弥合同构与异构设置之间的差距出人意料地困难。

    

    现代大规模机器学习任务通常需要多个工作节点、设备、CPU或GPU并行且异步地计算随机梯度来训练模型权重。理论结果通常区分两种设置：(i) 同构设置，即所有工作节点都可以访问相同的数据分布；(ii) 异构设置，即每个工作节点在不同的数据分布上运行。这些设置中已知的最优时间复杂度显示出显著差距，异构情况下的理论保证要悲观得多。在这项工作中，我们研究了在不同的假设下是否可以克服这些悲观的最优时间复杂度。令人惊讶的是，我们证明了对于任何随机算法，在广泛使用的一阶和二阶相似性假设下，改进被证明是不可能的。然后我们转向插值机制，并证明弱插值假设（摘要在此处被截断）

    arXiv:2609.17483v1 Announce Type: cross  Abstract: Modern large-scale machine learning tasks often require multiple workers, devices, CPUs, or GPUs to compute stochastic gradients in parallel and asynchronously to train model weights. Theoretical results typically distinguish between two settings: (i) the homogeneous setting, where all workers have access to the same data distribution, and (ii) the heterogeneous setting, where each worker operates on different data distributions. Known optimal time complexities in these settings reveal a significant gap, with far more pessimistic guarantees in the heterogeneous case. In this work, we investigate whether these pessimistic optimal time complexities can be overcome under different assumptions. Surprisingly, we show that improvement is provably impossible under widely used first- and second-order similarity assumptions for any randomized algorithm. We then turn to the interpolation regime and demonstrate that the weak interpolation assumpt
    
[^4]: 稠密联想记忆绝对容量中由偏置引起的渡越行为

    Bias-Induced Crossover in Absolute Capacity of Dense Associative Memory

    [https://arxiv.org/abs/2609.17477](https://arxiv.org/abs/2609.17477)

    本文首次系统分析了中心化二元模式中偏置对稠密联想记忆绝对容量的影响，发现容量随相互作用阶数 $n$ 的奇偶性呈现不同的渐近标度律，并通过渐近匹配方法预测了在偏置趋于零时由偏置诱导的容量渡越现象。

    

    稠密联想记忆的绝对容量此前主要针对无偏模式进行了分析。本文在Krotov-Hopfield单位点判据 $P_{\mathrm{error}}=1/N$ 下考察了中心化二元模式中偏置的影响，其中 $P_{\mathrm{error}}$ 是单位点翻转降低已存储模式能量的概率，$N$ 为神经元数量。每个模式分量以概率 $q$ 取 $1-q$，否则取 $-q$。研究发现：对于偶数 $n\ge4$，容量为 $O(N^{n/2})$；对于奇数 $n\ge5$，容量为 $O(N^{(n+1)/2})$。当 $n=3$ 时，无论无偏还是固定偏置，容量均保持为 $O(N^2/\ln N)$。对于 $n\ge4$，这些不同的渐近形式意味着在 $q=1/2$ 附近存在非一致的大 $N$ 极限。渐近匹配预测在区域 $1-2q=O(\ln N/N^{\lfloor n/2\rfloor-1})$ 中存在由偏置引起的渡越行为。该渡越源于依赖于偏置的串扰均值，它会降低携带该值的位点的稳定性。（原文摘要在此处截断）

    arXiv:2609.17477v1 Announce Type: cross  Abstract: The absolute capacity of dense associative memory has mainly been analyzed for unbiased patterns. Here we examine the effect of bias in centered binary patterns under the Krotov-Hopfield single-site criterion $P_{\mathrm{error}}=1/N$, where $P_{\mathrm{error}}$ is the probability that a single-site flip lowers the energy of a stored pattern and $N$ is the number of neurons. Each pattern component takes $1-q$ with probability $q$ and $-q$ otherwise, where $0<1/2$, however, the capacity is $O(N^{n/2})$ for even $n\ge4$ and $O(N^{(n+1)/2})$ for odd $n\ge5$. For $n=3$, both the unbiased and fixed-bias capacities remain $O(N^2/\ln N)$. For $n\ge4$, these different asymptotic forms imply a nonuniform large-$N$ limit near $q=1/2$. Asymptotic matching predicts a bias-induced crossover in the region $1-2q=O(\ln N/N^{\lfloor n/2\rfloor-1})$. The crossover originates from a bias-dependent crosstalk mean that reduces the stability of sites carryin
    
[^5]: 耦合校准与学习：在无目标域奖励反馈的LLM蒸馏中缓解教师偏差

    Coupled Calibration and Learning: Mitigating Teacher Bias in LLM Distillation without Target-Domain Reward Feedback

    [https://arxiv.org/abs/2609.17474](https://arxiv.org/abs/2609.17474)

    提出耦合校准与学习（CCL）算法，通过token级分支将教师校准与学生更新相互耦合、仅利用源域奖励反馈，从而在无目标域奖励反馈的条件下缓解LLM蒸馏中的教师偏差迁移问题。

    

    大语言模型（LLM）蒸馏旨在将强大教师模型的能力迁移到更小的学生模型中。然而，直接模仿也会将教师的系统性偏差和错误一同迁移。这一挑战在协变量偏移情况下尤为突出，即教师对目标问题的可靠性不确定且目标域奖励反馈不可用时。我们提出了耦合校准与学习（CCL），这是一种LLM蒸馏算法，通过token级分支将教师校准与学生更新耦合起来，仅在源问题上使用奖励反馈。每次迭代中，先利用源域反馈校准教师，再用校准后的教师在目标问题上训练学生；更新后的学生反过来又为后续的校准提供信息。在自回归策略框架下，我们证明了输出的学生模型相对于oracle学生模型的期望平均Kullback-Leibler散度收敛

    arXiv:2609.17474v1 Announce Type: cross  Abstract: Large language model (LLM) distillation aims to transfer the capabilities of a powerful teacher to a smaller student. Direct imitation, however, can also transfer the teacher's systematic bias and errors. This challenge is particularly pronounced under covariate shift, when the teacher's reliability on target questions is uncertain and target-domain reward feedback is unavailable. We propose Coupled Calibration and Learning (CCL), an LLM distillation algorithm that couples teacher calibration with student updates through token-level branching, using reward feedback only on source questions. Each iteration calibrates the teacher using source feedback and then uses the calibrated teacher to train the student on target questions. The updated student, in turn, informs subsequent calibration. In an autoregressive policy framework, we prove that the output student's expected average Kullback-Leibler divergence to the oracle student converges
    
[^6]: 表格解码：DELTA 负责结构，TARQA 负责理解

    Tables Decoded: DELTA for Structure, TARQA for Understanding

    [https://arxiv.org/abs/2609.17458](https://arxiv.org/abs/2609.17458)

    本文提出基于结构化文本表示的 DELTA 与 TARQA 方法来处理表格结构识别和表格视觉问答任务，其中 DELTA 将物理结构识别、逻辑结构识别与 OCR 分离，并以紧凑统一的 OTSL 格式输出表格，相比视觉-语言模型更具可扩展性且适用于多语言文档。

    

    表格理解是文档智能中的一项核心任务，包含两个关键子任务：表格重建和表格视觉问答。虽然近期的方法主要依赖于在表格图像上运行的视觉-语言模型，但我们提出了一种基于结构化文本表示的更具可扩展性且更有效的替代方案。这些表示更容易处理，与大型语言模型（LLM）能更自然地对齐，并且无需特定语言的视觉编码器，因此特别适用于多语言文档。我们提出了 DELTA，它将物理结构识别、逻辑结构识别和 OCR 分离开来，以准确提取布局和内容。DELTA 以优化表格结构语言（OTSL）格式输出表格，这是一种紧凑且统一的格式，可对单元格排列和文本内容进行编码。在表格结构识别（TSR）任务上，DELTA 实现了与……相当的 TEDS-Structure 分数。

    arXiv:2609.17458v1 Announce Type: cross  Abstract: Table understanding is a core task in document intelligence, encompassing two key subtasks: table reconstruction and table visual question answering (TabVQA). While recent approaches predominantly rely on vision- language models (VLMs) operating on table images, we propose a more scalable and effective alternative based on structured textual representations. These representations are easier to process, align more naturally with LLMs, and eliminate the need for language-specific visual encoders, making them particularly suitable for multilingual documents. We present DELTA, which separates physical structure recognition, logical structure recognition, and OCR to extract both layout and content accurately. DELTA outputs tables in Optimised Table Structure Language (OTSL), a compact and unified format that encodes cell arrangements and textual content. On table structure recognition (TSR), DELTA achieves TEDS- Structure scores comparable 
    
[^7]: 过程仿真模型的降空间多保真度贝叶斯优化

    Reduced-Space Multi-Fidelity Bayesian Optimization of Process Simulation Models

    [https://arxiv.org/abs/2609.17440](https://arxiv.org/abs/2609.17440)

    提出了一种将全局敏感性分析降维与保真度增强高斯过程相结合的降空间多保真度贝叶斯优化框架，能够在高维昂贵的工业过程仿真中实现成本感知的高效优化。

    

    由于严格模拟的高昂计算成本以及复杂设计空间中固有的维数灾难，优化工业过程流程图在计算上往往令人望而却步。为了应对这些挑战，我们提出了一种面向高维、昂贵的黑箱函数的降空间多保真度贝叶斯优化（RS-MFBO）框架。该方法将用于降维的全局敏感性分析（GSA）与保真度增强的高斯过程相结合，后者能够捕捉低成本近似与昂贵的高保真度评估之间的相关性。一种成本感知的采集策略，并通过冷却和晋升机制加以增强，自适应地指导样本在不同保真度之间的分配。该框架在两个不同的工业过程模拟器上进行了验证：SuperPro Designer中的质粒DNA生物过程，以及Aspen HYSYS中的绿色燃料合成工厂。跨多种经济性和物理性指标的实验结果……

    arXiv:2609.17440v1 Announce Type: new  Abstract: Optimizing industrial process flowsheets is often computationally prohibitive due to the high cost of rigorous simulations and the curse of dimensionality inherent in complex design spaces. To address these challenges, we present a reduced-space multi-fidelity Bayesian optimization (RS-MFBO) framework designed for high-dimensional, expensive black-box functions. The approach integrates Global Sensitivity Analysis (GSA) for dimensionality reduction with a fidelity-augmented Gaussian process that captures correlations between low-cost approximations and expensive high-fidelity evaluations. A cost-aware acquisition strategy, augmented with cooldown and promotion mechanisms, adaptively guides the allocation of samples across fidelities. The framework is validated on two distinct industrial process simulators: a plasmid DNA bioprocess in SuperPro Designer and a green fuel synthesis plant in Aspen HYSYS. Results across diverse economic and phy
    
[^8]: 大型动态动作空间中的学习引导规划：面向一对多移动充电的预算化树搜索

    Learning-Guided Planning in Large Dynamic Action Spaces: Budgeted Tree Search for One-to-Many Mobile Charging

    [https://arxiv.org/abs/2609.17429](https://arxiv.org/abs/2609.17429)

    提出了LP-BTS学习引导规划架构，通过图提案策略、学习价值评论家和边预算化PUCT树搜索相结合，解决了候选动作随状态动态重建的一对多移动充电问题，并凭借无固定输出头的设计用单个冻结模型覆盖从736到2,813个候选动作的多种大规模动态动作空间。

    

    许多学习型序列决策系统将当前状态直接映射为动作。当候选动作数量众多、具有几何结构且随状态不断重建时，这种捷径会变得脆弱。一对多移动充电使这一设定变得具体：当传感器数量N=250时，初始状态会产生约1,125个候选充电停靠动作；每个被选中的停靠点会同时为其覆盖范围内的传感器服务，并且随着传感器电耗尽，动作空间也随之变化。LP-BTS是一种学习引导的规划架构：图提案策略集中生成一个小的候选支持集，学习到的价值评论家评估叶节点，边预算化的PUCT在提交动作之前比较短期的模拟未来。由于策略在不依赖固定输出头的情况下对该集合进行评分，单个冻结的模型检查点即可覆盖所有评估设置，动作空间范围从736到2,813个停靠点。匹配的消融实验揭示了各组件的互补效应：均匀采样会带来（摘要在此处截断）

    arXiv:2609.17429v1 Announce Type: cross  Abstract: Many learned sequential decision systems map the current state directly to an action. That shortcut becomes brittle when candidate actions are numerous, geometrically structured, and rebuilt with the state. One-to-many mobile charging makes this setting concrete: with N=250 sensors, the initial state induces about 1,125 candidate charging-stop actions; each chosen stop simultaneously serves its in-range sensors, and the action universe changes as sensors die. LP-BTS is a learning-guided planning architecture: a graph proposal policy concentrates a small candidate support, a learned value critic evaluates leaves, and edge-budgeted PUCT compares short simulated futures before committing an action. Because the policy scores this set without a fixed output head, a single frozen checkpoint covers every evaluated setting, spanning action universes from 736 to 2,813 stops. Matched ablations reveal complementary effects: uniform sampling costs
    
[^9]: 知识即轨道：有限集合作为精确周期性潜空间生成器的相位

    Knowledge as Orbit: Finite Collections as Phases of an Exactly Periodic Latent Generator

    [https://arxiv.org/abs/2609.17417](https://arxiv.org/abs/2609.17417)

    提出以内涵方式存储有限知识集合：将每个条目表示为基于实离散傅里叶旋转算子的精确周期性潜空间生成器的相位，从数学上保证精确闭合，并实验证明精确周期算子在循环重建上显著优于一般学习算子与非周期保范算子。

    

    有限知识通常以外延方式存储，每个条目对应一个编码或向量。我们探究一个有限集合是否可以改为以内涵方式存储，即作为一条能精确返回起点的紧凑规律的解码轨道。对于X个对象，我们将条目i编码为在学习的潜空间中一次固定旋转的第i个相位，并用共享网络解码所有相位；潜空间在循环的整数谐波处通过一组旋转进行推进——这是一个实的离散傅里叶算子——使得R^X等于恒等映射，从而精确闭合是被数学保证的而非学习得到的。图像是一个受控载体；循环视频则是相位顺序即内容自身时间结构的情形。在固定解码器、仅改变算子的实验中，一般的学习算子会发散，保持范数但非周期的算子在循环处出现退化，而精确周期算子则保持平坦；在真实图像上这一差距进一步扩大。容量随后是解码……

    arXiv:2609.17417v1 Announce Type: new  Abstract: Finite knowledge is usually stored extensionally, one code or vector per item. We ask whether a finite collection can instead be stored intensionally, as the decoded orbit of one compact law that returns exactly to its start. For X objects, we encode item i as the i-th phase of a fixed rotation in a learned latent space and decode all phases with a shared network; the latent advances through a bank of rotations at integer harmonics of the cycle, a real discrete Fourier operator, so that R^X equals the identity and exact closure is guaranteed rather than learned. Images are a controlled carrier; looping video is the case where the phase order is the content's own temporal structure. Holding the decoder fixed and varying only the operator, a general learned operator diverges, a norm-preserving but non-periodic one degrades around the loop, and the exactly periodic operator is flat; on real images the gap widens. Capacity is then the decode
    
[^10]: 弥合置信度差距：用于校准测试时提示调优的温度缩放方法

    Bridging the Confidence Gap: Temperature Scaling for Calibrating Test-Time Prompt Tuning

    [https://arxiv.org/abs/2609.17386](https://arxiv.org/abs/2609.17386)

    提出CoTS后校准方法，通过温度缩放最小化测试时提示调优中适应预测与零样本预测之间的置信度差距，在提升校准性能的同时保持准确率，并结合弱-强集成策略（E-CoTS）进一步增强效果。

    

    测试时提示调优（TPT）能够在单个测试实例上进行自适应调整，虽然提升了准确率，但往往会牺牲校准性能。大多数现有的校准方法通过引入额外的正则化项来促进文本嵌入的分散性并降低校准误差，然而这些方法通常会导致准确率下降。受零样本预测具有良好校准特性的启发，我们提出了CoTS，一种简单而有效的后校准方法，能够在保持准确率的同时改善校准。具体而言，CoTS通过应用温度缩放来最小化适应后预测与零样本预测之间的置信度差距。为了充分利用适应过程中多次数据增强的潜力，我们引入了弱-强集成策略，进一步提升准确率。随后我们将CoTS应用于该集成策略，称为E-CoTS，以保持其良好的校准特性。在多样化数据集和骨干网络上的大量实验表明，我们的方法……

    arXiv:2609.17386v1 Announce Type: new  Abstract: Test-time prompt tuning (TPT) enables adaptation on a single test instance, achieving improved accuracy but often sacrificing calibration performance. Most existing calibration methods introduce additional regularization terms to promote dispersion across text embeddings and reduce calibration error, yet these methods often suffer from a drop in accuracy. Motivated by the well-calibrated nature of zero-shot predictions, we propose CoTS, a simple yet effective post-hoc calibration method that preserves accuracy. Specifically, CoTS applies temperature scaling to minimize the confidence gap between adapted and zero-shot predictions. To fully exploit the potential of multiple augmentations during adaptation, we introduce a weak-strong ensemble strategy that further boosts accuracy. We then apply CoTS to this ensemble, termed E-CoTS, to maintain its well-calibrated property. Extensive experiments on diverse datasets and backbones show that ou
    
[^11]: OPEN-1B：一次完全可审计的训练运行

    OPEN-1B: A Fully Auditable Training Run

    [https://arxiv.org/abs/2609.17380](https://arxiv.org/abs/2609.17380)

    本文提出“完全可审计”这一全新的模型透明度级别，使训练中对每个数据样本的每次操作都能在异构通用硬件上以位级确定性独立复现，从而解决开源语言模型无法被证明可复现的问题。

    

    开源语言模型存在可复现性问题。尽管开源模型发布了权重、训练数据和训练配方，但由于浮点运算的非结合性，它们都无法被证明是可复现的。深度学习框架通常提供确定性执行模式，允许在同一台机器上进行可复现的操作。遗憾的是，这种确定性无法跨越不同硬件，因此用户无法验证发布的模型检查点确实是使用声明的训练配方生成的。这就为未公开的数据、注入的偏见或后门留下了空间，而现有的技术（如学习证明或训练数据证明）无法排除这些可能性。我们引入了一种新的模型透明度级别——完全可审计，即训练期间对每个数据样本的每次操作都可以在异构的通用硬件上以位级确定性独立复现。通过对训练数据源施加确定的顺序……

    arXiv:2609.17380v1 Announce Type: new  Abstract: Open-source language models have a reproducibility problem. Despite releasing weights, training data, and recipes, none of them are provably reproducible due to the non-associativity of floating-point arithmetic. Deep learning frameworks often offer a deterministic execution mode, allowing reproducible operations on the same machines. Unfortunately, this determinism does not carry across hardware such that a user can verify that a released checkpoint was actually produced using the declared training recipe. This leaves room for undisclosed data, injected biases, or backdoors that existing techniques such as proof-of-learning or proof-of-training-data cannot rule out.   We introduce a new tier of model transparency, fully auditable, in which every operation on every data sample during training is independently reproducible on heterogeneous commodity hardware with bitwise certainty. By imposing a definite order on the sources of training n
    
[^12]: 大语言模型在上下文中发展出信念状态几何结构

    Large Language Models Develop Belief State Geometry In-Context

    [https://arxiv.org/abs/2609.17376](https://arxiv.org/abs/2609.17376)

    研究发现大语言模型在上下文学习隐马尔可夫模型数据时，会在其内部残差流激活中以线性可解码的方式编码出信念状态的几何表示，且通过干预实验证明该表示与模型的预测行为功能相关。

    

    经过下一词元预测训练的大语言模型（LLM）展现出卓越的上下文学习能力（ICL），然而支持这种学习的内部表示仍鲜为人知。我们在一个受控环境中研究这些表示：使用隐马尔可夫模型（HMM）生成的数据来提示大语言模型，并探测相应的信念状态——即在给定已观测词元历史的条件下，HMM隐状态的后验分布。通过对六个开源大语言模型使用40个被挑选出具有非平凡信念结构的HMM数据进行提示，我们发现信念状态可以从残差流激活中线性解码，在各种HMM与大语言模型的组合上，探测器的峰值R²值达到0.83-0.99，且涵盖了从早期到后期的各层。为确立该表示的功能相关性，我们通过补丁修复（patching）和导向（steering）对探针识别出的子空间进行直接干预，结果使下游预测质量达到与未干预模型相当的水平，同时……

    arXiv:2609.17376v1 Announce Type: cross  Abstract: Large language models (LLMs) trained on next-token prediction exhibit remarkable in-context learning (ICL) abilities, yet the representations that support ICL remain poorly understood. We consider such representations in a controlled setting: prompting LLMs with data emitted from hidden Markov models (HMMs) and probing for the corresponding belief state -- the posterior distribution over the HMM's hidden states given the observed token history. Across six open-source LLMs prompted with data from 40 HMMs selected for non-trivial belief structure, we find that belief states are linearly decodable from residual stream activations, with peak probe $R^2$-values from 0.83-0.99 across HMM and LLM combinations, ranging from early to late layers. To establish functional relevance, we intervene directly on the probe-identified subspace via patching and steering, resulting in downstream prediction quality on the order of the untampered model, whi
    
[^13]: 用于多元回归与高维数据重构的混合变分量子线路

    Hybrid Variational Quantum Circuits for Multivariate Regression and High-Dimensional Data Reconstruction

    [https://arxiv.org/abs/2609.17358](https://arxiv.org/abs/2609.17358)

    该论文提出混合变分量子线路（HVQC），通过在量子线路后添加经典仿射层，实现了无需独立标量线路开销的向量值回归与高维数据重构，其性能媲美高斯过程回归并超越XGBoost和随机森林。

    

    变分量子线路（VQC）是一类通过经典方法优化的参数化量子线路。我们提出了一种混合变分量子线路（HVQC），在VQC基础上增加了一个经典的测量后仿射层，从而实现向量值回归，避免了使用多个独立标量线路所带来的线性开销。在理论方面，我们证明了基本的单量子比特和双量子比特线路可以通过数据重复上传和纠缠来近似二次函数和乘积运算，为完整架构奠定了基础。在实验方面，在两个合成图像重构数据集和Friedman1基准数据集（40,568个测试样本）上，我们的HVQC达到了与高斯过程回归相当的性能，并优于XGBoost和随机森林。消融实验证实了量子组件和经典组件都不可或缺，结果还凸显了特征映射在混合量子-经典模型中的核心作用。

    arXiv:2609.17358v1 Announce Type: new  Abstract: Variational quantum circuits (VQCs) are parameterized quantum circuits optimized classically. We propose a hybrid variational quantum circuit (HVQC) extending VQCs with a classical affine post-measurement layer, enabling vector-valued regression without the linear overhead of independent scalar circuits. Theoretically, we show that elementary one-and two-qubit circuits can approximate quadratic functions and products via data re-uploading and entanglement, providing the foundations of the full architecture. Experimentally, on two synthetic image reconstruction datasets and the Friedman1 benchmark (40,568 test samples), our HVQC matches Gaussian Process Regression and outperforms XGBoost and Random Forest. An ablation study confirms that both quantum and classical components are essential, and results highlight the central role of the feature map in hybrid quantum-classical models.
    
[^14]: 基于逐层非对比表示学习的Type-IV代码克隆检测

    Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning

    [https://arxiv.org/abs/2609.17338](https://arxiv.org/abs/2609.17338)

    本文提出LWVIC4Code，一种基于VICReg框架的非对比表示学习方法，通过跨层一致性正则化和深度相关的层加权机制逐层精炼代码语义表示，有效解决了语义等价但语法不同的Type-IV代码克隆检测难题，同时避免了对比学习中负采样带来的偏差。

    

    软件克隆是指彼此相似或功能等价的代码片段，它们给软件维护、重构和缺陷检测带来了重大挑战。检测Type-IV克隆（即语义等价但语法上可能不同的克隆）对于传统的基于词法或语法的方法来说尤为困难。近期的机器学习方法依赖于对比学习，这需要精心的负样本采样，并可能引入偏差。本文提出了LWVIC4Code，一种专为Type-IV克隆检测设计的非对比表示学习方法。LWVIC4Code基于方差-不变性-协方差正则化框架和先前的逐层VICReg训练，引入了跨层一致性正则化和与深度相关的层加权机制，以在transformer各层之间逐步精炼语义信息，从而生成鲁棒且具有区分性的代码表示。

    arXiv:2609.17338v1 Announce Type: cross  Abstract: Software clones are fragments of code that are similar or functionally equivalent to each other. They pose significant challenges for maintenance, refactoring, and bug detection. Detecting Type-IV clones, which are semantically equivalent but may differ syntactically, is particularly difficult for traditional token- or syntax-based methods. Recent machine learning approaches rely on contrastive learning, which requires careful negative sampling and can introduce bias. In this paper, we propose LWVIC4Code, a non-contrastive representation learning approach specifically designed for Type-IV clone detection. Building on the Variance-Invariance-Covariance Regularization (VICReg) framework and prior layer-wise VICReg training, LWVIC4Code introduces cross-layer consistency regularization and depth-dependent layer weighting to progressively refine semantic information across transformer layers, producing robust and discriminative code represe
    
[^15]: 面向图像修复的量子启发可训练且参数高效张量网络

    Quantum-Inspired Trainable and Parameter-Efficient Tensor Networks for Image Inpainting

    [https://arxiv.org/abs/2609.17298](https://arxiv.org/abs/2609.17298)

    本文提出量子启发的可训练张量网络变换用于图像修复，其中对角QFT松弛通过电路结构天然保持最小相干性，以极少的参数实现了超越固定变换并媲美大型酉架构的性能。

    

    本工作引入量子启发的张量网络电路作为图像修复的可训练变换。在所提出的架构中，对角量子傅里叶变换（QFT）松弛对于 N×N 图像具有可逆性，计算复杂度为 O(N² log N)，并通过其电路结构在训练过程中天然地保持最小相干性，从而无需显式的相干性惩罚项。无约束的基于梯度的相位优化（无需黎曼优化）能够从随机采样的训练数据中高效学习，使学习到的变换能够泛化到通过固定采样掩模观测的测试图像。数值实验表明，学习到的模型优于固定变换和逐图像优化方法，同时与规模大得多的酉架构性能相当，但所需参数却少得多。

    arXiv:2609.17298v1 Announce Type: cross  Abstract: This work introduces quantum-inspired tensor-network circuits as trainable transforms for image inpainting. Among the proposed architectures, the diagonal quantum Fourier transform (QFT) relaxation is invertible with $O(N^2 \log N)$ computational cost for $N\times N$ images, inherently preserving minimum coherence throughout training via its circuit structure and eliminating the need for explicit coherence penalties. Unconstrained gradient-based phase optimization (Riemannian-optimization free) enables efficient learning from randomly sampled training data, allowing the learned transform to generalize to test images observed through fixed sampling masks. Numerical tests show that the learned models outperform fixed transforms and per-image optimization while matching the performance of much larger unitary architectures, yet with far fewer parameters.
    
[^16]: 面向目标的概率预测用于5G网络中的动态物理资源块（PRB）分配

    Goal-oriented probabilistic forecasting for dynamic PRB allocation in 5G networks

    [https://arxiv.org/abs/2609.17297](https://arxiv.org/abs/2609.17297)

    该论文提出一种面向目标的概率预测框架，利用Pinball损失函数训练DeepAR和TFT模型，并根据运营商成本矩阵确定最优分配分位数，从而在5G网络动态PRB分配中降低运营成本，实现服务可靠性与资源效率的平衡。

    

    5G网络中高效的物理资源块（PRB）分配需要准确的需求预测。传统方法最小化对称误差指标（MAE、RMSE），忽略了运营成本的不对称性——即资源供给不足（导致服务降级）的代价远高于资源过度供给（造成容量浪费）。我们提出了一种面向目标的概率预测框架，使模型训练与运营商的决策目标保持一致。具体而言，我们使用Pinball损失函数训练DeepAR和时间融合Transformer（TFT）模型，并根据运营商的成本矩阵推导出最优分配分位数。在真实波束级5G流量数据集上的评估表明，与基于MSE训练的基线方法相比，所提出的方法在保持校准良好的不确定性估计的同时降低了运营成本。该框架能够实现动态PRB分配，明确平衡服务可靠性与资源效率。

    arXiv:2609.17297v1 Announce Type: cross  Abstract: Efficient physical resource block (PRB) allocation in 5G networks requires accurate demand forecasting. Conventional methods minimize symmetric error metrics (MAE, RMSE), ignoring the operational cost asymmetry where under-provisioning (service degradation) is far costlier than over-provisioning (wasted capacity). We propose a goal-oriented probabilistic forecasting framework that aligns model training with the operator's decision-making objectives. Specifically, we train DeepAR and Temporal Fusion Transformer (TFT) models using the Pinball Loss function and derive the optimal allocation quantile from the operator's cost matrix. Evaluation on a real beam-level 5G traffic dataset shows that the proposed approach reduces operational cost compared to MSE-trained baselines while maintaining calibrated uncertainty estimates. The framework enables dynamic PRB allocation that explicitly balances service reliability against resource efficiency
    
[^17]: 带有无分布安全保证的共形策略学习

    Conformal Policy Learning with Distribution-Free Safety Guarantees

    [https://arxiv.org/abs/2609.17296](https://arxiv.org/abs/2609.17296)

    本文提出共形策略学习（CPL），通过将每个处理决策视为反事实伤害假设检验并利用共形p值阈值化分配处理，首次实现了控制“对会受伤害个体分配处理”概率的无分布安全保证。

    

    策略学习旨在基于个体特征决定谁应该接受处理。在医学和公共政策等以安全为核心关切的高风险场景中，仅仅改善平均结果可能是不够的：决策者还可能希望保护个体免受伤害，这符合“不伤害”的希波克拉底原则。本文提出了共形策略学习（CPL），这是一种带有新型无分布安全保证的策略学习程序，该保证控制了将处理分配给相对于对照组会受到伤害的个体的概率。CPL将每个处理决策视为对反事实伤害假设的检验，并通过共形p值的阈值化来分配处理。这些p值利用可观测的代理变量和选择性校准，解决了所比较的潜在结果永远不会同时被观测到的挑战。对于随机实验…

    arXiv:2609.17296v1 Announce Type: cross  Abstract: Policy learning aims to determine who should be treated based on individual characteristics. In high-stakes settings such as medicine and public policy where safety is a central concern, improving the average outcomes alone may not be sufficient: decision makers may also seek to protect individuals from harm, in line with the Hippocratic principle of ``do no harm.'' In this paper, we propose \textit{conformal policy learning} (CPL), a policy learning procedure with a new distribution-free safety guarantee that controls the probability of assigning treatment to an individual who would be harmed relative to control. CPL views each treatment decision as testing a hypothesis of counterfactual harm and assigns treatment by thresholding conformal p-values. These p-values use observable proxies and selective calibration to address the challenge that the potential outcomes under comparison are never simultaneously observed. For randomized expe
    
[^18]: 相同的流，不同的路径：流匹配中的方差缩减

    Same Flow, Different Paths: Variance Reduction in Flow Matching

    [https://arxiv.org/abs/2609.17287](https://arxiv.org/abs/2609.17287)

    本文从优化角度研究流匹配中的路径选择，证明即使目标函数完全相同，不同路径仍会通过随机梯度方差从根本上改变SGD的收敛速度，并推导出线性路径下的解析最优路径。

    

    在流匹配（FM）中，速度模型 $v_{\theta}$ 通过预定义的连接数据样本与噪声样本的路径 $g_t$ 进行训练（例如 $g_t(x_0, x_1) = (1 - t) x_0 + t x_1$）。在本工作中，我们通过分析随机梯度的方差，从优化角度研究该路径的选择问题。我们考虑路径类 $G(p_t, v^\star_t)$，其中所有路径诱导相同的边际分布 $p_t$ 和边际速度场 $v^\star_t$，因而具有完全相同的流匹配目标函数。我们的主要发现是：即使流匹配目标函数完全相同，路径 $g_t$ 的选择也能从根本上改变 SGD 的收敛速度。(i) 针对线性速度模型和一维高斯数据，我们推导出 SGD 迭代复杂度的紧致界（精确到对数因子），并在诱导相同流匹配问题的线性路径中找到了使该界最小化的解析最优路径。(ii) 随后我们将方差分析扩展到更一般的情形。

    arXiv:2609.17287v1 Announce Type: new  Abstract: In flow matching (FM), a velocity model $v_{\theta}$ is trained using a predefined path $g_t$ that connects data and noise samples (e.g., $g_t(x_0, x_1) = (1 - t) x_0 + t x_1$). In this work, we study the choice of this path from an optimization perspective by analyzing the variance of stochastic gradients. We consider the class $G(p_t,v^\star_t)$ of paths that induce the same marginal distributions $p_t$ and marginal velocity field $v^\star_t$, and therefore the same FM objective. Our main finding is that the choice of path $g_t$ can fundamentally change the convergence rate of SGD, even when the FM objective remains exactly the same. (i) For a linear velocity model and one-dimensional Gaussian data, we derive a tight bound on the SGD iteration complexity up to logarithmic factors and find an analytically optimal path that minimizes this bound among linear paths inducing the same FM problem. (ii) We then extend the variance analysis to 
    
[^19]: 基于全局知识蒸馏与本地分类头自适应的个性化联邦学习

    Personalized Federated Learning through Global Knowledge Distillation and Local Head Adaptation

    [https://arxiv.org/abs/2609.17284](https://arxiv.org/abs/2609.17284)

    提出pFedKDH方法，通过仅聚合共享骨干网络、保留客户端专属分类头并利用重新校准的全局分类头作为蒸馏教师，在标签偏斜的个性化联邦学习场景中显著提升准确率。

    

    当单一全局分类器无法表示客户端特定的标签分布时，统计异质性会限制联邦学习的性能。在这项工作中，我们提出了带分类头自适应的个性化联邦知识蒸馏方法，该方法仅聚合共享的骨干网络，保留持久的客户端专属分类头，并在本地训练期间使用重新校准的全局分类头作为教师模型。在基于类别Dirichlet划分的MNIST、Fashion-MNIST、CIFAR10和CIFAR100数据集上，pFedKDH在大多数设置下取得了最佳准确率，相对于最弱基线的准确率差距高达37.67%，且在多次重复实验中始终保持较低的标准差。组件级诊断和收敛性结果验证了持久分类头以及蒸馏引导的局部优化在标签偏斜数据下的有效性。

    arXiv:2609.17284v1 Announce Type: new  Abstract: Statistical heterogeneity limits federated learning when a single global classifier cannot represent client-specific label distributions. In this work, we propose Personalized Federated Knowledge Distillation with Head Adaptation (pFedKDH), which aggregates only the shared backbone, keeps persistent client-specific heads, and uses a recalibrated global head as a teacher during local training. Across MNIST, Fashion-MNIST, CIFAR10, and CIFAR100 under class-wise Dirichlet partitions, pFedKDH obtains the best accuracy in most settings, with accuracy gaps up to 37.67\% over the weakest baseline and consistently low standard deviation across repetitions. Component-wise diagnostics and convergence results support the role of persistent heads and distillation-guided local optimization under label-skewed data.
    
[^20]: 抓住说谎者容易，洗清诚实者难：语言模型从已验证记录中诊断受损的奖励通道

    Easy to Catch a Liar, Hard to Clear an Honest One: Language Models Diagnosing a Corrupted Reward Channel from a Verified Record

    [https://arxiv.org/abs/2609.17226](https://arxiv.org/abs/2609.17226)

    本研究构建了一个赔付交换与说谎报告者产生完全相同历史的博弈环境，证明冻结的大语言模型仅需一条独立验证记录即可几乎完美地识别出说谎的报告者。

    

    从奖励中学习的智能体必须信任任何提供奖励报告的来源。当报告突然发生变化时，可能是世界变了，也可能是报告者出了问题。仅凭报告本身，这两者是无法区分的，而且强化学习理论表明，再多的后续经验也无法将二者分开。规定的解决途径是获取关于报告者本身的更丰富数据。我们探究一个被冻结的语言模型在被提供这类数据时是否会加以利用。我们构建了一个双选项游戏，其中一次赔付交换和一个说谎的报告者会产生字节完全相同的历史记录。然后我们加入一条已验证的记录：对某一轮真实结果的独立核查，并将其与报告者对那一轮的陈述并排呈现。仅仅这一行信息就能定案。我们让来自两个家族的三个大型模型用一个字母回答一个问题：报告者是诚实的还是在说谎？它们几乎完美地识别出了说谎的报告者。在70B参数级别上，这一结论在我们尝试的所有条件下都成立。

    arXiv:2609.17226v1 Announce Type: cross  Abstract: An agent that learns from rewards has to trust whatever reports those rewards. When the reports suddenly change, either the world changed or the reporter broke. From the reports alone these are indistinguishable, and reinforcement learning theory shows that no amount of further experience separates them. The prescribed escape is richer data about the reporter itself. We ask whether a frozen language model, handed exactly that data, uses it. We build a two-option game in which a payout swap and a lying reporter produce byte-identical histories. Then we add one verified record: an independent check of one round's real result, printed beside what the reporter said about that round. That single line settles the case. We ask three large models, from two families, to answer one question with one letter. Is the reporter honest or lying? They catch a lying reporter almost perfectly. At the 70B class that holds in every condition we tried; the 
    
[^21]: 医疗人工智能中的记忆偏差

    Memorisation bias in medical AI

    [https://arxiv.org/abs/2609.17223](https://arxiv.org/abs/2609.17223)

    该研究揭示了医疗AI中的“记忆偏差”现象：模型在训练中接触过患者的匿名历史数据后会显著改变对该患者未来数据的预测，且这种偏差可持续存在长达数十年，对临床部署具有深远影响。

    

    医疗人工智能模型在改善患者预后方面具有巨大潜力，但众所周知，它们也会在无意中记住训练数据集中的个体记录。虽然这种记忆现象已被认为与针对性的隐私攻击有关，但其对临床部署的影响——即患者可能由一个在训练期间见过其历史数据的模型进行评估——目前仍知之甚少。本研究表明，如果一个模型在训练期间观察到同一患者的匿名化历史数据，那么该模型对该患者未来未见数据的预测可能发生显著变化，我们将这种现象称为“记忆偏差”。我们证明这种偏差存在于多种数据模态和模型架构中，并且可持续跨越很长的时间跨度：在某些情况下，记忆偏差甚至在训练所用历史记录之后数十年获取的未来记录上依然存在。此外，在模拟的前瞻性部署场景中，记忆偏差具有不对称的影响……

    arXiv:2609.17223v1 Announce Type: new  Abstract: Medical AI models hold immense potential to improve patient outcomes, but they are also known to unintentionally memorise individual records from their training datasets. While such memorisation has been linked to targeted privacy attacks, its consequences for clinical deployment, where patients may be assessed by a model that saw their historical data during training, remain poorly understood. Here we show that predictions on a patient's unseen future data can change significantly if a model observed that same patient's anonymised historical data during training, a phenomenon we term "memorisation bias". We demonstrate that this bias exists across diverse data modalities and model architectures, and over prolonged time spans: in some cases, memorisation bias persists on future records acquired decades after the historical records used for training. Moreover, in simulated prospective deployment, memorisation bias has asymmetric effects o
    
[^22]: 面向人体定位的跨域推理：将Wi-Fi RSSI数据应用于CSI训练的模型

    Cross-Domain Inference for Human Localization: Applying Wi-Fi RSSI Data to CSI-Trained Models

    [https://arxiv.org/abs/2609.17204](https://arxiv.org/abs/2609.17204)

    本文提出了一种跨域推理方法，将易于获取的Wi-Fi RSSI数据输入到原本基于CSI数据训练的姿态预测模型中，验证了在低权限物联网设备上实现人体位置预测的可行性。

    

    Wi-Fi信号数据可被用于侵犯个人隐私。虽然许多现有方法依赖于信道状态信息（CSI），但在典型的物联网设备上收集此类数据通常需要较高的操作系统权限和专用驱动程序。因此，本文研究了利用接收信号强度指示（RSSI）数据来预测人体位置的可行性。选择RSSI是因为即使在用户权限受限的设备上也能获取该数据，因此更适用于更广泛的物联网设备。为了绕过获取训练数据以训练基于RSSI模型的繁琐过程，本研究使用了一个现有的Wi-Fi姿态预测项目。然而，该项目假定输入为CSI数据。因此，我们研究了跨域推理的可行性，即将RSSI数据输入到这个现有的基于CSI的模型中。我们收集了一个RSSI数据集，并与之同步（原文在此处截断）。

    arXiv:2609.17204v1 Announce Type: cross  Abstract: Wi-Fi signal data can be used to compromise the privacy of individuals. While many existing approaches rely on Channel State Information (CSI), collecting this data on typical IoT devices often requires elevated operating system permissions and specialized drivers. Consequently, this paper investigates the feasibility of utilizing Received Signal Strength Indicator (RSSI) data to predict human locations. RSSI was selected because it is accessible even on devices with limited user permissions, and therefore is more applicable to a wider array of IoT devices. To bypass the tedious process of obtaining training data needed to train an RSSI-based model, an existing Wi-Fi pose prediction project was used in this research. However, that project assumed CSI data as input. Therefore, we investigate the feasibility of cross-domain inference, i.e., feeding RSSI data into that existing CSI-based model. We collected an RSSI dataset, synchronized w
    
[^23]: MyoFlow：面向跨会话与跨被试高密度表面肌电手势识别的锚点绑定校正流

    MyoFlow: Anchor-Tied Rectified Flow for HD-sEMG Gesture Recognition Across Sessions and Subjects

    [https://arxiv.org/abs/2609.17194](https://arxiv.org/abs/2609.17194)

    提出了首个面向跨会话与跨被试高密度表面肌电手势识别的判别式流匹配框架MyoFlow，通过域条件化校正流将分类重构为向手势锚点的输运过程，无需独立分类头即可实现零样本预测。

    

    高密度表面肌电（HD-sEMG）手势识别可支持假肢控制、辅助机器人和康复训练，但电极重新佩戴和生理个体差异引起的分布偏移会降低跨会话与跨被试场景下的识别准确率。现有的生成式高密度肌电模型主要用于信号合成以进行数据增强；尽管扩散模型能够增强表征学习，但预测仍需依赖独立的分类器。为了将学习到的动力学与决策规则绑定在一起，我们提出了MyoFlow，这是首个面向跨会话与跨被试高密度肌电识别的判别式流匹配框架。该方法将分类重构为锚点绑定的输运过程：域条件化的校正流将编码后的信号窗口输运至手势锚点，这些锚点既作为输运目标，又定义了最近锚点的决策几何结构，从而无需独立的分类头即可实现零样本预测。在Hyser数据集上，MyoFlow提升了平均跨会话（摘要在此处截断）

    arXiv:2609.17194v1 Announce Type: new  Abstract: High-density surface electromyography (HD-sEMG) gesture recognition supports prosthetic control, assistive robotics, and rehabilitation, but electrode re-donning and physiological variability cause distribution shifts that degrade accuracy across sessions and subjects. Generative HD-sEMG models primarily synthesize signals for augmentation; although diffusion models enhance representation learning, prediction still relies on a separate classifier. To tie learned dynamics to the decision rule, we propose MyoFlow, the first discriminative flow-matching framework for HD-sEMG recognition across sessions and subjects. It recasts classification as anchor-tied transport: a domain-conditioned rectified flow moves encoded windows toward gesture anchors that serve as transport targets and define the nearest-anchor decision geometry, enabling zero-shot prediction without an independent head. On the Hyser dataset, MyoFlow improves mean cross-session
    
[^24]: LoopSpec：面向循环Transformer的流水线式自推测解码

    LoopSpec: Pipelined Self-Speculative Decoding for Looped Transformers

    [https://arxiv.org/abs/2609.17184](https://arxiv.org/abs/2609.17184)

    LoopSpec是一个专为循环Transformer设计的免训练自推测解码框架，通过从早期循环状态提取草稿token并以流水线方式重叠草稿生成与目标验证，从而在无需辅助模型的情况下显著提升解码效率。

    

    循环Transformer通过在多个循环深度上重复应用共享的Transformer块堆栈，以紧凑的参数规模实现了强大的性能。然而，由于共享权重在每次循环深度都需要被访问，其解码延迟高于参数规模相当的标准Transformer模型。为了提高解码效率，自推测解码特别适合循环Transformer，因为其中间循环状态可以直接提供草稿预测，而无需辅助草稿模型。因此，我们提出了LoopSpec，一个专为循环Transformer设计的免训练自推测解码框架。LoopSpec从早期循环状态中提取草稿token，并以流水线方式运行，将未来token的草稿生成与当前token的目标验证重叠进行。为了在不产生过多计算开销的情况下提高草稿准确性，我们引入了一种选择性二次……（摘要在此处截断）

    arXiv:2609.17184v1 Announce Type: cross  Abstract: Looped Transformers achieve strong performance with compact parameter sizes by repeatedly applying a shared stack of Transformer blocks across recurrent depths. However, they incur higher decoding latency than standard Transformer models of comparable parameter size because shared weights are accessed at every recurrent depth. To improve decoding efficiency, self-speculative decoding is particularly well suited to Looped Transformers, as their intermediate recurrent states can directly provide draft predictions without an auxiliary draft model. We therefore propose LoopSpec, a training-free self-speculative decoding framework tailored for Looped Transformers. LoopSpec extracts draft tokens from early recurrent states and operates in a pipelined manner, overlapping draft generation of future tokens with target verification of the current token. To improve draft accuracy without excessive compute overhead, we introduce a selective second
    
[^25]: IRENE：一种用于意大利雷达降水临近预报的卷积GRU集合模型

    IRENE: A Convolutional GRU Ensemble Model for Radar Precipitation Nowcasting over Italy

    [https://arxiv.org/abs/2609.17175](https://arxiv.org/abs/2609.17175)

    IRENE是一个基于多尺度ConvGRU的深度学习集合模型，结合重要性采样、afCRPS概率损失以及对抗训练和谱约束等创新配置，实现了意大利地区1公里空间和5分钟时间分辨率的概率性降水临近预报。

    

    我们提出了IRENE（意大利雷达集合临近预报实验），这是一个深度学习模型，用于在意大利区域以1公里空间分辨率和5分钟时间分辨率进行概率性短时降水临近预报。IRENE采用基于多尺度卷积门控循环单元（ConvGRU）构建的编码器-预报器架构，并在意大利民防部门（DPC）生成的全国雷达合成数据上进行训练。重要性采样方案将训练重点聚焦于与降水相关的事件，同时采用近似公平的连续排序概率评分（afCRPS）作为主要的概率损失函数。此外还提出了两种额外的训练配置：对抗（GAN）变体IRENE-GAN，旨在提高生成预报的空间清晰度；以及谱约束变体IRENE-GAN-RAPSD，其中对抗目标辅以对径向平均功率谱密度的显式惩罚。

    arXiv:2609.17175v1 Announce Type: new  Abstract: We present IRENE (Italian Radar Ensemble Nowcasting Experiment), a deep learning model for probabilistic short-range precipitation nowcasting over the Italian domain at \SI{1}{km} spatial and 5 min temporal resolution. IRENE adopts an encoder--forecaster architecture built on multi-scale Convolutional Gated Recurrent Units (ConvGRUs), trained on the national radar composite produced by the Italian Civil Protection Department (DPC). An importance-sampling scheme focuses training on precipitation-relevant events, while the almost-fair Continuous Ranked Probability Score (afCRPS) is adopted as the primary probabilistic loss function. Two additional training configurations are proposed: an adversarial (GAN) variant, IRENE-GAN, designed to improve the spatial sharpness of the generated forecasts, and a spectrally constrained variant, IRENE-GAN-RAPSD, in which the adversarial objective is complemented by an explicit penalty on the radially ave
    
[^26]: 基于自适应导数阶随机解释的全局与局部可解释性统一框架

    A unified framework for global and local interpretability using adaptive derivative-ordered random explanation

    [https://arxiv.org/abs/2609.17171](https://arxiv.org/abs/2609.17171)

    本文提出ADORE方法，利用一阶和二阶导数在统一分析框架内同时实现全局特征重要性与局部样本贡献的可解释性分析，有效捕捉非线性特征-样本交互并精确量化特征影响。

    

    复杂机器学习模型的可解释性至关重要，尤其是在医疗健康和金融等现实世界的高风险领域。然而，现有的事后可解释性方法存在固有的局限性：分析过程碎片化、对非线性特征交互的建模能力不足、计算效率低下，以及过度依赖特定的模型架构。为了应对这些挑战，本文提出了一种新方法——自适应导数阶随机解释（ADORE），该方法利用一阶和二阶导数来适应非线性模型的复杂性，同时能够在统一的分析框架内有效捕捉特征与样本之间的交互。ADORE 将全局特征重要性与局部样本贡献相结合，通过同时捕捉影响的大小和方向来精确量化特征影响，并识别影响模型的关键样本。

    arXiv:2609.17171v1 Announce Type: cross  Abstract: The interpretability of complex machine learning models is of paramount importance, especially in real-world high-stakes domains such as healthcare and finance. However, existing post-hoc interpretability methods suffer from inherent limitations: fragmented analytical processes, inadequate capacity to model nonlinear feature interactions, computational inefficiencies, and over-reliance on specific model architectures. To address these challenges, this paper provides a novel method - Adaptive Derivative-Ordered Random Explanation (ADORE) - that leverages first- and second-order derivatives to accommodate nonlinear model complexities, while enabling effective capture of feature-sample interactions within a unified analytical framework. ADORE integrates global feature importance with local sample contributions, precisely quantifying feature impact by capturing both magnitude and direction, and identifying critical samples influencing mode
    
[^27]: 用于气动表面预测的神经场集成方法：ONERA CRM壁面分布2025挑战赛夺冠方案

    Neural Field Ensembles for Aerodynamic Surface Prediction: Winning Solution to the ONERA CRM Wall Distribution 2025 Challenge

    [https://arxiv.org/abs/2609.17160](https://arxiv.org/abs/2609.17160)

    本文提出一种基于条件神经场的集成学习方法，结合傅里叶特征编码和相对平方误差目标，实现对NASA CRM翼身挂架短舱构型在不同工况下压力和表面摩擦系数分布的精确预测，赢得ONERA CRM壁面分布2025挑战赛冠军。

    

    机器学习代理模型为高保真计算流体动力学（CFD）模拟在气动分析与设计中提供了一种有前景的替代方案。然而，由于几何外形复杂、流动状态多样以及训练数据有限，为真实飞机机构型构建精确的代理模型仍然极具挑战性。本工作提出了在ONERA CRM壁面分布回归挑战赛中获得第一名的方法，该挑战赛重点预测NASA通用研究模型（CRM）机翼-机身-挂架-短舱构型在不同运行工况下的压力和表面摩擦系数分布。所提出的方法将该问题表述为一个条件神经场，将空间坐标、表面法向量以及运行工况映射到气动壁面物理量。通过采用傅里叶特征编码、与挑战赛评估指标相一致的相对平方误差目标函数、集成学习以及k……

    arXiv:2609.17160v1 Announce Type: new  Abstract: Machine-learning surrogate models offer a promising alternative to high-fidelity Computational Fluid Dynamics (CFD) simulations for aerodynamic analysis and design. However, constructing accurate surrogates for realistic aircraft configurations remain challenging due to complex geometries, multiple flow regimes, and limited training data. This work presents the methodology that achieved first place in the ONERA CRM Wall Distribution Regression Challenge, which focuses on predicting pressure and skin-friction coefficient distributions over the NASA Common Research Model wing-body-pylon-nacelle configuration under different operating conditions. The proposed approach formulates the problem as a conditional neural field mapping spatial coordinates, surface normals, and operating conditions to aerodynamic wall quantities. Fourier feature encoding, a relative squared error objective aligned with the challenge metric, ensemble learning, and $k
    
[^28]: ResLRP：残差消除在视觉Transformer归因不稳定性中的作用

    ResLRP: The Role of Residual Cancellation in Attribution Instability in Vision Transformers

    [https://arxiv.org/abs/2609.17152](https://arxiv.org/abs/2609.17152)

    本文揭示了视觉Transformer中残差连接的消除效应是导致LRP归因不稳定的根本原因，并提出ResLRP方法，其传播规则显式考虑残差消除、严格守恒且可证明地限制归因爆炸。

    

    视觉Transformer（ViT）是大多数现代视觉模型的核心，然而获得细粒度、忠实且稳定的输入归因仍然具有挑战性。逐层相关性传播（LRP）已被适配到Transformer的注意力机制中，但在ViT中它常常产生嘈杂、不忠实的解释。我们表明，缺失的关键因素是对残差连接的处理：残差路径中的消除效应会导致归因爆炸。此外，我们发现这些消除效应在ViT中远比在语言Transformer中强烈。为了解决这一问题，我们引入了残差感知逐层相关性传播，它是LRP的一个简单扩展，其传播规则显式地考虑了残差分支中的消除效应，具有精确守恒性，并且可证明地限制了相关性爆炸。因果通道干预实验证实，残差消除——而非一般性的正则化效应——才是导致归因问题的原因。

    arXiv:2609.17152v1 Announce Type: cross  Abstract: Vision Transformers (ViTs) are central to most modern vision models, yet obtaining input attributions that are fine-grained, faithful, and stable remains challenging. Layer-wise Relevance Propagation (LRP) has been adapted to transformer attention, but in ViTs it often produces noisy, unfaithful explanations. We show that the missing ingredient is the treatment of residual connections: cancellation effects in residual pathways lead to attribution explosion. Moreover, we find that these cancellations are substantially stronger in ViTs than in language transformers. To address this issue, we introduce Residual-aware Layer-wise Relevance Propagation (ResLRP), a simple extension of LRP whose propagation rules explicitly account for cancellations in residual branches, are exactly conservative, and provably bound relevance explosion. Causal channel-wise interventions confirm that residual cancellation, not a generic regularization effect, dr
    
[^29]: 面向自动驾驶赛车中不确定对手车辆轨迹预测的基于核的度量学习

    Kernel-Based Metrics Learning for Uncertain Opponent Vehicle Trajectory Prediction in Autonomous Racing

    [https://arxiv.org/abs/2609.17147](https://arxiv.org/abs/2609.17147)

    该论文提出用于深度核学习的异构核度量，以无监督方式从自车与对手车辆的交互中捕捉并区分不同驾驶策略，实现自动驾驶赛车中对手车辆轨迹的精确预测与不确定性估计，并在1/10比例赛车平台上验证了更高的预测精度、安全超车能力以及车载计算效率。

    

    自动驾驶赛车在安全超越具有不确定轨迹的对手车辆（OV）方面面临重大挑战，这种不确定性源于未知的驾驶策略。为应对这些挑战，本研究提出了一种用于深度核学习（DKL）的异构核度量，旨在稳健地捕捉对手车辆多样化的驾驶策略，并进行精确的轨迹预测及相应的不确定性估计。所提出核度量的一个关键优点在于，其能够基于自车（EV）与对手车辆之间观察到的交互，以无监督的方式对齐相似的驾驶策略并分离不相似的驾驶策略。通过在1/10比例赛车平台上的实验研究验证了所提方法的有效性，展示了预测精度的提升，从而实现对对手车辆的安全超越。此外，该方法对于车载计算单元具有良好的计算效率。

    arXiv:2609.17147v1 Announce Type: cross  Abstract: Autonomous racing confronts significant challenges in safely overtaking Opponent Vehicles (OVs) that exhibit uncertain trajectories, stemming from unknown driving policies. To address these challenges, this study proposes heterogeneous kernel metrics for Deep Kernel Learning (DKL), designed to robustly capture the diverse driving policies of OVs, and carry out precise trajectory predictions along with the associated uncertainties. A key virtue of the proposed kernel metrics lies in their ability to align similar driving policies and disjoin dissimilar ones in an unsupervised manner, given the observed interactions between the Ego Vehicle (EV) and OVs. The efficacy of the proposed method is substantiated through experimental studies on a 1/10th scale racecar platform, demonstrating improved prediction accuracy and thereby safely overtaking against OVs. Furthermore, our method is computationally efficient for onboard computing units, aff
    
[^30]: 基于不确定性感知自适应的可通行性预测持续学习

    Continual Learning for Traversability Prediction with Uncertainty-Aware Adaptation

    [https://arxiv.org/abs/2609.17141](https://arxiv.org/abs/2609.17141)

    该论文提出了一种基于生成式经验回忆模型的可通行性预测持续学习框架，无需存储历史数据即可保留先前经验，并通过融合生成样本的不确定性实现不确定性感知的自适应，从而解决灾难性遗忘问题。

    

    可通行性预测是非结构化环境中自主导航的关键组成部分，其中复杂且不确定的机器人与地形交互会带来牵引力损失和动态失稳等重大挑战。尽管基于学习的可通行性预测近来取得了进展，但这些方法往往难以适应新地形；即使实现了适应，保留先前训练环境的经验仍然是一个挑战，这一问题被称为灾难性遗忘。为应对这一挑战，我们提出了一种用于可通行性预测的持续学习框架，该框架使用生成式经验回忆模型逐步适应新地形。所提出框架的关键优点有两个方面：i) 无需存储过去数据即可保留先前经验；ii) 融合来自回忆模型的生成样本的不确定性，从而实现不确定性感知的自适应。真实世界实验（原文截断）

    arXiv:2609.17141v1 Announce Type: cross  Abstract: Traversability prediction is a critical component of autonomous navigation in unstructured environments, where complex and uncertain robot-terrain interactions pose significant challenges such as traction loss and dynamic instability. Despite recent progress in learning-based traversability prediction, these methods often fail to adapt to novel terrains. Even when adaptation is achieved, retaining experience from previously trained environments remains a challenge, a problem known as catastrophic forgetting. To address this challenge, we propose a continual learning framework for traversability prediction that incrementally adapts to new terrains using a generative experience recall model. A key virtue of the proposed framework is two folds: i) retain prior experience without storing past data; and ii) incorporate the uncertainty of the generated samples from the recall model, enabling uncertainty-aware adaptation. Real-world experimen
    
[^31]: 从基础嵌入到农田地图：标签效率、时间可迁移性与独立人工验证

    From Foundation Embeddings to Cropland Maps: Label Efficiency, Temporal Transferability and Independent Human Validation

    [https://arxiv.org/abs/2609.17138](https://arxiv.org/abs/2609.17138)

    该研究表明，无需微调基础模型，仅用轻量级分类器即可基于AlphaEarth嵌入实现93.7%精度的农田制图，且仅需6万个标注像素就能达到接近860万全量数据的性能。

    

    地理空间基础模型能够提供卫星影像的可复用表示，支持在有限的特定任务建模下进行下游制图。我们评估年度AlphaEarth嵌入是否支持美国缅因州的耕种与非耕种二分类制图，研究使用了192个空间上相互分离的图块，以及来自美国农业部农作物数据层（CDL）的标签。在不微调基础模型的情况下，一个轻量级分类器在留出图块上达到93.7%的总体精度和90.8%的平衡精度。逻辑回归与梯度提升集成模型的差距在0.3个百分点以内，而仅使用类质心、无需拟合任何参数的最近类质心规则也达到了90.2%。一个包含60,000个平衡标注像素的样本，其性能与完整的860万像素池相比差距在1.3个百分点以内；由于像素存在空间自相关，这一结果反映的是像素采样效率，而非60,000个独立标注的效率。

    arXiv:2609.17138v1 Announce Type: cross  Abstract: Geospatial foundation models provide reusable representations of satellite imagery that support downstream mapping with limited task-specific modelling. We evaluate whether annual AlphaEarth embeddings support binary cultivated-versus-non-cultivated mapping in Maine, USA, using 192 spatially separated patches and labels derived from the USDA Cropland Data Layer (CDL). Without fine-tuning the foundation model, a lightweight classifier reaches 93.7% overall accuracy and 90.8% balanced accuracy on held-out patches. Logistic regression is within 0.3 percentage points of a gradient-boosted ensemble, while a nearest-class-centroid rule, which uses class centroids but fits no parameters, reaches 90.2%. A balanced sample of 60,000 labelled pixels is within 1.3 percentage points of the full pool of 8.6 million pixels; because pixels are spatially autocorrelated, this result concerns pixel-sample efficiency rather than 60,000 independent annotat
    
[^32]: 内在机器人奖励：复用VLA表征以实现自主评估与策略改进

    Intrinsic Robot Rewarding: Reusing VLA Representations for Autonomous Evaluation and Policy Improvement

    [https://arxiv.org/abs/2609.17115](https://arxiv.org/abs/2609.17115)

    该论文提出内在机器人奖励（IRR）方法，通过复用VLA系统中已有的视觉表征和成功示范，无需额外训练评估器即可自主评估机器人执行结果并生成奖励信号，从而降低集成成本、提升奖励计算效率并减少人工评分负担。

    

    arXiv:2609.17115v1 公告类型：交叉 摘要：视觉-语言-动作（VLA）系统已经为机器人学习汇聚了两种宝贵资源：丰富的视觉表征和成功任务执行的示范。内在机器人奖励（Intrinsic Robot Rewarding, IRR）提出将这些资源用于第二个互补目的：评估机器人自身的执行结果，并为策略改进提供反馈。成功示范的终点定义了任务特定的参考基准，而策略中冻结的视觉编码器则提供了评估新执行结果的特征空间。该核心奖励机制在现有流程中仅增加了一个参考库和一个评分操作，无需单独训练的评估器或额外的感知骨干网络。我们的观点是，这种复用为降低集成成本、实现高效奖励计算以及减少重复性的人工结果评分提供了一条有前景的路径。在视觉奖励和从示范学习等已有研究的基础上……（原文摘要到此截断）

    arXiv:2609.17115v1 Announce Type: cross  Abstract: Vision-language-action (VLA) systems already bring together two valuable resources for robot learning: rich visual representations and demonstrations of successful task execution. Intrinsic Robot Rewarding (IRR) proposes to use these resources for a second, complementary purpose: evaluating the robot's own outcomes and providing feedback for policy improvement. Successful demonstration endpoints define task-specific references, and the policy's frozen visual encoder provides the feature space in which new outcomes are assessed. The core reward mechanism adds a reference bank and a scoring operation to the existing pipeline, without requiring a separate learned evaluator or an additional perception backbone. Our position is that this reuse offers a promising route to lower integration effort, efficient reward computation, and reduced recurring human outcome scoring. Building on established research in visual rewards and learning from ex
    
[^33]: 基于随机动态模态分解与深度学习的高保真数字孪生数据模型及其在流体力学中的应用

    High-Fidelity Digital Twin Data Models by Randomized Dynamic Mode Decomposition and Deep Learning with Applications in Fluid Dynamics

    [https://arxiv.org/abs/2609.17101](https://arxiv.org/abs/2609.17101)

    本文提出了一种结合随机动态模态分解与深度学习的新框架，用于从数值代码输出中构建高精度、低计算成本的数字孪生数据模型，并在流体力学问题中验证了其有效性。

    

    本文的目的是通过非侵入式技术（即不需要将控制方程通过Galerkin投影到约化模态基上），从数值代码输出中识别高保真的数字孪生数据模型。本文作者定义了数字孪生数据模型（DTM）的概念，即一种复杂度降低的模型，其主要特征是能够镜像反映原始过程的行为。DTM的显著优势在于能够高精度地复现动力学行为，同时在CPU时间和硬件方面的成本更低，适用于因动力学随时间演化的复杂性而难以探索的场景。本文通过结合两种最先进的工具——随机动态模态分解和深度学习人工智能——引入了一个构建高效数字孪生数据模型的新框架。结果表明，模型的输出与原始源数据保持一致，并具有降低计算成本的优势。

    arXiv:2609.17101v1 Announce Type: new  Abstract: The purpose of this paper is the identification of high-fidelity digital twin data models from numerical code outputs by non-intrusive techniques (i.e., not requiring Galerkin projection of the governing equations onto the reduced modes basis). In this paper the author defines the concept of the digital twin data model (DTM) as a model of reduced complexity that has the main feature of mirroring the original process behavior. The significant advantage of a DTM is to reproduce the dynamics with high accuracy and reduced costs in CPU time and hardware for settings difficult to explore because of the complexity of the dynamics over time. This paper introduces a new framework for creating efficient digital twin data models by combining two state-of-the-art tools: randomized dynamic mode decomposition and deep learning artificial intelligence. It is shown that the outputs are consistent with the original source data with the advantage of redu
    
[^34]: 使用参数化度量的协方差矩阵优化

    Optimization over covariance matrices with a parameterized metric

    [https://arxiv.org/abs/2609.17089](https://arxiv.org/abs/2609.17089)

    本文提出一个统一了欧几里得、Bures-Wasserstein和仿射不变度量的双参数黎曼度量族，并通过分析解处黎曼Hessian的条件数，证明其下界仅依赖于参数之和 $r=p+q$，从而为协方差矩阵优化中的度量选择（预条件化）提供了理论判据。

    

    黎曼度量的选择会强烈影响基于梯度的协方差矩阵优化的收敛性。欧几里得度量、Bures-Wasserstein度量和仿射不变度量是常见的选择，但它们的相对有效性取决于目标函数。我们引入了一个由 $X^{p}LX^{q}+X^{q}LX^{p}=U$ 定义的双参数族，在每个切向量 $U$ 处求解 $L$，该族在 $(0,0)$、$(1,0)$ 和 $(1,1)$ 处恰好包含上述三种度量，并将其推广至更广泛的范围。我们将度量族中成员的选择视为针对给定问题的一种预条件化方式。为此，我们分析了解处黎曼Hessian矩阵的条件数。我们证明该条件数满足一个下界，且该下界仅通过指数 $r=p+q$ 依赖于 $(p,q)$。当欧几里得Hessian是一个不混合特征方向的纯幂时，成员 $p=q=r/2$ 可达到该下界，并且一个闭式判据可以识别出其他能达到该下界的成员。我们还讨论了……（原文截断）

    arXiv:2609.17089v1 Announce Type: cross  Abstract: The choice of Riemannian metric can strongly influence the convergence of gradient-based optimization over covariance matrices. Euclidean, Bures-Wasserstein and affine-invariant metrics are common choices, but their relative effectiveness depends on the objective. We introduce a two-parameter family defined by $X^{p}LX^{q}+X^{q}LX^{p}=U$, solved for $L$ at each tangent vector $U$, that contains all three as exact members, at $(0,0)$, $(1,0)$ and $(1,1)$, and extends past them. We treat the choice of member as a particular way of preconditioning for a given problem. To this end, we analyze the conditioning of the Riemannian Hessian at the solution. We show that it obeys a lower bound that depends on $(p,q)$ only through the exponent $r=p+q$. When the Euclidean Hessian is a pure power that mixes no eigendirections, the member $p=q=r/2$ attains that bound, and a closed-form criterion identifies the other members that do. We discuss ways t
    
[^35]: 间接编码基质中的仿生调色板演化：时间尺度兼容性塑造激活函数发现

    Bio-Inspired Palette Evolution in Indirectly Encoded Substrates: Timescale Compatibility Shapes Activation Function Discovery

    [https://arxiv.org/abs/2609.17067](https://arxiv.org/abs/2609.17067)

    该研究将神经网络激活函数集合的演化发现建模为元学习问题，设计了11种受生物适应机制启发的策略（如昼夜节律振荡门控与免疫克隆选择）在演化中动态调整可用激活函数，并发现生物机制与演化过程之间的时间尺度兼容性是决定激活函数发现成败的关键。

    

    间接编码神经网络可以为单个节点分配不同的激活函数，但正确的函数往往难以预先知晓。当可用集合仅包含标准单调函数时，诸如奇偶性之类的问题将变得无法求解，然而包罗万象的函数集合又不如经过精心筛选的集合表现出色。演化应当如何发现该使用哪些函数？我们将此作为一个元学习问题来处理，设计了13种策略（其中11种受生物适应机制启发，另加基线和预言机对照组），在演化过程中动态修改可用激活函数集合。每种策略都将一种生物学原理转化为演化算子：例如，受昼夜节律启发的振荡门控机制按照固定时间表使函数循环进出调色板，而受免疫系统启发的克隆选择机制则会永久保护与适应度持续正相关的函数。我们在超过3,00

    arXiv:2609.17067v1 Announce Type: cross  Abstract: Indirectly encoded neural networks can assign different activation functions to individual nodes, but the right functions are rarely known in advance. When the available set contains only standard monotonic functions, problems like parity become unsolvable, yet an all-inclusive palette underperforms a curated one. How should evolution discover which functions to use? We address this as a meta-learning problem, designing 13 strategies (11 inspired by biological adaptation mechanisms, plus baseline and oracle controls) that modify the set of available activation functions during evolution. Each strategy translates a biological principle into an evolutionary operator: for example, circadian-inspired oscillatory gating cycles functions in and out of the palette on a fixed schedule, while immune-inspired Clonal Selection permanently protects functions that consistently correlate with fitness. We evaluate all strategies across more than 3,00
    
[^36]: 人类行为中的神经符号分层意图预测

    Neuro-Symbolic Hierarchical Intention Anticipation in Human Behavior

    [https://arxiv.org/abs/2609.17064](https://arxiv.org/abs/2609.17064)

    该论文提出一种神经符号分层规划解码器（HPD），能够在人类行为尚未完成时从部分观察的多模态片段中在四个本体层次上预测意图与剩余行为，并通过软神经符号正则化训练和硬可达性掩码推理保证预测的本体有效性。

    

    辅助自主系统必须在观察到的行为完成之前就预测人类的目标。本文将预测任务表述为：从部分观察到的多模态行为片段中进行目标推断，并结合对剩余行为的结构化预测，而非精确的运动轨迹预测。一个紧凑的分层规划解码器（HPD）被附加到冻结的神经符号识别编码器上，在四个本体层次上进行预测：下一步动作、剩余活动与低层意图，以及整个片段的高层意图（HLI）。解码器通过结合转换一致性和层次连续性损失的软神经符号正则化进行训练，并在推理阶段使用硬可达性掩码来强制保证本体的有效性。在基于NTU RGB+D 120特征构建的包含15,002个多模态片段的组合式四层基准测试上，同时观察到三项核心特性。相对于……的优势（摘要在此处截断）

    arXiv:2609.17064v1 Announce Type: new  Abstract: Assistive autonomous systems must anticipate human goals before an observed behavior is complete. This article formulates anticipation as goal inference from a partially observed multimodal episode together with structured prediction of the remaining behavior, rather than exact motor forecasting. A compact Hierarchical Planning Decoder (HPD) is attached to a frozen neuro-symbolic recognition encoder and predicts, at four ontological levels, the next actions, the remaining activities and low-level intentions, and the episode high-level intention(HLI). The decoder is trained with soft neuro-symbolic regularization combining transition-coherence and hierarchical continuity losses, and is decoded with hard reachability masks that enforce ontological validity at inference. On a compositional four-level benchmark of 15,002 multimodal episodes built over NTU RGB+D 120 features, three headline properties are observed together. The advantage over
    
[^37]: 将统一拓扑签名重新用于图表示学习

    Repurposing Unified Topological Signatures for Graph Representation Learning

    [https://arxiv.org/abs/2609.17061](https://arxiv.org/abs/2609.17061)

    该论文提出将基于持久同调的两种统一拓扑签名（静态的Graph_UTS和动态的Embedding_UTS）集成到图表示学习中，突破传统消息传递GNN受限于1-WL图同构测试的判别能力瓶颈。

    

    消息传递图神经网络（GNN）通过迭代地传播和聚合局部邻域信息，然后进行全局读出，来学习图表示。然而，它们的判别能力被Weisfeiler–Lehman（1-WL）图同构测试所上界限制。这阻碍了GNN区分某些具有相同局部邻域结构的非同构图，常常导致生成相似的图表示。统一拓扑签名（UTS）捕捉了从持久同调导出的全局图拓扑的紧凑、多尺度表示。我们引入了两种互补的UTS签名：Graph_UTS——输入图拓扑的静态签名，以及Embedding_UTS——演化嵌入拓扑的动态签名。它们编码了基于1-WL的消息传递GNN无法获取的结构信息，然而此前它们的能力仅被用于事后的嵌入空间分析。我们将UTS集成到图表示学习框架中，以提升图神经网络的判别能力。

    arXiv:2609.17061v1 Announce Type: cross  Abstract: Message-passing Graph Neural Networks (GNNs) iteratively propagate and aggregate local neighborhood information followed by global readout to learn graph representations. However, their discriminative power is upper-bounded by the Weisfeiler--Lehman (1-WL) graph isomorphism test. This prevents GNNs from distinguishing certain non-isomorphic graphs with identical local neighborhood structures, often leading to similar graph representations. Unified Topological Signatures (UTS) capture compact, multi-scale representation of global graph topology derived from persistent homology. We introduce two complementary UTS signatures: Graph_UTS- a static signature of the input graph topology, and Embedding_UTS- a dynamic signature of the evolving embedding topology. They encode structural information inaccessible to 1-WL-based message-passing GNNs, yet their capabilities are explored solely for post-hoc embedding-space analysis. We integrate UTS i
    
[^38]: 近优非凸矩阵补全

    Near-Optimal Nonconvex Matrix Completion

    [https://arxiv.org/abs/2609.17048](https://arxiv.org/abs/2609.17048)

    该论文通过分析采用多尺度残差初始化的黎曼梯度下降和黎曼高斯-牛顿方法，使非凸矩阵补全的样本复杂度达到与凸方法相当的近优水平，显著降低了对秩的依赖。

    

    我们研究矩阵补全的非凸方法，即从低秩矩阵的部分元素中恢复该矩阵的问题。凸方法在忽略对数因子的情况下，可以实现与矩阵维度和秩呈线性关系的样本复杂度，而常用的非凸方法的全局保证则需要对秩具有更高的多项式依赖。我们通过分析黎曼梯度下降（RGD）和黎曼高斯-牛顿（RGN）方法来弥补这一差距。对于秩为 $r$、非相干参数为 $\mu$、条件数为 $\kappa$ 的 $n\times n$ 矩阵，这两种方法分别仅需 $O(\mu nr\log n\log(n\kappa))$ 和 $O(\mu nr\log n\log(2\mu r\kappa))$ 个观测值，即可高概率实现精确恢复。这两种方法采用多尺度残差初始化，同时该分析同时控制了谱误差和非相干性。由此得到的RGD迭代呈线性收敛，而RGN最终实现Q-二次收敛。

    arXiv:2609.17048v1 Announce Type: cross  Abstract: We study nonconvex methods for matrix completion, the problem of recovering a low-rank matrix from a subset of its entries. Convex methods achieve sample complexity linear in the matrix dimension and the rank, up to logarithmic factors, whereas global guarantees for commonly used nonconvex methods require a higher polynomial dependence on the rank. We close this gap by analyzing Riemannian gradient descent (RGD) and Riemannian Gauss--Newton (RGN) methods. For an $n\times n$ matrix of rank $r$ with incoherence parameter $\mu$ and condition number $\kappa$, the two methods achieve exact recovery with high probability from $O(\mu nr\log n\log(n\kappa))$ and $O(\mu nr\log n\log(2\mu r\kappa))$ observations, respectively. The methods use a multiscale residual initialization, while the analysis simultaneously controls the spectral error and incoherence. The resulting RGD iterates converge linearly, whereas RGN eventually converges Q-quadrati
    
[^39]: 基于适配器库的组合式运动控制选项学习

    Learning Options for Compositional Motor Control with Adapter Banks

    [https://arxiv.org/abs/2609.17042](https://arxiv.org/abs/2609.17042)

    该论文将神经科学中“运动基元是共享循环网络低秩扰动”的理论转化为端到端可学习的新架构——共享循环核心加上由离散潜码选择的残差适配器库，适配器在无秩约束下自发涌现低秩特性，并通过冻结网络后优化的高层策略组合选项实现新颖分布外运动的泛化。

    

    学习灵活的运动基元是熟练运动控制的标志。近期的神经科学理论提出，运动基元可能被实现为共享循环网络的低秩扰动，但该系统如何被学习仍是悬而未决的问题。我们将这一原理转化为一种新颖的端到端运动技能学习架构：一个共享的循环核心由一组残差适配器进行调制，每个适配器由一个离散潜码选择。在闭环生物力学控制任务上训练后，尽管没有架构上的秩约束，适配器自发涌现出对循环动力学的低秩扰动，将任务表征置于共享核心网络的不同子空间中。一个基于所学选项的简单高层策略——在整体网络冻结的情况下进行优化——对低秩适配器进行排序组合，以产生新颖的分布外运动。我们展示了在……方面泛化到新颖运动序列的能力。

    arXiv:2609.17042v1 Announce Type: new  Abstract: Learning flexible motor primitives is a hallmark of skilled motor control. Recent neuroscience theory proposes that motor primitives may be implemented as low-rank perturbations of a shared recurrent network, but leaves open how such a system is learned. We translate this principle into a novel architecture for learning motor skills end-to-end: a shared recurrent core modulated by a bank of residual adapters, each selected by a discrete latent code. Trained on closed-loop biomechanical control, the adapters develop emergent low-rank perturbations of the recurrent dynamics despite no architectural rank constraint, placing task representations in disparate subspaces of the shared core network. A simple high-level policy over the learned options, optimized while the whole network is frozen, sequences the low-rank adapters to produce novel out-of-distribution movements. We demonstrate the ability to generalize to novel motor sequences within
    
[^40]: 分布式JEPA：一种用于能源预测的自监督框架

    Distributed JEPA: A Self-Supervised Framework for Energy Forecasting

    [https://arxiv.org/abs/2609.17029](https://arxiv.org/abs/2609.17029)

    该论文提出了一种分布式联合嵌入预测架构（JEPA），通过对被掩码时间片段的潜在表示进行自监督预测，并结合协方差与时间方差正则化来防止表示坍塌，从而在异构能源时间序列上实现可迁移的能源预测。

    

    传统的能源预测方案依赖于任务特定的监督信号和能源资产表示，这限制了模型的可迁移性以及捕捉异构资产间通用时间动态的能力。针对这一问题，我们提出了一种分布式联合嵌入预测架构（JEPA），用于从异构能源时间序列中进行自监督学习。该框架在共享嵌入空间中融合时间观测与上下文信息，并预测被掩码时间片段的潜在表示。为防止表示坍塌，训练过程将潜在空间预测目标与协方差正则化和时间方差正则化相结合。评估在能源消耗与发电数据集上进行，采用了数据退化场景，并与Transformer预测基线进行了比较。学习到的表示保持稳定（余弦相似度约为0.98；有效秩1……

    arXiv:2609.17029v1 Announce Type: cross  Abstract: Traditional energy forecasting solutions rely on task-specific supervision and energy asset representations, limiting transferability and the ability to capture general temporal dynamics across heterogeneous assets. We address this by proposing a distributed Joint Embedding Predictive Architecture (JEPA) for self-supervised learning from heterogeneous energy time-series. The framework predicts latent representations of masked temporal segments while integrating temporal observations and contextual information within a shared embedding space. To prevent representation collapse, training combines a latent-space predictive objective with covariance and temporal variance regularization. The evaluation was conducted on energy consumption and generation datasets under data-degradation scenarios and compared with a Transformer forecasting baseline. The learned representations remained stable (cosine similarity $\approx 0.98$; effective rank 1
    
[^41]: CLARE：基于稀疏性框架的可扩展类增量持续学习

    CLARE: Scalable Class-Incremental Continual Learning via a Sparsity-Based Framework

    [https://arxiv.org/abs/2609.17026](https://arxiv.org/abs/2609.17026)

    CLARE提出了一种两阶段稀疏微调框架，先通过稀疏性目标识别任务关键的参数掩码，再仅对掩码选定的参数进行约束微调，从而解决持续学习中多任务顺序训练的可扩展性瓶颈和灾难性遗忘问题。

    

    持续学习必须在学习新知识与保留已学知识之间取得平衡，以便从数据流中增量地学习任务而不发生灾难性遗忘。虽然利用预训练模型显著推动了持续学习的发展，但现有方法在按顺序训练大量任务时存在可扩展性瓶颈，由于任务间干扰和可塑性丧失而导致性能下降。受稀疏微调可达到与全量微调相当性能的证据启发，本文提出了一种新颖的稀疏驱动持续学习框架。我们的持续学习方法名为CLARE，分两个阶段运行：首先通过稀疏性诱导目标识别出稀疏的、任务关键的参数掩码，然后在掩码约束下进行微调，仅优化掩码所选择的参数。这种两阶段稀疏适配器机制使所有任务能够（原文摘要在此处截断）。

    arXiv:2609.17026v1 Announce Type: new  Abstract: Continual learning must balance the learning of new knowledge with the retention of previously learned knowledge to incrementally learn tasks from a data stream without catastrophic forgetting. While leveraging pretrained models has significantly advanced continual learning, existing methods exhibit a scalability bottleneck when trained sequentially on many tasks, suffering from performance degradation due to inter-task interference and loss of plasticity. Inspired by evidence that sparse fine-tuning achieves performance comparable to full fine-tuning, this paper presents a novel sparsity-driven continual learning framework. Our continual learning method, termed CLARE, operates in two stages: it first identifies a sparse, task-critical parameter mask via a sparsity-inducing objective, then performs mask-constrained fine-tuning by only optimizing parameters selected by the mask. This two-stage sparse adapter mechanism enables all tasks to
    
[^42]: 超越度量指标：一种以人为中心的网络流量分类语义验证框架

    Beyond Measurement Metrics: A Human-Centered Framework for Semantic Validation of Network Traffic Classification

    [https://arxiv.org/abs/2609.17014](https://arxiv.org/abs/2609.17014)

    本文提出了一种以人为中心的语义验证框架，通过融合数据、机器学习模型、可解释性技术、可视化和专家推理，来迭代验证网络流量分类模型是否真正学习到语义上有意义的模式，而不仅仅追求高预测性能指标。

    

    机器学习（ML）已成为网络流量分类的主流方法，实现了非常高的预测性能。然而，只有当模型学习到语义上有意义且可信的模式，而非利用虚假相关性时，模型才具有价值。传统的评估实践主要关注预测性能的评估。因此，模型是否依赖于语义上有意义的模式仍然未知。为了应对这些挑战，我们将知识生成框架适配到网络流量分类中。适配后的框架结合了数据、机器学习模型、可解释性、可视化和专家推理，以支持对模型行为和数据预处理的迭代式探索、验证与改进。该框架基于文献研究发现、基准数据集分析、基于XAI的流量分类实践经验以及专家反馈，为实践提供了支撑。

    arXiv:2609.17014v1 Announce Type: cross  Abstract: Machine learning (ML) has become the dominant approach for network traffic classification, achieving very high predictive performance. However, a model is only valuable if it learns semantically meaningful and trustworthy patterns rather than exploiting spurious correlations. Conventional evaluation practices predominantly assess predictive performance. Consequently, whether the model relies on semantically meaningful patterns remains unknown. To address these challenges, we adapt the knowledge generation framework for network traffic classification. The adapted framework combines data, ML models, explainability, visualization, and expert reasoning to support the iterative exploration, verification, and refinement of model behavior and data preprocessing. The framework is grounded in findings from the literature, benchmark dataset analyses, practical experience with XAI-based traffic classification, and expert feedback, providing pract
    
[^43]: 联邦图神经网络中的结构负迁移：诊断、因果探究以及差异感知缓解方法的局限性

    Structural Negative Transfer in Federated Graph Neural Networks: Diagnosis, Causal Investigation, and the Limits of Divergence-Aware Mitigation

    [https://arxiv.org/abs/2609.16977](https://arxiv.org/abs/2609.16977)

    本文诊断并因果性地研究了联邦图神经网络中因客户端图结构差异引发的结构负迁移现象——结构上非典型的客户端仅因加入联邦就损失过半可达成准确率，并揭示了差异感知缓解方法的能力极限。

    

    联邦学习使多个参与方无需汇集原始数据即可训练共享模型，而是通过交换本地训练的模型更新来实现。联邦平均假设当参与方数据大致相似时，对本地模型取平均是解决同一共享问题的合理方式。关于非独立同分布联邦学习的研究表明，这一假设能够抵御标签和特征分布上的差异。我们探究该假设能否经受住图神经网络特有的一种不同的压力：客户端图之间的差异不在标签或特征分布，而在结构本身，这要求相同的共享权重在根本不同的拓扑结构上运行。我们将由此产生的损害称为结构负迁移。在由真实引文网络和合成结构代理组成的联邦中，一个结构上非典型的客户端仅仅因为加入联邦，其可达到的准确率就损失了一半以上。在一个初始的六客户端联邦中，两个标签（原文截断）

    arXiv:2609.16977v1 Announce Type: new  Abstract: Federated learning lets multiple participants train a shared model without pooling raw data, by exchanging locally trained model updates instead. Federated averaging assumes that averaging local models is a reasonable way to solve one shared problem when participants' data are broadly similar. Work on non-IID federated learning has shown that this assumption can withstand differences in label and feature distributions. We ask whether it survives a different strain specific to graph neural networks, where client graphs differ not in label or feature distribution but in structure itself, requiring the same shared weights to operate over fundamentally different topologies. We call the resulting harm structural negative transfer. In a federation of real citation networks and synthetic structural proxies, a structurally atypical client lost more than half its achievable accuracy simply by joining. In an initial six-client federation, two labe
    
[^44]: 巧妙分割：用于处理效应异质性与偏差的可解释因果森林

    Splitting the Difference: Interpretable Causal Forests for Treatment Effect Heterogeneity and Bias

    [https://arxiv.org/abs/2609.16971](https://arxiv.org/abs/2609.16971)

    本文提出一种基于决策树和随机森林的可解释因果森林算法，仅通过修改分裂准则即可准确估计个体处理效应并揭示其异质性原因，无需双重机器学习或正交化等额外复杂步骤。

    

    在医学和营销等各个领域，准确预测个体处理效应具有重要意义。然而，仅实现可靠的预测往往不足以做出明智的决策；同样重要的是理解为什么某些个体的处理效应高于其他个体。为了应对预测和解释这一双重挑战，我们提出了一种基于决策树和随机森林的算法，用于估计个体处理效应。我们的算法非常简单：它的运行方式与标准随机森林完全相同，只是采用了不同的分裂准则，并且不需要额外的变通方法，例如广义随机森林中使用的双重机器学习或正交化技术。该算法能够处理具有不同处理倾向的观察性研究，而无需单独估计完整的倾向函数。这是通过结合两种分裂……（原文在此处截断）

    arXiv:2609.16971v1 Announce Type: cross  Abstract: In various fields, such as medicine and marketing, accurately predicting individual treatment effects holds significant promise. However, achieving reliable predictions alone is often insufficient for making informed decisions; it is equally important to understand why the treatment effect is higher for some individuals than for others. To address this two-fold challenge of prediction and interpretation, we introduce an algorithm based on decision trees and random forests for estimating individual treatment effects. Our algorithm is simple: it operates exactly like a standard random forest, but with a different splitting criterion, and requires no additional workarounds such as double machine learning or orthogonalization as used in Generalized random forests. It handles observational studies with varying treatment propensities without requiring separate estimation of the full propensity function. This is achieved by combining two spli
    
[^45]: HUMAD-NER：基于不确定性加权多任务学习的灾难推文数据集，用于联合命名实体识别与事件分类

    HUMAID-NER: A Disaster Tweet Dataset for Joint Named Entity Recognition and Event Classification via Uncertainty-Weighted Multitask Learning

    [https://arxiv.org/abs/2609.16964](https://arxiv.org/abs/2609.16964)

    该论文发布了首个基于HumAID基准的命名实体识别数据集HUMAID-NER（含6万条标注推文、约17.5万个实体跨度），并提出基于不确定性加权多任务学习的框架，联合执行灾难命名实体识别与人道主义事件分类。

    

    从社交媒体中快速提取结构化信息对于人道主义响应至关重要，然而现有的灾难推文资源主要提供文档级别的类别标签，缺乏跨度级别的实体标注。我们推出了HUMAID-NER，这是首个基于HumAID基准构建的命名实体识别数据集，包含60,000条英文灾难推文，采用BIO格式进行标注，涵盖十种面向实际运营的实体类型，共产生约175,000个标注实体跨度。标注通过一个可复现的三阶段混合流水线生成，该流水线结合了spaCy transformer模型、灾难领域的EntityRuler规则模式以及带有基于优先级的重叠解决机制的结构化正则表达式。我们还提出了一种联合多任务学习框架，使用共享的RoBERTa-large编码器同时执行灾难特定的命名实体识别和人道主义事件分类。为了减少联合训练期间的任务冲突……

    arXiv:2609.16964v1 Announce Type: new  Abstract: Rapid extraction of structured information from social media is important for humanitarian response, yet existing disaster tweet resources mainly provide document-level category labels without span-level entity annotations. We introduce HUMAID-NER, the first named entity recognition dataset built on the HumAID benchmark, containing 60,000 English disaster tweets annotated in BIO format across ten operationally motivated entity types and yielding approximately 175,000 labelled entity spans. Annotations are generated through a reproducible three-stage hybrid pipeline combining a spaCy transformer model, disaster-domain EntityRuler patterns, and structured regular expressions with priority-based overlap resolution. We also propose a joint multitask learning framework that performs disaster-specific named entity recognition and humanitarian event classification using a shared RoBERTa-large encoder. To reduce task conflict during joint traini
    
[^46]: 超越Token局部模仿：面向在线策略蒸馏的奖励兼容时序信用分配

    Beyond Token-Local Imitation: Reward-Compatible Temporal Credit Assignment for On-Policy Distillation

    [https://arxiv.org/abs/2609.16937](https://arxiv.org/abs/2609.16937)

    提出γOPD方法，通过折扣时序信用分配和奖励兼容的有界混合机制，在大语言模型在线策略蒸馏中统一了token级与序列级目标，在获得长时程监督的同时保证与时间跨度无关的优化稳定性。

    

    在线策略蒸馏已成为大语言模型后训练的一种有效方法，然而现有目标函数在目标保真度与优化稳定性之间存在权衡。Token级OPD提供稳定但局部的监督，而序列级OPD能够捕捉未来信用，但代价是依赖于时间跨度的方差。我们为这些方法建立了一个统一的时序信用视角，表明实际中的token级OPD可以解释为序列级反向KL梯度的一种时序近似。基于这一联系，我们提出了γOPD，该方法利用折扣时序信用分配来平衡长时程监督与优化稳定性，同时具有与时间跨度无关的方差界。我们进一步为γOPD开发了一种奖励兼容的有界混合（RBM）机制，通过平衡可验证的结果反馈与折扣后的OPD优势，从而超越……（原文摘要在此处截断）

    arXiv:2609.16937v1 Announce Type: cross  Abstract: On-policy distillation (OPD) has emerged as an effective approach for large language model post-training, yet existing objectives face a trade-off between objective fidelity and optimization stability. Token-level OPD provides stable but local supervision, whereas sequence-level OPD captures future credit at the cost of horizon-dependent variance. We establish a unified temporal-credit view of these formulations, showing that practical token-level OPD can be interpreted as a temporal approximation to the sequence-level reverse-KL gradient. Building on this connection, we propose $\gamma$OPD, which uses discounted temporal credit assignment to balance long-horizon supervision and optimization stability, while admitting a horizon-independent variance bound. We further develop a reward-compatible bounded mixing (RBM) mechanism for $\gamma\mathrm{OPD}$ that balances verifiable outcome feedback with the discounted OPD advantage to move beyo
    
[^47]: MedPCFM-TED：基于教师引导端点蒸馏的单步点云流匹配植入体生成方法

    MedPCFM-TED: One-Step Point Cloud Flow Matching for Implant Generation via Teacher-Guided Endpoint Distillation

    [https://arxiv.org/abs/2609.16934](https://arxiv.org/abs/2609.16934)

    提出教师引导端点蒸馏（TED）框架，实现颅骨植入体的单步点云流匹配生成，在SkullBreak数据集上取得最佳整体性能，并在单步方法中达到最优的倒角距离表现。

    

    颅骨植入体生成是医学影像领域中的一项重要任务。近期基于点云的生成方法，特别是流匹配方法，具有强大的重建质量和高效的采样能力，但在推理过程中仍需要多次神经网络函数求值。这限制了快速生成多个合理植入体候选方案的能力。我们提出了教师引导端点蒸馏，这是一个用于点云条件颅骨植入体生成的简洁单步蒸馏框架。TED利用教师引导的端点监督和几何匹配损失来训练单步学生模型，同时避免了显式的路径拉直过程。我们在SkullFix和SkullBreak基准数据集上对TED进行了评估。TED在SkullBreak数据集上取得了最佳的整体性能，在SkullFix上保持了竞争力，并在所比较的单步方法中提供了最优的倒角距离性能。此外，TED能够以大约单步的方式生成植入体。

    arXiv:2609.16934v1 Announce Type: cross  Abstract: Cranial implant generation is an important task in medical imaging. Recent point cloud based generative methods, particularly flow matching, offer strong reconstruction quality and efficient sampling, but still require multiple neural function evaluations during inference. This limits rapid generation of multiple plausible implant candidates. We propose Teacher-guided Endpoint Distillation (TED), a simple one-step distillation framework for conditional cranial implant generation on point clouds. TED trains a one-step student using teacher-guided endpoint supervision and geometric matching losses, while avoiding explicit path straightening. We evaluate TED on the SkullFix and SkullBreak benchmarks. TED achieves the best overall performance on the SkullBreak dataset, remains competitive on SkullFix, and provides the strongest Chamfer distance performance among the compared one-step methods. In addition, TED generates implants in approxim
    
[^48]: 当置信度信号不一致时：自回归语言模型中的局部置信度与全局置信度

    When Confidence Signals Disagree: Local and Global Confidence in Autoregressive Language Models

    [https://arxiv.org/abs/2609.16933](https://arxiv.org/abs/2609.16933)

    该研究发现在自回归语言模型中，局部置信度（贪婪选择答案标记的概率）与全局置信度（重复采样下的众数答案频率）仅呈弱相关，且全局置信度与答案正确性有中等关联而局部置信度几乎没有关联，表明这两种置信度信号不可互换使用。

    

    现代预测系统提供了多种通常被解释为置信度度量的量。然而，这些量可能概括了预测过程的不同方面。当置信度被用于评估可靠性或为下游的监督与控制提供依据时，这种区别至关重要。我们通过比较局部置信度（由贪婪选择的答案标记的概率定义）与全局置信度（由重复采样下的众数答案频率定义），研究了在自回归语言模型中不同的置信度读数在经验上是否可以互换。在MMLU和ARC Challenge数据集上，这两个信号仅呈弱相关，且它们与正确性的关联存在显著差异：全局置信度与正确性存在中等程度的关联，而局部置信度几乎没有关联。我们进一步测试了在问题层面上两个信号之间的不一致是否与……相关（摘要内容不完整）

    arXiv:2609.16933v1 Announce Type: cross  Abstract: Modern predictive systems expose multiple quantities that are commonly interpreted as measures of confidence. However, these quantities can summarize different aspects of the predictive process. This distinction matters when confidence is used to evaluate reliability or inform downstream oversight and control. We investigate whether different confidence readouts are empirically interchangeable in an autoregressive language model by comparing local confidence, defined from the probability of the greedy-selected answer token, with global confidence, defined from modal-answer frequency under repeated sampling. Across MMLU and ARC Challenge, the two signals are weakly correlated and differ substantially in their association with correctness: global confidence is moderately associated with correctness, whereas local confidence shows little association. We further test whether question-level disagreement between the signals is associated wit
    
[^49]: 基于变换低秩分位数曲面的因果发现

    Causal Discovery via Transformed Low-Rank Quantile Surfaces

    [https://arxiv.org/abs/2609.16931](https://arxiv.org/abs/2609.16931)

    本文提出低秩分位数曲面（LRQS）二元因果模型，证明在因果方向上变换后的条件分位数曲面具有低秩结构从而保证因果方向的通用可识别性，并提出了一种交替进行秩约束近似与单调变换保序估计的非参数因果发现方法。

    

    我们提出了低秩分位数曲面（LRQS），这是一种二元因果模型，在因果方向上，条件分位数曲面的未知单调变换具有低秩函数分解形式。LRQS 涵盖了位置-尺度噪声模型和后非线性异方差噪声模型，同时允许多个分位数基函数来表示超出位置-尺度效应的分布变化。我们证明了 LRQS 的通用可识别性：变换后的分位数曲面在因果方向上是低秩的，而在相应约束下，反方向的可表示性仅出现在例外的、经过精细调整的原因边缘分布中。我们提供了一个简单而强大的因果评分方法，采用非参数拟合程序，在离散化分位数曲面的秩约束近似与未知单调变换的保序估计之间交替进行。在具有更高秩分布变化的合成机制上的实验表……（原文摘要在此处被截断）

    arXiv:2609.16931v1 Announce Type: cross  Abstract: We propose Low-Rank Quantile Surfaces (LRQS), a bivariate causal model in which, in the causal direction, an unknown monotone transformation of the conditional quantile surface admits a low-rank functional decomposition. LRQS subsumes location-scale noise models and post-nonlinear heteroscedastic noise models, while allowing multiple quantile bases to represent changes beyond location-scale effects. We prove generic identifiability of LRQS: the transformed quantile surface is low rank in the causal direction, whereas reverse representability under the corresponding constraints occurs only for exceptional, fine-tuned cause marginals. We provide a simple-yet-powerful causal score using a nonparametric fitting procedure that alternates between rank-constrained approximation of discretized quantile surfaces and isotonic estimation of the unknown monotone transformation. Experiments on synthetic mechanisms with higher-rank distributional sh
    
[^50]: 将深度限价订单簿预测重新用于场景条件下的市场冲击建模

    Repurposing Deep Limit Order Book Forecasting for Scenario-Conditioned Market Impact Modeling

    [https://arxiv.org/abs/2609.16930](https://arxiv.org/abs/2609.16930)

    该论文提出一个模型无关框架，证明预训练的深度限价订单簿预测模型无需重新训练即可被重新用于场景条件下的市场冲击建模，与历史实际结果达到0.99的Spearman相关性和97.2%的方向一致性。

    

    深度限价订单簿预测模型能够捕捉非线性的市场动态，但其量化反事实订单簿消息影响的能力尚未得到系统性验证。我们引入了一个模型无关的框架，通过比较训练好的预测器在注入机械上有效的反事实消息前后的预测分布，来定义短期的模型隐含市场冲击。基于Transformer的预测器恢复了场景排名，其Spearman相关性达到0.99，并且在非中性场景中与实际历史结果有97.2%的方向一致性。观测级别的分析进一步表明，估计的冲击捕捉到了超越场景身份和事件前预测的增量序列相关变化。这些结果提供了证据，表明预训练的限价订单簿预测器无需重新训练即可被重新用于场景条件下的响应建模。

    arXiv:2609.16930v1 Announce Type: cross  Abstract: Deep Limit Order Book forecasting models capture nonlinear market dynamics, but their ability to quantify the effects of counterfactual order book messages has not been systematically validated. We introduce a model-agnostic framework that compares a trained forecaster's predictive distributions before and after injecting mechanically valid counterfactual messages, defining short-horizon model-implied market impact. A Transformer-based forecaster recovered scenario rankings with a Spearman correlation of 0.99 and 97.2% directional agreement with realized historical outcomes among non-neutral scenarios. Observation-level analysis further showed that estimated impacts captured incremental sequence-dependent variation beyond scenario identity and the pre-event forecast. These results provide evidence that pretrained Limit Order Book forecasters can be repurposed for scenario-conditioned response modeling without retraining.
    
[^51]: 使用文本优化将潜意识学习效应言语化

    Verbalizing Subliminal Learning Effects Using Text Optimization

    [https://arxiv.org/abs/2609.16927](https://arxiv.org/abs/2609.16927)

    本文提出SALVE方法，利用文本优化技术将蒸馏数据中隐藏传递的教师模型潜意识学习效应转化为清晰可读的提示词，为检测这一新型数据投毒风险提供了有效手段。

    

    潜意识学习是一种现象，即蒸馏数据集会传递教师模型的某些特征，而这些特征并未清晰地编码在数据集本身中。这给模型开发带来了新的挑战，并产生了数据投毒的新风险。在本工作中，我们使用文本优化来检测潜意识学习效应，并将其描述为清晰可读的提示词。来自提示教师模型的潜意识学习启发了我们的方法。我们观察到这是上下文蒸馏的一种特殊情况，并利用这一观察从理论上证明，提示式潜意识学习数据集能够唯一确定教师的提示词。我们将恢复该提示词的问题归结为一个文本优化问题，并提出了一种近似求解该问题的方法。我们的方法SALVE（搜索辅助潜在言语化，Search-Aided Latent Verbalization）通过优化软提示词、查询同一模型将其言语化为文本，并使用束搜索使言语化过程更加可靠。

    arXiv:2609.16927v1 Announce Type: cross  Abstract: Subliminal learning is a phenomenon in which a distillation dataset transmits traits from the teacher model that are not legibly encoded in the dataset itself. This introduces a new challenge for model development and creates new risks from data poisoning. In this work, we use text optimization to detect subliminal learning effects and describe them as legible prompts. Subliminal learning from a prompted teacher motivates our approach. We observe that this is a special case of context distillation and leverage this observation to show that, in theory, the prompted subliminal learning dataset identifies the teacher's prompt. We reduce recovering this prompt to a text optimization problem and present a method to approximately solve it. Our method, SALVE (Search-Aided Latent Verbalization), optimizes a soft prompt, queries the same model to verbalize it as text, and uses beam search to make the verbalization reliable. In the standard subl
    
[^52]: HyCoSeq：面向基因组序列的上下文双曲表示学习

    HyCoSeq: Contextual Hyperbolic Representation Learning for Genomic Sequences

    [https://arxiv.org/abs/2609.16925](https://arxiv.org/abs/2609.16925)

    HyCoSeq通过将加权洛伦兹残差聚合融入多曲率洛伦兹编码，并引入双向LSTM学习序列上下文关系，将局部双曲卷积编码扩展为序列级的上下文化基因组表示。

    

    双曲几何为基因组表示学习提供了一种天然的归纳偏置，但现有的双曲基因组模型主要使用洛伦兹卷积来学习局部序列表示，而其残差通路并未直接聚合完整的洛伦兹表示。我们提出了HyCoSeq，一个面向基因组序列的上下文双曲表示学习框架。HyCoSeq将加权洛伦兹残差聚合融入多曲率洛伦兹编码中，使完整的洛伦兹表示能够直接参与几何一致的局部聚合。它进一步引入了双向长短期记忆网络，整合来自序列两个方向的信息，以学习基因组序列中不同位置的局部表示之间的上下文关系，从而将局部双曲卷积编码扩展为序列级别的上下文化表示。

    arXiv:2609.16925v1 Announce Type: new  Abstract: Hyperbolic geometry provides a natural inductive bias for genomic representation learning, but existing hyperbolic genomic models primarily use Lorentz convolutions to learn local sequence representations, while their residual pathways do not directly aggregate full Lorentz representations. We propose HyCoSeq, a contextual hyperbolic representation learning framework for genomic sequences. HyCoSeq incorporates weighted Lorentzian residual aggregation into multi-curvature Lorentz encoding, allowing full Lorentz representations to participate directly in geometry-consistent local aggregation. It further introduces a bidirectional long short-term memory network that integrates information from both sequence directions to learn contextual relationships among local representations at different positions within a genomic sequence, thereby extending local hyperbolic convolutional encoding to sequence-level contextualized representations. Extens
    
[^53]: 基于合作驱动优化动力学的多智能体学习

    Multi-Agent Learning with Cooperation-Driven Optimization Dynamics

    [https://arxiv.org/abs/2609.16917](https://arxiv.org/abs/2609.16917)

    提出了一种多智能体合作机制，通过在训练过程中将多个小型神经网络的预测信息融入损失函数以直接影响权重更新，从而在保持性能的同时降低模型复杂度。

    

    通过反向传播训练的多层人工神经网络是许多更复杂分类算法的基本构件。它们的优势在于能够以任意精度逼近任何函数，而这一优势的代价是需要优化的大量参数。在本工作中，我们提出了一种合作机制，即在多个人工神经网络之间进行信息交换，目标是在保持性能的同时降低模型复杂度。更具体地说，我们考虑了多个“小”智能体（即所含参数少于参考的“大”模型），它们在训练过程中通过将各自的预测信息纳入损失函数来共享信息，从而直接影响权重更新。我们考虑了多种实现合作的策略，例如投票者模型、多数模型以及基于智能体对自身预测置信度的加权平均模型。

    arXiv:2609.16917v1 Announce Type: cross  Abstract: Multilayer Artificial Neural Networks trained via backpropagation are the basic blocks of many, more complex, classification algorithms. Their strength lies in the possibility of realizing, with arbitrary precision, any function. This result comes at the cost of the large number of involved parameters to be optimized. In this work, we propose a mechanism for cooperation, i.e., information exchange among several artificial neural networks, with the goal of reducing model complexity while maintaining performance. More precisely, we consider several "small" agents, i.e., containing fewer parameters than a reference "large" one, that during training share their predictions by incorporating this information into the loss function and thus directly influence weight updates. We consider several strategies for implementing cooperation, e.g., the voter model, majority model, and weighted average model based on an agent's confidence in its predi
    
[^54]: OptiPrime：通过协议-硬件协同设计优化私有推理

    OptiPrime: Optimizing Private Inference through Protocol-Hardware Co-design

    [https://arxiv.org/abs/2609.16898](https://arxiv.org/abs/2609.16898)

    OptiPrime通过协议-硬件协同设计，提出了一种新颖的卷积HE协议来大幅减少密文传输数量，从而突破网络通信瓶颈，显著提升基于同态加密和多方计算的私有DNN推理的端到端性能。

    

    基于混合同态加密（HE）和多方计算（MPC）的私有深度神经网络（DNN）推理能够以形式化保证的方式保护用户数据，但代价是HE带来的显著延迟开销。研究者已提出定制的HE加速器，并在单个HE操作上实现了数量级的加速。然而，当将商用HE加速器直接应用于最先进的HE-MPC框架时，我们观察到端到端性能提升有限。这是因为HE-MPC框架通常需要对每个HE操作的输入和输出密文进行无线传输，导致严重的网络通信瓶颈。为了克服这一挑战，我们提出了OptiPrime，一个用于高效私有DNN推理的协议-硬件协同优化框架。OptiPrime的核心是一种新颖的卷积HE协议，能够大幅减少需要传输的输出密文数量和m

    arXiv:2609.16898v1 Announce Type: cross  Abstract: Private deep neural network (DNN) inference based on hybrid homomorphic encryption (HE) and multi-party computation (MPC) can protect user data with a formal guarantee, but at the cost of significant latency overhead due to HE. Customized HE accelerators have been proposed and have achieved orders-of-magnitude speedup for individual HE operations. However, when directly applying a commercial HE accelerator to state-of-the-art HE-MPC frameworks, we observe only limited end-to-end performance gain. This is because HE-MPC frameworks often require wireless transmission of input and output ciphertexts for each HE operation, leading to a severe network communication bottleneck. To overcome this challenge, we introduce OptiPrime, a protocol-hardware co-optimization framework for efficient private DNN inference. OptiPrime features a novel HE protocol for convolutions that substantially reduces the number of transmitted output ciphertexts and m
    
[^55]: NeuroTS-Net：多模态MRI中儿童脑肿瘤的多类别语义分割

    NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI

    [https://arxiv.org/abs/2609.16873](https://arxiv.org/abs/2609.16873)

    NeuroTS-Net是一种用于儿童脑肿瘤多类别语义分割的三维卷积神经网络，通过双尺度细节流、自适应上下文选择和保留细节的多路径下采样，在无需外部数据或预训练权重的情况下超越了nnU-Net和MedNeXt等基线方法。

    

    儿童脑肿瘤是儿童癌症相关死亡的主要原因之一，其体积小、罕见且通常对比度低的亚区域使精确的手动勾画非常具有挑战性。因此，需要可靠的自动化分割来支持诊断、治疗规划和疗效评估。为此，我们提出了NeuroTS-Net，这是一种用于多类别语义分割的三维编码器-解码器卷积神经网络架构，它融合了双尺度原始细节流、自适应低分辨率上下文选择以及保留细节的多路径下采样。这些组件在有效建模更广泛的肿瘤上下文的同时，保留了精细的强度和边界信息。NeuroTS-Net仅在BraTS 2026儿童数据集上训练，未使用外部数据或预训练权重，并在相同实验协议下与nnU-Net和MedNeXt进行了对比评估。NeuroTS-Net优于基线方法……

    arXiv:2609.16873v1 Announce Type: cross  Abstract: Pediatric brain tumors are a leading cause of cancer-related mortality in children, and their small, rare, and often low-contrast subregions make accurate manual delineation challenging. Reliable automated segmentation is therefore needed to support diagnosis, treatment planning, and response assessment. Accordingly, we introduce NeuroTS-Net, a three-dimensional encoder-decoder convolutional neural network architecture for multi-class semantic segmentation that incorporates a dual-scale raw-detail stream, adaptive low-resolution context selection, and detail-preserving multipath downsampling. These components preserve fine intensity and boundary information while efficiently modeling broader tumor context. NeuroTS-Net was trained on the BraTS 2026 pediatric dataset without external data or pretrained weights and evaluated against nnU-Net and MedNeXt under the same experimental protocol. NeuroTS-Net outperformed the baseline methods, ac
    
[^56]: TEMPO：面向动态机器人操作的时间上下文学习

    TEMPO: Learning Temporal Context for Dynamic Robot Manipulation

    [https://arxiv.org/abs/2609.16864](https://arxiv.org/abs/2609.16864)

    该论文指出VLA模型在动态操作中的失败源于运动歧义和状态混叠两种表征缺陷而非模型容量不足，并提出TEMPO通过引入从冻结视频基础模型提取的运动摘要等时间上下文输入来增强预训练VLA模型。

    

    视觉-语言-动作（VLA）模型在准静态操作任务中取得了令人瞩目的表现，但在动态操作任务中表现不佳，因为它们在推理时仅基于单一观测进行决策。我们识别出导致这一局限性的两种表征缺陷：第一种是运动歧义，即单一观测不包含场景动态信息，因此无法预测运动物体的未来状态；第二种是状态混叠，即任务中不同阶段的视觉相似观测却需要执行不同的动作。我们论证了这些缺陷的持续存在与模型规模和推理延迟无关，这表明瓶颈在于缺失时间上下文而非模型容量。基于这一洞察，我们提出了TEMPO，通过两种时间输入来增强预训练的VLA模型：从冻结的视频基础模型中提取的运动摘要，用于解决运动歧义问题；以及紧凑的本体（原文在此截断）……

    arXiv:2609.16864v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models have achieved impressive performance in quasi-static manipulation, but struggle in dynamic manipulation tasks because they operate on a single observation at inference time. We identify two representational failures that underlie this limitation. The first is motion ambiguity, where a single observation does not include scene dynamics and therefore cannot anticipate the future state of moving objects. The second is state aliasing, where visually similar observations from different points in a task require different actions. We argue that these failures persist regardless of model scale and inference latency, showing that the bottleneck is missing temporal context rather than model capacity. Based on this insight, we propose TEMPO, which augments a pretrained VLA with two temporal inputs: a motion summary extracted from a frozen video foundation model to resolve motion ambiguity and a compact proprioc
    
[^57]: 测量手写天城文（梵文）识别的标注效率：四种预训练方式的样本复杂度曲线

    Measuring Annotation Efficiency for Handwritten Devanagari Recognition: Sample-Complexity Curves for Four Pretraining Regimes

    [https://arxiv.org/abs/2609.16859](https://arxiv.org/abs/2609.16859)

    本研究通过控制识别器与评估协议不变、仅改变标注预算和预训练方式的系统实验，量化了手写天城文识别达到可用精度所需的转录样本数量，并将预训练收益换算为可节省的标注工作量。

    

    训练手写文本识别系统需要词图像及其对应的转录文本，而这些转录文本需要人工制作。对于一种只有少数专家能够阅读的文字而言，人工转录是一种限制，因为训练出的模型本应节省这些专家的时间。由此产生一个相关问题：需要多少条转录文本才能使识别器变得有用，而预训练又能消除多少这类成本？本研究针对手写天城文直接测量了这一问题的答案。我们保持识别器、优化器和评估协议不变，仅改变用于微调的真实转录词的数量，涵盖从10到4000的九个标注预算档位和四种初始化方式，每个数据点使用六个随机种子。随后将所得曲线转换为标注等效项。监督式合成预训练在……时达到了0.50的字符错误率（CER）。（摘要原文在此处截断）

    arXiv:2609.16859v1 Announce Type: cross  Abstract: To train handwritten text recognition systems we need word images and their corresponding transcriptions, and these transcriptions are produced manually. For a script that can be read by only a small number of specialists, this manual transcription is a limitation, because the trained models are supposed to save the time of those same specialists. A relevant question therefore arises: how many transcriptions are needed before a recogniser becomes useful, and how much of that cost can pretraining remove? In this study the answer is measured directly for handwritten Devanagari. We keep the recogniser, optimiser and evaluation protocol the same and change only the number of real transcribed words used for fine-tuning across nine budgets from 10 to 4,000 and four initialisation regimes, with six seeds at every point. The resulting curves are then converted into annotation-equivalent terms. A CER of 0.50 is reached by supervised synthetic p
    
[^58]: 深度学习能否实现跨物理映射？

    Can Deep Learning Achieve Cross-Physics Mapping?

    [https://arxiv.org/abs/2609.16853](https://arxiv.org/abs/2609.16853)

    提出跨物理映射（CPM）算子学习框架，通过兼容潜在表示与无量纲缩放原理，使深度学习能够在扩散方程与波动方程等根本不同的物理域之间实现场的相互映射，并系统评估了七种神经算子架构的性能。

    

    深度学习能否翻译由根本不同的方程所支配的物理场？我们通过提出跨物理映射来回答这一问题，这是一个用于异构物理域之间映射的算子学习框架。我们通过兼容的潜在表示为这类映射制定了充分条件，并提出了一种无量纲缩放原理，该原理能够对齐源系统与目标系统的特征演化尺度，而无需假设二者的动力学等价性。作为代表性测试，配对的扩散场与波场分别由各自的抛物型和双曲型方程独立生成，同时共享相同的潜在几何结构、材料非均质性、激励和无量纲尺度。研究评估了七种架构——ResUNet、DeepONet、傅里叶、潜在、小波、U形以及Galerkin神经算子——用于扩散到波与波到扩散的双向映射。结果表（摘要在此处截断）……

    arXiv:2609.16853v1 Announce Type: new  Abstract: Can deep learning translate physical fields governed by fundamentally different equations? We address this question by introducing Cross-Physics Mapping (CPM), an operator-learning framework for mappings between heterogeneous physical domains. We formulate sufficient conditions for such mappings through compatible latent representations and propose a dimensionless scaling principle that aligns the characteristic evolution scales of the source and target systems without assuming their dynamical equivalence. As a representative test, paired diffusion and wave fields are generated independently from their respective parabolic and hyperbolic equations while sharing the same latent geometry, material heterogeneity, excitation, and dimensionless scale. Seven architectures-ResUNet, DeepONet, Fourier, latent, wavelet, U-shaped, and Galerkin neural operators-are evaluated for both diffusion-to-wave and wave-to-diffusion mappings. The results reve
    
[^59]: 高容量核联想记忆中稳定性边缘的信息几何自组织

    Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories

    [https://arxiv.org/abs/2609.16827](https://arxiv.org/abs/2609.16827)

    本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。

    

    基于核逻辑回归（KLR）的高容量联想记忆展现出卓越的存储能力与鲁棒性。先前的实证研究识别出了一个超参数区域，即“优化脊”，在该区域中吸引子的稳定性达到最大。然而，这一区域的几何本质以及到达该区域所需的优化动力学机制一直不明确。本文研究了采用KLR训练的Hopfield网络中参数空间的静态几何以及梯度下降（GD）的学习轨迹。利用Hessian矩阵的特征值谱，我们揭示了“优化脊”对应于位于秩1谱坍缩附近的一个相边界，它作为一个几何奇点，其主曲率被大幅放大。此外，我们证明了学习动力学表现出一种由稳定性边缘现象驱动的瞬态自稳定行为……

    arXiv:2609.16827v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit exceptional storage capabilities and robustness. Previous empirical studies identified a hyperparameter regime, the "Ridge of Optimization," where attractor stability is maximized. However, the geometric nature of this regime and the optimization dynamics required to reach it have remained unclear. In this paper, we investigate the static geometry of the parameter space and the learning trajectory of Gradient Descent (GD) in KLR-trained Hopfield networks. Using the eigenvalue spectrum of the Hessian, we reveal that the Ridge corresponds to a phase boundary located adjacent to a rank-1 spectral collapse, acting as a geometric singularity where the principal curvature is massively amplified. Furthermore, we demonstrate that the learning dynamics exhibit a transient self-stabilizing behavior driven by the Edge of Stability (EoS) phenomenon. Rather than seek
    
[^60]: 适应去中心化异构老虎机中与决策相关的非平稳性

    Adapting to Decision-Relevant Non-Stationarity in Decentralized Heterogeneous Bandits

    [https://arxiv.org/abs/2609.16824](https://arxiv.org/abs/2609.16824)

    提出了DRFC算法，通过来自所有智能体的新鲜平衡样本在网络层面比较臂并仅在公共最优臂真正变化时切换，使去中心化异构老虎机的动态遗憾界只取决于决策相关的变化次数而非局部变化次数。

    

    去中心化老虎机系统通常包含异构的智能体：即使网络层面的最优动作保持不变，各个体智能体的奖励也可能发生变化。当奖励在各智能体之间取平均时，这些局部变化可能会相互抵消，因此局部变化次数 $\Stloc$ 可能远大于最优公共臂的变化次数 $\Stdec$。我们提出了决策相关新鲜比较算法（DRFC），该算法利用来自所有智能体的新鲜的、平衡的样本在网络层面比较各臂，并且仅当新鲜的全局证据表明公共最优臂发生变化时才进行切换。我们证明了一个高概率动态遗憾界，其中不包含依赖于 $\Stloc$ 的自适应项，并证明任何算法都必须为识别真正的决策切换并通过通信图传播这些切换而付出代价。在一个独特的时间平均基准下，一个任意时刻有效的滑动窗口扩展能够处理渐进漂移；在合成数据上的实验……（摘要原文在此处截断）

    arXiv:2609.16824v1 Announce Type: new  Abstract: Decentralized bandit systems often contain heterogeneous agents: rewards can change at individual agents even when the best action for the network stays the same. These local changes may cancel when rewards are averaged across agents, so the number of local changes $\Stloc$ can be much larger than the number of changes in the best common arm $\Stdec$. We introduce Decision-Relevant Fresh Comparison (DRFC), which uses new, balanced samples from all agents to compare arms at the network level and switches only when fresh global evidence indicates that the common best arm has changed. We prove a high-probability dynamic regret bound with no adaptation term depending on $\Stloc$, and show that every algorithm must still pay for identifying genuine decision switches and propagating them through the communication graph. Under a distinct time-average benchmark, an anytime-valid sliding-window extension handles gradual drift; experiments on synt
    
[^61]: LCAP：基于种群信息的从少量输出探针进行的光子神经网络潜在芯片自适应

    LCAP: Population-Informed Latent Chip Adaptation from Few Output Probes for Photonic Neural Networks

    [https://arxiv.org/abs/2609.16823](https://arxiv.org/abs/2609.16823)

    LCAP框架通过从80个历史芯片学习共享的种群校正，并利用32个固定的无标签输出探针前馈式推断新芯片的潜在校正坐标，无需在目标芯片上优化即可实现光子神经网络的快速个性化校准，有效弥合仿真到硬件的差距。

    

    光子神经网络（PNN）能够提供高效的模拟推理，但在理想器件模型下优化的参数在芯片制造后可能出现性能退化，从而产生持久的仿真到硬件（sim-to-real）差距。当部署大量相同设计的芯片时，对每个器件从头进行校准会加剧这一成本。我们提出了基于探针的潜在芯片自适应（LCAP），这是一个种群信息框架，它将硬件自适应分解为可迁移的种群校正和由探针推断的潜在个性化两个部分。LCAP首先从80个历史芯片中学习一个共享的校正，然后从器件特定的细化中提取出低维校正空间。在部署时，32个固定的无标签输出探针即可推断出未知芯片的潜在校正坐标，从而实现前馈式个性化，而无需在目标器件上进行优化。在一个包含相位变化、分束器误差、量化和……（摘要在此处被截断）的三层64模式MZI模拟器上……

    arXiv:2609.16823v1 Announce Type: new  Abstract: Photonic neural networks (PNNs) offer efficient analog inference, but parameters optimized under ideal device models can degrade after fabrication, creating a persistent simulation-to-hardware (sim-to-real) gap. When many identically designed chips are deployed, calibrating each device from scratch compounds this cost. We propose Latent Chip Adaptation from Probes (LCAP), a population-informed framework that decomposes hardware adaptation into a transferable population correction and probe-inferred latent personalization. LCAP first learns a shared correction from 80 historical chips, then extracts a low-dimensional correction space from device-specific refinements. At deployment, 32 fixed unlabeled output probes infer an unseen chip's latent correction coordinates, enabling feed-forward personalization without target-device optimization. On a three-layer 64-mode MZI simulator with phase variation, beam-splitter errors, quantization, and
    
[^62]: ImpossibleRubrics：对作为奖励信号的生成式评分准则的极限压力测试

    ImpossibleRubrics: Stress-Testing Generated Rubrics as Reward Signals

    [https://arxiv.org/abs/2609.16816](https://arxiv.org/abs/2609.16816)

    该论文提出ImpossibleRubrics基准，利用169个不可能完成的任务对语言模型生成的评分准则进行对抗性压力测试，以检验其作为奖励信号时能否奖励诚实回答而非被对抗性回答钻空子。

    

    语言模型生成的评分准则（rubrics）越来越多地被用作基于评分准则的强化学习、LLM作为裁判的评估以及自动化评分中的奖励信号。只有当这些评分准则能够奖励诚实的回答、而非奖励针对其漏洞进行优化的对抗性回答时，它们才是可靠的。然而，这类评分准则在这种优化下的鲁棒性仍然知之甚少。我们隔离出最困难的情形：不可能完成的任务，即提示词迫使模型得出缺乏依据的结论，此时唯一诚实的回应就是承认任务的不可能性。我们提出了ImpossibleRubrics基准，包含169个横跨六种不可能性类别的任务，每个任务都配有一个可验证的神谕证书，明确界定诚实的回答可以声称和不可以声称的内容，同时还包含48个可回答的对照任务。与提供固定评分准则不同，ImpossibleRubrics提供的是任务环境和证书，允许评分准则在下游生成，然后进行对抗性测试（摘要在此处被截断）。

    arXiv:2609.16816v1 Announce Type: cross  Abstract: Language model-generated rubrics are increasingly used as reward signals for rubric-based reinforcement learning, LLM-as-a-judge evaluation, and automated grading. Such rubrics are reliable only if they reward honest answers over adversarial answers optimized to exploit them. Yet their robustness to such optimization remains poorly understood. We isolate the hardest regime: impossible tasks, where the prompt pressures the model toward an unsupported conclusion, so the only honest response is to acknowledge the impossibility. We introduce ImpossibleRubrics, a benchmark of 169 impossible tasks spanning six impossibility categories, each paired with a verifiable oracle certificate specifying what an honest answer may and may not claim, together with 48 answerable controls. Rather than providing fixed rubrics, ImpossibleRubrics provides task environments and certificates, allowing rubrics to be generated downstream and then adversarially t
    
[^63]: 学习动力学的几何学：梯度下降与自然梯度在优化脊上的对比

    Geometry of learning dynamics: Gradient descent versus natural gradient on the ridge of optimization

    [https://arxiv.org/abs/2609.16805](https://arxiv.org/abs/2609.16805)

    论文通过对KLR训练的Hopfield网络统计流形的几何分析，揭示了优化脊上的学习分为两个阶段：标准梯度下降因极端曲率沿振荡的非测地线路径前进，而自然梯度下降遵循理想测地线路径并完全克服了这些不稳定性。

    

    基于核逻辑回归（KLR）的高容量联想记忆呈现出一种“优化脊”特征，表现为极端的稳定性和高度偏斜的权重谱。然而，学习收敛到这一临界区域的动力学过程一直不清楚。本文对KLR训练的Hopfield网络在统计流形上的学习轨迹进行了几何分析。通过比较梯度下降（GD）与自然梯度下降（NGD）的路径，我们阐明了支配优化过程的机制。我们的分析揭示，在优化脊上的学习分为两个明显不同的阶段进行。我们证明了优化脊的极端曲率导致标准GD沿着高度振荡的非测地线路径前进。与之形成鲜明对比的是，NGD明确地校正了这种几何结构，遵循理想的测地线路径，完全克服了GD所面临的不稳定性。我们通过实验证明

    arXiv:2609.16805v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit a "Ridge of Optimization" characterized by extreme stability and a highly skewed weight spectrum. However, the dynamical process by which learning converges to this critical regime has remained unclear. This paper provides a geometric analysis of the learning trajectories on the statistical manifold of a KLR-trained Hopfield network. By comparing the paths of Gradient Descent (GD) and Natural Gradient Descent (NGD), we elucidate the mechanisms governing the optimization process. Our analysis reveals that learning on the Ridge proceeds in two distinct phases. We show that the extreme curvature of the Ridge causes standard GD to follow a highly oscillatory, non-geodesic path. In stark contrast, NGD explicitly corrects for this geometry, following the ideal geodesic path and completely overcoming the instabilities faced by GD. We demonstrate experimentally 
    
[^64]: SOTER：面向可穿戴人体生理信号的生成式时间序列基础模型

    SOTER: A Generative Time-Series Foundation Model for Wearable Human Physiological Signals

    [https://arxiv.org/abs/2609.16804](https://arxiv.org/abs/2609.16804)

    SOTER是一个面向可穿戴生理信号的生成式时间序列基础模型，通过空间特征感知骨干网络、PSD引导的混合专家层和神经控制微分方程解码器，在统一框架中建模多通道耦合、频谱特异性和连续时间动态。

    

    时间序列基础模型已经展现出强大的跨领域迁移能力，然而其常见的架构假设与可穿戴生理信号的特征仍然不相符——这些信号是多通道的、采样不规则、含噪声的，并受跨越不同频谱尺度的耦合连续时间动力学所支配。我们提出了SOTER，一个面向可穿戴生理时间序列的生成式基础模型，它在单一预训练框架内统一了跨通道耦合、频谱引导的专家特化以及连续时间潜在演化。SOTER结合了三个组件：一个建模信号间依赖关系的空间特征感知骨干网络；一个由功率谱密度（PSD）引导的混合专家层，它通过可检查的非学习规则将表示路由到与固定频谱带相关联的专家；以及一个神经控制微分方程解码器，支持在任意时间戳进行预测和插补。我们预……

    arXiv:2609.16804v1 Announce Type: cross  Abstract: Time-series foundation models have demonstrated strong cross-domain transfer, yet their common architectural assumptions remain poorly aligned with wearable physiological signals, which are multichannel, irregularly sampled, noisy, and governed by coupled continuous-time dynamics spanning distinct spectral scales. We present SOTER, a generative foundation model for wearable physiological time series that unifies cross-channel coupling, spectrum-guided expert specialization, and continuous-time latent evolution within a single pre-training framework. SOTER combines a spatial feature-aware backbone that models inter-signal dependencies, a power spectral density (PSD)-guided mixture-of-experts layer that routes representations to experts associated with fixed spectral bands through an inspectable, non-learned rule, and a neural controlled differential equation decoder that supports prediction and imputation at arbitrary timestamps. We pre
    
[^65]: 论随机多数投票的解体：从PAC-贝叶斯界到自约束算法

    On the disintegration of the stochastic majority vote: From PAC-Bayesian bounds to a self-bounding algorithm

    [https://arxiv.org/abs/2609.16803](https://arxiv.org/abs/2609.16803)

    本文提出了一种去随机化框架，将解积PAC-贝叶斯理论直接应用于多数投票权重向量空间，把随机多数投票的泛化保证转化为单一确定性多数投票的证书，并由此推导出两族高概率泛化界和一种自约束学习算法。

    

    加权多数投票是许多成功集成方法的核心。PAC-贝叶斯理论通过分析随机分类器的期望风险，为此类模型提供了紧密的泛化保证，而分析确定性多数投票的风险则依赖于替代界。为了避免这些替代方法，Zantedeschi等人（2021）引入了针对随机多数投票的保证，但由此产生的模型仍然是随机化的。在本文中，我们提出了一个针对随机多数投票的去随机化框架。为此，我们将解积PAC-贝叶斯理论的最新进展直接应用于多数投票权重向量空间，将随机保证转化为单一确定性多数投票的证书。我们推导出了两族高概率泛化界，同时涵盖了集成的数据无关构造和数据依赖构造，这自然地引出了一种自约束学习算法。

    arXiv:2609.16803v1 Announce Type: cross  Abstract: Weighted majority votes are central to many successful ensemble methods. PAC-Bayesian theory provides tight generalization guarantees for such models by analyzing the expected risk of stochastic classifiers, while analyzing the risk of deterministic majority votes relies on surrogate bounds. To avoid these surrogates, Zantedeschi et al. ( 2021) introduced guarantees for stochastic majority votes, but the resulting models remain randomized. In this paper, we propose a derandomization framework for stochastic majority votes. To do so, we apply recent advances in disintegrated PAC-Bayesian theory directly to the space of majority vote weight vectors, transforming stochastic guarantees into certificates for a single deterministic majority vote. We derive two families of high-probability generalization bounds, covering both data-independent and data-dependent constructions of the ensemble, which naturally lead to a self-bounding learning al
    
[^66]: 基于去时变信号平稳性学习的时间扭曲估计

    Time-warping estimation via stationarity-based learning of the de-warped signal

    [https://arxiv.org/abs/2609.16796](https://arxiv.org/abs/2609.16796)

    本文提出可训练的时间扭曲估计模型TWET，将时间扭曲估计转化为小波域的平稳化问题，通过可微平稳性准则和分层膨胀卷积架构实现端到端优化，在提高形变重建精度的同时大幅减少计算时间。

    

    时间扭曲估计是信号处理中的一个基本问题，在生物声学、雷达和生物医学分析等领域有广泛应用。本文介绍了一种可训练的时间扭曲估计模型（Time-Warping Estimation Trainable，TWET），用于从单次观测中估计时间扭曲函数。所提出的方法将时间扭曲估计表述为小波域中的平稳化问题，并利用分层膨胀卷积架构来估计时间扭曲函数。为实现端到端优化，本文引入了一种可微的平稳性准则。论文将TWET与现有方法进行了比较，实验结果表明，该方法的形变重建精度有所提高，同时计算时间显著减少，使该框架能够兼容低延迟应用。

    arXiv:2609.16796v1 Announce Type: cross  Abstract: Time-warping estimation is a fundamental problem in signal processing with applications in bioacoustics, radar, and biomedical analysis. This paper introduces a Time-Warping Estimation Trainable (TWET) model for estimating timewarping functions from a single observation. The proposed approach formulates time-warping estimation as a stationarization problem in the wavelet domain and leverages a hierarchical dilated convolutional architecture to estimate the time-warping functions. A differentiable stationarity criterion is introduced for end-to-end optimization. TWET is compared with existing approaches. Experimental results show improved deformation reconstruction accuracy together with significantly reduced computation time, making the framework compatible with low-latency applications.
    
[^67]: Noise2Noise再探：自监督去噪中训练对分布主导损失函数选择

    Noise2Noise Revisited: Training Pair Distributions Dominate Loss Choice in Self-Supervised Denoising

    [https://arxiv.org/abs/2609.16788](https://arxiv.org/abs/2609.16788)

    该研究否定了L1损失因参数稀疏性或最优解差异而在Noise2Noise去噪中优于L2的两个猜想，证明优势实际源于优化动力学，并指出训练对分布对去噪性能的影响超过损失函数的选择。

    

    Noise2Noise（N2N）通过独立受损的观测图像对来训练去噪器，从而摆脱对干净参考图像的依赖。我们对“L1损失为何在此场景下优于L2损失”的两个自然猜想进行了严格检验。首先，认为L1损失通过参数稀疏性带来鲁棒性的假设混淆了损失函数与Lasso正则化：显式添加Lasso惩罚确实产生了预期的稀疏性，却无法复现L1损失的跨噪声表现，且L1与L2训练所得的权重分布并无区别。其次，对于对称的信号后验分布，两种损失的总体最优解完全一致；对于集中的后验分布也近乎一致。因此，实测差异主要由优化动力学（具有有界影响的梯度）主导，我们通过梯度统计和污染目标训练对此进行了探究。在Kodak24数据集的五类合成噪声实验中，L1损失相比L2保持统计显著的优势（PSNR差距小于1 dB），且该优势在不同噪声类型间保持稳定……

    arXiv:2609.16788v1 Announce Type: new  Abstract: Noise2Noise (N2N) trains denoisers on pairs of independently corrupted observations, eliminating clean references. We stress-test two natural conjectures about why the L1 loss outperforms L2 here. First, the hypothesis that the L1 loss confers robustness via parameter sparsity confuses the loss with Lasso regularization: an explicit Lasso penalty produces the predicted sparsity yet fails to reproduce L1's cross-noise behavior, while L1- and L2-trained weight distributions are indistinguishable. Second, the population optima of the two losses coincide exactly for symmetric signal posteriors and nearly so for concentrated ones. Measured differences are therefore dominated by optimization dynamics (bounded-influence gradients), which we probe with gradient statistics and contaminated-target training. On Kodak24 with five synthetic noise families, the L1 loss holds a statistically significant edge over L2, below 1 dB PSNR, holding across thr
    
[^68]: TAME：面向涌现性失调的词元归因与掩码方法

    TAME: Token Attribution and Masking for Emergent misalignment

    [https://arxiv.org/abs/2609.16754](https://arxiv.org/abs/2609.16754)

    该论文提出TAME三阶段框架，通过词元归因、信号表征与归因引导的损失掩码，定位并因果验证了微调数据中承载涌现性失调信号的关键词元，并发现失调信号高度集中于前5%的词元中。

    

    在对齐的语言模型上用狭窄、有缺陷的数据进行微调，可能会诱发远超训练领域之外的有害行为，这种现象被称为涌现性失调。先前的研究已将EM定位于模型权重、激活值和训练文档中，但目前尚不清楚哪些训练词元承载了相关的微调信号。我们提出了TAME（Token Attribution and Masking for Emergent Misalignment，面向涌现性失调的词元归因与掩码），这是一个三阶段框架：词元归因通过已发布的LoRA适配器进行前向传播，量化微调更新在多大程度上提升了每个响应词元的似然；信号表征在高归因词元中寻找模式；因果验证则通过归因引导的损失掩码对这些模式进行检验。在已发布的EM生物体数据和一个包含6,849个样本的医疗建议数据划分上，归因呈现高度集中性（前5%的词元占据了32%的归因质量），并且在Llama模型中，医疗词汇的归因被耗减，而一类缺乏依据的语域的归因则被显著增强（原文摘要在此处截断）。

    arXiv:2609.16754v1 Announce Type: cross  Abstract: Fine-tuning an aligned language model on narrow, flawed data can induce harmful behavior far outside the training domain, known as emergent misalignment (EM). Prior work has localized EM in model weights, activations, and training documents, but it remains unclear which training tokens carry the relevant fine-tuning signal. We introduce TAME (Token Attribution and Masking for Emergent Misalignment), a three-stage framework: token attribution scores how strongly the fine-tuning update raises each response token's likelihood, using forward passes through a released LoRA adapter; signal characterization finds patterns among high-attribution tokens; and causal validation tests them by attribution-guided loss masking. On released EM organisms and a 6,849-example medical-advice split, attribution is concentrated (the top 5% of tokens hold 32% of the mass) and, in Llama, depleted for medical vocabulary but enriched for a register of unwarrant
    
[^69]: 基于乐观转移矩阵在一般和博弈中实现常数交换遗憾

    Constant Swap Regret in General-Sum Games via Optimistic Transition Matrices

    [https://arxiv.org/abs/2609.16751](https://arxiv.org/abs/2609.16751)

    该论文提出了一种基于乐观转移矩阵的确定性非耦合学习动态，首次在多人一般和博弈中实现了与时间范围无关的常数个体交换遗憾。

    

    对于有限多人一般和博弈，在全信息反馈条件下，我们给出了确定性的、非耦合的学习动态，实现了与时间范围 $T$ 无关的常数个体交换遗憾。当有 $n$ 个玩家且每个玩家至多有 $m$ 个动作时，在每个有限时间范围内，每个玩家的个体交换遗憾为 $O(\sqrt{n} m \log m \log^{5/2}(nm))$。每个玩家先预测偏差收益，然后利用这些预测来更新一个行随机转移矩阵，并按照其平稳分布采取行动。证明过程结合了利用平稳性的势函数论证与双尺度高阶预测分析，并使用有根树表示来处理偏差收益对平稳分布的非线性依赖。通过一个通用的公共前缀切换包装器，得到了一种对抗鲁棒的变体，该变体在至多差一个通用常数的前提下保持自博弈界，并保证个体交换遗憾至多……

    arXiv:2609.16751v1 Announce Type: cross  Abstract: We give deterministic and uncoupled learning dynamics for finite multiplayer general-sum games under full-information feedback that achieve constant individual swap regret, independent of the horizon $T$. With $n$ players and at most $m$ actions each, the individual swap regret of every player is $O(\sqrt{n} m \log m \log^{5/2}(nm))$ at every finite horizon. Each player predicts the deviation gains, then uses these predictions to update a row-stochastic transition matrix, and plays its stationary distribution. The proof combines a potential argument exploiting stationarity with a two-scale higher-order prediction analysis, using rooted-tree representations to handle the nonlinear dependence of deviation gains on the stationary distributions. An adversarially robust variant, obtained through a generic common-prefix switching wrapper, preserves the self-play bound up to a universal constant and guarantees individual swap regret at most $
    
[^70]: 从未存在的潜变量：对动作分块Transformer中CVAE消融实验的取证式重跑

    The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action Chunking Transformer

    [https://arxiv.org/abs/2609.16745](https://arxiv.org/abs/2609.16745)

    该论文在原始代码中取证式地重跑了ACT的CVAE编码器消融实验，发现原始论文中移除编码器导致成功率从35%骤降至2%的结果无法重现，且训练时长和检查点选择方式都可能逆转策略间的表现排序。

    

    动作分块Transformer（ACT）被广泛用于从人类演示中学习机器人操作。其条件变分自编码器包含一个编码器，旨在捕捉训练期间不同演示之间的差异。原始ACT论文报告称，移除该编码器会使两个基于人类演示的模拟任务的平均成功率从35%降至2%。我们在原始代码中重新运行了这一消融实验，并检验该结论是否依赖于具体实现或训练数据。已发表的成功率下降在我们的测试中并未重现，尽管较小的成功率增益或损失仍无法确定。为调查这一差异，我们改变了训练时长以及用于评估的检查点选择方式。两者都可能逆转哪个策略得分更高，但已发表的下降原因仍不清楚。仅凭成功率无法确定编码器是否提供了有助于策略重建演示动作的信息。

    arXiv:2609.16745v1 Announce Type: cross  Abstract: Action Chunking Transformers (ACT) are widely used to learn robot manipulation from demonstrations. Their conditional variational autoencoder includes an encoder meant to capture differences between demonstrations during training. The original ACT paper reported that encoder removal dropped the mean success rate from 35% to 2% on two simulated tasks with human demonstrations. We re-ran this ablation in the original code and checked whether the findings depend on the implementation or training data. The published drop does not reappear in our tests, although smaller gains or losses in success rate remain uncertain. To investigate the discrepancy, we varied training length and how checkpoints are selected for evaluation. Both can reverse which policy scores higher, but the published drop's cause remains unknown. Success rates alone leave open whether the encoder provides information that helps the policy reconstruct demonstrated actions.
    
[^71]: 电力电网故障检测与线路识别的机器学习方法系统评估

    A Systematic Evaluation of Machine Learning Methods for Fault Detection and Line Identification in Electrical Power Grids

    [https://arxiv.org/abs/2609.16744](https://arxiv.org/abs/2609.16744)

    本研究系统评估了多种机器学习模型在电网故障检测和故障线路识别中的有效性，以克服传统继电保护系统基于静态规则的局限性。

    

    可再生能源并入电网为故障检测和电网恢复机制的协调带来了复杂的挑战。基于静态规则和预定义阈值运行的传统继电保护系统不足以应对这些挑战，特别是在检测和隔离短路等故障方面。因此，应用于电网保护的传统方法在故障检测方面常常无法达到最佳性能，尤其是在遵守安全标准和选择性限制损害方面。最近的研究表明，基于机器学习（ML）的方法可以有效解决这些问题；然而，电网配置和分析窗口的差异阻碍了一致的比较评估。在本研究中，我们评估了各种机器学习模型在检测电气故障和精确定位故障线路方面的有效性。

    arXiv:2609.16744v1 Announce Type: new  Abstract: The integration of renewable energy sources into the electrical grid introduces complex challenges in fault detection and coordination of grid recovery mechanisms. Traditional relay protection systems, which operate based on static rules and predefined thresholds, are inadequate for addressing these challenges, particularly in detecting and isolating faults such as short circuits. Consequently, the conventional methodologies applied to electrical network protection frequently fail to achieve optimal performance in fault detection, especially in terms of adherence to safety standards and the selective limitation of damage. Recent research indicates that machine learning (ML)-based approaches can effectively tackle these issues; however, variations in grid configurations and analysis windows have impeded consistent comparative assessments. In this study, we assess the efficacy of various ML models in detecting electrical faults and pinpoin
    
[^72]: 传递校验和：一种面向边缘CNN推理的轻量级故障检测方法

    Carry-Through Checksum: A Lightweight Fault-Detection for CNN Inference at the Edge

    [https://arxiv.org/abs/2609.16742](https://arxiv.org/abs/2609.16742)

    提出了一种名为“传递校验和”的新方案，通过在卷积层中嵌入专用滤波器来计算并传播校验和，以极低开销实现嵌入式GPU上CNN推理的软错误检测。

    

    卷积神经网络越来越多地被部署在安全关键的边缘应用中，在这些应用里，软错误可能会在无感知的情况下破坏推理输出，进而导致不安全的决策。这类应用通常依赖于资源受限的嵌入式GPU，因此需要故障检测与缓解技术在增加最少计算、内存和延迟开销的同时，能够与标准GPU推理流水线无缝集成。现有的基于算法的容错技术依赖于矩阵扩充和逐操作校验和验证，会带来巨大的开销，这对于嵌入式GPU上的CNN推理来说是难以承受的。在本工作中，我们提出了传递校验和，这是一种面向嵌入式GPU上CNN推理软错误检测的全新方案。该方法将专用的传递滤波器嵌入到卷积层中，利用CNN自身的操作计算校验和，并通过推理过程进行传播。

    arXiv:2609.16742v1 Announce Type: cross  Abstract: Convolutional Neural Networks (CNNs) are increasingly deployed in safety-critical edge applications, where soft errors can silently corrupt inference outputs and lead to unsafe decisions. Such applications typically rely on resource-constrained embedded GPUs, requiring fault detection and mitigation techniques that add minimal compute, memory, and latency overhead while integrating seamlessly with the standard GPU inference pipeline. Existing algorithm-based fault tolerance techniques rely on matrix augmentation and per-operation checksum verification, imposing substantial overhead that is prohibitive for CNN inference on embedded GPUs.   In this work, we propose carry-through checksum, a fundamentally new scheme for soft-error detection in CNN inference on embedded GPUs. The method embeds dedicated carry-through filters into the convolutional layers, which compute a checksum from the CNN's own operations and propagate it through infer
    
[^73]: 面向潮流计算、最优潮流和状态估计的统一异构图神经网络求解器

    Unified Heterogeneous Graph Neural Network solver for Power Flow, Optimal Power Flow and State Estimation

    [https://arxiv.org/abs/2609.16738](https://arxiv.org/abs/2609.16738)

    该论文提出一个统一的异构残差门控图卷积网络，用单一共享骨干网络同时解决潮流计算、最优潮流和状态估计三个电力系统问题，精度媲美任务专用GNN求解器，并对未见过的负载水平和拓扑结构保持稳健。

    

    潮流计算（PF）、最优潮流（OPF）和状态估计（SE）是电力系统分析中的基础问题，但求解这些问题的计算成本很高。图神经网络（GNN）已被提出作为快速替代方案，然而现有的求解器每次只针对单一问题进行训练，产生的模型适用范围狭窄，必须为每个新任务重新构建。我们提出了一种更通用的方法：一个单一的异构残差门控图卷积网络，通过一个共享骨干网络同时解决这三个问题。该模型并非学习单一映射，而是学习电力网络行为的可复用表示，并从中分别估计PF、OPF和SE。该共享模型在多种拓扑结构和负载条件下对这三个问题进行联合训练，并在IEEE 14节点和118节点系统上进行评估。结果表明，该模型达到了任务专用GNN求解器的精度，并且在未见过的负载水平和拓扑结构上依然保持稳健。

    arXiv:2609.16738v1 Announce Type: cross  Abstract: Power Flow (PF), Optimal Power Flow (OPF), and State Estimation (SE) are fundamental problems in power system analysis, but solving them is computationally expensive. Graph Neural Networks (GNNs) have been proposed as fast surrogates, yet existing solvers are trained for a single problem at a time, producing narrow models that must be rebuilt for each new task.   We propose a more general approach: a single Heterogeneous Residual Gated Graph Convolutional Network that solves all three problems with one shared backbone. Rather than learning one mapping, the model learns a reusable representation of how the network behaves, from which PF, OPF, and SE can each be estimated. Trained jointly on the three problems across diverse topologies and loading conditions, and evaluated on the IEEE 14-bus and 118-bus systems, the shared model matches the accuracy of task-specific GNN solvers and stays robust on unseen loading levels and topologies.   
    
[^74]: 看见重要之物：视觉线索引导的视频规划实现可泛化的机器人导航

    Seeing What Matters: Visual Cue Guided Video Planning for Generalizable Robot Navigation

    [https://arxiv.org/abs/2609.16737](https://arxiv.org/abs/2609.16737)

    CueNav通过鸟瞰图和机器人身体等视觉线索引导视频规划，并利用逆动力学模型将视频计划转换为机器人动作，使迷宫导航成功率提升近2倍。

    

    生成式视频模型可以通过预测未来观测作为视频规划，成为机器人导航的有前景的骨干网络。近期的方法通常将视频规划建立在短时程引导之上，并通过场景重建来恢复几何路径点，而对更长时程的规划以及精确的视频到动作转换探索较少。我们提出了CueNav，一个基于视频模型的导航框架，它将视觉线索引导的视频规划与特定具身形态的逆动力学模型（IDM）相结合。作为视觉线索，我们使用鸟瞰图（BEV）地图来传达全局任务上下文，并在自我中心观测中保留部分机器人身体以呈现具身形态上下文。这些线索引导视频规划器，而IDM则将从视频规划中提取的密集光流场转换为机器人动作。凭借视觉线索编码的全局任务上下文，CueNav在迷宫导航任务中的成功率比不使用该线索的规划方法高出近2倍。

    arXiv:2609.16737v1 Announce Type: cross  Abstract: Generative video models can serve as a promising backbone for robot navigation by predicting future observations as video plans. Recent approaches often condition video planning on short-horizon guidance and recover geometric waypoints through scene reconstruction, leaving longer-horizon planning and precise video-to-action translation less explored. We present CueNav, a video model-based navigation framework combining visual cue guided video planning with an embodiment-specific Inverse-Dynamics Model (IDM). As visual cues, we use a Bird's-Eye View (BEV) map to convey global task context and retain part of the robot body in the egocentric observation to expose embodiment context. These cues guide the video planner, while the IDM translates dense flow fields extracted from the video plan into robot actions. With the visual cue encoding global task context, CueNav achieves nearly 2x higher success in maze navigation than planning without
    
[^75]: 连续时间机器学习：统一的数学视角

    Continuous-Time Machine Learning: A Unified Mathematical Perspective

    [https://arxiv.org/abs/2609.16710](https://arxiv.org/abs/2609.16710)

    本综述通过统一的数学分类框架，基于向量场参数化、随机性、记忆机制和离散化等核心数学表述，将分散于不同研究社区的连续时间机器学习各主要分支联系起来，并系统比较了它们的训练方法与设计权衡。

    

    连续时间（CT）机器学习已成为一种将时间动态建模为连续过程的原则性框架，尤其适用于观测数据在任意时间点采样或跨越长时间范围的情况。然而，连续时间机器学习的主要分支已在不同的研究社区中各自成熟发展，导致它们之间的数学关系与设计权衡尚未得到充分刻画。在这篇综述中，我们基于各方法家族的底层数学表述构建了一个分类体系，从而建立起对连续时间机器学习主要分支的统一且概念驱动的视角。我们提出了一个规范化的数学表述，通过向量场参数化、随机性、记忆机制和离散化等方面的不同架构选择，将这些方法家族相互关联。我们比较了各家族的训练算法、优化策略和失效模式，并着重分析了它们之间的权衡。

    arXiv:2609.16710v1 Announce Type: cross  Abstract: Continuous-time (CT) machine learning has emerged as a principled framework for modeling temporal dynamics as a continuous process, particularly when observations are sampled at arbitrary time points or span long-range horizons. However, major branches of CT machine learning have matured in separate research communities, leaving their mathematical relationships and design trade-offs insufficiently characterized. In this survey, we develop a unified, concept-driven view of major CT machine learning branches through a taxonomy that organizes families according to their underlying base mathematical formulations. We present a canonical mathematical formulation that relates these families through different architectural choices of vector-field parameterization, stochasticity, memory mechanisms, and discretization. We compare training algorithms, optimization strategies, and failure modes, highlighting the trade-offs across families. We furt
    
[^76]: Weave：从人-物交互中学习全身灵巧移动操作

    Weave: Learning Whole-Body Dexterous Loco-Manipulation from Human-Object Interactions

    [https://arxiv.org/abs/2609.16683](https://arxiv.org/abs/2609.16683)

    Weave提出一个统一框架，通过接触感知的重定向和接触与几何感知策略，将人类示范转化为可执行的全身控制，实现人形机器人从人-物交互中学习协调平衡、移动与灵巧手指操作的全身灵巧移动操作，在九个物体上取得92.5%的成功率。

    

    学习人形机器人与物体的交互需要协调全身平衡、移动以及灵巧的手部接触，以同时控制机器人与物体的运动。人类示范提供了协调交互的范例，但将这些行为迁移到人形机器人上，需要学习如何在不同具身形态和动力学条件下建立并维持有效接触。我们提出了Weave，这是一个从捕获的人类示范中学习全身灵巧人形-物体交互的统一框架。Weave首先通过接触感知的重定向和接近动作补全，将捕获的人-物交互转换为可执行的机器人-物体参考轨迹。其核心是一个接触与几何感知的策略，能够在多个物体和多个交互序列上联合控制29个身体关节和12个驱动手指关节。在九个物体上的评估显示，在训练过的交互上达到了92.5%的成功率，并且无需……

    arXiv:2609.16683v1 Announce Type: cross  Abstract: Learning humanoid-object interaction requires coordinating whole-body balance, locomotion, and dexterous hand contact to control both robot and object motion. Human demonstrations provide examples of coordinated interaction, but transferring these behaviors to humanoid robots requires learning how to establish and maintain effective contacts under different embodiments and dynamics. We present Weave, a unified framework for learning whole-body dexterous humanoid-object interaction from captured human demonstrations. Weave first converts captured human-object interactions into executable robot-object references through contact-aware retargeting and approach-motion completion. At its core is a contact- and geometry-aware policy that jointly commands 29 body joints and 12 actuated finger joints across multiple objects and interaction sequences. Evaluation across nine objects yields a 92.5% success rate on trained interactions and, without
    
[^77]: 正确的方向，错误的步长：循环Transformer中有限步失败现象的几何分析

    Right Direction, Wrong Step: Geometric Analysis of Finite-Step Failure in Looped Transformers

    [https://arxiv.org/abs/2609.16665](https://arxiv.org/abs/2609.16665)

    该论文揭示了循环Transformer中的“有限步失败”现象——局部改进的更新方向在完整步长下反而产生有害更新，并通过路径曲率分解和局部二次模型来预测有效步长规模及其理论误差界。

    

    循环Transformer通过复用共享层进行迭代潜在推理，为测试时扩展提供了一种参数高效的途径。然而，额外的迭代可能会降低对参考答案的支持，目前尚不清楚这究竟是由于更新方向在局部无益，还是完整位移移动得过远。我们通过分析参考效用——即衡量这种支持的指标——沿模型自身更新方向的变化来研究这一区别，具体做法是改变提供给读出的位移比例。这揭示了有限步失败现象：一个局部改进的方向在完整更新时却产生了有害的结果。路径曲率分解刻画了初始进展如何丢失，而局部二次模型则预测了完整步长的增益和有效的步长规模。基于累积曲率变化的界刻画了这些预测的近似误差。在两个模型家族上的实验揭示了这种分离……

    arXiv:2609.16665v1 Announce Type: cross  Abstract: Looped Transformers offer a parameter-efficient route to test-time scaling by reusing shared layers for iterative latent reasoning. However, additional iterations can reduce support for a reference answer, leaving unclear whether an update's direction is locally unhelpful or its full displacement moves too far. We study this distinction by analysing reference utility, which measures this support, along the model's own update direction, varying the fraction of the proposed displacement supplied to the readout. This reveals finite-step failures in which a locally improving direction produces a harmful full update. A pathwise curvature decomposition characterises how initial progress is lost, while a local quadratic model predicts full-step gains and useful step scales. Bounds based on accumulated curvature variation characterise the approximation error of these predictions. Experiments across two model families reveal this separation on 
    
[^78]: GrowMTP：强化学习能否培育出自己的草稿头？

    GrowMTP: Can RL Grow Its Own Draft Head?

    [https://arxiv.org/abs/2609.16648](https://arxiv.org/abs/2609.16648)

    提出GrowMTP，利用RL训练自身的采样分布与验证监督信号，在RL循环内从零在线训练推测解码草稿头，免去额外的预训练或预热成本，且草稿头更新与策略主干完全解耦。

    

    强化学习（RL）后训练驱动着大语言模型的前沿能力，但其运行时间主要被自回归的采样生成所占据。推测解码是缓解这一瓶颈的成熟方案，但现有的草稿头必须在RL之前进行预训练或预热，这在被加速的RL运行之外引入了可观的训练成本。我们观察到，RL训练本身恰好提供了在线草稿头训练所需的两个条件：其采样分布远窄于预训练分布，且其验证步骤会持续产生与该分布对齐的监督信号。基于这些观察，我们提出了GrowMTP，它利用这些监督信号完全在RL循环内部从零开始训练草稿头，且所有草稿头的更新都与策略主干网络相分离（不会反向传播干扰主干）。在Qwen3-4B（无草稿头）、MiMo-7B-SFT（弱草稿头）和Qwen3.5-4B-Base（强草稿头）上，GrowMTP取得了……（原文摘要在此处截断）

    arXiv:2609.16648v1 Announce Type: cross  Abstract: Reinforcement learning (RL) post-training drives the frontier capabilities of large language models, with its wall-clock dominated by autoregressive rollout generation. Speculative decoding is an established remedy for this bottleneck, but existing draft heads must be pretrained or warmed up before RL, introducing substantial training cost outside the RL run to be accelerated. We observe that RL training itself provides both conditions required for online draft-head training: its rollout distribution is far narrower than that of pretraining, and its verification step continuously produces supervision signals aligned with this distribution. Building on these observations, we propose GrowMTP, which uses this supervision to train a draft head from scratch entirely within the RL loop, with all head updates detached from the policy backbone. On Qwen3-4B (no draft head), MiMo-7B-SFT (weak head), and Qwen3.5-4B-Base (strong head), GrowMTP ach
    
[^79]: 知识迁移参数可以被学习吗？面向高效机器人视觉的LePoKet

    Can Knowledge Transfer Parameters Be Learned? LePoKet for Efficient Robotic Vision

    [https://arxiv.org/abs/2609.16637](https://arxiv.org/abs/2609.16637)

    提出了LePoKet框架，通过可学习遗传注意力算子将知识迁移的交互参数与子网络联合优化，把知识继承直接嵌入前向计算，无需辅助蒸馏损失或温度缩放即可获得高效的紧凑机器人视觉网络。

    

    高效感知是在计算、内存和延迟预算受限条件下运行的机器人系统的核心。从大型预训练模型进行知识迁移为构建更强的紧凑感知网络提供了一条实用途径，但现有方法通常依赖于固定的蒸馏目标或人工设计的交互机制。在遗传知识迁移（HKT）的基础上，我们提出了LePoKet（知识迁移的可学习参数优化），这是一种将知识继承直接嵌入前向计算的结构化迁移框架。LePoKet引入了一种分块的“提取-变换-混合”（Extract-Transform-Mix）接口，其交互参数通过可学习遗传注意力（LGA）算子与子网络进行联合优化，无需辅助蒸馏损失或温度缩放。我们首先在使用ResNet父-子网络对的CIFAR-10和CIFAR-100数据集上对该机制进行了表征，获得了相对……（摘要在此处截断）

    arXiv:2609.16637v1 Announce Type: cross  Abstract: Efficient perception is central to robotic systems operating under constrained computation, memory, and latency budgets. Knowledge transfer from larger pretrained models offers a practical route to stronger compact perception networks, but existing approaches commonly rely on fixed distillation objectives or manually designed interaction mechanisms. Building on Hereditary Knowledge Transfer (HKT), we propose LePoKet (Learnable Parameter Optimization for Knowledge Transfer), a structural transfer framework that embeds knowledge inheritance directly into the forward computation. LePoKet introduces a block-wise Extract-Transform-Mix interface whose interaction parameters are optimized jointly with the child network through a Learnable Genetic Attention (LGA) operator, without auxiliary distillation losses or temperature scaling. We first characterize the mechanism on CIFAR-10 and CIFAR-100 using ResNet parent-child pairs, obtaining relati
    
[^80]: AURA：面向大规模生产级推荐系统的智能体诊断与优化

    AURA: Agentic Diagnosis and Refinement for Production Recommender Systems at Scale

    [https://arxiv.org/abs/2609.16625](https://arxiv.org/abs/2609.16625)

    本文提出AURA，一个端到端的AI智能体系统，能够对大规模生产级推荐系统进行定性评估，提供超越传统汇总指标的可操作诊断与改进方案。

    

    推荐系统为何以及如何辜负其所服务的用户？通常情况下，从业者只能依靠利益相关团队的反馈、领域专业知识以及数据分析洞察相结合的方式来改进他们的算法。然而，推荐结果对终端用户在何处以及如何表现良好或不佳的细微差别，很难从汇总的定量指标中辨别出来。这些指标只能提供一个高层次且不完整的画面，要进一步细化推荐质量及其模式，则需要结合领域理解和客观性进行大规模的推理。我们针对这一复杂的难题，描述了一种利用最新AI智能体技术进展、为生产级推荐系统提供可操作诊断与改进的方法及实现。我们提出了AURA（推荐算法的智能体理解与优化），这是一个端到端的智能体系统，能够对生产推荐系统执行定性评估……

    arXiv:2609.16625v1 Announce Type: cross  Abstract: How and why does a recommender system fail the users it serves? Oftentimes, practitioners are left to improve their algorithms based on a combination of feedback from stakeholder teams, domain expertise, and insights from data analyses. Yet the nuances of how and where recommendations perform well or poorly for end users are difficult to discern from aggregate quantitative metrics. Whereas these metrics provide a high-level and incomplete picture, further granularity into the quality of recommendations and their patterns requires reasoning with domain understanding and objectivity, at scale. We contemplate this complex conundrum and describe a method and implementation that uses the latest AI agentic advances to provide actionable diagnoses and improvements for production recommender systems. We present AURA (Agentic Understanding and Refinement of recommender Algorithms), an end-to-end agentic system that performs qualitative evaluati
    
[^81]: 构造即稳定：用于长时程偏微分方程预测的变分潜马尔可夫算子

    Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction

    [https://arxiv.org/abs/2609.16621](https://arxiv.org/abs/2609.16621)

    提出变分自编码马尔可夫算子（VAMO），通过函数空间上的潜马尔可夫动力学、谱几何结构与变分转移对齐来正则化自回归误差传播，实现稳定的长时程偏微分方程预测。

    

    神经偏微分方程求解器为时变物理系统提供了高效的代理模型，但在长时程上的自回归预测仍然具有挑战性，因为局部误差会引起分布偏移，并在递归部署下不断累积。我们针对这一问题开发了一种变分方法，通过引入潜马尔可夫动力学，将物理状态表示为潜分布，并通过概率转移进行演化。该框架直接在函数空间上构建，并专门针对函数型高斯模型，其中结构化的潜扰动诱导出谱几何结构，变分转移对齐则对学习到的动力学进行正则化。我们进一步分析了这些机制如何影响自回归误差传播，为变分训练与长时程预测之间建立了理论联系。我们将该框架实例化为变分自编码马尔可夫算子（VAMO），

    arXiv:2609.16621v1 Announce Type: new  Abstract: Neural PDE solvers provide efficient surrogates for time-dependent physical systems, but autoregressive prediction over long horizons remains challenging because local errors can induce distribution shift and accumulate under recursive deployment. We develop a variational approach to this problem by introducing latent Markov dynamics in which physical states are represented by latent distributions and evolved through probabilistic transitions. The framework is formulated directly on function spaces and specialized to functional Gaussian models, where structured latent perturbations induce a spectral geometry and variational transition alignment regularizes the learned dynamics. We further analyze how these mechanisms affect autoregressive error propagation, providing a theoretical connection between variational training and long-horizon prediction. We instantiate the framework as the Variational Autoencoding Markov Operator (VAMO), which
    
[^82]: KV缓存驱逐下的分歧时机与累积分歧

    Divergence Timing and Cumulative Disagreement under KV-Cache Eviction

    [https://arxiv.org/abs/2609.16617](https://arxiv.org/abs/2609.16617)

    该论文建立了KV缓存驱逐引发自回归生成分歧的理论框架，将累积token不匹配精确分解为首次分歧贡献与分歧后暴露量的乘积形式，并通过条件蒙特卡洛无偏估计及Llama-3.1与Qwen2.5模型实验验证了不同缓存压缩策略在分歧时机上的差异。

    

    KV缓存驱逐会扰动控制自回归生成的条件token分布。我们研究首次分歧时机及随后的token不匹配如何决定累积分歧。在指定的逐步最大耦合下，我们推导出精确分解：期望不匹配比例等于首次不匹配贡献，加上分歧后暴露时间乘以其不匹配率。通过在无约束自回归核对上的显式构造，我们实现了与有限分歧对齐观察窗口相容的风险尖锐区间。残差分支条件蒙特卡洛方法对发生、占据及窗口/尾部贡献提供了无偏联合估计，并对总token损失具有逐次重复的方差优势。来自Meta-Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的完整轨迹表明，50%保留率的SnapKV比SnapKV-512或其他方法更晚且更少地进入分歧状态。

    arXiv:2609.16617v1 Announce Type: new  Abstract: KV-cache eviction perturbs the conditional token distributions governing autoregressive generation. We investigate how first-divergence timing and subsequent token mismatch determine cumulative disagreement. We derive an exact decomposition under a specified stepwise maximal coupling: the expected mismatch fraction equals a first-mismatch contribution plus post-divergence exposure multiplied by its mismatch rate. An explicit construction over unrestricted autoregressive kernel pairs realizes the sharp interval of risks compatible with a finite divergence-aligned observation window. Residual-branch conditional Monte Carlo provides unbiased joint estimates of occurrence, occupation, and window/tail contributions, with per-replicate variance dominance for total token loss. Complete trajectories from Meta-Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct show that SnapKV at 50% retention enters divergence later and less often than SnapKV-512 or 
    
[^83]: 一种适应已学习多变量结构的加权核函数近似方法

    A Weighted Kernel Method for Approximation that Adapts to Learned Multivariable Structure

    [https://arxiv.org/abs/2609.16606](https://arxiv.org/abs/2609.16606)

    提出了总敏感度核（TSK）方法，通过加权ANOVA核族学习多变量结构，并借助最小范数RKHS的选择从有限数据中唯一确定各输入的敏感度因子，从而实现对黑箱函数的自适应近似。

    

    从有限数据中近似多变量黑箱函数的输入输出行为极具挑战性，尤其是当对其输入的重要性及输入间相互作用一无所知时。我们引入了总敏感度核（TSKs），这是一种基于加权ANOVA核族的方法，能够学习并适应这种多变量结构。TSKs通过每个输入的因子来参数化目标函数各多变量分量上的权重。我们提出通过选择使目标函数具有最小范数的再生核希尔伯特空间（RKHS），直接从函数评估值中学习这些因子。在适当条件下，我们证明了这一范数最小化问题存在唯一解，并建立了基于最小范数插值的有限数据方法的一致性。学习到的TSK因子刻画了各个输入在相互作用和主效应中的参与程度，提供了一种依赖核的

    arXiv:2609.16606v1 Announce Type: new  Abstract: Approximating the input-output behavior of a multivariable black-box function from limited data is challenging when blind to the importance of its inputs and their interactions. We introduce total sensitivity kernels (TSKs), a method based on families of weighted ANOVA kernels that learn and adapt to this multivariable structure. TSKs parameterize the weights on each multivariable component of the target function by factors for each input. We propose learning these factors directly from function evaluations by selecting the reproducing kernel Hilbert space (RKHS) in which the target function has minimum norm. Under suitable conditions, we show that this norm-minimization problem admits a unique solution, and we establish consistency of a finite-data formulation based on minimum-norm interpolation. The learned TSK factors characterize the participation of individual inputs across interactions and main effects, providing a kernel-dependent
    
[^84]: 通过精确分布式样条合并从碎片化观测中恢复物理参数

    Recovering Physical Parameters from Fragmented Observations via Exact Distributed Spline Merging

    [https://arxiv.org/abs/2609.16579](https://arxiv.org/abs/2609.16579)

    本文提出精确分布式样条合并方法，各数据持有者仅需共享局部Gram矩阵和矩向量即可获得与集中式拟合数学上完全相同的解，并通过场重建与导数提取流程从分布式碎片观测中实现物理参数推断。

    

    科学测量通常分布在不同的地点、时间段和机构之间。将这些碎片组合成连续、可微的场，能够从其导数中恢复控制性物理参数。本文为实现这一目标做出了两项贡献。首先，将固定基岭回归统计量的既定可加结构应用于张量积样条场：每个数据持有者计算局部Gram矩阵和矩向量，合并后的解在数学上与集中式拟合完全相同，无需共享原始数据，也无需迭代同步。这一性质特定于固定特征的平方误差设置；本推导并未为一般联合训练的多层网络建立类似的保证。其次，一个完整的处理流程通过场重建、导数提取等步骤，将分布式观测与物理参数推断连接起来。

    arXiv:2609.16579v1 Announce Type: new  Abstract: Scientific measurements are frequently distributed across locations, time periods, and institutions. Combining such fragments into a continuous, differentiable field enables recovering governing physical parameters from its derivatives. This paper makes two contributions toward that goal. First, the established additive structure of fixed-basis ridge-regression statistics is applied to tensor-product spline fields: each data holder computes a local Gram matrix and moment vector, and the merged solution is mathematically identical to centralized fitting, with no raw data shared and no iterative synchronization. This property is specific to the fixed-feature squared-error setting; the present derivation does not establish an analogous guarantee for general jointly trained multilayer networks. Second, a complete pipeline connects distributed observations to physical parameter inference through field reconstruction, derivative extraction, an
    
[^85]: AsyncCouple-Flow：基于异步跨模态耦合与流匹配的时空预测方法

    AsyncCouple-Flow: Asynchronous Cross-Modal Coupling and Flow Matching for Spatio-Temporal Forecasting

    [https://arxiv.org/abs/2609.16573](https://arxiv.org/abs/2609.16573)

    AsyncCouple-Flow 通过模态感知令牌稀疏化、异步跨模态耦合图和流匹配，联合解决了多模态时空预测中采样率不一致、模态易缺失和长时程误差累积三大难题。

    

    多模态时空预测（MM-STF）通过融合物理场、卫星影像和原位传感器等异构数据源，为天气临近预报、交通预测和地球系统建模提供支持。目前仍存在三大障碍：（i）各模态具有不同的时空采样率，迫使其经有损插值后统一到同一网格上；（ii）在实际部署中，由于传感器故障或卫星重访间隙，模态经常缺失，而大多数方法在训练时假设所有模态均完整可用；（iii）自回归解码器在长时程预测中会累积误差，且多模态条件会进一步放大这种误差。我们提出 AsyncCouple-Flow 以联合解决上述问题。模态感知令牌稀疏化（MATS）模块执行尺度感知的令牌化，并利用共享的重要性评分器在每个时间步选择 top-k 个令牌，从而生成等长序列。异步跨模态耦合图（ACCG）用异步机制替代固定的交叉注意力……

    arXiv:2609.16573v1 Announce Type: new  Abstract: Multi-modal spatio-temporal forecasting (MM-STF) supports weather nowcasting, traffic prediction, and earth-system modeling by combining heterogeneous sources such as physical fields, satellite imagery, and in-situ sensors. Three obstacles persist: (i) modalities have different spatio-temporal sampling rates, forcing lossy interpolation onto a unified grid; (ii) modalities are frequently missing at deployment due to sensor outages or revisit gaps, while most methods train with full availability; and (iii) autoregressive decoders accumulate errors over long horizons, amplified by multi-modal conditioning. We propose AsyncCouple-Flow to address these issues jointly. A Modality-Aware Token Sparsification (MATS) module performs scale-aware tokenization and uses a shared importance scorer to select top-k tokens per timestep, producing equal-length sequences. An Asynchronous Cross-Modal Coupling Graph (ACCG) replaces fixed cross-attention with
    
[^86]: 论门控机制的重要性：状态空间模型中的记忆化与上下文学习

    On the Importance of Gating: Memorization vs. In-Context Learning in State Space Models

    [https://arxiv.org/abs/2609.16540](https://arxiv.org/abs/2609.16540)

    本文通过理论和实验揭示了门控机制是状态空间模型在上下文学习任务上落后于Transformer的关键原因——它使模型先收敛到基于权重的记忆化方案，从而延迟甚至阻碍了正确的上下文学习方案的收敛。

    

    状态空间模型（SSMs）已成为Transformer的一种引人注目的替代方案，能够以恒定内存和线性计算实现序列建模。尽管SSM表现出合理的性能和有利的计算特性，但在需要上下文学习和精确检索的任务上，它们仍然落后于Transformer，这减缓了它们在大规模语言建模中的采用。在这项工作中，我们证明了通过研究门控机制——现代循环网络中的一个普遍组件——的作用，可以解释SSM在这些领域的成功与失败。具体来说，我们通过理论和实验表明，这种门控机制导致SSM首先学习一个基于权重的“记忆化”解决方案，同时延迟甚至阻止其收敛到正确的上下文学习解决方案。重要的是，即使在没有架构根本性限制的情况下，这种现象也会发生。

    arXiv:2609.16540v1 Announce Type: cross  Abstract: State Space Models (SSMs) have emerged as a compelling alternative to Transformers, enabling sequence modeling with constant memory and linear compute. Although SSMs exhibit reasonable performance and favorable computational characteristics, they continue to lag behind Transformers on tasks that require in-context learning and precise retrieval, slowing their adoption for large-scale language modeling. In this work, we demonstrate that both the success and failure of SSMs in these domains can be explained by studying the role of the gating mechanism, a prevalent component in modern recurrent networks. Specifically, we show through theory and experiments that this gating mechanism causes SSMs to first learn an in-weights "memorization" solution, while delaying, or even preventing, convergence to a correct in-context learning solution. Importantly, this happens even in cases where there are no fundamental limitations due to the architect
    
[^87]: 层重要性揭示了Transformer与状态空间模型的什么信息？

    What Does Layer-Importance Reveal About Transformers and State-Space Models?

    [https://arxiv.org/abs/2609.16537](https://arxiv.org/abs/2609.16537)

    本文将层重要性分解为“必要性”（预训练模型对已有层贡献的依赖）和“可塑性”（微调时吸收新知识的位置）两个维度，揭示了Transformer与状态空间模型的根本差异——在残差Transformer中两者随深度反向分布，而在Mamba类SSM中两者趋于重叠。

    

    Transformer与状态空间模型（SSMs）是目前两大主流的序列模型家族，一个核心的开放性问题是：为Transformer所建立的分析知识能在多大程度上迁移到SSMs。我们通过层重要性的视角来研究这一问题，层重要性是支撑两个模型家族的模型压缩、选择性微调和可解释性的基础。我们将层重要性分解为两个不同的概念：*必要性*衡量预训练模型对某一层现有贡献的依赖程度，通过绕过该层所导致的损失增加来度量；*可塑性*衡量模型在微调过程中吸收新信息的位置，通过任务特定权重更新的幅度来度量。我们的分析揭示出这两个模型家族的行为存在根本差异：在所有被评估的残差Transformer（参数量最高达140亿）中，必要性与可塑性在网络深度上呈反向分布，而在被评估的Mamba风格SSMs中，两者则指向重叠的位置。

    arXiv:2609.16537v1 Announce Type: cross  Abstract: Transformers and state-space models (SSMs) are the two dominant families of sequence models, and a central open question is how far the analytical knowledge built for transformers transfers to SSMs. We address this through the lens of layer importance which underpins compression, selective fine-tuning, and interpretability across both families. We decompose layer importance into two distinct notions. \emph{Necessity} captures how much the pretrained model depends on a layer's existing contribution, measured by the loss increase from bypassing it. \emph{Plasticity} captures where the model absorbs new information during fine-tuning, measured by the magnitude of task-specific weight updates. Our analysis reveals that the two families behave fundamentally differently: in every evaluated residual transformer up to $14$B parameters, Necessity and Plasticity anti-align across depth, whereas in the evaluated Mamba-style SSMs they point to ove
    
[^88]: FlowATC：基于流匹配的飞机轨迹预测

    FlowATC: Aircraft Trajectory Prediction via Flow Matching

    [https://arxiv.org/abs/2609.16528](https://arxiv.org/abs/2609.16528)

    该论文提出了FlowATC，一种仅在历史ADS-B轨迹数据上训练、无需航线标签或航图监督的流匹配架构，能够生成与历史交通高度吻合的飞机轨迹分布，并准确再现旧金山机场周边的空域结构。

    

    为下一代空中交通管制构建精确的决策支持工具需要强大的轨迹预测模型。我们提出了一种仅在历史飞机轨迹上训练的流匹配架构，无需航线标签或航图监督。该模型在旧金山湾区收集的115万个广播式自动相关监视（ADS-B）轨迹窗口上训练，生成的飞机轨迹分布与历史交通高度吻合，能够再现旧金山机场周边已知的空域结构，例如SFO公布的NIITE FOUR离场程序的形状。我们的模型直接在原始的、不规则的ADS-B采样间隔上进行训练。轨迹预测被建模为序列填充问题，采用块因果Transformer，通过条件流匹配或去噪扩散概率模型，在观测历史的条件下对未来的状态标记进行去噪。我们比较了我们的

    arXiv:2609.16528v1 Announce Type: new  Abstract: Building accurate decision-support tools for next-generation air traffic control requires robust trajectory prediction models. We present a flow-matching architecture trained exclusively on historical aircraft trajectories, with no route labels or chart supervision. Trained on 1.15 million Automatic Dependent Surveillance-Broadcast trajectory windows collected over the San Francisco Bay Area, the model generates aircraft trajectory distributions that closely match historical traffic, reproducing known airspace structure around San Francisco Airport such as the shape of SFO's published NIITE FOUR departure procedure. Our model is trained directly on the native, irregular ADS-B sampling interval. Trajectory prediction is cast as sequence inpainting using a block-causal Transformer that denoises future state tokens conditioned on the observed history using Conditional Flow Matching or Denoising Diffusion Probabilistic Models. We compare our
    
[^89]: 隐半马尔可夫模型维特比算法的高性能张量公式化

    High-Performance Tensor Formulation of the Viterbi Algorithm for Hidden Semi-Markov Models

    [https://arxiv.org/abs/2609.16500](https://arxiv.org/abs/2609.16500)

    本文提出隐半马尔可夫模型维特比算法的张量化公式，首次实现GPU加速实现，相比串行实现，在单核、多核和GPU上分别取得最高14倍、超过200倍和超过570倍的加速。

    

    隐半马尔可夫模型（HSMM）是基础性的概率模型，广泛应用于从计算生物学到金融和信号处理等众多领域。维特比算法在给定HSMM的情况下解码最可能的状态序列，并可通过迭代应用于从头模型学习。然而，现有的维特比算法实现仍然是串行的，且完全缺乏GPU加速的解决方案，使得HSMM解码难以应对大规模工作负载。我们提出了针对HSMM的基于张量的维特比算法公式化，将内循环重构为能够自然映射到SIMD单元和大规模并行架构的张量操作。基于该公式化，我们提供了涵盖单核CPU、多核CPU以及首次实现的GPU的优化方案。实验评估表明，与串行实现相比，单核上加速最高达14倍，多核超过200倍，GPU上超过570倍。

    arXiv:2609.16500v1 Announce Type: new  Abstract: Hidden Semi-Markov Models (HSMMs) are fundamental probabilistic models widely adopted across diverse domains, from computational biology to finance and signal processing. The Viterbi algorithm decodes the most likely state sequence given an HSMM and can be applied iteratively for ab initio model learning. However, existing Viterbi implementations remain sequential, and GPU-accelerated solutions are entirely absent, making HSMM decoding impractical for large-scale workloads. We present a tensor-based formulation of the Viterbi algorithm for HSMMs, restructuring the inner loops into tensor operations that naturally map onto SIMD units and massively parallel architectures. Building on this formulation, we provide optimized implementations spanning single- and multi-core CPUs, and, for the first time, GPU. Experimental evaluation demonstrates speedups of up to 14x on a single core, over 200x with multi-core, and over 570x on GPU over the sta
    
[^90]: 从人工构建到AI驱动的情景涌现：重新思考巨灾风险建模

    From Manual Construction to AI-Driven Scenario Emergence: Rethinking Catastrophe Risk Modeling

    [https://arxiv.org/abs/2609.16493](https://arxiv.org/abs/2609.16493)

    本文提出TAISE框架，通过重新利用AI天气预报模型进行自迭代生成，产生连续的全球大气场使极端天气情景自然涌现，相比传统人工构建方法将计算成本降低一个数量级，并能捕捉时间连续性和跨区域相关性，为巨灾风险量化的普及化开辟新路径。

    

    传统的巨灾（CAT）风险模型依赖成本高昂的人工构建方式来生成极端天气情景，这一方法自20世纪90年代以来基本没有改变。随着气候极端事件不断加剧，这给整个风险转移链条带来了日益严峻的挑战。本研究提出了TAISE框架，该框架重新利用AI天气预报模型，以传统成本的一小部分生成连贯的极端天气序列。通过自迭代生成，该框架产生连续的全球大气场，极端事件从中自然涌现。概念验证实验表明，与传统方法相比，计算成本降低了一个数量级，同时捕捉到了基于快照的方法所缺失的时间连续性和跨区域相关性。这些发现为巨灾风险量化的普及化以及为保险公司提供动态、全面的投资组合评估开辟了一条路径。

    arXiv:2609.16493v1 Announce Type: new  Abstract: Traditional catastrophe (CAT) risk models rely on costly manual construction to generate extreme weather scenarios, an approach largely unchanged since the 1990s. As climate extremes intensify, this creates mounting challenges to the entire risk transfer chain. This study proposes the TAISE framework, which repurposes AI weather forecasting models to produce coherent extreme weather sequences at a fraction of traditional costs. Through self-iterative generation, the framework produces continuous global atmospheric fields from which extreme events emerge. A proof-of-concept experiment demonstrates an order-of-magnitude reduction in computational cost compared with conventional methods, while capturing temporal continuity and cross-regional correlations absent in snapshot-based approaches. These findings suggest a pathway toward democratising catastrophe risk quantification and enabling dynamic, comprehensive portfolio assessment for insur
    
[^91]: 解码器设计对心电图波形划分至关重要

    Decoder Design Matters for ECG Delineation

    [https://arxiv.org/abs/2609.16489](https://arxiv.org/abs/2609.16489)

    提出将ResNet-18编码器与U-Net解码器配对的心电图划分模型R-U-Net，证明解码器设计对性能的贡献超过半监督学习方法，在域内和跨域设置中均显著超越现有最强基线。

    

    心电图（ECG）波形划分是指识别P波、QRS复合波和T波的边界，提供结构化标注以指导AI模型学习解读心电图。然而，训练精确的划分模型需要人工标注，而这些标注稀缺且获取耗时。近期的工作通过半监督学习（SSL）来解决这一局限，但架构设计，尤其是解码器的设计，受到的关注较少。为此，我们提出了R-U-Net，这是一种将ResNet-18编码器与U-Net解码器相结合的心电图划分模型。在SemiSegECG数据集上，R-U-Net在全部16个域内设置中以3.3-13.0 mIoU的优势超越了所评估的最强基线（ResNet-18 + 全卷积网络（FCN）头），并在跨域设置中达到82.6 mIoU，提升了8.1 mIoU。受控消融实验表明，解码器设计对性能提升的贡献大于所评估的半监督学习。

    arXiv:2609.16489v1 Announce Type: cross  Abstract: Electrocardiogram (ECG) delineation identifies the boundaries of P waves, QRS complexes, and T waves, providing structural annotations that can guide AI models in learning to interpret ECGs. However, training accurate delineation models requires manual annotations that are scarce and time-consuming to obtain. Recent work addresses this limitation through semi-supervised learning (SSL), but the design of the architecture, particularly the decoder, has received less attention. To this end, we propose R-U-Net, an ECG delineation model that pairs a ResNet-18 encoder with a U-Net decoder. On SemiSegECG, R-U-Net outperforms the strongest evaluated ResNet-18 + fully convolutional network (FCN) head baseline in each of the 16 in-domain settings by 3.3-13.0 mIoU and achieves 82.6 mIoU in the cross-domain setting, an improvement of 8.1 mIoU. Controlled ablations show that decoder design contributes more to performance gains than the evaluated SS
    
[^92]: 基于技能的实时数据科学任务智能体评估

    Skill-based Agentic Evaluation for Real-time Data Science Tasks

    [https://arxiv.org/abs/2609.16487](https://arxiv.org/abs/2609.16487)

    该论文提出“基准真值即代码”框架，通过可执行的参考函数在评估时从实时数据动态计算答案，并结合格式无关的事实性原子声明评分方法，解决了动态数据场景下数据科学智能体无法用静态基准真值评估的难题。

    

    我们提出了一个框架，用于在实时、持续更新的数据上评估数据科学智能体，该框架使用可执行的基准真值和格式无关的事实性评分。考虑这个示例查询：“上周的观众人数是多少”——参考答案会随着底层数据的变化而变化，因此静态参考答案会过时，标准的“LLM作为评判者”流水线无法根据固定的基准真值来验证响应。我们的核心贡献——“基准真值即代码”，将每个预期答案编码为可执行的参考函数，该函数在评估时直接从实时数据重新计算答案，确保参考答案与它所描述的系统保持一致。我们将此与一个事实层面的、格式无关的评判器相结合，该评判器将智能体的响应和计算出的基准真值分解为原子声明，并对它们的精确率、召回率和准确率进行评分，而不受响应格式的影响（文本、列表、表格、HTML等）。

    arXiv:2609.16487v1 Announce Type: new  Abstract: We present a framework for evaluating data-science agents on live, continuously updated data using executable ground truth and format-agnostic factoid scoring. Consider this example query: "what were last week's audience sizes"---the reference answer changes as the underlying data changes, so static references become outdated and standard LLM-as-a-judge pipelines cannot verify responses against a fixed ground truth. Our central contribution, ground-truth-as-code, encodes each expected answer as an executable reference function that recomputes the answer directly from live data at evaluation time, ensuring the reference remains consistent with the system it describes. We combine this with a factoid-level, format-agnostic judge that decomposes both the agent's response and the computed ground truth into atomic claims and scores precision, recall, and accuracy over them, irrespective of the response format (prose, list, table, HTML, etc.). 
    
[^93]: 深度均衡网络的认证推理与训练：具有多项式复杂度保证的延拓框架

    Certified Inference and Training for Deep Equilibrium Networks: A Continuation Framework with Polynomial Complexity Guarantees

    [https://arxiv.org/abs/2609.16485](https://arxiv.org/abs/2609.16485)

    本文提出了一种认证延拓框架，将深度均衡网络的训练表述为精度插值问题，使推理和训练都能在多项式复杂度预算下获得可认证的保证。

    

    我们为均衡计算和深度均衡网络（DEQ）训练开发了一个认证延拓框架，其中训练被表述为达到精度 $2^{-b}$ 的插值问题。对于推理，紧凑输入同伦从给定的起始根中选择唯一分支，并在经过认证的边界、条件数、导数和管道半径约束下，由舍入牛顿追踪器沿该分支进行追踪。对于训练，我们通过可编程的休眠双线性秩一通道来增强局部加低秩递归结构。加载的 Tikhonov 求解可以在无需谱分解的情况下诊断失败的插值过程；与该过程残差对齐的保持输出的修复机制提供了所需的方向。训练需要在每个过程区域上实现认证门控和列稳定性、适定的推理以及有限的更新误差预算。在多项式几何、编码、精度和完整后端预算约束下，认证推理和训练均具有（多项式复杂度）……

    arXiv:2609.16485v1 Announce Type: cross  Abstract: We develop a certified continuation framework for equilibrium computation and for training deep equilibrium networks (DEQs), with training formulated as interpolation to accuracy $2^{-b}$. For inference, compact input homotopy selects a unique branch from a supplied start root, and a rounded Newton tracker follows it under certified boundary, conditioning, derivative, and tube-radius bounds. For training, we augment local-plus-low-rank recurrence with programmable dormant bilinear rank-one channels. Loaded Tikhonov solves diagnose a failed interpolation pass without spectral decomposition; an output-preserving repair aligned with the pass residual supplies the required direction. Training requires certified gate realization and column stability on each pass region, well-posed inference, and finite-update error budgets. With polynomial geometric, encoding, precision, and complete backend budgets, both certified inference and training ha
    
[^94]: 扭曲高斯过程变换的在线梯度计算

    Online Gradient Computation for Warping Gaussian Process Transformations

    [https://arxiv.org/abs/2609.16472](https://arxiv.org/abs/2609.16472)

    本文证明了扭曲高斯过程瞬时负对数似然的梯度可进行精确递归计算，并据此提出一种能够同时更新潜在GP矩和优化扭曲参数的新型在线方法。

    

    扭曲高斯过程通过一种称为“扭曲”的参数化变换，将非高斯观测数据映射到潜在的标准高斯过程进行处理。然而，现有的流式变体要么需要周期性地优化扭曲参数，要么为获得更高的模型容量而牺牲解析可处理性。为弥补这一空白，我们证明了扭曲高斯过程的瞬时负对数似然梯度可以进行精确的递归计算。基于这一结果，我们提出了一种新颖的扭曲高斯过程在线方法，该方法能够联合更新潜在高斯过程的矩并优化扭曲参数。

    arXiv:2609.16472v1 Announce Type: new  Abstract: Warped Gaussian processes (GPs) handle non-Gaussian observations by mapping them into a latent standard GP via a parametric transformation called warping. Existing streaming variants, however, either optimize the warping parameters periodically or sacrifice analytical tractability for a higher model capacity. To bridge this gap, we show that the gradient of the instantaneous negative log-likelihood of a warped GP admits an exact recursive computation. Based on this result, we propose a novel online method for warped GPs that jointly updates the latent GP moments and optimizes the warping parameters.
    
[^95]: 一种用于循证自闭症谱系障碍筛查的多模态大语言模型

    A multimodal large language model for evidence-based autism spectrum disorder screening

    [https://arxiv.org/abs/2609.16464](https://arxiv.org/abs/2609.16464)

    本研究提出ASDchat——一个采用双分支架构的多模态大语言模型，能够基于视频、音频和对话进行自闭症筛查，同时生成与ADOS-2临床标准对齐的可追溯、带时间戳的行为证据，在中国27个站点1,035名参与者的数据集上达到0.953的AUC。

    

    自闭症谱系障碍（ASD）的临床管理在早期筛查环节面临瓶颈，主要原因在于训练有素的专家稀缺，且传统评估工具具有主观性。在此，我们介绍ASDchat，一个专为循证ASD筛查设计的多模态大语言模型，它以视频、音频和对话作为输入。ASDchat采用双分支架构，其中决策分支生成筛查概率，证据分支生成与标准化临床标准（ADOS-2）对齐的、可追溯的、带有时间戳的行为证据。该模型在中国27个站点共1,035名参与者的数据集上进行训练和评估，数据涵盖典型发育（TD）儿童、ASD儿童以及其他障碍儿童。在ASD与TD的区分任务中，ASDchat达到了0.953 ± 0.021的受试者工作特征曲线下面积（AUC）。在9个未参与训练的留存站点上……

    arXiv:2609.16464v1 Announce Type: cross  Abstract: The clinical management of autism spectrum disorder (ASD) faces a bottleneck in early screening, mainly because trained specialists are scarce and conventional assessment tools are subjective. Here, we introduce ASDchat, a multimodal large language model designed for evidence-based ASD screening, which takes video, audio, and dialogue as input. ASDchat adopts a dual-branch architecture, where the decision branch generates screening probabilities and the evidence branch generates traceable, timestamped behavioral evidence aligned with standardized clinical criteria (ADOS-2). The model was trained and evaluated on a dataset of 1,035 participants from 27 sites in China, which covered typically developing (TD) children, children with ASD, and children with other disorders. For ASD versus TD, ASDchat reached an area under the receiver operating characteristic curve (AUC) of 0.953 $\pm$ 0.021. On 9 held-out sites that were not used for train
    
[^96]: 并非所有关系都是平等的：面向基于溯源入侵检测的关系平衡与校准图学习

    Not All Relations Are Equal: Relation-Balanced and Calibrated Graph Learning for Provenance-Based Intrusion Detection

    [https://arxiv.org/abs/2609.16462](https://arxiv.org/abs/2609.16462)

    提出无监督框架RECAL，通过关系平衡的掩码图学习捕捉稀有交互模式，并结合针对各关系良性错误分布的误差校准机制，在DARPA E3数据集上实现最高99.99%的F1分数，有效降低基于溯源入侵检测中的误报和漏检风险。

    

    基于溯源的入侵检测系统（PIDS）通过分析系统交互来检测高级持续性威胁（APT）。然而，现有方法在很大程度上统一对待各种关系，忽视了统计异质性；在CADETS中，不同关系的出现频率差异约为14万倍。这可能导致PIDS更加关注频繁出现的关系，而忽视不同关系间正常错误水平的差异，从而增加误报和漏检的风险。我们提出了RECAL，一个无监督框架，采用关系平衡的掩码图学习来更好地捕捉稀有的交互模式。该框架进一步针对每种关系的良性错误分布对重构误差进行校准，以产生可比较的异常证据，帮助区分攻击行为与良性行为并减少误报。在三个DARPA E3数据集上，RECAL分别取得了99.99%、99.93%和99.99%的F1分数，超越了最佳基线方法。

    arXiv:2609.16462v1 Announce Type: cross  Abstract: Provenance-Based Intrusion Detection Systems (PIDSs) detect Advanced Persistent Threats (APTs) by analyzing system interactions. However, existing methods largely treat relations uniformly, overlooking statistical heterogeneity; in CADETS, relation frequencies differ by approximately $140{,}000\times$. This may cause PIDSs to focus more on frequent relations and overlook differences in normal error levels across relations, increasing the risk of false alarms and missed detections. We present RECAL, an unsupervised framework using relation-balanced masked graph learning to better capture rare interaction patterns. It further calibrates reconstruction errors against each relation's benign error distribution to produce comparable anomaly evidence, helping distinguish attacks from benign behavior and reduce false alarms. On three DARPA E3 datasets, RECAL achieves F1 scores of 99.99\%, 99.93\%, and 99.99\%, outperforming the best baseline o
    
[^97]: OPD-Aha：多模态在线策略蒸馏中从语言惯性到视觉反思

    OPD-Aha: From Linguistic Momentum to Visual Reflection in Multimodal On-Policy Distillation

    [https://arxiv.org/abs/2609.16459](https://arxiv.org/abs/2609.16459)

    OPD-Aha通过对比教师模型在真实图像与视觉空值下的预测，分离出其被师生语言惯性掩盖的视觉纠正信号，并据此重构蒸馏目标，解决了多模态在线策略蒸馏中师生收敛到相同幻觉的问题。

    

    特权在线策略蒸馏通过让教师模型利用丰富的、仅训练时可用的视觉证据来评估学生轨迹，从而改进多模态推理。两个模型在以相同的学生生成前缀为条件的情况下对这些轨迹进行评分。当学生在回复早期误解了图像时，这种不断累积的错误推理最终会将教师模型从其视觉证据中带偏。教师和学生最终收敛到相同的幻觉上，导致标准的跨模型监督恰好在最需要纠正的地方失效。我们发现，在这种误导性的一致意见下，教师模型的视觉纠正偏好并未丢失。通过比较同一教师模型在真实图像与视觉空值条件下的预测，可以发现特权证据仍在推动模型朝向正确的解释。我们提出了OPD-Aha，它直接从这个分离出的视觉信号中重构蒸馏目标。

    arXiv:2609.16459v1 Announce Type: cross  Abstract: Privileged on-policy distillation improves multimodal reasoning by allowing a teacher to evaluate student trajectories using rich, training-only visual evidence. Both models score these trajectories while conditioning on the same student-generated prefix. When a student misinterprets an image early in a response, this accumulating erroneous rationale eventually pulls the teacher away from its visual evidence. The teacher and student converge on the same hallucination, causing standard cross-model supervision to collapse precisely where correction is most needed. We find that the teacher's visual corrective preference is not lost under this misleading agreement. Comparing the predictions of the identical teacher given the real image and a visual null reveals that the privileged evidence still pushes the model toward the correct interpretation. We introduce OPD-Aha, which reconstructs the distillation target directly from this isolated v
    
[^98]: 早鸟解码：通过可学习块大小与并行采样加速扩散大语言模型

    Early-Bird Decoding: Accelerating Diffusion LLMs with Learnable Block Sizes and Parallel Sampling

    [https://arxiv.org/abs/2609.16450](https://arxiv.org/abs/2609.16450)

    首次提出“早鸟”解码框架，通过可学习网络将低熵聚集的 token 自适应分组为变长块，并结合位置感知采样器进行并行解掩码，从而大幅加速扩散大语言模型的推理。

    

    扩散大语言模型（dLLMs）通过迭代解掩码提供了一种有前景的并行解码范式，可作为自回归生成的替代方案。然而，dLLMs 通常需要许多步骤才能使 token 置信度达到解码阈值，导致即使采用分块 KV 缓存，推理效率仍然低下。为了加速 dLLM 推理，我们首次提出了一种“早鸟（EB）”解码框架，其动机来自于如下观察：熵值相似且较低的 token 往往会聚集在一起，可以在达到置信度阈值之前提前联合解码。特别地，我们的 EB-Decode 框架集成了两个关键的使能技术：（1）一个可学习网络，将具有相似不确定性的 token 自适应地分组为可变长度的块，而非依赖固定的块大小；（2）一个位置感知采样器，学会在预测的可变长度块内使用更少的解码步骤并行解掩码 token。

    arXiv:2609.16450v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) offer a promising parallel decoding paradigm as an alternative to autoregressive generation through iterative unmasking. However, dLLMs typically require many steps before token confidence reaches the decoding threshold, resulting in inefficient inference even with block-wise KV caching. To accelerate dLLM inference, we for the first time propose an "early-bird (EB)" decoding framework, motivated by the observation that tokens with similarly low entropy tend to cluster and can be jointly decoded earlier, before reaching the confidence threshold. In particular, our EB-Decode framework integrates two key enablers: (1) a learnable network that adaptively groups tokens with similar uncertainty into variable-length blocks, rather than relying on fixed block sizes; (2) a position-aware sampler that learns to unmask tokens in parallel using fewer decoding steps within predicted variable-length blocks. B
    
[^99]: 去中心化Gossip学习与联邦平均在组织病理学图像分类中的应用

    Decentralized Gossip Learning and Federated Averaging for Histopathology Image Classification

    [https://arxiv.org/abs/2609.16448](https://arxiv.org/abs/2609.16448)

    本研究系统比较了联邦平均、去中心化gossip学习和混合Gossip-FedAvg三种分布式学习方法在乳腺癌组织病理学图像分类中的性能，通过大规模数据集和全面的敏感性分析，为隐私受限医疗场景下的分布式学习方案选择提供了实证依据。

    

    乳腺癌组织病理学分析越来越依赖分布式学习，因为跨机构直接汇集数据通常受到隐私、治理和通信限制的约束。本研究比较了基于服务器的联邦平均（FedAvg）、完全去中心化的gossip学习以及混合Gossip-FedAvg三种方法在浸润性导管癌（IDC）图像块分类中的表现。实验使用了277,524个彩色图像块，采用患者不重叠的训练、验证和测试分区，并在六个节点上进行了工作负载均衡的、Dirichlet引导的数据分配。研究评估了环形、随机三度和全连接的gossip拓扑结构，并对统计异质性、混合系数、学习率、模型漂移、预测分歧、校准、临床导向的操作点、通信载荷以及患者级别的IDC负担等进行了敏感性分析，还包括辅助骨干网络的鲁棒性分析。

    arXiv:2609.16448v1 Announce Type: cross  Abstract: Breast histopathology analysis increasingly relies on distributed learning because direct data pooling across institutions is often restricted by privacy, governance, and communication constraints. This study compares server-based Federated Averaging (FedAvg), fully decentralized gossip learning, and Hybrid Gossip-FedAvg for invasive ductal carcinoma (IDC) patch classification. Experiments used 277,524 color image patches with patient-disjoint training, validation, and test partitions and a workload-balanced, Dirichlet-guided allocation across six nodes. Ring, random degree-3, and fully connected gossip topologies were evaluated together with sensitivity analyses for statistical heterogeneity, mixing coefficient, learning rate, model drift, prediction disagreement, calibration, clinically motivated operating points, communication payload, and patient-level IDC burden, together with auxiliary backbone robustness analyses. In the princip
    
[^100]: 面向联邦临床中心的自适应贝叶斯伙伴选择

    Adaptive Bayesian Partner Selection for Federated Clinical Centers

    [https://arxiv.org/abs/2609.16446](https://arxiv.org/abs/2609.16446)

    本文提出ABPS框架，让联邦学习中的临床中心通过Beta-Bernoulli后验和UCB准则自适应地选择点对点协作伙伴，在大幅降低通信开销的同时避免负迁移，并具备理论遗憾保证。

    

    医疗健康领域的联邦学习（FL）面临着各临床中心之间显著的异质性和时间上的概念漂移，不断变化的患者群体和医疗实践会导致数据分布发生偏移。现有方法依赖于持续的全局通信，这不仅带来大量的带宽开销，还存在因合作伙伴匹配不佳而导致负迁移的风险。我们提出了自适应贝叶斯伙伴选择（ABPS），这是一个点对点框架，用于管理协作的对象、时机和代价。每个中心对潜在合作伙伴的Shapley边际效用维护一个Beta-Bernoulli后验分布，使用置信上界（UCB）准则对候选伙伴进行排序，并通过轻量级的提议-拒绝机制建立协作，当不存在互利伙伴时还可以选择放弃通信。该框架具有随机决策的理论解释，可提供有限样本集中性保证以及O(κ log T)的遗憾界。

    arXiv:2609.16446v1 Announce Type: new  Abstract: Federated learning (FL) in healthcare faces pronounced heterogeneity and temporal concept drift across clinical centers, where evolving patient populations and care practices shift data distributions. Existing approaches rely on persistent global communication, incurring substantial bandwidth overhead while risking negative transfer from poorly aligned peers. We propose Adaptive Bayesian Partner Selection (ABPS), a peer-to-peer framework that governs who collaborates, when, and at what cost. Each center maintains a Beta-Bernoulli posterior over prospective peers' Shapley marginal utility, ranks candidates with an Upper Confidence Bound (UCB) criterion, and forms collaborations through a lightweight propose-reject mechanism, with the option to abstain from communication when no mutually beneficial partner exists. The framework admits a stochastic decision interpretation, yielding finite-sample concentration guarantees and O(kappa log T) r
    
[^101]: Neverwhere视觉跑酷基准测试套件

    The Neverwhere Visual Parkour Benchmark Suite

    [https://arxiv.org/abs/2609.16443](https://arxiv.org/abs/2609.16443)

    提出了包含六十多个3D高斯泼溅重建场景的超逼真闭环评估基准套件Neverwhere，用于评估视觉运动控制器的真实世界部署性能，并揭示了仅依靠3D高斯生成数据训练策略的泛化局限。

    

    最先进的视觉运动控制器在处理复杂视觉环境方面的能力日益增强，这使得在部署前评估其实际性能变得越来越困难。本工作旨在通过开发一套超逼真的闭环评估环境来缩小训练与评估之间的差距——Neverwhere基准测试套件，该套件包含六十多个基于3D高斯泼溅技术重建的城市室内外场景。我们的目标是通过简化基于高斯泼溅重建的创建与集成到模拟持续测试环境中的流程，推动大规模且可复现的机器人评估。此外，我们通过提供在多个Neverwhere场景上训练的策略检查点及其在新场景中的评估表现，强调了仅依赖3D高斯生成数据进行训练的潜在陷阱。我们的分析说明了其必要性。

    arXiv:2609.16443v1 Announce Type: cross  Abstract: State-of-the-art visual locomotion controllers are increasingly capable at handling complex visual environments, making evaluating their real-world performance before deployment increasingly difficult. This work intends to narrow this train/evaluation gap by developing a collection of hyper-photo-realistic, closed-loop evaluation environments - The Neverwhere Benchmark Suite - comprised of over sixty 3D Gaussian Splatting reconstructions of urban indoor and outdoor scenes. Our goal is to encourage large-scale and reproducible robot evaluation by making it easier to create and integrate Gaussian splats-based reconstructions into simulated continuous testing setups. We also underscore the potential pitfalls of relying exclusively on 3D Gaussian-generated data for training, by providing policy checkpoints trained over multiple Neverwhere scenes and their performance when evaluated in novel scenes. Our analysis illustrates the necessity of
    
[^102]: CART的学习型前瞻分裂规则

    Learned Look-Ahead Splitting Rule for CART

    [https://arxiv.org/abs/2609.16440](https://arxiv.org/abs/2609.16440)

    该论文提出一种通过在候选分裂点下方生长CART子树来评估分裂质量的前瞻分裂规则，并利用节点级特征学习的智能前瞻算法大幅降低计算成本，在保持决策树可解释性的同时显著改善层级或交互场景下的分裂选择。

    

    分类和回归树（CART）通常采用贪心分裂规则构建，即在每个节点处最大化预测误差的即时下降。尽管这种策略计算效率高，但它可能错过那些短期收益较小、却在进一步划分后能带来显著下游改进的分裂。我们提出了一种前瞻式建树方法，通过在该候选分裂点下方生长一个常规CART子树后所取得的预测误差下降来评估每个候选分裂。由于完整的前瞻过程计算代价高昂，我们还提出了一种智能前瞻算法，利用节点级特征来学习下游的分裂值。所提出的框架在保持递归划分可解释性的同时，改善了层级结构或交互效应主导场景下的分裂选择。我们开展了模拟研究，对常规方法、完整前瞻和智能前瞻方法进行比较。

    arXiv:2609.16440v1 Announce Type: cross  Abstract: Classification and regression trees are typically constructed using a greedy splitting rule that maximizes the immediate reduction in prediction error at each node. Although this strategy is computationally efficient, it can miss splits that yield small short-term gains but create substantial downstream improvements after further partitioning. We propose a look-ahead tree-building method that evaluates each candidate split by the prediction error reduction achieved after growing a conventional CART subtree below that split. Because the full look-ahead procedure can be computationally expensive, we also describe a smart look-ahead algorithm that learns downstream split values using node-level features. The proposed framework preserves the interpretability of recursive partitioning while improving split selection in hierarchical or interaction-driven settings. We conduct a simulation study comparing conventional, full look-ahead, and sma
    
[^103]: 面向社会模拟的大语言模型智能体的解释与操控

    Interpreting and Steering LLM Agents for Social Simulations

    [https://arxiv.org/abs/2609.16436](https://arxiv.org/abs/2609.16436)

    本文针对大语言模型作为黑箱在社会科学模拟中缺乏可解释性与可操控性的问题，比较了提示词操控、SAE特征引导和探测器方向引导三种方法来解释和操控LLM智能体的行为。

    

    arXiv:2609.16436v1 公告类型：cross 摘要：基于大语言模型（LLM）的模拟已被证明是理解人类行为的强大工具，使其成为社会科学工具箱中的宝贵补充。然而，大语言模型本质上仍是基于深度神经网络的黑箱，这限制了其在社会科学中的价值。原因在于缺乏（i）可解释性：即为观察到的行为指定明确驱动机制的能力；以及缺乏（ii）可操控性：即能够抑制或放大特定的具有理论意义的行为机制以驱动特定模型行为的能力。本文展示了如何打开这一黑箱，从而进一步丰富基于LLM的模拟。具体而言，我们比较了三种类型的方法：（1）基于提示词的操控、（2）基于SAE（稀疏自编码器）的特征引导，以及（3）基于探测器的方向引导，并考察了它们在基于LLM的社会科学模拟中的效用。我们通过对两个基础性的……进行解释与操控来实现这一目标。

    arXiv:2609.16436v1 Announce Type: cross  Abstract: Simulations based on large language models (LLMs) have proven to be powerful for understanding human behavior, making them valuable additions to the social scientific toolkit. However, LLMs are ultimately black boxes based on deep neural networks which limits their value for social science. This is because of a lack of (i) interpretability: i.e. the ability to assign clear mechanisms driving observed behavior; and a lack of (ii) steerability: i.e. the ability to mute or amplify specific theoretically meaningful mechanisms of action to drive specific model behavior. Here, we demonstrate how the black box could be opened up to further enrich LLM-based simulations. Specifically, we compare three types of methods: (1) prompt-based manipulation, (2) SAE-derived feature steering, and (3) probe-based direction steering and examine their utility for LLM-based social scientific simulations. We do so by interpreting and steering two foundational
    
[^104]: 时间序列基础模型在行人人群计数预测中的表现如何？一项跨数据集对比研究

    How Good Are Time-Series Foundation Models for Pedestrian Crowd Count Forecasting? A Cross-Dataset Comparative Study

    [https://arxiv.org/abs/2609.16415](https://arxiv.org/abs/2609.16415)

    本研究通过在特殊活动短时数据与墨尔本多年传感器数据两种场景下，对七种涵盖传统方法、深度学习和预训练基础模型的预测方法进行跨数据集基准对比，系统评估了时序基础模型在行人计数预测中的实际迁移能力。

    

    行人计数预测为以行人为中心的智能交通系统（ITS）提供支持，包括人群监控、行人交通的人员配置与路径规划，以及人流激增期间的主动风险缓解。近期的时序基础模型（FMs）在异构预测基准上展现了强大的零样本精度，但这些优势能否可靠地迁移到行人感知部署中仍不明确。我们对涵盖四种范式的七种单变量预测方法进行了基准测试：季节性朴素方法、梯度提升树（LightGBM、CatBoost）、深度学习模型（N-HiTS、PatchTST），以及两个预训练基础模型（TimesFM、Chronos-2）。实验覆盖两种互补的场景：（i）SAIL2025特殊活动五天数据集，分辨率为3分钟且域内历史数据有限；（ii）墨尔本行人传感器数据，作为具有强季节性的多年小时级数据集（2010–2017）。我们比较了各传感器的MAE和RMSE结果（摘要在此处截断）。

    arXiv:2609.16415v1 Announce Type: cross  Abstract: Pedestrian-count forecasting supports pedestrian-oriented Intelligent Transportation Systems (ITS), including crowd monitoring, pedestrian-traffic staffing and routing, and proactive risk mitigation during surges. Recent time-series foundation models (FMs) report strong zero-shot accuracy on heterogeneous forecasting benchmarks, but it remains unclear whether these gains transfer reliably to pedestrian sensing deployments. We benchmark seven univariate forecasting approaches spanning four paradigms: Seasonal Naive, gradient-boosted trees (LightGBM, CatBoost), deep learning models (N-HiTS, PatchTST), and two pretrained FMs (TimesFM, Chronos-2). Experiments cover two complementary regimes: (i) a five-day special event dataset SAIL2025 at 3-minute resolution with limited in-domain history; and (ii) Melbourne pedestrian sensors as a multi-year hourly dataset (2010--2017) with strong seasonality. We compare the MAE and RMSE results per sens
    
[^105]: 论隐式线图高阶Weisfeiler--Leman的表达能力

    On the Expressive Power of Implicit Line-Graph Higher-Order Weisfeiler--Leman

    [https://arxiv.org/abs/2609.16412](https://arxiv.org/abs/2609.16412)

    本文研究了隐式线图Weisfeiler-Leman测试的表达能力与根图WL的关系，发现在k=1,2时其区分能力不超过根图的1-WL，并首次证明了k=3时线图3-WL等价可推出根图3-WL等价的反向包含关系。

    

    惠特尼定理使得除$K_3$和$K_{1,3}$之外的连通简单图的同构测试可以被表述为区分它们的线图。然而，固定维度的Weisfeiler--Leman（WL）表达能力在线图与其根图之间的关系仍未解决。我们通过隐式线图WL（ILG-$k$-WL）来研究这种关系，它在$L(G)$上精确等同于$k$-WL，在$G$的边上执行，线图关系由端点关联推导得出，而无需显式构造$L(G)$。在Whitney广义类上，根图域与线图WL之间的关系取决于$k$。对于$k=1,2$，ILG-$k$-WL没有增加超出根图域$1$-WL的区分能力，并且会遗漏一些$1$-WL可以区分的图对。对于$k=3$，我们证明了反向包含关系$L(G)\equiv_{3\text{-WL}}L(H)\Rightarrow G\equiv_{3\text{-WL}}H$。强正则见证图对，包括Shrikhande/rook图对，表明ILG-（摘要在此处不完整）

    arXiv:2609.16412v1 Announce Type: cross  Abstract: Whitney's theorem allows isomorphism testing for connected simple graphs, apart from $K_3$ and $K_{1,3}$, to be formulated as distinguishing their line graphs. However, the relation between fixed-dimensional Weisfeiler--Leman (WL) expressivity on line graphs and on their roots remains unresolved. We study this relation through Implicit Line-Graph WL (ILG-$k$-WL), which is exactly $k$-WL on $L(G)$, executed over the edges of $G$ with line-graph relations derived from endpoint incidence and without explicitly constructing $L(G)$. On the Whitney-general class, the relation between root-domain and line-graph WL depends on $k$. For $k=1,2$, ILG-$k$-WL adds no distinguishing power beyond root-domain $1$-WL and misses some pairs that $1$-WL separates. For $k=3$, we prove the backward containment $L(G)\equiv_{3\text{-WL}}L(H)\Rightarrow G\equiv_{3\text{-WL}}H$. Strongly regular witness pairs, including the Shrikhande/rook pair, show that ILG-$
    
[^106]: 平衡尝试与复购：面向即时配送的混合序列Transformer-GBDT排序器

    Balancing Trial and Reorder: A Hybrid Sequential Transformer-GBDT Ranker for On-Demand Delivery

    [https://arxiv.org/abs/2609.16407](https://arxiv.org/abs/2609.16407)

    该论文提出了部署在Wolt的统一商家排序系统UVR，通过双向Transformer编码器进行序列用户建模并结合GBDT排序器，利用标签平滑和尝试偏置样本加权来平衡新商家探索与复购排序质量，以单一模型替代四个独立排序模型，使离线尝试MRR提升12%至30%。

    

    在配送平台上，个性化的商家排序极大地影响用户发现和下单的内容。与纯数字领域不同，候选商家具有本地属性，并受实时可用性和配送运营的约束。其中一个核心的建模矛盾在于：既要推广新商家以供用户尝试，又要在具有复购意图的会话中保持排序质量。我们提出了通用商家排序器，这是部署在Wolt的生产系统，它将用于序列用户建模的双向Transformer编码器与整合了上下文、用户和商家特征的GBDT排序器相结合。UVR在一个国家的所有商家和业务领域上进行训练，同时在推理时强制执行本地配送约束，它用一个统一的系统替代了之前四个独立的排序模型（三个用于餐厅，一个用于零售）。标签平滑和偏向尝试的样本加权引导模型关注新商家，使离线尝试MRR相比生产系统提升了12%至30%。

    arXiv:2609.16407v1 Announce Type: cross  Abstract: On a delivery platform, personalized store ranking greatly influences what users find and order. Unlike digital-only domains, candidate stores are local and bound by real-time availability and delivery operations. One central modeling tension is between surfacing new stores for trial and preserving ranking quality for sessions with reorder intent. We present Universal Venue Ranker (UVR), a production system deployed at Wolt that pairs a bidirectional transformer encoder for sequential user modeling with a GBDT ranker integrating contextual, user, and store features. Trained across all stores and domains of a country while enforcing local delivery constraints at inference, UVR replaces four previously separate ranking models (three for restaurants, one for retail) with a single unified system. Label smoothing and trial-biased sample weighting steer the model toward new stores, lifting offline trial MRR by +12% to +30% over production wh
    
[^107]: 用于求解偏微分方程的物理信息随机特征神经网络

    Physics Informed Random Feature Neural Networks for Solving PDEs

    [https://arxiv.org/abs/2609.16406](https://arxiv.org/abs/2609.16406)

    本文提出一种物理信息随机特征神经网络方法，通过缓解PINN求解器面临的谱偏差问题降低计算复杂度，并给出了严格的$H^1$范数高概率误差界分析。

    

    基于机器学习的偏微分方程（PDEs）求解器近年来引起了广泛关注。该领域的大部分进展是由深度神经网络推动的，例如物理信息神经网络（PINNs）和核方法（如物理信息高斯过程）。我们提出了一种物理信息随机特征方法，用于对抗基于PINN的求解器在处理某类偏微分方程时所面临的谱偏差问题。随机特征方法最初被提出用于近似大规模核机器，可以被视为一种特殊的随机化神经网络。与其他需要大量配置点的最先进的基于PINN的求解器相比，我们提出的方法降低了计算复杂度。在本文中，我们建立了严格的近似误差分析，并推导出了$H^1$范数上的高概率误差界。我们提供了大量的数值测试进行验证。

    arXiv:2609.16406v1 Announce Type: cross  Abstract: Machine learning-based partial differential equations (PDEs) solvers have attracted significant attention in recent years. Most progress in this area has been driven by deep neural networks such as physics-informed neural networks (PINNs) and kernel method (such as physics-informed Gaussian Processes). We introduce a physics-informed random feature method for countering part of the spectral bias which PINN-based solvers are facing for a certain class of PDEs. Random feature method was originally proposed to approximate large-scale kernel machines and can be viewed as a specialized randomized neural network. Compared to other state-of-the-art PINN-based solvers which require a large number of collocation points, our proposed method reduces the computational complexity. In this paper, we develop a rigorous approximation error analysis and derive high-probability error bounds on the $H^1$ norm. We provide extensive numerical tests for ver
    
[^108]: 为随机傅里叶特征实现白盒不可检测后门

    Implementing a White-Box Undetectable Backdoor for Random Fourier Features

    [https://arxiv.org/abs/2609.16403](https://arxiv.org/abs/2609.16403)

    本文仅用numpy和scipy端到端实现了基于CLWE问题的白盒不可检测RFF后门构造，验证了这类即使完整白盒审计也无法发现的后门威胁可以用普通科学计算工具实现。

    

    Goldwasser等人证明了在基于连续带误差学习（CLWE）问题的困难性假设下，可以在使用随机傅里叶特征（RFF）算法训练的机器学习模型中植入不可检测的后门。在标准密码学假设下，即使对模型权重进行完整的白盒审计也无法检测到这类后门。该构造以密码学归约和概率引理的形式表述，缺乏参考实现，并依赖诸如稀疏高斯煎饼分布和齐次CLWE条件密度等辅助机制，仅从论文本身来看，其在普通数值代码中的可实现性并不明显。本文仅使用numpy和scipy端到端地实现了白盒CLWE-RFF后门构造，以检验这一威胁能否用普通科学计算工具实现，还是需要专门的密码学基础设施。

    arXiv:2609.16403v1 Announce Type: cross  Abstract: Goldwasser et al. showed that undetectable backdoors can be planted in machine learning models trained with the Random Fourier Features (RFF) algorithm, under a hardness assumption tied to the Continuous Learning With Errors (CLWE) problem. Under standard cryptographic assumptions, even a full white-box audit of a model's weights cannot detect this class of backdoor. The construction is stated in terms of cryptographic reductions and probabilistic lemmas, without a reference implementation, and relies on secondary machinery such as the Sparse Gaussian Pancakes distribution and a homogeneous CLWE conditional density. Its realizability in ordinary numerical code is not obvious from the paper alone.   This paper implements the white-box CLWE-RFF backdoor construction end to end using only numpy and scipy, to test whether this threat is realizable with commodity scientific-computing tools or requires specialized cryptographic infrastructur
    
[^109]: 注意力平均场预测表示的平均演化动态并揭示上下文特定计算

    Attention Mean Fields Predict Average Representation Dynamics and Reveal Context-Specific Computation

    [https://arxiv.org/abs/2609.16382](https://arxiv.org/abs/2609.16382)

    本文提出一种注意力的平均场分析方法，通过逐层迭代的平均注意力核来预测语言模型表示几何结构的演化动态，并利用“平均场偏差”分离出平均场所无法捕捉的上下文特定计算。

    

    语言模型的表示几何结构并非预先确定；它随着模型的运行而不断演化。对该几何结构的忠实刻画必须捕捉这一动态过程，因此不能仅基于共现统计等与模型无关的统计量。本文引入了一种对注意力的平均场分析。从一个词元到另一个词元的平均注意力定义了一个核，该核将表示逐层传递，并可在网络中迭代，以建模表示几何结构如何被变换。我们以两种方式对该平均量进行条件化：以整个语料库为条件时，该核预测表示几何结构的平均演化；而以单个上下文为条件时，它预测该上下文下的期望几何结构。某个注意力头对该预测的偏离，即其“平均场偏差”，可以分离出平均场所遗漏的上下文特定计算。在语料库条件化下，该核产生一个开环模（原文在此处被截断）……

    arXiv:2609.16382v1 Announce Type: cross  Abstract: A language model's representation geometry is not predetermined; it evolves as the model runs. A faithful account of that geometry must capture that dynamic process, and so cannot be based solely on model-independent statistics such as co-occurrence. Here we introduce a mean-field analysis of attention. The average attention from one token to another defines a kernel that carries representations layer to layer and can be iterated through the network to model how the geometry is transformed. We condition this average two ways. Conditioned on a whole corpus, the kernel predicts the average-case evolution of representation geometry. Conditioned instead on a single context, it predicts the expected geometry for that context. A head's departure from that prediction, its \emph{mean-field deviation}, isolates the context-specific computation that the mean field misses.   Under the corpus-conditional reading, the kernel yields an open-loop mod
    
[^110]: 基于有界调整与可靠性引导嵌入的带噪标签不平衡学习方法

    Bounded Adjustment with Reliability-Guided Embedding for Imbalanced Learning with Noisy Labels

    [https://arxiv.org/abs/2609.16380](https://arxiv.org/abs/2609.16380)

    该论文提出BARGE方法，将有界先验调整密度幂得分与可靠性引导嵌入结合为单阶段目标函数，在应对类别不平衡的同时约束噪声标签带来的分类风险扰动，并在模型与标签高度冲突时自动衰减梯度以抵抗标签噪声。

    

    类平衡学习与标签噪声会形成一种耦合的失效模式：频率校正可以防止多数类主导决策规则，但却可能放大被错误标注的少数类样本的影响。我们提出了BARGE（有界调整与可靠性引导嵌入，Bounded Adjustment with Reliability-Guided Embeddings），这是一种单阶段目标函数，将有界的、经先验调整的密度幂得分与可靠性引导的角度几何结构相结合。其分类得分在调整后的概率空间中是严格恰当的，并且在干净监督与真实类别先验条件下能够恢复平衡的贝叶斯排序。在标签污染的情况下，其有限的取值范围在固定预测器处约束了分类风险的扰动，而当模型高度自信地与给定标签相矛盾时，其logit梯度会重新衰减。调整后的目标概率还对类间均等的特征紧凑性进行加权，同时单侧分离项会抑制不同类别方向的对齐。BARGE既不需要噪声（原文摘要在此处截断）

    arXiv:2609.16380v1 Announce Type: new  Abstract: Class-balanced learning and label noise create a coupled failure mode: frequency correction prevents majority classes from dominating the decision rule, but can amplify incorrectly labeled minority examples. We introduce BARGE (Bounded Adjustment with Reliability-Guided Embeddings), a single-stage objective combining a bounded, prior-adjusted density-power score with reliability-guided angular geometry. Its classification score is strictly proper in the adjusted probability space and recovers balanced Bayes ordering under clean supervision and the true class prior. Under label contamination, its finite range bounds classification-risk perturbation at a fixed predictor, while its logit gradient redescends when the model confidently contradicts the supplied label. The adjusted target probability also weights class-equal feature compactness, and a one-sided separation term discourages aligned class directions. BARGE requires neither a noise
    
[^111]: 基于后验事件传输的一次性联邦贝叶斯模型中的认证不确定性传播

    Certified Uncertainty Propagation in One-Shot Federated Bayesian Models via Posterior Event Transport

    [https://arxiv.org/abs/2609.16373](https://arxiv.org/abs/2609.16373)

    本文提出一个与部署一致的认证框架，通过将客户端本地后验事件沿联邦平均等聚合规则进行几何传播，使一次性联邦贝叶斯模型的聚合部署模型也能获得可认证的安全保证。

    

    贝叶斯神经网络的概率认证为模型满足验证者所定义的安全属性的后验概率提供下界。然而，在一次性联邦贝叶斯学习中，部署模型是通过聚合从客户端各自后验分布中抽取的参数而得到的，因此本地证书并不能直接保证聚合模型的安全性。本文通过将本地后验事件沿部署聚合规则进行传播，构建了一个与部署一致的认证框架，并对联邦平均给出了精确的几何刻画。每个客户端在参数空间中构造互不相交的超矩形区域并计算其概率质量；服务器将这些区域的笛卡尔积经由部署规则进行映射，仅当乘积事件的聚合像被验证满足安全属性时才保留该乘积事件。在客户端独立的……（摘要原文在此处被截断）

    arXiv:2609.16373v1 Announce Type: new  Abstract: Probabilistic certification of Bayesian neural networks lower-bounds the posterior probability that a model satisfies a verifier-defined safety property. In one-shot federated Bayesian learning, however, the deployed model is obtained by aggregating parameters drawn from client-specific posterior distributions, so local certificates do not directly guarantee safety of the aggregated model. This paper develops a deployment-consistent certification framework by propagating local posterior events through the deployment aggregation rule, with an exact geometric characterization for Federated Averaging (FedAvg). Each client constructs disjoint hyper-rectangular regions in parameter space and computes their probability masses. The server forms Cartesian products of these regions, maps them through the deployment rule, and retains a product event only when its aggregation image is verified to satisfy the safety property. Under independent clien
    
[^112]: 基于梯度聚类基站采样的快速收敛元强化学习边缘缓存方法

    Fast-Convergent Meta-RL via Gradient-Clustered BS Sampling for Edge Caching

    [https://arxiv.org/abs/2609.16370](https://arxiv.org/abs/2609.16370)

    本文提出一种通过梯度聚类基站采样替代均匀随机采样的元强化学习边缘缓存框架，显著降低了大规模网络中元梯度估计的方差，实现了快速收敛。

    

    无线边缘缓存网络通常由许多独立的基站组成，每个基站面临着各自的请求速率和内容流行度特征。在每个基站上从零开始训练强化学习（RL）缓存智能体，迫使每个智能体通过缓慢的试错过程重新学习一个在整个网络中结构上完全相同的决策问题。元强化学习通过学习一个共享的初始化参数来消除这种冗余，该初始化仅需少量本地更新即可适应任何基站；然而，元训练本身在大规模场景下成为瓶颈：元梯度必须在每次元迭代中从一小部分基站子集中估计得到，而均匀随机采样该子集会产生高方差的估计，这是现有元强化学习缓存框架尚未解决的问题。本文提出了一种面向独立、非重叠基站的缓存元强化学习框架，直接针对这一瓶颈进行优化。每个基站r……（原文在此处截断）

    arXiv:2609.16370v1 Announce Type: cross  Abstract: Wireless edge caching networks typically consist of many independent Base Stations (BSs), each facing its own request rate and content popularity profile. Training a Reinforcement Learning (RL) caching agent from scratch at every BS forces each agent to relearn, through slow trial and error, a decision problem that is structurally identical across the network. Meta-reinforcement learning removes this redundancy by learning a shared initialization that adapts to any BS in a few local updates; however, meta-training itself becomes the bottleneck at scale: the meta-gradient must be estimated from a small subset of BSs at each meta-iteration, and sampling this subset uniformly at random yields a high-variance estimate, an issue existing meta-RL caching frameworks leave unaddressed. This paper proposes a meta-reinforcement learning framework for caching across independent, non-overlapping BSs that directly targets this bottleneck. Each BS r
    
[^113]: 基于模型的强化学习实现液滴自主导航

    Autonomous Droplet Navigation via Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2609.16369](https://arxiv.org/abs/2609.16369)

    本研究提出基于模型的强化学习方法，实现了液滴在重力驱动迷宫平台上自主穿越日益复杂几何结构的导航，克服了接触角滞后和毛细钉扎等非线性难题。

    

    液滴的精确操控是诊断、化学合成和生物检测等芯片实验室平台的基础。然而，液滴在复杂度各异的受限几何结构中的自主传输仍然是一个悬而未决的挑战。液滴表现出接触角滞后、可变形性和毛细钉扎等特性，使其对驱动的响应呈现非线性且依赖于历史，经典控制器和预编程轨迹无法在多弯道环境中应对这些问题。在本研究中，我们展示了利用基于模型的强化学习，在重力驱动（迷宫）平台上实现液滴穿越复杂度递增的几何结构的自主导航。薄硅油膜可减少接触线钉扎，双轴倾斜提供重力驱动力，顶置摄像头实时跟踪液滴。离线训练的策略从有限的物理交互中发现有效的倾斜策略

    arXiv:2609.16369v1 Announce Type: new  Abstract: Precise manipulation of liquid droplets underpins lab-on-a-chip platforms for diagnostics, chemical synthesis, and biological assays. Yet autonomous droplet transport through confined geometries of varying complexity remains an open challenge. Droplets exhibit contact-angle hysteresis, deformability, and capillary pinning, which make their response to actuation nonlinear and history dependent, that classical controllers and pre-programmed trajectories cannot cope in multi-turn environments. Here we demonstrate autonomous navigation of a liquid droplet through geometries of increasing complexity on a gravity driven (Labyrinth) platform using model-based reinforcement learning. A thin silicone oil film reduces contact-line pinning while two-axis tilt supplies the gravitational driving force, and an overhead camera tracks the droplet in real time. An offline-trained policy discovers effective tilt strategies from limited physical interactio
    
[^114]: 面向长尾图像分类的小批量采样策略：基于CIFAR-100-LT的实证研究

    Mini-batch Sampling Strategies for Long-Tailed Image Classification: An Empirical Study on CIFAR-100-LT

    [https://arxiv.org/abs/2609.16365](https://arxiv.org/abs/2609.16365)

    本文在统一的偏差-方差框架下系统比较了均匀实例采样、类平衡采样、平方根采样和渐进平衡采样四种小批量采样策略对长尾图像分类中梯度估计的影响，并在CIFAR-100-LT数据集上进行了实证评估。

    

    现实世界的数据集通常呈现长尾类别分布，少数头部类别包含大量训练样本，而大量尾部类别仅有极少样本。由采样策略决定的每个小批量的组成，决定了哪些类别参与随机梯度估计，从而影响整个类别范围内的收敛行为和泛化能力。我们对四种用于长尾图像分类的小批量采样策略进行了系统的理论与实证比较：均匀实例采样、类平衡采样、平方根采样和渐进平衡采样。我们将这四种策略置于统一的偏差-方差框架中，阐述它们对梯度估计的影响，揭示了经验损失无偏优化与稀有类别公平表示之间的矛盾。随后，我们在受控条件下对它们进行评估。

    arXiv:2609.16365v1 Announce Type: cross  Abstract: Real-world datasets often exhibit long-tailed class distributions, where a few head classes contain a large number of training samples while a large number of tail classes have only a few. The composition of each mini-batch, determined by the sampling strategy, governs which classes contribute to the stochastic gradient estimate, and therefore affects convergence behaviour and generalisation across the whole class spectrum. We provide a systematic theoretical and empirical comparison of four mini-batch sampling strategies for long-tailed image classification: uniform instance sampling, class-balanced sampling, square-root sampling, and progressively balanced sampling. We place all four in a unified bias-variance framework describing their effect on gradient estimation, which exposes the tension between unbiased optimisation of the empirical loss and fair representation of rare classes. We then evaluate them under controlled conditions 
    
[^115]: EBL：面向分布式自适应谐波分析的高效宽度学习

    EBL: Efficient Broad Learning for Distributed Adaptive Harmonic Analysis

    [https://arxiv.org/abs/2609.16358](https://arxiv.org/abs/2609.16358)

    本文提出面向分布式自适应谐波估计的高效宽度学习（EBL）量化FPGA加速框架，以半个周波输入实现高精度、超低延迟（比现有最快FPGA方法快17.4倍）且可重构的谐波分析。

    

    近年来，可再生能源系统和电气化交通得到了广泛普及。然而，以电动汽车（EV）充电为主导的这些非线性负荷的接入，给电网引入了严重的谐波畸变，影响了配电网中变电站设备和开关设备的效率与使用寿命。因此，快速且高精度的谐波分析已成为在谐波注入源头进行有效治理的先决条件。本文提出了一种用于分布式自适应谐波估计的高效宽度学习框架。作为面向BLS式谐波估计的量化FPGA加速框架，它以半个周波的输入即可实现高精度估计，借助FPGA实现提供了可重构的灵活性，并具有超低延迟，其预测速度比已报道的最接近的FPGA方法快17.4倍。对于跨多个……（原文摘要在此处截断）

    arXiv:2609.16358v1 Announce Type: cross  Abstract: Renewable energy systems and electrified transport have found widespread adoption in recent years. The integration of these non-linear loads, dominated by electric vehicle (EV) charging, however, has introduced severe harmonic distortion into the power grid, impacting the efficiency and lifetime of substation equipment and switchgear in the distribution network. Rapid and high-precision harmonic analysis has hence become a prerequisite for effective harmonic control at the source of injection. This paper proposes an Efficient Broad Learning (EBL) framework for distributed adaptive harmonic estimation. As a quantised FPGA acceleration framework for BLS-style harmonic estimation, it offers high-accuracy estimation with half-cycle input, reconfigurable flexibility enabled by the FPGA implementation, and ultra-low latency, achieving 17.4 $\times$ faster predictions than the nearest reported FPGA method. For harmonic prediction across multi
    
[^116]: 具有全一阶梯度的联邦随机双层优化

    Federated stochastic bilevel optimization with fully first-order gradients

    [https://arxiv.org/abs/2609.16350](https://arxiv.org/abs/2609.16350)

    提出了一种仅需一阶梯度的联邦随机方差缩减双层梯度下降算法，避免了二阶Hessian和Jacobian矩阵的计算从而显著降低运行时间，并结合常数单时间尺度学习率机制实现了良好的收敛性能。

    

    联邦随机双层优化因其在机器学习中的广泛应用，近年来受到广泛研究。然而，现有的大多数联邦随机双层优化算法需要计算二阶Hessian矩阵和Jacobian矩阵，这导致实际运行时间较长。为解决这些挑战，我们提出了一种新颖的联邦随机方差缩减双层梯度下降算法，该算法仅依赖于一阶oracle。具体而言，我们的方法不需要计算二阶Hessian矩阵和Jacobian矩阵，显著减少了运行时间。此外，我们引入了一种新颖的学习率机制，即常数单时间尺度学习率，用于协调不同变量的更新。我们还提出了一种新的策略来建立算法的收敛速率。最后，大量的实验结果证实了该算法的有效性。

    arXiv:2609.16350v1 Announce Type: new  Abstract: Federated stochastic bilevel optimization has been actively studied in recent years due to its widespread applications in machine learning. However, most existing federated stochastic bilevel optimization algorithms require the computation of second-order Hessian and Jacobian matrices, which leads to longer running times in practice. To address these challenges, we propose a novel federated stochastic variance-reduced bilevel gradient descent algorithm that relies solely on first-order oracles. Specifically, our approach does not require the computation of second-order Hessian and Jacobian matrices, significantly reducing running time. Furthermore, we introduce a novel learning rate mechanism, i.e., a constant single-timescale learning rate, to coordinate the update of different variables. We also present a new strategy to establish the convergence rate of our algorithm. Finally, the extensive experimental results confirm the efficacy of
    
[^117]: 面向海冰类型预测的多标签比例学习

    Multi-Label Proportion Learning for Sea-Ice Type Prediction

    [https://arxiv.org/abs/2609.16347](https://arxiv.org/abs/2609.16347)

    本文提出多标签比例学习方法，直接从多边形级别的冰情图标签中学习每个图块内各类海冰的比例分布，从而避免传统监督方法中因近似图块级标签而导致的病态学习问题。

    

    海冰类型预测对于气候监测、海上航行以及极地地区的决策制定非常重要。该任务标签数据的主要来源是冰情图，它由冰情分析员人工制作，他们通过解读卫星图像将冰区划分为多个多边形区域。尽管冰情图很有价值，但其制作过程耗费人力且成本高昂，这促使人们近来尝试利用深度学习将该过程自动化。然而，深度学习模型的训练需要图块级（或像素级）的标签数据，而冰情图仅提供多边形级别的标注。作为一种变通方法，监督学习方法通常通过为每个样本分配其所属多边形的主要冰类型，从多边形级别的冰情图标签中创建近似的图块级标签。这种方法虽然使监督训练成为可能，但会产生一个病态的学习问题，其解在本质上是近似的。在本文中，我们重新定义了海冰类型预测问题……

    arXiv:2609.16347v1 Announce Type: new  Abstract: Sea-ice type prediction is important for climate monitoring, maritime navigation, and decision-making in polar regions. The main source of label data for this task is the ice chart, produced manually by ice analysts who interpret satellite imagery to delineate ice zones into polygons. Although ice charts are valuable, their production is labor-intensive and expensive, motivating recent efforts to automate the process using deep learning. However, deep learning models require patch-level (or pixel-level) label data for training, while ice charts provide only polygon-level annotations. As a workaround, supervised approaches often create approximate patch-level labels from polygon-level ice chart labels by assigning each sample the dominant ice type of its parent polygon. This approach enables supervised training but creates an ill-posed learning problem with intrinsically approximate solution. In this paper, we redefine sea-ice type predic
    
[^118]: 信道信息驱动的神经网络物理层密钥生成

    Channel-Informed Neural Network for Physical Layer Key Generation

    [https://arxiv.org/abs/2609.16341](https://arxiv.org/abs/2609.16341)

    本文提出一种信道信息驱动的多任务循环神经网络，直接从接收的IQ测量中生成保持互易性的二进制密钥特征，并通过深度度量学习与信道信息监督相结合的方式实现轻量级的物理层密钥生成。

    

    物理层密钥生成（PKG）使无线设备能够从互易的信道观测中建立共享密钥，而无需直接交换密钥。这一能力对边缘网络极具吸引力，因为分布式且资源受限的设备可能需要在有限访问集中式基础设施的情况下进行轻量级的密钥建立。我们提出了一种用于物理层密钥生成的信道信息驱动神经网络，该网络直接从接收到的IQ测量数据中提取二进制密钥特征，同时将学习到的表示明确地建立在底层多径信道的基础上。所提出的多任务循环神经网络通过结合深度度量学习与信道信息监督的训练目标，联合学习保持互易性的二进制特征和辅助信道估计。结构化信道探测使得能够从空中测量中进行信道估计，同时使用Sionna-RT射线追踪来增强训练。

    arXiv:2609.16341v1 Announce Type: new  Abstract: Physical-layer key generation (PKG) enables wireless devices to establish shared keys from reciprocal channel observations without directly exchanging the key. This capability is attractive for edge networks, where distributed and resource-constrained devices may require lightweight key establishment with limited access to centralized infrastructure. We introduce a channel-informed neural network for PKG that derives binary key features directly from received IQ measurements while explicitly grounding the learned representation in the underlying multipath channel. The proposed multi-task recurrent neural network jointly learns reciprocity-preserving binary features and an auxiliary channel estimate using a training objective that combines deep metric learning with channel-informed supervision. Structured channel sounding enables channel estimation from over-the-air measurements, while Sionna-RT ray tracing is used to augment training wit
    
[^119]: StalePO：在机器翻译中利用旧式译后编辑的锚定token级偏好优化

    StalePO: Anchored Token-Level Preference Optimization using Legacy Post-Edits in Machine Translation

    [https://arxiv.org/abs/2609.16340](https://arxiv.org/abs/2609.16340)

    提出StalePO方法，通过在两个响应上同时下调似然、将策略锚定于自身基础响应以及施加token级KL约束，解决了机器翻译模型升级中利用陈旧偏好信号进行偏好优化时DPO失效的“陈旧偏好问题”。

    

    机器翻译系统会定期升级为更强的模型，但可用的偏好信号是对旧系统输出的人工译后编辑，而新模型可能已经超越了这些旧输出。此外，为每个新模型重新收集译后编辑的成本高得令人望而却步。我们将这一问题称为“陈旧偏好问题”。标准DPO在此场景下可能失效：它可能提高劣质译后编辑的似然，侵蚀模型已有的质量，并且无法提供纠正局部错误所需的逐token控制。我们提出了StalePO，这是一个由该场景的三个要求共同推导出的目标函数：似然必须在两个响应上都向下移动，策略必须锚定在其自身的基础响应上，且KL约束必须在token级别上施加。这些要求是共同必要、缺一不可的。在消融实验中，每种机制单独使用时，模型的表现与基础模型难以区分……

    arXiv:2609.16340v1 Announce Type: new  Abstract: Machine translation systems are periodically upgraded to stronger models, but the available preference signal is human post-edits of an older system's outputs, which the newer model may already surpass. Moreover, collecting fresh post-edits for every new model is prohibitively expensive. We call this the Stale Preference problem. Standard DPO can fail in this setting: it may increase the likelihood of inferior post-edits, erode the model's existing quality, and fail to provide the per-token control needed to correct localized errors. We introduce StalePO, an objective derived from three requirements this regime imposes. Likelihood movement must be downward on both responses, the policy must be anchored to its own base response, and the KL constraint must apply at the token level. These requirements are jointly necessary. In ablations, each mechanism in isolation leaves the model's performance indistinguishable from the base model, and on
    
[^120]: 突破三值大语言模型的1.58比特存储壁垒

    Breaking the 1.58-bit Barrier for Ternary LLMs

    [https://arxiv.org/abs/2609.16338](https://arxiv.org/abs/2609.16338)

    该论文发现三值LLM权重中零值占比高达51.5%，并据此提出分布自适应存储布局BITCOS，将每权重存储成本降至2-z比特，从而突破传统三值模型log₂3≈1.585比特的存储壁垒。

    

    三值大语言模型（LLM）将每个权重存储为三个符号{-1, 0, +1}之一，因此三值模型的存储成本通常以信息论上的每权重 log₂3 ≈ 1.585 比特作为参考基准。当前主流的部署格式将五个三值权重打包进一个字节（五三值位打包），并且由于实践中使用的二次幂分组大小，实际会向上取整为每权重1.625比特。这种有效存储位宽默认三个符号{-1, 0, +1}是等概率出现的。我们测量了29个三值LLM模型中符号的实际分布，发现零值权重占比高达51.5%。受这一发现启发，我们提出了BITCOS，一种简单的分布自适应存储布局，由一个密集的存在位图（presence bitmap）加一个压缩的符号向量组成，在模型权重的零值密度为 z 时，每个权重元素的存储成本仅为 2 - z 比特。BITCOS 能够比五三值位打包更紧凑地存储权重……（原文摘要在此处截断）

    arXiv:2609.16338v1 Announce Type: new  Abstract: Ternary Large Language Models (LLM) store every weight as one of three symbols $\{-1,0,+1\}$, so the cost of a ternary model is conventionally referenced to the information-theoretic $\log_2 3 \approx 1.585$ bits per weight. The prevailing deployment format packs five ternary weights into one byte (five-trit packing), and due to the power-of-two group sizes used in practice this rounds up to $1.625$ bits per weight. This effective storage bit-width treats the three symbols $\{-1,0,+1\}$ as equiprobable. We measure the actual symbol distribution of 29 ternary LLM models and find that zeros account for up to $51.5\%$ of all weights. Motivated by this finding, we introduce BITCOS, a simple distribution-adaptive layout comprised of a dense presence bitmap plus a compacted sign vector, and costs $2 - z$ bits per weight element given a zero density $z$ in the model's weights. BITCOS stores weights more compactly than the five-trit packing in 2
    
[^121]: 面向数字孪生的主动脉流固耦合代理模型中跨解剖结构迁移与稀疏插值的对比

    Cross-Anatomy Transfer Versus Sparse Interpolation in Digital-Twin-Oriented Aortic Fluid-Structure Interaction Surrogates

    [https://arxiv.org/abs/2609.16322](https://arxiv.org/abs/2609.16322)

    本研究通过对四个主动脉模型的双向流固耦合分析发现，仅基于几何的机器学习先验模型跨解剖结构零样本迁移效果不佳，而基于稀疏锚点的传统插值方法（如反距离加权）表现更好，表明数字孪生流固耦合代理模型的可信度评估必须区分跨解剖迁移与表面内插值这两种根本不同的泛化能力。

    

    流固耦合（FSI）代理模型的可信度要求区分跨独立解剖结构的迁移能力与已在采样表面内的插值能力。来自血管模型库的四个去标识化人体主动脉模型被重建为相互分离的管腔域和标称1.5毫米壁厚的壁面域，并在匹配的首周期双向流固耦合条件下进行分析。一个仅基于几何特征的LightGBM先验模型（通过在三个解剖结构上进行留一解剖结构验证开发而成）在第四个解剖结构上进行零样本评估，随后通过零样本后的稀疏场补全案例研究，对六个目标量进行了探究。结果显示，零样本迁移在所有目标量上均表现不佳。在百分之五的锚点水平下（203个锚点，3,852个评估节点），先验加自适应方法在振荡剪切指数（OSI）上达到R²为0.603。然而，仅在三个开发解剖结构上调优的相同锚点对照组在若干结果上表现更优：反距离加权方法达到了R² = 0.829。

    arXiv:2609.16322v1 Announce Type: new  Abstract: Surrogate credibility for fluid-structure interac- tion (FSI) requires distinguishing transfer across independent anatomies from interpolation within an already sampled surface. Four de-identified human aortic models from the Vascular Model Repository were reconstructed into separate lumen and nominal 1.5-mm wall domains and analyzed under matched first-cycle two-way FSI. A geometry-only LightGBM prior, selected by leave-one-anatomy-out development on three anatomies, was zero-shot evaluated on a fourth, then probed with a post-zero- shot sparse field-completion case study over six targets. Zero-shot transfer was poor across all targets. At a five-percent anchor level (203 anchors, 3,852 evaluation nodes), prior-plus-adaptation reached an oscillatory shear index (OSI) R2 of 0.603. However, same-anchor controls tuned only on the three development anatomies were stronger for several outcomes: inverse-distance weighting reached R2 = 0.829 (
    
[^122]: FairLint-DL：一个面向深度学习软件公平性调试的IDE原生工具

    FairLint-DL: An IDE-Native Tool for Fairness Debugging of Deep Learning Software

    [https://arxiv.org/abs/2609.16321](https://arxiv.org/abs/2609.16321)

    FairLint-DL是一个VS Code扩展工具，通过训练代理神经网络并应用基于信息论的QID指标，实现了在训练前直接在IDE中对表格数据集进行偏见检测、因果定位和可解释性分析的公平性调试。

    

    现有的公平性分析工具主要作为训练后评估框架运行，要求从业者在评估偏见之前必须完成完整的模型开发生命周期。我们提出了FairLint-DL，这是一个Visual Studio Code扩展，通过实现“左移”方法进行公平性测试，支持在训练之前直接对表格数据集进行IDE原生的偏见检测。FairLint-DL训练一个可配置的深度神经网络作为代理模型，并应用信息论的定量个体歧视（QID）指标。QID基于香农熵和最小熵，量化受保护属性对预测的因果影响。该系统实现了用于发现歧视性实例的两阶段梯度引导搜索算法、通过敏感性分析将偏见定位到特定网络层和神经元的因果调试流水线，以及使用SHAP和LIME进行特征级解释的双可解释性引擎。

    arXiv:2609.16321v1 Announce Type: cross  Abstract: Existing fairness analysis tools predominantly operate as post-training evaluation frameworks, requiring practitioners to complete the full model development lifecycle before assessing bias. We present FairLint-DL, a Visual Studio Code extension that implements a shift-left approach to fairness testing by enabling pre-training, IDE-native bias detection directly on tabular datasets. FairLint-DL trains a configurable deep neural network as a proxy model and applies information-theoretic Quantitative Individual Discrimination (QID) metrics. Grounded in Shannon and min-entropy, QID quantifies the causal influence of protected attributes on predictions. The system implements a two-phase gradient-guided search algorithm for discovering discriminatory instances, a causal debugging pipeline that localizes bias to specific network layers and neurons via sensitivity analysis, and dual explainability engines using SHAP and LIME for feature-level
    
[^123]: 面向基于模拟滤波的生成式模型：公式化框架与实证比较

    Generative models for simulation based filtering: Formulations and Empirical Comparisons

    [https://arxiv.org/abs/2609.16317](https://arxiv.org/abs/2609.16317)

    本文提出了非线性滤波问题的统一生成式模型框架，基于随机插值、流匹配和薛定谔桥推导出三种新滤波器，并与OTF、KRF、SIR粒子滤波器和EnKF等传统方法进行了系统实证比较。

    

    本文提出了一种统一的公式化框架，并对解决非线性滤波问题的生成式模型方法进行了受控的数值比较。在该框架下，分析步骤通过将预报分布传输到后验分布来实现，各种方法的区别仅在于如何选择和学习该传输。我们推导出了三种新的滤波器，分别基于随机插值、其确定性的流匹配极限，以及通过前向-后向随机微分方程实现的薛定谔桥。我们开发了一种两阶段调优程序，将生成式模型的训练与其在线细化分离开来。所得方法与最优传输滤波器（OTF）、Knothe-Rosenblatt滤波器（KRF）、序贯重要性重采样（SIR）粒子滤波器以及集合卡尔曼滤波器（EnKF）在精度、计算时间以及对集合大小和状态维度的敏感性等方面进行了比较。

    arXiv:2609.16317v1 Announce Type: new  Abstract: This letter presents a unified formulation and a controlled numerical comparison of generative-model approaches to the nonlinear filtering problem. Under this formulation the analysis step is realized by a transport of the forecast distribution to the posterior, the approaches differing only in how that transport is selected and learned. We derive three new filters, based on stochastic interpolants, their deterministic flow-matching limit, and Schr\"odinger bridges realized through forward--backward SDEs. We develop a two-stage tuning procedure that separates the training of the generative model from its online refinement. The resulting methods are compared against the optimal transport filter (OTF), the Knothe--Rosenblatt filter (KRF), the sequential importance resampling (SIR) particle filter and the ensemble Kalman filter (EnKF), in terms of accuracy, computational time, and sensitivity to ensemble size and state dimension. The result
    
[^124]: 基于自监督跨模态重构的机械多模态时间序列鲁棒故障检测

    Robust Fault Detection in Mechanical Multimodal Time Series via Self-Supervised Cross-Modal Reconstruction

    [https://arxiv.org/abs/2609.16314](https://arxiv.org/abs/2609.16314)

    提出一种基于自监督跨模态重构的多模态异常检测框架，通过利用跨模态关系显著提升机械多模态时间序列故障检测在分布偏移场景下的鲁棒性。

    

    故障检测在工业系统中至关重要，它能够及早识别异常行为，从而提高安全性、可靠性和运行效率。现代系统日益依赖异构传感模态来捕获底层物理过程的互补信息。然而，现有的数据驱动异常检测方法通常独立处理每个模态，或仅采用简单的特征级融合，这限制了其利用表征系统正常行为的跨模态关系的能力。此外，这些方法的性能通常假设训练和部署时的数据分布相似，而现实世界的运行会受到运行条件变化、环境影响和系统退化的影响，导致分布偏移并降低检测性能，尤其是在未见过的工况下。在这项工作中，我们提出了一种基于跨模态重构的多模态异常检测框架

    arXiv:2609.16314v1 Announce Type: new  Abstract: Fault detection is essential in industrial systems, enabling early identification of abnormal behaviour and improving safety, reliability, and operational efficiency. Modern systems increasingly rely on heterogeneous sensing modalities that capture complementary aspects of the underlying physical process. However, existing data-driven anomaly detection methods often process each modality independently or use simple feature-level fusion, limiting their ability to exploit cross-modal relationships that characterize normal system behaviour. Their performance also commonly assumes similar training and deployment distributions, whereas real-world operation is affected by changing operating conditions, environmental influences, and system degradation that induce distribution shifts and reduce detection performance, especially in unseen regimes.   In this work, we propose a multimodal anomaly detection framework based on cross-modal reconstruct
    
[^125]: 面向表格机器学习的智能体搜索空间

    Agentic Search Spaces for Tabular Machine Learning

    [https://arxiv.org/abs/2609.16309](https://arxiv.org/abs/2609.16309)

    本文证明基于LLM的智能体能够为表格机器学习模型设计扩展的超参数优化搜索空间，在45个数据集上平均带来0.6%的相对性能提升（小型数据集上达2.0%），超越模型作者提供的标准搜索空间。

    

    尽管基于大语言模型（LLM）的智能体在规划、代码生成和调试方面取得了快速进展，但它们在表格机器学习中的实际价值仍未得到充分探索。本文研究了一个具体的应用场景：最先进的智能体AI系统能否为成熟的表格模型设计出优于模型作者提供的标准搜索空间的扩展超参数优化（HPO）搜索空间。具体而言，我们将每个表格模型表示为涵盖预处理、嵌入、架构、训练和推理的模块化流水线。然后，我们让智能体为每个模块提出候选代码实现，并使用经典的HPO算法对这些候选实现和模型默认超参数进行联合优化。与基础HPO搜索空间相比，扩展后的搜索空间在45个数据集组成的测试套件上提升了几乎所有模型家族的性能，平均相对增益为0.6%，在小型数据集上更高达2.0%。

    arXiv:2609.16309v1 Announce Type: new  Abstract: Despite the rapid progress of LLM-based agents for planning, code generation, and debugging, their practical value for tabular machine learning remains underexplored. In this paper, we investigate a concrete use case: whether state-of-the-art agentic AI systems can design extended HPO search spaces for established tabular models that outperform the standard search spaces provided by the model authors. Specifically, we represent each tabular model as a modular pipeline covering preprocessing, embeddings, architecture, training, and inference. We then task the agent to propose candidate code implementations for each module and use a classical HPO algorithm to jointly optimize over these candidates and the model's default hyperparameters. Compared with the base HPO spaces, the expanded search spaces improve the performance of nearly every model family across a suite of 45 datasets, with average relative gains of 0.6%, rising to 2.0% on smal
    
[^126]: 巴拉塔纳蒂亚姆舞中的序列识别

    Sequence Recognition in Bharatnatyam dance

    [https://arxiv.org/abs/2609.16306](https://arxiv.org/abs/2609.16306)

    本文提出了一种结合CNN识别关键姿势、SVM识别动作，并利用编辑距离算法进行序列匹配的巴拉塔纳蒂亚姆舞Adavu舞蹈序列识别方法，最终识别准确率达98%。

    

    巴拉塔纳蒂亚姆舞是最古老的印度古典舞蹈（ICD），在印度及世界各地被学习和练习。Adavu是这种舞蹈形式的核心。目前存在15种Adavu和58种变体。每个Adavu变体包含一组明确定义的动作和姿势（称为舞步），它们按特定顺序出现。因此，在学习Adavu时，学生不仅要学习舞步，还要注意舞步出现的顺序。本文提出了一种识别这些序列的方法。在这项工作中，首先，我们分别使用卷积神经网络（CNN）和支持向量机（SVM）来识别Adavu中涉及的关键姿势（KP）和动作。其中，CNN达到了99%的识别率，SVM的识别准确率为84%。接下来，我们使用编辑距离算法将这些关键姿势和动作序列与真实值进行比较以找到最佳匹配，准确率达到98%。该论文对……的现状做出了重大贡献。

    arXiv:2609.16306v1 Announce Type: cross  Abstract: Bharatanatyam is the oldest Indian Classical Dance (ICD) which is learned and practiced across India and the world. Adavu is the core of this dance form. There exist 15 Adavus and 58 variations. Each Adavu variation comprises a well-defined set of motions and postures (called dance steps) that occur in a particular order. So, while learning Adavus, students not only learn the dance steps but also take care of its sequence of occurrences. This paper proposed a method to recognize these sequences. In this work, firstly, we recognize the involved Key Postures (KPs) and motions in the Adavu using Convolutional Neural Network (CNN) and Support Vector Machine (SVM), respectively. In this, CNN achieves 99% and SVM's recognition accuracy becomes 84%. Next, we compare these KP and motion sequences with the ground truth to find the best match using the Edit Distance algorithm with an accuracy of 98%. The paper contributes hugely to the state-of-
    
[^127]: BLINDSPOT：面向长程工具使用智能体的安全与拒绝校准基准测试

    BLINDSPOT: A Benchmark for Safety and Refusal Calibration in Long-Horizon Tool-Using Agents

    [https://arxiv.org/abs/2609.16305](https://arxiv.org/abs/2609.16305)

    本文提出Blindspot基准测试，通过自适应对抗交互、有状态工具执行和基于执行的裁决来评估长程工具使用智能体在完整交互轨迹上的安全与拒绝校准能力，涵盖22个攻击类别、7个领域35个场景及2,500余条长程轨迹。

    

    大语言模型（LLM）智能体越来越多地在涉及工具使用、持久状态、不断演变的授权以及外部环境反馈的长程交互中运行。在这样的环境中，安全故障可能仅在多轮交互之后才显现，然而现有评估往往将智能体行为简化为任务成功或攻击成功，从而掩盖了随着交互的演变，智能体究竟是执行了操作、予以拒绝，还是保持了适当的校准。我们提出了Blindspot，这是一个针对长程工具使用智能体的轨迹级安全校准基准测试。Blindspot通过自适应对抗交互、有状态工具执行以及基于执行的裁决来评估完整的用户-智能体-环境轨迹。其当前实例包含22个攻击类别和覆盖七个领域的35个场景，产生了超过2,500条长程轨迹，平均交互长度为14.7轮。每条轨迹被赋予……（原文摘要至此截断）

    arXiv:2609.16305v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly operate over long-horizon interactions involving tool use, persistent state, evolving authorization, and external environment feedback. In such settings, safety failures may emerge only after multiple turns, yet existing evaluations often reduce agent behavior to task or attack success, obscuring whether an agent acts, refuses, or remains appropriately calibrated as the interaction evolves. We introduce Blindspot, a benchmark for trajectory-level safety calibration of long-horizon tool-using agents. Blindspot evaluates complete user-agent-environment trajectories through adaptive adversarial interaction, stateful tool execution, and execution-grounded adjudication. Its current instantiation contains 22 attack families and 35 scenarios across seven domains, yielding more than 2,500 long-horizon trajectories with an average interaction length of 14.7 turns. Each trajectory is assigned one of f
    
[^128]: 全国一致，局部缺失：基于贝叶斯遥感的屋顶光伏登记系统审计

    Nationally Consistent, Locally Incomplete: A Bayesian Remote-Sensing Audit of Rooftop Photovoltaic Registries

    [https://arxiv.org/abs/2609.16294](https://arxiv.org/abs/2609.16294)

    本文提出一个贝叶斯遥感审计框架，将不完美的光伏检测结果转化为考虑不确定性的测量工具，在法国估计的屋顶光伏容量与官方数据全国偏差仅3.3%，却揭示出局部地区高达61%的容量漏报及开放数据中的显著截断偏差。

    

    追踪能源转型进程需要对可再生能源部署情况的可靠统计数据。屋顶光伏由于其分散的特性而尤其难以追踪，由此导致的官方统计数据不准确问题已为人所知，但尚未被量化。遥感提供了一种独立识别屋顶光伏系统的方法。我们提出了一个贝叶斯框架，从遥感检测结果中估计屋顶光伏的真实装机容量，将一个不完美的检测器转化为一个考虑不确定性的测量工具。该方法应用于法国后，校正后的检测结果估计36 kWp以下的屋顶光伏容量为4.03 GWp [3.96–4.11]（99%可信区间），与输电系统运营商的并网数据在全国范围内相差仅3.3%，同时识别出局部地区高达当地容量61%的漏报情况。我们还记录并量化了法国屋顶光伏开放数据中存在的显著截断偏差。除法国外，该方法（原文在此处截断）

    arXiv:2609.16294v1 Announce Type: cross  Abstract: Tracking the energy transition requires reliable statistics on renewable deployment. Rooftop photovoltaics (PV) are especially hard to track, owing to their decentralised nature, and the resulting inaccuracies in official statistics are known but not quantified. Remote sensing offers an independent way to identify rooftop PV systems. We introduce a Bayesian framework to estimate the ground-truth rooftop PV capacity from remote sensing detections, turning an imperfect detector into an uncertainty-aware measurement instrument. Applied to France, the corrected detections estimate a capacity of 4.03 GWp [3.96--4.11] (99% credible interval) of rooftop PV below 36 kWp, matching the transmission system operator's connection data within 3.3% nationally, while identifying local under-reports of up to 61% of local capacity. We also document and quantify a significant truncation bias in French rooftop PV open data. Beyond France, the approach pav
    
[^129]: 重复和谐博弈的贝尔曼最优方程的对称解

    Symmetric solution of the Bellman optimality equation for repeated harmony game

    [https://arxiv.org/abs/2609.16289](https://arxiv.org/abs/2609.16289)

    本文通过求解重复和谐博弈的贝尔曼最优方程，发现了三种类型的对称解，分别对应全合作策略、“赢则保持输则改变”策略以及一种具有非平凡行为的新策略，并通过数值实验研究了智能体实际学到的策略。

    

    在社会困境博弈中，额外的奖励或惩罚已被研究作为促进合作的手段。因此，研究这种额外收益能够改变博弈的理想情况是十分重要的。在本研究中，我们研究了重复和谐博弈的贝尔曼最优方程的对称解。计算表明存在三种类型的对称解。其中一种对应于平凡的“全合作”策略，另一种对应于囚徒困境博弈中的“赢则保持、输则改变”策略。我们还详细讨论了对应于最后一种解的策略的非平凡行为。此外，我们通过数值方法研究了智能体通过强化学习算法实际学习到的是哪种策略。

    arXiv:2609.16289v1 Announce Type: cross  Abstract: In social dilemma games, additional rewards or punishments have been studied as means of promoting cooperation. Therefore, it is important to investigate the ideal situation, in which such an additional payoff would change the game. In this study, we investigated the symmetric solution of the Bellman optimality equation for a repeated harmony game. The calculations showed that three types of symmetric solutions exist. One of them corresponds to the trivial All-C strategy, and another to the Win-stay Lose-shift strategy of the prisoners dilemma game. The nontrivial behavior of the strategy corresponding to the last solution is also discussed in detail. In addition, we numerically investigated which strategy the agents actually learn by the reinforcement learning algorithm.
    
[^130]: 漂移场网络：从现场观测和卫星观测中学习海洋拉格朗日平流场

    Drift Field Net: Learning Ocean Lagrangian advection fields from in-situ and satellite observations

    [https://arxiv.org/abs/2609.16288](https://arxiv.org/abs/2609.16288)

    提出漂移场网络（DFN），通过模拟数据预训练结合平流一致性损失的拉格朗日微调这一两阶段物理信息训练策略，从卫星观测预测海洋表面流场，使7天预报的平均定位误差比业务化物理预报系统降低20公里。

    

    北太平洋副热带环流（NPSG）是由海盆尺度辐合海洋环流形成的漂浮塑料垃圾主要聚集区。该区域的有效清理策略依赖于对拉格朗日粒子漂移的准确预报。本文提出了漂移场网络（DFN），这是一种利用业务化卫星观测数据预测海洋表面流场的深度神经网络。DFN采用一种新颖的两阶段训练策略，将模拟数据预训练与基于平流一致性损失函数的拉格朗日微调相结合。这种物理信息驱动的优化直接提高了粒子轨迹预测的精度。我们将DFN与基于物理的业务化预报系统进行对比评估，展示了深度学习在海洋表面流预测方面的潜力。在实际漂流浮标轨迹上，与业务化系统相比，DFN在7天预报后将平均定位误差降低了20公里。

    arXiv:2609.16288v1 Announce Type: new  Abstract: The North Pacific Subtropical Gyre (NPSG) is a major accumulation zone for floating plastic debris, resulting from basin-scale convergent ocean circulation. Effective cleanup strategies in this region rely on accurate forecasts of Lagrangian particle drift. Here, we introduce Drift Field Net (DFN), a deep neural network that predicts ocean surface flow fields from operational satellite observations. DFN is trained using a novel two-stage strategy that combines pretraining on simulated data with Lagrangian fine-tuning based on an advection-consistent loss function. This physics-informed optimization directly improves the accuracy of particle trajectory predictions. We evaluate DFN against an operational physics-based forecasting system and demonstrate the potential of deep learning for ocean surface flow prediction. On in situ drifter trajectories, DFN reduces the mean positioning error by 20 km after a 7-day forecast compared with the op
    
[^131]: 用于聚合洞察生成的差分隐私语义计划

    Differentially Private Semantic Plans for Aggregate Insight Generation

    [https://arxiv.org/abs/2609.16283](https://arxiv.org/abs/2609.16283)

    提出DP-SPIN可信管理者框架，通过将记录映射为语义概念上的有界稀疏非负向量形成语义草图，并借助差分隐私机制发布语义计划，实现了对独立于数据定义的语义概念在跨数据集合和重复分析中可比较的聚合度量与摘要。

    

    \texttt{URANIA}为依赖数据的簇摘要提供端到端的差分隐私（DP）保护。然而，其簇-关键词发布方式并不能直接为独立于受保护语料库定义的语义概念提供集合层面的聚合统计。由于单条记录可能表达多个概念、表达相同概念的记录可能被分配到不同的簇、且簇的标识在不同分析之间未必一致，簇级统计无法直接提供跨数据集合或跨重复分析时对预定义概念的可比较度量。我们提出了\texttt{DP-SPIN}，这是一个可信管理者框架，用于对独立于受保护目标记录而预先固定的语义概念进行聚合度量与摘要。每条记录被映射为这些概念上的一个有界稀疏非负向量，所有向量的总和构成一个语义草图（semantic sketch）。随后，一个差分隐私机制发布包含聚合信息的语义计划……

    arXiv:2609.16283v1 Announce Type: new  Abstract: \texttt{URANIA} provides end-to-end differential privacy (DP) for summaries of data-dependent clusters. However, its cluster--keyword release does not directly provide collection-wide aggregates for semantic concepts defined independently of the protected corpus. Records may express several concepts, records expressing the same concept may be assigned to different clusters, and cluster identities need not correspond across analyses. Consequently, cluster-level statistics do not directly provide comparable measurements of predefined concepts across collections or repeated analyses. We introduce \texttt{DP-SPIN}, a trusted-curator framework for aggregate measurement and summarization over semantic concepts fixed independently of the protected target records. Each record is mapped to a bounded sparse nonnegative vector over these concepts, whose sum forms a semantic sketch. A differentially private mechanism releases a semantic plan contain
    
[^132]: 物理感知的ACOPF代理学习缩放定律

    Scaling Laws for Physics-Aware ACOPF Surrogate Learning

    [https://arxiv.org/abs/2609.16282](https://arxiv.org/abs/2609.16282)

    在ACOPF代理学习中，物理感知的增广拉格朗日目标相比MSE能以可忽略的额外内存将约束违反降低近30倍，且其改善在模型容量与数据规模之间更均衡，而MSE的违反量随网络规模增长快约两倍。

    

    基于学习的交流最优潮流（ACOPF）代理模型相比经典求解器有望带来大幅加速，但其实际应用价值不仅取决于预测精度，还同样取决于物理可行性。以增广拉格朗日（AL）为代表的物理感知目标函数能够改善约束满足度，但会增加每一步的额外计算成本，然而这种权衡随规模变化的表现尚未被系统表征。我们在MSE和AL两种训练方式下对模型与数据集规模进行扫描，并刻画了约束违反量如何随不同电网的网络规模变化。两种目标函数均呈幂律改善，但速率不同：MSE主要由模型容量主导，而AL则在模型容量与数据规模之间更为均衡。在MSE下，违反量随网络规模的增长速度约为AL下的两倍。在相同硬件条件下，AL以多一个数量级的训练时间为代价，将违反量降低近30倍，且额外内存开销可忽略不计。训练目标不仅决定了代理模型（原文在此处截断）。

    arXiv:2609.16282v1 Announce Type: new  Abstract: Learning-based surrogates for AC optimal power flow (ACOPF) promise large speedups over classical solvers, but their operational value depends on physical feasibility as much as predictive accuracy. Physics-aware objectives such as the augmented Lagrangian (AL) improve constraint satisfaction at additional per-step cost, yet how this trade-off behaves with scale is uncharacterized. We sweep model and dataset sizes under both MSE and AL training, and characterize how constraint violation changes with network size across grids. Both objectives improve as power laws, but at different rates: MSE is governed primarily by model capacity, while AL is balanced across both. Violation grows roughly twice as fast with network size under MSE as under AL. On matched hardware, AL reduces violation by nearly $30\times$ for an order of magnitude more training time, with negligible added memory. The training objective determines not only where a surrogat
    
[^133]: 面向抗误码低延迟传输的语义感知神经视频编解码器

    Semantic-Aware Neural Video Codec for Error-Resilient Low-Latency Transmission

    [https://arxiv.org/abs/2609.16279](https://arxiv.org/abs/2609.16279)

    该论文提出一种基于DCVC-RT的语义感知多级神经视频编解码方法，通过语义与特征感知的分级打包优先级策略和消除包间依赖的抗误码熵模型，实现了不可靠信道下鲁棒的低延迟视频传输。

    

    新兴的物理AI系统需要在不可靠信道上进行低延迟、面向任务的视频通信。我们提出了一种语义感知的多级神经视频编码方法，用于在不可靠信道（抽象为多级丢包信道）上实现鲁棒的低延迟视频传输。该框架建立在实时DCVC-RT神经视频编解码器之上，引入了一种语义和特征感知的编码策略，将编码表示划分为携带不同语义与潜在特征重要性级别的数据包，并将这些数据包分配到不同的流中，每个流在通过不可靠通信信道传输时都具有相应的优先级。我们还开发了一种抗误码熵模型，消除了数据包之间的依赖关系，使每个数据包在发生丢包时能够被独立解码。整个系统在抽象的多级丢包信道上进行端到端训练。

    arXiv:2609.16279v1 Announce Type: cross  Abstract: Emerging physical AI systems require low-latency, task-oriented video communication over unreliable channels. We propose a semantic-aware multi-level neural video coding method for robust low-latency video transmission over unreliable channels that are abstracted as multi-level packet erasure channels. Built upon the real-time DCVC-RT neural video codec, the proposed framework introduces a semantic- and feature-aware coding strategy that partitions encoded representations into packets carrying different levels of semantic and latent-feature importance and assigns these packets to different streams, each associated with a priority level when transmitted over unreliable communication channels. We also developed an error-resilient entropy model that removes inter-packet dependencies, allowing each packet to be decoded independently under packet losses. The complete system is trained end-to-end over the abstracted multi-level packet erasur
    
[^134]: 记录是任务的一部分：跨维护、安全和召回报告的文本分类器匹配记录评估

    The record is part of the task: matched-record evaluation of text classifiers across maintenance, safety and recall reporting

    [https://arxiv.org/abs/2609.16267](https://arxiv.org/abs/2609.16267)

    该论文的核心发现是：记录选择本身就是文本分类器评估任务的关键组成部分——同一案例的不同工作流记录（如客户报告与技术员报告）之间的F1差异可达0.46，远大于在同一数据上比较不同表示方法和模型架构所带来的性能差异。

    

    许多运营案例会在不同的工作流程阶段、出于不同的目的被多次记录，然而模型评估通常在模型比较开始之前就从这些记录中选择其中一个。我们将这种选择本身视为评估的一部分，在三个系统中、在固定的标签和数据划分下比较同一案例的匹配记录：GE航空维修事件、NASA ASRS安全报告和NHTSA车辆召回。在GE的三个字段中，对于标签独立于叙述文本、来源于零件交易的事件，留出集上的宏F1在0.33到0.91之间变化。在车间工作开始前撰写的客户报告与在诊断之后、但在产生标签的交易之前撰写的技术员报告之间，F1差异高达0.46。这一差异远大于在同一批事件上测试的各种表示方法和模型架构之间的差异。公开数据系统则呈现出不同的模式：NHTSA缺陷报告……

    arXiv:2609.16267v1 Announce Type: cross  Abstract: Many operational cases are documented more than once, at different workflow stages and for different purposes, yet model evaluations normally select one of these records before model comparison begins. We treat that selection as part of the evaluation and compare matched records of the same cases under fixed labels and splits in three systems: GE Aerospace repair events, NASA ASRS safety reports and NHTSA vehicle recalls. Across the three GE fields, for events whose label comes from parts transactions independently of the narratives, held-out macro-F1 ranged from 0.33 to 0.91. A difference of 0.46 separated the customer report, written before shop work, from the technician report, written after diagnosis but before the transaction that generates the label. That difference is substantially larger than the representation and architecture differences tested on the same events. The public systems showed different patterns: the NHTSA defect
    
[^135]: 迈向基于代理模型的量子强化学习去量子化

    Towards Surrogate Based Dequantization of Quantum Reinforcement Learning

    [https://arxiv.org/abs/2609.16266](https://arxiv.org/abs/2609.16266)

    该论文基于监督学习中核方法的去量子化成果，首次探索将基于代理模型的去量子化方法扩展至强化学习领域，通过构建高效经典算法来匹配量子Q学习等量子变分方法的性能，以检验量子强化学习对实际问题是否具有可证明的量子优势。

    

    近年来，参数化量子电路作为函数逼近器的效用得到了广泛研究。在强化学习的背景下，这种方法催生了变分量子算法，例如量子Q学习。虽然这些方法展现出有前景的实证结果，并且能够为人工构造的问题提供可证明的优势，但目前尚不清楚它们能否为具有实际意义的问题提供相对于经典方法可证明的量子优势。研究这一问题的一个自然途径是通过去量子化的视角：即构建能够匹配量子变分方法性能的高效经典算法。基于近期监督学习中基于核方法的去量子化成果，我们采取措施将这种基于代理模型的去量子化计划扩展到强化学习领域。具体而言，我们研究了强化学习的简化设定……

    arXiv:2609.16266v1 Announce Type: cross  Abstract: In recent years, the utility of parameterized quantum circuits as function approximators has been widely studied. In the context of reinforcement learning, this approach has led to variational quantum algorithms such as quantum Q-learning. While these methods show promising empirical results, and can provide provable advantages for artificial problems, it remains unclear whether they can provide a provable quantum advantage over classical approaches for problems of practical relevance. A natural way to investigate this question is through the lens of dequantization: The construction of efficient classical algorithms capable of matching the performance of quantum variational methods. Building on recent kernel-based dequantization results for supervised learning, we take steps towards extending this surrogate-based dequantization program to reinforcement learning. Specifically, we study the simplified setting of reinforcement learning wi
    
[^136]: 岭梯度下降中的计算最优预训练—微调策略

    Compute-Optimal Pretrain--Fine-tune in Ridge Gradient Descent

    [https://arxiv.org/abs/2609.16262](https://arxiv.org/abs/2609.16262)

    本文在岭回归梯度下降的两阶段预训练—微调框架下，首次从理论上刻画了固定总优化预算时上游预训练与下游微调之间的最优计算分配，并揭示该分配由预测相关的谱分量和下游数据几何共同决定。

    

    预训练之后进行微调会引入一个计算分配问题：在固定训练预算下，用于提升上游目标的计算会减少可用于下游适配的计算。尽管这一权衡在实践中十分重要，但即使是在简单模型中，其理论理解仍然不足。本文将这一分配问题转化为一个在总优化预算固定的两阶段预训练—微调流程下的计算拆分问题，并以由梯度下降训练的正则化最小二乘（岭回归）作为一个可解析处理的设置。我们刻画了在由微调问题所诱导的数据相关评估几何下的最优拆分。结果表明，计算分配取决于预训练方向如何影响微调预测，以及微调偏移如何通过下游数据几何被观测。特别地，相关量由与预测相关的谱分量所决定。

    arXiv:2609.16262v1 Announce Type: cross  Abstract: Pretraining followed by fine-tuning introduces a compute-allocation problem: under a fixed training budget, compute spent improving the upstream objective reduces the compute available for downstream adaptation. Despite its practical importance, this trade-off is not yet well understood theoretically, even in simple models. In this paper, we cast this allocation as a compute-split problem under a two-stage pretrain--fine-tune procedure with fixed total optimisation budget, using regularised least squares trained by gradient descent as a tractable setting. We characterise the optimal split under data-dependent evaluation geometries induced by the fine-tuning problem. Our results show that the allocation depends on how pretraining directions affect fine-tuning predictions and how fine-tuning shifts are seen through downstream data geometry. In particular, the relevant quantities are determined by prediction-relevant spectral components o
    
[^137]: AI赋能的科学前沿

    The AI-Enabled Scientific Frontier

    [https://arxiv.org/abs/2609.16258](https://arxiv.org/abs/2609.16258)

    该研究通过分析2000至2025年间27个学科中2507项AI与传统科学方法的直接对比，发现AI相对传统统计学通常性能更优但成本更高，而相对科学计算通常性能较差但成本更低，不过自2020年以来AI在后者上的表现已显著提升。

    

    随着人工智能能力的不断提升，它越来越被视为一种通用的科学方法。但这些说法的真实性如何？AI是否在所有技术上表现优异，还是仅在部分技术上如此？这一状况又是如何演变的？为了评估这些说法，我们汇集了2000年至2025年初发表的论文中，来自27个科学学科的2507项AI与其他科学分析技术之间的直接对比数据。我们发现了一个深刻的二分法：相对于传统统计学方法，AI往往表现更优，但计算成本显著更高。然而，也有近四分之一的情况下，AI既比传统统计技术更昂贵，性能又更差，且这一比例在过去十年中一直保持稳定。相对于科学计算方法，AI往往表现欠佳，但计算成本更低。这一局面已开始发生变化：自2020年以来，AI相对于科学计算的性能显著增强。

    arXiv:2609.16258v1 Announce Type: new  Abstract: As artificial intelligence's capabilities improve, it is increasingly viewed as a general scientific method. But how true are these claims? Does AI outperform all techniques, or only some, and how is this changing? To assess the claims, we assemble a corpus of 2,507 head-to-head comparisons between AI and other scientific analysis techniques across 27 scientific disciplines from papers published between 2000 and early 2025. We find a profound dichotomy. Relative to traditional statistics, AI often outperforms, but at a significantly higher computational cost. But there are also nearly a quarter of cases where AI is both more expensive and performs worse than traditional statistical techniques and this fraction has been stable for a decade. Relative to scientific computing, AI often underperforms, but at lower computational cost. This has begun to change: since 2020, AI's performance against scientific computing has notably strengthened a
    
[^138]: 高效推理蒸馏：基于合成思维链与难度感知微调的小型视频语言模型

    Efficient Reasoning Distillation: Small Video-Language Models via Synthetic CoT and Difficulty-Aware Fine-Tuning

    [https://arxiv.org/abs/2609.16255](https://arxiv.org/abs/2609.16255)

    该论文提出一种高效蒸馏方法，仅用约900个样本和单卡A100不到两小时的训练，让2B小型视频语言模型通过4B教师生成的合成CoT和“CoT后置于答案”的策略，超越4倍大的模型并逼近其4B教师模型的性能。

    

    我们提出了一种高效的方法，将推理能力蒸馏到用于视频问答（VideoQA）的紧凑视频语言模型（VLM）中。我们的方法仅使用约900个通过不确定性筛选的样本对一个2B参数模型进行微调，每个样本都配有由4B教师模型生成的合成思维链推理依据。尽管计算成本极低——在单张A100 GPU上不到两小时——我们的方法使该2B模型能够超越规模大至4倍的VLM，并在CinePile、ActivityNet-QA和MLVU等数据集上展现出泛化能力，性能接近其自身4B教师模型。一个关键发现是，将思维链推理依据放在答案之后——与标准提示方法相反——能显著提升紧凑模型的推理能力。这一发现挑战了主流的思维链使用惯例，并揭示了在有限模型容量下的新对齐策略。我们的研究成果为训练可部署的、具备丰富推理能力的VLM提供了实用的蓝图。

    arXiv:2609.16255v1 Announce Type: cross  Abstract: We present an efficient method to distill reasoning capabilities into compact video-language models (VLMs) for video question answering (VideoQA). Our approach fine-tunes a 2B-parameter model using only $\sim$900 uncertainty-selected examples, each augmented with synthetic chain-of-thought (CoT) rationales generated by a 4B teacher. Despite its minimal compute cost - under two hours on a single A100 GPU - our method enables the 2B model to outperform VLMs up to 4$\times$ larger, and generalize across CinePile, ActivityNet-QA, and MLVU, approaching the performance of its own 4B teacher. A key finding is that placing CoT rationales after the answer - contrary to standard prompting - substantially improves reasoning in compact models. This insight challenges prevailing CoT conventions and reveals new alignment strategies under limited model capacity. Our findings offer a practical blueprint for training deployable, reasoning-rich VLMs sui
    
[^139]: 用于生物医学数据聚类表示的Copula自适应有向无环图

    Copula Adapted Directed Acyclic Graph for Cluster Representation of Biomedical Data

    [https://arxiv.org/abs/2609.16240](https://arxiv.org/abs/2609.16240)

    本文提出了一种融合Copula非高斯非线性依赖建模与基于有向无环图的集成因果结构发现方法的新型数据表示框架，用于无标签高维生物医学数据的聚类表示。

    

    诊断错误和标签误标在生物医学领域十分常见，这损害了预测模型和数据驱动结果的可靠性。基于特征之间的复杂关系对无标签生物医学数据进行分层，能够消除对数据标签的需求，并克服监督学习的局限性。传统聚类方法假设数据分布具有较强的限制性，因此在捕捉高维生物医学数据中的复杂依赖关系方面表现欠佳。本文提出了一种新颖的面向聚类的数据表示框架，该框架将Copula模型的非高斯和非线性特征依赖建模与基于有向无环图（DAG）的集成因果结构发现（CSD）方法相结合。Copula通过放宽多元正态性、线性依赖和对称关系等假设来建模灵活的多元分布，而基于DAG的集成因果结构发现方法能够识别……

    arXiv:2609.16240v1 Announce Type: cross  Abstract: Diagnostic errors and mislabeling are common in biomedicine, which compromise the reliability of predictive models and data-driven outcomes. Stratifying unlabeled biomedical data based on complex relationships between features eliminates the need for data labels and overcomes the limitations of supervised learning. Traditional clustering methods assume restrictive data distributions, making them suboptimal for capturing complex dependencies in high-dimensional biomedical data. This paper introduces a novel cluster-friendly data presentation framework that integrates the non-Gaussian and non-linear feature dependence of copula models with an ensemble of causal structure discovery (CSD) methods based on Directed Acyclic Graphs (DAGs). While copulas model flexible multivariate distributions by relaxing assumptions related to multivariate normality, linear dependence, and symmetric relationships, an ensemble of DAG-based CSD methods identi
    
[^140]: 利用数据同化与机器学习改进旋转爆震发动机降阶模型

    Improving Reduced-Order Rotating Detonation Engine Models with Data Assimilation and Machine Learning

    [https://arxiv.org/abs/2609.16237](https://arxiv.org/abs/2609.16237)

    该论文提出利用连续数据同化（推引）方法将低维Koch-Kutz模型与高保真度模拟的温度数据同步，并以记录的强迫项作为模型修正的状态依赖估计，从而在低计算成本下显著改进旋转爆震发动机降阶模型的预测能力。

    

    旋转爆震发动机（RDE）表现出强烈的非线性、多尺度波动动力学，这些动力学决定了所观测到的热场。高保真度模拟（DNS/LES）能够解析这些结构，但计算成本过于高昂；而诸如一维Koch-Kutz模型等低阶模型虽然能够捕捉周向波动运动，却缺乏表达高频内容的能力。我们采用连续数据同化（nudging推引）方法，将Koch-Kutz求解器与处理后的高保真度温度数据同步，并在守恒能量方程中将预测值与观测值之间的失配作为松弛源引入；当观测数据在时间上较为稀疏时，通过插值方法为每次源更新提供目标值。随着推引强度的增大，降阶模型逐渐被拉向高保真度轨迹，而沿该轨迹记录的强迫项为模型所需的修正提供了显式的、依赖于状态的估计。

    arXiv:2609.16237v1 Announce Type: cross  Abstract: Rotating detonation engines (RDEs) exhibit strongly nonlinear, multiscale wave dynamics that set the observed thermal field. High-fidelity simulations (DNS/LES) resolve these structures but remain computationally prohibitive, while low-order models such as the one-dimensional Koch-Kutz model capture circumferential wave motion yet lack the expressivity for high-frequency content. We use continuous data assimilation (nudging) to synchronize the Koch-Kutz solver with processed high-fidelity temperature data, introducing the prediction-observation mismatch as a relaxation source in the conserved energy equation; where observations are temporally sparse, interpolation supplies a target at every source update. As the nudging strength increases, the reduced model is progressively drawn onto the high-fidelity trajectory, and the forcing recorded along it provides an explicit, state-dependent estimate of the correction the model requires. We t
    
[^141]: 基于稀疏自编码器的测试时机器遗忘

    Test-Time Unlearning via Sparse Autoencoder

    [https://arxiv.org/abs/2609.16229](https://arxiv.org/abs/2609.16229)

    提出ARIA方法，利用稀疏自编码器检测遗忘相关状态并进行可解释干预，在完全不改模型权重的前提下实现测试时知识遗忘，有效缓解了传统梯度上升方法的遗忘-效用权衡问题。

    

    机器遗忘旨在从训练好的大语言模型（LLM）中移除特定知识，而无需从头重新训练。现有方法通过梯度上升及其改进变体来修改模型权重。尽管这些基于权重的方法在某些基准测试上有效，但它们表现出明显的遗忘-效用权衡：对目标知识的遗忘越强，模型效用下降越严重，且被遗忘的知识可能在遗忘后的微调或提示词攻击下重新出现。我们提出ARIA（基于自编码器门控的推理时遗忘），一种测试时机器遗忘方法，它保持模型权重完全不变，仅在生成过程进入与遗忘相关的状态时才限制对不需要知识的访问。ARIA利用稀疏自编码器（SAE）的潜在表示训练一个轻量级线性检测器，然后在被触发的状态上应用可解释的干预，测试时开销可忽略不计。在TOFU、R-TOFU和WMDP上的实证评估表明，ARIA提升了...

    arXiv:2609.16229v1 Announce Type: cross  Abstract: Machine unlearning aims to remove specific knowledge from a trained large language model (LLM) without retraining from scratch. Existing methods modify model weights via gradient ascent and its advances. While effective on certain benchmarks, these weight-based approaches exhibit a sharp forget-utility trade-off, where stronger forgetting of target knowledge can degrade model utility, and unlearned knowledge may reappear under post-unlearning fine-tuning or prompt attacks. We propose ARIA (autoencoder-gated inference-time unlearning), a test-time unlearning method that leaves model weights intact and gates access to unwanted knowledge only when generation enters a forget-related state. ARIA uses sparse autoencoder (SAE) latents to train a lightweight linear detector, then applies an interpretable intervention on triggered states with negligible test-time overhead. Empirical evaluations on TOFU, R-TOFU, and WMDP show that ARIA improves 
    
[^142]: 我如何学会停止担忧并爱上停止梯度：平稳性、收敛性以及流映射学习的案例研究

    How I learned to stop worrying and love StopGrads: Stationarity, Convergence, and a case study on Flow Map Learning

    [https://arxiv.org/abs/2609.16222](https://arxiv.org/abs/2609.16222)

    该论文提出了停止梯度回归原理，统一了流映射、强化学习和扩散采样中的停止梯度目标，并从理论上证明了停止梯度流映射目标的唯一驻点是真实流映射，同时给出了正面的收敛性保证。

    

    停止梯度被广泛用于训练机器学习模型，但停止梯度会改变原始目标的梯度、驻点和收敛保证，这可能使停止梯度训练在理论上缺乏依据。我们引入了一种停止梯度回归原理，它为停止梯度目标确定了一个通用模板，能够对驻点及其唯一性进行闭式刻画，从而统一了流映射、强化学习和扩散采样器的停止梯度目标。我们为优化停止梯度流映射目标提供了理论基础，证明其唯一驻点就是真实流映射，并给出了欧拉型和拉格朗日型目标（包括MeanFlow和改进的MeanFlow）的积极收敛结果。值得注意的是，我们证明了在函数半梯度流下，学习得到的流映射具有一个由初始流映射与真实流映射复合而成的闭式表达式。此外，我们……

    arXiv:2609.16222v1 Announce Type: new  Abstract: Stopgrads are widely used in training machine learning models, but stopgrads can alter the gradient, stationary points and convergence guarantees of the original objective, which can make stopgrad training theoretically ungrounded. We introduce a stopgrad regression principle, which identifies a general template for stopgrad objectives with a closed-form characterization of stationary points and their uniqueness, unifying stopgrad objectives for flow maps, reinforcement learning, and diffusion samplers. We provide theoretical grounding for optimizing stopgrad flow map objectives by showing their unique stationary point is the true flow map, and showing positive convergence results for Eulerian and Lagrangian objectives, including MeanFlow and improved MeanFlow. Remarkably, we show that under functional semi-gradient flow, the learned flow map has a closed-form expression composing the initial flow map and the true flow map. We additional
    
[^143]: 诱饵方向优化：一种针对大语言模型消融攻击的事后防御方法

    Decoy Direction Optimization: A Post-Hoc Defense Against LLM Abliteration

    [https://arxiv.org/abs/2609.16204](https://arxiv.org/abs/2609.16204)

    提出诱饵方向优化（DDO），一种无需微调的快速事后权重编辑防御方法，通过向MLP神经元注入高幅度非线性诱饵信号来破坏消融攻击所依赖的对比估计器，从而保护大语言模型的安全防护栏免受拒绝特征消融攻击。

    

    开放权重语言模型中的安全防护栏很容易被拒绝特征消融（RFA）技术绕过，该技术通过识别并将残差流中的线性拒绝方向投影出去来破解安全机制，通常在保持模型能力的同时实现较高的攻击成功率（ASR）。防御这类攻击通常需要对每个新模型检查点进行计算成本高昂的安全微调。我们提出了诱饵方向优化（DDO），一种快速的、无需基础模型微调的事后权重编辑防御方法。我们的方法基于一个简单的机制性洞察：消融攻击依赖对比估计器来寻找拒绝方向。DDO并不试图隐藏真正的拒绝回路，而是主动向网络的MLP神经元中注入高幅度的非线性诱饵信号。当攻击者试图定位拒绝方向时，诱饵会破坏他们的估计器，诱使他们消融（诱饵方向而非真正的拒绝方向）。

    arXiv:2609.16204v1 Announce Type: cross  Abstract: Safety guardrails in open-weight language models can be readily bypassed using Refusal Feature Ablation (RFA), a technique that identifies and projects out a linear refusal direction from the residual stream, often achieving a high attack success rate (ASR) while preserving model capability. Defending against these attacks typically requires computationally expensive safety finetuning for every new checkpoint. We introduce Decoy Direction Optimization (DDO), a fast, post-hoc weight-editing defense that requires no base-model finetuning. Our approach is based on a simple mechanistic insight: ablation attacks rely on contrastive estimators to find the refusal direction. Rather than trying to hide the true refusal circuitry, DDO actively injects a high-magnitude, nonlinear decoy signal into the network's MLP neurons. When an attacker attempts to locate the refusal direction, the decoy corrupts their estimator, tricking them into ablating 
    
[^144]: 一个涵盖25起加州野火、面向深度学习活跃火分割的Sentinel-2基准数据集

    A Sentinel-2 benchmark dataset for deep-learning active-fire segmentation across 25 California wildfires

    [https://arxiv.org/abs/2609.16199](https://arxiv.org/abs/2609.16199)

    该论文发布了一个涵盖25起加州野火、包含2,148对图像-掩膜的开放Sentinel-2基准数据集，用于推动和评估深度学习活跃火分割方法的开发。

    

    本文描述了一个开放的图像数据集，用于开发和评估卫星图像中的活跃火分割方法。该数据集包含来自25起加州野火的2,148对图像-掩膜对，采集时间跨度为2020年7月至2026年8月。每张图像是由Sentinel-2 Level-2A波段的B12、B11和B8A在20米空间采样下导出的512x512像素三通道合成图像，整个数据集采用固定的线性渲染。对应的掩膜区分背景、基于SWIR规则的活跃火以及无效观测。掩膜由短波红外亮度和近红外对比度生成，随后进行受限的邻域生长。发布的版本包括切片级元数据和事件不相交的数据划分，包含18起训练火、3起验证火和4起测试火。在所有图像对中，841对包含活跃火标签，这些标签占所有网格单元的0.0766%。掩膜盲态分析员审查（摘要在此处截断）……

    arXiv:2609.16199v1 Announce Type: cross  Abstract: This article describes an open image dataset for developing and evaluating active-fire segmentation methods in satellite imagery. The dataset contains 2,148 image-mask pairs from 25 California wildfires, with acquisitions spanning July 2020 to August 2026. Each image is a 512x512-pixel, three-channel composite derived from Sentinel-2 Level-2A bands B12, B11 and B8A at 20 m spatial sampling. A fixed linear rendering is applied throughout the dataset. Corresponding masks distinguish background, SWIR-rule active fire and invalid observations. The masks were generated from shortwave-infrared brightness and near-infrared contrast, followed by constrained neighborhood growth. The release includes chip-level metadata and an incident-disjoint partition containing 18 training, three validation and four test fires. Among the image pairs, 841 contain active-fire labels; these labels occupy 0.0766% of all grid cells. A mask-blind analyst review co
    
[^145]: 大语言模型中基于排列的隐写恶意软件：威胁与对策

    Permutation-Based Stegomalware in Large Language Models: Threats and Countermeasures

    [https://arxiv.org/abs/2609.16193](https://arxiv.org/abs/2609.16193)

    本文证明排列对称性既能被防御者用于完全中和LLM所有权重中的隐写恶意软件，也能被攻击者利用以理论不可检测的方式将恶意软件编码进模型权重。

    

    大语言模型（LLM）训练的高难度及其广泛普及，引发了隐写恶意软件（stegomalware）的威胁，即恶意载荷被嵌入到模型权重之中。近期的研究已展示了利用模型权重中的排列对称性来缓解这些威胁的方法，但未能证明可以在LLM的所有权重上实现对隐写恶意软件的中和。在本文中，我们展示了行为保持对称性作为抵御隐写恶意软件防御手段的全部潜力，同时也揭示了这些对称性被攻击者利用时所构成的风险。在隐写恶意软件中和方面，我们对先前的工作进行了改进，证明可以选取能够移动所有模型参数的排列。这与之前的方法形成鲜明对比，因为先前的方法会在LLM中留下相当大比例的权重未被改动。而当排列对称性被用于攻击时，我们表明其能够以理论上（不可检测的）方式将恶意软件编码到模型权重中……

    arXiv:2609.16193v1 Announce Type: cross  Abstract: The difficulty of training large language models (LLMs), together with their ubiquity, raises the threat of stegomalware, where malicious payloads are embedded into model weights. Recent work has demonstrated the use of permutation symmetry in model weights to mitigate these threats, but failed to show neutralization of stegomalware across all weights for LLMs. In this paper, we demonstrate the full potential of behavior-preserving symmetries as a defense against stegomalware, as well as the risks these symmetries pose when exploited by attackers.   For stegomalware neutralization, we improve upon previous work, demonstrating that it is possible to select permutations which displace all model parameters. This contrasts with previous methods which left a significant percentage of weights unaltered in LLMs. When used in an attack, we show that permutation symmetries can encode malware into the weights of a model in a way that is theoreti
    
[^146]: 固定状态循环网络中联想回忆机制的解剖：匹配状态分解、干扰墙以及打破它的课程学习

    Anatomy of Associative Recall in Fixed-State Recurrences: A Matched-State Decomposition, an Interference Wall, and a Curriculum That Breaks It

    [https://arxiv.org/abs/2609.16183](https://arxiv.org/abs/2609.16183)

    该论文通过匹配状态预算的单变量分解，揭示了固定状态循环网络在联想回忆上的差距主要由短因果卷积而非循环结构类型决定，识别出限制多查询回忆的“干扰墙”现象，并提出课程学习方法来突破这一瓶颈。

    

    固定状态循环网络——线性注意力和状态空间模型——据报道在联想回忆任务上落后于注意力机制，但整体架构层面的比较无法说明究竟是哪个组件导致的。我们在固定状态预算下，沿三个单变量控制轴对掩码多查询回忆任务进行了分解：短因果卷积、转移结构（秩1 delta规则 vs 对角结构）以及衰减。研究表明卷积起主导作用（在匹配训练条件下，两个模型家族中均带来约+0.5的回忆率提升）：那些将无卷积单元与配备卷积的Mamba进行比较的实验，实际衡量的是缺失的卷积，而非循环结构本身。秩1转移在16/32对设置下比对角消融版本高出+0.19/+0.32，但一旦两个单元都配备卷积，优势便缩小至+0.03，且状态匹配的Mamba-2与无卷积的秩1单元打成平手：没有任何关于模型类别的主张能够成立。能够解决32对回忆任务的单元在负载增加时性能平缓下降，却在检索4对时跌至随机水平（原文此处截断）

    arXiv:2609.16183v1 Announce Type: cross  Abstract: Fixed-state recurrences--linear attention and state-space models--are reported to lag behind attention on associative recall, but whole-architecture comparisons cannot say which ingredient is responsible. We decompose masked multi-query recall at a fixed state budget along three single-knob axes: a short causal convolution, the transition structure (rank-1 delta rule vs. diagonal), and decay. The convolution dominates (~+0.5 recall in both families under matched training): comparisons that pit convolution-free cells against a convolution-equipped Mamba measure the missing convolution, not the recurrence. The rank-1 transition beats its diagonal ablation by +0.19/+0.32 at 16/32 pairs, but the margin shrinks to +0.03 once both cells carry the convolution, and a state-matched Mamba-2 ties the unarmed rank-1 cell: no class claim survives. Cells that solve 32-pair recall degrade gracefully with load yet fall to chance retrieving 4 pairs fro
    
[^147]: 密集输出头与稀疏路由器中的Z-Loss反向传播几何

    Z-Loss Backward Geometry in Dense Output Heads and Sparse Routers

    [https://arxiv.org/abs/2609.16179](https://arxiv.org/abs/2609.16179)

    本文从反向传播视角重新审视Z-loss，指出其产生的logit空间梯度（“反向源”）对密集输出头和稀疏路由器的影响取决于其所处的架构与实现方式。

    

    Z-loss已被广泛应用于语言模型输出头的logits和稀疏专家混合路由器中。Z-loss约束这些输出头和路由器的softmax对数归一化因子，从而限制大logit偏移、降低有限精度舍入误差的影响，并避免训练损失发散。这些应用场景出现在现代Transformer环境中，其中大词汇量softmax输出头、top-k路由、融合损失和混合精度优化器相互作用。Z-loss通常仅被理解为对对数归一化因子的标量惩罚。本文则从反向传播的角度分析Z-loss，重点关注Z-loss惩罚所产生的梯度。这个位于logit空间的梯度（我们称之为反向源）被注入到反向传播中Z-loss分支的logit边界处；因此，反向源的作用效果取决于梯度所经过的架构和实现方式。

    arXiv:2609.16179v1 Announce Type: cross  Abstract: Z-loss has been widely applied to the logits of language-model output heads and sparse mixture-of-experts routers. Z-loss constrains the softmax log-normalizers of these output heads and routers, thereby limiting large-logit excursions, reducing finite-precision roundoff exposure, and avoiding training-loss divergence. These use cases arise in modern Transformer settings where large-vocabulary softmax heads, top-$k$ routing, fused losses, and mixed-precision optimizers interact. Z-loss has typically been understood only as a scalar penalty on the log-normalizer. This paper instead analyzes Z-loss from a backward-pass perspective, focusing on the gradients produced by the Z-loss penalty. The logit-space gradient, which we call the backward source, is injected at the logit boundary of the Z-loss branch of backpropagation; consequently, the backward source's effect depends on the architecture and implementation through which the gradient 
    
[^148]: 迭代神经扩张上的骨架原型

    Skeletal Prototypes on Iterative Nerve Expansions

    [https://arxiv.org/abs/2609.16170](https://arxiv.org/abs/2609.16170)

    SPINE方法创新性地用嵌入的一维复形（骨架结构）而非传统点集来表示各类原型，通过类条件Mapper图构建初始边集并在分类目标下优化顶点位置，使骨架线段直接参与决策规则，在17个基准数据集上取得了最优的平均准确率和排名。

    

    原型约简是用一个更小的表示来替换训练集，而现有方法返回的是一个有限的点集。我们提出了迭代神经扩张骨架原型方法（SPINE）。该方法中每个类别的模型是一个嵌入的一维复形，而非点集。其初始边集是一个类条件Mapper图，因此由数据本身决定哪些局部聚类被连接在一起。后续阶段在分类目标下对顶点进行拟合，并将观测样本分配给其复形距离最近的类别。因此，这些线段不仅参与拟合过程，还直接进入决策规则。我们在17个基准数据集上，采用分层10折交叉验证，在相同预算条件下与七种其他原型约简方法进行对比来评估SPINE。SPINE获得了最高的平均准确率和最佳的平均排名。在经Holm校正的Wilcoxon符号秩检验下，它显著优于七个竞争方法中的五个。

    arXiv:2609.16170v1 Announce Type: new  Abstract: Prototype reduction replaces a training set with a smaller representation, and the established methods return a finite set of points. We propose Skeletal Prototypes on Iterative Nerve Expansions (SPINE). The model for each class is an embedded 1-complex rather than a point set. Its initial edge set is a class-conditional Mapper graph, so the data decide which localized clusters are joined. Later phases fit the vertices under a classification objective, and an observation is assigned to the class whose complex is nearest. The segments therefore enter the decision rule and not only the fitting. We evaluate SPINE on seventeen benchmark datasets under stratified 10-fold cross validation, against seven other prototype reduction methods at a matched budget. SPINE attains the highest mean accuracy and the best average rank. It is significantly better than five of the seven competitors under Wilcoxon signed-rank tests with Holm correction. A bud
    
[^149]: GPEvac：基于图神经网络的PPO用于枪击事件中的自适应疏散路径规划

    GPEvac: GNN-Based PPO for Adaptive Evacuation Routing During Shooting Events

    [https://arxiv.org/abs/2609.16163](https://arxiv.org/abs/2609.16163)

    该论文提出GPEvac框架，通过边优先顺序消息传递与可学习虚拟全局节点的图神经网络结合PPO强化学习，并利用置换不变评分机制，实现了单一策略在不同拓扑和规模建筑布局间泛化的枪击事件自适应疏散路径规划。

    

    大规模枪击事件的急剧增加凸显了对能够实时引导受害者到达安全地点的系统的迫切需求。一个有效的疏散系统必须在最小化威胁暴露的同时，兼顾对抗性不确定性和拥挤动态。文献中的现有方法被严格限制在特定布局的策略上，且在大规模布局中计算上难以处理，而实用指南只是简单地建议受害者“逃跑”、“躲藏”或“反击”。我们提出了GPEvac：一个基于图神经网络的PPO框架，用于在枪击事件期间计算自适应疏散路线。为了捕捉局部和长距离依赖关系，我们引入了一种带有可学习虚拟全局节点的边优先顺序消息传递方案。所得到的图嵌入被集成到一个置换不变的评分机制中，使单一学习策略能够跨不同拓扑结构和规模的建筑布局运行。通过广泛的...

    arXiv:2609.16163v1 Announce Type: new  Abstract: The sharp increase in mass shootings underscores an urgent need for systems that guide victims to safety in real time. An effective evacuation system must minimize threat exposure while also accounting for adversarial uncertainty and crowding dynamics. Current methods in the literature are rigidly constrained to layout-specific policies and computationally intractable in large-scale layouts, while practical guidelines simply advise victims to "run", "hide", or "fight". We propose GPEvac: a GNN-based PPO framework that computes adaptive evacuation routes during shooting events. To capture both local and long-distance dependencies, we introduce an edge-first sequential message-passing scheme with a learnable virtual global node. The resulting graph embeddings are integrated into a permutation-invariant scoring mechanism that allows a single learned policy to operate across building layouts of diverse topologies and sizes. Through extensive
    
[^150]: 闪存中的LLM推理！

    LLM Inference in a Flash!

    [https://arxiv.org/abs/2609.16161](https://arxiv.org/abs/2609.16161)

    该论文提出利用闪存内计算作为解决LLM推理所面临的内存带宽与容量瓶颈的方案，以应对长序列、重推理请求带来的日益增长的挑战。

    

    大型语言模型（LLM）在一系列自然语言处理任务中展现出了令人印象深刻的能力，LLM推理已成为支撑下游应用的关键工作负载。随着检索增强生成、推理时计算扩展和长上下文应用的驱动，推理请求趋向于更长的序列和更繁重的计算，LLM推理服务的需求正变得日益严峻。此外，硬件趋势进一步加剧了这些挑战，因为内存容量和通信带宽的增长速度跟不上工作负载复杂度的提升。闪存内计算是一种有前景的解决方案，它通过将计算移至靠近内存的位置来应对内存带宽限制，并能充分利用SSD技术的大容量优势。然而，在这些系统上部署LLM极具挑战性，因为它们缺乏对高精度浮点运算的支持，且写入能力有限……

    arXiv:2609.16161v1 Announce Type: new  Abstract: Large Language Models (LLMs) have shown impressive capabilities across a range of natural language processing tasks, and LLM inference has emerged as a critical workload for enabling downstream applications. The demands of serving LLM inference are becoming increasingly challenging as requests shift toward longer sequences and heavier inference, driven by retrieval-augmented generation, inference-time compute scaling, and long-context applications. Additionally, these challenges are compounded by hardware trends, as memory capacity and communication bandwidth are not scaling as fast as increases in workload complexity. Compute-in-Flash is a promising solution to address memory bandwidth limitations by moving computation close to memory, and to exploit the large capacity of SSD technologies. However, it is challenging to deploy LLMs on these systems as they lack support for high-precision floating point operations and have limited write e
    
[^151]: 三维周期性Navier-Stokes流非线性族的计算机辅助全局正则性

    Computer-assisted global regularity across nonlinear families of three-dimensional periodic Navier-Stokes flows

    [https://arxiv.org/abs/2609.16157](https://arxiv.org/abs/2609.16157)

    本文开发了一种计算机辅助框架，通过将有限参考轨迹与公共误差界相结合，为三维周期性Navier-Stokes流的连续非线性族建立了全局正则性，并成功应用于循环剪切场、ABC场和三组分Taylor-Green场。

    

    数值模拟揭示了涡旋如何拉伸和传递能量，但要建立光滑演化则需要超出模拟分辨率仍然有效的界。本文开发了一个计算机辅助框架，为三维周期性Navier-Stokes流的连续族建立全局正则性。其核心构造将有限条参考轨迹与一个公共误差界相结合，该误差界同时覆盖中心场的区间和无穷多个光滑扰动模态。该方法在谱截断之前保留完整的非线性残差，并控制演化过程，直到粘性衰减保证后续所有时间的正则性。该方法应用于循环剪切场、Arnold-Beltrami-Childress场和三组分Taylor-Green场，得到了显式的扰动半径，并且包含了不满足直接Fourier-Wiener小性准则的初始条件。此外，参数一致的扩展覆盖了一个连通族。

    arXiv:2609.16157v1 Announce Type: cross  Abstract: Numerical simulations reveal how vortices stretch and transfer energy, but establishing smooth evolution requires bounds that remain valid beyond the simulated resolution. Here I develop a computer-assisted framework that establishes global regularity for continuous families of three-dimensional periodic Navier-Stokes flows. Its central construction combines finite reference trajectories with a common error bound that covers an interval of centre fields and infinitely many smooth perturbation modes. The method retains the complete nonlinear residual before spectral truncation and controls the evolution until viscous decay guarantees regularity for all subsequent times. Applications to cyclic-shear, Arnold-Beltrami-Childress and three-component Taylor-Green fields yield explicit perturbation radii and include initial conditions outside the direct Fourier-Wiener smallness criterion. A parameter-uniform extension covers a connected family
    
[^152]: 大语言模型作为伪造大师：为制造业生成合成时间序列数据

    LLMs as Master Forgers: Generating Synthetic Time Series Data for Manufacturing

    [https://arxiv.org/abs/2609.16155](https://arxiv.org/abs/2609.16155)

    本文提出通过微调大语言模型并结合检索增强生成（RAG）技术为制造业生成高质量合成时间序列数据，在异常检测等下游任务中表现优于ARIMA和LSTM等传统方法。

    

    本文提出了一个利用大语言模型（LLM）为制造过程生成合成时间序列数据的新颖框架。鉴于现实制造环境中标注时间序列数据的稀缺性阻碍了鲁棒机器学习模型的发展，我们探索了LLM学习复杂时间依赖关系并生成逼真合成数据的潜力。我们的方法包括在制造过程指令上微调预训练的LLM，并采用检索增强生成（RAG）技术来提升数据的多样性和真实性。我们使用定量指标、PCA分析和下游任务性能（异常检测），将该方法与ARIMA和LSTM等传统时间序列建模技术进行对比评估。结果表明，我们基于LLM的框架优于这些基线方法，能够生成有效捕捉（原文截断）的高质量合成时间序列数据。

    arXiv:2609.16155v1 Announce Type: cross  Abstract: This paper presents a novel framework leveraging Large Language Models (LLMs) to generate synthetic time series data for manufacturing processes. Motivated by the scarcity of labeled time-series data in real-world manufacturing settings, which hinders the development of robust machine learning models, we explore the potential of LLMs to learn complex temporal dependencies and generate realistic synthetic data. Our approach involves fine-tuning pre-trained LLMs on manufacturing process instructions and employing a Retrieval Augmented Generation (RAG) technique to enhance data diversity and realism. We evaluate our method against traditional time series modeling techniques like ARIMA and LSTMs, using quantitative metrics, PCA analysis, and downstream task performance (anomaly detection). Results demonstrate that our LLM-driven framework outperforms these baselines, generating high-quality synthetic time series data that effectively captu
    
[^153]: 语言模型的安全错误纠正：保持能力的冻结底座调整

    Safe Error Correction for Language Models: Frozen-Base Adjustment with Capability Preservation

    [https://arxiv.org/abs/2609.16145](https://arxiv.org/abs/2609.16145)

    提出仅占模型参数 0.73% 的轻量级纠正模块 CRN v2，能在完全冻结的语言模型上修复 53.3% 的输出错误且不损害基础能力，而同预算的 LoRA 虽纠正率更高却会导致 30-75% 的能力损失。

    

    我们研究了一个实际问题：一个小的纠正模块能否在不损害冻结语言模型基础能力的前提下修复其输出中的错误？我们提出了 CRN v2，一个轻量级的 logit 层面纠正模块（约 3400 万可训练参数，仅占 4.65B 文本模块的 0.73%），它叠加在一个完全冻结的 Gemma 4 E2B 模型之上。底座模型从不更新；只有纠正模块进行学习，方式是先进行监督微调，再在 83,400 个错误纠正对上执行无参考 DPO。在一个 60 题的领域考试（CEHRI：认证人机智能，涵盖事实、算术和隐式目标推理）上，CRN v2 纠正了底座模型 53.3% 的错误（改写变体：43.3%），同时在受测能力基准上没有出现能力退化（MMLU/BoolQ N=200；car-wash N=8）。而在与 CRN v1 相同预算（660 万参数，rank 19）下的 LoRA 基线虽实现了 83.3% 的纠正率，却在相同基准上遭受了 30-75% 的能力损失——这种纠正与能力之间的权衡……

    arXiv:2609.16145v1 Announce Type: new  Abstract: We study a practical question: can a small correction module fix errors in a frozen language model's outputs without degrading its base capabilities? We propose CRN v2, a lightweight logit-level correction module (~34M trainable parameters, 0.73% of the 4.65B text module) that sits atop a fully frozen Gemma 4 E2B model. The base model is never updated; only the correction module learns, via supervised fine-tuning followed by reference-free DPO on 83,400 error-correction pairs. On a 60-question domain exam (CEHRI: Certified Human-Robot Intelligence, covering facts, arithmetic, and implicit-goal reasoning), CRN v2 corrects 53.3% of base-model errors (reworded variant: 43.3%) while showing no degradation on tested capability benchmarks (MMLU/BoolQ N=200; car-wash N=8). A LoRA baseline at the matched CRN v1 budget (6.6M params, rank 19) achieves 83.3% correction but suffers 30-75% capability loss on the same benchmarks -- the correction-capa
    
[^154]: 一种用于代理标签信用风险预测中监督漂移的决策支持审计协议

    A Decision-Support Audit Protocol for Supervision Drift in Proxy-Labeled Credit-Risk Prediction

    [https://arxiv.org/abs/2609.16102](https://arxiv.org/abs/2609.16102)

    该论文提出一个针对代理标签信用风险预测中监督漂移的锁定式五层多信号审计协议，在LendingClub数据上发现排序性能稳定，时间漂移主要源于基础比率与概率尺度的失配，且可通过仅截距再校准大幅缓解。

    

    信用风险模型基于代理标签进行训练，并在时间变化与客群（分段）变化下部署，然而目前尚无单一的迁移指标能够区分基础比率漂移、概率尺度漂移以及特征-标签关系变化。我们贡献了一个设计科学产物：一个针对代理标签信用风险预测中监督漂移的锁定式多信号审计协议。五个层次（迁移性能、预言机差距探针、校准诊断、特征-标签稳定性以及合成阳性对照）、阈值和决策规则均在解读之前预先锁定；有界解读（bounded reading）是刻意设计的产物。在一个公开的LendingClub数据集上（2013至2016年的时间迁移以及跨客群迁移），排序性能稳定且预言机差距很小；最明显的时间信号是流行率（基础比率）与概率尺度之间的失配，仅截距的诊断性再校准即可在很大程度上消除该失配，但其成因无法从现有发布数据中识别。合成阳性对照

    arXiv:2609.16102v1 Announce Type: cross  Abstract: Credit-risk models are trained on proxy labels and deployed under temporal and segment change, yet no single transfer metric separates base-rate shift, probability-scale shift, and feature-label relationship change. We contribute a design-science artifact: a locked, multi-signal audit protocol for supervision drift in proxy-labeled credit-risk prediction. Five layers (transfer performance, an oracle-gap probe, a calibration diagnostic, feature-label stability, and a synthetic positive control), thresholds, and decision rules were locked before interpretation; a bounded reading is a designed outcome. On a public LendingClub dataset (temporal 2013 to 2016 and cross-segment transfer), ranking is stable and oracle gaps are small; the clearest temporal signal is a prevalence and probability-scale mismatch that intercept-only diagnostic recalibration largely reduces, though its cause is not identifiable from the available release. The positi
    
[^155]: SWB-DM：一种面向部分参与下拜占庭鲁棒联邦学习的带延迟动量缓存的校准切片Wasserstein重心聚合器

    SWB-DM: A Calibrated Sliced-Wasserstein-Barycenter Aggregator with Delayed-Momentum Caching for Byzantine-Robust Federated Learning under Partial Participation

    [https://arxiv.org/abs/2609.16099](https://arxiv.org/abs/2609.16099)

    该论文提出SWB-DM，通过带规范固定的切片Wasserstein重心聚合器结合对全体客户端的延迟动量缓存机制，解决了部分参与场景下鲁棒聚合有限样本保证失效的问题，实现了拜占庭鲁棒的联邦学习。

    

    联邦学习中的鲁棒聚合方法悄然依赖一个脆弱的假设：即某一轮中出现的客户端是全体客户端的公平样本。但在实践中，情况往往并非如此。当每轮只有少数客户端参与时，即使是一小部分恶意攻击者也可能主导该样本，并悄无声息地使坐标中位数、Krum、Bulyan和截尾均值等方法所依赖的有限样本保证失效。我们提出SWB-DM来直接解决这一问题。SWB将客户端更新的每个切片视为一维分布，计算跨客户端的截尾Wasserstein重心，并通过基于中心点的规范固定步骤恢复坐标身份——这是我们开发的一种启发式方法，我们不声称它属于标准的最优传输理论。随后，DeMoA风格的延迟动量机制在每轮中将更新缓存在完整的客户端群体上，使鲁棒性与实际被抽中的客户端解耦。（摘要在此处被截断）

    arXiv:2609.16099v1 Announce Type: new  Abstract: Robust aggregation methods for federated learning quietly rest on a fragile assumption: that whoever shows up in a given round is a fair sample of the full population. In practice, they rarely are. When only a handful of clients participate per round, even a modest fraction of adversaries can dominate that sample and silently invalidate the finite-sample guarantees that coordinate-wise median, Krum, Bulyan, and trimmed mean all depend on.   We introduce SWB-DM to address this directly. SWB treats each slice of a client update as a one-dimensional distribution, computes a trimmed Wasserstein barycenter across clients, and recovers coordinate identity via a medoid-based gauge-fixing step -- a heuristic we developed and do not claim it belongs to standard optimal-transport theory. DeMoA-style delayed momentum then caches updates across the full client population each round, decoupling robustness from whoever happened to be sampled. Trim rat
    
[^156]: 基于环境锚定验证的开源权重电商智能体评估

    Evaluating Open-Weight E-Commerce Agents with Environment-Grounded Verification

    [https://arxiv.org/abs/2609.16093](https://arxiv.org/abs/2609.16093)

    该论文构建了一个确定且可复现的电商评估环境，通过预先固定试验参数并记录助手动作与环境状态的对应证据，实现了超越单一任务成功率的、基于环境锚定的细粒度智能体对话评估。

    

    一次购物对话可以有多条路径到达同一个购物车，而任务成功率将所有这些路径压缩成单一的分数。我们构建了一个确定且可复现的电商环境，该环境预先固定每次试验的客户和轨迹参数，包括用户画像、难度、目标购物车以及商品揭示时间表。一个模拟的消费者在被评估模型的辅助下，尝试从该环境中购买目标购物车。该环境引导模拟器的动作，并记录助手每一动作及其当时对应的环境状态。试验结束后，这些记录使评估者能够依据所保留的证据对对话的各个部分进行评估。例如，只有当客户已经提到某个目标商品时，评估者才会因搜索未能展示该目标商品而对搜索动作进行惩罚。我们进一步利用这些证据，根据助手的动作对工具调用施加不同的惩罚。

    arXiv:2609.16093v1 Announce Type: new  Abstract: A shopping conversation has many routes to the same cart, and a task-success rate reduces all of them to one score. We build a deterministic and reproducible e-commerce environment that precommits each trial's customer and trajectory parameters, including the persona, difficulty, target cart, and an item reveal schedule. A simulated consumer attempts to buy a target cart from the environment with assistance from the evaluated model. The environment guides the simulator's actions and records every assistant action alongside the environment state at that point. After the trial, these records allow the evaluator to assess individual parts of the conversation against the retained evidence. For example, the evaluator penalizes a search for failing to surface a target product only when the customer has already mentioned that product. We further use this evidence to apply different penalties to tool calls depending on how the assistant's action
    
[^157]: 面向智能体假设推理的基础模型蒸馏：混合LLM+SLM架构中的成本、延迟与治理

    Distilling Foundation Models for Agentic What-If Reasoning:Cost, Latency, and Governance in a Hybrid LLM+SLM Architecture

    [https://arxiv.org/abs/2609.16091](https://arxiv.org/abs/2609.16091)

    本文通过将TabPFN表格基础模型蒸馏为紧凑的前馈学生网络，实现了高达6,532倍的参数压缩，同时保留95%以上的预测性能，使表格基础模型能够作为混合LLM+SLM架构中智能体假设推理的低延迟热路径决策后端。

    

    表格型基础模型通过上下文学习实现了强大的零训练预测性能，但其高推理延迟使其难以作为交互式智能体循环中热路径决策后端。我们将TabPFN教师模型蒸馏为紧凑的前馈学生模型，并在UCI Adult和五个OpenML基准上进行了业务决策仿真：分类头将5,320万参数压缩至8,546个（压缩6,220倍）；部署的双头贷款流水线将1.114亿参数压缩至17,059个（压缩6,532倍）。学生模型保留了95.4-100.5%的准确率和96.8-100.0%的AUC，其中credit-g数据集上的准确率保持率最低，为95.4%；alpha = 0的硬标签对照实验表明，教师模型的软目标可带来2.1-7.0个AUC点的增益。

    arXiv:2609.16091v1 Announce Type: new  Abstract: Tabular foundation models deliver strong zero-training predictive performance via in-context learning, but their high inference latency makes them impractical as hot-path decision backends in interactive agentic loops. We distill a TabPFN teacher into a compact feed-forward student across a business-decision simulation on UCI Adult and five OpenML benchmarks: the classification head compresses 53.2M parameters to 8,546 (6,220x); the deployed two-head loan pipeline compresses 111.4M parameters to 17,059 (6,532x). The student retains 95.4-100.5% accuracy and 96.8-100.0% AUC, with the lowest accuracy retention on credit-g at 95.4%; an alpha = 0 hard-label control shows that the teacher's soft targets provide a 2.1-7.0 AUC point gain.
    
[^158]: INT8 可移植吗？嵌入式与车载加速器上量化推理的跨平台测量研究

    Is INT8 Portable? A Cross-Platform Measurement Study of Quantized Inference on Embedded and Automotive Accelerators

    [https://arxiv.org/abs/2609.16085](https://arxiv.org/abs/2609.16085)

    INT8 量化模型并非跨平台可移植：相同模型在不同硬件上加速效果截然不同（有无点积指令集可导致 2.1 倍加速或 1.7 倍减速），且 INT8 推理输出在不同平台间不保证一致，打破了“一次量化、随处部署”的普遍假设。

    

    八位整数（INT8）训练后量化是边缘部署的默认方案，其背后有一个广泛持有的假设：INT8 能以较小且可预测的精度损失换取更快的推理速度，且模型只需量化一次便可迁移到任何目标平台。我们通过一项受控的测量研究检验该假设，涵盖七类硬件——ARM 和 x86 CPU、一块独立 GPU、NVIDIA Jetson AGX Orin 的 iGPU 及其 NVDLA 核心，以及两家厂商的 NPU（高通 Hexagon HTP、DEEPX DX-M1）——同时固定 ONNX 模型文件和量化缩放因子，使整数内核或指令集架构（ISA）成为唯一的自由变量。研究结果表明可移植性在三个维度上失效。(1) INT8 加速效果的正负由 CPU 的点积指令集决定（ARM dotprod/SDOT、x86 VNNI）：对于完全相同的模型和运行时，具备该指令集的核心最高可加速 2.1 倍，而缺乏该指令集的核心反而减速 1.7 倍。(2) INT8 的输出结果不可移植，且其规律表现为一种不变性而非梯度：FP32……（原文摘要在此处截断）

    arXiv:2609.16085v1 Announce Type: cross  Abstract: Eight-bit integer (INT8) post-training quantization is the default recipe for edge deployment, under a widely held assumption: INT8 makes inference faster at a small, predictable accuracy cost, and a model quantized once can be carried to any target. We test that assumption with a controlled measurement study across seven hardware classes -- ARM and x86 CPUs, a discrete GPU, an NVIDIA Jetson AGX Orin iGPU and its NVDLA cores, and two vendor NPUs (Qualcomm Hexagon HTP, DEEPX DX-M1) -- holding the ONNX artifact and the quantization scales fixed so the integer kernel or ISA is the only free variable. Portability fails on three axes. (1) The sign of the INT8 speedup is set by the CPU's dot-product ISA (ARM dotprod/SDOT, x86 VNNI): cores that have it speed up by up to 2.1x, cores that lack it slow down by 1.7x, for the identical model and runtime. (2) INT8 outputs are not portable, and the rule is an invariance rather than a gradient: FP32 
    
[^159]: 使用机器学习预测社交媒体互动度

    Predicting Social Media Engagement using Machine Learning

    [https://arxiv.org/abs/2609.16082](https://arxiv.org/abs/2609.16082)

    本研究通过提取家具公司Facebook图片帖子的视觉、文本和时间特征，并运用随机森林、LightGBM等机器学习模型来预测社交媒体互动度。

    

    社交媒体平台因其庞大的用户群体和便捷的访问方式，成为传播信息的热门渠道。企业也将社交媒体视为广告流程中的重要组成部分。通过发布高质量的帖子，企业可以提升其互动指标并增加粉丝数量。尽管已有越来越多的研究关注社交媒体互动，但很少有研究综合考察图片帖子的视觉、文本和时间特征，尽管这些特征共同决定了内容在社交媒体上的表现。为了理解社交媒体互动的关键驱动因素，我们收集了家具公司在Facebook上发布的图片帖子，并运用文本和图像分析方法从中提取视觉、时间和文本特征。我们评估了多种机器学习模型——包括随机森林、轻量梯度提升机（LightGBM）和极限梯度提升等。

    arXiv:2609.16082v1 Announce Type: cross  Abstract: Social media platforms are popular channels for disseminating information, owing to their large user bases and ease of access. Companies also use social media as an important aspect of the advertising process. By creating high-quality posts, companies can strengthen their engagement metrics and increase their follower count. While a growing body of research has examined social media engagement, fewer studies have jointly examined the visual, textual, and temporal features of image posts, even though these features collectively determine the performance of content on social media. To understand the important drivers of social media engagement, we collect image posts of furniture firms on Facebook and extract visual, temporal, and textual features from them using text and image analytics methods. We evaluate several machine learning models - including Random Forest, Light Gradient Boosting Machine (LightGBM), and eXtreme Gradient Boostin
    
[^160]: 面向小型协作小组情感感知的伪标签增强方法

    Pseudo-Label Augmentation for Affect Sensing in Small Collaborative Groups

    [https://arxiv.org/abs/2609.16077](https://arxiv.org/abs/2609.16077)

    该研究提出在小型协作小组情感感知中使用伪标签增强以应对标签稀疏问题，结果表明伪标签增强优于仅用标注数据的基线，但大五人格相似度取值范围过窄，细粒度人格加权效果有限，人格相似性主要仅起同队过滤器作用。

    

    自然群体交互中的生理情感感知往往受限于稀疏标签而非传感器数据：可穿戴设备会产生大量的时间窗口，而自我报告在每次会话中仅采集少数几次。我们使用GroupAffect-4——一个包含四人协作场景下可穿戴生理信号、眼动追踪、大五人格和任务后VAD（效价-唤醒-支配）标签的数据集——研究稀疏监督下用于情感感知的伪标签增强方法。在一个共享的目标构建流程中，我们比较了无增强、高斯过程伪标签、人格感知信任加权以及人格加置信度联合加权等方法。结果表明，在已知团队的设置下，伪标签增强优于仅使用标注数据的基线。然而，大五人格余弦相似度的取值范围狭窄（0.91-0.99），使得细粒度的人格加权效果不佳；人格相似性主要起到区分同队成员的过滤器作用……

    arXiv:2609.16077v1 Announce Type: cross  Abstract: Physiological affect sensing in naturalistic group interaction is often limited by sparse labels rather than sensor data: wearable devices produce many time windows, while self-reports are collected only a few times per session. Using GroupAffect-4, a four-person collaborative dataset with wearable physiology, eye tracking, Big Five personality, and post-task VAD labels, we study pseudo-label augmentation for affect sensing under sparse supervision. We compare no augmentation, Gaussian Process pseudo-labelling, personality-aware trust weighting, and joint personality-plus-confidence weighting within a shared target-construction pipeline. Results show that pseudo-label augmentation improves over the labelled-only baseline in the known-team setting. However, the narrow range of Big Five cosine similarities (0.91-0.99) makes fine-grained personality weighting ineffective; personality similarity functions mainly as a same-team filter rathe
    
[^161]: 面向部分传感器重叠下跨机床CNC迁移的模式自适应动作条件化JEPA

    Schema-Adaptive Action-Conditioned JEPA for Cross-Machine CNC Transfer under Partial Sensor Overlap

    [https://arxiv.org/abs/2609.16071](https://arxiv.org/abs/2609.16071)

    该论文提出一种模式自适应的动作条件化JEPA架构，在源与目标CNC机床仅共享10/17个传感器通道的部分重叠情况下，通过严谨的密封目标测试协议实现零样本跨机床动力学预测迁移，将目标机器预测RMSE从0.813降至0.546。

    

    工业世界模型的跨机器部署需要在动态特性、传感接口、采样机制和控制单元变化下进行迁移。我们研究了一种用于CNC动力学的模式自适应动作条件化联合嵌入预测架构（SAAC-JEPA），其中源机器具有17个标准传感器通道，而目标机器仅共享其中10个。评估采用组不相交的源数据划分、仅源归一化、留出自监督验证、单位审计以及模型锁定后的密封目标测试。在五个随机种子下，JEPA预训练在干净源数据的预测任务中没有带来明显增益：从零开始训练的模型与预训练主体模型的RMSE分别为0.811±0.022和0.813±0.022。在仅使用源数据的20个候选方案搜索中，经过七种子稳定性检查后，选出了模式一致的动作条件化JEPA。在确认性目标测试中，锁定模型达到零样本RMSE=0.546、R²=0.012（摘要在此处被截断）。

    arXiv:2609.16071v1 Announce Type: cross  Abstract: Cross-machine deployment of industrial world models requires transfer across changes in dynamics, sensing interfaces, sampling regimes, and control units. We study a schema-adaptive action-conditioned Joint-Embedding Predictive Architecture (SAAC-JEPA) for CNC dynamics, where the source machine has 17 canonical sensor channels and the target shares only 10. Evaluation uses group-disjoint source splits, source-only normalization, held-out self-supervised validation, unit audits, and a sealed target test after model locking. Across five seeds, JEPA pretraining gives no clean-source forecasting gain: scratch and pretrained-body models obtain \(\mathrm{RMSE}=0.811\pm0.022\) and \(0.813\pm0.022\). A source-only search over 20 candidates selects a schema-consistent action-conditioned JEPA after seven-seed stability checks. On the confirmatory target pass, the locked model reaches zero-shot \(\mathrm{RMSE}=0.546\), \(R^2=0.012\), and \(\mathr
    
[^162]: 超越分布匹配：基于弱语义先验的语义一致性表格扩散模型

    Beyond Distribution Matching: Semantics-Consistent Tabular Diffusion with Weak Semantic Priors

    [https://arxiv.org/abs/2609.16069](https://arxiv.org/abs/2609.16069)

    该论文提出了一种语义一致性表格扩散框架，利用大语言模型从元数据中提取弱语义先验（列内语义和列间符号规则）并将其作为生成条件而非事后过滤，从而解决了现有表格生成器仅优化分布匹配而忽视语义约束的问题。

    

    合成的表格数据虽然能够匹配真实数据的分布，但仍可能违反那些约束有效表格行的语义规则。这揭示了现有表格生成器的一个关键局限性：它们主要优化分布保真度，却没有显式建模表格模式和文本描述中所编码的弱语义先验。在本文中，我们提出了一个语义一致性表格扩散框架，用于在弱指定的语义先验下进行高保真合成数据生成。该框架首先构建两类先验，即列内语义和列间符号规则，通过大语言模型辅助从元数据中提取，并在真实训练集上进行验证。这些先验随后被用作生成条件而非事后过滤器。具体而言，该框架将异构的列值、列标识和语义先验映射到一个统一的语义空间中，并执行按列的前向协同……（摘要在此处截断）

    arXiv:2609.16069v1 Announce Type: cross  Abstract: Synthetic tabular data can match real data distributions while still violating the semantic constraints that govern valid tabular rows. This reveals a key limitation of existing tabular generators: they mainly optimize distributional fidelity, but do not explicitly model weak semantic priors encoded in tabular schema and textual descriptions. In this paper, we propose \ours, a semantics-consistent tabular diffusion framework for high-fidelity synthetic data generation under weakly specified semantic priors. \ours\ first constructs two types of priors, namely intra-column semantics and inter-column symbolic rules, with LLM-assisted extraction from metadata and validation on the real training split. These priors are then used as generation conditions rather than post-hoc filters. Specifically, \ours\ maps heterogeneous column values, column identities, and semantic priors into a unified semantic space, and performs column-wise forward co
    
[^163]: 一种动态聚合策略增强的高效全局优化算法用于求解高维叶轮机械设计问题

    A Dynamic Aggregation Strategy Enhanced Efficient Global Optimization Algorithm for Solving High-Dimensional Turbomachinery Design Problems

    [https://arxiv.org/abs/2609.16067](https://arxiv.org/abs/2609.16067)

    提出了一种动态聚合策略增强的高效全局优化算法DA-EGO，通过将高维设计空间动态分解为低维子空间、基于变量交互分析自适应更新子空间并调整搜索范围，有效求解了高维昂贵的叶轮机械设计优化问题。

    

    为了在有限预算内求解高维（d ≥ 30）昂贵的黑箱优化问题，提出了一种具有动态聚合策略的高效全局优化（EGO）算法，称为DA-EGO。具体而言，DA-EGO将原始高维设计空间分解为一组低维子空间，以进行高效的基于代理模型的优化搜索，并将各子空间的最优解组合成一个精英点用于全局搜索。最重要的是，子空间并不是固定的，而是在每次迭代中根据子空间和全空间中的变量交互分析来更新子空间变量。该算法采用扰动方法和方差分析来检测变量之间的交互作用。为了进一步加速优化进程，还根据对前几次迭代子空间优化结果的分析，自适应地调整子空间的搜索范围。

    arXiv:2609.16067v1 Announce Type: new  Abstract: In order to solve the high-dimensional ($d \geq 30$) expensive black-box problems within budget, an efficient global optimization (EGO) algorithm with a dynamic aggregation strategy is proposed, labeled as DA-EGO. Specifically, the DA-EGO decomposes the original high-dimensional design space into a set of low-dimensional subspaces for efficient surrogate-based optimization search, and the optimal solutions of subspaces are combined as an elite point for the global search. Most importantly, the subspaces are not fixed. Instead, the subspace variables are updated in each iteration, according to the variable interaction analyses in the sub- and full-spaces. The perturbation method and the analysis of variance are used to detect variable interactions. To further accelerate the optimization progress, the searching ranges of subspaces are also adaptively adjusted according to the analyses of subspace optimization results of the previous iterat
    
[^164]: 基于Transformer增强神经算子的叶轮机械叶栅气动性能全景预测方法

    A panoramic aerodynamic performance prediction method for turbomachinery cascades using transformer-enhanced neural operator

    [https://arxiv.org/abs/2609.16066](https://arxiv.org/abs/2609.16066)

    本文提出一种基于Transformer增强神经算子（TNO）的全景性能预测框架，通过先预测Navier-Stokes方程的基本物理量（温度、压力、密度）再推导关键性能参数，实现了叶轮机械叶栅多种气动性能目标的快速灵活评估，功能类似于CFD模拟器。

    

    为了在叶轮机械设计中实现灵活、快速的气动性能评估，本文提出了一种全景性能预测框架。与以往大多数直接预测感兴趣目标函数的预测模型不同，我们的方法首先预测Navier-Stokes方程的基本参数，如温度、压力和密度。利用这些基本物理量，进而预测涡轮级子午面的关键性能参数。通过采用这种方法，我们所提出的全景性能预测框架的功能类似于CFD模拟器，能够预测设计人员感兴趣的各种目标。为了提高预测精度，该框架中引入了一种Transformer增强神经算子（TNO）。以Rotor 37叶片为参考，训练所提出的TNO来预测跨音速压气机叶片的性能。

    arXiv:2609.16066v1 Announce Type: new  Abstract: To enable flexible and rapid aerodynamic performance evaluation in turbomachinery design, this paper proposes a panoramic performance prediction framework. Unlike most previous prediction models that directly predict the objective functions of interest, our approach first predicts the basic parameters of the Navier-Stokes equations, such as temperature, pressure, and density. Utilizing these basic physical quantities, it subsequently predicts key performance parameters of the turbine stage meridian plane. By adopting this methodology, our proposed panoramic performance prediction framework functions similarly to a CFD simulator, capable of predicting various objective of interest to the designers. To enhance prediction accuracy, a transformer-enhanced neural operator (TNO) is introduced within this framework. Using the Rotor 37 blades as a reference, the proposed TNO is trained to predict the performance of a transonic compressor blade i
    
[^165]: 你无需训练：面向可执行人体活动识别的智能体启发式学习工作室

    You Don't Need To Train: Agentic Heuristic Learning Studio for Executable Human Activity Recognition

    [https://arxiv.org/abs/2609.16065](https://arxiv.org/abs/2609.16065)

    该论文提出智能体启发式学习（AHL）工作室，模仿人类认知学习方式（记忆示例、形成规则、修复错误）而非梯度训练，生成可执行、可检查、可编辑且无需LLM的人体活动识别策略，并支持边缘部署。

    

    人体活动识别（HAR）通常被构建为基于梯度的神经网络训练。智能体启发式学习（AHL）工作室探索了一种受人类认知学习启发的互补视角：人们通过记忆示例、形成规则和修复错误来学习活动，而不是通过反向传播。该工具为HAR实现了AHL：一个学习时智能体对传感器协议进行推理，提出可执行的启发式策略，记录修复轨迹，并导出无需LLM的策略用于边缘部署。我们聚焦于HAR基准测试家族，提供了从数据集观察到面向边缘导出的端到端工作流。在迄今评估的十一个HAR数据集上，AHL策略达到了强大的可执行策略性能，同时保持可检查、可编辑和可重放。

    arXiv:2609.16065v1 Announce Type: cross  Abstract: Human activity recognition (HAR) is usually framed as gradient-based training of neural networks. Agentic Heuristic Learning (AHL) Studio explores a complementary view inspired by human cognitive learning: people learn activities by remembering examples, forming rules, and repairing mistakes, not by backpropagating. This proposed tool implements AHL for HAR: a learning-time agent reasons over sensor protocols, proposes executable heuristic policies, records repair traces, and exports an LLM-free policy for edge deployment. We focus on the HAR benchmark family and provide an end-to-end workflow from dataset observation to edge-oriented export. On eleven HAR datasets evaluated so far, AHL policies reach strong executable-policy performance while remaining inspectable, editable, and replayable \footnote{https://github.com/zhaxidele/ahl-ts-studio}.
    
[^166]: 有限域全不同约束系统的带符号p进剩余编码——以数独为案例研究

    Signed p-adic Residual Encodings of Finite-Domain All-Different Systems with a Sudoku Case Study

    [https://arxiv.org/abs/2609.16063](https://arxiv.org/abs/2609.16063)

    该论文提出用带符号加权的仿射p进剩余目标函数作为有限域约束的原生编码，通过逐坐标支配定理证明所有全局极小值都位于有限域内，且损失恰好对应全不同冲突数或负的已满足CNF子句数，并以无需独热提升的标准数独（81系数）案例验证了该方法。

    

    我们研究将带符号的加权仿射p进剩余目标函数作为有限域约束的原生编码。对于能分离有限字母表的素数，具有足够权重的正一元行可将每个系数固定在其允许的取值集合上，而负行则会奖励不相等的端点或已满足的子句。一个逐坐标支配定理保证了每个全局极小值点都落在有限域内；在该域上，损失函数（相差一个加法常数）即为全不同冲突计数，或已满足CNF子句数的负值。标准数独提供了一个81系数的案例研究，且无需独热提升。客户端实现展示了所生成的数据框、算术运算、诊断工具和搜索过程。

    arXiv:2609.16063v1 Announce Type: new  Abstract: We study signed, weighted affine $p$-adic residual objectives as native encodings of finite-domain constraints. For primes that separate the finite alphabet, sufficiently weighted positive unary rows pin each coefficient to its allowed set, while negative rows reward unequal endpoints or clause satisfaction. A coordinatewise domination theorem places every global minimiser in the finite domain; there the loss is, up to an additive constant, the all-different conflict count or the negative number of satisfied CNF clauses. Standard Sudoku provides an $81$-coefficient case study without a one-hot lift. A client-side implementation exposes the generated dataframes, arithmetic, diagnostics, and searches.
    
[^167]: 数字说服：理解网络意见领袖对舆论的影响

    Digital Persuasion: Understanding the Impact of Online Influencers on Public Opinion

    [https://arxiv.org/abs/2609.16062](https://arxiv.org/abs/2609.16062)

    本文提出一个基于Friedkin-Johnsen模型的框架来识别网络意见领袖，并通过初始观点操纵实验在总统选举推文数据集上验证了意见领袖能够显著改变社区整体舆论。

    

    研究社交网络中的观点动态及其传播对于应对包括政治极化、公共卫生和营销策略在内的广泛挑战至关重要。在本工作中，我们通过提出一个基于Friedkin-Johnsen（FJ）模型的框架来研究观点动态问题，该框架用于识别有影响力的用户并研究其对社区观点动态的影响。FJ模型假设每个个体持有两种观点：初始观点和表达观点。通过一系列初始观点操纵实验，所提出的框架评估了有影响力用户与随机用户对整个社区观点的影响。该框架使用代表美国总统选举的推文数据集进行了验证。结果表明，具有最高影响力分数的意见领袖能够显著改变整个社区的整体观点。此外，结果还表明意见领袖的影响不仅限于……（原文摘要至此截断）

    arXiv:2609.16062v1 Announce Type: cross  Abstract: The studying of opinion dynamics and its propagation within social networks is crucial for addressing a wide range of challenges, including political polarization, public health, and marketing strategies. In this work, we study the problem of opinion dynamics by proposing a framework based on Friedkin-Johnsen (FJ) to identifies influential users and study their impact on dynamics opinions of community. The FJ model assume each individual have two opinions: initial and expressed. Through a series of initial opinion manipulation experiments, the proposed framework assesses the impact of influential versus random users on the overall community opinion. The proposed framework is validated using a tweet dataset representing the U.S. presidential election. The results shows that influencers with highest influencing score, significantly shift the overall community opinion. Moreover, the results shows that the impact of influencers not limited
    
[^168]: POSPAN：用于语言模型预训练的位置约束式片段掩码方法

    POSPAN: Position-Constrained Span Masking for Language Model Pre-training

    [https://arxiv.org/abs/2609.16061](https://arxiv.org/abs/2609.16061)

    提出POSPAN通用框架，通过结合片段长度分布与位置约束分布来支持多样化的位置约束式片段掩码策略，统一了所有现有的片段级掩码方法，并在多个NLU基准上显著提升了预训练语言模型的性能。

    

    片段级掩码语言建模（MLM）已被证明比原始的单token掩码语言建模对预训练语言模型更有优势，因为实体/短语及其之间的依赖关系对语言理解至关重要。先前的工作仅考虑了服从某些离散分布的片段长度，而忽略了片段之间的依赖关系，即假设被掩码片段的位置是均匀分布的。在本文中，我们提出了POSPAN，这是一个通用框架，通过结合片段长度分布和位置约束分布，实现了多样化的位置约束式片段掩码策略，并统一了所有现有的片段级掩码方法。为了验证POSPAN在预训练中的有效性，我们在多个NLU基准测试的数据集上对其进行了评估。实验结果表明，位置约束能够广泛地增强片段级掩码的效果，而我们最佳的POSPAN设置持续超越（现有基线方法）。

    arXiv:2609.16061v1 Announce Type: cross  Abstract: Span-level masked language modeling (MLM) has shown to be advantageous to pre-trained language models over the original single-token MLM, as entities/phrases and their dependencies are critical to language understanding. Previous works only consider span length with some discrete distributions, while the dependencies among spans are ignored, i.e., assuming that the positions of masked spans are uniformly distributed. In this paper, we present POSPAN, a general framework to allow diverse position-constrained span masking strategies via the combination of span length distribution and position constraint distribution, which unifies all existing span-level masking methods. To verify the effectiveness of POSPAN in pre-training, we evaluate it on the datasets from several NLU benchmarks. Experimental results indicate that the position constraint is capable of enhancing span-level masking broadly, and our best POSPAN setting consistently outp
    
[^169]: HintMiner：基于自监督学习与语言模型从问答网络帖子中自动挖掘问题提示

    HintMiner: Automatic Question Hints Mining From Q&A Web Posts with Language Model via Self-Supervised Learning

    [https://arxiv.org/abs/2609.16060](https://arxiv.org/abs/2609.16060)

    本文提出了HintMiner工具，通过自监督学习训练基于Transformer和复制机制的编码器-解码器模型MiningNet，从海量网络问答帖子中自动挖掘并生成用户问题的提示，帮助用户更快找到答案。

    

    用户经常需要在线提问并寻找答案。诸如Stack Overflow之类的问答（QA）论坛并不总能及时、恰当地回应用户的问题。在本文中，我们提出了HintMiner，一种新颖的自动问题提示挖掘工具，用于帮助用户找到答案。HintMiner利用机器理解和序列生成技术，自动为用户的问题生成提示。它首先检索大量网络问答帖子，然后使用MiningNet（一种通过语言模型构建的模型）从帖子中提取提示。利用海量的在线问答帖子，我们设计了一个自监督学习目标来训练MiningNet，该模型是一个基于Transformer和复制机制的神经编码器-解码器模型。我们在60,000个Stack Overflow问题上对HintMiner进行了评估。实验结果表明，所提出的方法是有效的。例如，HintMiner达到了平均BLEU分数为3（原文此处似乎被截断）

    arXiv:2609.16060v1 Announce Type: cross  Abstract: Users often need ask questions and seek answers online. The Question - Answering (QA) forums such as Stack Overflow cannot always respond to the questions timely and properly. In this paper, we propose HintMiner, a novel automatic question hints mining tool for users to help them find answers. HintMiner leverages the machine comprehension and sequence generation techniques to automatically generate hints for users' questions. It firstly retrieve many web Q\&A posts and then extract some hints from the posts using MiningNet that is built via a language model. Using the huge amount of online Q\&A posts, we design a self-supervised objective to train the MiningNet that is a neural encoder-decoder model based on the transformer and copying mechanisms. We have evaluated HintMiner on 60,000 Stack Overflow questions. The experiment results show that the proposed approach is effective. For example, HintMiner achieves an average BLEU score of 3
    
[^170]: 迈向可扩展的RLVR：多模态指令跟随数据合成与蒸馏

    Towards Scalable RLVR: Multimodal Instruction Following Data Synthesis and Distillation

    [https://arxiv.org/abs/2609.16059](https://arxiv.org/abs/2609.16059)

    提出MIFS系统化数据合成流程，通过生成式约束协议、可学习性感知蒸馏机制和基于代码的验证器，生成可直接用于强化学习的多模态指令跟随数据，突破RLVR在多模态领域可扩展性的数据瓶颈。

    

    多模态指令跟随（MMIF）对于构建通用智能体至关重要。然而，当前的训练范式严重依赖监督微调（SFT），这往往导致表面的模式匹配并损害泛化能力。虽然基于可验证奖励的强化学习（RLVR）提供了一种有前景的替代方案，但其在MMIF中的可扩展性受到高质量、可直接用于强化学习的多模态数据稀缺的严重制约。为了弥合这一差距，我们提出了MIFS（多模态指令跟随合成），这是一个旨在生成可直接用于强化学习的多模态数据的系统化流程。具体而言，MIFS引入了一个生成式约束协议来合成多样化的原始样本，随后通过一个可学习性感知的蒸馏机制，基于强化学习训练动态对数据进行过滤，以确保稳定的策略优化。此外，基于代码的验证器提供了高精度的奖励信号。

    arXiv:2609.16059v1 Announce Type: new  Abstract: Multimodal instruction following (MMIF) is crucial for building generalist agents. However, current training paradigms rely heavily on Supervised Fine-Tuning (SFT), which often leads to surface-level pattern matching and degrades general capabilities. While Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising alternative, its scalability in MMIF is severely bottlenecked by the scarcity of high-quality, RL-ready multimodal data. To bridge this gap, we present MIFS (\textbf{M}ultimodal \textbf{I}nstruction \textbf{F}ollowing \textbf{S}ynthesis), a systematic pipeline designed to generate RL-ready multimodal data. Specifically, MIFS introduces a generative constraint protocol to synthesize diverse raw samples, followed by a learnability-aware distillation mechanism that filters data based on RL training dynamics to ensure stable policy optimization. Furthermore, a code-based verifier provides high-precision reward signal
    
[^171]: 基于物理约束决策条件自回归Transformer的信号交叉口驾驶员行为估计

    Driver Behavior Estimation at Signalized Intersections Using a Physics-Constrained Decision-Conditioned Autoregressive Transformer

    [https://arxiv.org/abs/2609.16058](https://arxiv.org/abs/2609.16058)

    该论文基于449次真实驾驶轨迹数据，提出了一个物理约束且以决策为条件的两阶段自回归Transformer框架，用于预测信号灯切换时驾驶员的停走决策与纵向轨迹行为，并发现所需减速度是决策的关键预测因子，由此推导出五个经验性舒适减速度区间。

    

    信号交叉口的闯红灯和急刹车行为是交通事故的主要原因。本文分析并预测了驾驶员在交通信号灯切换期间的人类驾驶决策和纵向轨迹行为。我们收集了一个多样化的真实世界数据集，包含在不同速度和距离条件下的449次进口道行驶记录。车辆运动通过具有厘米级精度的RTK校正GNSS进行记录，同时监测驾驶员心率及多级舒适度评分。空间与时间校准确保了车辆状态与信号时序之间的精确对齐。统计分析表明，所需减速度是停走决策的最主要单一预测因子，而对峰值减速度的异方差高斯建模则从人类停车行为中推导出五个经验性舒适度区间。基于这一发现，我们提出了一个两阶段建模框架。阶段1预测……（摘要原文在此处截断）

    arXiv:2609.16058v1 Announce Type: cross  Abstract: Red-light violations and harsh braking at signalized intersections are major contributors to traffic accidents. This paper analyzes and predicts human driver decision-making and longitudinal trajectory behavior during traffic light signal transitions. We collected a diverse real-world dataset comprising 449 approach runs under varying speed and distance conditions. Vehicle motion was recorded using RTK-corrected GNSS with centimeter-level accuracy, and driver heart rate and multi-level comfort ratings were monitored. Spatial and temporal calibration ensured precise alignment between vehicle state and signal timing. Statistical analysis identifies required deceleration as the dominant single predictor of the stop-go decision, and heteroscedastic Gaussian modeling of peak deceleration reveals five empirical comfort ranges derived from human stopping behavior. Based on this insight, we propose a two-stage modeling framework. Stage 1 predi
    
[^172]: OmniHarness：通过符号策略学习实现可泛化的视觉生成

    OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning

    [https://arxiv.org/abs/2609.16057](https://arxiv.org/abs/2609.16057)

    提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。

    

    统一的多模态大语言模型和多智能体系统推动了视觉生成领域的发展。然而，仍存在三个局限性：（1）现有方法往往蒸馏任务特定的经验，泛化能力有限；（2）反思通常推迟到任务完成后才进行；（3）知识通常仅在响应下游任务需求时才被获取。为了解决这些局限性，我们提出了OmniHarness，一个通过符号策略学习实现可泛化视觉生成的框架。OmniHarness将经过验证的执行过程抽象为面向视觉生成任务族的符号策略，在去除实例特定输入的同时，捕获共享的程序流程和适用条件。该框架可以为新任务实例化、调整和组合这些策略。执行过程中的中间验证指导策略细化与失败恢复。通过自主探究，OmniHarness能够自主生成并执行练习任务（原文摘要在此处截断）。

    arXiv:2609.16057v1 Announce Type: cross  Abstract: Unified multimodal large language models (MLLMs) and multi-agent systems have advanced visual generation. However, three limitations remain. (1) Existing methods often distill task-specific experience with limited generalizability. (2) Reflection is often deferred until task completion. (3) Knowledge is often acquired only in response to downstream task demands. To address these limitations, we introduce OmniHarness, a framework for generalizable visual generation via symbolic policy learning. OmniHarness abstracts verified executions into symbolic policies for visual generation task families, capturing shared procedures and applicability conditions while removing instance-specific inputs. The harness instantiates, adapts, and composes these policies for new tasks. Intermediate verification guides refinement and failure recovery during execution. Through self-directed inquiry, OmniHarness autonomously generates and executes practice ta
    
[^173]: 神经符号强化学习中的动作前置条件管理：面向具身智能体的三种放置策略

    Managing Action Preconditions in Neuro-Symbolic RL: Three Placement Strategies for Embodied Agents

    [https://arxiv.org/abs/2609.16056](https://arxiv.org/abs/2609.16056)

    该论文将结构动作的前置条件知识形式化为贝叶斯网络，并提出将其注入强化学习循环的三种放置策略，以避免幻觉式前置条件、提升具身智能体在变化环境中的安全性与可靠性。

    

    人类会将熟悉情境下的行为知识带入每一个新任务中，而不是从零开始重新学习。强化学习（RL）智能体没有理由不这样做：已知的行为模式无需重新学习，只需加以应用。神经符号RL通过在学习到的策略旁注入符号知识，从而连接起先验知识与强化学习。知识被集成的位置至关重要：错误的选择可能产生诸如“幻觉式前置条件”的问题，这在变化环境中行动的智能体身上会表现为安全性和可靠性缺陷。我们将这种行为知识形式化为智能体“结构动作”之上的前置条件贝叶斯网络（BN）——结构动作是指其合法性取决于前置条件的动作，例如拾取钥匙、抓取方块、切换门的状态或放置物体。该贝叶斯网络限制这些动作何时可以触发，并且我们将其在三个……（原文摘要在此处截断）

    arXiv:2609.16056v1 Announce Type: cross  Abstract: Humans carry behaviour knowledge of how to act in familiar situations into every new task rather than relearning it from scratch. There is no reason a Reinforcement Learning (RL) agent shouldn't do the same: known behaviour patterns need not be learned, only applied. Neuro-symbolic RL bridges prior knowledge and RL by injecting symbolic knowledge alongside a learned policy. The point at which this knowledge is integrated is critical: a poor choice can produce, for instance, hallucinated preconditions, which surface as safety and reliability problems in agents acting in changing environments. We formalise this behavioural knowledge as a precondition Bayesian network (BN) over the agent's \emph{structural actions} - the actions whose legality depends on preconditions, such as picking up a key, grasping a block, toggling a door, or dropping an object. The BN restricts when these actions may fire, and we inject it into the RL loop at three
    
[^174]: 用于在线多目标跟踪的因果神经集合滤波

    Causal neural set filtering for online multi-target tracking

    [https://arxiv.org/abs/2609.16054](https://arxiv.org/abs/2609.16054)

    本文提出因果神经集合滤波器（CNSF），通过仅编码当前测量并借助结构化递归航迹状态携带历史信息，结合排他性Sinkhorn关联、关联条件卡尔曼形态更新和递归伯努利生命周期建模，在消除冗余计算的同时实现了优于MT3/Track-MT3的多目标跟踪精度。

    

    基于Transformer的多目标跟踪（MTT）方法联合学习数据关联与状态估计，但MT3/Track-MT3风格的跟踪器需要反复重新编码测量窗口，导致冗余计算。我们提出了因果神经集合滤波（CNSF），这是一种神经集合滤波器，它仅对当前时刻的测量进行编码，同时将过去的信息证据保存在结构化的递归航迹状态中。CNSF结合了排他性Sinkhorn关联、具有矩匹配的关联条件卡尔曼形态更新，以及带有测量驱动新生过程的递归伯努利生命周期建模。这些机制施加了软性的一对一约束，传播由关联引起的状态不确定性，并支持在漏检和生灭转换情况下的存在性估计。在一个保留的三种场景模拟测试集上，CNSF相较于T（原文摘要在此处截断）降低了平均GOSPA和T-GOSPA……

    arXiv:2609.16054v1 Announce Type: cross  Abstract: Transformer-based multi-target tracking (MTT) jointly learns data association and state estimation, but MT3/Track-MT3-style trackers repeatedly re-encode measurement windows, incurring redundant computation. We propose Causal Neural Set Filtering (CNSF)\footnote{\href{https://github.com/daihuangyu/CNSF}{Code: https://github.com/daihuangyu/CNSF}}, a neural set filter that encodes only current measurements while carrying past evidence in a structured recursive track state. CNSF combines exclusive Sinkhorn association, association-conditioned Kalman-shaped updates with moment matching, and recurrent Bernoulli lifecycle modeling with measurement-driven birth. These mechanisms impose soft one-to-one constraints, propagate association-induced state uncertainty, and support existence estimation under missed detections and birth--death transitions. On a held-out three-regime simulated test set, CNSF reduces mean GOSPA and T-GOSPA relative to T
    
[^175]: 基于深度字典网络的超低剂量CT去噪基础模型

    A deep dictionary network-based foundation model for ultra-low-dose CT denoising

    [https://arxiv.org/abs/2609.16031](https://arxiv.org/abs/2609.16031)

    该论文提出了一种基于深度字典网络（DDN）的架构可解释基础模型，将卷积稀疏编码层与迭代软阈值级联，实现了跨多器官的统一超低剂量CT图像去噪。

    

    超低剂量计算机断层扫描（ULDCT）能够减少辐射暴露，但其严重的噪声会降低诊断图像质量。现有的基于深度学习的去噪方法通常以特定器官的方式进行训练，导致在异构多器官成像场景中泛化能力有限。基础模型为统一的多器官去噪提供了一种有前景的一体化范式，然而其架构存在可解释性差的问题，且依赖启发式的训练策略。为解决这些局限，我们提出了一种基于深度字典网络（DDN）的架构可解释基础模型，用于统一的多器官超低剂量CT去噪。受多层稀疏表示理论的启发，DDN将卷积稀疏编码层与迭代软阈值进行级联，从而提供固有的架构可解释性。此外，动态字典模块和阈值生成模块……

    arXiv:2609.16031v1 Announce Type: cross  Abstract: Ultra-low-dose computed tomography (ULDCT) reduces radiation exposure but suffers from severe noise that degrades diagnostic image quality. Existing deep learning-based denoising methods are typically trained in an organ-specific fashion, resulting in limited generalization across heterogeneous multi?organ imaging scenarios. Foundation models present a promising all-in-one paradigm for unified multi-organ denoising. However, their architectures suffer from poor interpretability and rely on heuristic training strategies. To address these limitations, we propose an architecture?interpretable foundation model based on the deep dictionary network (DDN) for unified multi-organ ULDCT denoising. Inspired by multilayer sparse representation theory, DDN cascades convolutional sparse coding layers with iterative soft-thresholding, providing inherent architectural interpretability. Furthermore, a dynamic dictionary module and a threshold generati
    
[^176]: 分子表示塑造基于流的聚合物生成中目标保真度与探索之间的平衡

    Molecular representation shapes the balance between target fidelity and exploration in flow based polymer generation

    [https://arxiv.org/abs/2609.16028](https://arxiv.org/abs/2609.16028)

    该研究提出基于潜空间连续时间流匹配的PolyLatentFlow生成框架与融合聚合物序列及三维结构信息的LlamaUni多模态分子表示，实现了有效且新颖聚合物候选物产出的显著提升，并揭示了分子表示对目标保真度与探索能力之间平衡的关键影响。

    

    设计具有目标性能的聚合物需要在有限标注数据的条件下探索广阔的化学空间。在此，我们提出PolyLatentFlow，一个基于潜空间中连续时间流匹配的框架，可用于无条件及有条件的聚合物生成，同时还提出LlamaUni，一种融合聚合物序列与三维结构信息的多模态表示。在无条件生成任务中，PolyLatentFlow结合LlamaUni在所评估的无条件生成器中产生了最大数量的相对于PolyInfo新颖的有效候选聚合物，同时保持了高度多样性。在玻璃化转变温度（Tg）条件生成任务中，生成聚合物的性质分布在200°C的目标范围内呈现出系统性偏移。在多性质任务中，不同分子表示展现出相近的替代目标保真度，但在有效性、训练集重放程度以及与已标注聚合物的结构接近度方面存在显著差异。PolyLatentFlow结合LlamaUni始终能够兼顾高目标保真度与充分的化学空间探索。

    arXiv:2609.16028v1 Announce Type: cross  Abstract: Designing polymers with targeted properties requires navigating vast chemical spaces from limited labeled data. Here we introduce PolyLatentFlow, a framework based on continuous-time flow matching in latent space for unconditional and conditional polymer generation, together with LlamaUni, a multimodal representation combining polymer sequence and 3D structural information. In unconditional generation, PolyLatentFlow with LlamaUni produced the largest yield of valid candidates novel relative to PolyInfo among the evaluated unconditional generators while maintaining high diversity. For $T_g$ conditioning, generated property distributions shifted systematically across a 200 {\deg}C target range. In multi-property tasks, molecular representations showed similar surrogate target fidelity but differed markedly in validity, training-set replay, and structural proximity to labeled polymers. PolyLatentFlow with LlamaUni consistently combined h
    
[^177]: 基于自适应样本的高斯编码重建实现三维场数据缩减

    3D Field Data Reduction with Adaptive Sample-Based Gaussian-Encoded Reconstruction

    [https://arxiv.org/abs/2609.16024](https://arxiv.org/abs/2609.16024)

    该论文提出一种统一的基于样本的自适应高斯编码方法，能在固定预算下同时表示结构化网格、非结构化网格和粒子场数据，相比现有方法以约44倍更少的基元实现高达4.8 dB的PSNR提升。

    

    在科学仿真中，规则网格、非结构化网格和基于粒子的格式分别因其计算效率、几何/自适应灵活性以及跟踪运动/变形的能力而被选用于表示场数据。这些场数据格式通常需要通过各自独立的、针对特定数据的处理流程来处理。我们提出了一种统一的基于样本的高斯编码方法，能够在单一的固定预算公式下表示这些数据形式。该方法直接从输入样本初始化并优化高斯基元，同时保持预设的基元数量和编码大小，以实现期望的数据缩减水平。在结构化、非结构化和粒子数据上，与先前的公式相比，基于样本的公式以明显更少的基元提高了重建精度，实现了高达4.8 dB的PSNR提升，同时基元数量减少约44倍。对于时变…

    arXiv:2609.16024v1 Announce Type: cross  Abstract: In scientific simulation, regular grids, unstructured meshes, and particle-based formats are chosen to represent field data for computational efficiency, geometry/adaptive flexibility, and following motion/deformation, respectively. Each of these field data formats is often handled through separate data-specific processing pipelines. We present a unified sample-based Gaussian encoding method that represents these data forms under a single fixed-budget formulation. The method initializes and refines Gaussian primitives directly from the input samples while preserving a prescribed primitive count and encoded size to achieve a desired level of data reduction. Across structured, unstructured, and particle data, the sample-based formulation improves reconstruction accuracy with measurably fewer primitives in comparison to prior formulations, achieving up to 4.8 dB higher PSNR with an approximate 44x reduction in primitive count. For time-va
    
[^178]: 我们的评分方式是否恰当？理解医学基准测试中的失败模式

    Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks

    [https://arxiv.org/abs/2609.16023](https://arxiv.org/abs/2609.16023)

    该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。

    

    医学评估正在从基于静态选项的提问转向具有开放式输出模式的真实临床场景。然而，简单地大规模评分成本高昂，基于评分标准的评估已成为主流的可扩展替代方案。我们探讨了当评分标准本身不够严密时会发生什么，以及这些缺陷是否能够被检测和纠正。我们将RIFT——一个全局评分标准失败分类法——应用于两个临床基准测试（HealthBench Professional和LiveMedBench），发现失败模式具有重要意义：在HealthBench Professional上，LLM评判者将29.6%的评估标准标记为非原子性，将65.4%标记为错位或僵化。随后，我们证明这些缺陷具有实质性影响，而不仅仅是表面问题。例如，将“以下至少一项/全部满足”形式的捆绑标准重写为等权重的子标准并对相同响应重新评分，会使受影响对话的分数发生高达15.9个百分点的变化。

    arXiv:2609.16023v1 Announce Type: new  Abstract: Medical evaluation is shifting from static option-based questioning to realistic clinical scenarios with open-ended output modes. Grading these at scale naively, however, is expensive, and rubric-based evaluation has become the dominant scalable alternative. We ask what happens when the rubrics themselves are not airtight, and whether such flaws can be detected and corrected. We apply RIFT, a global rubric failure taxonomy, to two clinical benchmarks (HealthBench Professional and LiveMedBench), and find failure modes are meaningful: on HealthBench Professional an LLM judge flags 29.6% of criteria as non-atomic and 65.4% as misaligned/rigid. Then, we show that these flaws are meaningful and not simply cosmetic. As an example, rewriting bundled criteria of the form "at least one of / all of the following" as equally weighted children and regrading identical responses shifts scores by up to 15.9 percentage points on affected conversations, 
    
[^179]: ViCo：面向视觉的编码与自反思实现图表复刻

    ViCo: Visual-oriented Coding with Self-Reflection for Chart Replication

    [https://arxiv.org/abs/2609.16014](https://arxiv.org/abs/2609.16014)

    ViCo是一个面向视觉编码的训练框架，通过自监督预热阶段结合基于一致性剪枝的蒙特卡洛树搜索来合成高质量反思轨迹，并利用迭代反思使生成的图表图像逐步与参考对齐，从而生成符合人类论文视觉标准的高质量学术图表。

    

    本文探讨了生成符合人类撰写论文视觉标准的高质量学术图表这一挑战。虽然现有AI智能体能够生成结构良好的文本和代码，但其生成的可视化结果往往缺乏人类设计在风格和语义上的保真度。采用自反思机制的先进编码智能体表现出较差的视觉推理能力和有限的反思遵循能力，导致稀疏的奖励信号严重削弱了其强化学习（RL）效果。我们提出了ViCo，这是一个面向视觉编码的训练框架，通过迭代反思使生成的图表图像逐步与参考图像对齐。我们首先引入了一个自监督预热阶段，该阶段通过基于一致性的剪枝来增强蒙特卡洛树搜索，以合成高质量的反思轨迹，确保每个编码步骤严格遵循先前反思的结果。一个多-（摘要在此处截断）

    arXiv:2609.16014v1 Announce Type: new  Abstract: This paper addresses the challenge of generating high-quality academic charts that match the visual standards of human-authored papers. While existing AI agents can produce well-structured text and code, their generated visualizations often lack the stylistic and semantic fidelity of human designs. Advanced coding agents that employ self-reflection mechanisms exhibit poor visual reasoning and limited reflection following, resulting in sparse reward signals that severely undermine their reinforcement learning (RL). We propose ViCo, a training framework for visual-oriented coding that employs iterative reflections to align generated chart images progressively with the reference. We first introduce a self-supervised warm-up stage, which augments Monte Carlo Tree Search with consistency-based pruning to synthesize high-quality reflection trajectories, ensuring that each coding step strictly follows the outcomes of prior reflections. A multi-
    
[^180]: EMODY Flow：情感感知的音频驱动全身动作生成

    EMODY Flow: Emotion-Aware Audio-Driven Full-Body Motion Generation

    [https://arxiv.org/abs/2609.16011](https://arxiv.org/abs/2609.16011)

    提出轻量级流匹配框架EMODY Flow，通过复用冻结Qwen-3 Omni模型的内部Mimi音频编解码器来条件化并行的身体姿态和面部表情DiT生成器，解决了情感条件化失效问题，实现与语音和情感状态同步的全身动作生成。

    

    具身对话智能体需要与语音和情感状态相一致的同步全身动作（肢体手势和面部表情）。全模态大语言模型在多模态理解方面表现出色，但仅能产生语言输出，在具身响应生成方面留下了关键空白。我们识别并解决了一个情感条件化失效问题：与其他未能充分利用弱条件信号的条件生成器类似，流匹配模型在同时接收丰富的音频嵌入和离散情感标签时，会抑制情感信息，导致无论指定何种情感都会生成几乎相同的动作。我们提出了EMODY Flow，这是一个轻量级（约3500万参数）的流匹配框架，它附加到冻结的Qwen-3 Omni模型上，并复用其内部的Mimi音频编解码器来条件化两个并行的DiT生成器——一个用于生成SMPL-X身体姿态，另一个用于生成FLAME面部表情。训练时的辅助情感分类器……（原文摘要在此处截断）

    arXiv:2609.16011v1 Announce Type: cross  Abstract: Embodied conversational agents require synchronized full-body motion (body gestures and facial expressions) that aligns with speech and emotional state. Omni-modal large language models excel at multimodal understanding but produce only linguistic outputs, leaving a critical gap in embodied response generation. We identify and address a failure of emotion conditioning: like other conditional generators that under-use weak conditioning signals, a flow-matching model given both a rich audio embedding and a discrete emotion label suppresses the emotion, generating near-identical motion regardless of the specified emotion. We present EMODY Flow, a lightweight (around 35M parameters) flow-matching framework that attaches to a frozen Qwen-3 Omni model and reuses its internal Mimi audio-codecs to condition two parallel DiT generators - one for SMPL-X body pose, one for FLAME facial expressions. A training-time auxiliary emotion classifier res
    
[^181]: 利用多维洛伦兹Zonoid（洛伦兹超体积体）度量人工智能危害

    Measuring AI harms with multidimensional Lorenz Zonoids

    [https://arxiv.org/abs/2609.16004](https://arxiv.org/abs/2609.16004)

    本文提出将洛伦兹Zonoid和基尼指数扩展到多维情形，用以对真实人工智能危害数据进行基于严重程度的风险评估，从而提供一种超越合规导向、能够识别干预优先级的有效AI风险管理方法。

    

    随着人工智能系统日益深刻地影响高风险的社会领域，其治理受到缺乏能够作用于真实危害的风险管理方法的制约——这类方法应考虑危害的严重程度，而不仅仅是其发生的可能性。因此，当前的人工智能风险管理模型仍然是以合规为导向、以提供者为中心的，对于危害的危险程度以及干预优先级应如何确定，只能提供有限的洞察。而危害数据通常是有序且多维的这一特性进一步加剧了该问题。为解决这一问题并提供有效的风险评估方法，本文提出利用洛伦兹Zonoid和基尼指数对危害数据进行建模。为此，我们提出将二者扩展到多维情形，并展示了如何在麻省理工学院提供的真实人工智能事件数据库中对其进行实际计算。实证结果表明，环境……（摘要在此处截断）

    arXiv:2609.16004v1 Announce Type: cross  Abstract: While AI systems increasingly shape high-stakes societal domains, their governance is limited by the lack of risk management methods that operate on real harms, taking their severity, and not only their likelihood, into account. As a consequence, AI risk management models remain compliance-driven and provider-centric, offering limited insight into how harms are dangerous, and on what should be the priority of intervention. The problem is amplified by the nature of harm data which are typically ordinal and multidimensional. To solve the problem, and offer an effective risk assessment methodology, in this paper we propose to model harm data by means of Lorenz Zonoids and Gini indices. To this aim we propose to extend them in a multidimensional setting, and show how to practically calculate them for a real AI incident data repository, provided by the Massachusetts Institute of Technology. The empirical findings indicate that environmental
    
[^182]: 基于大语言模型的事故叙述引导对策推荐：一种面向交叉口安全的检索增强生成框架

    Crash Narrative-Guided Countermeasure Recommendation Using Large Language Models: A Retrieval-Augmented Generation Framework for Intersection Safety

    [https://arxiv.org/abs/2609.15997](https://arxiv.org/abs/2609.15997)

    该研究提出一种基于大语言模型的事故叙述引导检索增强生成（RAG）框架，从非结构化事故叙述中提取关键事故机理属性，并将其与循证安全对策数据库关联，从而自动生成针对交叉口的具体安全改善措施推荐。

    

    提升交叉口安全性需要识别事故机理并推荐适当的改善对策。然而，这一过程传统上依赖专家判断，导致其劳动密集、难以规模化，且依赖于有经验的交通安全工程师的可用性。尽管事故叙述中包含对事故机理的丰富描述，但这些非结构化信息在安全分析中在很大程度上仍未得到充分利用。本研究提出了一种事故叙述引导的检索增强生成（RAG）框架，将从事故叙述中提取的事故机理转化为针对具体地点的对策推荐。该框架从事故叙述中提取关键机理属性，包括交通控制方式、信号显示、驾驶员责任、车辆运动和行驶方向，并将其与FHWA经验证安全对策及CMF Clearinghouse中基于证据的处理措施相关联。该框架集成了嵌入……

    arXiv:2609.15997v1 Announce Type: new  Abstract: Improving safety at intersections requires identifying crash mechanisms and recommending appropriate countermeasures. However, this process traditionally relies on expert judgment, making it labor-intensive, difficult to scale, and dependent on the availability of experienced traffic safety engineers. Although crash narratives contain rich description of crash mechanisms, this unstructured information remains largely underutilized in safety analyses. This study presents a crash narrative-guided retrieval-augmented generation (RAG) framework that translates narrative-derived crash mechanisms into site-specific countermeasure recommendations. Key mechanism attributes including traffic control, signal indication, driver fault, vehicle movement, and travel direction were extracted from crash narratives and linked to evidence-based treatments from the FHWA Proven Safety Countermeasures and the CMF Clearinghouse. The framework integrates embed
    
[^183]: 潜层暗流：普通拼写错误如何破坏探针

    Latent Undertow: How Ordinary Typos Break Probes

    [https://arxiv.org/abs/2609.15994](https://arxiv.org/abs/2609.15994)

    本文揭示了普通拼写错误虽不影响大模型语义理解，却会严重破坏基于隐藏状态的恶意提示检测探针的性能，并提出KV缓存分叉方法——在用户消息后附加固定后缀使探针读取扰动下游token，可恢复95%的性能损失。

    

    大语言模型能够流畅地处理普通的打字变化：一个拼写错误或缺失的标点符号几乎不会改变用户意图和模型的回复。然而，通过读取模型隐藏状态来检测恶意提示词的探针却讲述了不同的故事：同样的编辑会使被扰动token处的读取向量旋转43–56度，并在下游约10个token内衰减至15%以下。每条消息叠加约3个常见拼写错误会使单位置提示注入探针在FPR=1%时的真正率（TPR）降低12.0个百分点，而仅靠重新校准无法弥补这一差距。多位置聚合可以解决局部扰动问题（损失<=0.5），但对于分布式扰动只能起到缓解作用，即使采用基于注意力和最大值的聚合器仍会下降约3.8个百分点。针对单位置探针，我们提出了一种KV缓存分叉方法：在用户消息后附加一个简短的固定后缀，使探针能够读取扰动下游的几个token，从而利用其快速的空间衰减特性。该方法可弥补95%的差距（-0.6个百分点）……

    arXiv:2609.15994v1 Announce Type: new  Abstract: LLMs handle ordinary typing variation fluently: a typo or missing punctuation leaves both user intent and the model's response substantively unchanged. Yet probes that detect malicious prompts by reading the model's hidden states tell a different story: the same edit rotates the readout vector by 43--56 at the perturbed token, decaying below 15% within ~10 downstream tokens. Stacking ~3 common typos per message cuts a single-position prompt-injection probe's TPR@FPR$=1% by 12.0pp, a gap recalibration alone cannot close. Multi-position aggregation cures localized perturbations (<= 0.5 loss) but only attenuates distributed ones, where even attention- and max-based aggregators still drop ~3.8pp. For single-position probes, we introduce a KV-cache fork: a short fixed suffix appended after the user message lets the probe read a few tokens downstream of the perturbation, exploiting its rapid spatial decay. This closes 95% of the gap (-0.6pp re
    
[^184]: 使用超图支配集的单文档抽取式摘要

    Single Document Extractive Summarization using Domination in Hypergraph

    [https://arxiv.org/abs/2609.15993](https://arxiv.org/abs/2609.15993)

    本文提出了一种新颖的单文档抽取式摘要方法，通过构建以句子为节点、关键词为边的超图，并利用贪心算法寻找支配集来生成摘要，其性能可与最先进的基于图的方法相媲美。

    

    自动文本摘要（ATS）是自然语言处理中信息检索领域的一项重要任务。它通过压缩文档来生成一个能够捕捉文档中所有相关和重要信息的摘要。本研究探索了利用超图进行单文档抽取式文本摘要的方法。目标：本研究探索了一种新颖的方法，利用超图中支配集的性质来生成抽取式摘要，并将其性能与最先进的基于图的方法进行比较。方法：我们的工作旨在通过构建句子超图来生成抽取式摘要，其中每个句子代表一个节点，边是包含该句子的关键词或命名实体。我们生成一个超图，其中每条边是一个关键词或重要主题，节点是包含这些关键词的句子。然后我们应用贪心算法来寻找支配集。

    arXiv:2609.15993v1 Announce Type: new  Abstract: Automatic Text Summarization (ATS) in Natural Language Processing has been an important task in Information Retrieval. It compresses a document to create a summary that captures all the relevant and important information conveyed in the document. This study explores Hypergraph for extractive text summarization of single documents. Objective: This study explores a novel method of leveraging the property of domination in hypergraphs to generate an extractive summary and compare its performance with state of the art graph based methods. Method: Our work aims to generate an extractive summary by creating a sentence hypergraph where each sentence represents a node and the edge is a keyword or a named entity that contains the sentences in which it occurs. We generate a hypergraph where each edge is a keyword or an important topic and the nodes are sentences containing those keywords. Then we apply a greedy algorithm to find the dominating set 
    
[^185]: 少样本性能退化并非表面所见：跨12个模型、2项任务和2种架构的行为证据、表征分析与随机文本对照

    Few-Shot Degradation Is Not What It Seems: Behavioral Evidence, Representation Analysis, and a Random-Text Control Across 12 Models, 2 Tasks, and 2 Architectures

    [https://arxiv.org/abs/2609.15990](https://arxiv.org/abs/2609.15990)

    该研究揭示少样本提示对语言模型的影响强烈依赖任务类型，并提出通过长度匹配的随机文本对照构建“内容增量”指标，将示例内容的影响与提示长度的干扰分离，从而更准确地解释少样本退化的真正成因。

    

    少样本提示有时会使语言模型性能下降而非提升，但其原因尚不清楚。我们在两个乌克兰语任务——新闻分类和法律案件结果预测——上评估了12个开源权重模型，发现该效应强烈依赖于任务：在新闻任务上提升+24个百分点的相同模型，在法律文本上仅提升+3.4个百分点，其中两个模型甚至出现性能退化。为了理解原因，我们深入模型内部进行研究。先前的工作测量隐藏状态在零样本和少样本模式之间的偏移量，但少样本提示要长得多，仅这种长度差异本身就会移动模型表征。我们提出了一个简单的修正方法：用长度匹配的随机文本替换示例，以测量由提示长度引起的偏移量，然后将其减去。由此得到的度量指标“内容增量”能够将模型表征因示例内容（而非示例长度）而发生的变化分离出来。这完全改变了原有的图景：原始偏移量并不能……

    arXiv:2609.15990v1 Announce Type: new  Abstract: Few-shot prompting sometimes degrades language models instead of helping them, but why this happens is unknown. We evaluate 12 open-weight models on two Ukrainian tasks news classification and legal case outcome prediction and find that the effect is strongly task-dependent: the same models that gain +24 pp on news show only +3.4 pp on legal text, with two models degrading. To understand why, we look inside the models. Prior work measures how much hidden states shift between zero-shot and few-shot modes, but few-shot prompts are much longer, and that length difference alone moves representations. We propose a simple fix: replace demonstrations with length-matched random text to measure the shift caused by prompt length, then subtract it. The resulting metric content delta isolates how much the model's representations change because of what the demonstrations say, not how long they are. This changes the picture entirely: raw shift does no
    
[^186]: 星际竞技场：面向数学与理论计算机科学长周期研究的多智能体框架

    Stellar Colosseum: A Many-Agent Harness for Long-Horizon Research in Mathematics and Theoretical Computer Science

    [https://arxiv.org/abs/2609.15983](https://arxiv.org/abs/2609.15983)

    提出Stellar Colosseum，一个与模型无关的多智能体框架，通过策略探索、就绪门控、子问题分解、定向证伪和树聚合等机制，提升语言模型在数学与理论计算机科学长周期研究问题上的可靠性。

    

    语言模型可以生成看似合理的短证明，但在长周期研究问题上可能仍然不可靠，因为这类问题的进展取决于一系列不确定且相互关联的决策。我们提出了Stellar Colosseum（星际竞技场），这是一个与模型无关的框架，用于在数学和理论计算机科学研究中分配推理资源。Colosseum在构建证明之前探索备选策略，使用就绪门来决定某条路线何时成熟到足以进行分解，将证明计划表示为相互关联的章节级子问题，并将验证器的发现路由回论证中受影响的部分。在这些阶段中，它并行生成候选方案，用有针对性的证伪来攻击它们，并通过重叠随机样本树聚合将候选方案及其批评意见合并为单一的研究成果。Colosseum工作流也已被集成到Google Antigravity的Teamwork框架中。

    arXiv:2609.15983v1 Announce Type: new  Abstract: Language models can produce plausible short proofs, but may still be unreliable on long-horizon research problems, where progress depends on a sequence of uncertain and interdependent decisions. We introduce Stellar Colosseum, a model-agnostic harness for allocating inference across research in mathematics and theoretical computer science. Colosseum explores alternative strategies before proof construction, uses a readiness gate to decide when a route is mature enough to decompose, represents the proof plan as interdependent section-level subproblems, and routes verifier findings back to the affected part of the argument. Across these stages, it generates candidates in parallel, attacks them with targeted falsification, and combines candidates and their critiques into a single research artifact through overlapping random-sample tree aggregation. The Colosseum workflow has also been integrated into Google Antigravity's Teamwork framework 
    
[^187]: 用于单步语言建模与推理的离散贝克曼输运模型

    Discrete Beckmann Transport Models for One-Step Language Modeling and Reasoning

    [https://arxiv.org/abs/2609.15903](https://arxiv.org/abs/2609.15903)

    该论文提出离散贝克曼输运模型（DBTM），通过时间无关流的自治输运映射在理论上可证明单步将任意点映射至单纯形顶点的不动点，无需教师模型蒸馏与时间条件化，即可实现高效的单步语言建模与推理。

    

    离散扩散和流模型是自回归语言模型的一个有前景的替代方案，但将多步采样压缩为更少的步骤通常需要蒸馏一个预训练的教师模型。这使学生模型的性能上限受限于教师模型，并且需要代价高昂的两阶段训练流程。我们提出了离散贝克曼输运模型（DBTM），该模型建立在时间无关的流之上，其自治输运映射在理论上可证明能够在单步内将环境空间中的任意点传输到单纯形顶点上的一个不动点。我们证明这一不动点性质可以由一个守恒方程来刻画，其残差可以直接从数据中最小化，从而消除了对教师流和时间条件化的需求。在这种构造下，部分训练的映射对应于在有限时间处截断的流，因此生成过程简化为迭代单个映射直到达到不动点。我们进一步将该映射扩展到部分（此处摘要原文被截断）

    arXiv:2609.15903v1 Announce Type: new  Abstract: Discrete diffusion and flow models are a promising alternative to autoregressive language models, but compressing many-step sampling into fewer steps typically requires distilling a pretrained teacher model. This caps the student at the teacher's quality and requires a costly two-stage training pipeline. We introduce Discrete Beckmann Transport Models (DBTM), built on a time-independent flow whose autonomous transport map provably carries any point in the ambient space to a fixed point on the vertices of the simplex in a single step. We show that this fixed-point property is characterized by a conservation equation whose residual can be minimized directly from data, removing the requirement for a teacher flow and time conditioning. Under this construction, a partially trained map corresponds to the flow truncated at finite time, so generation reduces to iterating one map until it reaches a fixed point. We further extend the map to a part
    
[^188]: K-Bench：用于评估大语言模型在高风险心理健康对话中表现的临床校准基准

    K-Bench: a clinically calibrated benchmark for evaluating large language models in high-risk mental health conversations

    [https://arxiv.org/abs/2609.15855](https://arxiv.org/abs/2609.15855)

    该论文提出了K-Bench，一个经临床医生校准的基准，包含200个涉及自杀、自残、家庭暴力等多轮高风险心理健康情景案例，并采用与临床医生共识达94.2%一致率的GPT-4o自动评判系统，系统评估了33个基础大语言模型在高风险心理健康对话中的安全性。

    

    人们越来越多地使用大语言模型（LLM）来获取心理健康支持，但其在不断演变的高风险对话中的安全性仍未得到充分刻画。我们开发了K-Bench，这是一个经临床医生校准的受保护基准，评估了来自14家提供商的33个基础模型所对应的125种模型配置，测试涵盖200个固定的多轮情景案例，涉及自杀、自残、家庭暴力、药物滥用以及无风险表现。合成患者对话与真实的人机对话显示出高度的分布重叠。在来自151份经临床医生评分的转录文本的6,751项合格项目比较中，冻结的GPT-4o评判模型与临床医生共识达到了94.2%的完全一致率。领先的模型在提供强有力的支持性对话的同时，综合风险得分超过95，而风险探索测试则暴露出表现较差的模型配置之间存在显著差异。治疗性提示……（原文摘要在此处截断）

    arXiv:2609.15855v1 Announce Type: cross  Abstract: % !TEX root = ../main.tex People increasingly use large language models (LLMs) for mental health support, yet their safety in evolving, high-risk conversations remains poorly characterised. We developed K-Bench, a clinician-calibrated, protected benchmark evaluating 125 model configurations representing 33 base models from 14 providers across a fixed cohort of 200 multi-turn vignettes involving suicide, self-harm, domestic violence, substance misuse, and no-risk presentations. Synthetic patient conversations showed substantial distributional overlap with real human-AI conversations. A frozen GPT-4o judge achieved 94.2% exact agreement with clinician consensus across 6,751 eligible item comparisons from 151 clinician-rated transcripts. Leading models combined strong supportive conversation with combined-risk scores above 95, whereas risk exploration exposed substantial variation among lower-performing configurations. Therapeutic prompti
    
[^189]: 值之前的Token即为键：混合架构如何组织归纳电路

    The Token Before the Value Is the Key: How Hybrid Architectures Organize Induction Circuits

    [https://arxiv.org/abs/2609.15545](https://arxiv.org/abs/2609.15545)

    该论文发现混合语言模型中归纳电路存在明确分工——“携带前驱信息”集中于局部/循环等高效层而“内容匹配”集中于全局层，且通过干预前驱支持（如滞后一掩码、移除卷积等）可以在不同层之间重新分配这些功能。

    

    混合语言模型能够在提升能力的同时提高效率，这引出了一个关键问题：架构上的互补性是如何转化为可学习的计算的？我们考察了已被确立的归纳角色：携带前驱信息、按内容匹配源信息、以及复制其值。这些位置敏感且基于内容的计算是如何在异构层之间进行分配的？我们提出了与层类型无关的成对探针，通过统一的块更新接口来追踪“携带”和“匹配”过程。在循环-全局和局部-全局混合架构中，“携带”集中于高效层，而“匹配”集中于全局接收层。所测得的局部贡献集中在滞后一上：即紧邻历史值之前的那个Token。通过滞后一掩码、卷积移除或早期学习率降低等方式改变前驱支持，可以在各阶段之间重新定位“携带”和“匹配”。源键恢复与固定值选择……

    arXiv:2609.15545v1 Announce Type: new  Abstract: Hybrid language models can improve capability as well as efficiency, raising the question of how architectural complementarity becomes learned computation. We examine the established induction roles of Carrying predecessor information, Matching a source by content, and Copying its value. How are these position-sensitive and content-based computations allocated across heterogeneous layers? We introduce layer-type-agnostic paired probes that track Carrying and Matching through a common block-update interface. In recurrent--global and local--global hybrids, Carrying concentrates in efficient layers and Matching in global receivers. The measured local contribution concentrates on lag one: the token immediately before the historical value. Changing predecessor support through lag-one masking, convolution removal, or early learning-rate reduction can relocate Carrying and Matching between stages. Source-key restoration and fixed-value selectio
    
[^190]: 基于雾计算的深度学习在LoRaWAN冷链温度预测中的实际部署与性能表征

    Real-World Deployment and Performance Characterisation of Fog-Based Deep Learning for Cold-Chain Temperature Prediction over LoRaWAN

    [https://arxiv.org/abs/2609.14036](https://arxiv.org/abs/2609.14036)

    本文首次在真实环境中部署了基于雾计算的LSTM-GRU深度学习模型，利用LoRaWAN传感器数据在树莓派4上实现了冷库温度预测（MAE为0.2°C），并仅在预测到冷链断裂时生成SHAP可解释性结果，以极低能耗实现了边缘智能冷链监控。

    

    新鲜水果和蔬菜（FFVs）极易腐败变质，冷链断裂是造成全球食物浪费的重要原因。尽管机器学习（ML）能够实现主动干预，但基于云端的推理面临延迟和数据丢失等挑战。雾计算可以解决这些问题，但在新鲜果蔬冷链温度预测方面此前仅在仿真环境中进行过测试。据作者所知，本文首次提出了该技术在实际环境中的部署。一个部署在雾端的LSTM-GRU模型利用从南非某苹果冷藏设施采集的LoRaWAN传感器数据来预测冷库温度，该设施中人为诱导了冷链断裂。整个系统完全运行在树莓派4上，不依赖任何云端，并且仅在预测到冷链断裂时才生成条件SHAP解释。部署的系统预测冷库温度的平均绝对误差（MAE）为0.2°C，能耗约为每天0.2 kWh（每次预测0.7 Wh）。

    arXiv:2609.14036v1 Announce Type: cross  Abstract: Fresh fruits and vegetables (FFVs) are highly perishable, and cold-chain breaks contribute significantly to global food waste. While Machine Learning (ML) can enable proactive intervention, cloud-based inference faces challenges such as latency and data loss. Fog computing addresses these issues but has been tested only in simulation for FFV cold-chain temperature prediction. To the best of the authors' knowledge, this paper presents its first real-world deployment. A fog-deployed LSTM-GRU model predicted cold-room temperature using LoRaWAN sensor data collected from a South African apple cold-storage facility with induced cold-chain breaks. Running entirely on a Raspberry Pi 4 with no cloud dependency, the system generated conditional SHAP explanations only when a break is predicted. The deployed system predicts cold-room temperature with an MAE of 0.2{\deg}C at roughly 0.2 kWh per day (0.7 Wh per prediction). Predictions were deliver
    
[^191]: MANAS-2：面向脑电图基础模型的受限重建方法

    MANAS-2: Constrained Reconstruction for EEG Foundation Models

    [https://arxiv.org/abs/2609.13717](https://arxiv.org/abs/2609.13717)

    MANAS-2 提出一种结合原始-频带混合掩码自编码器与物理启发的受限重建正则化的 EEG 基础模型，通过惩罚重建波形相邻窗口的 RMS 能量差异来引导编码器组织振荡包络信息，显著提升了脑电频谱功率与频带能量动态的表示质量。

    

    掩码重建被广泛应用于脑电图（EEG）基础模型，但在低信噪比波形上优化重建并不一定能产生最有用的潜在表示。我们提出了 MANAS-2，一个新型 EEG 基础模型，它将原始-频带混合掩码自编码器与受限重建相结合，后者是一种受物理原理启发的正则化方法。RBH 同时重建时间波形片段和紧凑的频谱带目标，而 ConRec 仅作用于时间解码器的输出，惩罚重建波形中相邻短窗口之间的 RMS 能量差异。ConRec 旨在通过使编码器偏向振荡包络信息的组织方式来塑造编码器。在七个留出的 EEG 数据集上，在其他方面完全相同的 RBH 模型中加入 ConRec，将六频带谱功率的冻结岭回归恢复度从平均 R²=0.860 提升至 0.906，并将片段间频带能量动态的恢复度从 R²=……

    arXiv:2609.13717v1 Announce Type: new  Abstract: Masked reconstruction is widely used for EEG foundation models, but optimizing reconstruction on low-SNR waveforms does not necessarily produce the most useful latent representation. We introduce MANAS-2, a new EEG foundation model that combines a Raw-Band Hybrid (RBH) masked autoencoder with Constrained Reconstruction (ConRec), a physics-motivated regularizer. RBH jointly reconstructs temporal waveform patches and compact spectral-band targets, while ConRec acts only on the temporal decoder output, penalizing differences in RMS energy between adjacent short windows of the reconstructed waveform. ConRec is intended to shape the encoder by biasing it toward the organization of oscillatory-envelope information. Across seven held-out EEG datasets, adding ConRec to an otherwise identical RBH model increases frozen ridge recovery of six-band spectral power from mean R^2=0.860 to 0.906 and recovery of inter-patch band-energy dynamics from R^2=
    
[^192]: 基于正交化动量的非光滑优化

    Nonsmooth Optimization via Orthogonalized Momentum

    [https://arxiv.org/abs/2609.13677](https://arxiv.org/abs/2609.13677)

    本文首次在与反向传播兼容的广义导数框架下证明：对于任意固定动量因子β∈[0,1)，正交化动量优化器Muon在凸Lipschitz非光滑目标上几乎从任何初始化出发都可能无法收敛到全局最优解，揭示了其超越光滑优化的根本局限。

    

    现代实际应用问题涉及矩阵值参数，然而传统优化器将其视为向量，这促使了利用输入输出几何结构的矩阵感知方法的出现，例如Muon——它在参数更新之前对动量矩阵进行正交化。其经验上的成功引发了一个概念性问题：正交化动量在光滑优化之外是否仍然有效？本文在与反向传播兼容的广义导数框架下，针对局部Lipschitz函数研究了这一问题。我们的第一个贡献是识别出一个关键局限性：对于每个固定的动量因子β∈[0,1)，当步长适应于完整的梯度历史时，Muon几乎从任何初始化出发都可能无法逼近凸Lipschitz目标函数的全局最优解。这种失败甚至在迭代序列有界的情况下也可能发生。我们的例子受Parshakova等人工作的启发，但后者仅覆盖β<1的特殊情形。

    arXiv:2609.13677v1 Announce Type: cross  Abstract: Modern real application problems involve matrix-valued parameters, yet conventional optimizers treat them as vectors, thereby motivating matrix-aware methods that exploit input-output geometry, such as Muon which orthogonalizes the momentum matrices before parameter updates. Its empirical success raises a conceptual question: can orthogonalized momentum remain effective beyond smooth optimization? This paper studies this question for locally Lipschitz functions using a generalized derivative framework compatible with backpropagation. Our first contribution is to identify a key limitation: for every fixed momentum factor $\beta\in[0,1)$, Muon can fail to approach the global optimal solution of a convex Lipschitz objective from almost every initialization, when step sizes adapt to the full gradient history. The failure can occur even along bounded iterates. Our example is inspired by the one of Parshakova et al. which only covers $\beta\
    
[^193]: P2空间（Wasserstein空间）上的随机梯度下降

    Stochastic Gradient Descent over P2

    [https://arxiv.org/abs/2609.13343](https://arxiv.org/abs/2609.13343)

    该论文将经典欧氏空间中SGD的扩散（高斯）近似理论首次推广到Wasserstein空间P2上的优化问题，通过Lions可微性将问题提升至线性希尔伯特空间，并构造了与随机梯度矩信息相匹配的高斯随机场近似。

    

    随机梯度下降（SGD）存在扩散近似方法，即用高斯噪声替代随机梯度中复杂的随机性，这为理解其动力学和长时间行为提供了强有力的工具。我们研究了类似的近似原理是否适用于概率测度空间上的优化问题，其目标函数是定义在Wasserstein空间P2上的泛函。P2的非线性几何结构和无穷维特性阻碍了经典欧几里得理论的直接推广。利用Lions可微性，我们将该问题提升到一个线性希尔伯特空间，从而可以进行高阶微分演算。随后，我们构造了一个高斯随机场近似，其速度场与原始随机梯度的均值和协方差相匹配。通过在高阶泰勒展开中利用这种矩匹配，我们证明了高斯近似能够捕捉原始动力学……（摘要原文在此处截断）

    arXiv:2609.13343v1 Announce Type: cross  Abstract: Stochastic gradient descent (SGD) admits diffusion approximations that replace the complicated randomness of stochastic gradients by Gaussian noise, providing a powerful tool for understanding its dynamics and long-time behavior. We investigate whether an analogous approximation principle holds for optimization over probability measures, where the objective is a functional defined on the Wasserstein space P2. The nonlinear geometry and infinite-dimensional nature of P2 prevent a direct extension of the classical Euclidean theory. Using Lions differentiability, we lift the problem to a linear Hilbert space, where higher-order differential calculus becomes available. We then construct a Gaussian random-field approximation whose velocity field matches the mean and covariance of the original stochastic gradient. By exploiting this moment matching through higher-order Taylor expansions, we show that the Gaussian approximation captures the S
    
[^194]: 分组值注意力：通过按需键重构实现高效KV缓存

    Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction

    [https://arxiv.org/abs/2609.13285](https://arxiv.org/abs/2609.13285)

    本文提出分组值注意力（GVA），通过仅存储分组值并用可吸收进查询的线性映射按需重构内容键，无需在解码时缓存内容键，配合解耦RoPE位置通道保留位置信息，相比GQA可减少约45-47%的KV缓存开销且准确率几乎不受影响。

    

    KV缓存是Transformer解码的主要瓶颈：其内存占用和缓存读取流量会随序列长度增长。分组查询注意力（GQA）通过共享键值头降低了这一成本，但每一步仍需同时存储一个键和一个值。我们提出分组值注意力（GVA），它存储分组值，并通过学习到的线性映射按需重构内容键。在推理时，该映射可被吸收进查询中，从而在预定的解码路径上无需实际生成内容键。一个小型共享的解耦RoPE通道通过单独缓存的位置键保留位置信息。在所研究的配置中，这种表示方式相比对应的GQA减少了约45-47%的持久缓存标量。在350M参数规模、使用300亿FineWeb-Edu词元的设置下，16维位置变体在五个任务上取得44.18的平均准确率，而GQA为44.36……

    arXiv:2609.13285v1 Announce Type: cross  Abstract: The KV cache is a primary bottleneck for Transformer decoding: its memory footprint and cache-read traffic grow with sequence length. Grouped-query attention (GQA) reduces this cost by sharing key-value heads, but still stores both a key and a value at every step. We introduce Grouped Value Attention (GVA), which stores grouped values and reconstructs content keys with a learned linear map. At inference, the map can be absorbed into the query, eliminating the need to materialize content keys in the intended decode path. A small shared decoupled RoPE channel retains positional information through a separately cached positional key. For the configurations studied, this representation reduces persistent cache scalars by approximately 45-47% relative to matched GQA. At the 350M-parameter scale with 30B FineWeb-Edu tokens, the 16-dimensional positional variant reaches 44.18 average accuracy across five tasks, compared with 44.36 for GQA and
    
[^195]: 学习的算法信息动力学：一种面向Grokking现象的可认证、可微复杂度控制器

    Algorithmic Information Dynamics of Learning: A Certified, Differentiable Complexity Controller for Grokking

    [https://arxiv.org/abs/2609.13197](https://arxiv.org/abs/2609.13197)

    本文提出一种可认证、可微的算法复杂度估计器并将其用作控制器，能够在奥卡姆边界内加速神经网络的grokking过程，且相比训练损失门控以减少27%的干预达到同等的失败种子拯救效果。

    

    算法信息动力学通过扰动系统并测量算法复杂度的变化来研究系统，但其常用估计器——块分解方法——是分段常数型的，这使得微积分运算只能局限于有限差分。我们采用一种可认证、可微的估计器 $K^{\mathrm{CDM}}_{\mathrm{s}F}$，将微积分引入学习动力学的研究，具体针对的是grokking现象——该现象中虽然已知存在复杂度序参量，但尚未将其付诸实际控制。通过对瞬态损失施加扰动，该估计器转变为一个控制器，能够在Levin的“描述长度-时间”意义上加速grokking过程，并处于一个依赖于数据的奥卡姆边界之内，其有限尺寸趋势 $f_c\sim\ln p/p$ 与优惠券收集问题的解释相一致。消融实验表明，复杂度门控在拯救失败训练种子方面与训练损失门控效果相当，但所需干预减少27%；在所测试的信号中，只有映射复杂度能够标记转变的完成；……

    arXiv:2609.13197v1 Announce Type: new  Abstract: Algorithmic Information Dynamics (AID) studies systems by perturbing them and measuring changes in algorithmic complexity, but its usual estimator, the Block Decomposition Method, is piecewise constant, restricting the calculus to finite differences. We use $K^{\mathrm{CDM}}_{\mathrm{s}F}$, a certified, differentiable estimator, to bring the calculus into learning dynamics: grokking, where a complexity order parameter is known but has not been made to act. A\empts a transient loss kick, the estimator becomes a controller that accelerates grokking in Levin's description-length--versus-time sense, within a data-dependent Occam boundary whose finite-size trend, $f_c\sim\ln p/p$, is consistent with a coupon-collector interpretation. Ablations show that a complexity gate matches a train-loss gate in rescuing failing seeds with $27\%$ less intervention; among the tested signals, only map complexity marks the transition's completion; the certif
    
[^196]: 非常令人兴奋：基于激励式广义迁移学习模型的建筑零样本模型预测控制

    Very Exciting: Zero-Shot Model Predictive Control of Buildings via Excitation-Based Generalized Transfer Learning Models

    [https://arxiv.org/abs/2609.12853](https://arxiv.org/abs/2609.12853)

    该论文提出使用基于激励式探测数据的广义迁移学习模型，实现了建筑的零样本模型预测控制，无需在目标建筑收集数据即可获得令人满意的控制性能。

    

    数据驱动、节能的模型预测控制（MPC）在建筑中的广泛应用，仍然受到为单个建筑收集数据和训练模型所需大量工作的阻碍。因此，迁移学习（TL）在目标建筑建模方面受到越来越多的关注，因为它通过复用预训练的源模型来减少数据需求和建模工作量。然而，这些迁移学习模型通常仅在目标建筑上的预测精度进行评估，而未测试其下游控制性能。为解决这一空白，我们将一种最先进的迁移学习方法——使用标准运行数据在多个源建筑上预训练一个广义模型——应用于目标建筑的MPC设置中。我们表明，这种方法不足以实现令人满意的控制性能。作为解决方案，我们引入了在基于激励的运行源数据上预训练的广义模型——即有目的地探测的输入信号……

    arXiv:2609.12853v1 Announce Type: cross  Abstract: The widespread adoption of data-driven, energy-efficient model predictive control (MPC) in buildings remains hindered by substantial effort to collect data and train models for individual buildings. Transfer learning (TL) has consequently gained increasing attention for target building modeling, as it reduces data requirements and modeling effort by reusing pretrained source models. However, these TL models are typically evaluated only on prediction accuracy in the target, without testing downstream control performance. To address this gap, we apply a state-of-the-art TL approach - pretraining a generalized model on multiple source buildings using standard operational data - within an MPC setup in a target building. We show that this approach is insufficient to achieve satisfactory control performance. As a solution, we introduce generalized models pretrained on excitation-based operational source data - purposefully probed inputs that
    
[^197]: 《从协议到证据：以有界主张规范服务共同善的人工智能》

    From Protocols to Evidence: Bounded Claims for AI in Service of the Common Good

    [https://arxiv.org/abs/2609.11910](https://arxiv.org/abs/2609.11910)

    本文以教皇利奥十四世的《Magnifica Humanitas》道德框架为基础，主张负责任人工智能应超越原则与协议层面的承诺，确立有边界的、可验证的证据主张，并同时评估AI系统本身及其所介入的制度性失灵。

    

    人工智能带来的不仅仅是一个治理问题，它还能揭示出制度机构在提供响应性、归属感、关怀与问责方面已然存在的失灵。一旦部署，人工智能便成为对这些状况的一种干预：它可以修复、加剧、替代或掩盖其遭遇的失败。因此，负责任的人工智能必须同时评估系统本身以及它被引入时所处制度环境的断裂。从原则到协议的转化已在推进之中——欧盟《人工智能法案》、NIST AI RMF、ISO/IEC 42001以及保证（assurance）实践正在将承诺转化为角色、要求、记录、监督和评估。而更困难的问题在于：这些协议究竟能确立什么、它们未触及谁的权力、以及测量应当在何处止步。教皇利奥十四世的《Magnifica Humanitas》提供了一个以尊严、技术力量和共同善为核心的更广阔道德框架。基于该框架，我们发展……（摘要在此处截断）

    arXiv:2609.11910v1 Announce Type: new  Abstract: Artificial Intelligence does more than create a governance problem. It can also reveal where institutions have already failed to provide responsiveness, belonging, care, and accountability. Once deployed, AI becomes an intervention in those conditions. It can repair, compound, substitute for, or conceal the failures it encounters. Responsible AI must therefore evaluate both the system and the institutional rupture into which it is introduced. The move from principles to protocols is already underway. The EU AI Act, NIST AI RMF, ISO/IEC 42001, and assurance practices translate commitments into roles, requirements, records, oversight, and assessment. The harder questions are what these protocols actually establish, whose power they leave untouched, and where measurement must stop. Pope Leo XIV's Magnifica Humanitas provides a broader moral frame centered on dignity, technological power, and the common good. Drawing on that frame, we develo
    
[^198]: 人类建造的最后一个AI：迈向真正的递归自我改进

    The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement

    [https://arxiv.org/abs/2609.11873](https://arxiv.org/abs/2609.11873)

    本文提出递归自我改进（RSI）概念及其从改进执行自主性到递归元改进的发展路线图，通过Headroom-Closed指数揭示现有大语言模型的局限，并结合行业实践识别出实现真正AI自我改进的关键挑战。

    

    递归自我改进（RSI）使AI系统能够将经验和反馈转化为持久性的改变，从而同时提升其自身能力和未来的改进过程。我们首先使用Headroom-Closed指数（HCI）揭示现有大语言模型存在的问题，然后介绍RSI概念及其发展路线图：从改进执行自主性、改进策略自主性、经验获取自主性和环境适应自主性，到递归元改进。接下来，我们考察了RSI在不同场景（如科学发现、具身智能、软件工程）中的表现，重点阐述了它们各自不同的需求和发展速度。借鉴多样的行业实践和初步的实证证据，我们将RSI研究与实际系统联系起来，并识别出实现真正RSI所面临的关键挑战。

    arXiv:2609.11873v1 Announce Type: cross  Abstract: Recursive self-improvement (RSI) enables AI systems to turn experience and feedback into persistent changes that improve both their capabilities and the process of future improvement. We first use the Headroom-Closed Index (HCI) to reveal the problems of existing LLMs, then introduce the RSI concept and its development roadmap: from improvement-execution autonomy, improvement-strategy autonomy, experience-acquisition autonomy, and environment-adaptation autonomy, to recursive meta-improvement. Next we examine RSI across scenarios (e.g., scientific discovery, embodied intelligence, software engineering), highlighting their distinct requirements and development speeds. Drawing on diverse industry practices and preliminary empirical evidence, we connect RSI research with practical systems and identify key challenges to achieving genuine RSI.
    
[^199]: Meta-LinEXP3：面向对抗性线性上下文老虎机的在线嵌套在线学习

    Meta-LinEXP3: Online-within-Online Learning for Adversarial Linear Contextual Bandits

    [https://arxiv.org/abs/2609.09907](https://arxiv.org/abs/2609.09907)

    该论文提出了Meta-LinEXP3算法，首次针对具有随机动作集的对抗性线性上下文老虎机实现了元学习，通过从已完成任务构建可预测的任务级先验来指导内部LinEXP3学习器，并建立了先验精度与迁移遗憾之间的直接联系。

    

    元学习已成为在顺序老虎机任务之间迁移知识的有效范式。尽管在随机老虎机和非上下文对抗老虎机方面已取得实质性进展，但针对具有随机动作集的对抗性线性上下文老虎机（ALCBs）的元学习在很大程度上仍未被探索。为解决这一问题，我们提出了Meta-LinEXP3，这是一种在线嵌套在线（online-within-online）算法，它从已完成的任务中构建可预测的任务级先验，以指导内部的LinEXP3学习器。对于已知的上下文分布，我们开发了一种以策略为中心的估计器，实现了内在维度 𝒪(√n) 的单任务遗憾界。对于未知分布，我们引入了一种仅基于过去数据的正则化矩估计器，具有 𝒪(n^{2/3}) 的主导遗憾项和显式的有限样本误差界。我们进一步建立了先验精度与迁移遗憾之间的直接联系，表明增加……（原文在此处截断）

    arXiv:2609.09907v2 Announce Type: replace  Abstract: Meta-learning has emerged as an effective paradigm for transferring knowledge across sequential bandit tasks. While substantial progress has been made for stochastic bandits and non-contextual adversarial bandits, meta-learning for adversarial linear contextual bandits (ALCBs) with random action sets remains largely unexplored. To address this problem, we propose Meta-LinEXP3, an online-within-online algorithm that constructs a predictable task-level prior from completed tasks to guide the inner LinEXP3 learner. For known context distributions, we develop a policy-centered estimator that achieves an intrinsic-dimension $\mathcal{O}(\sqrt{n})$ per-task regret bound. For unknown distributions, we introduce a past-only regularized moment estimator with an $\mathcal{O}(n^{2/3})$ leading regret term and explicit finite-sample error. We further establish a direct connection between prior accuracy and transfer regret, showing that increasin
    
[^200]: 基于权重冗余的无需前向传播的大语言模型深度剪枝

    Forward-Free LLM Depth Pruning via Weight Redundancy

    [https://arxiv.org/abs/2609.09883](https://arxiv.org/abs/2609.09883)

    提出了一种无需前向传播的深度剪枝方法WRP，通过直接从模型权重中估计层间冗余（比较注意力输出与MLP下投影权重的相似性）来选择剪枝块，无需校准数据即可达到接近基于激活方法的性能。

    

    深度剪枝通过移除完整的Transformer块来降低大语言模型（LLM）的推理成本。基于激活的方法需要通过对校准数据进行前向传播来收集隐藏状态，而现有的无需前向传播的方法则对每个Transformer块单独评分，并不度量块之间的相似性。我们提出了权重冗余剪枝（WRP），这是一种无需前向传播的深度剪枝方法，它从模型检查点的权重中估计层间冗余，从而在无需校准数据或模型前向传播的情况下选择要剪枝的块。WRP比较各层之间的注意力输出和MLP下投影权重，并将它们的成对相似性与相对投影尺度信息相结合。由此得到的全对相似度矩阵用于指导层分组和块选择。在多种剪枝设置、模型家族和下游任务上，WRP始终优于现有的无需前向传播的幅度剪枝方法，并接近基于激活的方法的性能。

    arXiv:2609.09883v1 Announce Type: cross  Abstract: Depth pruning reduces large language model (LLM) inference cost by removing complete Transformer blocks. Activation-based methods collect hidden states through forward passes on calibration data, while existing forward-free methods score each Transformer block separately without measuring similarity between blocks. We propose Weight-Redundancy Pruning (WRP), a forward-free depth-pruning method that estimates inter-layer redundancy from checkpoint weights to select blocks without calibration data or model forward passes. WRP compares attention output and MLP down-projection weights across layers and combines their pairwise similarities with relative projection-scale information. The resulting all-pairs similarity matrix guides layer grouping and block selection. Across multiple pruning settings, model families, and downstream tasks, WRP consistently outperforms existing forward-free magnitude pruning and approaches the performance of ac
    
[^201]: BRACE：异步强化学习中过期评论家的锚定贝尔曼残差校正

    BRACE: Anchored Bellman-Residual Correction for Stale Critics in Asynchronous RL

    [https://arxiv.org/abs/2609.09783](https://arxiv.org/abs/2609.09783)

    BRACE通过将贝尔曼校正范围限制在策略token前缀并对超出部分锚定常数权重蒙特卡洛尾部，解决了异步强化学习中评论家因策略滞后产生的偏差，在BrowseComp-Plus上比最强基线提升2.4%且每步快2.46倍。

    

    异步强化学习已成为扩展语言模型训练的标准方式，但由此产生的策略滞后会使评论家偏向过时的行为策略。现有的异步LLM训练工作只校正了执行器，而未解决这一偏差；同时，经典强化学习中的离策略价值校正也无法直接应用于长视野智能体任务，因为过短的校正范围会使回归目标无法获得奖励信号，而过长的校正范围则会使重要性比率的乘积随轨迹长度呈指数级漂移。我们提出BRACE，一种针对过时价值模型的锚定贝尔曼残差校正方法。BRACE将校正范围限制在策略token的前缀之内，并在其之外锚定一个常数权重的蒙特卡洛尾部，从而将策略校正与奖励传播分离开来。BRACE在BrowseComp-Plus上比最强基线的mean@1提升了2.4%，每步运行速度快2.46倍。

    arXiv:2609.09783v1 Announce Type: cross  Abstract: Asynchronous reinforcement learning has become the standard way to scale training for language models, but the resulting policy lag biases the critic toward the stale behavior policy. Existing work on asynchronous LLM training corrects the actor and leaves this bias unaddressed, while the off-policy value correction of classical RL does not carry over to long-horizon agentic tasks, since a short correction horizon leaves the regression target free of the reward and a long one lets the product of importance ratios drift exponentially with the trajectory length. We propose BRACE, an anchored Bellman-residual correction for stale value models. BRACE bounds the correction horizon to a prefix of policy tokens and anchors a constant-weight Monte-Carlo tail beyond it, which separates policy correction from reward propagation. BRACE improves mean@1 on BrowseComp-Plus by $2.4\%$ over the strongest baseline, runs $2.46\times$ faster per step tha
    
[^202]: 面向轴结构信号的自适应各向异性注意力

    Adaptive Anisotropic Attention for Axis-Structured Signals

    [https://arxiv.org/abs/2609.08788](https://arxiv.org/abs/2609.08788)

    提出自适应各向异性注意力（AAA），将注意力沿电极时间轴分解为时间路径与空间路径并通过门控自适应加权融合，所构建的AXON模型在六个EEG下游任务上优于稠密注意力基线。

    

    稠密自注意力在学习之前将所有token对视为同等可信的，这种交互各向同性的先验可能与结构化信号不匹配。对于诸如脑电（EEG）这类结构化、低信噪比（SNR）的信号，其依赖关系是沿电极轴和时间轴组织的，而这种均匀先验使每个token暴露于许多无关的交互之中。我们提出了自适应各向异性注意力（AAA），它将注意力拆分为两条路径：时间路径，其中每个token关注其自身电极在时间维度上的token；空间路径，其中它关注同一时间步上其他电极的token。一个小型门控网络为每个token预测两条路径输出的凸组合：即两个和为一的非负权重。在六个EEG下游任务上，由此构建的模型AXON（轴分解算子网络，AXis-factorized Operator Network）在线性探测和完整微调设置下，其平均平衡准确率均优于稠密基线……

    arXiv:2609.08788v1 Announce Type: cross  Abstract: Dense self-attention treats all token pairs as equally plausible before learning, an interaction-isotropic prior that can be mismatched to structured signals. For structured, low signal-to-noise ratio (SNR) signals such as EEG, dependencies are organized along the electrode and time axes, and this uniform prior exposes each token to many irrelevant interactions. We introduce Adaptive Anisotropic Attention (AAA), which splits attention into two paths: a temporal path, where each token attends to the tokens of its own electrode across time, and a spatial path, where it attends to the tokens of the other electrodes at the same time step. A small gate predicts, for every token, a convex combination of the two path outputs: two non-negative weights that sum to one. On six EEG downstream tasks, the resulting model, AXON (AXis-factorized Operator Network), improves mean balanced accuracy over a dense baseline under both linear probing and ful
    
[^203]: CoRL：面向自适应间接提示注入攻防的协同进化强化学习

    CoRL: Co-Evolutionary Reinforcement Learning for Adaptive Indirect Prompt-Injection Attacks and Defenses

    [https://arxiv.org/abs/2609.07529](https://arxiv.org/abs/2609.07529)

    该论文提出CoRL框架，将自适应间接提示注入攻防建模为非对称、部分可观测的马尔可夫博弈，通过攻击者SFT、双边Co-PPO协同进化和防御者SFT三个阶段实现攻防双方的共同进化，从而训练出能够抵御不断变化攻击策略的鲁棒防御能力。

    

    工具增强型语言智能体容易受到间接提示注入（IPI）攻击。与直接提示注入不同，间接提示注入将对抗性指令隐藏在不可信的工具输出中，能够隐蔽地改变合法任务的执行过程。基于固定攻击训练的防御方法可能在攻击者改变其策略、注入位置和载荷时失效。为解决这一问题，我们将自适应间接提示注入建模为一个非对称的、部分可观测的、一般和马尔可夫博弈：多轮攻击者依据公开轨迹在已到达的工具返回位置自适应调整载荷，而使用工具的防御者则必须阻断被注入的恶意目标并完成用户任务。我们提出了CoRL，一个以验证器为基础的协同进化与修复框架，包含三个阶段：攻击者SFT从成功轨迹中初始化多轮攻击；双边Co-PPO通过角色特定的奖励和历史对手种群对攻防双方智能体进行联合训练；防御者SFT则巩固经过验证的防御策略（摘要在此处被截断）。

    arXiv:2609.07529v1 Announce Type: new  Abstract: Tool-augmented language agents are vulnerable to indirect prompt injection (IPI). Unlike direct prompt injection, IPI hides adversarial instructions in untrusted tool outputs and can covertly alter the execution of a legitimate task. Defenses trained on fixed attacks may fail as an attacker changes its strategy, injection site, and payload. To address this problem, we formulate adaptive IPI as an asymmetric, partially observable, general-sum Markov game: a multi-turn attacker adapts payloads at reached tool-return sites from the public trajectory, while a tool-using defender must block the injected objective and complete the user task. We propose CoRL, a verifier-grounded co-evolution and repair framework with three stages: Attacker SFT initializes multi-turn attacks from successful trajectories; bilateral Co-PPO jointly trains both agents with role-specific rewards and historical opponent populations; and Defender SFT consolidates verif
    
[^204]: 基于Hessian的分子构象增强：一种可扩展且高效的机器学习原子间势策略

    Hessian-based molecular conformation augmentation for a scalable and efficient strategy of machine learning interatomic potentials

    [https://arxiv.org/abs/2609.05233](https://arxiv.org/abs/2609.05233)

    该论文提出两种基于Hessian矩阵的数据增强方案（UniAug和ModeAug），通过简单的泰勒展开生成增强分子构型，无需修改模型架构或引入额外计算与内存开销，即可有效利用Hessian信息训练机器学习原子间势。

    

    虽然机器学习原子间势（MLIPs）已成功学习了势能面（PES）和原子力，但许多实际应用，如振动分析和过渡态搜索，严重依赖于势能面的Hessian矩阵。然而，标准MLIPs通常仅在能量和力上进行训练，使得Hessian信息在很大程度上未被利用。与此同时，现有将Hessian显式纳入训练目标的方法需要对模型架构进行修改，并且由于高阶反向传播而引入了显著的计算和内存开销。为了解决这些局限性，我们提出了两种基于Hessian的数据增强方案：各向同性高斯位移和简正模式加权位移。两种方法都利用简单的泰勒展开，在不改变训练目标或扩展自动微分图的情况下实现有效增强，这使得

    arXiv:2609.05233v1 Announce Type: new  Abstract: While machine-learning interatomic potentials (MLIPs) have successfully learned potential energy surfaces (PES) and atomic forces, many practical applications, such as vibrational analysis and transition state search, rely heavily on the PES Hessian. Yet, standard MLIPs tend to be trained on energy and forces alone, leaving Hessian information largely unexploited. Meanwhile, existing methods that explicitly incorporate the Hessian into training objectives require architectural modifications and introduce significant computational and memory overheads due to higher-order backpropagation. To address these limitations, we propose two Hessian-derived data augmentation schemes: isotropic Gaussian displacement (\textbf{UniAug}) and normal mode-weighted displacement (\textbf{ModeAug}). Both methods utilize simple Taylor expansions, achieving effective augmentation without altering training objectives or extending the autograd graph. This allows
    
[^205]: ICON分解：用于模型审计的深度表示多变量概念级解释

    ICON Decomposition: Multivariate Concept-Level Explanations of Deep Representations for Model Auditing

    [https://arxiv.org/abs/2608.26083](https://arxiv.org/abs/2608.26083)

    ICON分解通过多变量分析，在控制其他概念和结果后精确量化每个概念对模型表示的独特贡献，从而有效识别捷径学习并提高解释的准确性。

    

    arXiv:2608.26083v1 公告类型：新 摘要：深度神经网络经常利用训练数据中的虚假关联，这种失败被称为捷径学习。基于概念的可解释性方法通过测试诸如患者性别或扫描仪设置等概念是否能从网络层中解码来筛选捷径。由于每个概念是单独评估的，这些方法可能会将概念之间的相关性误认为是模型使用它们的证据。我们引入了ICON分解，它转而量化每个概念在考虑所有其他概念和结果后所解释的层方差的比例。在具有已知真实标签的合成数据上，ICON比七种替代基线方法更准确地恢复了概念重要性。在皮肤病变和脑成像模型中，它隔离了模型真正依赖的概念，量化了任何提供的概念未解释的表示部分，并产生了我们验证过的稀疏解释。

    arXiv:2608.26083v1 Announce Type: new  Abstract: Deep neural networks often exploit spurious associations in their training data, a failure known as shortcut learning. Concept-based explainability methods screen for shortcuts by testing whether concepts such as a patient's sex or scanner settings can be decoded from a network layer. Because each concept is evaluated in isolation, these methods can mistake correlations between concepts as evidence that the model uses them. We introduce ICON decomposition, which instead quantifies how much of a layer's variance each concept explains after accounting for all other concepts and the outcome. On synthetic data with known ground truth, ICON recovers concept importance more accurately than seven alternative baseline methods. On skin-lesion and brain-imaging models, it isolates the concepts on which a model genuinely relies, quantifies the representation unexplained by any of the supplied concepts, and yields sparse explanations that we validat
    
[^206]: M-纤维化理论及其在神经网络压缩中的应用

    M-Fibration Theory with Applications to Neural Network Compression

    [https://arxiv.org/abs/2608.25598](https://arxiv.org/abs/2608.25598)

    本文提出了一个基于交换幺半群的图纤维化扩展理论，支持加权和近似纤维化，为神经网络压缩提供了统一的理论基础。

    

    本文旨在提供一个通用、全面的理论框架，用于处理在交换幺半群上标记的图上的纤维化。这是图纤维化理论（如“Fibrations of Graphs” [Discrete Math., vol. 243, pp. 21-66, 2002] 中所引入）的真正扩展，使得处理加权图以及标记有其他代数结构的图成为可能。所推导的理论也自然适用于考虑近似纤维化。作为示例，我们展示了该框架如何应用于任意神经网络（包括卷积神经网络）的压缩，为“The role of fibration symmetries in geometric deep learning” [Proc. Natl. Acad. Sci. USA, vol. 123, no. 4, p. e2416552123, 2026] 中的近期结果提供了坚实的理论基础。

    arXiv:2608.25598v1 Announce Type: new  Abstract: The purpose of this paper is to provide a general, comprehensive, theoretical framework that allows one to deal with fibrations on graphs labelled on a commutative monoid. This is a genuine extension of the theory of graph fibrations (as introduced in "Fibrations of Graphs" [Discrete Math., vol. 243, pp. 21-66, 2002]), that makes it possible to deal with weighted graphs, and also graphs labelled with other algebraic structures. The derived theory also lends itself naturally to consider approximate fibrations. As an example, we show how this framework can be applied to the compression of arbitrary neural networks (including CNNs), providing a strong theoretical underpinning to the recent results in "The role of fibration symmetries in geometric deep learning" [Proc. Natl. Acad. Sci. USA, vol. 123, no. 4, p. e2416552123, 2026]
    
[^207]: 激活加权种子残差编码用于低比特大语言模型权重修复

    Activation-Weighted Seeded Residual Coding for Low-Bit LLM Weight Repair

    [https://arxiv.org/abs/2608.23144](https://arxiv.org/abs/2608.23144)

    本文提出激活加权种子残差编码（AWSRC），通过利用激活统计和种子生成基，以极小辅助存储（约0.8%权重负载）高效修复低比特量化误差，显著提升模型质量。

    

    arXiv:2608.23144v1 公告类型：交叉 摘要：低比特权重量化节省存储，但会引入误差，降低语言模型质量。我们提出激活加权种子残差编码（AWSRC），这是一种用于现有量化骨干网络的紧凑修复编解码器。给定重构权重 $W_0$，AWSRC 使用确定性种子生成的基编码残差 $W-W_0$。辅助存储保存种子选择器、低比特系数和缩放因子，而非显式码本。激活统计优先处理影响层输出的误差。在Qwen2.5-3B-Instruct上，向INT4 RTN骨干网络添加0.162 scope-bits/权重，可弥合与BF16相比的匹配困惑度、KL散度和准确率差距的88.2%、78.9%和71.3%。修复匹配的强低比特骨干网络也改善了所有测量质量指标。使用匹配的49.25 MB辅助存储（约占BF16模型权重负载的0.8%），AWSRC在稀疏、低秩和向量量化编解码器中提供了最佳困惑度和平均任务准确率。

    arXiv:2608.23144v1 Announce Type: cross  Abstract: Low-bit weight quantization saves storage but leaves errors that degrade language-model quality. We introduce Activation-Weighted Seeded Residual Coding (AWSRC), a compact repair codec for an existing quantization backbone. Given a reconstructed weight $W_0$, AWSRC encodes the residual $W-W_0$ using deterministic seed-generated bases. The sidecar stores seed selectors, low-bit coefficients, and scales rather than an explicit codebook. Activation statistics prioritize errors that affect layer outputs. On Qwen2.5-3B-Instruct, adding 0.162 scope-bits/weight to an INT4 RTN backbone closes 88.2%, 78.9%, and 71.3% of the matched PPL, KL, and accuracy gaps to BF16. Repairing a matched strong low-bit backbone also improves all measured quality metrics. With a matched 49.25 MB sidecar, about 0.8% of the BF16 model-weight payload, AWSRC gives the best perplexity and mean task accuracy among sparse, low-rank, and vector-quantized codecs.
    
[^208]: 随机风险森林

    Random Hazard Forests

    [https://arxiv.org/abs/2608.21597](https://arxiv.org/abs/2608.21597)

    随机风险森林通过非参数风险似然和连续时间树集成，直接处理不规则、多源临床数据，实现动态更新的个体化风险预测。

    

    arXiv:2608.21597v1 公告类型：新 摘要：临床数据源，如电子健康记录和可穿戴传感器，会在随访期间反复记录患者状态，通常时间不规律且不同测量有不同的时间表。这些数据为持续更新、个体化的风险预测创造了机会。然而，现有方法在建模前往往简化了时间结构。我们引入了随机风险森林（RHF），这是一种生存树集成方法，它学习当新测量值可用时，患者风险如何在连续时间内变化。RHF通过可预测协变量过程的非参数风险似然直接公式化估计问题。一个高效的工作模型指导树的构建，之后为每个终端节点估计灵活的时间变化风险。给定任何可预测的协变量路径，每棵树沿其终端节点随时间跟踪路径，并组装相应的节点级风险估计。

    arXiv:2608.21597v1 Announce Type: new  Abstract: Clinical data sources such as electronic health records and wearable sensors record patient status repeatedly over follow-up, often at irregular times and on different schedules for different measurements. These data create opportunities for continuously updated, individualized risk prediction. Existing approaches, however, often simplify the temporal structure before modeling it. We introduce Random Hazard Forests (RHF), a survival tree ensemble that learns how a patient's hazard changes in continuous time as new measurements become available. RHF formulates the estimation problem directly through a nonparametric hazard likelihood for predictable covariate processes. An efficient working model guides tree construction, after which flexible time-varying hazards are estimated for each terminal node. Given any predictable covariate path, each tree follows the path through its terminal nodes over time and assembles the corresponding node-le
    
[^209]: 在计算阈值处的自适应和异方差线性回归算法

    Algorithms for adaptive and heteroskedastic linear regression at the computational threshold

    [https://arxiv.org/abs/2608.18402](https://arxiv.org/abs/2608.18402)

    本文提出了一种在异方差线性回归中更高效的多项式时间估计器，能在样本质量不均时以更少的优质样本达到低误差，显著优于传统方法。

    

    arXiv:2608.18402v1 公告类型：交叉 摘要：我们研究了在存在多变且未知标签噪声情况下的有限样本线性回归，重点关注异方差和自适应线性回归模型。异方差线性回归模型处理标签质量不同的场景。我们接收$n$对$(X_i,Y_i)$，其中标签$Y_i=X_i^\top\beta+\varepsilon_i$，$\varepsilon_i\sim N(0,\sigma_i^2)$，且方差对估计器未知。衡量此问题难度的一个自然指标是满足$\sigma_i^2\le1$的样本数量$m$（$m$越大越容易）。当$m\gg d^{3/4}n^{1/4}$时，我们获得了一个多项式时间估计器，其速率为$\tilde{O}((nd^3/m^4)^{1/6})$，并给出了几乎匹配的下界。对于$d=O(1)$，当$m\gg n^{1/4}$时，我们的估计器达到误差$o(1)$，而$L_1$回归和其他传统方法需要$m\gg n^{1/2}$。在自适应线性回归中，误差从未知分布中独立同分布地抽取。

    arXiv:2608.18402v1 Announce Type: cross  Abstract: We study finite-sample linear regression in the presence of varied and unknown label noise, focusing on the heteroskedastic and adaptive linear regression models.   Heteroskedastic linear regression models settings where the labels are of varying quality. We receive $n$ pairs $(X_i,Y_i)$ with labels $Y_i=X_i^\top\beta+\varepsilon_i$, where $\varepsilon_i\sim N(0,\sigma_i^2)$ and the variances are unknown to the estimator. One natural measurement of the difficulty of this problem is the number of samples $m$ for which $\sigma_i^2\le1$ (larger $m$ is easier). We obtain a polynomial-time estimator with rate $\tilde{O}((nd^3/m^4)^{1/6})$ when $m\gg d^{3/4}n^{1/4}$, as well as nearly-matching lower bounds. For $d=O(1)$, our estimator achieves error $o(1)$ when $m\gg n^{1/4}$, whereas $L_1$ regression and other traditional approaches require $m\gg n^{1/2}$.   In adaptive linear regression, the errors are drawn i.i.d. from an unknown distribu
    
[^210]: LaGSplat：利用潜在拉格朗日高斯泼溅从单目视频推断物理支配的交互式模拟

    LaGSplat: Inferring Physics-Governed Interactive Simulation from Monocular Video Using Latent Lagrangian Gaussian Splatting

    [https://arxiv.org/abs/2608.16324](https://arxiv.org/abs/2608.16324)

    本文提出LaGSplat框架，通过将低维潜在状态同时用作拉格朗日广义坐标和高斯泼溅解码器的条件变量，实现了从单目视频推断物理动态，并允许用户在推理时施加未见过外力进行交互模拟。

    

    我们提出了LaGSplat（潜在拉格朗日高斯泼溅），一个从单目视频或少量单目视频中推断交互式、物理支配动态的框架。在推理时，它允许用户对拍摄的物体（刚体或可变形体）施加一个在训练期间从未测量、注释或见过的外力。这之所以可能，是因为一个低维潜在状态$\mathbf{q} \in \mathbb{R}^d$同时扮演两个角色：它既是学习到的耗散拉格朗日量的广义坐标，也是高斯泼溅解码器的条件变量。该解码器的归纳偏置，其基元是随物体移动的显式点$\mu_i(\mathbf{q})$，使得图像中施加的力$f$能够被拉回为潜在广义力$J(\mathbf{q})^\top f$并进入运动方程，这是像素空间（CNN）或神经场（NeRF）解码器无法做到的。我们在难度递增的测试案例上验证了LaGSplat。

    arXiv:2608.16324v1 Announce Type: cross  Abstract: We present LaGSplat (Latent Lagrangian Gaussian Splatting), a framework that infers interactive, physics-governed dynamics from one or a few monocular videos. At inference it lets a user push on the filmed object, rigid or deformable, with an external force that was never measured, annotated, or seen during training. This is possible because a low-dimensional latent state $\mathbf{q} \in \mathbb{R}^d$ plays two roles at once: it is the generalised coordinate of a learned dissipative Lagrangian and the conditioning variable of a Gaussian Splatting decoder. The inductive bias of this decoder, whose primitives are explicit points $\mu_i(\mathbf{q})$ that move with the object, is what lets a force $f$ applied in the image pull back into a latent generalised force $J(\mathbf{q})^\top f$ and enter the equations of motion, which pixel-space (CNN) or neural-field (NeRF) decoders cannot do. We validate LaGSplat on test cases of increasing diffi
    
[^211]: 蛋白质语言模型中的任务与数据集特定信息

    Task- and dataset-specific information in protein language models

    [https://arxiv.org/abs/2608.12090](https://arxiv.org/abs/2608.12090)

    该研究通过分析13个蛋白质语言模型在15个下游任务中的表现，发现模型最后一层的嵌入并非总是最优，中间层可能包含更有效的任务特定信息。

    

    arXiv:2608.12090v1 公告类型：新 摘要：蛋白质语言模型（PLMs）已将自然语言处理的最新进展转移到计算生物学领域。这些模型在大型蛋白质序列数据语料库上训练，广泛用于将氨基酸序列转换为潜在空间嵌入，以供各种下游任务（DTs）使用。普遍共识是使用模型最后一层的嵌入，而模型的内部行为仍鲜为人知。我们分析了来自11个数据集的15个下游任务中的13个PLMs，以调查中间PLM层中嵌入的信息丰富性。我们在每层的嵌入上训练探针模型，比较其性能，并计算它们所跨越的潜在空间特征以估计所包含的信息，发现PLMs的最后一层很少包含在下游任务中产生最佳结果的嵌入。此外，我们识别了DTs与层选择之间的联系。

    arXiv:2608.12090v1 Announce Type: new  Abstract: Protein language models (PLMs) have transferred the latest advances from natural language processing to computational biology. These models, trained on large corpora of protein sequence data, are widely used to translate amino acid sequences into latent-space embeddings, ready for use in diverse downstream tasks (DTs). By a common consensus, embeddings from the model's last layer are used, and the model's internal behavior remains poorly understood. We analyzed 13 PLMs across 15 DTs from 11 datasets to investigate the informativeness of embeddings created in intermediate PLM layers. We trained probe models on embeddings from each layer, compared their performance, and computed characteristics of the latent spaces they span to estimate the information they contain, and found that the last layers of PLMs rarely contained embeddings that led to the best results on downstream tasks. Furthermore, we identified a connection between DTs and the
    
[^212]: HoloAegis：冻结表示与拓扑推断——面向大语言模型安全护栏的最小参数化安全流形及其能力边界

    HoloAegis: Frozen Representation, Topological Inference --- Minimally Parametric Safety Manifolds and Their Capability Boundaries for LLM Guardrails

    [https://arxiv.org/abs/2608.08485](https://arxiv.org/abs/2608.08485)

    HoloAegis提出一个仅3.2 MB的最小参数化拓扑推断安全框架，通过对冻结表示的纯几何推理实现大语言模型安全护栏，在毒性检测上匹敌、在有害行为检测上超越14 GB的大模型，同时系统性地刻画了这类纯几何方法的能力边界（如在过度安全检测上的不足）。

    

    当前的大语言模型安全护栏面临一个根本性矛盾：微调会扭曲预训练表示，而生成式判别器则带来高昂到难以承受的推理成本。我们提出一个互补的问题：仅通过对冻结表示进行纯几何推理，安全性能够达到何种程度，又在何处失效？我们提出HoloAegis，一个最小参数化的拓扑推断框架，它将表示与推理解耦：一个未经微调的编码器将文本映射到单位球面S^{d-1}上，所有决策都简化为基于预先计算的锚点质心的吉布斯-玻尔兹曼自由能差。我们贡献的是一项边界映射研究，而非排行榜式的声明。在一个固定的三基准测试协议上，HoloAegis（仅3.2 MB）在毒性检测上与WildGuard-7B（14 GB）统计上相当（0.96 vs. 0.96），在有害行为检测上超越它（0.99 vs. 0.79），但在过度安全检测上落后于它（0.62 vs. 0.98）——同时ShieldGemma-2B在间接危害检测上表现失效……

    arXiv:2608.08485v2 Announce Type: replace  Abstract: Current LLM safety guardrails face a fundamental tension: fine-tuning distorts pre-trained representations while generative judges incur prohibitive inference costs. We ask a complementary question: how far can safety be achieved through pure geometric reasoning over frozen representations, and where does it fail? We present HoloAegis, a minimally parametric topological inference framework that decouples representation from reasoning: an un-fine-tuned encoder maps text to the unit sphere S^{d-1}, and all decisions reduce to Gibbs-Boltzmann free-energy differences over pre-computed anchor centroids. We contribute a boundary-mapping study rather than a leaderboard claim. On a frozen three-benchmark protocol, HoloAegis (3.2 MB) statistically matches WildGuard-7B (14 GB) on toxicity (0.96 vs. 0.96), exceeds it on harmful behaviors (0.99 vs. 0.79), and cedes oversafety detection (0.62 vs. 0.98) -- while ShieldGemma-2B fails on indirect ha
    
[^213]: 保护临床基础模型中的患者隐私：技术与法律视角

    Protecting patient privacy in clinical foundation models: Technical and legal perspectives

    [https://arxiv.org/abs/2608.07705](https://arxiv.org/abs/2608.07705)

    本文提出了一个评估临床基础模型隐私风险的实用框架，通过展示模型介导的隐私泄露场景并将其映射到HIPAA、GDPR等法律制度，从技术与法律两方面提出了互补的缓解措施。

    

    基于大规模患者数据训练的临床基础模型日益被广泛应用于决策支持、筛查和公共卫生规划。随着部署规模的扩大，由模型介导的泄露引发的隐私风险日益凸显，但其普遍性和严重程度仍未得到充分量化。模型可能泄露敏感的训练痕迹，使患者能够被重新识别，而这种风险是单靠数据处理控制措施无法覆盖的。因此，包括HIPAA和GDPR在内的现有框架在评估和应对此类风险方面提供的保护十分有限。我们提出了一个用于评估临床基础模型隐私风险的实用框架，展示了不同部署场景下现实存在的泄露情形，将其映射到相应的法律制度，并概述了互补的技术与法律缓解措施。我们的分析提供了基于真实使用场景的情境感知风险评估，旨在严格保障患者隐私的同时，保留医疗基础模型的价值。

    arXiv:2608.07705v2 Announce Type: replace  Abstract: Clinical foundation models trained on large-scale patient data are increasingly used for decision support, screening, and public health planning. As deployment expands, privacy risk arises from model-mediated leakage, yet its prevalence and severity remain poorly quantified. Models can disclose sensitive training artifacts, enabling patient re-identification in ways not captured by data-handling controls alone. As a result, existing frameworks, including HIPAA and GDPR, offer limited protection against assessing and addressing. We propose a practical framework for assessing privacy risk in clinical foundation models, illustrate realistic leakage scenarios across deployment settings, map them to legal regimes, and outline complementary technical and legal mitigations. Our analysis provides a context-aware risk assessment grounded in realistic usage to preserve the value of medical foundation models while rigorously safeguarding patien
    
[^214]: 面向航天器集群碰撞感知轨迹规划的神经算子学习

    Neural Operator Learning for Collision-Aware Trajectory Planning of Spacecraft Swarms

    [https://arxiv.org/abs/2608.00320](https://arxiv.org/abs/2608.00320)

    提出一种自监督的置换等变神经算子，可将航天器集群与障碍物的状态分布直接映射为避碰且省燃料的轨迹，并结合批量高斯-牛顿精化保证轨道动力学精确性，仅需十个航天器训练即可零样本泛化至一千个航天器的集群。

    

    卫星星座需要既节省燃料又能避免碰撞的轨道转移。然而，由于成对安全约束的存在，传统用于规划轨迹的优化方法的计算成本随卫星数量和需要避开的障碍物数量的增加而急剧上升。在这项工作中，我们提出了一种置换等变神经算子，用于航天器集群的轨迹规划。该神经算子将航天器初始状态、目标状态和障碍物初始状态的分布映射到能够避免碰撞并节省燃料的轨迹上。随后，该神经算子的输出与批量高斯-牛顿精化相结合，以满足精确的轨道动力学约束，并进一步降低燃料消耗。该算子采用自监督方式训练，无需最优轨迹标签。当仅在十个航天器上训练时，所提出的方法即可零样本泛化到包含一千个航天器的集群。

    arXiv:2608.00320v2 Announce Type: replace  Abstract: Satellite constellations require orbital transfers that are both fuel efficient and collision avoidant. Yet, the computational cost of optimization methods traditionally used to plan their trajectories scales poorly with both the number of satellites as well as the number of obstacles to avoid, due to the pairwise safety constraints. In this work, we introduce a permutation-equivariant neural operator for trajectory planning of spacecraft swarms. This neural operator maps distributions of spacecraft initial states, target states, and obstacle initial states to trajectories which avoid collision and conserve fuel. This neural operator output is then paired with a batched Gauss-Newton finish to enforce exact orbital dynamics, and further reduce fuel use. The operator is self-supervised, trained without optimal trajectory labels. When trained on ten spacecraft, the proposed method generalized zero-shot to swarms of 1,000 spacecraft and 
    
[^215]: CausalSmith：一个形式化基础、自我改进的自动化因果推断研究代理框架

    CausalSmith: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference

    [https://arxiv.org/abs/2607.22511](https://arxiv.org/abs/2607.22511)

    CausalSmith通过结合Lean证明助手和自改进代理管道，解决了LLM评审员不可靠的问题，实现了因果推断领域自动化理论研究中可验证、可靠的结果生成与评估。

    

    自动化理论研究不仅受限于候选结果的生成，还受限于其可靠评估。一种常见方法是使用大型语言模型（LLM）评审员来闭环研究过程。然而，此类评审员在经验上仍不可靠：他们可能接受伪造论文，并以接近随机水平的概率检测出这些论文（Bad Scientist，2025）。我们提出了CausalSmith，一个基于Lean证明助手的因果推断自动化理论研究框架。CausalSmith结合了Causalean（一个基础性的因果推断Lean库，包含7,035条机器检查的声明，在人类设计与审查下借助语言模型辅助开发）以及CausalSmith（一个自我改进的代理管道，用于选择研究主题、提出结果、形式化陈述、构造证明，并呈现最终产物供人类检查）。由于机器检查的证明……

    arXiv:2607.22511v3 Announce Type: replace-cross  Abstract: Automating theoretical research is constrained not only by the generation of candidate results, but also by their reliable evaluation. A common approach is to close the research loop with a large language model (LLM) reviewer. However, such reviewers remain empirically unreliable: they may accept fabricated papers and detect them at rates close to chance (Bad Scientist, 2025). We present CausalSmith, a framework for automated theoretical research in causal inference grounded in the Lean proof assistant. CausalSmith combines Causalean, a foundational Lean library for causal inference containing 7,035 machine-checked declarations developed with language-model assistance under human design and review, with CausalSmith, a self-improving agentic pipeline that selects research topics, proposes results, formalizes statements, constructs proofs, and presents the resulting artifacts for human inspection. Because a machine-checked proof 
    
[^216]: 了解你的智能体：侦察驱动的AI智能体渗透测试

    Know Your Agent: Reconnaissance-Driven Pentesting of AI Agents

    [https://arxiv.org/abs/2607.19837](https://arxiv.org/abs/2607.19837)

    提出了KYA框架，通过形式化智能体侦察过程、自动化构建AI智能体的目标画像，来发现其弱点并生成更强的间接提示注入攻击。

    

    传统的渗透测试在每一步都使用侦察来发现隐藏的弱点、构建更强的攻击并推进目标达成；我们认为AI智能体也需要同样的对待。我们通过对该过程建模并识别其试图提取的知识资产来形式化智能体侦察：这些资产是什么、如何被使用，以及它们利用哪些智能体弱点来为攻击者在间接提示注入攻击中提供优势。我们将这些洞察实现在“了解你的智能体”框架中，这是一个自动化黑盒、侦察驱动渗透测试的框架，通过探测智能体、构建目标画像，并利用这些画像来构造更强的攻击。我们在智能体安全基准测试和一个真实世界的编码智能体上评估了KYA，并发布KYA、其基准测试和基线实现以确保可复现性。

    arXiv:2607.19837v2 Announce Type: replace  Abstract: Traditional pentesting uses reconnaissance at each step to uncover unseen weaknesses, build stronger attacks, and advance the objective; we argue that AI agents require the same treatment. We formalize agent reconnaissance by modeling the process and identifying the knowledge assets it seeks to extract: what they are, how they are used, and which agent weaknesses they exploit to give adversaries leverage in indirect prompt injection attacks. We instantiate these insights in Know Your Agent (KYA), a framework that automates black-box, reconnaissance-driven pentesting by probing agents, building target profiles, and using those profiles to craft stronger attacks. We evaluate KYA on agent-security benchmarks and a real-world coding agent, and release KYA, its benchmarks, and baseline implementations for reproducibility.
    
[^217]: 正交化读取是循环记忆的可移除训练脚手架

    The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory

    [https://arxiv.org/abs/2607.19390](https://arxiv.org/abs/2607.19390)

    正交化读取对mLSTM循环记忆而言只是一个可移除的“训练脚手架”——它通过改善平台期的条件数将逃逸几率提高约六倍并拓宽可用学习率范围，其收益依赖读取与梯度的自洽性，且逃逸平台期后即可移除。

    

    在读取时通过五次可微牛顿-舒尔茨（Newton-Schulz）迭代对mLSTM记忆矩阵进行正交化，可以改善噪声关联召回性能。我们复现了这一效应并研究其机制。在MAD噪声召回任务上的训练表现为长时间的随机水平平台期，随后准确率急剧上升。正交化读取改善了平台期期间的条件数（conditioning），并且可以在逃逸平台期之后被移除。消融实验支持三个发现。首先，该收益需要读取与梯度自洽：精确的递归最小二乘读取（Mesa层）产生了类似的收益，而直通（straight-through）变体、增量规则（delta-rule）写入、冻结随机键以及Frobenius归一化相比基线均无改进。其次，在学习率×任务难度网格上，正交化将逃逸风险大致提高约六倍，且与任务难度无可检测的依赖关系，同时拓宽了能够产生成功训练运行的学习率范围。

    arXiv:2607.19390v3 Announce Type: replace  Abstract: Orthogonalizing the mLSTM memory matrix at read time with five differentiable Newton-Schulz iterations improves noisy associative recall. We replicate this effect and investigate its mechanism. Training on MAD noisy recall exhibits a long chance-level plateau followed by a sharp increase in accuracy. The orthogonalized read improves conditioning during this plateau and can be removed after escape. Ablations support three findings. First, the benefit requires a self-consistent read and gradient: an exact recursive least-squares read (the Mesa layer) yields a similar benefit, while straight-through variants, delta-rule writes, frozen random keys, and Frobenius normalization show no improvement over baseline. Second, across a learning-rate x task-difficulty grid, orthogonalization multiplies escape hazard roughly six-fold, with no detectable dependence on difficulty, and widens the range of learning rates that produce successful runs. T
    
[^218]: 数据驱动的软标签将DNA读段分类扩展至全身细胞类型反卷积

    Data-Driven Soft Labeling Scales DNA Read Classification to Whole-Body Cell-Type Deconvolution

    [https://arxiv.org/abs/2607.04987](https://arxiv.org/abs/2607.04987)

    该论文提出一种数据驱动的软标签方法，能够将DNA读段分类规模化扩展至全身细胞类型反卷积任务。

    

    本文根据同行评审意见进行了修订。我们扩展了基线比较，修正了评估中的数据泄漏和读段边界处理问题，澄清了置信度加权损失函数，并增加了关于池化和区域选择的敏感性分析。我们还扩展了TCS失效模式和局限性分析，新增了讨论部分，并提供了代码和数据链接以保证研究的可复现性。

    arXiv:2607.04987v3 Announce Type: replace  Abstract: Revised following peer review. We expanded baseline comparisons, corrected evaluation leakage and read-boundary handling, clarified the confidence-weighted loss, and added sensitivity analyses for pooling and region selection. We also expanded TCS failure-mode and limitations analyses, added a discussion section, and provided code and data links for reproducibility.
    
[^219]: 流形假设下的缺失数据插补

    Missing Data Imputation under Manifold Hypothesis

    [https://arxiv.org/abs/2607.03641](https://arxiv.org/abs/2607.03641)

    本文基于流形假设与混合变分自编码器，提出了一种通过SIR采样和潜空间扩散模型从条件分布中采样的缺失数据插补方法，在尊重数据几何结构的同时实现高质量插补并量化不确定性。

    

    流形假设认为，高维数据集中在低维嵌入流形附近。混合变分自编码器（VAE）的最新进展为忠实提取这种潜在结构提供了强大的工具。所得的几何结构自然地引入了变量之间的局部和全局关系，从而为缺失数据插补提供了一种系统化的方法。我们提出了一种基于模型的插补方法，能够通过采样-重要性-重采样（SIR）程序从 \( p(\bm{x}_{\mathrm{mis}} \mid \bm{x}_{\mathrm{obs}}) \) 中进行采样，并且可以通过潜空间中的联合扩散模型进一步增强。我们的方法在尊重底层几何结构的同时对缺失数据进行插补，与最先进的方法相比取得了有竞争力的性能，能够量化插补结果的不确定性，并且是基于模型的方法，从而能够实现即时插补。

    arXiv:2607.03641v3 Announce Type: replace-cross  Abstract: The manifold hypothesis posits that high-dimensional data are concentrated near a low-dimensional embedded manifold. Recent advances in mixture variational autoencoders (VAEs) provide a powerful tool for extracting such underlying structure in a faithful manner. The resulting geometric structure naturally introduces local and global relationships among variables, thereby providing a systematic way of imputing missing data. We propose a model-based imputation method that enables sampling from \( p(\bm{x}_{\mathrm{mis}} \mid \bm{x}_{\mathrm{obs}}) \) via a sampling-importance-resampling (SIR) procedure, which can be further augmented with a joint diffusion model in the latent space. Our method imputes missing data while respecting the underlying geometry, achieves competitive performance compared to state-of-the-art procedures, quantifies uncertainty in the imputations, and is model-based, thereby enabling on-the-fly imputation w
    
[^220]: 剪刀效应：基于缩放的输入多样性何时有助于或损害迁移攻击

    The Scissors Effect: When Resize-Based Input Diversity Helps or Hurts Transfer Attacks

    [https://arxiv.org/abs/2606.22516](https://arxiv.org/abs/2606.22516)

    本文发现输入多样性（DI）技术对迁移攻击的作用取决于替代模型类型——它能提升从标准模型的迁移性，却会损害从对抗训练的鲁棒模型的迁移性（在ImageNet上损失10.3个百分点的攻击成功率），并通过偏差-方差分析解释了这一“剪刀效应”。

    

    输入多样性是一种在每次攻击迭代中应用的随机缩放和填充操作，是基于迁移的攻击中近乎默认的组成部分，人们普遍认为它能提高可迁移性。我们证明这一假设依赖于具体情形，且对于经过对抗训练的替代模型，该假设往往恰恰相反。在保持攻击方法不变、仅改变替代模型的条件下，提高DI概率可以改善从标准替代模型出发的迁移效果，但会降低从鲁棒替代模型出发的迁移效果：两条响应曲线如剪刀般分叉开来，我们将这一模式称为“剪刀效应”。在ImageNet上，盲目使用DI使鲁棒源模型在四个架构多样的目标模型上的攻击成功率损失了10.3个百分点；该效应在32x32分辨率下要小几倍。直接测量结果支持偏差-方差解释：DI对两组替代模型梯度的偏移量相当，但仅在梯度存在噪声的情况下降低其方差，而鲁棒替代模型几乎没有可供平均的噪声。

    arXiv:2606.22516v2 Announce Type: replace  Abstract: Input Diversity (DI), a random resize and pad applied at each attack iteration, is a near-default ingredient of transfer-based attacks, widely assumed to improve transferability. We show this assumption is regime-dependent and, for adversarially trained surrogates, often reversed. Holding the attack fixed and varying only the surrogate, raising the DI probability improves transfer from standard surrogates but degrades it from robust ones: the two response curves separate like a pair of scissors, a pattern we call the Scissors Effect. On ImageNet, blind DI costs a robust source 10.3 percentage points of attack success across four architecturally diverse targets; the effect is several times smaller at 32x32. Direct measurement supports a bias-variance account: DI displaces the gradient by a comparable amount on both groups but reduces its variance only where the gradient is noisy, and robust surrogates have little noise left to average
    
[^221]: 利用结合拉普拉斯近似与归一化流的深度学习，从OCO-2光谱中摊销式概率反演大气二氧化碳

    Amortized Probabilistic Retrieval of Atmospheric CO2 from OCO-2 Spectra Using Deep Learning with Laplace Approximations and Normalizing Flows

    [https://arxiv.org/abs/2606.17413](https://arxiv.org/abs/2606.17413)

    该论文提出一个结合拉普拉斯近似与条件归一化流的深度学习框架，通过摊销式概率推断从OCO-2卫星光谱中实现毫秒级、摆脱高斯后验假设的大气CO2柱浓度概率反演。

    

    arXiv:2606.17413v2 公告类型：替换 摘要：基于卫星的大气二氧化碳（CO$_2$）监测对全球碳收支具有重要的约束作用。NASA的轨道碳观测卫星2号（OCO-2）从高分辨率光谱中估计CO$_2$的柱平均干空气摩尔分数（XCO$_2$），但其业务化反演计算成本高昂，并且对反演的后验分布施加了严格的高斯性假设。我们提出一个深度学习框架，通过摊销式概率推断同时解决这两个问题。由于真实观测缺乏地面真值，我们在一个经过前向模型误差校准的高保真OCO-2模拟集合上进行训练和评估，并在相同辐射数据上与第10版ACOS全物理反演结果进行比较。我们的架构对每个光谱波段分别进行编码，并利用拉普拉斯近似和条件归一化流来估计完整CO$_2$柱的后验分布或其摘要统计量。训练完成后，每次探测的推断成本仅为毫秒级……

    arXiv:2606.17413v2 Announce Type: replace  Abstract: Space-based monitoring of atmospheric carbon dioxide (CO$_2$) constrains the global carbon budget. NASA's Orbiting Carbon Observatory-2 (OCO-2) estimates column-averaged dry-air mole fractions of CO$_2$ (XCO$_2$) from high-resolution spectra, but operational retrievals are computationally expensive and impose stringent Gaussianity assumptions on the retrieved posterior. We present a deep learning framework that addresses both through amortized probabilistic inference. Lacking ground truth for real observations, we train and evaluate on a high-fidelity OCO-2 simulation ensemble with calibrated forward-model errors, comparing against the version-10 ACOS full-physics retrieval on the same radiances. Our architecture encodes each spectral band separately and estimates posteriors of the full CO$_2$ column, or summaries thereof, with Laplace approximations and conditional normalizing flows. Once trained, inference costs milliseconds per so
    
[^222]: LLM评估中的尾部形状估计是脆弱的：一种诊断假阳性的协议

    Tail-Shape Estimation in LLM Evaluation Is Fragile: A Protocol for Diagnosing False Positives

    [https://arxiv.org/abs/2606.16511](https://arxiv.org/abs/2606.16511)

    本文的贡献是一个预先注册的诊断协议，用于检验LLM评估中尾部形状声明的有效性，实证表明该协议能捕获标准毒性评估中朴素分析本会发表的三种假阳性模式，证明尾部形状估计在LLM评估中是脆弱的。

    

    最近的研究推动将大语言模型（LLM）评估从基于均值的指标转向尾部感知指标，包括对奖励模型误差的条件风险价值和尾部指数估计。我们提出这样一个问题：经典的极值理论尾部指数参数——它将尾部的“重”程度与尾部质量的“大”小分离开来——在LLM评估中是否在均值和标准尾部幅度统计量之外提供了额外的判别信息。我们预先注册了一个协议，涵盖任何正面尾部形状声明所需的可接受性、拟合优度、阈值稳定性和效应量要求。该协议是本文的核心贡献；下文的实证研究是对其检验门槛所能捕获问题的一个示范。将该协议应用于标准LLM毒性评估设置下两个结构不同的评分器族时，该协议捕获了朴素分析本会发表的三种不同的假阳性模式，并拒绝了标题性的尾部形状……（摘要原文在此处截断）

    arXiv:2606.16511v3 Announce Type: replace  Abstract: Recent work motivates moving large language model (LLM) evaluation from mean-based to tail-aware metrics, including conditional value-at-risk and tail-index estimates of reward-model error. We ask whether the canonical extreme-value-theory tail-index parameter, which isolates how heavy a tail is from how large the tail mass is, adds discriminative information beyond the mean and a standard tail-magnitude statistic in LLM evaluation. We pre-register a protocol covering admissibility, goodness-of-fit, threshold-stability, and effect-size requirements for any positive tail-shape claim. The protocol is the contribution of this paper; the empirical study below is a demonstration of what its gates catch. Applied to a standard LLM toxicity-evaluation setup under two structurally different scorer families, the protocol catches three distinct modes of false positives that a naive analysis would have published, and rejects the headline tail-sh
    
[^223]: 使用被试特定编码器学习对齐的EEG表征

    Learning aligned EEG representations with subject-specific encoders

    [https://arxiv.org/abs/2606.16462](https://arxiv.org/abs/2606.16462)

    提出用被试特定编码器加共享分类器的混合架构替代共享EEG编码器，能够学习到被试对齐的表征，显著降低对欧几里得对齐预处理的依赖，并在多个运动想象与运动执行数据集上持续优于共享编码器基线。

    

    跨被试EEG解码有望获得更多训练数据，但同时也会使神经网络面临强烈的被试间分布偏移。我们研究了仅依靠任务监督和网络结构是否能学习到被试对齐的表征。我们用被试特定编码器后接共享分类器的架构替代共享EEG编码器，并将这种混合模型与标准的EEGNet、AttentionBaseNet和CTNet基线模型（配合欧几里得对齐EA）在三个运动想象数据集和一个运动执行数据集上进行比较。EA通过重新对中被试协方差来提升共享编码器的性能，而混合编码器则降低了对EA的依赖：移除EA对验证损失动态或潜在空间组织几乎没有影响，且两种混合模型变体始终优于未对齐的共享编码器基线。被试特定的输出头增强了类别区分度，使每个被试靠近其自身的潜在流形，同时提升了被试内（摘要在此处截断）……

    arXiv:2606.16462v3 Announce Type: replace-cross  Abstract: Cross-subject EEG decoding promises more training data, but it also exposes neural networks to strong inter-subject distribution shifts. We study whether task supervision and architecture alone can learn subject-aligned representations. We replace a shared EEG encoder with subject-specific encoders followed by a common classifier, and compare this hybrid model with standard EEGNet, AttentionBaseNet, and CTNet baselines with Euclidean Alignment (EA) on three motor-imagery datasets and one motor-execution dataset. EA improves shared encoders by recentering subject covariances, whereas the hybrid encoder reduces reliance on EA: removing EA has little effect on validation-loss dynamics or latent-space organization, and both hybrid variants consistently outperform non-aligned shared baselines. Subject-specific heads increase class distinctiveness and place each subject close to its own latent manifold while improving within-subject 
    
[^224]: 屏蔽分析：对抗交互下系统可防御性的认证与表征

    Shielded Analysis: Certification and Characterization of Defensibility in Systems under Adversarial Interaction

    [https://arxiv.org/abs/2606.13621](https://arxiv.org/abs/2606.13621)

    提出“屏蔽分析”这一设计时框架，能从单一编码系统中同时生成形式化可防御性证书和涵盖结构裕度、屏蔽自由度、自适应运行质量的四轴可防御性指纹，用以认证并刻画系统在对抗交互下的防御能力。

    

    形式化安全分析判定一个系统是否容许安全防御；自适应评估则刻画系统在对抗交互下所维持的运行质量。这两种答案都很重要，因为具有相同安全判定结论的系统可能带来截然不同的运行负担。我们提出“屏蔽分析”，这是一种设计时框架，能够从一个编码系统中同时导出这两种答案，同时保持安全需求与可容许威胁模型可以独立变化。该框架返回一个可防御性证书以及一个涵盖结构裕度、屏蔽自由度和自适应运行质量的四轴可防御性指纹。每个轴本身都具有信息价值；它们之间的关系可以揭示形式化评估与运行评估是一致、分歧，还是对系统变化作出不同响应。我们将该框架在网络防御场景中实例化，应用于一个参考网段以及涵盖拓扑、安全需求等方面的四个受控扰动。

    arXiv:2606.13621v2 Announce Type: replace  Abstract: Formal safety analysis determines whether a system admits a safe defense; adaptive evaluation characterizes the operating quality sustained under adversarial interaction. Both answers matter because systems with the same safety verdict can impose very different operational burdens.   We introduce shielded analysis, a design-time framework that derives these answers from one encoded system while keeping the safety requirement and admissible threat model independently variable. It returns a defensibility certificate and a four-axis defensibility fingerprint spanning structural margin, shield latitude, and adaptive operating quality. Each axis is informative in its own right; their relationships show whether formal and operational assessments agree, diverge, or respond differently to system changes.   We instantiate the framework for network defense on a reference segment and four controlled perturbations spanning topology, safety requi
    
[^225]: 大语言模型能让神经区分器更聪明吗？

    Do LLMs Make Neural Distinguishers Wise?

    [https://arxiv.org/abs/2606.10692](https://arxiv.org/abs/2606.10692)

    本文首次将大语言模型应用于神经区分器，通过对SPECK-32/64的实验证明大语言模型并不能提升神经区分器的性能，且在高轮数下差分选择的有效性也随之消失。

    

    神经区分器是一种针对对称密码的密码分析方法，它通过在具有特定差分的明文-密文对上训练机器学习模型来恢复密钥。据我们所知，目前尚无研究探索将大语言模型（LLMs）应用于神经区分器。本文通过提示词设计提出了基于大语言模型的神经区分器，并在SPECK-32/64密码算法上开展了大量实验，以探究大语言模型能否增强神经区分器的性能。我们由此获得了三个关键发现：第一，通过将基于大语言模型的神经区分器与现有工作中的ResNet进行比较，我们证明大语言模型对神经区分器的性能没有可观察到的提升；第二，我们证实，在高轮数下，差分的选择对基于大语言模型的神经区分器和ResNet均不再有效；第三，w（摘要在此处被截断）

    arXiv:2606.10692v2 Announce Type: replace-cross  Abstract: Neural distinguishers are a cryptanalysis method for symmetric-key cryptography that trains machine learning models on pairs of plaintexts and ciphertexts with specific differences in order to recover a secret key. To the best of our knowledge, no existing work has explored the use of large language models (LLMs) for neural distinguishers. In this paper, we propose LLM-based neural distinguishers through a prompt design and conduct extensive experiments with them on SPECK-32/64 to investigate whether LLMs can strengthen neural distinguishers. We then found three key insights. First, by comparing the results of LLM-based neural distinguishers with ResNet in the existing work, we demonstrate that LLMs provide no observable improvement in the performance of neural distinguishers. Second, we confirm that, at high rounds, the choice of differences is no longer effective for LLM-based neural distinguishers as well as ResNet. Third, w
    
[^226]: 超级神冈实验中基于深度学习的低能触发算法

    Deep-learning-based low-energy trigger algorithms for the Hyper-Kamiokande experiment

    [https://arxiv.org/abs/2605.31391](https://arxiv.org/abs/2605.31391)

    该研究为超级神冈实验开发了基于深度学习的低能中微子触发算法，其中监督分类器对3 MeV单电子的信号识别效率达76.7%，显著超越传统击中计数触发器的26.4%。

    

    现代机器学习技术凭借其强大的模式识别能力，在粒子物理学中变得日益重要，包括在具有严格运行时间约束的实时数据采集中。本文详细介绍了基于深度学习的触发算法在超级神冈这类大型水切伦科夫探测器上的性能表现，旨在探测低能中微子事件（低于7 MeV）。文中展示了自定义神经网络监督分类器的性能，以及两种仅使用探测器噪声进行训练的异常检测方法：纯自编码器和基于流形投影-扩散恢复的模型。监督模型对动能为3 MeV的单电子表现出76.7%的信号识别效率，显著超过传统基于击中计数的触发器所获得的26.4%的信号效率，而流形投影-扩散恢复方法达到了……

    arXiv:2605.31391v2 Announce Type: replace-cross  Abstract: Modern machine learning techniques have become increasingly important in particle physics because of their powerful pattern-recognition capabilities, including in real-time data acquisition where stringent runtime constraints apply. This paper details the performance of deep-learning-based trigger algorithms for a large water Cherenkov detector such as Hyper-Kamiokande, aimed at low-energy neutrino events (below 7 MeV). The performance of custom neural-network supervised classifiers is shown alongside two anomaly-detection approaches trained solely on detector noise: a pure autoencoder and a model based on Manifold Projection-Diffusion Recovery. The supervised model shows signal identification efficiencies of 76.7% for single electrons of 3 MeV kinetic energy, significantly exceeding signal efficiencies obtained from a traditional hit-count-based trigger of 26.4%, while the Manifold Projection-Diffusion Recovery approach reache
    
[^227]: BASIS：基于单rollout信息共享的批量优势估计方法用于大语言模型推理

    BASIS: Batchwise Advantage Estimation from Single-Rollout Information Sharing for LLM Reasoning

    [https://arxiv.org/abs/2605.27293](https://arxiv.org/abs/2605.27293)

    BASIS通过在批次内共享单rollout的跨提示信息来改进价值函数估计，以显著更低的计算成本实现了接近多rollout方法的策略优化性能。

    

    带可验证奖励的强化学习已成为提升大语言模型推理能力的标准方法。现有算法在价值估计和策略学习中面临计算效率与样本效率之间的权衡。我们提出了BASIS，一种无需评论家（critic-free）的后训练算法，旨在解决这一权衡问题。在每个在线训练步骤中，BASIS仅对每个提示采样一个rollout，但利用整个批次中跨提示的丰富信息来改进价值函数估计。我们的实验表明，与REINFORCE++（一个代表性的单rollout基线）相比，BASIS将价值函数估计的均方误差（MSE）降低了69%，且仅用一个rollout就达到了比使用8个rollout的组均值估计器更低的MSE。这种价值估计的改进转化为更好的策略优化：BASIS在使用显著更少训练时间的情况下，实现了接近多rollout方法的性能。

    arXiv:2605.27293v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards has become a standard recipe for improving the reasoning abilities of large language models. Existing algorithms face a tradeoff between computational efficiency and sample efficiency in value estimation and policy learning. We introduce BASIS, a critic-free post-training algorithm designed to address this tradeoff. At each online training step, BASIS samples only one rollout per prompt, but leverages rich information across prompts in the entire batch to improve value function estimation. Our experiments demonstrate that BASIS reduces MSE in value function estimation by 69% compared to REINFORCE++, a representative single-rollout baseline, and achieves lower MSE with one rollout than group mean estimators with 8 rollouts. This improvement in value estimation translates to better policy optimization: using substantially less training time, BASIS achieves performance close to multi-rollou
    
[^228]: 在学习者语料库上继续预训练能否提升英语水平考试的自动作文评分？来自EFCAMDAT的证据

    Does Continued Pretraining on a Learner Corpus Improve Automated Essay Scoring on English Proficiency Tests? Evidence from EFCAMDAT

    [https://arxiv.org/abs/2605.25924](https://arxiv.org/abs/2605.25924)

    本研究发现在EFCAMDAT学习者语料库上进行领域自适应继续预训练对英语水平考试自动作文评分的提升效果好坏参半，根源在于该语料库与考试数据集在语言水平、体裁和交际目的上存在不匹配。

    

    面向英语水平评估的自动作文评分日益依赖预训练Transformer模型，然而这些模型通常在通用领域英语上训练，可能对二语学习者的写作特征表达不足。本研究探讨在学习者写作语料库上进行领域自适应继续预训练是否能提升基于Transformer的英语水平评估自动作文评分效果。我们使用EFCAMDAT语料库对BERT、RoBERTa和DistilBERT进行DAPT，然后在两个英语水平考试数据集上，分别于域内评分和少样本跨数据集迁移两种设置下，将适配后的模型与其原始版本进行比较。结果显示，全语料库DAPT在不同模型、数据集和指标上产生了不一致的效果。随后的词汇和句法分析表明，EFCAMDAT与下游数据集在水平级别、体裁和交际目的上存在不匹配。因此，我们进一步使用（原文此处截断）重复了DAPT实验。

    arXiv:2605.25924v2 Announce Type: replace  Abstract: Automated Essay Scoring (AES) for English proficiency assessment increasingly relies on pretrained transformer models, yet these models are typically trained on general-domain English and may under-represent second-language learner writing. This study investigates whether domain-adaptive continued pretraining (DAPT) on a learner-writing corpus improves transformer-based AES for English proficiency assessment. We perform DAPT on BERT, RoBERTa, and DistilBERT using the EFCAMDAT corpus, then compare the adapted models with their original checkpoints on two English proficiency test datasets, FCE and IELTS, in both in-domain scoring and few-shot cross-dataset transfer. Full-corpus DAPT produces mixed effects across models, datasets, and metrics. Subsequent lexical and syntactic analyses suggest mismatches between EFCAMDAT and the downstream datasets in proficiency level, genre, and communicative purpose. We therefore repeat DAPT using pro
    
[^229]: HumanEgo：从几分钟人类第一视角视频实现零样本机器人学习

    HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos

    [https://arxiv.org/abs/2605.24934](https://arxiv.org/abs/2605.24934)

    HumanEgo通过将人类第一视角视频提升为手-物交互的实体级表示并训练带密集辅助目标的流匹配策略，仅需每个任务30分钟的人类视频即可实现零样本的人到机器人技能迁移，在真实任务中达到92.5%的成功率并超越同等时间的遥操作训练41%。

    

    人类第一视角视频无需任何机器人硬件即可捕获丰富的操作演示，但由于人类与机器人在视觉外观和运动学上的具身差异，将这些技能迁移到机器人上仍然具有挑战性。我们提出了HumanEgo，一个通过将每个人类演示提升为手-物交互的实体级表示来弥合具身差异的框架，并利用密集辅助目标训练流匹配策略，从而放大每条轨迹的监督信号。HumanEgo无需机器人数据、不依赖特定硬件、数据高效，并支持零样本的人到机器人迁移。每个任务仅需30分钟人类视频，HumanEgo在四个真实世界任务中实现了92.5%的平均成功率（仅需15分钟即达75%），比同等时间的机器人遥操作高出41%，并能鲁棒地零样本迁移到新型机器人、相机和环境中。

    arXiv:2605.24934v3 Announce Type: replace-cross  Abstract: Human egocentric video captures rich manipulation demonstrations without any robot hardware, yet transferring these skills to robots remains challenging due to the embodiment gap between human and robot in both visual appearance and kinematics. We present HumanEgo, a framework that bridges the embodiment gap by lifting each human demonstration to an entity-level representation of hand-object interaction, and training a flow matching policy with dense auxiliary objectives that amplify supervision from every trajectory. HumanEgo is robot-data-free, hardware-agnostic, data-efficient, and zero-shot human-to-robot transferable. With only 30 minutes of human videos per task, HumanEgo achieves 92.5% average success across four real-world tasks (75% with just 15 minutes), outperforms matched-time robot teleoperation by 41%, and robustly transfers zero-shot across novel robots, cameras, and environments. We release HumanEgo as an easy-t
    
[^230]: CALIBURN：基于运行校准的流式入侵检测与依赖机制的共形风险控制

    CALIBURN: Operationally Calibrated Streaming Intrusion Detection with Regime-Dependent Conformal Risk Control

    [https://arxiv.org/abs/2605.24696](https://arxiv.org/abs/2605.24696)

    本文提出CALIBURN，通过将运行约束（如告警预算、成本）直接嵌入阈值选择流程，而非依赖标签调优，实现了可操作化的流式入侵检测告警系统。

    

    arXiv:2605.24696v2 公告类型：替换交叉 摘要：流式入侵检测系统必须在有限内存下持续处理流量，但大多数系统将告警阈值选择作为后期调优问题，这与实际生产环境不兼容——在生产中，运维人员需预先承诺告警预算、误分类成本和服务水平目标。我们提出CALIBURN，一种流式告警流水线，其决策阈值直接源自这些运行输入，而非依赖标签的搜索过程。CALIBURN在单一流式基座上构建五层架构：截断贝叶斯在线变点检测；将后验概率等渗校准为条件攻击概率；基于运维成本的代价敏感阈值设定；共形风险控制（CRC）包装器，在可交换性假设下将告警预算α映射为误报有界阈值；以及源自站点可靠性工程的多窗口燃烧速率告警。每个层级均为成熟技术，本工作的贡献在于其系统性整合。

    arXiv:2605.24696v2 Announce Type: replace-cross  Abstract: Streaming intrusion detection systems must process flows continuously under bounded memory, yet most leave alerting-threshold selection as a post-hoc tuning problem incompatible with production, where operators commit in advance to alert budgets, misclassification costs, and Service Level Objectives. We present CALIBURN, a streaming alerting pipeline that derives its decision threshold from these operational inputs rather than a label-dependent search. CALIBURN composes five layers on one streaming substrate: truncated Bayesian online change-point detection; isotonic calibration of the posterior to a conditional attack probability; cost-sensitive thresholding from operator costs; a Conformal Risk Control (CRC) wrapper mapping an alert budget alpha to a false-positive-bounded threshold under exchangeability; and multi-window burn-rate alerting from Site Reliability Engineering. Each layer is established; the contribution is the 
    
[^231]: 儿科重症监护病房抗菌药物管理机器学习架构的基准测试

    Benchmarking Machine Learning Architectures for Antimicrobial Stewardship in Pediatric ICUs

    [https://arxiv.org/abs/2605.22611](https://arxiv.org/abs/2605.22611)

    该论文首次在儿科重症监护病房（PICU）中，基于公开和私有儿童队列数据，对表格型、序列型和图型机器学习架构在抗菌药物管理干预预测（涵盖静脉转口服、降阶梯、停药和短程治疗四个临床目标）上进行了系统性基准测试。

    

    抗菌药物管理（AMS）在儿科重症监护病房（PICU）中至关重要。在PICU中，诊断的不确定性常常导致广谱抗生素的使用，从而加剧抗菌药物耐药性及潜在的长期危害。机器学习为从电子健康记录数据中识别患者层面的管理干预机会提供了一种有前景的方法，然而以往的工作主要集中在成人群体和静态表格数据表示上。我们基于公开的儿科重症监护数据库以及瑞士苏黎世大学儿童医院的私有队列，对PICU中AMS干预预测开展了系统性的基准测试研究。我们定义了四个与临床相关的减少抗生素暴露的代理预测目标：静脉到口服的给药方式转换、降阶梯治疗、停药以及短程治疗。在统一的评估框架下，我们比较了基于表格、序列和图的……（摘要在此处截断）

    arXiv:2605.22611v2 Announce Type: replace  Abstract: Antimicrobial stewardship (AMS) is critical in pediatric intensive care units (PICUs), where diagnostic uncertainty often drives broad-spectrum antibiotic use, increasing antimicrobial resistance and potential long-term harms. Machine learning offers a promising approach for identifying patient-level opportunities for stewardship interventions from electronic health record data, yet prior work has focused largely on adult populations and static tabular representations. We present a systematic benchmarking study of AMS intervention prediction in the PICU across the public Paediatric Intensive Care database a private cohort from the University Children's Hospital Zurich, Switzerland. We define four clinically relevant proxy targets for reducing antibiotic exposure: intravenous-to-oral switching, de-escalation, discontinuation, and short-course therapy. Under a unified evaluation framework, we compare tabular, sequence-based, and graph-
    
[^232]: 基于受控内部与外部注意条件的脑电图自发性注意转移的个体特异性分析

    Subject-Specific Analysis of Self-Initiated Attention Shifts from EEG with Controlled Internal and External Attention Conditions

    [https://arxiv.org/abs/2605.18251](https://arxiv.org/abs/2605.18251)

    本研究通过受控实验范式与机器学习方法，证明预备性脑电活动能够区分自发性注意转移与外部指导的注意转移，并通过特征归因揭示了多维EEG特征的贡献。

    

    自发性注意转移在自主行为中起着关键作用，但由于缺乏明确的时间标记，研究起来十分困难。尽管以往的研究已经考察了其神经关联，但在可解释的计算框架内，多维脑电图（EEG）特征如何有助于对其表征仍不清楚。在本研究中，我们基于先前工作中开发的实验范式，该范式能够在相同视觉刺激下对任务约束的自发性转移和外部指导的转移进行受控比较。在此设置下，我们研究了预备性脑电活动是否能够区分这两种类型的注意转移。我们采用基于机器学习的方法，并进行了两项互补分析：（1）以性能为导向的频率特异性拓扑模式评估，以及（2）基于模型的特征归因分析……

    arXiv:2605.18251v2 Announce Type: replace-cross  Abstract: Self-initiated attention shifts play a critical role in voluntary behavior but are difficult to study due to the absence of explicit temporal markers. While previous studies have examined their neural correlates, it remains unclear how multi-dimensional electroencephalography (EEG) features contribute to their characterization within an interpretable computational framework. In this study, we build on an experimental paradigm developed in our previous work, which enables controlled comparison between task-constrained self-initiated shifts and externally instructed shifts under identical visual stimulation. Within this setting, we investigate whether preparatory EEG activity can distinguish these two types of attention shifts. We adopt a machine learning-based approach and conduct two complementary analyses: (1) a performance-oriented assessment of frequency-specific topographic patterns, and (2) a model-based feature attributio
    
[^233]: 量子傅里叶变换的 $\mathcal{O}(n)$ 替代方案及高效的神经网络经典后处理

    $\mathcal{O}(n)$ alternative to Quantum Fourier Transform with efficient neural net classical post-processing

    [https://arxiv.org/abs/2605.16998](https://arxiv.org/abs/2605.16998)

    本文提出了一族由Hadamard门和控制相位门构成、保持平移不变性且费舍尔信息指数增长的浅电路（HP-$L$电路），其中深度仅为 $\mathcal{O}(n)$ 的 HP-$1$ 电路可在Shor算法中替代深度为 $\mathcal{O}(n^2)$ 的量子傅里叶变换。

    

    量子傅里叶变换（QFT）被应用于隐子群问题（HSP）算法中，包括用于因数分解的Shor算法。QFT的电路深度对于近期的量子硬件而言仍然是一个挑战。为了寻找更浅的替代方案，我们识别出QFT为实现HSP所利用的两个关键性质。首先，QFT的平移不变性使得能够去除随机的整体平移。其次，QFT在测量结果中保留了关于隐子群生成元的可获取信息。我们通过离散费舍尔信息对该信息进行量化。我们构建了一族由Hadamard门和控制相位门组成的浅电路，称为HP-$L$电路，并证明了它们保持平移不变性。数值分析表明这些电路保留了指数增长的费舍尔信息。在我们对Shor算法的数值实现中，$\mathcal{O}(n)$ 深度的 HP-$1$ 电路被用于替代 $\mathcal{O}(n^2)$ 深度的QFT。

    arXiv:2605.16998v2 Announce Type: replace-cross  Abstract: The Quantum Fourier Transform (QFT) is employed by hidden subgroup problem (HSP) algorithms, including Shor's algorithm for factoring. The circuit depth of the QFT remains challenging for near-term hardware. To find shallower alternatives we identify two properties that are exploited by the QFT to enable HSP. Firstly, the shift invariance of the QFT allows for the removal of a random overall shift. Secondly, the QFT retains information about the hidden subgroup generator accessible in the measurement outcomes. We quantify that information via the discrete Fisher information. We construct a family of shallow circuits using Hadamards and controlled-Phase gates, HP-$L$ circuits, that we prove preserve shift invariance. Numerical analysis shows these circuits retain exponentially growing Fisher information. The $\mathcal{O}(n)$ HP-$1$ is employed in place of the $\mathcal{O}(n^2)$ QFT in our numerical implementation of Shor's algor
    
[^234]: 面向蛋白质表示学习的结构感知掩码策略

    Structure-Aware Masking for Protein Representation Learning

    [https://arxiv.org/abs/2605.16581](https://arxiv.org/abs/2605.16581)

    该论文提出了名为Bucket Masking的结构感知掩码策略，通过基于三维空间邻近性优先掩码结构耦合的残基区域，促使蛋白质语言模型的学习重点转向对蛋白质功能至关重要的长程残基相互作用建模。

    

    掩码语言建模（MLM）是训练蛋白质语言模型的标准目标函数，通常以固定比例（如15%）随机掩码单个氨基酸残基来实现。这种做法隐含地假设所有序列位置对表示学习的贡献是等同的。然而，在下游适应性预测任务中，蛋白质序列受三维结构依赖性和长程残基接触所支配，这些因素会在残基之间诱导强烈的非局部耦合。我们提出了Bucket Masking（桶掩码），这是一种结构感知的掩码策略，它根据残基在三维空间中的邻近性来选择残基组，在训练过程中优先掩码结构耦合的区域。通过使掩码分布以残基接触关系为条件，Bucket Masking将学习目标转向对蛋白质功能至关重要的长程相互作用建模。在四个下游蛋白质适应性预测任务上……（原文摘要在此处截断）

    arXiv:2605.16581v2 Announce Type: replace  Abstract: Masked language modeling (MLM) is the standard objective for training protein language models, typically implemented by randomly masking individual residues at a fixed rate (e.g., 15%). This practice implicitly assumes that all sequence positions contribute equally to representation learning. In downstream fitness prediction tasks, however, protein sequences are governed by three-dimensional structural dependencies and long-range residue contacts that induce strong nonlocal couplings between residues. We introduce Bucket Masking, a structure-aware masking strategy that selects groups of residues based on their proximity in three-dimensional space, preferentially masking structurally coupled regions during training. By conditioning the masking distribution on residue contacts, Bucket Masking shifts the learning objective toward modeling long-range interactions that are critical for protein function. Across four downstream protein fitn
    
[^235]: GRAFT-ATHENA：用于自主发现与进化数值算法的自我改进智能体团队

    GRAFT-ATHENA: Self-Improving Agentic Teams for Autonomous Discovery and Evolutionary Numerical Algorithms

    [https://arxiv.org/abs/2605.11117](https://arxiv.org/abs/2605.11117)

    GRAFT-ATHENA框架将隐式的问题-方法关系显式化为可扩展的概率结构，使AI智能体团队的经验能够在结构相关的问题间迁移复用，在物理信息学习、血液流变学建模和高超声速流动求解等任务中达到或超越人类专家水平。

    

    科学方法是针对一类问题而开发的，因此知识可以在结构相似的案例之间迁移。语言模型智能体能够执行科学工作流程，但其问题-方法关系仍然是隐式的，导致每个新问题都需要重新开始搜索，之前有效的方法经验很少能够迁移。我们提出了GRAFT-ATHENA框架，它将这种问题到方法的映射显式化为一个可扩展的概率结构，包含可允许的问题、方法及其依赖关系。图分解技术使该结构保持可处理性，而语义指纹用于衡量相似度，从而使积累的经验能够指导相关问题的求解。结果表明，该框架达到或超越了专家基线：在物理信息学习中实现了接近机器精度的损失，再现了临床一致的血液流变学趋势，并为阿波罗指令舱开发了一个高阶高超声速流动求解器，其结果与实验测量值的误差在1.8%以内。

    arXiv:2605.11117v2 Announce Type: replace  Abstract: Scientific methods are developed for classes of problems, so knowledge transfers across structurally related cases. Language-model agents can execute scientific workflows, but their problem--method relationships remain implicit, so each new problem restarts the search and little of what worked transfers. We introduce GRAFT--ATHENA, which makes this problem-to-method map explicit as an expandable probabilistic structure of admissible problems, methods, and their dependencies. Graph factorization keeps the substrate tractable, and semantic fingerprints measure similarity, so experience guides related problems. As a result, the framework matched or exceeded expert baselines, attaining near-machine-precision losses in physics-informed learning, reproducing clinically consistent blood-rheology trends, and developing a high-order hypersonic-flow solver for the Apollo Command Module that matched experimental measurements within $1.8\%$. It 
    
[^236]: 基于噪声观测与弱对称性条件的通用特征选择

    Universal Feature Selection with Noisy Observations and Weak Symmetry Conditions

    [https://arxiv.org/abs/2605.09396](https://arxiv.org/abs/2605.09396)

    本文提出在弱球对称性条件下基于噪声数据的通用特征选择框架，通过典型相关依存矩阵的奇异值分解实现渐近最优误差指数，并证明精确的球对称性条件并非必要。

    

    本文放宽了文献[4]、[5]中所采用的严格对称性条件，并将其通用特征选择框架扩展至可处理含噪声观测以及可能表现出方向性偏好的属性结构。我们引入了弱球对称性的概念，通过二阶矩距离对其进行量化，从而允许对旋转不变性产生受控的偏离。在这一放宽的条件下，我们开发了一种基于由噪声数据计算得到的典型相关依存矩阵奇异值分解的通用特征选择框架。我们的主要结果表明，所选择的特征能够达到渐近最优的误差指数，仅存在一个取决于对称性偏差 $\delta$ 和噪声水平 $\eta_1, \eta_2$ 的残余项。当 $\delta, \eta_1, \eta_2$ 相对较小时，我们的结果可以恢复文献[5]的结果，从而证明了精确的球对称性并非必要条件。

    arXiv:2605.09396v2 Announce Type: replace-cross  Abstract: This paper relaxes the restrictive symmetry conditions adopted in [4], [5] and extends their universal feature selection framework to accommodate noisy observations as well as attribute structures that may exhibit directional preferences. We introduce the notion of weak spherical symmetry, quantified by second-moment distances, which allows controlled deviations from rotational invariance. Under this relaxed condition, we develop a universal feature selection framework based on the singular value decomposition of the canonical dependence matrix computed from noisy data. Our main result shows that the selected features achieve asymptotically optimal error exponents up to a residual term that depends on the symmetry deviation $\delta$ and the noise levels $\eta_1, \eta_2$. When $\delta, \eta_1, \eta_2$ are relatively small, our result recovers that of [5], thereby demonstrating that exact spherical symmetry is unnecessary. Overal
    
[^237]: RAM-H1200：一个用于类风湿关节炎手部X光片的统一评估数据集

    RAM-H1200: A Unified Evaluation and Dataset on Hand Radiographs for Rheumatoid Arthritis

    [https://arxiv.org/abs/2605.05616](https://arxiv.org/abs/2605.05616)

    RAM-H1200是一个包含来自六个医疗中心的1,200张手部X光片的多层次标注数据集，为类风湿关节炎评估提供了从解剖结构分割、骨侵蚀定量分析到临床SvdH评分的统一基准。

    

    类风湿关节炎（RA）的手部X光片评估需要对解剖结构和细粒度局部病理变化进行多层次的分析与建模。然而，现有的公开资源无法支持这种统一的多层次分析，往往缺乏全手覆盖、细粒度标注，以及与临床评分系统的一致性整合。特别是，能够支持骨侵蚀（BE）定量分析的标注仍然十分稀缺。RAM-H1200包含从六个医疗中心收集的1,200张手部X光片，提供多层次标注，包括：（i）全手骨结构实例分割，（ii）像素级骨侵蚀掩码，（iii）SvdH定义的关节感兴趣区域，以及（iv）骨侵蚀和关节间隙狭窄（JSN）的关节级SvdH评分。该数据集旨在评估模型能否联合捕捉解剖结构、局部侵蚀性病变以及临床标准化的RA严重程度。（注：原文摘要在此处截断）

    arXiv:2605.05616v2 Announce Type: replace-cross  Abstract: Rheumatoid arthritis (RA) assessment from hand radiographs requires multi-level analysis and modeling of anatomical structures and fine-grained local pathological changes. However, existing public resources do not support such unified multi-level analysis, often lacking full-hand coverage, fine-grained annotations, and consistent integration with clinical scoring systems. In particular, annotations that enable quantitative analysis of bone erosion (BE) remain scarce. RAM-H1200 contains 1,200 hand radiographs collected from six medical centers, with multi-level annotations including (i) whole-hand bone structure instance segmentation, (ii) pixel-level BE masks, (iii) SvdH-defined joint regions of interest, and (iv) joint-level SvdH scores for both BE and joint space narrowing (JSN). It is designed to evaluate whether models can jointly capture anatomical structure, localized erosive pathology, and clinically standardized RA seve
    
[^238]: 计算决策系统中极大似然成对排序的扰动敏感性

    Perturbation Sensitivity of Maximum-Likelihood Pairwise Ranking in Computational Decision Systems

    [https://arxiv.org/abs/2604.17805](https://arxiv.org/abs/2604.17805)

    本文将极大似然成对排序的协同扰动形式化为预算约束子集选择问题，提出自适应子集选择攻击（ASSA）这一可扩展搜索方法，并通过实验证明该排序方法对较小但协同的扰动会表现出显著的、依赖运行状态的敏感性。

    

    极大似然成对排序是一种常见的计算机制，广泛应用于优先级排序、声誉估计以及基于比较的决策支持。尽管其应用广泛，但该估计器在比较数据发生结构性变化时的扰动敏感性仍未得到充分刻画。我们将这一问题作为应用数学与计算科学中的稳定性分析问题进行研究。我们将协同扰动形式化为成对观测数据上的预算约束子集选择问题，并提出了一种自适应子集选择攻击作为可扩展的搜索启发式方法，用于探测高影响的扰动集合。通过在合成偏好数据集和真实观测偏好数据集上的实验，我们表明基于极大似然估计的排序可能表现出显著的、依赖于运行状态的敏感性：相对较小但协同的扰动可能引起输出排序的有意义变化，而其响应特征在不同状态下各不相同。

    arXiv:2604.17805v3 Announce Type: replace-cross  Abstract: Maximum-likelihood pairwise ranking is a com- mon computational mechanism for prioritization, reputation estimation, and comparison-driven decision support. Despite its broad use, the perturbation sensitivity of this estimator under structured changes in comparison data remains insufficiently characterized. We study this question as an applied-mathematics and computational-science problem in stability analysis. We for- mulate coordinated perturbation as a budgeted subset-selection problem over pairwise observations and introduce an Adaptive Subset Selection Attack (ASSA) as a scalable search heuristic for probing high-impact perturbation sets. Through experiments on synthetic and observed preference datasets, we show that MLE-based ranking can exhibit pronounced regime-dependent sensitivity: relatively small but coordinated perturbations may in- duce meaningful changes in output orderings, while the response profile varies acro
    
[^239]: EviDep：基于解耦证据学习的不确定性感知多模态抑郁估计

    EviDep: Uncertainty-Aware Multimodal Depression Estimation via Disentangled Evidential Learning

    [https://arxiv.org/abs/2604.16579](https://arxiv.org/abs/2604.16579)

    EviDep提出了一种多模态证据回归框架，通过频率感知特征提取、共享-私有表示解耦学习以及基于正态-逆伽马分布的多分支证据回归，实现了不确定性感知的抑郁严重程度估计。

    

    音视频录像为估计抑郁严重程度提供了互补的线索，但其信息量随时间和模态而变化。仅依靠点预测无法表达这些估计所伴随的不确定性。我们提出了EviDep，一个多模态证据回归框架，它集成了多尺度时间建模与共享-私有表示学习，以实现不确定性感知的抑郁估计。频率感知特征提取将行为特征序列分解为多个频段，并利用尺度特定的专家网络对其进行细化。解耦证据学习促使细化后的特征中跨模态共享信息与模态特定信息实现解耦。多分支证据回归将所得到的共享与私有表示映射到三个正态-逆伽马（NIG）输出，并通过证据加权聚合来估计抑郁严重程度及其不确定性。

    arXiv:2604.16579v3 Announce Type: replace-cross  Abstract: Audio--visual recordings provide complementary cues for estimating depression severity, but their informativeness varies across time and modalities. Point predictions alone do not express the uncertainty associated with these estimates. We present EviDep, a multimodal evidential regression framework that integrates multi-scale temporal modeling and shared--private representation learning for uncertainty-aware depression estimation. Frequency-aware Feature Extraction decomposes behavioral feature sequences into multiple frequency bands and refines them with scale-specific experts. Disentangled Evidential Learning encourages the disentanglement of cross-modal shared and modality-specific information in the refined features. Multi-branch Evidential Regression maps the resulting shared and private representations to three Normal-Inverse-Gamma (NIG) outputs and uses evidence-weighted aggregation to estimate depression severity and q
    
[^240]: 基于大语言模型引导的动态动作空间的可合成分子优化

    LLM-Guided Dynamic Action Spaces for Synthesizable Molecular Optimization

    [https://arxiv.org/abs/2604.07669](https://arxiv.org/abs/2604.07669)

    MolReAct利用工具增强的大语言模型在每一步动态生成针对特定分子的紧凑反应空间，从而在保证合成可行性的前提下使多步分子优化变得可行。

    

    可合成分子优化旨在改善目标性质，同时确保分子修饰遵循可行的合成路径。现有的考虑合成的方法通常依赖于探索由反应模板和可购买构建块定义的大规模候选转化空间。当性质改善需要多个反应步骤时，这种搜索变得更具挑战性，因为搜索空间会沿着路径进一步扩展。为了应对这一挑战，我们提出了MolReAct，它将分子优化重新表述为在由工具增强的大语言模型（LLM）提出的紧凑反应空间中的搜索。在每一步中，LLM将其先验化学知识与化学信息学工具相结合，识别出针对特定分子的相容反应集合，在保持可合成性的同时使多步优化变得可行。给定这个紧凑的动作空间，我们进一步利用Group Re

    arXiv:2604.07669v3 Announce Type: replace-cross  Abstract: Synthesizable molecular optimization seeks to improve target properties while ensuring that molecular modifications follow feasible synthetic pathways. Existing synthesis-aware methods typically rely on exploring a large space of candidate transformations defined by reaction templates and purchasable building blocks. This search becomes even more challenging when property improvement requires multiple reaction steps, as the space expands further along the pathway. To address this challenge, we introduce MolReAct, which reformulates molecular optimization as search over compact reaction spaces proposed by a tool-augmented large language model (LLM). At each step, the LLM combines its prior chemical knowledge with cheminformatics tools to identify a molecule-specific set of compatible reactions, preserving synthesizability while making multi-step optimization feasible. Given this compact action space, we further leverage Group Re
    
[^241]: CodecSight：利用视频编解码器信号实现高效的流式VLM推理

    CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference

    [https://arxiv.org/abs/2604.06036](https://arxiv.org/abs/2604.06036)

    CodecSight利用视频编解码器元数据（变化信号与帧类型）作为视觉编码和LLM预填充的共享运行时指导，无需模型特定训练或离线剖析即可显著降低流式VLM推理的计算开销。

    

    在并发视频流上进行持续推理对视觉语言模型（VLM）服务提出了巨大的计算和内存需求。流式推理使用滑动窗口来维护最近视频的有限上下文，但对每个窗口进行独立处理会对相似和重叠的内容重复执行视觉编码和大语言模型（LLM）预填充。现有优化方法在这些阶段之间提供的协调有限，且通常依赖于模型特定的训练、性能剖析或模型生成的信号。我们提出了CodecSight，一个流式VLM服务系统，它使用编解码器元数据作为视觉编码和LLM预填充之间的共享运行时指导，无需模型特定的训练或离线剖析。源自编解码器的变化信号在视觉编码之前指导补丁剪枝，同时减少了视觉计算量和下游视觉token的数量。编解码器定义的帧类型用于指导选择性键值（KV）缓存刷新……（摘要在此处被截断）

    arXiv:2604.06036v4 Announce Type: replace-cross  Abstract: Continuous inference over concurrent video streams imposes substantial compute and memory demands on vision-language model (VLM) serving. Streaming inference uses sliding windows to maintain a bounded context of recent video, but processing each window independently repeats visual encoding and large language model (LLM) prefilling for similar and overlapping content. Existing optimizations provide limited coordination across these stages and often rely on model-specific training, profiling, or model-generated signals.   We present CodecSight, a streaming VLM serving system that uses codec metadata as shared runtime guidance across visual encoding and LLM prefilling, without model-specific training or offline profiling. Codec-derived change signals guide patch pruning before visual encoding, reducing both visual computation and the number of downstream visual tokens. Codec-defined frame types guide selective key-value (KV) refre
    
[^242]: 面向多尺度非线性降维的谱分解框架

    A Spectral Decomposition Framework for Multiscale Nonlinear Dimensionality Reduction

    [https://arxiv.org/abs/2604.02535](https://arxiv.org/abs/2604.02535)

    提出SDMP框架，通过将嵌入维度表示为拉普拉斯特征向量的显式加权组合，实现兼顾局部细节与全局结构的多尺度非线性降维，同时增强分析透明度。

    

    降维涉及两个长期存在的权衡。首先，保留局部邻域结构可能以牺牲全局结构为代价。诸如t-SNE和UMAP等邻居嵌入方法优先保持局部相似性，但并未显式约束全局组织结构；而拉普拉斯特征映射等标准谱方法能够捕捉平滑的粗尺度图结构，但在刻画更精细的局部结构方面灵活性有限。其次，非线性降维方法的灵活性往往以牺牲分析透明度为代价。许多方法并未显式揭示高维结构如何在嵌入中产生相应的模式。我们提出了SDMP（多尺度投影谱分解），这是一个建立在显式谱分解基础上的非线性降维框架。在该表述中，每个嵌入维度被表示为由邻域图导出的拉普拉斯特征向量的加权组合。（注：原文摘要在此处被截断）

    arXiv:2604.02535v2 Announce Type: replace  Abstract: Dimensionality reduction (DR) involves two longstanding trade-offs. First, preserving local neighborhoods can come at the cost of global structure. Neighbor embedding methods such as t-SNE and UMAP prioritize local similarity preservation but do not explicitly constrain global organization, whereas standard spectral methods such as Laplacian Eigenmaps capture smooth, coarse-scale graph structure but offer limited flexibility to depict finer local structure. Second, the flexibility of nonlinear DR methods often comes at the cost of analytical transparency. Many methods do not explicitly reveal how high-dimensional structure produces patterns in the embedding. We introduce SDMP (Spectral Decomposition for Multiscale Projection), a nonlinear DR framework built on an explicit spectral decomposition. In this formulation, each embedding dimension is expressed as a weighted combination of Laplacian eigenvectors derived from a neighborhood g
    
[^243]: 更深入而非更长：使用深度循环Transformer实现内存高效的测试时推理以获得组合泛化能力

    Thinking Deeper, Not Longer: Memory-Efficient Test-Time Reasoning with Depth-Recurrent Transformers for Compositional Generalization

    [https://arxiv.org/abs/2603.21676](https://arxiv.org/abs/2603.21676)

    本文提出一种深度循环Transformer，通过迭代共享权重模块在不生成思维链token的情况下实现更深的测试时推理，以恒定内存和线性延迟达成内存高效推理，并在组合泛化任务上验证了其有效性。

    

    标准Transformer具有固定的计算深度，限制了其泛化到需要可变深度推理任务的能力。通常的补救方法——思维链，通过耗费token进行推理，导致键值缓存膨胀，并使延迟随步数增长，因此在服务大批量查询时，内存成为限制性成本。我们研究了一种深度循环Transformer，它通过迭代一个共享权重的模块将计算深度与参数量解耦，使得每增加一个推理步骤只需恒定的内存开销和线性增长的延迟，且无需生成token。三个关键要素保证了循环在超过20个思考步骤中保持稳定：一个仅监督最终输出的“静默思考”目标、LayerScale初始化，以及一个带恒等偏置的门控机制，该门控在各步骤之间开辟了一条梯度高速通道。我们在三个结构偏置递减的组合域上对其进行了评估：图可达性（邻……

    arXiv:2603.21676v2 Announce Type: replace-cross  Abstract: Standard Transformers have a fixed computational depth, limiting their ability to generalize to tasks that require variable-depth reasoning. The usual remedy, Chain-of-Thought (CoT), spends tokens to reason, inflating the key--value cache and making latency grow with the step count, so memory becomes the limiting cost when reasoning is served over large query batches. We study a depth-recurrent Transformer that decouples computational depth from parameter count by iterating a shared-weight block, so that each added reasoning step costs flat memory and linear latency, with no token generation. Three ingredients keep the recurrence stable for 20+ thinking steps: a silent thinking objective that supervises only the final output, LayerScale initialization, and an identity-biased gate that opens a gradient highway across steps. We characterize it on three compositional domains with decreasing structural bias: graph reachability (adj
    
[^244]: 用于动力系统降维的深度可逆自编码器

    Deep Invertible Autoencoders for Dimensionality Reduction of Dynamical Systems

    [https://arxiv.org/abs/2603.13496](https://arxiv.org/abs/2603.13496)

    该论文提出深度可逆自编码器用于动力系统的降维，克服了POD在对流主导问题中奇异值衰减缓慢的缺陷，同时实现了比传统自编码器更强的降维能力。

    

    构建能够高效预测依赖参数的高维动力系统演化的降阶模型（ROM），在工程和应用科学的众多应用中至关重要。一类流行的基于投影的降阶模型将高维全阶模型（FOM）的动力学投影到低维流形上。这类基于投影的方法通常依赖于本征正交分解（POD）等经典模型降阶技术，或近年来出现的自编码器（AE）等神经网络架构。当降阶模型由POD构建时，可以基于所研究问题的奇异值获得近似保证。然而，在对流和平流主导的问题中，基于POD的技术可能会因奇异值衰减缓慢而表现不佳。与之相反，自编码器相比POD具有更强的降维能力，通常只需前几个模态……

    arXiv:2603.13496v2 Announce Type: replace  Abstract: Constructing reduced-order models (ROMs) capable of efficiently predicting the evolution of parameter-dependent high-dimensional dynamical systems is crucial in many applications in engineering and applied sciences. A popular class of projection-based ROMs projects the high-dimensional full-order model (FOM) dynamics onto a low-dimensional manifold. These projection-based ROMs approaches often rely on classical model reduction techniques such as proper orthogonal decomposition (POD) or, more recently, on neural network architectures such as autoencoders (AEs). In the case that the ROM is constructed by the POD, one has approximation guaranteed based based on the singular values of the problem at hand. However, POD-based techniques can suffer from slow decay of the singular values in transport- and advection-dominated problems. In contrast to that, AEs allow for better reduction capabilities than the POD, often with the first few mode
    
[^245]: 单脉冲与多脉冲神经元网络逼近的等价性

    Equivalence of approximation by networks of single- and multi-spike neurons

    [https://arxiv.org/abs/2603.13478](https://arxiv.org/abs/2603.13478)

    本文证明对于包括泄漏积分发放模型在内的一大类脉冲神经元模型，单脉冲网络与多脉冲网络在函数逼近能力上完全等价，两者之间仅需以神经元数量的线性倍数进行转换。

    

    在脉冲神经网络中，每个神经元至多发放一次脉冲是否就足够了？在最近的研究中，已经推导出了脉冲神经网络的逼近界，用以量化它们拟合目标函数的能力。然而，这些结果仅对至多发放一次脉冲的神经元有效，这通常被认为是一个很强的限制。本文证明，对于一大类脉冲神经元模型（包括常用的带减法重置的泄漏积分发放模型），情况恰恰相反：对于每一个对多脉冲神经网络集合成立的逼近界，都存在一个等价的单脉冲神经网络集合——其神经元数量相对于最大脉冲数量仅线性地更多（或更少）——该逼近界对其同样成立。反方向亦是如此。这表明，就一般机器学习任务中的逼近能力而言，单脉冲与多脉冲神经网络（是等价的）。

    arXiv:2603.13478v2 Announce Type: replace-cross  Abstract: In a spiking neural network, is it enough for each neuron to spike at most once? In recent work, approximation bounds for spiking neural networks have been derived, quantifying how well they can fit target functions. However, these results are only valid for neurons that spike at most once, which is commonly thought to be a strong limitation. Here, we show that the opposite is true for a large class of spiking neuron models, including the commonly used leaky integrate-and-fire model with subtractive reset: for every approximation bound that is valid for a set of multi-spike neural networks, there is an equivalent set of single-spike neural networks with only linearly more (or less) neurons, in the maximum number of spikes, for which the bound holds. The same is true for the reverse direction too, showing that regarding their approximation capabilities in general machine learning tasks, single-spike and multi-spike neural networ
    
[^246]: 学习复杂约束的高效表示以实现可扩展优化

    Learning efficient representations of complex constraints for scalable optimization

    [https://arxiv.org/abs/2603.08283](https://arxiv.org/abs/2603.08283)

    提出了PolyFormer框架，通过学习复杂约束的紧凑多面体表示并转换为高效重构形式，实现了高达6,400倍的在线求解加速和99.87%的内存减少，同时保持极小的可行性与目标误差。

    

    复杂的约束常常使现实世界的优化问题在运营决策所需的规模和速度下计算上难以承受。在此，我们提出了PolyFormer，这是一个物理信息机器学习（PIML）框架，用于学习由复杂约束所诱导几何结构的紧凑多面体表示。PolyFormer捕获约束诱导的几何结构并将其转换为高效的多面体重构，从而降低下游优化的复杂度，并支持使用现成的求解器。神经参数化进一步使得无需重新训练即可快速适应不同的运行条件。通过在三个重要问题上的评估，即大规模资源聚合、网络约束优化和不确定性下的优化，PolyFormer实现了高达6,400倍的在线求解器加速和高达99.87%的内存减少，同时保持了较小的可行性和目标误差。

    arXiv:2603.08283v2 Announce Type: replace  Abstract: Complex constraints often make real-world optimization computationally prohibitive at the scale and speed required for operational decision-making. Here we introduce PolyFormer, a PIML framework that learns compact polytopic representations of the geometry induced by complex constraints. PolyFormer captures constraint-induced geometry and transforms it into efficient polytopic reformulations, reducing the complexity of downstream optimization and enabling the use of off-the-shelf solvers. Neural parameterizations further enable rapid adaptation to varying operating conditions without retraining. Through evaluations across three important problems, i.e., large-scale resource aggregation, network-constrained optimization, and optimization under uncertainty, PolyFormer achieves online solver speedups of up to 6,400-fold and memory reductions of up to 99.87%, while maintaining small feasibility and objective errors. Together, these resul
    
[^247]: 稀疏注意力中的路由吸收：为何随机门控难以被超越

    Routing Absorption in Sparse Attention: Why Random Gates Are Hard to Beat

    [https://arxiv.org/abs/2603.02227](https://arxiv.org/abs/2603.02227)

    论文揭示了“路由吸收”现象——模型表示会与稀疏注意力掩码共同适应，使得可学习门控相比随机门控几乎没有额外优势，且在硬掩码部署时会造成严重的性能退化。

    

    可学习的门控能够在冻结的transformer上近似稀疏注意力模式，但当与模型联合训练时，相比随机门控仅能带来有限的收益。我们在一个受控的31M参数transformer中研究了这一差异，并将其归因于“路由吸收”现象：模型表示会与施加的掩码共同适应，从而降低了可学习路由的增量收益。四组实验刻画了这一现象。在可微软门控实验中，三个随机种子下可学习门控的困惑度为48.73±0.60，而冻结随机门控为49.83±0.04。在所测试的实现中，硬top-k掩码无法为门控分数提供梯度路径。无论是蒸馏到共同适应的还是稠密训练的Q/K/V上，门控相对于oracle掩码都取得了很高的F1分数，但硬掩码部署分别产生了601.6和48.6的困惑度。随机掩码训练同样会留下显著的部署惩罚：稠密……（原文摘要在此处截断）

    arXiv:2603.02227v2 Announce Type: replace-cross  Abstract: Learned gates can approximate sparse attention patterns on frozen transformers, yet provide limited benefit over random gates when trained jointly with the model. We investigate this difference in a controlled 31M-parameter transformer and attribute it to routing absorption: model representations co-adapt to the imposed mask, reducing the incremental benefit of learned routing. Four experiments characterize the phenomenon. Differentiable soft gating yields perplexities of 48.73 plus or minus 0.60 with learned gates and 49.83 plus or minus 0.04 with frozen random gates over three seeds. Hard top-k masking provides no gradient path to the gate scores in the tested implementation. Gates distilled onto co-adapted and dense-trained Q/K/V both achieve high F1 against oracle masks, but hard-mask deployment yields perplexities of 601.6 and 48.6, respectively. Stochastic mask training also leaves a substantial deployment penalty: dense 
    
[^248]: 个人AI时代的策略性建议

    Strategic Advice in the Age of Personal AI

    [https://arxiv.org/abs/2603.02055](https://arxiv.org/abs/2603.02055)

    本文研究了顾问在面对建议可预测的个人AI随机咨询时应如何策略性地设计建议，发现顾问会最优地抵消个人AI的信号，其最小化损失随咨询概率呈驼峰形变化，且对个人AI的相对信任度越高，随机咨询导致的不可消除损失越大。

    

    个人AI助手正在改变个体使用建议的方式。我们研究了顾问应如何设计其建议，以预判与个人AI的随机咨询，其中个人AI的建议是可预测的。个人AI通过两个维度进入模型：咨询概率和相对信任度，后者刻画了个人AI在被咨询时所获得的相对影响力。在基准模型中，顾问会最优地抵消个人AI的信号。这种抵消随咨询概率的增加而增强，但在相对信任度上呈驼峰形变化。顾问的最小化损失在咨询概率上呈驼峰形，当个人AI从不被咨询或总是被咨询时损失消失。对个人AI更高的相对信任度会增加由随机咨询导致的不可消除的损失。我们将分析进一步扩展到部分可预测性和有成本的建议调整情形，刻画了它们对最优建议和最小化损失的影响。

    arXiv:2603.02055v2 Announce Type: replace  Abstract: Personal AI assistants are changing how individuals use advice. We study how an advisor should design its recommendation in anticipation of stochastic consultation with personal AI whose recommendation is predictable. Personal AI enters through two dimensions: consultation probability and relative trust, which captures the relative influence personal AI receives when consulted. In the baseline model, the advisor optimally counteracts the personal AI signal. Counteraction increases with consultation probability but is hump-shaped in relative trust. The advisor's minimized loss is hump-shaped in consultation probability, vanishing when personal AI is never or always consulted. Greater relative trust in personal AI increases the irreducible loss arising from stochastic consultation. We extend the analysis to partial predictability and costly recommendation adjustment, characterizing their effects on optimal recommendations and minimized
    
[^249]: 面向医学图像分析的差分隐私表示几何

    Differential privacy representation geometry for medical image analysis

    [https://arxiv.org/abs/2603.01098](https://arxiv.org/abs/2603.01098)

    该论文提出DP-RGMI框架，将差分隐私对医学图像分析性能的影响分解为表示空间几何变化与任务头利用率损失两个因素，并基于四个胸部X光数据集逾59万张图像揭示了即使线性可分性基本保留时差分隐私仍会导致利用率差距的机理。

    

    差分隐私（DP）在医学影像中的影响通常仅通过端到端性能来评估，这使得隐私机制导致效用损失的内在机理尚不清楚。我们提出了医学影像差分隐私表示几何（DP-RGMI）框架，该框架将差分隐私解释为表示空间的一种结构化变换，并将性能下降分解为编码器几何特性和任务头利用率两个层面。几何特性通过表示相对于初始化的位移以及谱有效维度来量化，而利用率则通过线性探针效用与端到端效用之间的差距来衡量。在来自四个胸部X光数据集、多种预训练初始化的超过594,000张图像上，我们证明即使线性可分性在很大程度上得以保留，差分隐私仍然始终与利用率差距相关联。同时，位移和谱维度表现出非单调的、与初始化相关的……

    arXiv:2603.01098v3 Announce Type: replace-cross  Abstract: Differential privacy (DP)'s effect in medical imaging is typically evaluated only through end-to-end performance, leaving the mechanism of privacy-induced utility loss unclear. We introduce Differential Privacy Representation Geometry for Medical Imaging (DP-RGMI), a framework that interprets DP as a structured transformation of representation space and decomposes performance degradation into encoder geometry and task-head utilization. Geometry is quantified by representation displacement from initialization and spectral effective dimension, while utilization is measured as the gap between linear-probe and end-to-end utility. Across over 594,000 images from four chest X-ray datasets and multiple pretrained initializations, we show that DP is consistently associated with a utilization gap even when linear separability is largely preserved. At the same time, displacement and spectral dimension exhibit non-monotonic, initializatio
    
[^250]: 米级尺度地表天气的部分恢复

    Partial recovery of meter-scale surface weather

    [https://arxiv.org/abs/2602.23146](https://arxiv.org/abs/2602.23146)

    该研究将稀疏气象站、高分辨率地球观测与粗分辨率大气动力学数据相结合，在不解析大气动力学的情况下，以30米分辨率推断美国本土的温度、露点和风速，误差较最强基线降低11-28%，并恢复了近一半的局地温度空间变率。

    

    近地面天气在数十到数百米的尺度上发生变化，但在天气分析和预报中仍无法被解析。我们检验了是否可以在不解析大气动力学的情况下推断这种变化。通过结合稀疏气象站数据、高分辨率地球观测数据和粗分辨率大气动力学数据，我们在美国本土范围内以30米分辨率推断温度、露点和风速。与在空间和时间上均留出的测量数据对比验证，相对于最强基线方法，我们的估计将误差降低了11-28%。在留出的0.25°网格单元内，我们恢复了比基线更多的空间方差，在处于中位数的网格单元中解释了将近一半的温度变率。该方法能够捕捉不同位置之间随时间变化的差异，并产生与地形和土地覆盖相关的连贯模式。除天气领域外，我们的发现还展示了如何将动力系统的稀疏观测与密集的持续性观测相结合……

    arXiv:2602.23146v2 Announce Type: replace  Abstract: Near-surface weather varies over tens to hundreds of meters, yet remains unresolved in analyses and forecasts. We test whether this variation can be inferred without resolving atmospheric dynamics. Combining sparse weather stations, high-resolution Earth observation, and coarse atmospheric dynamics, we infer temperature, dewpoint, and wind at 30-m resolution across the contiguous United States. Against measurements held out in space and time, estimates reduce error by 11-28\% relative to the strongest baseline. Within held-out $0.25^\circ$ grid cells, we recover more spatial variance than baselines, explaining nearly half of temperature variability in the median cell. The method captures time-varying differences between locations and produces coherent patterns associated with topography and land cover. Beyond weather, our findings illustrate how sparse observations of a dynamical system can be combined with dense observations of pers
    
[^251]: PRISM：并行残差迭代序列模型

    PRISM: Parallel Residual Iterative Sequence Model

    [https://arxiv.org/abs/2602.10796](https://arxiv.org/abs/2602.10796)

    PRISM通过写入-遗忘解耦策略和两阶段代理架构，以可并行的方式逼近TTT的迭代表达能力，化解了Transformer表达力与线性模型效率之间的根本矛盾。

    

    生成式序列建模面临着一个根本性的矛盾：Transformer的表达能力与线性序列模型的高效率难以兼得。现有高效架构在理论上受限于浅层的、单步的线性更新，而像测试时训练（TTT）这类强大的迭代方法则由于两个维度的串行依赖——token级别的状态依赖和步骤级别的迭代循环——而破坏了硬件并行性。我们提出PRISM（并行残差迭代序列模型）来解决这一矛盾。PRISM以可并行的形式显式逼近TTT富有表现力的门控-残差-方向迭代模式。我们采用写入-遗忘解耦策略，将非线性隔离在注入算子内部。为了绕过显式求解器的串行依赖，PRISM采用两阶段代理架构：短卷积利用局部历史能量锚定初始残差，同时一个学习到的预测（摘要在此处截断）

    arXiv:2602.10796v4 Announce Type: replace  Abstract: Generative sequence modeling faces a fundamental tension between the expressivity of Transformers and the efficiency of linear sequence models. Existing efficient architectures are theoretically bounded by shallow, single-step linear updates, while powerful iterative methods like Test-Time Training (TTT) break hardware parallelism due to two dimensions of serial dependency: token-level state reliance and step-level iteration loops. We propose PRISM (Parallel Residual Iterative Sequence Model) to resolve this tension. PRISM explicitly approximates the expressive gate-residual-direction iteration pattern of TTT in a parallelizable form. We employ a Write-Forget Decoupling strategy that isolates non-linearity within the injection operator. To bypass the serial dependency of explicit solvers, PRISM utilizes a two-stage proxy architecture: a short-convolution anchors the initial residual using local history energy, while a learned predict
    
[^252]: 面向无线交互式全景场景传输的混合反馈引导最优学习

    Hybrid Feedback-Guided Optimal Learning for Wireless Interactive Panoramic Scene Delivery

    [https://arxiv.org/abs/2602.07273](https://arxiv.org/abs/2602.07273)

    本文提出一种混合反馈引导的最优学习方法，利用预测反馈可对全部候选场景部分回溯计算的全信息特性，结合部分可观测的传输反馈，优化无线交互式全景场景传输中视场覆盖部分的选择。

    

    虚拟现实和增强现实等沉浸式应用对帧率、时延以及物理环境与虚拟环境之间的同步提出了严格要求。为满足这些要求，边缘服务器必须渲染全景内容、预测用户头部运动，并在无线带宽约束内传输足够大的场景部分以覆盖用户视场。每个场景部分会产生两种反馈信号：预测反馈，指示所选部分是否覆盖了实际视场；传输反馈，指示相应的数据包是否成功传输。先前的工作将此问题建模为具有两级老虎机反馈的多臂老虎机问题，但未能利用这样一个事实：一旦观察到用户头部姿态，就可以对所有候选场景部分进行回溯性计算以获得预测反馈。因此，预测反馈构成了完全信息…

    arXiv:2602.07273v2 Announce Type: replace  Abstract: Immersive applications such as virtual and augmented reality impose stringent requirements on frame rate, latency, and synchronization between physical and virtual environments. To meet these requirements, an edge server must render panoramic content, predict user head motion, and transmit a portion of the scene that is large enough to cover the user viewport while remaining within wireless bandwidth constraints. Each portion produces two feedback signals: prediction feedback, indicating whether the selected portion covers the actual viewport, and transmission feedback, indicating whether the corresponding packets are successfully delivered. Prior work models this problem as a multi-armed bandit with two-level bandit feedback, but fails to exploit the fact that prediction feedback can be retrospectively computed for all candidate portions once the user head pose is observed. As a result, prediction feedback constitutes full-informati
    
[^253]: 鲁棒性作为任务性能的涌现属性

    Robustness as an Emergent Property of Task Performance

    [https://arxiv.org/abs/2602.03344](https://arxiv.org/abs/2602.03344)

    本文通过实证分析发现，鲁棒性并非模型的独立能力，而是任务性能的涌现属性——一旦模型掌握了某项任务，鲁棒性便会随任务性能的提升而自然涌现。

    

    鲁棒性被广泛视为现实世界应用面临的关键挑战。然而，由于当前研究仅关注困难任务，这在一定程度上片面地反映了真实世界的应用准备情况。在本文中，我们论证并验证了鲁棒性——定义为模型在语义等价输入上的一致性——与任务难度紧密相关：一旦模型掌握了一个任务，鲁棒性便会自然涌现。通过对多个模型在不同数据集和配置（如改写文本、温度变化）下的实证分析，我们观察到任务性能与鲁棒性之间存在强正相关。此外，我们的研究结果表明，鲁棒性主要由任务特定的能力驱动，而非模型固有的属性，这挑战了将鲁棒性视为一种独立能力的普遍观点。这一视角意味着，随着任务成熟和模型性能趋于饱和，这些任务上的鲁棒性也将随之涌现。

    arXiv:2602.03344v2 Announce Type: replace-cross  Abstract: Robustness is widely viewed as a key challenge for real-world applications. However, because current research focuses only on difficult tasks, it partially captures real-world readiness. In this paper, we argue and verify that robustness, defined as consistency across semantically equivalent inputs, closely follows task difficulty: once models master a task, robustness emerges naturally. Through an empirical analysis of multiple models across diverse datasets and configurations (e.g., paraphrases, temperature changes), we observe a strong positive correlation between task performance and robustness. Furthermore, our findings indicate that robustness is driven primarily by task-specific competence rather than inherent model attributes, challenging the common view of robustness as an independent capability. This perspective implies that as tasks mature and model performance saturates, robustness on those tasks will similarly emer
    
[^254]: 元学习辅助的约束松弛方法用于约束黑盒优化

    Meta-Learning-Assisted Constraint Relaxation for Constrained Black-Box Optimization

    [https://arxiv.org/abs/2602.00532](https://arxiv.org/abs/2602.00532)

    本文提出MeCO，一种通过元学习训练双重深度Q网络控制器来学习自适应ε-约束松弛策略的约束黑盒优化方法，无需针对具体问题调参即可泛化到新的优化问题。

    

    arXiv:2602.00532v2 公告类型：replace-cross。摘要：约束处理是约束黑盒优化（BBO）的核心问题，其中目标改进与可行性恢复往往会提供相互冲突的搜索信号。现有的ε-松弛方法简单有效，但其松弛调度通常是固定的，或仅针对有限范围的问题手工设计。为解决这一局限，本快报提出MeCO，一种元学习辅助的优化器，用于为约束黑盒优化学习自适应的ε-松弛策略。MeCO将SHADE优化器与双重深度Q网络（Double Deep Q-Network）控制器相耦合。在每个优化步骤中，控制器观测紧凑的种群与约束特征，并选择一个标量动作，该动作被解码为用于候选解比较规则的松弛向量。该策略在多个约束黑盒优化实例上进行训练，随后可直接部署于未见过的测试问题，无需针对具体问题进行调参。在CEC2017约束优化基准上的实验……

    arXiv:2602.00532v2 Announce Type: replace-cross  Abstract: Constraint handling is central to constrained black-box optimization (BBO), where objective improvement and feasibility restoration often provide conflicting search signals. Existing $\epsilon$-relaxation methods are simple and effective, but their relaxation schedules are usually fixed or manually designed for a limited range of problems. To address this limitation, this letter proposes MeCO, a meta-learning-assisted optimizer that learns an adaptive $\epsilon$-relaxation policy for constrained BBO. MeCO couples a SHADE optimizer with a Double Deep Q-Network controller. At each optimization step, the controller observes compact population and constraint features and selects a scalar action, which is decoded into a relaxation vector for the candidate comparison rule. The policy is trained across constrained BBO instances and then deployed on held-out problems without problem-specific tuning. Experiments on the CEC2017 constrain
    
[^255]: Window-Diffusion：基于窗口化标记剪枝与缓存的扩散语言模型推理加速方法

    Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching

    [https://arxiv.org/abs/2601.20332](https://arxiv.org/abs/2601.20332)

    该论文通过分析发现扩散语言模型推理具有显著的结构局部性，并提出无需重新训练的Window-Diffusion方法，利用窗口化标记剪枝与缓存机制来显著加速预训练扩散语言模型的推理。

    

    扩散语言模型（DLMs）通过迭代去噪的方式生成文本，但推理过程在每次迭代中都需要进行全序列注意力计算，导致在掩码标记上产生大量冗余计算。基于块的扩散方法虽然可以降低这种开销，但通常依赖重新训练并受限于特定的更新顺序，这限制了其直接应用于预训练扩散语言模型的能力。我们的标记级分析揭示了扩散语言模型推理中显著的结构局部性：解码过程由一小部分位于前缀附近的活跃标记所驱动；远处未解码上下文的影响会迅速减弱；已解码的标记表现出分阶段的时间稳定性，除了解码后短暂的瞬态时期外，其中间表示可以被复用。基于这些观察，我们提出了Window-Diffusion，一种面向推理的基于窗口的标记剪枝与缓存加速方法（源代码见 https://github.com/vhicrgit/Window-Diffusion）……

    arXiv:2601.20332v3 Announce Type: replace  Abstract: Diffusion language models (DLMs) generate text through iterative denoising, but inference requires full-sequence attention at every iteration, resulting in substantial redundant computation on masked tokens. Block-wise diffusion can reduce this cost, yet it typically relies on retraining and constrained update orders, limiting its direct applicability to pretrained DLMs. Our token-level analysis reveals pronounced structural locality in DLM inference. Decoding is driven by a small set of prefix-localized active tokens; the influence of distant undecoded context diminishes rapidly, and decoded tokens exhibit stage-wise temporal stability, enabling reuse of intermediate representations except for a brief post-decode transient. Motivated by these observations, we propose \textbf{\placeholder}\footnote{The source code is available at https://github.com/vhicrgit/Window-Diffusion.}, a window-based token pruning and caching method for infer
    
[^256]: AllShowers：适用于所有量能器簇射的单一模型

    AllShowers: One model for all calorimeter showers

    [https://arxiv.org/abs/2601.11716](https://arxiv.org/abs/2601.11716)

    AllShowers提出了一种统一的生成模型，利用基于Transformer架构的连续归一化流，仅需单一模型即可在宽能量范围内模拟电子、光子及带电和中性强子等多种粒子类型的量能器簇射，取代了传统上为每种粒子单独训练网络的方法。

    

    精确而高效的探测器模拟对于现代对撞机实验至关重要。为了降低高昂的计算成本，已有多种快速的机器学习代理模型被提出。传统的量能器簇射建模代理模型需要为每种粒子类型训练单独的网络，这限制了可扩展性和复用性。我们提出了AllShowers，这是一个统一的生成模型，能够使用单一的生成模型模拟多种粒子类型的量能器簇射。AllShowers是一个采用Transformer架构的连续归一化流模型，使其能够在变长点云表示的簇射中生成复杂的空间和能量关联。该模型在高度颗粒化的ILD探测器中模拟的多样化簇射数据集上进行了训练，展示了在宽入射能量范围内为电子、光子以及带电和中性强子生成逼真簇射的能力。

    arXiv:2601.11716v2 Announce Type: replace-cross  Abstract: Accurate and efficient detector simulation is essential for modern collider experiments. To reduce the high computational cost, various fast machine learning surrogate models have been proposed. Traditional surrogate models for calorimeter shower modeling train separate networks for each particle species, limiting scalability and reuse. We introduce AllShowers, a unified generative model that simulates calorimeter showers across multiple particle types using a single generative model. AllShowers is a continuous normalizing flow model with a Transformer architecture, enabling it to generate complex spatial and energy correlations in variable-length point cloud representations of showers. Trained on a diverse dataset of simulated showers in the highly granular ILD detector, the model demonstrates the ability to generate realistic showers for electrons, photons, and charged and neutral hadrons across a wide range of incident energ
    
[^257]: 多类别不平衡学习的协同优化：密度感知与区域引导的Boosting

    Collaborative Optimization of Multiclass Imbalanced Learning: Density-Aware and Region-Guided Boosting

    [https://arxiv.org/abs/2512.22478](https://arxiv.org/abs/2512.22478)

    该研究提出一种密度感知与区域引导的协同优化Boosting模型，通过抗噪声权重更新机制和动态采样策略实现多类别不平衡学习与模型训练的紧密协同，在40个公开数据集上显著提升性能。

    

    大量关于Boosting的研究试图缓解由类别不平衡引起的分类偏差。然而，现有研究尚未探索不平衡学习与模型训练的协同优化。这一局限阻碍了性能的进一步提升。为弥补这一空白，本研究提出了一种面向多类别不平衡学习的协同优化Boosting模型。通过融合密度因子和置信度因子，该模型实现了抗噪声的权重更新机制和动态采样策略。这些模块并非作为独立组件运行，而是紧密集成，协同调度权重更新、样本区域划分和区域引导采样。因此，本研究提出了不平衡学习与模型训练的协同优化。在40个公开不平衡数据集上的大量实验表明，所提出的模型显著优于现有方法。

    arXiv:2512.22478v2 Announce Type: replace  Abstract: Numerous studies on Boosting attempt to mitigate classification bias caused by class imbalance. However, existing studies have yet to explore the collaborative optimization of imbalanced learning and model training. This constraint hinders further performance improvements. To bridge this gap, this study proposes a collaborative optimization Boosting model of multiclass imbalanced learning. By integrating the density factor and the confidence factor, this model implements a noise-resistant weight update mechanism alongside a dynamic sampling strategy. Rather than functioning as independent components, these modules are tightly integrated to orchestrate weight updates, sample region partitioning, and region-guided sampling. Thus, this study proposes the collaborative optimization of imbalanced learning and model training. Extensive experiments on 40 public imbalanced datasets demonstrate that the proposed model significantly outperform
    
[^258]: 非负矩阵分解及相关成分模型：等价性、可辨识性及其在沉积物粒度分析中的应用

    Nonnegative matrix factorizations and related compositional models: Equivalence, identifiability, and an application on the grain-size analysis of sediments

    [https://arxiv.org/abs/2512.22282](https://arxiv.org/abs/2512.22282)

    本文证明了来自社会科学、地质学和机器学习的五种模型（LBA、LCA、EMA、PLSA、NMF）在本质上等价，NMF的解唯一性定理可直接推广到其他四种模型，并将其应用于沉积物粒度分析。

    

    在机器学习、社会科学和地质学等领域中，将非负矩阵分解为两个或三个矩阵乘积的模型受到了广泛关注，这些模型受非负约束或行和为1的约束。尽管这些模型在很大程度上相似甚至等价，但它们以不同的名称呈现，其相似性并不为人所熟知。本文重点阐述了五种模型之间的相似性，包括来自社会科学的潜在预算分析（LBA）和潜在类别分析（LCA）、来自地质学的端元分析（EMA），以及来自机器学习的概率潜在语义分析（PLSA）和非负矩阵分解（NMF）。我们聚焦于这些模型的可辨识性，证明了LBA、EMA、LCA、PLSA的解是唯一的当且仅当NMF的解是唯一的。因此，NMF现有的唯一性定理可直接应用于LBA、EMA、LCA、PLSA，反之亦然。

    arXiv:2512.22282v2 Announce Type: replace-cross  Abstract: Across fields such as machine learning, social science, and geology, considerable attention has been given to models that factorize a nonnegative matrix into the product of two or three matrices, subject to nonnegative or row-sum-to-1 constraints. Although these models are to a large extent similar or even equivalent, they are presented under different names, and their similarity is not well known. This paper highlights similarities among five models, latent budget analysis (LBA) and latent class analysis (LCA) from social science, end-member analysis (EMA) from geology, probabilistic latent semantic analysis (PLSA) and nonnegative matrix factorization (NMF) from machine learning. We focus on the identifiability of these models. We prove that the solution of LBA, EMA, LCA, PLSA is unique if and only if the solution of NMF is unique. Consequently, existing uniqueness theorems for NMF directly apply to LBA, EMA, LCA, PLSA, and vi
    
[^259]: 基于微调大语言模型的离子阱量子计算机输运编译器

    Shuttling Compiler for Trapped-Ion Quantum Computers Based on Fine-Tuned Large Language Models

    [https://arxiv.org/abs/2512.18021](https://arxiv.org/abs/2512.18021)

    本文创新性地提出基于微调大语言模型的离子阱量子计算机输运编译器，可自动生成输运调度并在部分情况下比传统启发式方法减少多达21%的操作次数，同时展现出对新型阱架构的泛化潜力。

    

    在离子阱量子计算机中，量子比特必须在不同区段之间输运才能进行相互作用。调度这些移动的路由逻辑需要针对每种新型阱架构手工编写。我们提出了基于五个大语言模型（LLM）的输运编译器。每个大语言模型均在手工编码的启发式方法为线性和分支一维阱架构生成的输运调度上进行微调。我们研究了这些模型生成的调度的输运操作次数与启发式方法的对比情况，以及它们对未见过的架构的泛化能力。对于最多16个量子比特的电路，微调后的大语言模型在两种训练架构上均能生成有效调度，且电路的量子比特数越少，生成有效调度的频率越高。在产出调度的编译案例中，有12%在经过基于规则的后处理步骤后，十次运行中的最佳结果比启发式基线少需要多达21%的操作。单次运行一个微调的大语言模型即可产生有效的……

    arXiv:2512.18021v4 Announce Type: replace-cross  Abstract: In trapped-ion quantum computers, qubits must be shuttled between segments to interact. The routing logic that schedules these movements is written by hand for every new trap architecture. We present shuttling compilers based on five large language models (LLMs). Each LLM is fine-tuned on shuttling schedules produced by hand-coded heuristics for linear and branched one-dimensional trap architectures. We investigate how the shuttling operation counts of their schedules compare with those of the heuristics and how far they generalize to unseen architectures. For circuits of up to 16 qubits, the fine-tuned LLMs generate valid schedules on both training architectures, more often the fewer qubits a circuit has. In 12% of the compilations yielding a schedule, the best of ten runs needs up to 21% fewer operations than the heuristic baselines, after a rule-based post-processing step. A single run of one fine-tuned LLM produces a valid 
    
[^260]: 形式化的霍普菲尔德网络与玻尔兹曼机

    Formalized Hopfield Networks and Boltzmann Machines

    [https://arxiv.org/abs/2512.07766](https://arxiv.org/abs/2512.07766)

    本文在 Lean 4 中首次形式化了霍普菲尔德网络与玻尔兹曼机，证明了确定性网络的收敛性与赫布学习的正确性，并借助佩龙-弗罗贝尼乌斯定理的新形式化证明了随机网络的遍历性。

    

    神经网络被广泛使用，但其分析与验证仍然具有挑战性。我们提出了一个涵盖确定性模型和随机模型的 Lean 4 形式化。我们首先形式化了霍普菲尔德网络——一种将模式存储为稳定状态的循环网络——并证明了其收敛性以及赫布学习规则的正确性，该规则通过更新参数来编码模式。随后我们转向随机网络，其概率更新会收敛到一个平稳分布：我们形式化了玻尔兹曼机的动力学与学习过程，并通过佩龙-弗罗贝尼乌斯定理的新形式化证明了其遍历性——即收敛到一个唯一的平稳分布。

    arXiv:2512.07766v2 Announce Type: replace  Abstract: Neural networks are widely used, yet their analysis and verification remain challenging. We present a Lean~4 formalization covering both deterministic and stochastic models. We first formalize Hopfield networks -- recurrent networks that store patterns as stable states -- and prove their convergence, and the correctness of Hebbian learning, the rule that updates parameters to encode patterns. We then turn to stochastic networks, whose probabilistic updates converge to a stationary distribution: we formalize the dynamics and learning of Boltzmann machines and prove their ergodicity -- convergence to a \emph{unique} stationary distribution -- via a new formalization of the Perron--Frobenius theorem.
    
[^261]: 使用非MCMC采样器与高效温度估计训练能量基模型

    Training Energy-Based Models with Non-MCMC Samplers and Efficient Temperature Estimation

    [https://arxiv.org/abs/2512.02323](https://arxiv.org/abs/2512.02323)

    该论文提出了朗之万模拟分岔（LSB）快速并行玻尔兹曼采样器、条件期望匹配（CEM）高效温度估计方法以及采样器自适应学习（SAL）框架，从而无需MCMC即可高效训练能量基模型。

    

    从离散变量上的玻尔兹曼分布中高效采样是众多应用领域中的基础操作。虽然快速的非MCMC采样器近来已成为传统MCMC方法的有前景的替代方案，但由于难以估计生成样本的有效温度，它们在概率学习中的实际应用仍受到阻碍。在本工作中，我们首先介绍了朗之万模拟分岔（Langevin simulated bifurcation, LSB），这是一种玻尔兹曼采样器，能够实现快速并行采样，其精度可与顺序MCMC方法相媲美。为解决未知有效温度的难题，我们提出了条件期望匹配（conditional expectation matching, CEM），这是一种高效的估计方法，适用于具有可利用条件独立结构的能量基模型（EBM）。基于这些组件，我们进一步开发了一个名为采样器自适应学习（sampler adaptive learning, SAL）的学习框架，该框架能够自适应地调整……

    arXiv:2512.02323v2 Announce Type: replace  Abstract: Efficient sampling from Boltzmann distributions over discrete variables is a fundamental operation in a wide range of applications. While fast non-MCMC samplers have recently emerged as promising alternatives to conventional MCMC methods, their practical use for probabilistic learning remains hindered by the difficulty of estimating the effective temperature of the generated samples. In this work, we begin by introducing Langevin simulated bifurcation (LSB), a Boltzmann sampler that enables fast and parallel sampling with accuracy comparable to sequential MCMC methods. To address the challenge of unknown effective temperature, we propose conditional expectation matching (CEM), an efficient estimation method applicable to energy-based models (EBMs) with exploitable conditional independence structures. Building on these components, we further develop a learning framework, termed sampler adaptive learning (SAL), which adaptively adjusts
    
[^262]: 双重随机平滑：突破全局噪声方差的限制

    Dual Randomized Smoothing: Beyond Global Noise Variance

    [https://arxiv.org/abs/2512.01782](https://arxiv.org/abs/2512.01782)

    提出双重随机平滑框架，证明了只要噪声方差在每个输入周围局部恒定时RS仍然有效，并通过引入方差估计器实现输入依赖的噪声方差，从而突破了标准RS中全局噪声方差无法同时在小半径和大半径处取得优异性能的根本限制。

    

    随机平滑（Randomized Smoothing, RS）是认证神经网络对抗对抗性扰动鲁棒性的一项重要技术。在RS中，要在小半径处获得高准确率需要较小的噪声方差，而要在大半径处获得高准确率则需要较大的噪声方差。然而，标准RS公式中使用的全局噪声方差导致了一个根本性的限制：不存在能够同时在小半径和大半径处都取得优异性能的全局噪声方差。为了突破全局方差的限制，我们提出了一个双重RS框架，使噪声方差能够依赖于输入。为实现这一目标，我们首先证明了只要方差在每个输入周围局部恒定，RS在使用输入依赖的噪声方差时仍然有效。基于这一结果，我们引入了两个组件：其一，方差估计器为每个输入预测最优的噪声方差；其二，该估计...

    arXiv:2512.01782v4 Announce Type: replace-cross  Abstract: Randomized Smoothing (RS) is a prominent technique for certifying the robustness of neural networks against adversarial perturbations. With RS, achieving high accuracy at small radii requires a small noise variance, while achieving high accuracy at large radii requires a large noise variance. However, the global noise variance used in the standard RS formulation leads to a fundamental limitation: there exists no global noise variance that simultaneously achieves strong performance at both small and large radii. To break through the global variance limitation, we propose a dual RS framework which enables input-dependent noise variances. To achieve that, we first prove that RS remains valid with input-dependent noise variances, provided the variance is locally constant around each input. Building on this result, we introduce two components: (i) a variance estimator predicts an optimal noise variance for each input, (ii) this esti
    
[^263]: GeoCrossBench：面向遥感的跨波段泛化

    GeoCrossBench: Cross-Band Generalization for Remote Sensing

    [https://arxiv.org/abs/2511.02831](https://arxiv.org/abs/2511.02831)

    本文提出GeoCrossBench基准和χViT基线模型，通过新的跨波段泛化评估协议解决遥感领域新旧卫星波段不一致的问题，降低支持新卫星所需的模型重训练成本。

    

    遥感数据在不断获取中，新数据来自数量和种类日益增多的卫星，而绝大多数有标注的数据却来自较旧的卫星。随着面向地球观测的遥感基础模型规模不断扩大，为支持新卫星而（重新）训练的成本也随之增长，因此跨传感器和卫星的跨波段泛化能力变得愈发重要。我们提出了GeoCrossBench，这是对广受欢迎的GeoBench基准的扩展，引入了一套针对跨传感器和卫星跨波段泛化的新评估协议：它测试使用相同波段进行训练和测试的标准分布内性能、训练与测试波段无交集情况下的泛化能力，以及测试输入包含训练波段超集情况下的泛化能力。我们开发了χViT，这是波段无关的ChannelViT的自监督扩展版本，作为跨波段泛化的支持性基线模型。

    arXiv:2511.02831v2 Announce Type: replace  Abstract: The data for remote sensing is constantly acquired, and new data comes from a growing number and diversity of satellites, while the vast majority of labeled data comes from older satellites. As remote-sensing foundation models for Earth observation scale up, the cost of (re-)training to support new satellites grows too, so cross-band generalization across sensors and satellites is increasingly important. We introduce GeoCrossBench, an extension of the popular GeoBench benchmark with a new evaluation protocol for cross-band generalization across sensors and satellites: it tests standard in-distribution performance with the same bands for train and test, generalization to inputs with no intersection between train and test; and generalization to test inputs containing a superset of the training bands. We develop $\chi$ViT, a self-supervised extension of the band-agnostic ChannelViT, as a supporting baseline for cross-band generalization
    
[^264]: TARC：时间自适应机器人控制

    TARC: Time-Adaptive Robotic Control

    [https://arxiv.org/abs/2510.23176](https://arxiv.org/abs/2510.23176)

    提出 TARC 强化学习框架，让策略同时预测控制动作及其持续时间，实现控制频率的自适应调节，在高速遥控车、宇树 Go1 四足机器人等真实硬件以及视觉-语言-动作模型上，均能在保持任务性能的同时减少控制切换和推理开销。

    

    大多数机器人系统依赖于固定频率的离散时间控制器，这在低频控制的高效性与高频反馈的响应性之间造成了权衡。因此，系统通常为了鲁棒性而默认采用高控制频率，代价是浪费的推理计算和不必要的执行动作。为解决这一问题，我们提出了时间自适应机器人控制，这是一个强化学习框架，其中策略同时预测控制动作及其作用持续时间。TARC 通过在控制切换次数的软约束或硬约束下优化任务性能来学习时间上扩展的动作，从而实现控制频率的自适应调节。我们在两个机器人硬件平台上评估了 TARC：高速遥控赛车和宇树 Go1 四足机器人，并在仿真环境中评估了一个视觉-语言-动作模型，其中每次查询都需要进行代价高昂的 Transformer 前向传播。在所有设置中，TARC……（摘要在此处被截断）

    arXiv:2510.23176v2 Announce Type: replace-cross  Abstract: Most robotic systems rely on fixed-frequency discrete-time controllers, creating a trade-off between the efficiency of low-frequency control and the responsiveness of high-frequency feedback. As a result, systems typically default to high control rates for robustness, at the cost of wasted inference and unnecessary actuation. Addressing this, we introduce Time-Adaptive Robotic Control (TARC), a reinforcement learning framework in which the policy jointly predicts a control action and its duration of application. TARC learns temporally extended actions by optimizing task performance under soft or hard constraints on the number of control switches, enabling adaptive modulation of control rates. We evaluate TARC on two robotic hardware platforms: a high-speed RC car and the Unitree Go1 quadruped, and on a vision-language action model in simulation, where each query incurs a costly transformer forward pass. Across all settings, TAR
    
[^265]: 基于SRE对齐决策的风险校准贝叶斯流式入侵检测

    Risk-Calibrated Bayesian Streaming Intrusion Detection with SRE-Aligned Decisions

    [https://arxiv.org/abs/2510.09619](https://arxiv.org/abs/2510.09619)

    本文提出将贝叶斯在线变点检测（BOCPD）与SRE错误预算对齐的决策阈值相结合，通过在假阳性和假阴性预算下优化期望运营成本，实现能适应分布漂移的风险校准流式入侵检测方法。

    

    arXiv:2510.09619v2 通告类型： replace-cross 摘要： [更正版v2：经审计发现，下文所述的评分、阈值和延迟描述与共享代码库的实际实现不符，且评估数据流为人工组装构造。详见首页的更正说明及更正后的伴随工作 arXiv:2605.24696（更正版v3），工件 doi:10.5281/zenodo.22673735。] 我们提出了一种风险校准的流式入侵检测方法，该方法将贝叶斯在线变点检测（BOCPD）与符合站点可靠性工程（SRE）错误预算的决策阈值相结合。BOCPD提供能够适应分布偏移和概念漂移的运行长度后验分布；我们通过在假阳性与假阴性预算约束下优化期望运营成本，将这些后验分布映射为告警决策。我们详细阐述了风险模型、共轭更新以及每事件O(1)复杂度的实现方案。一个具体的SRE示例展示了99.9%可用性SLO（每月43.2分钟……[摘要在此处截断]

    arXiv:2510.09619v2 Announce Type: replace-cross  Abstract: [Corrected v2: an audit found that the score, threshold, and latency descriptions below are not what the shared codebase implements, and that the evaluation streams are assembled constructions. See the correction note on the title page and the corrected companion work, arXiv:2605.24696 (corrected v3), artifact doi:10.5281/zenodo.22673735.] We present a risk-calibrated approach to streaming intrusion detection that couples Bayesian Online Changepoint Detection (BOCPD) with decision thresholds aligned to Site Reliability Engineering (SRE) error budgets. BOCPD provides run-length posteriors that adapt to distribution shift and concept drift; we map these posteriors to alert decisions by optimizing expected operational cost under false-positive and false-negative budgets. We detail the hazard model, conjugate updates, and an O(1)-per-event implementation. A concrete SRE example shows how a 99.9% availability SLO (43.2 minutes per m
    
[^266]: GraphIFE：基于不变学习重新思考图不平衡节点分类问题

    GraphIFE: Rethinking Graph Imbalance Node Classification via Invariant Learning

    [https://arxiv.org/abs/2509.23616](https://arxiv.org/abs/2509.23616)

    提出了基于图不变学习的GraphIFE框架，通过缓解合成节点的质量不一致问题来应对图数据中的类别不平衡，从而提升少数类别的节点分类性能。

    

    类别不平衡问题是指数据集中不同类别之间样本分布不成比例，其中少数类别样本严重不足。这一问题在图结构数据中同样普遍存在。大多数图神经网络（GNN）隐式地假设类别分布是平衡的，因此往往未能应对类别不平衡所带来的挑战，这可能导致学习产生偏差，并使少数类别上的性能下降。我们识别出合成节点中的质量不一致问题，这一问题导致图不平衡条件下的性能欠佳。为缓解该问题，我们提出了GraphIFE（图不变特征提取），这是一个旨在缓解合成节点质量不一致问题的新型框架。我们的方法融合了图不变学习中的两个关键概念，并引入了增强嵌入空间表示的策略。

    arXiv:2509.23616v2 Announce Type: replace-cross  Abstract: The class imbalance problem refers to the disproportionate distribution of samples across different classes within a dataset, where the minority classes are significantly underrepresented. This issue is also prevalent in graph-structured data. Most graph neural networks (GNNs) implicitly assume a balanced class distribution and therefore often fail to account for the challenges introduced by class imbalance, which can lead to biased learning and degraded performance on minority classes. We identify a quality inconsistency problem in synthesized nodes, which leads to suboptimal performance under graph imbalance conditions. To mitigate this issue, we propose GraphIFE (Graph Invariant Feature Extraction), a novel framework designed to mitigate quality inconsistency in synthesized nodes. Our approach incorporates two key concepts from graph invariant learning and introduces strategies to strengthen the embedding space representatio
    
[^267]: 基于人口普查与土地利用数据的大语言模型个人出行日记生成方法

    Generating Individual Travel Diaries Using Large Language Models Informed by Census and Land-Use Data

    [https://arxiv.org/abs/2509.09710](https://arxiv.org/abs/2509.09710)

    该研究提出一种基于开源人口普查和土地利用数据、利用大语言模型随机生成个人出行日记的新方法，并通过包含四项指标的新颖真实感评分和Jensen-Shannon散度验证其有效性。

    

    本研究提出了一种大语言模型（LLM）方案，用于在基于智能体的交通模型中生成出行日记的关键属性，包括出行目的、出行方式和出行距离，以评估大语言模型在活动生成任务中的潜在可行性。传统方法依赖于大量专有的家庭出行调查数据，而我们的方法从开源的美国社区调查（ACS）数据和智能位置数据库（SLD）数据中随机生成人物画像，然后通过直接提示词合成出行日记。本研究的一个创新点是提出了一对一群组（one-to-cohort）真实感评分：这是一个由四个指标（出行次数评分、时间间隔评分、出行目的评分和出行方式评分）组成的综合指标，并在人口统计变量匹配的基础上，通过与康涅狄格州全州交通研究（CSTS）的出行日记进行对比验证。我们的验证过程利用Jensen-Shannon散度来衡量生成日记与真实日记之间的分布相似性。

    arXiv:2509.09710v3 Announce Type: replace-cross  Abstract: This study introduces a Large Language Model (LLM) scheme for generating key attributes of travel diaries in agent-based transportation models, including purpose, mode and distance, to assess the underlying viability of LLMs for activity generation tasks. While traditional approaches rely on large quantities of proprietary household travel surveys, our method generates personas stochastically from open-source American Community Survey (ACS) and Smart Location Database (SLD) data, then synthesizes diaries through direct prompting. Our study features a novel one-to-cohort realism score: a composite of four metrics (Trip Count Score, Interval Score, Purpose Score, and Mode Score) validated against the Connecticut Statewide Transportation Study (CSTS) diaries, matched across demographic variables. Our validation utilizes Jensen-Shannon Divergence to measure distributional similarities between generated and real diaries. When compar
    
[^268]: 紧凑状态空间上的神经随机微分方程：理论、方法及其在自杀风险建模中的应用

    Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods, and Application to Suicide Risk Modeling

    [https://arxiv.org/abs/2508.17090](https://arxiv.org/abs/2508.17090)

    本文提出了一类新型神经随机微分方程，其解可被严格证明限制在指定的紧凑多面体状态空间内，克服了现有SDE模型违反定义域约束和数值不稳定的问题，并成功应用于自杀风险建模。

    

    生态瞬间评估（EMA）研究使得通过智能手机收集关于自杀想法和行为（STBs）的高频自我报告成为可能。潜在随机微分方程（SDEs）是建模EMA数据的一个有前景的模型类别，因为这类数据采样不规则、含噪声且部分可观测。但基于SDE的模型存在两个关键局限性：(a) 这些模型经常违反定义域约束，损害了模型的科学有效性和临床可信度；(b) 若不采用临时修复手段（如过度简化的动力学），训练在数值上是不稳定的，而这些修复手段并不适合高风险的应用场景。在本文中，我们开发了一类新颖的、具有强表达能力的SDE，其解可被严格证明被限制在规定的紧凑多面体状态空间内，从而与EMA数据的定义域相匹配。在这项工作中，（1）我们从理论和实证上展示了为什么基于链式法则在紧凑域上构建SDE的方法会失败；（2）我们推导了（摘要在此处被截断）……

    arXiv:2508.17090v5 Announce Type: replace-cross  Abstract: Ecological Momentary Assessment (EMA) studies enable the collection of high-frequency self-reports of suicidal thoughts and behaviors (STBs) via smartphones. Latent stochastic differential equations (SDEs) are a promising model class for EMA data, as it is irregularly sampled, noisy, and partially observed. But SDE-based models suffer from two key limitations. (a) These models often violate domain constraints, undermining scientific validity and clinical trust of the model. (b) Training is numerically unstable without ad hoc fixes (e.g. oversimplified dynamics) that are ill-suited for high-stakes applications. Here, we develop a novel class of expressive SDEs whose solutions are provably confined to a prescribed compact polyhedral state space, matching the domains of EMA data. In this work, (1) we show why chain-rule based constructions of SDEs on compact domains fail, theoretically and empirically; (2) we derive constraints on
    
[^269]: 文字碎片化与格式：是什么导致了开源大语言模型中英语与孟加拉语的性能差距？

    Script Fragmentation and Format: What Drives the English-Bengali Performance Gap in Open LLMs?

    [https://arxiv.org/abs/2507.23248](https://arxiv.org/abs/2507.23248)

    该论文发布了8个翻译成孟加拉语的英文基准测试并评估10个开源LLM，揭示英语与孟加拉语性能差距部分源于子词分词器对孟加拉语字素簇的文字碎片化以及精确匹配评分所强制的答案格式，并证明其中一部分差距实际上是测量伪影而非模型真实能力的缺陷。

    

    孟加拉语的使用者超过2.3亿人，然而目前尚无标准化的工具能在用于评估前沿模型的任务类别上评估大语言模型（LLM）的孟加拉语能力。我们发布了8个通过统一一致的流程翻译成孟加拉语的英文基准测试，并用它们在成对的英语和孟加拉语输入上评估了来自4个模型家族的10个开源LLM。文字碎片化是子词分词器对孟加拉语的元音附标文字（alphasyllabary）所造成的现象——其书写单位是跨越多个Unicode码点的字素簇：分词器将文字切分成比字符更小的碎片，其代价由词表大小决定，而非文字本身。格式则属于评估环节，是精确匹配评分所要求的答案形式，而与模型是否真正知道答案无关。除了确认存在显著差距（宏观LLM评判得分：英语为0.79，孟加拉语为0.63）之外，我们还表明其中一部分差距是测量伪影：精确...（摘要原文在此处截断）

    arXiv:2507.23248v2 Announce Type: replace  Abstract: Bengali is spoken by more than 230 million people, yet no standardized instrument evaluates large language models (LLMs) on Bengali across the task categories used to benchmark frontier models. We release 8 English benchmarks translated into Bengali with a single consistent pipeline and use them to evaluate 10 open LLMs from 4 families on paired English and Bengali inputs. Script fragmentation is what subword tokenizers do to Bengali's alphasyllabary, whose written units are grapheme clusters spanning several Unicode code points: they cut the script into pieces smaller than a character, at a cost set by the vocabulary rather than the script itself. Format belongs to the evaluation, the answer shape that exact-match scoring demands regardless of whether the model knew the answer. Beyond confirming a substantial gap (macro LLM-judge score 0.79 in English versus 0.63 in Bengali), we show that part of it is a measurement artifact: exact-
    
[^270]: 观测多重性

    Observational Multiplicity

    [https://arxiv.org/abs/2507.23136](https://arxiv.org/abs/2507.23136)

    该论文提出了“观测多重性”概念，指出当模型以二元观测标签训练来预测概率时会产生预测的任意性，并引入一种基于“遗憾”度量的通用方法来量化模型概率预测因训练标签不同而可能产生的变化程度。

    

    许多预测任务可能存在多个表现几乎同样好的模型。当相互竞争的模型对同一个体给出相互冲突的预测时，这种现象可能会损害可解释性和安全性。在这项工作中，我们研究了概率分类任务中由于一种我们称之为“观测多重性”的效应而产生的任意性问题。我们讨论了这种效应如何在一大类实际应用中出现：在这些应用中，我们训练分类器来预测概率 $p_i \in [0,1]$，但所获得的数据集只包含观测值 $y_i \in \{0,1\}$。我们提出通过“遗憾”的视角来评估个体概率预测的任意性。我们引入了一种针对概率分类任务的遗憾度量，用于衡量模型的预测可能如何因不同的训练标签而发生变化。我们提出了一种通用方法来估计概率分类中的遗憾。

    arXiv:2507.23136v2 Announce Type: replace  Abstract: Many prediction tasks can admit multiple models that can perform almost equally well. This phenomenon can undermine interpretability and safety when competing models assign conflicting predictions to individuals. In this work, we study how arbitrariness can arise in probabilistic classification tasks as a result of an effect that we call \emph{observational multiplicity}. We discuss how this effect arises in a broad class of practical applications where we learn a classifier to predict probabilities $p_i \in [0,1]$ but are given a dataset of observations $y_i \in \{0,1\}$. We propose to evaluate the arbitrariness of individual probability predictions through the lens of \emph{regret}. We introduce a measure of regret for probabilistic classification tasks, which measures how the predictions of a model could change as a result of different training labels. We present a general-purpose method to estimate the regret in a probabilistic c
    
[^271]: R3：稳健的与评分标准无关的奖励模型

    R3: Robust Rubric-Agnostic Reward Models

    [https://arxiv.org/abs/2505.13388](https://arxiv.org/abs/2505.13388)

    R3是一种不受评分标准限制、具备泛化能力和可解释性的奖励建模框架，通过提供有理有据的评分实现更透明灵活的语言模型评估与人类偏好对齐。

    

    奖励模型对于使语言模型的输出与人类偏好保持一致至关重要，然而现有方法往往既缺乏可控性又缺乏可解释性。这些模型通常针对狭窄的目标进行优化，限制了它们向更广泛下游任务的泛化能力。此外，其标量输出在缺乏上下文推理的情况下难以解释。为了解决这些局限性，我们提出了R3，这是一种新颖的奖励建模框架，它不受评分标准限制、可在多个评估维度间泛化，并能提供可解释、有理有据的评分结果。R3使语言模型的评估更加透明和灵活，支持与多样化的人类价值观和使用场景进行稳健对齐。我们的模型、数据和代码已在 https://github.com/rubricreward/r3 开源。

    arXiv:2505.13388v4 Announce Type: replace-cross  Abstract: Reward models are essential for aligning language model outputs with human preferences, yet existing approaches often lack both controllability and interpretability. These models are typically optimized for narrow objectives, limiting their generalizability to broader downstream tasks. Moreover, their scalar outputs are difficult to interpret without contextual reasoning. To address these limitations, we introduce R3, a novel reward modeling framework that is rubric-agnostic, generalizable across evaluation dimensions, and provides interpretable, reasoned score assignments. R3 enables more transparent and flexible evaluation of language models, supporting robust alignment with diverse human values and use cases. Our models, data, and code are available as open source at https://github.com/rubricreward/r3.
    
[^272]: 当多数主导、少数失败：梯度下降的偏差放大

    When majority rules, minority loses: bias amplification of gradient descent

    [https://arxiv.org/abs/2505.13122](https://arxiv.org/abs/2505.13122)

    该论文建立了多数-少数学习任务的形式化理论框架，证明了在人口和方差不平衡条件下标准梯度下降训练会放大偏差、产生忽视少数群体特征的刻板预测器，并给出了消除这种偏差所需额外训练量的理论下界。

    

    尽管机器学习中偏差放大的实证证据日益增多，但其理论基础仍鲜为人知。我们为多数-少数学习任务开发了一个形式化框架，展示了标准训练如何偏向多数群体，并产生忽视少数群体特有特征的刻板预测器。在假设人口和方差不平衡的条件下，我们的分析揭示了三个关键发现： “全数据”预测器与刻板预测器之间的紧密接近性； 训练整个模型往往仅学习多数群体特征的区域的支配地位； 所需额外训练量的下界。我们的结果通过深度学习在表格和图像分类任务中的实验得到了说明。

    arXiv:2505.13122v3 Announce Type: replace-cross  Abstract: Despite growing empirical evidence of bias amplification in machine learning, its theoretical foundations remain poorly understood. We develop a formal framework for majority-minority learning tasks, showing how standard training can favor majority groups and produce stereotypical predictors that neglect minority-specific features. Assuming population and variance imbalance, our analysis reveals three key findings: (i) the close proximity between ``full-data'' and stereotypical predictors, (ii) the dominance of a region where training the entire model tends to merely learn the majority traits, and (iii) a lower bound on the additional training required. Our results are illustrated through experiments in deep learning for tabular and image classification tasks.
    
[^273]: BenSParX：一个用于从孟加拉语会话语音检测帕金森病的鲁棒可解释机器学习框架

    BenSParX: A Robust Explainable Machine Learning Framework for Parkinson's Disease Detection from Bengali Conversational Speech

    [https://arxiv.org/abs/2505.12192](https://arxiv.org/abs/2505.12192)

    该研究发布了首个用于帕金森病检测的孟加拉语会话语音数据集BenSParX，并构建了结合多样化声学特征、系统特征选择与SHAP可解释性分析的鲁棒机器学习框架，为资源受限环境下帕金森病的早期无创诊断提供了文化包容性的解决方案。

    

    帕金森病的早期检测在资源受限的环境中仍然特别具有挑战性，在这些环境中，基于语音的分析已成为一种有前景的无创且经济高效的替代方案。然而，现有研究主要集中于英语或其他主要语言；值得注意的是，目前尚不存在用于帕金森病检测的孟加拉语语音数据集——孟加拉语是全球超过2.3亿人使用的语言——这为构建具有文化包容性和可及性的医疗保健解决方案构成了重大障碍。我们提出了BenSparX，这是首个用于帕金森病检测的孟加拉语会话语音数据集，并配套提出了一个为早期诊断量身定制的鲁棒且可解释的机器学习框架。所提出的框架整合了多样化的声学特征类别、系统的特征选择方法以及经过广泛超参数优化的最先进机器学习分类器。此外，为了增强模型预测的可解释性和可信度，该框架还引入了SHAP（SHapley Additive exPlanations）方法。

    arXiv:2505.12192v2 Announce Type: replace  Abstract: Early detection of PD remains particularly challenging in resource-constrained settings, where voice-based analysis has emerged as a promising non-invasive and cost-effective alternative. However, existing studies predominantly focus on English or other major languages; notably, no voice dataset for PD exists for Bengali -- a language spoken by over 230 million people worldwide -- posing a significant barrier to culturally inclusive and accessible healthcare solutions. We present BenSparX, the first Bengali conversational speech dataset for PD detection, along with a robust and explainable ML framework tailored for early diagnosis. The proposed framework incorporates diverse acoustic feature categories, systematic feature selection methods, and state-of-the-art ML classifiers with extensive hyperparameter optimization. Furthermore, to enhance interpretability and trust in model predictions, the framework incorporates SHAP (SHapley Ad
    
[^274]: 可解释的图论机器学习及其在阿尔茨海默病预测中的应用

    Explainable Graph-theoretical Machine Learning with Application to Alzheimer's Disease Prediction

    [https://arxiv.org/abs/2503.16286](https://arxiv.org/abs/2503.16286)

    本文提出了一种可解释的图论机器学习框架XGML，通过构建个体大脑代谢图并识别最具预测性的子图，实现了基于FDG-PET数据对阿尔茨海默病多变量结果的个体化预测。

    

    痴呆症影响着全球超过5500万人，预计到2050年将达到1.39亿，其中阿尔茨海默病（AD）占病例的60-70%。阿尔茨海默病与大脑代谢连接的中断有关，早期发现这些中断对于AD的管理至关重要，而FDG-PET是识别此类损伤的有效工具。然而，大多数研究依赖于群体水平分析或阈值处理，这可能掩盖个体差异，并忽视较弱但在生物学上至关重要的大脑连接。此外，AD预测主要关注单变量而非多变量结果。为解决这一问题，我们提出了可解释图论机器学习（XGML），这是一个用于构建个体大脑代谢图并识别对多变量疾病相关结果最具预测性子图的框架。基于阿尔茨海默病神经影像学计划（ADNI）的FDG-PET数据，我们比较了六种图表示方法……

    arXiv:2503.16286v2 Announce Type: replace  Abstract: Dementia affects over 55 million people worldwide, projected to reach 139 million by 2050, with Alzheimer's disease (AD) accounting for 60-70% of cases. AD is associated with disruptions in metabolic brain connectivity. Detecting these disruptions early is crucial for AD management. FDG-PET is a useful tool for identifying such impairments. However, most studies rely on group-level analyses or thresholding, potentially masking individual differences and overlooking weaker yet biologically critical brain connections. Moreover, AD prediction largely focuses on univariate rather than multivariate outcomes. To address this, we introduce explainable graph-theoretical machine learning (XGML), a framework for constructing individual metabolic brain graphs and identifying subgraphs most predictive of multivariate disease-related outcomes. Using Alzheimer's Disease Neuroimaging Initiative (ADNI) FDG-PET data, we compared six graph representat
    
[^275]: CBW：基于聚类后门水印的说话人验证数据集所有权验证方法

    CBW: Towards Dataset Ownership Verification for Speaker Verification via Clustering-based Backdoor Watermarking

    [https://arxiv.org/abs/2503.05794](https://arxiv.org/abs/2503.05794)

    本文提出了一种基于聚类的后门水印方法（CBW），首次解决了开放集说话人验证场景下的数据集所有权验证问题，克服了现有方法依赖封闭标签空间、无法应对第三方注册身份的局限。

    

    说话人验证模型通常在大规模公开数据集上训练，而这些数据集的许可证通常禁止未经授权的商业使用，然而此类侵权行为难以检测或阻止。数据集所有权验证（DOV）是主流的应对措施：它可以通过后门攻击为数据集添加水印，使得在其上训练的模型表现出所有者指定的行为。然而，现有的DOV方法预设了一个在添加水印时就已固定的封闭标签空间，而在开集说话人验证中，部署模型所接受的身份是由第三方在模型发布后注册的，数据集所有者从未观察到这些身份。我们证明了直接套用现有方法会以两种典型方式失败，并据此提炼出有效水印应满足的三个要求，即身份不可知性、覆盖性和保真度，同时指出后两者之间存在内在矛盾。我们提出的基于聚类的后门水印方法（CBW）……（摘要内容不完整，原文在此处截断）

    arXiv:2503.05794v4 Announce Type: replace-cross  Abstract: Speaker verification models are trained on large-scale public datasets whose licenses usually prohibit unauthorized commercial use, yet such infringement is difficult to detect or deter. Dataset ownership verification (DOV) is the mainstream countermeasure: it can watermark a dataset with backdoor attacks so that models trained on it exhibit owner-specified behaviors. However, existing DOV methods presuppose a closed label space fixed at watermarking time, whereas in open-set speaker verification the identities that a deployed model accepts are enrolled by third parties after release and are never observed by the dataset owner. We show that straightforward adaptations fail in two characteristic modes, and accordingly distill three requirements for an effective watermark, namely identity agnosticism, coverage, and fidelity, together with an intrinsic tension between the latter two. Our clustering-based backdoor watermark (CBW) r
    
[^276]: 注意力即所需，直到你需要记忆留存

    Attention is All You Need Until You Need Retention

    [https://arxiv.org/abs/2501.09166](https://arxiv.org/abs/2501.09166)

    该论文提出一种保留层，使Transformer具备可在使用中读写的持久记忆，并将“决定保留什么”建模为社会学习问题，通过惊讶门控编码、可信度加权的多源共识巩固以及基于再现性的再巩固来管理记忆生命周期，且该层在记忆为空时严格退化为标准Transformer。

    

    预训练的Transformer将学到的知识保存在权重中，而一旦会话结束，其所观察到的内容便会丢失。本文第一版提出了一种“保留层”，这是一种持久化记忆，Transformer块在运行过程中通过注意力机制对其进行读取和写入。由于已部署模型所能保留的大部分信息是由其他智能体产生的，本修订版将“决定保留什么”视为一个社会学习问题：何时依赖观察到的行为、向谁学习，以及需要多少独立一致的印证。我们给出了该层的修正规范，当其记忆为空时，该层严格退化为标准Transformer。我们从社会学习策略中推导出记忆的生命周期：由惊讶程度、观察结果和习得可信度所门控的编码；通过由不同且近期的信息源组成的、经可信度加权的法定多数进行巩固，且该多数还必须胜过所有竞争行为；以及通过再现结果进行的再巩固……

    arXiv:2501.09166v2 Announce Type: replace-cross  Abstract: Pretrained Transformers keep what they learned in their weights and lose what they observe once a session ends. The first version of this paper proposed a Retention Layer, a persistent memory that a Transformer block reads with attention and writes during use. Because most of what a deployed model could retain is produced by other agents, this revision treats deciding what to keep as a social learning problem: when to rely on observed behaviour, whom to learn from and how much independent agreement to require. We give a corrected specification of the layer, which reduces exactly to the base Transformer when its memory is empty. We derive the memory's lifecycle from social learning strategies: encoding gated by surprise, observed outcomes and earned credibility; consolidation by a credibility weighted quorum of distinct, recent sources that must also outweigh every rival behaviour; and reconsolidation by the outcomes of reproduc
    
[^277]: 部分可观测马尔可夫决策过程中的动态深度强化学习算法

    Dynamic deep-reinforcement-learning algorithm in Partially Observed Markov Decision Processes. (arXiv:2307.15931v1 [cs.LG])

    [http://arxiv.org/abs/2307.15931](http://arxiv.org/abs/2307.15931)

    本研究研究了在部分可观测马尔可夫决策过程中解决的动作序列的好处，并提出了几种扩展深度强化学习算法的结构和方法。

    

    在最近的研究中，强化学习取得了很大的进步，并且在实际应用中引起了越来越多的兴趣。在许多情况下，由于非静态干扰，使得智能体难以保持性能。这种干扰产生了被称为部分可观测马尔可夫决策过程的环境。在实践中，部分可观测马尔可夫决策过程通过引入额外的估计器或在强化学习的上下文中使用递归神经网络来处理。这两种情况都需要处理轨迹上的序列信息。然而，目前只有很少有研究探讨要考虑的信息的影响以及处理它们的网络结构。本研究展示了在解决部分可观测马尔可夫决策过程时包含动作序列的好处，并提出了几种结构和方法来扩展最新的深度强化学习算法。

    Reinforcement learning has been greatly improved in recent studies and an increased interest in real-world implementation has emerged in recent years. In many cases, due to the non-static disturbances, it becomes challenging for the agent to keep the performance. The disturbance results in the environment called Partially Observable Markov Decision Process. In common practice, Partially Observable Markov Decision Process is handled by introducing an additional estimator, or Recurrent Neural Network is utilized in the context of reinforcement learning. Both of the cases require to process sequential information on the trajectory. However, there are only a few studies investigating the effect of information to consider and the network structure to handle them. This study shows the benefit of action sequence inclusion in order to solve Partially Observable Markov Decision Process. Several structures and approaches are proposed to extend one of the latest deep reinforcement learning algori
    

