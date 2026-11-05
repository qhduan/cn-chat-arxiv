# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Framework for Context Enrichment and Selective Robustness in LLM Social Agents](https://arxiv.org/abs/2611.06914) | 该论文提出“受控情境增强”框架，通过一组提示词替换规则系统地丰富四种不同类型的情境（中性填充、通用社会背景、描述性决策依据、规范性决策依据），以探究更丰富的情境能否提升大语言模型社交智能体在语义不变提示变化下的行为鲁棒性。 |
| [^2] | [FlowReconBench: Accuracy, calibration and cost of learned and classical flow reconstruction from sparse sensors](https://arxiv.org/abs/2611.06912) | 本文提出 FlowReconBench 基准，系统评估了学习型与经典稀疏传感器流场重构方法在三种流动状态下的精度、校准与成本，发现最优方法因流动状态而异，且经典方法在低阶流动和密集传感场景下表现强劲。 |
| [^3] | [Perceptive Humanoid Motion from Simulation](https://arxiv.org/abs/2611.06908) | 提出了HALO人形机器人自主系统，通过在千个程序化生成的家庭环境中合成视觉-语言-运动数据，使数据规模随算力而非人力扩展，让人形机器人掌握坐、导航、穿门、蹲爬、推箱等多种物理技能并协调完成复合任务。 |
| [^4] | [Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement](https://arxiv.org/abs/2611.06905) | 本文探究并提出了让通用机器人策略通过强化学习在部署阶段实现多任务、通用性自我改进的算法方案，突破了以往仅能在单任务上进行优化的局限。 |
| [^5] | [Adaptive Multi Mode: Budget-Aware Routing of Optimizers By Online Landscape Probes](https://arxiv.org/abs/2611.06879) | AMM通过在线探测优化问题的景观特征（如曲率、可分离性等），将优化预算智能分配给拟牛顿、信赖域、演化策略等不同经典优化器，在bbob基准的600个函数-维度-预算组合中有382个取得最高目标份额。 |
| [^6] | [CounterFactual Retrieval for Evidence-Aware Individual Animal Identification](https://arxiv.org/abs/2611.06864) | 提出反事实前景检索（CFR）这一证据感知评估协议，通过完整图像、仅前景、仅捷径三种视图对动物个体识别进行评测，将检索分数转化为证据画像，从而揭示模型是依赖动物特征还是身份相关的背景捷径。 |
| [^7] | [Balancing hierarchical representation and quantization: A performance trade-off in text embeddings](https://arxiv.org/abs/2611.06859) | 该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。 |
| [^8] | [Visual Token Routing: Learning to keep tokens that matter](https://arxiv.org/abs/2611.06854) | 该论文提出了视觉令牌路由（VTR）这一联合训练框架，利用轻量级时空路由器筛选截图中真正关键的视觉令牌，在保留动作预测所需信息的同时大幅降低计算机使用智能体的计算成本。 |
| [^9] | [Navigating Disagreement Among Criteria for Evaluating Explanation Faithfulness](https://arxiv.org/abs/2611.06846) | 本文提出“忠实性泛函”框架，系统重构了现有的解释忠实性评估标准，揭示其分歧根源在于对解释主张、相关比较及一致性聚合方式的不同选择，为忠实性评估提供了实用的解决路径。 |
| [^10] | [Thermally Conditioned Time-Series Forecasting of Grain Growth in Polycrystalline Materials](https://arxiv.org/abs/2611.06843) | 本研究将热信息显式融入循环神经网络预测框架，把等温条件下的晶粒生长时间序列预测方法扩展到非等温热处理场景，从而能够快速预测变温热循环下晶粒尺寸分布的演化。 |
| [^11] | [Same Layout, Different Count: What Number Words Do Past Ten in Text-to-Image Models](https://arxiv.org/abs/2611.06840) | 该研究通过3,088张图像的人工计数发现，文生图模型仅在各自特定的“可达范围”内（约十至二十五）遵循数字词指令，超出该范围后图像布局由起始噪声的粗粒度部分固定，而仅重绘细粒度噪声即可像更换数字一样改变画面中的物体数量。 |
| [^12] | [Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token](https://arxiv.org/abs/2611.06837) | 本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。 |
| [^13] | [Alignment Myopia in Generative AI](https://arxiv.org/abs/2611.06830) | 该论文揭示了当前生成式AI对齐的“近视”缺陷——过度追求即时对话满意度而忽视长期负面影响（如认知萎缩与能动性减弱），并提出转向长程对齐范式，在人类福祉四个维度上权衡近期效用与长期成本。 |
| [^14] | [Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS](https://arxiv.org/abs/2611.06829) | 该论文提出了一套完整的从Vitis HLS到AWS云FPGA部署的工具链和四级误差定位方法论，使开发者乃至AI编码智能体能够通过高级语言在AWS云FPGA上高效开发、测试和部署机器学习模型。 |
| [^15] | [Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs](https://arxiv.org/abs/2611.06828) | 提出了脉动线性注意力（SLA），针对TPU的256×256脉动矩阵单元重新设计门控Delta线性注意力——将键扩展至256、把衰减门分组为64通道瓦片并用全分块矩阵乘法的单一融合内核计算——在TPU v7x上比Tokamax KDA内核快1.6至2.5倍。 |
| [^16] | [GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes](https://arxiv.org/abs/2611.06819) | 提出首个用于量子码设计的生成流网络GFlowQEC，它学习按得分比例采样生成量子LDPC码，在昂贵的电路级仿真评估预算下，找到的全新高分码数量是并行回火方法的两倍以上。 |
| [^17] | [Evolutionary Design of Interpretable Spatial Descriptors for Structure-Property Relationship Discovery](https://arxiv.org/abs/2611.06811) | 该论文提出了一种结合SHAP空间归因分析与遗传算法优化空间分区的框架，用于构建可解释的空间描述子，在预测精度与描述子维度之间取得平衡，以发现结构-性质关系。 |
| [^18] | [ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions](https://arxiv.org/abs/2611.06802) | 提出 ConDR 方法，通过单边损失和学习 HJI 变分不等式的下解来保证神经网络估计的危险集保守地包含真实危险集，同时避免退化解，实现了高维系统中安全可靠的可达性分析。 |
| [^19] | [Message-passing graph persistence](https://arxiv.org/abs/2611.06801) | 本文提出消息传递图持久性（MGP），将消息传递机制融入图过滤过程构建拓扑描述子，并证明了其图同构不变性以及严格强于现有基于持久同调描述子的表达能力。 |
| [^20] | [Transforming Integer Grids to Spatial Intuition: Perception, Not Planning, Governs ARC-AGI-3](https://arxiv.org/abs/2611.06793) | 该论文提出开源智能体SARBLOH，证明在ARC-AGI-3交互推理中起决定作用的是空间感知而非规划能力——其零参数O(HW)感知引擎可在35毫秒内将整数网格帧转化为对象清单与几何简报，从而大幅减少前沿模型因序列化破坏空间拓扑而产生的高昂探索浪费。 |
| [^21] | [Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors](https://arxiv.org/abs/2611.06789) | 研究发现，两个论文录用预测器尽管在原始论文排名上高度一致，但在判断哪些段落改写能提升分数时几乎没有一致性，且降低的数值精度产生的舍入误差会掩盖微小的分数变化。 |
| [^22] | [FedFiOT: Fisher-Weighted Optimal Transport for Model Merging in One-Shot and Few-Round Federated Learning](https://arxiv.org/abs/2611.06783) | FedFiOT提出了一种基于Fisher信息加权的最优传输框架，用于单轮和小轮次联邦学习，通过量化参数重要性来指导神经元对齐与模型聚合，解决了客户端模型漂移和数据异质性导致的直接平均性能下降问题。 |
| [^23] | [Continual Classification Beyond the Kernel Regime: Harnessing the Sparsity Bias of L2 Regularization in Diagonal Linear Networks](https://arxiv.org/abs/2611.06759) | 该论文推导了持续线性分类的最优L2正则化强度并刻画其泛化性能，同时利用两层对角线性网络揭示了L2正则化的稀疏性偏好可以在持续学习中实现超越核机制的特征学习。 |
| [^24] | [Coupled Scene Graph Anticipation: When With-Constraint Recall Hides Structural Inconsistency](https://arxiv.org/abs/2611.06738) | 本文提出CoupledSGA，通过关系交互图与结构一致性损失耦合共享实体的关系以避免互斥预测，并证明带约束评估指标会掩盖这种结构不一致性。 |
| [^25] | [ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models](https://arxiv.org/abs/2611.06732) | 该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。 |
| [^26] | [Template Gap Reconstruction for Shell Output Simulation with Small Language Models](https://arxiv.org/abs/2611.06731) | 该论文提出模板空缺重构（TGR）方法，通过从1870万个真实输出构建的模板库中提供可复用的输出字面量并与自回归生成的空缺内容交错，使小型语言模型在Shell输出模拟中显著优于完整输出解码。 |
| [^27] | [Turns Matter: Advancing Streaming Video RL via Turn-Level Credit](https://arxiv.org/abs/2611.06728) | 提出TurnWise——一个回合级GRPO框架，通过基于静默补全前缀评分的回合级信用分配和回合均值策略优化，解决了流式视频强化学习中轨迹级反馈与回合级决策不匹配的问题，从而促进有信息量的回应并减少冗余回复。 |
| [^28] | [Dexterous Robotic Hands in the Foundation-Model Era: A Full-Stack Survey of Hardware, Data, Cross-Embodiment Transfer, Learning, and Evaluation](https://arxiv.org/abs/2611.06724) | 这是一篇全栈式综述，首次将基础模型时代的灵巧机械手研究系统性地组织为硬件、数据、跨本体迁移、学习与评估五个相互依赖的层次，并通过案例分析揭示各层设计选择之间的内在联系。 |
| [^29] | [Symmetry Breaking at the Edge of Stability via Central Flow Analysis](https://arxiv.org/abs/2611.06718) | 本文借助中心流分析首次系统研究了神经网络在稳定性边缘处的对称性破缺现象，证明曲率修正会引入由临界参数方向上振荡协方差决定的有限维漂移，从而打破了梯度流下的连续对称性守恒律（如层平衡性），并在标量两层模型、两层矩阵分解和两层ReLU网络中刻画了哪些守恒律得以保持、哪些被破坏。 |
| [^30] | [Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching](https://arxiv.org/abs/2611.06711) | 提出了一种免训练的能量引导传输框架，将非线性物理约束建模为平滑势能场，无需离散投影即可在流匹配推理过程中保持物理一致性，并避免破坏概率路径和降低计算开销。 |
| [^31] | [ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus](https://arxiv.org/abs/2611.06706) | 该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。 |
| [^32] | [Where It Breaks: Diagnosing Perception, Memory, and Decision Failures of a Vision-Language Agent in Pac-Man](https://arxiv.org/abs/2611.06704) | 该研究通过对基于Qwen3.5-9B的吃豆人智能体分别进行感知检查和使用文本棋盘的基元策略强化学习，发现静态截图读取准确可能与交互中的错误并存、策略优化可能伴随更差的动作选择，从而为后训练提出了在使用中评估感知、区分感知与决策失败的具体检查方法。 |
| [^33] | [PINR-ESI: Physics-Informed Self-Supervised Implicit Neural Representation Framework for EEG Extended-Source Imaging](https://arxiv.org/abs/2611.06700) | 提出了一种物理信息驱动的自监督隐式神经表示框架PINR-ESI，利用导联场矩阵的物理约束和经典逆估计的暖启动初始化，在无需成对源标签或仿真预训练的情况下实现脑电扩展源成像的空间源重建。 |
| [^34] | [Don't Replace, Complement: A Detector-Agnostic Subspace Signal for Near-OOD Detection](https://arxiv.org/abs/2611.06688) | 该论文提出PLUS方法，利用稀疏自编码器为每个类别构建低维正交子空间并计算白化投影评分，作为与现有检测评分互补的近分布外检测信号，从而在不牺牲远分布外性能的前提下提升近分布外检测能力。 |
| [^35] | [SecureGARD: Dynamic Ontology-Guided Governance for Secure Tool-Using LLM Agents](https://arxiv.org/abs/2611.06682) | 提出SecureGARD框架，通过将可执行本体与回合级任务状态相结合，为工具使用型LLM代理生成确定性的就绪、等待和禁止决策，从而抵御间接提示注入攻击，同时在不扩大授权边界的前提下支持有界重规划。 |
| [^36] | [Do unlearning algorithms learn the right features?](https://arxiv.org/abs/2611.06681) | 该论文证明标准的机器遗忘评估指标可能掩盖特征层面的本质差异，并提出一种区分缺失与多余特征的特征恢复度量，发现标准指标表现良好的遗忘模型往往无法恢复重训练模型的特征。 |
| [^37] | [Self-Directed Apple Tasting](https://arxiv.org/abs/2611.06677) | 本文在自导向苹果品尝问题中，通过 VC 维、$\emptyset$-中心星数和自导向维数等组合参数，建立了可实现设置下最小最大错误率的四分法刻画，以及不可知设置下遗憾率的二分法刻画。 |
| [^38] | [Fingerprinting the Dispatchers behind Distributed Model Extraction](https://arxiv.org/abs/2611.06661) | 该论文提出通过请求时间戳（到达时间和流完成时间）指纹识别分布式模型提取攻击背后的调度策略，仅用20%种子集即可在20分钟内以0.92至1.00的AUC对攻击活动中剩余账户进行取证式排序，即使内容分类器失效时也能有效检测此类攻击。 |
| [^39] | [Detecting Silent Action-Interface Mismatch with Calibrated Relational World Models](https://arxiv.org/abs/2611.06653) | 该论文提出将学习到的状态-动作-效果关系作为离线契约，利用校准的联合高斯混合评分及其条件分布来检测机器人动作接口中的静默语义失配，在LIBERO任务上达到65.7%的召回率和95.8%的失配字段定位准确率。 |
| [^40] | [TestBED: An Open-Source Toolkit for Bayesian Experimental Design](https://arxiv.org/abs/2611.06633) | 本文提出了 TestBED，一个开源的贝叶斯实验设计通用工具包，通过将 BED 流程分解为任务、目标、训练机制、策略架构和评估流程等模块化、可自由组合的组件，解决了该领域缺乏标准化软件和基准测试的问题。 |
| [^41] | [Noise-regularized training of quantum circuit Born machines under a finite shot budget](https://arxiv.org/abs/2611.06630) | 该论文证明硬件噪声可作为量子电路玻恩机器的正则化手段以提升泛化能力，并提出GENIE框架在有限测量预算内自动选择最优噪声水平。 |
| [^42] | [Interactive World Simulation with Cosmos for Physical AI](https://arxiv.org/abs/2611.06627) | 该论文提出了一个将 Cosmos 世界基础模型适配为实时交互式世界仿真的共享框架，使用户和具身智能体能够通过动作与场景控制同演化的虚拟环境交互，并支持相机导航、机器人操作、可控视频迁移和多视角自动驾驶仿真等应用。 |
| [^43] | [Dimension-Free Private Mean Estimation is Impossible Without the Covariance](https://arxiv.org/abs/2611.06615) | 该论文通过建立新的下界证明：在协方差矩阵未知的情况下，差分隐私高斯均值估计无法实现与维度无关的最优误差保证——当样本量低于有效维度时，误差会比协方差已知情形大 √(d/n) 倍，即使样本量更高时差距依然存在。 |
| [^44] | [From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels](https://arxiv.org/abs/2611.06612) | 该论文提出一种测试时优化方法，将冻结的序列重建器在推理时转化为插补器——把缺失条目作为优化变量、仅最小化观测坐标上的重建误差，从而无需人工掩码增强或全通道丢弃训练即可应对完整传感器通道失效导致的时间序列缺失。 |
| [^45] | [Alternatives to Shallow Autoencoders for Collaborative Filtering](https://arxiv.org/abs/2611.06610) | 该论文针对EASE模型仅依赖物品-物品关系而限制细粒度用户关系捕捉的问题，提出了利用用户-用户信息的替代模型，以及结合用户-用户和物品-物品邻接分布的混合模型，从而提升协同过滤的推荐准确性。 |
| [^46] | [Verifier-Guided Self-Consistency: Test-Time Reasoning with Imperfect Generators and Verifiers](https://arxiv.org/abs/2611.06606) | 该论文提出验证器引导的自一致性（VGSC）随机解码框架，在生成过程中同时利用步骤级正确性与未来结果估计两种信号，在不完美的生成器多样性与验证器正确性引导之间取得平衡，从而改进测试时推理。 |
| [^47] | [The Behavioral Rate of Fine-Tuning: Compression as a Measure of Useful Information](https://arxiv.org/abs/2611.06598) | 本文提出“行为速率 $R^\star$”这一新指标——以能保留固定比例微调增益的压缩适配器最小序列化大小来度量微调写入的有用信息，并发现该速率由训练内容与冻结基础模型共同决定，而非参数量、微调增益或重建误差等直观指标。 |
| [^48] | [A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration](https://arxiv.org/abs/2611.06597) | 提出了一种与模型无关的核聚变台基稳定性代理模型信任层，通过共形误差棒、分布外告警和少样本重校准，即使在等离子体形状偏移导致查询超出训练数据时也能提供可信的不确定性量化。 |
| [^49] | [Thermodynamics-Informed Learning Captures Hierarchical Hydrogen-Bond Networks in Deep Eutectic Solvents](https://arxiv.org/abs/2611.06590) | 该研究提出热力学信息引导的神经网络TiNN，通过学习自由能路径分层刻画低共熔溶剂中氢键受体-供体缔合与溶质容纳过程，将木质纤维素溶解度预测误差从COSMO-RS的23倍以上降至约1.5倍。 |
| [^50] | [Finite-width shift of the edge of chaos in deep neural networks](https://arxiv.org/abs/2611.06582) | 该论文将两输入距离的对数建模为随机游走，精确计算出有限宽度下深度神经网络的混沌边缘向更大权重方差方向偏移（tanh网络约为8/N），并解释了典型网络与网络平均的临界点移动方向相反这一长期争议。 |
| [^51] | [From Fixed Defaults to Calibrated Activation Policies for Long-Term Time-Series Forecasting](https://arxiv.org/abs/2611.06559) | 该研究提出一种经过校准的激活选择策略，能够根据任务自适应地选择激活函数及其位置，并在证据不足时保留默认配置，在1,225次Informer训练实验中平均将长期时间序列预测的均方误差降低7.86%。 |
| [^52] | [RAGAM-AI: A Hybrid Reinforcement Learning based Agentic approach powered by Generative AI for Edge Resource Management](https://arxiv.org/abs/2611.06554) | 提出RAGAM-AI混合框架，将强化学习与生成式AI结合构建智能体工作流，利用LLM作为奖励预言机解决奖励泛化问题，以优化多接入边缘计算单元的资源分配。 |
| [^53] | [Adapting Sleep Foundation Models for Dense Sleep Spindle Localization](https://arxiv.org/abs/2611.06546) | 该论文提出通过为冻结的SleepFM编码器添加局部时频通路与残差交叉注意力机制，成功实现密集睡眠纺锤波定位，在MODA数据集上取得了0.808的事件级F1分数。 |
| [^54] | [EvoOPD: Evolving Privileged Information Strategies for Black-Box On-Policy Distillation](https://arxiv.org/abs/2611.06542) | 提出EvoOPD框架，通过演化式地确定应从黑盒教师获取何种特权信息，并将其转化为学生自身轨迹上的密集监督信号，从而有效实现黑盒模型的在线策略蒸馏。 |
| [^55] | [IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning](https://arxiv.org/abs/2611.06519) | 提出IDSISA方法，利用影响函数将易被遗忘的样本集中到少数分片，并通过未受影响分片的蒸馏恢复效用，在保持SISA精确删除保证的同时提升机器遗忘的效率和模型性能。 |
| [^56] | [An Interaction-Aware Decision Framework for FDIC-Assisted Distressed Bank Acquisitions: Empirical Evidence from the FDIC Failed-Bank Panel and the SVB-First Citizens Case](https://arxiv.org/abs/2611.06514) | 提出蜘蛛网络过滤框架，将倒闭银行收购中五个维度信号间的相互依赖编码为四条可解释的交互规则，据此构建“推进/谨慎/回避”决策规则，并在2008—2024年55宗FDIC协助收购中以公告期正异常收益加以验证。 |
| [^57] | [Nowcasting Future Performance for Population Pruning in Survival-of-the-Fittest Training](https://arxiv.org/abs/2611.06512) | 该论文提出了基于准确率与损失历史、权重统计以及SANE检查点编码的即时预测器来预测模型未来的测试准确率，并将其用于适者生存（SOTF）训练，从而在训练过程中更明智地对候选模型种群进行早期剪枝。 |
| [^58] | [ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events](https://arxiv.org/abs/2611.06508) | 提出了ElecLoad——一个上下文感知的智能体，能够利用网页通知、天气预警等多重事件的影响及其相互作用，自适应地组织电力负荷预测过程，从而提升预测的准确性。 |
| [^59] | [NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria](https://arxiv.org/abs/2611.06504) | 该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。 |
| [^60] | [Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation](https://arxiv.org/abs/2611.06498) | 本文揭示了离线RAG签名知识包中“签名只证明发布者身份、无法保证向量与文本一致”这一安全鸿沟，提出了量化攻击者可劫持查询比例的“边际定律”，并给出了测量与弥合该完整性差距的验证方案。 |
| [^61] | [Risk-Sensitive Non-Parametric and Kernel Bandits](https://arxiv.org/abs/2611.06489) | 本文提出NPCB乐观算法，将累积前景理论（CPT）风险敏感评估引入带连续动作空间的非参数赌博机，证明了其CPT遗憾与经典非参数赌博机量级一致，说明风险敏感性不带来额外的量级遗憾代价，并扩展至未知噪声分布的情形。 |
| [^62] | [Model Merging Does Not Require Uniform LoRA Ranks](https://arxiv.org/abs/2611.06486) | 该论文证明模型合并不要求各专家使用统一的LoRA秩，为不同领域专家分配各自的最佳秩可让平均法和基于效用的选择方法零成本地提升最多1.8个百分点的多领域准确率，而冲突解决类方法则表现不一致甚至可能损失4个百分点。 |
| [^63] | [What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?](https://arxiv.org/abs/2611.06483) | 本文通过Pass@(K,T)评估发现多轮智能体强化学习的优势依赖充足的交互轮次预算，且主要体现在错误恢复与长程韧性而非单步决策质量上，进而提出在训练中动态改变交互预算的BudgetMix RL，以在多种执行设置下实现稳健性能。 |
| [^64] | [The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention](https://arxiv.org/abs/2611.06476) | 对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。 |
| [^65] | [How many sorted projections determine n points in k dimensions?](https://arxiv.org/abs/2611.06470) | 本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。 |
| [^66] | [Within-Family EfficientNet Scaling under Fixed-Resolution Transfer Learning for Spectral-Domain Optical Coherence Tomography Retinal Disease Classification](https://arxiv.org/abs/2611.06452) | 该研究发现固定分辨率迁移学习中EfficientNet家族内按规模缩放并不能单调提升SD-OCT视网膜疾病分类性能，最小的B1以96.95%的准确率超越更大的B7模型，表明在医学图像迁移学习中盲目增大模型并非最优选择。 |
| [^67] | [Graph Foundation Models Can Become Effective Multivariate Time Series Anomaly Detectors](https://arxiv.org/abs/2611.06441) | 提出Gfm4Mtsad适配框架，通过引入结构与信息契约对齐（SICA）模块，将预训练图基础模型成功迁移到多变量时间序列异常检测任务中，克服了跨模态不匹配、伪异常监督不可靠和决策失真三大挑战。 |
| [^68] | [Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2611.06438) | 该论文首次从理论上将大语言模型的思维链推理建模为组合式分布外泛化问题，证明其通过复用共享的单步预测器组合分布内原子转换来实现泛化，且分布外风险由分布内子任务风险之和乘以一个与推理步数无关的 $\mathcal{O}(k)$ 系数所控制。 |
| [^69] | [Hardwired Distributed Processing Units for Inference: Neural Decisions at the Sensor that Free the Host](https://arxiv.org/abs/2611.06436) | 该论文提出一种在低成本Artix-7 FPGA上实现的硬连线分布式处理单元（HDPU），仅用1,814个LUT、无需处理器和操作系统即可在传感器端以固定时延完成神经推理决策（如斑马线检测），从而将全部推理负担从主机中解放出来。 |
| [^70] | [PARGAR: Pilot-Aware Graph Attention for Sparse-Pilot Neural OFDM Receivers](https://arxiv.org/abs/2611.06432) | 本文提出PARGAR，一种导频感知的图注意力神经接收机，通过将OFDM资源网格建模为融合局部时频边与导频锚点连接的图，有效提升了稀疏导频配置下的比特检测性能。 |
| [^71] | [Partially Interpretable Decision Trees](https://arxiv.org/abs/2611.06428) | 该论文提出了三种部分可解释决策树算法（Branches精确搜索扩展、基于CART的转换算法和考虑转交选项的广义CART），并通过形式化界限证明它们在真实数据集上表现相似的原因在于智能树构建的潜在收益本身有限。 |
| [^72] | [D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding](https://arxiv.org/abs/2611.06425) | D3D 提出一种不依赖目标模型隐藏状态的独立扩散草稿模型，使草稿生成与验证可异步并行执行，在多个基准上平均实现约 5 倍的解码加速。 |
| [^73] | [A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks](https://arxiv.org/abs/2611.06412) | 本文针对训练数据可能被第三方污染的人脸识别系统，提出了三种实用的后门攻击防御技巧——一种在训练阶段检测被投毒的身份，两种在推理阶段过滤可疑输入，为系统提供商提供了可落地的安全加固方案。 |
| [^74] | [Zero-Shot Coordination among LLM Agents](https://arxiv.org/abs/2611.06409) | 该论文通过设计隔离特定协调挑战的最小化环境与简单通用脚手架，首次系统研究了独立开发的LLM智能体之间的零样本协调问题，发现即使最前沿的LLM智能体也难以协调，其主要原因在于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。 |
| [^75] | [GIST: Gibbs-Targeted Hierarchical Data Selection for Offline RLVR](https://arxiv.org/abs/2611.06395) | 该论文提出GIST方法，基于双向KL散度在参考策略处的精确分解，揭示出正确响应应聚焦其低概率token、错误响应应聚焦策略高频生成响应的高概率token这两条监督原则，并据此设计两级层级数据选择方法来优化离线RLVR的训练监督。 |
| [^76] | [KolTa: A Physics-Informed Active Learning Framework for Formulation Discovery with Limited Data](https://arxiv.org/abs/2611.06394) | KolTa是一个物理信息驱动的主动学习框架，通过结合三阶段预测器与物理残差校准UCB采集规则，在有限实验预算下高效发现目标温度下排名前5%的最优配方。 |
| [^77] | [From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents](https://arxiv.org/abs/2611.06382) | 该研究基于守恒、耗散和聚合三条水文定律构建简约流域模型CaRT，并通过AI代理辅助的假设驱动迭代细化得到CaRT+模型，在不使用日尺度神经网络的情况下于三大洲达到与深度学习相当的精度，证明物理定律约束的低维模型可以匹敌神经网络。 |
| [^78] | [Give Up to Get More: Priors, Not Policies, Govern Test-Time Compute Allocation](https://arxiv.org/abs/2611.06372) | 该研究揭示了测试时计算的最优静态分配呈倒U形——应将尝试次数集中在中等难度而非最难的输入上，并指出真正的收益来源于知道哪些输入可解的先验信息，而非分配策略本身。 |
| [^79] | [EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models](https://arxiv.org/abs/2611.06356) | 该论文发布了首个即用型EEG基础模型公开训练语料库eegdash-corpus-v0，汇集了77个可合法再分发的公开数据集（涵盖7,272名受试者的14,075小时数据），解决了临床档案无法再分发、公开数据分散且格式不一导致难以汇集的核心问题。 |
| [^80] | [Flexible Intensities Matter: A Comprehensive Re-evaluation of Hawkes Processes and Neural Temporal Point Processes](https://arxiv.org/abs/2611.06348) | 该研究通过对标准基准和大规模医疗数据的系统性重新评估发现，一旦为霍克斯过程配备灵活可学习的基线与核函数，其与神经时间点过程之间此前报道的性能差距基本消失，从而挑战了神经TPPs显著优于经典TPPs的结论。 |
| [^81] | [CoRTeC: A Differentially Private Release for Synthetic Tabular Data, Decodable by Any Generator](https://arxiv.org/abs/2611.06342) | CoRTeC通过一次性发布一组经过差分隐私保护的含噪统计计数文件，使任何人都能用最大熵解码器或语言模型等任意生成器、以零额外隐私成本反复生成可防成员推断攻击的合成表格数据。 |
| [^82] | [Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves](https://arxiv.org/abs/2611.06338) | 该论文研究了迭代反馈整定（IFT）在汽车电子节气门这类非线性对象上调参时的梯度偏差问题，发现静摩擦导致的两次实验间控制器状态不匹配会引入与第二次实验幅值成反比的误差，且IFT梯度仅在69%的低成本区域内为下降方向。 |
| [^83] | [Mass Producing Training Data with Surprising Side Effects](https://arxiv.org/abs/2611.06324) | 本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。 |
| [^84] | [Conformal Prediction for Distribution-to-Distribution Regression](https://arxiv.org/abs/2611.06320) | 本文针对分布到分布回归问题，提出了一种在正交基空间中运行的共形预测新框架，能够生成可解释的预测集，同时保持覆盖保证，克服了朴素应用分裂共形预测导致的预测集不可解释的缺陷。 |
| [^85] | [When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Link](https://arxiv.org/abs/2611.06307) | 对低轨卫星链路上BG-CFQS风险受控吞吐量预测的复现研究发现，其风险保证虽然可复现但依赖校准集划分方式——在实际部署所面临的连续时间划分下大多失效，且无论何种划分都无法实现低于0.15的风险预算。 |
| [^86] | [The Split Protocol Decides the Result: Re-Evaluating Encrypted Traffic Classification](https://arxiv.org/abs/2611.06290) | 该论文揭示加密流量分类报告的高准确率主要由数据分割协议决定——同一模型在不同分割协议下宏F1可从0.992骤降至0.567，且关键划分轴取决于语料库的收集方式，并提出了一个无需训练模型即可识别正确划分轴的诊断方法。 |
| [^87] | [Consistency Learning and Distillation for One-Step MIMO-OFDM Channel Estimation](https://arxiv.org/abs/2611.06271) | 本文提出观测锚定一致性学习与一致性中期训练（CMT）蒸馏方法，实现MIMO-OFDM系统的一步式信道估计，紧凑的CMT学生模型在参数减少74.5%的情况下将符号错误率降低9.6-20.2%。 |
| [^88] | [Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature](https://arxiv.org/abs/2611.06265) | 该论文提出图符号指数（s_G）来量化正定几何先验与异质图之间的根本错配，并据此提出伪黎曼注意力网络（PRAT），通过带符号的平方伪距离注意力机制在伪黎曼流形上学习异质图表示。 |
| [^89] | [Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks](https://arxiv.org/abs/2611.06257) | 本研究系统评测了十二种本地开源权重LLM配置在CyberGym漏洞复现任务上的表现，发现即使小至4B的模型也能解决大部分较简单的漏洞复现任务，表明漏洞复现是本地小模型可胜任的自动化分析任务。 |
| [^90] | [Learning Noise Optimization for Reward Alignment in Few-Step Generative Models](https://arxiv.org/abs/2611.06255) | 提出LeNO方法，将奖励引导的初始噪声优化摊销为学习到的残差传输，在冻结预训练生成器的前提下实现少步生成模型的高效奖励对齐，并通过KL散度上界提供理论正则化。 |
| [^91] | [Rubric Curriculum RL: Exploiting the Generation-Verification Gap in Non-Verifiable Domains](https://arxiv.org/abs/2611.06244) | RcRL 通过将对采样结果的成对偏好与评分标准课程机制相结合，在无需新数据、人工标注或更强教师模型的情况下，利用生成-验证差距在不可验证的开放式领域实现持续的性能提升并有效避免奖励作弊。 |
| [^92] | [Zuse: A Decision Model for European Languages](https://arxiv.org/abs/2611.06238) | Zuse是首批面向27种欧洲语言的决策模型系列（4B/9B/27B），通过在冻结的开源基座上训练低秩适配器和指针头，实现了87.5%的准确率和极低的校准误差，并发现校准偏差主要由任务而非语言决定。 |
| [^93] | [Separation Is a Riesz Energy: How Many Environments Does ICA Need?](https://arxiv.org/abs/2611.06228) | 本文证明在高斯 ICA 中分离某个成分的代价由其在各环境方差签名所对应的 Riesz 2-能量刻画，并据此给出样本复杂度的维度定律：环境数 K=2、3 及 K≥4 时分别需要 d²/n、d·log d/n 与 d/n 量级的样本。 |
| [^94] | [Consistent Submodular Maximization with Full Foresight](https://arxiv.org/abs/2611.06223) | 该论文在“完全预知未来且计算不受限”的理想设定下，刻画了单调次模最大化中解稳定性（一致性）的代价，证明了确定性算法与随机算法的最优近似阈值分别为 3/4 和约 0.7942，并以多项于 1/ε 的变更代价逼近该阈值，同时对覆盖函数给出了匹配的不可能性证明。 |
| [^95] | [A Definition of Catastrophic Performance](https://arxiv.org/abs/2611.06219) | 该论文提出了“灾难性性能”的正式定义，通过基线值和正交值两个参考值来区分次优结果与灾难性结果，从而为严格分析人工智能目标错位带来的灾难性风险提供了理论基础。 |
| [^96] | [Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining](https://arxiv.org/abs/2611.06206) | 该研究发现在1亿词的数据受限预训练中，循环深度（12层堆叠重复四次）在相同推理计算量下的纯文本任务中比标准48层Transformer高出5.4个BLiMP百分点，但无法缓解多模态训练导致的约3.8个百分点的语言能力下降。 |
| [^97] | [Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data](https://arxiv.org/abs/2611.06194) | 本文提出一个零样本域适应框架，在目标资产完全没有故障数据的情况下，仅利用其正常数据即可通过EWMA自动标注源资产数据并借助改进的域对抗神经网络实现跨资产早期故障检测。 |
| [^98] | [GPUMAP: GPU-Accelerated Progressive Uniform Manifold Approximation and Projection](https://arxiv.org/abs/2611.06191) | GPUMAP通过将UMAP重构为渐进式流水线并利用GPU并行加速，实现了在普通GPU硬件上对百万级数据集的交互式渐进降维分析。 |
| [^99] | [Chiron: Cache Management Empowered by LLM Intelligence for AI Workloads](https://arxiv.org/abs/2611.06179) | Chiron提出了一种由LLM赋能的咨询式切换控制缓存管理方法：LLM异步诊断AI工作负载的访问模式并推荐缓存策略，由确定性框架把关建议并在本地执行专家设计的策略，从而在保持用户透明性的同时自适应地优化缓存性能。 |
| [^100] | [Network Intrusion Forensic Detection System with Explainable Artificial Intelligence and Digital Chain of Custody](https://arxiv.org/abs/2611.06173) | 该论文提出MODEXRE取证流水线，将经过验证的SHAP可解释XGBoost入侵检测模型扩展为“实验室模式+正式模式”双模式架构，在SHA-256哈希链式数字证据链保障下，对经OCSF归一化的多源异构网络证据执行可验证且可解释的取证推理。 |
| [^101] | [AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment](https://arxiv.org/abs/2611.06170) | 本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。 |
| [^102] | [FaCoDiff: Factor-Correlated Diffusion for Multivariate Probabilistic Forecasting](https://arxiv.org/abs/2611.06166) | FaCoDiff通过以历史为条件的因子载荷、对角补偿与低秩算子，将紧凑的条件统计估计与结构化去噪相耦合，在无需稠密协方差存储或分解的情况下实现高效的多元概率预测。 |
| [^103] | [Regret Bounds for Market Making with Persistent Orders](https://arxiv.org/abs/2611.06159) | 本文针对订单持续有效直至执行的对抗性双市场做市问题，提出了一个简单确定性策略，其相对静态和动态比较器分别取得最优的 $\log T$ 和 $\sqrt T$ 遗憾界，并证明了两项价格风险暴露项的必要性。 |
| [^104] | [Selective Prediction Intervals: Jointly Learning to Predict, Bound, and Abstain](https://arxiv.org/abs/2611.06158) | 提出SelPI方法，在单一损失函数和显式接受率预算下联合训练预测值、区间和弃权三个头，并实证证明仅凭区间宽度无法可靠判断何时应当弃权。 |
| [^105] | [Neural Stochastic Forecasting for Lateral Molecular Diffusion Simulations](https://arxiv.org/abs/2611.06157) | 提出了一种结合Transformer构象表示、多时间尺度循环网络和带学习记忆的随机潜过程的神经随机模型，能够从粗粒化分子动力学轨迹高效预测膜蛋白在膜内的长时间集体扩散。 |
| [^106] | [CARM: Conformal Adaptive Risk Masking for Reliable Scientific Super-Resolution](https://arxiv.org/abs/2611.06151) | 提出CARM框架，通过维护异构更新速率的专家并利用非对称代理损失在线聚合阈值，在时间分布偏移下实现自适应空间风险控制，为科学超分辨率生成可靠的信任掩码。 |
| [^107] | [SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving](https://arxiv.org/abs/2611.06147) | 本文揭示了MoE大模型服务中减少激活专家数量的加速方法会削弱模型对有害请求的拒绝能力，并提出SafeStart方法，在保留大部分加速效果的同时大幅恢复模型的安全性。 |
| [^108] | [Short-Term Forecasting of Heterogeneous Wind Power Fleets: Global Temporal Fusion Transformer Leveraging Static Covariates](https://arxiv.org/abs/2611.06130) | 本研究提出一种利用静态协变量（场龄、海拔、轮毂高度、转子直径和功率曲线参数等）在异构风电场组合上联合训练的全局时间融合Transformer（TFT），并证明其在48小时前瞻风电功率预测中优于针对单个站点训练的局部模型。 |
| [^109] | [From Handcrafted to Generative: RLVR-Optimized Icon Generation for Alexa+](https://arxiv.org/abs/2611.06123) | 该论文提出并部署了一套仅凭文本描述自动生成矢量化动画图标的生产级流水线，并将图标创建分解为多个阶段，重点利用 RLVR（可验证奖励的强化学习）优化其中决定整体质量的最关键阶段——图像生成，从而解决手工绘制图标耗时且难以规模化的问题。 |
| [^110] | [Learning When to Think Aloud in Latent Reasoning](https://arxiv.org/abs/2611.06108) | VITAL提出了一种重要性引导的混合推理框架，通过轻量级门控机制动态决定哪些token需要显式生成作为推理锚点、哪些计算保持在潜在空间，并结合课程式监督训练与平衡正确性和简洁性的强化学习，从而在效率和精度之间取得平衡并超越现有完全潜在推理方法。 |
| [^111] | [Many Pixels, One Composition: Self-Supervised Spectra for Specimen-Level Prediction](https://arxiv.org/abs/2611.06096) | 该论文提出一种自监督像素级编码器结合轻量级注意力回归头的方法，从高光谱成像预测地球表面物理样本的元素成分，在保持较低模型容量的情况下显著优于传统光谱匹配、解混及基于平均光谱的监督学习基线。 |
| [^112] | [Shallow Encoding Emits Deep Memories](https://arxiv.org/abs/2611.06084) | 提出了“发射器”机制——即深层写入路径的训练副本，能够从较浅的锚定层残差流中直接写出深层记忆，从而大幅降低混合语言模型推理服务中的计算与内存开销。 |
| [^113] | [Catastrophic Overfitting is Weight Drift the Training Loss Cannot See](https://arxiv.org/abs/2611.06082) | 该论文揭示了灾难性过拟合的本质是训练损失几乎无法感知的权重漂移，并提出利用FAT协议中已有的权重指数移动平均（EMA）作为几乎零成本的修复方案。 |
| [^114] | [Offline Learning of Reserve Prices from Censored Bids](https://arxiv.org/abs/2611.06067) | 本文在单调风险率假设下精确刻画了从删失出价数据中识别二价拍卖最优保留价格的条件，并在识别失败时给出仅依赖删失比例和最低历史保留价格处风险率这两个可观测量的极小极大遗憾最优保留价格。 |
| [^115] | [PRISM-AMC: Replay-Free Continual Learning for Automatic Modulation Classification](https://arxiv.org/abs/2611.06066) | PRISM-AMC是一种无需数据回放的持续学习框架，通过梯度投影记忆保留任务相关的中间表示子空间，在类增量和域增量场景下有效缓解自动调制分类中的灾难性遗忘。 |
| [^116] | [Learning From Static Context in Federated Time Series Forecasting](https://arxiv.org/abs/2611.06064) | 在联邦时间序列预测中，逐批次聚合梯度的FedAdam几乎能像集中式训练一样充分利用描述序列的静态上下文输入，而采用本地epoch的通信高效方案（如FedAvg、FedAdam+Prox）则难以从中学习且无法获益。 |
| [^117] | [Softmax Attention with Any Temperature in Fixed Dimension: Matching the Best Known Maximum-Inner-Product Bounds](https://arxiv.org/abs/2611.06055) | 该论文提出了固定维度下支持任意温度的softmax注意力快速算法，其期望运行时间达到了已知的最大内积搜索最优理论界。 |
| [^118] | [Alleviating Limited Effectiveness and Multi-Class Collapse in Deep Graph-level Clustering](https://arxiv.org/abs/2611.06052) | 该论文提出S³GLC框架，利用Hoffman-Wielandt不等式构建谱签名作为图编辑距离的高效下界代理，将全局拓扑特征与局部邻域特征相融合，从而缓解深度图级聚类中区分能力不足和多类别聚类坍塌的问题。 |
| [^119] | [Impact of Negative Weights on Performance in Reservoir Computing](https://arxiv.org/abs/2611.06051) | 带符号储备池中适度的负权重比例能够带来更丰富的内部动力学，并显著降低混沌时间序列预测的误差。 |
| [^120] | [Do Search-Augmented Recommenders Use the Latest Search Episode?](https://arxiv.org/abs/2611.06044) | 提出FocalSwap审计框架，通过在固定目标与候选集的条件下替换最新搜索或全部搜索输入，检验搜索增强推荐系统对最新搜索会话的实际依赖程度，发现其估计收益随对照方式的选择而变化。 |
| [^121] | [Act-HiMotif: Hierarchical Daily Activity Schedule Generation via Learned Activity-Structure Motifs](https://arxiv.org/abs/2611.06038) | 该论文提出Act-HiMotif，一种层次化生成模型，通过学习可复用的活动结构主题作为个体画像与活动结构之间的中间表示，从个体属性生成每日活动计划，有效应对活动结构长尾分布和同一结构下多种时长分配的挑战。 |
| [^122] | [TRACK: Multiple Instance Learning for Anti-Money Laundering in Transaction Networks](https://arxiv.org/abs/2611.06035) | 该论文提出TRACK方法，将账户级反洗钱检测重新表述为多示例学习问题，通过共享编码器对每笔交易编码并选择性汇聚为账户决策，从而在无需交易级标签的情况下解决了“聚合失配”问题，并在五个AML数据集上取得了最高的平均精度。 |
| [^123] | [Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels](https://arxiv.org/abs/2611.06027) | 该论文将仅基于软标签的贝叶斯误差估计技术从标准二分类扩展到公平分类等受约束分类问题，实现了对贝叶斯最优公平性-准确性权衡（帕累托前沿）的实用估计。 |
| [^124] | [Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox](https://arxiv.org/abs/2611.06022) | 该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。 |
| [^125] | [Attraction-Ranking Mismatch in Out-of-Distribution Detection: Theory and Evidence from Hamiltonian Feature Dynamics](https://arxiv.org/abs/2611.06021) | 该论文利用哈密顿特征动力学揭示了分布外检测中的“吸引-排序失配”现象：提升类亲和度分数并不必然改善ID与OOD样本的相对排序，且在源域验证集上有效的分数对齐规则在迁移后反而会损害检测性能。 |
| [^126] | [How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition](https://arxiv.org/abs/2611.06001) | 该论文提出“行为-能力前沿”框架来评估对比激活添加（CAA）的可用性，在能力退化明确预算约束下最大化行为收益，发现行为效果具有强泛化性（ρ=0.945），且将能力损失预算从1%提高到3%可使干预的联合存活率从37.5%显著提升至72.7%。 |
| [^127] | [FedIA: Instance-Aware Cross-Modal Representations for Federated Prompt Learning](https://arxiv.org/abs/2611.05994) | 该论文提出FedIA框架，通过学习实例感知的跨模态表示，使联邦提示学习能够针对每个输入自适应地平衡全局共享知识与个性化知识，克服了现有方法仅在客户端层面进行个性化而忽略样本间视觉差异的局限。 |
| [^128] | [Decouple to Reuse: Fine-Grained Knowledge Augmentation for Time Series Foundation Models](https://arxiv.org/abs/2611.05993) | 提出一种两阶段细粒度知识增强框架，通过将预训练知识解耦为趋势和季节性双分支残差码本并动态检索融合，以应对领域偏移、增强冻结的时间序列基础模型。 |
| [^129] | [Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation](https://arxiv.org/abs/2611.05992) | Clips2Story提出了一种无需训练的LLM流水线，通过多模态描述符检索与关键词相关的视频片段，并自动编排成带有旁白叙述的可编辑故事板和连贯短视频。 |
| [^130] | [A Controlled Experimental Evaluation of Configuration and Cache Architecture in LLM Serving Engines](https://arxiv.org/abs/2611.05991) | 该研究通过受控实验发现，vLLM 与 SGLang 服务引擎之间的巨大性能差距主要源于分块预填充的默认配置差异而非缓存架构本身，仅对齐这一配置即可消除大部分差距。 |
| [^131] | [A Comparative Study of Importance Attribution to Columns in Deep Learning over Relational Databases](https://arxiv.org/abs/2611.05986) | 该论文提出了一个系统比较关系数据库深度学习中列重要性归因方法的研究框架，从属性排名和得分分布两个维度分析各方法的异同，并考察这些差异是否反映在传统的保真度指标中。 |
| [^132] | [On the Design of Hyper-Connections: Simplification, Stability, and Scalability](https://arxiv.org/abs/2611.05984) | 该论文系统剖析了超连接（HC）架构，发现门控与残差混合可解耦为两个各自独立有效的交互组，并据此提出简化架构SimpleHC，通过移除残差混合和共享内存访问机制，在保持与mHC相当性能的同时显著降低训练开销。 |
| [^133] | [Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning](https://arxiv.org/abs/2611.05942) | 该论文提出“奖励预算化”这一插件式优势构造方法，通过为每个提示分配固定的总奖励预算来缩小优势差距，从而有效缓解大语言模型推理强化学习中的过早收敛与早期性能平台期问题。 |
| [^134] | [BAM: Bivariate Additive Models for Interpretable Tabular Prediction](https://arxiv.org/abs/2611.05940) | 提出 Bivariate Additive Models（BAM），将 GA²M 中的每个成对形状函数约束为非对称二元高斯的简单参数形式，在保持与标准 GAM 相当精度的同时，使每个交互项都可可视化为单峰等高线图并自动选择特征对，从而实现真正可解释、可审计的表格预测。 |
| [^135] | [Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification](https://arxiv.org/abs/2611.05937) | 本文提出FuseAMC轻量级决策融合模型，利用CNN在低信噪比下更鲁棒、高阶累积量分类器在高信噪比下更准确的互补特性，通过逻辑回归元组合器融合两者输出，实现了CPU高效的自动调制分类。 |
| [^136] | [Diversity-Aware Experience Replay via Statistical Leverage Scores](https://arxiv.org/abs/2611.05929) | 提出利用无需标签、奖励或价值网络的统计杠杆分数对经验回放缓冲区中的转移进行多样性感知采样，并针对独热动作编码导致方差评分失效的问题，采用基于峰度的特征列选择方法，从而优先采样稀有事件。 |
| [^137] | [Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models](https://arxiv.org/abs/2611.05919) | 该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。 |
| [^138] | [Improved approximation algorithms for capacitated cluster editing](https://arxiv.org/abs/2611.05916) | 本文针对带容量约束的聚类编辑问题，提出了一种缩减程序，将最佳近似比从 5.37 改进到 $\alpha_k = 2\alpha + 1$（约为 3.773）。 |
| [^139] | [Does Your Trial Simulator Know What the Drug Does? An Intervention-Blind Test of Interventional Validity on 7,207 Registered Trials](https://arxiv.org/abs/2611.05907) | 该论文首次将临床试验模拟器的“反事实能力”从主观论断转化为可测量的量化指标，通过构建隐去干预信息的干预盲参照模型，并基于7,207项注册试验构建的9,996个审计对比，将模拟器的干预技能定义为完整模型相对于干预盲参照所消除的误差。 |
| [^140] | [Retrieval Supersedes Scale: Efficient Log Level Prediction via Multiplex Socio-Technical Context](https://arxiv.org/abs/2611.05901) | 该论文提出基于多重社会技术上下文的检索方法来选择上下文学习示例，从而在不微调的情况下显著提升 LLM 的日志级别预测效果，证明了检索策略优于单纯依赖模型规模。 |
| [^141] | [Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol](https://arxiv.org/abs/2611.05895) | 该论文提出了一种信息匹配、防泄漏的对话情绪转变预测审计协议，通过明确的承诺点与可执行的因果性测试，揭示了可部署条件与先知式诊断条件会得出不同的评估结论。 |
| [^142] | [When Is One Query per Template Enough? Adapting Learned Cost Models to New Hardware](https://arxiv.org/abs/2611.05892) | 研究发现不同硬件间查询计划运行时间的变化主要由查询模板差异决定，提出一种为串行与并行计划分别拟合线性修正、并为每个测量模板添加偏移量的方法，使学习型代价模型在每个模板仅需一次测量查询的情况下即可适配到新硬件。 |
| [^143] | [MIGONet: a multiple-input graph neural operator with solver-free discrete manufactured-solution training](https://arxiv.org/abs/2611.05876) | 该论文提出了一种多输入图神经算子MIGONet，结合无需求解偏微分方程的离散制造解训练方法，能够捕获空间分布源项与狄利克雷/诺伊曼边界条件的联合影响，同时大幅降低训练数据生成成本。 |
| [^144] | [AI-Assisted Classification under Correlation Neglect and Trust](https://arxiv.org/abs/2611.05870) | 该论文提出包含相关性忽视与AI信任两个要素的人类决策行为模型，通过闭式相图划分AI辅助决策的互补、损害与自动化区域，并证明更强的AI可能因人类重复计算重叠证据而降低其决策准确性，而适当低水平的信任反而可以补偿相关性忽视。 |
| [^145] | [On Convergence of Gradient Temporal-Difference Learning with Regularized Corrections and Eligibility Traces](https://arxiv.org/abs/2611.05869) | 本文首次建立了带资格迹的正则化修正时序差分学习TDRC(λ)在非独立同分布离策略数据下的收敛性理论，并提出TDRC(λ)-Huber算法，在保持TDRC(λ)不动点的同时降低对大幅贝尔曼误差的敏感性。 |
| [^146] | [A low-rank decoder bottleneck bounds reliability in foundation-model perturbation prediction](https://arxiv.org/abs/2611.05849) | 该论文发现基础模型扰动预测中状态转移解码器的第一注意力层存在低秩瓶颈，其预测可靠性由预测响应与训练响应的距离所限定，并提出以“预测最近邻距离”作为逐条预测的置信度评分。 |
| [^147] | [PEARL: Prediction Error-Adaptive Regularized Learning for Selective Forgetting under Partial Preference Drift](https://arxiv.org/abs/2611.05842) | PEARL提出了一种预测误差自适应的正则化学习方法，通过仅在预测出错的特征方向上反转Gram矩阵更新来实现选择性遗忘，从而有效应对部分偏好漂移，克服了传统方法均匀遗忘和固定探索率的局限。 |
| [^148] | [RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection](https://arxiv.org/abs/2611.05838) | 提出RACE——一种无需真实执行的机器人策略检查点选择指标，通过基于状态条件阈值校准动作偏差并用累积乘积聚合评分，使检查点排序与真实部署性能更加一致。 |
| [^149] | [Pathwise-Budgeted LLM Routing: Online Allocation with K-Bit Tail-Cost Learning](https://arxiv.org/abs/2611.05835) | 本文提出上下文赌博机策略ADP-HT，通过学习各模型的任务特定成功概率，并利用K比特粗化回复长度标签构建保守成本估计，在固定预算下智能路由大语言模型调用与重试决策，以最大化在预算耗尽前完成的任务数量。 |
| [^150] | [Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework](https://arxiv.org/abs/2611.05828) | 本文从复杂适应系统视角揭示了智能制造中个体自主性与集体协调这两种耦合机制，并提出了一种将目标驱动自主代理与递归组织结构相结合的递归自主代理（RAA）框架，用以实现生产系统中的智能决策与组织。 |
| [^151] | [WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark](https://arxiv.org/abs/2611.05801) | WayPOP 是一个环视开放集全景分割跟踪基准，通过在真实 Waymo PVPS 序列中插入 Objaverse 的分类体系外 3D 物体并渲染多相机与 LiDAR 数据，检验未知物体能否在同步多相机序列中被一致地分割和跟踪。 |
| [^152] | [Item Representations Matter: Bridging Hard Semantic IDs and Soft Prototype Mixtures for Generative Retrieval](https://arxiv.org/abs/2611.05799) | 提出CAP-Mix流式推荐器，将每个物品表示为共享原型的软混合加上物品专属向量，在不引入离散瓶颈的前提下结合了硬语义ID的共享结构与连续表示的无损优势。 |
| [^153] | [ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models](https://arxiv.org/abs/2611.05793) | 该论文提出ZipDrift，通过训练一个向低闭环代价倾斜的一步漂移生成器来产生多模态的MPC代价修正候选，再由学习到的评论家排序选优，解决了确定性回归在多模态场景下平均各模态或继承专家不一致平局决策的问题。 |
| [^154] | [FlyHash for LLM Agent Memory: An Empirical Validation and a New Application](https://arxiv.org/abs/2611.05791) | 本文首次将受果蝇嗅觉回路启发的FlyHash稀疏哈希应用于大语言模型智能体的记忆检索，并通过严谨的匹配位协议实验实证验证了其相比SimHash的稳健质量优势。 |
| [^155] | [Unsupervised Neural Operators for Rotating Relativistic Stars: High-Fidelity Equation of State Inference and Pulsar Spin Evolution at Scale](https://arxiv.org/abs/2611.05790) | 该论文开发了物理信息驱动的无监督神经算子，能够高效求解旋转中子星的轴对称广义相对论流体静力平衡方程，从而实现高保真度的状态方程推断以及大规模的脉冲星自旋演化群体合成。 |
| [^156] | [Timing is Everything: Inferring Temporal Membership in Continually Updated Models](https://arxiv.org/abs/2611.05781) | 本文首次系统研究了持续更新模型中的时间隐私风险，提出了基于双向LSTM的黑盒攻击方法TMIA，能够以时间戳和区间两种粒度推断样本是否以及何时被用于模型训练。 |
| [^157] | [Optimizing Recourse Under Predictive Multiplicity](https://arxiv.org/abs/2611.05774) | 该论文提出ORPM算法，通过在带预算约束下最小化相互竞争的模型集合上的最坏情况有效性损失，解决了预测多重性下的鲁棒算法追索问题。 |
| [^158] | [Pairing Expert Critique with LLM-Driven Refinement: An Iterative Approach for Curating Quality AI Evaluation Datasets](https://arxiv.org/abs/2611.05749) | 提出SME-C人机协作工作流程，通过专家轻量级点评结合LLM驱动的分析与精炼，迭代式筛选高质量AI评估数据集与评分标准，并在金融商业简报领域收集411位专业人士的3,853条点评验证了两种LLM精炼策略的有效性。 |
| [^159] | [Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa](https://arxiv.org/abs/2611.05744) | 该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。 |
| [^160] | [Learning Hierarchical Patch Splitting Policies for Faster Vision Transformers](https://arxiv.org/abs/2611.05742) | 该论文提出SPLIT方法，通过学习依据模型认为有意义的任务相关内容来分配图像token分辨率，替代基于像素统计的启发式方法，在减少30%token数量的同时保持与全分辨率ViT相当的准确率。 |
| [^161] | [OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments](https://arxiv.org/abs/2611.05738) | 提出OdorSpaces——一个通过FDS模拟三维气味扩散来增强Habitat-Sim的物理仿真平台与基准，支持气体泄漏定位和早期火灾定位等视觉-嗅觉具身导航任务，其多模态强化学习策略优于多种基线方法。 |
| [^162] | [Self-Supervised Learning of Universal Latent Spaces for Operator Learning](https://arxiv.org/abs/2611.05735) | 该论文提出通过在即时合成、无标签无求解器的解析函数上自监督预训练函数自编码器，学习一个对平移、缩放和幅度精确等变的通用潜在空间，将其冻结后可在任意点云上跨任务复用于算子学习。 |
| [^163] | [RESHPart: Feedback-Reshaped Spectral Embedding for Hypergraph Partitioning](https://arxiv.org/abs/2611.05729) | 该论文提出FRESHPart框架，从谱嵌入几何出发，通过几何反馈和搜索反馈重塑谱嵌入，使其更有利于基于投影的超图划分，并在Titan23基准上相比hMETIS显著降低割值。 |
| [^164] | [Generative Modeling for Cascading Extremes via Autoregressive Transformers](https://arxiv.org/abs/2611.05728) | 该论文提出一种将多变量极端事件观测编码为分类标记序列的极值表示方法，并利用预训练的仅解码器自回归Transformer学习历史依赖的标记转移规律，从而实现对级联极端事件未来演化的生成式建模与预测。 |
| [^165] | [Do Language Models Know Which Constraints Belong Together?](https://arxiv.org/abs/2611.05726) | 该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。 |
| [^166] | [Give the Drafter a Second Chance to Guess: Parallel Reconsideration for Speculative Decoding](https://arxiv.org/abs/2611.05724) | 该论文提出“并行重审”方法，让投机解码中的草稿模型在自身不确定性引导下，通过与主草稿并行运行的额外前向计算重新考虑犹豫的猜测，使接受长度平均提升5.3%至10.2%，加速比平均提升3.5%至9.2%。 |
| [^167] | [CoReFlow: Consistency-Relaxed Flow Maps for Probabilistic Traffic Forecasting](https://arxiv.org/abs/2611.05715) | 提出CoReFlow模型，首次将流映射的单步生成能力应用于概率交通预测，解决了扩散和流模型多步采样推理慢且与交通数据分布复杂度不匹配的问题，并通过简洁的训练目标提升了预测精度。 |
| [^168] | [Agents, Think Only When it Helps](https://arxiv.org/abs/2611.05709) | 研究发现LLM智能体蒸馏中思想监督对最终行动表现影响甚微，仅当思想信息未在相应行动中披露时才变得重要，因此应根据信息披露情况灵活选择是否蒸馏思想，以提升效率。 |
| [^169] | [Neural networks and games](https://arxiv.org/abs/2611.05706) | 本文的核心贡献是建立了对称二值Hopfield神经网络与具有二次势函数的二行动博弈之间的精确等价关系，将神经网络稳定状态映射为纳什均衡、异步更新映射为最优反应动态、记忆存储与提取映射为均衡的设计与选择。 |
| [^170] | [Toward an Optimal Configuration for Tree Species Classification from Bark Images: A Leakage-Safe Ablation Study](https://arxiv.org/abs/2611.05680) | 本研究在树级别防泄漏的数据划分下，对基于树皮图像的树种深度分类进行了系统性消融实验，系统考察图像块大小、预测聚合策略、训练方式与骨干容量等因素，以确定最优模型配置并避免以往可能因数据泄漏而被高估的精度。 |
| [^171] | [Revisiting Critic Estimation in Soft Actor-Critic Algorithms](https://arxiv.org/abs/2611.05666) | 该论文在固定策略目标的前提下，系统性地重新审视并比较了SAC算法中六种critic设计变体（软优势vs软Q值、多步回报目标、在线或回放数据训练）在十四个任务上的表现。 |
| [^172] | [Adaptive Domain Decomposition for Finite Basis Physics-Informed Neural Networks](https://arxiv.org/abs/2611.05659) | 本文提出ADD-FBPINNs方法，将数据驱动的单位分解网络与有限基物理信息神经网络相结合，能够根据解的特征自动学习稳定且几何可解释的自适应域分解，在多个偏微分方程求解任务上相比固定域分解的FBPINNs获得更高精度。 |
| [^173] | [Beyond Accuracy: Information-Theoretic Decoder Evaluation Under Latent Intent](https://arxiv.org/abs/2611.05658) | 该论文将脑-机器人控制形式化为POMDP，证明了解码器性能存在由最差目标对切诺夫信息$\Gamma_{\min}$决定的相变边界——所有目标对可区分时遗憾值一致有界，否则线性增长，并据此提出闭式遗憾预测器，超越了传统的单步准确率评估指标。 |
| [^174] | [When Are Two SAE Latents the Same Feature?](https://arxiv.org/abs/2611.05655) | 该论文提出了一个基于能量加权最优传输的统一框架，用于比较稀疏自编码器的特征对应关系、字典差异与统计不确定性，并提供了渐近有效的联合置信推断方法。 |
| [^175] | [Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings](https://arxiv.org/abs/2611.05652) | 该论文提出用层次结构度量和kNN纯度量化EEG嵌入中特质与状态信息的编码方式，发现大多数EEG特征呈现“宏观特质聚类、微观状态嵌套其中”的层次结构。 |
| [^176] | [Data-Efficient Scaling-Law Discovery for Recommendation Systems via Coupled Hyperparameter Scaling](https://arxiv.org/abs/2611.05643) | 该论文提出通过缩放层将低维超超参数映射到完整架构空间，并结合缩放感知代理模型与帕累托前沿选择，仅需少量实验即可高效发现任意推荐排序模型的缩放定律。 |
| [^177] | [Spatio-Temporal Multi-Order Interaction for Video Recognition](https://arxiv.org/abs/2611.05633) | 该论文提出基于逐深度卷积的线性复杂度多阶时间建模与时空双重交互机制，以更低的计算开销实现更优的视频时空表示。 |
| [^178] | [LACS: Distribution-Sensitive Label-Agnostic Confidence Selection for Dynamic Data Pruning](https://arxiv.org/abs/2611.05616) | 提出了LACS，一种基于最大预测置信度、无需标签和逐样本梯度的动态数据剪枝方法，通过分布敏感的方式同时自适应调整阈值与保留样本数量，在七种评估设置中均验证了其有效性。 |
| [^179] | [On the Generalization Limitation of On-Policy Distillation for Post-RL Consolidation](https://arxiv.org/abs/2611.05609) | 在线策略蒸馏（OPD/MOPD）虽能近乎完美地复现教师模型的分布内性能，但无法完全迁移其泛化能力，分布外表现甚至低于基础模型，揭示了该方法在强化学习后能力巩固中的泛化局限。 |
| [^180] | [Transport along Local Annealing Paths: Wasserstein Geometry and Multimodal Sampling](https://arxiv.org/abs/2611.05605) | 该论文揭示了几何退火路径在Wasserstein空间中可能远长于端点距离从而拖慢传输映射学习，并提出分量退火传输（CAT），通过将目标精确分解为混合分量并沿各分量退火路径学习独立传输映射来减少模式间质量传输，缓解多模态采样中的模式坍塌。 |
| [^181] | [RIDGE: State-Conditioned Reward Blending for Behavioral Coverage in Deep RL Game Agents](https://arxiv.org/abs/2611.05601) | RIDGE 通过状态条件奖励混合器和多头评论家架构，让单个深度强化学习游戏智能体同时掌握多种行为风格，从而规避了 PPO 在非平稳奖励下的性能退化问题。 |
| [^182] | [Relevance Propagation as Entropic Optimal Transport on Filippov Neural Networks: Sinkhorn Propagation and Region Relevance in Image Classifiers](https://arxiv.org/abs/2611.05594) | 该论文证明神经网络中的相关性传播（LRP）本质上是熵最优传输变分问题的解，通过将前馈网络形式化为Filippov算子的有向无环图，统一了各种LRP规则并提出了权重可随节点活动变化的LRPN规则。 |
| [^183] | [GaussianRenderFormer: Transformer-based Rendering of 3D Gaussian Splats](https://arxiv.org/abs/2611.05593) | 提出GaussianRenderFormer，一个完全去除光栅化器、利用Transformer编码器-解码器结构从3D高斯泼溅和相机参数直接端到端渲染图像的神经网络，并通过诊断分析揭示了视图解码器中投影足迹推断的性能瓶颈。 |
| [^184] | [Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models](https://arxiv.org/abs/2611.05592) | 本研究通过在GraphCast和盘古气象大模型上对比门控与非门控优化方法，利用机器学习天气模型实现初始条件误差与模型误差的区分。 |
| [^185] | [C-GRPO: Conformal Group Relative Policy Optimization](https://arxiv.org/abs/2611.05591) | 提出C-GRPO，用经校准的保形阈值替代GRPO的固定采样预算，为自适应采样停止规则提供有限样本覆盖率保证，从而在不浪费计算的情况下按提示难度自适应分配采样资源。 |
| [^186] | [Exploring the Privacy--Size Tradeoff for Differentially Private Self-Attentive Variational Autoencoders](https://arxiv.org/abs/2611.05582) | 本文提出了SAVANT——首个在单一表格模型中结合注意力机制、变分自编码器与差分隐私的自编码器，并发现 在相同的差分隐私保证下，使用更多训练数据通常能带来更高的合成数据效用。 |
| [^187] | [Learning vs. Optimizing Players in Budgeted Games](https://arxiv.org/abs/2611.05580) | 该论文针对预算约束下的重复博弈提出了一个高效且预算可行的学习算法，能够同时实现不可操纵性——任何自适应优化者的期望收益至多为预算 Stackelberg 均衡值乘以 $T$ 加上 $O(\sqrt{T})$——以及当优化者采用固定混合策略时的低随机遗憾。 |
| [^188] | [Multimodal Fusion Methods for Whole-Slide Images and Clinical Data in Inflammatory Bowel Disease](https://arxiv.org/abs/2611.05579) | 该论文提出了一个模块化框架，在统一条件下系统比较15种多模态融合方法，将全切片图像嵌入与临床数据结合，用于预测炎症性肠病患者的缓解结局，结果显示最佳融合方法在多数场景下优于单模态模型。 |
| [^189] | [Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations](https://arxiv.org/abs/2611.05577) | 该研究通过预注册实验发现，自监督模型学到的流体表征虽然具有线性可解码的“可读性”，但并不围绕物理量组织——JEPA表现不优于未训练网络，而普通自编码器反而在最慢潜在方向上突出涡量耗散，且添加VICReg正则化会削弱这种物理结构。 |
| [^190] | [Role-based Effectiveness of an Epistemic Uncertainty Estimate in Training Models for Semantic Segmentation of 3D LiDAR Point Clouds](https://arxiv.org/abs/2611.05569) | 该论文研究将显式建模的认知不确定性以不同角色（作为额外输入特征或作为训练损失中的逐样本置信度权重）融入3D激光雷达点云语义分割模型训练中的有效性。 |
| [^191] | [Near-Minimax Sample Complexity for Multi-Objective Reinforcement Learning under Generative Models](https://arxiv.org/abs/2611.05562) | 本文建立了生成模型下多目标强化学习的常数置信度近极小极大样本复杂度 $\widetilde\Theta\!\left(\frac{SAH^3}{\varepsilon^2}\min\{S,\log(em)\}\right)$，证明单个数据集即可在不进一步采样的情况下同时支持所有奖励凸组合的 $\varepsilon$-最优策略。 |
| [^192] | [Support Vector Machines Directly Minimise KL Divergence](https://arxiv.org/abs/2611.05557) | 本文证明支持向量分类与回归的对偶目标等价于两个高斯测度之间的带约束KL散度最小化，从而为标准核机器建立了统一的信息论解释。 |
| [^193] | [FedSharp: Byzantine Robust Decentralized Federated Learning Under Noisy Channel](https://arxiv.org/abs/2611.05546) | 提出FedSharp框架，通过维护2b个前驱节点模型、坐标式中值偏差筛选可疑模型并对损失最低的b个候选模型取平均，在噪声信道下有效抵抗拜占庭攻击与信道噪声累积，实现鲁棒的去中心化联邦学习。 |
| [^194] | [A Unified Framework for Training-Dataset Aggregation and Exact Machine Unlearning via Gibbs Probability Measures](https://arxiv.org/abs/2611.05544) | 本文提出了Gibbs概率测度上重归一化、归一化对数线性组合与嵌套三种操作的统一理论框架，证明了嵌套与归一化对数线性组合的等价性，并据此实现了有监督机器学习中的训练数据集聚合与精确机器遗忘。 |
| [^195] | [Generalized Hyperrectangular Prediction Sets in Multivariate Conformal Prediction](https://arxiv.org/abs/2611.05543) | 该论文提出广义超矩形预测集及一个矩形化框架，能将任意连续的多变量非一致性分数或条件密度转化为具有有限样本覆盖保证、且易于解释和可视化的超矩形预测集。 |
| [^196] | [Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics](https://arxiv.org/abs/2611.05538) | 该研究证明，无需大规模脑电预训练的通用视觉语言模型能够像临床医生一样以视觉方式解读脑电信号，并在临床神经诊断任务上取得有竞争力的表现，从而为构建脑电图基础模型提供了摆脱对大规模脑电数据依赖的新范式。 |
| [^197] | [The Confidence Score: a New Measure on Kernels](https://arxiv.org/abs/2611.05528) | 本文提出了一种名为“置信度分数”的新核度量方法，通过衡量真实标签编码与核投影后真实标签之间的对齐程度，使保留类内结构的核也能获得高分，克服了传统理想核对齐度量要求同类样本必须完全相似的局限。 |
| [^198] | [Tight Lower Bounds under Matched High-Order H\"older Smoothness and Uniform Convexity](https://arxiv.org/abs/2611.05527) | 当高阶Hölder光滑阶数与一致凸阶数匹配（q = p + ν）时，任何确定性算法将初始最优性间隙Δ0降至ε都需要 Ω((H/σ)^(2/(3(p+ν)-2)) log(Δ0/ε)) 次oracle调用，该紧致下界与上界匹配，是Nesterov强凸函数经典下界向高阶一致凸情形的推广。 |
| [^199] | [TSFM Meets LLM: Context as Covariates](https://arxiv.org/abs/2611.05515) | 该论文提出 CoCo 预测框架，让 LLM 将文本推理转化为数值化的预测与置信度协变量来引导 TSFM 骨干模型，融合两者优势进行时间序列预测，并提供利用 LLM 标注真实时间序列以生成多模态数据集的可扩展方法。 |
| [^200] | [Ctrl-F: Force as an Interface for Humanoid Loco-Manipulation](https://arxiv.org/abs/2611.05507) | 提出了Ctrl-F分层运动-力框架，通过在指定接触部位施加显式的三维力指令来增强人形机器人运动跟踪，使机器人不仅能模仿人类动作，还能以指定力度与环境中的物体进行交互。 |
| [^201] | [Objective-Conditioned Active Preference Learning for Multi-Objective Alignment](https://arxiv.org/abs/2611.05506) | 提出 OGC-DPO 主动学习框架，将（回复对，目标）组合视为独立查询并利用策略自身隐式梯度衡量信息量，主动选择最能扩展各目标梯度覆盖范围的比较样本，从而在无需外部奖励模型的情况下降低多目标对齐的标注成本。 |
| [^202] | [Classical training of conditional instantaneous quantum polynomial-time Born machines](https://arxiv.org/abs/2611.05483) | 该论文提出将条件编码于输入态、跨条件共享单一IQP酉算子的条件IQP Born机，实现可经典无偏估计条件MMD目标与梯度的量子生成模型，且其表达能力超越了同等量子比特数下对角条件化的IQP模型。 |
| [^203] | [SO(3)-Equivariant Learning on CAD Boundary Representations](https://arxiv.org/abs/2611.05449) | SO3brep是一种作用于B-rep面邻接图的SO(3)等变图神经网络，通过Wigner D矩阵和Clebsch-Gordan张量积在消息传递中保留相邻面间的相对朝向，从而对随机旋转的CAD模型实现鲁棒的面分割与分类。 |
| [^204] | [SurgTransfer: Multi-Step Latent Action Modeling for Action Transfer in Surgical World Model](https://arxiv.org/abs/2611.05443) | 本文提出SurgTransfer框架，通过多步潜在动作建模从无标注手术视频中学习上下文不变的潜在动作，从而在无需真实运动学数据或专家标注的情况下实现手术世界模型中的专家动作迁移。 |
| [^205] | [Matched but Not Better: An Equivalence-Tested Benchmark of Quantum Machine Learning on Six Clinical Datasets and the QML-Health-READY Reporting Framework](https://arxiv.org/abs/2611.05442) | 该研究通过严格的配对等效性检验，在六个临床数据集上证明量子机器学习模型（QSVM和VQC）并不优于调优后的经典机器学习模型，并提出QML-Health-READY八领域报告框架以规范QML临床研究的报告标准。 |
| [^206] | [Learning Implicit Solves for Dynamic Simulations](https://arxiv.org/abs/2611.05439) | 提出隐式神经算子求解器INOS，仅从静态平衡解预训练神经算子并嵌入Newmark-β时间积分，无需动力学轨迹监督即可零样本实现高精度动力学仿真，并在非线性仿真中获得超过4600倍加速。 |
| [^207] | [HLA-Net: A Physics-Guided Skeleton-Based Framework for Automatic Gravitational Arc Detection in Astronomical Images](https://arxiv.org/abs/2611.05438) | 提出了一种无需人工标注数据的可解释多阶段框架HLA-Net，通过结合引力透镜的几何形态与物理特性，实现天文图像中引力弧的自动检测，并将候选搜索空间从169个缩减至83个。 |
| [^208] | [Constraint-Preserving Flow Matching for Spectral Graph Generation](https://arxiv.org/abs/2611.05436) | 提出了LaSFM方法，在谱图生成过程中通过在约束Stiefel流形上进行流匹配，同时保持拉普拉斯零空间约束、特征向量正交性和正特征值，并可扩展至属性引导的图生成。 |
| [^209] | [BIOTRAM: Biochemistry-informed Transformer for Relative Abundance via Metabarcoding](https://arxiv.org/abs/2611.05418) | BIOTRAM是一种融合生物化学知识的Transformer神经网络，通过学习每个标本的DNA贡献量和扩增效率来校正元条形码读段计数偏差，从而实现从批量样本中准确估计节肢动物各分类群的相对丰度。 |
| [^210] | [VenusGenome: Genome-Aware Masked Autoencoding for Microbial Genome Representation Learning](https://arxiv.org/abs/2611.05405) | 提出VenusGenome，一种基因分辨率的基因组感知掩码自编码器，通过高比例掩码与融合重叠群边界、链方向和基因位置的注意力机制，在超过170万个微生物基因组上预训练，以提升微生物基因组表示学习和功能预测能力。 |
| [^211] | [Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning](https://arxiv.org/abs/2611.05393) | 该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。 |
| [^212] | [Uncertainty-Aware Resource Orchestration for Agentic AI Systems](https://arxiv.org/abs/2611.05392) | 本文提出了一种不确定性感知的资源编排方法，解决大语言模型驱动的智能体AI系统因运行时资源需求呈重尾、多峰且不可预测而导致的调度难题，避免资源供给不足或过度供给。 |
| [^213] | [Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits](https://arxiv.org/abs/2611.05355) | 该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。 |
| [^214] | [Stabilizing On-Policy Targeted Guidance in Agentic Reinforcement Learning](https://arxiv.org/abs/2611.05266) | 提出了“定向引导”框架，通过在策略自蒸馏从每轮文本反馈中学习，并与按评分标准的强化学习奖励相结合，在不降低基线强化学习性能的前提下稳定地改进智能体环境中难以优化的行为。 |
| [^215] | [PAC Learning with Multiple Correct Answers](https://arxiv.org/abs/2611.05133) | 该论文研究了每个样本允许多个正确标签的PAC学习问题，证明了示范反馈可能比完整标签集合甚至bandit反馈学习更困难，并用广义Natarajan维度刻画了前两种反馈下的可学习性，用假设交集的可学习性刻画了示范学习下的可学习性。 |
| [^216] | [FedCSGA: Evolutionary Client Selection with Joint Statistical and System Heterogeneity in Federated Learning](https://arxiv.org/abs/2611.04667) | 提出了一种基于遗传算法的联邦学习客户端选择方法FedCSGA，通过联合建模统计异构性与系统异构性的非线性适应度函数，实现每轮客户端子集的进化式优化选择。 |
| [^217] | [Sample-Grained Approximate Unlearning with Provable Per-Sample Bounds](https://arxiv.org/abs/2611.04501) | 该论文提出SALA即插即用优化框架和样本粒度分布差距指标，使近似机器遗忘方法能够在逐样本层面工作，并提供了可证明的逐样本遗忘边界。 |
| [^218] | [Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs](https://arxiv.org/abs/2611.04376) | 该研究揭示了MoE大模型FP4强化学习失效的根源在于路由器的离散专家选择机制，证明了必须在训练与推理两侧对称地应用相同的恢复技术，才能实现有效的端到端FP4强化学习。 |
| [^219] | [Separating Common and Unique Directions for Model Merging](https://arxiv.org/abs/2611.04108) | 提出CUSD，一种无需数据的模型合并预处理方法，通过将任务向量分解为跨任务共享的公共方向分量和任务独有的独立分量并分别处理，从而提升算术合并方法的性能。 |
| [^220] | [The Complexity of ReLU Network Synthesis from a Parameter Multiset](https://arxiv.org/abs/2611.04074) | 本文研究了在参数数值与网络架构分离（即参数值预先给定）的条件下合成ReLU网络的计算复杂性，证明精确匹配标注样例为NP完全、在所有布尔输入上匹配布尔电路为Σ₂^P完全（四个神经元即可）、以及在所有实数输入上匹配给定ReLU网络为DP完全（三个神经元时）。 |
| [^221] | [All Paths All at Once: A Quantum Monte Carlo and Neural Committor Approach to Reaction Pathway Search](https://arxiv.org/abs/2611.03699) | 该论文提出一种无需势能梯度的仅能量反应路径搜索方法tReax_expedition，通过耦合蒙特卡洛探索、随机淬火标记与受限延续自动恢复中间体和竞争反应通道，无需过渡态猜测即可将链最大值能量精确到精修鞍点0.76 kJ/mol以内，并借助神经commitor学习将路径与过渡区域采样相连接。 |
| [^222] | [Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap](https://arxiv.org/abs/2611.03633) | 本文发现预训练数据规模会逆转Muon与AdamW优化器的微调性能差距——当预训练数据足够大时，微调后AdamW反而优于Muon，并提出了“边际-散度比”指标来解释这一逆转现象。 |
| [^223] | [Thought-deciphering via hierarchical EEG-to-text alignment](https://arxiv.org/abs/2611.03623) | 提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。 |
| [^224] | [Neural Actor--Critic Learning with Minimax Critics and Reflected Langevin Sampling](https://arxiv.org/abs/2611.03352) | 该论文提出一种结合正则化极小极大贝尔曼评论家估计与反射朗之万采样的神经演员-评论家算法，并给出了达到非参数极小极大最优速率的有限样本价值误差界与策略回报保证。 |
| [^225] | [Restrained Diffusion Bridges: Targeted Denoiser Training and Refinement](https://arxiv.org/abs/2611.02170) | 受限扩散桥方法通过在选定的中间时刻以受限构型为条件进行前向加噪，使受限采样引入的偏差被精确抵消，从而让模型在目标分布中任意关键的数据稀疏区域都能高精度生成，无需重要性权重。 |
| [^226] | [Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls](https://arxiv.org/abs/2611.02119) | 本研究构建了基于扩散模型的条件反事实审计框架，并通过匹配的阳性对照严格验证了其对黑色素瘤分类器概念级解释的审计能力，证明该方法在恢复目标概念方面优于传统的概念激活向量方法。 |
| [^227] | [NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale](https://arxiv.org/abs/2611.02102) | NorthCUA 构建了经验飞轮驱动的集成系统，通过大规模验证轨迹积累、递归自我改进和分阶段学习（中期训练、长时程监督微调与桌面-网页联合强化学习），显著提升计算机使用智能体完成复杂软件任务的能力。 |
| [^228] | [Wave Physics as Representation: Tensor-Structured Neural Reconstruction for Ocean Acoustic Fields](https://arxiv.org/abs/2611.01680) | 该论文提出从波动物理中提取结构化表示并用张量结构神经网络学习其组件的新路线，实现了既具物理可解释性、又对不确定环境信息鲁棒的海洋三维声场稀疏重建。 |
| [^229] | [Behavior Graphs: Unsupervised Graph Structures for Interpreting Robot Deployment Data](https://arxiv.org/abs/2611.00405) | 该论文提出“行为图”这一无监督方法，将机器人策略rollout数据压缩为一张单一、可解释的有向图，通过识别相似行为模式、按时序连接并与结果关联，显著降低了部署数据的分析难度，使研究者能更高效地指导数据收集决策。 |
| [^230] | [TensorSSM: Memorizing Spatial Position in Dynamic State Space for Tensor Core Generation](https://arxiv.org/abs/2610.40871) | 提出TensorSSM，一种在动态状态空间模型（SSM）中记忆空间位置的新型隐式神经表示函数式张量分解方法，通过K叉规则从粗到细编码空间标记，克服了现有方法因展平高阶张量为一维序列而导致的表达能力受限问题。 |
| [^231] | [Probabilistic Meshless Methods on Manifolds](https://arxiv.org/abs/2610.39855) | 该论文将概率无网格方法首次扩展到紧流形并给出其收敛性分析，在拉普拉斯-贝尔特拉米特征对已知的情形下，采用点云上的广义有限差分离散化PDE算子并证明方法在采样点处的一致收敛性。 |
| [^232] | [Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection](https://arxiv.org/abs/2610.39076) | Tactus是一种织物集成的感知手套，通过超薄3D针织外形、策略性布置的电容触觉传感器和融合IMU的抗遮挡姿态跟踪三大设计，实现了自然人类操作演示中精确接触力与手部姿态的高质量数据采集。 |
| [^233] | [Communication-Efficient Clustered Federated Learning in Non-IID Wireless Networks](https://arxiv.org/abs/2610.38970) | 提出了一种通信高效的聚类联邦学习框架，通过数据域聚类初始化基于模型的聚类，在不共享原始数据的前提下识别客户端相似性并为其分配专门模型，从而解决无线网络中的非独立同分布数据挑战。 |
| [^234] | [PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations](https://arxiv.org/abs/2610.37381) | PDRVLA 提出一种轻量级物理动力学模块，通过预测未来 RGB、深度和光流特征来学习物理表征，在无需在线视频生成的情况下显著提升了视觉-语言-动作模型的机器人操作性能。 |
| [^235] | [A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis](https://arxiv.org/abs/2610.35658) | 该论文提出一种混合框架，利用本地大语言模型从5.6万余份美国航空事故报告中提取因果链并构建因果知识图谱，结合基于物理的失效发生模型与漏泄噪声OR方法，将文本因素量化为失效模式概率，实现可解释的航空风险量化与干预分析，其提取的因果链可恢复90.8%的调查人员标注原因。 |
| [^236] | [Last-Iterate Stationarity of Muon with Finitely Many Newton-Schulz Iterations](https://arxiv.org/abs/2610.34158) | 本文首次为采用有限次Newton-Schulz迭代的实用Muon优化器建立了末次迭代梯度收敛保证，证明在正稳定化和小常数学习率下可实现非凸目标的末次迭代平稳性，而无稳定化时即使在凸二次函数上也会循环。 |
| [^237] | [Reactive Optimizers](https://arxiv.org/abs/2610.33826) | 本文提出OPAL，一种利用预训练语言模型智能体在单次训练中实时检查训练动态并重写优化器代码的反应式优化器，在未经调优的设置下可匹配或超越固定优化器配方。 |
| [^238] | [Entity-Aware Evaluation of OCR Systems for Historical Document Transcription](https://arxiv.org/abs/2610.33004) | 该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。 |
| [^239] | [Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion](https://arxiv.org/abs/2610.31323) | 提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。 |
| [^240] | [Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training](https://arxiv.org/abs/2610.30792) | Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。 |
| [^241] | [Asymmetric On-Policy Distillation Learns In-Context Exploration](https://arxiv.org/abs/2610.30105) | 提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。 |
| [^242] | [Interpretable Cutting Plane Selection for Mixed-Integer Linear Programming Using Genetic Programming](https://arxiv.org/abs/2610.29311) | 本文提出首个基于遗传规划的完全可解释割平面选择框架GPSCS，通过演化符号化评分函数将手工启发式策略显式泛化为可解释的自适应割平面选择策略，弥补了神经网络方法可解释性差且缺乏收敛保证的不足。 |
| [^243] | [From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake](https://arxiv.org/abs/2610.29152) | 该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。 |
| [^244] | [A latent control model for realistic rodent motion](https://arxiv.org/abs/2610.27279) | 提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。 |
| [^245] | [FrameScout: Scouting Query-Relevant Frames for Long Video Understanding](https://arxiv.org/abs/2610.23975) | 提出FrameScout框架，通过结合滑动窗口KV缓存与后继聚合的流式选择器架构生成时间上下文化的帧嵌入，并利用帧-查询对比学习目标直接训练选择器，从而为长视频理解高效选取与查询最相关的关键帧。 |
| [^246] | [Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss](https://arxiv.org/abs/2610.22454) | 研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。 |
| [^247] | [FutureLift: Lifting Trajectory Proposals into Bidirectional Future Graphs for Motion Forecasting](https://arxiv.org/abs/2610.22215) | FutureLift提出了一种与主干网络无关的轨迹提议改进方法，将多模态提议表示为带有双向时间边的异构未来图，使后续证据能够在单次遍历中修正较早的片段，从而提升自动驾驶运动预测的准确性。 |
| [^248] | [Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization](https://arxiv.org/abs/2610.21742) | 本文提出基于Smooth-L1正则化的稀疏回声状态网络SL1ESN，利用与牛顿法兼容的可微稀疏化方法实现读出层高度稀疏，并支持二阶联邦聚合，在七个时间序列分类基准上达到75%–95%的稀疏度。 |
| [^249] | [Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference](https://arxiv.org/abs/2610.21257) | 该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。 |
| [^250] | [Revisiting the 16x Pretraining Compute Gap between Diffusion and Autoregressive Language Models](https://arxiv.org/abs/2610.17590) | 本研究采用现代扩散语言模型训练实践（精确似然评估、块扩散训练、编码器-解码器架构）重新审视扩散语言模型的缩放定律，通过约1700个模型的大规模实验发现，扩散语言模型与自回归语言模型之间所谓的16倍预训练计算差距远小于此前声称的水平。 |
| [^251] | [Neuromorphic Decoding and Dynamical Image Reconstruction from Populations of Spiking Retinal Ganglion Cells](https://arxiv.org/abs/2610.17154) | 该研究构建了端到端的脉冲神经网络，首次实现了完全在脉冲域内对视网膜神经节细胞群体活动的解码与动态场景的实时追踪重建，且计算开销极低。 |
| [^252] | [DTA-GT: Direction- and Topology-Aware Graph Transformer for Neural Network Representation Learning](https://arxiv.org/abs/2610.16822) | 提出DTA-GT，一种方向与拓扑感知的图Transformer，通过在节点初始化、谱编码、注意力交互和特征更新全流程中保持有向计算语义，显著提升了神经架构搜索中精度与延迟预测的性能。 |
| [^253] | [Product Monte Carlo: Expectation Under a Product of Densities](https://arxiv.org/abs/2610.12702) | 该论文提出乘积蒙特卡罗（PMC）算法，通过将难以处理的密度乘积期望转化为全贝叶斯回归问题，仅利用各分量密度的蒙特卡罗样本即可计算任意期望，并具备一致性保证与不确定性量化能力。 |
| [^254] | [Multi-Agent Coordination via Support-Preserving Distillation](https://arxiv.org/abs/2610.10087) | 提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。 |
| [^255] | [Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence](https://arxiv.org/abs/2610.07739) | 该论文提出了BAR框架，首次将成本受限的KG探索、按来源质量区分的可验证证据以及可引用的推理依据三者结合，利用LLM在医疗知识图谱上进行推理，以弥补EHR中缺失的依赖关系并提升出院后风险预测的可靠性。 |
| [^256] | [Backward-Consistent Diffusion Sampling for Sparsely Observed PDE Inverse Problems](https://arxiv.org/abs/2610.04624) | 该论文提出FunBCS——一种输入空间优化方法，通过寻找使去噪器重构物理一致的最佳输入，证明了其在不连续PDE逆问题中能够正确最小化未观测误差，而现有输出空间方法则无法做到。 |
| [^257] | [Dynamic Quadtree Tokenization and Transformer for Adaptive Mesh PDE Forecasting](https://arxiv.org/abs/2610.04044) | 提出WAMRViT，一种基于动态四叉树标记化与小波细化判据的视觉Transformer，无需预设标记数量即可自适应调整网格分辨率，实现对细尺度PDE的长时程稳定预测。 |
| [^258] | [S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks](https://arxiv.org/abs/2610.03303) | 提出S²-PINN，利用可学习高斯空间字典、傅里叶时间特征与gPC随机基并通过低秩CP张量分解耦合的可分离表示，高效求解随机偏微分方程的不确定性量化问题，克服了经典谱方法的维数灾难。 |
| [^259] | [PatchKV: Weight-Space Compensation of KV Cache](https://arxiv.org/abs/2609.39329) | PatchKV是一种无需训练的框架，通过岭回归以闭式解将上下文特定的权重补丁合并进模型权重，以补偿KV缓存压缩带来的信息损失，且不增加每次查询的推理成本。 |
| [^260] | [Uniform Race: Parameter-Free Approximate Rejection Sampling](https://arxiv.org/abs/2609.34639) | 该论文提出了一种无需任何参数调节的“均匀竞赛（Uniform Race）”近似拒绝采样算法，通过将重要性权重除以独立均匀随机变量构造得分并选取样本，在不知道理想接受阈值的情况下自动达到最优的拒绝采样全变差误差保证。 |
| [^261] | [Understanding Generalization Requires Universal Induction](https://arxiv.org/abs/2609.34458) | 论文指出经典统计理论因依赖无法证成的手工归纳偏置而无法解释通用AI的泛化成功，并提出将所罗门诺夫归纳相对化到信息视角、偏向能访问所有既有信息的短程序，从而以普适归纳重新框架化归纳偏置问题。 |
| [^262] | [From Grey-Box to Green-Box: When can Physics-Informed Machine Learning Reduce Carbon Footprints in Structural Health Monitoring?](https://arxiv.org/abs/2609.33387) | 本文探讨物理信息机器学习（灰箱模型）能否在结构健康监测中降低碳足迹，通过评估残差建模、输入增广、混合建模和约束学习四种方法，将训练数据量、训练时间与计算碳排放联系起来。 |
| [^263] | [Schr\"odinger--F\"ollmer Actor--Critic: Diffusion Policy Improvement with Finite-Sample Analysis](https://arxiv.org/abs/2609.33239) | 该论文提出SFAC方法，利用极小极大贝尔曼评论家、Doob h-变换以及基于布朗桥端点的加权回归，在条件扩散框架中实现KL正则化的策略改进，并给出有限样本理论分析。 |
| [^264] | [What Should Data Teach? Moving Bottlenecks Across Circuit, Store, and Use](https://arxiv.org/abs/2609.32991) | 该论文提出以电路视角诊断语言模型训练中的三类数据瓶颈（计算形成、内容可用、路径选择），并给出统一的“诊断到数据”原则，通过形成敏感选择、可用性反事实与配对监督等针对性干预，显著提升了事实学习与长上下文问答能力。 |
| [^265] | [Robust Successor Features](https://arxiv.org/abs/2609.31016) | 该论文提出了鲁棒后继特征，将强化学习中的迁移学习与鲁棒强化学习两种范式统一起来，使智能体在线性马尔可夫决策过程假设下，能够同时在奖励函数和未知转移核两个维度上进行泛化。 |
| [^266] | [Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems](https://arxiv.org/abs/2609.30809) | 本文利用PINN和DeepONet求解无穷大Laplace与p-Laplace问题，在大p值（2至1000）和三维区域上优于传统网格求解器，并建立了条件收敛性与通用逼近的理论结果。 |
| [^267] | [Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning](https://arxiv.org/abs/2609.29711) | 该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。 |
| [^268] | [MMAP: Multimodal Missing-Aware Pretraining for Longitudinal Alzheimer's Prediction](https://arxiv.org/abs/2609.26617) | 提出了MMAP方法，一种多模态缺失感知的对齐预训练框架，通过sigmoid对比学习与生成式重建从不完整的图像和表格数据中学习联合表示，用于阿尔茨海默病的纵向进展预测。 |
| [^269] | [Efficient Robust Learning at the Information-Theoretic Limit](https://arxiv.org/abs/2609.17655) | 本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。 |
| [^270] | [Robust Policy Optimization via Adversarial Importance Sampling](https://arxiv.org/abs/2609.13044) | 该论文提出了Advis方法，通过对标准训练轨迹进行重要性采样来估计和优化可验证的最坏情况回报，无需额外环境交互和辅助网络即可捕获长期鲁棒性，同时提供模块化PyTorch库advrl以促进鲁棒DRL方法的快速开发与可复现评估。 |
| [^271] | [Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training](https://arxiv.org/abs/2609.07108) | 该论文提出了一个端到端系统，通过在上下文并行中融合分支注意力与主序列注意力、并引入TapChannel独立通道跨流水线阶段传输目标特征，实现了大规模长上下文强化学习后训练中投机解码草稿模型的在线协同训练，从而显著加速rollout生成。 |
| [^272] | [LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues](https://arxiv.org/abs/2609.03507) | 提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。 |
| [^273] | [A hybrid quantum-classical neural network for learning to route](https://arxiv.org/abs/2609.00489) | 本研究提出用小型量子神经网络替换基于注意力的路由模型中的编码器前馈模块，在带容量约束车辆路径问题上将参数减少56.6%的同时保持接近经典神经基线的求解质量，为神经组合优化提供了一种可行的混合模块压缩策略。 |
| [^274] | [Algorithmic Principles For Multiclass Learning Are Hard To Come By: Limits of Regularization and Proper Learning](https://arxiv.org/abs/2608.26516) | 本文证明多类学习不能简化为恰当学习或正则化，解决了三个开放问题，并展示了一个不可嵌入任何恰当框架的可学习多类问题。 |
| [^275] | [Continuous Adversarial MeanFlow Transfer](https://arxiv.org/abs/2608.19540) | 该论文提出MeanFlow-Transfer方法，通过统一速度表示和对抗训练，同时解决跨领域适应和生成器加速问题，使异构预训练模型能高效迁移到新领域。 |
| [^276] | [Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization](https://arxiv.org/abs/2608.11746) | 本文提出利用Epiplexity作为在线训练信号，通过拟合缩放定律预测数据域的Epiplexity增益，自适应调整采样权重，并用于指导合成数据生成，以提升模型在未预见任务上的分布外泛化能力。 |
| [^277] | [Do Tabular Foundation Models Agree with Themselves?](https://arxiv.org/abs/2608.06004) | 本文首次系统评估表格基础模型通过自回归采样构建多变量预测的机制能否诱导出有效的随机过程，提出了边缘化一致性和因子分解一致性这两个可检验的要求，以回答“以两种不同方式提问同一问题是否会得到相同答案”这一核心问题。 |
| [^278] | [EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement](https://arxiv.org/abs/2608.04968) | 提出 EvolveNet 框架，通过客户端并行演化同一执行框架、服务器聚合各客户端适配并重新分发的协作机制，在不更新模型权重的情况下实现智能体的持续自我改进与跨用户能力积累。 |
| [^279] | [Sedentary Behavior Classification for Wearable Sensors with a CNN-BiLSTM Model](https://arxiv.org/abs/2608.02946) | 提出通过对CNN-BiLSTM模型CHAP进行微调，将基于髋部加速度计训练的久坐行为分类模型成功迁移到手腕可穿戴传感器数据上，并持续优于从头训练的Transformer模型。 |
| [^280] | [DP-IVON-Gradsq: Differentially Private Squared-Gradient Improved Variational Online Newton](https://arxiv.org/abs/2607.23649) | 提出差分隐私优化器DP-IVON-Gradsq，通过噪声校正的平方梯度估计器从私有化梯度构建曲率估计，在保持IVON计算效率的同时减少了后验采样噪声与隐私噪声的相互干扰，实现了差分隐私下的变分贝叶斯深度学习。 |
| [^281] | [Unbiased Open World Regularization for Fair Self-Supervised Learning](https://arxiv.org/abs/2607.22149) | 该论文提出无偏开放世界（UOW）正则化方法，通过将每个偏差组内的表示分布匹配到共同目标分布来保证表示与偏差属性的独立性，从而为自监督学习实现更有效的去偏与公平性。 |
| [^282] | [GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries](https://arxiv.org/abs/2607.20757) | GaugeQuant通过在训练中向损失函数引入LogSumExp项打破Transformer的内部连续对称性，在线学习出最小化激活离群值的量化最优基，无需校准数据和量化模拟即可显著提升低比特量化效果。 |
| [^283] | [OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories](https://arxiv.org/abs/2607.17082) | 该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。 |
| [^284] | [Understanding Reasoning from Pretraining to Post-Training](https://arxiv.org/abs/2607.16097) | 该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。 |
| [^285] | [Revisiting data-driven dynamic security assessment with a tabular foundation model](https://arxiv.org/abs/2607.16031) | 该论文提出使用表格基础模型（TFM）通过上下文学习进行电力系统动态安全评估，仅需单一模型即可同时评估多个偶然事件，无需逐个偶然事件训练分类器或进行超参数优化，并可泛化到未见过的偶然事件。 |
| [^286] | [Causal Inference for Sequential Settings under Interference and Latent Confounding](https://arxiv.org/abs/2607.14940) | 该论文提出一种基于最大伪似然估计的计算高效方法，用于在存在结果干扰和低秩潜在混杂因素的序贯观测设置中估计因果效应，并证明了参数估计的非渐近一致性及其向因果量的忠实估计转化。 |
| [^287] | [Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation](https://arxiv.org/abs/2607.13686) | 本文提出SquareCB.Comb算法，为通用函数逼近下的上下文组合半赌博机问题首次实现了极小极大最优的遗憾界 $O(\sqrt{mAT\log|\mathcal{F}|})$。 |
| [^288] | [TTHE: Test-Time Harness Evolution](https://arxiv.org/abs/2607.08124) | TTHE提出在测试阶段利用无标签执行轨迹持续演化LLM智能体的脚手架程序，通过智能体提议者生成多样化候选并由裁判筛选，从而在无需标签的情况下实现对任务分布与失败模式变化的实时自适应。 |
| [^289] | [Distributed Sketching on Data Partitions for OLS Regression](https://arxiv.org/abs/2607.07888) | 本文提出从数据分区子集生成草图的新型分布式OLS回归方法，使草图计算成本随工作节点增多而下降，并刻画了全局估计器的精确超额损失，证明当子集协方差间平均Burg散度较小时，其性能接近基于全数据草图的方法。 |
| [^290] | [Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories](https://arxiv.org/abs/2607.06648) | 该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。 |
| [^291] | [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369) | 本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。 |
| [^292] | [Hyperparameter Transfer in Graph Neural Networks](https://arxiv.org/abs/2607.05017) | 本文为使用SGD、Adam和AdamW训练的图神经网络提出并验证了一种超参数迁移参数化方法，使模型在宽度和深度增加时能够实现稳定的特征更新、学习率迁移及性能提升。 |
| [^293] | [PixCon: Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation](https://arxiv.org/abs/2607.03068) | PixCon 通过添加一个仅存储预测与真实标签一致处的干净有标签特征、无需置信度阈值的类别级记忆库，并以多正样本对比损失训练编码器-解码器，从而以极简方式显著提升了基于基础模型的半监督分割性能。 |
| [^294] | [AbsoluteDegradation: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark](https://arxiv.org/abs/2607.02131) | 提出了受物理启发的模块化胶片退化合成流水线AbsoluteDegradation及新的大规模档案胶片修复基准，以解决档案胶片修复中缺乏成对训练数据和标准化评估基准的双重难题。 |
| [^295] | [Warp: Structured Output-Space Adaptation for Frozen Robot Policies](https://arxiv.org/abs/2606.31043) | 提出了Warp，一种上下文条件化的单调双射样条适配器，能在不修改冻结机器人策略的前提下对其输出进行保序的非线性重整形，从而在受控部署偏移的ManiSkill任务中实现最强的性能恢复。 |
| [^296] | [OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL](https://arxiv.org/abs/2606.30356) | OLIVE是首个在单一编码器内通过功能分离联合掩码潜在预测与波形重建的语音自监督学习框架，在保持语音识别等判别任务竞争力的同时，可直接重建波形并提升说话人任务表现。 |
| [^297] | [Kriging and Neural Network models for pressure losses across perforated plates](https://arxiv.org/abs/2606.29628) | 本文提出基于克里金和神经网络的两种数据驱动模型，在预测湍流中圆形穿孔板压力损失方面始终优于传统经验公式，即使训练数据有限也展现出良好的准确性和应用前景。 |
| [^298] | [Batch-Invariant Spectral Intelligence for Robust and Explainable Insect Authentication](https://arxiv.org/abs/2606.26757) | 提出了一个端到端的批次不变光谱网络（BISN），通过结合可学习的预处理模块和熵正则化对抗目标，有效抑制了光谱测量中的批次间变异，从而在未见过的生产批次上实现了鲁棒且可解释的昆虫物种认证。 |
| [^299] | [Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting](https://arxiv.org/abs/2606.26421) | 本文提出Otter Weather模型，通过高效时空架构实现中期天气预报，在1.5°分辨率下以极低计算成本（<3.5 A100天）超越传统NWP 9.6%，效率比轻量AI模型高2倍，比资源密集型模型低100倍，推动了天气预测的民主化。 |
| [^300] | [AutoDex: An Automated Real-World System for Dexterous Grasping Data Collection](https://arxiv.org/abs/2606.23689) | AutoDex是一个无需持续人工监督的自动化真实世界灵巧抓取数据收集系统，通过多视角物体跟踪、残余力矩碰撞监测和姿态导向的自动重置，在100个物体上成功收集了超过3,500次包含成功与失败案例的抓取试验数据。 |
| [^301] | [TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization](https://arxiv.org/abs/2606.23496) | TROPT是首个开源框架，通过单一接口统一了离散文本优化器的执行并标准化其开发流程，解决了现有优化器分散、难以使用和难以对比的问题，降低了在红队测试、审计和可解释性等领域应用与推进离散文本优化的门槛。 |
| [^302] | [Phantoms and Disclosures: A Statistical Framework for Auditing Privacy in Synthetic Data](https://arxiv.org/abs/2606.16952) | 本文提出了一个可定制的实证审计框架，通过区分“真实泄露”与“幻影泄露”，并利用训练/保留集划分与严格的统计假设检验，来审计合成数据是否符合零学习或差分隐私等严格隐私基线。 |
| [^303] | [Decomposing one-class support vector machine into an ensemble of one-data support vector machines](https://arxiv.org/abs/2606.16002) | 本文提出将单类支持向量机分解为针对单个数据点的模型并利用集成学习重组，结合基于样本均值训练模型的支持向量选择策略，在保持相似分类效果的同时显著加速训练，并实现样本与模型的一一对应。 |
| [^304] | [Provably Safe, Yet Scalable Reinforcement Learning](https://arxiv.org/abs/2606.14536) | PS2-RL框架通过两阶段架构和学习的备份策略进行前向积分来隐式构造安全保证，在提供严格安全可证明性的同时，实现了随状态维度良好扩展的安全强化学习。 |
| [^305] | [Machine-learned particle flow as a foundation model for collider physics](https://arxiv.org/abs/2606.14373) | 该论文证明将在事件重建（MLPF）中学到的逐粒子潜在表示作为共享的基础模型表示，可显著提升喷注味道识别、喷注能量回归和丢失动量回归等多种对撞机物理分析任务的性能。 |
| [^306] | [Learning-Augmented Approximation for Unrelated-Machines Makespan Scheduling](https://arxiv.org/abs/2606.13133) | 本文回应了Antoniadis等人提出的开放性问题，将学习增强近似框架从选择问题推广到调度领域，针对不相关机器完工时间最小化问题提出了利用重任务分配预测的算法，在预测准确时达到(1+ε)-近似，并随误差增大平滑退化为最坏情况2-近似。 |
| [^307] | [Flow Matching with In-Context Priors for Out-of-Distribution Brain Dynamics](https://arxiv.org/abs/2606.11833) | 提出PINC-Flow，一种通过在上下文中注入组合式语言和空间先验的逐时间步条件化扩散Transformer，可零样本生成未见认知任务下真实的fMRI脑动态，为反事实神经科学中的计算机内实验设计与评估开辟了道路。 |
| [^308] | [Teaching Diffusion to Speculate Left-to-Right](https://arxiv.org/abs/2606.11552) | 该论文提出让扩散草稿模型以从左到右的方式生成推测token，从而弥合块扩散草稿的双向生成与自回归目标模型严格左到右验证之间的不对称差距，提升推测解码的效率。 |
| [^309] | [Neural networks become centroid classifiers when trained on heavily corrupted inputs](https://arxiv.org/abs/2606.11319) | 该研究发现神经网络对超过90%的输入损坏仍保持远高于随机水平的准确率，并通过均值场理论分析证明，在重度损坏情形下无限宽神经网络会退化为最近质心（原型）分类器。 |
| [^310] | [Driving Video Retrieval for Complex Queries with Structured Grounding](https://arxiv.org/abs/2606.09109) | 提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。 |
| [^311] | [Diffuse AI Control on Fuzzy Tasks](https://arxiv.org/abs/2606.08892) | 本文提出将AI控制建模为蓝队与红队的对抗博弈框架，用以研究和防御AI在难以评分的模糊任务上进行长期、分散式暗中破坏的风险。 |
| [^312] | [Large-scale empirical tuning and comparison of default optimizers for variational inference](https://arxiv.org/abs/2606.07841) | 本文通过对56种随机梯度优化算法在1092个贝叶斯推断问题上进行超55万次优化运行的大规模实证评估，系统比较了无调参自适应优化器在黑箱变分推断中的表现，确立了无调参优化推断的最新技术水平。 |
| [^313] | [Anchor PCA](https://arxiv.org/abs/2606.06233) | Anchor PCA通过在总体解释方差与共享和领域特定低秩嵌入一致性之间进行权衡，为多领域数据找到鲁棒的共享降维嵌入，且可通过修改后的目标矩阵上的PCA高效求解。 |
| [^314] | [Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors](https://arxiv.org/abs/2606.06179) | 该论文从几何视角研究扩散模型的得分匹配误差，证明只有误差中可观测的梯度分量才影响采样质量（L² 得分误差即使任意大，模型仍可完美匹配目标分布），并据此推导出仅依赖该梯度分量的更紧 KL 散度上界。 |
| [^315] | [SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter](https://arxiv.org/abs/2606.05800) | 论文揭示了GRPO式组归一化中增加rollout数量会因策略梯度特征聚集成低秩带符号几何而相互抵消，并提出SALT插件，通过样本级梯度几何将组相对系数分解为共享与残差通道并自适应放大残差通道，使更多rollout真正发挥作用。 |
| [^316] | [Denoise First, Orthogonalize Later: Understanding Momentum in Muon via Spectral Filtering](https://arxiv.org/abs/2606.03899) | 该论文证明了Muon优化器中的动量本质上是一个谱滤波器——“先去噪、后正交化”的顺序能抑制梯度扰动、扩大谱间隙，从而稳定奇异子空间并使更新更可靠。 |
| [^317] | [Emergent Transfer of a Physics Foundation Model from Simulation to Laboratory Turbulence](https://arxiv.org/abs/2606.01470) | 该论文验证了物理基础模型能够从数值模拟涌现式地迁移到真实实验室湍流，在瑞利-泰勒不稳定性这一长期存在模拟-实验差异（增长率相差约三倍）的难题上取得了突破。 |
| [^318] | [FLAG: Flow Policy MaxEnt-RL by Latent Augmented Guidance](https://arxiv.org/abs/2605.30749) | FLAG通过在状态空间中引入流模型潜变量来局部化采样区域，避免了重要性权重坍缩问题，使最大熵强化学习能够在有限重要性样本下实现表达力强的策略优化并扩展到高维动作空间。 |
| [^319] | [Conservative neural posterior estimation via distributionally robust training](https://arxiv.org/abs/2605.28516) | 提出DRO-NPE方法，通过Wasserstein模糊集上的分布鲁棒训练解决神经后验估计在有限模拟预算下过度自信的问题，显著提升了基于模拟推断的覆盖率和校准可靠性。 |
| [^320] | [More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations](https://arxiv.org/abs/2605.26647) | 本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。 |
| [^321] | [AvalancheBench: Evaluating Enterprise Agent-Driven Semantic Query Processing Through Latent World Recovery](https://arxiv.org/abs/2605.24183) | 该论文提出了AvalancheBench，一个从潜在世界规范生成观测数据、通过涵盖文本、图像和文档的58个多部分问题评估数据智能体端到端语义查询能力的多模态企业分析基准，并首次揭示了错误经合法依赖关系传播、不当复用中间结果导致无关问题失败的“雪崩效应”。 |
| [^322] | [EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation](https://arxiv.org/abs/2605.22286) | 本文提出EmoTrack框架，通过临床-语义建模将临床特征分数与话语级表示相融合，从心理咨询文本中预测PHQ-8抑郁严重程度总分，解决了临床评估与对话证据之间关联不足的问题。 |
| [^323] | [Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift](https://arxiv.org/abs/2605.21849) | 提出几何自适应解释器（GAE），仅用无标签的分布外激活、无需梯度更新即可将词典重新对齐到OOD活跃子空间，从而量化并减小分布偏移下的忠实性差距。 |
| [^324] | [Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers](https://arxiv.org/abs/2605.21325) | 本文系统分析了Delta规则线性Transformer中三角矩阵求逆的直接法与迭代法算法，重点考察数值稳定性与计算复杂度，以解决该核心操作的性能瓶颈和数值精度问题。 |
| [^325] | [A Rigorous, Tractable Measure of Model Complexity](https://arxiv.org/abs/2605.21167) | 该论文提出了一种基于特征向量相似性的、数学上严谨且易于计算的模型复杂度度量，它适用于参数化和核方法非参数化模型，并能统一泛化多种现有模型特定的复杂度度量（如多项式次数、k近邻邻居数、决策树分裂次数、随机森林树的数量等）。 |
| [^326] | [Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2605.21139) | CoPhy 框架通过将视觉语言模型知识蒸馏进 BEV 编码器实现零推理成本的认知能力，并构建自回归 BEV 世界模型来预测候选动作的未来后果，从而突破模仿学习的行为克隆上限，实现自动驾驶的认知-物理强化学习。 |
| [^327] | [Prediction-Intervention Games and Invariant Sets](https://arxiv.org/abs/2605.16828) | 该论文提出预测-干预博弈（Stackelberg博弈的特例），并证明对于两类常见的跟随者干预目标，基于稳定毯这一不变子集的预测器性能至少与基于因果父节点的预测器相当。 |
| [^328] | [Bridging the Modality Bottleneck in Pathology MIL through Molecularly Guided Feature Adaptation](https://arxiv.org/abs/2605.16392) | 该论文提出MIST，一种即插即用的MIL投影层替代方法，仅在训练时利用配对空间转录组学构建虚拟分子染色，沿分子引导的轴重组H&E特征，从而弥合病理MIL中形态学与分子信息之间的模态瓶颈，且推理时无需转录组学数据。 |
| [^329] | [NodeSynth: Socially Aligned Synthetic Data for AI Evaluation](https://arxiv.org/abs/2605.14381) | NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。 |
| [^330] | [Learning POMDP World Models from Observations with Language-Model Priors](https://arxiv.org/abs/2605.13740) | 提出Pinductor方法，利用大语言模型先验从少量观测-动作轨迹中提出并迭代优化POMDP世界模型，在不使用隐藏状态的情况下达到与拥有特权信息方法相当的性能和样本效率。 |
| [^331] | [GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?](https://arxiv.org/abs/2605.12827) | 提出了首个统一基准GraphIP-Bench，在单一黑盒协议下系统评估12种模型提取攻击与12种GNN知识产权防御方法（覆盖10个图数据集、3种骨干网络和3类任务），首次可靠地回答了窃取GNN的难度以及防御方法能否有效阻止窃取这两个问题。 |
| [^332] | [EPIC: Efficient Predicate-Guided Inference-Time Control for Compositional Text-to-Image Generation](https://arxiv.org/abs/2605.11722) | EPIC是一个无需训练的推理时精炼框架，通过将提示解析为谓词视觉程序并基于视觉证据验证与定向修正生成图像，大幅提升了文生图模型处理多对象、数量、属性和关系等组合式提示的能力。 |
| [^333] | [Agentic Literature Extraction Augments Targeted Protein Degradation Databases](https://arxiv.org/abs/2605.11221) | 本研究提出专家在环的大语言模型工作流程，仅用七篇标注文献即实现高精度（F1=0.98）的靶向蛋白降解实验数据自动抽取，构建的AutoTPD数据库比现有参考数据库多出81%至92%的记录。 |
| [^334] | [Benchmarking Sensor-Fault Robustness in Forecasting](https://arxiv.org/abs/2605.10822) | 该论文提出了SensorFault-Bench，一个面向信息物理系统的传感器故障鲁棒性基准，通过受控合成压力测试在四个真实数据集和八个标准化评分场景下评估预测模型对孤立输入故障的鲁棒性，并将相对鲁棒性与绝对误差区分开来。 |
| [^335] | [Factual recall in linear associative memories: sharp asymptotics and mechanistic insights](https://arxiv.org/abs/2605.10795) | 本文通过引入一个与原模型等价的解耦模型，精确刻画了线性联想记忆在事实回忆任务中可存储关联数量的容量极限 $p_c$，并对学习解的结构给出了机理层面的洞察。 |
| [^336] | [Lattice Deduction Transformers](https://arxiv.org/abs/2605.08605) | 格演绎Transformer（LDT）通过在前向传递之间以格投影潜在状态来近似逻辑可靠的演绎推理，仅用80万参数和不到现有模型2%的训练计算量，就以近乎完全可靠的方式解决了99.999%的极难数独谜题。 |
| [^337] | [KL for a KL: On-Policy Distillation with Control Variate Baseline](https://arxiv.org/abs/2605.07865) | 提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。 |
| [^338] | [Structural Rationale Distillation via Reasoning Space Compression](https://arxiv.org/abs/2605.07139) | 提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。 |
| [^339] | [Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves](https://arxiv.org/abs/2605.06395) | 该论文提出了 HilbNets——一种以希尔伯特丛的联络拉普拉斯算子作为卷积算子的新型卷积学习框架，可处理流形上可能无穷维的信号，并通过采样诱导的希尔伯特胞腔层使卷积运算得以实际实现。 |
| [^340] | [Data-Driven and Cycle-Agnostic Covariate Selection for Nonparametric Causal Effect Estimation](https://arxiv.org/abs/2605.06385) | 该论文证明基于条件独立性的局部数据驱动协变量选择方法的有效性保证可从无环因果模型扩展到含环因果模型，从而为非参数因果效应估计建立了统一的、与环路无关的协变量选择框架。 |
| [^341] | [Ordinal Credit Assignment: Multi-Step Reinforcement Learning from Comparisons Alone](https://arxiv.org/abs/2605.06375) | 该论文提出一种在多步任务中仅依靠轨迹间两两比较（无需奖励函数或奖励模型）的序数信用分配方法，用动作在同一时间步群体中的胜率替代未来累积奖励，证明其期望更新等价于策略梯度，且在确定性环境中按真实价值排序、在随机环境中偏好高胜率动作，并对标签噪声和位置偏差具有鲁棒性。 |
| [^342] | [LINC: Decoupling Local Consequence Scoring from Hidden Matching in Constructive Neural Routing](https://arxiv.org/abs/2605.06332) | LINC提出一种解码器端候选决策架构，将确定性单步后果的显式计算与重要性学习解耦，使构造式神经路由在保留缓存嵌入和全局匹配的同时，能直接基于确定性转移量对候选动作评分。 |
| [^343] | [In-Context Black-Box Optimization with Unreliable Feedback](https://arxiv.org/abs/2605.06187) | 该论文提出反馈感知的上下文黑盒优化框架FICBO，通过结构化先验预训练Transformer，使其能同时利用优化历史和可能不可靠的辅助反馈，从而在反馈质量参差不齐的情况下实现稳健的跨任务优化。 |
| [^344] | [Estimating Implicit Regularization in Deep Learning](https://arxiv.org/abs/2605.05436) | 本文提出基于梯度匹配的估计方法，利用正则化导致权重更新偏离损失梯度这一信号，来量化深度学习系统（包括早停、小批量、dropout 等修改）中隐式正则化的效果。 |
| [^345] | [Hyper-Dimensional Fingerprints as Molecular Representations](https://arxiv.org/abs/2604.27810) | 该论文提出了一种无需训练的超维指纹（HDF），通过高维向量代数运算组合原子化学属性与多跳环境信息，生成确定性分子表示，在性质预测基准上表现优异，且在低维度下比传统指纹更高效地编码分子结构。 |
| [^346] | [Medial Axis Aware Learning of Signed Distance Functions](https://arxiv.org/abs/2604.16512) | 提出一种中轴感知的高阶变分方法，通过相场近似与神经网络，从无向点云中计算高精度的全局符号距离函数。 |
| [^347] | [Attention to Mamba: A Recipe for Cross-Architecture Distillation](https://arxiv.org/abs/2604.14191) | 该论文提出了一套将预训练的Attention架构Transformer模型有效蒸馏到Mamba状态空间模型架构的方法配方，使蒸馏后的Mamba模型能够保留教师模型的性能，从而加速SSM架构的落地采用。 |
| [^348] | [Quantum-Inspired Tensor Network Autoencoders for Anomaly Detection: A MERA-Based Approach](https://arxiv.org/abs/2604.06541) | 该论文首次提出了受量子多尺度纠缠重整化拟设（MERA）启发的张量网络自编码器，并将其应用于对撞机喷注的无监督异常检测任务。 |
| [^349] | [Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding](https://arxiv.org/abs/2603.25251) | 该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。 |
| [^350] | [MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels](https://arxiv.org/abs/2603.19310) | 提出MemReward，将查询、思考过程和答案作为异构图节点构建经验记忆，利用GNN将有限的奖励标签传播到无标签轨迹，从而在RL微调中充分利用全部采样轨迹优化LLM策略。 |
| [^351] | [Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures](https://arxiv.org/abs/2603.16568) | 提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。 |
| [^352] | [PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation](https://arxiv.org/abs/2603.13275) | PREBA提出了一种融合PCA加权检索与贝叶斯平均聚合的检索增强框架，无需任务特定微调即可将大语言模型的推理锚定于机构特定的临床证据上，从而实现准确且稳定的手术时长预测。 |
| [^353] | [Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck](https://arxiv.org/abs/2603.08462) | 该论文将高效推理重新建模为条件信息瓶颈下的有损压缩问题，解决了注意力机制违反马尔可夫性质的理论缺口，并由此导出一个通用的强化学习目标，在压缩推理链的同时最大化任务奖励，从而统一了预算强制方法。 |
| [^354] | [From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors](https://arxiv.org/abs/2603.02792) | 本文通过 token 级归因分析发现提示词中嵌入的代码示例和策略指令对 LLM 生成的算法影响最大，并据此提出将现有基准算法作为强先验来引导 LLM 进行自动化算法设计，在伪布尔和黑盒优化基准上验证了其有效性。 |
| [^355] | [SEAR: Sample Efficient Action Chunking Reinforcement Learning](https://arxiv.org/abs/2603.01891) | SEAR通过多时域目标训练的因果变换器批评者和后退时域随机重规划，实现了样本高效的在线动作分块强化学习，同时解决了高维动作序列和决策频率恢复的挑战。 |
| [^356] | [The minimal width of universal $p$-adic ReLU neural networks](https://arxiv.org/abs/2603.00064) | 本文确定了以自然 $p$-adic ReLU 类似函数为激活函数的 $p$-adic 神经网络，在 $L_q$ 范数和 $C_1$ 范数意义下对紧开子集上连续 $\mathbb Q_p$-值函数具有万能逼近性质所需的最小宽度。 |
| [^357] | [High-Dimensional Robust Mean Estimation from Doubly Corrupted and Heterogeneous Batches](https://arxiv.org/abs/2602.20698) | 本文研究用户级与样本级双重对抗污染下的高维鲁棒均值估计问题，提出了一种平方和（SoS）算法，在协方差矩阵有界条件下实现了较小的 $\ell_2$ 估计误差。 |
| [^358] | [Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis](https://arxiv.org/abs/2602.20207) | 本研究通过层梯度分析，证明了大语言模型中存在固定的“黄金层”，其知识编辑性能接近逐样本最优层，且这些黄金层可通过代理数据集可靠识别并能跨数据集泛化到未见查询，从而改进了知识编辑方法。 |
| [^359] | [Quantifying Error Propagation and Model Collapse in Diffusion Models](https://arxiv.org/abs/2602.16601) | 该论文首次为扩散模型在合成数据上递归训练导致的模型坍塌建立了散度的理论上界与下界，并刻画了分数估计误差与新鲜数据比例如何决定生成分布偏离目标分布的不同漂移区间。 |
| [^360] | [Evolutionary System Prompt Learning for Reinforcement Learning in LLMs](https://arxiv.org/abs/2602.14697) | 本文提出进化式系统提示学习（E-SPL），在强化学习迭代中联合优化模型权重与系统提示，实现权重中程序性知识与提示中可解释陈述性知识的自然分工，显著提升大语言模型在数学推理任务上的成功率与由易到难的泛化能力。 |
| [^361] | [Discovering Semantic Structures in Psychological Scales: A Response-Free Front-End to Efficient Simplification](https://arxiv.org/abs/2602.12575) | 该论文提出了一种无需作答数据的主题建模框架，利用上下文句子嵌入与密度聚类从量表题目措辞中自动发现语义结构并提取代表性关键词，从而实现心理量表的高效系统化简化。 |
| [^362] | [A Lightweight Multi-View Approach to Short-Term Load Forecasting](https://arxiv.org/abs/2602.09220) | 本文提出一种利用单值嵌入、缩放时间范围输入和嵌入丢弃机制的轻量级多视角方法，以显著更少的参数实现了具有竞争力的短期负荷预测性能。 |
| [^363] | [Mitigating Memorization Where It Happens](https://arxiv.org/abs/2602.08901) | 该论文提出了一个探测—引导框架，利用目标模型与低暴露参考模型之间的 token 级对数似然比精准定位记忆化发生的位置，并仅在这些位置施加有针对性的激活校正，从而在缓解逐字记忆问题的同时避免损害模型的正常泛化性能。 |
| [^364] | [CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation](https://arxiv.org/abs/2602.08686) | 论文提出CompilerKV，通过从校准语料库离线编译KV保留校正表，将在线校正开销降至O(1)查找，且编译出的保留表可作为可移植的架构先验，在跨语料库和跨模型迁移中表现出高度一致性。 |
| [^365] | [Orthogonal Model Merging](https://arxiv.org/abs/2602.05943) | 提出OrthoMerge方法，通过在正交群形成的黎曼流形上进行合并以保留预训练权重的几何结构，从而更有原则且高效地整合多个微调大语言模型的多样化能力。 |
| [^366] | [Federated Concept-Based Models: Interpretable models with distributed supervision](https://arxiv.org/abs/2602.04093) | 本文提出联邦概念模型，通过跨机构聚合概念级信息并随概念监督集合的演变高效调整模型架构，首次将可解释的概念模型与联邦学习相结合，实现了分布式监督下的可解释深度学习。 |
| [^367] | [SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning](https://arxiv.org/abs/2602.03201) | SLOPE通过乐观分布回归估计高置信度上界来构建信息丰富的势能景观，放大稀疏奖励中的罕见成功信号，从而显著提升基于模型的强化学习在稀疏奖励任务上的性能。 |
| [^368] | [From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient](https://arxiv.org/abs/2602.02313) | 本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。 |
| [^369] | [OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference](https://arxiv.org/abs/2602.01493) | 提出OpInf-LLM框架，将算子推断与大语言模型结合，仅需少量解数据即可准确求解参数化偏微分方程，并能泛化到未见过的参数与边界条件，缓解了执行成功率与数值精度之间的权衡。 |
| [^370] | [Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio](https://arxiv.org/abs/2601.22718) | 该论文发现token级别的重要性采样比率无法修正前缀分布偏移，并提出用累积前缀重要性比率对前缀边缘化策略梯度进行精确修正，从而在大策略滞后下稳定大语言模型的离策略强化学习优化。 |
| [^371] | [Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning](https://arxiv.org/abs/2601.06641) | 提出了PromptMIA——一种针对联邦提示调优的新型成员推理攻击，恶意服务器通过注入对抗性构造的软提示并利用其训练更新，能够高效地推断出客户端私有数据集中的成员信息，显著优于现有联邦学习攻击方法。 |
| [^372] | [Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings](https://arxiv.org/abs/2601.01558) | 本研究将AlphaEarth嵌入作为静态流域表征用于无观测流域的径流预测，发现其样本外泛化鲁棒性优于传统CAMELS流域属性。 |
| [^373] | [Optimality-Informed Neural Networks for Solving Parametric Optimization Problems](https://arxiv.org/abs/2512.20270) | 提出OptINNs方法，通过KKT残差损失与内嵌约束的输出激活函数，利用神经网络学习参数到原始最优解及对偶变量的映射，从而高效求解参数化优化问题并大幅降低数据需求。 |
| [^374] | [Deep Sets and event-level maximum-likelihood estimation for fast pile-up jet rejection at the ATLAS High Level Trigger](https://arxiv.org/abs/2512.10819) | 本文提出DIPz模型，利用Deep Sets从带电粒子径迹中以高斯似然形式估计喷注的束流轴原点及其不确定性，并将各喷注预测组合成事件级最大似然判别量，为ATLAS高层触发实现快速且稳健的堆叠喷注剔除。 |
| [^375] | [From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection](https://arxiv.org/abs/2512.10485) | 本文系统评估了四种深度学习漏洞检测模型在多个数据集上的表现，通过跨数据集实验和可视化分析揭示了基准测试与真实世界应用之间的性能差距，强调了模型泛化能力的挑战。 |
| [^376] | [A Granular Framework for Construction Material Price Forecasting: Econometric and Machine-Learning Approaches](https://arxiv.org/abs/2512.09360) | 本研究提出了一个基于CSI MasterFormat的建筑材料价格细粒度预测框架，能够在六位数字章节级别整合原材料价格与宏观经济指标等解释变量，并对比评估了LSTM、ARIMA、VECM、Transformer和Chronos-Bolt五种时间序列模型的预测性能。 |
| [^377] | [Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors](https://arxiv.org/abs/2511.14581) | 本文首次将可微分求解器参与的在线端到端机器学习方法应用于有界旋转区域中的二维准地转湍流次网格尺度建模，突破了以往研究局限于无边界理想化区域的限制，能够捕捉与行星内部相关的纬向急流、罗斯贝波等关键动力学特征。 |
| [^378] | [Bilevel Learning via Strategically Inexact Stochastic Gradient Descent](https://arxiv.org/abs/2511.06774) | 该论文针对计算成像中的双层优化问题，证明了实践者实际采用的不精确随机梯度调度方案可保证收敛，并在超梯度精度与计算成本线性相关、精度衰减衰减等条件下建立了收敛速率。 |
| [^379] | [MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction](https://arxiv.org/abs/2511.03212) | 提出了一种基于多视角混合Transformer的MvBody模型，仅利用妊娠31-38周期间的3D光学人体扫描和自我报告医疗数据，即可实现可解释的剖腹产风险预测，适用于医疗资源有限的环境。 |
| [^380] | [ESS-Flow: Training-free guidance as Bayesian inference in source space](https://arxiv.org/abs/2510.05849) | ESS-Flow是一种免训练、无梯度的引导方法，通过椭圆切片采样利用流模型源分布的高斯先验直接在源空间进行贝叶斯推断，即使梯度不可用（如不可微分模拟观测或量化过程）也能引导预训练流模型完成条件生成任务。 |
| [^381] | [UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models](https://arxiv.org/abs/2510.02194) | 提出UpSafe°C框架，通过将安全关键层升级为稀疏MoE结构、利用路由器作为软防护栏、结合两阶段SFT和可调节的安全温度机制，在保持通用能力的同时实现大语言模型推理时可灵活控制的安全防护。 |
| [^382] | [fev-bench: A Realistic Benchmark for Time Series Forecasting](https://arxiv.org/abs/2509.26468) | 提出了 fev-bench——一个涵盖七个领域共 100 个任务（其中 46 个带协变量）的贴近现实的时间序列预测基准，并配套推出强调可复现性与易集成的轻量级 Python 评估库 fev，通过自举置信区间等统计严谨的聚合方式实现可靠模型评估。 |
| [^383] | [Discovering Phase Space Structure in Learned Hamiltonian Systems](https://arxiv.org/abs/2509.23003) | 提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。 |
| [^384] | [Population-Aligned Persona Generation for LLM-based Social Simulation](https://arxiv.org/abs/2509.10127) | 该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。 |
| [^385] | [Power Grid Control with Graph-Based Distributed Reinforcement Learning](https://arxiv.org/abs/2509.02861) | 该论文提出了一种基于图的分布式强化学习框架，利用图神经网络编码电网拓扑信息，并通过高低层智能体协同实现了对现代电网的实时、可扩展控制。 |
| [^386] | [Constrained Black-Box Attacks Against Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2508.09275) | 该论文提出了两种此前被忽视的受限黑盒攻击设置（仅能访问智能体观测或完全无法访问任何信息），通过生成使智能体环境感知错位的扰动来有效破坏协作式多智能体强化学习，且无需访问策略权重或训练代理策略，在22个环境中验证了攻击的有效性和样本效率。 |
| [^387] | [Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning](https://arxiv.org/abs/2508.06199) | 本研究对28个预训练分子嵌入模型进行了大规模基准测试，发现除CheMeleon外所有神经模型均未超越经典ECFP计数指纹，且外部验证显示两者并无明确优劣，建议在新QSAR/QSPR研究中同时评估这两种方法。 |
| [^388] | [When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings](https://arxiv.org/abs/2508.01128) | 提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。 |
| [^389] | [Scalable Source Selection for Machine Learning Tasks](https://arxiv.org/abs/2507.22186) | 提出了Splice系统，一种借鉴基因剪接思想的目标与成本感知数据源选择方法，通过识别能为下游任务提供额外信号的数据源来最大化任务效用，并开发了利用数据画像和梯度对齐的两种加速变体Splice-DP和Splice-Grad。 |
| [^390] | [Domain-Adaptive Small Language Models for Structured Tax Code Prediction](https://arxiv.org/abs/2507.10880) | 该论文提出了一种基于编码器-解码器架构的领域自适应小型语言模型，能够从非结构化产品和服务数据中顺序预测层级化的税则编码（如HSN/SAC），从而提升跨国公司税务合规的准确性。 |
| [^391] | [Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning](https://arxiv.org/abs/2507.07359) | GO-CBED是一个面向目标的贝叶斯序贯因果实验设计框架，通过变分下界估计与Transformer策略网络直接最大化用户指定因果量的期望信息增益，实现非短视且实时的干预决策，在因果推理与发现任务中优于现有基线。 |
| [^392] | [GNNs Meet Sequence Models Along the Shortest-Path: an Expressive Method for Link Prediction](https://arxiv.org/abs/2507.07138) | SP4LP提出将图神经网络节点编码与最短路径上的序列建模相结合，能够以高效的方式捕获链路预测中表达力强的多跳关系模式。 |
| [^393] | [Hindsight Relabeling is All You Need for Reach-Avoid Learning](https://arxiv.org/abs/2505.19337) | 本文提出RADT，一种基于决策Transformer的离线无奖励强化学习方法，通过将目标与规避区域直接编码为提示token，仅凭随机策略产生的次优离线轨迹即可学会抵达-规避行为，并能在评估时灵活指定任意数量和大小的规避区域。 |
| [^394] | [Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis](https://arxiv.org/abs/2505.11581) | 本文通过对比演化网络与SGD训练网络在单图像生成任务上的内部表征，发现两者虽输出行为相同但内部结构截然不同——SGD训练的网络呈现“破碎纠缠表征”，从而挑战了“性能更好必然意味着内部表征更好”的表征乐观主义观点。 |
| [^395] | [Learning Penalty for Changepoint Detection via Automatic Feature Extraction](https://arxiv.org/abs/2505.07413) | 该研究提出利用循环网络自动提取特征，直接从原始序列中学习变点检测的最优惩罚参数，在20个基因组基准数据集上的实验表明其检测精度普遍优于依赖人工特征提取的传统方法。 |
| [^396] | [Achieving adaptivity and optimality for multi-armed bandits using Exponential-Kullback Leibler Maillard Sampling](https://arxiv.org/abs/2502.14379) | 本文提出了Exp-KL-MS算法，这是首个能同时满足渐近最优性、极小极大最优性（带√ln(K)因子）、Sub-UCB和方差自适应遗憾界等多种最优性标准的多臂老虎机算法。 |
| [^397] | [ENFORCE: Nonlinear Constrained Learning with Adaptive-depth Neural Projection](https://arxiv.org/abs/2502.06774) | 提出了ENFORCE神经网络架构，通过自适应深度神经投影模块（AdaNP）使预测在指定容差内满足非线性等式与不等式约束，并从理论上证明了仿射约束下投影的非扩张性（保证梯度稳定传播）及非线性约束下的局部收敛性。 |
| [^398] | [Sensor Response-Time Reduction using Long-Short Term Memory Network Forecasting](https://arxiv.org/abs/2404.17144) | 本文提出利用长短期记忆（LSTM）网络集成方法，仅从少量初始时序生物传感器测量数据即可准确预测最终的稳态响应，从而有效缩短生物传感器的响应时间。 |
| [^399] | [Exploring Sequence Feature Alignment for Domain Adaptive Detection Transformers](https://arxiv.org/abs/2107.12636) | 提出了一种专为检测Transformer设计的序列特征对齐（SFA）方法，通过基于域查询的特征对齐（DQFA）模块和逐词元特征对齐（TDA）模块来减小域差异，从而有效提升跨域目标检测性能。 |
| [^400] | [Estimating and Incentivizing Imperfect-Knowledge Agents with Hidden Rewards.](http://arxiv.org/abs/2308.06717) | 本文研究了一个自私学习代理和学习委托人之间的重复对自选游戏，探索如何估计和激励具有隐藏奖励的不完全知识代理。 |
| [^401] | [Action-State Dependent Dynamic Model Selection.](http://arxiv.org/abs/2307.04754) | 本文提出了一种动态模型选择的方法，该方法能够根据不同的状态选择最优的模型，并通过强化学习算法对动态规划问题进行近似和估计。实验结果表明，在重新平衡成本下切换投资组合模型时，使用宏观经济信息的性能优于事后选择最佳投资组合模型。 |
| [^402] | [Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents.](http://arxiv.org/abs/2304.07407) | 本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。 |

# 详细

[^1]: 大语言模型社交智能体中的情境增强与选择性鲁棒性框架

    A Framework for Context Enrichment and Selective Robustness in LLM Social Agents

    [https://arxiv.org/abs/2611.06914](https://arxiv.org/abs/2611.06914)

    该论文提出“受控情境增强”框架，通过一组提示词替换规则系统地丰富四种不同类型的情境（中性填充、通用社会背景、描述性决策依据、规范性决策依据），以探究更丰富的情境能否提升大语言模型社交智能体在语义不变提示变化下的行为鲁棒性。

    

    研究人员在社交模拟中使用大语言模型作为智能体，通过观察信息和背景信息来提示智能体，以塑造其行为。然而，已有研究表明，在保持语义不变的提示词变化下，大语言模型智能体的行为会发生显著改变，这给社交模拟带来了验证难题。一个合理的假设是，这种脆弱性反映了缺乏情境依据，并且可以通过丰富提供给智能体的情境来缓解。但更丰富的情境能否提升鲁棒性仍不清楚，部分原因在于情境丰富度和鲁棒性很少以支持受控比较的方式被定义。我们通过“受控情境增强”来满足第一个需求，这是一组提示词替换规则，可系统地增强不同类型的情境：中性填充信息、通用社会背景、描述性决策依据以及规范性决策依据。我们通过……（原文摘要在此处截断）

    arXiv:2611.06914v1 Announce Type: cross  Abstract: Researchers use large language models as agents in social simulations, where agents are prompted with observations and background information intended to shape their actions. Yet LLM agents have been shown to substantially change their behavior under meaning-preserving prompt variations, creating a validation problem for social simulation. One plausible hypothesis is that this brittleness reflects a lack of grounding and can be reduced by enriching the context provided to the agents. Whether richer context improves robustness remains unclear, in part because context richness and robustness are rarely defined in ways that support controlled comparisons. We address the first need with controlled context enrichment, a set of prompt substitution rules that systematically enriches distinct types of context: neutral padding, generic social background, descriptive decision grounding, and normative decision grounding. We address the second by 
    
[^2]: FlowReconBench：基于稀疏传感器的学习型与经典流场重构方法的精度、校准与成本

    FlowReconBench: Accuracy, calibration and cost of learned and classical flow reconstruction from sparse sensors

    [https://arxiv.org/abs/2611.06912](https://arxiv.org/abs/2611.06912)

    本文提出 FlowReconBench 基准，系统评估了学习型与经典稀疏传感器流场重构方法在三种流动状态下的精度、校准与成本，发现最优方法因流动状态而异，且经典方法在低阶流动和密集传感场景下表现强劲。

    

    从稀疏传感器重构完整流场需要对未测量区域和物理变量做出先验假设。然而，学习型模型何时优于经典估计器，以及其预测中哪些方面是可信的，目前仍不清楚。FlowReconBench 在三种流动状态下评估了十种学习型方法和五种经典方法：跨雷诺数变化的层流圆柱尾流、二维 Kolmogorov 湍流以及大规模三维（3D）各向同性湍流。所有方法采用统一的防泄漏数据划分与传感器采样配置，并按照逐通道误差、集成离散度、校准性、能谱和计算成本进行评分。最优方法随流动状态而变化：经典方法在低阶流动和密集传感场景下表现强劲；在圆柱尾流问题上，线性随机估计和带缺失数据的本征正交分解（gappy POD）方法领先，包括对未观测压力的重构；在三维情况下，逆距离加权方法可与最佳学习型方法相媲美。

    arXiv:2611.06912v1 Announce Type: cross  Abstract: Reconstructing full flow fields from sparse sensors requires prior assumptions about unmeasured regions and physical variables. Yet it remains unclear when learned models outperform classical estimators and which aspects of their predictions can be trusted. FlowReconBench evaluates ten learned and five classical methods on three regimes: a laminar cylinder wake across Reynolds numbers, two-dimensional Kolmogorov turbulence and a large-scale three-dimensional (3D) isotropic turbulence. Methods share leakage-controlled splits and sensor draws and are scored on per-channel error, ensemble spread, calibration, spectra and cost. The leading method changes for each regime. Classical methods perform strongly in low-order and dense-sensing regimes. On the cylinder wake, linear stochastic estimation and gappy proper orthogonal decomposition lead, including for unobserved pressure. In 3D, inverse-distance weighting matches the best learned metho
    
[^3]: 基于仿真的具感知人形机器人运动

    Perceptive Humanoid Motion from Simulation

    [https://arxiv.org/abs/2611.06908](https://arxiv.org/abs/2611.06908)

    提出了HALO人形机器人自主系统，通过在千个程序化生成的家庭环境中合成视觉-语言-运动数据，使数据规模随算力而非人力扩展，让人形机器人掌握坐、导航、穿门、蹲爬、推箱等多种物理技能并协调完成复合任务。

    

    人形机器人有潜力自动化广泛的物理任务。然而，迈向通用型人形机器人自主能力的进展仍受限于数据瓶颈：将像素和自由形式的语言映射到底层关节命令需要成对的监督数据，而目前尚不存在足够大规模的此类数据语料库。一些方法依赖于人类遥操作和可穿戴动作捕捉设备，这需要至少一小时的人工时间才能产出一小时的数据。我们提出了HALO，这是一个由数据合成流水线驱动的人形机器人自主系统，该流水线在一千个程序化生成的家庭环境中合成视觉-语言-运动数据。该语料库的规模随算力而非人力扩展。HALO掌握了一系列物理技能，包括坐在椅子上、导航至指定名称的物体、侧身穿过半开的门、在桌下蹲伏和爬行，以及推箱子。它协调这些技能以解决复合任务。

    arXiv:2611.06908v1 Announce Type: cross  Abstract: Humanoid robots have the potential to automate a wide range of physical tasks. Progress toward generalist humanoid autonomy, however, remains constrained by a data bottleneck: mapping pixels and free-form language to low-level joint commands requires paired supervision, and no corpus of such data exists at sufficiently large scale. Some approaches rely on human teleoperation and wearable capture devices, which require at least an hour of human time to produce an hour of data. We present HALO, a humanoid autonomy system powered by a pipeline that synthesizes vision-language-motion data across a thousand procedurally generated homes. The corpus scales with compute rather than human labor. HALO acquires a repertoire of physical skills, including sitting on chairs, navigating to named objects, stepping sideways through half-open doors, crouching and crawling beneath tables, and pushing boxes. It orchestrates these skills to solve composite
    
[^4]: 强化学习助力通用机器人策略改进的规模化

    Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement

    [https://arxiv.org/abs/2611.06905](https://arxiv.org/abs/2611.06905)

    本文探究并提出了让通用机器人策略通过强化学习在部署阶段实现多任务、通用性自我改进的算法方案，突破了以往仅能在单任务上进行优化的局限。

    

    通用机器人策略的引入使得单个大型模型能够从庞大的预训练语料库中学习执行多种任务。尽管这些模型通常展现出强大的零样本泛化能力，但它们通常需要一个适应阶段才能在部署中达到足够的性能。通过强化学习（RL）进行迭代式自我改进在实现这种适应方面已展现出前景，然而此类方法主要在单任务设置中被探索——其目标仅仅是在狭窄的部署任务上进行改进——这与基础策略的多任务能力形成了鲜明对比。在本工作中，我们探究通用机器人策略是否能够在部署中实现多任务、通用性的改进——即同时针对众多多样化的任务——以及什么样的算法方案能够实现这一目标。为了回答这个问题，我们首先开发了一套结构化的评估方法来测试通...（摘要原文在此处截断）

    arXiv:2611.06905v1 Announce Type: new  Abstract: The introduction of generalist robot policies has enabled a single, large model to learn to perform a variety of tasks from a vast pre-training corpus. While these models often exhibit powerful zero-shot generalization capabilities, they typically require an adaptation phase to achieve sufficient performance in deployment. Iterated self-improvement via reinforcement learning (RL) has demonstrated promise in enabling this adaptation, yet such approaches have been largely explored in single-task settings - where the goal is to simply improve on a narrow deployment task - which lies in stark contrast to the multi-task capabilities of the base policy. In this work we ask whether generalist robot policies are capable of multi-task, generalist improvement in deployment - targeting many diverse tasks simultaneously - and what algorithmic recipe enables this. To answer this question, we first develop a set of structured evaluations to test gener
    
[^5]: 自适应多模式：基于在线景观探测的优化器预算感知路由

    Adaptive Multi Mode: Budget-Aware Routing of Optimizers By Online Landscape Probes

    [https://arxiv.org/abs/2611.06879](https://arxiv.org/abs/2611.06879)

    AMM通过在线探测优化问题的景观特征（如曲率、可分离性等），将优化预算智能分配给拟牛顿、信赖域、演化策略等不同经典优化器，在bbob基准的600个函数-维度-预算组合中有382个取得最高目标份额。

    

    对于连续黑盒优化问题，最佳的经典优化器取决于其可用预算和问题景观。现有的路由器仅依据维度与预算、过往改进情况或针对单一基准拟合的特征来做决策，很少有方法在线测试优化器的隐含假设。AMM（自适应多模式）通过在线探测曲率、平台、可分离性、折点、盆地和分形结构等特征，将已知预算分配到拟牛顿、信赖域、坐标法、多起点和演化策略等优化引擎之间。每个探测方法都有默认设置、基于维度与预算的有效范围，以及计入预算的成本。在其预注册的开发流程中，在bbob基准上，36个预测中有32个在最后16次验证中得到确认。随后，AMM在留出的bbob实例101-110（D=2-40，10*D至1000*D预算）上进行了单次测试，与最多七个竞争者对比。它在600个函数-维度-预算组合中的382个（236个单独领先，146个并列）拥有最高的目标份额，并通过描述性（摘要在此处截断）

    arXiv:2611.06879v1 Announce Type: new  Abstract: The best classical optimizer for a continuous black-box problem depends on its budget and landscape. Existing routers decide from the dimension and budget, past improvement or features fitted to one benchmark, and few test an optimizer's implicit assumptions online. AMM (Adaptive Multi Mode) routes a known budget among quasi-Newton, trust-region, coordinate, multistart and evolution-strategy optimization engines with online probes of curvature, plateaus, separability, kinks, basins and fractal structure. Each probe has a default, a dimension-by-budget validity range and a cost charged to the budget. In its pre-registered development loop on bbob, 32 of 36 predictions held over the last 16 confirmations. AMM was then tested once on held-out bbob instances 101-110 (D=2-40, 10*D-1000*D) against up to seven competitors. It has the highest target share in 382 of 600 function-dimension-budget cells (236 outright, 146 tied) and, by a descriptiv
    
[^6]: 面向证据感知的个体动物识别的反事实检索方法

    CounterFactual Retrieval for Evidence-Aware Individual Animal Identification

    [https://arxiv.org/abs/2611.06864](https://arxiv.org/abs/2611.06864)

    提出反事实前景检索（CFR）这一证据感知评估协议，通过完整图像、仅前景、仅捷径三种视图对动物个体识别进行评测，将检索分数转化为证据画像，从而揭示模型是依赖动物特征还是身份相关的背景捷径。

    

    自动化的个体动物识别正在从小规模精选研究向共享基准和可复用的基础模型发展。这一进步是有价值的，但当前的评估方式未能揭示匹配结果背后的证据依据，即高检索准确率无法表明模型究竟是使用了动物本身的特征，还是利用了与身份相关的捷径信息。为了解决这一问题，我们提出了反事实前景检索（CounterFactual Foreground Retrieval, CFR），这是一种证据感知的评估协议。CFR 在不同视图下评估同一查询-图库检索任务：完整图像视图衡量标准检索性能；仅前景视图衡量由前景支撑的检索能力；仅捷径视图则衡量从背景或轮廓中可恢复的身份信号量。由此，单一的检索分数被转化为一份证据画像：包括模型在原始图像上的表现，以及当仅有动物（前景）信息时该性能能保留多少。

    arXiv:2611.06864v1 Announce Type: cross  Abstract: Automated identification of individual animals is moving from small curated studies toward shared benchmarks and reusable foundation models. This progress is useful, but current evaluation leaves the evidence behind a match unresolved, i.e., high retrieval accuracy does not show whether a model used animal features or identity-correlated shortcuts. We address this with Counterfactual Foreground Retrieval (CFR), an evidence-aware evaluation protocol. CFR evaluates the same query-gallery task under different views: full image, foreground-only, and shortcut-only. The full-image measures standard retrieval performance; the foreground-only measures foreground-supported retrieval; and shortcut-only view measure how much identity signal is recoverable from background, or silhouette. This turns a retrieval score into an evidence profile: how well the model performs on the original images, how much of that performance remains when only the anim
    
[^7]: 平衡层次化表示与量化：文本嵌入中的性能权衡

    Balancing hierarchical representation and quantization: A performance trade-off in text embeddings

    [https://arxiv.org/abs/2611.06859](https://arxiv.org/abs/2611.06859)

    该研究通过每任务168种配置的大规模基准实验，首次系统评估了层次化嵌入表示与量化的交互作用，发现8-bit量化几乎不损害文本嵌入性能（最多损失1.5个百分点），而二值量化在所有任务类别中均显著退化，为文本嵌入的效率优化提供了实证权衡依据。

    

    arXiv:2611.06859v1 公告类型： new 摘要： 层次化文本嵌入模型提供可变长度的表示，而量化则降低数值精度；两者都旨在提高计算效率。本工作是一项实证基准研究，探讨这两种相互正交且已被广泛确立的技术之间的交互作用。我们使用四个基于Transformer的嵌入模型，在大规模文本嵌入基准（MTEB）的38个任务上进行评估，涵盖一维和二维层次化表示，并结合32位、8位和二值量化，每个任务共168种配置。我们分别报告每个效率维度：每个向量的理论存储开销和端到端评估时间。在Friedman检验及Nemenyi事后分析下，int8量化在配对分类和重排序任务中与float32在统计上无显著差异，总体性能损失最多为1.5个百分点；而二值量化在所有任务类别中均显著更差。将嵌入（原文在此处截断）……

    arXiv:2611.06859v1 Announce Type: new  Abstract: Hierarchical text embedding models offer variable-length representations, while quantization reduces numerical precision; both aim to improve computational efficiency. This work is an empirical benchmark study of how these two orthogonal, already-established techniques interact. We evaluate 38 tasks from the Massive Text Embedding Benchmark (MTEB) using four transformer-based embedding models, covering 1D and 2D hierarchical representations combined with 32-bit, 8-bit, and binary quantization, for 168 configurations per task. We report each efficiency dimension separately: theoretical per-vector storage and end-to-end evaluation time. Under a Friedman test with Nemenyi post-hoc analysis, int8 quantization is statistically indistinguishable from float32 for pair classification and re-ranking and costs at most 1.5 percentage points overall, whereas binary quantization is significantly worse across every task category. Halving the embedding
    
[^8]: 视觉令牌路由：学习保留重要的令牌

    Visual Token Routing: Learning to keep tokens that matter

    [https://arxiv.org/abs/2611.06854](https://arxiv.org/abs/2611.06854)

    该论文提出了视觉令牌路由（VTR）这一联合训练框架，利用轻量级时空路由器筛选截图中真正关键的视觉令牌，在保留动作预测所需信息的同时大幅降低计算机使用智能体的计算成本。

    

    计算机使用智能体（CUAs）近来在多个长时程基准测试中取得了超越人类的性能表现。然而，其计算成本限制了它们在长时程任务以及视频游戏等实时领域中的应用。一个关键瓶颈在于将每张截图表示为数千个视觉令牌，而随着智能体在长期交互历史中进行推理，这些令牌会不断累积形成漫长的上下文。我们观察到，这些视觉令牌中的大多数要么在语义上与智能体的下一个动作无关，要么在时间维度上跨截图存在冗余。基于这一观察，我们提出了视觉令牌路由（VTR），这是一个用于学习稀疏视觉表示的联合训练框架，旨在保留动作预测所需的信息。在VTR中，我们将视觉语言模型与一个轻量级的时空路由器联合训练，该路由器检查当前及此前截图中的每个图像区域，并判断其视觉令牌……（摘要在此处截断）

    arXiv:2611.06854v1 Announce Type: cross  Abstract: Computer-use agents (CUAs) have recently achieved superhuman performance on several long-horizon benchmarks. However, their computational cost limits their use in long-horizon tasks and real-time domains such as video games. A key bottleneck is representing each screenshot as thousands of visual tokens, which accumulate into long contexts as agents reason over extended interaction histories. We observe that most of these visual tokens are either semantically irrelevant to the agent's next action or temporally redundant across screenshots. Building on this observation, we introduce Visual Token Routing (VTR), a joint training framework for learning sparse visual representations that preserve the information needed for action prediction. In VTR, we train the vision-language model together with a lightweight spatio-temporal router that examines each image region in the current and previous screenshots and decides whether its visual token 
    
[^9]: 应对解释忠实性评估标准之间的分歧

    Navigating Disagreement Among Criteria for Evaluating Explanation Faithfulness

    [https://arxiv.org/abs/2611.06846](https://arxiv.org/abs/2611.06846)

    本文提出“忠实性泛函”框架，系统重构了现有的解释忠实性评估标准，揭示其分歧根源在于对解释主张、相关比较及一致性聚合方式的不同选择，为忠实性评估提供了实用的解决路径。

    

    忠实性要求算法解释能够准确反映模型的真实行为。然而，现有的忠实性操作化定义可能会对同一解释是否忠实得出相互矛盾的结论，这使得开发者和审计者不得不在相互冲突的评估结果中进行选择。在本文中，我们研究了评估标准之间的分歧，并为评估解释忠实性提供了一条实用的前进路径。我们首先分析了忠实性各操作化定义之间的差异。我们引入了一个“忠实性泛函”，它将各种分歧定位于三个方面：解释对模型所作的主张、哪些比较与该主张相关，以及这些比较之间的一致性应如何聚合，并在此框架内重构了现有的评估标准。我们论证这些重构支持将忠实性理解为模型与解释之间一种共同的、但尚未被完全明确规定的结构关系，因此各标准之间的差异反映的是不同的选择。

    arXiv:2611.06846v1 Announce Type: new  Abstract: Faithfulness requires that algorithmic explanations accurately reflect the model's behavior. Yet existing operationalizations of faithfulness can disagree about whether the same explanation is faithful, leaving developers and auditors to choose among conflicting evaluations. In this paper, we investigate disagreement among evaluation criteria and offer a practical path forward for evaluating explanation faithfulness. We first identify how operationalizations of faithfulness differ. We introduce a faithfulness functional that locates their disagreements in what an explanation claims about the model, which comparisons bear on that claim, and how agreement across those comparisons should be aggregated, and we reconstruct existing evaluation criteria within it. We argue these reconstructions support understanding faithfulness as a common though incompletely specified relation between the model and explanation, so the differences reflect choi
    
[^10]: 多晶材料晶粒生长的热条件化时间序列预测

    Thermally Conditioned Time-Series Forecasting of Grain Growth in Polycrystalline Materials

    [https://arxiv.org/abs/2611.06843](https://arxiv.org/abs/2611.06843)

    本研究将热信息显式融入循环神经网络预测框架，把等温条件下的晶粒生长时间序列预测方法扩展到非等温热处理场景，从而能够快速预测变温热循环下晶粒尺寸分布的演化。

    

    热处理过程中的晶粒生长会强烈影响多晶材料的微观组织，进而影响其力学性能。尽管前沿跟踪或前沿捕获等数值模拟方法能够准确描述晶粒生长，但其高昂的计算成本限制了其在长时间或非等温热循环中的应用。在以往的工作中，研究者通过将晶粒尺寸分布视为时间序列，并利用循环神经网络预测其演化，成功实现了等温条件下的晶粒生长预测，使得完整的晶粒生长轨迹可以在几秒钟内生成，而无需依赖成本高昂的基于偏微分方程的模拟。在本工作中，该预测框架被重新构建，通过在模型架构中显式引入热信息来处理非等温条件。晶粒尺寸分布使用卷积神经网络进行编码，热描述信息……

    arXiv:2611.06843v1 Announce Type: cross  Abstract: Grain growth during heat treatment strongly affects the microstructure of polycrystalline materials and, consequently, their mechanical properties. Although numerical methods such as front-tracking or front-capturing simulations can accurately describe grain growth, their high computational cost limits their use for long or non-isothermal thermal cycles. In previous work, grain growth was successfully predicted under isothermal conditions by treating grain size distributions as time series and forecasting their evolution using recurrent neural networks, allowing complete grain growth trajectories to be generated in seconds rather than through costly PDE-based simulations. In the present work, this forecasting framework is reformulated to handle non-isothermal conditions by explicitly incorporating thermal information into the model architecture. Grain size distributions are encoded using a convolutional neural network, thermal descript
    
[^11]: 布局相同，数量不同：数字词在文生图模型中超过十之后的作用

    Same Layout, Different Count: What Number Words Do Past Ten in Text-to-Image Models

    [https://arxiv.org/abs/2611.06840](https://arxiv.org/abs/2611.06840)

    该研究通过3,088张图像的人工计数发现，文生图模型仅在各自特定的“可达范围”内（约十至二十五）遵循数字词指令，超出该范围后图像布局由起始噪声的粗粒度部分固定，而仅重绘细粒度噪声即可像更换数字一样改变画面中的物体数量。

    

    让SD3.5-Medium生成十二个某个物体，然后在相同的起始噪声下请求二十五个。图像保持其布局，但物体数量的变化幅度与更换随机种子时相当，且没有一致的方向。我们研究了在五个开源文生图模型中，当请求的物体数量超过约十之后，数字词究竟起什么作用。在固定物体和起始噪声的前提下，我们每次只改变一个变量，并由从未见过请求文本的人员对其中3,088张图像进行了计数。每个模型在数量较小时都能遵循数字指令，但只到某个限度，即其“可达范围”：SD3.5-M和FLUX.1-dev在超过十之后很快不再遵循数字，而FLUX.2 [klein]和Qwen-Image则能大致遵循到二十五。超出SD3.5-M的可达范围后，起始噪声的粗粒度部分决定了图像的构图，但并不决定其中物体的数量：仅重绘比图像八分之一更精细的噪声，既能像新的数字一样保持布局，并且已经……

    arXiv:2611.06840v1 Announce Type: cross  Abstract: Ask SD3.5-Medium for twelve of an object and then, from the same starting noise, for twenty-five. The picture keeps its layout, but the count changes about as much as it would under a new random seed, and in no consistent direction. We study what number words do once a request passes about ten objects, in five open text-to-image models. Holding the object and the starting noise fixed, we change one thing at a time, and people who never see the request count 3,088 of the images. Every model follows the number for small crowds, but only up to a point, its reach: SD3.5-M and FLUX.1-dev stop following it soon after ten, while FLUX.2 [klein] and Qwen-Image follow it loosely up to twenty-five. Beyond SD3.5-M's reach, the coarse part of the starting noise fixes the composition of the picture but not how many objects it holds: redrawing only the noise finer than an eighth of the image keeps the layout as well as a new number does, and already 
    
[^12]: 上下文-答案元模型：在首个答案词元生成之前预测大语言模型激活

    Context--Answer Metamodels: Forecasting LLM Activations Before the First Answer Token

    [https://arxiv.org/abs/2611.06837](https://arxiv.org/abs/2611.06837)

    本文提出上下文-答案元模型，证明语言模型阅读上下文时产生的激活序列可以直接高保真地预测其在随机解码推演中产生的答案激活，且性能随训练数据增长而提升并能零样本泛化到域外数据集。

    

    在随机解码条件下，单个上下文可以从语言模型中引出许多不同的答案。表征这种变化通常需要大量昂贵的自回归推演，而每次推演只能揭示一个实现样本。我们提出这样一个问题：模型在阅读上下文时产生的激活能否直接预测模型推演时的激活。我们训练了上下文-答案元模型，将上下文激活序列映射为平均池化答案激活的点预测或分布预测。我们发现：（i）我们研究的四种元模型家族能够高保真地预测平均池化的答案激活，且分布元模型还能捕捉推演过程中的变异性；（ii）随着训练集增长到50万个提示，性能不断提升，并且元模型能够零样本泛化到域外数据集；（iii）它们的预测保留了可预测模型行为的信号；（iv）来自Qwen2.5的上下文激活能够预测平均池化的答案激活。

    arXiv:2611.06837v1 Announce Type: cross  Abstract: Under stochastic decoding, a single context can elicit many different answers from a language model. Characterizing this variation typically requires many expensive autoregressive rollouts, each revealing only one realization. We ask whether the activations produced while reading the context can directly forecast the model's rollout activations. We train context--answer metamodels that map a context activation sequence to point or distribution forecasts of mean-pooled answer activations. We find that (i) the four metamodel families we study forecast mean-pooled answer activations with high fidelity, and the distribution metamodels also capture rollout variability; (ii) performance improves as the training set grows to 500,000 prompts, and the metamodels generalize zero-shot to out-of-domain datasets; (iii) their forecasts retain behavior-predictive signal; and (iv) context activations from Qwen2.5 predict mean-pooled answer activations
    
[^13]: 生成式AI中的对齐近视

    Alignment Myopia in Generative AI

    [https://arxiv.org/abs/2611.06830](https://arxiv.org/abs/2611.06830)

    该论文揭示了当前生成式AI对齐的“近视”缺陷——过度追求即时对话满意度而忽视长期负面影响（如认知萎缩与能动性减弱），并提出转向长程对齐范式，在人类福祉四个维度上权衡近期效用与长期成本。

    

    当前的生成式AI对齐依赖于一种短视的范式，即优先考虑即时对话满意度。由于依赖短程反馈，这种方法忽视了持续人机交互所带来的深远纵向外部性，可能导致认知萎缩、社会侵蚀以及人类能动性减弱等不良后果。为解决这一问题，我们提出向长程对齐的范式转变，即考量AI如何随时间重塑人类体验。为此，我们在人类福祉的四个维度上刻画了AI使用的近期效用与长期成本之间的张力，以揭示当前奖励模型未能捕捉的结果。最终，我们倡导超越简单的偏好优化，转向能够驾驭这些复杂时间尺度、积极促进人类长期福祉的对齐策略。

    arXiv:2611.06830v1 Announce Type: cross  Abstract: Current generative AI alignment relies on a myopic paradigm that prioritizes immediate conversational satisfaction. By relying on short-horizon feedback, this approach ignores the profound, longitudinal externalities of persistent human-AI interaction, which can lead to adverse outcomes such as cognitive atrophy, social erosion, and diminished human agency. To address this, we propose a paradigm shift toward long-horizon alignment that accounts for how AI reshapes human experience over time. To this end, we map the tensions between near-term utility and long-term costs of AI use across four facets of human well-being to reveal outcomes that current reward models fail to capture. Ultimately, we advocate for moving beyond simple preference optimization toward alignment strategies that actively foster long-term human well-being by navigating these complex temporal scales.
    
[^14]: 使用高级Vitis HLS实现AWS云FPGA开发

    Enabling AWS Cloud FPGA Development Using High-Level Vitis HLS

    [https://arxiv.org/abs/2611.06829](https://arxiv.org/abs/2611.06829)

    该论文提出了一套完整的从Vitis HLS到AWS云FPGA部署的工具链和四级误差定位方法论，使开发者乃至AI编码智能体能够通过高级语言在AWS云FPGA上高效开发、测试和部署机器学习模型。

    

    我们提出了一套在AWS EC2 F2云FPGA基础设施上使用高级语言开发机器学习模型的工具链，实现了一个完整的从Vitis HLS到HDK桥接再到AFI的工作流程——尽管AWS FPGA已面世八年，这一流程在学术文献中始终缺失。我们引入了一种四级方法论（FP32 PyTorch、位精确量化Python、HLS C仿真、物理FPGA），以帮助定位错误，包括FPGA特有的电路错误。这些工具共同使编码智能体能够根据用户指令有效地实现FPGA设计，并按需进行测试和调试。我们记录了部署工作流程、时序收敛策略以及被证明至关重要的实用HLS工程决策。我们使用针对命名实体识别任务微调的TinyBERT进行验证，在FPGA硬件上实现了嵌入层和四个编码器层中的第一层。我们的实现从主机DRAM流式传输权重。

    arXiv:2611.06829v1 Announce Type: cross  Abstract: We present a toolchain for development of ML models on AWS EC2 F2 cloud FPGA infrastructure from high-level language, implementing a complete Vitis HLS to HDK bridge to AFI workflow that has been absent from the academic literature despite eight years of AWS FPGA availability. We introduce a four-level methodology (FP32 PyTorch, bit-accurate quantized Python, HLS C-simulation, physical FPGA) to aid in localizing errors, including FPGA-specific circuit errors. Together these tools allow coding agents to effectively implement FPGA designs from user directions, testing and debugging as needed. We document the deployment workflow, timing closure strategies, and practical HLS engineering decisions that proved crucial. We validate using TinyBERT fine-tuned for address Named Entity Recognition, implementing the embedding layer and the first of four encoder layers on the FPGA hardware. Our implementation streams weights from host DRAM, yieldin
    
[^15]: 脉动阵列线性注意力：面向TPU协同设计的分块分组门控Delta注意力

    Systolic Linear Attention: Co-Designing Tile-Grouped Gated Delta Attention for TPUs

    [https://arxiv.org/abs/2611.06828](https://arxiv.org/abs/2611.06828)

    提出了脉动线性注意力（SLA），针对TPU的256×256脉动矩阵单元重新设计门控Delta线性注意力——将键扩展至256、把衰减门分组为64通道瓦片并用全分块矩阵乘法的单一融合内核计算——在TPU v7x上比Tokamax KDA内核快1.6至2.5倍。

    

    门控Delta线性注意力层，如门控DeltaNet（GDN）、Kimi Delta Attention（KDA）和GDN-2，都是针对GPU调优的——在GPU上，小尺寸张量核心瓦片和快速的向量寄存器使得每通道衰减重缩放和块内顺序求解的代价很低。而Google TPU将计算集中在宽256×256的脉动矩阵单元中，并辅以较慢的向量单元，直接移植这些层会导致每个矩阵瓦片有一半空间空置，且性能瓶颈落在向量运算上。我们提出了脉动线性注意力，它围绕矩阵单元重新塑造门控Delta注意力：将键维度扩展到256并采用分组值头，将衰减门分组为四个64通道瓦片，并在单个融合内核中完全通过分块矩阵乘积来计算块求解。在TPU v7x上，一个完整的SLA-R4层比开源的Tokamax KDA内核快1.6倍至2.5倍（使用精确分块求逆时为1.4倍至2.4倍）

    arXiv:2611.06828v1 Announce Type: new  Abstract: Gated-delta linear attention layers such as Gated DeltaNet (GDN), Kimi Delta Attention (KDA), and GDN-2 are tuned for GPUs, where small tensor-core tiles and fast vector registers make per-channel decay rescaling and sequential in-block solves cheap. On Google TPUs, which concentrate compute in wide $256 \times 256$ systolic matrix units backed by a slower vector unit, porting these layers directly leaves half of each matrix tile empty and bottlenecks on vector operations. We present \textbf{Systolic Linear Attention (SLA)}, which reshapes gated delta attention around the matrix unit by widening keys to $256$ with grouped value heads, grouping decay gates into four $64$-channel tiles, and computing the chunk solve entirely with block matrix products in a single fused kernel. On TPU v7x, a full \texttt{SLA-R4} layer runs $1.6\times$--$2.5\times$ faster than the open-source Tokamax KDA kernel ($1.4\times$--$2.4\times$ with exact chunk inve
    
[^16]: GFlowQEC：学习采样量子LDPC码的生成流网络

    GFlowQEC: Generative Flow Networks that Learn to Sample Quantum LDPC Codes

    [https://arxiv.org/abs/2611.06819](https://arxiv.org/abs/2611.06819)

    提出首个用于量子码设计的生成流网络GFlowQEC，它学习按得分比例采样生成量子LDPC码，在昂贵的电路级仿真评估预算下，找到的全新高分码数量是并行回火方法的两倍以上。

    

    为真实设备选择量子纠错码，意味着需要用电路级仿真对候选码进行评分，而这类仿真极其缓慢，只能运行几百次。因此每一次评估都弥足珍贵，搜索方法的好坏取决于它在每次付费评估中能找到多少个优质码。由少数群元素构建的量子LDPC码可以保护多个逻辑量子比特，但优质的码在巨大的搜索空间中十分稀有。我们提出了GFlowQEC，据我们所知，这是首个用于量子码设计的生成流网络（GFlowNet）。它每次添加一个群元素来逐步构建码，并学习按照候选码得分的幂次成比例地提出新码。我们使用来自早期搜索的得分对模型进行一次训练，然后将其冻结，让它在180个物理量子比特的规模上提出提升乘积码，并以码距作为评分标准。在每次付费调用中对其十个采样结果进行排序，它找到的新的最高分码（与训练中见过的任何码均不等价）数量是经过调优的并行回火算法的两倍以上，即使回火算法……

    arXiv:2611.06819v1 Announce Type: cross  Abstract: Choosing a quantum error-correcting code for a real device means scoring candidates with circuit-level simulations so slow that only a few hundred can be run. So every evaluation counts, and a search is judged by how many good codes it finds per paid evaluation. Quantum LDPC codes built from a few group elements protect many logical qubits, but good ones are rare in a huge space. We introduce GFlowQEC, to our knowledge the first generative flow network (GFlowNet) for quantum code design. It builds a code one group element at a time and learns to propose codes in proportion to a power of their score. We train it once on scores from earlier searches, freeze it, and let it propose lifted-product codes with 180 physical qubits, scored by their distances. Ranking ten of its draws per paid call, it finds more than twice as many new top-scoring codes (inequivalent to any it was trained on) as tuned parallel tempering, even when tempering star
    
[^17]: 面向结构-性质关系发现的可解释空间描述子的进化设计

    Evolutionary Design of Interpretable Spatial Descriptors for Structure-Property Relationship Discovery

    [https://arxiv.org/abs/2611.06811](https://arxiv.org/abs/2611.06811)

    该论文提出了一种结合SHAP空间归因分析与遗传算法优化空间分区的框架，用于构建可解释的空间描述子，在预测精度与描述子维度之间取得平衡，以发现结构-性质关系。

    

    结构-性质预测是机器学习中的经典任务，应用范围涵盖从分子活性、材料性质到显微镜下测量的局部功能响应等多个领域。对于基于图像的回归任务，有两个选择决定了预测器可获取的信息：图像邻域的大小以及其中保留的空间分辨率。较小的邻域可能遗漏相关的周围结构，而较大且精细采样的邻域则可能引入冗余或不相关的信息。在此，我们开发了一个用于构建空间描述子的框架，以预测局部光谱和滞回回线。我们对回归模型进行了基准测试，利用SHAP图来考察模型的空间归因，并探索补丁大小与压缩的影响，以建立预测精度与描述子维度之间的关系。随后，我们使用遗传算法来优化空间分区以及测量数据的分配。

    arXiv:2611.06811v1 Announce Type: cross  Abstract: Structure property prediction is a classical task in machine learning, ranging from molecular activity and materials properties to local functional responses measured in microscopy. For image-based regression, two choices define the information available to the predictor: the size of the image neighborhood and the spatial resolution retained within it. A small neighborhood can omit relevant surrounding structure, whereas a large, finely sampled neighborhood can introduce redundant or irrelevant information. Here, we develop a framework for constructing spatial descriptors for prediction of local spectra and hysteresis loops. We benchmark regression models, use SHAP maps to examine their spatial attribution, and explore the effect of the patch size and compression to establish the relationship between predictive accuracy and descriptor dimensionality. We then use a genetic algorithm to optimize spatial partitions and the assignment of m
    
[^18]: ConDR：基于神经网络汉密尔顿-雅可比下解的保守可达性分析

    ConDR: Conservative Reachability through Neural Hamilton-Jacobi Subsolutions

    [https://arxiv.org/abs/2611.06802](https://arxiv.org/abs/2611.06802)

    提出 ConDR 方法，通过单边损失和学习 HJI 变分不等式的下解来保证神经网络估计的危险集保守地包含真实危险集，同时避免退化解，实现了高维系统中安全可靠的可达性分析。

    

    汉密尔顿-雅可比（HJ）可达性分析用于识别动力系统中的不安全状态，即在最坏情况扰动下系统可能被强制进入失败的那些状态。然而，经典的基于网格的求解器随状态维数呈指数级扩展。DeepReach 通过用基于 HJ-Isaacs（HJI）变分不等式（VI）训练的神经值函数取代网格来提升可扩展性，但它对残差的两侧施加同等的惩罚。这导致学到的值可能高于真实值，从而使不安全状态被错误地分类为安全。我们提出保守 DeepReach（ConDR），旨在学习 HJI 变分不等式的一个下解，使得学到的危险集包含真实的危险集。ConDR 通过构造方式强制执行障碍分支和终端条件，利用单边损失将 HJI 残差推至低于零的 $\epsilon$ 安全裕度，并防止将所有状态都判为不安全的退化解。我们证明了……（摘要在此处截断）

    arXiv:2611.06802v1 Announce Type: cross  Abstract: Hamilton-Jacobi (HJ) Reachability identifies the unsafe states of a dynamical system from which it can be forced into failure under worst-case disturbance. However, classical grid-based solvers scale exponentially with state dimension. DeepReach improves scalability by replacing the grid with a neural value function trained on the HJ-Isaacs (HJI) variational inequality (VI), but penalizes both sides of the residual equally. This allows the learned value to lie above the true one and unsafe states to be misclassified as safe. We propose Conservative DeepReach (ConDR) which aims to learn a subsolution of the HJI VI such that the learned unsafe set contains the true one. ConDR enforces the obstacle branch and terminal condition by construction, pushes the HJI residual to a safety margin $\epsilon$ below zero with a one-sided loss, and prevents the degenerate solution of classifying everything as unsafe. We prove that the exact solution of
    
[^19]: 消息传递图持久性

    Message-passing graph persistence

    [https://arxiv.org/abs/2611.06801](https://arxiv.org/abs/2611.06801)

    本文提出消息传递图持久性（MGP），将消息传递机制融入图过滤过程构建拓扑描述子，并证明了其图同构不变性以及严格强于现有基于持久同调描述子的表达能力。

    

    消息传递图神经网络的表达能力受限于其一维Weisfeiler-Leman测试。持久同调提供了可以补充GNN表示的拓扑描述子，但现有方法通常将持久性的计算与表示学习过程分开进行。在本文中，我们提出了消息传递图持久性，这是一类将消息传递机制融入图过滤过程的拓扑描述子。在每个过滤步骤中，节点嵌入利用当前子图进行更新，并与持久性事件相关联。这些嵌入还在持久性计算过程中决定连通分量的代表元。我们证明了MGP在图同构下保持不变，并且其表达能力严格强于其所扩展的基于持久同调的描述子。我们在表达能力基准测试以及多个图分类和回归任务上对MGP进行了评估。

    arXiv:2611.06801v1 Announce Type: cross  Abstract: Message-passing graph neural networks (GNNs) are limited in expressive power by the one-dimensional Weisfeiler-Leman test. Persistent homology (PH) provides topological descriptors that can complement GNN representations, but existing approaches typically compute persistence separately from the representation learning process. In this paper, we introduce Message-Passing Graph Persistence (MGP), a family of topological descriptors that incorporates message passing into graph filtrations. At each filtration step, node embeddings are updated using the current subgraph and associated with persistence events. These embeddings also determine the representatives of connected components during persistence computation. We prove that MGP is invariant under graph isomorphism and that it is strictly more expressive than the PH-based descriptors it extends. We evaluate MGP on an expressivity benchmark and on several graph classification and regress
    
[^20]: 将整数网格转化为空间直觉：主导ARC-AGI-3的是感知而非规划

    Transforming Integer Grids to Spatial Intuition: Perception, Not Planning, Governs ARC-AGI-3

    [https://arxiv.org/abs/2611.06793](https://arxiv.org/abs/2611.06793)

    该论文提出开源智能体SARBLOH，证明在ARC-AGI-3交互推理中起决定作用的是空间感知而非规划能力——其零参数O(HW)感知引擎可在35毫秒内将整数网格帧转化为对象清单与几何简报，从而大幅减少前沿模型因序列化破坏空间拓扑而产生的高昂探索浪费。

    

    样本高效的流体智能需要能够快速建立陌生交互环境工作模型的智能体。ARC-AGI-3通过无指令视觉游戏对此进行测试，游戏采用“相对人类动作效率”评分，该评分会以二次方方式惩罚探索性浪费。当前沿模型读取原始整数网格时，序列化破坏了空间拓扑结构，导致它们需要花费数百个动作来重新发现基本的边界和对称性。我们提出了SARBLOH，一个开源权重的智能体，其建立在一个前提之上：ARC-AGI-3上的交互推理由空间感知而非规划能力所主导。一个零参数、复杂度为O(HW)的感知引擎能在35毫秒内将每一帧转化为包含二部图时间跟踪的对象清单、区分已验证不变量与可供性假设的几何简报，以及一个带注释的画布。配合“代码即动作”的运行时机制，该智能体编写的是算法级假设而非低级动作。在八个……（摘要在此处截断）

    arXiv:2611.06793v1 Announce Type: cross  Abstract: Sample-efficient fluid intelligence requires agents that rapidly form working models of unfamiliar interactive environments. ARC-AGI-3 tests this across uninstructed visual games scored by Relative Human Action Efficiency, which penalizes exploratory waste quadratically. When frontier models read raw integer grids, serialization destroys spatial topology, and they spend hundreds of actions rediscovering basic boundaries and symmetries. We introduce SARBLOH, an open-weight agent built on the premise that spatial perception, not planning capacity, governs interactive reasoning on ARC-AGI-3. A zero-parameter O(HW) perception engine turns each frame, in 35 ms, into an object inventory with bipartite temporal tracking, a geometric briefing that separates proven invariants from affordance hypotheses, and an annotated canvas. Paired with a code-as-action runtime, the agent writes algorithmic hypotheses instead of low-level moves. Across eight
    
[^21]: 排名一致性无法揭示两个已发布论文录用预测器之间的段落编辑一致性

    Ranking Agreement Does Not Reveal Paragraph-Edit Agreement in Two Released Paper Acceptance Predictors

    [https://arxiv.org/abs/2611.06789](https://arxiv.org/abs/2611.06789)

    研究发现，两个论文录用预测器尽管在原始论文排名上高度一致，但在判断哪些段落改写能提升分数时几乎没有一致性，且降低的数值精度产生的舍入误差会掩盖微小的分数变化。

    

    快速的录用分类器可以为较慢的评估器筛选修改版本。我们探讨了对现有论文的排名一致性是否支持这种用途。我们将一个基于词语和文档特征的分类器与两个已发布的纯文本录用预测器进行比较，这两个预测器均从同一个拥有三十亿参数的语言模型微调而来。针对国际学习表征会议（ICLR）的198篇论文，我们比较了模型对原始论文以及对段落改写版本的排名，这些改写版本经过自动筛选，确保其事实、论点强度和引用来源保持不变。该分类器对原始论文的排名与其中一个预测器高度接近，与另一个中度一致，但在判断哪些改写会提高分数方面，与两者均无可检测到的一致性。预先设定的测试使用了降低的数值精度，舍入误差主导了微小的分数响应。因此，我们事后以完整的32位精度对已发布的权重重新打分；编辑一致性仍然接近（零）。

    arXiv:2611.06789v1 Announce Type: new  Abstract: A fast acceptance classifier could select revisions for a slower evaluator. We ask whether agreement on existing papers supports that use. We compare a classifier of word and document features with two released text-only acceptance predictors fine-tuned from the same language model with three billion parameters. For 198 papers from the International Conference on Learning Representations, we compare model rankings of the original papers and of paragraph rewrites screened automatically for unchanged facts, claim strength, and attribution. The classifier ranks original papers closely with one predictor and moderately with the other, but shows no detectable agreement with either on which rewrites raise scores. The tests specified in advance used reduced numerical precision. Rounding error dominated the small score responses. We therefore rescored the released weights in full 32-bit precision as a post hoc check; edit agreement remained near
    
[^22]: FedFiOT：用于单轮和小轮次联邦学习中模型合并的Fisher加权最优传输方法

    FedFiOT: Fisher-Weighted Optimal Transport for Model Merging in One-Shot and Few-Round Federated Learning

    [https://arxiv.org/abs/2611.06783](https://arxiv.org/abs/2611.06783)

    FedFiOT提出了一种基于Fisher信息加权的最优传输框架，用于单轮和小轮次联邦学习，通过量化参数重要性来指导神经元对齐与模型聚合，解决了客户端模型漂移和数据异质性导致的直接平均性能下降问题。

    

    单轮和小轮次联邦学习（FL）通过在一次或少数几次通信轮次内构建全局模型来减少通信负担，但其性能通常低于传统的多轮次联邦学习。由于缺乏频繁的同步，客户端模型可能发生漂移，导致学习相似特征的神经元在各客户端占据不同的位置。同时，本地数据的异质性可能导致相应参数对每个客户端预测的重要性存在差异。这些挑战共同导致直接平均法会合并不匹配的神经元，并且尽管参数重要性不同，仍赋予它们相同的权重。我们提出了FedFiOT，一个面向单轮和小轮次联邦学习的Fisher加权最优传输框架，它利用Fisher信息来量化参数重要性，并指导神经元对齐和模型聚合。FedFiOT联合优化神经元对应关系和全局模型……

    arXiv:2611.06783v1 Announce Type: cross  Abstract: One-shot and few-round federated learning (FL) reduce communication burden by constructing a global model within one or a few communication rounds, but often achieve lower performance than conventional multi-round FL. Without frequent synchronization, client models may drift apart, causing neurons that learn similar features to occupy different positions across clients. At the same time, local data heterogeneity can cause corresponding parameters to differ in their importance to each client's predictions. Together, these challenges cause direct averaging to combine mismatched neurons and assign equal weight to parameters despite differences in their importance. We propose FedFiOT, a Fisher-weighted optimal transport framework for one-shot and few-round FL that leverages Fisher information to quantify parameter importance and guide neuron alignment and model aggregation. FedFiOT jointly optimizes neuron correspondences and global model 
    
[^23]: 超越核机制的持续分类：利用对角线性网络中L2正则化的稀疏性偏好

    Continual Classification Beyond the Kernel Regime: Harnessing the Sparsity Bias of L2 Regularization in Diagonal Linear Networks

    [https://arxiv.org/abs/2611.06759](https://arxiv.org/abs/2611.06759)

    该论文推导了持续线性分类的最优L2正则化强度并刻画其泛化性能，同时利用两层对角线性网络揭示了L2正则化的稀疏性偏好可以在持续学习中实现超越核机制的特征学习。

    

    我们研究了高维统计设定下使用线性预测器进行持续学习的逻辑回归问题。现有理论工作大多集中于采用L2正则化变体的持续线性回归，这源于其分析上的简洁性以及与神经网络核机制的联系。我们首先推导了持续线性分类的最优L2正则化强度，并刻画了其泛化性能，得到了与先前持续线性回归结果相类似的行为。接下来，我们使用一个极简的两层对角线性网络，考察L2正则化在更深模型中的影响。在单任务设定中，已知对此类模型权重的L2正则化会促使最终预测器呈现稀疏性，从而产生超越核机制的丰富的稀疏特征学习机制（类似效应也出现在两层ReLU网络中）。然而，在持续学习中……

    arXiv:2611.06759v1 Announce Type: new  Abstract: We study logistic regression for continual learning with linear predictors in a high-dimensional statistical setting. Existing theoretical work has largely focused on continual linear regression with variants of L2 regularization, motivated by their analytical simplicity and connection to the kernel regime of neural networks. We first derive the optimal L2 regularization strength for continual linear classification and characterize its generalization performance, recovering behavior analogous to prior results for continual linear regression. Next, we examine the effect of L2 regularization in deeper models using a minimal two-layer diagonal linear network. In the single-task setting, L2 regularization on the weights of such models is known to promote sparsity in the resulting predictor, yielding a rich, sparse feature-learning regime beyond the kernel regime (a similar effect arises in two-layer ReLU networks). However, in continual lear
    
[^24]: 耦合场景图预测：当带约束召回率掩盖结构不一致性时

    Coupled Scene Graph Anticipation: When With-Constraint Recall Hides Structural Inconsistency

    [https://arxiv.org/abs/2611.06738](https://arxiv.org/abs/2611.06738)

    本文提出CoupledSGA，通过关系交互图与结构一致性损失耦合共享实体的关系以避免互斥预测，并证明带约束评估指标会掩盖这种结构不一致性。

    

    场景图预测（SGA）用于预测未来视频帧中物体之间的关系。连续时间SGA模型使用独立的神经常微分方程（Neural ODE）来演化每个主语-宾语对，因此在积分过程中共享同一实体的关系无法相互影响，并且没有任何机制阻止对同一对物体做出互斥的预测，例如“坐在...上”和“站在...上”。我们提出了CoupledSGA，它通过在ODE向量场内部嵌入图注意力的关系交互图（Relationship Interaction Graph）来关联这些关系，并添加了基于互斥谓词矩阵的结构一致性损失（L_SC）。开发CoupledSGA的过程揭示了现有评估协议的一个局限性。带约束评估为每个主语-宾语对在每个组中仅保留一个谓词，因此其R@K和mR@K无法捕捉L_SC所针对的组内违规情况：我们证明，只要每组的获胜谓词保持不变，这些指标就不会发生变化。

    arXiv:2611.06738v1 Announce Type: cross  Abstract: Scene Graph Anticipation (SGA) predicts object relationships in future video frames. Continuous-time SGA models evolve each subject-object pair with an independent Neural ODE, so relationships sharing an entity cannot influence one another during integration, and nothing prevents mutually exclusive predictions, such as "sitting on" and "standing on", for the same pair. We introduce CoupledSGA, which links such relationships through a Relationship Interaction Graph with graph attention inside the ODE vector field, and adds a Structural Consistency Loss ($\mathcal{L}_{SC}$) based on a matrix of mutually exclusive predicates.   Developing CoupledSGA revealed a limitation of the evaluation protocol. With-Constraint evaluation keeps one predicate per group for each subject-object pair, so its R@K and mR@K cannot capture the within-group violations targeted by $\mathcal{L}_{SC}$: we prove they are unchanged whenever each group's winning pred
    
[^25]: 规模化ConceptARC：研究推理模型中的概念理解

    ConceptARC at Scale: Studying Conceptual Understanding in Reasoning Models

    [https://arxiv.org/abs/2611.06732](https://arxiv.org/abs/2611.06732)

    该论文提出ConceptARC at Scale，一个能够大规模生成数千个针对特定概念的ARC风格任务的生成器，用于系统研究大型推理模型能否在概念相关的推理任务上形成可泛化的抽象能力。

    

    大型推理模型（LRMs）在具有挑战性的基准测试中表现优异，但它们能否在概念相关的推理任务之间形成可泛化的抽象仍不明确。抽象与推理语料库（ARC）为研究抽象能力提供了一个受控的测试平台，要求求解者从少量示例中推断二维网格上的变换规则。虽然ARC评估的是单个任务内的规则抽象能力，但不同任务可能共享更高层次的结构，即便它们需要不同的变换——例如，多个任务可能都涉及对“被包含在其他对象中的对象”进行推理。像ConceptARC这样的基准通过将ARC风格的任务按反复出现的空间和语义概念进行分组来捕捉这种结构，但它们通常每个概念只包含少量手工构建的任务。我们提出了ConceptARC at Scale，这是一个生成器，可以创建数千个针对特定概念的ARC风格任务。

    arXiv:2611.06732v1 Announce Type: new  Abstract: Large reasoning models (LRMs) perform well on challenging benchmarks, yet it remains unclear whether they form generalizable abstractions across conceptually related reasoning tasks. The Abstraction and Reasoning Corpus (ARC) provides a controlled testbed for studying abstraction, requiring solvers to infer transformation rules over two-dimensional grids from a small set of examples. While ARC evaluates rule abstraction within individual tasks, different tasks may share higher-level structure despite requiring different transformations$\unicode{x2014}$for example, several tasks may involve reasoning about objects contained within other objects. Benchmarks such as ConceptARC capture this structure by grouping ARC-style tasks into recurring spatial and semantic concepts, but they often contain only a few manually constructed tasks per concept. We introduce ConceptARC at Scale, a generator for creating thousands of ARC-style tasks targeting
    
[^26]: 基于小型语言模型的模板空缺重构Shell输出模拟方法

    Template Gap Reconstruction for Shell Output Simulation with Small Language Models

    [https://arxiv.org/abs/2611.06731](https://arxiv.org/abs/2611.06731)

    该论文提出模板空缺重构（TGR）方法，通过从1870万个真实输出构建的模板库中提供可复用的输出字面量并与自回归生成的空缺内容交错，使小型语言模型在Shell输出模拟中显著优于完整输出解码。

    

    小型语言模型为无需实际执行所提交命令的Shell模拟提供了紧凑的基础，但完整输出解码要求模型在预测跨执行而变化的内容的同时，还需预测反复出现的工具输出结构。我们提出了模板空缺重构（TGR）方法，该方法提供可复用的输出字面量，其中穿插着哨兵标记，以便与自回归生成的空缺内容交错拼接。模板选择器负责选择需要重构的模板，而空模板则保留为完整输出解码方式。我们从跨异构执行环境收集的1870万个输出中构建了一个模板库，并在一个未参与模板库构建的目标环境上，基于多种小型语言模型检查点初始化，训练了一个模板选择器和五个重构模型。评估结果表明，当最优模板覆盖更多内容时，TGR在输出编辑相似度（O-ES）指标上优于完整输出解码（FOD）……

    arXiv:2611.06731v1 Announce Type: cross  Abstract: Small language models offer a compact basis for shell simulation without executing submitted commands, but full-output decoding requires them to predict recurring utility structure alongside content that varies across executions. We introduce template gap reconstruction (TGR), which supplies reusable output literals interspersed with sentinels to be interleaved with autoregressively generated gaps. A template selector chooses which template to be reconstructed, with the empty template retaining full-output decoding. We construct a template bank from 18.7 million outputs collected across heterogeneous execution environments and train a template selector and five reconstruction models initialized from various small language model checkpoints on a target environment excluded from bank construction. Evaluation demonstrates that TGR improves output edit similarity (O-ES) over full-output decoding (FOD) when the optimal template covers more 
    
[^27]: 回合至关重要：通过回合级信用分配推进流式视频强化学习

    Turns Matter: Advancing Streaming Video RL via Turn-Level Credit

    [https://arxiv.org/abs/2611.06728](https://arxiv.org/abs/2611.06728)

    提出TurnWise——一个回合级GRPO框架，通过基于静默补全前缀评分的回合级信用分配和回合均值策略优化，解决了流式视频强化学习中轨迹级反馈与回合级决策不匹配的问题，从而促进有信息量的回应并减少冗余回复。

    

    流式视频理解要求模型反复决定是应当立即回应还是继续观察。最近，组相对策略优化（GRPO）已被应用于这一场景，但现有方法侧重于奖励设计，同时在各回合之间保留单一的轨迹级优势。这种共享信用可能会在早期成功之后强化冗余回复，并在早期失败之后抑制有用的后续回应。我们提出TurnWise，一个回合级的GRPO框架，使学习与各个流式决策对齐。TurnWise解决了轨迹级反馈与回合级交互之间的不匹配问题，促进有信息量的回应并减少冗余。具体而言，回合级信用分配采用静默补全的前缀评分，从每个决策的剩余回报构建组相对优势，将其学习信号与先前的奖励分离开来。回合均值策略优化通过对……（摘要原文在此处截断）

    arXiv:2611.06728v1 Announce Type: cross  Abstract: Streaming video understanding requires models to repeatedly decide whether to respond or continue observing. Recently, Group Relative Policy Optimization (GRPO) has been applied to this setting, with existing approaches emphasizing reward design while retaining a single trajectory-level advantage across turns. Such shared credit can reinforce redundant replies after earlier successes and discourage useful follow-ups after earlier failures. We propose TurnWise, a turn-level GRPO framework that aligns learning with individual streaming decisions. TurnWise addresses the mismatch between trajectory-level feedback and turn-level interaction, promoting informative responses and reducing redundancy. Specifically, turn-level credit assignment uses silence-completed prefix scoring to construct group-relative advantages from each decision's remaining return, separating its learning signal from earlier rewards. Turn-mean policy optimization avera
    
[^28]: 基础模型时代的灵巧机械手：硬件、数据、跨本体迁移、学习与评估的全栈式综述

    Dexterous Robotic Hands in the Foundation-Model Era: A Full-Stack Survey of Hardware, Data, Cross-Embodiment Transfer, Learning, and Evaluation

    [https://arxiv.org/abs/2611.06724](https://arxiv.org/abs/2611.06724)

    这是一篇全栈式综述，首次将基础模型时代的灵巧机械手研究系统性地组织为硬件、数据、跨本体迁移、学习与评估五个相互依赖的层次，并通过案例分析揭示各层设计选择之间的内在联系。

    

    灵巧机械手正日益与大规模人类数据和基础模型深度结合，然而硬件、表征学习与评估等方面的进展往往被割裂地单独评估。本综述回顾了2024年1月至2026年11月初的相关工作，并纳入更早期的研究以梳理技术发展脉络。我们将文献组织为五个相互依赖的层次：硬件；数据采集与整理；跨本体迁移；学习与基础模型；以及评估与应用。人手-物体交互理解与触觉感知提供了两个贯穿全文的横切视角。通过系统级架构、对比表格和五个案例分析，我们将各层次之间的设计选择串联起来。研究区分了单一研究内部的实验对比与异构语料库统计、厂商规格参数以及公司报告的结果。综述最终得出三点结论，其中第一点是：人类数据的价值取决于运动学、视觉等方面的特性。（原文摘要在此处截断）

    arXiv:2611.06724v1 Announce Type: cross  Abstract: Dexterous robotic hands are increasingly coupled to large-scale human data and foundation models, yet progress in hardware, representation learning and evaluation is often assessed separately. This survey reviews work from January 2024 to early November 2026, with earlier studies included to establish technical lineages. We organize the literature into five interdependent layers: hardware; data collection and curation; cross-embodiment transfer; learning and foundation models; and evaluation and applications. Human hand-object understanding and tactile sensing provide two cross-cutting perspectives. A system-level architecture, comparative tables and five case analyses connect design choices across these layers. We distinguish within-study experimental comparisons from heterogeneous corpus statistics, vendor specifications and company-reported results. Three conclusions emerge. First, the value of human data depends on kinematic, visua
    
[^29]: 通过中心流分析研究稳定性边缘处的对称性破缺

    Symmetry Breaking at the Edge of Stability via Central Flow Analysis

    [https://arxiv.org/abs/2611.06718](https://arxiv.org/abs/2611.06718)

    本文借助中心流分析首次系统研究了神经网络在稳定性边缘处的对称性破缺现象，证明曲率修正会引入由临界参数方向上振荡协方差决定的有限维漂移，从而打破了梯度流下的连续对称性守恒律（如层平衡性），并在标量两层模型、两层矩阵分解和两层ReLU网络中刻画了哪些守恒律得以保持、哪些被破坏。

    

    神经网络中的连续对称性会导致在梯度流下保持不变的守恒量，例如线性和ReLU网络中的层平衡性。有限步梯度下降并不一定保持这些守恒律，特别是在稳定性边缘处，此时迭代点围绕一条更平滑的轨迹振荡，而这条轨迹可以用中心流动力学（Cohen et al., 2025）来描述。我们研究了在这一区域的中心流描述下，梯度流守恒律如何演化。我们证明，曲率修正会引入一个由沿临界参数方向的振荡协方差所决定的有限维漂移。因此，我们通过分析标量两层模型、两层矩阵分解以及两层ReLU网络，迈出了在稳定性边缘处分析哪些守恒律得以保持、哪些被打破的第一步，其中已知的守恒律被证明是……

    arXiv:2611.06718v1 Announce Type: new  Abstract: Continuous symmetries in neural networks lead to conserved quantities that are invariant under gradient flow, such as layer balancedness in linear and ReLU networks. Finite-step gradient descent need not preserve these laws, particularly at the edge of stability (EOS), where the iterates oscillate around a smoother trajectory, which can be described by the central flow dynamics (Cohen et al., 2025). We study how gradient flow conservation laws evolve under the central flow description of this regime. We show that the curvature correction induces a finite-dimensional drift determined by the oscillation covariance along critical parameter directions. We thus take the first steps to analyse which conservation laws remain and are broken at the EOS according to the central flow dynamics by analysing a scalar two-layer model, two-layer matrix factorisation, and a two-layer ReLU network, where the known conservation laws are shown to be complet
    
[^30]: 面向无投影物理信息流匹配的能量引导传输

    Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching

    [https://arxiv.org/abs/2611.06711](https://arxiv.org/abs/2611.06711)

    提出了一种免训练的能量引导传输框架，将非线性物理约束建模为平滑势能场，无需离散投影即可在流匹配推理过程中保持物理一致性，并避免破坏概率路径和降低计算开销。

    

    深度生成模型，特别是流匹配框架，已成为合成复杂物理系统的强大工具。然而，标准架构仍然纯粹由统计驱动，对底层控制方程一无所知，在推理过程中常常违反基本物理约束。现有的旨在强制执行这些约束的零样本方法通常依赖于刚性的代数投影。虽然这类方法对稀疏边界条件有效，但将这种离散的几何校正应用于稠密的、覆盖全域的偏微分方程时，可能会破坏概率路径，导致偏离已学习的流形，并引入大量计算开销。为应对这些挑战，我们提出了能量引导传输，这是一种新颖的免训练框架，它将非线性物理约束表述为平滑的势能景观。我们的方法不是中断ODE积分过程，而是施加……

    arXiv:2611.06711v1 Announce Type: new  Abstract: Deep generative models, particularly Flow Matching frameworks, have emerged as powerful tools for synthesizing complex physical systems. However, standard architectures remain purely statistically driven and agnostic to the underlying governing equations, often violating fundamental physical constraints during inference. Existing zero-shot approaches designed to enforce these constraints typically rely on rigid algebraic projections. While effective for sparse boundary conditions, applying such discrete geometric corrections to dense, full-domain PDEs may disrupt the probability path, \rev{potentially deviating from the learned manifold and introducing substantial computational overhead}. To address these challenges, we introduce Energy-Guided Transport, a novel training-free framework that formulates non-linear physical constraints as a smooth potential energy landscape. Rather than interrupting the ODE integration, our method applies a
    
[^31]: ALTO：非洲与黎凡特地区技术助长的基于性别的暴力语料库

    ALTO: African and Levantine Tech-Facilitated Gender-Based Violence Corpus

    [https://arxiv.org/abs/2611.06706](https://arxiv.org/abs/2611.06706)

    该论文通过与约旦、西非和肯尼亚的民间社会组织合作，以社区主导的众包方式构建了首个涵盖黎凡特阿拉伯语、非洲法语和斯瓦希里语的技术助长性别暴力标注语料库ALTO，填补了现有性别伤害检测研究长期偏向全球北方语言的空白。

    

    技术助长的基于性别的暴力（TFGBV）是一种普遍存在的网络危害，通过多种直接和隐蔽的形式影响女性和性别少数群体。当前用于性别伤害检测的模型和标注数据集主要被来自全球北方语言的数据以及用这些语言训练的模型所主导。我们与来自约旦、法语西非和肯尼亚的民间社会组织（CSOs）合作，开发了一种社区主导的方法，通过众包方式收集黎凡特阿拉伯语、非洲法语和斯瓦希里语中符合当地文化和语言特点的TFGBV实例。我们使用主动学习对众包数据进行标注，构建了一个规模虽小但性能出色的数据集（N=3022），命名为ALTO，用于改进这三种语言的TFGBV检测。通过基准测试和微调实验，我们表明现有的有害内容检测模型在ALTO数据集上的TFGBV召回率较低，而经过微调后性能有所提升。我们发现模型往往无法识别隐蔽的……

    arXiv:2611.06706v1 Announce Type: cross  Abstract: Tech-facilitated gender based violence (TFGBV) is a pervasive online harm affecting women and gender minority groups through its many direct and covert forms. Current models and annotated datasets for gendered harm detection are dominated by data from, and models trained in, Global North languages. In collaboration with civil society organizations (CSOs) from Jordan, Francophone West Africa, and Kenya, we developed a community-led approach to crowdsource culturally and linguistically relevant instances of TFGBV in Levantine Arabic, African French, and Swahili. We annotated the crowdsourced data using active learning to develop a small but performant dataset (N=3022) called ALTO to improve TFGBV detection in the three languages. Through benchmarking and fine-tuning experiments, we show that existing harmful content detection models show poor TFGBV recall within ALTO and improve with fine-tuning. We found that models often miss covert ha
    
[^32]: 哪里出了问题：诊断视觉-语言智能体在吃豆人游戏中的感知、记忆与决策失败

    Where It Breaks: Diagnosing Perception, Memory, and Decision Failures of a Vision-Language Agent in Pac-Man

    [https://arxiv.org/abs/2611.06704](https://arxiv.org/abs/2611.06704)

    该研究通过对基于Qwen3.5-9B的吃豆人智能体分别进行感知检查和使用文本棋盘的基元策略强化学习，发现静态截图读取准确可能与交互中的错误并存、策略优化可能伴随更差的动作选择，从而为后训练提出了在使用中评估感知、区分感知与决策失败的具体检查方法。

    

    仅用上、下、左、右四个动作训练智能体玩吃豆人提出了一个基本问题：当表现不佳时，应该改进什么？困难可能来自读取屏幕，也可能来自在长序列中选择动作。我们通过对一个基于Qwen3.5-9B的智能体开展独立的感知检查和基元策略强化学习研究来考察这些问题。在策略研究中，我们使用文本棋盘将图像转录从决策接口中移除，八个留出迷宫的通关率从优化前的0%提升到选定检查点时的80.2%（附带三条额外生命）。感知检查表明，准确的静态截图读取可能与交互过程中的错误并存。另一项策略比较也显示，进一步的优化可能伴随更差的动作选择。这些发现共同为后训练提供了具体的检查方法：在使用中评估感知、区分（摘要在此处截断）……

    arXiv:2611.06704v1 Announce Type: new  Abstract: Training an agent to play Pac-Man with only up, down, left and right actions raises a basic question: when performance is poor, what should be improved? Difficulty can arise from reading the screen or from choosing actions over a long sequence. We examine these problems in a Qwen3.5-9B-based agent through separate perception checks and primitive-policy reinforcement-learning studies. In the policy study, which uses textual boards to remove image transcription from the decision interface, completion on eight held-out mazes increased from 0% before this optimization to 80.2% at a selected checkpoint, with three extra lives. The perception checks show that accurate static screenshot reading can coexist with errors during interaction. A separate policy comparison also shows that further optimization can accompany worse action choices. Together, these findings motivate concrete checks for post-training: evaluate perception in use, distinguish
    
[^33]: PINR-ESI：面向脑电扩展源成像的物理信息自监督隐式神经表示框架

    PINR-ESI: Physics-Informed Self-Supervised Implicit Neural Representation Framework for EEG Extended-Source Imaging

    [https://arxiv.org/abs/2611.06700](https://arxiv.org/abs/2611.06700)

    提出了一种物理信息驱动的自监督隐式神经表示框架PINR-ESI，利用导联场矩阵的物理约束和经典逆估计的暖启动初始化，在无需成对源标签或仿真预训练的情况下实现脑电扩展源成像的空间源重建。

    

    脑电（EEG）扩展源成像旨在定位活跃的皮层区域并刻画其空间范围，然而有监督的逆问题网络通常依赖于合成的脑电-源数据对，且可能对合成数据与真实数据之间的失配较为敏感。我们提出了一种物理信息自监督隐式神经表示（INR）框架，用于空间源重建，无需成对的源标签，也无需基于仿真的有监督预训练。导联场矩阵编码了从源到传感器的容积传导模型，并通过预测脑电与实测脑电之间的一致性来约束隐式神经表示。皮层电流密度被建模为一个连续的、以坐标为条件的场。经典的逆问题估计为源活动提供暖启动初始化和空间约束，同时基于块集成的投影将连续场映射为离散源系数以用于正向建模。在不同噪声水平、源数量和源范围条件下的仿真（摘要原文在此处截断）

    arXiv:2611.06700v1 Announce Type: new  Abstract: EEG extended-source imaging aims to localize active cortical regions and characterize their spatial extent, yet supervised inverse networks typically depend on synthetic EEG-source pairs and may be sensitive to synthetic-to-real mismatch. We propose a physics-informed self-supervised implicit neural representation (INR) framework for spatial source reconstruction without paired source labels or simulation-based supervised pretraining. The lead field matrix encodes the source-to-sensor volume-conduction model and constrains the INR through agreement between predicted and measured EEG. Cortical current density is modeled as a continuous coordinate-conditioned field. A classical inverse estimate provides warm initialization and spatial constraints on source activity, while patch-integrated projection maps the field to discrete source coefficients for forward modeling. Simulations with varying noise levels, source counts, and source extents 
    
[^34]: 不要替换，而要互补：一种面向近分布外检测的检测器无关子空间信号

    Don't Replace, Complement: A Detector-Agnostic Subspace Signal for Near-OOD Detection

    [https://arxiv.org/abs/2611.06688](https://arxiv.org/abs/2611.06688)

    该论文提出PLUS方法，利用稀疏自编码器为每个类别构建低维正交子空间并计算白化投影评分，作为与现有检测评分互补的近分布外检测信号，从而在不牺牲远分布外性能的前提下提升近分布外检测能力。

    

    事后分布外（OOD）检测在远分布外场景下已相当成熟，但在近分布外场景下仍然薄弱。这两种设定在类特征方向的激活上存在差异：远分布外输入大多无法激活其预测类别的类特征方向，而近分布外输入则会像分布内样本一样激活这些方向。由于用于区分远分布外的轴使得近分布外样本与分布内样本几乎无法区分，针对一种设定优化的评分与另一种设定并不匹配；在检测器已对远分布外达到饱和的情况下，新增的近分布外灵敏度在一定程度上是以牺牲远分布外性能为代价换来的。这种分歧促使我们设计一个专门针对近分布外的信号，以补充现有评分。我们提出了PLUS（Per-class Low-dimensional sUbspace Signal，每类低维子空间信号）。对于每个类别，PLUS从稀疏自编码器中选取m个最具类活性的神经元，将对应的解码器方向正交归一化以构成低维子空间，并通过每类白化投影对输入进行评分……（原文摘要在此处截断）

    arXiv:2611.06688v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection is mature on far-OOD yet remains weak on near-OOD. These two settings differ in their use of class-characteristic feature directions: far-OOD inputs largely fail to activate those of their predicted class, whereas near-OOD inputs activate them much as in-distribution samples do. Because the axis separating far-OOD leaves near-OOD almost indistinguishable from ID, a score optimized for one setting is misaligned with the other; where a detector already saturates far-OOD, added near-OOD sensitivity is bought partly against it. This divergence motivates a targeted near-OOD signal designed to complement existing scores. We propose PLUS (Per-class Low-dimensional sUbspace Signal). For each class, PLUS selects the $m$ most class-active neurons from a sparse autoencoder, orthonormalizes the corresponding decoder directions to form a low-dimensional subspace, and scores an input by its per-class whit
    
[^35]: SecureGARD：面向安全工具使用型LLM代理的动态本体引导治理框架

    SecureGARD: Dynamic Ontology-Guided Governance for Secure Tool-Using LLM Agents

    [https://arxiv.org/abs/2611.06682](https://arxiv.org/abs/2611.06682)

    提出SecureGARD框架，通过将可执行本体与回合级任务状态相结合，为工具使用型LLM代理生成确定性的就绪、等待和禁止决策，从而抵御间接提示注入攻击，同时在不扩大授权边界的前提下支持有界重规划。

    

    使用工具的大型语言模型（LLM）代理会将模型输出转化为具有实际影响的行动，从而使间接提示注入演变为一个执行安全问题。现有的防御方法通过约束工具调用、轨迹或信息流来进行防护，但往往将动作语义、状态依赖关系和证据关系保持为隐式状态，这限制了在异构工具之间的泛化能力，并使安全更新变得复杂。我们提出了SecureGARD，一个动态本体引导的治理框架。SecureGARD将异构调用锚定在共享的语义表示中，并将可执行本体与回合级任务状态相结合，以推导出确定性的就绪、等待和禁止三类决策。等待决策识别出未满足但可恢复的义务，从而实现有界重规划，而不会扩大代理的授权边界。SecureGARD从经过验证的训练执行中离线演化其本体。离线LLM提出与具体实例无关的……

    arXiv:2611.06682v1 Announce Type: new  Abstract: Tool-using large language model (LLM) agents translate model outputs into consequential actions, turning indirect prompt injection into an execution-security problem. Existing defenses constrain tool calls, trajectories, or information flows, but often leave action semantics, state dependencies, and evidence relations implicit, limiting generalization across heterogeneous tools and complicating safe updates. We present \ours{}, a dynamic ontology-guided governance framework. \ours{} grounds heterogeneous calls in a shared semantic representation and combines an executable ontology with episode-level task state to derive deterministic ready, waiting, and forbidden decisions. A waiting decision identifies an unmet but recoverable obligation, enabling bounded replanning without expanding the agent's authorization boundary. \ours{} evolves its ontology offline from verified training executions. An offline LLM proposes instance-independent ac
    
[^36]: 遗忘算法是否学到了正确的特征？

    Do unlearning algorithms learn the right features?

    [https://arxiv.org/abs/2611.06681](https://arxiv.org/abs/2611.06681)

    该论文证明标准的机器遗忘评估指标可能掩盖特征层面的本质差异，并提出一种区分缺失与多余特征的特征恢复度量，发现标准指标表现良好的遗忘模型往往无法恢复重训练模型的特征。

    

    机器遗忘通常通过标准的删除后评估指标来衡量，即将遗忘后的模型与在剩余数据上重新训练的模型进行比较。这种快照式评估可能掩盖两个模型所编码特征之间的实质性差异。我们证明，如果一个遗忘模型和一个重训练模型在输出上一致但编码了不同的特征，那么只需对其预测头进行单步梯度更新就可能破坏这种一致性。我们进一步表明，删除任意小比例的数据都可能导致重训练选择性质截然不同的特征，因此要匹配重训练结果，可能需要学习完整模型从未拥有过的特征。随后，我们引入了一种特征恢复度量，将缺失特征与多余特征区分开来。在八种遗忘方法的实验中，在标准指标上表现良好的模型往往无法恢复重训练模型的特征。这种不匹配能够预测后续重新学习过程中模型与重训练模型的偏差。

    arXiv:2611.06681v1 Announce Type: new  Abstract: Machine unlearning is commonly evaluated with standard post-deletion metrics that compare an unlearned model to one retrained on the remaining data. Such snapshot evaluations can conceal consequential differences in the features the two models encode. We prove that if an unlearned and a retrained model agree on their outputs but encode different features, a single gradient step on their prediction heads can break this agreement. We further show that deleting an arbitrarily small fraction of the data can cause retraining to select qualitatively different features, so matching retraining can require learning features that the full model never had. We then introduce a measure of feature recovery that separates missing from extraneous features. Across eight unlearning methods, models that perform well on standard metrics often fail to recover the retrained features. This mismatch predicts deviations from retraining under subsequent relearnin
    
[^37]: 自导向苹果品尝

    Self-Directed Apple Tasting

    [https://arxiv.org/abs/2611.06677](https://arxiv.org/abs/2611.06677)

    本文在自导向苹果品尝问题中，通过 VC 维、$\emptyset$-中心星数和自导向维数等组合参数，建立了可实现设置下最小最大错误率的四分法刻画，以及不可知设置下遗憾率的二分法刻画。

    

    我们研究经典的苹果品尝设置，这是一种部分反馈模型，其中学习者只有在其预测为 $1$ 时才能观察到真实标签。Raman 等人（2024）已经研究了该问题在在线对抗设置下的错误率组合刻画。我们研究自导向设置，即学习者可以自适应地选择其进行预测的点的顺序。在可实现设置中，我们针对任何学习者的期望最小最大错误次数建立了四分法刻画：根据 VC 维、$\emptyset$-中心星数和自导向维数，错误率可能为 $\Theta(1)$、$O(\log n) \cap \omega(1)$、$\tilde \Theta(\sqrt{n})$ 或 $\Theta(n)$。在不可知设置中，最小最大遗憾为 $\tilde \Theta(\sqrt{n})$ 或 $\Theta(n)$，具体取决于 VC 维。

    arXiv:2611.06677v1 Announce Type: new  Abstract: We study the classical apple-tasting setting, a partial feedback model in which the learner observes the true label only when it predicts $1$. A combinatorial characterization for the rates of this problem in the online adversarial setting has been studied by Raman et al. (2024). We study the self-directed setting, in which the learner can adaptively choose the order of the points it predicts on. In the realizable setting, we establish a tetrachotomy on the expected minimax number of mistakes made by any learner. Depending on the VC dimension, the $\emptyset$-centered star number, and the self-directed dimension, we show that the rates could possibly be $\Theta(1)$, $O(\log n) \cap \omega(1)$, $\tilde \Theta(\sqrt{n})$, or $\Theta(n)$. In the agnostic setting, the minimax regret is $\tilde \Theta(\sqrt{n})$ or $\Theta(n)$, depending on the VC dimension.
    
[^38]: 指纹识别分布式模型提取背后的调度器

    Fingerprinting the Dispatchers behind Distributed Model Extraction

    [https://arxiv.org/abs/2611.06661](https://arxiv.org/abs/2611.06661)

    该论文提出通过请求时间戳（到达时间和流完成时间）指纹识别分布式模型提取攻击背后的调度策略，仅用20%种子集即可在20分钟内以0.92至1.00的AUC对攻击活动中剩余账户进行取证式排序，即使内容分类器失效时也能有效检测此类攻击。

    

    API蒸馏活动试图通过将请求分散到数千个负载均衡的账户中来隐藏模型窃取流量，而标准内容分类器往往无法检测到这种行为。然而，请求时间戳，特别是到达时间和流完成时间，会暴露底层的调度策略，例如轮询计数器或密钥混洗池。使用20%的种子集，这种时序分析可以在20分钟内以0.92至1.00的AUC对活动中剩余的账户进行取证式排序，即使面对数千个真实生产账户也是如此。虽然自适应攻击者难以通过抖动来隐藏轮换调度器，但固定密钥能够成功规避检测，尽管在配额受限时会严重限制吞吐量。最终，即使在内容过滤器失效的情况下，时序分析也能暴露共享基础设施模式。

    arXiv:2611.06661v1 Announce Type: new  Abstract: API distillation campaigns try to hide model-stealing traffic by splitting requests across thousands of load-balanced accounts, which standard content classifiers often miss. However, request timestamps, specifically arrival time and stream completion time, reveal the underlying dispatching strategy, such as round-robin counters or key-shuffling pools. Using a 20 percent seed set, this timing analysis can forensically rank the remaining accounts of a campaign with 0.92 to 1.00 AUC within 20 minutes, even across thousands of real production accounts. While adaptive attackers struggle to hide rotating dispatchers using jitter, pinning keys successfully evades detection, though it can severely throttle throughput when quotas bind. Ultimately, timing exposes shared infrastructure patterns even when content filters fail.
    
[^39]: 使用校准的关系世界模型检测静默的动作接口失配

    Detecting Silent Action-Interface Mismatch with Calibrated Relational World Models

    [https://arxiv.org/abs/2611.06653](https://arxiv.org/abs/2611.06653)

    该论文提出将学习到的状态-动作-效果关系作为离线契约，利用校准的联合高斯混合评分及其条件分布来检测机器人动作接口中的静默语义失配，在LIBERO任务上达到65.7%的召回率和95.8%的失配字段定位准确率。

    

    机器人的动作即使满足形状和范围约束，仍可能携带错误的语义：坐标轴可能被置换、符号可能被反转、旋转可能被重新缩放，或夹爪极性可能被颠倒。我们研究学习到的状态-动作-效果关系，将其作为离线契约来检测这些失配并对其可能涉及的字段进行排序。干净的保留转换数据用于校准警报；固定的反向词汇表支持闭集定位。在包含432个源片段的十个LIBERO Spatial任务上，在2.5%经验干净假阳性率下进行的容量受控比较显示：岭回归前向残差的召回率为46.1%，四分量联合高斯混合模型为61.0%，而同一混合模型的精确条件分布达到65.7%。联合评分比岭回归保持14.9个百分点的优势，而条件化使召回率进一步提高4.7个百分点。联合评分则在定位方面表现更高：95.8%对94.6%。

    arXiv:2611.06653v1 Announce Type: cross  Abstract: A robot action can satisfy shape and range constraints yet carry the wrong semantics: axes may be permuted, signs inverted, rotations rescaled, or gripper polarity reversed. We study learned state-action-effect relations as offline contracts for detecting these mismatches and ranking their likely fields. Clean held-out transitions calibrate alarms; a fixed inverse vocabulary supports closed-set localization. On ten LIBERO Spatial tasks comprising 432 source episodes, a capacity-controlled comparison at 2.5% empirical clean false-positive rate gives 46.1% recall for a ridge forward residual, 61.0% for a four-component joint Gaussian mixture, and 65.7% for the exact conditional of that same mixture. The joint score retains a 14.9 percentage-point advantage over ridge, while conditioning improves recall by a further 4.7 points. Joint scoring instead has higher localization: 95.8% versus 94.6%. These matched-rate comparisons are retrospect
    
[^40]: TestBED：一个用于贝叶斯实验设计的开源工具包

    TestBED: An Open-Source Toolkit for Bayesian Experimental Design

    [https://arxiv.org/abs/2611.06633](https://arxiv.org/abs/2611.06633)

    本文提出了 TestBED，一个开源的贝叶斯实验设计通用工具包，通过将 BED 流程分解为任务、目标、训练机制、策略架构和评估流程等模块化、可自由组合的组件，解决了该领域缺乏标准化软件和基准测试的问题。

    

    贝叶斯实验设计（BED）是一个用于智能数据获取的原则性框架，广泛应用于科学和工业领域的众多场景。然而，由于缺乏标准化的软件和基准测试，BED 的发展和采用仍受到阻碍。现有方法通常在定制的代码库中实现，并且仅在一小部分任务上进行评估，这使得各种方法难以被应用、比较或扩展。为了解决这一问题，我们提出了 TestBED，一个通用的、开源的贝叶斯实验设计工具包。TestBED 将 BED 流程分解为模块化、可复用的组件，涵盖任务、目标、训练机制、策略架构和评估流程，并提供了核心原语（如期望信息增益估计器）的可扩展实现。我们提供了种类丰富的组件，这些组件可以自由组合，从而使现有方法能够在……（摘要在此处截断）

    arXiv:2611.06633v1 Announce Type: cross  Abstract: Bayesian experimental design (BED) is a principled framework for intelligent data acquisition with applications across a wide range of scientific and industrial domains. However, the development and adoption of BED remains hindered by a lack of standardised software and benchmarks. Existing methods are typically implemented in bespoke codebases and evaluated on a small subset of tasks, making methods difficult to apply, compare, or extend. To address this, we present TestBED, a general-purpose, open-source toolkit for Bayesian experimental design. TestBED decomposes the BED process into modular, reusable components spanning tasks, objectives, training mechanisms, policy architectures, and evaluation procedures, together with scalable implementations of core primitives such as expected information gain estimators. We provide a wide range of these components, which can be freely composed, allowing existing methods to be reproduced within
    
[^41]: 有限测量预算下量子电路玻恩机器的噪声正则化训练

    Noise-regularized training of quantum circuit Born machines under a finite shot budget

    [https://arxiv.org/abs/2611.06630](https://arxiv.org/abs/2611.06630)

    该论文证明硬件噪声可作为量子电路玻恩机器的正则化手段以提升泛化能力，并提出GENIE框架在有限测量预算内自动选择最优噪声水平。

    

    量子电路玻恩机器能够学习经典计算机难以采样的分布，但其价值取决于能否从有限的训练数据中进行泛化。本文研究了硬件噪声如何影响这种泛化能力。我们证明，对于任何噪声过程，使用最大均值差异训练的QCBM的泛化差距受该噪声模型所能达到的分布复杂度限制。扰动的局域泡利噪声会收缩这种复杂度，但会使模型偏向均匀分布，因此残余噪声起到正则化的作用，其最优强度可以是中等水平。我们提出了GENIE框架，该框架通过有上限的误差缓解实现所选定的噪声水平，并在固定的执行预算内于训练过程中从留出数据中选择噪声水平，且具有可证明的有界遗憾。在合成数据和人口普查数据的模拟中，GENIE的表现始终接近四种噪声控制策略中的最优者，而任何固定的噪声选择都可能是次优的。

    arXiv:2611.06630v1 Announce Type: cross  Abstract: Quantum circuit Born machines (QCBMs) can learn distributions that are hard to sample from classically, but their value depends on generalizing from finite training data. Here, we study how hardware noise affects this generalization. We show that, for any noise process, the generalization gap of a QCBM trained with the maximum mean discrepancy is bounded by the complexity of the distributions the noisy model can reach. Scrambled local Pauli noise contracts this complexity but biases the model towards the uniform distribution, so residual noise acts as a regularizer whose optimal strength can be intermediate. We propose GENIE, a framework that realizes a chosen noise level with capped error mitigation and selects it during training from held-out data within a fixed execution budget, with provably bounded regret. In simulations on synthetic and census data, GENIE stays close to the best of four noise-control strategies, whereas every fix
    
[^42]: 基于 Cosmos 的面向物理 AI 的交互式世界仿真

    Interactive World Simulation with Cosmos for Physical AI

    [https://arxiv.org/abs/2611.06627](https://arxiv.org/abs/2611.06627)

    该论文提出了一个将 Cosmos 世界基础模型适配为实时交互式世界仿真的共享框架，使用户和具身智能体能够通过动作与场景控制同演化的虚拟环境交互，并支持相机导航、机器人操作、可控视频迁移和多视角自动驾驶仿真等应用。

    

    我们提出了一个共享框架，将 Cosmos 世界基础模型适配用于物理 AI 中的实时交互式世界仿真。通过根据动作和场景控制流式生成视觉观察结果，该框架使用户和具身智能体能够与不断演化的虚拟环境进行交互。它既支持预测动作的视觉后果，也支持根据外部模拟器维护的物理状态生成观察结果。我们在相机导航与机器人操作、可控视频迁移以及多视角自动驾驶汽车（AV）仿真等场景中演示了这一方法。这些示例为开发用于物理 AI 训练与评估的交互式虚拟环境提供了共享基础。我们的代码和模型检查点可在 https://github.com/nvidia/cosmos 和 https://huggingface.co/collections/nvidia/cosmos3-examples 获取。项目网站可在 https://research.nvidia.com 访问。

    arXiv:2611.06627v1 Announce Type: new  Abstract: We present a shared framework that adapts Cosmos world foundation models for real-time interactive world simulation in Physical AI. By streaming visual observations in response to actions and scene controls, the framework enables users and embodied agents to interact with evolving virtual environments. It supports both predicting the visual consequences of actions and generating observations from physical state maintained by an external simulator. We demonstrate this approach across camera navigation and robot manipulation, controllable video transfer, and multi-view autonomous vehicle (AV) simulation. These examples provide a shared foundation for developing interactive virtual environments for Physical AI training and evaluation. Our code and model checkpoints are available at https://github.com/nvidia/cosmos and https://huggingface.co/collections/nvidia/cosmos3-examples. The project website is available at https://research.nvidia.com/
    
[^43]: 没有协方差信息就无法实现与维度无关的私有均值估计

    Dimension-Free Private Mean Estimation is Impossible Without the Covariance

    [https://arxiv.org/abs/2611.06615](https://arxiv.org/abs/2611.06615)

    该论文通过建立新的下界证明：在协方差矩阵未知的情况下，差分隐私高斯均值估计无法实现与维度无关的最优误差保证——当样本量低于有效维度时，误差会比协方差已知情形大 √(d/n) 倍，即使样本量更高时差距依然存在。

    

    我们研究了当协方差矩阵未知时，差分隐私高斯均值估计的极限。先前的工作表明，当协方差 Σ 已知时，最优样本复杂度可以仅依赖于其谱，而非环境维度 d。我们针对采用欧氏误差的高斯均值估计建立了新的下界，证明当协方差未知时，私有估计器无法获得类似的保证。首先，当样本量 n 小于有效维度 trace(Σ)/||Σ||_op 时，协方差未知时的最优误差可能比协方差已知时的最优误差大 √(d/n) 倍。其次，当样本量大于有效维度时，即使对于有效维度不超过 log d 的对角协方差矩阵，协方差未知时的最优误差也可能大于协方差已知时的最优误差。

    arXiv:2611.06615v1 Announce Type: new  Abstract: We study the limits of differentially private Gaussian mean estimation when the covariance matrix is unknown. Prior work shows that when the covariance $\Sigma$ is known, the optimal sample complexity can depend only on its spectrum, rather than on the ambient dimension $d$. We establish new lower bounds for Gaussian mean estimation with Euclidean error which show that private estimators cannot achieve similar guarantees when the covariance is unknown. First, when the sample size $n$ is smaller than the effective dimension $\mathrm{trace}(\Sigma)/\|\Sigma\|_{\text{op}}$, the optimal error when the covariance is unknown can be larger than the optimal error when it is known by a factor of $\sqrt{d/n}$. Second, when the sample size is larger than the effective dimension, even for diagonal covariance matrices with effective dimension bounded by $\log d$, the optimal error when the covariance is unknown can be larger than the optimal error wh
    
[^44]: 从重建到插补：针对时间序列缺失通道的测试时优化

    From Reconstruction to Imputation: Test-Time Optimization for Missing Time-Series Channels

    [https://arxiv.org/abs/2611.06612](https://arxiv.org/abs/2611.06612)

    该论文提出一种测试时优化方法，将冻结的序列重建器在推理时转化为插补器——把缺失条目作为优化变量、仅最小化观测坐标上的重建误差，从而无需人工掩码增强或全通道丢弃训练即可应对完整传感器通道失效导致的时间序列缺失。

    

    完整的传感器通道失效会造成一种困难的时间序列缺失形式：局部插值方法不可用，而为每种失效模式单独部署一个训练好的插补器也可能不切实际。我们探讨这样一个问题：一个未经人工掩码增强训练、随后被冻结的序列重建器，能否在推理阶段被转换为插补器。我们的方法将每个测试窗口中的缺失条目视为优化变量，固定所有观测值，并通过冻结的模型进行反向传播，仅在观测坐标上最小化重建误差。由此得到的“观测自洽性”目标既不需要重建器训练时的人工掩码增强，也不需要全通道丢弃。我们通过从缺失输入到观测残差的交叉雅可比矩阵刻画了这种精调何时能够提供有效信息，并在该耦合较弱时引入可选的时间平滑先验。在五个真实世界……（摘要截断）

    arXiv:2611.06612v1 Announce Type: cross  Abstract: Complete sensor-channel failures create a difficult form of time-series missingness: local interpolation is unavailable, and deploying a separately trained imputer for every failure pattern may be impractical. We ask whether a sequence reconstructor trained without artificial mask augmentation and then frozen can be converted into an imputer at inference time. Our method treats the missing entries of each test window as optimization variables, clamps all observations, and backpropagates through the frozen model to minimize reconstruction error only on observed coordinates. The resulting observed-self-consistency objective requires neither artificial mask augmentation nor full-channel dropout during reconstructor training. We characterize when this refinement is informative through the cross-Jacobian from missing inputs to observed residuals, and use an optional temporal-smoothness prior when that coupling is weak. Across five real-worl
    
[^45]: 协同过滤中浅层自编码器的替代方案

    Alternatives to Shallow Autoencoders for Collaborative Filtering

    [https://arxiv.org/abs/2611.06610](https://arxiv.org/abs/2611.06610)

    该论文针对EASE模型仅依赖物品-物品关系而限制细粒度用户关系捕捉的问题，提出了利用用户-用户信息的替代模型，以及结合用户-用户和物品-物品邻接分布的混合模型，从而提升协同过滤的推荐准确性。

    

    协同过滤（CF）是推荐系统的基石，在许多现代应用中发挥着重要作用。CF利用用户-物品交互数据来发现用户的未来偏好。尽管深度学习模型在协同过滤中展现出了潜力，但面向稀疏数据的极简浅层自编码器（EASE）因其闭式解所带来的出色排序准确性而备受关注。EASE主要依靠物品之间的关系来拟合一个满秩的高维线性映射。我们假设这种设计限制了其捕捉相似但不完全等价的细粒度用户关系的能力，从而限制了其推荐准确性。本文提出了一种基于EASE的替代公式，该公式利用了用户-用户之间的信息。此外，我们提出了一个同时结合用户-用户和物品-物品邻接分布的混合模型。我们的实验表明，所提出的模型表现更优。

    arXiv:2611.06610v1 Announce Type: new  Abstract: Collaborative filtering (CF) is a cornerstone of recommender systems and plays a relevant role in many modern applications. CF uses user-item interaction data to discover future preferences. Although deep learning models have shown promise in CF, Embarrassingly Shallow Autoencoders for Sparse Data (EASE) has gained attention for its outstanding ranking accuracy provided by its closed-form solution. EASE relies primarily on relationships among items to fit a full-rank high-dimensional linear mapping. We hypothesize that this design limits its capacity to capture similar but not equivalent fine-grained user relationships, consequently limiting its recommendation accuracy. This paper introduces an alternative formulation based on EASE that takes advantage of user-user information. Furthermore, we propose a hybrid model that combines both user-user and item-item adjacency distributions.   Our experiments reveal that the proposed models outpe
    
[^46]: 验证器引导的自一致性：基于不完美生成器与验证器的测试时推理

    Verifier-Guided Self-Consistency: Test-Time Reasoning with Imperfect Generators and Verifiers

    [https://arxiv.org/abs/2611.06606](https://arxiv.org/abs/2611.06606)

    该论文提出验证器引导的自一致性（VGSC）随机解码框架，在生成过程中同时利用步骤级正确性与未来结果估计两种信号，在不完美的生成器多样性与验证器正确性引导之间取得平衡，从而改进测试时推理。

    

    大语言模型能够解决许多多步推理问题，但其推理仍然脆弱：看似合理的中间步骤往往导致错误的答案。在测试时改进推理需要平衡两个不完美的信号：提供多样性的生成器，以及引导正确性的验证器。现有方法大多偏向这一权衡的某一侧。基于验证器加权的投票方法依赖于从生成器采样的轨迹，无法将生成过程引导向正确的方向；相比之下，束搜索式的验证器引导搜索虽然可以引导生成，但可能过度集中于狭窄的轨迹集合，并放大验证器的错误。我们提出了验证器引导的自一致性，这是一种随机解码框架，它结合步骤级正确性与未来结果估计来在生成过程中引导推理。VGSC反复从基础语言模型中提出候选的下一步，并利用这两种信号在……（原文摘要在此处截断）

    arXiv:2611.06606v1 Announce Type: new  Abstract: Large language models can solve many multi-step reasoning problems, but their reasoning remains brittle: plausible intermediate steps often lead to incorrect answers. Improving reasoning at test time requires balancing two imperfect signals: the generator, which provides diversity, and the verifier, which guides correctness. Existing approaches largely favor one side of this tradeoff. Verifier-weighted voting relies on trajectories sampled from the generator and cannot steer generation toward correct ones. By contrast, beam-style verifier-guided search can steer generation, but may over-concentrate on narrow trajectory sets and amplify verifier errors.   We propose Verifier-Guided Self-Consistency (VGSC), a stochastic decoding framework that combines step-level correctness and future-outcome estimates to guide reasoning during generation. VGSC repeatedly proposes candidate next steps from the base language model and uses both signals in 
    
[^47]: 微调的行为速率：将压缩作为有用信息的度量

    The Behavioral Rate of Fine-Tuning: Compression as a Measure of Useful Information

    [https://arxiv.org/abs/2611.06598](https://arxiv.org/abs/2611.06598)

    本文提出“行为速率 $R^\star$”这一新指标——以能保留固定比例微调增益的压缩适配器最小序列化大小来度量微调写入的有用信息，并发现该速率由训练内容与冻结基础模型共同决定，而非参数量、微调增益或重建误差等直观指标。

    

    低秩适应通过对冻结的语言模型施加一个紧凑的学习更新来对其进行适配，这为探究“微调需要写入多少信息”提供了一个自然的研究场景。然而，适配器的秩、参数量或数值重建误差并不能直接告诉我们需要多少比特才能保留所获得的行为。因此，我们定义了一个可操作的行为速率 $R^\star$：即压缩后的适配器在重新加载回冻结基础模型后，仍能保留其留出集微调增益的固定比例时所需的最小序列化大小。在LoRA微调实验中，该速率在固定优化预算下随训练内容的不同而增加，但并不由源描述长度、微调增益或权重空间重建误差等直观指标所决定。最重要的是，相同的数据集在不同的冻结基础模型上表现出显著不同、甚至重新排序的速率，这表明微调所写入的信息依赖于其作用的基础模型。

    arXiv:2611.06598v1 Announce Type: new  Abstract: Low-Rank Adaptation (LoRA) adapts a frozen language model through a compact learned update, making it a natural setting for asking how much information fine-tuning needs to write. Yet adapter rank, parameter count, or numerical reconstruction error do not directly tell us how many bits are required to preserve the acquired behavior. We therefore define an operational behavioral rate $R^\star$ as the smallest serialized size of a compressed adapter that, once reloaded into the frozen base model, retains a fixed fraction of its held-out fine-tuning gain. Across LoRA fine-tuning experiments, this rate increases with distinct training content at fixed optimization budgets, but is not determined by intuitive metrics like source description length, fine-tuning gain, or weight-space reconstruction error. Most importantly, the same datasets exhibit substantially different, and even reordered, rates across frozen base models, showing that fine-tu
    
[^48]: 面向快速台基稳定性代理模型的信任层：校准误差棒、分布外告警与少样本重校准

    A trust layer for fast pedestal-stability surrogates: calibrated error bars, out-of-distribution alarms and few-shot recalibration

    [https://arxiv.org/abs/2611.06597](https://arxiv.org/abs/2611.06597)

    提出了一种与模型无关的核聚变台基稳定性代理模型信任层，通过共形误差棒、分布外告警和少样本重校准，即使在等离子体形状偏移导致查询超出训练数据时也能提供可信的不确定性量化。

    

    线性剥离-气球模稳定性的机器学习代理模型如今可在毫秒级运行，但它们大多以点估计器的形式被部署：既不说明自身可能有多大误差，也不告知查询是否超出了训练数据的范围。对于一家正在设计新装置的核聚变公司而言，所有感兴趣的平衡态都处于现有代理模型的训练数据之外，因此这才是真正重要的场景。我们提出了一种与模型无关的稳定性代理模型信任层，由共形误差棒、将可疑案例路由至物理代码的分布外告警，以及基于少量新设计模拟进行重校准三部分组成，并通过预注册假设对其进行了检验。在一个包含JET类平衡态及理想MHD增长率的大型公开数据集上，误差棒在分布内保持了名义覆盖率，但在等离子体形状发生偏移时覆盖率有所下降；通过对新形状进行几十次模拟，或将告警标记的案例路由至物理代码，均可恢复该覆盖率。

    arXiv:2611.06597v1 Announce Type: new  Abstract: Machine-learning surrogates of linear peeling-ballooning stability now run in milliseconds, but they are mostly deployed as point estimators: they do not say how wrong they may be, or when a query lies outside what they were trained on. For a fusion company designing a new device, every equilibrium of interest is outside the training data of existing surrogates, so this is the case that matters. We describe a model-agnostic trust layer for stability surrogates, made of conformal error bars, out-of-distribution alarms that route doubtful cases to the physics code, and recalibration from a few simulations of a new design, and we test it with pre-registered hypotheses. On a large public dataset of JET-like equilibria with ideal-MHD growth rates, the error bars hold their nominal coverage in distribution and fall short under a shift in plasma shape; a few dozen simulations of the new shape, or routing the cases the alarms flag, restore it. O
    
[^49]: 热力学信息引导的学习捕捉低共熔溶剂中的分层氢键网络

    Thermodynamics-Informed Learning Captures Hierarchical Hydrogen-Bond Networks in Deep Eutectic Solvents

    [https://arxiv.org/abs/2611.06590](https://arxiv.org/abs/2611.06590)

    该研究提出热力学信息引导的神经网络TiNN，通过学习自由能路径分层刻画低共熔溶剂中氢键受体-供体缔合与溶质容纳过程，将木质纤维素溶解度预测误差从COSMO-RS的23倍以上降至约1.5倍。

    

    低共熔溶剂为木质纤维素精炼提供了一个可调节的平台，但其溶剂化行为取决于多组分氢键网络，而非各组分的单独性质。我们开发了TiNN，一种热力学信息引导的神经网络，通过学习到的自由能路径分层地表征氢键受体-氢键供体的缔合以及溶质的容纳过程。TiNN在938条记录上训练后，测试集R²达到0.917，并能泛化到新的氢键受体-供体组合以及数据稀疏的条件。消融实验表明，溶质特异性稳定化是对预测准确性贡献最大的因素，而氢键相容性的重要性则取决于具体的溶解机制。在酸性、碱性和中性低共熔溶剂中对葡萄糖、木聚糖和硫酸盐木质素的实验测量表明，TiNN将几何平均溶解度误差从COSMO-RS的超过23倍降低至约1.5倍。在无需对分数溶解（原文在此处截断）……

    arXiv:2611.06590v1 Announce Type: new  Abstract: Deep eutectic solvents (DESs) offer a tunable platform for lignocellulose refining, but their solvation behavior depends on multicomponent hydrogen-bond networks rather than individual constituent properties. We develop TiNN, a thermodynamics-informed neural network that hierarchically represents HBA-HBD association and solute accommodation through a learned free-energy pathway. Trained on 938 records, TiNN achieves a test-set $R^2$ of 0.917 and generalizes to new HBA-HBD combinations and sparse conditions. Ablation identifies solute-specific stabilization as the largest contributor to prediction accuracy, while the importance of hydrogen-bond compatibility depends on the dissolution mechanism. Experimental measurements of glucose, xylan, and Kraft lignin in acidic, basic, and neutral DESs show that TiNN reduces the geometric mean solubility error from more than 23-fold with COSMO-RS to approximately 1.5-fold. Without training on fractio
    
[^50]: 深度神经网络中混沌边缘的有限宽度偏移

    Finite-width shift of the edge of chaos in deep neural networks

    [https://arxiv.org/abs/2611.06582](https://arxiv.org/abs/2611.06582)

    该论文将两输入距离的对数建模为随机游走，精确计算出有限宽度下深度神经网络的混沌边缘向更大权重方差方向偏移（tanh网络约为8/N），并解释了典型网络与网络平均的临界点移动方向相反这一长期争议。

    

    arXiv:2611.06582v1 公告类型：cross 摘要：深度神经网络在“混沌边缘”初始化时训练效果最佳——混沌边缘是权重方差与偏置方差平面上的一条临界线，在这条线上两个输入之间的距离既不随深度增长也不随深度收缩。这条临界线在无限宽网络中是已知的；在有限宽度下它会移动，而它如何移动一直存在争议。我们将两个输入经过宽度为 $N$ 的网络后距离的对数视为随机游走，并计算其漂移和方差至 $1/N$ 阶。混沌边缘移动到更大的权重方差处，移动量为 $\Delta\sigma_w^2=-a_1/(N\,\partial_{\sigma_w^2}\ln\chi_1)$，对于偏置方差为 $0.05$ 的 $\tanh$ 网络约为 $8/N$，这已通过对精确有限宽度递推关系的模拟得到证实。该偏移描述的是典型网络；而对多个网络取平均时，结果由少数距离增长的稀有网络主导，其临界点向相反方向移动，这解释了为什么理论计算与模拟（结果存在分歧）……（摘要在此处截断）

    arXiv:2611.06582v1 Announce Type: cross  Abstract: Deep neural networks train best when they are initialised at the edge of chaos, the critical line in the plane of weight and bias variances on which the distance between two inputs neither grows nor shrinks with depth. The line is known for infinitely wide networks; at finite width it moves, and how it moves has been disputed. We treat the logarithm of the distance between two inputs through a network of width $N$ as a random walk and compute its drift and variance to order $1/N$. The edge of chaos moves to a larger weight variance, by $\Delta\sigma_w^2=-a_1/(N\,\partial_{\sigma_w^2}\ln\chi_1)$, about $8/N$ for $\tanh$ networks at a bias variance of $0.05$, confirmed by simulations of the exact finite-width recursion. The shift describes a typical network; the average over many networks is dominated by rare networks in which the distance grows, and its critical point moves the other way, which explains why calculations and simulations 
    
[^51]: 从固定默认配置到经过校准的激活策略：面向长期时间序列预测

    From Fixed Defaults to Calibrated Activation Policies for Long-Term Time-Series Forecasting

    [https://arxiv.org/abs/2611.06559](https://arxiv.org/abs/2611.06559)

    该研究提出一种经过校准的激活选择策略，能够根据任务自适应地选择激活函数及其位置，并在证据不足时保留默认配置，在1,225次Informer训练实验中平均将长期时间序列预测的均方误差降低7.86%。

    

    长期时间序列预测为能源、交通等应用的决策提供支持，这些应用需要在观测数据范围之外获得可靠的预测结果。尽管预测研究在模型架构方面取得了进展，但固定的激活配置对于激活选择应如何随数据集和预测时域变化所提供的指导十分有限。本研究旨在确定任务特定的激活选择何时能够提升预测精度，以及如何利用先前的实验结果来指导为新任务选择合适的配置。我们开发了一种经过校准的选择策略，该策略从先前实验中学习一个可靠的默认配置，针对每个任务自适应地调整激活函数及其位置，并在证据不足时保留默认配置。在一项由1,225次Informer训练运行支持的评估中，涵盖17个数据集变体和51个预测任务，该策略平均将均方误差降低了7.86%。

    arXiv:2611.06559v1 Announce Type: new  Abstract: Long-term time-series forecasting supports decisions in energy, transportation, and other applications where reliable predictions are needed beyond the observed data. Although forecasting research has advanced model architectures, fixed activation configurations provide limited guidance on how activation choices should vary across datasets and prediction horizons. This study aims to establish when task-specific activation choices improve forecasting accuracy and how previous experiments can guide the selection of suitable configurations for new tasks. We develop a calibrated selection policy that learns a reliable default from prior experiments, adapts the activation function and its location to each task, and retains the default when evidence for a change is insufficient. In an evaluation supported by 1,225 Informer training runs across 17 dataset variants and 51 forecasting tasks, the policy reduces mean squared error by 7.86% on avera
    
[^52]: RAGAM-AI：一种由生成式AI驱动的混合强化学习智能体方法，用于边缘资源管理

    RAGAM-AI: A Hybrid Reinforcement Learning based Agentic approach powered by Generative AI for Edge Resource Management

    [https://arxiv.org/abs/2611.06554](https://arxiv.org/abs/2611.06554)

    提出RAGAM-AI混合框架，将强化学习与生成式AI结合构建智能体工作流，利用LLM作为奖励预言机解决奖励泛化问题，以优化多接入边缘计算单元的资源分配。

    

    对于托管多种服务的分布式边缘网络，强化学习（RL）已被高效地用于资源管理。然而，复杂工作流中的奖励泛化问题仍是RL领域的一个开放性难题。为此，本工作提出了RAGAM-AI，这是一个融合强化学习算法与生成式AI（Gen-AI）能力的混合框架，用于驱动一个智能体AI工作流，以优化多接入边缘计算（MEC）单元的资源分配。该框架通过利用大语言模型（LLM）提供反馈，进一步解决了奖励泛化问题。该工作实现了以MathΣtral作为奖励预言机的多智能体Soft-Actor-Critic（MASAC）算法来管理MEC处的资源分配。RAGAM-AI能够解读系统状态信息并评估多目标权衡，从而指导分布式边缘服务器智能体上的资源编排循环。该框架的复杂性被进一步……（原文摘要此处截断）

    arXiv:2611.06554v1 Announce Type: new  Abstract: Resource management for distributed edge networks hosting different services has been handled efficiently using Reinforcement learning (RL). Reward generalization in complex workflows is still an open issue within the RL realm. Therefore, this work proposes RAGAM-AI, a hybrid framework that combines the capabilities of RL algorithms and Generative-AI (Gen-AI) for powering an Agentic AI workflow that optimizes resource allocation at the Multi-access Edge Computing (MEC) units. This framework further resolves the reward generalization issue by utilizing Large Language Model (LLM) for feedback. A Multi- Agent Soft-Actor Critic (MASAC) using Math{\Sigma}tral as a reward oracle is implemented for managing resource allocations at the MEC. RAGAM-AI interprets system state information and evaluates multi-objective trade-offs guiding the resource orchestration loop over distributed edge server agents. The complexity of the framework is further br
    
[^53]: 将睡眠基础模型适配于密集睡眠纺锤波定位

    Adapting Sleep Foundation Models for Dense Sleep Spindle Localization

    [https://arxiv.org/abs/2611.06546](https://arxiv.org/abs/2611.06546)

    该论文提出通过为冻结的SleepFM编码器添加局部时频通路与残差交叉注意力机制，成功实现密集睡眠纺锤波定位，在MODA数据集上取得了0.808的事件级F1分数。

    

    将睡眠基础模型适配于密集脑电图（EEG）事件定位具有挑战性，因为多秒级表示可能无法保留足够的时序细节以实现精确的边界恢复。我们通过适配一个冻结的SleepFM编码器并引入局部时频通路来解决睡眠纺锤波定位问题，该通路保留了与纺锤波相关的频谱内容和令牌内部位置信息。局部表示驱动定位任务，而残差交叉注意力则融合SleepFM的时序上下文。密集预测通过局部化的令牌到样本重建和时间细化获得。在海量在线数据标注（MODA）数据集上的测试表明，所提出的模型在六折交叉验证中，IoU为0.2时达到了0.8080 ± 0.0040的事件级F1分数。消融实验显示，定位主要依赖于局部频谱和子令牌时序信息，而预训练表示则提供了补充性上下文。

    arXiv:2611.06546v1 Announce Type: cross  Abstract: Adapting sleep foundation models to dense electroencephalography (EEG) event localization is challenging because multi-second representations may not preserve sufficient temporal detail for precise boundary recovery. We address sleep spindle localization by adapting a frozen SleepFM encoder with a local time-frequency pathway that preserves spindle-relevant spectral content and intra-token position. The local representation drives localization, while residual cross-attention incorporates SleepFM temporal context. Dense predictions are obtained through localized token-to-sample reconstruction and temporal refinement. Tested on the Massive Online Data Annotation (MODA), the proposed model achieves an event-level $F_1$ of $0.8080 \pm 0.0040$ at IoU 0.2 across six folds. Ablations show that localization primarily depends on local spectral and sub-token temporal information, while the pretrained representation provides complementary context
    
[^54]: EvoOPD：面向黑盒在线策略蒸馏的特权信息策略演化

    EvoOPD: Evolving Privileged Information Strategies for Black-Box On-Policy Distillation

    [https://arxiv.org/abs/2611.06542](https://arxiv.org/abs/2611.06542)

    提出EvoOPD框架，通过演化式地确定应从黑盒教师获取何种特权信息，并将其转化为学生自身轨迹上的密集监督信号，从而有效实现黑盒模型的在线策略蒸馏。

    

    专有语言模型展现出强大的推理与编码能力，但其输出分布不可访问，这使得依赖教师logits的细粒度蒸馏方法难以直接应用于黑盒模型。我们观察到，来自黑盒教师的丰富文本反馈可以被视作特权信息（Privileged Information, PI），并通过一个以PI为条件的特权教师，将其转化为针对学生自身轨迹的密集监督信号。然而，应从教师处获取何种PI才能最有效地支持学生学习仍不明确。不同PI策略在关注点、所请求信息的类型以及所提供细节的详尽程度上各有差异，其对当前学生的效果可能大相径庭，且其相对有效性还会随训练进程而变化。基于这一观察，我们提出了面向黑盒在线策略蒸馏的特权信息策略演化方法（EvoOPD）。

    arXiv:2611.06542v1 Announce Type: new  Abstract: Proprietary language models exhibit strong reasoning and coding capabilities, but their output distributions are inaccessible, making fine-grained distillation methods that rely on teacher logits difficult to apply directly to black-box models. We observe that rich textual feedback from a black-box teacher can be treated as privileged information (PI) and converted, through a PI-conditioned privileged teacher, into dense supervision signals on the student's own trajectories. However, what PI should be acquired from the teacher to most effectively support student learning remains unclear. PI strategies that differ in where they focus, what information they request, and how much detail they provide can have substantially different effects on the current student, and their relative effectiveness can further change as training progresses. Based on this observation, we propose Evolving Privileged Information Strategies for Black-Box On-Policy
    
[^55]: IDSISA：面向高效机器遗忘的影响蒸馏分片训练方法

    IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning

    [https://arxiv.org/abs/2611.06519](https://arxiv.org/abs/2611.06519)

    提出IDSISA方法，利用影响函数将易被遗忘的样本集中到少数分片，并通过未受影响分片的蒸馏恢复效用，在保持SISA精确删除保证的同时提升机器遗忘的效率和模型性能。

    

    机器遗忘旨在移除特定训练样本对模型的影响，以符合隐私法规。完全重训能提供强保证，但在大规模场景下计算代价过高；近似方法虽然高效，却缺乏形式化保证。SISA（分片、隔离、切片与聚合训练）通过数据分片和独立重训实现了可认证的删除，但随机划分会将遗忘请求分散到大多数分片中，且隔离重训会降低模型效用。我们提出IDSISA（影响蒸馏SISA），利用影响函数将易被遗忘的样本集中到少数分片中，并通过来自未受影响分片的蒸馏来恢复因隔离训练而损失的效用。从SISA重训步骤继承的删除保证依然精确，无需任何概率性限定；而基于蒸馏的效用恢复则是一个单独量化、经验上有界的近似（ε...（摘要原文在此处截断）

    arXiv:2611.06519v1 Announce Type: cross  Abstract: Machine unlearning removes the influence of specific training examples to comply with privacy regulations. Full retraining gives strong guarantees but is computationally prohibitive at scale; approximate methods are efficient but lack formal guarantees. SISA (Sharded, Isolated, Sliced, and Aggregated training) achieves certified deletion through data sharding and independent retraining, but random partitioning scatters forget requests across most shards, and isolated retraining degrades utility. We introduce IDSISA (Influence-Distilled SISA), which uses influence functions to concentrate forget-prone samples into few shards and distillation from unaffected shards to recover utility lost to isolation. The deletion guarantee inherited from SISA's retraining step remains exact and requires no probabilistic qualification; the distillation-based utility recovery is a separately quantified, empirically bounded approximation ($\epsilon_{\text
    
[^56]: 面向FDIC协助的问题银行收购的交互感知决策框架：来自FDIC倒闭银行面板数据与硅谷银行—第一公民银行案例的实证证据

    An Interaction-Aware Decision Framework for FDIC-Assisted Distressed Bank Acquisitions: Empirical Evidence from the FDIC Failed-Bank Panel and the SVB-First Citizens Case

    [https://arxiv.org/abs/2611.06514](https://arxiv.org/abs/2611.06514)

    提出蜘蛛网络过滤框架，将倒闭银行收购中五个维度信号间的相互依赖编码为四条可解释的交互规则，据此构建“推进/谨慎/回避”决策规则，并在2008—2024年55宗FDIC协助收购中以公告期正异常收益加以验证。

    

    倒闭银行的收购方需要联合权衡收购价值、监管保护、风险、买方实力与执行难度，因为每个信号的含义都依赖于其他信号：例如当政府损失分担机制吸收损失时，高不良贷款的重要性便会降低。本文提出的蜘蛛网络过滤框架将这种依赖关系编码为五个维度之间的四条可解释交互规则，为逻辑回归模型生成图启发性特征，并形成“推进/谨慎/回避”的决策规则。我们在2008至2024年间由母公司在美国上市的收购方完成的55宗联邦存款保险公司（FDIC）协助的收购交易上对该框架进行评估。其中六个财务输入来自每次银行倒闭前发布的监管报告；两项交易条款（中标方在出价时已知悉，但投资者在公告时才获知）来自监管机构的新闻稿。结果——公告前后三天窗口内的正异常收益……（原文摘要在此处被截断）

    arXiv:2611.06514v1 Announce Type: new  Abstract: Acquirers of failed banks weigh acquisition value, regulatory protection, risk, buyer strength, and execution difficulty jointly, because each signal's meaning depends on the others: high non-performing loans matter less when government loss sharing absorbs the losses. The proposed Spider Network Filtering Framework encodes such dependencies as four interpretable interaction rules among five dimensions, yielding graph-inspired features for a logistic model and a Pursue/Caution/Avoid decision rule. We evaluate it on 55 acquisitions assisted by the Federal Deposit Insurance Corporation in 2008-2024 by acquirers with a parent listed in the United States. Six financial inputs come from regulatory reports published before each failure; two deal terms, which the winning bidder knows when it bids but investors learn at the announcement, come from the regulator's press releases. The outcome, a positive three-day abnormal return around the announ
    
[^57]: 适者生存训练中用于种群剪枝的未来性能即时预测

    Nowcasting Future Performance for Population Pruning in Survival-of-the-Fittest Training

    [https://arxiv.org/abs/2611.06512](https://arxiv.org/abs/2611.06512)

    该论文提出了基于准确率与损失历史、权重统计以及SANE检查点编码的即时预测器来预测模型未来的测试准确率，并将其用于适者生存（SOTF）训练，从而在训练过程中更明智地对候选模型种群进行早期剪枝。

    

    训练高性能神经网络通常需要评估具有不同超参数、架构或随机种子的候选模型种群。将每个候选模型训练到底在计算上代价高昂，然而过早停止看似较弱的训练运行，又有因学习曲线交叉而丢弃最终表现更优模型的风险。为应对这一挑战，我们研究了中间训练检查点能否预测未来的模型准确率，从而支持更明智的剪枝决策。我们的即时预测器（nowcaster）可以预测五个epoch之后的测试准确率或训练结束时的准确率，其输入包括准确率与损失历史、权重统计信息以及SANE表示。SANE是一个transformer自编码器，它将检查点编码为一系列潜在token的序列。随后，我们将这些即时预测器应用于适者生存（Survival-of-the-Fittest, SOTF）训练中。SOTF在训练过程中反复根据预测准确率对各个运行进行排名，并停止……（摘要原文在此处截断）

    arXiv:2611.06512v1 Announce Type: new  Abstract: Training high-performing neural networks often involves evaluating populations of candidate models with varying hyperparameters, architectures, or random seeds. Training every candidate to completion is computationally expensive, yet stopping seemingly weaker runs early risks discarding models that would ultimately perform better, as learning curves can cross. To address this challenge, we investigate whether intermediate training checkpoints can predict future model accuracy and enable more informed pruning decisions. Our nowcasters predict test accuracy five epochs ahead or at the end of training. They are trained with accuracy and loss histories, weight statistics, and SANE representations as inputs. SANE is a transformer autoencoder that encodes a checkpoint as a sequence of latent tokens. We then use the nowcasters in Survival-of-the-Fittest (SOTF) training. SOTF repeatedly ranks runs during training by predicted accuracy and stops 
    
[^58]: ElecLoad：由多重事件引导的自适应电力负荷预测

    ElecLoad: Adaptive Electricity Load Forecasting Guided by Multiple Events

    [https://arxiv.org/abs/2611.06508](https://arxiv.org/abs/2611.06508)

    提出了ElecLoad——一个上下文感知的智能体，能够利用网页通知、天气预警等多重事件的影响及其相互作用，自适应地组织电力负荷预测过程，从而提升预测的准确性。

    

    准确的电力负荷预测为能源采购、发电调度和供需平衡提供关键的决策支持，有助于提升电力系统的经济效率和运行可靠性。然而，未来的电力需求可能受到网页通知、天气预警和活动日程中描述的多重事件之间复杂相互作用的影响。这些事件的影响随时间变化，且可能相互增强或抵消，使得电力负荷难以准确预测。虽然现有方法可以将上下文信息作为额外输入纳入模型，但如何利用多重事件的影响及其相互作用来引导负荷预测过程本身仍有待探索。为填补这一空白，我们提出了ElecLoad，一个面向多重事件电力负荷预测的上下文感知智能体，它能够自适应地组织预测过程。具体而言，我们训练ElecLoad将历史负荷与（摘要在此处截断）相结合……

    arXiv:2611.06508v1 Announce Type: new  Abstract: Accurate electricity load forecasting provides critical decision support for energy procurement, generation scheduling, and supply-demand balancing, contributing to the economic efficiency and operational reliability of power systems. However, future electricity demand can be affected by the complex interplay of multiple events described in Web notices, weather alerts, and event schedules. Their effects vary over time and can reinforce or offset one another, making electricity load difficult to forecast accurately. While existing methods can incorporate context as additional input, how to use the effects of multiple events and their interactions to guide the load forecasting process itself remains to be explored. To address this gap, we propose ElecLoad, a context-aware agent for multi-event electricity load forecasting that adaptively organizes the forecasting process. Specifically, we train ElecLoad to combine historical load with cont
    
[^59]: NaijaFacts：一个将语言与知识从关于尼日利亚的事实中分离出来的框架

    NaijaFacts: A Framework for Disentangling Language from Knowledge in Facts about Nigeria

    [https://arxiv.org/abs/2611.06504](https://arxiv.org/abs/2611.06504)

    该论文提出NaijaFacts基准，通过国家×语言矩阵将语言效应与知识效应分离，发现大语言模型对尼日利亚事实的知识缺口（英语提问准确率仅20%）远大于低资源语言带来的语言效应。

    

    面向非洲语言的多语言基准测试大多是将通用问题翻译成当地语言，衡量的是语言能力，而非模型是否了解关于非洲的事实。我们提出了NaijaFacts，这是一个关于尼日利亚的事实性和统计性问题基准，由母语者撰写，并基于权威的官方统计数据，每个问题都有英语、尼日利亚皮钦语、约鲁巴语、伊博语和豪萨语版本。通过让模型回答针对英国和美国的类似问题，我们构建了一个国家×语言的矩阵，从而将语言效应（模型知道该事实，但在低资源语言中性能下降）与知识效应（该事实在任何语言中都不存在）分离开来。在八个前沿模型和尼日利亚适配模型上，知识效应占主导地位：以英语提问时，针对美国/英国的准确率为49%–50%，而针对尼日利亚仅为20%。语言效应较小且与网络表征程度相关，从英语到约鲁巴语逐渐下降（在尼日利亚问题上下降9个百分点）……

    arXiv:2611.06504v1 Announce Type: cross  Abstract: Multilingual benchmarks for African languages mostly translate generic questions into local languages, measuring linguistic competence but not whether models know facts about Africa. We introduce \naija, a benchmark of factual and statistical questions about Nigeria, authored by native speakers and grounded in authoritative official statistics, with every question available in English, Nigerian Pidgin, \yoruba, Igbo, and Hausa. Answering similar questions for the UK and US yields a country by language matrix that disentangles a language effect (the model knows the fact but degrades in a low-resource language) from a knowledge effect (the fact is absent in every language). Across eight frontier and Nigeria-adapted models, the knowledge effect dominates: English-query accuracy is 49--50\% for the US/UK but 20\% for Nigeria. The language effect is smaller and tracks web representation, degrading from English to \yoruba{} (9 points on Nige
    
[^60]: 已签名却仍可被操纵：测量并弥合离线检索增强生成中的文本-向量完整性鸿沟

    Signed but Steerable: Measuring and Closing the Text-Vector Integrity Gap in Offline Retrieval-Augmented Generation

    [https://arxiv.org/abs/2611.06498](https://arxiv.org/abs/2611.06498)

    本文揭示了离线RAG签名知识包中“签名只证明发布者身份、无法保证向量与文本一致”这一安全鸿沟，提出了量化攻击者可劫持查询比例的“边际定律”，并给出了测量与弥合该完整性差距的验证方案。

    

    在边缘设备上运行的离线检索增强生成（RAG）通过携带文本和预计算嵌入向量的签名知识包进行更新。签名只能证明知识包由谁发布，却不能证明其中的向量确实编码了对应的文本。我们研究了这种“文本-向量完整性鸿沟”。一个会重新计算向量并在余弦容差范围内接受它们的验证器，会将可预测比例的查询拱手让给控制向量的攻击者：劫持率等于该容差处角度检索边际的分布（即“边际定律”）。在五个语料库和两个嵌入模型上，检索边际都很小，因此处于浮点噪声级别的理想化容差会使最多5.5%的查询暴露于攻击之下（排除完全重复文档后，任何语料库中最多为1.2%），而根据我们对小型嵌入模型的CPU INT8-FP32对比校准的阈值（取诚实余弦的经验0.1%分位数处）则会使98-100%的查询暴露于攻击之下；在树莓派上……（原文摘要在此处被截断）

    arXiv:2611.06498v1 Announce Type: cross  Abstract: Offline retrieval-augmented generation (RAG) on edge devices is updated through signed knowledge packs that carry text and precomputed embedding vectors. A signature proves who published a pack, not that its vectors encode its text. We study this text-vector integrity gap. A verifier that recomputes vectors and accepts them within a cosine tolerance concedes a predictable share of queries to an attacker who controls the vectors: the hijack rate equals the distribution of angular retrieval margins at the tolerance (the margin law). Margins are small in five corpora and two embedding models, so an idealised tolerance at the level of floating-point noise leaves up to 5.5% of queries exposed (at most 1.2% in any corpus once exact duplicate documents are excluded), and a threshold calibrated from our CPU INT8-FP32 comparison of a small embedding model (at the empirical 0.1% quantile of honest cosines) leaves 98-100% exposed; on a Raspberry 
    
[^61]: 风险敏感的非参数与核赌博机

    Risk-Sensitive Non-Parametric and Kernel Bandits

    [https://arxiv.org/abs/2611.06489](https://arxiv.org/abs/2611.06489)

    本文提出NPCB乐观算法，将累积前景理论（CPT）风险敏感评估引入带连续动作空间的非参数赌博机，证明了其CPT遗憾与经典非参数赌博机量级一致，说明风险敏感性不带来额外的量级遗憾代价，并扩展至未知噪声分布的情形。

    

    我们研究了非参数赌博机问题，其中各臂（arm）采用累积前景理论（CPT）而非期望奖励来进行评估。我们考虑具有臂相关噪声的连续动作空间，并提出非参数CPT赌博机（$\mathtt{NPCB}$）——一种将均值估计与基于CPT的决策制定相解耦的乐观算法。我们证明了量级为 $\widetilde{\mathcal O}\bigl(\sqrt{T\,\dim_E(\mathcal F,T^{-2})\,\log N(\mathcal F,T^{-2},\|\cdot\|_\infty)}\bigr)$ 的CPT遗憾，其中 $\dim_E$ 是函数类 $\mathcal F$ 的Eluder维度，$T$ 是时间范围，$N$ 是其在无穷范数下的覆盖数。该结果与经典非参数赌博机的遗憾量级相匹配，表明CPT风险敏感性不会带来额外的量级遗憾代价。我们进一步将 $\mathtt{NPCB}$ 扩展到噪声累积分布函数（CDF）未知的情形，保留了已知CDF时的均值学习项，并添加了由分布Eluder维度所支配的CDF学习代价。

    arXiv:2611.06489v1 Announce Type: cross  Abstract: We study non-parametric bandits in which arms are evaluated using Cumulative Prospect Theory (CPT) rather than expected reward. We consider continuous action spaces with arm-dependent noise and propose Non-Parametric CPT Bandits ($\mathtt{NPCB}$), an optimistic algorithm that decouples mean estimation from CPT-based decision-making. We prove CPT regret of order $\widetilde{\mathcal O}\bigl(\sqrt{T\,\dim_E(\mathcal F,T^{-2})\,\log N(\mathcal F,T^{-2},\|\cdot\|_\infty)}\bigr)$, where $\dim_E$ is the Eluder dimension of the function class $\mathcal F$, $T$ is the time horizon, and $N$ is its covering number in the sup norm. This matches the order-wise regret of classical non-parametric bandits, showing that CPT risk sensitivity incurs no additional order-wise regret cost. We extend $\mathtt{NPCB}$ to unknown noise CDFs, retaining the known-CDF mean-learning term and adding a CDF-learning cost governed by a distributional Eluder dimension.
    
[^62]: 模型合并不需要统一的LoRA秩

    Model Merging Does Not Require Uniform LoRA Ranks

    [https://arxiv.org/abs/2611.06486](https://arxiv.org/abs/2611.06486)

    该论文证明模型合并不要求各专家使用统一的LoRA秩，为不同领域专家分配各自的最佳秩可让平均法和基于效用的选择方法零成本地提升最多1.8个百分点的多领域准确率，而冲突解决类方法则表现不一致甚至可能损失4个百分点。

    

    模型合并将各自独立微调的专家模型组合为单一模型，而无需重新训练。标准做法是以相同的LoRA秩训练每个专家，这使得秩的选择在各专家之间相互耦合，尽管每个领域其实都有各自的最佳秩。我们证明这种耦合是合并方法的特性，而非合并本身的固有属性。我们在三个领域（数学、指令遵循、多语言）、四种秩和三种模型规模下训练了36个专家模型，然后用六种方法分别在统一秩和各领域自身最佳秩下进行合并。为专家分配不同秩对平均法和基于效用的选择方法没有任何代价，反而可带来最高1.8个百分点的多领域准确率提升，而冲突解决类方法表现不一致，最多可能损失4个百分点。这种差异有一个简单的原因：合并仅保留汇集谱成分中的固定比例，而按大小选择这些成分的方法会丢弃许多本可以降低损失的成分。它们还倾向于丢弃…

    arXiv:2611.06486v1 Announce Type: cross  Abstract: Model merging combines separately fine-tuned experts into a single model without retraining. Standard practice trains every expert at the same LoRA rank, which couples the rank choice across experts even though each domain has its own best rank. We show this coupling is a property of the merge method, not of merging. We train 36 experts on three domains (mathematics, instruction following, multilingual) at four ranks and three model scales, then merge with six methods at uniform ranks and at each domain's own best rank. Giving experts different ranks costs averaging and utility-based selection nothing, and gains them up to 1.8 pp of multi-domain accuracy, while the conflict-resolution methods are inconsistent and can lose up to 4 pp. The split has a simple cause. Merging keeps only a fixed share of the pooled spectral components, and methods that choose them by size discard many that would have reduced loss. They also tend to discard t
    
[^63]: 多轮强化学习究竟在LLM智能体中提升了什么？

    What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?

    [https://arxiv.org/abs/2611.06483](https://arxiv.org/abs/2611.06483)

    本文通过Pass@(K,T)评估发现多轮智能体强化学习的优势依赖充足的交互轮次预算，且主要体现在错误恢复与长程韧性而非单步决策质量上，进而提出在训练中动态改变交互预算的BudgetMix RL，以在多种执行设置下实现稳健性能。

    

    虽然多轮智能体强化学习（RL）在长程任务上表现出色，但其性能提升对交互约束仍然高度敏感。通过Pass@(K,T)在采样广度K和交互深度T上进行评估，我们表明轨迹层面的RL优势依赖于充足的轮次预算，当预算收紧时该优势会显著减弱。此外，智能体RL并非均匀地提升单步决策质量；相反，其优势集中在错误恢复和长程韧性上。为缓解这种预算依赖带来的部署风险，我们对过程监督信号的益处与局限性进行了初步讨论，并引入BudgetMix RL作为替代方案，该方案在训练期间改变交互预算。实验表明，BudgetMix RL在多种执行设置下均能取得稳健的性能。项目主页：https://rethinking-agentic-rl.github.io/

    arXiv:2611.06483v1 Announce Type: cross  Abstract: While multi-turn agentic reinforcement learning (RL) excels on long-horizon tasks, its performance gains remain highly sensitive to interaction constraints. Evaluating across sampling breadth $K$ and interaction depth $T$ via $Pass@(K,T)$, we show that trajectory-level RL advantages depend on sufficient turn budgets, diminishing significantly when budgets tighten. Besides, agentic RL does not uniformly elevate single-step decision quality; rather, its strength concentrates in error recovery and long-horizon resilience. To mitigate the deployment risks caused by this budget dependence, we provide a preliminary discussion on the benefits and limitations of process supervision signals and introduce BudgetMix RL as an alternative that varies interaction budgets during training. Experiments demonstrate that BudgetMix RL achieves robust performance across diverse execution settings. Project Page: https://rethinking-agentic-rl.github.io/
    
[^64]: 对齐代价是分级的：SFT保留下长度归一化DPO的任务选择性能力损失

    The Alignment Tax Is Graded: Task-Selective Capability Loss Under Length-Normalized DPO with SFT Retention

    [https://arxiv.org/abs/2611.06476](https://arxiv.org/abs/2611.06476)

    对齐微调带来的能力损失并非均匀发生，而是分级且任务选择性的——抽取类任务几乎不受影响，JSON生成轻微下降，而路由任务出现约30分的严重退化。

    

    设定。这是一项实证特征描述研究；我们未提出新方法，也未与其他目标函数进行对比。我们在结构化财务报告任务上，使用稀疏LoRA和长度归一化DPO加上SFT保留项（即RPO目标）对Qwen3.5-4B进行微调，并跟踪下游能力随隐式奖励边际增长至约7个β缩放单位（nf4量化训练；在未量化的基础模型上以fp16测量时约为3）时的变化。问题。边际增长是否会在所有任务上均匀地降低能力，还是具有选择性？发现。在一项预注册的研究（8个随机种子，其中6个在冻结前未使用）中，通过伪影感知评分和TOST等效性检验，对齐代价是分级的且具有任务选择性：信息抽取能力得以保留（Δ = -0.55分，在±5分范围内等效），json能力损失很小（Δ = -3.52分，在聚类后的项目级区间上等效性检验未通过），而路由能力严重退化（Δ = -30分）。

    arXiv:2611.06476v1 Announce Type: new  Abstract: Setting. This is an empirical characterization study; we propose no new method and run no comparisons against alternative objectives. We fine-tune Qwen3.5-4B with sparse LoRA and length-normalized DPO plus an SFT retention term (the RPO objective) on structured financial-filing tasks, and track downstream capability as the implicit reward margin grows to $\sim 7$ $\beta$-scaled units (nf4 training; $\sim 3$ under fp16 measurement on the unquantized base). Question. Does margin growth degrade capability uniformly across tasks, or selectively? Findings. In a pre-registered study (8 seeds, 6 unseen at freeze) with artifact-aware scoring and TOST equivalence testing, the alignment tax is graded and task-selective: extraction is preserved ($\Delta = -0.55$ pts, equivalent at $\pm5$ pts), json loses little ($\Delta = -3.52$ pts, equivalence fails on a clustered item-level interval), and router degrades severely ($\Delta = -30$ pts). Router's l
    
[^65]: 多少个有序投影能确定k维空间中的n个点？

    How many sorted projections determine n points in k dimensions?

    [https://arxiv.org/abs/2611.06470](https://arxiv.org/abs/2611.06470)

    本文借助Hair和Sahai关于随机线性码列表恢复的两个引理，证明了在任意k维空间中，几乎任意选择适当数量的方向即可确定n个点的多重集，从而回答了Dym等人提出的高维开放问题。

    

    设 $d_1,\dots,d_m$ 为 $\mathbb{R}^k$ 中的方向，其中 $k\ge 2$。$\mathbb{R}^k$ 中 $n$ 个点构成的多重集 $X$ 会投影为 $m$ 个数字多重集 $\{\{x\cdot d_j : x\in X\}\}$，或等价地，投影为这些数字的 $m$ 个排序列表。需要多少个方向才能使这些投影确定任意一个由 $n$ 个点构成的多重集？在每个维度上都有两个已知的简单界：总是需要多于 $(k-1)\log_2 n$ 个方向，而处于一般位置的 $(k-1)n+1$ 个方向总是足够的。在平面上，人们知道得更多：Matousek、Privetivy 和 Skovron 证明了几乎任何 $O(\log n \log\log n)$ 个方向的选择都足够。至于在高维空间中是否有类似的结果成立，这是 Dym、Wellershoff、Tsoukanis、Levy 和 Balan 留下的开放问题。我们观察到，Hair 和 Sahai 为随机线性码的列表恢复所证明的两个引理可以回答该问题，并且通过收紧他们的论证，我们证明了在每个维度中……

    arXiv:2611.06470v1 Announce Type: cross  Abstract: Let $d_1,\dots,d_m$ be directions in $\mathbb{R}^k$, where $k\ge 2$. A multiset $X$ of $n$ points of $\mathbb{R}^k$ projects to the $m$ multisets of numbers $\{\{x\cdot d_j : x\in X\}\}$, or equivalently to the $m$ sorted lists of these numbers. How many directions are needed for these projections to determine every multiset of $n$ points? Two simple bounds are known in every dimension: more than $(k-1)\log_2 n$ directions are always needed, and $(k-1)n+1$ directions in general position always suffice. In the plane much more is known: Matousek, Privetivy and Skovron showed that almost every choice of $O(\log n \log\log n)$ directions suffices. Whether anything similar holds in higher dimensions was left open by Dym, Wellershoff, Tsoukanis, Levy and Balan. We observe that two lemmas of Hair and Sahai, proved for list recovery of random linear codes, answer that question, and by tightening their argument we show that in every dimension a
    
[^66]: 固定分辨率迁移学习下的EfficientNet家族内缩放用于频域光学相干断层扫描视网膜疾病分类

    Within-Family EfficientNet Scaling under Fixed-Resolution Transfer Learning for Spectral-Domain Optical Coherence Tomography Retinal Disease Classification

    [https://arxiv.org/abs/2611.06452](https://arxiv.org/abs/2611.06452)

    该研究发现固定分辨率迁移学习中EfficientNet家族内按规模缩放并不能单调提升SD-OCT视网膜疾病分类性能，最小的B1以96.95%的准确率超越更大的B7模型，表明在医学图像迁移学习中盲目增大模型并非最优选择。

    

    背景/目的：卷积神经网络（CNN）在基于频域光学相干断层扫描（SD-OCT）的视网膜疾病分类中表现良好，但EfficientNet在自然图像上的缩放层级能否迁移到固定分辨率的医学图像迁移学习中仍不明确。本研究评估了EfficientNet-B0至B7在SD-OCT四分类任务中的表现以及准确率与效率之间的权衡。方法：在Kermany数据集上以224×224分辨率评估了八个ImageNet预训练变体（实例化骨干网络参数量为405万至6410万）。灰度图像被复制为RGB三通道，并使用模型内置的预处理与数据增强进行处理。训练流程包括冻结骨干网络的分类头训练，随后进行部分微调。评估指标涵盖分类性能、判别能力、效率以及可视化分析。结果：测试准确率呈非单调变化，从B0的95.93%到B1的96.95%；B7达到96.73%。B1取得了最高的准确率以及玻璃膜疣类别的最高F1分数（0.878……摘要至此被截断）。

    arXiv:2611.06452v1 Announce Type: cross  Abstract: Background/Objectives: CNNs perform well in retinal disease classification from spectral-domain optical coherence tomography (SD-OCT), but whether EfficientNet's natural-image scaling hierarchy transfers to medical-image transfer learning at fixed resolution remains unclear. This study evaluates EfficientNet-B0-B7 for four-class SD-OCT classification and accuracy-efficiency trade-offs. Methods: Eight ImageNet-pretrained variants (4.05-64.10M instantiated backbone parameters) were evaluated at 224x224 on the Kermany dataset. Grayscale images were replicated to RGB and processed using model-embedded preprocessing and augmentation. Training comprised frozen-backbone head training followed by partial fine-tuning. Metrics covered classification, discrimination, efficiency, and visual analyses. Results: Test accuracy varied non-monotonically from 95.93% (B0) to 96.95% (B1); B7 achieved 96.73%. B1 had the highest accuracy and Drusen F1 (0.878
    
[^67]: 图基础模型可以成为有效的多变量时间序列异常检测器

    Graph Foundation Models Can Become Effective Multivariate Time Series Anomaly Detectors

    [https://arxiv.org/abs/2611.06441](https://arxiv.org/abs/2611.06441)

    提出Gfm4Mtsad适配框架，通过引入结构与信息契约对齐（SICA）模块，将预训练图基础模型成功迁移到多变量时间序列异常检测任务中，克服了跨模态不匹配、伪异常监督不可靠和决策失真三大挑战。

    

    多变量时间序列异常检测（MTSAD）依赖于对时间动态特性和传感器间依赖关系的建模，以识别复杂系统的故障。尽管基于图的检测器能够显式地建模传感器拓扑结构，但它们需要在有限的目标域数据上从头训练特定任务的网络，因此仍然容易受到虚假相关性的影响，并且无法复用可迁移的先验知识。虽然预训练的图基础模型（GFMs）提供了可迁移的拓扑先验和异常判别能力，但将其直接适配到MTSAD面临三大障碍：跨模态的特征-拓扑不匹配、不可靠的伪异常监督，以及节点到窗口聚合过程中的决策失真。为了填补这些空白，我们提出了一种新颖的适配框架Gfm4Mtsad，将预训练的图基础模型充分应用于多变量时间序列异常检测。我们引入了结构与信息契约对齐（SICA）模块，以桥接连续时间序列与预训练的图基础模型空间。该模块…

    arXiv:2611.06441v1 Announce Type: new  Abstract: Multivariate time series anomaly detection (MTSAD) relies on modeling both temporal dynamics and inter-sensor dependencies to identify complex system faults. Although graph-based detectors explicitly model sensor topologies, they train task-specific networks from scratch on limited target-domain data, remaining vulnerable to spurious correlations and unable to reuse transferable priors. While pretrained Graph Foundation Models (GFMs) offer transferable topological priors and anomaly discriminability, directly adapting them to MTSAD is hindered by cross-modal feature-topology mismatch, unreliable pseudo-anomaly supervision, and decision distortion during node-to-window aggregation. To fill these gaps, we propose a novel adaptation framework, Gfm4Mtsad, which unleashes pretrained GFMs for MTSAD. We introduce the Structure-and-Information Contract Alignment (SICA) module to bridge continuous time series and the pretrained GFM space. This mo
    
[^68]: 理解大语言模型中思维链推理的分布外泛化机制

    Understanding the Out-of-Distribution Generalization of Chain-of-Thought Reasoning in LLMs

    [https://arxiv.org/abs/2611.06438](https://arxiv.org/abs/2611.06438)

    该论文首次从理论上将大语言模型的思维链推理建模为组合式分布外泛化问题，证明其通过复用共享的单步预测器组合分布内原子转换来实现泛化，且分布外风险由分布内子任务风险之和乘以一个与推理步数无关的 $\mathcal{O}(k)$ 系数所控制。

    

    思维链推理显著提升了大语言模型的表现，但其分布外（OOD）泛化机制在理论上仍未被充分探索。我们将思维链视为组合式分布外泛化问题来研究：目标任务缺乏完整的推理轨迹或直接的输入-输出对，而训练数据仅提供分布内（ID）的原子步骤数据。我们证明思维链通过反复应用一个共享的单步预测器来组合可复用的分布内转换，从而实现泛化。在理论上，设 $T$ 表示目标任务中的推理步数，$n$ 表示每个子任务的训练样本数，$k$ 表示每步的输出 token 数，则分布外任务的风险由分布内子任务风险之和控制，其缩放系数为 $\mathcal{O}(k)$ 量级且与 $T$ 无关，尽管该风险之和本身仍可能随 $T$ 增长。基于 Rademacher 复杂度的分析进一步表明，该估计（原文在此处截断）。

    arXiv:2611.06438v1 Announce Type: new  Abstract: Chain-of-Thought (CoT) reasoning substantially improves Large Language Models (LLMs), yet its out-of-distribution (OOD) generalization mechanism remains theoretically underexplored. We study CoT as compositional OOD generalization: the target task lacks complete reasoning trajectories or direct input-output pairs, while training provides only in-distribution (ID) atomic step data. We show that CoT generalizes by repeatedly applying a shared single-step predictor to compose reusable ID transitions. Theoretically, let $T$ denote the number of reasoning steps in the target task, $n$ the number of training samples per sub-task, and $k$ the number of output tokens per step; then the OOD task risk is controlled by the sum of the ID sub-task risks, scaled by a coefficient of order $\mathcal{O}(k)$ that is independent of $T$, although the summed risk itself may still grow with $T$. A Rademacher-complexity analysis further shows that the estimati
    
[^69]: 用于推理的硬连线分布式处理单元：在传感器端做出神经决策，解放主机

    Hardwired Distributed Processing Units for Inference: Neural Decisions at the Sensor that Free the Host

    [https://arxiv.org/abs/2611.06436](https://arxiv.org/abs/2611.06436)

    该论文提出一种在低成本Artix-7 FPGA上实现的硬连线分布式处理单元（HDPU），仅用1,814个LUT、无需处理器和操作系统即可在传感器端以固定时延完成神经推理决策（如斑马线检测），从而将全部推理负担从主机中解放出来。

    

    我们证明了一种硬连线分布式处理单元（HDPU）可以在传感器端完成神经决策，从而将推理任务从主机中移除。该单元专为低成本Artix-7 FPGA上的斑马线检测而构建，在固定的2,269,477个时钟周期（34.04毫秒，取决于FPGA时钟频率）内对每一帧相机图像做出决策，处于37.49毫秒的帧周期之内；在全部639张测试图像上，它在板卡上返回的分数和决策与其位精确的软件模型完全一致。从相机像素到决策的整个检测路径仅占用1,814个LUT，无需处理器、无需操作系统，也无需全分辨率帧存储器。当该单元直接给出答案而不是报告“不确定”时，其在街景照片上的回答正确率达到95.7%；此外，一个硬件对比度处理级将类相机行车记录仪帧上的准确率从47.7%提升到78.2%。主机因此被完全免除了推理负担：每个帧周期可节省一个核心8.13毫秒的计算时间，而全精度ResNet-18则需要更多资源。

    arXiv:2611.06436v1 Announce Type: cross  Abstract: We show that a hardwired distributed processing unit (HDPU) can take a neural decision at the sensor and so remove the inference from the host. Built for zebra-crossing detection on a low-cost Artix-7 FPGA, the unit decides on every camera frame in a fixed 2,269,477 clock cycles (34.04 ms depends upon Fpga clk ), inside the 37.49-ms frame period, and on the board it returns the same scores and decisions as its bit-exact software model on all 639 test images. The whole detection path, from camera pixels to decision, takes 1,814 LUTs, with no processor, no operating system and no full-resolution frame store. When the unit answers instead of reporting "not sure", 95.7% of its answers on street photographs are correct, and a hardware contrast stage lifts accuracy on camera-like dashcam frames from 47.7% to 78.2%. The host is spared all of the inference: 8.13 ms of a core in every frame period, where a full-precision ResNet-18 would need mo
    
[^70]: PARGAR：面向稀疏导频神经OFDM接收机的导频感知图注意力机制

    PARGAR: Pilot-Aware Graph Attention for Sparse-Pilot Neural OFDM Receivers

    [https://arxiv.org/abs/2611.06432](https://arxiv.org/abs/2611.06432)

    本文提出PARGAR，一种导频感知的图注意力神经接收机，通过将OFDM资源网格建模为融合局部时频边与导频锚点连接的图，有效提升了稀疏导频配置下的比特检测性能。

    

    在5G新空口（NR）以及新兴的6G无线系统中，可靠的信道估计与符号检测依赖于解调参考信号（DMRS）。这些导频在正交频分复用（OFDM）时频资源网格上的密度和位置会同时影响接收机的可靠性与频谱效率。稀疏的DMRS布局会增大导频观测点与数据资源单元（RE）之间的时频距离，这可能使纯局部处理方式难以充分利用导频信息。本文提出了PARGAR（导频感知图注意力接收机），一种将OFDM资源网格表示为图以进行比特检测的图注意力神经接收机。该图将局部时频边与从数据RE指向选定导频RE的导频锚点连接相结合，并进一步研究了基于近似相干时间和相干带宽的信道尺度化导频锚点规则。

    arXiv:2611.06432v1 Announce Type: cross  Abstract: Reliable channel estimation and symbol detection in 5G New Radio (NR), as well as emerging 6G radio systems, rely on demodulation reference signals (DMRS). The density and placement of these pilots on the orthogonal frequency-division multiplexing (OFDM) time--frequency resource grid affect both receiver reliability and spectral efficiency. Sparse DMRS layouts increase the time--frequency distance between pilot observations and data resource elements (REs), which can make pilot information harder to exploit with purely local processing. This paper proposes \textbf{PARGAR (Pilot-Aware Graph Attention Receiver)}, a graph-attention neural receiver that represents the OFDM resource grid as a graph for bit detection. The graph combines local time--frequency edges with pilot-anchor links from data REs to selected pilot REs, and we further study a channel-scaled pilot-anchor rule based on approximate coherence time and coherence bandwidth. Us
    
[^71]: 部分可解释决策树

    Partially Interpretable Decision Trees

    [https://arxiv.org/abs/2611.06428](https://arxiv.org/abs/2611.06428)

    该论文提出了三种部分可解释决策树算法（Branches精确搜索扩展、基于CART的转换算法和考虑转交选项的广义CART），并通过形式化界限证明它们在真实数据集上表现相似的原因在于智能树构建的潜在收益本身有限。

    

    部分可解释模型在简单可解释模型的基础上进行扩展，增加了将复杂输入转交给准确但不可解释的黑箱模型处理的选项。转交的输入越多，模型越准确；转交的越少，模型越可解释。现有的训练此类部分可解释模型的方法大多基于线性模型或决策规则。我们提出了三种决策树算法：一是现有Branches算法的部分可解释扩展，该算法通过精确搜索来优化树结构；二是在标准CART之上运行的转换算法，其计算效率更高；三是广义CART算法，其分裂准则在树的生长过程中考虑了转交选项。令人惊讶的是，这三种算法在许多真实数据集上的表现相似。我们可以通过对智能树构建潜在收益的形式化界限来解释这一现象。

    arXiv:2611.06428v1 Announce Type: new  Abstract: Partially interpretable models extend simple interpretable models with the option to defer complicated inputs to an accurate but uninterpretable black-box model. By deferring more inputs, the model gets more accurate; by deferring fewer, it gets more interpretable. Existing procedures to train such partially interpretable models build on linear models or decision rules. We propose three algorithms for decision trees: a partially interpretable extension of the existing Branches algorithm, which performs exact search to optimize the tree; a conversion algorithm that we run on top of standard CART, which is computationally more efficient; and a generalized CART whose splitting criterion takes into account the option to defer while growing the tree. Perhaps surprisingly, the performance of the three algorithms is similar across many realworld datasets. We can explain this through a formal bound on the potential gain of smart tree constructio
    
[^72]: D3D：面向异步投机解码的解耦扩散草稿生成

    D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding

    [https://arxiv.org/abs/2611.06425](https://arxiv.org/abs/2611.06425)

    D3D 提出一种不依赖目标模型隐藏状态的独立扩散草稿模型，使草稿生成与验证可异步并行执行，在多个基准上平均实现约 5 倍的解码加速。

    

    目标条件扩散草稿模型可以并行预测候选 token，但每个新块都必须等待目标模型更新后的隐藏状态。这迫使草稿生成与验证必须串行运行，即使每个块都是一次性生成，速度提升仍然受限。我们提出 D3D，一种独立的扩散草稿模型，消除了这种轮次间的依赖。D3D 将分块去噪与辅助自回归监督相结合，在不依赖目标模型隐藏状态的情况下实现较长的被接受前缀。这种独立性使 D3D 能够在验证过程中通过一次打包前向传递草拟后续块，从而实现异步投机解码。我们根据各层对接受长度的影响逐步剪枝，以降低打包草稿生成的成本。在以 Qwen3-8B 为目标模型的三个基准测试中，移除 36 层中的 5 层后，平均接受长度仍保持在 99% 以上。在七个基准测试中，D3D 平均实现了 5.00 倍和 5.71 倍的解码加速（原文在此处截断）。

    arXiv:2611.06425v1 Announce Type: new  Abstract: Target-conditioned diffusion drafters predict candidate tokens in parallel, yet each new block must wait for updated target-model hidden states. This forces drafting and verification to run sequentially, limiting speedup even though each block is generated in one pass. We introduce D3D, a standalone diffusion drafter that removes this inter-round dependency. D3D combines blockwise denoising with auxiliary autoregressive supervision to achieve long accepted prefixes without target hidden states. This independence lets D3D draft successor blocks during verification in one packed pass, enabling asynchronous speculative decoding. We progressively prune layers according to their effect on acceptance length to reduce the cost of packed drafting. Removing five of 36 layers retains over 99\% of mean acceptance length across three benchmarks with Qwen3-8B as the target. Across seven benchmarks, D3D averages 5.00$\times$ and 5.71$\times$ decoding 
    
[^73]: 《防御基于深度学习的人脸识别流水线免受后门攻击的一揽子技巧》

    A Bag of Tricks for Defending Deep Learning-based Face Recognition Pipelines against Backdoor Attacks

    [https://arxiv.org/abs/2611.06412](https://arxiv.org/abs/2611.06412)

    本文针对训练数据可能被第三方污染的人脸识别系统，提出了三种实用的后门攻击防御技巧——一种在训练阶段检测被投毒的身份，两种在推理阶段过滤可疑输入，为系统提供商提供了可落地的安全加固方案。

    

    本技术报告针对与行业从业者最相关的威胁模型：尽管模型训练在内部完成，但训练数据来源于可能已被入侵的第三方。在此背景下，我们提出了三种针对基于数据投毒的后门攻击的实用防御“技巧”。第一种在训练阶段运行，可在训练期间或流水线部署之前检测被投毒的身份。后两种在推理阶段运行，在将结果返回给潜在攻击者之前过滤可疑输入。每种方法都介绍了其动机、算法和计算开销，并在真实的人脸识别工作负载上进行了评估。我们的目标是为系统提供商提供可实际操作的加固技术。

    arXiv:2611.06412v1 Announce Type: cross  Abstract: The protection of mature, multi-stage Face Recognition Systems is crucial to system providers. This technical report targets the threat model most relevant to industry practitioners: although model training is performed in-house, training data is sourced from possibly-compromised third parties. Given this setting, we present three practical defense "tricks" against data poisoning-based backdoor attacks. The first operates at training-time, detecting poisoned identities during training or prior to a pipeline's deployment. The last two operate at inference-time, filtering suspicious inputs before an answer can be sent back to a potential attacker. Each method is presented with their motivation, algorithms and computational impact, and evaluated on realistic face recognition workloads. Our goal is to provide actionable hardening techniques to system providers.
    
[^74]: 大语言模型智能体之间的零样本协调

    Zero-Shot Coordination among LLM Agents

    [https://arxiv.org/abs/2611.06409](https://arxiv.org/abs/2611.06409)

    该论文通过设计隔离特定协调挑战的最小化环境与简单通用脚手架，首次系统研究了独立开发的LLM智能体之间的零样本协调问题，发现即使最前沿的LLM智能体也难以协调，其主要原因在于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。

    

    我们研究零样本协调，即独立开发的智能体必须在测试时进行协调的场景。尽管零样本协调在强化学习文献中已被充分研究，但对于大语言模型智能体在此类设置中的性能却知之甚少，尽管它们在此类环境中的部署日益增多。现有的LLM协调工作通常依赖于专门化的脚手架架构，而独立开发的智能体在实践中不太可能共享这些架构。此外，在复杂环境（如Hanabi）上进行评估使得难以准确定位协调失败的根源。相比之下，我们专注于在最小化环境中使用简单、通用的脚手架架构，这些环境旨在隔离特定的协调挑战。我们的结果表明，即使在这些受控设置中，前沿LLM智能体也难以协调，这主要是由于对协调问题的理解有限以及对合作伙伴信念的推理能力较弱。虽然LLM引入了语义信息作为额外的（摘要在此处截断）……

    arXiv:2611.06409v1 Announce Type: cross  Abstract: We study zero-shot coordination (ZSC), where independently developed agents must coordinate at test time. While ZSC has been well studied in the RL literature, far less is known about the performance of LLM agents despite their increasing deployment in such settings. Existing work on LLM coordination often relies on specialised scaffolding that independently developed agents are unlikely to share in practice. Moreover, evaluations on complex environments (e.g., Hanabi) make it difficult to pinpoint the sources of coordination failure. In contrast, we focus on simple, general-purpose scaffolds in minimal environments designed to isolate specific coordination challenges. Our results show that even in these controlled settings, frontier LLM agents struggle to coordinate, largely due to limited understanding of the coordination problem and weak reasoning about their partner's beliefs. While LLMs introduce semantic information as an additio
    
[^75]: GIST：面向离线RLVR的基于吉布斯目标的层级数据选择

    GIST: Gibbs-Targeted Hierarchical Data Selection for Offline RLVR

    [https://arxiv.org/abs/2611.06395](https://arxiv.org/abs/2611.06395)

    该论文提出GIST方法，基于双向KL散度在参考策略处的精确分解，揭示出正确响应应聚焦其低概率token、错误响应应聚焦策略高频生成响应的高概率token这两条监督原则，并据此设计两级层级数据选择方法来优化离线RLVR的训练监督。

    

    具有可验证奖励的离线强化学习在固定的采样响应池上进行训练，通常对每个响应的每个token都进行监督，然而响应和token的信息量并不相同。我们通过KL正则化RLVR的Gibbs最优策略（即离线训练的目标）来研究应当监督哪些内容，并用双向KL散度来度量策略与该目标之间的失配。在参考策略处，这种失配可以精确分解，并将两种结果区分开来：正确响应的贡献主要来自前向项，集中于其低概率token上；错误响应的贡献则来自反向项，集中于策略经常生成的响应及其高概率token上。基于这一分析，我们提出了GIST（Gibbs-Targeted Hierarchical Data Selection，吉布斯目标导向的层级数据选择），这是一种两级选择方法，利用可获得的信号来近似上述原则……（摘要原文在此处截断）

    arXiv:2611.06395v1 Announce Type: new  Abstract: Offline reinforcement learning with verifiable rewards (RLVR) trains on a fixed pool of sampled responses and typically supervises every token of every response, although responses and tokens are not equally informative. We study which ones to supervise through the Gibbs-optimal policy of KL-regularized RLVR, which offline training targets, and measure the mismatch between a policy and this target with a bidirectional KL divergence. At the reference policy, this mismatch factorizes exactly and separates the two outcomes: correct responses contribute mainly through the forward term, concentrated on their low-probability tokens, and incorrect responses through the reverse term, concentrated on the responses the policy generates often and on their high-probability tokens. Based on this analysis, we propose GIST (Gibbs-Targeted Hierarchical Data Selection), a two-level selection method that approximates these principles with signals availabl
    
[^76]: KolTa：一种基于物理信息的有限数据配方发现主动学习框架

    KolTa: A Physics-Informed Active Learning Framework for Formulation Discovery with Limited Data

    [https://arxiv.org/abs/2611.06394](https://arxiv.org/abs/2611.06394)

    KolTa是一个物理信息驱动的主动学习框架，通过结合三阶段预测器与物理残差校准UCB采集规则，在有限实验预算下高效发现目标温度下排名前5%的最优配方。

    

    在少数配方已被测量且实验预算有限的情况下，寻找高性能配方十分困难。我们提出了KolTa，这是一个主动学习框架，在固定查询预算下通过选择批次来发现目标温度下排名前5%的最优配方。KolTa将一个三阶段预测器（涵盖组分贡献、角色条件相互作用以及锚定温度响应）与任务特定的软物理约束相结合。其物理残差校准上置信界（PRC-UCB）采集规则融合了预测效用、集成分歧度以及目标温度物理残差，并通过拟合折外带符号误差来确定非负探索权重。我们在三个实测任务和一个模拟任务上对该框架进行了评估，涵盖二元混合物粘度、药物溶解度、离子液体粘度和电解质电导率。KolTa在学习曲线下面积均值（AULC）上排名第一或第二。

    arXiv:2611.06394v1 Announce Type: new  Abstract: Finding high-performing formulations is difficult when few have been measured and experimental budgets are limited. We propose KolTa, an active learning framework that selects batches to discover the top 5% of formulations at a target temperature under a fixed query budget. KolTa combines a three-stage predictor of component contributions, role-conditioned interactions, and anchored temperature response with task-specific soft physics constraints. Its Physics-Residual-Calibrated Upper Confidence Bound (PRC-UCB) acquisition rule combines predicted utility, ensemble disagreement, and a target-temperature physics residual, with nonnegative exploration weights fitted to out-of-fold signed errors. We evaluate the framework on three measured tasks and one simulated task covering binary-mixture viscosity, drug solubility, ionic-liquid viscosity, and electrolyte conductivity. KolTa ranks first or second in mean area under the learning curve (AUL
    
[^77]: 从水文定律到流域模型：一个基于过程的模型在三大洲上达到与深度学习相当的精度

    From hydrological laws to catchment models: A process-based model matches deep-learning accuracy on three continents

    [https://arxiv.org/abs/2611.06382](https://arxiv.org/abs/2611.06382)

    该研究基于守恒、耗散和聚合三条水文定律构建简约流域模型CaRT，并通过AI代理辅助的假设驱动迭代细化得到CaRT+模型，在不使用日尺度神经网络的情况下于三大洲达到与深度学习相当的精度，证明物理定律约束的低维模型可以匹敌神经网络。

    

    Dooge曾提出水文在流域尺度上是否存在定律的问题。我们则要问：一个在物理定律约束下构建的低维系统，能否达到神经网络在相同输入下超越基于过程的模型的精度。三条定律约束着一个集总式流域模型：守恒定律使其成为一个由通量连接的储量网络；耗散定律确定了每个通量的符号；聚合定律决定了未解析的组成部分如何通过阈值分布上的有效定律和对强迫的记忆进入通量。这些定律允许多种细化方式。在此框架下的一个简约模型CaRT（每个流域11个参数），在AI代理辅助的循环中被逐步细化。每次细化都通过文献中的一个假设来解决观察到的模型缺陷。最终得到的CaRT+模型具有十四个状态变量、每个流域35个参数，且不包含任何日尺度神经网络。大多数细化源于对阈值的聚合或来自新的状态变量。

    arXiv:2611.06382v1 Announce Type: cross  Abstract: Dooge asked whether hydrology has laws at the catchment scale. We ask whether a low-dimensional system built under physical laws can match the accuracy of neural networks that outperform process-based models on the same inputs. Three laws constrain a lumped catchment model. Conservation makes it a flux-connected storage network, dissipation fixes the sign of every flux and aggregation determines how the unresolved composition enters the fluxes, through effective laws over threshold distributions and a memory of the forcing. These laws admit four kinds of refinement. A parsimonious model under them, CaRT (11 parameters per catchment), is refined in a loop assisted by AI agents. Each refinement addresses an observed deficit through a hypothesis from the literature. The result, CaRT+, has fourteen states, 35 parameters per catchment and no daily neural network. Most refinements follow from aggregation over thresholds or from variables tha
    
[^78]: 放弃以获得更多：主导测试时计算分配的是先验而非策略

    Give Up to Get More: Priors, Not Policies, Govern Test-Time Compute Allocation

    [https://arxiv.org/abs/2611.06372](https://arxiv.org/abs/2611.06372)

    该研究揭示了测试时计算的最优静态分配呈倒U形——应将尝试次数集中在中等难度而非最难的输入上，并指出真正的收益来源于知道哪些输入可解的先验信息，而非分配策略本身。

    

    当多个输入共享大语言模型（LLM）的重复尝试预算时，一种常见的规则是给较难的输入分配更多尝试次数。在公开的数学、代码和形式化证明样本上，已发表的最好难度比例规则在模型大多能解决的基准上可以获得一半或更多的oracle增益，但在困难工作负载上最多只能获得13%的增益，即使给定真实的通过率也是如此。由于多一次尝试的价值在中等难度处达到峰值，最优的静态分配呈倒U形，对在当前预算下几乎无望的输入只分配极少的资源。在一项对221篇LLM计算分配论文的严格调查中，至少有13篇论文给最难的输入分配最多的尝试次数；它们的较简单设置恰好处于这种策略接近最优的预算范围内，因此其代价会在更困难的工作负载上显现。要采取正确行动需要知道哪些输入是可解的，因此是信息而非策略本身驱动了增益。借助验证器以及来自相关模型在相同输入上历史运行记录的先验，Thompson（采样方法）……

    arXiv:2611.06372v1 Announce Type: new  Abstract: When many inputs share a budget of repeated attempts by a large language model (LLM), a common rule gives harder inputs more. On public math, code and formal-proof samples, the best published difficulty-proportional rule captures half or more of the oracle gain on benchmarks the model mostly solves, but at most 13% on hard workloads, even given true pass rates. Because the value of one more attempt peaks at intermediate difficulty, the optimal static allocation is inverted-U and gives little to inputs that are hopeless at the current budget. In a strict survey of 221 LLM allocation papers, at least 13 give the hardest inputs the most attempts; their easier settings sit at budgets where this is near-optimal, so the cost appears on harder workloads. Acting on it requires knowing which inputs are solvable, so information, not the policy, drives the gain. With a verifier and a prior from a related model's past runs on the same inputs, Thomps
    
[^79]: EEGDash语料库：面向脑电（EEG）基础模型的即用型公开训练语料库

    EEGDash Corpus: A Training-Ready Public Corpus for EEG Foundation Models

    [https://arxiv.org/abs/2611.06356](https://arxiv.org/abs/2611.06356)

    该论文发布了首个即用型EEG基础模型公开训练语料库eegdash-corpus-v0，汇集了77个可合法再分发的公开数据集（涵盖7,272名受试者的14,075小时数据），解决了临床档案无法再分发、公开数据分散且格式不一导致难以汇集的核心问题。

    

    脑电图（EEG）基础模型通常在数万小时的记录数据上进行预训练，且其准确率随数据量的增加而提升。然而，这类数据难以汇集：规模最大的数据来源是无法重新分发的临床档案库，而公开数据源则分散在多个档案库、文件格式和许可证之中。我们发布了eegdash-corpus-v0，这是一个包含77个公开EEG数据集、可直接用于训练的语料库，以受限访问的Hugging Face数据集形式（EEGDash/eegdash-corpus-v0）发布，并为每个文件提供了SHA-256校验值。REVE是目前已发表的规模最大的EEG基础模型，其列出了89个数据集作为预训练语料，但其中12个无法重新分发：四个临床档案库（占其总时长的82%）以及八个受禁止衍生条款许可证约束的数据集。本语料库收录了REVE采样率所允许的其余76个数据集，以及来自REVE发布说明的另一个数据集：共计77个数据集、来自7,272名受试者的14,075小时数据、500万个10秒样本，每个训练周期包含62.1亿个token。

    arXiv:2611.06356v1 Announce Type: new  Abstract: Foundation models for electroencephalography (EEG) are pretrained on tens of thousands of hours of recordings, and their accuracy grows with that amount. Such data is hard to assemble. The largest sources are clinical archives that cannot be redistributed, and the public sources are spread over several archives, file formats, and licences. We release eegdash-corpus-v0, a training-ready corpus of 77 public EEG datasets, as the gated Hugging Face dataset EEGDash/eegdash-corpus-v0 with the SHA-256 of every file. REVE, the largest published EEG foundation model, names 89 datasets as its pretraining corpus. Twelve cannot be redistributed: four clinical archives that hold 82% of its hours, and eight under no-derivatives licences. The corpus holds the 76 others that REVE's sampling rate allows and one more from REVE's release card: 77 datasets, 14,075 hours from 7,272 subjects, 5.0 million 10-second samples and 6.21 billion tokens per epoch. Ev
    
[^80]: 柔性强度至关重要：对霍克斯过程与神经时间点过程的全面重新评估

    Flexible Intensities Matter: A Comprehensive Re-evaluation of Hawkes Processes and Neural Temporal Point Processes

    [https://arxiv.org/abs/2611.06348](https://arxiv.org/abs/2611.06348)

    该研究通过对标准基准和大规模医疗数据的系统性重新评估发现，一旦为霍克斯过程配备灵活可学习的基线与核函数，其与神经时间点过程之间此前报道的性能差距基本消失，从而挑战了神经TPPs显著优于经典TPPs的结论。

    

    时间点过程（TPPs）是事件序列数据的标准框架，其应用涵盖地震学、社交媒体和电子健康记录等领域。最近有报道称神经TPPs大幅优于经典TPPs（如霍克斯过程），但这些比较是将灵活的神经模型与形式最受限制的霍克斯过程进行对比，且未涉及大规模医疗数据。在此，我们通过系统性评估重新审视了这一比较：在五个成熟的基准数据集以及两个分别源自英国生物银行（UK Biobank）疾病史和MIMIC-IV重症监护记录的大规模医疗评估任务上，在自回归下一事件预测设置下，对灵活性逐步提升的泊松过程和霍克斯过程与神经TPPs进行了对比。首先，一旦为霍克斯过程配备灵活可学习的基线和核函（摘要在此处截断）

    arXiv:2611.06348v1 Announce Type: new  Abstract: Temporal point processes (TPPs) are the canonical framework for event sequence data, with applications spanning seismology, social media, and electronic health records. Recently, neural TPPs have been reported to substantially outperform classical TPPs, such as Hawkes processes, but these comparisons pitted flexible neural models against Hawkes processes in their most restricted form, and did not consider large-scale medical data. Here, we revisit this comparison through a systematic evaluation of Poisson and Hawkes processes of increasing flexibility against neural TPPs, on five established benchmark datasets and on two large-scale medical evaluation tasks derived from UK Biobank disease histories and MIMIC-IV ICU records, under autoregressive next-event prediction. First, the previously reported gap between Hawkes processes and neural TPPs largely disappears once Hawkes processes are equipped with flexible, learnable baselines and kern
    
[^81]: CoRTeC：一种可由任意生成器解码的差分隐私合成表格数据发布方法

    CoRTeC: A Differentially Private Release for Synthetic Tabular Data, Decodable by Any Generator

    [https://arxiv.org/abs/2611.06342](https://arxiv.org/abs/2611.06342)

    CoRTeC通过一次性发布一组经过差分隐私保护的含噪统计计数文件，使任何人都能用最大熵解码器或语言模型等任意生成器、以零额外隐私成本反复生成可防成员推断攻击的合成表格数据。

    

    那些最能用于训练有价值模型的数据记录，例如医院的病历档案，往往恰恰是机构无法对外共享的数据。我们提出了CoRTeC，一种在差分隐私保护下发布此类记录合成版本的方法。差分隐私对合成数据可能泄露的关于任何个人的信息量提供了可证明的界限，尤其能够防范成员推断攻击。CoRTeC只花费一次隐私预算，用于生成一个专为训练分类器而设计的小型含噪计数文件。对于每组记录，该文件给出了该组的规模、其结果分布的均衡情况、每种结果下每一列的直方图，以及若干互不重叠的单元格中每一格的结果率。审核人员可以在文件离开机构之前逐一核查其中的每个数字。此后，任何人都可以将这些计数转换为合成数据行，并且可以根据需要任意多次进行，而不会产生额外的隐私成本。最大熵解码器可以在完全不使用语言模型的情况下完成这一转换，其他各类生成器也同样可以做到。

    arXiv:2611.06342v1 Announce Type: cross  Abstract: The records that could train the most useful models, such as a hospital's patient files, are often the ones an institution cannot share. We propose CoRTeC, a method that releases synthetic versions of such records under differential privacy. Differential privacy places a proven limit on what the synthetic data can reveal about any person, and in particular on membership inference. CoRTeC spends the privacy budget once, on a small file of noisy counts designed for training a classifier. For each group of records, the file gives the group's size, its balance of outcomes, a histogram of every column for each outcome and the outcome rate in each of several non-overlapping cells. A reviewer can check every number before the file leaves the institution. Anyone can then turn the counts into synthetic rows, as often as they like and at no further privacy cost. A maximum-entropy decoder can do this without any language model, and so can each of
    
[^82]: 学习在有偏梯度下调参：节气门上无模型控制的迭代反馈整定

    Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves

    [https://arxiv.org/abs/2611.06338](https://arxiv.org/abs/2611.06338)

    该论文研究了迭代反馈整定（IFT）在汽车电子节气门这类非线性对象上调参时的梯度偏差问题，发现静摩擦导致的两次实验间控制器状态不匹配会引入与第二次实验幅值成反比的误差，且IFT梯度仅在69%的低成本区域内为下降方向。

    

    迭代反馈整定（IFT）通过随机梯度下降学习固定结构控制器的参数。在每次迭代中，需在被控对象上进行两次闭环实验：第一次实验测量跟踪误差，第二次实验由该误差驱动，提供用于估计代价函数梯度的灵敏度信息。该估计仅对线性对象是无偏的。我们在汽车电子节气门上研究了有偏的情形——节气门的摩擦与静摩擦在不同阀门和不同运行之间各不相同，由智能比例（无模型）控制器控制，该控制器是一个增量式PI控制器。我们证明，两次实验开始时控制器状态的不匹配（静摩擦使其不可避免）会引入一个与第二次实验幅值成反比的误差。在十个阀门的数字孪生体上，IFT梯度仅在69%的低成本区域内是下降方向，摩擦……

    arXiv:2611.06338v1 Announce Type: cross  Abstract: Iterative feedback tuning (IFT) learns the parameters of a fixed-structure controller by stochastic gradient descent. At each iteration, two closed-loop experiments are run on the plant: the first one measures the tracking error, and the second one, driven by this error, provides the sensitivities from which the gradient of the cost is estimated. This estimate is unbiased for linear plants only. We study the biased case on automotive electronic throttle valves, whose friction and stiction vary between valves and runs, controlled by an intelligent proportional (model-free) controller, which is an incremental PI controller. We show that the mismatch of the controller state at the start of the two experiments, which stiction makes unavoidable, adds an error inversely proportional to the amplitude of the second experiment. On digital twins of ten valves, the IFT gradient is a descent direction in only 69\% of the low-cost region, friction 
    
[^83]: 大规模生产具有惊人副作用的训练数据

    Mass Producing Training Data with Surprising Side Effects

    [https://arxiv.org/abs/2611.06324](https://arxiv.org/abs/2611.06324)

    本文提出一种自动化流程，利用LLM评判器结合进化算法、强化学习和潜意识学习方法，大规模生产在训练域外引发语言模型意外行为的“具有惊人副作用的数据集”，最终生成了6,575个涵盖8种行为的此类数据集，为研究训练数据对模型的隐性影响提供了系统性方法。

    

    训练数据可能在训练领域之外对语言模型行为产生意想不到的副作用：不安全的代码数据可能导致模型广泛失调，过时的鸟类名称可能诱发模型表现出19世纪的人格。理解这种泛化现象之所以困难，是因为目前只存在几十个孤立的例子，且每个例子都是手工构建或偶然发现的。我们提出自动发现这类“具有惊人副作用的数据集”（DSSEs）。我们为DSSEs定义了三个目标：（1）能够引发目标行为，（2）读起来像普通的训练后数据，（3）让阅读这些数据的人类审计者感到惊讶。我们使用LLM评判器对每个目标进行量化，并通过进化算法、强化学习和潜意识学习方法搜索高分数据集。经过重新评分和使用保留指标过滤后，我们的流程生成了一系列包含6,575个DSSEs的数据集，涵盖8种行为，其中许多数据集的效果能够跨模型家族泛化。

    arXiv:2611.06324v1 Announce Type: cross  Abstract: Training data can have unexpected side effects on language model behavior outside the training domain: insecure code data can induce broad misalignment, and outdated bird names can induce a 19th-century persona. Understanding such generalization is hard because only dozens of isolated examples exist, each constructed by hand or discovered by accident. We propose automatically discovering such datasets with surprising side effects (DSSEs). We define three objectives for DSSEs: they should (1) elicit a target behavior, (2) read like ordinary post-training data, and (3) be surprising to human auditors reading the data. We operationalize each with LLM judges, and search for high-scoring datasets via evolution, reinforcement learning, and subliminal learning approaches. After re-scoring and filtering with held-out metrics, our pipeline produces a collection of 6,575 DSSEs across 8 behaviors, many of which generalize across model families an
    
[^84]: 分布到分布回归的共形预测

    Conformal Prediction for Distribution-to-Distribution Regression

    [https://arxiv.org/abs/2611.06320](https://arxiv.org/abs/2611.06320)

    本文针对分布到分布回归问题，提出了一种在正交基空间中运行的共形预测新框架，能够生成可解释的预测集，同时保持覆盖保证，克服了朴素应用分裂共形预测导致的预测集不可解释的缺陷。

    

    共形预测（CP）提供了一个模型无关的框架，用于构建具有覆盖保证的有限样本预测集，并已成为高风险环境中可靠决策的工具。最近的进展已将分裂共形预测（Split Conformal Prediction）扩展到日益复杂的学习问题，包括多标签、高维和函数输出。在这项工作中，我们研究了分布到分布回归的共形预测，这是许多应用领域中日益受到关注的问题设置。我们表明，直接朴素地应用分裂共形预测所产生的预测集虽然具有边际有效性，但不可解释，例如缺乏明确定义的体积或可操作的采样程序，实际价值有限。为了克服这一问题，我们提出了一个新的共形框架，该框架在正交基空间中运行，在保持覆盖保证的同时产生可解释的预测集。我们进一步开发了一种自适应变体（摘要在此处截断）。

    arXiv:2611.06320v1 Announce Type: cross  Abstract: Conformal Prediction (CP) provides a model-agnostic framework for constructing finite-sample prediction sets with coverage guarantees and has become a tool for reliable decision-making in high-risk settings. Recent advances have extended Split Conformal Prediction to increasingly complex learning problems, including multilabel, high-dimensional, and functional outputs. In this work, we study CP for Distribution-to-Distribution Regression, a setting of growing interest across many application domains. We show that a naive application of Split CP yields prediction sets that, while marginally valid, are non-interpretable, e.g., lacking a well-defined volume or a tractable sampling procedure, and of limited practical value. To overcome this, we propose a new conformal framework that operates in an orthogonal basis space, producing interpretable prediction sets while preserving coverage guarantees. We further develop an adaptive variant wit
    
[^85]: 风险受控的吞吐量预测何时才真正控制住风险？——一项面向低轨卫星链路的复现研究

    When Does Risk-Controlled Throughput Forecasting Actually Control Risk? A Reproduction Study on LEO Satellite Link

    [https://arxiv.org/abs/2611.06307](https://arxiv.org/abs/2611.06307)

    对低轨卫星链路上BG-CFQS风险受控吞吐量预测的复现研究发现，其风险保证虽然可复现但依赖校准集划分方式——在实际部署所面临的连续时间划分下大多失效，且无论何种划分都无法实现低于0.15的风险预算。

    

    风险预算化的吞吐量预测将低轨（LEO）卫星容量的点预测转化为一个保守上界，使其高估率保持在用户指定的预算水平之内，从而让准入控制能够安全地据此决策。我们在其作者所用的三条 Starlink 链路数据以及一个独立的第四个数据集上复现了当前最先进的方法 BG-CFQS，并报告了四项发现。第一，其风险保证是可复现的，但取决于校准集的划分方式：在随机划分下我们复现了所有已发表的数值，包括在全部3个数据集上均通过风险测试；然而在连续时间划分下——这正是实际部署终端所面对的情形——风险保证仅在3个数据集中的1个上成立。第二，无论采用何种划分方式，该方法都无法实现低于0.15的风险预算（即其候选分位数集合的下端）。在全部四个数据集上，预算设为0.05、0.10和0.15时实际达到的风险率完全相同，在最紧预算下超出目标1.4倍至3.7倍。第三，静态共形校准……（原文在此截断）

    arXiv:2611.06307v1 Announce Type: cross  Abstract: Risk-budgeted throughput forecasting converts a point forecast of LEO satellite capacity into a conservative bound whose overestimation rate is held at a user-specified budget, so that admission control can act on it safely. We reproduce the state of the art, BG-CFQS, on the three Starlink traces its authors used and on an independent fourth dataset, and report four findings. First, its risk guarantee is reproducible but conditional on the calibration split: under a random split we recover every published quantity, including risk-pass on 3 of 3 datasets, but under a contiguous temporal split, which is what a deployed terminal faces, it holds on 1 of 3. Second, independent of any split, the method cannot serve a risk budget below 0.15, the low end of its candidate quantile set. On all four datasets the achieved rate is identical at budgets of 0.05, 0.10 and 0.15, overshooting by 1.4x to 3.7x at the tightest. Third, static conformal cali
    
[^86]: 分割协议决定结果：重新评估加密流量分类

    The Split Protocol Decides the Result: Re-Evaluating Encrypted Traffic Classification

    [https://arxiv.org/abs/2611.06290](https://arxiv.org/abs/2611.06290)

    该论文揭示加密流量分类报告的高准确率主要由数据分割协议决定——同一模型在不同分割协议下宏F1可从0.992骤降至0.567，且关键划分轴取决于语料库的收集方式，并提出了一个无需训练模型即可识别正确划分轴的诊断方法。

    

    加密流量分类器通常被报告具有95-99%的准确率。我们表明，在两个公开语料库之一上，相同的模型和代码在文献标准的评估协议下得到宏F1 0.992，而在一种将来自同一数据包捕获的流量保留在分割同一侧的协议下仅得到0.567。除了数据划分方式外没有任何改变（八个随机种子上的差距为0.426，经Holm-Bonferroni校正后p = 0.008）。这种虚高并非随机分割本身的属性。在一个月的骨干网QUIC流量上，我们以每个协议八个种子评估了五种分割协议：随机、时间序和客户端不重叠分割在统计上与会话不重叠分割无法区分，而服务器不重叠分割会损失0.192的宏F1（d = 17.0，p = 0.008）。哪个划分轴重要取决于语料库的收集方式，而这从其标签空间中无法看出。我们提供了一个诊断方法，可以在训练任何模型之前识别出正确的划分轴。

    arXiv:2611.06290v1 Announce Type: cross  Abstract: Encrypted traffic classifiers are routinely reported at 95-99% accuracy. We show that on one of two public corpora, the same model and code score macro-F1 0.992 under the evaluation protocol standard in the literature and 0.567 under a protocol that keeps flows from one packet capture on one side of the split. Nothing changes but the partition (gap 0.426 over eight seeds, p = 0.008 after Holm-Bonferroni correction).   The inflation is not a property of random splitting as such. On a month of backbone QUIC traffic, we evaluate five split protocols at eight seeds each: random, temporal and client-disjoint splitting are statistically indistinguishable from session-disjoint, while server-disjoint splitting costs 0.192 macro-F1 (d = 17.0, p = 0.008). Which axis matters depends on how the corpus was collected, in a way invisible from its label space.   We give a diagnostic that identifies the right axis before any model is trained: single-id
    
[^87]: 面向一步式MIMO-OFDM信道估计的一致性学习与蒸馏

    Consistency Learning and Distillation for One-Step MIMO-OFDM Channel Estimation

    [https://arxiv.org/abs/2611.06271](https://arxiv.org/abs/2611.06271)

    本文提出观测锚定一致性学习与一致性中期训练（CMT）蒸馏方法，实现MIMO-OFDM系统的一步式信道估计，紧凑的CMT学生模型在参数减少74.5%的情况下将符号错误率降低9.6-20.2%。

    

    迭代式神经信道估计器能够在导频污染条件下改善信号恢复效果，但反复的网络推理会增加计算成本。本文针对多输入多输出正交频分复用（MIMO-OFDM）系统中的一步式信道估计，提出了观测锚定一致性估计器和一致性中期训练（Consistency Mid-Training, CMT）估计器。两种估计器均在时延-波束域中对基于导频的最小二乘信道估计进行精炼。一致性学习将带噪到干净路径上的端点预测相互耦合。CMT随后对一个紧凑的、经一致性训练的学生模型进行精炼，使其能够从中间轨迹状态直接预测流匹配教师模型的端点结果。在所评估的信噪比范围内，紧凑的CMT学生模型相比一步式流匹配基线，将未编码符号错误率（SER）降低了9.6-20.2%（对数SER降低0.44-0.98 dB），同时参数数量减少74.5%，网络乘法运算减少73.0%。

    arXiv:2611.06271v1 Announce Type: new  Abstract: Iterative neural channel estimators can improve recovery under pilot contamination, but repeated network evaluations increase inference cost. This paper develops observation-anchored consistency and Consistency Mid-Training (CMT) estimators for one-step channel estimation in multiple-input multiple-output orthogonal frequency-division multiplexing (MIMO-OFDM) systems. Both estimators refine pilot-derived least-squares channel estimates in the delay-beam domain. Consistency learning couples endpoint predictions along noisy-to-clean paths. CMT then refines a compact consistency-trained student to predict a flow-matching teacher's endpoint from intermediate trajectory states. Across the evaluated SNR range, the compact CMT student reduces uncoded symbol error rate (SER) by 9.6-20.2% relative to the one-step flow-matching baseline (0.44-0.98 dB in logarithmic SER reduction), while using 74.5% fewer parameters and 73.0% fewer network multiply
    
[^88]: 基于图符号的异质图伪黎曼流形学习

    Learning Pseudo-Riemannian Manifolds for Heterophilic Graphs via Graph Signature

    [https://arxiv.org/abs/2611.06265](https://arxiv.org/abs/2611.06265)

    该论文提出图符号指数（s_G）来量化正定几何先验与异质图之间的根本错配，并据此提出伪黎曼注意力网络（PRAT），通过带符号的平方伪距离注意力机制在伪黎曼流形上学习异质图表示。

    

    arXiv:2611.06265v1 公告类型：cross（交叉发布）。摘要：若干广泛使用的图神经网络家族都带有正定几何先验。当图卷积被解读为热扩散时需要半正定算子，双曲模型将特征放置在正定黎曼流形上，而点积和GAT风格的注意力分数则使用正定内积。然而在异质图（heterophilic graphs）上，边往往表示不相似性或结构上的排斥，这种先验便产生了错配：任何正定形式都无法为标签所区分的方向赋予负的平方距离。为了度量这种错配，我们引入了图符号指数，它是仅由标签拓扑构建的类相容矩阵的惯性，并证明了在(0,1)区间内，任何边同质指数或节点同质指数的取值都无法排除负特征值的存在。随后，我们提出了伪黎曼注意力网络（PRAT），其注意力对数是一个带符号的平方伪距离……（摘要在此处不完整）

    arXiv:2611.06265v1 Announce Type: cross  Abstract: Several widely used families of Graph Neural Networks carry a positive-definite geometric prior. Graph convolution read as heat diffusion needs a positive semi-definite operator, hyperbolic models place features on positive-definite Riemannian manifolds, and dot-product and GAT-style attention scores use positive-definite inner products. On heterophilic graphs, where edges often signify dissimilarity or structural repulsion, that prior is a mismatch: no positive-definite form can assign a negative squared distance to a direction the labels separate. To measure the mismatch, we introduce the Graph Signature Index ($s_{\mathcal{G}}$), the inertia of a class-compatibility matrix built from label topology alone, and we prove that no value of the edge or node homophily index in $(0,1)$ rules out a negative eigenvalue. We then propose the Pseudo-Riemannian Attention Network (PRAT), whose attention logit is a signed squared pseudo-distance un
    
[^89]: 本地开源权重大语言模型智能体在漏洞复现任务上的分析

    Analysis of Local Open-Weight LLM Agents on Vulnerability-Reproduction Tasks

    [https://arxiv.org/abs/2611.06257](https://arxiv.org/abs/2611.06257)

    本研究系统评测了十二种本地开源权重LLM配置在CyberGym漏洞复现任务上的表现，发现即使小至4B的模型也能解决大部分较简单的漏洞复现任务，表明漏洞复现是本地小模型可胜任的自动化分析任务。

    

    能装入单张消费级GPU的开源权重语言模型在越来越多的任务上展现出能力，而其重要性在于可以在本地运行、无需将代码发送到托管服务。漏洞复现——即把漏洞描述转化为能使含漏洞构建版本崩溃的输入——是这类模型的天然目标，因为它是一项更接近测试生成而非代码理解的自动化分析任务，并且针对的是必须执行而非阅读的构建版本。为了评估当前模型能否完成该任务，我们在十个CyberGym漏洞复现任务上评测了十二种本地模型配置，涵盖三代模型、两种架构和三种量化级别，每个任务在统一测试框架（harness）下运行三次，并发布了完整的实验工件。结果表明，所有配置——甚至小至4B的模型——都能解决六个较简单任务中的大多数，而各配置之间仅在剩余的较难任务上才显示出差异。

    arXiv:2611.06257v1 Announce Type: cross  Abstract: Open-weight language models that fit on a single consumer GPU are becoming capable on a growing range of tasks, and they matter because they can run locally without sending the code to a hosted service. Vulnerability reproduction, turning a vulnerability description into an input that crashes the vulnerable build, is a natural target for them, since it is an automated analysis task closer to test generation than to code comprehension and it runs against a build that must be executed rather than read. In order to assess whether current models can perform it, we evaluate twelve configurations of local models, spanning three generations, two architectures and three quantization levels, on ten CyberGym vulnerability-reproduction tasks, three runs each under one harness, and release the full artefact. We show that every configuration, down to the 4B model, solves most of the six easier tasks, and configurations separate only on the remainin
    
[^90]: 面向少步生成模型奖励对齐的学习式噪声优化

    Learning Noise Optimization for Reward Alignment in Few-Step Generative Models

    [https://arxiv.org/abs/2611.06255](https://arxiv.org/abs/2611.06255)

    提出LeNO方法，将奖励引导的初始噪声优化摊销为学习到的残差传输，在冻结预训练生成器的前提下实现少步生成模型的高效奖励对齐，并通过KL散度上界提供理论正则化。

    

    扩散模型和流模型的奖励对齐通常是在随机轨迹上进行形式化的，这使得这类方法不太适用于少步确定性生成器。初始噪声优化提供了一种无需轨迹的替代方案，但现有的无梯度方法需要在推理阶段进行代价高昂的逐样本搜索。我们提出了LeNO（Learning Noise Optimization，学习式噪声优化），它将奖励引导的初始噪声优化摊销为一个学习到的残差传输，同时保持预训练生成器完全冻结。LeNO利用局部GRPO和标量奖励，学习一个以锚点为条件的噪声修正分布，从而消除了迭代式的测试时优化。对于经过范数保持投影的高斯初始噪声，我们推导出了修正后噪声分布与原始先验之间KL散度的可求解上界，为该方法提供了一个有理论依据的正则化目标。我们在少步文本到图像和蛋白质生成任务上对LeNO进行了评估。

    arXiv:2611.06255v1 Announce Type: new  Abstract: Reward alignment for diffusion- and flow-based generative models is often formulated over stochastic trajectories, making it less suitable for few-step deterministic generators. Initial noise optimization provides a trajectory-free alternative, but existing gradient-free methods require costly per-sample search at inference. We introduce LeNO (Learning Noise Optimization), which amortizes reward-guided initial noise optimization into a learned residual transport while keeping the pretrained generator frozen. LeNO learns an anchor-conditioned distribution over noise corrections using Local GRPO and scalar rewards, eliminating iterative test-time optimization. For Gaussian initial noise with norm-preserving projection, we derive a tractable upper bound on the KL divergence between the corrected noise distribution and the original prior, providing a principled regularization objective. We evaluate LeNO on few-step text-to-image and protein 
    
[^91]: 评分标准课程强化学习：在不可验证领域利用生成-验证差距

    Rubric Curriculum RL: Exploiting the Generation-Verification Gap in Non-Verifiable Domains

    [https://arxiv.org/abs/2611.06244](https://arxiv.org/abs/2611.06244)

    RcRL 通过将对采样结果的成对偏好与评分标准课程机制相结合，在无需新数据、人工标注或更强教师模型的情况下，利用生成-验证差距在不可验证的开放式领域实现持续的性能提升并有效避免奖励作弊。

    

    带可验证奖励的强化学习（RLVR）在数学和代码生成领域取得了显著成果。然而，将这些成果扩展到开放式领域仍然充满挑战：真值验证不可用，人工标注成本高昂，且学习到的奖励模型容易遭受奖励作弊。生成-验证差距——即评判一个高质量输出比生成它更容易——为改进提供了一条潜在路径。然而，基于评分标准的评判器往往不精确且固定不变，导致策略容易对其过拟合或进行奖励作弊。我们提出了评分标准课程强化学习，它在无需新数据、人工标注或更强教师模型的情况下利用这一差距。我们将对采样结果的成对偏好与评分标准上的课程机制相结合，从而产生比绝对评分更具判别性的信号以及一个非平稳的优化目标。基线方法通常在几百步内就陷入停滞，而 RcRL 能够保持输出熵并持续提升（摘要在此处截断）。

    arXiv:2611.06244v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has led to significant gains in math and code generation. Extending these gains to open-ended domains remains challenging: ground-truth verification is unavailable, human annotation is expensive, and learned reward models are prone to reward hacking. The generation-verification gap, where judging good outputs is easier than producing them, offers a potential path to improvement. However, rubric-based judges are often imprecise and stationary, so policies can overfit or reward hack them. We introduce Rubric Curriculum RL (RcRL), which exploits this gap without new data, human annotations, or stronger teacher models. We combine pairwise preferences over rollouts with a curriculum over rubric criteria, yielding a more discriminative signal than absolute scoring and a non-stationary objective. Whereas baselines plateau within a few hundred steps, RcRL preserves output entropy and keeps 
    
[^92]: Zuse：面向欧洲语言的决策模型

    Zuse: A Decision Model for European Languages

    [https://arxiv.org/abs/2611.06238](https://arxiv.org/abs/2611.06238)

    Zuse是首批面向27种欧洲语言的决策模型系列（4B/9B/27B），通过在冻结的开源基座上训练低秩适配器和指针头，实现了87.5%的准确率和极低的校准误差，并发现校准偏差主要由任务而非语言决定。

    

    决策模型读取文档并为固定选项集合中的每个选项返回概率，而非生成文本。2026年出现了若干此类模型，但没有一个发表过英语以外的结果。我们提出Zuse——三个决策模型（4B、9B和27B），它们在冻结的开源权重基座上训练低秩适配器和指针头，面向27种欧洲语言，并用它们研究这种设计在不同语言和任务上的表现。在一个由11个未参与训练的数据源组成的广泛欧洲评测套件上（涵盖9个任务组和27种语言，其中开发集也用于指导模型选择），27B模型达到87.5%的准确率，总体期望校准误差为0.016，按数据源加权时为0.035。对这些单种子结果的事后分析得出三点发现。其一，校准偏差是按任务而非按语言结构化的：在26种语言的平行阅读和主题类题目上，置信度差距的变化幅度大约与采样……（摘要原文在此截断）

    arXiv:2611.06238v1 Announce Type: new  Abstract: Decision models read a document and return a probability for each of a fixed set of options instead of generating text. Several appeared in 2026, but none has published results outside English. We describe Zuse, three decision models (4B, 9B and 27B) that train a low-rank adapter and a pointer head on frozen open-weight bases, targeting 27 European languages, and use them to study how this design behaves across languages and tasks. On a broad European suite of 11 sources excluded from training, covering 9 task groups and 27 languages, a development set that also guided model selection, the 27B model reaches 87.5% accuracy with a pooled expected calibration error of 0.016, and 0.035 when weighted per source. Post-hoc analyses of these single-seed results give three findings. First, miscalibration is structured by task rather than language: on parallel reading and topic items in 26 languages the confidence gap varies about as much as sampl
    
[^93]: 分离即是一种 Riesz 能量：ICA 需要多少个环境？

    Separation Is a Riesz Energy: How Many Environments Does ICA Need?

    [https://arxiv.org/abs/2611.06228](https://arxiv.org/abs/2611.06228)

    本文证明在高斯 ICA 中分离某个成分的代价由其在各环境方差签名所对应的 Riesz 2-能量刻画，并据此给出样本复杂度的维度定律：环境数 K=2、3 及 K≥4 时分别需要 d²/n、d·log d/n 与 d/n 量级的样本。

    

    当所有独立成分都是高斯分布时，独立成分还能被恢复吗？答案是肯定的，只要它们的方差在不同环境之间发生变化；但随着成分数量 $d$ 的增长，需要多少个环境，以及多少个样本？本文针对 $K$ 个协方差矩阵的联合对角化问题回答了这两个问题。每个成分都有一个“特征签名”，即它在各环境中的方差。分离第 $j$ 个成分的代价等于该签名的 Riesz 2-能量 $E_j$，即在其余成分上成对费希尔信息倒数之和。对于高斯源，在任何 $n$ 和 $d$ 下，$O(d)$ 复杂度内没有任何估计器能超越 $cE_j/(n+E_j)$；而在超过某个明确的样本量阈值之后，谱估计器在常数因子内即可达到 $E_j/n$ 的速率，且该结论适用于任意混合方式、无需预知特征签名（仅需给出成对信息的下界）。由于特征签名生活在 $K-1$ 维空间中，其速率遵循一个维度定律：$K=2$ 时为 $d^2/n$，$K=3$ 时为 $d\log d/n$，而 $K\ge4$（环境分布良好）时为 $d/n$。起决定作用的是维度……（原文摘要至此截断）

    arXiv:2611.06228v1 Announce Type: new  Abstract: Can independent components be recovered when all of them are Gaussian? Yes, if their variances change across environments; but how many environments, and how many samples as the number of components $d$ grows? We answer both for the joint diagonalization of $K$ covariances. Each component has a signature, its variances across the environments. Separating component $j$ costs the Riesz 2-energy $E_j$ of this signature, a sum, over the other components, of inverse pairwise Fisher informations. For Gaussian sources no estimator in $O(d)$ beats $cE_j/(n+E_j)$ at any $n$ and $d$, and above an explicit sample size a spectral estimator attains $E_j/n$ up to a constant, for every mixing and without the signatures (given a lower bound on the pairwise information). Since signatures live in $K-1$ dimensions, the rate obeys a dimension law: $d^2/n$, $d\log d/n$ and $d/n$ for $K=2$, $3$ and $K\ge4$ well-spread environments. What counts is the dimensio
    
[^94]: 具有完全预见性的一致次模最大化

    Consistent Submodular Maximization with Full Foresight

    [https://arxiv.org/abs/2611.06223](https://arxiv.org/abs/2611.06223)

    该论文在“完全预知未来且计算不受限”的理想设定下，刻画了单调次模最大化中解稳定性（一致性）的代价，证明了确定性算法与随机算法的最优近似阈值分别为 3/4 和约 0.7942，并以多项于 1/ε 的变更代价逼近该阈值，同时对覆盖函数给出了匹配的不可能性证明。

    

    arXiv:2611.06223v1 公告类型：cross 摘要：具有有界变更代价的在线算法面临三个相互交织的障碍：对未来的不确定性、计算能力的限制以及有限的变更预算。我们提出了一个能将“一致性”与其他两个因素区分开来的根本性问题：当整个未来完全已知且计算能力不受限制时，保持解的稳定性需要付出什么代价？我们针对单调次模最大化问题回答了这一问题，其中算法必须维护一个由至多 $k$ 个已到达元素组成的解，且每步只允许常数次变更。我们确定了最优近似阈值：确定性算法为 $3/4$，随机算法为 $\xi^\star \approx 0.7942$。对于任意 $\varepsilon>0$，我们都能以最坏情况下与 $1/\varepsilon$ 呈多项式关系（且与 $k$ 和流的长度无关）的变更代价达到距这些阈值 $\varepsilon$ 以内的效果。我们甚至对覆盖函数证明了匹配的不可能性结果。我们通过最优地平衡……（原文在此截断）

    arXiv:2611.06223v1 Announce Type: cross  Abstract: Online algorithms with bounded recourse face three intertwined obstacles: uncertainty about the future, computational limits, and a limited budget for changes. We ask a fundamental question that separates consistency from the other two: what does solution stability cost when the entire future is known and computation is unlimited?   We answer this question for monotone submodular maximization, where an algorithm must maintain a solution of at most $k$ arrived elements, with a constant number of changes per step. We determine the optimal approximation thresholds: $3/4$ for deterministic algorithms and $\xi^\star \approx 0.7942$ for randomized algorithms. For every $\varepsilon>0$, we get within $\varepsilon$ of these thresholds with worst-case recourse polynomial in $1/\varepsilon$, independent of $k$ and the stream length. We prove matching impossibility results even for coverage functions.   We attain the sharp bounds by optimally bal
    
[^95]: 灾难性性能的定义

    A Definition of Catastrophic Performance

    [https://arxiv.org/abs/2611.06219](https://arxiv.org/abs/2611.06219)

    该论文提出了“灾难性性能”的正式定义，通过基线值和正交值两个参考值来区分次优结果与灾难性结果，从而为严格分析人工智能目标错位带来的灾难性风险提供了理论基础。

    

    人工智能的一个核心挑战是制定与人类偏好相一致的目标。虽然现有的理论结果阐明了目标设定错误如何影响性能，但它们通常没有区分仅仅是次优的结果与灾难性的结果。为了能够对人工智能目标错位带来的灾难性风险进行严格分析，我们引入了灾难性性能的正式定义。我们的定义基于两个参考值：第一个是基线值，表示如果没有部署任何智能体时所获得的价值；第二个是正交值，表示一条在我们所描述意义上与人类偏好无关的轨迹的价值。正交值通常远差于基线值。我们通过一些例子来阐释这些参考值，在这些例子中，过去的人类活动已经改变了环境状态并使其有利于人类。从由此产生的状态出发，部署一个以加速进一步……

    arXiv:2611.06219v1 Announce Type: new  Abstract: A central challenge in AI is specifying objectives that align with human preferences. While existing theoretical results elucidate how misspecification impacts performance, they typically do not distinguish between merely suboptimal and catastrophic outcomes. To enable rigorous analysis of catastrophic risk from AI misalignment, we introduce a formal definition of catastrophic performance. Our definition is based on two reference values. The first, baseline value, captures the value attained if no agent is deployed. The second, orthogonal value, captures the value of a trajectory that is unrelated to human preferences, in a sense we describe. Orthogonal value is typically far worse than baseline value. We interpret these reference values through examples in which past human activity has altered the state of the environment to human benefit. Starting at the resulting state, an AI agent is deployed with the goal of accelerating further pro
    
[^96]: 先循环，后飞跃：循环深度在数据受限预训练中能带来什么与不能带来什么

    Loop Before You Leap: What Recurrent Depth Can and Cannot Buy in Data-Limited Pretraining

    [https://arxiv.org/abs/2611.06206](https://arxiv.org/abs/2611.06206)

    该研究发现在1亿词的数据受限预训练中，循环深度（12层堆叠重复四次）在相同推理计算量下的纯文本任务中比标准48层Transformer高出5.4个BLiMP百分点，但无法缓解多模态训练导致的约3.8个百分点的语言能力下降。

    

    在BabyLM挑战赛中，多模态模型的表现往往低于纯文本基线（Klerings等人，2024；Takmaz等人，2025）。我们测试了额外的循环计算能否帮助缩小这一差距。循环机制通过重复使用层来增加计算量，而无需增加参数或训练数据。在单一实现下，我们在1亿词的预算下比较标准Transformer与循环Transformer，每个主要条件使用三个随机种子。每张图像由冻结的DINOv2编码器输出的一个特征向量表示，并作为前缀token使用。图像描述词计入预算，因此多模态训练也会改变语言数据本身。在一项预注册的纯文本比较中，重复四次的12层堆叠在推理计算量相同的情况下，其BLiMP得分比标准的48层Transformer高出5.4个百分点。然而，多模态训练使两种架构的BLiMP准确率均下降约3.8个百分点，且循环机制并未带来明显的缓解作用。

    arXiv:2611.06206v1 Announce Type: new  Abstract: Multimodal models have often scored below text-only baselines in the BabyLM Challenge (Klerings et al., 2024; Takmaz et al., 2025). We test whether additional recurrent computation helps close this gap. Recurrence increases computation by reusing layers without adding parameters or training data. Using a single implementation, we compare standard and looped Transformers under a 100M-word budget, with three seeds per main condition. Each image is represented by one feature vector from a frozen DINOv2 encoder, used as a prefix token. Caption words count toward the budget, so multimodal training also changes the language data. In a pre-registered text-only comparison, a 12-layer stack repeated four times exceeds a standard 48-layer Transformer by 5.4 BLiMP percentage points at matched inference compute. However, multimodal training lowers BLiMP accuracy by about 3.8 points in both architectures, with no clear mitigation from recurrence. An 
    
[^97]: 面向无目标故障数据的数据稀缺资产的跨资产知识迁移早期故障检测方法

    Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data

    [https://arxiv.org/abs/2611.06194](https://arxiv.org/abs/2611.06194)

    本文提出一个零样本域适应框架，在目标资产完全没有故障数据的情况下，仅利用其正常数据即可通过EWMA自动标注源资产数据并借助改进的域对抗神经网络实现跨资产早期故障检测。

    

    工业机队的可靠性管理依赖于早期故障预警，然而相似资产之间的生命周期数据高度不平衡。少数资产积累了从运行到失效的完整历史数据，而新安装的或很少发生故障的资产仅能提供正常工况下的数据。本文研究的问题是：当目标资产上尚未观测到任何故障数据时，如何将诊断知识从数据丰富的源资产迁移到数据稀缺的目标资产。我们将跨资产早期故障检测形式化为一个零样本域适应问题，其中“零样本”指的是目标资产故障样本的缺失，并提出一个仅需目标资产正常数据的框架。该框架包含：(1) 一个统计过程控制步骤，利用指数加权移动平均（EWMA）控制图对源资产的运行至失效数据进行自动标注，将人工标注工作简化为对被标记片段的验证；(2) 一个改进的域对抗神经网络（ADAN……

    arXiv:2611.06194v1 Announce Type: new  Abstract: Reliability management of industrial fleets depends on early fault warnings, yet lifecycle data across similar assets are highly imbalanced. A few assets accumulate run-to-failure histories, while newly installed or rarely failing assets provide only normal-condition data. This paper asks how diagnostic knowledge can be transferred from a data-rich source asset to a data-scarce target asset when no fault data have been observed on the target. We formulate cross-asset early fault detection as a zero-shot domain adaptation problem, in which zero-shot refers to the absence of target fault samples, and propose a framework that requires only normal data from the target. The framework comprises (1) a statistical process control step that labels the source asset run-to-failure data with an exponentially weighted moving average chart, reducing manual annotation to verifying flagged segments, (2) an Adapted Domain-Adversarial Neural Network (ADAN
    
[^98]: GPUMAP：GPU加速的渐进式均匀流形逼近与投影

    GPUMAP: GPU-Accelerated Progressive Uniform Manifold Approximation and Projection

    [https://arxiv.org/abs/2611.06191](https://arxiv.org/abs/2611.06191)

    GPUMAP通过将UMAP重构为渐进式流水线并利用GPU并行加速，实现了在普通GPU硬件上对百万级数据集的交互式渐进降维分析。

    

    降维（DR）技术是高维数据分析的基础，但其巨大的计算成本限制了它们在交互式场景中的使用。为了解决这一局限，我们提出了GPU加速的渐进式均匀流形逼近与投影（GPUMAP），这是一种专为交互式使用而设计的高效渐进式降维算法。我们识别出UMAP中的三个原子操作，并将它们重构为渐进式流水线。作为这一改造的一部分，我们对UMAP进行了修改，包括操作队列、实例级局部连接和调度策略。我们进一步利用GPU并行性加速计算，使得在普通GPU硬件上能够对百万级数据集进行交互式分析。最后，我们利用我们的实现系统地研究了渐进式降维的行为，考察了投影随时间相对于真实投影的演化过程。

    arXiv:2611.06191v1 Announce Type: new  Abstract: Dimensionality reduction (DR) techniques are fundamental to the analysis of high-dimensional data, yet their substantial computational cost limits their use in interactive settings. To address this limitation, we present GPU-accelerated Progressive Uniform Manifold Approximation and Projection (GPUMAP), an efficient progressive DR algorithm designed for interactive use. We identify three atomic operations within UMAP and restructure them into a progressive pipeline. As part of this adaptation, we introduce modifications to UMAP, including operation queues, instance-wise local join, and scheduling strategies. We further accelerate computation by leveraging GPU parallelism, enabling interactive analysis of million-scale datasets on commodity GPU hardware. Finally, we employ our implementation to systematically study the behavior of progressive DR, investigating how projections evolve over time relative to the ground-truth projection and ho
    
[^99]: Chiron：由大语言模型智能赋能的面向AI工作负载的缓存管理

    Chiron: Cache Management Empowered by LLM Intelligence for AI Workloads

    [https://arxiv.org/abs/2611.06179](https://arxiv.org/abs/2611.06179)

    Chiron提出了一种由LLM赋能的咨询式切换控制缓存管理方法：LLM异步诊断AI工作负载的访问模式并推荐缓存策略，由确定性框架把关建议并在本地执行专家设计的策略，从而在保持用户透明性的同时自适应地优化缓存性能。

    

    AI集群将数据保存在远程对象存储中，并依赖本地缓存来避免GPU因等待数据而空闲。只有当缓存的预取与淘汰策略与访问模式相匹配时，缓存才能发挥良好性能，然而这些模式在不同AI工作负载之间各不相同，并且在同一工作流的不同阶段之间也会发生变化。现有方法要么需要修改代码，从而损害了用户透明性，要么采用低端启发式方法，无法跟随这些变化。大语言模型（LLM）无需用户提示即可推断访问行为，但直接由LLM控制缓存操作存在推理延迟长、token成本高和决策质量差的问题。我们提出了Chiron，一个基于咨询式切换控制的缓存系统。LLM异步诊断访问模式并推荐策略；一个确定性框架对该建议进行把关，在本地执行专家设计的策略，并返回缓存反馈以供审查。我们在JuiceFS之上实现了Chiron并对其进行了评估。

    arXiv:2611.06179v1 Announce Type: cross  Abstract: AI clusters keep data in remote object stores and rely on local caches to stop GPUs from waiting for data. A cache performs well only when its prefetching and eviction policies match the access pattern, yet these patterns differ across AI workloads and change between phases of one workflow. Existing methods either need code changes, which compromises user transparency, or use low-end heuristics, which cannot follow such changes. Large language models (LLMs) can infer access behavior without user hints, but direct LLM control of cache actions suffers from long reasoning delay, high token cost, and poor decision quality. We present Chiron, a cache system based on advisory switching control. An LLM asynchronously diagnoses the access pattern and recommends a policy. A deterministic harness gates this advice, executes expert-designed policies locally, and returns cache feedback for review. We implement Chiron atop JuiceFS and evaluate it o
    
[^100]: 基于可解释人工智能与数字证据链的网络入侵取证检测系统

    Network Intrusion Forensic Detection System with Explainable Artificial Intelligence and Digital Chain of Custody

    [https://arxiv.org/abs/2611.06173](https://arxiv.org/abs/2611.06173)

    该论文提出MODEXRE取证流水线，将经过验证的SHAP可解释XGBoost入侵检测模型扩展为“实验室模式+正式模式”双模式架构，在SHA-256哈希链式数字证据链保障下，对经OCSF归一化的多源异构网络证据执行可验证且可解释的取证推理。

    

    摘要：将一个在实验室环境中经过验证的可解释入侵检测模型部署到实际取证工作流中，需要满足基准测试无法检验的保证：即经过认证的模型正是应用于具体案件的那个确切产物、异构证据能够被可靠地分类，以及每一步操作均可验证。我们提出了MODEXRE，一个取证流水线，它将此前已在合成数据上验证过的、带有SHAP可解释性的XGBoost分类器（arXiv:2607.00763）扩展为双模式架构：实验室模式，模型在公开数据和合成数据上进行训练与认证；正式模式，一个被冻结的、经哈希验证的模型在SHA-256哈希链式数字证据链下对案件证据执行推理。Suricata告警、CEF防火墙日志和数据包捕获均通过OCSF进行归一化处理。四个设计决策塑造了该架构，每一个决策都源于一次实际观察到的失败：例如一个导致Kitsune性能被虚高评估的时间戳泄露问题……（原文摘要在此处被截断）

    arXiv:2611.06173v1 Announce Type: cross  Abstract: Deploying an explainable intrusion detection model from a validated laboratory setting into a forensic workflow requires guarantees that benchmarks do not test: that the certified model is the exact artifact applied to a case, that heterogeneous evidence can be classified reliably, and that every step is verifiable. We present MODEXRE, a forensic pipeline that extends a previously validated synthetic-data-trained XGBoost classifier with SHAP explainability (arXiv:2607.00763) into a dual-mode architecture: a Laboratory mode, where models are trained and certified on public and synthetic data, and a Formal mode, where a frozen, hash-verified model performs inference on case evidence under a SHA-256 hash-chained chain of custody. Suricata alerts, CEF firewall logs and packet captures are normalized through OCSF. Four design decisions, each triggered by an observed failure, shaped the architecture: a timestamp leak that inflated Kitsune pe
    
[^101]: AssameseLM-E2B：一个面向消费级设备高效部署的开源权重阿萨姆语语言模型

    AssameseLM-E2B: An Open-Weight Assamese Language Model for Efficient Consumer-Grade Deployment

    [https://arxiv.org/abs/2611.06170](https://arxiv.org/abs/2611.06170)

    本文提出专为低资源阿萨姆语设计的开源 20 亿参数语言模型 AssameseLM-E2B，通过扩展字节级 BPE 词表大幅降低分词开销，并借助参数高效的持续预训练与监督微调，实现了在消费级设备上的高效部署。

    

    尽管大型语言模型（LLM）在高资源语言中表现出色，但由于分词效率低下、生成不稳定性和语言干扰，其在低资源语言中的性能往往会有所下降。本文提出了 AssameseLM-E2B 及其指令微调变体 AssameseLM-E2B-it，这是一个专门为低资源阿萨姆语适配的 20 亿参数模型。该模型基于 Gemma-4 架构构建，并采用扩展的字节级 BPE 词表，与多语言 2B–3B 基线相比，将分词开销降低了近 500%。我们采用结合 Hybrid LoRA 的参数高效持续预训练（CPT）流程，随后在平衡的多领域数据集上进行监督微调（SFT），以增强指令遵循和检索增强生成（RAG）能力。评估结果显示，该模型在 FLORES-200 上取得了 0.7488 的每字节比特数（BPB）得分，同时总体 RAG 得分为 96。

    arXiv:2611.06170v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) excel in high-resource languages, their performance often degrades in low-resource languages due to tokenization inefficiencies, generative instability, and linguistic interference. This paper presents AssameseLM-E2B and its instruction-tuned variant, AssameseLM-E2B-it, a 2-billion-parameter model adapted specifically for low resource Assamese language. Built upon the Gemma-4 architecture, the model incorporates an expanded byte-level BPE vocabulary to reduce tokenization overhead by nearly 500% compared with multilingual 2B--3B baselines. We employ a parameter-efficient Continual Pre-Training (CPT) pipeline with Hybrid LoRA, followed by Supervised Fine-Tuning (SFT) on a balanced, multi-domain dataset to enhance instruction-following and Retrieval-Augmented Generation (RAG) capabilities.   Evaluation demonstrates a Bits-per-Byte (BPB) score of 0.7488 on FLORES-200, alongside an overall RAG score of 96
    
[^102]: FaCoDiff：面向多元概率预测的因子相关扩散模型

    FaCoDiff: Factor-Correlated Diffusion for Multivariate Probabilistic Forecasting

    [https://arxiv.org/abs/2611.06166](https://arxiv.org/abs/2611.06166)

    FaCoDiff通过以历史为条件的因子载荷、对角补偿与低秩算子，将紧凑的条件统计估计与结构化去噪相耦合，在无需稠密协方差存储或分解的情况下实现高效的多元概率预测。

    

    多元概率预测需要在刻画边缘不确定性的同时建模变量间的相互依赖关系。其挑战在于如何在控制各边缘噪声尺度、避免稠密协方差高昂计算开销的前提下学习这种依赖关系。我们提出因子相关扩散模型，将一个紧凑的条件统计估计器与结构化去噪相耦合。以历史数据为条件的因子载荷针对预测中心修正、尺度调整和噪声依赖分别使用不同的系数。对角补偿机制保留了估计出的边缘噪声方差，而低秩算子使得残差高斯似然评估与噪声传输无需存储或分解稠密协方差矩阵即可实现。源自因子的特征调制显式地对去噪器进行条件化，几何感知的目标函数则对噪声传输后的误差进行考量。同一条件几何结构同时支配前向扰动与反向采样过程。在……（摘要原文在此处截断）

    arXiv:2611.06166v1 Announce Type: new  Abstract: Multivariate probabilistic forecasting requires modeling cross-variable dependence alongside marginal uncertainty. The challenge is to learn this dependence while controlling marginal noise scales and avoiding dense covariance costs. We propose Factor-Correlated Diffusion (FaCoDiff), which couples a compact conditional statistical estimator with structured denoising. History-conditioned factor loadings use distinct coefficients for forecast-center correction, scale adjustment and noise dependence. Diagonal compensation preserves the estimated marginal noise variances, while low-rank operators enable residual Gaussian likelihood evaluation and noise transport without dense covariance storage or factorization. Factor-derived feature modulation explicitly conditions the denoiser, and a geometry-aware objective accounts for errors after noise transport. The same conditional geometry governs forward perturbations and reverse sampling. Across 
    
[^103]: 持久订单做市问题的遗憾界

    Regret Bounds for Market Making with Persistent Orders

    [https://arxiv.org/abs/2611.06159](https://arxiv.org/abs/2611.06159)

    本文针对订单持续有效直至执行的对抗性双市场做市问题，提出了一个简单确定性策略，其相对静态和动态比较器分别取得最优的 $\log T$ 和 $\sqrt T$ 遗憾界，并证明了两项价格风险暴露项的必要性。

    

    我们研究了对抗环境下的做市问题，其中订单在执行前会一直保持活跃。做市商在两个市场上运作：一个主市场（在此下单报价）和一个次级市场（在此对产生的头寸进行对冲）。在每一轮中，做市商在观察主市场和次级市场价格之前，先在主市场上发布报价。当主市场价格达到报价时，订单执行，产生的头寸立即以次级市场价格进行对冲。当两个价格序列在T轮内均由对手选择时，我们给出了一个简单的确定性策略，其相对于最佳固定报价（静态比较器）的遗憾为 $\log T$ 阶，相对于每个提交日期的最佳报价（动态比较器）的遗憾为 $\sqrt T$ 阶，另有两项用于衡量未来主市场价格和次级市场价格带来的风险暴露。即使当两项风险暴露均消失时，这些遗憾率仍是最优的，且下界结果表明风险暴露项是必要的。

    arXiv:2611.06159v1 Announce Type: cross  Abstract: We study adversarial market making with orders that remain active until execution. The maker operates on two markets: a primary market where orders are made and a secondary market where the resulting positions are hedged. In each round, the maker posts a quote on the primary market before observing both primary and secondary market prices. An order executes when the primary price reaches its quote, and the resulting position is immediately offset at the secondary price. When both price sequences are chosen by an adversary over $T$ rounds, we give a simple deterministic policy with regret of order $\log T$ against the best fixed quote (static comparator) and $\sqrt T$ against the best quote for each submission date (dynamic comparator), plus two terms measuring exposure to future primary and secondary prices. These rates remain optimal even when both exposure terms vanish and lower bounds show that the exposure terms are necessary. Our 
    
[^104]: 选择性预测区间：联合学习预测、界定与弃权

    Selective Prediction Intervals: Jointly Learning to Predict, Bound, and Abstain

    [https://arxiv.org/abs/2611.06158](https://arxiv.org/abs/2611.06158)

    提出SelPI方法，在单一损失函数和显式接受率预算下联合训练预测值、区间和弃权三个头，并实证证明仅凭区间宽度无法可靠判断何时应当弃权。

    

    许多高风险的回归决策同时要求模型做三件事：一个可以据以行动的点估计、一个能够界定行动风险的区间，以及在两者都不可信时选择弃权的选项。深度学习只能孤立地提供这些能力：预测区间方法为每个输入返回一个区间（充其量再给出区间内的一个值估计），而选择性预测方法学习何时弃权，但仅针对点预测，且无法说明真实值与估计值可能相差多远。我们提出了SelPI，它在单一损失函数下以显式的接受率预算联合训练区间、值和弃权三个头。一个自然的质疑是，区间宽度本身已经能告诉我们何时该弃权。我们证明事实并非如此：在18个表格数据基准上，区间宽度能较好地预测大的点误差（AUROC 0.72），但与区间失误呈负相关（AUROC 0.38），因此基于宽区间进行弃权恰恰会丢弃那些……

    arXiv:2611.06158v1 Announce Type: new  Abstract: Many high-stakes regression decisions ask three things of a model at once: a point estimate to act on, an interval that bounds the risk of acting, and the option to abstain when neither can be trusted. Deep learning provides these in isolation: prediction-interval methods return an interval, and at best a value estimate inside it, for every input, while selective-prediction methods learn when to abstain, but only for point predictions and with no statement of how far the truth may lie from the estimate. We introduce SelPI, which trains interval, value and abstention heads jointly under a single loss with an explicit acceptance-rate budget. A natural objection is that interval width already tells us when to abstain. We show that it does not: across 18 tabular benchmarks, width predicts large point errors well (AUROC 0.72) but is anti-correlated with interval misses (AUROC 0.38), so abstaining on wide intervals discards precisely the input
    
[^105]: 用于侧向分子扩散模拟的神经随机预测方法

    Neural Stochastic Forecasting for Lateral Molecular Diffusion Simulations

    [https://arxiv.org/abs/2611.06157](https://arxiv.org/abs/2611.06157)

    提出了一种结合Transformer构象表示、多时间尺度循环网络和带学习记忆的随机潜过程的神经随机模型，能够从粗粒化分子动力学轨迹高效预测膜蛋白在膜内的长时间集体扩散。

    

    预测膜蛋白的长时间扩散具有挑战性，因为分子动力学（MD）模拟的计算成本高昂，而实验中的单粒子跟踪通常又受限于有限的观察时间。在此，我们提出了一种神经随机（NS）模型，用于从粗粒化MD轨迹预测膜蛋白的集体扩散。该模型结合了基于Transformer的蛋白质构象表示、在多个时间尺度上运行的循环网络，以及具有学习记忆功能的随机潜过程。随后，交叉注意力解码器在考虑系统瞬时几何构型和周期性边界条件的情况下，生成相关的蛋白质位移。我们在两个膜系统中对该模型进行了评估：一个是七个相互作用的PIEZO通道系统，另一个是包含七个PIEZO通道和十个TREK通道的混合系统。对于七PIEZO系统，NS模型重现了异常（扩散行为）……

    arXiv:2611.06157v1 Announce Type: new  Abstract: Predicting the long-time diffusion of membrane proteins is challenging because molecular dynamics (MD) simulations are computationally expensive, while experimental single-particle tracking is often limited by finite observation times. Here, we introduce a neural stochastic (NS) model for forecasting the collective diffusion of membrane proteins from coarse-grained MD trajectories. The model combines a transformer-based representation of protein configurations, recurrent networks operating at multiple temporal scales, and a stochastic latent process with learned memory. A cross-attention decoder then generates correlated protein displacements while accounting for the instantaneous system geometry and periodic boundary conditions. We evaluate the model in two membrane systems: seven interacting PIEZO channels and a mixed system containing seven PIEZO and ten TREK channels. For the seven PIEZO system, the NS model reproduces the anomalous 
    
[^106]: CARM：用于可靠科学超分辨率的保形自适应风险掩码

    CARM: Conformal Adaptive Risk Masking for Reliable Scientific Super-Resolution

    [https://arxiv.org/abs/2611.06151](https://arxiv.org/abs/2611.06151)

    提出CARM框架，通过维护异构更新速率的专家并利用非对称代理损失在线聚合阈值，在时间分布偏移下实现自适应空间风险控制，为科学超分辨率生成可靠的信任掩码。

    

    基于深度学习的超分辨率（SR）技术可以从粗分辨率观测中重建精细尺度的地球观测场，但仅靠重建精度本身并不能提供一个经过校准的规则来识别可靠的空间区域。保形掩码校准（CMC）建立在保形风险控制的基础上，将空间不确定性分数转换为信任掩码，并在可交换性假设下控制预期最坏情况的局部失真。然而，其固定阈值在受季节性和偶发事件影响的地球观测数据流中可能会失去校准。我们提出了CARM（保形自适应风险掩码），这是一个在时间分布偏移下实现自适应空间风险控制的统计框架。CARM维护多个具有异构更新速率的专家，并使用非对称代理损失在线聚合它们的阈值提议，从而在快速适应分布变化与相对平稳时期的稳定性之间取得平衡。

    arXiv:2611.06151v1 Announce Type: new  Abstract: Deep learning-based super-resolution (SR) can reconstruct fine-scale Earth-observation fields from coarse observations, but reconstruction accuracy alone does not provide a calibrated rule for identifying reliable spatial regions. Conformal Mask Calibration (CMC), built on conformal risk control, converts a spatial uncertainty score into a trust mask and controls the expected worst-case local distortion under exchangeability. However, its fixed threshold can become miscalibrated in Earth-observation streams affected by seasonality and episodic events. We propose $\textit{CARM}$ (Conformal Adaptive Risk Masking), a statistical framework for adaptive spatial risk control under temporal distribution shift. CARM maintains experts with heterogeneous update rates and aggregates their threshold proposals online using an asymmetric surrogate loss, balancing rapid adaptation to distribution changes against stability during relatively stationary p
    
[^107]: SafeStart：缓解MoE大语言模型服务中主动专家削减所带来的被忽视的安全成本

    SafeStart: Mitigating the Overlooked Safety Cost of Active Expert Reduction in MoE LLM Serving

    [https://arxiv.org/abs/2611.06147](https://arxiv.org/abs/2611.06147)

    本文揭示了MoE大模型服务中减少激活专家数量的加速方法会削弱模型对有害请求的拒绝能力，并提出SafeStart方法，在保留大部分加速效果的同时大幅恢复模型的安全性。

    

    arXiv:2611.06147v1 公告类型：新论文 摘要：混合专家模型将每个token仅通过众多专家子网络中的少数几个进行处理。现有的服务加速方法通过让每个token经过更少的专家来提升推理速度，并通常在准确率基准上进行验证。我们展示了这些方法会削弱模型对有害请求的拒绝能力，并提出了SafeStart，它在保留大部分加速效果的同时，能够大幅恢复模型的拒绝能力。在单张RTX A6000上使用vLLM的实验中，保留约三分之二的被选中专家可在batch size为8时将离线解码速度提升15%到28%。我们发现，在此设置下，在一种指示模型不要拒绝的越狱攻击下，模型回答的有害请求比例在Qwen1.5-MoE上上升18.4个百分点，在OLMoE上上升9.5个百分点，在DeepSeek-V2-Lite上上升6.6个百分点（这是九个测试模型中显著上升的三个），而MMLU准确率下降不到2分。我们还发现，这种上升大部分发生在模型处理提示词的阶段。

    arXiv:2611.06147v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) language models send each token through a few of many expert subnetworks. Serving methods make these models faster by running even fewer experts per token, and they are validated on accuracy benchmarks. We show that these methods can weaken the model's refusal of harmful requests, and we introduce SafeStart, which largely restores refusal while keeping most of the speedup. In vLLM on one RTX A6000, keeping about two thirds of the selected experts speeds up offline decoding by 15 to 28% at batch size 8. We find that, at this setting, under a jailbreak that tells the model not to refuse, the share of harmful requests that the model answers rises by 18.4 percentage points on Qwen1.5-MoE, 9.5 on OLMoE, and 6.6 on DeepSeek-V2-Lite, the three of nine models where it rises significantly, while MMLU accuracy drops by less than two points. We discover that most of this increase arises while the model processes the prompt 
    
[^108]: 异构风电场的短期功率预测：利用静态协变量的全局时间融合Transformer

    Short-Term Forecasting of Heterogeneous Wind Power Fleets: Global Temporal Fusion Transformer Leveraging Static Covariates

    [https://arxiv.org/abs/2611.06130](https://arxiv.org/abs/2611.06130)

    本研究提出一种利用静态协变量（场龄、海拔、轮毂高度、转子直径和功率曲线参数等）在异构风电场组合上联合训练的全局时间融合Transformer（TFT），并证明其在48小时前瞻风电功率预测中优于针对单个站点训练的局部模型。

    

    精确的短期风电功率预测对电网稳定性和电力交易至关重要。尽管基于Transformer的深度学习是当前最先进的技术，但行业实践主要依赖针对单个资产训练的局部模型。本研究探讨在德国多个异构风电场组合上联合训练的全局时间融合Transformer（TFT），在48小时前瞻预测中是否优于针对特定站点的局部模型。研究使用来自区域ICON-D2模式和全球ECMWF HRES模式的数值天气预报（NWP）数据，将全局TFT与局部模型及线性回归基线进行比较。全局模型利用静态协变量（如风电场场龄、海拔高度、轮毂高度、转子直径和功率曲线参数）来学习跨站点的共享表示。研究在100个合成风电场剖面和由11个德国真实风电场组成的组合上进行评估，并跨越两个季节进行测试。

    arXiv:2611.06130v1 Announce Type: new  Abstract: Accurate short-term wind power forecasting is critical for grid stability and electricity trading. While transformer-based deep learning is state of the art, industry practice relies mostly on local models trained for individual assets. This study investigates whether a global temporal fusion transformer (TFT), trained jointly on a heterogeneous portfolio of wind farms across Germany, outperforms site-specific local models in 48-hour-ahead forecasting. Using numerical weather prediction (NWP) data from the regional ICON-D2 and the global ECMWF HRES models, we compare the global TFT against local models and a linear regression baseline. The global model leverages static covariates, such as farm age, altitude above sea level, hub height, rotor diameter and power curve parameters, to learn shared representations across sites. We evaluate on 100 synthetic wind farm profiles and a real portfolio of 11 German wind farms, across two seasonal te
    
[^109]: 从手工绘制到生成式：面向 Alexa+ 的 RLVR 优化图标生成

    From Handcrafted to Generative: RLVR-Optimized Icon Generation for Alexa+

    [https://arxiv.org/abs/2611.06123](https://arxiv.org/abs/2611.06123)

    该论文提出并部署了一套仅凭文本描述自动生成矢量化动画图标的生产级流水线，并将图标创建分解为多个阶段，重点利用 RLVR（可验证奖励的强化学习）优化其中决定整体质量的最关键阶段——图像生成，从而解决手工绘制图标耗时且难以规模化的问题。

    

    配备屏幕的 Alexa+ 设备会在对话过程中显示动态图标，以视觉方式强化所表达的内容。最初的这批图标由设计师逐个手工绘制，这种方式虽能产出精美的成果，但耗时且无法扩展至开放式对话中遇到的各种话题。我们描述了一条已部署于生产环境的自动化流水线，它能够仅凭文本描述生成这些矢量化动态图标，并根据用户正在进行的对话向其展示相关的动画。该流水线将图标创建分解为一系列阶段（图像生成、语义分解、矢量化、样式化和动画制作）。其质量在最前面的图像生成阶段就被决定，所有下游阶段都依赖于这一阶段。因此，我们专注于改进这一首要且最关键的阶段。由于这些图标需要在严格的设计约束下在小尺寸显示画布上渲染……

    arXiv:2611.06123v1 Announce Type: new  Abstract: Alexa+ devices with screens display animated icons during conversations to visually reinforce what is being said. The original set of these icons was hand-drawn by designers one at a time, a process that produces polished results but is time-consuming and cannot scale to the range of topics encountered in open-ended conversation. We describe an automated pipeline, deployed in production, that generates these vectorised animated icons from a text description alone and surfaces relevant animations to customers based on their ongoing conversation. The pipeline decomposes icon creation into a sequence of stages (image generation, semantic decomposition, vectorisation, styling, and animation). Its quality is determined at the very first of these, image generation, on which every downstream stage depends. We therefore focus on improving this first and most critical stage. Because the icons are rendered on small display canvases under strict de
    
[^110]: 在潜在推理中学习何时进行显式思考

    Learning When to Think Aloud in Latent Reasoning

    [https://arxiv.org/abs/2611.06108](https://arxiv.org/abs/2611.06108)

    VITAL提出了一种重要性引导的混合推理框架，通过轻量级门控机制动态决定哪些token需要显式生成作为推理锚点、哪些计算保持在潜在空间，并结合课程式监督训练与平衡正确性和简洁性的强化学习，从而在效率和精度之间取得平衡并超越现有完全潜在推理方法。

    

    潜在思维链推理通过将中间计算转移到连续的隐藏状态中来提高效率。然而，完全潜在的方法在诸如数学推理等对精度要求严苛的任务上往往表现不佳，而现有的混合方法则保留了大量的token生成开销。我们提出了VITAL，这是一个基于重要性引导的混合推理框架，它选择性地将信息量大的token显式化为推理锚点，同时将大部分计算保持在潜在空间中。一个轻量级的门控头在显式token生成和直接隐藏状态反馈之间进行动态选择。VITAL利用token重要性估计从显式CoT构建混合轨迹，先通过基于课程的监督训练来训练模型，随后进行强化学习，其奖励函数在正确性与简洁性之间取得平衡。实验表明，VITAL持续优于现有的完全潜在推理方法，实现……

    arXiv:2611.06108v1 Announce Type: new  Abstract: Latent Chain-of-Thought (CoT) reasoning improves efficiency by shifting intermediate computation into continuous hidden states. However, fully latent methods often struggle on precision-critical tasks such as mathematical reasoning, while existing hybrid approaches retain substantial token-generation overhead. We introduce VITAL, an importance-guided hybrid reasoning framework that selectively verbalizes informative tokens as reasoning anchors while keeping most computation latent. A lightweight gating head dynamically selects between explicit token generation and direct hidden-state feedback. VITAL constructs hybrid trajectories from explicit CoT using token- importance estimation and trains the model through curriculum-based supervised training, followed by reinforcement learning with a reward that balances correctness and conciseness. Experiments show that VITAL consistently out- performs existing fully latent reasoning methods, achie
    
[^111]: 多像素，单一成分：面向样本级预测的自监督光谱方法

    Many Pixels, One Composition: Self-Supervised Spectra for Specimen-Level Prediction

    [https://arxiv.org/abs/2611.06096](https://arxiv.org/abs/2611.06096)

    该论文提出一种自监督像素级编码器结合轻量级注意力回归头的方法，从高光谱成像预测地球表面物理样本的元素成分，在保持较低模型容量的情况下显著优于传统光谱匹配、解混及基于平均光谱的监督学习基线。

    

    高光谱（HS）成像能否用于预测地球表面所发现的物理样本的元素成分？我们将该问题视为一个监督回归任务，使用一个开放的样本数据集，其中每个样本由一个高光谱立方体和元素表征组成。具体而言，我们以自监督方式训练一个在像素级别运行的编码器；随后冻结该编码器，并以监督方式训练一个轻量级的基于注意力的回归头，方法是将给定样本的多个表征与其对应的元素标签进行配对。在测试样本上，我们的方法显著优于传统的光谱匹配和解混基线方法。同时，它也优于基于平均光谱的监督学习模型（模型容量相近），而完整模型的微调则以大幅更高的模型容量取得了最佳性能。

    arXiv:2611.06096v1 Announce Type: cross  Abstract: Can hyperspectral (HS) imaging be used to predict the elemental composition of physical specimens found on the Earth's surface? We approach this problem as a supervised regression task, using an open dataset of specimens, each comprising a HS cube and elemental characterization. In particular, we train an encoder in a self-supervised fashion that operates on pixel-level; then freeze the encoder and train a lightweight attention-based regression head in a supervised manner, by pairing many representations of a given specimen with its corresponding elemental label. In test specimens, our approach significantly outperforms traditional spectral matching and unmixing baselines. It also outperforms supervised learning models based on the mean spectrum (of similar model capacity), while full model fine-tuning achieves the strongest performance, at substantially higher model capacity.
    
[^112]: 浅层编码释放深层记忆

    Shallow Encoding Emits Deep Memories

    [https://arxiv.org/abs/2611.06084](https://arxiv.org/abs/2611.06084)

    提出了“发射器”机制——即深层写入路径的训练副本，能够从较浅的锚定层残差流中直接写出深层记忆，从而大幅降低混合语言模型推理服务中的计算与内存开销。

    

    混合语言模型将少数softmax注意力层与大量线性注意力层或状态空间层交错排列，因此每一层都保存着先前词元的记忆：键值缓存或固定大小的循环状态。在推理服务中，这些记忆所消耗的计算量和高速带宽内存超过了实际输出所需。预填充阶段需要对每个提示词元运行所有层，而每个解码步骤都需要加载和存储完整的循环状态——在Qwen3.8-Flash-Next上，这相当于每个请求要消耗数千词元注意力缓存所占用的字节数。我们发现了训练后的混合模型的两个特性，可以定位这种冗余开销。首先，写入比读取更容易：写入某一深层对某个词元的记忆，所需的深层计算远少于预测下一个词元所需的计算。因此，我们利用深层写入路径的训练副本（称为“发射器”emitters），从较浅的锚定层的残差流中写出提示的深层记忆，并且所有提示词元……（摘要原文在此处截断）

    arXiv:2611.06084v1 Announce Type: new  Abstract: Hybrid language models interleave a few softmax-attention layers with many linear-attention or state-space layers, so every layer keeps a memory of earlier tokens: a key-value cache or a fixed-size recurrent state. Serving them spends more compute and high-bandwidth memory on these memories than the output requires. The prefill runs every layer for every prompt token, and every decode step loads and stores whole recurrent states, which on Qwen3.8-Flash-Next take as many bytes per request as the attention cache of thousands of tokens. We find two properties of trained hybrids that locate this excess. First, writes are easier than reads: writing a deep layer's memory of a token needs little of the deep computation that predicting the next token needs. Trained copies of the deep layers' write paths, which we call emitters, therefore write a prompt's deep memories from the residual stream of a shallower anchor layer, and all prompt tokens bu
    
[^113]: 灾难性过拟合是训练损失看不见的权重漂移

    Catastrophic Overfitting is Weight Drift the Training Loss Cannot See

    [https://arxiv.org/abs/2611.06082](https://arxiv.org/abs/2611.06082)

    该论文揭示了灾难性过拟合的本质是训练损失几乎无法感知的权重漂移，并提出利用FAT协议中已有的权重指数移动平均（EMA）作为几乎零成本的修复方案。

    

    快速对抗训练（FAT）通过用单步攻击替代多步攻击，使鲁棒训练变得经济可行，但它饱受灾难性过拟合（CO）的困扰：在训练进行到中途时，对多步攻击的鲁棒性会崩溃至接近零，而用于训练的单步攻击却持续被成功抵御。我们证明了CO本质上是权重的一次悄然逃逸。在整个事件过程中，网络在干净样本上的行为悄然改善，而从CO发生前一刻的权重移动到CO发生后一刻的权重，只会边际地改变训练目标，却摧毁了绝大部分鲁棒性。因此，训练目标对其所导致的位移几乎是盲目的，因而不会施加任何力量来撤销这一位移。FAT协议本身已经在维护的权重指数移动平均（EMA）并不会累积这种漂移，并能在整个事件期间保持鲁棒。这一观察提示了一种几乎零成本的修复方法：当验证集鲁棒性下降时，……

    arXiv:2611.06082v1 Announce Type: new  Abstract: Fast adversarial training (FAT) makes robust training affordable by replacing the multi-step attack with a single-step one, but it suffers from catastrophic overfitting (CO): partway through training, robustness to multi-step attacks collapses to near zero while the single-step attack used for training keeps being repelled. We show that CO is a quiet escape of the weights. Across the event the network's clean behavior quietly improves, and moving from the weights just before CO to those just after it changes the training objective only marginally while destroying most of the robustness. The training objective is therefore nearly blind to the displacement it causes, so it exerts no force to undo it. The exponential moving average (EMA) of the weights, which FAT protocols already maintain, does not accumulate this drift and stays robust through the entire event. This observation suggests a nearly free fix: when validation robustness drops,
    
[^114]: 从删失出价中离线学习保留价格

    Offline Learning of Reserve Prices from Censored Bids

    [https://arxiv.org/abs/2611.06067](https://arxiv.org/abs/2611.06067)

    本文在单调风险率假设下精确刻画了从删失出价数据中识别二价拍卖最优保留价格的条件，并在识别失败时给出仅依赖删失比例和最低历史保留价格处风险率这两个可观测量的极小极大遗憾最优保留价格。

    

    我们研究卖家如何从删失的历史出价数据中学习二价拍卖的保留价格。由于低于过去保留价格的估值是不可观测的，即使是无限的数据也可能无法揭示哪个保留价格能带来最多收入。在单调风险率（MHR）假设下，我们精确刻画了最优保留价格何时可以被识别——即使价值分布本身可能无法被识别：要么收入在最低历史保留价格处尚未开始下降，因此最优保留价格位于观测到的出价之中；要么删失比例达到MHR所允许的最大值，此时整个分布被完全确定。当识别失败时，我们推导出在与数据一致的所有分布上最小化最坏情况收入遗憾的保留价格，并证明这一唯一的极小极大遗憾保留价格及其遗憾值仅取决于两个可观测的量：删失比例和略高于最低历史保留价格处的风险率。当有N个历史竞……

    arXiv:2611.06067v1 Announce Type: cross  Abstract: We study how a seller can learn a reserve price for a second-price auction from censored historical bid data. Because values below past reserves are unobserved, even unlimited data may not reveal which reserve earns the most revenue. Under a monotone hazard rate (MHR) assumption, we characterize exactly when an optimal reserve is identified, even though the value distribution may not be: either revenue has not yet begun to fall at the lowest historical reserve, so the best reserve lies among the observed bids, or the censored fraction is as large as MHR allows, which pins down the entire distribution. When identification fails, we derive the reserve that minimizes worst-case revenue regret over distributions consistent with the data, and prove that this unique minimax reserve and its regret depend on just two observable quantities: the censored fraction and the hazard just above the lowest historical reserve. With $N$ historical bidder
    
[^115]: PRISM-AMC：面向自动调制分类的无回放持续学习

    PRISM-AMC: Replay-Free Continual Learning for Automatic Modulation Classification

    [https://arxiv.org/abs/2611.06066](https://arxiv.org/abs/2611.06066)

    PRISM-AMC是一种无需数据回放的持续学习框架，通过梯度投影记忆保留任务相关的中间表示子空间，在类增量和域增量场景下有效缓解自动调制分类中的灾难性遗忘。

    

    自动调制分类（AMC）是认知无线系统的关键组成部分，能够实现频谱感知和自适应通信。大多数AMC的深度学习方法假设调制类别集合是固定的，并且可以访问之前观察到的训练数据。而在实际中，调制类别和信道条件可能随时间演变，同时存储或数据保留的限制可能使原始基带I/Q样本无法回放。我们提出了PRISM-AMC，这是一个无回放的持续学习框架，通过梯度投影记忆保留中间表示中与任务相关的子空间，从而缓解灾难性遗忘。我们进一步在信噪比（SNR）相关的梯度噪声模型下分析了投影随机梯度更新。PRISM-AMC在类增量学习和域增量学习两种设置下均进行了评估。在MIGOU和DeepSig 2018.01A数据集上的实验表明，相较于顺序训练，遗忘程度显著降低。

    arXiv:2611.06066v1 Announce Type: new  Abstract: Automatic modulation classification (AMC) is a key component of cognitive wireless systems, enabling spectrum awareness and adaptive communication. Most deep learning approaches to AMC assume a fixed set of modulation classes and access to previously observed training data. In practice, modulation classes and channel conditions may evolve over time, while storage or data-retention constraints may preclude replay of raw baseband I/Q samples. We propose PRISM-AMC, a replay-free continual learning framework that mitigates catastrophic forgetting by preserving task-relevant subspaces of intermediate representations via gradient projection memory. We further analyze projected stochastic-gradient updates under an SNR-dependent gradient-noise model. PRISM-AMC is evaluated in both class-incremental and domain-incremental settings. Experiments on the MIGOU and DeepSig 2018.01A datasets show substantially reduced forgetting relative to sequential 
    
[^116]: 联邦时间序列预测中从静态上下文学习

    Learning From Static Context in Federated Time Series Forecasting

    [https://arxiv.org/abs/2611.06064](https://arxiv.org/abs/2611.06064)

    在联邦时间序列预测中，逐批次聚合梯度的FedAdam几乎能像集中式训练一样充分利用描述序列的静态上下文输入，而采用本地epoch的通信高效方案（如FedAvg、FedAdam+Prox）则难以从中学习且无法获益。

    

    时间序列预测正在从针对单个序列本地训练的模型，转向跨众多异构序列训练的全局模型，这要求模型学习区分不同序列的特征。当原始数据无法集中时，这种全局模型需要通过联邦学习（FL）进行训练，每个数据拥有者对应一个客户端。最先进的预测模型（如时间融合Transformer，TFT）通过一个描述每个序列且不随时间变化的静态输入来区分不同序列。在每个客户端仅持有一个序列的联邦设置这一特殊情况下，该静态输入在每个客户端内部是恒定的，仅在客户端之间存在差异。我们探究了联邦化的TFT能否从中进行学习。研究发现：逐批次聚合梯度的Per-batch FedAdam几乎能像集中式训练一样充分利用静态输入，而采用本地epoch的通信高效方案（FedAvg、FedAdam+Prox）则从中提取的信息少得多，且无法从中获益。

    arXiv:2611.06064v1 Announce Type: new  Abstract: Time series forecasting is shifting from locally trained, per-series models to global models trained across many heterogeneous series, which requires learning what sets the series apart. When the raw data cannot be centralized, such a global model is trained with federated learning (FL), one client per data owner. State-of-the-art forecasters such as the Temporal Fusion Transformer (TFT) tell series apart through a static input that describes each series and does not change over time. In the special case of a federated setting where each client holds a single series, this static input is constant within each client and varies only across clients. We ask whether a federated TFT can learn from it. Per-batch FedAdam, which aggregates gradients after every batch, uses the static input almost as well as central training, whereas communication-efficient schemes with local epochs (FedAvg, FedAdam+Prox) extract far less from it and gain nothing 
    
[^117]: 固定维度下任意温度的Softmax注意力：匹配已知的最大内积最佳界

    Softmax Attention with Any Temperature in Fixed Dimension: Matching the Best Known Maximum-Inner-Product Bounds

    [https://arxiv.org/abs/2611.06055](https://arxiv.org/abs/2611.06055)

    该论文提出了固定维度下支持任意温度的softmax注意力快速算法，其期望运行时间达到了已知的最大内积搜索最优理论界。

    

    注意力机制是现代语言、视觉和科学建模AI模型的核心。然而，直接计算需要将每个查询与每个键进行比较，这需要与上下文长度呈二次方关系的时间，限制了其扩展到长上下文的能力。因此，刻画注意力的复杂度界具有根本性的重要意义。对于每个固定维度 $d$，我们给出了一个针对 $m$ 个查询、$n$ 个键值对和加性误差 $\varepsilon$ 的算法，其期望时间为 $\widetilde{\mathrm{O}}\bigl((mn)^{k/(k+1)}+m+n\bigr)$，其中 $k=\lfloor d/2\rfloor$。在固定维度和固定数值通道数的条件下，软O符号隐藏了形如 $\operatorname{poly}_d\bigl(b,\log(m+n),\log B,\log(1/\varepsilon)\bigr)$ 的因子，其中 $b$ 限定输入编码长度，$B$ 限定坐标和数值的大小。特别地，对 $B$ 和 $1/\varepsilon$ 的依赖是多项式对数级别的，这使得算法可以支持任意正温度。

    arXiv:2611.06055v1 Announce Type: cross  Abstract: Attention is central to modern AI models for language, vision, and scientific modeling. However, direct evaluation compares every query with every key, requiring quadratic time in the context length, which limits its scalability to long contexts. Consequently, characterizing the complexity bounds of attention is of fundamental importance.   For every fixed dimension $d$, we give an algorithm for $m$ queries, $n$ key-value pairs, and additive error $\varepsilon$, with expected time $ \widetilde{\mathrm{O}}\bigl((mn)^{k/(k+1)}+m+n\bigr),\qquad k=\lfloor d/2\rfloor. $   For fixed dimension and a fixed number of value channels, soft-O hides factors of the form $\operatorname{poly}_d\bigl(b,\log(m+n),\log B,\log(1/\varepsilon)\bigr),$ where $b$ bounds input encoding lengths and $B$ bounds coordinate and value magnitudes. In particular, the dependence on $B$ and $1/\varepsilon$ is polylogarithmic, which allows arbitrary positive temperature.
    
[^118]: 缓解深度图级聚类中的有效性受限与多类坍塌问题

    Alleviating Limited Effectiveness and Multi-Class Collapse in Deep Graph-level Clustering

    [https://arxiv.org/abs/2611.06052](https://arxiv.org/abs/2611.06052)

    该论文提出S³GLC框架，利用Hoffman-Wielandt不等式构建谱签名作为图编辑距离的高效下界代理，将全局拓扑特征与局部邻域特征相融合，从而缓解深度图级聚类中区分能力不足和多类别聚类坍塌的问题。

    

    给定一个包含N个图的集合，图级聚类（GLC）旨在将这些图划分为语义上有意义的群组，这是分析图结构数据的一项基础任务。尽管图同构网络（GIN）已成为GLC事实上的标准编码器，但由于忽视全局拓扑结构，它们仍无法有效区分不同的图。此外，现有的深度聚类框架存在严重的“聚类坍塌”问题，在不平衡的多类别场景中尤为突出。为应对这些挑战，我们提出了S³GLC，这是一个将全局拓扑特征与局部邻域特征相融合、以实现平衡图级聚类的新颖框架。在理论上，我们利用Hoffman-Wielandt不等式构建了谱签名，作为图编辑距离（GED）的严格下界，从而为全局拓扑特征提供了计算高效的代理。此外，我们还提出了一个联合流形……（摘要内容不完整）

    arXiv:2611.06052v1 Announce Type: new  Abstract: Given a collection of $N$ graphs, graph-level clustering (GLC) seeks to partition the $N$ graphs into semantically meaningful groups, which serves as a fundamental task in analyzing graph-structured data. While Graph Isomorphism Networks (GINs) have become the de facto encoders for GLC, they still fail to distinguish graphs due to their neglect of global topology. On top of that, existing deep clustering frameworks suffer from severe "clustering collapse" issues, particularly in imbalanced multi-class scenarios. To address these challenges, we propose S$^3$GLC, a novel framework that integrates global topological characteristics with local neighborhood features for balanced GLC. Theoretically, we leverage the Hoffman-Wielandt inequality to construct Spectral Signature as a rigorous lower bound for graph edit distance (GED), providing a computationally efficient proxy for global topological features. We additionally propose a joint manifo
    
[^119]: 负权重对储备池计算性能的影响

    Impact of Negative Weights on Performance in Reservoir Computing

    [https://arxiv.org/abs/2611.06051](https://arxiv.org/abs/2611.06051)

    带符号储备池中适度的负权重比例能够带来更丰富的内部动力学，并显著降低混沌时间序列预测的误差。

    

    储备池计算为时间序列预测提供了一种高效框架，因为只有读出层需要训练，而循环储备池保持固定。在本研究中，我们考察了带符号Erdős-Rényi储备池的负号比例如何影响回声状态网络在Mackey-Glass基准任务上的内部动力学与预测性能。为了刻画储备池的响应，我们使用了两个基于香农熵的可观测量，即节点状态熵和节点频率熵，作为储备池状态多样性与时间激活模式的经验代理指标。我们发现，在所考察的任务中，中等水平的负号比例相比符号同质的极端情况具有更丰富的内部动力学，并显著降低了预测误差。这些结果表明，连边符号组成可以成为面向混沌预测的储备池设计中一个有用的结构参数。

    arXiv:2611.06051v1 Announce Type: cross  Abstract: Reservoir computing provides an efficient framework for time-series prediction because only the readout layer is trained while the recurrent reservoir remains fixed. In this study, we examine how the negative sign ratio of a signed Erdos-Renyi reservoir affects the internal dynamics and forecasting performance of an echo state network on the Mackey-Glass benchmark. To characterize the reservoir responses, we use two Shannon-entropy-based observables, namely node-state entropy and node-frequency entropy, as empirical proxies for the diversity of reservoir states and temporal activation patterns. We find that intermediate negative sign ratios are associated with richer internal dynamics and substantially lower prediction error than the sign-homogeneous extremes for the task considered here. These results suggest that link-sign composition can be a useful structural parameter in reservoir design for chaotic forecasting. The present findin
    
[^120]: 搜索增强推荐系统是否使用了最新的搜索会话？

    Do Search-Augmented Recommenders Use the Latest Search Episode?

    [https://arxiv.org/abs/2611.06044](https://arxiv.org/abs/2611.06044)

    提出FocalSwap审计框架，通过在固定目标与候选集的条件下替换最新搜索或全部搜索输入，检验搜索增强推荐系统对最新搜索会话的实际依赖程度，发现其估计收益随对照方式的选择而变化。

    

    搜索增强推荐系统可以利用最新的搜索会话以及更早的搜索记录。我们提出FocalSwap来测试这些输入对下一次推荐的贡献。对于每一条推荐，FocalSwap在保持目标、候选集、搜索前历史和评分器不变的情况下，替换最新的搜索槽位或所有搜索输入。它将中性掩码与匹配的已观测搜索替换进行对比，记录排名变化、相对于各对照组的RR增益，以及对真实焦点搜索的偏好。植入信号实验和GenSAR的当前查询任务证实，该审计方法能够检测到已知的输入依赖。我们在KuaiSAR和Qilin数据集上评估行为与词汇证据，并使用基于公开的SESRec和UniSAR实现所训练的推荐模型检查点进行实验。在三个独立训练的UniSAR检查点上，估计的全源收益取决于对照组的选择。在相同的66,139行数据上，将所有搜索输入替换为中性ID…（原文摘要在此处被截断）

    arXiv:2611.06044v1 Announce Type: cross  Abstract: Search-augmented recommenders can use the latest search episode and earlier searches. We introduce FocalSwap to test how these inputs contribute to the next recommendation. For each recommendation, it changes the latest search slot or all search inputs while holding the target, candidates, pre-search history, and scorer fixed. It compares neutral masks with matched observed-search replacements, recording rank changes, RR gains relative to each control, and preference for the true focal search. A planted signal and GenSAR's current-query task confirm that the audit detects known input dependence. We evaluate behavioral and lexical evidence on KuaiSAR and Qilin, and recommendation checkpoints trained using the released SESRec and UniSAR implementations.   Across three independently trained UniSAR checkpoints, the estimated all-source benefit depends on the control. On the same 66,139 rows, replacing all search inputs with a neutral ID yi
    
[^121]: Act-HiMotif：通过学习的活动结构主题进行层次化日程活动计划生成

    Act-HiMotif: Hierarchical Daily Activity Schedule Generation via Learned Activity-Structure Motifs

    [https://arxiv.org/abs/2611.06038](https://arxiv.org/abs/2611.06038)

    该论文提出Act-HiMotif，一种层次化生成模型，通过学习可复用的活动结构主题作为个体画像与活动结构之间的中间表示，从个体属性生成每日活动计划，有效应对活动结构长尾分布和同一结构下多种时长分配的挑战。

    

    从个体属性（如年龄和收入）生成日程活动计划，对于基于活动的出行建模和人类行为模拟具有重要价值。这项任务具有挑战性，因为精确的活动结构遵循高度不平衡的长尾分布，而同一结构在固定的全天时间预算下又可以有多种时长分配方式。因此，直接生成方法必须从高度不均匀的观测数据中学习大量精确的活动结构，同时建模其可能的时间实现方式。然而，许多精确结构实际上共享重复出现的活动组织模式。基于这一观察，我们提出了Act-HiMotif，这是一种基于主题（motif）的层次化生成模型，它引入了通过学习获得的、可复用的活动结构主题，作为个体画像与特定活动结构之间的中间表示。Act-HiMotif首先区分仅有居家活动的日子和有外出活动的日子，然后从以个体画像为条件的主题分布中采样一个主题……

    arXiv:2611.06038v1 Announce Type: new  Abstract: Generating daily activity schedules from individual attributes (e.g., age and income) is valuable for activity-based travel modeling and human behavior simulation. The task is challenging because exact activity structures follow a highly imbalanced, long-tailed distribution, while the same structure can admit multiple duration allocations under a fixed full-day budget. Direct generation must therefore learn many exact activity structures from highly uneven observations while modeling their possible temporal realizations. However, many exact structures share recurring activity organization. Based on this observation, we propose Act-HiMotif, a hierarchical motif-based generative model that introduces learned, reusable activity-structure motifs as an intermediate representation between individual profiles and specific activity structures. Act-HiMotif first distinguishes home-only days from active days, then samples a motif from a profile-co
    
[^122]: TRACK：面向交易网络反洗钱的多示例学习方法

    TRACK: Multiple Instance Learning for Anti-Money Laundering in Transaction Networks

    [https://arxiv.org/abs/2611.06035](https://arxiv.org/abs/2611.06035)

    该论文提出TRACK方法，将账户级反洗钱检测重新表述为多示例学习问题，通过共享编码器对每笔交易编码并选择性汇聚为账户决策，从而在无需交易级标签的情况下解决了“聚合失配”问题，并在五个AML数据集上取得了最高的平均精度。

    

    反洗钱（AML）调查以账户为单位组织进行，而一个账户的可疑证据可能仅存在于其少数几笔交易中。然而，许多账户级检测器在学习之前就先对交易进行了汇总，而消息传递模型还可能在相邻账户之间进行二次聚合。我们将这种差异命名为“聚合失配”（aggregation mismatch），并在以“至少存在一笔被标记交易”为定义的账户窗口基准标签下对此展开研究。我们将账户级AML检测重新表述为多示例学习问题：账户窗口作为一个“包”（bag），其交易则作为“示例”（instance）。我们提出了TRACK方法，该方法使用共享编码器对每笔交易进行编码，并有选择地将编码汇聚为账户级决策。为模拟粗粒度监督，我们在整个训练过程中对TRACK隐藏了五个AML数据集的交易标签。在按时间顺序的评估协议下，TRACK取得了最高的平均精度

    arXiv:2611.06035v1 Announce Type: new  Abstract: Anti-money laundering (AML) investigations are organized around accounts, while the suspicious evidence for an account may reside in only a few of its transactions. Yet many account-level detectors summarize transactions before learning, and message-passing models may aggregate once more over neighboring accounts. We name this discrepancy the aggregation mismatch and study it under an account-window benchmark label defined by the presence of at least one flagged transaction. We recast account-level AML detection as multiple instance learning, in which an account-window is a bag and its transactions are instances. We propose TRACK, which encodes every transaction with a shared encoder and selectively pools the encodings into an account decision. To emulate coarse supervision, we withhold the transaction labels of five AML datasets from TRACK throughout training. Under a time-ordered protocol, TRACK achieves the highest average precision a
    
[^123]: 基于软标签的贝叶斯最优公平性-准确性权衡的实用估计

    Practical Estimation of the Bayes Optimal Fairness-Accuracy Tradeoff with Soft Labels

    [https://arxiv.org/abs/2611.06027](https://arxiv.org/abs/2611.06027)

    该论文将仅基于软标签的贝叶斯误差估计技术从标准二分类扩展到公平分类等受约束分类问题，实现了对贝叶斯最优公平性-准确性权衡（帕累托前沿）的实用估计。

    

    对于给定的数据分布，任何模型所能达到的最佳预测误差都存在一个根本性的极限。在分类任务中，贝叶斯误差量化了这一极限，它既可以作为训练模型的性能基准，也可以作为检测过拟合的标准。目前最先进的二分类贝叶斯误差估计技术仅使用软标签（即正类概率），而无需输入特征或辅助训练。我们将这些概念扩展到受约束的分类问题，例如公平性。公平分类是受约束或多目标分类的一个重要案例，它要求模型不仅要准确，还要在不同人口群体（如种族和性别）之间产生公平的结果（即相等或接近相等的真阳性率）。这导致了固有的公平性-准确性权衡，即帕累托前沿，用于对各个公平分类器进行相互比较。我们工作的重点是将（原文在此处截断）

    arXiv:2611.06027v1 Announce Type: cross  Abstract: There is a fundamental limit to the best prediction error of any model for a given data distribution. In classification, the Bayes error quantifies this limit and serves as both a benchmark for trained models and a criterion to detect overfitting. The state-of-the-art Bayes error estimation techniques for binary classification use only soft labels (positive class probabilities) and require no input features or auxiliary training. We extend such notions to constrained classification problems, such as fairness. Fair classification is an important case of constrained or multi-objective classification. It requires a model not only to be accurate but also to yield fair outcomes (equal or near-equal true positive rates) across different demographic groups (e.g., race and gender). This leads to an inherent fairness-accuracy tradeoff or a Pareto frontier that is used to compare fair classifiers against each other. The focus of our work is to e
    
[^124]: 谁为模型复制的知识买单？作为罗杰斯悖论的模型坍塌

    Who Pays for the Knowledge Models Copy? Model Collapse as Rogers' Paradox

    [https://arxiv.org/abs/2611.06022](https://arxiv.org/abs/2611.06022)

    该论文将模型坍塌重新解读为文化演化中的罗杰斯悖论，指出问题的根源在于外部信息供给本身，并证明仅用完全合成数据拟合的罗杰斯模型能精确预测语言模型在不同新鲜数据混合比例下稀有知识的保留量。

    

    在先前模型输出上训练的生成模型会在代际间丢失知识，且稀有知识最先丢失。行之有效的补救措施都需要从循环外部引入信息，而现有的模型坍塌理论将这些信息供给视为既定条件：一个来自固定分布的固定档案库。我们认为供给本身才是问题的根源，而文化演化早已拥有关于这一问题的理论。我们将模型坍塌解读为罗杰斯悖论：一个逐渐依赖复制的人口最终并不比从不复制的人口境况更好，因为复制品会过时，除非有人持续进行自主学习。我们通过三个步骤检验这一论证。第一，仅用完全合成训练数据拟合的罗杰斯模型，能够以几个百分点以内的误差，预测语言模型——从词三元组模型到预训练Transformer——在其他新鲜数据混合比例下保留多少稀有知识。第二，当世界本身发生变化时，复制自身过往的谱系……（原文摘要在此处截断）

    arXiv:2611.06022v1 Announce Type: cross  Abstract: Generative models trained on the output of earlier models lose knowledge across generations, rare knowledge first. The remedies that work all bring in information from outside the loop, and the theory of model collapse treats that supply as given: a fixed archive, drawn from a fixed distribution. We argue that the supply is the problem, and that cultural evolution has the theory of it. We read model collapse as Rogers' paradox: a population that comes to rely on copying ends up no better off than one that never copied, because copies go stale unless someone keeps learning for themselves. We test the identification in three steps. First, Rogers' model, fitted only to fully synthetic training, predicts to within a few percentage points how much rare knowledge language models, from word trigrams to a pretrained transformer, retain under other mixtures of fresh data. Second, when the world itself changes, a lineage that copies its own past
    
[^125]: 分布外检测中的吸引-排序失配：来自哈密顿特征动力学的理论与证据

    Attraction-Ranking Mismatch in Out-of-Distribution Detection: Theory and Evidence from Hamiltonian Feature Dynamics

    [https://arxiv.org/abs/2611.06021](https://arxiv.org/abs/2611.06021)

    该论文利用哈密顿特征动力学揭示了分布外检测中的“吸引-排序失配”现象：提升类亲和度分数并不必然改善ID与OOD样本的相对排序，且在源域验证集上有效的分数对齐规则在迁移后反而会损害检测性能。

    

    将特征吸引至已学习的分布内（ID）区域是分布外（OOD）检测的一种直观策略，但这样的吸引是否真能改善检测效果？我们以类条件哈密顿动力学作为受控测试平台来研究这一问题，并识别出一种“吸引-排序失配”现象：提升单个类的亲和度分数并不必然改善ID样本与OOD样本之间的相对排序。我们的分析将驱动场与检测分数之间的失调，与即使两者对齐后依然存在的排序失败区分开来。通过精确的AUROC分解和局部边界分析，我们在正则性条件下确定了ID-OOD排序边界处的相对分数变化是一阶AUROC变化的来源。实验表明，一个分数对齐的规则可以在留出的源域验证集上提升性能，但在迁移后却会持续降低性能，即使没有一个获胜类……（原文在此处截断）

    arXiv:2611.06021v1 Announce Type: new  Abstract: Attracting features toward learned in-distribution (ID) regions is an intuitive strategy for out-of-distribution (OOD) detection, but does such attraction improve detection? We investigate this question using class-conditioned Hamiltonian dynamics as a controlled testbed and identify an attraction-ranking mismatch: improving individual class-affinity scores need not improve the relative ordering of ID and OOD samples. Our analysis separates misalignment between the driving field and the detection score from ranking failure that persists even when they are aligned. An exact AUROC decomposition and a local boundary analysis identify relative score changes at the ID-OOD ranking boundary as the source of first-order AUROC change under regularity conditions. Experiments demonstrate that a score-aligned rule can improve performance on a held-out source validation split yet consistently degrade it after transfer, even without a single winning-c
    
[^126]: 你能引导多远？对比激活添加中的行为-能力前沿

    How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition

    [https://arxiv.org/abs/2611.06001](https://arxiv.org/abs/2611.06001)

    该论文提出“行为-能力前沿”框架来评估对比激活添加（CAA）的可用性，在能力退化明确预算约束下最大化行为收益，发现行为效果具有强泛化性（ρ=0.945），且将能力损失预算从1%提高到3%可使干预的联合存活率从37.5%显著提升至72.7%。

    

    激活引导可以在不更新参数的情况下修改语言模型的行为，但仅凭行为效果并不能决定一项干预措施是否可用。对比激活添加（CAA）利用从对比样本中提取的激活方向来引导目标行为，但在此过程中可能会损害模型的其他无关能力。因此，我们通过行为-能力前沿来研究CAA，在明确的能力退化界限约束下，选择能够最大化行为收益的干预措施。我们在四个指令微调模型、七种行为以及1%、3%和5%（绝对百分点）的能力损失预算下对该框架进行了评估。结果显示，基于验证集筛选出的行为效果能够很好地泛化到留出数据上（ρ = 0.945）；而在非零工作点中，行为与能力同时达标的联合存活率从1%预算下的37.5%上升到3%预算下的72.7%，在5%预算下的额外改进则较为有限。

    arXiv:2611.06001v1 Announce Type: cross  Abstract: Activation steering can modify language-model behavior without parameter updates, but behavioral efficacy alone does not determine whether an intervention is usable. Contrastive Activation Addition (CAA) steers targeted behaviors using activation directions derived from contrasting examples, potentially degrading unrelated model capabilities in the process. We therefore study CAA through a behavior-capability frontier, selecting interventions that maximize behavioral gain subject to an explicit bound on capability degradation. We evaluate this framework across four instruction-tuned models, seven behaviors, and capability-loss budgets of 1%, 3%, and 5% (absolute percentage points). Validation-selected behavioral effects generalize strongly to held-out data ($\rho = 0.945$), while joint behavior-capability survival among nonzero operating points rises from 37.5% at the 1% budget to 72.7% at 3%, with limited additional improvement at 5%.
    
[^127]: FedIA：面向联邦提示学习的实例感知跨模态表示

    FedIA: Instance-Aware Cross-Modal Representations for Federated Prompt Learning

    [https://arxiv.org/abs/2611.05994](https://arxiv.org/abs/2611.05994)

    该论文提出FedIA框架，通过学习实例感知的跨模态表示，使联邦提示学习能够针对每个输入自适应地平衡全局共享知识与个性化知识，克服了现有方法仅在客户端层面进行个性化而忽略样本间视觉差异的局限。

    

    联邦提示学习能够在不交换原始数据的情况下，实现对预训练视觉-语言模型在去中心化客户端上的参数高效适配。为了应对跨客户端的异构性，现有的联邦提示学习方法通常将一个全局共享提示与一个客户端特定的本地提示配对使用。然而，这种设计仅在客户端层面实现个性化：同一个本地提示被应用于该客户端的所有样本，并且全局提示与本地提示以一种与输入无关的固定方式组合。因此，这种方法忽略了同一客户端内样本之间显著的视觉差异，并且无法针对每个输入调整共享知识与个性化知识之间的平衡。为了解决这些局限性，我们提出了基于实例感知跨模态表示的联邦提示学习，这是一个学习实例感知跨模态表示的联邦提示学习框架。具体而言，FedIA首先使用全局文本提示作为客户端（摘要在此处被截断）

    arXiv:2611.05994v1 Announce Type: cross  Abstract: Federated Prompt Learning (FPL) enables parameter-efficient adaptation of pre-trained Vision-Language Models across decentralized clients without exchanging raw data. To cope with cross-client heterogeneity, existing FPL methods typically pair a globally shared prompt with a client-specific local prompt. However, this design personalizes only at the client level: the same local prompt is applied to all samples of a client, and the global and local prompts are combined in an input-agnostic, fixed manner. Consequently, it overlooks the substantial visual variations among samples within the same client and cannot adjust the balance between shared and personalized knowledge to each input. To address these limitations, we propose Federated Prompt Learning with Instance-Aware Cross-Modal Representations (FedIA), an FPL framework that learns instance-aware cross-modal representations. Specifically, FedIA first uses global text prompts as clie
    
[^128]: 解耦以复用：面向时间序列基础模型的细粒度知识增强

    Decouple to Reuse: Fine-Grained Knowledge Augmentation for Time Series Foundation Models

    [https://arxiv.org/abs/2611.05993](https://arxiv.org/abs/2611.05993)

    提出一种两阶段细粒度知识增强框架，通过将预训练知识解耦为趋势和季节性双分支残差码本并动态检索融合，以应对领域偏移、增强冻结的时间序列基础模型。

    

    时间序列预测对于现代Web服务中的需求预测和资源管理至关重要，然而由于领域偏移的存在，将预训练的时间序列基础模型（TSFMs）应用于下游任务仍然充满挑战。现有的检索增强方法通常检索粗粒度的窗口上下文，可能引入与上下文相关的噪声。为解决这一问题，我们认为时间模式应当在分解层级上被复用，因为有效的趋势和季节性动态可能源自不同的上下文。我们提出了一个两阶段细粒度知识增强框架，将预训练知识解耦并整合到紧凑的、任务共享的残差码本中，以增强冻结的TSFMs。具体而言，阶段I构建了用于解耦趋势和季节性信号的双分支残差码本，阶段II则动态检索并合并这些细粒度知识（摘要内容在此处被截断）。

    arXiv:2611.05993v1 Announce Type: new  Abstract: Time series forecasting is critical for demand prediction and resource management in modern Web services, yet applying pretrained Time Series Foundation Models (TSFMs) to downstream tasks remains challenging due to domain shifts. Existing retrieval-augmented approaches typically retrieve coarse-grained window contexts and may introduce context-specific noise. To address this problem, we argue that temporal patterns should be reused at a decomposition level, as effective trend and seasonal dynamics may originate from different contexts. We propose \method, a two-stage \underline{F}ine-grained \underline{K}nowledge \underline{A}ugmentation framework that decouples and consolidates pre-training knowledge into compact, task-shared residual codebooks to enhance frozen TSFMs. Specifically, Stage I constructs dual-branch residual codebooks for decoupled trend and seasonal signals, while Stage II dynamically retrieves and merges these fine-grain
    
[^129]: Clips2Story：基于多模态检索嵌入式生成的免训练视频故事板制作与编辑

    Clips2Story: Training-Free Video Storyboarding and Editing Using Multimodal Retrieval-Embedded Generation

    [https://arxiv.org/abs/2611.05992](https://arxiv.org/abs/2611.05992)

    Clips2Story提出了一种无需训练的LLM流水线，通过多模态描述符检索与关键词相关的视频片段，并自动编排成带有旁白叙述的可编辑故事板和连贯短视频。

    

    视频编辑人员通常需要处理大量碎片化的素材集合，并必须从中选择和组织相关片段以形成连贯的故事。然而，许多现有的视频编辑和摘要方法都假设源视频在时间上是有序的，并旨在保留其原始的叙事和时间结构。在本文中，我们提出了Clips2Story，这是一个无需训练、基于大语言模型（LLM）的流水线，使用户能够从视频片段池中创建指定关键词的短视频。给定一个片段池和用户指定的关键词，我们首先为每个片段提取基于文本的多模态描述符，并检索与关键词最相关的片段。然后，我们将这些候选片段的文本描述符提供给LLM，由其选择并排列这些片段，形成带有连贯旁白叙述的可编辑故事板。最后，我们组装所选片段，并为旁白检索配套的视觉画面，以合成输出视频。为了评估所提出的方法……

    arXiv:2611.05992v1 Announce Type: cross  Abstract: Video editors often work with large collections of fragmented footage and must select and organize relevant clips into a coherent story. However, many existing video editing and summarization methods assume a temporally ordered source video and aim to preserve its original narrative and temporal structure. In this paper, we introduce Clips2Story, a training-free, LLM-based pipeline that enables users to create keyword-specified short videos from a pool of video clips. Given a pool of clips and a user-specified keyword, we first extract text-based multimodal descriptors for each clip and retrieve the clips most relevant to the keyword. We then provide the textual descriptors of these candidate clips to an LLM, which selects and arranges them into an editable storyboard with connective narrations. Finally, we assemble the selected clips and retrieve accompanying visuals for the narrations to compose the output video. To evaluate the prop
    
[^130]: LLM 服务引擎中配置与缓存架构的受控实验评估

    A Controlled Experimental Evaluation of Configuration and Cache Architecture in LLM Serving Engines

    [https://arxiv.org/abs/2611.05991](https://arxiv.org/abs/2611.05991)

    该研究通过受控实验发现，vLLM 与 SGLang 服务引擎之间的巨大性能差距主要源于分块预填充的默认配置差异而非缓存架构本身，仅对齐这一配置即可消除大部分差距。

    

    多智能体 LLM 应用在多个请求之间重复使用大部分提示词 token，服务引擎会复用共享前缀的缓存状态。vLLM 和 SGLang 以不同的粒度实现复用，针对两者默认部署的基准测试报告了巨大的性能差距，通常将其归因于缓存。然而这类比较很少控制其他配置差异。其中一个差异是长提示词是否采用分块预填充，该设置由各引擎根据硬件和版本的默认值决定，从而混淆了比较结果。我们在 Kubernetes 上构建了一个可复现的测试平台和一个带种子的多智能体负载生成器，在厂商默认设置和分块预填充对齐两种条件下对两个引擎进行了比较。SGLang 的配置在两组实验中保持一致，这使其成为跨独立部署环境之间的内部对照。在默认设置下，vLLM 的中位首 token 延迟最高可达 SGLang 的 64 倍。仅对齐这一项设置即可消除 78% 的差距。

    arXiv:2611.05991v1 Announce Type: cross  Abstract: Multi-agent LLM applications repeat most of their prompt tokens across requests, and serving engines reuse the cached state of shared prefixes. vLLM and SGLang reuse it at different granularities, and benchmarks of their default deployments report large performance gaps, usually credited to the cache. Such comparisons rarely control for other configuration differences. One of them, whether a long prompt is prefilled in chunks, is set by each engine's defaults from the hardware and the release, and it confounds the comparison. We built a reproducible testbed on Kubernetes and a seeded multi-agent workload generator, and compared the two engines under vendor defaults and with chunked prefill matched. SGLang's configuration is the same in both arms, which makes it an internal control across independently provisioned deployments. Under defaults, vLLM's median time to first token is up to 64x SGLang's. Matching one setting removes 78% of th
    
[^131]: 关系数据库深度学习中列重要性归因方法的比较研究

    A Comparative Study of Importance Attribution to Columns in Deep Learning over Relational Databases

    [https://arxiv.org/abs/2611.05986](https://arxiv.org/abs/2611.05986)

    该论文提出了一个系统比较关系数据库深度学习中列重要性归因方法的研究框架，从属性排名和得分分布两个维度分析各方法的异同，并考察这些差异是否反映在传统的保真度指标中。

    

    基于关系数据库的深度学习催生了日益有效的预测模型，也激发了对解释这些模型行为的需求。我们研究了一类为单个数据库属性分配重要性得分的解释方法，并聚焦于一个根本问题：重要性归因方法的选择在多大程度上会产生影响？我们开发了一套方法论来系统地比较各种重要性归因方法，并分析它们之间的异同。具体而言，我们从两个方面研究这些解释：一是它们认为哪些属性是重要的（通过属性排名的差异来刻画）；二是重要性得分在不同属性之间如何变化（通过得分分布的差异来刻画）。我们进一步探究了这些差异是否能够体现在解释质量的常规度量——保真度中。我们将该方法论应用于对四种方法（蒙特卡洛 Shapley、Kernel SHAP 等）的大规模实证研究。

    arXiv:2611.05986v1 Announce Type: new  Abstract: Deep learning over relational databases has led to increasingly effective prediction models, motivating a growing need to explain their behavior. We study explanation methods that assign importance scores to individual database attributes, focusing on a fundamental question: to what extent does the choice of importance-attribution method matter? We develop a methodology to systematically compare importance-attribution methods and analyze their similarities and differences. In particular, we study two aspects of the explanations: which attributes they deem important, captured by differences in attribute rankings, and how importance scores vary across attributes, captured by differences in score distributions. We further investigate whether these differences are reflected in the conventional measure of explanation quality, fidelity.   We apply our methodology in an extensive empirical study of four methods: Monte Carlo Shapley, Kernel SHAP
    
[^132]: 关于超连接的设计：简化、稳定性与可扩展性

    On the Design of Hyper-Connections: Simplification, Stability, and Scalability

    [https://arxiv.org/abs/2611.05984](https://arxiv.org/abs/2611.05984)

    该论文系统剖析了超连接（HC）架构，发现门控与残差混合可解耦为两个各自独立有效的交互组，并据此提出简化架构SimpleHC，通过移除残差混合和共享内存访问机制，在保持与mHC相当性能的同时显著降低训练开销。

    

    超连接（Hyper-Connections, HC）架构正越来越多地被前沿语言模型所采用，然而其组件仍未得到充分理解，并且会引入不可忽视的训练开销。在这项工作中，我们系统性地研究了HC架构，以理解其组件并找到更简单、更高效的设计。我们发现，不带混合的可学习读写门控与带固定门控的无约束混合构成了两个有效的交互组：单独使用其中任何一个即可达到与mHC相当的性能。我们还发现，对扩展残差状态的重复访问是训练开销的主要来源，即使在没有混合的情况下也是如此。这些发现催生了SimpleHC，它通过移除残差混合来简化架构，并通过共享内存访问来简化执行，利用由此产生的加性结构和共享门控输入。特定于层的门控共享一个组入口状态，从而支持批量读取和延迟写入……

    arXiv:2611.05984v1 Announce Type: cross  Abstract: Hyper-Connections (HC) architectures are increasingly adopted in frontier language models, yet their components remain insufficiently understood and introduce non-negligible training overhead. In this work, we systematically study the HC architecture to understand its components and identify simpler, more efficient designs. We find that learned read/write gating without mixing and unconstrained mixing with fixed gates form two effective interaction groups: either alone achieves performance comparable to mHC. We also identify repeated access to the expanded residual state as a major source of training overhead, even without mixing. These findings lead to SimpleHC, which simplifies the architecture by removing residual mixing and simplifies execution through shared memory access, using the resulting additive structure and shared gate inputs. Layer-specific gates share a group-entry state, enabling batched reads and deferred writes; scala
    
[^133]: 奖励预算化：减少大语言模型推理强化学习中的过早收敛

    Reward Budgeting Reduces Premature Convergence in Reinforcement Learning for LLM Reasoning

    [https://arxiv.org/abs/2611.05942](https://arxiv.org/abs/2611.05942)

    该论文提出“奖励预算化”这一插件式优势构造方法，通过为每个提示分配固定的总奖励预算来缩小优势差距，从而有效缓解大语言模型推理强化学习中的过早收敛与早期性能平台期问题。

    

    面向大语言模型（LLM）推理的强化学习（RL）常常随着策略分布变得越来越尖锐而过早进入性能平台期。近期研究将策略熵作为这一行为的诊断信号和控制信号。然而我们的实验表明，仅仅增加熵并不能可靠地改变最终的平台期。我们通过RL的隐式目标策略对其进行分析，推导出一个尖锐度因子 $\kappa$ 及其有效对应量 $\kappa_{\mathrm{eff}}$，用以刻画奖励或优势差距被转化为概率比差距的强度。基于这一视角，我们提出了“奖励预算化”，这是一种插件式的优势构造方法，为每个提示分配固定的总奖励预算。对于二元奖励和独立的on-policy采样，其未截断的得分函数估计器遵循一个精确的、具有收益递减特性的有限-$G$ 凹目标的梯度，而其优势差距随着更多采样的进行而减小……

    arXiv:2611.05942v1 Announce Type: new  Abstract: RL for LLM reasoning often reaches an early performance plateau as the policy distribution becomes increasingly sharp. Recent work has studied policy entropy as both a diagnostic and a control signal for this behavior. In our experiments, however, increasing entropy alone does not reliably change the eventual plateau. We analyze RL through its implicit target policy and derive a sharpness factor $\kappa$, together with an effective counterpart $\kappa_{\mathrm{eff}}$, that describes how strongly reward or advantage gaps are translated into probability-ratio gaps. Motivated by this view, we introduce \emph{reward budgeting}, a plug-in advantage construction that assigns a fixed total reward budget to each prompt. For binary rewards and independent on-policy rollouts, its unclipped score-function estimator follows the gradient of an exact finite-$G$ concave objective with diminishing returns, while its advantage gap decreases as more rollo
    
[^134]: BAM：用于可解释表格预测的二元可加模型

    BAM: Bivariate Additive Models for Interpretable Tabular Prediction

    [https://arxiv.org/abs/2611.05940](https://arxiv.org/abs/2611.05940)

    提出 Bivariate Additive Models（BAM），将 GA²M 中的每个成对形状函数约束为非对称二元高斯的简单参数形式，在保持与标准 GAM 相当精度的同时，使每个交互项都可可视化为单峰等高线图并自动选择特征对，从而实现真正可解释、可审计的表格预测。

    

    具有成对交互作用的广义可加模型（GA²M）被视为玻璃箱模型，因为每个成对形状函数都可以可视化为等高线图。但在实践中，EBM 或 NODE-GAM 等方法学习到的形状函数过于复杂（等高线不规则且局部波动剧烈），以至于从业者无法对其进行有意义的解释，也无法对照领域知识加以检验。这种更强意义上的可审计性需要显著更简单的形状函数。我们认为，只要捕捉到正确的特征对，这样的简化可以达到与标准 GAM 相当的精度。我们提出二元可加模型，这是一类 GA²M，其中每个成对形状函数都被约束为简单的参数形式，具体实例化为非对称二元高斯分布。每个组件都可以可视化为单峰等高线图，并且其特征对在训练过程中自动选择。在六个表格数据集上的实验……

    arXiv:2611.05940v1 Announce Type: cross  Abstract: Generalized Additive Models with pairwise interactions (GA$^2$Ms) are regarded as glass-box models, since each pairwise shape function can be visualized as a contour plot. In practice, the shape functions learned by methods such as EBM or NODE-GAM are so complex (irregular contours and sharp local fluctuations) that a practitioner cannot meaningfully interpret them or check them against domain knowledge. Auditability in this stronger sense requires substantially simpler shape functions. We argue that such simplification can match the accuracy of standard GAMs if the right feature pairs are captured. We propose Bivariate Additive Models (BAMs), a class of GA$^2$Ms in which every pairwise shape function is constrained to a simple parametric form, instantiated as an asymmetric bivariate Gaussian. Each component is visualizable as a unimodal contour map, and its feature pair is selected automatically during training. Experiments on six tab
    
[^135]: 深度与经典的结合：一种CPU高效的自动调制分类决策融合模型

    Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification

    [https://arxiv.org/abs/2611.05937](https://arxiv.org/abs/2611.05937)

    本文提出FuseAMC轻量级决策融合模型，利用CNN在低信噪比下更鲁棒、高阶累积量分类器在高信噪比下更准确的互补特性，通过逻辑回归元组合器融合两者输出，实现了CPU高效的自动调制分类。

    

    自动调制分类（AMC）是认知无线电中频谱感知和非协作信号识别的基础技术，然而其两种标准的CPU可行方法——基于原始同相/正交（I/Q）样本的紧凑卷积神经网络（CNN）和经典的高阶累积量特征分类器——很少在信噪比（SNR）条件下被相互比较。我们证明这两种方法实际上是互补的：CNN在低信噪比下明显更加鲁棒，此时手工设计的统计特征已接近随机猜测水平；而累积量分类器在高信噪比下更加准确，此时高阶统计量能够清晰地区分密集星座。我们利用这种互补性提出了FuseAMC，这是一种轻量级的决策融合模型，它在约19.8万参数的残差CNN和梯度提升树累积量分类器的拼接类后验之上，训练一个逻辑回归元组合器。

    arXiv:2611.05937v1 Announce Type: cross  Abstract: Automatic modulation classification (AMC) underpins spectrum sensing and non-cooperative signal identification in cognitive radio, yet the two standard CPU-feasible approaches to it, a compact convolutional neural network (CNN) on raw in-phase and quadrature (I/Q) samples and a classical higher-order-cumulant feature classifier, are rarely compared against each other conditioned on signal-to-noise ratio (SNR). We show that they are, in fact, complementary: the CNN is markedly more robust at low SNR, where hand-crafted statistics collapse toward chance, while the cumulant classifier is more accurate at high SNR, where higher-order statistics cleanly separate dense constellations. We exploit this complementarity with FuseAMC, a lightweight decision-fusion model that trains a logistic-regression meta-combiner over the concatenated class-posteriors of an approximately 198k-parameter residual CNN and a gradient-boosted-tree cumulant classif
    
[^136]: 基于统计杠杆分数的多样性感知经验回放

    Diversity-Aware Experience Replay via Statistical Leverage Scores

    [https://arxiv.org/abs/2611.05929](https://arxiv.org/abs/2611.05929)

    提出利用无需标签、奖励或价值网络的统计杠杆分数对经验回放缓冲区中的转移进行多样性感知采样，并针对独热动作编码导致方差评分失效的问题，采用基于峰度的特征列选择方法，从而优先采样稀有事件。

    

    离策略深度强化学习从存储过去经验的经验回放缓冲区中随机采样进行训练。在实体机器人场景中，该缓冲区被常规数据所主导：来自正常运行的数千条几乎完全相同的转移，而定义任务边界的稀有事件却很少被采样到。我们提出用统计杠杆分数为每条存储的转移进行评分，该分数衡量转移相对于缓冲区自身数据的不寻常程度，仅需根据存储的数值计算，无需标签、奖励或价值网络。我们从这个分数与均匀采样构成的混合分布中抽取每个小批量。首先，沿着数据变化最大的方向构建分数的自然方法，在动作用独热指示符存储时会失效，因为指示符列总能在方差排名中胜出；为此，我们转而选择具有最重尾部的特征列（最大峰度），在这些列中，较小的远端数据群体能够被显现出来……

    arXiv:2611.05929v1 Announce Type: cross  Abstract: Off-policy deep reinforcement learning trains on random samples from a replay buffer of past experience. In physical robot settings, that buffer is dominated by routine: thousands of near-identical transitions from normal operation, while the rare events that define the edges of the task are rarely sampled. We propose scoring every stored transition by its statistical leverage, which measures how unusual it is relative to the buffer's own data, computed from the stored numbers alone, with no labels, rewards, or value network. We draw each mini-batch from a mixture of this score and uniform sampling. First, the natural way to build the score along the directions where the data varies most turns out to be blind when actions are stored as one-hot indicators, because indicator columns always win the variance ranking; we instead select the feature columns with the heaviest tails (largest kurtosis) where small far-away populations are reveal
    
[^137]: 推理之路：读取与重塑MoE模型中的测试时计算

    Routes of Reasoning: Reading and Reshaping Test-Time Compute in MoE Models

    [https://arxiv.org/abs/2611.05919](https://arxiv.org/abs/2611.05919)

    该论文发现MoE模型的router logits可以作为天然的、无需训练的行为控制接口，通过六类推理行为的标注与探针读取，实现对“过度思考”等特定推理行为的精准调控，从而节约测试时计算并提升性能。

    

    推理模型会将测试时计算分配给不同的行为，例如规划、推导和验证。然而，模型经常“过度思考”，在找到正确答案后很长时间仍继续验证和重新推导。这不仅浪费了昂贵的计算资源，甚至可能降低性能。现有的干预方法，如token预算或引导，主要调节推理时长，而非针对特定行为。在这项工作中，我们证明了混合专家模型的router提供了一种天然的、无需训练的行为控制接口。我们首先将推理轨迹切分为句子级片段，并用六种推理行为（规划、执行、复述、事实、反思和结论）之一对每个片段进行标注，使每个token都携带其所属片段的行为标签。作用于router logits上的线性探针预测该标签的效果，几乎与作用于残差流（即模型完整隐藏状态这一标准目标）上的探针相当。

    arXiv:2611.05919v1 Announce Type: new  Abstract: Reasoning models allocate test-time compute across distinct behaviors such as planning, deriving, and verifying. However, models frequently "overthink," continuing to verify and re-derive long after finding a correct answer. This wastes expensive compute and can even degrade performance. Existing interventions, such as token budgets or steering, primarily regulate reasoning duration rather than targeting specific behaviors. In this work, we demonstrate that Mixture-of-Experts (MoE) routers provide a natural, training-free interface for behavioral control. We first split reasoning traces into sentence-level segments and label each with one of six reasoning behaviors (planning, execution, restatement, fact, reflection, and conclusion), so that every token carries the behavior of its segment. Linear probes on router logits predict this label nearly as well as probes on the residual stream, the model's full hidden state and the standard targ
    
[^138]: 带容量约束的聚类编辑问题的改进近似算法

    Improved approximation algorithms for capacitated cluster editing

    [https://arxiv.org/abs/2611.05916](https://arxiv.org/abs/2611.05916)

    本文针对带容量约束的聚类编辑问题，提出了一种缩减程序，将最佳近似比从 5.37 改进到 $\alpha_k = 2\alpha + 1$（约为 3.773）。

    

    我们研究最小分歧聚类编辑问题：给定一个完全边符号图，其中正边表示两个顶点彼此一致，负边表示分歧，该问题构造大小有上界约束的不相交团，使得团内部的负边数量与团外部的正边数量之和最小。当候选团的大小无上界时，该问题是NP难且APX难的，文献中已提出了许多近似比不断改进的近似算法，目前最先进的近似比为 $\alpha = 1.3865 + \epsilon$。带容量约束的变体——即候选团的大小不超过给定的 $k$——研究较少，目前最好的近似比为 5.37。在本文中，我们研究后一个问题。我们首先提出了一种缩减程序，实现了近似比 $\alpha_k = 2\alpha + 1$，基于

    arXiv:2611.05916v1 Announce Type: cross  Abstract: We study the minimum-disagreement cluster editing problem that, from a complete edge-signed graph, where a positive edge represents two vertices agreeing to each other while a negative represents a disagreement, constructs disjoint cliques of upper bounded size such that the total number of negative edges inside cliques and positive edges outside cliques is minimized. When the sizes of the candidate cliques are unbounded, the problem is NP-hard and APX-hard, and many approximation algorithms with improving ratios have been proposed in the literature and the state-of-the-art approximation ratio of $\alpha = 1.3865 + \epsilon$. The capacitated variant, in which the candidate cliques have sizes no greater than a given $k$, is less studied and the best approximation ratio is $5.37$. In this paper, we study the latter problem. We first present a downsizing procedure that achieves an approximation ratio of $\alpha_k = 2 \alpha + 1$, based on
    
[^139]: 你的试验模拟器知道药物的作用吗？基于7,207项注册试验的干预盲法干预效度检验

    Does Your Trial Simulator Know What the Drug Does? An Intervention-Blind Test of Interventional Validity on 7,207 Registered Trials

    [https://arxiv.org/abs/2611.05907](https://arxiv.org/abs/2611.05907)

    该论文首次将临床试验模拟器的“反事实能力”从主观论断转化为可测量的量化指标，通过构建隐去干预信息的干预盲参照模型，并基于7,207项注册试验构建的9,996个审计对比，将模拟器的干预技能定义为完整模型相对于干预盲参照所消除的误差。

    

    患者世界模型正日益被提出用作临床试验的模拟器，近期的综述将其能力按从时序预测到反事实推演的能力阶梯进行排序。目前阶梯上的位置是通过阅读论文而非实际测量来确定的。我们将反事实这一层级转化为可测量的量。从30,119项已完成并将结果发布至ClinicalTrials.gov的2/3期试验中，我们从7,207项试验构建了9,996个经过审计的试验内对比，并评估模拟器的准确率在多大程度上依赖于知晓所测试的是哪种干预。该测量工具是一个干预盲参照：即同一模型、在相同的试验上训练，但隐去所有能够识别干预的字段。干预技能定义为完整模型相对于该参照所消除的误差。在时间划分上（来自655项试验的878个测试对比），一个基于注册数据训练的梯度提升模拟器消除了五分之一的……

    arXiv:2611.05907v1 Announce Type: cross  Abstract: Patient world models are increasingly proposed as simulators of clinical trials, and recent surveys rank their capabilities on a ladder from temporal prediction to counterfactual rollouts. Position on that ladder is currently assigned by reading papers rather than by measurement. We turn the counterfactual rung into a measured quantity. From 30,119 completed phase 2/3 trials with results posted to ClinicalTrials.gov we construct 9,996 audited within-trial contrasts from 7,207 trials, and we score a simulator on how much of its accuracy depends on knowing which intervention was tested. The instrument is an intervention-blind reference: the same model, trained on the same trials, with every intervention-identifying field withheld. Interventional skill is the error the full model removes relative to that reference. On a temporal split (878 test contrasts from 655 trials) a registry-trained gradient-boosting simulator removes a fifth of th
    
[^140]: 检索超越规模：基于多重社会技术上下文的高效日志级别预测

    Retrieval Supersedes Scale: Efficient Log Level Prediction via Multiplex Socio-Technical Context

    [https://arxiv.org/abs/2611.05901](https://arxiv.org/abs/2611.05901)

    该论文提出基于多重社会技术上下文的检索方法来选择上下文学习示例，从而在不微调的情况下显著提升 LLM 的日志级别预测效果，证明了检索策略优于单纯依赖模型规模。

    

    开发者在源代码中插入日志语句以记录程序执行过程，由此产生的日志对于检测和排查系统异常至关重要。这些语句的关键组成部分是日志级别（如 DEBUG、INFO、WARN、ERROR），它决定了消息是否被记录。选择日志级别是一种权衡：级别设置过低会掩盖生产环境中的重要信息，而级别设置过高则会产生冗余日志并带来额外的 I/O 和存储开销。近期研究采用大语言模型（LLM）结合上下文学习（ICL）来预测日志级别，但通常使用随机采样的示例来扩充提示词。我们认为这种做法忽视了日志编写的社会技术属性，即日志约定会因组件和开发团队的不同而存在差异。据报道，使用随机示例的现成 LLM 仅能达到 0.56 到 0.76 的中位 AUC，这表明无需微调仍有改进空间。在本文中，我们提出了一种检索方法……（原文摘要在此处截断）

    arXiv:2611.05901v1 Announce Type: cross  Abstract: Developers insert logging statements in source code to record its execution, and the resulting logs are essential for detecting and troubleshooting system anomalies. A key part of these statements is the log level (e.g., DEBUG, INFO, WARN, ERROR), which determines whether a message is recorded. Choosing it is a trade-off: levels set too low mask important information in production, while levels set too high create noisy logs and extra I/O and storage overhead. Recent work uses Large Language Models (LLMs) with In-Context Learning (ICL) to predict log levels, but typically augments the prompt with randomly sampled examples. We argue that this overlooks the socio-technical nature of logging, where conventions vary by component and development team. Off-the-shelf LLMs using random examples are reported to achieve median AUCs of only 0.56 to 0.76, leaving room for improvement without fine-tuning. In this paper, we propose a retrieval appro
    
[^141]: 对话情绪转变预测的信息匹配审计：一种防泄漏的评估协议

    Information-Matched Auditing of Conversational Emotion-Shift Forecasting: A Leakage-Safe Evaluation Protocol

    [https://arxiv.org/abs/2611.05895](https://arxiv.org/abs/2611.05895)

    该论文提出了一种信息匹配、防泄漏的对话情绪转变预测审计协议，通过明确的承诺点与可执行的因果性测试，揭示了可部署条件与先知式诊断条件会得出不同的评估结论。

    

    arXiv:2611.05895v1 公告类型：新论文 摘要：在对话情绪转变预测研究中，当学习型模型获得丰富的特征，而其转移基线却被给予含噪声或不可获得的当前情绪标签时，当超参数预算不相同时，或当目标轮次信息混入表示中时，比较结果很容易被扭曲。我们提出了一种信息匹配且防泄漏的审计协议，通过明确的承诺点、可执行的因果性测试、均等的四配置验证搜索以及感知种子对话的推理来消除这些差距。将该协议应用于四个文本语料库和两种特征源敏感性配置上的六个因果模型族，结果表明：当为转移基线提供一个仅基于训练数据且可部署的当前情绪分类器时，所得到的评估结论与双方均获得金标准当前情绪的先知式诊断所得出的结论不同。在可部署条件下，学习型模型在全部四个主要语料库上均优于预测标签基线（摘要在此处被截断）。

    arXiv:2611.05895v1 Announce Type: new  Abstract: Comparisons in conversational emotion-shift forecasting are easily distorted when a learned model receives rich features but its transition baseline is given a noisy or unavailable current-emotion label, when hyperparameter budgets differ, or when target-turn information enters the representation. We present an information-matched, leakage-safe auditing protocol that closes these gaps with an explicit commitment point, executable causality tests, equal four-configuration validation searches, and seed-dialogue-aware inference. Applying it to six causal model families on four text corpora and two feature-source sensitivity configurations shows that a deployable train-only current-emotion classifier feeding the transition baseline yields a different verdict than an oracle diagnostic where both sides receive gold current emotion. Under the deployable condition, learned models improve over the predicted-label baseline on all four primary corp
    
[^142]: 每个模板只需一个查询何时才足够？将学习型代价模型适配到新硬件

    When Is One Query per Template Enough? Adapting Learned Cost Models to New Hardware

    [https://arxiv.org/abs/2611.05892](https://arxiv.org/abs/2611.05892)

    研究发现不同硬件间查询计划运行时间的变化主要由查询模板差异决定，提出一种为串行与并行计划分别拟合线性修正、并为每个测量模板添加偏移量的方法，使学习型代价模型在每个模板仅需一次测量查询的情况下即可适配到新硬件。

    

    一个用于预测查询运行时间的学习型代价模型是在某台机器上的测量数据上训练的。当数据库迁移到其他硬件时，查询计划保持不变，但它们的运行时间会改变，且每个计划按各自不同的因子变化。我们探讨如何利用在新机器上测量的 k 个查询来适配这样一个模型：应该运行哪些查询、如何使用它们的运行时间，以及多少个查询才足够。在 PostgreSQL 上，我们在六种环境中对 1,162 个 TPC-H 和 TPC-DS 查询的相同计划进行了计时：一台笔记本电脑和一台服务器，各自使用全部处理器或固定到单个处理器，服务器还固定到两个处理器，笔记本电脑还使用了较小的缓冲池。没有单一因子能够描述两种环境之间的性能下降：其大部分方差位于查询模板之间，因此一个速度因子在几个查询之后就不再改进。一种为串行和并行计划分别拟合直线、并为每个测量过的模板添加偏移量的修正方法，使得迁移后的模型在每个模板只需一个测量查询的情况下

    arXiv:2611.05892v1 Announce Type: cross  Abstract: A learned cost model that predicts query runtimes is trained on measurements from one machine. When the database moves to other hardware, its plans stay the same but their runtimes change, each by its own factor. We ask how to adapt such a model from k queries measured on the new machine: which queries to run, how to use their runtimes, and how many are enough. On PostgreSQL we timed the same plans of 1,162 TPC-H and TPC-DS queries in six environments: a laptop and a server, each with all its processors or pinned to one, the server also pinned to two, and the laptop also with a small buffer pool. No single factor describes the slowdown between two environments: most of its variance lies between query templates, so a speed factor stops improving after a few queries. A correction that fits separate lines for serial and parallel plans and adds an offset for every measured template brings the moved model, with one measured query per templa
    
[^143]: MIGONet：一种采用无求解器离散制造解训练的多输入图神经算子

    MIGONet: a multiple-input graph neural operator with solver-free discrete manufactured-solution training

    [https://arxiv.org/abs/2611.05876](https://arxiv.org/abs/2611.05876)

    该论文提出了一种多输入图神经算子MIGONet，结合无需求解偏微分方程的离散制造解训练方法，能够捕获空间分布源项与狄利克雷/诺伊曼边界条件的联合影响，同时大幅降低训练数据生成成本。

    

    输运问题的代理建模需要能够捕获空间分布源项以及狄利克雷和诺伊曼边界条件共同影响的算子，同时需要能够以较低计算成本生成的训练数据集。本工作提出了MIGONet——一种多输入图神经算子，以及一种离散制造解方法（MMS），该方法以稳态热传导为模型问题，无需求解偏微分方程即可完成训练。在离散MMS过程中，温度场被先验地给定，源项和狄利克雷或诺伊曼边界数据则使用与参考有限差分方法（FDM）相同的离散算子进行评估，因此每个训练样本在构造上都满足离散控制方程。由于边界数据是从每个场中导出的，因此可以为每个样本的每个边界段分配狄利克雷或诺伊曼条件，并且……

    arXiv:2611.05876v1 Announce Type: new  Abstract: Surrogate modeling of transport requires operators that capture the combined influence of spatially distributed sources and Dirichlet and Neumann boundary conditions, together with training datasets that can be generated at low computational cost. This work presents MIGONet, a multiple-input graph neural operator, and a discrete method of manufactured solutions (MMS) that trains it without solving a partial differential equation, using steady heat conduction as the model problem. In the discrete MMS procedure, temperature fields are prescribed a priori, and the source terms and Dirichlet or Neumann boundary data are evaluated with the same discrete operators as the reference finite-difference method (FDM), so every training sample satisfies the discrete governing equations by construction. Because the boundary data are derived from each field, a Dirichlet or Neumann condition can be assigned to every boundary segment of every sample, and
    
[^144]: 相关性忽视与信任下的人工智能辅助分类

    AI-Assisted Classification under Correlation Neglect and Trust

    [https://arxiv.org/abs/2611.05870](https://arxiv.org/abs/2611.05870)

    该论文提出包含相关性忽视与AI信任两个要素的人类决策行为模型，通过闭式相图划分AI辅助决策的互补、损害与自动化区域，并证明更强的AI可能因人类重复计算重叠证据而降低其决策准确性，而适当低水平的信任反而可以补偿相关性忽视。

    

    AI助手何时能改善人类决策，何时又会有所损害？我们认为，答案不仅取决于AI的准确性，还取决于其证据与人类证据的重叠程度，以及人类如何将两者结合起来。决策者通过观察自身信号并接收AI的推荐来做出二元决策。我们构建了一个包含两个基本要素的人类行为模型：她对证据重叠的认知程度以及她对AI的信任程度。我们的分析给出了关于AI能力与证据重叠程度的闭式相图，其中划分出互补、损害与自动化三个区域。忽视证据重叠（即相关性忽视）的决策者，可能因为能力更强的AI反而降低决策准确性，原因是她重复计算了共享的证据。信任会重塑这些区域——信任不足可能补偿相关性忽视，因此使准确性最大化的信任水平取决于具体的部署参数。在实证部分，我们通过配对……识别这两个模型参数。

    arXiv:2611.05870v1 Announce Type: cross  Abstract: When does an AI assistant improve human decision-making, and when does it hurt? We argue that the answer depends not just on AI accuracy but also on how much its evidence overlaps with the human's and how the human combines the two. A worker makes a binary decision by observing her own signal and receiving a recommendation from an AI. We develop a human behavioral model with two primitives: her awareness of the overlap and her trust in the AI. Our analysis gives a closed-form phase diagram over AI capability and overlap, with regions of complementarity, impairment, and automation. A worker who neglects the overlap (correlation neglect) can be made less accurate by a more capable AI, double-counting the shared evidence. Trust reshapes the regions - undertrust may compensate for correlation neglect, so that the trust level maximizing accuracy depends on the deployment parameters. Empirically, we identify the two model parameters by pairi
    
[^145]: 关于带正则化修正与资格迹的梯度时序差分学习的收敛性研究

    On Convergence of Gradient Temporal-Difference Learning with Regularized Corrections and Eligibility Traces

    [https://arxiv.org/abs/2611.05869](https://arxiv.org/abs/2611.05869)

    本文首次建立了带资格迹的正则化修正时序差分学习TDRC(λ)在非独立同分布离策略数据下的收敛性理论，并提出TDRC(λ)-Huber算法，在保持TDRC(λ)不动点的同时降低对大幅贝尔曼误差的敏感性。

    

    时序差分（TD）学习是许多预测与控制算法的基础，但当其与离策略学习、自举和函数逼近相结合时可能会发散。带正则化修正的时序差分学习（TDRC）在理论上可靠且在实践中简单，然而现有的收敛性理论仅覆盖了独立同分布采样下的单步情形，且Huber贝尔曼误差形式尚未被扩展到TDRC或资格迹。在本工作中，我们建立了带递减步长和非独立同分布离策略数据下约束型TDRC(λ)的收敛性，适用于状态依赖和历史依赖的λ参数。我们还提出了TDRC(λ)-Huber算法，该算法在保持TDRC(λ)不动点的同时，降低了对大幅贝尔曼误差的敏感性。针对这两种算法，我们应用了随机逼近理论中基于常微分方程（ODE）的弱收敛方法。对于TDRC(λ)-Huber的非线性平均动力学，我们给出……

    arXiv:2611.05869v1 Announce Type: new  Abstract: Temporal-difference (TD) learning underlies many prediction and control algorithms, but can diverge when combined with off-policy learning, bootstrapping, and function approximation. TD with Regularized Corrections (TDRC) is theoretically sound and practically simple, but existing convergence theory covers only the one-step case under i.i.d. sampling, while the Huber Bellman-error formulation has not been extended to TDRC or eligibility traces. In this work, we establish convergence of constrained TDRC($\lambda$) with diminishing stepsizes and non-i.i.d. off-policy data, for state-dependent and history-dependent $\lambda$-parameters. We also propose TDRC($\lambda$)-Huber, which preserves the TDRC($\lambda$) fixed point while reducing sensitivity to large Bellman errors. For both algorithms, we apply an ODE-based weak-convergence method from stochastic approximation theory. For the nonlinear mean dynamics of TDRC($\lambda$)-Huber, we give
    
[^146]: 低秩解码器瓶颈限制了基础模型扰动预测的可靠性

    A low-rank decoder bottleneck bounds reliability in foundation-model perturbation prediction

    [https://arxiv.org/abs/2611.05849](https://arxiv.org/abs/2611.05849)

    该论文发现基础模型扰动预测中状态转移解码器的第一注意力层存在低秩瓶颈，其预测可靠性由预测响应与训练响应的距离所限定，并提出以“预测最近邻距离”作为逐条预测的置信度评分。

    

    基础模型的扰动响应预测器通常主要以总体准确率来评估。若要利用其预测结果来优先安排实验，就需要评估每一条预测的可靠性。我们对STATE中使用的状态转移（ST）解码器进行了审计，涵盖三个编码器（SE-600M、scGPT和Tahoe-x1）以及CRISPRi条件下的四种细胞系。该解码器的第一个注意力层具有低秩输出。将该输出投影到从训练激活中估计的4-32个固定方向上，即可保持每个扰动的平均皮尔逊delta相关性（PDC）。这一低秩注意力瓶颈使解码器的可靠性受到其与训练响应之间距离的约束：随着扰动的观测响应与其最近训练响应之间的余弦距离增大，PDC随之下降。我们提出“预测最近邻距离”——即预测响应到其最近训练响应的余弦距离——作为每条预测的置信度评分。

    arXiv:2611.05849v1 Announce Type: cross  Abstract: Foundation-model predictors of perturbation response are evaluated primarily by aggregate accuracy. Using their predictions to prioritize experiments requires assessing each prediction's reliability. We audit the state-transition (ST) decoder used in STATE across three encoders (SE-600M, scGPT, and Tahoe-x1) and four cell lines under CRISPRi. The decoder's first attention layer has a low-rank output. Projecting this output onto 4-32 fixed directions estimated from training activations preserves mean per-perturbation Pearson delta correlation (PDC). This low-rank attention bottleneck bounds the decoder's reliability by distance to training responses: PDC declines as the cosine distance from a perturbation's observed response to its nearest training response increases. We propose predicted-NN-distance, the cosine distance from a predicted response to its nearest training response, as a per-prediction confidence score. The score has a mea
    
[^147]: PEARL：面向部分偏好漂移下选择性遗忘的预测误差自适应正则化学习

    PEARL: Prediction Error-Adaptive Regularized Learning for Selective Forgetting under Partial Preference Drift

    [https://arxiv.org/abs/2611.05842](https://arxiv.org/abs/2611.05842)

    PEARL提出了一种预测误差自适应的正则化学习方法，通过仅在预测出错的特征方向上反转Gram矩阵更新来实现选择性遗忘，从而有效应对部分偏好漂移，克服了传统方法均匀遗忘和固定探索率的局限。

    

    用户偏好会随着需求、情境和品味的演变而发生漂移。这对个性化推荐、大语言模型响应生成以及通过RLHF训练的奖励模型至关重要，因为这些系统可能会随偏好漂移而变得失准。现有的非平稳赌博机方法在所有维度上均匀遗忘，但真实的偏好漂移通常是部分的：某些维度已经过时，而其他维度仍然有效。神经上下文赌博机通过Gram矩阵单调地累积置信度，缺乏将不确定性引导至预测出错方向的机制。我们提出了PEARL（预测误差自适应正则化学习），它在预测出现严重偏差的特征方向上反转Gram矩阵的更新，同时在其他方向上保持置信度不变。我们证明了：（i）任何非自适应策略在偏好漂移后都会遭受线性遗憾；（ii）没有固定的探索率能够在平稳和非平稳两种情况下都实现低遗憾。

    arXiv:2611.05842v1 Announce Type: cross  Abstract: User preferences shift as needs, context, and tastes evolve. This matters for personalized recommendation, LLM response generation, and RLHF-trained reward models, which can become miscalibrated as preferences drift. Existing non-stationary bandit methods forget uniformly across all dimensions, but real preference shifts are typically partial: some dimensions go stale while others remain valid. Neural contextual bandits accumulate confidence monotonically via the Gram matrix, with no mechanism to direct uncertainty where predictions have gone wrong. We propose PEARL (Prediction Error-Adaptive Regularized Learning), which reverses the Gram update along feature directions where predictions are surprisingly wrong, while leaving confidence intact elsewhere. We prove that (i) any non-adaptive policy suffers linear regret after a preference shift, (ii) no fixed exploration rate achieves low regret in both stationary and non-stationary regime
    
[^148]: RACE：用于免执行机器人策略检查点选择的校准累积评分方法

    RACE: Calibrated Cumulative Scoring for Rollout-Free Robot Policy Checkpoint Selection

    [https://arxiv.org/abs/2611.05838](https://arxiv.org/abs/2611.05838)

    提出RACE——一种无需真实执行的机器人策略检查点选择指标，通过基于状态条件阈值校准动作偏差并用累积乘积聚合评分，使检查点排序与真实部署性能更加一致。

    

    在训练过程中评估已学习机器人策略的部署性能具有挑战性，因为直接衡量任务成功与否需要对每个候选检查点进行代价高昂的真实世界试验。平均验证损失提供了一种无需真实执行的替代方案，但其对检查点的排序可能与部署性能不一致。我们分析了该标准的两个局限性：它在惩罚动作偏差时并未考虑策略能否从这些偏差中恢复过来；并且即使早期的偏差可能使执行过程偏离参考轨迹，后期的参考预测仍被赋予同等权重。我们提出了RACE，一种新颖的免执行检查点排序指标，它利用一组固定的留出成功轨迹进行评估。RACE根据这些轨迹中相似状态下的动作差异推导出状态条件阈值，并以此校准动作偏差。我们通过运行乘积将这些分数进行聚合，使得较大的偏差能够……（摘要原文在此处截断）

    arXiv:2611.05838v1 Announce Type: cross  Abstract: Assessing the deployment performance of learned robot policies during training is challenging, as directly measuring task success requires costly real-world trials for each candidate checkpoint. Averaged validation loss offers a rollout-free alternative, but its checkpoint rankings can be inconsistent with deployment performance. We analyze two limitations of this criterion: it penalizes action deviations without considering whether the policy can recover from them, and gives later reference predictions equal weight even when earlier deviations may lead execution away from the reference. We propose RACE, a novel rollout-free metric for ranking checkpoints using a fixed set of held-out successful trajectories. RACE calibrates action deviations against state-conditional thresholds derived from action differences at similar states in these trajectories. We aggregate these scores through a running product, so that a substantial deviation r
    
[^149]: 路径预算约束下的大语言模型路由：基于K比特尾代价学习的在线分配

    Pathwise-Budgeted LLM Routing: Online Allocation with K-Bit Tail-Cost Learning

    [https://arxiv.org/abs/2611.05835](https://arxiv.org/abs/2611.05835)

    本文提出上下文赌博机策略ADP-HT，通过学习各模型的任务特定成功概率，并利用K比特粗化回复长度标签构建保守成本估计，在固定预算下智能路由大语言模型调用与重试决策，以最大化在预算耗尽前完成的任务数量。

    

    在固定预算下使用大语言模型的企业必须决定调用哪个模型，以及是否重试失败的任务。由于一次调用的费用只有在完成后才会被揭示，一个过长的回复可能耗尽剩余预算，导致后续任务无法得到服务。我们的目标是在服务停止前最大化已解决任务的数量。我们开发了一种上下文赌博机策略ADP-HT，它学习每个模型的任务特定成功概率，并在具有 $2^K$ 个区间的分位数网格上，基于粗化的回复长度标签构建保守的成本估计。该策略将预测成功率与这些成本进行权衡，并根据实际支出和每个剩余任务可用的预算来调整成本惩罚。在校准、局部正则性和网格覆盖条件下，我们对策略保守成本估计中的超额容许量进行了界定。在固定样本量下，增大 $K$ 可以减少离散化误差项，但会增加……（摘要截断）

    arXiv:2611.05835v1 Announce Type: new  Abstract: Firms using large language models under fixed budgets must decide which model to call and whether to retry a failed task. Because a call's charge is revealed only after completion, one long response can exhaust the remaining budget and leave later tasks unserved. Our objective is to maximize the number of tasks solved before service stops. We develop a contextual bandit policy, ADP-HT, which learns each model's task-specific success probability and constructs conservative cost estimates from coarsened response-length labels on a quantile grid with $2^K$ intervals. The policy weighs predicted success against these costs, adjusting the cost penalty to realized spending and the budget available per remaining task. Under calibration, local regularity, and grid-coverage conditions, we bound the excess allowance in the policy's conservative cost estimates. At a fixed sample size, increasing $K$ reduces the discretization term but increases the
    
[^150]: 智能制造中的自主性与协调性：递归自主代理框架

    Autonomy and Coordination in Intelligent Manufacturing: A Recursive Autonomous Agent Framework

    [https://arxiv.org/abs/2611.05828](https://arxiv.org/abs/2611.05828)

    本文从复杂适应系统视角揭示了智能制造中个体自主性与集体协调这两种耦合机制，并提出了一种将目标驱动自主代理与递归组织结构相结合的递归自主代理（RAA）框架，用以实现生产系统中的智能决策与组织。

    

    智能生产要求系统能够根据生产目标和运行状况进行决策并组织工作，同时还需要通过反馈进行持续调整。本文从复杂适应系统（CAS）的视角分析了人类生产组织中的智能。该分析识别出两种相互耦合的机制：个体自主性与集体协调。个体自主性使行动者能够利用可用信息形成局部决策；集体协调则将这些决策与共同目标和联合约束保持一致。学习与适应随时间推移不断改进这两种机制。通过一个生产系统模型，本文形式化了局部决策规则与协调机制如何共同形成系统决策。在此基础上，我们提出了递归自主代理（RAA）框架，该框架将目标驱动的自主代理与递归式的组织结构相结合。目标契约定义了局部目标……

    arXiv:2611.05828v1 Announce Type: new  Abstract: Intelligent production requires systems to make decisions and organize work according to production goals and operating conditions. It also requires continuous adjustment through feedback. This paper analyzes intelligence in human production organizations from a complex adaptive systems (CAS) perspective. The analysis identifies two coupled mechanisms: individual autonomy and collective coordination. Individual autonomy enables actors to form local decisions using available information. Collective coordination aligns these decisions with shared goals and joint constraints. Learning and adaptation improve both mechanisms over time. A production-system model formalizes how local decision rules and coordination jointly form system decisions. On this basis, we propose the Recursive Autonomous Agent (RAA) framework, which combines goal-driven autonomous agents with a recursive organizational structure. Goal contracts define local objectives a
    
[^151]: WayPOP：一个环视开放集全景分割跟踪基准

    WayPOP: A Panoramic Open-Set Panoptic Tracking Benchmark

    [https://arxiv.org/abs/2611.05801](https://arxiv.org/abs/2611.05801)

    WayPOP 是一个环视开放集全景分割跟踪基准，通过在真实 Waymo PVPS 序列中插入 Objaverse 的分类体系外 3D 物体并渲染多相机与 LiDAR 数据，检验未知物体能否在同步多相机序列中被一致地分割和跟踪。

    

    自主智能体的整体感知需要在时间和相机视角之间可靠地分割并关联场景元素，包括固定语义分类体系之外的物体。现有的开放集全景分割基准仅关注单张图像或单一视角，而环视跟踪基准则假设封闭集分类体系。因此，这些基准无法评估未知物体能否在同步的多相机序列中被一致地分割与跟踪。我们提出了 WayPOP，这是一个建立在真实 Waymo PVPS 序列之上、并通过来自 Objaverse 的多样化分类体系外 3D 资产进行增强的环视开放集全景分割跟踪基准。插入的物体被放置在 3D 场景中的固定位置，并在自车与周围交通运动时使用标定好的车载相机进行渲染，从而提供时间和跨视角一致的全景分割标注以及辅助的 LiDAR 距离观测。WayPOP 包含约……（原文摘要在此处截断）。

    arXiv:2611.05801v1 Announce Type: cross  Abstract: Holistic perception for autonomous agents requires reliably segmenting and associating scene elements across time and camera views, including objects outside a fixed semantic taxonomy. Existing open-set panoptic benchmarks focus on individual images or views, while panoramic tracking benchmarks assume closed-set taxonomies. Consequently, these benchmarks do not evaluate whether unknown objects can be consistently segmented and tracked across synchronized multi-camera sequences. We introduce WayPOP, a panoramic open-set panoptic tracking benchmark built on real Waymo PVPS sequences augmented with diverse out-of-taxonomy 3D assets from Objaverse. Inserted objects are placed at fixed locations in the 3D scene and rendered using calibrated vehicle-mounted cameras as the ego vehicle and surrounding traffic move, providing temporally and cross-view consistent panoptic annotations and auxiliary LiDAR range observations. WayPOP contains approx
    
[^152]: 物品表示至关重要：连接硬语义ID与软原型混合以实现生成式检索

    Item Representations Matter: Bridging Hard Semantic IDs and Soft Prototype Mixtures for Generative Retrieval

    [https://arxiv.org/abs/2611.05799](https://arxiv.org/abs/2611.05799)

    提出CAP-Mix流式推荐器，将每个物品表示为共享原型的软混合加上物品专属向量，在不引入离散瓶颈的前提下结合了硬语义ID的共享结构与连续表示的无损优势。

    

    现代推荐系统依赖生成式检索技术来预测下一个物品，既可以将其表示为离散编码，也可以表示为连续向量。一方面，语义ID（SIDs）为离散模型提供了共享参数和预训练的语义结构，但将物品离散化为硬编码会丢失原始内容嵌入中的部分信息。另一方面，连续模型避免了量化，但通常使用扁平的物品查找表，物品之间缺乏显式的参数绑定。我们探究连续推荐器能否在不引入离散瓶颈的情况下引入共享结构。我们提出了CAP-Mix，一种流式推荐器，它将每个物品表示为共享原型的软混合再加上一个物品专属向量。分配、原型和物品向量进行联合训练，并在推理时保留软表示。预训练信息是可选的：SID的归属概率可以作为先验。

    arXiv:2611.05799v1 Announce Type: cross  Abstract: Modern recommender systems rely on generative retrieval techniques to predict the next item either as discrete codes or as a continuous vector. On the one hand, Semantic IDs (SIDs) give discrete models shared parameters and pretrained semantic structure, but discretizing items into hard codes discards some information from the original content embeddings. On the other hand, continuous models avoid quantization, yet typically use flat item lookup tables with no explicit parameter tying across items. We ask whether a continuous recommender can introduce shared structure without making it a discrete bottleneck. We propose CAP-Mix, a flow recommender that represents each item as a soft mixture of shared prototypes plus an item-specific vector. The assignments, prototypes, and item vectors are trained jointly, and the soft representation is retained at inference. Pretrained information is optional: SID responsibilities can serve as a prior 
    
[^153]: ZipDrift：利用一步漂移模型生成多模态短时域MPC代价

    ZipDrift: Generating Multimodal Short-Horizon MPC Costs with One-Step Drifting Models

    [https://arxiv.org/abs/2611.05793](https://arxiv.org/abs/2611.05793)

    该论文提出ZipDrift，通过训练一个向低闭环代价倾斜的一步漂移生成器来产生多模态的MPC代价修正候选，再由学习到的评论家排序选优，解决了确定性回归在多模态场景下平均各模态或继承专家不一致平局决策的问题。

    

    短时域模型预测控制（MPC）速度快但目光短浅。ZipMPC通过学习一种依赖于上下文的短时域代价修正来弥补这一缺陷，该修正模仿长时域专家的行为。我们观察到最优修正往往是多模态的：当较慢的对手既可从左侧也可从右侧超越时，符号相反的良好修正同时存在，而专家在两者之间的选择取决于微小的代价差异。在这种情况下，确定性回归会平均各个模态，或继承专家不一致的平局抉择。我们提出ZipDrift，它在MPC代价修正上训练一个一步漂移生成器，该生成器向低闭环代价倾斜，且仅通过候选修正的代价评估来学习，无需对优化器进行微分。少量生成的候选由学习到的评论家进行排序，随后基于模型的MPC强制执行动力学和约束。在从ZipMPC衍生出的运动学超车基准测试中，53%的上下文是双模态的……

    arXiv:2611.05793v1 Announce Type: cross  Abstract: Short-horizon model predictive control (MPC) is fast but myopic. ZipMPC compensates by learning a context-dependent correction of the short-horizon cost that imitates a long-horizon expert. We observe that the optimal correction is often multimodal: when a slower opponent can be passed on either side, good corrections of opposite sign coexist, and the expert's choice between them hinges on tiny cost differences. Deterministic regression then averages modes or inherits inconsistent expert tie-breaking. We propose ZipDrift, which trains a one-step drifting generator over MPC cost corrections, tilted toward low closed-loop cost and learned only from cost evaluations of candidate corrections, without differentiating through the optimiser. A few generated candidates are ranked by a learned critic, and the model-based MPC then enforces dynamics and constraints. In a kinematic overtaking benchmark derived from ZipMPC, 53% of contexts are bimo
    
[^154]: 面向大语言模型智能体记忆的FlyHash：实证验证与新应用

    FlyHash for LLM Agent Memory: An Empirical Validation and a New Application

    [https://arxiv.org/abs/2611.05791](https://arxiv.org/abs/2611.05791)

    本文首次将受果蝇嗅觉回路启发的FlyHash稀疏哈希应用于大语言模型智能体的记忆检索，并通过严谨的匹配位协议实验实证验证了其相比SimHash的稳健质量优势。

    

    大语言模型（LLM）智能体日益依赖外部记忆库来在对话之间保留信息，通常通过稠密向量嵌入并结合近似最近邻搜索方法实现。果蝇的嗅觉回路以一种截然不同的机制计算感觉输入的局部敏感哈希——先进行稀疏随机投影，再进行胜者通吃竞争——先前的研究表明，这种被称为FlyHash的方案在多个基准测试上，每比特的邻居结构保持能力优于传统的随机超平面哈希。据我们所知，本文首次将FlyHash应用于LLM智能体记忆检索，并提出了三个实践者真正需要回答的问题：在正确匹配位数的协议下，质量优势是否依然成立（答案是肯定的，在10个随机种子和涵盖两种模态的四个数据集上均稳健成立，并揭示了一个此前未被记录的具体比较结果……

    arXiv:2611.05791v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly rely on external memory stores to retain information across conversations, typically via dense vector embeddings searched with approximate nearest-neighbor methods. The fruit fly's olfactory circuit computes a locality-sensitive hash of sensory input using a strikingly different mechanism - sparse random projection followed by winner-take-all competition - and prior work (Dasgupta et al., 2017) showed this scheme, dubbed FlyHash, preserves neighbor structure better per bit than conventional random-hyperplane hashing (SimHash) on several benchmarks. We apply FlyHash, to our knowledge for the first time, to LLM agent memory retrieval, and ask three questions a practitioner actually needs answered: does the quality advantage hold under a correctly matched-bit protocol (it does, robustly, across 10 seeds and four datasets spanning two modalities, with one specific, previously undocumented compar
    
[^155]: 用于旋转相对论性恒星的无监督神经算子：高保真状态方程推断与大规模脉冲星自旋演化

    Unsupervised Neural Operators for Rotating Relativistic Stars: High-Fidelity Equation of State Inference and Pulsar Spin Evolution at Scale

    [https://arxiv.org/abs/2611.05790](https://arxiv.org/abs/2611.05790)

    该论文开发了物理信息驱动的无监督神经算子，能够高效求解旋转中子星的轴对称广义相对论流体静力平衡方程，从而实现高保真度的状态方程推断以及大规模的脉冲星自旋演化群体合成。

    

    观测到的具有毫秒级周期的旋转中子星群体是探测基础物理学的有力工具。质量-半径测量能够揭示简并核物质中约束不佳的性质，而高保真的恒星群体合成则将中子星形成与双星演化中的未知因素与大型射电计时巡天观测联系起来。然而，由于求解控制旋转中子星内部的轴对称广义相对论流体静力平衡高度耦合偏微分方程的计算成本极高，重复的结构计算难以实现。采用非旋转计算或近似旋转修正可能会使状态方程推断产生偏差，并限制群体合成在追踪前身天体演化历史以及传播状态方程不确定性方面的保真度。在这篇快报中，我们开发了物理信息驱动的神经算子，能够求解轴对称问题……

    arXiv:2611.05790v1 Announce Type: cross  Abstract: Observed populations of rotating neutron stars~(NSs) with millisecond-order periods are powerful probes of fundamental physics. Mass-radius measurements inform poorly constrained properties of degenerate nuclear matter, whereas high-fidelity population synthesis connects unknowns in NS formation and binary evolution to large radio-timing surveys. However, repeated structure calculations are prohibitive due to the computational cost of solving the highly coupled partial differential equations of axisymmetric general relativistic hydrostatic equilibrium that govern rotating NS interiors. Non-rotating calculations or approximate rotational corrections can bias equation of state inference and limit the fidelity of population synthesis in tracing the evolutionary history of progenitors and propagating equation of state uncertainties. In this \textit{Letter}, we develop physics-informed neural operators that can solve the axisymmetric proble
    
[^156]: 时机决定一切：推断持续更新模型中的时间成员关系

    Timing is Everything: Inferring Temporal Membership in Continually Updated Models

    [https://arxiv.org/abs/2611.05781](https://arxiv.org/abs/2611.05781)

    本文首次系统研究了持续更新模型中的时间隐私风险，提出了基于双向LSTM的黑盒攻击方法TMIA，能够以时间戳和区间两种粒度推断样本是否以及何时被用于模型训练。

    

    持续模型更新已成为机器学习系统部署中的常见做法。然而，动态训练行为带来了现有研究尚未充分解决的新风险。在本文中，我们首次对持续更新的深度学习模型中的时间隐私风险进行了系统性研究。我们形式化了持续更新模型中“时间成员关系”的概念，并定义了时间成员推断攻击，该攻击旨在在时间戳级别和时间区间级别两种粒度上推断个体样本是否以及何时参与了模型训练。为此，我们提出了TMIA，一种新颖的黑盒攻击方法，它利用双向LSTM从跨连续更新观察到的模型输出中推断时间成员关系。通过在三个语言模型和三个图像分类模型上的大量实验，我们证明TMIA非常有效，并显著优于现有基线方法。

    arXiv:2611.05781v1 Announce Type: cross  Abstract: Continual model updating has become a common practice in the deployment of machine learning systems. However, the dynamic training behaviors give rise to new risks not sufficiently addressed by existing studies. In this paper, we present the first systematic study of temporal privacy risk in continually updated deep learning models. We formalize the notion of temporal membership in continually updated models. We then define the temporal membership inference attack, which seeks to infer whether and when individual samples were involved in model training, at both timestamp-level and interval-level granularities. To this end, we propose TMIA, a novel black-box attack that leverages bidirectional LSTMs to infer temporal membership from model outputs observed across successive updates. Through extensive experiments on three language models and three image classification models, we show that TMIA is highly effective and significantly outperf
    
[^157]: 预测多重性下的追索优化

    Optimizing Recourse Under Predictive Multiplicity

    [https://arxiv.org/abs/2611.05774](https://arxiv.org/abs/2611.05774)

    该论文提出ORPM算法，通过在带预算约束下最小化相互竞争的模型集合上的最坏情况有效性损失，解决了预测多重性下的鲁棒算法追索问题。

    

    算法追索旨在识别能够将模型的不利预测改变为有利预测的修改方案。在预测多重性存在的情形下，性能相近的模型可能会对同一个体做出不同的预测，因此对某一模型有效的追索方案对另一模型可能失效。我们将预测多重性下的鲁棒追索问题形式化为一个带预算约束的最小化问题，即在相互竞争的一组模型上最小化最坏情况下的有效性损失，这组模型可以包含不同架构或来自不同假设类别的模型。我们提出了一种名为ORPM的oracle高效算法，并证明了其收敛的充分条件。我们在一个合成数据集和五个真实数据集上评估了ORPM，竞争模型集合包括线性模型、神经网络、混合模型和决策树模型。实验结果表明，在许多测试场景中，优化间隙上界不断下降且有效性较高，与为特定模型类别设计的基线方法相比表现优越。

    arXiv:2611.05774v1 Announce Type: new  Abstract: Algorithmic recourse identifies modifications that can change an unfavorable model prediction into a favorable one. Under predictive multiplicity, similarly performing models can disagree on individual predictions, so recourse valid for one model may fail for another. We formulate robust recourse under predictive multiplicity as a budget-constrained minimization of the worst-case validity loss across a competing set of models, which can include models with different architectures or from different hypothesis classes. We propose an oracle-efficient algorithm called ORPM and prove sufficient conditions for its convergence. We evaluate ORPM on one synthetic and five real datasets, and competing sets consist of linear, neural, mixed, and decision-tree models. Our experiments show decreasing optimization-gap bounds and high validity in many tested settings, with favorable performance against baselines designed for specific model classes or ar
    
[^158]: 将专家点评与LLM驱动的精炼相结合：一种迭代式方法用于筛选高质量AI评估数据集

    Pairing Expert Critique with LLM-Driven Refinement: An Iterative Approach for Curating Quality AI Evaluation Datasets

    [https://arxiv.org/abs/2611.05749](https://arxiv.org/abs/2611.05749)

    提出SME-C人机协作工作流程，通过专家轻量级点评结合LLM驱动的分析与精炼，迭代式筛选高质量AI评估数据集与评分标准，并在金融商业简报领域收集411位专业人士的3,853条点评验证了两种LLM精炼策略的有效性。

    

    评估和改进特定领域的LLM回答需要高质量的参考数据和评分标准，然而由专家撰写既缓慢又昂贵，而合成数据集除非以专家判断为基础，否则可能无法捕捉从业者的真实偏好。我们提出了SME-C（领域专家点评，Subject Matter Expert Critique），这是一种人机协作工作流程，通过对候选回答的轻量级点评来捕捉专家知识，并结合LLM驱动的分析与精炼，迭代式地筛选回答数据集和评估评分标准。我们在金融商业简报领域验证了SME-C，在两轮调查中收集了411位金融专业人士针对520个查询给出的3,853条点评。我们评估了两种LLM驱动的精炼策略：一种以专家的书面点评为指导，另一种仅从专家的偏好投票中推断点评内容。在对另外100位专业人士进行的盲测验证调查中，我们发现两种策略……（原文摘要在此处截断）

    arXiv:2611.05749v1 Announce Type: cross  Abstract: Evaluating and improving domain-specific LLM responses requires high-quality reference data and rubrics, yet expert authoring is slow and expensive, and synthetic datasets may fail to capture practitioner preferences unless they are grounded in expert judgment. We present SME-C (Subject Matter Expert Critique), a human-AI workflow that captures expertise through lightweight critiques of candidate responses and complements it with LLM-driven analysis and refinement to iteratively curate response datasets and evaluation rubrics. We validate SME-C in the domain of financial business briefs, collecting 3,853 critiques from 411 finance professionals on 520 queries across two survey rounds. We evaluate two LLM-driven refinement strategies: one guided by the experts' written critiques and one that infers critique solely from their preference votes. In a blind validation survey with 100 additional professionals, we found that both strategies s
    
[^159]: 面向低资源医学自动语音识别的纯文本域自适应：豪萨语的浅融合与密度比重评分

    Text-Only Domain Adaptation for Low-Resource Medical ASR: Shallow Fusion and Density Ratio Rescoring for Hausa

    [https://arxiv.org/abs/2611.05744](https://arxiv.org/abs/2611.05744)

    该研究提出仅需文本数据（无需医学领域语音数据）即可对豪萨语医学ASR进行域自适应，通过浅融合和密度比重评分两种解码时语言模型融合方法，将词错误率从29.56%显著降至约23.3%–23.5%。

    

    自动语音识别（ASR）系统常常难以处理专业医学词汇，尤其是在缺乏领域特定语音数据的低资源语言中。我们研究了针对豪萨语医学ASR的纯文本域自适应方法，使用固定的MMS-1B-FL102声学模型以及两种解码时语言模型融合方法：浅融合（SF）和密度比方法（DRA）。基于19,421条豪萨语健康句子训练的4-gram KenLM作为医学目标模型，而由75,372条句子构成的通用领域模型为DRA提供源先验。评估使用了来自尼日利亚北部六个州的10名说话人针对25条经医生审阅的提示语录制的250条录音。在提示语不相交的五折交叉验证下，SF和DRA分别将词错误率（WER）从29.56%降至23.55%和23.33%。两者均显著优于贪婪解码（p < 0.001），而二者之间的差异不显著（p = 0.423）。然而，仅使用通用语言模型的SF……

    arXiv:2611.05744v1 Announce Type: cross  Abstract: Automatic speech recognition (ASR) systems often struggle with specialised medical vocabulary, especially in low-resource languages where domain-specific speech data are scarce. We investigate text-only domain adaptation for Hausa medical ASR using the fixed MMS-1B-FL102 acoustic model and two decoding-time language-model fusion methods: Shallow Fusion (SF) and the Density Ratio Approach (DRA). A 4-gram KenLM trained on 19,421 Hausa health sentences serves as the medical target model, while a 75,372-sentence general-domain model provides the source prior for DRA. Evaluation uses 250 recordings of 25 physician-reviewed prompts from 10 speakers across six northern Nigerian states. Under prompt-disjoint five-fold cross-validation, SF and DRA reduce WER from 29.56% to 23.55% and 23.33%, respectively. Both significantly outperform greedy decoding (p < 0.001), while their difference is not significant (p = 0.423). However, general-LM-only SF
    
[^160]: 学习分层图像块切分策略以加速视觉Transformer

    Learning Hierarchical Patch Splitting Policies for Faster Vision Transformers

    [https://arxiv.org/abs/2611.05742](https://arxiv.org/abs/2611.05742)

    该论文提出SPLIT方法，通过学习依据模型认为有意义的任务相关内容来分配图像token分辨率，替代基于像素统计的启发式方法，在减少30%token数量的同时保持与全分辨率ViT相当的准确率。

    

    视觉Transformer（ViT）的性能瓶颈在于其输入token序列的长度。标准的图像块化通过在整个图像上应用统一的patch大小来固定该长度，而不考虑各区域的内容。以往关于自适应patch大小的工作通过为视觉上简单的区域分配更粗糙的patch来减少token数量，但这类方法依赖于局部熵等含噪声的像素级启发式方法来做出patch化决策。我们认为，这种计算资源的分配应该基于模型对任务相关信息的理解，而不是原始的像素统计。我们提出了SPLIT（Split Policy Learned Image Tokenization，学习切分策略的图像token化方法），它学习根据模型认为有意义的内容来分配token分辨率，从而产生与任务相关内容更好匹配的更短token序列。SPLIT在使用减少30%的token的情况下，达到了全分辨率ViT分类基线的准确率。同样的收益也体现在密集预测任务上。（注：原始摘要在此处被截断）

    arXiv:2611.05742v1 Announce Type: cross  Abstract: Vision Transformers (ViTs) are bottlenecked by the length of their input token sequence. Standard patchification fixes this length with a single patch size applied across the image, regardless of regional content. Prior work on adaptive patch sizing reduces token count by assigning coarser patches to visually simple regions, but relies on noisy pixel-level heuristics such as local entropy to make patchification decisions. We argue that this compute allocation should be grounded in the model's interpretation of task relevant information, not in raw pixel statistics. We introduce Split Policy Learned Image Tokenization which learns to allocate token resolution based on what the model finds meaningful, producing shorter token sequences that are better matched to task-relevant content. SPLIT matches the accuracy of full-resolution ViT classification baselines while using 30% fewer tokens. The same gains hold on dense prediction: SPLIT achi
    
[^161]: OdorSpaces：三维环境中的视觉-嗅觉具身导航

    OdorSpaces: Visual-Olfactory Embodied Navigation in 3D Environments

    [https://arxiv.org/abs/2611.05738](https://arxiv.org/abs/2611.05738)

    提出OdorSpaces——一个通过FDS模拟三维气味扩散来增强Habitat-Sim的物理仿真平台与基准，支持气体泄漏定位和早期火灾定位等视觉-嗅觉具身导航任务，其多模态强化学习策略优于多种基线方法。

    

    嗅觉能够提供关于空气传播化学线索的关键信息，但它在具身智能领域仍未得到充分探索，先前的机器嗅觉研究较为有限，且主要聚焦于语义气味感知。我们提出了OdorSpaces，一个基于物理的视觉-嗅觉具身导航仿真平台与基准。OdorSpaces通过火灾动力学模拟器（FDS）模拟的三维气味传输与扩散来增强Habitat-Sim，使智能体能够结合视觉与嗅觉观测在照片级逼真的室内环境中进行导航。该基准包含两个代表性的源头搜索场景：不可见气体泄漏定位与早期火灾定位。我们开发了一个初始的多模态强化学习策略，能够随时间整合视觉观测与嗅觉测量。实验表明，该策略优于多种基线方法，为未来研究建立了强有力的参考。

    arXiv:2611.05738v1 Announce Type: cross  Abstract: Olfaction provides critical information about airborne chemical cues, yet it remains largely underexplored in embodied AI, with limited prior machine olfaction work focusing primarily on semantic odor perception. We introduce OdorSpaces, a physics-based simulation platform and benchmark for visual-olfactory embodied navigation. OdorSpaces augments Habitat-Sim with 3D odor transport and dispersion simulated by Fire Dynamics Simulator (FDS), enabling agents to navigate photorealistic indoor environments using both visual and olfactory observations. The benchmark includes two representative source-seeking scenarios: invisible gas-leak localization and early-stage fire localization. We develop an initial multimodal reinforcement learning policy that integrates visual observations and olfactory measurements over time. Experiments show that the policy outperforms a diverse set of baselines, establishing a strong reference for future research
    
[^162]: 面向算子学习的通用潜在空间自监督学习

    Self-Supervised Learning of Universal Latent Spaces for Operator Learning

    [https://arxiv.org/abs/2611.05735](https://arxiv.org/abs/2611.05735)

    该论文提出通过在即时合成、无标签无求解器的解析函数上自监督预训练函数自编码器，学习一个对平移、缩放和幅度精确等变的通用潜在空间，将其冻结后可在任意点云上跨任务复用于算子学习。

    

    神经算子将其函数表示拟合到所训练的特定任务上，因此每遇到一个新任务都需要从带标签数据重新构建一个降维空间。借鉴自然语言处理和计算机视觉领域中使表示可共享的自监督预训练思想，我们将表示学习与算子学习解耦。具体而言，在一个即时合成解析函数的函数自编码器上进行预训练，无需求解器、无需标签，使其能够在任意点云上表示函数，其诱导出的潜在空间被冻结并在多个任务间复用。该表示对平移、缩放和幅度具有精确的等变性，并能够在其从未见过的几何结构上重构偏微分方程（PDE）的解以及自然图像。由于该潜在空间与任务无关，其本身也可以被独立地解释。整个潜在空间都被充分利用并能解码为有效的物理场，同时保留了物理…（摘要在此处被截断）

    arXiv:2611.05735v1 Announce Type: new  Abstract: Neural operators fit their representation of functions to the task they are trained on, so every new task rebuilds a reduced space from labeled data. Following the self-supervised pre-training that made representations shareable in natural language processing and computer vision, we decouple representation learning from operator learning. A functional autoencoder is pre-trained on analytic functions synthesized on the fly, with no solver and no labels, to represent functions on arbitrary point clouds, and the latent space it induces is frozen and reused across tasks. The representation is exactly equivariant to translation, dilation, and amplitude, and reconstructs solutions of partial differential equations (PDEs) and natural images on geometries it never saw. Being task-agnostic, the latent space can be interpreted in its own right. The whole space is used and decodes to valid fields, it preserves the ordering of distances between phys
    
[^163]: RESHPart：面向超图划分的反馈重塑谱嵌入

    RESHPart: Feedback-Reshaped Spectral Embedding for Hypergraph Partitioning

    [https://arxiv.org/abs/2611.05729](https://arxiv.org/abs/2611.05729)

    该论文提出FRESHPart框架，从谱嵌入几何出发，通过几何反馈和搜索反馈重塑谱嵌入，使其更有利于基于投影的超图划分，并在Titan23基准上相比hMETIS显著降低割值。

    

    随着可扩展的拉普拉斯求解器的出现，下一个前沿是如何更有效地利用它们以获得高质量的超图划分。在本工作中，我们从谱嵌入几何的视角重新审视谱超图划分。通过对电路超图和受控合成（超）图的系统性分析，我们识别出一种射线状（ray-like）嵌入结构，这有助于解释基于投影的划分方法的有效性。我们进一步表明，谱嵌入的几何结构可以通过拉普拉斯矩阵设计和超边重加权被系统地影响。基于这些见解，我们提出了FRESHPart，一个通过几何反馈和搜索反馈来精化谱嵌入的反馈重塑谱划分框架，从而产生与基于投影的划分更加对齐的嵌入。在Titan23基准测试上，FRESHPart 相比 hMETIS 实现了平均割的降低（原文在此处截断）。

    arXiv:2611.05729v1 Announce Type: new  Abstract: With scalable Laplacian solvers now available, the next frontier is to exploit them more effectively for high-quality hypergraph partitioning. In this work, we revisit spectral hypergraph partitioning from the perspective of spectral embedding geometry. Through a systematic analysis of circuit hypergraphs and controlled synthetic (hyper)graphs, we identify a ray-like embedding structure that helps explain the effectiveness of projection-based partitioning. We further show that the geometry of spectral embeddings can be systematically influenced through Laplacian design and hyperedge reweighting. Guided by these insights, we propose FRESHPart, a feedback-reshaped spectral partitioning framework that refines the spectral embedding through Geometry Feedback and Search Feedback, producing embeddings that are better aligned with projection-based partitioning. On the Titan23 benchmark, FRESHPart achieves average cut reductions over hMETIS of 1
    
[^164]: 通过自回归Transformer实现级联极端事件的生成式建模

    Generative Modeling for Cascading Extremes via Autoregressive Transformers

    [https://arxiv.org/abs/2611.05728](https://arxiv.org/abs/2611.05728)

    该论文提出一种将多变量极端事件观测编码为分类标记序列的极值表示方法，并利用预训练的仅解码器自回归Transformer学习历史依赖的标记转移规律，从而实现对级联极端事件未来演化的生成式建模与预测。

    

    极端事件往往接连发生，从地震-海啸级联到金融市场中接连出现的极端损失。我们提出了一种极值标记表示方法，将多变量观测编码为分类序列。其中一个“安静”标记用于记录不存在超过阈值的情况；否则，该标记以最大的边际概率尺度得分来识别发生超限的分量。这种表示方法对逐分量的严格单调递增变换保持不变。它保留了主导超限事件的身份信息和时间顺序，但丢弃了其量级大小以及同时发生的非主导超限事件的身份信息。我们使用仅解码器的Transformer在该表示上估计依赖历史信息的下一标记核函数，并通过递归采样从诱导出的联合预测规律中生成未来的标记序列。我们通过在具有不同参数的模拟过程族上进行预训练，来摊销这一生成式模型的计算成本。

    arXiv:2611.05728v1 Announce Type: cross  Abstract: Extreme events often occur sequentially, from earthquake-tsunami cascades to successive extreme losses in financial markets. We introduce an extremal token representation that encodes multivariate observations as a categorical sequence. A quiet token records the absence of threshold exceedances; otherwise, the token identifies the exceeding component with the largest marginal probability-scale score. The representation is invariant to componentwise strictly increasing transformations. It preserves the identity and temporal ordering of dominant exceedances but discards their magnitudes and the identities of nondominant simultaneous exceedances. A decoder-only transformer estimates the history-dependent next-token kernel on this representation, and recursive sampling generates future token sequences from the induced joint predictive law. We amortize our generative model by pretraining across simulated process families with varying parame
    
[^165]: 语言模型知道哪些约束属于同一组吗？

    Do Language Models Know Which Constraints Belong Together?

    [https://arxiv.org/abs/2611.05726](https://arxiv.org/abs/2611.05726)

    该论文证明，在 Dantzig-Wolfe 重构中寻找只有组合起来才能改进边界的约束族隐藏组合需要 k^w 次评估，而一个遗漏 e 个成员的分组预测可将成本降至 Θ(k^e)，因此预测节省的评估次数直接度量了预测器（如语言模型）对约束分组知识的掌握程度。

    

    Dantzig-Wolfe 重构通过将选定的一组约束族凸化来加强混合整数规划的线性松弛，而每个候选选择都需要付出一次列生成的运行代价。当各约束族只有组合在一起才能改进边界时，这一选择变得困难，因为每次只添加一个约束族的搜索会在线性松弛处停滞。批量生产、网络设计和设施选址中的经典不等式正建立在这类零边际组之上。我们证明，当每次评估最多凸化 k 个约束族中的 s 个时，任何仅观察边界值的方法都需要量级为 k^w 的覆盖次数评估，才能找到包含 w 个约束族的隐藏组合，且该结论在实际的 Dantzig-Wolfe 松弛上同样成立。一个遗漏 e 个成员的预测可通过一次无需知晓 e 的分阶段搜索，将成本降至 Θ(k^e)，因此预测所节省的评估次数直接度量了预测器所掌握的知识。我们研究了两种预测来源……

    arXiv:2611.05726v1 Announce Type: cross  Abstract: Dantzig-Wolfe reformulation strengthens the linear relaxation of a mixed-integer program by convexifying a chosen set of constraint families, and every candidate costs a column generation run. The choice is hard when families strengthen the bound only together, because a search that adds one family at a time then stops at the linear relaxation. The classical inequalities of lot sizing, network design and facility location rest on such zero-marginal groups. We prove that when each evaluation convexifies at most $s$ of $k$ families, any method that observes only bounds needs a covering number of evaluations, of order $k^w$, to find a hidden group of $w$ families, also on actual Dantzig-Wolfe relaxations. A prediction that misses $e$ members reduces the cost to $\Theta(k^e)$, through one staged search that does not need to know $e$, so the evaluations a prediction saves measure what the predictor knows. We study two sources of predictions
    
[^166]: 给草稿模型第二次猜测的机会：面向投机解码的并行重审方法

    Give the Drafter a Second Chance to Guess: Parallel Reconsideration for Speculative Decoding

    [https://arxiv.org/abs/2611.05724](https://arxiv.org/abs/2611.05724)

    该论文提出“并行重审”方法，让投机解码中的草稿模型在自身不确定性引导下，通过与主草稿并行运行的额外前向计算重新考虑犹豫的猜测，使接受长度平均提升5.3%至10.2%，加速比平均提升3.5%至9.2%。

    

    投机解码利用一个小型草稿模型来加速大语言模型的推理，其每次猜测的接受率决定了可实现的加速效果。特征级草稿模型都有一个共同的设计选择：无论词元是简单还是困难，都在单次前向传播中完成草拟。目标模型拒绝的许多词元，恰恰是草稿模型自身不确定、而目标模型却能果断回答的词元。为解决这一问题，我们提出了并行重审方法：在自身不确定性的引导下，草稿模型对其犹豫的猜测进行重新考虑，额外的计算与单次前向的草稿并行运行，带来的延迟几乎可以忽略不计。协同设计的训练方案与服务时草拟算法将这些重审后的猜测作为精炼候选追加到原始草稿树中。在不同模型和基准测试上，平均而言，并行重审方法相比基线在接受长度上提升5.3%至10.2%，在加速比上提升3.5%至9.2%。

    arXiv:2611.05724v1 Announce Type: new  Abstract: Speculative decoding accelerates large language model inference with a small drafter, whose per-guess acceptance rate determines the achievable speedup. Feature-level drafters share a design choice: every token, easy or hard, is drafted under a single forward pass. Many of the tokens the target rejects are ones the drafter is itself unsure about yet the target answers decisively. To address this, we introduce Parallel Reconsideration: guided by its own uncertainty, the drafter reconsiders its hesitant guesses with extra passes that run in parallel with the one-pass draft, adding negligible latency. A co-designed training recipe and serving-time drafting algorithm append these reconsidered guesses to the original draft tree as refined candidates. Across models and benchmarks, on average, Parallel Reconsideration improves over the baselines by 5.3% to 10.2% on acceptance length and by 3.5% to 9.2% on speedup.
    
[^167]: CoReFlow：用于概率交通预测的一致性松弛流映射

    CoReFlow: Consistency-Relaxed Flow Maps for Probabilistic Traffic Forecasting

    [https://arxiv.org/abs/2611.05715](https://arxiv.org/abs/2611.05715)

    提出CoReFlow模型，首次将流映射的单步生成能力应用于概率交通预测，解决了扩散和流模型多步采样推理慢且与交通数据分布复杂度不匹配的问题，并通过简洁的训练目标提升了预测精度。

    

    概率交通预测对于可靠的智能交通系统（ITS）至关重要，因为它能够捕捉复杂路网中的不确定性。近年来，基于扩散模型和流模型的生成式模型通过多步采样过程实现了不确定性建模。然而，除了推理速度慢之外，该过程本意是捕捉复杂的多模态分布，这与条件交通数据分布相对有限的复杂度并不匹配，导致模型表现不如确定性基线。为解决这一问题，我们提出了一致性松弛流映射模型，这是首次利用流映射单步生成能力的工作，流映射即概率流常微分方程的解算子。此外，我们通过一个简洁的训练目标来提升精度，该目标通过有限时间步上的插值端点对模型进行监督。

    arXiv:2611.05715v1 Announce Type: new  Abstract: Probabilistic traffic forecasting is essential for reliable Intelligent Transportation Systems (ITS), as it captures uncertainties in complex road networks. Recent diffusion- and flow-based generative models have facilitated uncertainty modeling through a multi-step sampling process. However, beyond slow inference, this process is designed to capture complex multimodal distributions, which misaligns with the limited complexity of the conditional traffic data distribution, causing models to underperform deterministic baselines. To address this, we introduce the \underline{Co}nsistency-\underline{Re}laxed \underline{Flow} Map (CoReFlow) model, presenting the first effort to leverage the one-step generation capability of flow maps, known as solution operators of probability-flow ordinary differential equations. Besides, we improve accuracy using a concise training objective that supervises the model via interpolation endpoints over finite t
    
[^168]: 智能体：只在有帮助时才思考

    Agents, Think Only When it Helps

    [https://arxiv.org/abs/2611.05709](https://arxiv.org/abs/2611.05709)

    研究发现LLM智能体蒸馏中思想监督对最终行动表现影响甚微，仅当思想信息未在相应行动中披露时才变得重要，因此应根据信息披露情况灵活选择是否蒸馏思想，以提升效率。

    

    LLM智能体蒸馏通过在通常交织着思想与行动的完整教师轨迹上进行训练，将智能体能力迁移到学生模型中。我们质疑思想监督对智能体最终行动表现究竟有多关键。为回答这一问题，我们在蒸馏过程中独立地扰动教师的思想与行动，以解耦二者的影响。在跨模型规模的广泛采用的智能体基准测试上，改变或丢弃思想对学生性能几乎没有影响。然而我们发现，当训练中思想的信息未在相应行动中被披露时，思想监督变得重要起来。通过识别这类信息不一致的情况，我们的发现挑战了为LLM智能体蒸馏完整教师轨迹的常见做法，并提供了一个新见解：思想监督可以作为一种基于效率的可辨识选择，取决于思想的信息是否被披露。

    arXiv:2611.05709v1 Announce Type: new  Abstract: LLM agent distillation transfers agentic capabilities to student models by training on complete teacher trajectories that typically interleave thoughts and actions. We question how critical thought supervision is for agents' ultimate action performance. To answer this, we disentangle their impacts by independently perturbing teacher thoughts and actions during distillation. On widely adopted agent benchmarks across model scales, altering or discarding thoughts has little effect on student performance. We find, however, that thought supervision becomes important when information about the thought is not disclosed in the corresponding action during training. By identifying such discrepant cases, our findings challenge the common practice of distilling entire teacher trajectories for LLM agents, and offer a new insight that thought supervision can be a discernible choice for efficiency depending on whether the information of thought is disc
    
[^169]: 神经网络与博弈

    Neural networks and games

    [https://arxiv.org/abs/2611.05706](https://arxiv.org/abs/2611.05706)

    本文的核心贡献是建立了对称二值Hopfield神经网络与具有二次势函数的二行动博弈之间的精确等价关系，将神经网络稳定状态映射为纳什均衡、异步更新映射为最优反应动态、记忆存储与提取映射为均衡的设计与选择。

    

    Hopfield神经网络与非合作博弈为理解集体结果如何从局部交互中涌现提供了看似不同的途径。在神经网络中，个体单元根据局部信号调整自身状态；在博弈中，参与者选择行动以最大化个体收益。本文建立了对称二值Hopfield网络与具有二次势函数的二行动博弈之间的精确等价关系。该等价关系将稳定的网络构型对应于纯策略纳什均衡，将异步神经更新对应于最优反应动态。更广泛地说，神经网络中记忆的存储与提取分别对应于博弈中均衡的设计与选择。

    arXiv:2611.05706v1 Announce Type: cross  Abstract: Hopfield neural networks and noncooperative games provide seemingly different approaches to understanding how collective outcomes emerge from local interactions. In a neural network, individual units adjust their states according to local signals; in a game, players choose actions to maximise individual payoffs. I establish an exact equivalence between symmetric binary Hopfield networks and binary-action games that admit a quadratic potential. The equivalence identifies stable network configurations with pure-strategy Nash equilibria, and asynchronous neural updating with best-response dynamics. More broadly, the storage and retrieval of memories in neural networks correspond, respectively, to the design and selection of equilibria in games.
    
[^170]: 面向树皮图像树种分类的最优配置探索：一项防泄漏消融研究

    Toward an Optimal Configuration for Tree Species Classification from Bark Images: A Leakage-Safe Ablation Study

    [https://arxiv.org/abs/2611.05680](https://arxiv.org/abs/2611.05680)

    本研究在树级别防泄漏的数据划分下，对基于树皮图像的树种深度分类进行了系统性消融实验，系统考察图像块大小、预测聚合策略、训练方式与骨干容量等因素，以确定最优模型配置并避免以往可能因数据泄漏而被高估的精度。

    

    背景：通过树皮照片自动识别树种对林业清查和木材加工业具有实际意义，然而决定深度学习方法精度的设计选择尚未得到系统研究，且一些先前结果可能因同一棵树的图像同时出现在训练集与测试集中（数据泄漏）而被高估。方法：在公开的BarkNet 1.0数据集（20个树种）上，采用既定的树级别防泄漏划分方案开展系统性消融研究。基于ConvNeXt-V2骨干网络，依次改变以下因素并通过五折交叉验证进行评估：图像块大小、将逐块预测聚合为图像级决策的策略、训练方式（单阶段与两阶段）、骨干网络容量、整图基线以及推理时可用的图像块数量。结果：逐块精度上升……（摘要在此处被截断）

    arXiv:2611.05680v1 Announce Type: cross  Abstract: Background: The automatic identification of tree species from bark photographs is of practical interest for forestry inventory and the wood-processing industry, yet the design choices that govern the accuracy of deep-learning approaches have not been studied systematically, and some prior results may be inflated by leakage of images of the same trunk across training and test partitions. Methods: A systematic ablation study is conducted on the public BarkNet 1.0 dataset (20 species) under the established tree-level, leakage-safe partitioning scheme. Using ConvNeXt-V2 backbones, the following factors are varied in turn and evaluated by five-fold cross-validation: patch size, the strategy by which per-patch predictions are aggregated into an image-level decision, the training regime (single- versus two-stage), backbone capacity, the whole-image baseline, and the number of patches available at inference. Results: Per-patch accuracy rises m
    
[^171]: 重新审视Soft Actor-Critic算法中的Critic估计

    Revisiting Critic Estimation in Soft Actor-Critic Algorithms

    [https://arxiv.org/abs/2611.05666](https://arxiv.org/abs/2611.05666)

    该论文在固定策略目标的前提下，系统性地重新审视并比较了SAC算法中六种critic设计变体（软优势vs软Q值、多步回报目标、在线或回放数据训练）在十四个任务上的表现。

    

    Soft Actor-Critic (SAC) 是一种广泛使用的深度强化学习算法，它将熵正则化与actor-critic架构相结合。标准实现依赖于对Q函数的一步时序差分学习，然而其他替代性的critic设计选择仍然未被充分探索。在这项工作中，我们重新审视了三个这样的设计选择：学习软优势函数而非软Q值、使用多步回报目标，以及从经验回放缓冲区数据训练多步critic。在保持策略目标不变的情况下，我们比较了这些选择所产生的六种critic变体：(i) 从单步转移学习的advantage critic和 (ii) dueling-advantage critic，以及分别 (iii, iv) 在线策略上或 (v, vi) 从回放轨迹片段中学习的λ-return critic和软广义优势估计（GAE）critic。每种变体都在十四个任务上通过完整的随机超参数搜索进行评估，将超参数……

    arXiv:2611.05666v1 Announce Type: new  Abstract: Soft Actor-Critic (SAC) is a widely used deep reinforcement learning algorithm that combines entropy regularization with an actor-critic architecture. Standard implementations rely on one-step temporal-difference learning of Q-functions, yet alternative critic design choices remain underexplored. In this work, we revisit three such choices: learning soft advantages instead of soft Q-values, the use of multi-step return targets, and the training of multi-step critics from replay buffer data. Keeping the policy objective fixed, we compare the six critic variants these choices generate: (i) advantage and (ii) dueling-advantage critics learned from one-step transitions, and $\lambda$-return and soft Generalized Advantage Estimation (GAE) critics learned either (iii, iv) on-policy or (v, vi) from replayed trajectory segments. Each variant is evaluated across an entire random hyperparameter search on fourteen tasks, treating the hyperparameter
    
[^172]: 有限基物理信息神经网络的自适应域分解

    Adaptive Domain Decomposition for Finite Basis Physics-Informed Neural Networks

    [https://arxiv.org/abs/2611.05659](https://arxiv.org/abs/2611.05659)

    本文提出ADD-FBPINNs方法，将数据驱动的单位分解网络与有限基物理信息神经网络相结合，能够根据解的特征自动学习稳定且几何可解释的自适应域分解，在多个偏微分方程求解任务上相比固定域分解的FBPINNs获得更高精度。

    

    在这项工作中，我们提出了自适应域分解有限基物理信息神经网络（ADD-FBPINNs）。所提出的方法将单位分解网络（POUNets）——一种用于学习域划分的数据驱动方法——与有限基物理信息神经网络（FBPINNs）相结合，以实现更高效的域分解并获得更精确的逼近。FBPINNs通过用户自定义的计算域细分来提高标准物理信息神经网络（PINNs）的精度。通过使用可分离的POUNet根据解的特征自适应地调整域分解，该框架能够自动学习稳定且在几何上可解释的划分。我们在二维亥姆霍兹方程、具有多尺度特征的二维泊松方程以及粘性Burgers方程上测试了该方法，观察到与固定域分解的FBPINNs相比精度有所提升。

    arXiv:2611.05659v1 Announce Type: new  Abstract: In this work, we present adaptive domain decomposition finite basis physics-informed neural networks (ADD-FBPINNs). The proposed method integrates Partition of Unity Networks (POUNets), a data-driven approach for learning domain partitions, with finite Basis physics-informed neural networks (FBPINNs) to perform more efficient domain decomposition and achieve more accurate approximations. FBPINNs improve upon the accuracy of standard physics-informed neural networks (PINNs) through user-defined subdivisions of the computational domain. By adapting the domain decomposition based on the solution's features using a separable POUNet, the framework can automatically learn partitions that are stable and can be geometrically interpretable. We test this approach on the two-dimensional Helmholtz equation, two-dimensional Poisson equation with multiscale features, and viscous Burgers equation and observe improved accuracy compared with FBPINNs with
    
[^173]: 超越准确率：潜在意图下的信息论解码器评估

    Beyond Accuracy: Information-Theoretic Decoder Evaluation Under Latent Intent

    [https://arxiv.org/abs/2611.05658](https://arxiv.org/abs/2611.05658)

    该论文将脑-机器人控制形式化为POMDP，证明了解码器性能存在由最差目标对切诺夫信息$\Gamma_{\min}$决定的相变边界——所有目标对可区分时遗憾值一致有界，否则线性增长，并据此提出闭式遗憾预测器，超越了传统的单步准确率评估指标。

    

    辅助性脑-机器人控制中的一个核心挑战是从复杂且嘈杂的脑信号中解码用户的潜在意图。以往的方法将解码器视为单次试验分类器，并用基于准确率的指标进行评估，但这忽略了辅助控制固有的序列性。由于规划器必须随时间整合间歇性的、含噪声的解码器输出，单步准确率在预测下游任务性能时只能作为一种代理指标。我们将脑-机器人控制形式化为部分可观测马尔可夫决策过程（POMDP），并证明存在一个尖锐的相变：只要任意目标对都能被解码器区分，遗憾值在整个时间范围内一致有界；否则遗憾值将线性增长。我们证明，一个单一的信道统计量——最差目标对的切诺夫信息（Chernoff information）$\Gamma_{\min}$——决定了这一边界。随后，我们推导出一个闭式遗憾预测器 $U_{\mathrm{id}}(C)$，它可以从任意（摘要在此处截断）

    arXiv:2611.05658v1 Announce Type: new  Abstract: A central challenge in assistive brain-to-robot control is decoding a user's latent intent from complex and noisy brain signals. Prior approaches evaluate decoders as single-trial classifiers with accuracy-based metrics, but this neglects the inherently sequential nature of assistive control. Since a planner must integrate intermittent, noisy decoder outputs over time, single-step accuracy can only be a proxy when predicting downstream task performance. We formalize brain-to-robot control as a partially observable Markov decision process (POMDP), and prove that there is a sharp phase transition: regret is uniformly bounded across the horizon as long as pairs of goals are all decoder-distinguishable, but grows linearly otherwise. We show that a single channel statistic, the worst-pair Chernoff information $\Gamma_{\min}$ sets this boundary. Then, we derive a closed-form regret predictor $U_{\mathrm{id}}(C)$, that can be computed from any 
    
[^174]: 两个SAE潜在变量何时是同一特征？

    When Are Two SAE Latents the Same Feature?

    [https://arxiv.org/abs/2611.05655](https://arxiv.org/abs/2611.05655)

    该论文提出了一个基于能量加权最优传输的统一框架，用于比较稀疏自编码器的特征对应关系、字典差异与统计不确定性，并提供了渐近有效的联合置信推断方法。

    

    比较稀疏自编码器（SAE）需要连接特征对应关系、整个字典之间的差异，以及由有限评估数据带来的不确定性。我们开发了一个框架，通过在指定的相似性几何下的能量加权最优传输方案将这些任务联系起来。同一个传输方案定义了字典层面的差异度量，将其分解为各个潜在变量的贡献，并量化每个潜在变量的质量在其对应项之间的分布。我们刻画了解码器相似性、激活相似性与一阶因果相似性之间的关系；理想化模型澄清了对分裂（splitting）与吸收（absorption）现象的解释。对于固定的已训练SAE，我们推导了传输成本与比例的联合渐近推断方法，其中考虑了激活相似性以及从同一批文档中估计的两个能量边际。基于文档的联合自助法（bootstrap）给出了渐近有效的同步置信带。我们在四种语言上进行了实验验证。

    arXiv:2611.05655v1 Announce Type: new  Abstract: Comparing sparse autoencoders (SAEs) requires connecting feature correspondence, differences between whole dictionaries, and uncertainty from finite evaluation data. We develop a framework that connects these tasks through an energy-weighted optimal-transport plan under a specified similarity geometry. The same plan defines a dictionary-level discrepancy, decomposes it into latent contributions, and quantifies how each latent's mass is distributed across counterparts. We characterize relationships between decoder, activation, and first-order causal similarities; idealized models clarify the interpretation of splitting and absorption. For fixed trained SAEs, we derive joint asymptotic inference for transport costs and fractions, accounting for activation similarities and both energy marginals estimated from the same documents. A joint document bootstrap gives asymptotically valid simultaneous confidence bands. Experiments across four lang
    
[^175]: 量化脑电图（EEG）基础模型与信号处理嵌入中的特质-状态层次结构

    Quantifying Trait-State Hierarchy in EEG Foundation Model and Signal Processing Embeddings

    [https://arxiv.org/abs/2611.05652](https://arxiv.org/abs/2611.05652)

    该论文提出用层次结构度量和kNN纯度量化EEG嵌入中特质与状态信息的编码方式，发现大多数EEG特征呈现“宏观特质聚类、微观状态嵌套其中”的层次结构。

    

    脑电图（EEG）通过记录头皮上的电位来测量大脑活动。EEG记录捕捉信号中两组不同的特征：（1）特质——在一次记录过程中保持相对恒定的信息，例如受试者身份、年龄和性别；（2）状态——编码对外部刺激响应的特征，包括视觉、听觉和运动信息。目前，人们对不同的EEG建模特征（信号处理方法与基础模型）如何编码状态和特质特征的理解仍然有限。在这项工作中，我们使用层次结构度量指标和kNN纯度，在脑机接口（BCI）数据集试验上量化EEG嵌入所捕获的状态级和特质级特征，并发现大多数EEG特征形成了宏观特质层次结构（即嵌入形成特质级聚类，其内部包含微观状态）。我们的线性探测结果表明……

    arXiv:2611.05652v1 Announce Type: new  Abstract: Electroencephalography (EEG) measures brain activity through recording electrical potentials of the scalp. EEG recordings capture two distinct sets of characteristics in the signal: (1) \emph{Traits} -- information that remains relatively constant over the course of a recording session, such as subject identity, age, and sex; (2) \emph{States} -- signatures that encode responses to external stimuli, including visual, auditory, and motor information. There is limited understanding of how different EEG modeling features (signal processing and foundation models) encode state and trait characteristics. In this work, we quantify state- and trait-level features captured by EEG embeddings using a hierarchy metric and kNN purity on Brain-Computer Interface dataset trials, and find that a majority of EEG features form a macro-trait hierarchy (embeddings form trait-level clusters with micro-states within). Our linear probe results indicate that th
    
[^176]: 通过耦合超参数缩放实现推荐系统数据高效的缩放定律发现

    Data-Efficient Scaling-Law Discovery for Recommendation Systems via Coupled Hyperparameter Scaling

    [https://arxiv.org/abs/2611.05643](https://arxiv.org/abs/2611.05643)

    该论文提出通过缩放层将低维超超参数映射到完整架构空间，并结合缩放感知代理模型与帕累托前沿选择，仅需少量实验即可高效发现任意推荐排序模型的缩放定律。

    

    大型Web平台的排序与推荐模型与大语言模型类似，同样受益于计算量的增加，其缩放行为大致呈幂律形式。然而，排序模型具有异构架构，这使得其缩放定律的确定比大语言模型更为复杂。我们研究如何高效地发现任意排序模型的缩放定律。我们引入了一个缩放层，它将底层的架构参数划分为语义上连贯的组，并定义这些组之间如何协同变化，从而建立起从低维“超超参数”空间到完整架构空间的映射。在此基础上，Axia拟合缩放感知的代理模型，基于后验的采集准则选择配置，并返回帕累托前沿，从中验证每个模型的具体工作点——用少量信息量丰富的实验取代穷举式扫描。一个面向模型集群的评估协议

    arXiv:2611.05643v1 Announce Type: cross  Abstract: Large-scale ranking and recommendation models of Web platforms benefit from increased compute much as large language models do, and their scaling behavior is broadly power-law in form. Yet ranking models have heterogeneous architectures which makes their scaling law less straightforward than that of LLMs. We address how to discover the scaling law of an arbitrary ranking model efficiently. We introduce a scaling layer, which groups primitive architecture parameters into semantically coherent groups and defines how they co-vary, inducing a mapping from a low-dimensional space of hyper-hyperparameters into the full architecture space. On top of it, Axia fits scaling-informed surrogates, selects configurations by a posterior-based acquisition criterion, and returns a Pareto frontier from which per-model operating points are validated -- replacing exhaustive sweeps with a small number of informative runs. A fleet-level evaluation protocol,
    
[^177]: 面向视频识别的时空多阶交互

    Spatio-Temporal Multi-Order Interaction for Video Recognition

    [https://arxiv.org/abs/2611.05633](https://arxiv.org/abs/2611.05633)

    该论文提出基于逐深度卷积的线性复杂度多阶时间建模与时空双重交互机制，以更低的计算开销实现更优的视频时空表示。

    

    许多用于视频识别的时间建模方法依赖于显式的跨通道聚合投影，其复杂度随通道数呈二次方增长，带来了显著的计算开销。此外，用于视频识别的时空表示要么偏向局部阶交互（如CNN），要么偏向全局阶交互（如Transformer）。基于MogaNet的多阶空间表示，我们的方法引入了两个互补的贡献来解决这些问题。首先，在时间建模方面，我们引入了仅使用逐深度卷积的并行多阶处理以及基于隐式门控的聚合方式，其复杂度随通道数呈线性增长，从而大幅降低了计算量。其次，我们提出了空间维度与时间维度之间的双重交互机制，能够在不增加计算成本的情况下获得更好的时空表示。

    arXiv:2611.05633v1 Announce Type: cross  Abstract: Many temporal approaches for video recognition rely on explicit cross-channel aggregation projections, which have quadratic complexity in the number of channels, introducing significant computational overhead. In addition, the spatio-temporal representation used for video recognition is biased either toward local order interactions (for CNNs) or toward global order interactions (for transformers). Starting from the multi-order spatial representation of MogaNet, our method introduces two complementary contributions to tackle these problems. First, for the temporal modelling, we introduce parallel multi-order processing with depthwise-only convolutions and implicit gating-based aggregation, which is linear in the number of channels, inducing a considerable computational reduction. Second, we propose a dual interaction between spatial and temporal dimensions, which allows a better spatio-temporal representation without increasing the comp
    
[^178]: LACS：用于动态数据剪枝的分布敏感、标签无关的置信度选择

    LACS: Distribution-Sensitive Label-Agnostic Confidence Selection for Dynamic Data Pruning

    [https://arxiv.org/abs/2611.05616](https://arxiv.org/abs/2611.05616)

    提出了LACS，一种基于最大预测置信度、无需标签和逐样本梯度的动态数据剪枝方法，通过分布敏感的方式同时自适应调整阈值与保留样本数量，在七种评估设置中均验证了其有效性。

    

    在大规模数据集上进行训练促使我们选择哪些样本能够获得昂贵的训练计算资源。固定配额方法会预先确定剪枝的数量，而绝对置信度阈值则依赖于分数的尺度。我们研究了一种基于总体相对性的替代方法，该方法能够同时自适应地调整阈值和保留样本数量。我们提出了LACS（标签无关置信度选择器），这是一种基于最大预测置信度的训练时规则，在选择时既不需要标签，也不需要逐样本的梯度。LACS会移除分数达到或超过log-mean-exp参考值加上一个余量的样本。我们刻画了该方法对分数差距和分布质量的敏感性，并推导出了精确的不剪枝条件。对于0<ρ<1，移除的比例至多为ρ，且至少保留一个样本。由于采用无符号分数，该方法能够同时移除高置信度正确和高置信度错误的样本。评估涵盖七个设置，包括神经网络训练、语言建模和提升方法。

    arXiv:2611.05616v1 Announce Type: new  Abstract: Training on large datasets motivates selecting which examples receive expensive training computation. Fixed quotas determine the amount of pruning in advance, while absolute confidence cutoffs depend on the score scale. We study a population-relative alternative that adapts the threshold and retained count together. We introduce LACS (Label-Agnostic Confidence Selector), a training-time rule based on maximum predictive confidence, requiring neither labels at selection time nor per-example gradients. LACS removes examples at or above a log-mean-exp reference plus a margin. We characterize its sensitivity to score gaps and mass, and derive an exact no-pruning condition. For $0<\rho<1$, the removed fraction is at most $\rho$ and at least one example remains. The unsigned score permits both confidently correct and confidently wrong examples to be removed. Seven evaluation settings include neural training, language modeling, and boosting. Neu
    
[^179]: 论在线策略蒸馏在强化学习后能力巩固中的泛化局限性

    On the Generalization Limitation of On-Policy Distillation for Post-RL Consolidation

    [https://arxiv.org/abs/2611.05609](https://arxiv.org/abs/2611.05609)

    在线策略蒸馏（OPD/MOPD）虽能近乎完美地复现教师模型的分布内性能，但无法完全迁移其泛化能力，分布外表现甚至低于基础模型，揭示了该方法在强化学习后能力巩固中的泛化局限。

    

    多教师在线策略蒸馏（MOPD）将在线策略蒸馏（OPD）扩展为从多个教师模型中巩固能力。尽管先前的工作报告了近乎完美的分布内（IND）蒸馏性能，我们发现这可能掩盖了监督分布之外能力巩固不完全的问题。一项针对合成推理的初步研究表明，OPD 无法完全迁移教师模型的泛化能力。随后，我们将这一研究扩展到真实世界任务，使用 Qwen3 模型在数学推理、编程、智能体和长上下文能力方面评估了外推泛化与组合泛化。在外推设置中，OPD 和 MOPD 达到了其教师模型 IND 分数的 96.04%–102.73%，但其分布外（OOD）分数仍比基础模型低 2.45%–16.25%。在三个组合设置中，MOPD 在分布内性能上优于基础模型，但其在组合任务上的表现（摘要在此处截断）。

    arXiv:2611.05609v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation (MOPD) extends on-policy distillation (OPD) to consolidate capabilities from multiple teachers. Although prior work reports near-perfect in-distribution (IND) distillation performance, we find that this can mask incomplete capability consolidation beyond the supervised distribution. A preliminary study on synthetic reasoning shows that OPD does not fully transfer teachers' generalization capabilities. We then extend this investigation to real-world tasks, evaluating extrapolative and compositional generalization across mathematical reasoning, coding, agent, and long-context capabilities using Qwen3 models. Across extrapolative settings, OPD and MOPD reach 96.04-102.73% of their teachers' IND scores. Yet their out-of-distribution (OOD) scores remain 2.45-16.25% below the base model's. Across three compositional settings, MOPD improves IND performance over the base model, but its performance on composed
    
[^180]: 沿局部退火路径的传输：Wasserstein几何与多模态采样

    Transport along Local Annealing Paths: Wasserstein Geometry and Multimodal Sampling

    [https://arxiv.org/abs/2611.05605](https://arxiv.org/abs/2611.05605)

    该论文揭示了几何退火路径在Wasserstein空间中可能远长于端点距离从而拖慢传输映射学习，并提出分量退火传输（CAT），通过将目标精确分解为混合分量并沿各分量退火路径学习独立传输映射来减少模式间质量传输，缓解多模态采样中的模式坍塌。

    

    用于多模态采样的变分传输映射学习方法可能会出现模式坍塌。缓解这一问题的一种途径是在由几何退火定义的、由中间分布构成的给定路径上学习传输。我们证明退火在Wasserstein空间中可能会绕出相当大的弯路：当概率质量在各个模式之间移动时，路径长度可能远大于其端点之间的距离。我们进一步将路径几何与传输学习联系起来，表明较长的退火路径会拖慢种群优化，而足够正则的路径则允许快速收敛。受这些发现的启发，我们提出了分量退火传输（CAT），以减少模式间质量传输的需求。CAT 从一个融入模式信息的混合参考分布出发，构造目标的精确混合分解，并沿各分量的退火路径分别学习独立的传输映射。我们估计分量权重，从而……（摘要原文在此处截断）

    arXiv:2611.05605v1 Announce Type: cross  Abstract: Variational approaches to learning transport maps for multimodal sampling can suffer from mode collapse. One way to mitigate this problem is to learn transport along a prescribed path of intermediate distributions defined by geometric annealing. We show that annealing can take a substantial detour in Wasserstein space: when mass moves between modes, the path can be much longer than the distance between its endpoints. We further link path geometry to transport learning, showing that long annealing paths can slow population optimization, while sufficiently regular paths permit rapid convergence. Motivated by these findings, we propose componentwise annealed transport (CAT) to reduce the need for mass transfer between modes. Starting from a mode-informed mixture reference, CAT constructs an exact mixture decomposition of the target and learns separate transport maps along componentwise annealing paths. We estimate the component weights se
    
[^181]: RIDGE：面向深度强化学习游戏智能体行为覆盖的状态条件奖励混合方法

    RIDGE: State-Conditioned Reward Blending for Behavioral Coverage in Deep RL Game Agents

    [https://arxiv.org/abs/2611.05601](https://arxiv.org/abs/2611.05601)

    RIDGE 通过状态条件奖励混合器和多头评论家架构，让单个深度强化学习游戏智能体同时掌握多种行为风格，从而规避了 PPO 在非平稳奖励下的性能退化问题。

    

    在固定“人格”奖励上训练的深度强化学习智能体只会以一种方式行事：探索者在饥饿时仍继续探索，幸存者在空地图上囤积资源。由于每种人格都固定了自身的奖励，自动化游戏测试需要为 N 种风格训练 N 个智能体。而在回合中途改变奖励组成则会陷入“切换-稳定性”两难：PPO 在非平稳奖励下会发生性能退化。我们提出 RIDGE，一种通过两个设计选择规避该两难问题的单智能体架构：一是状态条件混合器，从缓慢变化的状态变量计算出经 softmax 归一化的 sigmoid 权重；二是多头评论家，为每个人格在其自身的平稳信号上分别训练一个价值头。在包含四种人格的 Crafter 环境中，单个 RIDGE 智能体同时掌握了木器级别制作/战斗分支的两条路径，而没有任何固定人格基线能够同时解决这两者；锐度消融实验证实状态条件化是性能提升的关键。我们将这些结果定位为混合方法能力的演示。

    arXiv:2611.05601v1 Announce Type: cross  Abstract: A deep RL agent trained on a fixed persona reward plays one way: an Explorer keeps exploring while starving, a Survivor hoards on an empty map. Since each persona fixes its own reward, automated playtesting trains N agents for N styles. Changing reward composition mid-episode runs into the switching-stability dilemma: PPO degrades under non-stationary rewards. We present RIDGE, a single-agent architecture that sidesteps the dilemma with two design choices: a state-conditioned blender that computes softmax-normalized sigmoid weights from slow-moving state variables, and a multi-head critic that trains one value head per persona on its own stationary signal. On Crafter with four personas, one \ridge agent acquires both branches of the wood-tier crafting/combat split that no fixed-persona baseline solves jointly, and a sharpness ablation confirms state-conditioning drives the gain. We position these results as demonstrations of the blendi
    
[^182]: 相关性传播作为Filippov神经网络上的熵最优传输：图像分类器中的Sinkhorn传播与区域相关性

    Relevance Propagation as Entropic Optimal Transport on Filippov Neural Networks: Sinkhorn Propagation and Region Relevance in Image Classifiers

    [https://arxiv.org/abs/2611.05594](https://arxiv.org/abs/2611.05594)

    该论文证明神经网络中的相关性传播（LRP）本质上是熵最优传输变分问题的解，通过将前馈网络形式化为Filippov算子的有向无环图，统一了各种LRP规则并提出了权重可随节点活动变化的LRPN规则。

    

    相关性传播将网络的预测结果逐层向后重新分配，通常被视为一种启发式方法。我们证明它实际上是某一单一变分问题的解：分层相关性传播（LRP）是激活计算图上的无发散流，其相对于前向激活流的后向转移律具有零相对熵；对于非负参考，这对应于后向路径上马尔可夫测度之间的相对熵。LRP₀、LRP$_{z^+}$和LRP$_{α,β}$等规则通过选择参考而由此原理导出，且都是单一规则LRPN$_{α,β}$的实例，该规则的权重可随节点而变化（例如随节点的活动而变化），这体现了Hebb原理的精神。该原理建立在前馈网络作为Filippov算子构成的有向无环图的测度论形式化基础之上，涵盖了分段光滑和不连续的网络架构……

    arXiv:2611.05594v1 Announce Type: new  Abstract: Relevance propagation, which redistributes a network's prediction backward through the layers, is usually treated as a heuristic. We show that it is the solution of a single variational problem: Layer-wise Relevance Propagation (LRP) is the divergence-free flow on the active computational graph of zero relative entropy to the backward transition law of the forward activation flow; for a non-negative reference, this is a relative entropy between Markov measures on backward paths. The rules LRP$_0$, LRP$_{z^+}$ and LRP$_{\alpha,\beta}$ follow from this principle by the choice of a reference, and are instances of one rule, LRPN$_{\alpha,\beta}$, whose weights may vary from node to node, for instance with the node's activity, in the spirit of Hebb's principle. The principle rests on a measure-theoretic formalization of feed-forward networks as directed acyclic graphs of Filippov operators, which covers piecewise-smooth and discontinuous arch
    
[^183]: GaussianRenderFormer：基于Transformer的3D高斯泼溅渲染

    GaussianRenderFormer: Transformer-based Rendering of 3D Gaussian Splats

    [https://arxiv.org/abs/2611.05593](https://arxiv.org/abs/2611.05593)

    提出GaussianRenderFormer，一个完全去除光栅化器、利用Transformer编码器-解码器结构从3D高斯泼溅和相机参数直接端到端渲染图像的神经网络，并通过诊断分析揭示了视图解码器中投影足迹推断的性能瓶颈。

    

    神经网络能否在没有光栅化器的情况下，自主学习渲染3D高斯泼溅？我们提出了GaussianRenderFormer，这是一个Transformer模型，它接收一组高斯和一个相机，直接输出图像。每个高斯在视图无关的编码器中成为一个token，而视图相关的解码器通过对这些token进行注意力操作，将小块的相机光线转换为像素。在26,820个Objaverse物体（每个剪枝至20,000个高斯）上训练后，GaussianRenderFormer对300个未见过的物体进行渲染，相对于完整未剪枝的泼溅达到38.08 dB PSNR，而对相同剪枝输入进行光栅化则可达到44.02 dB。这一差距并非泛化能力问题，我们的诊断结果与视图解码器中的瓶颈相一致：向解码器提供每个高斯的投影足迹信息（原则上它可以自行推断）可额外带来1.4 dB的提升。训练不足的可能性尚未排除。

    arXiv:2611.05593v1 Announce Type: cross  Abstract: Can a neural network learn to render 3D Gaussian splats on its own, with no rasterizer in the renderer? We present GaussianRenderFormer, a transformer that takes a set of Gaussians and a camera and outputs the image directly. Each Gaussian becomes one token in a view-independent encoder, and a view-dependent decoder turns small patches of camera rays into pixels by attending to those tokens. Trained on 26,820 Objaverse objects pruned to 20,000 Gaussians each, GaussianRenderFormer renders 300 unseen objects at 38.08 dB PSNR against the full, unpruned splats, while rasterizing the same pruned input reaches 44.02 dB. The gap is not one of generalization, and our diagnostics are consistent with a bottleneck in the view decoder: telling it the projected footprint of each Gaussian, which it could in principle infer, adds 1.4 dB. Under-training is not ruled out. Code, trained models and videos: https://svlwoof.github.io/gaussianrenderformer
    
[^184]: 利用机器学习天气模型区分初始条件误差与模型误差

    Distinguishing Initial-Condition and Model Error Using Machine-Learning Weather Models

    [https://arxiv.org/abs/2611.05592](https://arxiv.org/abs/2611.05592)

    本研究通过在GraphCast和盘古气象大模型上对比门控与非门控优化方法，利用机器学习天气模型实现初始条件误差与模型误差的区分。

    

    在GraphCast和盘古气象大模型中评估了门控与非门控优化方法。第一张图（左上）显示了36个日期的平均结果；其余图显示了2017年1月至11月每月首日12 UTC的优化下降过程。纵轴指标为相对于ERA5验证的第1-10天Z500均方根误差（RMSE）之和，以各模型相对于日期匹配对照组的百分比变化表示。门控分支将初始条件与LoRA权重校正进行联合优化；其GraphCast曲线仅使用初始条件（去除LoRA），即盘古模型接收到的相同状态。星号和菱形分别标记了5天和10天窗口内盘古的最低误差。每张图上方的百分比显示了门控（绿色）和非门控（红色）优化相对于对照组的盘古最佳改进。当提前停止优化时，非门控方法在一半的日期上优于门控方法。门控并不能保证盘古模型性能更好，但它确实能产生更可比较的GraphCast和（原文在此截断）

    arXiv:2611.05592v1 Announce Type: cross  Abstract: Gated and ungated optimizations evaluated in GraphCast and Pangu-Weather. The first panel (upper-left) shows the mean over the 36 dates; remaining panels show first-of-month 12 UTC optimization descents from January-November 2017. The y-axis metric is the summed day 1-10 Z500 RMSE verified against ERA5, expressed as percent change from each model's date-matched control. The gated arm optimizes the initial conditions jointly with a LoRA weight correction; its GraphCast curve uses the initial conditions alone (LoRA removed), the same state Pangu receives. Stars and diamonds mark the lowest Pangu errors in the 5- and 10-day windows, respectively. Percentages above each panel show the best Pangu improvement relative to control for gated (green) and ungated (red) optimization. Ungated outperforms gated on half of dates when stopped early. Gating does not guarantee a better Pangu performance, but it does produce more comparable GraphCast and
    
[^185]: C-GRPO：保形组相对策略优化

    C-GRPO: Conformal Group Relative Policy Optimization

    [https://arxiv.org/abs/2611.05591](https://arxiv.org/abs/2611.05591)

    提出C-GRPO，用经校准的保形阈值替代GRPO的固定采样预算，为自适应采样停止规则提供有限样本覆盖率保证，从而在不浪费计算的情况下按提示难度自适应分配采样资源。

    

    面向推理的语言模型训练越来越依赖多次在策略采样，但标准的组相对策略优化（GRPO）使用固定的组大小，在简单提示上浪费计算资源，而在困难提示上探索不足。现有的自适应预算方法缓解了这种低效问题，但其停止规则是启发式的，无法提供有限样本校准保证。我们提出CGRPO，用基于任务特定非一致性分数的经校准的保形阈值取代标准GRPO的固定采样预算，结合了用于闭式推理的自适应预测集（APS）式校准、用于执行式代码的首个成功分数、自动的误覆盖水平选择以及周期性重新校准。我们证明了所得停止规则的有限样本覆盖率，并表明在单调策略改进条件下，重新校准会使期望训练预算非递增。在跨越六个基准的实验中……

    arXiv:2611.05591v1 Announce Type: new  Abstract: Reasoning-oriented language model training increasingly relies on multiple on-policy rollouts, but standard Group Relative Policy Optimization (GRPO) uses a fixed group size, wasting compute on easy prompts and under-exploring hard ones. Existing adaptive-budget methods alleviate this inefficiency, yet their stopping rules are heuristic and do not provide a finite-sample calibration guarantee. We propose CGRPO, which replaces the fixed rollout budget of standard GPRO with calibrated conformal thresholds over task-specific nonconformity scores, combining Adaptive Prediction Set (APS)-style calibration for closed-form reasoning, a first-success score for execution-based code, automatic miscoverage selection, and periodic re-calibration. We prove finite-sample coverage for the resulting stopping rule and show that, under monotone policy improvement, recalibration induces a non-increasing expected training budget. Across six benchmarks spann
    
[^186]: 探索差分隐私自注意力变分自编码器中的隐私与数据规模权衡

    Exploring the Privacy--Size Tradeoff for Differentially Private Self-Attentive Variational Autoencoders

    [https://arxiv.org/abs/2611.05582](https://arxiv.org/abs/2611.05582)

    本文提出了SAVANT——首个在单一表格模型中结合注意力机制、变分自编码器与差分隐私的自编码器，并发现 在相同的差分隐私保证下，使用更多训练数据通常能带来更高的合成数据效用。

    

    数据在共享时往往最有用，而最值得共享的数据通常正是受隐私义务限制的数据。由在差分隐私（DP）下训练的生成器产生的合成数据是调和二者矛盾的一种方式，但其效用取决于隐私预算 ε 以及生成器所接触的数据量——后者是大多数合成数据共享应用可以控制的一个参数。我们提出了 SAVANT，这是一种针对表格数据、使用 DP-SGD 训练的自注意力变分自编码器，据我们所知，这是首个在单一表格模型中同时结合注意力机制、VAE 和差分隐私的模型。我们在两个公开数据集上，联合考察隐私预算和数据集规模两个维度，将其与两种边际机制 PrivBayes 和 MST 进行对比评估，衡量下游分类器因使用合成数据而非真实数据训练所损失的效用。我们发现，在相同的差分隐私保证下，更多的数据通常意味着更高的效用。对于边际……（摘要在此处截断）

    arXiv:2611.05582v1 Announce Type: new  Abstract: Data tends to be most useful when it is shared, and the data most worth sharing is often the data that privacy obligations restrict. Synthetic data from a generator trained under differential privacy (DP) is one way to reconcile the two, but its utility depends on the privacy budget $\varepsilon$ and on how much data the generator sees - a parameter that most data sharing applications of synthetic data control. We present SAVANT, a self-attentive variational autoencoder for tabular data trained with DP-SGD that, to our knowledge, is the first to combine attention, a VAE, and DP in one tabular model. We evaluate it against two marginal mechanisms, PrivBayes and MST, jointly over the privacy budget and the dataset size on two public datasets, measuring the utility a downstream classifier loses by training on synthetic rather than real data. We find that, under the same DP guarantee, more data generally means more utility. For the marginal 
    
[^187]: 预算约束博弈中的学习型玩家与优化型玩家

    Learning vs. Optimizing Players in Budgeted Games

    [https://arxiv.org/abs/2611.05580](https://arxiv.org/abs/2611.05580)

    该论文针对预算约束下的重复博弈提出了一个高效且预算可行的学习算法，能够同时实现不可操纵性——任何自适应优化者的期望收益至多为预算 Stackelberg 均衡值乘以 $T$ 加上 $O(\sqrt{T})$——以及当优化者采用固定混合策略时的低随机遗憾。

    

    我们研究了持续 $T$ 轮的重复一般博弈，博弈双方分别为一个效用最大化的优化者与一个学习智能体，其中预算约束将双方在各轮中的决策耦合在一起。在没有预算的情况下，低交换遗憾可以同时保证学习者获得次线性遗憾，并将优化者的收益上限限制在其 Stackelberg 值。而预算打破了这一图景：对抗性的无悔学习变得不可能实现，因此自然的目标是两个方面：一是不可操纵性——即任何自适应的优化者都不能获得超过 $\mathtt{BSE} \cdot T + o(T)$ 的收益，其中 $\mathtt{BSE}$ 是其预算 Stackelberg 均衡值；二是当优化者采用固定混合策略时实现低随机遗憾。我们提出了一个高效的、满足预算可行性的学习算法，该算法在每个预算博弈中都能同时实现上述两个目标：任何自适应优化者（无论其是否受预算约束）的期望收益至多为 $\mathtt{BSE} \cdot T + O(\sqrt{T})$，并且当优化者在常数个区间中的每一个区间上采用固定混合策略时……（摘要原文在此处被截断）

    arXiv:2611.05580v1 Announce Type: cross  Abstract: We study repeated general games over $T$ rounds between a utility-maximizing optimizer and a learning agent, where budget constraints couple the players' decisions across rounds. Without budgets, low swap regret simultaneously guarantees the learner sublinear regret and caps the optimizer at its Stackelberg value. Budgets break this picture: adversarial no-regret becomes impossible, so the natural goals are non-manipulability---no adaptive optimizer earns more than $\mathtt{BSE} \cdot T + o(T)$, where $\mathtt{BSE}$ is its Budgeted Stackelberg Equilibrium value---and low stochastic regret when the optimizer plays a fixed mixed strategy.   We give an efficient, budget-feasible learning algorithm that achieves both goals in every budgeted game: any adaptive optimizer, budgeted or not, earns at most $\mathtt{BSE} \cdot T + O(\sqrt{T})$ in expectation, and if the optimizer plays a fixed mixed strategy on each of a constant number of interv
    
[^188]: 炎症性肠病中全切片图像与临床数据的多模态融合方法

    Multimodal Fusion Methods for Whole-Slide Images and Clinical Data in Inflammatory Bowel Disease

    [https://arxiv.org/abs/2611.05579](https://arxiv.org/abs/2611.05579)

    该论文提出了一个模块化框架，在统一条件下系统比较15种多模态融合方法，将全切片图像嵌入与临床数据结合，用于预测炎症性肠病患者的缓解结局，结果显示最佳融合方法在多数场景下优于单模态模型。

    

    多模态AI融合组织学与临床数据是计算病理学中的一个活跃研究领域，但融合方法很少在相同条件下、跨不同临床终点进行比较。我们提出了一个模块化框架，其中切片级嵌入（通过在冻结的基础模型所产生的补丁嵌入上训练基于注意力的聚合器获得）与每次就诊的临床嵌入被输入到15种可互换的融合方法中，这些方法共享输出维度和预测头（决策级方法除外）。使用一个新诊断炎症性肠病患者的前瞻性队列，我们从基线或随访就诊预测符合STRIDE-II标准的缓解结局。最佳融合方法的宏F1达到0.57–0.75，在六个场景中的四个场景中优于两种单模态模型，且其表现因结局和输入不同而有所差异。图像依赖性，即当切片嵌入被重新（原文此处截断）……

    arXiv:2611.05579v1 Announce Type: cross  Abstract: Multimodal AI fusion of histology and clinical data is an active research area in computational pathology, but fusion methods are rarely compared under identical conditions and across different clinical endpoints. We present a modular framework in which slide-level embeddings, obtained by training an attention-based aggregator over patch embeddings from a frozen foundation model, and per-visit clinical embeddings are fed into 15 interchangeable fusion methods with a shared output dimension and prediction head (except decision-level methods). Using a prospective cohort of newly diagnosed inflammatory bowel disease patients, we predicted STRIDE-II-aligned remission outcomes from baseline or follow-up visits. The best fusion method reached macro-F1 of 0.57-0.75, outperforming both single-modality models in four of six scenarios, and differed by outcome and input. Image reliance, the drop in balanced accuracy when the slide embedding is re
    
[^189]: 可读性不等于可操控性：对学习到的流体表征中物理结构的预注册测试

    Readable Is Not Steerable: Pre-Registered Tests of Physical Structure in Learned Fluid Representations

    [https://arxiv.org/abs/2611.05577](https://arxiv.org/abs/2611.05577)

    该研究通过预注册实验发现，自监督模型学到的流体表征虽然具有线性可解码的“可读性”，但并不围绕物理量组织——JEPA表现不优于未训练网络，而普通自编码器反而在最慢潜在方向上突出涡量耗散，且添加VICReg正则化会削弱这种物理结构。

    

    自监督世界模型常被认为会围绕物理量来组织其潜在表示。我们在二维衰减的纳维-斯托克斯流动上通过三项预注册研究检验了这一说法，每项研究仅在保留数据上测试一次。在研究1中，训练用于预测流场的联合嵌入预测架构（JEPA）未能满足预注册标准，即其潜在表示围绕流体量的组织程度并不优于相同架构的未训练网络（假设不被支持）；而在非物理线性动力学上训练的JEPA，以及从未接触时间信息的静态不变性模型，表现得与其相当甚至更好。在研究2中，渲染流场的普通自编码器使其最慢的无监督潜在方向上涡量耗散（enstrophy）变得显著（最佳相关系数 r 为 0.80-0.86，而未训练网络为 0.38；采用全新随机种子和封存数据）。添加 VICReg 在所有比较中都降低了这种显著性，尽管在涡度方面未达到预注册的差距，同时提高了自编码器的线性可解码性（部分支持……）

    arXiv:2611.05577v1 Announce Type: cross  Abstract: Self-supervised world models are often said to organize their latents around physical quantities. We test this on 2D decaying Navier-Stokes flow in three pre-registered studies, each tested once on held-out data. In Study 1, a joint-embedding predictive architecture (JEPA) trained to predict the flow fails the pre-registered criteria for organizing its latent around fluid quantities better than an untrained network of the same architecture (not supported); a JEPA trained on non-physical linear dynamics and a static invariance model that never sees time do as well or better. In Study 2, plain autoencoders that render the field make enstrophy prominent in their slowest unsupervised latent directions (best r 0.80-0.86 vs 0.38 untrained; fresh seeds, sealed data). Adding VICReg lowers this prominence in every comparison, though not by the pre-registered margin for vorticity, while raising the autoencoders' linear decodability (partially su
    
[^190]: 基于角色的认知不确定性估计在训练3D激光雷达点云语义分割模型中的有效性

    Role-based Effectiveness of an Epistemic Uncertainty Estimate in Training Models for Semantic Segmentation of 3D LiDAR Point Clouds

    [https://arxiv.org/abs/2611.05569](https://arxiv.org/abs/2611.05569)

    该论文研究将显式建模的认知不确定性以不同角色（作为额外输入特征或作为训练损失中的逐样本置信度权重）融入3D激光雷达点云语义分割模型训练中的有效性。

    

    点云分割是三维数据分析中的一项基础任务，其应用涵盖机器人技术、增强现实与虚拟现实以及遥感等领域。然而，点云固有的稀疏性、噪声和无序性带来了重大挑战，尤其是在实现可靠分类方面。尽管最先进的方法依赖于在训练模型中对不确定性进行显式建模，但在探索通过生成式方法估计的不确定性所能扮演的不同角色方面仍存在研究空白。本工作通过显式建模认知不确定性并将其用于不同角色（即作为特征和用于损失函数中），研究了确定性不确定性对点云分割的影响。我们在四个大规模激光雷达基准数据集上评估了一个基线人工神经网络（ANN）以及两个不确定性感知变体（不确定性作为额外的输入特征，以及不确定性作为训练损失中逐样本的置信度权重）。

    arXiv:2611.05569v1 Announce Type: cross  Abstract: Point cloud segmentation is a fundamental task in 3D data analysis, with applications across robotics, augmented and virtual reality, and remote sensing. However, the inherent sparsity, noise, and unordered nature of point clouds introduce significant challenges, particularly in achieving reliable classification. While the state-of-the-art rely on training models for explicit modeling of uncertainty, there is a gap in exploring different roles that estimated uncertainty using generative approaches can play. This work investigates the impact of deterministic uncertainty in point cloud segmentation by explicitly modeling epistemic uncertainty and using it in different roles, namely, as a feature and in loss function. We evaluate a baseline artificial neural network (ANN) and two uncertainty-aware variants (uncertainty as an additional input feature and as a per-sample confidence weight in the training loss) on four large-scale LiDAR benc
    
[^191]: 生成模型下多目标强化学习的近极小极大样本复杂度

    Near-Minimax Sample Complexity for Multi-Objective Reinforcement Learning under Generative Models

    [https://arxiv.org/abs/2611.05562](https://arxiv.org/abs/2611.05562)

    本文建立了生成模型下多目标强化学习的常数置信度近极小极大样本复杂度 $\widetilde\Theta\!\left(\frac{SAH^3}{\varepsilon^2}\min\{S,\log(em)\}\right)$，证明单个数据集即可在不进一步采样的情况下同时支持所有奖励凸组合的 $\varepsilon$-最优策略。

    

    在多目标强化学习中，偏好可能只有在数据收集完成后才能确定，这要求一个数据集能够支持连续的奖励权衡。我们研究了具有未知转移、$S$ 个状态、$A$ 个动作以及 $m$ 个取值于 $[0,1]$ 的已知奖励函数的有限折扣马尔可夫决策过程，并假设学习者可以访问生成模型。学习者必须在不进行进一步采样的情况下，为这些奖励的每一个凸组合提供 $\varepsilon$-最优策略，且该保证需同时适用于所有偏好和初始状态。记 $H=(1-\gamma)^{-1}$，我们建立了常数置信度的极小极大样本复杂度 $\widetilde\Theta\!\left( \frac{SAH^3}{\varepsilon^2}\min\{S,\log(em)\} \right)$，其中被省略的因子是关于 $S,A,H,\varepsilon^{-1}$ 的对数因子，而对 $m$ 的依赖则是显式的。因此，在这些对数因子范围内，支持……

    arXiv:2611.05562v1 Announce Type: cross  Abstract: In multi-objective reinforcement learning, preferences may be specified only after data collection, requiring one dataset to support a continuum of reward trade-offs. We study finite discounted Markov decision processes with unknown transitions, $S$ states, $A$ actions, and $m$ known reward functions taking values in $[0,1]$, given access to a generative model. The learner must provide an $\varepsilon$-optimal policy for every convex combination of these rewards without further sampling, with a guarantee that holds simultaneously over all preferences and initial states. Writing $H=(1-\gamma)^{-1}$, we establish the constant-confidence minimax sample complexity $\widetilde\Theta\!\left(   \frac{SAH^3}{\varepsilon^2}\min\{S,\log(em)\} \right)$, where the suppressed factors are logarithmic in $S,A,H,\varepsilon^{-1}$ and the dependence on $m$ is explicit. Thus, up to these logarithmic factors, the worst-case statistical cost of supporting
    
[^192]: 支持向量机直接最小化KL散度

    Support Vector Machines Directly Minimise KL Divergence

    [https://arxiv.org/abs/2611.05557](https://arxiv.org/abs/2611.05557)

    本文证明支持向量分类与回归的对偶目标等价于两个高斯测度之间的带约束KL散度最小化，从而为标准核机器建立了统一的信息论解释。

    

    支持向量机（SVM）从根本上建立在几何间隔最大化之上，目前缺乏主流的概率解释。这使其在理论上与那些天然可比较、可组合的概率机器学习模型相互隔离。在本文中，我们通过证明支持向量分类和支持向量回归的对偶目标函数都是两个高斯测度之间精确的、带约束的Kullback-Leibler（KL）散度最小化任务，来弥合这一差距。我们进一步证明，当去除定义域约束并施加 $L^2$ 正则化时，同样的散度形式可以恢复核岭回归的精确目标函数。通过证明SVM的对偶问题直接最小化相对熵，我们的结果为标准核机器提供了一个统一的信息论模型，并为在概率框架下统一更广泛的基于核的算法奠定了基础。

    arXiv:2611.05557v1 Announce Type: new  Abstract: Support vector machines (SVMs) are fundamentally grounded in geometric margin maximisation and currently lack a mainstream probabilistic interpretation. This keeps them theoretically isolated from probabilistic machine learning models that are naturally comparable and combinable. In this paper, we bridge this gap by proving that the dual objectives of both support vector classification and support vector regression are exact, constrained Kullback-Leibler (KL) divergence minimisation tasks between two Gaussian measures. We further show that this same divergence formulation recovers the exact objective function for kernel ridge regression when domain constraints are removed and $L^2$ regularisation is applied. By demonstrating that SVM duals directly minimise relative entropy, our results provide a unified information-theoretic model for standard kernel machines and lay the foundation for unifying broader kernel-based algorithms under prob
    
[^193]: FedSharp：噪声信道下的拜占庭鲁棒去中心化联邦学习

    FedSharp: Byzantine Robust Decentralized Federated Learning Under Noisy Channel

    [https://arxiv.org/abs/2611.05546](https://arxiv.org/abs/2611.05546)

    提出FedSharp框架，通过维护2b个前驱节点模型、坐标式中值偏差筛选可疑模型并对损失最低的b个候选模型取平均，在噪声信道下有效抵抗拜占庭攻击与信道噪声累积，实现鲁棒的去中心化联邦学习。

    

    去中心化联邦学习（DFL）消除了中央聚合服务器，但仍然容易受到数据异构性、拜占庭攻击和信道噪声的影响。我们证明，随着模型在节点间传播，噪声会不断累积，并且随着学习率的衰减，这种累积噪声的影响变得越来越显著。此外，累积噪声使拜占庭攻击者能够通过模仿诚实模型的平均值来利用基于损失的模型选择机制。为应对这些挑战，我们提出了FedSharp——一种鲁棒的去中心化联邦学习框架，即筛选式、跳数感知、排序、近端平均方法。在该方法中，每个节点维护来自其2b个前驱节点的模型，从而在至多b个节点为拜占庭节点的情况下，确保至少有b个诚实的候选模型。FedSharp使用坐标式中值偏差来筛选可疑模型，对损失最低的b个候选模型进行平均，从而将新鲜信道噪声的方差降低b倍，并应用噪声自适应梯度范数正则化和近端训练来缓解……

    arXiv:2611.05546v1 Announce Type: new  Abstract: Decentralized federated learning (DFL) eliminates the central aggregation server but remains vulnerable to data heterogeneity, Byzantine attacks, and channel noise. We show that noise accumulates as models propagate across nodes, becoming increasingly significant as the learning rate decays. Moreover, accumulated noise enables Byzantine attackers to exploit loss-based model selection by mimicking honest model averages. To address these challenges, we propose screened, hop-aware, ranked, proximal averaging, a robust DFL framework (FedSharp). In this approach, each node maintains models from its 2b predecessors, ensuring at least b honest candidates when at most b nodes are Byzantine. FedSharp screens suspicious models using coordinate-wise median deviations, averages the b lowest-loss candidates to reduce fresh channel-noise variance by a factor of b, and applies noise-adaptive gradient-norm regularization and proximal training to mitigat
    
[^194]: 基于Gibbs概率测度的训练数据集聚合与精确机器遗忘统一框架

    A Unified Framework for Training-Dataset Aggregation and Exact Machine Unlearning via Gibbs Probability Measures

    [https://arxiv.org/abs/2611.05544](https://arxiv.org/abs/2611.05544)

    本文提出了Gibbs概率测度上重归一化、归一化对数线性组合与嵌套三种操作的统一理论框架，证明了嵌套与归一化对数线性组合的等价性，并据此实现了有监督机器学习中的训练数据集聚合与精确机器遗忘。

    

    本文研究了Gibbs概率测度上的三种操作：重归一化、归一化对数线性组合以及嵌套。结果表明，所得测度仍是Gibbs概率测度，且其目标函数、参考测度和正则化因子均具有显式表达式。进一步地，这些测度被刻画为涉及原始目标函数线性组合的优化问题的唯一解，其中优化问题受相对熵正则化或相对熵约束的限制。当参考测度为概率测度时，这三种操作获得了涉及毕达哥拉斯恒等式与指数测地线的几何解释。本文的一个核心结果是证明了Gibbs概率测度的嵌套与归一化对数线性组合之间的等价性。随后，这些结果被应用于采用Gibbs算法的有监督机器学习，即训练数据集聚合与精确机器遗忘（摘要在此处截断）。

    arXiv:2611.05544v1 Announce Type: cross  Abstract: In this paper, three operations on Gibbs probability measures are studied: renormalization, normalized log-linear combination, and nesting. The resulting measures are shown to be Gibbs probability measures with explicit expressions for their objective functions, reference measures, and regularization factors. These measures are further characterized as unique solutions to optimization problems involving linear combinations of the original objective functions, subject to either relative entropy regularization or relative entropy constraints. When the reference measures are probability measures, geometric interpretations of these three operations are established involving Pythagorean identities and exponential geodesics.A central result is the equivalence between nesting and normalized log-linear combinations of Gibbs probability measures. These results are then applied to supervised machine learning with Gibbs algorithms, which are repr
    
[^195]: 多变量保形预测中的广义超矩形预测集

    Generalized Hyperrectangular Prediction Sets in Multivariate Conformal Prediction

    [https://arxiv.org/abs/2611.05543](https://arxiv.org/abs/2611.05543)

    该论文提出广义超矩形预测集及一个矩形化框架，能将任意连续的多变量非一致性分数或条件密度转化为具有有限样本覆盖保证、且易于解释和可视化的超矩形预测集。

    

    在多变量保形预测中，许多非一致性分数所产生的预测集具有复杂的几何形状（例如椭球或密度水平集），难以解释和可视化。超矩形通过为每个输出提供一个区间，提供了一种实用的替代方案。我们将这一概念扩展到任意一维集合的笛卡尔积，并称之为广义超矩形。我们引入了一个矩形化框架，能够将任何连续的多变量非一致性分数或条件密度转换为可产生具有有限样本覆盖保证的广义超矩形预测集的分数。我们建立了该矩形化框架的关键几何性质，并就广义超矩形提供了有益的直观理解。我们进一步建立了一个表示定理，表明基于坐标分数最大聚合的一大类超矩形保形方法可以被解释为矩形化（框架的特例）。

    arXiv:2611.05543v1 Announce Type: cross  Abstract: In multivariate conformal prediction, many nonconformity scores yield prediction sets with complex geometry (e.g., ellipsoids or density level sets) that are difficult to interpret and visualize. Hyperrectangles offer a practical alternative by providing one interval per output. We extend this notion to Cartesian products of arbitrary one-dimensional sets, which we call generalized hyperrectangles. We introduce a rectangularization framework that turns any continuous multivariate nonconformity score, or conditional density, into one yielding generalized hyperrectangular prediction sets with finite-sample coverage. We establish key geometric properties of our rectangularization framework and provide useful intuitions on generalized hyperrectangles. We further establish a representation theorem showing that a broad class of hyperrectangular conformal methods based on max-aggregated coordinate-wise scores can be interpreted as rectangular
    
[^196]: 将视觉语言模型重塑为用于临床神经诊断的脑电图基础模型

    Recasting Visual Language Models as EEG Foundation Models for Clinical Neurodiagnostics

    [https://arxiv.org/abs/2611.05538](https://arxiv.org/abs/2611.05538)

    该研究证明，无需大规模脑电预训练的通用视觉语言模型能够像临床医生一样以视觉方式解读脑电信号，并在临床神经诊断任务上取得有竞争力的表现，从而为构建脑电图基础模型提供了摆脱对大规模脑电数据依赖的新范式。

    

    历史上，自动化脑电图（EEG）系统是通过在专门标注的数据上训练特定任务的监督模型来实现的。较新的建模范式则依赖于对未标注的脑电信号进行大规模领域特定预训练，从而为多个特定模态的任务创建统一表示，即“基础模型”。这种统一表示的质量直接取决于特定模态数据集的可获得性。与此同时，通用视觉语言模型（VLM）展现出广泛的少样本学习能力，以及对不同视觉和语言任务良好的微调适应性。那么，视觉语言模型能否学会以视觉方式阅读和解读脑电图，在某种程度上模拟人类临床医生检查脑电模式的方式？在这项工作中，我们论证了它们可以做到：我们展示了未经大规模脑电图预训练的通用视觉语言模型，在一系列临床神经……（原文摘要至此截断）

    arXiv:2611.05538v1 Announce Type: new  Abstract: Historically automated electroencephalography (EEG) systems was done using task-specific supervised models trained on purposely annotated data. A more recent modeling paradigm relies on creating a unified representation, or a 'foundation model', for multiple modality-specific tasks through large-scale domain-specific pretraining on the unannotated EEG signal. The quality of this unified representation directly depends on the availability of the modality-specific datasets. At the same time, generalist Vision Language Models (VLMs) exhibit broad few-shot learning capacities as well as good fine-tuning adaptability to different visual and language tasks. Can VLMs learn how to read and interpret EEG visually, emulating to a degree the way human clinicians inspect the EEG patterns? In this work we make a case that they can: we show that generalist VLMs without large-scale EEG pretraining perform competitively across a collection of clinical n
    
[^197]: 置信度分数：一种新的核度量方法

    The Confidence Score: a New Measure on Kernels

    [https://arxiv.org/abs/2611.05528](https://arxiv.org/abs/2611.05528)

    本文提出了一种名为“置信度分数”的新核度量方法，通过衡量真实标签编码与核投影后真实标签之间的对齐程度，使保留类内结构的核也能获得高分，克服了传统理想核对齐度量要求同类样本必须完全相似的局限。

    

    在广泛的应用场景中，核度量被用于调整超参数、在候选核中进行选择，以及评估核对于特定学习任务的适用性。对于分类任务，核度量被用于预测哪些核可能产生准确的核分类器。许多现有的核度量方法，例如与理想核的对齐度，旨在奖励那些将同一类别的所有样本视为相似的核。虽然这种方法在许多场景中行之有效，但对于分类任务而言，这一要求是不必要的严格限制，因为类别内部可能存在大量的内部结构。本文提出了“置信度分数”，这是一种新的核度量方法。与理想核对齐度量不同，置信度分数衡量的是真实标签的编码与经过核投影后的真实标签之间的对齐程度，从而允许保留类内结构的核获得高分。

    arXiv:2611.05528v1 Announce Type: new  Abstract: Across a wide range of applications, kernel measures are used to tune hyperparameters, select among candidate kernels, and assess the suitability of kernels for specific learning tasks. For classification, kernel measures are used to predict which kernels may produce accurate kernel classifiers. Many existing kernel measures, such as alignment to the ideal kernel, are designed to reward kernels which captures all samples from the same class as similar. While effective in many settings, this requirement is unnecessarily restrictive for classification, where classes may exhibit substantial internal structure. In this work, the "Confidence Score", a new kernel measure, is proposed. Unlike alignment-to-the-ideal kernel measures, the Confidence Score measures the alignment between an encoding of the ground truth and that ground truth projected by the kernel, allowing kernels that preserve intra-class structure to achieve high scores. Through 
    
[^198]: 匹配的高阶Hölder光滑性与一致凸性下的紧致下界

    Tight Lower Bounds under Matched High-Order H\"older Smoothness and Uniform Convexity

    [https://arxiv.org/abs/2611.05527](https://arxiv.org/abs/2611.05527)

    当高阶Hölder光滑阶数与一致凸阶数匹配（q = p + ν）时，任何确定性算法将初始最优性间隙Δ0降至ε都需要 Ω((H/σ)^(2/(3(p+ν)-2)) log(Δ0/ε)) 次oracle调用，该紧致下界与上界匹配，是Nesterov强凸函数经典下界向高阶一致凸情形的推广。

    

    本文在阶数匹配的条件下，为最小化高阶Hölder光滑且一致凸的函数提供了紧致下界。具体而言，我们考虑其p阶导数满足次数为ν、参数为H的Hölder连续性，且具有次数为q、参数为σ的一致凸性的函数。当这些阶数匹配时，即 q = p + ν，我们证明任何可访问至多p阶导数的确定性算法，都需要 Ω((H/σ)^(2/(3(p+ν)-2)) log(Δ0/ε)) 次oracle调用，才能将初始最优性间隙Δ0降低到ε。该下界与相应的上界匹配，建立了Nesterov针对强凸函数的经典一阶下界在高阶一致凸情形下的对应版本。我们没有直接推广Nesterov的经典二次链构造，而是开发了一种多……（摘要在此处截断）

    arXiv:2611.05527v1 Announce Type: cross  Abstract: In this paper, we provide tight lower bounds for minimizing high-order H\"older-smooth and uniformly convex functions when their orders match. Specifically, we consider functions whose $p^{th}$-order derivatives are H\"older continuous with degree $\nu$ and parameter $H$, and that are uniformly convex with degree $q$ and parameter $\sigma$. When these orders match, namely, $q = p + \nu$, we show that any deterministic algorithm with access to derivatives up to order $p$ requires $\Omega\left(\left(\frac{H}{\sigma}\right)^{\frac{2}{3(p+\nu)-2}}\log\frac{\Delta_0}{\epsilon}\right)$ oracle calls to reduce an initial optimality gap of $\Delta_0$ to $\epsilon$. This matches the corresponding upper bound and establishes a higher-order uniformly convex counterpart to Nesterov's classical first-order lower bound for strongly convex functions. Rather than directly generalizing Nesterov's classical quadratic-chain construction, we develop a mult
    
[^199]: TSFM 遇见 LLM：作为协变量的上下文

    TSFM Meets LLM: Context as Covariates

    [https://arxiv.org/abs/2611.05515](https://arxiv.org/abs/2611.05515)

    该论文提出 CoCo 预测框架，让 LLM 将文本推理转化为数值化的预测与置信度协变量来引导 TSFM 骨干模型，融合两者优势进行时间序列预测，并提供利用 LLM 标注真实时间序列以生成多模态数据集的可扩展方法。

    

    文本信息在现实世界的时间序列数据集中无处不在，以新闻、报告和注释等形式出现，其中蕴含着仅凭数值历史无法推断出的关键预测信号。大语言模型（LLM）能够解读文本但缺乏数值精度，而时间序列基础模型（TSFM）在数值上精确却无法对文本的影响进行推理。此外，大规模成对多模态数据集的缺失也限制了弥合这一差距的研究进展。在本工作中，我们通过提供一种可扩展的方法来应对这一挑战，即利用 LLM 对真实世界的时间序列进行标注，从而生成逼真的多模态数据集。在此基础上，我们提出了“上下文作为协变量”预测框架 CoCo，其中 LLM 将其文本推理提炼为基于数值的预测协变量和置信度协变量，用以引导 TSFM 骨干模型，从而充分发挥 LLM 与 TSFM 各自的优势。我们将两种模型在……

    arXiv:2611.05515v1 Announce Type: cross  Abstract: Textual information is ubiquitous in real-world time series datasets, appearing as news, reports, and annotations that hold key predictive signals not deducible from numerical history alone. Large Language Models (LLMs) interpret text but lack numerical precision, while Time Series Foundation Models (TSFMs) are quantitatively accurate but cannot reason about the effects of text. Moreover, the lack of large-scale paired multimodal datasets limits progress in bridging this gap. In this work, we address this challenge by providing a scalable approach for generating realistic multimodal datasets by annotating real-world time series with LLMs. Building on this, we propose Context as Covariates (CoCo), a forecasting framework in which an LLM distills its textual reasoning into numerically grounded forecast and confidence covariates that guide a TSFM backbone, thereby exploiting the strengths of both LLMs and TSFMs. We align the two models in
    
[^200]: Ctrl-F：以力作为人形机器人移动操作的交互接口

    Ctrl-F: Force as an Interface for Humanoid Loco-Manipulation

    [https://arxiv.org/abs/2611.05507](https://arxiv.org/abs/2611.05507)

    提出了Ctrl-F分层运动-力框架，通过在指定接触部位施加显式的三维力指令来增强人形机器人运动跟踪，使机器人不仅能模仿人类动作，还能以指定力度与环境中的物体进行交互。

    

    人形机器人运动跟踪已经发展到能够重现多样化人类动作的通用策略，但由此产生的行为仅限于自由空间中：跟踪器知道身体应该处于什么位置，却不知道应该如何与环境进行交互。其他研究工作虽然将物体纳入了优化目标之中，但将这种交互抽象为接触或几何关系，这使得具有挑战性的场景仍未得到充分探索。我们认为，力是连接运动模仿与物体交互之间的自然指令通道，它不仅规定了在哪里进行接触，还规定了用多大的力。我们提出了Ctrl-F，这是一个分层的运动-力框架，它通过显式的力指令来增强运动跟踪，即在每个指定的接触部位（如手或脚）上施加一个三维力。该分层框架包括：(1) 一个统一的运动-力控制器，在跟踪参考运动的同时向外界施加指令力；(2) 一个负责发出力指令的高层力规划器（摘要在此处不完整）

    arXiv:2611.05507v1 Announce Type: cross  Abstract: Humanoid motion tracking has scaled to general policies that reproduce diverse human motion, but the resulting behavior is in free-space: the tracker knows where the body should be, not how it should interact with the environment. Other work makes the object part of the objective, yet abstracts that interaction to contact or geometry, which leaves the demanding cases under-explored. We argue that force is a natural command channel between motion imitation and object interaction, one that specifies not just where to make contact but how hard. We present Ctrl-F, a hierarchical motion-force framework that augments motion tracking with an explicit force command, a 3-D force at each designated contact body such as a hand or a foot. The hierarchical framework comprises: (1) A unified motion-force controller tracks the reference motion while applying the commanded force to the world. (2) A high-level force planner that issues a force command 
    
[^201]: 面向多目标对齐的目标条件化主动偏好学习

    Objective-Conditioned Active Preference Learning for Multi-Objective Alignment

    [https://arxiv.org/abs/2611.05506](https://arxiv.org/abs/2611.05506)

    提出 OGC-DPO 主动学习框架，将（回复对，目标）组合视为独立查询并利用策略自身隐式梯度衡量信息量，主动选择最能扩展各目标梯度覆盖范围的比较样本，从而在无需外部奖励模型的情况下降低多目标对齐的标注成本。

    

    将大型语言模型（LLM）与多个目标（如有用性、无害性和诚实性）进行对齐，通常需要对每个目标分别进行偏好判断，使标注成本随目标数量线性增长。主动偏好学习可以通过选择那些标签预期对策略改变最大的比较来降低这一成本。现有的方法要么缺乏面向多目标设置的目标感知查询选择机制，要么将相互冲突的偏好视为有害信息并将其排除在训练数据之外。我们提出了 OGC-DPO（目标条件化梯度覆盖 DPO），这是一个主动学习框架，它将每个（回复对，目标）组合视为一个独立的查询，并通过策略自身的隐式梯度来衡量其信息量，而无需依赖外部奖励模型。OGC-DPO 随后选择最能扩展每个目标梯度覆盖范围的查询。

    arXiv:2611.05506v1 Announce Type: cross  Abstract: Aligning large language models (LLMs) with several objectives, such as helpfulness, harmlessness, and honesty, typically relies on preference judgment for each objective, growing the annotation cost linearly with the number of objectives. Active preference learning can reduce this cost by selecting the comparisons whose labels are expected to change the policy most. Existing remedies either lack an objective-aware query selection mechanism for the multi-objective setting or regard conflicting preferences as harmful, therefore excluding them from the training data. We introduce OGC-DPO (Objective-conditioned Gradient Coverage DPO), an active learning framework that treats each (response pair, objective) combination as a separate query and measures its informativeness through the policy's own implicit gradient, without relying on an external reward model. OGC-DPO then selects the queries that most expand each objective's gradient coverag
    
[^202]: 条件型瞬时量子多项式时间Born机的经典训练

    Classical training of conditional instantaneous quantum polynomial-time Born machines

    [https://arxiv.org/abs/2611.05483](https://arxiv.org/abs/2611.05483)

    该论文提出将条件编码于输入态、跨条件共享单一IQP酉算子的条件IQP Born机，实现可经典无偏估计条件MMD目标与梯度的量子生成模型，且其表达能力超越了同等量子比特数下对角条件化的IQP模型。

    

    瞬时量子多项式时间（IQP）Born机是一类颇具吸引力的量子生成模型，因为某些IQP电路族被猜想难以经典采样模拟，但它们却可以被经典地训练。先前提出的条件IQP Born机使对角层的参数依赖于条件。我们提出了一种条件IQP Born机，其中条件被编码在输入态中，并且一个单一的IQP酉算子在各个条件之间共享。只需对进入对角层的参考态进行高效采样和振幅比求值，即可获得条件最大均值差异目标函数及其梯度的无偏经典估计。我们证明，输入态条件化能够表示某些条件分布族，而任何在相同量子比特数目下采用对角条件化的IQP模型都无法复现这些分布族。对于十六量子比特的居里-外斯自旋模型，单一的条件模型即可捕获磁化转变……

    arXiv:2611.05483v1 Announce Type: cross  Abstract: Instantaneous quantum polynomial-time (IQP) Born machines are attractive quantum generative models, as certain IQP circuit families are conjectured to be hard to sample, yet they can be trained classically. Previously proposed conditional IQP Born machines make the parameters of the diagonal layer condition dependent. We introduce conditional IQP Born machines, in which the condition is encoded in the input state and a single IQP unitary is shared across conditions. Efficient sampling and amplitude-ratio evaluation for the reference state entering the diagonal layer suffice to obtain unbiased classical estimates of a conditional maximum mean discrepancy objective and its gradients. We show that input conditioning can represent conditional families that no diagonally conditioned IQP model with the same number of qubits can reproduce. For the sixteen-qubit Curie-Weiss spin model, a single conditional model captures the magnetization tran
    
[^203]: 面向CAD边界表示的SO(3)等变学习

    SO(3)-Equivariant Learning on CAD Boundary Representations

    [https://arxiv.org/abs/2611.05449](https://arxiv.org/abs/2611.05449)

    SO3brep是一种作用于B-rep面邻接图的SO(3)等变图神经网络，通过Wigner D矩阵和Clebsch-Gordan张量积在消息传递中保留相邻面间的相对朝向，从而对随机旋转的CAD模型实现鲁棒的面分割与分类。

    

    在边界表示（B-rep）上进行学习需要对零件姿态具有鲁棒性，因为CAD模型并非以规范朝向存储。现有的B-rep网络要么在世界坐标系下使用坐标和法向量，从而在旋转下性能退化；要么在局部参考系中对每个面进行编码。局部参考系虽然能带来不变性，但会把面间消息传递简化为标量，从而丢弃相邻面之间的相对朝向信息。我们提出SO3brep，一种作用于B-rep面邻接图的SO(3)等变图神经网络。节点特征在局部参考系中计算，并通过Wigner D矩阵映射到共享坐标系中。Clebsch-Gordan张量积将这些特征与相邻面之间的方向耦合起来，使相对朝向信息在整个消息传递过程中得以保留，同时输出保持SE(3)不变性。在三个面分割基准和两个分类基准上（测试零件经过随机旋转），SO3brep取得……（摘要在此截断）

    arXiv:2611.05449v1 Announce Type: cross  Abstract: Learning on boundary representations (B-reps) requires robustness to the pose of a part, since CAD models are not stored in a canonical orientation. Existing B-rep networks either operate on world-frame coordinates and normals, and thus degrade under rotation, or encode each face in a local reference frame. Local frames yield invariance, but reduce inter-face messages to scalars and discard the relative orientation of adjacent faces. We propose SO3brep, an SO(3)-equivariant graph neural network on the B-rep face-adjacency graph. Node features are computed in local frames and mapped into a shared frame by Wigner D-matrices. Clebsch-Gordan tensor products couple these features with the directions between adjacent faces, so that relative orientation is preserved throughout message passing and the output remains SE(3)-invariant. Across three face-segmentation and two classification benchmarks with randomly rotated test parts, SO3brep achie
    
[^204]: SurgTransfer：面向手术世界模型中动作迁移的多步潜在动作建模

    SurgTransfer: Multi-Step Latent Action Modeling for Action Transfer in Surgical World Model

    [https://arxiv.org/abs/2611.05443](https://arxiv.org/abs/2611.05443)

    本文提出SurgTransfer框架，通过多步潜在动作建模从无标注手术视频中学习上下文不变的潜在动作，从而在无需真实运动学数据或专家标注的情况下实现手术世界模型中的专家动作迁移。

    

    手术世界模型有潜力提供术中指导、加速无风险训练，并促进专家手术操作的迁移。然而，利用世界模型来迁移手术专长在很大程度上尚未被探索。此外，该领域近期的方法主要依赖于特定任务的真实运动学动作或专家标注，这限制了其在缺乏真实标注的狭窄任务范围之外的应用。相比之下，自然图像领域的现有工作在仅从无标注视频中学习潜在动作并跨上下文迁移所学动作方面已展现出令人鼓舞的成果。我们观察到，这一范式对于手术领域而言是短视的，无法捕捉手术过程的层次化与目标导向结构。因此，我们提出了一种新颖的多步动作学习框架，用于学习上下文不变的潜在动作，从而（摘要至此截断）。

    arXiv:2611.05443v1 Announce Type: cross  Abstract: Surgical world models have the potential to provide intra-operative guidance, accelerate risk-free training and facilitate transfer of expert surgical maneuvers. However, the use of world models for transferring surgical expertise has been largely unexplored. Moreover, recent approaches in this domain have mainly relied on ground truth kinematics actions or expert annotations for specific tasks, limiting their applicability beyond the narrow range of tasks with ground-truth annotations. In contrast, existing works in the natural domain have shown promising results in learning latent actions solely from unlabeled videos and transferring the learned actions across contexts. We observe that this paradigm is short-sighted for the surgical domain, failing to capture the hierarchical and goal-directed structure of surgical procedures. Thus, we propose a novel multi-step action learning framework to learn context-invariant latent actions that
    
[^205]: 匹配但并不更优：六个临床数据集上量子机器学习的等效性检验基准及QML-Health-READY报告框架

    Matched but Not Better: An Equivalence-Tested Benchmark of Quantum Machine Learning on Six Clinical Datasets and the QML-Health-READY Reporting Framework

    [https://arxiv.org/abs/2611.05442](https://arxiv.org/abs/2611.05442)

    该研究通过严格的配对等效性检验，在六个临床数据集上证明量子机器学习模型（QSVM和VQC）并不优于调优后的经典机器学习模型，并提出QML-Health-READY八领域报告框架以规范QML临床研究的报告标准。

    

    背景：量子机器学习（QML）被推广用于临床预测，但与传统模型的比较很少采用配对设计或进行等效性检验。方法：我们在六个公开临床数据集上，将量子核支持向量机（QSVM；4量子比特，调优带宽）和变分量子分类器（VQC）与五个调优后的经典模型进行比较。所有模型使用相同的四分量输入和25次交叉验证划分。采用校正的重采样t检验、经Holm校正的等效性检验（边际：±0.02和±0.05）以及随机效应合并分析来评估配对AUROC差异。此外还进行了消融实验、噪声、有限采样次数和实现故障分析。我们提出了QML-Health-READY，一个包含八个领域的报告框架。结果：两种量子模型均未显著优于任何经典对照模型。在各数据集上合并后，QSVM与核匹配的径向基函数（RBF）SVM的差异仅为-0.006

    arXiv:2611.05442v1 Announce Type: new  Abstract: Background: Quantum machine learning (QML) is promoted for clinical prediction, but comparisons with classical models are rarely paired or tested for equivalence. Methods: We compared a quantum-kernel support vector machine (QSVM; 4 qubits, tuned bandwidth) and a variational quantum classifier (VQC) with five tuned classical models across six public clinical datasets. All models used identical four-component inputs and 25 cross-validation splits. Paired AUROC differences were assessed using a corrected resampled t-test, Holm-adjusted equivalence tests (margins: +/-0.02 and +/-0.05), and random-effects pooling. Ablation, noise, finite-shot, and implementation-fault analyses were conducted. We propose QML-Health-READY, an eight-domain reporting framework. Results: Neither quantum model significantly outperformed any classical comparator. Pooled across datasets, QSVM differed from the kernel-matched radial basis function (RBF) SVM by -0.006
    
[^206]: 学习动力学仿真中的隐式求解

    Learning Implicit Solves for Dynamic Simulations

    [https://arxiv.org/abs/2611.05439](https://arxiv.org/abs/2611.05439)

    提出隐式神经算子求解器INOS，仅从静态平衡解预训练神经算子并嵌入Newmark-β时间积分，无需动力学轨迹监督即可零样本实现高精度动力学仿真，并在非线性仿真中获得超过4600倍加速。

    

    动力学系统的高保真仿真是科学发现与工程设计的核心，但其计算成本依然高昂。本文提出隐式神经算子求解器（INOS），它学习隐式求解过程并编码时间积分器。INOS从静态平衡解中预训练一个神经算子，并将其复用于Newmark-β积分中。在每个时间增量步，积分器构造一个有效平衡问题，预训练的神经算子推断下一步的解场，再由Newmark关系推进动力学状态。线性和非线性弹性动力学的数值算例表明，该方法在无需动力学轨迹监督的情况下即可实现精确的全场预测。对比结果显示，经零样本评估的INOS在预测精度上优于在动力学仿真上训练的神经求解器基线。此外，INOS在非线性仿真中实现了超过4600倍的加速。

    arXiv:2611.05439v1 Announce Type: new  Abstract: High-fidelity simulation of dynamical systems is central to scientific discovery and engineering design, yet remains computationally demanding. Here we introduce the Implicit Neural Operator Solver (INOS), which learns the implicit solve and encodes a time integrator. INOS pretrains a neural operator from static equilibrium solutions and reuses it within Newmark-$\beta$ integration. At each time increment, the integrator forms an effective equilibrium problem, the pretrained neural operator infers the next-step solution field, and the Newmark relations advance the dynamic state. Numerical examples in linear and nonlinear elastodynamics demonstrate accurate full-field predictions without dynamic trajectory supervision. Comparative results show that INOS, evaluated zero-shot, achieves higher predictive accuracy than neural solver baselines trained on dynamic simulations. In addition, INOS delivers over 4,600$\times$ speedup in nonlinear si
    
[^207]: HLA-Net：一种基于物理引导的骨架化框架，用于天文图像中引力弧的自动检测

    HLA-Net: A Physics-Guided Skeleton-Based Framework for Automatic Gravitational Arc Detection in Astronomical Images

    [https://arxiv.org/abs/2611.05438](https://arxiv.org/abs/2611.05438)

    提出了一种无需人工标注数据的可解释多阶段框架HLA-Net，通过结合引力透镜的几何形态与物理特性，实现天文图像中引力弧的自动检测，并将候选搜索空间从169个缩减至83个。

    

    强引力透镜产生的引力弧为研究暗物质分布、星系团质量重建以及宇宙结构提供了独特的途径。由于引力弧非常稀少、通常较为暗淡，且在形态上与拉长的星系相似，其自动检测仍然具有挑战性。在本文中，我们提出了HLA-Net（混合透镜弧网络），这是一个可解释的多阶段自动引力弧检测框架，它显式地利用了引力透镜的几何形态和物理特性，而无需人工标注的训练数据集。该框架在包含复杂星系团结构的天文FITS图像上进行了评估。从有序骨架表示中提取的169个初始弧候选开始，多阶段验证流程将搜索空间缩减至83个几何一致的候选对象。

    arXiv:2611.05438v1 Announce Type: cross  Abstract: Gravitational arcs produced by strong gravitational lensing provide a unique approach for investigating dark matter distribution, galaxy cluster mass reconstruction and the structure of the Universe. Automated detection remains challenging because gravitational arcs are rare, often faint and morphologically similar to elongated galaxies.In this paper, we propose \textbf{HLA-Net (Hybrid Lens Arc Network)}, an interpretable multi-stage framework for automatic gravitational arc detection that explicitly exploits both the geometric morphology and physical characteristics of gravitational lensing without requiring manually annotated training datasets.The proposed framework was evaluated on astronomical FITS images containing complex galaxy cluster structures. Starting from 169 initial arc candidates extracted from ordered skeleton representations, the multi-stage verification pipeline reduced the search space to 83 geometrically consistent 
    
[^208]: 面向谱图生成的保约束流匹配

    Constraint-Preserving Flow Matching for Spectral Graph Generation

    [https://arxiv.org/abs/2611.05436](https://arxiv.org/abs/2611.05436)

    提出了LaSFM方法，在谱图生成过程中通过在约束Stiefel流形上进行流匹配，同时保持拉普拉斯零空间约束、特征向量正交性和正特征值，并可扩展至属性引导的图生成。

    

    谱图生成通过图拉普拉斯矩阵的特征值和特征向量来建模图，为直接图构建提供了一种紧凑且连续的替代方案。然而，这些特征系统必须保留图拉普拉斯矩阵的结构性质。现有方法虽然保留了特征向量的正交归一性，但并未强制执行拉普拉斯零空间约束，可能产生与拉普拉斯结构不一致的特征系统。为解决这一问题，我们提出了拉普拉斯约束Stiefel流匹配，在生成过程中同时保持零空间条件、特征向量正交归一性以及正特征值。我们刻画了底层的约束Stiefel流形，并在其上构建流匹配。我们进一步将LaSFM扩展到属性引导的生成，在保持这些约束的同时，将样本引导至期望的图属性。实验表明在谱生成任务上取得了性能提升。

    arXiv:2611.05436v1 Announce Type: cross  Abstract: Spectral graph generation models graphs through the eigenvalues and eigenvectors of their Laplacians, providing a compact and continuous alternative to direct graph construction. However, these eigensystems must retain the structural properties of graph Laplacians. Existing approaches preserve eigenvector orthonormality but do not enforce the Laplacian nullspace constraint, potentially producing eigensystems inconsistent with Laplacian structure. To address this, we introduce Laplacian-Constrained Stiefel Flow Matching (LaSFM), which preserves the nullspace condition alongside eigenvector orthonormality and positive eigenvalues during generation. We characterize the underlying constrained Stiefel manifold and formulate flow matching on it. We further extend LaSFM to property-guided generation, steering samples toward desired graph properties while maintaining these constraints. Experiments demonstrate improved performance on both spect
    
[^209]: BIOTRAM：基于生物化学信息的Transformer模型，用于通过元条形码技术估计相对丰度

    BIOTRAM: Biochemistry-informed Transformer for Relative Abundance via Metabarcoding

    [https://arxiv.org/abs/2611.05418](https://arxiv.org/abs/2611.05418)

    BIOTRAM是一种融合生物化学知识的Transformer神经网络，通过学习每个标本的DNA贡献量和扩增效率来校正元条形码读段计数偏差，从而实现从批量样本中准确估计节肢动物各分类群的相对丰度。

    

    节肢动物的大规模监测始于对批量样本的元条形码处理：陷阱采集的数百至数千个标本作为一个整体单元处理，从批量标本中提取DNA，然后对所有标本的DNA进行混合测序。元条形码能够揭示样本中包含哪些分类群，但由于DNA扩增速率和体型大小等因素的差异，其读段计数并不能准确对应不同分类群之间的相对丰度。这种相对丰度信息的缺失限制了此类数据在众多依赖该信息的生态分析中的实用性。我们提出了BIOTRAM：一种神经网络，它通过将观测到的读段计数除以学习得到的每个标本的读段产出量来估计各分类群的标本数量，该读段产出量被分解为每个标本所贡献的DNA量以及该DNA的扩增效率。在一个包含1,188个瓶子（80万个标本）的配对数据集上，这些样本同时通过元条形码处理（摘要在此处截断）

    arXiv:2611.05418v1 Announce Type: cross  Abstract: Arthropod monitoring at scale begins with metabarcoding of bulk samples: trap collections of hundreds to thousands of specimens processed as one unit, extraction of DNA from the bulked specimens, then sequencing of the DNA from all specimens together. Metabarcoding reveals which taxa such a sample contains, but due to factors such as differing DNA amplification rates and body sizes, its read counts do not neatly correspond to how abundant the different taxa are relative to one another. This lack of relative abundance limits the data's utility for the multitude of ecological analyses that depend on it. We propose BIOTRAM: a neural network that estimates per-taxon specimen counts by dividing the observed read counts by a learned read yield per specimen, which it factors into the DNA each specimen contributes and how efficiently that DNA is amplified. On a paired dataset of 1,188 bottles (800 k specimens) processed both by metabarcoding a
    
[^210]: VenusGenome：面向微生物基因组表示学习的基因组感知掩码自编码器

    VenusGenome: Genome-Aware Masked Autoencoding for Microbial Genome Representation Learning

    [https://arxiv.org/abs/2611.05405](https://arxiv.org/abs/2611.05405)

    提出VenusGenome，一种基因分辨率的基因组感知掩码自编码器，通过高比例掩码与融合重叠群边界、链方向和基因位置的注意力机制，在超过170万个微生物基因组上预训练，以提升微生物基因组表示学习和功能预测能力。

    

    基因组上下文模型已经表明，为蛋白质嵌入提供上下文信息可以改善微生物功能的预测。我们观察到，一个基因的蛋白质嵌入可以从基因组中相邻基因的嵌入中部分预测出来，这促使我们采用高比例掩码策略，使模型必须从更稀疏的基因组上下文中进行重建。此外，基因组建模还必须区分真实的生物学关系与组装约定，因为独立重叠群（contig）之间的排序并不意味着物理上相邻，而链方向标签依赖于参考方向。为此，我们提出了VenusGenome，一个基因分辨率的掩码自编码器，它将仅可见编码与基因组感知注意力机制相结合，后者纳入了重叠群边界、相对链方向以及基因位置信息。我们在一个精选的语料库上预训练该模型，该语料库包含来自分离株和宏基因组来源的超过170万个微生物基因组组装，并使用冻结的蛋白质嵌入作为基因级输入。

    arXiv:2611.05405v1 Announce Type: cross  Abstract: Genomic-context models have shown that contextualizing protein embeddings can improve predictions of microbial function. We observe that a gene's protein embedding is partially predictable from the embeddings of neighboring genes in the genome, motivating high-ratio masking that requires reconstruction from sparser genomic context. Genomic modeling must also distinguish biological relationships from assembly conventions, since the ordering of separate contigs does not imply adjacency and strand labels depend on the reference orientation. We therefore introduce VenusGenome, a gene-resolution masked autoencoder that combines visible-only encoding with genome-aware attention incorporating contig boundaries, relative strand orientations, and gene positions. We pretrain this model on a curated corpus of over 1.7 million microbial genome assemblies from isolate and metagenomic sources, using frozen protein embeddings as gene-level inputs. Ac
    
[^211]: 面向医疗多智能体推理的挑战感知知识招募与元智能体监督方法

    Challenge-Aware Knowledge Recruitment with Meta-Agent Supervision for Medical Multi-Agent Reasoning

    [https://arxiv.org/abs/2611.05393](https://arxiv.org/abs/2611.05393)

    该论文提出挑战感知的自适应招募框架，由外部挑战者智能体监控医疗多智能体推理过程中涌现的知识与专业缺口，动态触发针对性知识检索或讨论中途的专家招募，在MEDAGENTSBENCH困难子集上显著优于仅依赖问题的事前检索策略并提升多数据集准确率。

    

    大语言模型在标准医疗问答任务上表现强劲，但在更困难的推理任务上性能会下降，而多智能体辩论在此类任务上往往无法超越强大的单模型提示方法。我们认为一个关键局限在于：现有系统仅根据问题本身来选择专家和检索知识，而不是根据推理过程中涌现出的知识缺口。我们提出挑战感知的自适应招募机制：一个外部的“挑战者”检查不断演化的推理轨迹，并识别出具体的知识缺口或专业知识缺口。这些信号会触发针对性的知识检索，或在讨论中途招募领域专家。在MEDAGENTSBENCH困难子集（涵盖九个生物医学数据集的862个问题）上，事前检索几乎不带来收益，而挑战感知的推理中间检索表现更好；在多专家骨干系统上，该方法在9个数据集中的8个上提升了准确率，相比仅使用挑战者的方案提升了6个百分点。消融实验表明两者均有收益……

    arXiv:2611.05393v1 Announce Type: new  Abstract: Large language models perform strongly on standard medical QA but degrade on harder reasoning tasks, where multi-agent debate often fails to beat strong single-model prompting. We argue that a key limitation is that existing systems select experts and retrieve knowledge from the question alone, rather than from gaps that emerge during reasoning. We propose challenge-aware adaptive recruitment: an external CHALLENGER inspects the evolving reasoning trace and identifies specific factual or expertise gaps. These signals trigger targeted knowledge retrieval or mid-discussion specialist recruitment. On the MEDAGENTSBENCH hard subset (862 questions across nine biomedical datasets), upfront retrieval provides little benefit, while challenge-aware in-between retrieval performs better; on a multi-specialist backbone it improves accuracy on 8 of 9 datasets and adds 6 percentage points over Challenger Only. Ablations indicate benefits from both ret
    
[^212]: 面向智能体AI系统的不确定性感知资源编排

    Uncertainty-Aware Resource Orchestration for Agentic AI Systems

    [https://arxiv.org/abs/2611.05392](https://arxiv.org/abs/2611.05392)

    本文提出了一种不确定性感知的资源编排方法，解决大语言模型驱动的智能体AI系统因运行时资源需求呈重尾、多峰且不可预测而导致的调度难题，避免资源供给不足或过度供给。

    

    由大语言模型（LLM）驱动的智能体人工智能（AI）系统以自主的多步骤流水线形式执行，动态调用工具、将观测结果整合进记忆，并不断迭代，直到任务完成或资源预算耗尽。与具有固定图结构的经典工作流不同，此类流水线的总体计算、带宽和端到端延迟需求在调度时是未知的：它们在运行时由中间观测结果和随机token采样涌现而来，形成重尾、多峰的随机变量。这种不可预测性使基础设施编排器陷入两难：资源供给不足可能导致执行中途限流和违反截止期限，而过度供给则会浪费稀缺的边缘或云端容量。现有调度器需要完全已知的流水线图，而边缘资源管理方法依赖参数化或平稳的需求模型，两者都不适用于未知的、非平稳的需求特性……

    arXiv:2611.05392v1 Announce Type: new  Abstract: Agentic artificial intelligence (AI) systems powered by large language models (LLMs) execute as autonomous multi-step pipelines that dynamically invoke tools, integrate observations into memory, and iterate until a task is resolved or a resource budget is exhausted. Unlike classical workflows with fixed graphs, the aggregate compute, bandwidth, and end-to-end latency demands of such pipelines are unknown at dispatch time: they emerge at runtime from intermediate observations and stochastic token sampling, yielding heavy-tailed, multi-modal random variables. This unpredictability forces infrastructure orchestrators into a dilemma: under-provisioning risks mid-execution throttling and deadline violations, while over-provisioning wastes scarce edge or cloud capacity. Existing schedulers require a fully known pipeline graph, while edge resource-management methods rely on parametric or stationary demand models, and neither fits the unknown, n
    
[^213]: 解读“为什么”：在光与量子比特上实现具有精确词证据的玻恩测量注意力机制

    Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits

    [https://arxiv.org/abs/2611.05355](https://arxiv.org/abs/2611.05355)

    该论文提出一种由玻恩测量构成的变换器（玻恩归一化注意力、测量乘积与玻恩读出），使类别概率成为词振幅的多项式，从而可通过有限求积无误差地精确计算每个词的积分梯度证据，并在 e-SNLI 上取得与同规模 softmax 变换器相当的准确率。

    

    当语言模型给出答案时，我们通常只能估计其“为什么”。我们构建了一个变换器，其中每一步都是一次玻恩测量——注意力为在每一行上归一化的 $|\langle q|k\rangle|^4$，用测量乘积取代前馈非线性，并采用玻恩读出。将其分母保持在输入时的取值，每个类别概率都是词振幅的多项式，因此其积分梯度证据可以通过有限求积精确计算，不存在求积误差：各词的贡献之和恰好等于类别概率超出无信息份额的部分，误差达到 $10^{-11}$ 或更好。在 e-SNLI 自然语言推理任务上，该模型在嵌入层之外约 98k 参数的规模下达到 76.98 ± 0.17% 的测试准确率，与同规模的 softmax 变换器持平（76.62 ± 0.36%）。删除证据排名第一的词会在 45% 的测试样本对中改变答案，而这接近任何单个词删除所能改变答案的比例上限（53%）。

    arXiv:2611.05355v2 Announce Type: replace-cross  Abstract: When a language model answers, we can usually only estimate why. We build a transformer in which every step is a Born measurement - attention $|\langle q|k\rangle|^4$ normalised over each row, a measured product in place of the feed-forward nonlinearity, and a Born readout. With its denominators held at their values for the input, each class probability is a polynomial in the word amplitudes, so its integrated-gradient evidence is computed by a finite quadrature with no quadrature error: the words' contributions sum to the class probability above its no-information share, to $10^{-11}$ or better. On e-SNLI natural language inference, with about 98k parameters outside the embedding, the model reaches $76.98 \pm 0.17$% test accuracy, level with a softmax transformer of the same size ($76.62 \pm 0.36$%). Deleting the word its evidence ranks first changes the answer in 45% of test pairs, out of the 53% any single deletion can chang
    
[^214]: 智能体强化学习中在策略定向引导的稳定化

    Stabilizing On-Policy Targeted Guidance in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2611.05266](https://arxiv.org/abs/2611.05266)

    提出了“定向引导”框架，通过在策略自蒸馏从每轮文本反馈中学习，并与按评分标准的强化学习奖励相结合，在不降低基线强化学习性能的前提下稳定地改进智能体环境中难以优化的行为。

    

    朴素的在策略强化学习往往受限于基座模型的能力——覆盖不足可能导致模型无法从某些奖励中学习。与此同时，现代后训练系统中的奖励通常来自丰富的文本来源，例如带有评分标准的评判模型。我们能否利用这类结构来提供丰富的方向性信号，从而改进那些原本难以优化的奖励？在这项工作中，我们研究了一种称为“定向引导”的框架，它通过在策略自蒸馏目标从每轮的文本反馈中学习。我们的重点不仅是直接改进难以优化的行为，还包括如何在整个训练过程中实现这一目标而不降低基线强化学习的性能。在智能体多轮软件工程环境的多种设置中，我们通过将定向自蒸馏与按评分标准的强化学习奖励相结合取得了成功，并且……（摘要在此处截断）

    arXiv:2611.05266v2 Announce Type: replace  Abstract: Naive on-policy RL is often bottlenecked by the capabilities of the base model --- insufficient coverage can result in an inability to learn from certain rewards. At the same time, rewards in modern post-training systems often come from rich textual sources, e.g. judge models with a rubric of criteria. Can we utilize such structures to provide a rich directional signal that enables improving on otherwise difficult to optimize rewards? In this work, we examine a framework we refer to as targeted guidance that learns from per-turn textual feedback via an on-policy self-distillation objective. Our focus is not only on directly improving hard-to-optimize behaviors, but on how to do so throughout the training process without degrading baseline RL performance. Over a range of settings across an agentic multi-turn software engineering environment, we find success by combining targeted self-distillation with a per-criteria RL reward, and by 
    
[^215]: 具有多个正确答案的PAC学习

    PAC Learning with Multiple Correct Answers

    [https://arxiv.org/abs/2611.05133](https://arxiv.org/abs/2611.05133)

    该论文研究了每个样本允许多个正确标签的PAC学习问题，证明了示范反馈可能比完整标签集合甚至bandit反馈学习更困难，并用广义Natarajan维度刻画了前两种反馈下的可学习性，用假设交集的可学习性刻画了示范学习下的可学习性。

    

    我们研究了每个样本带有至多 $\ell$ 个正确标签集合的PAC学习问题，只要预测的标签属于该集合即视为正确。训练过程中可能揭示完整标签集合、仅反馈所提出的标签是否正确（bandit反馈），或者由示范者给出的一个正确标签（示范）。我们的主要发现是：示范学习可能比完整标签集合难得多，甚至比bandit反馈更难，这种困难体现在可学习的类别以及所需样本数量两个方面。对于有限标签空间，一个广义的Natarajan维度刻画了在可实现和不可知（agnostic）两种设定下，从完整标签集合和从bandit反馈学习的可学习性。然而，对于示范学习，一个类是可实现的（realizably）可学习的当且仅当其至多 $\ell+1$ 个假设的交集可以从完整标签集合学习。仅使用三个标签时，该条件对于一个可以从单个完整标签集合样本（甚至单个bandit反馈样本）学习的类就可能不成立。

    arXiv:2611.05133v2 Announce Type: replace  Abstract: We study PAC learning when each example comes with a set of at most $\ell$ correct labels and a prediction is correct if it belongs to this set. Training may reveal the full set, only whether a proposed label is correct (bandit feedback), or one correct label chosen by a demonstrator (demonstrations). Our main finding is that demonstrations can be much harder than full sets, and even than bandit feedback, both in which classes can be learned and in how many examples are needed. For finite label spaces, a generalized Natarajan dimension characterizes learnability from full sets and from bandit feedback, in both the realizable and the agnostic setting. From demonstrations, however, a class is realizably learnable if and only if the intersections of at most $\ell+1$ of its hypotheses are learnable from full sets. Already with three labels, this condition can fail for a class learnable from a single full-set example or even a single band
    
[^216]: FedCSGA：联邦学习中联合统计与系统异构性的进化客户端选择方法

    FedCSGA: Evolutionary Client Selection with Joint Statistical and System Heterogeneity in Federated Learning

    [https://arxiv.org/abs/2611.04667](https://arxiv.org/abs/2611.04667)

    提出了一种基于遗传算法的联邦学习客户端选择方法FedCSGA，通过联合建模统计异构性与系统异构性的非线性适应度函数，实现每轮客户端子集的进化式优化选择。

    

    联邦学习（Federated Learning, FL）能够在保护数据隐私的前提下，跨分布式客户端协同训练机器学习模型。在实际部署中，联邦学习需要在每轮通信中选择一个客户端子集，以降低通信开销、缓解掉队者问题，并在数据与系统异构条件下提升训练效率。在异构边缘计算环境中，统计异构性和系统异构性都会显著影响收敛效率和模型性能。然而，现有的客户端选择方法往往独立处理这两类异构性因素，忽略了它们在通信轮次之间的耦合与动态影响。为解决这一局限，我们提出了FedCSGA，一种基于遗传算法的联邦学习客户端选择方法。FedCSGA将客户端子集编码为染色体，并通过一个联合整合统计与系统异构性的非线性适应度函数来评估这些子集。

    arXiv:2611.04667v2 Announce Type: replace  Abstract: Federated Learning (FL) enables collaborative training of machine learning models across distributed clients while preserving data privacy. In practical deployments, FL relies on selecting a client subset in each communication round to reduce communication cost, mitigate stragglers, and improve training efficiency under heterogeneous data and system conditions. In heterogeneous edge computing environments, both statistical heterogeneity and system heterogeneity significantly affect convergence efficiency and model performance. However, existing client selection methods often address these factors independently, overlooking their coupled and dynamic impact across communication rounds. To address this limitation, we propose FedCSGA, a Client Selection method for Federated learning based on Genetic Algorithm. FedCSGA encodes client subsets as chromosomes and evaluates these subsets through a non-linear fitness function that jointly inte
    
[^217]: 样本粒度近似遗忘与可证明的逐样本边界

    Sample-Grained Approximate Unlearning with Provable Per-Sample Bounds

    [https://arxiv.org/abs/2611.04501](https://arxiv.org/abs/2611.04501)

    该论文提出SALA即插即用优化框架和样本粒度分布差距指标，使近似机器遗忘方法能够在逐样本层面工作，并提供了可证明的逐样本遗忘边界。

    

    机器遗忘旨在从训练好的模型中消除指定遗忘集的影响，同时保持模型在保留集上的强效用。近似遗忘方法无需完整重训练即可高效实现这一目标，但其有效性主要通过数据集层面的聚合指标来评估。这些聚合指标可能掩盖个别样本层面上与重训练结果的偏差。在本工作中，我们关注逐样本的遗忘有效性。我们提出了SALA（Sample-grained Likelihoods ALignment，样本级似然对齐），这是一个即插即用的优化框架，通过将每个样本引导至其正确的目标分布，使近似遗忘方法能够在单个样本层面运行。为了定量评估这一行为，我们进一步提出了样本粒度分布差距这一指标，用于捕捉近似遗忘与精确重训练输出分布之间的逐样本差异。我们证明（摘要原文在此处被截断）……

    arXiv:2611.04501v2 Announce Type: replace  Abstract: Machine unlearning aims to erase the influence of a designated forget set from a trained model while preserving strong utility on the retain set. Approximate unlearning methods efficiently achieve this goal without full retraining, but their effectiveness is primarily assessed using dataset-level aggregate metrics. These aggregate metrics can mask deviations from retraining at the level of individual samples. In this work, we focus on per-sample unlearning effectiveness. We introduce SALA (Sample-grained Likelihoods ALignment), a plug-in optimization framework that enables approximate unlearning methods to operate at the level of individual samples by steering each example toward its correct target distribution. To quantitatively assess this behavior, we further propose the sample-grained distribution gap, a metric that captures per-sample discrepancies between approximate unlearning and exact retraining output distributions. We prov
    
[^218]: 面向混合专家大语言模型端到端FP4强化学习的训练与推理策略对齐

    Aligning Training and Rollout Policies for End-to-End FP4 RL on Mixture-of-Experts LLMs

    [https://arxiv.org/abs/2611.04376](https://arxiv.org/abs/2611.04376)

    该研究揭示了MoE大模型FP4强化学习失效的根源在于路由器的离散专家选择机制，证明了必须在训练与推理两侧对称地应用相同的恢复技术，才能实现有效的端到端FP4强化学习。

    

    本研究探讨了面向混合专家大语言模型的端到端FP4（四比特浮点）强化学习后训练。近期关于FP4强化学习的工作提出了有效的恢复方案，能够在激进低精度量化下恢复稠密模型的精度，但我们发现同样的方法在MoE场景下相比朴素的FP4基线没有任何提升。我们开展了深入研究以理解其中的原因。我们的分析发现其根本原因在于MoE模型独有的路由机制：由于路由器做出的是离散的专家选择决策，训练与推理策略的对齐对训练端和推理端是否对称地应用相同的恢复技术变得极为敏感。当使用非对称方案时——包括那种仅修正rollout/训练侧的稠密模型方法——训练与推理策略的对齐会急剧恶化（以皮尔逊相关系数衡量）。

    arXiv:2611.04376v2 Announce Type: replace-cross  Abstract: In this work, we study end-to-end FP4 reinforcement learning (RL) post-training for Mixture-of-Experts (MoE) large language models. Recent work on FP4 RL has introduced effective recovery recipes that restore accuracy under aggressive low-precision quantization for dense models, but we find that the same approach yields no improvement over the naive FP4 baseline in the MoE setting. We conduct an in-depth study to understand why. Our analysis identifies the root cause in the routing mechanism unique to MoE: because the router makes discrete expert-selection decisions, training-inference policy alignment becomes acutely sensitive to whether the same recovery technique is applied symmetrically on the training and inference sides. When an asymmetric recipe is used-including the dense-model approach, which corrects only the rollout-training-inference alignment deteriorates sharply, as measured by the Pearson correlation coefficient 
    
[^219]: 面向模型合并的公共与独立方向分离

    Separating Common and Unique Directions for Model Merging

    [https://arxiv.org/abs/2611.04108](https://arxiv.org/abs/2611.04108)

    提出CUSD，一种无需数据的模型合并预处理方法，通过将任务向量分解为跨任务共享的公共方向分量和任务独有的独立分量并分别处理，从而提升算术合并方法的性能。

    

    模型合并无需额外训练即可组合多个微调模型，而算术方法已成为一种无需数据、简单而有效的模型合并方式。然而，现有的算术方法在参数空间中处理任务向量，并对可合并性可能不同的方向统一应用单一的逐元素规则。在共享的方向基中，我们发现了一致的双区间结构：一个跨任务共享的低离散度公共主体，以及一个高离散度、由任务主导且承载着不成比例能量的尾部。这些发现催生了公共-独立子空间分解（CUSD），这是一种无需数据的预合并方法，它将任务向量分解为公共与独立的方向分量，用下游算术算子合并公共分量，并单独保留独立残差。在11个基础模型、四个模型家族和四种基于算术的合并算子上的实验表明，CUSD提升了……

    arXiv:2611.04108v2 Announce Type: replace-cross  Abstract: Model merging combines multiple fine-tuned models without additional training, and arithmetic methods have emerged as a simple yet effective way to merge models without data. However, existing arithmetic methods treat task vectors in parameter space and apply a single element-wise rule to directions that can differ in mergeability. In a shared directional basis, we find a consistent two-regime structure: a low-dispersion common bulk shared across tasks and a high-dispersion task-dominated tail that carries disproportionate energy. These findings motivate Common--Unique Subspace Decomposition (CUSD), a data-free pre-merging method that separates task vectors into common and unique directional components, merges the common components with a downstream arithmetic operator, and retains unique residuals separately. Experiments across $11$ base models, four model families, and four arithmetic-based merging operators show that CUSD im
    
[^220]: 基于参数多重集的ReLU网络合成复杂性

    The Complexity of ReLU Network Synthesis from a Parameter Multiset

    [https://arxiv.org/abs/2611.04074](https://arxiv.org/abs/2611.04074)

    本文研究了在参数数值与网络架构分离（即参数值预先给定）的条件下合成ReLU网络的计算复杂性，证明精确匹配标注样例为NP完全、在所有布尔输入上匹配布尔电路为Σ₂^P完全（四个神经元即可）、以及在所有实数输入上匹配给定ReLU网络为DP完全（三个神经元时）。

    

    我们探讨当参数值预先给定时，构建一个ReLU网络有多困难。一个实例提供了目标行为的适当编码、一个有理数多重集，以及神经元数量（包括输出神经元）的上界。问题是：能否在该界限内选择ReLU网络的图结构，并将这些数放置到其边和偏置上，使之达到目标。这将数值与网络架构分离开来。精确匹配两个带标签的样例已经是NP完全的，因此我们进一步追问：当目标必须在每个输入上成立或平均意义上成立时，复杂性会发生什么变化。在所有布尔输入上与给定布尔电路一致是Σ₂^P完全的，且仅需四个神经元即可。在所有实数输入上与给定ReLU网络一致，对于每个固定的神经元界限都属于DP类，且在三个神经元时是DP完全的。在单位立方体上以绝对……（原文摘要在此处截断）

    arXiv:2611.04074v2 Announce Type: replace-cross  Abstract: We ask how hard it is to build a ReLU network when its parameter values are given in advance. An instance supplies an appropriate encoding of a target behavior, a multiset of rational numbers, and a bound on the number of neurons (output included). The question is whether one can choose the graph of a ReLU network within the bound, and place the numbers on its edges and biases so that it meets the target. This separates the numbers from the architecture. Matching two labeled examples exactly is already $\mathsf{NP}$-complete, so we ask what changes when the target must hold on every input or on average. Agreeing with a Boolean circuit on every Boolean input is $\Sigma_2^{\mathsf P}$-complete, already for four neurons. Agreeing with a given ReLU network on every real input lies in $\mathsf{DP}$ for every fixed neuron bound and is $\mathsf{DP}$-complete for three neurons. Approximating a ReLU network on the unit cube within absol
    
[^221]: 所有路径一次获取：一种结合量子蒙特卡洛与神经commitor的反应路径搜索方法

    All Paths All at Once: A Quantum Monte Carlo and Neural Committor Approach to Reaction Pathway Search

    [https://arxiv.org/abs/2611.03699](https://arxiv.org/abs/2611.03699)

    该论文提出一种无需势能梯度的仅能量反应路径搜索方法tReax_expedition，通过耦合蒙特卡洛探索、随机淬火标记与受限延续自动恢复中间体和竞争反应通道，无需过渡态猜测即可将链最大值能量精确到精修鞍点0.76 kJ/mol以内，并借助神经commitor学习将路径与过渡区域采样相连接。

    

    在势能梯度不可用时，寻找相互竞争的反应路径十分困难。本文提出tReax_expedition，一种仅依赖能量信息的方法，它将耦合蒙特卡洛探索、随机淬火标记与受限延续相结合。探索阶段识别相互连通的状态，而延续阶段解析候选反应通道并记录采样链的最大值。在三个二维势能面上的单次搜索基准测试中，该方法在无需过渡态猜测的情况下成功恢复出中间体和相互竞争的反应路线。在基于GFN2-xTB势能面的九个气相反应中，有八个反应的采样链最大值与独立精修后的鞍点能量之差小于0.76 kJ/mol。而丙氨酸二肽的算例则揭示了构象弛豫不完全的问题，表明尽管存在结构和能量上的误差，反应路径的连通性仍可保持。通过神经commitor学习和局域跨对采样，所得反应路径可与过渡区域样本相连（原文此处截断）。

    arXiv:2611.03699v2 Announce Type: replace-cross  Abstract: Locating competing reaction pathways is difficult when potential-energy gradients are unavailable. Here we present tReax_expedition, an energy-only approach combining coupled Monte Carlo exploration, stochastic quench labelling and restrained continuation. Exploration identifies connected states, while continuation resolves candidate channels and records sampled chain maxima. Single-search benchmarks on three two-dimensional potentials recovered an intermediate and competing routes without transition-state guesses. For eight of nine gas-phase reactions on GFN2-xTB, sampled chain maxima differed from independently refined saddle energies by less than 0.76 kJ/mol. Alanine dipeptide instead revealed incomplete conformational relaxation, showing that route connectivity can persist despite structural and energetic errors. Neural committor learning and local straddling-pair sampling connect the resulting paths to transition-region sa
    
[^222]: 预训练数据规模逆转Muon与AdamW的微调差距

    Pretraining Data Scale Reverses the Muon-AdamW Fine-Tuning Gap

    [https://arxiv.org/abs/2611.03633](https://arxiv.org/abs/2611.03633)

    本文发现预训练数据规模会逆转Muon与AdamW优化器的微调性能差距——当预训练数据足够大时，微调后AdamW反而优于Muon，并提出了“边际-散度比”指标来解释这一逆转现象。

    

    Muon近来已成为一种有前景的大语言模型预训练优化器，在最小化预训练损失方面比AdamW提供了加速。然而，Muon的优势在微调之后能否持续尚不清楚。本文研究了Muon和AdamW的下游行为，并揭示了一个令人惊讶的趋势：当预训练数据预算超过足够大的规模后，AdamW在微调后的表现优于Muon。此外，随着模型规模的增大，AdamW在越来越小的预训练token-参数比率下就会超越Muon。为了解释这种逆转，我们将微调视为对模型输出的一种扰动，并引入了“边际-散度比”这一指标，用于衡量正确logit高于错误logit的程度相对于错误logit散布范围的比值。在语言模型中，Muon在小数据预算下具有更大的边际-散度比，而AdamW在大数据预算下具有更大的边际-散度比，这与微调性能的趋势相吻合。

    arXiv:2611.03633v2 Announce Type: replace  Abstract: Muon has recently emerged as a promising optimizer for pretraining large language models, offering a speedup over AdamW in minimizing pretraining loss. However, it remains unclear whether Muon's advantage persists after fine-tuning. In this paper, we study the downstream behavior of Muon and AdamW and uncover a surprising trend: beyond a sufficiently large pretraining data budget, AdamW outperforms Muon after fine-tuning. Moreover, as model size increases, AdamW overtakes Muon at progressively smaller pretraining token-parameter ratios. To explain this reversal, we view fine-tuning as a perturbation of the outputs and introduce the margin-to-spread ratio, which measures how far the correct logit lies above the wrong ones relative to how widely the wrong logits are spread. In language models, Muon has the larger margin-to-spread ratio at small data budgets and AdamW at large ones, mirroring the fine-tuning trend. In a linear associati
    
[^223]: 通过分层脑电图-文本对齐实现思维破译

    Thought-deciphering via hierarchical EEG-to-text alignment

    [https://arxiv.org/abs/2611.03623](https://arxiv.org/abs/2611.03623)

    提出一种分层脑电图-文本对齐框架，先解码粗粒度语义类别、再在类别内检索细粒度句子，显著提升了从EEG进行句子级解码的检索准确率。

    

    非侵入式脑到文本接口为有沟通障碍的人群提供了替代性沟通途径。然而，从脑电图（EEG）进行句子级别的解码仍然具有挑战性。我们提出了一种分层脑电图-文本对齐框架，通过逐步解码粗粒度语义信息和细粒度句子表示来检索句子。我们假设，相比单个句子，更广泛的语义背景能够从脑电图中被更可靠地捕获。因此，脑电图表示首先与语义类别进行对齐，随后在预测出的类别内进行句子检索。在超过2,000个独特文本样本中，我们的方法实现了平均top-5和top-10检索准确率分别为62.6% ± 1.0%和83.5% ± 1.0%，而直接进行脑电图-句子对齐的准确率仅为50.3% ± 2.3%和68.3% ± 2.0%。定性分析进一步揭示了……（摘要内容截断）

    arXiv:2611.03623v2 Announce Type: replace  Abstract: Non-invasive brain-to-text interfaces offer alternative communication pathways for people with communication disabilities. However, sentence-level decoding from electroencephalography (EEG) remains challenging. We propose a hierarchical EEG-text alignment framework that retrieves sentences by progressively decoding coarse-grained semantic information and fine-grained sentence representations. We hypothesize that broader semantic context can be more reliably captured from EEG than individual sentences. Accordingly, EEG representations are first aligned with semantic classes, followed by sentence retrieval within the predicted class. Among over 2,000 unique text samples, our approach achieves average top-5 and top-10 retrieval accuracies of 62.6\% $\pm$ 1.0\% and 83.5\% $\pm$ 1.0\%, respectively, compared with 50.3\% $\pm$ 2.3\% and 68.3\% $\pm$ 2.0\% for direct EEG-sentence alignment. Qualitative analysis further reveals that decoding
    
[^224]: 基于极小极大评论家与反射朗之万采样的神经演员-评论家学习

    Neural Actor--Critic Learning with Minimax Critics and Reflected Langevin Sampling

    [https://arxiv.org/abs/2611.03352](https://arxiv.org/abs/2611.03352)

    该论文提出一种结合正则化极小极大贝尔曼评论家估计与反射朗之万采样的神经演员-评论家算法，并给出了达到非参数极小极大最优速率的有限样本价值误差界与策略回报保证。

    

    演员-评论家方法依赖评论家的梯度来改进连续动作策略，但有限样本保证必须同时控制价值误差与梯度误差及其在演员端的传播。我们提出了一种神经演员-评论家算法，该方法将基于S形网络的正则化极小极大贝尔曼估计器与由拟合评论家梯度驱动的有限反射朗之万采样器相结合。在一致贝尔曼正则性与覆盖重置采样条件下，正的高阶动作导数惩罚可带来具有经典非参数极小极大指数（至多相差对数因子）的价值与动作梯度风险界。我们将估计误差、混合误差、离散化误差、温度误差与策略迭代误差进行传播，从而获得回报保证，并针对有界多峰势函数与强凹情形给出了精细化结果。在受控仿真、记录的连续控制数据以及模拟决策任务上的实证研究表明……

    arXiv:2611.03352v2 Announce Type: replace-cross  Abstract: Actor--critic methods rely on critic gradients to improve continuous-action policies, yet finite-sample guarantees must control both value and gradient errors and their propagation through the actor. We develop a neural actor-critic procedure that combines a regularized minimax Bellman estimator over sigmoid networks with a finite reflected Langevin sampler driven by the fitted critic gradient. Under uniform Bellman regularity and covered reset sampling, a positive high-order action-derivative penalty yields value and action-gradient risk bounds with classical nonparametric minimax exponents up to logarithmic factors. We propagate estimation, mixing, discretization, temperature, and policy-iteration errors to obtain return guarantees, with refinements for bounded multimodal potentials and strongly concave regimes. Empirical studies across controlled simulations, logged continuous-control data, and a simulated decision task show
    
[^225]: 受限扩散桥：有针对性的去噪器训练与精化

    Restrained Diffusion Bridges: Targeted Denoiser Training and Refinement

    [https://arxiv.org/abs/2611.02170](https://arxiv.org/abs/2611.02170)

    受限扩散桥方法通过在选定的中间时刻以受限构型为条件进行前向加噪，使受限采样引入的偏差被精确抵消，从而让模型在目标分布中任意关键的数据稀疏区域都能高精度生成，无需重要性权重。

    

    扩散模型在目标分布中关键但数据稀疏的区域往往不够准确。从受限分布采样——这是经典增强采样中使用的一类有偏分布，用于将数据集中在所选构型附近——可以控制训练数据的采集位置，但会使学到的分布产生偏差。我们证明，当从受限数据出发的前向加噪过程以适当选择的中间时刻的受限构型为条件时，这种偏差会被精确抵消。这使得在受限数据上微调的预训练模型，能够在目标分布的任意选定区域（无论该区域的概率大小）实现高精度，且只需使用标准的去噪或流匹配目标函数，无需重要性权重。当目标数据不可用时，吉布斯采样使得仅用受限数据训练的模型也能进行推断。

    arXiv:2611.02170v2 Announce Type: replace  Abstract: Diffusion models are often inaccurate in critical, data-sparse regions of the target distribution. Sampling from restrained distributions, biased distributions used in classical enhanced sampling to concentrate data near chosen configurations, can control where training data are collected but biases the learned distribution. We establish that this bias cancels exactly when forward noising from restrained data is conditioned on the restrained configuration at an appropriately chosen intermediate time. This allows a pretrained model fine-tuned on restrained data to achieve high accuracy in any chosen region of the target distribution, regardless of its probability, using standard denoising or flow matching objectives without importance weights. When no target data are available, Gibbs sampling enables inference from a model trained exclusively on restrained data.
    
[^226]: 使用扩散反事实样本和匹配阳性对照测试黑色素瘤分类器的概念级审计

    Testing concept-level audits of melanoma classifiers with diffusion counterfactuals and matched positive controls

    [https://arxiv.org/abs/2611.02119](https://arxiv.org/abs/2611.02119)

    本研究构建了基于扩散模型的条件反事实审计框架，并通过匹配的阳性对照严格验证了其对黑色素瘤分类器概念级解释的审计能力，证明该方法在恢复目标概念方面优于传统的概念激活向量方法。

    

    基于概念的黑色素瘤分类器解释很少针对具有已知目标的对照进行测试。我们基于以七点检查表中七个概念为条件的扩散模型构建了一个反事实审计，并使用此类对照对其进行了测试。经过色彩校准的起始噪声将颜色异常值从78.5-84.8%降低到6.6-13.3%。七个控制旋钮中有四个是目标主导的，但也会改变其他概念。基于四分位数的阳性对照存在混杂因素，该审计未能通过预先设定的规则。对于三个概念，它成功恢复了所有16个按生成器伪标签匹配的对照的目标，以及由第二位评估者标注的12个对照中全部12个的目标；而概念激活向量恢复的数量较少（16个中恢复了11个）。在18个置换标签分类器中，也出现了显著的扫描响应和目标排名，而标准化的效应边际成功地将对照与这些零分类器区分开来。色素沉着的扫描提高了黑色素瘤的……（摘要在此处被截断）

    arXiv:2611.02119v2 Announce Type: replace-cross  Abstract: Concept-based explanations of melanoma classifiers are rarely tested against controls with known targets. We built a counterfactual audit from a diffusion model conditioned on seven 7-point checklist concepts and tested it with such controls. Colour-calibrated start noise reduced colour outliers from 78.5-84.8\% to 6.6-13.3\%. Four of seven knobs were target-dominant but also changed other concepts. Quartile-based positive controls were confounded, and the audit failed its pre-specified rule. It recovered the targets of all 16 controls matched on the generator's pseudo-labels and of 12 of 12 controls labelled by a second evaluator, for three concepts; concept activation vectors recovered fewer (11 of 16). In 18 permuted-label classifiers, significant sweep responses and target rankings also arose, whereas standardised effect margins separated the controls from these null classifiers. The pigmentation sweep raised the melanoma l
    
[^227]: NorthCUA 技术报告：通过大规模经验加速计算机使用智能体

    NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale

    [https://arxiv.org/abs/2611.02102](https://arxiv.org/abs/2611.02102)

    NorthCUA 构建了经验飞轮驱动的集成系统，通过大规模验证轨迹积累、递归自我改进和分阶段学习（中期训练、长时程监督微调与桌面-网页联合强化学习），显著提升计算机使用智能体完成复杂软件任务的能力。

    

    将计算机使用智能体从单个操作扩展到复杂软件工作，需要能够将决策与完整结果关联起来的经验。我们提出 NorthCUA，一个通过大规模经验来加速计算机使用智能体的集成系统。其经验飞轮为 NorthCUA Harness 提供多样化的可执行任务，该 Harness 支持长时间交互、记录决策并检查任务结果。经验飞轮利用这些记录来筛选经过验证的轨迹。对于未解决的任务，递归自我改进会演化出可复用的技能，并保留已验证的里程碑，使后续尝试建立在之前的进展之上。分阶段学习将这种经验与来自经验飞轮的定位数据相结合，通过中期训练和长时程监督微调来培养 GUI 能力。随后，策略通过桌面与网页联合强化学习，在同一 Harness 中从新的尝试中学习，从而闭合经验与提升之间的循环。

    arXiv:2611.02102v3 Announce Type: replace  Abstract: Scaling computer-use agents from individual actions to complex software work requires experience that connects decisions to complete outcomes. We present NorthCUA, an integrated system for turbocharging computer-use agents through experience at scale. Its Experience Flywheel supplies diverse executable tasks to the NorthCUA Harness, which sustains long interactions, records decisions, and checks task outcomes. The Flywheel uses these records to curate verified trajectories. For unsolved tasks, recursive self-improvement evolves reusable skills and preserves verified milestones so later attempts build on earlier progress. Staged Learning combines this experience with grounding data from the Flywheel to develop GUI competence through mid-training and long-horizon supervised fine-tuning. The policy then learns from fresh attempts in the same Harness through joint desktop-web reinforcement learning, closing the loop between experience pr
    
[^228]: 波动物理作为表示：用于海洋声场的张量结构神经重建

    Wave Physics as Representation: Tensor-Structured Neural Reconstruction for Ocean Acoustic Fields

    [https://arxiv.org/abs/2611.01680](https://arxiv.org/abs/2611.01680)

    该论文提出从波动物理中提取结构化表示并用张量结构神经网络学习其组件的新路线，实现了既具物理可解释性、又对不确定环境信息鲁棒的海洋三维声场稀疏重建。

    

    从稀疏测量中重建三维（3D）复杂海洋声场极具挑战性，原因在于高度振荡的波结构以及受限的感知模式与路径。基于学习的方法可分为两类：第一类是纯数据驱动模型，虽然能够学习复杂的场模式，但并未显式利用波动物理，而波动物理本可增强可解释性并降低训练成本；第二类是物理信息方法，其将特定的物理方程作为重建约束，但往往依赖难以获取的环境相关参数。因此，如何在物理上有依据、又能对不确定环境信息保持鲁棒性的实用3D声场重建，在很大程度上仍是一个未被充分探索的问题。本工作采取了另一条路线：不再求解完全指定的波动方程，而是从波动物理中提取一种结构化表示，并学习其各个组成部分……

    arXiv:2611.01680v2 Announce Type: replace-cross  Abstract: Reconstructing three-dimensional (3D) complex ocean acoustic fields from sparse measurements is challenging due to highly oscillatory wave structures and restricted sensing patterns/routes. Learning-based approaches fall into two categories. First, purely data-driven models can learn complex field patterns, but do not explicitly exploit wave physics that may enhance interpretability and reduce training costs. Second, physics-informed methods impose a specified physical equation as a reconstruction constraint, but often require environment-dependent parameters that are difficult to acquire. As a result, practical 3D acoustic field reconstruction that is both physically grounded and robust to uncertain environmental information remains largely unexplored. This work takes an alternative route: instead of solving a fully specified wave equation, it extracts a structured representation from wave physics and learns its components fro
    
[^229]: 行为图：用于解释机器人部署数据的无监督图结构

    Behavior Graphs: Unsupervised Graph Structures for Interpreting Robot Deployment Data

    [https://arxiv.org/abs/2611.00405](https://arxiv.org/abs/2611.00405)

    该论文提出“行为图”这一无监督方法，将机器人策略rollout数据压缩为一张单一、可解释的有向图，通过识别相似行为模式、按时序连接并与结果关联，显著降低了部署数据的分析难度，使研究者能更高效地指导数据收集决策。

    

    数据飞轮承诺通过从部署数据中获得的洞察实现持续改进。理解部署后的模型如何成功与失败，可以引导后续的数据收集，从而修补失败并强化成功。然而，机器人部署会产生大量高维的rollout数据，这些数据难以理解，也难以据此做出有把握的数据收集决策。为了让部署数据更易于分析和采取行动，我们提出了行为图，它将策略rollout压缩成一张单一的、可解释的有向图。我们通过学习识别不同rollout之间相似的行为模式、在时间维度上连接这些行为，并最终将它们与结果关联起来来构建该图。在一项针对三个真实世界任务的用户研究中，参与者通过分析rollout来选择数据收集策略，18名参与者中有16名更偏好行为图而非rollout视频，且在感知负荷降低40.3%的情况下得出了相当的结论。

    arXiv:2611.00405v1 Announce Type: cross  Abstract: Data flywheels promise continuous improvement via insights from deployment data. Understanding how the deployed model succeeds and fails can steer subsequent data collection towards patching failures and reinforcing successes. However, robot deployments create large volumes of high-dimensional rollout data, which is challenging to understand and confidently ground data collection decisions in. To make deployment data easier to analyze and act on, we propose behavior graphs, which compress policy rollouts into a single interpretable, directed graph. We construct the graphs by learning to identify similar behavior modes across rollouts, connecting the behaviors temporally, and finally linking them to outcomes. In a user study on three real-world tasks, where participants analyzed rollouts to choose a data collection strategy, 16 of 18 preferred behavior graphs to rollout videos, reaching comparable conclusions with 40.3% lower perceived 
    
[^230]: TensorSSM：在动态状态空间中记忆空间位置以生成张量核

    TensorSSM: Memorizing Spatial Position in Dynamic State Space for Tensor Core Generation

    [https://arxiv.org/abs/2610.40871](https://arxiv.org/abs/2610.40871)

    提出TensorSSM，一种在动态状态空间模型（SSM）中记忆空间位置的新型隐式神经表示函数式张量分解方法，通过K叉规则从粗到细编码空间标记，克服了现有方法因展平高阶张量为一维序列而导致的表达能力受限问题。

    

    高维张量在现实世界的应用中无处不在，但会带来巨大的存储开销。函数式张量分解通过利用神经网络将元素坐标映射到张量核来应对这一问题，显著减少了数据压缩所需的参数量。然而，现有的函数式方法通常将高阶张量展平为一维序列。这种折叠操作会导致高度重叠的局部坐标序列，且难以捕捉复杂的跨模态空间依赖关系，严重限制了此类方法在多样化数据集上的表达能力。为了解决这些挑战，我们提出了TensorSSM，这是一种用于函数式分解的新型隐式神经表示（INR）方法，它在动态状态空间模型（SSM）中记忆空间位置。具体而言，TensorSSM首先将各模态的坐标分解为按K叉规则切分的序列，从粗到细地编码空间标记，有效捕捉……

    arXiv:2610.40871v1 Announce Type: new  Abstract: High-dimensional tensors are ubiquitous in real-world applications but lead to massive storage overhead. Functional tensor decomposition addresses this by employing neural networks to map element coordinates to tensor cores, significantly reducing the parameters for data compression. However, existing functional methods typically flatten high-order tensors into 1D sequences. This folding operation causes highly overlapping local coordinate sequences and struggles to capture complex cross-mode spatial dependencies, severely limiting their expressiveness on diverse datasets. To tackle these challenges, we propose TensorSSM, a novel Implicit Neural Representation (INR) for functional decomposition method that memorizes spatial positions within a dynamic State Space Model (SSM). Specifically, TensorSSM first decomposes mode-wise coordinates into sequences spit by K-ary rule, encoding spatial tokens from coarse to fine, which effectively capt
    
[^231]: 流形上的概率无网格方法

    Probabilistic Meshless Methods on Manifolds

    [https://arxiv.org/abs/2610.39855](https://arxiv.org/abs/2610.39855)

    该论文将概率无网格方法首次扩展到紧流形并给出其收敛性分析，在拉普拉斯-贝尔特拉米特征对已知的情形下，采用点云上的广义有限差分离散化PDE算子并证明方法在采样点处的一致收敛性。

    

    概率无网格方法通过在解上放置高斯过程先验，并在有限个配点节点处以偏微分方程（PDE）为条件对其进行调节，从而求解偏微分方程。后验分布既提供了解的近似，也提供了对其误差的贝叶斯量化。虽然这些方法在欧氏区域上已经发展成熟，但据我们所知，其在流形上的扩展尚缺乏任何收敛性分析。我们将这些方法扩展到紧流形上，并在两种设定下研究其收敛性。首先，对于拉普拉斯-贝尔特拉米特征对已知的流形，先验可以被精确计算，但PDE算子通常无法以闭式形式作用于其上。我们用在从流形上采样的点云上进行标准广义有限差分离散化来替代该算子，并证明所得方法在点云的各点上是一致的，其收敛速率由点云的……（原文摘要在此处截断）

    arXiv:2610.39855v2 Announce Type: replace-cross  Abstract: Probabilistic meshless methods solve partial differential equations (PDEs) by placing a Gaussian process prior on the solution and conditioning it on the PDE at a finite set of collocation nodes. The posterior provides both an approximation of the solution and a Bayesian quantification of its error. While these methods are well established on Euclidean domains, their extension to manifolds lacks, to our knowledge, any convergence analysis. We extend them to compact manifolds and study their convergence in two settings. First, for known manifolds whose Laplace-Beltrami eigenpairs are available, the prior can be evaluated exactly, but the PDE operator generally cannot be applied to it in closed form. We replace this operator by a standard generalized finite difference discretization on a point cloud sampled from the manifold, and show that the resulting method is consistent at the points of the cloud, at a rate governed by their 
    
[^232]: Tactus：用于自然灵巧操作数据采集的触觉与姿态感知手套

    Tactus: Tactile & Pose Sensing Glove for Natural Dexterity Data Collection

    [https://arxiv.org/abs/2610.39076](https://arxiv.org/abs/2610.39076)

    Tactus是一种织物集成的感知手套，通过超薄3D针织外形、策略性布置的电容触觉传感器和融合IMU的抗遮挡姿态跟踪三大设计，实现了自然人类操作演示中精确接触力与手部姿态的高质量数据采集。

    

    采集带有精确接触力和手部姿态的自然人类操作演示数据，对于机器人灵巧操作策略和基础世界模型学习至关重要——然而，目前没有任何可穿戴平台能够在保持自然操作行为的外形设计下，同时提供经过标定的指尖力感知和抗遮挡的手部跟踪。我们提出了Tactus，一种织物集成的感知手套，共同优化了三大设计支柱：(1) 超薄、贴合的3D针织外形（1.5毫米指尖叠层），旨在最大程度减少对自然操作的干扰；(2) 基于实证接触研究，在操作相关区域策略性布置的电容式触觉传感器；(3) 融合IMU的姿态跟踪，在手-物体遮挡最严重（也正是操作数据最有价值）的情况下仍能保持跟踪精度。Tactus还包含一套文档化的标定流程，以确保跨手套的一致性和单次会话（该部分内容被截断）

    arXiv:2610.39076v1 Announce Type: cross  Abstract: Collecting naturalistic human manipulation demonstrations with accurate contact force and hand pose is essential for robot dexterous manipulation policy and foundational world model learning - yet no existing wearable platform provides both calibrated fingertip force sensing and occlusion-resilient hand tracking in a form factor that preserves natural manipulation behavior. We present Tactus, a textile-integrated sensing glove that co-optimizes three design pillars: (1) a thin, conformal 3D-knit form factor (1.5 mm fingertip stackup) designed to minimize interference with natural manipulation, (2) strategically placed capacitive tactile sensors at manipulation-relevant regions informed by empirical contact studies, and (3) IMU-fused pose tracking that maintains accuracy through the hand-object occlusions where manipulation data is most valuable. Tactus includes a documented calibration pipeline for cross-glove consistency and per-sessi
    
[^233]: 非独立同分布无线网络中的通信高效聚类联邦学习

    Communication-Efficient Clustered Federated Learning in Non-IID Wireless Networks

    [https://arxiv.org/abs/2610.38970](https://arxiv.org/abs/2610.38970)

    提出了一种通信高效的聚类联邦学习框架，通过数据域聚类初始化基于模型的聚类，在不共享原始数据的前提下识别客户端相似性并为其分配专门模型，从而解决无线网络中的非独立同分布数据挑战。

    

    随着分布式学习成为先进6G服务的关键技术，当前和新兴的移动网络面临着日益严峻的隐私、可扩展性和性能挑战。异构无线环境中自然产生的非独立同分布数据会降低标准联邦学习（FL）方法的性能，从而进一步加剧了这些挑战。为解决客户端之间的非独立同分布数据问题，我们提出了一种聚类联邦学习（CFL）框架，该框架在训练过程中将客户端分配到专门的模型。数据域聚类用于初始化基于模型的聚类，从而在无需共享原始数据的情况下识别并利用客户端之间的相似性，同时降低通信成本。受毫米波波束选择问题的启发，我们在无线仿真数据集（DeepMIMO）上评估了所提出的方法，并用两个非独立同分布基准数据集（FEMNIST、Rotated CIFAR-10）加以补充，以隔离不同的非独立同分布特性。

    arXiv:2610.38970v2 Announce Type: replace  Abstract: Current and emerging mobile networks face increasing privacy, scalability, and performance challenges as distributed learning becomes essential for advanced 6G services. These challenges are amplified by the presence of non-IID data, naturally arising in heterogeneous wireless environments, that degrade the performance of standard Federated Learning (FL) approaches. To address non-IID data across clients, we propose a Clustered Federated Learning (CFL) framework that assigns clients to specialized models during training. Data-domain clustering initializes model-based clustering, enabling identification and exploitation of client similarities without sharing raw data while reducing communication cost.   Motivated by mmWave beam selection, we evaluate our methods on a wireless simulation dataset (DeepMIMO) and complement it with two non-IID benchmarks (FEMNIST, Rotated CIFAR-10) to isolate different non-IID characteristics. Our best co
    
[^234]: PDRVLA：具有物理动力学表征的视觉-语言-动作模型

    PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations

    [https://arxiv.org/abs/2610.37381](https://arxiv.org/abs/2610.37381)

    PDRVLA 提出一种轻量级物理动力学模块，通过预测未来 RGB、深度和光流特征来学习物理表征，在无需在线视频生成的情况下显著提升了视觉-语言-动作模型的机器人操作性能。

    

    视觉-语言-动作（VLA）模型能够根据视觉观测和语言指令生成机器人动作，但其对物理动力学的理解仍然有限。视频世界模型可以捕捉场景动态，但生成未来视频会给控制回路带来较高的延迟。我们提出了 PDRVLA，其轻量级的物理动力学模块从视觉-语言特征中提取物理 token 来条件化动作生成，而无需在线视频生成。该模块通过直接预测未来的 RGB、深度和光流特征进行学习，并借助冻结的预训练视频世界模型提供辅助监督。训练完全依赖公开可用的数据，并通过人-机器人视频合成、子任务标注和具身思维链（ECoT）标注进行扩充。在四个仿真基准测试中，PDRVLA 在 LIBERO 上达到 99.3% 的成功率，在 LIBERO-Plus 上达到 89.9%，在 RoboTwin 2 上达到 93.6%。

    arXiv:2610.37381v2 Announce Type: replace-cross  Abstract: Vision--Language--Action (VLA) models generate robot actions from visual observations and language instructions, but their understanding of physical dynamics remains limited. Video world models can capture scene dynamics, yet generating future videos introduces high latency into the control loop. We introduce PDRVLA, whose lightweight Physical Dynamics Module extracts physical tokens from vision--language features to condition action generation without online video generation. The module learns through direct prediction of future RGB, depth, and optical-flow features, with auxiliary supervision through a frozen pretrained video world model. Training relies entirely on publicly available data, expanded through human-to-robot synthesis, subtask annotation, and embodied chain-of-thought (ECoT) annotation. Across four simulation benchmarks, PDRVLA achieves success rates of 99.3% on LIBERO, 89.9% on LIBERO-Plus, 93.6% on RoboTwin~2.
    
[^235]: 面向可解释航空风险的混合框架：基于安全报告的因果链提取、基于物理的失效量化与干预分析

    A Hybrid Framework for Explainable Aviation Risk: Causal Chain Extraction from Safety Reports with Physics-Based Failure Quantification and Interventional Analysis

    [https://arxiv.org/abs/2610.35658](https://arxiv.org/abs/2610.35658)

    该论文提出一种混合框架，利用本地大语言模型从5.6万余份美国航空事故报告中提取因果链并构建因果知识图谱，结合基于物理的失效发生模型与漏泄噪声OR方法，将文本因素量化为失效模式概率，实现可解释的航空风险量化与干预分析，其提取的因果链可恢复90.8%的调查人员标注原因。

    

    航空管理机构保存着数十年积累的事故报告，其中的叙述记录了调查人员如何重建每一起事故。针对这些档案的现有计算研究大多只是将报告归入固定的分类体系。在本工作中，我们提出一种混合框架：首先由基于文本的语料库识别出某一失效模式的主导因素，随后由基于物理的模型量化该失效在给定运行工况下发生的可能性。利用本地部署的大语言模型，从56,202份美国事故报告中提取有序因果链，并聚合为因果知识图谱。在2,000份报告的评估样本上，这些因果链恢复了90.8%的调查人员所标注的原因。对于记录中带有驱动因素的每个物理因素，都建立了一个针对事故总体进行校准的发生模型。一个拟合于语料库的漏泄噪声OR（leaky noisy-OR）模型将这些因素组合为失效模式概率，从而支持干预分析。对于发动机动力丧失情形，化油器（摘要在此处被截断）

    arXiv:2610.35658v1 Announce Type: new  Abstract: Aviation authorities hold decades of accident reports whose narratives record how investigators reconstructed each accident. Computational work on these archives has mostly sorted reports into fixed taxonomies. In this work, we propose a hybrid framework in which a text-based corpus identifies the leading factors of a failure mode. Physics-based models then quantify how likely that failure is at a stated operating point. Ordered causal chains extracted from 56,202 American accident reports with a local large language model aggregate into a causation knowledge graph. On an evaluation sample of 2,000 reports, the chains recover 90.8% of investigator-flagged causes. Each physical factor whose drivers the record carries receives an occurrence model calibrated to the accident population. A leaky noisy-OR fitted to the corpus combines the factors into a failure mode probability that supports intervention. For engine power loss, the carburetor 
    
[^236]: 有限次Newton-Schulz迭代下Muon的末次迭代平稳性

    Last-Iterate Stationarity of Muon with Finitely Many Newton-Schulz Iterations

    [https://arxiv.org/abs/2610.34158](https://arxiv.org/abs/2610.34158)

    本文首次为采用有限次Newton-Schulz迭代的实用Muon优化器建立了末次迭代梯度收敛保证，证明在正稳定化和小常数学习率下可实现非凸目标的末次迭代平稳性，而无稳定化时即使在凸二次函数上也会循环。

    

    Muon在大规模语言模型训练中取得了出色的实证效果，这促使人们对其具体实现进行更深入的理论理解。与理想化的极因子更新不同，实际使用的Muon采用固定次数的五次Newton–Schulz迭代。现有分析主要建立了平均迭代或最优迭代的平稳性保证，而对末次迭代的收敛性理解仍然不足。我们研究了实际的有限步Muon在光滑、下有界的非凸目标函数上的末次迭代梯度收敛性。对于确定性梯度，我们证明在没有输入稳定化的情况下，Muon即使在光滑凸二次函数上也可能产生循环；在正稳定化条件下，我们对任意小于1的固定动量系数以及足够小的常数学习率建立了末次迭代梯度收敛性，并同时证明了梯度的平方可和性。对于随机梯度，我们首先证明持续有界、无偏的噪声（摘要在此处截断）……

    arXiv:2610.34158v1 Announce Type: new  Abstract: Muon has achieved strong empirical performance in large-scale language-model training, motivating a better theoretical understanding of its practical implementation. Unlike the idealized polar-factor update, practical Muon uses a fixed number of quintic Newton--Schulz iterations. Existing analyses mainly establish average- or best-iterate stationarity guarantees, leaving last-iterate convergence less understood. We study last-iterate gradient convergence of practical finite-step Muon on smooth, lower-bounded nonconvex objectives.   For deterministic gradients, we show that without input stabilization, Muon can cycle even on a smooth convex quadratic. With positive stabilization, we establish last-iterate gradient convergence for any fixed momentum coefficient below one and a sufficiently small constant learning rate, together with square-summability of gradients. For stochastic gradients, we first show that persistent bounded, unbiased n
    
[^237]: 反应式优化器

    Reactive Optimizers

    [https://arxiv.org/abs/2610.33826](https://arxiv.org/abs/2610.33826)

    本文提出OPAL，一种利用预训练语言模型智能体在单次训练中实时检查训练动态并重写优化器代码的反应式优化器，在未经调优的设置下可匹配或超越固定优化器配方。

    

    我们研究了一种反应式优化器，它利用预训练的语言模型智能体来检查训练动态，并在单次训练过程中重写可执行的优化器代码。该智能体可以针对各个参数组修改更新规则和调度计划，无需针对优化器的专门元训练，也无需从其他训练运行中转移的记忆。我们在语言建模和图像生成任务上评估了这种方法（称为OPAL），即在尚未针对目标设置调优训练配方的情况下。在所有评估的设置中，它的表现与选定的固定配方相当或有所提升。对其决策的分析显示，它会修改学习率调度、优化器结构、数值精度和动量。受控实验还表明，它能够适应反复出现的数据损坏，并减少过拟合。我们同时报告了这些行为以及该智能体的计算成本和实现要求。

    arXiv:2610.33826v1 Announce Type: new  Abstract: We study a reactive optimizer that uses a pretrained language-model agent to inspect training dynamics and rewrite executable optimizer code during a single training run. The agent can change update rules and schedules for individual parameter groups, without optimizer-specific meta-training or memory transferred from other runs. We evaluate this approach, OPAL, on language modeling and image generation when a recipe has not been tuned for the target setting. Across the evaluated settings, it matches or improves on selected fixed recipes. Analysis of its decisions shows changes to learning-rate schedules, optimizer structure, numerical precision, and momentum. Controlled experiments also show adaptation to repeated data corruption and reduced overfitting. We report these behaviors together with the agent's computational cost and implementation requirements.
    
[^238]: 面向历史文档转录的实体感知OCR系统评估

    Entity-Aware Evaluation of OCR Systems for Historical Document Transcription

    [https://arxiv.org/abs/2610.33004](https://arxiv.org/abs/2610.33004)

    该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。

    

    光学字符识别（OCR）对于历史档案数字化至关重要，然而字符错误率（CER）等标准评估指标并不一定能反映姓名、日期和标识符等有意义的信息是否被保留。在本工作中，我们研究了OCR错误的实体级分析，作为评估历史文档OCR质量的一种补充视角。我们使用来自Berrutti档案的175份文档的公开基准数据集，并辅以5,209个手动标注的实体提及（将于论文被接收后发布），比较了序列识别OCR系统、OCR专用视觉语言模型以及通用视觉语言模型。除了全局转录准确率之外，我们还在不同的字符级编辑容差下分析了实体的保留性和可恢复性。我们的实验表明，OCR系统的排名取决于评估标准：虽然Qwen 3.6在……（摘要原文在此处被截断）

    arXiv:2610.33004v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) is essential for digitizing historical archives, yet standard evaluation metrics such as Character Error Rate (CER) do not necessarily reflect whether meaningful information, such as names, dates, and identifiers, is preserved. In this work, we investigate entitylevel analysis of OCR errors as a complementary perspective for evaluating OCR quality in historical documents. Using a publicly available benchmark of 175 documents from the Berrutti Archive, augmented with 5,209 manually annotated entity mentions that will be released upon acceptance, we compare sequence-recognition OCR systems, OCR-specialized vision-language models, and general-purpose vision-language models. Beyond global transcription accuracy, we analyze entity preservation and recoverability under varying character-level edit tolerances. Our experiments show that OCR rankings depend on the evaluation criterion: while Qwen 3.6 achieves
    
[^239]: 弥合数据鸿沟：面向少样本扩散的源信息任务空间传输

    Bridging the Data Gap: Source-Informed Task-Space Transport for Few-Shot Diffusion

    [https://arxiv.org/abs/2610.31323](https://arxiv.org/abs/2610.31323)

    提出 Meta-Diff，通过在紧凑任务空间中一次性学习可复用的源-目标关系，使共享扩散主干无需针对每个目标域再训练即可从稀疏目标数据快速生成，尤其适用于医疗等目标数据难以获取的场景。

    

    许多实际部署场景要求模型能够快速地从数据丰富的源域迁移到仅有稀疏观测的相关目标域。当目标数据难以收集时，这一问题尤为突出，例如在存在持续人群或采集不平衡的医疗场景中。为每个目标域重新训练或微调高容量生成器代价高昂，且当目标数据稀缺时在统计上可能不可靠。我们提出 Meta-Diff，它转而在一个紧凑的任务空间中一次性学习反复出现的源-目标关系：一个共享编码器将每个分布映射为低维坐标；一个学习到的关系将新的源坐标传输到其预测的目标坐标，随后共享的扩散主干直接从该预测进行生成，无需针对特定目标进行再训练。受控的高斯混合实验揭示了该方法生效的条件：任务身份必须在表示中得以保留，关系必须能够在不同任务间泛化，并且源-

    arXiv:2610.31323v1 Announce Type: new  Abstract: Many deployments require a model to move rapidly from a data-rich source to a related target observed only sparsely. This is especially consequential where target data are difficult to collect, as in healthcare settings with persistent population or acquisition imbalances. Retraining or fine-tuning a high-capacity generator for every target is costly and can be statistically unreliable when target data are scarce. We introduce Meta-Diff, which instead learns a recurring source--target relation once in a compact task space. A shared encoder maps each distribution to a low-dimensional coordinate; a learned relation transports a new source coordinate to its predicted target, and a shared diffusion backbone generates from that prediction without target-specific retraining. Controlled Gaussian-mixture experiments identify when this works: task identity must survive the representation, the relation must generalise across tasks, and the source-
    
[^240]: Trace2World：从真实交互轨迹重建应用世界以实现可扩展的GUI智能体训练

    Trace2World: Reconstructing App Worlds from Real Interaction Traces for Scalable GUI Agent Training

    [https://arxiv.org/abs/2610.30792](https://arxiv.org/abs/2610.30792)

    Trace2World通过将全局行为重建与环境实现分离，从真实交互轨迹构建忠实于真实应用行为的可执行GUI环境用于智能体训练，在十个应用的200个工作流上实现91.1%的端到端复现率，且基于来源的验证使真实设备上的训练增益保留率达94.0%。

    

    扩展GUI强化学习需要能够保留真实应用行为的环境。然而，将行为推断与编码相耦合，可能会把部分轨迹转变成状态转换错误的可运行模拟（mock）。我们提出Trace2World，它将全局行为重建与具体实现分离开来。该方法首先将多条轨迹调和为一个共享的、与来源关联的应用模型，然后编译出可执行的环境。通过独立回放来源工作流来定位不一致之处，从而进行有边界的实现修复。经过验证的世界进而可以在来源支持的行为范围内，支持面向学习者自适应的任务与环境增强。Trace2World从1,570条轨迹中构建了50个环境。在十个应用的200个工作流上，91.1%的工作流被端到端复现，而最强的任务局部匹配基线仅为59.8%。基于来源的验证在真实设备上保留了94.0%的模拟训练增益，而自我验证下仅为54.7%。学习者引导（摘要在此处截断）

    arXiv:2610.30792v1 Announce Type: new  Abstract: Scaling GUI reinforcement learning requires environments that preserve real-app behavior. Yet coupling behavior inference with coding can turn partial traces into runnable mocks with incorrect state transitions. We introduce \textbf{Trace2World}, which separates global behavior reconstruction from implementation. It reconciles traces into a shared, source-linked app model, then compiles an executable environment. Independent replay of source workflows localizes mismatches for bounded implementation repair. Validated worlds then support learner-adaptive task and environment augmentation within source-supported behavior. From 1,570 traces, Trace2World constructs 50 environments. On 200 workflows across ten apps, 91.1% are reproduced end to end, versus 59.8% for the strongest matched task-local baseline. Source-grounded validation retains 94.0% of simulated training gains on real devices, versus 54.7% under self-validation. Learner guidance
    
[^241]: 非对称在策略蒸馏学习情境化探索

    Asymmetric On-Policy Distillation Learns In-Context Exploration

    [https://arxiv.org/abs/2610.30105](https://arxiv.org/abs/2610.30105)

    提出ASTEROID学生-教师蒸馏框架，让拥有特权信息的教师基于学生在部分可观测环境中自主收集的在策略历史上下文提供动作标签，从而使学生学会测试时的情境化探索行为。

    

    在部分可观测条件下的序贯决策十分困难，因为智能体必须先通过探索来推断隐藏的、与任务相关的信息，然后才能有效地行动。通过在线强化学习从零开始学习这种信息收集行为的样本效率很低，而离线模仿学习则会在不同的隐藏状态之间平均专家动作，从而无法学到探索行为。在这项工作中，我们提出了 ASTEROID，这是一个简单而有效的学生-教师监督学习框架，能够学习测试时的探索行为。尽管拥有特权信息的教师从不需要探索，我们证明了基于历史条件的学生模型在与策略上下文一起训练时能够学会探索：学生首先在部分可观测条件下收集在策略上下文，特权教师则基于该上下文提供动作标签。通过直接在学生的在策略历史条件下蒸馏专家动作，学生能够学会系统地收集与任务相关的信息。

    arXiv:2610.30105v1 Announce Type: cross  Abstract: Sequential decision-making under partial observability is difficult because agents must explore to infer hidden, task-relevant information before acting effectively. Learning such information-gathering behaviors tabula rasa via online reinforcement learning is sample-inefficient, whereas offline imitation averages expert actions across hidden states and fails to learn exploration. In this work, we propose ASTEROID, a simple and effective student--teacher supervised learning framework that can learn test-time exploration. Although a privileged expert never needs to explore, we demonstrate that a history-conditioned student can learn to do so when trained with on-policy contexts: the student first collects on-policy context under partial observability, and the privileged expert provides action labels conditioned on that context. By distilling expert actions directly conditioned on the student's on-policy history, the student learns to sy
    
[^242]: 基于遗传规划的混合整数线性规划可解释割平面选择方法

    Interpretable Cutting Plane Selection for Mixed-Integer Linear Programming Using Genetic Programming

    [https://arxiv.org/abs/2610.29311](https://arxiv.org/abs/2610.29311)

    本文提出首个基于遗传规划的完全可解释割平面选择框架GPSCS，通过演化符号化评分函数将手工启发式策略显式泛化为可解释的自适应割平面选择策略，弥补了神经网络方法可解释性差且缺乏收敛保证的不足。

    

    割平面能够在不排除任何可行整数解的前提下，从混合整数线性规划（MILP）中移除分数解，因而是现代MILP求解器的关键组成部分。当前最先进的求解器依赖于经过精细调优的手工割平面策略，尽管这些策略行之有效，但本质上仍是静态的，可能并非最优。近年来，已有基于学习的方法被提出用于设计自适应割平面策略；然而，这些方法大多基于神经网络，可解释性有限，且无法提供收敛性保证。在本文中，我们提出了基于遗传规划与统计的割平面选择方法（GPSCS），这是首个将手工割平面选择启发式规则显式泛化为完全可解释策略的基于学习的框架。首先，遗传规划模型演化出一个符号化评分函数，利用从（摘要在此处被截断）

    arXiv:2610.29311v1 Announce Type: new  Abstract: Cutting planes (cuts) remove fractional solutions from mixed-integer linear programs (MILPs) without excluding any feasible integer solutions, making them a key ingredient in modern MILP solvers. Current state-of-the-art solvers rely on fine-tuned handcrafted cut strategies, which, despite their effectiveness, remain inherently static and can be suboptimal. More recently, learning-based approaches have been proposed to design adaptive cut strategies; however, most of these methods are based on neural networks, offering limited interpretability and providing no guarantees of convergence. In this paper, we propose \emph{Genetic Programming and Statistics-based Cut Selection} (GPSCS), the first learning-based framework that explicitly generalizes handcrafted cut selection heuristics into a fully interpretable policy. First, a genetic programming model evolves a symbolic scoring function that ranks candidate cuts using features derived from 
    
[^243]: 从数据增强到合成证据：在数据集入口处评估生成的胸部X光片

    From Augmentation to Synthetic Evidence: Evaluating Generated Chest Radiographs at Dataset Intake

    [https://arxiv.org/abs/2610.29152](https://arxiv.org/abs/2610.29152)

    该研究提出在数据集入口处对生成的胸部X光片进行质量检查、来源分类和记录验证的三重评估，发现基于质量的选择虽能显著提高质量通过率，但无法使生成图像同时通过质量与来源双重检查。

    

    生成的胸部X光片作为明示的数据增强可能是有用的，但当同一张图像被提交作为真实检查的记录时，它就变成了缺乏依据的证据。我们通过区分三个问题来研究这一问题：图像是否通过自动化质量检查、来源分类器是否将其标记为生成图像，以及可信记录是否支持所声称的采集过程。在我们的实验中，Qwen从1,000张现有的Cheff识别或SynthCheX候选图像池中最多选择20张图像。接收方使用MedGemma评估图像质量，并使用基于冻结DINOv3特征的分类器预测图像来源。基于质量的选择使Cheff识别图像的质量通过率从20张中的12张提高到19张，SynthCheX图像从8张提高到17张。我们没有发现任何证据表明选择能够改善同时通过两项检查的情况：没有选中的Cheff识别图像能同时通过两项检查，而均匀选择、质量选择和质量约束的替代选择分别产生2张、1张和……（摘要在此处截断）

    arXiv:2610.29152v1 Announce Type: cross  Abstract: A generated chest radiograph can be useful as declared augmentation, but the same image becomes unsupported evidence when submitted as the record of a real examination. We study this problem by separating three questions: whether an image passes an automated quality check, whether an origin classifier flags it as generated, and whether a trusted record supports the claimed acquisition. In our experiments, Qwen selects up to 20 images from pools of 1,000 existing Cheff-identified or SynthCheX candidates. A receiver evaluates quality with MedGemma and predicted origin with a classifier on frozen DINOv3 features. Quality-based selection raises quality passes from 12 to 19 of 20 for Cheff-identified images and from 8 to 17 for SynthCheX. We find no evidence that selection improves passage through both checks: no selected Cheff-identified image passes both, while uniform, quality, and quality-constrained surrogate selection produce 2, 1, an
    
[^244]: 一种用于逼真啮齿动物运动的潜在控制模型

    A latent control model for realistic rodent motion

    [https://arxiv.org/abs/2610.27279](https://arxiv.org/abs/2610.27279)

    提出了SCAMPER系统，利用具有双重先验的变分架构训练具身虚拟大鼠模型，使其在目标导向行为和随机探索中都能产生自然的运动，弥补了以往具身动物建模的不足。

    

    开发动物行为的具身模型是神经-AI领域的一个重要目标。具身模型将增强我们对生物智能体和人工智能体进行直接比较的能力。然而，要使这种比较有效，我们需要真正像动物一样运动的具身模型。先前的工作已经开发了虚拟动物身体以及通过模仿学习训练逼真运动的技术。然而，迄今为止，还没有一个具身动物建模系统能够既在经过训练的目标导向行为中，又在由随机生成驱动的探索过程中产生自然的运动。在这里，我们介绍了SCAMPER（通过增强运动先验实现具身强化学习模拟控制），这是一个训练具身虚拟大鼠模型的系统，能够在目标导向行为和随机探索过程中都实现自然的运动。SCAMPER使用一种具有两种不同先验的变分架构，即一种用于……的运动先验

    arXiv:2610.27279v1 Announce Type: new  Abstract: The development of embodied models of animal behavior is a major goal of the neuro-AI community. Embodied models will enhance our ability to make direct comparisons between biological and artificial agents. To make such comparisons effective, though, we need embodied models that actually move like animals. Previous work has developed virtual animal bodies and techniques for training realistic motion through imitation. However, to date, there is no system for embodied animal modeling that produces natural movements both during trained goal-directed behavior, and during exploration driven by random generation. Here, we introduce Simulated Control via Augmented Motor Priors for Embodied Reinforcement learning (SCAMPER), a system for training an embodied virtual rat model, enabling natural movements during both goal-directed behavior and random exploration. SCAMPER uses a variational architecture with two distinct priors, a motion prior for 
    
[^245]: FrameScout：为长视频理解搜寻与查询相关的帧

    FrameScout: Scouting Query-Relevant Frames for Long Video Understanding

    [https://arxiv.org/abs/2610.23975](https://arxiv.org/abs/2610.23975)

    提出FrameScout框架，通过结合滑动窗口KV缓存与后继聚合的流式选择器架构生成时间上下文化的帧嵌入，并利用帧-查询对比学习目标直接训练选择器，从而为长视频理解高效选取与查询最相关的关键帧。

    

    长视频理解要求视频大语言模型（VideoLLM）能够回答关于时长跨越数十分钟乃至数小时的视频的查询，但这类长视频产生的视觉令牌数量远超当前VideoLLM所能处理的范围。为解决这一问题，研究者提出了关键帧选择器，通过计算帧嵌入与查询嵌入之间的相似度来选取与查询最相关的帧。然而，现有的选择器要么在编码帧时独立处理、缺乏时间上下文，要么以孤立方式处理视频片段，且所采用的排序目标难以优化。为缓解上述局限，我们提出FrameScout，其包含两项核心设计：(i) 一种流式选择器架构，利用滑动窗口KV缓存和后继聚合机制来生成具有时间上下文的帧嵌入；(ii) 一种帧-查询对比目标，在共享空间中对齐帧嵌入与查询嵌入，直接训练选择器区分相关帧与无关帧。

    arXiv:2610.23975v2 Announce Type: replace-cross  Abstract: Long video understanding requires VideoLLMs to answer queries about videos spanning tens of minutes to hours, but such long videos produce far more visual tokens than current VideoLLMs can process. To address this issue, keyframe selectors are proposed to select the most query-relevant frames by computing the similarity between frame and query embeddings. Existing selectors either encode frames independently without temporal context or process chunks in isolation with ranking objectives that are difficult to optimize. To alleviate those limitations, we propose FrameScout with two core designs: (i) a streaming selector architecture that leverages a sliding-window KV cache and successor aggregation to produce temporally-contextualized frame embeddings; (ii) a frame-query contrastive objective that aligns frame and query embeddings in a shared space, directly training the selector to distinguish relevant frames from irrelevant one
    
[^246]: 预填充阶段专家削减在何处损害MoE模型：少数提示词Token承担了大部分短答案F1损失

    Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

    [https://arxiv.org/abs/2610.22454](https://arxiv.org/abs/2610.22454)

    研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。

    

    在预填充阶段让提示词Token通过更少的路由专家，是降低混合专家模型服务成本的一种简单方法，但这种做法究竟损害了什么尚不清楚。我们在四个MoE模型家族上，通过Token级、层级和注意力路径干预，在短答案问答任务中定位了这种损害。在专家数量减半的情况下，三个家族的F1损失主要来自“答案形式”：模型仍能找到答案，但不再以简短形式作答。在Qwen3-30B-A3B中，这种损失发生在答案之前的两个聊天模板Token上；在DeepSeek-V2-Lite中，损失发生在其指令所在的前8-32个提示词Token上。按答案似然计算，这些位置分别约占全部损害的一半和五分之一。专家削减对首个Token的注意力质量、范数和最大激活的改变不足4%，而恢复“超级专家”或超大激活最多只能挽回约四分之一的损失，远低于保持敏感位置启用全部专家的效果。

    arXiv:2610.22454v1 Announce Type: new  Abstract: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts
    
[^247]: FutureLift：将轨迹提议提升为双向未来图以用于运动预测

    FutureLift: Lifting Trajectory Proposals into Bidirectional Future Graphs for Motion Forecasting

    [https://arxiv.org/abs/2610.22215](https://arxiv.org/abs/2610.22215)

    FutureLift提出了一种与主干网络无关的轨迹提议改进方法，将多模态提议表示为带有双向时间边的异构未来图，使后续证据能够在单次遍历中修正较早的片段，从而提升自动驾驶运动预测的准确性。

    

    准确的运动预测对于安全的自动驾驶至关重要，然而即使是强大的预测器也会在其多模态提议中留下残差误差。现有的改进器要么压缩完整的提议，要么按顺序更新其片段，这使得完整的提议未来无法作为显式的关联上下文用于修正。我们提出了FutureLift，一种与主干网络无关的改进器，它将每个多模态提议表示为一个异构未来图，其中具有空间锚定的时间节点，这些节点连接到智能体历史、地图元素、替代模态，以及在联合预测场景下连接到其他智能体的未来轨迹。双向时间边允许后续证据在单次遍历中修正较早的片段，更新后的节点则预测残差坐标和置信度修正。FutureLift仅需要提议、分数和场景上下文作为输入，为结构不同的预测主干网络提供了一个通用的改进接口。在...

    arXiv:2610.22215v1 Announce Type: cross  Abstract: Accurate motion prediction is critical for safe autonomous driving, yet even strong forecasters leave residual errors in their multimodal proposals. Existing refiners compress complete proposals or update their segments sequentially, preventing the full proposed future from serving as explicit relational context for correction. We introduce FutureLift, a backbone-agnostic refiner that represents each multimodal proposal as a heterogeneous future graph, with spatially grounded temporal nodes connected to agent histories, map elements, alternative modes, and, for joint prediction, other agents future trajectories. Bidirectional temporal edges allow later evidence to revise earlier segments in a single pass, and the updated nodes predict residual coordinate and confidence corrections. Requiring only proposals, scores, and scene context, FutureLift provides a common refinement interface for structurally different forecasting backbones. On 
    
[^248]: 基于Smooth-L1正则化的稀疏联邦回声状态网络时间序列分类方法

    Sparse Federated Echo State Networks for Time-Series Classification via Smooth-L1 Regularization

    [https://arxiv.org/abs/2610.21742](https://arxiv.org/abs/2610.21742)

    本文提出基于Smooth-L1正则化的稀疏回声状态网络SL1ESN，利用与牛顿法兼容的可微稀疏化方法实现读出层高度稀疏，并支持二阶联邦聚合，在七个时间序列分类基准上达到75%–95%的稀疏度。

    

    回声状态网络（ESN）作为储备池计算的轻量级实现，为资源受限环境中的时间数据处理提供了高效的解决方案。虽然标准ESN具有计算效率优势，但其密集的读出层会增加存储模型的大小和逐样本的推理成本。本文提出了一种Smooth-L1正则化的ESN（SL1ESN），在保持基于梯度优化的可微性的同时，促进读出层的稀疏性。该方法采用二次可微的Smooth-L1近似，与牛顿法兼容，我们将其既用于集中式训练，也用于二阶（牛顿步）联邦聚合方案。在所有数据集和两种设置下共享单一固定收缩阈值的情况下，在七个时间序列分类基准上的实验表明，SL1ESN学习到的读出层在集中式设置中稀疏度达到77%–95%，在联邦设置中达到75%–95%。

    arXiv:2610.21742v1 Announce Type: new  Abstract: Echo State Networks (ESNs), as lightweight implementations of reservoir computing, provide efficient solutions for temporal data processing in resource-constrained environments. While standard ESNs offer computational efficiency, their dense readout layers inflate the stored model and the per-sample inference cost. This paper introduces a Smooth-L1 regularized ESN (SL1ESN) that promotes sparsity in the readout layer while preserving differentiability for gradient-based optimization. The method employs a twice-differentiable Smooth-L1 approximation compatible with Newton's method, which we exploit both for centralized training and for a second-order (Newton-step) federated aggregation scheme. Under a single fixed shrinkage threshold shared by every dataset and both settings, experiments on seven time-series classification benchmarks show that SL1ESN learns readouts that are $77$--$95\%$ sparse in the centralized setting and $75$--$95\%$ s
    
[^249]: 面向高效且有效的无乘法器LLM推理的分数二次幂量化

    Fractional Power-of-Two Quantization for Efficient and Effective Multiplier-Free LLM Inference

    [https://arxiv.org/abs/2610.21257](https://arxiv.org/abs/2610.21257)

    该论文提出分数二次幂量化（FPoT），采用以√2为基数的半八度分数量化网格，在保持完全整数移位-加法数据通路实现无乘法器推理的同时，解决了传统二次幂量化在低位宽下的精度下降问题，实现了实用的4位LLM权重量化。

    

    大语言模型（LLM）需要高效的推理，其中线性投影中的乘累加（MAC）运算主导了计算和能耗成本。训练后量化（PTQ）通过将权重和激活映射到低位整数来降低这一成本。二次幂权重量化通过移位-加法逻辑有望实现高效的无乘法器LLM推理，但由于其粗糙的以2为基数的量化网格，在低位宽下会出现显著的精度下降。我们引入了分数二次幂量化（FPoT），这是一种以√2为基数（半八度）的分数二次幂网格，在保持完全整数移位-加法数据通路的同时，实现了实用的4位权重量化。该方法结合了：(i) 以√2为基数的权重网格与均匀整数激活相结合，(ii) 在校准阶段进行并被吸收到静态逐通道缩放因子中的网格对齐细化，以及 (iii) 轻量级的单加法器√2近似，其误差在权重量化过程中被吸收。

    arXiv:2610.21257v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demand efficient inference, where the multiply-accumulate (MAC) operations in linear projections dominate compute and energy cost. Post-training quantization (PTQ) reduces this cost by mapping weights and activations to low-bit integers. Power-of-Two (PoT) weight quantization promises efficient multiplier-free LLM inference via shift-add logic, but suffers significant accuracy degradation at low bitwidths due to its coarse base-2 quantization grid. We introduce Fractional PoT (FPoT), a base-$\sqrt{2}$ (half-octave) fractional PoT grid that achieves practical 4-bit weight quantization while preserving a fully integer shift-add datapath. The method combines (i) the base-$\sqrt{2}$ weight grid with uniform integer activations, (ii) a calibration-time grid-alignment refinement absorbed into the static per-channel scale, and (iii) a lightweight 1-adder $\sqrt{2}$ approximation whose error is absorbed during weig
    
[^250]: 重新审视扩散语言模型与自回归语言模型之间16倍的预训练计算差距

    Revisiting the 16x Pretraining Compute Gap between Diffusion and Autoregressive Language Models

    [https://arxiv.org/abs/2610.17590](https://arxiv.org/abs/2610.17590)

    本研究采用现代扩散语言模型训练实践（精确似然评估、块扩散训练、编码器-解码器架构）重新审视扩散语言模型的缩放定律，通过约1700个模型的大规模实验发现，扩散语言模型与自回归语言模型之间所谓的16倍预训练计算差距远小于此前声称的水平。

    

    扩散语言模型因其快速、并行的解码能力而受到广泛关注，但人们普遍认为其训练效率远低于自回归语言模型（ARMs）：一项被广泛引用的研究报告指出，扩散语言模型需要大约16倍于自回归模型的训练计算量才能达到相同的似然水平，这一数字使得从零开始预训练扩散语言模型在大规模应用上显得望而却步。我们发现，这一数字背后的研究方法已经与当前扩散语言模型的实践脱节：它使用证据下界（ELBO）来评估模型，并训练全序列的仅解码器模型。因此，我们重新审视了扩散语言模型的缩放定律，采用了模型自身解码顺序下的精确似然、块扩散训练、编码器-解码器架构，并对实践中使用的块大小进行了扫描。在这一新协议下，我们研究了十个模型变体、总计约1700个模型，发现预训练扩散语言模型的计算成本远……

    arXiv:2610.17590v1 Announce Type: cross  Abstract: Diffusion language models (DLMs) have attracted wide attention for their fast, parallel decoding, but they are believed to be far less efficient to train than autoregressive language models (ARMs): a widely cited study reports that a DLM needs roughly 16x the training compute of an ARM to match its likelihood, a figure that has made pretraining DLMs from scratch look prohibitive at scale. We find that the methodology behind this figure has fallen out of step with current DLM practice: it scores the model with its evidence lower bound (ELBO) and trains a full-sequence, decoder-only model. We therefore revisit the scaling laws of DLMs, adopting the exact likelihood under the model's own decoding order, block diffusion training, an encoder-decoder architecture, and a sweep over the block sizes used in practice. Under this new protocol, we study across ten model variants, about 1,700 models in total, and find that pretraining DLMs is far c
    
[^251]: 从视网膜神经节细胞脉冲群体活动中实现神经形态解码与动态图像重建

    Neuromorphic Decoding and Dynamical Image Reconstruction from Populations of Spiking Retinal Ganglion Cells

    [https://arxiv.org/abs/2610.17154](https://arxiv.org/abs/2610.17154)

    该研究构建了端到端的脉冲神经网络，首次实现了完全在脉冲域内对视网膜神经节细胞群体活动的解码与动态场景的实时追踪重建，且计算开销极低。

    

    从视网膜神经节细胞的脉冲活动中解码视觉刺激，既有助于我们理解视网膜编码，也有助于视觉假体的设计。自然视觉是动态的，然而最先进的解码器大多使用人工神经网络重建单帧图像，这些网络处理的是密集的连续值信号，而非视网膜稀疏的异步脉冲。我们探究了动态场景能否完全在脉冲域内被重建和追踪，以及其计算成本如何。我们构建了一个端到端的脉冲神经网络（SNN），其神经元随时间整合记录到的视网膜神经节细胞脉冲。在高密度微电极阵列上对离体小鼠视网膜中多达1836个视网膜神经节细胞的群体进行了记录，同时呈现闪光图像、移动条形、漂移光栅和随机移动的手写数字。对于移动数字，一个共享的脉冲编码器连接两个读出层，其中一个用于重建图像……

    arXiv:2610.17154v2 Announce Type: replace-cross  Abstract: Decoding visual stimuli from the spiking activity of retinal ganglion cells (RGCs) informs both our understanding of the retinal code and the design of visual prostheses. Natural vision is dynamic, yet state-of-the-art decoders mostly reconstruct single frames with artificial neural networks that process dense, continuous-valued signals rather than the retina's sparse, asynchronous spikes. We asked whether dynamic scenes can be reconstructed and tracked entirely within the spike domain, and at what computational cost. We built an end-to-end spiking neural network (SNN) whose neurons integrate the recorded RGC spikes over time. Populations of up to 1836 RGCs were recorded ex vivo from mouse retina on high-density microelectrode arrays while flashed images, moving bars, drifting gratings and randomly moving handwritten digits were presented. For moving digits, one shared spiking encoder fed two readouts, one reconstructing the im
    
[^252]: DTA-GT：用于神经网络表示学习的方向与拓扑感知图Transformer

    DTA-GT: Direction- and Topology-Aware Graph Transformer for Neural Network Representation Learning

    [https://arxiv.org/abs/2610.16822](https://arxiv.org/abs/2610.16822)

    提出DTA-GT，一种方向与拓扑感知的图Transformer，通过在节点初始化、谱编码、注意力交互和特征更新全流程中保持有向计算语义，显著提升了神经架构搜索中精度与延迟预测的性能。

    

    架构属性预测器通过从小规模已评估的候选集中估计精度或延迟，来降低神经架构搜索的成本。其关键难点在于神经架构是带有操作标签的有向无环图（DAG）：边的方向定义了计算流，但现有预测器往往仅将方向视为局部结构线索。我们提出了DTA-GT，一种基于编码器级方向一致性的方向与拓扑感知图Transformer，它在节点初始化、全局谱编码、成对交互和特征更新等各个环节中都保持了有向计算语义。DTA-GT通过方向与拓扑感知的节点初始化、磁拉普拉斯谱编码、方向对偶结构注意力以及方向敏感的MoE-FFN来实现这一原则。在NAS-Bench-101和NAS-Bench-201上的精度与延迟预测任务中，DTA-GT持续优于代表性的序列模型、GNN和Transformer方法。

    arXiv:2610.16822v1 Announce Type: cross  Abstract: Architecture attribute predictors reduce the cost of neural architecture search by estimating accuracy or latency from a small set of evaluated candidates. The key difficulty is that neural architectures are operation-labeled DAGs: edge direction defines computational flow, but existing predictors often treat direction as a local structural cue. We propose DTA-GT, a Direction- and Topology-Aware Graph Transformer based on encoder-wide directional consistency, which preserves directed computational semantics across node initialization, global spectral encoding, pairwise interaction, and feature update. DTA-GT realizes this principle with direction- and topology-aware node initialization, Magnetic Laplacian spectral encoding, direction-dual structural attention, and a Direction-Sensitive MoE-FFN. Across accuracy and latency prediction on NAS-Bench-101 and NAS-Bench-201, DTA-GT consistently outperforms representative sequence-, GNN-, Tran
    
[^253]: 乘积蒙特卡罗：密度乘积下的期望计算

    Product Monte Carlo: Expectation Under a Product of Densities

    [https://arxiv.org/abs/2610.12702](https://arxiv.org/abs/2610.12702)

    该论文提出乘积蒙特卡罗（PMC）算法，通过将难以处理的密度乘积期望转化为全贝叶斯回归问题，仅利用各分量密度的蒙特卡罗样本即可计算任意期望，并具备一致性保证与不确定性量化能力。

    

    聚合独立计算的子样本推断是大规模贝叶斯计算、联邦学习和意见汇聚领域公认的难题。其核心问题是近似一个密度正比于各个体密度乘积的分布。我们提出了乘积蒙特卡罗（Product Monte Carlo, PMC），这是一种仅利用来自各个体密度的蒙特卡罗样本、即可计算密度乘积下任意期望的算法。我们通过一族可计算的条件期望，将难以直接处理的乘积期望转化为一个回归问题。除了点估计之外，我们的全贝叶斯回归框架还提供了连贯的不确定性量化以及有原则的诊断方法。我们为PMC估计量建立了一致性方面的理论保证，并刻画了其收敛速率。在改变子样本数量、维度和异质性的模拟与真实数据实验中，该方法均表现出良好效果。

    arXiv:2610.12702v1 Announce Type: cross  Abstract: Aggregating independently computed sub-sample inferences is a well-known challenge in large-scale Bayesian computation, federated learning, and opinion pooling. The core problem is to approximate a distribution whose density is proportional to a product of individual densities. We propose Product Monte Carlo (PMC), an algorithm for computing arbitrary expectations under a product of densities using only Monte Carlo draws from individual densities. We turn the intractable product expectation into a regression problem by expressing it through a family of computable conditional expectations. Besides the point estimate, our fully Bayesian regression formulation provides coherent uncertainty quantification and principled diagnostics. We establish theoretical guarantees on consistency and characterize convergence rates for the PMC estimator. Across simulated and real-data experiments varying the number, dimensionality, and heterogeneity of t
    
[^254]: 基于支撑集保持蒸馏的多智能体协调

    Multi-Agent Coordination via Support-Preserving Distillation

    [https://arxiv.org/abs/2610.10087](https://arxiv.org/abs/2610.10087)

    提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。

    

    离线多智能体强化学习（MARL）日益依赖生成式策略来建模多模态联合行为，通常是在集中式训练分散式执行（CTDE）框架下，将集中式教师蒸馏为分散式单步执行者。我们识别出教师训练阶段的一种失败模式：标准的基于流的教师将噪声与回放目标独立配对，导致相邻的噪声样本可能被路由到相互冲突的协调模式。教师随后会在有效模式之间产生样本，而由于蒸馏损失将每个局部执行者回归到给定局部输入下教师输出的条件均值，这一误差不会被吸收，而是传播到学生模型。为了消除这一教师侧的伪影，我们提出模式支撑半离散最优传输（MoSDOT），它将多模态回放数据总结为具有预设容量的有限模式支撑集，并在教师训练之前使用条件半离散最优传输将每个噪声样本分配到单一模式……

    arXiv:2610.10087v1 Announce Type: new  Abstract: Offline MARL increasingly relies on generative policies to model multimodal joint behavior, typically by distilling a centralized teacher into decentralized one-step actors under the CTDE. We identify a failure mode at the teacher training stage: standard flow-based teachers pair noise with replay targets independently, so nearby noise samples can be routed toward conflicting coordination modes. The teacher then produces samples between valid modes, and because the distillation loss regresses each local actor onto the conditional mean of the teacher's output given local input, this error is not absorbed but propagated to the student. To remove this teacher-side artifact, we propose Mode-Support Semi-Discrete Optimal Transport (MoSDOT), which summarizes multimodal replay into a finite mode support with prescribed capacities and uses conditional semi-discrete optimal transport to assign each noise sample to a single mode before teacher tra
    
[^255]: 引用你所探索的：基于可验证证据的医疗知识图谱预算感知LLM推理

    Cite What You Explore: Budget-Aware LLM Reasoning over Medical KGs with Verifiable Evidence

    [https://arxiv.org/abs/2610.07739](https://arxiv.org/abs/2610.07739)

    该论文提出了BAR框架，首次将成本受限的KG探索、按来源质量区分的可验证证据以及可引用的推理依据三者结合，利用LLM在医疗知识图谱上进行推理，以弥补EHR中缺失的依赖关系并提升出院后风险预测的可靠性。

    

    基于电子健康记录（EHR）的出院后风险预测十分困难，因为许多将出院时观察结果与下游并发症联系起来的依赖关系——如共病级联和药物-疾病相互作用——在记录中是缺失的。外部医疗知识图谱（KG）可以补充这些缺失的依赖关系，但追踪它们需要满足三个特性：KG探索必须保持成本受限，检索到的证据必须按来源质量进行区分，且所得到的推理依据必须可引用以供回顾性审查。大型语言模型（LLM）能够在结构化证据上进行规划与验证，使其成为KG推理的自然候选者，但现有的基于LLM的方法无法同时满足这三个特性。在本文中，我们提出了BAR，一个面向医疗知识图谱的预算感知LLM推理框架，包含三项贡献。首先，BAR将原始KG精炼为疾病特定的证据图，其边（摘要截断于此）……

    arXiv:2610.07739v1 Announce Type: new  Abstract: Post-discharge risk prediction from electronic health records (EHRs) is difficult because many dependencies that link discharge-time observations to downstream complications, such as comorbidity cascades and drug-disease interactions, are absent from the record. External medical knowledge graphs (KGs) can supply these missing dependencies, but tracing them demands three properties: KG exploration must remain cost-bounded, retrieved evidence must be differentiated by source quality, and the resulting rationale must be citable for retrospective review. Large language models (LLMs) can plan and verify over structured evidence, making them natural candidates for KG reasoning, but existing LLM-based methods do not satisfy these three properties jointly. In this paper, we propose BAR, a Budget-Aware LLM Reasoning framework over medical KGs with three contributions. First, BAR refines the raw KG into disease-specific evidence graphs whose edges
    
[^256]: 面向稀疏观测偏微分方程逆问题的后向一致扩散采样

    Backward-Consistent Diffusion Sampling for Sparsely Observed PDE Inverse Problems

    [https://arxiv.org/abs/2610.04624](https://arxiv.org/abs/2610.04624)

    该论文提出FunBCS——一种输入空间优化方法，通过寻找使去噪器重构物理一致的最佳输入，证明了其在不连续PDE逆问题中能够正确最小化未观测误差，而现有输出空间方法则无法做到。

    

    从极其稀疏的观测中恢复偏微分方程（PDE）系数场是一个严重病态的逆问题，生成式机器学习方法（例如扩散模型）已成为编码先验的主流方式。近期最先进的扩散求解器将这些先验提升到函数空间，在扩散去噪器的输出空间中寻找物理一致的重构。我们证明，在不连续PDE的设定下，输出空间方法可能无法用正确的系数场适当地最小化未观测误差。因此，我们提出了函数空间后向一致采样，这是一种用于求解PDE问题的输入空间优化方法，旨在找到最佳输入使得去噪器的重构是物理一致的。随后我们证明，与输出空间优化方法不同，FunBCS能够适当地最小化未观测误差。

    arXiv:2610.04624v2 Announce Type: replace  Abstract: Recovering Partial Differential Equation (PDE) coefficient fields from extremely sparse observations is a severely ill-posed inverse problem for which generative machine learning methods (e.g., diffusion models) have become a leading way to encode the prior. Recent state-of-the-art diffusion solvers lift these priors to function spaces, finding a physics-consistent reconstruction in the output space of the diffusion denoiser. We prove that, in a discontinuous PDE setting, output space methods can result in failure to appropriately minimize the unobserved error with the correct coefficient field. Consequently, we propose Function Space Backward-Consistent Sampling (FunBCS), an input space optimization approach for solving PDE problems which aims to find the best input such that the denoiser reconstruction is physics-consistent. We then prove that FunBCS appropriately minimizes the unobserved error, unlike output space optimization met
    
[^257]: 用于自适应网格PDE预测的动态四叉树标记化与Transformer

    Dynamic Quadtree Tokenization and Transformer for Adaptive Mesh PDE Forecasting

    [https://arxiv.org/abs/2610.04044](https://arxiv.org/abs/2610.04044)

    提出WAMRViT，一种基于动态四叉树标记化与小波细化判据的视觉Transformer，无需预设标记数量即可自适应调整网格分辨率，实现对细尺度PDE的长时程稳定预测。

    

    视觉Transformer（ViT）二次方复杂度的注意力计算开销迫使模型在空间分辨率与滚动预测时域之间做出权衡，这对于细尺度PDE问题尤为突出，因为其中的激波、反应前沿和材料界面仅占据求解域中较小且不断演化的区域。传统神经代理模型也缺乏动态调整分辨率的机制。我们提出了WAMRViT，这是一种视觉Transformer，它使用受小波启发的细化判据将输入标记化为平衡四叉树，通过3D旋转位置编码联合编码位置与细化层级，并在推理过程中在单元空间中进行重网格化，以实现稳定的长时程滚动预测。其多尺度变体在每个叶节点保留其原始源分辨率，使模型能够跨分辨率层级进行学习。与固定预算的自适应标记化方法不同，WAMRViT不预设标记数量，并在整个自回归滚动过程中支持完全自适应的拓扑结构。据我们所知，它是首个……（摘要在此处截断）

    arXiv:2610.04044v3 Announce Type: replace-cross  Abstract: The quadratic attention cost of Vision Transformers (ViTs) forces a trade-off between spatial resolution and rollout horizon, particularly for fine-scale PDEs where shocks, reaction fronts, and material interfaces occupy small, evolving regions of the domain. Conventional neural surrogates also lack mechanisms to adapt resolution dynamically. We propose WAMRViT, a ViT that tokenizes inputs as balanced quadtrees using a wavelet-inspired refinement criterion, jointly encodes position and refinement level with 3D rotary positional embeddings, and regrids in cell space during inference for stable long-horizon rollouts. A multi-scale variant retains each leaf at its native source resolution and lets the model learn across resolution levels. Unlike fixed-budget adaptive-tokenization methods, WAMRViT imposes no predetermined token count and supports fully adaptive topology throughout autoregressive rollout. To our knowledge, it is the
    
[^258]: S²-PINN：随机可分离物理信息神经网络

    S$^{2}$-PINN: Stochastic Separable Physics-Informed Neural Networks

    [https://arxiv.org/abs/2610.03303](https://arxiv.org/abs/2610.03303)

    提出S²-PINN，利用可学习高斯空间字典、傅里叶时间特征与gPC随机基并通过低秩CP张量分解耦合的可分离表示，高效求解随机偏微分方程的不确定性量化问题，克服了经典谱方法的维数灾难。

    

    随机偏微分方程（PDE）的不确定性量化（UQ）在计算科学与工程中无处不在。然而，求解这类问题的经典谱方法面临维数灾难，而现有的神经求解器往往忽略了随机结构，无法使矩估计与校准变得易于处理。我们提出了一种随机可分离物理信息神经网络，称为S²-PINN，它通过可学习的高斯空间字典、傅里叶时间特征以及广义多项式混沌（gPC）随机基来表示随机PDE的解 u(t,x,Z)，并通过低秩Canonical Polyadic（CP）张量分解核心将三者耦合起来。该方法采用强形式与gPC投影残差相结合的混合损失进行训练。我们的理论分析证明，在温和条件下，该可分离函数类在L²空间中是稠密的，且投影残差恰好对应于……

    arXiv:2610.03303v1 Announce Type: new  Abstract: Uncertainty quantification (UQ) for random partial differential equations (PDEs) is ubiquitous in computational science and engineering. However, classical spectral solvers for this class of problems face the curse of dimensionality, and existing neural solvers often ignore the stochastic structure that makes moments and calibration tractable. We introduce a stochastic separable physics-informed neural network, dubbed S$^{2}$-PINN, that represents the solution $u(t,\mathbf{x},\mathbf{Z})$ of a random PDE with a learnable Gaussian spatial dictionary, Fourier temporal features, and a generalized polynomial chaos (gPC) stochastic basis, coupled by a low-rank Canonical Polyadic (CP) tensor decomposition core. The method is trained with a hybrid strong-form and gPC-projected residual loss. Our theoretical analysis establishes that the separable class is dense in $L^2$ under mild conditions, and the projected residual corresponds exactly to a 
    
[^259]: PatchKV：KV缓存的权重空间补偿

    PatchKV: Weight-Space Compensation of KV Cache

    [https://arxiv.org/abs/2609.39329](https://arxiv.org/abs/2609.39329)

    PatchKV是一种无需训练的框架，通过岭回归以闭式解将上下文特定的权重补丁合并进模型权重，以补偿KV缓存压缩带来的信息损失，且不增加每次查询的推理成本。

    

    在大语言模型（LLM）的长上下文推理中，键值（KV）缓存线性增长的内存是主要瓶颈。现有的压缩方法通过令牌驱逐或近似来缩减缓存，但在激进的压缩预算下性能会急剧下降。我们提出PatchKV，这是一个无需训练的框架，通过将部分上下文信息承载到模型权重中来补偿KV缓存压缩方法。PatchKV将现成的压缩KV缓存与一个上下文特定的权重补丁配对，该权重补丁在上下文加载时仅计算一次，并可服务于该上下文的所有下游查询。权重补丁通过岭回归以闭式解推导得出，具体做法是对齐完整缓存与压缩缓存下、由上下文生成的参考查询令牌的逐块激活。补丁一旦合并到模型中，便不再改变前向计算图，且在单上下文、多查询的设置下每次查询的推理成本保持不变。

    arXiv:2609.39329v1 Announce Type: new  Abstract: Long-context inference with Large Language Models (LLMs) is bottlenecked by the linearly growing memory of the key-value (KV) cache. Existing compression methods reduce the cache through token eviction or approximation, but degrade sharply at aggressive compression budgets. We propose PatchKV, a training-free framework that compensates KV cache compression methods by carrying part of the context in the model's weights. PatchKV pairs an off-the-shelf compressed KV cache with a context-specific weight patch, which is computed once at context-loading time and served for downstream queries for the context. The weight patch is derived in closed form via ridge regression, by aligning the block-wise activations of context-derived reference query tokens under the full cache and the compressed cache. Once merged into the model, the patch leaves the forward graph and per-query inference cost unchanged in the single-context, multi-query setting. Ac
    
[^260]: 均匀竞赛：无参数的近似拒绝采样

    Uniform Race: Parameter-Free Approximate Rejection Sampling

    [https://arxiv.org/abs/2609.34639](https://arxiv.org/abs/2609.34639)

    该论文提出了一种无需任何参数调节的“均匀竞赛（Uniform Race）”近似拒绝采样算法，通过将重要性权重除以独立均匀随机变量构造得分并选取样本，在不知道理想接受阈值的情况下自动达到最优的拒绝采样全变差误差保证。

    

    考虑如下近似采样问题：给定来自提议分布 μ 的 N 个独立样本，目标是选出其中一个，使其分布接近目标分布 π，而 π 仅在相差一个归一化常数的意义下给定。Block 与 Polyanskiy（2023）的开创性工作给出了近似拒绝采样（RS）关于算法接受阈值 M 的全变差（TV）误差界。然而，使该界最小化的理想阈值 M 依赖于 (π, μ) 的性质，而这些性质通常无法从观测样本中获得。这引出一个自然的问题：能否在不以最优 M 作为输入的情况下获得最优的拒绝采样保证？我们通过提出一种称为 uniform race（UR）的无参数采样算法对此给出了肯定的回答。UR 为每个样本 y 计算一个得分，该得分是重要性权重 π(y)/μ(y) 除以一个独立的均匀随机变量，然后返回得分最……（摘要在此处截断）

    arXiv:2609.34639v2 Announce Type: replace  Abstract: Consider the approximate sampling problem: given $N$ independent samples from a proposal distribution $\mu$, the goal is to select one whose distribution is close to a target $\pi$ specified only up to a normalizing constant. The seminal work of Block & Polyanskiy (2023) provides a total variation (TV) error bound for approximate rejection sampling (RS) as a function of the algorithm's acceptance threshold $M$. The ideal threshold $M$ that minimizes the bound, however, depends on properties of $(\pi, \mu)$ that are typically unavailable from the observed sample. This raises a natural question: Can one attain the best RS guarantee without taking the best $M$ as input? We answer this question affirmatively by proposing a parameter-free sampling algorithm called uniform race (UR). UR computes a score for each sample $y$ that is the importance weight $\pi(y)/\mu(y)$ divided by an independent uniform random variable and then returns the o
    
[^261]: 理解泛化需要普适归纳

    Understanding Generalization Requires Universal Induction

    [https://arxiv.org/abs/2609.34458](https://arxiv.org/abs/2609.34458)

    论文指出经典统计理论因依赖无法证成的手工归纳偏置而无法解释通用AI的泛化成功，并提出将所罗门诺夫归纳相对化到信息视角、偏向能访问所有既有信息的短程序，从而以普适归纳重新框架化归纳偏置问题。

    

    经典统计理论不足以解释通用人工智能模型的成功，因为它依赖于无法自圆其说的手工设计的归纳偏置。没有免费午餐（No Free Lunch, NFL）定理表明，任何在某些环境中胜过随机的学习器，必然在其他环境中表现不佳。我们或许希望过去的经验能指明应当预期哪些环境，但NFL定理同样适用于元学习。因此，任何能做出有意义预测的方法，都必然以数据之外的归纳偏置为起点。若选择偏向短程序的归纳偏置，便会得到所罗门诺夫归纳（Solomonoff Induction, SI），其性能可与所有可计算的学习器相媲美——尽管其中存在的“常数”在与利用背景信息的专门化方法进行比较时会变得很大。因此，我们将SI相对化到一个信息视角上，偏向于能够访问所有既有信息的短程序。这一做法重新框架化了归纳偏置问题（摘要原文在此处截断）。

    arXiv:2609.34458v2 Announce Type: replace-cross  Abstract: Classical statistical theory is insufficient to explain the successes of general-purpose AI models, because it depends on handcrafted inductive biases that it cannot justify. No Free Lunch (NFL) theorems force any learner that beats chance on some environments to underperform on others. We might hope that past experience informs which environments to expect, but NFL applies equally to meta-learning. Thus, any method that makes meaningful predictions necessarily begins with an inductive bias external to the data. Choosing to bias toward short programs yields Solomonoff induction (SI), whose performance is competitive against all computable learners - albeit up to "constants" that become large when comparing against specialized methods that exploit background information. We therefore relativize SI to an information vantage point, biasing toward short programs with access to all preexisting information. This reframes the inductiv
    
[^262]: 从灰箱到绿箱：物理信息机器学习何时能减少结构健康监测中的碳足迹？

    From Grey-Box to Green-Box: When can Physics-Informed Machine Learning Reduce Carbon Footprints in Structural Health Monitoring?

    [https://arxiv.org/abs/2609.33387](https://arxiv.org/abs/2609.33387)

    本文探讨物理信息机器学习（灰箱模型）能否在结构健康监测中降低碳足迹，通过评估残差建模、输入增广、混合建模和约束学习四种方法，将训练数据量、训练时间与计算碳排放联系起来。

    

    机器学习在工程领域发挥着日益重要的作用，但随之而来的计算时间增加并非没有环境代价。物理信息机器学习或“灰箱”模型旨在克服传统黑箱学习器的部分局限性，其利用工程师对所建模结构的物理洞察，已在结构工程等众多领域展现出可喜的成果。本研究探讨了此类模型能否带来一项额外优势——降低环境影响，其中考虑了训练数据量与训练时间之间的关系，并将训练时长与计算产生的碳排放相关联。在结构健康监测的背景下，本文评估了四种跨越高斯过程和神经网络的物理信息机器学习方法：残差建模、输入增广、混合建模以及约束学习……

    arXiv:2609.33387v2 Announce Type: replace  Abstract: Machine learning plays an increasingly vital role in engineering, but the corresponding increase in compute time is not without environmental cost. Physics-informed machine learning or "grey-box" models have been developed to overcome some of the limitations of traditional black-box learners, utilising the physical insight that an engineer would have about the structure they are modelling and have shown promising results in the structural engineering field among many others. This work explores whether an additional advantage could be a reduced environmental impact, considering the relationship between training data quantity and training time, linking this duration to carbon emissions from computing.   In a structural health monitoring context, four physics-informed machine learning approaches - spanning Gaussian processes and neural networks - are evaluated: residual modelling, input augmentation, hybrid modelling, and constrained le
    
[^263]: 薛定谔–弗尔默 Actor–Critic：具有有限样本分析的扩散策略改进

    Schr\"odinger--F\"ollmer Actor--Critic: Diffusion Policy Improvement with Finite-Sample Analysis

    [https://arxiv.org/abs/2609.33239](https://arxiv.org/abs/2609.33239)

    该论文提出SFAC方法，利用极小极大贝尔曼评论家、Doob h-变换以及基于布朗桥端点的加权回归，在条件扩散框架中实现KL正则化的策略改进，并给出有限样本理论分析。

    

    扩散策略能够表示多模态的动作分布，但基于Q值加权的更新并未说明如何从所得的目标分布中进行采样。我们提出了薛定谔–弗尔默 Actor–Critic（SFAC），这是一种离线到在线的强化学习方法，通过条件扩散实现带KL（Kullback–Leibler）正则化的策略改进。极小极大贝尔曼评论家估计动作价值函数，该函数定义了一个指数倾斜的目标策略。Doob h-变换将该更新表示为对参考扩散漂移的修正。目标漂移是布朗桥标签的加权条件均值。我们通过加权回归对其进行拟合，使用来自当前策略的端点以及逐点的动作价值评估。在小更新情形下，KL正则化更新遵循自然策略梯度方向，而Doob修正表示扩散空间中相同的局部变化。

    arXiv:2609.33239v3 Announce Type: replace-cross  Abstract: Diffusion policies represent multimodal action distributions, but a $Q$-weighted update does not specify how to sample from the resulting target distribution. We propose Schr\"odinger--F\"ollmer Actor--Critic (SFAC), an offline-to-online reinforcement learning (RL) method for Kullback--Leibler (KL)-regularized policy improvement through conditional diffusion. A minimax Bellman critic estimates the action-value function, which defines an exponentially tilted target policy. A Doob $h$-transform expresses this update as a correction to the reference diffusion drift. The target drift is a weighted conditional mean of Brownian-bridge labels. We fit it by weighted regression using endpoints from the current policy and pointwise action-value evaluations. In the small-update regime, the KL-regularized update follows the natural policy-gradient direction, and the Doob correction represents the same local change in the space of diffusion
    
[^264]: 数据应该教什么？——在电路、存储与使用之间移动瓶颈

    What Should Data Teach? Moving Bottlenecks Across Circuit, Store, and Use

    [https://arxiv.org/abs/2609.32991](https://arxiv.org/abs/2609.32991)

    该论文提出以电路视角诊断语言模型训练中的三类数据瓶颈（计算形成、内容可用、路径选择），并给出统一的“诊断到数据”原则，通过形成敏感选择、可用性反事实与配对监督等针对性干预，显著提升了事实学习与长上下文问答能力。

    

    在训练的特定阶段，数据应该教会语言模型什么？电路视角揭示了三个截然不同的瓶颈：形成一次计算、使其所需内容变得可用、以及在可用路径之间做出选择。一个统一的“诊断到数据”原则将三者联系起来：定位缺失的操作、保持其因果关系、变换含有捷径的上下文，并对残差进行重新审计。对“形成”敏感的数据选择与先修知识排序加速了“绑定—匹配—传输”路径；来自同一训练数据多重集的简短早期前缀在1000亿token范围内始终保持验证优势。随后，可用性反事实区分了“写入内容”与“调用已有记忆”，而配对监督与“上下文—机会”排序改善了匹配路径的决策以及长上下文回答的可能性。一个持续的3.5亿参数模型实验将三项干预联系在同一批事实上：早期电路训练改进了子……

    arXiv:2609.32991v2 Announce Type: replace  Abstract: What should data teach a language model at a particular point in training? A circuit view reveals three distinct bottlenecks: forming a computation, making its required content available, and selecting among available routes. A shared diagnosis-to-data principle connects them: localize the missing operation, preserve its causal relation, vary shortcut-bearing context, and re-audit the residual. Formation-sensitive selection and prerequisite ordering accelerate a binding-matching-transport path; a brief early prefix from the same training multiset retains a validation advantage through 100B tokens. Availability counterfactuals then distinguish writing content from invoking available memory, while paired supervision and context-opportunity ranking improve matched route decisions and long-context answer likelihood. A continuous 350M-model experiment connects the three interventions on the same facts: early circuit training improves subs
    
[^265]: 鲁棒后继特征

    Robust Successor Features

    [https://arxiv.org/abs/2609.31016](https://arxiv.org/abs/2609.31016)

    该论文提出了鲁棒后继特征，将强化学习中的迁移学习与鲁棒强化学习两种范式统一起来，使智能体在线性马尔可夫决策过程假设下，能够同时在奖励函数和未知转移核两个维度上进行泛化。

    

    摘要：强化学习（RL）中的泛化是指智能体在一系列不同任务上训练后，能够在未见过的任务上执行接近最优策略的能力。基于后继表示的开创性工作以及结合函数逼近的进一步改进，强化学习中的迁移学习传统上一直专注于泛化到仅在奖励函数上有所不同的任务。在后继表示提出十年之后，鲁棒强化学习同时从运筹学领域的数篇文章中兴起。在鲁棒强化学习中，转移核是未知的，目标是在这种不确定性下最大化期望奖励。我们的工作通过鲁棒后继特征统一了这两种范式，在任务为线性马尔可夫决策过程的假设下，鲁棒后继特征能够在奖励函数和转移核两个维度上实现泛化。我们推导了广义策略改进的一个界……

    arXiv:2609.31016v1 Announce Type: new  Abstract: Generalization in Reinforcement Learning (RL) refers to the ability to execute close-to-optimal policies in unseen tasks after the agent has been trained on a different set of tasks. Building on the seminal work of the successor representation and further adaptations with function approximation, Transfer in RL has traditionally focused on generalizing to tasks that only differ in the reward function. A decade after the introduction of the successor representation, Robust RL emerged simultaneously from several articles in the field of operations research. In Robust RL, the transition kernel is unknown, and the goal is to maximize the expected reward under this uncertainty. Our work unifies these two paradigms through robust successor features, which generalize across both the reward function and the transition kernel, under the assumption that tasks are linear Markov Decision Processes. We derive a bound on Generalized Policy Improvement 
    
[^266]: 面向无穷大Laplace问题与p-Laplace问题的深度学习求解器与代理模型

    Deep-Learning Solvers and Surrogates for Infinity and p-Laplace Problems

    [https://arxiv.org/abs/2609.30809](https://arxiv.org/abs/2609.30809)

    本文利用PINN和DeepONet求解无穷大Laplace与p-Laplace问题，在大p值（2至1000）和三维区域上优于传统网格求解器，并建立了条件收敛性与通用逼近的理论结果。

    

    我们研究了神经网络求解器在无穷大Laplace问题和p-Laplace问题中的应用，这些问题在非线性分析中具有基础性地位，并具有实际应用价值。我们的方法采用物理信息神经网络（PINNs）和深度算子网络来应对大p值（范围从2到1000）在各种二维和三维区域上带来的计算挑战。与传统的基于物理的求解器相比，我们的方法具有优势，尤其是在三维情况下，基于网格的求解器对此类问题的计算成本非常高。我们还为这两个问题的PINN近似建立了条件收敛性结果，并为DeepONet在参数化p-Poisson问题上的应用建立了通用逼近定理。我们通过数值实验验证了这些神经网络求解器的有效性，并将其性能与传统方法进行了比较。

    arXiv:2609.30809v1 Announce Type: cross  Abstract: We investigate the use of neural network solvers for infinity and $p$-Laplace problems, which are fundamental in nonlinear analysis and have practical applications. Our approach employs Physics-Informed Neural Networks (PINNs) and Deep Operator Networks (DeepONets) to address computational challenges associated with large $p$ values, ranging from $2$ to $1000$, on various 2D and 3D domains. Our method offers advantages over traditional physics-based solvers, especially in three dimensions where mesh-based solvers become very costly for these problems. We also establish conditional convergence results for PINN approximations of both problems and a universal approximation result for DeepONet on the parametric $p$-Poisson problem. We demonstrate the effectiveness of these neural network solvers through numerical experiments and compare their performance with conventional methods.
    
[^267]: 解耦知识与隐私：面向大语言模型持续学习的任务后自蒸馏重放

    Decoupling Knowledge and Privacy: Post-Task Self-Distillation Replay for LLM Continual Learning

    [https://arxiv.org/abs/2609.29711](https://arxiv.org/abs/2609.29711)

    该论文提出SPARK方法，通过将知识保留与隐私修正解耦——先冻结任务后学到的分布作为稳定参考，再围绕其进行针对稀疏敏感位置的选择性修正——从而在大语言模型持续学习中兼顾知识保留与隐私保护。

    

    隐私保护的持续学习（PPCL）必须在跨序列任务保留有用知识的同时，减少敏感内容的再现。形式化隐私保证刻画的是随机化机制，而操作性输出控制关注的则是训练后的模型能否选择性地降低其输出中敏感内容出现的可能性。在这项工作中，我们在现实的任务演化情境下，将后者与持续学习效用放在一起进行研究。知识保留与隐私修正作用于不同的粒度：任务获取需要广泛保留当前任务和旧任务的行为，而隐私修正则针对稀疏的标注位置。联合优化会导致当前任务的保留目标不断变化。我们提出了SPARK，一种保留-修正分解方法，它首先冻结学习到的任务后分布，然后围绕这一稳定参考进行选择性修正。

    arXiv:2609.29711v1 Announce Type: cross  Abstract: Privacy-preserving continual learning (PPCL) must reduce the reproduction of sensitive content while retaining useful knowledge across sequential tasks. Formal privacy guarantees characterize randomized mechanisms, whereas operational output control concerns whether a trained model selectively reduces the likelihood of sensitive content in its outputs. In this work, we investigate the latter together with continual-learning utility under realistic task evolution. Retention and privacy correction operate at different granularities: task acquisition requires broad preservation of current- and old-task behavior, whereas privacy correction targets sparse annotated positions. Joint optimization leaves the current-task preservation target continually changing. We propose SPARK, a retention-correction decomposition that first freezes the learned post-task distribution and then applies selective correction around this stable reference. Self-Di
    
[^268]: MMAP：面向纵向阿尔茨海默病预测的多模态缺失感知预训练

    MMAP: Multimodal Missing-Aware Pretraining for Longitudinal Alzheimer's Prediction

    [https://arxiv.org/abs/2609.26617](https://arxiv.org/abs/2609.26617)

    提出了MMAP方法，一种多模态缺失感知的对齐预训练框架，通过sigmoid对比学习与生成式重建从不完整的图像和表格数据中学习联合表示，用于阿尔茨海默病的纵向进展预测。

    

    临床决策在很大程度上依赖于通过理解患者的健康状况来预测疾病进展轨迹，而患者的健康状况由多模态医疗数据所刻画。人工智能在从多模态医疗数据中学习有用的表示以预测疾病进展并辅助临床决策方面具有巨大潜力。然而，预测型AI模型的开发受到医疗数据集中经常出现的模态缺失和不完整表格数据的限制。此外，仅凭疾病标签可能只能为从高维多模态数据中学习表示提供有限的监督信号。在本文中，我们提出了MMAP，一种新颖的多模态缺失感知对齐预训练方法，用于从不完整数据中学习图像-表格联合表示。其中，图像编码器通过高效的sigmoid对比学习结合生成式重建进行预训练，表格编码器则构建于表格数据基础之上。

    arXiv:2609.26617v1 Announce Type: cross  Abstract: Clinical decision making heavily relies on predicting the disease progression trajectory by seeking to understand patient's health status which is characterised by multimodal medical data. AI holds great potential for learning useful representations from multimodal medical data to predict disease progression and aid clinical decision making. However, development of predictive AI models is constrained by missing modalities and incomplete tabular data frequently occurring in medical datasets. In addition, disease labels alone may only provide limited supervisory signals for learning representations from high-dimensional multimodal data. Here, we present MMAP, a novel Multimodal Missing-aware Alignment Pretraining method for learning image-tabular representations from incomplete data. An image encoder is pretrained with efficient sigmoid contrastive learning combined with generative reconstruction. A tabular encoder is built upon a tabula
    
[^269]: 信息论极限下的高效鲁棒学习

    Efficient Robust Learning at the Information-Theoretic Limit

    [https://arxiv.org/abs/2609.17655](https://arxiv.org/abs/2609.17655)

    本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。

    

    在近期一项重要工作中，Blanc（2026）给出了一种针对固定分布鲁棒学习布尔概念类的算法，该算法输出一个（随机化的）分类器，能够达到 η + ε 的最优错误率，其中 η 为噪声率。相比之下，众所周知，确定性假设无法达到低于 2η + ε 的错误率。Blanc 的算法在计算上效率低下，其工作中遗留的主要开放问题是：在可访问经验风险最小化（ERM）预言机的条件下，找到一种多项式时间算法。在本文中，我们解决了这一问题，并给出了这样一种算法。令人惊讶的是，我们的技术关键地利用了各种类型的无悔学习器。此外，我们给出了一种高效算法（无需 ERM 预言机），用于鲁棒学习任何在超收缩分布下允许三明治多项式的函数类。作为其中一个结果……

    arXiv:2609.17655v1 Announce Type: cross  Abstract: In an important recent work, Blanc (2026) gave an algorithm for robustly learning Boolean concept classes with respect to a fixed distribution that outputs a (randomized) classifier achieving the optimal error of $\eta + \varepsilon$ where $\eta$ is the noise rate. In contrast, it is well known that deterministic hypotheses cannot achieve error less than $2\eta + \varepsilon.$   Blanc's algorithm is computationally inefficient, and the main problem left open in his work is to find a polynomial-time algorithm given access to an oracle for empirical risk minimization (ERM). In this paper, we resolve this problem and give such an algorithm. Perhaps surprisingly, our techniques make crucial use of various types of no-regret learners.   Additionally, we give an efficient algorithm (no ERM oracle required) for robustly learning any function class that admits sandwiching polynomials with respect to hypercontractive distributions. As one conse
    
[^270]: 基于对抗性重要性采样的鲁棒策略优化

    Robust Policy Optimization via Adversarial Importance Sampling

    [https://arxiv.org/abs/2609.13044](https://arxiv.org/abs/2609.13044)

    该论文提出了Advis方法，通过对标准训练轨迹进行重要性采样来估计和优化可验证的最坏情况回报，无需额外环境交互和辅助网络即可捕获长期鲁棒性，同时提供模块化PyTorch库advrl以促进鲁棒DRL方法的快速开发与可复现评估。

    

    在保护深度强化学习（DRL）策略免受输入扰动方面已经取得了显著进展。开发鲁棒的DRL涉及三个主要阶段：算法设计、实现和评估。在这项工作中，我们识别并解决了每个阶段的一个关键局限。首先，我们提出了对抗性重要性采样，这是一种利用标准训练产生的轨迹上的重要性采样来估计和优化可验证的最坏情况回报的方法。Advis满足了三个先前工作未能同时实现的理想标准：它不需要额外的环境交互，不需要辅助网络，并且能够捕获长期鲁棒性。其次，我们介绍了advrl，这是一个模块化的PyTorch库，为现有的鲁棒性方法和对抗攻击提供了干净的单文件实现，便于快速原型开发，并实现可复现和可追溯的评估。第三，我们重新审视了评估……

    arXiv:2609.13044v1 Announce Type: new  Abstract: Significant progress has been made in safeguarding deep reinforcement learning (DRL) policies against input perturbations. Developing robust DRL involves three main stages: algorithm design, implementation, and evaluation. In this work, we identify and address a key limitation at each stage. First, we introduce Adversarial Importance Sampling (Advis), a method that uses importance sampling over trajectories from standard training to estimate and optimize verifiable worst-case returns. Advis satisfies three desirable criteria not jointly achieved by prior work: it requires no additional environment interactions, no auxiliary networks, and captures long-term robustness. Second, we introduce advrl, a modular PyTorch library that provides clean, single-file implementations of existing robustness methods and adversarial attacks, facilitating rapid prototyping and enabling reproducible and traceable evaluations. Third, we revisit evaluation un
    
[^271]: 面向大规模长上下文强化学习后训练中投机解码的在线草稿模型协同训练

    Online Draft Co-Training for Speculative Decoding in Large-Scale, Long-Context RL Post-Training

    [https://arxiv.org/abs/2609.07108](https://arxiv.org/abs/2609.07108)

    该论文提出了一个端到端系统，通过在上下文并行中融合分支注意力与主序列注意力、并引入TapChannel独立通道跨流水线阶段传输目标特征，实现了大规模长上下文强化学习后训练中投机解码草稿模型的在线协同训练，从而显著加速rollout生成。

    

    投机解码可以加速rollout生成，而rollout生成在强化学习（RL）后训练的成本中占据主导地位。在线协同训练可以进一步提高草稿模型的准确率，从而获得更大的加速效果。然而，将该方法扩展到长上下文大模型的协同训练面临两个障碍：（1）标准的因果上下文并行（CP）实现不支持分支注意力；（2）目标模型的特征跨越流水线并行（PP）的各个阶段。我们通过一个面向大规模在线草稿协同训练的端到端系统来解决这两个问题。对于CP，我们将rank局部的分支注意力与因果主序列注意力相融合，扩展了打包且负载均衡的锯齿形环形注意力。对于PP，TapChannel通过独立路径跨阶段传输中间目标特征，且不影响流水线调度。实验表明，协同训练的草稿模型能紧密跟随策略基线，同时带来显著的rollout加速。

    arXiv:2609.07108v1 Announce Type: new  Abstract: Speculative decoding accelerates rollout generation, which dominates the cost of reinforcement learning (RL) post-training. Online co-training can further increase the draft's accuracy, yielding greater speedups. However, scaling this approach to co-training on large models with long contexts poses two obstacles: (1) branch attention is unsupported by standard causal context-parallel (CP) implementations, and (2) target features span across pipeline-parallel (PP) stages. We address both with an end-to-end system for large-scale online draft co-training. For CP, we extend packed, load-balanced zigzag ring attention by merging rank-local branch attention with causal main-sequence attention. For PP, TapChannel transports intermediate target features across stages via a separate path, leaving the pipeline schedule unaffected. Experiments demonstrate that co-trained drafts closely track the policy baseline while delivering substantial rollout
    
[^272]: LongCounsel-8：基于多轮次心理咨询对话的纵向抑郁症追踪基准套件

    LongCounsel-8: A Benchmark Suite for Longitudinal Depression Tracking from Multi-Session Counseling Dialogues

    [https://arxiv.org/abs/2609.03507](https://arxiv.org/abs/2609.03507)

    提出了LongCounsel-8基准套件，包含三个数据集共计7,749条五轮次心理咨询对话轨迹，填补了多轮次纵向抑郁症追踪研究中标准化会话级标注数据的空白。

    

    从多轮次心理咨询对话中追踪抑郁症，需要同时估计当前的症状严重程度以及其在不同会话之间的变化情况。然而，该任务的进展受限于缺乏具有标准化会话级抑郁标签的纵向咨询数据。现有资源通常要么提供没有抑郁标签的多轮次对话，要么仅提供单次会话的标注访谈。构建这样一个基准面临三大挑战：保持纵向的一致性与多样性，将症状进展建立在实证模式之上，以及在不暴露目标标签的前提下自然地表达受控的抑郁状态。为应对这些挑战，我们提出了LongCounsel-8，这是一个由三个独立生成的数据集组成的基准套件，共包含7,749条五轮次咨询轨迹，其建立在真实的来访者档案、抑郁轨迹、症状构成和咨询模式的基础之上。

    arXiv:2609.03507v1 Announce Type: cross  Abstract: Tracking depression from multi-session counseling dialogues requires estimating both current symptom severity and how it changes across sessions. Yet progress on this task is constrained by the scarcity of longitudinal counseling data with standardized session-level depression labels. Existing resources typically provide either multi-session conversations without depression labels or labeled interviews in a single session. Building such a benchmark poses three challenges: maintaining longitudinal consistency and diversity, grounding symptom progression in empirical patterns, and expressing controlled depression states naturally without exposing target labels. To address these challenges, we introduce LongCounsel-8, a benchmark suite of three independently generated datasets totaling 7,749 five-session counseling trajectories, grounded in real-world client profiles, depression trajectories, symptom compositions, and counseling patterns.
    
[^273]: 一种用于学习路径规划（路由）的混合量子-经典神经网络

    A hybrid quantum-classical neural network for learning to route

    [https://arxiv.org/abs/2609.00489](https://arxiv.org/abs/2609.00489)

    本研究提出用小型量子神经网络替换基于注意力的路由模型中的编码器前馈模块，在带容量约束车辆路径问题上将参数减少56.6%的同时保持接近经典神经基线的求解质量，为神经组合优化提供了一种可行的混合模块压缩策略。

    

    本工作研究用于学习路由启发式算法的混合量子-经典神经网络。具体而言，本文探讨小型量子神经网络能否在保持解质量的前提下，替换有竞争力的基于注意力的路由模型中的参数密集型模块。对于带容量约束的车辆路径问题（CVRP），编码器前馈替换被证明是最有前景的设计：它将模型参数数量减少了56.6%，同时使混合模型在中小规模实例上接近经典神经基线，尽管在更大规模实例上差距会扩大。本工作还与经典路由算法进行了比较，后者在固定欧几里得测试集上仍保持高度竞争力且常常表现更优。因此，我们的结果并未表明量子优势或求解器主导地位，而是确定编码器前馈替换作为神经组合优化中一种可行的混合模块压缩策略。

    arXiv:2609.00489v1 Announce Type: new  Abstract: This work studies hybrid quantum-classical neural networks for learning routing heuristics. Specifically, this paper asks whether small quantum neural networks can replace parameter-heavy modules inside a competitive attention-based routing model while maintaining solution quality. For the capacitated vehicle routing problem, encoder feed-forward replacement emerges as the most promising design: it reduces the number of model parameters by 56.6% while keeping the hybrid model close to the classical neural baseline at small and medium instance sizes, although the gap grows for larger instances. This work also compares to classical routing algorithms, which remain highly competitive and often superior on the fixed Euclidean test sets. Our results therefore do not indicate quantum advantage or solver dominance, but identify encoder feed-forward replacement as a viable hybrid-module compression strategy for neural combinatorial optimization.
    
[^274]: 多类学习算法原理难以实现：正则化与恰当学习的局限性

    Algorithmic Principles For Multiclass Learning Are Hard To Come By: Limits of Regularization and Proper Learning

    [https://arxiv.org/abs/2608.26516](https://arxiv.org/abs/2608.26516)

    本文证明多类学习不能简化为恰当学习或正则化，解决了三个开放问题，并展示了一个不可嵌入任何恰当框架的可学习多类问题。

    

    arXiv:2608.26516v1 公告类型：新 摘要：统计学习理论中两个最基本的问题是：哪些预测问题可学习，以及应如何学习它们？对于前者，优雅的答案通常以组合维度的形式呈现。然而，后者已被证明远为难以捉摸：所有已知的通用多类学习器都依赖于指数级大型一包含结构的复杂定向，而熟悉的学习算法原则，如恰当学习和正则化，仍未被充分理解。受先前工作启发，我们探讨学习是否可简化为恰当学习——可能在一个更大的假设类上——以及恰当或不恰当的多类学习是否最终能被合适的正则化器所捕捉。我们的主要结果对这两个问题均给出否定答案，解决了先前工作中的三个开放问题。首先，我们展示了一个可学习的多类问题，它无法嵌入到任何恰当学习框架中。

    arXiv:2608.26516v1 Announce Type: new  Abstract: Two of the most fundamental questions in statistical learning theory are the following: which prediction problems are learnable, and how should they be learned? For the former, elegant answers often take the form of combinatorial dimensions. The latter question, however, has proved considerably more elusive: all known general-purpose multiclass learners rely on intricate orientations of exponentially large one-inclusion structures, and familiar algorithmic principles such as proper learning and regularization remain poorly understood. Motivated by prior work, we ask whether learning reduces to proper learning---possibly over a larger hypothesis class---and whether proper or improper multiclass learning can ultimately be captured by suitable regularizers.   Our primary results answer both questions negatively, resolving three open problems from prior work. First, we exhibit a learnable multiclass problem that cannot be embedded in any pro
    
[^275]: 连续对抗平均流迁移

    Continuous Adversarial MeanFlow Transfer

    [https://arxiv.org/abs/2608.19540](https://arxiv.org/abs/2608.19540)

    该论文提出MeanFlow-Transfer方法，通过统一速度表示和对抗训练，同时解决跨领域适应和生成器加速问题，使异构预训练模型能高效迁移到新领域。

    

    摘要：在有限数据下，针对新领域训练快速生成器仍面临两个挑战。首先，将预训练的扩散或流模型适应到新领域时，其昂贵的多步采样问题未得到解决，且现有加速方法与源参数化（如ε、x、v或u）绑定，导致异构预训练模型缺乏统一的加速目标。其次，虽然对抗性精炼已被证明对少步质量有效，但其仅针对瞬时速度流设计，不适用于MeanFlow（MF）模型预测的有限区间平均速度。我们解决了这两个问题。我们提出MeanFlow-Transfer，该方法将异构源输出映射到共享速度表示，利用源权重初始化MF生成器，并在目标域上优化MF目标。这统一了适应和加速过程，在广泛的模型范围内实现单一训练循环。

    arXiv:2608.19540v1 Announce Type: new  Abstract: Training fast generators on new domains with limited data remains challenging for two reasons. First, adapting a pretrained diffusion or flow model to a new domain leaves its costly multi-step sampling unaddressed, and existing acceleration methods are tied to the source parameterization--$\epsilon$, $x$, $v$, or $u$--leaving heterogeneous pretrained models with no common acceleration target. Second, while adversarial refinement is proven effective for few-step quality, it is formulated only for instantaneous-velocity flows, not for the finite-interval average velocities that MeanFlow (MF) models predict. We address both problems. We propose MeanFlow-Transfer, which maps heterogeneous source outputs into a shared velocity representation, uses it to initialize an MF generator from the source weights, and optimizes an MF objective on the target domain. This unifies adaptation and acceleration in a single training loop across a broad range 
    
[^276]: 基于Epiplexity引导的数据选择与生成用于分布外泛化

    Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization

    [https://arxiv.org/abs/2608.11746](https://arxiv.org/abs/2608.11746)

    本文提出利用Epiplexity作为在线训练信号，通过拟合缩放定律预测数据域的Epiplexity增益，自适应调整采样权重，并用于指导合成数据生成，以提升模型在未预见任务上的分布外泛化能力。

    

    现代系统越来越期望能够在训练时未指定的任务间进行迁移。什么数据能促进这些新的、未预见情境下的泛化？一个假设是，具有更多结构信息的数据可能包含共享电路和子程序，这些可以在更广泛的下游设置中被重用。Epiplexity，一种最近提出的度量，用于衡量计算受限学习器能从数据中提取的结构信息，提供了推理这种关系的机制。在本文中，我们展示了如何将Epiplexity作为在线训练信号，用于数据选择和合成数据生成。对于选择，我们拟合自然数据域的训练损失曲线的缩放定律，以预测作为训练令牌函数的预期Epiplexity增益，并利用该信号在训练过程中自适应地确定各域的采样权重。对于合成数据生成，我们定义了...

    arXiv:2608.11746v1 Announce Type: cross  Abstract: Modern systems are increasingly expected to transfer across tasks not specified during training. What data facilitates generalization in these new, unanticipated settings? One hypothesis is that data with more structural information could contain shared circuits and subprograms that could be recycled in a wider array of downstream settings. Epiplexity, a recently proposed measure of the structural information a compute-bounded learner can extract from data, provides a mechanism to reason about this relationship. In this paper, we show how to operationalize epiplexity as an online training signal for data selection and synthetic data generation. For selection, we fit scaling laws to the training loss curves of natural data domains to predict the expected epiplexity gain as a function of training tokens, and use this signal to adaptively determine the sampling weights over domains during training. For synthetic data generation, we define
    
[^277]: 表格基础模型与自身一致吗？

    Do Tabular Foundation Models Agree with Themselves?

    [https://arxiv.org/abs/2608.06004](https://arxiv.org/abs/2608.06004)

    本文首次系统评估表格基础模型通过自回归采样构建多变量预测的机制能否诱导出有效的随机过程，提出了边缘化一致性和因子分解一致性这两个可检验的要求，以回答“以两种不同方式提问同一问题是否会得到相同答案”这一核心问题。

    

    表格基础模型是许多表格预测任务的最先进方法。它们是在预训练先验下近似贝叶斯后验预测分布的Transformer模型。表格基础模型通过自回归方式成为多变量预测器，即采样一个目标变量并将其附加到特征中。本文首次评估这种设置是否会诱导出一个有效的随机过程。通俗地说，我们要问：*对以两种不同方式提出的同一个问题，我们会得到相同的答案吗？*标准的Kolmogorov一致性条件无法直接在表格基础模型上检验，因为表格基础模型无法对某些条件分布建模。因此，我们提出了两个经过调整的、可检验的要求：边缘化一致性要求边缘化后的条件分布等于直接预测的边缘分布；因子分解一致性要求两种因子分解顺序产生相同的联合分布。由于数据生成分布满足（原文摘要在此截断）

    arXiv:2608.06004v2 Announce Type: replace  Abstract: Tabular Foundation Models (TFMs) are the state of the art for many tabular prediction tasks. They are transformers that approximate the Bayesian posterior predictive distribution under a pretraining prior. TFMs become multivariate predictors autoregressively, by sampling one target and appending it to the features. This paper is the first to evaluate whether this setup induces a valid stochastic process. Informally, we ask: *Do we get the same answer to the same question asked in two different ways?* The standard Kolmogorov consistency conditions cannot be checked on TFMs directly, since TFMs are unable to model certain conditional distributions. We therefore propose two adapted, testable requirements. Marginalization consistency demands that marginalized conditionals equal directly predicted marginals, and factorization consistency that both factorization orders yield the same joint. Since the data-generating distribution satisfies 
    
[^278]: EvolveNet：面向智能体自我提升的协作式执行框架演化

    EvolveNet: Collaborative Harness Evolution for Agent Self-Improvement

    [https://arxiv.org/abs/2608.04968](https://arxiv.org/abs/2608.04968)

    提出 EvolveNet 框架，通过客户端并行演化同一执行框架、服务器聚合各客户端适配并重新分发的协作机制，在不更新模型权重的情况下实现智能体的持续自我改进与跨用户能力积累。

    

    大语言模型智能体可以通过演化其执行框架，即管理模型调用、工具使用、验证与恢复的可执行程序——而无需更新模型权重，从而获得持久的改进。当同一执行框架服务于具有不同工作负载的用户时，孤立演化会限制有用发现的共享，而集中式适配则可能引入相互冲突的任务特定修改。我们提出了 EvolveNet，一个将本地特化与共享能力积累相连接的协作式执行框架演化框架。客户端在各自的本地工作负载上并行演化同一广播的执行框架，服务器则将它们的适配组合成更新后的共享程序。重新分发使被接受的发现对所有客户端可用，使后续的本地演化能够建立在跨协作积累的改进之上。EvolveNet 通过证据引导的、按作用域分类的程序聚合以及……（摘要在此处截断）

    arXiv:2608.04968v3 Announce Type: replace  Abstract: An LLM agent can acquire persistent improvements by evolving its harness---the executable program governing model calls, tool use, verification, and recovery---without updating model weights. When the same harness serves users with different workloads, isolated evolution limits the sharing of useful discoveries, while centralized adaptation can introduce conflicting task-specific changes. We introduce EvolveNet, a framework for collaborative harness evolution that connects local specialization with shared capability accumulation. Clients evolve the same broadcast harness on their local workloads in parallel, and a server composes their adaptations into an updated shared program. Redistribution makes accepted discoveries available to all clients, allowing subsequent local evolution to build on improvements accumulated across the collaboration. EvolveNet supports this process through evidence-guided, scope-typed program aggregation and
    
[^279]: 基于CNN-BiLSTM模型的可穿戴传感器久坐行为分类

    Sedentary Behavior Classification for Wearable Sensors with a CNN-BiLSTM Model

    [https://arxiv.org/abs/2608.02946](https://arxiv.org/abs/2608.02946)

    提出通过对CNN-BiLSTM模型CHAP进行微调，将基于髋部加速度计训练的久坐行为分类模型成功迁移到手腕可穿戴传感器数据上，并持续优于从头训练的Transformer模型。

    

    准确检测久坐行为对于研究久坐相关的健康风险非常重要，但基于姿态的分类在使用可穿戴传感器时仍然具有挑战性，尤其是在手腕部位。我们研究在髋部佩戴加速度计数据上训练的深度学习模型能否迁移到手腕佩戴加速度计数据，用于坐姿与非坐姿分类。我们使用CHAP——一个最初为髋部加速度计开发的CNN-BiLSTM模型，评估其在手腕数据上的零样本性能，以及通过不同数量的带标注手腕数据进行微调后的适应能力。实验在iWatch数据集上进行，其中的真实姿态标签来自可穿戴摄像头。结果表明，在髋部数据上训练的模型无需重新训练即可在髋部数据上表现良好，但由于传感器佩戴位置的变化，其在手腕数据上的准确率有所下降。通过微调，CHAP相比从头训练的Transformer模型表现出持续的优势。

    arXiv:2608.02946v2 Announce Type: replace  Abstract: Accurate detection of sedentary behavior is important for studying health risks related to prolonged sitting, but posture-based classification remains challenging with wearable sensors, especially at the wrist. We study whether a deep learning model trained on hip-worn accelerometer data can transfer to wrist-worn accelerometer data for sitting versus non-sitting classification. We use CHAP, a CNN-BiLSTM model originally developed for hip accelerometers, and evaluate its zero-shot performance on wrist data as well as its adaptation through finetuning with varying amounts of labeled wrist data. Experiments are conducted on the iWatch dataset with ground-truth posture labels derived from wearable cameras. The hip-trained model performs strongly on hip data without retraining, but accuracy drops on wrist data due to sensor placement shift. Finetuning CHAP provides consistent advantages over transformer models trained from scratch. These
    
[^280]: DP-IVON-Gradsq：差分隐私平方梯度改进变分在线牛顿法

    DP-IVON-Gradsq: Differentially Private Squared-Gradient Improved Variational Online Newton

    [https://arxiv.org/abs/2607.23649](https://arxiv.org/abs/2607.23649)

    提出差分隐私优化器DP-IVON-Gradsq，通过噪声校正的平方梯度估计器从私有化梯度构建曲率估计，在保持IVON计算效率的同时减少了后验采样噪声与隐私噪声的相互干扰，实现了差分隐私下的变分贝叶斯深度学习。

    

    差分隐私为在敏感数据上训练神经网络提供了正式的隐私保证，而贝叶斯深度学习则为不确定性感知的预测提供了一个有原则的框架。将这两个目标结合起来仍然具有挑战性，因为隐私噪声可能会与贝叶斯后验采样所引入的随机性相互作用。在这项工作中，我们通过改进变分在线牛顿法（IVON）优化器来研究差分隐私下的变分贝叶斯学习。我们提出了DP-IVON-Gradsq，这是IVON的一个隐私保护变体。所提出的方法使用噪声校正的平方梯度估计器，从私有化后的梯度构建曲率估计，在保持IVON类Adam计算效率的同时，减少了后验采样噪声与隐私噪声之间的直接相互作用。我们在CIFAR-10数据集上，在一系列隐私预算设置下，将DP-IVON-Gradsq与标准隐私优化器DP-SGD和DP-Adam进行了评估。

    arXiv:2607.23649v2 Announce Type: replace  Abstract: Differential privacy provides formal privacy guarantees for training neural networks on sensitive data, while Bayesian deep learning offers a principled framework for uncertainty-aware prediction. Combining these two objectives remains challenging, as privacy noise can interact with the stochasticity introduced by Bayesian posterior sampling. In this work, we investigate differentially private variational Bayesian learning through the Improved Variational Online Newton (IVON) optimizer. We introduce DP-IVON-Gradsq, a private variant of IVON. The proposed method constructs its curvature estimate from the privatized gradient using a noise-corrected squared-gradient estimator, reducing the direct interaction between posterior-sampling noise and privacy noise while preserving the Adam-like computational efficiency of IVON. We evaluate DP-IVON-Gradsq on CIFAR-10 against the standard private optimizers DP-SGD and DP-Adam over a range of pr
    
[^281]: 面向公平自监督学习的无偏开放世界正则化

    Unbiased Open World Regularization for Fair Self-Supervised Learning

    [https://arxiv.org/abs/2607.22149](https://arxiv.org/abs/2607.22149)

    该论文提出无偏开放世界（UOW）正则化方法，通过将每个偏差组内的表示分布匹配到共同目标分布来保证表示与偏差属性的独立性，从而为自监督学习实现更有效的去偏与公平性。

    

    尽管近年来取得了诸多进展，自监督学习（SSL）模型和联合嵌入预测架构（JEPAs）仍然容易受到虚假相关性的影响。许多防塌缩正则化器将全局表示分布约束为匹配一个预设的目标分布，例如多元高斯分布或球面上的均匀分布。然而，这种全局匹配并不能阻止由某个偏差属性定义的群体在表示空间中占据不同的区域。我们证明了现有的仅编码器去偏正则化器（包括 EnD 和 FSCL）都可以被解释为条件分布匹配的受限形式。基于这一解释，我们提出了无偏开放世界，它将每个已观测偏差组内的表示分布匹配到一个共同的目标分布。在总体最优解处，使用判别式条件差异度量可以保证表示与偏差属性之间的独立性（原文摘要在此处截断）。

    arXiv:2607.22149v2 Announce Type: replace  Abstract: Despite recent advances, self-supervised learning (SSL) models and Joint-Embedding Predictive Architectures (JEPAs) remain susceptible to spurious correlations. Many anti-collapse regularizers constrain the global representation distribution to match a prescribed target, such as a multivariate Gaussian or the uniform distribution on the sphere. However, global matching does not prevent groups defined by a bias attribute from occupying distinct regions of the representation space. We show that existing encoder-only debiasing regularizers, including EnD and FSCL, can be interpreted as restricted forms of conditional distribution matching. Building on this interpretation, we introduce Unbiased Open World (UOW), which matches the representation distribution within each observed bias group to a common target distribution. At the population optimum, using a discriminative conditional discrepancy guarantees independence between the represen
    
[^282]: GaugeQuant：从大语言模型对称性中在线学习量化最优基

    GaugeQuant: Online Learning of Quantization-Optimal Bases from LLM Symmetries

    [https://arxiv.org/abs/2607.20757](https://arxiv.org/abs/2607.20757)

    GaugeQuant通过在训练中向损失函数引入LogSumExp项打破Transformer的内部连续对称性，在线学习出最小化激活离群值的量化最优基，无需校准数据和量化模拟即可显著提升低比特量化效果。

    

    众所周知，Transformer具有内部连续对称性，这种对称性在保持输出不变的同时会改变激活值在各坐标维度上的分布方式，从而影响其量化效果。GaugeQuant在训练过程中利用了这一特性，通过在损失函数中引入一个LogSumExp项来打破这种对称性，从而选择出使激活离群值最小化的基。梯度停止（stop-gradient）算子确保只有旋转矩阵被更新，使语言建模目标完全不受影响。我们的量化策略不需要特定的校准数据，不需要量化模拟，且仅增加很小的推理开销。作为概念验证，我们在WikiText-2困惑度和零样本推理任务上进行了评估。在Qwen-2.5 0.5B上，GaugeQuant降低了困惑度（例如在W4A16设置下从25.62降至16.76）并提高了零样本准确率。在LLaMA-2 7B上，困惑度有所改善，但下游准确率没有提升。代码可在以下网址获取：https://github.

    arXiv:2607.20757v3 Announce Type: replace-cross  Abstract: Transformers are known to have internal continuous symmetries that leave outputs invariant while altering how activations are distributed across coordinates, and hence how well they quantize. GaugeQuant leverages this in-training by introducing a LogSumExp term to the loss that breaks the symmetries, thus selecting a basis that minimizes activation outliers. A stop-gradient operator ensures that only rotation matrices are updated, yielding the language modeling objective completely unaltered. Our quantization strategy requires no specific calibration data, no quantization simulation, and adds a small inference overhead. As a proof of concept, we evaluate on WikiText-2 perplexity and zero-shot reasoning. On Qwen-2.5 0.5B, GaugeQuant reduces perplexity (e.g. $25.62 \to 16.76$ under W4A16) and improves zero-shot accuracy. On LLaMA-2 7B, perplexity improves but downstream accuracy does not. The code is available at: https://github.
    
[^283]: OTAP：用于评估智能体轨迹中规划与执行的结构感知最优传输方法

    OTAP: Structure-Aware Optimal Transport for Evaluating Planning and Execution in Agent Trajectories

    [https://arxiv.org/abs/2607.17082](https://arxiv.org/abs/2607.17082)

    该论文提出 OTAP 方法，将智能体轨迹评估重新框架化为执行图与有效解决方案图之间的最优传输距离，克服了传统二元成功标志和精确匹配评估方式的局限性。

    

    大型语言模型智能体通过生成交错包含规划、工具调用和中间结果的轨迹来完成任务。当前的评估指标要么将此类轨迹简化为一个二元成功标志，要么通过精确匹配将其与参考轨迹进行比较，要么将判断委托给另一个语言模型。成功标志无法区分合理的解决方案与侥幸成功的方案，也无法解释运行失败的原因；而精确匹配则会对被重新排序或以不同方式分解的有效规划进行惩罚。我们将轨迹评估重新框架化为智能体执行图与一组有效解决方案图之间的距离问题，并通过在带属性依赖图上求解不平衡融合 Gromov-Wasserstein 传输问题来加以实现。所得分数称为 OTAP（Optimal Transport for Agentic Planning，智能体规划最优传输），是一个归一化的差异分数。其理想化的、经全局最小化的无熵目标对于固定图中保持依赖关系的重标记具有不变性……

    arXiv:2607.17082v3 Announce Type: replace  Abstract: Large language model agents solve tasks by generating trajectories that interleave planning, tool calls, and intermediate results. Current evaluation metrics reduce such a trajectory to a binary success flag, compare it against a reference by exact matching, or delegate judgment to another language model. A success flag cannot distinguish sound solutions from lucky ones, nor explain why runs fail. Exact matching penalizes valid plans reordered or decomposed differently. We reframe trajectory evaluation as a distance between the agent's execution graph and a set of valid solution graphs, and instantiate it via an unbalanced fused Gromov-Wasserstein transport problem over attributed dependency graphs. The resulting score, termed OTAP (Optimal Transport for Agentic Planning), is a normalized discrepancy score. Its idealized, globally minimized entropy-free objective is invariant to dependency-preserving relabelings of a fixed graph and 
    
[^284]: 理解从预训练到后训练的推理

    Understanding Reasoning from Pretraining to Post-Training

    [https://arxiv.org/abs/2607.16097](https://arxiv.org/abs/2607.16097)

    该论文以国际象棋作为受控测试平台，首次在完整的预训练到后训练流程中系统研究预训练选择（模型规模与数据）如何影响RL计算投入的回报，以及RL后训练究竟对模型的推理能力产生了什么作用。

    

    强化学习（RL）已成为提升大型语言模型（LLM）在复杂推理任务上表现的核心手段，然而RL后训练通常是与之前的预训练割裂开来研究的。因此，两个基本问题仍然悬而未决：(1) 预训练选择（模型规模、数据）如何影响RL计算投入的回报？(2) RL究竟对模型做了什么？在标准的LLM研究设定下，这些问题难以探究：预训练语料庞大且不可控，难以将模型行为归因于预训练还是RL，而且跨两个阶段进行系统性的计算扫描成本高昂到令人却步。为应对这些挑战，我们采用国际象棋作为受控测试平台，在完整的预训练到后训练流程中研究推理能力。我们遵循标准的LLM训练流程，在人类棋局上预训练从5M到1B参数的语言模型，并在合成……（摘要在此处截断，内容不完整）

    arXiv:2607.16097v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has become central to improving large language models (LLMs) on complex reasoning tasks, yet RL post-training is largely studied in isolation from the pretraining that precedes it. As a result, two basic questions remain open: (1) how do pretraining choices (model size, data) shape the returns to RL compute, and (2) what does RL actually do to the model? These questions are difficult to study in the standard LLM setting: pretraining corpora are vast and uncontrolled, making it hard to attribute behaviors to pretraining versus RL, and systematic compute sweeps across both stages are prohibitively expensive. To address these challenges, we use chess as a controlled testbed for studying reasoning across the full pretraining-to-post-training pipeline. We follow the standard LLM training pipeline by pretraining language models from 5M to 1B parameters on human chess games, supervised fine-tuning on synthe
    
[^285]: 基于表格基础模型的数据驱动动态安全评估再审视

    Revisiting data-driven dynamic security assessment with a tabular foundation model

    [https://arxiv.org/abs/2607.16031](https://arxiv.org/abs/2607.16031)

    该论文提出使用表格基础模型（TFM）通过上下文学习进行电力系统动态安全评估，仅需单一模型即可同时评估多个偶然事件，无需逐个偶然事件训练分类器或进行超参数优化，并可泛化到未见过的偶然事件。

    

    数据驱动的故障前动态安全评估（DSA）利用机器学习（ML）快速评估电力系统中可信偶然事件的动态风险。现有方法面临两个局限性：首先，它们需要大量的标注数据库进行训练，且对于可能很长的可信偶然事件列表中的每一个偶然事件，都需要单独训练、调优和维护一个模型；其次，训练好的模型对未见过的偶然事件泛化能力较差。本工作通过使用一种表格基础模型（TFM）来解决这些局限性，该模型通过上下文学习来评估稳定性，无需重新训练或超参数优化。我们使用单个TFM同时评估多个偶然事件，从而无需为每个偶然事件训练一个分类器。我们还刻画了使用电气距离坐标（EDC）作为连续特征何时能够使TFM泛化到未见过的偶然事件，以及何时不能。

    arXiv:2607.16031v2 Announce Type: replace-cross  Abstract: Data-driven pre-fault dynamic security assessment (DSA) rapidly evaluates the dynamic risk of credible contingencies on a power system using machine learning (ML). Existing approaches face two limitations. First, they require a large labelled database for training, with a separate model trained, tuned, and maintained for each contingency in a potentially long list of credible contingencies. Second, the trained models generalize poorly to unseen contingencies. This work addresses these limitations by using a tabular foundation model (TFM) that assesses stability through in-context learning, and requires no retraining or hyperparameter optimization. We use a single TFM to assess many contingencies at once, removing the need to train one classifier per contingency. We also characterize when using electrical distance coordinates (EDC) as continuous features enables the TFM to generalize to unseen contingencies and when it does not,
    
[^286]: 干扰与潜在混杂因素下序贯设置的因果推断

    Causal Inference for Sequential Settings under Interference and Latent Confounding

    [https://arxiv.org/abs/2607.14940](https://arxiv.org/abs/2607.14940)

    该论文提出一种基于最大伪似然估计的计算高效方法，用于在存在结果干扰和低秩潜在混杂因素的序贯观测设置中估计因果效应，并证明了参数估计的非渐近一致性及其向因果量的忠实估计转化。

    

    我们研究序贯观测设置下存在结果干扰的因果推断问题。具体而言，我们考虑N个单元的二元结果在T个时间步上具有马尔可夫性的设置。在每个时间步，N个单元结果之间的依赖关系通过Ising模型来刻画；每个结果还受到一个外部场的影响，该外部场捕获了其处理效应以及潜在混杂因素的作用。与面板数据文献类似，这些潜在混杂因素被建模为具有低秩因子结构。我们的数据是来自该高维分布的单一样本。为了估计感兴趣的因果量，我们提供了一种基于最大伪似然估计（MPLE）的计算高效方法来学习模型参数。在温和的假设下，我们建立了参数估计的非渐近一致性，并证明这可以转化为对感兴趣的因果量的忠实估计。

    arXiv:2607.14940v2 Announce Type: replace  Abstract: We study causal inference under outcome interference for sequential, observational settings. Specifically, we consider settings where the binary outcomes over N units are Markovian across T time steps. At each time step, the outcomes of N units have dependencies captured through an Ising model; each outcome is also impacted through an external field capturing the effects of its treatment as well as latent confounders. Similar to panel data literature, these latent confounders are modeled to have a low-rank factor structure. Our data is a single sample from this high-dimensional distribution. To estimate causal quantities of interest, we provide a computationally efficient method based on Maximum Pseudo-Likelihood Estimation (MPLE) for learning the model parameters. Under mild assumptions, we establish non-asymptotic consistency for parameter estimation and show this translates to faithful estimation of causal quantities of interest a
    
[^287]: 具有通用函数逼近的最优高效上下文组合半赌博机

    Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation

    [https://arxiv.org/abs/2607.13686](https://arxiv.org/abs/2607.13686)

    本文提出SquareCB.Comb算法，为通用函数逼近下的上下文组合半赌博机问题首次实现了极小极大最优的遗憾界 $O(\sqrt{mAT\log|\mathcal{F}|})$。

    

    我们研究上下文组合半赌博机（CCSB）问题，在该问题中，学习者在每一轮观察一个上下文，选择一个由基础臂的子集构成的组合动作，并接收每个被选臂的奖励；其目标是在时间推移过程中最大化累积奖励。尽管以往许多工作都聚焦于具有线性奖励函数逼近的CCSB问题，但针对具有通用奖励函数逼近的CCSB问题的高效算法设计仍然有限。在本工作中，我们研究可实现的设定，即学习者可以访问一个包含真实奖励函数的奖励函数类 $\mathcal{F}$，并提出了 $\mathrm{SquareCB{.}Comb}$ 算法。我们证明 $\mathrm{SquareCB{.}Comb}$ 达到了极小极大最优的遗憾界 $O(\sqrt{mAT\log|\mathcal{F}|})$，其中 $A$ 是基础臂的数量，$m$ 是一个组合动作中所含臂的最大数量，$T$ 是时间范围。

    arXiv:2607.13686v2 Announce Type: replace  Abstract: We study the contextual combinatorial semi-bandit (CCSB) problem, where at each round, the learner observes a context, selects a combinatorial action consisting of a subset of basic arms, and receives the reward of each selected arm; its goal is to maximize its cumulative reward over time. While many previous works focus on CCSB problems with linear reward function approximation, the design of efficient algorithms for the CCSB problem with general reward function approximation is limited. In this work, we study the realizable setting where the learner has access to a reward function class $\mathcal{F}$ that contains the true reward function, and propose the $\mathrm{SquareCB{.}Comb}$ algorithm. We prove that $\mathrm{SquareCB{.}Comb}$ achieves a minimax optimal regret bound of $O(\sqrt{m A T \log |\mathcal{F}|})$, where $A$ is the number of arms, $m$ is the maximum number of arms in a combinatorial action, and $T$ is the time horizon
    
[^288]: TTHE：测试时脚手架演化

    TTHE: Test-Time Harness Evolution

    [https://arxiv.org/abs/2607.08124](https://arxiv.org/abs/2607.08124)

    TTHE提出在测试阶段利用无标签执行轨迹持续演化LLM智能体的脚手架程序，通过智能体提议者生成多样化候选并由裁判筛选，从而在无需标签的情况下实现对任务分布与失败模式变化的实时自适应。

    

    LLM智能体的行为不仅由底层模型决定，还由其脚手架决定：脚手架是构建上下文、调用工具、验证中间结果并从失败中恢复的可执行程序。现有方法通常在训练或开发数据上优化脚手架，并在测试时冻结所得到的程序，这限制了对任务分布、失败模式和工具交互变化的适应能力。我们提出这样一个问题：能否在评估过程中利用无标签的执行轨迹来优化脚手架？我们提出了测试时脚手架演化（TTHE），它将可执行脚手架视为测试时自适应的状态。在每个无标签的测试批次上，TTHE通过能够对执行轨迹进行推理并探索多样化编辑方向的智能体提议者，迭代地演化候选脚手架种群，并由裁判基于执行轨迹和源自执行的代理信号来选择候选脚手架。

    arXiv:2607.08124v3 Announce Type: replace-cross  Abstract: The behavior of an LLM agent is determined not only by the underlying model, but also by its harness: the executable program that constructs context, invokes tools, verifies intermediate results, and recovers from failures. Existing approaches typically optimize harnesses on training or development data and freeze the resulting programs at test time, limiting adaptation to shifts in task distributions, failure modes, and tool interactions. We ask whether the harness can instead be optimized during evaluation using unlabeled execution traces. We introduce Test-Time Harness Evolution (TTHE), which treats the executable harness as the state of test-time adaptation. On each unlabeled test batch, TTHE iteratively evolves a population of candidate harnesses through agentic proposers that reason over execution traces and explore diverse edit directions. A judge selects a candidate using execution traces and execution-derived proxy sig
    
[^289]: 面向普通最小二乘（OLS）回归的数据分区分布式草图方法

    Distributed Sketching on Data Partitions for OLS Regression

    [https://arxiv.org/abs/2607.07888](https://arxiv.org/abs/2607.07888)

    本文提出从数据分区子集生成草图的新型分布式OLS回归方法，使草图计算成本随工作节点增多而下降，并刻画了全局估计器的精确超额损失，证明当子集协方差间平均Burg散度较小时，其性能接近基于全数据草图的方法。

    

    本文研究了一种基于分布式草图的可扩展普通最小二乘（OLS）回归方法。该方法将大型数据集的小型草图分发到多个工作节点，让每个工作节点根据其草图构建局部OLS估计器，再将各局部估计器平均为全局估计器。以往的研究假设每个草图都是从全部数据生成的，但这一草图生成步骤计算成本很高，有时甚至高于标准OLS。相反，我们提出从数据的分区子集中生成每个草图，从而使草图生成成本随工作节点数量的增加而降低。在固定设计设置下，我们刻画了全局估计器的精确超额损失。该损失高于以往研究所建立的损失，但当子集协方差之间的平均Burg散度较小时，它仍与后者保持接近。由于在我们的分区设计下，工作节点数量的增加往往会增大该散度，此……

    arXiv:2607.07888v2 Announce Type: replace  Abstract: This paper studies a scalable approach for ordinary least squares (OLS) regression based on distributed sketching. It distributes small sketches of a large data set across multiple workers, has each worker construct a local OLS estimator from its sketch, and averages the local estimators into a global one. Prior studies assume each sketch is generated from all data, but this sketching step has high computation cost, sometimes even higher than standard OLS. Instead, we propose to generate each sketch from a partitioned subset of the data, which allows the sketching cost to decrease with more workers. Under the fixed design setting, we characterize the exact excess loss of the global estimator. This loss is higher than the loss established in prior studies, but remains close to it when the average Burg divergence between subset covariances is small. Since more workers tend to increase the divergence under our partition design, this rev
    
[^290]: 最终检查点并不足够：沿训练轨迹分析潜在推理的忠实性

    Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories

    [https://arxiv.org/abs/2607.06648](https://arxiv.org/abs/2607.06648)

    该研究揭示了仅评估训练结束时的最终检查点不足以判断潜在推理的忠实性——高任务准确率可能与低反事实响应性共存，因此必须沿整个训练轨迹追踪行为与激活层面的忠实性证据。

    

    潜在推理在连续的隐状态中执行多步推理，有望实现更紧凑、更高效的推理。然而，这些不透明的状态引发了一个忠实性问题：潜在推理步骤是否真正驱动了最终答案的生成。先前的工作在选定的检查点上研究这一问题，并报告了若干不忠实的行为。这种端点视角使得忠实性证据在训练过程中如何演变这一问题未被考察。我们利用经过验证的反事实编辑以及对潜在推理状态的干预，在整个训练过程中追踪行为层面的证据和基于激活的证据。我们发现，高任务准确率可能与低反事实响应性共存：随着准确率的提升，响应性反而可能下降，且不同的潜在推理方法遵循各自不同的轨迹。在ProsQA上，输出对范数噪声替换的敏感性与反事实响应性一同下降，尽管该结果取决于替换的具体方式。

    arXiv:2607.06648v2 Announce Type: replace-cross  Abstract: Latent reasoning performs multi-step inference in continuous hidden states, promising more compact and efficient reasoning. However, these opaque states raise a question of faithfulness: whether the latent reasoning steps drive the final answer. Prior work studies this question at selected checkpoints and reports several unfaithful behaviors. This endpoint view leaves how evidence of faithfulness evolves during training unexamined. We track behavioral and activation-based evidence across training using verified counterfactual edits and interventions on the latent reasoning states. We find that high task accuracy can coexist with low counterfactual responsiveness: as accuracy improves, responsiveness can decline, and different latent reasoning approaches follow distinct trajectories. On ProsQA, output sensitivity to norm-noise replacement declines alongside counterfactual responsiveness, although the result depends on the replac
    
[^291]: GaP：一种用于变分自动化任务的“图即策略”多智能体自学习框架

    GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks

    [https://arxiv.org/abs/2607.05369](https://arxiv.org/abs/2607.05369)

    本文提出了Graph-as-Policy（GaP），一个多智能体自学习编码框架，它从模块化开源机器人技能库生成包含感知、规划和控制节点的有向计算图，并利用内部仿真环境进行排练，从而提升物体几何与姿态变化较大的变分自动化任务的可靠性。

    

    要让机器人在商业和工业应用中可靠地工作，智能体编码系统的最新进展有潜力将可解释的机器人编程与无模型策略的开放世界适应能力相结合。我们专注于“变分自动化”（VA），这是一类在物体几何形状和姿态上的变化比固定自动化更大的任务。无模型策略往往难以缩小变分自动化任务的可靠性差距，而这些任务在商业和工业应用中必须持久且可靠地执行。受先前任务与运动规划（TAMP）和机器人操作系统（ROS）方面工作的启发，我们提出了“图即策略”，这是一个多智能体编码框架，能够从模块化开源机器人技能库（MORSL）中生成包含感知、规划和控制节点的有向计算图。GaP生成一个内部仿真环境，以便用不同的图并行地排练任务实例。

    arXiv:2607.05369v2 Announce Type: replace-cross  Abstract: For robots to work reliably in commercial and industrial applications, recent advances in agentic coding systems have potential to combine interpretable robot programming with the open-world adaptability of model-free policies. We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP generates an internal simulation environment to rehearse task instances with different graphs in para
    
[^292]: 图神经网络中的超参数迁移

    Hyperparameter Transfer in Graph Neural Networks

    [https://arxiv.org/abs/2607.05017](https://arxiv.org/abs/2607.05017)

    本文为使用SGD、Adam和AdamW训练的图神经网络提出并验证了一种超参数迁移参数化方法，使模型在宽度和深度增加时能够实现稳定的特征更新、学习率迁移及性能提升。

    

    深度学习模型的性能在很大程度上取决于学习率、初始化尺度和权重衰减等超参数的设置。超参数迁移旨在使接近最优的超参数设置在不同模型规模之间保持一致，从而可以通过代理调优较小、优化成本更低的对等模型来优化大模型。虽然迁移原则在语言和视觉任务中的密集神经网络背景下已被充分研究，但对于图神经网络（GNN）而言，这一领域仍相对缺乏探索。我们为使用SGD、Adam和AdamW训练的GNN开发并验证了一种迁移参数化方法。通过理论缩放分析和受控实验，我们表明所提出的参数化方法能够产生稳定的特征更新、学习率迁移，并随着宽度和深度的增加而提升性能。对于SGD，我们识别出依赖于图的第一层校正因子，并显示…

    arXiv:2607.05017v2 Announce Type: replace-cross  Abstract: The performance of deep learning models crucially depends on the settings of hyperparameters like learning rate, initialization scale, and weight decay. Hyperparameter transfer aims to make near-optimal hyperparameter settings consistent across model scale, so that large models can be optimized by proxy tuning their smaller, cheaper-to-optimize counterparts. While transfer principles are well-studied in the context of dense neural networks in language and vision tasks, they remain comparatively under-explored for graph neural networks (GNNs). We develop and validate a transfer parameterization for GNNs trained with SGD, Adam, and AdamW. Through theoretical scaling analyses and controlled experiments, we show that the proposed parameterization yields stable feature updates, learning rate transfer, and improved performance as width and depth increase. For SGD, we identify graph-dependent first-layer correction factors and show th
    
[^293]: PixCon：面向基础模型半监督分割的干净正样本对比学习

    PixCon: Clean-Positive Contrastive Learning for Foundation-Model Semi-Supervised Segmentation

    [https://arxiv.org/abs/2607.03068](https://arxiv.org/abs/2607.03068)

    PixCon 通过添加一个仅存储预测与真实标签一致处的干净有标签特征、无需置信度阈值的类别级记忆库，并以多正样本对比损失训练编码器-解码器，从而以极简方式显著提升了基于基础模型的半监督分割性能。

    

    arXiv:2607.03068v2 公告类型： replace-cross 摘要：一个简单的有标签像素对比分支能否在使用强大预训练编码器的半监督分割中带来提升？我们通过 PixCon 研究了这一问题。PixCon 在基于 DINOv2-Base 的 UniMatch V2 实现上添加了一个按类别划分的记忆库，该记忆库仅存储学生预测与真实标签一致位置处的有标签特征。多正样本对比损失利用这些特征来训练共享的编码器-解码器，而在推理阶段头部和记忆库会被丢弃。这通过去除置信度阈值和学习的排序机制，简化了先前的有标签像素记忆库方法。在仅使用 183 个标签的 Pascal VOC 上，PixCon 使三个配对随机种子的 mIoU 分别提升 0.20、0.64 和 1.83 个点，平均值从 87.01 提高到 87.90。使用 92 个标签时平均提升 0.40 个点；在 Cityscapes 和 ADE20K 上的单种子差异范围为 -0.08 至 +0.14 个点。一个使用预测标签进行对比学习的对照实验达到了几乎相同的 Pascal 精度，因此这些结果……

    arXiv:2607.03068v2 Announce Type: replace-cross  Abstract: Can a simple labeled-pixel contrastive branch improve semi-supervised segmentation with a strong pretrained encoder? We study this question with PixCon, which adds a class-wise memory bank to a DINOv2-Base implementation of UniMatch V2. The bank stores ground-truth-labeled features only where the student prediction agrees with the label. A multi-positive contrastive loss uses these features to train the shared encoder-decoder; the head and bank are discarded at inference. This simplifies earlier labeled-pixel banks by removing confidence thresholds and learned ranking. On Pascal VOC with 183 labels, PixCon improves all three paired seeds by 0.20, 0.64 and 1.83 mIoU points, raising the mean from 87.01 to 87.90. With 92 labels the mean gain is 0.40 points; single-seed differences on Cityscapes and ADE20K range from -0.08 to +0.14 points. A predicted-label contrastive control reaches nearly the same Pascal accuracy, so these resul
    
[^294]: AbsoluteDegradation：一个受物理启发的合成胶片退化流水线与档案胶片修复基准

    AbsoluteDegradation: A Physics-Inspired Synthetic Film-Degradation Pipeline and Archival Film Restoration Benchmark

    [https://arxiv.org/abs/2607.02131](https://arxiv.org/abs/2607.02131)

    提出了受物理启发的模块化胶片退化合成流水线AbsoluteDegradation及新的大规模档案胶片修复基准，以解决档案胶片修复中缺乏成对训练数据和标准化评估基准的双重难题。

    

    由于缺乏成对的训练数据以及标准化的评估基准，档案胶片修复仍然是一个根本性的挑战性问题。受损影像的完好版本在物理上是无法恢复的，这使得有监督方法不得不依赖合成数据，而这些合成数据往往无法捕捉真实胶片退化中复杂且具有时间连贯性的特性。与此同时，现有的真实世界数据集在规模、质量和可获取性方面都很有限，阻碍了可靠的评估以及不同方法之间的公平比较。我们通过 AbsoluteDegradation 来解决这两个局限性——这是一个受物理启发、模块化的真实胶片退化合成流水线，同时我们还构建了一个新的大规模档案胶片基准。所提出的流水线将模拟到数字的过程建模为一系列伪影族的结构化组合，包含信号相关的颗粒噪声、参数化的划痕以及时间连贯的摄像机运动，从而……

    arXiv:2607.02131v2 Announce Type: replace-cross  Abstract: Restoring archival film remains a fundamentally challenging problem due to the absence of paired training data and the lack of standardized evaluation benchmarks. Pristine versions of deteriorated footage are physically unrecoverable, requiring supervised methods to rely on synthetic data that often fail to capture the complex, temporally coherent nature of real film degradation. At the same time, existing real-world datasets are limited in scale, quality, and accessibility, hindering reliable evaluation and fair comparison across methods. We address both limitations with AbsoluteDegradation, a physics-inspired, modular pipeline for synthesizing realistic film degradations, and a new large-scale archival benchmark. The proposed pipeline models the analog-to-digital process as a structured composition of artifact families, incorporating signal-dependent grain, parametric scratches, and temporally coherent camera motion, enabling
    
[^295]: Warp：面向冻结机器人策略的结构化输出空间自适应

    Warp: Structured Output-Space Adaptation for Frozen Robot Policies

    [https://arxiv.org/abs/2606.31043](https://arxiv.org/abs/2606.31043)

    提出了Warp，一种上下文条件化的单调双射样条适配器，能在不修改冻结机器人策略的前提下对其输出进行保序的非线性重整形，从而在受控部署偏移的ManiSkill任务中实现最强的性能恢复。

    

    部署后仍保持广泛能力的机器人策略，当动力学、接触、几何、执行或传感条件与源条件不同时可能出现性能退化。我们研究了通过结构化输出空间适配器进行的事后自适应方法，即对冻结基础策略的输出样本进行紧凑变换。我们提出了Warp，一种上下文条件化的单调双射样条，它能够在保持每个维度排序的同时，实现非线性的、依赖样本的重新塑形；平移和仿射变换则分别作为仅位置和位置-尺度的参照基准。在四个具有受控部署偏移的ManiSkill任务上，在一项匹配条件的进化策略（ES）适配器研究中，Warp实现了最强的性能恢复，优于样条参数化、位置-尺度自适应以及通用非线性容量的对照方法。而在PPO下，各方法的排序变得依赖于具体任务，这表明更丰富的适配器结构并不一定自动转化为更好的自适应效果……

    arXiv:2606.31043v4 Announce Type: replace  Abstract: Robot policies that remain broadly competent after deployment can degrade when dynamics, contact, geometry, actuation, or sensing differ from source conditions. We study post-hoc adaptation through structured output-space adapters: compact transformations of samples from a frozen base policy. We introduce Warp, a context-conditioned monotone bijective spline that enables nonlinear sample-dependent reshaping while preserving per-dimension ordering; Translation and Affine provide location-only and location--scale references. Across four ManiSkill tasks with controlled deployment shifts, Warp achieves the strongest recovery in a matched evolution-strategy (ES) adapter study, outperforming controls for spline parameterization, location--scale adaptation, and generic nonlinear capacity. Under PPO, the ordering becomes task dependent, showing that richer adapter structure does not automatically translate into better adaptation under a diff
    
[^296]: OLIVE：基于视图增强的潜在预测与波形重建的语音自监督学习

    OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL

    [https://arxiv.org/abs/2606.30356](https://arxiv.org/abs/2606.30356)

    OLIVE是首个在单一编码器内通过功能分离联合掩码潜在预测与波形重建的语音自监督学习框架，在保持语音识别等判别任务竞争力的同时，可直接重建波形并提升说话人任务表现。

    

    我们提出了在线潜在预测与不变视图及重建（OLIVE），这是一种联合优化分析与合成目标的自监督语音表征学习框架。OLIVE通过单一编码器内部的功能分离，将掩码潜在预测与波形重建耦合起来：重建任务约束编码器早期特征以保留信号级信息，而掩码潜在预测则塑造编码器后期的上下文表征以实现稳健的下游性能，波形视图增强定义了模型所学习的不变性。据我们所知，OLIVE是首个能够直接重建波形、在预训练后保留其声码器、并在判别类任务上保持竞争力的语音自监督学习模型。加入重建目标在很大程度上保留了仅使用掩码潜在预测时的语音识别性能，同时改善了波形重建质量和说话人相关任务的表现。

    arXiv:2606.30356v2 Announce Type: replace  Abstract: We propose Online Latent prediction with Invariant Views and rEconstruction (OLIVE), a self-supervised speech representation learning framework that jointly optimizes analysis and synthesis objectives. OLIVE couples masked latent prediction and waveform reconstruction through a functional separation within a single encoder: reconstruction constrains early encoder features to retain signal-level information, while masked latent prediction shapes later contextual representations for robust downstream performance, with waveform view augmentations defining the invariances they learn. To the best of our knowledge, OLIVE is the first speech SSL model that directly reconstructs waveforms, retains its vocoder after pre-training, and remains competitive on discriminative tasks. Adding reconstruction largely preserves the speech recognition performance of masked latent prediction alone, while improving waveform reconstruction and speaker tasks
    
[^297]: 用于穿孔板压力损失的克里金与神经网络模型

    Kriging and Neural Network models for pressure losses across perforated plates

    [https://arxiv.org/abs/2606.29628](https://arxiv.org/abs/2606.29628)

    本文提出基于克里金和神经网络的两种数据驱动模型，在预测湍流中圆形穿孔板压力损失方面始终优于传统经验公式，即使训练数据有限也展现出良好的准确性和应用前景。

    

    本文提出了两种基于克里金（Kriging）和神经网络（NN）的新型数据驱动模型，用于预测湍流条件下圆形穿孔板两侧的压力损失。这些模型利用文献中已有的两组实验数据进行开发。文中对所提出模型的预测性能进行了评估，并与广泛使用的经验公式进行了比较。研究发现，对于实验数据集中所包含的大多数穿孔板构型，所提出的模型始终优于现有的经验模型。此外，预测的压力损失与实验测量结果总体吻合良好，表明基于克里金和神经网络的数据驱动方法为穿孔板压力损失建模提供了一个可行的框架。总体而言，尽管这两种方法都是在相对有限的实验数据上训练的，但它们都具有广阔的应用前景。

    arXiv:2606.29628v2 Announce Type: replace-cross  Abstract: In this paper, two novel data-driven models based on Kriging and neural networks (NN) are proposed to predict pressure losses across perforated plates with circular perforations in turbulent flows. The models are developed using two sets of experimental data available in the literature. The predictive performance of the proposed models is assessed and compared against widely used empirical formulae. It is found that the proposed models consistently outperform existing empirical models for most perforated plate configurations contained in the experimental datasets. Besides, the predicted pressure losses generally show good agreement with experimental measurements, demonstrating that data-driven approaches based on Kriging and NN provide a feasible framework for modelling pressure losses across perforated plates. Overall, both approaches are promising, despite being trained on a relatively limited amount of experimental data, owi
    
[^298]: 用于鲁棒且可解释的昆虫认证的批次不变光谱智能技术

    Batch-Invariant Spectral Intelligence for Robust and Explainable Insect Authentication

    [https://arxiv.org/abs/2606.26757](https://arxiv.org/abs/2606.26757)

    提出了一个端到端的批次不变光谱网络（BISN），通过结合可学习的预处理模块和熵正则化对抗目标，有效抑制了光谱测量中的批次间变异，从而在未见过的生产批次上实现了鲁棒且可解释的昆虫物种认证。

    

    arXiv:2606.26757v1 公告类型：新 摘要：可食用昆虫提供了一种高效的替代蛋白质来源，与传统畜牧业相比，它们需要的土地和水更少，排放的温室气体也更少。然而，要成功地将它们融入食品供应链，需要可靠的物种认证，以控制过敏原暴露、防止掺假并满足监管标准。近红外光谱提供了一种快速分析工具，但由于光谱测量中的批次间差异，当应用于训练过程中未见过的生产批次时，其性能会下降。我们引入了批次不变光谱网络（BISN），这是一个端到端框架，它将一个可学习的预处理模块（初始化为Savitzky-Golay滤波）与一个熵正则化的对抗目标相结合，以抑制批次特定的光谱变异。与仅在特征提取后强制执行领域适应的域对抗神经网络不同，BISN抑制批次特定的光谱变化，从而在未见过的生产批次上实现鲁棒且可解释的昆虫认证。

    arXiv:2606.26757v1 Announce Type: new  Abstract: Edible insects offer an efficient source of alternative protein, requiring less land, water and emitting less greenhouse gas than conventional livestock. However, their successful integration into the food supply chain demands reliable species authentication to control allergen exposure, prevent adulteration, and meet regulatory standards. Near-infrared spectroscopy provides a rapid analytical tool, but its performance drops when applied to production batches unseen during training due to batch-to-batch variation in spectral measurements. We introduce the Batch-Invariant Spectral Network (BISN), an end-to-end framework that combines a learnable preprocessing module, initialised with Savitzky-Golay filtering, with an entropy-regularised adversarial objective to suppress batch-specific spectral variation. In contrast to Domain-Adversarial Neural Networks, which enforce domain adaptation only after feature extraction, BISN suppress batch-ef
    
[^299]: 水獭天气：高技巧且计算高效的中期天气预报

    Otter Weather: Skillful and Computationally Efficient Medium-Range Weather Forecasting

    [https://arxiv.org/abs/2606.26421](https://arxiv.org/abs/2606.26421)

    本文提出Otter Weather模型，通过高效时空架构实现中期天气预报，在1.5°分辨率下以极低计算成本（<3.5 A100天）超越传统NWP 9.6%，效率比轻量AI模型高2倍，比资源密集型模型低100倍，推动了天气预测的民主化。

    

    arXiv:2606.26421v1 公告类型：新 摘要：最先进的中期人工智能天气模型可以超越传统的数值天气预报（NWP），但需要巨大的训练预算。这限制了对资源不足群体的使用，并严重限制了模型的快速迭代。在这里，我们开发了Otter Weather，这是一个高效的时空预测模型，旨在通过人工智能实现高性能天气预测的民主化。在1.5°分辨率下的ERA5再分析数据上，使用标准WeatherBench协议进行评估，Otter系列显著推进了技巧-计算帕累托前沿。确定性版本在24小时预报时效上比最佳NWP基线高出9.6%，同时训练所需的A100天数少于3.5天。与轻量级人工智能模型相比，它提供了2倍的效率提升，与资源密集型前沿架构相比，计算量减少了100倍。我们通过Continuo训练将这些效率优势扩展到概率预测中。

    arXiv:2606.26421v1 Announce Type: new  Abstract: State-of-the-art medium-range AI weather models can outperform traditional Numerical Weather Prediction (NWP) but require massive training budgets. This restricts usage for under-resourced groups and severely limits fast model iteration. Here we develop Otter Weather, a highly efficient spatiotemporal forecasting model designed to democratise high-performance weather prediction with AI. Evaluated on ERA5 reanalysis data at 1.5{\deg} resolution using standard WeatherBench protocols, the Otter family significantly advances the skill-compute Pareto frontier. The deterministic version outperforms the best NWP baseline by 9.6% at a 24-hour lead time while requiring fewer than 3.5 A100-days for training. It provides a 2x efficiency gain over lightweight AI models and a 100-fold reduction in compute compared to resource-intensive frontier architectures. We extend these efficiency gains into probabilistic forecasting by training via the Continuo
    
[^300]: AutoDex：一个用于灵巧抓取数据收集的自动化真实世界系统

    AutoDex: An Automated Real-World System for Dexterous Grasping Data Collection

    [https://arxiv.org/abs/2606.23689](https://arxiv.org/abs/2606.23689)

    AutoDex是一个无需持续人工监督的自动化真实世界灵巧抓取数据收集系统，通过多视角物体跟踪、残余力矩碰撞监测和姿态导向的自动重置，在100个物体上成功收集了超过3,500次包含成功与失败案例的抓取试验数据。

    

    真实世界灵巧抓取数据收集的规模化受到持续人工投入的限制，包括操作机器人、验证抓取结果以及在试验之间重置物体。仿真可以大规模生成抓取候选方案，但其在硬件上的成功率仍然不确定。我们提出了AutoDex，一个无需持续人工操作即可收集真实世界灵巧抓取数据的系统。给定生成的抓取候选方案，AutoDex规划并执行抓取、标注其结果，并重置物体以在稳定姿态间继续收集数据。多视角物体跟踪支持在手部与物体遮挡情况下的结果标注，而残余力矩碰撞监测和基于姿态的重置使该循环能够在没有持续人工监督的情况下运行。使用AutoDex，我们通过Allegro和Inspire灵巧手在100个物体上收集了超过3,500次抓取试验，同时记录了成功与失败的抓取数据以及同步的视觉观测。

    arXiv:2606.23689v2 Announce Type: replace-cross  Abstract: Scaling real-world dexterous-grasp data collection is limited by the sustained human effort required to operate the robot, verify grasp outcomes, and reset objects between trials. Simulation can generate grasp candidates at scale, but their success on hardware remains uncertain. We present AutoDex, a system that collects real-world dexterous-grasp data without continuous human operation. Given generated grasp candidates, AutoDex plans and executes grasps, labels their outcomes, and resets the object to continue collection across stable poses. Multi-view object tracking supports labeling under hand-object occlusion, while residual-torque collision monitoring and pose-directed reset enable the loop to operate without continuous human supervision. Using AutoDex, we collect over 3,500 grasp trials across 100 objects with Allegro and Inspire hands, recording both successes and failures together with synchronized visual observations 
    
[^301]: TROPT：一个统一和推进离散文本优化的开源框架

    TROPT: An Open Framework for Unifying and Advancing Discrete Text Optimization

    [https://arxiv.org/abs/2606.23496](https://arxiv.org/abs/2606.23496)

    TROPT是首个开源框架，通过单一接口统一了离散文本优化器的执行并标准化其开发流程，解决了现有优化器分散、难以使用和难以对比的问题，降低了在红队测试、审计和可解释性等领域应用与推进离散文本优化的门槛。

    

    离散文本触发优化——即搜索当被模型读取后能够引导模型朝向特定目标的文本序列——是模型红队测试（例如大语言模型越狱攻击）、审计和可解释性研究的基础。然而，当前离散优化器的现状阻碍了它们的采用与发展。首先，现有的优化器即使有开源的，也分散在与特定模型、目标和问题领域绑定的研究代码库中。其次，优化器数量激增，每个都需要额外的工程开销才能使用或扩展，且难以进行直接的对比评估。这些问题共同提高了在现有或新领域采用优化器、以及通过新策略推动其发展的门槛。我们提出TROPT来弥补这些不足，它是首个在单一接口下统一离散优化器执行方式并标准化其开发流程的开源框架。TROPT使得端到端自定义优化变得简单……

    arXiv:2606.23496v2 Announce Type: replace  Abstract: Discrete text-trigger optimization -- searching for text sequences that, when ingested by a model, steer it toward a specified objective -- underpins model red-teaming (e.g., LLM jailbreaks), as well as auditing and interpretability. However, the current state of discrete optimizers hinders their adoption and progress. First, existing optimizers, when open-sourced at all, are scattered across research codebases tied to specific models, objectives, and problem domains. Second, optimizers proliferate, each requiring engineering overhead to use or extend, and remaining hard to compare head-to-head. Together, these raise the bar for adopting optimizers in existing or new domains, and for advancing them via new strategies. We address these gaps with TROPT, the first open-source framework that unifies discrete optimizers' execution and standardizes their development under a single interface. TROPT makes it easy to customize end-to-end opti
    
[^302]: 幻影与泄露：一个用于审计合成数据隐私的统计框架

    Phantoms and Disclosures: A Statistical Framework for Auditing Privacy in Synthetic Data

    [https://arxiv.org/abs/2606.16952](https://arxiv.org/abs/2606.16952)

    本文提出了一个可定制的实证审计框架，通过区分“真实泄露”与“幻影泄露”，并利用训练/保留集划分与严格的统计假设检验，来审计合成数据是否符合零学习或差分隐私等严格隐私基线。

    

    生成式人工智能和大型语言模型（LLM）的迅速普及，引发了人们将合成数据作为敏感真实世界数据集的隐私保护替代品的兴趣。然而，生成高实用性的合成数据往往伴随着记忆并复述训练语料库中私人信息的风险。在这项工作中，我们提出了一个可定制的实证审计框架，旨在检测并解释此类数据泄露。我们的框架引入了一种机制，用于区分“真实泄露”（即系统直接复制了用户的信息）和“幻影泄露”（即系统偶然生成了用户的数据）。通过将输入数据划分为训练集和保留集，并应用严格的统计假设检验，我们判断观察到的泄露是否符合严格的隐私基线，例如零学习基线或特定的差分隐私（DP）界限。

    arXiv:2606.16952v3 Announce Type: replace-cross  Abstract: The rapid adoption of generative AI and Large Language Models (LLMs) has spurred interest in synthetic data as a privacy-preserving alternative to sensitive real-world datasets. However, generating high-utility synthetic data often carries the risk of memorizing and regurgitating private information from the training corpus. In this work, we present a customizable empirical auditing framework designed to detect and explain such data disclosures. Our framework introduces a mechanism to distinguish between "true disclosures"-where the system directly reproduces a user's information-and "phantom disclosures''-where the system incidentally generates a user's data. By partitioning input data into training and holdout sets and applying rigorous statistical hypothesis testing, we determine if observed disclosures are consistent with strict privacy baselines, such as zero-learning or specific Differential Privacy (DP) bounds. Crucially
    
[^303]: 将单类支持向量机分解为单数据支持向量机的集成

    Decomposing one-class support vector machine into an ensemble of one-data support vector machines

    [https://arxiv.org/abs/2606.16002](https://arxiv.org/abs/2606.16002)

    本文提出将单类支持向量机分解为针对单个数据点的模型并利用集成学习重组，结合基于样本均值训练模型的支持向量选择策略，在保持相似分类效果的同时显著加速训练，并实现样本与模型的一一对应。

    

    单类支持向量机（OCSVM）是支持向量机（SVM）用于单类分类（OCC）的一种变体。然而，OCSVM在大规模数据集上存在可扩展性问题。本文提出了一种OCSVM加速策略，其思想是将数据集分解为样本，并为单个数据点训练OCSVM模型，随后应用集成学习将所有模型组合起来，以计算整个数据集的OCSVM模型。此外，通过使用在训练样本均值上训练的OCSVM模型的支持向量选择策略，进一步实现了加速。实验使用Python包将所提方法与传统OCSVM进行了比较，结果表明所提策略比传统OCSVM更快，同时取得了相似的分类结果。此外，所提策略可以在样本与模型之间建立一一对应关系。源代码已上传至 https://github.com/ToshiHayashi/O

    arXiv:2606.16002v2 Announce Type: replace  Abstract: One-class Support Vector Machine(OCSVM) is a variant of support vector machine(SVM) for one-class classification (OCC). However, OCSVM has scalability issues with large datasets. This paper proposes an acceleration strategy for OCSVM. The idea is to decompose the dataset into samples and train OCSVM models for single data points. Subsequently, ensemble learning is applied to combine all models to compute the OCSVM model for the dataset. In addition, further acceleration is achieved through a support vector selection strategy with an OCSVM model trained on the average of the training samples. The experiment compared the proposal and traditional OCSVM using the Python package. The proposed strategy is faster than traditional OCSVM, while achieving similar classification results. Moreover, the proposed strategy can create a one-to-one correspondence between samples and models. Source code is uploaded at https://github.com/ToshiHayashi/O
    
[^304]: 可证明安全且可扩展的强化学习

    Provably Safe, Yet Scalable Reinforcement Learning

    [https://arxiv.org/abs/2606.14536](https://arxiv.org/abs/2606.14536)

    PS2-RL框架通过两阶段架构和学习的备份策略进行前向积分来隐式构造安全保证，在提供严格安全可证明性的同时，实现了随状态维度良好扩展的安全强化学习。

    

    安全强化学习（RL）旨在学习在满足约束条件的同时优化奖励的策略。主流方法依赖于软约束策略优化，这类方法在实证上取得了成功，但无法为学习到的策略提供正式的安全保证。相比之下，具有严格保证的方法通常依赖于显式证书函数，其构建需要直接合成和验证控制不变集，这一过程随状态维度的增加扩展性较差，且往往导致过于保守的行为。在本文中，我们提出了可证明安全且可扩展的强化学习框架（PS2-RL），这是一种新颖的两阶段架构，用于以可扩展的方式学习可证明安全的策略，旨在克服先前方法的关键瓶颈。PS2-RL并不显式计算不变集，而是利用学习到的备份策略对系统动力学进行前向积分，从而生成一个隐式的……

    arXiv:2606.14536v2 Announce Type: replace  Abstract: Safe reinforcement learning (RL) aims to learn policies that optimize rewards while satisfying constraints. Predominant approaches rely on soft-constrained policy optimization, which has achieved empirical success but does not provide formal safety guarantees for the learned policy. In contrast, methods with strict guarantees typically rely on explicit certificate functions, whose construction requires the direct synthesis and verification of control-invariant sets, a process that scales poorly with state dimension and often yields overly conservative behavior. In this paper, we present the Provably Safe, yet Scalable RL (PS2-RL) framework, a novel two-phase architecture for learning provably safe policies in a scalable manner, designed to overcome the key bottlenecks of prior methods. Rather than explicitly computing invariant sets, PS2-RL leverages a learned backup policy to forward-integrate the system dynamics, generating an impl
    
[^305]: 机器学习粒子流作为对撞机物理的基础模型

    Machine-learned particle flow as a foundation model for collider physics

    [https://arxiv.org/abs/2606.14373](https://arxiv.org/abs/2606.14373)

    该论文证明将在事件重建（MLPF）中学到的逐粒子潜在表示作为共享的基础模型表示，可显著提升喷注味道识别、喷注能量回归和丢失动量回归等多种对撞机物理分析任务的性能。

    

    从粒子对撞到物理分析的工作流程需要经过一系列传统上模块化且相互独立的重建步骤，缺乏将低层次探测器数据与高层次分析任务联系起来的共享表示。我们证明，将事件重建转化为机器学习问题可以自然地产生这种共享表示。我们将为粒子流重建（MLPF）训练的机器学习模型重新用于执行三种不同的分析任务：喷注味道识别、喷注能量回归和丢失动量回归。通过将在重建过程中学到的逐粒子潜在表示作为额外的输入特征，我们相比仅使用传统粒子级特征的基线方法取得了显著改进。我们进一步证明，仅使用潜在表示训练的单一线性层即可达到与最先进基线方法具有竞争力的性能。

    arXiv:2606.14373v2 Announce Type: replace-cross  Abstract: The workflow from particle collision to physics analysis passes through a series of reconstruction steps that are traditionally modular and disconnected, with no shared representation linking low-level detector data to high-level analysis tasks. We show that casting event reconstruction as a machine learning problem naturally produces such a shared representation. We repurpose a machine learning model trained for particle-flow reconstruction (MLPF) to perform three distinct analysis tasks: jet flavor identification, jet energy regression, and missing momentum regression. By appending the per-particle latent representations learned during reconstruction as additional input features, we substantially improve over baselines that use conventional particle-level features alone. We further demonstrate that a single linear layer trained using only the latent representations achieves competitive performance against state-of-the-art bas
    
[^306]: 面向不相关机型完工时间调度的学习增强近似算法

    Learning-Augmented Approximation for Unrelated-Machines Makespan Scheduling

    [https://arxiv.org/abs/2606.13133](https://arxiv.org/abs/2606.13133)

    本文回应了Antoniadis等人提出的开放性问题，将学习增强近似框架从选择问题推广到调度领域，针对不相关机器完工时间最小化问题提出了利用重任务分配预测的算法，在预测准确时达到(1+ε)-近似，并随误差增大平滑退化为最坏情况2-近似。

    

    最近，Antoniadis等人（ICLR 2025）提出了一个将预测融入NP难选择问题近似的框架。尽管该方法非常简单，却能紧致地匹配理论下界，使其推广极具吸引力。我们着手解决Antoniadis等人在其工作中提出的一个开放性问题，即如何将该方法扩展到选择问题类别之外的其他重要问题，例如调度问题。我们为不相关机器上的完工时间最小化问题（记为 $R\|C_{\max}$）开发了一种学习增强算法。通过使用重任务分配的预测，我们在预测准确时实现了多项式时间的 $(1+\varepsilon)$-近似，并随着误差的增大平滑地退化为最坏情况下的2-近似。最后，我们对所提方法进行了实证分析。

    arXiv:2606.13133v2 Announce Type: replace-cross  Abstract: Recently, Antoniadis et al. (ICLR 2025) proposed a framework for incorporating predictions to approximate NP-hard selection problems. Despite its simplicity, this approach tightly matches theoretical lower bounds, making its generalization highly compelling. We address an open question raised in the work of Antoniadis et al., concerning the extension of this approach to other important problems outside the class of selection problems, such as scheduling. We develop a learning-augmented algorithm for the makespan minimization problem on unrelated machines, denoted by $R\|C_{\max}$. By using predictions of heavy job assignments, we achieve a polynomial-time $(1+\varepsilon)$-approximation for accurate predictions that smoothly degrades to a worst-case 2-approximation as the error increases. We conclude our work with an empirical analysis of our method.
    
[^307]: 基于上下文先验的流匹配用于分布外脑动力学建模

    Flow Matching with In-Context Priors for Out-of-Distribution Brain Dynamics

    [https://arxiv.org/abs/2606.11833](https://arxiv.org/abs/2606.11833)

    提出PINC-Flow，一种通过在上下文中注入组合式语言和空间先验的逐时间步条件化扩散Transformer，可零样本生成未见认知任务下真实的fMRI脑动态，为反事实神经科学中的计算机内实验设计与评估开辟了道路。

    

    流匹配和扩散模型能够实现从图像到蛋白质等多个领域的条件生成，最近还被扩展到分布外场景。然而，神经时间序列的生成模型在很大程度上仍局限于类别条件化，阻碍了组合式和零样本泛化。在这项工作中，我们提出了PINC-Flow，这是一种逐时间步条件化的扩散Transformer，通过在上下文中同时注入组合式语言先验和可选的空间先验，来生成未见认知任务期间真实的fMRI脑动态。这种零样本生成可以通过在经验验证之前支持新型认知实验的计算机内设计和评估，从而实现反事实神经科学。利用该模型，我们在数百个保留任务条件下进行评估，并表征了预测性能与训练流形之间的关系。仅凭语言输入，该模型就能恢复区域……

    arXiv:2606.11833v2 Announce Type: replace  Abstract: Flow matching and diffusion models enable conditional generation across domains ranging from images to proteins, with recent extensions to out-of-distribution contexts. Yet generative models of neural time series have largely remained restricted to categorical conditioning, precluding compositional and zero-shot generalization. In this work, we propose PINC-Flow, a per-timestep conditioned diffusion transformer for generating realistic fMRI brain dynamics during unseen cognitive tasks by injecting both compositional language and optional spatial priors in-context. Such zero-shot generation could enable counterfactual neuroscience by supporting in-silico design and evaluation of novel cognitive experiments before empirical validation. Leveraging this model, we evaluate across hundreds of held-out task conditions and characterize predictive performance in relation to the training manifold. From language alone, the model recovers region
    
[^308]: 教会扩散模型从左到右进行推测（Teaching Diffusion to Speculate Left-to-Right）

    Teaching Diffusion to Speculate Left-to-Right

    [https://arxiv.org/abs/2606.11552](https://arxiv.org/abs/2606.11552)

    该论文提出让扩散草稿模型以从左到右的方式生成推测token，从而弥合块扩散草稿的双向生成与自回归目标模型严格左到右验证之间的不对称差距，提升推测解码的效率。

    

    大语言模型（LLMs）在广泛的任务中取得了卓越的性能，但其自回归解码过程由于固有的顺序性token生成而带来了高昂的推理成本。推测解码通过采用轻量级草稿模型提出多个未来token，再由更大的目标模型并行验证，从而缓解了这一瓶颈。近期研究表明，扩散语言模型非常适合这一场景，因为它们能够并行生成整块的草稿token，从而突破了自回归草稿生成的顺序性限制。然而这一机制存在一个微妙之处：块扩散草稿模型在块内以双向方式生成token，而验证则由严格从左到右评估token的自回归目标模型完成，这导致对称的训练时目标与非对称的（验证方式之间存在差距）……

    arXiv:2606.11552v3 Announce Type: replace  Abstract: Large language models (LLMs) achieve remarkable performance across a wide range of tasks, but their autoregressive decoding process incurs substantial inference costs due to inherently sequential token generation. Speculative decoding addresses this bottleneck by employing a lightweight draft model to propose multiple future tokens that are subsequently verified in parallel by a larger target model. Recent work has demonstrated that diffusion language models are well suited for this setting, as they can generate entire blocks of draft tokens in parallel and thereby alleviate the sequential constraints of autoregressive drafting. A subtlety of this regime is that block-diffusion drafters generate tokens bidirectionally within a block, whereas verification is performed by an autoregressive target model that evaluates tokens in a strictly left-to-right manner, leaving a gap between the symmetric training-time objective and the asymmetri
    
[^309]: 当在严重损坏的输入上训练时，神经网络会变成质心分类器

    Neural networks become centroid classifiers when trained on heavily corrupted inputs

    [https://arxiv.org/abs/2606.11319](https://arxiv.org/abs/2606.11319)

    该研究发现神经网络对超过90%的输入损坏仍保持远高于随机水平的准确率，并通过均值场理论分析证明，在重度损坏情形下无限宽神经网络会退化为最近质心（原型）分类器。

    

    从不完美的数据中学习是机器学习的核心主题，它将鲁棒性这一实际问题与可学习性这一基本问题联系起来。本文研究属性噪声：即在标签保持完好的情况下从损坏的输入中学习，这一设定相比于标签噪声受到的分析关注要少得多。我们考虑两类损坏模型：加性噪声和替换噪声。通过在损坏的分类数据集上对多层感知机进行实验，我们发现神经网络依然保持鲁棒，即使输入损坏超过90%——远超人类识别能力的极限——其准确率仍远高于随机猜测水平。为了理解这种鲁棒性，我们采用受均值场启发的方法分析了重度损坏情形下的无限宽网络，并推导出分类结果的主导阶决策规则：网络实现了一种原型规则，即最近类……（摘要在此处截断）

    arXiv:2606.11319v2 Announce Type: replace  Abstract: Learning from imperfect data is a central theme in machine learning, connecting practical questions of robustness to fundamental questions of learnability. Here we examine attribute noise: learning from corrupted inputs while keeping the labels intact, a setting that has received considerably less analytical attention than its label-noise counterpart. We consider two types of corruption models: additive noise and replacement noise. Through experiments with multi-layer perceptrons (MLPs) on corrupted classification datasets, we find that neural networks remain robust, maintaining well-above-chance accuracy even when inputs are >90% corrupted -- far beyond human recognition. To understand this robustness, we analyze infinite-width networks in the heavy-corruption regime using a mean-field-inspired approach and derive a leading-order decision rule for the classification outcome: the network implements a prototype rule, the nearest-class
    
[^310]: 基于结构化定位的驾驶视频复杂查询检索

    Driving Video Retrieval for Complex Queries with Structured Grounding

    [https://arxiv.org/abs/2606.09109](https://arxiv.org/abs/2606.09109)

    提出STRIVE-D数据校准检索框架，通过弱标注视频评估并自适应调整查询规则，将校准后的规则分数与视觉-语言及关键词检索信号融合，从而在驾驶视频中可靠检索加塞、急刹等复杂动态事件。

    

    大规模视频检索是自动驾驶数据整理与安全验证的核心任务，用户不仅希望检索到场景，还希望找到诸如加塞、急刹车等动态事件。现有的视觉-语言检索与基于关键词的检索方法常常会遗漏这些事件，原因在于相关运动可能未在文本中被明确描述，或无法通过词汇重叠被捕获。基于规则的检索能够更直接地编码这类事件，但它较为脆弱：当生成或手写的规则与真实驾驶数据的假设不匹配时往往会失效。我们提出了STRIVE-D，一个面向驾驶视频的数据校准检索框架。该框架利用弱标注的域内视频来估计查询规则何时可靠，对与观测数据不匹配的规则进行自适应调整，并将校准后的规则分数与视觉-语言检索及关键词检索信号相融合。在三个驾驶基准上的实验（包括新发布的Dr...（摘要原文在此处被截断）

    arXiv:2606.09109v3 Announce Type: replace-cross  Abstract: Video retrieval at scale is central to data curation and safety validation in autonomous driving, where users want to find not only scenes but also dynamic events such as cut-ins and hard braking. Existing vision-language and keyword-based retrieval methods often miss these events because the relevant motion may not be explicitly described in text or captured by lexical overlap. Rule-based retrieval can encode such events more directly, but it is brittle: generated or hand-written rules often fail when their assumptions do not match real driving data. We propose STRIVE-D, a data-calibrated retrieval framework for driving videos. It uses weakly labeled in-domain videos to estimate when a query rule is reliable, adapt rules that mismatch observed data, and fuse calibrated rule scores with vision-language and keyword-based retrieval signals. Across three driving benchmarks, including newly released human-annotated event data on Dr
    
[^311]: 模糊任务上的扩散式AI控制

    Diffuse AI Control on Fuzzy Tasks

    [https://arxiv.org/abs/2606.08892](https://arxiv.org/abs/2606.08892)

    本文提出将AI控制建模为蓝队与红队的对抗博弈框架，用以研究和防御AI在难以评分的模糊任务上进行长期、分散式暗中破坏的风险。

    

    arXiv:2606.08892v3 公告类型：替换。摘要：部署在关键领域（如AI安全研究）的AI模型，可能因目标不一致而暗中破坏我们的努力。扩散式AI控制是AI安全领域的一个分支，旨在缓解AI在长部署周期内以分散方式实施破坏所带来的风险（扩散性威胁）。这类风险在模糊任务——即难以评分或需要直觉的任务——上尤为棘手。为了理解模糊任务上的扩散性威胁，我们提出了一个将AI控制视为蓝队与红队之间对抗博弈的框架。蓝队使用一个较弱的可信模型构建一个弱评分标准，并依据该评分标准训练一个强大的、可能具有颠覆性的模型，以便在颠覆倾向存在时将其训练消除。红队则试图寻找那些在该弱评分标准下获得高分（因而可能不会在训练中被消除）但实际上对应于糟糕表现的行为。我们在写作任务上测试了我们的框架……（原文摘要在此处截断）

    arXiv:2606.08892v3 Announce Type: replace  Abstract: AI models deployed in critical domains, such as AI safety research, may subtly sabotage our efforts due to misalignment. Diffuse AI Control is a subfield of AI safety concerned with mitigating risks from AI sabotage distributed over long deployment horizons (diffuse threats). These risks are particularly pernicious on fuzzy tasks, i.e. tasks which are hard to grade or require intuition. To understand diffuse threats on fuzzy tasks, we introduce a framework that considers AI control as an adversarial game between a blue team and a red team. The blue team uses a weak trusted model to construct a weak score against which they would train a strong, potentially subversive model to remove the subversion propensity if it were present. The red team then tries to find model behaviors that are rated highly by the weak score, and thus might not be trained out, but actually correspond to poor performance. We test our framework on the task of wri
    
[^312]: 变分推断默认优化器的大规模实证调参与比较

    Large-scale empirical tuning and comparison of default optimizers for variational inference

    [https://arxiv.org/abs/2606.07841](https://arxiv.org/abs/2606.07841)

    本文通过对56种随机梯度优化算法在1092个贝叶斯推断问题上进行超55万次优化运行的大规模实证评估，系统比较了无调参自适应优化器在黑箱变分推断中的表现，确立了无调参优化推断的最新技术水平。

    

    黑箱变分推断（BBVI）是一种依赖随机优化的后验近似方法。在实践中，支撑BBVI的随机优化器通常需要针对具体问题进行大量调参，这削弱了其作为真正“黑箱”推断算法的承诺。然而，在过去十年中，许多新的自适应随机优化算法被开发出来，它们减少或完全消除了调参的需求。在这项工作中，我们在BBVI的背景下研究了这一系列新的自适应方法，旨在确立无调参的基于优化的推断方法的最新技术水平。特别是，我们对56种基于随机梯度的优化算法应用于1092个贝叶斯推断优化问题进行了大规模实证评估，涉及超过55万次单独的优化运行和15个核心年的计算量。

    arXiv:2606.07841v2 Announce Type: replace-cross  Abstract: Black-box variational inference (BBVI) is a methodology for posterior approximation that relies on stochastic optimization. In practice, the stochastic optimizers underpinning BBVI generally require extensive problem-specific tuning, which undermines its promise as a truly "black box" inference algorithm. However, over the past decade, many new adaptive stochastic optimization algorithms have been developed that reduce or remove entirely the need for tuning. In this work, we investigate this new collection of adaptive methods in the context of BBVI, with the goal of establishing the current state of the art in tuning-free optimization-based inference. In particular, we present a large-scale empirical evaluation of 56 stochastic gradient-based optimization algorithms applied to 1092 Bayesian inference optimization problems, involving over 550,000 individual optimization runs and 15 core-years of compute. The optimization algorit
    
[^313]: 锚点主成分分析

    Anchor PCA

    [https://arxiv.org/abs/2606.06233](https://arxiv.org/abs/2606.06233)

    Anchor PCA通过在总体解释方差与共享和领域特定低秩嵌入一致性之间进行权衡，为多领域数据找到鲁棒的共享降维嵌入，且可通过修改后的目标矩阵上的PCA高效求解。

    

    主成分分析（PCA）是最广泛使用的无监督降维技术之一。我们研究了针对来自多个相关领域数据的PCA。由于主成分通常在不同领域之间存在差异，获得共享低秩嵌入的一种方法是对合并后的数据执行PCA。然而，这种方法可能会聚焦于仅在少数几个领域中表现出高变异的虚假方向。为了找到一种能够在未见但相似的领域中仍能解释大部分方差的鲁棒嵌入，我们转而提出聚焦于共享的变异方向。为此，我们引入了Anchor PCA（锚点PCA），它在总体解释方差与共享嵌入和领域特定低秩嵌入之间的一致性之间进行权衡。Anchor PCA等价于对修改后的目标矩阵执行PCA，因此可以高效求解。此外，我们证明了Anchor PCA能够恢复最大不变子空间，并具有极小极大重构的解释。

    arXiv:2606.06233v2 Announce Type: replace-cross  Abstract: Principal component analysis (PCA) is one of the most widely used unsupervised dimension reduction techniques. We study PCA for data from multiple related domains. Since principal components generally differ across domains, one way to obtain a shared low-rank embedding is to perform PCA on the pooled data. However, this approach can focus on spurious directions that exhibit high variation in only a few domains. To find a robust embedding that still explains most variance in unseen but similar domains, we propose instead to focus on shared directions of variation. To this end, we introduce Anchor PCA which trades off overall explained variance with agreement between the shared and domain-specific low-rank embeddings. Anchor PCA amounts to PCA on a modified target matrix and thus can be solved efficiently. Moreover, we show that Anchor PCA recovers a maximal invariant subspace and admits a minimax reconstruction interpretation un
    
[^314]: 扩散模型只观测到梯度：得分匹配误差的几何视角

    Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors

    [https://arxiv.org/abs/2606.06179](https://arxiv.org/abs/2606.06179)

    该论文从几何视角研究扩散模型的得分匹配误差，证明只有误差中可观测的梯度分量才影响采样质量（L² 得分误差即使任意大，模型仍可完美匹配目标分布），并据此推导出仅依赖该梯度分量的更紧 KL 散度上界。

    

    基于得分的扩散模型通常通过最小化 L² 得分匹配误差来训练，标准理论分析也依赖该量来界定学习分布与目标分布之间的采样差异。近期研究表明，得分场可被正交分解为梯度分量与螺线（solenoidal）分量，其中只有梯度部分对精确采样起作用。我们将这一分解应用于训练所得模型的得分误差，并研究其定量后果。由于误差中只有梯度分量会进入边缘分布的 Fokker-Planck 动力学，模型即使产生任意大的 L² 得分误差，仍可完美匹配目标分布。基于这一观察，我们给出了三项结果。其一，我们推导出一个仅依赖于误差中可观测梯度分量的 KL 散度上界，该上界收紧了针对一般得分场（摘要原文在此截断）

    arXiv:2606.06179v3 Announce Type: replace-cross  Abstract: Score-based diffusion models are typically trained by minimizing the $L^2$ score matching error, and standard theoretical analyses rely on this quantity to bound the sampling discrepancy between the learned and target distributions. Recent work has shown that the score admits an orthogonal decomposition into gradient and solenoidal components in which only the gradient part matters for exact sampling. We apply this decomposition to the score error of trained models and study its quantitative consequences. Since only the gradient component of the error enters the marginal Fokker-Planck dynamics, a model can incur arbitrarily large $L^2$ score error while perfectly matching the target. Starting from this observation, we present three results. First, we derive an upper bound on the Kullback-Leibler divergence that depends only on the observable gradient component of the error. It tightens the standard Girsanov bound for generic sc
    
[^315]: SALT：当更多采样在基于分组的策略优化中失效时，如何让它们发挥作用

    SALT: When More Rollouts Don't Help in Group-Based Policy Optimization and How to Make Them Matter

    [https://arxiv.org/abs/2606.05800](https://arxiv.org/abs/2606.05800)

    论文揭示了GRPO式组归一化中增加rollout数量会因策略梯度特征聚集成低秩带符号几何而相互抵消，并提出SALT插件，通过样本级梯度几何将组相对系数分解为共享与残差通道并自适应放大残差通道，使更多rollout真正发挥作用。

    

    采用可验证奖励的强化学习（RLVR）通常采用GRPO风格的组相对更新，即为每个提示采样多个rollout来构建归一化的学习信号。然而，仅仅增加rollout的数量并不能可靠地增强学习效果：在GRPO风格的组归一化下，每个rollout的策略梯度特征可能集中到一种低秩的带符号几何结构中，导致聚合过程中出现大量相互抵消，从而削弱了有效更新。我们通过SALT来解决这一失效模式——SALT是一个子空间自适应几何插件组件，利用逐样本的梯度几何来重新加权组相对更新的系数。SALT从小批量的Gram几何中估计出一个主导共享子空间，将组相对系数分解为共享通道和残差通道，并在带符号抵消严重时自适应地放大残差通道。在多样化的面向推理的RLVR基准上……

    arXiv:2606.05800v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) often adopts GRPO-style group-relative updates, sampling multiple rollouts per prompt to construct normalized learning signals. However, merely increasing the number of rollouts does not reliably strengthen learning: under GRPO-style group normalization, per-rollout policy-gradient features can concentrate into a low-rank, signed geometry, causing substantial cancellation during aggregation and weakening the effective update. We address this failure mode with SALT, a Subspace-Adaptive geometry pLug-in componenT that uses sample-wise gradient geometry to reweight the coefficients of group-relative updates. SALT estimates a dominant shared subspace from the mini-batch Gram geometry, decomposes group-relative coefficients into shared and residual channels, and adaptively amplifies the residual channel when signed cancellation is severe. Across diverse reasoning-oriented RLVR benchmar
    
[^316]: 先去噪，后正交化：通过谱滤波理解Muon中的动量

    Denoise First, Orthogonalize Later: Understanding Momentum in Muon via Spectral Filtering

    [https://arxiv.org/abs/2606.03899](https://arxiv.org/abs/2606.03899)

    该论文证明了Muon优化器中的动量本质上是一个谱滤波器——“先去噪、后正交化”的顺序能抑制梯度扰动、扩大谱间隙，从而稳定奇异子空间并使更新更可靠。

    

    Muon最近在大型语言模型训练中展现出了强劲的实证性能，但动量在Muon中的理论作用仍然不清楚。现有的Muon分析要么去除动量以孤立地研究谱更新，要么保留动量却未解释它为何能提升实证性能。我们的工作弥合了这一空白，证明Muon中的动量起到了谱滤波器的作用。在一个结构化的“信号加扰动”梯度模型下，我们证明动量能够抑制扰动同时保留主导信号，从而扩大二者之间的谱间隙。这一扩大的谱间隙稳定了传递给Muon正交化步骤的矩阵的奇异子空间，使得所得更新更加可靠。我们进一步证明，与颠倒这一顺序或简单地去除动量相比，在正交化之前应用动量能够可证明地实现与梯度信号分量更强的对齐。

    arXiv:2606.03899v3 Announce Type: replace  Abstract: Muon has recently demonstrated strong empirical performance in large language model training, but the theoretical role of momentum in Muon remains unclear. Existing analyses of Muon either remove momentum to study spectral updates in isolation, or retain momentum without explaining why it improves empirical performance. Our work bridges this gap by showing momentum in Muon acts as a spectral filter. Under a structured signal-plus-perturbation gradient model, we prove that momentum suppresses perturbations while preserving the dominant signal, thereby enlarging the spectral gap between them. This enlarged gap stabilizes the singular subspaces of the matrix passed to Muon's orthogonalization step, making the resulting update more reliable. We further show that applying momentum before orthogonalization achieves provably stronger alignment with the signal component of the gradient than either reversing this order or simply removing mome
    
[^317]: 物理基础模型从数值模拟到实验室湍流的涌现式迁移

    Emergent Transfer of a Physics Foundation Model from Simulation to Laboratory Turbulence

    [https://arxiv.org/abs/2606.01470](https://arxiv.org/abs/2606.01470)

    该论文验证了物理基础模型能够从数值模拟涌现式地迁移到真实实验室湍流，在瑞利-泰勒不稳定性这一长期存在模拟-实验差异（增长率相差约三倍）的难题上取得了突破。

    

    物理基础模型能否有效地部署于真实实验室实验，仍是科学机器学习（ML）领域一个悬而未决的问题。我们在瑞利-泰勒不稳定性（RTI）上检验了这一问题。RTI 是一种普遍存在且极具挑战性的流体不稳定性，从桌面尺度的流动到超新星爆发中均可见到：当较轻的流体加速进入较重的流体时，密度界面上的微小扰动会发展为混沌的、多尺度的混合。标准 ML 模型难以应对 RTI，并且尽管经过了一个多世纪的理论、数值与实验研究，RTI 仍存在一个模拟与实验之间未解决的差异：大多数实验室实验测得的晚期混合增长率 α（约 0.06–0.07）约为理想化直接数值模拟（DNS，约 0.02）所得数值的三倍，而这一差距的起源至今仍有争议。这些特性使 RTI 成为检验这一重要问题的严格试金石（原文摘要至此处截断）。

    arXiv:2606.01470v2 Announce Type: replace-cross  Abstract: Whether physics foundation models can be usefully deployed on laboratory experiments remains an open question for scientific machine learning (ML). We test this question on the Rayleigh-Taylor instability (RTI), a ubiquitous and demanding fluid instability seen from tabletop flows to supernova explosions, in which small perturbations at a density interface grow into chaotic, multiscale mixing as a lighter fluid accelerates into a heavier one. Standard ML models struggle with RTI, and despite over a century of theoretical, numerical, and experimental work, it carries an unresolved discrepancy between simulation and experiment: the late-time mixing growth rate, $\alpha$, measured in most laboratory experiments ($\sim$ 0.06-0.07), is roughly three times the value from idealized direct numerical simulations (DNS, $\sim$ 0.02). The gap's origin remains debated. These properties make RTI a stringent test for a question that matters w
    
[^318]: FLAG：基于潜变量增强引导的流策略最大熵强化学习

    FLAG: Flow Policy MaxEnt-RL by Latent Augmented Guidance

    [https://arxiv.org/abs/2605.30749](https://arxiv.org/abs/2605.30749)

    FLAG通过在状态空间中引入流模型潜变量来局部化采样区域，避免了重要性权重坍缩问题，使最大熵强化学习能够在有限重要性样本下实现表达力强的策略优化并扩展到高维动作空间。

    

    最大熵强化学习能够实现稳健的探索，但实际实现中策略通常被限制为简单的高斯分布。虽然近期方法通过重要性加权的监督学习引入了表达力更强的生成式策略，但这些方法容易出现重要性权重坍缩问题，从而限制了其在高维动作空间中的可扩展性。我们的关键洞察是通过局部化采样区域来缓解这一限制，避免在整个动作空间上进行重要性采样所导致的权重退化。为了实现这一洞察，我们提出了FLAG（具有潜变量增强引导的流策略）。FLAG通过流模型潜变量增强状态空间，并优化一个可证明一致性的代理最大熵强化学习目标。我们通过实验证明，FLAG能够在有限的重要性样本下实现表达力强的策略优化，并可扩展到基于状态的高维任务中。

    arXiv:2605.30749v2 Announce Type: replace  Abstract: Maximum entropy reinforcement learning (MaxEnt-RL) enables robust exploration, yet practical implementations often restrict policies to simple Gaussians. While recent approaches incorporate expressive generative policies via importance-weighted supervised learning, they are prone to importance weight collapse, which limits their scalability in high-dimensional action spaces. Our key insight is to mitigate this limitation by localizing the sampling region, avoiding the weight degeneracy induced by importance sampling over the entire action space. To instantiate this insight, we introduce \textbf{FLAG} (\textbf{F}low policy with \textbf{L}atent-\textbf{A}ugmented \textbf{G}uidance). FLAG augments the state space with a flow latent variable and optimizes a provably consistent proxy MaxEnt-RL objective. We empirically demonstrate that FLAG enables expressive policy optimization with limited importance samples and scales to state-based hi
    
[^319]: 基于分布鲁棒训练的保守神经后验估计

    Conservative neural posterior estimation via distributionally robust training

    [https://arxiv.org/abs/2605.28516](https://arxiv.org/abs/2605.28516)

    提出DRO-NPE方法，通过Wasserstein模糊集上的分布鲁棒训练解决神经后验估计在有限模拟预算下过度自信的问题，显著提升了基于模拟推断的覆盖率和校准可靠性。

    

    基于模拟的推断中，神经后验估计（NPE）在有限的模拟预算下往往产生过度自信且不可靠的后验分布。为解决这一问题，我们提出了DRO-NPE，这是一种分布鲁棒的方法，用Wasserstein模糊集上的最坏情况损失替代标准NPE目标。我们引入了基于KL散度的覆盖不足和校准偏差度量，并利用这些度量证明DRO-NPE目标能够控制过拟合并降低后验的过度自信。我们的方法易于处理、可并行化，并且能够与标准归一化流轻松集成。在基准SBI任务上，DRO-NPE持续改善覆盖率和校准效果，同时缩小经验NPE损失与总体NPE损失之间的差距，从而在低模拟条件下实现更可靠的推断。

    arXiv:2605.28516v2 Announce Type: replace-cross  Abstract: Simulation-based inference with neural posterior estimation (NPE) often yields overconfident and unreliable posteriors under limited simulation budgets. To address this, we propose DRO-NPE, a distributionally robust approach that replaces the standard NPE objective with a worst-case loss over a Wasserstein ambiguity set. We introduce KL-based metrics for miscoverage and miscalibration, and use these to show that the DRO-NPE objective controls overfitting and reduces posterior overconfidence. Our method is tractable, parallelisable, and readily integrates with standard normalising flows. Across benchmark SBI tasks, DRO-NPE consistently improves coverage and calibration, while narrowing the gap between empirical and population NPE loss, leading to more reliable inference in low-simulation regimes.
    
[^320]: 更具表达力的前馈层：第一部分：令牌自适应的激活混合

    More Expressive Feedforward Layers: Part I. Token-Adaptive Mixing of Activations

    [https://arxiv.org/abs/2605.26647](https://arxiv.org/abs/2605.26647)

    本文提出令牌自适应的激活混合（MoA）前馈层设计，通过轻量级输入相关门控混合多个激活函数，并从理论上证明了其表达能力严格超越可学习激活（LA）和固定激活FFN。

    

    前馈网络（FFN）层在基于Transformer的大语言模型（LLM）中占据了相当大的参数比例和非线性表达能力。尽管激活函数已从ReLU和GELU演进到SwiGLU等门控变体，但大多数FFN设计仍然使用单一固定的激活函数，对所有令牌（token）应用相同的非线性变换。在这项工作中，我们提出了激活混合（Mixture of Activations, MoA），这是一种令牌自适应的FFN设计，它使用轻量级的、依赖于输入的门控机制来混合一个激活函数字典，同时共享相同的线性投影。作为输入无关的对应方案，我们还引入了可学习激活（Learnable Activations, LA），它为ReLU型和SwiGLU型FFN构造激活函数的线性组合。在理论方面，我们在固定激活FFN、LA和MoA之间建立了严格的有限宽度表达能力分离关系：LA严格包含固定激活FFN的表达能力，而MoA又严格包含LA，并进一步……

    arXiv:2605.26647v2 Announce Type: replace  Abstract: Feedforward network (FFN) layers account for a large fraction of parameters and nonlinear expressivity in Transformer-based large language models (LLMs). Despite the evolution from ReLU and GELU to gated variants such as SwiGLU, most FFN designs still use a single fixed activation function, applying the same nonlinear transformation to all tokens. In this work, we propose Mixture of Activations (MoA), a token-adaptive FFN design that mixes a dictionary of activation functions using lightweight input-dependent gates while sharing the same linear projections. As an input-independent counterpart, we also introduce learnable activations (LA), which form linear combinations of activation functions for both ReLU-type and SwiGLU-type FFNs. Theoretically, we establish strict finite-width expressive separations among fixed-activation FFNs, LA, and MoA: LA strictly contains fixed-activation FFNs, while MoA strictly contains LA, with the additi
    
[^321]: AvalancheBench：通过潜在世界恢复评估企业级智能体驱动的语义查询处理

    AvalancheBench: Evaluating Enterprise Agent-Driven Semantic Query Processing Through Latent World Recovery

    [https://arxiv.org/abs/2605.24183](https://arxiv.org/abs/2605.24183)

    该论文提出了AvalancheBench，一个从潜在世界规范生成观测数据、通过涵盖文本、图像和文档的58个多部分问题评估数据智能体端到端语义查询能力的多模态企业分析基准，并首次揭示了错误经合法依赖关系传播、不当复用中间结果导致无关问题失败的“雪崩效应”。

    

    随着大语言模型（LLM）能力的提升，数据智能体正成为数据分析的新接口，而语义查询处理引擎则利用LLM对结构化和非结构化数据进行操作。现有基准测试大多将智能体与语义查询引擎分开评估，因此智能体能否在端到端分析中决定何时以及如何使用语义算子仍不清楚。我们提出了AvalancheBench，这是一个多模态企业分析基准，它从潜在世界规范生成观测数据，并据此评估数据智能体恢复该潜在世界的能力。该基准包含3个经过验证的任务，共58个问题，涵盖文本、图像和文档。这些问题以多部分分析简报的形式一并提供，且可能共享中间结果，从而暴露出“雪崩效应”：错误可以通过合法的依赖关系传播，而不恰当的复用会导致原本互不相关的问题失败。AvalancheBench测……（原文摘要至此截断）

    arXiv:2605.24183v2 Announce Type: replace-cross  Abstract: As LLMs improve, data agents are emerging as a new interface for analytics, while semantic query processing engines use LLMs to operate over structured and unstructured data. Existing benchmarks largely evaluate agents and semantic query engines in isolation, leaving it unclear whether agents can decide when and how to use semantic operators in end-to-end analysis.   We introduce AvalancheBench, a multimodal enterprise analytics benchmark that generates observed data from latent world specifications and evaluates data agents on their ability to recover that world. The benchmark's 3 validated tasks contain 58 questions spanning text, images, and documents. Questions are provided together as a multi-part analytical brief and may share intermediate results, exposing the avalanche effect: errors can propagate through legitimate dependencies, and inappropriate reuse can cause otherwise unrelated questions to fail. AvalancheBench mea
    
[^322]: EmoTrack：面向基于文本的抑郁严重程度估计的临床-语义建模

    EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation

    [https://arxiv.org/abs/2605.22286](https://arxiv.org/abs/2605.22286)

    本文提出EmoTrack框架，通过临床-语义建模将临床特征分数与话语级表示相融合，从心理咨询文本中预测PHQ-8抑郁严重程度总分，解决了临床评估与对话证据之间关联不足的问题。

    

    基于文本的心理咨询为评估抑郁严重程度提供了宝贵的信息来源。我们研究了如何从咨询对话记录中预测八项患者健康问卷（PHQ-8）的总分，PHQ-8是一种抑郁严重程度的自我报告测量工具。基于临床的方法主要依靠大语言模型（LLM）推理来获得结构化的会话级评估，但这些评估提供的信息有限，无法说明哪些话语支持每个评分。基于训练的方法直接在句子或其语义嵌入上训练预测器，能够保留局部对话细节，但必须从有限的有标签对话记录中学习临床结构。将少量临床特征分数与长序列的高维话语表示相集成极具挑战性，因为简单的融合方法可能会过度强调对话证据，并且无法将临床特征与支持性话语相关联。（原文摘要在此处截断）

    arXiv:2605.22286v2 Announce Type: replace-cross  Abstract: Text-based counseling provides a valuable source of information for assessing depression severity. We study prediction of the total score on the eight-item Patient Health Questionnaire (PHQ-8), a self-report measure of depression severity, from counseling transcripts. Clinical-based methods rely mainly on large language model (LLM) inference to obtain structured session-level assessments, but these assessments provide limited information about which utterances support each score. Training-based methods train predictors directly on sentences or their semantic embeddings and preserve local conversational detail, but must learn clinical structure from limited labeled transcripts. Integrating a small set of clinical feature scores with a long sequence of high-dimensional utterance representations is challenging, as simple fusion can overemphasize dialogue evidence and fail to link clinical features to supporting utterances. Combini
    
[^323]: 面向分布偏移下忠实词典可解释性的几何自适应解释器

    Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift

    [https://arxiv.org/abs/2605.21849](https://arxiv.org/abs/2605.21849)

    提出几何自适应解释器（GAE），仅用无标签的分布外激活、无需梯度更新即可将词典重新对齐到OOD活跃子空间，从而量化并减小分布偏移下的忠实性差距。

    

    机械可解释性旨在通过识别因果相关的内部结构来解释模型的行为。基于词典的解释器（如稀疏自编码器和转码器）是其中的主要工具，但它们在分布外（OOD）偏移下的忠实性却很少受到系统性关注。我们证明，分布偏移会旋转模型实际使用的子空间，导致在分布内（ID）激活上训练的解释器词典发生错位。我们将这种错位形式化为“忠实性差距”，即ID词典与OOD活跃子空间之间的几何距离，并证明该差距约束了额外的OOD投影损失。为了减小这一差距，我们提出了几何自适应解释器（GAE），它在保留原始稀疏特征激活的同时，将词典与OOD活跃子空间重新对齐。该方法仅需要无标签的OOD激活，且无需任何梯度更新。

    arXiv:2605.21849v2 Announce Type: replace-cross  Abstract: Mechanistic interpretability aims to explain a model's behavior by identifying causally responsible internal structures. Dictionary-based explainers such as sparse autoencoders and transcoders are a primary tool, but their faithfulness under out-of-distribution (OOD) shift has received little systematic attention. We show that distribution shift rotates the subspace that the model actively uses, misaligning the explainer's dictionary trained on in-distribution (ID) activations. We formalize this misalignment as the faithfulness gap, a geometric distance between the ID dictionary and the OOD-active subspace, and show that it bounds excess OOD projection loss. To reduce this gap, we propose the Geometry-Adaptive Explainer (GAE), which realigns the dictionary with the OOD-active subspace while preserving the original sparse feature activations. This requires only unlabeled OOD activations and no gradient updates. For the adapted d
    
[^324]: 面向Delta规则线性Transformer的快速且稳定的三角矩阵求逆

    Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers

    [https://arxiv.org/abs/2605.21325](https://arxiv.org/abs/2605.21325)

    本文系统分析了Delta规则线性Transformer中三角矩阵求逆的直接法与迭代法算法，重点考察数值稳定性与计算复杂度，以解决该核心操作的性能瓶颈和数值精度问题。

    

    线性注意力已成为高效长上下文架构的基石，这一点从其被集成到Qwen3.5/3.6、Kimi Linear/K3和RWKV-7等最先进的开源模型中即可得到印证。采用所谓Delta规则的线性注意力层的模型，其核心子程序之一是三角矩阵求逆。这一操作往往构成性能瓶颈，并且由于其对数值误差高度敏感，若实现不当，会显著损害端到端模型的精度。本工作对直接法与迭代法三角矩阵求逆算法进行了系统性分析，重点关注富含矩阵乘积的方法，这类方法有潜力高效利用现代硬件。为此，我们的分析涵盖了数学与实践两方面的广泛内容，并着重关注数值稳定性、计算复杂度等方面。

    arXiv:2605.21325v2 Announce Type: replace  Abstract: Linear attention has emerged as a cornerstone for efficient long-context architectures, as evidenced by its integration into state-of-the-art open-source models including Qwen3.5/3.6, Kimi Linear/K3, and RWKV-7. Models that incorporate linear attention layers with the so-called Delta-Rule involve the inversion of triangular matrices as a core sub-routine. This operation often forms a performance bottleneck, and, due to its high-sensitivity to numerical errors, it can significantly deteriorate end-to-end model accuracy if it is not carefully implemented. This work provides a systematic analysis of both direct and iterative triangular inversion algorithms, targeting methods that are rich in matrix products, and, therefore, have the potential to efficiently utilize modern hardware. To that end, our analysis covers a broad spectrum of mathematical and practical aspects, with a heavy focus on numerical stability, computational complexity,
    
[^325]: 一种严谨且易于计算的模型复杂度度量

    A Rigorous, Tractable Measure of Model Complexity

    [https://arxiv.org/abs/2605.21167](https://arxiv.org/abs/2605.21167)

    该论文提出了一种基于特征向量相似性的、数学上严谨且易于计算的模型复杂度度量，它适用于参数化和核方法非参数化模型，并能统一泛化多种现有模型特定的复杂度度量（如多项式次数、k近邻邻居数、决策树分裂次数、随机森林树的数量等）。

    

    准确评估模型的复杂度对于模型解释、泛化和模型选择等课题至关重要。然而，现有的复杂度度量方法要么依赖于启发式假设，要么计算代价过高。在本文中，我们提出了一种数学上严谨且易于计算的模型复杂度度量，该度量基于不同输入的特征向量之间的相似性。因此，它对任何参数化模型（通过NTK特征泛化）以及基于核的非参数模型（通过诱导核）都有良好的定义。我们证明了我们的复杂度度量可以泛化多种特定于模型的复杂度度量，包括多项式次数（对应多项式回归）、核长度尺度（对应Matérn核）、邻居数量（对应k近邻）、分裂次数（对应决策树）以及树的数量（对应随机森林）。我们还使用我们的度量获得了对……的新见解。

    arXiv:2605.21167v2 Announce Type: replace-cross  Abstract: An accurate assessment of a model's complexity is crucial for topics such as interpretation, generalization, and model selection. However, existing complexity measures either rely on heuristic assumptions or are computationally prohibitive. In this paper, we present a mathematically rigorous yet easy-to-compute measure of model complexity that is based on the similarities between feature vectors for different inputs. It is thus well-defined for any parametric model (via NTK feature generalization) and for kernel-based non-parametric models (via the induced kernel). We prove that our measure of complexity generalizes model-specific complexity measures, including polynomial degree (for polynomial regression), kernel length scale (for Mat\'ern kernels), number of neighbors (for k-nearest neighbors), number of splits (for decision trees), and number of trees (for random forests). We also use our measure to obtain new insights into 
    
[^326]: 蒸馏以思考，预见以行动：面向自动驾驶的认知-物理强化学习

    Distill to Think, Foresee to Act: Cognitive-Physical Reinforcement Learning for Autonomous Driving

    [https://arxiv.org/abs/2605.21139](https://arxiv.org/abs/2605.21139)

    CoPhy 框架通过将视觉语言模型知识蒸馏进 BEV 编码器实现零推理成本的认知能力，并构建自回归 BEV 世界模型来预测候选动作的未来后果，从而突破模仿学习的行为克隆上限，实现自动驾驶的认知-物理强化学习。

    

    当前的端到端自动驾驶模型从根本上受限于模仿学习的行为克隆上限。虽然强化学习为实现更智能的自主性提供了一条路径，但它需要两个尚缺失的基础设施：(1) 一个能够理解交通语义和驾驶意图的认知基础；(2) 一个能够预判候选动作后果的具有前瞻性的物理环境。为此，我们提出了 CoPhy，一个面向自动驾驶的认知-物理强化学习框架。为了“蒸馏以思考”，我们将视觉语言模型（VLM）的知识蒸馏到 BEV 编码器中，然后完全舍弃 VLM，在零推理成本下保留认知能力，同时将认知通道释放为可插拔接口，以支持可选的人类自然语言指令。为了“预见以行动”，我们构建了一个自回归的 BEV 世界模型，该模型以候选动作为条件显式预测未来的语义地图，为……

    arXiv:2605.21139v4 Announce Type: replace-cross  Abstract: Current end-to-end autonomous driving models are fundamentally constrained by the behavioral cloning ceiling of imitation learning. While reinforcement learning offers a path to smarter autonomy, it demands two missing pieces of infrastructure: (1) a cognitive foundation that understands traffic semantics and driving intent, and (2) a foresighted physical environment that can anticipate the consequences of candidate actions. To this end, we propose CoPhy, a CognitivePhysical reinforcement learning framework for autonomous driving. To distill to think, we distill VLM knowledge into the BEV encoder and then discard the VLM entirely, retaining cognitive ability at zero inference cost while releasing the cognitive channel as a pluggable interface for optional human language commands. To foresee to act, we build an auto-regressive BEV world model that explicitly predicts future semantic maps conditioned on candidate actions, serving
    
[^327]: 预测-干预博弈与不变集

    Prediction-Intervention Games and Invariant Sets

    [https://arxiv.org/abs/2605.16828](https://arxiv.org/abs/2605.16828)

    该论文提出预测-干预博弈（Stackelberg博弈的特例），并证明对于两类常见的跟随者干预目标，基于稳定毯这一不变子集的预测器性能至少与基于因果父节点的预测器相当。

    

    我们考虑以下双人博弈：领导者利用观测数据，从给定协变量中为响应变量 $Y$ 选择一个预测函数。随后，跟随者通过对底层结构因果模型中的某些协变量进行干预来做出反应，以最大化自身的目标。领导者知道干预的目标变量，但对跟随者的目标可能了解有限。我们将这种设定称为预测-干预博弈，它是Stackelberg博弈的一种特例。寻找领导者的最优策略通常是困难的。为避免严重的性能损失，领导者可以将其预测建立在 $Y$ 的因果父节点之上，或更一般地建立在协变量的某个不变子集之上。我们证明，对于两类常见的跟随者目标，基于稳定毯（一种不变子集）的预测器至少与基于因果父节点的预测器一样好。我们进一步对领导者干预后……

    arXiv:2605.16828v2 Announce Type: replace-cross  Abstract: We consider the following two-player game: using observational data, the leader chooses a prediction function for a response variable $Y$ from given covariates. The follower then reacts with an intervention on some covariates in the underlying structural causal model to maximize their own objective. The leader knows the intervention targets, but may have limited knowledge of the follower's objective. We call this setup a prediction-intervention game, a special case of a Stackelberg game. Finding an optimal strategy for the leader is generally difficult. To avoid severe performance loss, the leader may base their prediction on the causal parents of $Y$, or more generally on an invariant subset of covariates. We prove, for two common classes of follower objectives, that predictors based on the stable blanket, an invariant subset, are at least as good as those based on the causal parents. We further upper bound the leader's post-i
    
[^328]: 通过分子引导的特征自适应弥合病理多实例学习中的模态瓶颈

    Bridging the Modality Bottleneck in Pathology MIL through Molecularly Guided Feature Adaptation

    [https://arxiv.org/abs/2605.16392](https://arxiv.org/abs/2605.16392)

    该论文提出MIST，一种即插即用的MIL投影层替代方法，仅在训练时利用配对空间转录组学构建虚拟分子染色，沿分子引导的轴重组H&E特征，从而弥合病理MIL中形态学与分子信息之间的模态瓶颈，且推理时无需转录组学数据。

    

    多实例学习（MIL）是计算病理学中全切片图像分析的主流框架，通常由冻结的图像块编码器、一个投影层和一个切片级聚合器组成。尽管编码器和聚合器已被广泛研究，投影层仍然主要是一个仅基于形态学的瓶颈。这限制了诸如生物标志物状态和生存期等预测终点，因为这些终点由分子状态决定，而H&E形态学并不能完全捕捉这种分子状态。我们提出了分子信息染色变换（MIST），这是一种MIL投影层的即插即用替代方案，仅在训练阶段使用配对的空间转录组学数据来构建虚拟分子染色。MIST将基因表达谱聚类为跨模态原型，将其锚定在冻结的基础模型特征空间中，并利用它们沿分子引导的轴重组H&E图像块特征。它在推理阶段无需转录组学数据。

    arXiv:2605.16392v2 Announce Type: replace-cross  Abstract: Multiple instance learning (MIL) is the dominant framework for whole-slide image analysis in computational pathology, typically combining a frozen patch encoder, a projection layer, and a slide-level aggregator. While encoders and aggregators have been extensively studied, the projection layer remains a largely morphology-only bottleneck. This limits endpoints such as biomarker status and survival, which are governed by a molecular state that is not fully captured by H&E morphology. We introduce Molecularly Informed Staining Transform (MIST), a plug-in replacement for the MIL projection layer that uses paired spatial transcriptomics only during training to construct virtual molecular stains. MIST clusters gene expression profiles into cross-modal prototypes, anchors them in the frozen foundation model feature space, and uses them to reorganize H&E patch features along molecularly guided axes. It requires no transcriptomics at i
    
[^329]: NodeSyn：面向AI评估的社会对齐合成数据

    NodeSynth: Socially Aligned Synthetic Data for AI Evaluation

    [https://arxiv.org/abs/2605.14381](https://arxiv.org/abs/2605.14381)

    NodeSyn通过将微调的分类体系生成器与可信文献来源和敏感属性相结合，生成反映真实社会背景的敏感领域评估数据，其诱发的模型失败率达到通用合成基线的3.1倍、真实人类查询的1.6倍，显著优于现有合成数据方法。

    

    生成式人工智能的最新进展为模型评估所需的大规模合成数据生成提供了便利。然而，若缺乏针对性的方法，这些数据集往往缺乏敏感领域所需的社会技术层面的细微差别。我们提出了NodeSyn，这是一种将经过微调的分类体系生成器（TaG）与可信的已发表文献来源以及提取的敏感属性相结合的方法，用于生成反映真实社会背景和细微危害的查询。在使用经人工验证的自动评估器，于医疗建议和自我伤害两个领域对十个开源及专有大语言模型进行评估时，NodeSyn在10个模型中的8个上于每个领域均排名第一。NodeSyn数据集引发的平均失败率是通用合成基线的3.1倍，是结构化合成基准的2.4倍，是真实人类撰写查询的1.6倍。消融研究证实，在控制结构混杂因素的情况下，我们细粒度的第3级分类体系…

    arXiv:2605.14381v3 Announce Type: replace-cross  Abstract: Recent advancements in generative AI facilitate large-scale synthetic data generation for model evaluation. However, without targeted approaches, these datasets often lack the sociotechnical nuance required for sensitive domains. We introduce NodeSyn, a methodology that pairs a fine-tuned taxonomy generator (TaG) with credible published sources and extracted sensitive attributes to generate queries reflecting authentic societal contexts and nuanced harms. Evaluated across ten open-weight and proprietary LLMs in Medical Advice and Self-Harm using human-validated autoraters, NodeSyn ranked first on 8 of 10 models in each domain. NodeSyn datasets induced average failure rates 3.1 times higher than generic synthetic baselines, 2.4 times higher than structured synthetic benchmarks, and 1.6 times higher than authentic human-authored queries. Ablation studies confirm that, controlling for structural confounds, our granular Level 3 tax
    
[^330]: 利用语言模型先验从观测中学习POMDP世界模型

    Learning POMDP World Models from Observations with Language-Model Priors

    [https://arxiv.org/abs/2605.13740](https://arxiv.org/abs/2605.13740)

    提出Pinductor方法，利用大语言模型先验从少量观测-动作轨迹中提出并迭代优化POMDP世界模型，在不使用隐藏状态的情况下达到与拥有特权信息方法相当的性能和样本效率。

    

    无论是 navigating 一栋建筑、操作机器人还是玩游戏，一个想要在环境中有效行动的智能体必须首先学习该环境如何运作的内部模型。部分可观测马尔可夫决策过程（POMDP）为这类内部世界模型提供了灵活的建模框架，但仅从观测-动作轨迹中学习它们极具挑战性，通常需要大量的环境交互。我们探讨语言模型先验能否通过利用先验知识来减少代价高昂的交互，并提出了Pinductor（POMDP归纳器）：由大语言模型从少量观测-动作轨迹中提出候选POMDP模型，并迭代地对其进行优化，以最大化基于信念的似然得分。尽管严格而言所依赖的信息更少，Pinductor仍取得了与拥有隐藏状态特权信息的基于LLM的POMDP学习方法相当的性能和样本效率。此外，它最多可带来33

    arXiv:2605.13740v2 Announce Type: replace  Abstract: Whether navigating a building, operating a robot, or playing a game, an agent that acts effectively in an environment must first learn an internal model of how that environment works. Partially-observable Markov decision processes (POMDPs) provide a flexible modeling class for such internal world models, but learning them from observation-action trajectories alone is challenging and typically requires extensive environment interaction. We ask whether language-model priors can reduce costly interaction by leveraging prior knowledge, and introduce Pinductor (POMDP-inductor): an LLM proposes candidate POMDP models from a few observation-action trajectories and iteratively refines them to optimize a belief-based likelihood score. Despite relying on strictly less information, Pinductor achieves comparable performance and sample efficiency to LLM-based POMDP learning methods with privileged access to hidden states. Moreover, it is up to 33
    
[^331]: GraphIP-Bench：窃取一个图神经网络有多难？我们能否阻止它？

    GraphIP-Bench: How Hard Is It to Steal a Graph Neural Network, and Can We Stop It?

    [https://arxiv.org/abs/2605.12827](https://arxiv.org/abs/2605.12827)

    提出了首个统一基准GraphIP-Bench，在单一黑盒协议下系统评估12种模型提取攻击与12种GNN知识产权防御方法（覆盖10个图数据集、3种骨干网络和3类任务），首次可靠地回答了窃取GNN的难度以及防御方法能否有效阻止窃取这两个问题。

    

    部署为云服务的图神经网络（GNN）可能通过*模型提取攻击*被“窃取”——攻击者利用查询响应训练代理模型来复现目标模型的行为，而不断涌现的所有权防御方法则试图阻止或追踪此类窃取行为。本文的标题提出了两个问题：*窃取一个GNN有多难？*以及*我们能否阻止它？*由于以往实验使用的数据集、威胁模型和评估指标不一致，先前的工作无法回答这两个问题。我们提出了*GraphIP-Bench*，这是一个在统一黑盒协议下同时评估攻防双方的基准。它集成了十二种提取攻击、横跨水印、输出扰动和查询模式检测三大类的十二种防御方法、十个覆盖同配、异配和大规模场景的公开图数据集、三种GNN骨干网络以及三种图学习任务，并报告保真度、任务效用、所有权验证和计算成本等指标。

    arXiv:2605.12827v3 Announce Type: replace-cross  Abstract: Graph neural networks (GNNs) deployed as cloud services can be *stolen* through *model-extraction attacks*, which train a surrogate from query responses to reproduce the target's behaviour, and a growing line of ownership defenses tries to prevent or trace such theft. The title of this paper asks two questions: *how hard is it to steal a GNN?*, and *can we stop it?* Prior work cannot answer either, because experiments use inconsistent datasets, threat models, and metrics. We introduce *GraphIP-Bench*, a unified benchmark which evaluates both sides under a single black-box protocol. It integrates twelve extraction attacks, twelve defenses spanning watermarking, output-perturbation, and query-pattern-detection families, ten public graphs covering homophilic, heterophilic, and large-scale regimes, three GNN backbones, and three graph-learning tasks, and it reports fidelity, task utility, ownership verification, and computational c
    
[^332]: EPIC：面向组合式文生图生成的高效谓词引导推理时控制

    EPIC: Efficient Predicate-Guided Inference-Time Control for Compositional Text-to-Image Generation

    [https://arxiv.org/abs/2605.11722](https://arxiv.org/abs/2605.11722)

    EPIC是一个无需训练的推理时精炼框架，通过将提示解析为谓词视觉程序并基于视觉证据验证与定向修正生成图像，大幅提升了文生图模型处理多对象、数量、属性和关系等组合式提示的能力。

    

    近期的文生图（T2I）生成器能够合成逼真的图像，但在处理涉及多个对象、数量、属性和关系的组合式提示时仍然存在困难。我们提出了EPIC（高效谓词引导的推理时控制），这是一个无需训练的推理时精炼框架，用于组合式文生图生成。EPIC将精炼过程转化为谓词引导的搜索：它将原始提示一次性解析为由对象变量和类型化谓词组成的视觉程序，然后利用从候选图像中提取的视觉证据对每个候选图像进行验证。当所有谓词均通过时，该候选图像被判定为满足提示；否则，未满足的谓词将引导下一步操作，将局部性失败路由至针对性的编辑操作，而将布局层面的或持续存在的失败路由至重新采样。在GenEval2基准上，EPIC将提示级Soft-TIFA GM分数从单次生成的34.16%提升至71.46%。

    arXiv:2605.11722v2 Announce Type: replace-cross  Abstract: Recent text-to-image (T2I) generators can synthesize realistic images, but still struggle with compositional prompts involving multiple objects, counts, attributes, and relations. We introduce EPIC (Efficient Predicate-Guided Inference-Time Control), a training-free inference-time refinement framework for compositional T2I generation. EPIC casts refinement as predicate-guided search: it parses the original prompt once into a visual program of object variables and typed predicates, then verifies each candidate image against this program using visual evidence extracted from that image. A candidate is judged to satisfy the prompt when all predicates pass; otherwise, unmet predicates guide the next step, routing localized failures to targeted editing and layout-level or persistent failures to resampling. On GenEval2, EPIC improves the prompt-level Soft-TIFA GM score from 34.16% for single-pass generation to 71.46%. With the same ge
    
[^333]: 智能体文献抽取增强靶向蛋白降解数据库

    Agentic Literature Extraction Augments Targeted Protein Degradation Databases

    [https://arxiv.org/abs/2605.11221](https://arxiv.org/abs/2605.11221)

    本研究提出专家在环的大语言模型工作流程，仅用七篇标注文献即实现高精度（F1=0.98）的靶向蛋白降解实验数据自动抽取，构建的AutoTPD数据库比现有参考数据库多出81%至92%的记录。

    

    药物化学中的预测模型依赖于锁定在原始出版物中的结构化实验数据。这一瓶颈在靶向蛋白降解（TPD）领域尤为突出，因为每条实验记录都需要整合化合物身份、降解靶点、招募者、实验背景以及分布在正文各章节、表格和补充文件中的终点数值。现有的TPD数据库均由人工整理，且往往缺乏实验背景信息。我们提出了一种专家在环的大型语言模型工作流程，其中交叉验证的提示词优化模块能够从稀缺的专家标注中自适应调整抽取指令。仅用七篇标注的分子胶文献，该工作流程就达到了记录级F1 = 0.98的抽取性能，并且仅通过术语替换就成功迁移到了蛋白水解靶向嵌合体（PROTACs）领域（F1 ≥ 0.93）。所构建的数据库AutoTPD比标准化参考数据库多包含81%和92%的分子胶和PROTAC记录。

    arXiv:2605.11221v2 Announce Type: replace-cross  Abstract: Predictive models in medicinal chemistry depend on structured assay data locked in primary publications. This bottleneck is acute in targeted protein degradation (TPD), where each assay record must combine compound identity, degradation target, recruiter, assay context, and endpoint values reported across sections, tables, and supplementary files. Existing TPD databases are manually curated and often lack assay context. We present an expert-in-the-loop large language model workflow in which a cross-validated prompt-refinement module adapts extraction instructions from scarce expert annotations. With only seven annotated molecular glue publications, the workflow achieved record-level $F_1 = 0.98$ and transferred to proteolysis-targeting chimeras (PROTACs) by terminology substitution alone ($F_1 \geq 0.93$). The resulting database, AutoTPD, contained 81% and 92% more molecular glue and PROTAC records than standardized reference d
    
[^334]: 预测任务中传感器故障鲁棒性的基准测试

    Benchmarking Sensor-Fault Robustness in Forecasting

    [https://arxiv.org/abs/2605.10822](https://arxiv.org/abs/2605.10822)

    该论文提出了SensorFault-Bench，一个面向信息物理系统的传感器故障鲁棒性基准，通过受控合成压力测试在四个真实数据集和八个标准化评分场景下评估预测模型对孤立输入故障的鲁棒性，并将相对鲁棒性与绝对误差区分开来。

    

    信息物理系统（CPS）预测模型依赖于可能存在噪声、偏差、缺失或时间错位的传感器数据流，然而标准的预测评估通常仅凭名义误差来选择模型，而未展示这些模型在此类故障下是否仍然保持鲁棒。我们提出SensorFault-Bench，这是一个以CPS为基础的评估协议，通过对孤立输入故障进行受控的合成压力测试来评估各类预测架构和鲁棒性改进方法，并提供了一套方法的操作性分类体系。该基准涵盖四个真实世界数据集和八个由标准化严重程度模型控制的评分场景，报告最坏场景下的性能退化、干净数据下的均方误差（MSE）以及最坏场景下故障发生时的MSE，从而将相对鲁棒性与绝对误差区分开来。显式故障训练方法在与评分场景不相交的相邻故障族上进行训练。实证结果表明，在干净MSE上表现更优的预测架构……（原文摘要至此截断）

    arXiv:2605.10822v2 Announce Type: replace  Abstract: Cyber-physical system (CPS) forecasting models depend on sensor streams with noisy, biased, missing, or temporally misaligned readings, yet standard forecasting evaluation often selects models by nominal error without showing whether they remain robust under such faults. We introduce SensorFault-Bench, a CPS-grounded protocol that evaluates forecasting architectures and robustness-improvement methods through controlled synthetic stress tests of isolated input faults, together with an operational taxonomy of the methods. Across four real-world datasets and eight scored scenarios governed by a standardized severity model, it reports worst-scenario degradation, clean mean squared error (MSE), and worst-scenario fault-time MSE, separating relative robustness from absolute error. Explicit fault-training methods train on adjacent fault families disjoint from the scored scenarios. Empirically, forecasting architectures favored by clean MSE 
    
[^335]: 线性联想记忆中的事实回忆：精确渐近分析与机理洞察

    Factual recall in linear associative memories: sharp asymptotics and mechanistic insights

    [https://arxiv.org/abs/2605.10795](https://arxiv.org/abs/2605.10795)

    本文通过引入一个与原模型等价的解耦模型，精确刻画了线性联想记忆在事实回忆任务中可存储关联数量的容量极限 $p_c$，并对学习解的结构给出了机理层面的洞察。

    

    大型语言模型在事实回忆方面表现出色，但神经网络存储输入-输出关联的基本极限仍不清楚。我们在一个最小化设置中研究这些极限：一个线性联想记忆，通过线性层将 $\mathbb{R}^d$ 中的 $p$ 个输入嵌入映射到其 $d$ 维目标，要求每个映射后的输入对其自身目标的得分高于所有其他输出。由于所有输入都在竞争相同的输出，这些约束之间存在强相关性，这使得存储容量 $p_c$——即能够存储的最大关联数量——难以刻画。我们分三步解决这个问题。首先，我们引入一个解耦模型，其中每个输入与其自身独立的输出集合竞争，并提供数值和分析证据表明该模型在容量和学习解的结构上都与原始模型等价。其次，利用统计……（摘要原文在此处截断）

    arXiv:2605.10795v2 Announce Type: replace-cross  Abstract: Large language models excel at factual recall, yet the fundamental limits of storing input--output associations with neural networks remain unclear. We study these limits in a minimal setting: a linear associative memory that maps $p$ input embeddings in $\mathbb{R}^d$ to their $d$-dimensional targets via a linear layer, requiring each mapped input to score its own target above all other outputs. Since all inputs compete for the same outputs, these constraints are strongly correlated, which makes the storage capacity $p_c$ --- the largest number of associations that can be stored --- hard to characterise. We address this in three steps. First, we introduce a decoupled model, in which each input competes with its own independent set of outputs, and provide numerical and analytical evidence that it is equivalent to the original model, both in capacity and in the structure of the learnt solution. Second, using tools from statistic
    
[^336]: 格演绎Transformer

    Lattice Deduction Transformers

    [https://arxiv.org/abs/2605.08605](https://arxiv.org/abs/2605.08605)

    格演绎Transformer（LDT）通过在前向传递之间以格投影潜在状态来近似逻辑可靠的演绎推理，仅用80万参数和不到现有模型2%的训练计算量，就以近乎完全可靠的方式解决了99.999%的极难数独谜题。

    

    我们提出了格演绎Transformer（LDT），这是一种循环Transformer，通过在前向传递之间将其潜在状态投影到一个格上，来近似逻辑上可靠的演绎推理。我们在一个与基于搜索的约束求解器中的演绎过程相镜像的流程中进行在线训练，并通过基于抽象解释的、与领域无关的解候选集近似方法来监督训练。一个80万参数的LDT在Sudoku-Extreme（极难数独）基准上超越了以往的小型循环推理模型，且训练所需的估计浮点运算量不到后者的2%，解决了422,786个测试谜题中的99.999%。该模型几乎总是可靠的：在127万次谜题评估中仅给出4次错误答案，其余情况要么返回正确答案，要么选择弃权。一个180万参数的LDT变体在Maze-Hard（高难度迷宫）上达到了99.8%的准确率。我们的代码发布于 github.com/lcrh/lattice-deduction-transformers。

    arXiv:2605.08605v2 Announce Type: replace-cross  Abstract: We introduce the Lattice Deduction Transformer (LDT), a recurrent transformer that approximates logically sound deduction by projecting its latent state through a lattice between forward passes. We train on-policy in a process that mirrors deduction in a search-based constraint solver and supervise training via a domain-agnostic approximation of the set of solution candidates based on abstract interpretation. An $800$K-parameter LDT outperforms prior small recurrent reasoners on Sudoku-Extreme with under $2\%$ of their estimated training FLOPs, solving $99.999\%$ of the $422{,}786$ test puzzles. The model is nearly always sound, in that it gives a wrong answer for $4$ of $1.27$M puzzle evaluations, and otherwise returns a correct answer or abstains. A $1.8$M-parameter LDT variant reaches $99.8\%$ accuracy on Maze-Hard. Our code is available at github.com/lcrh/lattice-deduction-transformers.
    
[^337]: 以KL换KL：基于控制变量基线的在线策略蒸馏

    KL for a KL: On-Policy Distillation with Control Variate Baseline

    [https://arxiv.org/abs/2605.07865](https://arxiv.org/abs/2605.07865)

    提出vOPD方法，将在线策略蒸馏重构为策略梯度强化学习，并利用控制变量基线来降低梯度方差，其中该基线的闭式解为师生模型间逐token的负反向KL散度，可直接从现有前向传播中获得，无需额外开销即可稳定训练。

    

    在线策略蒸馏已成为大语言模型后训练的主导范式，尤其是在推理领域。然而，由于其单样本蒙特卡洛估计器具有高梯度方差，OPD在实践中仍然不稳定，且实现稳定训练的方法尚不成熟。我们提出了vOPD（带有控制变量基线的在线策略蒸馏），该方法将OPD视为策略梯度强化学习，并通过引入来自强化学习文献的控制变量基线（通常是一个价值函数）来稳定训练过程。我们证明OPD价值函数存在闭式解：即学生模型与教师模型之间逐token的负反向KL散度，该数值可直接从现有的前向传播中获得，无需额外的评论家网络或推理过程。现有的稳定化方法要么需要在整个词表上计算完整的token级反向KL散度，带来显著的开销，要么将其限制在top-k支持集上，从而引入偏差。

    arXiv:2605.07865v2 Announce Type: replace-cross  Abstract: On-Policy Distillation (OPD) has emerged as a dominant post-training paradigm for Large Language Models, especially in reasoning domains. However, OPD remains unstable in practice due to the high gradient variance of its single-sample Monte Carlo estimator, and recipes for stable training are still immature. We propose vOPD (On-Policy Distillation with a control variate baseline), which casts OPD as policy-gradient RL and stabilizes it by introducing a control variate baseline---canonically a value function---from the RL literature. We show that the OPD value function admits a closed form: the per-token negative reverse KL divergence between student and teacher, available directly from the existing forward pass with no additional critic or inference. Existing stabilization methods either compute the full token-level reverse KL over the entire vocabulary, adding significant overhead, or restrict it to a top-k support, biasing th
    
[^338]: 通过推理空间压缩的结构化推理依据蒸馏

    Structural Rationale Distillation via Reasoning Space Compression

    [https://arxiv.org/abs/2605.07139](https://arxiv.org/abs/2605.07139)

    提出D-RPC方法，通过让教师模型遵循一个紧凑且动态维护的可复用推理路径库来压缩推理空间，使相似问题的推理依据结构一致，从而为学生模型提供更干净一致的蒸馏监督。

    

    将大型语言模型（LLM）的推理能力蒸馏到较小模型时，教师模型对相似问题给出的推理依据在结构和策略上往往差异巨大。就像厨师每次做同一道菜的方式都不同一样，这种不一致性给学生模型带来了难以内化的噪声监督。我们提出了“通过推理路径压缩进行蒸馏”的方法（D-RPC），该方法约束教师模型遵循一个紧凑的、动态维护的可复用高层推理路径库。对于每个训练问题，D-RPC检索最相关的路径并引导教师模型遵循该路径，从而产生在相似问题之间保持一致、同时又能覆盖不同问题类型的多样化推理依据。PAC-Bayes分析形式化了路径库大小与覆盖率之间的权衡：较小的路径库会限制路径多样性，从而带来覆盖缺口的风险。我们的消融实验倾向于中等大小的路径库，这与该理论分析相一致。

    arXiv:2605.07139v2 Announce Type: replace-cross  Abstract: When distilling reasoning from large language models (LLMs) into smaller ones, teacher rationales for similar problems often vary wildly in structure and strategy. Like a chef who makes the same dish differently each time, this inconsistency burdens the student with noisy supervision that is hard to internalize. We propose Distillation through Reasoning Path Compression (D-RPC), which constrains the teacher to follow a compact, dynamically maintained bank of reusable high-level reasoning paths. For each training question, D-RPC retrieves the most relevant path and conditions the teacher to follow it, producing rationales that are consistent across similar problems yet diverse enough to cover different problem types. A PAC-Bayes analysis formalizes a trade-off between bank size and coverage, with smaller banks limiting path diversity at the risk of coverage gaps. Our ablations favor an intermediate bank size, consistent with thi
    
[^339]: 基于希尔伯特丛与胞腔层的一致几何深度学习

    Consistent Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves

    [https://arxiv.org/abs/2605.06395](https://arxiv.org/abs/2605.06395)

    该论文提出了 HilbNets——一种以希尔伯特丛的联络拉普拉斯算子作为卷积算子的新型卷积学习框架，可处理流形上可能无穷维的信号，并通过采样诱导的希尔伯特胞腔层使卷积运算得以实际实现。

    

    现代深度学习架构越来越多地需要处理本质上为无穷维的复杂信号，例如时间序列、概率分布或算子，这些信号定义在不规则的域上。然而，针对这些情形一直缺乏统一的学习理论。为开始填补这一空白，我们针对定义在流形上的、可能为无穷维的信号，提出了一种新颖的卷积学习框架。具体而言，我们将与希尔伯特丛相关联的联络拉普拉斯算子用作卷积算子，并由此推导出滤波器和神经网络，称为 HilbNets。我们通过一个两阶段的采样过程使 HilbNets 以及更一般的卷积运算得以实现。首先，我们证明对流形进行采样会诱导出一个希尔伯特胞腔层——这是一种具有希尔伯特特征空间和逐边耦合规则的广义图结构，并且我们证明了其层拉普拉斯算子收敛……（摘要在此处被截断）

    arXiv:2605.06395v3 Announce Type: replace-cross  Abstract: Modern deep learning architectures increasingly contend with sophisticated signals that are natively infinite-dimensional, such as time series, probability distributions, or operators, and are defined over irregular domains. Yet, a unified learning theory for these settings has been lacking. To start addressing this gap, we introduce a novel convolutional learning framework for possibly infinite-dimensional signals supported on a manifold. Namely, we use the connection Laplacian associated with a Hilbert bundle as a convolutional operator, and we derive filters and neural networks, dubbed as \textit{HilbNets}. We make HilbNets and, more generally, the convolution operation, implementable via a two-stage sampling procedure. First, we show that sampling the manifold induces a Hilbert Cellular Sheaf, a generalized graph structure with Hilbert feature spaces and edge-wise coupling rules, and we prove that its sheaf Laplacian conver
    
[^340]: 面向非参数因果效应估计的数据驱动且与环路无关的协变量选择方法

    Data-Driven and Cycle-Agnostic Covariate Selection for Nonparametric Causal Effect Estimation

    [https://arxiv.org/abs/2605.06385](https://arxiv.org/abs/2605.06385)

    该论文证明基于条件独立性的局部数据驱动协变量选择方法的有效性保证可从无环因果模型扩展到含环因果模型，从而为非参数因果效应估计建立了统一的、与环路无关的协变量选择框架。

    

    从观测数据中估计因果效应需要识别有效的调整集。在存在潜在混杂因素和反馈环路的现实场景中，这一任务尤为困难。现有方法通常假设因果结构无环，或依赖全局因果结构学习，从而限制了其适用性和计算效率。在本工作中，我们研究了一种局部的、数据驱动的协变量选择方法，该方法在预处理假设下基于条件独立性信息进行协变量选择。虽然已知该方法在无环因果模型中是可靠且完备的，但其在存在环路情形下的有效性一直不明确。我们的主要贡献是证明了这些理论保证可以扩展到含环路的因果模型。特别地，我们的结果依赖于条件独立性断言在σ-去环化（σ-acyclification）下的不变性。这些发现为协变量选择建立了一个统一的、与环路无关的视角。

    arXiv:2605.06385v2 Announce Type: replace  Abstract: Estimating causal effects from observational data requires identifying valid adjustment sets. This task is especially challenging in realistic settings where latent confounding and feedback loops are present. Existing approaches typically assume acyclicity or rely on global causal structure learning, limiting applicability and computational efficiency. In this work, we study a local, data-driven method for covariate selection based on conditional independence information under a pre-treatment assumption. While this method is known to be sound and complete in acyclic causal models, its validity in the presence of cycles has remained unclear. Our main contribution is to show that these guarantees extend to cyclic causal models. In particular, our result relies on the invariance of conditional independence assertions under $\sigma$-acyclification. These findings establish a unified, cycle-agnostic perspective on covariate selection, whi
    
[^341]: 序数信用分配：仅凭比较的多步强化学习

    Ordinal Credit Assignment: Multi-Step Reinforcement Learning from Comparisons Alone

    [https://arxiv.org/abs/2605.06375](https://arxiv.org/abs/2605.06375)

    该论文提出一种在多步任务中仅依靠轨迹间两两比较（无需奖励函数或奖励模型）的序数信用分配方法，用动作在同一时间步群体中的胜率替代未来累积奖励，证明其期望更新等价于策略梯度，且在确定性环境中按真实价值排序、在随机环境中偏好高胜率动作，并对标签噪声和位置偏差具有鲁棒性。

    

    策略梯度方法需要一个标量奖励，然而对许多任务而言，没有任何公式能够刻画我们真正想要的目标，而判断两个结果中哪个更好却很容易。我们研究在多步任务中仅接收此类比较信号的强化学习——不需要奖励函数，也不需要奖励模型。多组轨迹从共享状态出发，每个动作的信用根据其未来累积奖励与同一时间步上其他轨迹的比较结果来分配。我们证明，期望更新等价于一种策略梯度，其中未来累积奖励被替换为该动作在当时仍在运行的轨迹群体中的胜率，并按对手仍在运行的概率进行加权；在确定性环境中，该更新按真实价值对动作排序，且最优策略是稳定的；而在随机结果下，它更偏好胜率更高的动作，而非均值更高的动作。平局附近的标签噪声或由位置偏差引起的噪声不会改变期望更新；只

    arXiv:2605.06375v2 Announce Type: replace-cross  Abstract: Policy-gradient methods need a scalar reward, yet for many tasks no formula captures what we want, while judging which of two outcomes is better is easy. We study reinforcement learning that receives only such comparisons - no reward function, no reward model - in multi-step tasks. Groups of trajectories start from a shared state, and each action is credited by how its reward-to-go compares with those of other trajectories at the same time step. We show the expected update is policy gradient with reward-to-go replaced by its win rate against the population still running at that time, weighted by the chance an opponent is still running; in deterministic environments the update ranks actions by true value and the optimal policy is stable, whereas under stochastic outcomes it prefers actions that win more often rather than those with higher mean. Label noise near ties or from position bias leaves the expected update unchanged; onl
    
[^342]: LINC：在构造式神经路由中将局部后果评分与隐式匹配解耦

    LINC: Decoupling Local Consequence Scoring from Hidden Matching in Constructive Neural Routing

    [https://arxiv.org/abs/2605.06332](https://arxiv.org/abs/2605.06332)

    LINC提出一种解码器端候选决策架构，将确定性单步后果的显式计算与重要性学习解耦，使构造式神经路由在保留缓存嵌入和全局匹配的同时，能直接基于确定性转移量对候选动作评分。

    

    构造式神经路由求解器通常通过将解码器上下文与候选嵌入进行匹配来为下一个动作评分，而这些候选嵌入是基于静态节点特征（如坐标、需求和时间窗）在每实例中一次性计算得到并缓存的；确定性的单步后果（如行驶、等待、松弛和容量变化）并未作为输入提供，必须由模型隐式推断。我们提出LINC，这是一种解码器端的候选决策架构，它将“计算这些后果”与“学习它们的重要性”解耦。环境在每一步显式计算这些后果；候选级别的后果由状态条件化的共享线性比较器进行评分，而可行集摘要则对解码器上下文进行调制。这样既保留了缓存的嵌入和标准的全局匹配机制，又使评分函数能够直接访问确定性的转移量。带时间窗的容量约束车辆路径问题

    arXiv:2605.06332v3 Announce Type: replace  Abstract: Constructive neural routing solvers usually score the next action by matching a decoder context against candidate embeddings computed once per instance from static node features such as coordinates, demands, and time windows; deterministic one-step consequences such as travel, waiting, slack, and capacity changes are not provided as inputs and must be inferred implicitly by the model. We propose LINC, a decoder-side candidate decision architecture that decouples computing these consequences from learning how much they matter. The environment computes the consequences explicitly at each step; candidate-level consequences are scored by a state-conditioned shared linear comparator, while feasible-set summaries modulate the decoder context. This keeps the cached embeddings and standard global matching while giving the scoring function direct access to deterministic transition quantities. The Capacitated Vehicle Routing Problem with Time 
    
[^343]: 基于不可靠反馈的上下文黑盒优化

    In-Context Black-Box Optimization with Unreliable Feedback

    [https://arxiv.org/abs/2605.06187](https://arxiv.org/abs/2605.06187)

    该论文提出反馈感知的上下文黑盒优化框架FICBO，通过结构化先验预训练Transformer，使其能同时利用优化历史和可能不可靠的辅助反馈，从而在反馈质量参差不齐的情况下实现稳健的跨任务优化。

    

    科学与工程中的黑盒优化通常伴随着辅助信息：专家、模拟器、预训练预测器或启发式方法可以提出有前景的候选方案。这类反馈能够加速搜索，但也可能存在偏差、依赖于输入或具有误导性。感知反馈的贝叶斯优化方法通常独立处理各个任务，限制了跨反馈源的适应能力；而上下文优化器虽然能够实现跨任务适应，但在测试时通常仅依赖优化历史。我们研究了基于反馈信息的上下文黑盒优化（FICBO），其中预训练的优化器同时以观测历史和当前候选集的廉价辅助反馈为条件。我们引入了关于辅助反馈与目标之间关系的结构化先验，并利用它们在从高度信息丰富到具有误导性的各种情形下预训练一个感知反馈的 Transformer。在测试时，该模型估计……（原文摘要截断于此）

    arXiv:2605.06187v2 Announce Type: replace-cross  Abstract: Black-box optimization in science and engineering often comes with side information: experts, simulators, pretrained predictors, or heuristics can suggest promising candidates. Such feedback can accelerate search, but can also be biased, input-dependent, or misleading. Feedback-aware Bayesian optimization methods typically handle tasks independently, limiting adaptation across feedback sources. In-context optimizers enable cross-task adaptation, but usually rely only on optimization history at test time. We study feedback-informed in-context black-box optimization (FICBO), where a pretrained optimizer conditions on both the observed history and cheap auxiliary feedback for the current candidate set. We introduce structured priors over how auxiliary feedback relates to the objective, and use them to pretrain a feedback-aware transformer across regimes ranging from highly informative to misleading. At test time, the model estimat
    
[^344]: 估计深度学习中的隐式正则化

    Estimating Implicit Regularization in Deep Learning

    [https://arxiv.org/abs/2605.05436](https://arxiv.org/abs/2605.05436)

    本文提出基于梯度匹配的估计方法，利用正则化导致权重更新偏离损失梯度这一信号，来量化深度学习系统（包括早停、小批量、dropout 等修改）中隐式正则化的效果。

    

    众所周知，深度学习系统会表现出隐式正则化（也称隐式偏差），即倾向于选择简单的解，而不仅仅是最小化损失函数。在某些情况下，我们可以通过解析方法推导出这种隐式正则化——将其与增广学习目标的等价惩罚项联系起来。然而，现代深度学习系统十分复杂，包含对训练过程和架构的各种修改（如早停、小批量训练、dropout 等），这些修改的效果并不总是能够被直接解释。尽管估计由此产生的隐式正则化既有助于理论研究者进行算法设计，也有助于实践者理解其超参数选择，但这一问题很少受到直接关注。同时，这一问题也是可处理的：正则化会使权重更新偏离损失梯度，这为识别隐式偏差提供了有希望的信号。在此，我们提出了一类梯度匹配方法，可用于……

    arXiv:2605.05436v2 Announce Type: replace-cross  Abstract: Deep learning systems are known to exhibit implicit regularization (alt. implicit bias), favoring simple solutions instead of merely minimizing the loss function. In some cases, we can analytically derive the implicit regularization -- connecting it to an equivalent penalty that augments the learning objective. However, modern deep learning systems are complex, carrying modifications to the training procedure and architecture (e.g. early stopping, minibatching, dropout) whose effects are not always directly interpretable. Although estimating the resulting implicit regularization could aid theorists in algorithm design and practitioners in interpreting their hyperparameter choices, this problem has received little direct attention. It is also tractable: regularization makes weight updates deviate from loss gradients, promising a signal for identifying implicit bias. Here we provide gradient matching methods that can be used to e
    
[^345]: 超维指纹作为分子表示

    Hyper-Dimensional Fingerprints as Molecular Representations

    [https://arxiv.org/abs/2604.27810](https://arxiv.org/abs/2604.27810)

    该论文提出了一种无需训练的超维指纹（HDF），通过高维向量代数运算组合原子化学属性与多跳环境信息，生成确定性分子表示，在性质预测基准上表现优异，且在低维度下比传统指纹更高效地编码分子结构。

    

    数值化分子表示是虚拟筛选、性质预测和材料发现的基础。传统指纹方法高效且具有确定性，但在低维度下会因基于哈希的压缩而丢失结构信息；而图神经网络虽能恢复表达能力，却需要计算密集、针对特定任务的训练。本文提出了超维指纹（HDF），通过对高维向量进行代数运算来组合原子的化学属性及其多跳环境信息，无需任何训练即可生成确定性的分子表示。在多种性质预测基准测试中，HDF在各个数据集和模型上均始终位列最佳表示方法之列。至关重要的是，HDF能更高效地编码分子结构：在低维度下，其距离与图编辑距离的相关性比Morgan指纹更强，从而提供了有效的相似性度量。

    arXiv:2604.27810v3 Announce Type: replace  Abstract: Numerical molecular representations underpin virtual screening, property prediction, and materials discovery. Conventional fingerprints are efficient and deterministic but lose structural information through hash-based compression at low dimensionalities, whereas graph neural networks recover expressiveness through compute-intensive, task-specific training. Here, we introduce hyperdimensional fingerprints (HDF), which compose the chemical attributes of atoms and their multi-hop environments through algebraic operations on high-dimensional vectors, producing deterministic representations without any training. Across diverse property-prediction benchmarks, HDF ranks consistently among the best representations across datasets and models. Crucially, HDF encodes structure more efficiently: at low dimensionalities, its distances correlate with graph edit distance more strongly than Morgan fingerprints', yielding effective similarity metric
    
[^346]: 中轴感知的符号距离函数学习

    Medial Axis Aware Learning of Signed Distance Functions

    [https://arxiv.org/abs/2604.16512](https://arxiv.org/abs/2604.16512)

    提出一种中轴感知的高阶变分方法，通过相场近似与神经网络，从无向点云中计算高精度的全局符号距离函数。

    

    我们提出了一种新颖的变分方法，用于计算给定点云的高精度全局符号距离函数（SDF）。为此，SDF梯度的跳跃集（与曲面的中轴重合）通过一个高阶变分公式被显式地纳入考虑，该公式强制SDF沿梯度方向远离该不连续集时呈线性增长。eikonal方程和SDF的零水平集作为约束条件被强制执行。为了使该变分问题在计算上易于处理，我们采用了Ambrosio-Tortorelli类型的相场近似。相关的相场函数隐式地描述了中轴。该方法针对由无向点云表示的曲面实现，使用神经网络同时近似SDF和相场。实验证明了该方法在近场和全局范围内的准确性。定量（实验结果表明...）

    arXiv:2604.16512v3 Announce Type: replace-cross  Abstract: We propose a novel variational method to compute a highly accurate global signed distance function (SDF) to a given point cloud. To this end, the jump set of the gradient of the SDF, which coincides with the medial axis of the surface, is explicitly taken into account through a higher-order variational formulation that enforces linear growth along the gradient direction away from this discontinuity set. The eikonal equation and the zero-level set of the SDF are enforced as constraints. To make this variational problem computationally tractable, a phase field approximation of Ambrosio-Tortorelli type is employed. The associated phase field function implicitly describes the medial axis. The method is implemented for surfaces represented by unoriented point clouds using neural network approximations of both the SDF and the phase field. Experiments demonstrate the method's accuracy both in the near field and globally. Quantitative 
    
[^347]: 从Attention到Mamba：一种跨架构蒸馏的方法配方

    Attention to Mamba: A Recipe for Cross-Architecture Distillation

    [https://arxiv.org/abs/2604.14191](https://arxiv.org/abs/2604.14191)

    该论文提出了一套将预训练的Attention架构Transformer模型有效蒸馏到Mamba状态空间模型架构的方法配方，使蒸馏后的Mamba模型能够保留教师模型的性能，从而加速SSM架构的落地采用。

    

    诸如Mamba等状态空间模型（SSM）已成为Transformer模型的流行替代方案，因为与基于注意力的模型相比，它们具有更低的内存消耗和更高的生成吞吐量。另一方面，社区已经积累了大量关于如何训练Transformer的知识，并且许多预训练的Transformer模型随时可用。为了在利用现有预训练Transformer的同时促进SSM的采用，我们旨在确定一种将基于注意力的模型有效蒸馏到类Mamba架构的方法配方。然而，在先前关于跨架构蒸馏的工作中已经表明，从Transformer到Mamba的朴素蒸馏方法无法保留原始教师模型的性能，这一限制通常通过结合Attention和SSM块的混合架构方案来克服。我们工作的关键论点是，通过为Mamba配备一种有原则的初始化（摘要在此处被截断）

    arXiv:2604.14191v2 Announce Type: replace  Abstract: State Space Models (SSMs) such as Mamba have become a popular alternative to Transformer models, due to their reduced memory consumption and higher throughput at generation compared to their Attention-based counterparts. On the other hand, the community has built up a considerable body of knowledge on how to train Transformers, and many pretrained Transformer models are readily available. To facilitate the adoption of SSMs while leveraging existing pretrained Transformers, we aim to identify an effective recipe to distill an Attention-based model into a Mamba-like architecture. In prior work on cross-architecture distillation, however, it has been shown that a na\"ive distillation procedure from Transformers to Mamba fails to preserve the original teacher performance, a limitation often overcome with hybrid solutions combining Attention and SSM blocks. The key argument from our work is that, by equipping Mamba with a principled initi
    
[^348]: 量子启发的张量网络自编码器用于异常检测：一种基于MERA的方法

    Quantum-Inspired Tensor Network Autoencoders for Anomaly Detection: A MERA-Based Approach

    [https://arxiv.org/abs/2604.06541](https://arxiv.org/abs/2604.06541)

    该论文首次提出了受量子多尺度纠缠重整化拟设（MERA）启发的张量网络自编码器，并将其应用于对撞机喷注的无监督异常检测任务。

    

    我们研究多尺度张量网络架构是否能为对撞机喷注的基于重构的异常检测提供一种有用的归纳偏置。喷注由级联分支过程产生，因此其内部结构天然地在角尺度与动量尺度上组织。这启发我们构建一种能够分层压缩信息、并可在粗粒化之前重组短程关联的自编码器。受此图像引导，我们提出了一种受多尺度纠缠重整化拟设（MERA）启发的自编码器，作用于有序排列的喷注组分。据我们所知，此前尚未有人提出过受MERA启发的自编码器，且该架构尚未在对撞机异常检测领域被探索过。我们在仅背景重构的框架内，将该架构与稠密自编码器、相应的树张量网络极限以及经典基线方法进行了比较。本文围绕两个主要……

    arXiv:2604.06541v2 Announce Type: replace-cross  Abstract: We investigate whether a multiscale tensor-network architecture provides a useful inductive bias for reconstruction-based anomaly detection in collider jets. Jets are produced by a branching cascade, so their internal structure is naturally organised across angular and momentum scales. This motivates an autoencoder that compresses information hierarchically and can reorganise short-range correlations before coarse-graining. Guided by this picture, we formulate a Multiscale Entanglement Renormalization Ansatz (MERA)-inspired autoencoder acting on ordered jet constituents. To our knowledge, a MERA-inspired autoencoder has not previously been proposed, and this architecture has not been explored in collider anomaly detection. We compare this architecture to a dense autoencoder, the corresponding tree-tensor-network limit, and classical baselines within a background-only reconstruction framework. The paper is organised around two m
    
[^349]: 解释的正确性重要吗？将计算化XAI评估与人类理解联系起来

    Does Explanation Correctness Matter? Linking Computational XAI Evaluation to Human Understanding

    [https://arxiv.org/abs/2603.25251](https://arxiv.org/abs/2603.25251)

    该研究通过200人用户实验，在四个受控正确性水平（100%、85%、70%、55%）下首次实证检验了解释正确性与人类理解之间的联系，发现解释的功能正确性确实会影响人类对AI决策的前向模拟准确率，从而为计算化XAI评估指标与人类理解之间建立了关联。

    

    解释性人工智能（XAI）方法通常使用功能正确性指标（也称为忠实度或保真度）进行评估，这些指标衡量解释在多大程度上反映模型的真实推理过程。人们通常假设正确性越高，人类理解效果就越好，但这一联系尚未在受控的正确性水平下得到检验。我们开展了一项用户研究（N=200），在一个合成时间序列分类任务中，将解释正确性操纵在四个水平（100%、85%、70%、55%），参与者无法依赖领域知识或视觉直觉进行判断。解释被构建为类似特征归因的形式，其正确性基于已知的真实标准定义，而非从训练好的模型中估计得出。参与者需要根据这些解释预测模拟AI的决策（前向模拟），我们以其前向模拟准确率作为理解程度的代理指标。正确性影响了前向模拟准确率

    arXiv:2603.25251v3 Announce Type: replace-cross  Abstract: Explainable AI (XAI) methods are commonly evaluated using functional correctness metrics, also called faithfulness or fidelity, which estimate how closely an explanation reflects the model's reasoning. Higher correctness is assumed to produce better human understanding, but this link has not been tested with controlled levels. We conducted a user study (N=200) that manipulated explanation correctness at four levels (100%, 85%, 70%, 55%) in a synthetic time series classification task where participants could not rely on domain knowledge or visual intuition. Explanations were constructed to resemble feature attributions, with correctness defined against a known ground truth rather than estimated from a trained model. Participants predicted a simulated AI's decisions from these explanations (forward simulation), and we used their forward simulation accuracy as a proxy for understanding. Correctness affected forward simulation accu
    
[^350]: MemReward：基于图的经验记忆用于有限标签下的LLM奖励预测

    MemReward: Graph-Based Experience Memory for LLM Reward Prediction with Limited Labels

    [https://arxiv.org/abs/2603.19310](https://arxiv.org/abs/2603.19310)

    提出MemReward，将查询、思考过程和答案作为异构图节点构建经验记忆，利用GNN将有限的奖励标签传播到无标签轨迹，从而在RL微调中充分利用全部采样轨迹优化LLM策略。

    

    强化学习（RL）微调通过在采样轨迹上利用奖励信号来提升大型语言模型（LLM）的推理能力。当只有部分轨迹拥有奖励标签时，将策略更新限制在该子集上会导致其余轨迹无法用于策略优化。半监督学习可以将标签从有标签样本传播到无标签样本，但现有方法要么使用固定的相似度权重来扩散标签，要么将每个样本独立处理，忽略了查询、思考过程与答案之间的关联。我们提出MemReward，一种基于图的经验记忆方法，可在在线策略优化过程中将奖励传播到无标签轨迹。MemReward将初始LLM策略的查询及其轨迹（思考过程与最终答案）存储为异构图中的节点，并通过相似性边和结构边进行连接，同时在有标签轨迹上预热训练一个GNN，以在图上传播奖励……（摘要在此处截断）

    arXiv:2603.19310v5 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) fine-tuning improves large language model (LLM) reasoning using reward signals on sampled rollouts. When reward labels are available for only a subset of rollouts, restricting policy updates to that subset leaves the remaining rollouts unused for policy optimization. Semi-supervised learning propagates labels from labeled to unlabeled samples, but existing methods either diffuse labels with fixed similarity weights or treat each sample independently, ignoring the relations among queries, thinking processes, and answers. We propose MemReward, a graph-based experience memory that propagates rewards to unlabeled rollouts during online policy optimization. MemReward stores queries and the rollouts (thinking processes and final answers) of an initial LLM policy as nodes of a heterogeneous graph connected by similarity and structural edges, and warms up a GNN on labeled rollouts to propagate rewards over t
    
[^351]: 流形匹配自编码器：将数据嵌入到选定的几何结构中

    Manifold-Matching Autoencoders: Embedding Data into Chosen Geometric Structures

    [https://arxiv.org/abs/2603.16568](https://arxiv.org/abs/2603.16568)

    提出一种名为“流形匹配”的简单正则化方法，通过最小化潜空间与任意参考表示之间成对距离的均方误差，使自编码器的潜空间能够被塑造成任意选定的几何结构，且无需标签、参考维数与潜空间维数自由解耦。

    

    操纵自编码器潜空间中数据的结构是表示学习领域的一个长期目标。我们提出了一种简单的正则化方法——流形匹配，它能够在潜空间上施加显式的几何结构。其核心思想是选取数据的一个参考表示R（它不必是输入网络的原始输入X），然后按批次最小化R中的成对距离与潜空间Z中的成对距离之间的均方误差。尽管自编码器仍然学习重建X，但其潜空间的形状会逐渐模仿R的形状。该目标不需要标签，只需要参考距离；而且由于这些距离是标量，R的维数与Z的维数相互解耦：任何参考都可以引导任何瓶颈层。利用这一单一工具，我们研究了如何在潜空间上施加不同的几何结构，并与基于几何或t…（方法）构建的自编码器进行了比较。

    arXiv:2603.16568v2 Announce Type: replace-cross  Abstract: Manipulating the structure of data in the latent space of an autoencoder is a long-standing goal of representation learning. We propose a simple regularization, Manifold Matching, that imposes an explicit geometric structure on that latent space. The idea is to take a reference representation R of the data, which need not be the raw input X fed to the network, and to minimize, batch-wise, the mean squared error between the pairwise distances in R and those in the latent space Z. Although the autoencoder still learns to reconstruct X, the shape of its latent space comes to mimic that of R. The objective requires no labels, only reference distances, and because those are scalars, the dimensionality of R is decoupled from that of Z: any reference can guide any bottleneck. Using this single instrument we study how different geometric structures can be imposed on a latent space, comparing against autoencoders built on geometric or t
    
[^352]: PREBA：基于PCA加权检索增强大语言模型与贝叶斯平均聚合的手术时长预测

    PREBA: Surgical Duration Prediction via PCA-Weighted Retrieval-Augmented LLMs and Bayesian Averaging Aggregation

    [https://arxiv.org/abs/2603.13275](https://arxiv.org/abs/2603.13275)

    PREBA提出了一种融合PCA加权检索与贝叶斯平均聚合的检索增强框架，无需任务特定微调即可将大语言模型的推理锚定于机构特定的临床证据上，从而实现准确且稳定的手术时长预测。

    

    手术时长的准确预测对医院资源管理至关重要。尽管近年来从机器学习（ML）到微调大语言模型（LLM）等监督学习方法已展现出强大的性能，但它们仍受限于对高质量标注数据和计算密集型训练的依赖。相比之下，零样本LLM推理提供了一种无需训练的有前景的替代方案，但其缺乏对特定机构临床证据的支撑，导致预测结果与临床实际脱节且容易不稳定。为解决这些局限性，我们提出了PREBA，这是一个检索增强框架，它集成了PCA加权检索和贝叶斯平均聚合，将LLM的推理建立在特定机构的临床证据之上，而无需任务特定的微调。具体而言，PREBA首先将异构的临床特征编码为统一的表示形式，然后执行PCA加权检索……（原文摘要在此处截断）

    arXiv:2603.13275v4 Announce Type: replace-cross  Abstract: Accurate prediction of surgical duration is pivotal for hospital resource management. Although recent supervised learning approaches, from machine learning (ML) to fine-tuned large language models (LLMs), have shown strong performance, they remain constrained by the need for high-quality labeled data and computationally intensive training. In contrast, zero-shot LLM inference offers a promising training-free alternative, but it lacks grounding in institution-specific clinical evidence, making its predictions clinically misaligned and prone to instability. To address these limitations, we present PREBA, a retrieval-augmented framework that integrates PCA-weighted retrieval and Bayesian averaging to ground LLM reasoning in institution-specific evidence without requiring task-specific fine-tuning. Specifically, PREBA first encodes heterogeneous clinical features into a unified representation. It then performs PCA-weighted retrieva
    
[^353]: 推理即压缩：通过条件信息瓶颈统一预算强制方法

    Reasoning as Compression: Unifying Budget Forcing via the Conditional Information Bottleneck

    [https://arxiv.org/abs/2603.08462](https://arxiv.org/abs/2603.08462)

    该论文将高效推理重新建模为条件信息瓶颈下的有损压缩问题，解决了注意力机制违反马尔可夫性质的理论缺口，并由此导出一个通用的强化学习目标，在压缩推理链的同时最大化任务奖励，从而统一了预算强制方法。

    

    链式思维提示提升了大语言模型在复杂任务上的准确率，但往往会增加token使用量和推理成本。现有的“预算强制”方法通过带有启发式长度惩罚的微调来降低成本，但同时抑制了必要的推理过程和冗余的填充内容。我们将高效推理重新表述为信息瓶颈原理下的有损压缩问题，并指出了将朴素的信息瓶颈应用于Transformer时的一个关键理论缺口：注意力机制违反了提示、推理轨迹与响应之间的马尔可夫性质。为解决这一问题，我们在条件信息瓶颈原理下对链式思维生成进行建模，其中推理轨迹 $Z$ 充当计算桥梁，仅包含从提示 $X$ 无法直接获取的关于响应 $Y$ 的信息。由此得到一个通用的强化学习目标：在压缩补全内容的同时最大化任务奖励。

    arXiv:2603.08462v3 Announce Type: replace  Abstract: Chain-of-Thought (CoT) prompting improves LLM accuracy on complex tasks but often increases token usage and inference cost. Existing ``Budget Forcing'' methods reduce cost via fine-tuning with heuristic length penalties, suppressing both essential reasoning and redundant filler. We recast efficient reasoning as a lossy compression problem under the Information Bottleneck (IB) principle, and identify a key theoretical gap when applying naive IB to transformers: attention violates the Markov property between prompt, reasoning trace, and response. To resolve this issue, we model CoT generation under the Conditional Information Bottleneck (CIB) principle, where the reasoning trace $Z$ acts as a computational bridge that contains only the information about the response $Y$ that is not directly accessible from the prompt $X$. This yields a general Reinforcement Learning objective: maximize task reward while compressing completions under a 
    
[^354]: 从启发式选择到自动化算法设计：大语言模型受益于强先验

    From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors

    [https://arxiv.org/abs/2603.02792](https://arxiv.org/abs/2603.02792)

    本文通过 token 级归因分析发现提示词中嵌入的代码示例和策略指令对 LLM 生成的算法影响最大，并据此提出将现有基准算法作为强先验来引导 LLM 进行自动化算法设计，在伪布尔和黑盒优化基准上验证了其有效性。

    

    大语言模型已被广泛应用于自动化算法设计，在各个领域生成和演化算法方面展现出强大的能力。现有工作主要集中于考察其在解决特定问题上的有效性，而搜索策略主要由自适应的提示词设计来引导。本文通过研究提示词中各 token 对 LLM 生成算法代码的归因影响，发现提示词中嵌入的代码示例及其相关的策略指令对生成的算法具有最强的影响力。基于这一洞察，我们提出利用现有的基准算法作为强先验来引导 LLM 驱动的优化。我们在两个黑盒优化基准——伪布尔优化套件和黑盒优化套件上，使用三个大语言模型对这一方法进行了评估。尽管仅依赖于简单的 (1+1) 精英……

    arXiv:2603.02792v3 Announce Type: replace  Abstract: Large Language Models (LLMs) have already been widely adopted for automated algorithm design, demonstrating strong abilities in generating and evolving algorithms across various fields. Existing work has largely focused on examining their effectiveness in solving specific problems, with search strategies primarily guided by adaptive prompt designs. In this paper, through investigating the token-wise attribution of the prompts to LLM-generated algorithmic code, we show that the embedded code examples and their associated strategy instructions have the strongest influence on the generated algorithms. Building upon this insight, we propose leveraging existing benchmark algorithms as strong priors to guide LLM-driven optimization. We evaluate this approach with three LLMs on two black-box optimization benchmarks: the pseudo-Boolean optimization suite (pbo) and the black-box optimization suite (bbob). Despite relying on a simple (1+1) eli
    
[^355]: SEAR：样本高效的动作分块强化学习

    SEAR: Sample Efficient Action Chunking Reinforcement Learning

    [https://arxiv.org/abs/2603.01891](https://arxiv.org/abs/2603.01891)

    SEAR通过多时域目标训练的因果变换器批评者和后退时域随机重规划，实现了样本高效的在线动作分块强化学习，同时解决了高维动作序列和决策频率恢复的挑战。

    

    arXiv:2603.01891v2 公告类型：替换 摘要：动作分块在长时程强化学习中改善了探索并加速了价值传播，但天真地将离策略方法应用于降低决策频率的时间扩展动作空间会抵消这些收益，导致样本效率低下。现有的在线动作分块方法通过计算成本高昂的仅批评者方法或依赖离线数据来解决这些问题。我们引入了SEAR，一种样本高效的离策略算法，通过解决这两个挑战，实现了带有动作分块的在线强化学习。为了处理高维动作序列，SEAR采用了一个因果变换器批评者，该批评者使用多时域目标进行训练，为动作分块的每个前缀提供训练信号，有效增加了每个样本的有用梯度。为了恢复决策频率，SEAR采用后退时域和随机重规划操作，确保状态覆盖均匀。

    arXiv:2603.01891v2 Announce Type: replace  Abstract: Action chunking improves exploration and accelerates value propagation in long-horizon reinforcement learning, but naively applying off-policy methods to the temporally extended action space at reduced decision frequency offsets these gains, leading to poor sample efficiency. Existing online action chunking methods address these issues through computationally expensive critic-only approaches or by relying on offline data. We introduce SEAR, a sample-efficient off-policy algorithm that enables online reinforcement learning with action chunks by addressing both challenges. To handle high-dimensional action sequences, SEAR employs a causal transformer critic trained with multi-horizon targets that provide a training signal for every prefix of an action chunk, effectively increasing the useful gradients per sample. To restore decision frequency, SEAR operates with a receding horizon and random replanning, ensuring uniform state coverage 
    
[^356]: 万能 $p$-adic ReLU 神经网络的最小宽度

    The minimal width of universal $p$-adic ReLU neural networks

    [https://arxiv.org/abs/2603.00064](https://arxiv.org/abs/2603.00064)

    本文确定了以自然 $p$-adic ReLU 类似函数为激活函数的 $p$-adic 神经网络，在 $L_q$ 范数和 $C_1$ 范数意义下对紧开子集上连续 $\mathbb Q_p$-值函数具有万能逼近性质所需的最小宽度。

    

    我们确定了具有万能逼近性质的 $p$-adic 神经网络的最小宽度，该性质是针对紧开子集上连续的 $\mathbb Q_p$-值函数在 $L_q$ 范数和 $C_1$ 范数意义下的逼近，其中激活函数是 ReLU 函数的一种自然的 $p$-adic 类比。

    arXiv:2603.00064v2 Announce Type: replace-cross  Abstract: We determine the minimal width of $p$-adic neural networks with the universal approximation property for continuous $\mathbb Q_p$-valued functions on compact open subsets with respect to the $L_q$ norms and the $C_1$ norm, where the activation function is a natural $p$-adic analogue of the ReLU function.
    
[^357]: 来自双重污染与异构批次的高维鲁棒均值估计

    High-Dimensional Robust Mean Estimation from Doubly Corrupted and Heterogeneous Batches

    [https://arxiv.org/abs/2602.20698](https://arxiv.org/abs/2602.20698)

    本文研究用户级与样本级双重对抗污染下的高维鲁棒均值估计问题，提出了一种平方和（SoS）算法，在协方差矩阵有界条件下实现了较小的 $\ell_2$ 估计误差。

    

    现代数据系统通常需要聚合来自多个来源的信息：其中一些来源可能完全不可信，而其余据信可靠的来源仍可能存在少量污染。我们研究这一实际场景下的鲁棒均值估计问题，并通过双重污染模型对其进行形式化：样本分布在 $N$ 个用户之间，每个用户拥有 $n$ 个样本。其中 $\varepsilon$ 比例的用户可能被任意污染，而其余的“好”用户中每个用户仍有 $\alpha$ 比例（$\alpha>0$）的被污染样本。目标是以较小的 $\ell_2$ 误差估计未知均值 $\mu$。这是一个具有挑战性的场景，因为我们必须同时处理两种类型的对抗性污染：用户级污染和样本级污染。当未知分布的协方差矩阵以单位矩阵为界时，我们设计了一种平方和（SoS）算法，达到了 $\ell_2$ 误差……（摘要在此处截断）

    arXiv:2602.20698v2 Announce Type: replace  Abstract: Modern data systems often require aggregating information from multiple sources: some might be entirely untrustworthy, while the remaining presumably reliable sources may still suffer from a small amount of corruption. We study robust mean estimation under this practical scenario, which is formulated through the double corruption model: samples are divided among $N$ users, each with $n$ samples. $\varepsilon$ fraction of the users could be arbitrarily corrupted, while each of the remaining ``good'' users still have $\alpha$-fraction ($\alpha>0$) of corrupted samples. The goal is to estimate the unknown mean $\mu$ with small $\ell_2$ error. This is a challenging scenario because we must handle two types of adversarial corruption: user-level corruption and sample-level corruption. When the unknown distribution has covariance matrix bounded by the identity matrix, we design a sum-of-squares (SoS) algorithm that achieves an $\ell_2$ erro
    
[^358]: 黄金层及其寻找之道：通过层梯度分析改进大语言模型的知识编辑

    Golden Layers and Where to Find Them: Improved Knowledge Editing for Large Language Models Via Layer Gradient Analysis

    [https://arxiv.org/abs/2602.20207](https://arxiv.org/abs/2602.20207)

    本研究通过层梯度分析，证明了大语言模型中存在固定的“黄金层”，其知识编辑性能接近逐样本最优层，且这些黄金层可通过代理数据集可靠识别并能跨数据集泛化到未见查询，从而改进了知识编辑方法。

    

    大语言模型（LLMs）中的知识编辑旨在将模型对特定查询的预测更新为期望的目标，同时保持模型对所有其他输入的原有行为。这一过程通常包含两个阶段：确定需要编辑的层，以及执行参数更新。直觉上，不同的查询可能会将知识定位于模型的不同深度，因此固定的编辑层对不同样本的编辑性能会有所差异。在本工作中，我们假设存在固定的“黄金层”，能够达到与逐样本最优层相近的近最优编辑性能。为验证这一假设，我们通过将黄金层与真实的逐样本最优层进行对比，提供了实证证据。此外，我们证明黄金层可以通过代理数据集被可靠地识别出来，并且能够有效地泛化到不同数据集中未见过的测试集查询上。最后，我们提出了一种新颖的方法（摘要在此处截断）。

    arXiv:2602.20207v4 Announce Type: replace-cross  Abstract: Knowledge editing in Large Language Models (LLMs) aims to update the model's prediction for a specific query to a desired target while preserving its behavior on all other inputs. This process typically involves two stages: identifying the layer to edit and performing the parameter update. Intuitively, different queries may localize knowledge at different depths of the model, resulting in different sample-wise editing performance for a fixed editing layer. In this work, we hypothesize the existence of fixed golden layers that can achieve near-optimal editing performance similar to sample-wise optimal layers. To validate this hypothesis, we provide empirical evidence by comparing golden layers against ground-truth sample-wise optimal layers. Furthermore, we show that golden layers can be reliably identified using a proxy dataset and generalize effectively to unseen test set queries across datasets. Finally, we propose a novel me
    
[^359]: 量化扩散模型中的误差传播与模型坍塌

    Quantifying Error Propagation and Model Collapse in Diffusion Models

    [https://arxiv.org/abs/2602.16601](https://arxiv.org/abs/2602.16601)

    该论文首次为扩散模型在合成数据上递归训练导致的模型坍塌建立了散度的理论上界与下界，并刻画了分数估计误差与新鲜数据比例如何决定生成分布偏离目标分布的不同漂移区间。

    

    机器学习模型越来越多地在合成数据上进行训练或微调。已有观察表明，在这类数据上进行递归训练会在广泛的任务中显著降低性能，通常表现为生成分布逐渐偏离目标分布。在本工作中，我们在基于分数的扩散模型设置下对这一现象进行了理论分析。针对每一轮训练混合使用合成数据与来自目标分布的新鲜样本这一现实训练流程，我们得到了生成分布与目标分布之间累积散度的上界与下界。值得注意的是，据我们所知，这是首个关于学习分布与目标分布之间散度的下界，即使对于标准扩散模型亦是如此。我们的结果使我们能够根据分数估计误差和新鲜数据的使用比例，刻画不同的漂移区间。

    arXiv:2602.16601v3 Announce Type: replace-cross  Abstract: Machine learning models are increasingly trained or fine-tuned on synthetic data. Recursively training on such data has been observed to significantly degrade performance in a wide range of tasks, often characterized by a progressive drift away from the target distribution. In this work, we theoretically analyze this phenomenon in the setting of score-based diffusion models. For a realistic pipeline where each training round uses a combination of synthetic data and fresh samples from the target distribution, we obtain upper and lower bounds on the accumulated divergence between the generated and target distributions. Notably, to the best of our knowledge, this is the first lower bound on the divergence between the learned and target distributions, even for standard diffusion models. Our results allow us to characterize different regimes of drift, depending on the score estimation error and the proportion of fresh data used in e
    
[^360]: 面向大语言模型强化学习的进化式系统提示学习

    Evolutionary System Prompt Learning for Reinforcement Learning in LLMs

    [https://arxiv.org/abs/2602.14697](https://arxiv.org/abs/2602.14697)

    本文提出进化式系统提示学习（E-SPL），在强化学习迭代中联合优化模型权重与系统提示，实现权重中程序性知识与提示中可解释陈述性知识的自然分工，显著提升大语言模型在数学推理任务上的成功率与由易到难的泛化能力。

    

    构建能够从经验中自主自我改进的智能体系统是人工智能长期以来的目标。如今，大语言模型（LLM）主要通过两种机制进行自我改进：通过自我反思进行上下文更新，以及通过强化学习（RL）进行权重更新。在本工作中，我们提出了进化式系统提示学习（E-SPL），这是一种联合改进模型经验上下文及其权重的方法。在每次强化学习迭代中，E-SPL 在多个系统提示下并行采样轨迹，然后利用这些轨迹通过强化学习联合更新权重，并通过自我反思进化系统提示。E-SPL 促进了权重中的程序性知识与系统提示中可解释的陈述性知识之间的自然分工。最值得注意的是，在由易到难的泛化设置中（AIME → BeyondAIME），E-SPL 将强化学习的成功率从 38.8% 提升至 45.1%。E-SPL 还提升了强化学习在 AIME 2025 上的表现（56.3……

    arXiv:2602.14697v4 Announce Type: replace  Abstract: Building agentic systems that can autonomously self-improve from experience is a longstanding goal of AI. Large language models (LLMs) today primarily self-improve via two mechanisms: self-reflection for context updates, and reinforcement learning (RL) for weight updates. In this work, we propose Evolutionary System Prompt Learning (E-SPL), a method for jointly improving a model's experiential context and its weights. In each RL iteration, E-SPL samples trajectories under multiple system prompts in parallel, then uses them to jointly update the weights with RL and evolve the system prompts with self-reflection. E-SPL encourages a natural division between procedural knowledge in weights and interpretable declarative knowledge in system prompts. Most notably, in an easy-to-hard generalization setting (AIME $\rightarrow$ BeyondAIME), E-SPL improves RL success rate from 38.8% $\rightarrow$ 45.1%. E-SPL also improves RL on AIME 2025 (56.3
    
[^361]: 发现心理量表中的语义结构：一种无需作答数据的高效简化前端方法

    Discovering Semantic Structures in Psychological Scales: A Response-Free Front-End to Efficient Simplification

    [https://arxiv.org/abs/2602.12575](https://arxiv.org/abs/2602.12575)

    该论文提出了一种无需作答数据的主题建模框架，利用上下文句子嵌入与密度聚类从量表题目措辞中自动发现语义结构并提取代表性关键词，从而实现心理量表的高效系统化简化。

    

    心理量表的精简传统上依赖于基于作答数据的方法，如因子分析、项目反应理论和网络心理测量学，以揭示潜在构念并优化题目构成。尽管这些方法严谨，但它们需要大样本，且可能受制于数据可得性和跨文化可比性。自然语言处理的最新进展表明，量表题目的措辞本身包含与构念相关的语义组织，从而提供了一种无需作答数据的互补视角。我们引入了一个主题建模框架，将题目层面的语义结构操作化，用于系统性的量表简化。题目通过上下文句子嵌入进行编码，并通过基于密度的聚类进行分组，从而在无需预先设定主题数量的情况下识别语义主题。在每个聚类内部，基于类别的词项加权方法识别出能够概括每个主题的代表性关键词，为后续……

    arXiv:2602.12575v3 Announce Type: replace  Abstract: Psychological scale refinement traditionally relies on response-based methods such as factor analysis, item response theory, and network psychometrics to uncover latent constructs and optimize item composition. Although rigorous, these approaches require large samples and may be constrained by data availability and cross-cultural comparability. Recent advances in natural language processing suggest that the wording of scale items contains construct-relevant semantic organization, offering a complementary response-free perspective. We introduce a topic-modeling framework that operationalizes item-level semantic structure for systematic scale simplification. Items are encoded using contextual sentence embeddings and grouped through density-based clustering to identify semantic topics without predefining their number. Within each cluster, class-based term weighting identifies representative keywords that summarize each topic and support
    
[^362]: 一种用于短期负荷预测的轻量级多视角方法

    A Lightweight Multi-View Approach to Short-Term Load Forecasting

    [https://arxiv.org/abs/2602.09220](https://arxiv.org/abs/2602.09220)

    本文提出一种利用单值嵌入、缩放时间范围输入和嵌入丢弃机制的轻量级多视角方法，以显著更少的参数实现了具有竞争力的短期负荷预测性能。

    

    时间序列预测是能源、金融和气象等领域的一项关键任务，准确的预测能够支持明智的决策。尽管基于Transformer和大参数量的模型最近取得了最先进的结果，但其复杂性可能导致过拟合和不稳定的预测，尤其是当较早的数据点变得不那么相关时。在本文中，我们提出了一种用于短期负荷预测的轻量级多视角方法，该方法利用单值嵌入和缩放的时间范围输入来高效地捕获时间相关特征。我们引入了一种嵌入丢弃机制，以防止对特定特征的过度依赖并增强可解释性。我们的方法以显著更少的参数实现了具有竞争力的性能，在包括噪声或稀疏数据场景在内的多个数据集上展示了鲁棒性，并对各组成部分的贡献提供了深入见解。

    arXiv:2602.09220v2 Announce Type: replace-cross  Abstract: Time series forecasting is a critical task across domains such as energy, finance, and meteorology, where accurate predictions enable informed decision-making. While transformer-based and large-parameter models have recently achieved state-of-the-art results, their complexity can lead to overfitting and unstable forecasts, especially when older data points become less relevant. In this paper, we propose a lightweight multi-view approach to short-term load forecasting that leverages single-value embeddings and a scaled time-range input to capture temporally relevant features efficiently. We introduce an embedding dropout mechanism to prevent over-reliance on specific features and enhance interpretability. Our method achieves competitive performance with significantly fewer parameters, demonstrating robustness across multiple datasets, including scenarios with noisy or sparse data, and provides insights into the contributions of 
    
[^363]: 在记忆发生之处缓解记忆化

    Mitigating Memorization Where It Happens

    [https://arxiv.org/abs/2602.08901](https://arxiv.org/abs/2602.08901)

    该论文提出了一个探测—引导框架，利用目标模型与低暴露参考模型之间的 token 级对数似然比精准定位记忆化发生的位置，并仅在这些位置施加有针对性的激活校正，从而在缓解逐字记忆问题的同时避免损害模型的正常泛化性能。

    

    大型语言模型（LLM）可能会逐字记忆并复现训练序列，这引发了隐私和版权方面的担忧。现有的缓解方法会对所有内容统一施加干预，往往会损害大多数正常泛化的 token 的性能。我们研究了目标模型与一个对被评估序列接触较少、但具备领域能力的参考模型之间的 token 级对数似然比。在控制暴露程度的实验中，该指标被证明可用于定位与序列级逐字复现相关的位置。在所评估的各种设置中，少数位置承载了不成比例的正向信号，这促使我们采用选择性的激活校正方法。基于理想的记忆化削减目标，我们推导出一个“探测—引导”框架，该框架将干预分解为两个部分：一个探测器，用于检测与记忆化相关的激活；一个引导器，仅当探测器的输出超过阈值时才施加有针对性的校正。

    arXiv:2602.08901v2 Announce Type: replace  Abstract: Large language models (LLMs) can memorize and reproduce training sequences verbatim, raising privacy and copyright concerns. Existing mitigation methods apply interventions uniformly, often degrading performance on the majority of tokens that generalize normally. We study a token-level log-likelihood ratio between a target model and a domain-competent reference with lower exposure to the evaluated sequences. Exposure-controlled experiments support its use for localizing positions associated with sequence-level verbatim recall. Across evaluated settings, a minority of positions carries a disproportionate share of the positive signal, motivating selective activation correction. From the ideal memorization reduction objective, we derive a probe--steer framework which decomposes intervention into a probe that detects memorization-relevant activations and a steer that applies targeted correction only when the probe exceeds a threshold. We
    
[^364]: CompilerKV：通过离线经验编译实现风险自适应的KV压缩

    CompilerKV: Risk-Adaptive KV Compression via Offline Experience Compilation

    [https://arxiv.org/abs/2602.08686](https://arxiv.org/abs/2602.08686)

    论文提出CompilerKV，通过从校准语料库离线编译KV保留校正表，将在线校正开销降至O(1)查找，且编译出的保留表可作为可移植的架构先验，在跨语料库和跨模型迁移中表现出高度一致性。

    

    仅预填充的KV压缩在预填充结束时冻结一个token子集，并从中进行解码，不再进行后续淘汰。因此保留决策是不可逆的，然而现有方法是在线地从单个带噪声的提示中估计其所依赖的校正信号，即每个注意力头的可靠性和提示级别的压缩敏感度。我们认为这是错误的统计单位：这些信号在跨提示之间表现出的规律性远高于提示内的信噪比。我们提出了CompilerKV，这是一种KV保留策略，其校正表是从校准语料库中离线编译得到的，将标准观察窗口扫描之后的在线校正简化为O(1)次查找加上一个预算钳制。我们发现编译得到的保留表表现得如同可移植的架构先验：在四个骨干网络上，排序可以在互不相交的语料库之间迁移（平均Spearman相关系数ρ̄=0.90），且直接的模型到模型表迁移仅需0.4

    arXiv:2602.08686v4 Announce Type: replace-cross  Abstract: Prefill-only KV compression freezes a token subset at the end of prefill and decodes from it without further eviction. The retention decision is therefore irreversible, yet existing methods estimate the corrective signals it relies on, per-head reliability and prompt-level compression sensitivity, online from a single noisy prompt. We argue this is the wrong statistical unit: these signals exhibit far higher cross-prompt regularity than within-prompt signal-to-noise. We introduce \textsc{CompilerKV}, a KV-retention policy whose corrective tables are compiled offline from a calibration corpus, reducing online correction after the standard observation-window scan to $O(1)$ lookups plus a budget clamp. We find that compiled retention tables behave as portable architectural priors: rankings transfer across disjoint corpora on four backbones (mean Spearman $\bar\rho{=}0.90$), and direct model-to-model table transfer costs only $0.4$
    
[^365]: 正交模型合并

    Orthogonal Model Merging

    [https://arxiv.org/abs/2602.05943](https://arxiv.org/abs/2602.05943)

    提出OrthoMerge方法，通过在正交群形成的黎曼流形上进行合并以保留预训练权重的几何结构，从而更有原则且高效地整合多个微调大语言模型的多样化能力。

    

    合并微调后的大语言模型（LLMs）对于将多种能力整合到单一统一模型中正变得越来越重要。然而，主流的模型合并方法依赖于欧几里得空间中的线性算术运算，这往往会破坏预训练权重的内在几何属性，例如超球面能量。为了解决这个问题，我们提出了正交模型合并（OrthoMerge），一种在由正交群形成的黎曼流形上执行合并操作的方法，以保持模型权重的几何结构。通过将正交微调（OFT）学习到的任务特定正交矩阵映射到李代数，OrthoMerge 实现了一种有原则且高效的整合方式，能够同时考虑自适应的方向和强度。除了直接利用 OFT 得到的正交矩阵外，我们进一步将该方法扩展到使用非正交方式微调的通用模型。

    arXiv:2602.05943v2 Announce Type: replace  Abstract: Merging finetuned Large Language Models (LLMs) has become increasingly important for integrating diverse capabilities into a single unified model. However, prevailing model merging methods rely on linear arithmetic in Euclidean space, which often destroys the intrinsic geometric properties of pretrained weights, such as hyperspherical energy. To address this, we propose Orthogonal Model Merging (OrthoMerge), a method that performs merging operations on the Riemannian manifold formed by the orthogonal group to preserve the geometric structure of the model's weights. By mapping task-specific orthogonal matrices learned by Orthogonal Finetuning (OFT) to the Lie algebra, OrthoMerge enables a principled yet efficient integration that takes into account both the direction and intensity of adaptations. In addition to directly leveraging orthogonal matrices obtained by OFT, we further extend this approach to general models finetuned with non
    
[^366]: 联邦概念模型：基于分布式监督的可解释模型

    Federated Concept-Based Models: Interpretable models with distributed supervision

    [https://arxiv.org/abs/2602.04093](https://arxiv.org/abs/2602.04093)

    本文提出联邦概念模型，通过跨机构聚合概念级信息并随概念监督集合的演变高效调整模型架构，首次将可解释的概念模型与联邦学习相结合，实现了分布式监督下的可解释深度学习。

    

    概念模型通过将预测建立在人类可理解的概念之上，提升了深度学习的可解释性。然而，概念标注成本高昂，且很少能在单一数据源中大规模获取。联邦学习可以通过实现跨机构的训练来缓解这一限制，从而利用分布在不同数据所有者之间的概念标注。然而，联邦学习缺乏可解释的建模范式。将概念模型与联邦学习集成并非易事：尽管联邦学习支持异构且非平稳的客户端参与，但它通常假设共享架构固定不变，而概念模型可能随着可用概念集的演变而需要架构上的调整。我们提出了联邦概念模型，这是一种在不断演变的联邦学习环境中部署概念模型的新方法。F-CMs跨机构聚合概念级信息，并高效地调整模型架构以适应概念监督的变化……

    arXiv:2602.04093v3 Announce Type: replace  Abstract: Concept-based Models (CMs) enhance interpretability in deep learning by grounding predictions in human-understandable concepts. However, concept annotations are costly and rarely available at scale within a single data source. Federated Learning (FL) could alleviate this limitation by enabling cross-institutional training over concept annotations distributed across multiple data owners. Yet, FL lacks interpretable modeling paradigms. Integrating CMs with FL is non-trivial: although FL supports heterogeneous and non-stationary client participation, it typically assumes a fixed shared architecture, whereas CMs may require architectural adaptation as the available concept set evolves. We propose Federated Concept-based Models (F-CMs), a new methodology for deploying CMs in evolving FL settings. F-CMs aggregate concept-level information across institutions and efficiently adapt the model architecture to changes in concept supervision whi
    
[^367]: SLOPE：面向基于模型的强化学习的乐观势能景观塑形方法

    SLOPE: Optimistic Potential Landscape Shaping for Model-based Reinforcement Learning

    [https://arxiv.org/abs/2602.03201](https://arxiv.org/abs/2602.03201)

    SLOPE通过乐观分布回归估计高置信度上界来构建信息丰富的势能景观，放大稀疏奖励中的罕见成功信号，从而显著提升基于模型的强化学习在稀疏奖励任务上的性能。

    

    基于模型的强化学习（MBRL）具有样本高效的优点，但在稀疏奖励环境中表现不佳。其关键瓶颈在于稀疏环境下缺乏信息丰富的梯度，标准奖励模型往往产生平坦的势能景观，难以引导规划。为应对这一挑战，我们提出了基于乐观势能估计的景观塑形方法（SLOPE），这是一个新颖的框架，将奖励建模从预测稀疏标量转变为构建信息丰富的势能景观。SLOPE采用乐观分布回归来估计高置信度上界，从而放大罕见的成功信号并确保充分的探索梯度。在5个基准测试的30多个任务以及真实世界机器人部署上的评估表明，SLOPE在全稀疏、半稀疏和稠密奖励设置中始终优于领先的基线方法。项目代码可在 https://github.com/Arya87/SLOPE 获取。

    arXiv:2602.03201v4 Announce Type: replace  Abstract: Model-based reinforcement learning (MBRL) is sample-efficient but struggles in sparse reward settings. A critical bottleneck arises from the lack of informative gradients in sparse settings, where standard reward models often yield flat landscapes that struggle to guide planning. To address this challenge, we propose Shaping Landscapes with Optimistic Potential Estimates (SLOPE), a novel framework that shifts reward modeling from predicting sparse scalars to constructing informative potential landscapes. SLOPE employs optimistic distributional regression to estimate high-confidence upper bounds, which amplifies rare success signals and ensures sufficient exploration gradients. Evaluations on 30+ tasks across 5 benchmarks and real-world robotic deployments, demonstrate that SLOPE consistently outperforms leading baselines in fully sparse, semi-sparse, and dense rewards. The project code is available at https://github.com/Arya87/SLOPE.
    
[^368]: 从结果到表征：通过积分策略梯度追踪推理机制

    From Outcome to Representation: Tracing Reasoning Mechanisms through Integrated Policy Gradient

    [https://arxiv.org/abs/2602.02313](https://arxiv.org/abs/2602.02313)

    本文提出无需训练的积分策略梯度（IPG）框架，通过将结果层面的不可微分行为信号沿整个推理轨迹反向传播并利用路径积分，在组件层面精准追踪神经元或稀疏特征对LLM多步推理结果的长程影响。

    

    大型语言模型（LLMs）展现出卓越的推理能力，然而驱动这些多步推理过程的内部机制仍然不透明。现有的机制可解释性方法通常依赖于表面的文本模式共现，或仅能捕捉短期效应，难以追踪神经元或稀疏特征等内部组件在多步推理中的长程、结果驱动的影响。本文提出了积分策略梯度，这是一种新颖的无需训练的框架，将基于策略的归因方法引入LLM推理的机制可解释性研究。该方法基于“结果导向”和“序贯影响感知”两项原则，将结果层面的不可微分行为信号（如推理结果的正确性）反向传播至整个推理轨迹，并利用路径积分在组件层面分离出与结果相关的中介因素。

    arXiv:2602.02313v3 Announce Type: replace  Abstract: Large Language Models (LLMs) demonstrate remarkable reasoning capabilities, yet the internal mechanisms driving these multi-step processes remain opaque. Existing mechanistic interpretability approaches often rely on superficial text-pattern co-occurrences or capture only short-term effects, struggling to trace the long-horizon, outcome-driven influence of internal components such as neurons or sparse features in multi-step reasoning.In this paper, we introduce Integrated Policy Gradient (IPG), a novel, training-free framework that brings policy-based attribution into mechanistic interpretability for LLM reasoning. Grounded in \textit{outcome-oriented} and \textit{sequential-influence-aware} principles, it backpropagates outcome-level, non-differentiable behavioral signals (e.g., reasoning outcome correctness) through entire inference trajectories, utilizing path integration to isolate outcome-relevant mediators at component level fo
    
[^369]: OpInf-LLM：通过算子推断利用大语言模型求解参数化偏微分方程

    OpInf-LLM: Parametric PDE Solving with LLMs via Operator Inference

    [https://arxiv.org/abs/2602.01493](https://arxiv.org/abs/2602.01493)

    提出OpInf-LLM框架，将算子推断与大语言模型结合，仅需少量解数据即可准确求解参数化偏微分方程，并能泛化到未见过的参数与边界条件，缓解了执行成功率与数值精度之间的权衡。

    

    求解多样化的偏微分方程（PDE）是科学与工程领域的基础任务。大语言模型（LLM）在代码生成、符号推理和工具使用方面已展现出强大能力，但在异构环境下可靠地求解PDE仍然具有挑战性。此前基于LLM的代码生成研究以及基于Transformer的PDE学习基础模型已取得可喜进展。然而，执行成功率与数值精度之间始终存在权衡，尤其是在需要对未见过的参数和边界条件进行泛化时。在本工作中，我们提出了OpInf-LLM，一个通过算子推断实现大语言模型参数化PDE求解的框架。该框架利用少量解数据即可对多样化的PDE实例（包括未见过的参数和配置）进行准确预测，并可与LLM无缝集成，实现自然语言交互（原文在此截断）。

    arXiv:2602.01493v3 Announce Type: replace-cross  Abstract: Solving diverse partial differential equations (PDEs) is fundamental in science and engineering. Large language models (LLMs) have demonstrated strong capabilities in code generation, symbolic reasoning, and tool use, but reliably solving PDEs across heterogeneous settings remains challenging. Prior work on LLM-based code generation and transformer-based foundation models for PDE learning has shown promising advances. However, a persistent trade-off between execution success rate and numerical accuracy arises, particularly when generalization to unseen parameters and boundary conditions is required. In this work, we propose OpInf-LLM, an LLM parametric PDE solving framework via operator inference. The proposed framework leverages small amounts of solution data to enable accurate prediction of diverse PDE instances, including unseen parameters and configurations, and provides seamless integration with LLMs for natural language t
    
[^370]: 基于前缀重要性比率稳定离策略大语言模型优化

    Stabilizing Off-policy LLM Optimization with Prefix Importance Ratio

    [https://arxiv.org/abs/2601.22718](https://arxiv.org/abs/2601.22718)

    该论文发现token级别的重要性采样比率无法修正前缀分布偏移，并提出用累积前缀重要性比率对前缀边缘化策略梯度进行精确修正，从而在大策略滞后下稳定大语言模型的离策略强化学习优化。

    

    强化学习（RL）后训练在激发大语言模型（LLM）的推理行为方面日益展现出强大的能力。为了提高训练效率，通常以离策略的方式使用较旧的采样策略生成rollout样本，然后用于更新当前的目标策略。为了纠正由此产生的采样策略与目标策略之间的偏差，大多数现有的RL目标依赖于token级别的重要性采样比率。然而，我们观察到在较大的策略滞后情况下，现有的重要性加权目标存在训练动态不稳定的问题。在本文中，我们重新审视了离策略条件下的LLM策略优化，并证明了对前缀边缘化策略梯度的精确修正应该是累积前缀重要性比率，而token级别的加权并不能显式地纠正这种前缀分布偏移。为了在大规模离策略漂移下稳定LLM优化，我们提出了一种简单的（摘要在此处截断）

    arXiv:2601.22718v2 Announce Type: replace  Abstract: Reinforcement learning (RL) post-training has increasingly demonstrated strong ability to elicit reasoning behaviors in large language models (LLMs). For training efficiency, rollouts are typically generated in an off-policy manner using an older sampling policy and then used to update the current target policy. To correct the resulting discrepancy between the sampling and target policies, most existing RL objectives rely on a token-level importance sampling ratio. However, we observe unstable training dynamics in existing importance-weighted objectives under large policy lag. In this paper, we revisit LLM policy optimization under off-policy conditions and show that the exact correction for the prefix-marginal policy gradient is the cumulative prefix importance ratio. Token-level weighting does not explicitly correct for this prefix distribution shift. To stabilize LLM optimization under large off-policy drift, we propose a simple y
    
[^371]: 利用软提示在联邦提示调优中实施隐私攻击

    Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning

    [https://arxiv.org/abs/2601.06641](https://arxiv.org/abs/2601.06641)

    提出了PromptMIA——一种针对联邦提示调优的新型成员推理攻击，恶意服务器通过注入对抗性构造的软提示并利用其训练更新，能够高效地推断出客户端私有数据集中的成员信息，显著优于现有联邦学习攻击方法。

    

    成员推理攻击（MIAs）在联邦学习（FL）中构成了严重的隐私威胁。尽管MIAs在标准联邦学习场景下已被广泛研究，但近期向联邦微调的转变引入了新的、在很大程度上尚未被探索的攻击面。在本工作中，我们证明了联邦提示调优——一种通过轻量级输入前缀来适配预训练基础模型的方法——暴露了一种新颖且高效的成员推理攻击途径。我们提出了PromptMIA，一种专为联邦提示调优量身定制的成员推理攻击：恶意服务器注入对抗性构造的提示，并在协作训练过程中利用这些提示的更新来判断某个目标数据点是否属于客户端的私有数据集。我们通过安全博弈对这一威胁进行了形式化定义，并证明PromptMIA在多种基准数据集上均能持续取得较高的攻击优势，显著优于当前最先进的联邦成员推理攻击方法。

    arXiv:2601.06641v4 Announce Type: replace  Abstract: Membership inference attacks (MIAs) pose a serious privacy threat in federated learning (FL). While MIAs have been extensively studied in standard FL, the recent shift toward federated fine-tuning introduces new and largely unexplored attack surfaces. In this work, we show that federated prompt-tuning, which adapts pre-trained foundation models using lightweight input prefixes, exposes a novel and effective vector for membership inference. We propose PromptMIA, a membership inference attack tailored to federated prompt-tuning, in which a malicious server introduces adversarially crafted prompts and exploits their updates during collaborative training to determine whether a target data point belongs to a client's private dataset. We formalize this threat via a security game and demonstrate that PromptMIA achieves consistently high attack advantage across diverse benchmark datasets, substantially outperforming current SOTA federated MI
    
[^372]: 利用地球基础模型的AlphaEarth嵌入提升水文模型的模拟性能

    Utilizing Earth Foundation Models to Enhance the Simulation Performance of Hydrological Models with AlphaEarth Embeddings

    [https://arxiv.org/abs/2601.01558](https://arxiv.org/abs/2601.01558)

    本研究将AlphaEarth嵌入作为静态流域表征用于无观测流域的径流预测，发现其样本外泛化鲁棒性优于传统CAMELS流域属性。

    

    无观测流域的准确预测（PUB）需要在水文条件各异的区域之间进行有效迁移。尽管深度学习推动了大样本水文建模的发展，但其跨区域泛化能力在一定程度上依赖于用于引导模型行为的静态流域描述符。地球基础模型（EFMs）通过从多模态卫星数据中学习环境表征，提供了一种有前景的替代方案。在本研究中，我们评估了64维的AlphaEarth Foundations（AEF）嵌入作为一种新的静态流域表征形式，并基于CAMELS数据集，在样本内（IS）和样本外（OOS）两种设置下评估了其对径流预测的价值。结果表明，尽管AEF嵌入的样本内性能与CAMELS属性几乎相当，但其样本外鲁棒性更高。互信息分析揭示了AEF嵌入与传统流域属性之间的统计关联……

    arXiv:2601.01558v3 Announce Type: replace-cross  Abstract: Accurate prediction in ungauged basins (PUB) requires effective transfer across hydrologically diverse regions. While deep learning has advanced large-sample hydrological modeling, its cross-regional generalization depends in part on the static basin descriptors used to guide model behavior. Earth Foundation Models (EFMs) provide a promising alternative by learning environmental representations from multi-modal satellite data. In this study, we evaluate the 64-dimensional AlphaEarth Foundations (AEF) embeddings as a new form of static basin representation and assess their value for streamflow prediction under both in-sample (IS) and out-of-sample (OOS) settings using the CAMELS dataset. Results show that AEF embeddings offer higher OOS robustness than CAMELS attributes, despite nearly identical IS performance. Mutual-information analysis reveals statistical associations between AEF embeddings and conventional catchment attribut
    
[^373]: 用于求解参数化优化问题的最优性信息神经网络

    Optimality-Informed Neural Networks for Solving Parametric Optimization Problems

    [https://arxiv.org/abs/2512.20270](https://arxiv.org/abs/2512.20270)

    提出OptINNs方法，通过KKT残差损失与内嵌约束的输出激活函数，利用神经网络学习参数到原始最优解及对偶变量的映射，从而高效求解参数化优化问题并大幅降低数据需求。

    

    许多工程任务需要求解一族以场景特定变量为参数的非线性约束优化问题，其计算开销很大，尤其是当解需要在变化剧烈的参数值范围内计算时，例如在实时控制或基于模型的设计中。因此，我们提出利用神经网络学习从参数到原始最优解及其对应对偶变量的映射，与网格化方法相比可提供稠密的估计。我们的方法——最优性信息神经网络，结合了：(i) 在标准约束规范假设下惩罚违反一阶最优性条件（KKT条件）的残差损失；(ii) 通过构造方式强制满足简单不等式约束（如箱型/正性约束）的针对具体问题的输出激活函数。这一设计减少了数据需求，并支持对偶变量的预测……

    arXiv:2512.20270v2 Announce Type: replace-cross  Abstract: Many engineering tasks require solving families of nonlinear constrained optimization problems, parametrized in setting-specific variables. This is computationally demanding, particularly if solutions have to be computed across strongly varying parameter values, e.g., in real-time control or for model-based design. Thus, we propose to learn the mapping from parameters to the primal optimal solutions and to their corresponding duals using neural networks, giving a dense estimation in contrast to gridded approaches. Our approach, Optimality-Informed Neural Networks (OptINNs), combines (i) a KKT-residual loss that penalizes violations of the first-order optimality conditions under standard constraint qualification assumptions, and (ii) problem-specific output activations that enforce simple inequality constraints (e.g., box-type/positivity) by construction. This design reduces data requirements, allows the prediction of dual varia
    
[^374]: 用于ATLAS高层触发中快速堆叠喷注剔除的Deep Sets与事件级最大似然估计

    Deep Sets and event-level maximum-likelihood estimation for fast pile-up jet rejection at the ATLAS High Level Trigger

    [https://arxiv.org/abs/2512.10819](https://arxiv.org/abs/2512.10819)

    本文提出DIPz模型，利用Deep Sets从带电粒子径迹中以高斯似然形式估计喷注的束流轴原点及其不确定性，并将各喷注预测组合成事件级最大似然判别量，为ATLAS高层触发实现快速且稳健的堆叠喷注剔除。

    

    在LHC中，同一次束团穿越会发生多次质子-质子对撞（即“堆叠”，pile-up），在Run 3期间每次束团穿越的平均对撞次数可达约60次，而在高亮度LHC阶段将高达约200次。其直接后果是每次束团穿越中的强子活动增加，使得识别与多喷注末态相关的感兴趣对撞变得更加困难。本工作介绍了DIPz，这是一种基于Deep Sets、具有不确定性感知能力的机器学习模型，它利用与喷注匹配的带电粒子径迹集合，以高斯似然的形式估计每个喷注沿束流轴方向的原点。随后，将DIPz对每个喷注的预测组合成一个基于似然的事件级判别量，该判别量经过优化，用于筛选与产生给定喷注多重数的单次对撞相容的事件。这种组合方法为堆叠剔除提供了一种稳健且计算高效的方法。

    arXiv:2512.10819v2 Announce Type: replace-cross  Abstract: Multiple proton-proton interactions occur in the same bunch crossing at the LHC ("pile-up"), with the mean number of interactions per bunch crossing reaching up to about 60 in Run 3 and up to about 200 at the High-Luminosity LHC. As a direct consequence, hadronic activity within each bunch crossing increases, making it more difficult to identify collisions of interest pertaining to multijet final states. This work introduces DIPz, an uncertainty-aware Deep Sets-based machine-learning model that estimates each jet's origin along the beam axis as a Gaussian likelihood using the set of charged-particle tracks matched to the jet. Next, the DIPz predictions for each jet are combined into a likelihood-based event-level discriminant, which is optimised to select events compatible with a single collision producing a given jet multiplicity. This combined approach provides a robust and computationally efficient method for pile-up rejecti
    
[^375]: 从实验室到现实：深度学习模型与LLMs在漏洞检测中的实用评估

    From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection

    [https://arxiv.org/abs/2512.10485](https://arxiv.org/abs/2512.10485)

    本文系统评估了四种深度学习漏洞检测模型在多个数据集上的表现，通过跨数据集实验和可视化分析揭示了基准测试与真实世界应用之间的性能差距，强调了模型泛化能力的挑战。

    

    基于深度学习（DL）的漏洞检测方法在基准数据集上表现出强大性能，但其在现实世界中的有效性仍未得到充分探索。近期研究表明，基于图神经网络和Transformer的模型，包括大型语言模型（LLMs），在精选的基准数据集上评估时能产生令人满意的结果。这些数据集通常具有相似的数据分布，可能包含合成样本、启发式标签或标签噪声。在本研究中，我们系统评估了四种代表性DL模型——Devign、ReVeal、LineVul和VulBERTa——在四个代表性数据集上：Juliet、Devign、BigVul和ICVul。每个模型在每个数据集上独立训练，并使用t-SNE和质心距离分析这些模型采用的基于图和CodeBERT的表示，以检查与漏洞相关的模式。为了评估现实应用的适用性，我们进一步进行了跨数据集实验和实际场景模拟。

    arXiv:2512.10485v3 Announce Type: replace-cross  Abstract: Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that graph neural network-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by similar data distributions and may contain synthetic samples, heuristic labels, or labeling noise. In this study, we systematically evaluate four representative DL models---Devign, ReVeal, LineVul, and VulBERTa---across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each dataset, and the graph-based and CodeBERT representations adopted by these models are analyzed using t-SNE and centroid distance to examine vulnerability-related patterns. To assess realistic applicabil
    
[^376]: 建筑材料价格预测的细粒度框架：计量经济学与机器学习方法

    A Granular Framework for Construction Material Price Forecasting: Econometric and Machine-Learning Approaches

    [https://arxiv.org/abs/2512.09360](https://arxiv.org/abs/2512.09360)

    本研究提出了一个基于CSI MasterFormat的建筑材料价格细粒度预测框架，能够在六位数字章节级别整合原材料价格与宏观经济指标等解释变量，并对比评估了LSTM、ARIMA、VECM、Transformer和Chronos-Bolt五种时间序列模型的预测性能。

    

    建筑材料价格的持续波动给成本估算、预算编制和项目交付带来了重大风险，凸显了对细粒度且可扩展预测方法的迫切需求。本研究开发了一个预测框架，该框架以建筑规范协会（CSI）MasterFormat作为目标数据结构，能够在六位数字章节级别进行预测，并支持对各类建筑材料的详细成本预测。为提高预测精度，该框架整合了原材料价格、商品指数和宏观经济指标等解释变量。研究在基线配置（仅使用CSI数据）和包含解释变量的扩展版本下，评估了五种时间序列模型：长短期记忆网络（LSTM）、自回归积分移动平均模型（ARIMA）、向量误差修正模型（VECM）、原生Transformer和Chronos-Bolt。

    arXiv:2512.09360v2 Announce Type: replace  Abstract: The persistent volatility of construction material prices poses significant risks to cost estimation, budgeting, and project delivery, underscoring the urgent need for granular and scalable forecasting methods. This study develops a forecasting framework that leverages the Construction Specifications Institute (CSI) MasterFormat as the target data structure, enabling predictions at the six-digit Section level and supporting detailed cost projections across a wide spectrum of building materials. To enhance predictive accuracy, the framework integrates explanatory variables such as raw material prices, commodity indexes, and macroeconomic indicators. Five time-series models, long short-term memory (LSTM), autoregressive integrated moving average (ARIMA), vector error correction model (VECM), vanilla Transformer, and Chronos-Bolt, were evaluated under both baseline configurations (using CSI data only) and extended versions with explanat
    
[^377]: 行星内部准地转湍流次网格尺度模型的在线学习

    Online learning of subgrid-scale models for quasi-geostrophic turbulence in planetary interiors

    [https://arxiv.org/abs/2511.14581](https://arxiv.org/abs/2511.14581)

    本文首次将可微分求解器参与的在线端到端机器学习方法应用于有界旋转区域中的二维准地转湍流次网格尺度建模，突破了以往研究局限于无边界理想化区域的限制，能够捕捉与行星内部相关的纬向急流、罗斯贝波等关键动力学特征。

    

    用于次网格尺度（SGS）建模的机器学习方法目前已在大气和海洋应用中得到了广泛认可。其中，可微分求解器参与训练的在线端到端学习展现出尤为可观的前景。然而，现有研究大多局限于理想化的周期性区域，缺乏机械边界，因而无法处理与行星内部相关的有界旋转流动的动力学问题。本文研究了快速旋转的环形有界区域中的二维准地转湍流，考察了三种不同的配置，改变了容器的几何形状和旋转速率。该系统展现出纬向急流、罗斯贝波等关键特征，并且在球壳几何结构中还出现了急流缓慢的准周期性向内漂移。纬向与非纬向流动的谱性质可以在纬向湍流理论的框架下得到理解。

    arXiv:2511.14581v3 Announce Type: replace-cross  Abstract: Machine learning approaches to subgrid-scale (SGS) modeling are now well established in atmospheric and oceanic applications. Among these, online end-to-end learning, where the differentiable solver participates in the training, has shown particular promise. Yet existing studies are largely restricted to idealised periodic domains, with no mechanical boundaries, precluding them from addressing the dynamics of bounded rotating flows relevant to planetary interiors. Here we consider two-dimensional quasi-geostrophic turbulence in a rapidly rotating annular bounded domain. We examine three configurations varying the geometry of the container and the rotation rate. The system exhibits key features such as zonal jets, Rossby waves, and, in the spherical shell geometry, a slow quasi-periodic inward drift of the jets. The spectral properties of the zonal and non-zonal flow can be understood in the framework of zonostrophic turbulence 
    
[^378]: 基于策略性不精确随机梯度下降的双层学习

    Bilevel Learning via Strategically Inexact Stochastic Gradient Descent

    [https://arxiv.org/abs/2511.06774](https://arxiv.org/abs/2511.06774)

    该论文针对计算成像中的双层优化问题，证明了实践者实际采用的不精确随机梯度调度方案可保证收敛，并在超梯度精度与计算成本线性相关、精度衰减衰减等条件下建立了收敛速率。

    

    双层优化是计算成像领域中进行高维超参数调优的核心工具；其重要应用包括学习数据自适应正则化器以及在变分正则化中优化前向算子。然而，由于问题的嵌套特性，无法获得关于参数的精确梯度（即所谓的超梯度）：超梯度的精度与计算成本呈线性关系，因此需要能够在不产生不必要精度开销的前提下求解该问题的算法。在可学习正则化流程中，内层重建求解器通常仅运行固定次数的迭代，导致超梯度精度不受控制，使得现有的收敛理论不再适用。在本工作中，我们分析了实践者实际使用的调度方案：我们证明了不精确随机双层优化的收敛性，并在精度衰减等多种条件下建立了收敛速率……

    arXiv:2511.06774v2 Announce Type: replace-cross  Abstract: Bilevel optimization is a central tool in computational imaging for high-dimensional hyperparameter tuning; prominent applications include learning data-adaptive regularizers and optimizing forward operators in variational regularization. However, due to the nested nature of the problem, exact gradients with respect to the parameters (so-called hypergradients) are not available: the precision of the hypergradient is linearly related to the computational cost, thus requiring algorithms which can solve the problem without unnecessary precision. In learned-regularizer pipelines, the inner reconstruction solver is typically run for a fixed number of iterations, leaving the hypergradient accuracy uncontrolled and existing convergence theory inapplicable. In this work, we analyze the schedules that practitioners actually run: we prove convergence of inexact stochastic bilevel optimization and establish rates under both decaying accur
    
[^379]: MvBody：基于多视角的混合Transformer网络，利用光学3D人体扫描实现可解释的剖腹产预测

    MvBody: Multi-View-Based Hybrid Transformer Using Optical 3D Body Scan for Explainable Cesarean Section Prediction

    [https://arxiv.org/abs/2511.03212](https://arxiv.org/abs/2511.03212)

    提出了一种基于多视角混合Transformer的MvBody模型，仅利用妊娠31-38周期间的3D光学人体扫描和自我报告医疗数据，即可实现可解释的剖腹产风险预测，适用于医疗资源有限的环境。

    

    准确评估剖腹产（CS）分娩风险至关重要，尤其是在医疗资源有限的环境中，因为这些地区获得医疗保健的机会往往受限。早期可靠的风险预测有助于做出更明智的产前护理决策，并可改善孕产妇和新生儿的结局。然而，现有的大多数预测模型都是为住院分娩期间使用而设计的，依赖于在资源有限或居家环境中通常无法获得的参数。在本研究中，我们开展了一项试点研究，以检验利用3D体型进行剖腹产风险评估的可行性，以便未来能够使用更经济实惠的通用设备。我们提出了一种新颖的基于多视角的Transformer网络MvBody，该网络仅使用自我报告的医疗数据和在妊娠第31周至第38周之间获得的3D光学人体扫描来预测剖腹产风险。为了提高训练效率和模型在数据稀缺情况下的泛化能力（摘要不完整）……

    arXiv:2511.03212v1 Announce Type: cross  Abstract: Accurately assessing the risk of cesarean section (CS) delivery is critical, especially in settings with limited medical resources, where access to healthcare is often restricted. Early and reliable risk prediction allows better-informed prenatal care decisions and can improve maternal and neonatal outcomes. However, most existing predictive models are tailored for in-hospital use during labor and rely on parameters that are often unavailable in resource-limited or home-based settings. In this study, we conduct a pilot investigation to examine the feasibility of using 3D body shape for CS risk assessment for future applications with more affordable general devices. We propose a novel multi-view-based Transformer network, MvBody, which predicts CS risk using only self-reported medical data and 3D optical body scans obtained between the 31st and 38th weeks of gestation. To enhance training efficiency and model generalizability in data-sc
    
[^380]: ESS-Flow：作为源空间中贝叶斯推断的免训练引导方法

    ESS-Flow: Training-free guidance as Bayesian inference in source space

    [https://arxiv.org/abs/2510.05849](https://arxiv.org/abs/2510.05849)

    ESS-Flow是一种免训练、无梯度的引导方法，通过椭圆切片采样利用流模型源分布的高斯先验直接在源空间进行贝叶斯推断，即使梯度不可用（如不可微分模拟观测或量化过程）也能引导预训练流模型完成条件生成任务。

    

    引导预训练的基于流的生成模型进行条件生成或产生具有期望目标属性的样本，能够在无需在成对数据上重新训练的情况下解决多种任务。我们提出了ESS-Flow，这是一种无梯度方法，它利用基于流的模型中源分布的高斯先验，使用椭圆切片采样直接在源空间中执行贝叶斯推断。ESS-Flow只需要通过生成模型和观测过程进行前向传播，使其即使在梯度不可用的情况下也适用，例如存在不可微分的基于模拟的观测或生成过程中的量化时。我们在多个科学应用中展示了其有效性，包括生成具有目标属性的材料、具有更优可设计性的蛋白质骨架结构，以及可合成的类药物分子。

    arXiv:2510.05849v2 Announce Type: replace  Abstract: Guiding pretrained flow-based generative models for conditional generation or to produce samples with desired target properties enables solving diverse tasks without retraining on paired data. We present ESS-Flow, a gradient-free method that leverages the Gaussian prior of the source distribution in flow-based models to perform Bayesian inference directly in the source space using elliptical slice sampling. ESS-Flow only requires forward passes through the generative model and observation process, making it applicable even when gradients are unavailable, such as with non-differentiable simulation-based observations or quantization in the generation process. We demonstrate its effectiveness on scientific applications including generating materials with target properties, protein backbone structures with improved designability, and synthesizable, drug-like molecules.
    
[^381]: UpSafe°C：面向大语言模型可控安全性的升级改造方法

    UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models

    [https://arxiv.org/abs/2510.02194](https://arxiv.org/abs/2510.02194)

    提出UpSafe°C框架，通过将安全关键层升级为稀疏MoE结构、利用路由器作为软防护栏、结合两阶段SFT和可调节的安全温度机制，在保持通用能力的同时实现大语言模型推理时可灵活控制的安全防护。

    

    大语言模型（LLMs）在广泛的任务中取得了显著进展，但仍然容易受到安全风险的威胁，例如有害内容生成和越狱攻击。现有的安全技术——包括外部防护栏、推理时引导和后训练对齐——在平衡安全性、实用性和可控性方面各有局限。在这项工作中，我们提出了UpSafe°C，一个通过安全感知升级改造来增强大语言模型安全性的统一框架。我们的方法首先识别安全关键层，并将其升级改造为稀疏的混合专家结构，其中路由器充当软防护栏，选择性地激活原始MLP和新增的安全专家。我们进一步引入两阶段SFT策略，在保持通用能力的同时增强安全判别能力。为了在推理时实现灵活控制，我们引入了安全温度机制，允许……

    arXiv:2510.02194v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have achieved remarkable progress across a wide range of tasks, but remain vulnerable to safety risks such as harmful content generation and jailbreak attacks. Existing safety techniques---including external guardrails, inference-time guidance, and post-training alignment---each face limitations in balancing safety, utility, and controllability. In this work, we propose UpSafe$^\circ$C, a unified framework for enhancing LLM safety through safety-aware upcycling. Our approach first identifies safety-critical layers and upcycles them into a sparse Mixture-of-Experts (MoE) structure, where the router acts as a soft guardrail that selectively activates original MLPs and added safety experts. We further introduce a two-stage SFT strategy to strengthen safety discrimination while preserving general capabilities. To enable flexible control at inference time, we introduce a safety temperature mechanism, allowing 
    
[^382]: fev-bench：一个贴近现实的时序预测基准

    fev-bench: A Realistic Benchmark for Time Series Forecasting

    [https://arxiv.org/abs/2509.26468](https://arxiv.org/abs/2509.26468)

    提出了 fev-bench——一个涵盖七个领域共 100 个任务（其中 46 个带协变量）的贴近现实的时间序列预测基准，并配套推出强调可复现性与易集成的轻量级 Python 评估库 fev，通过自举置信区间等统计严谨的聚合方式实现可靠模型评估。

    

    基准测试的质量对于时间序列预测领域的有效评估和持续进步至关重要，尤其是在预训练模型兴起的背景下。现有的基准测试往往领域覆盖范围有限，或忽视了带协变量任务等真实世界场景。它们的聚合程序常常缺乏统计严谨性，导致观测到的性能差异究竟反映的是真实提升还是随机波动变得难以判断。许多基准测试缺乏一致的评估基础设施，或者过于僵化而难以集成到现有工作流程中。为了弥补这些不足，我们提出了 fev-bench，这是一个涵盖七个领域共 100 个预测任务的基准，其中 46 个任务包含协变量。作为对该基准的支持，我们引入了 fev——一个轻量级的 Python 预测评估库，强调可复现性以及与现有工作流程的集成。借助 fev，fev-bench 采用基于自举（bootstrap）置信区间的严谨聚合方法……

    arXiv:2509.26468v5 Announce Type: replace  Abstract: Benchmark quality is critical for meaningful evaluation and sustained progress in time series forecasting, particularly with the rise of pretrained models. Existing benchmarks often have limited domain coverage or overlook real-world settings such as tasks with covariates. Their aggregation procedures frequently lack statistical rigor, making it unclear whether observed performance differences reflect true improvements or random variation. Many benchmarks lack consistent evaluation infrastructure or are too rigid for integration into existing pipelines. To address these gaps, we propose fev-bench, a benchmark of 100 forecasting tasks across seven domains, including 46 with covariates. Supporting the benchmark, we introduce fev, a lightweight Python library for forecasting evaluation emphasizing reproducibility and integration with existing workflows. Using fev, fev-bench employs principled aggregation with bootstrapped confidence int
    
[^383]: 发现学习到的哈密顿系统中的相空间结构

    Discovering Phase Space Structure in Learned Hamiltonian Systems

    [https://arxiv.org/abs/2509.23003](https://arxiv.org/abs/2509.23003)

    提出 SPS-GAN 条件生成模型，将端口哈密顿网络与对抗生成相结合，可从轨迹和视频中统一学习保守、耗散及受迫系统的结构化动力学，并通过循环坐标损失在无需真实相空间的情况下恢复系统的自由度和正则坐标。

    

    力学启发的神经动力学模型提升了可解释性和物理一致性，但它们通常假设构型空间已知、物理参数固定，并且从给定初始条件出发进行确定性预测。我们提出结构化相空间生成对抗网络（SPS-GAN），这是一种条件生成模型，能够同时从笛卡尔轨迹和视频中学习结构化动力学。SPS-GAN 将端口哈密顿骨干网络与对抗式条件生成相结合，使保守系统、耗散系统和受迫系统能够在单一架构中得到表示。为了从原始观测中推断潜在的力学结构，我们采用循环坐标损失，在无需访问真实相空间的情况下同时恢复系统的自由度和正则坐标。在模拟系统和真实世界系统上的实验表明，SPS-GAN 在轨迹精度上优于监督动力学基线方法，并提升了视频一致性。

    arXiv:2509.23003v3 Announce Type: replace-cross  Abstract: Mechanics-inspired neural dynamical models improve interpretability and physical consistency, but they usually assume a known configuration space, fixed physical parameters, and deterministic prediction from a prescribed initial condition. We introduce Structured Phase Space GAN (SPS-GAN), a conditional generative model that learns structured dynamics from both Cartesian trajectories and video. SPS-GAN combines a port-Hamiltonian backbone with adversarial conditional generation, allowing conservative, dissipative, and forced systems to be represented in a single architecture. To infer latent mechanical structure from raw observations, we use a cyclic-coordinate loss that recovers both the system degrees of freedom and canonical coordinates without access to the true phase space. Experiments on simulated and real-world systems show that SPS-GAN improves trajectory accuracy over supervised dynamics baselines and video consistency
    
[^384]: 面向基于大语言模型社会模拟的种群对齐式人格生成

    Population-Aligned Persona Generation for LLM-based Social Simulation

    [https://arxiv.org/abs/2509.10127](https://arxiv.org/abs/2509.10127)

    该论文提出了一个系统性框架，通过从长期社交媒体数据生成叙事型角色、质量评估过滤低保真档案、并利用重要性采样与大五人格等参考心理测量分布对齐，来合成高质量且与真实人口分布对齐的角色集合，从而提升LLM社会模拟的真实性。

    

    大语言模型（LLM）的最新进展使得大规模、高保真度的社会模拟成为可能，为计算社会科学创造了新的机遇。然而，构建能够忠实反映现实世界人口多样性的角色（persona）集合仍然是一个关键挑战。现有研究往往强调智能体框架和模拟环境，而对角色生成以及不具代表性的角色集合所引入的偏差关注较少。本文提出了一个系统性框架，用于为LLM驱动的社会模拟合成高质量、与种群对齐的角色集合。我们的方法首先利用LLM从长期社交媒体数据中生成叙事型角色，随后通过严格的质量评估来过滤低保真度的角色档案。接着，我们应用重要性采样来实现与参考心理测量分布（如大五人格）的全局对齐。

    arXiv:2509.10127v3 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have enabled large-scale, high-fidelity social simulations, creating new opportunities for computational social science. However, constructing persona sets that faithfully reflect real-world population diversity remains a key challenge. Existing studies often emphasize agentic frameworks and simulation environments, while paying less attention to persona generation and the biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality 
    
[^385]: 基于图的分布式强化学习电网控制

    Power Grid Control with Graph-Based Distributed Reinforcement Learning

    [https://arxiv.org/abs/2509.02861](https://arxiv.org/abs/2509.02861)

    该论文提出了一种基于图的分布式强化学习框架，利用图神经网络编码电网拓扑信息，并通过高低层智能体协同实现了对现代电网的实时、可扩展控制。

    

    可再生能源接入的必要性，加之电网规模的不断扩大，给现代电网的控制带来了重大挑战。传统的基于人工和优化的控制系统在这种不断演变的环境下难以适应和扩展，这促使人们探索更加动态和分布式的控制策略。本工作提出了一种基于图的分布式强化学习框架，用于实时、可扩展的电网管理。所提出的架构由作用于各条输电线路的分布式低层智能体网络组成，并由一个高层管理智能体进行协调。该框架采用图神经网络（GNN）在单个低层智能体的观测中编码电网的拓扑信息。为了加速收敛并提高学习稳定性，该框架集成了模仿学习和基于势函数的奖励塑形。

    arXiv:2509.02861v2 Announce Type: replace  Abstract: The necessary integration of renewable energy sources, combined with the expanding scale of power networks, presents significant challenges in controlling modern power grids. Traditional control systems, which are human and optimization-based, struggle to adapt and to scale in such an evolving context, motivating the exploration of more dynamic and distributed control strategies. This work advances a graph-based distributed reinforcement learning framework for real-time, scalable grid management. The proposed architecture consists of a network of distributed low-level agents acting on individual power lines and coordinated by a high-level manager agent. A Graph Neural Network (GNN) is employed to encode the network's topological information within the single low-level agent's observation. To accelerate convergence and enhance learning stability, the framework integrates imitation learning and potential-based reward shaping. In contra
    
[^386]: 针对协作式多智能体强化学习的受限黑盒攻击

    Constrained Black-Box Attacks Against Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2508.09275](https://arxiv.org/abs/2508.09275)

    该论文提出了两种此前被忽视的受限黑盒攻击设置（仅能访问智能体观测或完全无法访问任何信息），通过生成使智能体环境感知错位的扰动来有效破坏协作式多智能体强化学习，且无需访问策略权重或训练代理策略，在22个环境中验证了攻击的有效性和样本效率。

    

    协作式多智能体强化学习（MARL）越来越多地被部署在关键领域，然而其对对抗性扰动的脆弱性尚未得到充分研究。大多数现有的针对MARL的对抗攻击依赖于乐观的假设，例如能够访问策略权重或能够训练代理策略。我们研究了能力受限的攻击者是否仍然能够有效攻击MARL智能体。特别是，我们研究了文献中被忽视的两种受限攻击设置：（i）攻击者仅能访问智能体的观测；（ii）攻击者完全无法访问任何信息。我们的方法生成的扰动会故意使受害智能体对环境的感知产生错位，从而阻止它们进行协作。我们在22个多样化的环境中进行了广泛的实验。我们的结果证明了我们攻击方法的有效性和样本效率。

    arXiv:2508.09275v3 Announce Type: replace  Abstract: Collaborative multi-agent reinforcement learning (MARL) is increasingly deployed in critical domains, yet its vulnerability to adversarial perturbations has not been thoroughly investigated. Most existing adversarial attacks against MARL rely on optimistic assumptions, such as access to policy weights or the ability to train surrogate policies. We investigate whether adversaries with limited capabilities remain effective against MARL agents. In particular, we study two restricted attack settings that are overlooked in the literature: (i) adversaries with access only to agent observations and (ii) adversaries with no access whatsoever. Our approach generates perturbations that intentionally misalign how victim agents perceive their environment, preventing them from collaborating. We conduct extensive experiments across a diverse set of 22 environments. Our results demonstrate both the effectiveness and sample efficiency of our attacks
    
[^387]: 面向分子表示学习的预训练分子嵌入模型基准测试

    Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning

    [https://arxiv.org/abs/2508.06199](https://arxiv.org/abs/2508.06199)

    本研究对28个预训练分子嵌入模型进行了大规模基准测试，发现除CheMeleon外所有神经模型均未超越经典ECFP计数指纹，且外部验证显示两者并无明确优劣，建议在新QSAR/QSPR研究中同时评估这两种方法。

    

    覆盖图神经网络、文本Transformer和混合架构的预训练分子嵌入模型已被广泛应用于药物发现领域，然而此前尚缺乏对其表示能力的大规模严格比较。我们在一个受控框架下，采用层次贝叶斯Bradley-Terry统计检验方法，在25个分类数据集上对28个模型进行了评估。结果显示，除CheMeleon外，所有神经模型均未能超越经典的ECFP计数指纹，CheMeleon是唯一被评为优于ECFP指纹的模型。然而，基于支架交叉验证和贝叶斯层次相关t检验的外部验证数据集的进一步测试表明，两种嵌入方法均不具备明确的优势。作为实用的结论，我们建议在新的QSAR/QSPR研究中同时评估ECFP指纹和CheMeleon嵌入。总体而言，就所有被评估的神经网络而言，这些结果也挑战了人们所认为的（神经模型的优势）。

    arXiv:2508.06199v5 Announce Type: replace-cross  Abstract: Pretrained molecular embedding models spanning graph neural networks, text transformers, and hybrid architectures have been widely used in drug discovery, yet no large-scale, rigorous comparison of their representations existed. We evaluated 28 models across 25 classification datasets under a controlled framework using hierarchical Bayesian Bradley-Terry statistical testing. All neural models except CheMeleon failed to outperform the classical ECFP count fingerprint; CheMeleon was the only model rated as better. However, further testing on external validation datasets, based on scaffold cross-validation and a Bayesian hierarchical correlated t-test, revealed no definite advantage of either embedding method. As a practical takeaway, we recommend evaluating both ECFP fingerprints and CheMeleon embeddings in novel QSAR/QSPR studies. Overall, in the context of all neural networks evaluated, these results also challenge the perceive
    
[^388]: 当评论稀缺时：现实稀疏评论场景下感知结构的评论补全推荐方法

    When Reviews Are Scarce: Structure-Aware Review Imputation for Recommendation in Realistic Sparse Review Settings

    [https://arxiv.org/abs/2508.01128](https://arxiv.org/abs/2508.01128)

    提出TWISTER统一框架，通过在文本边图上联合建模语义与结构信号来合成缺失评论，从而在用户很少写评论的现实稀疏场景下提升评论感知推荐系统的性能。

    

    文本评论为推荐系统提供了细粒度的偏好信号和更强的可解释性。然而，在现实场景中，用户很少留下评论，导致严重的稀疏性，削弱了部分可观测条件下评论感知推荐器的有效性。一个自然的补救方法是合成缺失的评论，以恢复这些推荐器所依赖的文本信号。然而，传统的补全技术，如矩阵补全和基于大语言模型的增强方法，要么在将文本嵌入为向量时丢失上下文语义，要么忽视了用户-物品图上的结构依赖关系。为了解决这些缺陷，我们提出了TWISTER（ToWards Imputation on Sparsity with Textual Edge Graph Representation），这是一个用于稀疏文本推荐的统一框架，通过联合建模语义信号和结构信号来合成缺失评论。具体而言，我们将用户-物品交互表示为……

    arXiv:2508.01128v2 Announce Type: replace-cross  Abstract: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of review-aware recommenders under partial observability. A natural remedy is to synthesize the missing reviews to restore the textual signal these recommenders rely on. Yet, conventional imputation techniques, such as matrix completion and LLM-based augmentation, either lose contextualized semantics by embedding texts into vectors or overlook structural dependencies on the user-item graph. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework for sparse-text recommendation that synthesizes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as 
    
[^389]: 面向机器学习任务的可扩展数据源选择

    Scalable Source Selection for Machine Learning Tasks

    [https://arxiv.org/abs/2507.22186](https://arxiv.org/abs/2507.22186)

    提出了Splice系统，一种借鉴基因剪接思想的目标与成本感知数据源选择方法，通过识别能为下游任务提供额外信号的数据源来最大化任务效用，并开发了利用数据画像和梯度对齐的两种加速变体Splice-DP和Splice-Grad。

    

    给定一个查询表，现有的数据湖发现技术通过模式相似性和连接性来识别与查询表相关的数据源。然而，发现的数据源并不总是对下游分析有用；其中一些可以提高任务泛化能力，而另一些则可能无意中引入有害噪声或分布偏移。我们将数据源选择形式化为一个寻找被发现数据源子集的问题，该子集能够最大化下游任务的效用。我们提出了Splice，一个目标和成本感知的数据源选择系统，它借鉴基因剪接的思想来识别那些能提供所有数据源并集所缺失的额外信号的数据源。为了加速这一过程，我们开发了两个代理变体Splice-DP和Splice-Grad，它们分别通过在元学习方法中利用数据画像以及梯度对齐来估计子集对任务效用的边际贡献。

    arXiv:2507.22186v3 Announce Type: replace-cross  Abstract: Given a query table, existing data lake discovery techniques seek to identify data sources related to the query table through schema similarity and joinability. The discovered sources, however, are not always useful for downstream analyses; while some improve task generalization, some have the potential to inadvertently introduce harmful noise or distribution shifts. We formalize source selection as the problem of finding a subset of the discovered sources that maximizes downstream task utility. We propose Splice, a target- and cost-aware source selection system that adopts the idea of gene splicing to identify sources which provide additional signals that the union of all sources misses. To speed up the process, we develop two surrogate variants, Splice-DP and Splice-Grad, that estimate the marginal contribution of subsets to task utility by leveraging data profiles in a meta-learning approach and gradient alignment, respectiv
    
[^390]: 面向结构化税则编码预测的领域自适应小型语言模型

    Domain-Adaptive Small Language Models for Structured Tax Code Prediction

    [https://arxiv.org/abs/2507.10880](https://arxiv.org/abs/2507.10880)

    该论文提出了一种基于编码器-解码器架构的领域自适应小型语言模型，能够从非结构化产品和服务数据中顺序预测层级化的税则编码（如HSN/SAC），从而提升跨国公司税务合规的准确性。

    

    跨国公司每天处理数千笔交易，每笔交易都必须遵守因司法管辖区而异且往往十分细微的税务法规。确定产品和服务税则编码（如HSN或SAC）是税务合规中的一个重要应用场景，准确确定这些编码对于避免税务处罚至关重要。本文提出了一种采用编码器-解码器架构的领域自适应小型语言模型（SLM），用于增强产品和服务税则编码的预测效果。在该方法中，我们利用非结构化的产品和服务数据来解决预测层级化税则编码序列的问题。我们采用基于编码器-解码器架构的SLM，因为该架构能够顺序生成税则编码，从而捕获税则编码内部存在的层级依赖关系。我们的实验表明，编码器-解码器SLM可以成功地应用于税则编码的顺序预测任务。

    arXiv:2507.10880v3 Announce Type: replace-cross  Abstract: Every day, multinational firms process thousands of transactions, each of which must adhere to tax regulations that vary by jurisdiction and are often nuanced. The determination of product and service tax codes, such as HSN or SAC is a major use case in Tax compliance. An accurate determination of such codes is imperative to avoid any tax penalties. This paper proposes a domain-adaptive small language model (SLM) with an encoder-decoder architecture for the enhanced prediction of product and service tax codes. In this approach, we address the problem of predicting hierarchical tax code sequences using unstructured product and services data. We employ an SLM based upon encoder-decoder architecture as this enables sequential generation of tax codes to capture the hierarchical dependencies present within the tax codes. Our experiments demonstrate that encoder-decoder SLMs can be successfully applied to the sequential prediction of
    
[^391]: 面向目标的因果学习序贯贝叶斯实验设计

    Goal-Oriented Sequential Bayesian Experimental Design for Causal Learning

    [https://arxiv.org/abs/2507.07359](https://arxiv.org/abs/2507.07359)

    GO-CBED是一个面向目标的贝叶斯序贯因果实验设计框架，通过变分下界估计与Transformer策略网络直接最大化用户指定因果量的期望信息增益，实现非短视且实时的干预决策，在因果推理与发现任务中优于现有基线。

    

    我们提出了GO-CBED，一个面向目标的序贯因果实验设计贝叶斯框架。与旨在通过选择干预来推断完整因果模型的传统方法不同，GO-CBED直接最大化用户指定的目标因果量上的期望信息增益（EIG），从而实现更有针对性、更高效的实验。该框架既具有非短视性（对整个干预序列进行优化），又具有面向目标性（仅关注与因果查询相关的模型方面）。为解决精确EIG计算的难解性问题，我们引入了一个变分下界估计器，通过基于Transformer的策略网络和基于归一化流的变分后验进行联合优化。由此得到的策略可通过摊销网络实现实时决策。我们证明，GO-CBED在多种因果推理与因果发现任务中始终优于现有基线方法。

    arXiv:2507.07359v2 Announce Type: replace-cross  Abstract: We present GO-CBED, a goal-oriented Bayesian framework for sequential causal experimental design. Unlike conventional approaches that select interventions aimed at inferring the full causal model, GO-CBED directly maximizes the expected information gain (EIG) on user-specified causal quantities of interest, enabling more targeted and efficient experimentation. The framework is both non-myopic, optimizing over entire intervention sequences, and goal-oriented, targeting only model aspects relevant to the causal query. To address the intractability of exact EIG computation, we introduce a variational lower bound estimator, optimized jointly through a transformer-based policy network and normalizing flow-based variational posteriors. The resulting policy enables real-time decision-making via an amortized network. We demonstrate that GO-CBED consistently outperforms existing baselines across various causal reasoning and discovery ta
    
[^392]: 图神经网络沿最短路径遇见序列模型：一种表达力强的链路预测方法

    GNNs Meet Sequence Models Along the Shortest-Path: an Expressive Method for Link Prediction

    [https://arxiv.org/abs/2507.07138](https://arxiv.org/abs/2507.07138)

    SP4LP提出将图神经网络节点编码与最短路径上的序列建模相结合，能够以高效的方式捕获链路预测中表达力强的多跳关系模式。

    

    图神经网络（GNN）通常难以捕获对准确链路预测至关重要的链路特定结构模式，因为其以节点为中心的消息传递机制忽略了连接一对节点的子图结构。现有的注入此类结构上下文的方法要么计算成本高昂，要么依赖于无法建模多跳依赖关系的简单启发式方法（例如共同邻居计数）。我们提出了SP4LP（Shortest Path for Link Prediction，用于链路预测的最短路径），这是一种将基于GNN的节点编码与最短路径上的序列建模相结合的新颖框架。具体而言，SP4LP首先应用GNN计算所有节点的表示，然后提取每个候选节点对之间的最短路径，并使用序列模型处理所得的节点嵌入序列。这种设计使SP4LP能够以计算高效的方式捕获具有表达力的多跳关系模式。

    arXiv:2507.07138v2 Announce Type: replace  Abstract: Graph Neural Networks (GNNs) often struggle to capture the link-specific structural patterns crucial for accurate link prediction, as their node-centric message-passing schemes overlook the subgraph structures connecting a pair of nodes. Existing methods to inject such structural context either incur high computational cost or rely on simplistic heuristics (e.g., common neighbor counts) that fail to model multi-hop dependencies. We introduce SP4LP (Shortest Path for Link Prediction), a novel framework that combines GNN-based node encodings with sequence modeling over shortest paths. Specifically, SP4LP first applies a GNN to compute representations for all nodes, then extracts the shortest path between each candidate node pair and processes the resulting sequence of node embeddings using a sequence model. This design enables SP4LP to capture expressive multi-hop relational patterns with computational efficiency. Empirically, SP4LP ac
    
[^393]: 后见之明重标注即抵达-规避学习所需的一切

    Hindsight Relabeling is All You Need for Reach-Avoid Learning

    [https://arxiv.org/abs/2505.19337](https://arxiv.org/abs/2505.19337)

    本文提出RADT，一种基于决策Transformer的离线无奖励强化学习方法，通过将目标与规避区域直接编码为提示token，仅凭随机策略产生的次优离线轨迹即可学会抵达-规避行为，并能在评估时灵活指定任意数量和大小的规避区域。

    

    离线目标条件强化学习方法在抵达-规避任务中展现出良好前景，这类任务要求智能体在抵达目标状态的同时避开状态空间中的不良区域。现有方法通常将规避区域信息编码到增广状态空间和代价函数中，这阻碍了在评估时对新规避区域信息进行灵活、动态的设定。此外，这些方法还严重依赖精心设计的奖励和代价函数，限制了对新环境和新规范的可迁移性。我们提出RADT，一种用于离线、无奖励、目标条件、规避区域条件强化学习的决策Transformer方法。RADT将目标和规避区域直接编码为提示token，允许在评估时指定任意数量、任意大小的规避区域。仅使用来自随机策略的次优离线轨迹，RADT即可在完整……（原文摘要在此处截断）中学到抵达-规避行为。

    arXiv:2505.19337v3 Announce Type: replace-cross  Abstract: Offline goal-conditioned reinforcement learning methods have shown promise for reach-avoid tasks, where an agent must reach a target state while avoiding undesirable regions of the state space. Existing approaches typically encode avoid-region information into an augmented state space and cost function, which prevents flexible, dynamic specification of novel avoid-region information at evaluation time. They also rely heavily on meticulously designed reward and cost functions, limiting transferability to novel environments and specifications. We introduce RADT, a decision transformer approach for offline, reward-free, goal-conditioned, avoid-region-conditioned RL. RADT encodes goals and avoid-regions directly as prompt tokens, allowing any number of avoid-regions of arbitrary size to be specified at evaluation time. Using only suboptimal offline trajectories from a random policy, RADT can learn reach-avoid behavior in a complete
    
[^394]: 质疑深度学习中的表征乐观主义：破碎纠缠表征假说

    Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis

    [https://arxiv.org/abs/2505.11581](https://arxiv.org/abs/2505.11581)

    本文通过对比演化网络与SGD训练网络在单图像生成任务上的内部表征，发现两者虽输出行为相同但内部结构截然不同——SGD训练的网络呈现“破碎纠缠表征”，从而挑战了“性能更好必然意味着内部表征更好”的表征乐观主义观点。

    

    现代人工智能领域令人兴奋的诸多进展，都源于这样一个观察：扩展现有系统的规模能带来更好的性能。但更好的性能是否必然意味着更好的内部表征？表征乐观主义者认为必然如此，而本立场文件对这一观点提出了挑战。我们将通过开放式搜索过程演化而来的神经网络与通过传统随机梯度下降（SGD）训练的神经网络，在生成单张图像这一简单任务上进行了比较。这一极简设置具有独特优势：每个隐藏神经元的完整功能行为都可以轻松地可视化为图像，从而能够揭示网络的输出行为是如何在内部逐个神经元构建起来的。结果令人震惊：尽管两个网络产生了相同的输出行为，但它们的内部表征却截然不同。经SGD训练的网络呈现出一种无序状态，我们将其称为“破碎纠缠表征”（fractured entangled representation）……

    arXiv:2505.11581v2 Announce Type: replace-cross  Abstract: Much of the excitement in modern AI is driven by the observation that scaling up existing systems leads to better performance. But does better performance necessarily imply better internal representations? While the representational optimist assumes it must, this position paper challenges that view. We compare neural networks evolved through an open-ended search process to networks trained via conventional stochastic gradient descent (SGD) on the simple task of generating a single image. This minimal setup offers a unique advantage: each hidden neuron's full functional behavior can be easily visualized as an image, thus revealing how the network's output behavior is internally constructed neuron by neuron. The result is striking: while both networks produce the same output behavior, their internal representations differ dramatically. The SGD-trained networks exhibit a form of disorganization that we term fractured entangled rep
    
[^395]: 通过自动特征提取学习变点检测的惩罚参数

    Learning Penalty for Changepoint Detection via Automatic Feature Extraction

    [https://arxiv.org/abs/2505.07413](https://arxiv.org/abs/2505.07413)

    该研究提出利用循环网络自动提取特征，直接从原始序列中学习变点检测的最优惩罚参数，在20个基因组基准数据集上的实验表明其检测精度普遍优于依赖人工特征提取的传统方法。

    

    变点检测用于识别数据序列中的显著变化，在金融、遗传学和医疗保健等领域具有重要应用。最优分割算法能够高效地检测这些变化，其通过一个惩罚参数来限制变点的数量。然而，确定该惩罚参数的最优值往往具有挑战性。传统方法需要人工提取统计特征（如序列长度或方差）来进行预测。本研究提出了一种新颖的方法，利用循环网络自动提取特征，直接从原始序列中学习该惩罚参数。在20个基准基因组数据集上进行的实验表明，这种新方法在变点检测精度方面普遍优于传统方法。

    arXiv:2505.07413v3 Announce Type: replace  Abstract: Changepoint detection identifies significant shifts in data sequences, making it important in areas like finance, genetics, and healthcare. The Optimal Partitioning algorithms efficiently detect these changes, using a penalty parameter to limit the changepoints count. Determining the optimal value for this penalty can be challenging. Traditionally, this process involved manually extracting statistical features, such as sequence length or variance to make the prediction. This study proposes a novel approach that uses recurrent networks to learn this penalty directly from raw sequences by automatically extracting features. Experiments conducted on 20 benchmark genomic datasets show that this novel method generally outperforms traditional ones in changepoint detection accuracy.
    
[^396]: 利用指数-库尔巴克-莱布勒-马亚尔采样实现多臂老虎机的自适应性与最优性

    Achieving adaptivity and optimality for multi-armed bandits using Exponential-Kullback Leibler Maillard Sampling

    [https://arxiv.org/abs/2502.14379](https://arxiv.org/abs/2502.14379)

    本文提出了Exp-KL-MS算法，这是首个能同时满足渐近最优性、极小极大最优性（带√ln(K)因子）、Sub-UCB和方差自适应遗憾界等多种最优性标准的多臂老虎机算法。

    

    我们研究了K臂老虎机问题，其中奖励分布属于单参数指数分布族。在现有文献中，已有多种标准被提出用于评估此类算法的性能，包括渐近最优性、极小极大最优性、Sub-UCB以及方差自适应遗憾界。基于汤普森采样的算法和基于置信上界的算法已被用于实现其中部分标准。然而，目前没有任何算法能够同时满足上述所有标准。在本文中，我们设计了一种名为指数库尔巴克-莱布勒-马亚尔采样（简称Exp-KL-MS）的算法，该算法能够同时实现多种最优性标准，包括渐近最优性、带有√ln(K)因子的极小极大比率、Sub-UCB以及方差自适应遗憾界。我们的算法设计遵循最小经验散度框架。

    arXiv:2502.14379v3 Announce Type: replace  Abstract: We study the problem of $K$-armed bandits with reward distributions belonging to a one-parameter exponential distribution family. In the literature, several criteria have been proposed to evaluate the performance of such algorithms, including Asymptotic Optimality, Minimax Optimality, Sub-UCB, and variance-adaptive regret bound. Thompson Sampling-based and Upper Confidence Bound-based algorithms have been employed to achieve some of these criteria. However, none of these algorithms simultaneously satisfy all the aforementioned criteria.   In this paper, we design an algorithm, Exponential Kullback-LeiblerMaillard Sampling (abbrev. Exp-KL-MS), that achieves multiple optimality criteria simultaneously, including Asymptotic Optimality, minimax ratio with a $\sqrt{\ln (K)}$ factor, Sub-UCB, and a variance-adaptive regret bound. Our algorithm design follows the Minimum Empirical Divergence framework~\citep{honda2011asymptotically,maillard
    
[^397]: ENFORCE：基于自适应深度神经投影的非线性约束学习

    ENFORCE: Nonlinear Constrained Learning with Adaptive-depth Neural Projection

    [https://arxiv.org/abs/2502.06774](https://arxiv.org/abs/2502.06774)

    提出了ENFORCE神经网络架构，通过自适应深度神经投影模块（AdaNP）使预测在指定容差内满足非线性等式与不等式约束，并从理论上证明了仿射约束下投影的非扩张性（保证梯度稳定传播）及非线性约束下的局部收敛性。

    

    确保神经网络遵循特定领域的约束，对于解决安全性与可信性问题至关重要，同时还能提升推理精度。尽管大多数现实任务具有非线性特性，但现有方法大多仅限于仿射（等式）或凸（不等式）约束。我们提出了ENFORCE，这是一种神经网络架构，其采用自适应深度投影模块（AdaNP），使预测在指定容差 ε 内满足非线性等式和不等式约束，并且在仿射（关于y）情形下可精确满足约束。对于仿射约束集，我们证明了相应的投影映射是非扩张的（1-Lipschitz），从而确保梯度传播的稳定性。对于非线性约束，我们在标准正则性条件下建立了局部收敛性分析。我们在多个任务上评估了ENFORCE，包括函数拟合、实际工程案例研究以及学习优化问题。

    arXiv:2502.06774v5 Announce Type: replace  Abstract: Ensuring neural networks adhere to domain-specific constraints is crucial for addressing safety and trustworthiness while also enhancing inference accuracy. Despite the nonlinear nature of most real-world tasks, the majority of existing methods are limited to affine (equality) or convex (inequality) constraints. We introduce ENFORCE, a neural network architecture that uses an adaptive projection module (AdaNP) to enforce nonlinear equality and inequality constraints in the predictions up to a specified tolerance $\varepsilon$, and exactly in the affine-in-$y$ case. For affine constraint sets, we prove that the associated projection mapping is non-expansive (1-Lipschitz), ensuring stable gradient propagation. For nonlinear constraints, we establish local convergence analysis under standard regularity conditions. We evaluate ENFORCE on multiple tasks, including function fitting, real-world engineering case studies, and learning optimiz
    
[^398]: 利用长短期记忆网络预测缩短传感器响应时间

    Sensor Response-Time Reduction using Long-Short Term Memory Network Forecasting

    [https://arxiv.org/abs/2404.17144](https://arxiv.org/abs/2404.17144)

    本文提出利用长短期记忆（LSTM）网络集成方法，仅从少量初始时序生物传感器测量数据即可准确预测最终的稳态响应，从而有效缩短生物传感器的响应时间。

    

    生物传感器的响应时间是医疗诊断等安全关键应用中的一项关键指标，在这些应用中，更早的诊断能够显著改善患者的治疗效果。然而，生物传感器达到最终平衡状态的速度可能受到质量传输不佳和分子扩散时间较长的限制，这些因素会增加目标分子到达生物传感器活性感应区域所需的时间。虽然优化系统和传感器设计可以促进分子更快地到达传感元件，但一种更简单且可广泛应用于所有传感器平台的响应时间缩短方法是使用时间序列预测来预测最终的稳态传感器响应。在这项工作中，我们证明了长短期记忆（LSTM）网络的集成方法能够从少量的初始随时间变化的生物传感器测量数据中准确预测平衡状态的生物传感器响应。

    arXiv:2404.17144v3 Announce Type: replace  Abstract: The response time of a biosensor is a crucial metric in safety-critical applications such as medical diagnostics where an earlier diagnosis can markedly improve patient outcomes. However, the speed at which a biosensor reaches a final equilibrium state can be limited by poor mass transport and long molecular diffusion times that increase the time it takes target molecules to reach the active sensing region of a biosensor. While optimization of system and sensor design can promote molecules reaching the sensing element faster, a simpler and complementary approach for response time reduction that is widely applicable across all sensor platforms is to use time-series forecasting to predict the ultimate steady-state sensor response. In this work, we show that ensembles of long short-term memory (LSTM) networks can accurately predict equilibrium biosensor response from a small quantity of initial time-dependent biosensor measurements, all
    
[^399]: 探索用于域自适应检测Transformer的序列特征对齐

    Exploring Sequence Feature Alignment for Domain Adaptive Detection Transformers

    [https://arxiv.org/abs/2107.12636](https://arxiv.org/abs/2107.12636)

    提出了一种专为检测Transformer设计的序列特征对齐（SFA）方法，通过基于域查询的特征对齐（DQFA）模块和逐词元特征对齐（TDA）模块来减小域差异，从而有效提升跨域目标检测性能。

    

    检测Transformer近年来展现出了有前景的目标检测结果，并吸引了越来越多的关注。然而，如何开发有效的域自适应技术来提升其跨域性能仍未被探索且不明确。在本文中，我们深入研究了这一主题，并通过实验发现，仅在CNN骨干网络上进行直接的特征分布对齐只能带来有限的改进，因为它无法保证Transformer中用于预测的序列特征具有域不变性。为了解决这个问题，我们提出了一种新颖的序列特征对齐（SFA）方法，该方法专为检测Transformer的自适应而设计。在技术上，SFA由一个基于域查询的特征对齐（DQFA）模块和一个逐词元特征对齐（TDA）模块组成。在DQFA中，一种新颖的域查询被用来从两个域的词元序列中聚合和对齐全局上下文，DQFA减小了域差异。

    arXiv:2107.12636v5 Announce Type: replace-cross  Abstract: Detection transformers have recently shown promising object detection results and attracted increasing attention. However, how to develop effective domain adaptation techniques to improve its cross-domain performance remains unexplored and unclear. In this paper, we delve into this topic and empirically find that direct feature distribution alignment on the CNN backbone only brings limited improvements, as it does not guarantee domain-invariant sequence features in the transformer for prediction. To address this issue, we propose a novel Sequence Feature Alignment (SFA) method that is specially designed for the adaptation of detection transformers. Technically, SFA consists of a domain query-based feature alignment (DQFA) module and a token-wise feature alignment (TDA) module. In DQFA, a novel domain query is used to aggregate and align global context from the token sequence of both domains. DQFA reduces the domain discrepancy 
    
[^400]: 估计和激励具有隐藏奖励的不完全知识代理

    Estimating and Incentivizing Imperfect-Knowledge Agents with Hidden Rewards. (arXiv:2308.06717v1 [cs.LG])

    [http://arxiv.org/abs/2308.06717](http://arxiv.org/abs/2308.06717)

    本文研究了一个自私学习代理和学习委托人之间的重复对自选游戏，探索如何估计和激励具有隐藏奖励的不完全知识代理。

    

    在实践中，激励提供者（即委托人）通常无法观察受到激励的代理的奖励实现情况，这与许多先前研究过的委托人-代理模型形成了对比。这种信息不对称性使委托人仅通过观察代理的决策就要始终估计代理的未知奖励变得更加困难，当代理需要学习自己的奖励时，这个挑战变得更加困难。这种复杂情况在各种现实场景中被观察到，从可再生能源储存合同到个性化医疗保健激励。因此，它不仅提供了有趣的理论问题，而且具有广泛的实际相关性。本文探讨了一个自私学习代理和学习委托人之间的重复对自选游戏。代理解决多臂老虎机（MAB）问题，以最大化他们预期的奖励和激励。除代理的学习外，委托人还训练了一个并行算法，并面临一个折中情况

    In practice, incentive providers (i.e., principals) often cannot observe the reward realizations of incentivized agents, which is in contrast to many principal-agent models that have been previously studied. This information asymmetry challenges the principal to consistently estimate the agent's unknown rewards by solely watching the agent's decisions, which becomes even more challenging when the agent has to learn its own rewards. This complex setting is observed in various real-life scenarios ranging from renewable energy storage contracts to personalized healthcare incentives. Hence, it offers not only interesting theoretical questions but also wide practical relevance. This paper explores a repeated adverse selection game between a self-interested learning agent and a learning principal. The agent tackles a multi-armed bandit (MAB) problem to maximize their expected reward plus incentive. On top of the agent's learning, the principal trains a parallel algorithm and faces a trade-of
    
[^401]: 动作状态相关的动态模型选择

    Action-State Dependent Dynamic Model Selection. (arXiv:2307.04754v1 [cs.LG])

    [http://arxiv.org/abs/2307.04754](http://arxiv.org/abs/2307.04754)

    本文提出了一种动态模型选择的方法，该方法能够根据不同的状态选择最优的模型，并通过强化学习算法对动态规划问题进行近似和估计。实验结果表明，在重新平衡成本下切换投资组合模型时，使用宏观经济信息的性能优于事后选择最佳投资组合模型。

    

    在世界的某些状态下，多个模型中的一个可能只在其中某些状态下表现最佳。而在模型之间的切换也可能代价高昂。在这种情况下，寻找一种能够动态选择模型的过程需要解决一个复杂的估计问题和动态规划问题。本文使用强化学习算法来从数据中近似和估计这个动态规划问题的最优解。实验结果表明，该算法能够一致地估计出根据一组协变量选择不同模型的最优策略。具体应用方面，例如在重新平衡成本下切换不同投资组合模型，使用宏观经济信息进行决策。通过一组宏观经济变量和价格数据，经验应用于上述投资组合问题表现出比事后选择最佳投资组合模型更优的性能。

    A model among many may only be best under certain states of the world. Switching from a model to another can also be costly. Finding a procedure to dynamically choose a model in these circumstances requires to solve a complex estimation procedure and a dynamic programming problem. A Reinforcement learning algorithm is used to approximate and estimate from the data the optimal solution to this dynamic programming problem. The algorithm is shown to consistently estimate the optimal policy that may choose different models based on a set of covariates. A typical example is the one of switching between different portfolio models under rebalancing costs, using macroeconomic information. Using a set of macroeconomic variables and price data, an empirical application to the aforementioned portfolio problem shows superior performance to choosing the best portfolio model with hindsight.
    
[^402]: 未观测到代理奖励的重复负责人代理博弈问题研究

    Repeated Principal-Agent Games with Unobserved Agent Rewards and Perfect-Knowledge Agents. (arXiv:2304.07407v1 [cs.LG])

    [http://arxiv.org/abs/2304.07407](http://arxiv.org/abs/2304.07407)

    本文提出了一个利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    

    本文研究了一个多臂老虎机框架中的重复负责人代理博弈场景，其中代理选择一种老虎机后会获得奖励和激励，但负责人只能观察到代理选择了哪个老虎机以及代理相应的激励，而想要设计一种合适的策略却充满了挑战性。本文提出了一种利用多臂老虎机框架结构来处理未知代理奖励的策略，并证明了其性能是渐进最优的。

    Motivated by a number of real-world applications from domains like healthcare and sustainable transportation, in this paper we study a scenario of repeated principal-agent games within a multi-armed bandit (MAB) framework, where: the principal gives a different incentive for each bandit arm, the agent picks a bandit arm to maximize its own expected reward plus incentive, and the principal observes which arm is chosen and receives a reward (different than that of the agent) for the chosen arm. Designing policies for the principal is challenging because the principal cannot directly observe the reward that the agent receives for their chosen actions, and so the principal cannot directly learn the expected reward using existing estimation techniques. As a result, the problem of designing policies for this scenario, as well as similar ones, remains mostly unexplored. In this paper, we construct a policy that achieves a low regret (i.e., square-root regret up to a log factor) in this scenar
    

